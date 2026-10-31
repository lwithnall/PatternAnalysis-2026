from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import DATA_DIR, generate_datasets, generate_dataloaders
from models import ADNICNN, ADNI_ConvNeXt
from transforms import MRINormalise
from utils import output_cuda_resource_usage


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the model for one epoch

    Assumes loss_fn uses reduction="mean", so each batch's
    loss is averaged across its samples.

    Returns:
        Average loss and accuracy for the trained epoch
    """

    model.train()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimiser.zero_grad(set_to_none=True)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()

        # Metrics
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = pred.argmax(dim=1)
        correct_predictions += (predictions == y).sum().item()

    if total_samples == 0:
        raise ValueError("Train loader is empty")

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model on one epoch

    Doesn't calculate gradients nor update model weights

    Returns:
        Average loss and accuracy across validation dataset
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(X)
            loss = loss_fn(pred, y)

            # Metrics
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            predictions = pred.argmax(dim=1)

            correct_predictions += (predictions == y).sum().item()
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Evaluate loader is empty")

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

    return average_loss, accuracy


@output_cuda_resource_usage
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    save_path: Path,
):
    """
    Train and evaluate model iteratively over provided number of epochs.
    Model will be saved in the provided save_path destination
    """

    if epochs < 1:
        raise ValueError("Need at least 1 epoch")

    if val_loader is None:
        raise ValueError("Need validation data loader")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    print(f"Training on: {device}")
    print(f"Epochs: {epochs}")
    print("=" * 30)

    # Metrics
    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Storing model with best validation loss
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(epochs):
        # Training
        train_loss, train_accuracy = train_epoch(
            model, train_loader, loss_fn, optimiser, device
        )

        # Validation
        val_loss, val_accuracy = evaluate_epoch(model, val_loader, loss_fn, device)

        # Recordings
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())

        # Epoch results
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train loss: {train_loss:.4f} "
            f"| Train acc: {train_accuracy:.2%} "
            f"| Val loss: {val_loss:.4f} "
            f"| Val acc: {val_accuracy:.2%}"
        )

    if best_model_state is None:
        raise RuntimeError("Couldn't produce model checkpoint")

    model.load_state_dict(best_model_state)
    torch.save(best_model_state, save_path)
    print(f"\nSaved model from epoch #{best_epoch}")

    print("-" * 30)

    return model, metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    val_split = 0.2
    random_state = 42
    batch_size = 32
    epochs = 30
    save_path_root = Path("Models/")
    
    train_transforms = v2.Compose([
        v2.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03), fill=0,),
        MRINormalise(),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    eval_transforms = v2.Compose([MRINormalise()])

    print("Getting datasets...")
    train_ds, val_ds, test_ds = generate_datasets(
        DATA_DIR, 
        validation_split=val_split,  
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
        random_state=random_state,
    )

    print("Getting preparing dataloaders...")
    train_loader, val_loader, test_loader = generate_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size,
    )

    print("Configuring ConvNeXt model...")
    convnext_model = ADNI_ConvNeXt(
        drop_path_rate=0.2, layer_scale_init_value=1e-6,
        depths=(2, 2, 4, 2  ),
        dims=(48, 96, 192, 384),
    ).to(device)
    next_loss_fn = nn.CrossEntropyLoss()
    next_optimiser = torch.optim.AdamW(
        convnext_model.parameters(), lr=1e-4, weight_decay=0.1
    )

    print("Training ConvNext...")
    train_model(
        convnext_model,
        train_loader,
        val_loader,
        next_loss_fn,
        next_optimiser,
        epochs,
        device,
        save_path = save_path_root / "convnext.pt"
    )

    del convnext_model, next_optimiser

    print("Configuring CNN model...")
    cnn_model = ADNICNN().to(device)
    cnn_loss_fn = nn.CrossEntropyLoss()
    cnn_optimiser = torch.optim.Adam(
        cnn_model.parameters(), lr=0.001
    )

    print("Training CNN...")
    train_model(
        cnn_model,
        train_loader,
        val_loader,
        cnn_loss_fn,
        cnn_optimiser,
        epochs,
        device,
        save_path = save_path_root / "cnn.pt"
    )
