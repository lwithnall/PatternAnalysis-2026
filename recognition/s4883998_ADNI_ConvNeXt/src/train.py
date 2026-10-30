from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

        optimiser.zero_grad()

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
        val_loss, val_accuracy = evaluate_epoch(
            model, val_loader, loss_fn, device
        )

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
