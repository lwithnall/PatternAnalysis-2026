from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import (
    DATA_DIR,
    generate_datasets,
    ADNIDataset,
    NC_LABEL,
    AD_LABEL,
)

from models import ADNICNN, ADNI_ConvNeXt
from transforms import MRINormalise
from plotting import generate_evaluation_plots

from utils import (
    calculate_classification_metrics,
    calculate_computational_metrics,
    calculate_confidence_metrics
)

MODEL_DIR = Path("Models")
OUTPUT_DIR = MODEL_DIR / "evaluation"

CLASS_NAMES = ["NC", "AD"]


#################
# CONFIGURATION
#################

def load_model(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    """Load and return model saved at path in evaluation mode"""

    if not path.is_file():
        raise FileNotFoundError(f"Invalid path supplied: {path}")

    state_dict = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


##########
# OUTPUT
##########


def print_evaluation_results(
    model_name: str,
    y_true: np.ndarray,
    total_slices: int,
    classification_metrics: dict,
    confidence_metrics: dict,
    computational_metrics: dict,
):
    """
    Print classification, confidence and computational evaluation metrics.
    
    AI DECLARATION: FUNCTION MADE BY GPT; REFER TO AI DEC. ENTRY 4 
    """

    print(f"\n{'=' * 60}")
    print(f" EVALUATION RESULTS: {model_name}")
    print(f"{'=' * 60}")

    print(f"Patients evaluated: {len(y_true)}")
    print(f"MRI slices evaluated: {total_slices}")

    # Classification metrics
    print(f"\n{'-' * 60}")
    print("CLASSIFICATION METRICS")
    print(f"{'-' * 60}")

    print(f"Accuracy:       {classification_metrics['accuracy']:.4f}")
    print(f"Macro-F1:       {classification_metrics['macro_f1']:.4f}")
    print(f"AUROC:          {classification_metrics['auroc']:.4f}")
    print(f"Average Prec.:  {classification_metrics['pr_auc']:.4f}")

    precision = classification_metrics["precision"]
    recall = classification_metrics["recall"]
    f1 = classification_metrics["f1"]

    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-score':>12}")

    for i, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<10} "
            f"{precision[i]:>12.4f} "
            f"{recall[i]:>12.4f} "
            f"{f1[i]:>12.4f}"
        )

    # Confusion matrix
    tn, fp, fn, tp = classification_metrics["confusion_matrix"]

    print("\nConfusion matrix:")
    print(f"{'':>12} {'Predicted NC':>15} {'Predicted AD':>15}")
    print(f"{'Actual NC':<12} {tn:>15} {fp:>15}")
    print(f"{'Actual AD':<12} {fn:>15} {tp:>15}")

    # Confidence metrics
    print(f"\n{'-' * 60}")
    print("CONFIDENCE METRICS")
    print(f"{'-' * 60}")

    print(f"Confidence threshold:      {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Coverage:                  {confidence_metrics['coverage']:.2%}")
    print(
        f"High-confidence accuracy: "
        f"{confidence_metrics['high_confidence_accuracy']:.4f}"
    )

    # Computational metrics
    print(f"\n{'-' * 60}")
    print("COMPUTATIONAL METRICS")
    print(f"{'-' * 60}")

    print(f"Total parameters:       {computational_metrics['parameters']:,}")
    print(f"Trainable parameters:   {computational_metrics['trainable_parameters']:,}")
    print(f"Inference runtime:      {computational_metrics['runtime']:.2f} s")
    print(f"Latency per slice:      {computational_metrics['latency'] * 1000:.2f} ms")
    print(f"Throughput:             {computational_metrics['throughput']:.2f} slices/s")

    if not np.isnan(computational_metrics["peak_allocated"]):
        print(f"Peak GPU allocated:     {computational_metrics['peak_allocated']:.2f} MB")
        print(f"Peak GPU reserved:      {computational_metrics['peak_reserved']:.2f} MB")
    else:
        print("GPU memory:             Unavailable (CPU inference)")

    print(f"\n{'=' * 60}\n")


############################
# LINK EVERYTHING TOGETHER
############################

def run_model_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device 
):
    """Run given model on test data to output predictions w/ probabilities"""

    model.eval()

    slice_probabilities = [] # Classification probabilities
    slice_labels = [] # Predicted label
    total_slices = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter() # Time taken to run inference code

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)

            # Predictions for slice X
            logits = model(X)
            probabilties = torch.softmax(logits, dim=1).cpu()

            slice_probabilities.extend(probabilties.tolist())
            slice_labels.extend(y.cpu().tolist())

            total_slices += len(y)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    runtime = time.perf_counter() - start_time
    return slice_probabilities, slice_labels, total_slices, runtime


def get_predictions(
    slice_probabilities: list,
    slice_labels: list,
):
    """
    Convert slice probabilities and true labels into arrays
    for classification metrics

    y_true: True labels (NC = 0, AD = 1)
    y_pred: Predicted labels (NC = 0, AD = 1)
    y_ad_prob: Predicted probability of AD
    """

    probabilities = np.array(slice_probabilities)

    y_true = np.array(slice_labels)
    y_pred = probabilities.argmax(axis=1)
    y_ad_prob = probabilities[:, AD_LABEL]

    return y_true, y_pred, y_ad_prob


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
):
    """
    Run model evaluation, calculate metrics and generate evaluation plots.
    """

    # Running model on given dataloader
    slice_probabilities, slice_labels, total_slices, runtime = run_model_test(
        model, dataloader, device
    )

    y_true, y_pred, y_ad_prob = get_predictions(
        slice_probabilities, slice_labels
    )

    # Calculating metrics
    classification_metrics = calculate_classification_metrics(
        y_true, y_pred, y_ad_prob,
    )

    confidence_metrics = calculate_confidence_metrics(
        y_true, y_pred, y_ad_prob
    )

    computational_metrics = calculate_computational_metrics(
        model, runtime, total_slices, device
    )

    # Evaluation results
    print_evaluation_results(
        model_name,
        y_true,
        total_slices,
        classification_metrics,
        confidence_metrics,
        computational_metrics,
    )

    # Plotting
    generate_evaluation_plots(
        y_true,
        y_pred,
        y_ad_prob,
        classification_metrics,
        OUTPUT_DIR / model_name.lower(),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    confidence_threshold = 0.90
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_transforms = v2.Compose([MRINormalise()])

    # Load test dataset
    print("Loading test dataset...")
    _, _, test_ds = generate_datasets(
        DATA_DIR,
        validation_split=0.0,
        random_state=42,
        eval_transforms=eval_transforms,
    )

    # Test dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # CONVNEXT IMPLEMENTATION ################

    # Loading saved model
    print("Loading ConvNeXt...")
    convnext_model = ADNI_ConvNeXt(
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        depths=(2, 2, 4, 2),
        dims=(48, 96, 192, 384),
    )

    convnext_model = load_model(
        convnext_model,
        MODEL_DIR / "convnext.pt",
        device,
    )

    print("Evaluating ConvNeXt...")
    convnext_error_cases = evaluate_model(
        convnext_model,
        test_ds,
        test_loader,
        device,
        "ConvNeXt",
    )

    del convnext_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nConvNeXt stuff done =====================\n\n")

     # CNN IMPLEMENTATION ################

    print("Loading CNN...")
    cnn_model = ADNICNN()

    cnn_model = load_model(
        cnn_model,
        MODEL_DIR / "cnn.pt",
        device,
    )

    print("Evaluating CNN...")
    cnn_error_cases =evaluate_model(
        cnn_model,
        test_ds,
        test_loader,
        device,
        "CNN",
    )

    del cnn_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nCNN stuff done =====================\n\n")

