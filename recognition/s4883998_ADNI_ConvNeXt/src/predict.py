
from pathlib import Path
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from dataset import (
    DATA_DIR,
    generate_datasets,
    ADNIDataset,
    NC_LABEL,
    AD_LABEL,
)

from models import ADNICNN, ADNI_ConvNeXt
from transforms import MRINormalise


#################
# CONFIGURATION
#################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("Models")
OUTPUT_DIR = MODEL_DIR / "evaluation"
BATCH_SIZE = 32
CLASS_NAMES = ["NC", "AD"]
CONFIDENCE_THRESHOLD = 0.90


def load_model(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    """Load and return model saved at path in evaluation mode"""

    if not path.is_file():
        raise FileNotFoundError(f"Invalid path supplied: {path}")

    state_dict = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


##############################
# RUN / INTERPRET MODEL DATA
##############################


def run_inference(
    model: nn.Module,
    dataset: ADNIDataset,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Run the given model and collect slice predictions by patient
    """
    
    model.eval()

    patient_probabilities = {}
    patient_labels = {}
    total_slices = 0

    # Reset GPU memory statistics before inference.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    offset = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)

            # Generate predictions for slice X
            logits = model(X)
            probabilities = torch.softmax(logits, dim=1).cpu()
            batch_entries = dataset.entries[offset:offset + len(y)]

            for entry, probability in zip(batch_entries, probabilities):
                patient_id = entry.patient_id
                if patient_id not in patient_probabilities:
                    patient_probabilities[patient_id] = []
                    patient_labels[patient_id] = entry.label

                elif patient_labels[patient_id] != entry.label:
                    raise ValueError(f"Conflicting labels for patient {patient_id}")

                patient_probabilities[patient_id].append(probability)

            offset += len(y)
            total_slices += len(y)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    runtime = time.perf_counter() - start_time
    return patient_probabilities, patient_labels, total_slices, runtime


def aggregate_patient_predictions(
    patient_probabilities: dict,
    patient_labels: dict,
):
    """
    Get a patients average classification probability across their 
    corresponding slices
    """
    y_true = []
    y_pred = []
    y_prob = []

    for patient_id, probabilities in patient_probabilities.items():

        # Average prob across all slices
        average_probability = torch.stack(probabilities).mean(dim=0)

        # AD probability
        ad_probability = average_probability[AD_LABEL].item()
        prediction = average_probability.argmax().item()

        y_true.append(patient_labels[patient_id])
        y_pred.append(prediction)
        y_prob.append(ad_probability)

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
    )


#####################
# CALCULATE METRICS
#####################


def calculate_classification_metrics(y_true, y_pred, y_prob):
    """
    Calculating relevant classificaiton metrics based on values provided
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[NC_LABEL, AD_LABEL],
        average=None,
        zero_division=0,
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[NC_LABEL, AD_LABEL],
    ).ravel()

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        auroc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "confusion_matrix": (tn, fp, fn, tp),
    }

def calculate_confidence_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification coverage and accuracy for high-confidence
    predictions.
    """

    # Calculate confidence in the predicted class.
    confidence = np.where(
        y_pred == AD_LABEL,
        y_prob,
        1 - y_prob,
    )

    # Identify predictions meeting the confidence threshold.
    accepted = confidence >= CONFIDENCE_THRESHOLD
    coverage = np.mean(accepted)

    if np.any(accepted):
        high_confidence_accuracy = accuracy_score(
            y_true[accepted],
            y_pred[accepted],
        )
    else:
        high_confidence_accuracy = float("nan")

    return {
        "coverage": coverage,
        "high_confidence_accuracy": high_confidence_accuracy,
    }

def calculate_computational_metrics(
    model: nn.Module,
    runtime: float,
    total_slices: int,
    device: torch.device,
):
    """
    Calculate model complexity, inference performance and GPU memory usage.
    """

    # Model complexity.
    parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Inference performance.
    latency = runtime / total_slices
    throughput = total_slices / runtime

    # GPU memory usage.
    if device.type == "cuda":
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
    else:
        peak_allocated = float("nan")
        peak_reserved = float("nan")

    return {
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "runtime": runtime,
        "latency": latency,
        "throughput": throughput,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
    }


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


def evaluate_model(
    model: nn.Module,
    dataset: ADNIDataset,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
):
    # Run inference on individual MRI slices.
    patient_probabilities, patient_labels, total_slices, runtime = (
        run_inference(model, dataset, dataloader, device)
    )

    # Aggregate slice predictions into patient-level predictions.
    y_true, y_pred, y_prob = aggregate_patient_predictions(
        patient_probabilities,
        patient_labels,
    )


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
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
        batch_size=BATCH_SIZE,
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
        DEVICE,
    )

    print("Evaluating ConvNeXt...")
    convnext_results = evaluate_model(
        convnext_model,
        test_ds,
        test_loader,
        DEVICE,
        "ConvNeXt",
    )

    del convnext_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\nConvNeXt stuff done =====================\n\n")

     # CNN IMPLEMENTATION ################

    print("Loading CNN...")
    cnn_model = ADNICNN()

    cnn_model = load_model(
        cnn_model,
        MODEL_DIR / "cnn.pt",
        DEVICE,
    )

    print("Evaluating CNN...")
    cnn_results = evaluate_model(
        cnn_model,
        test_ds,
        test_loader,
        DEVICE,
        "CNN",
    )

    del cnn_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\nCNN stuff done =====================\n\n")

