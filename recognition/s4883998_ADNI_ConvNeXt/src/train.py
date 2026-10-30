from pathlib import Path
from models import ADNICNN

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
    """
    ...


def evaluate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model on one epoch
    """
    ...


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | None,
    save_path: Path,
):
    """
    Train and evaluate model iteratively over provided number of epochs.
    Model will be saved in the provided save_path destination
    """
    ...
