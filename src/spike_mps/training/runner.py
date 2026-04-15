from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from spike_mps.models.mps_classifier import (
    MPSClassifier,
    MPSModelConfig,
    select_predicted_class,
)
from spike_mps.training.checkpoints import load_model_from_checkpoint, save_checkpoint
from spike_mps.training.config import (
    get_experiment_config,
    load_training_config,
)
from spike_mps.training.data import (
    load_dataset_bundle,
)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    predictions: list[int]
    labels: list[int]


def run_training_experiment(
    *,
    base_path: Path,
    config_path: Path | None,
    experiment_name: str,
) -> Path:
    app_config = load_training_config(
        base_path=base_path,
        explicit_path=config_path,
    )
    experiment_config = get_experiment_config(
        app_config=app_config, experiment_name=experiment_name
    )
    experiment_dir = (
        base_path / "output" / "processed_data" / "experiments" / experiment_config.name
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = base_path / "datasets" / "generated" / experiment_config.dataset_name
    dataset_bundle = load_dataset_bundle(dataset_dir=dataset_dir)
    device = _resolve_device(experiment_config.device)
    torch.manual_seed(experiment_config.seed)

    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=dataset_bundle.sequence_length,
            input_dim=2,
            num_classes=dataset_bundle.num_classes,
            bond_dim=experiment_config.bond_dim,
            task=dataset_bundle.task,
        )
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )

    train_loader = DataLoader(
        dataset_bundle.train_dataset,
        batch_size=experiment_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_bundle.val_dataset,
        batch_size=experiment_config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset_bundle.test_dataset,
        batch_size=experiment_config.batch_size,
        shuffle=False,
    )

    history_rows: list[dict[str, float | int]] = []
    checkpoint_path = experiment_dir / "checkpoint_best.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, experiment_config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_accuracy": val_metrics.accuracy,
            }
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                experiment_name=experiment_config.name,
                dataset_name=experiment_config.dataset_name,
                experiment_config=experiment_config.to_dict(),
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                metrics={
                    "train_loss": train_metrics.loss,
                    "val_loss": val_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "val_accuracy": val_metrics.accuracy,
                },
                seed=experiment_config.seed,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= experiment_config.patience:
            break

    best_model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, map_location=device
    )
    test_metrics = _run_epoch(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
    )

    _write_history(path=experiment_dir / "history.csv", history_rows=history_rows)
    _write_confusion_matrix(
        path=experiment_dir / "confusion_matrix.csv",
        labels=test_metrics.labels,
        predictions=test_metrics.predictions,
        num_classes=dataset_bundle.num_classes,
    )
    _write_metrics(
        path=experiment_dir / "metrics.yaml",
        payload={
            "experiment_name": experiment_config.name,
            "dataset_name": experiment_config.dataset_name,
            "task": dataset_bundle.task,
            "sequence_length": dataset_bundle.sequence_length,
            "num_classes": dataset_bundle.num_classes,
            "bond_dim": experiment_config.bond_dim,
            "device": str(device),
            "best_epoch": checkpoint["best_epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
        },
    )
    return checkpoint_path


def _run_epoch(
    *,
    model: MPSClassifier,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> EpochMetrics:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    correct_predictions = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            scores = model(inputs)
            loss = criterion(scores.abs(), labels)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        predictions = select_predicted_class(scores)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        correct_predictions += int((predictions == labels).sum().item())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / max(total_examples, 1)
    accuracy = correct_predictions / max(total_examples, 1)
    return EpochMetrics(
        loss=average_loss,
        accuracy=accuracy,
        predictions=all_predictions,
        labels=all_labels,
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def _write_history(
    *,
    path: Path,
    history_rows: list[dict[str, float | int]],
) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def _write_confusion_matrix(
    *,
    path: Path,
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> None:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[label][prediction] += 1

    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        header = ["true/pred"] + [str(index) for index in range(num_classes)]
        writer.writerow(header)
        for index, row in enumerate(matrix):
            writer.writerow([index, *row])


def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
