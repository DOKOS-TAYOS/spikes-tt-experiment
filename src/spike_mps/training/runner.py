from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F
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
from spike_mps.training.data import load_dataset_bundle


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    cross_entropy_loss: float
    output_concentration_penalty: float
    tensor_concentration_penalty: float
    accuracy: float
    target_component_mean: float
    off_target_component_mean: float
    best_incorrect_component_mean: float
    target_margin_mean: float
    correct_predictions: int
    total_examples: int
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
    model.prepare_for_training()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )

    train_loader = DataLoader(
        dataset_bundle.full_dataset,
        batch_size=experiment_config.batch_size,
        shuffle=True,
    )
    full_loader = DataLoader(
        dataset_bundle.full_dataset,
        batch_size=experiment_config.batch_size,
        shuffle=False,
    )

    history_rows: list[dict[str, float | int]] = []
    checkpoint_path = experiment_dir / "checkpoint_best.pt"
    best_full_accuracy = float("-inf")
    best_full_loss = float("inf")
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
            output_concentration_penalty_weight=(
                experiment_config.output_concentration_penalty_weight
            ),
            tensor_concentration_penalty_weight=(
                experiment_config.tensor_concentration_penalty_weight
            ),
        )
        full_metrics = _run_epoch(
            model=model,
            dataloader=full_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
            output_concentration_penalty_weight=(
                experiment_config.output_concentration_penalty_weight
            ),
            tensor_concentration_penalty_weight=(
                experiment_config.tensor_concentration_penalty_weight
            ),
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_cross_entropy_loss": train_metrics.cross_entropy_loss,
                "train_output_concentration_penalty": (
                    train_metrics.output_concentration_penalty
                ),
                "train_tensor_concentration_penalty": (
                    train_metrics.tensor_concentration_penalty
                ),
                "full_loss": full_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_target_component_mean": train_metrics.target_component_mean,
                "train_off_target_component_mean": (
                    train_metrics.off_target_component_mean
                ),
                "train_best_incorrect_component_mean": (
                    train_metrics.best_incorrect_component_mean
                ),
                "train_target_margin_mean": train_metrics.target_margin_mean,
                "full_cross_entropy_loss": full_metrics.cross_entropy_loss,
                "full_output_concentration_penalty": (
                    full_metrics.output_concentration_penalty
                ),
                "full_tensor_concentration_penalty": (
                    full_metrics.tensor_concentration_penalty
                ),
                "full_accuracy": full_metrics.accuracy,
                "full_target_component_mean": full_metrics.target_component_mean,
                "full_off_target_component_mean": (
                    full_metrics.off_target_component_mean
                ),
                "full_best_incorrect_component_mean": (
                    full_metrics.best_incorrect_component_mean
                ),
                "full_target_margin_mean": full_metrics.target_margin_mean,
            }
        )
        _log_epoch(
            epoch=epoch,
            total_epochs=experiment_config.epochs,
            train_metrics=train_metrics,
            full_metrics=full_metrics,
        )

        if _is_better_full_checkpoint(
            full_accuracy=full_metrics.accuracy,
            full_loss=full_metrics.loss,
            best_full_accuracy=best_full_accuracy,
            best_full_loss=best_full_loss,
        ):
            best_full_accuracy = full_metrics.accuracy
            best_full_loss = full_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                experiment_name=experiment_config.name,
                dataset_name=experiment_config.dataset_name,
                experiment_config=experiment_config.to_dict(),
                best_epoch=best_epoch,
                best_full_loss=best_full_loss,
                metrics={
                    "train_loss": train_metrics.loss,
                    "train_cross_entropy_loss": train_metrics.cross_entropy_loss,
                    "train_output_concentration_penalty": (
                        train_metrics.output_concentration_penalty
                    ),
                    "train_tensor_concentration_penalty": (
                        train_metrics.tensor_concentration_penalty
                    ),
                    "full_loss": full_metrics.loss,
                    "full_cross_entropy_loss": full_metrics.cross_entropy_loss,
                    "full_output_concentration_penalty": (
                        full_metrics.output_concentration_penalty
                    ),
                    "full_tensor_concentration_penalty": (
                        full_metrics.tensor_concentration_penalty
                    ),
                    "train_accuracy": train_metrics.accuracy,
                    "train_target_component_mean": (
                        train_metrics.target_component_mean
                    ),
                    "train_off_target_component_mean": (
                        train_metrics.off_target_component_mean
                    ),
                    "train_best_incorrect_component_mean": (
                        train_metrics.best_incorrect_component_mean
                    ),
                    "train_target_margin_mean": train_metrics.target_margin_mean,
                    "full_accuracy": full_metrics.accuracy,
                    "full_target_component_mean": full_metrics.target_component_mean,
                    "full_off_target_component_mean": (
                        full_metrics.off_target_component_mean
                    ),
                    "full_best_incorrect_component_mean": (
                        full_metrics.best_incorrect_component_mean
                    ),
                    "full_target_margin_mean": full_metrics.target_margin_mean,
                },
                seed=experiment_config.seed,
            )
        else:
            epochs_without_improvement += 1

        if _reached_perfect_full_accuracy(full_metrics.accuracy):
            _log_perfect_accuracy_stop(epoch=epoch)
            break

        if epochs_without_improvement >= experiment_config.patience:
            break

    best_model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, map_location=device
    )
    full_metrics = _run_epoch(
        model=best_model,
        dataloader=full_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        output_concentration_penalty_weight=(
            experiment_config.output_concentration_penalty_weight
        ),
        tensor_concentration_penalty_weight=(
            experiment_config.tensor_concentration_penalty_weight
        ),
    )
    _log_final_summary(full_metrics=full_metrics)

    _write_history(path=experiment_dir / "history.csv", history_rows=history_rows)
    _write_confusion_matrix(
        path=experiment_dir / "confusion_matrix.csv",
        labels=full_metrics.labels,
        predictions=full_metrics.predictions,
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
            "output_concentration_penalty_weight": (
                experiment_config.output_concentration_penalty_weight
            ),
            "tensor_concentration_penalty_weight": (
                experiment_config.tensor_concentration_penalty_weight
            ),
            "device": str(device),
            "best_epoch": checkpoint["best_epoch"],
            "best_full_accuracy": best_full_accuracy,
            "best_full_loss": checkpoint["best_full_loss"],
            "full_loss": full_metrics.loss,
            "full_cross_entropy_loss": full_metrics.cross_entropy_loss,
            "full_output_concentration_penalty": (
                full_metrics.output_concentration_penalty
            ),
            "full_tensor_concentration_penalty": (
                full_metrics.tensor_concentration_penalty
            ),
            "full_accuracy": full_metrics.accuracy,
            "full_target_component_mean": full_metrics.target_component_mean,
            "full_off_target_component_mean": (full_metrics.off_target_component_mean),
            "full_best_incorrect_component_mean": (
                full_metrics.best_incorrect_component_mean
            ),
            "full_target_margin_mean": full_metrics.target_margin_mean,
            "full_correct_predictions": full_metrics.correct_predictions,
            "full_total_examples": full_metrics.total_examples,
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
    output_concentration_penalty_weight: float,
    tensor_concentration_penalty_weight: float,
) -> EpochMetrics:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cross_entropy_loss = 0.0
    total_output_concentration_penalty = 0.0
    total_tensor_concentration_penalty = 0.0
    total_examples = 0
    correct_predictions = 0
    total_target_component = 0.0
    total_off_target_component = 0.0
    total_best_incorrect_component = 0.0
    total_target_margin = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            scores = model(inputs)
            (
                loss,
                cross_entropy_loss,
                output_concentration_penalty,
                tensor_concentration_penalty,
            ) = _compute_loss_components(
                effective_site_tensors=model.effective_site_tensors(),
                scores=scores,
                labels=labels,
                output_concentration_penalty_weight=(
                    output_concentration_penalty_weight
                ),
                tensor_concentration_penalty_weight=(
                    tensor_concentration_penalty_weight
                ),
                criterion=criterion,
            )
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        predictions = select_predicted_class(scores)
        output_metrics = _compute_output_metrics(scores=scores, labels=labels)
        total_loss += loss.item() * batch_size
        total_cross_entropy_loss += cross_entropy_loss.item() * batch_size
        total_output_concentration_penalty += (
            output_concentration_penalty.item() * batch_size
        )
        total_tensor_concentration_penalty += (
            tensor_concentration_penalty.item() * batch_size
        )
        total_examples += batch_size
        correct_predictions += int((predictions == labels).sum().item())
        total_target_component += output_metrics["target_component_mean"] * batch_size
        total_off_target_component += (
            output_metrics["off_target_component_mean"] * batch_size
        )
        total_best_incorrect_component += (
            output_metrics["best_incorrect_component_mean"] * batch_size
        )
        total_target_margin += output_metrics["target_margin_mean"] * batch_size
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / max(total_examples, 1)
    average_cross_entropy_loss = total_cross_entropy_loss / max(total_examples, 1)
    average_output_concentration_penalty = total_output_concentration_penalty / max(
        total_examples, 1
    )
    average_tensor_concentration_penalty = total_tensor_concentration_penalty / max(
        total_examples, 1
    )
    accuracy = correct_predictions / max(total_examples, 1)
    return EpochMetrics(
        loss=average_loss,
        cross_entropy_loss=average_cross_entropy_loss,
        output_concentration_penalty=average_output_concentration_penalty,
        tensor_concentration_penalty=average_tensor_concentration_penalty,
        accuracy=accuracy,
        target_component_mean=total_target_component / max(total_examples, 1),
        off_target_component_mean=total_off_target_component / max(total_examples, 1),
        best_incorrect_component_mean=(
            total_best_incorrect_component / max(total_examples, 1)
        ),
        target_margin_mean=total_target_margin / max(total_examples, 1),
        correct_predictions=correct_predictions,
        total_examples=total_examples,
        predictions=all_predictions,
        labels=all_labels,
    )


def _compute_loss_components(
    *,
    effective_site_tensors: list[torch.Tensor],
    scores: torch.Tensor,
    labels: torch.Tensor,
    output_concentration_penalty_weight: float,
    tensor_concentration_penalty_weight: float,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    cross_entropy_loss = resolved_criterion(scores, labels)
    output_concentration_penalty = _compute_output_concentration_penalty(scores)
    tensor_concentration_penalty = _compute_tensor_concentration_penalty(
        effective_site_tensors=effective_site_tensors
    )
    total_loss = (
        cross_entropy_loss
        + output_concentration_penalty_weight * output_concentration_penalty
        + tensor_concentration_penalty_weight * tensor_concentration_penalty
    )
    return (
        total_loss,
        cross_entropy_loss,
        output_concentration_penalty,
        tensor_concentration_penalty,
    )


def _compute_output_concentration_penalty(
    scores: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if scores.shape[1] <= 1:
        return scores.new_tensor(0.0)
    probabilities = torch.softmax(scores, dim=1)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=1)
    return (entropy / math.log(float(scores.shape[1]))).mean()


def _compute_tensor_concentration_penalty(
    effective_site_tensors: list[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for tensor in effective_site_tensors:
        flattened = tensor.reshape(-1)
        if flattened.numel() <= 1:
            penalties.append(flattened.new_tensor(0.0))
            continue
        probabilities = flattened / (flattened.sum() + eps)
        entropy = -(probabilities * torch.log(probabilities + eps)).sum()
        penalties.append(entropy / math.log(float(flattened.numel())))
    if not penalties:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.stack(penalties).mean()


def _compute_output_metrics(
    *,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    target_mask = F.one_hot(labels, num_classes=scores.shape[1]).to(torch.bool)
    target_components = scores.masked_select(target_mask)
    off_target_components = scores.masked_select(~target_mask)
    incorrect_scores = scores.masked_fill(target_mask, float("-inf"))
    best_incorrect_components = incorrect_scores.max(dim=1).values
    target_margins = target_components - best_incorrect_components
    return {
        "target_component_mean": float(target_components.mean().item()),
        "off_target_component_mean": float(off_target_components.mean().item()),
        "best_incorrect_component_mean": float(best_incorrect_components.mean().item()),
        "target_margin_mean": float(target_margins.mean().item()),
    }


def _log_epoch(
    *,
    epoch: int,
    total_epochs: int,
    train_metrics: EpochMetrics,
    full_metrics: EpochMetrics,
) -> None:
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train loss={train_metrics.loss:.4f} "
        f"ce={train_metrics.cross_entropy_loss:.4f} "
        f"out_conc={train_metrics.output_concentration_penalty:.4f} "
        f"tensor_conc={train_metrics.tensor_concentration_penalty:.4f} "
        f"acc={train_metrics.accuracy:.4f} "
        f"target={train_metrics.target_component_mean:.4f} "
        f"off={train_metrics.off_target_component_mean:.4f} "
        f"margin={train_metrics.target_margin_mean:.4f} | "
        f"full loss={full_metrics.loss:.4f} "
        f"ce={full_metrics.cross_entropy_loss:.4f} "
        f"out_conc={full_metrics.output_concentration_penalty:.4f} "
        f"tensor_conc={full_metrics.tensor_concentration_penalty:.4f} "
        f"acc={full_metrics.accuracy:.4f} "
        f"target={full_metrics.target_component_mean:.4f} "
        f"off={full_metrics.off_target_component_mean:.4f} "
        f"margin={full_metrics.target_margin_mean:.4f}"
    )


def _log_final_summary(*, full_metrics: EpochMetrics) -> None:
    print("Final full metrics:")
    print(
        "  accuracy: "
        f"{full_metrics.accuracy:.4f} "
        f"({full_metrics.correct_predictions}/{full_metrics.total_examples})"
    )
    print(f"  total_loss: {full_metrics.loss:.4f}")
    print(f"  cross_entropy_loss: {full_metrics.cross_entropy_loss:.4f}")
    print(
        "  output_concentration_penalty: "
        f"{full_metrics.output_concentration_penalty:.4f}"
    )
    print(
        "  tensor_concentration_penalty: "
        f"{full_metrics.tensor_concentration_penalty:.4f}"
    )
    print(f"  target_component_mean: {full_metrics.target_component_mean:.4f}")
    print(f"  off_target_component_mean: {full_metrics.off_target_component_mean:.4f}")
    print(
        "  best_incorrect_component_mean: "
        f"{full_metrics.best_incorrect_component_mean:.4f}"
    )
    print(f"  target_margin_mean: {full_metrics.target_margin_mean:.4f}")


def _is_better_full_checkpoint(
    *,
    full_accuracy: float,
    full_loss: float,
    best_full_accuracy: float,
    best_full_loss: float,
) -> bool:
    if full_accuracy > best_full_accuracy:
        return True
    if full_accuracy < best_full_accuracy:
        return False
    return full_loss < best_full_loss


def _reached_perfect_full_accuracy(full_accuracy: float) -> bool:
    return full_accuracy >= 1.0


def _log_perfect_accuracy_stop(*, epoch: int) -> None:
    print(f"Reached perfect full accuracy at epoch {epoch}. Stopping early.")


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
    fieldnames = [
        "epoch",
        "train_loss",
        "train_cross_entropy_loss",
        "train_output_concentration_penalty",
        "train_tensor_concentration_penalty",
        "full_loss",
        "train_accuracy",
        "train_target_component_mean",
        "train_off_target_component_mean",
        "train_best_incorrect_component_mean",
        "train_target_margin_mean",
        "full_cross_entropy_loss",
        "full_output_concentration_penalty",
        "full_tensor_concentration_penalty",
        "full_accuracy",
        "full_target_component_mean",
        "full_off_target_component_mean",
        "full_best_incorrect_component_mean",
        "full_target_margin_mean",
    ]
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
