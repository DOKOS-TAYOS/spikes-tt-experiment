from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from spike_mps.models.mps_classifier import (
    MPSClassifier,
    MPSModelConfig,
    select_predicted_class,
)
from spike_mps.training.checkpoints import load_model_from_checkpoint, save_checkpoint
from spike_mps.training.data import DatasetBundle, load_dataset_bundle

CanonicalizationMode = Literal["svd", "qr"]


@dataclass(frozen=True)
class CanonicalizationResult:
    output_path: Path
    accuracy_before: float
    accuracy_after: float
    bond_dims_before: tuple[int, ...]
    bond_dims_after: tuple[int, ...]


def canonicalize_model(
    *,
    model: MPSClassifier,
    mode: CanonicalizationMode = "svd",
) -> MPSClassifier:
    site_tensors = [
        tensor.detach().clone() for tensor in model.effective_site_tensors()
    ]
    canonical_site_tensors = _left_canonicalize_site_tensors(
        site_tensors=site_tensors,
        mode=mode,
    )
    canonical_config = _build_config_from_site_tensors(
        site_tensors=canonical_site_tensors,
        task=model.config.task,
        fallback_bond_dim=model.config.bond_dim,
    )
    canonical_model = MPSClassifier(config=canonical_config)
    canonical_model.to(_resolve_model_device(model=model))
    for node, tensor in zip(
        canonical_model.network.site_nodes,
        canonical_site_tensors,
        strict=True,
    ):
        node.tensor = tensor.to(
            device=_resolve_model_device(model=model),
            dtype=_resolve_model_dtype(model=model),
        )
    canonical_model.eval()
    return canonical_model


def canonicalize_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path | None = None,
    mode: CanonicalizationMode = "svd",
) -> CanonicalizationResult:
    model, checkpoint = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location="cpu",
    )
    dataset_dir = _resolve_dataset_dir(
        checkpoint_path=checkpoint_path,
        dataset_name=str(checkpoint["dataset_name"]),
    )
    dataset_bundle = load_dataset_bundle(dataset_dir=dataset_dir)

    accuracy_before = _evaluate_full_accuracy(
        model=model,
        dataset_bundle=dataset_bundle,
    )
    if accuracy_before < 1.0:
        raise RuntimeError(
            "The checkpoint must already reach 100% accuracy on the full dataset "
            "before canonicalization."
        )

    canonical_model = canonicalize_model(model=model, mode=mode)
    accuracy_after = _evaluate_full_accuracy(
        model=canonical_model,
        dataset_bundle=dataset_bundle,
    )
    if accuracy_after < 1.0:
        raise RuntimeError(
            "Canonicalization changed the predictions and the model no longer "
            "reaches 100% accuracy on the full dataset."
        )

    resolved_output_path = (
        output_path
        if output_path is not None
        else checkpoint_path.with_name("checkpoint_canonical.pt")
    )
    save_checkpoint(
        path=resolved_output_path,
        model=canonical_model,
        experiment_name=str(checkpoint["experiment_name"]),
        dataset_name=str(checkpoint["dataset_name"]),
        experiment_config=dict(checkpoint["experiment_config"]),
        best_epoch=int(checkpoint["best_epoch"]),
        best_full_loss=float(checkpoint["best_full_loss"]),
        metrics={
            str(key): float(value)
            for key, value in dict(checkpoint.get("metrics", {})).items()
        },
        seed=int(checkpoint["seed"]),
    )
    return CanonicalizationResult(
        output_path=resolved_output_path,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        bond_dims_before=model.config.resolved_bond_dims,
        bond_dims_after=canonical_model.config.resolved_bond_dims,
    )


def _left_canonicalize_site_tensors(
    *,
    site_tensors: list[torch.Tensor],
    mode: CanonicalizationMode,
) -> list[torch.Tensor]:
    if mode not in {"svd", "qr"}:
        raise ValueError("mode must be either 'svd' or 'qr'.")
    if len(site_tensors) <= 1:
        return [tensor.clone() for tensor in site_tensors]

    canonical_tensors = [tensor.clone() for tensor in site_tensors]
    for index in range(len(canonical_tensors) - 1):
        current_tensor = canonical_tensors[index]
        reshaped_tensor = current_tensor.reshape(-1, current_tensor.shape[-1])
        left_tensor, transfer_tensor = _factor_site_matrix(
            matrix=reshaped_tensor,
            mode=mode,
        )
        canonical_tensors[index] = left_tensor.reshape(
            *current_tensor.shape[:-1],
            left_tensor.shape[1],
        )
        canonical_tensors[index + 1] = _absorb_transfer_into_next_site(
            transfer_tensor=transfer_tensor,
            next_tensor=canonical_tensors[index + 1],
        )
    return canonical_tensors


def _factor_site_matrix(
    *,
    matrix: torch.Tensor,
    mode: CanonicalizationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "qr":
        q_matrix, r_matrix = torch.linalg.qr(matrix, mode="reduced")
        return q_matrix, r_matrix

    u_matrix, singular_values, vh_matrix = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    transfer_tensor = singular_values.unsqueeze(1) * vh_matrix
    return u_matrix, transfer_tensor


def _absorb_transfer_into_next_site(
    *,
    transfer_tensor: torch.Tensor,
    next_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ab,bcd->acd", transfer_tensor, next_tensor)


def _build_config_from_site_tensors(
    *,
    site_tensors: list[torch.Tensor],
    task: str,
    fallback_bond_dim: int | tuple[int, ...],
) -> MPSModelConfig:
    if len(site_tensors) == 1:
        return MPSModelConfig(
            sequence_length=1,
            input_dim=int(site_tensors[0].shape[0]),
            num_classes=int(site_tensors[0].shape[1]),
            bond_dim=fallback_bond_dim,
            task=task,
            parameterization="direct",
        )
    bond_dims = tuple(int(site_tensor.shape[-1]) for site_tensor in site_tensors[:-1])
    return MPSModelConfig(
        sequence_length=len(site_tensors),
        input_dim=int(site_tensors[0].shape[0]),
        num_classes=int(site_tensors[-1].shape[-1]),
        bond_dim=bond_dims,
        task=task,
        parameterization="direct",
    )


def _evaluate_full_accuracy(
    *,
    model: MPSClassifier,
    dataset_bundle: DatasetBundle,
) -> float:
    dataloader = DataLoader(dataset_bundle.full_dataset, batch_size=64, shuffle=False)
    total_examples = 0
    correct_predictions = 0
    device = _resolve_model_device(model=model)
    dtype = _resolve_model_dtype(model=model)

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            predictions = select_predicted_class(model(inputs))
            correct_predictions += int((predictions == labels).sum().item())
            total_examples += int(labels.shape[0])
    return correct_predictions / max(total_examples, 1)


def _resolve_dataset_dir(
    *,
    checkpoint_path: Path,
    dataset_name: str,
) -> Path:
    search_roots = [Path.cwd().resolve(), *checkpoint_path.resolve().parents]
    seen_roots: set[Path] = set()
    for root in search_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        candidate = root / "datasets" / "generated" / dataset_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset directory for '{dataset_name}' near {checkpoint_path}."
    )


def _resolve_model_device(*, model: MPSClassifier) -> torch.device:
    return next(model.parameters()).device


def _resolve_model_dtype(*, model: MPSClassifier) -> torch.dtype:
    return next(model.parameters()).dtype
