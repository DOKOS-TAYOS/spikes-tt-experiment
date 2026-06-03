from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spike_mps.models.mps_classifier import (
    MPSClassifier,
    MPSModelConfig,
    select_predicted_class,
)
from spike_mps.training.checkpoints import load_model_from_checkpoint, save_checkpoint
from spike_mps.training.data import DatasetBundle, load_dataset_bundle


@dataclass(frozen=True)
class BondOptimizationResult:
    u_matrix: torch.Tensor
    concentration_before: float
    concentration_after: float


@dataclass(frozen=True)
class BondConcentrationResult:
    bond_index: int
    concentration_before: float
    concentration_after: float


@dataclass(frozen=True)
class ModelConcentrationResult:
    model: MPSClassifier
    concentration_before: float
    concentration_after: float
    bond_results: tuple[BondConcentrationResult, ...]


@dataclass(frozen=True)
class CheckpointConcentrationResult:
    output_path: Path
    accuracy_before: float
    accuracy_after: float
    concentration_before: float
    concentration_after: float
    max_score_difference: float


@dataclass(frozen=True)
class _EvaluationResult:
    accuracy: float
    scores: torch.Tensor


def compute_magnitude_entropy(
    tensor: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    magnitudes = tensor.abs().reshape(-1)
    if magnitudes.numel() <= 1:
        return magnitudes.new_tensor(0.0)

    total_mass = magnitudes.sum()
    if float(total_mass.detach().cpu()) <= eps:
        return magnitudes.new_tensor(0.0)

    probabilities = magnitudes / (total_mass + eps)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum()
    return entropy / math.log(float(magnitudes.numel()))


def compute_network_magnitude_entropy(site_tensors: list[torch.Tensor]) -> float:
    if not site_tensors:
        return 0.0
    entropies = [compute_magnitude_entropy(tensor) for tensor in site_tensors]
    return float(torch.stack(entropies).mean().item())


def apply_bond_transform(
    *,
    site_tensors: list[torch.Tensor],
    bond_index: int,
    u_matrix: torch.Tensor,
) -> list[torch.Tensor]:
    _validate_bond_transform_inputs(
        site_tensors=site_tensors,
        bond_index=bond_index,
        u_matrix=u_matrix,
    )
    transformed_tensors = [tensor.clone() for tensor in site_tensors]
    transformed_tensors[bond_index] = torch.tensordot(
        site_tensors[bond_index],
        u_matrix,
        dims=([-1], [0]),
    )
    transformed_tensors[bond_index + 1] = torch.tensordot(
        u_matrix.T,
        site_tensors[bond_index + 1],
        dims=([1], [0]),
    )
    return transformed_tensors


def concentrate_model(
    *,
    model: MPSClassifier,
    steps: int,
    learning_rate: float,
    restarts: int,
    seed: int,
) -> ModelConcentrationResult:
    site_tensors = [
        tensor.detach().clone() for tensor in model.effective_site_tensors()
    ]
    concentration_before = compute_network_magnitude_entropy(site_tensors)
    bond_results: list[BondConcentrationResult] = []

    for bond_index in range(len(site_tensors) - 1):
        bond_concentration_before = float(
            compute_magnitude_entropy(site_tensors[bond_index]).item()
        )
        optimization_result = optimize_bond_transform(
            site_tensor=site_tensors[bond_index],
            steps=steps,
            learning_rate=learning_rate,
            restarts=restarts,
            seed=seed + bond_index,
        )
        site_tensors = apply_bond_transform(
            site_tensors=site_tensors,
            bond_index=bond_index,
            u_matrix=optimization_result.u_matrix,
        )
        bond_concentration_after = float(
            compute_magnitude_entropy(site_tensors[bond_index]).item()
        )
        bond_results.append(
            BondConcentrationResult(
                bond_index=bond_index,
                concentration_before=bond_concentration_before,
                concentration_after=bond_concentration_after,
            )
        )

    concentrated_model = _build_direct_model_from_site_tensors(
        source_model=model,
        site_tensors=site_tensors,
    )
    return ModelConcentrationResult(
        model=concentrated_model,
        concentration_before=concentration_before,
        concentration_after=compute_network_magnitude_entropy(site_tensors),
        bond_results=tuple(bond_results),
    )


def concentrate_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path | None = None,
    steps: int,
    learning_rate: float,
    restarts: int,
    seed: int,
    max_score_difference_tolerance: float = 1e-4,
) -> CheckpointConcentrationResult:
    model, checkpoint = load_model_from_checkpoint(
        path=checkpoint_path,
        map_location="cpu",
    )
    dataset_dir = _resolve_dataset_dir(
        checkpoint_path=checkpoint_path,
        dataset_name=str(checkpoint["dataset_name"]),
    )
    dataset_bundle = load_dataset_bundle(dataset_dir=dataset_dir)

    before_evaluation = _evaluate_model(model=model, dataset_bundle=dataset_bundle)
    if before_evaluation.accuracy < 1.0:
        raise RuntimeError(
            "The checkpoint must already reach 100% accuracy on the full dataset "
            "before concentration."
        )

    concentration_result = concentrate_model(
        model=model,
        steps=steps,
        learning_rate=learning_rate,
        restarts=restarts,
        seed=seed,
    )
    after_evaluation = _evaluate_model(
        model=concentration_result.model,
        dataset_bundle=dataset_bundle,
    )
    max_score_difference = float(
        (before_evaluation.scores - after_evaluation.scores).abs().max().item()
    )
    if after_evaluation.accuracy < 1.0:
        raise RuntimeError(
            "Concentration changed the predictions and the model no longer "
            "reaches 100% accuracy on the full dataset."
        )
    if max_score_difference > max_score_difference_tolerance:
        raise RuntimeError(
            "Concentration changed model scores more than allowed: "
            f"{max_score_difference:.6g} > {max_score_difference_tolerance:.6g}."
        )

    resolved_output_path = (
        output_path
        if output_path is not None
        else checkpoint_path.with_name("checkpoint_concentrated.pt")
    )
    save_checkpoint(
        path=resolved_output_path,
        model=concentration_result.model,
        experiment_name=str(checkpoint["experiment_name"]),
        dataset_name=str(checkpoint["dataset_name"]),
        experiment_config=dict(checkpoint["experiment_config"]),
        best_epoch=int(checkpoint["best_epoch"]),
        best_full_loss=float(checkpoint["best_full_loss"]),
        metrics={
            **{
                str(key): float(value)
                for key, value in dict(checkpoint.get("metrics", {})).items()
            },
            "concentration_before": concentration_result.concentration_before,
            "concentration_after": concentration_result.concentration_after,
            "concentration_accuracy_before": before_evaluation.accuracy,
            "concentration_accuracy_after": after_evaluation.accuracy,
            "concentration_max_score_difference": max_score_difference,
        },
        seed=int(checkpoint["seed"]),
    )
    return CheckpointConcentrationResult(
        output_path=resolved_output_path,
        accuracy_before=before_evaluation.accuracy,
        accuracy_after=after_evaluation.accuracy,
        concentration_before=concentration_result.concentration_before,
        concentration_after=concentration_result.concentration_after,
        max_score_difference=max_score_difference,
    )


def optimize_bond_transform(
    *,
    site_tensor: torch.Tensor,
    steps: int,
    learning_rate: float,
    restarts: int,
    seed: int,
) -> BondOptimizationResult:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if restarts <= 0:
        raise ValueError("restarts must be greater than zero.")
    if site_tensor.ndim < 1:
        raise ValueError("site_tensor must have at least one axis.")

    bond_dim = int(site_tensor.shape[-1])
    identity = torch.eye(
        bond_dim,
        device=site_tensor.device,
        dtype=site_tensor.dtype,
    )
    concentration_before = float(compute_magnitude_entropy(site_tensor).item())
    if bond_dim <= 1 or steps == 0:
        return BondOptimizationResult(
            u_matrix=identity,
            concentration_before=concentration_before,
            concentration_after=concentration_before,
        )

    generator = torch.Generator(device=site_tensor.device)
    generator.manual_seed(seed)
    best_u_matrix = identity
    best_concentration = float(
        compute_magnitude_entropy(
            torch.tensordot(site_tensor, identity, dims=([-1], [0]))
        ).item()
    )

    for _ in range(restarts):
        raw_matrix = _initial_skew_source(
            size=bond_dim,
            generator=generator,
            reference_tensor=site_tensor,
        )
        optimizer = torch.optim.Adam([raw_matrix], lr=learning_rate)

        for _ in range(steps):
            optimizer.zero_grad()
            u_matrix = _orthogonal_from_raw(raw_matrix)
            transformed_tensor = torch.tensordot(
                site_tensor,
                u_matrix,
                dims=([-1], [0]),
            )
            loss = compute_magnitude_entropy(transformed_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            candidate_u_matrix = _orthogonal_from_raw(raw_matrix)
            candidate_concentration = float(
                compute_magnitude_entropy(
                    torch.tensordot(
                        site_tensor,
                        candidate_u_matrix,
                        dims=([-1], [0]),
                    )
                ).item()
            )
        if candidate_concentration < best_concentration:
            best_concentration = candidate_concentration
            best_u_matrix = candidate_u_matrix.detach().clone()

    return BondOptimizationResult(
        u_matrix=best_u_matrix,
        concentration_before=concentration_before,
        concentration_after=best_concentration,
    )


def _initial_skew_source(
    *,
    size: int,
    generator: torch.Generator,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    initial_value = 0.05 * torch.randn(
        (size, size),
        generator=generator,
        device=reference_tensor.device,
        dtype=reference_tensor.dtype,
    )
    return torch.nn.Parameter(initial_value)


def _orthogonal_from_raw(raw_matrix: torch.Tensor) -> torch.Tensor:
    skew_matrix = raw_matrix - raw_matrix.T
    return torch.linalg.matrix_exp(skew_matrix)


def _validate_bond_transform_inputs(
    *,
    site_tensors: list[torch.Tensor],
    bond_index: int,
    u_matrix: torch.Tensor,
) -> None:
    if len(site_tensors) < 2:
        raise ValueError("At least two site tensors are needed for a bond transform.")
    if bond_index < 0 or bond_index >= len(site_tensors) - 1:
        raise ValueError("bond_index must identify an internal MPS bond.")
    if u_matrix.ndim != 2 or u_matrix.shape[0] != u_matrix.shape[1]:
        raise ValueError("u_matrix must be a square matrix.")
    left_bond_dim = int(site_tensors[bond_index].shape[-1])
    right_bond_dim = int(site_tensors[bond_index + 1].shape[0])
    u_dim = int(u_matrix.shape[0])
    if left_bond_dim != right_bond_dim or u_dim != left_bond_dim:
        raise ValueError("u_matrix shape must match the selected bond dimension.")


def _build_direct_model_from_site_tensors(
    *,
    source_model: MPSClassifier,
    site_tensors: list[torch.Tensor],
) -> MPSClassifier:
    config = _build_config_from_site_tensors(
        site_tensors=site_tensors,
        task=source_model.config.task,
        fallback_bond_dim=source_model.config.bond_dim,
    )
    concentrated_model = MPSClassifier(
        config=config,
        device=_resolve_model_device(model=source_model),
        dtype=_resolve_model_dtype(model=source_model),
    )
    for node, tensor in zip(
        concentrated_model.network.site_nodes,
        site_tensors,
        strict=True,
    ):
        node.tensor = tensor.to(
            device=_resolve_model_device(model=source_model),
            dtype=_resolve_model_dtype(model=source_model),
        )
    concentrated_model.eval()
    return concentrated_model


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


def _evaluate_model(
    *,
    model: MPSClassifier,
    dataset_bundle: DatasetBundle,
) -> _EvaluationResult:
    dataloader = DataLoader(dataset_bundle.full_dataset, batch_size=64, shuffle=False)
    total_examples = 0
    correct_predictions = 0
    score_batches: list[torch.Tensor] = []
    device = _resolve_model_device(model=model)
    dtype = _resolve_model_dtype(model=model)

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            scores = model(inputs)
            predictions = select_predicted_class(scores)
            correct_predictions += int((predictions == labels).sum().item())
            total_examples += int(labels.shape[0])
            score_batches.append(scores.detach().cpu())
    return _EvaluationResult(
        accuracy=correct_predictions / max(total_examples, 1),
        scores=torch.cat(score_batches, dim=0),
    )


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
