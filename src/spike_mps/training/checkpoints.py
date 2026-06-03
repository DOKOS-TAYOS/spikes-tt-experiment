from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from spike_mps.filesystem import open_binary_for_read, open_binary_for_write
from spike_mps.models.mps_classifier import MPSClassifier, MPSModelConfig

FORMAT_VERSION = 2


def save_checkpoint(
    *,
    path: Path,
    model: MPSClassifier,
    experiment_name: str,
    dataset_name: str,
    experiment_config: dict[str, Any],
    best_epoch: int,
    best_full_loss: float,
    metrics: dict[str, float],
    seed: int,
) -> None:
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "model_config": model.config.to_dict(),
        "parameterization": model.config.parameterization,
        "experiment_config": experiment_config,
        "best_epoch": best_epoch,
        "best_full_loss": best_full_loss,
        "metrics": metrics,
        "seed": seed,
        "state_dict": _build_raw_site_state_dict(model=model),
    }
    with open_binary_for_write(path) as file_handle:
        torch.save(checkpoint, file_handle)


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    with open_binary_for_read(path) as file_handle:
        return torch.load(file_handle, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MPSClassifier, dict[str, Any]]:
    checkpoint = load_checkpoint(path=path, map_location=map_location)
    raw_model_config = dict(checkpoint["model_config"])
    raw_model_config.setdefault(
        "parameterization",
        str(checkpoint.get("parameterization", "direct")),
    )
    model_config = MPSModelConfig.from_dict(raw_model_config)
    model = MPSClassifier(config=model_config)
    state_dict = _normalize_state_dict(
        state_dict=dict(checkpoint["state_dict"]),
        model_config=model_config,
    )
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model, checkpoint


def _build_raw_site_state_dict(*, model: MPSClassifier) -> dict[str, torch.Tensor]:
    return {
        f"network.param_site_{index}": node.tensor.detach().clone()
        for index, node in enumerate(model.network.site_nodes)
    }


def _normalize_state_dict(
    *,
    state_dict: dict[str, torch.Tensor],
    model_config: MPSModelConfig,
) -> dict[str, torch.Tensor]:
    if "network.param_virtual_result_stack" not in state_dict:
        return state_dict

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for index in range(model_config.sequence_length):
        site_key = f"network.param_site_{index}"
        if site_key in state_dict:
            normalized_state_dict[site_key] = state_dict[site_key]

    stacked_key = "network.param_virtual_result_stack"
    stacked_tensor = state_dict[stacked_key]
    expected_internal_sites = max(model_config.sequence_length - 2, 0)
    if int(stacked_tensor.shape[0]) != expected_internal_sites:
        raise ValueError(
            "Legacy checkpoint internal site count does not match the model "
            "sequence length."
        )
    for offset, tensor in enumerate(torch.unbind(stacked_tensor, dim=0), start=1):
        normalized_state_dict[f"network.param_site_{offset}"] = tensor
    return normalized_state_dict
