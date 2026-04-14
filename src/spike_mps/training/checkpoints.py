from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from spike_mps.models.mps_classifier import MPSClassifier, MPSModelConfig

FORMAT_VERSION = 1


def save_checkpoint(
    *,
    path: Path,
    model: MPSClassifier,
    experiment_name: str,
    dataset_name: str,
    experiment_config: dict[str, Any],
    best_epoch: int,
    best_val_loss: float,
    metrics: dict[str, float],
    seed: int,
) -> None:
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "model_config": model.config.to_dict(),
        "experiment_config": experiment_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "seed": seed,
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[MPSClassifier, dict[str, Any]]:
    checkpoint = load_checkpoint(path=path, map_location=map_location)
    model_config = MPSModelConfig.from_dict(checkpoint["model_config"])
    model = MPSClassifier(config=model_config)
    state_dict = checkpoint["state_dict"]
    if any("virtual_result_stack" in key for key in state_dict):
        example = torch.zeros(
            (1, model_config.sequence_length, model_config.input_dim),
            dtype=torch.float32,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`tensor` is being cropped to fit the shape of node .*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Using a non-tuple sequence for multidimensional indexing.*",
                category=UserWarning,
            )
            model.network.trace(example)
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model, checkpoint
