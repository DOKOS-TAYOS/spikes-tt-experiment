from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import tensorkrowch as tk
import torch
from torch import nn

from spike_mps.config import TaskName


@dataclass(frozen=True)
class MPSModelConfig:
    sequence_length: int
    input_dim: int
    num_classes: int
    bond_dim: int
    task: TaskName

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.bond_dim <= 0:
            raise ValueError("bond_dim must be greater than zero.")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "bond_dim": self.bond_dim,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MPSModelConfig:
        return cls(
            sequence_length=int(data["sequence_length"]),
            input_dim=int(data["input_dim"]),
            num_classes=int(data["num_classes"]),
            bond_dim=int(data["bond_dim"]),
            task=str(data["task"]),
        )


class MPSClassifier(nn.Module):
    def __init__(
        self,
        config: MPSModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.network = tk.models.MPSLayer(
            n_features=config.sequence_length,
            in_dim=config.input_dim,
            out_dim=config.num_classes,
            bond_dim=config.bond_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
            return self.network(inputs)
