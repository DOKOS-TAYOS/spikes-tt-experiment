from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import tensorkrowch as tk
import torch
from torch import nn

from spike_mps.config import TaskName

type BondDimensionSpec = int | tuple[int, ...]


@dataclass(frozen=True)
class MPSModelConfig:
    sequence_length: int
    input_dim: int
    num_classes: int
    bond_dim: BondDimensionSpec
    task: TaskName

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if isinstance(self.bond_dim, int):
            if self.bond_dim <= 0:
                raise ValueError("bond_dim must be greater than zero.")
            return
        if len(self.bond_dim) != max(self.sequence_length - 1, 0):
            raise ValueError("bond_dim sequence must have one entry per internal bond.")
        if any(bond_dimension <= 0 for bond_dimension in self.bond_dim):
            raise ValueError("All bond dimensions must be greater than zero.")

    @property
    def resolved_bond_dims(self) -> tuple[int, ...]:
        if self.sequence_length == 1:
            return ()
        if isinstance(self.bond_dim, int):
            return (self.bond_dim,) * (self.sequence_length - 1)
        return self.bond_dim

    def to_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "bond_dim": (
                self.bond_dim if isinstance(self.bond_dim, int) else list(self.bond_dim)
            ),
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MPSModelConfig:
        raw_bond_dim = data["bond_dim"]
        bond_dim: BondDimensionSpec
        if isinstance(raw_bond_dim, list):
            bond_dim = tuple(int(value) for value in raw_bond_dim)
        else:
            bond_dim = int(raw_bond_dim)
        return cls(
            sequence_length=int(data["sequence_length"]),
            input_dim=int(data["input_dim"]),
            num_classes=int(data["num_classes"]),
            bond_dim=bond_dim,
            task=str(data["task"]),
        )


def select_predicted_class(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch_size, num_classes).")
    return torch.argmax(scores.abs(), dim=1)


class ManualMPSNetwork(tk.TensorNetwork):
    def __init__(
        self,
        config: MPSModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name="ManualMPSNetwork")
        self.config = config
        self.site_nodes = self._build_site_nodes(device=device, dtype=dtype)
        self.auto_stack = True
        self.auto_unbind = False

    def set_data_nodes(self) -> None:
        input_edges = [node["input"] for node in self.site_nodes]
        super().set_data_nodes(input_edges, num_batch_edges=1)

    def contract(self) -> tk.Node:
        data_nodes = list(self.data_nodes.values())

        if len(self.site_nodes) == 1:
            result = self.site_nodes[0] @ data_nodes[0]
        else:
            result = self.site_nodes[0] @ data_nodes[0]

            if len(self.site_nodes) > 2 and self._can_use_stacked_bulk_contraction():
                stacked_site_nodes = tk.stack(self.site_nodes[1:-1])
                stacked_data_nodes = tk.stack(data_nodes[1:-1])
                stacked_site_nodes ^ stacked_data_nodes
                contracted_bulk_nodes = tk.unbind(
                    stacked_site_nodes @ stacked_data_nodes
                )

                for node in contracted_bulk_nodes:
                    result @= node
            else:
                for site_node, data_node in zip(
                    self.site_nodes[1:-1], data_nodes[1:-1], strict=True
                ):
                    result @= site_node @ data_node

            last_result = self.site_nodes[-1] @ data_nodes[-1]
            result @= last_result

        return tk.permute(result, ("batch", "output"))

    def _can_use_stacked_bulk_contraction(self) -> bool:
        bulk_site_nodes = self.site_nodes[1:-1]
        if len(bulk_site_nodes) <= 1:
            return False
        reference_shape = tuple(
            int(dimension) for dimension in bulk_site_nodes[0].shape
        )
        return all(
            tuple(int(dimension) for dimension in node.shape) == reference_shape
            for node in bulk_site_nodes[1:]
        )

    def _build_site_nodes(
        self,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> list[tk.ParamNode]:
        site_nodes: list[tk.ParamNode] = []

        for index in range(self.config.sequence_length):
            shape, axes_names = self._site_spec(index=index)
            node = tk.ParamNode(
                shape=shape,
                axes_names=axes_names,
                name=f"site_{index}",
                network=self,
                device=device,
                dtype=dtype,
            )
            node.tensor = _initialize_site_tensor(
                shape=shape,
                device=device,
                dtype=dtype,
            )
            site_nodes.append(node)

        for index in range(len(site_nodes) - 1):
            site_nodes[index]["right"] ^ site_nodes[index + 1]["left"]

        return site_nodes

    def _site_spec(
        self,
        *,
        index: int,
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        bond_dims = self.config.resolved_bond_dims
        if self.config.sequence_length == 1:
            return (
                (self.config.input_dim, self.config.num_classes),
                ("input", "output"),
            )
        if index == 0:
            return (
                (self.config.input_dim, bond_dims[0]),
                ("input", "right"),
            )
        if index == self.config.sequence_length - 1:
            return (
                (bond_dims[-1], self.config.input_dim, self.config.num_classes),
                ("left", "input", "output"),
            )
        return (
            (bond_dims[index - 1], self.config.input_dim, bond_dims[index]),
            ("left", "input", "right"),
        )


def _initialize_site_tensor(
    *,
    shape: tuple[int, ...],
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
    std = 1.0 / math.sqrt(max(shape[0], shape[-1], 1))
    return torch.randn(shape, device=device, dtype=resolved_dtype) * std


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
        self.network = ManualMPSNetwork(
            config=config,
            device=device,
            dtype=dtype,
        )

    def prepare_for_training(self) -> None:
        reference_parameter = next(self.parameters())
        example = torch.zeros(
            (1, self.config.sequence_length, self.config.input_dim),
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
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
            self.network.trace(example)

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
            return self.network(inputs).squeeze(dim=1)
