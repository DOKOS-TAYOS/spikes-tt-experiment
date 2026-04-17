from __future__ import annotations

import contextlib
import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from tensor_network_viz import PlotConfig, TensorElementsConfig, show_tensor_network

from spike_mps.models.mps_classifier import (
    ManualMPSNetwork,
    MPSClassifier,
    select_predicted_class,
)
from spike_mps.training.checkpoints import load_model_from_checkpoint
from spike_mps.training.data import (
    DatasetBundle,
    DatasetRecord,
    encode_spike_train,
    load_dataset_bundle,
)

DatasetSplitName = Literal["full", "train", "val", "test"]


@dataclass(frozen=True)
class SampleVisualizationSummary:
    sample_index: int
    spike_train: str
    label: int
    clean_label: int
    is_noisy: bool
    predicted_label: int
    scores: tuple[float, ...]


def visualize_checkpoint(
    *,
    checkpoint_path: Path,
    show: bool = True,
) -> tuple[object, object]:
    # if show:
    #     backend = matplotlib.get_backend().lower()
    #     if "agg" in backend:
    #         raise RuntimeError(
    #             "Interactive visualization requires a GUI backend. "
    #             "Run in a GUI environment or pass --no-show."
    #         )

    model, _checkpoint = load_model_from_checkpoint(checkpoint_path)
    display_network = model.build_visualization_network()
    display_network.reset()
    visible_nodes = _build_visible_mps_nodes(network=display_network)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using a non-tuple sequence for multidimensional indexing.*",
            category=UserWarning,
        )
        fig, ax = show_tensor_network(
            visible_nodes,
            engine="tensorkrowch",
            config=_build_plot_config(),
            show=show,
        )
    _close_figure(fig=fig, show=show)
    return fig, ax


def visualize_checkpoint_per_sample(
    *,
    checkpoint_path: Path,
    split: DatasetSplitName = "full",
    limit: int | None = None,
    show: bool = True,
) -> list[SampleVisualizationSummary]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to zero.")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    dataset_dir = _resolve_dataset_dir(
        checkpoint_path=checkpoint_path,
        dataset_name=str(checkpoint["dataset_name"]),
    )
    dataset_bundle = load_dataset_bundle(dataset_dir=dataset_dir)
    records = _select_split_records(dataset_bundle=dataset_bundle, split=split)

    summaries: list[SampleVisualizationSummary] = []
    for sample_index, record in enumerate(records):
        if limit is not None and sample_index >= limit:
            break

        scores = _contract_sample(model=model, record=record)
        predicted_label = int(select_predicted_class(scores).item())
        score_values = tuple(float(value) for value in scores.squeeze(0).tolist())
        display_network = model.build_visualization_network()
        _contract_display_network(network=display_network, record=record)
        visible_nodes = _build_visible_contracted_nodes(network=display_network)
        fig, _ax = show_tensor_network(
            visible_nodes,
            engine="tensorkrowch",
            config=_build_plot_config(
                show_contraction_scheme=True,
                contraction_scheme_by_name=_build_contraction_scheme_by_name(
                    visible_nodes=visible_nodes,
                ),
            ),
            show=show,
        )
        _close_figure(fig=fig, show=show)
        summaries.append(
            SampleVisualizationSummary(
                sample_index=sample_index,
                spike_train=record.spike_train,
                label=record.label,
                clean_label=record.clean_label,
                is_noisy=record.is_noisy,
                predicted_label=predicted_label,
                scores=score_values,
            )
        )
    return summaries


def _contract_sample(
    *,
    model: MPSClassifier,
    record: DatasetRecord,
) -> torch.Tensor:
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
        model.network.reset()
        reference_parameter = next(model.parameters())
        encoded_sample = (
            encode_spike_train(record.spike_train)
            .unsqueeze(0)
            .to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        )
        with torch.no_grad():
            return model(encoded_sample).detach().cpu()


def _contract_display_network(
    *,
    network: ManualMPSNetwork,
    record: DatasetRecord,
) -> None:
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
        network.reset()
        reference_tensor = network.site_nodes[0].tensor
        encoded_sample = (
            encode_spike_train(record.spike_train)
            .unsqueeze(0)
            .to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )
        )
        with torch.no_grad():
            network(encoded_sample)


def _build_visible_contracted_nodes(*, network: ManualMPSNetwork) -> list[object]:
    data_nodes = list(network.data_nodes.values())
    return [*network.site_nodes, *data_nodes]


def _build_visible_mps_nodes(*, network: ManualMPSNetwork) -> list[object]:
    return list(network.site_nodes)


def _build_plot_config(
    *,
    show_contraction_scheme: bool = False,
    contraction_scheme_by_name: tuple[tuple[str, ...], ...] | None = None,
) -> PlotConfig:
    plot_config_kwargs: dict[str, object] = {
        "show_contraction_scheme": show_contraction_scheme,
        "contraction_tensor_inspector": True,
        "theme": "paper",
        "contraction_scheme_by_name": contraction_scheme_by_name,
    }
    if _supports_keyword_argument(
        callable_object=TensorElementsConfig,
        parameter_name="theme",
    ) and _supports_keyword_argument(
        callable_object=PlotConfig,
        parameter_name="tensor_inspector_config",
    ):
        plot_config_kwargs["tensor_inspector_config"] = _build_tensor_inspector_config()
    return PlotConfig(**plot_config_kwargs)


def _build_tensor_inspector_config() -> TensorElementsConfig:
    return TensorElementsConfig(theme="grayscale")


def _supports_keyword_argument(
    *,
    callable_object: object,
    parameter_name: str,
) -> bool:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


def _build_contraction_scheme_by_name(
    *,
    visible_nodes: list[object],
) -> tuple[tuple[str, ...], ...]:
    site_names = [
        str(node.name)
        for node in visible_nodes
        if str(getattr(node, "name", "")).startswith("site_")
    ]
    data_names = [
        str(node.name)
        for node in visible_nodes
        if str(getattr(node, "name", "")).startswith("data_")
    ]
    if len(site_names) != len(data_names):
        raise ValueError(
            "The visible contracted network must contain one data node per site node."
        )

    contraction_steps: list[tuple[str, ...]] = []
    cumulative_names: list[str] = []
    for site_name, data_name in zip(site_names, data_names, strict=True):
        pair_step = (site_name, data_name)
        contraction_steps.append(pair_step)
        cumulative_names.extend(pair_step)
        if len(cumulative_names) > 2:
            contraction_steps.append(tuple(cumulative_names))
    return tuple(contraction_steps)


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


def _select_split_records(
    *,
    dataset_bundle: DatasetBundle,
    split: DatasetSplitName,
) -> list[DatasetRecord]:
    split_records = {
        "full": dataset_bundle.full_dataset.records,
        "train": dataset_bundle.train_dataset.records,
        "val": dataset_bundle.val_dataset.records,
        "test": dataset_bundle.test_dataset.records,
    }
    return list(split_records[split])


def _close_figure(*, fig: object, show: bool) -> None:
    if show:
        return
    with contextlib.suppress(TypeError, ValueError):
        plt.close(fig)
