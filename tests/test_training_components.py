from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from spike_mps.config import DatasetConfig, SplitRatios
from spike_mps.generation import build_dataset_artifacts
from spike_mps.models.mps_classifier import (
    MPSClassifier,
    MPSModelConfig,
    select_predicted_class,
)
from spike_mps.training.canonicalization import canonicalize_model
from spike_mps.training.checkpoints import (
    load_model_from_checkpoint,
    save_checkpoint,
)
from spike_mps.training.data import encode_spike_train, load_dataset_bundle
from spike_mps.training.runner import (
    _compute_loss_components,
    _compute_output_metrics,
    _is_better_full_checkpoint,
    _reached_perfect_full_accuracy,
)
from spike_mps.training.visualization import (
    visualize_checkpoint,
    visualize_checkpoint_per_sample,
)
from spike_mps.writer import write_dataset_artifacts


def test_encode_spike_train_returns_local_feature_map() -> None:
    encoded = encode_spike_train("0101")

    assert encoded.shape == (4, 2)
    assert torch.equal(
        encoded,
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_load_dataset_bundle_infers_num_classes_from_metadata(
    workspace_dir: Path,
) -> None:
    dataset_dir = _create_dataset_directory(
        workspace_dir=workspace_dir,
        name="bundle_example",
        task="adjacent_ones_score",
        sequence_length=4,
        dataset_size=16,
    )

    bundle = load_dataset_bundle(dataset_dir=dataset_dir)

    assert bundle.dataset_name == "bundle_example"
    assert bundle.sequence_length == 4
    assert bundle.num_classes == 5
    assert bundle.task == "adjacent_ones_score"
    assert len(bundle.full_dataset) == 16


def test_mps_classifier_returns_scores_for_each_class() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=6,
            task="count_ones",
        )
    )
    batch = torch.stack(
        [encode_spike_train("01010"), encode_spike_train("11100")], dim=0
    )

    scores = model(batch)

    assert scores.shape == (2, 6)


def test_mps_classifier_builds_one_site_tensor_per_spike_position() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=6,
            task="count_ones",
        )
    )

    site_nodes = model.network.site_nodes

    assert len(site_nodes) == 5
    assert tuple(site_nodes[0].shape) == (2, 6)
    assert tuple(site_nodes[1].shape) == (6, 2, 6)
    assert tuple(site_nodes[-1].shape) == (6, 2, 6)
    assert site_nodes[0].axes_names == ["input", "right"]
    assert site_nodes[-1].axes_names == ["left", "input", "output"]
    assert site_nodes[-1]["output"].size() == 6


def test_single_site_mps_uses_only_input_and_output_axes() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=1,
            input_dim=2,
            num_classes=2,
            bond_dim=2,
            task="count_ones",
        )
    )
    batch = torch.stack([encode_spike_train("0")], dim=0)

    scores = model(batch)

    assert tuple(model.network.site_nodes[0].shape) == (2, 2)
    assert model.network.site_nodes[0].axes_names == ["input", "output"]
    assert scores.shape == (1, 2)


def test_prepare_for_training_stabilizes_active_mps_parameters() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=6,
            task="count_ones",
        )
    )
    batch = torch.stack(
        [encode_spike_train("01010"), encode_spike_train("11100")], dim=0
    )

    model.prepare_for_training()
    parameter_names_after_prepare = {name for name, _ in model.named_parameters()}
    _ = model(batch)
    parameter_names_after_forward = {name for name, _ in model.named_parameters()}

    assert "network.param_virtual_result_stack" in parameter_names_after_prepare
    assert "network.param_site_1" not in parameter_names_after_prepare
    assert "network.param_site_2" not in parameter_names_after_prepare
    assert "network.param_site_3" not in parameter_names_after_prepare
    assert parameter_names_after_forward == parameter_names_after_prepare


def test_prepare_for_training_allows_optimizer_to_track_active_parameters() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=6,
            task="count_ones",
        )
    )

    model.prepare_for_training()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }

    for _, parameter in model.named_parameters():
        assert id(parameter) in optimizer_parameter_ids


def test_select_predicted_class_uses_largest_absolute_score() -> None:
    scores = torch.tensor(
        [
            [-0.1, 0.2, -0.9],
            [0.5, -0.7, 0.6],
        ],
        dtype=torch.float32,
    )

    predicted = select_predicted_class(scores)

    assert torch.equal(predicted, torch.tensor([2, 1], dtype=torch.long))


def test_compute_loss_components_combines_cross_entropy_and_one_hot_penalty() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.2, 0.3, 0.8],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 2], dtype=torch.long)

    total_loss, cross_entropy_loss, one_hot_penalty = _compute_loss_components(
        scores=scores,
        labels=labels,
        num_classes=3,
        one_hot_penalty_weight=0.25,
    )

    abs_scores = scores.abs()
    expected_cross_entropy = nn.CrossEntropyLoss()(abs_scores, labels)
    expected_penalty = torch.mean(
        (
            abs_scores
            - torch.nn.functional.one_hot(labels, num_classes=3).to(torch.float32)
        )
        ** 2
    )

    assert torch.isclose(cross_entropy_loss, expected_cross_entropy)
    assert torch.isclose(one_hot_penalty, expected_penalty)
    assert torch.isclose(total_loss, expected_cross_entropy + 0.25 * expected_penalty)


def test_compute_output_metrics_summarizes_target_and_off_target_components() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.2, 0.3, 0.8],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 2], dtype=torch.long)

    metrics = _compute_output_metrics(scores=scores, labels=labels)

    assert metrics["target_component_mean"] == pytest.approx(0.85)
    assert metrics["off_target_component_mean"] == pytest.approx(0.2)
    assert metrics["best_incorrect_component_mean"] == pytest.approx(0.25)
    assert metrics["target_margin_mean"] == pytest.approx(0.6)


def test_better_checkpoint_prefers_higher_full_accuracy_over_lower_loss() -> None:
    assert _is_better_full_checkpoint(
        full_accuracy=0.9,
        full_loss=10.0,
        best_full_accuracy=0.8,
        best_full_loss=0.1,
    )


def test_better_checkpoint_uses_loss_as_tiebreaker_for_equal_accuracy() -> None:
    assert _is_better_full_checkpoint(
        full_accuracy=0.9,
        full_loss=0.4,
        best_full_accuracy=0.9,
        best_full_loss=0.5,
    )
    assert not _is_better_full_checkpoint(
        full_accuracy=0.9,
        full_loss=0.6,
        best_full_accuracy=0.9,
        best_full_loss=0.5,
    )


def test_reached_perfect_full_accuracy_detects_full_memorization() -> None:
    assert _reached_perfect_full_accuracy(1.0)
    assert not _reached_perfect_full_accuracy(0.999)


def test_checkpoint_roundtrip_reconstructs_model_with_same_predictions(
    workspace_dir: Path,
) -> None:
    torch.manual_seed(0)
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=6,
            task="count_ones",
        )
    )
    batch = torch.stack(
        [encode_spike_train("01010"), encode_spike_train("11100")], dim=0
    )
    expected_scores = model(batch).detach()

    checkpoint_path = workspace_dir / "checkpoint_best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        experiment_name="roundtrip",
        dataset_name="bundle_example",
        experiment_config={"epochs": 2, "bond_dim": 6},
        best_epoch=1,
        best_full_loss=0.5,
        metrics={"full_accuracy": 0.8},
        seed=123,
    )

    loaded_model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    actual_scores = loaded_model(batch).detach()

    assert checkpoint["experiment_name"] == "roundtrip"
    assert torch.allclose(expected_scores, actual_scores)


def test_canonicalize_model_preserves_outputs_and_reduces_bond_dim() -> None:
    model = _build_perfect_count_ones_length2_model()
    batch = torch.stack(
        [
            encode_spike_train("00"),
            encode_spike_train("01"),
            encode_spike_train("10"),
            encode_spike_train("11"),
        ],
        dim=0,
    )
    expected_scores = model(batch).detach()

    canonical_model = canonicalize_model(model=model, mode="svd")
    actual_scores = canonical_model(batch).detach()

    assert canonical_model.config.bond_dim == (2,)
    assert torch.allclose(expected_scores, actual_scores)


def test_visualize_checkpoint_uses_visible_theme_and_spectral_inspector(
    workspace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=4,
            input_dim=2,
            num_classes=5,
            bond_dim=5,
            task="count_ones",
        )
    )
    checkpoint_path = workspace_dir / "checkpoint_best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        experiment_name="viz_theme_example",
        dataset_name="viz_theme_example_dataset",
        experiment_config={"epochs": 1, "bond_dim": 5},
        best_epoch=1,
        best_full_loss=0.5,
        metrics={"full_accuracy": 0.8},
        seed=123,
    )

    visualized_calls: list[dict[str, object]] = []

    def _fake_show_tensor_network(
        network: object,
        *,
        engine: str,
        config: object,
        show: bool,
    ) -> tuple[object, object]:
        visualized_calls.append(
            {
                "network": network,
                "engine": engine,
                "config": config,
                "show": show,
            }
        )
        return object(), object()

    monkeypatch.setattr(
        "spike_mps.training.visualization.show_tensor_network",
        _fake_show_tensor_network,
    )

    visualize_checkpoint(checkpoint_path=checkpoint_path, show=False)

    assert len(visualized_calls) == 1
    assert visualized_calls[0]["network"] is not None
    assert visualized_calls[0]["engine"] == "tensorkrowch"
    assert visualized_calls[0]["show"] is False
    assert visualized_calls[0]["config"].theme == "paper"
    assert visualized_calls[0]["config"].contraction_tensor_inspector is True
    assert visualized_calls[0]["config"].tensor_inspector_config.theme == "spectral"


def test_visualize_checkpoint_per_sample_uses_dataset_records_and_contraction_scheme(
    workspace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = _create_dataset_directory(
        workspace_dir=workspace_dir,
        name="viz_samples_example",
        task="count_ones",
        sequence_length=4,
        dataset_size=16,
    )
    bundle = load_dataset_bundle(dataset_dir=dataset_dir)
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=bundle.sequence_length,
            input_dim=2,
            num_classes=bundle.num_classes,
            bond_dim=5,
            task=bundle.task,
        )
    )
    checkpoint_path = (
        workspace_dir
        / "output"
        / "processed_data"
        / "experiments"
        / "viz_samples_example_exp"
        / "checkpoint_best.pt"
    )
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        experiment_name="viz_samples_example_exp",
        dataset_name="viz_samples_example",
        experiment_config={"epochs": 1, "bond_dim": 5},
        best_epoch=1,
        best_full_loss=0.5,
        metrics={"full_accuracy": 0.8},
        seed=123,
    )

    visualized_calls: list[dict[str, object]] = []

    def _fake_show_tensor_network(
        network: object,
        *,
        engine: str,
        config: object,
        show: bool,
    ) -> tuple[object, object]:
        visualized_calls.append(
            {
                "network": network,
                "engine": engine,
                "config": config,
                "show": show,
            }
        )
        return object(), object()

    monkeypatch.setattr(
        "spike_mps.training.visualization.show_tensor_network",
        _fake_show_tensor_network,
    )

    summaries = visualize_checkpoint_per_sample(
        checkpoint_path=checkpoint_path,
        split="full",
        limit=2,
        show=False,
    )

    assert len(summaries) == 2
    assert len(visualized_calls) == 2
    assert all(call["engine"] == "tensorkrowch" for call in visualized_calls)
    assert all(call["show"] is False for call in visualized_calls)
    assert all(
        getattr(call["config"], "show_contraction_scheme", False) is True
        for call in visualized_calls
    )
    assert all(
        getattr(call["config"], "contraction_tensor_inspector", False) is True
        for call in visualized_calls
    )
    assert all(
        getattr(call["config"].tensor_inspector_config, "theme", None) == "spectral"
        for call in visualized_calls
    )
    assert [node.name for node in visualized_calls[0]["network"]] == [
        "site_0",
        "site_1",
        "site_2",
        "site_3",
        "data_0",
        "data_1",
        "data_2",
        "data_3",
    ]
    assert visualized_calls[0]["config"].contraction_scheme_by_name == (
        ("site_0", "data_0"),
        ("site_1", "data_1"),
        ("site_0", "data_0", "site_1", "data_1"),
        ("site_2", "data_2"),
        ("site_0", "data_0", "site_1", "data_1", "site_2", "data_2"),
        ("site_3", "data_3"),
        (
            "site_0",
            "data_0",
            "site_1",
            "data_1",
            "site_2",
            "data_2",
            "site_3",
            "data_3",
        ),
    )

    first_record = bundle.full_dataset.records[0]
    second_record = bundle.full_dataset.records[1]
    assert summaries[0].sample_index == 0
    assert summaries[0].spike_train == first_record.spike_train
    assert summaries[0].label == first_record.label
    assert summaries[1].sample_index == 1
    assert summaries[1].spike_train == second_record.spike_train
    assert len(summaries[0].scores) == bundle.num_classes


def _create_dataset_directory(
    *,
    workspace_dir: Path,
    name: str,
    task: str,
    sequence_length: int,
    dataset_size: int,
) -> Path:
    config = DatasetConfig(
        name=name,
        task=task,
        sequence_length=sequence_length,
        dataset_size=dataset_size,
        seed=9,
        split_ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
        label_noise_pct=0.0,
        noise_scope="train",
    )
    artifacts = build_dataset_artifacts(config=config)
    output_root = workspace_dir / "datasets" / "generated"
    return write_dataset_artifacts(artifacts=artifacts, output_root=output_root)


def _build_perfect_count_ones_length2_model() -> MPSClassifier:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=2,
            input_dim=2,
            num_classes=3,
            bond_dim=3,
            task="count_ones",
        )
    )
    model.network.site_nodes[0].tensor = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    site_1_tensor = torch.zeros((3, 2, 3), dtype=torch.float32)
    site_1_tensor[0, 0, 0] = 1.0
    site_1_tensor[0, 1, 1] = 1.0
    site_1_tensor[1, 0, 1] = 1.0
    site_1_tensor[1, 1, 2] = 1.0
    model.network.site_nodes[1].tensor = site_1_tensor
    return model
