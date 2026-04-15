from __future__ import annotations

from pathlib import Path

import torch

from spike_mps.config import DatasetConfig, SplitRatios
from spike_mps.generation import build_dataset_artifacts
from spike_mps.models.mps_classifier import (
    MPSClassifier,
    MPSModelConfig,
    select_predicted_class,
)
from spike_mps.training.checkpoints import (
    load_model_from_checkpoint,
    save_checkpoint,
)
from spike_mps.training.data import encode_spike_train, load_dataset_bundle
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
