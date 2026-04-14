from __future__ import annotations

from pathlib import Path

import torch

from spike_mps.config import DatasetConfig, SplitRatios
from spike_mps.generation import build_dataset_artifacts
from spike_mps.models.mps_classifier import MPSClassifier, MPSModelConfig
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
        task="count_ones",
        sequence_length=4,
        dataset_size=16,
    )

    bundle = load_dataset_bundle(dataset_dir=dataset_dir)

    assert bundle.dataset_name == "bundle_example"
    assert bundle.sequence_length == 4
    assert bundle.num_classes == 5
    assert bundle.task == "count_ones"


def test_mps_classifier_returns_logits_for_each_class() -> None:
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=4,
            task="count_ones",
        )
    )
    batch = torch.stack(
        [encode_spike_train("01010"), encode_spike_train("11100")], dim=0
    )

    logits = model(batch)

    assert logits.shape == (2, 6)


def test_checkpoint_roundtrip_reconstructs_model_with_same_predictions(
    workspace_dir: Path,
) -> None:
    torch.manual_seed(0)
    model = MPSClassifier(
        config=MPSModelConfig(
            sequence_length=5,
            input_dim=2,
            num_classes=6,
            bond_dim=4,
            task="count_ones",
        )
    )
    batch = torch.stack(
        [encode_spike_train("01010"), encode_spike_train("11100")], dim=0
    )
    expected_logits = model(batch).detach()

    checkpoint_path = workspace_dir / "checkpoint_best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        experiment_name="roundtrip",
        dataset_name="bundle_example",
        experiment_config={"epochs": 2, "bond_dim": 4},
        best_epoch=1,
        best_val_loss=0.5,
        metrics={"val_accuracy": 0.8},
        seed=123,
    )

    loaded_model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    actual_logits = loaded_model(batch).detach()

    assert checkpoint["experiment_name"] == "roundtrip"
    assert torch.allclose(expected_logits, actual_logits)


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
