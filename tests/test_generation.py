from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spike_mps.config import DatasetConfig, SplitRatios, load_app_config
from spike_mps.generation import build_dataset_artifacts


def make_config(
    *,
    name: str = "tiny",
    task: str = "count_ones",
    sequence_length: int = 3,
    dataset_size: int = 8,
    seed: int = 7,
    label_noise_pct: float = 0.0,
    noise_scope: str = "train",
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        task=task,
        sequence_length=sequence_length,
        dataset_size=dataset_size,
        seed=seed,
        split_ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
        label_noise_pct=label_noise_pct,
        noise_scope=noise_scope,
    )


def test_build_dataset_artifacts_creates_full_exhaustive_space() -> None:
    artifacts = build_dataset_artifacts(make_config(dataset_size=8))

    assert len(artifacts.full_rows) == 8
    assert {row.spike_train for row in artifacts.full_rows} == {
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    }


def test_build_dataset_artifacts_samples_without_replacement() -> None:
    artifacts = build_dataset_artifacts(make_config(dataset_size=4, seed=11))

    sequences = [row.spike_train for row in artifacts.full_rows]
    assert len(sequences) == 4
    assert len(set(sequences)) == 4


def test_build_dataset_artifacts_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        build_dataset_artifacts(make_config(dataset_size=9))


def test_build_dataset_artifacts_is_reproducible_for_same_seed() -> None:
    first = build_dataset_artifacts(make_config(dataset_size=4, seed=19))
    second = build_dataset_artifacts(make_config(dataset_size=4, seed=19))
    third = build_dataset_artifacts(make_config(dataset_size=4, seed=20))

    first_rows = [(row.spike_train, row.split) for row in first.full_rows]
    second_rows = [(row.spike_train, row.split) for row in second.full_rows]
    third_rows = [(row.spike_train, row.split) for row in third.full_rows]

    assert first_rows == second_rows
    assert first_rows != third_rows


def test_label_noise_is_applied_only_to_train_and_stays_local() -> None:
    artifacts = build_dataset_artifacts(
        make_config(
            task="count_ones",
            sequence_length=5,
            dataset_size=32,
            label_noise_pct=50.0,
            noise_scope="train",
            seed=3,
        )
    )

    train_rows = artifacts.splits["train"]
    val_rows = artifacts.splits["val"]
    test_rows = artifacts.splits["test"]
    max_label = 5

    assert any(row.is_noisy for row in train_rows)
    assert all(not row.is_noisy for row in val_rows + test_rows)

    for row in train_rows:
        assert 0 <= row.label <= max_label
        if row.is_noisy:
            assert abs(row.label - row.clean_label) == 1
        else:
            assert row.label == row.clean_label


def test_load_app_config_supports_legacy_config_name(workspace_dir: Path) -> None:
    config_path = workspace_dir / "configDatsets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
                    "label_noise_pct": 0.0,
                    "noise_scope": "train",
                },
                "datasets": [
                    {
                        "name": "legacy",
                        "task": "count_zeros",
                        "sequence_length": 4,
                        "dataset_size": 16,
                        "seed": 5,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app_config = load_app_config(base_path=workspace_dir)

    assert app_config.source_path == config_path
    assert app_config.datasets[0].task == "count_zeros"
