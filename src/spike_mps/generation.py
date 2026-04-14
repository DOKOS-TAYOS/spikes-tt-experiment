from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Literal

from spike_mps.config import DatasetConfig, TaskName
from spike_mps.labels import compute_label, max_label_for_task

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetRow:
    sample_id: int
    spike_train: str
    clean_label: int
    label: int
    is_noisy: bool
    split: SplitName


@dataclass(frozen=True)
class DatasetArtifacts:
    config: DatasetConfig
    full_rows: list[DatasetRow]
    splits: dict[SplitName, list[DatasetRow]]
    metadata: dict[str, object]


def build_dataset_artifacts(config: DatasetConfig) -> DatasetArtifacts:
    total_unique = config.unique_space_size
    if config.dataset_size > total_unique:
        raise ValueError(
            "Requested dataset_size cannot exceed the number of unique spike trains "
            f"for sequence_length={config.sequence_length}. Requested "
            f"{config.dataset_size}, "
            f"maximum {total_unique}."
        )

    selected_sequences = _select_sequences(config)
    split_assignments = _assign_splits(
        sequences=selected_sequences,
        split_ratios=config.split_ratios,
        seed=config.seed + 1,
    )
    clean_rows = _build_clean_rows(
        task=config.task,
        split_assignments=split_assignments,
    )
    noisy_rows = _apply_label_noise(rows=clean_rows, config=config)
    splits = {
        split_name: [row for row in noisy_rows if row.split == split_name]
        for split_name in ("train", "val", "test")
    }
    metadata = _build_metadata(config=config, rows=noisy_rows, splits=splits)
    return DatasetArtifacts(
        config=config,
        full_rows=noisy_rows,
        splits=splits,
        metadata=metadata,
    )


def _select_sequences(config: DatasetConfig) -> list[str]:
    all_sequences = [
        format(number, f"0{config.sequence_length}b")
        for number in range(config.unique_space_size)
    ]
    if config.dataset_size == config.unique_space_size:
        return all_sequences

    selection_rng = Random(config.seed)
    return selection_rng.sample(all_sequences, k=config.dataset_size)


def _assign_splits(
    sequences: list[str],
    split_ratios: object,
    seed: int,
) -> list[tuple[str, SplitName]]:
    train_ratio = split_ratios.train
    val_ratio = split_ratios.val

    shuffled_sequences = list(sequences)
    Random(seed).shuffle(shuffled_sequences)
    total = len(shuffled_sequences)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    split_names: list[SplitName] = (
        ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
    )
    return list(zip(shuffled_sequences, split_names, strict=True))


def _build_clean_rows(
    task: TaskName,
    split_assignments: list[tuple[str, SplitName]],
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    for sample_id, (spike_train, split_name) in enumerate(split_assignments):
        clean_label = compute_label(task=task, spike_train=spike_train)
        rows.append(
            DatasetRow(
                sample_id=sample_id,
                spike_train=spike_train,
                clean_label=clean_label,
                label=clean_label,
                is_noisy=False,
                split=split_name,
            )
        )
    return rows


def _apply_label_noise(
    rows: list[DatasetRow],
    config: DatasetConfig,
) -> list[DatasetRow]:
    if config.label_noise_pct == 0 or config.noise_scope == "none":
        return rows

    target_splits: set[SplitName] = {"train", "val", "test"}
    if config.noise_scope == "train":
        target_splits = {"train"}

    max_label = max_label_for_task(config.task, config.sequence_length)
    if max_label == 0:
        return rows

    eligible_indices = [
        index for index, row in enumerate(rows) if row.split in target_splits
    ]
    requested_noisy_count = round(
        len(eligible_indices) * config.label_noise_pct / 100.0
    )
    actual_noisy_count = min(requested_noisy_count, len(eligible_indices))
    noisy_indices = set(
        Random(config.seed + 2).sample(eligible_indices, k=actual_noisy_count)
    )
    noise_rng = Random(config.seed + 3)

    noisy_rows: list[DatasetRow] = []
    for index, row in enumerate(rows):
        if index not in noisy_indices:
            noisy_rows.append(row)
            continue

        noisy_label = _perturb_label_local(
            label=row.clean_label,
            max_label=max_label,
            noise_rng=noise_rng,
        )
        noisy_rows.append(
            DatasetRow(
                sample_id=row.sample_id,
                spike_train=row.spike_train,
                clean_label=row.clean_label,
                label=noisy_label,
                is_noisy=noisy_label != row.clean_label,
                split=row.split,
            )
        )

    return noisy_rows


def _perturb_label_local(
    label: int,
    max_label: int,
    noise_rng: Random,
) -> int:
    if label <= 0:
        return 1
    if label >= max_label:
        return max_label - 1
    return noise_rng.choice([label - 1, label + 1])


def _build_metadata(
    config: DatasetConfig,
    rows: list[DatasetRow],
    splits: dict[SplitName, list[DatasetRow]],
) -> dict[str, object]:
    return {
        "name": config.name,
        "task": config.task,
        "sequence_length": config.sequence_length,
        "requested_size": config.dataset_size,
        "generated_size": len(rows),
        "unique_space_size": config.unique_space_size,
        "seed": config.seed,
        "split_ratios": config.split_ratios.to_dict(),
        "split_counts": {
            split_name: len(split_rows) for split_name, split_rows in splits.items()
        },
        "label_noise_pct": config.label_noise_pct,
        "noise_scope": config.noise_scope,
        "noise_summary": {
            split_name: sum(1 for row in split_rows if row.is_noisy)
            for split_name, split_rows in splits.items()
        },
        "label_distribution": {
            "full": _count_labels(rows=rows, attribute_name="label"),
            **{
                split_name: _count_labels(rows=split_rows, attribute_name="label")
                for split_name, split_rows in splits.items()
            },
        },
        "clean_label_distribution": {
            "full": _count_labels(rows=rows, attribute_name="clean_label"),
            **{
                split_name: _count_labels(rows=split_rows, attribute_name="clean_label")
                for split_name, split_rows in splits.items()
            },
        },
    }


def _count_labels(rows: list[DatasetRow], attribute_name: str) -> dict[int, int]:
    counts = Counter(getattr(row, attribute_name) for row in rows)
    return dict(sorted(counts.items()))
