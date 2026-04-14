from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from spike_mps.config import TaskName


def encode_spike_train(spike_train: str) -> torch.Tensor:
    encoding: list[list[float]] = []
    for bit in spike_train:
        if bit == "0":
            encoding.append([1.0, 0.0])
        elif bit == "1":
            encoding.append([0.0, 1.0])
        else:
            raise ValueError("spike_train must contain only '0' and '1'.")
    return torch.tensor(encoding, dtype=torch.float32)


@dataclass(frozen=True)
class DatasetRecord:
    spike_train: str
    label: int
    clean_label: int
    is_noisy: bool


class SpikeTrainTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[DatasetRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        return encode_spike_train(record.spike_train), torch.tensor(
            record.label, dtype=torch.long
        )


@dataclass(frozen=True)
class DatasetBundle:
    dataset_name: str
    task: TaskName
    sequence_length: int
    num_classes: int
    train_dataset: SpikeTrainTensorDataset
    val_dataset: SpikeTrainTensorDataset
    test_dataset: SpikeTrainTensorDataset
    metadata: dict[str, Any]


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    metadata = yaml.safe_load(
        (dataset_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(metadata, dict):
        raise ValueError("metadata.yaml must contain a YAML mapping.")

    full_distribution = metadata["label_distribution"]["full"]
    labels = [int(label) for label in full_distribution.keys()]
    num_classes = max(labels) + 1

    return DatasetBundle(
        dataset_name=str(metadata["name"]),
        task=str(metadata["task"]),
        sequence_length=int(metadata["sequence_length"]),
        num_classes=num_classes,
        train_dataset=SpikeTrainTensorDataset(
            _load_split_records(dataset_dir / "train.csv")
        ),
        val_dataset=SpikeTrainTensorDataset(
            _load_split_records(dataset_dir / "val.csv")
        ),
        test_dataset=SpikeTrainTensorDataset(
            _load_split_records(dataset_dir / "test.csv")
        ),
        metadata=metadata,
    )


def _load_split_records(split_path: Path) -> list[DatasetRecord]:
    with split_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return [
            DatasetRecord(
                spike_train=str(row["spike_train"]),
                label=int(row["label"]),
                clean_label=int(row["clean_label"]),
                is_noisy=str(row["is_noisy"]).lower() == "true",
            )
            for row in reader
        ]
