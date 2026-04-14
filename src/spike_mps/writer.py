from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import yaml

from spike_mps.generation import DatasetArtifacts, DatasetRow


def write_dataset_artifacts(
    artifacts: DatasetArtifacts,
    output_root: Path,
) -> Path:
    dataset_dir = output_root / artifacts.config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_rows_csv(path=dataset_dir / "full.csv", rows=artifacts.full_rows)
    for split_name in ("train", "val", "test"):
        _write_rows_csv(
            path=dataset_dir / f"{split_name}.csv",
            rows=artifacts.splits[split_name],
        )
    _write_metadata(path=dataset_dir / "metadata.yaml", metadata=artifacts.metadata)
    return dataset_dir


def _write_rows_csv(path: Path, rows: list[DatasetRow]) -> None:
    fieldnames = [
        "sample_id",
        "spike_train",
        "clean_label",
        "label",
        "is_noisy",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
