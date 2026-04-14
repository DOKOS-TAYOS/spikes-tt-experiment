from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

TaskName = Literal["count_ones", "count_zeros", "adjacent_ones_score"]
NoiseScope = Literal["train", "all", "none"]

_VALID_TASKS: set[str] = {"count_ones", "count_zeros", "adjacent_ones_score"}
_VALID_NOISE_SCOPES: set[str] = {"train", "all", "none"}
_DEFAULT_CONFIG_NAME = "configDatasets.yaml"
_LEGACY_CONFIG_NAME = "configDatsets.yaml"


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("Split ratios must be non-negative.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0.")

    def to_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SplitRatios:
        return cls(
            train=float(data["train"]),
            val=float(data["val"]),
            test=float(data["test"]),
        )


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: TaskName
    sequence_length: int
    dataset_size: int
    seed: int
    split_ratios: SplitRatios
    label_noise_pct: float
    noise_scope: NoiseScope

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset name cannot be empty.")
        if self.task not in _VALID_TASKS:
            raise ValueError(f"Unknown task: {self.task}")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be greater than zero.")
        if self.label_noise_pct < 0 or self.label_noise_pct > 100:
            raise ValueError("label_noise_pct must be between 0 and 100.")
        if self.noise_scope not in _VALID_NOISE_SCOPES:
            raise ValueError(f"Unknown noise_scope: {self.noise_scope}")

    @property
    def unique_space_size(self) -> int:
        return 2**self.sequence_length


@dataclass(frozen=True)
class AppConfig:
    datasets: list[DatasetConfig]
    source_path: Path


def resolve_config_path(base_path: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()

    canonical = base_path / _DEFAULT_CONFIG_NAME
    if canonical.exists():
        return canonical

    legacy = base_path / _LEGACY_CONFIG_NAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"Could not find '{_DEFAULT_CONFIG_NAME}' or '{_LEGACY_CONFIG_NAME}' "
        f"in {base_path}."
    )


def load_app_config(
    base_path: Path,
    explicit_path: Path | None = None,
) -> AppConfig:
    source_path = resolve_config_path(base_path=base_path, explicit_path=explicit_path)
    raw_data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    defaults = _parse_defaults(cast(dict[str, Any], raw_data.get("defaults", {})))
    raw_datasets = raw_data.get("datasets", [])
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("Configuration must include a non-empty 'datasets' list.")

    datasets = [
        _build_dataset_config(
            defaults=defaults,
            raw_dataset=cast(dict[str, Any], raw_dataset),
        )
        for raw_dataset in raw_datasets
    ]
    return AppConfig(datasets=datasets, source_path=source_path)


def _parse_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    parsed_defaults = {
        "split_ratios": SplitRatios.from_mapping(
            cast(
                dict[str, Any],
                defaults.get(
                    "split_ratios",
                    {"train": 0.7, "val": 0.15, "test": 0.15},
                ),
            )
        ),
        "label_noise_pct": float(defaults.get("label_noise_pct", 0.0)),
        "noise_scope": str(defaults.get("noise_scope", "train")),
    }

    noise_scope = parsed_defaults["noise_scope"]
    if noise_scope not in _VALID_NOISE_SCOPES:
        raise ValueError(f"Unknown noise_scope: {noise_scope}")

    return parsed_defaults


def _build_dataset_config(
    defaults: dict[str, Any],
    raw_dataset: dict[str, Any],
) -> DatasetConfig:
    split_ratios = defaults["split_ratios"]
    if "split_ratios" in raw_dataset:
        split_ratios = SplitRatios.from_mapping(
            cast(dict[str, Any], raw_dataset["split_ratios"])
        )

    task = str(raw_dataset["task"])
    noise_scope = str(raw_dataset.get("noise_scope", defaults["noise_scope"]))

    return DatasetConfig(
        name=str(raw_dataset["name"]),
        task=cast(TaskName, task),
        sequence_length=int(raw_dataset["sequence_length"]),
        dataset_size=int(raw_dataset["dataset_size"]),
        seed=int(raw_dataset["seed"]),
        split_ratios=split_ratios,
        label_noise_pct=float(
            raw_dataset.get("label_noise_pct", defaults["label_noise_pct"])
        ),
        noise_scope=cast(NoiseScope, noise_scope),
    )
