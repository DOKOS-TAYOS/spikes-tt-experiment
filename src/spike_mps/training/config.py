from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

DeviceName = Literal["auto", "cpu", "cuda"]
_VALID_DEVICES: set[str] = {"auto", "cpu", "cuda"}
_DEFAULT_TRAINING_CONFIG = "configTraining.yaml"


@dataclass(frozen=True)
class TrainingExperimentConfig:
    name: str
    dataset_name: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    bond_dim: int
    one_hot_penalty_weight: float
    patience: int
    device: DeviceName
    seed: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Experiment name cannot be empty.")
        if not self.dataset_name:
            raise ValueError("dataset_name cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.bond_dim <= 0:
            raise ValueError("bond_dim must be greater than zero.")
        if self.one_hot_penalty_weight < 0:
            raise ValueError("one_hot_penalty_weight must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be greater than zero.")
        if self.device not in _VALID_DEVICES:
            raise ValueError(f"Unknown device: {self.device}")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "bond_dim": self.bond_dim,
            "one_hot_penalty_weight": self.one_hot_penalty_weight,
            "patience": self.patience,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TrainingAppConfig:
    experiments: list[TrainingExperimentConfig]
    source_path: Path


def resolve_training_config_path(
    base_path: Path,
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()

    config_path = base_path / _DEFAULT_TRAINING_CONFIG
    if config_path.exists():
        return config_path

    raise FileNotFoundError(
        f"Could not find '{_DEFAULT_TRAINING_CONFIG}' in {base_path}."
    )


def load_training_config(
    base_path: Path,
    explicit_path: Path | None = None,
) -> TrainingAppConfig:
    source_path = resolve_training_config_path(
        base_path=base_path, explicit_path=explicit_path
    )
    raw_data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("Training configuration file must contain a YAML mapping.")

    defaults = cast(dict[str, Any], raw_data.get("defaults", {}))
    raw_experiments = raw_data.get("experiments", [])
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError(
            "Training configuration must include a non-empty 'experiments' list."
        )

    experiments = [
        _build_experiment_config(
            defaults=defaults, raw_experiment=cast(dict[str, Any], raw_experiment)
        )
        for raw_experiment in raw_experiments
    ]
    return TrainingAppConfig(experiments=experiments, source_path=source_path)


def get_experiment_config(
    app_config: TrainingAppConfig,
    experiment_name: str,
) -> TrainingExperimentConfig:
    for experiment in app_config.experiments:
        if experiment.name == experiment_name:
            return experiment
    raise ValueError(f"Unknown experiment: {experiment_name}")


def _build_experiment_config(
    defaults: dict[str, Any],
    raw_experiment: dict[str, Any],
) -> TrainingExperimentConfig:
    return TrainingExperimentConfig(
        name=str(raw_experiment["name"]),
        dataset_name=str(raw_experiment["dataset_name"]),
        batch_size=int(
            raw_experiment.get("batch_size", defaults.get("batch_size", 32))
        ),
        epochs=int(raw_experiment.get("epochs", defaults.get("epochs", 100))),
        learning_rate=float(
            raw_experiment.get("learning_rate", defaults.get("learning_rate", 1e-3))
        ),
        weight_decay=float(
            raw_experiment.get("weight_decay", defaults.get("weight_decay", 0.0))
        ),
        bond_dim=int(raw_experiment.get("bond_dim", defaults.get("bond_dim", 8))),
        one_hot_penalty_weight=float(
            raw_experiment.get(
                "one_hot_penalty_weight",
                defaults.get("one_hot_penalty_weight", 0.25),
            )
        ),
        patience=int(raw_experiment.get("patience", defaults.get("patience", 15))),
        device=str(raw_experiment.get("device", defaults.get("device", "auto"))),
        seed=int(raw_experiment.get("seed", defaults.get("seed", 0))),
    )
