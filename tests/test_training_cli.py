from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from spike_mps.config import DatasetConfig, SplitRatios
from spike_mps.generation import build_dataset_artifacts
from spike_mps.writer import write_dataset_artifacts


def test_train_mps_cli_creates_expected_artifacts(workspace_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _create_dataset_directory(workspace_dir=workspace_dir, name="train_cli_example")
    config_path = _write_training_config(
        workspace_dir=workspace_dir,
        experiment_name="train_cli_example_exp",
        dataset_name="train_cli_example",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_mps.py"),
            "--config",
            str(config_path),
            "--experiment",
            "train_cli_example_exp",
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )

    assert completed.returncode == 0, completed.stderr

    experiment_dir = (
        workspace_dir
        / "output"
        / "processed_data"
        / "experiments"
        / "train_cli_example_exp"
    )
    assert (experiment_dir / "checkpoint_best.pt").exists()
    assert (experiment_dir / "history.csv").exists()
    assert (experiment_dir / "metrics.yaml").exists()
    assert (experiment_dir / "confusion_matrix.csv").exists()

    metrics = yaml.safe_load(
        (experiment_dir / "metrics.yaml").read_text(encoding="utf-8")
    )
    assert metrics["experiment_name"] == "train_cli_example_exp"
    assert metrics["dataset_name"] == "train_cli_example"


def test_visualize_mps_cli_reloads_checkpoint_and_prepares_visualization(
    workspace_dir: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _create_dataset_directory(workspace_dir=workspace_dir, name="viz_cli_example")
    config_path = _write_training_config(
        workspace_dir=workspace_dir,
        experiment_name="viz_cli_example_exp",
        dataset_name="viz_cli_example",
    )

    train_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_mps.py"),
            "--config",
            str(config_path),
            "--experiment",
            "viz_cli_example_exp",
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )
    assert train_completed.returncode == 0, train_completed.stderr

    checkpoint_path = (
        workspace_dir
        / "output"
        / "processed_data"
        / "experiments"
        / "viz_cli_example_exp"
        / "checkpoint_best.pt"
    )
    visualize_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "visualize_mps.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--no-show",
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )

    assert visualize_completed.returncode == 0, visualize_completed.stderr
    assert "Loaded checkpoint" in visualize_completed.stdout


def _create_dataset_directory(*, workspace_dir: Path, name: str) -> Path:
    config = DatasetConfig(
        name=name,
        task="count_ones",
        sequence_length=4,
        dataset_size=16,
        seed=9,
        split_ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
        label_noise_pct=0.0,
        noise_scope="train",
    )
    artifacts = build_dataset_artifacts(config=config)
    output_root = workspace_dir / "datasets" / "generated"
    return write_dataset_artifacts(artifacts=artifacts, output_root=output_root)


def _write_training_config(
    *,
    workspace_dir: Path,
    experiment_name: str,
    dataset_name: str,
) -> Path:
    config_path = workspace_dir / "configTraining.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "batch_size": 4,
                    "epochs": 2,
                    "learning_rate": 1e-2,
                    "weight_decay": 0.0,
                    "bond_dim": 3,
                    "patience": 2,
                    "device": "cpu",
                    "seed": 7,
                },
                "experiments": [
                    {
                        "name": experiment_name,
                        "dataset_name": dataset_name,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _pythonpath_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    return env
