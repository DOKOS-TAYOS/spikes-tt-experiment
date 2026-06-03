from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from spike_mps.config import DatasetConfig, SplitRatios
from spike_mps.generation import build_dataset_artifacts
from spike_mps.models.mps_classifier import MPSClassifier, MPSModelConfig
from spike_mps.runtime import (
    PROJECT_VENV_REEXEC_ENV_VAR,
    ensure_project_venv_python,
)
from spike_mps.training.checkpoints import load_model_from_checkpoint, save_checkpoint
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
    assert metrics["output_concentration_penalty_weight"] == 0.5
    assert metrics["tensor_concentration_penalty_weight"] == 0.75
    assert metrics["post_training_concentration_enabled"] is False
    assert metrics["post_training_concentration_steps"] == 5
    assert metrics["post_training_concentration_learning_rate"] == 0.05
    assert metrics["post_training_concentration_restarts"] == 1
    assert metrics["post_training_concentration_output_path"] is None
    assert "best_full_loss" in metrics
    assert "full_loss" in metrics
    assert "full_cross_entropy_loss" in metrics
    assert "full_output_concentration_penalty" in metrics
    assert "full_tensor_concentration_penalty" in metrics
    assert "full_accuracy" in metrics
    assert "full_target_component_mean" in metrics
    assert "full_off_target_component_mean" in metrics
    assert "full_best_incorrect_component_mean" in metrics
    assert "full_target_margin_mean" in metrics
    assert "full_correct_predictions" in metrics
    assert "full_total_examples" in metrics
    assert "best_val_loss" not in metrics
    assert "test_loss" not in metrics
    assert "test_accuracy" not in metrics
    assert "Epoch 1" in completed.stdout
    assert "Final full metrics" in completed.stdout

    with (experiment_dir / "history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file_handle:
        history_rows = list(csv.DictReader(file_handle))
    assert history_rows
    assert "train_cross_entropy_loss" in history_rows[0]
    assert "train_output_concentration_penalty" in history_rows[0]
    assert "train_tensor_concentration_penalty" in history_rows[0]
    assert "train_target_component_mean" in history_rows[0]
    assert "train_off_target_component_mean" in history_rows[0]
    assert "train_best_incorrect_component_mean" in history_rows[0]
    assert "train_target_margin_mean" in history_rows[0]
    assert "full_loss" in history_rows[0]
    assert "full_cross_entropy_loss" in history_rows[0]
    assert "full_output_concentration_penalty" in history_rows[0]
    assert "full_tensor_concentration_penalty" in history_rows[0]
    assert "full_accuracy" in history_rows[0]
    assert "full_target_component_mean" in history_rows[0]
    assert "full_off_target_component_mean" in history_rows[0]
    assert "full_best_incorrect_component_mean" in history_rows[0]
    assert "full_target_margin_mean" in history_rows[0]
    assert "val_loss" not in history_rows[0]
    assert "val_accuracy" not in history_rows[0]

    with (
        workspace_dir / "datasets" / "generated" / "train_cli_example" / "full.csv"
    ).open("r", encoding="utf-8", newline="") as file_handle:
        dataset_rows = list(csv.DictReader(file_handle))
    with (experiment_dir / "confusion_matrix.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file_handle:
        confusion_rows = list(csv.reader(file_handle))

    predicted_total = sum(
        sum(int(value) for value in row[1:]) for row in confusion_rows[1:]
    )
    assert predicted_total == len(dataset_rows)


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


def test_visualize_mps_cli_can_iterate_sample_contractions(
    workspace_dir: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _create_dataset_directory(
        workspace_dir=workspace_dir,
        name="viz_per_sample_example",
    )
    config_path = _write_training_config(
        workspace_dir=workspace_dir,
        experiment_name="viz_per_sample_example_exp",
        dataset_name="viz_per_sample_example",
    )

    train_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_mps.py"),
            "--config",
            str(config_path),
            "--experiment",
            "viz_per_sample_example_exp",
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
        / "viz_per_sample_example_exp"
        / "checkpoint_best.pt"
    )
    visualize_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "visualize_mps.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--per-sample",
            "--limit",
            "2",
            "--no-show",
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )

    assert visualize_completed.returncode == 0, visualize_completed.stderr
    assert "Sample 0 | spike_train=" in visualize_completed.stdout
    assert "Sample 1 | spike_train=" in visualize_completed.stdout
    assert "Visualized 2 sample contraction(s) from split full." in (
        visualize_completed.stdout
    )


def test_canonicalize_mps_cli_saves_exact_canonical_checkpoint(
    workspace_dir: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = _create_perfect_count_ones_checkpoint(workspace_dir=workspace_dir)

    canonicalize_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "canonicalize_mps.py"),
            "--checkpoint",
            str(checkpoint_path),
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )

    assert canonicalize_completed.returncode == 0, canonicalize_completed.stderr
    assert "Accuracy before canonicalization: 1.0000" in canonicalize_completed.stdout
    assert "Accuracy after canonicalization: 1.0000" in canonicalize_completed.stdout
    canonical_checkpoint_path = checkpoint_path.parent / "checkpoint_canonical.pt"
    assert _artifact_exists(canonical_checkpoint_path)
    canonical_model, _ = load_model_from_checkpoint(canonical_checkpoint_path)
    assert canonical_model.config.parameterization == "direct"


def test_concentrate_mps_cli_saves_direct_concentrated_checkpoint(
    workspace_dir: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = _create_perfect_count_ones_checkpoint(workspace_dir=workspace_dir)

    concentrate_completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "concentrate_mps.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--steps",
            "20",
            "--learning-rate",
            "0.05",
            "--restarts",
            "1",
            "--seed",
            "0",
        ],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=_pythonpath_env(repo_root),
    )

    assert concentrate_completed.returncode == 0, concentrate_completed.stderr
    assert "Accuracy before concentration: 1.0000" in concentrate_completed.stdout
    assert "Accuracy after concentration: 1.0000" in concentrate_completed.stdout
    concentrated_checkpoint_path = checkpoint_path.parent / "checkpoint_concentrated.pt"
    assert _artifact_exists(concentrated_checkpoint_path)
    concentrated_model, _ = load_model_from_checkpoint(concentrated_checkpoint_path)
    assert concentrated_model.config.parameterization == "direct"


def test_ensure_project_venv_python_reexecs_when_running_outside_venv(
    monkeypatch: pytest.MonkeyPatch,
    workspace_dir: Path,
) -> None:
    project_root = workspace_dir / "project"
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    exec_calls: list[tuple[str, list[str]]] = []

    def _fake_execv(executable: str, argv: list[str]) -> None:
        exec_calls.append((executable, argv))
        raise RuntimeError("reexec requested")

    monkeypatch.delenv(PROJECT_VENV_REEXEC_ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "executable", str(project_root / "python.exe"))
    monkeypatch.setattr(sys, "argv", ["scripts/visualize_mps.py", "--no-show"])
    monkeypatch.setattr("os.execv", _fake_execv)

    with pytest.raises(RuntimeError, match="reexec requested"):
        ensure_project_venv_python(project_root=project_root)

    assert exec_calls == [
        (
            str(venv_python),
            [str(venv_python), "scripts/visualize_mps.py", "--no-show"],
        )
    ]
    assert os.environ[PROJECT_VENV_REEXEC_ENV_VAR] == "1"


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


def _create_perfect_count_ones_checkpoint(*, workspace_dir: Path) -> Path:
    config = DatasetConfig(
        name="canonical_cli_example",
        task="count_ones",
        sequence_length=2,
        dataset_size=4,
        seed=9,
        split_ratios=SplitRatios(train=0.5, val=0.25, test=0.25),
        label_noise_pct=0.0,
        noise_scope="train",
    )
    artifacts = build_dataset_artifacts(config=config)
    output_root = workspace_dir / "datasets" / "generated"
    write_dataset_artifacts(artifacts=artifacts, output_root=output_root)

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
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    site_1_tensor = torch.zeros((3, 2, 3), dtype=torch.float32)
    site_1_tensor[0, 0, 0] = 1.0
    site_1_tensor[0, 1, 1] = -1.0
    site_1_tensor[1, 0, 1] = 1.0
    site_1_tensor[1, 1, 2] = -1.0
    model.network.site_nodes[1].tensor = site_1_tensor

    checkpoint_path = (
        workspace_dir
        / "output"
        / "processed_data"
        / "experiments"
        / "canonical_cli_example_exp"
        / "checkpoint_best.pt"
    )
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        experiment_name="canonical_cli_example_exp",
        dataset_name="canonical_cli_example",
        experiment_config={"epochs": 1, "bond_dim": 3},
        best_epoch=1,
        best_full_loss=0.0,
        metrics={"full_accuracy": 1.0},
        seed=0,
    )
    return checkpoint_path


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
                    "bond_dim": 5,
                    "output_concentration_penalty_weight": 0.5,
                    "tensor_concentration_penalty_weight": 0.75,
                    "post_training_concentration_enabled": False,
                    "post_training_concentration_steps": 5,
                    "post_training_concentration_learning_rate": 0.05,
                    "post_training_concentration_restarts": 1,
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


def _artifact_exists(path: Path) -> bool:
    if os.name != "nt":
        return path.exists()
    path_text = str(path.resolve())
    if path_text.startswith("\\\\?\\"):
        return os.path.exists(path_text)
    return os.path.exists("\\\\?\\" + path_text)
