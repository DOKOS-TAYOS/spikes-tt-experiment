from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_generate_datasets_cli_creates_expected_files(workspace_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_datasets.py"

    config_path = workspace_dir / "configDatasets.yaml"
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
                        "name": "cli_example",
                        "task": "adjacent_ones_score",
                        "sequence_length": 4,
                        "dataset_size": 16,
                        "seed": 13,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        check=False,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr

    dataset_dir = workspace_dir / "datasets" / "generated" / "cli_example"
    assert (dataset_dir / "full.csv").exists()
    assert (dataset_dir / "train.csv").exists()
    assert (dataset_dir / "val.csv").exists()
    assert (dataset_dir / "test.csv").exists()
    assert (dataset_dir / "metadata.yaml").exists()

    metadata = yaml.safe_load(
        (dataset_dir / "metadata.yaml").read_text(encoding="utf-8")
    )
    assert metadata["name"] == "cli_example"
    assert metadata["task"] == "adjacent_ones_score"
    assert metadata["generated_size"] == 16
