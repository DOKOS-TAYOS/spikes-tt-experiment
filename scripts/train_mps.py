from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run() -> int:
    import argparse

    from spike_mps.training.runner import run_training_experiment

    parser = argparse.ArgumentParser(
        description="Train an MPS experiment from YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configTraining.yaml.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name to train.",
    )
    args = parser.parse_args()

    checkpoint_path = run_training_experiment(
        base_path=Path.cwd(),
        config_path=args.config,
        experiment_name=args.experiment,
    )
    print(f"Saved best checkpoint to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
