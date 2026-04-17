from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spike_mps.training.visualization import SampleVisualizationSummary

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ensure_project_venv_python() -> None:
    from spike_mps.runtime import ensure_project_venv_python

    ensure_project_venv_python(project_root=ROOT)


_ensure_project_venv_python()


def _run() -> int:
    import argparse

    from spike_mps.training.visualization import (
        visualize_checkpoint,
        visualize_checkpoint_per_sample,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Reload a trained MPS checkpoint and open the tensor-network visualizer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_best.pt.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Prepare the visualization without opening an interactive window.",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help=(
            "Contract the checkpoint with each example from its dataset and "
            "show one contraction scheme per sample."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("full", "train", "val", "test"),
        default="full",
        help="Dataset split used with --per-sample. Defaults to full.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to visualize with --per-sample.",
    )
    args = parser.parse_args()

    if args.per_sample:
        summaries = visualize_checkpoint_per_sample(
            checkpoint_path=args.checkpoint,
            split=args.split,
            limit=args.limit,
            show=not args.no_show,
        )
        print(f"Loaded checkpoint {args.checkpoint}")
        for summary in summaries:
            print(_format_sample_summary(summary))
        print(
            "Visualized "
            f"{len(summaries)} sample contraction(s) from split {args.split}."
        )
        return 0

    visualize_checkpoint(checkpoint_path=args.checkpoint, show=not args.no_show)
    print(f"Loaded checkpoint {args.checkpoint}")
    return 0


def _format_sample_summary(summary: SampleVisualizationSummary) -> str:
    return (
        f"Sample {summary.sample_index} | "
        f"spike_train={summary.spike_train} | "
        f"label={summary.label} | "
        f"clean_label={summary.clean_label} | "
        f"is_noisy={summary.is_noisy} | "
        f"predicted={summary.predicted_label} | "
        f"scores={list(summary.scores)}"
    )


if __name__ == "__main__":
    raise SystemExit(_run())
