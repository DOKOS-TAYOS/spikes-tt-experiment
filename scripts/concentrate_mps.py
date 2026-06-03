from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run() -> int:
    import argparse

    from spike_mps.training.concentration import concentrate_checkpoint

    parser = argparse.ArgumentParser(
        description=(
            "Concentrate a trained MPS checkpoint with orthogonal bond "
            "transformations and save it only if 100% accuracy is preserved."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint_best.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the concentrated checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Optimization steps per MPS bond.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Adam learning rate used to optimize each orthogonal U matrix.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Random restarts per MPS bond.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for orthogonal matrix optimization.",
    )
    parser.add_argument(
        "--max-score-difference",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute score difference after concentration.",
    )
    args = parser.parse_args()

    result = concentrate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        steps=args.steps,
        learning_rate=args.learning_rate,
        restarts=args.restarts,
        seed=args.seed,
        max_score_difference_tolerance=args.max_score_difference,
    )
    print(f"Accuracy before concentration: {result.accuracy_before:.4f}")
    print(f"Accuracy after concentration: {result.accuracy_after:.4f}")
    print(f"Concentration before: {result.concentration_before:.6f}")
    print(f"Concentration after: {result.concentration_after:.6f}")
    print(f"Max score difference: {result.max_score_difference:.6g}")
    print(f"Saved concentrated checkpoint to {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
