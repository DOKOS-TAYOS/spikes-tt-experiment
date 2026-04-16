from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run() -> int:
    import argparse

    from spike_mps.training.canonicalization import canonicalize_checkpoint

    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize a trained MPS checkpoint and save it only if the model "
            "still reaches 100% accuracy on the full dataset."
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
        help="Optional output path for the canonical checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=("svd", "qr"),
        default="svd",
        help="Exact factorization used during canonicalization.",
    )
    args = parser.parse_args()

    result = canonicalize_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        mode=args.mode,
    )
    print(f"Accuracy before canonicalization: {result.accuracy_before:.4f}")
    print(f"Accuracy after canonicalization: {result.accuracy_after:.4f}")
    print(f"Bond dimensions before: {list(result.bond_dims_before)}")
    print(f"Bond dimensions after: {list(result.bond_dims_after)}")
    print(f"Saved canonical checkpoint to {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
