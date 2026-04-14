from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run() -> int:
    import argparse

    from spike_mps.training.visualization import visualize_checkpoint

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
    args = parser.parse_args()

    visualize_checkpoint(checkpoint_path=args.checkpoint, show=not args.no_show)
    print(f"Loaded checkpoint {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
