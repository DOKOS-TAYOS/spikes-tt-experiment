from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run() -> int:
    from spike_mps.tools import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
