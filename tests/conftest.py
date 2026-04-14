from __future__ import annotations

import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def workspace_dir() -> Path:
    base_dir = ROOT / ".tmp" / "test-workspaces"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"workspace-{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
