from __future__ import annotations

import shutil
from pathlib import Path


def clean_workspace(project_root: Path) -> None:
    removable_paths = [
        project_root / ".pytest_cache",
        project_root / ".ruff_cache",
        project_root / ".tmp",
    ]

    for cache_dir in removable_paths:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

    for stray_dir in project_root.glob("pytest-cache-files-*"):
        if stray_dir.is_dir():
            shutil.rmtree(stray_dir, ignore_errors=True)

    for pycache_dir in project_root.rglob("__pycache__"):
        if project_root in pycache_dir.parents:
            shutil.rmtree(pycache_dir, ignore_errors=True)

    for compiled_file in project_root.rglob("*.pyc"):
        if project_root in compiled_file.parents and compiled_file.exists():
            compiled_file.unlink()


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    clean_workspace(project_root=project_root)
    print("Removed temporary Python caches from the project workspace.")
    return 0
