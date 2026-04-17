from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_VENV_REEXEC_ENV_VAR = "SPIKE_MPS_PROJECT_VENV_REEXEC"


def ensure_project_venv_python(*, project_root: Path) -> None:
    if os.environ.get(PROJECT_VENV_REEXEC_ENV_VAR) == "1":
        return

    venv_python = _resolve_project_venv_python(project_root=project_root)
    if venv_python is None:
        return

    current_executable = Path(sys.executable).resolve()
    if current_executable == venv_python.resolve():
        return

    os.environ[PROJECT_VENV_REEXEC_ENV_VAR] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def _resolve_project_venv_python(*, project_root: Path) -> Path | None:
    for candidate in _candidate_project_venv_pythons(project_root=project_root):
        if candidate.is_file():
            return candidate
    return None


def _candidate_project_venv_pythons(*, project_root: Path) -> tuple[Path, ...]:
    return (
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "bin" / "python3",
    )
