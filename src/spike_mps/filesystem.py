from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, TextIO


def ensure_parent_dir(path: Path) -> None:
    os.makedirs(_platform_path(path.parent), exist_ok=True)


def open_binary_for_read(path: Path) -> BinaryIO:
    return open(_platform_path(path), "rb")


def open_binary_for_write(path: Path) -> BinaryIO:
    ensure_parent_dir(path)
    return open(_platform_path(path), "wb")


def open_text_for_write(path: Path, *, newline: str | None = None) -> TextIO:
    ensure_parent_dir(path)
    return open(_platform_path(path), "w", encoding="utf-8", newline=newline)


def _platform_path(path: Path) -> str:
    if os.name != "nt":
        return os.fspath(path)

    path_text = str(path.resolve(strict=False))
    if path_text.startswith("\\\\?\\"):
        return path_text
    if path_text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_text[2:]
    return "\\\\?\\" + path_text
