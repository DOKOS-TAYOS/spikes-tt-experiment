from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from spike_mps.config import load_app_config
from spike_mps.generation import build_dataset_artifacts
from spike_mps.writer import write_dataset_artifacts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible synthetic spike-train datasets from "
            "YAML configuration."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to configDatasets.yaml. If omitted, the generator searches "
            "for configDatasets.yaml and then configDatsets.yaml in the "
            "current directory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    working_directory = Path.cwd()
    app_config = load_app_config(
        base_path=working_directory,
        explicit_path=args.config,
    )
    output_root = working_directory / "datasets" / "generated"
    output_root.mkdir(parents=True, exist_ok=True)

    generated_directories: list[Path] = []
    for dataset_config in app_config.datasets:
        artifacts = build_dataset_artifacts(config=dataset_config)
        generated_directories.append(
            write_dataset_artifacts(artifacts=artifacts, output_root=output_root)
        )

    for dataset_dir in generated_directories:
        print(f"Generated dataset at {dataset_dir}")
    return 0
