#!/usr/bin/env python3
"""Run the complete DGM and TF3 data pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINE = (
    ("DGM download and processing", "download_newest.py"),
    ("river mask download", "river_downloader.py"),
    ("biome and terrain mask download", "biome_downloader.py"),
)


def run_step(label: str, script: Path, config: Path) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    subprocess.run(
        [sys.executable, str(script), str(config)],
        cwd=script.parent,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete DGM and TF3 download pipeline."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="User configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--skip-rivers",
        action="store_true",
        help="Do not generate the OSM river mask.",
    )
    parser.add_argument(
        "--skip-biomes",
        action="store_true",
        help="Do not generate the TF3 biome and terrain masks.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = Path.cwd() / config
    config = config.resolve()

    if not config.is_file():
        parser.error(f"Configuration file not found: {config}")

    steps = [PIPELINE[0]]
    if not args.skip_rivers:
        steps.append(PIPELINE[1])
    if not args.skip_biomes:
        steps.append(PIPELINE[2])

    try:
        for label, script_name in steps:
            script = project_dir / script_name
            if not script.is_file():
                raise FileNotFoundError(f"Pipeline script not found: {script}")
            run_step(label, script, config)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(
                f"\nPipeline stopped: {error.cmd[1]} exited with code "
                f"{error.returncode}.",
                file=sys.stderr,
            )
            return error.returncode or 1
        print(f"\nPipeline stopped: {error}", file=sys.stderr)
        return 1

    print("\nPipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
