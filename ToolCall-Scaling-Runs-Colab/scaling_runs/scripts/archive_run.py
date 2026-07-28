#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Archive one run for download")
    parser.add_argument("run_name")
    parser.add_argument("--runs-dir", help="Runs directory (defaults to scaling_runs/runs)")
    parser.add_argument("--output", help="Output .zip path")
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include large .pt checkpoint files",
    )
    args = parser.parse_args()

    runs_dir = (
        Path(args.runs_dir).expanduser().resolve()
        if args.runs_dir
        else project_root / "runs"
    )
    run_dir = runs_dir / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (project_root / f"{args.run_name}.zip")
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            if not args.include_checkpoints and "checkpoints" in path.parts:
                continue
            archive.write(path, Path("runs") / args.run_name / path.relative_to(run_dir))
    print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
