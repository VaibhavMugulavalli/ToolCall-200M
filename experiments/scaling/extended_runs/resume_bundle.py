#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
NUMBERED_CHECKPOINT = re.compile(r"checkpoint_step_(\d+)\.pt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(run_dir: Path) -> Path:
    final = run_dir / "checkpoints" / "checkpoint_final.pt"
    if final.is_file():
        return final
    numbered: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("checkpoint_step_*.pt"):
        match = NUMBERED_CHECKPOINT.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError(f"No resumable checkpoint found under {run_dir}")
    return sorted(numbered)[-1][1]


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def pack(run_name: str, runs_dir: Path, output: Path) -> None:
    run_dir = runs_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    checkpoint = latest_checkpoint(run_dir)
    checkpoint_hash = sha256(checkpoint)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "run_name": run_name,
        "checkpoint": f"{run_name}/checkpoints/{checkpoint.name}",
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "resume_bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        for name in ("config.json", "metrics.jsonl", "summary.json"):
            path = run_dir / name
            if path.is_file():
                archive.write(path, f"{run_name}/{name}")
        tensorboard = run_dir / "tensorboard"
        if tensorboard.is_dir():
            for path in sorted(tensorboard.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(run_dir).as_posix()
                    archive.write(path, f"{run_name}/{relative}")
        archive.write(
            checkpoint,
            f"{run_name}/checkpoints/{checkpoint.name}",
        )
    temporary.replace(output)
    print(
        f"Created {output} ({output.stat().st_size / 1024**2:.1f} MiB). "
        "The checkpoint is stored without ZIP compression for faster Colab export."
    )


def restore(archive_path: Path, runs_dir: Path, expected_run: str | None) -> str:
    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(f"Not a valid ZIP archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if any(not safe_member(name) for name in names):
            raise RuntimeError("Resume archive contains an unsafe path")
        try:
            manifest = json.loads(
                archive.read("resume_bundle_manifest.json").decode("utf-8")
            )
        except KeyError as error:
            raise RuntimeError("Resume bundle manifest is missing") from error
        if manifest.get("format_version") != 1:
            raise RuntimeError("Unsupported resume bundle format")
        run_name = str(manifest["run_name"])
        if expected_run is not None and run_name != expected_run:
            raise RuntimeError(
                f"Expected resume bundle for {expected_run!r}, found {run_name!r}"
            )
        prefix = f"{run_name}/"
        payload_names = [
            name
            for name in names
            if name != "resume_bundle_manifest.json" and not name.endswith("/")
        ]
        if not payload_names or any(not name.startswith(prefix) for name in payload_names):
            raise RuntimeError("Resume bundle contains files outside its run directory")

        runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs_dir.parent) as temporary:
            temporary_root = Path(temporary)
            archive.extractall(temporary_root)
            source = temporary_root / run_name
            checkpoint_relative = PurePosixPath(str(manifest["checkpoint"]))
            checkpoint = temporary_root.joinpath(*checkpoint_relative.parts)
            if not checkpoint.is_file():
                raise RuntimeError("Resume checkpoint is missing from the archive")
            if checkpoint.stat().st_size != int(manifest["checkpoint_bytes"]):
                raise RuntimeError("Resume checkpoint byte count does not match manifest")
            if sha256(checkpoint) != manifest["checkpoint_sha256"]:
                raise RuntimeError("Resume checkpoint checksum verification failed")

            destination = runs_dir / run_name
            existing_config = destination / "config.json"
            incoming_config = source / "config.json"
            if existing_config.is_file() and incoming_config.is_file():
                if existing_config.read_bytes() != incoming_config.read_bytes():
                    raise RuntimeError(
                        f"Refusing to merge a different config into {destination}"
                    )
            shutil.copytree(source, destination, dirs_exist_ok=True)

    print(f"Restored {run_name} under {runs_dir / run_name}")
    return run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or restore a minimal Colab resume bundle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--run-name", required=True)
    pack_parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    pack_parser.add_argument("--output")

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("archive")
    restore_parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    restore_parser.add_argument("--expected-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if args.command == "pack":
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else PROJECT_ROOT / f"{args.run_name}_resume.zip"
        )
        pack(args.run_name, runs_dir, output)
    else:
        restore(
            Path(args.archive).expanduser().resolve(),
            runs_dir,
            args.expected_run,
        )


if __name__ == "__main__":
    main()
