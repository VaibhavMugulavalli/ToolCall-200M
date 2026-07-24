#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.config import load_config
from scaling.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one scaling-law model run")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit checkpoint path (default: auto)",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, or cuda:1"
    )
    parser.add_argument(
        "--runs-dir",
        help="Runs directory for checkpoints, metrics, logs, and summaries",
    )
    parser.add_argument("--train-dir", help="Override the config training corpus")
    parser.add_argument("--validation-dir", help="Override the validation corpus")
    parser.add_argument(
        "--structured-validation-dir",
        help="Override or enable the structured validation corpus",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, data capacity, model, and config without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runs_root = (
        Path(args.runs_dir).expanduser().resolve()
        if args.runs_dir
        else PROJECT_ROOT / "runs"
    )
    existing_run_dir = runs_root / config.run_name
    existing_state = (
        (existing_run_dir / "metrics.jsonl").exists()
        or (existing_run_dir / "summary.json").exists()
        or any((existing_run_dir / "checkpoints").glob("*.pt"))
    )
    if (
        not args.preflight_only
        and str(args.resume).lower() == "none"
        and existing_state
    ):
        raise RuntimeError(
            f"Run {config.run_name!r} already contains state. Use --resume auto or "
            "choose a new run_name; refusing to mix two experiments."
        )
    trainer = Trainer(
        config=config,
        project_root=PROJECT_ROOT,
        runs_root=runs_root,
        device_name=args.device,
        train_dir_override=args.train_dir,
        validation_dir_override=args.validation_dir,
        structured_validation_dir_override=args.structured_validation_dir,
    )
    description = trainer.describe()
    print(json.dumps(description, indent=2))
    if args.preflight_only:
        print("Preflight passed. No training was started.")
        return
    resumed = trainer.resume(args.resume)
    if resumed:
        print(f"Resumed from {resumed}")
    else:
        if existing_state:
            raise RuntimeError(
                f"Run {config.run_name!r} contains metrics but no resumable checkpoint. "
                "Choose a new run_name to keep the experiment history clean."
            )
        print("Starting a new run")
    trainer.train()


if __name__ == "__main__":
    main()
