#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "m30": PROJECT_ROOT / "configs/m30_standard_470m.json",
    "m60": PROJECT_ROOT / "configs/m60_swiglu_gqa_470m.json",
    "m30-debug": PROJECT_ROOT / "configs/m30_standard_debug_1m.json",
    "m60-debug": PROJECT_ROOT / "configs/m60_swiglu_gqa_debug_1m.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume one full-dataset extended scaling run"
    )
    parser.add_argument("--run", choices=sorted(CONFIGS), required=True)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit checkpoint path (default: auto)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument(
        "--session-token-budget",
        type=int,
        help="Pause cleanly after this many new prediction tokens in this session",
    )
    parser.add_argument(
        "--max-session-minutes",
        type=float,
        help="Pause cleanly after this many minutes and save a resume checkpoint",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(CONFIGS[args.run])
    runs_root = Path(args.runs_dir).expanduser().resolve()
    run_dir = runs_root / config.run_name
    existing_state = (
        (run_dir / "metrics.jsonl").exists()
        or (run_dir / "summary.json").exists()
        or any((run_dir / "checkpoints").glob("*.pt"))
    )
    if str(args.resume).lower() == "none" and existing_state:
        raise RuntimeError(
            f"Run {config.run_name!r} already contains state. Use --resume auto."
        )

    trainer = Trainer(
        config=config,
        project_root=PROJECT_ROOT,
        runs_root=runs_root,
        device_name=args.device,
    )
    print(json.dumps(trainer.describe(), indent=2))
    if args.preflight_only:
        print("Preflight passed. No training was started.")
        return

    resumed = trainer.resume(args.resume)
    if resumed is not None:
        print(f"Resumed from {resumed}")
    elif existing_state:
        raise RuntimeError(
            f"Run {config.run_name!r} contains state but no resumable checkpoint."
        )
    else:
        print("Starting a new run from random initialization")

    trainer.train(
        session_token_budget=args.session_token_budget,
        max_session_minutes=args.max_session_minutes,
    )


if __name__ == "__main__":
    main()

