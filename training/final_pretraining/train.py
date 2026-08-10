#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .trainer import DistributedTrainer


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/toolcall_200m_4b.json"
DEFAULT_DEEPSPEED_CONFIG = PACKAGE_ROOT / "deepspeed_zero2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume the frozen ToolCall-200M 4B-token run"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED_CONFIG
    )
    parser.add_argument(
        "--hub-repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face model repo (defaults to HF_REPO_ID)",
    )
    parser.add_argument(
        "--resume",
        choices=("auto", "none", "required"),
        default="auto",
        help="auto starts fresh only when the Hub has no checkpoint",
    )
    parser.add_argument(
        "--qualification-steps",
        type=int,
        help="Run a disposable ZeRO-2 qualification and do not access the Hub/W&B",
    )
    parser.add_argument(
        "--allow-no-wandb",
        action="store_true",
        help="Allow real training without live W&B graphs (not recommended)",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    qualification = args.qualification_steps is not None
    trainer = DistributedTrainer(
        config=config,
        config_path=args.config,
        deepspeed_config_path=args.deepspeed_config,
        data_root=args.data_root,
        run_directory=args.run_dir,
        hub_repo_id=args.hub_repo_id,
        hub_token=os.environ.get("HF_TOKEN"),
        resume_mode="none" if qualification else args.resume,
        allow_no_wandb=args.allow_no_wandb,
        qualification=qualification,
    )
    if trainer.is_main:
        print(json.dumps(trainer.describe(), indent=2))
    trainer.train(qualification_steps=args.qualification_steps)


if __name__ == "__main__":
    main()
