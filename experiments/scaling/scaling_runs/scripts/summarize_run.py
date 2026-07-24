#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.dashboard_data import build_run_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a concise run summary")
    parser.add_argument("run_name")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    args = parser.parse_args()
    run_dir = Path(args.runs_dir).expanduser().resolve() / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    payload = build_run_payload(run_dir)
    train = [record for record in payload["metrics"] if record.get("type") == "train"]
    validation = [
        record for record in payload["metrics"] if record.get("type") == "validation"
    ]
    summary = {
        **payload["summary"],
        "first_logged_train_loss": train[0]["loss"] if train else None,
        "last_logged_train_loss": train[-1]["loss"] if train else None,
        "last_validation_loss": validation[-1]["loss"] if validation else None,
        "logged_train_points": len(train),
        "logged_validation_points": len(validation),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
