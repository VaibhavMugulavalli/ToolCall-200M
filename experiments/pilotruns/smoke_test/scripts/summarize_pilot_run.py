#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metadata_path = args.run_dir / "run_metadata.json"
    log_path = args.run_dir / "train_log.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError("No log rows found.")

    df = pd.DataFrame(rows)
    first = df.iloc[0]
    last = df.iloc[-1]
    loss_change = float(last["loss"]) - float(first["loss"])

    out_path = args.out or (args.run_dir / "result_summary.md")

    md = f"""# Pilot Run Summary

Run directory: `{args.run_dir}`

## Setup

- Data dir: `{metadata.get("data_dir")}`
- Tokenizer: `{metadata.get("tokenizer")}`
- Device: `{metadata.get("device")}`
- Parameters: `{metadata.get("parameters"):,}`
- Dataset tokens: `{metadata.get("dataset_tokens"):,}`
- Sequence length: `{metadata.get("seq_len")}`
- Batch size: `{metadata.get("batch_size")}`
- Gradient accumulation: `{metadata.get("grad_accum")}`
- Effective batch tokens: `{metadata.get("effective_batch_tokens"):,}`
- Max steps: `{metadata.get("max_steps")}`
- Learning rate: `{metadata.get("learning_rate")}`

## Result

- First logged step: `{int(first["step"])}`
- First loss: `{float(first["loss"]):.4f}`
- Last logged step: `{int(last["step"])}`
- Last loss: `{float(last["loss"]):.4f}`
- Loss change: `{loss_change:.4f}`
- Last tokens/sec: `{float(last["tokens_per_second"]):,.0f}`

## Interpretation

- Training loop completed: `TODO`
- Loss decreased meaningfully: `TODO`
- CUDA OOM: `TODO`
- Checkpoints saved: `TODO`
- Generation sample usable for sanity check: `TODO`

## Notes

TODO: Add observations from Colab logs.
"""
    out_path.write_text(md, encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
