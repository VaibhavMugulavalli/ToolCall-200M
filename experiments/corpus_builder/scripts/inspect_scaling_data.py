#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode random samples from token shards")
    parser.add_argument("data_root")
    parser.add_argument("--split", default="validation_general")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(root / "tokenizer" / "toolcall_spm_32k.model")
    )
    split_root = root / args.split
    records = [
        json.loads(line)
        for line in (split_root / "manifest_shards.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    for index in range(args.samples):
        record = rng.choice(records)
        shard = np.memmap(split_root / record["file"], mode="r", dtype="<u2")
        if shard.size < args.tokens:
            start = 0
        else:
            start = rng.randrange(0, shard.size - args.tokens + 1)
        ids = shard[start : start + args.tokens].astype(np.int32).tolist()
        print(f"\n--- sample {index + 1}: {record['file']} @ token {start:,} ---")
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
