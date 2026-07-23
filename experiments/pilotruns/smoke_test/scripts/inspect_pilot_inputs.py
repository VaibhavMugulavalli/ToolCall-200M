#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sample-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")

    shard_dir = args.data_dir / "shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard dir not found: {shard_dir}")

    shard_paths = sorted(shard_dir.glob("shard_*.bin"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.bin files found under {shard_dir}")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    vocab_size = sp.vocab_size()

    total_tokens = 0
    global_max = -1

    print("Tokenizer:", args.tokenizer)
    print("Vocab size:", vocab_size)
    print("Data dir:", args.data_dir)
    print("Shards:", len(shard_paths))

    for path in shard_paths:
        arr = np.fromfile(path, dtype=np.uint16)
        if arr.size == 0:
            raise RuntimeError(f"Empty shard: {path}")
        max_id = int(arr.max())
        if max_id >= vocab_size:
            raise RuntimeError(f"{path.name} has token id {max_id} >= vocab size {vocab_size}")

        total_tokens += int(arr.size)
        global_max = max(global_max, max_id)
        print(f"{path.name}: {arr.size:,} tokens, max_id={max_id}")

    print("\nTotal tokens:", f"{total_tokens:,}")
    print("Global max token id:", global_max)

    first = np.fromfile(shard_paths[0], dtype=np.uint16)
    ids = first[: args.sample_tokens].astype(np.int64).tolist()
    print("\nDecoded sample:")
    print("=" * 100)
    print(sp.decode(ids)[:2000])

    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        print("\nManifest found:", manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Manifest actual_tokens:", manifest.get("actual_tokens"))
        print("Manifest dtype:", manifest.get("dtype"))
    else:
        print("\nNo manifest.json found beside shards.")


if __name__ == "__main__":
    main()
