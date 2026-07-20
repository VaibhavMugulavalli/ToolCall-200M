#!/usr/bin/env python3
"""Validate and inspect packed uint16 token shards."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-tokens", type=int, default=256)
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    vocab_size = sp.vocab_size()
    paths = sorted((args.data_dir / "shards").glob("shard_*.bin"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.bin files found in {args.data_dir / 'shards'}")
    total_tokens, max_token, min_token = 0, 0, vocab_size
    print(f"Found {len(paths)} shards")
    for path in paths:
        arr = np.fromfile(path, dtype=np.uint16)
        if arr.size == 0:
            raise RuntimeError(f"Empty shard: {path}")
        total_tokens += int(arr.size)
        max_token = max(max_token, int(arr.max()))
        min_token = min(min_token, int(arr.min()))
        if int(arr.max()) >= vocab_size:
            raise RuntimeError(f"{path.name} contains token id {int(arr.max())} >= vocab size {vocab_size}")
        print(f"{path.name}: {arr.size:,} tokens, max_id={int(arr.max())}")
    print("\nSummary")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Min token ID: {min_token}")
    print(f"Max token ID: {max_token}")
    print(f"EOS ID: {sp.eos_id()}")
    for i in range(args.num_samples):
        path = random.choice(paths)
        arr = np.fromfile(path, dtype=np.uint16)
        start = 0 if arr.size <= args.sample_tokens else random.randint(0, arr.size - args.sample_tokens)
        ids = arr[start:start + args.sample_tokens].astype(np.int64).tolist()
        text = sp.decode(ids)
        print("\n" + "=" * 100)
        print(f"Sample {i}: {path.name}, start={start}")
        print("=" * 100)
        print(text[:2000])

if __name__ == "__main__":
    main()
