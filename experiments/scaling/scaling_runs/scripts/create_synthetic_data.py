#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_split(
    root: Path, token_count: int, vocab_size: int, seed: int, shard_tokens: int
) -> None:
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    shard_index = 0
    while written < token_count:
        count = min(shard_tokens, token_count - written)
        tokens = rng.integers(0, vocab_size, size=count, dtype=np.uint16)
        tokens.tofile(shards_dir / f"shard_{shard_index:05d}.bin")
        written += count
        shard_index += 1
    print(f"Wrote {written:,} tokens to {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create random shards for a systems test")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--train-tokens", type=int, default=2_000_000)
    parser.add_argument("--validation-tokens", type=int, default=500_000)
    parser.add_argument("--shard-tokens", type=int, default=500_000)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    write_split(
        output / "train",
        args.train_tokens,
        args.vocab_size,
        args.seed,
        args.shard_tokens,
    )
    write_split(
        output / "validation",
        args.validation_tokens,
        args.vocab_size,
        args.seed + 1,
        args.shard_tokens,
    )


if __name__ == "__main__":
    main()

