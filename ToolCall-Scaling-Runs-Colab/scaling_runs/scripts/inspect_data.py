#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.config import load_config, resolve_project_path
from scaling.data import PackedTokenCorpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect packed scaling corpora")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", help="Override the config training corpus")
    parser.add_argument("--validation-dir", help="Override the validation corpus")
    parser.add_argument(
        "--structured-validation-dir",
        help="Override or enable the structured validation corpus",
    )
    parser.add_argument(
        "--tokenizer",
        help="Optional native SentencePiece .model used to decode samples",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def inspect_split(
    name: str,
    corpus: PackedTokenCorpus,
    vocab_size: int,
    tokenizer,
    sample_count: int,
    sample_tokens: int,
    rng: random.Random,
) -> None:
    print(f"\n[{name}]")
    print(json.dumps(corpus.describe().__dict__, indent=2))
    for index in range(sample_count):
        offset = rng.randrange(0, corpus.total_tokens - sample_tokens)
        token_ids = corpus.read(offset, sample_tokens).astype("int64").tolist()
        maximum = max(token_ids)
        minimum = min(token_ids)
        if maximum >= vocab_size:
            raise RuntimeError(
                f"{name} contains token id {maximum}, outside vocab size {vocab_size}"
            )
        print(
            f"sample={index + 1} offset={offset:,} min_id={minimum} max_id={maximum}"
        )
        if tokenizer is not None:
            decoded = tokenizer.decode(token_ids).replace("\n", " ")
            print(decoded[:1000])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = None
    if args.tokenizer:
        import sentencepiece as spm

        tokenizer_path = resolve_project_path(PROJECT_ROOT, args.tokenizer)
        assert tokenizer_path is not None
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if tokenizer.vocab_size() != config.model.vocab_size:
            raise RuntimeError(
                f"Tokenizer vocab is {tokenizer.vocab_size()}, config expects "
                f"{config.model.vocab_size}"
            )

    rng = random.Random(args.seed)
    splits = {
        "train": args.train_dir or config.data.train_dir,
        "validation_general": args.validation_dir or config.data.validation_dir,
        "validation_structured": (
            args.structured_validation_dir or config.data.structured_validation_dir
        ),
    }
    for name, value in splits.items():
        path = resolve_project_path(PROJECT_ROOT, value)
        if path is None:
            continue
        inspect_split(
            name,
            PackedTokenCorpus(path, config.data.token_dtype),
            config.model.vocab_size,
            tokenizer,
            args.samples,
            args.sample_tokens,
            rng,
        )
    print("\nData inspection passed.")


if __name__ == "__main__":
    main()
