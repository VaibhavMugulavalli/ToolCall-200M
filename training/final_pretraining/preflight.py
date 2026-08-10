#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import TokenPlan, load_config
from .data import ShardedTokenCorpus, discover_one


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only final-run data preflight")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs/toolcall_200m_4b.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train = ShardedTokenCorpus(
        args.data_root, config.data.train_patterns, config.data.token_dtype
    )
    general = ShardedTokenCorpus(
        args.data_root,
        config.data.validation_general_patterns,
        config.data.token_dtype,
    )
    structured = ShardedTokenCorpus(
        args.data_root,
        config.data.validation_structured_patterns,
        config.data.token_dtype,
    )
    tokenizer = discover_one(
        args.data_root, config.data.tokenizer_patterns, "SentencePiece model"
    )
    actual_counts = {
        "train": train.total_tokens,
        "validation_general": general.total_tokens,
        "validation_structured": structured.total_tokens,
    }
    expected_counts = {
        "train": config.data.expected_train_source_tokens,
        "validation_general": config.data.expected_validation_general_tokens,
        "validation_structured": config.data.expected_validation_structured_tokens,
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Dataset counts do not match: actual={actual_counts}, expected={expected_counts}"
        )

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if processor.vocab_size() != config.model.vocab_size:
        raise RuntimeError("Tokenizer vocabulary does not match the model")
    plan = TokenPlan.from_corpus(config, train.total_tokens)
    report = {
        "status": "passed",
        "parameter_count": config.model.expected_parameter_count,
        "context_length": config.model.sequence_length,
        "tokenizer": str(tokenizer),
        "tokenizer_vocab_size": processor.vocab_size(),
        "splits": {
            "train": asdict(train.describe()),
            "validation_general": asdict(general.describe()),
            "validation_structured": asdict(structured.describe()),
        },
        "token_plan": asdict(plan),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
