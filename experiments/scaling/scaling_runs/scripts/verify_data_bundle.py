#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def count_tokens(split_root: Path) -> tuple[int, int]:
    shards = sorted((split_root / "shards").glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"No .bin shards found under {split_root / 'shards'}")
    total = 0
    for shard in shards:
        if shard.stat().st_size % np.dtype(np.uint16).itemsize:
            raise RuntimeError(f"Invalid uint16 byte size: {shard}")
        total += shard.stat().st_size // 2
    return total, len(shards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the shared scaling_470m bundle")
    parser.add_argument("data_root", nargs="?", default="data/scaling_470m")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if (root / "BUILDING").exists() or not (root / "COMPLETE").is_file():
        raise RuntimeError(f"Incomplete data bundle: {root}")
    bundle = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    required = {
        "train": 470_000_000,
        "validation_general": 5_000_000,
        "validation_structured": 1_000_000,
    }
    for split, required_tokens in required.items():
        actual, shard_count = count_tokens(root / split)
        if actual < required_tokens:
            raise RuntimeError(
                f"{split} has {actual:,} tokens; at least {required_tokens:,} required"
            )
        manifest_path = root / split / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["tokens"]) != actual:
            raise RuntimeError(f"Manifest/token mismatch for {split}")
        print(f"PASS {split}: {actual:,} tokens in {shard_count} shards")
    tokenizer_path = root / "tokenizer" / "toolcall_spm_32k.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != 32_000:
        raise RuntimeError(
            f"Expected 32,000 tokenizer pieces, found {tokenizer.get_piece_size():,}"
        )
    print("PASS tokenizer: 32,000 pieces")
    expected_mix = {
        "clean_english_general": 0.45,
        "structured_text": 0.20,
        "api_docs_tool_schemas": 0.15,
        "code": 0.10,
        "task_action_instruction": 0.10,
    }
    actual_mix: dict[str, float] = {}
    for source in bundle.get("sources", []):
        category = source.get("category")
        actual_mix[category] = actual_mix.get(category, 0.0) + float(
            source["realized_general_token_fraction"]
        )
    for category, expected_fraction in expected_mix.items():
        actual_fraction = actual_mix.get(category, 0.0)
        if abs(actual_fraction - expected_fraction) > 0.001:
            raise RuntimeError(
                f"Mixture mismatch for {category}: expected {expected_fraction:.2%}, "
                f"found {actual_fraction:.2%}"
            )
        print(f"PASS mixture {category}: {actual_fraction:.2%}")
    print("Scaling data bundle is ready for all nine runs.")


if __name__ == "__main__":
    main()
