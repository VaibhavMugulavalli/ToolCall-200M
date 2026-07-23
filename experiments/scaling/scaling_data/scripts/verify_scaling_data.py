#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sentencepiece as spm


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_signature(config: dict) -> list[dict]:
    return sorted(
        (
            {
                "name": source["name"],
                "category": source.get("category"),
                "kind": source.get("kind", "huggingface"),
                "weight": float(source["weight"]),
            }
            for source in config["sources"]
        ),
        key=lambda source: source["name"],
    )


def mixture_tolerance(config: dict, general_tokens: int) -> float:
    """Allow one document of scheduling overshoot on either side of a target."""
    document_granularity = 2 * int(config["max_document_tokens"]) / general_tokens
    return max(0.001, document_granularity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a generated scaling data bundle")
    parser.add_argument("data_root")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument(
        "--expected-config",
        default="configs/scaling_470m.json",
        help="Project config whose source mixture the bundle must match",
    )
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    expected_config_path = Path(args.expected_config).expanduser()
    if not expected_config_path.is_absolute():
        expected_config_path = project_root / expected_config_path
    expected_config = json.loads(expected_config_path.read_text(encoding="utf-8"))
    if (root / "BUILDING").exists() or not (root / "COMPLETE").exists():
        raise RuntimeError("Data bundle is incomplete")
    bundle = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((root / "build_config.json").read_text(encoding="utf-8"))
    if (
        source_signature(config) != source_signature(expected_config)
        or config.get("category_weights") != expected_config.get("category_weights")
    ):
        raise RuntimeError(
            "Bundle was generated with a different dataset mixture. "
            "Move or delete this bundle and rebuild it with the current project config."
        )
    print(f"PASS declared mixture matches {expected_config_path}")
    expected = {
        "train": int(config["train_tokens"]),
        "validation_general": int(config["validation_general_tokens"]),
        "validation_structured": int(config["validation_structured_tokens"]),
    }
    for split, expected_tokens in expected.items():
        split_root = root / split
        manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (split_root / "manifest_shards.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual_tokens = 0
        for record in records:
            shard = split_root / record["file"]
            if shard.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Byte-size mismatch: {shard}")
            actual_tokens += shard.stat().st_size // 2
            if args.checksums and digest(shard) != record["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {shard}")
        if actual_tokens != expected_tokens or manifest["tokens"] != expected_tokens:
            raise RuntimeError(
                f"{split}: expected {expected_tokens:,} tokens, found {actual_tokens:,}"
            )
        print(f"PASS {split}: {actual_tokens:,} tokens in {len(records)} shards")
    tokenizer_path = root / "tokenizer" / "toolcall_spm_32k.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != int(config["vocab_size"]):
        raise RuntimeError("Tokenizer vocabulary does not match the build config")
    if bundle.get("status") != "complete":
        raise RuntimeError("Bundle manifest is not marked complete")
    expected_weights = {
        source["name"]: float(source["weight"])
        for source in expected_config["sources"]
    }
    general_tokens = expected["train"] + expected["validation_general"]
    fraction_tolerance = mixture_tolerance(config, general_tokens)
    realized_sources = bundle.get("sources", [])
    realized_names = {source.get("name") for source in realized_sources}
    if realized_names != set(expected_weights):
        raise RuntimeError(
            "Bundle source manifest does not contain exactly the configured sources"
        )
    for source in realized_sources:
        expected_fraction = expected_weights[source["name"]]
        actual_fraction = float(source["realized_general_token_fraction"])
        if abs(actual_fraction - expected_fraction) > fraction_tolerance:
            raise RuntimeError(
                f"Mixture mismatch for {source['name']}: expected {expected_fraction:.2%}, "
                f"found {actual_fraction:.2%}, allowed deviation "
                f"{fraction_tolerance:.2%}"
            )
        print(
            f"PASS mixture {source['category']}: {actual_fraction:.2%} "
            f"({source['name']})"
        )
    print(f"PASS tokenizer: {tokenizer.get_piece_size():,} pieces, eos={tokenizer.eos_id()}")
    print(f"PASS mixture tolerance: {fraction_tolerance:.3%}")
    print("Data bundle verification passed.")


if __name__ == "__main__":
    main()
