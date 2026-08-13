#!/usr/bin/env python3
"""Measure unique token capacity of finite sources before the 4B build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import sentencepiece as spm

from build_scaling_data import (
    OpenAPICatalog,
    atomic_json,
    normalize_text,
    sha256_file,
    utc_now,
    validate_config,
)


WORD_PATTERN = re.compile(r"[A-Za-z0-9_./:{}-]+")


def simhash64(text: str, maximum_words: int = 4_000) -> int:
    words = WORD_PATTERN.findall(text.lower())[:maximum_words]
    if not words:
        return 0
    width = 5 if len(words) >= 5 else 1
    scores = [0] * 64
    for start in range(max(1, len(words) - width + 1)):
        shingle = "\x1f".join(words[start : start + width]).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for bit in range(64):
            scores[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            result |= 1 << bit
    return result


class NearDuplicateSample:
    """Small deterministic LSH sample; it reports but does not reject documents."""

    def __init__(self, sample_rate: float, maximum: int, threshold: int = 3) -> None:
        self.sample_rate = sample_rate
        self.maximum = maximum
        self.threshold = threshold
        self.fingerprints: list[int] = []
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.near_duplicates = 0

    def consider(self, digest: bytes, text: str) -> None:
        if len(self.fingerprints) >= self.maximum:
            return
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        if bucket >= self.sample_rate:
            return
        fingerprint = simhash64(text)
        candidates: set[int] = set()
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            candidates.update(self.buckets[(band, value)][-64:])
        if any((fingerprint ^ self.fingerprints[index]).bit_count() <= self.threshold for index in candidates):
            self.near_duplicates += 1
        index = len(self.fingerprints)
        self.fingerprints.append(fingerprint)
        for band in range(4):
            value = (fingerprint >> (band * 16)) & 0xFFFF
            self.buckets[(band, value)].append(index)


def source_iterator(catalog: OpenAPICatalog, kind: str) -> Iterator[dict[str, str]]:
    if kind == "openapi_raw":
        return catalog.raw_documents()
    if kind == "openapi_docs":
        return catalog.documentation()
    if kind == "openapi_actions":
        return catalog.action_documents()
    raise ValueError(f"Capacity audit only supports finite OpenAPI sources, got {kind!r}")


def audit_source(
    definition: dict[str, Any],
    catalog: OpenAPICatalog,
    tokenizer: spm.SentencePieceProcessor,
    config: dict[str, Any],
    tokenizer_path: Path,
    config_path: Path,
    near_sample_rate: float,
    near_sample_maximum: int,
    progress_documents: int,
) -> dict[str, Any]:
    started = time.time()
    seen: set[bytes] = set()
    near = NearDuplicateSample(near_sample_rate, near_sample_maximum)
    emitted = 0
    accepted = 0
    rejected_short = 0
    rejected_exact = 0
    truncated = 0
    raw_characters = 0
    encoded_before_truncation = 0
    retained_tokens = 0
    max_document_tokens = int(config["max_document_tokens"])
    min_document_chars = int(config["min_document_chars"])

    for row in source_iterator(catalog, str(definition["kind"])):
        emitted += 1
        raw = row.get(definition.get("text_field", "text"))
        if not isinstance(raw, str):
            continue
        text = normalize_text(raw)
        if len(text) < min_document_chars:
            rejected_short += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        key = digest[:16]
        if key in seen:
            rejected_exact += 1
            continue
        seen.add(key)
        near.consider(digest, text)
        ids_before_eos = tokenizer.encode(text, out_type=int)
        encoded_with_eos = len(ids_before_eos) + 1
        retained = min(encoded_with_eos, max_document_tokens)
        accepted += 1
        raw_characters += len(text)
        encoded_before_truncation += encoded_with_eos
        retained_tokens += retained
        if encoded_with_eos > retained:
            truncated += 1
        if progress_documents > 0 and emitted % progress_documents == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"source={definition['name']} emitted={emitted:,} accepted={accepted:,} "
                f"tokens={retained_tokens:,} docs/s={emitted / elapsed:,.1f}",
                flush=True,
            )

    target_predictions = round(
        int(config["prediction_token_target"]) * float(definition["weight"])
    )
    target_stored = round(int(config["train_tokens"]) * float(definition["weight"]))
    elapsed = time.time() - started
    sample_count = len(near.fingerprints)
    return {
        "format_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "dataset_name": config["dataset_name"],
        "source": definition["name"],
        "category": definition.get("category"),
        "kind": definition["kind"],
        "finite": bool(definition.get("finite", False)),
        "weight_cap": float(definition["weight"]),
        "target_prediction_tokens": target_predictions,
        "target_stored_tokens": target_stored,
        "usable_unique_tokens": retained_tokens,
        "capacity_fraction_of_cap": retained_tokens / target_stored,
        "shortfall_tokens": max(0, target_stored - retained_tokens),
        "documents_emitted": emitted,
        "documents_accepted": accepted,
        "documents_rejected_short": rejected_short,
        "documents_rejected_exact_duplicate": rejected_exact,
        "documents_truncated": truncated,
        "raw_characters": raw_characters,
        "encoded_tokens_before_truncation": encoded_before_truncation,
        "retained_tokens_after_truncation": retained_tokens,
        "exact_duplicate_fraction": rejected_exact / max(emitted, 1),
        "near_duplicate_sample": {
            "method": "64-bit 5-word SimHash, four 16-bit LSH bands",
            "selection_rate": near_sample_rate,
            "maximum_documents": near_sample_maximum,
            "sampled_documents": sample_count,
            "near_duplicates_hamming_le_3": near.near_duplicates,
            "estimated_fraction_in_sample": near.near_duplicates / max(sample_count, 1),
            "note": "Diagnostic estimate only; near matches are not removed by this audit.",
        },
        "elapsed_seconds": elapsed,
        "documents_per_second": emitted / max(elapsed, 1e-9),
        "tokens_per_second": retained_tokens / max(elapsed, 1e-9),
        "tokenizer": {
            "pieces": tokenizer.get_piece_size(),
            "eos_id": tokenizer.eos_id(),
            "sha256": sha256_file(tokenizer_path),
        },
        "config_sha256": sha256_file(config_path),
        "requested_revision": definition.get("revision"),
        "resolved_revision": catalog.revision,
        "license_note": definition.get("license_note"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/toolcall_4b.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/toolcall_spm_32k.model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/content/toolcall_4b_cache")
    parser.add_argument("--source", action="append", help="Finite source name; repeat as needed")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--near-sample-rate", type=float, default=0.02)
    parser.add_argument("--near-sample-maximum", type=int, default=100_000)
    parser.add_argument("--progress-documents", type=int, default=10_000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    tokenizer_path = Path(args.tokenizer).expanduser()
    if not tokenizer_path.is_absolute():
        tokenizer_path = root / tokenizer_path
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Frozen tokenizer not found: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != int(config["vocab_size"]):
        raise RuntimeError(
            f"Expected {config['vocab_size']:,} tokenizer pieces, "
            f"found {tokenizer.get_piece_size():,}"
        )
    if not 0.0 < args.near_sample_rate <= 1.0:
        parser.error("--near-sample-rate must be in (0, 1]")

    finite = [source for source in config["sources"] if source.get("finite")]
    requested = set(args.source or [source["name"] for source in finite])
    known = {source["name"] for source in finite}
    unknown = requested - known
    if unknown:
        parser.error(f"Unknown finite sources: {sorted(unknown)}")
    selected = [source for source in finite if source["name"] in requested]
    catalog = OpenAPICatalog(
        Path(args.cache_dir).expanduser().resolve(),
        int(config["seed"]),
        str(config["openapi_revision"]),
    )
    print(
        f"Pinned OpenAPI revision: {catalog.revision}; specs={len(catalog.spec_files):,}",
        flush=True,
    )
    for definition in selected:
        destination = output_dir / f"{definition['name']}.capacity.json"
        if destination.is_file() and args.skip_existing:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("config_sha256") == sha256_file(config_path)
                and existing.get("tokenizer", {}).get("sha256") == sha256_file(tokenizer_path)
                and existing.get("resolved_revision") == catalog.revision
            ):
                print(f"SKIP verified existing report: {destination}", flush=True)
                continue
            raise RuntimeError(
                f"Existing report does not match this config/tokenizer/revision: {destination}"
            )
        print(f"Auditing {definition['name']} ({definition['kind']})", flush=True)
        report = audit_source(
            definition,
            catalog,
            tokenizer,
            config,
            tokenizer_path,
            config_path,
            args.near_sample_rate,
            args.near_sample_maximum,
            args.progress_documents,
        )
        atomic_json(destination, report)
        print(
            f"COMPLETE {definition['name']}: {report['usable_unique_tokens']:,} unique "
            f"tokens; cap={report['target_stored_tokens']:,}; "
            f"shortfall={report['shortfall_tokens']:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
