from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_build_corpus import (
    TransactionalShardWriter,
    assemble,
    atomic_json,
    sha256_file,
    verify_final,
)


def test_partial_snapshots_are_immutable_across_resume(tmp_path: Path) -> None:
    durable = tmp_path / "durable"
    first = TransactionalShardWriter(durable, tmp_path / "local-1", 23, 10, None)
    first.write(list(range(7)))
    first.commit_files(1)
    snapshot_1 = first.snapshot()
    old_partial = durable / snapshot_1["current_snapshot"]["file"]
    old_digest = sha256_file(old_partial)

    second = TransactionalShardWriter(
        durable, tmp_path / "local-2", 23, 10, {"writer": snapshot_1}
    )
    second.write(list(range(7, 16)))
    second.commit_files(2)
    snapshot_2 = second.snapshot()

    assert sha256_file(old_partial) == old_digest
    assert snapshot_2["total_tokens"] == 16
    assert [record["tokens"] for record in snapshot_2["completed_records"]] == [10]

    third = TransactionalShardWriter(
        durable, tmp_path / "local-3", 23, 10, {"writer": snapshot_2}
    )
    third.write(list(range(16, 23)))
    third.finalize_current()
    third.commit_files(3)
    assert [record["tokens"] for record in third.records] == [10, 10, 3]


def test_move_only_assembly_and_checksum_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    final_root = tmp_path / "final"
    tokenizer = tmp_path / "toolcall_spm_32k.model"
    tokenizer.write_bytes(b"test-tokenizer")
    config = {
        "dataset_name": "tiny-test",
        "train_tokens": 12,
        "validation_general_tokens": 3,
        "validation_structured_tokens": 2,
        "shard_tokens": 5,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "approved_allocations": {"a": 7, "b": 5},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")

    def create_stage(name: str, sizes: list[int], category: str | None = None) -> None:
        root = workspace / "staging" / name
        (root / "shards").mkdir(parents=True)
        records = []
        for index, tokens in enumerate(sizes):
            shard = root / "shards" / f"shard_{index:05d}.bin"
            shard.write_bytes(bytes([index + 1, 0]) * tokens)
            records.append(
                {
                    "file": f"shards/{shard.name}",
                    "tokens": tokens,
                    "bytes": tokens * 2,
                    "sha256": sha256_file(shard),
                }
            )
        atomic_json(
            root / "manifest.json",
            {
                "name": name,
                "tokens": sum(sizes),
                "records": records,
                "category": category,
                "resolved_revision": "test",
                "license_note": "test",
            },
        )
        (root / "COMPLETE").write_text("complete\n", encoding="utf-8")

    create_stage("a", [5, 2], "category-a")
    create_stage("b", [5], "category-b")
    create_stage("validation_general", [3])
    create_stage("validation_structured", [2])

    assemble(config, config_path, tokenizer, audit_path, workspace, final_root)
    verify_final(config, final_root, checksums=True)
    assert (final_root / "COMPLETE").is_file()
    assert not any((workspace / "staging" / "a" / "shards").glob("*.bin"))
