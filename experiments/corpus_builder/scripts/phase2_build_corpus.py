#!/usr/bin/env python3
"""Restart-safe Phase 2 builder for the approved ToolCall 4B corpus.

CPU work and Hugging Face caches stay under /content. Completed shard
transactions, cursor checkpoints, dedup hashes, and progress JSON are committed
to the persistent workspace (normally Google Drive). The progress JSON is
written last and is therefore the transaction authority after a disconnect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sentencepiece as spm

from build_scaling_data import (
    OpenAPICatalog,
    atomic_json,
    encode_document,
    normalize_text,
    sha256_file,
    structured_documents,
    utc_now,
)


FORMAT_VERSION = 2


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if source.stat().st_size != temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete copy to persistent storage: {destination}")
    os.replace(temporary, destination)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_approval(
    config_path: Path,
    tokenizer_path: Path,
    audit_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    audit = load_json(audit_summary_path)
    if config.get("approval_status") != "approved" or int(config.get("phase", 0)) != 2:
        raise RuntimeError("Phase 2 requires the frozen approved configuration")
    if sha256_file(tokenizer_path) != config["tokenizer_sha256"]:
        raise RuntimeError("Frozen tokenizer SHA-256 does not match the approved config")
    if sha256_file(audit_summary_path) != config["audit_binding"]["capacity_summary_sha256"]:
        raise RuntimeError("Capacity summary SHA-256 does not match the approved config")
    if audit.get("config_sha256") != config["audit_binding"]["stage1_config_sha256"]:
        raise RuntimeError("Capacity report was produced from a different Stage 1 config")
    audited_allocations = {
        row["source"]: int(row["provisional_allocation_tokens"])
        for row in audit.get("provisional_allocations", [])
    }
    approved = {name: int(value) for name, value in config["approved_allocations"].items()}
    if audited_allocations != approved:
        raise RuntimeError("Approved allocations differ from the audited allocation table")
    if sum(approved.values()) != int(config["train_tokens"]):
        raise RuntimeError("Approved source allocations do not sum to train_tokens")
    by_source = {source["name"]: source for source in config["sources"]}
    if set(by_source) != set(approved):
        raise RuntimeError("Approved allocation/source names do not match")
    for name, target in approved.items():
        if int(by_source[name]["target_tokens"]) != target:
            raise RuntimeError(f"Source target mismatch for {name}")
    return config, audit


class InteractiveHFIterator:
    """One-row handshake around hf_stream_worker for transaction-safe cursors."""

    def __init__(
        self,
        definition: dict[str, Any],
        cache_dir: Path,
        local_state: Path,
        resume_state: Path | None,
    ) -> None:
        worker = Path(__file__).with_name("hf_stream_worker.py")
        command = [
            sys.executable,
            str(worker),
            "--dataset",
            definition["dataset"],
            "--split",
            definition.get("split", "train"),
            "--revision",
            definition["revision"],
            "--text-field",
            definition.get("text_field", "text"),
            "--license-field",
            definition.get("license_field", "license"),
            "--cache-dir",
            str(cache_dir / "huggingface"),
            "--source-name",
            definition["name"],
            "--checkpoint-state",
            str(local_state),
            "--interactive-checkpoints",
        ]
        if definition.get("subset") is not None:
            command.extend(["--subset", str(definition["subset"])])
        if resume_state is not None:
            command.extend(["--resume-state", str(resume_state)])
        self.name = definition["name"]
        self.local_state = local_state
        self.pending = False
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"Could not start Hugging Face worker for {self.name}")

    def _command(self, command: str, expect_control: bool = False) -> None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"Hugging Face worker {self.name} is closed")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        self.pending = False
        if expect_control:
            line = self.process.stdout.readline()
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid checkpoint response from {self.name}") from exc
            if value.get("_control") != "checkpoint_saved":
                raise RuntimeError(f"Unexpected checkpoint response from {self.name}: {value}")

    def __iter__(self) -> "InteractiveHFIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.pending:
            self._command("ACK")
        if self.process.stdout is None:
            raise StopIteration
        line = self.process.stdout.readline()
        if line:
            value = json.loads(line)
            if not isinstance(value, dict) or "_control" in value:
                raise RuntimeError(f"Unexpected row from Hugging Face worker {self.name}")
            self.pending = True
            return value
        return_code = self.process.wait()
        if return_code == 0:
            raise StopIteration
        raise RuntimeError(
            f"Hugging Face worker {self.name} exited with status {return_code}"
        )

    def checkpoint(self) -> Path:
        if not self.pending:
            raise RuntimeError("A cursor checkpoint requires one processed pending row")
        self._command("CHECKPOINT", expect_control=True)
        if not self.local_state.is_file():
            raise RuntimeError(f"Worker did not create cursor state: {self.local_state}")
        return self.local_state

    def close(self) -> None:
        if self.process.poll() is None:
            if self.pending:
                try:
                    self._command("STOP", expect_control=True)
                except Exception:
                    self.process.terminate()
            else:
                self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


class TransactionalShardWriter:
    """Write locally; copy only checkpoint-consistent files to persistent storage."""

    def __init__(
        self,
        durable_root: Path,
        local_root: Path,
        target_tokens: int,
        shard_tokens: int,
        progress: dict[str, Any] | None,
    ) -> None:
        self.durable_root = durable_root
        self.local_root = local_root
        self.target_tokens = target_tokens
        self.shard_tokens = shard_tokens
        self.local_shards = local_root / "shards"
        self.durable_shards = durable_root / "shards"
        self.local_shards.mkdir(parents=True, exist_ok=True)
        self.durable_shards.mkdir(parents=True, exist_ok=True)
        writer = (progress or {}).get("writer", {})
        self.total_tokens = int(writer.get("total_tokens", 0))
        self.shard_index = int(writer.get("shard_index", 0))
        self.current_tokens = int(writer.get("current_tokens", 0))
        self.records = list(writer.get("completed_records", []))
        self.current_snapshot = writer.get("current_snapshot")
        self._handle = None
        self._dirty: set[str] = set()
        if self.current_tokens:
            name = self.current_name
            if not isinstance(self.current_snapshot, dict):
                raise RuntimeError("Committed partial shard snapshot is missing")
            source = self.durable_root / self.current_snapshot["file"]
            if (
                not source.is_file()
                or source.stat().st_size != self.current_tokens * 2
                or sha256_file(source) != self.current_snapshot["sha256"]
            ):
                raise RuntimeError(f"Missing or invalid committed partial shard: {source}")
            shutil.copy2(source, self.local_shards / name)

    @property
    def current_name(self) -> str:
        return f"shard_{self.shard_index:05d}.bin"

    @property
    def complete(self) -> bool:
        return self.total_tokens >= self.target_tokens

    def _open(self) -> None:
        if self._handle is None:
            self._handle = (self.local_shards / self.current_name).open("ab")

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def _finish_full_shard(self) -> None:
        self._close_handle()
        path = self.local_shards / self.current_name
        self.records.append(
            {
                "file": f"shards/{path.name}",
                "tokens": self.current_tokens,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self._dirty.add(path.name)
        self.shard_index += 1
        self.current_tokens = 0

    def write(self, token_ids: list[int]) -> int:
        if self.complete:
            return 0
        values = np.asarray(token_ids, dtype=np.int64)
        values = values[: self.target_tokens - self.total_tokens]
        if values.size and (values.min() < 0 or values.max() > 65535):
            raise RuntimeError("Token ID does not fit uint16")
        offset = 0
        while offset < values.size:
            self._open()
            room = self.shard_tokens - self.current_tokens
            take = min(room, values.size - offset)
            payload = values[offset : offset + take].astype("<u2", copy=False).tobytes()
            self._handle.write(payload)
            self.current_tokens += take
            self.total_tokens += take
            offset += take
            self._dirty.add(self.current_name)
            if self.current_tokens == self.shard_tokens:
                self._finish_full_shard()
        return int(values.size)

    def commit_files(self, transaction_index: int) -> None:
        self._close_handle()
        for name in sorted(self._dirty):
            if self.current_tokens and name == self.current_name:
                continue
            source = self.local_shards / name
            destination = self.durable_shards / name
            copy_verified(source, destination)
        if self.current_tokens:
            source = self.local_shards / self.current_name
            relative = (
                f"partials/transaction_{transaction_index:05d}_"
                f"{self.current_name}"
            )
            destination = self.durable_root / relative
            copy_verified(source, destination)
            self.current_snapshot = {
                "file": relative,
                "tokens": self.current_tokens,
                "bytes": self.current_tokens * 2,
                "sha256": sha256_file(destination),
            }
        else:
            self.current_snapshot = None
        self._dirty.clear()

    def finalize_current(self) -> None:
        self._close_handle()
        if self.current_tokens:
            path = self.local_shards / self.current_name
            self.records.append(
                {
                    "file": f"shards/{path.name}",
                    "tokens": self.current_tokens,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            self._dirty.add(path.name)
            self.shard_index += 1
            self.current_tokens = 0
            self.current_snapshot = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "target_tokens": self.target_tokens,
            "shard_index": self.shard_index,
            "current_tokens": self.current_tokens,
            "current_snapshot": self.current_snapshot,
            "completed_records": self.records,
        }


def source_definition(config: dict[str, Any], name: str) -> dict[str, Any]:
    for source in config["sources"]:
        if source["name"] == name:
            return source
    raise KeyError(name)


def finite_iterator(
    definition: dict[str, Any], config: dict[str, Any], cache_dir: Path
) -> tuple[Iterator[dict[str, str]], str]:
    catalog = OpenAPICatalog(cache_dir, int(config["seed"]), definition["revision"])
    if catalog.revision != definition["revision"]:
        raise RuntimeError("Resolved OpenAPI commit differs from approved revision")
    kind = definition["kind"]
    if kind == "openapi_raw":
        return catalog.raw_documents(), catalog.revision
    if kind == "openapi_docs":
        return catalog.documentation(), catalog.revision
    if kind == "openapi_actions":
        return catalog.action_documents(), catalog.revision
    raise RuntimeError(f"Not a finite OpenAPI source: {kind}")


def read_dedup_chunks(stage_root: Path, progress: dict[str, Any] | None) -> set[bytes]:
    seen: set[bytes] = set()
    for record in (progress or {}).get("dedup_chunks", []):
        path = stage_root / record["file"]
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Dedup chunk size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Dedup chunk checksum mismatch: {path}")
        payload = path.read_bytes()
        if len(payload) % 16:
            raise RuntimeError(f"Invalid dedup chunk: {path}")
        seen.update(payload[index : index + 16] for index in range(0, len(payload), 16))
    return seen


def make_progress(
    *,
    name: str,
    config_sha: str,
    tokenizer_sha: str,
    writer: TransactionalShardWriter,
    counters: dict[str, int],
    chunks: list[dict[str, Any]],
    cursor_file: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "name": name,
        "updated_at": utc_now(),
        "config_sha256": config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "writer": writer.snapshot(),
        "counters": counters,
        "dedup_chunks": chunks,
        "cursor_file": cursor_file,
    }


def build_stream(
    *,
    name: str,
    definition: dict[str, Any],
    target_tokens: int,
    config: dict[str, Any],
    config_path: Path,
    tokenizer_path: Path,
    workspace: Path,
    cache_dir: Path,
    local_root: Path,
    max_session_minutes: float,
    selection: str,
) -> None:
    stage_root = workspace / "staging" / name
    complete_path = stage_root / "COMPLETE"
    if complete_path.is_file():
        print(f"SKIP complete stage: {name}")
        return
    stage_root.mkdir(parents=True, exist_ok=True)
    progress_path = stage_root / "progress.json"
    progress = load_json(progress_path) if progress_path.is_file() else None
    config_sha = sha256_file(config_path)
    tokenizer_sha = sha256_file(tokenizer_path)
    if progress and (
        progress.get("config_sha256") != config_sha
        or progress.get("tokenizer_sha256") != tokenizer_sha
    ):
        raise RuntimeError(f"Existing progress uses different inputs: {progress_path}")

    session_root = local_root / name
    if session_root.exists():
        shutil.rmtree(session_root)
    session_root.mkdir(parents=True)
    writer = TransactionalShardWriter(
        stage_root,
        session_root,
        target_tokens,
        int(config["shard_tokens"]),
        progress,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != int(config["vocab_size"]):
        raise RuntimeError("Tokenizer vocabulary size mismatch")
    counters = dict(
        (progress or {}).get(
            "counters",
            {
                "documents_read": 0,
                "documents_accepted": 0,
                "rejected_short": 0,
                "rejected_duplicate": 0,
                "rejected_license": 0,
                "rejected_split": 0,
                "documents_truncated": 0,
            },
        )
    )
    chunks = list((progress or {}).get("dedup_chunks", []))
    transaction_index = int((progress or {}).get("transaction_index", 0))
    seen = read_dedup_chunks(stage_root, progress)
    new_hashes: list[bytes] = []
    hf_worker: InteractiveHFIterator | None = None
    cursor_local: Path | None = None
    cursor_durable = stage_root / "cursor" / "hf_state.pkl"
    resolved_revision = definition.get("revision")

    if definition["kind"] == "huggingface":
        cursor_resume: Path | None = None
        if progress and progress.get("cursor_file"):
            durable = stage_root / progress["cursor_file"]
            cursor_resume = session_root / "resume_hf_state.pkl"
            shutil.copy2(durable, cursor_resume)
        cursor_local = session_root / "next_hf_state.pkl"
        hf_worker = InteractiveHFIterator(
            definition, cache_dir, cursor_local, cursor_resume
        )
        iterator: Iterator[dict[str, Any]] = hf_worker
    else:
        iterator, resolved_revision = finite_iterator(definition, config, cache_dir)
        for _ in range(int(counters["documents_read"])):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(f"Finite cursor exceeds source length for {name}") from exc

    allowed = {str(value).lower() for value in definition.get("allowed_licenses", [])}
    fraction = float(config["general_validation_selection_fraction"])
    started = time.time()
    last_committed_tokens = writer.total_tokens

    def commit(status: str) -> None:
        nonlocal new_hashes, chunks, transaction_index
        transaction_index += 1
        cursor_file: str | None = None
        if hf_worker is not None:
            assert cursor_local is not None
            checkpoint = hf_worker.checkpoint()
        else:
            checkpoint = None
        writer.commit_files(transaction_index)
        if new_hashes:
            chunk_name = f"hashes/chunk_{len(chunks):05d}.bin"
            chunk_path = stage_root / chunk_name
            payload = b"".join(new_hashes)
            write_bytes_atomic(chunk_path, payload)
            chunks.append(
                {
                    "file": chunk_name,
                    "digests": len(new_hashes),
                    "bytes": len(payload),
                    "sha256": sha256_file(chunk_path),
                }
            )
            new_hashes = []
        if checkpoint is not None:
            versioned_cursor = (
                stage_root / "cursor" / f"hf_state_{transaction_index:05d}.pkl"
            )
            copy_verified(checkpoint, versioned_cursor)
            cursor_file = versioned_cursor.relative_to(stage_root).as_posix()
        next_progress = make_progress(
                name=name,
                config_sha=config_sha,
                tokenizer_sha=tokenizer_sha,
                writer=writer,
                counters=counters,
                chunks=chunks,
                cursor_file=cursor_file,
                status=status,
            )
        next_progress["transaction_index"] = transaction_index
        atomic_json(progress_path, next_progress)

    try:
        while not writer.complete:
            try:
                row = next(iterator)
            except StopIteration:
                raise RuntimeError(
                    f"Source {name} exhausted at {writer.total_tokens:,}/"
                    f"{target_tokens:,} tokens"
                )
            counters["documents_read"] += 1
            if allowed:
                field = definition.get("license_field", "license")
                if str(row.get(field, "")).lower() not in allowed:
                    counters["rejected_license"] += 1
                    continue
            raw = row.get(definition.get("text_field", "text"))
            if not isinstance(raw, str):
                continue
            text = normalize_text(raw)
            if len(text) < int(config["min_document_chars"]):
                counters["rejected_short"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            key = digest[:16]
            if key in seen:
                counters["rejected_duplicate"] += 1
                continue
            is_general_validation = (
                int.from_bytes(digest[:8], "big") / 2**64 < fraction
            )
            keep = True
            if selection == "general_validation":
                keep = is_general_validation
            elif selection == "train" and definition.get("exclude_general_validation"):
                keep = not is_general_validation
            if not keep:
                counters["rejected_split"] += 1
                continue
            seen.add(key)
            new_hashes.append(key)
            raw_ids = tokenizer.encode(text, out_type=int)
            if len(raw_ids) + 1 > int(config["max_document_tokens"]):
                counters["documents_truncated"] += 1
            ids = encode_document(tokenizer, text, int(config["max_document_tokens"]))
            writer.write(ids)
            counters["documents_accepted"] += 1

            crossed_boundary = (
                writer.total_tokens // int(config["shard_tokens"])
                > last_committed_tokens // int(config["shard_tokens"])
            )
            if writer.complete or crossed_boundary:
                commit("complete" if writer.complete else "building")
                last_committed_tokens = writer.total_tokens
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"{name}: {writer.total_tokens:,}/{target_tokens:,} tokens "
                    f"({writer.total_tokens / target_tokens:.2%}); "
                    f"session={elapsed / 60:.1f}m; {writer.total_tokens / elapsed:,.0f} tok/s",
                    flush=True,
                )
                if not writer.complete and elapsed >= max_session_minutes * 60:
                    print(f"PAUSED {name}: session time budget reached", flush=True)
                    return
    except KeyboardInterrupt:
        if hf_worker is not None and not hf_worker.pending:
            print("Interrupt arrived between rows; last transaction remains authoritative.")
        elif writer.total_tokens > last_committed_tokens:
            commit("paused")
        raise
    finally:
        if hf_worker is not None:
            hf_worker.close()

    writer.finalize_current()
    writer.commit_files(transaction_index + 1)
    # The last row cursor and all final data were already committed above. Rewrite
    # progress with the finalized partial-shard record before declaring complete.
    final_progress = load_json(progress_path)
    final_progress["writer"] = writer.snapshot()
    final_progress["status"] = "complete"
    final_progress["updated_at"] = utc_now()
    atomic_json(progress_path, final_progress)
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "name": name,
        "category": definition.get("category"),
        "kind": definition["kind"],
        "target_tokens": target_tokens,
        "tokens": writer.total_tokens,
        "bytes": writer.total_tokens * 2,
        "shard_count": len(writer.records),
        "shard_tokens": int(config["shard_tokens"]),
        "resolved_revision": resolved_revision,
        "config_sha256": config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "counters": counters,
        "records": writer.records,
        "license_note": definition.get("license_note"),
        "completed_at": utc_now(),
    }
    atomic_json(stage_root / "manifest.json", manifest)
    complete_path.write_text(f"Completed {utc_now()}\n", encoding="utf-8")
    print(f"COMPLETE {name}: {writer.total_tokens:,} tokens in {len(writer.records)} shards")


def build_structured_validation(
    config: dict[str, Any],
    config_path: Path,
    tokenizer_path: Path,
    workspace: Path,
    local_root: Path,
) -> None:
    name = "validation_structured"
    target = int(config["validation_structured_tokens"])
    stage_root = workspace / "staging" / name
    if (stage_root / "COMPLETE").is_file():
        print(f"SKIP complete stage: {name}")
        return
    session_root = local_root / name
    if session_root.exists():
        shutil.rmtree(session_root)
    session_root.mkdir(parents=True)
    writer = TransactionalShardWriter(
        stage_root, session_root, target, int(config["shard_tokens"]), None
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    documents = 0
    for text in structured_documents(int(config["seed"]) + 99_991):
        writer.write(encode_document(tokenizer, text, int(config["max_document_tokens"])))
        documents += 1
        if writer.complete:
            break
    writer.finalize_current()
    writer.commit_files(1)
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "name": name,
        "tokens": writer.total_tokens,
        "bytes": writer.total_tokens * 2,
        "shard_count": len(writer.records),
        "shard_tokens": int(config["shard_tokens"]),
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "documents": documents,
        "records": writer.records,
        "note": "Deterministic diagnostic only; excluded from training.",
        "completed_at": utc_now(),
    }
    atomic_json(stage_root / "manifest.json", manifest)
    (stage_root / "COMPLETE").write_text(f"Completed {utc_now()}\n", encoding="utf-8")
    print(f"COMPLETE {name}: {writer.total_tokens:,} tokens")


def weighted_plan(manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    positions = {name: 0 for name in manifests}
    scheduled = {name: 0 for name in manifests}
    result: list[dict[str, Any]] = []
    while any(positions[name] < len(manifests[name]["records"]) for name in manifests):
        active = [
            name
            for name in sorted(manifests)
            if positions[name] < len(manifests[name]["records"])
        ]
        name = min(
            active,
            key=lambda value: (
                scheduled[value] / int(manifests[value]["tokens"]),
                value,
            ),
        )
        record = dict(manifests[name]["records"][positions[name]])
        result.append({"source": name, "source_record": record})
        positions[name] += 1
        scheduled[name] += int(record["tokens"])
    return result


def move_verified(source: Path, destination: Path, record: dict[str, Any]) -> None:
    expected_bytes = int(record["bytes"])
    expected_sha = record["sha256"]
    if destination.is_file():
        if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_sha:
            raise RuntimeError(f"Existing final shard is invalid: {destination}")
        if source.is_file():
            raise RuntimeError(f"Both source and final shard exist; inspect before continuing: {source}")
        return
    if not source.is_file():
        raise FileNotFoundError(f"Missing staged shard: {source}")
    if source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha:
        raise RuntimeError(f"Staged shard failed verification: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    write_bytes_atomic(path, payload.encode("utf-8"))


def assemble(
    config: dict[str, Any],
    config_path: Path,
    tokenizer_path: Path,
    audit_path: Path,
    workspace: Path,
    final_root: Path,
) -> None:
    if (final_root / "COMPLETE").is_file():
        print(f"SKIP complete final corpus: {final_root}")
        return
    manifests: dict[str, dict[str, Any]] = {}
    for name, target in config["approved_allocations"].items():
        stage = workspace / "staging" / name
        if not (stage / "COMPLETE").is_file():
            raise RuntimeError(f"Source is not complete: {name}")
        manifest = load_json(stage / "manifest.json")
        if int(manifest["tokens"]) != int(target):
            raise RuntimeError(f"Source token mismatch: {name}")
        manifests[name] = manifest
    for name, target in (
        ("validation_general", config["validation_general_tokens"]),
        ("validation_structured", config["validation_structured_tokens"]),
    ):
        stage = workspace / "staging" / name
        if not (stage / "COMPLETE").is_file():
            raise RuntimeError(f"Validation split is not complete: {name}")
        if int(load_json(stage / "manifest.json")["tokens"]) != int(target):
            raise RuntimeError(f"Validation token mismatch: {name}")

    final_root.mkdir(parents=True, exist_ok=True)
    (final_root / "BUILDING").write_text(
        "Assembly is incomplete until this marker is removed.\n", encoding="utf-8"
    )
    plan_path = workspace / "finalization_plan.json"
    if plan_path.is_file():
        plan = load_json(plan_path)["train"]
    else:
        plan = weighted_plan(manifests)
        atomic_json(
            plan_path,
            {
                "format_version": FORMAT_VERSION,
                "created_at": utc_now(),
                "config_sha256": sha256_file(config_path),
                "train": plan,
            },
        )

    final_train_records: list[dict[str, Any]] = []
    for index, item in enumerate(plan):
        source_name = item["source"]
        source_record = item["source_record"]
        source = workspace / "staging" / source_name / source_record["file"]
        destination = final_root / "train" / "shards" / f"shard_{index:05d}.bin"
        move_verified(source, destination, source_record)
        final_train_records.append(
            {
                "file": f"shards/{destination.name}",
                "tokens": int(source_record["tokens"]),
                "bytes": int(source_record["bytes"]),
                "sha256": source_record["sha256"],
                "source": source_name,
                "source_file": source_record["file"],
            }
        )

    split_manifests: dict[str, dict[str, Any]] = {}
    train_manifest = {
        "format_version": FORMAT_VERSION,
        "split": "train",
        "dtype": "uint16",
        "endianness": "little",
        "tokens": sum(int(record["tokens"]) for record in final_train_records),
        "bytes": sum(int(record["bytes"]) for record in final_train_records),
        "shard_count": len(final_train_records),
        "shard_tokens": int(config["shard_tokens"]),
    }
    (final_root / "train").mkdir(parents=True, exist_ok=True)
    atomic_json(final_root / "train" / "manifest.json", train_manifest)
    write_jsonl(final_root / "train" / "manifest_shards.jsonl", final_train_records)
    split_manifests["train"] = train_manifest

    for split in ("validation_general", "validation_structured"):
        source_root = workspace / "staging" / split
        source_manifest = load_json(source_root / "manifest.json")
        final_records: list[dict[str, Any]] = []
        for index, record in enumerate(source_manifest["records"]):
            source = source_root / record["file"]
            destination = final_root / split / "shards" / f"shard_{index:05d}.bin"
            move_verified(source, destination, record)
            final_records.append(
                {
                    "file": f"shards/{destination.name}",
                    "tokens": int(record["tokens"]),
                    "bytes": int(record["bytes"]),
                    "sha256": record["sha256"],
                }
            )
        manifest = {
            "format_version": FORMAT_VERSION,
            "split": split,
            "dtype": "uint16",
            "endianness": "little",
            "tokens": sum(int(record["tokens"]) for record in final_records),
            "bytes": sum(int(record["bytes"]) for record in final_records),
            "shard_count": len(final_records),
            "shard_tokens": int(config["shard_tokens"]),
        }
        (final_root / split).mkdir(parents=True, exist_ok=True)
        atomic_json(final_root / split / "manifest.json", manifest)
        write_jsonl(final_root / split / "manifest_shards.jsonl", final_records)
        split_manifests[split] = manifest

    tokenizer_destination = final_root / "tokenizer" / "toolcall_spm_32k.model"
    copy_verified(tokenizer_path, tokenizer_destination)
    shutil.copy2(config_path, final_root / "build_config.json")
    shutil.copy2(audit_path, final_root / "capacity_summary.json")
    bundle = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "dataset_name": config["dataset_name"],
        "created_at": utc_now(),
        "config_sha256": sha256_file(config_path),
        "audit_summary_sha256": sha256_file(audit_path),
        "tokenizer_sha256": sha256_file(tokenizer_destination),
        "splits": split_manifests,
        "sources": [
            {
                "name": name,
                "category": manifest.get("category"),
                "tokens": int(manifest["tokens"]),
                "fraction": int(manifest["tokens"]) / int(config["train_tokens"]),
                "resolved_revision": manifest.get("resolved_revision"),
                "license_note": manifest.get("license_note"),
                "counters": manifest.get("counters"),
            }
            for name, manifest in manifests.items()
        ],
        "assembly": {
            "method": "deterministic weighted-fair interleaving of source shards",
            "plan": "../finalization_plan.json",
            "note": "Staged shard files are moved, not duplicated, to stay within free Drive capacity.",
        },
    }
    atomic_json(final_root / "bundle_manifest.json", bundle)
    verify_final(config, final_root, checksums=True)
    (final_root / "BUILDING").unlink(missing_ok=True)
    (final_root / "COMPLETE").write_text(f"Completed {utc_now()}\n", encoding="utf-8")
    for name in list(manifests) + ["validation_general", "validation_structured"]:
        (workspace / "staging" / name / "MOVED_TO_FINAL").write_text(
            f"Moved to {final_root} at {utc_now()}\n", encoding="utf-8"
        )
    print(f"COMPLETE final corpus: {final_root}")


def verify_final(config: dict[str, Any], final_root: Path, checksums: bool) -> None:
    expected = {
        "train": int(config["train_tokens"]),
        "validation_general": int(config["validation_general_tokens"]),
        "validation_structured": int(config["validation_structured_tokens"]),
    }
    for split, target in expected.items():
        root = final_root / split
        manifest = load_json(root / "manifest.json")
        records = [
            json.loads(line)
            for line in (root / "manifest_shards.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        total = 0
        for record in records:
            path = root / record["file"]
            if not path.is_file() or path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Missing or wrong-sized shard: {path}")
            if int(record["bytes"]) != int(record["tokens"]) * 2:
                raise RuntimeError(f"Invalid uint16 byte count: {path}")
            if checksums and sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {path}")
            total += int(record["tokens"])
        if total != target or int(manifest["tokens"]) != target:
            raise RuntimeError(f"{split}: expected {target:,}, found {total:,}")
        print(f"PASS {split}: {total:,} tokens in {len(records)} shards")
    tokenizer = final_root / "tokenizer" / "toolcall_spm_32k.model"
    if sha256_file(tokenizer) != config["tokenizer_sha256"]:
        raise RuntimeError("Final tokenizer SHA-256 mismatch")
    print("PASS tokenizer SHA-256")


def status(config: dict[str, Any], workspace: Path, final_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    targets = dict(config["approved_allocations"])
    targets.update(
        {
            "validation_general": config["validation_general_tokens"],
            "validation_structured": config["validation_structured_tokens"],
        }
    )
    for name, target_value in targets.items():
        target = int(target_value)
        stage = workspace / "staging" / name
        progress_path = stage / "progress.json"
        manifest_path = stage / "manifest.json"
        if manifest_path.is_file():
            tokens = int(load_json(manifest_path).get("tokens", 0))
        elif progress_path.is_file():
            tokens = int(load_json(progress_path).get("writer", {}).get("total_tokens", 0))
        else:
            tokens = 0
        rows.append(
            {
                "name": name,
                "tokens": tokens,
                "target_tokens": target,
                "fraction": tokens / target,
                "complete": (stage / "COMPLETE").is_file(),
                "moved_to_final": (stage / "MOVED_TO_FINAL").is_file(),
            }
        )
    result = {
        "workspace": str(workspace),
        "final_root": str(final_root),
        "final_complete": (final_root / "COMPLETE").is_file(),
        "stages": rows,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["build-source", "build-general-validation", "build-structured-validation", "assemble", "verify", "status"],
    )
    parser.add_argument("--source")
    parser.add_argument("--config", default="configs/toolcall_4b_approved.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--final-root")
    parser.add_argument("--cache-dir", default="/content/toolcall_4b_cache")
    parser.add_argument("--local-root", default="/content/toolcall_4b_phase2")
    parser.add_argument("--max-session-minutes", type=float, default=300.0)
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    audit_path = Path(args.audit_summary).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    final_root = (
        Path(args.final_root).expanduser().resolve()
        if args.final_root
        else workspace / "final" / "toolcall_4b"
    )
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    local_root = Path(args.local_root).expanduser().resolve()
    config, _audit = validate_approval(config_path, tokenizer_path, audit_path)
    workspace.mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        status(config, workspace, final_root)
    elif args.command == "build-source":
        if not args.source:
            parser.error("build-source requires --source")
        definition = source_definition(config, args.source)
        build_stream(
            name=args.source,
            definition=definition,
            target_tokens=int(definition["target_tokens"]),
            config=config,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            workspace=workspace,
            cache_dir=cache_dir,
            local_root=local_root,
            max_session_minutes=args.max_session_minutes,
            selection="train",
        )
    elif args.command == "build-general-validation":
        definition = dict(source_definition(config, "general_fineweb_edu"))
        build_stream(
            name="validation_general",
            definition=definition,
            target_tokens=int(config["validation_general_tokens"]),
            config=config,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            workspace=workspace,
            cache_dir=cache_dir,
            local_root=local_root,
            max_session_minutes=args.max_session_minutes,
            selection="general_validation",
        )
    elif args.command == "build-structured-validation":
        build_structured_validation(
            config, config_path, tokenizer_path, workspace, local_root
        )
    elif args.command == "assemble":
        assemble(config, config_path, tokenizer_path, audit_path, workspace, final_root)
    elif args.command == "verify":
        verify_final(config, final_root, checksums=args.checksums)


if __name__ == "__main__":
    main()
