from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_scaling_data as builder
from build_scaling_data import (
    OpenAPICatalog,
    ShardWriter,
    normalize_text,
    split_is_validation,
)
from verify_scaling_data import mixture_tolerance, source_signature


def test_writer_exact_target_and_rollover(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path / "train", target_tokens=11, shard_tokens=4)
    assert writer.write([1, 2, 3]) == 3
    assert writer.write([4, 5, 6, 7, 8, 9, 10, 11, 12]) == 8
    manifest = writer.finish({"split": "train"})
    assert manifest["tokens"] == 11
    assert manifest["shard_count"] == 3
    records = [
        json.loads(line)
        for line in (tmp_path / "train" / "manifest_shards.jsonl").read_text().splitlines()
    ]
    assert [record["tokens"] for record in records] == [4, 4, 3]
    combined = np.concatenate(
        [np.fromfile(tmp_path / "train" / record["file"], dtype="<u2") for record in records]
    )
    assert combined.tolist() == list(range(1, 12))
    for record in records:
        path = tmp_path / "train" / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_normalization_and_split_are_deterministic() -> None:
    text = normalize_text("Ａ\r\n\r\n\r\n  B  \x00")
    assert text == "A\n\n  B"
    digest = hashlib.sha256(text.encode()).digest()
    assert split_is_validation(digest, 0.5) == split_is_validation(digest, 0.5)


def test_openapi_views_docs_and_actions(tmp_path: Path) -> None:
    repository = tmp_path / "openapi-directory"
    spec_path = repository / "APIs/example.com/1.0/openapi.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        """
openapi: 3.0.0
info:
  title: Example Calendar
  version: '1.0'
  created: 2026-02-31
paths:
  /events:
    post:
      operationId: createEvent
      summary: Create an event
      parameters:
        - name: calendar_id
          in: query
          required: true
          schema: {type: string}
      responses:
        '200': {description: Created}
""".strip(),
        encoding="utf-8",
    )
    catalog = OpenAPICatalog.__new__(OpenAPICatalog)
    catalog.root = repository
    catalog.seed = 42
    catalog.spec_files = [spec_path]
    loaded = catalog.load_spec(spec_path)
    assert loaded is not None
    assert loaded["info"]["created"] == "2026-02-31"
    raw = list(catalog.raw_documents())
    docs = list(catalog.documentation())
    actions = list(catalog.action_documents())
    assert len(raw) >= 3
    assert any("2026-02-31" in document["text"] for document in raw)
    assert len(docs) == 4
    assert any("createEvent" in document["text"] for document in docs)
    assert any('"decision": "call"' in document["text"] for document in actions)
    assert any("ask_clarification" in document["text"] for document in actions)
    assert all('"confidence"' in document["text"] for document in actions)


def test_source_signature_detects_mixture_changes() -> None:
    config = {
        "sources": [
            {
                "name": "general",
                "category": "clean_english_general",
                "kind": "huggingface",
                "weight": 0.45,
            },
            {
                "name": "structured",
                "category": "structured_text",
                "kind": "openapi_raw",
                "weight": 0.20,
            },
        ]
    }
    changed = json.loads(json.dumps(config))
    changed["sources"][0]["weight"] = 0.80
    assert source_signature(config) != source_signature(changed)


def test_mixture_tolerance_scales_with_bundle_size() -> None:
    config = {"max_document_tokens": 16_384}
    assert mixture_tolerance(config, 1_050_000) > 0.03
    assert mixture_tolerance(config, 475_000_000) == 0.001


def test_huggingface_source_uses_isolated_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeInfo:
        sha = "resolved-sha"

    class FakeApi:
        def dataset_info(self, *args, **kwargs):
            return FakeInfo()

    workers = []

    class FakeWorker:
        def __init__(self, definition, revision, cache_dir):
            self.rows = iter([{"text": "usable source text"}])
            self.closed = False
            workers.append(self)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.rows)

        def close(self):
            self.closed = True

    builder.close_source_states()
    monkeypatch.setattr(builder, "HfApi", FakeApi)
    monkeypatch.setattr(builder, "HuggingFaceSubprocessIterator", FakeWorker)
    states = builder.open_sources(
        {
            "seed": 42,
            "sources": [
                {
                    "name": "general",
                    "dataset": "example/data",
                    "kind": "huggingface",
                    "weight": 1.0,
                }
            ],
        },
        tmp_path,
    )
    assert next(states[0].iterator)["text"] == "usable source text"
    builder.close_source_states()
    assert workers[0].closed
    assert states[0].resource is None
    with pytest.raises(StopIteration):
        next(states[0].iterator)
