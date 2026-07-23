#!/usr/bin/env python3
"""Build deterministic uint16 train/validation shards for the scaling pilot."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sentencepiece as spm
import yaml
from huggingface_hub import HfApi


class OpenAPISafeLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves timestamp-looking metadata as text."""


# OpenAPI descriptions frequently contain unquoted date-like metadata, and the
# public directory includes invalid calendar dates. PyYAML's timestamp resolver
# would turn valid values into datetime/date objects and raise ValueError for an
# invalid one before the document can be inspected. OpenAPI does not require us
# to interpret these scalars as Python dates, so keep them as their source text.
OpenAPISafeLoader.yaml_implicit_resolvers = {
    first_character: [
        (tag, expression)
        for tag, expression in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for first_character, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert PyYAML-native scalars and non-string mapping keys to JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def safe_json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(json_safe(value), **kwargs)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            normalized.append(line)
        else:
            blank_count += 1
            if blank_count <= 1:
                normalized.append("")
    return "\n".join(normalized).strip()


@dataclass
class SourceState:
    definition: dict[str, Any]
    revision: str
    iterator: Iterator[dict[str, Any]]
    resource: Any | None = None
    allowed_licenses: set[str] | None = None
    documents_read: int = 0
    documents_written_train: int = 0
    documents_written_validation: int = 0
    tokens_written_train: int = 0
    tokens_written_validation: int = 0


_OPEN_SOURCE_STATES: list[SourceState] = []


def close_source_states() -> None:
    """Close partially consumed streaming iterators before HTTP worker shutdown."""
    while _OPEN_SOURCE_STATES:
        state = _OPEN_SOURCE_STATES.pop()
        iterator = state.iterator
        state.iterator = iter(())
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        resource = state.resource
        state.resource = None
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        del iterator, resource
    gc.collect()


atexit.register(close_source_states)


class HuggingFaceSubprocessIterator:
    """Stream HF rows through a child so Arrow/native failures stay isolated."""

    def __init__(
        self,
        definition: dict[str, Any],
        revision: str,
        cache_dir: Path,
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
            revision,
            "--text-field",
            definition.get("text_field", "text"),
            "--license-field",
            definition.get("license_field", "license"),
            "--cache-dir",
            str(cache_dir / "huggingface"),
            "--source-name",
            definition["name"],
        ]
        if definition.get("subset") is not None:
            command.extend(["--subset", str(definition["subset"])])
        self.name = definition["name"]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        if self.process.stdout is None:
            raise RuntimeError(f"Could not open source-worker pipe for {self.name}")

    def __iter__(self) -> "HuggingFaceSubprocessIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise StopIteration
        line = self.process.stdout.readline()
        if line:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid row from Hugging Face worker {self.name}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object row from Hugging Face worker {self.name}")
            return value
        return_code = self.process.wait()
        self.process.stdout.close()
        self.process.stdout = None
        if return_code == 0:
            raise StopIteration
        if return_code < 0:
            detail = f"signal {-return_code}"
        else:
            detail = f"exit status {return_code}"
        raise RuntimeError(
            f"Hugging Face worker {self.name} failed with {detail}. "
            "The builder process and partial BUILDING bundle remain inspectable."
        )

    def close(self) -> None:
        stdout = self.process.stdout
        if stdout is not None:
            stdout.close()
            self.process.stdout = None
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def chunk_document(text: str, header: str, max_chars: int = 32_000) -> Iterator[dict[str, str]]:
    """Split large specs without silently discarding their later sections."""
    if len(text) <= max_chars:
        yield {"text": f"{header}\n{text}"}
        return
    part_count = math.ceil(len(text) / max_chars)
    for part, start in enumerate(range(0, len(text), max_chars), start=1):
        yield {
            "text": (
                f"{header}\nPart {part} of {part_count}\n"
                + text[start : start + max_chars]
            )
        }


def safe_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value)
    return "_".join(part for part in slug.split("_") if part)[:120] or "tool"


class OpenAPICatalog:
    """Local, commit-recorded view of APIs-guru OpenAPI definitions."""

    repository_url = "https://github.com/APIs-guru/openapi-directory.git"

    def __init__(self, cache_dir: Path, seed: int) -> None:
        self.root = cache_dir / "openapi-directory"
        self.seed = seed
        self._ensure_repository()
        self.revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        names = {"openapi.yaml", "swagger.yaml", "openapi.json", "swagger.json"}
        self.spec_files = sorted(
            path
            for path in (self.root / "APIs").rglob("*")
            if path.is_file() and path.name in names
        )
        if not self.spec_files:
            raise RuntimeError(f"No OpenAPI definitions found under {self.root / 'APIs'}")

    def _ensure_repository(self) -> None:
        if (self.root / ".git").is_dir():
            return
        if self.root.exists():
            raise RuntimeError(
                f"Incomplete OpenAPI cache exists at {self.root}; remove that exact "
                "cache directory and rerun the source check."
            )
        self.root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                self.repository_url,
                str(self.root),
            ],
            check=True,
        )

    def shuffled_files(self, offset: int) -> list[Path]:
        paths = list(self.spec_files)
        random.Random(self.seed + offset).shuffle(paths)
        return paths

    @staticmethod
    def load_spec(path: Path) -> dict[str, Any] | None:
        try:
            loaded = yaml.load(
                path.read_text(encoding="utf-8", errors="ignore"),
                Loader=OpenAPISafeLoader,
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def raw_documents(self) -> Iterator[dict[str, str]]:
        for path in self.shuffled_files(1_001):
            relative = path.relative_to(self.root).as_posix()
            original = path.read_text(encoding="utf-8", errors="ignore")
            source_format = "JSON" if path.suffix == ".json" else "YAML"
            yield from chunk_document(
                original, f"OpenAPI source: {relative}\nFormat: {source_format}"
            )
            spec = self.load_spec(path)
            if spec is None:
                continue
            canonical = safe_json_dumps(
                spec, indent=2, sort_keys=True, ensure_ascii=False
            )
            yield from chunk_document(
                canonical, f"OpenAPI source: {relative}\nFormat: canonical JSON"
            )
            component_view = {
                "info": spec.get("info", {}),
                "components": spec.get("components", {}),
                "definitions": spec.get("definitions", {}),
                "parameters": spec.get("parameters", {}),
                "securityDefinitions": spec.get("securityDefinitions", {}),
            }
            yield from chunk_document(
                safe_json_dumps(
                    component_view, indent=2, sort_keys=True, ensure_ascii=False
                ),
                f"OpenAPI schema/component view: {relative}",
            )
            path_view = {
                "info": spec.get("info", {}),
                "servers": spec.get("servers", []),
                "basePath": spec.get("basePath"),
                "paths": spec.get("paths", {}),
            }
            yield from chunk_document(
                safe_json_dumps(
                    path_view, indent=2, sort_keys=True, ensure_ascii=False
                ),
                f"OpenAPI path/operation view: {relative}",
            )

    def operations(self, offset: int) -> Iterator[tuple[str, dict[str, Any], str, str, dict[str, Any]]]:
        methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
        for path in self.shuffled_files(offset):
            spec = self.load_spec(path)
            if spec is None:
                continue
            info = spec.get("info") if isinstance(spec.get("info"), dict) else {}
            api_title = str(info.get("title") or path.parent.parent.name)
            paths = spec.get("paths")
            if not isinstance(paths, dict):
                continue
            for route, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                path_parameters = path_item.get("parameters", [])
                for method, operation in path_item.items():
                    if str(method).lower() not in methods or not isinstance(operation, dict):
                        continue
                    operation = dict(operation)
                    operation["_path_parameters"] = path_parameters
                    yield api_title, spec, str(route), str(method).upper(), operation

    def documentation(self) -> Iterator[dict[str, str]]:
        for api_title, _spec, route, method, operation in self.operations(2_003):
            operation_id = str(operation.get("operationId") or f"{method}_{route}")
            parameters = []
            for value in list(operation.get("_path_parameters") or []) + list(
                operation.get("parameters") or []
            ):
                if isinstance(value, dict):
                    parameters.append(value)
            document = {
                "parameters": parameters,
                "requestBody": operation.get("requestBody"),
                "responses": operation.get("responses", {}),
                "security": operation.get("security"),
            }
            text = (
                f"# {api_title} API\n\n"
                f"## {method} {route}\n\n"
                f"Operation: `{operation_id}`\n\n"
                f"Summary: {operation.get('summary', '')}\n\n"
                f"Description:\n{operation.get('description', '')}\n\n"
                "### Parameters, request body, responses, and security\n\n"
                "```json\n"
                + safe_json_dumps(
                    document, indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n```"
            )
            yield from chunk_document(text, "API reference document")
            request_guide = (
                f"# Calling {operation_id}\n\n"
                f"API: {api_title}\nEndpoint: {method} {route}\n\n"
                f"## Parameters and request body\n\n```json\n"
                + safe_json_dumps(
                    {
                        "parameters": parameters,
                        "requestBody": operation.get("requestBody"),
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n```"
            )
            yield from chunk_document(request_guide, "API request guide")
            response_guide = (
                f"# Responses from {operation_id}\n\n"
                f"API: {api_title}\nEndpoint: {method} {route}\n\n```json\n"
                + safe_json_dumps(
                    operation.get("responses", {}),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n```"
            )
            yield from chunk_document(response_guide, "API response guide")
            parameter_names = [
                str(parameter.get("name"))
                for parameter in parameters
                if isinstance(parameter, dict) and parameter.get("name")
            ]
            signature = (
                f"# SDK-style function reference\n\n"
                f"`{safe_slug(operation_id)}({', '.join(parameter_names)})`\n\n"
                f"Calls `{method} {route}` in {api_title}.\n\n"
                f"{operation.get('summary', '')}\n\n{operation.get('description', '')}"
            )
            yield from chunk_document(signature, "Function and method documentation")

    @staticmethod
    def operation_schema(
        api_title: str, route: str, method: str, operation: dict[str, Any]
    ) -> tuple[str, dict[str, Any], list[str]]:
        operation_id = str(operation.get("operationId") or f"{method}_{route}")
        tool_name = safe_slug(f"{api_title}_{operation_id}")
        properties: dict[str, Any] = {}
        required: list[str] = []
        parameters = list(operation.get("_path_parameters") or []) + list(
            operation.get("parameters") or []
        )
        for parameter in parameters:
            if not isinstance(parameter, dict) or "$ref" in parameter:
                continue
            name = str(parameter.get("name") or "argument")
            schema = parameter.get("schema")
            if not isinstance(schema, dict):
                schema = {"type": parameter.get("type", "string")}
            properties[name] = {
                **schema,
                "description": parameter.get("description", ""),
            }
            if parameter.get("required") and name not in required:
                required.append(name)
        request_body = operation.get("requestBody")
        if isinstance(request_body, dict):
            content = request_body.get("content")
            if isinstance(content, dict) and content:
                media = next((value for value in content.values() if isinstance(value, dict)), {})
                body_schema = media.get("schema", {}) if isinstance(media, dict) else {}
                properties["body"] = body_schema if isinstance(body_schema, dict) else {"type": "object"}
                if request_body.get("required"):
                    required.append("body")
        schema = {
            "name": tool_name,
            "description": operation.get("summary") or operation.get("description") or f"{method} {route}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        return tool_name, schema, required

    def action_documents(self) -> Iterator[dict[str, str]]:
        rng = random.Random(self.seed + 3_007)
        for api_title, _spec, route, method, operation in self.operations(3_007):
            tool_name, schema, required = self.operation_schema(
                api_title, route, method, operation
            )
            arguments = {
                name: f"sample_{safe_slug(name)}_{rng.getrandbits(32):08x}"
                for name in schema["parameters"]["properties"]
            }
            direct = {
                "decision": "call",
                "tool_name": tool_name,
                "arguments": arguments,
                "missing_required_fields": [],
                "confidence": 0.93,
            }
            request = f"Use {api_title} to perform {method} {route} with the supplied details."
            yield {
                "text": (
                    "<|user|>\n"
                    + request
                    + "\n<|tool_schema|>\n"
                    + safe_json_dumps(schema, sort_keys=True, ensure_ascii=False)
                    + "\n<|assistant|>\n"
                    + safe_json_dumps(direct, sort_keys=True, ensure_ascii=False)
                )
            }
            if required:
                missing = rng.choice(required)
                incomplete = {key: value for key, value in arguments.items() if key != missing}
                clarification = {
                    "decision": "ask_clarification",
                    "tool_name": tool_name,
                    "arguments": incomplete,
                    "missing_required_fields": [missing],
                    "confidence": 0.97,
                }
                yield {
                    "text": (
                        "<|user|>\n"
                        + f"Please run {tool_name}, but one required detail is absent."
                        + "\n<|tool_schema|>\n"
                        + safe_json_dumps(
                            schema, sort_keys=True, ensure_ascii=False
                        )
                        + "\n<|assistant|>\n"
                        + safe_json_dumps(
                            clarification, sort_keys=True, ensure_ascii=False
                        )
                    )
                }
            no_call = {
                "decision": "no_call",
                "tool_name": None,
                "arguments": {},
                "missing_required_fields": [],
                "confidence": 0.91,
            }
            yield {
                "text": (
                    "<|user|>\n"
                    + f"Write a poem instead of using the available {tool_name} tool."
                    + "\n<|tool_schema|>\n"
                    + safe_json_dumps(schema, sort_keys=True, ensure_ascii=False)
                    + "\n<|assistant|>\n"
                    + safe_json_dumps(no_call, sort_keys=True, ensure_ascii=False)
                )
            }


class ShardWriter:
    """Write an exact-size logical token stream into checksummed uint16 shards."""

    def __init__(self, root: Path, target_tokens: int, shard_tokens: int) -> None:
        if target_tokens <= 0 or shard_tokens <= 0:
            raise ValueError("target_tokens and shard_tokens must be positive")
        self.root = root
        self.target_tokens = target_tokens
        self.shard_tokens = shard_tokens
        self.shards_dir = root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=False)
        self.total_tokens = 0
        self.shard_index = -1
        self.shard_count = 0
        self._handle = None
        self._digest = None
        self._shard_path: Path | None = None
        self._shard_written = 0
        self.records: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return self.target_tokens - self.total_tokens

    @property
    def complete(self) -> bool:
        return self.total_tokens >= self.target_tokens

    def _open_shard(self) -> None:
        self.shard_index += 1
        self._shard_path = self.shards_dir / f"shard_{self.shard_index:05d}.bin"
        self._handle = self._shard_path.open("wb")
        self._digest = hashlib.sha256()
        self._shard_written = 0

    def _close_shard(self) -> None:
        if self._handle is None or self._shard_path is None or self._digest is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.records.append(
            {
                "file": f"shards/{self._shard_path.name}",
                "tokens": self._shard_written,
                "bytes": self._shard_written * 2,
                "sha256": self._digest.hexdigest(),
            }
        )
        self.shard_count += 1
        self._handle = None
        self._digest = None
        self._shard_path = None
        self._shard_written = 0

    def write(self, token_ids: list[int] | np.ndarray) -> int:
        if self.complete:
            return 0
        values = np.asarray(token_ids, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint16).max):
            raise ValueError("token id is outside uint16 range")
        values = values[: self.remaining]
        written = 0
        while written < values.size:
            if self._handle is None:
                self._open_shard()
            room = self.shard_tokens - self._shard_written
            take = min(room, values.size - written)
            payload = values[written : written + take].astype("<u2", copy=False).tobytes()
            assert self._handle is not None and self._digest is not None
            self._handle.write(payload)
            self._digest.update(payload)
            self._shard_written += take
            self.total_tokens += take
            written += take
            if self._shard_written == self.shard_tokens:
                self._close_shard()
        return written

    def finish(self, metadata: dict[str, Any]) -> dict[str, Any]:
        self._close_shard()
        if not self.complete:
            raise RuntimeError(
                f"Incomplete writer at {self.root}: {self.total_tokens:,}/"
                f"{self.target_tokens:,} tokens"
            )
        manifest = {
            "format_version": 1,
            "dtype": "uint16",
            "endianness": "little",
            "tokens": self.total_tokens,
            "bytes": self.total_tokens * 2,
            "shard_tokens": self.shard_tokens,
            "shard_count": self.shard_count,
            "created_at": utc_now(),
            **metadata,
        }
        atomic_json(self.root / "manifest.json", manifest)
        with (self.root / "manifest_shards.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return manifest


def validate_config(config: dict[str, Any]) -> None:
    required_positive = (
        "train_tokens",
        "validation_general_tokens",
        "validation_structured_tokens",
        "shard_tokens",
        "min_document_chars",
        "max_document_tokens",
    )
    for key in required_positive:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["vocab_size"]) > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size does not fit uint16")
    if not 0.0 < float(config["validation_fraction"]) < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("At least one source is required")
    if any(float(source["weight"]) <= 0 for source in sources):
        raise ValueError("Every source weight must be positive")
    total_weight = sum(float(source["weight"]) for source in sources)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Source weights must sum to 1.0, got {total_weight}")
    supported_kinds = {"huggingface", "openapi_raw", "openapi_docs", "openapi_actions"}
    for source in sources:
        kind = source.get("kind", "huggingface")
        if kind not in supported_kinds:
            raise ValueError(f"Unsupported source kind: {kind}")
        if kind == "huggingface" and not source.get("dataset"):
            raise ValueError(f"Hugging Face source {source.get('name')} needs dataset")
    expected_categories = config.get("category_weights", {})
    realized_categories: dict[str, float] = {}
    for source in sources:
        category = str(source.get("category", ""))
        realized_categories[category] = realized_categories.get(category, 0.0) + float(
            source["weight"]
        )
    if realized_categories != expected_categories:
        raise ValueError(
            f"Source/category weights {realized_categories} do not match "
            f"category_weights {expected_categories}"
        )


def open_sources(config: dict[str, Any], cache_dir: Path) -> list[SourceState]:
    api = HfApi()
    states: list[SourceState] = []
    openapi_catalog = (
        OpenAPICatalog(cache_dir, int(config["seed"]))
        if any(
            str(source.get("kind", "huggingface")).startswith("openapi_")
            for source in config["sources"]
        )
        else None
    )
    for definition in config["sources"]:
        kind = definition.get("kind", "huggingface")
        requested_revision = definition.get("revision", "main")
        resource = None
        if kind == "huggingface":
            info = api.dataset_info(definition["dataset"], revision=requested_revision)
            resolved_revision = info.sha
            # Datasets/PyArrow lives only in this worker. A native failure is
            # reported as a worker error instead of segfaulting the shard writer.
            iterator = HuggingFaceSubprocessIterator(
                definition, resolved_revision, cache_dir
            )
        else:
            if openapi_catalog is None:
                raise RuntimeError(
                    f"OpenAPI catalog was not initialized for {definition['name']}"
                )
            resolved_revision = openapi_catalog.revision
            if kind == "openapi_raw":
                iterator = openapi_catalog.raw_documents()
            elif kind == "openapi_docs":
                iterator = openapi_catalog.documentation()
            elif kind == "openapi_actions":
                iterator = openapi_catalog.action_documents()
            else:
                raise ValueError(f"Unsupported source kind: {kind}")
        state = SourceState(
            definition=definition,
            revision=resolved_revision,
            iterator=iterator,
            resource=resource,
            allowed_licenses=(
                {str(value).lower() for value in definition["allowed_licenses"]}
                if definition.get("allowed_licenses")
                else None
            ),
        )
        states.append(state)
        _OPEN_SOURCE_STATES.append(state)
        print(
            f"source={definition['name']} category={definition.get('category')} "
            f"kind={kind} revision={resolved_revision} weight={definition['weight']}",
            flush=True,
        )
    return states


def split_is_validation(content_digest: bytes, validation_fraction: float) -> bool:
    bucket = int.from_bytes(content_digest[:8], "big") / 2**64
    return bucket < validation_fraction


def encode_document(
    tokenizer: spm.SentencePieceProcessor,
    text: str,
    max_tokens: int,
) -> list[int]:
    ids = tokenizer.encode(text, out_type=int)
    eos_id = tokenizer.eos_id()
    if eos_id < 0:
        raise RuntimeError("SentencePiece model has no EOS token")
    if len(ids) >= max_tokens:
        ids = ids[: max_tokens - 1]
    ids.append(eos_id)
    return ids


def structured_documents(seed: int) -> Iterator[str]:
    rng = random.Random(seed)
    tools = [
        ("calendar.create_event", ["title", "start_time"], ["end_time", "attendees"]),
        ("gmail.search_emails", ["query"], ["after", "before", "max_results"]),
        ("tasks.create", ["title"], ["due_date", "priority", "project"]),
        ("weather.forecast", ["location"], ["date", "units"]),
        ("github.create_issue", ["repository", "title"], ["body", "labels"]),
        ("files.search", ["query"], ["directory", "extension"]),
        ("web.search", ["query"], ["recency_days", "domains"]),
        ("database.query", ["database", "sql"], ["limit"]),
    ]
    names = ["Aarav", "Maya", "Noah", "Priya", "Sam", "Zoe"]
    cities = ["Bengaluru", "Delhi", "London", "Singapore", "Toronto", "Tokyo"]
    while True:
        serial = rng.getrandbits(64)
        tool_name, required, optional = rng.choice(tools)
        properties = {
            field: {"type": "string", "description": f"Value for {field}"}
            for field in required + optional
        }
        schema = {
            "name": tool_name,
            "description": f"Execute {tool_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        arguments: dict[str, Any] = {}
        for field in required + optional:
            if field in optional and rng.random() < 0.45:
                continue
            if field == "location":
                value: Any = rng.choice(cities)
            elif field == "attendees":
                value = [f"{rng.choice(names).lower()}@example.com"]
            elif field in {"max_results", "limit", "recency_days"}:
                value = rng.choice([5, 10, 20, 30])
            elif "time" in field or "date" in field or field in {"after", "before", "due_date"}:
                value = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 20):02d}:30:00+05:30"
            else:
                value = f"{field.replace('_', ' ')} {serial:x}"
            arguments[field] = value

        mode = rng.random()
        if mode < 0.15:
            missing = rng.choice(required)
            arguments.pop(missing, None)
            request = f"Please use {tool_name} for request {serial:x}, but I forgot one detail."
            answer = {
                "decision": "ask_clarification",
                "tool_name": tool_name,
                "arguments": arguments,
                "missing_required_fields": [missing],
                "confidence": 0.96,
            }
        elif mode < 0.22:
            request = f"Tell me a joke about request {serial:x}; do not use any tool."
            answer = {
                "decision": "no_call",
                "tool_name": None,
                "arguments": {},
                "missing_required_fields": [],
                "confidence": 0.92,
            }
        else:
            request = f"Run {tool_name} with these details: {json.dumps(arguments, ensure_ascii=False)}"
            answer = {
                "decision": "call",
                "tool_name": tool_name,
                "arguments": arguments,
                "missing_required_fields": [],
                "confidence": 0.94,
            }
        yield (
            "<|user|>\n"
            + request
            + "\n<|tool_schema|>\n"
            + json.dumps(schema, sort_keys=True, ensure_ascii=False)
            + "\n<|assistant|>\n"
            + json.dumps(answer, sort_keys=True, ensure_ascii=False)
        )


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite only when you "
                "intend to rebuild this exact directory."
            )
        protected = {
            Path("/").resolve(),
            Path.home().resolve(),
            Path("/content").resolve(),
            Path("/content/drive").resolve(),
            Path("/content/drive/MyDrive").resolve(),
        }
        if output_root.resolve() in protected or len(output_root.resolve().parts) < 3:
            raise RuntimeError(f"Refusing to overwrite broad path: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    (output_root / "BUILDING").write_text(
        "Generation is incomplete until this marker is removed.\n", encoding="utf-8"
    )


def copy_tokenizer_files(tokenizer_path: Path, output_root: Path) -> list[dict[str, Any]]:
    destination = output_root / "tokenizer"
    destination.mkdir()
    candidates = [tokenizer_path]
    for suffix in (".vocab",):
        sibling = tokenizer_path.with_suffix(suffix)
        if sibling.is_file():
            candidates.append(sibling)
    manifest_sibling = tokenizer_path.parent / "tokenizer_manifest.json"
    if manifest_sibling.is_file():
        candidates.append(manifest_sibling)
    copied: list[dict[str, Any]] = []
    for source in candidates:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(
            {
                "file": f"tokenizer/{target.name}",
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaling_470m.json")
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer/toolcall_spm_32k.model"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-dir",
        default="/content/toolcall_scaling_cache",
        help="Local cache for the APIs-guru repository (keep this off Google Drive)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit-train-tokens",
        type=int,
        help="Testing override; also scales validation targets down",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    tokenizer_path = Path(args.tokenizer).expanduser()
    if not tokenizer_path.is_absolute():
        tokenizer_path = project_root / tokenizer_path
    output_root = Path(args.output).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.limit_train_tokens:
        config["dataset_name"] += f"_test_{args.limit_train_tokens}"
        config["train_tokens"] = args.limit_train_tokens
        config["validation_general_tokens"] = max(10_000, args.limit_train_tokens // 20)
        config["validation_structured_tokens"] = max(10_000, args.limit_train_tokens // 50)
        config["shard_tokens"] = min(config["shard_tokens"], args.limit_train_tokens)
        config["progress_every_tokens"] = max(10_000, args.limit_train_tokens // 10)
    validate_config(config)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Copy the frozen "
            "toolcall_spm_32k.model into scaling_data/artifacts/tokenizer first."
        )
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size() != int(config["vocab_size"]):
        raise RuntimeError(
            f"Expected a {config['vocab_size']:,}-piece tokenizer, found "
            f"{tokenizer.get_piece_size():,}"
        )

    prepare_output(output_root, args.overwrite)
    started = time.time()
    train_writer = ShardWriter(
        output_root / "train", int(config["train_tokens"]), int(config["shard_tokens"])
    )
    validation_writer = ShardWriter(
        output_root / "validation_general",
        int(config["validation_general_tokens"]),
        int(config["shard_tokens"]),
    )
    structured_writer = ShardWriter(
        output_root / "validation_structured",
        int(config["validation_structured_tokens"]),
        int(config["shard_tokens"]),
    )

    states = open_sources(config, cache_dir)
    rng = random.Random(int(config["seed"]))
    seen_hashes: set[bytes] = set()
    rejected_short = 0
    rejected_duplicate = 0
    next_progress = int(config["progress_every_tokens"])
    active = list(states)
    while not (train_writer.complete and validation_writer.complete):
        if not active:
            raise RuntimeError("All configured streams ended before token targets were met")
        # Weighted fair scheduling uses tokens already written, not document
        # counts, so different average document lengths do not skew the mix.
        normalized_tokens = [
            (
                state.tokens_written_train + state.tokens_written_validation
            )
            / float(state.definition["weight"])
            for state in active
        ]
        minimum = min(normalized_tokens)
        candidates = [
            state
            for state, normalized in zip(active, normalized_tokens)
            if math.isclose(normalized, minimum, rel_tol=0.0, abs_tol=1e-9)
        ]
        source = rng.choice(candidates)
        try:
            row = next(source.iterator)
        except StopIteration:
            written = source.tokens_written_train + source.tokens_written_validation
            expected = round(
                (int(config["train_tokens"]) + int(config["validation_general_tokens"]))
                * float(source.definition["weight"])
            )
            raise RuntimeError(
                f"Source {source.definition['name']} exhausted after {written:,} tokens; "
                f"approximately {expected:,} were required to preserve the configured mix."
            )
        source.documents_read += 1
        if source.allowed_licenses:
            license_field = source.definition.get("license_field", "license")
            if str(row.get(license_field, "")).lower() not in source.allowed_licenses:
                continue
        raw = row.get(source.definition.get("text_field", "text"))
        if not isinstance(raw, str):
            continue
        text = normalize_text(raw)
        if len(text) < int(config["min_document_chars"]):
            rejected_short += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        short_digest = digest[:16]
        if short_digest in seen_hashes:
            rejected_duplicate += 1
            continue
        seen_hashes.add(short_digest)
        validation = split_is_validation(digest, float(config["validation_fraction"]))
        if validation and validation_writer.complete:
            continue
        if not validation and train_writer.complete:
            continue
        ids = encode_document(tokenizer, text, int(config["max_document_tokens"]))
        if validation:
            written = validation_writer.write(ids)
            source.documents_written_validation += 1
            source.tokens_written_validation += written
        else:
            written = train_writer.write(ids)
            source.documents_written_train += 1
            source.tokens_written_train += written
        total = train_writer.total_tokens + validation_writer.total_tokens
        if total >= next_progress:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"general={total:,}/{config['train_tokens'] + config['validation_general_tokens']:,} "
                f"train={train_writer.total_tokens:,} val={validation_writer.total_tokens:,} "
                f"rate={total / elapsed:,.0f} tok/s unique_docs={len(seen_hashes):,}",
                flush=True,
            )
            next_progress += int(config["progress_every_tokens"])

    close_source_states()

    for text in structured_documents(int(config["seed"]) + 99_991):
        if structured_writer.complete:
            break
        structured_writer.write(
            encode_document(tokenizer, text, int(config["max_document_tokens"]))
        )

    source_summary = [
        {
            "name": state.definition["name"],
            "category": state.definition.get("category"),
            "kind": state.definition.get("kind", "huggingface"),
            "dataset": state.definition.get("dataset"),
            "repository": (
                OpenAPICatalog.repository_url
                if str(state.definition.get("kind", "")).startswith("openapi_")
                else None
            ),
            "subset": state.definition.get("subset"),
            "requested_revision": state.definition.get("revision", "main"),
            "resolved_revision": state.revision,
            "weight": state.definition["weight"],
            "allowed_licenses": state.definition.get("allowed_licenses"),
            "license_note": state.definition.get("license_note"),
            "documents_read": state.documents_read,
            "documents_written_train": state.documents_written_train,
            "documents_written_validation": state.documents_written_validation,
            "tokens_written_train": state.tokens_written_train,
            "tokens_written_validation": state.tokens_written_validation,
            "realized_general_token_fraction": (
                state.tokens_written_train + state.tokens_written_validation
            )
            / (train_writer.total_tokens + validation_writer.total_tokens),
        }
        for state in states
    ]
    shared_metadata = {
        "dataset_name": config["dataset_name"],
        "tokenizer": "tokenizer/toolcall_spm_32k.model",
        "vocab_size": tokenizer.get_piece_size(),
        "eos_id": tokenizer.eos_id(),
        "seed": config["seed"],
    }
    manifests = {
        "train": train_writer.finish({**shared_metadata, "split": "train"}),
        "validation_general": validation_writer.finish(
            {**shared_metadata, "split": "validation_general"}
        ),
        "validation_structured": structured_writer.finish(
            {
                **shared_metadata,
                "split": "validation_structured",
                "note": "Deterministic diagnostic only; excluded from training and scaling fit.",
            }
        ),
    }
    tokenizer_files = copy_tokenizer_files(tokenizer_path, output_root)
    config_snapshot = output_root / "build_config.json"
    atomic_json(config_snapshot, config)
    bundle = {
        "format_version": 1,
        "status": "complete",
        "dataset_name": config["dataset_name"],
        "created_at": utc_now(),
        "elapsed_seconds": time.time() - started,
        "config_file": "build_config.json",
        "tokenizer_files": tokenizer_files,
        "splits": manifests,
        "sources": source_summary,
        "deduplication": {
            "method": "normalized-text SHA-256 (first 128 bits retained in memory)",
            "unique_documents": len(seen_hashes),
            "rejected_duplicates": rejected_duplicate,
            "rejected_short_documents": rejected_short,
        },
    }
    atomic_json(output_root / "bundle_manifest.json", bundle)
    (output_root / "BUILDING").unlink()
    (output_root / "COMPLETE").write_text(
        f"Completed {utc_now()}\n", encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, sort_keys=True), flush=True)
    print(f"\nComplete data bundle: {output_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. The BUILDING marker remains; rerun with --overwrite.", file=sys.stderr)
        raise
    finally:
        # Run cleanup while Python is still fully initialized. Relying only on
        # interpreter finalization is unsafe with Colab's HF/parquet workers.
        close_source_states()
