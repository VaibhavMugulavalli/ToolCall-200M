#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

from build_scaling_data import OpenAPICatalog, validate_config


def next_usable(
    iterator: Iterator[dict[str, Any]], definition: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    allowed = {
        str(value).lower() for value in definition.get("allowed_licenses", [])
    }
    license_field = definition.get("license_field", "license")
    text_field = definition.get("text_field", "text")
    for row in iterator:
        if allowed and str(row.get(license_field, "")).lower() not in allowed:
            continue
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            return row, text
    raise RuntimeError(f"{definition['name']} returned no usable text")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every configured source")
    parser.add_argument("--config", default="configs/scaling_470m.json")
    parser.add_argument("--cache-dir", default="/content/toolcall_scaling_cache")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = Path(args.config).expanduser()
    if not path.is_absolute():
        path = root / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    openapi = (
        OpenAPICatalog(cache_dir, int(config["seed"]))
        if any(
            str(source.get("kind", "huggingface")).startswith("openapi_")
            for source in config["sources"]
        )
        else None
    )
    api = HfApi()
    for definition in config["sources"]:
        kind = definition.get("kind", "huggingface")
        if kind == "huggingface":
            revision = api.dataset_info(
                definition["dataset"], revision=definition.get("revision", "main")
            ).sha
            # Deliberately do not shuffle this one-row probe. A shuffled streaming
            # dataset fills a large buffer and can leave HTTP workers alive while
            # this short-lived process is shutting down.
            stream = load_dataset(
                definition["dataset"],
                definition.get("subset"),
                split=definition.get("split", "train"),
                revision=revision,
                streaming=True,
            )
            iterator = iter(stream)
            row, text = next_usable(iterator, definition)
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            del iterator, stream
            gc.collect()
        else:
            if openapi is None:
                raise RuntimeError("OpenAPI catalog is unavailable")
            revision = openapi.revision
            if kind == "openapi_raw":
                iterator = openapi.raw_documents()
            elif kind == "openapi_docs":
                iterator = openapi.documentation()
            elif kind == "openapi_actions":
                iterator = openapi.action_documents()
            else:
                raise ValueError(f"Unsupported source kind: {kind}")
            row, text = next_usable(iterator, definition)
            iterator.close()
        print(
            f"PASS {definition['name']}: category={definition.get('category')} "
            f"revision={revision} chars={len(text):,} fields={sorted(row)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
