#!/usr/bin/env python3
"""Isolated Hugging Face streaming worker; stdout is reserved for JSON rows."""

from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import sys
import time
from pathlib import Path

# Xet is unnecessary for HTTP parquet streaming and adds another native layer.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    Timeout as RequestsTimeout,
)
from urllib3.exceptions import (
    IncompleteRead as Urllib3IncompleteRead,
    NewConnectionError,
    ProtocolError,
    ReadTimeoutError,
)


TRANSIENT_NETWORK_ERRORS = (
    ChunkedEncodingError,
    RequestsConnectionError,
    RequestsTimeout,
    Urllib3IncompleteRead,
    http.client.IncompleteRead,
    NewConnectionError,
    ProtocolError,
    ReadTimeoutError,
)


def exception_chain(error: BaseException):
    """Yield an exception and its explicit/implicit causes once each."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_network_error(error: BaseException) -> bool:
    """Recognize retryable HTTP failures even when PyArrow wraps them."""
    return any(isinstance(item, TRANSIENT_NETWORK_ERRORS) for item in exception_chain(error))


def close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def retry_delay(attempt: int, initial: float, maximum: float) -> float:
    return min(maximum, initial * (2 ** max(0, attempt - 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--license-field", default="license")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--max-reconnects", type=int, default=100)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.max_retries < 0 or args.max_reconnects < 0:
        parser.error("retry limits must be non-negative")
    if args.retry_initial_seconds < 0 or args.retry_max_seconds < 0:
        parser.error("retry delays must be non-negative")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_name = args.source_name or args.dataset
    checkpoint: dict | None = None
    emitted_rows = 0
    consecutive_failures = 0
    reconnects = 0

    while True:
        stream = None
        iterator = None
        try:
            stream = load_dataset(
                args.dataset,
                args.subset,
                split=args.split,
                revision=args.revision,
                streaming=True,
                cache_dir=str(cache_dir),
            )
            if checkpoint is not None:
                stream.load_state_dict(checkpoint)
            iterator = iter(stream)
            while True:
                row = next(iterator)
                selected = {args.text_field: row.get(args.text_field)}
                if args.license_field in row:
                    selected[args.license_field] = row.get(args.license_field)
                # ASCII escaping keeps every logical row on one UTF-8-safe line.
                sys.stdout.write(json.dumps(selected, ensure_ascii=True) + "\n")
                sys.stdout.flush()
                # This state points to the next example. Update it only after the
                # row has reached the parent process, so reconnects neither skip
                # nor repeat a document.
                checkpoint = stream.state_dict()
                emitted_rows += 1
                consecutive_failures = 0
        except StopIteration:
            return
        except Exception as error:
            if not is_transient_network_error(error):
                raise
            consecutive_failures += 1
            reconnects += 1
            if (
                consecutive_failures > args.max_retries
                or reconnects > args.max_reconnects
            ):
                raise RuntimeError(
                    f"Hugging Face stream {source_name} exceeded its retry limit "
                    f"after {emitted_rows:,} emitted rows and {reconnects} reconnects"
                ) from error
            delay = retry_delay(
                consecutive_failures,
                args.retry_initial_seconds,
                args.retry_max_seconds,
            )
            print(
                f"HF stream {source_name}: transient {type(error).__name__}; "
                f"reconnect {reconnects}/{args.max_reconnects}, "
                f"attempt {consecutive_failures}/{args.max_retries} in "
                f"{delay:g}s; resuming after {emitted_rows:,} rows",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        finally:
            if iterator is not None:
                close_iterator(iterator)
            del iterator, stream
            gc.collect()


if __name__ == "__main__":
    main()
