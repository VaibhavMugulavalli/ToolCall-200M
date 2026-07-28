from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scaling.io_utils import utc_now


class JsonlMetricsLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        payload = {"timestamp": utc_now(), **record}
        line = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            # A process can be interrupted between append and flush. Ignore only
            # an incomplete final record; malformed earlier records are real errors.
            if line_number == len(lines):
                break
            raise
    return records
