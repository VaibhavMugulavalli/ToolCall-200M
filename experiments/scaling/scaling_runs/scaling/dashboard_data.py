from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scaling.io_utils import atomic_write_json
from scaling.metrics import read_jsonl


def safe_run_filename(run_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-")
    return f"{slug or 'run'}.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_payload(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json") or {}
    summary = load_json(run_dir / "summary.json") or {}
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    return {
        "name": run_dir.name,
        "config": config,
        "summary": summary,
        "metrics": metrics,
    }


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return runs
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        config = load_json(run_dir / "config.json")
        if config is None:
            continue
        summary = load_json(run_dir / "summary.json") or {}
        training = config.get("training", {})
        model = config.get("model", {})
        runs.append(
            {
                "name": run_dir.name,
                "data_file": safe_run_filename(run_dir.name),
                "status": summary.get("status", "running"),
                "tokens_seen": summary.get("tokens_seen", 0),
                "target_tokens": training.get("target_tokens", 0),
                "parameter_count": summary.get("parameter_count"),
                "model": f"{model.get('num_layers', '?')}L/{model.get('hidden_size', '?')}D",
                "updated_at": summary.get("updated_at"),
            }
        )
    return runs


def export_static_dashboard(runs_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = list_runs(runs_dir)
    for run in runs:
        payload = build_run_payload(runs_dir / run["name"])
        atomic_write_json(output_dir / run["data_file"], payload)
    atomic_write_json(output_dir / "runs.json", {"runs": runs})
    return len(runs)
