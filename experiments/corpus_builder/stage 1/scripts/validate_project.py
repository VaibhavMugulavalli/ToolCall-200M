#!/usr/bin/env python3
"""Dependency-free structural validation for the Stage 1 package."""

from __future__ import annotations

import json
import math
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/toolcall_4b.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    sequence_length = int(config["sequence_length"])
    global_tokens = int(config["global_prediction_tokens_per_step"])
    prediction_target = int(config["prediction_token_target"])
    steps = math.ceil(prediction_target / global_tokens)
    actual_predictions = steps * global_tokens
    sequences_per_step = global_tokens // sequence_length
    required_stored = steps * sequences_per_step * (sequence_length + 1)

    assert global_tokens % sequence_length == 0
    assert prediction_target == 4_000_000_000
    assert steps == 122_071
    assert actual_predictions == 4_000_022_528
    assert required_stored == int(config["train_tokens"]) == 4_007_835_072
    assert config["vocab_size"] == 32_000
    assert abs(sum(config["category_weights"].values()) - 1.0) < 1e-12
    assert abs(sum(source["weight"] for source in config["sources"]) - 1.0) < 1e-12
    assert all(source.get("revision") != "main" for source in config["sources"])
    assert config["openapi_revision"] != "main"

    finite = {source["name"] for source in config["sources"] if source.get("finite")}
    assert finite == {
        "structured_openapi",
        "api_tool_docs",
        "synthetic_tool_actions",
    }

    notebook_path = root / "notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook_path.name}:cell-{index}", "exec")

    for script in (
        "audit_source_capacity.py",
        "merge_capacity_reports.py",
        "build_scaling_data.py",
        "check_sources.py",
    ):
        path = root / "scripts" / script
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    print(f"PASS exact optimizer steps: {steps:,}")
    print(f"PASS actual prediction tokens: {actual_predictions:,}")
    print(f"PASS required stored tokens: {required_stored:,}")
    print(f"PASS finite sources: {sorted(finite)}")
    print(f"PASS notebook code cells: {notebook_path}")
    print("Stage 1 project validation passed.")


if __name__ == "__main__":
    main()
