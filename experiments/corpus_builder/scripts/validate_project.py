#!/usr/bin/env python3
"""Dependency-free structural validation for the Stage 1 and Phase 2 package."""

from __future__ import annotations

import json
import math
import hashlib
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

    approved_path = root / "configs/toolcall_4b_approved.json"
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    exact_allocations = {
        "general_fineweb_edu": 2_632_512_585,
        "codeparrot_clean_permissive": 658_128_146,
        "structured_openapi": 400_783_507,
        "api_tool_docs": 197_553_288,
        "synthetic_tool_actions": 118_857_546,
    }
    assert approved["phase"] == 2
    assert approved["approval_status"] == "approved"
    assert approved["approved_allocations"] == exact_allocations
    assert sum(exact_allocations.values()) == required_stored
    assert approved["train_tokens"] == required_stored
    assert approved["optimizer_steps"] == steps
    assert approved["actual_prediction_tokens"] == actual_predictions
    assert approved["tokenizer_sha256"] == (
        "427581f5bf5a38ab9ce5b8900fead0780baae72cdef5e0b1761da6349302ff2b"
    )
    assert approved["audit_binding"]["stage1_config_sha256"] == (
        "e3c09361026adf650cf20de250f723e00bf612e91ef666bea01ad122d1150b59"
    )
    bundled_audit = root / "artifacts/approval/toolcall_4b_capacity_summary.json"
    assert hashlib.sha256(bundled_audit.read_bytes()).hexdigest() == (
        approved["audit_binding"]["capacity_summary_sha256"]
    )
    by_name = {source["name"]: source for source in approved["sources"]}
    assert set(by_name) == set(exact_allocations)
    assert all(
        int(by_name[name]["target_tokens"]) == target
        for name, target in exact_allocations.items()
    )

    notebook_paths = [
        root / "notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb",
        root / "notebooks/02_build_toolcall_4b_corpus_cpu_colab.ipynb",
    ]
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) >= 9
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell["source"]),
                    f"{notebook_path.name}:cell-{index}",
                    "exec",
                )

    for script in (
        "audit_source_capacity.py",
        "merge_capacity_reports.py",
        "build_scaling_data.py",
        "check_sources.py",
        "hf_stream_worker.py",
        "phase2_build_corpus.py",
        "create_phase2_notebook.py",
    ):
        path = root / "scripts" / script
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    print(f"PASS exact optimizer steps: {steps:,}")
    print(f"PASS actual prediction tokens: {actual_predictions:,}")
    print(f"PASS required stored tokens: {required_stored:,}")
    print(f"PASS finite sources: {sorted(finite)}")
    print(f"PASS approved source allocations: {sum(exact_allocations.values()):,}")
    for notebook_path in notebook_paths:
        print(f"PASS notebook code cells: {notebook_path}")
    print("Stage 1 + Phase 2 project validation passed.")


if __name__ == "__main__":
    main()
