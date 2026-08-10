#!/usr/bin/env python3
"""Merge per-source capacity audits and calculate provisional 4B allocations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_scaling_data import atomic_json, sha256_file, utc_now, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/toolcall_4b.json")
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    expected_config_sha = sha256_file(config_path)

    reports: dict[str, dict[str, Any]] = {}
    for source in config["sources"]:
        if not source.get("finite"):
            continue
        path = reports_dir / f"{source['name']}.capacity.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing capacity report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"Incomplete capacity report: {path}")
        if report.get("config_sha256") != expected_config_sha:
            raise RuntimeError(f"Report was made with a different config: {path}")
        reports[source["name"]] = report

    allocations: dict[str, int] = {}
    total_shortfall = 0
    for source in config["sources"]:
        cap = round(int(config["train_tokens"]) * float(source["weight"]))
        if source.get("finite"):
            usable = int(reports[source["name"]]["usable_unique_tokens"])
            allocation = min(cap, usable)
            total_shortfall += cap - allocation
        else:
            allocation = cap
        allocations[source["name"]] = allocation

    fill_weights = config["capacity_policy"]["shortfall_fill_sources"]
    if abs(sum(float(value) for value in fill_weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("Shortfall fill weights must sum to 1.0")
    distributed = 0
    fill_names = list(fill_weights)
    for index, name in enumerate(fill_names):
        if name not in allocations:
            raise RuntimeError(f"Unknown shortfall fill source: {name}")
        if index == len(fill_names) - 1:
            addition = total_shortfall - distributed
        else:
            addition = round(total_shortfall * float(fill_weights[name]))
            distributed += addition
        allocations[name] += addition

    difference = int(config["train_tokens"]) - sum(allocations.values())
    if difference:
        allocations[fill_names[0]] += difference
    if sum(allocations.values()) != int(config["train_tokens"]):
        raise RuntimeError("Provisional allocations do not sum to train_tokens")

    rows = []
    by_name = {source["name"]: source for source in config["sources"]}
    for name, allocated in allocations.items():
        source = by_name[name]
        cap = round(int(config["train_tokens"]) * float(source["weight"]))
        rows.append(
            {
                "source": name,
                "category": source["category"],
                "finite": bool(source.get("finite")),
                "configured_cap_tokens": cap,
                "audited_unique_tokens": (
                    int(reports[name]["usable_unique_tokens"])
                    if name in reports
                    else None
                ),
                "provisional_allocation_tokens": allocated,
                "provisional_fraction": allocated / int(config["train_tokens"]),
            }
        )

    summary = {
        "format_version": 1,
        "status": "awaiting_human_approval",
        "created_at": utc_now(),
        "dataset_name": config["dataset_name"],
        "config_sha256": expected_config_sha,
        "prediction_token_target": config["prediction_token_target"],
        "stored_training_token_target": config["train_tokens"],
        "finite_source_shortfall_tokens": total_shortfall,
        "shortfall_fill_policy": fill_weights,
        "provisional_allocations": rows,
        "finite_source_reports": reports,
        "gate": (
            "Review the audited capacities, near-duplicate estimates, and license notes. "
            "Do not launch Stage 2 until these allocations are explicitly frozen."
        ),
    }
    atomic_json(output, summary)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
