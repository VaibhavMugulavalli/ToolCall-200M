#!/usr/bin/env python3
"""Generate the checked-in CPU Colab notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


cells = [
    markdown(
        """
# ToolCall-200M — Stage 1 capacity audit for the 4B corpus

Run this notebook in a **CPU Colab runtime**. It pins the same frozen 32k
SentencePiece tokenizer and source revisions, audits every finite
OpenAPI-derived source, and calculates provisional 4B allocations.

This stage deliberately does **not** generate the 8 GB final shard set. The
finite-source report is a safety gate: it prevents filling a quota by repeating
templates after a source is exhausted.

Completed per-source reports are stored on Google Drive. If Colab disconnects,
rerun the notebook; completed reports are verified and skipped.
"""
    ),
    markdown(
        """
## 1. Upload and extract the project

Upload `ToolCall-4B-Corpus-CPU-Colab.zip`. This notebook expects exactly one
project archive and refuses ambiguous layouts.
"""
    ),
    code(
        """
from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

uploaded = files.upload()
archives = [Path(name) for name in uploaded if name.lower().endswith(".zip")]
if len(archives) != 1:
    raise RuntimeError(f"Upload exactly one project ZIP, found: {archives}")

archive = archives[0]
workspace = Path("/content/toolcall_4b_workspace")
if workspace.exists():
    shutil.rmtree(workspace)
workspace.mkdir(parents=True)

with zipfile.ZipFile(archive) as zf:
    for info in zf.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
    zf.extractall(workspace)

candidates = [
    path.parent
    for path in workspace.rglob("configs/toolcall_4b.json")
    if (path.parent / "scripts/audit_source_capacity.py").is_file()
]
if len(candidates) != 1:
    raise RuntimeError(f"Expected one ToolCall 4B project, found {len(candidates)}")
PROJECT = candidates[0]
os.chdir(PROJECT)
print("Project:", PROJECT)
"""
    ),
    markdown(
        """
## 2. Mount persistent storage

The computation remains CPU-only. Drive is used only to preserve the tokenizer
and small audit reports across ephemeral Colab sessions. The large Hugging Face
and Git caches remain under local `/content` for speed.
"""
    ),
    code(
        """
from google.colab import drive

drive.mount("/content/drive")
PERSIST_ROOT = Path("/content/drive/MyDrive/ToolCall-200M/toolcall_4b_corpus")
INPUT_ROOT = PERSIST_ROOT / "inputs"
AUDIT_ROOT = PERSIST_ROOT / "capacity_audit"
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

free_gib = shutil.disk_usage(PERSIST_ROOT).free / 1024**3
print("Persistent root:", PERSIST_ROOT)
print(f"Drive free space: {free_gib:.1f} GiB")
if free_gib < 10:
    print("WARNING: Stage 2 will need at least about 10 GiB free.")
"""
    ),
    markdown(
        """
## 3. Create the pinned CPU environment

Stop if dependency validation fails. A GPU is neither requested nor used.
"""
    ),
    code(
        """
ENV_ROOT = Path("/content/toolcall_4b_data_env")
READY = ENV_ROOT / "READY"
requirements_digest = hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest()

if ENV_ROOT.exists() and (
    not READY.is_file() or READY.read_text().strip() != requirements_digest
):
    shutil.rmtree(ENV_ROOT)

if not (ENV_ROOT / "bin/python").is_file():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "virtualenv==20.29.3"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "virtualenv", "--no-download", str(ENV_ROOT)],
        check=True,
    )

DATA_PYTHON = str(ENV_ROOT / "bin/python")
if not READY.is_file():
    subprocess.run(
        [DATA_PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=True,
    )
    subprocess.run(
        [DATA_PYTHON, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"],
        cwd=PROJECT,
        check=True,
    )
    subprocess.run([DATA_PYTHON, "scripts/validate_data_runtime.py"], cwd=PROJECT, check=True)
    READY.write_text(requirements_digest + "\\n")
else:
    subprocess.run([DATA_PYTHON, "scripts/validate_data_runtime.py"], cwd=PROJECT, check=True)

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("Using HF_TOKEN from Colab secrets.")
else:
    print("No HF_TOKEN found. Public source checks still work with lower rate limits.")
"""
    ),
    markdown(
        """
## 4. Install the frozen tokenizer once

On the first session, upload `toolcall_spm_32k.model`. It is copied to Drive
and reused in later sessions. Do not substitute or retrain the tokenizer.
"""
    ),
    code(
        """
PERSISTED_TOKENIZER = INPUT_ROOT / "toolcall_spm_32k.model"
if not PERSISTED_TOKENIZER.is_file():
    print("Upload toolcall_spm_32k.model")
    uploaded = files.upload()
    models = [Path(name) for name in uploaded if name.endswith(".model")]
    if len(models) != 1:
        raise RuntimeError(f"Upload exactly one tokenizer model, found: {models}")
    shutil.copy2(models[0], PERSISTED_TOKENIZER)

PROJECT_TOKENIZER = PROJECT / "artifacts/tokenizer/toolcall_spm_32k.model"
PROJECT_TOKENIZER.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(PERSISTED_TOKENIZER, PROJECT_TOKENIZER)
tokenizer_sha = hashlib.sha256(PROJECT_TOKENIZER.read_bytes()).hexdigest()
print("Tokenizer:", PROJECT_TOKENIZER)
print("SHA-256:", tokenizer_sha)
"""
    ),
    markdown(
        """
## 5. Validate configuration, tokenizer, and source access

The configuration pins FineWeb-Edu, CodeParrot Clean, and APIs-guru revisions.
This cell reads only a probe document from each source.
"""
    ),
    code(
        """
CONFIG = PROJECT / "configs/toolcall_4b.json"
CACHE_ROOT = Path("/content/toolcall_4b_cache")

subprocess.run(
    [
        DATA_PYTHON,
        "scripts/check_sources.py",
        "--config",
        str(CONFIG),
        "--cache-dir",
        str(CACHE_ROOT),
    ],
    cwd=PROJECT,
    check=True,
)
print("Source preflight passed.")
"""
    ),
    markdown(
        """
## 6. Audit each finite source

Each source writes its own completed JSON report. This may take a while because
all pinned OpenAPI definitions are parsed and encoded. If the runtime ends,
rerun the notebook: reports that match the tokenizer, config, and source commit
are skipped.
"""
    ),
    code(
        """
FINITE_SOURCES = [
    "structured_openapi",
    "api_tool_docs",
    "synthetic_tool_actions",
]

for source_name in FINITE_SOURCES:
    print("\\n" + "=" * 80)
    print("AUDITING:", source_name)
    subprocess.run(
        [
            DATA_PYTHON,
            "scripts/audit_source_capacity.py",
            "--config",
            str(CONFIG),
            "--tokenizer",
            str(PROJECT_TOKENIZER),
            "--output-dir",
            str(AUDIT_ROOT),
            "--cache-dir",
            str(CACHE_ROOT),
            "--source",
            source_name,
            "--skip-existing",
        ],
        cwd=PROJECT,
        check=True,
    )

print("All finite-source audits are complete.")
"""
    ),
    markdown(
        """
## 7. Merge reports and calculate provisional allocations

This applies the configured fallback rule to any finite-source shortfall:
80% goes to FineWeb-Edu and 20% to permissively licensed code. The result is
still marked `awaiting_human_approval`; it is not yet authorization to build.
"""
    ),
    code(
        """
SUMMARY_JSON = AUDIT_ROOT / "toolcall_4b_capacity_summary.json"
subprocess.run(
    [
        DATA_PYTHON,
        "scripts/merge_capacity_reports.py",
        "--config",
        str(CONFIG),
        "--reports-dir",
        str(AUDIT_ROOT),
        "--output",
        str(SUMMARY_JSON),
    ],
    cwd=PROJECT,
    check=True,
)

summary = json.loads(SUMMARY_JSON.read_text())
print("Status:", summary["status"])
print("Finite-source shortfall:", f"{summary['finite_source_shortfall_tokens']:,}")
"""
    ),
    code(
        """
import pandas as pd

table = pd.DataFrame(summary["provisional_allocations"])
display(table)

for name, report in summary["finite_source_reports"].items():
    near = report["near_duplicate_sample"]
    print(
        f"{name}: unique={report['usable_unique_tokens']:,}, "
        f"cap={report['target_stored_tokens']:,}, "
        f"shortfall={report['shortfall_tokens']:,}, "
        f"sampled-near-duplicate-rate={near['estimated_fraction_in_sample']:.2%}"
    )
"""
    ),
    markdown(
        """
## 8. Download the audit summary

Download both files and send them back for review. **Stop here.** Stage 2 will
freeze the approved allocations and build the resumable 4.008B-token shard set.
"""
    ),
    code(
        """
SUMMARY_CSV = SUMMARY_JSON.with_suffix(".csv")
print("Downloading:", SUMMARY_JSON.name)
files.download(str(SUMMARY_JSON))
print("Downloading:", SUMMARY_CSV.name)
files.download(str(SUMMARY_CSV))

print("\\nSTOP CONDITION: audit files downloaded; do not launch corpus generation yet.")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "01_audit_toolcall_4b_sources_cpu_colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).resolve().parents[1] / "notebooks/01_audit_toolcall_4b_sources_cpu_colab.ipynb"
destination.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(destination)
