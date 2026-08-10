#!/usr/bin/env python3
"""Generate the standalone CPU-Colab notebook for Phase 2."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "notebooks" / "02_build_toolcall_4b_corpus_cpu_colab.ipynb"
    cells = [
        markdown(
            """# ToolCall-200M — Phase 2: build the approved 4B corpus on CPU Colab

This notebook turns the approved Stage 1 allocation into restart-safe `uint16`
shards. CPU performs all tokenization. Google Drive stores only committed
transactions and the final Kaggle-ready corpus.

Run on a **CPU** runtime. Rerun the notebook after any disconnect; completed
stages are verified and skipped. Never delete `progress.json` or individual
staged files to force a resume.
"""
        ),
        code(
            """# One Colab session budget. The active source pauses at its next safe shard checkpoint.
SESSION_MINUTES = 300
MINIMUM_DRIVE_FREE_GIB = 9.5

print("Session budget:", SESSION_MINUTES, "minutes")
"""
        ),
        code(
            """from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
        zf.extractall(destination)


print("Upload ToolCall-4B-Corpus-CPU-Colab.zip")
uploaded = files.upload()
archives = [Path(name) for name in uploaded if zipfile.is_zipfile(name)]
if len(archives) != 1:
    raise RuntimeError(f"Upload exactly one project ZIP, found: {archives}")

unpack = Path(tempfile.mkdtemp(prefix="toolcall_4b_project_", dir="/content"))
safe_extract_zip(archives[0], unpack)
candidates = sorted({
    path.parent.parent
    for path in unpack.rglob("configs/toolcall_4b_approved.json")
    if (path.parent.parent / "scripts/phase2_build_corpus.py").is_file()
})
if len(candidates) != 1:
    raise RuntimeError(f"Expected one ToolCall 4B Phase 2 project, found {len(candidates)}")
PROJECT = candidates[0]
os.chdir(PROJECT)
print("Project:", PROJECT)
"""
        ),
        code(
            """# Install into an isolated environment so Colab's preinstalled packages stay untouched.
VENV = Path("/content/toolcall_4b_venv")
if not (VENV / "bin/python").is_file():
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
DATA_PYTHON = str(VENV / "bin/python")
subprocess.run(
    [DATA_PYTHON, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt")],
    check=True,
)
subprocess.run([DATA_PYTHON, "scripts/validate_project.py"], cwd=PROJECT, check=True)
print("Environment ready:", DATA_PYTHON)
"""
        ),
        code(
            """from google.colab import drive

drive.mount("/content/drive")
PERSIST_ROOT = Path("/content/drive/MyDrive/ToolCall-200M/toolcall_4b_corpus")
INPUT_ROOT = PERSIST_ROOT / "inputs"
AUDIT_ROOT = PERSIST_ROOT / "capacity_audit"
WORKSPACE = PERSIST_ROOT / "phase2"
FINAL_ROOT = WORKSPACE / "final" / "toolcall_4b"
CACHE_ROOT = Path("/content/toolcall_4b_cache")
LOCAL_ROOT = Path("/content/toolcall_4b_phase2")
for path in (INPUT_ROOT, AUDIT_ROOT, WORKSPACE, CACHE_ROOT, LOCAL_ROOT):
    path.mkdir(parents=True, exist_ok=True)

free_gib = shutil.disk_usage(PERSIST_ROOT).free / 1024**3
print("Persistent root:", PERSIST_ROOT)
print(f"Drive free space: {free_gib:.1f} GiB")
if free_gib < MINIMUM_DRIVE_FREE_GIB:
    raise RuntimeError(
        f"Phase 2 needs at least {MINIMUM_DRIVE_FREE_GIB:.1f} GiB free; "
        f"only {free_gib:.1f} GiB is available."
    )
"""
        ),
        code(
            """TOKENIZER = INPUT_ROOT / "toolcall_spm_32k.model"
AUDIT_SUMMARY = AUDIT_ROOT / "toolcall_4b_capacity_summary.json"
CONFIG = PROJECT / "configs/toolcall_4b_approved.json"
BUNDLED_AUDIT = PROJECT / "artifacts/approval/toolcall_4b_capacity_summary.json"

if not AUDIT_SUMMARY.is_file():
    shutil.copy2(BUNDLED_AUDIT, AUDIT_SUMMARY)
    print("Installed the approved Stage 1 summary from the project bundle.")

missing = []
if not TOKENIZER.is_file():
    missing.append("toolcall_spm_32k.model")

if missing:
    print("Upload missing approved inputs:", missing)
    uploaded = files.upload()
    for name in list(uploaded):
        source = Path(name)
        if source.name == "toolcall_spm_32k.model":
            shutil.copy2(source, TOKENIZER)

if not TOKENIZER.is_file() or not AUDIT_SUMMARY.is_file():
    raise RuntimeError("The frozen tokenizer and Stage 1 capacity summary are required")

config = json.loads(CONFIG.read_text())
print("Tokenizer SHA-256:", hashlib.sha256(TOKENIZER.read_bytes()).hexdigest())
print("Audit SHA-256:", hashlib.sha256(AUDIT_SUMMARY.read_bytes()).hexdigest())
print("Approved train tokens:", f"{config['train_tokens']:,}")
"""
        ),
        code(
            """BASE_COMMAND = [
    DATA_PYTHON,
    "scripts/phase2_build_corpus.py",
    "--config", str(CONFIG),
    "--tokenizer", str(TOKENIZER),
    "--audit-summary", str(AUDIT_SUMMARY),
    "--workspace", str(WORKSPACE),
    "--final-root", str(FINAL_ROOT),
    "--cache-dir", str(CACHE_ROOT),
    "--local-root", str(LOCAL_ROOT),
]

subprocess.run(BASE_COMMAND[:2] + ["status"] + BASE_COMMAND[2:], cwd=PROJECT, check=True)
"""
        ),
        markdown(
            """## Build the next incomplete stages

This cell uses the remaining session budget automatically. A stage commits
roughly every 10M tokens. If the session limit is reached, it exits cleanly;
start a new CPU runtime and run the notebook again.
"""
        ),
        code(
            """stage_order = [
    ("validation_structured", "build-structured-validation", None),
    ("structured_openapi", "build-source", "structured_openapi"),
    ("api_tool_docs", "build-source", "api_tool_docs"),
    ("synthetic_tool_actions", "build-source", "synthetic_tool_actions"),
    ("validation_general", "build-general-validation", None),
    ("general_fineweb_edu", "build-source", "general_fineweb_edu"),
    ("codeparrot_clean_permissive", "build-source", "codeparrot_clean_permissive"),
]

deadline = time.monotonic() + SESSION_MINUTES * 60
for stage_name, command_name, source_name in stage_order:
    if (WORKSPACE / "staging" / stage_name / "COMPLETE").is_file():
        print("SKIP complete:", stage_name)
        continue
    remaining_minutes = (deadline - time.monotonic()) / 60
    if remaining_minutes < 2:
        print("Session budget exhausted before:", stage_name)
        break
    command = BASE_COMMAND[:2] + [command_name] + BASE_COMMAND[2:]
    command += ["--max-session-minutes", str(max(1, remaining_minutes - 1))]
    if source_name:
        command += ["--source", source_name]
    print("\\nBUILDING:", stage_name)
    result = subprocess.run(command, cwd=PROJECT)
    if result.returncode != 0:
        raise RuntimeError(f"Phase 2 stage failed: {stage_name}")
    if not (WORKSPACE / "staging" / stage_name / "COMPLETE").is_file():
        print("Stage paused safely; continue in the next CPU Colab session.")
        break

subprocess.run(BASE_COMMAND[:2] + ["status"] + BASE_COMMAND[2:], cwd=PROJECT, check=True)
"""
        ),
        code(
            """# Assemble automatically only after every source and validation stage is complete.
required = [name for name, _command, _source in stage_order]
all_complete = all((WORKSPACE / "staging" / name / "COMPLETE").is_file() for name in required)

if all_complete:
    subprocess.run(BASE_COMMAND[:2] + ["assemble"] + BASE_COMMAND[2:], cwd=PROJECT, check=True)
    subprocess.run(
        BASE_COMMAND[:2] + ["verify"] + BASE_COMMAND[2:] + ["--checksums"],
        cwd=PROJECT,
        check=True,
    )
    print("Kaggle-ready corpus:", FINAL_ROOT)
else:
    print("Phase 2 is not complete yet. Start another CPU session and rerun this notebook.")
"""
        ),
        markdown(
            """## Completion condition

The corpus is ready only when this file exists:

`MyDrive/ToolCall-200M/toolcall_4b_corpus/phase2/final/toolcall_4b/COMPLETE`

The final checksum pass must also print exact totals for train (4,007,835,072),
general validation (5,000,000), and structured validation (2,000,000). The
finalizer moves staged shards into the final directory, so it does not require
a second 8 GB copy.
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
