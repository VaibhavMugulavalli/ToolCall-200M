#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and consolidate the latest ZeRO-2 checkpoint"
    )
    parser.add_argument(
        "--hub-repo-id", default=os.environ.get("HF_REPO_ID"), required=False
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not args.hub_repo_id or not token:
        raise RuntimeError("HF_REPO_ID/--hub-repo-id and HF_TOKEN are required")

    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    from huggingface_hub import HfApi, snapshot_download

    download_root = args.output_dir.parent / f"{args.output_dir.name}-download"
    snapshot_download(
        repo_id=args.hub_repo_id,
        repo_type="model",
        token=token,
        allow_patterns=["resume/**", "artifacts/**"],
        local_dir=download_root,
    )
    resume_root = download_root / "resume"
    tag = (resume_root / "latest").read_text(encoding="utf-8").strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    convert_zero_checkpoint_to_fp32_state_dict(
        str(resume_root),
        str(args.output_dir),
        max_shard_size="2GB",
        safe_serialization=True,
        tag=tag,
    )
    for source_name, destination_name in (
        ("model_config.json", "model_config.json"),
        ("toolcall_spm_32k.model", "toolcall_spm_32k.model"),
    ):
        shutil.copy2(
            download_root / "artifacts" / source_name,
            args.output_dir / destination_name,
        )
    manifest = {
        "source_repo": args.hub_repo_id,
        "source_checkpoint_tag": tag,
        "format": "consolidated fp32 safetensors",
        "loader": "training.final_pretraining.model.load_exported_model",
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.upload:
        result = HfApi(token=token).upload_folder(
            repo_id=args.hub_repo_id,
            repo_type="model",
            folder_path=args.output_dir,
            path_in_repo=f"exports/{tag}",
            commit_message=f"Consolidated model export for {tag}",
        )
        print(f"Uploaded export: {result}")
    print(f"Consolidated checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
