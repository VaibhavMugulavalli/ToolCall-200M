from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def validate_hub_repo(repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi

    HfApi(token=token).repo_info(repo_id=repo_id, repo_type="model")


def remote_checkpoint_exists(
    repo_id: str,
    token: str,
    checkpoint_path: str,
) -> bool:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model")
    return f"{checkpoint_path}/latest" in files


def download_latest_checkpoint(
    repo_id: str,
    token: str,
    checkpoint_path: str,
    local_parent: str | Path,
) -> Path | None:
    """Download the atomic Hub resume snapshot into a shared local directory."""

    from huggingface_hub import snapshot_download

    if not remote_checkpoint_exists(repo_id, token, checkpoint_path):
        return None
    local_parent = Path(local_parent)
    local_parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns=[f"{checkpoint_path}/**"],
        local_dir=local_parent,
    )
    checkpoint_root = local_parent / checkpoint_path
    if not (checkpoint_root / "latest").is_file():
        raise RuntimeError("Hub snapshot downloaded without its resume/latest marker")
    return checkpoint_root


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _stage_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            _link_or_copy(path, destination / path.relative_to(source))


def upload_latest_checkpoint(
    *,
    repo_id: str,
    token: str,
    checkpoint_root: str | Path,
    checkpoint_tag: str,
    checkpoint_path: str,
    run_directory: str | Path,
    artifact_files: dict[str, str | Path],
    commit_message: str,
) -> str:
    """Atomically replace the Hub's restartable checkpoint and run metadata."""

    from huggingface_hub import HfApi

    checkpoint_root = Path(checkpoint_root)
    run_directory = Path(run_directory)
    tag_directory = checkpoint_root / checkpoint_tag
    if not tag_directory.is_dir() or not (checkpoint_root / "latest").is_file():
        raise FileNotFoundError(
            f"Incomplete local DeepSpeed checkpoint for tag {checkpoint_tag!r}"
        )

    with tempfile.TemporaryDirectory(prefix="hub-stage-", dir=run_directory) as temp:
        stage = Path(temp)
        remote_resume = stage / checkpoint_path
        _stage_tree(tag_directory, remote_resume / checkpoint_tag)
        _link_or_copy(checkpoint_root / "latest", remote_resume / "latest")

        remote_run = stage / "run"
        for name in (
            "config.json",
            "manifest.json",
            "model_config.json",
            "metrics.jsonl",
            "summary.json",
            "wandb_run_id.txt",
        ):
            path = run_directory / name
            if path.is_file():
                _link_or_copy(path, remote_run / name)
        tensorboard = run_directory / "tensorboard"
        if tensorboard.is_dir():
            _stage_tree(tensorboard, remote_run / "tensorboard")

        for remote_name, local_path in artifact_files.items():
            source = Path(local_path)
            if not source.is_file():
                raise FileNotFoundError(f"Required Hub artifact is missing: {source}")
            _link_or_copy(source, stage / "artifacts" / remote_name)

        result = HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=stage,
            path_in_repo="",
            delete_patterns=[
                f"{checkpoint_path}/*",
                f"{checkpoint_path}/**",
                "run/*",
                "run/**",
            ],
            commit_message=commit_message,
        )
    return str(result)
