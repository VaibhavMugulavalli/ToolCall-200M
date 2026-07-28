from __future__ import annotations

import math
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scaling.io_utils import atomic_write_json, utc_now


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    non_embedding = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "token_embedding" not in name
    )
    return total, non_embedding


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def system_information(device: torch.device, project_root: Path) -> dict[str, Any]:
    information: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "git_commit": git_commit(project_root),
    }
    if device.type == "cuda":
        information.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return information
