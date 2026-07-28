from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last: int) -> None:
        self.directory = run_dir / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, state: dict[str, Any], step: int, label: str | None = None) -> Path:
        filename = f"checkpoint_{label}.pt" if label else f"checkpoint_step_{step:08d}.pt"
        path = self.directory / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        if label is None:
            self._prune_numbered_checkpoints()
        return path

    def latest(self) -> Path | None:
        final = self.directory / "checkpoint_final.pt"
        if final.exists():
            return final
        numbered = self._numbered_checkpoints()
        if numbered:
            return numbered[-1][1]
        return None

    def _numbered_checkpoints(self) -> list[tuple[int, Path]]:
        pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")
        checkpoints: list[tuple[int, Path]] = []
        for path in self.directory.glob("checkpoint_step_*.pt"):
            match = pattern.match(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        return sorted(checkpoints)

    def _prune_numbered_checkpoints(self) -> None:
        checkpoints = self._numbered_checkpoints()
        for _, path in checkpoints[: -self.keep_last]:
            path.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
