from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class CorpusExhausted(RuntimeError):
    """Raised when a no-repeat packed corpus cannot satisfy another batch."""


@dataclass(frozen=True)
class CorpusDescription:
    root: str
    shard_count: int
    total_tokens: int
    total_bytes: int
    first_shard: str
    last_shard: str


class PackedTokenCorpus:
    """Read a deterministic global token stream from sorted uint16 .bin shards."""

    def __init__(self, root: str | Path, dtype: str = "uint16") -> None:
        self.root = Path(root).resolve()
        if dtype != "uint16":
            raise ValueError("Only uint16 token shards are supported")
        self.dtype = np.dtype(np.uint16)
        self.shards = self._discover_shards(self.root)
        self.shard_token_counts: list[int] = []
        self.shard_offsets: list[int] = [0]
        for shard in self.shards:
            byte_count = shard.stat().st_size
            if byte_count % self.dtype.itemsize != 0:
                raise ValueError(f"Shard has invalid byte size for uint16: {shard}")
            token_count = byte_count // self.dtype.itemsize
            if token_count == 0:
                raise ValueError(f"Shard is empty: {shard}")
            self.shard_token_counts.append(token_count)
            self.shard_offsets.append(self.shard_offsets[-1] + token_count)
        self.total_tokens = self.shard_offsets[-1]
        self._memmaps: dict[int, np.memmap] = {}

    @staticmethod
    def _discover_shards(root: Path) -> list[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Token corpus directory does not exist: {root}")
        candidates = sorted((root / "shards").glob("*.bin"))
        if not candidates:
            candidates = sorted(root.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(
                f"No .bin shards found under {root} or {root / 'shards'}"
            )
        return candidates

    def _memmap(self, shard_index: int) -> np.memmap:
        if shard_index not in self._memmaps:
            self._memmaps[shard_index] = np.memmap(
                self.shards[shard_index], mode="r", dtype=self.dtype
            )
        return self._memmaps[shard_index]

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count <= 0:
            raise ValueError("start must be non-negative and count must be positive")
        end = start + count
        if end > self.total_tokens:
            raise CorpusExhausted(
                f"Requested tokens [{start}, {end}), but corpus contains "
                f"{self.total_tokens:,} tokens. The experiment never wraps training data."
            )

        output = np.empty(count, dtype=self.dtype)
        output_cursor = 0
        global_cursor = start
        while global_cursor < end:
            shard_index = bisect.bisect_right(self.shard_offsets, global_cursor) - 1
            local_start = global_cursor - self.shard_offsets[shard_index]
            available = self.shard_token_counts[shard_index] - local_start
            take = min(available, end - global_cursor)
            output[output_cursor : output_cursor + take] = self._memmap(shard_index)[
                local_start : local_start + take
            ]
            output_cursor += take
            global_cursor += take
        return output

    def batch(
        self,
        cursor: int,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_tokens = batch_size * (sequence_length + 1)
        packed = self.read(cursor, source_tokens).astype(np.int64, copy=False)
        sequences = torch.from_numpy(packed.reshape(batch_size, sequence_length + 1))
        inputs = sequences[:, :-1].to(device=device, non_blocking=True)
        targets = sequences[:, 1:].to(device=device, non_blocking=True)
        return inputs, targets, cursor + source_tokens

    def maximum_batches(self, batch_size: int, sequence_length: int) -> int:
        return self.total_tokens // (batch_size * (sequence_length + 1))

    def describe(self) -> CorpusDescription:
        return CorpusDescription(
            root=str(self.root),
            shard_count=len(self.shards),
            total_tokens=self.total_tokens,
            total_bytes=self.total_tokens * self.dtype.itemsize,
            first_shard=self.shards[0].name,
            last_shard=self.shards[-1].name,
        )

    def fingerprint(self) -> str:
        """Hash manifests plus shard names/sizes so resume cannot silently change data."""
        digest = hashlib.sha256()
        for manifest_name in ("manifest.json", "manifest_shards.jsonl"):
            manifest = self.root / manifest_name
            if manifest.is_file():
                digest.update(manifest_name.encode("utf-8"))
                digest.update(manifest.read_bytes())
        for shard, token_count in zip(self.shards, self.shard_token_counts):
            digest.update(shard.name.encode("utf-8"))
            digest.update(str(token_count).encode("ascii"))
        return digest.hexdigest()
