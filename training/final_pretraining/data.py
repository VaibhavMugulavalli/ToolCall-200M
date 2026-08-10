from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import torch


class CorpusExhausted(RuntimeError):
    """Raised when a no-repeat token stream cannot provide another batch."""


def discover_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    root = root.resolve()
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    return sorted(matches, key=lambda path: path.relative_to(root).as_posix())


def discover_one(root: Path, patterns: Iterable[str], label: str) -> Path:
    matches = discover_files(root, patterns)
    if len(matches) != 1:
        rendered = ", ".join(str(path.relative_to(root)) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {label} under {root}, found {len(matches)}: {rendered}"
        )
    return matches[0]


@dataclass(frozen=True)
class CorpusDescription:
    root: str
    patterns: tuple[str, ...]
    shard_count: int
    total_tokens: int
    total_bytes: int
    first_shard: str
    last_shard: str
    fingerprint: str


class ShardedTokenCorpus:
    """Deterministic, no-repeat reader over sorted raw uint16 token shards."""

    def __init__(
        self,
        root: str | Path,
        patterns: Iterable[str],
        dtype: str = "uint16",
    ) -> None:
        self.root = Path(root).resolve()
        self.patterns = tuple(patterns)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if dtype != "uint16":
            raise ValueError("Only raw uint16 token shards are supported")
        self.dtype = np.dtype(np.uint16)
        self.shards = discover_files(self.root, self.patterns)
        if not self.shards:
            raise FileNotFoundError(
                f"No shards matched {self.patterns!r} below {self.root}"
            )

        self.shard_token_counts: list[int] = []
        self.shard_offsets: list[int] = [0]
        for shard in self.shards:
            byte_count = shard.stat().st_size
            if byte_count <= 0 or byte_count % self.dtype.itemsize:
                raise ValueError(f"Invalid uint16 shard size: {shard} ({byte_count} bytes)")
            token_count = byte_count // self.dtype.itemsize
            self.shard_token_counts.append(token_count)
            self.shard_offsets.append(self.shard_offsets[-1] + token_count)
        self.total_tokens = self.shard_offsets[-1]
        self._memmaps: dict[int, np.memmap] = {}
        self._fingerprint = self._calculate_fingerprint()

    def _calculate_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.dtype).encode("ascii"))
        for shard, token_count in zip(self.shards, self.shard_token_counts):
            digest.update(shard.relative_to(self.root).as_posix().encode("utf-8"))
            digest.update(str(token_count).encode("ascii"))
        return digest.hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def describe(self) -> CorpusDescription:
        return CorpusDescription(
            root=str(self.root),
            patterns=self.patterns,
            shard_count=len(self.shards),
            total_tokens=self.total_tokens,
            total_bytes=self.total_tokens * self.dtype.itemsize,
            first_shard=self.shards[0].relative_to(self.root).as_posix(),
            last_shard=self.shards[-1].relative_to(self.root).as_posix(),
            fingerprint=self.fingerprint(),
        )

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
                f"Requested [{start:,}, {end:,}), but the split contains "
                f"{self.total_tokens:,} tokens. Training data never wraps."
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
        device: "torch.device",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        source_tokens = batch_size * (sequence_length + 1)
        packed = self.read(cursor, source_tokens).astype(np.int64, copy=False)
        sequences = torch.from_numpy(packed.reshape(batch_size, sequence_length + 1))
        inputs = sequences[:, :-1].to(device=device, non_blocking=True)
        targets = sequences[:, 1:].to(device=device, non_blocking=True)
        return inputs, targets

    def maximum_local_batches(self, batch_size: int, sequence_length: int) -> int:
        return self.total_tokens // (batch_size * (sequence_length + 1))


def rank_batch_start(
    global_cursor: int,
    rank: int,
    local_batch_size: int,
    sequence_length: int,
) -> int:
    """Return a rank-disjoint slice within one global distributed micro-batch."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    local_source_tokens = local_batch_size * (sequence_length + 1)
    return global_cursor + rank * local_source_tokens
