"""Workload generation: exact-length prompts and the workload matrix.

We pin the input sequence length (ISL) and output sequence length (OSL) *exactly*
so that TPS/latency comparisons across configs are not polluted by variable prompt
or generation lengths. ISL is pinned by sending the prompt as a list of token IDs;
OSL is pinned with max_tokens + ignore_eos on the server side.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# The Moreh-style workload matrix (ISL/OSL in tokens) plus E-specific short-ISL cells.
DEFAULT_ISL_OSL = [
    (1024, 1024),   # 1K/1K   light
    (1024, 8192),   # 1K/8K   decode-heavy
    (8192, 1024),   # 8K/1K   prefill-heavy
    (8192, 8192),   # 8K/8K   heavy
]
# Config-E lives or dies at short ISL; add extra cells there.
E_EXTRA_ISL_OSL = [(128, 1024), (256, 1024), (512, 1024), (256, 4096)]

DEFAULT_CONCURRENCY = [1, 4, 8, 16, 32]


def build_prompt_token_ids(isl: int, vocab_size: int, seed: int = 0,
                           lo: int = 1000, hi: int | None = None) -> list[int]:
    """A deterministic pseudo-random token-id prompt of exactly `isl` tokens.

    We avoid the low ids (special/control tokens) by starting at `lo`. Random
    mid-vocab ids are fine for benchmarking: prefill compute depends on sequence
    length, not on the semantic content of the tokens.
    """
    hi = hi if hi is not None else max(lo + 1, vocab_size - 100)
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(isl)]


@dataclass
class Cell:
    """One (ISL, OSL, concurrency) benchmark cell."""
    isl: int
    osl: int
    concurrency: int

    @property
    def name(self) -> str:
        return f"isl{self.isl}_osl{self.osl}_c{self.concurrency}"


def build_matrix(isl_osl: list[tuple[int, int]] | None = None,
                 concurrency: list[int] | None = None) -> list[Cell]:
    isl_osl = isl_osl or DEFAULT_ISL_OSL
    concurrency = concurrency or DEFAULT_CONCURRENCY
    cells: list[Cell] = []
    for isl, osl in isl_osl:
        for c in concurrency:
            cells.append(Cell(isl, osl, c))
    return cells
