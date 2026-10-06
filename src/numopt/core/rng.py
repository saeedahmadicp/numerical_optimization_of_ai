"""A tiny, portable pseudo-random generator (Mulberry32) shared with the web app.

NumPy's PCG64 cannot be reproduced in the browser, so every stochastic method in numopt
draws from :class:`Rng` instead. ``web/src/core/rng.ts`` implements the identical
bit-level algorithm, so the same seed gives the same stream of uniforms in both languages.
Mulberry32 is not cryptographic and has a 2^32 period, which is ample for demonstrations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")
_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & _MASK


class Rng:
    """Seeded generator: ``random()`` ∈ [0, 1) with 32-bit resolution."""

    def __init__(self, seed: int = 0) -> None:
        self._state = int(seed) & _MASK

    def random(self) -> float:
        self._state = (self._state + 0x6D2B79F5) & _MASK
        t = self._state
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t + _imul(t ^ (t >> 7), t | 61)) & _MASK ^ t
        return ((t ^ (t >> 14)) & _MASK) / 4294967296.0

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * self.random()

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Box–Muller (one variate per call; the second variate is discarded for portability)."""
        u1 = 1.0 - self.random()  # (0, 1]
        u2 = self.random()
        return mean + std * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def integers(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        return min(int(self.random() * n), n - 1)

    def permutation(self, n: int) -> list[int]:
        """Fisher–Yates shuffle of range(n)."""
        out = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.integers(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def choice(self, items: Sequence[T]) -> T:
        return items[self.integers(len(items))]
