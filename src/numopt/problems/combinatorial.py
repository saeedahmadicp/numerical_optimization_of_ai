"""Combinatorial test problems: 0-1 knapsack and Euclidean traveling-salesman instances.

Two instance types live here (kind ``"combinatorial"``):

* :class:`KnapsackInstance` — the 0-1 knapsack problem (Martello & Toth 1990, eq. 2.1–2.3)

      maximize   Σᵢ vᵢ xᵢ
      subject to Σᵢ wᵢ xᵢ ≤ C,   xᵢ ∈ {0, 1},

  with positive integer weights wᵢ, non-negative integer values vᵢ and an integer capacity C.

* :class:`TspInstance` — the symmetric Euclidean TSP on n cities with coordinates ``coords``
  (n × 2): find a cyclic permutation π minimizing Σₖ ‖c_{π(k)} − c_{π(k+1 mod n)}‖₂.

Seeded instances draw only from :class:`numopt.core.rng.Rng` (Mulberry32), in the order the
builder functions document, so the TypeScript port regenerates identical data. Every
``optimum_value`` / ``optimum_length`` stored here is checked by the test-suite against an
independent oracle (brute force and SciPy's MILP solver), never against a numopt method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.rng import Rng
from ..core.types import to_jsonable
from .registry import factory


@dataclass(frozen=True)
class KnapsackInstance:
    """A 0-1 knapsack instance: maximize Σ vᵢxᵢ subject to Σ wᵢxᵢ ≤ C, xᵢ ∈ {0, 1}.

    Attributes:
        id: Registry id.
        name: Display name.
        values: Item values vᵢ ≥ 0 (integers).
        weights: Item weights wᵢ ≥ 1 (integers).
        capacity: Knapsack capacity C ≥ 0 (integer).
        optimum_value: The optimal objective value z*, when known.
        description: One or two sentences for the UI.
    """

    id: str
    name: str
    values: tuple[int, ...]
    weights: tuple[int, ...]
    capacity: int
    optimum_value: int | None = None
    description: str = ""

    @property
    def n(self) -> int:
        """Number of items."""
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "values": list(self.values),
                "weights": list(self.weights),
                "capacity": self.capacity,
                "optimum_value": self.optimum_value,
                "description": self.description,
            }
        )


@dataclass(frozen=True)
class TspInstance:
    """A symmetric Euclidean TSP instance on ``n = len(coords)`` cities.

    Attributes:
        id: Registry id.
        name: Display name.
        coords: City coordinates, one ``(x, y)`` pair per city (n × 2).
        optimum_length: Length of an optimal closed tour, when known.
        description: One or two sentences for the UI.
    """

    id: str
    name: str
    coords: tuple[tuple[float, float], ...]
    optimum_length: float | None = None
    description: str = ""

    @property
    def n(self) -> int:
        """Number of cities."""
        return len(self.coords)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "coords": [list(c) for c in self.coords],
                "optimum_length": self.optimum_length,
                "description": self.description,
            }
        )


# --------------------------------------------------------------------------------------
# Knapsack instances
# --------------------------------------------------------------------------------------


def weakly_correlated_knapsack(n: int, seed: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Weakly correlated items: a class modeled on Pisinger's, with an asymmetric value offset.

    Draw order, for i = 0, …, n−1: ``wᵢ = 10 + rng.integers(41)`` (so wᵢ ∈ [10, 50]), then
    ``vᵢ = wᵢ + rng.integers(21) − 5`` (so vᵢ ∈ [wᵢ − 5, wᵢ + 15]). Capacity C = ⌊Σwᵢ / 2⌋.
    Weak correlation between value and weight makes the ratio ordering informative but not
    decisive, so greedy is close to (but not at) the optimum and branch-and-bound must search.

    # NOTE: this is not Pisinger's weakly correlated benchmark class (Pisinger 2005, Computers
    # & Operations Research 32(9), 2271–2284; Kellerer, Pferschy & Pisinger 2004), which draws
    # wᵢ uniformly in [1, R] and vᵢ uniformly in the symmetric interval [wᵢ − R/10, wᵢ + R/10].
    # Here the offset vᵢ − wᵢ ∈ {−5, …, 15} is asymmetric (mean +5), so every item is worth on
    # average a little more than it weighs. The draw order is kept because the stored optima
    # (and the TypeScript port) depend on it.
    """
    rng = Rng(seed)
    weights: list[int] = []
    values: list[int] = []
    for _ in range(n):
        w = 10 + rng.integers(41)
        v = w + rng.integers(21) - 5
        weights.append(w)
        values.append(v)
    return tuple(values), tuple(weights), sum(weights) // 2


@factory("combinatorial")
def knapsack_10() -> KnapsackInstance:
    values, weights, capacity = weakly_correlated_knapsack(10, seed=7)
    return KnapsackInstance(
        id="knapsack_10",
        name="Knapsack, 10 items",
        values=values,
        weights=weights,
        capacity=capacity,
        optimum_value=173,
        description="10 weakly correlated items (seed 7), capacity = half the total weight. "
        "Greedy by value/weight ratio finds 172; the optimum is 173.",
    )


@factory("combinatorial")
def knapsack_20() -> KnapsackInstance:
    values, weights, capacity = weakly_correlated_knapsack(20, seed=3)
    return KnapsackInstance(
        id="knapsack_20",
        name="Knapsack, 20 items",
        values=values,
        weights=weights,
        capacity=capacity,
        optimum_value=407,
        description="20 weakly correlated items (seed 3), capacity = half the total weight. "
        "Greedy finds 397; the optimum is 407 (2²⁰ ≈ 10⁶ subsets for brute force).",
    )


@factory("combinatorial")
def knapsack_greedy_trap() -> KnapsackInstance:
    return KnapsackInstance(
        id="knapsack_greedy_trap",
        name="Greedy trap",
        values=(3, 50, 50),
        weights=(1, 50, 50),
        capacity=100,
        optimum_value=100,
        description="The small item has the best value/weight ratio, so greedy packs it first "
        "and then only one large item fits: greedy = 53, best single item = 50, optimum = 100. "
        "Even the 'best single item' fix stays near the ½-approximation bound.",
    )


# --------------------------------------------------------------------------------------
# TSP instances
# --------------------------------------------------------------------------------------


def _shuffle(points: list[tuple[float, float]], seed: int) -> tuple[tuple[float, float], ...]:
    """City i gets ``points[perm[i]]`` with ``perm = Rng(seed).permutation(n)``.

    Shuffling hides the optimal order, so the identity tour 0, 1, …, n−1 is not optimal.
    """
    perm = Rng(seed).permutation(len(points))
    return tuple(points[p] for p in perm)


@factory("combinatorial")
def tsp_circle_12() -> TspInstance:
    radius = 40.0
    points = [
        (
            50.0 + radius * math.cos(2.0 * math.pi * k / 12),
            50.0 + radius * math.sin(2.0 * math.pi * k / 12),
        )
        for k in range(12)
    ]
    return TspInstance(
        id="tsp_circle_12",
        name="12 cities on a circle",
        coords=_shuffle(points, seed=12),
        # Points in convex position: the optimal tour is the convex-hull order (any crossing
        # tour is shortened by 2-opt, and the only non-crossing tour is the hull), so the
        # optimum is the perimeter of the regular 12-gon: 12 · 2r·sin(π/12).
        optimum_length=24.0 * radius * math.sin(math.pi / 12),
        description="Regular 12-gon of radius 40 centered at (50, 50), city order shuffled "
        "(seed 12). The optimum is the polygon perimeter 960·sin(π/12) ≈ 248.466.",
    )


@factory("combinatorial")
def tsp_grid_16() -> TspInstance:
    spacing = 10.0
    points = [(spacing * (k % 4), spacing * (k // 4)) for k in range(16)]
    return TspInstance(
        id="tsp_grid_16",
        name="4 × 4 grid",
        coords=_shuffle(points, seed=16),
        # Every tour has 16 edges of length ≥ spacing, and a 4 × 4 grid (even side) has a
        # Hamiltonian cycle of unit grid steps, so the optimum is exactly 16 · spacing.
        optimum_length=16.0 * spacing,
        description="A 4 × 4 grid with spacing 10, city order shuffled (seed 16). Every edge "
        "has length ≥ 10 and a boustrophedon cycle uses only length-10 edges: optimum = 160. "
        "Many optimal tours exist (ties everywhere).",
    )


@factory("combinatorial")
def tsp_random_15() -> TspInstance:
    # Draw order: for each city i, x = rng.integers(101), then y = rng.integers(101).
    rng = Rng(15)
    coords = tuple((float(rng.integers(101)), float(rng.integers(101))) for _ in range(15))
    return TspInstance(
        id="tsp_random_15",
        name="15 random cities",
        coords=coords,
        # Verified in tests by Held–Karp (NumPy) and by a subtour-elimination MILP (SciPy).
        optimum_length=399.1256152125949,
        description="15 cities with integer coordinates drawn uniformly from {0, …, 100}² "
        "(seed 15). Optimum ≈ 399.126.",
    )


_CLUSTER_CENTERS = ((20, 25), (75, 20), (30, 75), (80, 70))


@factory("combinatorial")
def tsp_cities_20() -> TspInstance:
    # Draw order: city i belongs to cluster i mod 4; x = cx + rng.integers(21) − 10,
    # then y = cy + rng.integers(21) − 10.
    rng = Rng(21)
    coords: list[tuple[float, float]] = []
    for i in range(20):
        cx, cy = _CLUSTER_CENTERS[i % 4]
        coords.append((float(cx + rng.integers(21) - 10), float(cy + rng.integers(21) - 10)))
    return TspInstance(
        id="tsp_cities_20",
        name="20 clustered cities",
        coords=tuple(coords),
        # Verified in tests by a subtour-elimination MILP (SciPy).
        optimum_length=280.94443474166826,
        description="Four clusters of five cities (integer offsets in [−10, 10] around fixed "
        "centers, seed 21). Good tours visit each cluster once. Optimum ≈ 280.944.",
    )


__all__ = [
    "KnapsackInstance",
    "TspInstance",
    "knapsack_10",
    "knapsack_20",
    "knapsack_greedy_trap",
    "tsp_circle_12",
    "tsp_cities_20",
    "tsp_grid_16",
    "tsp_random_15",
    "weakly_correlated_knapsack",
]
