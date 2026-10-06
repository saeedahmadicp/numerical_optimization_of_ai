"""Combinatorial problem library: every stored optimum is checked against independent oracles.

Oracles (none of them uses numopt's methods):
* knapsack: vectorized brute force over all 2ⁿ subsets, and SciPy's MILP solver (HiGHS);
* TSP: a NumPy Held–Karp written from the recursion, and a DFJ subtour-elimination MILP
  (SciPy/HiGHS, cuts added until the solution is a single cycle).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from numopt import problems
from numopt.core.rng import Rng
from numopt.problems.combinatorial import (
    KnapsackInstance,
    TspInstance,
    weakly_correlated_knapsack,
)

KNAPSACK_IDS = ("knapsack_10", "knapsack_20", "knapsack_greedy_trap")
TSP_IDS = ("tsp_circle_12", "tsp_random_15", "tsp_cities_20", "tsp_grid_16")


# --------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------


def knapsack_brute_force(values, weights, capacity) -> int:
    n = len(values)
    X = (np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1  # (2ⁿ, n)
    W = X @ np.asarray(weights, dtype=np.int64)
    V = X @ np.asarray(values, dtype=np.int64)
    return int(np.max(np.where(W <= capacity, V, -1)))


def knapsack_milp(values, weights, capacity) -> float:
    v = np.asarray(values, dtype=float)
    res = milp(
        -v,
        constraints=LinearConstraint(np.asarray(weights, dtype=float)[None, :], -np.inf, capacity),
        integrality=np.ones_like(v),
        bounds=Bounds(0, 1),
    )
    assert res.success
    return -float(res.fun)


def euclid(coords) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    return np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)


def held_karp_numpy(D: np.ndarray) -> float:
    n = len(D)
    m = n - 1
    dp = np.full((1 << m, m), np.inf)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, 1 << m):
        bits = [j for j in range(m) if mask >> j & 1]
        if len(bits) < 2:
            continue
        for j in bits:
            dp[mask, j] = np.min(dp[mask ^ (1 << j)] + D[1:, j + 1])
    return float(np.min(dp[-1] + D[1:, 0]))


def tsp_milp(D: np.ndarray) -> float:
    n = len(D)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cost = np.array([D[i, j] for i, j in edges])
    A = np.zeros((n, len(edges)))
    for e, (i, j) in enumerate(edges):
        A[i, e] = A[j, e] = 1.0
    rows, lo, hi = [A], [np.full(n, 2.0)], [np.full(n, 2.0)]
    for _ in range(100):
        res = milp(
            cost,
            constraints=LinearConstraint(np.vstack(rows), np.concatenate(lo), np.concatenate(hi)),  # pyright: ignore[reportArgumentType]
            integrality=np.ones(len(edges)),
            bounds=Bounds(0, 1),
            options={"mip_rel_gap": 0.0},
        )
        assert res.success
        x = np.round(res.x)
        G = np.zeros((n, n))
        for e, (i, j) in enumerate(edges):
            if x[e] > 0.5:
                G[i, j] = G[j, i] = 1.0
        n_comp, label = connected_components(csr_matrix(G), directed=False)
        if n_comp == 1:
            return float(res.fun)
        for c in range(n_comp):
            S = set(np.flatnonzero(label == c).tolist())
            rows.append(np.array([[1.0 if i in S and j in S else 0.0 for i, j in edges]]))
            lo.append(np.array([-np.inf]))
            hi.append(np.array([len(S) - 1.0]))
    raise AssertionError("subtour elimination did not converge")


# --------------------------------------------------------------------------------------
# Registry and serialization
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", KNAPSACK_IDS + TSP_IDS)
def test_registered_and_serializable(pid):
    p = problems.get(pid)
    assert problems.kind_of(pid) == "combinatorial"
    assert p.id == pid
    d = p.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["id"] == pid and d["description"]


def test_instance_types_and_shapes():
    for pid in KNAPSACK_IDS:
        p = problems.get(pid)
        assert isinstance(p, KnapsackInstance)
        assert len(p.values) == len(p.weights) == p.n
        assert all(isinstance(w, int) and w >= 1 for w in p.weights)
        assert all(isinstance(v, int) and v >= 0 for v in p.values)
    for pid in TSP_IDS:
        p = problems.get(pid)
        assert isinstance(p, TspInstance)
        assert np.asarray(p.coords).shape == (p.n, 2)
        assert len(set(p.coords)) == p.n, "cities must be distinct"
    assert problems.get("knapsack_10").n == 10 and problems.get("knapsack_20").n == 20
    assert [problems.get(t).n for t in TSP_IDS] == [12, 15, 20, 16]


def test_instances_are_frozen():
    p = problems.get("knapsack_10")
    with pytest.raises(AttributeError):
        p.capacity = 1  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Seeded generation follows the documented draw order
# --------------------------------------------------------------------------------------


def test_knapsack_generation_draw_order():
    for pid, n, seed in (("knapsack_10", 10, 7), ("knapsack_20", 20, 3)):
        rng = Rng(seed)
        w, v = [], []
        for _ in range(n):
            wi = 10 + min(int(rng.random() * 41), 40)
            vi = wi + min(int(rng.random() * 21), 20) - 5
            w.append(wi)
            v.append(vi)
        p = problems.get(pid)
        assert p.weights == tuple(w) and p.values == tuple(v)
        assert p.capacity == sum(w) // 2


def test_weakly_correlated_generator_ranges_match_its_docstring():
    # The documented class: wᵢ ∈ [10, 50] and an asymmetric offset vᵢ − wᵢ ∈ {−5, …, 15}
    # (not Pisinger's symmetric [wᵢ − R/10, wᵢ + R/10]); C = ⌊Σwᵢ / 2⌋.
    offsets: set[int] = set()
    weights: set[int] = set()
    for seed in range(20):
        values, w, capacity = weakly_correlated_knapsack(200, seed)
        assert capacity == sum(w) // 2
        weights.update(w)
        offsets.update(v - wi for v, wi in zip(values, w, strict=True))
    assert weights == set(range(10, 51))
    assert offsets == set(range(-5, 16))


def test_tsp_generation_draw_order():
    rng = Rng(15)
    pts = [(float(rng.integers(101)), float(rng.integers(101))) for _ in range(15)]
    assert problems.get("tsp_random_15").coords == tuple(pts)
    rng = Rng(21)
    centres = ((20, 25), (75, 20), (30, 75), (80, 70))
    pts = []
    for i in range(20):
        cx, cy = centres[i % 4]
        x = cx + rng.integers(21) - 10
        y = cy + rng.integers(21) - 10
        pts.append((float(x), float(y)))
    assert problems.get("tsp_cities_20").coords == tuple(pts)


def test_shuffled_geometric_instances_hold_the_right_point_sets():
    circle = problems.get("tsp_circle_12").coords
    c = np.asarray(circle) - 50.0
    np.testing.assert_allclose(np.hypot(c[:, 0], c[:, 1]), 40.0, rtol=1e-14)
    angles = np.sort(np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi))
    np.testing.assert_allclose(np.diff(angles), 2 * np.pi / 12, rtol=1e-12)
    grid = problems.get("tsp_grid_16").coords
    assert sorted(grid) == sorted((10.0 * (k % 4), 10.0 * (k // 4)) for k in range(16))
    # The shuffle hides the optimal order: the identity tour is not optimal.
    for pid in ("tsp_circle_12", "tsp_grid_16"):
        p = problems.get(pid)
        D = euclid(p.coords)
        identity = sum(D[k, (k + 1) % p.n] for k in range(p.n))
        assert identity > 1.2 * p.optimum_length


# --------------------------------------------------------------------------------------
# Stored optima versus independent oracles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", KNAPSACK_IDS)
def test_knapsack_optimum_brute_force_and_milp(pid):
    p = problems.get(pid)
    assert knapsack_brute_force(p.values, p.weights, p.capacity) == p.optimum_value
    assert knapsack_milp(p.values, p.weights, p.capacity) == pytest.approx(
        p.optimum_value, abs=1e-6
    )


def test_greedy_trap_claims():
    p = problems.get("knapsack_greedy_trap")
    # Ratio-greedy by hand: item 0 (ratio 3) then item 1 (ratio 1) fit; item 2 does not.
    assert p.values[0] / p.weights[0] > p.values[1] / p.weights[1] == p.values[2] / p.weights[2]
    greedy = p.values[0] + p.values[1]
    assert p.weights[0] + p.weights[1] <= p.capacity < p.weights[0] + p.weights[1] + p.weights[2]
    assert (greedy, max(p.values)) == (53, 50)
    assert p.optimum_value == 100 and greedy < 0.55 * p.optimum_value


def test_circle_optimum_is_the_polygon_perimeter():
    p = problems.get("tsp_circle_12")
    perimeter = 12 * 2 * 40.0 * math.sin(math.pi / 12)
    assert p.optimum_length == pytest.approx(perimeter, rel=1e-15)
    # NOTE: rtol 1e-12 — the oracle sums 12 rounded distances (each relative error ~1e-16).
    np.testing.assert_allclose(held_karp_numpy(euclid(p.coords)), perimeter, rtol=1e-12)


def test_random_15_optimum_held_karp_and_milp():
    p = problems.get("tsp_random_15")
    D = euclid(p.coords)
    np.testing.assert_allclose(held_karp_numpy(D), p.optimum_length, rtol=1e-12)
    # NOTE: rtol 1e-9 — HiGHS reports the objective with its own (feasibility-tolerance) rounding.
    np.testing.assert_allclose(tsp_milp(D), p.optimum_length, rtol=1e-9)


@pytest.mark.parametrize("pid", ("tsp_cities_20", "tsp_grid_16"))
def test_large_tsp_optimum_milp(pid):
    p = problems.get(pid)
    np.testing.assert_allclose(tsp_milp(euclid(p.coords)), p.optimum_length, rtol=1e-9)


def test_grid_optimum_is_sixteen_spacings():
    p = problems.get("tsp_grid_16")
    D = euclid(p.coords)
    off = D[~np.eye(16, dtype=bool)]
    assert off.min() == 10.0  # every edge ≥ spacing, so every tour ≥ 16 · spacing
    assert p.optimum_length == 160.0
