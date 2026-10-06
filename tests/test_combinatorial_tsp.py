"""TSP methods: construction, local search, metaheuristics and the exact Held–Karp DP.

Oracles: brute force over all (n−1)! tours (n ≤ 8), SciPy's ``cdist`` for distances, the
analytic optimum of points in convex position, and exhaustive neighbourhood scans for
2-optimality / Or-optimality.
"""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.distance import cdist

import numopt
from numopt import problems
from numopt.combinatorial import tsp
from numopt.combinatorial.tsp import distance_matrix, order_crossover, tour_length
from numopt.core.rng import Rng
from numopt.problems.combinatorial import TspInstance

ALL = (
    "tsp_nearest_neighbor",
    "tsp_two_opt",
    "tsp_or_opt",
    "tsp_simulated_annealing",
    "tsp_genetic",
    "tsp_ant_colony",
    "tsp_held_karp",
)
STOCHASTIC = ("tsp_simulated_annealing", "tsp_genetic", "tsp_ant_colony")
TSP_IDS = ("tsp_circle_12", "tsp_random_15", "tsp_cities_20", "tsp_grid_16")


def num(value: float | None) -> float:
    """Narrow an optional objective value (pyright) and fail loudly if it is missing."""
    assert value is not None
    return value


def make(coords) -> TspInstance:
    return TspInstance("custom", "custom", tuple((float(x), float(y)) for x, y in coords))


def brute_force(D: np.ndarray) -> float:
    n = len(D)
    if n <= 2:
        return 2.0 * D[0, n - 1] if n == 2 else 0.0
    perms = np.array(list(itertools.permutations(range(1, n))))  # ((n−1)!, n−1)
    tours = np.hstack([np.zeros((len(perms), 1), dtype=int), perms])
    return float(np.min(D[tours, np.roll(tours, -1, axis=1)].sum(axis=1)))


def is_permutation(t, n: int) -> bool:
    return sorted(t) == list(range(n))


def circle(n: int, perm) -> TspInstance:
    pts = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
    return make([pts[p] for p in perm])


def underflows(coords) -> bool:
    """True when the guard must refuse: a nonzero squared coordinate difference is below
    2^-1022 and the largest distance is below 2^-485.

    Evaluated in exact rational arithmetic (squares) and with np.hypot, which scales its
    arguments and so stays accurate for subnormal differences.
    """
    c = np.asarray(coords, dtype=float)  # (n, 2)
    tiny = Fraction(2) ** -1022
    sub = any(
        0 < Fraction(float(a)) - Fraction(float(b))
        and (Fraction(float(a)) - Fraction(float(b))) ** 2 < tiny
        for col in (c[:, 0], c[:, 1])
        for a in col
        for b in col
    )
    spread = float(np.max(np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])))
    return sub and spread < tsp.MIN_DISTANCE_SCALE


# Valid instances only: refused (underflowing) ones are covered by test_underflowing_*.
coords_strategy = (
    st.integers(1, 8)
    .flatmap(
        lambda n: st.lists(
            st.tuples(st.floats(0, 100, allow_nan=False), st.floats(0, 100, allow_nan=False)),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    .filter(lambda coords: not underflows(coords))
)
local_coords = st.integers(4, 12).flatmap(
    lambda n: st.lists(
        st.tuples(st.integers(0, 100), st.integers(0, 100)), min_size=n, max_size=n, unique=True
    )
)


# --------------------------------------------------------------------------------------
# Helpers: distances and tour length
# --------------------------------------------------------------------------------------


def test_distance_matrix_matches_cdist():
    c = np.asarray(problems.get("tsp_cities_20").coords)
    D = distance_matrix(c)
    np.testing.assert_allclose(D, cdist(c, c), rtol=1e-15, atol=0)
    assert np.array_equal(D, D.T) and not np.diag(D).any()


def test_tour_length_matches_numpy():
    inst = problems.get("tsp_random_15")
    D = distance_matrix(inst.coords)
    t = list(range(15))[::-1]
    ref = D[t, np.roll(t, -1)].sum()
    np.testing.assert_allclose(tour_length(D, t), ref, rtol=1e-14)


# --------------------------------------------------------------------------------------
# Contract on the library instances
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("pid", TSP_IDS)
def test_contract_on_library(method, pid):
    inst = problems.get(pid)
    if method == "tsp_held_karp" and inst.n > tsp.HELD_KARP_MAX_N:
        with pytest.raises(ValueError, match="n ≤ 16"):
            numopt.run(method, inst)
        return
    res = numopt.run(method, inst)
    assert_valid_result(res)
    assert len(res.trace) <= 400
    assert res.trace[-1].k == res.n_iter
    assert is_permutation(res.x, inst.n)
    np.testing.assert_allclose(
        num(res.fun), tour_length(distance_matrix(inst.coords), res.x), rtol=1e-14
    )
    assert num(res.fun) >= inst.optimum_length * (1 - 1e-12)
    assert res.extra["optimum_length"] == inst.optimum_length
    for s in res.trace:
        assert set(s.info["tour"]) <= set(range(inst.n))


def test_exact_and_local_search_reach_known_optima():
    circ = problems.get("tsp_circle_12")
    # NOTE: rtol 1e-12 — 12 rounded edge lengths are summed (each relative error ≤ 1 ulp).
    res = numopt.run("tsp_held_karp", circ)
    assert res.converged
    np.testing.assert_allclose(num(res.fun), circ.optimum_length, rtol=1e-12)
    for strategy in ("first", "best"):
        res = numopt.run("tsp_two_opt", circ, strategy=strategy)
        assert res.converged
        np.testing.assert_allclose(num(res.fun), circ.optimum_length, rtol=1e-12)
    grid = problems.get("tsp_grid_16")
    assert num(numopt.run("tsp_two_opt", grid).fun) == pytest.approx(160.0, rel=1e-12)
    cities = problems.get("tsp_cities_20")
    res = numopt.run("tsp_two_opt", cities, strategy="best", init="nearest_neighbor")
    assert num(res.fun) == pytest.approx(cities.optimum_length, rel=1e-12)


def test_held_karp_trace_layers():
    inst = problems.get("tsp_circle_12")
    res = numopt.run("tsp_held_karp", inst)
    assert [s.k for s in res.trace] == list(range(13))
    m = inst.n - 1
    for s in res.trace[1:-1]:
        size = s.info["subset_size"]
        assert s.info["states"] == size * math.comb(m, size)
        assert len(s.info["tour"]) == size + 1 and s.info["tour"][0] == 0
        assert not s.info["closed"]
    assert res.trace[-1].info["closed"]
    assert res.extra["states"] == sum(s * math.comb(m, s) for s in range(1, m + 1))


def test_held_karp_refuses_large_n():
    assert tsp.HELD_KARP_MAX_N == 16
    with pytest.raises(ValueError, match="n ≤ 16"):
        numopt.run("tsp_held_karp", problems.get("tsp_cities_20"))
    with pytest.raises(ValueError, match="n ≤ 16"):
        numopt.run("tsp_held_karp", problems.get("tsp_cities_20").coords[:17])


@pytest.mark.parametrize("pid", ("tsp_random_15", "tsp_grid_16"))
def test_held_karp_solves_library_instances_up_to_the_limit(pid):
    # Regression: the limit was n ≤ 13, which excluded these instances. The stored optima are
    # verified independently (SciPy MILP, grid argument) in test_problems_combinatorial.py.
    inst = problems.get(pid)
    res = numopt.run("tsp_held_karp", inst)
    assert_valid_result(res)
    assert res.converged and is_permutation(res.x, inst.n) and res.x[0] == 0
    assert [s.k for s in res.trace] == list(range(inst.n + 1))
    # NOTE: rtol 1e-12 — ≤ 16 rounded distances summed in a different order than the oracle.
    np.testing.assert_allclose(num(res.fun), inst.optimum_length, rtol=1e-12)


@settings(max_examples=1000, deadline=None)
@given(coords_strategy)
def test_held_karp_equals_brute_force(coords):
    inst = make(coords)
    D = distance_matrix(inst.coords)
    res = tsp.tsp_held_karp(inst)
    assert res.converged and is_permutation(res.x, inst.n) and res.x[0] == 0
    # NOTE: rtol 1e-12 — both sides sum ≤ 8 rounded distances in different orders.
    np.testing.assert_allclose(num(res.fun), brute_force(D), rtol=1e-12, atol=1e-12)


def test_held_karp_n9_brute_force():
    inst = make(problems.get("tsp_random_15").coords[:9])
    D = distance_matrix(inst.coords)
    np.testing.assert_allclose(num(tsp.tsp_held_karp(inst).fun), brute_force(D), rtol=1e-12)


# --------------------------------------------------------------------------------------
# Nearest neighbour
# --------------------------------------------------------------------------------------


def naive_nearest_neighbor(D: np.ndarray, start: int) -> list[int]:
    tour, left = [start], set(range(len(D))) - {start}
    while left:
        cur = tour[-1]
        nxt = min(left, key=lambda j: (D[cur, j], j))
        tour.append(nxt)
        left.remove(nxt)
    return tour


@pytest.mark.parametrize("start", (0, 4, 14))
def test_nearest_neighbor_matches_naive(start):
    inst = problems.get("tsp_random_15")  # integer coordinates: no near-ties below tol
    D = distance_matrix(inst.coords)
    res = numopt.run("tsp_nearest_neighbor", inst, start=start)
    assert res.x == naive_nearest_neighbor(D, start)
    assert [s.k for s in res.trace] == list(range(16)) and res.trace[-1].info["closed"]
    for s in res.trace[:-1]:
        np.testing.assert_allclose(
            num(s.fun), sum(D[a, b] for a, b in itertools.pairwise(s.x)), rtol=1e-14
        )


def test_nearest_neighbor_rejects_bad_start():
    with pytest.raises(ValueError, match="start"):
        numopt.run("tsp_nearest_neighbor", problems.get("tsp_random_15"), start=15)


def test_nearest_neighbor_tie_break_goes_to_smaller_index():
    inst = problems.get("tsp_grid_16")  # exact ties everywhere
    D = distance_matrix(inst.coords)
    res = numopt.run("tsp_nearest_neighbor", inst)
    assert res.x == naive_nearest_neighbor(D, 0)


# --------------------------------------------------------------------------------------
# 2-opt and Or-opt
# --------------------------------------------------------------------------------------


def best_two_opt_delta(D, t) -> float:
    n = len(t)
    best = 0.0
    for i in range(n - 2):
        for j in range(i + 2, n if i > 0 else n - 1):
            new = list(t)
            new[i + 1 : j + 1] = new[i + 1 : j + 1][::-1]
            best = min(best, tour_length(D, new) - tour_length(D, t))
    return best


def best_or_opt_delta(D, t) -> float:
    """Exhaustive: move every segment of length 1–3 to every other gap, both orientations."""
    n = len(t)
    base = tour_length(D, t)
    best = 0.0
    for L in (1, 2, 3):
        if L > n - 3:
            continue
        for i in range(n):
            seg = [t[(i + s) % n] for s in range(L)]
            rest = [t[(i + L + s) % n] for s in range(n - L)]
            for m in range(n - L - 1):
                for piece in (seg, seg[::-1]):
                    new = [*rest[: m + 1], *piece, *rest[m + 1 :]]
                    best = min(best, tour_length(D, new) - base)
    return best


@settings(max_examples=1000, deadline=None)
@given(
    local_coords,
    st.sampled_from(("first", "best")),
    st.sampled_from(("identity", "nearest_neighbor")),
)
def test_two_opt_reaches_a_two_optimal_tour(coords, strategy, init):
    inst = make(coords)
    D = distance_matrix(inst.coords)
    res = tsp.tsp_two_opt(inst, strategy=strategy, init=init)
    assert res.converged and is_permutation(res.x, inst.n)
    tol = tsp.REL_TOL * D.max()
    assert best_two_opt_delta(D, res.x) >= -tol * (1 + 1e-6) - 1e-12
    lengths = [num(s.fun) for s in res.trace]
    for prev, cur in itertools.pairwise(res.trace):
        move = cur.info["move"]
        assert num(cur.fun) < num(prev.fun), "every applied move strictly shortens the tour"
        np.testing.assert_allclose(
            num(cur.fun) - num(prev.fun), move["delta"], rtol=1e-9, atol=1e-9
        )
        a, b = move["removed"][0]
        c, d = move["removed"][1]
        assert move["added"] == [[a, c], [b, d]]
    assert num(res.fun) == lengths[-1]


@settings(max_examples=300, deadline=None)
@given(st.integers(4, 12).flatmap(lambda n: st.permutations(list(range(n)))))
def test_two_opt_is_optimal_for_points_in_convex_position(perm):
    # In convex position a 2-optimal tour has no crossing edges, so it is the hull order.
    n = len(perm)
    inst = circle(n, perm)
    res = tsp.tsp_two_opt(inst)
    np.testing.assert_allclose(num(res.fun), 2 * n * math.sin(math.pi / n), rtol=1e-12)


@settings(max_examples=500, deadline=None)
@given(local_coords, st.sampled_from(("identity", "nearest_neighbor")))
def test_or_opt_reaches_an_or_optimal_tour(coords, init):
    inst = make(coords)
    D = distance_matrix(inst.coords)
    res = tsp.tsp_or_opt(inst, init=init)
    assert res.converged and is_permutation(res.x, inst.n)
    assert best_or_opt_delta(D, res.x) >= -tsp.REL_TOL * D.max() * (1 + 1e-6) - 1e-12
    for prev, cur in itertools.pairwise(res.trace):
        move = cur.info["move"]
        assert num(cur.fun) < num(prev.fun)
        np.testing.assert_allclose(
            num(cur.fun) - num(prev.fun), move["delta"], rtol=1e-9, atol=1e-9
        )
        # The reported move is the net edge exchange: no edge is both removed and added, and
        # it turns the previous tour's edge set into the new one.
        removed = {frozenset(e) for e in move["removed"]}
        added = {frozenset(e) for e in move["added"]}
        assert not removed & added
        assert 2 <= len(removed) == len(added) <= 3
        assert (tour_edges(prev.x) - removed) | added == tour_edges(cur.x)
        assert removed <= tour_edges(prev.x) and added <= tour_edges(cur.x)


def tour_edges(t) -> set[frozenset[int]]:
    return {frozenset((t[k], t[(k + 1) % len(t)])) for k in range(len(t))}


def test_or_opt_degenerate_moves_report_the_net_exchange():
    # Regression: step k = 2 of this fixture reinserts the one-city segment [6] next to q = 8.
    # It used to report removed [[2, 6], [6, 8], [8, 4]] and added [[2, 8], [8, 6], [6, 4]],
    # i.e. edge {6, 8} removed and re-added although it stays in the tour.
    res = numopt.run("tsp_or_opt", problems.get("tsp_random_15"), init="nearest_neighbor")
    move = res.trace[2].info["move"]
    assert move["segment"] == [6]
    assert move["removed"] == [[2, 6], [8, 4]]
    assert move["added"] == [[2, 8], [6, 4]]
    for pid in TSP_IDS:
        for init in ("identity", "nearest_neighbor"):
            for s in numopt.run("tsp_or_opt", problems.get(pid), init=init).trace[1:]:
                mv = s.info["move"]
                assert not {frozenset(e) for e in mv["removed"]} & {
                    frozenset(e) for e in mv["added"]
                }


@pytest.mark.parametrize("n", (5, 8, 12))
def test_or_opt_full_scan_counts_each_distinct_move_once(n):
    # Cities in hull order: the identity tour is optimal, so one full scan finds nothing.
    # A one-city segment has one orientation; L = 2, 3 have two. Each segment of length L has
    # n − L − 1 insertion edges, and L ≤ n − 3.
    res = tsp.tsp_or_opt(circle(n, range(n)))
    assert res.converged and res.n_iter == 0
    expected = sum(n * (n - L - 1) * (1 if L == 1 else 2) for L in (1, 2, 3) if L <= n - 3)
    assert res.n_fev == expected


@pytest.mark.parametrize("method", ("tsp_two_opt", "tsp_or_opt"))
def test_local_search_max_iter_is_reported(method):
    res = numopt.run(method, problems.get("tsp_cities_20"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter=3" in res.message
    assert res.n_iter == 3


@pytest.mark.parametrize("method", ("tsp_two_opt", "tsp_or_opt"))
def test_local_search_record_every_keeps_final_step(method):
    full = numopt.run(method, problems.get("tsp_cities_20"))
    thin = numopt.run(method, problems.get("tsp_cities_20"), record_every=7)
    assert thin.x == full.x and thin.n_iter == full.n_iter
    assert [s.k for s in thin.trace][:-1] == list(range(0, full.n_iter + 1, 7))[
        : len(thin.trace) - 1
    ]
    assert thin.trace[-1].k == full.n_iter


def test_two_opt_rejects_bad_choices():
    inst = problems.get("tsp_circle_12")
    with pytest.raises(ValueError, match="strategy"):
        numopt.run("tsp_two_opt", inst, strategy="random")
    with pytest.raises(ValueError, match="init"):
        numopt.run("tsp_two_opt", inst, init="greedy")


# --------------------------------------------------------------------------------------
# Stochastic metaheuristics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", STOCHASTIC)
def test_stochastic_methods_are_deterministic_given_seed(method):
    inst = problems.get("tsp_random_15")
    a = numopt.run(method, inst, seed=5).to_dict()
    b = numopt.run(method, inst, seed=5).to_dict()
    c = numopt.run(method, inst, seed=6).to_dict()
    assert a == b
    assert a["trace"] != c["trace"]


@pytest.mark.parametrize("method", STOCHASTIC)
def test_stochastic_methods_use_only_the_portable_rng(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("np.random must not be used")

    for name in ("default_rng", "random", "rand", "randint", "permutation", "shuffle", "choice"):
        monkeypatch.setattr(np.random, name, forbidden)
    numopt.run(method, problems.get("tsp_circle_12"), seed=1)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_best_length_is_monotone_and_returned(method):
    inst = problems.get("tsp_cities_20")
    res = numopt.run(method, inst, seed=2)
    best = [s.info["best_length"] for s in res.trace if s.info["best_length"] is not None]
    assert all(b2 <= b1 for b1, b2 in itertools.pairwise(best))
    assert num(res.fun) == pytest.approx(best[-1], rel=1e-14)


def test_simulated_annealing_schedule_and_metropolis():
    inst = problems.get("tsp_random_15")
    res = numopt.run(
        "tsp_simulated_annealing", inst, seed=3, record_every=1, max_iter=2000, t_min=1e-9
    )
    D = distance_matrix(inst.coords)
    n = inst.n
    d_bar = D[np.triu_indices(n, 1)].mean()
    T0 = res.trace[0].info["temperature"]
    np.testing.assert_allclose(T0, d_bar, rtol=1e-13)
    for s in res.trace[1:]:
        # NOTE: rtol 1e-12 — k repeated multiplications by alpha (relative error ≤ k·eps).
        np.testing.assert_allclose(s.info["temperature"], T0 * 0.9995 ** (s.k - 1), rtol=1e-12)
        move = s.info["move"]
        if move["delta"] <= 0:
            assert move["accepted"], "improving moves are always accepted"
        a, b = move["removed"][0]
        c, d = move["removed"][1]
        assert len({a, b, c, d}) == 4, "the two removed edges are never adjacent"
    assert not res.converged and "max_iter=2000" in res.message
    assert res.n_fev == 2000


def test_simulated_annealing_freezes():
    res = numopt.run("tsp_simulated_annealing", problems.get("tsp_circle_12"), seed=0)
    assert res.converged and "frozen" in res.message
    expected = math.ceil(math.log(1e-3) / math.log(0.9995))
    assert abs(res.n_iter - expected) <= 1


# Regression (audit): t_min ≥ t0 was accepted. The run made one proposal, declared the system
# "frozen" and returned converged=True although the annealing schedule never ran.


@pytest.mark.parametrize(("t0", "t_min"), [(1e-3, 1.0), (0.5, 0.5), (1.0, 2.0)])
def test_simulated_annealing_rejects_an_empty_schedule(t0, t_min):
    with pytest.raises(ValueError, match=r"t_min = \S+ must be below t0"):
        numopt.run("tsp_simulated_annealing", problems.get("tsp_circle_12"), t0=t0, t_min=t_min)


def test_simulated_annealing_ui_ranges_cannot_give_an_empty_schedule():
    params = {p.name: p for p in numopt.get_method("tsp_simulated_annealing").params}
    t0, t_min = params["t0"], params["t_min"]
    assert t0.min is not None and t_min.max is not None
    assert t_min.max < t0.min
    assert t_min.default < t0.default


def test_simulated_annealing_proposals_are_uniform_pairs():
    # With n = 6 the 9 non-adjacent edge pairs should be proposed equally often.
    inst = make([(math.cos(k), math.sin(2 * k)) for k in range(6)])
    res = numopt.run(
        "tsp_simulated_annealing", inst, seed=11, record_every=1, max_iter=9000, t_min=1e-12
    )
    counts: dict[tuple[int, int], int] = {}
    for s in res.trace[1:]:
        key = (s.info["move"]["i"], s.info["move"]["j"])
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 9
    assert all(abs(c - 1000) < 150 for c in counts.values())  # ~4.7 σ for Binomial(9000, 1/9)


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(1, 12).flatmap(
        lambda n: st.tuples(
            st.permutations(list(range(n))),
            st.permutations(list(range(n))),
            st.integers(0, n - 1),
            st.integers(0, n - 1),
        )
    )
)
def test_order_crossover_properties(args):
    p1, p2, a, b = args
    a, b = min(a, b), max(a, b)
    child = order_crossover(list(p1), list(p2), a, b)
    n = len(p1)
    assert sorted(child) == list(range(n))
    assert child[a : b + 1] == list(p1[a : b + 1])
    # The other cities appear in p2's cyclic order starting after position b.
    seg = set(p1[a : b + 1])
    from_p2 = [p2[(b + 1 + s) % n] for s in range(n) if p2[(b + 1 + s) % n] not in seg]
    rest = [child[(b + 1 + s) % n] for s in range(n - (b - a + 1))]
    assert rest == from_p2


def test_genetic_population_statistics():
    res = numopt.run("tsp_genetic", problems.get("tsp_grid_16"), seed=4)
    for s in res.trace:
        info = s.info
        assert info["length"] <= info["mean_length"] <= info["worst_length"]
        assert info["length"] == info["best_length"]  # elitism: exact, not up to tol


def test_genetic_max_iter_is_reported():
    res = numopt.run("tsp_genetic", problems.get("tsp_cities_20"), seed=0, max_iter=5, patience=100)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and "max_iter=5" in res.message
    assert res.n_fev == 40 + 5 * 39


def test_ant_system_pheromone_update_one_ant():
    inst = problems.get("tsp_random_15")
    D = distance_matrix(inst.coords)
    rho = 0.3
    res = numopt.run("tsp_ant_colony", inst, seed=9, n_ants=1, rho=rho, max_iter=3, patience=50)
    nn_len = res.trace[0].info["length"]
    tau = np.full((15, 15), 1.0 / nn_len)
    np.fill_diagonal(tau, 0.0)
    np.testing.assert_allclose(res.trace[0].info["pheromone"], tau, rtol=1e-15)
    for s in res.trace[1:]:
        tour = s.info["tour"]
        L = tour_length(D, tour)
        tau = (1 - rho) * tau
        for a, b in zip(tour, np.roll(tour, -1), strict=True):
            tau[a, b] += 1 / L
            tau[b, a] += 1 / L
        P = np.asarray(s.info["pheromone"])
        np.testing.assert_allclose(P, tau, rtol=1e-13)
        assert np.array_equal(P, P.T)
    assert not res.converged and "max_iter=3" in res.message
    np.testing.assert_allclose(nn_len, tour_length(D, naive_nearest_neighbor(D, 0)), rtol=1e-14)


def test_ant_system_greedy_limit():
    # alpha = 0 and a large beta: ants follow almost pure nearest-neighbour choices.
    inst = problems.get("tsp_random_15")
    res = numopt.run("tsp_ant_colony", inst, seed=0, alpha=0.0, beta=10.0, n_ants=15, max_iter=5)
    nn_best = min(num(numopt.run("tsp_nearest_neighbor", inst, start=s).fun) for s in range(15))
    assert num(res.fun) <= nn_best * 1.05


# --------------------------------------------------------------------------------------
# Edge cases, validation, fixtures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("n", (1, 2, 3))
def test_tiny_instances(method, n):
    inst = make([(0, 0), (3, 4), (6, 0)][:n])
    res = numopt.run(method, inst)
    assert_valid_result(res)
    assert res.converged and is_permutation(res.x, n)
    expected = {1: 0.0, 2: 10.0, 3: 16.0}[n]
    assert num(res.fun) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("method", ALL)
def test_bare_coordinate_array_is_accepted(method):
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 2]])
    res = numopt.run(method, coords)
    assert is_permutation(res.x, 5)
    assert "optimum_length" not in res.extra


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("bad", ([], [[0, 0, 0]], [[0, 0], [np.nan, 1]], [[0, 0], [np.inf, 1]]))
def test_invalid_coordinates_raise(method, bad):
    with pytest.raises(ValueError):
        numopt.run(method, np.asarray(bad, dtype=float))


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("tsp_genetic", {"crossover_rate": 1.5}),
        ("tsp_genetic", {"pop_size": 1}),
        ("tsp_ant_colony", {"rho": 0.0}),
        ("tsp_ant_colony", {"beta": -1.0}),
        ("tsp_simulated_annealing", {"alpha": 1.0}),
        ("tsp_simulated_annealing", {"t0": 0.0}),
        ("tsp_two_opt", {"record_every": 0}),
    ],
)
def test_invalid_parameters_raise(method, params):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("tsp_circle_12"), **params)


def test_fixture_cases_are_valid_and_small():
    assert {m for m, _, _ in tsp.FIXTURE_CASES} == set(ALL)
    for method, pid, params in tsp.FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) <= 400
        json.dumps({"params": params, "result": res.to_dict()}, allow_nan=False)


# --------------------------------------------------------------------------------------
# Spec conformance: re-implementations written from the docstrings (the TS port's contract)
# --------------------------------------------------------------------------------------


def _delta(D, t, i, j) -> float:
    n = len(t)
    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
    return (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])


def reference_sa(D, seed, t0=1.0, alpha=0.9995, t_min=1e-3, max_iter=20_000):
    n = len(D)
    d_bar = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d_bar += D[i, j]
    d_bar /= n * (n - 1) // 2
    T, T_min = t0 * d_bar, t_min * d_bar
    rng = Rng(seed)
    t = list(range(n))
    tol = 1e-10 * D.max()
    best_t, best_len = list(t), tour_length(D, t)
    k = 0
    for k in range(1, max_iter + 1):  # noqa: B007
        i = int(rng.random() * n)
        j = (i + 2 + int(rng.random() * (n - 3))) % n
        i, j = min(i, j), max(i, j)
        delta = _delta(D, t, i, j)
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            t = t[: i + 1] + t[i + 1 : j + 1][::-1] + t[j + 1 :]
            if tour_length(D, t) < best_len - tol:
                best_t, best_len = list(t), tour_length(D, t)
        T *= alpha
        if T < T_min:
            break
    return best_t, best_len, k


def reference_ga(D, seed, pop_size=40, cx=0.9, mut=0.2, ts=3, patience=60, max_iter=300):
    n = len(D)
    rng = Rng(seed)
    tol = 1e-10 * D.max()

    def randint(m):
        return min(int(rng.random() * m), m - 1)

    def perm(m):
        out = list(range(m))
        for i in range(m - 1, 0, -1):
            j = randint(i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    pop = [perm(n) for _ in range(pop_size)]
    lens = [tour_length(D, p) for p in pop]
    best = min(lens)
    stall = 0
    gen = 0
    for gen in range(1, max_iter + 1):  # noqa: B007
        e = lens.index(min(lens))
        new, new_l = [list(pop[e])], [lens[e]]
        while len(new) < pop_size:
            parents = []
            for _ in range(2):
                draws = [randint(pop_size) for _ in range(ts)]
                parents.append(min(draws, key=lambda c: (lens[c], draws.index(c))))
            if rng.random() < cx:
                a, b = sorted((randint(n), randint(n)))
                child = order_crossover(pop[parents[0]], pop[parents[1]], a, b)
            else:
                child = list(pop[parents[0]])
            if rng.random() < mut:
                i, j = randint(n), randint(n)
                child[i], child[j] = child[j], child[i]
            new.append(child)
            new_l.append(tour_length(D, child))
        pop, lens = new, new_l
        if min(lens) < best - tol:
            best, stall = min(lens), 0
        else:
            stall += 1
        if stall >= patience:
            break
    return pop[lens.index(min(lens))], min(lens), gen


def reference_aco(D, seed, m=20, alpha=1.0, beta=5.0, rho=0.5, patience=30, max_iter=100):
    n = len(D)
    rng = Rng(seed)
    tol = 1e-10 * D.max()
    nn = naive_nearest_neighbor(D, 0)
    tau = np.full((n, n), m / tour_length(D, nn))
    np.fill_diagonal(tau, 0.0)
    with np.errstate(divide="ignore"):
        eta = np.where(np.eye(n, dtype=bool), 0.0, 1.0 / D)
    best_t, best_len, stall, it = [], math.inf, 0, 0
    for it in range(1, max_iter + 1):  # noqa: B007
        W = tau**alpha * eta**beta
        tours = []
        for _ in range(m):
            tour = [min(int(rng.random() * n), n - 1)]
            while len(tour) < n:
                i = tour[-1]
                cand = [j for j in range(n) if j not in tour]
                total = sum(W[i, j] for j in cand)
                target = rng.random() * total
                if total == 0.0:  # every unvisited τ_ij = 0 (ρ = 1): nearest unvisited city
                    tour.append(min(cand, key=lambda j: (D[i, j], j)))
                    continue
                run = 0.0
                for j in cand:
                    run += W[i, j]
                    if target < run:
                        tour.append(j)
                        break
            tours.append(tour)
        lens = [tour_length(D, tr) for tr in tours]
        tau = (1 - rho) * tau
        for tr, L in zip(tours, lens, strict=True):
            for a, b in zip(tr, tr[1:] + tr[:1], strict=True):
                tau[a, b] += 1 / L
                tau[b, a] += 1 / L
        k = lens.index(min(lens))
        if lens[k] < best_len - tol:
            best_t, best_len, stall = tours[k], lens[k], 0
        else:
            stall += 1
        if stall >= patience:
            break
    return best_t, best_len, it


@pytest.mark.parametrize("seed", (0, 1, 7))
def test_simulated_annealing_follows_documented_draw_order(seed):
    inst = problems.get("tsp_random_15")
    D = distance_matrix(inst.coords)
    tour, length, k = reference_sa(D, seed)
    res = numopt.run("tsp_simulated_annealing", inst, seed=seed)
    assert (res.x, num(res.fun), res.n_iter) == (tour, length, k)


@pytest.mark.parametrize("seed", (0, 3))
def test_genetic_follows_documented_draw_order(seed):
    inst = problems.get("tsp_grid_16")
    D = distance_matrix(inst.coords)
    tour, length, k = reference_ga(D, seed)
    res = numopt.run("tsp_genetic", inst, seed=seed)
    assert (res.x, num(res.fun), res.n_iter) == (tour, length, k)


@pytest.mark.parametrize("seed", (0, 4))
def test_ant_colony_follows_documented_draw_order(seed):
    inst = problems.get("tsp_random_15")
    D = distance_matrix(inst.coords)
    tour, length, k = reference_aco(D, seed)
    res = numopt.run("tsp_ant_colony", inst, seed=seed)
    assert (res.x, res.n_iter) == (tour, k)
    np.testing.assert_allclose(num(res.fun), length, rtol=1e-14)


# Regression (audit): raw τ^α·η^β under- or overflowed at large or small coordinate scales, and
# the all-zero guard then silently replaced the random-proportional rule by nearest-neighbour
# steps. AS is scale invariant (τ, η ∝ 1/c), so the run must not depend on the scale.


@pytest.mark.parametrize(
    ("scale", "params"),
    [
        (1e-25, {"alpha": 5.0, "beta": 10.0, "max_iter": 10}),
        (1e20, {"alpha": 5.0, "beta": 10.0, "max_iter": 10}),
        (1e60, {"max_iter": 30}),
        (1e100, {"max_iter": 30}),
    ],
)
def test_ant_colony_is_scale_invariant(scale, params):
    inst = problems.get("tsp_random_15")
    coords = np.asarray(inst.coords, dtype=float)
    base = numopt.run("tsp_ant_colony", coords, seed=3, **params)
    with np.errstate(all="raise"):
        res = numopt.run("tsp_ant_colony", coords * scale, seed=3, **params)
    assert_valid_result(res, max_iter=params["max_iter"])
    assert res.extra["nn_fallback_steps"] == 0 and "nearest" not in res.message
    assert res.x == base.x and res.n_iter == base.n_iter
    np.testing.assert_allclose(num(res.fun) / scale, num(base.fun), rtol=1e-12)
    # the unscaled run is not the nearest-neighbour tour the old guard fell back to
    assert num(base.fun) < res.trace[0].info["length"] / scale * (1 - 1e-3)


@given(
    pts=st.lists(
        st.tuples(st.integers(-50, 50), st.integers(-50, 50)), min_size=4, max_size=9, unique=True
    ),
    exponent=st.integers(-120, 450),
    alpha=st.sampled_from((0.0, 1.0, 2.5, 5.0)),
    beta=st.sampled_from((0.0, 1.0, 5.0, 10.0)),
    rho=st.sampled_from((0.1, 0.5, 0.99)),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=1000, deadline=None)
def test_ant_colony_power_of_two_scaling_is_exact(pts, exponent, alpha, beta, rho, seed):
    # Scaling by 2^e is exact in IEEE arithmetic (no subnormals, no overflow for these ranges),
    # so d_ref/d_ij and log(τ/τ₀) are bit-identical and the whole run must replay exactly.
    coords = np.asarray(pts, dtype=float)
    scale = math.ldexp(1.0, exponent)
    kw = {"seed": seed, "alpha": alpha, "beta": beta, "rho": rho, "n_ants": 3, "max_iter": 6}
    base = numopt.run("tsp_ant_colony", coords, **kw)
    with np.errstate(all="raise"):
        res = numopt.run("tsp_ant_colony", coords * scale, **kw)
    assert res.x == base.x and res.n_iter == base.n_iter
    assert num(res.fun) == num(base.fun) * scale
    assert res.extra["nn_fallback_steps"] == base.extra["nn_fallback_steps"] == 0
    for s, s0 in zip(res.trace, base.trace, strict=True):
        assert s.info["tour"] == s0.info["tour"]
        np.testing.assert_allclose(
            np.asarray(s.info["pheromone"]) * scale, s0.info["pheromone"], rtol=1e-15
        )


def test_ant_colony_zero_pheromone_rows_fall_back_and_are_reported():
    # ρ = 1 erases all pheromone except the last deposits, so from a city whose two tour
    # neighbours are already visited every unvisited τ_ij is 0 and p_ij = 0/0. The run must
    # replay the documented nearest-city fallback and report how often it was used.
    inst = problems.get("tsp_random_15")
    D = distance_matrix(inst.coords)
    kw = {"n_ants": 2, "rho": 1.0, "max_iter": 4, "patience": 50}
    res = numopt.run("tsp_ant_colony", inst, seed=5, **kw)
    assert_valid_result(res, max_iter=4)
    steps = res.extra["nn_fallback_steps"]
    assert steps > 0
    assert f"{steps} construction steps had zero pheromone" in res.message
    assert not res.converged and "max_iter=4" in res.message
    tour, length, k = reference_aco(D, 5, m=2, rho=1.0, patience=50, max_iter=4)
    assert (res.x, res.n_iter) == (tour, k)
    np.testing.assert_allclose(num(res.fun), length, rtol=1e-14)
    # with α = 0 the pheromone is ignored, so no row is ever all-zero
    res0 = numopt.run("tsp_ant_colony", inst, seed=5, alpha=0.0, **kw)
    assert res0.extra["nn_fallback_steps"] == 0 and "nearest" not in res0.message


# Regression (audit): the visibility guard clamped every d_ij < tol = 1e-10·max d up to tol, so
# distinct cities closer than tol all got the same η and Eq. 3.1 was silently changed. With
# d01 = 1e-11, d02 = 5e-11 and β = 5, p_01/p_02 = 5^5 = 3125, but the clamp gave 1. Only
# d_ij = 0 may be replaced, and its substitute must not make a coincident city less
# attractive than a distinct one (η is decreasing in d).
NEAR = [(0.0, 0.0), (1e-11, 0.0), (5e-11, 0.0), (1.0, 0.0), (0.0, 1.0)]


def first_moves_from_city_0(pts, n_seeds):
    """Next city after a start at city 0, over seeds (one ant, one iteration: τ is uniform)."""
    hits: dict[int, int] = {}
    for seed in range(n_seeds):
        res = numopt.run("tsp_ant_colony", pts, seed=seed, n_ants=1, max_iter=1, beta=5.0)
        tour = res.trace[1].info["tour"]
        if tour[0] == 0:
            hits[tour[1]] = hits.get(tour[1], 0) + 1
    return hits


def test_ant_colony_near_coincident_cities_follow_eq_3_1():
    pts = np.asarray(NEAR)
    D = distance_matrix(pts)
    assert 0.0 < D[0, 1] < D[0, 2] < tsp._tolerance(D)  # distinct, both closer than tol
    hits = first_moves_from_city_0(pts, 1000)
    starts = sum(hits.values())
    assert starts > 100
    # E[hits to 2] = starts/3126 ≈ 0.06; the clamp gave ≈ starts/2
    assert hits.get(2, 0) <= 3 and hits.get(1, 0) >= starts - 3
    for seed in (0, 3):  # full replay against the naive η = 1/d oracle
        tour, length, k = reference_aco(D, seed, max_iter=20)
        res = numopt.run("tsp_ant_colony", pts, seed=seed, max_iter=20)
        assert_valid_result(res, max_iter=20)
        assert (res.x, res.n_iter) == (tour, k)
        np.testing.assert_allclose(num(res.fun), length, rtol=1e-14)


def test_ant_colony_coincident_city_is_as_attractive_as_the_nearest_distinct_one():
    # d01 = 0 and d02 = 1e-11 < tol: the substitute d_0 = min(tol, 1e-11) gives w1 = w2, so
    # the move 0 → 1 is Binomial(starts, 1/2). Substituting tol would give w1/w2 ≈ 2e-6.
    pts = np.asarray([(0.0, 0.0), (0.0, 0.0), (1e-11, 0.0), (1.0, 0.0), (0.0, 1.0)])
    hits = first_moves_from_city_0(pts, 1000)
    starts = sum(hits.values())
    assert starts > 100 and hits.get(1, 0) + hits.get(2, 0) == starts
    assert abs(hits.get(1, 0) - starts / 2) <= 4.0 * math.sqrt(starts / 4)  # 4σ


@given(
    gap=st.floats(1e-16, 1e-11),
    ratio=st.floats(1.5, 20.0),
    beta=st.floats(0.5, 5.0),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=1000, deadline=None)
def test_ant_colony_matches_naive_oracle_below_tol(gap, ratio, beta, seed):
    # Distinct cities at distances gap and ratio·gap (both < tol): the run must replay the
    # naive τ^α·(1/d)^β oracle draw for draw.
    pts = np.asarray([(0.0, 0.0), (gap, 0.0), (ratio * gap, 0.0), (1.0, 0.0), (0.0, 1.0)])
    D = distance_matrix(pts)
    assert 0.0 < D[0, 1] < D[0, 2]
    tour, length, k = reference_aco(D, seed, m=3, beta=beta, max_iter=3)
    res = numopt.run("tsp_ant_colony", pts, seed=seed, n_ants=3, beta=beta, max_iter=3)
    assert (res.x, res.n_iter) == (tour, k)
    np.testing.assert_allclose(num(res.fun), length, rtol=1e-14)


def test_ant_colony_extreme_distance_ratio_does_not_overflow():
    # d01 ≈ 3e-162 next to d ≈ 1e154: d_ref/d01 overflows, so log η̂ must come from
    # log d_ref − log d01. An inf there made every row look all-zero (nearest-city fallback).
    pts = np.asarray([(0.0, 0.0), (3e-162, 0.0), (9e153, 0.0), (9e153, 9e153), (0.0, 9e153)])
    with np.errstate(under="ignore"):
        D = distance_matrix(pts)
    with np.errstate(over="ignore"):
        assert D[0, 1] > 0.0 and np.isinf(D.max() / D[0, 1])  # the case this test is about
    res = numopt.run("tsp_ant_colony", pts, seed=0, max_iter=5)
    assert_valid_result(res, max_iter=5)
    assert res.extra["nn_fallback_steps"] == 0 and "nearest" not in res.message


def test_metropolis_acceptance_frequency():
    # Over all uphill proposals, Σ(accepted − p) with p = exp(−Δ/T) has mean 0 and
    # variance Σ p(1 − p); a reversed or missing Metropolis test fails this by far.
    inst = problems.get("tsp_random_15")
    res = numopt.run(
        "tsp_simulated_annealing", inst, seed=8, record_every=1, max_iter=6000, t_min=1e-9
    )
    excess, var, count = 0.0, 0.0, 0
    for s in res.trace[1:]:
        move = s.info["move"]
        if move["delta"] > 0:
            p = math.exp(-move["delta"] / s.info["temperature"])
            excess += float(move["accepted"]) - p
            var += p * (1 - p)
            count += 1
    assert count > 1000 and var > 50
    assert abs(excess) < 5 * math.sqrt(var)


@pytest.mark.parametrize("method", ALL)
def test_overflowing_distances_raise(method):
    # Regression: finite coordinates whose differences square to inf used to give D = inf,
    # tol = inf, NaN comparisons and non-permutation tours reported as converged.
    coords = [[0, 0], [1e200, 0], [1e200, 1e200], [0, 1e200], [5e199, 2e200]]
    with pytest.raises(ValueError, match="overflow"):
        numopt.run(method, np.asarray(coords))
    with pytest.raises(ValueError, match="overflow"):
        numopt.run(method, np.asarray([[-1e154, 0.0], [1e154, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("method", ALL)
def test_large_but_representable_coordinates_are_solved(method):
    # |dx| = 2e150: dx·dx = 4e300 is finite, so the guard must not refuse the instance.
    scale = 1e150
    pts = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 3)]
    res = numopt.run(method, np.asarray(pts, dtype=float) * scale)
    assert_valid_result(res)
    assert res.converged and is_permutation(res.x, 5)
    optimum = brute_force(distance_matrix(np.asarray(pts, dtype=float))) * scale
    assert num(res.fun) >= optimum * (1 - 1e-12)
    if method == "tsp_held_karp":
        np.testing.assert_allclose(num(res.fun), optimum, rtol=1e-12)


@pytest.mark.parametrize("method", ALL)
def test_coincident_cities(method):
    res = numopt.run(method, np.zeros((6, 2)))
    assert_valid_result(res)
    assert res.converged and num(res.fun) == 0.0 and is_permutation(res.x, 6)


# Regression (audit): coordinates below ~1e-154 made dx·dx + dy·dy subnormal or 0. Distinct
# cities got d = 0, Held–Karp certified a tour 85 % above the optimum with fun = 0, and SA said
# "all cities coincide". The guard refuses every instance whose largest distance is < 2^-485.


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("scale", (1e-160, 1e-165, 1e-170))
def test_underflowing_distances_raise(method, scale):
    coords = np.asarray(problems.get("tsp_random_15").coords, dtype=float) * scale
    if method == "tsp_held_karp":
        coords = coords[:8]
    assert underflows(coords)
    with pytest.raises(ValueError, match="underflow"):
        numopt.run(method, coords)
    # Hypothesis shrunk counterexample: two distinct cities whose distance rounds to 0.
    with pytest.raises(ValueError, match="underflow"):
        numopt.run(method, np.array([[0.0, 0.0], [0.0, 2.225073858507203e-309]]))


@pytest.mark.parametrize("method", ALL)
def test_tiny_but_safe_scales_are_solved(method):
    # A tiny cluster next to normal-size cities is fine: its rounding error (≤ 2^-537) is far
    # below eps·max d. Identical tiny coordinates coincide exactly and are still valid.
    mixed = np.array([[0.0, 0.0], [1e-160, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    res = numopt.run(method, mixed)
    assert_valid_result(res)
    assert res.converged and is_permutation(res.x, 5)
    np.testing.assert_allclose(num(res.fun), 8.0, rtol=1e-15)
    same = numopt.run(method, np.full((5, 2), 1e-170))
    assert same.converged and num(same.fun) == 0.0 and is_permutation(same.x, 5)
    # Scale 1e-150: every square (≥ 1e-300) stays normal, so nothing underflows although
    # max d < 2^-485. The run must equal the unscaled one up to the rounding of the scaling.
    pts = np.asarray(problems.get("tsp_random_15").coords, dtype=float)[:8]
    base = numopt.run(method, pts)
    small = numopt.run(method, pts * 1e-150)
    assert small.converged == base.converged
    if method == "tsp_held_karp":
        # NOTE: rtol 1e-14 — coords·1e-150 is rounded (not a power of two); ≤ 8 distances summed.
        np.testing.assert_allclose(num(small.fun) * 1e150, num(base.fun), rtol=1e-14)


@given(
    pts=st.lists(
        st.tuples(st.integers(-50, 50), st.integers(-50, 50)), min_size=2, max_size=8, unique=True
    ),
    exponent=st.integers(-560, -440),
)
@settings(max_examples=1000, deadline=None)
def test_underflow_guard_threshold_and_exactness(pts, exponent):
    # Integer coordinates times 2^e: every square k²·2^(2e) (k² < 2^14) is exact for e ≥ −537,
    # so an accepted instance must replay the unscaled Held–Karp run exactly. The guard must
    # refuse exactly when a nonzero square k²·2^(2e) < 2^-1022 and max d < 2^-485.
    coords = np.asarray(pts, dtype=float)
    scale = math.ldexp(1.0, exponent)
    base = numopt.run("tsp_held_karp", coords)
    d_max = float(distance_matrix(coords).max()) * scale  # exact: a normal power-of-two scaling
    k_min = min(abs(a - b) for col in zip(*pts, strict=True) for a in col for b in col if a != b)
    square_underflows = Fraction(k_min**2) * Fraction(2) ** (2 * exponent) < Fraction(2) ** -1022
    if square_underflows and d_max < tsp.MIN_DISTANCE_SCALE:
        with pytest.raises(ValueError, match="underflow"):
            numopt.run("tsp_held_karp", coords * scale)
        return
    res = numopt.run("tsp_held_karp", coords * scale)
    assert res.converged and res.x == base.x
    assert num(res.fun) == num(base.fun) * scale


# Regression (audit): the UI range of ``start`` was [0, 1000] although the largest library
# instance has 20 cities, so a slider could select an index the method rejects.


def test_nearest_neighbor_start_range_fits_the_library():
    spec = next(p for p in numopt.get_method("tsp_nearest_neighbor").params if p.name == "start")
    library = [p for p in problems.list_problems("combinatorial") if isinstance(p, TspInstance)]
    assert len(library) >= 4
    largest = max(library, key=lambda inst: inst.n)
    assert spec.min == 0 and spec.max == largest.n - 1
    for start in range(largest.n):
        res = numopt.run("tsp_nearest_neighbor", largest, start=start)
        assert res.converged and res.x[0] == start and is_permutation(res.x, largest.n)


# Regression (audit): best_length moved only on improvements larger than tol, so it differed
# from the elite length by rounding (2.8e-14 on tsp_grid_16, seed 22) and from the returned fun.


@pytest.mark.parametrize("pid", ("tsp_grid_16", "tsp_circle_12"))
@pytest.mark.parametrize("seed", range(30))
def test_genetic_best_length_equals_elite_length_exactly(pid, seed):
    res = numopt.run("tsp_genetic", problems.get(pid), seed=seed)
    assert all(s.info["best_length"] == s.info["length"] for s in res.trace)
    assert num(res.fun) == res.trace[-1].info["best_length"]


def test_genetic_stall_counter_still_uses_tol():
    # tsp_grid_16, seed 22 improves by 2.8e-14 < tol at some generation: best_length follows
    # it, but the stall counter must not reset (it measures improvements larger than tol).
    inst = problems.get("tsp_grid_16")
    res = numopt.run("tsp_genetic", inst, seed=22)
    tol = tsp.REL_TOL * float(distance_matrix(inst.coords).max())
    for prev, cur in itertools.pairwise(res.trace):
        drop = prev.info["best_length"] - cur.info["best_length"]
        if cur.info["stall"] == 0:
            assert drop > tol
        else:
            assert cur.info["stall"] == prev.info["stall"] + 1


# Regression (audit): method-specific extra keys were not documented in the module docstring.


@pytest.mark.parametrize(
    ("method", "keys"),
    [
        ("tsp_nearest_neighbor", {"distance_evaluations"}),
        ("tsp_two_opt", {"initial_length"}),
        ("tsp_or_opt", {"initial_length"}),
        ("tsp_simulated_annealing", {"final_temperature"}),
        ("tsp_genetic", set()),
        ("tsp_ant_colony", {"tau0", "nn_fallback_steps"}),
        ("tsp_held_karp", {"states", "transitions"}),
    ],
)
def test_extra_keys_are_documented_and_present(method, keys):
    doc = tsp.__doc__ or ""
    extra_doc = doc[doc.index("Extra keys") : doc.index("Info keys")]
    for coords in (problems.get("tsp_circle_12"), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])):
        res = numopt.run(method, coords)
        assert set(res.extra) - {"optimum_length", "gap"} == keys
    for key in keys | {"optimum_length", "gap"}:
        assert f"{key}:" in extra_doc
    res = numopt.run("tsp_nearest_neighbor", problems.get("tsp_random_15"))
    assert res.extra["distance_evaluations"] == 15 * 14 // 2
