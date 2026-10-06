"""Tests for the benchmark-profile harness (method.py).

Oracles:
  * hand-computed values from the definitions (Dolan–Moré 2002 eq. 1, Moré–Wild 2009 eq. 2.2
    and 2.7, BenDFO's perf_profile.m / data_profile.m, COCO's ERT);
  * an independent code path: a raw per-evaluation log fed through the BenDFO port must give
    the same t_{p,s} as numopt.bench (improvement-only recorder) under the BenDFO cutoff;
  * exact integration of the step functions that numopt.bench returns, for the area formulas;
  * scipy.stats.kendalltau for τ_b.

Every comparison of integer counts or of count/|P| fractions is exact. Areas are sums of
O(|P|) logs, so rtol = 1e-12 (f64 rounding only, no conditioning issue).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from numopt import bench, problems
from numopt.core.rng import Rng

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("benchmark_profiles_method", _HERE / "method.py")
_conftest = _load("numopt_tests_conftest", _HERE.parents[1] / "tests" / "conftest.py")
assert_valid_result = _conftest.assert_valid_result

HYP = settings(max_examples=1000, deadline=None)


# --------------------------------------------------------------------------------------
# Hand-computed values
# --------------------------------------------------------------------------------------


def test_bendfo_solve_counts_hand_example() -> None:
    # One problem, two solvers, four evaluations each (raw values, not running minima).
    H = np.array([[10.0, 12.0], [8.0, 2.0], [3.0, 9.0], [5.0, 1.0]])[:, None, :]  # (4, 1, 2)
    # f(x0) = H[0, 0, 0] = 10, f_L = 1.
    # gate 0.5: cutoff 1 + 0.5·9 = 5.5; running minima [10, 8, 3, 3] and [12, 2, 2, 1].
    assert_array_equal(M.bendfo_solve_counts(H, 0.5), [[3.0, 2.0]])
    # gate 0.1: cutoff 1.9; only solver 2 gets there, at evaluation 4.
    assert_array_equal(M.bendfo_solve_counts(H, 0.1), [[math.inf, 4.0]])
    # An explicit f(x0) larger than every value moves the cutoff: 1 + 0.1·(21 − 1) = 3.
    assert_array_equal(M.bendfo_solve_counts(H, 0.1, f0=[21.0]), [[3.0, 2.0]])


def test_performance_area_hand_example() -> None:
    T = np.array([[10.0, 20.0], [30.0, 15.0], [math.inf, 40.0]])
    R = M.performance_ratios(T)
    assert_array_equal(R, [[1.0, 2.0], [2.0, 1.0], [math.inf, 1.0]])
    # L = log2 4 = 2: solver 1: (1 + 1/2 + 0)/3; solver 2: (1/2 + 1 + 1)/3.
    assert_allclose(M.performance_area(R, 4.0), [0.5, 2.5 / 3.0], rtol=1e-15)


def test_data_area_hand_example() -> None:
    K = np.array([[0.5, 4.0], [2.0, math.inf]])
    # L = log2 16 = 4: κ = 0.5 → clipped to 1 → 1; κ = 2 → 3/4; κ = 4 → 1/2; ∞ → 0.
    assert_allclose(M.data_area(K, 16.0), [(1.0 + 0.75) / 2, 0.25], rtol=1e-15)


def test_dominance_hand_example() -> None:
    y = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0], [0.5, 0.5, 0.5], [0.0, 0.5, 1.0]])
    D = M.dominance(y)
    assert D[0, 1] == 1 and D[1, 0] == -1  # 0 ≥ 1 everywhere, > at one point
    assert D[0, 2] == 2 and D[2, 0] == 2  # curves 0 and 2 cross
    assert D[0, 3] == 0  # equal curves
    assert_array_equal(np.diag(D), 0)


def test_rank_scores_ties() -> None:
    assert_array_equal(M.rank_scores([0.9, 0.5, 0.9, 0.1]), [1.5, 3.0, 1.5, 4.0])


def _synthetic_result(
    hist: dict[tuple[str, str], tuple[list[float], list[float], float]],
    insts: list[tuple[str, float, float | None, int]],
) -> bench.BenchmarkResult:
    """A BenchmarkResult from hand-written best-so-far histories (cost, best, total cost)."""
    labels = tuple(sorted({k[0] for k in hist}))
    instances = tuple(
        bench.Instance(iid, iid, (0.0,), 0, n, f0, f_known, f_known if f_known is not None else 0.0)
        for iid, f0, f_known, n in insts
    )
    runs = {
        k: bench.RunHistory(
            k[0],
            k[1],
            np.array(c),
            np.array(b),
            tot,
            int(tot),
            0,
            0,
            "budget",
            False,
            b[-1] if b else None,
            "",
        )
        for k, (c, b, tot) in hist.items()
    }
    return bench.BenchmarkResult(labels, instances, runs, budget=100.0, cost_model="nfev")


def test_ert_and_ecdf_hand_example() -> None:
    # Two instances with f_opt = 0 (n = 2); solver "a" reaches Δf = 1e-3 at cost 10 on p1 and
    # never on p2 (spends 100); solver "b" reaches it at 30 and 50.
    res = _synthetic_result(
        {
            ("a", "p1"): ([1.0, 10.0], [1.5, 1e-4], 100.0),
            ("a", "p2"): ([1.0], [1.5], 100.0),
            ("b", "p1"): ([1.0, 30.0], [1.5, 1e-3], 60.0),
            ("b", "p2"): ([1.0, 50.0], [1.5, 0.0], 50.0),
        },
        [("p1", 5.0, 0.0, 2), ("p2", 3.0, 0.0, 2)],
    )
    # ERT_a = (10 + 100) / 1, ERT_b = (30 + 50) / 2.
    assert_array_equal(M.expected_running_time(res, 1e-3), [110.0, 40.0])
    ecdf = M.runtime_ecdf(res, [1e-3, 2.0])
    # Runtimes / n: a: p1 → (5, 0.5), p2 → (∞, 0.5);  b: p1 → (15, 0.5), p2 → (25, 0.5).
    assert_array_equal(ecdf.runtimes[0], [[5.0, 0.5], [math.inf, 0.5]])
    assert_array_equal(ecdf.runtimes[1], [[15.0, 0.5], [25.0, 0.5]])
    j = int(np.searchsorted(ecdf.x, 15.0, side="right")) - 1
    assert_array_equal(ecdf.y[:, j], [0.75, 0.75])
    assert_array_equal(ecdf.y[:, -1], [0.75, 1.0])


# --------------------------------------------------------------------------------------
# Independent oracle: raw evaluation log + BenDFO port vs numopt.bench
# --------------------------------------------------------------------------------------

SOLVERS = {
    "nelder_mead": {"xtol": 1e-14, "ftol": 1e-16, "max_iter": 100_000},
    "compass_search": {"xtol": 1e-14, "max_iter": 100_000},
    "bfgs": {"gtol": 1e-14, "max_iter": 100_000},
    "cma_es": {"xtol": 1e-14, "ftol": 1e-16, "max_iter": 100_000},
}
PROBS = ["rosenbrock", "goldstein_price", "six_hump_camel", "ackley"]
STARTS = [None, [0.7, -0.4]]
BUDGET = 400


@pytest.fixture(scope="module")
def small_bench() -> bench.BenchmarkResult:
    cases = [
        bench.BenchmarkCase(dataclasses.replace(problems.get(p), grad=None, hess=None), STARTS)
        for p in PROBS
    ]
    return bench.run_benchmark(list(SOLVERS), cases, budget=BUDGET, params=SOLVERS, seeds=(0, 1))


@pytest.fixture(scope="module")
def raw_logs() -> dict[tuple[str, str], list[float]]:
    out = {}
    for pid in PROBS:
        prob = problems.get(pid)
        for i, x0 in enumerate(STARTS):
            for seed in (0, 1):
                iid = f"{pid}[{i}]@seed{seed}"
                for m, params in SOLVERS.items():
                    _, vals = M.log_evaluations(
                        m,
                        prob,
                        x0=prob.x0 if x0 is None else x0,
                        budget=BUDGET,
                        seed=seed if m == "cma_es" else None,
                        params=params,
                    )
                    out[(m, iid)] = vals
    return out


@pytest.mark.parametrize("tau", [1e-1, 1e-3, 1e-5, 1e-7])
def test_bendfo_port_matches_bench(small_bench, raw_logs, tau: float) -> None:
    res = M.with_bendfo_cutoff(small_bench)
    ids = [inst.id for inst in res.instances]
    H = M.history_array([[raw_logs[(m, iid)] for m in res.labels] for iid in ids], BUDGET)
    f0 = [inst.f0 for inst in res.instances]
    T_port = M.bendfo_solve_counts(H, tau, f0=f0)
    T_bench = res.solve_costs(tau)
    assert_array_equal(T_bench, T_port)
    # Profiles: bench's step function at every break point vs counting the port's values.
    perf = bench.performance_profile(res, tau)
    R = M.bendfo_performance_ratios(T_port)
    for a in perf.x:
        assert_array_equal(perf.at(a), (R <= a).mean(axis=0))
    data = bench.data_profile(res, tau)
    K = M.bendfo_data_values(T_port, [inst.n for inst in res.instances])
    for k in data.x:
        assert_array_equal(data.at(k), (K <= k).mean(axis=0))


def test_bendfo_cutoff_every_instance_solved(small_bench) -> None:
    res = M.with_bendfo_cutoff(small_bench)
    for tau in (1e-1, 1e-4, 1e-7, 1e-12):
        T = res.solve_costs(tau)
        assert np.isfinite(T).any(axis=1).all()
    # Under the known-minimum cutoff, some instance can be unsolved by every solver; the
    # BenDFO f_L is never above the known minimum unless every solver missed it.
    for a, b in zip(small_bench.instances, res.instances, strict=True):
        assert b.f_star <= a.f0
        if a.f_known is not None and b.f_star < a.f_known:
            assert a.f_star == b.f_star  # bench falls back to the found value as well


@pytest.mark.parametrize("method", list(SOLVERS))
def test_logger_counts_match_solver_counts_and_contract(method: str) -> None:
    prob = problems.get("beale")
    params = dict(SOLVERS[method])
    params["max_iter"] = 60
    res, vals = M.log_evaluations(
        method,
        prob,
        x0=prob.x0,
        budget=10**6,
        seed=0 if method == "cma_es" else None,
        params=params,
    )
    assert res is not None
    assert_valid_result(res, max_iter=60)
    assert res.n_fev == len(vals)  # every f evaluation is logged exactly once
    if method == "bfgs":  # central-difference gradients: 2n = 4 logged f values each
        assert res.n_fev >= 4 * res.n_gev
    assert min(vals) <= res.fun + 1e-12 * abs(res.fun)


def test_log_evaluations_stops_at_budget() -> None:
    res, vals = M.log_evaluations(
        "nelder_mead",
        problems.get("rosenbrock"),
        x0=[-1.2, 1.0],
        budget=17,
        params={"max_iter": 10_000},
    )
    assert res is None and len(vals) == 17


# --------------------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------------------

costs_tables = arrays(
    np.float64,
    st.tuples(st.integers(1, 8), st.integers(1, 5)),
    elements=st.one_of(st.integers(1, 2000).map(float), st.just(math.inf)),
)


def _exact_log2_area(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """∫ of the right-continuous step function y(x) over log₂ x ∈ [log₂ lo, log₂ hi], / L.

    ``y[:, j]`` is the value on [x[j], x[j+1]); x contains every jump, so the integral is a
    finite sum. Values left of x[0] are 0 (the grids start below the first jump).
    """
    u = np.log2(np.clip(x, lo, hi))
    edges = np.append(u, math.log2(hi))
    widths = np.diff(edges)  # (G,)
    return (y * widths).sum(axis=1) / math.log2(hi / lo)


@HYP
@given(costs_tables, st.sampled_from([2.0, 8.0, 32.0, 1000.0]))
def test_performance_area_equals_integral_of_bench_profile(T: np.ndarray, alpha_max: float) -> None:
    prof = bench.performance_profile_from_costs(T)
    grid = np.unique(np.concatenate([prof.x, [alpha_max]]))
    y = np.stack([prof.at(a) for a in grid], axis=1)
    want = _exact_log2_area(grid, y, 1.0, alpha_max)
    got = M.performance_area(M.performance_ratios(T), alpha_max)
    assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@HYP
@given(costs_tables, st.sampled_from([2.0, 50.0, 500.0]))
def test_data_area_equals_integral_of_bench_profile(T: np.ndarray, kappa_max: float) -> None:
    n = np.full(T.shape[0], 2)
    prof = bench.data_profile_from_costs(T, n)
    grid = np.unique(np.concatenate([prof.x, [1.0, kappa_max]]))
    grid = grid[grid >= 1.0]
    # Left of the grid start (strictly below the first jump) the profile is 0.
    y = np.stack([prof.at(k) if k >= prof.x[0] else np.zeros(T.shape[1]) for k in grid], axis=1)
    want = _exact_log2_area(grid, y, 1.0, kappa_max)
    got = M.data_area(M.bendfo_data_values(T, n), kappa_max)
    assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@HYP
@given(costs_tables, st.integers(-6, 6), st.integers(0, 7))
def test_performance_area_invariant_to_problem_scaling(T: np.ndarray, e: int, p: int) -> None:
    # r_{p,s} does not change when every solver's cost on p is scaled by c > 0 (c = 2^e,
    # exact in floating point).
    T2 = T.copy()
    T2[p % T.shape[0]] *= 2.0**e
    assert_array_equal(
        M.performance_area(M.performance_ratios(T2), 32.0),
        M.performance_area(M.performance_ratios(T), 32.0),
    )


@HYP
@given(costs_tables, arrays(np.float64, 8, elements=st.integers(1, 2000).map(float)))
def test_data_area_independent_of_other_solvers_and_bounded(
    T: np.ndarray, extra: np.ndarray
) -> None:
    n = np.full(T.shape[0], 2)
    T2 = np.column_stack([T, extra[: T.shape[0]]])
    a = M.data_area(M.bendfo_data_values(T, n), 500.0)
    b = M.data_area(M.bendfo_data_values(T2, n), 500.0)
    assert_array_equal(a, b[:-1])
    pa = M.performance_area(M.performance_ratios(T), 32.0)
    assert ((0.0 <= pa) & (pa <= 1.0)).all() and ((0.0 <= a) & (a <= 1.0)).all()
    solved_any = np.isfinite(T).any(axis=1)
    # Each instance solved by someone gives exactly one ratio 1 → Σ_s A_s ≥ fraction solved.
    assert pa.sum() >= solved_any.mean() - 1e-12


raw_histories = arrays(
    np.float64,
    st.tuples(st.integers(1, 12), st.integers(1, 5), st.integers(1, 4)),
    elements=st.integers(-50, 50).map(float),
)


@HYP
@given(raw_histories, st.integers(1, 1023), st.integers(-3, 3), st.integers(-20, 20))
def test_bendfo_counts_properties(H: np.ndarray, g: int, e: int, b: int) -> None:
    gate = g / 1024.0  # dyadic: every quantity below is exact in f64
    T = M.bendfo_solve_counts(H, gate)
    # (i) the solver that attains f_L solves every problem: no all-∞ row.
    assert np.isfinite(T).any(axis=1).all()
    # (ii) affine invariance of eq. 2.2: f → 2^e f + b leaves T unchanged.
    assert_array_equal(M.bendfo_solve_counts(2.0**e * H + b, gate), T)
    # (iii) a looser gate never needs more evaluations.
    T_loose = M.bendfo_solve_counts(H, min(gate * 2, 1023 / 1024))
    assert (T_loose <= T).all()
    # (iv) brute force from the definition.
    nf, P, S = H.shape
    for p in range(P):
        best = [min(H[: k + 1, p, s]) for k in range(nf) for s in range(S)]
        f_L = min(min(best), H[0, p, 0])
        cut = f_L + gate * (H[0, p, 0] - f_L)
        for s in range(S):
            ks = [k + 1 for k in range(nf) if min(H[: k + 1, p, s]) <= cut]
            assert T[p, s] == (ks[0] if ks else math.inf)


@HYP
@given(
    arrays(np.float64, st.integers(2, 7), elements=st.integers(0, 4).map(float)),
    arrays(np.float64, 7, elements=st.integers(0, 4).map(float)),
)
def test_kendall_tau_b_matches_scipy(a: np.ndarray, b: np.ndarray) -> None:
    b = b[: a.size]
    want = float(stats.kendalltau(a, b, variant="b").statistic)  # pyright: ignore[reportAttributeAccessIssue]
    got = M.kendall_tau_b(a, b)
    if math.isnan(want):
        assert math.isnan(got)
    else:
        assert_allclose(got, want, rtol=1e-12, atol=1e-15)


@settings(max_examples=1000, deadline=None)
@given(
    arrays(np.float64, st.tuples(st.integers(2, 10), st.integers(2, 4)), elements=st.floats(0, 1)),
    st.integers(1, 4),
    st.integers(0, 2**31),
)
def test_bootstrap_support_properties(X: np.ndarray, n_groups: int, seed: int) -> None:
    groups = [f"g{i % n_groups}" for i in range(X.shape[0])]
    bs = M.bootstrap_order_support(X, groups, n_boot=40, seed=seed)
    S = X.shape[1]
    assert_allclose(bs.p_greater + bs.p_greater.T, np.ones((S, S)), atol=1e-12)
    assert_allclose(np.diag(bs.p_greater), 0.5)
    # Every replicate score is a weighted mean of the columns: inside [min, max].
    assert (bs.scores >= X.min(axis=0) - 1e-12).all() and (bs.scores <= X.max(axis=0) + 1e-12).all()
    # A column that dominates another in every instance wins (or ties) in every replicate.
    Y = np.column_stack([X[:, 0], X[:, 0] * 0.5])
    bs2 = M.bootstrap_order_support(Y, groups, n_boot=40, seed=seed)
    assert bs2.p_greater[0, 1] >= 0.5


def test_bootstrap_is_deterministic_and_single_group_is_degenerate() -> None:
    X = np.array([[0.2, 0.4], [0.9, 0.1], [0.5, 0.5]])
    a = M.bootstrap_order_support(X, ["a", "b", "c"], n_boot=200, seed=7)
    b = M.bootstrap_order_support(X, ["a", "b", "c"], n_boot=200, seed=7)
    assert_array_equal(a.scores, b.scores)
    one = M.bootstrap_order_support(X, ["z"] * 3, n_boot=50, seed=1)
    assert_allclose(one.scores, np.tile(X.mean(axis=0), (50, 1)), rtol=1e-15)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        M.bendfo_solve_counts(np.zeros((2, 1, 1)), 1.0)
    with pytest.raises(ValueError):
        M.performance_area(np.ones((2, 2)), 1.0)
    with pytest.raises(ValueError):
        M.data_area(np.array([[np.nan]]), 10.0)
    with pytest.raises(ValueError):
        M.runtime_ecdf(
            _synthetic_result({("a", "p"): ([1.0], [1.0], 1.0)}, [("p", 1.0, 0.0, 1)]), [0.0]
        )
    assert_allclose(M.coco_targets()[[0, -1]], [1e2, 1e-8])
    assert M.coco_targets().size == 51


# --------------------------------------------------------------------------------------
# Cutoff audit: an unbounded problem must be caught before any analysis
# --------------------------------------------------------------------------------------


def test_cutoff_audit_hand_example() -> None:
    res = _synthetic_result(
        {
            ("a", "p1"): ([1.0], [-5.0], 10.0),  # far below f_known = 0: flagged
            ("a", "p2"): ([1.0], [-1e-15], 10.0),  # rounding level: not flagged
            ("a", "p3"): ([1.0], [-7.0], 10.0),  # no stated minimum: not audited
            ("a", "p4"): ([1.0], [0.5], 10.0),  # above f_known: not flagged
        },
        [("p1", 5.0, 0.0, 2), ("p2", 5.0, 0.0, 2), ("p3", 5.0, None, 2), ("p4", 5.0, 0.0, 2)],
    )
    audit = M.cutoff_audit(res)
    assert audit.flagged == ("p1",)
    assert_allclose(audit.gap[[0, 1, 3]], [1.0, 2e-16, -0.1], rtol=1e-15)
    assert math.isnan(audit.gap[2])
    # rtol scales with f(x0) − f_known: a gap of 1e-9 of that scale passes at rtol = 1e-8.
    res2 = _synthetic_result({("a", "p"): ([1.0], [-5e-9], 1.0)}, [("p", 5.0, 0.0, 2)])
    assert M.cutoff_audit(res2, rtol=1e-10).flagged == ("p",)
    assert M.cutoff_audit(res2, rtol=1e-8).flagged == ()


def test_cutoff_audit_flags_unbounded_mccormick_not_bounded_problems(small_bench) -> None:
    # McCormick's f_min holds on its box only; along x − y = 1, f = σ/2 + sin σ → −∞.
    mc = dataclasses.replace(problems.get("mccormick"), grad=None, hess=None)
    assert "bounded-domain" in mc.tags
    # (i) CMA-ES from the default x0 = (3, −2), seed 1, leaves the box and diverges (the
    # review's mccormick[0]@seed1 run, f ≈ −7.7e13).
    case = bench.BenchmarkCase(mc, [None])
    res = bench.run_benchmark(
        ["cma_es"], [case], budget=1500, seeds=(1,), params={"cma_es": SOLVERS["cma_es"]}
    )
    audit = M.cutoff_audit(res)
    assert audit.flagged == (res.instances[0].id,) and audit.gap[0] > 1e6
    # (ii) A start on the valley x − y = 1, outside the box, is already below f_min.
    case = bench.BenchmarkCase(mc, [[-10.0, -11.0]])
    res = bench.run_benchmark(
        ["nelder_mead"], [case], budget=400, params={"nelder_mead": SOLVERS["nelder_mead"]}
    )
    audit = M.cutoff_audit(res)
    assert audit.flagged == (res.instances[0].id,) and audit.gap[0] == math.inf
    # The library problems of the oracle benchmark have their minimum on ℝ²: nothing flagged.
    assert M.cutoff_audit(small_bench).flagged == ()


def test_run_excludes_bounded_domain_problems_and_keeps_start_points() -> None:
    sys.path.insert(0, str(_HERE))
    try:
        R = _load("benchmark_profiles_run", _HERE / "run.py")
    finally:
        sys.path.remove(str(_HERE))
    cases = R.make_cases()
    ids = [c.resolve().id for c in cases]
    two_d = [p for p in problems.list_problems("unconstrained") if p.dim == 2]
    assert ids == [p.id for p in two_d if "bounded-domain" not in p.tags]
    assert "mccormick" not in ids and len(ids) == 15
    # The draws of an excluded problem are still consumed: the last problem's starts equal the
    # draws made over every 2-D problem in library order.
    rng = Rng(R.START_SEED)
    want = []
    for p in two_d:
        want = [[rng.uniform(*p.domain[0]), rng.uniform(*p.domain[1])] for _ in range(7)]
    assert [list(x) for x in list(cases[-1].x0s or [])[1:]] == want


# --------------------------------------------------------------------------------------
# Multiple comparisons and leave-one-problem-out
# --------------------------------------------------------------------------------------


def test_holm_hand_example() -> None:
    # Sorted p: 0.005, 0.01, 0.03, 0.04 → 4·0.005, 3·0.01, 2·0.03, 1·0.04 = 0.02, 0.03, 0.06,
    # 0.04 → running max 0.02, 0.03, 0.06, 0.06.
    assert_allclose(M.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02], rtol=1e-15)
    assert_array_equal(M.holm([1.0, 1.0]), [1.0, 1.0])
    with pytest.raises(ValueError):
        M.holm([0.5, 1.5])


def _holm_step_down(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm's sequential procedure, as stated in Holm (1979): reject H_(1), H_(2), … while
    p_(k) ≤ α / (m − k + 1); stop at the first acceptance."""
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for k, idx in enumerate(np.argsort(p, kind="stable")):
        if p[idx] * (m - k) <= alpha:
            reject[idx] = True
        else:
            break
    return reject


@HYP
@given(
    arrays(np.float64, st.integers(1, 40), elements=st.integers(0, 1024).map(lambda k: k / 1024)),
    st.sampled_from([0.01, 0.05, 0.1, 0.25]),
)
def test_holm_matches_step_down_procedure(p: np.ndarray, alpha: float) -> None:
    adj = M.holm(p)
    assert_array_equal(adj <= alpha, _holm_step_down(p, alpha))
    assert (adj >= p).all() and (adj <= np.minimum(1.0, p.size * p)).all()  # ≤ Bonferroni
    order = np.argsort(p, kind="stable")
    assert (np.diff(adj[order]) >= 0).all()  # monotone in p


def test_order_support_is_direction_aware() -> None:
    bs = M.BootstrapSupport(
        p_greater=np.array([[0.5, 0.97], [0.03, 0.5]]), scores=np.zeros((1, 2)), ci=np.zeros((2, 2))
    )
    assert M.order_support(bs, 0, 1, 0.2) == 0.97
    assert M.order_support(bs, 0, 1, -0.2) == 0.03  # the point estimate disagrees with P*
    assert M.order_support(bs, 0, 1, 0.0) == 0.5


def test_leave_one_group_out_matches_recomputed_areas(small_bench) -> None:
    res = M.with_bendfo_cutoff(small_bench)
    groups = [inst.problem_id for inst in res.instances]
    Ta, Tb = res.solve_costs(1e-1), res.solve_costs(1e-7)
    Xa = M.area_terms(M.performance_ratios(Ta), 32.0)
    Xb = M.area_terms(M.performance_ratios(Tb), 32.0)
    i, j = 0, 3  # nelder_mead vs cma_es
    out = M.leave_one_group_out(Xa, Xb, groups, i, j, n_boot=300, seed=4, confirm=0.9)
    assert sorted(out) == sorted(set(groups))
    for g, v in out.items():
        keep = np.array([x != g for x in groups])
        sub = [x for x in groups if x != g]
        # Independent path: the area from the cost table of the remaining instances.
        Aa = M.performance_area(M.performance_ratios(Ta[keep]), 32.0)
        Ab = M.performance_area(M.performance_ratios(Tb[keep]), 32.0)
        assert_allclose([v["diff_a"], v["diff_b"]], [Aa[i] - Aa[j], Ab[i] - Ab[j]], atol=1e-14)
        ba = M.bootstrap_order_support(Xa[keep], sub, n_boot=300, seed=4)
        pa = ba.p_greater[i, j] if v["diff_a"] > 0 else ba.p_greater[j, i]
        assert v["support_a"] == pa
        flip = v["diff_a"] * v["diff_b"] < 0
        assert v["kept"] == (flip and min(v["support_a"], v["support_b"]) >= 0.9)


def test_leave_one_group_out_hand_example() -> None:
    # Three problems; solver 0 beats solver 1 in setting a everywhere, and loses in setting b
    # everywhere except problem "c". Dropping "c" keeps the flip; the bootstrap of the
    # remaining two problems is unanimous, so the flip is kept at any confirm level.
    A = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    B = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    out = M.leave_one_group_out(A, B, ["a", "b", "c"], 0, 1, n_boot=200, seed=0, confirm=0.99)
    assert out["c"]["kept"] and out["c"]["support_a"] == 1.0 and out["c"]["support_b"] == 1.0
    assert out["a"]["diff_b"] == 0.0 and not out["a"]["flip"] and not out["a"]["kept"]
