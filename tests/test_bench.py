"""Tests for numopt.bench: profiles against hand-computed and naive definitions, the runner."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

import numopt
from numopt import bench, problems
from numopt.cli import main as cli_main
from numopt.core.types import Problem, Result, Step

INF = math.inf

# Hand table: 4 problems × 3 solvers (A, B, C); ∞ = not solved.
#   p0: best 10 → r = (1, 2, ∞)     p1: best 15 → r = (2, 1, 4)
#   p2: nobody  → r = (∞, ∞, ∞)     p3: best 8  → r = (1, 1, 2)
HAND = np.array(
    [
        [10.0, 20.0, INF],
        [30.0, 15.0, 60.0],
        [INF, INF, INF],
        [8.0, 8.0, 16.0],
    ]
)
HAND_N = np.array([1, 2, 3, 1])  # κ = t / (n + 1): p0 (5, 10, ∞), p1 (10, 5, 20), p3 (4, 4, 8)


# --------------------------------------------------------------------------------------
# Profiles: exact values on a hand-built table
# --------------------------------------------------------------------------------------


def test_performance_profile_hand_table_exact() -> None:
    prof = bench.performance_profile_from_costs(HAND, ["A", "B", "C"])
    expected = {
        1.0: [2 / 4, 2 / 4, 0 / 4],
        1.5: [2 / 4, 2 / 4, 0 / 4],
        2.0: [3 / 4, 3 / 4, 1 / 4],
        3.9: [3 / 4, 3 / 4, 1 / 4],
        4.0: [3 / 4, 3 / 4, 2 / 4],
        1e300: [3 / 4, 3 / 4, 2 / 4],
    }
    for alpha, values in expected.items():
        assert_array_equal(prof.at(alpha), values)
    assert_array_equal(prof.solved, [3 / 4, 3 / 4, 2 / 4])
    # Every break point is on the grid, and the grid starts at α = 1.
    assert prof.x[0] == 1.0
    assert {1.0, 2.0, 4.0} <= set(prof.x.tolist())
    assert_array_equal(prof.y[:, -1], prof.solved)


def test_data_profile_hand_table_exact() -> None:
    prof = bench.data_profile_from_costs(HAND, HAND_N, ["A", "B", "C"])
    expected = {
        4.0: [1 / 4, 1 / 4, 0 / 4],
        4.5: [1 / 4, 1 / 4, 0 / 4],
        5.0: [2 / 4, 2 / 4, 0 / 4],
        8.0: [2 / 4, 2 / 4, 1 / 4],
        10.0: [3 / 4, 3 / 4, 1 / 4],
        20.0: [3 / 4, 3 / 4, 2 / 4],
        1e300: [3 / 4, 3 / 4, 2 / 4],
    }
    for kappa, values in expected.items():
        assert_array_equal(prof.at(kappa), values)
    assert_array_equal(prof.y[:, 0], [0.0, 0.0, 0.0])  # the grid starts strictly below min κ = 4
    assert_array_equal(prof.solved, [3 / 4, 3 / 4, 2 / 4])


def test_profile_on_explicit_grid() -> None:
    prof = bench.performance_profile_from_costs(HAND, alphas=[4.0, 1.0, 2.0])
    assert_array_equal(prof.x, [1.0, 2.0, 4.0])
    assert_array_equal(prof.y, [[0.5, 0.75, 0.75], [0.5, 0.75, 0.75], [0.0, 0.25, 0.5]])
    assert prof.labels == ("s0", "s1", "s2")


# --------------------------------------------------------------------------------------
# Profiles: properties against a naive implementation of the definitions
# --------------------------------------------------------------------------------------


def _naive_performance(T: np.ndarray, alpha: float) -> list[float]:
    """ρ_s(α) straight from Dolan & Moré (2002), with Python loops."""
    P, S = T.shape
    out = []
    for s in range(S):
        count = 0
        for p in range(P):
            best = min(float(T[p, j]) for j in range(S))
            if math.isfinite(T[p, s]) and T[p, s] / best <= alpha:
                count += 1
        out.append(count / P)
    return out


def _naive_data(T: np.ndarray, n: np.ndarray, kappa: float) -> list[float]:
    """d_s(κ) straight from Moré & Wild (2009, eq. 2.7), with Python loops."""
    P, S = T.shape
    return [
        sum(1 for p in range(P) if T[p, s] / (float(n[p]) + 1.0) <= kappa) / P for s in range(S)
    ]


@st.composite
def cost_tables(draw: Any) -> tuple[np.ndarray, np.ndarray]:
    P = draw(st.integers(1, 7))
    S = draw(st.integers(1, 5))
    cell = st.one_of(st.integers(1, 60).map(float), st.just(INF))
    T = np.array([[draw(cell) for _ in range(S)] for _ in range(P)])
    n = np.array([draw(st.integers(1, 6)) for _ in range(P)])
    return T, n


@settings(max_examples=1000, deadline=None)
@given(cost_tables())
def test_profiles_match_naive_definitions(table: tuple[np.ndarray, np.ndarray]) -> None:
    T, n = table
    perf = bench.performance_profile_from_costs(T)
    data = bench.data_profile_from_costs(T, n)
    for j, alpha in enumerate(perf.x):
        assert perf.y[:, j].tolist() == _naive_performance(T, float(alpha))
    for j, kappa in enumerate(data.x):
        assert data.y[:, j].tolist() == _naive_data(T, n, float(kappa))
    for prof in (perf, data):
        assert np.all(np.diff(prof.x) > 0)
        assert np.all(np.diff(prof.y, axis=1) >= 0), "a profile must be non-decreasing"
        assert np.all((prof.y >= 0) & (prof.y <= 1))
        # ρ_s(∞) = d_s(∞) = fraction solved, reached on the last grid point.
        assert_array_equal(prof.y[:, -1], np.isfinite(T).mean(axis=0))
        assert_array_equal(prof.at(1e308), prof.solved)
    # Every problem someone solves has a winner: Σ_s ρ_s(1) ≥ fraction solved by anyone.
    assert perf.y[:, 0].sum() >= np.isfinite(T).any(axis=1).mean() - 1e-15


@settings(max_examples=1000, deadline=None)
@given(cost_tables(), st.data())
def test_dominated_solver_leaves_other_profiles_unchanged(
    table: tuple[np.ndarray, np.ndarray], data: Any
) -> None:
    T, n = table
    # A dominated solver costs at least as much as every other solver on every problem.
    extra = np.array(
        [
            data.draw(
                st.one_of(st.just(INF), st.floats(1.0, 4.0).map(lambda f, r=row: f * r.max()))
            )
            for row in T
        ]
    )
    T2 = np.column_stack([T, extra])
    perf = bench.performance_profile_from_costs(T)
    perf2 = bench.performance_profile_from_costs(T2, alphas=perf.x)
    assert_array_equal(perf2.y[:-1], perf.y)
    data1 = bench.data_profile_from_costs(T, n)
    data2 = bench.data_profile_from_costs(T2, n, kappas=data1.x)
    assert_array_equal(data2.y[:-1], data1.y)


def test_non_dominated_solver_does_change_ratios() -> None:
    """Sanity check of the invariance test: a faster solver shifts the others' ratios."""
    T2 = np.column_stack([HAND, [1.0, 1.0, 1.0, 1.0]])
    prof = bench.performance_profile_from_costs(T2, alphas=[1.0, 2.0])
    assert_array_equal(prof.y[:3, 0], [0.0, 0.0, 0.0])


def test_profile_input_validation() -> None:
    with pytest.raises(ValueError):
        bench.performance_profile_from_costs(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        bench.performance_profile_from_costs([[1.0, math.nan]])
    with pytest.raises(ValueError):
        bench.performance_profile_from_costs(np.ones((0, 2)))
    with pytest.raises(ValueError):
        bench.performance_profile_from_costs(HAND, alphas=[0.5, 2.0])
    with pytest.raises(ValueError):
        bench.data_profile_from_costs(HAND, [1, 2])
    with pytest.raises(ValueError):
        bench.performance_profile_from_costs(HAND, ["A", "B"])


# --------------------------------------------------------------------------------------
# Convergence test (Moré–Wild eq. 2.2)
# --------------------------------------------------------------------------------------


def test_is_solved_boundary() -> None:
    # f_L = 1, f(x0) = 11, τ = 0.5 → threshold 6 (all exactly representable).
    assert bench.is_solved(6.0, 11.0, 1.0, 0.5)
    assert not bench.is_solved(math.nextafter(6.0, 7.0), 11.0, 1.0, 0.5)
    assert bench.is_solved(1.0, 1.0, 1.0, 0.5)  # starting at the minimum solves at once
    for tau in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            bench.is_solved(0.0, 1.0, 0.0, tau)


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------

F_MIN = 3.0
X_STAR = np.array([1.0, -2.0])


def _quad_f(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + 4.0 * (x[1] + 2.0) ** 2 + F_MIN)


def _quad_grad(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)])


QUAD = Problem(
    id="test_quad",
    name="test quadratic",
    latex="",
    f=_quad_f,
    dim=2,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    grad=_quad_grad,
    x0=[3.0, 1.0],
    extra={"f_min": F_MIN},
)


def _scripted(problem: Problem, *, x0: Any, use_grad: bool = True) -> Result:
    """Evaluate f at x* + 10^{-k}(x0 − x*), k = 0..5, with one gradient after each f.

    On this quadratic f − f* = 10^{-2k} (f(x0) − f*), so the τ = 1e-3 test first passes at k = 2.
    """
    x0v = np.asarray(x0, dtype=float)
    trace = []
    for k in range(6):
        x = X_STAR + 10.0**-k * (x0v - X_STAR)
        fx = problem.f(x)
        if use_grad:
            assert problem.grad is not None
            problem.grad(x)
        trace.append(Step(k=k, x=x, fun=fx))
    return Result("scripted", trace[-1].x, trace[-1].fun, True, "script done", 5, trace=trace)


def _lazy(problem: Problem, *, x0: Any) -> Result:
    """A dominated solver: it evaluates f(x0) ten times and never moves."""
    x = np.asarray(x0, dtype=float)
    fx = 0.0
    for _ in range(10):
        fx = problem.f(x)
    return Result("lazy", x, fx, False, "lazy", 0, trace=[Step(0, x, fx)])


@pytest.mark.parametrize(
    ("cost", "t_expected", "total"), [("nfev", 3.0, 6.0), ("nfev+n*ngev", 7.0, 18.0)]
)
def test_scripted_solver_exact_costs(cost: str, t_expected: float, total: float) -> None:
    res = bench.run_benchmark(
        [("scripted", _scripted)],
        [bench.BenchmarkCase(QUAD)],
        budget=100,
        cost=cost,  # type: ignore[arg-type]
    )
    (inst,) = res.instances
    assert inst.f0 == _quad_f(np.array([3.0, 1.0])) == 43.0
    assert inst.f_known == F_MIN and inst.f_star == F_MIN
    h = res.history("scripted", inst.id)
    # f evals at costs 1,2,3,.. (nfev) or 1,4,7,.. (gradient = n = 2 evaluations).
    step = 1.0 if cost == "nfev" else 3.0
    assert_array_equal(h.cost, 1.0 + step * np.arange(6))
    assert np.all(np.diff(h.best) < 0)
    assert h.total_cost == total and (h.n_fev, h.n_gev) == (6, 6)
    assert res.solve_costs(1e-3)[0, 0] == t_expected
    assert res.solve_costs(0.5)[0, 0] == 1.0 + step  # k = 1: 1e-2 ≤ 0.5
    assert res.solve_costs(1e-11)[0, 0] == INF  # k = 5 reaches 1e-10 only
    assert h.best_at(0.5) == INF and h.best_at(1.0) == 43.0


def test_runner_counts_agree_with_solver_counts() -> None:
    """The recorder (oracle: each method's own Counted wrappers) sees the same evaluations."""
    res = bench.run_benchmark(
        ["bfgs", "nelder_mead", "gauss_newton"],
        ["rosenbrock_ls"],
        budget=10_000,
        cost="nfev+n*ngev",
    )
    prob = problems.get("rosenbrock_ls")
    for label in res.labels:
        r = numopt.run(label, prob)
        h = res.history(label, "rosenbrock_ls")
        assert h.stopped == "solver"
        assert (h.n_fev, h.n_gev, h.n_hev) == (r.n_fev, r.n_gev, r.n_hev)
        assert h.total_cost == r.n_fev + prob.dim * r.n_gev
        assert h.converged == r.converged
        assert r.fun is not None and h.best[-1] <= r.fun


def test_budget_truncates_and_history_is_monotone() -> None:
    res = bench.run_benchmark(
        ["gradient_descent", "nelder_mead"],
        ["rosenbrock"],
        budget=101,
        cost="nfev+n*ngev",
        params={"gradient_descent": {"max_iter": 100_000}},
    )
    for label in res.labels:
        h = res.history(label, "rosenbrock")
        assert h.stopped == "budget" and not h.converged
        assert h.total_cost <= 101
        assert np.all(np.diff(h.best) < 0) and np.all(np.diff(h.cost) > 0)
        assert h.cost[-1] <= h.total_cost
        assert h.total_cost == h.n_fev + 2 * h.n_gev


def test_dominated_solver_in_a_real_benchmark() -> None:
    cases = [bench.BenchmarkCase("rosenbrock"), bench.BenchmarkCase("beale", x0s=[[1, 1], [2, 0]])]
    kw: dict[str, Any] = {"budget": 3000, "cost": "nfev+n*ngev"}
    base = bench.run_benchmark(["bfgs", "nelder_mead"], cases, **kw)
    more = bench.run_benchmark(["bfgs", "nelder_mead", ("lazy", _lazy)], cases, **kw)
    assert [i.f_star for i in base.instances] == [i.f_star for i in more.instances]
    for tau in (1e-1, 1e-3, 1e-6):
        p1 = bench.performance_profile(base, tau)
        p2 = bench.performance_profile(more, tau, alphas=p1.x)
        assert_array_equal(p2.y[:2], p1.y)
        assert_array_equal(p2.solved[2], 0.0)
        d1 = bench.data_profile(base, tau)
        assert_array_equal(bench.data_profile(more, tau, kappas=d1.x).y[:2], d1.y)


def test_f_star_without_known_minimum_is_best_found() -> None:
    quad = Problem(
        id="quad_unknown", name="", latex="", f=_quad_f, dim=2, domain=(), grad=_quad_grad,
        x0=[3.0, 1.0],
    )  # fmt: skip
    res = bench.run_benchmark(["bfgs", "nelder_mead"], [quad], budget=500)
    (inst,) = res.instances
    assert inst.f_known is None
    best = min(res.history(s, inst.id).best[-1] for s in res.labels)
    assert inst.f_star == best
    # The best solver solves at every τ; f_L is never below what was found.
    T = res.solve_costs(1e-12)
    assert np.isfinite(T).any()


def test_seeds_stochastic_and_deterministic() -> None:
    res = bench.run_benchmark(
        ["bfgs", "cma_es"], ["himmelblau"], budget=1500, seeds=(0, 1), cost="nfev"
    )
    ids = [i.id for i in res.instances]
    assert ids == ["himmelblau@seed0", "himmelblau@seed1"]
    b0, b1 = (res.history("bfgs", i) for i in ids)
    assert_array_equal(b0.best, b1.best)
    c0, c1 = (res.history("cma_es", i) for i in ids)
    assert not np.array_equal(c0.best, c1.best)
    # Deterministic: the same benchmark twice gives identical JSON.
    again = bench.run_benchmark(
        ["bfgs", "cma_es"], ["himmelblau"], budget=1500, seeds=(0, 1), cost="nfev"
    )
    assert again.to_json() == res.to_json()


def test_crashing_and_budget_swallowing_solvers() -> None:
    def crash(problem: Problem, *, x0: Any) -> Result:
        problem.f(np.asarray(x0, dtype=float))
        raise RuntimeError("boom")

    def swallow(problem: Problem, *, x0: Any) -> Result:
        x = np.asarray(x0, dtype=float)
        for _ in range(50):
            try:
                problem.f(x)
            except Exception:
                pass
        return Result("swallow", x, None, True, "done", 0, trace=[Step(0, x, None)])

    res = bench.run_benchmark([("crash", crash), ("swallow", swallow), "bfgs"], [QUAD], budget=10)
    h = res.history("crash", "test_quad")
    assert h.stopped == "error" and "boom" in h.message and h.n_fev == 1
    h = res.history("swallow", "test_quad")
    assert h.stopped == "budget" and h.total_cost == 10 and not h.converged
    assert res.history("bfgs", "test_quad").n_fev >= 1


def test_runner_input_validation() -> None:
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs"], ["rosenbrock"], budget=10, cost="nfev+ngev")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs", "bfgs"], ["rosenbrock"], budget=10)
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs"], ["rosenbrock", "rosenbrock"], budget=10)
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs"], ["rosenbrock"], budget=0)
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs"], [bench.BenchmarkCase("rosenbrock", x0s=[[1.0]])], budget=10)
    with pytest.raises(ValueError):
        bench.run_benchmark(["bfgs"], ["rosenbrock"], budget=10, params={"lbfgs": {}})
    with pytest.raises(ValueError):
        bench.performance_profile(bench.run_benchmark(["bfgs"], ["rosenbrock"], budget=10), tau=1.0)


# --------------------------------------------------------------------------------------
# Serialization and plots
# --------------------------------------------------------------------------------------


def test_json_round_trip(tmp_path: Any) -> None:
    res = bench.run_benchmark(
        ["bfgs", "nelder_mead", ("lazy", _lazy)],
        [bench.BenchmarkCase("beale", x0s=[[1, 1], [-2, 2]]), "rosenbrock"],
        budget=400,
        cost="nfev+n*ngev",
        params={"nelder_mead": {"max_iter": 10_000}},
    )
    path = res.save(tmp_path / "bench.json")
    json.loads(path.read_text())  # strict JSON
    back = bench.BenchmarkResult.load(path)
    assert back.to_dict() == res.to_dict()
    assert back.labels == res.labels and back.instances == res.instances
    for key, h in res.runs.items():
        assert_array_equal(back.runs[key].cost, h.cost)
        assert_array_equal(back.runs[key].best, h.best)
    for tau in (1e-1, 1e-4):
        assert_array_equal(back.solve_costs(tau), res.solve_costs(tau))  # ∞ survives
        assert_array_equal(
            bench.performance_profile(back, tau).y, bench.performance_profile(res, tau).y
        )
        assert_array_equal(bench.data_profile(back, tau).y, bench.data_profile(res, tau).y)
    assert np.isinf(res.solve_costs(1e-4)).any()
    with pytest.raises(ValueError):
        bench.BenchmarkResult.from_dict({"format": "other"})


def test_plots_write_png_and_svg(tmp_path: Any) -> None:
    pytest.importorskip("matplotlib")
    perf = bench.performance_profile_from_costs(HAND, ["A", "B", "C"], tau=1e-3)
    data = bench.data_profile_from_costs(HAND, HAND_N, ["A", "B", "C"], tau=1e-3)
    for suffix in ("png", "svg"):
        p1, p2 = tmp_path / f"perf.{suffix}", tmp_path / f"data.{suffix}"
        fig = bench.plot_performance_profile(perf, p1)
        bench.plot_data_profile(data, p2)
        assert p1.stat().st_size > 0 and p2.stat().st_size > 0
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    colors = [line.get_color() for line in ax.get_lines()]
    assert len(set(colors)) == 3
    with pytest.raises(ValueError):
        bench.plot_data_profile(perf)
    from matplotlib.figure import Figure

    outer = Figure()
    sub_ax = outer.add_subfigure(outer.add_gridspec(1, 2)[0, 1]).add_subplot()
    assert bench.plot_data_profile(data, ax=sub_ax) is outer  # an Axes in a SubFigure


def test_cli_bench(tmp_path: Any, capsys: Any) -> None:
    out = tmp_path / "b.json"
    code = cli_main(
        ["bench", "booth", "beale", "--methods", "bfgs", "nelder_mead", "--budget", "500",
         "--tau", "1e-3", "--set", "nelder_mead:max_iter=1000", "--json", str(out)]
    )  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "bfgs" in text and "nelder_mead" in text
    res = bench.BenchmarkResult.load(out)
    assert res.params == {"nelder_mead": {"max_iter": 1000}}
    assert [i.id for i in res.instances] == ["booth", "beale"]
