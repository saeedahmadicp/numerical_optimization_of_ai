"""Tests for numopt.unconstrained.global_ (SA, PSO, DE, CMA-ES, basin hopping)."""

from __future__ import annotations

import inspect
import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.optimize as so
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import numopt
from numopt import problems
from numopt.core.rng import Rng
from numopt.core.types import Problem
from numopt.unconstrained import global_ as glob
from numopt.unconstrained.derivative_free import nelder_mead

METHODS = (
    "simulated_annealing",
    "particle_swarm",
    "differential_evolution",
    "cma_es",
    "basin_hopping",
)
BOXED = ("simulated_annealing", "particle_swarm", "differential_evolution")
SEEDS = range(20)


def _val(v: float | None) -> float:
    """Narrow an optional objective value (Step.fun / Result.fun) to float."""
    assert v is not None
    return v


def _global_dist(prob, x) -> float:
    """∞-distance from x to the nearest *global* minimizer listed by the problem."""
    n_global = prob.extra["n_global"]
    return min(float(np.max(np.abs(np.asarray(x) - np.asarray(m)))) for m in prob.minima[:n_global])


def _box(prob) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([p[0] for p in prob.domain], dtype=float)
    hi = np.array([p[1] for p in prob.domain], dtype=float)
    return lo, hi


def _sphere_box(f, x0=(1.0, -1.0), box=((-5.0, 5.0), (-5.0, 5.0))) -> Problem:
    return Problem(id="test_box", name="test", latex="", f=f, dim=len(x0), domain=box, x0=x0)


# --------------------------------------------------------------------------------------
# Contract checks shared by every method
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_registry_metadata(method):
    spec = numopt.get_method(method)
    assert spec.family == "global" and spec.deterministic is False and spec.needs == ("f",)
    sig = inspect.signature(spec.fn)
    assert sig.parameters["seed"].default == 0
    declared = {p.name for p in spec.params}
    assert declared == set(sig.parameters) - {"problem", "x0", "seed"}
    for p in spec.params:
        assert sig.parameters[p.name].default == p.default


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rastrigin", "ackley", "himmelblau", "six_hump_camel"])
def test_contract(method, pid):
    prob = problems.get(pid)
    res = numopt.run(method, prob, seed=3)
    assert_valid_result(res, max_iter=numopt.get_method(method).defaults()["max_iter"])
    assert res.n_iter == res.trace[-1].k
    assert_allclose(res.x, res.trace[-1].x, rtol=0, atol=0)
    assert res.fun == res.trace[-1].fun == res.trace[-1].info["best_f"]
    assert res.n_gev == 0 and res.n_hev == 0
    # Step.x / Step.fun are the best point so far: f is consistent and never increases.
    funs = [_val(s.fun) for s in res.trace]
    assert all(b <= a for a, b in pairwise(funs))
    for s in res.trace:
        assert_allclose(s.x, s.info["best"], rtol=0, atol=0)
        assert s.fun == pytest.approx(float(prob.f(np.asarray(s.x))), rel=0, abs=0)


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_same_trace_other_seed_other_trace(method):
    prob = problems.get("himmelblau")
    a = numopt.run(method, prob, seed=11, max_iter=15)
    b = numopt.run(method, prob, seed=11, max_iter=15)
    c = numopt.run(method, prob, seed=12, max_iter=15)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict()["trace"] != c.to_dict()["trace"]


@pytest.mark.parametrize("method", METHODS)
def test_reports_max_iter(method):
    res = numopt.run(method, problems.get("rastrigin"), seed=0, max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4


@pytest.mark.parametrize("method", METHODS)
def test_minus_infinity_means_unbounded(method):
    prob = _sphere_box(
        lambda x: -math.inf if x[0] > 0.0 else float(x[0] ** 2 + x[1] ** 2), x0=(-1.0, 0.5)
    )
    res = numopt.run(method, prob, seed=0)
    assert_valid_result(res)
    assert not res.converged and "unbounded" in res.message
    assert res.fun == -math.inf


@pytest.mark.parametrize("method", METHODS)
def test_needs_a_search_box(method):
    with pytest.raises(ValueError, match="search box"):
        numopt.minimize(lambda x: x[0] ** 2 + x[1] ** 2, x0=[1.0, 1.0], method=method)


@pytest.mark.parametrize("method", BOXED)
def test_start_outside_the_box_is_rejected(method):
    with pytest.raises(ValueError, match="outside"):
        numopt.run(method, problems.get("himmelblau"), x0=[6.0, 0.0])


@pytest.mark.parametrize("method", METHODS)
def test_one_dimensional_problem_gets_a_float_in_every_method(method):
    """Regression: basin hopping (through its local Nelder–Mead) passed f the float x[0] of a
    dim-1 Problem, the other four methods a length-1 array, so a 1-D f written for one
    convention crashed in the others ("'float' object is not subscriptable"). Every method
    now follows the scalar convention of core.types, and n_fev counts every call."""
    seen: list[type] = []

    def f(x: float) -> float:
        seen.append(type(x))
        return float(x * x)

    prob = _sphere_box(f, x0=(1.5,), box=((-2.0, 2.0),))
    res = numopt.run(method, prob, seed=0, max_iter=5)
    assert_valid_result(res)
    assert seen and set(seen) == {float}
    assert res.n_fev == len(seen)


@pytest.mark.parametrize("method", METHODS)
def test_one_dimensional_box(method):
    # A dim-1 Problem follows the scalar convention of core.types: f takes the float x.
    prob = _sphere_box(
        lambda x: float((x - 1.0) ** 2 + 0.5 * math.sin(5.0 * x) ** 2),
        x0=(-2.0,),
        box=((-3.0, 3.0),),
    )
    res = numopt.run(method, prob, seed=1)
    assert_valid_result(res)
    assert res.x.shape == (1,)
    # The global minimizer of (x − 1)² + ½ sin²(5x) on [−3, 3], by a dense grid + Brent polish.
    grid = np.linspace(-3.0, 3.0, 200_001)
    xg = grid[np.argmin([(t - 1.0) ** 2 + 0.5 * math.sin(5.0 * t) ** 2 for t in grid])]
    ref = so.minimize_scalar(
        lambda t: (t - 1.0) ** 2 + 0.5 * math.sin(5.0 * t) ** 2,
        bracket=(xg - 1e-4, xg, xg + 1e-4),
        tol=1e-12,
    )
    x_ref = float(ref.x)  # pyright: ignore[reportAttributeAccessIssue]
    # NOTE: SA resolves x only to its final proposal standard deviation
    # σ = step·(hi − lo)·√(T_final/T₀) ≈ 6e-3; the other methods to their xtol.
    sa = method == "simulated_annealing"
    tol = float(np.max(res.trace[-1].info["proposal_sd"])) if sa else 1e-5
    assert abs(res.x[0] - x_ref) <= tol


@pytest.mark.parametrize("method", METHODS)
def test_fixture_cases_are_valid(method):
    cases = [c for c in glob.FIXTURE_CASES if c[0] == method]
    assert cases
    for mid, pid, params in cases:
        res = numopt.run(mid, problems.get(pid), **params)
        assert_valid_result(res)
        assert res.converged, (mid, pid, res.message)
        assert len(res.trace) < 300


# --------------------------------------------------------------------------------------
# Finding global minima (fixed seeds)
# --------------------------------------------------------------------------------------


def _hits(method: str, pid: str, tol: float, **kw: Any) -> tuple[int, list[Any]]:
    prob = problems.get(pid)
    results = [numopt.run(method, prob, seed=s, **kw) for s in SEEDS]
    return sum(_global_dist(prob, r.x) <= tol for r in results), results


# NOTE: 1e-5 bounds the localization error of the collapse tests (xtol = 1e-6 for PSO/DE),
# of CMA-ES's TolX and of the local Nelder–Mead (xtol = 1e-8) of basin hopping.
@pytest.mark.parametrize(
    "method,pid,kw,min_hits",
    [
        ("particle_swarm", "rastrigin", {}, 20),
        ("particle_swarm", "ackley", {}, 20),
        ("particle_swarm", "himmelblau", {}, 20),
        ("differential_evolution", "rastrigin", {}, 20),
        ("differential_evolution", "ackley", {}, 20),
        ("differential_evolution", "himmelblau", {}, 20),
        ("differential_evolution", "ackley", {"strategy": "best/1/bin"}, 20),
        # best/1 is greedy; one seed of 20 settles on a local minimum (see the test below).
        ("differential_evolution", "himmelblau", {"strategy": "best/1/bin"}, 19),
        ("cma_es", "ackley", {}, 20),
        ("cma_es", "himmelblau", {}, 20),
        # Hansen (2016), §B.4: multimodal functions such as Rastrigin need a large λ.
        ("cma_es", "rastrigin", {"pop_size": 100}, 20),
        ("basin_hopping", "ackley", {}, 20),
        ("basin_hopping", "himmelblau", {}, 20),
        # A hop of ±0.1·width (±1.02) often lands in a neighbouring Rastrigin basin and is then
        # rejected; 3 of 20 seeds exhaust the patience of 20 hops on a ring of value ≈ 0.995.
        ("basin_hopping", "rastrigin", {}, 17),
    ],
)
def test_finds_global_minimum(method, pid, kw, min_hits):
    hits, results = _hits(method, pid, 1e-5, **kw)
    assert hits >= min_hits, [r.message for r in results]
    prob = problems.get(pid)
    for r in results:
        if _global_dist(prob, r.x) <= 1e-5:
            assert r.fun <= prob.extra["f_min"] + 1e-8


def test_differential_evolution_best_1_exploits_faster_but_finds_less_on_rastrigin():
    """DE/best/1 mutates around the current best: fewer generations, more premature
    convergence on a multimodal function than DE/rand/1 (Price, Storn & Lampinen 2005, §2.6)."""
    prob = problems.get("rastrigin")
    rand_hits, rand_res = _hits("differential_evolution", prob.id, 1e-5)
    best_hits, best_res = _hits("differential_evolution", prob.id, 1e-5, strategy="best/1/bin")
    assert all(r.converged for r in rand_res + best_res)
    assert np.median([r.n_iter for r in best_res]) < np.median([r.n_iter for r in rand_res])
    assert best_hits < rand_hits
    # The runs that miss stop at a local minimum: a value of the grid ≈ 0.995·(i² + j²).
    for r in best_res:
        assert min(abs(r.fun - 0.9949590570932898 * q) for q in (0, 1, 2, 4, 5)) <= 1e-2


@pytest.mark.parametrize("pid", ["himmelblau", "ackley"])
def test_simulated_annealing_ends_in_the_global_basin(pid):
    prob = problems.get(pid)
    results = [numopt.run("simulated_annealing", prob, seed=s) for s in SEEDS]
    assert all(r.converged and "frozen" in r.message for r in results)
    # The chain's resolution at freezing is the final proposal standard deviation
    # σ = step·(hi − lo)·√(T_final/T₀) ≈ 0.01; the best point is within it of a global minimum.
    hits = sum(
        _global_dist(prob, r.x) <= float(np.max(r.trace[-1].info["proposal_sd"])) for r in results
    )
    assert hits >= 18


def test_simulated_annealing_escapes_the_start_basin_on_rastrigin():
    prob = problems.get("rastrigin")
    local = nelder_mead(prob)  # the local minimum of the basin of x0 = (3.3, −2.6)
    f_local = _val(local.fun)
    assert local.converged and f_local > 10.0
    for s in SEEDS:
        res = numopt.run("simulated_annealing", prob, seed=s)
        assert _val(res.fun) < f_local - 5.0


def test_cma_es_default_population_converges_to_local_minima_of_rastrigin():
    """converged=True certifies a settled search, not a global minimum (module docstring)."""
    prob = problems.get("rastrigin")
    results = [numopt.run("cma_es", prob, seed=s) for s in SEEDS]
    assert all(r.converged for r in results)
    assert sum(_global_dist(prob, r.x) <= 1e-5 for r in results) <= 2


@pytest.mark.parametrize(
    "method,oracle",
    [
        ("differential_evolution", "differential_evolution"),
        ("particle_swarm", "dual_annealing"),
        ("basin_hopping", "basinhopping"),
    ],
)
@pytest.mark.parametrize("pid", ["rastrigin", "ackley", "six_hump_camel"])
def test_agrees_with_scipy_global_optimizers(method, oracle, pid):
    prob = problems.get(pid)
    lo, hi = _box(prob)
    bounds = list(zip(lo, hi, strict=True))

    def oracle_run(r: int) -> Any:
        if oracle == "differential_evolution":
            return so.differential_evolution(prob.f, bounds, rng=r, tol=1e-12, polish=True)
        if oracle == "dual_annealing":
            return so.dual_annealing(prob.f, bounds, rng=r)
        return so.basinhopping(prob.f, np.asarray(prob.x0, dtype=float), niter=200, rng=r)

    # NOTE: the oracles are stochastic too (SciPy's default best1bin DE misses the Rastrigin
    # optimum for 2 of rng = 0…4), so the reference is the best of five SciPy runs.
    ref = min((oracle_run(r) for r in range(5)), key=lambda o: float(o.fun))
    res = numopt.run(method, prob, seed=0)
    # Both reach a global minimizer (six_hump_camel has two, ±; the nearest one counts).
    assert _global_dist(prob, ref.x) <= 1e-5 and _global_dist(prob, res.x) <= 1e-5
    # NOTE: f is Lipschitz near the minimizer with |∇f| ≲ 4 (Ackley's cone: f ≈ 2√2‖x‖₂), so
    # a distance of 1e-5 allows f − f_min ≤ 1e-4; SciPy's FD-gradient polish stops at f ≈ 3e-8
    # on Ackley's kink, so the values are compared at that resolution.
    f_min = prob.extra["f_min"]
    assert f_min <= res.fun <= f_min + 1e-4 and f_min <= ref.fun <= f_min + 1e-4


# --------------------------------------------------------------------------------------
# Exact evaluation counts
# --------------------------------------------------------------------------------------


def test_evaluation_counts():
    prob = problems.get("rastrigin")
    sa = numopt.run("simulated_annealing", prob, seed=2)
    inside = sum(1 for s in sa.trace[1:] if s.info["inside"])
    assert sa.n_fev == 1 + inside
    assert all((s.info["candidate_f"] is None) == (not s.info["inside"]) for s in sa.trace[1:])

    pso = numopt.run("particle_swarm", prob, seed=2, n_particles=9)
    assert pso.n_fev == 9 * (pso.n_iter + 1)

    de = numopt.run("differential_evolution", prob, seed=2, pop_size=11)
    assert de.n_fev == 11 * (de.n_iter + 1)

    cma = numopt.run("cma_es", prob, seed=2)
    lam = 4 + math.floor(3 * math.log(2))
    assert cma.n_fev == 1 + lam * cma.n_iter
    assert all(len(s.info["population"]) == lam for s in cma.trace[1:])

    bh = numopt.run("basin_hopping", prob, seed=2)
    s = 0.1 * 10.24
    local = [
        nelder_mead(
            prob,
            x0=step_.info["start"],
            xtol=glob._LOCAL_XTOL,
            ftol=glob._LOCAL_FTOL,
            initial_step=0.5 * s,
            max_iter=glob._LOCAL_MAX_ITER,
        )
        for step_ in bh.trace
    ]
    assert bh.n_fev == sum(r.n_fev for r in local)
    for step_, r in zip(bh.trace, local, strict=True):
        assert_allclose(step_.info["local_min"], r.x, rtol=0, atol=0)
        assert step_.info["local_f"] == r.fun


# --------------------------------------------------------------------------------------
# Independent replays of the documented formulas and draw order
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("cooling", ["geometric", "logarithmic"])
def test_simulated_annealing_replay(cooling):
    prob = problems.get("himmelblau")
    seed, T0, alpha, step = 4, 10.0, 0.97, 0.1
    res = numopt.run(
        "simulated_annealing", prob, seed=seed, T0=T0, alpha=alpha, cooling=cooling, max_iter=150
    )
    lo, hi = _box(prob)
    rng = Rng(seed)
    x = np.asarray(prob.x0, dtype=float)
    fx = float(prob.f(x))
    for k, s in enumerate(res.trace[1:], start=1):
        T = T0 * alpha ** (k - 1) if cooling == "geometric" else T0 * math.log(2) / math.log(k + 1)
        z = np.array([rng.normal() for _ in range(2)])
        u = rng.random()
        sd = step * (hi - lo) * math.sqrt(T / T0)
        y = x + sd * z
        assert s.info["temperature"] == pytest.approx(T, rel=1e-15)
        assert_allclose(s.info["candidate"], y, rtol=1e-15, atol=0)
        if np.all((y >= lo) & (y <= hi)):
            fy = float(prob.f(y))
            p = min(1.0, math.exp(-(fy - fx) / T))
            assert s.info["accept_prob"] == pytest.approx(p, rel=1e-12)
            if u < p:
                x, fx = y, fy
        else:
            assert s.info["accepted"] is False and s.info["candidate_f"] is None
        assert_allclose(s.info["current"], x, rtol=1e-15, atol=0)
    if cooling == "logarithmic":
        # T_k = T₀ ln 2/ln(k + 1) ≤ T_min ⇔ k ≥ 2^(T₀/T_min) − 1: here 2^(10⁴) steps.
        assert not res.converged and "max_iter" in res.message


def test_particle_swarm_replay():
    prob = problems.get("ackley")
    seed, N, w, c1, c2 = 9, 6, 0.6, 1.7, 1.3
    res = numopt.run(
        "particle_swarm", prob, seed=seed, n_particles=N, w=w, c1=c1, c2=c2, max_iter=12
    )
    lo, hi = _box(prob)
    n = 2
    rng = Rng(seed)
    X = [np.asarray(prob.x0, dtype=float)]
    X += [
        np.array([lo[j] + (hi[j] - lo[j]) * rng.random() for j in range(n)]) for _ in range(N - 1)
    ]
    V = [
        0.5 * (np.array([lo[j] + (hi[j] - lo[j]) * rng.random() for j in range(n)]) - x) for x in X
    ]
    P = [x.copy() for x in X]
    Pf = [float(prob.f(x)) for x in X]
    for s in res.trace:
        if s.k > 0:
            g = P[int(np.argmin(Pf))]
            for i in range(N):
                r1 = np.array([rng.random() for _ in range(n)])
                r2 = np.array([rng.random() for _ in range(n)])
                v = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (g - X[i])
                x = X[i] + v
                out = (x < lo) | (x > hi)
                X[i], V[i] = np.clip(x, lo, hi), np.where(out, 0.0, v)
            for i in range(N):
                fi = float(prob.f(X[i]))
                if fi < Pf[i]:
                    P[i], Pf[i] = X[i].copy(), fi
        assert_allclose(s.info["particles"], X, rtol=1e-14, atol=1e-15)
        assert_allclose(s.info["velocities"], V, rtol=1e-14, atol=1e-15)
        assert_allclose(s.info["personal_best"], P, rtol=1e-14, atol=1e-15)
        assert_allclose(s.info["global_best"], P[int(np.argmin(Pf))], rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("strategy", ["rand/1/bin", "best/1/bin"])
def test_differential_evolution_replay(strategy):
    prob = problems.get("rastrigin")
    seed, NP, F, CR = 5, 7, 0.7, 0.6
    res = numopt.run(
        "differential_evolution",
        prob,
        seed=seed,
        pop_size=NP,
        F=F,
        CR=CR,
        strategy=strategy,
        max_iter=10,
    )
    lo, hi = _box(prob)
    n = 2
    rng = Rng(seed)

    def pick(exclude: list[int]) -> int:
        while True:
            r = min(int(rng.random() * NP), NP - 1)
            if r not in exclude:
                return r

    pop = [np.asarray(prob.x0, dtype=float)]
    pop += [
        np.array([lo[j] + (hi[j] - lo[j]) * rng.random() for j in range(n)]) for _ in range(NP - 1)
    ]
    fpop = [float(prob.f(x)) for x in pop]
    for s in res.trace[1:]:
        best = pop[int(np.argmin(fpop))]
        new, fnew = [x.copy() for x in pop], list(fpop)
        for i in range(NP):
            r1 = pick([i])
            r2 = pick([i, r1])
            if strategy == "rand/1/bin":
                r3 = pick([i, r1, r2])
                v = pop[r1] + F * (pop[r2] - pop[r3])
            else:
                v = best + F * (pop[r1] - pop[r2])
            j_rand = min(int(rng.random() * n), n - 1)
            U = [rng.random() for _ in range(n)]
            u = np.array([v[j] if (U[j] <= CR or j == j_rand) else pop[i][j] for j in range(n)])
            for j in range(n):
                if u[j] < lo[j]:
                    u[j] = 0.5 * (lo[j] + pop[i][j])
                elif u[j] > hi[j]:
                    u[j] = 0.5 * (hi[j] + pop[i][j])
            fu = float(prob.f(u))
            assert_allclose(s.info["mutants"][i], v, rtol=1e-15, atol=1e-15)
            assert_allclose(s.info["trials"][i], u, rtol=1e-15, atol=1e-15)
            assert s.info["accepted"][i] == (fu <= fpop[i])
            if fu <= fpop[i]:
                new[i], fnew[i] = u, fu
        pop, fpop = new, fnew
        assert_allclose(s.info["population"], pop, rtol=1e-15, atol=1e-15)
        assert s.info["best_index"] == int(np.argmin(fpop))


def test_cma_es_replay_against_the_tutorial_equations():
    """Replay Hansen (2016) eqs. (38)–(47) with scipy.linalg.sqrtm as the oracle for C^{±1/2}."""
    prob = problems.get("ackley")
    seed = 21
    res = numopt.run("cma_es", prob, seed=seed, max_iter=60)
    n = 2
    lam = 4 + math.floor(3 * math.log(n))
    mu = lam // 2
    w = np.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w**2)
    cs = (mueff + 2) / (n + mueff + 5)
    ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (0.25 + mueff + 1 / mueff - 2) / ((n + 2) ** 2 + mueff))
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    # Table 1 at n = 2: λ = 6, μ = 3, μ_eff ≈ 2.0286 (values printed by Hansen's pycma).
    assert (lam, mu) == (6, 3) and mueff == pytest.approx(2.0286, abs=1e-4)

    rng = Rng(seed)
    m = np.asarray(prob.x0, dtype=float)
    sigma = 0.3 * 10.0
    C = np.eye(n)
    ps, pc = np.zeros(n), np.zeros(n)
    for g, s in enumerate(res.trace[1:], start=1):
        info = s.info
        assert_allclose(info["sample_mean"], m, rtol=1e-11, atol=1e-13)
        assert info["sample_sigma"] == pytest.approx(sigma, rel=1e-11)
        assert_allclose(info["sample_covariance"], C, rtol=1e-10, atol=1e-13)
        # Use the method's own (m, σ, C) from here on so that rounding does not accumulate.
        m, sigma = np.asarray(info["sample_mean"]), info["sample_sigma"]
        C = np.asarray(info["sample_covariance"])
        root = np.real(sla.sqrtm(C))
        Z = np.array([[rng.normal() for _ in range(n)] for _ in range(lam)])
        X = m + sigma * Z @ root.T  # (λ, n)
        assert_allclose(info["population"], X, rtol=1e-10, atol=1e-12 * sigma)
        fX = [float(prob.f(x)) for x in np.asarray(info["population"])]
        assert info["population_f"] == fX
        sel = list(np.argsort(fX, kind="stable")[:mu])
        assert info["selected"] == sel
        Y = (np.asarray(info["population"])[sel] - m) / sigma  # (μ, n)
        yw = w @ Y
        m = m + sigma * yw
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * np.linalg.solve(root, yw)
        hs = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * g)) < (1.4 + 2 / (n + 1)) * chi_n
        pc = (1 - cc) * pc + hs * math.sqrt(cc * (2 - cc) * mueff) * yw
        dh = (1 - hs) * cc * (2 - cc)
        C = (1 + c1 * dh - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (Y.T * w) @ Y
        sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
        assert info["h_sigma"] == hs
        assert_allclose(info["mean"], m, rtol=1e-10, atol=1e-12 * sigma)
        assert_allclose(info["p_sigma"], ps, rtol=1e-8, atol=1e-10)
        assert_allclose(info["p_c"], pc, rtol=1e-8, atol=1e-10)
        assert info["sigma"] == pytest.approx(sigma, rel=1e-10)
        assert_allclose(info["covariance"], C, rtol=1e-9, atol=1e-12 * np.max(np.abs(C)))
        ps, pc = np.asarray(info["p_sigma"]), np.asarray(info["p_c"])


def test_cma_es_on_an_ill_conditioned_quadratic():
    """CMA-ES learns the Hessian shape: TolX convergence to the minimizer and C ∝ H⁻¹."""
    theta = math.pi / 6.0
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    H = (R * np.array([1000.0, 1.0])) @ R.T  # eigenvalues 1e3 and 1, axes rotated by 30°
    assert np.linalg.cond(H) == pytest.approx(1e3, rel=1e-12)
    c = np.array([0.7, -1.2])
    prob = _sphere_box(lambda x: 0.5 * float((x - c) @ H @ (x - c)), x0=(3.0, 3.0))
    res = numopt.run("cma_es", prob, seed=0)
    assert res.converged and ("TolX" in res.message or "TolFun" in res.message)
    # On a quadratic, ‖x − c‖₂ ≤ √(2 f(x)/λ_min(H)) exactly; f(best) ≤ 1e-12 gives ≲ 1.5e-6.
    lam_min = float(np.linalg.eigvalsh(H)[0])
    f_best = _val(res.fun)
    assert f_best <= 1e-12
    assert np.linalg.norm(res.x - c) <= math.sqrt(2.0 * f_best / lam_min) * (1 + 1e-8) + 1e-15
    C = np.asarray(res.trace[-1].info["covariance"])
    Hinv = np.linalg.solve(H, np.eye(2))
    # Directions agree: the cosine of the angle between C and H⁻¹ (Frobenius) is near 1.
    cos = float(np.sum(C * Hinv) / (np.linalg.norm(C) * np.linalg.norm(Hinv)))
    assert cos > 0.95


def test_basin_hopping_replay():
    prob = problems.get("rastrigin")
    seed, T, step = 8, 1.5, 0.15
    res = numopt.run("basin_hopping", prob, seed=seed, T=T, step=step, max_iter=12)
    lo, hi = _box(prob)
    s_box = step * (hi - lo)
    rng = Rng(seed)
    x = np.asarray(res.trace[0].info["current"])
    fx = res.trace[0].info["current_f"]
    for s in res.trace[1:]:
        u = np.array([rng.random() for _ in range(2)])
        u_acc = rng.random()
        y = x + s_box * (2.0 * u - 1.0)
        assert_allclose(s.info["start"], y, rtol=1e-15, atol=0)
        fz = s.info["local_f"]
        p = min(1.0, math.exp(-(fz - fx) / T))
        assert s.info["accept_prob"] == pytest.approx(p, rel=1e-12)
        assert s.info["accepted"] == (u_acc < p)
        if u_acc < p:
            x, fx = np.asarray(s.info["local_min"]), fz
        assert_allclose(s.info["current"], x, rtol=0, atol=0)
    # The minima list holds distinct minima whose hit counts add up to the number of searches.
    minima = res.trace[-1].info["minima"]
    assert sum(r["hits"] for r in minima) == len(res.trace)


def test_basin_hopping_stall_counter():
    res = numopt.run("basin_hopping", problems.get("himmelblau"), seed=0, patience=5)
    assert res.converged and "5 hops" in res.message
    stall = 0
    for prev, s in pairwise(res.trace):
        f_best = prev.info["best_f"]
        improved = s.info["local_f"] < f_best - 1e-8 * (1.0 + abs(f_best))
        stall = 0 if improved else stall + 1
        assert s.info["stall"] == stall
    assert stall == 5


def _best_local_converged(res) -> bool:
    """Replay which hop produced the best minimum; return its ``local_converged`` flag."""
    ok = res.trace[0].info["local_converged"]
    for prev, s in pairwise(res.trace):
        if s.info["local_f"] < prev.info["best_f"]:
            ok = s.info["local_converged"]
    return ok


@pytest.mark.parametrize("pid", ["himmelblau", "rastrigin", "ackley"])
@pytest.mark.parametrize("seed", range(6))
def test_basin_hopping_converged_iff_the_best_local_search_converged(pid, seed):
    """A stall stop certifies the best point only if its own local search converged."""
    res = numopt.run("basin_hopping", problems.get(pid), seed=seed, patience=5)
    assert_valid_result(res, max_iter=100)
    if "max_iter" not in res.message:
        assert res.converged == _best_local_converged(res)


def test_basin_hopping_stall_without_converged_local_searches_is_not_convergence(monkeypatch):
    """Regression: with every local search cut off, the stall stop must report converged=False."""
    monkeypatch.setattr(glob, "_LOCAL_MAX_ITER", 5)
    res = numopt.run("basin_hopping", problems.get("himmelblau"), seed=0, patience=5)
    assert_valid_result(res, max_iter=100)
    assert not any(s.info["local_converged"] for s in res.trace)
    assert "unchanged for 5 hops" in res.message
    assert not res.converged and "not a converged local minimum" in res.message


def test_basin_hopping_on_quadratic_nd_agrees_with_scipy_success_flag():
    """Audit case: n = 20, Nelder–Mead needs > 2000 iterations per local search. The stall rule
    fires, but no local search converged; SciPy's basinhopping reports success=False here too."""
    prob = problems.get("quadratic_nd")
    res = numopt.run("basin_hopping", prob, seed=0, patience=3)
    assert_valid_result(res, max_iter=100)
    assert "unchanged for 3 hops" in res.message
    assert not res.converged
    assert float(np.linalg.norm(prob.grad(res.x))) > 1e-3  # truly not a stationary point
    ref = so.basinhopping(
        prob.f,
        np.asarray(prob.x0, dtype=float),
        niter=100,
        niter_success=3,
        minimizer_kwargs={
            "method": "Nelder-Mead",
            "options": {"maxiter": glob._LOCAL_MAX_ITER, "xatol": 1e-8, "fatol": 1e-10},
        },
        rng=0,
    )
    assert not ref.success


# --------------------------------------------------------------------------------------
# Failure paths and parameter validation
# --------------------------------------------------------------------------------------


def test_particle_swarm_stagnation_is_reported_honestly():
    """A particle whose personal best lies in another basin keeps oscillating between p_i and
    g (Clerc & Kennedy 2002), so the swarm never collapses: converged=False at max_iter, even
    though g is a global minimizer."""
    prob = problems.get("himmelblau")
    res = numopt.run("particle_swarm", prob, seed=12)
    assert not res.converged and "max_iter" in res.message
    assert _global_dist(prob, res.x) <= 1e-8
    info = res.trace[-1].info
    g = np.asarray(info["global_best"])
    far = [p for p in info["personal_best"] if np.max(np.abs(np.asarray(p) - g)) > 1.0]
    assert far  # some particle remembers another basin


@pytest.mark.parametrize(
    "method,kw",
    [
        ("simulated_annealing", {"alpha": 1.0}),
        ("simulated_annealing", {"cooling": "linear"}),
        ("simulated_annealing", {"T0": 0.0}),
        ("particle_swarm", {"n_particles": 1}),
        ("particle_swarm", {"w": -0.1}),
        ("differential_evolution", {"pop_size": 3}),
        ("differential_evolution", {"strategy": "rand/2/exp"}),
        ("differential_evolution", {"CR": 1.5}),
        ("cma_es", {"pop_size": -1}),
        ("cma_es", {"sigma0": 0.0}),
        ("basin_hopping", {"patience": 0}),
        ("basin_hopping", {"T": -1.0}),
    ],
)
def test_validates_parameters(method, kw):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("himmelblau"), **kw)


@pytest.mark.parametrize("max_iter", [0, -1, 2.5])
@pytest.mark.parametrize("method", METHODS)
def test_rejects_invalid_max_iter(method, max_iter):
    """Regression: the ``k == max_iter`` test never fires for these, so the limit was ignored."""
    prob = _sphere_box(lambda x: -x[0] + x[1] ** 2, x0=(0.5, 0.5))
    with pytest.raises(ValueError, match="max_iter"):
        numopt.run(method, prob, max_iter=max_iter)


def test_best_1_bin_needs_three_members():
    res = numopt.run("differential_evolution", problems.get("himmelblau"), pop_size=3,
                     strategy="best/1/bin", max_iter=5)  # fmt: skip
    assert_valid_result(res)


def test_nonfinite_start_of_simulated_annealing():
    prob = _sphere_box(lambda x: math.nan, x0=(0.0, 0.0))
    res = numopt.run("simulated_annealing", prob)
    assert_valid_result(res)
    assert not res.converged and "not finite" in res.message and res.n_fev == 1


def test_barrier_region_is_never_accepted():
    # f is NaN (→ +∞) outside the disc ‖x‖ < 3; the minimizer (1, 1) is inside.
    def f(x):
        if x[0] ** 2 + x[1] ** 2 >= 9.0:
            return math.nan
        return float((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)

    prob = _sphere_box(f, x0=(0.0, 0.0))
    for method in METHODS:
        res = numopt.run(method, prob, seed=0)
        assert_valid_result(res)
        assert math.isfinite(_val(res.fun)) and np.hypot(*res.x) < 3.0
        tol = 1e-2 if method == "simulated_annealing" else 1e-5
        assert_allclose(res.x, [1.0, 1.0], atol=tol)


# --------------------------------------------------------------------------------------
# Hypothesis invariants
# --------------------------------------------------------------------------------------

_PIDS = ("rastrigin", "ackley", "himmelblau", "six_hump_camel", "levi13")


@given(
    seed=st.integers(0, 2**32 - 1),
    pid=st.sampled_from(_PIDS),
    method=st.sampled_from(BOXED),
)
@settings(max_examples=1000, deadline=None)
def test_boxed_methods_stay_inside_the_box(seed, pid, method):
    prob = problems.get(pid)
    lo, hi = _box(prob)
    kw: dict[str, Any] = {"max_iter": 8}
    if method != "simulated_annealing":
        kw[{"particle_swarm": "n_particles", "differential_evolution": "pop_size"}[method]] = 6
    res = numopt.run(method, prob, seed=seed, **kw)

    def inside(p) -> bool:
        p = np.asarray(p)
        return bool(np.all((p >= lo) & (p <= hi)))

    for s in res.trace:
        assert inside(s.x)
        if method == "simulated_annealing":
            assert inside(s.info["current"])
        elif method == "particle_swarm":
            assert all(inside(p) for p in s.info["particles"])
            # Absorbing walls: a particle that left the box sits on the wall with zero velocity.
            for p, v in zip(s.info["particles"], s.info["velocities"], strict=True):
                on_wall = (np.asarray(p) == lo) | (np.asarray(p) == hi)
                assert np.all(np.asarray(v)[on_wall] == 0.0) or s.k == 0
        else:
            assert all(inside(p) for p in s.info["population"])
            assert all(inside(p) for p in s.info["trials"])


@given(seed=st.integers(0, 2**32 - 1), pid=st.sampled_from(_PIDS), method=st.sampled_from(METHODS))
@settings(max_examples=300, deadline=None)
def test_best_is_the_minimum_of_all_evaluated_values(seed, pid, method):
    prob = problems.get(pid)
    kw: dict[str, Any] = {"max_iter": 6}
    if method == "basin_hopping":
        kw["max_iter"] = 3
    res = numopt.run(method, prob, seed=seed, **kw)
    seen: list[float] = []
    for s in res.trace:
        info = s.info
        if method == "simulated_annealing":
            seen += [info["current_f"]] + (
                [info["candidate_f"]] if info["candidate_f"] is not None else []
            )
        elif method == "particle_swarm":
            seen += info["particles_f"]
        elif method == "differential_evolution":
            seen += info["population_f"] + info["trials_f"]
        elif method == "cma_es":
            seen += info["population_f"] + (
                [float(prob.f(np.asarray(prob.x0)))] if s.k == 0 else []
            )
        else:
            seen += [m["f"] for m in info["minima"]]
        assert s.fun == min(seen)


# --------------------------------------------------------------------------------------
# Regression tests for audit findings
# --------------------------------------------------------------------------------------

EPS = float(np.finfo(np.float64).eps)


def _offset_sphere(c: float) -> Problem:
    return Problem(
        id="offset_sphere",
        name="c + ‖x‖²",
        latex="",
        f=lambda x: c + float(x[0] ** 2 + x[1] ** 2),
        dim=2,
        domain=((-10.0, 10.0), (-10.0, 10.0)),
        x0=(3.0, -2.0),
    )


@pytest.mark.parametrize("method", ["particle_swarm", "differential_evolution"])
@pytest.mark.parametrize("c", [0.0, 3e4, 1e5, 1e6])
def test_population_methods_stop_on_the_rounding_plateau(method, c):
    """f = c + ‖x‖²: rounding makes f ≡ c on the ball ‖x‖² ≤ ulp(c)/2 ≤ εc/2, whose radius
    (4.7e-6 at c = 1e5) exceeds xtol = 1e-6. Before the plateau exit, PSO and DE never stopped
    for c ≥ 1e5 (0 of 10 seeds)."""
    prob = _offset_sphere(c)
    for seed in range(10):
        res = numopt.run(method, prob, seed=seed)
        assert_valid_result(res)
        assert res.converged, (seed, res.message)
        assert "collapsed" in res.message or "rounding plateau" in res.message
        # The best value is c to rounding, so the best point lies in the plateau ball.
        f_best = _val(res.fun)
        assert c <= f_best <= c * (1.0 + 2.0 * EPS) + 1e-8
        assert float(res.x @ res.x) <= max(EPS * c, f_best - c)
        if "rounding plateau" in res.message:
            # The documented test: over the last 10 generations every f is within 2ε|f_best|.
            key = "particles_f" if method == "particle_swarm" else "population_f"
            window = [max(s.info[key]) for s in res.trace[-glob._PLATEAU_GENERATIONS :]]
            assert max(window) - f_best <= 2.0 * EPS * abs(f_best)
    if c >= 1e5:
        # The plateau, not the collapse, ends these runs (the x-spread stays above xtol).
        assert "rounding plateau" in numopt.run(method, prob, seed=0).message


@pytest.mark.parametrize("method", ["particle_swarm", "differential_evolution"])
def test_constant_function_stops_on_the_plateau_after_ten_generations(method):
    res = numopt.run(method, _sphere_box(lambda x: 5.0), seed=0)
    assert_valid_result(res)
    assert res.converged and "rounding plateau" in res.message
    assert res.n_iter == glob._PLATEAU_GENERATIONS and res.fun == 5.0


@pytest.mark.parametrize(
    "T0,T_min",
    [(1.0, 0.1), (1.0, 0.2), (2.0, 0.5)],
)
def test_logarithmic_cooling_freezes_after_2_to_the_T0_over_Tmin_steps(T0, T_min):
    """T_k = T₀ ln 2/ln(k + 1) ≤ T_min ⇔ k ≥ 2^(T₀/T_min) − 1 (the audit found 2^(T₀ ln 2/T_min)
    in the docstring, which gives 122 instead of 1023 for T₀ = 1, T_min = 0.1)."""
    res = numopt.run(
        "simulated_annealing",
        problems.get("himmelblau"),
        cooling="logarithmic",
        T0=T0,
        T_min=T_min,
        max_iter=5000,
    )
    assert res.converged and "frozen" in res.message
    # Oracle: the first k whose temperature is ≤ T_min, by direct search.
    k_star = next(k for k in range(1, 10**6) if T0 * math.log(2.0) / math.log(k + 1.0) <= T_min)
    assert res.n_iter == k_star
    assert abs(k_star - (2.0 ** (T0 / T_min) - 1.0)) <= 1.0  # rounding at the boundary only
    assert any("Hajek (1988)" in r for r in numopt.get_method("simulated_annealing").references)


def test_cma_es_pop_size_zero_and_one_select_the_default_lambda():
    prob = problems.get("himmelblau")
    default = numopt.run("cma_es", prob, seed=4, max_iter=20)
    for lam in (0, 1):
        res = numopt.run("cma_es", prob, seed=4, max_iter=20, pop_size=lam)
        assert res.to_dict() == default.to_dict()
    assert len(default.trace[1].info["population"]) == 4 + math.floor(3 * math.log(2))


def test_cma_es_condition_cov_is_reported():
    """f = 10¹⁶x₁² + x₂²: C must reach κ ≈ 10¹⁶ to fit H⁻¹, so ConditionCov (κ(C) > 10¹⁴,
    Hansen 2016, App. B.3) stops the run before the tight TolX/TolFun can pass."""
    prob = _sphere_box(lambda x: float(1e16 * x[0] ** 2 + x[1] ** 2), x0=(3.0, -2.0))
    for seed in range(3):
        res = numopt.run("cma_es", prob, seed=seed, ftol=1e-15, xtol=1e-14, max_iter=5000)
        assert_valid_result(res, max_iter=5000)
        assert not res.converged and "ConditionCov" in res.message
        C = np.asarray(res.trace[-1].info["covariance"])
        assert np.linalg.cond(C) > 1e14


def test_cma_es_step_size_overflow_is_reported():
    """f = −log(1 + |x|) decreases without bound but is never −∞ at a finite x: the mean runs
    away, σ grows geometrically and overflows to +∞ (n = 1, so κ(C) = 1 cannot stop it)."""
    prob = Problem(
        id="runaway",
        name="runaway",
        latex="",
        f=lambda x: -math.log1p(abs(x)) if math.isfinite(x) else math.nan,  # dim 1: a float
        dim=1,
        domain=((-5.0, 5.0),),
        x0=(3.0,),
    )
    res = numopt.run("cma_es", prob, seed=0, max_iter=100_000)
    assert_valid_result(res)
    assert not res.converged and "non-finite" in res.message
    assert res.n_iter < 100_000 and math.isfinite(_val(res.fun))


def test_cma_es_loss_of_definiteness_is_reported(monkeypatch):
    """C stays SPD in exact arithmetic, so the guard is exercised with an eigensolver that
    reports λ_min(C) < 0 after the first update."""
    real_eigh = np.linalg.eigh
    calls = {"n": 0}

    def eigh(a, *args, **kwargs):
        w, v = real_eigh(a, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] >= 2:  # call 1 is C = I at the start
            w = w.copy()
            w[0] = -1e-3 * abs(w[-1])
        return w, v

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    res = numopt.run("cma_es", problems.get("himmelblau"), seed=0)
    assert_valid_result(res)
    assert not res.converged and "positive definiteness" in res.message
    assert res.n_iter == 1 and len(res.trace) == 2


@pytest.mark.parametrize("method", METHODS)
def test_every_value_in_the_param_spec_range_is_accepted(method):
    """A UI builds its controls from the ParamSpec; every value it offers must run."""
    spec = numopt.get_method(method)
    prob = problems.get("himmelblau")
    for p in spec.params:
        if p.name == "max_iter":
            values: list[Any] = [int(p.min)] if p.min is not None else []
        elif p.kind == "choice":
            values = list(p.choices)
        elif p.kind == "bool":
            values = [False, True]
        else:
            values = [v for v in (p.min, p.max) if v is not None]
            values = [int(v) for v in values] if p.kind == "int" else values
        for v in values:
            kw = {p.name: v} if p.name == "max_iter" else {p.name: v, "max_iter": 3}
            res = numopt.run(method, prob, seed=0, **kw)
            assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_bare_callable_with_a_box_through_vector_problem(method):
    """A bare callable has no box: the error names the remedy, and the remedy works."""
    from numopt.core.counting import vector_problem

    def f(x):
        return float((x[0] - 1.0) ** 2 + (x[1] + 0.5) ** 2)

    with pytest.raises(ValueError, match="vector_problem"):
        numopt.minimize(f, x0=[2.0, 2.0], method=method)
    prob = vector_problem(f, x0=[2.0, 2.0], domain=((-5.0, 5.0), (-5.0, 5.0)))
    res = numopt.minimize(prob, method=method, seed=0)
    assert_valid_result(res)
    assert res.converged, res.message
    tol = 1e-2 if method == "simulated_annealing" else 1e-5
    assert_allclose(res.x, [1.0, -0.5], atol=tol)


def test_basin_hopping_start_is_always_a_point():
    """The docstring documents ``start: [n]`` (x0 at k = 0, never null)."""
    prob = problems.get("himmelblau")
    res = numopt.run("basin_hopping", prob, seed=0, max_iter=5)
    assert_allclose(res.trace[0].info["start"], prob.x0, rtol=0, atol=0)
    assert all(len(s.info["start"]) == 2 for s in res.trace)
    assert "start: [n]            the perturbed" in (glob.__doc__ or "")
