"""Tests for numopt.unconstrained.derivative_free (Nelder–Mead, Powell, Hooke–Jeeves, compass)."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
import scipy.optimize as so
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import numopt
from numopt import problems
from numopt.unconstrained import derivative_free as dfo

EPS = float(np.finfo(np.float64).eps)
METHODS = ("nelder_mead", "powell", "hooke_jeeves", "compass_search")
SLOW = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _val(v: float | None) -> float:
    """Narrow an optional objective value (Step.fun / Result.fun) to float."""
    assert v is not None
    return v


def _nearest_min_dist(prob, x) -> float:
    return min(float(np.max(np.abs(np.asarray(x) - np.asarray(m)))) for m in prob.minima)


def _spd(seed: int, n: int, log_cond: float) -> tuple[np.ndarray, np.ndarray]:
    """A random SPD matrix with eigenvalues in [1, 10^log_cond] and a random centre."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = 10.0 ** rng.uniform(0.0, log_cond, size=n)
    return (Q * lam) @ Q.T, rng.uniform(-3.0, 3.0, size=n)


# --------------------------------------------------------------------------------------
# Contract checks shared by every method
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["quadratic_bowl", "himmelblau", "beale", "six_hump_camel"])
def test_converges_to_a_listed_minimum(method, pid):
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=numopt.get_method(method).defaults()["max_iter"])
    assert res.converged, res.message
    assert res.n_iter == res.trace[-1].k
    # NOTE: 1e-6 bounds the localization error of the step/simplex tests (xtol = 1e-8) times
    # the conditioning of these minima; every method lands far closer in practice.
    assert _nearest_min_dist(prob, res.x) <= 1e-6
    assert res.fun == res.trace[-1].fun
    assert res.n_gev == 0 and res.n_hev == 0


@pytest.mark.parametrize("method", METHODS)
def test_rosenbrock(method):
    prob = problems.get("rosenbrock")
    kw = {"max_iter": 20_000} if method == "compass_search" else {}
    res = numopt.run(method, prob, **kw)
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_reports_max_iter(method):
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_start_is_reported(method):
    res = numopt.minimize(lambda x: math.nan, x0=[1.0, 2.0], method=method)
    assert_valid_result(res)
    assert not res.converged and "not finite" in res.message
    assert res.n_iter == 0 and res.n_fev == 1


@pytest.mark.parametrize("method", METHODS)
def test_extreme_barrier_keeps_iterates_feasible(method):
    # f is NaN outside the disc ‖x‖ < 2; the minimizer (1, 0.5) is inside.
    def f(x):
        if x[0] ** 2 + x[1] ** 2 >= 4.0:
            return math.nan
        return (x[0] - 1.0) ** 2 + 4.0 * (x[1] - 0.5) ** 2

    res = numopt.minimize(f, x0=[-1.0, -1.0], method=method)
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, 0.5], atol=1e-6)
    assert all(math.isfinite(_val(s.fun)) for s in res.trace)


@pytest.mark.parametrize("method", METHODS)
def test_minus_infinity_means_unbounded(method):
    def f(x):
        return -math.inf if x[0] > 1.5 else (x[0] - 3.0) ** 2 + x[1] ** 2

    res = numopt.minimize(f, x0=[0.0, 0.0], method=method)
    assert_valid_result(res)
    assert not res.converged and "unbounded" in res.message


@pytest.mark.parametrize("method", METHODS)
def test_one_dimensional_problem(method):
    """A bare callable with a one-entry x0 is a dim = 1 Problem (core.counting), so it receives
    floats (the scalar convention of core.types), directly and through numopt.minimize."""

    def f(x: float) -> float:
        assert isinstance(x, float), type(x)
        return (x - 2.0) ** 2 + 1.0

    direct = getattr(dfo, method)(f, x0=[0.0])
    via_minimize = numopt.minimize(f, x0=[0.0], method=method)
    for res in (direct, via_minimize):
        assert_valid_result(res)
        assert res.converged
        assert res.x.shape == (1,)
        assert abs(res.x[0] - 2.0) <= 1e-6 and abs(_val(res.fun) - 1.0) <= 1e-12
    assert (direct.n_iter, direct.n_fev) == (via_minimize.n_iter, via_minimize.n_fev)


@pytest.mark.parametrize("method", METHODS)
def test_does_not_mutate_x0(method):
    x0 = np.array([-1.2, 1.0])
    numopt.minimize(problems.get("rosenbrock").f, x0=x0, method=method, max_iter=20)
    assert x0.tolist() == [-1.2, 1.0]


@pytest.mark.parametrize("method", METHODS)
def test_fixture_cases_are_valid(method):
    for mid, pid, params in dfo.FIXTURE_CASES:
        if mid != method:
            continue
        res = numopt.run(mid, problems.get(pid), **params)
        assert_valid_result(res)
        assert res.converged, (mid, pid, res.message)
        assert len(res.trace) < 300


@given(seed=st.integers(0, 2**31 - 1), log_cond=st.floats(0.0, 3.0), n=st.integers(2, 4))
@settings(max_examples=150, deadline=None)
def test_best_value_never_increases(seed, log_cond, n):
    A, c = _spd(seed, n, log_cond)

    def f(x):
        d = np.asarray(x) - c
        return 0.5 * float(d @ A @ d)

    for method in METHODS:
        res = numopt.minimize(f, x0=np.zeros(n), method=method, max_iter=200)
        funs = [_val(s.fun) for s in res.trace]
        assert all(b <= a for a, b in pairwise(funs)), method


# --------------------------------------------------------------------------------------
# Nelder–Mead
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pid", ["rosenbrock", "himmelblau", "beale", "booth", "six_hump_camel", "goldstein_price"]
)
@pytest.mark.parametrize("adaptive", [False, True])
def test_nelder_mead_matches_scipy_exactly(pid, adaptive):
    """SciPy's Nelder–Mead is also LRWW (1998): from the same simplex, the same path."""
    prob = problems.get(pid)
    x0 = np.asarray(prob.x0, dtype=float)
    h = 0.5
    simplex = np.vstack([x0] + [x0 + h * e for e in np.eye(2)])
    res = numopt.run("nelder_mead", prob, initial_step=h, adaptive=adaptive)
    ref = so.minimize(
        prob.f,
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-8, "fatol": 1e-8, "adaptive": adaptive},
    )
    assert res.converged and ref.success
    assert res.n_fev == ref.nfev
    assert res.n_iter == ref.nit - 1  # SciPy starts counting iterations at 1
    # NOTE: rounding-level differences only (SciPy's argsort is not stable; centroid summation order).
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-12)


def test_nelder_mead_adaptive_matches_scipy_in_ten_dimensions():
    prob = problems.get("rosenbrock_nd")
    x0 = np.asarray(prob.x0, dtype=float)
    n = x0.size
    simplex = np.vstack([x0] + [x0 + 0.5 * e for e in np.eye(n)])
    res = numopt.run("nelder_mead", prob, adaptive=True, max_iter=10_000)
    opts = {"initial_simplex": simplex, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True}
    ref = so.minimize(prob.f, x0, method="Nelder-Mead", options={**opts, "maxiter": 10_000})
    assert res.converged and ref.success
    # NOTE: paths agree for many iterations and then split on rounding of near-ties, so only the
    # minimizers are compared: both lie within the size tolerance of the minimum (1, …, 1).
    assert_allclose(res.x, ref.x, atol=1e-6)
    assert_allclose(res.x, np.ones(n), atol=1e-6)


def test_nelder_mead_adaptive_helps_in_high_dimension():
    prob = problems.get("quadratic_nd")  # n = 20
    std = numopt.run("nelder_mead", prob, max_iter=50_000)
    ada = numopt.run("nelder_mead", prob, adaptive=True, max_iter=50_000)
    assert std.converged and ada.converged
    assert ada.n_iter < std.n_iter
    assert _nearest_min_dist(prob, ada.x) <= 1e-6


def test_nelder_mead_coefficients():
    assert dfo._nm_coefficients(5, False) == (1.0, 2.0, 0.5, 0.5)
    rho, chi, gamma, sigma = dfo._nm_coefficients(4, True)
    assert (rho, chi, gamma, sigma) == (1.0, 1.5, 0.625, 0.75)
    # n = 2: Gao–Han reduce to the standard coefficients.
    assert dfo._nm_coefficients(2, True) == (1.0, 2.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        numopt.minimize(lambda x: x[0] ** 2, x0=[1.0], method="nelder_mead", adaptive=True)


@pytest.mark.parametrize(
    "pid", ["rosenbrock", "himmelblau", "goldstein_price", "rastrigin", "rosenbrock_nd"]
)
def test_nelder_mead_operations_follow_lrww(pid):
    """Every recorded operation satisfies the LRWW acceptance conditions of its trial values."""
    prob = problems.get(pid)
    res = numopt.run("nelder_mead", prob, max_iter=400)
    n = prob.dim
    expected_fev = n + 1
    for prev, cur in pairwise(res.trace):
        fo = prev.info["simplex_f"]  # old simplex values, sorted
        f1, fn, fworst = fo[0], fo[n - 1], fo[n]
        trials = {t["op"]: t["f"] for t in cur.info["trials"]}
        ops = [t["op"] for t in cur.info["trials"]]
        op = cur.info["operation"]
        fr = trials["reflect"]
        assert ops[0] == "reflect"
        if op == "reflect":
            assert (f1 <= fr < fn) or (fr < f1 and trials["expand"] >= fr)
        elif op == "expand":
            assert fr < f1 and trials["expand"] < fr
        elif op == "contract_outside":
            assert fn <= fr < fworst and trials["contract_outside"] <= fr
        elif op == "contract_inside":
            assert fr >= fworst and trials["contract_inside"] < fworst
        else:
            assert op == "shrink"
            assert ("contract_outside" in trials and trials["contract_outside"] > fr) or (
                "contract_inside" in trials and trials["contract_inside"] >= fworst
            )
        # The vertices stay sorted and the best value never increases.
        fs = cur.info["simplex_f"]
        assert fs == sorted(fs) and fs[0] <= f1
        expected_fev += len(cur.info["trials"]) + (n if op == "shrink" else 0)
        # Geometry: the reflected point is x̄ + ρ(x̄ − x_worst).
        xr = next(t["x"] for t in cur.info["trials"] if t["op"] == "reflect")
        cen, worst = np.asarray(cur.info["centroid"]), np.asarray(cur.info["worst"])
        assert_allclose(xr, 2.0 * cen - worst, rtol=0, atol=1e-12)
        assert_allclose(cen, np.mean(np.asarray(prev.info["simplex"])[:n], axis=0), atol=1e-12)
    assert res.n_fev == expected_fev


@pytest.mark.parametrize("pid", ["rastrigin", "ackley", "levi13"])
def test_nelder_mead_shrink_geometry(pid):
    """A shrink replaces x_i by x_1 + σ(x_i − x_1), σ = ½ (these runs each contain one)."""
    res = numopt.run("nelder_mead", problems.get(pid))
    shrinks = [(p, c) for p, c in pairwise(res.trace) if c.info["operation"] == "shrink"]
    assert shrinks
    for prev, cur in shrinks:
        old = np.asarray(prev.info["simplex"])
        new = np.asarray(cur.info["simplex"])
        expect = old[0] + 0.5 * (old - old[0])
        assert sorted(map(tuple, new.tolist())) == sorted(map(tuple, expect.tolist()))
    assert res.converged


def test_nelder_mead_validates_parameters():
    with pytest.raises(ValueError):
        numopt.run("nelder_mead", problems.get("rosenbrock"), xtol=0.0)
    with pytest.raises(ValueError):
        numopt.run("nelder_mead", problems.get("rosenbrock"), initial_step=-1.0)


# --------------------------------------------------------------------------------------
# Powell
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pid",
    [
        "rosenbrock",
        "himmelblau",
        "beale",
        "booth",
        "quadratic_ill",
        "six_hump_camel",
        "goldstein_price",
        "three_hump_camel",
        "styblinski_tang",
        "rosenbrock_nd",
    ],
)
def test_powell_matches_scipy(pid):
    prob = problems.get(pid)
    x0 = np.asarray(prob.x0, dtype=float)
    res = numopt.run("powell", prob)
    ref = so.minimize(
        prob.f, x0, method="Powell", options={"xtol": 1e-10, "ftol": 1e-12, "maxfev": 10**6}
    )
    assert res.converged and ref.success
    # NOTE: different bracketing details, same local minimizer; 1e-7 covers Brent's relative
    # tolerance 3e-8 on each line and the flatness of f at the minimum.
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-7)
    assert res.fun <= ref.fun + 1e-12 * max(1.0, abs(ref.fun))


def test_powell_matyas_where_scipy_stalls():
    res = numopt.run("powell", problems.get("matyas"))
    assert res.converged
    assert np.max(np.abs(res.x)) <= 1e-10


@given(seed=st.integers(0, 2**31 - 1), log_cond=st.floats(0.0, 4.0), n=st.integers(2, 5))
@settings(max_examples=200, deadline=None)
def test_powell_solves_quadratics(seed, log_cond, n):
    A, c = _spd(seed, n, log_cond)
    res = numopt.minimize(
        lambda x: 0.5 * float((x - c) @ A @ (x - c)), x0=np.zeros(n), method="powell"
    )
    assert res.converged
    # Error in the A-norm, relative to the start: Brent's tolerance limits it to ~1e-8.
    e = res.x - c
    assert math.sqrt(float(e @ A @ e)) <= 1e-7 * max(1.0, math.sqrt(float(c @ A @ c)))


def test_powell_counts_and_line_geometry():
    prob = problems.get("rosenbrock")
    res = numopt.run("powell", prob)
    fev = 1
    for prev, step in pairwise(res.trace):
        lines = step.info["lines"]
        x = np.asarray(prev.x)
        for i, ln in enumerate(lines):
            assert_allclose(ln["origin"], x, atol=0)
            assert_allclose(ln["point"], x + ln["alpha"] * np.asarray(ln["direction"]), atol=1e-15)
            assert ln["f"] == pytest.approx(prob.f(ln["point"]), rel=1e-15, abs=1e-300)
            # Every line search decreases f (NR's mnbrak may discard a lower trial, so the
            # result need not be the lowest trial).
            assert ln["f"] <= prob.f(ln["origin"])
            if i < len(step.info["directions"]):
                assert_allclose(ln["direction"], step.info["directions"][i], atol=0)
            x = np.asarray(ln["point"])
            fev += len(ln["trials"])
        fev += 0 if step.info["extrapolated"] is None else 1
        assert_allclose(step.x, x, atol=0)
        if step.info["replaced"] is not None:
            assert len(lines) == len(step.info["directions"]) + 1
            # The new direction is Powell's u = x_n − x_0 of the sweep (unscaled; see NOTE).
            u = np.asarray(lines[-1]["origin"]) - np.asarray(prev.x)
            assert_allclose(lines[-1]["direction"], u, rtol=0, atol=0)
            assert_allclose(step.info["new_directions"][-1], u, rtol=0, atol=0)
    assert res.n_fev == fev


def test_powell_replacement_rule():
    """NR §10.7: replace iff f_E < f_0 and t < 0; the discarded direction is the one of largest decrease."""
    res = numopt.run("powell", problems.get("rosenbrock"))
    replaced_any = False
    for prev, step in pairwise(res.trace):
        info = step.info
        if info["extrapolated"] is None:
            continue
        f0, fe = prev.fun, info["f_extrapolated"]
        if info["replaced"] is not None:
            replaced_any = True
            assert fe < f0 and info["replace_test"] < 0.0
            assert info["replaced"] == info["largest_index"]
            old, new = info["directions"], info["new_directions"]
            ib = info["replaced"]
            kept = [d for j, d in enumerate(old) if j != ib]
            assert_allclose(
                new[:-1], [kept[-1] if j == ib else old[j] for j in range(len(old) - 1)]
            )
        else:
            assert fe >= f0 or info["replace_test"] >= 0.0
    assert replaced_any


def test_powell_unbounded_line():
    res = numopt.minimize(lambda x: x[0] + x[1], x0=[0.0, 0.0], method="powell")
    assert_valid_result(res)
    assert not res.converged and "bracket" in res.message


@given(c=st.floats(-50, 50), a=st.floats(1e-3, 1e3), b=st.floats(-10, 10))
@settings(max_examples=1000, deadline=None)
def test_line_minimizer_on_parabolas(c, a, b):
    res = dfo._line_minimize(lambda t: a * (t - c) ** 2 + b, a * c * c + b)
    assert res.ok
    # Brent stops with the minimizer within tol2 = 2(tol·|α| + ZEPS) of α (tol = 3e-8, NR3).
    # Rounding in φ adds the flat zone a(α − c)² ≤ 2ε|b|, where φ cannot see the vertex.
    flat = math.sqrt(4.0 * EPS * abs(b) / a)
    assert abs(res.alpha - c) <= 2.0 * (3e-8 * abs(c) + 1e-15) + flat + 4.0 * EPS * abs(c)


@pytest.mark.parametrize(
    "phi,lo,hi",
    [
        (lambda t: math.cos(t) + 0.1 * t, 1.0, 5.0),
        (lambda t: (t - 1.3) ** 4 + 0.5 * t, -2.0, 4.0),
        (lambda t: math.exp(t) - 3.0 * t, -2.0, 4.0),
    ],
)
def test_line_minimizer_matches_scipy_brent(phi, lo, hi):
    res = dfo._line_minimize(phi, phi(0.0))
    ref = so.minimize_scalar(phi, bracket=(0.0, 1.0), method="brent", tol=3e-8)
    assert res.ok
    assert res.alpha == pytest.approx(float(ref.x), abs=1e-6)  # pyright: ignore[reportAttributeAccessIssue]
    assert res.f <= float(ref.fun) + 1e-14  # pyright: ignore[reportAttributeAccessIssue]


# --------------------------------------------------------------------------------------
# Hooke–Jeeves and compass search
# --------------------------------------------------------------------------------------


@given(seed=st.integers(0, 2**31 - 1), log_cond=st.floats(0.0, 3.0), n=st.integers(1, 4))
@settings(max_examples=300, deadline=None)
def test_pattern_searches_certify_stationarity_on_quadratics(seed, log_cond, n):
    """At an unsuccessful iteration with step h, f(x ± h e_i) ≥ f(x) ⇒ |∂_i f(x)| ≤ ½ A_ii h.

    This is the exact quadratic form of KLT (2003) eq. (3.3), ‖∇f‖ ≤ √n M Δ (checked as well).
    """
    A, c = _spd(seed, n, log_cond)

    def f(x):
        d = np.asarray(x) - c
        return 0.5 * float(d @ A @ d)

    M = float(np.linalg.eigvalsh(A)[-1])
    diag = np.diag(A)
    for method, failed in (
        ("hooke_jeeves", lambda s: s.info["outcome"] == "step_reduced"),
        ("compass_search", lambda s: s.info["success"] is False),
    ):
        # NOTE: compass search needs O(κ) successful polls per step level on an ill-conditioned
        # quadratic (up to ~3·10⁴ iterations at κ ≈ 10³, n = 4), hence the large limit.
        res = numopt.minimize(f, x0=np.zeros(n), method=method, max_iter=100_000)
        assert res.converged, res.message
        checked = 0
        for s in res.trace[1:]:
            if not failed(s):
                continue
            d = np.asarray(s.x) - c
            g = A @ d
            h = s.info["step"]
            # f(x ± h e_i) ≥ f(x) holds for the *computed* f; each value has an absolute error
            # ≤ δ = 2nε·M(‖d‖ + h)² (Higham 2002, §3.1, dot products), which adds 2δ/h.
            delta = 2.0 * n * EPS * M * (float(np.linalg.norm(d)) + h) ** 2
            assert np.all(np.abs(g) <= 0.5 * diag * h + 2.0 * delta / h)
            assert np.linalg.norm(g) <= math.sqrt(n) * M * h + 2.0 * math.sqrt(n) * delta / h
            checked += 1
        assert checked >= 1


def test_hooke_jeeves_moves():
    res = numopt.run("hooke_jeeves", problems.get("rosenbrock"))
    fev = 1
    for prev, s in pairwise(res.trace):
        info = s.info
        fev += len(info["probes"])
        base_prev = np.asarray(prev.info["base"])
        if info["move"] == "pattern":
            # The pattern point is 2·base − previous base, and it is probed first.
            pp = 2.0 * base_prev - np.asarray(prev.info["previous_base"])
            assert_allclose(info["pattern_point"], pp, atol=1e-15)
            assert_allclose(info["probes"][0]["x"], pp, atol=0)
            assert prev.info["outcome"] in ("explore_success", "pattern_success")
        else:
            assert info["pattern_point"] is None
            assert len(info["probes"]) <= 2 * len(base_prev)
        if info["outcome"] in ("explore_success", "pattern_success"):
            assert _val(s.fun) < _val(prev.fun)
            assert_allclose(info["previous_base"], base_prev, atol=0)
        else:
            assert s.fun == prev.fun
            assert_allclose(info["base"], base_prev, atol=0)
        expected_step = info["step"] * (0.5 if info["outcome"] == "step_reduced" else 1.0)
        assert info["new_step"] == expected_step
    assert res.n_fev == fev


def test_compass_search_polls_and_counts():
    res = numopt.run("compass_search", problems.get("beale"))
    n = 2
    D = list(np.eye(n)) + list(-np.eye(n))
    fev = 1
    for prev, s in pairwise(res.trace):
        info = s.info
        x = np.asarray(prev.x)
        for j, poll in enumerate(info["polls"]):
            assert_allclose(poll["x"], x + info["step"] * D[j], atol=1e-15)
        fev += len(info["polls"])
        if info["success"]:
            assert len(info["polls"]) == info["direction"] + 1
            assert _val(s.fun) < _val(prev.fun) and info["new_step"] == info["step"]
            assert all(p["f"] >= prev.fun for p in info["polls"][:-1])
        else:
            assert len(info["polls"]) == 2 * n
            assert all(p["f"] >= prev.fun for p in info["polls"])
            assert info["new_step"] == 0.5 * info["step"]
    assert res.n_fev == fev
    assert res.converged and res.trace[-1].info["new_step"] < 1e-8


@pytest.mark.parametrize("method", ["hooke_jeeves", "compass_search"])
def test_pattern_search_validates_parameters(method):
    prob = problems.get("booth")
    with pytest.raises(ValueError):
        numopt.run(method, prob, shrink=1.0)
    with pytest.raises(ValueError):
        numopt.run(method, prob, step=0.0)


@pytest.mark.parametrize("max_iter", [0, -1, 2.5])
@pytest.mark.parametrize("method", METHODS)
def test_rejects_invalid_max_iter(method, max_iter):
    """Regression: ``k == max_iter`` never fires for these values, so pattern searches looped
    forever on an f that is unbounded below and Nelder–Mead ran until f = −∞."""
    with pytest.raises(ValueError, match="max_iter"):
        numopt.minimize(lambda x: -x[0] + x[1] ** 2, x0=[0.5, 0.5], method=method,
                        max_iter=max_iter)  # fmt: skip


def test_max_iter_one_is_accepted():
    res = numopt.run("compass_search", problems.get("booth"), max_iter=1)
    assert_valid_result(res, max_iter=1)
    assert res.n_iter == 1 and not res.converged


def test_minimize_accepts_bare_callable():
    res = numopt.minimize(
        lambda x: (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2, x0=[0, 0], method="nelder_mead"
    )
    assert res.converged
    assert_allclose(res.x, [1.0, -2.0], atol=1e-7)


# --------------------------------------------------------------------------------------
# Regression tests for audit findings
# --------------------------------------------------------------------------------------


def test_powell_order_string_matches_the_discarding_rule():
    """NR's discarding rule loses quadratic termination: random SPD quadratics need more than n
    sweeps (the old order string claimed n)."""
    order = numopt.get_method("powell").order
    assert "no quadratic termination" in order
    n = 5
    sweeps = []
    for seed in range(10):
        A, c = _spd(seed, n, 3.0)

        def f(x, A=A, c=c):
            return 0.5 * float((x - c) @ A @ (x - c))

        f0 = f(np.zeros(n))
        res = numopt.minimize(f, x0=np.zeros(n), method="powell")
        assert res.converged
        sweeps.append(next(s.k for s in res.trace if _val(s.fun) <= 1e-12 * f0))
    assert max(sweeps) > n


@pytest.mark.parametrize("c", [0.0, 1e4, 1e6, 1e8])
def test_powell_with_a_large_constant_in_f(c):
    """f = c + Rosenbrock. NR's relative test alone stopped at ‖x − x*‖ = 0.11 for c = 1e8
    (f − c = 4.8e-3 ≫ ulp(1e8) = 1.5e-8). With the x test, Powell reaches the rounding
    plateau: Rosenbrock(x) at most a few ulps of c."""
    rb = problems.get("rosenbrock").f
    res = numopt.minimize(lambda x: c + float(rb(x)), x0=[-1.2, 1.0], method="powell")
    assert_valid_result(res)
    assert res.converged, res.message
    assert float(rb(res.x)) <= 2.0 * EPS * c + 1e-20
    # ½ λ_min(H) ‖x − x*‖² ≤ Rosenbrock(x) near x* (λ_min(∇²f(x*)) ≈ 0.3994).
    lam_min = float(np.linalg.eigvalsh(np.array([[802.0, -400.0], [-400.0, 200.0]]))[0])
    assert np.linalg.norm(res.x - 1.0) <= math.sqrt(2.0 * (2.0 * EPS * c + 1e-20) / lam_min) * 1.01


@given(
    seed=st.integers(0, 2**31 - 1),
    log_cond=st.floats(0.0, 3.0),
    n=st.integers(2, 4),
    log_offset=st.floats(-2.0, 9.0),
    sign=st.sampled_from([-1.0, 1.0]),
)
@settings(max_examples=1000, deadline=None)
def test_powell_stopping_test_and_accuracy_with_offsets(seed, log_cond, n, log_offset, sign):
    """The final sweep meets both stopping tests, and the minimizer is found to the rounding
    plateau of f = b + ½(x − c)ᵀA(x − c), whatever the offset b."""
    A, c = _spd(seed, n, log_cond)
    b = sign * 10.0**log_offset

    def g(x):
        return 0.5 * float((x - c) @ A @ (x - c))

    res = numopt.minimize(lambda x: b + g(x), x0=np.zeros(n), method="powell")
    assert res.converged, res.message
    prev, last = res.trace[-2], res.trace[-1]
    f0, fn = _val(prev.fun), _val(last.fun)
    assert 2.0 * (f0 - fn) <= 1e-10 * (abs(f0) + abs(fn)) + 1e-25
    step = float(np.max(np.abs(np.asarray(last.x) - np.asarray(prev.x))))
    assert step <= 1e-8 + 2.0 * EPS * float(np.max(np.abs(last.x))) or f0 - fn <= 2.0 * EPS * abs(
        fn
    )
    # Accuracy: the A-norm error of the offset-free test (Brent's tolerance, ~1e-8 relative),
    # plus the rounding plateau g ≤ 2ε|b| (one ulp of b in the sum, one in the comparison).
    e = res.x - c
    err_a = math.sqrt(float(e @ A @ e))
    assert err_a <= 1e-7 * max(1.0, math.sqrt(float(c @ A @ c))) + math.sqrt(4.0 * EPS * abs(b))


def test_info_is_null_at_the_start_where_documented():
    """The module docstring documents move, outcome and success as null at k = 0."""
    hj = numopt.run("hooke_jeeves", problems.get("booth"))
    assert hj.trace[0].info["move"] is None and hj.trace[0].info["outcome"] is None
    assert all(isinstance(s.info["move"], str) for s in hj.trace[1:])
    cs = numopt.run("compass_search", problems.get("booth"))
    assert cs.trace[0].info["success"] is None
    assert all(isinstance(s.info["success"], bool) for s in cs.trace[1:])
    doc = dfo.__doc__ or ""
    assert '"pattern" | null' in doc and "success: bool | null" in doc


@pytest.mark.parametrize("method", METHODS)
def test_every_value_in_the_param_spec_range_is_accepted(method):
    """A UI builds its controls from the ParamSpec; every value it offers must run."""
    spec = numopt.get_method(method)
    prob = problems.get("himmelblau")
    for p in spec.params:
        if p.name == "max_iter":
            values: list[object] = [int(p.min)] if p.min is not None else []
        elif p.kind == "bool":
            values = [False, True]
        else:
            bounds = [v for v in (p.min, p.max) if v is not None]
            values = [int(v) for v in bounds] if p.kind == "int" else list(bounds)
        for v in values:
            kw = {p.name: v} if p.name == "max_iter" else {p.name: v, "max_iter": 3}
            assert_valid_result(numopt.run(method, prob, **kw))


# --------------------------------------------------------------------------------------
# dim = 1 library problems (web-port regression)
# --------------------------------------------------------------------------------------

ONE_D = (
    "quadratic_1d",
    "quartic_1d",
    "sin_1d",
    "x_log_x",
    "abs_shifted",
    "multimodal_1d",
    "rational_1d",
    "drug_concentration",
)


@pytest.mark.parametrize("pid", ONE_D)
@pytest.mark.parametrize("method", METHODS)
def test_one_dimensional_library_problems(method, pid):
    """Regression: with NumPy ≥ 2.5, float() of the shape-(1,) value of a vectorized 1-D f
    raised TypeError, and f's that accept only floats (sin_1d, x_log_x, …) raised on the
    array argument. Oracle: the minimizer is the nearest listed local minimizer, and f there
    agrees with scipy's bounded Brent search on a small interval around it."""
    p = problems.get(pid)
    res = numopt.run(method, p)
    assert_valid_result(res)
    assert res.converged, res.message
    assert res.x.shape == (1,)
    x = float(res.x[0])
    x_star = min((float(m) for m in p.minima), key=lambda m: abs(m - x))
    # 1e-4: the methods' default x tolerances are ≈ 1e-6 to 1e-8; abs_shifted has a kink.
    assert abs(x - x_star) <= 1e-4, (x, x_star)
    ref = so.minimize_scalar(
        p.f, bounds=(x_star - 0.05, x_star + 0.05), method="bounded", options={"xatol": 1e-10}
    )
    assert _val(res.fun) <= float(ref.fun) + 1e-8  # pyright: ignore[reportAttributeAccessIssue]


def test_objective_value_shapes():
    """A size-1 array result is accepted as f(x); a result with more entries is invalid input."""
    arr = dfo.compass_search(lambda x: np.array([(x[0] - 1.0) ** 2]), x0=[0.0, 0.0])
    assert arr.converged and _val(arr.fun) <= 1e-10
    one_d = dfo.nelder_mead(lambda x: np.array([(x - 2.0) ** 2 + 1.0]), x0=[0.5])
    assert one_d.converged and abs(_val(one_d.fun) - 1.0) <= 1e-12
    with pytest.raises(ValueError, match="scalar"):
        numopt.minimize(lambda x: np.array([1.0, 2.0]) * x, x0=[0.5], method="powell")
