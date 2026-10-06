"""Tests for numopt.unconstrained.conjugate_gradient (nonlinear CG, five β formulas)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import warnings
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.unconstrained import conjugate_gradient as cg

METHODS = (
    "cg_fletcher_reeves",
    "cg_polak_ribiere",
    "cg_hestenes_stiefel",
    "cg_dai_yuan",
    "cg_hager_zhang",
)
INFO_KEYS = {
    "direction",
    "beta",
    "beta_formula",
    "restart",
    "powell_ratio",
    "descent",
    "alpha",
    "trials",
}
EPS = float(np.finfo(float).eps)


def _quadratic_problem(A: np.ndarray, c: np.ndarray) -> Problem:
    """f(x) = ½(x − c)ᵀA(x − c) as a Problem (used by the property tests)."""
    return Problem(
        id="test_quadratic",
        name="test quadratic",
        latex="",
        f=lambda x: 0.5 * float((x - c) @ A @ (x - c)),
        dim=c.size,
        domain=(),
        grad=lambda x: A @ (x - c),
        hess=lambda x: A.copy(),
        x0=np.zeros(c.size),
    )


def _spd(n: int, cond: float, seed: int) -> np.ndarray:
    """SPD matrix Q diag(λ) Qᵀ with λ geometric from 1 to ``cond``."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.geomspace(1.0, cond, n)
    A = Q @ np.diag(lam) @ Q.T
    return 0.5 * (A + A.T)


# --------------------------------------------------------------------------------------
# Convergence (≥ 2 problems per method) and the contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "himmelblau", "beale", "quadratic_ill"])
def test_converges_to_a_known_minimizer(method: str, pid: str) -> None:
    prob = problems.get(pid)
    # gtol = 1e-6 (tighter than the default 1e-5) makes the distance bound below meaningful.
    res = numopt.run(method, prob, gtol=1e-6)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    assert float(np.max(np.abs(prob.grad(res.x)))) <= 1e-6
    # NOTE: ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)); 1e-5 covers λ_min ≥ 0.1 on these problems.
    dist = min(float(np.linalg.norm(res.x - np.asarray(m))) for m in prob.minima)
    assert dist <= 1e-5
    assert res.n_iter == res.trace[-1].k
    assert res.trace[-1].info["direction"] is None


@pytest.mark.parametrize("method", METHODS)
def test_rosenbrock_nd(method: str) -> None:
    prob = problems.get("rosenbrock_nd")
    res = numopt.run(method, prob, gtol=1e-6)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    dist = min(float(np.linalg.norm(res.x - np.asarray(m))) for m in prob.minima)
    assert dist <= 1e-5


@pytest.mark.parametrize("method", METHODS)
def test_info_keys_documented_and_json(method: str) -> None:
    res = numopt.run(method, problems.get("rosenbrock"))
    for s in res.trace:
        assert set(s.info) == INFO_KEYS
    json.dumps(res.to_dict(), allow_nan=False)
    assert cg.__doc__ is not None
    for key in INFO_KEYS:
        assert f"    {key}:" in cg.__doc__
    first = res.trace[0]
    assert first.info["restart"] == "initial" and first.info["alpha"] is None
    assert first.step_size is None and first.info["trials"] == []
    assert_allclose(first.info["direction"], -problems.get("rosenbrock").grad(first.x), rtol=0)


# --------------------------------------------------------------------------------------
# Oracles: SciPy CG (PR+), linear CG in exact rational arithmetic
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "booth", "matyas"])
def test_minimizer_matches_scipy_cg(method: str, pid: str) -> None:
    prob = problems.get(pid)
    ref = minimize(prob.f, prob.x0, jac=prob.grad, method="CG", options={"gtol": 1e-9})
    assert ref.success
    res = numopt.run(method, prob, gtol=1e-8)
    assert res.converged, res.message
    # NOTE: both stop at ‖∇f‖ ≈ 1e-8–1e-9; the minimizers are isolated with λ_min(∇²f) ≥ 0.4,
    # so the iterates agree to ≈ 1e-8/0.4; 1e-7 leaves room for that.
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-7)


def _exact_linear_cg(A: np.ndarray, c: np.ndarray, x0: np.ndarray, iters: int) -> np.ndarray:
    """Linear CG (N&W Alg. 5.2) on A x = A c in exact rational arithmetic; returns x_0..x_iters."""
    n = c.size
    Af = [[Fraction(v) for v in row] for row in A.tolist()]
    cf = [Fraction(v) for v in c.tolist()]

    def mv(v: list[Fraction]) -> list[Fraction]:
        return [sum((Af[i][j] * v[j] for j in range(n)), Fraction(0)) for i in range(n)]

    def dot(u: list[Fraction], v: list[Fraction]) -> Fraction:
        return sum((a * b for a, b in zip(u, v, strict=True)), Fraction(0))

    x = [Fraction(v) for v in x0.tolist()]
    r = mv([x[i] - cf[i] for i in range(n)])
    p = [-v for v in r]
    out = [[float(v) for v in x]]
    for _ in range(iters):
        Ap = mv(p)
        rr = dot(r, r)
        alpha = rr / dot(p, Ap)
        x = [x[i] + alpha * p[i] for i in range(n)]
        r_new = [r[i] + alpha * Ap[i] for i in range(n)]
        beta = dot(r_new, r_new) / rr
        p = [-r_new[i] + beta * p[i] for i in range(n)]
        r = r_new
        out.append([float(v) for v in x])
    return np.array(out)


@pytest.fixture(scope="module")
def exact_cg_quadratic_nd() -> np.ndarray:
    prob = problems.get("quadratic_nd")
    A, c = np.array(prob.extra["A"]), np.array(prob.extra["c"])
    return _exact_linear_cg(A, c, np.zeros(c.size), c.size)


def test_exact_linear_cg_oracle_terminates_in_n_steps(exact_cg_quadratic_nd) -> None:
    """The oracle itself: in exact arithmetic linear CG reaches x* = c at step n = 20 (N&W
    Thm. 5.1), with A and c taken exactly as their float64 values."""
    c = np.array(problems.get("quadratic_nd").extra["c"])
    assert exact_cg_quadratic_nd.shape == (21, 20)
    assert np.array_equal(exact_cg_quadratic_nd[20], c)
    assert float(np.max(np.abs(exact_cg_quadratic_nd[19] - c))) > 1e-5  # measured 1.4e-4


@pytest.mark.parametrize("method", METHODS)
def test_exact_line_search_reproduces_linear_cg(method: str, exact_cg_quadratic_nd) -> None:
    """With α = −gᵀd/dᵀAd every β equals β^FR and the iterates are those of linear CG."""
    prob = problems.get("quadratic_nd")
    n = prob.dim
    res = numopt.run(method, prob, line_search="exact_quadratic")
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    X = np.array([s.x for s in res.trace[:11]])
    # NOTE: the iterates of CG on quadratic_nd are very sensitive to rounding: the forward error
    # of float64 CG against exact arithmetic grows ≈ 10× per iteration after k ≈ 8. Measured at
    # k = 10: 1e-14 (FR), 4e-14 (DY), 4e-13 (PR+), 8e-13 (HS), 1e-12 (HZ, whose correction term
    # 2‖y‖²dᵀg/(dᵀy)² is zero in exact arithmetic but amplifies the rounding of dᵀg). 1e-10
    # bounds k ≤ 10 with a 100× margin.
    assert_allclose(X, exact_cg_quadratic_nd[:11], rtol=0, atol=1e-10)
    # NOTE: finite termination is lost in float64 on this problem: ‖g₂₀‖ ≈ 1e-3 although exact
    # CG ends at k = 20 (test above). Its spectrum is geometric, dense near λ_min = 1, and g₀
    # has weights ≈ 1e-3 on the two smallest eigenvectors: the delay of convergence caused by
    # the loss of orthogonality described by Strakoš (1991), LAA 154–156, 535–549, and
    # Greenbaum & Strakoš (1992), SIAM J. Matrix Anal. Appl. 13(1), 121–137. With the
    # periodic restart at k = n the run still ends within 2n iterations.
    assert n < res.n_iter <= 2 * n
    # All β formulas coincide with β^FR along the way (g_kᵀg_{k−1} = 0 = g_kᵀd_{k−1}).
    for s in res.trace[1:9]:
        assert s.info["restart"] is None
        g = prob.grad(np.asarray(s.x))
        assert s.info["beta"] == pytest.approx(s.info["beta_formula"], rel=1e-12)
        assert s.info["powell_ratio"] < 1e-12
        assert s.info["descent"] == pytest.approx(-float(g @ g), rel=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["quadratic_bowl", "quadratic_ill", "booth", "matyas"])
def test_n_step_termination_2d(method: str, pid: str) -> None:
    """N&W Thm. 5.1: at most n = 2 iterations on a 2-D convex quadratic with exact line search."""
    res = numopt.run(method, problems.get(pid), line_search="exact_quadratic", gtol=1e-10)
    assert res.converged, res.message
    assert res.n_iter <= 2
    assert res.n_hev == res.n_iter  # one Hessian per exact line search


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(6, 20),
    log_cond=st.floats(0.0, 2.0),
    seed=st.integers(0, 2**31 - 1),
    method=st.sampled_from(METHODS),
)
def test_n_step_termination_up_to_n_20(n: int, log_cond: float, seed: int, method: str) -> None:
    """N&W Thm. 5.1 for n up to 20 (the size of quadratic_nd), with eigenvalues evenly spaced in
    [1, κ], κ ≤ 100: for such spectra float64 CG keeps its finite termination (unlike the
    geometric spectrum of quadratic_nd, see test_exact_line_search_reproduces_linear_cg)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.linspace(1.0, 10.0**log_cond, n)) @ Q.T
    A = 0.5 * (A + A.T)
    c = rng.uniform(-2.0, 2.0, n)
    prob = _quadratic_problem(A, c)
    assert prob.grad is not None
    g0 = float(np.max(np.abs(prob.grad(prob.x0))))
    res = numopt.run(method, prob, line_search="exact_quadratic", gtol=1e-10 * g0)
    assert res.converged, res.message
    assert res.n_iter <= n
    # ‖x − c‖ ≤ ‖A⁻¹‖‖∇f‖ ≤ √n·1e-10·‖g₀‖∞ with λ_min = 1 and ‖g₀‖∞ ≤ 100·2√n.
    assert_allclose(res.x, c, rtol=0, atol=1e-8 * n)


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(2, 5),
    log_cond=st.floats(0.0, 2.0),
    seed=st.integers(0, 2**31 - 1),
    method=st.sampled_from(METHODS),
)
def test_n_step_termination_random_spd(n: int, log_cond: float, seed: int, method: str) -> None:
    A = _spd(n, 10.0**log_cond, seed)
    c = np.random.default_rng(seed + 1).uniform(-2.0, 2.0, n)
    prob = _quadratic_problem(A, c)
    assert prob.grad is not None
    g0 = float(np.max(np.abs(prob.grad(prob.x0))))
    # NOTE: after n exact-arithmetic steps g = 0; in float64 the gradient left after n steps is
    # ≈ κ·ε·‖g₀‖ (κ ≤ 100, n ≤ 5), so 1e-10·‖g₀‖ is reached at iteration n.
    res = numopt.run(method, prob, line_search="exact_quadratic", gtol=1e-10 * g0)
    assert res.converged, res.message
    assert res.n_iter <= n
    assert_allclose(res.x, c, rtol=0, atol=1e-8 * (1.0 + float(np.max(np.abs(c)))))


# --------------------------------------------------------------------------------------
# Invariants: Wolfe conditions, descent, monotone decrease, β rules, restarts
# --------------------------------------------------------------------------------------


def _check_trace(method: str, prob: Problem, res: Any, c2: float = cg.C2_DEFAULT) -> None:
    """Check the strong Wolfe conditions, descent and β/restart bookkeeping of a trace."""
    assert prob.grad is not None
    n = prob.dim
    since = 0
    for prev, cur in itertools.pairwise(res.trace):
        x0, x1 = np.asarray(prev.x), np.asarray(cur.x)
        d = np.asarray(prev.info["direction"])
        g0, g1 = prob.grad(x0), prob.grad(x1)
        alpha = cur.info["alpha"]
        assert alpha > 0.0 and cur.step_size == alpha
        assert_allclose(x1, x0 + alpha * d, rtol=1e-15, atol=1e-15)
        slope0 = float(g0 @ d)
        assert slope0 < 0.0
        # Strong Wolfe (N&W eq. 3.7), with c₁ = 1e-4 and the run's c₂.
        assert cur.fun <= prev.fun + cg.C1 * alpha * slope0
        assert abs(float(g1 @ d)) <= -c2 * slope0 * (1.0 + 1e-10)
        assert cur.fun <= prev.fun
        assert cur.info["trials"] and cur.info["trials"][-1][0] == alpha
    for k, s in enumerate(res.trace):
        info = s.info
        if info["direction"] is None:
            continue
        g = prob.grad(np.asarray(s.x))
        d = np.asarray(info["direction"])
        assert info["descent"] == pytest.approx(float(g @ d), rel=1e-12, abs=1e-300)
        assert info["descent"] < 0.0
        restart = info["restart"]
        if k == 0:
            assert restart == "initial"
            continue
        ratio = info["powell_ratio"]
        if restart is not None:
            assert info["beta"] == 0.0
            assert_allclose(d, -g, rtol=0, atol=0)
            if restart == "periodic":
                assert since + 1 >= n
            elif restart == "powell":
                assert ratio >= cg.POWELL_NU
            since = 0
            continue
        since += 1
        assert since < n and ratio < cg.POWELL_NU
        beta, raw = info["beta"], info["beta_formula"]
        d_prev = np.asarray(res.trace[k - 1].info["direction"])
        assert_allclose(d, -g + beta * d_prev, rtol=1e-14, atol=1e-300)
        if method == "cg_polak_ribiere":
            assert beta == max(raw, 0.0)
        elif method == "cg_hager_zhang":
            assert beta >= raw
            # Hager & Zhang (2005), Thm. 1.1: gᵀd ≤ −⅞‖g‖² for β ∈ [β^N, 0] ∪ {β^N}.
            assert info["descent"] <= -0.875 * float(g @ g) * (1.0 - 1e-12)
        else:
            assert beta == raw
        if method == "cg_fletcher_reeves" and c2 < 0.5:
            # N&W Lemma 5.6: −1/(1−c₂) ≤ gᵀd/‖g‖² ≤ (2c₂−1)/(1−c₂).
            q = info["descent"] / float(g @ g)
            assert -1.0 / (1.0 - c2) - 1e-12 <= q <= (2.0 * c2 - 1.0) / (1.0 - c2) + 1e-12


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "himmelblau", "rosenbrock_nd"])
def test_trace_invariants(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert res.converged, res.message
    _check_trace(method, prob, res)


@settings(max_examples=60, deadline=None)
@given(
    x=st.floats(-1.8, 1.8),
    y=st.floats(-0.8, 2.8),
    method=st.sampled_from(METHODS),
    c2=st.sampled_from([0.1, 0.3, 0.45]),
)
def test_trace_invariants_random_starts(x: float, y: float, method: str, c2: float) -> None:
    prob = problems.get("rosenbrock")
    res = numopt.run(method, prob, x0=[x, y], c2=c2, max_iter=400)
    assert_valid_result(res, max_iter=400)
    _check_trace(method, prob, res, c2)


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(2, 8),
    log_cond=st.floats(0.0, 3.0),
    seed=st.integers(0, 2**31 - 1),
    method=st.sampled_from(METHODS),
)
def test_random_quadratic_strong_wolfe(n: int, log_cond: float, seed: int, method: str) -> None:
    A = _spd(n, 10.0**log_cond, seed)
    c = np.random.default_rng(seed + 7).uniform(-2.0, 2.0, n)
    prob = _quadratic_problem(A, c)
    res = numopt.run(method, prob, gtol=1e-8)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    _check_trace(method, prob, res)
    # ‖x − c‖ ≤ ‖A⁻¹‖‖∇f‖ ≤ √n·gtol/λ_min with λ_min = 1.
    assert float(np.linalg.norm(res.x - c)) <= math.sqrt(n) * 1e-8 * (1.0 + 1e-6)


def test_dai_yuan_never_needs_a_descent_restart() -> None:
    """Dai & Yuan (1999), Thm. 2.1: under the Wolfe conditions every DY direction is descent."""
    for pid in ("rosenbrock", "beale", "himmelblau", "rosenbrock_nd", "six_hump_camel"):
        res = numopt.run("cg_dai_yuan", problems.get(pid))
        assert all(s.info["restart"] != "not_descent" for s in res.trace)
        assert all(s.info["restart"] != "breakdown" for s in res.trace)


def test_beta_formulas_by_hand() -> None:
    g_prev = np.array([1.0, -2.0, 0.5])
    g = np.array([0.3, 0.4, -1.0])
    d_prev = np.array([-1.5, 1.0, 0.2])
    y = g - g_prev
    dy = float(d_prev @ y)
    assert (
        cg._beta("fletcher_reeves", g, g_prev, d_prev)
        == (pytest.approx(float(g @ g) / float(g_prev @ g_prev), rel=1e-15),) * 2
    )
    pr = float(g @ y) / float(g_prev @ g_prev)
    assert cg._beta("polak_ribiere", g, g_prev, d_prev) == (max(pr, 0.0), pytest.approx(pr))
    assert cg._beta("hestenes_stiefel", g, g_prev, d_prev)[0] == pytest.approx(float(g @ y) / dy)
    assert cg._beta("dai_yuan", g, g_prev, d_prev)[0] == pytest.approx(float(g @ g) / dy)
    beta_n = float((y - 2.0 * d_prev * float(y @ y) / dy) @ g) / dy
    eta_k = -1.0 / (float(np.linalg.norm(d_prev)) * min(0.01, float(np.linalg.norm(g_prev))))
    used, raw = cg._beta("hager_zhang", g, g_prev, d_prev)
    assert raw == pytest.approx(beta_n, rel=1e-14)
    assert used == pytest.approx(max(beta_n, eta_k), rel=1e-14)
    # dᵀy ≤ 0 makes HS, DY and HZ undefined (breakdown); ‖g_{k−1}‖ = 0 does so for FR and PR.
    assert dy > 0.0
    for rule in ("hestenes_stiefel", "dai_yuan", "hager_zhang"):
        assert cg._beta(rule, g, g_prev, 0.0 * d_prev) == (None, None)
        assert cg._beta(rule, g, g_prev, -d_prev) == (None, None)
    for rule in ("fletcher_reeves", "polak_ribiere"):
        assert cg._beta(rule, g, 0.0 * g_prev, d_prev) == (None, None)


def test_periodic_restart_on_2d_problem() -> None:
    """n = 2: after one CG direction the next direction is steepest descent."""
    res = numopt.run("cg_fletcher_reeves", problems.get("rosenbrock"))
    labels = [s.info["restart"] for s in res.trace if s.info["direction"] is not None]
    assert labels[0] == "initial"
    for a, b in itertools.pairwise(labels[1:]):
        assert not (a is None and b is None)  # never two CG directions in a row when n = 2
    assert "periodic" in labels
    assert res.extra["n_restarts"] == sum(lab not in (None, "initial") for lab in labels)


# --------------------------------------------------------------------------------------
# Exact evaluation counts
# --------------------------------------------------------------------------------------


def _counting(prob: Problem) -> tuple[Problem, dict[str, int]]:
    counts = {"f": 0, "g": 0, "h": 0}

    def f(x: Any) -> Any:
        counts["f"] += 1
        return prob.f(x)

    def g(x: Any) -> Any:
        counts["g"] += 1
        assert prob.grad is not None
        return prob.grad(x)

    def h(x: Any) -> Any:
        counts["h"] += 1
        assert prob.hess is not None
        return prob.hess(x)

    return dataclasses.replace(prob, f=f, grad=g, hess=h), counts


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("line_search", ["strong_wolfe", "exact_quadratic"])
def test_evaluation_counts_are_exact(method: str, line_search: str) -> None:
    pid = "rosenbrock" if line_search == "strong_wolfe" else "quadratic_ill"
    prob, counts = _counting(problems.get(pid))
    res = numopt.run(method, prob, line_search=line_search)
    assert res.converged, res.message
    assert (res.n_fev, res.n_gev, res.n_hev) == (counts["f"], counts["g"], counts["h"])
    trials = sum(len(s.info["trials"]) for s in res.trace)
    assert res.n_fev == 1 + trials  # f(x0) + one f per trial step
    if line_search == "exact_quadratic":
        assert res.n_hev == res.n_iter and res.n_gev == 1 + res.n_iter


def test_bare_callable_uses_finite_differences() -> None:
    res = numopt.minimize(
        lambda x: (x[0] - 1.0) ** 2 + 4.0 * (x[1] + 0.5) ** 2,
        x0=[3.0, 1.0],
        method="cg_polak_ribiere",
    )
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, -0.5], atol=1e-6)
    assert res.n_fev >= 4 * res.n_gev  # every FD gradient costs 2n = 4 evaluations of f


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_max_iter(method: str) -> None:
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4


def test_nonfinite_start() -> None:
    prob = dataclasses.replace(problems.get("rosenbrock"), f=lambda x: math.nan)
    res = numopt.run("cg_fletcher_reeves", prob)
    assert_valid_result(res)
    assert not res.converged and "not finite" in res.message
    assert res.n_iter == 0


def test_exact_line_search_needs_positive_curvature() -> None:
    """At the origin of Himmelblau ∇²f is negative definite, so dᵀ∇²f d < 0 for d = −g."""
    res = numopt.run("cg_polak_ribiere", problems.get("himmelblau"), line_search="exact_quadratic")
    assert_valid_result(res)
    assert not res.converged and "dᵀ∇²f d" in res.message
    assert res.n_iter == 0


def test_unbounded_below_fails_in_the_line_search() -> None:
    prob = Problem(
        id="linear",
        name="linear",
        latex="",
        f=lambda x: -float(x[0]) + 0.0 * float(x[1]),
        dim=2,
        domain=(),
        grad=lambda x: np.array([-1.0, 0.0]),
        x0=[0.0, 0.0],
    )
    res = numopt.run("cg_hager_zhang", prob)
    assert_valid_result(res)
    assert not res.converged and "line search failed" in res.message


def test_rounding_limited_failure_is_reported() -> None:
    """Goldstein–Price (local minimizer, f = 30, λ_max ≈ 4.3e3): f-based line searches cannot
    certify decreases once ‖g‖ ≲ √(2ε·30·4.3e3) ≈ 8e-6, so gtol = 1e-9 is out of reach."""
    res = numopt.run("cg_polak_ribiere", problems.get("goldstein_price"), gtol=1e-9)
    assert_valid_result(res)
    assert not res.converged
    assert "line search failed" in res.message and "‖∇f‖∞" in res.message
    last = res.trace[-1].grad_norm
    assert last is not None and last < 1e-4  # it failed next to the minimizer, not far away


@pytest.mark.parametrize("method", METHODS)
def test_default_gtol_reaches_goldstein_price_local_minimizer(method: str) -> None:
    """Regression: the default gtol must lie above the attainable accuracy of the f-based line
    search, ≈ √(2ε·30·λ_max) = 7.6e-6 at the local minimizer (−0.6, −0.4) of Goldstein–Price that
    the default x0 = (−1, 1) reaches (module docstring). With the old default 1e-6, PR+, HS and
    DY stopped with "line search failed"; SciPy's CG (gtol 1e-5) succeeds from the same start."""
    spec = numopt.core.registry.get_method(method)
    assert [p.default for p in spec.params if p.name == "gtol"] == [1e-5]
    prob = problems.get("goldstein_price")
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    ref = minimize(prob.f, prob.x0, jac=prob.grad, method="CG")
    assert ref.success
    # ‖x − x*‖ ≤ ‖∇f‖₂/λ_min ≤ √2·1e-5/448 = 3.2e-8 for both runs (λ_min(∇²f(x*)) ≈ 448).
    assert_allclose(res.x, [-0.6, -0.4], rtol=0, atol=4e-8)
    assert_allclose(res.x, ref.x, rtol=0, atol=8e-8)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["matyas", "quadratic_ill"])
def test_gtol_zero_stops_at_gradient_underflow(method: str, pid: str) -> None:
    """Regression: with gtol = 0 the iterates reach ‖g‖₂ < 1.5e-154, where ‖g‖₂² underflows.
    The driver divided by ‖g‖₂² (Powell's test) and by ‖g‖₂ (first trial step) and raised
    ZeroDivisionError. It must stop with converged=False, no exception and no RuntimeWarning, and
    report the true ‖g‖₂ > 0 (np.linalg.norm returns 0 there)."""
    prob = problems.get(pid)
    assert prob.grad is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run(method, prob, gtol=0.0)
    assert_valid_result(res, max_iter=1000)
    g = np.asarray(prob.grad(res.x))
    last = res.trace[-1]
    assert last.info["direction"] is None
    if res.converged:
        # On matyas PR+, HS and HZ land on ∇f = 0 exactly: gtol = 0 is then honestly met.
        assert not np.any(g) and last.grad_norm == 0.0
        return
    assert "underflows" in res.message, res.message
    assert float(np.max(np.abs(g))) > 0.0
    assert last.grad_norm is not None and 0.0 < last.grad_norm < 1.5e-154
    # Oracle: math.hypot scales internally (correctly rounded up to a few ulp).
    assert last.grad_norm == pytest.approx(math.hypot(*g), rel=4 * EPS)


def test_gradient_below_float_range_at_x0() -> None:
    """f = c‖x‖² with c = 2⁻⁵⁶⁶ ≈ 1.6e-170: ‖g₀‖₂² underflows at x₀ = (3, −1). Before the fix the
    first trial step α₀ = 1/‖g₀‖₂ raised ZeroDivisionError (‖g₀‖₂ computed as 0)."""
    c = 2.0**-566
    prob = Problem(
        id="tiny",
        name="tiny",
        latex="",
        f=lambda x: c * float(x @ x),
        dim=2,
        domain=(),
        grad=lambda x: 2.0 * c * x,
        x0=[3.0, -1.0],
    )
    for method in METHODS:
        res = numopt.run(method, prob, gtol=1e-200)
        assert_valid_result(res)
        assert not res.converged and "underflows" in res.message and res.n_iter == 0
        assert res.trace[0].grad_norm == pytest.approx(2.0 * c * math.sqrt(10.0), rel=2 * EPS)
        assert (res.n_fev, res.n_gev) == (1, 1)


@settings(max_examples=1000, deadline=None)
@given(
    v=arrays(np.float64, st.integers(1, 8), elements=st.floats(-1.0, 1.0)),
    e=st.integers(-1070, 1020),
)
def test_scaled_norm(v: np.ndarray, e: int) -> None:
    """_norm2 equals math.hypot (which scales) to a few ulp over the whole float range, and equals
    np.linalg.norm bit for bit whenever no square under- or overflows (the CG path is unchanged
    in the normal range)."""
    with np.errstate(under="ignore", over="ignore"):
        w = np.ldexp(v, e)
    if not np.all(np.isfinite(w)):
        return
    ref = math.hypot(*w)
    got = cg._norm2(w)
    assert got == pytest.approx(ref, rel=4 * EPS, abs=0.0) or (ref == 0.0 and got == 0.0)
    nz = np.abs(w[w != 0.0])
    if nz.size and float(nz.min()) >= 1.5e-154 and float(nz.max()) <= 1e153:
        assert got == float(np.linalg.norm(w))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"line_search": "backtracking"},
        {"c2": 1.0},
        {"c2": 1e-5},
        {"max_iter": 0},
        {"gtol": -1.0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        numopt.run("cg_dai_yuan", problems.get("rosenbrock"), **kwargs)


def test_fixture_cases_are_valid() -> None:
    for method, pid, params in cg.FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert res.converged, (method, pid, res.message)
        assert len(res.trace) < 300
    assert {m for m, _, _ in cg.FIXTURE_CASES} == set(METHODS)


# --------------------------------------------------------------------------------------
# Regression: huge objective scales (overflow side of the float range)
# --------------------------------------------------------------------------------------


def _scaled(pid: str, c: float) -> Problem:
    """The library problem with f, ∇f and ∇²f multiplied by c (exact for c = 2ᵉ)."""
    base = problems.get(pid)
    f_fn, g_fn, h_fn = base.f, base.grad, base.hess
    assert g_fn is not None and h_fn is not None
    return dataclasses.replace(
        base,
        f=lambda x: c * float(f_fn(x)),
        grad=lambda x: c * np.asarray(g_fn(x)),
        hess=lambda x: c * np.asarray(h_fn(x)),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("line_search", ["strong_wolfe", "exact_quadratic"])
@pytest.mark.parametrize("pid", ["booth", "rosenbrock", "quadratic_ill", "himmelblau", "beale"])
@pytest.mark.parametrize("e", [488, 496, 500, 512, 600])
def test_huge_objective_scale_never_raises(method: str, line_search: str, pid: str, e: int) -> None:
    """Regression: with f scaled by 2^e, e ≳ 488, the driver raised. ‖g‖₂ ≳ 1.3e154 made
    g_kᵀd_k overflow to −inf (ValueError in the line search); ‖g‖₂ ≈ 1e146 to 1e154 drove the
    step lengths below 1e-154, and the zoom divided by an underflowed h² (ZeroDivisionError).
    The contract forbids raising on numerical breakdown: the run must end with a Result, without
    a RuntimeWarning, and converged=True only when ‖∇f‖∞ ≤ gtol holds."""
    prob = _scaled(pid, 2.0**e)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run(method, prob, line_search=line_search)
    assert_valid_result(res, max_iter=1000)
    assert prob.grad is not None
    ginf = float(np.max(np.abs(np.asarray(prob.grad(res.x)))))
    if res.converged:
        assert ginf <= 1e-5
    else:
        assert res.message


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("line_search", ["strong_wolfe", "exact_quadratic"])
def test_gradient_above_float_range_at_x0(method: str, line_search: str) -> None:
    """f = c‖x‖² with c = 2⁶⁰⁰ ≈ 4.1e180: ‖g₀‖₂ = 2c√10 ≈ 2.6e181, so ‖g₀‖₂² and the slope
    g₀ᵀd₀ = −‖g₀‖₂² overflow. Before the fix the line search raised ValueError on the slope −inf.
    The mirror of ``test_gradient_below_float_range_at_x0``."""
    c = 2.0**600
    prob = Problem(
        id="huge",
        name="huge",
        latex="",
        f=lambda x: c * float(x @ x),
        dim=2,
        domain=(),
        grad=lambda x: 2.0 * c * x,
        hess=lambda x: 2.0 * c * np.eye(2),
        x0=[3.0, -1.0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run(method, prob, line_search=line_search)
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0
    assert "overflows" in res.message and "rescale f" in res.message
    assert res.trace[0].info["direction"] is None
    # Oracle: ‖g₀‖₂ = 2c‖x₀‖₂ = 2c√10 (math.hypot scales internally).
    assert res.trace[0].grad_norm == pytest.approx(2.0 * c * math.sqrt(10.0), rel=2 * EPS)
    assert (res.n_fev, res.n_gev, res.n_hev) == (1, 1, 0)


@pytest.mark.parametrize(
    ("pid", "method"),
    [
        ("booth", "cg_fletcher_reeves"),
        ("booth", "cg_dai_yuan"),
        ("rosenbrock", "cg_fletcher_reeves"),
        ("rosenbrock", "cg_polak_ribiere"),
        ("rosenbrock", "cg_dai_yuan"),
    ],
)
def test_tiny_steps_in_the_zoom_are_reported_not_raised(pid: str, method: str) -> None:
    """f scaled by 2⁴⁹⁶ ≈ 2e149: ∇²f ≈ 1e150, so the steps are α ≈ 1e-150. Once α‖d‖ reaches
    the spacing of floats at x (‖∇f‖ ≈ λ·ulp(x) ≈ 1e134 ≫ gtol), the zoom's brackets fall below
    ≈ 1.5e-162, where h² underflows. In these five runs ZeroDivisionError escaped from the line
    search before the fix. The run must fail with a message, and the counts must stay exact."""
    prob, counts = _counting(_scaled(pid, 2.0**496))
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=1000)
    assert not res.converged
    # "broke down" from the driver's guard, or "line search failed" once the zoom bisects.
    assert "broke down" in res.message or "line search failed" in res.message, res.message
    assert (res.n_fev, res.n_gev, res.n_hev) == (counts["f"], counts["g"], counts["h"])


def test_overflowing_cg_direction_restarts_as_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """β_k = 1e308 makes β_k d_{k−1} (or g_kᵀd_k) overflow whenever |d_{k−1}| has an entry > 1.8.
    Before the fix −inf passed the descent test g_kᵀd_k < 0 and reached the line search. The
    direction must restart as d_k = −g_k with restart = "breakdown" and a finite slope."""
    monkeypatch.setattr(cg, "_beta", lambda rule, g, g_prev, d_prev: (1e308, 1e308))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run("cg_fletcher_reeves", problems.get("rosenbrock_nd"), max_iter=40)
    assert_valid_result(res, max_iter=40)
    restarts = [s.info["restart"] for s in res.trace if s.info["direction"] is not None]
    assert "breakdown" in restarts
    for s in res.trace:
        if s.info["restart"] == "breakdown":
            g = -np.asarray(s.info["direction"])
            assert s.info["beta"] == 0.0
            assert s.info["descent"] == pytest.approx(-float(g @ g), rel=4 * EPS)
        if s.info["descent"] is not None:
            assert math.isfinite(s.info["descent"]) and s.info["descent"] < 0.0


# --------------------------------------------------------------------------------------
# Regression: dim = 1 Problems follow the scalar convention (floats in, floats out)
# --------------------------------------------------------------------------------------


def _scalar_only_problem(a: float, b: float, x0: float) -> Problem:
    """f(x) = a(x − b)² + 1 with callables that accept only floats (the dim = 1 convention)."""

    def f(x: float) -> float:
        assert isinstance(x, float), type(x)
        return a * (x - b) ** 2 + 1.0

    def grad(x: float) -> float:
        assert isinstance(x, float), type(x)
        return 2.0 * a * (x - b)

    def hess(x: float) -> float:
        assert isinstance(x, float), type(x)
        return 2.0 * a

    return Problem(
        id="q1", name="q1", latex="", f=f, grad=grad, hess=hess, dim=1, domain=(-10, 10), x0=x0
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("line_search", ["strong_wolfe", "exact_quadratic"])
def test_library_problem_with_dim_1(method: str, line_search: str) -> None:
    """Regression: problems.get("quadratic_1d") (dim = 1, f = (x − 2)² + 1) raised TypeError
    ("only 0-dimensional arrays can be converted to Python scalars"): the driver passed an
    array of shape (1,) to callables that follow the float convention of core.types."""
    prob = problems.get("quadratic_1d")
    res = numopt.run(method, prob, line_search=line_search, gtol=1e-10)
    assert_valid_result(res)
    assert res.converged, res.message
    assert np.shape(res.x) == (1,)
    # Oracle: the closed-form minimizer x* = 2, f* = 1; |x − x*| = |f′(x)|/2 ≤ gtol/2.
    assert_allclose(res.x, [2.0], rtol=0, atol=0.5e-10)
    assert res.fun == pytest.approx(1.0, rel=0, abs=1e-15)
    if line_search == "exact_quadratic":
        assert res.n_iter == 1  # linear CG terminates in n = 1 step (N&W Thm. 5.1)


@pytest.mark.parametrize("method", METHODS)
def test_scalar_only_callables_with_dim_1(method: str) -> None:
    """f(x) = cos x + x²/10 written with math.* (rejects arrays); from x₀ = 1 descent reaches
    the local minimizer x* ≈ 2.5957 where f′(x) = −sin x + x/5 = 0 (oracle: SciPy brentq)."""
    from scipy.optimize import brentq

    prob = Problem(
        id="cos",
        name="cos",
        latex="",
        f=lambda x: math.cos(x) + 0.1 * x * x,
        grad=lambda x: -math.sin(x) + 0.2 * x,
        hess=lambda x: -math.cos(x) + 0.2,
        dim=1,
        domain=(-5.0, 5.0),
        x0=1.0,
    )
    res = numopt.run(method, prob, gtol=1e-9)
    assert_valid_result(res)
    assert res.converged, res.message
    x_star = cast(float, brentq(lambda x: -math.sin(x) + 0.2 * x, 2.0, 3.0, xtol=1e-15))
    # f″(x*) = −cos x* + 0.2 ≈ 1.05, so |x − x*| ≈ |f′(x)|/f″ ≤ 1e-9/1.05 (+ rounding).
    assert_allclose(res.x, [x_star], rtol=0, atol=1e-9)


@settings(max_examples=1000, deadline=None)
@given(
    a=st.floats(1e-2, 1e2),
    b=st.floats(-10.0, 10.0),
    x0=st.floats(-10.0, 10.0),
    method=st.sampled_from(METHODS),
)
def test_dim_1_quadratic_exact_line_search_one_step(
    a: float, b: float, x0: float, method: str
) -> None:
    """On f(x) = a(x − b)² + 1 (dim = 1, float-only callables) CG with the exact line search is
    linear CG with n = 1: it stops after at most one step (N&W Thm. 5.1), and strong convexity
    gives |x − b| = |f′(x)|/(2a) ≤ gtol/(2a)."""
    gtol = 1e-8
    res = numopt.run(
        method, _scalar_only_problem(a, b, x0), line_search="exact_quadratic", gtol=gtol
    )
    assert res.converged, res.message
    assert res.n_iter <= 1
    assert abs(float(res.x[0]) - b) <= gtol / (2.0 * a)


def test_bare_callable_in_one_dimension() -> None:
    """A bare callable with a one-entry x0 becomes a dim = 1 Problem (core.counting), so it
    receives floats like the library's scalar problems, directly and through numopt.minimize.
    A size-1 array result is accepted as f(x); a result with more entries is invalid input."""

    def f(x: float) -> float:
        assert isinstance(x, float), type(x)
        return math.cos(x) + 0.1 * x * x

    via_minimize = numopt.minimize(f, x0=[1.0], method="cg_hager_zhang", gtol=1e-9)
    direct = cg.cg_hager_zhang(f, x0=[1.0], gtol=1e-9)
    for res in (via_minimize, direct):
        assert_valid_result(res)
        assert res.converged, res.message
        assert_allclose(res.x, [2.595739079649799], rtol=0, atol=1e-9)  # brentq root of f′
    arr = cg.cg_polak_ribiere(lambda x: np.array([(x - 2.0) ** 2 + 1.0]), x0=[0.5])
    assert arr.converged and arr.fun == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError, match="scalar"):
        numopt.minimize(lambda x: np.array([1.0, 2.0]) * x, x0=[0.5], method="cg_dai_yuan")
