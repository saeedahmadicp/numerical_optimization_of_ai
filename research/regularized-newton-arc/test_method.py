"""Tests for ARC and gradient-regularized Newton (research/regularized-newton-arc/method.py).

Run: .venv/bin/python -m pytest research/regularized-newton-arc -q
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import problems as study  # noqa: E402
from method import (  # noqa: E402
    _EPS,
    _ROUNDING,
    PARAMS,
    _reg_solve,
    arc,
    cubic_cauchy,
    cubic_model,
    cubic_subproblem,
    reg_newton,
)

from numopt import problems  # noqa: E402
from numopt.core.types import Problem  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "numopt_conftest", HERE.parents[1] / "tests" / "conftest.py"
)
assert _spec is not None and _spec.loader is not None
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
assert_valid_result = _conftest.assert_valid_result

VARIANTS = ("adan", "fixed", "super_universal")


def _quadratic(A: np.ndarray, x0: np.ndarray) -> Problem:
    return Problem(
        id="quad",
        name="quad",
        latex="",
        # dim 1 follows the scalar convention of numopt.core.types (x arrives as a float).
        f=lambda x: 0.5 * float(np.atleast_1d(x) @ A @ np.atleast_1d(x)),
        grad=lambda x: A @ np.atleast_1d(x),
        hess=lambda x: A,
        dim=A.shape[0],
        domain=(),
        x0=x0,
    )


# --------------------------------------------------------------------------------------
# The cubic subproblem
# --------------------------------------------------------------------------------------


def _brute_force_cubic(g: np.ndarray, B: np.ndarray, sigma: float, seed: int) -> float:
    """min_s gᵀs + ½sᵀBs + (σ/3)‖s‖³ by multistart BFGS (independent of Thm. 3.1)."""
    n = g.size
    lam1 = float(np.linalg.eigvalsh(B)[0])
    # Every global minimizer has ‖s‖ = λ/σ ≤ R (the bracket bound hi of cubic_subproblem).
    R = (abs(lam1) + math.sqrt(lam1 * lam1 + 4.0 * sigma * float(np.linalg.norm(g)))) / sigma
    rng = np.random.default_rng(seed)

    def m(s: np.ndarray) -> float:
        return cubic_model(g, B, sigma, s)

    def dm(s: np.ndarray) -> np.ndarray:
        return g + B @ s + sigma * float(np.linalg.norm(s)) * s

    best = math.inf
    starts = [np.zeros(n), *(R * rng.uniform(-1, 1, size=n) for _ in range(40))]
    if n == 2:  # a polar grid as well, to catch a missed basin
        for r in np.linspace(0.05, 1.0, 12) * R:
            for t in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
                starts.append(np.array([r * np.cos(t), r * np.sin(t)]))
    for s0 in starts:
        res = minimize(m, s0, jac=dm, method="BFGS", options={"gtol": 1e-13, "maxiter": 2000})
        best = min(best, float(res.fun))
    return best


def test_cubic_subproblem_zero_hessian_closed_form() -> None:
    # B = 0: s = −g/λ with λ = σ‖s‖ ⇒ λ = √(σ‖g‖) (CGT Thm. 3.1).
    g = np.array([3.0, 4.0])
    sub = cubic_subproblem(g, np.zeros((2, 2)), 2.0)
    lam = math.sqrt(2.0 * 5.0)
    assert sub.solved and not sub.hard_case
    assert_allclose(sub.lam, lam, rtol=1e-12)
    assert_allclose(sub.s, -g / lam, rtol=1e-12, atol=0)


def test_cubic_subproblem_hard_case_hand_computed() -> None:
    # quartic_saddle at (2, 0): g = (4, 0), B = diag(2, −2), σ = 1. γ is 0 on E₁ = span(e₂),
    # s_⊥ = −4/(2 + 2) e₁, ‖s_⊥‖ = 1 ≤ −λ₁/σ = 2 ⇒ hard case, s = (−1, ±√3), λ = 2 (CGT eq. 6.6).
    sub = cubic_subproblem(np.array([4.0, 0.0]), np.diag([2.0, -2.0]), 1.0)
    assert sub.hard_case and sub.solved
    assert_allclose(sub.lam, 2.0, rtol=1e-14)
    assert_allclose(sub.s, [-1.0, math.sqrt(3.0)], rtol=1e-14, atol=1e-15)


def test_cubic_subproblem_matches_reg_newton_when_hessian_is_zero() -> None:
    # Mishchenko (2023) §1: with B = 0 the cubic step is the regularized step with λ = √(σ‖g‖).
    g = np.array([0.3, -1.7, 2.2])
    sigma = 0.7
    sub = cubic_subproblem(g, np.zeros((3, 3)), sigma)
    eigvals, Q = np.linalg.eigh(np.zeros((3, 3)))
    s, pd = _reg_solve(eigvals, Q, g, math.sqrt(sigma * float(np.linalg.norm(g))))
    assert s is not None and pd
    assert_allclose(sub.s, s, rtol=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_cubic_subproblem_is_global_minimizer(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2 if seed < 25 else int(rng.integers(3, 6))
    M = rng.normal(size=(n, n))
    B = 0.5 * (M + M.T) * 10.0 ** rng.uniform(-1, 1)
    g = rng.normal(size=n) * 10.0 ** rng.uniform(-2, 2)
    sigma = 10.0 ** rng.uniform(-2, 2)
    sub = cubic_subproblem(g, B, sigma)
    m_ours = cubic_model(g, B, sigma, sub.s)
    m_oracle = _brute_force_cubic(g, B, sigma, seed)
    scale = abs(m_oracle) + float(np.linalg.norm(g)) * float(np.linalg.norm(sub.s))
    # NOTE: 1e-10·scale: the oracle (BFGS, gtol 1e-13) and m(s) are both evaluated in float64 from
    # terms of size ‖g‖‖s‖; a missed basin would differ at O(scale).
    assert m_ours <= m_oracle + 1e-10 * scale
    assert_allclose(sub.predicted, -m_ours, rtol=1e-9, atol=1e-12 * scale)


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(1, 6),
    seed=st.integers(0, 2**32 - 1),
    log_sigma=st.floats(-6, 6),
    log_g=st.floats(-8, 8),
    log_b=st.floats(-4, 4),
    hard=st.booleans(),
)
def test_cubic_subproblem_optimality_conditions(
    n: int, seed: int, log_sigma: float, log_g: float, log_b: float, hard: bool
) -> None:
    """CGT Thm. 3.1: (B + λI)s = −g, λ = σ‖s‖, B + λI ⪰ 0; and m(s) ≤ m(s^C) (eq. 2.2)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.sort(rng.normal(size=n)) * 10.0**log_b
    B = (Q * lam) @ Q.T
    B = 0.5 * (B + B.T)
    g = rng.normal(size=n)
    if hard and n >= 2 and lam[0] < 0:
        g = g - (Q[:, 0] @ g) * Q[:, 0]  # g ⊥ eigenvector of λ₁: the (potential) hard case
    g = g / np.linalg.norm(g) * 10.0**log_g
    sigma = 10.0**log_sigma
    sub = cubic_subproblem(g, B, sigma)
    assert sub.solved
    s = sub.s
    bnorm = float(np.max(np.abs(lam)))
    snorm = float(np.linalg.norm(s))
    # NOTE: tolerances: the eigendecomposition is backward stable, so ‖(B + λI)s + g‖ is at the
    # rounding level of the terms, ~n·ε·(‖B‖ + λ)‖s‖ + ε‖g‖; 1e3 allows for cond(Q) and n ≤ 6.
    resid = float(np.linalg.norm((B + sub.lam * np.eye(n)) @ s + g))
    assert resid <= 1e3 * _EPS * ((bnorm + sub.lam) * snorm + float(np.linalg.norm(g)))
    assert abs(sub.lam - sigma * snorm) <= 1e-10 * sub.lam
    assert sub.lam + lam[0] >= -1e3 * _EPS * max(bnorm, sub.lam)
    m_s = cubic_model(g, B, sigma, s)
    m_c = cubic_model(g, B, sigma, cubic_cauchy(g, B, sigma))
    assert m_s <= m_c + 1e3 * _EPS * (abs(m_c) + float(np.linalg.norm(g)) * snorm)


def test_cubic_cauchy_point_minimizes_along_minus_gradient() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        M = rng.normal(size=(3, 3))
        B = M + M.T
        g = rng.normal(size=3)
        sigma = 10.0 ** rng.uniform(-2, 2)
        sc = cubic_cauchy(g, B, sigma)
        t = float(np.linalg.norm(sc))
        u = g / np.linalg.norm(g)
        ts = np.linspace(0.0, 3 * t + 1.0, 20001)
        vals = [cubic_model(g, B, sigma, -tt * u) for tt in ts]
        assert cubic_model(g, B, sigma, sc) <= min(vals) + 1e-12 * (1 + abs(min(vals)))
        assert_allclose(sc / t, -u, rtol=1e-12)


# --------------------------------------------------------------------------------------
# ARC
# --------------------------------------------------------------------------------------


def test_arc_first_step_hand_computed_escapes_saddle_manifold() -> None:
    p = study.get("quartic_saddle")
    res = arc(p, x0=(2.0, 0.0))
    assert_valid_result(res, max_iter=200)
    s1 = res.trace[1]
    assert s1.info["hard_case"] is True
    assert_allclose(s1.x, [1.0, math.sqrt(3.0)], rtol=1e-14)
    # f(2,0) = 4, m(s) = 4 + (−4) + ½(2·1 − 2·3) + (1/3)·8 = 2/3 → predicted 10/3; f(1,√3) = −5/4.
    assert_allclose(s1.info["predicted"], 10.0 / 3.0, rtol=1e-13)
    assert_allclose(s1.info["actual"], 4.0 - (1.0 - 3.0 + 9.0 / 4.0), rtol=1e-13)
    assert res.converged
    assert_allclose(np.abs(res.x), [0.0, math.sqrt(2.0)], atol=1e-8)


@pytest.mark.parametrize(
    "pid, minima",
    [
        ("rosenbrock", [(1.0, 1.0)]),
        ("himmelblau", None),
        ("beale", [(3.0, 0.5)]),
    ],
)
def test_arc_converges_on_library_problems(pid: str, minima) -> None:
    p = problems.get(pid)
    res = arc(p)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    targets = minima if minima is not None else p.minima
    assert min(float(np.linalg.norm(res.x - np.asarray(m))) for m in targets) < 1e-6
    assert res.extra["lambda_min"] > 0.0


@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "six_hump_camel"])
def test_arc_agrees_with_scipy_trust_exact(pid: str) -> None:
    p = problems.get(pid)
    ours = arc(p)
    ref = minimize(
        p.f,
        np.asarray(p.x0, float),
        jac=p.grad,
        hess=p.hess,
        method="trust-exact",
        options={"gtol": 1e-8},
    )
    assert ours.converged and ref.success
    assert_allclose(ours.x, ref.x, atol=1e-7)


@pytest.mark.parametrize("pid", study.CONVEX + study.NONCONVEX_LOCAL)
def test_arc_invariants_on_study_problems(pid: str) -> None:
    p = study.get(pid)
    x0s = [(9.0, -7.0), (-4.5, 10.5), (0.3, -0.2)]
    for x0 in x0s:
        res = arc(p, x0=x0, max_iter=500)
        assert_valid_result(res, max_iter=500)
        assert res.converged, res.message
        fs = [s.fun for s in res.trace]
        sig = res.trace[0].info["sigma"]
        for prev, step in zip(res.trace, res.trace[1:], strict=False):
            info = step.info
            assert info["sigma"] == sig
            assert info["accepted"] == (info["rho"] is not None and info["rho"] >= 0.1)
            if info["iteration"] == "very_successful":
                assert info["new_sigma"] <= info["sigma"]
            elif info["iteration"] == "unsuccessful":
                assert info["new_sigma"] == 2.0 * info["sigma"]
                assert np.array_equal(step.x, prev.x)
            sig = info["new_sigma"]
        # Accepted steps decrease f (up to the rounding allowance δ of the module docstring).
        for a, b in itertools.pairwise(fs):
            assert b <= a + _ROUNDING * _EPS * max(abs(a), abs(b))
        # Exact Hessian count: one at x0 plus one per accepted step.
        assert res.n_hev == 1 + sum(bool(s.info["accepted"]) for s in res.trace[1:])
        assert res.n_fev == 1 + res.n_iter


def test_arc_counts_are_exact() -> None:
    base = problems.get("rosenbrock")
    calls = {"f": 0, "g": 0, "h": 0}

    def f(x):
        calls["f"] += 1
        return base.f(x)

    def g(x):
        calls["g"] += 1
        return base.grad(x)

    def h(x):
        calls["h"] += 1
        return base.hess(x)

    p = Problem(id="r", name="r", latex="", f=f, grad=g, hess=h, dim=2, domain=(), x0=base.x0)
    for fn in (arc, reg_newton):
        calls.update(f=0, g=0, h=0)
        res = fn(p)
        assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], calls["h"])


def test_arc_stops_at_stationary_start_and_reports_saddle() -> None:
    # The second-order part of the stopping test: ‖∇f‖∞ = 0 ≤ gtol at the saddle of
    # x² − y² + y⁴/4, but ∇²f = diag(2, −2) ⇒ not a minimizer ⇒ converged=False (numopt Newton
    # convention).
    res = arc(study.get("quartic_saddle"), x0=(0.0, 0.0))
    assert not res.converged and res.n_iter == 0
    assert "NOT a minimizer" in res.message and "a saddle point" in res.message
    assert res.extra["lambda_min"] == -2.0
    assert_valid_result(res)


@pytest.mark.parametrize("variant", ["adan", "super_universal", "fixed"])
def test_reg_newton_saddle_stop_is_not_converged(variant: str) -> None:
    # From (3, 0), ∇f ⟂ e₂ and every iterate stays on y = 0 (the stable manifold of the saddle):
    # the gradient test passes at (0, 0), where ∇²f = diag(2, −2).
    res = reg_newton(study.get("quartic_saddle"), x0=(3.0, 0.0), variant=variant, H=0.5)
    assert float(np.max(np.abs(res.x))) <= 1e-8
    assert not res.converged and "NOT a minimizer" in res.message, res.message
    assert res.extra["lambda_min"] == pytest.approx(-2.0)
    assert_valid_result(res)


def test_second_order_stop_maximizer_and_semidefinite() -> None:
    # Maximizer: f = −x² − y² at its stationary point 0.
    mx = Problem(
        id="m",
        name="m",
        latex="",
        f=lambda x: -float(x @ x),
        grad=lambda x: -2.0 * np.asarray(x),
        hess=lambda x: -2.0 * np.eye(2),
        dim=2,
        domain=(),
        x0=(0.0, 0.0),
    )
    for fn in (arc, reg_newton):
        res = fn(mx)
        assert not res.converged and "a maximizer" in res.message, res.message
    # Positive semidefinite, singular: f = x⁴ + y² at 0 (a minimizer; ∇²f = diag(0, 2)).
    psd = Problem(
        id="p",
        name="p",
        latex="",
        f=lambda x: x[0] ** 4 + x[1] ** 2,
        grad=lambda x: np.array([4.0 * x[0] ** 3, 2.0 * x[1]]),
        hess=lambda x: np.diag([12.0 * x[0] ** 2, 2.0]),
        dim=2,
        domain=(),
        x0=(0.0, 0.0),
    )
    for fn in (arc, reg_newton):
        res = fn(psd)
        assert res.converged and "semidefinite" in res.message, res.message
    # Central-difference ∇f and ∇²f at a minimizer of (x + y)² (∇²f = [[2, 2], [2, 2]], singular):
    # the finite-difference eigenvalue 0 ± O(ε^{1/3}) must not be read as a saddle.
    for fn in (arc, reg_newton):
        res = fn(lambda x: (x[0] + x[1]) ** 2, x0=[1.0, -1.0], gtol=1e-6)
        assert res.converged, res.message


def test_arc_failure_paths() -> None:
    p = problems.get("rosenbrock")
    res = arc(p, max_iter=2)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=2)
    bad = Problem(
        id="b",
        name="b",
        latex="",
        f=lambda x: math.nan,
        grad=lambda x: np.ones(2),
        hess=lambda x: np.eye(2),
        dim=2,
        domain=(),
        x0=(1.0, 1.0),
    )
    res = arc(bad)
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)
    for kw in ({"sigma0": 0.0}, {"eta1": 0.95, "eta2": 0.9}, {"gamma": 1.0}, {"max_iter": 0}):
        with pytest.raises(ValueError):
            arc(p, **kw)


def test_arc_bare_callable_and_fd_derivatives() -> None:
    res = arc(
        lambda x: (x[0] - 1.0) ** 2 + 10.0 * (x[1] + 2.0) ** 2 + 0.1 * x[0] ** 4,
        x0=[3.0, 3.0],
        gtol=1e-6,
    )
    assert res.converged
    # x* solves 2(x − 1) + 0.4x³ = 0 (the real root of 0.2x³ + x − 1), y* = −2.
    roots = np.roots([0.2, 0.0, 1.0, -1.0])
    x_star = float(roots[np.abs(roots.imag) < 1e-12].real[0])
    # NOTE: atol 1e-6: ∇f and ∇²f are central differences; gtol = 1e-6 bounds ‖x − x*‖ by
    # ≈ gtol/λ_min(∇²f) = 1e-6/2.9.
    assert_allclose(res.x, [x_star, -2.0], atol=1e-6)


# --------------------------------------------------------------------------------------
# Gradient-regularized Newton
# --------------------------------------------------------------------------------------


def test_reg_newton_fixed_first_step_hand_computed() -> None:
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x0 = np.array([1.0, -2.0])
    H = 0.5
    res = reg_newton(_quadratic(A, x0), x0=x0, variant="fixed", H=H)
    g = A @ x0
    lam = math.sqrt(H * float(np.linalg.norm(g)))
    x1 = x0 - np.linalg.solve(A + lam * np.eye(2), g)
    assert_allclose(res.trace[1].x, x1, rtol=1e-14)
    assert_allclose(res.trace[1].info["lambda"], lam, rtol=1e-15)
    assert res.converged
    assert_allclose(res.x, [0.0, 0.0], atol=1e-8)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("pid", ["lse", "sqrt1p", "logistic_ridge"])
def test_reg_newton_converges_on_convex_study_problems(variant: str, pid: str) -> None:
    p = study.get(pid)
    # fixed: Assumption 1 needs ∇²f to be 2H-Lipschitz. H = 1 exceeds L̂₂/2 for all three
    # (run.py's grid estimates: L̂₂ ≈ 1.22 for lse, 0.85 for sqrt1p, 0.18 for logistic_ridge).
    res = reg_newton(p, x0=(9.0, -7.0), variant=variant, max_iter=500)
    assert_valid_result(res, max_iter=500)
    assert res.converged, res.message
    assert_allclose(res.x, p.minima[0], atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_reg_newton_library_problems(variant: str) -> None:
    for pid in ("rosenbrock", "six_hump_camel"):
        res = reg_newton(problems.get(pid), variant=variant)
        assert_valid_result(res, max_iter=200)
        assert res.converged, res.message


def test_reg_newton_fixed_beats_pure_newton_divergence() -> None:
    # Pure Newton maps r → −r³ on sqrt(1 + ‖x‖²) (diverges from ‖x0‖ = 5).
    from numopt import run

    p = study.get("sqrt1p")
    assert not run("pure_newton", p, x0=(3.0, 4.0)).converged
    res = reg_newton(p, x0=(3.0, 4.0), variant="fixed", H=1.0)
    assert res.converged and float(np.linalg.norm(res.x)) < 1e-8


@settings(max_examples=1000, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(1, 5),
    log_h=st.floats(-3, 3),
    log_scale=st.floats(-3, 3),
    variant=st.sampled_from(VARIANTS),
)
def test_reg_newton_monotone_and_acceptance_tests_on_convex_quadratics(
    seed: int, n: int, log_h: float, log_scale: float, variant: str
) -> None:
    """Mishchenko Lemma 3 / eq. (12): f(x_{k+1}) ≤ f(x_k) for convex f (∇²f constant ⇒ any H
    satisfies Assumption 1); AdaN's two acceptance tests and DMN's test hold at every accepted step.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M @ M.T + 0.1 * np.eye(n)
    x0 = rng.normal(size=n) * 10.0**log_scale
    p = _quadratic(A, x0)
    res = reg_newton(p, x0=x0, variant=variant, H=10.0**log_h, max_iter=300)
    assert_valid_result(res, max_iter=300)
    if variant == "fixed":
        # A large H on a quadratic (true H = 0) makes λ ≫ ∇²f: gradient-like steps, still in the
        # O(1/k²) phase of Thm. 1 at max_iter (e.g. H = 100, ‖x0‖ = 10³). Only max_iter may stop it.
        assert res.converged or res.message.startswith("reached max_iter"), res.message
    else:
        assert res.converged, res.message
    for prev, step in zip(res.trace, res.trace[1:], strict=False):
        delta = _ROUNDING * _EPS * max(abs(prev.fun), abs(step.fun))
        assert step.fun <= prev.fun + delta
        lam = step.info["lambda"]
        r = float(np.linalg.norm(step.info["direction"]))
        if variant == "adan":
            assert step.grad_norm <= 2.0 * lam * r * (1 + 1e-12)
            assert step.fun <= prev.fun - (2.0 / 3.0) * lam * r * r + delta
        if variant == "super_universal":
            g1 = A @ np.asarray(step.x)
            lhs = float(g1 @ (np.asarray(prev.x) - np.asarray(step.x)))
            assert lhs >= float(g1 @ g1) / (4.0 * lam) * (1 - 1e-12)
        assert step.info["pd"] is True and step.info["descent"] is True


def test_reg_newton_adan_doubling_schedule() -> None:
    # Alg. 2: the first trial at k = 0 uses 2H₀; every later iteration starts from H_{k−1}/2.
    res = reg_newton(study.get("lse"), x0=(9.0, -7.0), variant="adan", H=1.0)
    trials = res.trace[1].info["trials"]
    assert trials[0][0] == 2.0
    for j in range(1, len(trials)):
        assert trials[j][0] == 2.0 * trials[j - 1][0]
    for prev, step in zip(res.trace[1:], res.trace[2:], strict=False):
        assert step.info["trials"][0][0] == prev.info["H_reg"] / 2.0


def test_reg_newton_super_universal_schedule() -> None:
    res = reg_newton(study.get("lse"), x0=(9.0, -7.0), variant="super_universal", H=1.0)
    for prev, step in zip(res.trace[1:], res.trace[2:], strict=False):
        assert step.info["trials"][0][0] == prev.info["H_reg"] / 4.0  # H_{k+1} = 4^{j_k}H_k/4
        for j, t in enumerate(step.info["trials"]):
            assert t[0] == step.info["trials"][0][0] * 4.0**j


def test_reg_newton_failure_paths() -> None:
    # f = −½x² + ½y² at (1, 0): ‖g‖ = 1, H = 1 ⇒ λ = 1 and ∇²f + λI = diag(0, 2) is singular.
    p = Problem(
        id="s",
        name="s",
        latex="",
        f=lambda x: -0.5 * x[0] ** 2 + 0.5 * x[1] ** 2,
        grad=lambda x: np.array([-x[0], x[1]]),
        hess=lambda x: np.diag([-1.0, 1.0]),
        dim=2,
        domain=(),
        x0=(1.0, 0.0),
    )
    res = reg_newton(p, variant="fixed", H=1.0)
    assert not res.converged and "singular" in res.message
    assert_valid_result(res)
    res = reg_newton(problems.get("rosenbrock"), max_iter=3)
    assert not res.converged and "max_iter" in res.message
    with pytest.raises(ValueError):
        reg_newton(p, variant="nope")
    with pytest.raises(ValueError):
        reg_newton(p, alpha=0.5)
    with pytest.raises(ValueError):
        reg_newton(p, H=-1.0)


def test_params_dict_matches_signatures() -> None:
    import inspect

    for name, fn in (("arc", arc), ("reg_newton", reg_newton)):
        sig = inspect.signature(fn)
        kw = {k for k in sig.parameters if k not in ("problem", "x0")}
        assert {p.name for p in PARAMS[name]} == kw
        for spec in PARAMS[name]:
            assert sig.parameters[spec.name].default == spec.default


# --------------------------------------------------------------------------------------
# Study problems: exact derivatives against central differences
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", study.CONVEX + study.NONCONVEX_LOCAL)
def test_study_problem_derivatives(pid: str) -> None:
    p = study.get(pid)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-10, 10, size=2)
        h = 1e-5
        E = np.eye(2)
        g_fd = np.array([(p.f(x + h * e) - p.f(x - h * e)) / (2 * h) for e in E])
        H_fd = np.array([(p.grad(x + h * e) - p.grad(x - h * e)) / (2 * h) for e in E]).T
        # NOTE: central differences with h = 1e-5: truncation O(h²‖f'''‖) + rounding O(ε|f|/h).
        assert_allclose(p.grad(x), g_fd, rtol=1e-6, atol=1e-8 * (1 + abs(p.f(x))))
        assert_allclose(p.hess(x), H_fd, rtol=1e-6, atol=1e-8)
    for xm in p.minima:
        assert float(np.max(np.abs(p.grad(np.asarray(xm))))) <= 1e-12
