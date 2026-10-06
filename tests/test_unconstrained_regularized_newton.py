"""Tests for numopt.unconstrained.regularized_newton: ARC and gradient-regularized Newton.

Oracles: a multistart BFGS minimization of the cubic model (independent of CGT Thm. 3.1); the
optimality conditions of CGT Thm. 3.1; hand-computed steps (closed form for B = 0, the hard case of
CGT eq. 6.6, the first regularized Newton step); a naive ``scipy.linalg.solve`` of
(∇²f + λI)s = −∇f at every accepted step; SciPy ``trust-exact`` for the minimizers; the second-order
classification of the final point (saddle points and maximizers give converged=False).
"""

from __future__ import annotations

import collections
import inspect
import itertools
import json
import math
import zlib
from typing import Any

import numpy as np
import pytest
import scipy.linalg
import scipy.special
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import minimize as sp_minimize

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import Problem
from numopt.unconstrained import regularized_newton as rn
from numopt.unconstrained.regularized_newton import (
    FIXTURE_CASES,
    VARIANTS,
    _reg_solve,
    arc,
    cubic_cauchy,
    cubic_model,
    cubic_subproblem,
    reg_newton,
)

EPS = float(np.finfo(float).eps)
ROUNDING = rn._ROUNDING


# --------------------------------------------------------------------------------------
# Test problems that the library does not have (from research/regularized-newton-arc)
# --------------------------------------------------------------------------------------


def _quadratic(A: np.ndarray, x0: Any) -> Problem:
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


def _quartic_saddle() -> Problem:
    """x² − y² + y⁴/4: a strict saddle at 0 (∇²f = diag(2, −2)), minima (0, ±√2) with f = −1."""
    return Problem(
        id="quartic_saddle",
        name="quartic saddle",
        latex="",
        f=lambda x: float(x[0] ** 2 - x[1] ** 2 + 0.25 * x[1] ** 4),
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1] + x[1] ** 3]),
        hess=lambda x: np.array([[2.0, 0.0], [0.0, -2.0 + 3.0 * x[1] ** 2]]),
        dim=2,
        domain=(),
        x0=(3.0, 0.0),
        minima=((0.0, math.sqrt(2.0)), (0.0, -math.sqrt(2.0))),
    )


def _sqrt1p() -> Problem:
    """√(1 + ‖x‖²): convex; pure Newton maps the radius r to −r³ (diverges for r > 1)."""

    def f(x: Any) -> float:
        v = np.asarray(x, dtype=float)
        return float(np.sqrt(1.0 + v @ v))

    def grad(x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        return v / np.sqrt(1.0 + v @ v)

    def hess(x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        s = float(np.sqrt(1.0 + v @ v))
        return (np.eye(v.size) - np.outer(v, v) / s**2) / s

    return Problem(
        id="sqrt1p",
        name="sqrt1p",
        latex="",
        f=f,
        grad=grad,
        hess=hess,
        dim=2,
        domain=(),
        x0=(3.0, 4.0),
        minima=((0.0, 0.0),),
    )


_LSE_A = np.array([[1.0, 0.5], [-1.0, 1.0], [0.2, -1.0]])  # (3, 2); 0 interior to conv{aᵢ}
_LSE_B = np.array([0.0, 0.5, -0.5])  # (3,)


def _lse() -> Problem:
    """log Σ exp(aᵢᵀx − bᵢ): strictly convex, ∇²f → 0 far from the unique minimizer."""

    def probs(x: Any) -> np.ndarray:
        z = _LSE_A @ np.asarray(x, dtype=float) - _LSE_B
        e = np.exp(z - np.max(z))
        return e / np.sum(e)

    def f(x: Any) -> float:
        z = _LSE_A @ np.asarray(x, dtype=float) - _LSE_B
        return float(np.asarray(scipy.special.logsumexp(z)))

    def grad(x: Any) -> np.ndarray:
        return _LSE_A.T @ probs(x)

    def hess(x: Any) -> np.ndarray:
        p = probs(x)
        return _LSE_A.T @ (np.diag(p) - np.outer(p, p)) @ _LSE_A

    # The minimizer: ∇f = Aᵀp = 0 with p on the simplex ⇒ p ∝ null(Aᵀ); then z − z₀ = log p.
    null = scipy.linalg.null_space(_LSE_A.T)[:, 0]
    p_star = null / null.sum()
    # A x − b = log p + c·1: solve the 3×3 system for (x, c).
    M = np.column_stack([_LSE_A, -np.ones(3)])
    sol = np.linalg.solve(M, np.log(p_star) + _LSE_B)
    return Problem(
        id="lse",
        name="lse",
        latex="",
        f=f,
        grad=grad,
        hess=hess,
        dim=2,
        domain=(),
        x0=(8.0, 8.0),
        minima=(tuple(float(t) for t in sol[:2]),),
    )


#: Relative evaluation error of f in ``_noisy_quartic``: the spread 2·100ε between two points is
#: below the rounding allowance δ/|f| = 10³ε of the module docstring.
_NOISE_REL = 100.0 * EPS
_NOISY_OFFSET = 1e4


def _noisy_quartic() -> Problem:
    """(10⁴ + ‖x‖⁴/4)·(1 + 100ε·u(x)), u(x) ∈ [−1, 1) a CRC-32 hash of the bits of x.

    A model of f evaluated in floating point: a deterministic, non-smooth relative error of 100ε
    on f (≈ 2·10⁻¹⁰ absolute), while ∇f = ‖x‖²x and ∇²f = ‖x‖²I + 2xxᵀ are exact. The minimizer 0
    is degenerate (∇²f(0) = 0), so the convergence is linear and the last iterations ask for a
    decrease (2/3)λr² below the error of f. Then f(x₊) > f(x) − (2/3)λr² happens on correct steps,
    and the rounding allowance δ = 10³ε·max(|f(x)|, |f(x₊)|) is what accepts them.
    """

    def f(x: Any) -> float:
        v = np.ascontiguousarray(x, dtype=np.float64)
        u = zlib.crc32(v.tobytes()) / 2.0**31 - 1.0
        return (_NOISY_OFFSET + 0.25 * float(v @ v) ** 2) * (1.0 + _NOISE_REL * u)

    def grad(x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        return float(v @ v) * v

    def hess(x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        return float(v @ v) * np.eye(v.size) + 2.0 * np.outer(v, v)

    return Problem(
        id="noisy_quartic",
        name="noisy quartic",
        latex="",
        f=f,
        grad=grad,
        hess=hess,
        dim=2,
        domain=(),
        x0=(1.0, -0.5),
        minima=((0.0, 0.0),),
    )


def _counting(base: Problem) -> tuple[Problem, dict[str, int]]:
    calls = {"f": 0, "g": 0, "h": 0}
    assert base.grad is not None and base.hess is not None
    bg, bh = base.grad, base.hess

    def f(x: Any) -> float:
        calls["f"] += 1
        return base.f(x)

    def g(x: Any) -> Any:
        calls["g"] += 1
        return bg(x)

    def h(x: Any) -> Any:
        calls["h"] += 1
        return bh(x)

    p = Problem(
        id="c", name="c", latex="", f=f, grad=g, hess=h, dim=base.dim, domain=(), x0=base.x0
    )
    return p, calls


# --------------------------------------------------------------------------------------
# Registry and contract
# --------------------------------------------------------------------------------------


def test_registered_specs_match_signatures() -> None:
    for mid, fn in (("arc", arc), ("reg_newton", reg_newton)):
        spec = get_method(mid)
        assert spec.fn is fn
        assert spec.family == "unconstrained"
        assert spec.needs == ("f", "grad", "hess")
        sig = inspect.signature(fn)
        kw = {k for k in sig.parameters if k not in ("problem", "x0")}
        assert {p.name for p in spec.params} == kw
        for p in spec.params:
            assert sig.parameters[p.name].default == p.default
            if p.kind in ("float", "int"):
                assert p.min is not None and p.max is not None and p.min <= p.default <= p.max
        assert spec.references and spec.summary


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_fixture_cases_are_valid_and_short(case: tuple[str, str, dict[str, Any]]) -> None:
    mid, pid, params = case
    res = numopt.run(mid, problems.get(pid), **params)
    assert_valid_result(res, max_iter=200)
    assert len(res.trace) < 400
    assert res.n_iter == res.trace[-1].k
    json.dumps(res.to_dict(), allow_nan=False)


def test_fixture_cases_outcomes() -> None:
    # The honest flag on the nonconvex fixture: Himmelblau's local maximizer.
    res = numopt.run("reg_newton", problems.get("himmelblau"), variant="fixed", H=0.5)
    assert not res.converged and "a maximizer" in res.message, res.message
    assert_allclose(res.x, [-0.270845, -0.923039], atol=1e-6)
    assert res.extra["lambda_min"] < 0.0
    # ARC from the same start reaches a minimizer.
    res = numopt.run("arc", problems.get("himmelblau"))
    assert res.converged and res.extra["lambda_min"] > 0.0
    assert_allclose(res.x, [3.0, 2.0], atol=1e-9)


def test_unknown_parameter_rejected_by_run() -> None:
    with pytest.raises(TypeError):
        numopt.run("arc", problems.get("rosenbrock"), sigma=1.0)


# --------------------------------------------------------------------------------------
# The cubic subproblem
# --------------------------------------------------------------------------------------


def _brute_force_cubic(g: np.ndarray, B: np.ndarray, sigma: float, seed: int) -> float:
    """min_s gᵀs + ½sᵀBs + (σ/3)‖s‖³ by multistart BFGS (independent of Thm. 3.1)."""
    n = g.size
    lam1 = float(np.linalg.eigvalsh(B)[0])
    # Every global minimizer has ‖s‖ = λ/σ ≤ R (the bracket bound of cubic_subproblem).
    R = (abs(lam1) + math.sqrt(lam1 * lam1 + 4.0 * sigma * float(np.linalg.norm(g)))) / sigma
    rng = np.random.default_rng(seed)

    def m(s: np.ndarray) -> float:
        return float(g @ s + 0.5 * s @ B @ s + sigma * np.linalg.norm(s) ** 3 / 3.0)

    def dm(s: np.ndarray) -> np.ndarray:
        return g + B @ s + sigma * float(np.linalg.norm(s)) * s

    best = math.inf
    starts = [np.zeros(n), *(R * rng.uniform(-1, 1, size=n) for _ in range(40))]
    if n == 2:  # a polar grid as well, to catch a missed basin
        for r in np.linspace(0.05, 1.0, 12) * R:
            for t in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
                starts.append(np.array([r * np.cos(t), r * np.sin(t)]))
    for s0 in starts:
        res = sp_minimize(m, s0, jac=dm, method="BFGS", options={"gtol": 1e-13, "maxiter": 2000})
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
    # f − m(s) = ‖g‖²/λ − ‖g‖²/λ²·... : m(s) = −‖g‖²/λ + (σ/3)(‖g‖/λ)³ = −5/√10·(1 − 1/3).
    assert_allclose(sub.predicted, (2.0 / 3.0) * 25.0 / lam, rtol=1e-14)


def test_cubic_subproblem_hard_case_hand_computed() -> None:
    # g = (4, 0), B = diag(2, −2), σ = 1: γ is 0 on E₁ = span(e₂), s_⊥ = −4/(2 + 2) e₁,
    # ‖s_⊥‖ = 1 ≤ −λ₁/σ = 2 ⇒ hard case, s = (−1, ±√3), λ = 2 (CGT eq. 6.6).
    sub = cubic_subproblem(np.array([4.0, 0.0]), np.diag([2.0, -2.0]), 1.0)
    assert sub.hard_case and sub.solved
    assert_allclose(sub.lam, 2.0, rtol=1e-14)
    assert_allclose(np.abs(sub.s), [1.0, math.sqrt(3.0)], rtol=1e-14, atol=1e-15)
    assert sub.s[0] < 0.0


def test_cubic_subproblem_matches_regularized_step_when_hessian_is_zero() -> None:
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
    assert resid <= 1e3 * EPS * ((bnorm + sub.lam) * snorm + float(np.linalg.norm(g)))
    assert abs(sub.lam - sigma * snorm) <= 1e-10 * sub.lam
    assert sub.lam + lam[0] >= -1e3 * EPS * max(bnorm, sub.lam)
    m_s = cubic_model(g, B, sigma, s)
    m_c = cubic_model(g, B, sigma, cubic_cauchy(g, B, sigma))
    assert m_s <= m_c + 1e3 * EPS * (abs(m_c) + float(np.linalg.norm(g)) * snorm)


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


def test_arc_first_step_hand_computed_hard_case() -> None:
    res = arc(_quartic_saddle(), x0=(2.0, 0.0))
    assert_valid_result(res, max_iter=200)
    s1 = res.trace[1]
    assert s1.info["hard_case"] is True
    assert_allclose(np.abs(s1.x), [1.0, math.sqrt(3.0)], rtol=1e-14)
    # f(2,0) = 4, m(s) = 4 + (−4) + ½(2·1 − 2·3) + (1/3)·8 = 2/3 → predicted 10/3; f(1,±√3) = −5/4.
    assert_allclose(s1.info["predicted"], 10.0 / 3.0, rtol=1e-13)
    assert_allclose(s1.info["actual"], 4.0 - (1.0 - 3.0 + 9.0 / 4.0), rtol=1e-13)
    assert res.converged
    assert_allclose(np.abs(res.x), [0.0, math.sqrt(2.0)], atol=1e-8)


@pytest.mark.parametrize(
    "pid", ["rosenbrock", "himmelblau", "beale", "six_hump_camel", "three_hump_camel", "booth"]
)
def test_arc_converges_on_library_problems(pid: str) -> None:
    p = problems.get(pid)
    res = arc(p)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    assert float(np.max(np.abs(p.grad(res.x)))) <= 1e-8
    assert min(float(np.linalg.norm(res.x - np.asarray(m))) for m in p.minima) < 1e-6
    assert res.extra["lambda_min"] > 0.0


@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "six_hump_camel"])
def test_arc_agrees_with_scipy_trust_exact(pid: str) -> None:
    p = problems.get(pid)
    ours = arc(p)
    ref = sp_minimize(
        p.f,
        np.asarray(p.x0, float),
        jac=p.grad,
        hess=p.hess,
        method="trust-exact",
        options={"gtol": 1e-8},
    )
    assert ours.converged and ref.success
    # ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)) ≤ 1e-8/0.3 for both runs.
    assert_allclose(ours.x, ref.x, rtol=0, atol=1e-7)


_ARC_FAR_STARTS = ((9.0, -7.0), (-4.5, 10.5), (0.3, -0.2))
_ARC_LIBRARY = ("rosenbrock", "himmelblau", "beale", "six_hump_camel", "three_hump_camel", "booth")
#: (problem id, x0, parameters) for the ARC invariant checks. The library problems from their
#: default x0 give the "successful" class of CGT eq. 2.6 (the convex test problems never do).
_ARC_RUNS: tuple[tuple[str, Any, dict[str, Any]], ...] = (
    *((pid, x0, {}) for pid in ("lse", "sqrt1p", "quartic") for x0 in _ARC_FAR_STARTS),
    *((pid, None, {}) for pid in _ARC_LIBRARY),
    ("rosenbrock", None, {"sigma0": 100.0, "eta1": 0.3, "eta2": 0.6, "gamma": 5.0}),
    ("beale", None, {"sigma0": 0.01, "eta1": 0.05, "eta2": 0.95, "gamma": 3.0}),
)
_ARC_PARAMS = {"sigma0": 1.0, "eta1": 0.1, "eta2": 0.9, "gamma": 2.0}  # CGT §7 (the defaults)


def _arc_problem(pid: str) -> Problem:
    local = {"lse": _lse, "sqrt1p": _sqrt1p, "quartic": _quartic_saddle}
    return local[pid]() if pid in local else problems.get(pid)


def _check_arc_run(pid: str, x0: Any, kw: dict[str, Any]) -> collections.Counter[str]:
    """Run ARC and check every iteration against CGT Alg. 2.1; return the class census."""
    p = _arc_problem(pid)
    par = _ARC_PARAMS | kw
    eta1, eta2, gamma = par["eta1"], par["eta2"], par["gamma"]
    res = arc(p, x0=x0, max_iter=500, **kw)
    assert_valid_result(res, max_iter=500)
    assert res.converged, res.message
    census: collections.Counter[str] = collections.Counter()
    sig = res.trace[0].info["sigma"]
    assert sig == par["sigma0"]
    for prev, step in itertools.pairwise(res.trace):
        info = step.info
        assert info["sigma"] == sig
        rho = info["rho"]
        # The class of CGT eq. 2.6 follows from ρ (a non-finite f(x + s) gives ρ = None).
        if rho is not None and rho > eta2:
            expected = "very_successful"
        elif rho is not None and rho >= eta1:
            expected = "successful"
        else:
            expected = "unsuccessful"
        assert info["iteration"] == expected, (rho, info["iteration"])
        census[expected] += 1
        assert info["accepted"] == (expected != "unsuccessful")
        # The σ update of CGT eq. 2.6 with the choices of §7 (ε_M = machine epsilon), exactly.
        sigma = info["sigma"]
        if expected == "very_successful":
            gnorm = float(np.linalg.norm(np.asarray(info["grad"])))
            assert info["new_sigma"] == max(min(sigma, gnorm), EPS), (sigma, gnorm)
        elif expected == "successful":
            assert info["new_sigma"] == sigma
        else:
            assert info["new_sigma"] == gamma * sigma
            assert np.array_equal(step.x, prev.x)
        # ρ itself: (actual + δ)/(predicted + δ) from the trace values (eq. 2.4 + allowance δ).
        f_center = prev.fun
        f_trial = p.f(np.asarray(info["trial_point"]))
        assert f_center is not None and math.isfinite(f_trial)
        delta = ROUNDING * EPS * max(abs(f_center), abs(f_trial))
        assert rho is not None
        assert_allclose(rho, (f_center - f_trial + delta) / (info["predicted"] + delta), rtol=1e-14)
        # The step obeys CGT Thm. 3.1 at the center: (B + λI)s = −g, λ = σ‖s‖.
        B = np.asarray(info["H"])
        s = np.asarray(info["step"])
        g = np.asarray(info["grad"])
        lam = info["lambda"]
        resid = (B + lam * np.eye(2)) @ s + g
        scale = (float(np.max(np.abs(np.linalg.eigvalsh(B)))) + lam) * float(
            np.linalg.norm(s)
        ) + float(np.linalg.norm(g))
        assert float(np.linalg.norm(resid)) <= 1e3 * EPS * scale
        assert_allclose(lam, info["sigma"] * float(np.linalg.norm(s)), rtol=1e-10)
        # Accepted steps decrease f (up to the rounding allowance δ of the module docstring).
        assert step.fun is not None and prev.fun is not None
        assert step.fun <= prev.fun + ROUNDING * EPS * max(abs(prev.fun), abs(step.fun))
        sig = info["new_sigma"]
    # Exact counts: f once per iteration, ∇f and ∇²f once per accepted point (+ x₀).
    n_acc = sum(bool(s.info["accepted"]) for s in res.trace[1:])
    assert res.n_hev == res.n_gev == 1 + n_acc
    assert res.n_fev == 1 + res.n_iter
    assert res.extra["n_rejected"] == res.n_iter - n_acc
    return census


@pytest.mark.parametrize(
    "run", _ARC_RUNS, ids=lambda r: f"{r[0]}-{r[1]}-{'-'.join(map(str, r[2].values()))}"
)
def test_arc_invariants(run: tuple[str, Any, dict[str, Any]]) -> None:
    _check_arc_run(*run)


def test_arc_invariant_runs_reach_every_class_of_eq_2_6() -> None:
    # Without a class in the runs above, its σ rule is not checked: require all three, and a very
    # successful step where ‖g‖ < σ (min(σ, ‖g‖) picks ‖g‖) and one where σ < ‖g‖.
    census: collections.Counter[str] = collections.Counter()
    for run in _ARC_RUNS:
        census += _check_arc_run(*run)
    assert census["very_successful"] >= 50, census
    assert census["successful"] >= 5, census
    assert census["unsuccessful"] >= 5, census
    picks = collections.Counter()
    for pid in _ARC_LIBRARY:
        for step in arc(problems.get(pid)).trace[1:]:
            if step.info["iteration"] == "very_successful":
                gnorm = float(np.linalg.norm(np.asarray(step.info["grad"])))
                picks["grad" if gnorm < step.info["sigma"] else "sigma"] += 1
    assert picks["grad"] >= 5 and picks["sigma"] >= 5, picks


@pytest.mark.parametrize(
    "kw",
    [{}, {"sigma0": 100.0}, {"gamma": 5.0, "eta1": 0.3}],
)
def test_arc_counts_are_exact(kw: dict[str, Any]) -> None:
    p, calls = _counting(problems.get("rosenbrock"))
    res = arc(p, **kw)
    assert res.converged
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], calls["h"])


# --------------------------------------------------------------------------------------
# The study's key property: ARC escapes saddles; the second-order stop is honest
# --------------------------------------------------------------------------------------


def test_arc_escapes_the_saddle_where_regularized_newton_stalls() -> None:
    # From (3, 0), ∇f ⟂ e₂: every regularized Newton iterate stays on y = 0, the stable manifold of
    # the saddle 0. ARC's cubic model sees the negative curvature (hard case) and leaves the line.
    p = _quartic_saddle()
    res = arc(p, x0=(3.0, 0.0))
    assert res.converged, res.message
    assert_allclose(np.abs(res.x), [0.0, math.sqrt(2.0)], atol=1e-8)
    assert any(s.info["hard_case"] for s in res.trace[1:])
    for variant in VARIANTS:
        rr = reg_newton(p, x0=(3.0, 0.0), variant=variant, H=0.5)
        assert_valid_result(rr)
        assert float(np.max(np.abs(rr.x))) <= 1e-8
        assert not rr.converged and "a saddle point" in rr.message, rr.message
        assert "not a minimizer" in rr.message
        assert rr.extra["lambda_min"] == pytest.approx(-2.0)


def test_arc_stops_at_stationary_saddle_start() -> None:
    res = arc(_quartic_saddle(), x0=(0.0, 0.0))
    assert not res.converged and res.n_iter == 0
    assert "not a minimizer" in res.message and "a saddle point" in res.message
    assert res.extra["lambda_min"] == -2.0
    assert_valid_result(res)


def test_second_order_stop_maximizer_and_semidefinite() -> None:
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
        res = fn(mx)
        assert not res.converged and "a maximizer" in res.message, res.message
        res = fn(psd)
        assert res.converged and "semidefinite" in res.message, res.message
        # Central-difference ∇f and ∇²f at a minimizer of (x + y)² (∇²f singular): the
        # finite-difference eigenvalue 0 ± O(ε^{1/3}) must not be read as a saddle.
        res = fn(lambda x: (x[0] + x[1]) ** 2, x0=[1.0, -1.0], gtol=1e-6)
        assert res.converged, res.message


# --------------------------------------------------------------------------------------
# ARC failure paths and input conventions
# --------------------------------------------------------------------------------------


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
    assert not res.converged and "not finite" in res.message and res.n_iter == 0
    assert_valid_result(res)
    # f = +inf outside the unit disk: trial steps that leave it are rejected (rho None), σ grows.
    disk = Problem(
        id="d",
        name="d",
        latex="",
        f=lambda x: float(x @ x) - 4.0 * float(x[0]) if float(x @ x) < 1.0 else math.inf,
        grad=lambda x: 2.0 * np.asarray(x) - np.array([4.0, 0.0]),
        hess=lambda x: 2.0 * np.eye(2),
        dim=2,
        domain=(),
        x0=(0.0, 0.0),
    )
    res = arc(disk, max_iter=50)
    assert_valid_result(res, max_iter=50)
    assert not res.converged
    rejected = [s for s in res.trace[1:] if s.info["rho"] is None]
    assert rejected and all(s.info["iteration"] == "unsuccessful" for s in rejected)
    assert all(float(np.linalg.norm(s.x)) < 1.0 for s in res.trace)
    for kw in (
        {"sigma0": 0.0},
        {"eta1": 0.95, "eta2": 0.9},
        {"gamma": 1.0},
        {"max_iter": 0},
        {"gtol": -1.0},
    ):
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


def test_one_dimensional_problem_uses_scalar_convention() -> None:
    # f(x) = x⁴/4 − x: x* = 1; the callables receive a float (numopt.core.types scalar convention).
    p = Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: x**4 / 4.0 - x,
        grad=lambda x: x**3 - 1.0,
        hess=lambda x: 3.0 * x**2,
        dim=1,
        domain=(-3.0, 3.0),
        x0=-2.0,
    )
    for fn in (arc, reg_newton):
        res = fn(p)
        assert res.converged, res.message
        assert_allclose(res.x, [1.0], atol=1e-9)


# --------------------------------------------------------------------------------------
# Gradient-regularized Newton
# --------------------------------------------------------------------------------------


def test_reg_newton_fixed_first_step_hand_computed() -> None:
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    x0 = np.array([1.0, -2.0])
    H = 0.5
    res = reg_newton(_quadratic(A, x0), variant="fixed", H=H)
    g = A @ x0  # (1, −3)
    lam = math.sqrt(H * math.sqrt(10.0))
    x1 = x0 - scipy.linalg.solve(A + lam * np.eye(2), g, assume_a="pos")
    assert_allclose(res.trace[1].x, x1, rtol=1e-14)
    assert_allclose(res.trace[1].info["lambda"], lam, rtol=1e-15)
    assert res.converged
    assert_allclose(res.x, [0.0, 0.0], atol=1e-8)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "six_hump_camel", "himmelblau"])
def test_reg_newton_every_step_is_the_regularized_newton_step(variant: str, pid: str) -> None:
    """x_k = x_{k−1} − (∇²f + λI)⁻¹∇f by a naive LU solve, and λ follows the variant's rule."""
    p = problems.get(pid)
    H, alpha = 1.0, 0.8
    res = reg_newton(p, variant=variant, H=H, alpha=alpha)
    assert_valid_result(res, max_iter=200)
    assert p.grad is not None and p.hess is not None
    for prev, step in itertools.pairwise(res.trace):
        xp = np.asarray(prev.x)
        g = np.asarray(p.grad(xp))
        B = np.asarray(p.hess(xp))
        lam, H_reg = step.info["lambda"], step.info["H_reg"]
        gnorm = float(np.linalg.norm(g))
        if variant == "super_universal":
            assert_allclose(lam, H_reg * gnorm**alpha, rtol=1e-14)
        else:
            assert_allclose(lam, math.sqrt(H_reg * gnorm), rtol=1e-14)
        if variant == "fixed":
            assert H_reg == H
        M = B + lam * np.eye(2)
        s_ref = -scipy.linalg.solve(M, g)
        # NOTE: rtol = κ(M)·1e-13: eigh-based and LU solves agree to O(κ ε).
        kappa = float(np.linalg.cond(M))
        assert_allclose(step.x, xp + s_ref, rtol=kappa * 1e-13, atol=kappa * 1e-13 * gnorm)
        assert step.info["inner_iters"] == len(step.info["trials"])
        assert step.info["trials"][-1][4] is True
        assert all(t[4] is False for t in step.info["trials"][:-1])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("make", [_lse, _sqrt1p], ids=["lse", "sqrt1p"])
def test_reg_newton_converges_on_convex_problems_from_far(variant: str, make: Any) -> None:
    p = make()
    # fixed: Assumption 1 needs ∇²f to be 2H-Lipschitz; H = 1 exceeds L₂/2 for both problems
    # (the study's grid estimates L̂₂ ≈ 1.22 for lse, 0.85 for sqrt1p).
    for x0 in [(9.0, -7.0), (-4.5, 10.5)]:
        res = reg_newton(p, x0=x0, variant=variant, max_iter=500)
        assert_valid_result(res, max_iter=500)
        assert res.converged, res.message
        assert_allclose(res.x, p.minima[0], atol=1e-6)
        assert all(s.info["pd"] and s.info["descent"] for s in res.trace[1:])


def test_reg_newton_fixed_beats_pure_newton_divergence() -> None:
    # Pure Newton maps r → −r³ on √(1 + ‖x‖²) (diverges from ‖x0‖ = 5).
    p = _sqrt1p()
    assert not numopt.run("pure_newton", p, x0=(3.0, 4.0)).converged
    res = reg_newton(p, x0=(3.0, 4.0), variant="fixed", H=1.0)
    assert res.converged and float(np.linalg.norm(res.x)) < 1e-8


@pytest.mark.parametrize("variant", VARIANTS)
def test_reg_newton_counts_are_exact(variant: str) -> None:
    p, calls = _counting(problems.get("rosenbrock"))
    res = reg_newton(p, variant=variant)
    assert res.converged
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], calls["h"])
    n_trials = sum(len(s.info["trials"]) for s in res.trace[1:])
    assert res.extra["n_trials"] == n_trials
    assert res.n_hev == 1 + res.n_iter
    if variant == "fixed":
        assert res.n_fev == res.n_gev == 1 + res.n_iter
    elif variant == "adan":
        assert res.n_fev == res.n_gev == 1 + n_trials
    else:
        assert res.n_gev == 1 + n_trials and res.n_fev == 1 + res.n_iter


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

    On a quadratic the acceptance tests hold for every H and λ (∇f(x₊) = −λs), so no trial is ever
    rejected here; ``test_reg_newton_adaptive_trials_replay_the_acceptance_tests`` checks the
    decisions on non-quadratic problems, rejected trials included.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M @ M.T + 0.1 * np.eye(n)
    x0 = rng.normal(size=n) * 10.0**log_scale
    p = _quadratic(A, x0 if n > 1 else float(x0[0]))
    res = reg_newton(p, variant=variant, H=10.0**log_h, max_iter=300)
    assert_valid_result(res, max_iter=300)
    if variant == "fixed":
        # A large H on a quadratic (true H = 0) makes λ ≫ ∇²f: gradient-like steps, still in the
        # O(1/k²) phase of Thm. 1 at max_iter (e.g. H = 100, ‖x0‖ = 10³). Only max_iter may stop it.
        assert res.converged or res.message.startswith("reached max_iter"), res.message
    else:
        assert res.converged, res.message
    for prev, step in itertools.pairwise(res.trace):
        assert step.fun is not None and prev.fun is not None
        delta = ROUNDING * EPS * max(abs(prev.fun), abs(step.fun))
        assert step.fun <= prev.fun + delta
        lam = step.info["lambda"]
        r = float(np.linalg.norm(step.info["direction"]))
        if variant == "adan":
            assert step.grad_norm <= 2.0 * lam * r * (1 + 1e-12)
            assert step.fun <= prev.fun - (2.0 / 3.0) * lam * r * r + delta
        if variant == "super_universal":
            g1 = A @ np.atleast_1d(np.asarray(step.x))
            lhs = float(g1 @ (np.atleast_1d(np.asarray(prev.x)) - np.atleast_1d(step.x)))
            assert lhs >= float(g1 @ g1) / (4.0 * lam) * (1 - 1e-12)
        assert step.info["pd"] is True and step.info["descent"] is True


def _kleene_and(a: bool | None, b: bool | None) -> bool | None:
    """a ∧ b where None is "not decided beyond rounding" (False ∧ None = False)."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _decide(margin: float, band: float) -> bool | None:
    """margin ≥ 0, or None when |margin| ≤ band (the test is at its rounding level)."""
    return None if abs(margin) <= band else margin > 0.0


def _replay_reg_newton_trials(
    p: Problem, res: Any, variant: str, alpha: float
) -> collections.Counter[str]:
    """Recompute every trial of the adaptive search, rejected ones included, from p.f, p.grad,
    p.hess and an LU solve s = −(∇²f(x) + λI)⁻¹∇f(x) (independent of the code's eigh solve), and
    check the recorded decision trial[4] against the acceptance test of the variant:

        adan:             ‖∇f(x₊)‖ ≤ 2λr  and  f(x₊) ≤ f(x) − (2/3)λr² + δ,  r = ‖s‖
                          (Mishchenko (2023), Alg. 2; δ = 10³ε·max(|f(x)|, |f(x₊)|), module docs)
        super_universal:  ⟨∇f(x₊), x − x₊⟩ ≥ ‖∇f(x₊)‖²/(4λ)   (DMN (2024), Alg. 2)

    The recorded f(x₊) and ‖∇f(x₊)‖ must equal p.f and p.grad at the recomputed x₊ (to the solve
    difference and the noise of ``_noisy_quartic``); the thresholds, the constants 2, 2/3, 4 and δ
    are then applied to them here. A test within its rounding band of the threshold counts as
    undecided. Returns a census of the trials by which condition decided them.
    """
    assert p.grad is not None and p.hess is not None
    census: collections.Counter[str] = collections.Counter()
    for prev, step in itertools.pairwise(res.trace):
        x = np.asarray(prev.x, dtype=float)  # (2,)
        fx = float(p.f(x))
        assert fx == prev.fun
        g = np.asarray(p.grad(x), dtype=float)  # (2,)
        B = np.asarray(p.hess(x), dtype=float)  # (2, 2)
        gnorm = float(np.linalg.norm(g))
        trials = step.info["trials"]
        assert trials[-1][4] is True and all(t[4] is False for t in trials[:-1])
        for H_used, lam, f_rec, gn_rec, ok in trials:
            if variant == "adan":
                assert_allclose(lam, math.sqrt(H_used * gnorm), rtol=1e-15)
            else:
                assert_allclose(lam, H_used * gnorm**alpha, rtol=1e-15)
            M = B + lam * np.eye(2)  # (2, 2)
            if gn_rec is None:  # a singular trial: rejected without evaluating anything
                assert ok is False and f_rec is None
                assert float(np.min(np.abs(np.linalg.eigvalsh(M)))) <= 1e-12 * max(
                    float(np.max(np.abs(np.linalg.eigvalsh(B)))), lam
                )
                census["singular"] += 1
                continue
            kappa = float(np.linalg.cond(M))
            s = -scipy.linalg.solve(M, g, assume_a="sym")  # (2,)
            xp = x + s
            r = float(np.linalg.norm(s))
            fp = float(p.f(xp))
            gp = np.asarray(p.grad(xp), dtype=float)  # (2,)
            gpn = float(np.linalg.norm(gp))
            # NOTE: rel = 1e-12·κ(M): the LU and eigh solves differ by O(κε)‖s‖ (Higham §7), which
            # moves ‖∇f(x₊)‖ and r by O(κε) relative; 1e-12 ≈ 4500ε covers the constant.
            rel = 1e-12 * kappa
            assert_allclose(gn_rec, gpn, rtol=rel, atol=rel * float(np.linalg.norm(B)) * r)
            if variant == "adan":
                # The recorded f(x₊) is f at the code's x₊, which differs from the LU x₊ by
                # O(κε)‖s‖: check it against p.f, then decide with it. NOTE: tolerance 64ε|f| for
                # the rounding of f, 2·100ε|f| for the evaluation error of _noisy_quartic (its
                # hash differs between last-bit-different points), and ‖∇f‖·O(κε)r for Δx₊.
                assert f_rec is not None
                f_scale = max(abs(fx), abs(fp))
                f_tol = (64.0 * EPS + 2.0 * _NOISE_REL) * f_scale + rel * (gpn + gnorm) * r
                assert abs(f_rec - fp) <= f_tol, (f_rec, fp, f_tol)
                delta = ROUNDING * EPS * max(abs(fx), abs(f_rec))
                # NOTE: decision bands: 8ε|f| for the rounding of fx − (2/3)λr² + δ − f(x₊), and
                # the O(κε) relative error of r between the two solves (r² twice that).
                f_band = 8.0 * EPS * f_scale + 2.0 * rel * lam * r * r
                f_margin = fx - (2.0 / 3.0) * lam * r * r - f_rec
                cond_grad = _decide(2.0 * lam * r - gn_rec, rel * (2.0 * lam * r + gn_rec))
                cond_f = _decide(f_margin + delta, f_band)
                expected = _kleene_and(cond_grad, cond_f)
                if expected is None:
                    census["undecided"] += 1
                    continue
                assert ok is expected, (H_used, lam, cond_grad, cond_f)
                if ok:
                    census["accepted"] += 1
                    if _decide(f_margin, f_band) is False:
                        census["accepted_by_delta"] += 1  # f(x₊) > f(x) − (2/3)λr², within δ
                elif cond_grad is False and cond_f is True:
                    census["rejected_by_grad_only"] += 1
                elif cond_grad is True and cond_f is False:
                    census["rejected_by_f_only"] += 1
                    if f_margin + (1.0 / 3.0) * lam * r * r + delta > f_band:
                        census["rejected_f_between_1/3_and_2/3"] += 1
                else:
                    census["rejected_by_both"] += 1
            else:
                assert f_rec is None  # super_universal evaluates f at accepted points only
                inner = float(gp @ (x - xp))
                rhs = gpn * gpn / (4.0 * lam)
                expected = _decide(inner - rhs, rel * (abs(inner) + rhs + gpn * r))
                if expected is None:
                    census["undecided"] += 1
                    continue
                assert ok is expected, (H_used, lam, inner, rhs)
                census["accepted" if ok else "rejected"] += 1
                if ok and inner < 2.0 * rhs * (1.0 - rel):
                    census["accepted_inner_below_2x"] += 1  # rejected if 4λ were 2λ
    return census


#: (problem, x0) for the replay: convex problems from far starts (the search rejects on them) and
#: nonconvex library problems from their default x0.
_REPLAY_RUNS: tuple[tuple[str, Any], ...] = (
    *((pid, x0) for pid in ("lse", "sqrt1p") for x0 in [(9.0, -7.0), (-4.5, 10.5), (30.0, 40.0)]),
    ("noisy_quartic", (1.0, -0.5)),
    ("noisy_quartic", (-3.0, 2.0)),
    *((pid, None) for pid in ("rosenbrock", "beale", "six_hump_camel", "himmelblau")),
)


def _replay_problem(pid: str) -> Problem:
    local = {"lse": _lse, "sqrt1p": _sqrt1p, "noisy_quartic": _noisy_quartic}
    return local[pid]() if pid in local else problems.get(pid)


@pytest.mark.parametrize("H", [1.0, 1e-3])
def test_reg_newton_adaptive_trials_replay_the_acceptance_tests(H: float) -> None:
    census: collections.Counter[str] = collections.Counter()
    su: collections.Counter[str] = collections.Counter()
    for pid, x0 in _REPLAY_RUNS:
        p = _replay_problem(pid)
        res = reg_newton(p, x0=x0, variant="adan", H=H, max_iter=500)
        assert_valid_result(res, max_iter=500)
        assert res.converged, (pid, res.message)
        census += _replay_reg_newton_trials(p, res, "adan", 1.0)
        for alpha in (1.0, 2.0 / 3.0):
            res = reg_newton(p, x0=x0, variant="super_universal", H=H, alpha=alpha, max_iter=500)
            assert_valid_result(res, max_iter=500)
            if res.converged:
                su += _replay_reg_newton_trials(p, res, "super_universal", alpha)
            else:  # the theory is for convex f: only a nonconvex library problem may fail
                assert pid in ("six_hump_camel", "himmelblau"), (pid, alpha, res.message)
    # Every kind of decision occurs, so dropping or weakening a condition changes some trial:
    # the gradient test alone rejects, the f-test alone rejects (some with f(x₊) between the 1/3
    # and 2/3 thresholds), and δ alone accepts (the noisy f near x*).
    assert census["undecided"] == 0, census
    assert census["rejected_by_grad_only"] >= 5, census
    assert census["rejected_by_f_only"] >= 5, census
    assert census["rejected_f_between_1/3_and_2/3"] >= 1, census
    assert census["accepted_by_delta"] >= 1, census
    assert su["undecided"] == 0, su
    assert su["rejected"] >= 20, su
    assert su["accepted_inner_below_2x"] >= 1, su


@pytest.mark.parametrize("x0", [(1.0, -0.5), (-3.0, 2.0)])
def test_reg_newton_adan_rounding_allowance_accepts_noisy_steps(
    x0: tuple[float, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The purpose of δ (module docstring): near x* the 100ε evaluation error of f exceeds the
    # decrease (2/3)λr² that AdaN asks for. With δ the run converges (‖∇f‖∞ = ‖x‖³ ≤ 1e-8 ⇒
    # ‖x‖ ≤ 2.2e-3) and some step is accepted only through δ; without δ correct steps are
    # rejected, H grows, and the run does not converge.
    p = _noisy_quartic()
    res = reg_newton(p, x0=x0, variant="adan", max_iter=500)
    assert res.converged, res.message
    assert float(np.linalg.norm(res.x)) < 2.5e-3
    assert _replay_reg_newton_trials(p, res, "adan", 1.0)["accepted_by_delta"] >= 1
    monkeypatch.setattr(rn, "_ROUNDING", 0.0)
    res = reg_newton(p, x0=x0, variant="adan", max_iter=500)
    assert not res.converged, res.message


def test_reg_newton_adan_doubling_schedule() -> None:
    # Alg. 2: the first trial at k = 0 uses 2H₀; every later iteration starts from H_{k−1}/2.
    res = reg_newton(_lse(), x0=(9.0, -7.0), variant="adan", H=1.0)
    assert res.converged
    trials = res.trace[1].info["trials"]
    assert trials[0][0] == 2.0
    for j in range(1, len(trials)):
        assert trials[j][0] == 2.0 * trials[j - 1][0]
    for prev, step in itertools.pairwise(res.trace[1:]):
        assert step.info["trials"][0][0] == prev.info["H_reg"] / 2.0


def test_reg_newton_super_universal_schedule() -> None:
    res = reg_newton(_lse(), x0=(9.0, -7.0), variant="super_universal", H=1.0)
    assert res.converged
    for prev, step in itertools.pairwise(res.trace[1:]):
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
    assert not res.converged and "singular" in res.message and res.n_iter == 0
    assert_valid_result(res)
    # super_universal tries λ = H‖g‖ = 1 first: singular, so the trial is rejected (f and ‖∇f‖ are
    # None) and H grows. f is unbounded below along x, so both adaptive variants follow it out to
    # max_iter and report converged=False.
    for variant in ("adan", "super_universal"):
        res = reg_newton(p, variant=variant, H=1.0, max_iter=30)
        assert_valid_result(res, max_iter=30)
        assert not res.converged and "max_iter" in res.message
        assert res.fun is not None and res.fun < -1e15
    trials = reg_newton(p, variant="super_universal", H=1.0, max_iter=1).trace[1].info["trials"]
    assert trials[0] == [1.0, 1.0, None, None, False]
    assert trials[1][4] is True
    res = reg_newton(problems.get("rosenbrock"), max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=3)
    # Non-finite ∇f at x₀.
    nan_grad = Problem(
        id="n",
        name="n",
        latex="",
        f=lambda x: float(x @ x),
        grad=lambda x: np.array([math.nan, 0.0]),
        hess=lambda x: np.eye(2),
        dim=2,
        domain=(),
        x0=(1.0, 1.0),
    )
    res = reg_newton(nan_grad)
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)
    for kw in (
        {"variant": "nope"},
        {"alpha": 0.5},
        {"alpha": 1.5},
        {"H": -1.0},
        {"H": math.inf},
        {"max_iter": 0},
        {"max_iter": 2.5},
    ):
        with pytest.raises(ValueError):
            reg_newton(p, **kw)
    # The ParamSpec minimum α = 2/3 is accepted.
    assert reg_newton(_lse(), variant="super_universal", alpha=2.0 / 3.0).converged
