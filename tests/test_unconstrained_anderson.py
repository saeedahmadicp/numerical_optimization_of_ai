"""Tests for numopt.unconstrained.anderson: Anderson-accelerated gradient descent AA(m).

Oracles: gradient descent with a fixed step (AA(0) is that method, bit for bit); a hand
computation of the first two steps; GMRES built independently in the test from an Arnoldi basis
(Walker & Ni (2011), Thm. 2.2); the Fang–Saad multisecant identities; a KKT solve
(``scipy.linalg.solve``) and a null-space parametrization (LAPACK gelsy) for the coefficient
problem; SciPy BFGS for the minimizers; and ``scipy.linalg.eigvalsh`` of the analytic Hessian
for the classification of every stopping point. The key property of
``research/anderson-acceleration`` (E4) is reproduced: plain AA(5)-GD stops at a saddle point
or a maximizer from 31 of 64 himmelblau starts, λ = 10⁻² from 17, gradient descent from 0.
"""

from __future__ import annotations

import inspect
import math
from collections import Counter
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import minimize as sp_minimize

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import Problem, Result
from numopt.unconstrained import anderson as aa_mod
from numopt.unconstrained.anderson import _aa_coefficients, anderson_gd

EPS = float(np.finfo(float).eps)

INFO_KEYS = {
    "grad",
    "alpha",
    "memory",
    "coefficients",
    "history",
    "x_bar",
    "lsq_residual",
    "cond",
    "lam_eff",
    "direction",
    "hess_eigs",
}


def _quadratic(A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> Problem:
    """f(x) = ½xᵀAx − bᵀx with exact derivatives."""
    return Problem(
        id="quad",
        name="quad",
        latex="q",
        f=lambda x: float(0.5 * x @ A @ x - b @ x),
        grad=lambda x: A @ x - b,
        hess=lambda x: A,
        dim=b.size,
        domain=(),
        x0=x0,
    )


def _poly2(f: Any, grad: Any, hess: Any, x0: Any) -> Problem:
    return Problem(id="p", name="p", latex="p", f=f, grad=grad, hess=hess, dim=2, domain=(), x0=x0)


def _kind(res: Result) -> str:
    """The outcome class of the study (E4) from the Result."""
    if res.converged:
        return "minimizer"
    if "not a minimizer" in res.message:
        return "saddle/max"
    if "diverged" in res.message or "not finite" in res.message:
        return "diverged"
    return "no convergence"


# --------------------------------------------------------------------------------------
# Registration and contract
# --------------------------------------------------------------------------------------


def test_registration_matches_the_signature() -> None:
    spec = get_method("anderson_gd")
    assert spec.family == "unconstrained"
    assert spec.needs == ("f", "grad")
    assert all(ref for ref in spec.references)
    sig = inspect.signature(anderson_gd)
    keywords = {n for n, p in sig.parameters.items() if p.kind is p.KEYWORD_ONLY} - {"x0"}
    assert {p.name for p in spec.params} == keywords
    for p in spec.params:
        assert sig.parameters[p.name].default == p.default
        assert p.min is not None and p.max is not None and p.min <= p.default <= p.max


@pytest.mark.parametrize(
    ("pid", "kw"),
    [
        ("rosenbrock", {}),
        ("quadratic_nd", {"m": 3, "lr": 0.01}),
        ("himmelblau", {"x0": [1.0, 1.0], "lr": 0.01, "lam": 0.01, "beta": 0.7}),
        ("beale", {"m": 0}),
    ],
)
def test_info_keys_and_trace_contract(pid: str, kw: dict[str, Any]) -> None:
    prob = problems.get(pid)
    m = kw.get("m", 5)
    res = numopt.run("anderson_gd", prob, **kw)
    assert_valid_result(res, max_iter=500)
    assert res.n_iter == res.trace[-1].k == len(res.trace) - 1
    s0 = res.trace[0]
    assert set(s0.info) == INFO_KEYS
    assert s0.step_size is None and s0.info["memory"] is None and s0.info["coefficients"] == []
    for prev, s in zip(res.trace, res.trace[1:], strict=False):
        assert set(s.info) == INFO_KEYS
        x_prev, x = np.asarray(prev.x), np.asarray(s.x)
        c = np.asarray(s.info["coefficients"])
        hist = np.stack(s.info["history"], axis=1)
        assert s.info["memory"] == min(m, s.k - 1) == c.size - 1 == hist.shape[1] - 1
        assert math.isclose(c.sum(), 1.0, rel_tol=0, abs_tol=1e-12)
        assert_allclose(hist[:, -1], x_prev, rtol=0, atol=0)
        assert_allclose(s.info["x_bar"], hist @ c, rtol=1e-13, atol=1e-13)
        assert_allclose(s.info["direction"], x - x_prev, rtol=0, atol=0)
        assert s.step_size == pytest.approx(float(np.linalg.norm(x - x_prev)), rel=1e-15)
        assert_allclose(s.info["grad"], prob.grad(x), rtol=0, atol=0)
        assert s.grad_norm == pytest.approx(float(np.linalg.norm(prob.grad(x))), rel=1e-15)
        assert s.fun == pytest.approx(float(prob.f(x)), rel=1e-15)
        assert s.info["alpha"] == kw.get("lr", 1e-3)
        assert s.info["cond"] >= 1.0
    # hess_eigs only at a step that passed the gradient test (the last one, here).
    assert all(s.info["hess_eigs"] is None for s in res.trace[:-1])
    if res.trace[-1].grad_norm is not None and res.trace[-1].grad_norm <= 1e-6:
        assert_allclose(
            res.trace[-1].info["hess_eigs"],
            scipy.linalg.eigvalsh(prob.hess(np.asarray(res.x))),
            rtol=1e-12,
            atol=1e-12,
        )


# --------------------------------------------------------------------------------------
# Convergence and oracles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pid", "kw"),
    [
        ("rosenbrock", {}),
        ("beale", {}),
        ("beale", {"beta": 0.5}),
        ("quadratic_bowl", {}),
        ("booth", {}),
        ("six_hump_camel", {"lr": 0.01}),
        ("quadratic_nd", {"lr": 0.01, "m": 10}),
    ],
)
def test_converges_to_the_scipy_bfgs_minimizer(pid: str, kw: dict[str, Any]) -> None:
    prob = problems.get(pid)
    res = numopt.run("anderson_gd", prob, gtol=1e-8, **kw)
    assert_valid_result(res, max_iter=500)
    assert res.converged, res.message
    assert "strict local minimizer" in res.message
    assert np.linalg.norm(prob.grad(res.x)) <= 1e-8
    ref = sp_minimize(prob.f, prob.x0, jac=prob.grad, method="BFGS", options={"gtol": 1e-10})
    # Both stop near the same strict minimizer: ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)), λ_min ≥ 0.3.
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-6)


def test_m0_is_gradient_descent_with_a_fixed_step_bit_for_bit() -> None:
    prob = problems.get("six_hump_camel")
    ref = numopt.run("gradient_descent", prob, step_rule="fixed", lr=0.01, gtol=1e-150, max_iter=40)
    res = anderson_gd(prob, m=0, lr=0.01, gtol=1e-150, max_iter=40)
    xs_ref = np.array([s.x for s in ref.trace])
    xs = np.array([s.x for s in res.trace])
    assert xs.shape == xs_ref.shape == (41, 2)
    assert_allclose(xs, xs_ref, rtol=0, atol=0)


def test_hand_computed_first_two_steps() -> None:
    """x₁ = x₀ + βf₀; γ = f₁ᵀΔf/‖Δf‖², x₂ = x₁ + βf₁ − (Δx + βΔf)γ, f = −α∇f."""
    prob = problems.get("rosenbrock")
    alpha, beta = 1e-3, 0.7
    x0 = np.array([-1.2, 1.0])
    f0 = -alpha * prob.grad(x0)
    x1 = x0 + beta * f0
    f1 = -alpha * prob.grad(x1)
    df, dx = f1 - f0, x1 - x0
    gamma = float(f1 @ df / (df @ df))
    x2 = x1 + beta * f1 - (dx + beta * df) * gamma
    res = anderson_gd(prob, x0=x0, m=1, lr=alpha, beta=beta, gtol=0.0, max_iter=2)
    assert_allclose(res.trace[1].x, x1, rtol=1e-15)
    assert_allclose(res.trace[2].x, x2, rtol=1e-13)
    assert_allclose(res.trace[2].info["coefficients"], [gamma, 1.0 - gamma], rtol=1e-12)
    assert res.trace[1].info["coefficients"].tolist() == [1.0]  # the gradient step


def _gmres_iterates(B: np.ndarray, rhs: np.ndarray, x0: np.ndarray, kmax: int) -> list[Any]:
    """GMRES written from its definition: x_k = argmin_{x ∈ x₀ + K_k(B, r₀)} ‖rhs − Bx‖₂.

    Arnoldi with modified Gram–Schmidt (Saad (2003), Alg. 6.2), then the small least-squares
    problem by ``scipy.linalg.lstsq``. Returns [(x_k, ‖r_k‖)] for k = 0, …, kmax.
    """
    r0 = rhs - B @ x0
    beta0 = float(np.linalg.norm(r0))
    n = x0.size
    V = np.zeros((n, kmax + 1))
    H = np.zeros((kmax + 1, kmax))
    V[:, 0] = r0 / beta0
    out: list[Any] = [(x0.copy(), beta0)]
    for k in range(kmax):
        w = B @ V[:, k]
        for i in range(k + 1):
            H[i, k] = V[:, i] @ w
            w = w - H[i, k] * V[:, i]
        H[k + 1, k] = float(np.linalg.norm(w))
        if H[k + 1, k] > 0:
            V[:, k + 1] = w / H[k + 1, k]
        e1 = np.zeros(k + 2)
        e1[0] = beta0
        y, *_ = scipy.linalg.lstsq(H[: k + 2, : k + 1], e1)
        xk = x0 + V[:, : k + 1] @ y
        out.append((xk, float(np.linalg.norm(rhs - B @ xk))))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_untruncated_aa_gd_is_gmres_walker_ni_thm_2_2(seed: int) -> None:
    """f = ½xᵀAx − bᵀx: g(x) = (I − αA)x + αb, so AA(∞) is GMRES on αA x = αb.

    At step k + 1: x̄ = Σ c*_i x_i = x_k^GMRES, ‖F c*‖ = ‖r_k^GMRES‖, x_{k+1} = g(x_k^GMRES).
    """
    rng = np.random.default_rng(seed)
    n = 8
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.geomspace(1.0, 50.0, n)) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(n)
    x0 = rng.standard_normal(n)
    alpha = 0.03
    res = anderson_gd(_quadratic(A, b, x0), m=50, lr=alpha, gtol=0.0, max_iter=n)
    gm = _gmres_iterates(alpha * A, alpha * b, x0, n - 1)
    kappa = float(np.linalg.cond(A))
    r0 = gm[0][1]
    for k in range(n):
        xk, rk = gm[k]
        s = res.trace[k + 1]
        # The residual norm is well conditioned: 100·κ(A)·ε·‖r₀‖ absolute.
        assert abs(s.info["lsq_residual"] - rk) <= 100 * kappa * EPS * r0
        # κ(ΔF) from the recorded iterates and the exact residuals f_i = −α(Ax_i − b), not from
        # info["cond"]: a wrong cond must not loosen the asserts below.
        hist = np.stack(s.info["history"], axis=1)  # (n, m_k + 1)
        dF = np.diff(-alpha * (A @ hist - b[:, None]), axis=1)  # (n, m_k)
        kappa_dF = float(np.linalg.cond(dF)) if dF.shape[1] else 1.0
        assert 1.0 <= kappa_dF < 1e8
        # NOTE: the iterate x̄ = x_k − ΔXγ* inherits the least-squares forward error of γ*, of
        # order κ(ΔF)·ε (Higham (2002), Thm 20.1, first-order term); ΔF = −αAΔX is a Krylov
        # power basis, so κ(ΔF) grows to ≈ 3e6 at k = 7 here. Measured
        # error / (κ(ΔF)·ε·max(1, ‖x_k‖)) ≤ 2 over the 5 seeds; the bound is 10× that scale.
        tol = 10 * kappa_dF * EPS * max(1.0, float(np.linalg.norm(xk)))
        assert_allclose(s.info["x_bar"], xk, rtol=0, atol=tol)
        assert_allclose(s.x, xk - alpha * (A @ xk - b), rtol=0, atol=tol)


def test_finite_termination_on_a_quadratic_in_n_plus_1_steps() -> None:
    """AA(m ≥ n)-GD on a strictly convex quadratic: x_{n+1} = x* (GMRES terminates at step n)."""
    prob = problems.get("quadratic_ill")
    res = anderson_gd(prob, m=50, lr=0.01, gtol=0.0, max_iter=prob.dim + 1)
    kappa = float(np.linalg.cond(prob.hess(np.asarray(prob.x0))))
    g0, gn = res.trace[0].grad_norm, res.trace[prob.dim + 1].grad_norm
    assert g0 is not None and gn is not None
    # NOTE: exact in exact arithmetic; the floating-point residual is O(κ·ε·‖∇f(x₀)‖).
    assert gn <= 10 * kappa * EPS * g0


@pytest.mark.parametrize(("pid", "beta"), [("rosenbrock", 1.0), ("beale", 0.6)])
def test_multisecant_form_fang_saad(pid: str, beta: float) -> None:
    """x_{k+1} = x_k − H_k f_k with H_k = −βI + (ΔX + βΔF)(ΔFᵀΔF)⁻¹ΔFᵀ, and H_k ΔF = ΔX."""
    prob = problems.get(pid)
    alpha = 1e-3
    res = anderson_gd(prob, m=1, lr=alpha, beta=beta, gtol=0.0, max_iter=12)
    resid = {s.k: -alpha * np.asarray(s.info["grad"]) for s in res.trace}
    for s in res.trace[2:]:
        hist = np.stack(s.info["history"], axis=1)  # x_{k−1−m}, …, x_{k−1}
        k = s.k - 1
        Fw = np.stack([resid[k - hist.shape[1] + 1 + i] for i in range(hist.shape[1])], axis=1)
        dX, dF = np.diff(hist, axis=1), np.diff(Fw, axis=1)
        H = -beta * np.eye(2) + (dX + beta * dF) @ scipy.linalg.solve(dF.T @ dF, dF.T)
        assert_allclose(H @ dF, dX, rtol=1e-9, atol=1e-14)
        assert_allclose(s.x, hist[:, -1] - H @ Fw[:, -1], rtol=1e-11, atol=1e-14)


# --------------------------------------------------------------------------------------
# The coefficient problem (property tests against independent oracles)
# --------------------------------------------------------------------------------------


def _kkt_coefficients(F: np.ndarray, lam_eff: float) -> np.ndarray:
    """Oracle: c from the KKT system of min ‖Fc‖² + λ'‖c‖² s.t. 1ᵀc = 1 (no differences)."""
    p = F.shape[1]
    K = np.zeros((p + 1, p + 1))
    K[:p, :p] = 2.0 * (F.T @ F + lam_eff * np.eye(p))
    K[:p, p] = K[p, :p] = 1.0
    rhs = np.zeros(p + 1)
    rhs[p] = 1.0
    return scipy.linalg.solve(K, rhs, assume_a="sym")[:p]


@st.composite
def _full_rank_windows(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    p = draw(st.integers(2, 7))  # m + 1 residuals
    n = draw(st.integers(p + 1, 14))
    seed = draw(st.integers(0, 2**31 - 1))
    F = np.random.default_rng(seed).standard_normal((n, p))
    scale = 10.0 ** draw(st.floats(-8, 8))
    lam = draw(st.sampled_from([0.0, 1e-10, 1e-6, 1e-2, 1.0]))
    return F, scale, lam


@settings(max_examples=1500, deadline=None)
@given(_full_rank_windows())
def test_coefficients_match_the_kkt_oracle_and_are_scale_invariant(
    case: tuple[np.ndarray, float, float],
) -> None:
    F, scale, lam = case
    c, _, cond, lam_eff = _aa_coefficients(scale * F, lam)
    assert math.isclose(c.sum(), 1.0, rel_tol=0, abs_tol=1e-12)
    c_ref = _kkt_coefficients(F, lam * np.linalg.norm(F, 2) ** 2)
    kappa = float(np.linalg.cond(F))
    # NOTE: the KKT oracle forms FᵀF, so its error is ≈ κ(F)²ε‖c‖ (Higham (2002), §20.4); the
    # bound is 1e3·κ²·ε for these Gaussian windows (n ≥ p + 1, κ(F) ≲ 30).
    tol = 1e3 * kappa**2 * EPS * max(1.0, float(np.linalg.norm(c_ref)))
    assert_allclose(c, c_ref, rtol=0, atol=tol)
    # λ is relative to ‖F‖₂²: c* does not depend on the units of F.
    assert math.isclose(lam_eff, lam * scale**2 * np.linalg.norm(F, 2) ** 2, rel_tol=1e-12)
    assert cond >= 1.0


@st.composite
def _underdetermined_windows(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    """m = p − 1 > n residual differences in ℝⁿ: ΔF (n × m) has rank ≤ n < m (m > n in 2-D)."""
    p = draw(st.integers(3, 9))
    n = draw(st.integers(1, p - 2))
    seed = draw(st.integers(0, 2**31 - 1))
    F = np.random.default_rng(seed).standard_normal((n, p))
    scale = 10.0 ** draw(st.floats(-8, 8))
    lam = draw(st.sampled_from([0.0, 1e-10, 1e-6, 1e-2, 1.0]))
    return F, scale, lam


@settings(max_examples=1500, deadline=None)
@given(_underdetermined_windows())
def test_coefficients_of_an_underdetermined_window(case: tuple[np.ndarray, float, float]) -> None:
    """λ = 0: ‖Fc*‖ is optimal and γ* ⟂ null(ΔF) (the minimum-norm solution); λ > 0: c* unique.

    Oracle: the parametrization c = 1/p + Nz (N spans {1ᵀv = 0}) solved by LAPACK gelsy on F,
    or on the stacked rows [F; √λ' I] for λ > 0; it never forms FᵀF.
    """
    F, scale, lam = case
    n, p = F.shape
    c, gamma, _, lam_eff = _aa_coefficients(scale * F, lam)
    assert math.isclose(c.sum(), 1.0, rel_tol=0, abs_tol=1e-12)
    N = scipy.linalg.null_space(np.ones((1, p)))  # (p, p − 1)
    c0 = np.full(p, 1.0 / p)
    lam_rel = lam * np.linalg.norm(F, 2) ** 2
    G = np.vstack([F, math.sqrt(lam_rel) * np.eye(p)]) if lam > 0 else F
    A = G @ N
    z, *_ = scipy.linalg.lstsq(A, -G @ c0, lapack_driver="gelsy")
    c_ref = c0 + N @ z
    s = scipy.linalg.svdvals(A)
    kappa = s[0] / s[s > 1e-12 * s[0]][-1]
    if lam > 0:
        # NOTE: least-squares perturbation bound (Higham (2002), Thm. 20.1); with
        # ‖r‖/‖A‖ ≲ √λ both terms are O(κε). Bound: 1e3·κ·ε·max(1, ‖c*‖).
        tol = 1e3 * kappa * EPS * max(1.0, float(np.linalg.norm(c_ref)))
        assert_allclose(c, c_ref, rtol=0, atol=tol)
        assert math.isclose(lam_eff, lam * scale**2 * np.linalg.norm(F, 2) ** 2, rel_tol=1e-12)
        return
    v_ref = float(np.linalg.norm(F @ c_ref))
    scale_c = max(1.0, float(np.linalg.norm(c)), float(np.linalg.norm(c_ref)))
    tol_v = 1e3 * kappa * EPS * float(np.linalg.norm(F, 2)) * scale_c
    assert abs(float(np.linalg.norm(F @ c)) - v_ref) <= tol_v
    dF = np.diff(F, axis=1)
    sd = scipy.linalg.svdvals(dF)
    r = int(np.sum(sd > 1e-12 * sd[0]))
    assert r <= n < dF.shape[1]
    _, _, Vt = scipy.linalg.svd(dF)
    null_dF = Vt[r:].T
    # NOTE: the null-space basis carries an O(κ_r ε) rotation error (Golub & Van Loan, Thm 8.6.5).
    kappa_r = sd[0] / sd[r - 1]
    bound = 1e3 * kappa_r * EPS * max(1.0, float(np.linalg.norm(gamma)))
    assert float(np.linalg.norm(null_dF.T @ gamma)) <= bound


def _stacked_cond(F: np.ndarray, lam: float) -> float:
    """Oracle: σ_max/σ_min (``scipy.linalg.svdvals``) of the stacked matrix [ΔF; √λ'·D].

    D (p × (p − 1)) maps γ to the zero-sum part of c = e + Dγ; the sign of a block of rows does
    not change the singular values. λ' = λ‖F‖₂².
    """
    p = F.shape[1]
    dF = F[:, 1:] - F[:, :-1]
    if lam > 0:
        D = -np.diff(np.eye(p), axis=0).T  # D_jj = 1, D_{j+1,j} = −1
        dF = np.vstack([dF, math.sqrt(lam * np.linalg.norm(F, 2) ** 2) * D])
    s = scipy.linalg.svdvals(dF)
    return float(s[0] / s[-1])


@settings(max_examples=1000, deadline=None)
@given(_full_rank_windows())
def test_cond_is_the_condition_number_of_the_stacked_matrix(
    case: tuple[np.ndarray, float, float],
) -> None:
    F, scale, lam = case
    _, _, cond, _ = _aa_coefficients(scale * F, lam)
    kappa = _stacked_cond(F, lam)
    # NOTE: a backward-stable SVD moves σ_min by O(ε σ_max), so κ moves by O(κ²ε).
    assert abs(cond - kappa) <= 1e2 * kappa**2 * EPS, (cond, kappa)


@pytest.mark.parametrize("lam", [0.0, 1e-2])
def test_info_cond_matches_the_window_of_a_run(lam: float) -> None:
    """info["cond"] = σ_max/σ_min of [ΔF; √λ'D], rebuilt from the recorded history and ∇f.

    rosenbrock_nd (n = 10) with m = 5: every window ΔF (10 × m_k) has full column rank.
    """
    prob = problems.get("rosenbrock_nd")
    alpha = 1e-3
    res = anderson_gd(prob, m=5, lr=alpha, lam=lam, gtol=0.0, max_iter=30)
    assert res.n_iter == 30
    assert res.trace[1].info["cond"] == 1.0  # the gradient step
    for s in res.trace[2:]:
        hist = np.stack(s.info["history"], axis=1)  # (n, m_k + 1)
        F = np.stack([-alpha * prob.grad(xi) for xi in hist.T], axis=1)
        kappa = _stacked_cond(F, lam)
        assert kappa < 1e7
        assert abs(s.info["cond"] - kappa) <= 1e2 * kappa**2 * EPS, (s.k, s.info["cond"], kappa)
        assert s.info["lam_eff"] == pytest.approx(lam * np.linalg.norm(F, 2) ** 2, rel=1e-12)


def test_large_lambda_gives_uniform_weights() -> None:
    F = np.random.default_rng(1).standard_normal((6, 4))
    c, *_ = _aa_coefficients(F, 1e12)
    assert_allclose(c, np.full(4, 0.25), rtol=0, atol=1e-10)


def test_rank_deficient_window_stays_finite() -> None:
    f = np.array([1.0, 2.0, 3.0])
    c, _, cond, _ = _aa_coefficients(np.stack([f, 2 * f, 3 * f], axis=1), 0.0)
    assert np.all(np.isfinite(c)) and math.isclose(c.sum(), 1.0, abs_tol=1e-12)
    # ΔF = [f, f] has rank 1: σ_min is rounding (or 0), so the reported cond is ≥ 1/(10ε).
    assert cond >= 0.1 / EPS


# --------------------------------------------------------------------------------------
# Honest saddle detection (the key finding of research/anderson-acceleration)
# --------------------------------------------------------------------------------------


def test_study_e4_saddle_counts_on_himmelblau() -> None:
    """research/anderson-acceleration E4, m = 5, α = 0.01, 8 × 8 starts in [−5, 5]², 2000 its.

    Study table: GD 64 / 0, λ = 0: 33 minimizers / 31 saddle-max, λ = 10⁻²: 47 / 17.
    """
    prob = problems.get("himmelblau")
    grid = np.linspace(-5.0, 5.0, 8)
    starts = [np.array([a, b]) for b in grid for a in grid]
    counts: dict[str, Counter[str]] = {}
    for key, kw in {"gd": {"m": 0}, "lam0": {"m": 5}, "lam1e-2": {"m": 5, "lam": 1e-2}}.items():
        c: Counter[str] = Counter()
        for x0 in starts:
            res = anderson_gd(prob, x0=x0, lr=0.01, gtol=1e-6, max_iter=2000, **kw)
            kind = _kind(res)
            if kind in ("minimizer", "saddle/max"):
                # The classification agrees with an independent eigensolver on the exact ∇²f.
                lam_min = float(scipy.linalg.eigvalsh(prob.hess(np.asarray(res.x)))[0])
                assert (lam_min > 0) == (kind == "minimizer"), (x0, res.message)
            c[kind] += 1
        counts[key] = c
    assert counts["gd"] == Counter({"minimizer": 64})
    assert counts["lam0"] == Counter({"minimizer": 33, "saddle/max": 31})
    assert counts["lam1e-2"] == Counter({"minimizer": 47, "saddle/max": 17})


def test_rosenbrock_nd_default_start_stops_at_the_study_saddle() -> None:
    """Study E3: AA(5)-GD stops at the saddle f = 9.606, λ_min(∇²f) = −2.81 (47 gradients there)."""
    prob = problems.get("rosenbrock_nd")
    res = numopt.run("anderson_gd", prob)
    assert_valid_result(res, max_iter=500)
    assert not res.converged
    assert "stopped at a saddle point, not a minimizer" in res.message
    assert res.fun == pytest.approx(9.606, abs=5e-4)
    eigs = scipy.linalg.eigvalsh(prob.hess(np.asarray(res.x)))
    assert eigs[0] == pytest.approx(-2.81, abs=5e-3) and eigs[-1] > 0
    assert_allclose(res.trace[-1].info["hess_eigs"], eigs, rtol=1e-10, atol=1e-10)
    assert res.n_hev == 1 and res.n_gev == res.n_iter + 1


@settings(max_examples=1000, deadline=None)
@given(st.floats(-5, 5), st.floats(-5, 5), st.sampled_from([0, 1, 3, 5, 10]))
def test_every_stop_is_classified_by_the_true_hessian(u: float, v: float, m: int) -> None:
    """Property: converged ⇔ ∇²f(x) ⪰ 0 at a point with ‖∇f‖₂ ≤ gtol (himmelblau, any start)."""
    prob = problems.get("himmelblau")
    res = anderson_gd(prob, x0=[u, v], m=m, lr=0.01, gtol=1e-6, max_iter=500)
    assert_valid_result(res, max_iter=500)
    g_final = float(np.linalg.norm(prob.grad(np.asarray(res.x))))
    lam = scipy.linalg.eigvalsh(prob.hess(np.asarray(res.x)))
    if res.converged:
        assert g_final <= 1e-6 and lam[0] > 0
    elif "not a minimizer" in res.message:
        assert g_final <= 1e-6 and lam[0] < 0
        expected = "a maximizer" if lam[-1] < 0 else "a saddle point"
        assert f"stopped at {expected}," in res.message
    else:
        assert g_final > 1e-6 and res.n_hev == 0


@pytest.mark.parametrize(
    ("f", "grad", "hess", "x0", "conv", "phrase"),
    [
        # x² − y²: AA solves ∇f = 0 and reaches the saddle point (GD would move away from it).
        (
            lambda x: x[0] ** 2 - x[1] ** 2,
            lambda x: np.array([2 * x[0], -2 * x[1]]),
            lambda x: np.diag([2.0, -2.0]),
            [0.7, 0.3],
            False,
            "stopped at a saddle point,",
        ),
        # −x² − y²: AA reaches the maximizer.
        (
            lambda x: -(x[0] ** 2) - x[1] ** 2,
            lambda x: np.array([-2 * x[0], -2 * x[1]]),
            lambda x: np.diag([-2.0, -2.0]),
            [0.7, 0.3],
            False,
            "stopped at a maximizer,",
        ),
        # −x² + y⁴ at 0: ∇²f = diag(−2, 0) cannot tell a saddle from a maximizer.
        (
            lambda x: -(x[0] ** 2) + x[1] ** 4,
            lambda x: np.array([-2 * x[0], 4 * x[1] ** 3]),
            lambda x: np.diag([-2.0, 12 * x[1] ** 2]),
            [0.0, 0.0],
            False,
            "a saddle point or a maximizer",
        ),
        # x⁴ + y² at 0: a minimizer with a singular ∇²f; converged, second order not verified.
        (
            lambda x: x[0] ** 4 + x[1] ** 2,
            lambda x: np.array([4 * x[0] ** 3, 2 * x[1]]),
            lambda x: np.diag([12 * x[0] ** 2, 2.0]),
            [0.0, 0.0],
            True,
            "positive semidefinite but numerically singular",
        ),
    ],
)
def test_second_order_classification(
    f: Any, grad: Any, hess: Any, x0: list[float], conv: bool, phrase: str
) -> None:
    res = anderson_gd(_poly2(f, grad, hess, x0), m=3, lr=0.1, gtol=1e-10)
    assert_valid_result(res, max_iter=500)
    assert res.converged is conv, res.message
    assert phrase in res.message
    assert res.n_hev == 1
    if x0 == [0.0, 0.0]:
        assert res.n_iter == 0 and res.message.startswith("at x0:")


# Two problems whose global minimizers are not isolated: f ≥ 0, f = 0 on a curve, and the only
# other stationary point is the origin (f = 1): the maximizer of the ring, the saddle of the
# hyperbola (∇²f(0) = [[0, −2], [−2, 0]]). Oracle: a stop with f ≤ 1e-10 is a global minimizer.
_VALLEYS = {
    "ring": (
        lambda x: float((x @ x - 1.0) ** 2),
        lambda x: 4.0 * (x @ x - 1.0) * x,
        lambda x: 4.0 * (x @ x - 1.0) * np.eye(2) + 8.0 * np.outer(x, x),
        "stopped at a maximizer,",
    ),
    "hyperbola": (
        lambda x: float((x[0] * x[1] - 1.0) ** 2),
        lambda x: 2.0 * (x[0] * x[1] - 1.0) * np.array([x[1], x[0]]),
        lambda x: np.array(
            [
                [2.0 * x[1] ** 2, 2.0 * (2.0 * x[0] * x[1] - 1.0)],
                [2.0 * (2.0 * x[0] * x[1] - 1.0), 2.0 * x[0] ** 2],
            ]
        ),
        "stopped at a saddle point,",
    ),
}


@pytest.mark.parametrize(
    ("name", "m", "expected"),
    [
        ("ring", 0, (50, 40)),
        ("ring", 3, (39, 15)),
        ("hyperbola", 0, (50, 0)),
        ("hyperbola", 3, (18, 12)),
    ],
)
def test_non_isolated_minimizers_are_not_reported_as_saddles(
    name: str, m: int, expected: tuple[int, int]
) -> None:
    """Regression: λ_min(∇²f(x)) ≈ −c‖∇f(x)‖ < 0 next to a valley of minimizers.

    expected = (runs that stop on the valley, runs among them with λ_min < −tol_H). Before the
    √(‖∇f‖·|λ|_max) allowance, every run in the second count was labelled "saddle point", e.g.
    λ_min = −7.2e-7 against tol_H = 3.6e-15 at ‖∇f‖ = 7.2e-7 on the ring.
    """
    f, grad, hess, origin_phrase = _VALLEYS[name]
    starts = np.random.default_rng(0).uniform(-0.9, 0.9, (50, 2))
    n_valley = n_gradient_band = 0
    for x0 in starts:
        res = anderson_gd(_poly2(f, grad, hess, list(x0)), m=m, lr=0.05, max_iter=5000)
        assert_valid_result(res, max_iter=5000)
        eigs = res.trace[-1].info["hess_eigs"]
        assert eigs is not None, res.message  # every run passes the gradient test
        if f(np.asarray(res.x)) <= 1e-10:
            n_valley += 1
            assert res.converged, (x0, res.message)
            assert "strict local minimizer" not in res.message
            # The bug regime: λ_min is negative far beyond the rounding of eigvalsh.
            if eigs[0] < -2 * 2 * EPS * abs(eigs).max():
                n_gradient_band += 1
                assert "positive semidefinite to the accuracy of the gradient test" in res.message
        else:
            # The origin: the second-order test still rejects it.
            assert_allclose(res.x, [0.0, 0.0], rtol=0, atol=1e-6)
            assert not res.converged and origin_phrase in res.message, res.message
    assert (n_valley, n_gradient_band) == expected


@settings(max_examples=1000, deadline=None)
@given(st.floats(-2, 2), st.floats(-2, 2), st.sampled_from([0, 1, 3, 5]), st.sampled_from([0, 1]))
def test_valley_stops_are_classified_by_the_value(u: float, v: float, m: int, which: int) -> None:
    """Property: converged ⇔ f(x) ≤ 1e-10 (a global minimizer) on the ring and the hyperbola."""
    f, grad, hess, _ = _VALLEYS[("ring", "hyperbola")[which]]
    res = anderson_gd(_poly2(f, grad, hess, [u, v]), m=m, lr=0.02, max_iter=2000)
    assert_valid_result(res, max_iter=2000)
    if res.trace[-1].info["hess_eigs"] is None:
        assert not res.converged and "not a minimizer" not in res.message
        return
    fx = f(np.asarray(res.x))
    assert res.converged == (fx <= 1e-10), (fx, res.message)


@pytest.mark.parametrize(
    ("eigs", "gnorm", "conv", "phrase"),
    [
        # tol_g = √(1e-6·8) ≈ 2.8e-3 > 1e-7: within the accuracy of the gradient test.
        ([-1e-7, 8.0], 1e-6, True, "accuracy of the gradient test"),
        # 1e-2 > tol_g: a saddle point, as before.
        ([-1e-2, 8.0], 1e-6, False, "stopped at a saddle point,"),
        # ‖∇f‖ = 0: tol_g = 0, only the rounding tolerance of ∇²f applies.
        ([-1e-7, 8.0], 0.0, False, "stopped at a saddle point,"),
        # λ_min > 0 but below tol_g: positive definite is not claimed.
        ([1e-7, 8.0], 1e-6, True, "accuracy of the gradient test"),
        ([1e-2, 8.0], 1e-6, True, "strict local minimizer"),
        # ∇²f ⪯ 0 and λ_max within tol_g of 0: a saddle point or a maximizer.
        ([-1.0, 1e-4], 1e-6, False, "a saddle point or a maximizer"),
    ],
)
def test_classification_tolerance_includes_the_gradient_norm(
    eigs: list[float], gnorm: float, conv: bool, phrase: str
) -> None:
    lam = np.array(eigs)
    tol_h = aa_mod._eig_tol(lam, 0.0, hess_fd=False, grad_fd=False)
    tol_g = aa_mod._stationarity_tol(lam, gnorm)
    assert tol_g == pytest.approx(math.sqrt(gnorm * max(abs(e) for e in eigs)), rel=1e-15)
    ok, why = aa_mod._classify(lam, tol_h, tol_g, hess_fd=False)
    assert ok is conv and phrase in why, why


def test_finite_difference_hessian_classifies_a_saddle() -> None:
    """No analytic ∇²f: a central difference of ∇f (2n gradient calls) is used and counted."""
    prob = Problem(
        id="s",
        name="s",
        latex="s",
        f=lambda x: x[0] ** 2 - x[1] ** 2,
        grad=lambda x: np.array([2 * x[0], -2 * x[1]]),
        dim=2,
        domain=(),
        x0=[0.7, 0.3],
    )
    res = anderson_gd(prob, m=3, lr=0.1, gtol=1e-10)
    assert not res.converged
    assert "stopped at a saddle point, not a minimizer: the finite-difference ∇²f" in res.message
    assert res.n_hev == 1 and res.n_gev == res.n_iter + 1 + 2 * 2


# --------------------------------------------------------------------------------------
# Evaluation counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ["rosenbrock", "himmelblau", "quadratic_nd"])
def test_evaluation_counts_are_exact(pid: str) -> None:
    prob = problems.get(pid)
    calls = Counter[str]()

    def wrap(name: str, fn: Any) -> Any:
        def inner(x: Any) -> Any:
            calls[name] += 1
            return fn(x)

        return inner

    counted = Problem(
        id=prob.id,
        name=prob.name,
        latex=prob.latex,
        f=wrap("f", prob.f),
        grad=wrap("g", prob.grad),
        hess=wrap("h", prob.hess),
        dim=prob.dim,
        domain=prob.domain,
        x0=prob.x0,
    )
    res = anderson_gd(counted, lr=0.01 if pid != "rosenbrock" else 1e-3)
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], calls["h"])
    assert res.n_fev == res.n_gev == res.n_iter + 1
    assert res.n_hev == (1 if res.trace[-1].info["hess_eigs"] is not None else 0)


def test_bare_callable_uses_finite_differences_and_counts_them() -> None:
    calls = Counter[str]()

    def f(x: np.ndarray) -> float:
        calls["f"] += 1
        return float((x[0] - 1.0) ** 2 + 2.0 * x[1] ** 2)

    res = anderson_gd(f, x0=[3.0, 1.0], m=3, lr=0.1)
    assert res.converged, res.message
    assert "finite-difference ∇²f" in res.message
    assert_allclose(res.x, [1.0, 0.0], rtol=0, atol=1e-6)
    n = 2
    # Per iterate: f once plus 2n for the central-difference ∇f; the FD Hessian at the end
    # takes 2n FD gradients (2n·2n f calls), counted in n_gev as well.
    assert res.n_fev == calls["f"] == (res.n_iter + 1) * (1 + 2 * n) + (2 * n) * (2 * n)
    assert res.n_gev == res.n_iter + 1 + 2 * n and res.n_hev == 1


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


def test_max_iter_is_reported() -> None:
    res = numopt.run("anderson_gd", problems.get("rosenbrock"), m=1, max_iter=20)
    assert_valid_result(res, max_iter=20)
    assert not res.converged and res.message == "max_iter = 20 reached" and res.n_iter == 20
    assert res.n_hev == 0


def test_converged_at_x0() -> None:
    prob = problems.get("quadratic_bowl")
    res = anderson_gd(prob, x0=prob.minima[0])
    assert res.converged and res.n_iter == 0 and res.n_hev == 1
    assert res.message.startswith("at x0:")


def test_divergence_is_reported_and_a_zero_window_stays_finite() -> None:
    """f = x + y: ∇f is constant, ΔF = 0 (cond = inf), and x drifts by α per step."""
    prob = _poly2(
        lambda x: float(x[0] + x[1]), lambda x: np.ones(2), lambda x: np.zeros((2, 2)), [0, 0]
    )
    res = anderson_gd(prob, m=3, lr=1e11, max_iter=200)
    assert_valid_result(res, max_iter=200)
    assert not res.converged and res.message.startswith("diverged")
    assert all(s.info["cond"] == math.inf for s in res.trace[2:])
    assert all(np.all(np.isfinite(np.asarray(s.x))) for s in res.trace)


def test_non_finite_value_at_x0_is_reported() -> None:
    res = anderson_gd(lambda x: float("inf"), x0=[0.0, 0.0])
    assert_valid_result(res)
    assert not res.converged and "not finite at x0" in res.message and res.n_iter == 0


def test_overflow_raised_by_f_is_reported_not_raised() -> None:
    """f = exp(x₀²) + x₁²: the first step jumps to x₀ ≈ −5400, where math.exp overflows."""

    def f(x: np.ndarray) -> float:
        return math.exp(x[0] ** 2) + x[1] ** 2  # OverflowError for x₀² > 709.78

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * x[0] * math.exp(x[0] ** 2), 2.0 * x[1]])

    prob = Problem(id="e", name="e", latex="e", f=f, grad=grad, dim=2, domain=(), x0=[1.0, 1.0])
    res = anderson_gd(prob, m=2, lr=1e3, max_iter=50)
    assert_valid_result(res, max_iter=50)
    assert not res.converged
    assert res.message == "f or ∇f is not finite at iteration 1" and res.n_iter == 1


def test_overflowing_residual_difference_is_reported() -> None:
    """f_0 = 1e308, f_1 = −1e308: ΔF = −inf; no SVD is tried and nothing is raised."""
    prob = Problem(
        id="o",
        name="o",
        latex="o",
        f=lambda x: 0.0,
        grad=lambda x: np.array([-1e308 if x[0] < 5e307 else 1e308]),
        dim=1,
        domain=(),
        x0=[1e307],
    )
    res = anderson_gd(prob, m=2, lr=1.0, max_iter=10)
    assert_valid_result(res, max_iter=10)
    assert not res.converged
    assert res.message == "non-finite residual difference at iteration 2"


def test_stall_is_reported() -> None:
    """x = 1e20 and a step α‖∇f‖ = 1e-10 below its spacing: x_{k+1} = x_k in floating point."""
    prob = Problem(
        id="l",
        name="l",
        latex="l",
        f=lambda x: float(x[0]),
        grad=lambda x: np.ones(1),
        dim=1,
        domain=(),
        x0=[1e20],
    )
    res = anderson_gd(prob, lr=1e-10)
    assert not res.converged and res.message.startswith("stalled") and res.n_iter == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"m": -1},
        {"m": 1.5},
        {"lr": 0.0},
        {"lr": math.inf},
        {"beta": 0.0},
        {"beta": math.nan},
        {"lam": -1.0},
        {"gtol": -1.0},
        {"max_iter": 0},
        {"x0": [1.0, 2.0, 3.0]},
    ],
)
def test_invalid_input_raises(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        anderson_gd(problems.get("rosenbrock"), **kw)


# --------------------------------------------------------------------------------------
# Fixtures for the web app
# --------------------------------------------------------------------------------------


def test_fixture_cases_are_short_deterministic_and_registered() -> None:
    assert 3 <= len(aa_mod.FIXTURE_CASES) <= 6
    for method, pid, params in aa_mod.FIXTURE_CASES:
        assert method == "anderson_gd"
        a = numopt.run(method, problems.get(pid), **params)
        b = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(a)
        assert len(a.trace) < 400
        assert a.to_dict() == b.to_dict()


def test_fixture_pair_shows_the_rna_effect() -> None:
    """The two himmelblau fixtures share x₀ = (1, 1): λ = 0 stops at a saddle, λ = 10⁻² at (3, 2)."""
    prob = problems.get("himmelblau")
    plain = anderson_gd(prob, x0=[1.0, 1.0], lr=0.01)
    rna = anderson_gd(prob, x0=[1.0, 1.0], lr=0.01, lam=0.01)
    assert not plain.converged and "saddle point" in plain.message
    assert rna.converged
    assert_allclose(rna.x, [3.0, 2.0], rtol=0, atol=1e-6)
