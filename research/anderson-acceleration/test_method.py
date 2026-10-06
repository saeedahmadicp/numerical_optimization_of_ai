"""Tests for AA(m) / RNA (method.py). Run: .venv/bin/pytest research/anderson-acceleration -q"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from numopt import problems, run
from numopt.core.types import LinearSystem, Problem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from method import aa_coefficients, anderson, anderson_gd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "numopt_conftest", HERE.parents[1] / "tests" / "conftest.py"
)
assert _spec is not None and _spec.loader is not None
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
assert_valid_result = _conftest.assert_valid_result

EPS = np.finfo(float).eps


def cos_map(A: np.ndarray) -> Problem:
    """F(x) = cos(Ax) − x, so g(x) = x + F(x) = cos(Ax)."""
    return Problem(
        id="cos_map",
        name="cos(Ax)",
        latex=r"\cos(Ax)-x",
        f=lambda x: np.cos(A @ x) - x,
        dim=2,
        domain=((-2, 2), (-2, 2)),
        x0=(0.0, 0.0),
    )


A_CONTRACT = np.array([[1.0, 0.5], [0.2, 0.6]])  # ρ(g'(x*)) ≈ 0.91


def kkt_coefficients(F: np.ndarray, lam_eff: float) -> np.ndarray:
    """Oracle: c from the KKT system of min ‖Fc‖² + λ'‖c‖² s.t. 1ᵀc = 1 (no differences)."""
    p = F.shape[1]
    K = np.zeros((p + 1, p + 1))
    K[:p, :p] = 2.0 * (F.T @ F + lam_eff * np.eye(p))
    K[:p, p] = K[p, :p] = 1.0
    rhs = np.zeros(p + 1)
    rhs[p] = 1.0
    return scipy.linalg.solve(K, rhs, assume_a="sym")[:p]


# ---------------------------------------------------------------- contract & convergence


@pytest.mark.parametrize("m", [0, 1, 3, 10])
def test_contract_and_convergence_fixed_point(m: int) -> None:
    res = anderson(cos_map(A_CONTRACT), m=m, ftol=1e-10, max_iter=500)
    assert_valid_result(res, max_iter=500)
    assert res.converged and res.fun is not None and res.fun <= 1e-10
    assert res.n_fev == res.n_iter + 1  # one F evaluation per iterate
    for s in res.trace[1:]:
        assert math.isclose(sum(s.info["coefficients"]), 1.0, rel_tol=0, abs_tol=1e-12)
        assert s.info["memory"] == min(m, s.k - 1)


@pytest.mark.parametrize("pid", ["trig_system", "intersecting_circles"])
def test_converges_on_library_systems(pid: str) -> None:
    p = problems.get(pid)
    res = anderson(p, m=3, omega=-0.2, ftol=1e-10, max_iter=300)
    assert_valid_result(res, max_iter=300)
    assert res.converged
    assert np.linalg.norm(np.asarray(p.f(res.x))) <= 1e-10


@pytest.mark.parametrize(("pid", "lr"), [("quadratic_bowl", 0.1), ("rosenbrock", 1e-3)])
def test_anderson_gd_converges(pid: str, lr: float) -> None:
    p = problems.get(pid)
    res = anderson_gd(p, m=5, lr=lr, gtol=1e-8, max_iter=3000)
    assert_valid_result(res, max_iter=3000)
    assert res.converged
    assert np.linalg.norm(p.grad(np.asarray(res.x))) <= 1e-8
    assert_allclose(res.x, p.minima[0], atol=1e-6)
    assert res.n_gev == res.n_iter + 1 and res.n_fev == res.n_iter + 1


def test_bare_callable_uses_fd_gradient() -> None:
    res = anderson_gd(lambda x: float((x[0] - 1) ** 2 + 2 * x[1] ** 2), x0=[3.0, 1.0], lr=0.1)
    assert res.converged and res.n_gev == 0 and res.n_fev == (res.n_iter + 1) * 5  # f + 2n FD
    assert_allclose(res.x, [1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------- exact first steps


def test_m0_is_numopt_fixed_point_iteration() -> None:
    """AA(0), β = 1 on F = cos x − x is x ← cos x = numopt fixed_point with λ = −1."""
    p = problems.get("cos_minus_x")
    ref = run("fixed_point", p, lam=-1.0, xtol=0.0, max_iter=40)
    res = anderson(lambda x: np.cos(x) - x, x0=[p.x0], m=0, ftol=0.0, max_iter=40)
    xs_ref = [s.x for s in ref.trace]
    xs = [float(s.x[0]) for s in res.trace[: len(xs_ref)]]
    assert_allclose(xs, xs_ref, rtol=0, atol=0)  # the same floating-point operations


def test_hand_computed_first_two_steps() -> None:
    """k = 0: x₁ = x₀ + βf₀. k = 1: γ = f₁ᵀΔf/‖Δf‖², x₂ = x₁ + βf₁ − (Δx + βΔf)γ."""
    A, beta = A_CONTRACT, 0.7
    g = lambda x: np.cos(A @ x)  # noqa: E731
    x0 = np.array([0.3, -0.4])
    f0 = g(x0) - x0
    x1 = x0 + beta * f0
    f1 = g(x1) - x1
    df, dx = f1 - f0, x1 - x0
    gamma = f1 @ df / (df @ df)
    x2 = x1 + beta * f1 - (dx + beta * df) * gamma
    res = anderson(cos_map(A), x0=x0, m=1, beta=beta, ftol=0.0, max_iter=2)
    assert_allclose(res.trace[1].x, x1, rtol=1e-15)
    assert_allclose(res.trace[2].x, x2, rtol=1e-13)
    assert_allclose(res.trace[2].info["coefficients"], [gamma, 1 - gamma], rtol=1e-13)


# ---------------------------------------------------------------- Walker–Ni Thm. 2.2


@pytest.mark.parametrize("seed", range(5))
def test_untruncated_aa_reproduces_gmres(seed: int) -> None:
    """Linear g(x) = Mx + b: ‖F_k c*‖ = ‖r_k^GMRES‖, x̄_k = x_k^GMRES, x_{k+1} = g(x_k^GMRES)."""
    rng = np.random.default_rng(seed)
    n = 10
    M = rng.standard_normal((n, n))
    M *= 0.9 / max(abs(np.linalg.eigvals(M)))  # ρ(M) = 0.9, non-normal
    b = rng.standard_normal(n)
    x0 = rng.standard_normal(n)
    gm = run("gmres", LinearSystem("t", "t", np.eye(n) - M, b), x0=x0, restart=n, tol=1e-15)
    aa = anderson(lambda x: M @ x + b - x, x0=x0, m=100, ftol=0.0, max_iter=n)
    r0 = np.linalg.norm(b - (np.eye(n) - M) @ x0)
    kappa = np.linalg.cond(np.eye(n) - M)
    K = min(len(gm.trace), n)
    aa_res = np.array([aa.trace[k + 1].info["lsq_residual"] for k in range(K)])
    gm_res = np.array([gm.trace[k].fun for k in range(K)])
    # NOTE: both methods carry rounding of order κ(I − M)·ε·‖r₀‖ in the residual; the bound
    # below is 100·κ·ε·‖r₀‖ absolute (κ ≈ 10–60 for these matrices), no relative slack.
    atol = 100 * kappa * EPS * r0
    assert_allclose(aa_res, gm_res, rtol=0, atol=atol)
    for k in range(K):
        xg = np.asarray(gm.trace[k].x)
        scale = 100 * kappa * EPS * max(1.0, np.linalg.norm(xg))
        assert_allclose(aa.trace[k + 1].info["x_bar"], xg, rtol=0, atol=scale)
        assert_allclose(aa.trace[k + 1].x, M @ xg + b, rtol=0, atol=scale)


def test_anderson_gd_terminates_on_quadratic_in_n_plus_1_steps() -> None:
    """AA(∞)-GD on a strictly convex quadratic is GMRES on Ax = c: x_{n+1} = g(x_n^GMRES) = x*."""
    p = problems.get("quadratic_ill")
    res = anderson_gd(p, m=50, lr=0.01, gtol=0.0, max_iter=p.dim + 1)
    kappa = np.linalg.cond(np.asarray(p.extra["A"]))
    g0 = res.trace[0].grad_norm
    gn = res.trace[p.dim + 1].grad_norm
    assert g0 is not None and gn is not None
    # NOTE: exact in exact arithmetic; the floating-point residual is O(κ·ε·‖∇f(x₀)‖)
    # (measured 2.3e-12 vs κ·ε·‖∇f(x₀)‖ = 1.6e-12); the bound is 10× that scale.
    assert gn <= 10 * kappa * EPS * g0


# ---------------------------------------------------------------- Fang–Saad multisecant form


def test_multisecant_form_fang_saad() -> None:
    """x_{k+1} = x_k − H_k f_k, H_k = −βI + (ΔX + βΔF)(ΔFᵀΔF)⁻¹ΔFᵀ, and H_k ΔF = ΔX."""
    beta = 0.8
    res = anderson(cos_map(A_CONTRACT), x0=[1.5, -1.0], m=1, beta=beta, ftol=0.0, max_iter=6)
    resid = {s.k: np.asarray(s.info["residual"]) for s in res.trace}
    for s in res.trace[2:]:
        hist = np.stack(s.info["history"], axis=1)  # x_{k−m}, …, x_k
        k = s.k - 1
        Fw = np.stack([resid[k - hist.shape[1] + 1 + i] for i in range(hist.shape[1])], axis=1)
        dX, dF = np.diff(hist, axis=1), np.diff(Fw, axis=1)
        H = -beta * np.eye(2) + (dX + beta * dF) @ np.linalg.solve(dF.T @ dF, dF.T)
        assert_allclose(H @ dF, dX, rtol=1e-10, atol=1e-14)
        assert_allclose(s.x, hist[:, -1] - H @ Fw[:, -1], rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------- coefficient solve (property)


@st.composite
def residual_windows(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    p = draw(st.integers(2, 7))  # m + 1 residuals
    n = draw(st.integers(p + 1, 14))
    seed = draw(st.integers(0, 2**31 - 1))
    F = np.random.default_rng(seed).standard_normal((n, p))
    scale = 10.0 ** draw(st.floats(-8, 8))
    lam = draw(st.sampled_from([0.0, 1e-10, 1e-6, 1e-2, 1.0]))
    return F, scale, lam


@settings(max_examples=1500, deadline=None)
@given(residual_windows())
def test_coefficients_match_kkt_oracle_and_are_scale_invariant(
    case: tuple[np.ndarray, float, float],
) -> None:
    F, scale, lam = case
    c, _, cond, lam_eff = aa_coefficients(scale * F, lam)
    assert math.isclose(c.sum(), 1.0, rel_tol=0, abs_tol=1e-12)
    c_ref = kkt_coefficients(F, lam * np.linalg.norm(F, 2) ** 2)
    kappa = np.linalg.cond(F)
    # NOTE: the KKT oracle squares κ(F) (it forms FᵀF); its error is ≈ κ(F)²ε‖c‖. The bound
    # is 1e3·κ²·ε, κ(F) ≤ ~30 for these Gaussian windows (n ≥ p + 1).
    tol = 1e3 * kappa**2 * EPS * max(1.0, np.linalg.norm(c_ref))
    assert_allclose(c, c_ref, rtol=0, atol=tol)
    # λ is relative to ‖F‖₂², so c does not depend on the units of F.
    assert math.isclose(lam_eff, lam * scale**2 * np.linalg.norm(F, 2) ** 2, rel_tol=1e-12)
    assert cond >= 1.0


@st.composite
def underdetermined_windows(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    """m = p − 1 > n residual differences in R^n: ΔF (n × m) has rank ≤ n < m (E2 has n = 2)."""
    p = draw(st.integers(3, 9))
    n = draw(st.integers(1, p - 2))
    seed = draw(st.integers(0, 2**31 - 1))
    F = np.random.default_rng(seed).standard_normal((n, p))
    scale = 10.0 ** draw(st.floats(-8, 8))
    lam = draw(st.sampled_from([0.0, 1e-10, 1e-6, 1e-2, 1.0]))
    return F, scale, lam


@settings(max_examples=1500, deadline=None)
@given(underdetermined_windows())
def test_coefficients_underdetermined_window(case: tuple[np.ndarray, float, float]) -> None:
    """Rank-deficient ΔF (m > n): γ* is not unique for λ = 0; lstsq must return the min-norm γ*.

    λ = 0: (i) ‖Fc‖ equals the optimal value of min_{1ᵀc=1} ‖Fc‖, from an independent
    parametrization c = 1/p + N z (N spans {1ᵀv = 0}) solved by LAPACK gelsy, and (ii) γ ⟂ null(ΔF),
    which with (i) characterizes the minimum-norm least-squares solution.
    λ > 0: the problem is strictly convex and c* is unique; the same parametrization with the
    stacked rows [F; √λ' I] gives an oracle that never forms FᵀF.
    """
    F, scale, lam = case
    n, p = F.shape
    c, gamma, _, lam_eff = aa_coefficients(scale * F, lam)
    assert math.isclose(c.sum(), 1.0, rel_tol=0, abs_tol=1e-12)
    N = scipy.linalg.null_space(np.ones((1, p)))  # (p, p − 1)
    c0 = np.full(p, 1.0 / p)
    lam_rel = lam * np.linalg.norm(F, 2) ** 2  # λ' for the unscaled F
    G = np.vstack([F, math.sqrt(lam_rel) * np.eye(p)]) if lam > 0 else F  # (n [+ p], p)
    A = G @ N
    z, *_ = scipy.linalg.lstsq(A, -G @ c0, lapack_driver="gelsy")
    c_ref = c0 + N @ z
    s = scipy.linalg.svdvals(A)
    kappa = s[0] / s[s > 1e-12 * s[0]][-1]  # condition number on the range of A
    if lam > 0:
        # NOTE: least-squares perturbation bound (Higham 2002, Thm. 20.1) ≈ (κ + κ²·‖r‖/(‖A‖‖z‖))ε;
        # here ‖r‖/‖A‖ ≲ √λ ≈ 1/κ, so both terms are O(κε). Bound: 1e3·κ·ε·max(1, ‖c*‖).
        tol = 1e3 * kappa * EPS * max(1.0, np.linalg.norm(c_ref))
        assert_allclose(c, c_ref, rtol=0, atol=tol)
        assert math.isclose(lam_eff, lam * scale**2 * np.linalg.norm(F, 2) ** 2, rel_tol=1e-12)
        return
    # λ = 0: optimal value (unique), then the minimum-norm selection of γ.
    v_ref = float(np.linalg.norm(F @ c_ref))
    tol_v = (
        1e3
        * kappa
        * EPS
        * np.linalg.norm(F, 2)
        * max(1.0, np.linalg.norm(c), np.linalg.norm(c_ref))
    )
    assert abs(float(np.linalg.norm(F @ c)) - v_ref) <= tol_v
    dF = np.diff(F, axis=1)  # (n, p − 1), rank ≤ n < p − 1
    sd = scipy.linalg.svdvals(dF)
    r = int(np.sum(sd > 1e-12 * sd[0]))
    assert r <= n < dF.shape[1]
    _, _, Vt = scipy.linalg.svd(dF)
    null_dF = Vt[r:].T  # (p − 1, p − 1 − r)
    kappa_r = sd[0] / sd[r - 1]
    # NOTE: the null-space basis carries an O(κ_r ε) rotation error (Golub & Van Loan, Thm. 8.6.5).
    assert np.linalg.norm(null_dF.T @ gamma) <= 1e3 * kappa_r * EPS * max(
        1.0, np.linalg.norm(gamma)
    )


def test_large_lambda_gives_uniform_weights() -> None:
    F = np.random.default_rng(1).standard_normal((6, 4))
    c, *_ = aa_coefficients(F, 1e12)
    assert_allclose(c, np.full(4, 0.25), atol=1e-10)


def test_rank_deficient_window_stays_finite() -> None:
    f = np.array([1.0, 2.0, 3.0])
    F = np.stack([f, 2 * f, 3 * f], axis=1)  # rank 1
    c, *_ = aa_coefficients(F, 0.0)
    assert np.all(np.isfinite(c)) and math.isclose(c.sum(), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------- SciPy oracle (same root)


def test_same_root_as_scipy_anderson() -> None:
    p = cos_map(A_CONTRACT)
    ref = scipy.optimize.anderson(p.f, np.zeros(2), M=3, f_tol=1e-12)
    res = anderson(p, m=3, ftol=1e-12)
    assert res.converged
    assert_allclose(res.x, ref, atol=1e-10)


# ---------------------------------------------------------------- failure paths


def test_max_iter_is_reported() -> None:
    res = anderson(cos_map(A_CONTRACT), m=0, ftol=1e-14, max_iter=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and "max_iter" in res.message and res.n_iter == 5


def test_divergence_is_reported() -> None:
    res = anderson(lambda x: x + 1.0, x0=[1.0], m=0, max_iter=200)  # g(x) = 2x + 1
    assert_valid_result(res)
    assert not res.converged and "diverged" in res.message


def test_non_finite_residual_is_reported() -> None:
    res = anderson(lambda x: np.array([np.inf]), x0=[0.0], m=2)
    assert_valid_result(res)
    assert not res.converged and "non-finite" in res.message and res.n_iter == 0


def test_converged_at_x0() -> None:
    res = anderson_gd(problems.get("quadratic_bowl"), x0=problems.get("quadratic_bowl").minima[0])
    assert res.converged and res.n_iter == 0


@pytest.mark.parametrize(
    "kw", [{"m": -1}, {"m": 1.5}, {"beta": 0.0}, {"lam": -1.0}, {"max_iter": 0}, {"omega": 0.0}]
)
def test_invalid_input_raises(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        anderson(cos_map(A_CONTRACT), **kw)  # type: ignore[arg-type]
