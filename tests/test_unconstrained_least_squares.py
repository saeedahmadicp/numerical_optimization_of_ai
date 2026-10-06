"""Tests for Gauss–Newton and Levenberg–Marquardt (numopt.unconstrained.least_squares).

Oracles: scipy.optimize.least_squares(method="lm") (MINPACK lmder) on the problem library,
numpy.linalg.lstsq on linear least squares, the listed (Newton-refined) minimizers, and
independent re-computations of the gain ratio, the damped normal equations and Nielsen's
damping update from the trace.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose
from scipy.optimize import least_squares

import numopt
from numopt import problems

METHODS = ("gauss_newton", "levenberg_marquardt")
LS = ("exp_decay_fit", "rosenbrock_ls", "circle_fit", "michaelis_menten")
INFO_KEYS = {"step", "lambda", "gain_ratio", "residual_norm", "jtj_cond", "accepted"}


def _val(v: float | None) -> float:
    """Narrow an optional Result/Step value to float (the methods always set it)."""
    assert v is not None
    return v


def _f(p, x):
    r = p.residual(np.asarray(x))
    return 0.5 * float(r @ r)


def _accuracy_bound(p, x_star, *, gtol: float = 1e-8, xtol: float = 1e-10) -> float:
    """Attainable ‖x − x*‖∞ for a method that accepts steps by comparing values of f.

    Comparisons of f resolve decreases down to its rounding noise, ≈ ε·Σ|rᵢ|(|rᵢ| + |modelᵢ|)
    ≤ 100·ε·f* for these data (|model|/|r| ≲ 50), so x is pinned only to within
    √(2·100 ε f*/λ_min) — the classic √ε accuracy of f-based tests. Add the gtol term
    gtol/λ_min and the step-test term xtol·‖x*‖, each with a factor 2 for the final step.
    """
    lam_min = float(np.linalg.eigvalsh(p.hess(np.asarray(x_star))).min())
    f_star = _f(p, x_star)
    eps = float(np.finfo(float).eps)
    return (
        math.sqrt(2 * 100 * eps * f_star / lam_min)
        + 2 * gtol / lam_min
        + 2 * xtol * float(np.linalg.norm(x_star))
    )


# --------------------------------------------------------------------------------------
# Convergence on the problem library and the SciPy oracle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", LS)
@pytest.mark.parametrize("tols", [{}, {"gtol": 1e-10, "xtol": 1e-14}])
def test_converges_to_listed_minimum(method, pid, tols):
    p = problems.get(pid)
    res = numopt.run(method, p, **tols)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    bound = _accuracy_bound(p, p.minima[0], **tols)
    assert np.abs(res.x - np.array(p.minima[0])).max() <= bound
    # f − f* ≈ ½ eᵀHe: second order in the error, so 1e-12 relative is ample.
    assert math.isclose(_val(res.fun), p.extra["f_min"], rel_tol=1e-12, abs_tol=1e-15)
    assert res.n_hev == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", LS)
def test_matches_scipy_lm(method, pid):
    p = problems.get(pid)
    res = numopt.run(method, p)
    sol = least_squares(
        p.residual, p.x0, jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    assert res.converged and sol.success
    # NOTE: rtol 1e-7 — MINPACK's ftol stop leaves it ~1e-9 relative short of x* (5e-9 on K for
    # michaelis_menten, κ(JᵀJ) ≈ 1.7e6), and our defaults (gtol 1e-8) allow ≈ 4e-9 on K.
    assert_allclose(res.x, sol.x, rtol=1e-7)
    assert math.isclose(_val(res.fun), sol.cost, rel_tol=1e-12, abs_tol=1e-15)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(METHODS), st.floats(-0.5, 0.5), st.floats(-0.5, 0.5))
def test_matches_scipy_lm_from_random_starts_exp_decay(method, da, db):
    p = problems.get("exp_decay_fit")
    x0 = np.array(p.minima[0]) * (1 + np.array([da, db]))
    res = numopt.run(method, p, x0=x0, gtol=1e-10)
    sol = least_squares(p.residual, x0, jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    assert res.converged, res.message
    assert_allclose(res.x, sol.x, rtol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_circle_fit_mirror_local_minimum(method):
    """From above the arc both methods find the worse mirror centre and say so honestly."""
    p = problems.get("circle_fit")
    res = numopt.run(method, p, x0=[1.0, 4.0], gtol=1e-10)
    assert res.converged, res.message
    assert np.abs(res.x - np.array(p.minima[1])).max() <= _accuracy_bound(
        p, p.minima[1], gtol=1e-10
    )
    assert res.fun > p.extra["f_min"]


def test_michaelis_menten_bates_watts_values():
    res = numopt.run("levenberg_marquardt", problems.get("michaelis_menten"))
    assert res.converged
    assert round(res.x[0], 2) == 212.68 and round(res.x[1], 5) == 0.06412
    assert round(2 * _val(res.fun), 2) == 1195.45


def test_gauss_newton_zero_residual_square_system_is_newton():
    """rosenbrock_ls: r₂ = 1 − x is linear, so one full GN step gives x = 1, the next y = 1."""
    p = problems.get("rosenbrock_ls")
    res = numopt.run("gauss_newton", p, line_search="none")
    assert res.converged and res.n_iter == 2
    # The SVD solve is backward stable: |error in p| ≲ κ(J)·ε·‖p‖ (Higham (2002), §20.1).
    J0 = p.jac(np.array(p.x0))
    step = np.linalg.norm(res.trace[1].info["step"])
    eps = float(np.finfo(float).eps)
    assert abs(res.trace[1].x[0] - 1.0) <= 4 * np.linalg.cond(J0) * eps * step
    assert_allclose(res.x, [1.0, 1.0], rtol=0, atol=1e-14)


# --------------------------------------------------------------------------------------
# Linear least squares (oracle: numpy.linalg.lstsq)
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(2, 4).flatmap(
        lambda n: st.tuples(
            hnp.arrays(np.float64, (n + 3, n), elements=st.floats(-3, 3)),
            hnp.arrays(np.float64, (n + 3,), elements=st.floats(-3, 3)),
        )
    ),
    st.sampled_from(METHODS),
)
def test_linear_least_squares_matches_lstsq(Ab, method):
    A, b = Ab
    s = np.linalg.svd(A, compute_uv=False)
    # κ(A) ≤ 1e3 (so κ(AᵀA) ≤ 1e6) and σ_min ≥ 1e-3: gtol is an absolute test, so for a
    # tiny-scaled A (Hypothesis found entries ≈ 2e-308) ‖Aᵀ(Ax − b)‖ ≤ gtol already holds at x0.
    assume(s[-1] >= 1e-3 and s[-1] > 1e-3 * s[0])
    x_star = np.linalg.lstsq(A, b, rcond=None)[0]
    res = numopt.run(method, lambda x: A @ x - b, x0=np.zeros(A.shape[1]), gtol=1e-10)
    assert res.converged, res.message
    if method == "gauss_newton":
        assert res.n_iter <= 2  # one exact step, then possibly one rounding-level step
    # Error bound: κ(A)·ε·‖x*‖ for the solve, gtol/λ_min from the stopping test, and (LM
    # accepts steps by comparing f) the f-resolution term √(2·100 ε f*/λ_min), λ_min = σ_min².
    eps = float(np.finfo(float).eps)
    lam_min = s[-1] ** 2
    f_star = 0.5 * float((A @ x_star - b) @ (A @ x_star - b))
    tol = (
        10 * (s[0] / s[-1]) * eps * (1 + np.abs(x_star).max())
        + 2e-10 / lam_min
        + math.sqrt(2 * 100 * eps * f_star / lam_min)
        + 2e-10 * np.linalg.norm(x_star)
    )
    assert_allclose(res.x, x_star, rtol=0, atol=tol)
    assert_valid_result(res)


# --------------------------------------------------------------------------------------
# Properties of the iteration (monotone decrease, gain ratio, damping update, counts)
# --------------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(METHODS), st.sampled_from(LS), st.floats(0.05, 0.95), st.floats(0.05, 0.95))
def test_objective_never_increases(method, pid, u, v):
    p = problems.get(pid)
    (a_lo, a_hi), (b_lo, b_hi) = p.domain
    x0 = [a_lo + u * (a_hi - a_lo), b_lo + v * (b_hi - b_lo)]
    res = numopt.run(method, p, x0=x0, max_iter=60)
    assert_valid_result(res, max_iter=60)
    fs = [_val(s.fun) for s in res.trace]
    for prev, cur in pairwise(fs):
        assert cur <= prev
    if method == "gauss_newton":
        # Armijo: f_k ≤ f_{k−1} + c₁ α ∇f_{k−1}ᵀ p_{k−1}.
        for prev, cur in pairwise(res.trace):
            slope = float(p.grad(prev.x) @ np.array(cur.info["step"]))
            f0 = _val(prev.fun)
            assert _val(cur.fun) <= f0 + 1e-4 * cur.info["alpha"] * slope + 1e-15 * f0


@pytest.mark.parametrize("pid", LS)
def test_levenberg_marquardt_trace_reproduces_nielsen_update(pid):
    p = problems.get(pid)
    res = numopt.run("levenberg_marquardt", p, tau=1.0)
    tr = res.trace
    J0 = p.jac(np.array(p.x0))
    assert math.isclose(tr[0].info["lambda"], float(np.max(np.diag(J0.T @ J0))), rel_tol=1e-14)
    assert tr[0].info["nu"] == 2.0
    for prev, cur in pairwise(tr):
        info = cur.info
        x = np.asarray(prev.x)
        h = np.array(info["step"])
        mu = info["lambda"]
        # (1) The step solves the damped normal equations (oracle: a dense solve of JᵀJ + μI).
        J = p.jac(x)
        g = J.T @ p.residual(x)
        h_ref = np.linalg.solve(J.T @ J + mu * np.eye(2), -g)
        assert_allclose(h, h_ref, rtol=1e-8, atol=1e-14 * (1 + np.abs(h_ref).max()))
        # (2) Gain ratio against the model L(h) = ½‖r + Jh‖²: L(0) − L(h) = −gᵀh − ½‖Jh‖²,
        # expanded so that ½‖r‖² cancels exactly (the subtraction ½‖r‖² − ½‖r + Jh‖² loses
        # ~log10(f/pred) digits). The method uses MNT eq. 3.14 instead, which relies on the
        # LM equation; the two agree only if h solves it.
        r = p.residual(x)
        Jh = J @ h
        pred = -float(g @ h) - 0.5 * float(Jh @ Jh)
        rho = (_f(p, x) - _f(p, x + h)) / pred
        # The two predictions differ by ½hᵀ[(JᵀJ + μI)h + g], i.e. by how well the computed h
        # solves the LM equation. The SVD solve leaves a residual ≈ ε(‖J‖‖r‖ + (‖J‖² + μ)‖h‖),
        # which near the solution (‖g‖ ≪ ‖J‖‖r‖) is large relative to pred ≈ ‖g‖‖h‖.
        eps = float(np.finfo(float).eps)
        nJ, nh = float(np.linalg.norm(J, 2)), float(np.linalg.norm(h))
        d_pred = 10 * eps * (nJ * float(np.linalg.norm(r)) + (nJ**2 + mu) * nh) * nh
        assert abs(info["gain_ratio"] - rho) <= abs(rho) * d_pred / pred + 1e-12
        # (3) Acceptance and Nielsen's update (MNT Alg. 3.16).
        assert info["accepted"] == (info["gain_ratio"] > 0)
        nxt = tr[cur.k + 1].info["lambda"] if cur.k + 1 < len(tr) else None
        if info["accepted"]:
            assert_allclose(cur.x, x + h, rtol=0, atol=0)
            assert info["nu"] == 2.0
            expect = mu * max(1 / 3, 1 - (2 * info["gain_ratio"] - 1) ** 3)
        else:
            assert np.array_equal(cur.x, prev.x) and cur.fun == prev.fun
            assert info["nu"] == 2 * prev.info["nu"]
            expect = mu * prev.info["nu"]
        if nxt is not None:
            assert math.isclose(nxt, expect, rel_tol=1e-15)


def test_levenberg_marquardt_damping_limits():
    """Small τ: first step ≈ Gauss–Newton step (rel. diff ≤ τκ); large τ: ≈ −g/μ."""
    p = problems.get("exp_decay_fit")
    x0 = np.array(p.x0)
    J, r = p.jac(x0), p.residual(x0)
    p_gn = np.linalg.lstsq(J, -r, rcond=None)[0]
    tau = 1e-8
    small = numopt.run("levenberg_marquardt", p, tau=tau, max_iter=1).trace[1].info
    kappa = float(np.linalg.cond(J) ** 2)
    h = np.array(small["step"])
    assert np.linalg.norm(h - p_gn) <= 2 * tau * kappa * np.linalg.norm(p_gn)
    big = numopt.run("levenberg_marquardt", p, tau=100.0, max_iter=1).trace[1].info
    g = J.T @ r
    h = np.array(big["step"])
    cos = float(-g @ h) / (np.linalg.norm(g) * np.linalg.norm(h))
    assert cos > 1 - 1e-4


@pytest.mark.parametrize("pid", LS)
def test_evaluation_counts_are_exact(pid):
    p = problems.get(pid)
    gn = numopt.run("gauss_newton", p)
    assert gn.n_gev == 1 + gn.n_iter
    assert gn.n_fev == 1 + sum(len(s.info["trials"]) for s in gn.trace)
    lm = numopt.run("levenberg_marquardt", p)
    assert lm.n_fev == 1 + lm.n_iter
    assert lm.n_gev == 1 + sum(bool(s.info["accepted"]) for s in lm.trace)


def test_finite_difference_jacobian_for_bare_residual():
    p = problems.get("exp_decay_fit")
    res = numopt.run("levenberg_marquardt", p.residual, x0=p.x0, gtol=1e-9)
    assert res.converged
    assert_allclose(res.x, p.minima[0], rtol=1e-7)
    n_trials = res.n_iter
    # Each FD Jacobian costs 2n + 1 = 5 residual evaluations (core.diff.jacobian).
    assert res.n_fev == 1 + n_trials + 5 * res.n_gev


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", LS)
def test_trace_info_contract(method, pid):
    p = problems.get(pid)
    res = numopt.run(method, p)
    for s in res.trace:
        assert INFO_KEYS <= set(s.info)
        assert math.isclose(s.info["residual_norm"], math.sqrt(2 * _val(s.fun)), rel_tol=1e-14)
        assert s.info["jtj_cond"] >= 1.0
        assert math.isclose(
            _val(s.grad_norm), float(np.linalg.norm(p.grad(np.asarray(s.x)))), rel_tol=1e-10
        )
        if method == "gauss_newton":
            assert s.info["lambda"] == 0.0
            assert {"alpha", "trials"} <= set(s.info)
        else:
            assert "nu" in s.info
    first = res.trace[0].info
    assert first["step"] is None and first["gain_ratio"] is None and first["accepted"] is None
    assert len(res.trace) == res.n_iter + 1 and res.trace[-1].k == res.n_iter


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_reported(method):
    res = numopt.run(method, problems.get("michaelis_menten"), max_iter=1)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 1
    assert_valid_result(res, max_iter=1)


def test_gauss_newton_rank_deficient_fails_lm_succeeds():
    def r(x):
        return np.array([x[0] + x[1] - 2.0])  # m = 1 < n = 2: J has rank 1

    gn = numopt.run("gauss_newton", r, x0=[0.0, 0.0])
    assert not gn.converged and "rank-deficient" in gn.message
    assert gn.n_iter == 0
    assert_valid_result(gn)
    lm = numopt.run("levenberg_marquardt", r, x0=[0.0, 0.0])
    assert lm.converged
    assert abs(lm.x[0] + lm.x[1] - 2.0) < 1e-8
    assert_valid_result(lm)


def test_non_finite_trial_points():
    """r = (log x, y) from (5, 1): the full GN step lands at x < 0 (log undefined)."""

    def r(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([np.log(x[0]), x[1]])

    def jac(x):
        return np.array([[1.0 / x[0], 0.0], [0.0, 1.0]])

    prob = numopt.Problem(
        id="log_test", name="log", latex="", f=lambda x: 0.5 * float(r(x) @ r(x)), dim=2, domain=(),
        residual=r, jac=jac, x0=[5.0, 1.0],
    )  # fmt: skip
    full = numopt.run("gauss_newton", prob, line_search="none")
    assert not full.converged and "not finite" in full.message
    assert_valid_result(full)
    damped = numopt.run("gauss_newton", prob)
    assert damped.converged and np.allclose(damped.x, [1.0, 0.0], atol=1e-9)
    assert damped.trace[1].info["alpha"] < 1.0
    lm = numopt.run("levenberg_marquardt", prob)
    assert lm.converged and np.allclose(lm.x, [1.0, 0.0], atol=1e-8)


def test_levenberg_marquardt_does_not_claim_convergence_when_blocked():
    def r(x):
        return np.array([x[0] - 3.0]) if x[0] == 1.0 else np.array([np.nan])

    res = numopt.run("levenberg_marquardt", r, x0=[1.0])
    assert not res.converged and "not finite" in res.message
    assert all(s.info["accepted"] is False for s in res.trace[1:])
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_start(method):
    res = numopt.run(method, lambda x: np.array([np.nan, 1.0]), x0=[0.0])
    assert not res.converged and res.n_iter == 0 and "x0" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_invalid_inputs(method):
    with pytest.raises(ValueError, match="residual"):
        numopt.run(method, problems.get("rosenbrock"))
    with pytest.raises(ValueError, match="x0"):
        numopt.run(method, lambda x: x)
    with pytest.raises(TypeError):
        numopt.run(method, problems.get("exp_decay_fit"), bogus=1)
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("exp_decay_fit"), x0=[1.0, 2.0, 3.0])


def test_invalid_method_parameters():
    with pytest.raises(ValueError):
        numopt.run("gauss_newton", problems.get("exp_decay_fit"), line_search="wolfe")
    with pytest.raises(ValueError):
        numopt.run("levenberg_marquardt", problems.get("exp_decay_fit"), tau=0.0)


def test_fixture_cases_run_and_converge():
    from numopt.unconstrained.least_squares import FIXTURE_CASES

    assert {m for m, _, _ in FIXTURE_CASES} == set(METHODS)
    for method, pid, params in FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert res.converged and len(res.trace) < 300
        assert_valid_result(res)


def test_gauss_newton_ftol_stop_and_rounding_stall():
    """On michaelis_menten f ≈ 598 resolves decreases only to ~ε·f, so ‖g‖∞ stalls near 1e-4."""
    p = problems.get("michaelis_menten")
    res = numopt.run("gauss_newton", p, gtol=1e-14, xtol=1e-15)
    assert res.converged and "ftol" in res.message
    assert np.abs(res.x - np.array(p.minima[0])).max() <= _accuracy_bound(p, p.minima[0])
    # With ftol below the rounding floor the line search reports the stall honestly.
    stall = numopt.run("gauss_newton", p, gtol=1e-14, xtol=1e-15, ftol=1e-18)
    assert not stall.converged and "no measurable decrease" in stall.message
    assert_valid_result(stall)
    assert np.abs(stall.x - np.array(p.minima[0])).max() <= _accuracy_bound(p, p.minima[0])


@pytest.mark.parametrize(
    "pid, x0",
    [
        ("circle_fit", [0.46573169294277905, -1.2038309048861635]),
        ("michaelis_menten", [245.51758097738417, 0.0794776761058022]),
    ],
)
def test_gauss_newton_default_ftol_is_above_the_rounding_noise_of_f(pid, x0):
    """Regression: with the old default ftol = 1e-15 ≈ 4.5ε these starts reached x* with
    ‖Jᵀr‖∞ just above gtol and ‖Jp‖²/‖r‖² ≈ 2e-15 (below the noise of f), and the Armijo search
    ended the run with converged=False ('no measurable decrease')."""
    from numopt.core.registry import get_method

    p = problems.get(pid)
    spec = {s.name: s.default for s in get_method("gauss_newton").params}
    assert spec["ftol"] == 1e-14
    res = numopt.run("gauss_newton", p, x0=x0)
    assert_valid_result(res)
    assert res.converged and "ftol" in res.message, res.message
    x_star = p.minima[0]
    assert np.abs(res.x - np.array(x_star)).max() <= _accuracy_bound(p, x_star)
    # Oracle: MINPACK restarted at the result does not lower f beyond the rounding noise.
    sol = least_squares(
        p.residual, res.x, jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    eps = float(np.finfo(float).eps)
    assert sol.cost >= _val(res.fun) * (1 - 1e3 * eps)


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sampled_from(("exp_decay_fit", "circle_fit", "michaelis_menten")),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_gauss_newton_converged_flag_from_domain_starts(pid, u, v):
    """Default Gauss–Newton (backtracking) from any start in the plotting box either converges
    to within the attainable accuracy of a listed minimum, or fails for a documented reason
    (rank-deficient J or max_iter). A rounding stall is not acceptable at the defaults."""
    p = problems.get(pid)
    (a_lo, a_hi), (b_lo, b_hi) = p.domain
    x0 = [a_lo + u * (a_hi - a_lo), b_lo + v * (b_hi - b_lo)]
    res = numopt.run("gauss_newton", p, x0=x0)
    assert_valid_result(res)
    if res.converged:
        dist = [np.abs(res.x - np.asarray(m)).max() / _accuracy_bound(p, m) for m in p.minima]
        assert min(dist) <= 1.0, (res.message, res.x)
    else:
        assert "rank-deficient" in res.message or "max_iter" in res.message, res.message


def _scalar_problem(r, jac, x0) -> numopt.Problem:
    return numopt.Problem(
        id="scalar_ls", name="scalar", latex="", f=lambda x: 0.5 * float(r(x) @ r(x)), dim=len(x0),
        domain=(), residual=r, jac=jac, x0=list(x0),
    )  # fmt: skip


FAILED_START_KEYS = {
    "gauss_newton": INFO_KEYS | {"alpha", "trials"},
    "levenberg_marquardt": INFO_KEYS | {"nu"},
}


@pytest.mark.parametrize(
    "method, kw",
    [
        ("gauss_newton", {"ftol": 0.0}),
        ("gauss_newton", {"ftol": 0.0, "line_search": "none"}),
        ("gauss_newton", {}),
        ("levenberg_marquardt", {}),
    ],
)
def test_f_underflow_to_zero_is_a_global_minimum_not_a_crash(method, kw):
    """r = x² from x = 1 with gtol = xtol = 0: f = ½x⁴ underflows to 0 while g = 2x³ ≠ 0.

    Regression: the ftol ratio ‖Jp‖²/(2f) raised ZeroDivisionError there.
    """
    prob = _scalar_problem(lambda x: np.array([x[0] ** 2]), lambda x: np.array([[2 * x[0]]]), [1.0])
    res = numopt.run(method, prob, gtol=0.0, xtol=0.0, max_iter=3000, **kw)
    assert_valid_result(res, max_iter=3000)
    assert res.converged and res.fun == 0.0 and "= 0 in floating point" in res.message
    x = float(res.x[0])
    assert x != 0.0 and 0.5 * (x * x) ** 2 == 0.0  # f underflowed, r = x² did not vanish
    assert all(_val(s.fun) > 0.0 for s in res.trace[:-1])  # the stop fires at the first f = 0


@settings(max_examples=1000, deadline=None)
@given(
    hnp.arrays(np.float64, (5, 2), elements=st.floats(-10, 10)),
    hnp.arrays(np.float64, (5,), elements=st.floats(-10, 10)),
    st.integers(-1000, 1000),
)
def test_gauss_newton_ratio_is_scale_free_and_matches_projection(J, r, e):
    """‖Jp‖²/‖r‖² = ‖P r‖²/‖r‖², P the orthogonal projector onto range J (oracle: lstsq).

    Scaling r by 2ᵉ is exact in binary floating point, so the ratio must be bit-identical
    even where ½‖r‖² underflows to 0 or overflows to inf (|e| up to 1000).
    """
    from numopt.unconstrained.least_squares import _linearize

    s = np.linalg.svd(J, compute_uv=False)
    # σ_min ≥ 1e-3 keeps the oracle's p = −J⁺r representable: Hypothesis found J with
    # subnormal entries (5e-324) where lstsq returns p = inf although the ratio is exactly 1.
    assume(s[-1] > 1e-6 * s[0] and s[-1] >= 1e-3 and np.abs(r).max() > 1e-3)
    ratio = _linearize(r, J).gauss_newton_ratio()
    p = np.linalg.lstsq(J, -r, rcond=None)[0]
    Jp = J @ p
    # Both sides are backward stable; the projector loses ≈ κ(J)·ε relative.
    assert ratio == pytest.approx(float(Jp @ Jp) / float(r @ r), rel=1e-9, abs=1e-12)
    r_scaled = np.ldexp(r, e)
    # The scaling is exact only while no entry becomes subnormal (or inf): Hypothesis found
    # r ∋ 1.8e-127 with e = −601, where 2ᵉ·r ≈ 4e-308 < 2.2e-308 loses bits.
    assume(np.array_equal(np.ldexp(r_scaled, -e), r))
    scaled = _linearize(r_scaled, J).gauss_newton_ratio()
    assert scaled == ratio


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "x0",
    [[1.0, -85.0], [1.0, -100.0]],  # max|r| ≈ 1.1e155 and 1e182: f = ½‖r‖² = inf
)
def test_f_overflow_at_start_is_reported(method, x0):
    """Regression: LM returned converged=True at x0 with f = inf (μ₀ = inf ⇒ h = 0)."""
    p = problems.get("exp_decay_fit")
    res = numopt.run(method, p, x0=x0)
    assert not res.converged and res.n_iter == 0
    assert "f = ½‖r‖² overflows at x0" in res.message
    assert res.fun == math.inf
    assert set(res.trace[0].info) == FAILED_START_KEYS[method]
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_residual_scaled_to_overflow_is_reported(method):
    p = problems.get("exp_decay_fit")
    res = numopt.run(method, lambda x: 1e200 * p.residual(x), x0=p.x0)
    assert not res.converged and "overflows at x0" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_gradient_overflow_at_start_is_reported(method):
    """f = ½·1e300 is finite, but g = Jᵀr = 1e310 is not."""
    prob = _scalar_problem(
        lambda x: np.array([1e150 + 0 * x[0]]), lambda x: np.array([[1e160]]), [0.0]
    )
    res = numopt.run(method, prob)
    assert not res.converged and "∇f = Jᵀr overflows at x0" in res.message
    assert math.isfinite(_val(res.fun))
    assert set(res.trace[0].info) == FAILED_START_KEYS[method]
    assert_valid_result(res)


def test_levenberg_marquardt_initial_damping_overflow_is_reported():
    """r = 1e160·x from x = 1e-300: f, g finite, μ₀ = 10⁻³·1e320 = inf (h would be 0)."""
    prob = _scalar_problem(lambda x: 1e160 * x, lambda x: np.array([[1e160]]), [1e-300])
    lm = numopt.run("levenberg_marquardt", prob)
    assert not lm.converged and "μ₀" in lm.message and "overflows" in lm.message
    assert lm.trace[0].info["lambda"] == math.inf
    assert_valid_result(lm)
    # The undamped method is unaffected: ‖p‖ = 1e-300 ≤ xtol·(‖x‖ + xtol) at once. The scaled
    # norm reports the true ‖p‖ (np.linalg.norm underflows to 0 there).
    gn = numopt.run("gauss_newton", prob)
    assert gn.converged and "‖h‖ = 1e-300" in gn.message
    with_zero_xtol = numopt.run("gauss_newton", prob, xtol=0.0)
    assert with_zero_xtol.converged and with_zero_xtol.x[0] == 0.0
    assert "gtol" in with_zero_xtol.message


def test_gradient_norm_does_not_overflow_when_g_is_finite():
    """exp_decay_fit at (1, −84): ‖g‖∞ ≈ 4e306 is finite; ‖g‖₂ must not be reported as inf."""
    p = problems.get("exp_decay_fit")
    x0 = np.array([1.0, -84.0])
    g = p.grad(x0)
    assert np.all(np.isfinite(g))
    res = numopt.run("levenberg_marquardt", p, x0=x0, max_iter=1)
    m = np.abs(g).max()
    assert math.isclose(_val(res.trace[0].grad_norm), m * np.linalg.norm(g / m), rel_tol=1e-14)


@pytest.mark.parametrize("x0", [[0.1, -4.0], [1.0, -6.0], [0.001, -4.0]])
def test_levenberg_marquardt_damping_limited_step_is_not_convergence(x0):
    """Regression: from these starts μ₀ ≈ 1e11–1e20 stays ≫ σ_min² ≈ 1e-5 (Nielsen lowers μ by
    at most 3× per step), so ‖h‖ ≈ ‖g‖/μ < xtol·‖x‖ while ‖g‖∞ ≈ 1e2–5e3 and f ≈ 5.65 ≫
    f* = 0.0071. The old code reported converged=True there."""
    p = problems.get("exp_decay_fit")
    res = numopt.run("levenberg_marquardt", p, x0=x0)
    assert_valid_result(res, max_iter=200)
    assert not res.converged
    assert "damping μ" in res.message and "Gauss–Newton model still predicts" in res.message
    # Oracle: MINPACK from our final point still lowers f by a large factor.
    sol = least_squares(p.residual, res.x, jac=p.jac, method="lm")
    assert sol.cost < 1e-2 * _val(res.fun)


LM_TAUS = (1e-3, 1.0, 100.0)


def _start_in_band(pid: str, u: float, v: float) -> list[float]:
    """A start in the plotting domain; for exp_decay_fit the wider band a ∈ [1e-3, 10] (log),
    b ∈ [−6, 3], where the old damping-limited stop failed on 364 of 1500 starts."""
    if pid == "exp_decay_fit":
        return [10.0 ** (-3.0 + 4.0 * u), -6.0 + 9.0 * v]
    (a_lo, a_hi), (b_lo, b_hi) = problems.get(pid).domain
    return [a_lo + u * (a_hi - a_lo), b_lo + v * (b_hi - b_lo)]


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sampled_from(("exp_decay_fit", "michaelis_menten", "circle_fit")),
    st.sampled_from(LM_TAUS),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_levenberg_marquardt_converged_means_scipy_cannot_improve(pid, tau, u, v):
    """Whenever LM reports converged=True, MINPACK restarted at the result does not lower f
    by more than the rounding noise of f.

    NOTE: rel. tolerance 1e3·ε — the rounding stop certifies f − f* ≲ 100·ε·f, and MINPACK's
    own f carries noise of the same order (worst observed over 1350 starts: 1.0e-14 ≈ 46ε).
    The audited √ε stop left f − f* = 1.2e-8·f on michaelis_menten with τ = 1."""
    p = problems.get(pid)
    res = numopt.run("levenberg_marquardt", p, x0=_start_in_band(pid, u, v), tau=tau)
    assert_valid_result(res, max_iter=200)
    if res.converged:
        sol = least_squares(
            p.residual, res.x, jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
        eps = float(np.finfo(float).eps)
        assert sol.cost >= _val(res.fun) * (1 - 1e3 * eps) - 1e-300
    else:
        assert "damping μ" in res.message or "max_iter" in res.message


def _soft_direction(p) -> np.ndarray:
    """Unit eigenvector of ∇²f(x*) for λ_min: the direction in which f resolves x worst."""
    _, V = np.linalg.eigh(p.hess(np.asarray(p.minima[0])))
    return V[:, 0]


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sampled_from(LS),
    st.sampled_from(LM_TAUS),
    st.floats(-6.0, -1.0),
    st.sampled_from((1.0, -1.0)),
)
def test_levenberg_marquardt_converged_means_within_accuracy_bound(pid, tau, log_d, sign):
    """Starts x* + d·v along the soft eigenvector, d = 10^log_d·max(1, ‖x*‖∞): a converged
    run must be within the attainable accuracy of x*; otherwise the damping blocked it."""
    p = problems.get(pid)
    x_star = np.asarray(p.minima[0])
    d = sign * 10.0**log_d * max(1.0, float(np.abs(x_star).max()))
    res = numopt.run("levenberg_marquardt", p, x0=x_star + d * _soft_direction(p), tau=tau)
    assert_valid_result(res, max_iter=200)
    if res.converged:
        assert np.abs(res.x - x_star).max() <= _accuracy_bound(p, x_star)
    else:
        assert "damping μ" in res.message and "Gauss–Newton model still predicts" in res.message


def test_levenberg_marquardt_large_damping_is_not_a_rounding_stop():
    """Regression: the 'unresolvable decrease' threshold was √ε ≈ 1.5e-8, so with τ ≥ 1 LM
    stopped at k = 0 with converged=True where comparisons of f still resolve the decrease."""
    p = problems.get("michaelis_menten")
    x_star = np.asarray(p.minima[0])
    x0 = x_star + 2.5e-3 * _soft_direction(p)
    bound = _accuracy_bound(p, x_star)
    assert np.abs(x0 - x_star).max() > 700 * bound  # the old stop was 730× outside the bound
    # τ = 1: the damped steps predict resolvable decreases, so LM continues and converges.
    res = numopt.run("levenberg_marquardt", p, x0=x0, tau=1.0)
    assert res.converged and res.n_iter > 0, res.message
    assert np.abs(res.x - x_star).max() <= bound
    # τ = 100: the damped step predicts a decrease below the noise of f, it is rejected, and
    # LM says that the damping (not convergence) ended the run.
    big = numopt.run("levenberg_marquardt", p, x0=x0, tau=100.0)
    assert not big.converged and "damping μ" in big.message
    assert_valid_result(big)

    # Linear LS r = (1e5(x₁ − 1), x₂ − 1, 1e3) from (1, 1.1): f(x0) − f* = 5e-3 ≫ ε·f = 1.1e-10.
    def r(x):
        return np.array([1e5 * (x[0] - 1.0), x[1] - 1.0, 1e3])

    A, b = np.array([[1e5, 0.0], [0.0, 1.0], [0.0, 0.0]]), np.array([1e5, 1.0, -1e3])
    x_ls = np.linalg.lstsq(A, b, rcond=None)[0]  # oracle: (1, 1)
    gn = numopt.run("gauss_newton", r, x0=[1.0, 1.1])
    assert gn.converged and np.abs(gn.x - x_ls).max() <= 1e-12
    for tau in (1.0, 100.0):
        lm = numopt.run("levenberg_marquardt", r, x0=[1.0, 1.1], tau=tau)
        assert not lm.converged and "damping μ" in lm.message, lm.message
        assert_valid_result(lm)


@pytest.mark.parametrize(
    "method, params",
    [
        ("gauss_newton", {}),
        ("gauss_newton", {"gtol": 1e-14, "xtol": 1e-15}),  # ftol stop on michaelis_menten
        ("gauss_newton", {"xtol": 1e-6}),
        ("levenberg_marquardt", {}),
        ("levenberg_marquardt", {"tau": 1.0}),
        ("levenberg_marquardt", {"xtol": 1e-6}),
    ],
)
@pytest.mark.parametrize("pid", LS)
def test_capped_run_reports_the_same_stop_as_uncapped(method, params, pid):
    """Regression: the step and ftol tests were not re-run after the last iteration, so
    max_iter = n_iter of a converged run gave converged=False at the same iterate."""
    p = problems.get(pid)
    free = numopt.run(method, p, **params)
    assert free.n_iter >= 1
    capped = numopt.run(method, p, max_iter=free.n_iter, **params)
    assert (capped.converged, capped.message) == (free.converged, free.message)
    assert capped.n_iter == free.n_iter and np.array_equal(capped.x, free.x)
    # The final tests cost no evaluations.
    assert (capped.n_fev, capped.n_gev) == (free.n_fev, free.n_gev)
    assert_valid_result(capped, max_iter=free.n_iter)
    if free.n_iter >= 2:
        short = numopt.run(method, p, max_iter=free.n_iter - 1, **params)
        assert not short.converged and "max_iter" in short.message


@pytest.mark.parametrize("pid", ["exp_decay_fit", "circle_fit"])
def test_gain_ratio_none_only_in_documented_cases(pid):
    """gain_ratio is None at k ≥ 1 only when f(x + h) is not finite or L(0) − L(h) ≤ 0. With
    gtol = xtol = 0, μ grows to ≈ 1e307, h becomes subnormal and L(0) − L(h) underflows."""
    p = problems.get(pid)
    res = numopt.run("levenberg_marquardt", p, gtol=0.0, xtol=0.0, max_iter=2000)
    assert not res.converged and "μ overflowed" in res.message
    assert_valid_result(res, max_iter=2000)
    nones = 0
    for prev, cur in pairwise(res.trace):
        if cur.info["gain_ratio"] is not None:
            continue
        nones += 1
        x, h, mu = np.asarray(prev.x), np.array(cur.info["step"]), cur.info["lambda"]
        g = p.grad(x)
        predicted = 0.5 * float(h @ (mu * h - g))  # MNT eq. 3.14
        assert predicted <= 0.0 or not math.isfinite(_f(p, x + h))
        assert math.isfinite(_f(p, x + h))  # here the finite-f case of the documentation
        assert cur.info["accepted"] is False
    assert nones >= 1


@pytest.mark.parametrize("method", METHODS)
def test_step_test_is_normwise_relative(method):
    """Documented limitation of the MNT step test: r = (10³·x₁, x₂ − 10¹²) from (1, 10¹²).

    ‖x − x*‖ = 1 ≤ xtol·‖x*‖ = 100, so the normwise test passes at once although f = ½·10⁶;
    the message reports ‖Jᵀr‖∞ = 10⁶ so the user sees it.
    """
    prob = _scalar_problem(
        lambda x: np.array([1e3 * x[0], x[1] - 1e12]),
        lambda x: np.array([[1e3, 0.0], [0.0, 1.0]]),
        [1.0, 1e12],
    )
    res = numopt.run(method, prob)
    assert res.converged and res.n_iter == 0 and "xtol" in res.message
    assert "‖Jᵀr‖∞ = 1e+06" in res.message
    assert np.linalg.norm(res.x - np.array([0.0, 1e12])) <= 1e-10 * 1e12


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "residual, x0",
    [
        (lambda x: np.array([np.nan, 1.0]), [0.0]),  # r not finite
        (lambda x: np.array([np.log(x[0] - 1.0)]), [0.0]),  # r = nan, J (FD) not reached
    ],
)
def test_failed_start_info_has_the_documented_key_set(method, residual, x0):
    with np.errstate(invalid="ignore"):
        res = numopt.run(method, residual, x0=x0)
    assert not res.converged and res.n_iter == 0
    info = res.trace[0].info
    assert set(info) == FAILED_START_KEYS[method]
    assert info["step"] is None and info["gain_ratio"] is None and info["accepted"] is None
    assert info["jtj_cond"] is None  # J was never evaluated
    if method == "gauss_newton":
        assert info["lambda"] == 0.0 and info["alpha"] is None and info["trials"] == []
    else:
        assert info["lambda"] is None and info["nu"] == 2.0
    assert_valid_result(res)


def test_jtj_cond_is_inf_when_rank_deficient():
    def r(x):
        return np.array([x[0] + x[1] - 2.0])  # m = 1 < n = 2

    gn = numopt.run("gauss_newton", r, x0=[0.0, 0.0])
    assert gn.trace[0].info["jtj_cond"] == math.inf and "σ_min/σ_max" in gn.message
    lm = numopt.run("levenberg_marquardt", r, x0=[0.0, 0.0])
    assert all(s.info["jtj_cond"] == math.inf for s in lm.trace)
    # Numerically rank-deficient but with σ_min > 0: still inf, not a finite 1e51.
    p = problems.get("exp_decay_fit")
    gn = numopt.run("gauss_newton", p, x0=[1.0, -84.0])
    assert not gn.converged and "rank-deficient" in gn.message
    assert gn.trace[-1].info["jtj_cond"] == math.inf


def test_levenberg_marquardt_order_is_superlinear_not_quadratic():
    """The registry says 'superlinear for zero residual': on rosenbrock_ls the error ratios
    e_{k+1}/e_k fall toward 0 (superlinear) but only ≈ 3× per step, tracking μ_k, so the order
    estimates log(e_{k+1}/e_k)/log(e_k/e_{k−1}) stay well below 2."""
    from numopt.core.registry import list_methods

    spec = next(s for s in list_methods() if s.id == "levenberg_marquardt")
    assert spec.order.startswith("superlinear for zero residual")
    res = numopt.run("levenberg_marquardt", problems.get("rosenbrock_ls"), gtol=1e-15, xtol=1e-16)
    assert res.converged
    acc = [s for s in res.trace if s.info["accepted"] in (None, True)]
    e = [float(np.linalg.norm(np.asarray(s.x) - 1.0)) for s in acc]
    e = [v for v in e if v > 0.0][-7:]
    ratios = [b / a for a, b in pairwise(e)]
    assert all(b < a for a, b in pairwise(ratios)) and ratios[-1] < 1e-2  # superlinear
    q = [math.log(ratios[i + 1]) / math.log(ratios[i]) for i in range(len(ratios) - 1)]
    assert max(q[-3:]) < 1.5  # not quadratic (q → 2)


@pytest.mark.parametrize("line_search", ["backtracking", "none"])
def test_gauss_newton_ftol_never_fires_on_zero_residual(line_search):
    """Zero residual: r ∈ range J, so ‖Jp‖²/‖r‖² = 1 and only gtol can stop the run."""
    res = numopt.run(
        "gauss_newton", problems.get("rosenbrock_ls"), ftol=1e-4, line_search=line_search
    )
    assert res.converged and "gtol" in res.message
