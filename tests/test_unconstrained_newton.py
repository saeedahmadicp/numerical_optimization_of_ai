"""Tests for numopt.unconstrained.newton: pure, damped and modified Newton.

Oracles: SciPy (``scipy.linalg.solve``/``cho_solve`` for the linear algebra, ``minimize`` with
Newton-CG / trust-exact for minimizers and iteration counts), the eigenvalues of the matrix
for N&W Alg. 3.3, closed-form minimizers of quadratics, and known stationary points.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.optimize import minimize as sp_minimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.unconstrained import newton as newton_mod
from numopt.unconstrained.newton import _cholesky_shift, _cholesky_solve, _second_order

METHODS = ("pure_newton", "damped_newton", "modified_newton")
LS_METHODS = ("damped_newton", "modified_newton")
EPS = float(np.finfo(float).eps)


def _nearest_min(problem: Problem, x: Any) -> np.ndarray:
    mins = [np.asarray(m, dtype=float) for m in problem.minima]
    return min(mins, key=lambda m: float(np.linalg.norm(m - np.asarray(x))))


def _quadratic(A: np.ndarray, b: np.ndarray) -> Problem:
    """f(x) = ½xᵀAx − bᵀx with exact derivatives."""
    return Problem(
        id="quad",
        name="quad",
        latex="",
        f=lambda x: 0.5 * float(x @ A @ x) - float(b @ x),
        grad=lambda x: A @ x - b,
        hess=lambda x: A,
        dim=b.size,
        domain=(),
        x0=np.zeros(b.size),
    )


def _poly(f: Any, g: Any, h: Any, x0: Any) -> Problem:
    return Problem(id="t", name="t", latex="", f=f, grad=g, hess=h, dim=2, domain=(), x0=x0)


# --------------------------------------------------------------------------------------
# Convergence on library problems and the SciPy oracle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "pid", ["rosenbrock", "quadratic_bowl", "booth", "quadratic_nd", "three_hump_camel", "matyas"]
)
def test_converges_to_a_known_minimum(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    assert np.max(np.abs(prob.grad(res.x))) <= 1e-8
    x_star = _nearest_min(prob, res.x)
    # f-error ≤ ½λ_max‖x − x*‖² and ‖x − x*‖ ≤ ‖∇f‖/λ_min: gtol = 1e-8 with λ_min ≥ 0.04
    # (matyas) bounds the x error by ~1e-6·√n.
    assert_allclose(res.x, x_star, rtol=0, atol=1e-6)


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize("pid", ["beale", "himmelblau", "goldstein_price", "styblinski_tang"])
def test_line_search_newton_reaches_local_minimizers(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    assert np.all(np.linalg.eigvalsh(prob.hess(res.x)) > 0.0)
    assert "positive definite" in res.message


# Problems whose start lies in the basin of one minimizer for every method compared (on
# himmelblau the methods legitimately reach different local minimizers).
@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "three_hump_camel"])
def test_matches_scipy_minimizer_and_iteration_ballpark(pid: str) -> None:
    prob = problems.get(pid)
    # SciPy's Newton-type references at a tolerance comparable to ours (Newton-CG's default
    # xtol = 1e-5 would make the reference, not the method under test, the inaccurate side).
    ref = sp_minimize(
        prob.f,
        prob.x0,
        jac=prob.grad,
        hess=prob.hess,
        method="trust-exact",
        options={"gtol": 1e-10},
    )
    assert ref.success
    ref_cg = sp_minimize(
        prob.f,
        prob.x0,
        jac=prob.grad,
        hess=prob.hess,
        method="Newton-CG",
        options={"xtol": 1e-12},
    )
    assert_allclose(ref_cg.x, ref.x, rtol=0, atol=1e-8)
    for mid in LS_METHODS:
        res = numopt.run(mid, prob)
        assert res.converged
        assert_allclose(res.x, ref.x, rtol=0, atol=1e-6)
        assert res.n_iter <= 2 * ref.nit + 5


def test_first_newton_step_matches_scipy_solve() -> None:
    prob = problems.get("rosenbrock")
    res = numopt.run("pure_newton", prob)
    x0 = np.asarray(prob.x0, dtype=float)
    p_ref = scipy.linalg.solve(prob.hess(x0), -prob.grad(x0), assume_a="sym")
    # κ(∇²f(x0)) ≈ 1e3, so the spectral solve and LU agree to ~κε.
    assert_allclose(res.trace[1].info["direction"], p_ref, rtol=1e-12)
    assert_allclose(res.trace[1].x, x0 + p_ref, rtol=1e-12)
    assert res.trace[1].info["alpha"] == 1.0
    assert res.trace[1].info["trials"] == []


def test_pure_newton_rosenbrock_is_quadratically_convergent() -> None:
    prob = problems.get("rosenbrock")
    res = numopt.run("pure_newton", prob, gtol=1e-12)
    assert res.converged
    e = [float(np.linalg.norm(np.asarray(s.x) - 1.0)) for s in res.trace]
    # Quadratic rate (N&W Thm 3.5): e_{k+1} ≤ C e_k² near x*, with C ≈ ‖∇²f(x*)⁻¹‖·L/2.
    tail = [(a, b) for a, b in itertools.pairwise(e) if 0.0 < a < 0.1]
    assert len(tail) >= 2
    for a, b in tail:
        assert b <= 10.0 * a * a


@given(
    st.integers(2, 6),
    st.lists(st.floats(-2.0, 2.0), min_size=6, max_size=6),
    arrays(np.float64, (6, 6), elements=st.floats(-1, 1)),
    arrays(np.float64, 6, elements=st.floats(-10, 10)),
)
@settings(max_examples=300, deadline=None)
def test_pure_newton_solves_spd_quadratic_in_one_step(
    n: int, log_eigs: list[float], M: np.ndarray, b: np.ndarray
) -> None:
    Q, _ = np.linalg.qr(M[:n, :n] + 3.0 * np.eye(n))
    lam = 10.0 ** np.asarray(log_eigs[:n])  # eigenvalues in [1e-2, 1e2]: κ ≤ 1e4
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    bb = b[:n]
    assume(float(np.max(np.abs(bb))) > 1e-6)  # x0 = 0 must not already be optimal
    x_star = scipy.linalg.solve(A, bb, assume_a="pos")
    res = numopt.run("pure_newton", _quadratic(A, bb), gtol=1e-7)
    assert res.converged, res.message
    assert res.n_iter == 1
    kappa = lam.max() / lam.min()
    scale = max(1.0, float(np.max(np.abs(x_star))))
    # Backward-stable solves: forward error ≲ n·κ·ε relative to ‖x*‖.
    assert_allclose(res.x, x_star, rtol=0, atol=10 * n * kappa * EPS * scale)


# --------------------------------------------------------------------------------------
# Non-convex behaviour: saddles, maxima, singular Hessians, divergence
# --------------------------------------------------------------------------------------


def test_pure_newton_reports_a_maximizer_on_himmelblau() -> None:
    res = numopt.run("pure_newton", problems.get("himmelblau"))
    assert_valid_result(res)
    assert not res.converged
    # ∇²f ≺ 0 there: a strict local maximizer, not the undecidable singular case.
    assert "stopped at a maximizer" in res.message
    # Himmelblau's local maximum.
    assert_allclose(res.x, [-0.270845, -0.923039], atol=1e-6)


def test_saddle_point_is_not_reported_as_converged() -> None:
    prob = _poly(
        lambda x: float(x[0] ** 2 - x[1] ** 2),
        lambda x: np.array([2 * x[0], -2 * x[1]]),
        lambda x: np.diag([2.0, -2.0]),
        [1.0, 0.5],
    )
    for method in ("pure_newton", "damped_newton"):
        res = numopt.run(method, prob)
        assert_valid_result(res)
        # Newton jumps to the saddle (0, 0); for damped Newton the Newton direction is a
        # descent direction there (gᵀp = −1.5), so it is accepted as well.
        assert not res.converged
        assert "saddle" in res.message
        assert_allclose(res.x, [0.0, 0.0], atol=1e-15)
        assert res.n_iter == 1


_CLASSIFY_CASES = [
    # (f, ∇f, ∇²f, expected phrase): stationary point at 0 with ∇²f(0) = diag(−2, ·).
    (  # ∇²f = diag(−2, 0), and y⁴ makes 0 a saddle point.
        lambda x: float(-(x[0] ** 2) + x[1] ** 4),
        lambda x: np.array([-2.0 * x[0], 4.0 * x[1] ** 3]),
        lambda x: np.diag([-2.0, 12.0 * x[1] ** 2]),
        "saddle point or a maximizer",
    ),
    (  # The same ∇²f(0) = diag(−2, 0), but −y⁴ makes 0 a maximizer.
        lambda x: float(-(x[0] ** 2) - x[1] ** 4),
        lambda x: np.array([-2.0 * x[0], -4.0 * x[1] ** 3]),
        lambda x: np.diag([-2.0, -12.0 * x[1] ** 2]),
        "saddle point or a maximizer",
    ),
    (  # ∇²f = diag(−2, −2) ≺ 0: a strict local maximizer.
        lambda x: float(-(x[0] ** 2) - x[1] ** 2),
        lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
        lambda x: np.diag([-2.0, -2.0]),
        "stopped at a maximizer,",
    ),
    (  # ∇²f = diag(−2, 2): indefinite, a saddle point.
        lambda x: float(-(x[0] ** 2) + x[1] ** 2),
        lambda x: np.array([-2.0 * x[0], 2.0 * x[1]]),
        lambda x: np.diag([-2.0, 2.0]),
        "stopped at a saddle point,",
    ),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("f", "g", "h", "phrase"), _CLASSIFY_CASES)
def test_stationary_point_classification(method: str, f: Any, g: Any, h: Any, phrase: str) -> None:
    # Start at the stationary point 0: every method stops at k = 0 with converged=False.
    res = numopt.run(method, _poly(f, g, h, [0.0, 0.0]))
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0
    assert phrase in res.message
    if "saddle point or a maximizer" in phrase:
        # Second-order information is the same for a saddle and a maximizer here.
        assert "cannot decide" in res.message and "λ_max = 0:" in res.message
    elif method == "pure_newton":
        # ∇²f is nonsingular: pure Newton from (1, 0) lands on 0 in one step (the x-part is
        # quadratic, y = 0). With ∇²f = diag(−2, 0) it would stop at x_0 (singular ∇²f).
        res = numopt.run(method, _poly(f, g, h, [1.0, 0.0]))
        assert not res.converged and res.n_iter == 1 and phrase in res.message
        assert_allclose(res.x, [0.0, 0.0], rtol=0, atol=0)


@given(
    st.lists(st.sampled_from([-3.0, -1.0, 0.0, 1.0, 3.0]), min_size=1, max_size=6),
    st.booleans(),
    st.data(),
)
@settings(max_examples=1000, deadline=None)
def test_second_order_classification_matches_eigenvalue_signs(
    signs: list[float], fd_hessian: bool, data: st.DataObject
) -> None:
    # Oracle: the textbook classification of a stationary point by the signs of the exact
    # eigenvalues of ∇²f (N&W Thms 2.3–2.4; with a zero eigenvalue and no positive one,
    # second-order information cannot separate a saddle point from a maximizer). The
    # eigenvalues are perturbed by up to 0.9·tol (the error of the Hessian source), which
    # must never change the verdict.
    exact = np.asarray(signs)
    n = exact.size
    lam_max = float(np.max(np.abs(exact)))
    tol = n * EPS ** (1 / 3) * max(1.0, lam_max) if fd_hessian else n * EPS * lam_max
    noise = data.draw(arrays(np.float64, n, elements=st.floats(-0.9, 0.9)))
    lam = np.sort(exact + noise * tol)
    ok, msg = _second_order(lam, tol, fd_hessian)
    neg, zero, pos = bool(np.any(exact < 0)), bool(np.any(exact == 0)), bool(np.any(exact > 0))
    if neg:
        assert not ok
        if pos:
            assert "stopped at a saddle point," in msg
        elif zero:
            assert "saddle point or a maximizer" in msg and "cannot decide" in msg
        else:
            assert "stopped at a maximizer," in msg
    elif zero:
        assert ok and "semidefinite" in msg and "not verified" in msg
    else:
        assert ok and "strict local minimizer" in msg
    assert ("finite-difference" in msg) == fd_hessian


def test_singular_hessian() -> None:
    prob = _poly(
        lambda x: float((x[0] + x[1]) ** 2),
        lambda x: 2.0 * (x[0] + x[1]) * np.ones(2),
        lambda x: np.full((2, 2), 2.0),
        [1.0, 2.0],
    )
    res = numopt.run("pure_newton", prob)
    assert_valid_result(res)
    assert not res.converged and "singular" in res.message
    assert res.n_iter == 0
    # Damped Newton falls back to −∇f and reaches the valley x + y = 0, where ∇²f is
    # positive semidefinite: converged, with the semidefinite caveat in the message.
    res = numopt.run("damped_newton", prob)
    assert_valid_result(res)
    assert res.converged and "semidefinite" in res.message
    assert res.trace[1].info["direction_type"] == "steepest"
    assert abs(res.x[0] + res.x[1]) <= 1e-8


# Audit regression (finite-difference Hessian): every point of x + y = 0 is a global minimizer
# of f = (x + y)² + c, and ∇²f = [[2, 2], [2, 2]] is singular. Given f alone, ∇f and ∇²f are
# central differences; their noise (≈ 1e-11 here) must not be read as negative curvature
# (a "saddle point", converged=False) or as positive curvature ("a strict local minimizer").


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize("c", [0.0, 3.0, 1e3])
def test_fd_hessian_does_not_misclassify_a_singular_minimizer(method: str, c: float) -> None:
    rng = np.random.default_rng(12345)
    for _ in range(40):
        x0 = rng.uniform(-3.0, 3.0, 2)
        res = numopt.run(method, lambda x: float((x[0] + x[1]) ** 2 + c), x0=x0)
        assert_valid_result(res)
        assert res.converged, res.message
        assert "within its accuracy" in res.message and "not verified" in res.message
        assert "saddle" not in res.message and "strict" not in res.message
        assert abs(res.x[0] + res.x[1]) <= 1e-6


def test_fd_hessian_singularity_stops_pure_newton() -> None:
    # The estimate has |λ|_min ≈ 1e-11 ≠ 0; with the analytic tolerance 2ε·4 pure Newton would
    # divide by that noise and jump ≈ 1e11·‖∇f‖ along the valley.
    res = numopt.run("pure_newton", lambda x: float((x[0] + x[1]) ** 2), x0=[1.0, 2.0])
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0
    assert "finite-difference ∇²f(x_0) is numerically singular" in res.message


def test_fd_hessian_still_classifies_clear_curvature() -> None:
    # Negative curvature far above the FD accuracy is still a saddle point / maximizer, and a
    # positive definite FD Hessian is still a strict local minimizer.
    for method in METHODS:
        res = numopt.run(method, lambda x: float(x[0] ** 2 - x[1] ** 2), x0=[0.0, 0.0])
        assert not res.converged and "stopped at a saddle point," in res.message
        assert "finite-difference ∇²f has the eigenvalue" in res.message
        res = numopt.run(method, lambda x: float(-(x[0] ** 2) - x[1] ** 2), x0=[0.0, 0.0])
        assert not res.converged and "stopped at a maximizer," in res.message
        res = numopt.run(method, problems.get("rosenbrock").f, x0=[-1.2, 1.0], gtol=1e-6)
        assert_valid_result(res)
        assert res.converged, res.message
        assert "finite-difference ∇²f is positive definite" in res.message
        assert_allclose(res.x, [1.0, 1.0], atol=1e-5)
    # With an analytic gradient and Hessian, the analytic tolerance n·ε·|λ|_max is used.
    res = numopt.run("damped_newton", problems.get("rosenbrock"))
    assert res.converged and "finite-difference" not in res.message


@given(
    arrays(np.float64, 2, elements=st.floats(-10.0, 10.0)),
    st.sampled_from([0.0, 1.0, 3.0, 1e3]),
    st.booleans(),
)
@settings(max_examples=1000, deadline=None)
def test_fd_eigenvalue_tolerance_bounds_the_fd_error(
    x: np.ndarray, c: float, analytic_grad: bool
) -> None:
    # Oracle: the exact eigenvalues {0, 4} of ∇²[(x + y)² + c]. The tolerance must cover the
    # error of the central-difference Hessian (from a central-difference or an analytic ∇f),
    # otherwise noise is classified as curvature.
    def f(z: np.ndarray) -> float:
        return float((z[0] + z[1]) ** 2 + c)

    def g(z: np.ndarray) -> np.ndarray:
        return 2.0 * (z[0] + z[1]) * np.ones(2)

    prob = Problem(
        id="t", name="t", latex="", f=f, grad=g if analytic_grad else None, dim=2, domain=()
    )
    x0, oracle = newton_mod._resolve(prob, x)
    assert oracle.hess_fd and oracle.grad_fd == (not analytic_grad)
    lam = np.asarray(np.linalg.eigvalsh(newton_mod._evaluate_hessian(oracle, x0)), dtype=np.float64)
    tol = newton_mod._eig_tol(lam, f(x0), oracle)
    assert np.max(np.abs(lam - np.array([0.0, 4.0]))) <= tol


# Audit regression (gtol = 0): ∇f eventually becomes so small that ∇fᵀp underflows to 0, and
# the line search (which requires ∇fᵀp < 0) used to raise ValueError. The contract: never
# raise on numerical breakdown.


def _quartic() -> Problem:
    return Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: float(x[0] ** 4),
        grad=lambda x: np.array([4.0 * x[0] ** 3]),
        hess=lambda x: np.array([[12.0 * x[0] ** 2]]),
        dim=1,
        domain=(),
    )


@pytest.mark.parametrize("method", LS_METHODS)
@pytest.mark.parametrize("ls", ["backtracking", "strong_wolfe", "weak_wolfe", "goldstein"])
def test_gtol_zero_reports_the_underflow_instead_of_raising(method: str, ls: str) -> None:
    res = numopt.run(method, _quartic(), x0=[1.0], gtol=0.0, max_iter=10_000, line_search=ls)
    assert_valid_result(res, max_iter=10_000)
    # ∇f = 0 exactly is unreachable here: the iterates approach 0 linearly (Newton on x⁴).
    assert not res.converged
    assert "line search failed" in res.message
    assert all(s.fun is not None and math.isfinite(s.fun) for s in res.trace)


@pytest.mark.parametrize("method", LS_METHODS)
def test_subnormal_gradient_stops_before_the_line_search(method: str) -> None:
    # x0 = 1e-107: ∇f = 4e-321 is subnormal and ∇fᵀp underflows to 0 at k = 0.
    res = numopt.run(method, _quartic(), x0=[1e-107], gtol=0.0)
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0
    assert "∇fᵀp underflowed to 0" in res.message and "rounding level" in res.message


def test_pure_newton_with_gtol_zero_is_honest() -> None:
    # Pure Newton has no line search; it stops when ∇f underflows to exactly 0.
    res = numopt.run("pure_newton", _quartic(), x0=[1.0], gtol=0.0, max_iter=10_000)
    assert_valid_result(res, max_iter=10_000)
    assert res.converged and res.trace[-1].info["grad"] == [0.0]


def _sqrt_problem() -> Problem:
    # f = √(1+x²) + √(1+y²): strictly convex, but Newton's x⁺ = −x³ diverges for |x| > 1.
    return _poly(
        lambda x: float(np.sum(np.sqrt(1.0 + x**2))),
        lambda x: x / np.sqrt(1.0 + x**2),
        lambda x: np.diag((1.0 + x**2) ** -1.5),
        [1.5, 0.5],
    )


def test_pure_newton_divergence_and_damped_newton_rescue() -> None:
    res = numopt.run("pure_newton", _sqrt_problem())
    assert_valid_result(res)
    assert not res.converged
    assert "diverge" in res.message
    xs = [s.x[0] for s in res.trace[:4]]
    assert_allclose(xs, [1.5, -(1.5**3), 1.5**9, -(1.5**27)], rtol=1e-12)
    for method in LS_METHODS:
        res = numopt.run(method, _sqrt_problem())
        assert res.converged, res.message
        assert_allclose(res.x, [0.0, 0.0], atol=1e-8)


def test_minimizer_far_from_the_origin_is_converged_not_diverged() -> None:
    # Audit regression: x* = (1e9, 1) lies beyond 10⁸·max(1, ‖x_0‖∞) from x_0 = 0. The single
    # Newton step lands exactly on x* (∇f = 0); the stopping test must run before the
    # divergence test, so pure Newton agrees with the line-search methods.
    x_star = np.array([1e9, 1.0])
    prob = _poly(
        lambda x: float((x[0] - 1e9) ** 2 + (x[1] - 1.0) ** 2),
        lambda x: 2.0 * (x - x_star),
        lambda x: 2.0 * np.eye(2),
        [0.0, 0.0],
    )
    for method in METHODS:
        res = numopt.run(method, prob)
        assert_valid_result(res)
        assert res.converged, res.message
        assert "strict local minimizer" in res.message
        assert_allclose(res.x, x_star, rtol=1e-15, atol=0)
    assert numopt.run("pure_newton", prob).n_iter == 1


def _dw_grad(x: np.ndarray) -> np.ndarray:
    return np.array([4 * x[0] ** 3 - 2 * x[0], 2 * x[1]])


def _dw_hess(x: np.ndarray) -> np.ndarray:
    return np.diag([12 * x[0] ** 2 - 2.0, 2.0])


def _double_well() -> Problem:
    # f = x⁴ − x² + y²: ∇²f is indefinite for |x| < 1/√6.
    return _poly(lambda x: float(x[0] ** 4 - x[0] ** 2 + x[1] ** 2), _dw_grad, _dw_hess, [0.3, 0.2])


def test_damped_newton_falls_back_to_steepest_descent() -> None:
    prob = _double_well()
    g0 = _dw_grad(np.array([0.3, 0.2]))
    # The pure Newton step goes uphill in x (the audit's ascent-step bug in the legacy code).
    pure = numopt.run("pure_newton", prob, max_iter=1)
    assert pure.trace[1].info["descent"] is False
    res = numopt.run("damped_newton", prob)
    assert_valid_result(res)
    assert res.trace[1].info["direction_type"] == "steepest"
    assert_allclose(res.trace[1].info["direction"], -g0, rtol=0, atol=0)
    assert res.converged
    assert_allclose(res.x, [1 / math.sqrt(2), 0.0], atol=1e-8)


def test_modified_newton_shifts_an_indefinite_hessian() -> None:
    prob = _double_well()
    res = numopt.run("modified_newton", prob)
    assert_valid_result(res)
    info = res.trace[1].info
    # ∇²f(x0) = diag(−0.92, 2): Alg. 3.3 starts at τ₀ = 0.92 + β and succeeds at once.
    assert info["tau"] == pytest.approx(0.92 + 1e-3, rel=1e-12)
    assert info["chol_attempts"] == 1
    x0 = np.array([0.3, 0.2])
    B = _dw_hess(x0) + info["tau"] * np.eye(2)
    assert_allclose(info["direction"], scipy.linalg.solve(B, -_dw_grad(x0)), rtol=1e-12)
    assert res.converged
    assert_allclose(res.x, [1 / math.sqrt(2), 0.0], atol=1e-8)
    # Near the minimizer ∇²f ≻ 0: no shift, i.e. Newton's method.
    assert res.trace[-1].info["tau"] == 0.0


# --------------------------------------------------------------------------------------
# N&W Alg. 3.3 and the triangular solves against independent oracles
# --------------------------------------------------------------------------------------


def _tau_oracle(A: np.ndarray, beta: float) -> float | None:
    """First τ of Alg. 3.3's sequence with λ_min(A) + τ > 0; None if ambiguous at rounding."""
    lam_min = float(np.linalg.eigvalsh(A)[0])
    a_min = float(np.min(np.diag(A)))
    tau = 0.0 if a_min > 0.0 else -a_min + beta
    scale = float(np.max(np.abs(A))) + 1.0
    while True:
        margin = lam_min + tau
        if abs(margin) <= 1e-9 * (scale + tau):
            return None  # Cholesky success is decided by rounding here
        if margin > 0.0:
            return tau
        tau = max(2.0 * tau, beta)


@given(
    st.integers(1, 5),
    arrays(np.float64, (5, 5), elements=st.floats(-10, 10)),
    st.floats(1e-4, 1.0),
)
@settings(max_examples=1000, deadline=None)
def test_cholesky_shift_matches_eigenvalue_oracle(n: int, M: np.ndarray, beta: float) -> None:
    A = 0.5 * (M[:n, :n] + M[:n, :n].T)
    # Shifts that leave A + τI nearly singular make the solve comparison meaningless.
    expected = _tau_oracle(A, beta)
    L, tau, attempts = _cholesky_shift(A, beta)
    assert L is not None
    if expected is not None:
        assert tau == expected
    assert attempts >= 1
    assert np.all(np.diag(L) > 0.0)
    assert_allclose(L @ L.T, A + tau * np.eye(n), rtol=0, atol=1e-12 * (np.abs(A).max() + tau))
    b = M[:n, 0] + 1.0
    kappa = np.linalg.cond(L @ L.T)
    assume(kappa < 1e10)
    z = _cholesky_solve(L, b)
    z_ref = scipy.linalg.cho_solve((L, True), b)
    assert_allclose(z, z_ref, rtol=0, atol=10 * n * kappa * EPS * max(1.0, np.abs(z_ref).max()))


# --------------------------------------------------------------------------------------
# Properties: descent and monotone decrease
# --------------------------------------------------------------------------------------


@given(
    st.sampled_from(LS_METHODS),
    st.sampled_from(["rosenbrock", "himmelblau", "six_hump_camel", "beale"]),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.sampled_from(["backtracking", "strong_wolfe", "weak_wolfe", "goldstein"]),
)
@settings(max_examples=150, deadline=None)
def test_line_search_newton_is_a_descent_method(
    method: str, pid: str, u: float, v: float, ls: str
) -> None:
    prob = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = prob.domain[0], prob.domain[1]
    x0 = [x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)]
    res = numopt.run(method, prob, x0=x0, line_search=ls, max_iter=60)
    assert_valid_result(res, max_iter=60)
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        p = np.asarray(cur.info["direction"])
        slope = float(np.asarray(prev.info["grad"]) @ p)
        assert slope < 0.0
        assert cur.info["descent"] is True
        # Armijo (or Goldstein's upper test) with c₁ = 1e-4 (Goldstein: c = 0.25).
        c = 0.25 if ls == "goldstein" else 1e-4
        assert cur.step_size is not None and cur.fun is not None and prev.fun is not None
        assert cur.fun <= prev.fun + c * cur.step_size * slope


# --------------------------------------------------------------------------------------
# Evaluation counts, finite-difference fallbacks, failure paths, contract
# --------------------------------------------------------------------------------------


class _Calls:
    def __init__(self) -> None:
        self.f = self.g = self.h = 0


def _counted_rosenbrock(calls: _Calls) -> Problem:
    base = problems.get("rosenbrock")

    def f(x: np.ndarray) -> float:
        calls.f += 1
        return base.f(x)

    def g(x: np.ndarray) -> np.ndarray:
        calls.g += 1
        return base.grad(x)

    def h(x: np.ndarray) -> np.ndarray:
        calls.h += 1
        return base.hess(x)

    return _poly(f, g, h, base.x0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("ls", ["backtracking", "strong_wolfe"])
def test_evaluation_counts_are_exact(method: str, ls: str) -> None:
    calls = _Calls()
    kw = {} if method == "pure_newton" else {"line_search": ls}
    res = numopt.run(method, _counted_rosenbrock(calls), **kw)
    assert res.converged
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls.f, calls.g, calls.h)
    assert res.n_hev == res.n_iter + 1
    if method == "pure_newton":
        assert res.n_fev == res.n_gev == res.n_iter + 1
    else:
        trials = sum(len(s.info["trials"]) for s in res.trace)
        assert res.n_fev == 1 + trials


def test_finite_difference_fallbacks_are_counted() -> None:
    calls = _Calls()
    base = problems.get("rosenbrock")

    def f(x: np.ndarray) -> float:
        calls.f += 1
        return base.f(x)

    res = numopt.minimize(f, x0=[-1.2, 1.0], method="damped_newton", gtol=1e-6)
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, 1.0], atol=1e-5)
    assert res.n_fev == calls.f
    # every gradient costs 2n = 4 values of f; every Hessian 2n = 4 gradients.
    assert res.n_gev >= 4 * res.n_hev


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_is_reported(method: str) -> None:
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=2)
    assert_valid_result(res, max_iter=2)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 2 and len(res.trace) == 3


def test_line_search_failure_when_f_is_unbounded_below() -> None:
    # f = x − y² (y-direction unbounded below, ∇²f = diag(0, −2) singular and indefinite).
    prob = _poly(
        lambda x: float(x[0] - x[1] ** 2),
        lambda x: np.array([1.0, -2.0 * x[1]]),
        lambda x: np.diag([0.0, -2.0]),
        [0.0, 0.5],
    )
    res = numopt.run("modified_newton", prob, line_search="strong_wolfe")
    assert_valid_result(res)
    assert not res.converged
    assert "line search failed" in res.message and "unbounded" in res.message


@pytest.mark.parametrize("method", METHODS)
def test_invalid_input_raises(method: str) -> None:
    prob = problems.get("rosenbrock")
    with pytest.raises(ValueError):
        numopt.run(method, prob, x0=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        numopt.run(method, prob, gtol=-1.0)
    with pytest.raises(ValueError):
        numopt.run(method, prob, max_iter=0)
    with pytest.raises(ValueError):
        numopt.minimize(lambda x: float("nan"), x0=[0.0, 0.0], method=method)
    if method != "pure_newton":
        with pytest.raises(ValueError):
            numopt.run(method, prob, line_search="exact")
    if method == "modified_newton":
        with pytest.raises(ValueError):
            numopt.run(method, prob, beta=0.0)


_COMMON_KEYS = {"grad", "hess", "hess_eigs", "direction", "alpha", "trials", "descent"}
_EXTRA_KEYS = {
    "pure_newton": set(),
    "damped_newton": {"direction_type"},
    "modified_newton": {"tau", "chol_attempts"},
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "quadratic_nd"])
def test_info_keys_and_trace_contract(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=100)
    assert res.n_iter == res.trace[-1].k
    for step in res.trace:
        assert set(step.info) == _COMMON_KEYS | _EXTRA_KEYS[method]
        assert (step.info["hess"] is None) == (prob.dim > 2)
        assert_allclose(step.info["hess_eigs"], np.linalg.eigvalsh(prob.hess(step.x)), atol=1e-9)
        assert step.grad_norm == pytest.approx(float(np.linalg.norm(prob.grad(step.x))))
        assert step.fun == pytest.approx(prob.f(step.x), rel=1e-15, abs=1e-300)
    first = res.trace[0]
    assert first.step_size is None and first.info["direction"] is None
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        p = np.asarray(cur.info["direction"])
        assert_allclose(cur.x, np.asarray(prev.x) + cur.step_size * p, rtol=0, atol=0)


@pytest.mark.parametrize(("method", "pid", "params"), newton_mod.FIXTURE_CASES)
def test_fixture_cases_run(method: str, pid: str, params: dict[str, Any]) -> None:
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300


def test_modified_newton_reports_when_alg_3_3_finds_no_shift() -> None:
    # λ_min(A) ≈ −1e17 lies beyond the largest shift Alg. 3.3 tries (β·2⁶³ ≈ 9.2e15).
    A = np.array([[1.0, 1e17], [1e17, 1.0]])
    L, tau, attempts = _cholesky_shift(A, 1e-3)
    assert L is None and attempts == 65 and tau == 1e-3 * 2.0**63
    res = numopt.run("modified_newton", _quadratic(A, np.zeros(2)), x0=[1.0, 0.0])
    assert_valid_result(res)
    assert not res.converged and "no shift" in res.message
    assert res.n_iter == 0


# --------------------------------------------------------------------------------------
# A non-finite slope ∇fᵀp is a failed run, not an exception (web-port regression)
# --------------------------------------------------------------------------------------


def _concave_bowl() -> Problem:
    """f = −(x² + y²): unbounded below, so the iterates grow until ∇fᵀp overflows to −inf."""
    return Problem(
        id="concave_bowl",
        name="concave bowl",
        latex="",
        f=lambda x: float(-(x[0] ** 2 + x[1] ** 2)),
        grad=lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: -2.0 * np.eye(2),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=np.array([1.0, 0.5]),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the user f overflows, not the method
@pytest.mark.parametrize("line_search", ["backtracking", "strong_wolfe"])
@pytest.mark.parametrize("method", LS_METHODS)
def test_non_finite_slope_is_a_failed_run(method: str, line_search: str) -> None:
    """Regression: `if not float(g @ p) < 0.0` let ∇fᵀp = −inf through, and search() raised
    ValueError. The run must end with converged=False and a finite last iterate."""
    res = numopt.run(method, _concave_bowl(), line_search=line_search, max_iter=2000)
    assert_valid_result(res)
    assert not res.converged
    assert res.fun is not None and np.all(np.isfinite(res.x)) and math.isfinite(res.fun)
    if line_search == "backtracking":  # α = 1 is accepted until ‖x‖ ≈ 1e154
        assert "∇fᵀp overflowed to -inf, so no step can be tested" in res.message, res.message
        assert res.message.startswith(f"line search failed at iteration {res.n_iter + 1}")
        grad_norm = res.trace[-1].grad_norm
        assert grad_norm is not None and grad_norm > 1e153
    else:  # strong Wolfe: φ still decreases at α_max on the first search
        assert "phi still decreases at alpha_max" in res.message, res.message


def test_slope_helpers() -> None:
    big = np.array([1e200, 1e200])
    assert newton_mod._slope(big, -big) == -math.inf  # no RuntimeWarning (errstate)
    assert (
        newton_mod._slope_failure(-math.inf) == "∇fᵀp overflowed to -inf, so no step can be tested"
    )
    assert newton_mod._slope_failure(math.nan) == "∇fᵀp overflowed to nan, so no step can be tested"
    assert newton_mod._slope_failure(0.0) == "∇fᵀp underflowed to 0, so no step can be tested"
    assert newton_mod._slope(np.array([1.0, 2.0]), np.array([-3.0, 0.5])) == -2.0
