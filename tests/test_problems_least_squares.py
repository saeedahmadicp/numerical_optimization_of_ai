"""Tests for the least-squares problem library (numopt.problems.least_squares).

Derivative oracle: the 5-point central difference with a per-component step
h_i = ε^{1/5}·max(|x_i|, 10⁻²). The step is relative because the Michaelis–Menten parameters
differ in scale by ~3000× (V ≈ 200, K ≈ 0.06; an absolute step 7·10⁻⁴ on K would put the
truncation error h⁴|∂⁵r/∂K⁵|/30 near 1). Truncation ≈ 10⁻¹⁴·|x|⁴|r⁽⁵⁾| and rounding
≈ ε|r|/h stay ≥ 10³ below the 1e-7 tolerance on every problem's domain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import least_squares, rosen

from numopt import problems
from numopt.core.types import Problem

REQUIRED = ("exp_decay_fit", "rosenbrock_ls", "circle_fit", "michaelis_menten")
ALL = [p.id for p in problems.list_problems("least_squares")]
_H5 = float(np.finfo(float).eps) ** 0.2


def _fd5(fun, x: np.ndarray) -> np.ndarray:
    cols = []
    for i in range(x.size):
        h = _H5 * max(abs(x[i]), 1e-2)
        e = np.zeros_like(x)
        e[i] = h
        d = (
            -np.asarray(fun(x + 2 * e))
            + 8 * np.asarray(fun(x + e))
            - 8 * np.asarray(fun(x - e))
            + np.asarray(fun(x - 2 * e))
        )
        cols.append(d / (12 * h))
    return np.stack(cols, axis=-1)


def _check(p: Problem, x: np.ndarray) -> None:
    assert p.residual and p.jac and p.grad and p.hess
    r = p.residual(x)
    J = p.jac(x)
    m = p.extra["m"]
    assert r.shape == (m,) and J.shape == (m, p.dim)
    # f, ∇f = Jᵀr and ∇²f against their definitions and against finite differences.
    assert_allclose(p.f(x), 0.5 * r @ r, rtol=1e-14, atol=0)
    assert_allclose(p.grad(x), J.T @ r, rtol=1e-14, atol=1e-14 * (1 + np.abs(J.T @ r).max()))
    assert_allclose(J, _fd5(p.residual, x), rtol=1e-7, atol=1e-7 * (1 + np.abs(J).max()))
    g = p.grad(x)
    assert_allclose(g, _fd5(p.f, x), rtol=1e-7, atol=1e-7 * (1 + abs(p.f(x)) + np.abs(g).max()))
    H = p.hess(x)
    assert_allclose(H, H.T, rtol=0, atol=0)
    assert_allclose(H, _fd5(p.grad, x), rtol=1e-7, atol=1e-7 * (1 + np.abs(H).max()))


def test_required_ids_and_metadata():
    for pid in REQUIRED:
        p = problems.get(pid)
        assert problems.kind_of(pid) == "least_squares"
        assert p.residual is not None and p.jac is not None and p.grad and p.hess
        assert p.dim == 2 and len(p.x0) == 2 and len(p.domain) == 2
        assert "least-squares" in p.tags
        assert p.minima and len(p.extra["minima_f"]) == len(p.minima)
        p.to_dict()
    for pid in ("exp_decay_fit", "michaelis_menten"):
        e = problems.get(pid).extra
        assert isinstance(e["t"], list) and isinstance(e["y"], list)
        assert len(e["t"]) == len(e["y"]) == e["m"]
    c = problems.get("circle_fit").extra
    assert isinstance(c["points_x"], list) and len(c["points_x"]) == len(c["points_y"]) == c["m"]


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_at_x0_and_minima(pid):
    p = problems.get(pid)
    for x in (p.x0, *p.minima):
        _check(p, np.array(x, dtype=float))


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_hypothesis(pid):
    p = problems.get(pid)
    lows = np.array([lo for lo, _ in p.domain])
    highs = np.array([hi for _, hi in p.domain])

    @settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
    def check(u, v):
        x = lows + np.array([u, v]) * (highs - lows)
        if pid == "circle_fit":
            # NOTE: rᵢ = ‖pᵢ − c‖ − R is not smooth at c = pᵢ and its k-th derivatives grow like
            # d^{1−k}; the FD truncation error ≈ (h/d)⁴/30 reached 2.4e-7 at d = 0.038 (h = 2e-3).
            # Keep d ≥ 0.25, where it is ≤ 4e-9; test_circle_fit_jacobian_rows_are_unit_vectors
            # covers the analytic Jacobian everywhere else.
            d = np.hypot(np.array(p.extra["points_x"]) - x[0], np.array(p.extra["points_y"]) - x[1])
            assume(d.min() >= 0.25)
        _check(p, x)

    check()


@pytest.mark.parametrize("pid", ALL)
def test_minima_are_stationary_and_strict(pid):
    p = problems.get(pid)
    for m, fm in zip(p.minima, p.extra["minima_f"], strict=True):
        x = np.array(m)
        H = p.hess(x)
        # |∇f| at rounding level: ε·Σ|J||r| ≲ 1e-11 for these data scales.
        assert np.abs(p.grad(x)).max() <= 1e-10 * (1 + np.abs(H).max()), (m, p.grad(x))
        assert np.linalg.eigvalsh(H).min() > 0
        assert fm == p.f(x)
    fs = p.extra["minima_f"]
    assert all(f > fs[0] for f in fs[p.extra["n_global"] :])


@pytest.mark.parametrize("pid", ALL)
def test_global_minimum_not_beaten_on_dense_grid(pid):
    p = problems.get(pid)
    (a_lo, a_hi), (b_lo, b_hi) = p.domain
    A, B = np.meshgrid(np.linspace(a_lo, a_hi, 401), np.linspace(b_lo, b_hi, 401))
    F = p.f(np.stack([A, B]))
    assert F.shape == A.shape
    assert F.min() >= p.extra["f_min"] * (1 - 1e-12)


@pytest.mark.parametrize("pid", ALL)
def test_scipy_least_squares_oracle_finds_listed_minimum(pid):
    """Oracle: MINPACK lmder via scipy.optimize.least_squares(method='lm') from x0."""
    p = problems.get(pid)
    sol = least_squares(
        p.residual, p.x0, jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    assert sol.success
    # NOTE: rtol 1e-7, not 1e-12 — MINPACK stops on its ftol test ~1e-9 (relative) short of the
    # minimizer (measured: 6e-10 on V and 5e-9 on K for michaelis_menten, κ(JᵀJ) ≈ 1.7e6).
    assert_allclose(sol.x, p.minima[0], rtol=1e-7)
    assert_allclose(sol.cost, p.extra["f_min"], rtol=1e-12)


def test_michaelis_menten_reproduces_bates_watts():
    """Bates & Watts (1988), §2.2.1: θ̂ = (212.68, 0.06412), S(θ̂) = 1195 (1195.449 in R)."""
    p = problems.get("michaelis_menten")
    V, K = p.minima[0]
    assert round(V, 2) == 212.68
    assert round(K, 5) == 0.06412
    assert round(2 * p.extra["f_min"], 3) == 1195.449


def test_rosenbrock_ls_is_half_rosenbrock():
    p = problems.get("rosenbrock_ls")
    rng = np.random.default_rng(1)
    for _ in range(100):
        x = rng.uniform(-2, 2, 2)
        assert_allclose(p.f(x), 0.5 * rosen(x), rtol=1e-14)
    assert np.linalg.det(p.jac(np.array([0.3, -0.7]))) == pytest.approx(10.0)


def test_rosenbrock_ls_description_claims_hold():
    """The description's numbers: 2 full GN steps via (1, −3.84); Armijo GN takes 10 steps."""
    import numopt

    p = problems.get("rosenbrock_ls")
    full = numopt.run("gauss_newton", p, line_search="none")
    assert full.converged and full.n_iter == 2
    # Newton on r = 0: x₁ = 1 solves the linear r₂ exactly; y₁ = 2x₀x₁ − x₀² = −2.4 − 1.44.
    assert_allclose(full.trace[1].x, [1.0, -3.84], rtol=1e-14)
    damped = numopt.run("gauss_newton", p)
    assert damped.converged and damped.n_iter == 10
    first_alpha, first_f = damped.trace[1].info["trials"][0]
    assert first_alpha == 1.0 and round(first_f) == 1171 and round(p.f(np.array(p.x0)), 1) == 12.1
    assert damped.trace[1].info["alpha"] < 1.0
    for claim in ("two steps", "(1, −3.84)", "1171", "12.1", "10 steps"):
        assert claim in p.description


def test_data_are_reproducible_and_plausible():
    from numopt.problems.least_squares import circle_fit, exp_decay_fit

    a, b = exp_decay_fit(), problems.get("exp_decay_fit")
    assert a.extra["y"] == b.extra["y"]
    t, y = np.array(b.extra["t"]), np.array(b.extra["y"])
    a0, b0 = b.extra["true_params"]
    noise = y - a0 * np.exp(-b0 * t)
    assert np.abs(noise).max() < 4 * b.extra["noise_std"]
    assert_allclose(t, 0.3 * np.arange(15), rtol=0, atol=1e-15)

    c, d = circle_fit(), problems.get("circle_fit")
    assert c.extra["points_x"] == d.extra["points_x"] and c.extra["points_y"] == d.extra["points_y"]
    cx, cy = d.extra["true_params"]
    dist = np.hypot(np.array(d.extra["points_x"]) - cx, np.array(d.extra["points_y"]) - cy)
    assert np.abs(dist - d.extra["radius"]).max() < 4 * d.extra["noise_std"]
    # The points lie on a 120° arc above the centre (θ ∈ [π/6, 5π/6]).
    theta = np.arctan2(np.array(d.extra["points_y"]) - cy, np.array(d.extra["points_x"]) - cx)
    assert theta.min() > math.pi / 6 - 1e-12 and theta.max() < 5 * math.pi / 6 + 1e-12


def test_circle_fit_mirror_minimum_reached_from_above():
    p = problems.get("circle_fit")
    sol = least_squares(p.residual, [1.0, 4.0], jac=p.jac, method="lm", xtol=1e-15, ftol=1e-15)
    assert_allclose(sol.x, p.minima[1], rtol=1e-7)


@settings(max_examples=1000, deadline=None)
@given(st.floats(-2.0, 4.0), st.floats(-3.0, 4.5))
def test_circle_fit_jacobian_rows_are_unit_vectors(a, b):
    """Exact identity: ∂dᵢ/∂c = −(pᵢ − c)/dᵢ, a unit vector, and (pᵢ − c) = −dᵢ·Jᵢ."""
    p = problems.get("circle_fit")
    x = np.array([a, b])
    P = np.column_stack([p.extra["points_x"], p.extra["points_y"]])
    d = np.hypot(P[:, 0] - a, P[:, 1] - b)
    assume(d.min() > 0.0)
    J = p.jac(x)
    assert_allclose(np.hypot(J[:, 0], J[:, 1]), 1.0, rtol=1e-15)
    assert_allclose(-d[:, None] * J, P - x, rtol=1e-13, atol=1e-15)


def test_circle_fit_centre_on_a_data_point_is_finite():
    p = problems.get("circle_fit")
    x = np.array([p.extra["points_x"][0], p.extra["points_y"][0]])
    J = p.jac(x)
    assert np.all(np.isfinite(J)) and np.all(J[0] == 0.0)
    assert np.all(np.isfinite(p.hess(x)))
