"""Tests for the unconstrained problem library (numopt.problems.unconstrained).

Derivative oracle: the 5-point central difference
    f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h),  h = ε^{1/5} ≈ 7.4·10⁻⁴,
whose truncation error is h⁴|f⁽⁵⁾|/30 ≈ 10⁻¹⁴|f⁽⁵⁾| and rounding error ≈ ε|f|/h ≈ 3·10⁻¹³|f|.
The step is absolute, not scaled by |x|: the oscillatory problems have period ~1 independent
of |x| (a step ε^{1/5}·|x| ≈ 3.6·10⁻³ at |x| ≈ 5 gave a 10⁻⁶ truncation error on the Ackley
Hessian, whose analytic value agrees with a sympy evaluation to 6·10⁻¹⁵). All domains have
|x| ≤ 10, so representing x ± h costs at most ε·10/h ≈ 3·10⁻¹² relative.
The tolerance below (1e-7 relative to the local scale of f and its derivative) leaves a margin
of ~10² over that estimate for the oscillatory problems (|f⁽⁶⁾| ≈ π(2π)⁵ for Ackley).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import rosen, rosen_der, rosen_hess

from numopt import problems
from numopt.core.types import Problem

REQUIRED = (
    "quadratic_bowl",
    "quadratic_ill",
    "rosenbrock",
    "himmelblau",
    "beale",
    "booth",
    "matyas",
    "three_hump_camel",
    "six_hump_camel",
    "goldstein_price",
    "rastrigin",
    "ackley",
    "styblinski_tang",
    "mccormick",
    "bohachevsky",
    "levi13",
    "rosenbrock_nd",
    "quadratic_nd",
)
ALL = [p.id for p in problems.list_problems("unconstrained")]
TWO_D = [pid for pid in ALL if problems.get(pid).dim == 2]
_H5 = float(np.finfo(float).eps) ** 0.2


def _fd5(fun, x: np.ndarray) -> np.ndarray:
    """5-point central-difference derivative of fun (scalar or vector valued) w.r.t. x."""
    cols = []
    for i in range(x.size):
        h = _H5
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


def _check_derivatives(p: Problem, x: np.ndarray) -> None:
    assert p.grad and p.hess
    g = p.grad(x)
    H = p.hess(x)
    assert g.shape == (p.dim,) and H.shape == (p.dim, p.dim)
    assert_allclose(H, H.T, rtol=0, atol=1e-12 * (1 + np.abs(H).max()))
    scale_g = 1.0 + abs(float(p.f(x))) + float(np.abs(g).max())
    assert_allclose(g, _fd5(p.f, x), rtol=1e-7, atol=1e-7 * scale_g)
    scale_h = 1.0 + float(np.abs(g).max()) + float(np.abs(H).max())
    assert_allclose(H, _fd5(p.grad, x), rtol=1e-7, atol=1e-7 * scale_h)


def test_required_ids_registered_with_metadata():
    for pid in REQUIRED:
        p = problems.get(pid)
        assert problems.kind_of(pid) == "unconstrained"
        assert isinstance(p, Problem)
        assert p.grad is not None and p.hess is not None
        assert len(p.domain) == p.dim and all(lo < hi for lo, hi in p.domain)
        assert len(p.x0) == p.dim
        assert p.minima and len(p.extra["minima_f"]) == len(p.minima)
        assert 1 <= p.extra["n_global"] <= len(p.minima)
        assert p.tags and p.description and p.latex
        p.to_dict()  # JSON-able metadata
    assert problems.get("rosenbrock_nd").dim == 10
    assert problems.get("quadratic_nd").dim == 20
    assert len(problems.get("himmelblau").minima) == 4
    assert problems.get("himmelblau").extra["n_global"] == 4


@pytest.mark.parametrize("pid", ALL)
def test_x0_inside_domain_and_f_finite(pid):
    p = problems.get(pid)
    x0 = np.array(p.x0)
    assert all(lo <= v <= hi for v, (lo, hi) in zip(x0, p.domain, strict=True))
    assert math.isfinite(p.f(x0))


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_match_finite_differences_at_x0(pid):
    p = problems.get(pid)
    _check_derivatives(p, np.array(p.x0, dtype=float))


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_match_finite_differences_hypothesis(pid):
    p = problems.get(pid)
    lows = np.array([lo for lo, _ in p.domain])
    highs = np.array([hi for _, hi in p.domain])

    @settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(st.lists(st.floats(0.0, 1.0), min_size=p.dim, max_size=p.dim))
    def check(u):
        x = lows + np.array(u) * (highs - lows)
        if pid == "ackley":
            # Ackley has a cone-shaped kink at 0; keep the FD stencil (width 4h ≈ 3e-3) away.
            assume(np.linalg.norm(x) > 0.05)
        _check_derivatives(p, x)

    check()


@pytest.mark.parametrize("pid", ALL)
def test_listed_minima_are_strict_local_minima(pid):
    p = problems.get(pid)
    fs = p.extra["minima_f"]
    n_global = p.extra["n_global"]
    for m, fm in zip(p.minima, fs, strict=True):
        x = np.array(m, dtype=float)
        g = p.grad(x)
        H = p.hess(x)
        # Stationarity to rounding level: |∇f| ≤ 1e-11 · (1 + |∇²f|·(1 + |x|)).
        assert np.abs(g).max() <= 1e-11 * (1 + np.abs(H).max() * (1 + np.abs(x).max())), (m, g)
        assert np.linalg.eigvalsh(H).min() > 0, m
        assert math.isclose(float(p.f(x)), fm, rel_tol=1e-12, abs_tol=1e-12)
        assert all(lo <= v <= hi for v, (lo, hi) in zip(x, p.domain, strict=True))
    assert fs[0] == p.extra["f_min"]
    for fg in fs[:n_global]:
        assert math.isclose(fg, fs[0], rel_tol=1e-12, abs_tol=1e-12)
    for fl in fs[n_global:]:
        assert fl > fs[0] + 1e-6


@pytest.mark.parametrize("pid", TWO_D)
def test_global_minimum_not_beaten_on_dense_grid(pid):
    """Independent check of the global label: no point of a 601×601 grid of the box is lower."""
    p = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, 601), np.linspace(y_lo, y_hi, 601))
    F = p.f(np.stack([X, Y]))
    assert F.shape == X.shape
    assert F.min() >= p.extra["f_min"] - 1e-12 * (1 + abs(p.extra["f_min"]))


@pytest.mark.parametrize("pid", ALL)
def test_f_vectorizes_over_grids(pid):
    p = problems.get(pid)
    pts = [np.array(p.x0, dtype=float), np.array(p.minima[0], dtype=float)]
    grid = np.stack(pts, axis=-1).reshape(p.dim, 1, 2)  # (n, 1, 2)
    F = p.f(grid)
    assert F.shape == (1, 2)
    assert_allclose(F[0], [p.f(q) for q in pts], rtol=1e-14, atol=1e-14)


def test_rosenbrock_matches_scipy():
    rng = np.random.default_rng(0)
    for pid, n in (("rosenbrock", 2), ("rosenbrock_nd", 10)):
        p = problems.get(pid)
        for _ in range(50):
            x = rng.uniform(-2, 2, size=n)
            assert_allclose(p.f(x), rosen(x), rtol=1e-14)
            assert_allclose(p.grad(x), rosen_der(x), rtol=1e-13, atol=1e-12)
            assert_allclose(p.hess(x), rosen_hess(x), rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize(
    ("pid", "x", "value"),
    [
        # Reference values: closed forms or the J&Y survey.
        ("himmelblau", (0.0, 0.0), 170.0),
        ("beale", (0.0, 0.0), 1.5**2 + 2.25**2 + 2.625**2),
        ("booth", (0.0, 0.0), 74.0),
        ("goldstein_price", (0.0, 0.0), 600.0),
        ("goldstein_price", (0.0, -1.0), 3.0),
        ("rastrigin", (1.0, 1.0), 2.0),
        ("ackley", (0.0, 0.0), 0.0),
        ("styblinski_tang", (0.0, 0.0), 0.0),
        ("matyas", (1.0, 1.0), 0.04),
        ("levi13", (1.0, 1.0), 0.0),
        ("bohachevsky", (0.0, 0.0), 0.0),
    ],
)
def test_reference_values(pid, x, value):
    assert math.isclose(
        float(problems.get(pid).f(np.array(x))), value, rel_tol=1e-13, abs_tol=1e-14
    )


def test_literature_minimum_values():
    # J&Y: six-hump camel f* = −1.0316 (Dixon & Szegő 1978: −1.031628453489877),
    # Styblinski–Tang f* = −39.16599·n, McCormick f* = −1.9133 (exact −π/3 − √3/2).
    assert math.isclose(
        problems.get("six_hump_camel").extra["f_min"], -1.031628453489877, rel_tol=1e-13
    )
    assert math.isclose(
        problems.get("styblinski_tang").extra["f_min"] / 2, -39.16616570377142, rel_tol=1e-13
    )
    assert math.isclose(problems.get("mccormick").extra["f_min"], -1.913222954981037, rel_tol=1e-13)
    assert math.isclose(
        problems.get("three_hump_camel").extra["minima_f"][1], 0.29863844223686, rel_tol=1e-10
    )


def test_quadratic_conditioning_and_minimizer():
    for pid, kappa in (("quadratic_bowl", 2.0), ("quadratic_ill", 50.0), ("quadratic_nd", 100.0)):
        p = problems.get(pid)
        A = np.array(p.extra["A"])
        c = np.array(p.extra["c"])
        lam = np.linalg.eigvalsh(A)
        assert_allclose(A, A.T, rtol=0, atol=0)
        assert lam.min() > 0
        assert_allclose(lam.max() / lam.min(), kappa, rtol=1e-12)
        assert_allclose(p.hess(p.x0), A, rtol=0, atol=0)
        # Oracle: the minimizer solves A x = A c (independent of how c was stored).
        assert_allclose(np.linalg.solve(A, A @ c), p.minima[0], rtol=1e-12, atol=1e-12)
    q = problems.get("quadratic_nd")
    assert_allclose(np.linalg.eigvalsh(np.array(q.extra["A"])), q.extra["eigenvalues"], rtol=1e-12)


def test_quadratic_nd_is_deterministic():
    from numopt.problems.unconstrained import quadratic_nd

    a = quadratic_nd()  # the factory returns the builder; rebuilding replays Rng(20)
    b = problems.get("quadratic_nd")
    assert a.extra["A"] == b.extra["A"] and a.minima == b.minima


def test_ackley_gradient_finite_at_and_near_origin():
    p = problems.get("ackley")
    assert np.array_equal(p.grad(np.zeros(2)), np.zeros(2))
    assert np.all(np.isfinite(p.hess(np.zeros(2))))
    for x in (np.array([1e-300, 0.0]), np.array([1e-12, -1e-12]), np.array([-1e-8, 3e-9])):
        assert np.all(np.isfinite(p.grad(x))) and np.all(np.isfinite(p.hess(x)))
        # The cone term dominates near 0: |∇f| → 4/√2·(radial unit) = 2√2.
        assert math.isclose(float(np.linalg.norm(p.grad(x))), 2 * math.sqrt(2), rel_tol=1e-6)


def test_mccormick_is_unbounded_below_off_the_box():
    """The description claims f → −∞ along x − y = 1; the box minimum is only box-global."""
    p = problems.get("mccormick")
    s = -2 * math.pi / 3 - 2 * math.pi  # the next minimizer to the left, outside the box
    x = np.array([(s + 1) / 2, (s - 1) / 2])
    assert x[0] < p.domain[0][0]
    assert np.abs(p.grad(x)).max() < 1e-12 and p.f(x) < p.extra["f_min"]
