"""Tests for the scalar_min problem library: derivatives, minima, domains, metadata."""

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

from numopt import problems
from numopt.problems import scalar_min

REQUIRED = (
    "quadratic_1d",
    "quartic_1d",
    "sin_1d",
    "x_log_x",
    "abs_shifted",
    "multimodal_1d",
    "flat_valley",
    "rational_1d",
    "drug_concentration",
)
ALL = [p.id for p in problems.list_problems("scalar_min")]
# Points where f is not differentiable (excluded from finite-difference checks).
KINKS = {"abs_shifted": (0.3,)}


def central(fn, x: float) -> float:
    """Central difference with h ≈ ε^{1/3}·max(1,|x|): error ~ 1e-10 relative for smooth fn."""
    h = np.finfo(float).eps ** (1 / 3) * max(1.0, abs(x))
    return (fn(x + h) - fn(x - h)) / (2 * h)


def interior_points(pid: str, n: int = 41) -> np.ndarray:
    lo, hi = problems.get(pid).domain
    pts = np.linspace(lo, hi, n + 2)[1:-1]
    return np.array([x for x in pts if all(abs(x - k) > 1e-3 for k in KINKS.get(pid, ()))])


def test_required_problems_registered():
    for pid in REQUIRED:
        assert problems.kind_of(pid) == "scalar_min"
    assert set(REQUIRED) <= set(ALL)


@pytest.mark.parametrize("pid", ALL)
def test_metadata(pid):
    p = problems.get(pid)
    assert p.dim == 1
    assert callable(p.f) and callable(p.grad) and callable(p.hess)
    lo, hi = p.domain
    a, b = p.bracket
    assert lo <= a < b <= hi, "the bracket must lie inside the plotting domain"
    assert lo <= p.x0 <= hi
    assert p.minima, "every problem lists its minimizers"
    for xm in p.minima:
        assert lo < xm < hi
    # At least one listed minimizer lies in the bracket.
    assert any(a <= xm <= b for xm in p.minima)
    json.dumps(p.to_dict(), allow_nan=False)
    assert isinstance(p.f(float(p.x0)), float)


@pytest.mark.parametrize("pid", ALL)
def test_gradient_matches_finite_differences(pid):
    p = problems.get(pid)
    for x in interior_points(pid):
        g, fd = p.grad(float(x)), central(p.f, float(x))
        # Central-difference error ≈ ε^{2/3}·(|f| + |f'''|); 1e-7 is ~100x above it.
        assert_allclose(g, fd, rtol=1e-7, atol=1e-7 * (1 + abs(p.f(float(x)))))


@pytest.mark.parametrize("pid", ALL)
def test_hessian_matches_finite_differences_of_gradient(pid):
    p = problems.get(pid)
    for x in interior_points(pid):
        h, fd = p.hess(float(x)), central(p.grad, float(x))
        assert_allclose(h, fd, rtol=1e-7, atol=1e-7 * (1 + abs(p.grad(float(x)))))


@pytest.mark.parametrize("pid", sorted(set(ALL) - {"abs_shifted", "flat_valley"}))
def test_minima_are_strict_local_minimizers(pid):
    p = problems.get(pid)
    for xm in p.minima:
        # |f'(x*)| at the rounding level of f' near x* (f' is O(1..10) on these problems).
        assert abs(p.grad(xm)) <= 1e-13 * max(1.0, abs(p.hess(xm)) * abs(xm)) * 10
        assert p.hess(xm) > 0.0
        # Strict local minimum on a small neighbourhood.
        for d in (1e-3, -1e-3, 1e-2, -1e-2):
            assert p.f(xm + d) > p.f(xm)


def test_abs_shifted_subgradient_condition():
    p = problems.get("abs_shifted")
    (xs,) = p.minima
    # 0 ∈ ∂f(x*) = [-1, 1] + x*  ⇔  |x*| ≤ 1.
    assert abs(xs) <= 1.0
    assert p.grad(xs) == 0.0  # the minimum-norm subgradient
    assert p.grad(xs - 1e-9) < 0.0 < p.grad(xs + 1e-9)
    assert math.isclose(p.f(xs), 0.5 * 0.3**2)


def test_flat_valley_is_flat_and_derivatives_never_nan():
    p = problems.get("flat_valley")
    assert p.f(0.0) == 0.0 and p.grad(0.0) == 0.0 and p.hess(0.0) == 0.0
    for x in (5e-324, 1e-300, 1e-200, 1e-20, 0.01, 0.036, -0.036, -1e-200):
        assert p.f(x) == 0.0  # underflow: exp(-1/x²) < smallest subnormal
        assert p.grad(x) == 0.0 and p.hess(x) == 0.0  # log form avoids inf·0 = nan
    assert p.f(0.037) > 0.0
    # The gradient test |f'| ≤ 1e-8 already passes at x = 0.2.
    assert abs(p.grad(0.2)) < 1e-8 < abs(p.grad(0.25))


@pytest.mark.parametrize("pid", ALL)
def test_listed_global_minimizer_beats_dense_grid(pid):
    p = problems.get(pid)
    lo, hi = p.domain
    grid = np.linspace(lo, hi, 20001)
    fg = np.array([p.f(float(x)) for x in grid])
    f_star = p.f(p.minima[0])
    # The first listed minimizer is the global one on the domain (end points excluded for
    # sin_1d, whose f(0) = f(2π) = 0 > f(3π/2)).
    assert f_star <= np.nanmin(fg) + 1e-12
    for xm in p.minima[1:]:
        assert p.f(xm) > f_star


def test_closed_forms():
    q = problems.get("quartic_1d")
    for x in (
        scalar_min.QUARTIC_GLOBAL_MIN,
        scalar_min.QUARTIC_LOCAL_MIN,
        scalar_min.QUARTIC_LOCAL_MAX,
    ):
        assert abs(4 * x**3 - 8 * x + 1) < 1e-13
    assert q.hess(scalar_min.QUARTIC_LOCAL_MAX) < 0
    np.testing.assert_allclose(
        sorted(np.roots([4, 0, -8, 1]).real),
        sorted((*q.minima, scalar_min.QUARTIC_LOCAL_MAX)),
        rtol=1e-13,
    )
    r = problems.get("rational_1d")
    assert math.isclose(r.minima[0], math.sqrt(6) - 1, rel_tol=1e-15)
    assert math.isclose(r.f(r.minima[0]), 2 * math.sqrt(6) - 4, rel_tol=1e-14)
    x = problems.get("x_log_x")
    assert math.isclose(x.f(math.exp(-1)), -math.exp(-1), rel_tol=1e-15)
    d = problems.get("drug_concentration")
    assert math.isclose(d.minima[0], math.log(5) / 0.8, rel_tol=1e-15)
    assert math.isclose(-d.f(d.minima[0]), 5.349922439811, rel_tol=1e-11)
    # The model stops being convex at the inflection point t = 2 t*.
    assert abs(d.hess(2 * d.minima[0])) < 1e-14
    s = problems.get("sin_1d")
    assert s.f(s.minima[0]) == -1.0


def test_values_outside_the_domain_are_nan_not_errors():
    assert math.isnan(problems.get("x_log_x").f(-0.5))
    assert math.isnan(problems.get("x_log_x").grad(-0.5))
    assert problems.get("x_log_x").f(0.0) == 0.0
    assert math.isnan(problems.get("rational_1d").f(-1.0))
    assert math.isnan(problems.get("rational_1d").hess(-1.0))


@pytest.mark.parametrize("pid", sorted(set(ALL) - {"flat_valley"}))
def test_scipy_bounded_finds_the_listed_minimizer_in_the_bracket(pid):
    """Oracle: SciPy's fminbound on the default bracket lands on a listed minimizer."""
    p = problems.get(pid)
    res: Any = minimize_scalar(p.f, bounds=p.bracket, method="bounded", options={"xatol": 1e-10})
    assert min(abs(res.x - xm) for xm in p.minima) < 1e-7


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(sorted(set(ALL) - {"abs_shifted"})), st.floats(0.0, 1.0))
def test_derivatives_property(pid, t):
    """Hypothesis: f' and f'' match central differences anywhere inside the domain."""
    p = problems.get(pid)
    lo, hi = p.domain
    x = lo + (0.02 + 0.96 * t) * (hi - lo)
    if pid == "flat_valley" and abs(x) < 0.05:
        return  # f underflows to 0 here; FD of exactly-zero values carries no information
    scale_f = 1 + abs(p.f(x))
    assert_allclose(p.grad(x), central(p.f, x), rtol=1e-6, atol=1e-6 * scale_f)
    assert_allclose(p.hess(x), central(p.grad, x), rtol=1e-6, atol=1e-6 * (1 + abs(p.grad(x))))
