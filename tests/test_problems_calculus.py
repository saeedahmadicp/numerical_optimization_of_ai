"""Checks of the calculus problem library against independent oracles (SciPy, closed forms)."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import differentiate, integrate

import numopt
from numopt import problems
from numopt.core.types import Problem

REQUIRED = (
    "poly3",
    "exp_0_1",
    "sin_0_pi",
    "runge",
    "sqrt_0_1",
    "gaussian",
    "oscillatory",
    "arctan_deriv",
    "abs_kink",
)

# Points to keep away from when checking derivatives (kink, singular endpoint).
_AVOID = {"abs_kink": 0.3, "sqrt_0_1": 0.0}


def _calc(pid: str) -> Problem:
    return problems.get(pid)


def _exact(pid: str) -> float:
    value = _calc(pid).exact
    assert value is not None
    return value


def _derivs(p: Problem) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    assert p.grad is not None and p.hess is not None
    return p.grad, p.hess


def test_required_ids_are_registered_as_calculus():
    ids = {p.id for p in problems.list_problems("calculus")}
    assert set(REQUIRED) <= ids
    for pid in REQUIRED:
        assert problems.kind_of(pid) == "calculus"


@pytest.mark.parametrize("pid", REQUIRED)
def test_metadata(pid):
    p = _calc(pid)
    a, b = p.domain
    assert p.dim == 1 and a < b
    assert p.exact is not None and math.isfinite(p.exact)
    assert p.grad is not None and p.hess is not None
    assert p.x0 is not None and a <= p.x0 <= b
    p.to_dict()  # JSON-able metadata


@pytest.mark.parametrize("pid", REQUIRED)
def test_exact_integral_matches_quad(pid):
    p = _calc(pid)
    a, b = p.domain
    points = [0.3] if pid == "abs_kink" else None
    value, abserr = integrate.quad(p.f, a, b, epsabs=1e-14, epsrel=1e-13, limit=500, points=points)
    # quad's own error estimate + a few ulps of the value
    exact = _exact(pid)
    assert abs(value - exact) <= max(abserr, 4e-16 * max(1.0, abs(exact)))


def test_exact_integrals_closed_forms():
    assert _exact("poly3") == 2.0
    assert_allclose(_exact("exp_0_1"), math.e - 1.0, rtol=1e-15)
    assert _exact("sin_0_pi") == 2.0
    assert_allclose(_exact("runge"), 0.4 * math.atan(5.0), rtol=1e-15)
    assert_allclose(_exact("sqrt_0_1"), 2.0 / 3.0, rtol=1e-15)
    assert_allclose(_exact("gaussian"), math.sqrt(math.pi) * math.erf(2.0), rtol=1e-15)
    assert _exact("oscillatory") == 0.0
    assert_allclose(_exact("arctan_deriv"), math.pi / 4.0, rtol=1e-15)
    assert_allclose(_exact("abs_kink"), 0.29, rtol=1e-15)


@pytest.mark.parametrize("pid", REQUIRED)
@settings(max_examples=1000, deadline=None)
@given(u=st.floats(0.02, 0.98))
def test_derivatives_match_scipy_differentiate(pid, u):
    p = _calc(pid)
    grad, hess = _derivs(p)
    a, b = p.domain
    x = a + u * (b - a)
    avoid = _AVOID.get(pid)
    if avoid is not None and abs(x - avoid) < 0.05:
        x = avoid + 0.05 if x >= avoid else avoid - 0.05
    # scipy.differentiate.derivative: adaptive central differences with Richardson
    # extrapolation; its own error estimate bounds the comparison.
    d1 = differentiate.derivative(
        p.f, x, initial_step=0.01, tolerances={"rtol": 1e-12, "atol": 1e-10}
    )
    d2 = differentiate.derivative(
        grad, x, initial_step=0.01, tolerances={"rtol": 1e-12, "atol": 1e-10}
    )
    assert d1.success and d2.success
    assert abs(grad(x) - d1.df) <= 1e-9 * max(1.0, abs(d1.df)) + 10 * d1.error
    assert abs(hess(x) - d2.df) <= 1e-9 * max(1.0, abs(d2.df)) + 10 * d2.error


@pytest.mark.parametrize("pid", REQUIRED)
@settings(max_examples=1000, deadline=None)
@given(u=st.floats(0.02, 0.98))
def test_f_is_complex_analytic(pid, u):
    """Im f(x + ih)/h = f′(x) to rounding with h = 1e-200 (complex step needs this)."""
    p = _calc(pid)
    a, b = p.domain
    x = a + u * (b - a)
    avoid = _AVOID.get(pid)
    if avoid is not None and abs(x - avoid) < 1e-6:
        return
    h = 1e-200
    w = p.f(complex(x, h))
    assert np.iscomplexobj(w)
    # NOTE: complex x**3 may differ from real x**3 by an ulp, hence the absolute floor.
    assert_allclose(complex(w).real, float(p.f(x)), rtol=1e-14, atol=1e-15)
    assert_allclose(complex(w).imag / h, float(_derivs(p)[0](x)), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("pid", REQUIRED)
def test_vectorized_evaluation(pid):
    p = _calc(pid)
    a, b = p.domain
    xs = np.linspace(a, b, 7)[1:-1]
    for fn in (p.f, *_derivs(p)):
        out = np.asarray(fn(xs))
        assert out.shape == xs.shape
        assert_allclose(out, [float(fn(float(x))) for x in xs], rtol=1e-15)


def test_abs_kink_one_sided_complex_slopes():
    p = _calc("abs_kink")
    assert complex(p.f(0.2 + 1e-20j)).imag / 1e-20 == -1.0
    assert complex(p.f(0.4 + 1e-20j)).imag / 1e-20 == 1.0
    assert _derivs(p)[0](0.3) == 0.0  # midpoint of the subdifferential [-1, 1]


def test_sqrt_is_nan_left_of_zero():
    with np.errstate(invalid="ignore"):
        assert math.isnan(float(_calc("sqrt_0_1").f(-0.1)))


def test_oscillatory_is_odd_about_midpoint():
    p = _calc("oscillatory")
    xs = np.linspace(0.0, math.pi, 41)
    assert_allclose(p.f(math.pi - xs), -p.f(xs), atol=1e-14)


@pytest.mark.parametrize(
    ("method", "n", "order"),
    [
        ("left_riemann", 4, 1.0),
        ("right_riemann", 4, 1.0),
        ("midpoint_rule", 4, 1.5),
        ("trapezoid", 4, 1.5),
        ("simpson", 4, 1.5),
        ("simpson_38", 3, 1.5),
        ("boole", 4, 1.5),
    ],
)
def test_sqrt_observed_orders_match_the_docstring(method, n, order):
    """Audit regression: the docs said that every rule converges like h^{3/2} on √x, but the
    Riemann sums keep O(h) (slower than h^{3/2}); rules of order ≥ 2 drop to h^{3/2}."""
    res = numopt.run(method, _calc("sqrt_0_1"), n=n, levels=10)
    errs = [s.info["error"] for s in res.trace]
    assert abs(math.log2(errs[9] / errs[10]) - order) < 0.02


def test_sqrt_romberg_and_gauss_orders_match_the_docstring():
    p = _calc("sqrt_0_1")
    rom = numopt.run("romberg", p, tol=1e-300, max_levels=14)
    e = [s.info["error"] for s in rom.trace]
    assert abs(math.log2(e[13] / e[14]) - 1.5) < 0.02  # h^{3/2}, not the h^{2k+2} of row k
    gauss = numopt.run("gauss_legendre", p, n=32)
    g = [s.info["error"] for s in gauss.trace]  # g[m − 1] is the m-point error
    assert abs(math.log2(g[15] / g[31]) - 3.0) < 0.1  # n^{−3}
