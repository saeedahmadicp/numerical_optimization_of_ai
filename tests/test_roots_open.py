"""Tests of the open root finders (newton, secant, halley, steffensen, muller,
inverse_quadratic_interpolation, fixed_point)."""

from __future__ import annotations

import math
import sys
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy import optimize

import numopt
from numopt import problems
from numopt.core.types import Problem, Step
from numopt.roots import open as open_mod

EPS = sys.float_info.epsilon
METHODS = (
    "newton",
    "secant",
    "halley",
    "steffensen",
    "muller",
    "inverse_quadratic_interpolation",
    "fixed_point",
)
#: (method, problem, params) that must converge from the problem's default x0.
CONVERGING = [
    ("newton", "sqrt2", {}),
    ("newton", "cubic", {}),
    ("newton", "kepler", {}),
    ("newton", "double_root", {}),
    ("newton", "wilkinson5", {}),
    ("secant", "cubic", {}),
    ("secant", "kepler", {}),
    ("secant", "newton_cycle", {}),
    ("halley", "x10_minus_1", {}),
    ("halley", "atan_newton", {}),
    ("halley", "wilkinson5", {}),
    ("steffensen", "sqrt2", {}),
    ("steffensen", "cos_minus_x", {}),
    ("steffensen", "kepler", {}),
    ("muller", "cos_minus_x", {}),
    ("muller", "kepler", {}),
    ("muller", "atan_newton", {}),
    ("inverse_quadratic_interpolation", "cubic", {}),
    ("inverse_quadratic_interpolation", "steep_exp", {}),
    ("fixed_point", "cos_minus_x", {"lam": -1.0}),
    ("fixed_point", "sqrt2", {"lam": 0.3}),
    ("fixed_point", "cubic", {}),
]


def _counting(p: Problem) -> tuple[Problem, dict[str, int]]:
    """Copy of p with independent counters on f, f' and f''."""
    n = {"f": 0, "g": 0, "h": 0}

    def f(x):
        n["f"] += 1
        return p.f(x)

    def g(x):
        n["g"] += 1
        return p.grad(x)  # type: ignore[misc]

    def h(x):
        n["h"] += 1
        return p.hess(x)  # type: ignore[misc]

    q = Problem(p.id, p.name, p.latex, f, 1, p.domain, grad=g, hess=h, x0=p.x0, roots=p.roots)
    return q, n


@pytest.mark.parametrize(("method", "pid", "params"), CONVERGING)
def test_converges_to_a_listed_root_with_exact_counts(method, pid, params):
    p, n = _counting(problems.get(pid))
    res = numopt.run(method, p, **params)
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    root = min(p.roots, key=lambda r: abs(r - res.x))
    # NOTE: at the double root of (x−1)²(x+2) Newton-type steps converge linearly (rate ½
    # for Newton), so the step test bounds the error by about one step: allow 2·tol.
    assert abs(res.x - root) <= 2 * (1e-10 + 2 * EPS * abs(root)), (res.x, root)
    assert (res.n_fev, res.n_gev, res.n_hev) == (n["f"], n["g"], n["h"])
    assert res.trace[-1].x == res.x and res.trace[-1].fun == res.fun == p.f(res.x)
    for s in res.trace[1:]:
        assert s.info["previous"] is not None
        assert s.step_size == abs(s.x - s.info["previous"])


ORACLE_PROBLEMS = [p.id for p in problems.list_problems("roots")]


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # arctan, |x| → ∞
@pytest.mark.parametrize("pid", ORACLE_PROBLEMS)
def test_newton_iterates_equal_scipy_newton(pid):
    """Oracle: scipy.optimize.newton evaluates f at exactly the same points."""
    p = problems.get(pid)
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return p.f(x)

    try:
        optimize.newton(f, p.x0, fprime=p.grad, tol=1e-10, rtol=0.0, maxiter=100)
    except (RuntimeError, OverflowError):
        pass
    res = numopt.run("newton", p)
    mine = [s.x for s in res.trace]
    n = min(len(mine), len(calls))
    assert n >= 3
    assert mine[:n] == calls[:n]


@pytest.mark.parametrize(
    "pid",
    ["sqrt2", "cubic", "cos_minus_x", "x10_minus_1", "double_root", "steep_exp", "wilkinson5"],
)
def test_halley_iterates_equal_scipy_halley(pid):
    """Oracle: SciPy's Halley (newton with fprime2). SciPy damps the step when
    |f f''/(2f'²)| ≥ 1; on these problems that safeguard never fires."""
    p = problems.get(pid)
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return p.f(x)

    ref = optimize.newton(f, p.x0, fprime=p.grad, fprime2=p.hess, tol=1e-10, rtol=0.0)
    res = numopt.run("halley", p)
    mine = [s.x for s in res.trace]
    n = min(len(mine), len(calls))
    np.testing.assert_allclose(mine[:n], calls[:n], rtol=4 * EPS, atol=0)
    assert abs(res.x - ref) <= 1e-10


@pytest.mark.parametrize("pid", ["sqrt2", "cubic", "cos_minus_x", "kepler", "wilkinson5"])
def test_secant_root_equals_scipy_secant(pid):
    """Oracle: SciPy's secant (it reorders the two start points, so compare roots)."""
    p = problems.get(pid)
    ref = optimize.newton(p.f, p.x0 + 0.1, x1=p.x0, tol=1e-12, rtol=0.0, maxiter=100)
    res = numopt.run("secant", p, xtol=1e-12)
    assert res.converged
    assert abs(res.x - ref) <= 2 * (1e-12 + 2 * EPS * abs(ref))


@pytest.mark.parametrize("pid", ["sqrt2", "cubic", "cos_minus_x", "kepler", "newton_cycle"])
def test_steffensen_equals_scipy_del2(pid):
    """Oracle: Steffensen = Aitken Δ² on g(x) = x + f(x) (scipy.optimize.fixed_point, del2)."""
    p = problems.get(pid)
    ref = float(optimize.fixed_point(lambda x: x + p.f(x), p.x0, xtol=1e-13, maxiter=500))
    res = numopt.run("steffensen", p, xtol=1e-12)
    assert res.converged
    assert abs(res.x - ref) <= 2 * (1e-12 + 2 * EPS * abs(ref))


def test_fixed_point_equals_scipy_iteration_and_its_linear_rate():
    ref = float(optimize.fixed_point(math.cos, 1.0, xtol=1e-14, method="iteration", maxiter=2000))
    res = numopt.run("fixed_point", problems.get("cos_minus_x"), lam=-1.0, xtol=1e-12)
    assert res.converged and abs(res.x - ref) <= 2e-12
    # g(x) = cos x, so x_{k+1} = cos x_k exactly and the rate is g'(r) = −sin r.
    xs = [s.x for s in res.trace]
    assert all(b == math.cos(a) for a, b in pairwise(xs))
    # e_{k+1}/e_k = g'(ξ_k) differs from g'(r) by ≈ |g''|·e_k: use iterates with e ≲ 1e-5.
    e = [x - ref for x in xs[25:45]]
    rates = [b / a for a, b in pairwise(e)]
    np.testing.assert_allclose(rates, -math.sin(ref), rtol=1e-3)


def test_fixed_point_slow_alternating_convergence_is_not_a_cycle():
    """g(x) = −0.99x: the iterates alternate and creep in; no false cycle, error ≤ tol."""
    res = numopt.find_root(lambda x: x, x0=1.0, method="fixed_point", lam=1.99, max_iter=5000)
    assert res.converged and "cycle_period" not in res.extra
    assert abs(res.x) <= 1e-10
    assert_valid_result(res, max_iter=5000)


def test_newton_converges_quadratically():
    p = problems.get("cubic")
    r = p.roots[0]
    res = numopt.run("newton", p, xtol=1e-14)
    e = [abs(s.x - r) for s in res.trace if abs(s.x - r) > 1e-14]
    c = p.hess(r) / (2 * p.grad(r))  # e_{k+1} ≈ |f''/2f'|·e_k²
    for a, b in pairwise(e[1:]):
        assert b <= 1.5 * abs(c) * a * a + 4 * EPS


def test_newton_is_linear_at_the_double_root():
    res = numopt.run("newton", problems.get("double_root"))
    e = [abs(s.x - 1.0) for s in res.trace]
    rates = [b / a for a, b in pairwise(e[5:21])]
    np.testing.assert_allclose(rates, 0.5, rtol=1e-2)


def test_tangent_hyperbola_parabola_geometry():
    p = problems.get("kepler")
    for s in numopt.run("newton", p).trace[1:]:
        (x, y), m = s.info["tangent"]["point"], s.info["tangent"]["slope"]
        assert (x, y, m) == (s.info["previous"], p.f(x), p.grad(x))
        assert math.isclose(x - y / m, s.x, rel_tol=4 * EPS)
    for s in numopt.run("halley", p).trace[1:]:
        h = s.info["hyperbola"]
        c, a, b, g = h["center"], h["alpha"], h["beta"], h["gamma"]
        f0, f1, f2 = s.info["derivatives"]
        assert math.isclose(a / g, f0, rel_tol=1e-12, abs_tol=1e-15)  # y(c) = f
        assert math.isclose((g - a * b) / g**2, f1, rel_tol=1e-12)  # y'(c) = f'
        assert math.isclose(-2 * b * (g - a * b) / g**3, f2, rel_tol=1e-10, abs_tol=1e-14)
        assert c - a == s.x
    for s in numopt.run("muller", p).trace[1:]:
        q = s.info["parabola"]
        for x, y in s.info["points"]:
            t = x - q["center"]
            assert math.isclose(
                q["a"] * t * t + q["b"] * t + q["c"], y, rel_tol=1e-9, abs_tol=1e-12
            )
        t = s.x - q["center"]
        assert abs(q["a"] * t * t + q["b"] * t + q["c"]) <= 1e-12 * (1 + abs(q["c"]))
    for s in numopt.run("inverse_quadratic_interpolation", p).trace[1:]:
        q = s.info["inverse_parabola"]
        for x, y in s.info["points"]:
            assert math.isclose(q["a"] * y * y + q["b"] * y + q["c"], x, rel_tol=1e-9)
        assert math.isclose(q["c"], s.x, rel_tol=1e-9)
    for s in numopt.run("secant", p).trace[1:]:
        (x0, y0), (x1, y1) = s.info["chord"]
        assert math.isclose(x1 - y1 * (x1 - x0) / (y1 - y0), s.x, rel_tol=4 * EPS)
    for s in numopt.run("steffensen", p).trace[1:]:
        (x0, y0), (z, fz) = s.info["chord"]
        # The slope uses the probe actually taken, z − x (= f(x) up to the rounding of z).
        assert z == x0 + y0
        assert math.isclose(s.info["slope"], (fz - y0) / (z - x0), rel_tol=4 * EPS)
    for s in numopt.run("fixed_point", p, lam=1.0).trace[1:]:
        (a0, a1), (b0, b1), (c0, c1) = s.info["cobweb"]
        assert a0 == a1 == b0 and b1 == c0 == c1 == s.x


def test_auxiliary_points_are_reported_at_step_zero():
    """h = max(delta, √ε|x₀|) is the plain delta for moderate x₀ (cubic: x₀ = 2)."""
    p = problems.get("cubic")
    assert numopt.run("secant", p, delta=0.25).trace[0].info["auxiliary"][0][0] == 2.25
    for method in ("muller", "inverse_quadratic_interpolation"):
        aux = numopt.run(method, p, delta=0.25).trace[0].info["auxiliary"]
        assert [a[0] for a in aux] == [1.75, 2.25]
    res = numopt.find_root(lambda x: x / 3e16 - 1.0, x0=1e16, method="secant")
    h = math.sqrt(EPS) * 1e16
    assert res.trace[0].info["auxiliary"][0][0] == 1e16 + h
    for method in METHODS:
        assert numopt.run(method, p).trace[0].x == p.x0


# --- failure paths -------------------------------------------------------------------


def test_newton_diverges_on_arctan_beyond_the_critical_start():
    p = problems.get("atan_newton")
    res = numopt.run("newton", p)
    assert not res.converged and "diverged" in res.message
    assert_valid_result(res)
    xs = [s.x for s in res.trace]
    assert all(abs(b) > abs(a) and a * b < 0 for a, b in pairwise(xs))


def test_newton_detects_the_two_cycle():
    res = numopt.run("newton", problems.get("newton_cycle"))
    assert not res.converged and "cycle" in res.message
    assert res.extra["cycle_period"] == 2 and res.n_iter == 2
    assert [s.x for s in res.trace] == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_zero_derivative_breaks_down(method):
    res = numopt.run(method, problems.get("sqrt2"), x0=0.0)
    assert not res.converged and "f'(x) = 0" in res.message
    assert_valid_result(res)


def test_secant_and_iqi_break_down_on_equal_values():
    res = numopt.find_root(lambda x: x * x - 1, x0=-0.05, delta=0.1, method="secant")
    assert not res.converged and "horizontal chord" in res.message
    res = numopt.find_root(
        lambda x: x * x - 1, x0=0.0, delta=0.1, method="inverse_quadratic_interpolation"
    )
    assert not res.converged and "equal f values" in res.message


def test_muller_stops_on_a_complex_step():
    res = numopt.find_root(lambda x: x * x + 1, x0=0.5, method="muller")
    assert not res.converged and "no real root" in res.message
    assert res.n_iter == 0
    assert_valid_result(res)


def test_steffensen_does_not_claim_a_false_convergence():
    """From x₀ = 4 on eˣ − 10 the probe is 44.6 long, the slope estimate is ~3e19 and the
    step is below the float spacing. SciPy's del2 returns 4.0 here; we report a stall."""
    p = problems.get("steep_exp")
    res = numopt.run("steffensen", p)
    assert not res.converged and "stalled" in res.message
    for x0 in (3.5, 3.6, 3.8):
        res = numopt.run("steffensen", p, x0=x0)
        assert not res.converged or abs(res.x - math.log(10)) <= 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_is_reported(method):
    res = numopt.run(method, problems.get("kepler"), max_iter=2, xtol=1e-15)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=2)


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_values_stop_the_run(method):
    def f(x: float) -> float:
        return math.nan if x < 0.75 else x - 0.5

    extra: dict[str, Any] = {"lam": 4.0} if method == "fixed_point" else {}
    res = numopt.find_root(f, x0=1.0, method=method, fprime=lambda x: 4.0, **extra)
    assert not res.converged
    assert_valid_result(res)


def test_finite_difference_fallback_counts():
    res = numopt.find_root(lambda x: x**3 - 2 * x - 5, x0=2.0, method="newton")
    assert res.converged and res.extra["derivatives"] == "finite_difference"
    assert res.n_gev == 0 and res.n_fev == 1 + 3 * res.n_iter
    res = numopt.find_root(lambda x: x**3 - 2 * x - 5, x0=2.0, method="halley")
    assert res.converged and res.n_fev == 1 + 6 * res.n_iter  # f, f' (2) and f'' (3)


def test_exact_root_at_the_start():
    for method in METHODS:
        res = numopt.find_root(lambda x: x - 2.0, x0=2.0, method=method, fprime=lambda x: 1.0)
        assert res.converged and res.n_iter == 0 and res.x == 2.0


# --- property tests ---------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(
        [
            "newton",
            "secant",
            "halley",
            "steffensen",
            "muller",
            "inverse_quadratic_interpolation",
        ]
    ),
    st.floats(-100, 100),
    st.floats(0.1, 10) | st.floats(-10, -0.1),
    st.floats(-1, 1),
)
def test_property_one_step_exactness(method, r, slope, offset):
    """Each method is exact on the functions its model reproduces:
    lines (Newton, secant, Steffensen), Möbius maps (Halley), parabolas (Müller) and
    functions with a quadratic inverse (IQI); the first step lands on the root."""
    x0 = r + offset
    if method in ("newton", "secant", "steffensen"):
        f, df = (lambda x: slope * (x - r)), (lambda x: slope)
        res = numopt.find_root(f, x0=x0, method=method, fprime=df)
    elif method == "halley":
        # y = (x − r)/(0.5(x − r) + 2): its only pole is 4 units left of r, x0 is within 1.
        f = lambda x: (x - r) / (0.5 * (x - r) + 2.0)  # noqa: E731
        df = lambda x: 2.0 / (0.5 * (x - r) + 2.0) ** 2  # noqa: E731
        d2f = lambda x: -2.0 / (0.5 * (x - r) + 2.0) ** 3  # noqa: E731
        res = numopt.find_root(f, x0=x0, method=method, fprime=df, fprime2=d2f)
    elif method == "muller":
        f = lambda x: slope * (x - r) * (x - r - 5.0)  # noqa: E731  roots r and r + 5
        res = numopt.find_root(f, x0=x0, method=method)
    else:
        # x(y) = r + y + y²/(4·10) ⇔ y = 20(√(1 + (x − r)/10) − 1): inverse is quadratic.
        # Written as 2t/(√(1+t) + 1), t = (x − r)/10, to avoid cancellation in √(1+t) − 1.
        def f(x: float) -> float:
            t = (x - r) / 10.0
            return 20.0 * t / (math.sqrt(1.0 + t) + 1.0)

        res = numopt.find_root(f, x0=x0, method=method)
    if res.n_iter == 0:  # x0 hit the root exactly
        assert res.fun == 0.0
        return
    x1 = res.trace[1].x
    bound = 64 * EPS * (1 + abs(r))
    if method == "muller":
        # NOTE: perturbing the data fᵢ by δᵢ moves the parabola's root by Σ Lᵢ(x₁)δᵢ / p'(x₁)
        # (Lᵢ: Lagrange basis at the three points), so rounding of f is amplified by that.
        pts = res.trace[1].info["points"]
        q = res.trace[1].info["parabola"]
        slope_at_root = abs(2 * q["a"] * (x1 - q["center"]) + q["b"])
        sens = 0.0
        for i, (xi, yi) in enumerate(pts):
            li = 1.0
            for j, (xj, _) in enumerate(pts):
                if j != i:
                    li *= (x1 - xj) / (xi - xj)
            sens += abs(li) * abs(yi)
        bound += 16 * EPS * sens / slope_at_root
    if method == "inverse_quadratic_interpolation":
        # NOTE: extrapolating x(y) to y = 0 amplifies the rounding of the data by the
        # Lebesgue sum Σ|L_i(0)| (≈ 100 when the three f values sit near 1, 0.1 apart).
        (xa, ya), (xb, yb), (xc, yc) = res.trace[1].info["points"]
        lam = (
            abs(yb * yc / ((ya - yb) * (ya - yc)))
            + abs(ya * yc / ((yb - ya) * (yb - yc)))
            + abs(ya * yb / ((yc - ya) * (yc - yb)))
        )
        bound += 16 * EPS * lam * (max(abs(xa), abs(xb), abs(xc)) + max(abs(ya), abs(yb), abs(yc)))
    assert abs(x1 - r) <= bound, (method, x1, r)
    # The exact x₁ can coincide with an auxiliary start point (e.g. x₀ = −0.1, δ = 0.1:
    # x₀ + δ rounds to the root); the next interpolation is then degenerate and the
    # method reports that breakdown instead of converging.
    assert res.converged or "coincide" in res.message or "equal f values" in res.message


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(METHODS), st.floats(0.5, 3.0), st.floats(-0.3, 0.3))
def test_property_result_contract(method, r, offset):
    """f(x) = (x − r)(x + 4): every run satisfies the contract and a converged run is at a root."""

    def f(x: float) -> float:
        return (x - r) * (x + 4.0)

    extra: dict[str, Any] = {"lam": 0.1} if method == "fixed_point" else {}
    res = numopt.find_root(f, x0=r + offset, method=method, fprime=lambda x: 2 * x + 4 - r, **extra)
    assert_valid_result(res, max_iter=100)
    assert len(res.trace) == res.n_iter + 1
    if res.converged:
        assert min(abs(res.x - r), abs(res.x + 4.0)) <= 1e-8 * (1 + abs(r))


@pytest.mark.parametrize(("method", "pid", "params"), open_mod.FIXTURE_CASES)
def test_fixture_cases_run_and_serialize(method, pid, params):
    """The parity fixtures exported for the web app: valid, JSON-clean, < 300 steps."""
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300
    assert numopt.get_method(method).family == "roots"


def test_steffensen_when_the_probe_rounds_to_x():
    """|f(x)| below half an ulp: no new slope. A sign probe at x ± tol toward the root
    predicted by the last slope judges x (one extra, counted evaluation)."""
    # f(x) = 0.1(x − 0.7) from x₀ = 2: one step lands 2 ulps below 0.7 (|f| ≈ 4e-17).
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return 0.1 * (x - 0.7)

    res = numopt.find_root(f, x0=2.0, method="steffensen")
    assert res.converged and "rounds to x" in res.message and abs(res.x - 0.7) <= 1e-15
    assert res.n_fev == len(calls) and res.n_iter == 1
    x_p, f_p = res.trace[-1].info["slope_probe"]
    assert res.x < x_p <= res.x + _tol(res.x)  # toward the root, within tol
    assert res.fun is not None and f_p * res.fun < 0
    # At the double root the last slope is ≈ 6(x − 1) ≈ 0, so |f/s| is far above tol.
    res = numopt.run("steffensen", problems.get("double_root"))
    assert not res.converged and "rounds to x" in res.message
    assert_valid_result(res)


# --- regression tests: no false convergence, no exceptions -----------------------------


def _tol(x: float, xtol: float = 1e-10) -> float:
    return xtol + 2 * EPS * abs(x)


def _size(s: Step) -> float:
    assert s.step_size is not None
    return s.step_size


def _grid_cases() -> list[tuple[str, str, dict[str, Any]]]:
    cases = []
    for p in problems.list_problems("roots"):
        for method in METHODS:
            for extra in (
                [{}] if method != "fixed_point" else [{"lam": v} for v in (0.1, -1.0, 1.0)]
            ):
                cases.append((method, p.id, extra))
    return cases


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("method", "pid", "extra"), _grid_cases())
def test_open_methods_never_raise_and_never_converge_falsely(method, pid, extra):
    """Every open method from a grid of starts over three widths of the plotting domain:
    no exception (math.exp / x**10 used to raise OverflowError), a valid Result, and a
    converged x within 2·tol of a listed root (the lists hold every real root)."""
    p = problems.get(pid)
    lo, hi = p.domain
    for x0 in np.linspace(lo - (hi - lo), hi + (hi - lo), 61):
        res = numopt.run(method, p, x0=float(x0), **extra)
        assert_valid_result(res, max_iter=100)
        if res.converged:
            root = min(p.roots, key=lambda r: abs(r - res.x))
            assert abs(res.x - root) <= 2 * _tol(root), (float(x0), res.x, res.message)


@pytest.mark.parametrize("multiplicity", [2, 3, 4, 5, 8])
def test_newton_at_a_multiple_root_meets_the_tolerance(multiplicity):
    """Linear rate 1 − 1/m: the error is (m − 1)·step. The plain step test stopped at 4·tol
    for m = 5; the a posteriori factor ρ/(1 − ρ) = m − 1 keeps it within tol."""
    m = multiplicity
    res = numopt.find_root(
        lambda x: (x - 1.0) ** m,
        x0=2.0,
        fprime=lambda x: m * (x - 1.0) ** (m - 1),
        method="newton",
        max_iter=500,
    )
    assert res.converged and "step test" in res.message
    assert abs(res.x - 1.0) <= _tol(1.0)
    # The Newton map of (x − 1)^m is exactly e ↦ (1 − 1/m)e. Early steps (e ≥ 1e-3) keep the
    # rounding of x − 1 (≈ ε/e relative) far below the tolerance.
    rho = [_size(s) / _size(p) for p, s in pairwise(res.trace[1:12])]
    np.testing.assert_allclose(rho, 1 - 1 / m, rtol=1e-10)


def test_newton_near_a_pole_is_not_converged_after_one_tiny_step():
    """f = 1/x − 1 from x₀ = 1e−11: |f/f'| = 1e−11 ≤ tol while f = 1e11. The plain step test
    returned x = 2e−11; the a posteriori test needs a decreasing step (here they double)."""
    res = numopt.find_root(
        lambda x: 1.0 / x - 1.0, x0=1e-11, fprime=lambda x: -1.0 / (x * x), method="newton"
    )
    assert res.n_iter > 30
    assert res.converged and abs(res.x - 1.0) <= 2 * _tol(1.0)
    steps = [_size(s) for s in res.trace[1:10]]
    assert all(b > a for a, b in pairwise(steps))


def test_steffensen_on_a_steeply_scaled_function_converges_at_the_root():
    """f = 1e6(x² − 2): |f| one float spacing from √2 is 4.4e−10 > tol, so the probe test
    can never pass; the iterates settle on the floats around √2 and the sign change between
    them proves a root within tol. It used to report a false 'cycle'."""
    res = numopt.find_root(
        lambda x: 1e6 * (x * x - 2.0), x0=math.sqrt(2) + 1e-6, method="steffensen"
    )
    assert res.converged and "changes sign" in res.message
    assert abs(res.x - math.sqrt(2)) <= 2 * _tol(math.sqrt(2))
    prev = res.trace[-2]
    assert prev.fun is not None and res.fun is not None and prev.fun * res.fun < 0


def test_steffensen_step_does_not_underflow():
    """f(x) = x from x₀ = 3e−288: f²/(f(x + f) − f) underflows to 0 and froze x; f·(f/den)
    does not."""
    res = numopt.find_root(lambda x: x, x0=2.9678379414886136e-288, method="steffensen")
    assert res.converged and res.x == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_exceptions_in_f_and_derivatives_become_non_finite(method):
    def f(x: float) -> float:
        return math.exp(x) - 10.0 if x < 50 else math.exp(1e6)  # OverflowError beyond 50

    extra: dict[str, Any] = {"lam": 100.0} if method == "fixed_point" else {}
    res = numopt.find_root(f, x0=-5.0, method=method, fprime=math.exp, fprime2=math.exp, **extra)
    assert_valid_result(res)
    if res.converged:
        assert abs(res.x - math.log(10)) <= 2 * _tol(math.log(10))
    res = numopt.find_root(
        lambda x: x - 1.0, x0=3.0, method="newton", fprime=lambda x: math.log(-x)
    )
    assert not res.converged and "f'(x) is not finite" in res.message
    res = numopt.find_root(f, x0=1e3, method=method, fprime=math.exp, **extra)
    assert not res.converged and "OverflowError" in res.message
    with pytest.raises(TypeError):  # a bug in f is not swallowed
        numopt.find_root(lambda x: x + "1", x0=1.0, method=method)  # type: ignore[operator]


@settings(max_examples=2000, deadline=None)
@given(
    st.sampled_from(METHODS),
    st.sampled_from(["exp", "pole", "multiple"]),
    st.floats(-3.0, 3.0),
    st.floats(0.1, 30.0),
    st.integers(2, 6),
    st.floats(-12.0, -4.0).map(lambda e: 10.0**e),
)
# Shrunk counterexample: a zero step whose forward slope probe pointed away from the root
# accepted x = 1 + 2⁻²³ at error 3.8·tol (see test_zero_step_without_a_local_slope_*).
@example("secant", "multiple", 2.0**-23, 1.0, 3, 10.0**-7.5)
def test_property_no_false_convergence(method, family, offset, c, m, xtol):
    """converged ⇒ x within 2·tol of the root, on steep exponentials e^{c(x−1)} − 1, the
    pole-like f = 1/x − 1 and multiple roots (x − 1)^m, from starts around the root, with
    xtol ∈ [1e-12, 1e-4]. At xtol = 1e-10 alone Steffensen never converged at (x − 1)^m, so
    the multiple-root family was vacuous for it; at 1e-6 it returned 4–6·tol errors.
    Halley gets f' only, so its f'' is a finite difference (the audit's 5.4·tol case)."""
    if family == "exp":

        def f(x: float) -> float:
            return math.expm1(c * (x - 1.0))

        def df(x: float) -> float:
            return c * math.exp(c * (x - 1.0))

    elif family == "pole":

        def f(x: float) -> float:
            return 1.0 / x - 1.0

        def df(x: float) -> float:
            return -1.0 / (x * x)

    else:

        def f(x: float) -> float:
            return (x - 1.0) ** m

        def df(x: float) -> float:
            return m * (x - 1.0) ** (m - 1)

    x0 = 1.0 + offset if family != "pole" else abs(offset) / 10 + 1e-12
    extra: dict[str, Any] = {"lam": 0.5 / c} if method == "fixed_point" else {}
    res = numopt.find_root(f, x0=x0, method=method, fprime=df, max_iter=300, xtol=xtol, **extra)
    assert_valid_result(res, max_iter=300)
    if res.converged:
        assert abs(res.x - 1.0) <= 2 * _tol(1.0, xtol), (res.x, xtol, res.message)


# --- regression tests: multiple roots, noise floor, auxiliary offset ---------------------


def _power(m: int) -> tuple[Any, Any]:
    return (lambda x: (x - 1.0) ** m), (lambda x: m * (x - 1.0) ** (m - 1))


@pytest.mark.parametrize(
    ("method", "m", "x0", "xtol"),
    [
        ("steffensen", 3, 1.3, 1e-6),  # was x = 1 + 4.1e-6, "|f(x)/slope| ≤ tol"
        ("halley", 6, 2.3, 1e-6),  # was 5.4·tol: FD f'' = 4.6e-16, true 2.6e-20
        ("halley", 6, -0.7, 1e-6),  # was 4.9·tol
        ("halley", 6, 3.3, 1e-6),
    ],
)
def test_multiple_root_audit_cases_are_not_false_convergences(method, m, x0, xtol):
    """At an m-fold root the Newton/secant correction is ≈ e/m. Guards that accepted a tiny
    step on that correction alone returned errors of m·tol with converged=True."""
    f, df = _power(m)
    res = numopt.find_root(f, x0=x0, method=method, fprime=df, xtol=xtol, max_iter=300)
    assert_valid_result(res, max_iter=300)
    if res.converged:
        assert abs(res.x - 1.0) <= 2 * _tol(1.0, xtol), res.message
    else:
        assert "cycle" not in res.message, res.message


@pytest.mark.parametrize("method", ["steffensen", "halley", "newton", "secant"])
def test_multiple_roots_over_a_grid_of_starts_and_tolerances(method):
    """The audit's grid (multiplicity 2..6, 25 starts, xtol 1e-4 and 1e-6): converged ⇒
    error ≤ 2·tol (it was up to 5.9·tol for Steffensen and 5.4·tol for Halley with a
    finite-difference f''), and Steffensen does converge on part of the grid."""
    n_conv = 0
    for m in range(2, 7):
        f, df = _power(m)
        for offset in np.linspace(-3.0, 3.0, 25):
            for xtol in (1e-4, 1e-6):
                res = numopt.find_root(
                    f, x0=1.0 + float(offset), method=method, fprime=df, xtol=xtol, max_iter=300
                )
                if res.converged:
                    n_conv += 1
                    assert abs(res.x - 1.0) <= 2 * _tol(1.0, xtol), (m, offset, xtol, res.message)
    assert n_conv > 0


def test_zero_step_without_a_local_slope_needs_both_probes():
    """Regression (Hypothesis counterexample): secant on (x − 1)³ from x₀ = 1 + 2⁻²³ with
    xtol = 10^-7.5. The chord through x₀ + 0.1 rounds the step to zero; the forward probe at
    x₀ + tol points away from the root and its correction |f|·tol/|f(x₀ + tol) − f| =
    e/3.87 = 3.08e-8 ≤ tol accepted an error e = 1.19e-7 = 3.8·tol as converged. The back
    probe at x₀ − tol gives the correction 5.2e-8 > tol: the run stalls."""
    xtol = 10.0**-7.5
    f, df = _power(3)
    x0 = 1.0 + 2.0**-23
    res = numopt.find_root(f, x0=x0, method="secant", fprime=df, max_iter=300, xtol=xtol)
    assert_valid_result(res, max_iter=300)
    assert not res.converged and res.message.startswith("stalled: xₖ = xₖ₋₁"), res.message
    info = res.trace[-1].info
    tol = _tol(res.x, xtol)
    (x_f, f_f), (x_b, f_b) = info["slope_probe"], info["slope_probe_back"]
    assert res.x < x_f <= res.x + tol and res.x - tol <= x_b < res.x
    assert (f_f, f_b) == (f(x_f), f(x_b))
    assert res.n_fev == 1 + 2 + 2  # x₀, the auxiliary point, x₁ (= x₀), and the two probes


@pytest.mark.parametrize("m", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("ratio", [-3.0, -1.9, -1.2, -0.6, -0.2, 0.2, 0.6, 1.2, 1.9, 3.0, 3.77])
def test_zero_step_judgment_on_pure_powers(m, ratio):
    """Secant from x₀ = 1 + ratio·tol with a far auxiliary point (delta = 1) on (x − 1)ᵐ: the
    chord step |f(x₀)|·h/|f(x₀ + h) − f(x₀)| ≈ |e|ᵐ is far below ½ ulp, so x₁ = x₀ and the
    zero-step judgment decides alone. It must certify only |x − 1| ≤ tol (it accepted up to
    8·tol for m ≥ 6 with the forward probe alone), and a sign change at a probe certifies the
    odd multiplicities within tol."""
    xtol = 1e-7
    tol = _tol(1.0, xtol)
    f, df = _power(m)
    x0 = 1.0 + ratio * tol
    res = numopt.find_root(f, x0=x0, method="secant", fprime=df, delta=1.0, xtol=xtol)
    assert_valid_result(res)
    assert res.n_iter == 1 and res.x == x0  # the zero step
    if res.converged:
        assert abs(res.x - 1.0) <= tol, (res.message, ratio)
    else:
        assert res.message.startswith("stalled"), res.message
    if m % 2 == 1 and abs(ratio) < 1.0:
        assert res.converged and "changes sign" in res.message, res.message
    if abs(ratio) > 1.0:
        assert not res.converged, res.message


def _wilkinson10_monomial(x: float) -> float:
    """∏(x − i), i = 1..10, expanded and evaluated by Horner: rounding noise dominates f
    within ~1e-10 of each root (|f| ≈ 1e-7 there), the textbook noisy polynomial."""
    s = 0.0
    for a in np.poly(np.arange(1, 11)).tolist():
        s = s * x + a
    return s


@pytest.mark.parametrize(("method", "x0"), [("secant", 5.95), ("newton", 6.05)])
def test_noise_floor_is_not_reported_as_a_cycle(method, x0):
    """The iterates jitter within tol of the root 6 at the noise floor of f. This used to
    end in 'cycle of period 2/3' with converged=False although x was within xtol."""
    res = numopt.find_root(_wilkinson10_monomial, x0=x0, method=method)
    assert_valid_result(res)
    assert res.converged and "cycle" not in res.message, res.message
    assert "cycle_period" not in res.extra
    assert abs(res.x - 6.0) <= 2 * _tol(6.0)


@pytest.mark.parametrize(
    "method", ["newton", "secant", "halley", "muller", "inverse_quadratic_interpolation"]
)
def test_noise_floor_messages_are_certified(method):
    """Over a grid of starts on the noisy polynomial: a 'noise floor' convergence has an
    iterate within tol of x where f has the other sign (a root of the computed f within
    tol); a reported cycle has an iterate in its period farther than tol from x_k, and no
    iterate of opposite sign within the period's spread."""
    for x0 in np.linspace(0.6, 10.4, 50):
        for xtol in (1e-10, 1e-12):
            res = numopt.find_root(_wilkinson10_monomial, x0=float(x0), method=method, xtol=xtol)
            assert_valid_result(res)
            xs = [s.x for s in res.trace]
            fs = [cast(float, s.fun) for s in res.trace]
            x_k, f_k, tol = xs[-1], fs[-1], _tol(xs[-1], xtol)
            opposite = [abs(x - x_k) for x, fx in zip(xs, fs, strict=True) if fx * f_k < 0]
            if res.message.startswith("noise floor"):
                probe = res.trace[-1].info.get("slope_probe")
                if probe is not None and probe[1] * f_k < 0:  # sign probe at x_k ± tol
                    opposite.append(abs(probe[0] - x_k))
                assert res.converged and min(opposite) <= tol
            if "cycle_period" in res.extra:
                j = res.n_iter - res.extra["cycle_period"]
                spread = max(abs(x - x_k) for x in xs[j:])
                assert spread > tol
                assert not opposite or min(opposite) > spread


def test_auxiliary_offset_is_relative():
    """x₀ + 0.1 == x₀ for |x₀| ≥ 1e16: secant, Müller and IQI broke down at step 0 with
    'horizontal chord' / 'points coincide'. h = max(delta, √ε|x₀|) keeps them working."""
    for x0, root in [(1e16, 3e16), (1e18, 3e18)]:
        for method in ("secant", "muller", "inverse_quadratic_interpolation"):
            res = numopt.find_root(lambda x, root=root: x / root - 1.0, x0=x0, method=method)
            assert res.converged, (method, res.message)
            assert abs(res.x - root) <= 2 * _tol(root)


@pytest.mark.parametrize("method", ["secant", "muller", "inverse_quadratic_interpolation"])
def test_auxiliary_point_outside_the_domain_names_delta(method):
    """log(x/2e-6) from x₀ = 1e-6: x₀ − 0.1 leaves the domain (the message used to be a
    bare 'f is not finite at an auxiliary starting point'). The message names delta and
    the exception; with a smaller delta (below the ParamSpec range, allowed in the API)
    the auxiliary points are inside the domain."""
    f = lambda x: math.log(x / 2e-6)  # noqa: E731
    if method != "secant":  # secant uses only x₀ + h, which is inside the domain
        res = numopt.find_root(f, x0=1e-6, method=method)
        assert not res.converged and res.n_iter == 0
        assert "delta" in res.message and "ValueError" in res.message, res.message
    res = numopt.find_root(f, x0=1e-6, method=method, delta=5e-7)
    assert_valid_result(res)
    assert "auxiliary" not in res.message
    assert all(math.isfinite(y) for _, y in res.trace[0].info["auxiliary"])


@pytest.mark.parametrize("method", ["secant", "muller", "inverse_quadratic_interpolation"])
def test_invalid_delta_is_rejected(method):
    for delta in (0.0, -0.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="delta must be"):
            numopt.find_root(lambda x: x - 1.0, x0=2.0, method=method, delta=delta)
    # A delta below the float spacing at x₀ is raised to √ε|x₀|: the run still works.
    res = numopt.find_root(lambda x: x - 1.0, x0=2.0, method=method, delta=1e-17)
    assert res.converged and abs(res.x - 1.0) <= 2 * _tol(1.0)


def test_steffensen_zero_slope_estimate_is_judged_by_a_sign_probe():
    """f = x/4 near 0 in the subnormal range: f(x + f(x)) = f(x) after one step. x is
    within tol of the root, and the sign probe at x − tol proves it."""
    res = numopt.find_root(lambda x: 0.25 * x, x0=2.225073858507e-311, method="steffensen")
    assert res.converged and "zero slope estimate" in res.message and abs(res.x) <= 1e-300
