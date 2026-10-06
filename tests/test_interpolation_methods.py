"""Tests for numopt.interpolation.methods (oracles: SciPy interpolators, numpy.polynomial)."""

import json
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.polynomial import chebyshev as npcheb
from numpy.testing import assert_allclose
from scipy.interpolate import BarycentricInterpolator, CubicSpline, PchipInterpolator

import numopt
from numopt import problems
from numopt.core.types import Dataset
from numopt.interpolation import methods as im

POLY = ("lagrange", "barycentric", "newton_divided_differences", "neville")
PIECEWISE = (
    "linear_spline",
    "cubic_spline_natural",
    "cubic_spline_clamped",
    "cubic_spline_not_a_knot",
    "pchip",
)
ALL = (*POLY, *PIECEWISE, "chebyshev_interpolation")
DATA_IDS = (
    "runge_equispaced",
    "runge_chebyshev",
    "sine_samples",
    "step_data",
    "noisy_linear",
    "noisy_quadratic",
    "anscombe_1",
    "outliers_linear",
    "exponential_growth",
)
EPS = np.finfo(float).eps
#: Absolute underflow term of the floating-point model, fl(a op b) = (a op b)(1+δ) + η,
#: |η| ≤ ETA (Higham, ASNA 2nd ed., §2.1). Error bounds of subnormal data need it.
ETA = float(np.nextafter(0.0, 1.0))


def _fun(res) -> float:
    assert res.fun is not None
    return float(res.fun)


def _grid_y(res):
    return np.asarray(res.extra["eval"]["y"], dtype=float)


def _grid_x(res):
    return np.asarray(res.extra["eval"]["x"], dtype=float)


# Hypothesis strategy: n distinct nodes with gaps in [gmin, gmax], shuffled or sorted.
def nodes_strategy(n_min, n_max, gmin, gmax, *, shuffle=False):
    @st.composite
    def build(draw):
        n = draw(st.integers(n_min, n_max))
        gaps = draw(st.lists(st.floats(gmin, gmax), min_size=n - 1, max_size=n - 1))
        start = draw(st.floats(-5, 5))
        x = start + np.concatenate(([0.0], np.cumsum(gaps)))
        y = np.array(draw(st.lists(st.floats(-10, 10), min_size=n, max_size=n)))
        if shuffle:
            perm = draw(st.permutations(range(n)))
            x, y = x[list(perm)], y[list(perm)]
        return x, y

    return build()


# --------------------------------------------------------------------------------------
# Contract, fixtures, JSON
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
@pytest.mark.parametrize("mid", ALL)
def test_contract_on_every_dataset(mid, pid):
    res = numopt.run(mid, problems.get(pid))
    assert_valid_result(res)
    assert res.converged, res.message
    assert res.method == mid
    extra = res.extra
    for key in ("kind", "coefficients", "nodes", "values", "domain", "eval", "node_residual"):
        assert key in extra
    assert len(extra["eval"]["x"]) == len(extra["eval"]["y"]) == im.N_GRID
    d = problems.get(pid)
    if d.f_true is None:
        assert res.fun is None and extra["eval"]["f_true"] is None
    else:
        assert res.fun == pytest.approx(np.max(np.abs(_grid_y(res) - d.f_true(_grid_x(res)))))
    assert res.n_iter == res.trace[-1].k
    # Every step of the polynomial methods carries the current curve on the grid.
    if mid in POLY or mid == "chebyshev_interpolation":
        assert all(len(s.info["curve"]) == im.N_GRID for s in res.trace)
        assert_allclose(res.trace[-1].info["curve"], _grid_y(res), rtol=1e-12, atol=1e-12)


def test_fixture_cases_cover_every_method():
    cases = im.FIXTURE_CASES
    assert {m for m, _, _ in cases} == set(ALL)
    for mid, pid, params in cases:
        res = numopt.run(mid, problems.get(pid), **params)
        assert res.converged and len(res.trace) < 300
        json.dumps(res.to_dict(), allow_nan=False)


def test_accepts_xy_pair_and_does_not_mutate():
    x = np.array([2.0, 0.0, 1.0])
    y = np.array([4.0, 0.0, 1.0])
    x0, y0 = x.copy(), y.copy()
    for mid in ALL:
        res = numopt.run(mid, (x, y))
        assert res.converged
        assert res.fun is None  # no f_true for a bare pair
    assert_allclose(x, x0, rtol=0, atol=0)
    assert_allclose(y, y0, rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# Invalid input
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mid", ALL)
def test_rejects_invalid_data(mid):
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # repeated node
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0, 1.0, 2.0], [1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0, 1.0, 2.0], [1.0, 2.0]))
    with pytest.raises(TypeError):
        numopt.run(mid, "not data")


@pytest.mark.parametrize("mid", PIECEWISE)
def test_piecewise_needs_two_points(mid):
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0], [1.0]))


def test_single_node_polynomials_are_constants():
    for mid in (*POLY, "chebyshev_interpolation"):
        res = numopt.run(mid, ([0.5], [3.0]))
        assert res.converged
        assert_allclose(_grid_y(res), 3.0, rtol=0, atol=1e-15)


def test_bad_parameters():
    d = problems.get("sine_samples")
    with pytest.raises(ValueError):
        numopt.run("neville", d, x_frac=1.5)
    with pytest.raises(ValueError):
        numopt.run("chebyshev_interpolation", d, n_nodes=-1)
    with pytest.raises(ValueError):
        numopt.run("cubic_spline_clamped", d, fprime_a=math.inf)
    with pytest.raises(TypeError):
        numopt.run("lagrange", d, tol=1e-3)


# --------------------------------------------------------------------------------------
# Failure paths: converged=False with a message, never an exception
# --------------------------------------------------------------------------------------


def test_barycentric_weights_overflow_is_reported():
    x = np.linspace(0.0, 4000.0, 300)
    res = numopt.run("barycentric", (x, np.sin(x)))
    assert not res.converged and "weights" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12, 1e6])
def test_newton_form_roundoff_is_reported(scale):
    # 300 nodes: the divided differences stay finite but the nested form no longer
    # reproduces the data; Neville and the Lagrange product form still do. The flag
    # must not depend on the units of y (audit: the limit was 1e-6·max(1, max|y|), an
    # absolute 1e-6 for |y| < 1).
    x = np.linspace(0.0, 4000.0, 300)
    y = scale * np.sin(x)
    res = numopt.run("newton_divided_differences", (x, y))
    assert not res.converged and "node" in res.message
    assert_valid_result(res)
    assert numopt.run("neville", (x, y)).converged


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-8, 2.0**-40, 1e8])
def test_moderate_newton_roundoff_is_reported_in_any_units(scale):
    # Audit reproduction: 70 equispaced nodes, y = s·cos(3x). The Newton form misses the
    # data by 10–14% of max|y| at every s; it was converged=True for s = 1e-6 and 1e-8.
    # The Lagrange, barycentric and Chebyshev forms reproduce the data to ≤ 1e-12·s.
    x = np.linspace(-1.0, 1.0, 70)
    y = scale * np.cos(3.0 * x)
    res = numopt.run("newton_divided_differences", (x, y))
    assert_valid_result(res)
    assert not res.converged, res.message
    assert res.extra["node_residual"] > 0.05 * scale
    for mid in ("lagrange", "barycentric", "chebyshev_interpolation"):
        other = numopt.run(mid, (x, y))
        assert other.converged, (mid, other.message)
        assert other.extra["node_residual"] <= 1e-12 * scale


@given(nodes_strategy(1, 12, 0.05, 2.0, shuffle=True), st.integers(-60, 60))
@settings(max_examples=1000, deadline=None)
def test_convergence_flag_is_invariant_under_scaling_of_y(xy, k):
    # y → 2^k·y is exact in binary floating point (no underflow/overflow here), and every
    # method is homogeneous in y, so the node residual scales by exactly 2^k and the
    # relative node test must give the same flag.
    x, y = xy
    assume(len(set(x.tolist())) == x.size)
    assume(np.all((y == 0.0) | (np.abs(y) >= 1e-200)))
    for mid in ALL:
        if mid in PIECEWISE and x.size < 2:
            continue
        base = numopt.run(mid, (x, y))
        scaled = numopt.run(mid, (x, 2.0**k * y))
        assert scaled.converged == base.converged, (mid, base.message, scaled.message)
        assert scaled.extra["node_residual"] == 2.0**k * base.extra["node_residual"], mid


@pytest.mark.parametrize("mid", ALL)
def test_overflow_is_reported(mid):
    x = np.linspace(0.0, 1.0, 12)
    y = 1e308 * (-1.0) ** np.arange(12)
    res = numopt.run(mid, (x, y))
    assert not res.converged and res.message
    assert_valid_result(res)


def test_chebyshev_non_finite_f_true_is_reported():
    # Resampling runs only when the data are samples of f_true (the sample check costs
    # one f_true call per data point, counted in n_fev).
    def log1p_shift(t):
        return math.log(t + 1.0)

    x = np.linspace(-0.5, 1.0, 4)
    d = Dataset("pole", "pole", x, np.log(x + 1.0), log1p_shift, domain=(-1.0, 1.0))
    res = numopt.run("chebyshev_interpolation", d, n_nodes=6)
    assert res.converged and res.extra["source"] == "f_true"  # the nodes avoid t = -1
    assert res.n_fev == 4 + 6

    # f_true agrees with the data at the data points but is NaN at every Chebyshev node.
    def nan_inside(t):
        return 0.0 if t in (-1.0, 1.0) else math.nan

    bad = Dataset("nan", "nan", np.array([-1.0, 1.0]), np.zeros(2), nan_inside)
    res = numopt.run("chebyshev_interpolation", bad, n_nodes=5)
    assert not res.converged and "not finite" in res.message and res.n_fev == 2 + 5
    assert_valid_result(res)


# --------------------------------------------------------------------------------------
# Known values (Runge's phenomenon) and convergence orders
# --------------------------------------------------------------------------------------


def test_runge_phenomenon_reference_values():
    # Degree-10 interpolation of 1/(1+25x²): max error ≈ 1.9156 on 11 equispaced nodes
    # (near x = ±0.94) and ≈ 0.1092 on the 11 Chebyshev nodes.
    for mid in POLY:
        assert numopt.run(mid, problems.get("runge_equispaced")).fun == pytest.approx(
            1.9156, abs=1e-4
        )
        assert numopt.run(mid, problems.get("runge_chebyshev")).fun == pytest.approx(
            0.1091, abs=1e-4
        )
    # Chebyshev resampling of the equispaced dataset gives the Chebyshev-node polynomial
    # (11 f_true calls for the sample check, 11 at the Chebyshev nodes).
    res = numopt.run("chebyshev_interpolation", problems.get("runge_equispaced"))
    assert res.fun == pytest.approx(0.1091, abs=1e-4) and res.n_fev == 22


@pytest.mark.parametrize(
    ("mid", "params", "order"),
    [
        ("cubic_spline_clamped", {"fprime_a": 1.0, "fprime_b": math.e}, 4.0),
        ("cubic_spline_not_a_knot", {}, 4.0),
        # f'' ≠ 0 at the ends, so the natural end condition is wrong by O(1): O(h²).
        ("cubic_spline_natural", {}, 2.0),
        ("linear_spline", {}, 2.0),
        ("pchip", {}, 3.0),
    ],
)
def test_spline_convergence_order_on_exp(mid, params, order):
    errors = []
    for n in (11, 21, 41, 81):
        x = np.linspace(0.0, 1.0, n)
        d = Dataset("exp", "exp", x, np.exp(x), np.exp, domain=(0.0, 1.0))
        res = numopt.run(mid, d, **params)
        assert res.converged
        errors.append(_fun(res))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(rates - order) < 0.25), rates


def test_chebyshev_converges_geometrically_on_analytic_f():
    d = problems.get("sine_samples")
    errs = [_fun(numopt.run("chebyshev_interpolation", d, n_nodes=n)) for n in (4, 8, 12, 16)]
    assert errs[0] > errs[1] > errs[2] > errs[3] and errs[3] < 1e-9


def test_step_data_monotone_vs_overshoot():
    d = problems.get("step_data")
    for mid in ("pchip", "linear_spline"):
        y = _grid_y(numopt.run(mid, d))
        assert np.all(np.diff(y) >= -1e-15) and y.min() >= -1e-15 and y.max() <= 1 + 1e-15
    y = _grid_y(numopt.run("cubic_spline_natural", d))
    assert y.max() > 1.05 and y.min() < -0.05  # the C² spline overshoots the jump
    assert _fun(numopt.run("lagrange", d)) > 1.0  # the degree-11 polynomial is far worse


# --------------------------------------------------------------------------------------
# Oracle comparisons (SciPy / NumPy)
# --------------------------------------------------------------------------------------


def _exact_lagrange(x, y, ts):
    """p(t) in exact rational arithmetic (the data are binary floats, hence rationals)."""
    xs = [Fraction(v) for v in x]
    ys = [Fraction(v) for v in y]
    out = []
    for t in ts:
        tt, total = Fraction(t), Fraction(0)
        for j in range(len(xs)):
            ell = Fraction(1)
            for m in range(len(xs)):
                if m != j:
                    ell *= (tt - xs[m]) / (xs[j] - xs[m])
            total += ys[j] * ell
        out.append(float(total))
    return np.array(out)


def _lebesgue_constant(x, ts):
    total = np.zeros_like(ts)
    for j in range(x.size):
        ell = np.ones_like(ts)
        for m in range(x.size):
            if m != j:
                ell = ell * (ts - x[m]) / (x[j] - x[m])
        total += np.abs(ell)
    return max(1.0, float(total.max()))


def _poly_error_bound(mid, x, y, res, t):
    """Forward-error bound plus the underflow term 64·n²·η (see :func:`_poly_error_rel`)."""
    return _poly_error_rel(mid, x, y, res, t) + 64 * x.size**2 * ETA


def _poly_error_rel(mid, x, y, res, t):
    """Forward-error bound for each polynomial form on the grid ``t`` (u = eps/2).

    * lagrange (product form): (5n+5)·u·Λ·‖y‖∞            (Higham 2004, first form)
    * barycentric (second form): [(3n+4)Λ + (3n+2)Λ²]·u·‖y‖∞  (Higham 2004)
    * newton: 3n·u·max_t Σ_k D_k Π_{j<k}|t - x_j|, D_k = the divided-difference
      recurrence applied to magnitudes, D[x_i..x_j] = (D[x_{i+1}..x_j] + D[x_i..x_{j-1}])
      / |x_j - x_i| from |y| (= |L⁻¹||y| of Higham, ASNA 2nd ed., Thm. 5.3); divided
      differences + nested evaluation; depends on the node order, not on Λ. (The explicit
      Σ_{j≤k}|y_j|/Π|x_j - x_m| is too small for unsorted nodes, where the recurrence
      cancels: x = [0, 1, 2, 2.375, 3.375, 4.375, 5.375, 2.25], y = e_6 gives D_7 = 13.3
      vs 6.7e-4, and the error exceeded 8× that bound.)
    * neville: running-error bound 4n·u·M(t), where M is the tableau recurrence applied
      to magnitudes, M_{i,0} = |y_i|, M_{i,j} = (|t-x_{i-j}| M_{i,j-1} + |t-x_i| M_{i-1,j-1})
      / |x_i - x_{i-j}| (Higham, ASNA 2nd ed., §3.3 running error analysis)
    * chebyshev (data mode): n·u·κ₂(T)·Σ|c_k|, T the Chebyshev–Vandermonde matrix
    The constants are not sharp. Measured on 3000 random cases (clustered and rounded
    nodes, sparse data): max error/bound = 0.13, 0.15, 0.33, 0.29, 1.3 respectively.
    """
    n = x.size
    u = EPS / 2
    lam = _lebesgue_constant(x, t)
    ys = max(1.0, float(np.abs(y).max()))
    if mid == "lagrange":
        return (5 * n + 5) * u * lam * ys
    bary = ((3 * n + 4) * lam + (3 * n + 2) * lam**2) * u * ys
    if mid == "barycentric":
        return bary
    if mid == "neville":
        mag = np.repeat(np.abs(y)[:, None], t.size, axis=1).astype(float)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                mag[i] = (np.abs(t - x[i - j]) * mag[i] + np.abs(t - x[i]) * mag[i - 1]) / abs(
                    x[i] - x[i - j]
                )
        return 4 * n * u * max(float(mag[n - 1].max()), ys * EPS)
    if mid == "newton_divided_differences":
        total = np.zeros_like(t)
        col = [abs(float(v)) for v in y]  # level k: D[x_i..x_{i+k}], i = 0..n-1-k
        for k in range(n):
            if k > 0:
                col = [(col[i + 1] + col[i]) / abs(x[i + k] - x[i]) for i in range(n - k)]
            d_k = col[0]
            prod = np.ones_like(t)
            for j in range(k):
                prod = prod * np.abs(t - x[j])
            total += d_k * prod
        return 3 * n * u * max(float(total.max()), ys)
    a, b = res.extra["domain"]
    unit = np.clip((2 * x - a - b) / (b - a), -1.0, 1.0)
    vander = np.cos(np.outer(np.arccos(unit), np.arange(n)))
    return n * u * np.linalg.cond(vander) * float(np.abs(np.asarray(res.x)).sum())


@given(nodes_strategy(1, 8, 0.1, 2.0, shuffle=True))
@settings(max_examples=1000, deadline=None)
def test_polynomial_methods_match_exact_and_scipy(xy):
    x, y = xy
    # Oracle: p(t) in exact rational arithmetic. Clustered random nodes give Lebesgue
    # constants up to ~10³ and badly ordered Newton forms, so the tolerance is each
    # method's own forward-error bound (see _poly_error_bound), never a fixed number.
    exact = None
    for mid in (*POLY, "chebyshev_interpolation"):
        res = numopt.run(mid, (x, y))
        assert res.converged, res.message
        t = _grid_x(res)
        if exact is None:
            exact = _exact_lagrange(x, y, t[::10])
        bound = _poly_error_bound(mid, x, y, res, t)
        assert_allclose(_grid_y(res)[::10], exact, rtol=0, atol=8 * bound)
        # SciPy's BarycentricInterpolator carries the second-form error itself.
        both = 8 * (bound + _poly_error_bound("barycentric", x, y, res, t))
        assert_allclose(_grid_y(res), BarycentricInterpolator(x, y)(t), rtol=0, atol=both)
        assert res.extra["node_residual"] <= 8 * bound


@given(nodes_strategy(1, 8, 0.2, 2.0), st.data())
@settings(max_examples=1000, deadline=None)
def test_polynomial_methods_reproduce_polynomials(xy, data):
    # Exactness: the interpolant of a polynomial of degree < n is that polynomial.
    x, _ = xy
    deg = data.draw(st.integers(0, x.size - 1))
    coef = np.array(data.draw(st.lists(st.floats(-3, 3), min_size=deg + 1, max_size=deg + 1)))
    y = np.polynomial.polynomial.polyval(x, coef)
    for mid in (*POLY, "chebyshev_interpolation"):
        res = numopt.run(mid, (x, y))
        t = _grid_x(res)
        expect = np.polynomial.polynomial.polyval(t, coef)
        bound = np.polynomial.polynomial.polyval(np.abs(t), np.abs(coef))  # Σ|c_k||t|^k
        assert_allclose(_grid_y(res), expect, rtol=0, atol=1e-11 * max(1.0, float(bound.max())))


def _exact_spline_slopes(x, y, bc, fa=0.0, fb=0.0):
    """Exact node slopes of the cubic spline, in rational arithmetic.

    Independent of the method's reduced rows: not-a-knot is imposed directly as
    d_0 = d_1 and d_{n-3} = d_{n-2}, d_i = (s_i + s_{i+1} - 2m_i)/h_i² (needs n ≥ 4).
    """
    xs = [Fraction(v) for v in x]
    ys = [Fraction(v) for v in y]
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    m = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    rows = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for i in range(1, n - 1):
        rows[i][i - 1], rows[i][i], rows[i][i + 1] = h[i], 2 * (h[i - 1] + h[i]), h[i - 1]
        rows[i][n] = 3 * (h[i] * m[i - 1] + h[i - 1] * m[i])
    if bc == "natural":
        rows[0][0], rows[0][1], rows[0][n] = Fraction(2), Fraction(1), 3 * m[0]
        rows[-1][n - 2], rows[-1][n - 1], rows[-1][n] = Fraction(1), Fraction(2), 3 * m[-1]
    elif bc == "clamped":
        rows[0][0], rows[0][n] = Fraction(1), Fraction(fa)
        rows[-1][n - 1], rows[-1][n] = Fraction(1), Fraction(fb)
    else:
        for row, (i, j) in ((0, (0, 1)), (n - 1, (n - 3, n - 2))):
            rows[row] = [Fraction(0)] * (n + 1)
            rows[row][i] = 1 / h[i] ** 2
            rows[row][i + 1] = 1 / h[i] ** 2 - 1 / h[j] ** 2
            rows[row][j + 1] = -1 / h[j] ** 2
            rows[row][n] = 2 * m[i] / h[i] ** 2 - 2 * m[j] / h[j] ** 2
    for c in range(n):  # Gauss–Jordan, exact
        p = next(i for i in range(c, n) if rows[i][c] != 0)
        rows[c], rows[p] = rows[p], rows[c]
        for i in range(n):
            if i != c and rows[i][c] != 0:
                f = rows[i][c] / rows[c][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[c], strict=True)]
    return np.array([float(rows[i][n] / rows[i][i]) for i in range(n)])


def _spline_error_scale(res):
    """n·κ·u·(Horner condition): the forward-error scale of solve + evaluation.

    κ = κ₂ of the row-equilibrated tridiagonal system (from the trace); the Horner
    condition is max_t Σ_k |coef_k| |t - x_i|^k on the grid. Measured on 1500 random
    cases (spacing ratios up to 10³): error/scale ≤ 0.68 for ours and for SciPy.
    """
    info = res.trace[0].info
    t_mat = np.diag(info["diag"]) + np.diag(info["lower"], -1) + np.diag(info["upper"], 1)
    kappa = np.linalg.cond(t_mat / np.abs(t_mat).sum(axis=1, keepdims=True))
    coef = np.asarray(res.extra["coefficients"])
    nodes = np.asarray(res.extra["nodes"])
    t = _grid_x(res)
    idx = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, nodes.size - 2)
    dt = np.abs(t - nodes[idx])
    horner = np.zeros_like(t)
    for k in range(4):
        horner += np.abs(coef[idx, k]) * dt**k
    return nodes.size * kappa * (EPS / 2) * max(1.0, float(horner.max()))


@given(nodes_strategy(2, 10, 0.01, 10.0), st.floats(-20, 20), st.floats(-20, 20))
@settings(max_examples=1000, deadline=None)
def test_cubic_splines_match_exact_and_scipy(xy, fa, fb):
    x, y = xy
    # Spacing ratios up to 10³ make the not-a-knot system ill-conditioned (κ up to ~10⁶):
    # the tolerance is the derived forward-error scale, not a fixed number.
    cases: tuple[tuple[str, str, Any, dict[str, float]], ...] = (
        ("cubic_spline_natural", "natural", "natural", {}),
        ("cubic_spline_clamped", "clamped", ((1, fa), (1, fb)), {"fprime_a": fa, "fprime_b": fb}),
        ("cubic_spline_not_a_knot", "not-a-knot", "not-a-knot", {}),
    )
    for mid, bc, scipy_bc, params in cases:
        res = numopt.run(mid, (x, y), **params)
        assert res.converged, res.message
        scale = _spline_error_scale(res)
        t = _grid_x(res)
        slopes = np.asarray(res.extra["slopes"])
        if bc != "not-a-knot" or x.size >= 4:
            exact_s = _exact_spline_slopes(x, y, bc, fa, fb)
            exact = im._pp_eval(x, im._hermite_coefficients(x, y, exact_s), t)
            assert_allclose(_grid_y(res), exact, rtol=0, atol=4 * scale)
            h = float(np.min(np.diff(x)))
            assert_allclose(slopes, exact_s, rtol=0, atol=4 * scale / h)
        ref = CubicSpline(x, y, bc_type=scipy_bc)
        assert_allclose(_grid_y(res), ref(t), rtol=0, atol=8 * scale)


def test_cubic_spline_special_sizes_match_scipy():
    for x, y in (([0.0, 1.0], [1.0, 3.0]), ([0.0, 1.0, 3.0], [1.0, -2.0, 5.0])):
        for mid, bc in (
            ("cubic_spline_natural", "natural"),
            ("cubic_spline_not_a_knot", "not-a-knot"),
        ):
            res = numopt.run(mid, (np.array(x), np.array(y)))
            expect = CubicSpline(x, y, bc_type=bc)(_grid_x(res))
            assert_allclose(_grid_y(res), expect, rtol=1e-13, atol=1e-13)


@given(nodes_strategy(2, 12, 0.01, 10.0, shuffle=True))
@settings(max_examples=1000, deadline=None)
def test_pchip_and_linear_spline_match_scipy(xy):
    x, y = xy
    res = numopt.run("pchip", (x, y))
    t = _grid_x(res)
    order = np.argsort(x)
    expect = PchipInterpolator(x[order], y[order])(t)
    assert_allclose(_grid_y(res), expect, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(expect))))
    res = numopt.run("linear_spline", (x, y))
    assert_allclose(_grid_y(res), np.interp(t, x[order], y[order]), rtol=0, atol=1e-12 * 10)


@given(nodes_strategy(2, 12, 0.01, 5.0), st.booleans())
@settings(max_examples=1000, deadline=None)
def test_pchip_preserves_monotonicity(xy, increasing):
    # Fritsch–Carlson: slopes with 0 ≤ α, β ≤ 3 make each cubic piece monotone.
    x, y = xy
    y = np.cumsum(np.abs(y)) * (1 if increasing else -1)
    res = numopt.run("pchip", (x, y))
    slopes_step = res.trace[1].info
    for a, b in zip(slopes_step["alpha"], slopes_step["beta"], strict=True):
        if a is not None:
            assert -1e-12 <= a <= 3 + 1e-12 and -1e-12 <= b <= 3 + 1e-12
    xs = np.linspace(x[0], x[-1], 2001)
    vals = im._pp_eval(np.asarray(res.extra["nodes"]), np.asarray(res.extra["coefficients"]), xs)
    dv = np.diff(vals) * (1 if increasing else -1)
    assert np.all(dv >= -1e-12 * max(1.0, np.max(np.abs(y))))


@given(st.integers(1, 30), st.floats(-3, 3), st.floats(-2, 2), st.floats(0.5, 4))
@settings(max_examples=1000, deadline=None)
def test_chebyshev_coefficients_match_numpy_chebinterpolate(n, c, a, width):
    b = a + width

    def f(t):
        return math.exp(c * t) * math.cos(t)

    d = Dataset("f", "f", np.array([a, b]), np.array([f(a), f(b)]), f, domain=(a, b))
    res = numopt.run("chebyshev_interpolation", d, n_nodes=n)
    assert res.converged and res.n_fev == 2 + n and res.extra["source"] == "f_true"
    ref = npcheb.chebinterpolate(
        lambda u: (
            np.exp(c * (0.5 * (a + b) + 0.5 * (b - a) * u))
            * np.cos(0.5 * (a + b) + 0.5 * (b - a) * u)
        ),
        n - 1,
    )
    # Both are O(n)-term sums of values ≤ max|f|; agreement to a few ulps of max|f|.
    fmax = float(np.max(np.abs(res.extra["values"])))
    assert_allclose(res.x, ref, rtol=0, atol=50 * EPS * max(1.0, fmax))


# --------------------------------------------------------------------------------------
# Method-specific invariants (independent formulas)
# --------------------------------------------------------------------------------------


@given(nodes_strategy(1, 8, 0.1, 2.0, shuffle=True))
@settings(max_examples=1000, deadline=None)
def test_barycentric_weights_formula(xy):
    # Berrut & Trefethen (2004) eq. (3.2): w_j Π_{m≠j}(x_j - x_m) = 1.
    x, y = xy
    res = numopt.run("barycentric", (x, y))
    w = np.asarray(res.x)
    for j in range(x.size):
        prod = float(np.prod(np.delete(x[j] - x, j)))
        assert w[j] * prod == pytest.approx(1.0, rel=1e-13 * x.size)
    for k, step in enumerate(res.trace):
        assert len(step.x) == k + 1  # one node per step


@given(nodes_strategy(1, 8, 0.1, 2.0, shuffle=True))
@settings(max_examples=1000, deadline=None)
def test_newton_coefficients_symmetric_formula(xy):
    # f[x_0..x_k] = Σ_{j≤k} y_j / Π_{m≤k, m≠j} (x_j - x_m)   (independent of the table)
    x, y = xy
    res = numopt.run("newton_divided_differences", (x, y))
    coef = np.asarray(res.x)
    for k in range(x.size):
        terms = [y[j] / np.prod(np.delete(x[j] - x[: k + 1], j)) for j in range(k + 1)]
        mag = float(np.sum(np.abs(terms)))
        assert abs(coef[k] - float(np.sum(terms))) <= 1e-12 * max(1.0, mag) * (k + 1)
        row = res.trace[k].info["new_row"]
        assert len(row) == k + 1 and row[0] == y[k] and row[-1] == coef[k]


@given(nodes_strategy(1, 8, 0.1, 2.0, shuffle=True), st.floats(0, 1))
@settings(max_examples=1000, deadline=None)
def test_neville_diagonal_estimates(xy, frac):
    x, y = xy
    res = numopt.run("neville", (x, y), x_frac=frac)
    a, b = res.extra["domain"]
    x_star = a + frac * (b - a)
    assert res.extra["x_eval"] == pytest.approx(x_star)
    t_star = np.array([res.extra["x_eval"]])
    for k, step in enumerate(res.trace):
        # Q_{k,k} = P_{0..k}(x*): exact rational oracle (SciPy's barycentric evaluator
        # returns NaN when x* is a subnormal distance from a node), pointwise bound.
        xk, yk = x[: k + 1], y[: k + 1]
        expect = float(_exact_lagrange(xk, yk, t_star)[0])
        bound = _poly_error_bound("neville", xk, yk, res, t_star)
        assert abs(step.x - expect) <= 8 * bound
        assert len(step.info["column"]) == x.size - k
    assert res.x == res.extra["value"] == res.trace[-1].x


def test_lagrange_trace_is_the_partial_sum():
    d = problems.get("sine_samples")
    res = numopt.run("lagrange", d)
    total = np.zeros(im.N_GRID)
    for k, step in enumerate(res.trace):
        total = total + d.y[k] * np.asarray(step.info["basis"])
        assert_allclose(step.info["curve"], total, rtol=0, atol=1e-15)
        assert_allclose(step.x, d.y[: k + 1], rtol=0, atol=0)
    # Σ_j ℓ_j ≡ 1 (the basis reproduces constants).
    basis_sum = np.sum([s.info["basis"] for s in res.trace], axis=0)
    assert_allclose(basis_sum, 1.0, rtol=0, atol=1e-13)


def test_spline_stages_and_system():
    d = problems.get("runge_equispaced")
    res = numopt.run("cubic_spline_natural", d)
    stages = [s.info["stage"] for s in res.trace]
    assert stages == ["assemble", "forward_sweep", "back_substitution", "coefficients"]
    info = res.trace[0].info
    n = d.x.size
    t_mat = np.diag(info["diag"]) + np.diag(info["lower"], -1) + np.diag(info["upper"], 1)
    slopes = np.asarray(res.trace[2].x)
    assert_allclose(t_mat @ slopes, info["rhs"], rtol=1e-13, atol=1e-13)
    assert len(res.x) == 4 * (n - 1)
    # C² continuity at interior knots: S_i''(x_{i+1}) = S_{i+1}''(x_{i+1}).
    coef = np.asarray(res.extra["coefficients"])
    h = np.diff(d.x)
    left = 2 * coef[:-1, 2] + 6 * coef[:-1, 3] * h[:-1]
    assert_allclose(left, 2 * coef[1:, 2], rtol=0, atol=1e-12)
    assert abs(coef[0, 2]) < 1e-14  # S''(x_0) = 0
    assert abs(2 * coef[-1, 2] + 6 * coef[-1, 3] * h[-1]) < 1e-12  # S''(x_n) = 0
    assert [s.info["stage"] for s in numopt.run("pchip", d).trace] == [
        "secants",
        "slopes",
        "coefficients",
    ]


def test_left_endpoint_uses_first_piece():
    # Legacy bug (AUDIT §1.20): searchsorted gave i = -1 at x_0. Data (0,1),(1,2),(2,0),(3,5).
    res = numopt.run("cubic_spline_natural", ([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.0, 5.0]))
    nodes = np.asarray(res.extra["nodes"])
    vals = im._pp_eval(nodes, np.asarray(res.extra["coefficients"]), nodes)
    assert_allclose(vals, [1.0, 2.0, 0.0, 5.0], rtol=0, atol=1e-14)


def test_not_a_knot_reproduces_cubics_and_clamped_with_exact_slopes():
    x = np.array([-1.0, -0.3, 0.2, 0.9, 1.5, 2.0])

    def p(t):
        return 2 - t + 0.5 * t**2 - 0.7 * t**3

    def dp(t):
        return -1 + t - 2.1 * t**2

    for mid, params in (
        ("cubic_spline_not_a_knot", {}),
        ("cubic_spline_clamped", {"fprime_a": dp(-1.0), "fprime_b": dp(2.0)}),
    ):
        res = numopt.run(mid, Dataset("c", "c", x, p(x), p), **params)
        assert _fun(res) < 1e-13


def test_exact_extrapolation_domain_and_sorting():
    # Unsorted input: piecewise methods sort; nodes come back ascending.
    res = numopt.run("pchip", ([3.0, 1.0, 2.0], [9.0, 1.0, 4.0]))
    assert res.extra["nodes"].tolist() == [1.0, 2.0, 3.0]
    assert res.extra["values"].tolist() == [1.0, 4.0, 9.0]


@given(nodes_strategy(2, 8, 0.1, 2.0, shuffle=True))
@settings(max_examples=1000, deadline=None)
def test_every_method_interpolates(xy):
    x, y = xy
    assume(len(set(x.tolist())) == x.size)
    for mid in ALL:
        res = numopt.run(mid, (x, y))
        assert res.converged, (mid, res.message)
        if mid in PIECEWISE:
            # p(x_i) = a_i = y_i exactly, except at the last node: one Horner evaluation,
            # error ≤ γ₆·Σ_k |c_k| h^k (Higham, ASNA 2nd ed., §5.1), γ₆ ≈ 6u.
            coef = np.abs(np.asarray(res.extra["coefficients"]))
            h = np.diff(np.asarray(res.extra["nodes"]))[:, None]
            horner = float(np.max(np.sum(coef * h ** np.arange(4), axis=1)))
            limit = 8 * (EPS / 2) * horner + 64 * ETA
        else:
            limit = 8 * _poly_error_bound(mid, x, y, res, _grid_x(res))
        assert res.extra["node_residual"] <= limit


# --------------------------------------------------------------------------------------
# Audit regressions
# --------------------------------------------------------------------------------------


def test_chebyshev_data_mode_reports_an_ignored_n_nodes():
    # Without f_true the data are the nodes: a UI value n_nodes = 50 on anscombe_1 used to
    # give 11 terms with no word about it.
    d = problems.get("anscombe_1")
    plain = numopt.run("chebyshev_interpolation", d)
    res = numopt.run("chebyshev_interpolation", d, n_nodes=50)
    assert_valid_result(res)
    assert res.converged and res.extra["source"] == "data"
    assert len(res.trace) == d.x.size
    assert "n_nodes=50 ignored" in res.message
    assert_allclose(res.x, plain.x, rtol=0, atol=0)  # the parameter changes nothing else
    assert "ignored" not in plain.message
    same = numopt.run("chebyshev_interpolation", d, n_nodes=d.x.size)
    assert "ignored" not in same.message
    resampled = numopt.run("chebyshev_interpolation", problems.get("runge_equispaced"), n_nodes=20)
    assert "ignored" not in resampled.message and len(resampled.trace) == 20


@pytest.mark.parametrize("mid", [*PIECEWISE, "chebyshev_interpolation"])
def test_step_data_grid_error_is_just_below_one_half(mid):
    # Any continuous p has sup|p - H| ≥ 1/2 (H jumps by 1 at 0), but Result.fun is the
    # max over the 200-point grid, which has no point at 0: the reported value is a
    # little below 1/2, as the step_data description now says (it claimed ≥ 1/2).
    d = problems.get("step_data")
    res = numopt.run(mid, d)
    grid = np.asarray(res.extra["eval"]["x"])
    assert not np.any(grid == 0.0)
    assert res.fun is not None and 0.45 < res.fun < 0.5, res.fun
    # The bound is attained near the jump: either side of 0 the error tends to
    # |p(0)| or |1 - p(0)|, and max(|v|, |1 - v|) ≥ 1/2 for every v.
    left, right = grid[grid < 0].max(), grid[grid > 0].min()
    curve = np.asarray(res.extra["eval"]["y"])
    near = max(abs(curve[grid == left][0]), abs(1.0 - curve[grid == right][0]))
    assert near == pytest.approx(res.fun)  # the grid maximum is at the jump
    assert "0.46–0.48" in d.description and "below 1/2" not in d.description


def test_clamped_spline_end_slopes_count_as_data():
    # Hypothesis counterexample of the relative node test: y = 0 with f'_a = 20 is a
    # nonzero spline, and the Horner evaluation at the last node leaves a rounding
    # residual (5.7e-18) that a limit of 1e-6·max|y| = 0 rejects. The end slopes are
    # interpolation data in units of |f'|·h.
    x = np.array([0.0, 0.5, 2.0, 2.3, 7.0])
    y = np.zeros(5)
    res = numopt.run("cubic_spline_clamped", (x, y), fprime_a=20.0, fprime_b=-13.0)
    assert_valid_result(res)
    assert res.converged, res.message
    bc: Any = ((1, 20.0), (1, -13.0))  # SciPy's stubs type bc_type as str only
    ref = CubicSpline(x, y, bc_type=bc)
    assert_allclose(_grid_y(res), ref(_grid_x(res)), rtol=0, atol=1e-12 * 20.0 * 4.7)
    assert np.max(np.abs(_grid_y(res))) > 1.0  # the spline is not the zero function


NOISY_IDS = ("noisy_linear", "noisy_quadratic", "outliers_linear", "exponential_growth")
NOISE_FREE_IDS = ("runge_equispaced", "runge_chebyshev", "sine_samples", "step_data")


@pytest.mark.parametrize("pid", NOISY_IDS)
def test_chebyshev_interpolates_noisy_data_not_the_hidden_truth(pid):
    # Audit: on the noisy sets f_true is the hidden mean curve. Resampling it gave a max
    # error of 9e-15 while the curve missed the user's data by up to 0.99 (noisy_linear);
    # every other interpolant goes through the data.
    d = problems.get(pid)
    res = numopt.run("chebyshev_interpolation", d)
    assert_valid_result(res)
    assert res.converged, res.message
    assert res.extra["source"] == "data"
    assert "not samples of f_true" in res.message
    assert res.n_fev == d.x.size  # the sample check only
    deviation = float(np.max(np.abs(d.f_true(d.x) - d.y)))
    assert res.extra["sample_deviation"] == pytest.approx(deviation, rel=1e-15)
    assert deviation > im.NODE_RTOL * float(np.max(np.abs(d.y)))
    # The curve is the interpolant of the data (oracle: SciPy barycentric) and goes
    # through every data point.
    a, b = res.extra["domain"]
    at_data = npcheb.chebval((2.0 * d.x - (a + b)) / (b - a), np.asarray(res.x))
    ymax = float(np.max(np.abs(d.y)))
    assert np.max(np.abs(at_data - d.y)) <= 1e-9 * ymax
    ref = BarycentricInterpolator(d.x, d.y)(_grid_x(res))
    bary = numopt.run("barycentric", d)
    curve_size = float(np.max(np.abs(ref)))
    # Both evaluate the same degree-(m-1) polynomial; the Chebyshev–Vandermonde solve
    # loses ~log10 κ₂(T) digits. Measured: κ₂(T) ≤ 1.1e5 (noisy_quadratic, 25 points),
    # relative grid difference ≤ 5.9e-12; the tolerance 1e-9 ≈ 40·κ·ε.
    assert_allclose(_grid_y(res), ref, rtol=0, atol=1e-9 * curve_size)
    assert res.fun == pytest.approx(bary.fun, rel=1e-9)


@pytest.mark.parametrize("pid", NOISE_FREE_IDS)
def test_chebyshev_resamples_noise_free_data(pid):
    d = problems.get(pid)
    res = numopt.run("chebyshev_interpolation", d, n_nodes=15)
    assert res.converged and res.extra["source"] == "f_true"
    assert res.extra["sample_deviation"] == 0.0  # the library samples are exact
    assert res.n_fev == d.x.size + 15 and len(res.trace) == 15
    assert "not samples" not in res.message and "ignored" not in res.message


def test_chebyshev_noisy_data_ignore_n_nodes():
    d = problems.get("noisy_linear")
    res = numopt.run("chebyshev_interpolation", d, n_nodes=40)
    assert res.extra["source"] == "data" and len(res.trace) == d.x.size
    assert "n_nodes=40 ignored: in data mode" in res.message
