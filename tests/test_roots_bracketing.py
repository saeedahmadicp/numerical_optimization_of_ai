import math
from itertools import pairwise

import pytest
from conftest import assert_valid_result
from hypothesis import given
from hypothesis import strategies as st

import numopt


def test_bisection_cube_root():
    res = numopt.find_root(lambda x: x**3 - 2, bracket=(0, 2), method="bisection", xtol=1e-12)
    assert_valid_result(res, max_iter=200)
    assert res.converged
    assert abs(res.x - 2 ** (1 / 3)) <= 1e-12
    # Bracket must always contain the root and halve every step.
    for prev, cur in pairwise(res.trace):
        a0, b0 = prev.info["bracket"]
        a1, b1 = cur.info["bracket"]
        assert a1 <= 2 ** (1 / 3) <= b1
        assert math.isclose(b1 - a1, 0.5 * (b0 - a0))


def test_bisection_rejects_bad_bracket():
    with pytest.raises(ValueError):
        numopt.find_root(lambda x: x**2 + 1, bracket=(-1, 1), method="bisection")


def test_bisection_reports_max_iter():
    res = numopt.find_root(lambda x: x - 0.3, bracket=(0, 1), method="bisection", max_iter=5)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=5)


@given(st.floats(-50, 50), st.floats(0.1, 20))
def test_bisection_iteration_bound(root, width):
    a, b = root - width * 0.37, root + width * 0.63
    res = numopt.find_root(lambda x: x - root, bracket=(a, b), method="bisection", xtol=1e-9)
    assert res.converged
    assert abs(res.x - root) <= 1e-9 + 1e-12 * abs(root)
    assert res.n_iter <= math.ceil(math.log2((b - a) / 1e-9))


# ======================================================================================
# All bracketing methods
# ======================================================================================

import sys  # noqa: E402
from fractions import Fraction  # noqa: E402
from typing import cast  # noqa: E402

import numpy as np  # noqa: E402
from hypothesis import settings  # noqa: E402
from scipy import optimize  # noqa: E402
from scipy.optimize import elementwise  # noqa: E402

from numopt import problems  # noqa: E402
from numopt.core.types import Problem  # noqa: E402
from numopt.roots import bracketing  # noqa: E402

EPS = sys.float_info.epsilon
RTOL = np.float64(4 * EPS)  # SciPy's smallest allowed rtol
METHODS = (
    "bisection",
    "regula_falsi",
    "illinois",
    "pegasus",
    "anderson_bjorck",
    "ridders",
    "brent",
    "chandrupatla",
    "itp",
)
FALSE_POSITION = ("regula_falsi", "illinois", "pegasus", "anderson_bjorck")
ROOT_PROBLEMS = [p.id for p in problems.list_problems("roots")]
#: Documented legitimate failures with default parameters: (method, problem).
KNOWN_FAILURES = {("regula_falsi", "steep_exp")}


def _counting(p: Problem) -> tuple[Problem, list[float]]:
    """Copy of p whose f records every argument (an independent evaluation counter)."""
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return p.f(x)

    return Problem(p.id, p.name, p.latex, f, 1, p.domain, bracket=p.bracket, roots=p.roots), calls


def _tol(x: float, xtol: float = 1e-10) -> float:
    return xtol + 2 * EPS * abs(x)


def _check_brackets(res, f) -> None:
    """Every step: x_k inside its bracket; both brackets keep a sign change; nesting."""
    for s in res.trace:
        lo, hi = s.info["bracket"]
        nlo, nhi = s.info["new_bracket"]
        assert lo <= s.x <= hi
        assert lo <= nlo <= nhi <= hi
        assert f(lo) * f(hi) <= 0
        assert f(nlo) * f(nhi) <= 0
    for prev, cur in pairwise(res.trace):
        assert cur.info["bracket"] == prev.info["new_bracket"]


@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
@pytest.mark.parametrize("method", METHODS)
def test_method_on_every_problem(method, pid):
    p, calls = _counting(problems.get(pid))
    res = numopt.run(method, p)
    assert_valid_result(res, max_iter=200)
    assert res.n_fev == len(calls), "n_fev must equal the true number of f evaluations"
    assert res.n_gev == 0 and res.n_hev == 0
    _check_brackets(res, p.f)
    if (method, pid) in KNOWN_FAILURES:
        assert not res.converged and "max_iter" in res.message
        return
    assert res.converged, res.message
    assert res.fun == p.f(res.x)
    root = min(p.roots, key=lambda r: abs(r - res.x))
    assert p.bracket is not None
    a, b = p.bracket
    assert a <= root <= b
    assert abs(res.x - root) <= 2 * _tol(root), (res.x, root)


@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
def test_brent_iterates_equal_scipy_brentq(pid):
    """Oracle: SciPy's brentq evaluates f at the same points (xtol ↔ 2·xtol, rtol = 4ε)."""
    p = problems.get(pid)
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return p.f(x)

    root = float(cast(float, optimize.brentq(f, *p.bracket, xtol=2e-10, rtol=RTOL)))
    res = numopt.run("brent", p, xtol=1e-10)
    np.testing.assert_allclose([s.x for s in res.trace], calls[2:], rtol=1e-12, atol=0)
    assert res.x == root


@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
def test_chandrupatla_iterates_equal_scipy_find_root(pid):
    """Oracle: scipy.optimize.elementwise.find_root implements Chandrupatla (1997)."""
    p = problems.get(pid)
    calls: list[float] = []

    def f(x):
        calls.extend(np.ravel(x).tolist())
        return np.vectorize(p.f)(x)

    tol = dict(xatol=2e-10, xrtol=4 * EPS, fatol=0.0, frtol=0.0)
    ref = elementwise.find_root(f, p.bracket, tolerances=tol)
    res = numopt.run("chandrupatla", p, xtol=1e-10)
    mine = [s.x for s in res.trace]
    np.testing.assert_allclose(mine, calls[-len(mine) :], rtol=1e-12, atol=0)
    assert len(calls) == len(mine) + 2
    assert res.x == float(ref.x)


@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
def test_ridders_iterates_equal_scipy_ridder(pid):
    """Oracle: SciPy's ridder; it differs only in its final step (it clamps x to stay
    tol/2 inside the bracket), so every point but the last two is compared."""
    p = problems.get(pid)
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return p.f(x)

    ref = float(cast(float, optimize.ridder(f, *p.bracket, xtol=2e-10, rtol=RTOL)))
    res = numopt.run("ridders", p, xtol=1e-10)
    mine: list[float] = []
    for s in res.trace:
        if s.info["step"] != "ridders":  # our verifying probe: SciPy stops before it
            break
        mine += [s.info["midpoint"], s.x] if s.info["f_mid"] != 0.0 else [s.x]
    n = min(len(mine), len(calls) - 2) - 2
    # NOTE: atol 1e-15 — the square root is computed as hypot(f_m, √|f_a|·√|f_b|) here,
    # which differs from SciPy's sqrt(f_m² − f_a f_b) in the last bit near x = 0.
    np.testing.assert_allclose(mine[:n], calls[2 : 2 + n], rtol=1e-12, atol=1e-15)
    assert abs(res.x - ref) <= 2 * _tol(ref)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")  # SciPy internals
@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
@pytest.mark.parametrize("method", METHODS)
def test_root_agrees_with_toms748(method, pid):
    if (method, pid) in KNOWN_FAILURES:
        return
    p = problems.get(pid)
    res = numopt.run(method, p, xtol=1e-12)
    a, b = res.trace[-1].info["new_bracket"]
    if a == b:
        assert p.f(a) == 0.0
        return
    # The root the method converged to lies in its final bracket; TOMS 748 finds it there.
    ref = float(cast(float, optimize.toms748(p.f, a, b, xtol=1e-15, rtol=RTOL)))
    assert abs(res.x - ref) <= 2 * _tol(ref, 1e-12)


def _false_position_reference(m_rule: str, n: int) -> list[Fraction]:
    """Exact rational transcription of Ford (1995) Alg. 1 on f = x² − 2, [a, b] = [1, 2]."""

    def f(x: Fraction) -> Fraction:
        return x * x - 2

    a, b = Fraction(1), Fraction(2)
    fa, fb = f(a), f(b)
    out = []
    for _ in range(n):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        out.append(c)
        if fc * fb < 0:
            a, fa = b, fb
        else:
            m = {
                "regula_falsi": Fraction(1),
                "illinois": Fraction(1, 2),
                "pegasus": fb / (fb + fc),
                "anderson_bjorck": (1 - fc / fb) if 1 - fc / fb > 0 else Fraction(1, 2),
            }[m_rule]
            fa *= m
        b, fb = c, fc
    return out


@pytest.mark.parametrize("method", FALSE_POSITION)
def test_false_position_first_iterates_are_exact(method):
    res = numopt.find_root(lambda x: x * x - 2, bracket=(1, 2), method=method, max_iter=3)
    ref = _false_position_reference(method, 4)
    assert ref[0] == Fraction(4, 3) and ref[1] == Fraction(7, 5)  # hand-computed
    if method == "regula_falsi":
        assert ref[2] == Fraction(24, 17)
    if method == "illinois":
        assert ref[2] == Fraction(37, 26)
    np.testing.assert_allclose([s.x for s in res.trace], [float(r) for r in ref], rtol=4 * EPS)


def test_regula_falsi_keeps_the_far_end_of_a_convex_function():
    """The chord steps never move the right end; only the final probe closes the bracket."""
    res = numopt.run("regula_falsi", problems.get("sqrt2"))
    assert res.converged
    chords = [s for s in res.trace if s.info["step"] == "chord"]
    assert chords == res.trace[:-1]
    assert all(s.info["new_bracket"][1] == 2.0 for s in chords)
    assert all(s.info["scale"] == 1.0 for s in res.trace)
    last = res.trace[-1]
    assert last.info["step"] == "verify" and "probe" in res.message
    lo, hi = last.info["new_bracket"]
    # NOTE: the probe x_(k−1) + tol is rounded to the float spacing at x (≤ ε|x|/2).
    assert hi < 2.0 and hi - lo <= _tol(res.x) + EPS * abs(res.x)


def test_illinois_type_methods_free_the_stuck_end():
    p = problems.get("x10_minus_1")
    plain = numopt.run("regula_falsi", p)
    for method in ("illinois", "pegasus", "anderson_bjorck"):
        res = numopt.run(method, p)
        assert res.converged and res.n_iter < plain.n_iter / 3
        assert any(s.info["scale"] != 1.0 for s in res.trace)
        assert res.trace[-1].info["new_bracket"][1] < 1.3  # the right end moved


def test_illinois_scales_by_one_half():
    res = numopt.run("illinois", problems.get("x10_minus_1"))
    assert {s.info["scale"] for s in res.trace} <= {1.0, 0.5}


def test_chords_cross_zero_at_the_iterate():
    for method in FALSE_POSITION:
        res = numopt.run(method, problems.get("kepler"))
        for s in res.trace:
            if s.info["step"] != "chord":
                assert s.info["chord"] is None
                continue
            (x1, y1), (x2, y2) = s.info["chord"]
            assert math.isclose(x2 - y2 * (x2 - x1) / (y2 - y1), s.x, rel_tol=1e-14, abs_tol=0)


def test_ridders_is_exact_for_exponential_times_linear():
    """h(x) = f(x)e^{2x} is linear for f = (x − 0.3)e^{−2x}: one Ridders step is exact."""
    res = numopt.find_root(lambda x: (x - 0.3) * math.exp(-2 * x), bracket=(0, 1), method="ridders")
    assert abs(res.trace[0].x - 0.3) <= 4 * EPS
    q = res.trace[0].info["exp_factor"]
    assert math.isclose(q, math.exp(2 * 0.5), rel_tol=1e-14)


def test_ridders_transformed_points_are_collinear_and_cross_at_x():
    for pid in ROOT_PROBLEMS:
        for s in numopt.run("ridders", problems.get(pid)).trace:
            if s.info["transformed"] is None:
                continue
            (x0, y0), (x1, y1), (x2, y2) = s.info["transformed"]
            scale = max(abs(y0), abs(y1), abs(y2))
            # NOTE: rounding of the abscissa differences gives the ratio (x1−x0)/(x2−x0) a
            # relative error ≈ ε·|x|/(x2 − x0), large once the bracket is tiny.
            ratio_err = 8 * EPS * max(abs(x0), abs(x2)) / (x2 - x0)
            assert abs(y0 + (y2 - y0) * (x1 - x0) / (x2 - x0) - y1) <= (1e-12 + ratio_err) * scale
            x_cross = x0 - y0 * (x2 - x0) / (y2 - y0)
            assert abs(x_cross - s.x) <= 1e-12 * abs(s.x) + 8 * EPS * max(abs(x0), abs(x2))


def test_ridders_bracket_at_least_halves():
    for pid in ROOT_PROBLEMS:
        for s in numopt.run("ridders", problems.get(pid)).trace:
            if s.info["step"] != "ridders":
                continue
            lo, hi = s.info["bracket"]
            nlo, nhi = s.info["new_bracket"]
            assert nhi - nlo <= 0.5 * (hi - lo) * (1 + 4 * EPS)


def test_brent_step_geometry():
    kinds = set()
    for pid in ROOT_PROBLEMS:
        res = numopt.run("brent", problems.get(pid))
        for s in res.trace:
            i = s.info
            kinds.add(i["step"])
            assert i["step"] in ("bisection", "secant", "inverse_quadratic")
            assert i["step"] == "bisection" or i["step"] == i["attempted"]
            assert (
                len(i["points"]) == {None: 0, "secant": 2, "inverse_quadratic": 3}[i["attempted"]]
            )
            assert s.step_size >= i["tol"] * (1 - 4 * EPS)  # never a step below tol
        best = res.trace[-1].info["best"]
        assert res.x == best
    assert kinds == {"bisection", "secant", "inverse_quadratic"}


def test_brent_and_secant_are_exact_on_a_line():
    for method in ("brent", "illinois", "regula_falsi"):
        res = numopt.find_root(lambda x: 3.0 * x - 1.0, bracket=(-2, 5), method=method)
        assert abs(res.trace[0].x - 1 / 3) <= 4 * EPS and res.converged


def test_chandrupatla_starts_with_bisection_and_uses_iqi():
    res = numopt.run("chandrupatla", problems.get("cubic"))
    assert res.trace[0].info["step"] == "bisection" and res.trace[0].x == 2.5
    assert any(s.info["step"] == "inverse_quadratic" for s in res.trace)
    for s in res.trace:
        if s.info["step"] == "inverse_quadratic":
            xi, phi = s.info["xi"], s.info["phi"]
            assert 1 - math.sqrt(1 - xi) < phi < math.sqrt(xi)


def test_itp_worst_case_bound_and_validation():
    p = problems.get("x10_minus_1")
    a, b = p.bracket
    for n0 in (0, 1, 3):
        res = numopt.run("itp", p, n0=n0, xtol=1e-10)
        n_half = math.ceil(math.log2((b - a) / 2e-10))
        assert res.converged and res.n_iter + 1 <= n_half + n0
    for bad in ({"kappa1": 0.0}, {"kappa2": 2.7}, {"kappa2": 0.5}, {"n0": -1}):
        with pytest.raises(ValueError):
            numopt.run("itp", p, **bad)


def test_itp_is_superlinear_on_smooth_problems():
    for pid in ("sqrt2", "cubic", "cos_minus_x", "kepler", "steep_exp"):
        res = numopt.run("itp", problems.get(pid))
        assert res.converged and res.n_iter <= 9


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_and_validation(method):
    res = numopt.run(method, problems.get("kepler"), max_iter=2, xtol=1e-15)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=2)
    with pytest.raises(ValueError, match="opposite signs"):
        numopt.find_root(lambda x: x * x + 1, bracket=(-1, 1), method=method)
    with pytest.raises(ValueError, match="a < b"):
        numopt.find_root(lambda x: x, bracket=(1, -1), method=method)
    with pytest.raises(ValueError, match="finite"):
        numopt.find_root(lambda x: math.nan if x > 0.9 else x, bracket=(-1, 1), method=method)


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_value_inside_the_bracket(method):
    res = numopt.find_root(
        lambda x: math.nan if -0.6 < x < 0.6 else x, bracket=(-1.0, 1.3), method=method
    )
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_exact_zero_at_an_end(method):
    res = numopt.find_root(lambda x: x - 1.0, bracket=(1.0, 3.0), method=method)
    assert res.converged and res.x == 1.0 and res.n_iter == 0 and res.n_fev == 2
    assert_valid_result(res)


@pytest.mark.parametrize("method", METHODS)
def test_large_root_is_reachable(method):
    """tol = xtol + 2ε|x| keeps the test reachable when xtol < float spacing at the root."""
    res = numopt.find_root(lambda x: x - 1e10 - 0.5, bracket=(0.0, 3e10), method=method, xtol=1e-12)
    assert res.converged
    assert abs(res.x - (1e10 + 0.5)) <= 1e-12 + 4 * EPS * 1e10 + 2 * _tol(1e10)


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(METHODS),
    st.floats(-1e3, 1e3),
    st.floats(1e-3, 1e2),
    st.floats(0.05, 0.95),
    st.floats(0.0, 10.0),
    st.sampled_from([-1.0, 1.0]),
)
def test_property_single_root_is_found_inside_every_bracket(method, root, width, frac, c, s):
    """f(x) = s·(x − r)(1 + c(x − r)²) has exactly one root r; brackets must contain it."""

    def f(x: float) -> float:
        d = x - root
        return s * d * (1.0 + c * d * d)

    a, b = root - frac * width, root + (1 - frac) * width
    res = numopt.find_root(f, bracket=(a, b), method=method, max_iter=500)
    for st_ in res.trace:
        lo, hi = st_.info["bracket"]
        assert lo <= root <= hi or f(lo) == 0 or f(hi) == 0
    if not res.converged:
        # Plain regula falsi is linear with a fixed-end rate that can be arbitrarily close to
        # 1 (here 1 − |f'(r)|·|e − r|/|f(e)|); it is the only method allowed to be this slow.
        assert method == "regula_falsi" and "max_iter" in res.message
        return
    # NOTE: f(x) itself has relative error ~ε near the root, which can shift the computed
    # sign change by a few ulps of r beyond the method's own tolerance.
    assert abs(res.x - root) <= 2 * _tol(root) + 8 * EPS * abs(root)


@pytest.mark.parametrize(("method", "pid", "params"), bracketing.FIXTURE_CASES)
def test_fixture_cases_run_and_serialize(method, pid, params):
    """The parity fixtures exported for the web app: valid, JSON-clean, < 300 steps."""
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300
    assert numopt.get_method(method).family == "roots"


# ======================================================================================
# Regression tests: verified step test, exceptions in f, ITP bound in floating point
# ======================================================================================

VERIFIED = (*FALSE_POSITION, "ridders")
LN10 = math.log(10.0)


def _assert_honest(res, root: float, xtol: float = 1e-10) -> None:
    """converged ⇒ x within 2·tol of the root; not converged ⇒ an iteration-limit message."""
    assert_valid_result(res)
    if res.converged:
        assert abs(res.x - root) <= 2 * _tol(root, xtol) + 8 * EPS * abs(root), (res.x, root)
    else:
        assert "max_iter" in res.message


@pytest.mark.parametrize("method", VERIFIED)
def test_chord_that_rounds_onto_a_bracket_end_is_not_convergence(method):
    """eˣ − 10 on (−50, 60): f(60) ≈ 1.1e26, so the first chord points round onto x = −50.
    Two equal iterates used to end the run with converged=True at x = −50."""
    res = numopt.find_root(lambda x: math.exp(x) - 10.0, bracket=(-50, 60), method=method)
    _assert_honest(res, LN10)
    if method != "regula_falsi":
        assert res.converged, res.message
    _check_brackets(res, lambda x: math.exp(x) - 10.0)


@pytest.mark.parametrize("method", VERIFIED)
def test_steep_power_is_never_a_false_convergence(method):
    """x⁴⁰ − 1 on (0.5, 3): f ≈ −1 on most of the bracket. Previously every false-position
    method returned x = 0.5 with converged=True. Regula falsi and Anderson–Björck crawl on
    the flat part (an honest max_iter); Illinois, Pegasus and Ridders find x = 1."""
    res = numopt.find_root(lambda x: x**40 - 1.0, bracket=(0.5, 3.0), method=method)
    _assert_honest(res, 1.0)
    if method in ("illinois", "pegasus", "ridders"):
        assert res.converged, res.message


@pytest.mark.parametrize(
    ("n", "b", "xtol"), [(20, 1.5, 1e-3), (25, 3.0, 1e-10), (30, 3.0, 1e-10), (50, 2.0, 1e-10)]
)
def test_anderson_bjorck_single_ratio_estimate_is_verified(n, b, xtol):
    """After a jump across the bracket ρ = s_k/s_{k−1} is tiny and the a posteriori estimate
    collapses to a tiny step far from the root (x = 0.21 for n = 20, xtol = 1e-3)."""
    res = numopt.find_root(
        lambda x: x**n - 1.0, bracket=(0.0, b), method="anderson_bjorck", xtol=xtol
    )
    _assert_honest(res, 1.0, xtol)
    if n == 20:
        assert res.converged
        kinds = [s.info["step"] for s in res.trace]
        assert "verify" in kinds and "bisection" in kinds  # a misled trigger, then bisection


@pytest.mark.parametrize("b", [100.0, 300.0])
def test_regula_falsi_slow_crawl_is_not_convergence(b):
    """eˣ − 10 on (0, b): the chord points creep up from 0 with ρ ≈ 1 − 1e−39; the step
    test used to accept x = 1.5e−39."""
    res = numopt.find_root(
        lambda x: math.expm1(min(x, 700.0)) - 9.0, bracket=(0.0, b), method="regula_falsi"
    )
    assert not res.converged and "max_iter" in res.message


@pytest.mark.parametrize("xtol", [1e-6, 1e-10, 1e-14])
@pytest.mark.parametrize("power", [3, 5])
def test_ridders_multiple_root_within_tolerance(power, xtol):
    """(x − 0.3)^p: two consecutive Ridders points used to land close together on one side
    (error 27·2·tol for p = 5, xtol = 1e-6). SciPy's ridder is the oracle."""

    def f(x: float) -> float:
        d = x - 0.3
        return d**power

    res = numopt.find_root(f, bracket=(-1.0, 2.0), method="ridders", xtol=xtol)
    assert res.converged
    assert abs(res.x - 0.3) <= 2 * _tol(0.3, xtol)
    ref = float(cast(float, optimize.ridder(f, -1.0, 2.0, xtol=2 * xtol, rtol=RTOL)))
    assert abs(ref - 0.3) <= 2 * _tol(0.3, xtol)


def test_ridders_point_on_a_bracket_end_is_not_convergence():
    def f(x: float) -> float:
        return math.expm1(min(98.47 * (x - 0.99884), 700.0))

    res = numopt.find_root(f, bracket=(-2.2899, 36.083), method="ridders")
    _assert_honest(res, 0.99884)
    assert res.converged


@pytest.mark.parametrize("pid", ROOT_PROBLEMS)
@pytest.mark.parametrize("method", VERIFIED)
def test_verify_steps_geometry_and_keys(method, pid):
    """A probe lies tol(x_(k−1)) inside the bracket from the last iterate; it follows a step
    whose estimate is ≤ tol; a failed probe is followed by a bisection (false position)."""
    res = numopt.run(method, problems.get(pid))
    rule = "ridders" if method == "ridders" else "chord"
    keys = {"bracket", "new_bracket", "step", "estimate"} | (
        {"midpoint", "f_mid", "exp_factor", "transformed"}
        if method == "ridders"
        else {"chord", "scale"}
    )
    for i, s in enumerate(res.trace):
        assert set(s.info) == keys
        kind = s.info["step"]
        assert kind in (rule, "verify", "bisection")
        if kind != rule:
            assert s.info["estimate"] is None
        if kind == "verify":
            prev = res.trace[i - 1]
            assert prev.info["step"] == rule and prev.info["estimate"] <= _tol(prev.x)
            assert abs(abs(s.x - prev.x) - _tol(prev.x)) <= EPS * abs(prev.x)
            if s is not res.trace[-1]:
                nxt = res.trace[i + 1].info["step"]
                assert nxt == ("ridders" if method == "ridders" else "bisection")
        if kind == "bisection":
            lo, hi = s.info["bracket"]
            assert s.x == lo + 0.5 * (hi - lo)


@pytest.mark.parametrize("method", METHODS)
def test_exceptions_in_f_become_non_finite_values(method):
    """OverflowError / ValueError / ZeroDivisionError raised by f count as f = nan."""

    def f(x: float) -> float:
        if -0.6 < x < 0.6:
            return math.exp(1e6)  # raises OverflowError
        return x

    res = numopt.find_root(f, bracket=(-1.0, 1.3), method=method)
    assert not res.converged and "not finite" in res.message and "OverflowError" in res.message
    assert_valid_result(res)
    with pytest.raises(ValueError, match=r"finite at the bracket ends.*math domain error"):
        numopt.find_root(lambda x: math.log(x), bracket=(-1.0, 2.0), method=method)
    with pytest.raises(TypeError):  # a bug in f is not swallowed
        numopt.find_root(lambda x: x + "1", bracket=(-1.0, 2.0), method=method)  # type: ignore[operator]


STEEP_FAMILIES = ("exp", "power", "cube", "fifth")


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(METHODS),
    st.sampled_from(STEEP_FAMILIES),
    st.floats(-5.0, 5.0),
    st.floats(0.1, 100.0),
    st.floats(0.01, 0.99),
    st.floats(0.01, 100.0),
    st.integers(2, 50),
)
def test_property_steep_and_multiple_roots(method, family, r, c, frac, width, n):
    """Steep exponentials e^{c(x−r)} − 1 (c ≤ 100, wide brackets), powers xⁿ − 1 (n ≤ 50)
    and the multiple roots (x − r)³, (x − r)⁵. The sign of each f is exact (it is the sign
    of the computed x − r), so converged ⇒ |x − r| ≤ 2·tol; only the false-position family
    may fail, and only by max_iter."""
    if family == "power":
        root = 1.0
        a, b = 1.0 - frac, 1.0 + width / 50.0

        def f(x: float) -> float:
            return x**n - 1.0

    else:
        root = r
        a, b = r - frac * width, r + (1.0 - frac) * width

        def f(x: float) -> float:
            d = x - r
            if family == "exp":
                return math.expm1(min(c * d, 700.0))
            return d * d * d if family == "cube" else d * d * d * d * d

    if not f(a) * f(b) < 0:
        return
    res = numopt.find_root(f, bracket=(a, b), method=method, max_iter=500)
    assert_valid_result(res, max_iter=500)
    for s in res.trace:
        lo, hi = s.info["bracket"]
        assert lo <= root <= hi or f(lo) == 0 or f(hi) == 0
    if not res.converged:
        assert method in FALSE_POSITION and "max_iter" in res.message, res.message
        return
    assert abs(res.x - root) <= 2 * _tol(root) + 8 * EPS * abs(root), (res.x, root, res.message)


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(["lin", "cube", "step", "exp", "pow"]),
    st.floats(-10.0, 10.0),
    st.floats(-3.0, 3.0),
    st.floats(0.01, 0.99),
    st.floats(-12.0, -3.0),
    st.sampled_from([0, 1, 3]),
)
def test_property_itp_worst_case_bound_in_floating_point(kind, r, log_w, frac, log_tol, n0):
    """At most n½ + n₀ new evaluations, n½ = ⌈log₂((b − a)/2xtol)⌉, on random brackets.
    A literal float transcription needed one more on ~13% of such brackets (the projection
    makes every later step tight; see the NOTE in itp)."""
    w, xtol = 10.0**log_w, 10.0**log_tol
    a, b = r - frac * w, r + (1.0 - frac) * w

    def f(x: float) -> float:
        d = x - r
        if kind == "lin":
            return d
        if kind == "cube":
            return d * d * d
        if kind == "step":
            return -1.0 if d < 0 else 1.0
        if kind == "exp":
            return math.expm1(min(30.0 * d, 50.0))
        return math.copysign(abs(d) ** 0.2, d)

    if not f(a) * f(b) < 0:
        return
    res = numopt.find_root(f, bracket=(a, b), method="itp", xtol=xtol, n0=n0)
    assert res.converged
    n_half = math.ceil(math.log2((b - a) / (2 * xtol)))
    assert bracketing.itp_n_half(a, b, xtol) in (n_half - 1, n_half, n_half + 1)
    assert res.n_fev - 2 <= max(0, bracketing.itp_n_half(a, b, xtol)) + n0


def test_itp_audit_example_meets_the_bound():
    a, b, xtol = -73.44897905214071, 53.362908165977345, 4.568883945207533e-4
    res = numopt.find_root(
        lambda x: (x - 1.2143464205738734) ** 3, bracket=(a, b), method="itp", xtol=xtol, n0=3
    )
    assert res.converged and res.n_fev - 2 <= bracketing.itp_n_half(a, b, xtol) + 3 == 21


def test_itp_n_half_is_exact():
    for a, b, eps in [(0.0, 1.0, 2.0**-30), (0.0, 1.0, 2.0**-30 * 1.0000001), (-3.0, 5.0, 1e-7)]:
        n = bracketing.itp_n_half(a, b, eps)
        assert math.ldexp(2 * eps, n) >= b - a and (n == 0 or math.ldexp(2 * eps, n - 1) < b - a)


# ======================================================================================
# Regression tests: bracket and xtol validation, overflow, Ridders containment
# ======================================================================================


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("bracket", "match"),
    [
        ((-math.inf, math.inf), "must be finite"),
        ((-math.inf, 1.0), "must be finite"),
        ((0.0, math.inf), "must be finite"),
        ((math.nan, 1.0), "must be finite"),
        ((-1e308, 1e308), "overflows"),
        ((-1.7e308, 1.7e308), "overflows"),
    ],
)
def test_infinite_or_overflowing_bracket_is_rejected(method, bracket, match):
    """An infinite end or a width b − a that overflows made the half-width test read
    inf ≤ inf: bisection and brent returned x = ±inf with converged=True, ITP raised
    OverflowError. Such a bracket is invalid input."""
    with pytest.raises(ValueError, match=match):
        numopt.find_root(math.tanh, bracket=bracket, method=method)


@pytest.mark.parametrize("method", METHODS)
def test_huge_finite_bracket_is_honest(method):
    """Width 1.6e308 is finite: every derived quantity stays finite. atan(x − 1) is ±π/2
    at the ends, so f·(b − a) overflows in the chord formula (ITP used to evaluate f(nan))."""

    def f(x: float) -> float:
        return math.atan(x - 1.0)

    res = numopt.find_root(f, bracket=(-8e307, 8e307), method=method, max_iter=2000)
    assert_valid_result(res, max_iter=2000)
    assert all(math.isfinite(s.x) for s in res.trace)
    _check_brackets(res, f)
    assert res.converged, res.message
    assert abs(res.x - 1.0) <= 2 * _tol(1.0) + 8 * EPS
    res = numopt.find_root(lambda x: math.tanh(x - 3e300), bracket=(-8e307, 8e307), method=method)
    assert res.converged and abs(res.x - 3e300) <= 2 * _tol(3e300), res.message


@pytest.mark.parametrize("method", [m for m in METHODS if m != "itp"])
@pytest.mark.parametrize("pid", ["sqrt2", "cubic", "kepler"])
def test_xtol_zero_converges_to_full_precision(method, pid):
    """xtol = 0 leaves tol = 2ε|x|: the bracket closes to a few ulps of the root."""
    p = problems.get(pid)
    res = numopt.run(method, p, xtol=0.0)
    assert_valid_result(res)
    assert res.converged, res.message
    root = min(p.roots, key=lambda r: abs(r - res.x))
    assert abs(res.x - root) <= 2 * _tol(root, 0.0) + 8 * EPS * abs(root)


@pytest.mark.parametrize("method", METHODS)
def test_invalid_xtol_is_rejected(method):
    """ITP needs ε = xtol > 0 for its worst-case bound (xtol = 0 used to raise
    ZeroDivisionError, xtol < 0 a bare 'math domain error'); the others need xtol ≥ 0."""
    p = problems.get("sqrt2")
    for bad in (-1.0, -1e-12, math.nan, math.inf):
        with pytest.raises(ValueError, match="xtol"):
            numopt.run(method, p, xtol=bad)
    if method == "itp":
        with pytest.raises(ValueError, match="xtol must be finite and > 0"):
            numopt.run(method, p, xtol=0.0)


def test_itp_n_half_is_exact_without_overflow():
    """(b − a)/(2ε) overflows a float for b − a = 1.6e308, ε = 1e-10; the exact rational
    comparison does not."""
    a, b, eps = -8e307, 8e307, 1e-10
    n = bracketing.itp_n_half(a, b, eps)
    ratio = Fraction(b - a) / (2 * Fraction(eps))
    assert Fraction(2) ** n >= ratio > Fraction(2) ** (n - 1)
    assert n == 1057  # ⌈log₂(8e317)⌉ = ⌈1056.06⌉
    with pytest.raises(ValueError):
        bracketing.itp_n_half(0.0, 1.0, 0.0)


def test_itp_huge_bracket_meets_its_worst_case_bound():
    a, b, xtol = -8e307, 8e307, 1e-10
    res = numopt.find_root(
        lambda x: math.atan(x - 1.0), bracket=(a, b), method="itp", xtol=xtol, max_iter=2000
    )
    assert res.converged and abs(res.x - 1.0) <= 2 * xtol
    assert res.n_fev - 2 <= bracketing.itp_n_half(a, b, xtol) + 1


def test_ridders_point_is_clamped_into_the_bracket():
    """√(x − 0.1) − 1e−20 on (0.1, 0.7): f(m)/s rounds to −1 and m − (m − lo) rounds to
    0.09999999999999998 < lo, where f is undefined. Ridders used to stop there with
    'f(x) is not finite'; every other bracketing method converged."""
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return math.sqrt(x - 0.1) - 1e-20

    res = numopt.find_root(f, bracket=(0.1, 0.7), method="ridders")
    assert res.converged, res.message
    assert abs(res.x - 0.1) <= 2 * _tol(0.1)
    assert all(0.1 <= x <= 0.7 for x in calls)
    assert res.n_fev == len(calls)


def test_ridders_audit_polynomial_stays_inside_its_brackets():
    """The audit's random case 404: step 11 put x one ulp above the bracket's upper end."""
    r, c = -4.241164374282028, 10.565189155387035

    def f(x: float) -> float:
        d = x - r
        return d**3 * (1.0 + c * d * d) + 0.01 * math.sin(5.0 * x) * d

    res = numopt.find_root(
        f,
        bracket=(-14.783340621126005, -3.7929302143484245),
        method="ridders",
        xtol=3.340965622894971e-14,
        max_iter=500,
    )
    assert res.converged
    _check_brackets(res, f)


@settings(max_examples=1500, deadline=None)
@given(
    st.sampled_from(METHODS),
    st.floats(-5.0, 5.0),
    st.sampled_from([1, 3, 5]),
    st.floats(0.0, 20.0),
    st.sampled_from([-1.0, 1.0]),
    st.floats(-3.0, 1.5),
    st.floats(-3.0, 1.5),
    st.one_of(st.just(0.0), st.floats(-14.0, -2.0).map(lambda e: 10.0**e)),
)
def test_property_every_step_stays_inside_its_bracket(method, r, k, c, s, log_l, log_r, xtol):
    """Per step: lo ≤ x_k ≤ hi, new_bracket ⊆ bracket with a sign change, consecutive
    brackets chain, and f is never evaluated outside the initial bracket; on random
    perturbed polynomials s·(x − r)^k(1 + c(x − r)²) + 0.01·s·sin(5x)(x − r), xtol ∈ {0} ∪
    [1e-14, 1e-2] (the ranges of the audit sweep that found the Ridders escape)."""
    if method == "itp" and xtol == 0.0:
        xtol = 1e-14
    a, b = r - 10.0**log_l, r + 10.0**log_r
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        d = x - r
        return s * d**k * (1.0 + c * d * d) + s * 0.01 * math.sin(5.0 * x) * d

    try:
        res = numopt.find_root(f, bracket=(a, b), method=method, xtol=xtol, max_iter=500)
    except ValueError:  # no sign change over (a, b)
        return
    assert_valid_result(res, max_iter=500)
    assert all(a <= x <= b for x in calls)
    assert res.n_fev == len(calls)
    _check_brackets(res, f)


def test_chandrupatla_point_stays_inside_when_tol_is_below_the_float_spacing():
    """Found by the containment property: xtol = 0, root 5.2e−255, bracket (r − 1, r + 1).
    t = 1 − t_l rounded to 1 and x₁ + t(x₂ − x₁) gave x = 0.0 outside [2.6e−255, 0.5]."""
    r = 5.153345104033259e-255

    def f(x: float) -> float:
        d = x - r
        return -(d * (1.0 + d * d) + 0.01 * math.sin(5.0 * x) * d)

    res = numopt.find_root(f, bracket=(r - 1.0, r + 1.0), method="chandrupatla", xtol=0.0)
    assert_valid_result(res)
    _check_brackets(res, f)
    assert not res.converged or abs(res.x - r) <= 2 * _tol(r, 0.0)


def test_itp_order_string() -> None:
    """Regression (web-port): the worst-case bound is written with explicit parentheses and the
    subscript n₀, as in the docstring: ⌈log₂((b−a)/(2·xtol))⌉ + n₀."""
    order = numopt.get_method("itp").order
    assert order == "superlinear on smooth f; worst case ⌈log₂((b−a)/(2·xtol))⌉ + n₀ iterations"
