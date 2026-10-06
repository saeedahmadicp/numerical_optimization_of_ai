"""Tests for numopt.scalar.methods (1-D minimization on an interval)."""

import inspect
import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import optimize as so
from scipy.optimize._optimize import BracketError

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.scalar import methods as sm

METHODS = (
    "golden_section",
    "fibonacci_search",
    "dichotomous_search",
    "ternary_search",
    "parabolic_interpolation",
    "brent_minimize",
    "newton_1d",
    "bracket_minimum",
)
INTERVAL = ("golden_section", "fibonacci_search", "dichotomous_search", "ternary_search")
BRACKETED = (*INTERVAL, "parabolic_interpolation", "brent_minimize")
PROBLEMS = [p.id for p in problems.list_problems("scalar_min")]
# Problems that are unimodal on their default bracket (interval elimination is exact there).
UNIMODAL = (
    "quadratic_1d",
    "quartic_1d",
    "x_log_x",
    "abs_shifted",
    "rational_1d",
    "drug_concentration",
)
PHI = (1 + math.sqrt(5)) / 2
EPS = float(np.finfo(float).eps)


def recording(fn):
    """Wrap fn so every evaluation point is recorded (for evaluation-sequence oracles)."""
    pts: list[float] = []

    def g(x):
        x = float(np.asarray(x).reshape(-1)[0])
        pts.append(x)
        return fn(x)

    return g, pts


def scipy_scalar(f, **kw) -> Any:
    """scipy.optimize.minimize_scalar; its OptimizeResult is untyped for pyright."""
    return so.minimize_scalar(f, **kw)


def bare(f, bracket=None, x0=None) -> Problem:
    return Problem(id="t", name="t", latex="", f=f, dim=1, domain=(-1, 1), bracket=bracket, x0=x0)


def x_star(pid: str) -> float:
    p = problems.get(pid)
    a, b = p.bracket
    return next(x for x in p.minima if a <= x <= b)


def comparison_floor(pid: str) -> float:
    """The accuracy floor h of f-comparisons near x* (module docstring, "Accuracy floor").

    Within √(8ε|f*|/f''(x*)) of x* the f-values differ by less than 4 ulps of f*. The
    factor 2 allows f-evaluation errors of up to 16 ulps; 4ε(1 + |x*|) covers the rounding
    of the points themselves. The value is 2e-8 to 2e-7 on the unimodal library problems.
    """
    p = problems.get(pid)
    xs = x_star(pid)
    return 2 * math.sqrt(8 * EPS * abs(p.f(xs)) / p.hess(xs)) + 4 * EPS * (1 + abs(xs))


# --------------------------------------------------------------------------------------
# Registry and contract
# --------------------------------------------------------------------------------------


def test_registry_and_param_specs():
    specs = {s.id: s for s in numopt.list_methods("scalar")}
    assert set(METHODS) <= set(specs)
    for mid in METHODS:
        spec = specs[mid]
        sig = inspect.signature(spec.fn)
        kw = {n: p for n, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
        declared = {p.name for p in spec.params}
        assert set(kw) - {"x0", "bracket", "seed"} == declared, mid
        for p in spec.params:
            assert kw[p.name].default == p.default, (mid, p.name)
            if p.kind in ("float", "int"):
                assert p.min is not None and p.max is not None and p.min <= p.default <= p.max
        assert spec.references and spec.summary and spec.order


@pytest.mark.parametrize("mid", METHODS)
@pytest.mark.parametrize("pid", PROBLEMS)
def test_contract_on_every_problem(mid, pid):
    res = numopt.run(mid, problems.get(pid))
    max_iter = numopt.get_method(mid).defaults()["max_iter"]
    assert_valid_result(res, max_iter=max_iter)
    assert res.n_iter == res.trace[-1].k
    assert [s.k for s in res.trace] == list(range(len(res.trace)))
    if mid != "newton_1d":
        assert res.n_gev == 0 and res.n_hev == 0
    for s in res.trace:
        assert "bracket" in s.info or mid == "newton_1d"


def _corners(spec) -> list[dict[str, Any]]:
    """Every combination of the numeric ParamSpec ends: (min, max) for floats; (min, default)
    for the integer max_iter (its max only makes the run longer)."""
    axes: list[list[tuple[str, Any]]] = []
    for p in spec.params:
        if p.kind == "float":
            axes.append([(p.name, p.min), (p.name, p.max)])
        elif p.kind == "int":
            axes.append([(p.name, p.min), (p.name, p.default)])
    out: list[dict[str, Any]] = [{}]
    for axis in axes:
        out = [{**d, name: v} for d in out for name, v in axis]
    return out


@pytest.mark.parametrize("mid", METHODS)
@pytest.mark.parametrize("pid", PROBLEMS)
def test_param_spec_corners_are_valid_on_every_problem(mid, pid):
    """Regression: a UI built from the ParamSpecs must never raise. dichotomous_search with
    xtol = 1e-12 and delta_ratio = 1e-3 (δ = 1e-15) raised ValueError on 8 of 9 problems."""
    spec = numopt.get_method(mid)
    for params in _corners(spec):
        res = numopt.run(mid, problems.get(pid), **params)
        assert_valid_result(res, max_iter=params["max_iter"])


def test_fixture_cases_are_valid():
    assert 3 <= len(sm.FIXTURE_CASES) <= 8
    assert {c[0] for c in sm.FIXTURE_CASES} == set(METHODS)
    for mid, pid, params in sm.FIXTURE_CASES:
        res = numopt.run(mid, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) < 300
        assert res.converged, (mid, pid, res.message)


# --------------------------------------------------------------------------------------
# Accuracy on unimodal problems (shared by all bracketed methods)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mid", BRACKETED)
@pytest.mark.parametrize("pid", UNIMODAL)
def test_bracketed_methods_meet_xtol_on_unimodal_problems(mid, pid):
    # xtol = 1e-6 is far above the √ε rounding floor (~1e-8) of f-comparisons, so the
    # interval guarantee |x - x*| ≤ xtol must hold exactly.
    xtol = 1e-6
    res = numopt.run(mid, problems.get(pid), xtol=xtol)
    assert res.converged, res.message
    assert abs(res.x - x_star(pid)) <= xtol
    if mid in INTERVAL or mid == "parabolic_interpolation":
        a, b = res.trace[-1].info["bracket"]
        assert a <= x_star(pid) <= b
        assert b - a <= xtol


@pytest.mark.parametrize("mid", BRACKETED)
@pytest.mark.parametrize("pid", UNIMODAL)
def test_bracket_always_contains_the_minimizer(mid, pid):
    res = numopt.run(mid, problems.get(pid), xtol=1e-6)
    xs = x_star(pid)
    for s in res.trace:
        a, b = s.info["bracket"]
        assert a <= xs <= b, (s.k, a, b)
        assert a <= s.x <= b


@pytest.mark.parametrize("mid", BRACKETED)
@pytest.mark.parametrize("pid", UNIMODAL)
def test_accuracy_at_param_spec_corners_on_unimodal_problems(mid, pid):
    """Regression: at UI corners (xtol = 1e-10, delta_ratio = 1e-3) dichotomous_search
    reported converged=True with |x - x*| = 3.3e-3 (3.3e7·xtol) and x* outside the final
    bracket, because rounding decided its comparisons. Every bracketed method must keep x*
    in every bracket up to the comparison floor h, and a converged run must meet
    max(documented tolerance, h); the only allowed failure is dichotomous_search's
    "comparison resolution" stop, which keeps x* in its bracket."""
    spec = numopt.get_method(mid)
    default_iter = spec.defaults()["max_iter"]
    xs, h = x_star(pid), comparison_floor(pid)
    for params in _corners(spec):
        if params["max_iter"] != default_iter:
            continue  # max_iter = 1 corners are covered by the validity test
        res = numopt.run(mid, problems.get(pid), **params)
        for s in res.trace:
            lo, hi = s.info["bracket"]
            assert lo - h <= xs <= hi + h, (params, s.k, lo, hi)
        err = abs(res.x - xs)
        if res.converged:
            xtol = params["xtol"]
            tol = 2 * (params["rtol"] * abs(res.x) + xtol / 3) if mid == "brent_minimize" else xtol
            assert err <= max(tol, h), (params, err)
        else:
            assert mid == "dichotomous_search", (params, res.message)
            assert "comparison resolution" in res.message, (params, res.message)
            lo, hi = res.extra["bracket"]
            assert err <= hi - lo + h, (params, err)


def test_param_help_states_the_comparison_floor():
    """Regression: the UI help promised |x - x*| ≤ xtol (or 2·tol for Brent) below the √ε
    floor, and called a Brent rtol below √ε "meaningless" while allowing it."""
    for mid in BRACKETED:
        helps = {p.name: p.help for p in numopt.get_method(mid).params}
        assert "√ε" in helps["xtol"], mid
        assert "|x - x*| ≤ xtol" not in helps["xtol"], mid
    rtol_help = {p.name: p.help for p in numopt.get_method("brent_minimize").params}["rtol"]
    assert "meaningless" not in rtol_help and "√ε" in rtol_help


@pytest.mark.parametrize("mid", (*INTERVAL, "parabolic_interpolation", "brent_minimize"))
def test_multimodal_result_is_a_local_minimizer_inside_the_final_bracket(mid):
    p = problems.get("multimodal_1d")
    res = numopt.run(mid, p, xtol=1e-6)
    assert res.converged
    a, b = res.trace[-1].info["bracket"]
    assert a <= res.x <= b
    assert min(abs(res.x - xm) for xm in p.minima) <= 1e-6


@pytest.mark.parametrize("mid", BRACKETED)
def test_scipy_bounded_oracle(mid):
    """Every bracketed method agrees with scipy.optimize.minimize_scalar(method='bounded')."""
    for pid in ("rational_1d", "drug_concentration", "sin_1d", "quartic_1d"):
        p = problems.get(pid)
        ref = scipy_scalar(p.f, bounds=p.bracket, method="bounded", options={"xatol": 1e-9})
        res = numopt.run(mid, p, xtol=1e-6)
        # Both are within xtol (mine) and ~1e-8 (SciPy) of x*.
        assert abs(res.x - ref.x) <= 1e-6 + 1e-7, (pid, res.x, ref.x)


# --------------------------------------------------------------------------------------
# Golden section
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PROBLEMS)
def test_golden_one_evaluation_per_iteration(pid):
    res = numopt.run("golden_section", problems.get(pid))
    assert res.n_fev == res.n_iter + 2
    for s in res.trace[1:]:
        assert len(s.info["evaluated"]) == 1
        x1, x2 = s.info["interior"]
        assert x1 < x2
        # The reused point is exactly one of the previous interior points.
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        assert set(prev.info["interior"]) & set(cur.info["interior"])


def test_golden_width_shrinks_by_one_over_phi():
    res = numopt.run("golden_section", problems.get("drug_concentration"), xtol=1e-6)
    widths = [s.info["bracket"][1] - s.info["bracket"][0] for s in res.trace]
    for w0, w1 in pairwise(widths):
        # Rounding in a + ρ(b - a) perturbs each width by ~ε·max|x| = ~2e-15 absolute.
        assert_allclose(w1, w0 / PHI, rtol=0, atol=1e-13)
    # The interior points sit at the golden fractions of the bracket.
    for s in res.trace:
        a, b = s.info["bracket"]
        x1, x2 = s.info["interior"]
        assert_allclose([(x1 - a) / (b - a), (x2 - a) / (b - a)], [sm.RHO, 1 - sm.RHO], atol=1e-6)


def test_golden_matches_scipy_golden():
    """Oracle: scipy's golden (NR §10.2) from the same initial pair reaches the same x*."""
    for pid in ("rational_1d", "drug_concentration", "sin_1d", "x_log_x"):
        p = problems.get(pid)
        a, b = p.bracket
        x1, x2 = a + sm.RHO * (b - a), b - sm.RHO * (b - a)
        xb = x1 if p.f(x1) < min(p.f(a), p.f(b)) else x2
        ref = scipy_scalar(p.f, bracket=(a, xb, b), method="golden", tol=1e-10)
        res = numopt.run("golden_section", p, xtol=1e-8)
        # Both are limited by the √ε floor of f-comparisons: |x - x*| ≲ 1e-8·(1 + |x*|).
        assert abs(res.x - ref.x) <= 5e-8 * (1 + abs(ref.x)), pid
        assert abs(res.x - x_star(pid)) <= 5e-8 * (1 + abs(x_star(pid)))


def test_golden_max_iter_failure():
    res = numopt.run("golden_section", problems.get("quadratic_1d"), max_iter=5)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=5)
    assert res.n_fev == 7


# --------------------------------------------------------------------------------------
# Fibonacci search
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PROBLEMS)
@pytest.mark.parametrize("xtol", [1e-2, 1e-5, 1e-8])
def test_fibonacci_plan_and_counts(pid, xtol):
    p = problems.get(pid)
    a, b = p.bracket
    r = 0.05
    res = numopt.run("fibonacci_search", p, xtol=xtol, eps_ratio=r)
    n = res.extra["n_planned"]
    fib = sm.fibonacci_numbers(1)
    while len(fib) <= n + 1:
        fib.append(fib[-1] + fib[-2])
    slack = sm.FIB_SLACK_ULPS * EPS * max(abs(a), abs(b))
    assert xtol > 2 * slack  # the absolute margin applies at these tolerances
    target = (1 + r) * (b - a) / (xtol - slack)
    # n is the smallest index ≥ 3 with F_n ≥ (1 + r) L / xtol' (F₀ = F₁ = 1).
    assert fib[n] >= target and (n == 3 or fib[n - 1] < target)
    assert res.n_fev == n == res.n_iter + 2
    assert res.converged
    a_f, b_f = res.extra["bracket"]
    assert b_f - a_f <= xtol
    # Width ratios L_{k+1}/L_k = F_{n-k}/F_{n-k+1} for the regular steps.
    widths = [s.info["bracket"][1] - s.info["bracket"][0] for s in res.trace]
    # Each end point carries a rounding error of a few ε·|x|; a width is only known to
    # that absolute accuracy, not relatively (it can be 1e-8 at x ≈ 5).
    w_err = 16 * np.finfo(float).eps * max(abs(a), abs(b))
    for k in range(1, n - 2):
        assert abs(widths[k] - widths[k - 1] * fib[n - k] / fib[n - k + 1]) <= w_err
        # Each interior point lies the fraction ``ratio`` of the width from the far end.
        info = res.trace[k].info
        ratio = info["ratio"]
        assert ratio == fib[n - k - 1] / fib[n - k]
        (lo, hi), (x1, x2) = info["bracket"], info["interior"]
        assert abs((hi - x1) - ratio * (hi - lo)) <= w_err
        assert abs((x2 - lo) - ratio * (hi - lo)) <= w_err
    # The last iteration reduces by F₂/F₃ = 2/3 and then halves (plus ε if [a, q] is
    # kept): L_last = L_prev/3 or L_prev/3 + ε.
    eps = res.extra["eps"]
    assert res.trace[-1].info["eps"] == eps
    last = widths[-1]
    assert math.isclose(last, widths[-2] / 3, rel_tol=1e-6) or math.isclose(
        last, widths[-2] / 3 + eps, rel_tol=1e-6
    )


def test_fibonacci_uses_no_more_evaluations_than_golden():
    """Fibonacci search is the optimal fixed-budget plan (Kiefer 1953)."""
    for pid in ("quadratic_1d", "rational_1d", "drug_concentration"):
        for xtol in (1e-3, 1e-6, 1e-9):
            fib = numopt.run("fibonacci_search", problems.get(pid), xtol=xtol)
            gold = numopt.run("golden_section", problems.get(pid), xtol=xtol)
            assert fib.n_fev <= gold.n_fev


def test_fibonacci_max_iter_failure():
    """max_iter = 4 allows 6 evaluations: the run is the complete 6-point plan (the optimal
    one for that budget, ending with the ε step) and reports converged=False."""
    res = numopt.run("fibonacci_search", problems.get("quadratic_1d"), max_iter=4)
    assert not res.converged and "max_iter + 2 = 6" in res.message
    assert_valid_result(res, max_iter=4)
    assert res.n_fev == 6 == res.extra["n_planned"] and res.n_iter == 4
    assert res.trace[-1].info["ratio"] == 0.5  # the plan's last (ε-shifted) step ran
    a, b = res.extra["bracket"]
    # F₆ = 13: final width L/F₆ or L/F₆ + ε with L = 5.
    assert math.isclose(b - a, 5 / 13, rel_tol=1e-12) or math.isclose(
        b - a, 5 / 13 + res.extra["eps"], rel_tol=1e-12
    )


@pytest.mark.parametrize("xtol", [1e-15, 1e-20, 1e-30, 1e-100])
def test_fibonacci_long_plan_keeps_order_and_minimizer(xtol):
    """Regression: Bazaraa's fixed-fraction placement amplified the reused point's rounding
    error by φ per step; with x* = 0 and xtol = 1e-30 the interior points crossed at
    k = 103, x* left the bracket and a negative final width passed the test."""
    res = numopt.run(
        "fibonacci_search", bare(lambda x: x * x, bracket=(-1.0, 2.0)), xtol=xtol, max_iter=2000
    )
    gold = numopt.run(
        "golden_section", bare(lambda x: x * x, bracket=(-1.0, 2.0)), xtol=xtol, max_iter=2000
    )
    assert res.converged, res.message
    assert_valid_result(res)
    for s in res.trace:
        a, b = s.info["bracket"]
        x1, x2 = s.info["interior"]
        assert a <= x1 < x2 <= b, s.k
        assert a <= 0.0 <= b, s.k
    a, b = res.extra["bracket"]
    assert 0 < b - a <= xtol and abs(res.x) <= xtol
    assert res.n_fev == res.extra["n_planned"] <= gold.n_fev


def test_fibonacci_tiny_xtol_terminates():
    """Regression: (1 + r)L/xtol = inf made the plan loop endless (xtol = 1e-308)."""
    assert len(sm.fibonacci_numbers(math.inf, n_max=50)) == 51
    assert sm.fibonacci_numbers(math.inf)[-1] > sm.FIB_MAX
    for xtol, bracket in [(1e-308, (0.0, 6.0)), (5e-324, (0.0, 1.0)), (1e-308, (-1.0, 2.0))]:
        res = numopt.run(
            "fibonacci_search", bare(lambda x: (x - 0.3) ** 2, bracket=bracket), xtol=xtol
        )
        assert_valid_result(res, max_iter=200)
        assert res.extra["n_planned"] <= 202
        # The run stops at the floating-point floor 4ε·max(|a|, |b|) around x* = 0.3.
        assert res.converged and "resolution" in res.message, res.message
        assert res.n_fev < res.extra["n_planned"]
        assert abs(res.x - 0.3) <= 1e-7


def test_fibonacci_capped_plan_reports_failure():
    """A plan longer than max_iter + 2 evaluations cannot reach xtol: converged=False."""
    res = numopt.run("fibonacci_search", bare(lambda x: x * x, bracket=(-1.0, 2.0)), xtol=1e-308)
    assert not res.converged and "max_iter + 2 = 202" in res.message
    assert res.n_fev == 202 and res.n_iter == 200
    a, b = res.extra["bracket"]
    assert a <= 0.0 <= b and b - a > 1e-308


def test_fibonacci_plan_at_equality_converges():
    """Regression: F₉ = 55 = (1 + 0.1)·5/0.1 exactly gave a final width 1 ulp above xtol
    and converged=False although the plan ran correctly."""
    res = numopt.run("fibonacci_search", problems.get("quadratic_1d"), xtol=0.1, eps_ratio=0.1)
    assert res.converged, res.message
    a, b = res.extra["bracket"]
    assert b - a <= 0.1 and a <= 2.0 <= b


def test_fibonacci_no_fit_judges_the_reduced_bracket():
    """Regression (Hypothesis counterexample): on (-7, -2) with xtol = 1.001·5/F₇₄ the
    last step found no double between p and b. The guard tested the previous bracket
    (5 ulps wide, floor 4ε·4.77 ≈ 4.8 ulps) and reported converged=False. The λ-μ
    comparison already justifies the reduced bracket, which is at the floor."""
    lo, hi = -7.0, -2.0
    c = lo + 0.4453125 * (hi - lo)
    xtol = 1.001 * (hi - lo) / sm.fibonacci_numbers(1e20)[74]
    res = numopt.run(
        "fibonacci_search",
        bare(lambda x: (x - c) * (x - c), bracket=(lo, hi)),
        xtol=xtol,
        eps_ratio=1e-3,
    )
    assert_valid_result(res)
    assert res.converged and "no point fits" in res.message, res.message
    a, b = res.extra["bracket"]
    last_a, last_b = res.trace[-1].info["bracket"]
    assert last_a <= a < b <= last_b and b - a < last_b - last_a
    assert b - a <= 4 * EPS * max(abs(a), abs(b))
    assert a <= c <= b and a <= res.x <= b
    assert res.n_fev == 2 + res.n_iter


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(4, 75),
    st.sampled_from([(0.0, 1.0), (-1.0, 2.0), (0.0, 12.0), (-7.0, -2.0)]),
    st.floats(0.0, 1.0),
    st.sampled_from([1e-3, 0.05, 0.5]),
)
def test_fibonacci_exact_plan_boundary_property(n, bracket, frac, r):
    """Hypothesis: with xtol = (1 + r)L/F_n exactly (the worst case of the plan rule) the run
    converges with final width ≤ xtol and the minimizer in every bracket."""
    fib = sm.fibonacci_numbers(1e20)
    lo, hi = bracket
    c = lo + frac * (hi - lo)
    xtol = (1 + r) * (hi - lo) / fib[n]
    res = numopt.run(
        "fibonacci_search",
        bare(lambda x: (x - c) * (x - c), bracket=bracket),
        xtol=xtol,
        eps_ratio=r,
    )
    assert res.converged, res.message
    a, b = res.extra["bracket"]
    assert 0 < b - a <= max(xtol, 4 * EPS * max(abs(a), abs(b)))
    # Comparisons are exact until f(x) - f(c) drops below rounding: |x - c| ≲ √ε·(1 + |c|).
    floor = 1e-7 * (1 + abs(c))
    for s in res.trace:
        a, b = s.info["bracket"]
        assert a - floor <= c <= b + floor
        x1, x2 = s.info["interior"]
        assert a <= x1 < x2 <= b


def test_fibonacci_shift_below_resolution_uses_next_double():
    """ε = r·L/F_n below the spacing of doubles at p would give q = p (no information);
    the last step then compares p with the next double."""
    fib = sm.fibonacci_numbers(1e20)
    r, n = 1e-3, 65
    xtol = (1 + r) * 3.0 / fib[n]
    res = numopt.run(
        "fibonacci_search",
        bare(lambda x: (x - 1.31) ** 2, bracket=(-1.0, 2.0)),
        xtol=xtol,
        eps_ratio=r,
    )
    assert res.extra["eps"] < math.ulp(1.31)
    last = res.trace[-1].info
    p, q = last["interior"]
    assert last["ratio"] == 0.5 and q == math.nextafter(p, math.inf) and last["eps"] == q - p
    assert res.converged, res.message
    a, b = res.extra["bracket"]
    assert b - a <= xtol


TINY = 5e-324  # the smallest subnormal: on [0, n·TINY] every point is a multiple of it


@pytest.mark.parametrize(
    ("mid", "n", "k", "phrase"),
    [
        ("fibonacci_search", 16, 1, "rounding broke the order"),
        ("fibonacci_search", 24, 13, "no point fits"),
        ("parabolic_interpolation", 34, 1, "cannot leave x_m"),
    ],
)
def test_guards_on_a_subnormal_grid(mid, n, k, phrase):
    """The guard paths that the 4ε·max(|a|,|b|) floor normally pre-empts: on a bracket of
    n subnormal spacings that floor underflows to 0, so the interior points collide. Each
    guard must stop with converged=False and keep the minimizer c = k·TINY in the bracket."""
    c = k * TINY
    res = numopt.run(mid, bare(lambda x: abs(x - c), bracket=(0.0, n * TINY)), xtol=TINY)
    assert not res.converged and phrase in res.message, res.message
    assert_valid_result(res)
    assert res.n_iter == res.trace[-1].k
    lo, hi = res.extra["bracket"]
    assert lo <= c <= hi and lo <= res.x <= hi
    assert hi - lo > TINY  # the width test did not pass


# --------------------------------------------------------------------------------------
# Dichotomous and ternary search
# --------------------------------------------------------------------------------------


def test_dichotomous_stops_when_rounding_decides_the_comparison():
    """Regression: with δ = 1e-13 the pair's f-difference f''·d·δ fell below rounding at
    d ≈ 4ε|f*|/(f''δ) ≈ 4e-2; the tie rule kept [a, x₂] and lost x* at k = 11, and the run
    reported converged=True with |x - x*| = 3.3e-3. Now it stops before that elimination."""
    pid = "drug_concentration"
    xs = x_star(pid)
    res = numopt.run("dichotomous_search", problems.get(pid), xtol=1e-10, delta_ratio=1e-3)
    assert not res.converged and "comparison resolution" in res.message, res.message
    assert_valid_result(res, max_iter=200)
    assert res.n_fev == 2 * (res.n_iter + 1)
    for s in res.trace:
        lo, hi = s.info["bracket"]
        assert lo <= xs <= hi, s.k
    # Every elimination used a comparison that rounding did not decide; the last pair did.
    for s in res.trace[:-1]:
        assert not sm._rounding_tie(*s.info["f_interior"]), s.k
    assert sm._rounding_tie(*res.trace[-1].info["f_interior"])
    lo, hi = res.extra["bracket"]
    assert hi - lo > 1e-10 and abs(res.x - xs) <= hi - lo
    # Golden section on the same problem and xtol meets its floor (oracle for x*).
    gold = numopt.run("golden_section", problems.get(pid), xtol=1e-10)
    assert gold.converged and abs(gold.x - xs) <= comparison_floor(pid)


def test_dichotomous_exact_symmetric_tie_also_stops():
    """Documented limit of the tie stop: on x² over (-1, 1) the first midpoint is x* and
    f(x₁) = f(x₂) exactly. In floating point this cannot be told from a rounded tie, so the
    run stops at k = 0 with converged=False; the result is still the bracket's best point."""
    res = numopt.run("dichotomous_search", bare(lambda x: x * x, bracket=(-1.0, 1.0)), xtol=1e-6)
    assert not res.converged and "comparison resolution" in res.message
    assert res.n_iter == 0 and res.n_fev == 2
    assert res.extra["bracket"] == [-1.0, 1.0]
    assert abs(res.x) <= 1e-7  # x₁ = -δ/2
    # The same f off-centre converges normally.
    res = numopt.run("dichotomous_search", bare(lambda x: x * x, bracket=(-1.0, 3.0)), xtol=1e-6)
    assert res.converged and abs(res.x) <= 1e-6


@settings(max_examples=1000, deadline=None)
@given(
    st.floats(-50.0, 50.0),
    st.floats(0.01, 0.99),
    st.floats(1e-2, 50.0),
    st.floats(-10.0, 10.0),
    st.floats(0.1, 10.0),
    st.floats(-10.0, -1.0),
    st.floats(-3.0, math.log10(0.9)),
)
def test_dichotomous_never_discards_the_minimizer_on_noise(
    center, frac, width, f_star, half_curv, log_xtol, log_ratio
):
    """Hypothesis over the whole UI range of (xtol, delta_ratio) and random scales: on
    f = A(x - c)² + f* every bracket holds c up to the floor h = √(8ε|f*|/(2A)) (times 2,
    plus the rounding of the points); a converged run meets max(xtol, h); otherwise the
    run stopped at a rounding-decided comparison and c is still in its bracket."""

    def f(x):
        return half_curv * (x - center) ** 2 + f_star

    a = center - frac * width
    b = a + width
    xtol, ratio = 10.0**log_xtol, 10.0**log_ratio
    h = 2 * math.sqrt(8 * EPS * abs(f_star) / (2 * half_curv)) + 8 * EPS * (abs(a) + abs(b))
    if 2.0 * (0.5 * ratio * xtol) >= b - a:  # δ ≥ b - a: the pair would leave [a, b]
        with pytest.raises(ValueError, match="smaller than the bracket width"):
            numopt.run("dichotomous_search", bare(f, bracket=(a, b)), xtol=xtol, delta_ratio=ratio)
        return
    res = numopt.run(
        "dichotomous_search", bare(f, bracket=(a, b)), xtol=xtol, delta_ratio=ratio, max_iter=500
    )
    assert res.n_fev == 2 * (res.n_iter + 1)
    for s in res.trace:
        lo, hi = s.info["bracket"]
        assert lo - h <= center <= hi + h, (s.k, lo, hi)
    lo, hi = res.extra["bracket"]
    if res.converged:
        assert abs(res.x - center) <= max(xtol, h)
    else:
        assert "comparison resolution" in res.message, res.message
        assert abs(res.x - center) <= hi - lo + h


def test_dichotomous_width_recurrence_and_counts():
    xtol, ratio = 1e-6, 0.1
    delta = ratio * xtol
    res = numopt.run(
        "dichotomous_search", problems.get("rational_1d"), xtol=xtol, delta_ratio=ratio
    )
    assert res.n_fev == 2 * (res.n_iter + 1)
    widths = [s.info["bracket"][1] - s.info["bracket"][0] for s in res.trace]
    for w0, w1 in pairwise(widths):
        assert_allclose(w1, w0 / 2 + delta / 2, rtol=0, atol=1e-14)
    for s in res.trace:
        x1, x2 = s.info["interior"]
        a, b = s.info["bracket"]
        assert_allclose(x2 - x1, delta, rtol=1e-6)
        assert_allclose(0.5 * (x1 + x2), 0.5 * (a + b), rtol=0, atol=1e-14)
    # Iteration count from the recurrence: L_k = (L_0 - δ)/2^k + δ ≤ xtol.
    a, b = problems.get("rational_1d").bracket
    assert res.n_iter == math.ceil(math.log2((b - a - delta) / (xtol - delta)))


@pytest.mark.parametrize(
    ("bracket", "xtol"), [((0.0, 0.005), 0.1), ((0.3, 0.305), 0.1), ((0.36, 0.3601), 1e-2)]
)
def test_dichotomous_rejects_delta_not_below_the_bracket_width(bracket, xtol):
    """Regression (audit): with δ = delta_ratio·xtol ≥ b - a the setup pair m ± δ/2 lay
    outside [a, b]. On x_log_x (0, 0.005), xtol 0.1 the run evaluated f(-0.0025) = nan; on
    (0.3, 0.305) it returned converged=True with x = 0.3075 outside the bracket. Now this
    is invalid input. Golden section on the same input converges inside the bracket."""
    p = problems.get("x_log_x")
    with pytest.raises(ValueError, match="smaller than the bracket width"):
        numopt.run("dichotomous_search", p, bracket=bracket, xtol=xtol)
    gold = numopt.run("golden_section", p, bracket=bracket, xtol=xtol)
    assert gold.converged and bracket[0] <= gold.x <= bracket[1]


def test_dichotomous_delta_equal_to_the_width_is_rejected_and_just_below_is_inside():
    p = problems.get("x_log_x")
    # δ = 0.5·0.01 = 0.005 = b - a: the pair would be the end points a and b.
    with pytest.raises(ValueError, match="smaller than the bracket width"):
        numopt.run("dichotomous_search", p, bracket=(0.0, 0.005), xtol=0.01, delta_ratio=0.5)
    # δ = 0.0049 < b - a: the pair is (0.00005, 0.00495), strictly inside, and the run
    # converges at k = 0 (b - a ≤ xtol) without leaving the bracket.
    g, pts = recording(p.f)
    res = numopt.run(
        "dichotomous_search", bare(g, bracket=(0.0, 0.005)), xtol=0.01, delta_ratio=0.49
    )
    assert_valid_result(res)
    assert res.converged and res.n_iter == 0 and res.n_fev == 2
    assert all(0.0 < z < 0.005 for z in pts), pts
    assert 0.0 <= res.x <= 0.005


@settings(max_examples=1000, deadline=None)
@given(
    st.floats(-100.0, 100.0),
    st.floats(-12.0, 1.0),
    st.floats(1e-12, 1.0),
    st.floats(0.01, 0.99),
    st.floats(-3.0, math.log10(0.9)),
    st.floats(0.0, 1.0),
)
def test_dichotomous_evaluates_only_inside_the_bracket(a, log_w, frac_d, frac_c, log_ratio, sl):
    """Hypothesis: for every δ < b - a (including δ within ulps of b - a, where rounding
    of m ± δ/2 used to leave the bracket) every evaluated point lies in [a, b], every trace
    pair lies in its bracket, and x lies in the final bracket; δ ≥ b - a raises."""
    b = a + 10.0**log_w
    if not a < b:
        return
    width = b - a
    ratio = 10.0**log_ratio
    # δ ranges from 1e-12·width to (just above) width; xtol = δ/ratio.
    delta = width * (1.0 - frac_d) if sl < 0.5 else width * (1.0 + frac_d)
    xtol = delta / ratio
    center = a + frac_c * width

    def f(x):
        return (x - center) ** 2

    g, pts = recording(f)
    prob = bare(g, bracket=(a, b))
    resolution = 2.0 * 0.5 * ratio * xtol <= sm.RESOLUTION_ULPS * EPS * max(abs(a), abs(b))
    if resolution or 2.0 * (0.5 * ratio * xtol) >= width:
        with pytest.raises(ValueError):
            numopt.run("dichotomous_search", prob, xtol=xtol, delta_ratio=ratio, max_iter=500)
        return
    res = numopt.run("dichotomous_search", prob, xtol=xtol, delta_ratio=ratio, max_iter=500)
    assert all(a <= z <= b for z in pts)
    for s in res.trace:
        lo, hi = s.info["bracket"]
        x1, x2 = s.info["interior"]
        assert lo <= x1 < x2 <= hi, s.k
    lo, hi = res.extra["bracket"]
    assert a <= lo <= res.x <= hi <= b


def test_ternary_width_shrinks_by_two_thirds():
    res = numopt.run("ternary_search", problems.get("x_log_x"), xtol=1e-6)
    assert res.n_fev == 2 * (res.n_iter + 1)
    widths = [s.info["bracket"][1] - s.info["bracket"][0] for s in res.trace]
    for w0, w1 in pairwise(widths):
        assert_allclose(w1, 2 * w0 / 3, rtol=0, atol=1e-14)
    assert res.n_iter == math.ceil(math.log(2.0 / 1e-6) / math.log(1.5))


@pytest.mark.parametrize("mid", ["dichotomous_search", "ternary_search"])
def test_two_point_max_iter_failure(mid):
    res = numopt.run(mid, problems.get("sin_1d"), max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=3)
    assert res.n_fev == 8


# --------------------------------------------------------------------------------------
# Successive parabolic interpolation
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    st.lists(st.floats(-10, 10), min_size=3, max_size=3, unique=True),
    st.lists(st.floats(-10, 10), min_size=3, max_size=3),
)
def test_parabola_vertex_formulations_agree(xs, fs):
    """NR eq. 10.3.1, the centred divided-difference form and numpy.polyfit agree."""
    xl, xm, xr = sorted(xs)
    if min(xm - xl, xr - xm) < 1e-2:
        return  # ill-conditioned fit: the three formulas legitimately differ
    fl, fm, fr = fs
    coef = np.polyfit([xl, xm, xr], [fl, fm, fr], 2)  # oracle: c2 x² + c1 x + c0
    if abs(coef[0]) < 1e-3:
        return  # nearly linear: the vertex is ill-conditioned (|u| ~ 1/c2)
    u_np = -coef[1] / (2 * coef[0])
    u_nr = sm.parabola_vertex(xl, xm, xr, fl, fm, fr)
    par = sm._parabola(xm, fm, xl, fl, xr, fr)
    assert par is not None
    c0, c1, c2 = par["coef"]
    u_dd = xm - c1 / (2 * c2)
    # κ of the vertex ~ (|c1| + |c2|·|x|)/|c2|; 1e-9 relative to that scale is ~1e5 ε.
    scale = 1 + abs(u_np) + abs(coef[1] / coef[0])
    assert abs(u_nr - u_np) <= 1e-9 * scale
    assert abs(u_dd - u_np) <= 1e-9 * scale
    # The centred parabola interpolates the three points.
    for x, fx in ((xl, fl), (xm, fm), (xr, fr)):
        assert_allclose(c0 + c1 * (x - xm) + c2 * (x - xm) ** 2, fx, rtol=1e-9, atol=1e-9)


def test_parabolic_is_exact_on_a_quadratic():
    res = numopt.run("parabolic_interpolation", problems.get("quadratic_1d"), xtol=1e-6)
    first = res.trace[1]
    assert first.info["step"] == "parabolic"
    assert first.info["trial"] == 2.0  # vertex of (x - 2)² + 1 through 0, 2.5, 5
    assert res.converged and abs(res.x - 2.0) <= 1e-6


@pytest.mark.parametrize("pid", PROBLEMS)
def test_parabolic_invariants(pid):
    xtol = 1e-6
    res = numopt.run("parabolic_interpolation", problems.get(pid), xtol=xtol)
    assert res.n_fev == res.n_iter + 3
    pattern_seen = False
    for s in res.trace:
        xl, xm, xr = s.info["triple"]
        assert xl < xm < xr
        pattern_seen = pattern_seen or s.info["pattern"]
        if pattern_seen:  # once established, the three-point pattern is never lost
            assert s.info["pattern"]
        if s.k > 0:
            assert s.info["step"] in ("bisect", "parabolic", "probe", "golden")
            prev = res.trace[s.k - 1].info
            # No evaluation closer to the previous x_m than δ = tol/2 (minimum step).
            if s.info["step"] in ("parabolic", "probe"):
                # δ = tol/2 ≥ xtol/2; the rounding fallback (segment midpoint) is > δ/2.
                assert abs(s.info["trial"] - prev["triple"][1]) >= 0.25 * xtol * (1 - 1e-6)
                par = s.info["parabola"]
                assert par is not None and par["center"] == prev["triple"][1]
                # The vertex is the minimizer of the plotted parabola.
                assert par["coef"][2] >= 0
    if res.converged:
        xl, xm, xr = res.trace[-1].info["triple"]
        assert max(xm - xl, xr - xm) <= 0.5 * xtol


def test_parabolic_on_flat_valley_stops_in_the_underflow_region():
    """f = exp(-1/x²) underflows to exactly 0 for |x| < 0.0367: like every f-comparison
    method, SPI converges to some point of that computed-minimum region (before the
    progress test it stalled there until max_iter)."""
    res = numopt.run("parabolic_interpolation", problems.get("flat_valley"))
    assert res.converged, res.message
    assert_valid_result(res, max_iter=200)
    assert res.fun == 0.0 and abs(res.x) < 0.0367
    gold = numopt.run("golden_section", problems.get("flat_valley"))
    assert gold.fun == 0.0 and abs(gold.x) < 0.0367


@pytest.mark.parametrize(("lo", "hi", "c"), [(0.0, 10.0, 3.7), (-3.0, 5.0, 1.88), (0.0, 4.0, 1.2)])
def test_parabolic_progress_test_prevents_stalling(lo, hi, c):
    """Regression: on the asymmetric f = exp(5(x - c)) - 5(x - c) the steep end x_r stayed
    fixed and the accepted vertex steps crept toward x* (500 iterations, |x - x*| = 0.88 on
    (-3, 5)). Brent's test |vertex - x_m| < ½·(step before last) forces golden steps."""

    def f(x):
        return math.exp(5 * (x - c)) - 5 * (x - c)

    res = numopt.run("parabolic_interpolation", bare(f, bracket=(lo, hi)), xtol=1e-6)
    gold = numopt.run("golden_section", bare(f, bracket=(lo, hi)), xtol=1e-6)
    assert res.converged, res.message
    assert abs(res.x - c) <= 1e-6
    assert res.n_fev <= 2 * gold.n_fev
    steps = [s.info["step"] for s in res.trace[1:]]
    assert "golden" in steps
    for s in res.trace[1:]:
        info, prev = s.info, res.trace[s.k - 1].info
        if s.k <= 2:
            assert info["max_step"] is None
            continue
        assert info["max_step"] is not None
        if info["step"] in ("parabolic", "probe"):
            assert abs(info["vertex"] - prev["triple"][1]) < info["max_step"]


def test_parabolic_not_fooled_by_symmetric_data():
    """x_log_x: the second vertex equals x_m = 0.5 exactly; the method must not stop there."""
    res = numopt.run("parabolic_interpolation", problems.get("x_log_x"), xtol=1e-6)
    assert res.trace[2].info["step"] == "probe"
    assert res.converged and abs(res.x - math.exp(-1)) <= 1e-6


def test_parabolic_end_point_singularity_is_reported():
    """Regression (audit): parabolic_interpolation evaluates f at the bracket ends, so a
    bracket ending on the pole of rational_1d (x = -1) stops at k = 0. The precondition is
    documented and the message names the end point; the methods that evaluate only interior
    points converge on the same input (oracle: x* = √6 - 1)."""
    p = problems.get("rational_1d")
    res = numopt.run("parabolic_interpolation", p, bracket=(-1.0, 6.0))
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0 and res.n_fev == 3
    assert "not finite" in res.message and "bracket end point a" in res.message, res.message
    assert res.x == -1.0
    for mid in (*INTERVAL, "brent_minimize"):
        other = numopt.run(mid, p, bracket=(-1.0, 6.0))
        assert other.converged and abs(other.x - x_star("rational_1d")) <= 1e-6, mid
    # A bracket strictly inside the domain converges.
    res = numopt.run("parabolic_interpolation", p, bracket=(-0.999, 6.0))
    assert res.converged and abs(res.x - x_star("rational_1d")) <= 1e-7
    # The right end: f = x² + 1/(1 - x) has a pole at b = 1.
    pole = bare(lambda x: math.nan if x >= 1.0 else x * x + 1.0 / (1.0 - x), bracket=(-2.0, 1.0))
    res = numopt.run("parabolic_interpolation", pole)
    assert not res.converged and res.n_iter == 0 and "bracket end point b" in res.message
    spec = next(s for s in numopt.list_methods("scalar") if s.id == "parabolic_interpolation")
    assert "both bracket ends" in spec.summary


# --------------------------------------------------------------------------------------
# Brent
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PROBLEMS)
@pytest.mark.parametrize("xtol", [1e-5, 1e-8])
def test_brent_reproduces_scipy_fminbound_exactly(pid, xtol):
    """Oracle: SciPy's 'bounded' method is the same FMM fmin; with SciPy's √(2.2e-16) as
    rtol the evaluation sequences must be bit-identical."""
    p = problems.get(pid)
    g_ref, ref_pts = recording(p.f)
    ref = scipy_scalar(
        g_ref, bounds=p.bracket, method="bounded", options={"xatol": xtol, "maxiter": 500}
    )
    g, pts = recording(p.f)
    res = numopt.run(
        "brent_minimize", bare(g, bracket=p.bracket), xtol=xtol, rtol=math.sqrt(2.2e-16)
    )
    assert pts == ref_pts
    assert res.x == ref.x and res.fun == ref.fun
    assert res.n_fev == ref.nfev == res.n_iter + 1
    assert res.converged


@pytest.mark.parametrize("pid", PROBLEMS)
def test_brent_info_geometry(pid):
    res = numopt.run("brent_minimize", problems.get(pid))
    for s in res.trace[1:]:
        a, b = s.info["bracket"]
        x = s.info["xwv"][0]
        fx, fw = s.info["f_xwv"][:2]
        assert a <= x <= b and fx <= fw and fx == s.fun
        assert s.info["step"] in ("parabolic", "golden")
        par, vert = s.info["parabola"], s.info["vertex"]
        prev = res.trace[s.k - 1].info
        if s.info["step"] == "parabolic":
            assert vert is not None
        px, pw, pv = prev["xwv"]
        spread = max(px, pw, pv) - min(px, pw, pv)
        gap = min(abs(px - pw), abs(px - pv), abs(pw - pv))
        if par is not None and vert is not None and gap > 1e-3:
            # Brent's x + p/q equals the vertex of the plotted parabola (well-separated points).
            assert par["center"] == px
            _, c1, c2 = par["coef"]
            assert c2 != 0
            assert (
                abs(par["center"] - c1 / (2 * c2) - vert) <= 1e-8 * (1 + abs(vert)) * spread / gap
            )


def test_brent_beats_golden():
    for pid in ("quadratic_1d", "rational_1d", "drug_concentration", "sin_1d"):
        brent = numopt.run("brent_minimize", problems.get(pid))
        gold = numopt.run("golden_section", problems.get(pid))
        assert brent.n_fev < gold.n_fev / 2


def test_brent_tolerance_is_relative_far_from_the_origin():
    """Documented (no defect): Brent's guarantee is 2(rtol·|x| + xtol/3), not xtol. On
    (1e8, 1e8 + 1) 2·tol ≈ 2.98 exceeds the bracket, so Brent stops at k = 0 after one
    evaluation, exactly like SciPy's bounded method; golden section meets xtol."""
    xs = 1e8 + 0.25

    def f(x):
        return (x - xs) ** 2

    p = bare(f, bracket=(1e8, 1e8 + 1))
    res = numopt.run("brent_minimize", p, xtol=1e-12)
    ref = scipy_scalar(f, bounds=(1e8, 1e8 + 1), method="bounded", options={"xatol": 1e-12})
    assert res.converged and res.n_iter == 0 and res.n_fev == 1 == ref.nfev
    assert res.x == ref.x
    tol = math.sqrt(EPS) * abs(res.x) + 1e-12 / 3
    assert 1e-12 < abs(res.x - xs) <= 2 * tol
    gold = numopt.run("golden_section", p, xtol=1e-12)
    # golden is limited only by the floor 4ε·1e8 ≈ 9e-8 of the bracket width.
    assert gold.converged and abs(gold.x - xs) <= 4 * EPS * 1e8


def test_brent_max_iter_failure():
    res = numopt.run("brent_minimize", problems.get("abs_shifted"), max_iter=4)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=4)


# --------------------------------------------------------------------------------------
# Newton
# --------------------------------------------------------------------------------------


def test_newton_exact_on_quadratic():
    res = numopt.run("newton_1d", problems.get("quadratic_1d"))
    assert res.converged and res.n_iter == 1 and res.x == 2.0
    assert res.n_fev == res.n_gev == res.n_hev == 2
    assert res.trace[1].info["step"] == "newton"
    assert res.trace[1].info["parabola"] == {"center": 0.5, "coef": [3.25, -3.0, 1.0]}


@pytest.mark.parametrize(
    "pid", ["drug_concentration", "rational_1d", "x_log_x", "quartic_1d", "multimodal_1d"]
)
def test_newton_matches_scipy_newton_on_f_prime(pid):
    """Oracle: pure Newton on f' (scipy.optimize.newton with fprime=f'') gives the same x*."""
    p = problems.get(pid)
    ref = so.newton(p.grad, p.x0, fprime=p.hess, tol=1e-15, maxiter=100)
    res = numopt.run("newton_1d", p, gtol=1e-13)
    assert res.converged
    assert all(s.info["step"] == "newton" for s in res.trace[1:])
    assert abs(res.x - ref) <= 1e-12 * (1 + abs(ref))
    assert min(abs(res.x - xm) for xm in p.minima) <= 1e-10


def test_newton_quadratic_convergence():
    p = problems.get("drug_concentration")
    res = numopt.run("newton_1d", p, gtol=1e-14)
    xs = p.minima[0]
    err = [abs(s.x - xs) for s in res.trace]
    # e_{k+1} ≤ C e_k² with C ≈ |f'''/(2f'')| at x* (≈ 1 here); check while e is above noise.
    for e0, e1 in pairwise(err):
        if 1e-7 < e0 < 0.5:
            assert e1 <= 2.0 * e0**2


@pytest.mark.parametrize(
    ("pid", "x0"), [("sin_1d", 3.0), ("quartic_1d", 0.0), ("drug_concentration", 6.0)]
)
def test_newton_negative_curvature_safeguard(pid, x0):
    p = problems.get(pid)
    assert p.hess(x0) <= 0
    res = numopt.run("newton_1d", p, x0=x0)
    first = res.trace[1].info
    assert first["step"] == "gradient" and first["trials"] and first["alpha"] is not None
    # Armijo: f decreased by at least c₁ α f'².
    g0, f0 = p.grad(x0), p.f(x0)
    assert res.trace[1].fun <= f0 - sm.ARMIJO_C1 * first["alpha"] * g0 * g0
    for s in res.trace[1:]:
        if s.info["step"] == "gradient":
            assert res.trace[s.k - 1].info["h"] <= 0
        else:
            assert res.trace[s.k - 1].info["h"] > 0
    assert res.converged
    assert min(abs(res.x - xm) for xm in p.minima) <= 1e-9
    assert res.n_fev == res.n_gev + sum(
        len(s.info["trials"]) - 1 for s in res.trace if s.info["trials"]
    )


def test_newton_step_is_not_globalized_near_an_inflection_point():
    """From x0 = 2 two gradient steps land at x ≈ 3.16 where 0 < f'' ≈ 0.02: the pure
    Newton step then jumps ≈ 44 to another period of sin; it still finds a minimizer."""
    res = numopt.run("newton_1d", problems.get("sin_1d"), x0=2.0)
    assert res.trace[3].info["step"] == "newton" and (res.trace[3].step_size or 0.0) > 40
    assert res.converged
    assert math.isclose(math.sin(res.x), -1.0, abs_tol=1e-15) and abs(math.cos(res.x)) <= 1e-8


def test_newton_stops_at_a_local_maximum():
    res = numopt.run("newton_1d", problems.get("sin_1d"), x0=math.pi / 2)
    assert not res.converged and "maximum" in res.message
    assert_valid_result(res)


def test_newton_oscillates_on_the_kink():
    res = numopt.run("newton_1d", problems.get("abs_shifted"))
    assert not res.converged and "max_iter" in res.message
    assert {round(s.x, 12) for s in res.trace[1:]} == {-1.0, 1.0}


def test_newton_leaves_the_domain():
    res = numopt.run("newton_1d", problems.get("x_log_x"), x0=2.0)
    assert not res.converged and "non-finite" in res.message
    assert res.n_iter == 1 and res.x < 0
    assert_valid_result(res)


def test_newton_gradient_backtracking_failure():
    """f = (x - 1)(3 - x) on its domain x ≥ 1 (nan below), x0 = 1: f'' < 0 forces a
    gradient step, and -f'(1) = -2 points out of the domain. Every trial is nan until
    1 - 2α rounds to 1, where f does not decrease: Armijo fails 60 times."""
    p = Problem(
        id="t",
        name="t",
        latex="",
        f=lambda x: (x - 1.0) * (3.0 - x) if x >= 1.0 else math.nan,
        grad=lambda x: 4.0 - 2.0 * x,
        hess=lambda x: -2.0,
        dim=1,
        domain=(1.0, 3.0),
        x0=1.0,
    )
    res = numopt.run("newton_1d", p)
    assert not res.converged
    assert res.message.startswith(f"gradient-step backtracking failed after {sm.MAX_BACKTRACK}")
    assert_valid_result(res, max_iter=100)
    assert res.n_iter == 0 and res.x == 1.0 and res.fun == 0.0
    assert res.n_fev == 1 + sm.MAX_BACKTRACK and res.n_gev == res.n_hev == 1


def test_newton_flat_minimizer_passes_the_gradient_test_far_from_x_star():
    res = numopt.run("newton_1d", problems.get("flat_valley"))
    assert res.converged and 0.1 < res.x < 0.3  # documented weakness of the gradient test


def test_newton_converges_to_an_inflection_point_of_x_cubed():
    """Documented weakness: f = x³ has no minimizer, but near x = 0 both |f'| ≤ gtol and
    f'' ≥ 0 hold, so the necessary-condition test reports converged=True."""
    res = numopt.run(
        "newton_1d",
        Problem(
            id="t",
            name="t",
            latex="",
            f=lambda x: x**3,
            grad=lambda x: 3 * x * x,
            hess=lambda x: 6 * x,
            dim=1,
            domain=(-1, 1),
            x0=1.0,
        ),
    )
    assert res.converged
    assert 0 < res.x < 1e-4  # Newton halves x each step: x_k = 2^-k
    assert res.trace[-1].info["g"] <= 1e-8 and res.trace[-1].info["h"] >= 0
    assert all(s.info["step"] == "newton" for s in res.trace[1:])
    assert all(s.x == 0.5**s.k for s in res.trace)


def test_newton_bare_callable_uses_finite_differences():
    res = numopt.run("newton_1d", bare(lambda x: (x - 1.5) ** 4 + (x - 1.5) ** 2, x0=0.0))
    assert res.converged and abs(res.x - 1.5) <= 1e-6
    assert res.n_gev == 0 and res.n_hev == 0 and res.n_fev > res.n_iter + 1


# --------------------------------------------------------------------------------------
# bracket_minimum
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PROBLEMS)
@pytest.mark.parametrize("step", [0.1, -0.3, 1.0])
def test_bracket_reproduces_scipy_bracket_exactly(pid, step):
    """Oracle: scipy.optimize.bracket is NR mnbrak; the evaluation sequence is identical."""
    p = problems.get(pid)
    g, pts = recording(p.f)
    res = numopt.run("bracket_minimum", bare(g, x0=p.x0), step=step, grow_limit=100.0)
    g_ref, ref_pts = recording(p.f)
    try:
        xa, xb, xc, *_ = so.bracket(g_ref, xa=p.x0, xb=p.x0 + step, grow_limit=100.0)
    except (BracketError, RuntimeError, ValueError):
        assert not res.converged  # e.g. the walk leaves the domain (nan) on x_log_x
        return
    assert res.converged
    assert pts == ref_pts
    assert res.extra["triple"] == sorted([xa, xb, xc])
    assert res.x == xb


@pytest.mark.parametrize("pid", PROBLEMS)
def test_bracket_trace_accounting(pid):
    res = numopt.run("bracket_minimum", problems.get(pid))
    assert res.n_fev == sum(len(s.info["trials"]) for s in res.trace)
    assert res.trace[0].info["step"] == "init" and len(res.trace[0].info["trials"]) == 3
    for s in res.trace[1:]:
        assert 1 <= len(s.info["trials"]) <= 2
        assert s.info["step"] in ("golden", "parabolic", "parabolic_far", "limit")
    if res.converged:
        (a, b, c), (fa, fb, fc) = res.extra["triple"], res.extra["f_triple"]
        assert a < b < c and fb <= fa and fb <= fc


@settings(max_examples=1000, deadline=None)
@given(
    st.floats(-50, 50),
    st.floats(-50, 50),
    st.floats(1e-3, 5.0),
    st.sampled_from([-1.0, 1.0]),
    st.floats(0.0, 3.0),
)
def test_bracket_property_on_convex_functions(center, x0, step, sign, quartic):
    """Hypothesis: on f = (x - c)² + q(x - c)⁴ a valid bracket around c is always found."""

    def f(x):
        return (x - center) ** 2 + quartic * (x - center) ** 4

    res = numopt.run("bracket_minimum", bare(f, x0=x0), step=sign * step, max_iter=200)
    assert res.converged, res.message
    (a, b, c), (fa, fb, fc) = res.extra["triple"], res.extra["f_triple"]
    assert a < b < c and fb <= fa and fb <= fc
    assert a <= center <= c


def test_bracket_failure_paths():
    res = numopt.run("bracket_minimum", bare(lambda x: -x, x0=0.0), max_iter=10)
    assert not res.converged and "unbounded" in res.message
    assert_valid_result(res, max_iter=10)
    res = numopt.run("bracket_minimum", problems.get("x_log_x"), x0=0.05, step=-0.1)
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize(("x0", "step"), [(1.0, 1e-17), (1e16, 1e-6), (-3e10, -1e-6)])
def test_bracket_step_lost_in_rounding_is_invalid_input(x0, step):
    """Regression: x0 + step == x0 gave a = b = c and converged=True after 3 evaluations
    (f(b) ≤ f(c) holds trivially). SciPy's bracket (NR mnbrak) raises BracketError."""

    def f(x):
        return (x - 3.0) ** 2

    with pytest.raises(BracketError):
        so.bracket(f, xa=x0, xb=x0 + step)
    with pytest.raises(ValueError, match="lost in rounding"):
        numopt.run("bracket_minimum", bare(f, x0=x0), step=step)


def test_bracket_grow_limit_must_exceed_one():
    """u_lim = b + grow_limit·(c - b) must lie beyond c: grow_limit = 1 re-evaluated c and
    reported the collapsed triple (b, c, c) as a bracket."""
    for gl in (1.0, 0.5, 0.0, -2.0, math.inf):
        with pytest.raises(ValueError, match="grow_limit"):
            numopt.run("bracket_minimum", problems.get("rational_1d"), grow_limit=gl)


@pytest.mark.parametrize(("pid", "step"), [("rational_1d", 0.1), ("drug_concentration", 0.1)])
def test_bracket_collapsed_triple_is_not_a_bracket(pid, step):
    """With grow_limit one ulp above 1, u_lim rounds onto c; the guard in done() must report
    converged=False instead of the trivial f(b) ≤ f(c) with b = c."""
    gl = math.nextafter(1.0, 2.0)
    res = numopt.run("bracket_minimum", problems.get(pid), step=step, grow_limit=gl)
    assert not res.converged and "collapsed" in res.message, res.message
    assert_valid_result(res, max_iter=50)
    a, b, c = res.extra["triple"]
    assert a <= b <= c and not a < b < c


def test_bracket_extra_triple_is_sorted_on_every_path():
    """Regression: on failure extra["triple"] stayed in search order (decreasing for a walk
    to the left) and a non-finite stop had no triple."""
    # f = x: no minimizer; the walk goes left, so the search order decreases.
    res = numopt.run("bracket_minimum", bare(lambda x: x, x0=0.0), max_iter=3)
    assert not res.converged
    last = res.trace[-1].info["triple"]
    assert last[0] > last[1] > last[2]
    assert res.extra["triple"] == sorted(last)
    assert res.extra["f_triple"] == sorted(res.trace[-1].info["f_triple"])  # f = x
    # Non-finite stop inside the loop (f = x on x > 0 walks left out of its domain): the
    # last finite triple, sorted.
    res = numopt.run("bracket_minimum", bare(lambda x: x if x > 0 else math.nan, x0=1.0), step=-0.1)
    assert not res.converged and "not finite" in res.message and res.n_iter >= 1
    t, ft = res.extra["triple"], res.extra["f_triple"]
    assert t == sorted(t) and all(math.isfinite(v) for v in ft)
    # Non-finite f(c) at k = 0: the triple holds that c, sorted.
    res = numopt.run(
        "bracket_minimum", bare(lambda x: math.log(x) if x > 0 else math.nan, x0=1.0), step=-0.5
    )
    assert not res.converged and res.n_iter == 0 and res.n_fev == 3
    assert res.extra["triple"] == sorted(res.extra["triple"])
    # Non-finite f(x0): no triple was formed.
    res = numopt.run("bracket_minimum", bare(lambda x: math.nan, x0=1.0))
    assert not res.converged and "triple" not in res.extra


@pytest.mark.parametrize("pid", PROBLEMS)
def test_bracket_info_is_a_true_bracket_only_on_the_converged_step(pid):
    """Info key ``bracket`` of bracket_minimum is the search interval [min, max] of the
    triple; the bracket condition f(b) ≤ f(a), f(c) holds on the final converged step."""
    res = numopt.run("bracket_minimum", problems.get(pid))
    for s in res.trace:
        assert s.info["bracket"] == [min(s.info["triple"]), max(s.info["triple"])]
    if res.converged:
        fa, fb, fc = res.trace[-1].info["f_triple"]
        assert fb <= fa and fb <= fc


# --------------------------------------------------------------------------------------
# Hypothesis: interval elimination keeps the minimizer
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from((*INTERVAL, "parabolic_interpolation", "brent_minimize")),
    st.floats(-100, 100),
    st.floats(0.01, 0.99),
    st.floats(1e-2, 1e2),
    st.floats(0.0, 2.0),
    st.sampled_from([1e-3, 1e-5]),
    st.sampled_from(["kink", "exp", "exp_neg"]),
)
def test_interval_methods_property(mid, center, frac, width, kink, xtol, shape):
    """Hypothesis: on a strictly convex f with c anywhere in a random bracket, every bracket
    contains c, the result converges and |x - c| ≤ xtol. Shapes: the symmetric
    (x-c)² + k|x-c| and the asymmetric smooth exp(t) - t with t = ±s(x - c), s = 1 + 2k
    (its steep side stalled parabolic interpolation before the progress test)."""
    s_exp = 1.0 + 2.0 * kink
    sign = -1.0 if shape == "exp_neg" else 1.0

    def f(x):
        if shape == "kink":
            return (x - center) ** 2 + kink * abs(x - center)
        t = sign * s_exp * (x - center)
        return math.exp(t) - t

    a = center - frac * width
    b = a + width
    res = numopt.run(mid, bare(f, bracket=(a, b)), xtol=xtol, max_iter=500)
    # exp shapes: f* = 1 and f'' = s², so comparisons are decided by rounding within
    # √(2ε f*/f'') ≤ 2.1e-8 of c (module docstring, "Accuracy floor"); the kink is exact.
    floor = 0.0 if shape == "kink" else 3 * math.sqrt(2 * EPS) / s_exp
    for s in res.trace:
        lo, hi = s.info["bracket"]
        assert lo - floor <= center <= hi + floor
    if not res.converged:
        # Only dichotomous search may stop early: when its pair ties to rounding (e.g. the
        # kink with c exactly at a midpoint), with c still in the bracket.
        assert mid == "dichotomous_search" and "comparison resolution" in res.message, res.message
        return
    tol = (
        xtol
        if mid != "brent_minimize"
        else 2 * (math.sqrt(np.finfo(float).eps) * abs(res.x) + xtol / 3)
    )
    assert abs(res.x - center) <= tol + 1e-12 + floor


# --------------------------------------------------------------------------------------
# Input validation and non-finite values
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mid", BRACKETED)
def test_invalid_input_raises(mid):
    p = problems.get("quadratic_1d")
    with pytest.raises(ValueError):
        numopt.run(mid, p, bracket=(1.0, 1.0))
    with pytest.raises(ValueError):
        numopt.run(mid, p, bracket=(2.0, 1.0))
    with pytest.raises(ValueError):
        numopt.run(mid, p, bracket=(0.0, math.inf))
    with pytest.raises(ValueError):
        numopt.run(mid, p, xtol=0.0)
    with pytest.raises(ValueError):
        numopt.run(mid, bare(lambda x: x * x))  # no bracket anywhere
    with pytest.raises(ValueError, match="overflows"):
        numopt.run(mid, p, bracket=(-1e308, 1e308))  # b - a = inf
    with pytest.raises(ValueError):
        numopt.run(mid, p, max_iter=0)


def test_other_invalid_inputs():
    p = problems.get("quadratic_1d")
    with pytest.raises(ValueError):
        numopt.run("fibonacci_search", p, eps_ratio=1.0)
    with pytest.raises(ValueError):
        numopt.run("dichotomous_search", p, delta_ratio=1.0)
    with pytest.raises(ValueError):
        numopt.run("bracket_minimum", p, step=0.0)
    with pytest.raises(ValueError):
        numopt.run("newton_1d", bare(lambda x: x * x))  # no x0
    with pytest.raises(ValueError):
        numopt.run("newton_1d", p, gtol=-1.0)
    with pytest.raises(TypeError):
        numopt.run("golden_section", p, tol=1e-3)


@pytest.mark.parametrize("mid", BRACKETED)
def test_non_finite_values_stop_honestly(mid):
    # x log x is nan for x < 0: the bracket (-1, 1) makes some trial point negative.
    res = numopt.run(mid, problems.get("x_log_x"), bracket=(-1.0, 1.0))
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)
    assert res.n_iter == res.trace[-1].k


def test_xtol_below_resolution_terminates():
    """xtol far below the spacing of doubles still terminates (floating-point floor)."""
    for mid in BRACKETED:
        if mid == "dichotomous_search":
            # δ = 1e-301 would make x₁ = x₂: rejected as invalid input.
            with pytest.raises(ValueError):
                numopt.run(mid, problems.get("rational_1d"), xtol=1e-300)
            continue
        res = numopt.run(mid, problems.get("rational_1d"), xtol=1e-300, max_iter=2000)
        assert_valid_result(res)
        assert res.converged, (mid, res.message)
        # The √ε floor of f-comparisons: √(2ε|f*|/f''(x*)) ≈ 2e-8 here.
        assert abs(res.x - x_star("rational_1d")) <= 1e-7
