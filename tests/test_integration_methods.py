"""Tests of the quadrature methods: oracles (SciPy/NumPy), exactness degrees, orders, failure paths."""

import json
import math
from itertools import pairwise

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import integrate

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.integration import methods as M

CALCULUS = [p.id for p in problems.list_problems("calculus")]
COMPOSITE = (
    "left_riemann",
    "right_riemann",
    "midpoint_rule",
    "trapezoid",
    "simpson",
    "simpson_38",
    "boole",
)
SPAN = {
    "left_riemann": 1,
    "right_riemann": 1,
    "midpoint_rule": 1,
    "trapezoid": 1,
    "simpson": 2,
    "simpson_38": 3,
    "boole": 4,
}
ORDER = {
    "left_riemann": 1,
    "right_riemann": 1,
    "midpoint_rule": 2,
    "trapezoid": 2,
    "simpson": 4,
    "simpson_38": 4,
    "boole": 6,
}
DEGREE = {
    "left_riemann": 0,
    "right_riemann": 0,
    "midpoint_rule": 1,
    "trapezoid": 1,
    "simpson": 3,
    "simpson_38": 3,
    "boole": 5,
}
ALL = (*COMPOSITE, "romberg", "gauss_legendre", "adaptive_simpson", "monte_carlo_integration")

COMMON_KEYS = {"estimate", "error", "err_est"}
INFO_KEYS = {
    **{
        m: COMMON_KEYS | {"ratio", "confirmed", "h", "n_panels", "panels", "nodes", "weights"}
        for m in COMPOSITE
    },
    "romberg": COMMON_KEYS | {"ratio", "confirmed", "h", "n_panels", "row", "nodes"},
    "gauss_legendre": COMMON_KEYS | {"ratio", "n_points", "nodes", "weights"},
    "adaptive_simpson": COMMON_KEYS
    | {"interval", "pending", "depth", "tol_local", "passed", "parent_passed", "forced", "nodes"},
    "monte_carlo_integration": COMMON_KEYS | {"samples", "n_samples"},
}


def _small_params(method: str) -> dict:
    if method in COMPOSITE:
        return {"n": SPAN[method] * 2, "levels": 4}
    return {
        "romberg": {"max_levels": 10},
        "gauss_legendre": {"n": 8},
        "adaptive_simpson": {"tol": 1e-6, "max_iter": 400},
        "monte_carlo_integration": {"n": 20, "levels": 4},
    }[method]


def _poly_problem(coefs: list[float], a: float, b: float) -> Problem:
    p = np.polynomial.Polynomial(coefs)
    P = p.integ()
    return Problem(
        id="poly",
        name="poly",
        latex="p(x)",
        f=p,
        dim=1,
        domain=(a, b),
        exact=float(P(b) - P(a)),
    )


def _ex(prob: Problem) -> float:
    assert prob.exact is not None
    return prob.exact


def _poly_scale(coefs: list[float], a: float, b: float) -> float:
    m = max(1.0, abs(a), abs(b))
    return (b - a) * sum(abs(c) * m**j for j, c in enumerate(coefs))


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("pid", CALCULUS)
def test_contract_on_every_problem(method, pid):
    res = numopt.run(method, problems.get(pid), **_small_params(method))
    assert_valid_result(res)
    assert res.n_iter == res.trace[-1].k
    for step in res.trace:
        assert set(step.info) == INFO_KEYS[method]
        assert step.fun == step.info["estimate"] == step.x
    if res.converged:
        assert math.isfinite(res.x)


@pytest.mark.parametrize("case", M.FIXTURE_CASES)
def test_fixture_cases_run(case):
    method, pid, params = case
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300


def test_every_method_is_covered_by_fixtures():
    # The nested rules (clenshaw_curtis, gauss_patterson) are tested in test_integration_promoted.py.
    nested = {"clenshaw_curtis", "gauss_patterson"}
    assert {c[0] for c in M.FIXTURE_CASES} == set(ALL) | nested


# --------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ["exp_0_1", "runge", "gaussian", "sqrt_0_1"])
def test_trapezoid_and_simpson_match_scipy_on_every_level(pid):
    p = problems.get(pid)
    a, b = p.domain
    for method, ref in (("trapezoid", integrate.trapezoid), ("simpson", integrate.simpson)):
        res = numopt.run(method, p, n=4, levels=5)
        for step in res.trace:
            x = np.array([pt[0] for pt in step.info["nodes"]])
            n_sub = step.info["n_panels"]
            assert_allclose(x, np.linspace(a, b, n_sub + 1), rtol=0, atol=4e-16 * (b - a))
            expected = ref(p.f(x), x=x)
            assert_allclose(step.fun, expected, rtol=1e-14)


@pytest.mark.parametrize("pid", ["exp_0_1", "arctan_deriv", "gaussian"])
def test_newton_cotes_rules_match_scipy_weights(pid):
    """Simpson 3/8 and Boole against scipy.integrate.newton_cotes panel weights."""
    p = problems.get(pid)
    a, b = p.domain
    for method, rn in (("simpson_38", 3), ("boole", 4), ("simpson", 2), ("trapezoid", 1)):
        an, _ = integrate.newton_cotes(rn, 1)
        n_sub = rn * 8
        h = (b - a) / n_sub
        x = np.linspace(a, b, n_sub + 1)
        fx = p.f(x)
        expected = sum(h * float(an @ fx[i : i + rn + 1]) for i in range(0, n_sub, rn))
        res = numopt.run(method, p, n=n_sub, levels=0)
        assert_allclose(res.trace[0].info["estimate"], expected, rtol=1e-14)


@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "runge"])
def test_riemann_and_midpoint_match_direct_sums(pid):
    p = problems.get(pid)
    a, b = p.domain
    n_sub = 32
    h = (b - a) / n_sub
    x = np.linspace(a, b, n_sub + 1)
    mid = a + (np.arange(n_sub) + 0.5) * h
    for method, expected in (
        ("left_riemann", h * math.fsum(p.f(x[:-1]))),
        ("right_riemann", h * math.fsum(p.f(x[1:]))),
        ("midpoint_rule", h * math.fsum(p.f(mid))),
    ):
        res = numopt.run(method, p, n=n_sub, levels=0)
        assert_allclose(res.trace[0].info["estimate"], expected, rtol=1e-14)


@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "runge", "gaussian", "arctan_deriv"])
def test_romberg_rows_match_scipy_romb(pid):
    p = problems.get(pid)
    a, b = p.domain
    res = numopt.run("romberg", p, tol=1e-300, max_levels=8)
    for step in res.trace[1:]:
        k = step.k
        x = np.linspace(a, b, 2**k + 1)
        assert_allclose(step.info["row"][-1], integrate.romb(p.f(x), dx=(b - a) / 2**k), rtol=1e-13)


def test_romberg_first_columns_are_simpson_and_boole():
    for pid in ("exp_0_1", "runge", "sqrt_0_1"):
        p = problems.get(pid)
        res = numopt.run("romberg", p, tol=1e-300, max_levels=4)
        assert_allclose(
            res.trace[1].info["row"][1], numopt.run("simpson", p, n=2, levels=0).x, rtol=1e-14
        )
        assert_allclose(
            res.trace[2].info["row"][2], numopt.run("boole", p, n=4, levels=0).x, rtol=1e-14
        )
        # column 1 of every row is composite Simpson on that grid
        for step in res.trace[1:]:
            simp = numopt.run("simpson", p, n=step.info["n_panels"], levels=0).x
            assert_allclose(step.info["row"][1], simp, rtol=1e-13)


def _legendre_rule_extended(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference rule in np.longdouble (quad precision on aarch64, 80-bit on x86):
    Newton on P_n from NumPy's nodes, then wᵢ = 2/((1 − tᵢ²)·P_n′(tᵢ)²) (A&S 25.4.29)."""
    x = np.polynomial.legendre.leggauss(n)[0].astype(np.longdouble)

    def pair(x):
        p0, p1 = np.ones_like(x), x.copy()
        for k in range(1, n):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        if n == 1:
            p0 = np.ones_like(x)
        return p1, n * (x * p1 - p0) / (x * x - 1)

    for _ in range(5):
        p, dp = pair(x)
        x = x - p / dp
    _, dp = pair(x)
    return x, 2 / ((1 - x * x) * dp * dp)


@pytest.mark.parametrize("n", range(1, 65))
def test_golub_welsch_matches_extended_precision_reference(n):
    t, w = M.gauss_legendre_rule(n)
    t_ref, w_ref = _legendre_rule_extended(n)
    # Measured: node error ≤ 3.3e-16, weight relative error ≤ 8e-14 for n ≤ 64 (the small
    # end weights come from small eigenvector components, error ~ ε/eigengap).
    assert np.max(np.abs(t - t_ref)) <= 1e-15
    assert np.max(np.abs((w - w_ref) / w_ref)) <= 2e-13
    assert_allclose(t, -t[::-1], rtol=0, atol=0)  # exact symmetry
    assert_allclose(w.sum(), 2.0, rtol=4e-15)
    # NOTE: numpy's leggauss weights are themselves only good to ~1e-12 relative at n = 48,
    # so it is a looser cross-check, not the reference.
    t_np, w_np = np.polynomial.legendre.leggauss(n)
    assert_allclose(t, t_np, rtol=0, atol=1e-15)
    assert_allclose(w, w_np, rtol=5e-12)


@pytest.mark.parametrize("pid", ["exp_0_1", "runge", "gaussian", "sqrt_0_1", "abs_kink"])
def test_gauss_matches_scipy_fixed_quad(pid):
    p = problems.get(pid)
    a, b = p.domain
    res = numopt.run("gauss_legendre", p, n=12)
    for step in res.trace:
        ref, _ = integrate.fixed_quad(p.f, a, b, n=step.info["n_points"])
        assert_allclose(step.info["estimate"], ref, rtol=1e-13, atol=1e-15)


_ACCURATE = (
    ("simpson", {"n": 4, "levels": 10, "tol": 1e-12}),
    # levels = 9: Boole converges faster than O(h⁶) before it reaches the rounding level
    # (runge: ρ ≈ 2·10⁴ at N = 256), so the ratios disagree, and the asymptotic regime is
    # confirmed only when two differences are at the rounding level (k = 8 or 9).
    ("boole", {"n": 4, "levels": 9, "tol": 1e-12}),
    ("romberg", {"tol": 1e-12}),
    ("gauss_legendre", {"n": 30, "tol": 1e-12}),
    ("adaptive_simpson", {"tol": 1e-11, "max_iter": 5000}),
)


@pytest.mark.parametrize(
    ("method", "params", "pid"),
    [
        (m, prm, pid)
        for m, prm in _ACCURATE
        for pid in ("exp_0_1", "sin_0_pi", "gaussian", "arctan_deriv", "runge")
        # Gauss on runge needs n ≈ 80: see test_gauss_reports_slow_convergence_on_runge
        if (m, pid) != ("gauss_legendre", "runge")
    ],
)
def test_accurate_methods_agree_with_quad(method, params, pid):
    p = problems.get(pid)
    a, b = p.domain
    ref, _ = integrate.quad(p.f, a, b, epsabs=1e-14, epsrel=1e-14)
    res = numopt.run(method, p, **params)
    assert res.converged, res.message
    assert abs(res.x - ref) <= 1e-10


# --------------------------------------------------------------------------------------
# Exactness degrees (Hypothesis)
# --------------------------------------------------------------------------------------

# Subnormal coefficients have no relative precision; they are outside the tested domain.
_coef = st.floats(-1, 1, allow_subnormal=False)
_interval = st.tuples(st.floats(-1.0, 0.9), st.floats(0.05, 1.0)).map(
    lambda t: (t[0], min(t[0] + t[1], 1.0))
)


@pytest.mark.parametrize("method", COMPOSITE)
@settings(max_examples=1000, deadline=None)
@given(data=st.data())
def test_composite_rules_exact_to_their_degree(method, data):
    d = DEGREE[method]
    coefs = data.draw(st.lists(_coef, min_size=d + 1, max_size=d + 1))
    a, b = data.draw(_interval)
    n_mult = data.draw(st.integers(1, 6))
    prob = _poly_problem(coefs, a, b)
    res = numopt.run(method, prob, n=SPAN[method] * n_mult, levels=1)
    for step in res.trace:
        assert abs(step.info["estimate"] - _ex(prob)) <= 1e-13 * _poly_scale(coefs, a, b)


@pytest.mark.parametrize("method", COMPOSITE)
def test_composite_rules_not_exact_one_degree_higher(method):
    d = DEGREE[method] + 1
    prob = _poly_problem([0.0] * d + [1.0], 0.0, 1.0)  # x^(d+1) on [0, 1]
    res = numopt.run(method, prob, n=SPAN[method], levels=0)
    assert abs(res.x - _ex(prob)) > 1e-4


@settings(max_examples=1000, deadline=None)
@given(n=st.integers(1, 20), data=st.data())
def test_gauss_exact_for_degree_2n_minus_1(n, data):
    coefs = data.draw(st.lists(_coef, min_size=2 * n, max_size=2 * n))
    a, b = data.draw(_interval)
    prob = _poly_problem(coefs, a, b)
    res = numopt.run("gauss_legendre", prob, n=n)
    assert abs(res.trace[-1].info["estimate"] - _ex(prob)) <= 1e-13 * _poly_scale(coefs, a, b)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_gauss_not_exact_for_degree_2n(n):
    prob = _poly_problem([0.0] * (2 * n) + [1.0], -1.0, 1.0)
    assert abs(numopt.run("gauss_legendre", prob, n=n).x - _ex(prob)) > 1e-6


@settings(max_examples=1000, deadline=None)
@given(k=st.integers(1, 4), data=st.data())
def test_romberg_row_k_exact_for_degree_2k_plus_1(k, data):
    coefs = data.draw(st.lists(_coef, min_size=2 * k + 2, max_size=2 * k + 2))
    a, b = data.draw(_interval)
    prob = _poly_problem(coefs, a, b)
    res = numopt.run("romberg", prob, tol=1e-300, max_levels=k)
    scale = _poly_scale(coefs, a, b)
    if res.n_iter < k:  # stopped early: |R(j,j) − R(j−1,j−1)| was exactly 0
        assert res.converged and abs(res.x - _ex(prob)) <= 1e-13 * scale
    else:
        assert abs(res.trace[k].info["row"][k] - _ex(prob)) <= 1e-13 * scale


@settings(max_examples=1000, deadline=None)
@given(data=st.data())
def test_adaptive_simpson_exact_for_cubics_in_one_step(data):
    """S₁ = S₂ for a cubic, so the pure Lyness routine (min_depth = 0) accepts [a, b] at once;
    with the default min_depth = 2 the four depth-2 intervals are accepted instead."""
    coefs = data.draw(st.lists(_coef, min_size=4, max_size=4))
    a, b = data.draw(_interval)
    prob = _poly_problem(coefs, a, b)
    res = numopt.run("adaptive_simpson", prob, tol=1e-10, min_depth=0)
    assert res.converged and res.n_iter == 1 and res.n_fev == 5
    assert abs(res.x - _ex(prob)) <= 1e-13 * _poly_scale(coefs, a, b)
    res = numopt.run("adaptive_simpson", prob, tol=1e-10)
    assert res.converged and res.n_iter == 4 and res.n_fev == 3 + 2 * 7
    assert [s.info["depth"] for s in res.trace[1:]] == [2, 2, 2, 2]
    assert abs(res.x - _ex(prob)) <= 1e-13 * _poly_scale(coefs, a, b)


# --------------------------------------------------------------------------------------
# Convergence orders and error estimates
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", COMPOSITE)
def test_observed_order_on_smooth_integrand(method):
    p = problems.get("exp_0_1")
    levels = {6: 3}.get(ORDER[method], 5)
    res = numopt.run(method, p, n=SPAN[method] * 2, levels=levels)
    errs = [s.info["error"] for s in res.trace]
    observed = math.log2(errs[-2] / errs[-1])
    assert abs(observed - ORDER[method]) < 0.1
    # Richardson estimate within 10% of the true error in the asymptotic regime
    assert_allclose(res.trace[-1].info["err_est"], errs[-1], rtol=0.1)


@pytest.mark.parametrize("method", ["trapezoid", "midpoint_rule", "simpson", "simpson_38", "boole"])
def test_order_drops_to_1_5_for_sqrt(method):
    """f = √x has f′ unbounded at 0: every rule's error behaves like h^{3/2}."""
    res = numopt.run(method, problems.get("sqrt_0_1"), n=SPAN[method] * 4, levels=8)
    errs = [s.info["error"] for s in res.trace]
    assert abs(math.log2(errs[-2] / errs[-1]) - 1.5) < 0.05


def test_romberg_converges_fast_on_smooth_and_slowly_on_singular():
    fast = numopt.run("romberg", problems.get("exp_0_1"), tol=1e-12)
    assert fast.converged and fast.n_iter <= 6 and abs(fast.extra["error"]) < 1e-14
    slow = numopt.run("romberg", problems.get("sqrt_0_1"), tol=1e-10, max_levels=12)
    assert not slow.converged and "max_levels" in slow.message


def test_gauss_reports_slow_convergence_on_runge():
    """Poles at ±i/5: the Bernstein ellipse parameter is ρ = 0.2 + √1.04 ≈ 1.22, so the
    n-point error decays like ρ^(−2n) — about 6e-6 at n = 30 (Trefethen, ATAP Thm. 19.3)."""
    rho = 0.2 + math.sqrt(1.04)
    res = numopt.run("gauss_legendre", problems.get("runge"), n=30, tol=1e-12)
    assert not res.converged
    assert 1e-2 * rho**-60 < res.extra["error"] < 1e2 * rho**-60


def _documented_estimate(diffs: list[float], noises: list[float], rate: float | None) -> float:
    """The estimate exactly as the module docstring states it (an independent transcription)."""
    d, base = diffs[-1], diffs[-1] / (rate - 1) if rate else diffs[-1]
    if len(diffs) == 1 or (d <= noises[-1] and diffs[-2] <= noises[-2]):
        return base
    rho = diffs[-2] / d if d > 0 else math.inf
    if rho <= 1:
        est = d
    elif rate:
        est = max(base, d / (rho - 1), diffs[-2] / (rate * (rate - 1)))
    else:
        est = max(d, d / (rho - 1))
    return max(est, *diffs[-3:]) if rate is None else est


@pytest.mark.parametrize("method", [*COMPOSITE, "gauss_legendre"])
@pytest.mark.parametrize("pid", CALCULUS)
def test_err_est_follows_the_documented_rule(method, pid):
    params = {"n": 12} if method == "gauss_legendre" else {"n": SPAN[method] * 2, "levels": 6}
    res = numopt.run(method, problems.get(pid), **params)
    rate = None if method == "gauss_legendre" else 2.0 ** ORDER[method]
    diffs: list[float] = []
    noises: list[float] = []
    for s0, s1 in pairwise(res.trace):
        mass = [
            math.fsum(
                abs(w * v) for w, (_, v) in zip(s.info["weights"], s.info["nodes"], strict=True)
            )
            for s in (s0, s1)
        ]
        diffs.append(abs(s1.info["estimate"] - s0.info["estimate"]))
        noises.append(4 * np.finfo(float).eps * (mass[0] + mass[1]))
        assert s1.info["err_est"] == _documented_estimate(diffs, noises, rate)
        expected_ratio = diffs[-2] / diffs[-1] if len(diffs) > 1 and diffs[-1] > 0 else None
        assert s1.info["ratio"] == expected_ratio


def test_estimate_corrects_the_order_drop_on_sqrt():
    """Boole on √x converges like h^{3/2}: the textbook d/63 is ~30× too small, the
    observed-ratio tail d/(ρ − 1) with ρ → 2^{3/2} is not."""
    res = numopt.run("boole", problems.get("sqrt_0_1"), n=4, levels=8)
    last = res.trace[-1].info
    d = abs(res.trace[-1].info["estimate"] - res.trace[-2].info["estimate"])
    assert last["error"] > 20 * d / 63
    assert 0.5 * last["error"] <= last["err_est"] <= 2 * last["error"]
    assert abs(math.log2(last["ratio"]) - 1.5) < 0.05
    g = numopt.run("gauss_legendre", problems.get("sqrt_0_1"), n=64, tol=1e-15).trace[-1].info
    assert g["error"] > 10 * abs(
        g["estimate"] - numopt.run("gauss_legendre", problems.get("sqrt_0_1"), n=63).x
    )  # one difference is far too small
    assert g["err_est"] >= 0.5 * g["error"]


def test_coincident_levels_at_a_kink_do_not_converge():
    """Midpoint rule on |x − 0.3| with N = 4, 8: the kink panel's error is d² = 0.05² both
    times, so I_1 = I_2 exactly. One difference (levels = 1) cannot converge, and with
    levels = 3 the term d_{k−1}/(R(R − 1)) keeps the estimate above the true error."""
    p = problems.get("abs_kink")
    res = numopt.run("midpoint_rule", p, n=2, levels=2, tol=1e-12)
    assert res.trace[2].fun == res.trace[1].fun and res.extra["error"] == pytest.approx(0.0025)
    assert res.trace[2].info["err_est"] >= res.extra["error"]
    assert not res.converged
    one = numopt.run("midpoint_rule", p, n=4, levels=1, tol=1.0)
    assert not one.converged and "levels = 1" in one.message


def test_gauss_window_covers_irregular_convergence_at_a_kink():
    """Gauss on |x − 0.3|: |G_63 − G_62| ≈ 7e-7 while the error is 2.5e-5; the window
    max(d_k, d_{k−1}, d_{k−2}) keeps the estimate above the error."""
    res = numopt.run("gauss_legendre", problems.get("abs_kink"), n=63, tol=1e-5)
    last = res.trace[-1].info
    assert (
        abs(res.trace[-1].info["estimate"] - res.trace[-2].info["estimate"]) < 0.1 * last["error"]
    )
    assert last["err_est"] >= last["error"]
    assert not res.converged


def _runge_c(c: float) -> Problem:
    return Problem(
        id=f"runge_{c:g}",
        name="1/(1 + c x²)",
        latex="f",
        f=lambda x: 1 / (1 + c * x * x),
        dim=1,
        domain=(-1.0, 1.0),
        exact=2 * math.atan(math.sqrt(c)) / math.sqrt(c),
    )


@pytest.mark.parametrize(
    ("method", "c", "params"),
    [
        ("simpson", 100.0, {"n": 4, "levels": 2, "tol": 1e-3}),
        ("boole", 100.0, {"n": 4, "levels": 2, "tol": 1e-3}),
        ("boole", 426.8, {"n": 4, "levels": 3, "tol": 3.2e-4}),
        ("simpson", 112.6, {"n": 2, "levels": 3, "tol": 1e-3}),
        ("romberg", 48.3, {"max_levels": 15, "tol": 1.3e-3}),
    ],
)
def test_pre_asymptotic_runge_variants_are_not_falsely_converged(method, c, params):
    """Audit regressions: before the grid resolves the peak of 1/(1 + c·x²), the differences
    can contract at a rate near 2^p while the error grows (Simpson, c = 100: errors 0.094,
    0.0095, 0.0129). These runs converged with errors of 13–23·tol. Now they converge only
    after the asymptotic regime is confirmed, and then the error is within tol."""
    res = numopt.run(method, _runge_c(c), **params)
    assert_valid_result(res)
    tol = params["tol"]
    if res.converged:
        assert res.extra["error"] <= tol * max(1.0, abs(res.x))
    else:
        assert "not confirmed" in res.message or "> tol" in res.message


def test_confirmation_needs_four_signed_differences_of_one_sign():
    """The rule of the module docstring, on hand-made sequences."""
    noise = [1e-30] * 6
    geometric = [-(16.0**-j) for j in range(6)]  # δ_j of a sequence with rate 16
    assert M._confirmed(geometric[:4], noise[:4])
    assert not M._confirmed(geometric[:3], noise[:3])  # only two ratios
    flipped = [*geometric[:3], -geometric[3]]  # the last difference changes sign
    assert not M._confirmed(flipped, noise[:4])
    erratic = [-1.0, -0.1, -0.05, -0.0001]  # ratios 10, 2, 500 disagree
    assert not M._confirmed(erratic, noise[:4])
    slow = [-1.0, -1 / 2.83, -1 / 2.83**2, -1 / 2.83**3]  # order drop to h^{3/2}: confirmed
    assert M._confirmed(slow, noise[:4])
    at_rounding = [-1.0, -0.5, 1e-17, -2e-17]
    assert M._confirmed(at_rounding, [1e-16] * 4)
    assert not M._confirmed([-1.0, 1e-17], [1e-16, 1e-16])  # only one at the rounding level


@pytest.mark.parametrize("method", [*COMPOSITE, "romberg"])
def test_converged_runs_carry_a_confirmed_last_step(method):
    for pid in ("exp_0_1", "runge", "sqrt_0_1", "gaussian"):
        params = {"tol": 1e-6} | (
            {"max_levels": 16} if method == "romberg" else {"n": SPAN[method] * 2, "levels": 9}
        )
        res = numopt.run(method, problems.get(pid), **params)
        if res.converged:
            assert res.trace[-1].info["confirmed"]


_FAMILY_METHODS = (*COMPOSITE, "romberg", "gauss_legendre", "adaptive_simpson")


def _family_problem(family: str, c: float, xc: float) -> Problem:
    if family == "runge":
        return _runge_c(c)
    s = math.sqrt(c)
    exact = math.sqrt(math.pi) / (2 * s) * (math.erf(s * (1 - xc)) + math.erf(s * xc))
    return Problem(
        id="peak",
        name="exp(−c(x − x_c)²)",
        latex="f",
        f=lambda x: math.exp(-c * (x - xc) ** 2),
        dim=1,
        domain=(0.0, 1.0),
        exact=exact,
    )


@settings(max_examples=1000, deadline=None)
@given(
    method=st.sampled_from(_FAMILY_METHODS),
    family=st.sampled_from(["runge", "peak"]),
    log_c=st.floats(1.0, 3.0),
    xc=st.floats(0.0, 1.0),
    size=st.integers(1, 8),
    levels=st.integers(2, 9),
    log_tol=st.floats(-12, -1),
)
def test_converged_is_honest_on_parametric_families(
    method, family, log_c, xc, size, levels, log_tol
):
    """The honesty invariant beyond the library (audit: the safeguards were tuned on the 9
    library problems and failed on 1/(1 + c·x²) with c slightly above 25): on
    1/(1 + c·x²) on [−1, 1] and exp(−c(x − x_c)²) on [0, 1], c ∈ [10, 1000], converged ⇒
    error ≤ 10·tol·max(1, |I|), and for the composite rules error ≤ 10·err_est. Known limit
    (module docstring, "Preconditions", item 1): a peak that no node sees is missed; with
    c ≤ 1000 (width 1/√c ≥ 0.03) the nodes see it before the regime can be confirmed."""
    p = _family_problem(family, 10.0**log_c, xc)
    tol = 10.0**log_tol
    if method in COMPOSITE:
        params = {"n": SPAN[method] * size, "levels": levels}
    elif method == "romberg":
        params = {"max_levels": 2 + levels}
    elif method == "gauss_legendre":
        params = {"n": 8 * size}
    else:
        params = {}  # the defaults, as in the audit
    res = numopt.run(method, p, tol=tol, **params)
    if not res.converged:
        return
    slack = 16 * np.finfo(float).eps * max(1.0, abs(_ex(p)))
    assert res.extra["error"] <= 10 * tol * max(1.0, abs(res.x)) + slack
    if method in COMPOSITE:
        assert res.extra["error"] <= 10 * res.extra["err_est"] + slack


_SWEEP = (*COMPOSITE, "romberg", "gauss_legendre", "adaptive_simpson")


@settings(max_examples=1000, deadline=None)
@given(
    method=st.sampled_from(_SWEEP),
    pid=st.sampled_from(CALCULUS),
    size=st.integers(1, 16),
    levels=st.integers(1, 9),
    log_tol=st.floats(-14, -2),
)
def test_converged_means_the_error_estimate_covers_the_error(method, pid, size, levels, log_tol):
    """The honesty invariant: converged ⇒ |I − exact| ≤ 10·err_est (+ rounding of the
    exact value); for adaptive Simpson, whose err_est sums local estimates, ≤ 10·max(err_est,
    tol). The estimates are not rigorous bounds; a 20 000-run sweep found no violation.
    The midpoint rule at a kink is the documented blind spot (see the next test)."""
    assume((method, pid) != ("midpoint_rule", "abs_kink"))
    p = problems.get(pid)
    tol = 10.0**log_tol
    if method in COMPOSITE:
        params = {"n": SPAN[method] * size, "levels": levels}
    elif method == "romberg":
        params = {"max_levels": size}
    elif method == "gauss_legendre":
        params = {"n": 4 * size}
    else:
        params = {"max_depth": 50, "max_iter": 5000}
    res = numopt.run(method, p, tol=tol, **params)
    if not res.converged:
        return
    est = res.extra["err_est"]
    slack = 16 * np.finfo(float).eps * max(1.0, abs(_ex(p)))
    if method == "adaptive_simpson":
        est = max(est, tol)
    assert res.extra["error"] <= 10 * est + slack


def test_midpoint_blind_spot_at_a_kink_is_as_documented():
    """Characterization of the documented limitation: on |x − 0.3| with N = 13, 26, 52 the
    nearest grid node to the kink is 4/13 at every level, so the kink-panel error
    (4/13 − 0.3)² is the same and the three estimates are identical. The estimate is 0 and
    the rule reports convergence although the error is 5.9e-5."""
    res = numopt.run("midpoint_rule", problems.get("abs_kink"), n=13, levels=2, tol=1e-2)
    assert res.trace[0].fun == res.trace[1].fun == res.trace[2].fun
    assert res.extra["error"] == pytest.approx((4 / 13 - 0.3) ** 2, rel=1e-9)
    assert res.extra["err_est"] == 0.0 and res.converged
    # one more level puts the node 31/104 nearer to the kink and the estimate sees it
    more = numopt.run("midpoint_rule", problems.get("abs_kink"), n=13, levels=3, tol=1e-12)
    assert more.trace[3].fun != more.trace[2].fun
    assert more.extra["err_est"] >= more.extra["error"]


def _simpson(f, a: float, b: float) -> float:
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


def _lyness_diff(f, a: float, b: float) -> float:
    """S₂ − S₁ on [a, b], computed independently of the module."""
    m = (a + b) / 2
    return _simpson(f, a, m) + _simpson(f, m, b) - _simpson(f, a, b)


def test_lyness_acceptance_needs_the_parent_to_pass_on_runge():
    """The failure of Lyness' original routine (min_depth = 0): on runge with tol = 1e-3,
    [−1, 0] and [0, 1] pass their own test |S₂ − S₁| ≤ 15·tol/2, but the root [−1, 1] fails,
    and accepting the halves gives an error of 0.026. An interval now needs a parent that
    passed, so the halves are bisected and the result meets tol."""
    p = problems.get("runge")
    f, tol = p.f, 1e-3
    assert abs(_lyness_diff(f, -1.0, 1.0)) > 15 * tol
    halves = [(-1.0, 0.0), (0.0, 1.0)]
    assert all(abs(_lyness_diff(f, a, b)) <= 15 * tol / 2 for a, b in halves)
    one_level = sum(
        _simpson(f, a, (a + b) / 2) + _simpson(f, (a + b) / 2, b) + _lyness_diff(f, a, b) / 15
        for a, b in halves
    )
    assert abs(one_level - _ex(p)) > 0.02  # what the original routine returns
    res = numopt.run("adaptive_simpson", p, tol=tol, min_depth=0)
    assert res.converged and res.extra["error"] <= tol
    assert res.extra["intervals"][:2] != [[-1.0, 0.0], [0.0, 1.0]]
    assert all(s.info["passed"] and s.info["parent_passed"] for s in res.trace[1:])
    assert not any(s.info["forced"] for s in res.trace[1:])


def test_adaptive_audit_case_runge_100_is_not_falsely_converged():
    """Audit regression: on 1/(1 + 100x²) with tol = 1e-3 and the defaults, [0, 0.5] (depth 2)
    passes its own test although its error is 6.6e-3, while its parent [0, 1] fails; the
    one-level rule converged with error 0.0132 = 13·tol."""
    c, tol = 100.0, 1e-3

    def f(x):
        return 1 / (1 + c * x * x)

    assert abs(_lyness_diff(f, 0.0, 0.5)) <= 15 * tol / 4
    assert abs(_lyness_diff(f, 0.0, 1.0)) > 15 * tol / 2
    prob = Problem(
        id="r100",
        name="r100",
        latex="f",
        f=f,
        dim=1,
        domain=(-1.0, 1.0),
        exact=2 * math.atan(math.sqrt(c)) / math.sqrt(c),
    )
    res = numopt.run("adaptive_simpson", prob, tol=tol)
    assert res.converged and res.extra["error"] <= tol
    assert [0.0, 0.5] not in res.extra["intervals"]
    assert_valid_result(res)


def test_aliasing_blind_spot_is_as_documented():
    """Characterization of documented precondition 1: cos(32πx) is 1 at every node of
    N ≤ 32 dyadic subintervals of [0, 1], so the trapezoid rule and Romberg see equal
    estimates (d_k = 0, at the rounding level) and report I = 1 although I = 0. Gauss uses
    non-nested nodes and does not converge."""

    def f(x):
        return math.cos(32 * math.pi * x)

    trap = numopt.run("trapezoid", f, bracket=(0.0, 1.0), n=4, levels=2, tol=1e-8)
    assert trap.converged and trap.x == 1.0 and trap.extra["err_est"] == 0.0
    rom = numopt.run("romberg", f, bracket=(0.0, 1.0), max_levels=16, tol=1e-8)
    assert rom.converged and rom.x == 1.0
    gauss = numopt.run("gauss_legendre", f, bracket=(0.0, 1.0), n=30, tol=1e-8)
    assert not gauss.converged


def test_gauss_converges_spectrally_on_analytic_integrand():
    res = numopt.run("gauss_legendre", problems.get("exp_0_1"), n=8, tol=1e-13)
    assert res.converged
    errs = [s.info["error"] for s in res.trace]
    # n points: truncation error (n!)⁴/((2n+1)((2n)!)³)·e, i.e. 5e-16 for n = 6 and 1e-18 for
    # n = 7, below the rounding error of the sum (a few ulp of e − 1 ≈ 1.7). The errors of
    # n = 1..6 decrease strictly; from n = 6 on they are rounding noise, whose order depends on
    # the CPU.
    assert errs[5] < 1e-14 and errs[6] < 1e-14
    assert all(e1 < e0 for e0, e1 in pairwise(errs[:6]))


# --------------------------------------------------------------------------------------
# Evaluation counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", COMPOSITE)
def test_composite_counts(method):
    n, L = SPAN[method] * 2, 4
    res = numopt.run(method, problems.get("exp_0_1"), n=n, levels=L)
    expected = {
        "left_riemann": n * 2**L,
        "right_riemann": n * 2**L,
        "midpoint_rule": n * (2 ** (L + 1) - 1),
    }.get(method, n * 2**L + 1)
    assert res.n_fev == expected


def test_romberg_gauss_adaptive_mc_counts():
    p = problems.get("runge")
    rom = numopt.run("romberg", p, tol=1e-300, max_levels=7)
    assert rom.n_fev == 2**7 + 1
    gl = numopt.run("gauss_legendre", p, n=9)
    # Σ m for m = 1..9, minus the shared centre node of the odd rules m = 3, 5, 7, 9
    assert gl.n_fev == 45 - 4
    ad = numopt.run("adaptive_simpson", p, tol=1e-7)
    n_acc = ad.n_iter  # accepted = leaves of a binary tree; processed = 2·leaves − 1
    assert ad.n_fev == 3 + 2 * (2 * n_acc - 1)
    mc = numopt.run("monte_carlo_integration", p, n=30, levels=3)
    assert mc.n_fev == 30 * 8


# --------------------------------------------------------------------------------------
# Adaptive Simpson invariants
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(pid=st.sampled_from(CALCULUS), tol=st.floats(1e-10, 1e-3))
def test_adaptive_intervals_tile_the_domain_in_order(pid, tol):
    p = problems.get(pid)
    a, b = p.domain
    res = numopt.run("adaptive_simpson", p, tol=tol, max_iter=3000)
    ivs = res.extra["intervals"]
    if res.converged:
        assert ivs[0][0] == a and ivs[-1][1] == b
        for (x0, x1), (y0, y1) in pairwise(ivs):
            assert x1 == y0 and x0 < x1 and y0 < y1
        assert not res.trace[-1].info["pending"]
    # Step k reports the k-th accepted interval, so the viewer can accumulate them
    assert [step.info["interval"] for step in res.trace[1:]] == ivs


@pytest.mark.parametrize(
    "pid", ["exp_0_1", "sin_0_pi", "gaussian", "runge", "arctan_deriv", "abs_kink"]
)
def test_adaptive_meets_tolerance(pid):
    p = problems.get(pid)
    res = numopt.run("adaptive_simpson", p, tol=1e-8)
    assert res.converged, res.message
    assert res.extra["error"] <= 1e-8


def test_adaptive_concentrates_work_near_singularity():
    res = numopt.run("adaptive_simpson", problems.get("sqrt_0_1"), tol=1e-6, max_depth=50)
    widths = [b - a for a, b in res.extra["intervals"]]
    assert widths[0] < 1e-5 * widths[-1]  # tiny intervals at x = 0, wide ones at x = 1


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------


def test_monte_carlo_is_deterministic_per_seed():
    p = problems.get("gaussian")
    r1 = numopt.run("monte_carlo_integration", p, seed=7, n=50, levels=3)
    r2 = numopt.run("monte_carlo_integration", p, seed=7, n=50, levels=3)
    r3 = numopt.run("monte_carlo_integration", p, seed=8, n=50, levels=3)
    assert r1.x == r2.x and r1.trace[2].info["samples"] == r2.trace[2].info["samples"]
    assert r1.x != r3.x


def test_monte_carlo_samples_are_the_rng_stream():
    from numopt.core.rng import Rng

    p = problems.get("exp_0_1")
    res = numopt.run("monte_carlo_integration", p, seed=3, n=5, levels=1)
    rng = Rng(3)
    xs = [rng.uniform(0.0, 1.0) for _ in range(10)]
    got = [s[0] for step in res.trace for s in step.info["samples"]]
    assert got == xs
    assert_allclose(res.x, np.mean(np.exp(xs)), rtol=1e-14)
    assert_allclose(res.extra["err_est"], np.std(np.exp(xs), ddof=1) / math.sqrt(10), rtol=1e-12)


@settings(max_examples=1000, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), pid=st.sampled_from(["sin_0_pi", "exp_0_1", "runge"]))
def test_monte_carlo_error_within_six_standard_errors(seed, pid):
    res = numopt.run("monte_carlo_integration", problems.get(pid), seed=seed, n=100, levels=4)
    assert res.extra["error"] <= 6.0 * res.extra["err_est"]


def test_monte_carlo_standard_error_shrinks_like_inverse_sqrt_n():
    res = numopt.run("monte_carlo_integration", problems.get("sin_0_pi"), seed=1, n=1000, levels=6)
    se = [s.info["err_est"] for s in res.trace]
    for s0, s1 in pairwise(se):
        assert 1.2 < s0 / s1 < 1.65  # √2 ≈ 1.414


def test_monte_carlo_constant_integrand_is_exact():
    res = numopt.run("monte_carlo_integration", lambda x: 3.0, bracket=(1.0, 3.0), n=10, levels=1)
    assert res.x == 6.0 and res.extra["err_est"] == 0.0 and res.converged


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


def _reciprocal(x):
    return 1.0 / x


@pytest.mark.parametrize(
    "method", ["trapezoid", "left_riemann", "simpson", "boole", "romberg", "adaptive_simpson"]
)
def test_nonfinite_integrand_reports_failure(method):
    res = numopt.run(method, _reciprocal, bracket=(0.0, 1.0))
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)


def _opposite_infinities(x):
    return 1.0 / np.float64(x * (x - 1.0))  # f(0) = −inf, f(1) = +inf


def _huge_constant(x):
    return 1e308  # I = 4e308 on [0, 4] overflows


def _huge_opposite(x):
    return 1e308 if x < 0.5 else -1e308  # I = 0, but partial sums overflow


_OVERFLOW = {
    "opposite_infinities": (_opposite_infinities, (0.0, 1.0)),
    "huge_constant": (_huge_constant, (0.0, 4.0)),
    "huge_opposite": (_huge_opposite, (0.0, 1.0)),
}


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("case", sorted(_OVERFLOW))
def test_overflow_and_opposite_infinities_never_raise(method, case):
    """Audit regression: math.fsum raised ValueError on inf + (−inf) and OverflowError on an
    intermediate overflow, so the methods crashed; adaptive Simpson returned NaN with a
    max_iter message. Now every sum falls back to the IEEE sum and the methods stop with
    converged=False."""
    f, bracket = _OVERFLOW[case]
    res = numopt.run(method, f, bracket=bracket, **_small_params(method))
    assert_valid_result(res)
    if case == "huge_constant":
        assert not res.converged and res.message == "the estimate overflowed"
    elif case == "opposite_infinities":
        assert not res.converged
        if method not in ("midpoint_rule", "gauss_legendre", "monte_carlo_integration"):
            assert "not finite at x = " in res.message  # these rules evaluate an endpoint
    elif res.converged:  # huge_opposite: I = 0 exactly
        assert res.x == 0.0


def test_adaptive_overflow_is_detected_not_iterated():
    """Audit regression: with f = 1e308, S₁ = inf and S₂ − S₁ = NaN, so no interval passed and
    the run ended at max_iter with x = NaN."""
    res = numopt.run("adaptive_simpson", _huge_constant, bracket=(0.0, 4.0))
    assert not res.converged and res.message == "the estimate overflowed"
    assert res.n_iter == 0 and res.n_fev == 3
    mixed = numopt.run("adaptive_simpson", _huge_opposite, bracket=(0.0, 1.0))
    assert not mixed.converged and mixed.message == "the estimate overflowed"


def test_fsum_falls_back_to_the_ieee_sum():
    assert M._fsum([1.0, 1e-16, -1.0]) == 1e-16  # correctly rounded
    assert math.isnan(M._fsum([math.inf, -math.inf]))
    assert M._fsum([1e308, 1e308]) == math.inf
    assert M._fsum([1e308, 1e308, -1e308]) == math.inf  # plain IEEE left-to-right sum


def test_midpoint_and_gauss_handle_endpoint_singularity_without_nan():
    for method in ("midpoint_rule", "gauss_legendre"):
        res = numopt.run(method, lambda x: 1.0 / math.sqrt(x), bracket=(0.0, 1.0))
        assert math.isfinite(res.x) and not res.converged  # slow convergence, honest flag


@pytest.mark.parametrize("method", COMPOSITE)
def test_levels_zero_is_not_converged(method):
    res = numopt.run(method, problems.get("poly3"), n=SPAN[method] * 2, levels=0)
    assert not res.converged and "levels = 0" in res.message
    assert_valid_result(res, max_iter=0)


def test_single_estimate_methods_are_not_converged():
    one = numopt.run("gauss_legendre", problems.get("exp_0_1"), n=1)
    assert not one.converged and one.n_iter == one.trace[-1].k == 0
    two = numopt.run("gauss_legendre", problems.get("exp_0_1"), n=2, tol=1.0)
    assert not two.converged and "n = 2" in two.message
    # audit regression: the n ≤ 2 branch returned n_iter = 0 for the two steps k = 0, 1
    assert two.n_iter == two.trace[-1].k == 1
    assert_valid_result(two, max_iter=1)
    mc = numopt.run("monte_carlo_integration", problems.get("exp_0_1"), n=1, levels=0)
    assert not mc.converged and "one sample" in mc.message


def test_slow_methods_report_tolerance_not_met():
    res = numopt.run("trapezoid", problems.get("exp_0_1"), n=2, levels=3, tol=1e-10)
    assert not res.converged and "> tol" in res.message
    res = numopt.run("gauss_legendre", problems.get("sqrt_0_1"), n=6, tol=1e-10)
    assert not res.converged and "increase n" in res.message


def test_adaptive_limits_report_failure():
    deep = numopt.run("adaptive_simpson", problems.get("sqrt_0_1"), tol=1e-12, max_depth=5)
    assert not deep.converged and "Lyness" in deep.message
    assert any(s.info["forced"] for s in deep.trace[1:])
    short = numopt.run("adaptive_simpson", problems.get("runge"), tol=1e-10, max_iter=3)
    assert not short.converged and "max_iter" in short.message
    assert_valid_result(short, max_iter=3)


def test_romberg_max_levels():
    res = numopt.run("romberg", problems.get("runge"), tol=1e-14, max_levels=3)
    assert not res.converged and "max_levels=3" in res.message
    assert_valid_result(res, max_iter=3)
    # the test needs rows k − 1 ≥ 2 and k, so max_levels ≤ 2 cannot converge; the UI range
    # starts at 3
    for max_levels in (1, 2):
        few = numopt.run("romberg", problems.get("poly3"), tol=1e-1, max_levels=max_levels)
        assert not few.converged and "max_levels ≥ 3" in few.message


def test_romberg_cost_is_capped_like_the_composite_rules():
    """Audit regression: max_levels = 25 meant 2^25 + 1 evaluations and a trace with 2^24
    nodes in its last row. Row k uses 2^k subintervals, so max_levels ≤ 18 = log2 MAX_POINTS."""
    assert 2**M._ROMBERG_LEVELS_MAX == M.MAX_POINTS
    with pytest.raises(ValueError, match="max_levels"):
        numopt.run("romberg", problems.get("sqrt_0_1"), max_levels=M._ROMBERG_LEVELS_MAX + 1)


def _spec(method: str) -> dict:
    return {ps.name: ps for ps in numopt.get_method(method).params}


@pytest.mark.parametrize("method", ALL)
def test_paramspec_ranges_never_raise(method):
    """Audit regression: every UI combination inside the ParamSpec ranges must be valid. The
    limits are joint (n·2^levels), so the corners decide; checked without running them."""
    spec = _spec(method)
    if method in COMPOSITE:
        n_max, lv_max = spec["n"].max, spec["levels"].max
        assert n_max is not None and lv_max is not None
        assert n_max * 2**lv_max <= M.MAX_POINTS
        assert n_max % SPAN[method] == 0 and spec["n"].min == SPAN[method]
    elif method == "monte_carlo_integration":
        n_max, lv_max = spec["n"].max, spec["levels"].max
        assert n_max is not None and lv_max is not None
        assert n_max * 2**lv_max <= M.MAX_SAMPLES
        assert spec["n"].min <= spec["n"].default <= n_max
    elif method == "romberg":
        lv = spec["max_levels"]
        assert lv.min == 3 and lv.max is not None and 2**lv.max <= M.MAX_POINTS
    elif method == "adaptive_simpson":
        # audit regression: min_depth = 2 (default) with max_depth = 1 raised ValueError; the
        # joint corners must run (min_depth is clamped to max_depth)
        md, xd = spec["min_depth"], spec["max_depth"]
        assert md.max is not None and xd.min is not None and md.min is not None
        for min_depth, max_depth in ((md.max, xd.min), (md.default, xd.min), (md.max, md.max)):
            res = numopt.run(
                "adaptive_simpson",
                problems.get("exp_0_1"),
                min_depth=min_depth,
                max_depth=max_depth,
                max_iter=50,
            )
            assert_valid_result(res, max_iter=50)
            assert res.extra["min_depth"] == min(min_depth, max_depth)
    for ps in spec.values():
        if ps.min is not None and ps.max is not None:
            assert ps.min <= ps.default <= ps.max, ps.name


@pytest.mark.parametrize(("method", "n"), [("simpson_38", 63), ("monte_carlo_integration", 128)])
def test_paramspec_corner_runs(method, n):
    """The largest corner that is cheap enough to run in a unit test: n at its maximum with
    a few levels (the joint limit is checked statically above)."""
    res = numopt.run(method, problems.get("exp_0_1"), n=n, levels=3)
    assert_valid_result(res, max_iter=3)


def test_min_depth_above_max_depth_is_clamped():
    res = numopt.run("adaptive_simpson", problems.get("poly3"), min_depth=6, max_depth=5)
    assert res.extra["min_depth"] == 5
    assert all(s.info["depth"] == 5 for s in res.trace[1:])
    assert res.converged and res.n_iter == 32  # a cubic passes everywhere


@pytest.mark.parametrize(
    ("method", "pid", "params"),
    [
        ("trapezoid", "exp_0_1", {"n": 64, "levels": 12}),
        ("left_riemann", "exp_0_1", {"n": 64, "levels": 12}),
        ("boole", "exp_0_1", {"n": 64, "levels": 12}),
        ("monte_carlo_integration", "exp_0_1", {"n": 128, "levels": 13}),
        ("romberg", "sqrt_0_1", {"max_levels": 18, "tol": 1e-15}),
    ],
)
def test_trace_size_is_bounded_at_the_paramspec_corners(method, pid, params):
    """Audit regression: at the UI maxima every level stored all its nodes, weights and panels
    (trapezoid: 53 MB of JSON; Monte Carlo every sample: 44 MB; Romberg 11 MB). A Step now
    stores at most MAX_DISPLAY points; the estimate is unchanged."""
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(json.dumps(res.to_dict(), allow_nan=False)) < 2_000_000
    key = "samples" if method == "monte_carlo_integration" else "nodes"
    for step in res.trace:
        shown = step.info[key]
        assert shown is None or len(shown) <= M.MAX_DISPLAY
    if method in COMPOSITE:
        last = res.trace[-1].info
        assert last["nodes"] is None and last["weights"] is None and last["panels"] is None
        assert res.trace[0].info["nodes"] is not None  # small levels keep their geometry


def test_adaptive_trace_grows_linearly():
    """Audit regression: every Step copied all accepted intervals ("intervals"), so the trace
    was O(N²): 5 000 steps gave 539 MB of JSON. Now Step k holds its own interval and the stack,
    which has at most one entry per depth."""
    max_depth = 30
    res = numopt.run(
        "adaptive_simpson",
        problems.get("oscillatory"),
        tol=1e-14,
        max_depth=max_depth,
        max_iter=2000,
    )
    assert len(res.trace) == 2001 and not res.converged
    pairs = sum(1 + len(step.info["pending"]) for step in res.trace)
    assert all(len(step.info["pending"]) <= max_depth + 1 for step in res.trace)
    assert pairs <= (max_depth + 2) * len(res.trace)
    assert len(res.extra["intervals"]) == 2000


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("simpson", {"n": 3}),
        ("simpson_38", {"n": 4}),
        ("boole", {"n": 6}),
        ("trapezoid", {"n": 0}),
        ("trapezoid", {"tol": 0.0}),
        ("trapezoid", {"levels": -1}),
        ("trapezoid", {"n": 512, "levels": 12}),
        ("gauss_legendre", {"n": 0}),
        ("romberg", {"max_levels": 0}),
        ("romberg", {"max_levels": 19}),
        ("monte_carlo_integration", {"n": 128, "levels": 14}),
        ("adaptive_simpson", {"max_depth": 0}),
        ("adaptive_simpson", {"min_depth": -1}),
        ("monte_carlo_integration", {"n": 0}),
    ],
)
def test_invalid_parameters_raise(method, params):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("exp_0_1"), **params)


@pytest.mark.parametrize("bracket", [(1.0, 0.0), (0.0, 0.0), (0.0, math.inf)])
def test_invalid_interval_raises(bracket):
    with pytest.raises(ValueError):
        numopt.run("trapezoid", problems.get("exp_0_1"), bracket=bracket)


def test_bracket_overrides_domain_and_custom_callable_has_no_error():
    res = numopt.run("simpson", lambda x: x**2, bracket=(0.0, 3.0), n=2, levels=1)
    assert_allclose(res.x, 9.0, rtol=1e-15)
    assert res.trace[0].info["error"] is None and res.extra["exact"] is None
