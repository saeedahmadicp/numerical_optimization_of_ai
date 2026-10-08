"""Tests of the numerical-differentiation methods: textbook formulas, orders, V-curve, oracles."""

import math
from itertools import pairwise

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import differentiate

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.differentiation import methods as M

CALCULUS = [p.id for p in problems.list_problems("calculus")]
FD = ("forward_difference", "backward_difference", "central_difference", "five_point_stencil")
ALL = (*FD, "second_derivative_central", "richardson_extrapolation", "complex_step")
ORDER = {
    "forward_difference": 1,
    "backward_difference": 1,
    "central_difference": 2,
    "five_point_stencil": 4,
    "second_derivative_central": 2,
    "complex_step": 2,
}
BASE_KEYS = {
    "h",
    "estimate",
    "error",
    "err_est",
    "roundoff",
    "roundoff_obs",
    "confirms",
    "stencil",
    "weights",
    "x0",
    "collapsed",
}
EXTRA_KEYS = {"richardson_extrapolation": {"row"}, "complex_step": {"imag"}}
EPS = np.finfo(float).eps


def _textbook(method: str, f, x: float, h: float) -> float:
    """The formulas exactly as printed (Burden & Faires Eqs. 4.1, 4.5, 4.6, 4.9)."""
    if method == "forward_difference":
        return (f(x + h) - f(x)) / h
    if method == "backward_difference":
        return (f(x) - f(x - h)) / h
    if method == "central_difference":
        return (f(x + h) - f(x - h)) / (2 * h)
    if method == "five_point_stencil":
        return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)
    if method == "second_derivative_central":
        return (f(x - h) - 2 * f(x) + f(x + h)) / h**2
    raise KeyError(method)


def _horner_bound(coefs: list[float], x: float) -> float:
    """Higham (2002), Eq. 5.3: |fl(p(x)) − p(x)| ≤ γ_{2d}·Σ_j |a_j||x|^j for Horner's rule."""
    d = len(coefs) - 1
    gamma = 2 * d * EPS / (1 - 2 * d * EPS)
    return gamma * math.fsum(abs(a) * abs(x) ** j for j, a in enumerate(coefs))


def _poly_problem(coefs: list[float], x0: float) -> Problem:
    p = np.polynomial.Polynomial(coefs)
    return Problem(
        id="poly",
        name="poly",
        latex="p",
        f=p,
        grad=p.deriv(),
        hess=p.deriv(2),
        dim=1,
        domain=(-1, 1),
        x0=x0,
    )


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL)
@pytest.mark.parametrize("pid", CALCULUS)
def test_contract_on_every_problem(method, pid):
    res = numopt.run(method, problems.get(pid), levels=30)
    assert_valid_result(res, max_iter=30)
    assert res.n_iter == res.trace[-1].k == 30
    for k, step in enumerate(res.trace):
        assert set(step.info) == BASE_KEYS | EXTRA_KEYS.get(method, set())
        assert step.step_size == step.info["h"] == 0.1 / 2**k
        if step.info["estimate"] is not None and math.isfinite(step.info["estimate"]):
            assert step.fun == step.x == step.info["estimate"]
    for key in (
        "exact",
        "error",
        "k_best",
        "h_best",
        "bound_best",
        "k_trunc_end",
        "k_confirm_last",
        "h_opt",
        "h_opt_rule",
        "h_min_error",
        "order",
    ):
        assert key in res.extra


@pytest.mark.parametrize("case", M.FIXTURE_CASES)
def test_fixture_cases_run(case):
    method, pid, params = case
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300


def test_every_method_is_covered_by_fixtures():
    assert {c[0] for c in M.FIXTURE_CASES} == set(ALL)


@pytest.mark.parametrize("method", [*FD, "second_derivative_central"])
def test_stencil_and_weights_reproduce_estimate(method):
    res = numopt.run(method, problems.get("gaussian"), levels=6)
    for step in res.trace:
        total = math.fsum(
            w * pt[1] for w, pt in zip(step.info["weights"], step.info["stencil"], strict=True)
        )
        # products wᵢ·fᵢ are rounded before the sum: within the round-off model
        assert abs(total - step.info["estimate"]) <= 2 * step.info["roundoff"]


# --------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", [*FD, "second_derivative_central"])
@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "runge", "gaussian", "oscillatory"])
def test_estimates_equal_the_textbook_formula(method, pid):
    p = problems.get(pid)
    res = numopt.run(method, p, levels=12)
    for step in res.trace:
        h = step.info["h"]
        expected = _textbook(method, p.f, p.x0, h)
        # the two differ only in the rounding of the numerator: ≤ a few ε·Σ|cᵢ fᵢ|/h^q
        assert abs(step.info["estimate"] - expected) <= 4 * step.info["roundoff"] + 1e-300


@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "runge", "gaussian", "arctan_deriv"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_richardson_equals_polynomial_extrapolation_in_h_squared(pid, k):
    """D(k, k) is the value at h = 0 of the degree-k polynomial in h² through (h_i², D(i, 0))."""
    p = problems.get(pid)
    res = numopt.run("richardson_extrapolation", p, h0=0.2, levels=k)
    hs = np.array([s.info["h"] for s in res.trace])
    base = np.array([s.info["row"][0] for s in res.trace])
    coeffs = np.polynomial.polynomial.polyfit(hs**2, base, k)
    assert_allclose(res.trace[k].fun, coeffs[0], rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize(
    "pid", ["exp_0_1", "sin_0_pi", "runge", "gaussian", "oscillatory", "arctan_deriv", "sqrt_0_1"]
)
@pytest.mark.parametrize("method", [*FD, "richardson_extrapolation", "complex_step"])
def test_best_estimate_agrees_with_scipy_differentiate(pid, method):
    p = problems.get(pid)
    exact = float(p.grad(p.x0))
    ref = differentiate.derivative(
        p.f, p.x0, initial_step=0.01, tolerances={"rtol": 1e-13, "atol": 1e-13}
    )
    res = numopt.run(method, p)
    assert res.converged, res.message
    # the method's own bound (times a safety factor) covers the true error
    assert abs(res.x - exact) <= 10 * res.extra["bound_best"] + 2 * EPS * abs(exact)
    # SciPy agrees to its true error, which its own estimate ref.error can understate: the
    # estimate is the change between two of its iterations, 0 on some CPUs while the error is
    # 1e-11 (sin_0_pi on x86-64). Its true error is at most 2.2e-11 on these problems.
    assert abs(res.x - ref.df) <= 10 * res.extra["bound_best"] + 1e-10 * max(1.0, abs(exact))


@pytest.mark.parametrize("pid", CALCULUS)
def test_complex_step_reaches_machine_precision(pid):
    p = problems.get(pid)
    res = numopt.run("complex_step", p, h0=1e-20, levels=2)
    assert_allclose(res.x, p.grad(p.x0), rtol=2 * EPS, atol=1e-300)
    assert res.converged


# --------------------------------------------------------------------------------------
# Orders, V-curve and h_opt
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ORDER)
@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "gaussian"])
def test_truncation_order_in_the_asymptotic_regime(method, pid):
    p = problems.get(pid)
    res = numopt.run(method, p, h0=0.05, levels=4)
    errs = [s.info["error"] for s in res.trace]
    observed = math.log2(errs[-2] / errs[-1])
    assert abs(observed - ORDER[method]) < 0.1, errs
    # the Richardson estimate tracks the true error within 15 % there
    assert_allclose(res.trace[-1].info["err_est"], errs[-1], rtol=0.15)


@pytest.mark.parametrize(
    ("method", "h_slope"),
    [
        ("forward_difference", 1),
        ("central_difference", 1),
        ("five_point_stencil", 1),
        ("second_derivative_central", 2),
    ],
)
def test_v_shaped_error_curve(method, h_slope):
    res = numopt.run(method, problems.get("exp_0_1"), h0=0.1, levels=45)
    errs = np.array([s.info["error"] for s in res.trace])
    k_min = int(np.argmin(errs))
    assert 0 < k_min < 45
    assert errs[-1] > 100 * errs[k_min]  # round-off branch is visible
    assert errs[0] > 100 * errs[k_min]  # truncation branch is visible
    # round-off model grows like h^(-q): ×2^q per halving
    rho = [s.info["roundoff"] for s in res.trace]
    ratios = [r1 / r0 for r0, r1 in pairwise(rho[10:30])]
    assert_allclose(ratios, 2.0**h_slope, rtol=0.05)
    # the textbook h_opt lies within a factor 10 of the observed optimum
    assert 0.1 < res.extra["h_opt"] / res.extra["h_min_error"] < 10


def test_complex_step_error_curve_has_no_v():
    res = numopt.run("complex_step", problems.get("exp_0_1"), h0=0.1, levels=60)
    errs = [s.info["error"] for s in res.trace]
    assert max(errs[30:]) <= 2 * EPS * math.e


def test_h_opt_formulas():
    x0 = 3.0
    expected = {
        "forward_difference": 2 * math.sqrt(EPS) * x0,
        "central_difference": (3 * EPS) ** (1 / 3) * x0,
        "five_point_stencil": (11.25 * EPS) ** (1 / 5) * x0,
        "second_derivative_central": (48 * EPS) ** 0.25 * x0,
        "complex_step": math.sqrt(6 * EPS) * x0,
    }
    for method, h in expected.items():
        res = numopt.run(method, problems.get("exp_0_1"), x0=x0, levels=3)
        assert_allclose(res.extra["h_opt"], h, rtol=1e-12)
    assert (
        numopt.run("richardson_extrapolation", problems.get("exp_0_1"), levels=3).extra["h_opt"]
        is None
    )


def test_best_level_is_not_fooled_by_coincident_roundoff_estimates():
    """At x0 = 1, f(x) = x³ − 2x + 1 has f(1) = 0 and h0 = 0.1 makes successive deep
    estimates coincide exactly (0.1·2^(52−k) has the same fractional part ratio). The
    abscissa-rounding term of the model must keep the selection in the truncation regime."""
    res = numopt.run("forward_difference", problems.get("poly3"), h0=0.1, levels=45)
    assert res.converged
    assert res.extra["error"] <= 10 * res.extra["bound_best"]
    assert res.extra["error"] < 1e-7


@pytest.mark.parametrize("method", ["central_difference", "five_point_stencil"])
def test_aliased_levels_are_rejected(method):
    """sin at x0 = π/3 with h0 = 2π: D_0 = D_1 = 0 (the stencil spans whole periods), so the
    local bound b_1 ≈ 0 at a wrong value. Later levels disagree with D_1 by 0.5, and the
    consistency term must move the selection to a level near the true value cos(π/3)."""
    p = problems.get("sin_0_pi")
    res = numopt.run(method, p, h0=2 * math.pi, levels=30)
    assert abs(res.trace[1].info["estimate"]) < 1e-15 and res.trace[1].info["err_est"] < 1e-15
    assert res.extra["k_best"] > 1
    assert res.converged
    assert res.extra["error"] <= res.extra["bound_best"]
    assert res.extra["error"] < 1e-9


def test_unconfirmed_bound_is_not_converged():
    """Hypothesis counterexample: e^{−x²} at x0 = 1 with h0 = 10, levels = 1 samples only the
    tails (e^{−121}, e^{−36}), so D_0 ≈ D_1 ≈ 0 with err_est ≈ 4e-9 while f′(1) = −0.736.
    No later level can test the bound, so the result must not claim convergence."""
    res = numopt.run("central_difference", problems.get("gaussian"), x0=1.0, h0=10.0, levels=1)
    assert res.extra["bound_best"] < 1e-8 and res.extra["error"] > 0.7
    assert not res.converged and "no later level confirms" in res.message
    # with more levels the sweep reaches the scale of f and converges to the right value
    res = numopt.run("central_difference", problems.get("gaussian"), x0=1.0, h0=10.0, levels=40)
    assert res.converged and res.extra["error"] < 1e-9


def _shifted_exp(x0: float) -> Problem:
    return Problem(
        id="exp_shift",
        name="exp_shift",
        latex="e^{x-x_0}",
        f=lambda x: np.exp(x - x0),
        grad=lambda x: np.exp(x - x0),
        hess=lambda x: np.exp(x - x0),
        dim=1,
        domain=(x0 - 1, x0 + 1),
        x0=x0,
    )


@pytest.mark.parametrize("method", [*FD, "second_derivative_central", "richardson_extrapolation"])
def test_collapsed_levels_are_flagged_and_never_selected(method):
    """x0 = 1e8 has ulp(x0) ≈ 1.49e-8: once h < ulp/2 the abscissae x0 ± h round to x0."""
    x0 = 1e8
    res = numopt.run(method, _shifted_exp(x0), h0=1e-3, levels=30)
    assert_valid_result(res)
    ulp = math.ulp(x0)
    for step in res.trace:
        info = step.info
        offsets = {0, *M._STENCILS.get(method, M._STENCILS["central_difference"]).offsets}
        grid = [x0 + o * info["h"] for o in sorted(offsets)]
        assert info["collapsed"] == any(b <= a for a, b in pairwise(grid))
        if info["collapsed"]:
            assert info["err_est"] is None and info["roundoff"] is None
    flags = [s.info["collapsed"] for s in res.trace]
    assert not flags[0] and flags[-1]  # 1e-3 is far above ulp, 1e-3/2^30 ≈ 9e-13 below it
    assert all(flags[k] for k in range(len(flags)) if res.trace[k].info["h"] < ulp / 4)
    assert not res.trace[res.extra["k_best"]].info["collapsed"]
    # the level after a collapsed one has no truncation estimate
    for a, b in pairwise(res.trace):
        if a.info["collapsed"]:
            assert b.info["err_est"] is None


#: f that cancel inside their own formula: the absolute error of f is about ε, not ε|f|.
_CANCELLING = {
    "expm1_naive": (lambda x: np.exp(x) - 1.0, np.exp, np.exp),
    "one_minus_cos": (lambda x: 1.0 - np.cos(x), np.sin, np.cos),
}


def _cancelling(name: str, x0: float) -> Problem:
    f, g, h = _CANCELLING[name]
    return Problem(
        id=name, name=name, latex=name, f=f, grad=g, hess=h, dim=1, domain=(-1, 1), x0=x0
    )


def test_cancellation_regression_from_the_audit():
    """Audit finding: central difference on f = exp(x) − 1 at x0 = 1e-9 with levels = 60 (the UI
    maximum). From k = 52 on, f(x0 ± h) are bitwise equal, so D_k = 0 with err_est_k = 0 and a
    model roundoff of 1e-6..1e-4 at f′ = 1; those levels confirmed each other and refuted every
    correct level, and the old rule returned x = 0 with converged = True."""
    res = numopt.run("central_difference", _cancelling("expm1_naive", 1e-9), levels=60)
    last = res.extra["k_confirm_last"]
    deep = res.trace[last + 1 :]
    # the trap is still in the trace ...
    assert any(s.info["estimate"] == 0.0 and s.info["err_est"] == 0.0 for s in deep)
    # ... but those levels cannot confirm, and the result is the true derivative
    assert not any(s.info["confirms"] for s in deep)
    assert res.converged and abs(res.x - math.exp(1e-9)) < 1e-10
    best = res.trace[res.extra["k_best"]].info
    # the model sees ε|f| ≈ 1e-25 per value; the observed round-off is the real ε/h level
    assert best["roundoff_obs"] > 1000 * best["roundoff"]
    assert res.extra["error"] <= 10 * res.extra["bound_best"]


@pytest.mark.parametrize("name", list(_CANCELLING))
@pytest.mark.parametrize("x0", [1e-9, 1e-7, 1e-5, 1e-3])
@pytest.mark.parametrize("method", [*FD, "second_derivative_central", "richardson_extrapolation"])
def test_internal_cancellation_at_the_audit_settings(name, x0, method):
    """Default h0 = 0.1, levels = 60: every real-stencil method finds f′ (f″) within the
    promise, and the round-off tail of the sweep is never a confirming level."""
    res = numopt.run(method, _cancelling(name, x0), levels=60)
    assert_valid_result(res, max_iter=60)
    assert res.converged, res.message
    assert res.extra["error"] <= 10 * 1e-6 * max(1.0, abs(res.x))
    last = res.extra["k_confirm_last"]
    assert res.extra["k_best"] < last < 60
    assert [s.info["confirms"] for s in res.trace[last + 1 :]] == [False] * (60 - last)


# NOTE: restricted to the methods for which the invariant holds on these functions. A
# 20 000-run sweep of this domain found no violation for them (worst error = 3.5·tol·max(1,
# |D|)), but 0.3 % violations for the forward, backward and five-point formulas: there the
# round-off regime can start with one jump to bitwise-equal f values right after a run, which
# the trace cannot tell apart from a kink (module docstring, precondition 2).
@settings(max_examples=1000, deadline=None, derandomize=True)
@given(
    method=st.sampled_from(
        ["central_difference", "second_derivative_central", "richardson_extrapolation"]
    ),
    name=st.sampled_from(list(_CANCELLING)),
    log_x0=st.floats(-9, -1),
    log_h0=st.floats(-3, 0),
    levels=st.integers(1, 60),
    tol=st.sampled_from([1e-3, 1e-6, 1e-9, 1e-12]),
)
def test_converged_is_honest_when_f_cancels_internally(method, name, log_x0, log_h0, levels, tol):
    """converged ⇒ |D − exact| ≤ 10·tol·max(1, |D|) for f whose round-off the a-priori model
    underestimates by up to 1/|f| (here up to 1e18), with h0 above the round-off regime."""
    res = numopt.run(
        method, _cancelling(name, 10.0**log_x0), h0=10.0**log_h0, levels=levels, tol=tol
    )
    if res.converged:
        assert res.extra["error"] <= 10 * tol * max(1.0, abs(res.x))


@pytest.mark.parametrize(
    "method",
    [
        "backward_difference",
        "central_difference",
        "five_point_stencil",
        "second_derivative_central",
        "richardson_extrapolation",
    ],
)
def test_infinite_stencil_value_gives_a_nan_estimate(method):
    """f = 1/x in NumPy at x0 = 0.1, h0 = 0.1: x0 − h = 0 gives f = inf (no exception). The
    documented estimate is NaN (JSON null), not ±inf; the stencil keeps the true value."""
    res = numopt.run(method, lambda x: np.reciprocal(np.float64(x)), x0=0.1, h0=0.1, levels=3)
    assert_valid_result(res, max_iter=3)
    first = res.trace[0]
    assert math.isnan(first.info["estimate"]) and math.isnan(first.x)
    assert any(math.isinf(pt[1]) for pt in first.info["stencil"])
    as_json = first.to_dict()
    assert as_json["info"]["estimate"] is None and as_json["x"] is None
    # the five-point stencil also reaches 0 at k = 1 (x0 − 2h_1 = 0); later levels are finite
    assert math.isfinite(res.trace[-1].info["estimate"])


_SWEEP_METHODS = (*FD, "second_derivative_central", "richardson_extrapolation", "complex_step")


# NOTE: derandomized so that CI is reproducible. The invariant holds with the factor 10 for
# functions evaluated to a relative accuracy ε; a function that cancels inside its own formula
# can exceed it slightly. A 150 000-run random sweep found one such case in 79 904 converged
# runs: poly3 at x0 = 0.827 (|f| = 0.088 but |x³| + |2x| + 1 = 3.2), error 10.3·B.
@settings(max_examples=1000, deadline=None, derandomize=True)
@given(
    method=st.sampled_from(_SWEEP_METHODS),
    pid=st.sampled_from(CALCULUS),
    log_h0=st.floats(-12, 1),
    levels=st.integers(1, 60),
    u=st.floats(0.05, 0.95),
    tol=st.sampled_from([1e-3, 1e-6, 1e-9, 1e-12]),
)
def test_converged_means_the_bound_covers_the_true_error(method, pid, log_h0, levels, u, tol):
    """The honesty invariant of the selection rule. Random x0 inside the domain and h0 over 13
    decades (the ParamSpec range) exercise aliasing (h ≈ period), kinks, the √x singularity
    and round-off-dominated sweeps. Precondition (module docstring): the sweep must reach
    steps below the length scale of f (≥ 0.1 for every library problem away from its kink or
    singularity), here h0/2^levels ≤ 1e-4. converged ⇒ |D − exact| ≤ 10·B_k* up to the
    precision of the oracle itself; with B_k* ≤ tol·max(1, |D|) this gives the promise
    error ≤ 10·tol·max(1, |D|). (B is an estimate, not a rigorous bound.)"""
    assume(10.0**log_h0 / 2.0**levels <= 1e-4)
    p = problems.get(pid)
    a, b = p.domain
    x0 = a + u * (b - a)
    res = numopt.run(method, p, x0=x0, h0=10.0**log_h0, levels=levels, tol=tol)
    exact = res.extra["exact"]
    if not res.converged or exact is None:
        return
    err = res.extra["error"]
    assert res.extra["bound_best"] <= tol * max(1.0, abs(res.x))
    # The exact derivative g(x0) evaluated in floating point is itself uncertain by about
    # ε(|g| + |x0|·|g′|) (the rounding of x0 inside g); g′ = f″ is known for first derivatives.
    dg = abs(float(p.hess(x0))) if method != "second_derivative_central" else 0.0
    oracle = 8 * EPS * (max(1.0, abs(exact)) + abs(x0) * dg)
    assert err <= 10 * res.extra["bound_best"] + oracle


# --------------------------------------------------------------------------------------
# Exactness on polynomials (Hypothesis)
# --------------------------------------------------------------------------------------

# Subnormal coefficients have no relative precision; they are outside the tested domain.
_coef2 = st.floats(-2, 2, allow_subnormal=False)

EXACT_DEGREE = {
    "forward_difference": 1,
    "backward_difference": 1,
    "central_difference": 2,
    "five_point_stencil": 4,
    "second_derivative_central": 3,
}


@pytest.mark.parametrize("method", EXACT_DEGREE)
@settings(max_examples=1000, deadline=None)
@given(data=st.data())
def test_formulas_exact_up_to_their_degree(method, data):
    d = EXACT_DEGREE[method]
    coefs = data.draw(st.lists(_coef2, min_size=d + 1, max_size=d + 1))
    x0 = data.draw(st.floats(-1, 1))
    h0 = data.draw(st.floats(1e-3, 0.5))
    prob = _poly_problem(coefs, x0)
    res = numopt.run(method, prob, h0=h0, levels=3)
    exact = res.extra["exact"]
    for step in res.trace:
        # The round-off model assumes f values with a relative error ε. A polynomial that
        # cancels inside its own Horner evaluation (e.g. a double root at x0) has the larger
        # absolute error γ_{2d}·Σ|a_j||x|^j (Higham, Accuracy and Stability, Eq. 5.3).
        eval_noise = math.fsum(
            abs(w) * _horner_bound(coefs, pt[0])
            for w, pt in zip(step.info["weights"], step.info["stencil"], strict=True)
        )
        bound = 8 * step.info["roundoff"] + eval_noise + 1e-14 * max(1.0, abs(exact))
        assert abs(step.info["estimate"] - exact) <= bound


@settings(max_examples=1000, deadline=None)
@given(k=st.integers(0, 3), data=st.data())
def test_richardson_row_k_exact_for_degree_2k_plus_2(k, data):
    coefs = data.draw(st.lists(_coef2, min_size=2 * k + 3, max_size=2 * k + 3))
    x0 = data.draw(st.floats(-1, 1))
    prob = _poly_problem(coefs, x0)
    res = numopt.run("richardson_extrapolation", prob, h0=0.25, levels=k)
    scale = sum(abs(c) for c in coefs) * 2.0 ** len(coefs)
    assert abs(res.trace[k].fun - res.extra["exact"]) <= 1e-12 * scale


@settings(max_examples=1000, deadline=None)
@given(
    coefs=st.lists(_coef2, min_size=1, max_size=12),
    x0=st.floats(-1, 1),
    h0=st.floats(1e-12, 1e-6),
)
def test_complex_step_exact_on_any_polynomial(coefs, x0, h0):
    prob = _poly_problem(coefs, x0)
    res = numopt.run("complex_step", prob, h0=h0, levels=1)
    exact = res.extra["exact"]
    h = res.trace[-1].info["h"]
    scale = sum(abs(c) * j for j, c in enumerate(coefs))  # ≥ |p′(x)| on [−1, 1]
    third = sum(abs(c) * j * (j - 1) * (j - 2) for j, c in enumerate(coefs))  # ≥ |p‴(x)|
    # truncation h²|p‴|/6 (+ higher terms, ≤ the same bound for h ≤ 1e-6) plus rounding
    assert abs(res.trace[-1].fun - exact) <= 2 * h**2 * third / 6 + 64 * EPS * max(1.0, scale)


@settings(max_examples=1000, deadline=None)
@given(pid=st.sampled_from(CALCULUS), u=st.floats(0.05, 0.95), h0=st.floats(1e-4, 0.1))
def test_forward_backward_average_is_central(pid, u, h0):
    p = problems.get(pid)
    a, b = p.domain
    x0 = a + u * (b - a)
    fw = numopt.run("forward_difference", p, x0=x0, h0=h0, levels=0).x
    bw = numopt.run("backward_difference", p, x0=x0, h0=h0, levels=0).x
    ce = numopt.run("central_difference", p, x0=x0, h0=h0, levels=0)
    if not all(map(math.isfinite, (fw, bw, ce.x))):
        return
    assert abs(0.5 * (fw + bw) - ce.x) <= 4 * ce.trace[0].info["roundoff"] + 1e-300


# --------------------------------------------------------------------------------------
# Evaluation counts
# --------------------------------------------------------------------------------------


def test_counts():
    p = problems.get("exp_0_1")
    L = 10
    expected = {
        "forward_difference": L + 2,  # f(x0) once, f(x0 + h_k) per level
        "backward_difference": L + 2,
        "central_difference": 2 * (L + 1),
        "five_point_stencil": 4 + 2 * L,  # x0 ± 2h_k = x0 ± h_(k−1) are reused
        "second_derivative_central": 2 * (L + 1) + 1,
        "richardson_extrapolation": 2 * (L + 1),
        "complex_step": L + 1,
    }
    for method, n in expected.items():
        res = numopt.run(method, p, levels=L)
        assert res.n_fev == n, method
        assert res.n_gev == 0 and res.n_hev == 0  # the exact derivative is not counted


# --------------------------------------------------------------------------------------
# Edge cases and failure paths
# --------------------------------------------------------------------------------------


def test_abs_kink_straddling_stencils_are_wrong_then_exact():
    res = numopt.run("central_difference", problems.get("abs_kink"), h0=0.2, levels=6)
    errs = [s.info["error"] for s in res.trace]
    # h = 0.2, 0.1: the stencil straddles the kink at 0.3 (x0 = 0.35)
    assert errs[0] > 0.2 and errs[1] > 0.2
    # h ≤ 0.05: both points on the linear branch, exact slope 1
    assert all(e <= 1e-14 for e in errs[2:])
    assert res.converged and abs(res.x - 1.0) <= 1e-14
    # one-sided differences: forward never straddles from x0 = 0.35
    fw = numopt.run("forward_difference", problems.get("abs_kink"), h0=0.2, levels=3)
    assert all(e <= 1e-14 for e in (s.info["error"] for s in fw.trace))


def test_nan_levels_are_skipped():
    """√x at x0 = 0.05 with h0 = 0.4: x0 − h < 0 gives NaN for k = 0..2; later levels are fine."""
    p = problems.get("sqrt_0_1")
    res = numopt.run("central_difference", p, x0=0.05, h0=0.4, levels=25)
    assert_valid_result(res)
    assert all(not math.isfinite(s.x) for s in res.trace[:3])
    assert all(math.isfinite(s.x) for s in res.trace[3:])
    assert res.converged and "skipped" in res.message
    assert abs(res.extra["error"]) < 1e-7
    rich = numopt.run("richardson_extrapolation", p, x0=0.05, h0=0.4, levels=15)
    assert rich.converged and abs(rich.extra["error"]) < 1e-9  # the table restarted


def test_all_nonfinite_reports_failure():
    res = numopt.run("central_difference", lambda x: math.nan, x0=0.0, levels=5)
    assert not res.converged and "non-finite" in res.message
    assert_valid_result(res)


def test_levels_zero_is_not_converged():
    for method in ALL:
        res = numopt.run(method, problems.get("exp_0_1"), levels=0)
        assert not res.converged and "levels = 0" in res.message
        assert_valid_result(res, max_iter=0)


def test_unreachable_tolerance_is_reported():
    res = numopt.run("forward_difference", problems.get("exp_0_1"), tol=1e-14)
    assert not res.converged and "> tol" in res.message
    assert math.isfinite(res.x)  # still returns the best estimate


def test_custom_callable_without_exact_derivative():
    res = numopt.run("central_difference", lambda x: x**3, x0=2.0, levels=20)
    assert res.converged and abs(res.x - 12.0) < 1e-8
    assert res.trace[0].info["error"] is None and res.extra["h_min_error"] is None


def test_complex_step_rejects_non_analytic_functions():
    with pytest.raises(ValueError, match="complex"):
        numopt.run("complex_step", math.sin, x0=1.0)
    with pytest.raises(ValueError, match="real value"):
        numopt.run("complex_step", abs, x0=1.0)


@pytest.mark.parametrize(
    "params", [{"h0": 0.0}, {"h0": -1.0}, {"h0": math.inf}, {"levels": -1}, {"tol": 0.0}]
)
@pytest.mark.parametrize("method", ALL)
def test_invalid_parameters_raise(method, params):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("exp_0_1"), **params)


def test_missing_x0_raises():
    with pytest.raises(ValueError):
        numopt.run("central_difference", lambda x: x)
