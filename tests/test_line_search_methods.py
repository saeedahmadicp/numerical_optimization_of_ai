"""Tests for numopt.line_search.methods (the search() helper and the five demo methods).

Acceptance is always re-checked here with independent one-line formulas of N&W eqs. (3.4),
(3.6), (3.7) and (3.11), never with the module's own predicates.
"""

from __future__ import annotations

import inspect
import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose
from scipy import optimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.line_search import methods as ls
from numopt.line_search.methods import KINDS, search

EPS = float(np.finfo(float).eps)
TINY = float(np.finfo(float).smallest_subnormal)  # 2⁻¹⁰⁷⁴
SEARCH_KINDS = ("backtracking", "strong_wolfe", "weak_wolfe", "goldstein")
C_GOLD = 0.25

# --------------------------------------------------------------------------------------
# Inline problems with exact derivatives
# --------------------------------------------------------------------------------------


def rosen(x: np.ndarray) -> float:
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def rosen_grad(x: np.ndarray) -> np.ndarray:
    return np.array(
        [-400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]), 200.0 * (x[1] - x[0] ** 2)]
    )


def rosen_hess(x: np.ndarray) -> np.ndarray:
    return np.array(
        [[1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]], [-400.0 * x[0], 200.0]]
    )


ROSEN = Problem(
    "rosen_inline", "Rosenbrock", "", rosen, 2, (), grad=rosen_grad, hess=rosen_hess, x0=[-1.2, 1.0]
)

A_ILL = np.array([[1.0, 0.0], [0.0, 50.0]])
B_ILL = np.array([1.0, -2.0])


def quad_ill(x: np.ndarray) -> float:
    return float(0.5 * x @ A_ILL @ x - B_ILL @ x)


QUAD = Problem(
    "quad_inline",
    "ill-conditioned quadratic",
    "",
    quad_ill,
    2,
    (),
    grad=lambda x: A_ILL @ x - B_ILL,
    hess=lambda x: A_ILL,
    x0=[3.0, 1.0],
)


def quadratic(A: np.ndarray, b: np.ndarray, c: float = 0.0) -> tuple[Callable, Callable, Callable]:
    """f = ½xᵀAx - bᵀx + c with its gradient and Hessian."""
    return (
        lambda x: float(0.5 * x @ A @ x - b @ x) + c,
        lambda x: A @ x - b,
        lambda x: A,
    )


def gamma(k: int) -> float:
    """γ_k = kε / (1 - kε), Higham (2002) Lemma 3.1."""
    return k * EPS / (1.0 - k * EPS)


def quadratic_f_err(A: np.ndarray, b: np.ndarray, c: float, *zs: np.ndarray) -> float:
    """A bound on the error of a computed f = float(½zᵀAz - bᵀz) + c at every z in ``zs``.

    Higham (2002) Thm 3.5 and Lemma 3.3: A @ z, the dot products and the three remaining
    operations give at most γ_{2n+3}·T(z), T(z) = ½|z|ᵀ|A||z| + |b|ᵀ|z| + |c|. One more ε·T
    covers the rounding of the trial point fl(x + αp), which moves f by at most
    ε|y|ᵀ|Ay - b| ≤ 2ε·T(y) (shared by the two values: 2·γ_{2n+4} ≥ 2·γ_{2n+3} + 2ε). The
    bound is built from the magnitudes of the terms, so the size of f itself does not set it.

    Gradual underflow adds an absolute error (Higham §2.1, fl(x·y) = xy(1 + δ) + η,
    |η| ≤ 2⁻¹⁰⁷⁵ = TINY/2): at most TINY for each of the n² + 2n + 1 products, and
    ½‖|A||y| + |b|‖₁·TINY for the subnormal rounding of the trial point y. Without it the
    relative bound underflows to 0 when f is subnormal (shrunk example: A = 750,
    b = x = 1.8754795309089233e-156, f ≈ -2.3e-315 differs from Brent's value by one subnormal ulp).
    """
    n = b.size
    return max(
        gamma(2 * n + 4)
        * (float(0.5 * np.abs(z) @ np.abs(A) @ np.abs(z) + np.abs(b) @ np.abs(z)) + abs(c))
        + TINY * (n * n + 2 * n + 1 + 0.5 * float(np.sum(np.abs(A) @ np.abs(z) + np.abs(b))))
        for z in zs
    )


# --------------------------------------------------------------------------------------
# Independent condition checks (N&W §3.1)
# --------------------------------------------------------------------------------------


def holds(
    kind: str,
    f: Callable,
    grad: Callable,
    x: np.ndarray,
    p: np.ndarray,
    alpha: float,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> bool:
    f0, d0 = f(x), float(grad(x) @ p)
    fa = f(x + alpha * p)
    if kind == "goldstein":
        return f0 + (1 - c1) * alpha * d0 <= fa <= f0 + c1 * alpha * d0
    armijo = fa <= f0 + c1 * alpha * d0
    if kind == "backtracking":
        return armijo
    da = float(grad(x + alpha * p) @ p)
    if kind == "weak_wolfe":
        return armijo and da >= c2 * d0
    return armijo and abs(da) <= c2 * abs(d0)  # strong Wolfe


def c1_for(kind: str) -> float:
    return C_GOLD if kind == "goldstein" else 1e-4


# --------------------------------------------------------------------------------------
# Registry contract
# --------------------------------------------------------------------------------------


def test_registered_ids_and_params():
    specs = {s.id: s for s in numopt.list_methods("line_search")}
    assert set(specs) == set(KINDS)
    for spec in specs.values():
        sig = inspect.signature(spec.fn)
        kw = {n for n, prm in sig.parameters.items() if prm.kind is prm.KEYWORD_ONLY}
        assert kw - {"x0"} == {p.name for p in spec.params}
        for p in spec.params:
            assert sig.parameters[p.name].default == p.default
            if p.kind == "float":
                assert p.min is not None and p.max is not None and p.min <= p.default <= p.max


def test_fixture_cases_are_well_formed():
    assert 3 <= len(ls.FIXTURE_CASES) <= 8
    assert {m for m, _, _ in ls.FIXTURE_CASES} == set(KINDS)
    for method, problem_id, params in ls.FIXTURE_CASES:
        prob = problems.get(problem_id)  # a missing id fails here (the export would fail too)
        res = numopt.run(method, prob, **params)
        assert_valid_result(res, max_iter=params.get("max_iter", 50))
        assert len(res.trace) < 300


ZOOM_LABELS = {"bisection", "cubic", "cubic_clamped", "quadratic", "quadratic_clamped"}
PHASES = {"start", "backtrack", "expand", "zoom", "bisect", "exact"}


def test_fixture_cases_cover_every_phase_and_zoom_branch():
    """The FIXTURE_CASES comment promises every phase and zoom interpolation branch, so the TS
    parity test checks each of them (audit: bisection and cubic_clamped were missing)."""
    doc = ls.__doc__ or ""
    for label in ZOOM_LABELS:
        assert f'"{label}"' in doc  # the documented set of info["interp"] values
    interp: set[str] = set()
    phases: set[str] = set()
    zoom_orientations: set[bool] = set()
    accepted_in_bracketing = False
    for method, problem_id, params in ls.FIXTURE_CASES:
        res = numopt.run(method, problems.get(problem_id), **params)
        assert res.converged, (method, problem_id, res.message)
        phases |= {s.info["phase"] for s in res.trace}
        if method == "strong_wolfe":
            zoom = [s for s in res.trace if s.info["phase"] == "zoom"]
            interp |= {s.info["interp"] for s in zoom}
            zoom_orientations |= {s.info["alpha_lo"] < s.info["alpha_hi"] for s in zoom}
            accepted_in_bracketing |= res.trace[-1].info["phase"] == "expand"
    assert interp == ZOOM_LABELS
    assert phases == PHASES
    # Both calls of Alg. 3.5: zoom(α_{i-1}, α_i) and zoom(α_i, α_{i-1}) (α_lo > α_hi).
    assert zoom_orientations == {True, False}
    assert accepted_in_bracketing


# --------------------------------------------------------------------------------------
# Demo methods: trace semantics on two problems and both directions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prob", [ROSEN, QUAD], ids=["rosenbrock", "quadratic"])
@pytest.mark.parametrize("direction", ["steepest", "newton"])
def test_demo_trace_contract(kind, prob, direction):
    res = numopt.run(kind, prob, direction=direction)
    assert_valid_result(res, max_iter=50)
    assert res.converged, res.message
    x0 = np.asarray(prob.x0, dtype=float)
    g0 = prob.grad(x0)
    p = -g0 if direction == "steepest" else -np.linalg.solve(prob.hess(x0), g0)
    assert_allclose(res.extra["direction"], p, rtol=1e-15)
    phi0 = prob.f(x0)

    s0 = res.trace[0]
    assert s0.step_size == 0.0 and s0.info["alpha"] == 0.0 and s0.info["phase"] == "start"
    assert s0.fun == phi0 and s0.info["dphi"] == pytest.approx(float(g0 @ p), rel=1e-15)
    assert res.n_iter == len(res.trace) - 1 == res.trace[-1].k
    for s in res.trace[1:]:
        a = s.step_size
        assert a == s.info["alpha"] > 0
        assert np.array_equal(s.x, x0 + a * p)
        assert s.fun == prob.f(x0 + a * p) == s.info["phi"]
        # φ(0) is never overwritten (legacy bug, AUDIT §1.14).
        assert s.info["phi0"] == phi0
        # The reported Armijo flag agrees with an independent evaluation (difference form).
        assert s.info["conditions"]["armijo"] == (
            s.fun - phi0 <= s.info["c1"] * a * s.info["dphi0"]
        )
        if s.info["dphi"] is not None:
            assert s.info["dphi"] == float(prob.grad(x0 + a * p) @ p)
            assert s.grad_norm == pytest.approx(np.linalg.norm(prob.grad(x0 + a * p)), rel=1e-15)
        if s.info["interval"] is not None:
            lo, hi = s.info["interval"]
            assert lo < a < hi or (s.info["phase"] == "expand" and a == hi)
    assert [s.info["accepted"] for s in res.trace] == [False] * res.n_iter + [True]
    cond = res.trace[-1].info["conditions"]
    if kind == "exact_quadratic":  # the demo has f_err = 0: decrease; armijo is diagnostic
        assert cond["decrease"] and cond["strong_curvature"] is None
    else:
        assert all(v for v in cond.values())
    a_star = res.extra["alpha"]
    assert np.array_equal(res.x, x0 + a_star * p) and res.fun < phi0
    if kind != "exact_quadratic":
        assert holds(kind, prob.f, prob.grad, x0, p, a_star, c1=c1_for(kind))

    # Exact evaluation counts: φ(0) + one f per trial; ∇f(x0) + one per computed φ'(α).
    assert res.n_fev == len(res.trace)
    assert res.n_gev == 1 + sum(s.info["dphi"] is not None for s in res.trace[1:])
    assert res.n_hev == (1 if direction == "newton" or kind == "exact_quadratic" else 0)


def test_demo_on_scalar_problem():
    prob = Problem(
        "s", "s", "", lambda x: (x - 2.0) ** 2 + math.exp(x), 1, (-1.0, 3.0),
        grad=lambda x: 2.0 * (x - 2.0) + math.exp(x), hess=lambda x: 2.0 + math.exp(x), x0=-1.0,
    )  # fmt: skip
    for kind in KINDS:
        res = numopt.run(kind, prob)
        assert_valid_result(res)
        assert res.converged and isinstance(res.x, float) and res.fun < prob.f(-1.0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("direction", ["steepest", "newton"])
def test_demo_bare_callable_uses_finite_differences(kind, direction):
    """A bare callable (no ∇f, no ∇²f) runs with central differences, as in the other n-D
    families (audit finding). Oracle: the same demo with the exact derivatives."""
    res = numopt.run(kind, rosen, x0=[-1.2, 1.0], direction=direction)
    ref = numopt.run(kind, ROSEN, direction=direction)
    assert_valid_result(res, max_iter=50)
    assert res.converged, res.message
    # NOTE: tolerances from the difference errors (N&W §8.1): a central-difference gradient
    # has relative error ~ε^{2/3} ≈ 1e-10 here, so steepest-descent trials agree to 1e-6 (margin
    # for the interpolation). The Hessian is a central difference of that gradient: its error
    # is ~(gradient error)/h ≈ ε^{1/3} ≈ 6e-6 relative (measured 1e-5), hence 1e-4 for Newton.
    rtol = 1e-6 if direction == "steepest" else 1e-4
    assert [s.info["phase"] for s in res.trace] == [s.info["phase"] for s in ref.trace]
    assert_allclose(
        [s.info["alpha"] for s in res.trace], [s.info["alpha"] for s in ref.trace], rtol=rtol
    )
    assert_allclose(res.extra["direction"], ref.extra["direction"], rtol=rtol)
    # Exact counts: each difference gradient costs 2n f evaluations, each difference Hessian
    # 2n gradient evaluations (numopt.core.diff), and every call is counted.
    n = 2
    assert res.n_hev == ref.n_hev
    assert res.n_gev == ref.n_gev + 2 * n * res.n_hev
    assert res.n_fev == ref.n_fev + 2 * n * res.n_gev


def test_demo_bare_callable_with_a_three_dimensional_start():
    # The dimension comes from x0 (vector_problem(..., x0=x0)), not from a default of 2.
    res = numopt.run("strong_wolfe", lambda x: float(np.sum((x - 1.0) ** 2)), x0=[0.0, 2.0, 5.0])
    assert_valid_result(res)
    assert res.converged and len(res.extra["direction"]) == 3


def test_demo_problem_without_hessian_uses_a_difference_hessian():
    prob = Problem("nh", "nh", "", rosen, 2, (), grad=rosen_grad, x0=[-1.2, 1.0])
    for kind in KINDS:
        res = numopt.run(kind, prob, direction="newton")
        ref = numopt.run(kind, ROSEN, direction="newton")
        assert res.converged and ref.converged
        assert_allclose(res.extra["direction"], ref.extra["direction"], rtol=1e-6)
        assert res.n_hev == 1 and res.n_gev == ref.n_gev + 2 * 2 and res.n_fev == ref.n_fev


# --------------------------------------------------------------------------------------
# search(): exact counts, f0/g0 reuse, scale invariance
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_search_counts_and_trials_are_exact(kind):
    calls_f: list[np.ndarray] = []
    calls_g: list[np.ndarray] = []

    def f(x):
        calls_f.append(x.copy())
        return rosen(x)

    def g(x):
        calls_g.append(x.copy())
        return rosen_grad(x)

    x = np.array([-1.2, 1.0])
    p = -rosen_grad(x)
    r = search(kind, f, g, x, p, hess=rosen_hess)
    assert r.success and r.n_fev == len(calls_f) and r.n_gev == len(calls_g)
    # f(x) first, then one call per trial at x + α p in the recorded order.
    assert np.array_equal(calls_f[0], x)
    assert len(r.trials) == len(calls_f) - 1
    for (a, phi), xc in zip(r.trials, calls_f[1:], strict=True):
        assert np.array_equal(xc, x + a * p) and phi == rosen(xc)
    assert r.trials[-1][0] == r.alpha and r.f_new == rosen(x + r.alpha * p)
    if kind in ("strong_wolfe", "weak_wolfe"):
        assert r.g_new is not None and np.array_equal(r.g_new, rosen_grad(x + r.alpha * p))
    else:
        assert r.g_new is None
    assert r.n_hev == (1 if kind == "exact_quadratic" else 0)

    # Supplying f0 / g0 skips exactly those two evaluations and changes nothing else.
    r2 = search(kind, rosen, rosen_grad, x, p, f0=rosen(x), g0=rosen_grad(x), hess=rosen_hess(x))
    assert (r2.alpha, r2.trials) == (r.alpha, r.trials)
    assert (r2.n_fev, r2.n_gev, r2.n_hev) == (r.n_fev - 1, r.n_gev - 1, 0)


@settings(max_examples=300, deadline=None)
@given(
    k=st.integers(-30, 30),
    x=hnp.arrays(np.float64, 2, elements=st.floats(-2, 2)),
    kind=st.sampled_from(SEARCH_KINDS),
)
def test_search_invariant_under_power_of_two_scaling_of_f(k, x, kind):
    """Scaling f by 2^k is exact in floating point and the tests are homogeneous in f,
    so every trial (and the interpolation formulas) must be reproduced bit for bit."""
    g = rosen_grad(x)
    assume(np.linalg.norm(g) > 1e-6)
    s = 2.0**k
    r1 = search(kind, rosen, rosen_grad, x, -g)
    r2 = search(kind, lambda z: s * rosen(z), lambda z: s * rosen_grad(z), x, -g)
    assert [a for a, _ in r1.trials] == [a for a, _ in r2.trials]
    assert r1.alpha == r2.alpha and r1.success == r2.success


# --------------------------------------------------------------------------------------
# Hypothesis: accepted steps satisfy the conditions (convex quadratics, Rosenbrock)
# --------------------------------------------------------------------------------------


@st.composite
def convex_quadratic_case(draw):
    n = draw(st.integers(1, 5))
    M = draw(hnp.arrays(np.float64, (n, n), elements=st.floats(-3, 3)))
    lam = draw(st.floats(0.1, 5.0))
    scale = 10.0 ** draw(st.integers(-3, 3))
    A = scale * (M @ M.T + lam * np.eye(n))
    b = draw(hnp.arrays(np.float64, n, elements=st.floats(-5, 5)))
    x = draw(hnp.arrays(np.float64, n, elements=st.floats(-5, 5)))
    r = draw(hnp.arrays(np.float64, n, elements=st.floats(-1, 1)))
    use_random_dir = draw(st.booleans())
    return A, b, x, r, use_random_dir


def _descent_direction(g: np.ndarray, r: np.ndarray, use_random: bool) -> np.ndarray:
    if not use_random:
        return -g
    p = r.copy()
    return -p if p @ g > 0 else p


def _resolvable(x: np.ndarray, g: np.ndarray, p: np.ndarray) -> bool:
    """p is a descent direction that rounding does not erase: ∇fᵀp is clearly negative and
    the step x + αp moves x for α ~ 1 (a tiny p makes x + αp == x in floating point)."""
    return bool(
        g @ p < -1e-8 * np.linalg.norm(g) * np.linalg.norm(p)
        and np.linalg.norm(p) > 1e-6 * (1.0 + np.linalg.norm(x))
    )


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=convex_quadratic_case(), kind=st.sampled_from(SEARCH_KINDS))
# Shrunk Hypothesis counterexample: the best decrease Δ* = 3.0e-18 is below ½ ulp(f(x)) =
# 6.9e-18, and no trial α = 2⁻ᵏ gives a computed f(x + αp) < f(x). The search must fail.
@example(
    case=(
        np.diag([9.125, 0.125]),
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.array([1.0, 2**-24]),
        True,
    ),
    kind="backtracking",
)
def test_accepted_steps_satisfy_conditions_on_convex_quadratics(case, kind):
    A, b, x, r, use_random = case
    f, grad, _ = quadratic(A, b)
    g = grad(x)
    p = _descent_direction(g, r, use_random)
    assume(_resolvable(x, g, p))
    res = search(kind, f, grad, x, p)
    # Every acceptable step is within a factor 2 of the line minimizer α* = -∇fᵀp / pᵀAp.
    # From α₀ = 1, 50 trials reach α ∈ [2⁻⁴⁹, α_max = 1e3] (halving / doubling), so success
    # is required when α* is well inside that range; outside it the search must fail honestly.
    # Success also needs a decrease that the computed f can show: the exact decrease at α*,
    # Δ* = (∇fᵀp)²/(2pᵀAp), must exceed the rounding of f at the trial points (Higham
    # Thm 3.5 bound, quadratic_f_err) by a margin; below it a rounding-level decrease is luck.
    alpha_star = -float(g @ p) / float(p @ A @ p)
    delta_star = 0.5 * alpha_star * -float(g @ p)
    f_err = quadratic_f_err(A, b, 0.0, x, x + alpha_star * p, x + 2.0 * alpha_star * p)
    if 1e-10 <= alpha_star <= 1e2 and delta_star > 100.0 * f_err:
        assert res.success, res.message
    if res.success:
        assert holds(kind, f, grad, x, p, res.alpha, c1=c1_for(kind))
        assert res.f_new < f(x)
    else:
        assert res.alpha == 0.0 and res.f_new == f(x)


@settings(max_examples=1000, deadline=None)
@given(
    x=hnp.arrays(np.float64, 2, elements=st.floats(-2, 2)),
    kind=st.sampled_from(SEARCH_KINDS),
    newton=st.booleans(),
    alpha0=st.sampled_from([1e-4, 1e-2, 1.0, 10.0]),
)
def test_accepted_steps_satisfy_conditions_on_rosenbrock(x, kind, newton, alpha0):
    g = rosen_grad(x)
    assume(np.linalg.norm(g) > 1e-6)
    if newton:
        try:
            p = -np.linalg.solve(rosen_hess(x), g)
        except np.linalg.LinAlgError:
            assume(False)
            return
        assume(g @ p < -1e-8 * np.linalg.norm(g) * np.linalg.norm(p))
    else:
        p = -g
    res = search(kind, rosen, rosen_grad, x, p, alpha0=alpha0)
    assert res.success, res.message
    assert holds(kind, rosen, rosen_grad, x, p, res.alpha, c1=c1_for(kind))
    assert res.n_fev == len(res.trials) + 1 <= 51


@settings(max_examples=300, deadline=None)
@given(x=hnp.arrays(np.float64, 2, elements=st.floats(-2, 2)))
def test_strong_wolfe_zoom_brackets_shrink(x):
    """Every zoom trial lies strictly inside its bracket, which shrinks by ≥ 10% per trial.

    Safeguard geometry (δ = 0.1): an interpolated trial lies in [lo + δw, hi - δw], a clamped
    one on an end of that interval, a bisection trial at the midpoint; and a bracket that did
    not shrink by 0.66 over two trials forces bisection (Moré & Thuente 1994, §4).
    """
    g = rosen_grad(x)
    assume(np.linalg.norm(g) > 1e-6)
    prob = Problem("r", "r", "", rosen, 2, (), grad=rosen_grad, x0=x)
    res = numopt.run("strong_wolfe", prob, alpha0=1.0)
    zoom = [s for s in res.trace if s.info["phase"] == "zoom"]
    for prev, cur in itertools.pairwise(zoom):
        lo0, hi0 = prev.info["interval"]
        lo1, hi1 = cur.info["interval"]
        assert lo0 <= lo1 < hi1 <= hi0
        assert hi1 - lo1 <= 0.9 * (hi0 - lo0) * (1 + 1e-12)
    widths = [s.info["interval"][1] - s.info["interval"][0] for s in zoom]
    for j, s in enumerate(zoom):
        lo, hi = s.info["interval"]
        t, w, how = s.step_size, hi - lo, s.info["interp"]
        assert lo < t < hi
        assert {s.info["alpha_lo"], s.info["alpha_hi"]} == {lo, hi}
        if how == "bisection":
            assert t == lo + 0.5 * w
        elif how in ("cubic_clamped", "quadratic_clamped"):
            assert t in (lo + 0.1 * w, hi - 0.1 * w)
        else:
            assert how in ("cubic", "quadratic") and lo + 0.1 * w <= t <= hi - 0.1 * w
        if j >= 2 and widths[j] > 0.66 * widths[j - 2]:
            assert how == "bisection"


# Moré & Thuente (1994), ACM TOMS 20:286–307, Table 1 test functions φ(α) and φ'(α) with their
# (c₁, c₂) = (μ, η). NOTE: functions 2–6 use μ = η, which N&W's 0 < c₁ < c₂ excludes; we take
# c₁ = 0.99 μ.
def _mt_gamma(b: float) -> float:
    return math.sqrt(1.0 + b * b) - b


def _mt_456(b1: float, b2: float) -> Callable[[float], tuple[float, float]]:
    def phi(a: float) -> tuple[float, float]:
        u, v = math.sqrt((1.0 - a) ** 2 + b2**2), math.sqrt(a * a + b1**2)
        g1, g2 = _mt_gamma(b1), _mt_gamma(b2)
        return g1 * u + g2 * v, g1 * (a - 1.0) / u + g2 * a / v

    return phi


def _mt3(a: float, b: float = 0.01, ell: int = 39) -> tuple[float, float]:
    if a <= 1.0 - b:
        p0, d0 = 1.0 - a, -1.0
    elif a >= 1.0 + b:
        p0, d0 = a - 1.0, 1.0
    else:
        p0, d0 = (a - 1.0) ** 2 / (2.0 * b) + b / 2.0, (a - 1.0) / b
    w = ell * math.pi / 2.0
    return p0 + 2.0 * (1.0 - b) / (ell * math.pi) * math.sin(w * a), d0 + (1.0 - b) * math.cos(
        w * a
    )


MORE_THUENTE: dict[str, tuple[Callable[[float], tuple[float, float]], float, float]] = {
    "mt1": (lambda a: (-a / (a * a + 2.0), (a * a - 2.0) / (a * a + 2.0) ** 2), 1e-3, 0.1),
    "mt2": (
        lambda a: (
            (a + 0.004) ** 5 - 2 * (a + 0.004) ** 4,
            5 * (a + 0.004) ** 4 - 8 * (a + 0.004) ** 3,
        ),
        0.099,
        0.1,
    ),
    "mt3": (_mt3, 0.099, 0.1),
    "mt4": (_mt_456(1e-3, 1e-3), 0.99e-3, 1e-3),
    "mt5": (_mt_456(1e-2, 1e-3), 0.99e-3, 1e-3),
    "mt6": (_mt_456(1e-3, 1e-2), 0.99e-3, 1e-3),
}


@pytest.mark.parametrize("name", sorted(MORE_THUENTE))
@pytest.mark.parametrize("alpha0", [1e-3, 1e-1, 1.0, 10.0, 1e3])
def test_strong_wolfe_on_more_thuente_functions(name, alpha0):
    fn, c1, c2 = MORE_THUENTE[name]
    f = lambda x: fn(float(x[0]))[0]  # noqa: E731
    g = lambda x: np.array([fn(float(x[0]))[1]])  # noqa: E731
    x, p = np.zeros(1), np.ones(1)
    r = search("strong_wolfe", f, g, x, p, c1=c1, c2=c2, alpha0=alpha0, alpha_max=1e4)
    assert r.success, r.message
    assert holds("strong_wolfe", f, g, x, p, r.alpha, c1=c1, c2=c2)
    # Regression for the zoom safeguard: midpoint replacement of rejected interpolation
    # estimates took 25–34 trials on function 2 (the clamp takes ≤ 22; SciPy needs 10 from
    # α = 1).
    assert len(r.trials) <= 24


# --------------------------------------------------------------------------------------
# Oracle: SciPy's strong Wolfe search (compare satisfaction, not equality)
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::scipy.optimize._linesearch.LineSearchWarning")
@settings(max_examples=500, deadline=None)
@given(
    x=hnp.arrays(np.float64, 2, elements=st.floats(-2, 2)),
    c2=st.sampled_from([0.1, 0.5, 0.9]),
)
def test_strong_wolfe_agrees_with_scipy_on_satisfaction(x, c2):
    g = rosen_grad(x)
    assume(np.linalg.norm(g) > 1e-6)
    p = -g
    alpha_sp = optimize.line_search(
        rosen, rosen_grad, x, p, gfk=g, old_fval=rosen(x), c1=1e-4, c2=c2, amax=1e3, maxiter=50
    )[0]
    ours = search("strong_wolfe", rosen, rosen_grad, x, p, c2=c2)
    assert ours.success, ours.message
    assert holds("strong_wolfe", rosen, rosen_grad, x, p, ours.alpha, c2=c2)
    if alpha_sp is not None:  # SciPy's answer passes the same checker (validates the checker)
        assert holds("strong_wolfe", rosen, rosen_grad, x, p, alpha_sp, c2=c2)


def test_strong_wolfe_matches_scipy_on_fixed_cases():
    cases = [
        (rosen, rosen_grad, np.array([-1.2, 1.0])),
        (rosen, rosen_grad, np.array([0.0, 0.0])),
        (rosen, rosen_grad, np.array([2.0, -1.0])),
        (quad_ill, QUAD.grad, np.array([3.0, 1.0])),
    ]
    for f, grad, x in cases:
        p = -grad(x)
        alpha_sp = optimize.line_search(f, grad, x, p, c1=1e-4, c2=0.9)[0]
        assert alpha_sp is not None
        ours = search("strong_wolfe", f, grad, x, p)
        assert ours.success
        assert holds("strong_wolfe", f, grad, x, p, ours.alpha)
        assert holds("strong_wolfe", f, grad, x, p, alpha_sp)


# --------------------------------------------------------------------------------------
# exact_quadratic exactness
# --------------------------------------------------------------------------------------


@st.composite
def exact_quadratic_case(draw):
    """A convex quadratic ½xᵀAx - bᵀx + c and a start x that is either anywhere in [-5, 5]ⁿ or
    within 10⁻¹⁴…10⁻² (relative) of the minimizer A⁻¹b, where f differences are rounding error.
    The constant c is 0, -f(x*) (so min f = 0 and f(x) near x* is pure rounding noise, audit
    finding) or ±10^k for k up to 12."""
    A, b, x, r, use_random = draw(convex_quadratic_case())
    x_star = np.linalg.solve(A, b)
    if draw(st.booleans()):
        offset = draw(hnp.arrays(np.float64, x.size, elements=st.floats(-1, 1)))
        x = x_star + offset * 10.0 ** draw(st.integers(-14, -2)) * (1.0 + np.abs(x_star).max())
    c = draw(
        st.one_of(
            st.just(0.0),
            st.just(0.5 * float(b @ x_star)),
            st.builds(lambda k, s: s * 10.0**k, st.integers(-3, 12), st.sampled_from([-1, 1])),
        )
    )
    return A, b, c, x, r, use_random


#: b = x with f ≈ -2.3e-315 (subnormal): the old relative f_err bound underflowed to 0 here.
B_SUB = 1.8754795309089233e-156


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=exact_quadratic_case())
# Shrunk counterexamples in the subnormal range (see quadratic_f_err and alpha_underflow).
@example(
    case=(np.array([[2000.0]]), np.array([2.22507386e-309]), 0.0, np.zeros(1), np.ones(1), True)
)
@example(case=(np.array([[750.0]]), np.array([B_SUB]), 0.0, np.array([B_SUB]), np.zeros(1), False))
def test_exact_quadratic_is_the_exact_line_minimizer(case):
    A, b, c, x, r, use_random = case
    f, grad, hess = quadratic(A, b, c)
    g = grad(x)
    p = _descent_direction(g, r, use_random)
    n = x.size

    # ∇f(z)ᵀp has an absolute rounding error ≤ γ_n |p|ᵀ(|A||z| + |b|) (Higham Thm 3.5) with a
    # factor 100 margin, plus the underflow error of subnormal results (Higham §2.1).
    def slope_err(z: np.ndarray) -> float:
        return float(
            100 * n * EPS * (np.abs(p) @ (np.abs(A) @ np.abs(z) + np.abs(b)))
            + 100 * n * TINY * (1.0 + np.abs(p).sum())
        )

    # The computed slope must be a descent slope with correct digits: ∇f(x)ᵀp below -100 times
    # its rounding error. Otherwise the direction is noise and the model step is meaningless.
    # pᵀAp must be a normal float: a subnormal one (e.g. |p| = 1e-157) has fewer than 53
    # significant bits (Higham §2.1), so α = -∇fᵀp/pᵀAp itself is inaccurate (shrunk example:
    # A = 1, x = 1, p = 1.6e-157 gives α·p = 1 - 8.8e-11).
    assume(float(g @ p) < -100 * slope_err(x) and float(p @ A @ p) >= np.finfo(float).tiny)
    assume(np.all(np.isfinite(x)))
    # α depends on ∇f and A only, so the caller can bound the rounding of f at x and x + αp.
    alpha = -float(g @ p) / float(p @ (A @ p))
    y = x + alpha * p
    f_err = quadratic_f_err(A, b, c, x, y)
    res = search("exact_quadratic", f, grad, x, p, hess=hess, f_err=f_err)
    assert res.n_hev == 1 and res.trials[0][0] == alpha and len(res.trials) == 1
    phi_alpha = res.trials[0][1]
    # φ'(α*) = 0 up to the rounding of the two dot products ∇f(y)ᵀp and ∇f(x)ᵀp, plus the
    # underflow of a subnormal α or y (|Δα| ≤ TINY/2 moves φ' by pᵀAp·Δα, |Δyᵢ| ≤ TINY/2 by
    # ½|p|ᵀ|A|𝟙·TINY; shrunk example: A = 2000, b = 2.2e-309, x = 0, α = 1.1e-312 gives
    # φ'(α) = 4.7e-321, which no relative bound covers).
    alpha_underflow = TINY * float(np.abs(p) @ np.abs(A) @ (np.abs(p) + 1.0))
    assert abs(grad(y) @ p) <= slope_err(x) + slope_err(y) + alpha_underflow
    # Independent oracle: Brent's method on φ (its xtol limits the agreement to ~1e-8 rel.);
    # φ(α) may exceed φ(α_Brent) only by the rounding of the two values.
    phi = lambda a: f(x + a * p)  # noqa: E731
    a_brent = optimize.minimize_scalar(phi, bracket=(0.0, 2 * alpha), tol=1e-12).x
    f_err_brent = quadratic_f_err(A, b, c, x + a_brent * p)
    assert phi(alpha) <= phi(a_brent) + 2 * max(f_err, f_err_brent) + 1e-12 * abs(phi(a_brent))
    # On a quadratic the exact step is the line minimizer, so a rise of the computed f is the
    # evaluation rounding of f at x and y, at most 2·f_err (the bound is rigorous), and the
    # search must accept it (audit: "f increased" on exact quadratics).
    f0 = f(x)
    assert phi_alpha - f0 <= 2 * f_err
    assert res.success, res.message
    assert res.alpha == alpha and res.f_new == phi_alpha
    if phi_alpha > f0:
        assert "within the rounding error of f" in res.message
        # The gradient test ran, and its ∇f(y) is returned for reuse.
        assert res.n_gev == 2 and res.g_new is not None and np.array_equal(res.g_new, grad(y))
        # Without the caller's bound (f_err = 0) no rise is called rounding: an honest failure
        # that names the rounding bound, not an "overshoot".
        r0 = search("exact_quadratic", f, grad, x, p, hess=hess)
        assert not r0.success and r0.trials == res.trials and r0.alpha == 0.0
        assert "f_err" in r0.message and "overshoots" not in r0.message
    else:
        assert res.g_new is None and res.n_gev == 1


@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(1, 6),
    seed=st.integers(0, 2**32 - 1),
    log_kappa=st.floats(0.0, 8.0),
    log_offset=st.floats(-14.0, -1.0),
    steepest=st.booleans(),
    offset_kind=st.sampled_from(["zero", "min_zero", "large"]),
)
def test_exact_quadratic_succeeds_on_ill_conditioned_quadratics(
    n, seed, log_kappa, log_offset, steepest, offset_kind
):
    """A = Q diag(λ) Qᵀ with κ(A) up to 1e8 and x near A⁻¹b: the computed f can rise by up to
    ~1e6·nε|f| (cancellation in ½xᵀAx - bᵀx + c), far above any |f|-relative rounding level,
    while f decreases in exact arithmetic. With min f = 0 (c = -f(x*)) |f(x)| is itself
    rounding noise (audit finding). With the term-magnitude bound f_err the search must accept
    every such rise; the verdict must not depend on c."""
    rng = np.random.default_rng(seed)  # test data only (the methods never draw randoms)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = 10.0 ** rng.uniform(0.0, log_kappa, n)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.normal(size=n) * 10.0 ** rng.uniform(-2, 3)
    x_star = np.linalg.solve(A, b)
    x = x_star + rng.normal(size=n) * 10.0**log_offset * (1.0 + np.abs(x_star).max())
    c = {"zero": 0.0, "min_zero": 0.5 * float(b @ x_star), "large": 1e6 * rng.normal()}[offset_kind]
    f, grad, _ = quadratic(A, b, c)
    g = grad(x)
    p = -g if steepest else rng.normal(size=n)
    p = -p if g @ p > 0 else p
    # The slope ∇f(x)ᵀp must have correct digits (100x its rounding bound, Higham Thm 3.5).
    slope_err = n * EPS * (np.abs(p) @ (np.abs(A) @ np.abs(x) + np.abs(b)))
    assume(float(g @ p) < -100 * slope_err and float(p @ A @ p) >= np.finfo(float).tiny)
    alpha = -float(g @ p) / float(p @ (A @ p))
    f_err = quadratic_f_err(A, b, c, x, x + alpha * p)
    res = search("exact_quadratic", f, grad, x, p, hess=A, f_err=f_err)
    assert res.success, res.message
    assert res.trials[0][0] == alpha and res.alpha == alpha and res.f_new == res.trials[0][1]
    if res.f_new > f(x):
        assert "within the rounding error of f" in res.message
        assert res.n_gev == 2 and res.g_new is not None
    else:
        assert res.n_gev == 1 and res.g_new is None


@settings(max_examples=500, deadline=None)
@given(case=convex_quadratic_case())
def test_exact_quadratic_along_newton_direction_is_one(case):
    A, b, x, _, _ = case
    f, grad, _ = quadratic(A, b)
    g = grad(x)
    assume(np.linalg.norm(g) > 1e-8 * (np.linalg.norm(A) * np.linalg.norm(x) + 1.0))
    p = -np.linalg.solve(A, g)
    assume(g @ p < 0)
    res = search("exact_quadratic", f, grad, x, p, hess=A)
    # NOTE: rtol 1e-9 = κ(A)·eps margin; κ(A) ≤ ~1e3 for these draws (λ_min ≥ 0.1·scale).
    assert_allclose(res.alpha, 1.0, rtol=1e-9)
    # The step lands on the minimizer A⁻¹b (oracle: numpy solve).
    assert_allclose(x + res.alpha * p, np.linalg.solve(A, b), rtol=1e-8, atol=1e-8)


# --------------------------------------------------------------------------------------
# Interpolation formulas (N&W eqs. 3.58, 3.59)
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    c0=st.floats(0.05, 5) | st.floats(-5, -0.05),
    a=st.floats(-3, 3),
    width=st.floats(0.1, 4),
    u=st.floats(-1, 2),
    sep=st.floats(0.1, 10),
    swap=st.booleans(),
)
def test_cubic_interpolation_is_exact_on_cubics(c0, a, width, u, sep, swap):
    """Eq. (3.59) on P with P'(t) = 3c₀(t - r_min)(t - r_other): the oracle is r_min.

    The local minimizer r_min lies in [a - w, b + w]; the zoom only uses interior points
    (safeguard), and far extrapolation or merging critical points (separation → 0) are
    ill-conditioned (errors grow like (distance/w)² and like √ε/separation).
    """
    r_min = a + u * width
    r_other = r_min - math.copysign(sep, c0)  # P''(r_min) = 3c₀(r_min - r_other) > 0
    coef = [c0, -1.5 * c0 * (r_min + r_other), 3.0 * c0 * r_min * r_other, 0.7]
    P = np.poly1d(coef)
    dP = P.deriv()
    b = a + width
    lo, hi = (b, a) if swap else (a, b)
    t = ls._cubic_minimizer(lo, P(lo), dP(lo), hi, P(hi), dP(hi))
    assert t is not None
    # NOTE: atol 1e-9 covers rounding of the coefficients (|r| ≤ 13) amplified by 1/sep ≤ 10.
    assert_allclose(t, r_min, rtol=1e-10, atol=1e-9)


@settings(max_examples=500, deadline=None)
@given(
    q=st.floats(0.1, 10), m=st.floats(-5, 5), c=st.floats(-5, 5), a=st.floats(-3, 3),
    width=st.floats(-4, 4),
)  # fmt: skip
def test_quadratic_interpolation_is_exact_on_quadratics(q, m, c, a, width):
    assume(abs(width) > 0.1)
    phi = lambda t: q * (t - m) ** 2 + c  # noqa: E731
    dphi = lambda t: 2 * q * (t - m)  # noqa: E731
    b = a + width
    t = ls._quadratic_minimizer(a, phi(a), dphi(a), b, phi(b))
    assert t is not None
    assert_allclose(t, m, rtol=1e-10, atol=1e-10)


def test_interpolation_reports_no_minimizer():
    # Concave quadratic data: no minimizer.
    assert ls._quadratic_minimizer(0.0, 0.0, -1.0, 1.0, -2.0) is None
    # φ = -t³ - t has no stationary point (d₁² - φ'(a)φ'(b) < 0).
    phi = lambda t: -(t**3) - t  # noqa: E731
    dphi = lambda t: -3 * t**2 - 1  # noqa: E731
    assert ls._cubic_minimizer(1.0, phi(1.0), dphi(1.0), 2.0, phi(2.0), dphi(2.0)) is None
    # φ = -t³ has only an inflection point at 0 (d₁² - φ'(a)φ'(b) = 0): not a minimizer.
    assert ls._cubic_minimizer(1.0, -1.0, -3.0, 2.0, -8.0, -12.0) is None
    # Coincident end points: no interpolant (and no division by b - a = 0).
    assert ls._cubic_minimizer(1.0, 0.0, -1.0, 1.0, 0.0, 1.0) is None
    assert ls._quadratic_minimizer(1.0, 0.0, -1.0, 1.0, 0.0) is None


# --------------------------------------------------------------------------------------
# Weak versus strong Wolfe (AUDIT §1.14: legacy "wolfe" was strong Wolfe)
# --------------------------------------------------------------------------------------


def test_weak_wolfe_accepts_a_step_that_strong_wolfe_rejects():
    f = lambda x: float(0.5 * (x[0] - 1.0) ** 2)  # noqa: E731
    grad = lambda x: np.array([x[0] - 1.0])  # noqa: E731
    x, p = np.array([0.0]), np.array([1.0])
    weak = search("weak_wolfe", f, grad, x, p, alpha0=1.95)
    assert weak.success and weak.alpha == 1.95 and len(weak.trials) == 1
    strong = search("strong_wolfe", f, grad, x, p, alpha0=1.95)
    assert strong.success and strong.alpha != 1.95
    assert abs(strong.g_new[0]) <= 0.9  # type: ignore[index]


def test_strong_wolfe_cubic_zoom_is_exact_on_a_quadratic():
    # With c₂ = 0.1 and α₀ = 1.75α*, φ'(α₀) > 0 and |φ'(α₀)| > c₂|φ'(0)|, so Alg. 3.5 calls
    # zoom(α₀, 0) with φ' known at both ends; the cubic through quadratic data is exact.
    x0 = np.asarray(QUAD.x0, dtype=float)
    g = A_ILL @ x0 - B_ILL
    alpha_star = float(g @ g) / float(g @ A_ILL @ g)
    res = numopt.run("strong_wolfe", QUAD, alpha0=1.75 * alpha_star, c2=0.1)
    assert res.converged and res.n_iter == 2
    assert res.trace[-1].info["interp"] == "cubic"
    assert_allclose(res.extra["alpha"], alpha_star, rtol=1e-12)


def test_goldstein_expands_a_short_first_step():
    res = numopt.run("goldstein", QUAD, alpha0=1e-4)
    assert res.converged
    phases = [s.info["phase"] for s in res.trace[1:]]
    assert phases[0] == "expand"
    s = res.trace[-1]
    a, fa, phi0, d0 = s.info["alpha"], s.info["phi"], s.info["phi0"], s.info["dphi0"]
    assert phi0 + 0.75 * a * d0 <= fa <= phi0 + 0.25 * a * d0


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "kw", "match"),
    [
        ("wolfe", {}, "unknown"),
        ("backtracking", {"c1": 0.0}, "c1"),
        ("backtracking", {"rho": 1.0}, "rho"),
        ("backtracking", {"alpha0": -1.0}, "alpha0"),
        ("backtracking", {"max_iter": 0}, "max_iter"),
        ("strong_wolfe", {"c1": 0.5, "c2": 0.4}, "c2"),
        ("weak_wolfe", {"c2": 1.0}, "c2"),
        ("goldstein", {"c1": 0.5}, "1/2"),
        ("strong_wolfe", {"alpha0": 10.0, "alpha_max": 1.0}, "alpha_max"),
        # audit: alpha_max = inf doubled α to inf and evaluated f at x + inf·p
        ("strong_wolfe", {"alpha_max": math.inf}, "finite alpha_max"),
        ("weak_wolfe", {"alpha_max": math.inf}, "finite alpha_max"),
        ("goldstein", {"alpha_max": math.inf}, "finite alpha_max"),
        ("weak_wolfe", {"alpha_max": math.nan}, "alpha_max"),
        # audit: max_iter = inf raised OverflowError, nan a ValueError from int()
        ("backtracking", {"max_iter": math.inf}, "max_iter must be a positive integer"),
        ("strong_wolfe", {"max_iter": math.nan}, "max_iter must be a positive integer"),
        ("goldstein", {"max_iter": 2.5}, "max_iter must be a positive integer"),
        ("weak_wolfe", {"max_iter": "10"}, "max_iter must be a positive integer"),
        ("exact_quadratic", {}, "Hessian"),
        ("exact_quadratic", {"hess": -np.eye(2)}, "pᵀ"),
        ("exact_quadratic", {"hess": np.eye(3)}, "shape"),
        # f_err is an absolute rounding bound of f: finite and ≥ 0
        ("exact_quadratic", {"hess": rosen_hess, "f_err": -1e-12}, "f_err"),
        ("exact_quadratic", {"hess": rosen_hess, "f_err": math.nan}, "f_err"),
        ("exact_quadratic", {"hess": rosen_hess, "f_err": math.inf}, "f_err"),
        ("exact_quadratic", {"hess": rosen_hess, "f_err": "1e-9"}, "f_err"),
    ],
)
def test_search_rejects_invalid_input(kind, kw, match):
    x = np.array([-1.2, 1.0])
    with pytest.raises(ValueError, match=match):
        search(kind, rosen, rosen_grad, x, -rosen_grad(x), **kw)


def test_search_validation_accepts_what_the_kind_does_not_use():
    x = np.array([-1.2, 1.0])
    p = -rosen_grad(x)
    # backtracking never expands, so alpha_max (even inf) is unused; an integral float or a
    # NumPy integer is a valid max_iter.
    r = search("backtracking", rosen, rosen_grad, x, p, alpha_max=math.inf, max_iter=50.0)  # pyright: ignore[reportArgumentType]
    assert r.success
    r2 = search("strong_wolfe", rosen, rosen_grad, x, p, max_iter=np.int64(50))  # pyright: ignore[reportArgumentType]
    assert r2.success
    # f_err belongs to exact_quadratic only; the other kinds ignore it.
    assert search("goldstein", rosen, rosen_grad, x, p, f_err=math.nan).success
    with pytest.raises(ValueError, match="finite alpha_max"):
        numopt.run("strong_wolfe", ROSEN, alpha_max=math.inf)


def test_exact_quadratic_accepts_a_scalar_hessian_in_one_dimension():
    # The Problem convention for 1-D problems: hess returns a float (audit: search() raised
    # "hess must have shape (1, 1), got ()"). f = (x - 2)² + eˣ is not quadratic, so the step
    # is the Newton step α = 1 along p = -f'(x)/f''(x) (oracle: the scalar Newton formula).
    f = lambda z: float((z[0] - 2.0) ** 2 + math.exp(z[0]))  # noqa: E731
    g = lambda z: np.array([2.0 * (z[0] - 2.0) + math.exp(z[0])])  # noqa: E731
    h = lambda z: 2.0 + math.exp(z[0])  # noqa: E731
    x = np.array([-1.0])
    p = -g(x) / h(x)
    for hess in (h, h(x), np.array(h(x)), np.array([h(x)]), np.array([[h(x)]])):
        r = search("exact_quadratic", f, g, x, p, hess=hess)
        assert r.success and r.alpha == pytest.approx(1.0, rel=1e-15)
    with pytest.raises(ValueError, match="shape"):
        search("exact_quadratic", f, g, x, p, hess=np.eye(2))
    with pytest.raises(ValueError, match="shape"):  # n = 2: a scalar is not a 2x2 Hessian
        search(
            "exact_quadratic", rosen, rosen_grad, np.zeros(2), -rosen_grad(np.zeros(2)), hess=2.0
        )


@pytest.mark.parametrize("kind", KINDS)
def test_search_rejects_non_descent_directions(kind):
    # At x = (1, 2), ∇f = (−400, 200) exactly, so ∇fᵀp is exact for the orthogonal p = (200, 400)
    # on every IEEE platform (−80000 + 80000 = 0, also with FMA or another summation order). At
    # a point such as (−1.2, 1) the products round and ∇fᵀp is ±1e-14 depending on the CPU.
    x = np.array([1.0, 2.0])
    g = rosen_grad(x)
    assert np.array_equal(g, [-400.0, 200.0])
    for p in (g, np.zeros(2), np.array([g[1], -g[0]])):  # ascent, zero, orthogonal
        with pytest.raises(ValueError, match="descent"):
            search(kind, rosen, rosen_grad, x, p, hess=rosen_hess)
    with pytest.raises(ValueError, match="finite"):
        search(kind, lambda z: math.nan, rosen_grad, x, -g, hess=rosen_hess)


def _linear_grad(x: np.ndarray) -> np.ndarray:
    return np.array([-1.0, -1.0])


LINEAR = Problem(
    "linear", "unbounded", "", lambda x: float(-x[0] - x[1]), 2, (),
    grad=_linear_grad, hess=lambda x: np.zeros((2, 2)), x0=[0.0, 0.0],
)  # fmt: skip


@pytest.mark.parametrize("kind", ["strong_wolfe", "weak_wolfe", "goldstein"])
def test_unbounded_below_stops_at_alpha_max(kind):
    res = numopt.run(kind, LINEAR, alpha_max=100.0)
    assert_valid_result(res, max_iter=50)
    assert not res.converged and "alpha_max" in res.message
    assert res.trace[-1].step_size == 100.0 and not res.trace[-1].info["accepted"]
    assert np.array_equal(res.x, [0.0, 0.0]) and res.extra["alpha"] == 0.0
    r = search(kind, LINEAR.f, _linear_grad, np.zeros(2), np.ones(2), alpha_max=100.0)
    assert not r.success and r.alpha == 0.0 and r.f_new == 0.0


def test_backtracking_accepts_first_step_on_linear_function():
    res = numopt.run("backtracking", LINEAR)
    assert res.converged and res.n_iter == 1 and res.extra["alpha"] == 1.0


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("backtracking", {"alpha0": 100.0, "max_iter": 3}),
        ("strong_wolfe", {"alpha0": 100.0, "max_iter": 1}),
        ("weak_wolfe", {"alpha0": 100.0, "max_iter": 2}),
        ("goldstein", {"alpha0": 100.0, "max_iter": 2}),
    ],
)
def test_max_iter_is_reported(kind, params):
    res = numopt.run(kind, ROSEN, **params)
    assert_valid_result(res, max_iter=params["max_iter"])
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == params["max_iter"]
    assert not any(s.info["accepted"] for s in res.trace)
    assert np.array_equal(res.x, ROSEN.x0) and res.fun == rosen(np.asarray(ROSEN.x0))


def _nan_beyond(cut: float) -> tuple[Callable, Callable]:
    """f = (x₀ - 3)² + x₁² for x₀ < cut, NaN beyond (a function with a bounded domain)."""

    def f(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2 + x[1] ** 2) if x[0] < cut else math.nan

    def g(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (x[0] - 3.0), 2.0 * x[1]]) if x[0] < cut else np.full(2, math.nan)

    return f, g


@pytest.mark.parametrize("kind", SEARCH_KINDS)
def test_non_finite_trial_values_shrink_the_step(kind):
    # Along p = (6, 0) from 0: φ(α) = (6α - 3)², α* = ½, NaN for α ≥ 2/3. Every kind's
    # acceptable set meets [0, 2/3) (Goldstein with c = ¼: α ∈ [¼, ¾]).
    f, g = _nan_beyond(4.0)
    x, p = np.array([0.0, 0.0]), np.array([6.0, 0.0])
    r = search(kind, f, g, x, p)
    assert r.success and math.isfinite(r.f_new) and x[0] + r.alpha * p[0] < 4.0
    assert any(math.isnan(phi) for _, phi in r.trials)
    assert holds(kind, f, g, x, p, r.alpha, c1=c1_for(kind))


def test_goldstein_fails_when_its_acceptable_set_is_outside_the_domain():
    # NaN for α ≥ ¼, but the Goldstein set is α ∈ [¼, ¾]: no acceptable step exists.
    f, g = _nan_beyond(1.5)
    x, p = np.array([0.0, 0.0]), np.array([6.0, 0.0])
    r = search("goldstein", f, g, x, p)
    assert not r.success and r.alpha == 0.0 and r.f_new == 9.0
    assert all(a < 0.25 or math.isnan(phi) for a, phi in r.trials)
    # Wolfe steps exist there ((1 - c₂)α* = 0.05 ≤ α < ¼), so the Wolfe searches succeed.
    for kind in ("strong_wolfe", "weak_wolfe", "backtracking"):
        rk = search(kind, f, g, x, p)
        assert rk.success and holds(kind, f, g, x, p, rk.alpha)


def _demo_failure(kind: str, prob: Problem, match: str, **params: Any) -> None:
    res = numopt.run(kind, prob, **params)
    assert_valid_result(res)
    assert not res.converged and match in res.message
    assert res.n_iter == 0 and len(res.trace) == 1


def test_demo_failures_do_not_raise():
    stationary = Problem(
        "st", "st", "", lambda x: float(x @ x), 2, (), grad=lambda x: 2 * x,
        hess=lambda x: 2 * np.eye(2), x0=[0.0, 0.0],
    )  # fmt: skip
    saddle = Problem(
        "sa", "sa", "", lambda x: float(x[0] ** 2 - x[1] ** 2), 2, (),
        grad=lambda x: np.array([2 * x[0], -2 * x[1]]),
        hess=lambda x: np.diag([2.0, -2.0]), x0=[1.0, 1.0],
    )  # fmt: skip
    singular = Problem(
        "sg", "sg", "", lambda x: float(x[0] ** 2), 2, (),
        grad=lambda x: np.array([2 * x[0], 0.0]), hess=lambda x: np.diag([2.0, 0.0]), x0=[1.0, 1.0],
    )  # fmt: skip
    concave = Problem(
        "cc", "cc", "", lambda x: float(-x[0] ** 2 + x[1] ** 2), 2, (),
        grad=lambda x: np.array([-2 * x[0], 2 * x[1]]),
        hess=lambda x: np.diag([-2.0, 2.0]), x0=[1.0, 0.0],
    )  # fmt: skip
    f_nan, g_nan = _nan_beyond(1.5)
    bad_start = Problem("nan", "nan", "", f_nan, 2, (), grad=g_nan, x0=[2.0, 0.0])
    for kind in KINDS:
        _demo_failure(kind, stationary, "stationary")
        _demo_failure(kind, saddle, "not a descent direction", direction="newton")
        _demo_failure(kind, singular, "singular", direction="newton")
    for kind in SEARCH_KINDS:
        _demo_failure(kind, bad_start, "not finite")
    _demo_failure("exact_quadratic", concave, "no minimizer")


def test_exact_quadratic_fails_when_f_increases():
    # f = √(1 + x₀²) + x₁²: the curvature at x₀ = 2 is tiny, so the model step overshoots
    # (f rises from √5 ≈ 2.24 to ≈ 8.06): an uphill step is not a success (audit finding).
    prob = Problem(
        "hyp", "hyp", "", lambda x: float(math.sqrt(1 + x[0] ** 2) + x[1] ** 2), 2, (),
        grad=lambda x: np.array([x[0] / math.sqrt(1 + x[0] ** 2), 2 * x[1]]),
        hess=lambda x: np.diag([(1 + x[0] ** 2) ** -1.5, 2.0]), x0=[2.0, 0.0],
    )  # fmt: skip
    res = numopt.run("exact_quadratic", prob)
    assert_valid_result(res)
    assert not res.converged and "f increased" in res.message and "overshoots" in res.message
    s, f_first = res.trace[-1], res.trace[0].fun
    assert res.n_iter == 1 and s.fun is not None and f_first is not None and s.fun > f_first
    assert s.step_size == pytest.approx(5.0**1.5)  # 1 / f''(2) = (1 + 4)^{3/2}
    assert s.info["conditions"] == {"armijo": False, "decrease": False, "strong_curvature": False}
    assert not s.info["accepted"]
    # f rose, so the gradient test ran: φ'(α) > 0 past the line minimizer.
    assert s.info["dphi"] is not None and s.info["dphi"] > 0 and res.n_gev == 2
    assert np.array_equal(res.x, [2.0, 0.0]) and res.fun == f_first and res.extra["alpha"] == 0.0
    x = np.array([2.0, 0.0])
    grad = prob.grad
    assert grad is not None
    # Even a generous rounding bound does not excuse a step that ∇f shows to overshoot.
    for f_err in (0.0, 1e-10, 10.0):
        r = search("exact_quadratic", prob.f, grad, x, -grad(x), hess=prob.hess, f_err=f_err)
        assert not r.success and r.alpha == 0.0 and r.f_new == math.sqrt(5.0)
        assert len(r.trials) == 1 and r.trials[0][1] > math.sqrt(5.0)


@pytest.mark.parametrize(
    ("s", "rise", "bump", "f_err", "ok", "match"),
    [
        (1.0, 0.0, 0.0, 0.0, True, "exact minimizer"),  # no rise: decrease, no ∇f evaluation
        # a rise of 4 ulps with ∇f at the line minimizer: without a rounding bound it is not
        # called rounding error (the old 100·n·ε·|f| level did, audit finding)
        (1.0, 8 * EPS, 0.0, 0.0, False, "exceeds the rounding bound 2·f_err = 0"),
        (1e-8, 8 * EPS, 0.0, 0.0, False, "pass f_err"),
        # with the caller's bound: rise ≤ 2·f_err is accepted, rise > 2·f_err is not
        (1.0, 8 * EPS, 0.0, 4 * EPS, True, "within the rounding error of f"),
        (1.0, 8 * EPS, 0.0, 3 * EPS, False, "exceeds the rounding bound"),
        # |φ'(α)| = 0.15·s ≤ 0.1·|φ'(0)| = 0.2·s: ∇f confirms the line minimizer
        (1.0, 1e3 * EPS, 0.15, 1e3 * EPS, True, "within the rounding error of f"),
        # |φ'(α)| = 0.25·s > 0.2·s: the model step misses the line minimizer, whatever f_err
        (1.0, 1e3 * EPS, 0.25, 1.0, False, "overshoots"),
        (1.0, 1e3 * EPS, -0.25, 1.0, False, "falls short of"),
        # a real rise of 10⁻⁵ is not rounding error, at any |f| (the old 10⁻⁶|f| test only
        # rejected it because φ(0) = 1, audit finding)
        (1.0, 1e-5, 0.0, 0.0, False, "exceeds the rounding bound"),
    ],
)
def test_exact_quadratic_acceptance_tests(s, rise, bump, f_err, ok, match):
    # φ(0) = 1, φ(α) = 1 + rise for α ≠ 0. The model q(α) = 1 - 2sα + α² (∇f(0)ᵀp = -2s,
    # pᵀHp = 2) puts the step at α = s; ∇f(s) = bump·s, so |φ'(α)| / |φ'(0)| = |bump| / 2.
    def f(x: np.ndarray) -> float:
        return 1.0 if x[0] == 0.0 else 1.0 + rise

    def g(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (x[0] - s) + (bump * s if x[0] != 0.0 else 0.0)])

    r = search("exact_quadratic", f, g, np.zeros(1), np.ones(1), hess=lambda x: 2.0, f_err=f_err)
    n_gev = 1 if rise == 0.0 else 2  # ∇f(x), plus ∇f(x + αp) when f rose
    assert r.trials == [(s, 1.0 + rise)]
    assert r.success is ok and r.alpha == (s if ok else 0.0)
    assert match in r.message and r.n_gev == n_gev and r.n_fev == 2 and r.n_hev == 1
    assert ("f increased" in r.message) is not ok
    if ok and n_gev == 2:
        assert r.g_new is not None and r.g_new[0] == bump * s
    # The demo (p = -∇f(0) = 2s, α = ½, the same point x = s) has f_err = 0: it accepts only
    # a decrease, and reports the same gradient test.
    prob = Problem("q1", "q1", "", lambda x: f(np.array([x])), 1, (),
                   grad=lambda x: float(g(np.array([x]))[0]), hess=lambda x: 2.0, x0=0.0)  # fmt: skip
    res = numopt.run("exact_quadratic", prob)
    assert_valid_result(res)
    assert res.converged is (rise == 0.0) and res.trace[-1].info["accepted"] is (rise == 0.0)
    assert res.trace[-1].x == s and res.n_gev == n_gev
    cond = res.trace[-1].info["conditions"]
    assert cond["decrease"] is (rise == 0.0)
    assert cond["strong_curvature"] is (None if rise == 0.0 else abs(bump) <= 0.2)


# φ(t) = C - t + ½t² + 2t³ - 1.4t⁴ (audit a2): φ'(0) = -1, φ''(0) = 1, so the model step is
# α = 1, where φ(1) - φ(0) = 0.1 and φ'(1) = 0.4: a real overshoot.
OVERSHOOT = (
    lambda C: lambda x: C - x[0] + 0.5 * x[0] ** 2 + 2 * x[0] ** 3 - 1.4 * x[0] ** 4,
    lambda x: np.array([-1 + x[0] + 6 * x[0] ** 2 - 5.6 * x[0] ** 3]),
    lambda x: np.array([[1 + 12 * x[0] - 16.8 * x[0] ** 2]]),
)
# φ(t) = C - t + ½t² + 4t³ - 3t⁴: the model step α = 1 is a local MAXIMUM of φ (φ'(1) = 0,
# φ''(1) = -11) with φ(1) - φ(0) = ½. The gradient test alone would accept it.
LOCAL_MAX = (
    lambda C: lambda x: C - x[0] + 0.5 * x[0] ** 2 + 4 * x[0] ** 3 - 3 * x[0] ** 4,
    lambda x: np.array([-1 + x[0] + 12 * x[0] ** 2 - 12 * x[0] ** 3]),
    lambda x: np.array([[1 + 24 * x[0] - 36 * x[0] ** 2]]),
)


@pytest.mark.parametrize("C", [0.0, -1e3, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12])
@pytest.mark.parametrize(("case", "match"), [(OVERSHOOT, "overshoots"), (LOCAL_MAX, "maximum")])
def test_exact_quadratic_real_rise_fails_for_every_offset(C, case, match):
    """Audit: with |f(x0)| ≥ 1e5 the old 10⁻⁶|f| test called the rise of 0.1 'rounding error'.
    The verdict must not depend on C, also with the caller's term-magnitude bound of f."""
    f_of, g, h = case
    f = f_of(C)
    # |terms of f| ≤ |C| + 10 on [0, 1]; γ_8 covers the few operations of the polynomial.
    f_err = gamma(8) * (abs(C) + 10.0)
    for err in (0.0, f_err):
        r = search("exact_quadratic", f, g, np.zeros(1), np.ones(1), hess=h, f_err=err)
        assert not r.success and r.alpha == 0.0 and r.f_new == f(np.zeros(1))
        assert [a for a, _ in r.trials] == [1.0] and r.n_gev == 2
        assert "f increased" in r.message and match in r.message
    if case is OVERSHOOT:  # the registered demo reports the same (audit: converged at C = 1e6)
        prob = Problem("a2", "a2", "", lambda t: f(np.array([t])), 1, (),
                       grad=lambda t: float(g(np.array([t]))[0]),
                       hess=lambda t: float(h(np.array([t]))[0, 0]), x0=0.0)  # fmt: skip
        res = numopt.run("exact_quadratic", prob)
        assert_valid_result(res)
        assert (
            not res.converged
            and res.fun == C
            and res.trace[-1].info["conditions"]
            == {
                "armijo": False,
                "decrease": False,
                "strong_curvature": False,
            }
        )


# --------------------------------------------------------------------------------------
# Offset invariance: f and f + C give the same search (audit finding)
# --------------------------------------------------------------------------------------


def _log_rosen(C: float, q: int) -> tuple[Callable, Callable, Callable]:
    """f = C + round_q(log(1 + rosen(x))), rounded to multiples of 2⁻q so that f + C is exact,
    with the derivatives of log(1 + rosen). Values stay below 2¹⁰ for every finite x (unlike
    rosen itself, whose huge trial values would make f + C inexact), and C is a multiple of
    2⁻q with |C| < 2^(42-q), so |f + C| < 2^(53-q) is a float. A coarse grid (q = 4) makes
    many trials tie, φ(α) = φ(0), which is where the rounding of a sum φ(0) + c₁αφ'(0) at the
    scale of |C| decides the verdict."""
    step = 2.0**-q

    def f(x: np.ndarray) -> float:
        v = math.log1p(rosen(x))
        if not math.isfinite(v):
            return v + C
        r = round(v / step) * step
        out = r + C
        assert out - C == r  # f + C is exact
        return out

    def g(x: np.ndarray) -> np.ndarray:
        return rosen_grad(x) / (1.0 + rosen(x))

    def h(x: np.ndarray) -> np.ndarray:
        r, gr = 1.0 + rosen(x), rosen_grad(x)
        return rosen_hess(x) / r - np.outer(gr, gr) / r**2

    return f, g, h


@st.composite
def exact_offset(draw):
    """A grid 2⁻q (q ∈ {24, 12, 4}) and C = ±m·2⁻q with m log-uniform in [1, 2⁴²)."""
    q = draw(st.sampled_from([24, 12, 4]))
    e = draw(st.integers(0, 41))
    m = draw(st.integers(2**e, 2 ** (e + 1) - 1))
    return q, draw(st.sampled_from([-1, 1])) * m * 2.0**-q


# Trial steps far along p overflow rosen to inf (then f = inf in both runs): expected.
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    x=hnp.arrays(np.float64, 2, elements=st.floats(-2, 2)),
    kind=st.sampled_from(KINDS),
    qC=exact_offset(),
    alpha0=st.sampled_from([1e-3, 1.0, 10.0]),
)
def test_search_is_invariant_under_an_added_constant(x, kind, qC, alpha0):
    """Every test compares differences of f values, so f + C must give the same trials, the
    same step and the same verdict as f when f + C is computed exactly. The sum form
    φ(0) + c₁αφ'(0) of the Armijo test rounds at the scale of |C| and failed this (about one
    sample in five on the grid 2⁻⁴), as did the |f|-relative tolerances of exact_quadratic."""
    q, C = qC
    f0, g, h = _log_rosen(0.0, q)
    fC, _, _ = _log_rosen(C, q)
    p = -g(x)
    assume(np.linalg.norm(p) > 1e-6)
    if kind == "exact_quadratic":
        assume(float(p @ h(x) @ p) > 0.0)
    r0 = search(kind, f0, g, x, p, hess=h, alpha0=alpha0)
    rC = search(kind, fC, g, x, p, hess=h, alpha0=alpha0)
    assert rC.success == r0.success and rC.alpha == r0.alpha
    assert [a for a, _ in rC.trials] == [a for a, _ in r0.trials]
    assert np.array_equal([v - C for _, v in rC.trials], [v for _, v in r0.trials], equal_nan=True)
    assert (rC.n_fev, rC.n_gev, rC.n_hev) == (r0.n_fev, r0.n_gev, r0.n_hev)


@pytest.mark.parametrize("C", [0.0, 2.0**42, -(2.0**42)])
def test_armijo_compares_differences_not_sums(C):
    # φ(α) = C + α² - α along p = 1 from 0: φ(1) = φ(0), so α = 1 fails the Armijo test
    # (0 > -c₁ = c₁αφ'(0)), and α = ½ passes it. With C = 2⁴² (spacing 2⁻¹⁰ > c₁) the sum
    # φ(0) + c₁αφ'(0) rounds to φ(0) and the sum form accepted α = 1.
    f = lambda x: C + float(x[0] ** 2 - x[0])  # noqa: E731
    g = lambda x: np.array([2.0 * x[0] - 1.0])  # noqa: E731
    r = search("backtracking", f, g, np.zeros(1), np.ones(1))
    assert r.success and r.alpha == 0.5 and r.trials == [(1.0, C), (0.5, C - 0.25)]


COMMON_INFO_KEYS = {
    "alpha", "phi", "dphi", "phi0", "dphi0", "c1", "c2", "direction", "phase", "interval",
    "accepted", "conditions",
}  # fmt: skip
KIND_INFO_KEYS = {
    "backtracking": ({"rho"}, {"armijo"}),
    "strong_wolfe": (
        {"alpha_lo", "alpha_hi", "interp"},
        {"armijo", "curvature", "strong_curvature"},
    ),
    "weak_wolfe": (set(), {"armijo", "curvature"}),
    "goldstein": (set(), {"armijo", "goldstein_lower"}),
    "exact_quadratic": ({"pHp", "model_phi"}, {"armijo", "decrease", "strong_curvature"}),
}


@pytest.mark.parametrize("kind", KINDS)
def test_info_keys_match_the_documented_set(kind):
    """The module docstring's "Info keys:" section is the contract for the web port."""
    extra, cond = KIND_INFO_KEYS[kind]
    for res in (numopt.run(kind, ROSEN), numopt.run(kind, ROSEN, direction="newton")):
        for s in res.trace:
            assert set(s.info) == COMMON_INFO_KEYS | extra
            assert set(s.info["conditions"]) == cond
    doc = ls.__doc__ or ""
    for key in COMMON_INFO_KEYS | extra | cond:
        assert key in doc


# --------------------------------------------------------------------------------------
# Rounding guard: steps that do not move x (audit: backtracking reported alpha = 0)
# --------------------------------------------------------------------------------------


def _ramp_then_nan(x_end: float) -> tuple[Callable, Callable]:
    """φ(α) = f(x_end + α) = -(x_end + α) for α ≤ 0, NaN beyond: no step α > 0 is acceptable."""

    def f(x: np.ndarray) -> float:
        return float(-x[0]) if x[0] <= x_end else math.nan

    def g(x: np.ndarray) -> np.ndarray:
        return np.array([-1.0])

    return f, g


@pytest.mark.parametrize("kw", [{"max_iter": 2000}, {"alpha0": 1e-310}])
def test_backtracking_fails_when_the_step_underflows(kw):
    f, g = _ramp_then_nan(0.0)
    r = search("backtracking", f, g, np.zeros(1), np.ones(1), **kw)
    assert not r.success and r.alpha == 0.0 and r.f_new == 0.0
    assert "underflowed" in r.message
    # Every evaluated trial was a positive step (alpha = 0 itself is never a trial).
    assert r.trials and all(a > 0.0 and math.isnan(phi) for a, phi in r.trials)
    assert r.trials[-1][0] == TINY and r.n_fev == len(r.trials) + 1


@pytest.mark.parametrize("kind", SEARCH_KINDS)
def test_steps_lost_to_rounding_are_never_accepted(kind):
    # x = -1e16 (float spacing 2): x + αp == x for α ≤ 1, and every step that moves x lands
    # where f is NaN, so no acceptable step exists. Before the guard, backtracking and
    # Goldstein accepted alpha = 1 (x unchanged, Armijo true only by rounding).
    x0 = -1e16
    f, g = _ramp_then_nan(x0)
    x, p = np.array([x0]), np.ones(1)
    r = search(kind, f, g, x, p, alpha0=4.0)
    assert not r.success and r.alpha == 0.0 and r.f_new == -x0
    if kind == "backtracking":
        # α = 4 and 2 move x (into the NaN region); -1e16 + 1 rounds to -1e16 (ties to even).
        assert "underflowed" in r.message and [a for a, _ in r.trials] == [4.0, 2.0]
    if kind in ("weak_wolfe", "goldstein"):  # non-moving steps are not even evaluated
        assert all(x0 + a * 1.0 != x0 for a, _ in r.trials)


@pytest.mark.parametrize("kind", ["weak_wolfe", "goldstein"])
def test_bracket_searches_expand_past_steps_lost_to_rounding(kind):
    # φ(α) = (x0 + αp - c)² with x0 = 1e16 (float spacing 2), c = x0 + 2²⁰, p = 2²¹: α* = ½.
    # α₀ = 2⁻²³ gives αp = ¼, which rounds away; the search must treat it as too short.
    x0, c = 1e16, 1e16 + 2.0**20
    f = lambda x: float((x[0] - c) ** 2)  # noqa: E731
    g = lambda x: np.array([2.0 * (x[0] - c)])  # noqa: E731
    x, p = np.array([x0]), np.array([2.0**21])
    r = search(kind, f, g, x, p, alpha0=2.0**-23)
    assert r.success, r.message
    assert holds(kind, f, g, x, p, r.alpha, c1=c1_for(kind))
    assert all(x0 + a * p[0] != x0 for a, _ in r.trials)
    assert r.n_fev == len(r.trials) + 1


FROZEN = Problem(
    "frozen", "p too short to move x", "", lambda x: -1e-30 * x, 1, (),
    grad=lambda x: -1e-30, hess=lambda x: 0.0, x0=1e16,
)  # fmt: skip


@pytest.mark.parametrize("kind", SEARCH_KINDS)
def test_demo_reports_a_direction_too_short_to_move_x(kind):
    # p = 1e-30 at x0 = 1e16: x0 + αp == x0 for every α ≤ alpha_max.
    res = numopt.run(kind, FROZEN)
    assert_valid_result(res, max_iter=50)
    assert not res.converged and res.x == 1e16 and res.fun == -1e-30 * 1e16
    assert not any(s.info["accepted"] for s in res.trace)
    assert res.n_fev == len(res.trace)
    if kind == "backtracking":
        assert "underflowed" in res.message and res.n_iter == 0
    else:
        # The expanding searches do not evaluate non-moving steps: α doubles to alpha_max.
        # (Before the guard, strong Wolfe evaluated them; φ(2) = φ(1) sent it to a zoom of
        # non-moving steps that spent all 50 trials.)
        assert "even at alpha_max" in res.message and res.n_iter == 0
        r = numopt.run(kind, FROZEN, alpha0=100.0, alpha_max=100.0)
        assert not r.converged and "even at alpha_max" in r.message and r.n_iter == 0


# --------------------------------------------------------------------------------------
# More failure paths (audit: untested branches)
# --------------------------------------------------------------------------------------


def _nan_grad_beyond(cut: float) -> tuple[Callable, Callable]:
    """f = (x₀ - 3)² + x₁² everywhere, but ∇f is NaN for x₀ ≥ cut."""

    def f(x: np.ndarray) -> float:
        return float((x[0] - 3.0) ** 2 + x[1] ** 2)

    def g(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (x[0] - 3.0), 2.0 * x[1]]) if x[0] < cut else np.full(2, math.nan)

    return f, g


@pytest.mark.parametrize(
    ("kind", "alpha0", "trials"),
    [
        ("strong_wolfe", 0.5, [(0.5, 0.0)]),  # bracketing phase (Alg. 3.5)
        ("strong_wolfe", 1.0, [(1.0, 9.0), (0.5, 0.0)]),  # zoom (quadratic interpolation)
        ("weak_wolfe", 0.5, [(0.5, 0.0)]),
    ],
)
def test_non_finite_gradient_stops_the_wolfe_searches(kind, alpha0, trials):
    # Along p = (6, 0) from 0: φ(α) = (6α - 3)², α* = ½ lands at x₀ = 3 where ∇f is NaN.
    f, g = _nan_grad_beyond(2.0)
    r = search(kind, f, g, np.zeros(2), np.array([6.0, 0.0]), alpha0=alpha0)
    assert not r.success and r.alpha == 0.0 and r.f_new == 9.0
    assert r.message == "non-finite gradient at α = 0.5"
    assert r.trials == trials and r.n_gev == 2 and r.n_fev == len(trials) + 1


def test_strong_wolfe_max_iter_in_the_bracketing_phase():
    r = search("strong_wolfe", LINEAR.f, _linear_grad, np.zeros(2), np.ones(2), max_iter=3)
    assert not r.success and "bracketing phase" in r.message
    assert [a for a, _ in r.trials] == [1.0, 2.0, 4.0]


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("backtracking", "underflowed"),
        ("strong_wolfe", "zoom interval"),
        ("weak_wolfe", "bracket"),
        ("goldstein", "bracket"),
    ],
)
def test_bracket_collapse_is_reported(kind, match):
    # No α > 0 is acceptable (f is NaN for every step), so each search halves α down to the
    # smallest subnormal and must then stop with a failure, never at alpha = 0.
    f, g = _ramp_then_nan(0.0)
    r = search(kind, f, g, np.zeros(1), np.ones(1), alpha0=1e-300, max_iter=200)
    assert not r.success and r.alpha == 0.0 and match in r.message
    assert "machine precision" in r.message or kind == "backtracking"
    assert all(a > 0.0 for a, _ in r.trials) and len(r.trials) < 200
    assert min(a for a, _ in r.trials) == TINY


def test_exact_quadratic_non_finite_step():
    f, g = _nan_beyond(1.5)  # the model step α = ½ lands at x₀ = 3, where f is NaN
    r = search("exact_quadratic", f, g, np.zeros(2), np.array([6.0, 0.0]), hess=2 * np.eye(2))
    assert not r.success and "not finite" in r.message and r.f_new == 9.0
    assert len(r.trials) == 1 and r.trials[0][0] == 0.5 and math.isnan(r.trials[0][1])
    # The internal guard (both callers check pᵀHp first) reports, never divides.
    line = ls._Line(f, g, np.zeros(2), np.array([6.0, 0.0]))
    out = ls._exact_quadratic(line, 9.0, -36.0, 0.0)
    assert not out.success and "no minimizer" in out.message and not out.trials
    assert line.n_fev == 0


@pytest.mark.parametrize(
    ("x", "p", "g0", "match"),
    [
        (np.zeros(2), np.ones(3), None, "same shape"),
        (np.zeros(2), -np.ones(2), np.ones(3), "g0 must have shape"),
    ],
)
def test_search_rejects_shape_mismatch(x, p, g0, match):
    with pytest.raises(ValueError, match=match):
        search("backtracking", rosen, rosen_grad, x, p, g0=g0)


def test_demo_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        numopt.run("backtracking", ROSEN, direction="foo")


def test_demo_reports_non_finite_newton_direction():
    # ∇²f = diag(1e-320, 2) is not singular for LAPACK, but 1 / 1e-320 overflows: p = (-inf, 0).
    prob = Problem(
        "tiny", "tiny", "", lambda x: float(x[0] + x[1] ** 2), 2, (),
        grad=lambda x: np.array([1.0, 2.0 * x[1]]), hess=lambda x: np.diag([1e-320, 2.0]),
        x0=[0.0, 0.0],
    )  # fmt: skip
    for kind in KINDS:
        _demo_failure(kind, prob, "not finite", direction="newton")


# --------------------------------------------------------------------------------------
# Slider ranges (audit: c1 ≥ c2 or alpha0 > alpha_max raised from the web app)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", numopt.list_methods("line_search"), ids=lambda s: s.id)
def test_every_slider_corner_is_a_valid_run(spec):
    axes = [
        [(prm.name, c) for c in prm.choices]
        if prm.kind == "choice"
        else [(prm.name, prm.min), (prm.name, prm.max)]
        for prm in spec.params
    ]
    for combo in itertools.product(*axes):
        params = dict(combo)
        res = numopt.run(spec.id, ROSEN, **params)  # must not raise
        assert_valid_result(res, max_iter=params.get("max_iter", 50))
        if res.converged and spec.id != "exact_quadratic":
            p = np.asarray(res.extra["direction"])
            x0 = np.asarray(ROSEN.x0, dtype=float)
            c1, c2 = params.get("c1", c1_for(spec.id)), params.get("c2", 0.9)
            assert holds(spec.id, rosen, rosen_grad, x0, p, res.extra["alpha"], c1=c1, c2=c2)
