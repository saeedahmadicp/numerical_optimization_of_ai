"""Interior-point methods: oracle comparison with linprog(method="highs"), interiority and
feasibility invariants, divergence detection on unbounded / infeasible LPs."""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.optimize import linprog

import numopt
from numopt import problems
from numopt.core.types import LinearProgram
from numopt.lp import interior_point as ipm_module
from numopt.lp.interior_point import equality_form, independent_rows

FEASIBLE = (
    "wyndor",
    "diet_2d",
    "degenerate_2d",
    "klee_minty_3",
    "transport_small",
    "beale_cycling",
)
METHODS = (
    ("primal_dual_ipm", {}),
    ("affine_scaling", {}),
    ("affine_scaling", {"variant": "short"}),
)


def oracle(lp: LinearProgram):
    sign = 1.0 if lp.sense == "min" else -1.0
    r = linprog(
        sign * np.asarray(lp.c),
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        method="highs",
        # NOTE: presolve off; HiGHS presolve can misreport an unbounded LP as infeasible.
        options={"presolve": False},
    )
    return r, sign


def funs(res) -> list[float]:
    """Step objective values (never None for these methods on feasible problems)."""
    out = [s.fun for s in res.trace]
    assert all(v is not None for v in out)
    return [float(v) for v in out if v is not None]


def scale(lp: LinearProgram) -> float:
    b = [np.abs(v).max() for v in (lp.b_ub, lp.b_eq) if v is not None and v.size]
    return 1.0 + max(b, default=0.0)


def scaling_reference(lp: LinearProgram):
    """The scaling of the module docstring, restated from its formulas (not the implementation):
    Â = R A D, b̂ = R b / β, ĉ = c̃ / γ with R = diag(1/ρ), ρᵢ = maxⱼ≤ₙ |A_ij| (else |bᵢ| or 1),
    D = diag(1 on x,
    ρᵢ on the slack of row i), β = ‖R b‖∞, γ = ‖c̃‖∞ (all rows, none dropped)."""
    A, b, c, n, sign = equality_form(lp)
    rho = np.abs(A[:, :n]).max(axis=1) if A.shape[0] else np.zeros(0)
    tiny = np.finfo(np.float64).tiny
    rho = np.where(rho > 0, rho, np.where(np.abs(b) >= tiny, np.abs(b), 1.0))
    col = np.ones(A.shape[1])
    col[n:] = rho[: A.shape[1] - n]
    A_hat = A * col / rho[:, None]
    beta = float(np.abs(b / rho).max(initial=0.0)) or 1.0
    gamma = float(np.abs(c).max(initial=0.0)) or 1.0
    return A_hat, b / rho / beta, c / gamma, rho, beta, gamma, n, sign


@pytest.mark.parametrize("pid", FEASIBLE)
@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_matches_linprog(pid, method, kw):
    lp = problems.get(pid)
    res = numopt.run(method, lp, **kw)
    assert_valid_result(res)
    assert res.converged, res.message
    ref, _ = oracle(lp)
    value = -ref.fun if lp.sense == "max" else ref.fun
    assert res.fun is not None
    # Stopping test on the scaled LP: ẑᵀŝ ≤ 1e-8·(1+|ĉᵀẑ|), i.e. the objective error is
    # ≲ 1e-8·(βγ + |cᵀx|) in the original units, plus the residual times the dual size;
    # 1e-7 leaves a factor-10 margin.
    *_, beta, gamma, _, _ = scaling_reference(lp)
    np.testing.assert_allclose(res.fun, value, rtol=1e-7, atol=1e-7 * beta * gamma)
    # Every library optimum is unique (tests/test_problems_lp.py), so x converges to it.
    # The vertex distance is ~ gap / (smallest nonzero reduced cost): 1e-6 relative to ‖b‖.
    np.testing.assert_allclose(res.x, lp.optimum, rtol=0, atol=1e-6 * scale(lp) + 1e-6)


@pytest.mark.parametrize("pid", FEASIBLE)
def test_ipm_iterates_stay_interior_and_converge(pid):
    res = numopt.run("primal_dual_ipm", problems.get(pid))
    for s in res.trace:
        assert np.all(np.asarray(s.info["x"]) > 0)
        assert s.info["mu"] > 0
    first, last = res.trace[0].info, res.trace[-1].info
    assert last["mu"] < 1e-6 * first["mu"]
    assert last["primal_residual"] <= 1e-7 * scale(problems.get(pid))
    assert res.n_iter <= 30  # Mehrotra typically needs 5–15 iterations


@pytest.mark.parametrize("pid", FEASIBLE)
@pytest.mark.parametrize("variant", ("long", "short"))
def test_affine_scaling_descent_and_feasibility(pid, variant):
    lp = problems.get(pid)
    big_m = 1e6
    res = numopt.run("affine_scaling", lp, variant=variant, big_m=big_m)
    A_hat, b_hat, _, rho, beta, gamma, _, sign = scaling_reference(lp)
    r0_hat = b_hat - A_hat @ np.ones(A_hat.shape[1])
    # Augmented objective of the scaled big-M problem: ĉᵀẑ + M t = sign·cᵀx/(βγ) + M t.
    augmented = [
        sign * fun / (beta * gamma) + big_m * s.info["artificial"]
        for s, fun in zip(res.trace, funs(res), strict=True)
    ]
    # Affine scaling is a descent method on the augmented (big-M) objective.
    assert all(b2 <= a + 1e-9 * (1 + abs(a)) for a, b2 in pairwise(augmented))
    for s in res.trace:
        assert np.all(np.asarray(s.info["x"]) > 0) and s.info["artificial"] > 0
        # Â ẑ + r̂₀ t = b̂ on every iterate, so A z − b = −β t (ρ ∘ r̂₀) in original units.
        expected = beta * s.info["artificial"] * np.linalg.norm(rho * r0_hat)
        assert abs(s.info["primal_residual"] - expected) <= 1e-8 * scale(lp)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_unbounded_and_infeasible_are_reported(method, kw):
    unb = numopt.run(method, problems.get("unbounded_2d"), **kw)
    assert_valid_result(unb)
    assert not unb.converged and unb.extra["status"] == "unbounded"
    assert "unbounded" in unb.message
    inf = numopt.run(method, problems.get("infeasible_2d"), **kw)
    assert_valid_result(inf)
    assert not inf.converged and inf.extra["status"] == "infeasible"
    assert "infeasible" in inf.message


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_max_iter_is_reported(method, kw):
    res = numopt.run(method, problems.get("transport_small"), max_iter=2, **kw)
    assert_valid_result(res, max_iter=2)
    assert not res.converged and "max_iter" in res.message and res.n_iter == 2


def test_redundant_and_inconsistent_equalities():
    a_eq = np.array([[1.0, 1.0], [2.0, 2.0]])
    ok = LinearProgram("r", "r", np.array([1.0, 2.0]), A_eq=a_eq, b_eq=np.array([1.0, 2.0]))
    keep, consistent = independent_rows(a_eq, np.array([1.0, 2.0]))
    assert keep == [0] and consistent
    for method, kw in METHODS:
        res = numopt.run(method, ok, **kw)
        assert res.converged
        np.testing.assert_allclose(res.x, [1.0, 0.0], atol=1e-7)
    bad = LinearProgram("r", "r", np.array([1.0, 2.0]), A_eq=a_eq, b_eq=np.array([1.0, 3.0]))
    for method, kw in METHODS:
        res = numopt.run(method, bad, **kw)
        assert_valid_result(res)
        assert not res.converged and "inconsistent" in res.message


def test_info_keys_present():
    res = numopt.run("primal_dual_ipm", problems.get("wyndor"))
    k0, k1 = res.trace[0].info, res.trace[1].info
    assert k0["sigma"] is None and k0["x_affine"] is None
    assert 0 <= k1["sigma"] <= 1 and 0 < k1["alpha_primal"] <= 1 and 0 < k1["alpha_dual"] <= 1
    assert len(k1["x_affine"]) == 2
    aff = numopt.run("affine_scaling", problems.get("wyndor"))
    assert aff.trace[0].info["alpha"] is None and aff.trace[1].info["alpha"] > 0
    assert len(aff.trace[1].info["direction"]) == 2


@pytest.mark.parametrize("pid", ("wyndor", "diet_2d", "transport_small"))
def test_ipm_start_point_is_mehrotra_heuristic(pid):
    """Step 0 equals Nocedal & Wright (14.41)–(14.42) on the scaled data, computed here with
    numpy.linalg, mapped back by x = β x̂."""
    lp = problems.get(pid)
    A, b, c, _, beta, _, n, _ = scaling_reference(lp)
    keep, _ = independent_rows(A, b)
    A, b = A[keep], b[keep]
    gram = A @ A.T
    x = A.T @ np.linalg.solve(gram, b)
    s = c - A.T @ np.linalg.solve(gram, A @ c)
    x_hat = x + max(-1.5 * x.min(), 0.0)
    s_hat = s + max(-1.5 * s.min(), 0.0)
    x0 = x_hat + 0.5 * (x_hat @ s_hat) / s_hat.sum()
    res = numopt.run("primal_dual_ipm", lp)
    # κ(AAᵀ) ≤ 1e3 for these problems: ~13 digits survive; 1e-10 leaves a margin.
    np.testing.assert_allclose(res.trace[0].x, beta * x0[:n], rtol=1e-10, atol=1e-10 * beta)


def test_ipm_duals_match_linprog_marginals():
    """The final λ of min c̃ᵀz equals HiGHS's ∂(min value)/∂b_ub (Wyndor's dual is unique)."""
    lp = problems.get("wyndor")
    res = numopt.run("primal_dual_ipm", lp, tol=1e-10)
    ref, _ = oracle(lp)
    np.testing.assert_allclose(res.extra["lambda"], ref.ineqlin.marginals, rtol=0, atol=1e-7)
    # Hillier & Lieberman §6.2: shadow prices of the max problem are (0, 3/2, 1).
    np.testing.assert_allclose(-res.extra["lambda"], [0.0, 1.5, 1.0], atol=1e-7)


# --------------------------------------------------------------------------------------
# Scale invariance and infeasibility / unboundedness certificates (audit findings)
# --------------------------------------------------------------------------------------


def wyndor_scaled(c_factor: float, row_factor: float, b_factor: float = 1.0) -> LinearProgram:
    lp = problems.get("wyndor")
    assert lp.A_ub is not None and lp.b_ub is not None
    return replace(
        lp,
        c=lp.c * c_factor,
        A_ub=lp.A_ub * row_factor,
        b_ub=lp.b_ub * row_factor * b_factor,
    )


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize(
    ("c_factor", "row_factor"), [(1e-8, 1.0), (1.0, 1e-6), (1e-8, 1e-6), (1e3, 1e4)]
)
def test_stopping_tests_are_scale_invariant(method, kw, c_factor, row_factor):
    """Audit: c×1e-8 with A, b×1e-6 converged with a 0.9% objective error and x = (2.07, 5.90);
    A, b×1e-6 alone made affine_scaling report "unbounded". The scaled LPs are the same LP."""
    ref = numopt.run(method, problems.get("wyndor"), **kw)
    res = numopt.run(method, wyndor_scaled(c_factor, row_factor), **kw)
    assert_valid_result(res)
    assert res.converged, res.message
    # Same scaled problem (up to rounding of the scale factors): same iterates.
    assert res.n_iter == ref.n_iter and res.fun is not None
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.fun, 36.0 * c_factor, rtol=1e-8)


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("b_factor", (1e-6, 1e6))
def test_scaling_b_scales_the_solution(method, kw, b_factor):
    """b → κ b scales x* by κ; the relative accuracy must not depend on κ."""
    res = numopt.run(method, wyndor_scaled(1.0, 1.0, b_factor), **kw)
    assert res.converged, res.message
    np.testing.assert_allclose(res.x, b_factor * np.array([2.0, 6.0]), rtol=1e-7)


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("a", (1e-9, 1e-10, 1e-11))
def test_bounded_lp_with_a_large_optimum_is_not_divergent(method, kw, a):
    """Audit: max x1 + x2 s.t. a·x1 ≤ 1, x2 ≤ 1 (x* = (1/a, 1)) was declared "diverged ...
    likely infeasible / unbounded" by a fixed 1e10 limit on the iterates."""
    lp = LinearProgram(
        "big",
        "big",
        np.array([1.0, 1.0]),
        A_ub=np.array([[a, 0.0], [0.0, 1.0]]),
        b_ub=np.array([1.0, 1.0]),
        sense="max",
    )
    res = numopt.run(method, lp, **kw)
    assert_valid_result(res)
    assert res.converged and res.fun is not None, res.message
    # Normwise relative accuracy: |x1| = 1/a dominates, so x2 is only known to ~tol/a.
    np.testing.assert_allclose(res.fun, 1.0 / a + 1.0, rtol=1e-7)
    np.testing.assert_allclose(res.x[0], 1.0 / a, rtol=1e-7)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_infeasible_lp_with_a_ray_is_infeasible(method, kw):
    """Audit: max x3 s.t. x1 + x2 ≤ 2, x1 + x2 ≥ 4 is infeasible, but its big-M problem is
    unbounded along x3; affine_scaling reported "unbounded". Both methods must say infeasible."""
    lp = LinearProgram(
        "inf_ray",
        "inf_ray",
        np.array([0.0, 0.0, 1.0]),
        A_ub=np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]),
        b_ub=np.array([2.0, -4.0]),
        sense="max",
    )
    res = numopt.run(method, lp, **kw)
    assert_valid_result(res)
    assert not res.converged and res.extra["status"] == "infeasible", res.message
    assert "unbounded" not in res.message.split(";")[0]


def test_big_m_too_small_is_not_called_unbounded():
    """max Σx (20 variables) s.t. Σx ≤ 1: with ĉ = e, each unit of t frees r̂₀ = 20 units of Σx,
    so big_m = 10 < 20 makes the big-M problem unbounded only through t."""
    n = 20
    lp = LinearProgram(
        "m", "m", np.ones(n), A_ub=np.ones((1, n)), b_ub=np.array([1.0]), sense="max"
    )
    small = numopt.run("affine_scaling", lp, big_m=10.0)
    assert_valid_result(small)
    assert not small.converged
    # The LP is feasible (the zero-objective run converges), so only big_m is to blame.
    assert small.extra["status"] == "big_m_too_small"
    assert small.extra["feasibility_check"] == "optimal"
    assert "big_m" in small.message
    ok = numopt.run("affine_scaling", lp)
    assert ok.converged and ok.fun == pytest.approx(1.0, rel=1e-7)


def test_unbounded_needs_a_feasible_point():
    """A ray alone is not unboundedness: the zero-objective feasibility run decides."""
    res = numopt.run("primal_dual_ipm", problems.get("unbounded_2d"))
    assert res.extra["status"] == "unbounded"
    assert res.extra.get("feasibility_check") in (None, "optimal")


@st.composite
def random_lps(draw):
    """Small integer LPs of every status (optimal, infeasible, unbounded)."""
    n = draw(st.integers(2, 6))
    m = draw(st.integers(1, 5))
    m_eq = draw(st.integers(0, 2))
    a = draw(hnp.arrays(np.float64, (m, n), elements=st.integers(-2, 2)))
    b = draw(hnp.arrays(np.float64, (m,), elements=st.integers(-2, 3)))
    a_eq = draw(hnp.arrays(np.float64, (m_eq, n), elements=st.integers(-2, 2)))
    b_eq = draw(hnp.arrays(np.float64, (m_eq,), elements=st.integers(0, 3)))
    c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-3, 3)))
    sense = draw(st.sampled_from(["min", "max"]))
    return LinearProgram(
        "any",
        "any",
        c,
        A_ub=a,
        b_ub=b,
        A_eq=a_eq if m_eq else None,
        b_eq=b_eq if m_eq else None,
        sense=sense,
    )


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_lps(), st.sampled_from(METHODS))
def test_random_lp_status_matches_linprog(lp, method_kw):
    """Audit: 12 of 135 random infeasible LPs were labeled "unbounded" by affine_scaling."""
    method, kw = method_kw
    ref, _ = oracle(lp)
    assume(ref.status in (0, 2, 3))
    res = numopt.run(method, lp, **kw)
    assert_valid_result(res)
    expected = {0: "optimal", 2: "infeasible", 3: "unbounded"}[ref.status]
    assert res.extra["status"] == expected, res.message
    if expected == "optimal":
        assert res.fun is not None
        value = -ref.fun if lp.sense == "max" else ref.fun
        *_, beta, gamma, _, _ = scaling_reference(lp)
        np.testing.assert_allclose(res.fun, value, rtol=1e-6, atol=1e-6 * beta * gamma)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.data(),
    st.sampled_from(METHODS),
    st.integers(-8, 8),
)
def test_random_row_and_cost_scaling_property(data, method_kw, c_exp):
    """Scaling rows by 10^kᵢ and c by 10^k gives the same x and a 10^k times larger value."""
    method, kw = method_kw
    lp = data.draw(bounded_feasible_lps())
    assert lp.A_ub is not None and lp.b_ub is not None
    m = lp.b_ub.size
    exps = data.draw(hnp.arrays(np.int64, (m,), elements=st.integers(-8, 8)))
    rows = 10.0 ** exps.astype(np.float64)
    scaled = replace(lp, c=lp.c * 10.0**c_exp, A_ub=lp.A_ub * rows[:, None], b_ub=lp.b_ub * rows)
    ref = numopt.run(method, lp, **kw)
    res = numopt.run(method, scaled, **kw)
    assert ref.converged and res.converged, (ref.message, res.message)
    assert ref.fun is not None and res.fun is not None
    *_, beta, gamma, _, _ = scaling_reference(lp)
    np.testing.assert_allclose(res.fun / 10.0**c_exp, ref.fun, rtol=1e-6, atol=1e-6 * beta * gamma)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        numopt.run("primal_dual_ipm", problems.get("wyndor"), eta=1.0)
    with pytest.raises(ValueError):
        numopt.run("affine_scaling", problems.get("wyndor"), variant="medium")
    with pytest.raises(ValueError):
        numopt.run("affine_scaling", problems.get("wyndor"), beta=1.5)


# --------------------------------------------------------------------------------------
# Hypothesis: random bounded feasible LPs against linprog
# --------------------------------------------------------------------------------------


@st.composite
def bounded_feasible_lps(draw):
    """Random LPs that are feasible (a known interior point) and bounded (a box row)."""
    n = draw(st.integers(1, 4))
    m = draw(st.integers(1, 4))
    a = draw(hnp.arrays(np.float64, (m, n), elements=st.integers(-4, 6)))
    x_feas = draw(hnp.arrays(np.float64, (n,), elements=st.floats(0.1, 3.0)))
    slack = draw(hnp.arrays(np.float64, (m,), elements=st.floats(0.0, 4.0)))
    b = a @ x_feas + slack
    a = np.vstack(
        [a, np.ones((1, n))]
    )  # Σx ≤ 10 + Σx_feas keeps the LP bounded and x_feas feasible
    b = np.append(b, 10.0 + float(np.sum(x_feas)))
    c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-5, 5)))
    sense = draw(st.sampled_from(["min", "max"]))
    return LinearProgram("rand", "rand", c, A_ub=a, b_ub=b, sense=sense)


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(bounded_feasible_lps(), st.sampled_from(METHODS))
def test_random_bounded_lps_match_linprog(lp, method_kw):
    method, kw = method_kw
    res = numopt.run(method, lp, **kw)
    assert_valid_result(res)
    assert res.converged, res.message
    ref, _ = oracle(lp)
    value = -ref.fun if lp.sense == "max" else ref.fun
    assert res.fun is not None
    # Objective error ≤ gap + dual-norm × residual; data entries ≤ 10, values ≤ 50 here.
    np.testing.assert_allclose(res.fun, value, rtol=1e-6, atol=1e-6)
    x = np.asarray(res.x)
    assert np.all(lp.A_ub @ x <= lp.b_ub + 1e-6)


def documented_info_keys(module) -> set[str]:
    """Names listed in the module docstring's "Info keys:" section."""
    doc = module.__doc__ or ""
    section = doc.split("Info keys", 1)[1]
    keys: set[str] = set()
    for line in section.splitlines():
        m = re.match(r"^\s{4}([a-z_]+(?:, [a-z_]+)*)(?::| —)", line)
        if m:
            keys.update(k.strip() for k in m.group(1).split(","))
    return keys


def test_every_info_key_is_documented():
    documented = documented_info_keys(ipm_module)
    for method, kw in METHODS:
        for pid in ("wyndor", "unbounded_2d", "infeasible_2d"):
            res = numopt.run(method, problems.get(pid), **kw)
            for s in res.trace:
                assert set(s.info) <= documented, (method, set(s.info) - documented)
