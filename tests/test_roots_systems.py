"""Tests of the nonlinear-system solvers (newton_system, broyden)."""

from __future__ import annotations

import math
import sys
from itertools import pairwise

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import optimize

import numopt
from numopt import problems
from numopt.core.types import Problem, Step
from numopt.roots import systems as systems_mod

EPS = sys.float_info.epsilon
SYSTEMS = [p.id for p in problems.list_problems("systems")]
#: (method, problem, params) that converge from the problem's default start.
CONVERGING = [
    ("newton_system", "circle_line", {}),
    ("newton_system", "rosenbrock_system", {}),
    ("newton_system", "freudenstein_roth", {}),
    ("newton_system", "trig_system", {}),
    ("newton_system", "intersecting_circles", {}),
    ("newton_system", "circle_line", {"damping": True}),
    ("newton_system", "trig_system", {"damping": True}),
    ("newton_system", "intersecting_circles", {"damping": True}),
    ("broyden", "circle_line", {}),
    ("broyden", "trig_system", {}),
    ("broyden", "intersecting_circles", {}),
    ("broyden", "circle_line", {"jacobian0": "finite_difference"}),
    ("broyden", "trig_system", {"jacobian0": "finite_difference"}),
]


def _fun(s: Step) -> float:
    assert s.fun is not None
    return s.fun


def _counting(p: Problem) -> tuple[Problem, dict[str, int]]:
    n = {"f": 0, "j": 0}

    def F(x):
        n["f"] += 1
        return p.f(x)

    def J(x):
        n["j"] += 1
        return p.jac(x)  # type: ignore[misc]

    return Problem(p.id, p.name, p.latex, F, 2, p.domain, jac=J, x0=p.x0, roots=p.roots), n


@pytest.mark.parametrize(("method", "pid", "params"), CONVERGING)
def test_converges_to_a_listed_root(method, pid, params):
    p, n = _counting(problems.get(pid))
    res = numopt.run(method, p, **params)
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    assert any(np.allclose(res.x, r, rtol=1e-9, atol=1e-9) for r in p.roots), res.x
    assert (res.n_fev, res.n_gev) == (n["f"], n["j"])
    # NOTE: the solver uses an overflow-safe scaled 2-norm; it agrees with numpy's to ~ε.
    assert math.isclose(_fun(res.trace[-1]), float(np.linalg.norm(p.f(res.x))), rel_tol=4 * EPS)
    assert res.fun == _fun(res.trace[-1])
    for s in res.trace:
        assert s.fun == s.info["residual_norm"]
        assert math.isclose(_fun(s), float(np.linalg.norm(s.info["residual"])), rel_tol=4 * EPS)
    for prev, cur in pairwise(res.trace):
        np.testing.assert_array_equal(cur.info["step"], cur.x - prev.x)


@pytest.mark.parametrize("pid", SYSTEMS)
def test_newton_root_agrees_with_scipy_hybr(pid):
    """Oracle: MINPACK hybrd (scipy.optimize.root, method='hybr') started at our root."""
    p = problems.get(pid)
    res = numopt.run("newton_system", p)
    if not res.converged:
        return
    ref = optimize.root(p.f, res.x, jac=p.jac, method="hybr", options={"xtol": 1e-14})
    assert ref.success
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-10, atol=1e-12)


def test_newton_converges_quadratically():
    p = problems.get("circle_line")
    root = np.array(p.roots[1])
    res = numopt.run("newton_system", p, ftol=0.0, xtol=1e-15)
    e = [float(np.linalg.norm(s.x - root)) for s in res.trace]
    e = [v for v in e if v > 1e-13]
    ratios = [b / a**2 for a, b in pairwise(e)]
    assert max(ratios[1:]) < 2.0  # bounded e_{k+1}/e_k²


def test_newton_steps_solve_the_linearization():
    p = problems.get("rosenbrock_system")
    for prev, cur in pairwise(numopt.run("newton_system", p).trace):
        J = np.asarray(p.jac(prev.x))
        np.testing.assert_array_equal(cur.info["jacobian"], J)
        np.testing.assert_allclose(J @ cur.info["newton_step"], -p.f(prev.x), rtol=1e-12, atol=1e-9)
        assert cur.info["alpha"] == 1.0 and cur.info["trials"] == []


def test_damping_keeps_newton_in_the_domain():
    """From (−1, 1.25) the full Newton step leaves the plotting window and lands on a
    2π-shifted copy of the root; the damped method stays and finds (π/6, π/6)."""
    p = problems.get("trig_system")
    full = numopt.run("newton_system", p, x0=[-1.0, 1.25])
    damped = numopt.run("newton_system", p, x0=[-1.0, 1.25], damping=True)
    assert full.converged and np.max(np.abs(full.x)) > 5
    assert damped.converged
    np.testing.assert_allclose(damped.x, [math.pi / 6, math.pi / 6], atol=1e-12)
    norms = [_fun(s) for s in damped.trace]
    assert all(b < a for a, b in pairwise(norms))  # Armijo ⇒ monotone ‖F‖
    for s in damped.trace[1:]:
        alphas = [t[0] for t in s.info["trials"]]
        assert alphas == [2.0**-i for i in range(len(alphas))]
        assert s.info["alpha"] == alphas[-1]


def test_armijo_condition_holds_for_accepted_steps():
    p = problems.get("trig_system")
    res = numopt.run("newton_system", p, x0=[-1.0, 1.25], damping=True)
    for prev, cur in pairwise(res.trace):
        phi0 = 0.5 * _fun(prev) ** 2
        alpha = cur.info["alpha"]
        assert 0.5 * _fun(cur) ** 2 <= (1 - 2e-4 * alpha) * phi0 * (1 + 4 * EPS)
        last_alpha, last_phi = cur.info["trials"][-1]
        assert last_alpha == alpha
        phi = 0.5 * float(cur.info["residual"] @ cur.info["residual"])
        assert math.isclose(last_phi, phi, rel_tol=8 * EPS)  # scaled vs plain norm


def test_damped_newton_reports_the_freudenstein_trap():
    res = numopt.run("newton_system", problems.get("freudenstein_roth"), damping=True)
    assert not res.converged and "line search failed" in res.message
    assert res.x[1] == pytest.approx(-0.8968, abs=1e-3)  # near the singular line
    assert_valid_result(res)


def test_singular_jacobian_is_reported():
    # On the line through both centres det J = 4(1 − x − 2y) = 0.
    for method in ("newton_system", "broyden"):
        res = numopt.run(method, problems.get("intersecting_circles"), x0=[0.0, 0.5])
        assert not res.converged and "singular" in res.message
        assert res.n_iter == 0
        assert_valid_result(res)


def test_max_iter_and_slow_damped_newton():
    res = numopt.run("newton_system", problems.get("rosenbrock_system"), damping=True)
    assert not res.converged and "max_iter" in res.message  # crawls along the valley
    assert_valid_result(res, max_iter=100)
    for method in ("newton_system", "broyden"):
        res = numopt.run(method, problems.get("circle_line"), max_iter=2)
        assert not res.converged and "max_iter" in res.message
        assert_valid_result(res, max_iter=2)


def test_broyden_failure_far_from_the_root_is_honest():
    res = numopt.run("broyden", problems.get("rosenbrock_system"))
    assert not res.converged
    assert_valid_result(res)


def _broyden_b_form(F, J0, x0, n_steps):
    """Reference: good Broyden in direct form — solve B_k p = −F each step and update
    B_{k+1} = B_k + (y − B_k s)sᵀ/(sᵀs) (N&W §11.1). Equivalent to the inverse update."""
    x = np.array(x0, dtype=float)
    B = np.array(J0, dtype=float)
    Fx = F(x)
    xs = [x.copy()]
    for _ in range(n_steps):
        s = np.linalg.solve(B, -Fx)
        x = x + s
        F_new = F(x)
        y = F_new - Fx
        B = B + np.outer(y - B @ s, s) / (s @ s)
        Fx = F_new
        xs.append(x.copy())
    return xs


@pytest.mark.parametrize("pid", ["circle_line", "trig_system", "intersecting_circles"])
def test_broyden_inverse_update_equals_direct_update(pid):
    p = problems.get(pid)
    res = numopt.run("broyden", p, ftol=1e-8)
    ref = _broyden_b_form(p.f, p.jac(np.array(p.x0)), p.x0, res.n_iter)
    # NOTE: the two forms round differently; agreement degrades as cond(B) grows.
    np.testing.assert_allclose([s.x for s in res.trace], ref, rtol=1e-9, atol=1e-11)


def test_broyden_secant_equation_and_counts():
    p = problems.get("intersecting_circles")
    res = numopt.run("broyden", p)
    assert res.n_gev == 1 and res.n_fev == res.n_iter + 1
    for cur, nxt in pairwise(res.trace[1:]):
        s, y = np.array(cur.info["secant"]["s"]), np.array(cur.info["secant"]["y"])
        B_next = np.array(nxt.info["jacobian"])
        np.testing.assert_allclose(B_next @ s, y, rtol=1e-10, atol=1e-12)
    fd = numopt.run("broyden", p, jacobian0="finite_difference")
    assert fd.n_gev == 0 and fd.n_fev == 1 + (2 * 2 + 1) + fd.n_iter
    assert fd.extra["jacobian"] == "finite_difference"


def test_broyden_root_agrees_with_scipy_broyden1():
    p = problems.get("circle_line")
    res = numopt.run("broyden", p)
    ref = optimize.root(p.f, [1.5, 0.5], method="broyden1", options={"fatol": 1e-12})
    assert ref.success
    np.testing.assert_allclose(res.x, ref.x, rtol=1e-9, atol=1e-10)


def test_bare_callable_uses_finite_differences():
    def F(v):
        return np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[1] - v[0] + 1.0])

    res = numopt.run("newton_system", F, x0=[2.0, 2.0])
    assert res.converged and res.extra["jacobian"] == "finite_difference"
    np.testing.assert_allclose(res.x, problems.get("circle_line").roots[1], atol=1e-10)
    assert res.n_gev == 0 and res.n_fev == 1 + res.n_iter * (1 + 5)


def test_invalid_input():
    with pytest.raises(ValueError, match="square"):
        numopt.run("newton_system", lambda v: np.array([v[0], v[1], v[0] * v[1]]), x0=[1.0, 2.0])
    with pytest.raises(ValueError, match="jacobian0"):
        numopt.run("broyden", problems.get("circle_line"), jacobian0="identity")
    with pytest.raises(ValueError):
        numopt.run("newton_system", problems.get("circle_line"), x0=[1.0, 2.0, 3.0])


def test_non_finite_residual():
    def F(v):
        return np.array([math.nan if v[0] < 0 else v[0] - 1.0, v[1]])

    for method in ("newton_system", "broyden"):
        res = numopt.run(method, F, x0=[-1.0, 0.0])
        assert not res.converged and "not finite" in res.message
        res = numopt.run(
            method,
            lambda v: (
                np.array([v[0] - 3.0, math.sqrt(v[1])]) if v[1] >= 0 else np.array([math.inf, 0.0])
            ),
            x0=[1.0, 1.0],
        )
        assert not res.converged
        assert_valid_result(res)


@settings(max_examples=1000, deadline=None)
@given(
    st.lists(st.floats(-5, 5), min_size=4, max_size=4),
    st.lists(st.floats(-5, 5), min_size=2, max_size=2),
    st.sampled_from(["newton_system", "broyden"]),
)
def test_property_linear_systems_are_solved_in_one_step(a, b, method):
    """F(x) = Ax − b with the exact Jacobian A: Newton and Broyden (B₀ = A) are exact
    after one step, up to cond(A)·ε."""
    A = np.array(a).reshape(2, 2)
    bb = np.array(b)
    # The property needs a representable solution: cond(A) ≤ 1e6 and max|Aᵢⱼ| > 1e-6 give
    # ‖A⁻¹‖ ≲ 1e12 (subnormal matrices would put the root beyond the float range).
    assume(np.abs(A).max() > 1e-6 and np.linalg.cond(A) <= 1e6)
    lin = Problem("lin", "lin", "Ax-b", lambda v: A @ v - bb, 2, (), jac=lambda v: A)
    res = numopt.run(method, lin, x0=[0.3, -0.7])
    ref = np.linalg.solve(A, bb)
    assert res.converged
    if res.n_iter == 0:  # ‖F(x₀)‖ is already ≤ ftol (tiny A and b)
        assert res.fun is not None and res.fun <= 1e-10
        return
    bound = 16 * np.linalg.cond(A) * EPS * (1 + np.abs(ref).max() + 1.0)
    np.testing.assert_allclose(res.trace[1].x, ref, rtol=0, atol=bound)
    assert res.n_iter <= 2


@settings(max_examples=1000, deadline=None)
@given(st.floats(-3, 3), st.floats(-3, 3), st.booleans())
def test_property_circle_line_converges_only_to_its_two_roots(x, y, damping):
    p = problems.get("circle_line")
    res = numopt.run("newton_system", p, x0=[x, y], damping=damping)
    assert_valid_result(res, max_iter=100)
    if res.converged:
        assert any(np.allclose(res.x, r, atol=1e-9) for r in p.roots)
    else:
        assert res.message


@pytest.mark.parametrize(("method", "pid", "params"), systems_mod.FIXTURE_CASES)
def test_fixture_cases_run_and_serialize(method, pid, params):
    """The parity fixtures exported for the web app: valid, JSON-clean, < 300 steps."""
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300
    assert numopt.get_method(method).family == "systems"


def test_norms_do_not_overflow_for_huge_roots():
    """A ≈ 1e-154·I puts the root near 1e154, where ‖x‖² overflows in plain arithmetic."""
    A = np.eye(2) * 1e-154
    b = np.array([0.0, 3.0])
    lin = Problem("lin", "lin", "Ax-b", lambda v: A @ v - b, 2, (), jac=lambda v: A)
    with np.errstate(over="raise", invalid="raise"):
        res = numopt.run("newton_system", lin, x0=[0.3, -0.7])
    assert res.converged and math.isfinite(_fun(res.trace[1]))
    np.testing.assert_allclose(res.x, [0.0, 3e154], rtol=1e-15)
    assert res.trace[1].step_size == pytest.approx(3e154, rel=1e-15)


# --- regression tests: Broyden stall, exceptions, convergence only at true roots --------


def _trig_root_distance(x: np.ndarray) -> float:
    """Distance to the nearest root of trig_system: (π/6, π/6) or (5π/6, 5π/6) + 2π·ℤ²."""
    best = math.inf
    for base in (math.pi / 6, 5 * math.pi / 6):
        shift = np.round((x - base) / (2 * math.pi))
        best = min(best, float(np.linalg.norm(x - base - 2 * math.pi * shift)))
    return best


def _root_distance(pid: str, x: np.ndarray) -> float:
    if pid == "trig_system":
        return _trig_root_distance(x)
    return min(float(np.linalg.norm(x - np.array(r))) for r in problems.get(pid).roots)


@pytest.mark.parametrize("jacobian0", ["exact", "finite_difference"])
def test_broyden_zero_step_is_a_stall_not_convergence(jacobian0):
    """F(x) = ((x₁ − 2⁶⁰) + 1, x₂) at x₀ = (2⁶⁰, 0): F(x₀) = (1, 0) exactly and J = I, so the
    step is p = (−1, 0), half the spacing of floats below 2⁶⁰, and x₀ + p rounds to x₀ on every
    IEEE platform (round to nearest even). The run must report a stall with ‖F‖ = 1, not
    convergence (a zero step used to pass the step test; the audit case was circle_line from
    (0.6, −0.6), where the zero step at x ≈ −9.6e10 depends on the CPU's rounding)."""
    big = 2.0**60
    p = Problem(
        id="far_line",
        name="far line",
        latex="",
        f=lambda v: np.array([(v[0] - big) + 1.0, v[1]]),
        jac=lambda v: np.eye(2),
        dim=2,
        domain=((0.0, 2.0 * big), (-1.0, 1.0)),
        x0=(big, 0.0),
    )
    res = numopt.run("broyden", p, jacobian0=jacobian0)
    assert not res.converged and "stalled" in res.message, res.message
    assert res.n_iter == 1 and res.trace[-1].step_size == 0.0
    assert np.array_equal(res.x, [big, 0.0]) and res.fun == 1.0
    assert_valid_result(res)


@pytest.mark.parametrize("x0", [[-0.55, 0.35], [1.475, -0.55], [3.5, 0.575]])
def test_broyden_far_from_the_domain_converges_only_to_true_roots(x0):
    """From these starts Broyden wanders to ‖x‖ ≈ 1e8–1e9. A converged x must be a root of
    the 2π-periodic system to the documented relative accuracy xtol·(1 + ‖x‖) (MINPACK's
    convention); ‖F‖ there is about ‖J‖ times that distance."""
    res = numopt.run("broyden", problems.get("trig_system"), x0=x0)
    assert_valid_result(res)
    if res.converged:
        x = np.asarray(res.x)
        # NOTE: reducing x mod 2π in floats costs ≈ ε‖x‖ (≈ 2e-7 at 1e9).
        assert _trig_root_distance(x) <= 1e-12 * (1 + np.linalg.norm(x)) + 4 * EPS * np.linalg.norm(
            x
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(SYSTEMS),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.sampled_from(
        [
            ("newton_system", {}),
            ("newton_system", {"damping": True}),
            ("broyden", {}),
            ("broyden", {"jacobian0": "finite_difference"}),
        ]
    ),
)
def test_property_converged_only_at_true_roots(pid, u, v, method_params):
    """From any start in the plotting window: a valid Result, and converged ⇒ x near a true
    root (the root lists are complete; trig_system is 2π-periodic)."""
    method, params = method_params
    p = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    x0 = [x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)]
    res = numopt.run(method, p, x0=x0, **params)
    assert_valid_result(res, max_iter=100)
    if res.converged:
        x = np.asarray(res.x)
        assert _root_distance(pid, x) <= 1e-6 * (1 + np.linalg.norm(x)), (x0, res.message)


@pytest.mark.parametrize("method", ["newton_system", "broyden"])
def test_exceptions_in_F_and_J_become_non_finite(method):
    def F(v):
        return np.array([math.exp(v[0]) - 10.0, v[1]])  # math.exp raises beyond 709.78

    def J(v):
        return np.array([[math.exp(v[0]), 0.0], [0.0, 1.0]])

    prob = Problem("expsys", "expsys", "F", F, 2, (), jac=J, x0=(-10.0, 0.0))
    res = numopt.run(method, prob)  # the first Newton step jumps to x ≈ 2.2e5
    assert not res.converged and "non-finite" in res.message
    assert_valid_result(res)
    bad_jac = Problem(
        "bad", "bad", "F", F, 2, (), jac=lambda v: np.log(-1.0 - np.abs(v)), x0=(0.0, 0.0)
    )
    with np.errstate(invalid="ignore"):
        res = numopt.run(method, bad_jac)
    assert not res.converged and "non-finite" in res.message
    with pytest.raises(TypeError):  # a bug in F is not swallowed
        numopt.run(method, lambda v: v + "1", x0=[1.0, 2.0])
