"""Tests for the constrained problem library (numopt.problems.constrained).

Derivative oracle: the 5-point central difference
    f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h),  h = ε^{1/5} ≈ 7.4·10⁻⁴,
with truncation error h⁴|f⁽⁵⁾|/30 ≈ 10⁻¹⁴|f⁽⁵⁾| and rounding error ≈ ε|f|/h ≈ 3·10⁻¹³|f|.
The domains have |x| ≤ 52 (hs21), so representing x ± h costs ≤ ε·52/h ≈ 1.6·10⁻¹¹ relative;
the tolerance 1e-7 (relative to the local scale) leaves a margin of > 10³.

KKT oracle: every listed minimizer must satisfy, with the listed multipliers,
    ∇f(x*) + Σ λ_i ∇c_i(x*) = 0,  c_E(x*) = 0,  c_I(x*) ≤ 0,  λ_I ≥ 0,  λ_i c_i(x*) = 0,
to 1e-12 relative to the scale of the terms (the values are closed forms, or KKT-Newton
refinements whose residuals are ≤ 1e-13), and the second-order sufficient condition: ∇²_xx L
is positive definite on the null space of the gradients of the equalities and of the
inequalities with λ_i > 0 (a superset of the critical cone, so the test is sufficient).
SciPy's SLSQP and trust-constr are independent oracles for the minimizers and multipliers.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import NonlinearConstraint, minimize
from scipy.spatial import ConvexHull

from numopt import problems
from numopt.core.types import Problem

REQUIRED = (
    "quadratic_disk",
    "rosenbrock_disk",
    "box_quadratic",
    "linear_eq_quadratic",
    "hs21",
    "halfplanes_quadratic",
    "circle_eq",
    "mishra_bird_constrained",
)
ALL = [p.id for p in problems.list_problems("constrained")]
CONVEX = [pid for pid in ALL if "convex" in problems.get(pid).tags]
_H5 = float(np.finfo(float).eps) ** 0.2
EPS = float(np.finfo(float).eps)


def _fd5(fun, x: np.ndarray) -> np.ndarray:
    """5-point central-difference derivative of fun (scalar or vector valued) w.r.t. x."""
    cols = []
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = _H5
        d = (
            -np.asarray(fun(x + 2 * e))
            + 8 * np.asarray(fun(x + e))
            - 8 * np.asarray(fun(x - e))
            + np.asarray(fun(x - 2 * e))
        )
        cols.append(d / (12 * _H5))
    return np.stack(cols, axis=-1)


def _check_gradient_hessian(fun, grad, hess, x: np.ndarray) -> None:
    g = np.asarray(grad(x), dtype=float)
    H = np.asarray(hess(x), dtype=float)
    assert g.shape == (2,) and H.shape == (2, 2)
    assert_allclose(H, H.T, rtol=0, atol=1e-12 * (1 + np.abs(H).max()))
    scale_g = 1.0 + abs(float(fun(x))) + float(np.abs(g).max())
    assert_allclose(g, _fd5(fun, x), rtol=1e-7, atol=1e-7 * scale_g)
    scale_h = 1.0 + float(np.abs(g).max()) + float(np.abs(H).max())
    assert_allclose(H, _fd5(grad, x), rtol=1e-7, atol=1e-7 * scale_h)


def _cvals(p: Problem, x: np.ndarray) -> np.ndarray:
    return np.array([float(c.fun(x)) for c in p.constraints])


def _slsqp_constraints(p: Problem) -> list[dict]:
    """numopt c_i ≤ 0 ↦ SciPy −c_i ≥ 0; equalities unchanged."""
    out = []
    for c in p.constraints:
        if c.kind == "ineq":
            out.append(
                {"type": "ineq", "fun": lambda x, c=c: -c.fun(x), "jac": lambda x, c=c: -c.grad(x)}
            )
        else:
            out.append({"type": "eq", "fun": c.fun, "jac": c.grad})
    return out


# ======================================================================================
# Registration and metadata
# ======================================================================================


def test_required_ids_registered_with_metadata():
    for pid in REQUIRED:
        assert pid in ALL
    for pid in ALL:
        p = problems.get(pid)
        assert problems.kind_of(pid) == "constrained"
        assert isinstance(p, Problem)
        assert p.dim == 2 and p.grad is not None and p.hess is not None
        assert p.constraints, f"{pid}: a constrained problem needs constraints"
        assert all(c.kind in ("ineq", "eq") and c.latex for c in p.constraints)
        assert len(p.domain) == 2 and all(lo < hi for lo, hi in p.domain)
        assert len(p.x0) == 2 and p.minima and p.latex and p.description
        m = len(p.constraints)
        ex = p.extra
        k = len(p.minima)
        assert len(ex["minima_f"]) == len(ex["multipliers"]) == len(ex["active"]) == k
        assert all(len(lam) == m for lam in ex["multipliers"])
        assert len(ex["affine"]) == len(ex["constraint_hess"]) == m
        assert 1 <= ex["n_global"] <= k
        assert ex["f_min"] == ex["minima_f"][0]
        for xm in p.minima:  # the minimizers lie inside the plotting domain
            assert all(lo <= v <= hi for v, (lo, hi) in zip(xm, p.domain, strict=True))


def test_to_dict_is_strict_json():
    for pid in ALL:
        d = problems.get(pid).to_dict()
        json.dumps(d, allow_nan=False)
        assert len(d["constraints"]) == len(problems.get(pid).constraints)
        assert all(set(c) == {"kind", "latex"} for c in d["constraints"])


# ======================================================================================
# Derivatives (objective and constraints) against finite differences
# ======================================================================================


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_at_minima_and_x0(pid):
    p = problems.get(pid)
    for x in [np.asarray(p.x0, dtype=float), *(np.asarray(xm) for xm in p.minima)]:
        _check_gradient_hessian(p.f, p.grad, p.hess, x)
        for c, ch in zip(p.constraints, p.extra["constraint_hess"], strict=True):
            _check_gradient_hessian(c.fun, c.grad, ch, x)


@settings(max_examples=1000, deadline=None)
@given(
    st.sampled_from(ALL),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_derivatives_property(pid, u, v):
    p = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    x = np.array([x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)])
    _check_gradient_hessian(p.f, p.grad, p.hess, x)
    for c, ch in zip(p.constraints, p.extra["constraint_hess"], strict=True):
        _check_gradient_hessian(c.fun, c.grad, ch, x)


@pytest.mark.parametrize("pid", ALL)
def test_grid_evaluation_matches_pointwise(pid):
    p = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    X, Y = np.meshgrid(np.linspace(x_lo, x_hi, 7), np.linspace(y_lo, y_hi, 5))
    G = np.stack([X, Y])  # (2, 5, 7)
    funs = [p.f, *(c.fun for c in p.constraints)]
    for fun in funs:
        Z = np.broadcast_to(np.asarray(fun(G), dtype=float), X.shape)
        ref = np.array([[fun(np.array([X[i, j], Y[i, j]])) for j in range(7)] for i in range(5)])
        assert_allclose(Z, ref, rtol=1e-15, atol=0)


@pytest.mark.parametrize("pid", ALL)
def test_affine_flags(pid):
    p = problems.get(pid)
    pts = np.array([[0.3, -1.7], [2.1, 0.4], [-1.1, 0.9]])
    for c, aff, ch in zip(
        p.constraints, p.extra["affine"], p.extra["constraint_hess"], strict=True
    ):
        if aff:
            # c(x + y) − c(x) − c(y) + c(0) = 0 for affine c, and ∇c is constant.
            a, b = pts[0], pts[1]
            z = np.zeros(2)
            assert abs(c.fun(a + b) - c.fun(a) - c.fun(b) + c.fun(z)) <= 1e-13
            assert_allclose(c.grad(a), c.grad(b), rtol=0, atol=0)
            assert np.all(ch(pts[2]) == 0.0)
        else:
            assert np.any(ch(pts[2]) != 0.0)


# ======================================================================================
# Known solutions: KKT conditions, second-order sufficiency, values
# ======================================================================================


@pytest.mark.parametrize("pid", ALL)
def test_listed_minima_satisfy_kkt(pid):
    p = problems.get(pid)
    is_eq = np.array([c.kind == "eq" for c in p.constraints])
    for xm, lam, act, fm in zip(
        p.minima, p.extra["multipliers"], p.extra["active"], p.extra["minima_f"], strict=True
    ):
        x = np.asarray(xm, dtype=float)
        lam = np.asarray(lam, dtype=float)
        c = _cvals(p, x)
        J = np.array([c_.grad(x) for c_ in p.constraints])
        g = p.grad(x)
        scale = 1.0 + float(np.abs(g).max()) + float(np.abs(J.T * np.abs(lam)).max())
        # Stationarity of the Lagrangian.
        assert np.abs(g + J.T @ lam).max() <= 1e-12 * scale
        # Primal feasibility; the listed active set is exactly {i : c_i(x*) = 0}.
        c_scale = 1.0 + float(np.abs(x).max()) * float(np.abs(J).max())
        assert np.all(np.abs(c[is_eq]) <= 1e-13 * c_scale)
        assert np.all(c[~is_eq] <= 1e-13 * c_scale)
        on_boundary = [i for i in range(c.size) if abs(c[i]) <= 1e-13 * c_scale]
        assert on_boundary == sorted(act)
        # Dual feasibility and complementarity.
        assert np.all(lam[~is_eq] >= 0.0)
        inactive = [i for i in range(c.size) if i not in act]
        assert np.all(lam[inactive] == 0.0)
        # Listed values.
        assert math.isclose(float(p.f(x)), fm, rel_tol=1e-14, abs_tol=1e-14)


@pytest.mark.parametrize("pid", ALL)
def test_listed_minima_second_order_sufficient(pid):
    p = problems.get(pid)
    for xm, lam, act in zip(p.minima, p.extra["multipliers"], p.extra["active"], strict=True):
        x = np.asarray(xm, dtype=float)
        H = p.hess(x) + sum(
            lam_i * ch(x) for lam_i, ch in zip(lam, p.extra["constraint_hess"], strict=True)
        )
        strongly = [
            i for i in act if p.constraints[i].kind == "eq" or lam[i] > 0.0
        ]  # constraints that fix the critical cone to a subspace
        A = np.array([p.constraints[i].grad(x) for i in strongly]).reshape(-1, 2)
        _, sv, Vt = np.linalg.svd(A) if A.size else (None, np.zeros(0), np.eye(2))
        rank = int(np.sum(sv > 1e-12 * (sv.max() if sv.size else 1.0)))
        Z = Vt[rank:].T  # orthonormal basis of the null space (2, 2 − rank)
        if Z.shape[1] == 0:
            continue  # a vertex: the critical cone is {0}
        eig = np.linalg.eigvalsh(Z.T @ H @ Z)
        assert eig.min() > 1e-8 * (1.0 + np.abs(H).max()), f"{pid}: SOSC fails at {xm}"


def test_closed_form_solutions():
    s5, s2 = math.sqrt(5.0), math.sqrt(2.0)
    expected = {
        "quadratic_disk": ([2 / s5, 1 / s5], 6 - 2 * s5, [s5 - 1]),
        "rosenbrock_disk": ([1.0, 1.0], 0.0, [0.0]),
        "box_quadratic": ([1.0, 0.5], 0.75, [0.0, 1.5, 0.0, 0.0]),
        "linear_eq_quadratic": ([2 / 3, 1 / 3], 2 / 3, [-4 / 3]),
        "hs21": ([2.0, 0.0], -99.96, [0.0, 0.04, 0.0, 0.0, 0.0]),
        "halfplanes_quadratic": ([2 / 3, 2 / 3], 19 / 6, [4 / 3, 2 / 3]),
        "circle_eq": ([-1 / s2, -1 / s2], -s2, [1 / s2]),
    }
    for pid, (x, f, lam) in expected.items():
        p = problems.get(pid)
        assert_allclose(p.minima[0], x, rtol=1e-15, atol=1e-16)
        assert math.isclose(p.extra["f_min"], f, rel_tol=1e-15, abs_tol=1e-15)
        assert_allclose(p.extra["multipliers"][0], lam, rtol=1e-15, atol=0)


def test_global_minima_come_first():
    for pid in ALL:
        p = problems.get(pid)
        fs = p.extra["minima_f"]
        n_g = p.extra["n_global"]
        assert all(math.isclose(fs[i], fs[0], rel_tol=1e-12, abs_tol=1e-12) for i in range(n_g))
        assert all(fs[i] > fs[0] + 1e-9 for i in range(n_g, len(fs)))


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(ALL), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_no_feasible_point_beats_the_global_minimum(pid, u, v):
    p = problems.get(pid)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    x = np.array([x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)])
    c = _cvals(p, x)
    if any(con.kind == "eq" for con in p.constraints):
        return  # random points are never on an equality manifold
    if np.all(c <= 0.0):
        assert float(p.f(x)) >= p.extra["f_min"] - 1e-12 * (1.0 + abs(p.extra["f_min"]))


@pytest.mark.parametrize("pid", ALL)
def test_default_start_is_interior(pid):
    p = problems.get(pid)
    x0 = np.asarray(p.x0, dtype=float)
    c = _cvals(p, x0)
    for i, con in enumerate(p.constraints):
        if con.kind == "ineq":
            assert c[i] < 0.0
        elif p.extra["affine"][i]:
            assert abs(c[i]) <= 1e-15


# ======================================================================================
# Simple feasible sets: bounds, disk, polyhedron, vertices
# ======================================================================================


@pytest.mark.parametrize("pid", [pid for pid in ALL if "projection" in problems.get(pid).extra])
def test_projection_metadata_describes_the_constraints(pid):
    p = problems.get(pid)
    ex = p.extra
    rng = np.random.default_rng(7)
    X = rng.uniform(-60, 60, size=(50, 2))
    is_eq = np.array([c.kind == "eq" for c in p.constraints])
    for x in X:
        c = _cvals(p, x)
        if ex["projection"] == "box":
            lo, hi = np.asarray(ex["bounds"], dtype=float).T
            inside = bool(np.all((lo <= x) & (x <= hi)))
            assert inside == bool(np.all(c <= 0.0))
        if ex["projection"] == "disk":
            d = ex["disk"]
            r = float(np.linalg.norm(x - np.asarray(d["center"])))
            assert (r <= d["radius"]) == bool(c[0] <= 0.0)
        if ex["projection"] in ("box", "polyhedron"):
            A_ub = np.asarray(ex["A_ub"], dtype=float).reshape(-1, 2)
            A_eq = np.asarray(ex["A_eq"], dtype=float).reshape(-1, 2)
            assert_allclose(A_ub @ x - np.asarray(ex["b_ub"]), c[~is_eq], rtol=1e-14, atol=1e-12)
            assert_allclose(A_eq @ x - np.asarray(ex["b_eq"]), c[is_eq], rtol=1e-14, atol=1e-12)
            assert all(ex["affine"])


@pytest.mark.parametrize("pid", [pid for pid in ALL if "vertices" in problems.get(pid).extra])
def test_vertices_span_the_feasible_set(pid):
    p = problems.get(pid)
    V = np.asarray(p.extra["vertices"], dtype=float)
    # Counter-clockwise (positive signed area, shoelace formula).
    area = 0.5 * float(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
    assert area > 0.0
    # Each vertex is feasible with at least two active constraints (n = 2).
    for v in V:
        c = _cvals(p, v)
        assert np.all(c <= 1e-12)
        assert int(np.sum(np.abs(c) <= 1e-12)) >= 2
    # conv(V) ⊇ C: every feasible point of a grid over the domain lies in the hull.
    hull = ConvexHull(V)
    assert math.isclose(hull.volume, area, rel_tol=1e-12)
    (x_lo, x_hi), (y_lo, y_hi) = p.domain
    for x in np.linspace(x_lo, x_hi, 41):
        for y in np.linspace(y_lo, y_hi, 41):
            pt = np.array([x, y])
            if np.all(_cvals(p, pt) <= 0.0):
                assert np.all(hull.equations[:, :2] @ pt + hull.equations[:, 2] <= 1e-9)


# ======================================================================================
# SciPy oracles
# ======================================================================================


@pytest.mark.parametrize("pid", ALL)
def test_slsqp_oracle_finds_a_listed_minimizer_with_its_multipliers(pid):
    p = problems.get(pid)
    res = minimize(
        p.f,
        np.asarray(p.x0, dtype=float),
        jac=p.grad,
        constraints=_slsqp_constraints(p),
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    errs = [float(np.abs(res.x - np.asarray(xm)).max()) for xm in p.minima]
    j = int(np.argmin(errs))
    # SLSQP stops on |Δf| ≤ ftol, so x is accurate to ~√ftol·(curvature ratio).
    assert errs[j] <= 1e-6, f"{pid}: SLSQP x = {res.x}, nearest listed minimizer {p.minima[j]}"
    if pid in CONVEX:
        assert j == 0  # convex problems have the global minimizer only
    # SciPy's Lagrangian is f − μᵀc_scipy: inequalities (−c ≥ 0) keep the sign of λ,
    # equalities flip it.
    sign = np.array([1.0 if c.kind == "ineq" else -1.0 for c in p.constraints])
    assert_allclose(sign * res.multipliers, p.extra["multipliers"][j], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("pid", CONVEX)
def test_trust_constr_oracle_on_convex_problems(pid):
    p = problems.get(pid)
    cons = p.constraints
    chess = p.extra["constraint_hess"]
    nlc = NonlinearConstraint(
        lambda x: np.array([c.fun(x) for c in cons]),
        [-np.inf if c.kind == "ineq" else 0.0 for c in cons],
        np.zeros(len(cons)),
        jac=lambda x: np.array([c.grad(x) for c in cons]),  # pyright: ignore[reportArgumentType]
        hess=lambda x, v: sum(vi * h(x) for vi, h in zip(v, chess, strict=True)),
    )
    res = minimize(
        p.f,
        np.asarray(p.x0, dtype=float),
        jac=p.grad,
        hess=p.hess,
        constraints=[nlc],
        method="trust-constr",
        options={"gtol": 1e-12, "xtol": 1e-14, "barrier_tol": 1e-12, "maxiter": 5000},
    )
    # trust-constr stops on its gtol test at a central point of a barrier subproblem whose
    # parameter μ is still 1e-7 to 1e-5 (measured: f − f* = |A|·μ, |A| = number of active
    # constraints); a central point has f − f* ≤ m·μ (B&V §11.2.2). Its iterates
    # are feasible, and for a convex problem ∇f(x*)ᵀ(x − x*) ≥ 0 on the feasible set, so
    # f(x) − f* ≥ (σ/2)‖x − x*‖² with σ = λ_min(∇²f) (the objectives here are quadratics).
    x = np.asarray(res.x, dtype=float)
    c = _cvals(p, x)
    is_eq = np.array([con.kind == "eq" for con in p.constraints])
    assert np.all(c[~is_eq] <= 1e-12) and np.all(np.abs(c[is_eq]) <= 1e-12)
    mu = float(res.get("barrier_parameter", 0.0))
    gap = float(res.fun) - p.extra["f_min"]
    assert -1e-12 <= gap <= 2.0 * int(np.sum(~is_eq)) * mu + 1e-10
    sigma = float(np.linalg.eigvalsh(p.hess(x)).min())
    assert np.linalg.norm(x - p.minima[0]) <= math.sqrt(2.0 * (max(gap, 0.0) + 1e-12) / sigma)


@pytest.mark.parametrize("pid", ALL)
def test_descriptions_mark_optimal_values_with_star_operator(pid: str) -> None:
    """Brand rule (docs/brand/brand.md): the solution carries the superscript ⋆ (U+22C6), never
    an ASCII '*' (regression: λ*, ν* in the descriptions). No other '*' is used in prose."""
    p = problems.get(pid)
    assert "*" not in p.description, p.description
    assert "*" not in p.name


def test_multiplier_descriptions_use_the_star_operator() -> None:
    expected = {
        "quadratic_disk": "λ⋆ = √5 − 1",
        "rosenbrock_disk": "λ⋆ = 0",
        "rosenbrock_unit_disk": "λ⋆ ≈ 0.1215",
        "linear_eq_quadratic": "ν⋆ = −4/3",
        "circle_eq": "2ν⋆I",
    }
    for pid, text in expected.items():
        assert text in problems.get(pid).description, (pid, text)
