"""Constrained test problems: min f(x) subject to smooth constraints, all in ℝ².

Every problem is 2-D so the web app can draw the contours of f, the feasible region and the
path of a method. Each problem supplies exact derivatives of f and of every constraint, a
plotting ``domain``, a default ``x0`` that satisfies every inequality strictly and every
affine equality exactly (so that interior-point methods can start from it; a nonlinear
equality has no interior, and ``circle_eq`` starts off its circle), and its known constrained
minimizers with their Lagrange multipliers.

Conventions:
    * ``Constraint(kind="ineq")`` means ``c_i(x) ≤ 0``; ``kind="eq"`` means ``c_i(x) = 0``.
    * Lagrangian ``L(x, λ) = f(x) + Σ_i λ_i c_i(x)``. A KKT point satisfies
      ``∇f(x*) + Σ_i λ_i ∇c_i(x*) = 0``, ``λ_i ≥ 0`` and ``λ_i c_i(x*) = 0`` for every
      inequality, and ``c_i(x*) = 0`` for every equality (equality multipliers have any sign).
    * ``f(x)`` and every ``Constraint.fun`` accept ``x`` of shape ``(2,)`` (returning a float)
      or ``(2, *grid)`` (returning an array of shape ``grid``), so the visualizer can evaluate
      whole grids at once. ``grad`` / ``hess`` take ``x`` of shape ``(2,)``.
    * ``minima`` lists known local constrained minimizers, **global first**; every listed
      point is verified in the tests (feasibility, KKT conditions with the listed
      multipliers, and second-order sufficiency on the critical cone).

Extra keys (``Problem.extra``):
    minima_f: [float] — f at each entry of ``minima`` (same order).
    n_global: int — the first ``n_global`` entries of ``minima`` are global minimizers.
    f_min: float — the global constrained minimum value.
    multipliers: [[float]] — for each minimizer, the multiplier λ_i of every constraint
        (problem order, sign convention above).
    active: [[int]] — for each minimizer, the indices of the constraints with c_i(x*) = 0.
    affine: [bool] — per constraint, True when c_i is affine (c_i(x) = a_iᵀx − b_i).
    constraint_hess: (callable, ...) — per constraint, x ↦ ∇²c_i(x) of shape (2, 2).
        (Not JSON: ``Problem.to_dict`` does not export ``extra``.)
    projection: "box" | "disk" | "polyhedron" — present when the feasible set is a simple
        convex set with an exact Euclidean projection (projected gradient, Frank–Wolfe):
        box: ``bounds = [[lo_1, hi_1], [lo_2, hi_2]]``;
        disk: ``disk = {"center": [c_1, c_2], "radius": r}``;
        polyhedron: ``A_ub, b_ub`` (A_ub x ≤ b_ub) and ``A_eq, b_eq`` (A_eq x = b_eq), whose
        rows are the problem's linear constraints in problem order.
    vertices: [[x, y], ...] — the vertices (counter-clockwise) of a bounded polyhedral
        feasible set; the linear-minimization oracle of Frank–Wolfe minimizes over them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.types import Constraint, Problem
from .registry import factory

Array = NDArray[np.float64]
Fn = Callable[[Any], Any]
_SQRT2 = math.sqrt(2.0)
_SQRT5 = math.sqrt(5.0)


def _arr(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


class _Con:
    """A constraint together with its Hessian and affinity flag (assembled into a Problem)."""

    def __init__(
        self,
        kind: Literal["ineq", "eq"],
        fun: Fn,
        grad: Callable[[Any], Array],
        hess: Callable[[Any], Array],
        latex: str,
        affine: bool,
    ) -> None:
        self.kind: Literal["ineq", "eq"] = kind
        self.fun = fun
        self.grad = grad
        self.hess = hess
        self.latex = latex
        self.affine = affine


def _linear(kind: Literal["ineq", "eq"], a: Sequence[float], b: float, latex: str) -> _Con:
    """The affine constraint c(x) = a₁x + a₂y − b (≤ 0 or = 0)."""
    a1, a2 = float(a[0]), float(a[1])
    a_vec = np.array([a1, a2])

    def fun(x: ArrayLike) -> Any:
        x = _arr(x)
        return a1 * x[0] + a2 * x[1] - b

    def grad(x: ArrayLike) -> Array:
        return a_vec.copy()

    def hess(x: ArrayLike) -> Array:
        return np.zeros((2, 2))

    return _Con(kind, fun, grad, hess, latex, True)


def _disk(kind: Literal["ineq", "eq"], center: Sequence[float], radius: float, latex: str) -> _Con:
    """c(x) = (x − c₁)² + (y − c₂)² − r² (≤ 0: inside the disk; = 0: on the circle)."""
    c1, c2 = float(center[0]), float(center[1])
    r_sq = float(radius) ** 2

    def fun(x: ArrayLike) -> Any:
        x = _arr(x)
        return (x[0] - c1) ** 2 + (x[1] - c2) ** 2 - r_sq

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([2.0 * (x[0] - c1), 2.0 * (x[1] - c2)])

    def hess(x: ArrayLike) -> Array:
        return 2.0 * np.eye(2)

    return _Con(kind, fun, grad, hess, latex, False)


def _box(bounds: Sequence[Sequence[float]], names: Sequence[str] = ("x", "y")) -> list[_Con]:
    """The four bound constraints lo_j − x_j ≤ 0, x_j − hi_j ≤ 0 (j = 1, 2, in this order)."""
    cons: list[_Con] = []
    for j, (lo, hi) in enumerate(bounds):
        e = [0.0, 0.0]
        e[j] = 1.0
        ne = [-v for v in e]
        cons.append(_linear("ineq", ne, -float(lo), rf"{_fmt(lo)} - {names[j]} \le 0"))
        cons.append(_linear("ineq", e, float(hi), rf"{names[j]} - {_fmt(hi)} \le 0"))
    return cons


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def _problem(
    *,
    id: str,
    name: str,
    latex: str,
    f: Fn,
    grad: Callable[[Any], Array],
    hess: Callable[[Any], Array],
    constraints: Sequence[_Con],
    domain: Sequence[tuple[float, float]],
    x0: Sequence[float],
    minima: Sequence[Sequence[float]],
    minima_f: Sequence[float],
    multipliers: Sequence[Sequence[float]],
    active: Sequence[Sequence[int]],
    n_global: int,
    description: str,
    tags: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> Problem:
    """Assemble a constrained Problem and its metadata (see the module docstring)."""
    m = len(constraints)
    if not len(minima) == len(minima_f) == len(multipliers) == len(active):
        raise ValueError(f"{id}: inconsistent minima metadata")
    if any(len(lam) != m for lam in multipliers) or not 1 <= n_global <= len(minima):
        raise ValueError(f"{id}: multipliers must have one entry per constraint")
    return Problem(
        id=id,
        name=name,
        latex=latex,
        f=f,
        dim=2,
        domain=tuple((float(lo), float(hi)) for lo, hi in domain),
        grad=grad,
        hess=hess,
        x0=[float(v) for v in x0],
        minima=tuple([float(v) for v in xm] for xm in minima),
        constraints=tuple(Constraint(c.kind, c.fun, c.grad, c.latex) for c in constraints),
        description=description,
        tags=tags,
        extra={
            "minima_f": [float(v) for v in minima_f],
            "n_global": n_global,
            "f_min": float(minima_f[0]),
            "multipliers": [[float(v) for v in lam] for lam in multipliers],
            "active": [[int(i) for i in act] for act in active],
            "affine": [c.affine for c in constraints],
            "constraint_hess": tuple(c.hess for c in constraints),
            **(extra or {}),
        },
    )


def _polyhedron(constraints: Sequence[_Con]) -> dict[str, Any]:
    """``A_ub, b_ub, A_eq, b_eq`` of a problem whose constraints are all affine."""
    if not all(c.affine for c in constraints):
        raise ValueError("a polyhedron needs affine constraints")
    origin = np.zeros(2)
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    for c in constraints:
        a = [float(v) for v in c.grad(origin)]
        b = -float(c.fun(origin))  # c(x) = aᵀx − b  ⇒  b = −c(0)
        (A_ub if c.kind == "ineq" else A_eq).append(a)
        (b_ub if c.kind == "ineq" else b_eq).append(b)
    return {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq}


# --------------------------------------------------------------------------------------
# Quadratic over the unit disk (active constraint, closed-form solution)
# --------------------------------------------------------------------------------------


@factory("constrained")
def quadratic_disk() -> Problem:
    # f = ‖x − c‖² with c = (2, 1) outside the unit disk: x* = c/‖c‖ = (2, 1)/√5.
    # ∇f(x*) + λ∇g(x*) = 2(x* − c) + 2λx* = 0 with x* = c/√5 ⇒ (1/√5 − 1) + λ/√5 = 0
    # ⇒ λ* = √5 − 1;  f* = (√5 − 1)² = 6 − 2√5.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])

    def hess(x: ArrayLike) -> Array:
        return 2.0 * np.eye(2)

    return _problem(
        id="quadratic_disk",
        name="Quadratic over the unit disk",
        latex=r"\min\ (x-2)^2 + (y-1)^2 \quad \text{s.t.}\ x^2 + y^2 \le 1",
        f=f,
        grad=grad,
        hess=hess,
        constraints=[_disk("ineq", (0.0, 0.0), 1.0, r"x^2 + y^2 - 1 \le 0")],
        domain=((-1.5, 2.5), (-1.5, 1.75)),
        x0=(-0.5, 0.5),
        minima=((2.0 / _SQRT5, 1.0 / _SQRT5),),
        minima_f=(6.0 - 2.0 * _SQRT5,),
        multipliers=((_SQRT5 - 1.0,),),
        active=((0,),),
        n_global=1,
        description="The unconstrained minimizer (2, 1) lies outside the unit disk, so the "
        "constraint is active: the solution is the boundary point (2, 1)/√5 where the level "
        "circle of f touches the disk, with multiplier λ⋆ = √5 − 1.",
        tags=("convex", "quadratic", "active", "disk", "2d"),
        extra={"projection": "disk", "disk": {"center": [0.0, 0.0], "radius": 1.0}},
    )


# --------------------------------------------------------------------------------------
# Rosenbrock inside disks
# --------------------------------------------------------------------------------------


def _rosen(x: ArrayLike) -> Any:
    x = _arr(x)
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def _rosen_grad(x: ArrayLike) -> Array:
    x = _arr(x)
    return np.array(
        [-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2), 200.0 * (x[1] - x[0] ** 2)]
    )


def _rosen_hess(x: ArrayLike) -> Array:
    x = _arr(x)
    return np.array(
        [[2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0]], [-400.0 * x[0], 200.0]]
    )


_ROSEN_LATEX = r"(1-x)^2 + 100\,(y-x^2)^2"


@factory("constrained")
def rosenbrock_disk() -> Problem:
    return _problem(
        id="rosenbrock_disk",
        name="Rosenbrock in the disk x² + y² ≤ 2",
        latex=rf"\min\ {_ROSEN_LATEX} \quad \text{{s.t.}}\ x^2 + y^2 \le 2",
        f=_rosen,
        grad=_rosen_grad,
        hess=_rosen_hess,
        constraints=[_disk("ineq", (0.0, 0.0), _SQRT2, r"x^2 + y^2 - 2 \le 0")],
        domain=((-1.6, 1.6), (-1.6, 1.6)),
        x0=(-1.2, 0.5),
        minima=((1.0, 1.0),),
        minima_f=(0.0,),
        multipliers=((0.0,),),
        active=((0,),),
        n_global=1,
        description="The Rosenbrock minimizer (1, 1) lies exactly on the circle x² + y² = 2. "
        "The constraint is active with multiplier λ⋆ = 0 (weakly active: strict "
        "complementarity fails), so penalty and barrier methods approach (1, 1) without the "
        "constraint pushing back. Rosenbrock is not convex: det ∇²f = 400 − 80000(y − x²), so "
        "∇²f is indefinite where y > x² + 1/200 and Newton steps there need a Hessian "
        "modification.",
        tags=("nonconvex", "degenerate", "weakly-active", "disk", "2d"),
        extra={"projection": "disk", "disk": {"center": [0.0, 0.0], "radius": _SQRT2}},
    )


@factory("constrained")
def rosenbrock_unit_disk() -> Problem:
    # KKT point computed by Newton's method on (∇f + 2λx, x² + y² − 1) = 0 (residual ≤ 4e-14);
    # the tests re-verify the KKT conditions and second-order sufficiency.
    x_star = (0.78641515416842789, 0.61769831252339347)
    return _problem(
        id="rosenbrock_unit_disk",
        name="Rosenbrock in the unit disk",
        latex=rf"\min\ {_ROSEN_LATEX} \quad \text{{s.t.}}\ x^2 + y^2 \le 1",
        f=_rosen,
        grad=_rosen_grad,
        hess=_rosen_hess,
        constraints=[_disk("ineq", (0.0, 0.0), 1.0, r"x^2 + y^2 - 1 \le 0")],
        domain=((-1.5, 1.5), (-1.5, 1.5)),
        x0=(-0.5, 0.0),
        minima=(x_star,),
        minima_f=(0.045674808719500221,),
        multipliers=((0.12149655699928837,),),
        active=((0,),),
        n_global=1,
        description="Rosenbrock's banana valley cut off by the unit circle (the classic "
        "example of MATLAB's fmincon documentation). The minimizer "
        "(0.7864, 0.6177) lies on the circle where the valley meets it, with multiplier "
        "λ⋆ ≈ 0.1215.",
        tags=("nonconvex", "active", "disk", "2d"),
        extra={"projection": "disk", "disk": {"center": [0.0, 0.0], "radius": 1.0}},
    )


# --------------------------------------------------------------------------------------
# Box-constrained quadratic
# --------------------------------------------------------------------------------------


@factory("constrained")
def box_quadratic() -> Problem:
    # f = ½(x − c)ᵀA(x − c), A = [[2, 1], [1, 2]], c = (2, 0), box [−1, 1]².
    # With x = 1 fixed: ∂f/∂y = (x − 2) + 2y = 0 ⇒ y = ½ (inside); ∂f/∂x = 2(x − 2) + y = −3/2,
    # so the multiplier of x − 1 ≤ 0 is λ = 3/2 > 0. f* = ½(2·1 − 2·½ + 2·¼) = 3/4.
    # Note that clipping the unconstrained minimizer (2, 0) to the box gives (1, 0) ≠ x*.
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    c = np.array([2.0, 0.0])
    bounds = [[-1.0, 1.0], [-1.0, 1.0]]

    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        d0, d1 = x[0] - 2.0, x[1]
        return d0**2 + d0 * d1 + d1**2

    def grad(x: ArrayLike) -> Array:
        return A @ (_arr(x) - c)

    def hess(x: ArrayLike) -> Array:
        return A.copy()

    cons = _box(bounds)
    return _problem(
        id="box_quadratic",
        name="Quadratic in a box",
        latex=r"\min\ (x-2)^2 + (x-2)\,y + y^2 \quad \text{s.t.}\ -1 \le x \le 1,\ -1 \le y \le 1",
        f=f,
        grad=grad,
        hess=hess,
        constraints=cons,
        domain=((-1.5, 2.5), (-1.5, 1.5)),
        x0=(-0.5, -0.5),
        minima=((1.0, 0.5),),
        minima_f=(0.75,),
        multipliers=((0.0, 1.5, 0.0, 0.0),),
        active=((1,),),
        n_global=1,
        description="A coupled convex quadratic whose unconstrained minimizer (2, 0) lies "
        "outside the box [−1, 1]². Only the bound x ≤ 1 is active at the solution (1, ½); "
        "clipping (2, 0) to the box would give the wrong point (1, 0).",
        tags=("convex", "quadratic", "box", "linear-constraints", "2d"),
        extra={
            "projection": "box",
            "bounds": bounds,
            "vertices": [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            **_polyhedron(cons),
        },
    )


# --------------------------------------------------------------------------------------
# Equality-constrained quadratic
# --------------------------------------------------------------------------------------


@factory("constrained")
def linear_eq_quadratic() -> Problem:
    # ∇f + ν∇h = (2x, 4y) + ν(1, 1) = 0 ⇒ x = 2y; x + y = 1 ⇒ x* = (2/3, 1/3), ν* = −4/3,
    # f* = 4/9 + 2/9 = 2/3.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return x[0] ** 2 + 2.0 * x[1] ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([2.0 * x[0], 4.0 * x[1]])

    def hess(x: ArrayLike) -> Array:
        return np.diag([2.0, 4.0])

    cons = [_linear("eq", (1.0, 1.0), 1.0, r"x + y - 1 = 0")]
    return _problem(
        id="linear_eq_quadratic",
        name="Quadratic on a line",
        latex=r"\min\ x^2 + 2y^2 \quad \text{s.t.}\ x + y = 1",
        f=f,
        grad=grad,
        hess=hess,
        constraints=cons,
        domain=((-1.5, 2.0), (-1.5, 2.0)),
        x0=(-0.5, 1.5),
        minima=((2.0 / 3.0, 1.0 / 3.0),),
        minima_f=(2.0 / 3.0,),
        multipliers=((-4.0 / 3.0,),),
        active=((0,),),
        n_global=1,
        description="The textbook Lagrange-multiplier example: the solution (2/3, 1/3) is "
        "where an ellipse x² + 2y² = const touches the line x + y = 1; ∇f = (4/3)(1, 1) is "
        "parallel to the line's normal, with multiplier ν⋆ = −4/3. The default start lies on "
        "the line.",
        tags=("convex", "quadratic", "equality", "linear-constraints", "2d"),
        extra={"projection": "polyhedron", **_polyhedron(cons)},
    )


# --------------------------------------------------------------------------------------
# Hock–Schittkowski problem 21
# --------------------------------------------------------------------------------------


@factory("constrained")
def hs21() -> Problem:
    # Hock & Schittkowski (1981), problem 21: f = 0.01x₁² + x₂² − 100 subject to
    # 10x₁ − x₂ ≥ 10, 2 ≤ x₁ ≤ 50, −50 ≤ x₂ ≤ 50. x* = (2, 0), f* = −99.96.
    # Only the bound x₁ ≥ 2 is active: ∇f(x*) = (0.04, 0) = −λ·(−1, 0) ⇒ λ = 0.04.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return 0.01 * x[0] ** 2 + x[1] ** 2 - 100.0

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([0.02 * x[0], 2.0 * x[1]])

    def hess(x: ArrayLike) -> Array:
        return np.diag([0.02, 2.0])

    cons = [
        _linear("ineq", (-10.0, 1.0), -10.0, r"10 - 10x + y \le 0"),
        *_box([[2.0, 50.0], [-50.0, 50.0]]),
    ]
    return _problem(
        id="hs21",
        name="Hock–Schittkowski 21",
        latex=r"\min\ 0.01x^2 + y^2 - 100 \quad \text{s.t.}\ 10x - y \ge 10,\ "
        r"2 \le x \le 50,\ -50 \le y \le 50",
        f=f,
        grad=grad,
        hess=hess,
        constraints=cons,
        domain=((0.0, 52.0), (-52.0, 52.0)),
        x0=(10.0, 10.0),
        minima=((2.0, 0.0),),
        minima_f=(-99.96,),
        multipliers=((0.0, 0.04, 0.0, 0.0, 0.0),),
        active=((1,),),
        n_global=1,
        description="Hock & Schittkowski (1981) test problem 21: a convex quadratic with one "
        "general linear inequality and bounds. Only the bound x ≥ 2 is active at (2, 0). "
        "The published start (−1, −1) is infeasible; the default start (10, 10) is strictly "
        "feasible so that interior-point methods can use it.",
        tags=("convex", "quadratic", "linear-constraints", "bounds", "hock-schittkowski", "2d"),
        extra={
            "projection": "polyhedron",
            **_polyhedron(cons),
            "vertices": [[2.0, -50.0], [50.0, -50.0], [50.0, 50.0], [6.0, 50.0], [2.0, 10.0]],
            "x0_hs": [-1.0, -1.0],
        },
    )


# --------------------------------------------------------------------------------------
# Two half-planes, both active
# --------------------------------------------------------------------------------------


@factory("constrained")
def halfplanes_quadratic() -> Problem:
    # f = (x − 2)² + 2(y − 3/2)², g₁ = x + 2y − 2 ≤ 0, g₂ = 2x + y − 2 ≤ 0.
    # The lines meet at (2/3, 2/3); ∇f there = (−8/3, −10/3). Solving
    # ∇f + λ₁(1, 2) + λ₂(2, 1) = 0 gives λ₁ = 4/3, λ₂ = 2/3 (both > 0, so the vertex is optimal).
    # f* = (4/3)² + 2(5/6)² = 16/9 + 25/18 = 19/6.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return (x[0] - 2.0) ** 2 + 2.0 * (x[1] - 1.5) ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([2.0 * (x[0] - 2.0), 4.0 * (x[1] - 1.5)])

    def hess(x: ArrayLike) -> Array:
        return np.diag([2.0, 4.0])

    cons = [
        _linear("ineq", (1.0, 2.0), 2.0, r"x + 2y - 2 \le 0"),
        _linear("ineq", (2.0, 1.0), 2.0, r"2x + y - 2 \le 0"),
    ]
    return _problem(
        id="halfplanes_quadratic",
        name="Quadratic in a wedge of two half-planes",
        latex=r"\min\ (x-2)^2 + 2(y-\tfrac32)^2 \quad \text{s.t.}\ x + 2y \le 2,\ 2x + y \le 2",
        f=f,
        grad=grad,
        hess=hess,
        constraints=cons,
        domain=((-2.0, 3.0), (-2.0, 3.0)),
        x0=(-1.0, -1.0),
        minima=((2.0 / 3.0, 2.0 / 3.0),),
        minima_f=(19.0 / 6.0,),
        multipliers=((4.0 / 3.0, 2.0 / 3.0),),
        active=((0, 1),),
        n_global=1,
        description="Two linear inequalities form an unbounded wedge; the unconstrained "
        "minimizer (2, 3/2) lies outside it and the solution is the corner (2/3, 2/3) where "
        "both constraints are active with multipliers 4/3 and 2/3.",
        tags=("convex", "quadratic", "linear-constraints", "vertex-solution", "2d"),
        extra={"projection": "polyhedron", **_polyhedron(cons)},
    )


# --------------------------------------------------------------------------------------
# Linear objective on the unit circle (nonlinear equality)
# --------------------------------------------------------------------------------------


@factory("constrained")
def circle_eq() -> Problem:
    # ∇f + ν∇h = (1, 1) + 2ν(x, y) = 0 on x² + y² = 1 ⇒ x = y = ∓1/√2.
    # Minimizer (−1/√2, −1/√2): ν* = 1/√2, ∇²L = 2ν*I ≻ 0, f* = −√2.
    # The other KKT point (1/√2, 1/√2) (ν = −1/√2) is the constrained maximizer.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return x[0] + x[1]

    def grad(x: ArrayLike) -> Array:
        return np.array([1.0, 1.0])

    def hess(x: ArrayLike) -> Array:
        return np.zeros((2, 2))

    return _problem(
        id="circle_eq",
        name="Linear function on the unit circle",
        latex=r"\min\ x + y \quad \text{s.t.}\ x^2 + y^2 = 1",
        f=f,
        grad=grad,
        hess=hess,
        constraints=[_disk("eq", (0.0, 0.0), 1.0, r"x^2 + y^2 - 1 = 0")],
        domain=((-1.6, 1.6), (-1.6, 1.6)),
        x0=(0.3, 1.2),
        minima=((-1.0 / _SQRT2, -1.0 / _SQRT2),),
        minima_f=(-_SQRT2,),
        multipliers=((1.0 / _SQRT2,),),
        active=((0,),),
        n_global=1,
        description="A linear objective on a nonlinear equality constraint (a nonconvex "
        "feasible set). The Lagrangian's curvature 2ν⋆I comes only from the constraint. "
        "The point (1/√2, 1/√2) is also a KKT point (ν = −1/√2), but it is the maximizer.",
        tags=("nonconvex", "equality", "nonlinear-constraints", "2d"),
    )


# --------------------------------------------------------------------------------------
# Mishra's bird, constrained (multimodal)
# --------------------------------------------------------------------------------------


def _mishra_parts(x: Array) -> tuple[Any, ...]:
    """A = e^{(1−cos x)²}, B = e^{(1−sin y)²} and the logarithmic derivatives a = A'/A, b = B'/B."""
    sx, cx, sy, cy = np.sin(x[0]), np.cos(x[0]), np.sin(x[1]), np.cos(x[1])
    A = np.exp((1.0 - cx) ** 2)
    B = np.exp((1.0 - sy) ** 2)
    a = 2.0 * (1.0 - cx) * sx
    b = -2.0 * (1.0 - sy) * cy
    return sx, cx, sy, cy, A, B, a, b


@factory("constrained")
def mishra_bird_constrained() -> Problem:
    # Mishra (2006), "Some new test functions for global optimization": the global minimum
    # inside the disk is f(−3.1302468, −1.5821422) = −106.7645367. The list of local
    # minimizers is complete: the interior ones are all the points where Newton's method on
    # ∇f = 0, started from a 101 × 101 grid over the disk, converged with ∇²f ≻ 0 (the other
    # 11 stationary points are saddles); the boundary ones are all the KKT points with λ > 0
    # and positive tangential curvature of ∇²L found by Newton's method on the KKT system
    # started from 721 points of the circle. All are refined to KKT residuals ≤ 1e-13.
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        _, cx, sy, _, A, B, _, _ = _mishra_parts(x)
        return sy * A + cx * B + (x[0] - x[1]) ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        sx, cx, sy, cy, A, B, a, b = _mishra_parts(x)
        d = x[0] - x[1]
        return np.array([sy * A * a - sx * B + 2.0 * d, cy * A + cx * B * b - 2.0 * d])

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        sx, cx, sy, cy, A, B, a, b = _mishra_parts(x)
        a_prime = 2.0 * (sx**2 + (1.0 - cx) * cx)  # da/dx
        b_prime = 2.0 * cy**2 + 2.0 * (1.0 - sy) * sy  # db/dy
        fxx = sy * A * (a * a + a_prime) - cx * B + 2.0
        fxy = cy * A * a - sx * B * b - 2.0
        fyy = -sy * A + cx * B * (b * b + b_prime) + 2.0
        return np.array([[fxx, fxy], [fxy, fyy]])

    return _problem(
        id="mishra_bird_constrained",
        name="Mishra's bird (constrained)",
        latex=r"\min\ \sin y\, e^{(1-\cos x)^2} + \cos x\, e^{(1-\sin y)^2} + (x-y)^2"
        r" \quad \text{s.t.}\ (x+5)^2 + (y+5)^2 \le 25",
        f=f,
        grad=grad,
        hess=hess,
        constraints=[_disk("ineq", (-5.0, -5.0), 5.0, r"(x+5)^2 + (y+5)^2 - 25 \le 0")],
        domain=((-10.0, 0.0), (-10.0, 0.0)),
        x0=(-5.0, -5.0),
        minima=(
            (-3.1302468034546562, -1.5821421769300335),
            (-9.1907144922242008, -7.727253571757136),
            (-3.1757265085378856, -7.8198477790263903),
            (-8.9438004526852257, -1.9264941858848208),
            (-5.3776666083326097, -5.6179076792316662),
        ),
        minima_f=(
            -106.76453674926471,
            -97.894220784141453,
            -87.310882733003581,
            -21.518248963023638,
            1.4870191265420765,
        ),
        multipliers=((0.0,), (6.4201335864045408,), (0.0,), (8.0114423573423998,), (0.0,)),
        active=((), (0,), (), (0,), ()),
        n_global=1,
        description="A multimodal benchmark (Mishra 2006) restricted to a disk. It has five "
        "local minimizers: three interior ones (including the global one) and two on the "
        "boundary circle with an active constraint. Local methods find whichever basin they "
        "start in.",
        tags=("nonconvex", "multimodal", "disk", "2d"),
        extra={"projection": "disk", "disk": {"center": [-5.0, -5.0], "radius": 5.0}},
    )
