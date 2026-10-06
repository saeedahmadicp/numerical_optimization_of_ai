"""Nonlinear systems F(x) = 0 in R² (kind ``"systems"``).

Every problem is a 2-D :class:`~numopt.core.types.Problem` with

* ``f``   — the residual map F, ``(2,) -> (2,)``;
* ``jac`` — the exact Jacobian J(x) = ∂F/∂x, ``(2,) -> (2, 2)`` (row i = ∇Fᵢ);
* ``x0``  — a default starting point;
* ``roots`` — every real root inside ``domain`` (as ``(x, y)`` tuples);
* ``domain`` — the plotting window ``((x_min, x_max), (y_min, y_max))``.

The web app draws the zero contours of F₁ and F₂; their intersections are the roots.
Roots are given in closed form where one exists and are verified in
``tests/test_problems_systems.py``.

Every F and J follows IEEE semantics far from the domain: overflow gives ±inf and a
non-finite argument gives nan, never an exception (squares are written as products,
because the float ``**`` operator raises OverflowError, and :func:`_sin` / :func:`_cos`
return nan where ``math.sin`` / ``math.cos`` raise ValueError).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..core.types import Problem, Vector
from .registry import add

VecFn = Callable[[Vector], Vector]


def _sin(t: float) -> float:
    """sin t, or nan for a non-finite t (``math.sin`` raises ValueError there)."""
    return math.sin(t) if math.isfinite(t) else math.nan


def _cos(t: float) -> float:
    """cos t, or nan for a non-finite t (``math.cos`` raises ValueError there)."""
    return math.cos(t) if math.isfinite(t) else math.nan


def _system(
    id: str,
    name: str,
    latex: str,
    F: VecFn,
    J: VecFn,
    *,
    x0: tuple[float, float],
    roots: tuple[tuple[float, float], ...],
    domain: tuple[tuple[float, float], tuple[float, float]],
    description: str,
    tags: tuple[str, ...] = (),
) -> Problem:
    return add(
        "systems",
        Problem(
            id=id,
            name=name,
            latex=latex,
            f=F,
            jac=J,
            dim=2,
            domain=domain,
            x0=x0,
            roots=roots,
            description=description,
            tags=tags,
        ),
    )


# --- circle and line -----------------------------------------------------------------------
# x² + y² = 4 and y = x − 1  ⇒  2x² − 2x − 3 = 0  ⇒  x = (1 ± √7)/2, y = x − 1.

_S7 = math.sqrt(7.0)

CIRCLE_LINE = _system(
    "circle_line",
    "Circle and line",
    r"F(x,y) = \begin{pmatrix} x^2 + y^2 - 4 \\ y - x + 1 \end{pmatrix}",
    lambda v: np.array([v[0] * v[0] + v[1] * v[1] - 4.0, v[1] - v[0] + 1.0]),
    lambda v: np.array([[2.0 * v[0], 2.0 * v[1]], [-1.0, 1.0]]),
    x0=(2.0, 2.0),
    roots=(((1.0 - _S7) / 2.0, (-1.0 - _S7) / 2.0), ((1.0 + _S7) / 2.0, (_S7 - 1.0) / 2.0)),
    domain=((-3.0, 3.0), (-3.0, 3.0)),
    description=(
        "The circle of radius 2 meets the line y = x − 1 in two points. J is singular where "
        "x + y = 0 (the line is tangent to a circle centered at 0 there)."
    ),
    tags=("polynomial", "two-roots"),
)

# --- gradient of Rosenbrock's function ----------------------------------------------------
# f(x, y) = (1 − x)² + 100 (y − x²)²;  F = ∇f;  J = ∇²f.

ROSENBROCK_SYSTEM = _system(
    "rosenbrock_system",
    "Rosenbrock gradient = 0",
    r"F(x,y) = \nabla\big[(1-x)^2 + 100(y-x^2)^2\big]",
    lambda v: np.array(
        [-400.0 * v[0] * (v[1] - v[0] * v[0]) - 2.0 * (1.0 - v[0]), 200.0 * (v[1] - v[0] * v[0])]
    ),
    lambda v: np.array(
        [
            [1200.0 * v[0] * v[0] - 400.0 * v[1] + 2.0, -400.0 * v[0]],
            [-400.0 * v[0], 200.0],
        ]
    ),
    x0=(-1.2, 1.0),
    roots=((1.0, 1.0),),
    domain=((-2.0, 2.0), (-1.0, 3.0)),
    description=(
        "Stationarity conditions of Rosenbrock's function: the only root is the minimizer "
        "(1, 1). J = ∇²f is singular on the parabola y = x² + 1/200."
    ),
    tags=("polynomial", "badly-scaled"),
)

# --- Freudenstein & Roth (Moré, Garbow & Hillstrom 1981, problem 2) ----------------------
# F₁ − F₂ = −2(y − 4)(y² + 2y + 2), so y = 4 and x = 5 is the only real root.

FREUDENSTEIN_ROTH = _system(
    "freudenstein_roth",
    "Freudenstein–Roth",
    r"F(x,y) = \begin{pmatrix} -13 + x + ((5-y)y - 2)y \\ -29 + x + ((y+1)y - 14)y \end{pmatrix}",
    lambda v: np.array(
        [
            -13.0 + v[0] + ((5.0 - v[1]) * v[1] - 2.0) * v[1],
            -29.0 + v[0] + ((v[1] + 1.0) * v[1] - 14.0) * v[1],
        ]
    ),
    lambda v: np.array(
        [
            [1.0, (10.0 - 3.0 * v[1]) * v[1] - 2.0],
            [1.0, (3.0 * v[1] + 2.0) * v[1] - 14.0],
        ]
    ),
    x0=(0.5, -2.0),
    roots=((5.0, 4.0),),
    domain=((-2.0, 14.0), (-3.0, 6.0)),
    description=(
        "Moré–Garbow–Hillstrom test problem 2. The only root is (5, 4), but ½‖F‖² also has a "
        "non-zero local minimum near (11.41, −0.897), close to where J is singular "
        "(6y² − 8y − 12 = 0)."
    ),
    tags=("polynomial", "MGH", "local-minimum-trap"),
)

# --- trigonometric system ------------------------------------------------------------------
# cos x = cos y ⇒ y = ±x (mod 2π); y = x gives 2 sin x = 1 ⇒ x = π/6 or 5π/6; y = −x gives 0 = 1.

TRIG_SYSTEM = _system(
    "trig_system",
    "Trigonometric system",
    r"F(x,y) = \begin{pmatrix} \sin x + \sin y - 1 \\ \cos x - \cos y \end{pmatrix}",
    lambda v: np.array([_sin(v[0]) + _sin(v[1]) - 1.0, _cos(v[0]) - _cos(v[1])]),
    lambda v: np.array(
        [
            [_cos(v[0]), _cos(v[1])],
            [-_sin(v[0]), _sin(v[1])],
        ]
    ),
    x0=(1.0, 0.2),
    roots=((math.pi / 6.0, math.pi / 6.0), (5.0 * math.pi / 6.0, 5.0 * math.pi / 6.0)),
    domain=((-1.0, 3.5), (-1.0, 3.5)),
    description=(
        "Two roots on the diagonal, at (π/6, π/6) and (5π/6, 5π/6). det J = sin(x + y) "
        "vanishes on the line x + y = π that separates them."
    ),
    tags=("transcendental", "two-roots"),
)

# --- two intersecting circles --------------------------------------------------------------
# (x−1)² + y² = 4 and (x+1)² + (y−1)² = 4; subtracting gives y = 2x + ½, then 5x² = 11/4.

_XC = math.sqrt(0.55)

INTERSECTING_CIRCLES = _system(
    "intersecting_circles",
    "Two intersecting circles",
    r"F(x,y) = \begin{pmatrix} (x-1)^2 + y^2 - 4 \\ (x+1)^2 + (y-1)^2 - 4 \end{pmatrix}",
    lambda v: np.array(
        [
            (v[0] - 1.0) * (v[0] - 1.0) + v[1] * v[1] - 4.0,
            (v[0] + 1.0) * (v[0] + 1.0) + (v[1] - 1.0) * (v[1] - 1.0) - 4.0,
        ]
    ),
    lambda v: np.array(
        [
            [2.0 * (v[0] - 1.0), 2.0 * v[1]],
            [2.0 * (v[0] + 1.0), 2.0 * (v[1] - 1.0)],
        ]
    ),
    x0=(2.0, 2.0),
    roots=((-_XC, 0.5 - 2.0 * _XC), (_XC, 0.5 + 2.0 * _XC)),
    domain=((-3.5, 3.5), (-2.5, 3.5)),
    description=(
        "Circles of radius 2 centered at (1, 0) and (−1, 1). det J = 4(1 − x − 2y) vanishes "
        "on the line through both centers, which separates the two roots."
    ),
    tags=("polynomial", "two-roots"),
)

__all__ = [
    "CIRCLE_LINE",
    "FREUDENSTEIN_ROTH",
    "INTERSECTING_CIRCLES",
    "ROSENBROCK_SYSTEM",
    "TRIG_SYSTEM",
]
