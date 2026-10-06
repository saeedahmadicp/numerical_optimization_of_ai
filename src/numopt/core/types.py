"""Core data types shared by every method in :mod:`numopt`.

Every method returns a :class:`Result` whose ``trace`` holds one :class:`Step` per
iteration (``k = 0`` is the starting state). The trace is the didactic record that the
web visualizer animates, so ``Step.info`` carries the geometry of the iteration
(brackets, simplices, trust radii, tableaux, ...) as JSON-serializable values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]
Scalar = float
Point = float | Vector


def as_vector(x: ArrayLike) -> Vector:
    """Return ``x`` as a fresh 1-D float64 array (a copy, never a view)."""
    arr = np.array(x, dtype=np.float64, copy=True)
    return arr.reshape(-1) if arr.ndim != 1 else arr


def to_jsonable(value: Any) -> Any:
    """Convert numpy containers and non-finite floats into plain JSON values.

    ``NaN`` becomes ``None``; ``±inf`` becomes the strings ``"inf"`` / ``"-inf"``
    (JSON has no infinity literal; the web loader maps them back).
    """
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def jsonable_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON-safe part of an ``extra`` mapping: callables (and containers of them) are dropped."""

    def ok(v: Any) -> bool:
        if callable(v):
            return False
        if isinstance(v, Mapping):
            return all(ok(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return all(ok(x) for x in v)
        return True

    return {str(k): to_jsonable(v) for k, v in extra.items() if ok(v)}


@dataclass(frozen=True)
class Step:
    """One iteration of a method.

    Attributes:
        k: Iteration index; ``0`` is the initial state before any update.
        x: The current iterate (a float for 1-D methods, a vector otherwise).
        fun: Objective value f(x) for minimization, residual f(x) or ||F(x)|| for root finding,
            or the current estimate for non-iterative numerical methods.
        grad_norm: ||∇f(x)||₂ when the method has it, else ``None``.
        step_size: The step length / trust radius / spacing used to *produce* this iterate.
        info: Method-specific, JSON-serializable geometry for the visualizer.
    """

    k: int
    x: Any
    fun: float | None
    grad_norm: float | None = None
    step_size: float | None = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "k": self.k,
                "x": self.x,
                "fun": self.fun,
                "grad_norm": self.grad_norm,
                "step_size": self.step_size,
                "info": dict(self.info),
            }
        )


@dataclass
class Result:
    """The outcome of running a method.

    ``converged`` is ``True`` only when the method's documented tolerance test passed.
    Stopping at ``max_iter``, on a non-finite value, a singular system, or a lost bracket
    gives ``converged=False`` and a message that says why.
    """

    method: str
    x: Any
    fun: float | None
    converged: bool
    message: str
    n_iter: int
    n_fev: int = 0
    n_gev: int = 0
    n_hev: int = 0
    trace: list[Step] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "x": self.x,
            "fun": self.fun,
            "converged": self.converged,
            "message": self.message,
            "n_iter": self.n_iter,
            "n_fev": self.n_fev,
            "n_gev": self.n_gev,
            "n_hev": self.n_hev,
            "extra": self.extra,
        }
        if include_trace:
            out["trace"] = [s.to_dict() for s in self.trace]
        return to_jsonable(out)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"Result({self.method}: {status} after {self.n_iter} iterations, "
            f"x={self.x!r}, fun={self.fun!r}, message={self.message!r})"
        )


# --------------------------------------------------------------------------------------
# Problem definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """A smooth problem: a scalar or vector function with optional derivatives.

    Used by the families ``roots``, ``scalar``, ``line_search``, ``unconstrained``,
    ``global``, ``systems``, ``least_squares``, ``constrained``, ``integration`` and
    ``differentiation``.

    Conventions:
        * ``dim == 1``: ``f``, ``grad`` (= f') and ``hess`` (= f'') take and return floats.
        * ``dim >= 2``: ``f(x) -> float``, ``grad(x) -> (n,)``, ``hess(x) -> (n, n)``.
        * Systems ``F(x) = 0``: ``f(x) -> (m,)`` and ``jac(x) -> (m, n)``.
        * Least squares: ``residual(x) -> (m,)``, ``jac`` is its Jacobian and
          ``f(x) = ½‖r(x)‖²``.
    """

    id: str
    name: str
    latex: str
    f: Callable[..., Any]
    dim: int
    domain: tuple[Any, ...]
    grad: Callable[..., Any] | None = None
    hess: Callable[..., Any] | None = None
    jac: Callable[..., Any] | None = None
    residual: Callable[..., Any] | None = None
    x0: Any = None
    bracket: tuple[float, float] | None = None
    minima: tuple[Any, ...] = ()
    roots: tuple[Any, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    exact: float | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Metadata only (callables are not exported)."""
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "latex": self.latex,
                "dim": self.dim,
                "domain": self.domain,
                "x0": self.x0,
                "bracket": self.bracket,
                "minima": self.minima,
                "roots": self.roots,
                "constraints": [c.to_dict() for c in self.constraints],
                "exact": self.exact,
                "description": self.description,
                "tags": self.tags,
                "extra": jsonable_extra(self.extra),
            }
        )


@dataclass(frozen=True)
class Constraint:
    """A smooth constraint. ``kind == "ineq"`` means ``fun(x) <= 0``; ``"eq"`` means ``fun(x) == 0``."""

    kind: Literal["ineq", "eq"]
    fun: Callable[[Vector], float]
    grad: Callable[[Vector], Vector]
    latex: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "latex": self.latex}


@dataclass(frozen=True)
class LinearProgram:
    """``min/max cᵀx`` s.t. ``A_ub x ≤ b_ub``, ``A_eq x = b_eq``, ``x ≥ 0`` (optionally integer)."""

    id: str
    name: str
    c: Vector
    A_ub: Vector | None = None
    b_ub: Vector | None = None
    A_eq: Vector | None = None
    b_eq: Vector | None = None
    sense: Literal["min", "max"] = "min"
    integer: tuple[bool, ...] = ()
    optimum: Vector | None = None
    optimal_value: float | None = None
    description: str = ""
    domain: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "c": self.c,
                "A_ub": self.A_ub,
                "b_ub": self.b_ub,
                "A_eq": self.A_eq,
                "b_eq": self.b_eq,
                "sense": self.sense,
                "integer": self.integer,
                "optimum": self.optimum,
                "optimal_value": self.optimal_value,
                "description": self.description,
                "domain": self.domain,
            }
        )


@dataclass(frozen=True)
class LinearSystem:
    """``A x = b`` with an optional known solution."""

    id: str
    name: str
    A: Vector
    b: Vector
    solution: Vector | None = None
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "A": self.A,
                "b": self.b,
                "solution": self.solution,
                "description": self.description,
                "tags": self.tags,
            }
        )


@dataclass(frozen=True)
class Dataset:
    """Points ``(x_i, y_i)`` for interpolation / regression, with an optional true function."""

    id: str
    name: str
    x: Vector
    y: Vector
    f_true: Callable[[Any], Any] | None = None
    latex: str = ""
    domain: tuple[float, float] | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "x": self.x,
                "y": self.y,
                "latex": self.latex,
                "domain": self.domain,
                "description": self.description,
            }
        )


ProblemLike = Problem | LinearProgram | LinearSystem | Dataset | Any

__all__ = [
    "Constraint",
    "Dataset",
    "LinearProgram",
    "LinearSystem",
    "Point",
    "Problem",
    "ProblemLike",
    "Result",
    "Scalar",
    "Sequence",
    "Step",
    "Vector",
    "as_vector",
    "jsonable_extra",
    "to_jsonable",
]
