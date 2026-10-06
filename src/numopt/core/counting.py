"""Evaluation counting and problem resolution helpers used by every method."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .types import Problem, as_vector


class Counted:
    """Wrap a callable and count its calls: ``fc = Counted(f); fc(x); fc.n``."""

    __slots__ = ("fn", "n")

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.n = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.n += 1
        return self.fn(*args, **kwargs)


def scalar_problem(problem: Problem | Callable[[float], float], **fields: Any) -> Problem:
    """Accept a :class:`Problem` or a bare ``f(x) -> float`` and return a 1-D Problem."""
    if isinstance(problem, Problem):
        return problem
    if not callable(problem):
        raise TypeError("problem must be a numopt Problem or a callable f(x)")
    return Problem(
        id="custom",
        name="custom",
        latex="f(x)",
        f=problem,
        dim=1,
        domain=fields.pop("domain", (-1.0, 1.0)),
        **fields,
    )


def vector_problem(problem: Problem | Callable[[Any], float], **fields: Any) -> Problem:
    """Accept a :class:`Problem` or a bare ``f(x) -> float`` and return an n-D Problem."""
    if isinstance(problem, Problem):
        return problem
    if not callable(problem):
        raise TypeError("problem must be a numopt Problem or a callable f(x)")
    x0 = fields.get("x0")
    dim = int(np.size(x0)) if x0 is not None else fields.pop("dim", 2)
    fields.pop("dim", None)
    return Problem(
        id="custom",
        name="custom",
        latex="f(x)",
        f=problem,
        dim=dim,
        domain=fields.pop("domain", ()),
        **fields,
    )


def start_point(problem: Problem, x0: Any) -> np.ndarray:
    """Resolve the starting vector: the explicit ``x0`` or the problem's default."""
    if x0 is None:
        x0 = problem.x0
    if x0 is None:
        raise ValueError(f"{problem.id}: no starting point given and the problem has no default x0")
    x = as_vector(x0)
    if problem.dim and x.size != problem.dim:
        raise ValueError(f"{problem.id}: x0 has {x.size} entries, expected {problem.dim}")
    return x


def start_scalar(problem: Problem, x0: Any) -> float:
    """Resolve a scalar starting point."""
    if x0 is None:
        x0 = problem.x0
    if x0 is None:
        raise ValueError(f"{problem.id}: no starting point given and the problem has no default x0")
    return float(np.asarray(x0, dtype=float).reshape(-1)[0])


def finite(*values: Any) -> bool:
    """True when every value is finite (works for scalars and arrays)."""
    return all(bool(np.all(np.isfinite(v))) for v in values)
