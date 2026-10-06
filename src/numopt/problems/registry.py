"""Registry of named test problems, keyed by id and grouped by kind.

Kinds mirror the method families that consume them (``scalar`` problems serve the
``roots``, ``scalar``, ``line_search``, ``integration`` and ``differentiation`` families).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_PROBLEMS: dict[str, tuple[str, Any]] = {}

KINDS = (
    "roots",  # scalar f(x) = 0 problems with brackets / starting points
    "systems",  # F(x) = 0 in R^n (mostly R^2)
    "scalar_min",  # 1-D minimization on an interval (also line-search demos)
    "unconstrained",  # n-D smooth minimization (mostly 2-D for plotting), also global methods
    "least_squares",  # residual problems r(x), f = ½‖r‖²
    "stochastic",  # finite-sum ML objectives
    "constrained",  # smooth constrained problems
    "lp",  # linear / integer programs
    "combinatorial",  # knapsack, TSP instances
    "linalg",  # linear systems A x = b
    "calculus",  # integrands and functions to differentiate
    "data",  # datasets for interpolation and regression
)


def add(kind: str, problem: T) -> T:
    """Register ``problem`` (which must have an ``id``) under ``kind`` and return it."""
    if kind not in KINDS:
        raise ValueError(f"unknown problem kind {kind!r}")
    pid = problem.id  # type: ignore[attr-defined]
    if pid in _PROBLEMS:
        raise ValueError(f"problem id {pid!r} is already registered")
    _PROBLEMS[pid] = (kind, problem)
    return problem


def get(id: str) -> Any:
    """Return the problem with this id."""
    try:
        return _PROBLEMS[id][1]
    except KeyError:
        raise KeyError(f"unknown problem {id!r}; run `numopt problems` to list them") from None


def kind_of(id: str) -> str:
    return _PROBLEMS[id][0]


def list_problems(kind: str | None = None) -> list[Any]:
    return [p for k, p in _PROBLEMS.values() if kind is None or k == kind]


def factory(kind: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Decorator: register the problem built by a zero-argument function at import time."""

    def deco(build: Callable[[], T]) -> Callable[[], T]:
        add(kind, build())
        return build

    return deco
