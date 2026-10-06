"""The test-problem library. ``problems.get("rosenbrock")`` returns a ready-to-use problem."""

from . import (  # noqa: F401  (imports register the problems)
    calculus,
    combinatorial,
    constrained,
    data,
    least_squares,
    linalg,
    lp,
    roots,
    scalar_min,
    stochastic,
    systems,
    unconstrained,
)
from .registry import KINDS, add, factory, get, kind_of, list_problems

__all__ = ["KINDS", "add", "factory", "get", "kind_of", "list_problems"]
