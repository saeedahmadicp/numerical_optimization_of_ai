"""numopt — numerical optimization and numerical methods, implemented clearly and checked carefully.

Quick start::

    import numopt
    from numopt import problems

    res = numopt.run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1.0])
    res.x, res.fun, res.converged, len(res.trace)

    numopt.find_root(lambda x: x**3 - 2, bracket=(0, 2), method="brent")
    numopt.minimize(lambda x: (x[0] - 1) ** 2 + 4 * x[1] ** 2, x0=[3, 1], method="newton")

Every method is registered with an id; ``numopt.list_methods()`` lists them all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Importing the family packages registers their methods.
from . import (  # noqa: F401
    combinatorial,
    constrained,
    differentiation,
    integration,
    interpolation,
    linalg,
    line_search,
    lp,
    problems,
    regression,
    roots,
    scalar,
    stochastic,
    unconstrained,
)
from .core import (
    FAMILIES,
    Constraint,
    Dataset,
    LinearProgram,
    LinearSystem,
    MethodSpec,
    ParamSpec,
    Problem,
    Result,
    Step,
    get_method,
    list_methods,
    run,
)
from .core.counting import scalar_problem, vector_problem

__version__ = "1.0.0"


def find_root(
    f: Callable[..., Any] | Problem,
    *,
    method: str = "brent",
    x0: Any = None,
    bracket: tuple[float, float] | None = None,
    fprime: Callable[..., Any] | None = None,
    fprime2: Callable[..., Any] | None = None,
    **params: Any,
) -> Result:
    """Find a root of a scalar function with any method of the ``roots`` family."""
    prob = scalar_problem(f, grad=fprime, hess=fprime2) if not isinstance(f, Problem) else f
    kw: dict[str, Any] = dict(params)
    if x0 is not None:
        kw["x0"] = x0
    if bracket is not None:
        kw["bracket"] = bracket
    return run(method, prob, **kw)


def minimize(
    f: Callable[..., Any] | Problem,
    x0: Any = None,
    *,
    method: str = "bfgs",
    grad: Callable[..., Any] | None = None,
    hess: Callable[..., Any] | None = None,
    **params: Any,
) -> Result:
    """Minimize a function of n variables with any method of the ``unconstrained`` family."""
    prob = f if isinstance(f, Problem) else vector_problem(f, grad=grad, hess=hess, x0=x0)
    if x0 is not None:
        params["x0"] = x0
    return run(method, prob, **params)


__all__ = [
    "FAMILIES",
    "Constraint",
    "Dataset",
    "LinearProgram",
    "LinearSystem",
    "MethodSpec",
    "ParamSpec",
    "Problem",
    "Result",
    "Step",
    "find_root",
    "get_method",
    "list_methods",
    "minimize",
    "problems",
    "run",
]
