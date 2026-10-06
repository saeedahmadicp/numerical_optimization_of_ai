"""Unconstrained minimization of smooth (or derivative-free) functions of n variables."""

from typing import Any

from . import (
    accelerated,
    anderson,
    conjugate_gradient,
    derivative_free,
    first_order,
    global_,
    least_squares,
    newton,
    quasi_newton,
    regularized_newton,
    trust_region,
)

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(first_order, "FIXTURE_CASES", []),
    *getattr(newton, "FIXTURE_CASES", []),
    *getattr(quasi_newton, "FIXTURE_CASES", []),
    *getattr(conjugate_gradient, "FIXTURE_CASES", []),
    *getattr(trust_region, "FIXTURE_CASES", []),
    *getattr(derivative_free, "FIXTURE_CASES", []),
    *getattr(global_, "FIXTURE_CASES", []),
    *getattr(least_squares, "FIXTURE_CASES", []),
    *getattr(accelerated, "FIXTURE_CASES", []),
    *getattr(anderson, "FIXTURE_CASES", []),
    *getattr(regularized_newton, "FIXTURE_CASES", []),
]
