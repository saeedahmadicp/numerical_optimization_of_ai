"""Polynomial and spline interpolation."""

from typing import Any

from . import (
    methods,
    rational,
)

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(methods, "FIXTURE_CASES", []),
    *getattr(rational, "FIXTURE_CASES", []),
]
