"""Least-squares regression and goodness of fit."""

from typing import Any

from . import methods

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(methods, "FIXTURE_CASES", []),
]
