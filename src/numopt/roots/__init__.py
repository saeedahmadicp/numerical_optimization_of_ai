"""Root finding for scalar equations f(x) = 0 and nonlinear systems F(x) = 0."""

from typing import Any

from . import bracketing, open, systems

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(bracketing, "FIXTURE_CASES", []),
    *getattr(open, "FIXTURE_CASES", []),
    *getattr(systems, "FIXTURE_CASES", []),
]
