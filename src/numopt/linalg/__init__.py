"""Direct and iterative solvers for linear systems A x = b."""

from typing import Any

from . import direct, iterative

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(direct, "FIXTURE_CASES", []),
    *getattr(iterative, "FIXTURE_CASES", []),
]
