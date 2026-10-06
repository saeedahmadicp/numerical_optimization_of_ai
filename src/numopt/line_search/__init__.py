"""Line searches: choose a step length α along a descent direction p."""

from typing import Any

from . import methods

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(methods, "FIXTURE_CASES", []),
]
