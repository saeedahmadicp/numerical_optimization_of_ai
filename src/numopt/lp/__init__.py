"""Linear and integer programming."""

from typing import Any

from . import (
    integer,
    interior_point,
    pdhg,
    simplex,
)

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(simplex, "FIXTURE_CASES", []),
    *getattr(interior_point, "FIXTURE_CASES", []),
    *getattr(integer, "FIXTURE_CASES", []),
    *getattr(pdhg, "FIXTURE_CASES", []),
]
