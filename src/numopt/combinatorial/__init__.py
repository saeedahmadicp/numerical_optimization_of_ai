"""Combinatorial optimization: knapsack and traveling-salesman methods."""

from typing import Any

from . import knapsack, tsp

#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    *getattr(knapsack, "FIXTURE_CASES", []),
    *getattr(tsp, "FIXTURE_CASES", []),
]
