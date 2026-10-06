"""Reference results for tests/lp/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/lp/gen_lp_reference.py

It writes web/tests/lp/fixtures/lp_reference.json with full results (trace and info included)
of the LP family on the cases the parity fixtures (src/generated/fixtures/lp.json) do not
cover: every pivot rule, phase 1 with equality rows, cycling, unbounded and infeasible
programs, the zero-objective feasibility check of the interior-point methods, the integer
methods on every library problem with integer data, budgets, and the ValueErrors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from numopt import problems, run
from numopt.core.types import to_jsonable

OUT = Path(__file__).with_name("fixtures") / "lp_reference.json"

SIMPLEX_FAMILY = ("simplex", "two_phase_simplex", "big_m", "revised_simplex")
RULES = ("dantzig", "bland", "steepest_edge")
LP_IDS = [p.id for p in problems.list_problems("lp")]

CASES: list[tuple[str, str, dict[str, Any]]] = []
for method in SIMPLEX_FAMILY:
    for pid in LP_IDS:
        for rule in RULES:
            CASES.append((method, pid, {"pivot_rule": rule}))
for pid in LP_IDS:
    for rule in ("dantzig", "bland"):
        CASES.append(("dual_simplex", pid, {"pivot_rule": rule}))
CASES += [
    ("simplex", "klee_minty_3", {"max_iter": 3}),
    ("two_phase_simplex", "transport_small", {"max_iter": 2}),
    ("two_phase_simplex", "diet_2d", {"tol": 1e-4}),
    ("big_m", "diet_2d", {"max_iter": 1}),
    ("revised_simplex", "transport_small", {"max_iter": 4}),
]
for pid in LP_IDS:
    CASES.append(("primal_dual_ipm", pid, {}))
    CASES.append(("affine_scaling", pid, {}))
    CASES.append(("affine_scaling", pid, {"variant": "short", "beta": 0.5}))
    CASES.append(("branch_and_bound", pid, {}))
    CASES.append(("branch_and_bound", pid, {"strategy": "depth_first"}))
    CASES.append(("gomory_cuts", pid, {}))
CASES += [
    ("primal_dual_ipm", "wyndor", {"eta": 0.6, "tol": 1e-10}),
    ("primal_dual_ipm", "klee_minty_3", {"max_iter": 3}),
    ("affine_scaling", "wyndor", {"max_iter": 5}),
    ("affine_scaling", "diet_2d", {"big_m": 20.0}),
    ("branch_and_bound", "ilp_knapsack_like_2d", {"max_iter": 2}),
    ("branch_and_bound", "ilp_3var", {"int_tol": 0.3}),
    ("gomory_cuts", "ilp_3var", {"max_iter": 1}),
]


def main() -> None:
    out = []
    for method, pid, params in CASES:
        entry: dict[str, Any] = {"method": method, "problem": pid, "params": params}
        try:
            res = run(method, problems.get(pid), **params)
            entry["result"] = res.to_dict()
        except ValueError as e:
            entry["error"] = str(e)
        out.append(entry)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {len(out)} cases to {OUT}")


if __name__ == "__main__":
    main()
