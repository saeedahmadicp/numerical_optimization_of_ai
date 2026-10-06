"""Reference results for tests/pdhg/pdhg.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/pdhg/gen_pdhg_reference.py

It writes web/tests/pdhg/fixtures/pdhg_reference.json with results of ``restarted_pdhg`` on the
cases the parity fixtures (src/generated/fixtures/lp.json) do not cover: every library LP, every
restart scheme, primal weight and preconditioner, a primal start x0, budgets on infeasible and
unbounded programs, and the ValueErrors. To keep the file small, each trace is cut to its first
12 steps; ``restarts`` lists every k with a restart and ``n_trace`` the full trace length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from numopt import problems, run
from numopt.core.types import to_jsonable

OUT = Path(__file__).with_name("fixtures") / "pdhg_reference.json"
M = "restarted_pdhg"
LP_IDS = [p.id for p in problems.list_problems("lp")]
STUCK = {"unbounded_2d", "infeasible_2d"}

CASES: list[tuple[str, dict[str, Any]]] = []
for pid in LP_IDS:
    budget = {"max_iter": 400} if pid in STUCK else {}
    CASES.append((pid, dict(budget)))
    CASES.append((pid, {"restart": "fixed", **budget}))
    CASES.append((pid, {"restart": "none", "max_iter": 500}))
    CASES.append((pid, {"primal_weight": "unit", **budget}))
    CASES.append((pid, {"primal_weight": "adaptive", **budget}))
    CASES.append((pid, {"precondition": "none", "max_iter": 3000}))
CASES += [
    ("wyndor", {"restart_check_every": 1}),
    ("wyndor", {"restart_check_every": 8, "beta": 0.6}),
    ("diet_2d", {"restart": "fixed", "restart_period": 16}),
    ("wyndor", {"x0": [1.0, 2.0]}),
    ("wyndor", {"x0": [-3.0, 9.0]}),
    ("diet_2d", {"x0": [10.0, 10.0], "tol": 1e-5}),
    ("klee_minty_3", {"tol": 1e-4}),
    ("transport_small", {"max_iter": 7}),
    ("wyndor", {"tol": 1e3}),
    # ValueErrors
    ("wyndor", {"restart": "sometimes"}),
    ("wyndor", {"primal_weight": "heavy"}),
    ("wyndor", {"precondition": "ruiz"}),
    ("wyndor", {"beta": 1.0}),
    ("wyndor", {"restart_period": 0}),
    ("wyndor", {"tol": 0.0}),
    ("wyndor", {"x0": [1.0]}),
]

KEEP = 12


def main() -> None:
    out = []
    for pid, params in CASES:
        entry: dict[str, Any] = {"method": M, "problem": pid, "params": params}
        try:
            res = run(M, problems.get(pid), **params)
            d = res.to_dict()
            entry["restarts"] = [s.k for s in res.trace if s.info["restarted"]]
            entry["n_trace"] = len(res.trace)
            entry["last"] = to_jsonable(res.trace[-1].to_dict() if hasattr(res.trace[-1], "to_dict") else d["trace"][-1])
            d["trace"] = d["trace"][:KEEP]
            entry["result"] = d
        except ValueError as e:
            entry["error"] = str(e)
        out.append(entry)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(out), ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {len(out)} cases to {OUT}")


if __name__ == "__main__":
    main()
