"""Reference values for tests/unconstrained-promoted/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/unconstrained-promoted/gen_promoted_fixture.py

It writes web/tests/unconstrained-promoted/fixtures/promoted_python.json with full results of the
promoted unconstrained methods (accelerated.py, anderson.py, regularized_newton.py) on cases the
parity fixtures do not cover: other problems and parameters, failure paths, the ValueError
messages of invalid input, and the schedule helpers (silver steps and rates, the κ-aware block,
the auto horizon, OGM's θ).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from numopt import problems
from numopt.core.registry import run
from numopt.core.types import to_jsonable
from numopt.unconstrained import accelerated as acc

OUT = Path(__file__).with_name("fixtures") / "promoted_python.json"

#: (method, problem, params). x0 may be given in params.
CASES: list[tuple[str, str, dict[str, Any]]] = [
    # silver_gd
    ("silver_gd", "quadratic_bowl", {"max_iter": 63}),
    ("silver_gd", "booth", {"max_iter": 127}),
    ("silver_gd", "rosenbrock", {"L": 2500.0, "max_iter": 31}),
    ("silver_gd", "matyas", {"max_iter": 255}),
    # silver_gd_strongly_convex
    ("silver_gd_strongly_convex", "quadratic_bowl", {}),
    ("silver_gd_strongly_convex", "quadratic_ill", {"horizon": 8, "max_iter": 100}),
    ("silver_gd_strongly_convex", "booth", {}),
    ("silver_gd_strongly_convex", "matyas", {"horizon": 16}),
    # long_step_gd
    ("long_step_gd", "quadratic_ill", {"pattern": "2", "max_iter": 200}),
    ("long_step_gd", "booth", {"pattern": "7", "max_iter": 140}),
    ("long_step_gd", "quadratic_bowl", {"pattern": "127", "max_iter": 127}),
    # ogm
    ("ogm", "quadratic_bowl", {"max_iter": 50}),
    ("ogm", "booth", {"max_iter": 200}),
    ("ogm", "matyas", {"max_iter": 30}),
    # fista
    ("fista", "rosenbrock", {"restart": "none", "backtracking": False, "lr": 1e-3, "max_iter": 200}),
    ("fista", "himmelblau", {}),
    ("fista", "quadratic_ill", {"backtracking": False, "lr": 0.02, "restart": "function"}),
    ("fista", "booth", {"restart": "none"}),
    ("fista", "six_hump_camel", {"lr": 10.0, "eta": 3.0}),
    # anderson_gd
    ("anderson_gd", "quadratic_bowl", {"m": 0, "lr": 0.1}),
    ("anderson_gd", "himmelblau", {"m": 2, "lr": 0.005}),
    ("anderson_gd", "beale", {"lam": 0.001, "max_iter": 60}),
    ("anderson_gd", "booth", {"m": 1, "lr": 0.02}),
    ("anderson_gd", "six_hump_camel", {"x0": [0.05, 0.05], "lr": 0.01}),
    # arc
    ("arc", "beale", {}),
    ("arc", "six_hump_camel", {}),
    ("arc", "booth", {}),
    ("arc", "himmelblau", {"x0": [-0.27, -0.92]}),
    ("arc", "rosenbrock", {"sigma0": 100.0, "gamma": 3.0}),
    ("arc", "rosenbrock", {"max_iter": 5}),
    # reg_newton
    ("reg_newton", "rosenbrock", {"variant": "fixed"}),
    ("reg_newton", "himmelblau", {}),
    ("reg_newton", "quadratic_ill", {"variant": "super_universal", "alpha": 2.0 / 3.0}),
    ("reg_newton", "beale", {"H": 10.0}),
    ("reg_newton", "six_hump_camel", {"variant": "super_universal", "x0": [0.05, 0.05]}),
    ("reg_newton", "rosenbrock", {"max_iter": 5}),
]

#: Invalid input: (method, problem, params) whose ValueError message the port must repeat.
ERRORS: list[tuple[str, str, dict[str, Any]]] = [
    ("silver_gd", "rosenbrock", {}),
    ("silver_gd", "quadratic_bowl", {"L": -1.0}),
    ("silver_gd_strongly_convex", "quadratic_ill", {"L": 0.5}),
    ("silver_gd_strongly_convex", "quadratic_ill", {"horizon": 3}),
    ("long_step_gd", "rosenbrock", {"L": 10.0, "pattern": "5"}),
    ("ogm", "himmelblau", {}),
    ("fista", "booth", {"restart": "always"}),
    ("fista", "booth", {"eta": 1.0}),
    ("anderson_gd", "booth", {"lr": 0.0}),
    ("arc", "booth", {"eta1": 0.95}),
    ("reg_newton", "booth", {"variant": "cubic"}),
    ("reg_newton", "booth", {"alpha": 0.5}),
]


def main() -> None:
    cases = []
    for method, pid, params in CASES:
        result = run(method, problems.get(pid), **params)
        cases.append(
            {
                "method": method,
                "problem": pid,
                "params": to_jsonable(params),
                "result": result.to_dict(),
            }
        )
    errors = []
    for method, pid, params in ERRORS:
        try:
            run(method, problems.get(pid), **params)
        except ValueError as exc:
            errors.append({"method": method, "problem": pid, "params": params, "message": str(exc)})
        else:
            raise AssertionError(f"{method} {params} did not raise")
    helpers = {
        "silver_schedule_15": acc.silver_schedule(15).tolist(),
        "silver_rates": [acc.silver_rate(k) for k in range(8)],
        "silver_sc": [
            {
                "kappa": kappa,
                "n": n,
                "h": acc.silver_sc_schedule(kappa, n)[0].tolist(),
                "tau": acc.silver_sc_schedule(kappa, n)[1],
            }
            for kappa, n in ((1.0, 4), (1.5, 8), (10.0, 8), (100.0, 16), (1e6, 4))
        ],
        "auto_horizon": {str(k): acc.silver_sc_auto_horizon(k) for k in (1.0, 4.0, 50.0, 100.0, 1e4)},
        "ogm_thetas_6": acc.ogm_thetas(6).tolist(),
    }
    out = {"cases": cases, "errors": errors, "helpers": helpers}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(out), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cases)} cases, {len(errors)} errors)")
    assert all(math.isfinite(v) for v in helpers["silver_rates"])


if __name__ == "__main__":
    main()
