"""Extra Python reference values for the systems port (web/tests/systems/*.test.ts).

Run from the repo root:  .venv/bin/python web/tests/systems/fixtures/gen_systems_extra.py

Writes systems_extra.json next to this file:
  values: F and J of every systems problem at x0, its roots and seeded points;
  runs:   method runs that the exported fixtures do not cover (singular Jacobian, budget,
          divergence, finite-difference Jacobians, every problem's default start);
  cond:   np.linalg.cond of a few matrices;
  full:   every iterate of the long runs the lab's presets show (Newton on freudenstein_roth
          takes 43 steps, Broyden on rosenbrock_system 100), where the port must replay NumPy's
          rounding (fused multiply-adds in OpenBLAS) to stay on the same path.
"""

import json
import math
from pathlib import Path

import numpy as np

import numopt
from numopt import problems
from numopt.core.rng import Rng


def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


values = []
rng = Rng(7)
for p in problems.list_problems("systems"):
    pts = [list(p.x0), *[list(r) for r in p.roots]]
    (x0, x1), (y0, y1) = p.domain
    for _ in range(4):
        pts.append([x0 + (x1 - x0) * rng.random(), y0 + (y1 - y0) * rng.random()])
    for x in pts:
        v = np.array(x, dtype=float)
        values.append({"id": p.id, "x": x, "F": p.f(v), "J": p.jac(v)})

RUNS = [
    ("newton_system", "circle_line", {"x0": [1.0, -1.0]}),  # J singular at x0 (x + y = 0)
    ("newton_system", "rosenbrock_system", {"max_iter": 3}),  # budget
    ("newton_system", "freudenstein_roth", {}),
    ("newton_system", "intersecting_circles", {}),
    ("newton_system", "trig_system", {}),
    ("newton_system", "trig_system", {"x0": [2.0, 1.0], "damping": True}),
    ("newton_system", "circle_line", {"x0": [-2.5, 2.5], "damping": True}),
    ("broyden", "rosenbrock_system", {}),
    ("broyden", "freudenstein_roth", {}),
    ("broyden", "trig_system", {}),
    ("broyden", "circle_line", {"x0": [0.6, -0.6], "jacobian0": "finite_difference"}),
    ("broyden", "circle_line", {"x0": [1.0, -1.0]}),  # initial J singular
    ("broyden", "intersecting_circles", {"x0": [-3.0, 3.0], "max_iter": 4}),
    ("broyden", "freudenstein_roth", {"x0": [2.0, 1.386]}),
]
runs = []
for method, pid, params in RUNS:
    r = numopt.run(method, problems.get(pid), **params)
    runs.append(
        {
            "method": method,
            "problem": pid,
            "params": params,
            "x": r.x,
            "converged": r.converged,
            "message": r.message,
            "n_iter": r.n_iter,
            "n_fev": r.n_fev,
            "n_gev": r.n_gev,
            "extra": r.extra,
            "xs": [s.x for s in r.trace[:10]],
        }
    )

mats = [
    [[4.0, 4.0], [-1.0, 1.0]],
    [[1.0, 2.0], [2.0, 4.0]],
    [[1e8, 1.0], [0.0, 1e-8]],
    [[2.0, -2.0], [-1.0, 1.0 + 1e-15]],
    [[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 5.0]],
]
cond = [{"A": A, "cond": float(np.linalg.cond(np.array(A)))} for A in mats]

FULL = [
    ("newton_system", "freudenstein_roth", {}),
    ("newton_system", "freudenstein_roth", {"damping": True}),
    ("broyden", "rosenbrock_system", {}),
    ("newton_system", "rosenbrock_system", {}),
    ("newton_system", "trig_system", {"x0": [-1.0, 1.25]}),
    ("newton_system", "trig_system", {"x0": [-1.0, 1.25], "damping": True}),
    ("broyden", "intersecting_circles", {}),
    ("newton_system", "intersecting_circles", {}),
    ("broyden", "freudenstein_roth", {}),
    ("broyden", "trig_system", {}),
]
full = []
for method, pid, params in FULL:
    r = numopt.run(method, problems.get(pid), **params)
    full.append(
        {
            "method": method,
            "problem": pid,
            "params": params,
            "converged": r.converged,
            "n_iter": r.n_iter,
            "xs": [s.x for s in r.trace],
        }
    )

out = Path(__file__).with_name("systems_extra.json")
out.write_text(json.dumps(jsonable({"values": values, "runs": runs, "cond": cond, "full": full}), indent=1))
print(f"wrote {out} ({len(values)} values, {len(runs)} runs)")
