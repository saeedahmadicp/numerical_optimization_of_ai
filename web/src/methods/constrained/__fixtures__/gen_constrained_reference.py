"""Reference runs for src/methods/constrained/methods.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/src/methods/constrained/__fixtures__/gen_constrained_reference.py

It writes constrained_reference.json next to this file: for every case the result summary
(n_iter, converged, message, x, fun, counts, extra), the first 12 steps and the last step in
full (info included), or the ValueError message; plus direct calls of ``solve_qp`` and the
problem values (f, ∇f, ∇²f, c, ∇c, ∇²c) at the start, the minima and seeded points.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.constrained import methods as cm
from numopt.core.registry import run
from numopt.core.types import to_jsonable

OUT = Path(__file__).with_name("constrained_reference.json")
METHODS = ["projected_gradient", "frank_wolfe", "quadratic_penalty", "augmented_lagrangian",
           "log_barrier", "sqp"]
PROBS = [p.id for p in problems.list_problems("constrained")]


def summary(res: Any) -> dict[str, Any]:
    tr = [to_jsonable(s.to_dict() if hasattr(s, "to_dict") else s.__dict__) for s in res.trace]
    head = tr[:12]
    return {
        "n_iter": res.n_iter, "converged": res.converged, "message": res.message,
        "x": to_jsonable(res.x), "fun": to_jsonable(res.fun),
        "n_fev": res.n_fev, "n_gev": res.n_gev, "n_hev": res.n_hev,
        "extra": to_jsonable(res.extra), "head": head, "last": tr[-1], "n_steps": len(tr),
    }


cases: list[dict[str, Any]] = []


def add(method: str, prob: str, params: dict[str, Any]) -> None:
    p = problems.get(prob)
    try:
        res = run(method, p, **params)
        cases.append({"method": method, "problem": prob, "params": to_jsonable(params),
                      "result": summary(res)})
    except ValueError as exc:
        cases.append({"method": method, "problem": prob, "params": to_jsonable(params),
                      "error": str(exc)})


for m in METHODS:
    for pid in PROBS:
        add(m, pid, {})
# Fixture-like and lab-preset cases.
add("frank_wolfe", "quadratic_disk", {"step": "armijo"})
add("frank_wolfe", "box_quadratic", {"step": "armijo"})
add("frank_wolfe", "hs21", {"step": "armijo"})
add("quadratic_penalty", "quadratic_disk", {"tol": 1e-5})
add("sqp", "hs21", {"x0": [-1.0, -1.0]})
add("projected_gradient", "box_quadratic", {"x0": [2.0, -1.4]})
add("projected_gradient", "quadratic_disk", {"x0": [-1.2, 1.2]})
add("projected_gradient", "hs21", {"x0": [-1.0, -1.0]})
add("projected_gradient", "linear_eq_quadratic", {"x0": [-0.5, -1.0]})
add("projected_gradient", "rosenbrock_unit_disk", {"s_bar": 0.01, "max_iter": 300})
add("log_barrier", "quadratic_disk", {"t0": 0.1, "mu": 4.0})
add("log_barrier", "linear_eq_quadratic", {"x0": [1.5, 1.2]})
add("log_barrier", "box_quadratic", {"x0": [0.9, 0.9]})
add("log_barrier", "quadratic_disk", {"x0": [2.0, 0.0]})
add("quadratic_penalty", "circle_eq", {"mu0": 0.1, "rho": 3.0})
add("quadratic_penalty", "box_quadratic", {"max_iter": 5})
add("augmented_lagrangian", "quadratic_disk", {"mu0": 2.0, "mu_factor": 5.0})
add("augmented_lagrangian", "halfplanes_quadratic", {"x0": [2.5, 2.5]})
add("sqp", "circle_eq", {"x0": [0.05, 0.05]})
add("sqp", "mishra_bird_constrained", {"x0": [-8.0, -2.0]})
add("sqp", "rosenbrock_disk", {"x0": [-1.2, 1.0]})
add("sqp", "quadratic_disk", {"max_iter": 2})
add("frank_wolfe", "rosenbrock_unit_disk", {"max_iter": 200})
add("frank_wolfe", "mishra_bird_constrained", {"step": "armijo", "x0": [-6.0, -1.0]})
add("projected_gradient", "box_quadratic", {"beta": 2.0})

qp_cases: list[dict[str, Any]] = []


def qp(G, a, Ae=None, be=None, Ai=None, bi=None) -> None:
    r = cm.solve_qp(G, a, Ae, be, Ai, bi)
    qp_cases.append({
        "args": to_jsonable([G, a, Ae, be, Ai, bi]),
        "ok": r.ok, "x": to_jsonable(r.x), "lam_eq": to_jsonable(r.lam_eq),
        "lam_ub": to_jsonable(r.lam_ub), "active": list(r.active), "n_iter": r.n_iter,
        "message": r.message,
    })


I2 = [[1.0, 0.0], [0.0, 1.0]]
qp(I2, [-2.0, -1.0], None, None, [[1.0, 1.0]], [1.0])
qp([[2.0, 0.5], [0.5, 1.0]], [1.0, -3.0], [[1.0, -1.0]], [0.5], [[-1.0, 0.0], [0.0, -1.0], [1.0, 2.0]], [0.0, 0.0, 2.0])
qp(I2, [0.0, 0.0], None, None, [[1.0, 0.0], [-1.0, 0.0]], [-1.0, -1.0])  # inconsistent
qp(I2, [-3.0, -3.0], None, None, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 1.0, 2.0])  # degenerate vertex
qp(I2, [-3.0, -3.0], [[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0])  # duplicated equality
qp([[1.0, 2.0], [2.0, 1.0]], [0.0, 0.0])  # not PD
qp([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]], [-1.0, 2.0, -3.0], [[1.0, 1.0, 1.0]], [1.0], [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [0.0, 0.0, 0.0])

values: dict[str, Any] = {}
rng = np.random.default_rng(7)
for pid in PROBS:
    p = problems.get(pid)
    (x0lo, x0hi), (y0lo, y0hi) = p.domain
    pts = [list(p.x0)] + [list(m) for m in p.minima] + [
        [float(rng.uniform(x0lo, x0hi)), float(rng.uniform(y0lo, y0hi))] for _ in range(4)]
    vals = []
    for x in pts:
        xa = np.array(x)
        vals.append({
            "x": x, "f": float(p.f(xa)), "grad": to_jsonable(p.grad(xa)),
            "hess": to_jsonable(p.hess(xa)),
            "c": [float(c.fun(xa)) for c in p.constraints],
            "cgrad": [to_jsonable(c.grad(xa)) for c in p.constraints],
            "chess": [to_jsonable(h(xa)) for h in p.extra["constraint_hess"]],
        })
    meta = p.to_dict() if hasattr(p, "to_dict") else {}
    extra = {k: v for k, v in p.extra.items() if k != "constraint_hess"}
    values[pid] = {"points": vals, "extra": to_jsonable(extra), "meta": to_jsonable(meta)}

OUT.write_text(json.dumps({"cases": cases, "qp": qp_cases, "problems": values}, indent=None))
print(f"wrote {OUT} ({len(cases)} cases, {len(qp_cases)} QPs)")
