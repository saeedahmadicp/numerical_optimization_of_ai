"""Reference values for tests/least-squares/least_squares.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/least-squares/gen_least_squares_fixture.py

It writes web/tests/least-squares/fixtures/least_squares_python.json with

* ``problems``: r, J, f, ∇f, ∇²f of every least-squares problem at its x0, its minima and six
  seeded points of its domain;
* ``runs``: full results of Gauss–Newton and Levenberg–Marquardt on cases the parity fixtures do
  not cover (other starts, parameters, budgets, failure paths: overflow, rank deficiency,
  a non-finite start, a bare residual callable without a Jacobian);
* ``presets``: the lab's first view and its "Try this" presets (src/labs/least-squares/
  presets.ts), so every claim of a preset note is checked against Python — including the
  rank-loss preset, whose end state is rounding noise (a = O(10⁻¹⁵) with a sign that differs
  between NumPy and the port, and an f that follows the sign).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems, run
from numopt.core.rng import Rng
from numopt.core.types import to_jsonable

OUT = Path(__file__).with_name("fixtures") / "least_squares_python.json"
IDS = ("exp_decay_fit", "rosenbrock_ls", "circle_fit", "michaelis_menten")


def sample_points(p: Any, seed: int) -> list[list[float]]:
    rng = Rng(seed)
    (a0, a1), (b0, b1) = p.domain
    return [[rng.uniform(a0, a1), rng.uniform(b0, b1)] for _ in range(6)]


def problem_values() -> list[dict[str, Any]]:
    out = []
    for i, pid in enumerate(IDS):
        p = problems.get(pid)
        pts = [list(p.x0), *[list(m) for m in p.minima], *sample_points(p, 100 + i)]
        for x in pts:
            xa = np.array(x, dtype=float)
            out.append(
                {
                    "id": pid,
                    "x": x,
                    "r": p.residual(xa).tolist(),
                    "J": np.asarray(p.jac(xa)).tolist(),
                    "f": float(p.f(xa)),
                    "grad": p.grad(xa).tolist(),
                    "hess": p.hess(xa).tolist(),
                }
            )
    return out


def runs() -> list[dict[str, Any]]:
    cases: list[tuple[str, str, dict[str, Any]]] = []
    for i, pid in enumerate(IDS):
        p = problems.get(pid)
        starts = sample_points(p, 200 + i)[:4]
        for x0 in starts:
            cases.append(("gauss_newton", pid, {"x0": x0}))
            cases.append(("gauss_newton", pid, {"x0": x0, "line_search": "none"}))
            cases.append(("levenberg_marquardt", pid, {"x0": x0}))
            cases.append(("levenberg_marquardt", pid, {"x0": x0, "tau": 1.0}))
        cases.append(("gauss_newton", pid, {"line_search": "none"}))
        cases.append(("gauss_newton", pid, {"max_iter": 3}))
        cases.append(("levenberg_marquardt", pid, {"max_iter": 3}))
        cases.append(("levenberg_marquardt", pid, {"tau": 10.0}))
        cases.append(("levenberg_marquardt", pid, {"tau": 1e-6, "gtol": 1e-6}))
    cases += [
        # The module docstring's examples.
        ("levenberg_marquardt", "exp_decay_fit", {"x0": [1.0, -84.0]}),
        ("gauss_newton", "exp_decay_fit", {"x0": [1.0, -84.0]}),
        ("levenberg_marquardt", "exp_decay_fit", {"x0": [0.1, -4.0]}),
        ("gauss_newton", "exp_decay_fit", {"x0": [0.1, -4.0]}),
        # Overflow at the start (f = ½‖r‖² is inf).
        ("gauss_newton", "exp_decay_fit", {"x0": [1.0, -400.0]}),
        ("levenberg_marquardt", "exp_decay_fit", {"x0": [1.0, -400.0]}),
        # Non-finite residual at the start (exp overflows to inf).
        ("gauss_newton", "exp_decay_fit", {"x0": [1.0, -1000.0]}),
        ("levenberg_marquardt", "exp_decay_fit", {"x0": [1.0, -1000.0]}),
        # Rank-deficient J: a = 0 makes the ∂r/∂b column zero.
        ("gauss_newton", "exp_decay_fit", {"x0": [0.0, 1.0]}),
        ("levenberg_marquardt", "exp_decay_fit", {"x0": [0.0, 1.0]}),
        # The circle's mirror minimum and a start on a data point.
        ("gauss_newton", "circle_fit", {"x0": [1.2, 3.5]}),
        ("levenberg_marquardt", "circle_fit", {"x0": [1.2, 3.5]}),
        ("levenberg_marquardt", "circle_fit", {"x0": [1.0, 1.0]}),
        # Rosenbrock: full Gauss–Newton step = Newton for r = 0.
        ("gauss_newton", "rosenbrock_ls", {"x0": [-1.2, 1.0], "line_search": "none"}),
        ("levenberg_marquardt", "rosenbrock_ls", {"tau": 100.0}),
        ("levenberg_marquardt", "michaelis_menten", {"tau": 1.0}),
        ("gauss_newton", "michaelis_menten", {"ftol": 1e-18}),
    ]
    out = []
    for method, pid, params in cases:
        res = run(method, problems.get(pid), **params)
        out.append(
            {"method": method, "problem": pid, "params": to_jsonable(params), "result": res.to_dict()}
        )

    # A bare residual callable without a Jacobian (central differences, counted in n_fev).
    def rosen_r(x: Any) -> Any:
        x = np.asarray(x, dtype=float)
        return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    for method in ("gauss_newton", "levenberg_marquardt"):
        res = run(method, rosen_r, x0=[-1.2, 1.0])
        out.append(
            {
                "method": method,
                "problem": "bare:rosen_r",
                "params": {"x0": [-1.2, 1.0]},
                "result": res.to_dict(),
            }
        )
    return out


# (preset id, method, problem, params) — mirrors src/labs/least-squares/presets.ts.
PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("default", "gauss_newton", "exp_decay_fit", {"x0": [0.2, 2.8]}),
    ("default", "levenberg_marquardt", "exp_decay_fit", {"x0": [0.2, 2.8]}),
    ("newton", "gauss_newton", "rosenbrock_ls", {"x0": [-1.2, 1.0], "line_search": "none"}),
    ("newton", "levenberg_marquardt", "rosenbrock_ls", {"x0": [-1.2, 1.0]}),
    ("rank", "gauss_newton", "exp_decay_fit", {"x0": [0.5, 2.8], "line_search": "none"}),
    ("rank", "levenberg_marquardt", "exp_decay_fit", {"x0": [0.5, 2.8]}),
    ("mirror", "gauss_newton", "circle_fit", {"x0": [1.2, 3.5]}),
    ("mirror", "levenberg_marquardt", "circle_fit", {"x0": [1.2, 3.5]}),
    ("scale", "gauss_newton", "michaelis_menten", {"x0": [205.0, 0.08]}),
    ("scale", "levenberg_marquardt", "michaelis_menten", {"x0": [205.0, 0.08], "tau": 1.0}),
]


def presets() -> list[dict[str, Any]]:
    out = []
    for pid, method, prob, params in PRESETS:
        res = run(method, problems.get(prob), **params)
        out.append(
            {
                "preset": pid,
                "method": method,
                "problem": prob,
                "params": to_jsonable(params),
                "result": res.to_dict(),
            }
        )
    return out


def main() -> None:
    data = {"problems": problem_values(), "runs": runs(), "presets": presets()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(data['problems'])} problem points, {len(data['runs'])} runs)")


if __name__ == "__main__":
    main()
