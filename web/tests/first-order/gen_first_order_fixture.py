"""Reference values for tests/first-order/first_order.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/first-order/gen_first_order_fixture.py
    (cd web && npx prettier --write tests/first-order/fixtures)   # keeps `npm run format:check` green

It writes web/tests/first-order/fixtures/first_order_python.json with

* ``runs``: results of ``numopt.unconstrained.first_order`` methods on cases the parity fixtures
  do not cover (failure paths, max_iter, other problems and start points, n-D problems, a bare
  callable without a gradient). Each run keeps the first 12 steps, the last step and the totals.
* ``errors``: the ValueError messages of invalid input.

The custom problems below (``linear_2d``, ``rosen_nograd``) are rebuilt in the TS test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.core.types import Problem, to_jsonable
from numopt.unconstrained import first_order as fo

OUT = Path(__file__).with_name("fixtures") / "first_order_python.json"

LINEAR = Problem(
    id="linear_2d",
    name="linear",
    latex="x + 2y",
    f=lambda x: float(x[0] + 2.0 * x[1]),
    grad=lambda x: np.array([1.0, 2.0]),
    hess=lambda x: np.zeros((2, 2)),
    dim=2,
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    x0=(0.0, 0.0),
)


def _rosen(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _tiny_gradient(c: float) -> Problem:
    """f(x) = a·(x₁ + x₂) + ½c‖x‖² with a = 1e-148: each Barzilai–Borwein step moves ∇f by
    y ≈ c·1e-138 per entry, so the unscaled yᵀy underflows to 0 for c ≲ 1e-24 while sᵀy > 0
    (Python raised ZeroDivisionError before the power-of-2 scaling of ``_bb_steps``). Plain float
    arithmetic, so the TS test rebuilds it with the same operations."""
    a = 1e-148
    return Problem(
        id=f"tiny_gradient_{c!r}",
        name="tiny gradient",
        latex="",
        f=lambda x: (
            a * (float(x[0]) + float(x[1]))
            + 0.5 * c * (float(x[0]) * float(x[0]) + float(x[1]) * float(x[1]))
        ),
        grad=lambda x: a + c * np.asarray(x, dtype=float),
        dim=2,
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        x0=(0.0, 0.0),
    )


def _scaled(pid: str, e: int) -> Problem:
    """2ᵉ·f with ∇f and ∇²f scaled to match (exact: a power of 2)."""
    p = problems.get(pid)
    f, g, h = p.f, p.grad, p.hess
    assert g is not None and h is not None
    c = 2.0**e
    return dataclasses.replace(
        p,
        f=lambda x: c * f(x),
        grad=lambda x: c * np.asarray(g(x)),
        hess=lambda x: c * np.asarray(h(x)),
    )


def _problem(pid: str) -> Any:
    if pid == "linear_2d":
        return LINEAR
    if pid == "rosen_nograd":
        return _rosen
    if pid.startswith("tiny_gradient_"):
        return _tiny_gradient(float(pid.removeprefix("tiny_gradient_")))
    if "_x2^" in pid:
        base, e = pid.split("_x2^")
        return _scaled(base, int(e))
    return problems.get(pid)


RUNS: list[tuple[str, str, dict[str, Any]]] = [
    # gradient descent: every rule beyond the fixtures, and every failure path
    ("gradient_descent", "rosenbrock", {"max_iter": 300}),
    ("gradient_descent", "rosenbrock", {"step_rule": "strong_wolfe", "max_iter": 200}),
    ("gradient_descent", "beale", {"step_rule": "strong_wolfe"}),
    ("gradient_descent", "quadratic_bowl", {"step_rule": "exact_quadratic"}),
    ("gradient_descent", "himmelblau", {"step_rule": "exact_quadratic"}),
    ("gradient_descent", "goldstein_price", {"step_rule": "fixed"}),
    ("gradient_descent", "quadratic_bowl", {"step_rule": "fixed", "max_iter": 40}),
    ("gradient_descent", "quadratic_bowl", {"x0": [1.0, -0.5]}),
    ("gradient_descent", "linear_2d", {}),
    ("gradient_descent", "linear_2d", {"step_rule": "strong_wolfe", "max_iter": 60}),
    ("gradient_descent", "rosen_nograd", {"x0": [-1.2, 1.0], "max_iter": 30}),
    ("gradient_descent", "rosenbrock_nd", {"max_iter": 50}),
    # Barzilai–Borwein
    ("barzilai_borwein", "quadratic_bowl", {}),
    ("barzilai_borwein", "rosenbrock", {"variant": "bb2"}),
    ("barzilai_borwein", "rosenbrock", {"nonmonotone": False, "max_iter": 200}),
    ("barzilai_borwein", "himmelblau", {"x0": [0.5, -3.0]}),
    ("barzilai_borwein", "linear_2d", {"max_iter": 300}),
    ("barzilai_borwein", "quadratic_nd", {"max_iter": 100}),
    # yᵀy underflows to 0 with sᵀy > 0 (regression: Python raised ZeroDivisionError)
    ("barzilai_borwein", "tiny_gradient_1e-24", {"gtol": 1e-150, "max_iter": 6}),
    ("barzilai_borwein", "tiny_gradient_3e-25", {"variant": "bb2", "gtol": 1e-150, "max_iter": 6}),
    (
        "barzilai_borwein",
        "tiny_gradient_1e-163",
        {"nonmonotone": False, "gtol": 1e-150, "max_iter": 20},
    ),
    # f scaled by 2⁴⁹⁶: the strong Wolfe zoom divides by an h² that underflows (regression:
    # ZeroDivisionError escaped from gradient_descent on beale)
    ("gradient_descent", "beale_x2^496", {"step_rule": "strong_wolfe", "max_iter": 5000}),
    ("gradient_descent", "booth_x2^496", {"step_rule": "strong_wolfe", "max_iter": 5000}),
    # momentum methods
    ("momentum", "quadratic_bowl", {"max_iter": 300}),
    ("momentum", "rosenbrock", {"lr": 0.01, "max_iter": 300}),
    ("momentum", "booth", {"lr": 0.2, "beta": 0.5, "max_iter": 300}),
    ("nesterov", "rosenbrock", {"max_iter": 300}),
    ("nesterov", "rosenbrock", {"lr": 0.01, "max_iter": 300}),
    ("nesterov", "quadratic_bowl", {"beta": 0.0, "max_iter": 50}),
    # adaptive methods
    ("adagrad", "rosenbrock", {"max_iter": 300}),
    ("adagrad", "himmelblau", {"lr": 0.5, "max_iter": 300}),
    ("rmsprop", "quadratic_bowl", {}),
    ("rmsprop", "rosenbrock", {"lr": 0.01, "max_iter": 200}),
    ("adadelta", "beale", {"rho": 0.9, "eps": 1e-4, "max_iter": 300}),
    ("adam", "rosenbrock", {"max_iter": 300}),
    ("adam", "quadratic_nd", {"lr": 0.1, "max_iter": 150}),
    ("adamw", "quadratic_ill", {"max_iter": 300}),
    ("adamw", "himmelblau", {"weight_decay": 0.0, "max_iter": 200}),
    ("adamax", "rosenbrock", {"max_iter": 300}),
    ("adamax", "booth", {"beta2": 0.0, "max_iter": 200}),
    ("nadam", "himmelblau", {"max_iter": 300}),
    ("nadam", "rosenbrock", {"max_iter": 300}),
    ("amsgrad", "rosenbrock", {"max_iter": 300}),
    ("amsgrad", "beale", {"lr": 0.05, "max_iter": 300}),
    # coordinate descent
    ("coordinate_descent", "quadratic_bowl", {}),
    ("coordinate_descent", "rosenbrock", {"max_iter": 300}),
    ("coordinate_descent", "himmelblau", {"x0": [0.0, 0.0], "max_iter": 60}),
    ("coordinate_descent", "booth", {"x0": [1.0, 3.0]}),
    ("coordinate_descent", "rosenbrock_nd", {"max_iter": 100}),
    ("coordinate_descent", "rosen_nograd", {"x0": [-1.2, 1.0], "max_iter": 40}),
]

ERRORS: list[tuple[str, str, dict[str, Any]]] = [
    ("gradient_descent", "rosenbrock", {"step_rule": "newton"}),
    ("gradient_descent", "rosenbrock", {"gtol": 0.0}),
    ("gradient_descent", "rosenbrock", {"max_iter": 0}),
    ("gradient_descent", "rosenbrock", {"max_iter": 2.5}),
    ("gradient_descent", "rosenbrock", {"lr": -1.0}),
    ("gradient_descent", "rosenbrock", {"x0": [1.0, 2.0, 3.0]}),
    ("barzilai_borwein", "rosenbrock", {"variant": "bb3"}),
    ("momentum", "rosenbrock", {"beta": 1.0}),
    ("nesterov", "rosenbrock", {"lr": 0.0}),
    ("adagrad", "rosenbrock", {"eps": 0.0}),
    ("rmsprop", "rosenbrock", {"rho": -0.5}),
    ("adadelta", "rosenbrock", {"rho": 1.0}),
    ("adam", "rosenbrock", {"beta2": 1.0}),
    ("adamw", "rosenbrock", {"weight_decay": -0.001}),
    ("adamax", "rosenbrock", {"beta1": 1.5}),
    ("nadam", "rosenbrock", {"eps": -1e-8}),
    ("amsgrad", "rosenbrock", {"lr": 0.0}),
    ("coordinate_descent", "rosenbrock", {"gtol": 1e-200}),
]


def _trimmed(result: Any) -> dict[str, Any]:
    out = result.to_dict()
    trace = out.pop("trace")
    out["head"] = trace[:12]
    out["last"] = trace[-1]
    out["n_steps"] = len(trace)
    return out


def main() -> None:
    runs = []
    for method_id, pid, params in RUNS:
        fn = getattr(fo, method_id)
        result = fn(_problem(pid), **params)
        runs.append(
            {"method": method_id, "problem": pid, "params": to_jsonable(params), **_trimmed(result)}
        )
    errors = []
    for method_id, pid, params in ERRORS:
        fn = getattr(fo, method_id)
        try:
            fn(_problem(pid), **params)
        except ValueError as e:
            errors.append(
                {
                    "method": method_id,
                    "problem": pid,
                    "params": to_jsonable(params),
                    "error": str(e),
                }
            )
        else:
            raise AssertionError(f"{method_id} {params}: no ValueError")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"runs": runs, "errors": errors}, indent=1, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    print(f"wrote {OUT} ({len(runs)} runs, {len(errors)} errors)")


if __name__ == "__main__":
    main()
