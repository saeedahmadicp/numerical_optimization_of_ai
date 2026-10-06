"""Reference values for tests/stochastic/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/stochastic/gen_stochastic_fixture.py
    (cd web && npx prettier --write tests/stochastic/fixtures)

It writes web/tests/stochastic/fixtures/stochastic_python.json with

* ``problems``: the data (X, y), metadata, and f, ∇f, ∇²f, a mini-batch gradient and per-sample
  gradients of every stochastic problem at its x0, its minimizer and four seeded points;
* ``schedules``: ``learning_rate`` for every schedule on a grid of updates;
* ``runs``: full results (with traces) of cases the parity fixtures do not cover: every method on
  every problem, record_every > 1, b > 32 (no batch indices), full batches, a converged run, a
  start-converged run, divergence, a non-finite start, and every lr schedule;
* ``errors``: invalid inputs that raise ValueError / TypeError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import numopt
from numopt import problems
from numopt.core.rng import Rng
from numopt.core.types import to_jsonable
from numopt.stochastic import methods as sm

OUT = Path(__file__).with_name("fixtures") / "stochastic_python.json"
IDS = ("linreg_2d", "logreg_2d", "ill_conditioned_ls", "huber_regression_2d")
METHODS = (
    "sgd",
    "sgd_momentum",
    "sgd_nesterov",
    "stochastic_adagrad",
    "stochastic_rmsprop",
    "stochastic_adam",
    "svrg",
    "saga",
    "sag",
)


def problem_data() -> list[dict[str, Any]]:
    out = []
    for pid in IDS:
        p = problems.get(pid)
        rng = Rng(99)
        (lo0, hi0), (lo1, hi1) = p.domain
        pts = [list(p.x0), list(p.minima[0])] + [
            [rng.uniform(lo0, hi0), rng.uniform(lo1, hi1)] for _ in range(4)
        ]
        idx = [7, 3, 150, 3, 99]
        evals = [
            {
                "w": w,
                "f": p.f(w),
                "grad": p.grad(w),
                "hess": p.hess(w),
                "grad_batch": p.grad_batch(w, idx),
                "grad_samples": p.grad_samples(w, idx),
            }
            for w in pts
        ]
        out.append({**p.to_dict(), "idx": idx, "evals": evals})
    return out


def schedules() -> list[dict[str, Any]]:
    out = []
    for s in sm.SCHEDULES:
        for U, T in ((20, 400), (7, 21), (1, 3), (200, 1000)):
            out.append(
                {
                    "schedule": s,
                    "U": U,
                    "T": T,
                    "values": [sm.learning_rate(s, 0.3, t, U, T) for t in range(T)],
                }
            )
    return out


def runs() -> list[dict[str, Any]]:
    cases: list[tuple[str, str, dict[str, Any]]] = []
    small = {"epochs": 3, "batch_size": 25}
    for m in METHODS:
        for pid in IDS:
            lr = 1e-4 if pid == "ill_conditioned_ls" else 0.05
            cases.append((m, pid, {**small, "lr": lr, "seed": 3}))
    cases += [
        ("sgd", "linreg_2d", {"epochs": 10, "batch_size": 5, "lr": 0.1}),  # automatic every = 2
        ("sgd", "linreg_2d", {"epochs": 4, "batch_size": 7, "record_every": 3}),
        ("sgd", "linreg_2d", {"epochs": 3, "batch_size": 40}),  # b > 32: batch = None
        ("sgd", "linreg_2d", {"epochs": 3, "batch_size": 500}),  # full batch
        ("svrg", "linreg_2d", {"epochs": 20, "batch_size": 5, "lr": 0.1, "record_every": 20}),
        ("saga", "logreg_2d", {"epochs": 30, "batch_size": 10, "lr": 0.5, "record_every": 25}),
        ("sgd", "linreg_2d", {"epochs": 2, "gtol": 50.0}),  # converged at the start
        ("sgd", "ill_conditioned_ls", {"epochs": 3, "lr": 0.05}),  # diverges (f blow-up)
        ("sag", "linreg_2d", {"epochs": 20, "batch_size": 5, "lr": 0.1, "record_every": 40}),
        ("sgd", "linreg_2d", {"epochs": 2, "x0": [1e200, 1e200]}),  # non-finite start
        ("sgd", "linreg_2d", {"epochs": 2, "x0": [1e150, 1e150], "lr": 1.0}),  # overflow
        ("stochastic_adam", "logreg_2d", {"epochs": 8, "batch_size": 1, "record_every": 100}),
    ]
    for s in sm.SCHEDULES:
        cases.append(
            ("sgd_momentum", "huber_regression_2d", {"epochs": 8, "lr_schedule": s, "record_every": 8})
        )
    out = []
    for m, pid, params in cases:
        r = numopt.run(m, problems.get(pid), **params)
        out.append({"method": m, "problem": pid, "params": to_jsonable(params), "result": r.to_dict()})
    return out


def errors() -> list[dict[str, Any]]:
    cases = [
        ("sgd", {"lr": 0.0}),
        ("sgd", {"lr": float("inf")}),
        ("sgd", {"batch_size": 0}),
        ("sgd", {"batch_size": 2.5}),
        ("sgd", {"epochs": 0}),
        ("sgd", {"record_every": -1}),
        ("sgd", {"gtol": -1.0}),
        ("sgd", {"lr_schedule": "linear"}),
        ("sgd", {"x0": [1.0, 2.0, 3.0]}),
        ("sgd", {"x0": [float("nan"), 0.0]}),
        ("sgd_momentum", {"momentum": 1.0}),
        ("stochastic_rmsprop", {"rho": -0.1}),
        ("stochastic_adam", {"beta2": 1.0}),
        ("stochastic_adagrad", {"eps": 0.0}),
    ]
    out = []
    for m, params in cases:
        try:
            numopt.run(m, problems.get("linreg_2d"), **params)
        except (ValueError, TypeError) as e:
            out.append({"method": m, "params": to_jsonable(params), "error": str(e)})
        else:
            raise AssertionError(f"{m} {params} did not raise")
    return out


def main() -> None:
    data = {
        "problems": problem_data(),
        "schedules": schedules(),
        "runs": runs(),
        "errors": errors(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_jsonable(data)))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
