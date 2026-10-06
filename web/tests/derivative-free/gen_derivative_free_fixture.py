"""Reference results for tests/derivative-free/derivative_free.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/derivative-free/gen_derivative_free_fixture.py
    (cd web && npx prettier --write tests/derivative-free/fixtures)

It writes web/tests/derivative-free/fixtures/derivative_free_python.json with

* ``runs``: results (every info key of the first 20 (n > 2: 4) and the last 2 Steps) of the four derivative-free methods of
  ``numopt.unconstrained.derivative_free`` on cases the parity fixtures do not cover: more library
  problems, explicit x0, max_iter stops, the extreme barrier (NaN / +inf), f = -inf, a non-finite
  f(x0), a 1-D problem and dim = 1 library problems;
* ``errors``: the ValueError messages of invalid parameters.

The custom problems below are rebuilt in the TS test with the same formulas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.core.types import Problem, to_jsonable
from numopt.unconstrained import derivative_free as dfo

OUT = Path(__file__).with_name("fixtures") / "derivative_free_python.json"
HEAD, TAIL = 20, 2
METHODS = {
    "nelder_mead": dfo.nelder_mead,
    "powell": dfo.powell,
    "hooke_jeeves": dfo.hooke_jeeves,
    "compass_search": dfo.compass_search,
}


def _barrier(x: Any) -> float:
    # NaN left of x = -0.25, +inf above y = 2.6: both are "+inf" for the methods.
    if x[0] < -0.25:
        return math.nan
    if x[1] > 2.6:
        return math.inf
    return (x[0] - 1.0) ** 2 + 3.0 * (x[1] - 2.5) ** 2 + 0.5 * x[0] * x[1]


def _unbounded(x: Any) -> float:
    if x[1] < -1.0:
        return -math.inf
    return x[0] ** 2 + x[1]


def _one_d(x: Any) -> float:
    # A custom 1-D objective that accepts x as a shape-(1,) array or as a plain float. (The
    # library's dim = 1 problems take floats; the methods pass them a float since the fix for
    # NumPy >= 2.5, see the runs on quadratic_1d etc. below.)
    t = float(np.asarray(x, dtype=float).reshape(-1)[0])
    return (t - 2.0) ** 2 + 0.1 * t**4


CUSTOM: dict[str, Problem] = {
    "barrier_2d": Problem(
        id="barrier_2d", name="barrier", latex="", f=_barrier, dim=2, domain=(), x0=[0.0, 0.0]
    ),
    "unbounded_2d": Problem(
        id="unbounded_2d", name="unbounded", latex="", f=_unbounded, dim=2, domain=(), x0=[0.0, 0.0]
    ),
    "nan_start": Problem(
        id="nan_start", name="nan", latex="", f=lambda x: math.nan, dim=2, domain=(), x0=[1.0, 2.0]
    ),
    "minus_inf_start": Problem(
        id="minus_inf_start",
        name="-inf",
        latex="",
        f=lambda x: -math.inf,
        dim=2,
        domain=(),
        x0=[1.0, 2.0],
    ),
    "one_d": Problem(id="one_d", name="1-D", latex="", f=_one_d, dim=1, domain=(-3.0, 3.0), x0=0.5),
}

RUNS: list[tuple[str, str, dict[str, Any]]] = [
    ("nelder_mead", "beale", {}),
    ("nelder_mead", "booth", {}),
    ("nelder_mead", "quadratic_ill", {}),
    ("nelder_mead", "rosenbrock", {"initial_step": 0.1, "x0": [0.5, -0.5]}),
    ("nelder_mead", "rosenbrock", {"max_iter": 20}),
    ("nelder_mead", "rosenbrock_nd", {"adaptive": True, "max_iter": 3000}),
    ("nelder_mead", "quadratic_nd", {}),
    ("nelder_mead", "six_hump_camel", {"xtol": 1e-12, "ftol": 1e-12}),
    ("powell", "quadratic_bowl", {}),
    ("powell", "himmelblau", {}),
    ("powell", "booth", {}),
    ("powell", "six_hump_camel", {}),
    ("powell", "quadratic_ill", {}),
    # (powell on quadratic_nd is left out: its last sweeps run on the rounding plateau of f, where
    # numpy's BLAS product and the TS loop differ by an ulp, and it stops one sweep apart.)
    ("powell", "rosenbrock_nd", {}),
    ("powell", "rosenbrock", {"max_iter": 3}),
    ("powell", "rosenbrock", {"x0": [2.0, -1.0], "ftol": 1e-6, "xtol": 1e-4}),
    ("hooke_jeeves", "rosenbrock", {}),
    ("hooke_jeeves", "beale", {"step": 0.25, "shrink": 0.3}),
    ("hooke_jeeves", "quadratic_nd", {}),
    ("hooke_jeeves", "rosenbrock", {"max_iter": 30}),
    ("compass_search", "rosenbrock", {}),
    ("compass_search", "booth", {}),
    ("compass_search", "himmelblau", {"shrink": 0.3, "x0": [-1.0, -1.0]}),
    ("compass_search", "quadratic_nd", {"step": 1.0}),
]
for _m in METHODS:
    RUNS += [
        (_m, "barrier_2d", {}),
        (_m, "unbounded_2d", {}),
        (_m, "nan_start", {}),
        (_m, "minus_inf_start", {}),
    ]
RUNS += [
    ("nelder_mead", "one_d", {}),
    ("powell", "one_d", {}),
    ("hooke_jeeves", "one_d", {}),
    ("compass_search", "one_d", {}),
]
# dim = 1 library problems (regression: Python raised TypeError on them; the methods now pass a
# float to the scalar callables). Problems whose f calls log/exp/sin are left out for powell: V8
# and glibc can round those one ulp apart, and powell then stops a sweep earlier or later.
RUNS += [(_m, "quadratic_1d", {}) for _m in METHODS]
RUNS += [
    ("nelder_mead", "sin_1d", {}),
    ("powell", "quartic_1d", {}),
    ("hooke_jeeves", "rational_1d", {}),
    ("compass_search", "abs_shifted", {}),
]

ERRORS: list[tuple[str, str, dict[str, Any]]] = [
    ("nelder_mead", "rosenbrock", {"max_iter": 0}),
    ("nelder_mead", "rosenbrock", {"max_iter": 2.5}),
    ("nelder_mead", "rosenbrock", {"xtol": 0.0}),
    ("nelder_mead", "rosenbrock", {"ftol": -1e-3}),
    ("nelder_mead", "rosenbrock", {"initial_step": 0.0}),
    ("nelder_mead", "one_d", {"adaptive": True}),
    ("nelder_mead", "rosenbrock", {"x0": [1.0, 2.0, 3.0]}),
    ("powell", "rosenbrock", {"ftol": 0.0}),
    ("powell", "rosenbrock", {"xtol": -1.0}),
    ("powell", "rosenbrock", {"max_iter": -3}),
    ("hooke_jeeves", "rosenbrock", {"shrink": 1.0}),
    ("hooke_jeeves", "rosenbrock", {"step": 0.0}),
    ("compass_search", "rosenbrock", {"shrink": 0.0}),
    ("compass_search", "rosenbrock", {"xtol": 0.0}),
]


def get(pid: str) -> Any:
    return CUSTOM[pid] if pid in CUSTOM else problems.get(pid)


def main() -> None:
    runs = []
    for method, pid, params in RUNS:
        res = to_jsonable(METHODS[method](get(pid), **params).to_dict())
        # Keep the file small: the first HEAD steps and the last TAIL steps of every trace.
        trace = res.pop("trace")
        res["trace_len"] = len(trace)
        head = HEAD if get(pid).dim <= 2 else 4  # n-D traces are large (Powell: n lines per step)
        res["trace_head"] = trace[:head]
        res["trace_tail"] = trace[-TAIL:] if len(trace) > head else []
        runs.append({"method": method, "problem": pid, "params": params, "result": res})
    errors = []
    for method, pid, params in ERRORS:
        try:
            METHODS[method](get(pid), **params)
        except ValueError as exc:
            errors.append({"method": method, "problem": pid, "params": params, "message": str(exc)})
        else:
            raise AssertionError(f"{method} {pid} {params}: no ValueError")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runs": runs, "errors": errors}, allow_nan=False) + "\n")
    print(f"wrote {OUT} ({len(runs)} runs, {len(errors)} errors)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
