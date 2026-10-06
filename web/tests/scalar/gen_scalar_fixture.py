"""Reference values for tests/scalar/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/scalar/gen_scalar_fixture.py
    (cd web && npx prettier --write tests/scalar/fixtures)   # keeps `npm run format:check` green

It writes web/tests/scalar/fixtures/scalar_python.json with

* ``problems``: f, f', f'' of every scalar_min problem at its x0, its minima, its bracket ends
  and seeded points of its domain (plus a few points outside the domain);
* ``runs``: full results (every step, every info key, messages and counts) of every scalar
  method on every scalar_min problem with default parameters, plus parameter variations and
  failure paths that the parity fixtures do not cover;
* ``errors``: inputs for which Python raises ValueError (the message is checked loosely);
* ``fibonacci``: fibonacci_numbers(target, n_max) for a few targets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from numopt import problems
from numopt.core.rng import Rng
from numopt.core.types import to_jsonable
from numopt.scalar import methods as sm

OUT = Path(__file__).with_name("fixtures") / "scalar_python.json"

METHODS = [
    "golden_section",
    "fibonacci_search",
    "dichotomous_search",
    "ternary_search",
    "parabolic_interpolation",
    "brent_minimize",
    "newton_1d",
    "bracket_minimum",
]

# (method, problem, kwargs): variations beyond the defaults.
EXTRA: list[tuple[str, str, dict[str, Any]]] = [
    ("golden_section", "quadratic_1d", {"xtol": 1e-3}),
    ("golden_section", "quadratic_1d", {"max_iter": 5}),
    ("golden_section", "x_log_x", {"bracket": [-1.0, 2.0]}),
    ("golden_section", "quartic_1d", {"bracket": [-2.2, 2.2]}),
    ("fibonacci_search", "quadratic_1d", {"xtol": 0.1, "eps_ratio": 0.1}),
    ("fibonacci_search", "sin_1d", {"max_iter": 10}),
    ("fibonacci_search", "flat_valley", {"xtol": 1e-12}),
    ("fibonacci_search", "quadratic_1d", {"xtol": 1e-12, "eps_ratio": 0.001}),
    ("dichotomous_search", "drug_concentration", {"xtol": 1e-10, "delta_ratio": 0.001}),
    ("dichotomous_search", "quadratic_1d", {"xtol": 1e-8}),
    ("dichotomous_search", "sin_1d", {"max_iter": 4}),
    ("ternary_search", "abs_shifted", {"xtol": 1e-10}),
    ("parabolic_interpolation", "rational_1d", {"bracket": [-1.0, 6.0]}),
    ("parabolic_interpolation", "abs_shifted", {"xtol": 1e-10}),
    ("parabolic_interpolation", "quadratic_1d", {"xtol": 1e-12}),
    ("parabolic_interpolation", "sin_1d", {"max_iter": 3}),
    ("brent_minimize", "drug_concentration", {"xtol": 1e-12, "rtol": 1e-10}),
    ("brent_minimize", "abs_shifted", {}),
    ("brent_minimize", "x_log_x", {"bracket": [-0.5, 2.0]}),
    ("brent_minimize", "quadratic_1d", {"max_iter": 2}),
    ("newton_1d", "quartic_1d", {"x0": 0.1}),
    ("newton_1d", "quartic_1d", {"x0": 0.0, "alpha0": 0.01}),
    ("newton_1d", "x_log_x", {"x0": 1.5}),
    ("newton_1d", "abs_shifted", {"x0": -0.5, "max_iter": 12}),
    ("newton_1d", "flat_valley", {"x0": 0.5}),
    ("newton_1d", "drug_concentration", {"x0": 6.0}),
    ("newton_1d", "sin_1d", {"x0": 1.5707963267948966}),
    ("bracket_minimum", "quartic_1d", {"x0": 0.0, "step": 0.01}),
    ("bracket_minimum", "x_log_x", {"x0": 1.5, "step": 0.5}),
    ("bracket_minimum", "rational_1d", {"x0": 5.0, "step": -0.1, "grow_limit": 2.0}),
    ("bracket_minimum", "multimodal_1d", {"x0": 2.7, "step": 1.0}),
    ("bracket_minimum", "flat_valley", {"x0": 0.5, "max_iter": 3}),
]

ERRORS: list[tuple[str, str, dict[str, Any]]] = [
    ("golden_section", "quadratic_1d", {"bracket": [1.0, 1.0]}),
    ("golden_section", "quadratic_1d", {"xtol": 0.0}),
    ("fibonacci_search", "quadratic_1d", {"eps_ratio": 1.0}),
    ("dichotomous_search", "quadratic_1d", {"xtol": 1e-10, "delta_ratio": 1e-6}),
    ("dichotomous_search", "quadratic_1d", {"bracket": [0.0, 1e-7], "xtol": 1e-6}),
    ("ternary_search", "quadratic_1d", {"max_iter": 0}),
    ("brent_minimize", "quadratic_1d", {"rtol": -1.0}),
    ("newton_1d", "quadratic_1d", {"x0": math.inf}),
    ("bracket_minimum", "quadratic_1d", {"step": 1e-20, "x0": 1e5}),
    ("bracket_minimum", "quadratic_1d", {"grow_limit": 1.0}),
]


def _run(method: str, problem: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    fn = sm.__dict__[method]
    res = fn(problems.get(problem), **kwargs)
    return {"method": method, "problem": problem, "params": kwargs, "result": res.to_dict()}


def main() -> None:
    rng = Rng(7)
    probs = []
    for p in problems.list_problems("scalar_min"):
        lo, hi = p.domain
        xs = [
            p.x0,
            *p.minima,
            *p.bracket,
            lo,
            hi,
            *(lo + (hi - lo) * rng.random() for _ in range(6)),
        ]
        xs += [lo - 0.5, -1.0]
        rows = []
        for x in xs:
            x = float(x)
            rows.append({"x": x, "f": p.f(x), "grad": p.grad(x), "hess": p.hess(x)})
        probs.append({"id": p.id, "values": rows})

    runs = []
    for p in problems.list_problems("scalar_min"):
        for m in METHODS:
            runs.append(_run(m, p.id, {}))
    for m, p, kw in EXTRA:
        runs.append(_run(m, p, kw))

    errors = []
    for m, p, kw in ERRORS:
        try:
            sm.__dict__[m](problems.get(p), **kw)
        except ValueError as e:
            errors.append({"method": m, "problem": p, "params": kw, "error": str(e)})
        else:
            raise AssertionError(f"{m} {p} {kw} did not raise")

    fib = [
        {"target": t, "n_max": n, "fib": [float(v) for v in sm.fibonacci_numbers(t, n_max=n)]}
        for t, n in [(1.0, None), (10.0, None), (1e6, None), (1e6, 10), (math.inf, 40)]
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"problems": probs, "runs": runs, "errors": errors, "fibonacci": fib}
    OUT.write_text(json.dumps(to_jsonable(payload), separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
