"""Reference runs for tests/roots/oracle.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/roots/gen_roots_oracle.py

It writes web/tests/roots/fixtures/roots_oracle.json.gz (gzip: prettier and
reviewers skip it; 1.7 MB of JSON) with

* ``problems``: f, f', f'' of every roots problem at 21 points of its domain, its x0, its
  bracket ends and its roots;
* ``runs``: the full results (every trace step with its info, counts, message, extra) of every
  registered method of ``numopt.roots.bracketing`` and ``numopt.roots.open`` on every roots
  problem with default parameters, plus start variants, budgets, tolerances, failure paths and
  bare callables without derivatives (finite differences). A run that raises records
  ``{"error": message}`` instead.

The bare callables below (``cubic_f``, ``cubic_nograd``) are rebuilt in the TS test.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

from numopt import problems, run
from numopt.core.types import Problem, to_jsonable
from numopt.roots.bracketing import itp_n_half

OUT = Path(__file__).with_name("fixtures") / "roots_oracle.json.gz"
KEEP_HEAD, KEEP_TAIL = 20, 3

BRACKETING = (
    "bisection",
    "regula_falsi",
    "illinois",
    "pegasus",
    "anderson_bjorck",
    "ridders",
    "brent",
    "chandrupatla",
    "itp",
)
OPEN = (
    "newton",
    "secant",
    "halley",
    "steffensen",
    "muller",
    "inverse_quadratic_interpolation",
    "fixed_point",
)

CUSTOM = {
    # f only: Newton and Halley fall back to central differences.
    "cubic_nograd": Problem(
        id="cubic_nograd",
        name="cubic without derivatives",
        latex="",
        f=lambda x: (x * x - 2.0) * x - 5.0,
        dim=1,
        domain=(0.0, 3.5),
        x0=2.0,
        bracket=(2.0, 3.0),
    ),
}


def get(pid: str) -> Any:
    return CUSTOM[pid] if pid in CUSTOM else problems.get(pid)


def record(method: str, pid: str, params: dict[str, Any]) -> dict[str, Any]:
    case: dict[str, Any] = {"method": method, "problem": pid, "params": params}
    try:
        res = run(method, get(pid), **params)
    except (ValueError, TypeError) as exc:
        case["error"] = str(exc)
        return case
    out = res.to_dict()
    trace = out["trace"]
    if len(trace) > KEEP_HEAD + KEEP_TAIL:
        # Long runs: the first KEEP_HEAD and the last KEEP_TAIL steps (the test checks both ends
        # and the length).
        out["trace_length"] = len(trace)
        out["trace"] = trace[:KEEP_HEAD] + trace[-KEEP_TAIL:]
    case["result"] = out
    return case


def main() -> None:
    roots = problems.list_problems("roots")
    probs = {}
    for p in roots:
        a, b = p.domain
        xs = [a + (b - a) * i / 20 for i in range(21)] + [p.x0, *p.bracket, *p.roots]
        probs[p.id] = [[x, p.f(x), p.grad(x), p.hess(x)] for x in xs]

    cases: list[tuple[str, str, dict[str, Any]]] = []
    for p in roots:
        for m in (*BRACKETING, *OPEN):
            cases.append((m, p.id, {}))
    # Bracketing: wide and awkward brackets, tolerances, budgets, input errors.
    for m in BRACKETING:
        cases += [
            (m, "steep_exp", {"bracket": [-50.0, 60.0]}),
            (m, "x10_minus_1", {"bracket": [0.0, 1.3], "xtol": 1e-14}),
            (m, "sqrt2", {"bracket": [0.0, 2.0], "xtol": 0.0}),
            (m, "cubic", {"max_iter": 3}),
            (m, "kepler", {"ftol": 1e-6}),
            (m, "wilkinson5", {"bracket": [0.6, 2.7]}),
            (m, "double_root", {"bracket": [-3.0, 1.0]}),  # f(b) = 0 exactly
            (m, "sqrt2", {"bracket": [2.0, 3.0]}),  # no sign change
            (m, "sqrt2", {"bracket": [2.0, 1.0]}),  # a > b
            (m, "atan_newton", {"bracket": [-1e300, 1e300]}),
            (m, "cos_minus_x", {"xtol": 1e-3}),
            (m, "cubic_nograd", {}),
        ]
    cases += [
        ("itp", "cubic", {"kappa1": 1.0, "kappa2": 1.0, "n0": 0}),
        ("itp", "steep_exp", {"kappa1": 1e-4, "kappa2": 2.5, "n0": 5}),
        ("itp", "sqrt2", {"xtol": 0.0}),  # ITP needs xtol > 0
        ("itp", "cubic", {"kappa2": 2.7}),
    ]
    # Open methods: start points that show the failure modes.
    starts = {
        "atan_newton": [1.5, 1.3, -1.39, 1.3917452002707353, 10.0],
        "x10_minus_1": [0.5, -0.63, 2.0],
        "kepler": [0.3, 3.0, 2.0],
        "double_root": [2.0, 0.0, -4.0],
        "steep_exp": [4.0, -6.0, 8.0],
        "newton_cycle": [0.0, -1.0, 0.5],
        "cubic": [1.0, 0.0, -3.0],
        "wilkinson5": [0.5, 2.5, 3.6],
        "cos_minus_x": [1.0, 10.0, -3.0],
        "sqrt2": [1.0, 0.0, 100.0],
    }
    for pid, xs0 in starts.items():
        for x0 in xs0:
            for m in OPEN:
                cases.append((m, pid, {"x0": x0}))
    for lam in (-1.0, -0.5, 0.5, 1.5, -2.5):
        cases.append(("fixed_point", "cos_minus_x", {"lam": lam}))
    for lam in (0.1, 0.4, 0.6, -0.2):
        cases.append(("fixed_point", "sqrt2", {"lam": lam, "x0": 1.0}))
    for m in OPEN:
        cases += [
            (m, "cubic", {"max_iter": 2}),
            (m, "cubic", {"xtol": 1e-4, "ftol": 1e-3}),
            (m, "cubic_nograd", {}),
            (m, "kepler", {"xtol": 1e-15}),
        ]
    for m in ("secant", "muller", "inverse_quadratic_interpolation"):
        cases += [
            (m, "cubic", {"delta": 1.0}),
            (m, "steep_exp", {"delta": 1e-6}),
            (m, "sqrt2", {"delta": 0.0}),  # delta must be > 0
            (m, "kepler", {"x0": 1e17}),
        ]

    # libm references (glibc via Python) and Python number formatting.
    rnd = random.Random(7)
    xs = [rnd.uniform(-6.0, 6.0) for _ in range(3000)]
    ys = [rnd.uniform(0.0, 1.4) for _ in range(3000)]
    libm = {
        "x": xs,
        "y": ys,
        "exp": [math.exp(x) for x in xs],
        "sin": [math.sin(x) for x in xs],
        "cos": [math.cos(x) for x in xs],
        "atan": [math.atan(x) for x in xs],
        "hypot": [math.hypot(x, y) for x, y in zip(xs, ys, strict=True)],
    }
    nums = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        0.1,
        1e-5,
        1.5e-7,
        123456.789,
        1e16,
        1.5e16,
        9.999999e15,
        2.0**-1074,
        1.7976931348623157e308,
        0.30000000000000004,
        1e22,
        5e-324,
        0.0001,
        0.00012345678,
        33.0,
        1.4142135623730951,
        -1.3022231416374677,
        6.02e23,
        1e-10,
        5.82e-11,
        2.2250738585072014e-308,
        100.0,
        1234567.0,
        0.5,
        2.0**-10,
        999999.5,
    ]
    fmt = {
        "repr": [[v, repr(v)] for v in nums],
        "g3": [[v, format(v, ".3g")] for v in nums],
        "g6": [[v, format(v, ".6g")] for v in nums],
    }
    ulps = [
        [v, math.ulp(v), math.nextafter(v, math.inf), math.nextafter(v, -math.inf)]
        for v in nums
        if math.isfinite(v)
    ]
    nhalf = []
    for a, b, eps in [
        (0.0, 6.0, 1e-10),
        (2.0, 3.0, 1e-10),
        (-50.0, 60.0, 1e-10),
        (0.0, 1.0, 0.5),
        (0.0, 1.0, 0.25),
        (0.0, 1.0, 0.3),
        (-1e300, 1e300, 1e-15),
        (1.0, 1.0 + 2**-40, 1e-12),
        (0.0, 1e-300, 1e-310),
    ]:
        nhalf.append([a, b, eps, itp_n_half(a, b, eps)])

    out = {
        "libm": libm,
        "format": fmt,
        "ulp": ulps,
        "itp_n_half": nhalf,
        "problems": probs,
        "runs": [to_jsonable(record(m, p, params)) for m, p, params in cases],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(out), separators=(",", ":"), allow_nan=False)
    OUT.write_bytes(gzip.compress(text.encode(), mtime=0))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(cases)} runs)")


if __name__ == "__main__":
    main()
