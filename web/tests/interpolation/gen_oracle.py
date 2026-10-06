"""Oracle for tests/interpolation/oracle.test.ts: Python results on cases the parity fixtures
do not cover (data mode, tiny n, unsorted data, overflow, messages, info payloads).

Run from the repo root:  .venv/bin/python web/tests/interpolation/gen_oracle.py
"""
import json
import math
from pathlib import Path

import numpy as np

import numopt
from numopt import problems

OUT = Path(__file__).parent / "fixtures" / "oracle.json"


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def thin(info):
    """Every 7th grid value of the 200-point curves keeps the file small."""
    return {k: (np.asarray(v)[::7] if k in ("curve", "basis") else v) for k, v in info.items()}


def result(res):
    extra = dict(res.extra)
    if "eval" in extra:
        ev = extra["eval"]
        extra["eval"] = {k: (None if v is None else np.asarray(v)[::7]) for k, v in ev.items()}
    return clean({
        "x": res.x, "fun": res.fun, "converged": res.converged, "message": res.message,
        "n_iter": res.n_iter, "n_fev": res.n_fev,
        "trace": [{"k": s.k, "x": s.x, "fun": s.fun, "info": thin(s.info)} for s in res.trace],
        "extra": extra,
    })


cases = []


def add(name, method, problem, problem_ref, **params):
    try:
        res = numopt.run(method, problem, **params)
        cases.append({"name": name, "method": method, "problem": problem_ref, "params": params,
                      "result": result(res)})
    except Exception as e:  # noqa: BLE001
        cases.append({"name": name, "method": method, "problem": problem_ref, "params": params,
                      "error": f"{type(e).__name__}: {e}"})


def ds(pid):
    return problems.get(pid), {"id": pid}


def pair(x, y):
    return (np.array(x, float), np.array(y, float)), {"x": list(map(float, x)), "y": list(map(float, y))}


ALL = ["lagrange", "barycentric", "newton_divided_differences", "neville", "linear_spline",
       "cubic_spline_natural", "cubic_spline_clamped", "cubic_spline_not_a_knot", "pchip",
       "chebyshev_interpolation"]

for m in ALL:
    for pid in ["runge_equispaced", "runge_chebyshev", "sine_samples", "step_data",
                "anscombe_1", "noisy_linear", "exponential_growth"]:
        add(f"{m}/{pid}", m, *ds(pid))

add("cheb n_nodes=25", "chebyshev_interpolation", *ds("runge_equispaced"), n_nodes=25)
add("cheb n_nodes ignored", "chebyshev_interpolation", *ds("noisy_linear"), n_nodes=7)
add("cheb n_nodes ignored (pair)", "chebyshev_interpolation", *pair([0, 1, 3], [1, 2, 0]), n_nodes=5)
add("neville x_frac=0", "neville", *ds("sine_samples"), x_frac=0.0)
add("neville x_frac=0.5", "neville", *ds("runge_chebyshev"), x_frac=0.5)
add("neville bad x_frac", "neville", *ds("sine_samples"), x_frac=1.5)
add("clamped big slopes", "cubic_spline_clamped", *pair([0, 1, 2, 3], [0, 0, 0, 0]), fprime_a=20.0, fprime_b=-3.5)
for m in ["cubic_spline_natural", "cubic_spline_not_a_knot", "pchip", "linear_spline", "cubic_spline_clamped"]:
    add(f"{m} n=2", m, *pair([0, 2], [1, 3]))
    add(f"{m} n=3", m, *pair([0, 1, 3], [1, -1, 2]))
    add(f"{m} n=1", m, *pair([0], [1]))
for m in ["lagrange", "barycentric", "newton_divided_differences", "neville", "chebyshev_interpolation"]:
    add(f"{m} n=1", m, *pair([0.5], [2.0]))
    add(f"{m} duplicate", m, *pair([0, 1, 1], [1, 2, 3]))
add("pchip flat+extremum", "pchip", *pair([0, 1, 2, 3, 4, 5], [0, 1, 1, 3, 2, 2]))
add("pchip unsorted", "pchip", *pair([3, 0, 2, 1], [1, 0, 4, 2]))
add("natural unsorted", "cubic_spline_natural", *pair([3, 0, 2, 1], [1, 0, 4, 2]))
xs = np.linspace(0, 1000, 400)
add("barycentric overflow", "barycentric", *pair(xs, np.sin(xs)))
x70 = np.linspace(-1, 1, 70)
add("newton 70 nodes", "newton_divided_differences", *pair(x70, np.cos(3 * x70)))
add("lagrange 40 nodes", "lagrange", *pair(np.linspace(-1, 1, 40), np.cos(3 * np.linspace(-1, 1, 40))))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cases))
print(f"{len(cases)} cases -> {OUT} ({OUT.stat().st_size // 1024} KB)")
