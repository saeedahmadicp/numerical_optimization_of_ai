"""Oracle for tests/regression/oracle.test.ts: Python results on cases the parity fixtures do
not cover (every method on every regression dataset, data-mode inputs, rank deficiency, large x
offsets, exact fits, messages, extra statistics, info payloads, invalid input).

Run from the repo root:  .venv/bin/python web/tests/regression/gen_oracle.py
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
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def result(res):
    extra = dict(res.extra)
    if "eval" in extra:
        extra["eval"] = {k: np.asarray(v)[::13] for k, v in extra["eval"].items()}
    trace = res.trace
    # Long IRLS traces: keep the first 12 steps and the last one.
    keep = trace if len(trace) <= 13 else [*trace[:12], trace[-1]]
    return clean(
        {
            "x": res.x,
            "fun": res.fun,
            "converged": res.converged,
            "message": res.message,
            "n_iter": res.n_iter,
            "n_fev": res.n_fev,
            "trace_len": len(trace),
            "trace": [
                {"k": s.k, "x": s.x, "fun": s.fun, "step_size": s.step_size, "info": s.info}
                for s in keep
            ],
            "extra": extra,
        }
    )


cases = []


def add(name, method, data, **params):
    """`data` is a problem id (str) or an (x, y) pair."""
    prob = problems.get(data) if isinstance(data, str) else (np.asarray(data[0]), np.asarray(data[1]))
    entry = {
        "name": name,
        "method": method,
        "problem": data if isinstance(data, str) else None,
        "data": None if isinstance(data, str) else clean([list(data[0]), list(data[1])]),
        "params": params,
    }
    try:
        entry["result"] = result(numopt.run(method, prob, **params))
    except (ValueError, TypeError) as e:
        entry["error"] = str(e)
    cases.append(entry)


DATASETS = ["noisy_linear", "noisy_quadratic", "anscombe_1", "outliers_linear", "exponential_growth"]
for ds in DATASETS:
    add(f"ols qr {ds}", "linear_regression", ds)
    add(f"ols svd {ds}", "linear_regression", ds, solver="svd")
    add(f"ols ne {ds}", "linear_regression", ds, solver="normal_equations")
    add(f"huber {ds}", "huber_regression", ds)
    add(f"lad {ds}", "lad_regression", ds)
    add(f"theil-sen {ds}", "theil_sen", ds)
    add(f"minimax {ds}", "chebyshev_minimax_line", ds)
    for d in (0, 1, 3, 6, 10):
        add(f"poly d={d} {ds}", "polynomial_regression", ds, degree=d)
    for lam in (1e-6, 1.0, 100.0):
        add(f"ridge d=4 lam={lam} {ds}", "ridge_regression", ds, lam=lam, degree=4)

# Interpolation datasets used by the lab's degree sweep.
for ds in ["runge_equispaced", "sine_samples", "step_data"]:
    for d in (2, 8, 10, 12):
        add(f"poly d={d} {ds}", "polynomial_regression", ds, degree=d)
    add(f"huber {ds}", "huber_regression", ds)
    add(f"lad {ds}", "lad_regression", ds)
    add(f"minimax {ds}", "chebyshev_minimax_line", ds)

# Rank deficiency and minimum-norm solutions.
add("poly d=15 anscombe", "polynomial_regression", "anscombe_1", degree=15)
add("poly d=4 three points", "polynomial_regression", ([0.0, 1.0, 2.0], [1.0, 3.0, 2.0]), degree=4)
add("poly d=3 repeated x", "polynomial_regression", ([1.0, 1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0, 1.0]), degree=3)
add("ols qr all x equal", "linear_regression", ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
add("ols svd all x equal", "linear_regression", ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]), solver="svd")
add("ols single point", "linear_regression", ([1.0], [2.0]))
add("ridge lam=0 rank deficient", "ridge_regression", ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]), lam=0.0, degree=1)
add("ridge lam=0 ols", "ridge_regression", "noisy_linear", lam=0.0, degree=1)
add("ridge d=0", "ridge_regression", "noisy_linear", lam=3.0, degree=0)
add("ridge d=12 lam=1e-8", "ridge_regression", "noisy_quadratic", lam=1e-8, degree=12)
add("ridge d=15 lam=1e4", "ridge_regression", "noisy_quadratic", lam=1e4, degree=15)

# Large x offsets: equilibration keeps QR accurate; the normal equations lose the digits.
xs = [1e8 + i for i in range(12)]
ys = [0.475 * i + 0.1 * math.sin(3.0 * i) for i in range(12)]
add("ols qr offset 1e8", "linear_regression", (xs, ys))
add("ols ne offset 1e8", "linear_regression", (xs, ys), solver="normal_equations")
add("huber offset 1e8", "huber_regression", (xs, ys))
add("lad offset 1e8", "lad_regression", (xs, ys))
add("minimax offset 1e8", "chebyshev_minimax_line", (xs, ys))
xs6 = [1e6 + i for i in range(12)]
add("minimax offset 1e6", "chebyshev_minimax_line", (xs6, ys))

# Exact fits, constant y, ties and edits the lab can make.
line = ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 5.0, 7.0, 9.0])
add("huber exact line", "huber_regression", line)
add("lad exact line", "lad_regression", line)
add("theil-sen exact line", "theil_sen", line)
add("minimax exact line", "chebyshev_minimax_line", line)
add("ols constant y", "linear_regression", ([0.0, 1.0, 2.0], [0.2, 0.2, 0.2]))
add("theil-sen repeated x", "theil_sen", ([1.0, 1.0, 2.0, 3.0, 3.0], [0.0, 2.0, 1.0, 5.0, 4.0]))
add("theil-sen all x equal", "theil_sen", ([1.0, 1.0], [0.0, 2.0]))
add("huber all x equal", "huber_regression", ([1.0, 1.0, 1.0], [0.0, 2.0, 1.0]))
add("minimax repeated x", "chebyshev_minimax_line", ([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 0.0]))
add("minimax too few", "chebyshev_minimax_line", ([0.0, 1.0], [0.0, 1.0]))
add("minimax unsorted", "chebyshev_minimax_line", ([3.0, 0.0, 5.0, 1.0, 4.0, 2.0], [2.0, 0.5, 1.0, 1.5, 4.0, -1.0]))
add("huber delta 0.5", "huber_regression", "outliers_linear", delta=0.5)
add("huber delta 10", "huber_regression", "outliers_linear", delta=10.0)
add("huber max_iter 3", "huber_regression", "outliers_linear", max_iter=3)
add("lad eps 1e-2", "lad_regression", "outliers_linear", eps=1e-2)
add("lad max_iter 20", "lad_regression", "outliers_linear", max_iter=20)
add("minimax max_iter 1", "chebyshev_minimax_line", "noisy_linear", max_iter=1)
add("minimax outliers", "chebyshev_minimax_line", "outliers_linear")
add("huber small", "huber_regression", ([0.0, 1.0, 2.0, 3.0], [0.0, 1.1, 1.9, 9.0]))
add("lad two points", "lad_regression", ([0.0, 1.0], [0.0, 1.0]))
add("ols nan", "linear_regression", ([0.0, float("nan")], [0.0, 1.0]))
add("ols length", "linear_regression", ([0.0, 1.0, 2.0], [0.0, 1.0]))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cases, indent=None, separators=(",", ":")))
print(f"wrote {len(cases)} cases to {OUT}")
