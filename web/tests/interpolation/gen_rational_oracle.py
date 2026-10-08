"""Oracle for tests/interpolation/rational.test.ts: Python AAA and Floater–Hormann results on
cases the six parity fixtures do not cover (every dataset, both scalings, tol / max_terms
limits, real poles, data given as an (x, y) pair, input errors), and the pole helpers
``barycentric_poles`` / ``real_denominator_roots`` on fixed weight vectors.

Run from the repo root:  .venv/bin/python web/tests/interpolation/gen_rational_oracle.py
"""
import json
import math
from pathlib import Path

import numpy as np

import numopt
from numopt import problems
from numopt.interpolation.methods import _barycentric_eval
from numopt.interpolation.rational import barycentric_poles, real_denominator_roots

OUT = Path(__file__).parent / "fixtures" / "rational_oracle.json"


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
    if isinstance(v, complex):
        return [v.real, v.imag]
    return v


def thin(info):
    """Every 7th grid value of the 200-point curves keeps the file small."""
    return {k: (np.asarray(v)[::7] if k == "curve" else v) for k, v in info.items()}


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


TIE_RTOL = 1e-9


def ties(problem, res, tol=1e-13):
    """AAA steps whose outcome rounding decides: the greedy pick (two samples within TIE_RTOL
    of the largest error), the sign of w (two entries within TIE_RTOL of the largest |w_j|) and
    the stop (a sample error at the rounding level, ≤ 1e3·ε·max|f|, within a factor 100 of the
    threshold tol·max|f|). The TS test compares steps up to the first rounding-decided pick or
    stop, and w up to sign. Steps with a weight that is 0 in exact arithmetic (zero_w) have
    rounding-decided poles."""
    data = problem if isinstance(problem, tuple) else (problem.x, problem.y)
    z_all, f_all = np.asarray(data[0], float), np.asarray(data[1], float)
    pick, sign, zero_w, w_cond, stop = [], [], [], [0.0], []
    f_scale = float(np.max(np.abs(f_all)))
    atol = float(tol) * f_scale
    for k in range(1, len(res.trace)):
        prev = res.trace[k - 1].info
        used = [s.info["node_index"] for s in res.trace[1:k]]
        cand = np.array([i for i in range(z_all.size) if i not in used])
        if k == 1:
            r = np.full(cand.size, np.mean(f_all))
        else:
            r = _barycentric_eval(np.asarray(prev["support"]), np.asarray(prev["weights"]),
                                  np.asarray(prev["support_values"]), z_all[cand])
        e = np.abs(f_all[cand] - r)
        if np.sum(e >= (1 - TIE_RTOL) * e.max()) > 1:
            pick.append(k)
        a = np.abs(np.asarray(res.trace[k].x))
        if np.sum(a >= (1 - TIE_RTOL) * a.max()) > 1:
            sign.append(k)
        # Condition of w: eps·σ_1/(σ_{m-1} − σ_m) of the scaled Loewner matrix (the bound on
        # the angle error of a computed singular vector; Golub & Van Loan 2013, §8.6.1).
        cand_k = np.array([i for i in range(z_all.size) if i not in used + [res.trace[k].info["node_index"]]])
        zs, fs = np.asarray(res.trace[k].info["support"]), np.asarray(res.trace[k].info["support_values"])
        cmat = 1.0 / (z_all[cand_k, None] - zs[None, :]) if cand_k.size else np.zeros((0, zs.size))
        loew = f_all[cand_k, None] * cmat - cmat * fs[None, :] if cand_k.size else cmat
        cn = np.linalg.norm(loew, axis=0) if res.extra.get("scaling", "columns") == "columns" else np.ones(zs.size)
        cn[~(cn > 0)] = 1.0
        sv = np.linalg.svd(loew / cn, compute_uv=False) if loew.size else np.zeros(0)
        if cand_k.size >= zs.size and zs.size >= 2:
            gap = sv[-2] - sv[-1]
            w_cond.append(float(np.finfo(float).eps * sv[0] / gap) if gap > 0 else math.inf)
        else:
            w_cond.append(0.0)
        # Weights that are 0 in exact arithmetic come out as 0 or as rounding noise; a
        # noise weight is a support point with its own sign, so it can add a real pole.
        if np.any((a > 0) & (a < 1e-12 * a.max())) or np.any(a == 0):
            zero_w.append(k)
        err = float(res.trace[k].info["sample_error"])
        if err <= 1e3 * np.finfo(float).eps * f_scale and atol / 100 <= err <= 100 * atol:
            stop.append(k)
    return {"pick": pick, "sign": sign, "zero_w": zero_w, "w_cond": w_cond, "stop": stop}


cases = []


def add(name, method, problem, problem_ref, **params):
    try:
        res = numopt.run(method, problem, **params)
        out = result(res)
        if method == "aaa":
            out["ties"] = clean(ties(problem, res, params.get("tol", 1e-13)))
        cases.append({"name": name, "method": method, "problem": problem_ref, "params": params,
                      "result": out})
    except Exception as e:  # noqa: BLE001
        cases.append({"name": name, "method": method, "problem": problem_ref, "params": params,
                      "error": f"{type(e).__name__}: {e}"})


def ds(pid):
    return problems.get(pid), {"id": pid}


def pair(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (x, y), {"x": x.tolist(), "y": y.tolist()}


DATASETS = ["runge_equispaced", "runge_chebyshev", "sine_samples", "step_data", "noisy_linear",
            "noisy_quadratic", "anscombe_1", "outliers_linear", "exponential_growth"]

for pid in DATASETS:
    add(f"aaa/{pid}", "aaa", *ds(pid))
    add(f"aaa/{pid}/none", "aaa", *ds(pid), scaling="none")
    for d in (0, 1, 3, 5, 8):
        add(f"fh/{pid}/d={d}", "floater_hormann", *ds(pid), d=d)

add("aaa tol=1e-3", "aaa", *ds("runge_equispaced"), tol=1e-3)
add("aaa max_terms=2", "aaa", *ds("sine_samples"), max_terms=2)
add("aaa max_terms=1", "aaa", *ds("noisy_linear"), max_terms=1)
x40 = np.linspace(-1, 1, 40)
add("aaa |x| 40", "aaa", *pair(x40, np.abs(x40)))
add("aaa tanh 40", "aaa", *pair(x40, np.tanh(8 * x40)))
add("aaa runge 40", "aaa", *pair(x40, 1 / (1 + 25 * x40**2)))
xr = np.array([0.0, 0.13, 0.29, 0.41, 0.6, 0.77, 0.91, 1.0, 1.24, 1.5])
add("aaa unsorted", "aaa", *pair(xr[::-1], np.exp(-xr[::-1]) * np.cos(4 * xr[::-1])))
add("aaa one point", "aaa", *pair([0.5], [2.0]))
add("aaa two points", "aaa", *pair([0.0, 1.0], [1.0, 3.0]))
add("aaa constant", "aaa", *pair([0, 1, 2, 3], [2, 2, 2, 2]))
add("aaa duplicate", "aaa", *pair([0, 1, 1], [1, 2, 3]))
add("aaa bad tol", "aaa", *ds("sine_samples"), tol=-1.0)
add("aaa bad max_terms", "aaa", *ds("sine_samples"), max_terms=0)
add("aaa bad scaling", "aaa", *ds("sine_samples"), scaling="rows")
add("fh one point", "floater_hormann", *pair([0.5], [2.0]), d=0)
add("fh d > n", "floater_hormann", *pair([0, 1, 2], [1, 0, 1]), d=3)
add("fh unsorted", "floater_hormann", *pair(xr[::-1], np.sin(3 * xr[::-1])), d=2)
add("fh runge 40 d=4", "floater_hormann", *pair(x40, 1 / (1 + 25 * x40**2)), d=4)
add("fh duplicate", "floater_hormann", *pair([0, 1, 1], [1, 2, 3]), d=1)

# The pole helpers on fixed (z, w, f): random weights give real poles in most gaps.
rng = np.random.default_rng(7)
helpers = []
for m in (3, 5, 8, 12):
    z = np.sort(rng.uniform(-1, 1, m))
    w = rng.standard_normal(m)
    f = rng.standard_normal(m)
    poles, res = barycentric_poles(z, w, f)
    br = real_denominator_roots(z, w, -1.0, 1.0)
    helpers.append(clean({"z": z, "w": w, "f": f, "poles": [complex(p) for p in poles],
                          "residues": [complex(r) for r in res], "brackets": br}))
# Odd data on symmetric support points: Σ w_j = 0, one vanishing moment.
z = np.linspace(-1, 1, 6)
w = np.array([1.0, -2.0, 1.5, -1.5, 2.0, -1.0])
poles, res = barycentric_poles(z, w, z**3)
helpers.append(clean({"z": z, "w": w, "f": z**3, "poles": [complex(p) for p in poles],
                      "residues": [complex(r) for r in res],
                      "brackets": real_denominator_roots(z, w, -1.0, 1.0)}))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"cases": cases, "helpers": helpers}))
print(f"{len(cases)} cases, {len(helpers)} helper sets -> {OUT} ({OUT.stat().st_size // 1024} KB)")
