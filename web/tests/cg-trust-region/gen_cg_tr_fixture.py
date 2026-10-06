"""Reference values for tests/cg-trust-region/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/cg-trust-region/gen_cg_tr_fixture.py
    (cd web && npx prettier --write tests/cg-trust-region/fixtures)

It writes web/tests/cg-trust-region/fixtures/cg_tr_python.json with full results of the
conjugate-gradient and trust-region methods on cases the parity fixtures do not cover: n-D
problems, the exact line search, failure paths (max_iter, failed line search, non-finite values,
‖g‖² under/overflow, radius collapse), indefinite Hessians (dogleg fallback, Steihaug negative
curvature, the hard case of the exact solver), finite-difference derivatives, 1-D problems,
the radius clamp, ValueError messages for invalid input, and np.linalg.eigh of 2×2 matrices.

The custom problems below are rebuilt in the TS test (cases.test.ts) with the same formulas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.core.types import Problem
from numopt.unconstrained import conjugate_gradient as cg
from numopt.unconstrained import trust_region as tr

OUT = Path(__file__).with_name("fixtures") / "cg_tr_python.json"
HEAD = 10
TAIL = 2
#: n-D problems: the test compares only k, x and f of their steps (rounding drift, see the test).
SLIM = {"rosenbrock_nd", "quadratic_nd"}

METHODS = {
    **{
        m: getattr(cg, m)
        for m in (
            "cg_fletcher_reeves",
            "cg_polak_ribiere",
            "cg_hestenes_stiefel",
            "cg_dai_yuan",
            "cg_hager_zhang",
        )
    },
    **{
        m: getattr(tr, m)
        for m in (
            "trust_region_cauchy",
            "trust_region_dogleg",
            "trust_region_steihaug",
            "trust_region_exact",
        )
    },
}


def _rosen(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _only_at_start(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    return 2.0 if (x[0] == 1.0 and x[1] == 1.0) else math.inf


CUSTOM: dict[str, Problem] = {
    # Rosenbrock without derivatives: central-difference gradient and Hessian.
    "rosen_nograd": Problem(
        id="rosen_nograd",
        name="Rosenbrock (f only)",
        latex="",
        f=_rosen,
        dim=2,
        domain=((-2.0, 2.0), (-1.0, 3.0)),
        x0=[-1.2, 1.0],
    ),
    # A saddle: g ⊥ the eigenvector of the negative eigenvalue at x0 (the hard case).
    "saddle": Problem(
        id="saddle",
        name="saddle",
        latex="x^2 - y^2",
        f=lambda x: float(x[0] ** 2 - x[1] ** 2),
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.array([[2.0, 0.0], [0.0, -2.0]]),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[1.0, 0.0],
    ),
    # A 1-D problem in the scalar convention: f, f', f'' take and return floats.
    "quartic_1d": Problem(
        id="quartic_1d",
        name="quartic",
        latex="(x-2)^4 + x^2",
        f=lambda x: (x - 2.0) ** 4 + x * x,
        grad=lambda x: 4.0 * (x - 2.0) ** 3 + 2.0 * x,
        hess=lambda x: 12.0 * (x - 2.0) ** 2 + 2.0,
        dim=1,
        domain=(-1.0, 4.0),
        x0=-1.0,
    ),
    # Non-finite at x0.
    "log_neg": Problem(
        id="log_neg",
        name="log",
        latex="",
        f=lambda x: float(np.log(x[0]) + x[1] ** 2) if x[0] > 0 else math.nan,
        grad=lambda x: np.array([1.0 / x[0], 2.0 * x[1]]),
        hess=lambda x: np.array([[-1.0 / x[0] ** 2, 0.0], [0.0, 2.0]]),
        dim=2,
        domain=((0.1, 2.0), (-1.0, 1.0)),
        x0=[-1.0, 0.5],
    ),
    # Tiny and huge scales: ‖g‖₂² under- and overflows.
    "tiny_bowl": Problem(
        id="tiny_bowl",
        name="tiny",
        latex="",
        f=lambda x: float(1e-300 * (x[0] ** 2 + x[1] ** 2)),
        grad=lambda x: np.array([2e-300 * x[0], 2e-300 * x[1]]),
        hess=lambda x: np.array([[2e-300, 0.0], [0.0, 2e-300]]),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[1.0, 1.0],
    ),
    "huge_bowl": Problem(
        id="huge_bowl",
        name="huge",
        latex="",
        f=lambda x: float(1e300 * (x[0] ** 2 + x[1] ** 2)),
        grad=lambda x: np.array([2e300 * x[0], 2e300 * x[1]]),
        hess=lambda x: np.array([[2e300, 0.0], [0.0, 2e300]]),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[1.0, 1.0],
    ),
    # Finite only at x0: every trial step is rejected and the radius collapses.
    "only_at_start": Problem(
        id="only_at_start",
        name="only at start",
        latex="",
        f=_only_at_start,
        grad=lambda x: np.array([2.0 * x[0], 2.0 * x[1]]),
        hess=lambda x: np.array([[2.0, 0.0], [0.0, 2.0]]),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=[1.0, 1.0],
    ),
}


def _quartic_exp(pid: str, A: list, c: list, w: list, v: list, x0: list) -> Problem:
    """f(x) = ½(x − c)ᵀA(x − c) + Σ w_i x_i⁴ + Σ v_i exp(x_i), written with explicit loops so the
    TS test can repeat every operation in the same order."""
    n = len(c)

    def f(x: Any) -> float:
        d = [float(x[i]) - c[i] for i in range(n)]
        s = 0.0
        for i in range(n):
            for j in range(n):
                s += d[i] * A[i][j] * d[j]
        r = 0.5 * s
        for i in range(n):
            xi = float(x[i])
            r += w[i] * (xi * xi * xi * xi)
        for i in range(n):
            r += v[i] * math.exp(float(x[i]))
        return r

    def grad(x: Any) -> Any:
        d = [float(x[i]) - c[i] for i in range(n)]
        out = []
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i][j] * d[j]
            xi = float(x[i])
            out.append(s + 4.0 * w[i] * (xi * xi * xi) + v[i] * math.exp(xi))
        return np.array(out)

    def hess(x: Any) -> Any:
        H = [[A[i][j] for j in range(n)] for i in range(n)]
        for i in range(n):
            xi = float(x[i])
            H[i][i] = A[i][i] + (12.0 * w[i] * (xi * xi) + v[i] * math.exp(xi))
        return np.array(H)

    return Problem(id=pid, name=pid, latex="", f=f, grad=grad, hess=hess, dim=n, domain=(), x0=x0)


#: Random instances found by search: Fletcher–Reeves with the exact step meets a non-descent CG
#: direction, Hestenes–Stiefel with the exact step meets dᵀy ≤ 0 (the "breakdown" restart).
QUARTIC_DATA: dict[str, dict[str, list]] = {
    "quartic_nd4": {
        "A": [
            [1.891914552213195, 0.6007455973719564, -0.08950997657080804, 0.5282510854709683],
            [0.6007455973719564, 2.6779641910289715, 1.5039933396270706, 2.593241113330091],
            [-0.08950997657080804, 1.5039933396270706, 3.678232468332131, 2.488999198102653],
            [0.5282510854709683, 2.593241113330091, 2.488999198102653, 3.1605869655872194],
        ],
        "c": [
            -0.5006334623822092,
            0.018099797491261945,
            -2.0325002592230628,
            -0.002855936559750958,
        ],
        "w": [0.4439180282742583, 1.4439960510989136, 0.06836633815039028, 0.04515210939571812],
        "v": [0.0, 0.0, 0.0, 0.0],
        "x0": [-1.1683773702075044, 2.9098428372427407, -1.9362625841623893, 4.201988347459885],
    },
    "quartic_exp3": {
        "A": [
            [0.22058726226321498, -0.1288061430140342, -0.08047878627283564],
            [-0.1288061430140342, 0.1533189409440372, 0.04275409639684979],
            [-0.08047878627283564, 0.04275409639684979, 0.0423685570969853],
        ],
        "c": [-4.682158198777026, -0.54148450099491, -3.5057891911919628],
        "w": [0.017812199484818613, 0.6976217753129912, -0.006223244419023173],
        "v": [0.2869628040843012, 0.6112996831998543, 0.7891874540779672],
        "x0": [-0.5016336058756649, -3.3220433336700244, 0.4314702948058381],
    },
}
for _pid, _d in QUARTIC_DATA.items():
    CUSTOM[_pid] = _quartic_exp(_pid, _d["A"], _d["c"], _d["w"], _d["v"], _d["x0"])

CG_IDS = [m for m in METHODS if m.startswith("cg_")]
TR_IDS = [m for m in METHODS if m.startswith("trust_region_")]

RUNS: list[tuple[str, str, dict[str, Any]]] = [
    # n-D problems (periodic restarts, Jacobi vs LAPACK on n = 10, 20).
    *[(m, "rosenbrock_nd", {}) for m in CG_IDS],
    *[(m, "quadratic_nd", {"line_search": "exact_quadratic", "gtol": 1e-8}) for m in CG_IDS],
    *[(m, "quadratic_nd", {}) for m in CG_IDS],
    *[(m, "rosenbrock_nd", {}) for m in TR_IDS],
    *[(m, "quadratic_nd", {}) for m in TR_IDS],
    # 2-D problems the fixtures do not use.
    *[
        (m, p, {})
        for m in CG_IDS
        for p in ("booth", "matyas", "goldstein_price", "three_hump_camel")
    ],
    *[
        (m, p, {})
        for m in TR_IDS
        for p in (
            "beale",
            "six_hump_camel",
            "three_hump_camel",
            "goldstein_price",
            "booth",
            "mccormick",
        )
    ],
    # CG: max_iter, failed line search (f-based search at its rounding level), c2, exact search.
    ("cg_fletcher_reeves", "rosenbrock", {"max_iter": 3}),
    ("cg_polak_ribiere", "goldstein_price", {"gtol": 1e-12}),
    ("cg_hestenes_stiefel", "rosenbrock", {"c2": 0.45}),
    ("cg_dai_yuan", "rosenbrock", {"line_search": "exact_quadratic"}),
    ("cg_fletcher_reeves", "himmelblau", {"line_search": "exact_quadratic", "x0": [0.0, 0.0]}),
    ("cg_fletcher_reeves", "quadratic_ill", {"line_search": "exact_quadratic"}),
    # CG: custom problems.
    *[(m, "rosen_nograd", {}) for m in ("cg_polak_ribiere", "cg_hager_zhang")],
    ("cg_fletcher_reeves", "rosen_nograd", {"line_search": "exact_quadratic", "max_iter": 4}),
    *[(m, "quartic_1d", {}) for m in CG_IDS],
    ("cg_dai_yuan", "log_neg", {}),
    ("cg_fletcher_reeves", "tiny_bowl", {}),
    ("cg_fletcher_reeves", "tiny_bowl", {"gtol": 0.0}),
    ("cg_fletcher_reeves", "huge_bowl", {}),
    ("cg_hager_zhang", "saddle", {"max_iter": 5}),
    ("cg_fletcher_reeves", "quartic_nd4", {"line_search": "exact_quadratic", "max_iter": 60}),
    ("cg_hestenes_stiefel", "quartic_exp3", {"line_search": "exact_quadratic", "max_iter": 60}),
    # Trust region: max_iter, radius clamp, eta, indefinite Hessians, hard case.
    ("trust_region_dogleg", "rosenbrock", {"max_iter": 4}),
    ("trust_region_exact", "rosenbrock", {"radius0": 50.0, "max_radius": 2.0}),
    ("trust_region_steihaug", "himmelblau", {"eta": 0.0, "radius0": 0.01}),
    *[(m, "himmelblau", {"x0": [0.0, 0.0]}) for m in TR_IDS],
    *[(m, "himmelblau", {"x0": [-0.27, -0.92]}) for m in TR_IDS],
    *[(m, "saddle", {"max_iter": 6}) for m in TR_IDS],
    *[(m, "rosen_nograd", {}) for m in TR_IDS],
    *[(m, "quartic_1d", {}) for m in TR_IDS],
    ("trust_region_exact", "log_neg", {}),
    ("trust_region_cauchy", "tiny_bowl", {}),
    *[(m, "tiny_bowl", {"gtol": 0.0}) for m in TR_IDS],
    *[(m, "huge_bowl", {}) for m in TR_IDS],
    ("trust_region_cauchy", "only_at_start", {}),
    ("trust_region_exact", "only_at_start", {}),
]

ERRORS: list[tuple[str, str, dict[str, Any]]] = [
    ("cg_fletcher_reeves", "rosenbrock", {"line_search": "backtracking"}),
    ("cg_fletcher_reeves", "rosenbrock", {"c2": 1e-5}),
    ("cg_fletcher_reeves", "rosenbrock", {"c2": 1.0}),
    ("cg_fletcher_reeves", "rosenbrock", {"gtol": -1.0}),
    ("cg_fletcher_reeves", "rosenbrock", {"max_iter": 0}),
    ("cg_fletcher_reeves", "rosenbrock", {"x0": [1.0, 2.0, 3.0]}),
    ("trust_region_dogleg", "rosenbrock", {"eta": 0.25}),
    ("trust_region_dogleg", "rosenbrock", {"radius0": 0.0}),
    ("trust_region_dogleg", "rosenbrock", {"max_radius": math.inf}),
    ("trust_region_dogleg", "rosenbrock", {"gtol": -1e-3}),
    ("trust_region_dogleg", "rosenbrock", {"max_iter": 2.5}),
]


def eigh2_cases() -> list[dict[str, Any]]:
    """np.linalg.eigh of symmetric 2×2 matrices: special cases, extreme scales, random ones."""
    rng = np.random.default_rng(1)
    mats = [
        [[1330.0, 480.0], [480.0, 200.0]],
        [[2.0, 0.0], [0.0, -2.0]],
        [[1.0, 1e-20], [1e-20, 1.0]],
        [[-3.0, 2.0], [2.0, -3.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[5.0, 5.0], [5.0, 5.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[1e-130, 3e-131], [3e-131, 2e-130]],
        [[1e200, 3e199], [3e199, -2e200]],
        [[1e-300, 1e-301], [1e-301, 3e-300]],
        [[2e300, 0.0], [0.0, 2e300]],
    ]
    for _ in range(60):
        a = rng.normal(size=(2, 2)) * 10.0 ** rng.uniform(-8, 8)
        mats.append(((a + a.T) / 2).tolist())
    out = []
    for m in mats:
        w, Q = np.linalg.eigh(np.array(m))
        out.append({"A": m, "w": w.tolist(), "Q": Q.tolist()})
    return out


def problem_of(pid: str) -> Problem:
    return CUSTOM[pid] if pid in CUSTOM else problems.get(pid)


def main() -> None:
    runs = []
    for method, pid, params in RUNS:
        result = METHODS[method](problem_of(pid), **params)
        res = result.to_dict()
        trace = res.pop("trace")
        # Long traces: keep the first HEAD and the last TAIL steps (the test checks those, every
        # count, the message and the final point); short ones are kept whole.
        res["trace_len"] = len(trace)
        if pid in SLIM:
            trace = [{"k": s["k"], "x": s["x"], "fun": s["fun"]} for s in trace]
        res["trace_head"] = trace[:HEAD]
        res["trace_tail"] = trace[HEAD:][-TAIL:] if len(trace) > HEAD else []
        runs.append({"method": method, "problem": pid, "params": params, "result": res})
    errors = []
    for method, pid, params in ERRORS:
        try:
            METHODS[method](problem_of(pid), **params)
        except ValueError as exc:
            errors.append({"method": method, "problem": pid, "params": params, "error": str(exc)})
        else:
            raise SystemExit(f"{method} {params}: expected a ValueError")
    data = {"runs": runs, "errors": errors, "eigh2": eigh2_cases(), "quartic": QUARTIC_DATA}
    OUT.parent.mkdir(exist_ok=True)

    def default(o: Any) -> Any:
        raise TypeError(type(o))

    text = json.dumps(_clean(data), ensure_ascii=False, default=default)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(runs)} runs, {len(errors)} errors)")


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


if __name__ == "__main__":
    main()
