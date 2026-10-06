"""Reference values for tests/shared-ports/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/shared-ports/gen_shared_ports_fixture.py
    (cd web && npx prettier --write tests/shared-ports/fixtures)   # keeps `npm run format:check` green

It writes web/tests/shared-ports/fixtures/shared_ports_python.json with

* ``problems``: f, ∇f, ∇²f of every unconstrained and calculus problem at its x0, its minima
  and six seeded random points of its domain;
* ``search``: direct calls of ``numopt.line_search.methods.search`` (results and ValueErrors);
* ``demos``: full results of the five registered line-search demos on cases the parity fixtures
  do not cover (Newton directions, failure paths, 1-D problems, finite-difference derivatives).

The custom problems below (``linear_2d``, ``rosen_nograd``) are rebuilt in the TS tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.core.types import Problem, to_jsonable
from numopt.line_search import methods as ls

OUT = Path(__file__).with_name("fixtures") / "shared_ports_python.json"


def _rosen(x: Any) -> Any:
    x = np.asarray(x, dtype=float)
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


CUSTOM: dict[str, Problem] = {
    # f is unbounded below along -grad: the expanding searches hit alpha_max.
    "linear_2d": Problem(
        id="linear_2d",
        name="linear",
        latex="x + 2y",
        f=lambda x: float(np.asarray(x)[0] + 2.0 * np.asarray(x)[1]),
        grad=lambda x: np.array([1.0, 2.0]),
        hess=lambda x: np.zeros((2, 2)),
        dim=2,
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        x0=[0.0, 0.0],
    ),
    # Rosenbrock without derivatives: the demos use central differences.
    "rosen_nograd": Problem(
        id="rosen_nograd",
        name="Rosenbrock (f only)",
        latex="",
        f=_rosen,
        dim=2,
        domain=((-2.0, 2.0), (-1.0, 3.0)),
        x0=[-1.2, 1.0],
    ),
}


def get(pid: str) -> Any:
    return CUSTOM[pid] if pid in CUSTOM else problems.get(pid)


def sample_points(p: Any, rng: np.random.Generator) -> list[Any]:
    pts: list[Any] = []
    if p.dim == 1:
        a, b = p.domain
        pts.append(float(p.x0))
        pts += [float(v) for v in rng.uniform(a, b, 6)]
        return pts
    pts.append(list(map(float, p.x0)))
    pts += [list(map(float, m)) for m in p.minima]
    lo = np.array([d[0] for d in p.domain])
    hi = np.array([d[1] for d in p.domain])
    for _ in range(6):
        pts.append(rng.uniform(lo, hi).tolist())
    return pts


def problem_values() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20261005)
    out = []
    for kind in ("unconstrained", "calculus"):
        for p in problems.list_problems(kind):
            for x in sample_points(p, rng):
                xa = x if p.dim == 1 else np.asarray(x, dtype=float)
                with np.errstate(all="ignore"):
                    out.append(
                        {
                            "id": p.id,
                            "x": x,
                            "f": float(p.f(xa)),
                            "grad": p.grad(xa),
                            "hess": p.hess(xa),
                        }
                    )
    return to_jsonable(out)


def search_case(
    kind: str, pid: str, x: list[float] | None = None, p: Any = "-grad", **opts: Any
) -> dict[str, Any]:
    prob = get(pid)
    xv = np.asarray(prob.x0 if x is None else x, dtype=float)
    g = np.asarray(prob.grad(xv), dtype=float)
    pv = -g if isinstance(p, str) and p == "-grad" else (g if p == "+grad" else np.asarray(p))
    call = dict(opts)
    if call.pop("pass_f0g0", False):
        call["f0"] = float(prob.f(xv))
        call["g0"] = g
    if call.pop("hess_matrix", False):
        call["hess"] = prob.hess(xv)
    elif call.pop("hess_callable", False):
        call["hess"] = prob.hess
    case = {"kind": kind, "problem": pid, "x": xv.tolist(), "p": pv.tolist(), "opts": opts}
    try:
        r = ls.search(kind, prob.f, prob.grad, xv, pv, **call)
        case["result"] = {
            "alpha": r.alpha,
            "f_new": r.f_new,
            "g_new": None if r.g_new is None else r.g_new.tolist(),
            "n_fev": r.n_fev,
            "n_gev": r.n_gev,
            "n_hev": r.n_hev,
            "success": r.success,
            "trials": [list(t) for t in r.trials],
            "message": r.message,
        }
    except ValueError as exc:
        case["error"] = str(exc)
    return to_jsonable(case)


def search_cases() -> list[dict[str, Any]]:
    c = search_case
    return [
        c("backtracking", "rosenbrock"),
        c("backtracking", "rosenbrock", pass_f0g0=True, rho=0.3, alpha0=2.0),
        c("backtracking", "rosenbrock", max_iter=3),
        c("backtracking", "rosenbrock", alpha0=1e-320),
        c("backtracking", "beale", c1=0.4),
        c("strong_wolfe", "rosenbrock"),
        c("strong_wolfe", "rosenbrock", c2=0.1),
        c("strong_wolfe", "rosenbrock", alpha0=1e-3, c2=0.1),
        c("strong_wolfe", "himmelblau", [1.0, -2.0], alpha0=0.5),
        c("strong_wolfe", "beale", alpha0=10.0, c2=0.1),
        c("strong_wolfe", "goldstein_price", [0.5, 0.5], c2=0.1),
        c("strong_wolfe", "ackley", [1.3, 0.4], alpha0=0.01, c2=0.1),
        c("strong_wolfe", "levi13", alpha0=0.001, c2=0.1),
        c("strong_wolfe", "rastrigin", alpha0=0.5, c2=0.5),
        c("strong_wolfe", "rosenbrock", max_iter=2, c2=0.1),
        c("strong_wolfe", "linear_2d", alpha_max=50.0),
        c("strong_wolfe", "rosenbrock_nd", alpha0=1e-3, c2=0.1),
        c("strong_wolfe", "quadratic_nd", c2=0.1),
        c("strong_wolfe", "six_hump_camel", pass_f0g0=True, c1=0.01, c2=0.5),
        c("weak_wolfe", "rosenbrock"),
        c("weak_wolfe", "himmelblau", [1.0, -2.0], alpha0=10.0),
        c("weak_wolfe", "beale", alpha0=1e-3),
        c("weak_wolfe", "linear_2d", alpha_max=100.0),
        c("weak_wolfe", "styblinski_tang", alpha0=0.2, c2=0.5),
        c("weak_wolfe", "rosenbrock_nd", max_iter=4),
        c("goldstein", "himmelblau"),
        c("goldstein", "rosenbrock", alpha0=5.0),
        c("goldstein", "bohachevsky", c1=0.1),
        c("goldstein", "linear_2d", alpha0=2.0, alpha_max=40.0),
        c("goldstein", "mccormick", alpha0=1e-4),
        c("exact_quadratic", "quadratic_ill", hess_matrix=True),
        c("exact_quadratic", "quadratic_bowl", [2.0, 1.0], hess_callable=True),
        c("exact_quadratic", "quadratic_nd", hess_callable=True),
        c("exact_quadratic", "rosenbrock", hess_callable=True),
        c("exact_quadratic", "rosenbrock", [0.0, 0.0], hess_callable=True, f_err=10.0),
        c("exact_quadratic", "three_hump_camel", hess_callable=True),
        c("exact_quadratic", "matyas", pass_f0g0=True, hess_matrix=True),
        # ValueError paths
        c("backtracking", "rosenbrock", p="+grad"),
        c("backtracking", "rosenbrock", p=[0.0, 0.0]),
        c("bogus", "rosenbrock"),
        c("backtracking", "rosenbrock", c1=1.5),
        c("goldstein", "rosenbrock", c1=0.6),
        c("strong_wolfe", "rosenbrock", c1=0.5, c2=0.4),
        c("backtracking", "rosenbrock", rho=1.0),
        c("backtracking", "rosenbrock", alpha0=0.0),
        c("weak_wolfe", "rosenbrock", alpha_max=math.inf),
        c("strong_wolfe", "rosenbrock", alpha0=10.0, alpha_max=5.0),
        c("backtracking", "rosenbrock", max_iter=0),
        c("exact_quadratic", "rosenbrock"),
        c("exact_quadratic", "rosenbrock", hess_callable=True, f_err=-1.0),
        c("exact_quadratic", "himmelblau", hess_callable=True),
    ]


def demo_case(method: str, pid: str, **params: Any) -> dict[str, Any]:
    from numopt.core.registry import get_method

    spec = get_method(method)
    case: dict[str, Any] = {"method": method, "problem": pid, "params": params}
    try:
        r = spec.fn(get(pid), **params)
        case["result"] = r.to_dict()
    except ValueError as exc:
        case["error"] = str(exc)
    return to_jsonable(case)


def demo_cases() -> list[dict[str, Any]]:
    d = demo_case
    out = []
    for m in ("backtracking", "strong_wolfe", "weak_wolfe", "goldstein", "exact_quadratic"):
        out += [
            d(m, "rosenbrock", direction="newton"),
            d(m, "himmelblau", x0=[1.0, 1.0], direction="newton"),
            d(m, "beale"),
            d(m, "rosenbrock", x0=[1.0, 1.0]),  # stationary point
            d(m, "himmelblau", x0=[-0.27, -0.92], direction="newton"),  # ascent Newton step
            d(m, "rosen_nograd"),
            d(m, "rosen_nograd", direction="newton"),
            d(m, "runge"),
            d(m, "gaussian", x0=1.2, direction="newton"),
            d(m, "rosenbrock_nd"),
            d(m, "goldstein_price", x0=[0.3, -0.6]),
        ]
        if m != "exact_quadratic":
            out += [
                d(m, "rosenbrock", max_iter=1),
                d(m, "quadratic_bowl", alpha0=100.0),
                d(m, "linear_2d"),
            ]
    out += [
        d("exact_quadratic", "himmelblau"),  # pᵀHp < 0 at the origin
        d("strong_wolfe", "rosenbrock", c1=0.09, c2=0.1),  # invalid? no: c1 < c2
        d("strong_wolfe", "rosenbrock", c1=0.05, c2=0.01),  # ValueError (c2 < c1)
        d("backtracking", "rosenbrock", direction="sideways"),  # ValueError
        d("backtracking", "rosenbrock", x0=[1.0, 2.0, 3.0]),  # ValueError (wrong size)
        d("goldstein", "himmelblau", alpha0=100.0, alpha_max=100.0, c1=0.4),
        d("weak_wolfe", "ackley", alpha0=1e-4, c2=0.1, max_iter=200),
    ]
    return out


def format_cases() -> list[list[Any]]:
    """Python's ``format(v, '.6g')``, ``format(v, '.3g')`` and ``repr(v)`` of awkward floats."""
    rng = np.random.default_rng(7)
    values = [2.0**k for k in range(-1074, 1024, 37)]
    values += [0.5 * 10.0**k for k in range(-12, 12)]
    values += [1.25e-5, 9.9999995, 0.00012345675, 999999.5, 1e16, 123456789.0, 5e-324]
    values += (rng.standard_normal(150) * 10.0 ** rng.uniform(-9, 9, 150)).tolist()
    values += (rng.integers(1, 2**20, 50) / 2.0 ** rng.integers(1, 30, 50)).tolist()
    out = []
    for v in values:
        for w in (v, -v):
            out.append([w, format(w, ".6g"), format(w, ".3g"), repr(w)])
    return out


def main() -> None:
    data = {
        "formats": format_cases(),
        "problems": problem_values(),
        "search": search_cases(),
        "demos": demo_cases(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
