"""Reference values for tests/newton_qn/*.test.ts (test data only; never bundled).

Run from the repository root:

    .venv/bin/python web/tests/newton_qn/gen_newton_qn_fixture.py
    (cd web && npx prettier --write tests/newton_qn/fixtures)

It writes web/tests/newton_qn/fixtures/newton_qn_python.json with full results (every step,
every info key, counts, messages) of ``numopt.unconstrained.newton`` and
``numopt.unconstrained.quasi_newton`` on cases the parity fixtures do not cover: other line
searches, finite-difference derivatives, saddle points, maximizers, singular and divergent runs,
max_iter stops, n > 2, and the ValueError messages of invalid input.

The custom problems below are rebuilt in the TS tests (tests/newton_qn/custom.ts).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import Problem, to_jsonable

OUT = Path(__file__).with_name("fixtures") / "newton_qn_python.json"

D2 = ((-2.0, 2.0), (-2.0, 2.0))
HEAD, TAIL = 10, 1


def _rosen(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _rosen_grad(x: Any) -> Any:
    x = np.asarray(x, dtype=float)
    return np.array(
        [-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2), 200.0 * (x[1] - x[0] ** 2)]
    )


CUSTOM: dict[str, Problem] = {
    # Rosenbrock with f and ∇f only: ∇²f is a central difference of ∇f.
    "rosen_nohess": Problem(
        id="rosen_nohess", name="", latex="", f=_rosen, grad=_rosen_grad, dim=2, domain=D2,
        x0=[-1.2, 1.0],
    ),
    # Rosenbrock with f only: ∇f and ∇²f are central differences.
    "rosen_fonly": Problem(id="rosen_fonly", name="", latex="", f=_rosen, dim=2, domain=D2, x0=[-1.2, 1.0]),
    # x² − y²: a saddle point at 0.
    "saddle": Problem(
        id="saddle", name="", latex="", dim=2, domain=D2, x0=[1.0, 0.5],
        f=lambda x: float(x[0] ** 2 - x[1] ** 2),
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.array([[2.0, 0.0], [0.0, -2.0]]),
    ),
    # −(x² + y²): a maximizer at 0.
    "bowl_down": Problem(
        id="bowl_down", name="", latex="", dim=2, domain=D2, x0=[1.0, 0.5],
        f=lambda x: float(-(x[0] ** 2 + x[1] ** 2)),
        grad=lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.array([[-2.0, 0.0], [0.0, -2.0]]),
    ),
    # (x + y)²: singular ∇²f everywhere, minimizers on x + y = 0.
    "valley": Problem(
        id="valley", name="", latex="", dim=2, domain=D2, x0=[1.0, 2.0],
        f=lambda x: float((x[0] + x[1]) ** 2),
        grad=lambda x: np.array([2.0 * (x[0] + x[1]), 2.0 * (x[0] + x[1])]),
        hess=lambda x: np.array([[2.0, 2.0], [2.0, 2.0]]),
    ),
    "valley_fonly": Problem(
        id="valley_fonly", name="", latex="", dim=2, domain=D2, x0=[1.0, 2.0],
        f=lambda x: float((x[0] + x[1]) ** 2),
    ),
    # −x² + y⁴: ∇²f = diag(−2, 0) at 0, "a saddle point or a maximizer".
    "negsemi": Problem(
        id="negsemi", name="", latex="", dim=2, domain=D2, x0=[0.0, 0.0],
        f=lambda x: float(-x[0] ** 2 + x[1] ** 4),
        grad=lambda x: np.array([-2.0 * x[0], 4.0 * x[1] ** 3]),
        hess=lambda x: np.array([[-2.0, 0.0], [0.0, 12.0 * x[1] ** 2]]),
    ),
    # √(1 + x²) + √(1 + y²): pure Newton maps x to −x³ and diverges from |x| > 1.
    "soft_abs": Problem(
        id="soft_abs", name="", latex="", dim=2, domain=D2, x0=[1.5, 0.5],
        f=lambda x: float(math.sqrt(1.0 + x[0] ** 2) + math.sqrt(1.0 + x[1] ** 2)),
        grad=lambda x: np.array([x[0] / math.sqrt(1.0 + x[0] ** 2), x[1] / math.sqrt(1.0 + x[1] ** 2)]),
        hess=lambda x: np.array(
            [[(1.0 + x[0] ** 2) ** -1.5, 0.0], [0.0, (1.0 + x[1] ** 2) ** -1.5]]
        ),
    ),
    # A 3-D convex quartic: n > 2 (no hess / H info overlay), several Jacobi sweeps.
    "quartic3": Problem(
        id="quartic3", name="", latex="", dim=3, domain=((-2.0, 2.0),) * 3, x0=[1.5, -1.0, 0.7],
        f=lambda x: float(x[0] ** 4 + (x[0] - x[1]) ** 2 + 2.0 * (x[1] + x[2]) ** 2 + x[2] ** 4 + 0.5 * x[0] * x[2]),
        grad=lambda x: np.array([
            4.0 * x[0] ** 3 + 2.0 * (x[0] - x[1]) + 0.5 * x[2],
            -2.0 * (x[0] - x[1]) + 4.0 * (x[1] + x[2]),
            4.0 * (x[1] + x[2]) + 4.0 * x[2] ** 3 + 0.5 * x[0],
        ]),
        hess=lambda x: np.array([
            [12.0 * x[0] ** 2 + 2.0, -2.0, 0.5],
            [-2.0, 6.0, 4.0],
            [0.5, 4.0, 4.0 + 12.0 * x[2] ** 2],
        ]),
    ),
}


def get(pid: str) -> Any:
    return CUSTOM[pid] if pid in CUSTOM else problems.get(pid)


def case(method: str, pid: str, **params: Any) -> dict[str, Any]:
    spec = get_method(method)
    out: dict[str, Any] = {"method": method, "problem": pid, "params": params}
    try:
        res = spec.fn(get(pid), **params).to_dict()
        # Keep the file small: the first HEAD steps and the last TAIL steps (each has its k).
        trace = res["trace"]
        if len(trace) > HEAD + TAIL:
            res["trace"] = trace[:HEAD] + trace[-TAIL:]
        out["result"] = res
    except ValueError as exc:
        out["error"] = str(exc)
    return to_jsonable(out)


LS = ("backtracking", "strong_wolfe", "weak_wolfe", "goldstein")


def newton_cases() -> list[dict[str, Any]]:
    c = case
    out = []
    lib = ("rosenbrock", "himmelblau", "beale", "six_hump_camel", "booth", "three_hump_camel")
    for pid in lib + ("rosen_nohess", "rosen_fonly", "quartic3"):
        out.append(c("pure_newton", pid))
    for pid in ("saddle", "bowl_down", "valley", "valley_fonly", "soft_abs", "negsemi"):
        out.append(c("pure_newton", pid))
    out.append(c("pure_newton", "rosenbrock", max_iter=2))
    out.append(c("pure_newton", "himmelblau", x0=[-0.27, -0.92]))
    for m in ("damped_newton", "modified_newton"):
        for ls in LS:
            for pid in ("rosenbrock", "himmelblau", "six_hump_camel", "beale", "quartic3"):
                out.append(c(m, pid, line_search=ls))
        for pid in ("rosen_nohess", "rosen_fonly", "valley", "valley_fonly", "saddle", "negsemi", "soft_abs"):
            out.append(c(m, pid))
        out.append(c(m, "himmelblau", x0=[-0.27, -0.92]))
        out.append(c(m, "himmelblau", x0=[0.0, 0.0]))
        out.append(c(m, "six_hump_camel", x0=[1.5, 0.5]))
        out.append(c(m, "rosenbrock", max_iter=3))
        out.append(c(m, "rosenbrock", gtol=0.0, max_iter=60))
    out.append(c("modified_newton", "himmelblau", x0=[-0.27, -0.92], beta=2.0))
    # f unbounded below: ∇fᵀp overflows to −∞, which ends the run with a message.
    out.append(c("modified_newton", "bowl_down"))
    # ValueErrors.
    out.append(c("pure_newton", "rosenbrock", gtol=-1.0))
    out.append(c("pure_newton", "rosenbrock", max_iter=0))
    out.append(c("damped_newton", "rosenbrock", line_search="exact_quadratic"))
    out.append(c("modified_newton", "rosenbrock", beta=0.0))
    out.append(c("pure_newton", "rosenbrock", x0=[1.0, 2.0, 3.0]))
    return out


def qn_cases() -> list[dict[str, Any]]:
    c = case
    out = []
    # Every line search on four 2-D problems; two searches on the quadratics; the default
    # (strong Wolfe) on the n = 10 Rosenbrock (keeps the file small).
    per_problem = {
        **{pid: LS for pid in ("rosenbrock", "himmelblau", "beale", "six_hump_camel")},
        "booth": ("strong_wolfe", "backtracking"),
        "quadratic_ill": ("strong_wolfe", "backtracking"),
        "rosenbrock_nd": ("strong_wolfe",),
    }
    # (method, extra params, full line-search matrix?)
    variants: list[tuple[str, dict[str, Any], bool]] = [
        ("bfgs", {}, True),
        ("dfp", {}, False),
        ("sr1", {}, True),
        ("broyden_class", {"phi": 0.0}, False),
        ("broyden_class", {"phi": 0.3}, True),
        ("broyden_class", {"phi": 1.0}, False),
        ("lbfgs", {"m": 1}, False),
        ("lbfgs", {"m": 3}, True),
        ("lbfgs", {}, False),
    ]
    two = ("strong_wolfe", "backtracking")
    for m, extra, full in variants:
        # Every line search (or two) on four 2-D problems, two on the quadratics.
        for pid in ("rosenbrock", "himmelblau", "beale", "six_hump_camel"):
            for ls in LS if full else two:
                out.append(c(m, pid, line_search=ls, **extra))
        for pid in ("booth", "quadratic_ill"):
            for ls in two:
                out.append(c(m, pid, line_search=ls, **extra))
        for pid in ("rosen_fonly", "quartic3", "saddle"):
            out.append(c(m, pid, **extra))
        out.append(c(m, "rosenbrock", max_iter=4, **extra))
        out.append(c(m, "rosenbrock", gtol=0.0, max_iter=200, **extra))
    # n = 10 (NumPy's BLAS sums in a SIMD order: compared loosely in the tests).
    for m, extra in (("bfgs", {}), ("sr1", {}), ("lbfgs", {"m": 3})):
        out.append(c(m, "rosenbrock_nd", **extra))
    # f unbounded below: ∇fᵀp overflows to −∞, which ends the run with a message.
    out.append(c("bfgs", "bowl_down", line_search="backtracking", max_iter=2000))
    # ValueErrors.
    out.append(c("bfgs", "rosenbrock", line_search="exact_quadratic"))
    out.append(c("bfgs", "rosenbrock", max_iter=0))
    out.append(c("broyden_class", "rosenbrock", phi=1.5))
    out.append(c("lbfgs", "rosenbrock", m=0))
    out.append(c("dfp", "rosenbrock", gtol=float("nan")))
    return out


def main() -> None:
    data = {"newton": newton_cases(), "quasi_newton": qn_cases()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
