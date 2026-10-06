"""Performance-estimation (PEP) oracle for the certified schedules (needs cvxpy).

The worst case of a fixed-step first-order method over all L-smooth (μ-strongly) convex
functions in any dimension is the value of a small SDP (Drori & Teboulle 2014; Taylor,
Hendrickx & Glineur 2017, Theorem 4 for the interpolation conditions). With L = 1, x* = 0,
g* = 0, f* = 0 and the Gram matrix G of the basis [x₀, g₀, ..., g_N]:

    maximize   f_N − f*                     (or ‖x_N − x*‖² for the strongly convex case)
    subject to ‖x₀ − x*‖² ≤ 1,  G ⪰ 0,
               f_i ≥ f_j + ⟨g_j, x_i − x_j⟩ + 1/(2(1 − μ)) (‖g_i − g_j‖² + μ‖x_i − x_j‖²
                     − 2μ⟨g_j − g_i, x_j − x_i⟩)          for all i ≠ j ∈ {*, 0, ..., N}.

Every iterate is a fixed linear combination of the basis vectors, so the constraints are
linear in (G, f). The SDP value is the exact worst case (up to solver tolerance), which makes
it an oracle *independent* of the papers' proofs: a certified bound must be ≥ this value.

Run ``python3 pep.py`` (a Python with cvxpy + Clarabel) to write ``results/pep.json``.
This file is not imported by ``run.py`` (the experiment needs only numpy/scipy/matplotlib).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _fixed_step_points(h: Sequence[float]) -> list[np.ndarray]:
    """Coefficients of x₀, ..., x_N in the basis [x₀, g₀, ..., g_N] for GD with steps h."""
    N = len(h)
    dim = N + 2
    x = np.zeros(dim)
    x[0] = 1.0
    pts = [x.copy()]
    for t, ht in enumerate(h):
        x = x.copy()
        x[1 + t] -= ht  # x_{t+1} = x_t − h_t g_t  (L = 1)
        pts.append(x)
    return pts


def _ogm_points(N: int) -> list[np.ndarray]:
    """Coefficients of OGM1's x₀, ..., x_N (Kim & Fessler 2016, Algorithm OGM1), L = 1.

    Written independently of ``method.ogm`` (it acts on coefficient vectors, not points).
    """
    dim = N + 2
    th = [1.0]
    for i in range(N):
        c = 8.0 if i == N - 1 else 4.0
        th.append(0.5 * (1.0 + math.sqrt(1.0 + c * th[i] ** 2)))
    x = np.zeros(dim)
    x[0] = 1.0
    y = x.copy()
    pts = [x.copy()]
    for i in range(N):
        g = np.zeros(dim)
        g[1 + i] = 1.0
        y_new = x - g
        x = y_new + (th[i] - 1.0) / th[i + 1] * (y_new - y) + th[i] / th[i + 1] * (y_new - x)
        y = y_new
        pts.append(x.copy())
    return pts


def worst_case(points: list[np.ndarray], *, mu: float = 0.0, measure: str = "f") -> float:
    """SDP value of the PEP for iterates ``points`` (L = 1, R = 1, strong convexity μ < 1)."""
    import cvxpy as cp  # pyright: ignore[reportMissingImports]

    N = len(points) - 1
    dim = N + 2
    G = cp.Variable((dim, dim), PSD=True)
    F = cp.Variable(N + 1)
    # index −1 is the minimizer: x* = 0, g* = 0, f* = 0
    X = [*points, np.zeros(dim)]
    Gr = [*[np.eye(dim)[1 + i] for i in range(N + 1)], np.zeros(dim)]
    Fv = [*[F[i] for i in range(N + 1)], 0.0]

    def ip(a: np.ndarray, b: np.ndarray) -> cp.Expression:
        return cp.sum(cp.multiply(G, np.outer(a, b)))

    cons = [ip(X[0], X[0]) <= 1.0]
    m = len(X)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            dx, dg = X[i] - X[j], Gr[i] - Gr[j]
            rhs = (
                Fv[j]
                + ip(Gr[j], dx)
                + (ip(dg, dg) + mu * ip(dx, dx) - 2.0 * mu * ip(-dg, -dx)) / (2.0 * (1.0 - mu))
            )
            cons.append(Fv[i] >= rhs)
    obj = F[N] if measure == "f" else ip(X[N], X[N])
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"PEP solve failed: {prob.status}")
    return float(prob.value)


def worst_case_gd(h: Sequence[float], *, mu: float = 0.0, measure: str = "f") -> float:
    """Worst case of f(x_N) − f* (or ‖x_N − x*‖²) for GD with normalized steps h."""
    return worst_case(_fixed_step_points(h), mu=mu, measure=measure)


def worst_case_ogm(N: int) -> float:
    """Worst case of f(x_N) − f* for OGM1 with horizon N."""
    return worst_case(_ogm_points(N))


def main() -> None:
    import method as M

    out: dict[str, list[dict[str, float]]] = {
        "silver": [],
        "constant": [],
        "ogm": [],
        "silver_sc": [],
        "long_step": [],
    }
    for k in range(1, 6):  # n = 1, 3, 7, 15, 31
        n = 2**k - 1
        pep = worst_case_gd(M.silver_schedule(n).tolist())
        out["silver"].append({"n": n, "pep": pep, "bound": M.silver_rate(k)})
        out["constant"].append(
            {"n": n, "pep": worst_case_gd([1.0] * n), "bound": 1.0 / (4 * n + 2)}
        )
        out["ogm"].append({"n": n, "pep": worst_case_ogm(n), "bound": M.ogm_rate(n)})
    for kappa in (4.0, 16.0, 100.0):
        for n in (1, 2, 4, 8, 16):
            h, tau = M.silver_sc_schedule(kappa, n)
            pep = worst_case_gd(h.tolist(), mu=1.0 / kappa, measure="dist")
            out["silver_sc"].append({"kappa": kappa, "n": n, "pep": pep, "bound": tau})
    for pat in ("2", "3", "7", "15"):
        h = M.LONG_STEP_PATTERNS[pat]
        for reps in (1, 2):
            T = reps * len(h)
            pep = worst_case_gd(list(h) * reps)
            out["long_step"].append(
                {
                    "pattern": float(pat),
                    "T": T,
                    "pep": pep,
                    "c_over_T": 1.0 / (M.LONG_STEP_RATES[pat] * T),
                }
            )
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "pep.json").write_text(json.dumps(out, indent=1))
    for key, rows in out.items():
        for r in rows:
            print(key, r)


if __name__ == "__main__":
    main()
