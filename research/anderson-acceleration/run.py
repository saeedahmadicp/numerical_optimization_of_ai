"""Reproducible experiments for AA(m) / RNA. Run: .venv/bin/python research/anderson-acceleration/run.py

E1  Walker–Ni Thm. 2.2: untruncated AA vs full GMRES on linear maps g(x) = Mx + b.
E2  2-D contractions g(x) = cos(Ax): AA(m) vs plain iteration, good Broyden, Newton, SciPy anderson.
E3  AA(m)-GD vs L-BFGS(m), GD, Nesterov on quadratic_nd and rosenbrock_nd (gradient evaluations).
E4  RNA λ on himmelblau and beale from a grid of (nonconvex) starts.

Deterministic: start points come from numopt.core.rng.Rng with fixed seeds; no network.
Writes results/*.json and figures/*.svg.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from numopt import bench, problems, run
from numopt.core.rng import Rng
from numopt.core.types import LinearSystem, Problem, to_jsonable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from method import anderson, anderson_gd  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
EPS = float(np.finfo(float).eps)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
warnings.filterwarnings("ignore", category=RuntimeWarning)  # overflow in diverging baselines


def savefig(fig: Any, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150)
    plt.close(fig)


def save(name: str, obj: Any) -> None:
    (RES / name).write_text(json.dumps(to_jsonable(obj), indent=1, allow_nan=False))


class FirstHit:
    """Count calls of a vector map and record the first call whose 2-norm is ≤ tol.

    ``hit_any`` is the first call with norm ≤ tol at any point, and ``x_any`` is that point.
    ``hit`` is the first call with norm ≤ tol at a point that ``accept`` accepts (all points
    when ``accept`` is None). E3 uses ``accept`` to count only minimizers, not saddle points.
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], Any],
        tol: float,
        accept: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.fn, self.tol, self.accept, self.n = fn, tol, accept, 0
        self.hit = self.hit_any = math.inf
        self.x_any: np.ndarray | None = None
        self.best: list[float] = []  # running minimum of the norm, one entry per call

    def __call__(self, x: np.ndarray) -> Any:
        self.n += 1
        v = self.fn(x)
        r = float(np.linalg.norm(np.asarray(v, dtype=float)))
        if not math.isfinite(r):
            r = math.inf
        if r <= self.tol:
            if self.hit_any == math.inf:
                self.hit_any, self.x_any = self.n, np.array(x, dtype=float)
            if self.hit == math.inf and (self.accept is None or self.accept(x)):
                self.hit = self.n
        self.best.append(min(r, self.best[-1]) if self.best else r)
        return v


def uniform_starts(seed: int, n: int, box: tuple[tuple[float, float], ...], count: int) -> list:
    rng = Rng(seed)  # draw order: point by point, coordinate by coordinate
    return [np.array([rng.uniform(lo, hi) for lo, hi in box[:n]]) for _ in range(count)]


# ======================================================================================
# E1  AA(∞) vs GMRES on linear maps
# ======================================================================================


def e1_gmres() -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    n = 30
    for seed in range(5):
        rng = Rng(100 + seed)
        M = np.array([[rng.normal() for _ in range(n)] for _ in range(n)])
        M *= 0.95 / max(abs(np.linalg.eigvals(M)))  # ρ(M) = 0.95, non-normal
        b = np.array([rng.normal() for _ in range(n)])
        x0 = np.zeros(n)
        A = np.eye(n) - M
        gm = run("gmres", LinearSystem("lin", "lin", A, b), x0=x0, restart=n, tol=1e-14)
        aa = anderson(lambda x, M=M, b=b: M @ x + b - x, x0=x0, m=10_000, ftol=0.0, max_iter=n)
        pic = anderson(lambda x, M=M, b=b: M @ x + b - x, x0=x0, m=0, ftol=0.0, max_iter=n)
        K = min(len(gm.trace), n)
        r0 = float(np.linalg.norm(b))
        gm_res = [float(gm.trace[k].fun or 0.0) for k in range(K)]
        aa_res = [float(aa.trace[k + 1].info["lsq_residual"]) for k in range(K)]
        dev_res = max(abs(a - g) for a, g in zip(aa_res, gm_res, strict=True)) / r0
        dev_x = max(
            float(np.linalg.norm(aa.trace[k + 1].x - (M @ np.asarray(gm.trace[k].x) + b)))
            / max(1.0, float(np.linalg.norm(gm.trace[k].x)))
            for k in range(K)
        )
        out.append(
            {
                "seed": 100 + seed,
                "n": n,
                "cond_I_minus_M": float(np.linalg.cond(A)),
                "gmres_steps": gm.n_iter,
                "max_abs_residual_dev_over_r0": dev_res,
                "max_rel_iterate_dev": dev_x,
                "gmres_residual": gm_res,
                "aa_lsq_residual": aa_res,
                "aa_true_residual": [float(s.fun or 0.0) for s in aa.trace[1 : K + 1]],
                "picard_residual": [float(s.fun or 0.0) for s in pic.trace[1 : K + 1]],
            }
        )
    d = out[0]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    k = np.arange(len(d["gmres_residual"]))
    ax.semilogy(
        k, d["gmres_residual"], "-", color=COLORS[0], lw=2.5, label="full GMRES  $\\|r_k\\|$"
    )
    ax.semilogy(
        k,
        d["aa_lsq_residual"],
        "o",
        ms=3.5,
        color=COLORS[1],
        label="AA($\\infty$)  $\\|F_k c^*\\|$",
    )
    ax.semilogy(
        k + 1,
        d["aa_true_residual"],
        ":",
        color=COLORS[1],
        label="AA($\\infty$)  $\\|f(x_{k+1})\\|$",
    )
    ax.semilogy(k + 1, d["picard_residual"], "--", color="0.5", label="plain iteration")
    ax.set_xlabel("iteration k")
    ax.set_ylabel("residual 2-norm")
    ax.set_title(f"Linear map, n = {d['n']}, ρ(M) = 0.95 (seed {d['seed']})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "e1_gmres_equivalence")
    summary = {
        "max_abs_residual_dev_over_r0": max(o["max_abs_residual_dev_over_r0"] for o in out),
        "max_rel_iterate_dev": max(o["max_rel_iterate_dev"] for o in out),
        "max_cond": max(o["cond_I_minus_M"] for o in out),
    }
    save("e1_gmres.json", {"summary": summary, "instances": out})
    return summary


# ======================================================================================
# E2  2-D contractions g(x) = cos(Ax)
# ======================================================================================

MAPS = {
    "C1": [[0.5, 0.25], [0.25, 0.5]],
    "C2": [[0.9, -0.6], [0.6, 0.9]],
    "C3": [[1.0, 0.5], [0.2, 0.6]],
    "C4": [[1.05, 0.5], [0.2, 0.7]],
    "N1": [[1.0, 0.5], [0.5, 1.0]],
}
E2_TOL, E2_BUDGET = 1e-10, 500


def cos_problem(A: np.ndarray, F: Callable[[np.ndarray], np.ndarray]) -> Problem:
    return Problem(
        id="cos_map", name="cos(Ax) − x", latex=r"\cos(Ax)-x", f=F, dim=2,
        domain=((-2, 2), (-2, 2)),
        jac=lambda x: -np.diag(np.sin(A @ x)) @ A - np.eye(2),
    )  # fmt: skip


def e2_contractions() -> dict[str, Any]:
    starts = uniform_starts(7, 2, ((-2.0, 2.0), (-2.0, 2.0)), 10)
    methods: dict[str, Callable[[Problem, np.ndarray], Any]] = {
        "plain (AA(0))": lambda p, x: anderson(p, x0=x, m=0, ftol=E2_TOL, max_iter=E2_BUDGET),
        **{
            f"AA({m})": (lambda p, x, m=m: anderson(p, x0=x, m=m, ftol=E2_TOL, max_iter=E2_BUDGET))
            for m in (1, 2, 3, 5, 10)
        },
        "Broyden (exact J0)": lambda p, x: run("broyden", p, x0=x, ftol=E2_TOL, xtol=0.0, max_iter=E2_BUDGET),
        "Broyden (FD J0)": lambda p, x: run(
            "broyden", p, x0=x, ftol=E2_TOL, xtol=0.0, max_iter=E2_BUDGET, jacobian0="finite_difference"
        ),
        "Newton": lambda p, x: run("newton_system", p, x0=x, ftol=E2_TOL, xtol=0.0, max_iter=E2_BUDGET),
        "SciPy anderson M=3": lambda p, x: scipy.optimize.anderson(p.f, x, M=3, f_tol=E2_TOL / 2, maxiter=E2_BUDGET),
    }  # fmt: skip
    table: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[float]] = {}
    for name, Araw in MAPS.items():
        A = np.array(Araw)
        ref = anderson(
            cos_problem(A, lambda x, A=A: np.cos(A @ x) - x), x0=np.zeros(2), m=3, ftol=1e-14
        )
        J = -np.diag(np.sin(A @ ref.x)) @ A
        rho = float(max(abs(np.linalg.eigvals(J))))
        row: dict[str, Any] = {"A": Araw, "x_star": ref.x, "rho_g_prime": rho}
        for label, solve in methods.items():
            fevals, jevals = [], []
            for i, x0 in enumerate(starts):
                F = FirstHit(lambda x, A=A: np.cos(A @ x) - x, E2_TOL)
                p = cos_problem(A, F)
                try:
                    r = solve(p, x0)
                    jevals.append(int(getattr(r, "n_gev", 0)))
                except Exception:  # SciPy raises NoConvergence at maxiter
                    jevals.append(0)
                fevals.append(F.hit if F.hit <= E2_BUDGET else math.inf)
                if name == "C3" and i == 0:
                    curves[label] = F.best[:120]
            solved = [v for v in fevals if math.isfinite(v)]
            row[label] = {
                "solved": len(solved),
                "of": len(starts),
                "median_F_evals": float(np.median(solved)) if solved else None,
                "max_F_evals": max(solved) if solved else None,
                "median_jacobians": float(np.median(jevals)),
                "F_evals": fevals,
            }
        table[name] = row
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    styles = {  # AA variants solid; baselines dashed / dotted in distinct colours
        "plain (AA(0))": ("0.55", "-"), "AA(1)": (COLORS[5], "-"), "AA(3)": (COLORS[0], "-"),
        "AA(2)": ("#882255", "-"), "AA(5)": (COLORS[3], "-"), "AA(10)": (COLORS[4], "-"), "Broyden (exact J0)": (COLORS[1], "--"),
        "Broyden (FD J0)": (COLORS[1], ":"), "Newton": ("#000000", "-."), "SciPy anderson M=3": (COLORS[2], "--"),
    }  # fmt: skip
    for label, c in curves.items():
        col, ls = styles[label]
        ax.semilogy(
            np.arange(1, len(c) + 1), np.maximum(c, 1e-17), color=col, ls=ls, lw=1.5, label=label
        )
    ax.axhline(E2_TOL, color="0.6", lw=0.8)
    ax.set_xlabel("F evaluations")
    ax.set_ylabel("best $\\|F(x)\\|_2$ so far")
    ax.set_title(f"g(x) = cos(Ax), map C3 (ρ = {table['C3']['rho_g_prime']:.3f}), start 1")
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-16, 10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    savefig(fig, "e2_contraction_C3")
    save(
        "e2_contractions.json",
        {"tol": E2_TOL, "budget": E2_BUDGET, "starts": starts, "maps": table},
    )
    return table


# ======================================================================================
# E3  AA(m)-GD vs L-BFGS(m) and first-order baselines
# ======================================================================================

E3_TOL, E3_BUDGET = 1e-6, 2000
MEMS = (1, 3, 5, 10)
# NOTE: AA does not need the GD stability limit α < 2/L, so the grids extend past it (L = 100 on
# quadratic_nd). The grids are wide enough that every tuned AA(m)-GD step is interior.
LR_GRID = {
    "quadratic_nd": (0.005, 0.01, 0.015, 0.0199, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1),
    "rosenbrock_nd": (6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3),
}
NEST_BETA = (0.9, 0.95, 0.99)
F_GLOBAL_TOL = 1e-8  # f* = 0 on both E3 problems


def is_minimizer(prob: Problem, x: np.ndarray) -> bool:
    """Second-order test at a point with ‖∇f‖ ≤ E3_TOL: smallest Hessian eigenvalue > 0."""
    assert prob.hess is not None
    return bool(np.linalg.eigvalsh(prob.hess(np.asarray(x, dtype=float)))[0] > 0)


def grad_cost(prob: Problem, solve: Callable[[Problem], Any]) -> dict[str, Any]:
    """Gradient calls until the first call at a MINIMIZER with ‖∇f‖₂ ≤ E3_TOL (∞ beyond budget).

    A call with ‖∇f‖₂ ≤ E3_TOL at a point whose Hessian is not positive definite (a saddle point)
    does not count as solved. ``end`` classifies the first tolerance hit of any kind:
    "minimizer", "saddle" (λ_min(∇²f) ≤ 0) or "none" (no hit within the budget)."""
    G = FirstHit(prob.grad, E3_TOL, accept=lambda x: is_minimizer(prob, x))  # type: ignore[arg-type]
    p = dataclasses.replace(prob, grad=G)
    r = solve(p)
    hit = G.hit if G.hit <= E3_BUDGET else math.inf
    if G.hit_any <= E3_BUDGET and G.x_any is not None:
        assert prob.hess is not None
        lam_min = float(np.linalg.eigvalsh(prob.hess(G.x_any))[0])
        end, f_hit = ("minimizer" if lam_min > 0 else "saddle"), float(prob.f(G.x_any))
    else:
        lam_min, end, f_hit = math.nan, "none", math.nan
    return {
        "hit": hit,
        "best": G.best,
        "message": r.message,
        "f_final": float(prob.f(np.asarray(r.x, dtype=float))),
        "end": end,
        "f_at_first_hit": f_hit,
        "lam_min_at_first_hit": lam_min,
        "first_hit_any": G.hit_any if G.hit_any <= E3_BUDGET else math.inf,
    }


def e3_gd() -> dict[str, Any]:
    cases = {}
    for pid in ("quadratic_nd", "rosenbrock_nd"):
        p = problems.get(pid)
        box = p.domain
        cases[pid] = (
            p,
            [np.asarray(p.x0, float), *uniform_starts(11 if pid[0] == "q" else 13, p.dim, box, 9)],
        )
    solvers: dict[str, Callable[..., Any]] = {}
    for m in MEMS:
        solvers[f"AA({m})-GD"] = lambda p, x, lr, m=m, **_: anderson_gd(
            p, x0=x, m=m, lr=lr, gtol=E3_TOL, max_iter=E3_BUDGET
        )
        solvers[f"L-BFGS({m})"] = lambda p, x, m=m, **_: run(
            "lbfgs", p, x0=x, m=m, gtol=E3_TOL / math.sqrt(p.dim), max_iter=E3_BUDGET
        )
    solvers["GD fixed α"] = lambda p, x, lr, **_: run(
        "gradient_descent", p, x0=x, step_rule="fixed", lr=lr, gtol=E3_TOL, max_iter=E3_BUDGET
    )
    solvers["GD backtracking"] = lambda p, x, **_: run(
        "gradient_descent", p, x0=x, gtol=E3_TOL, max_iter=E3_BUDGET
    )
    solvers["Nesterov"] = lambda p, x, lr, mu, **_: run(
        "nesterov", p, x0=x, lr=lr, beta=mu, gtol=E3_TOL, max_iter=E3_BUDGET
    )  # fmt: skip
    tuned = {
        "AA": True,
        "L-": False,
        "GD fixed α": True,
        "GD backtracking": False,
        "Nesterov": True,
    }

    def configs(label: str, pid: str) -> list[dict[str, float]]:
        if label.startswith("AA") or label == "GD fixed α":
            return [{"lr": lr} for lr in LR_GRID[pid]]
        if label == "Nesterov":
            return [{"lr": lr, "mu": mu} for lr in LR_GRID[pid] for mu in NEST_BETA]
        return [{}]

    def key(costs: list[float]) -> tuple[int, float]:
        fin = [c for c in costs if math.isfinite(c)]
        return (-len(fin), float(np.median(costs)) if fin else math.inf)

    results: dict[str, Any] = {"tol": E3_TOL, "budget": E3_BUDGET, "problems": {}}
    best_curves: dict[str, dict[str, dict[str, Any]]] = {}
    labels = list(solvers)
    cost_table = np.full((20, len(labels)), math.inf)
    dims = np.zeros(20)
    for pi, (pid, (p, starts)) in enumerate(cases.items()):
        per: dict[str, Any] = {}
        best_curves[pid] = {}
        dims[pi * 10 : pi * 10 + 10] = p.dim
        for si, label in enumerate(labels):
            sweep = []
            for cfg in configs(label, pid):
                runs = [
                    grad_cost(
                        p, lambda q, x0=x0, cfg=cfg, label=label: solvers[label](q, x0, **cfg)
                    )
                    for x0 in starts
                ]
                sweep.append({"config": cfg, "grad_evals": [r["hit"] for r in runs], "runs": runs})
            chosen = min(sweep, key=lambda s: key(s["grad_evals"]))
            runs = chosen["runs"]
            fin = [c for c in chosen["grad_evals"] if math.isfinite(c)]
            grid = LR_GRID[pid]
            lr = chosen["config"].get("lr")
            per[label] = {
                "tuned": next(v for k, v in tuned.items() if label.startswith(k)),
                "chosen": chosen["config"],
                "chosen_at_grid_edge": lr is not None and lr in (grid[0], grid[-1]),
                "solved": len(fin),  # first ‖∇f‖ ≤ tol at a point with a positive definite Hessian
                "of": len(starts),
                "median_grad_evals": float(np.median(chosen["grad_evals"])) if fin else None,
                "grad_evals": chosen["grad_evals"],
                "saddle": sum(r["end"] == "saddle" for r in runs),
                "solved_at_global_min": sum(
                    math.isfinite(r["hit"]) and r["f_at_first_hit"] <= F_GLOBAL_TOL for r in runs
                ),
                "end": [r["end"] for r in runs],
                "first_hit_any": [r["first_hit_any"] for r in runs],
                "f_at_first_hit": [r["f_at_first_hit"] for r in runs],
                "lam_min_at_first_hit": [r["lam_min_at_first_hit"] for r in runs],
                "messages": [r["message"] for r in runs],
                "f_final": [r["f_final"] for r in runs],
                "sweep": [
                    {"config": s["config"], "solved": sum(map(math.isfinite, s["grad_evals"])),
                     "saddle": sum(r["end"] == "saddle" for r in s["runs"]),
                     "median": float(np.median(s["grad_evals"]))} for s in sweep
                ],
            }  # fmt: skip
            cost_table[pi * 10 : pi * 10 + 10, si] = chosen["grad_evals"]
            best_curves[pid][label] = {
                "curve": runs[0]["best"][:E3_BUDGET],
                "end": runs[0]["end"],
                "first_hit_any": runs[0]["first_hit_any"],
                "f_at_first_hit": runs[0]["f_at_first_hit"],
            }
        results["problems"][pid] = per
    save("e3_gd.json", results)

    # Figures: convergence on the default x0, and profiles over the 20 instances.
    show = ["AA(5)-GD", "L-BFGS(5)", "AA(10)-GD", "L-BFGS(10)", "GD fixed α", "Nesterov"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, pid in zip(axes, best_curves, strict=True):
        for j, label in enumerate(show):
            d = best_curves[pid][label]
            c = d["curve"]
            name = label
            if d["end"] == "saddle":  # the run stops at a saddle point: not a solve
                name = f"{label}: SADDLE (f = {d['f_at_first_hit']:.3g})"
            ax.semilogy(
                np.arange(1, len(c) + 1),
                c,
                color=COLORS[j],
                ls="-" if j % 2 == 0 else "--",
                label=name,
            )
            if d["end"] == "saddle":
                k = d["first_hit_any"]
                ax.plot(k, c[k - 1], "X", color=COLORS[j], ms=9, mec="black", mew=0.8)
        ax.axhline(E3_TOL, color="0.6", lw=0.8)
        ax.set_xlim(0, 600)
        ax.set_xlabel("gradient evaluations")
        ax.set_ylabel("best $\\|\\nabla f\\|_2$ so far")
        ax.set_title(f"{pid}, default $x_0$ (X marker = run stops at a saddle point)", fontsize=8)
        ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "e3_gd_convergence")
    sel = [
        labels.index(s)
        for s in (
            *[f"AA({m})-GD" for m in MEMS],
            *[f"L-BFGS({m})" for m in MEMS],
            "Nesterov",
            "GD fixed α",
        )
    ]
    perf = bench.performance_profile_from_costs(cost_table[:, sel], [labels[i] for i in sel])
    fig = bench.plot_performance_profile(
        perf, title="Gradient evals to ‖∇f‖₂ ≤ 1e-6 at a minimizer (20 instances)"
    )
    savefig(fig, "e3_performance_profile")
    data = bench.data_profile_from_costs(cost_table[:, sel], dims, [labels[i] for i in sel])
    fig = bench.plot_data_profile(
        data, title="Data profile (minimizer only): κ = gradient evals / (n + 1)"
    )
    savefig(fig, "e3_data_profile")
    save("e3_profiles.json", {
        "labels": [labels[i] for i in sel],
        "performance_rho_at_1": dict(zip(perf.labels, perf.at(1.0).tolist(), strict=True)),
        "performance_rho_at_2": dict(zip(perf.labels, perf.at(2.0).tolist(), strict=True)),
        "solved_fraction": dict(zip(perf.labels, perf.solved.tolist(), strict=True)),
        "criterion": "first ‖∇f‖₂ ≤ 1e-6 at a point with a positive definite Hessian",
    })  # fmt: skip
    return results


# ======================================================================================
# E4  RNA λ from nonconvex starts
# ======================================================================================

E4 = {"himmelblau": (0.01, 5.0), "beale": (0.01, 2.0)}  # (lr, half-width of the start box)
LAMS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
E4_MEMS = (3, 5, 10)
E4_ITERS = 2000


def outcome(p: Problem, r: Any) -> str:
    if r.converged:
        assert p.hess is not None
        return "minimizer" if np.linalg.eigvalsh(p.hess(np.asarray(r.x)))[0] > 0 else "saddle/max"
    if "diverged" in r.message or "non-finite" in r.message:
        return "diverged"
    return "no convergence"


def e4_rna() -> dict[str, Any]:
    out: dict[str, Any] = {"lams": LAMS, "iters": E4_ITERS, "problems": {}}
    grids: dict[str, dict[str, list[str]]] = {}
    for pid, (lr, h) in E4.items():
        p = problems.get(pid)
        g = np.linspace(-h, h, 8)
        starts = [np.array([a, b]) for b in g for a in g]
        rows: dict[str, Any] = {"lr": lr, "box": h, "n_starts": len(starts)}
        gd = [
            outcome(p, anderson_gd(p, x0=s, m=0, lr=lr, gtol=1e-6, max_iter=E4_ITERS))
            for s in starts
        ]
        rows["GD (AA(0))"] = dict(Counter(gd))
        grids[pid] = {"GD": gd}
        for m in E4_MEMS:
            for lam in LAMS:
                res = [
                    anderson_gd(p, x0=s, m=m, lr=lr, lam=lam, gtol=1e-6, max_iter=E4_ITERS)
                    for s in starts
                ]
                oc = [outcome(p, r) for r in res]
                iters = [r.n_iter for r, o in zip(res, oc, strict=True) if o == "minimizer"]
                # Weight norms of the diverged runs. Step 1 is always the Picard step, c = [1], so
                # ‖c‖₂ = 1 there in every run; only steps k ≥ 2 carry information about AA.
                div_diag = [
                    {
                        "x0": s,
                        "n_iter": r.n_iter,
                        "gd_outcome": q,
                        "max_c_norm_k_ge_2": max(
                            (float(np.linalg.norm(st.info["coefficients"])) for st in r.trace[2:]),
                            default=None,
                        ),
                        "last3_c_norm": [
                            float(np.linalg.norm(st.info["coefficients"])) for st in r.trace[-3:]
                        ],
                        "last_memory": r.trace[-1].info.get("memory"),
                        "max_abs_c_last": float(np.max(np.abs(r.trace[-1].info["coefficients"]))),
                        "c_last": r.trace[-1].info["coefficients"],  # oldest … newest iterate
                    }
                    for s, r, o, q in zip(starts, res, oc, gd, strict=True)
                    if o == "diverged"
                ]
                rows[f"m={m}, lam={lam:g}"] = {
                    "diverged_runs": div_diag,
                    **Counter(oc),
                    "median_iters_to_min": float(np.median(iters)) if iters else None,
                    "diverged_where_gd_converged": sum(
                        o == "diverged" and q in ("minimizer", "saddle/max") for o, q in zip(oc, gd, strict=True)
                    ),
                    "diverged_where_gd_diverged": sum(
                        o == "diverged" and q == "diverged" for o, q in zip(oc, gd, strict=True)
                    ),
                }  # fmt: skip
                if m == 5:
                    grids[pid][f"lam={lam:g}"] = oc
        out["problems"][pid] = rows
    out["grids_m5"] = grids
    save("e4_rna.json", out)

    # Figure: outcome fraction vs λ (m = 5), and the himmelblau start map for λ = 0 and 1e-2.
    cats = ("minimizer", "saddle/max", "no convergence", "diverged")
    ccol = {
        "minimizer": COLORS[2],
        "saddle/max": COLORS[1],
        "no convergence": COLORS[4],
        "diverged": "0.25",
    }
    fig, axes = plt.subplots(
        1, 4, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1, 1]}
    )
    for ax, pid in zip(axes[:2], E4, strict=True):
        cols = ["GD", *[f"lam={lam:g}" for lam in LAMS]]
        bottom = np.zeros(len(cols))
        for c in cats:
            v = np.array([grids[pid][k].count(c) for k in cols]) / 64
            ax.bar(range(len(cols)), v, bottom=bottom, color=ccol[c], label=c, width=0.8)
            bottom += v
        ax.set_xticks(
            range(len(cols)),
            ["GD", "λ=0", *[f"{lam:.0e}" for lam in LAMS[1:]]],
            rotation=45,
            fontsize=7,
        )
        ax.set_title(f"{pid}: AA(5)-GD, 64 starts")
        ax.set_ylabel("fraction of starts")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False, fontsize=8)
    p = problems.get("himmelblau")
    xs = np.linspace(-5.5, 5.5, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = np.vectorize(lambda a, b: p.f(np.array([a, b])))(X, Y)
    g = np.linspace(-5, 5, 8)
    pts = np.array([[a, b] for b in g for a in g])
    for ax, k in zip(axes[2:], ("lam=0", "lam=0.01"), strict=True):
        ax.contour(X, Y, np.log1p(Z), levels=14, colors="0.75", linewidths=0.6)
        for c in cats:
            idx = [i for i, o in enumerate(grids["himmelblau"][k]) if o == c]
            ax.scatter(
                pts[idx, 0], pts[idx, 1], s=18, color=ccol[c], edgecolor="white", linewidth=0.4
            )
        mins = np.array(p.minima)
        ax.scatter(mins[:, 0], mins[:, 1], marker="x", s=30, color="black", linewidth=1.2)
        ax.set_aspect("equal")
        ax.set_title(
            f"himmelblau start map, λ = {float(k[4:]):g}\n(colour = outcome, × = minimizer)",
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "e4_rna_outcomes")
    return out


def main() -> None:
    t0 = time.perf_counter()
    timings = {}
    for name, fn in (("e1", e1_gmres), ("e2", e2_contractions), ("e3", e3_gd), ("e4", e4_rna)):
        t = time.perf_counter()
        fn()
        timings[name] = round(time.perf_counter() - t, 1)
        print(f"{name} done in {timings[name]} s", flush=True)
    save("timings.json", {**timings, "total_s": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    main()
