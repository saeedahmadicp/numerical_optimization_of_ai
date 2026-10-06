"""Experiment: certified step-size schedules vs numopt's first-order baselines.

Questions (see README.md):
  Q1. Does f(x_n) − f* under the convex silver schedule stay below the proved envelope
      r_k L‖x₀ − x*‖² at every n = 2^k − 1 (k = 1..12)?
  Q2. At which horizon n does convex silver lose to constant 1/L steps on strongly convex
      problems, and does the κ-aware silver schedule (Part I) remove that crossover?
  Q3. How do all methods compare on the same oracle budget (performance profiles)?

Protocol: 4 problems × 5 start points (the problem's default x₀ and 4 points drawn with
numopt.core.rng.Rng uniformly from the plotting box); budget N = 4095 oracle calls. An oracle
call is one point at which the method evaluates f and/or ∇f (a line-search trial point is
one call; numopt's Nesterov also evaluates ∇f at x_k only for its stopping test, so it is
charged k calls for k iterations, the textbook count; a sensitivity profile charges it every
gradient it evaluates, ≈ 2k). Every number in README.md is read from results/*.json written
here. Deterministic: no network, no unseeded randomness.

Checks beyond the main protocol (summary.json keys):
  ablation_L, ablation_L_starts  crossover with L_used = s·λmax, s ∈ L_FACTORS, on the
                                 default start (all methods) and on all 5 starts of
                                 quadratic_nd and quadratic_ill (silver vs GD 1/L_used).
  logreg_hypothesis              the quadratic model of logreg_2d at x*, with L_used = λmax(H*)
                                 and with the global L of logreg_2d.
  robustness_extra_starts        Q1/Q2 on 12 fresh starts per problem (Rng(900 + index)).
  Q3_profiles_nesterov_strict    Q3 with Nesterov charged every gradient it evaluates.

Run:  .venv/bin/python research/certified-stepsize-schedules/run.py      (≈ 1 min)
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import method as M  # noqa: E402

from numopt import bench, problems  # noqa: E402
from numopt.core.registry import run as run_registered  # noqa: E402
from numopt.core.rng import Rng  # noqa: E402
from numopt.core.types import Problem, Result  # noqa: E402

N_ITER = 4095  # = 2^12 − 1, the last convex-silver checkpoint (k = 12)
N_STARTS = 5
GTOL_BASELINE = 1e-150  # numopt's GTOL_MIN: run to the budget
EPS = float(np.finfo(np.float64).eps)
#: f is at rounding level when f − f* ≤ 64ε·|f*| (f* ≠ 0) ...
TIE = 64 * EPS
RES = HERE / "results"
FIG = HERE / "figures"


# --------------------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Case:
    problem: Problem
    L: float
    mu: float
    x_star: np.ndarray
    f_star: float
    strongly_convex: bool
    note: str


def _logreg_full() -> Case:
    fs = problems.get("logreg_2d")
    p = Problem(
        id="logreg_2d",
        name="Logistic regression, full batch",
        latex=fs.latex,
        f=fs.f,
        grad=fs.grad,
        hess=fs.hess,
        dim=2,
        domain=fs.domain,
        x0=list(fs.x0),
        minima=(list(fs.minima[0]),),
        tags=("convex", "strongly-convex"),
        extra={"L": fs.extra["L"], "mu": fs.extra["mu"], "f_min": fs.extra["f_min"]},
    )
    return Case(
        p,
        float(fs.extra["L"]),
        float(fs.extra["mu"]),
        np.array(fs.minima[0]),
        float(fs.extra["f_min"]),
        True,
        "L = ¼λmax(XᵀX/n) + λ (global), μ = λ = 1e-2",
    )


def _decay_quadratic() -> Case:
    # Convex control: f(x) = ½ Σ_j λ_j (x_j − c_j)², λ_j = 1/j² (j = 1..200), c_j ~ U(−1, 1)
    # from Rng(7). κ = 4·10⁴ ≫ N, so every horizon here is in the merely convex regime.
    n = 200
    lam = 1.0 / np.arange(1, n + 1) ** 2
    rng = Rng(7)
    c = np.array([rng.uniform(-1.0, 1.0) for _ in range(n)])

    def f(x: Any) -> float:
        d = np.asarray(x, dtype=np.float64) - c
        return 0.5 * float(np.dot(lam * d, d))

    def grad(x: Any) -> np.ndarray:
        return lam * (np.asarray(x, dtype=np.float64) - c)

    p = Problem(
        id="decay_quadratic",
        name="Quadratic with eigenvalues 1/j² (n = 200)",
        latex=r"f(x) = \tfrac12\sum_{j=1}^{200} j^{-2}(x_j - c_j)^2",
        f=f,
        grad=grad,
        hess=lambda x: np.diag(lam),
        dim=n,
        domain=((-2.0, 2.0),) * n,
        x0=[0.0] * n,
        minima=(c.tolist(),),
        tags=("quadratic", "convex"),
        extra={"L": 1.0, "mu": float(lam[-1]), "f_min": 0.0},
    )
    return Case(p, 1.0, float(lam[-1]), c, 0.0, False, "λ_j = 1/j², κ = 40000 (convex regime)")


def _quadratic(pid: str) -> Case:
    p = problems.get(pid)
    A = np.array(p.extra["A"])
    ev = np.linalg.eigvalsh(A)
    return Case(
        p,
        float(ev[-1]),
        float(ev[0]),
        np.array(p.minima[0], dtype=float),
        0.0,
        True,
        f"L = λmax(A) = {ev[-1]:.4g}, μ = λmin(A) = {ev[0]:.4g}",
    )


def cases() -> list[Case]:
    return [
        _quadratic("quadratic_nd"),
        _quadratic("quadratic_ill"),
        _logreg_full(),
        _decay_quadratic(),
    ]


def start_points(case: Case, seed: int) -> list[np.ndarray]:
    """The default x₀, then N_STARTS − 1 points uniform in the plotting box (Rng(seed))."""
    rng = Rng(seed)
    box = case.problem.domain
    pts = [np.asarray(case.problem.x0, dtype=np.float64)]
    for _ in range(N_STARTS - 1):
        pts.append(np.array([rng.uniform(lo, hi) for lo, hi in box]))
    return pts


# --------------------------------------------------------------------------------------
# Methods (the baselines are numopt registry ids with stated parameters)
# --------------------------------------------------------------------------------------

Runner = Callable[[Problem, np.ndarray, Case], Result]


def _reg(mid: str, **kw: Any) -> Callable[[Case], dict[str, Any]]:
    return lambda c: {"_id": mid, **kw}


def method_table() -> dict[str, tuple[Callable[[Case], dict[str, Any]], str]]:
    """label → (case → keyword parameters, stated choice). ``_id`` = registry id or ours."""
    rk = lambda c: math.sqrt(c.L / c.mu)  # noqa: E731
    return {
        "silver (convex)": (lambda c: {"_id": "silver_gd", "L": c.L}, "h_t = 1 + ρ^{ν(t+1)−1}"),
        "silver (κ-aware)": (
            lambda c: {"_id": "silver_gd_strongly_convex", "L": c.L, "mu": c.mu},
            "Part I schedule, auto horizon",
        ),
        "long steps (t=127)": (
            lambda c: {"_id": "long_step_gd", "L": c.L, "pattern": "127"},
            "Grimmer Table 1, t = 127",
        ),
        "OGM1": (lambda c: {"_id": "ogm", "L": c.L}, "N = 4095"),
        "GD 1/L": (
            lambda c: {"_id": "gradient_descent", "step_rule": "fixed", "lr": 1 / c.L},
            "fixed lr = 1/L",
        ),
        "GD Armijo": (
            _reg("gradient_descent", step_rule="backtracking"),
            "numopt default (c₁ = 1e-4, ρ = ½)",
        ),
        "Nesterov": (
            lambda c: {"_id": "nesterov", "lr": 1 / c.L, "beta": (rk(c) - 1) / (rk(c) + 1)},
            "lr = 1/L, β = (√κ−1)/(√κ+1)",
        ),
        "heavy ball": (
            lambda c: {
                "_id": "momentum",
                "lr": 4 / (math.sqrt(c.L) + math.sqrt(c.mu)) ** 2,
                "beta": ((rk(c) - 1) / (rk(c) + 1)) ** 2,
            },
            "Polyak's optimal lr, β for [μ, L]",
        ),
        "BB": (_reg("barzilai_borwein"), "numopt default (BB1, nonmonotone)"),
        "CG-PR+": (_reg("cg_polak_ribiere"), "numopt default (strong Wolfe, c₂ = 0.1)"),
    }


OURS = {"silver_gd", "silver_gd_strongly_convex", "long_step_gd", "ogm"}


class Recorder:
    """Counts oracle calls of line-search methods: distinct points where f or ∇f is evaluated.

    A trial point costs one call whether the method asks for f, ∇f or both there.
    """

    def __init__(self) -> None:
        self.first_cost: dict[bytes, int] = {}
        self.hist: list[tuple[int, float]] = []  # (cost, f) per f evaluation

    def _touch(self, x: Any) -> int:
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key not in self.first_cost:
            self.first_cost[key] = len(self.first_cost) + 1
        return self.first_cost[key]

    def wrap(self, p: Problem) -> Problem:
        f0, g0 = p.f, p.grad
        assert g0 is not None

        def f(x: Any) -> float:
            c = self._touch(x)
            v = float(f0(x))
            self.hist.append((c, v))
            return v

        def g(x: Any) -> np.ndarray:
            self._touch(x)
            return g0(x)

        return dataclasses.replace(p, f=f, grad=g)

    def cost_of(self, x: Any) -> int:
        return self.first_cost[np.asarray(x, dtype=np.float64).tobytes()] - 1


def run_one(label: str, params: dict[str, Any], case: Case, x0: np.ndarray) -> dict[str, Any]:
    """One run; returns per-iterate (cost, gap) and the best-so-far history (cost, gap)."""
    kw = dict(params)
    mid = kw.pop("_id")
    rec = Recorder()
    prob = rec.wrap(case.problem)
    t0 = time.perf_counter()
    if mid in OURS:
        res = M.METHODS[mid](
            prob, x0=x0, gtol=0.0, max_iter=N_ITER + (mid == "silver_gd_strongly_convex"), **kw
        )
    else:
        res = run_registered(mid, prob, x0=x0.tolist(), gtol=GTOL_BASELINE, max_iter=N_ITER, **kw)
    secs = time.perf_counter() - t0
    one_call = mid in OURS or mid in ("nesterov", "momentum") or kw.get("step_rule") == "fixed"
    if one_call:
        # one gradient per iteration by construction: x_k costs k calls (x₀ costs 0); see
        # the module docstring for Nesterov
        costs = [s.k for s in res.trace]
    else:
        costs = [rec.cost_of(s.x) for s in res.trace]
    gaps = [np.nan if s.fun is None else float(s.fun) - case.f_star for s in res.trace]
    dist2 = [
        float(np.sum((np.asarray(s.x, dtype=np.float64) - case.x_star) ** 2)) for s in res.trace
    ]
    if one_call and res.converged and costs[-1] < N_ITER:
        # A run that stopped at its gtol test (here: ∇f(x_k) = 0 or ≤ 1e-150) would keep
        # x_k fixed if continued; pad so every method has a value at every n ≤ N.
        pad = N_ITER - costs[-1]
        costs += list(range(costs[-1] + 1, N_ITER + 1))
        gaps += [gaps[-1]] * pad
        dist2 += [dist2[-1]] * pad
    keep = [i for i, c in enumerate(costs) if c <= N_ITER]
    # best-so-far over every f evaluation (Moré–Wild); Nesterov: over its iterates
    if one_call:
        hist = [(costs[i], gaps[i]) for i in keep]
    else:
        hist = [(c - 1, v - case.f_star) for c, v in rec.hist if c - 1 <= N_ITER]
    out: dict[str, Any] = {
        "label": label,
        "cost": [costs[i] for i in keep],
        "gap": [gaps[i] for i in keep],
        "dist2": [dist2[i] for i in keep],
        "hist": hist,
        "n_iter": res.n_iter,
        "n_fev": res.n_fev,
        "n_gev": res.n_gev,
        "message": res.message,
        "seconds": secs,
    }
    if mid == "nesterov":
        # Sensitivity (README, Q3): the strict count charges every distinct point where numopt's
        # Nesterov evaluates ∇f — the look-ahead y_{k−1} and x_k (its stopping test) — so x_k
        # costs ≈ 2k calls instead of k.
        strict = [(rec.cost_of(s.x), gaps[i]) for i, s in enumerate(res.trace)]
        out["hist_strict"] = [(c, g) for c, g in strict if c <= N_ITER]
    if mid == "silver_gd_strongly_convex":
        out["horizon"] = res.extra["horizon"]
        out["tau_horizon"] = res.extra["tau_horizon"]
    return out


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def _gap_at(run: dict[str, Any], n: int) -> float:
    """Gap of the iterate produced after exactly n oracle calls (fixed-step methods)."""
    i = run["cost"].index(n)
    return float(run["gap"][i])


def dist_floor(case: Case) -> float:
    """Rounding floor of ‖x − x*‖²: an ulp-scale error 8ε·max(1, ‖x*‖∞) per coordinate."""
    return case.problem.dim * (8 * EPS * max(1.0, float(np.max(np.abs(case.x_star))))) ** 2


def gap_floor(case: Case) -> float:
    """Rounding floor of f − f*: max(64ε|f*|, (L/2)·dist_floor). Gaps below it are noise."""
    return max(TIE * abs(case.f_star), 0.5 * case.L * dist_floor(case))


def best_by(hist: list[tuple[int, float]], budget: int) -> float:
    vals = [v for c, v in hist if c <= budget]
    return min(vals) if vals else math.inf


def _dist_at(run: dict[str, Any], n: int) -> float:
    """‖x − x*‖² of the iterate produced after exactly n oracle calls (fixed-step methods)."""
    return float(run["dist2"][run["cost"].index(n)])


def crossover(
    a: list[float], b: list[float], lvl: float, checkpoints: list[int]
) -> tuple[int | None, list[int], list[int]]:
    """First checkpoint from which b is strictly below a (not a tie at ``lvl``) at every later one.

    Returns (crossover n or None, checkpoints where a is strictly ahead, checkpoints that tie).
    """
    tie = [x <= lvl and y <= lvl for x, y in zip(a, b, strict=True)]
    b_wins = [y < x and not t for x, y, t in zip(a, b, tie, strict=True)]
    a_wins = [x < y and not t for x, y, t in zip(a, b, tie, strict=True)]
    cross = next((checkpoints[j] for j in range(len(b_wins)) if all(b_wins[j:])), None)
    return (
        cross,
        [n for n, w in zip(checkpoints, a_wins, strict=True) if w],
        [n for n, t in zip(checkpoints, tie, strict=True) if t],
    )


def analyse(all_runs: dict[str, Any], cs: list[Case]) -> dict[str, Any]:
    checkpoints = [2**k - 1 for k in range(1, 13)]
    env_rows, cross_rows, ka_rows = [], [], []
    for case in cs:
        pid = case.problem.id
        for si, inst in enumerate(all_runs[pid]):
            R2 = inst["R2"]
            sil, con, ka = (
                inst["runs"]["silver (convex)"],
                inst["runs"]["GD 1/L"],
                inst["runs"]["silver (κ-aware)"],
            )
            # Q1: envelope
            for k, n in enumerate(checkpoints, start=1):
                gap = _gap_at(sil, n)
                env = M.silver_rate(k) * case.L * R2
                env_rows.append(
                    {
                        "problem": pid,
                        "start": si,
                        "k": k,
                        "n": n,
                        "gap": gap,
                        "envelope": env,
                        "ratio": gap / env,
                    }
                )
            # Q2: crossover of convex silver vs constant at checkpoints 2^k − 1
            g_s = [_gap_at(sil, n) for n in checkpoints]
            g_c = [_gap_at(con, n) for n in checkpoints]
            cross, ahead, ties = crossover(g_s, g_c, gap_floor(case), checkpoints)
            # the same comparison in ‖x − x*‖², whose rounding floor is far below that of f − f*
            # when f* ≠ 0 (logreg_2d: 2.6e-29 vs 4.0e-15)
            d_s = [_dist_at(sil, n) for n in checkpoints]
            d_c = [_dist_at(con, n) for n in checkpoints]
            cross_d, ahead_d, ties_d = crossover(d_s, d_c, dist_floor(case), checkpoints)
            cross_rows.append(
                {
                    "problem": pid,
                    "start": si,
                    "crossover_n": cross,
                    "silver_ahead_at": ahead,
                    "tie_at": ties,
                    "silver": g_s,
                    "constant": g_c,
                    "crossover_n_dist": cross_d,
                    "silver_ahead_at_dist": ahead_d,
                    "tie_at_dist": ties_d,
                    "silver_dist2": d_s,
                    "constant_dist2": d_c,
                }
            )
            # Q2': κ-aware vs constant at every iterate n ≤ N, at its own checkpoints, and at 2^k − 1
            hz = ka["horizon"]
            g_k_all = np.array(ka["gap"][: N_ITER + 1])
            g_c_all = np.array(con["gap"][: N_ITER + 1])
            lvl = gap_floor(case)
            loses = (g_c_all < g_k_all) & ~((g_k_all <= lvl) & (g_c_all <= lvl))
            own = [m * hz for m in range(1, N_ITER // hz + 1)]
            d_ka = np.array(ka["dist2"])
            # rounding floor of ‖x − x*‖²: one ulp-scale error per coordinate (x* itself is
            # known only to ~1e-16 for logreg_2d)
            floor = dist_floor(case)
            viol = [
                n for n in own if d_ka[n] > ka["tau_horizon"] ** (n // hz) * R2 * (1 + 1e-9) + floor
            ]
            bound_ok = None if not own else not viol
            ka_rows.append(
                {
                    "problem": pid,
                    "start": si,
                    "horizon": hz,
                    "tau_horizon": ka["tau_horizon"],
                    "loses_at_any_iterate": int(loses.sum()),
                    "last_loss_iterate": int(np.flatnonzero(loses)[-1]) if loses.any() else None,
                    "loses_at_own_checkpoints": int(sum(loses[n] for n in own)),
                    "loses_at_2k-1": int(sum(loses[n] for n in checkpoints)),
                    "distance_bound_holds": bound_ok,  # None: no checkpoint ≤ N
                    "distance_checks": len(own),
                    "distance_floor": floor,
                    "kappa_aware_at_2k-1": [float(g_k_all[n]) for n in checkpoints],
                }
            )
    return {
        "checkpoints": checkpoints,
        "envelope": env_rows,
        "crossover": cross_rows,
        "kappa_aware": ka_rows,
    }


def profile_tables(
    all_runs: dict[str, Any],
    labels: list[str],
    taus: tuple[float, ...],
    nesterov_strict: bool = False,
) -> dict[str, Any]:
    """Performance profiles; ``nesterov_strict`` charges Nesterov every gradient it evaluates."""
    out: dict[str, Any] = {}
    for tau in taus:
        rows, ids = [], []
        for pid, insts in all_runs.items():
            for si, inst in enumerate(insts):
                f0gap = inst["f0_gap"]
                row = []
                for lab in labels:
                    r = inst["runs"][lab]
                    h = r["hist_strict"] if nesterov_strict and "hist_strict" in r else r["hist"]
                    hit = [c for c, v in h if v <= tau * f0gap]
                    row.append(float(max(min(hit), 1)) if hit else math.inf)
                rows.append(row)
                ids.append(f"{pid}[{si}]")
        T = np.array(rows)
        prof = bench.performance_profile_from_costs(T, labels, tau=tau)
        out[str(tau)] = {
            "instances": ids,
            "costs": [[None if not math.isfinite(v) else v for v in r] for r in T.tolist()],
            "rho_at_1": dict(zip(labels, prof.at(1.0).tolist(), strict=True)),
            "rho_at_2": dict(zip(labels, prof.at(2.0).tolist(), strict=True)),
            "solved": dict(zip(labels, prof.solved.tolist(), strict=True)),
            "_profile": prof,
        }
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

COLORS = {
    "silver (convex)": "#0072B2",
    "silver (κ-aware)": "#D55E00",
    "long steps (t=127)": "#CC79A7",
    "OGM1": "#009E73",
    "GD 1/L": "#000000",
    "GD Armijo": "#999999",
    "Nesterov": "#E69F00",
    "heavy ball": "#56B4E9",
    "BB": "#8C6D31",
    "CG-PR+": "#7F3C8D",
}


def _style(ax: Any) -> None:
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_schedules(path: Path) -> None:
    t = np.arange(127)
    h_sc, _ = M.silver_sc_schedule(100.0, 128)
    rows = [
        (
            "silver, convex: $h_t = 1+\\rho^{\\nu(t+1)-1}$",
            M.silver_schedule(127),
            "silver (convex)",
        ),
        ("silver, κ-aware (κ = 100, horizon 128): steps ≤ (κ+1)/2", h_sc[:127], "silver (κ-aware)"),
        (
            "Grimmer long steps, pattern t = 127",
            np.array(M.LONG_STEP_PATTERNS["127"]),
            "long steps (t=127)",
        ),
    ]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 5.6), layout="constrained", sharex=True, sharey=True
    )
    for ax, (title, h, lab) in zip(axes, rows, strict=True):
        ax.bar(t, h, width=0.8, color=COLORS[lab])
        ax.axhline(1.0, color="k", lw=0.9)
        ax.axhline(2.0, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9, loc="left")
        ax.text(127.5, 2.0, " 2 (descent limit)", fontsize=7, va="center")
        ax.text(127.5, 1.0, " 1 (GD 1/L)", fontsize=7, va="center")
        _style(ax)
    axes[1].set_ylabel("normalized step $h_t = L\\,\\alpha_t$")
    axes[-1].set_xlabel("iteration t")
    fig.savefig(path)
    plt.close(fig)


def fig_envelope(
    analysis: dict[str, Any], cs: list[Case], pep: dict[str, Any] | None, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    ks = np.arange(1, 13)
    ns = 2**ks - 1
    ax.plot(
        ns,
        [M.silver_rate(int(k)) for k in ks],
        "-",
        color=COLORS["silver (convex)"],
        lw=2,
        label="proved envelope $r_k$ (silver)",
    )
    ax.plot(
        ns, 1 / (4 * ns + 2), "--", color="k", lw=1.2, label="tight bound for GD 1/L: $1/(4n+2)$"
    )
    if pep is not None:
        pn = [r["n"] for r in pep["silver"]]
        ax.plot(
            pn,
            [r["pep"] for r in pep["silver"]],
            "s",
            ms=7,
            mfc="none",
            color=COLORS["silver (convex)"],
            label="PEP worst case (silver, exact SDP)",
        )
    for case in cs:
        pid = case.problem.id
        rows = [
            r for r in analysis["envelope"] if r["problem"] == pid and r["gap"] > gap_floor(case)
        ]
        x = [r["n"] for r in rows]
        y = [r["ratio"] * M.silver_rate(r["k"]) for r in rows]  # gap / (L R²)
        ax.scatter(
            x, y, s=16, marker=PMARK[pid], color=PCOLORS[pid], alpha=0.6, label=f"observed: {pid}"
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1)
    ax.set_xlabel("iteration $n = 2^k - 1$")
    ax.set_ylabel("$(f(x_n) - f^*) / (L\\,\\|x_0 - x^*\\|^2)$")
    ax.set_title(
        "Q1: silver stays under its certified envelope\n(5 starts per problem; gaps at rounding level omitted)",
        fontsize=10,
    )
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_curves(
    all_runs: dict[str, Any], cs: list[Case], analysis: dict[str, Any], path: Path
) -> None:
    show = [
        "silver (convex)",
        "silver (κ-aware)",
        "GD 1/L",
        "OGM1",
        "Nesterov",
        "long steps (t=127)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    for ax, case in zip(axes.ravel(), cs, strict=True):
        pid = case.problem.id
        inst = all_runs[pid][0]
        f0 = inst["f0_gap"]
        for lab in show:
            r = inst["runs"][lab]
            g = np.maximum(np.array(r["gap"], dtype=float), gap_floor(case)) / f0
            ax.plot(
                np.array(r["cost"]) + 1,
                g,
                lw=1.1 if lab != "GD 1/L" else 1.6,
                color=COLORS[lab],
                label=lab,
                alpha=0.9,
            )
        cr = next(c for c in analysis["crossover"] if c["problem"] == pid and c["start"] == 0)
        if cr["crossover_n"] is not None:
            ax.axvline(cr["crossover_n"] + 1, color="#0072B2", ls=":", lw=1)
            ax.text(
                cr["crossover_n"] + 1,
                1e2,
                f"crossover n = {cr['crossover_n']} ",
                color="#0072B2",
                fontsize=8,
                ha="right",
                va="top",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhspan(1e-40, gap_floor(case) / f0 * 1.5, color="#999999", alpha=0.12, lw=0)
        ax.set_ylim(max(gap_floor(case) / f0 / 10, 1e-34), 1e3)
        ax.set_title(f"{pid}  ({case.note})", fontsize=9)
        ax.set_xlabel("oracle calls + 1")
        ax.set_ylabel("$(f(x_k) - f^*)/(f(x_0) - f^*)$")
        _style(ax)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(
        "Q2: convergence from the default start (raw iterates, not best-so-far; grey band = rounding floor)"
    )
    fig.savefig(path)
    plt.close(fig)


PCOLORS = {
    "quadratic_nd": "#0072B2",
    "quadratic_ill": "#D55E00",
    "logreg_2d": "#009E73",
    "decay_quadratic": "#CC79A7",
}
PMARK = {"quadratic_nd": "o", "quadratic_ill": "^", "logreg_2d": "D", "decay_quadratic": "v"}


def fig_ratio(analysis: dict[str, Any], cs: list[Case], path: Path) -> None:
    """Gap ratios at n = 2^k − 1, both gaps clipped at the rounding floor (ties → ratio 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained", sharey=True)
    ns = np.array(analysis["checkpoints"])
    for case in cs:
        pid = case.problem.id
        lvl = gap_floor(case)
        for r in [c for c in analysis["crossover"] if c["problem"] == pid]:
            kr = next(
                k
                for k in analysis["kappa_aware"]
                if k["problem"] == pid and k["start"] == r["start"]
            )
            c = np.maximum(np.array(r["constant"]), lvl)
            for ax, num in zip(axes, (r["silver"], kr["kappa_aware_at_2k-1"]), strict=True):
                ratio = np.maximum(np.array(num), lvl) / c
                ax.plot(
                    ns,
                    ratio,
                    "-",
                    marker=PMARK[pid],
                    ms=3.5,
                    lw=0.9,
                    alpha=0.75,
                    color=PCOLORS[pid],
                    label=pid if r["start"] == 0 else None,
                )
    for ax, t in zip(axes, ("convex silver ÷ GD 1/L", "κ-aware silver ÷ GD 1/L"), strict=True):
        ax.axhline(1.0, color="k", lw=1)
        ax.axhspan(1.0, 1e22, color="#999999", alpha=0.08, lw=0)
        ax.text(1.1, 3e15, "constant step better", fontsize=8, color="#555555")
        ax.text(1.1, 1e-17, "schedule better", fontsize=8, color="#555555")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-22, 1e22)
        ax.set_xlabel("iteration $n = 2^k - 1$")
        ax.set_title(t, fontsize=10)
        _style(ax)
    axes[0].set_ylabel("ratio of gaps $f(x_n) - f^*$ (clipped at rounding floor)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Q2: where constant 1/L steps overtake each schedule (5 starts per problem)")
    fig.savefig(path)
    plt.close(fig)


def fig_profiles(prof: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
    for ax, (tau, d) in zip(axes, prof.items(), strict=True):
        bench.plot_performance_profile(
            d["_profile"], ax=ax, title=f"performance profile, τ = {float(tau):g}"
        )
    handles, names = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, names, loc="outside right center", fontsize=8, frameon=False)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------------


L_FACTORS = (1.0, 1.0001, 1.001, 1.003, 1.01, 1.1, 2.0)
CHECKPOINTS = [2**k - 1 for k in range(1, 13)]


def ablation_L(case: Case) -> list[dict[str, Any]]:
    """Each method with an overestimated L (still a valid constant), default start.

    The schedules are tuned to the curvature L itself; with L_used = s·L the largest
    normalized curvature λ/L_used is 1/s < 1. μ is unchanged, so κ grows with s.
    Returns relative gaps (f − f*)/(f(x₀) − f*) at n = 2^k − 1 and the crossover n of convex
    silver vs GD 1/L_used (same definition as Q2).
    """
    table = method_table()
    x0 = np.asarray(case.problem.x0, dtype=np.float64)
    f0 = float(case.problem.f(x0) - case.f_star)
    out = []
    for s_ in L_FACTORS:
        c2 = dataclasses.replace(case, L=s_ * case.L)
        row: dict[str, Any] = {"L_factor": s_, "gaps": {}}
        for lab in ("silver (convex)", "silver (κ-aware)", "long steps (t=127)", "OGM1", "GD 1/L"):
            r = run_one(lab, table[lab][0](c2), case, x0)
            row["gaps"][lab] = [_gap_at(r, n) / f0 for n in CHECKPOINTS]
        g_s, g_c = row["gaps"]["silver (convex)"], row["gaps"]["GD 1/L"]
        row["crossover_n"] = crossover(g_s, g_c, gap_floor(case) / f0, CHECKPOINTS)[0]
        out.append(row)
    return out


def crossovers_with_L(
    case: Case, starts: list[np.ndarray], factors: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Crossover n of convex silver vs GD 1/L_used, L_used = s·case.L, for every start.

    Both in f − f* (``crossover_n``) and in ‖x − x*‖² (``crossover_n_dist``); same
    definition and rounding floors as Q2.
    """
    table = method_table()
    out = []
    for s_ in factors:
        c2 = dataclasses.replace(case, L=s_ * case.L)
        cr, cr_d, last_ahead = [], [], []
        for x0 in starts:
            sil = run_one("silver (convex)", table["silver (convex)"][0](c2), case, x0)
            con = run_one("GD 1/L", table["GD 1/L"][0](c2), case, x0)
            c, ahead, _ = crossover(
                [_gap_at(sil, n) for n in CHECKPOINTS],
                [_gap_at(con, n) for n in CHECKPOINTS],
                gap_floor(case),
                CHECKPOINTS,
            )
            cd = crossover(
                [_dist_at(sil, n) for n in CHECKPOINTS],
                [_dist_at(con, n) for n in CHECKPOINTS],
                dist_floor(case),
                CHECKPOINTS,
            )[0]
            cr.append(c)
            cr_d.append(cd)
            last_ahead.append(max(ahead) if ahead else None)
        out.append(
            {
                "L_factor": s_,
                "L_used": s_ * case.L,
                "crossover_n": cr,
                "crossover_n_dist": cr_d,
                "silver_last_ahead_at": last_ahead,
            }
        )
    return out


def _logreg_local_quadratic() -> Case:
    """The quadratic model of logreg_2d at x*: f(x) = ½(x − x*)ᵀH*(x − x*), H* = ∇²f(x*).

    It has the curvature of logreg_2d near x* but no non-quadratic part. It isolates the
    hypothesis that logreg_2d shows no crossover because its global L is ≈ 6× λmax(H*).
    """
    fs = problems.get("logreg_2d")
    xs = np.array(fs.minima[0], dtype=np.float64)
    H = np.asarray(fs.hess(xs), dtype=np.float64)
    H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)

    def f(x: Any) -> float:
        d = np.asarray(x, dtype=np.float64) - xs
        return 0.5 * float(d @ H @ d)

    def grad(x: Any) -> np.ndarray:
        return H @ (np.asarray(x, dtype=np.float64) - xs)

    p = Problem(
        id="logreg_2d_local_quadratic",
        name="Quadratic model of logreg_2d at its minimizer",
        latex=r"f(x) = \tfrac12 (x - x^\star)^\top \nabla^2 f_{\mathrm{logreg}}(x^\star)(x - x^\star)",
        f=f,
        grad=grad,
        hess=lambda x: H,
        dim=2,
        domain=fs.domain,
        x0=list(fs.x0),
        minima=(xs.tolist(),),
        tags=("quadratic", "convex", "strongly-convex"),
        extra={"L": float(ev[-1]), "mu": float(ev[0]), "f_min": 0.0},
    )
    return Case(p, float(ev[-1]), float(ev[0]), xs, 0.0, True, "H* = ∇²f(x*) of logreg_2d")


def logreg_hypothesis(cs: list[Case]) -> dict[str, Any]:
    """Test: is L_global/λmax(H*) ≈ 6 why logreg_2d has no crossover?

    Runs convex silver vs GD 1/L_used on the quadratic model of logreg_2d at x* from the same
    5 starts as logreg_2d, with L_used = λmax(H*) (curvature equal to L: a crossover is
    predicted) and L_used = the global L of logreg_2d (ratio ≈ 6.2: no crossover predicted).
    """
    ci = next(i for i, c in enumerate(cs) if c.problem.id == "logreg_2d")
    starts = start_points(cs[ci], seed=100 + ci)
    loc = _logreg_local_quadratic()
    ratio = cs[ci].L / loc.L
    return {
        "lambda_H_star": [loc.mu, loc.L],
        "L_global": cs[ci].L,
        "L_global_over_lambda_max": ratio,
        "rows": crossovers_with_L(loc, starts, (1.0, ratio)),
    }


N_EXTRA = 12
ROBUST_LABELS = ("silver (convex)", "silver (κ-aware)", "GD 1/L")


def robustness_starts(cs: list[Case]) -> dict[str, Any]:
    """Q1/Q2 on N_EXTRA fresh starts per problem (Rng(900 + problem index), same box).

    These starts are not used anywhere else: they check that the 5-start numbers generalize.
    """
    table = method_table()
    runs: dict[str, Any] = {}
    for ci, case in enumerate(cs):
        rng = Rng(900 + ci)
        insts = []
        for _ in range(N_EXTRA):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in case.problem.domain])
            insts.append(
                {
                    "x0": x0.tolist(),
                    "R2": float(np.sum((x0 - case.x_star) ** 2)),
                    "f0_gap": float(case.problem.f(x0)) - case.f_star,
                    "runs": {
                        lab: run_one(lab, table[lab][0](case), case, x0) for lab in ROBUST_LABELS
                    },
                }
            )
        runs[case.problem.id] = insts
    an = analyse(runs, cs)
    env = an["envelope"]
    return {
        "starts_per_problem": N_EXTRA,
        "seed": "Rng(900 + problem index)",
        "envelope_checks": len(env),
        "envelope_violations": sum(r["gap"] > r["envelope"] for r in env),
        "envelope_max_ratio": max(r["ratio"] for r in env),
        "crossover": {
            c.problem.id: [
                r["crossover_n"] for r in an["crossover"] if r["problem"] == c.problem.id
            ]
            for c in cs
        },
        "crossover_dist": {
            c.problem.id: [
                r["crossover_n_dist"] for r in an["crossover"] if r["problem"] == c.problem.id
            ]
            for c in cs
        },
        "kappa_aware_last_loss": {
            c.problem.id: [
                r["last_loss_iterate"] for r in an["kappa_aware"] if r["problem"] == c.problem.id
            ]
            for c in cs
        },
        "kappa_aware_horizon": {
            c.problem.id: next(
                r["horizon"] for r in an["kappa_aware"] if r["problem"] == c.problem.id
            )
            for c in cs
        },
        "kappa_aware_distance_bound_holds": {
            c.problem.id: [
                r["distance_bound_holds"] for r in an["kappa_aware"] if r["problem"] == c.problem.id
            ]
            for c in cs
        },
    }


def fig_ablation(
    abl: list[dict[str, Any]], abl_starts: dict[str, list[dict[str, Any]]], case: Case, path: Path
) -> None:
    ns = np.array(CHECKPOINTS)
    lvl = gap_floor(case) / float(case.problem.f(case.problem.x0) - case.f_star)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), layout="constrained")
    cmap = plt.get_cmap("viridis")
    for i, row in enumerate(abl):
        col = cmap(i / max(1, len(abl) - 1))
        lab = f"$L_{{used}}$ = {row['L_factor']:g}·λmax"
        for ax, key in zip(axes[:2], ("silver (convex)", "GD 1/L"), strict=True):
            ax.plot(ns, np.maximum(row["gaps"][key], lvl), "-o", ms=3, color=col, label=lab)
    for ax, t in zip(axes[:2], ("convex silver", "GD with step 1/L_used"), strict=True):
        ax.axhspan(1e-40, lvl * 1.5, color="#999999", alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(lvl / 10, 2)
        ax.set_xlabel("iteration $n = 2^k - 1$")
        ax.set_title(f"{t} on quadratic_nd (default start)", fontsize=10)
        _style(ax)
    axes[1].sharey(axes[0])
    axes[0].set_ylabel("$(f(x_n) - f^*)/(f(x_0) - f^*)$")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    # panel 3: crossover n vs s, every start of both quadratics
    ax = axes[2]
    NONE = 2**14
    for off, (pid, rows) in zip((-0.12, 0.12), abl_starts.items(), strict=True):
        for i, row in enumerate(rows):
            ys = [NONE if c is None else c + 1 for c in row["crossover_n"]]
            jit = np.linspace(-0.08, 0.08, len(ys))
            ax.scatter(
                i + off + jit,
                ys,
                s=18,
                marker=PMARK[pid],
                color=PCOLORS[pid],
                alpha=0.8,
                label=pid if i == 0 else None,
            )
    ax.axhline(NONE / 1.4, color="k", lw=0.6, ls=":")
    ax.set_yscale("log", base=2)
    ax.set_yticks(
        [2, 8, 32, 128, 512, 1024, 2048, NONE],
        ["1", "7", "31", "127", "511", "1023", "2047", "none"],
    )
    ax.set_xticks(range(len(L_FACTORS)), [f"{s_:g}" for s_ in L_FACTORS], fontsize=8)
    ax.set_xlabel(r"$s = L_{used}/\lambda_{\max}$")
    ax.set_ylabel("crossover $n$ (GD 1/L_used ahead from here on)")
    ax.set_title("crossover per start (5 starts per problem)", fontsize=10)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        "Ablation: within 4095 steps, the crossover needs λmax within 0.1–0.3 % of the L given "
        "to the schedule"
    )
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    cs = cases()
    table = method_table()
    labels = list(table)
    all_runs: dict[str, Any] = {}
    for ci, case in enumerate(cs):
        insts = []
        for si, x0 in enumerate(start_points(case, seed=100 + ci)):
            f0 = float(case.problem.f(x0))
            inst = {
                "x0": x0.tolist(),
                "R2": float(np.sum((x0 - case.x_star) ** 2)),
                "f0_gap": f0 - case.f_star,
                "runs": {},
            }
            for lab in labels:
                inst["runs"][lab] = run_one(lab, table[lab][0](case), case, x0)
            insts.append(inst)
            print(f"{case.problem.id}[{si}] done ({time.perf_counter() - t_start:.1f} s)")
        all_runs[case.problem.id] = insts

    analysis = analyse(all_runs, cs)
    prof = profile_tables(all_runs, labels, (1e-3, 1e-6))
    prof_strict = profile_tables(all_runs, labels, (1e-3, 1e-6), nesterov_strict=True)
    pep_path = RES / "pep.json"
    pep = json.loads(pep_path.read_text()) if pep_path.exists() else None

    env = analysis["envelope"]
    summary = {
        "N": N_ITER,
        "starts_per_problem": N_STARTS,
        "problems": {
            c.problem.id: {
                "L": c.L,
                "mu": c.mu,
                "kappa": c.L / c.mu,
                "note": c.note,
                "dim": c.problem.dim,
            }
            for c in cs
        },
        "methods": {lab: desc for lab, (_, desc) in table.items()},
        "Q1_envelope": {
            "checks": len(env),
            "violations": sum(r["gap"] > r["envelope"] for r in env),
            "max_ratio": max(r["ratio"] for r in env),
            "max_ratio_at": max(env, key=lambda r: r["ratio"]),
            "max_ratio_per_problem": {
                c.problem.id: max(r["ratio"] for r in env if r["problem"] == c.problem.id)
                for c in cs
            },
        },
        "Q2_crossover": [{k: v for k, v in r.items()} for r in analysis["crossover"]],
        "Q2_kappa_aware": analysis["kappa_aware"],
        "Q3_profiles": {
            tau: {k: v for k, v in d.items() if k != "_profile"} for tau, d in prof.items()
        },
        "final_gap_median": {
            c.problem.id: {
                lab: float(
                    np.median(
                        [
                            best_by(i["runs"][lab]["hist"], N_ITER) / i["f0_gap"]
                            for i in all_runs[c.problem.id]
                        ]
                    )
                )
                for lab in labels
            }
            for c in cs
        },
        "oracle_calls_used": {
            c.problem.id: {
                lab: [max(i["runs"][lab]["cost"]) for i in all_runs[c.problem.id]] for lab in labels
            }
            for c in cs
        },
        "Q3_profiles_nesterov_strict": {
            tau: {k: v for k, v in d.items() if k not in ("_profile", "costs", "instances")}
            for tau, d in prof_strict.items()
        },
        "ablation_L": ablation_L(cs[0]),
        "ablation_L_starts": {
            c.problem.id: crossovers_with_L(c, start_points(c, seed=100 + ci), L_FACTORS)
            for ci, c in enumerate(cs[:2])
        },
        "logreg_hypothesis": logreg_hypothesis(cs),
        "robustness_extra_starts": robustness_starts(cs),
        "runtime_seconds": None,
    }
    fig_schedules(FIG / "schedules.png")
    fig_envelope(analysis, cs, pep, FIG / "envelope.png")
    fig_curves(all_runs, cs, analysis, FIG / "convergence.png")
    fig_ratio(analysis, cs, FIG / "crossover_ratio.png")
    fig_profiles(prof, FIG / "performance_profiles.png")
    fig_ablation(summary["ablation_L"], summary["ablation_L_starts"], cs[0], FIG / "ablation_L.png")
    summary["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float, allow_nan=True))
    (RES / "envelope.json").write_text(json.dumps(env, indent=1))
    print(
        json.dumps(
            {k: summary[k] for k in ("Q1_envelope", "runtime_seconds")}, indent=1, default=float
        )
    )
    for r in analysis["crossover"]:
        print("crossover", r["problem"], r["start"], r["crossover_n"])
    for r in analysis["kappa_aware"]:
        print("kappa", {k: v for k, v in r.items() if k != "kappa_aware_at_2k-1"})
    for tau, d in summary["Q3_profiles"].items():
        print(tau, "rho(1)", d["rho_at_1"], "\n solved", d["solved"])
    for pid, rows in summary["ablation_L_starts"].items():
        for r in rows:
            print("ablation", pid, r["L_factor"], r["crossover_n"], r["crossover_n_dist"])
    print("logreg hypothesis", json.dumps(summary["logreg_hypothesis"], default=float))
    print("robustness", json.dumps(summary["robustness_extra_starts"], default=float))


if __name__ == "__main__":
    main()
