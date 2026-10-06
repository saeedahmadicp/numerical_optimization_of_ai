"""Experiment: ARC and gradient-regularized Newton vs the numopt Newton / trust-region baselines.

Run from the repository root:

    .venv/bin/python research/regularized-newton-arc/run.py

Deterministic (no random numbers, no network). Writes results/*.json and figures/*.{svg,png} next
to this file. See README.md for the protocol and the questions.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import problems as study  # noqa: E402
from method import arc, reg_newton  # noqa: E402

from numopt import bench, problems, run  # noqa: E402
from numopt.core.types import Problem, Result  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
GTOL = 1e-8  # ‖∇f‖∞ ≤ GTOL: the stopping test of every method here
MAX_ITER = 500  # the same iteration budget for every method
EIG_TOL = 1e-8  # λ_min ≥ −EIG_TOL·max(1, ‖∇²f‖) counts as a local minimizer

# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------

Solver = Callable[[Problem, Any], Result]


def _registry(mid: str) -> Solver:
    return lambda p, x0: run(mid, p, x0=x0, gtol=GTOL, max_iter=MAX_ITER)


#: label → (solver factory taking the problem's H for the fixed variant, colour, line style).
#: Colours: the reference categorical palette in fixed slot order; pure Newton is the recessive
#: reference series (gray, dotted). Line style is the secondary encoding: studied methods solid,
#: baselines dashed.
STYLE: dict[str, tuple[str, str]] = {
    "ARC": ("#eb6834", "-"),
    "RegN-AdaN": ("#2a78d6", "-"),
    "RegN-SU": ("#4a3aa7", "-"),
    "RegN-fixed": ("#1baf7a", "-"),
    "trust_region_exact": ("#eda100", "--"),
    "trust_region_steihaug": ("#e87ba4", "--"),
    "damped_newton": ("#008300", "--"),
    "modified_newton": ("#e34948", "--"),
    "pure_newton": ("#8a8985", ":"),
}


def methods_for(fixed_H: float | None) -> dict[str, Solver]:
    out: dict[str, Solver] = {
        "ARC": lambda p, x0: arc(p, x0=x0, gtol=GTOL, max_iter=MAX_ITER),
        "RegN-AdaN": lambda p, x0: reg_newton(
            p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, variant="adan", H=1.0
        ),
        "RegN-SU": lambda p, x0: reg_newton(
            p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, variant="super_universal", H=1.0, alpha=1.0
        ),
    }
    if fixed_H is not None:
        out["RegN-fixed"] = lambda p, x0: reg_newton(
            p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, variant="fixed", H=fixed_H
        )
    for mid in (
        "trust_region_exact",
        "trust_region_steihaug",
        "damped_newton",
        "modified_newton",
        "pure_newton",
    ):
        out[mid] = _registry(mid)
    return out


# --------------------------------------------------------------------------------------
# Problems and start points
# --------------------------------------------------------------------------------------


def convex_starts() -> list[tuple[float, float]]:
    """8 × 8 grid on [−10.5, 10.5]² (spacing 3; it contains no minimizer of the study problems)."""
    t = np.linspace(-10.5, 10.5, 8)
    return [(float(a), float(b)) for a in t for b in t]


def nonconvex_starts(p: Problem) -> list[tuple[float, float]]:
    """9 × 9 grid on the problem's plotting domain, minus starts with ‖∇f(x0)‖∞ ≤ GTOL."""
    (a0, a1), (b0, b1) = p.domain
    pts = [(float(a), float(b)) for a in np.linspace(a0, a1, 9) for b in np.linspace(b0, b1, 9)]
    return [x for x in pts if float(np.max(np.abs(p.grad(np.array(x))))) > GTOL]


def hessian_lipschitz_estimate(p: Problem, R: float = 12.0, n_grid: int = 61) -> float:
    """L̂₂ = max over a grid on [−R, R]² and 36 directions u of ‖(∇²f(x+hu) − ∇²f(x−hu))/2h‖₂.

    A grid estimate, i.e. a lower bound on the true L₂ = sup_x ‖∇³f(x)‖ (for sqrt1p the radial
    third derivative gives 0.8587 exactly; see README).
    """
    h = 1e-4
    ts = np.linspace(0.0, np.pi, 36, endpoint=False)
    U = np.stack([np.cos(ts), np.sin(ts)], axis=1)  # (36, 2)
    best = 0.0
    for a in np.linspace(-R, R, n_grid):
        for b in np.linspace(-R, R, n_grid):
            x = np.array([a, b])
            for u in U:
                D = (p.hess(x + h * u) - p.hess(x - h * u)) / (2.0 * h)
                best = max(best, float(np.linalg.norm(D, 2)))
    return best


# --------------------------------------------------------------------------------------
# Running and classifying
# --------------------------------------------------------------------------------------


def classify(p: Problem, r: Result, *, has_minimizer: bool) -> str:
    """'min' | 'saddle' | 'fail', from the final x with the exact (uncounted) ∇f and ∇²f.

    'min': ‖∇f‖∞ ≤ GTOL and λ_min(∇²f) ≥ −EIG_TOL·max(1, ‖∇²f‖) ('solved' on logistic_sep, which
    has no minimizer: there only the gradient test applies). 'saddle': ‖∇f‖∞ ≤ GTOL with a negative
    eigenvalue. 'fail': anything else (divergence, max_iter, breakdown).
    """
    x = np.asarray(r.x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return "fail"
    with np.errstate(all="ignore"):
        g = p.grad(x)
    if not (np.all(np.isfinite(g)) and float(np.max(np.abs(g))) <= GTOL):
        return "fail"
    if not has_minimizer:
        return "min"
    lam = np.linalg.eigvalsh(p.hess(x))
    return "min" if lam[0] >= -EIG_TOL * max(1.0, float(np.max(np.abs(lam)))) else "saddle"


def near_minimizer(p: Problem, r: Result, *, has_minimizer: bool) -> bool:
    """A failed run that stopped at a local minimizer but above GTOL: ‖∇f‖∞ ≤ 1e-6 and
    λ_min(∇²f) > 0 (e.g. an Armijo search that cannot resolve a decrease below the rounding level
    of f). Reported separately so that such stops are not mistaken for saddle stalls."""
    x = np.asarray(r.x, dtype=np.float64)
    if not (has_minimizer and np.all(np.isfinite(x))):
        return False
    with np.errstate(all="ignore"):
        g = p.grad(x)
        H = p.hess(x)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return False
    return float(np.max(np.abs(g))) <= 1e-6 and float(np.linalg.eigvalsh(H)[0]) > 0.0


def hessians_along_trace(r: Result) -> np.ndarray:
    """Cumulative Hessian evaluations at each trace step (1 at k = 0, +1 per accepted iterate)."""
    acc = (
        [1]
        + [
            1 if s.info.get("accepted", True) else 0  # Newton / RegN steps are always accepted
            for s in r.trace[1:]
        ]
    )
    return np.cumsum(acc)


def run_set(
    pid: str, p: Problem, starts: list[tuple[float, float]], meths: dict[str, Solver], has_min: bool
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for label, fn in meths.items():
        rows = []
        for x0 in starts:
            r = fn(p, x0)
            outcome = classify(p, r, has_minimizer=has_min)
            rows.append(
                {
                    "x0": list(x0),
                    "outcome": outcome,
                    "near_min": outcome == "fail" and near_minimizer(p, r, has_minimizer=has_min),
                    "n_hev": r.n_hev,
                    "n_gev": r.n_gev,
                    "n_fev": r.n_fev,
                    "n_iter": r.n_iter,
                    "converged": r.converged,
                    "fun": r.fun if r.fun is not None and math.isfinite(r.fun) else None,
                    "x": [float(v) for v in np.asarray(r.x, dtype=float)]
                    if np.all(np.isfinite(r.x))
                    else None,
                    "message": r.message[:160],
                }
            )
        out[label] = rows
    return out


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [r for r in rows if r["outcome"] == "min"]
    h = np.array([r["n_hev"] for r in solved], dtype=float)
    return {
        "n": len(rows),
        "min": len(solved),
        "saddle": sum(r["outcome"] == "saddle" for r in rows),
        "fail": sum(r["outcome"] == "fail" for r in rows),
        "fail_near_min": sum(bool(r["near_min"]) for r in rows),
        "fev_median": float(np.median([r["n_fev"] for r in solved])) if solved else None,
        "gev_median": float(np.median([r["n_gev"] for r in solved])) if solved else None,
        "hev_median": float(np.median(h)) if h.size else None,
        "hev_q25": float(np.percentile(h, 25)) if h.size else None,
        "hev_q75": float(np.percentile(h, 75)) if h.size else None,
        "hev_max": int(h.max()) if h.size else None,
    }


def paired(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], key: str = "n_hev"
) -> dict[str, Any]:
    """Counts ``key`` (Hessians by default) of A vs B on the starts both solve: wins/ties/losses
    and a two-sided sign test on the non-tied pairs (binomial, p = ½)."""
    both = [
        (a[key], b[key])
        for a, b in zip(rows_a, rows_b, strict=True)
        if a["outcome"] == "min" and b["outcome"] == "min"
    ]
    if not both:
        return {"n_both": 0}
    a = np.array([t[0] for t in both], dtype=float)
    b = np.array([t[1] for t in both], dtype=float)
    wins, losses = int(np.sum(a < b)), int(np.sum(a > b))
    p_value = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return {
        "n_both": len(both),
        "a_fewer": wins,
        "tie": int(np.sum(a == b)),
        "a_more": losses,
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff_a_minus_b": float(np.median(a - b)),
        "sign_test_p": float(p_value),
    }


#: The grid of the sensitivity sweep: σ₀ (ARC), H₀ (AdaN, SU) and Δ₀ (trust_region_exact).
#: radius0 > 100 = max_radius is clamped to 100, so the grid stops there.
SENS_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def sensitivity_sweep() -> dict[str, Any]:
    """Median Hessian count on the convex set for every constant of SENS_GRID.

    Returns {"grid", "by_method": {method: {value: {pid: summary}}}, "best": {method: {pid: …}},
    "best_ranking": {pid: [(method, value, median), …]}, "best_paired": {pid: {"A vs B": paired}}}.
    The best value is chosen on the same 64 starts it is scored on (in-sample: an optimistic
    estimate for every method alike).
    """
    sweeps: dict[str, Callable[[float], Solver]] = {
        "ARC sigma0": lambda v: lambda p, x0: arc(p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, sigma0=v),
        "RegN-AdaN H0": lambda v: (
            lambda p, x0: reg_newton(p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, variant="adan", H=v)
        ),
        "RegN-SU H0": lambda v: (
            lambda p, x0: reg_newton(
                p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, variant="super_universal", H=v, alpha=1.0
            )
        ),
        "trust_region_exact radius0": lambda v: (
            lambda p, x0: run(
                "trust_region_exact", p, x0=x0, gtol=GTOL, max_iter=MAX_ITER, radius0=v
            )
        ),
    }
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    all_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for lab, make in sweeps.items():
        by_method[lab] = {}
        for val in SENS_GRID:
            by_method[lab][f"{val:g}"] = {}
            for pid in study.CONVEX:
                p = study.get(pid)
                rows = run_set(
                    pid, p, convex_starts(), {lab: make(val)}, has_min=pid != "logistic_sep"
                )[lab]
                all_rows[(lab, f"{val:g}", pid)] = rows
                by_method[lab][f"{val:g}"][pid] = summarize(rows)
    best: dict[str, dict[str, Any]] = {}
    ranking: dict[str, list[list[Any]]] = {}
    for pid in study.CONVEX:
        entries = []
        for lab in sweeps:
            cands = [
                (s["hev_median"], s["gev_median"], float(v), v)
                for v, per in by_method[lab].items()
                if (s := per[pid])["min"] == s["n"]
            ]
            if not cands:
                best.setdefault(lab, {})[pid] = None
                continue
            med, gmed, _, v = min(cands)
            best.setdefault(lab, {})[pid] = {
                "value": float(v),
                "hev_median": med,
                "gev_median": gmed,
            }
            entries.append([lab.rsplit(" ", 1)[0], float(v), med, gmed])
        ranking[pid] = sorted(entries, key=lambda e: (e[2], e[3]))
    # Paired Hessian counts with every method at its best constant for the problem.
    best_paired: dict[str, dict[str, Any]] = {}
    pairs = (
        ("RegN-AdaN H0", "ARC sigma0"),
        ("RegN-SU H0", "ARC sigma0"),
        ("RegN-AdaN H0", "trust_region_exact radius0"),
        ("RegN-SU H0", "trust_region_exact radius0"),
        ("ARC sigma0", "trust_region_exact radius0"),
    )
    for pid in study.CONVEX:
        best_paired[pid] = {}
        for a, b in pairs:
            ba, bb = best[a][pid], best[b][pid]
            if ba is None or bb is None:
                continue
            key = f"{a} = {ba['value']:g} vs {b} = {bb['value']:g}"
            best_paired[pid][key] = paired(
                all_rows[(a, f"{ba['value']:g}", pid)], all_rows[(b, f"{bb['value']:g}", pid)]
            )
    return {
        "grid": list(SENS_GRID),
        "by_method": by_method,
        "best": best,
        "best_ranking": ranking,
        "best_paired": best_paired,
    }


def start_distances(pid: str, p: Problem) -> dict[str, Any]:
    """Distance of each convex start to the (unique) minimizer x*, for the Setup section."""
    if not p.minima:
        return {"x_star": None}
    xs = np.asarray(p.minima[0], dtype=float)
    d = np.array([float(np.linalg.norm(np.asarray(x0) - xs)) for x0 in convex_starts()])
    return {
        "x_star": xs.tolist(),
        "min_distance": float(d.min()),
        "nearest_start": list(convex_starts()[int(d.argmin())]),
        "n_within_3": int(np.sum(d < 3.0)),
        "median_distance": float(np.median(d)),
    }


def converged_flag_audit(
    data: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, int]]:
    """Result.converged against the independent outcome: how often converged=True at a stop that
    is not a minimizer (saddle or fail), and converged=False at a minimizer."""
    labels = list(next(iter(data.values())).keys())
    out: dict[str, dict[str, int]] = {}
    for lab in labels:
        rows = [r for per in data.values() for r in per[lab]]
        out[lab] = {
            "runs": len(rows),
            "converged_at_saddle": sum(r["converged"] and r["outcome"] == "saddle" for r in rows),
            "converged_at_fail": sum(r["converged"] and r["outcome"] == "fail" for r in rows),
            "not_converged_at_min": sum(
                (not r["converged"]) and r["outcome"] == "min" for r in rows
            ),
        }
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"


def _style_axes(ax: Any) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=8)


def plot_outcome_maps(data: dict[str, dict[str, list[dict[str, Any]]]], labels: list[str]) -> None:
    """One panel per (method, problem): each start coloured by its Hessian count (one sequential
    hue), × for a failure, ▲ for a stop at a saddle."""
    pids = list(data)
    hmax = max(
        r["n_hev"]
        for pid in pids
        for lab in labels
        for r in data[pid].get(lab, [])
        if r["outcome"] == "min"
    )
    cmap = plt.get_cmap("Blues")
    hmin = min(
        r["n_hev"]
        for pid in pids
        for lab in labels
        for r in data[pid].get(lab, [])
        if r["outcome"] == "min"
    )
    # The lightest step starts above white so that every solved start stays visible.
    norm = matplotlib.colors.LogNorm(vmin=hmin / 2.0, vmax=hmax)
    fig, axes = plt.subplots(
        len(labels), len(pids), figsize=(2.0 * len(pids) + 2.6, 1.9 * len(labels)), squeeze=False
    )
    for j, pid in enumerate(pids):
        for i, lab in enumerate(labels):
            ax = axes[i][j]
            rows = data[pid].get(lab)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if rows is None:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    color=TEXT2,
                    fontsize=8,
                    transform=ax.transAxes,
                )
                continue
            xs = np.array([r["x0"] for r in rows])
            ok = np.array([r["outcome"] == "min" for r in rows])
            sad = np.array([r["outcome"] == "saddle" for r in rows])
            fl = np.array([r["outcome"] == "fail" and not r["near_min"] for r in rows])
            nm = np.array([bool(r["near_min"]) for r in rows])
            h = np.array([r["n_hev"] for r in rows], dtype=float)
            ax.scatter(
                xs[ok, 0],
                xs[ok, 1],
                c=h[ok],
                cmap=cmap,
                norm=norm,
                s=34,
                marker="s",
                edgecolors="white",
                linewidths=0.6,
            )
            ax.scatter(xs[fl, 0], xs[fl, 1], c=TEXT, s=22, marker="x", linewidths=1.0)
            ax.scatter(xs[sad, 0], xs[sad, 1], c=TEXT, s=20, marker="^", linewidths=0)
            ax.scatter(
                xs[nm, 0],
                xs[nm, 1],
                facecolors="none",
                edgecolors=TEXT,
                s=22,
                marker="o",
                linewidths=1.0,
            )
            ax.set_aspect("equal", adjustable="datalim")
            if i == 0:
                ax.set_title(pid, fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8, color=TEXT, rotation=0, ha="right", va="center")
            n_ok = int(ok.sum())
            ax.text(
                0.5,
                -0.02,
                f"{n_ok}/{len(rows)} solved",
                ha="center",
                va="top",
                fontsize=7,
                color=TEXT2,
                transform=ax.transAxes,
            )
    cax = fig.add_axes((0.86, 0.3, 0.015, 0.4))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [t for t in (2, 4, 8, 16, 32, 64, 128) if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.minorticks_off()
    cb.set_label("Hessian evaluations (start solved)", fontsize=8, color=TEXT)
    cb.ax.tick_params(labelsize=7, colors=TEXT2)
    fig.text(
        0.5,
        0.005,
        "■ reached a minimizer with ‖∇f‖∞ ≤ 1e-8 (colour = Hessian count)   "
        "× failed / diverged   ▲ stopped at a saddle point\n○ stopped at a minimizer "
        "with 1e-8 < ‖∇f‖∞ ≤ 1e-6 (line search at the rounding level of f)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT2,
    )
    fig.subplots_adjust(left=0.22, right=0.83, top=0.96, bottom=0.05, wspace=0.08, hspace=0.2)
    for ext in ("png", "svg"):
        fig.savefig(
            FIGURES / f"outcome_maps_{'convex' if 'lse' in pids else 'nonconvex'}.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_sensitivity(sens: dict[str, Any]) -> None:
    """Median Hessian count on the convex set against each method's constant (log axis)."""
    lines = {
        "ARC sigma0": ("ARC", "σ₀"),
        "RegN-AdaN H0": ("RegN-AdaN", "H₀"),
        "RegN-SU H0": ("RegN-SU", "H₀"),
        "trust_region_exact radius0": ("trust_region_exact", "Δ₀"),
    }
    defaults = {"ARC sigma0": 1.0, "RegN-AdaN H0": 1.0, "RegN-SU H0": 1.0}
    defaults["trust_region_exact radius0"] = 1.0
    pids = list(study.CONVEX)
    fig, axes = plt.subplots(1, len(pids), figsize=(3.6 * len(pids), 3.4), squeeze=False)
    for ax, pid in zip(axes[0], pids, strict=True):
        for key, (lab, sym) in lines.items():
            per = sens["by_method"][key]
            xs = np.array([float(v) for v in per])
            ys = np.array([per[v][pid]["hev_median"] for v in per], dtype=float)
            color, ls = STYLE[lab]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=ls,
                linewidth=2,
                marker="o",
                markersize=3.5,
                label=f"{lab} ({sym})",
            )
            d = defaults[key]
            ax.plot(
                [d],
                [per[f"{d:g}"][pid]["hev_median"]],
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                linestyle="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_title(pid, fontsize=10, color=TEXT)
        ax.set_xlabel("constant: σ₀ (ARC), H₀ (RegN), Δ₀ (TR)", fontsize=8, color=TEXT)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([t for t in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96)])
        _style_axes(ax)
    axes[0][0].set_ylabel("median Hessian evaluations (64 starts)", fontsize=8, color=TEXT)
    handles, labs = axes[0][0].get_legend_handles_labels()
    handles.append(
        matplotlib.lines.Line2D(
            [],
            [],
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=TEXT2,
            linestyle="none",
        )
    )
    labs.append("default constant (main tables)")
    fig.legend(
        handles,
        labs,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"sensitivity.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(profiles: dict[str, bench.Profile]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.2 * len(profiles), 3.9), squeeze=False)
    for ax, (title, prof) in zip(axes[0], profiles.items(), strict=True):
        for s, lab in enumerate(prof.labels):
            color, ls = STYLE[lab]
            ax.step(
                prof.x, prof.y[s], where="post", color=color, linestyle=ls, linewidth=2, label=lab
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(1, prof.x[-1])
        ax.set_xlabel(
            "α  (Hessian count ≤ α × best Hessian count on the instance)", fontsize=8, color=TEXT
        )
        ax.set_ylabel("fraction of instances  ρ_s(α)", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=10, color=TEXT)
        _style_axes(ax)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"performance_profiles.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    cases: list[tuple[str, Problem, tuple[float, float], dict[str, Solver], bool]],
) -> dict[str, Any]:
    """‖∇f‖₂ against cumulative Hessian evaluations from one start per problem."""
    ncol = 4
    nrow = math.ceil(len(cases) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow), squeeze=False)
    record: dict[str, Any] = {}
    for idx, (pid, p, x0, meths, has_min) in enumerate(cases):
        ax = axes[idx // ncol][idx % ncol]
        record[pid] = {"x0": list(x0)}
        runs = []
        for lab, fn in meths.items():
            r = fn(p, x0)
            h = hessians_along_trace(r)
            assert h[-1] == r.n_hev, (pid, lab, h[-1], r.n_hev)  # trace ↔ counter consistency
            gn = np.array([s.grad_norm if s.grad_norm is not None else np.nan for s in r.trace])
            ok = np.isfinite(gn) & (gn > 0)
            runs.append((lab, r, h, gn, ok, classify(p, r, has_minimizer=has_min)))
        # x-range: the runs that reached a minimizer; longer runs are clipped at the edge (▸).
        solved_h = [int(h[ok][-1]) for _, _, h, _, ok, out in runs if out == "min" and ok.any()]
        xmax = 1.3 * max(solved_h) + 1 if solved_h else 50.0
        for lab, r, h, gn, ok, outcome in runs:
            color, ls = STYLE[lab]
            hh, gg = h[ok], gn[ok]
            inside = hh <= xmax
            ax.semilogy(
                hh[inside],
                gg[inside],
                color=color,
                linestyle=ls,
                linewidth=2 if ls == "-" else 1.6,
                label=lab,
            )
            if ok.any():
                if hh[-1] > xmax:
                    y_edge = float(np.interp(xmax, hh, gg))
                    ax.plot(xmax, y_edge, marker=">", color=color, markersize=7, linestyle="none")
                    where = {"min": "", "saddle": ", at a saddle", "fail": ", failed"}[outcome]
                    ax.annotate(
                        f"{lab}: {r.n_hev} at stop{where}",
                        (xmax, y_edge),
                        xytext=(-4, 6),
                        textcoords="offset points",
                        ha="right",
                        fontsize=7,
                        color=TEXT2,
                    )
                elif outcome != "min":  # where a run ended without reaching a minimizer
                    ax.plot(
                        hh[-1],
                        max(gg[-1], 2e-13),
                        marker="^" if outcome == "saddle" else "x",
                        color=color,
                        markersize=7,
                        markeredgewidth=1.5,
                        linestyle="none",
                    )
            record[pid][lab] = {
                "n_hev": r.n_hev,
                "converged": r.converged,
                "outcome": outcome,
                "final_grad_norm": float(gg[-1]) if ok.any() else None,
            }
        ax.axhline(GTOL, color=TEXT2, linewidth=0.8, linestyle=(0, (1, 2)))
        ax.set_title(f"{pid}, x₀ = ({x0[0]:g}, {x0[1]:g})", fontsize=9, color=TEXT)
        ax.set_xlabel("Hessian evaluations", fontsize=8, color=TEXT)
        ax.set_ylabel("‖∇f(x_k)‖₂", fontsize=8, color=TEXT)
        ax.set_ylim(1e-13, 1e3)
        ax.set_xlim(0, xmax * 1.03)
        _style_axes(ax)
    for idx in range(len(cases), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    handles, labs = [], []
    for row in axes:
        for ax in row:
            for hnd, lb in zip(*ax.get_legend_handles_labels(), strict=True):
                if lb not in labs:
                    handles.append(hnd)
                    labs.append(lb)
    handles += [
        matplotlib.lines.Line2D([], [], color=TEXT2, marker="^", linestyle="none"),
        matplotlib.lines.Line2D([], [], color=TEXT2, marker="x", linestyle="none"),
    ]
    handles.append(matplotlib.lines.Line2D([], [], color=TEXT2, marker=">", linestyle="none"))
    labs += [
        "run ended at a saddle point",
        "run failed (max_iter, divergence)",
        "run continues past the axis (count at stop)",
    ]
    axes[-1][-1].legend(handles, labs, fontsize=8, frameon=False, loc="center")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"convergence.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return record


def plot_saddle_paths(p: Problem, x0: tuple[float, float], meths: dict[str, Solver]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    t = np.linspace(-3.2, 3.2, 400)
    X, Y = np.meshgrid(t, t)
    Z = X**2 - Y**2 + Y**4 / 4
    ax.contour(X, Y, Z, levels=np.linspace(-1, 12, 27), colors=GRID, linewidths=0.8)
    ax.plot([0], [0], marker="+", color=TEXT, markersize=10)
    ax.annotate("saddle (0, 0)", (0, 0), xytext=(0.15, -0.45), fontsize=8, color=TEXT2)
    for m in p.minima:
        ax.plot([m[0]], [m[1]], marker="*", color=TEXT, markersize=9)
    for lab, fn in meths.items():
        r = fn(p, x0)
        xs = np.array([s.x for s in r.trace])
        color, ls = STYLE[lab]
        end = "minimizer" if classify(p, r, has_minimizer=True) == "min" else "saddle"
        ax.plot(
            xs[:, 0],
            xs[:, 1],
            color=color,
            linestyle=ls,
            linewidth=2,
            marker="o",
            markersize=3.5,
            label=f"{lab}: ends at the {end} ({r.n_hev} Hessians)",
        )
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_title(
        f"x² − y² + y⁴/4 from x₀ = ({x0[0]:g}, {x0[1]:g}) (on the saddle's stable manifold)",
        fontsize=9,
        color=TEXT,
    )
    ax.annotate(
        "four methods move along y = 0\n(g is orthogonal to the negative-curvature direction)\n"
        "and stop at the saddle",
        (1.6, 0.0),
        xytext=(1.05, -1.2),
        fontsize=8,
        color=TEXT2,
        arrowprops={"arrowstyle": "->", "color": TEXT2, "lw": 0.8},
    )
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"saddle_paths.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # Part 1: convex problems.
    convex: dict[str, dict[str, list[dict[str, Any]]]] = {}
    lipschitz: dict[str, dict[str, float]] = {}
    distances: dict[str, Any] = {}
    for pid in study.CONVEX:
        p = study.get(pid)
        L2 = hessian_lipschitz_estimate(p)
        H_fixed = L2 / 2.0  # Assumption 1 of Mishchenko (2023): ∇²f is 2H-Lipschitz
        lipschitz[pid] = {"L2_hat": L2, "H_fixed": H_fixed}
        distances[pid] = start_distances(pid, p)
        convex[pid] = run_set(
            pid, p, convex_starts(), methods_for(H_fixed), has_min=pid != "logistic_sep"
        )

    # Part 2: nonconvex problems.
    nonconvex: dict[str, dict[str, list[dict[str, Any]]]] = {}
    nc_problems = {
        "himmelblau": problems.get("himmelblau"),
        "six_hump_camel": problems.get("six_hump_camel"),
        "quartic_saddle": study.get("quartic_saddle"),
    }
    for pid, p in nc_problems.items():
        assert isinstance(p, Problem)
        nonconvex[pid] = run_set(pid, p, nonconvex_starts(p), methods_for(None), has_min=True)

    # Summaries.
    summary: dict[str, Any] = {"convex": {}, "nonconvex": {}}
    for part, data in (("convex", convex), ("nonconvex", nonconvex)):
        for pid, per in data.items():
            summary[part][pid] = {lab: summarize(rows) for lab, rows in per.items()}

    q1 = {
        pid: {
            "RegN-AdaN vs ARC": paired(convex[pid]["RegN-AdaN"], convex[pid]["ARC"]),
            "RegN-SU vs ARC": paired(convex[pid]["RegN-SU"], convex[pid]["ARC"]),
            "RegN-fixed vs ARC": paired(convex[pid]["RegN-fixed"], convex[pid]["ARC"]),
            "RegN-AdaN vs trust_region_exact": paired(
                convex[pid]["RegN-AdaN"], convex[pid]["trust_region_exact"]
            ),
            "RegN-SU vs trust_region_exact": paired(
                convex[pid]["RegN-SU"], convex[pid]["trust_region_exact"]
            ),
            # Gradient counts: the adaptive searches pay ∇f at every rejected trial.
            "RegN-AdaN vs ARC (gradients)": paired(
                convex[pid]["RegN-AdaN"], convex[pid]["ARC"], key="n_gev"
            ),
            "RegN-SU vs ARC (gradients)": paired(
                convex[pid]["RegN-SU"], convex[pid]["ARC"], key="n_gev"
            ),
        }
        for pid in convex
    }
    q2: dict[str, Any] = {}
    for pid, per in nonconvex.items():
        starts = [tuple(r["x0"]) for r in per["ARC"]]
        saddle_d = [i for i, r in enumerate(per["damped_newton"]) if r["outcome"] == "saddle"]
        saddle_m = [i for i, r in enumerate(per["modified_newton"]) if r["outcome"] == "saddle"]
        saddle_both = sorted(set(saddle_d) & set(saddle_m))
        nonmin_d = [i for i, r in enumerate(per["damped_newton"]) if r["outcome"] != "min"]
        nonmin_m = [i for i, r in enumerate(per["modified_newton"]) if r["outcome"] != "min"]
        nonmin_both = sorted(set(nonmin_d) & set(nonmin_m))

        def outcomes(idx: list[int], lab: str, per=per) -> dict[str, int]:
            out = {"min": 0, "saddle": 0, "fail": 0}
            for i in idx:
                out[per[lab][i]["outcome"]] += 1
            return out

        compare = ("ARC", "trust_region_exact", "trust_region_steihaug", "RegN-AdaN")
        q2[pid] = {
            "n_starts": len(starts),
            "saddle_stops": {
                "damped_newton": len(saddle_d),
                "modified_newton": len(saddle_m),
                "both": len(saddle_both),
            },
            "saddle_both_starts": [list(starts[i]) for i in saddle_both],
            "on_damped_saddle_stops": {lab: outcomes(saddle_d, lab) for lab in compare},
            "on_modified_saddle_stops": {lab: outcomes(saddle_m, lab) for lab in compare},
            "on_both_saddle_stops": {lab: outcomes(saddle_both, lab) for lab in compare},
            "non_min": {
                "damped_newton": len(nonmin_d),
                "modified_newton": len(nonmin_m),
                "both": len(nonmin_both),
                "both_near_min_fail": {
                    lab: sum(bool(per[lab][i]["near_min"]) for i in nonmin_both)
                    for lab in ("damped_newton", "modified_newton")
                },
            },
            "on_both_non_min": {lab: outcomes(nonmin_both, lab) for lab in compare},
            "ARC vs trust_region_exact": paired(per["ARC"], per["trust_region_exact"]),
            "ARC vs trust_region_steihaug": paired(per["ARC"], per["trust_region_steihaug"]),
        }

    # Sensitivity to the one tuning constant of each adaptive method, and to the initial radius of
    # the trust-region baseline (convex set only). Every value of the grid is run on all 256
    # starts; "best" is the value with the smallest median Hessian count among those that solve
    # every start of the problem (ties broken by the smaller value).
    sens = sensitivity_sweep()
    # Performance profiles on the Hessian count (Dolan–Moré), via numopt.bench.
    profiles: dict[str, bench.Profile] = {}
    prof_json: dict[str, Any] = {}
    for title, data in (
        ("Convex set (4 problems × 64 starts)", convex),
        ("Nonconvex set (3 problems × 9×9 grid)", nonconvex),
    ):
        labels = list(next(iter(data.values())).keys())
        # costs[p, s]: Hessian evaluations of method s on instance p (∞ unless it reached a minimizer).
        costs = np.array(
            [
                [
                    data[pid][lab][i]["n_hev"]
                    if data[pid][lab][i]["outcome"] == "min"
                    else math.inf
                    for lab in labels
                ]
                for pid in data
                for i in range(len(data[pid][labels[0]]))
            ],
            dtype=float,
        )
        prof = bench.performance_profile_from_costs(costs, labels)
        profiles[title] = prof
        prof_json[title] = {
            "labels": list(prof.labels),
            "rho_at_1": dict(zip(prof.labels, prof.at(1.0).tolist(), strict=True)),
            "rho_at_2": dict(zip(prof.labels, prof.at(2.0).tolist(), strict=True)),
            "solved_fraction": dict(zip(prof.labels, prof.solved.tolist(), strict=True)),
            "n_instances": int(costs.shape[0]),
        }

    # Figures.
    plot_outcome_maps(convex, list(STYLE))
    plot_outcome_maps(nonconvex, [lab for lab in STYLE if lab != "RegN-fixed"])
    plot_profiles(profiles)
    plot_sensitivity(sens)
    conv_cases = [
        (
            pid,
            study.get(pid),
            (-10.5, 7.5),
            methods_for(lipschitz[pid]["H_fixed"]),
            pid != "logistic_sep",
        )
        for pid in study.CONVEX
    ] + [
        ("himmelblau", nc_problems["himmelblau"], (-1.25, 0.0), methods_for(None), True),
        ("six_hump_camel", nc_problems["six_hump_camel"], (0.0, -0.6), methods_for(None), True),
        ("quartic_saddle", nc_problems["quartic_saddle"], (3.0, 0.0), methods_for(None), True),
    ]
    conv_record = plot_convergence(conv_cases)
    plot_saddle_paths(
        nc_problems["quartic_saddle"],
        (3.0, 0.0),
        {
            k: v
            for k, v in methods_for(None).items()
            if k
            in (
                "ARC",
                "trust_region_exact",
                "damped_newton",
                "modified_newton",
                "RegN-AdaN",
                "trust_region_steihaug",
            )
        },
    )

    meta = {
        "gtol": GTOL,
        "max_iter": MAX_ITER,
        "eig_tol": EIG_TOL,
        "convex_starts": "8x8 grid on [-10.5, 10.5]^2",
        "nonconvex_starts": "9x9 grid on the plotting domain minus stationary starts",
        "method_params": {
            "ARC": "sigma0=1, eta1=0.1, eta2=0.9, gamma=2 (CGT 2011 §7)",
            "RegN-AdaN": "H0=1",
            "RegN-SU": "H0=1, alpha=1",
            "RegN-fixed": "H = L2_hat/2 (per problem)",
            "baselines": "numopt defaults except gtol=1e-8, max_iter=500",
        },
        "sensitivity_grid": list(SENS_GRID),
        "runtime_s": None,
    }
    for name, obj in (
        ("runs_convex", convex),
        ("runs_nonconvex", nonconvex),
        ("summary", summary),
        ("q1_convex_paired", q1),
        ("q2_nonconvex", q2),
        ("sensitivity", sens),
        ("lipschitz", lipschitz),
        ("profiles", prof_json),
        ("convergence_examples", conv_record),
        ("start_distances", distances),
        (
            "converged_flag",
            {"convex": converged_flag_audit(convex), "nonconvex": converged_flag_audit(nonconvex)},
        ),
    ):
        (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, allow_nan=False))
    meta["runtime_s"] = round(time.time() - t0, 1)
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=1))

    # Console report.
    print(f"runtime {meta['runtime_s']} s")
    for part in ("convex", "nonconvex"):
        for pid, per in summary[part].items():
            for lab, s in per.items():
                print(
                    f"{part:9s} {pid:15s} {lab:22s} min {s['min']:3d}/{s['n']:3d} saddle "
                    f"{s['saddle']:3d} fail {s['fail']:3d} (near-min {s['fail_near_min']:2d})"
                    f"  H median {s['hev_median']} [{s['hev_q25']}, {s['hev_q75']}] max "
                    f"{s['hev_max']}  f {s['fev_median']} g {s['gev_median']}"
                )
    print(json.dumps(q1, indent=1))
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "saddle_both_starts"}
                for k, v in q2.items()
            },
            indent=1,
        )
    )
    print(json.dumps(sens, indent=1))
    print(json.dumps(lipschitz, indent=1))
    print(json.dumps(prof_json, indent=1))


if __name__ == "__main__":
    main()
