"""Experiment: does the ranking of numopt's derivative-free solvers depend on τ or on the profile?

Run:  .venv/bin/python research/benchmark-profiles/run.py      (about 45 s, no network)

Writes results/costs.json (the solve-cost tables t_{p,s}), results/summary.json (every number
quoted in README.md) and figures/*.svg.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import method as M
import numpy as np
from matplotlib.figure import Figure

from numopt import bench, problems
from numopt.core.rng import Rng

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# ---- Setup (every choice is stated in README.md §Setup) -------------------------------
# Tolerances are tightened and max_iter raised so a solver stops at rounding level or at the
# budget, not at a default tolerance looser than τ = 1e-7 needs. bfgs has no analytic
# gradient: numopt falls back to central differences, and every f call is charged.
SOLVERS: dict[str, dict[str, Any]] = {
    "nelder_mead": {"xtol": 1e-14, "ftol": 1e-16, "max_iter": 100_000},
    "powell": {"xtol": 1e-14, "ftol": 1e-16, "max_iter": 100_000},
    "compass_search": {"xtol": 1e-14, "max_iter": 100_000},
    "hooke_jeeves": {"xtol": 1e-14, "max_iter": 100_000},
    "bfgs": {"gtol": 1e-14, "max_iter": 100_000},
    "cma_es": {"xtol": 1e-14, "ftol": 1e-16, "max_iter": 100_000},
}
BUDGET = 1500  # f evaluations per run = 500 simplex gradients for n = 2
N_RANDOM_STARTS = 7  # plus the library default x0 → 8 start points per problem
START_SEED = 20261005
SEEDS = (0, 1, 2, 3, 4)  # CMA-ES seeds; deterministic solvers are run once and shared
TAUS = (1e-1, 1e-3, 1e-5, 1e-7)
ALPHA_MAX = float(M.PARAMS["alpha_max"].default)  # 32
KAPPA_MAX = BUDGET / 3  # 500 simplex gradients: the whole budget
KAPPA_READOUTS = (10.0, 100.0)  # fixed budgets, in simplex gradients
# NOTE: 20 000 replicates, not the PARAMS default of 2000. The Monte Carlo standard error of a
# support near 0.95 is then √(0.05·0.95/20 000) ≈ 0.0015 (0.005 at 2000), and a bootstrap
# p-value resolves 5e-5, below the smallest Holm threshold 0.05/120 ≈ 4.2e-4 of any family.
N_BOOT = 20_000
BOOT_SEED = 1
CONFIRM = 0.95  # bootstrap support needed to call an order "resolved" in one setting
ALPHA_FWER = 0.05  # family-wise error rate of the Holm-corrected flip tests
# A problem tagged "bounded-domain" has its stated minimum on a box only and is unbounded below
# on ℝⁿ (mccormick: f → −∞ along x − y = 1). An unconstrained solver can leave the box and
# diverge, which makes f_L meaningless, so such a problem is not an unconstrained test case.
EXCLUDE_TAGS = ("bounded-domain",)
AUDIT_RTOL = 1e-10  # M.cutoff_audit: largest allowed (f_known − f_found) / (f(x₀) − f_known)

# Okabe-style categorical slots 1–6 (validated: CVD ΔE ≥ 9.1, normal-vision ΔE ≥ 19.6);
# line styles and markers add a second, colour-free encoding.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P")
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def make_cases() -> list[bench.BenchmarkCase]:
    """Every 2-D unconstrained library problem with a minimum on ℝ², without derivatives,
    from 8 start points (problems tagged with an ``EXCLUDE_TAGS`` entry are left out).

    Draw order: every 2-D problem in library order, the excluded ones included; for each, 7
    points; for each point, x then y, uniform on the problem's plotting domain, from one
    Rng(START_SEED). An excluded problem still consumes its draws, so the start points of the
    other problems do not depend on which problems are excluded.
    """
    rng = Rng(START_SEED)
    cases = []
    for p in problems.list_problems("unconstrained"):
        if p.dim != 2:
            continue
        starts: list[Any] = [None]
        for _ in range(N_RANDOM_STARTS):
            starts.append([rng.uniform(*p.domain[0]), rng.uniform(*p.domain[1])])
        if set(p.tags) & set(EXCLUDE_TAGS):
            continue
        cases.append(bench.BenchmarkCase(dataclasses.replace(p, grad=None, hess=None), starts))
    return cases


def setting_tables(res: bench.BenchmarkResult) -> dict[str, dict[str, Any]]:
    """Per (profile, τ): per-instance area terms, scores, readouts and dominance."""
    n = np.array([inst.n for inst in res.instances])
    out: dict[str, dict[str, Any]] = {}
    for tau in TAUS:
        T = res.solve_costs(tau)
        R = M.performance_ratios(T)
        K = M.bendfo_data_values(T, n)
        perf = bench.performance_profile(res, tau)
        data = bench.data_profile(res, tau)
        out[f"perf@{tau:.0e}"] = {
            "kind": "performance",
            "tau": tau,
            "terms": M.area_terms(R, ALPHA_MAX),
            "score": M.performance_area(R, ALPHA_MAX),
            "profile": perf,
            "readout": {
                "rho(1)": perf.at(1.0),
                f"rho({ALPHA_MAX:g})": perf.at(ALPHA_MAX),
                "solved": perf.solved,
            },
            "dominance": M.dominance(perf.y),
        }
        out[f"data@{tau:.0e}"] = {
            "kind": "data",
            "tau": tau,
            "terms": M.area_terms(K, KAPPA_MAX),
            "score": M.data_area(K, KAPPA_MAX),
            "profile": data,
            "readout": {
                "d(10)": data.at(10.0),
                "d(100)": data.at(100.0),
                "d(500)": data.at(KAPPA_MAX),
            },
            "dominance": M.dominance(data.y),
        }
        # Fixed-budget readouts of the data profile (Moré–Wild 2009, §2: "solver S2
        # outperforms S1 with a computational budget of k simplex gradients, k ∈ [20, 100]").
        for kappa in KAPPA_READOUTS:
            hit = (K <= kappa).astype(np.float64)  # (P, S): solved within κ simplex gradients
            score = hit.mean(axis=0)
            out[f"d{kappa:g}@{tau:.0e}"] = {
                "kind": f"data at kappa={kappa:g}",
                "tau": tau,
                "terms": hit,
                "score": score,
                "profile": data,
                "readout": {},
                "dominance": M.dominance(score[:, None]),  # a single point: just the order
            }
    return out


def analyse(res: bench.BenchmarkResult) -> dict[str, Any]:
    labels = list(res.labels)
    S = len(labels)
    groups = [inst.problem_id for inst in res.instances]
    tabs = setting_tables(res)
    names = list(tabs)
    boot = {
        k: M.bootstrap_order_support(v["terms"], groups, n_boot=N_BOOT, seed=BOOT_SEED)
        for k, v in tabs.items()
    }  # same seed → the same replicates in every setting

    settings: dict[str, Any] = {}
    for k, v in tabs.items():
        settings[k] = {
            "kind": v["kind"],
            "tau": v["tau"],
            "score": v["score"].tolist(),
            "rank": M.rank_scores(v["score"]).tolist(),
            "ci95": boot[k].ci.T.tolist(),
            "p_greater": boot[k].p_greater.tolist(),
            "readout": {r: np.asarray(x).tolist() for r, x in v["readout"].items()},
            "dominance": v["dominance"].tolist(),
            "order": [labels[i] for i in np.argsort(-v["score"], kind="stable")],
        }

    kendall = {
        a: {b: M.kendall_tau_b(tabs[a]["score"], tabs[b]["score"]) for b in names} for a in names
    }

    groups_list = [str(g) for g in groups]

    # Rank changes between two settings, pair by pair. Every pair is a test (p = 1 when the
    # point estimates do not flip), so the Holm family counts all 15 pairs per comparison.
    def pair_tests(a: str, b: str) -> list[dict[str, Any]]:
        out = []
        sa, sb = tabs[a]["score"], tabs[b]["score"]
        for i, j in combinations(range(S), 2):
            da, db = float(sa[i] - sa[j]), float(sb[i] - sb[j])
            flip = da * db < 0
            supp_a = M.order_support(boot[a], i, j, da)
            supp_b = M.order_support(boot[b], i, j, db)
            # Intersection-union test (Berger 1982): the flip needs both orders.
            p_iut = 1.0 - min(supp_a, supp_b) if flip else 1.0
            # Joint bootstrap: fraction of replicates whose order also flips.
            ra = boot[a].scores[:, i] - boot[a].scores[:, j]
            rb = boot[b].scores[:, i] - boot[b].scores[:, j]
            Da, Db = tabs[a]["dominance"][i, j], tabs[b]["dominance"][i, j]
            out.append(
                {
                    "pair": [labels[i], labels[j]],
                    "ij": [i, j],
                    "flip": bool(flip),
                    "score_a": [float(sa[i]), float(sa[j])],
                    "score_b": [float(sb[i]), float(sb[j])],
                    "support_a": supp_a,
                    "support_b": supp_b,
                    "mc_se_a": math.sqrt(supp_a * (1 - supp_a) / N_BOOT),
                    "mc_se_b": math.sqrt(supp_b * (1 - supp_b) / N_BOOT),
                    "p_iut": p_iut,
                    "p_flip_joint": float(np.mean(np.sign(ra) * np.sign(rb) < 0)),
                    "resolved_both": bool(flip and min(supp_a, supp_b) >= CONFIRM),
                    "resolved_one": bool(flip and (supp_a >= CONFIRM) != (supp_b >= CONFIRM)),
                    "dominance_a": int(Da),
                    "dominance_b": int(Db),
                    "dominance_reversal": bool({int(Da), int(Db)} == {1, -1}),
                }
            )
        return out

    lo, hi = f"{TAUS[0]:.0e}", f"{TAUS[-1]:.0e}"
    families: dict[str, list[tuple[str, str]]] = {
        "Q1 tolerance": [(f"perf@{lo}", f"perf@{hi}"), (f"data@{lo}", f"data@{hi}")],
        "Q2 whole-budget areas": [(f"perf@{t:.0e}", f"data@{t:.0e}") for t in TAUS],
        "Q2 fixed-budget readouts": [
            (f"perf@{t:.0e}", f"d{k:g}@{t:.0e}") for k in KAPPA_READOUTS for t in TAUS
        ],
    }
    comparisons: dict[str, list[dict[str, Any]]] = {}
    family_size: dict[str, int] = {}
    for fam, pairs in families.items():
        tests = [(a, b, t) for a, b in pairs for t in pair_tests(a, b)]
        p_adj = M.holm([t["p_iut"] for _, _, t in tests])
        family_size[fam] = len(tests)
        for (a, b, t), q in zip(tests, p_adj, strict=True):
            t["family"] = fam
            t["p_holm"] = float(q)
            t["confirmed_holm"] = bool(t["flip"] and q <= ALPHA_FWER)
            if t["resolved_both"] or t["confirmed_holm"]:
                i, j = t["ij"]
                logo = M.leave_one_group_out(
                    tabs[a]["terms"],
                    tabs[b]["terms"],
                    groups_list,
                    i,
                    j,
                    n_boot=N_BOOT,
                    seed=BOOT_SEED,
                    confirm=CONFIRM,
                )
                t["jackknife"] = {
                    "kept": sum(v["kept"] for v in logo.values()),
                    "of": len(logo),
                    "lost_without": [g for g, v in logo.items() if not v["kept"]],
                    "min_support": min(min(v["support_a"], v["support_b"]) for v in logo.values()),
                }
            if t["flip"]:
                comparisons.setdefault(f"{a} vs {b}", []).append(t)
            else:
                comparisons.setdefault(f"{a} vs {b}", [])

    return {
        "labels": labels,
        "settings": settings,
        "kendall_tau_b": kendall,
        "comparisons": comparisons,
        "family_size": family_size,
        "boot": boot,
        "tabs": tabs,
    }


# ---- Figures ---------------------------------------------------------------------------


def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _step(ax: Any, x: np.ndarray, y: np.ndarray, j: int, label: str) -> None:
    ax.step(x, y, where="post", color=COLORS[j], linestyle=STYLES[j], linewidth=1.6, label=label)


def fig_profiles(an: dict[str, Any], labels: list[str], path: Path) -> None:
    fig = Figure(figsize=(12.0, 5.6), layout="constrained")
    axes = fig.subplots(2, len(TAUS), sharey=True)
    for c, tau in enumerate(TAUS):
        for r, kind in enumerate(("perf", "data")):
            ax = axes[r, c]
            prof = an["tabs"][f"{kind}@{tau:.0e}"]["profile"]
            for j, lab in enumerate(labels):
                _step(ax, prof.x, prof.y[j], j, lab)
            ax.set_xscale("log", base=2)
            if kind == "perf":
                ax.set_xlim(1, 2**10)
                ax.set_title(rf"$\tau = 10^{{{round(math.log10(tau))}}}$", color=INK, fontsize=10)
                ax.set_xlabel(r"performance ratio $\alpha$", color=INK, fontsize=9)
            else:
                ax.set_xlim(0.25, KAPPA_MAX)
                ax.set_xlabel(r"budget $\kappa$ (simplex gradients)", color=INK, fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            _style_axes(ax)
        axes[0, 0].set_ylabel(r"performance profile $\rho_s(\alpha)$", color=INK, fontsize=9)
        axes[1, 0].set_ylabel(r"data profile $d_s(\kappa)$", color=INK, fontsize=9)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labs,
        loc="outside upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.savefig(path)


def fig_ranks(
    an: dict[str, Any], labels: list[str], ecdf_scores: np.ndarray, ecdf_ci: np.ndarray, path: Path
) -> None:
    names = [f"perf@{t:.0e}" for t in TAUS] + [f"data@{t:.0e}" for t in TAUS]
    names += [f"d10@{t:.0e}" for t in TAUS]
    ticks = [rf"$10^{{{round(math.log10(t))}}}$" for t in TAUS] * 3 + ["ECDF"]
    xs = np.concatenate([np.arange(4) + 4.6 * g for g in range(3)] + [[13.8]])
    fig = Figure(figsize=(13.0, 4.8), layout="constrained")
    ax_s, ax_r = fig.subplots(1, 2, width_ratios=(1.25, 1.0))
    S = len(labels)
    for j, lab in enumerate(labels):
        sc = [an["settings"][k]["score"][j] for k in names] + [float(ecdf_scores[j])]
        ci = [an["settings"][k]["ci95"][j] for k in names] + [list(ecdf_ci[:, j])]
        rk = [an["settings"][k]["rank"][j] for k in names]
        rk.append(float(M.rank_scores(ecdf_scores)[j]))
        off = (j - (S - 1) / 2) * 0.07
        lo = [s - c[0] for s, c in zip(sc, ci, strict=True)]
        hi = [c[1] - s for s, c in zip(sc, ci, strict=True)]
        for seg in (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 13)):
            ax_s.errorbar(
                xs[seg] + off,
                sc[seg],
                yerr=[lo[seg], hi[seg]],
                color=COLORS[j],
                linestyle=STYLES[j],
                marker=MARKERS[j],
                markersize=5,
                linewidth=1.4,
                elinewidth=0.9,
                capsize=0,
                label=lab if seg.start == 0 else None,
            )
            ax_r.plot(
                xs[seg],
                rk[seg],
                color=COLORS[j],
                linestyle=STYLES[j],
                marker=MARKERS[j],
                markersize=6,
                linewidth=1.6,
            )
        ax_r.annotate(lab, (xs[-1] + 0.25, rk[-1]), va="center", fontsize=8, color=INK)
    for ax in (ax_s, ax_r):
        ax.set_xticks(xs, ticks)
        _style_axes(ax)
        for x0, txt in (
            (1.5, r"area of $\rho_s$, by $\tau$"),
            (6.1, r"area of $d_s$, by $\tau$"),
            (10.7, r"$d_s(\kappa = 10)$, by $\tau$"),
            (13.8, "COCO"),
        ):
            ax.text(
                x0,
                -0.13,
                txt,
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=8,
                color=MUTED,
            )
    ax_s.set_ylabel("score (95 % cluster-bootstrap CI)", color=INK, fontsize=9)
    ax_s.set_title("a  Summary score per setting", loc="left", color=INK, fontsize=10)
    ax_s.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2, loc="lower left")
    ax_r.set_ylim(S + 0.5, 0.5)
    ax_r.set_yticks(range(1, S + 1))
    ax_r.set_xlim(-0.4, 16.2)
    ax_r.set_ylabel("rank (1 = best)", color=INK, fontsize=9)
    ax_r.set_title("b  Rank per setting", loc="left", color=INK, fontsize=10)
    fig.savefig(path)


def fig_ecdf(ecdf: M.Ecdf, labels: list[str], path: Path) -> None:
    fig = Figure(figsize=(6.4, 4.2), layout="constrained")
    ax = fig.subplots()
    for j, lab in enumerate(labels):
        _step(ax, ecdf.x, ecdf.y[j], j, lab)
    ax.set_xscale("log", base=10)
    ax.set_xlim(0.5, BUDGET / 2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("function evaluations / dimension", color=INK, fontsize=9)
    ax.set_ylabel("fraction of (run, target) pairs", color=INK, fontsize=9)
    ax.set_title(
        r"Runtime ECDF, 51 targets $\Delta f \in [10^{-8}, 10^{2}]$", color=INK, fontsize=10
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.savefig(path)


def fig_convergence(res: bench.BenchmarkResult, labels: list[str], path: Path) -> None:
    show = ("rosenbrock", "beale", "goldstein_price", "six_hump_camel")
    grid = np.unique(np.round(np.geomspace(1, BUDGET, 240)))  # log-spaced evaluation counts
    fig = Figure(figsize=(10.0, 6.4), layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    for ax, pid in zip(axes, show, strict=True):
        insts = [i for i in res.instances if i.problem_id == pid]
        for j, lab in enumerate(labels):
            gaps = []
            for inst in insts:
                h = res.runs[(lab, inst.id)]
                idx = np.searchsorted(h.cost, grid, side="right") - 1
                best = np.where(idx >= 0, h.best[np.maximum(idx, 0)], inst.f0)
                f_ref = inst.f_known if inst.f_known is not None else inst.f_star
                gaps.append(np.maximum(best - f_ref, 1e-16))
            G = np.array(gaps)
            med = np.median(G, axis=0)
            q1, q3 = np.quantile(G, [0.25, 0.75], axis=0)
            ax.fill_between(grid, q1, q3, color=COLORS[j], alpha=0.10, linewidth=0)
            ax.plot(grid, med, color=COLORS[j], linestyle=STYLES[j], linewidth=1.6, label=lab)
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_ylim(1e-16, None)
        ax.set_title(
            f"{pid}  ({len(insts) // len(SEEDS)} starts × {len(SEEDS)} seeds)",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("function evaluations", color=INK, fontsize=9)
        ax.set_ylabel(r"best $f - f^*$ (median, IQR)", color=INK, fontsize=9)
        _style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2, loc="lower left")
    fig.savefig(path)


# ---- Main ------------------------------------------------------------------------------


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf" if v < 0 else None)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    cases = make_cases()
    raw = bench.run_benchmark(
        list(SOLVERS), cases, budget=BUDGET, cost="nfev", params=SOLVERS, seeds=SEEDS
    )
    t_run = time.perf_counter() - t0
    labels = list(raw.labels)

    # f_L must not come from runs that escape to −∞ (README §Setup); stop before any analysis.
    audit = M.cutoff_audit(raw, rtol=AUDIT_RTOL)
    if audit.flagged:
        raise RuntimeError(
            f"{len(audit.flagged)} instances have a best value far below f_known "
            f"(first: {audit.flagged[:5]}); the problem set has an unbounded problem"
        )
    res = M.with_bendfo_cutoff(raw)  # primary convention
    an = analyse(res)
    an_known = analyse(raw)  # sensitivity: f_L = known global minimum (numopt.bench default)

    deltas = M.coco_targets()
    ecdf = M.runtime_ecdf(raw, deltas)
    ecdf_x_max = BUDGET / 2
    ecdf_scores = M.ecdf_area(ecdf, ecdf_x_max)
    flat = ecdf.runtimes.reshape(len(labels), len(raw.instances), -1)
    # Per-instance ECDF terms: mean over the 51 targets of each run's clipped log runtime.
    per_inst = np.stack(
        [M.area_terms(flat[s], ecdf_x_max).mean(axis=1) for s in range(len(labels))], axis=1
    )
    ecdf_boot = M.bootstrap_order_support(
        per_inst, [i.problem_id for i in raw.instances], n_boot=N_BOOT, seed=BOOT_SEED
    )
    ert = {f"{d:.0e}": M.expected_running_time(raw, d).tolist() for d in (1e-1, 1e-4, 1e-8)}

    stops: dict[str, dict[str, int]] = {}
    for lab in labels:
        hs = [raw.runs[(lab, i.id)] for i in raw.instances]
        stops[lab] = {k: sum(h.stopped == k for h in hs) for k in ("solver", "budget", "error")}

    n_unique = len({(i.problem_id, i.x0) for i in raw.instances})
    summary = {
        "setup": {
            "solvers": SOLVERS,
            "budget": BUDGET,
            "cost_model": "nfev",
            "problems": [c.resolve().id for c in cases],
            "excluded_tags": list(EXCLUDE_TAGS),
            "excluded_problems": [
                p.id
                for p in problems.list_problems("unconstrained")
                if p.dim == 2 and set(p.tags) & set(EXCLUDE_TAGS)
            ],
            "n_problems": len(cases),
            "starts_per_problem": 1 + N_RANDOM_STARTS,
            "start_seed": START_SEED,
            "seeds": list(SEEDS),
            "n_instances": len(raw.instances),
            "n_problem_start_pairs": n_unique,
            "taus": list(TAUS),
            "alpha_max": ALPHA_MAX,
            "kappa_max": KAPPA_MAX,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "confirm": CONFIRM,
            "alpha_fwer": ALPHA_FWER,
            "holm_family_size": an["family_size"],
            "coco_targets": {
                "n": int(deltas.size),
                "max": float(deltas[0]),
                "min": float(deltas[-1]),
            },
        },
        "labels": labels,
        "stops": stops,
        "cutoff_audit": {
            "rtol": AUDIT_RTOL,
            "n_flagged": len(audit.flagged),
            "max_gap": float(np.nanmax(audit.gap)),
            "n_below_f_known": int(np.sum(audit.gap > 0)),
        },
        "bendfo": {k: an[k] for k in ("settings", "kendall_tau_b", "comparisons")},
        "known_fmin": {k: an_known[k] for k in ("settings", "kendall_tau_b", "comparisons")},
        "coco": {
            "ecdf_area": ecdf_scores.tolist(),
            "ecdf_x_max": ecdf_x_max,
            "ecdf_rank": M.rank_scores(ecdf_scores).tolist(),
            "ecdf_area_bootstrap_mean_ci95": ecdf_boot.ci.T.tolist(),
            "ecdf_p_greater": ecdf_boot.p_greater.tolist(),
            "ecdf_final_fraction": ecdf.y[:, -1].tolist(),
            "ert": ert,
        },
        "kendall_ecdf_vs": {
            k: M.kendall_tau_b(ecdf_scores, np.array(v["score"])) for k, v in an["settings"].items()
        },
        "runtime_s": {"benchmark": round(t_run, 1)},
    }

    fig_profiles(an, labels, FIGURES / "profiles.svg")
    fig_ranks(an, labels, ecdf_scores, ecdf_boot.ci, FIGURES / "ranks.svg")
    fig_ecdf(ecdf, labels, FIGURES / "ecdf.svg")
    fig_convergence(res, labels, FIGURES / "convergence.svg")
    # The solve-cost tables t_{p,s} reproduce every profile number (the full run histories
    # are 10 MB; rerun this script to get them).
    costs = {
        "format": "t[p][s] = evaluations until eq. 2.2 holds; 'inf' = not solved",
        "labels": labels,
        "instances": [inst.to_dict() for inst in res.instances],
        "f_L_numopt_bench_default": [inst.f_star for inst in raw.instances],
        "bendfo": {f"{t:.0e}": res.solve_costs(t).tolist() for t in TAUS},
        "known_fmin": {f"{t:.0e}": raw.solve_costs(t).tolist() for t in TAUS},
    }
    (RESULTS / "costs.json").write_text(
        json.dumps(_jsonable(costs), allow_nan=False), encoding="utf-8"
    )
    summary["runtime_s"]["total"] = round(time.perf_counter() - t0, 1)
    (RESULTS / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=1, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(_jsonable({k: summary[k] for k in ("stops", "runtime_s")}), indent=1))
    for name, a in (("bendfo", an), ("known", an_known)):
        print(f"--- f_L = {name}")
        for k, v in a["settings"].items():
            print(f"{k:12s}", " ".join(f"{s:.3f}" for s in v["score"]), v["order"])
        for k, v in a["comparisons"].items():
            for f in v:
                print(
                    f"{k:26s} {f['pair']} sa={f['support_a']:.4f} sb={f['support_b']:.4f}"
                    f" holm={f['p_holm']:.4f} both={f['resolved_both']}"
                    f" one={f['resolved_one']} jk={f.get('jackknife', {}).get('kept')}"
                )
    print("ECDF", ecdf_scores.round(3), M.rank_scores(ecdf_scores))


if __name__ == "__main__":
    main()
