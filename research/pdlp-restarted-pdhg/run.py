"""Experiment: matrix-vector products PDHG needs to reach a relative KKT error of 1e-8,
without restarts, with fixed-frequency restarts and with adaptive restarts.

Every comparison is made twice, in two configurations:

* ``unit`` — ω = 1, no preconditioning (the 2023 paper's setting);
* ``pc``   — balanced ω and Ruiz + Pock–Chambolle preconditioning (the method's defaults).

Each configuration has its own plain-PDHG baseline, its own fixed-period sweep and its own
adaptive scheme, so that the restart gain and the adaptive/fixed ratio are measured inside one
configuration and never across a change of preconditioning.

The hindsight best fixed period is reported on three nested grids (powers of 4, powers of 2,
half octaves) so that the effect of the grid on the adaptive/fixed ratio is visible.

Reproduce:  .venv/bin/python research/pdlp-restarted-pdhg/run.py      (≈ 2 min on 16 cores)

Writes results/summary.json, results/curves.json and figures/*.svg|png.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from numopt import bench, problems
from numopt.core.registry import get_method
from numopt.core.types import LinearProgram, Result, to_jsonable

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"

_spec = importlib.util.spec_from_file_location("pdlp_restarted_pdhg_method", HERE / "method.py")
assert _spec is not None and _spec.loader is not None
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

TOL = 1e-8
MAX_ITER = 50_000  # budget: 2 + 2·50 000 = 100 002 products with A or Aᵀ for every variant
BUDGET = 2 + 2 * MAX_ITER
# Fixed restart periods: half octaves 2^(j/2), j = 4 … 28, i.e. 4 … 16384 (25 periods).
# The 2023 paper sweeps 4¹ … 4⁹; our budget is 5·10⁴ iterations, so 16384 = 4⁷ is the
# largest period that still restarts more than twice.
PERIODS = tuple(sorted({round(2.0 ** (j / 2)) for j in range(4, 29)}))
GRIDS: dict[str, tuple[int, ...]] = {
    "pow4": tuple(4**j for j in range(1, 8)),  # 4 … 16384 (7 periods)
    "pow2": tuple(2**j for j in range(2, 15)),  # 4 … 16384 (13 periods)
    "half_octave": PERIODS,  # 4 … 16384 (25 periods)
}
assert all(set(g) <= set(PERIODS) for g in GRIDS.values())
LIBRARY = (
    "wyndor",
    "diet_2d",
    "degenerate_2d",
    "beale_cycling",
    "klee_minty_3",
    "transport_small",
    "ilp_knapsack_like_2d",
    "ilp_3var",
)
RANDOM_SIZES = ((5, 12), (10, 25), (20, 50), (40, 100))
RANDOM_SEEDS = (1, 2)
CASE_SEED = 761  # Hypothesis counterexample of test_random_sharp_lps_solved_to_known_optimum
START_SEEDS = (11, 12)  # two random starts per library LP, besides the zero start

UNIT = {"primal_weight": "unit", "precondition": "none"}
PC = {"primal_weight": "balanced", "precondition": "ruiz_pc"}  # = the method's defaults
# The two configurations: suffix of the labels, settings, plain label, adaptive labels.
CONFIGS: dict[str, dict[str, Any]] = {
    "unit": {"suffix": "", "kw": UNIT, "plain": "plain", "adaptive": ("adaptive/40", "adaptive/1")},
    "pc": {"suffix": "+pc", "kw": PC, "plain": "plain+pc", "adaptive": ("default",)},
}
# Runs with tol = TOL (the two plain runs are separate: they go to the full budget).
VARIANTS: dict[str, dict[str, Any]] = {
    **{
        f"fixed-{p}{c['suffix']}": {"restart": "fixed", "restart_period": p, **c["kw"]}
        for c in CONFIGS.values()
        for p in PERIODS
    },
    "adaptive/1": {"restart": "adaptive", "restart_check_every": 1, **UNIT},
    "adaptive/40": {"restart": "adaptive", "restart_check_every": 40, **UNIT},
    "default": {},  # adaptive/40 + ruiz_pc + balanced ω (method defaults)
    "adaptive/40+pc": {
        "restart": "adaptive",
        "restart_check_every": 40,
        **UNIT,
        "precondition": "ruiz_pc",
    },  # ablation: unit ω with preconditioning
    "pdlp-ω/40": {"primal_weight": "adaptive"},  # default + PDLP Algorithm 3 at every restart
    "pdlp-ω/1": {"primal_weight": "adaptive", "restart_check_every": 1},
}
PLAIN_RUNS = {"plain": UNIT, "plain+pc": PC}
LABELS = [
    "plain/last",
    "plain/avg",
    "plain+pc/last",
    "plain+pc/avg",
    *VARIANTS,
]
BASELINES = ("primal_dual_ipm", "affine_scaling", "two_phase_simplex", "revised_simplex")
# Colours by role (Okabe–Ito); the same role has the same colour in both configurations.
ROLE_COLORS = {
    "plain/last": "#7f7f7f",
    "plain/avg": "#000000",
    "best fixed": "#0072B2",
    "adaptive/1": "#E69F00",
    "adaptive": "#D55E00",
    "pdlp-ω/40": "#CC79A7",
    "adaptive/40+pc": "#8C564B",
    "single fixed": "#56B4E9",
    "oracle": "#009E73",
}
CURVE_INSTS = ("wyndor@0", "diet_2d@0", "beale_cycling@0", "rand_20x50_s1@0")


# --------------------------------------------------------------------------------------
# Problem set
# --------------------------------------------------------------------------------------


def random_lp(seed: int, m: int, n: int) -> LinearProgram:
    """Equality LP with a known strictly complementary optimum (z*, y*): b = A z*, c = Aᵀy* + s*."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    z = np.zeros(n)
    basis = rng.choice(n, size=m, replace=False)
    z[basis] = rng.uniform(0.5, 2.0, size=m)
    y = rng.normal(size=m)
    s = rng.uniform(0.5, 2.0, size=n)
    s[basis] = 0.0
    return LinearProgram(
        id=f"rand_{m}x{n}_s{seed}",
        name=f"random {m}×{n}",
        c=A.T @ y + s,
        A_eq=A,
        b_eq=A @ z,
        optimum=z,
        optimal_value=float((A.T @ y + s) @ z),
    )


def oracle(lp: LinearProgram) -> tuple[np.ndarray, float]:
    sign = 1.0 if lp.sense == "min" else -1.0
    res = linprog(
        sign * np.asarray(lp.c),
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=(0, None),
        method="highs",
    )
    assert res.status == 0, res.message
    return np.asarray(res.x), float(sign * res.fun)


def instances() -> list[tuple[str, LinearProgram, np.ndarray | None]]:
    """(label, LP, x0): library LPs × {zero, 2 random starts} + random LPs × {zero}."""
    out: list[tuple[str, LinearProgram, np.ndarray | None]] = []
    for pid in LIBRARY:
        lp = problems.get(pid)
        x_star, _ = oracle(lp)
        scale = 2.0 * max(1.0, float(np.max(np.abs(x_star))))
        out.append((f"{pid}@0", lp, None))
        for s in START_SEEDS:
            x0 = np.random.default_rng(s).uniform(0.0, scale, size=np.size(lp.c))
            out.append((f"{pid}@s{s}", lp, x0))
    for m, n in RANDOM_SIZES:
        for seed in RANDOM_SEEDS:
            lp = random_lp(seed, m, n)
            out.append((f"{lp.id}@0", lp, None))
    return out


def lp_of(name: str) -> str:
    """The LP of an instance label ('wyndor@s11' → 'wyndor')."""
    return name.split("@")[0]


def siblings(names: list[str], i: int) -> list[int]:
    """Tuning set of instance i: the other starts of the same LP (library LPs), or the random
    LP of the same size with the other seed (random LPs). Never contains i."""
    n = names[i]
    if n.startswith("rand_"):
        size = n.rsplit("_s", 1)[0]
        return [j for j, m in enumerate(names) if j != i and m.rsplit("_s", 1)[0] == size]
    return [j for j, m in enumerate(names) if j != i and lp_of(m) == lp_of(n)]


# --------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------


def value(res: Result) -> float:
    """The objective value of a Result (every method here reports one)."""
    assert res.fun is not None, res.message
    return float(res.fun)


def first_hit(values: list[float], tol: float = TOL) -> float:
    """Products with A/Aᵀ (2 + 2k) at the first k with values[k] ≤ tol; ∞ if never."""
    for k, v in enumerate(values):
        if v <= tol:
            return float(2 + 2 * k)
    return math.inf


def shape_fit(values: list[float]) -> dict[str, Any] | None:
    """Linear vs power-law fit of the running-minimum envelope e_k on the segment e ≤ 1e-3.

    Linear convergence: log10 e ≈ a − r·k (straight on a log-y plot). Sublinear O(k^-p):
    log10 e ≈ a − p·log10 k. Returns both R² and the fitted rate/exponent, with k in products.
    """
    env = np.minimum.accumulate(np.asarray(values, dtype=np.float64))
    k = 2.0 + 2.0 * np.arange(env.size)
    seg = (env <= 1e-3) & (env >= TOL * 0.5) & (env > 0)
    if seg.sum() < 50:
        return None
    x, y = k[seg], np.log10(env[seg])

    def r2(u: np.ndarray) -> tuple[float, float]:
        A = np.stack([np.ones_like(u), u], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        return 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()), float(coef[1])

    r2_lin, slope_lin = r2(x)
    r2_pow, slope_pow = r2(np.log10(x))
    return {
        "segment_products": [float(x[0]), float(x[-1])],
        "r2_linear": r2_lin,
        "decades_per_1000_products": -1000.0 * slope_lin,
        "r2_power": r2_pow,
        "power_exponent": -slope_pow,
        "better_fit": "linear" if r2_lin >= r2_pow else "power",
        "reached_tol": bool(env[-1] <= TOL),
        "final_envelope": float(env[-1]),
    }


def downsample(values: list[float], n: int = 600) -> dict[str, list[float]]:
    idx = np.unique(np.linspace(0, len(values) - 1, min(n, len(values))).astype(int))
    return {"products": [2.0 + 2.0 * i for i in idx], "kkt": [float(values[i]) for i in idx]}


def run_pdhg(lp: LinearProgram, x0: np.ndarray | None, **kw: Any) -> Result:
    return M.restarted_pdhg(lp, x0=x0, max_iter=MAX_ITER, **kw)


def kkts(res: Result) -> list[float]:
    return [s.info["kkt"] for s in res.trace]


def one_run(task: tuple[int, str]) -> dict[str, Any]:
    """One (instance, run) pair in a worker process; deterministic. Returns small summaries."""
    p, run = task
    name, lp, x0 = instances()[p]
    _, f_ref = oracle(lp)
    keep_curve = name in CURVE_INSTS
    out: dict[str, Any] = {
        "name": name,
        "costs": {},
        "fits": {},
        "accuracy": {},
        "restarts": {},
        "curves": {},
        "omega_min": {},
    }
    if run in PLAIN_RUNS:  # plain PDHG to the full budget: both sequences are measured
        res = run_pdhg(lp, x0, restart="none", tol=1e-300, **PLAIN_RUNS[run])
        for seq in ("last", "avg"):
            vals = [s.info[f"kkt_{seq}"] for s in res.trace]
            out["costs"][f"{run}/{seq}"] = first_hit(vals)
            out["fits"][f"{run}/{seq}"] = shape_fit(vals)
            if keep_curve:
                out["curves"][f"{run}/{seq}"] = downsample(vals)
        return out
    res = run_pdhg(lp, x0, tol=TOL, **VARIANTS[run])
    vals = kkts(res)
    out["costs"][run] = float(res.extra["n_matvec"]) if res.converged else math.inf
    if res.converged:
        out["accuracy"][run] = abs(value(res) - f_ref) / (1.0 + abs(f_ref))
    out["restarts"][run] = int(res.extra["n_restarts"])
    out["fits"][run] = shape_fit(vals)
    if run.startswith("pdlp"):
        out["omega_min"][run] = float(min(s.info["omega"] for s in res.trace))
    if keep_curve:
        out["curves"][run] = downsample(vals)
        out["curves"][f"restarts_{run}"] = [
            2.0 + 2.0 * s.k for s in res.trace if s.info["restarted"]
        ]
    return out


def omega_case_study() -> dict[str, Any]:
    """The 3×10 random LP (seed 761) on which the property test found adaptive ω failing."""
    lp = random_lp(CASE_SEED, 3, 10)
    out: dict[str, Any] = {"instance": lp.id, "max_iter": MAX_ITER}
    for lab, kw in (("default", {}), ("pdlp-ω/40", {"primal_weight": "adaptive"})):
        res = run_pdhg(lp, None, tol=TOL, **kw)
        out[lab] = {
            "converged": res.converged,
            "n_matvec": res.extra["n_matvec"],
            "best_kkt": res.extra["kkt"],
            "min_omega": float(min(s.info["omega"] for s in res.trace)),
            "kkt": downsample(kkts(res)),
            "omega": downsample([s.info["omega"] for s in res.trace]),
            "first_kkt_below_3e-8_products": first_hit(kkts(res), 3e-8),
        }
    return out


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def ratio_stats(num: np.ndarray, den: np.ndarray) -> dict[str, Any]:
    """Ratios num/den on the rows both solve, and the rows only one of them solves."""
    ok = np.isfinite(num) & np.isfinite(den)
    r = num[ok] / den[ok]
    return {
        "n_both_solved": int(ok.sum()),
        "n_ratio_le_1": int((r <= 1.0).sum()),
        "n_ratio_gt_1": int((r > 1.0).sum()),
        "median_ratio": float(np.median(r)) if r.size else None,
        "geomean_ratio": float(np.exp(np.mean(np.log(r)))) if r.size else None,
        "min_ratio": float(r.min()) if r.size else None,
        "max_ratio": float(r.max()) if r.size else None,
        "n_only_numerator_solved": int((np.isfinite(num) & ~np.isfinite(den)).sum()),
        "n_only_denominator_solved": int((~np.isfinite(num) & np.isfinite(den)).sum()),
    }


def per_lp(costs: np.ndarray, names: list[str]) -> tuple[list[str], np.ndarray]:
    """Aggregate the starts of each LP: geometric mean over its starts, ∞ if any start fails.

    Since geomean(a)/geomean(b) = geomean(a/b), a ratio of aggregated costs is the geometric
    mean of the per-start ratios. This removes the pseudo-replication of the 3 starts.
    """
    lps = list(dict.fromkeys(lp_of(n) for n in names))
    out = np.full((len(lps), *costs.shape[1:]), np.inf)
    for i, lp in enumerate(lps):
        rows = costs[[j for j, n in enumerate(names) if lp_of(n) == lp]]
        ok = np.all(np.isfinite(rows), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.exp(np.mean(np.log(np.where(np.isfinite(rows), rows, 1.0)), axis=0))
        out[i] = np.where(ok, g, np.inf)
    return lps, out


def shape_summary(fits: list[dict[str, Any] | None]) -> dict[str, Any]:
    f = [x for x in fits if x]
    if not f:
        return {"n_fitted": 0}
    return {
        "n_fitted": len(f),
        "n_linear_better": sum(x["better_fit"] == "linear" for x in f),
        "n_power_better": sum(x["better_fit"] == "power" for x in f),
        "n_reached_tol": sum(x["reached_tol"] for x in f),
        "median_r2_linear": float(np.median([x["r2_linear"] for x in f])),
        "median_r2_power": float(np.median([x["r2_power"] for x in f])),
        "median_power_exponent": float(np.median([x["power_exponent"] for x in f])),
        "range_power_exponent": [
            float(min(x["power_exponent"] for x in f)),
            float(max(x["power_exponent"] for x in f)),
        ],
    }


def save(fig: Any, stem: str) -> None:
    """Write figures/<stem>.svg and figures/<stem>.png."""
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.png", dpi=150)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    insts = instances()
    names = [n for n, _, _ in insts]
    labels = LABELS
    col = {lab: j for j, lab in enumerate(labels)}
    # Heavy instances (random LPs, at the end of the list) first, for load balance.
    order = sorted(range(len(insts)), key=lambda i: (not names[i].startswith("rand_"), i))
    tasks = [(i, r) for i in order for r in (*PLAIN_RUNS, *VARIANTS)]
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        case_future = pool.submit(omega_case_study)
        outs = list(pool.map(one_run, tasks, chunksize=1))
        omega_case = case_future.result()

    P = len(names)
    costs = np.full((P, len(labels)), np.inf)  # (P, S) products to KKT ≤ 1e-8, ∞ = failed
    fits: dict[str, dict[str, Any]] = {n: {} for n in names}
    accuracy: dict[str, dict[str, float]] = {n: {} for n in names}
    restarts: dict[str, dict[str, int]] = {n: {} for n in names}
    omega_min: dict[str, dict[str, float]] = {n: {} for n in names}
    raw_curves: dict[str, dict[str, Any]] = {n: {} for n in CURVE_INSTS}
    for o in outs:
        i = names.index(o["name"])
        for lab, c in o["costs"].items():
            costs[i, col[lab]] = c
        fits[o["name"]].update(o["fits"])
        accuracy[o["name"]].update(o["accuracy"])
        restarts[o["name"]].update(o["restarts"])
        omega_min[o["name"]].update(o["omega_min"])
        if o["curves"]:
            raw_curves[o["name"]].update(o["curves"])

    lib_rows = [i for i, n in enumerate(names) if not n.startswith("rand_")]
    rand_rows = [i for i, n in enumerate(names) if n.startswith("rand_")]
    lp_names, lp_costs = per_lp(costs, names)

    def by_class(num: np.ndarray, den: np.ndarray) -> dict[str, Any]:
        """ratio_stats on the library instances and on the random LPs separately."""
        return {
            "library": ratio_stats(num[lib_rows], den[lib_rows]),
            "random": ratio_stats(num[rand_rows], den[rand_rows]),
        }

    def fixed_cols(cfg: str, grid: tuple[int, ...]) -> list[int]:
        return [col[f"fixed-{p}{CONFIGS[cfg]['suffix']}"] for p in grid]

    def hindsight(cfg: str, grid: tuple[int, ...], table: np.ndarray) -> tuple[np.ndarray, list]:
        """Per-row hindsight-best fixed cost on a grid, and the arg-min period (None if none)."""
        sub = table[:, fixed_cols(cfg, grid)]
        best = sub.min(axis=1)
        arg = [grid[int(np.argmin(r))] if np.isfinite(r.min()) else None for r in sub]
        return best, arg

    def choose_period(cfg: str, grid: tuple[int, ...], rows: list[int]) -> int | None:
        """The period a user would pick from the instances ``rows``: most instances solved;
        ties broken by the geometric mean over the rows that every tied period solves.
        None if no period of the grid solves any of the rows."""
        cols = fixed_cols(cfg, grid)
        sub = costs[np.ix_(rows, cols)]
        n_solved = np.isfinite(sub).sum(axis=0)
        if n_solved.max() == 0:
            return None
        tied = [j for j in range(len(cols)) if n_solved[j] == n_solved.max()]
        common = np.all(np.isfinite(sub[:, tied]), axis=1)
        gm = [float(np.exp(np.mean(np.log(sub[common, j])))) for j in tied]
        return grid[tied[int(np.argmin(gm))]]

    def transfer(cfg: str, grid: tuple[int, ...]) -> tuple[np.ndarray, list[int | None]]:
        """Cost of the period tuned on the siblings and applied to the held-out instance."""
        periods = [choose_period(cfg, grid, siblings(names, i)) for i in range(P)]
        sfx = CONFIGS[cfg]["suffix"]
        cost = np.array(
            [
                math.inf if q is None else costs[i, col[f"fixed-{q}{sfx}"]]
                for i, q in enumerate(periods)
            ]
        )
        return cost, periods

    def best_single(cfg: str) -> int:
        """Best single period over the whole set, chosen in-sample (same rule)."""
        q = choose_period(cfg, PERIODS, list(range(P)))
        assert q is not None
        return q

    configs: dict[str, Any] = {}
    oracles: dict[str, dict[str, np.ndarray]] = {}
    single: dict[str, int] = {}
    transfers: dict[str, dict[str, np.ndarray]] = {}
    for cfg, c in CONFIGS.items():
        main_ad = c["adaptive"][0]
        plain_last = costs[:, col[f"{c['plain']}/last"]]
        res: dict[str, Any] = {"adaptive_labels": list(c["adaptive"]), "oracle": {}}
        oracles[cfg] = {}
        for g, grid in GRIDS.items():
            best, arg = hindsight(cfg, grid, costs)
            # NOTE: per LP we aggregate the per-instance oracle (geomean of per-start best),
            # which is the stronger oracle; then ratio = geomean of per-start ratios.
            _, best_lp_inst = per_lp(best[:, None], names)
            oracles[cfg][g] = best
            res["oracle"][g] = {
                "periods": list(grid),
                "best_period": dict(zip(names, arg, strict=True)),
                "best_products": dict(
                    zip(
                        names, [None if not math.isfinite(v) else int(v) for v in best], strict=True
                    )
                ),
                **{f"{ad}_vs_oracle": ratio_stats(costs[:, col[ad]], best) for ad in c["adaptive"]},
                f"{main_ad}_vs_oracle_by_class": by_class(costs[:, col[main_ad]], best),
                **{
                    f"{ad}_vs_oracle_per_lp": ratio_stats(lp_costs[:, col[ad]], best_lp_inst[:, 0])
                    for ad in c["adaptive"]
                },
            }
        single[cfg] = best_single(cfg)
        sc = col[f"fixed-{single[cfg]}{c['suffix']}"]
        res["best_single_period"] = single[cfg]
        res[f"{main_ad}_vs_best_single"] = ratio_stats(costs[:, col[main_ad]], costs[:, sc])
        res[f"{main_ad}_vs_each_period"] = {
            str(p): ratio_stats(costs[:, col[main_ad]], costs[:, col[f"fixed-{p}{c['suffix']}"]])
            for p in PERIODS
        }
        res["restart_gain_plain_last_vs_adaptive"] = ratio_stats(plain_last, costs[:, col[main_ad]])
        res["restart_gain_plain_last_vs_adaptive_per_lp"] = ratio_stats(
            lp_costs[:, col[f"{c['plain']}/last"]], lp_costs[:, col[main_ad]]
        )
        res["plain_last_vs_oracle_half_octave"] = ratio_stats(
            plain_last, oracles[cfg]["half_octave"]
        )
        res["transfer"] = {}
        transfers[cfg] = {}
        for g, grid in GRIDS.items():
            tcost, tper = transfer(cfg, grid)
            transfers[cfg][g] = tcost
            _, tcost_lp = per_lp(tcost[:, None], names)
            res["transfer"][g] = {
                "period": dict(zip(names, tper, strict=True)),
                **{
                    f"{ad}_vs_transfer": ratio_stats(costs[:, col[ad]], tcost)
                    for ad in c["adaptive"]
                },
                f"{main_ad}_vs_transfer_per_lp": ratio_stats(
                    lp_costs[:, col[main_ad]], tcost_lp[:, 0]
                ),
                "transfer_vs_oracle": ratio_stats(tcost, oracles[cfg][g]),
                f"{main_ad}_vs_transfer_by_class": by_class(costs[:, col[main_ad]], tcost),
            }
        configs[cfg] = res

    table = {}
    for i, n in enumerate(names):
        row: dict[str, Any] = {
            lab: (None if not math.isfinite(costs[i, j]) else int(costs[i, j]))
            for j, lab in enumerate(labels)
        }
        for cfg in CONFIGS:
            for g in GRIDS:
                row[f"oracle[{cfg},{g}]"] = configs[cfg]["oracle"][g]["best_products"][n]
                row[f"oracle period[{cfg},{g}]"] = configs[cfg]["oracle"][g]["best_period"][n]
        table[n] = row

    shape_labels = [
        "plain/last",
        "plain/avg",
        "plain+pc/last",
        "plain+pc/avg",
        "adaptive/40",
        "default",
    ]
    best_fixed_fit = {
        cfg: {
            n: fits[n].get(
                f"fixed-{configs[cfg]['oracle']['half_octave']['best_period'][n]}"
                f"{CONFIGS[cfg]['suffix']}"
            )
            for n in names
            if configs[cfg]["oracle"]["half_octave"]["best_period"][n] is not None
        }
        for cfg in CONFIGS
    }
    shape_by_class: dict[str, Any] = {}
    for lab in shape_labels:
        shape_by_class[lab] = {
            cls: shape_summary([fits[names[i]].get(lab) for i in rows])
            for cls, rows in (("library", lib_rows), ("random", rand_rows))
        }
    for cfg in CONFIGS:
        shape_by_class[f"best fixed ({cfg}, half_octave)"] = {
            cls: shape_summary([best_fixed_fit[cfg].get(names[i]) for i in rows])
            for cls, rows in (("library", lib_rows), ("random", rand_rows))
        }

    # ---------------------------------------------------------------- baselines (exact)
    baselines: dict[str, dict[str, Any]] = {}
    for pid in LIBRARY:
        lp = problems.get(pid)
        _, f_ref = oracle(lp)
        brow: dict[str, Any] = {"linprog_highs_value": f_ref}
        for mid in BASELINES:
            kw = {"pivot_rule": "bland"} if mid.endswith("simplex") else {}
            try:
                r = get_method(mid).fn(lp, **kw)
                brow[mid] = {
                    "converged": r.converged,
                    "n_iter": r.n_iter,
                    "rel_obj_error": abs(value(r) - f_ref) / (1.0 + abs(f_ref)),
                }
            except ValueError as e:
                brow[mid] = {"not_applicable": str(e)}
        d = run_pdhg(lp, None, tol=TOL)
        brow["restarted_pdhg (default)"] = {
            "converged": d.converged,
            "n_iter": d.n_iter,
            "n_matvec": d.extra["n_matvec"],
            "rel_obj_error": abs(value(d) - f_ref) / (1.0 + abs(f_ref)),
        }
        baselines[pid] = brow

    all_acc = [v for a in accuracy.values() for v in a.values()]
    worst_acc = max(
        ((v, n, lab) for n, a in accuracy.items() for lab, v in a.items()), key=lambda t: t[0]
    )
    summary: dict[str, Any] = {
        "settings": {
            "tol": TOL,
            "max_iter": MAX_ITER,
            "budget_products": BUDGET,
            "periods": PERIODS,
            "grids": GRIDS,
            "configs": {
                k: {"kw": v["kw"], "plain": v["plain"], "adaptive": v["adaptive"]}
                for k, v in CONFIGS.items()
            },
            "variants": VARIANTS,
            "beta": math.exp(-1.0),
            "start_seeds": START_SEEDS,
            "random_sizes": RANDOM_SIZES,
            "random_seeds": RANDOM_SEEDS,
            "n_instances": P,
            "n_library_instances": len(lib_rows),
            "n_distinct_lps": len(lp_names),
        },
        "labels": labels,
        "instances": names,
        "products_to_tol": table,
        "solved_count": {lab: int(np.isfinite(costs[:, j]).sum()) for j, lab in enumerate(labels)},
        "solved_count_library": {
            lab: int(np.isfinite(costs[lib_rows, j]).sum()) for j, lab in enumerate(labels)
        },
        "solved_count_per_lp": {
            lab: int(np.isfinite(lp_costs[:, j]).sum()) for j, lab in enumerate(labels)
        },
        "configs": configs,
        "preconditioning_effect_default_vs_adaptive40": ratio_stats(
            costs[:, col["default"]], costs[:, col["adaptive/40"]]
        ),
        "balanced_vs_unit_omega_with_pc": ratio_stats(
            costs[:, col["default"]], costs[:, col["adaptive/40+pc"]]
        ),
        "pdlp_omega40_vs_default": ratio_stats(
            costs[:, col["pdlp-ω/40"]], costs[:, col["default"]]
        ),
        "shape_fits": fits,
        "shape_by_class": shape_by_class,
        "rel_objective_error_vs_highs": accuracy,
        "max_rel_objective_error": {
            "value": worst_acc[0],
            "instance": worst_acc[1],
            "variant": worst_acc[2],
            "n_runs": len(all_acc),
        },
        "n_restarts": restarts,
        "baselines": baselines,
        "min_omega_adaptive_weight": omega_min,
        "omega_case": {
            k: (
                {kk: vv for kk, vv in v.items() if kk not in ("kkt", "omega")}
                if isinstance(v, dict)
                else v
            )
            for k, v in omega_case.items()
        },
    }

    # ---------------------------------------------------------------- curves for the figure
    curves: dict[str, Any] = {}
    for n in CURVE_INSTS:
        rc = raw_curves[n]
        cv: dict[str, Any] = {}
        for cfg, c in CONFIGS.items():
            bp = configs[cfg]["oracle"]["half_octave"]["best_period"][n]
            ad = c["adaptive"][0]
            cv[cfg] = {
                "plain/last": rc[f"{c['plain']}/last"],
                "plain/avg": rc[f"{c['plain']}/avg"],
                **({f"best fixed (T = {bp})": rc[f"fixed-{bp}{c['suffix']}"]} if bp else {}),
                ad: rc[ad],
                f"restarts_{ad}": rc[f"restarts_{ad}"],
            }
        curves[n] = cv

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cfg_title = {"unit": "unit ω, no preconditioning", "pc": "balanced ω + Ruiz/Pock–Chambolle"}

    # 1. Convergence curves: one row per instance, one column per configuration.
    fig, axes = plt.subplots(len(CURVE_INSTS), 2, figsize=(9.6, 11.0), constrained_layout=True)
    for r, n in enumerate(CURVE_INSTS):
        for cidx, cfg in enumerate(CONFIGS):
            ax = axes[r, cidx]
            right = 0.0
            cv = curves[n][cfg]
            ad = CONFIGS[cfg]["adaptive"][0]
            for lab, data in cv.items():
                if lab.startswith("restarts_"):
                    continue
                role = (
                    "best fixed"
                    if lab.startswith("best fixed")
                    else "adaptive"
                    if lab == ad
                    else lab
                )
                style = {"plain/last": ":", "plain/avg": "--"}.get(lab, "-")
                ax.semilogy(
                    data["products"], data["kkt"], style, color=ROLE_COLORS[role], lw=1.4, label=lab
                )
                if not lab.startswith("plain"):
                    right = max(right, max(data["products"]))
            rk = cv[f"restarts_{ad}"]
            yk = np.interp(rk, cv[ad]["products"], cv[ad]["kkt"])
            ax.semilogy(rk, yk, "o", ms=2.5, color=ROLE_COLORS["adaptive"], label="restart")
            ax.axhline(TOL, color="#bbbbbb", lw=0.8)
            ax.set_title(f"{n.replace('@0', '')}, x₀ = 0 — {cfg_title[cfg]}", fontsize=8.5)
            ax.set_ylim(1e-11, 1e2)
            ax.set_xlim(0, min(BUDGET, 2.5 * right))
            ax.legend(
                fontsize=6.5, frameon=True, framealpha=0.85, edgecolor="none", loc="upper right"
            )
            if r == len(CURVE_INSTS) - 1:
                ax.set_xlabel("products with A or Aᵀ")
            if cidx == 0:
                ax.set_ylabel("relative KKT error")
    fig.suptitle(
        "Relative KKT error against work, in each configuration "
        "(best fixed = hindsight best on the half-octave grid)"
    )
    save(fig, "convergence")
    plt.close(fig)

    # 2. Fixed-period sweep relative to the adaptive scheme (zero start; library | random).
    fig, panels = plt.subplots(2, 2, figsize=(10.0, 8.6), sharey=True, constrained_layout=True)
    for ridx, (cfg, c) in enumerate(CONFIGS.items()):
        ad = c["adaptive"][0]
        for cidx, rand in enumerate((False, True)):
            ax = panels[ridx, cidx]
            for p4 in GRIDS["pow4"]:
                ax.axvline(p4, color="#e0e0e0", lw=0.8, zorder=0)
            for n in names:
                if not n.endswith("@0") or n.startswith("rand_") != rand:
                    continue
                i = names.index(n)
                ref = costs[i, col[ad]]
                if not math.isfinite(ref):
                    continue
                vals = costs[i, fixed_cols(cfg, PERIODS)] / ref
                ok = np.isfinite(vals)
                per = np.array(PERIODS, dtype=float)
                (line,) = ax.loglog(per[ok], vals[ok], "-o", ms=2.2, lw=1.1, label=n[:-2])
                if (~ok).any():
                    ax.plot(
                        per[~ok], np.full(int((~ok).sum()), 40.0), "x", ms=3, color=line.get_color()
                    )
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xticks(GRIDS["pow4"], [str(q) for q in GRIDS["pow4"]])
            ax.minorticks_off()
            ax.set_title(
                f"{'random LPs (Ax = b)' if rand else 'numopt lp library'} — {cfg_title[cfg]}",
                fontsize=8.5,
            )
            if ridx == 1:
                ax.set_xlabel("fixed restart period T (iterations; grey lines: powers of 4)")
            if cidx == 0:
                ax.set_ylabel(f"products(fixed-T) / products({ad})")
            ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(
        "Fixed-period sweep on 25 half-octave periods, x₀ = 0 "
        "(× at 40: not solved in the budget; below 1: fixed is cheaper)"
    )
    save(fig, "fixed_period_sweep")
    plt.close(fig)

    # 3. Grid sensitivity: ECDF of adaptive ÷ hindsight-best fixed for each grid.
    fig, panels = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True)
    grid_style = {
        "pow4": ("powers of 4 (7 periods)", ":"),
        "pow2": ("powers of 2 (13 periods)", "--"),
        "half_octave": ("half octaves (25 periods)", "-"),
    }
    for ax, (cfg, c) in zip(panels, CONFIGS.items(), strict=True):
        ad = c["adaptive"][0]
        for g, (glab, ls) in grid_style.items():
            num, den = costs[:, col[ad]], oracles[cfg][g]
            ok = np.isfinite(num) & np.isfinite(den)
            rr = np.sort(num[ok] / den[ok])
            ax.step(
                rr,
                np.arange(1, rr.size + 1) / rr.size,
                ls,
                where="post",
                color=ROLE_COLORS["adaptive"],
                lw=1.4,
                label=f"hindsight best, {glab}: median {np.median(rr):.2f}",
            )
        num, den = costs[:, col[ad]], transfers[cfg]["half_octave"]
        ok = np.isfinite(num) & np.isfinite(den)
        rr = np.sort(num[ok] / den[ok])
        ax.step(
            rr,
            np.arange(1, rr.size + 1) / rr.size,
            "-",
            where="post",
            color=ROLE_COLORS["oracle"],
            lw=1.4,
            label=f"tuned on sibling instances, half octaves: median {np.median(rr):.2f}",
        )
        ax.axvline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"products({ad}) / products(fixed period)")
        ax.set_title(cfg_title[cfg], fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    panels[0].set_ylabel("fraction of instances (both solved)")
    fig.suptitle(
        "Adaptive ÷ fixed period: the hindsight-best period gets stronger as its grid gets finer"
    )
    save(fig, "grid_sensitivity")
    plt.close(fig)

    # 4. Performance profiles (Dolan–Moré), one per configuration.
    prof_sets = {
        "unit": ["plain/last", "plain/avg", f"fixed-{single['unit']}", "adaptive/1", "adaptive/40"],
        "pc": [
            "plain+pc/last",
            "plain+pc/avg",
            f"fixed-{single['pc']}+pc",
            "default",
            "adaptive/40+pc",
            "pdlp-ω/40",
        ],
    }
    fig, panels = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    summary["performance_profile"] = {}
    for ax, (cfg, labs) in zip(panels, prof_sets.items(), strict=True):
        orc = f"best fixed (hindsight, {cfg})"
        tab = np.column_stack([costs[:, [col[lab] for lab in labs]], oracles[cfg]["half_octave"]])
        all_labs = [*labs, orc]
        prof = bench.performance_profile_from_costs(tab, all_labs, tau=TOL)
        bench.plot_performance_profile(prof, ax=ax, title=f"{cfg_title[cfg]} ({P} instances)")
        summary["performance_profile"][cfg] = {
            "labels": all_labs,
            "rho_at_1": dict(zip(all_labs, prof.at(1.0).tolist(), strict=True)),
            "rho_at_2": dict(zip(all_labs, prof.at(2.0).tolist(), strict=True)),
            "solved_fraction": dict(zip(all_labs, prof.solved.tolist(), strict=True)),
        }
    fig.suptitle(
        "Performance profiles, cost = products to KKT ≤ 1e-8 "
        "(the hindsight oracle is a reference, not a method)"
    )
    save(fig, "performance_profile")
    plt.close("all")

    # 5. Primal-weight case study.
    fig, (a1x, a2x) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for lab, color in (
        ("default", ROLE_COLORS["adaptive"]),
        ("pdlp-ω/40", ROLE_COLORS["pdlp-ω/40"]),
    ):
        d = omega_case[lab]
        a1x.semilogy(d["kkt"]["products"], d["kkt"]["kkt"], color=color, label=lab)
        a2x.semilogy(d["omega"]["products"], d["omega"]["kkt"], color=color, label=lab)
    # (downsample() stores any per-step series under the key "kkt")
    a1x.axhline(TOL, color="#bbbbbb", lw=0.8)
    a1x.set_ylabel("relative KKT error")
    a2x.set_ylabel("primal weight ω")
    for a in (a1x, a2x):
        a.set_xlabel("products with A or Aᵀ")
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"PDLP primal-weight update on {omega_case['instance']}")
    save(fig, "primal_weight")
    plt.close(fig)

    summary["runtime_seconds"] = time.perf_counter() - t_start
    (RESULTS / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=1, allow_nan=False)
    )
    (RESULTS / "curves.json").write_text(
        json.dumps(to_jsonable({"curves": curves, "omega_case": omega_case}), allow_nan=False)
    )
    brief = {
        "solved_count": summary["solved_count"],
        "best_single": single,
        **{
            f"{cfg}:{k}": v
            for cfg, r in configs.items()
            for k, v in r.items()
            if k.startswith("restart_gain") or k.endswith("best_single")
        },
        **{
            f"{cfg}:{g}:{k}": v
            for cfg, r in configs.items()
            for g, o in r["oracle"].items()
            for k, v in o.items()
            if "_vs_oracle" in k
        },
        **{
            f"{cfg}:transfer:{g}:{k}": v
            for cfg, r in configs.items()
            for g, o in r["transfer"].items()
            for k, v in o.items()
            if k != "period"
        },
        "runtime_seconds": summary["runtime_seconds"],
    }
    print(json.dumps(to_jsonable(brief), indent=1))


if __name__ == "__main__":
    main()
