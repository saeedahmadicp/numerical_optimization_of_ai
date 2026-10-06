"""Experiment: does rotating a problem slow down coordinate-wise Adam successors?

For each problem P, angle θ and method M we run M on g_θ(x) = P(R(θ)x) from the rotated start
points R(θ)ᵀx₀, with no gradient stop, for a fixed horizon of H iterations, and record

    N = the settling iteration: the smallest k such that g_θ(x_j) − g* < 1e-6 for every
        j = k, …, H (g* = 0 for both problems).

A run counts as solved only if N ≤ N_max ≤ H/2, i.e. f is seen to stay below the target for at
least N_max further iterations. A run that only passes through the target (oscillates,
spikes, or ends above it) is unsolved. The first-hit count (first k with f(x_k) < 1e-6) is
recorded as a diagnostic only.

Hyperparameters come from a fixed grid per method (stated below; every tuned hyperparameter
is checked against its grid edges). Two protocols:

* tuned: at every angle, the grid point with the fewest unsolved starts, then the smallest
  median N, is chosen (the best each method can do at that orientation);
* fixed: the grid point chosen at the reference orientation is kept at every angle (the
  practitioner's view, and the protocol of Xie et al. 2025: same hyperparameters on the
  original and the rotated loss).

Writes results/*.json and figures/*.svg. Deterministic: no randomness except Sophia-H's
Hutchinson draws, which use numopt.core.rng.Rng(seed = start index). Run:

    .venv/bin/python research/adam-successors-rotation/run.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

# 2-D linear algebra: BLAS threads only add contention between the pool workers.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from method import (  # noqa: E402
    GTOL_MIN,
    adabelief,
    adan,
    hessian_11_norm,
    lion,
    rotate,
    rotation_2d,
    sophia,
)

from numopt import bench, problems  # noqa: E402
from numopt.core.registry import get_method  # noqa: E402
from numopt.core.types import Problem, Result  # noqa: E402

TARGET = 1e-6  # solved: f(x_k) − f* < TARGET from the settling iteration N to the horizon
# NOTE: no gradient stop. gtol = 1e-150 is the smallest value the methods accept (‖g‖₂
# underflows below it); a run that reaches it sits at a stationary point with f < TARGET.
GTOL = GTOL_MIN
#: A trajectory whose peak f(x_k)/f(x₀) exceeds this is flagged as an overshoot.
OVERSHOOT = 10.0
PHI0 = math.atan2(0.6, 0.8)  # quadratic_ill = ½xᵀQ diag(1, 50)Qᵀx with Q = R(PHI0)
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def lr_grid(lo: float, hi: float, per_decade: int) -> list[float]:
    """10^lo, 10^(lo + 1/per_decade), …, 10^hi."""
    n = round((hi - lo) * per_decade)
    return [float(10.0 ** (lo + i / per_decade)) for i in range(n + 1)]


def ring(
    center: np.ndarray, radius: float, frame: np.ndarray, angles_deg: list[float]
) -> list[list[float]]:
    """Start points center + frame·r(cos ψ, sin ψ) for ψ in ``angles_deg``."""
    out = []
    for a in angles_deg:
        psi = math.radians(a)
        out.append(
            [float(v) for v in center + frame @ (radius * np.array([math.cos(psi), math.sin(psi)]))]
        )
    return out


# --------------------------------------------------------------------------------------
# Problems, angles and start points
# --------------------------------------------------------------------------------------

Q = rotation_2d(PHI0)
# quadratic_ill: the literal grid θ = 0, 7.5, …, 45° plus θ = φ₀ − φ for φ = 0, 15, 30, 45°, so
# that the misalignment φ = |φ₀ − θ| between the Hessian eigenbasis and the axes covers [0°, 45°].
QUAD_THETAS = [7.5 * i for i in range(7)] + [math.degrees(PHI0) - phi for phi in (0, 15, 30, 45)]
PROBLEMS: dict[str, dict[str, Any]] = {
    "quadratic_ill": {
        "thetas_deg": QUAD_THETAS,
        "ref_deg": math.degrees(PHI0),  # φ = 0: eigenbasis aligned with the axes
        # Starts on a circle of radius 2√2 (= ‖default x₀‖) in the eigenbasis frame at
        # ψ = 22.5°, 67.5°, 112.5°, 157.5°. With the starts at ψ + 180° (equal counts for every
        # method: f(−x) = f(x) and −I is a signed permutation) the set is symmetric under a
        # reflection of either eigen-axis, so the counts depend on φ only.
        "starts": ring(np.zeros(2), 2 * math.sqrt(2), Q, [22.5, 67.5, 112.5, 157.5]),
        "n_max": 5000,  # largest N that counts as solved
        "horizon": 40000,  # iterations of the sweep; f must stay below TARGET from N to here
        "certify_horizon": 40000,  # horizon of every reported selection (see certify())
        "L_mu": (50.0, 1.0),
    },
    "rosenbrock": {
        "thetas_deg": [7.5 * i for i in range(7)],
        "ref_deg": 0.0,  # the original problem
        # Six starts at distance 1.2 from x* = (1, 1), ψ = 15° + 60°j.
        "starts": ring(np.ones(2), 1.2, np.eye(2), [15.0 + 60.0 * j for j in range(6)]),
        "n_max": 20000,
        "horizon": 40000,
        "certify_horizon": 160000,
    },
}


# --------------------------------------------------------------------------------------
# Methods and search grids
# --------------------------------------------------------------------------------------


def _registered(id_: str) -> Callable[..., Result]:
    return get_method(id_).fn


# label → (function, fixed keyword arguments)
METHODS: dict[str, tuple[Callable[..., Result], dict[str, Any]]] = {
    "gd": (_registered("gradient_descent"), {"step_rule": "fixed"}),
    "momentum": (_registered("momentum"), {"beta": 0.9}),  # heavy ball at Adam's β₁ = 0.9
    "heavy_ball": (_registered("momentum"), {}),  # heavy ball with β tuned as well
    "adam": (_registered("adam"), {}),
    "adamw": (_registered("adamw"), {}),
    "amsgrad": (_registered("amsgrad"), {}),
    "nadam": (_registered("nadam"), {}),
    "rmsprop": (_registered("rmsprop"), {}),
    "adabelief": (adabelief, {}),
    "adan": (adan, {}),
    "sophia": (sophia, {}),
    "sophia_h": (sophia, {"estimator": "hutchinson"}),
    "lion": (lion, {}),
}
NEW = ("adabelief", "lion", "adan", "sophia", "sophia_h")


def _lr(grid: list[float]) -> list[dict[str, float]]:
    return [{"lr": v} for v in grid]


def _product(lrs: list[float], key: str, values: Sequence[float]) -> list[dict[str, float]]:
    return [{"lr": v, key: d} for d in values for v in lrs]


def lion_decays(j_lo: int, j_hi: int) -> list[float]:
    """ρ = 1 − 10^(−j/4), j = j_lo, …, j_hi: quarter-decade steps in the decay rate 1 − ρ."""
    return [1.0 - 10.0 ** (-j / 4) for j in range(j_lo, j_hi + 1)]


# Wide grids, the same for every adaptive method on a problem (half-decade steps on
# quadratic_ill, third-decade on rosenbrock). results/summary.json lists every tuned choice
# that lies on an edge of any tuned hyperparameter (`tuned_best_on_grid_edge`).
ADAPTIVE = {"quadratic_ill": lr_grid(-4, 2, 2), "rosenbrock": lr_grid(-5, 1, 3)}
HB_BETAS = {
    "quadratic_ill": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98),
    "rosenbrock": (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995),
}
ADAPTIVE_METHODS = [m for m in METHODS if m not in ("gd", "momentum", "heavy_ball", "lion")]
GRIDS: dict[str, dict[str, list[dict[str, float]]]] = {
    "quadratic_ill": {
        "gd": _lr(lr_grid(-3, -1, 4)),
        "momentum": _lr(lr_grid(-4, -1, 2)),
        "heavy_ball": _product(lr_grid(-3, -1, 4), "beta", HB_BETAS["quadratic_ill"]),
        **{m: _lr(ADAPTIVE["quadratic_ill"]) for m in ADAPTIVE_METHODS},
        "lion": _product(lr_grid(-3, 2, 2), "lr_decay", lion_decays(2, 14)),
    },
    "rosenbrock": {
        "gd": _lr(lr_grid(-4, -2, 3)),
        "momentum": _lr(lr_grid(-5, -2, 3)),
        "heavy_ball": _product(lr_grid(-5, -2, 3), "beta", HB_BETAS["rosenbrock"]),
        **{m: _lr(ADAPTIVE["rosenbrock"]) for m in ADAPTIVE_METHODS},
        "lion": _product(lr_grid(-4, 1.5, 2), "lr_decay", lion_decays(4, 16)),
    },
}


# --------------------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------------------


def first_hit(fs: Sequence[float | None], ks: Sequence[int], target: float = TARGET) -> int | None:
    """First k with f(x_k) < target (diagnostic only: a passing crossing also counts)."""
    for f, k in zip(fs, ks, strict=True):
        if f is not None and f < target:
            return k
    return None


def settle_iteration(
    fs: Sequence[float | None], ks: Sequence[int], target: float = TARGET
) -> int | None:
    """Smallest k_i such that f_j < target for every j ≥ i (to the end of the trace).

    None when the last value is not below the target (NaN, None and ∞ count as not below).
    """
    last_bad = -1
    for i, f in enumerate(fs):
        if f is None or not f < target:
            last_bad = i
    if last_bad == len(fs) - 1:
        return None
    return ks[last_bad + 1]


def solved_n(settle: int | None, n_max: int) -> int | None:
    """N if the run settled no later than n_max (so it stayed below for ≥ horizon − n_max)."""
    return settle if settle is not None and settle <= n_max else None


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------


def rotated(problem_id: str, theta_deg: float) -> tuple[Problem, np.ndarray]:
    R = rotation_2d(math.radians(theta_deg))
    return rotate(problems.get(problem_id), R, tag=f"rot{theta_deg:g}"), R


def run_one(
    problem_id: str,
    theta_deg: float,
    method: str,
    cfg: dict[str, Any],
    start: int,
    horizon: int | None = None,
) -> tuple[Result, Problem]:
    spec = PROBLEMS[problem_id]
    horizon = spec["horizon"] if horizon is None else horizon
    P, R = rotated(problem_id, theta_deg)
    fn, fixed = METHODS[method]
    extra: dict[str, Any] = {"seed": start} if method == "sophia_h" else {}
    x0 = R.T @ np.asarray(spec["starts"][start])
    res = fn(P, x0=x0, gtol=GTOL, max_iter=horizon, **fixed, **cfg, **extra)
    return res, P


def lion_frozen(res: Result, P: Problem, cfg: dict[str, Any]) -> bool:
    """True when Lion's remaining schedule cannot carry x_K to x*.

    After iteration K every coordinate moves by at most η_t = lr·ρ^(t−1) per step (λ = 0), so
    the total remaining travel is Σ_{t>K} η_t = lr·ρ^K/(1 − ρ). If that is below
    ‖x_K − x*‖_∞ the iterate is frozen by the schedule and can never reach x*.
    """
    rho = cfg["lr_decay"]
    if rho >= 1.0:
        return False
    remaining = cfg["lr"] * rho**res.n_iter / (1.0 - rho)
    dist = float(np.max(np.abs(np.asarray(res.x) - np.asarray(P.minima[0]))))
    return remaining < dist


Key = tuple[str, int, str, int, int]  # (problem, angle index, method, config index, start)


def _task(args: tuple[str, int, str, int, int, int]) -> tuple[Key, dict[str, Any]]:
    problem_id, a, method, c, s, horizon = args
    t0 = time.process_time()
    spec = PROBLEMS[problem_id]
    cfg = GRIDS[problem_id][method][c]
    res, P = run_one(problem_id, spec["thetas_deg"][a], method, cfg, s, horizon)
    fs = [st.fun for st in res.trace]
    ks = [st.k for st in res.trace]
    settle = settle_iteration(fs, ks)
    finite = [f for f in fs if f is not None and math.isfinite(f)]
    f0 = fs[0] if fs[0] is not None else math.nan
    peak = max(finite) if len(finite) == len(fs) else math.inf
    rec = {
        "N": solved_n(settle, spec["n_max"]),
        "horizon": horizon,
        "settle": settle,
        "first": first_hit(fs, ks),
        "n_iter": res.n_iter,
        "converged": res.converged,
        "f_final": res.fun if res.fun is not None and math.isfinite(res.fun) else None,
        "peak_ratio": peak / f0 if math.isfinite(peak) and f0 > 0 else None,
        "frozen": lion_frozen(res, P, cfg) if method == "lion" else None,
        "cpu": time.process_time() - t0,
    }
    return (problem_id, a, method, c, s), rec


def all_tasks() -> list[tuple[str, int, str, int, int, int]]:
    tasks = []
    for pid, spec in PROBLEMS.items():
        for m, grid in GRIDS[pid].items():
            for a in range(len(spec["thetas_deg"])):
                for c in range(len(grid)):
                    for s in range(len(spec["starts"])):
                        tasks.append((pid, a, m, c, s, spec["horizon"]))
    # Longest jobs first (rosenbrock has the longer horizon) for a better pool schedule.
    tasks.sort(key=lambda t: t[0] != "rosenbrock")
    return tasks


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def median_inf(v: np.ndarray) -> float:
    """Median with inf for unsolved starts (inf sorts last, as a cost of ∞ should)."""
    return float(np.median(v))


def choose(N: np.ndarray) -> int:
    """Index of the best config in N[config, start]: fewest unsolved, then smallest median."""
    keys = [(int(np.isinf(row).sum()), median_inf(row), float(np.mean(row))) for row in N]
    return min(range(len(keys)), key=lambda i: keys[i])


def edge_keys(grid: list[dict[str, float]], i: int) -> list[str]:
    """Hyperparameters of grid[i] that sit on the smallest or largest value of their grid."""
    out = []
    for key in grid[i]:
        values = sorted({g[key] for g in grid})
        if len(values) > 1 and grid[i][key] in (values[0], values[-1]):
            out.append(key)
    return out


def phi_of(problem_id: str, theta_deg: float) -> float:
    """Misalignment (degrees, folded to [0, 45]) of the Hessian eigenbasis at x* and the axes."""
    if problem_id == "quadratic_ill":
        H = np.asarray(problems.get("quadratic_ill").hess(np.zeros(2)))
    else:
        H = np.asarray(problems.get("rosenbrock").hess(np.ones(2)))
    R = rotation_2d(math.radians(theta_deg))
    _, V = np.linalg.eigh(R.T @ H @ R)
    ang = math.degrees(math.atan2(V[1, -1], V[0, -1])) % 90.0
    return min(ang, 90.0 - ang)


def h11_at_minimizer(problem_id: str, theta_deg: float) -> float:
    P, _ = rotated(problem_id, theta_deg)
    assert P.hess is not None
    return hessian_11_norm(P.hess(np.asarray(P.minima[0])))


#: Stopping test of the GD path used for the path-averaged ‖H‖₁,₁ (near x*, μ ≈ 0.399, so
#: ‖∇f‖ ≤ 5e-4 gives f − f* ≲ 3.1e-7): the average is over the approach, not the tail at x*.
PATH_GTOL = 5e-4


def h11_along_gd(problem_id: str, thetas_deg: list[float], gd_lr: float) -> list[float]:
    """Mean ‖∇²g_θ(x_k)‖₁,₁ over the GD iterates from all starts, for every θ.

    GD is rotation-equivariant: on g_θ its iterates are R(θ)ᵀx_k with x_k the iterates on the
    unrotated problem, and ∇²g_θ(R(θ)ᵀx_k) = R(θ)ᵀ∇²f(x_k)R(θ). So one GD run per start on f
    gives the ℓ∞-smoothness surrogate along the same path at every angle.
    """
    base = problems.get(problem_id)
    gd, fixed = METHODS["gd"]
    spec = PROBLEMS[problem_id]
    assert base.hess is not None
    hs = []
    for x0 in spec["starts"]:
        res = gd(base, x0=x0, gtol=PATH_GTOL, max_iter=spec["n_max"], lr=gd_lr, **fixed)
        hs.extend(np.asarray(base.hess(np.asarray(st.x))) for st in res.trace)
    H = np.array(hs)  # (K, 2, 2)
    out = []
    for t in thetas_deg:
        R = rotation_2d(math.radians(t))
        HR = np.einsum("ji,kjl,lm->kim", R, H, R)  # Rᵀ H_k R for every k
        out.append(float(np.abs(HR).sum(axis=(1, 2)).mean()))
    return out


def spearman(x: list[float], y: list[float]) -> dict[str, float] | None:
    """Spearman ρ and its two-sided p-value (None when one side is constant)."""
    if len(set(y)) < 2 or len(set(x)) < 2:
        return None
    rho, p = cast(tuple[float, float], spearmanr(x, y))
    return {"rho": float(rho), "p": float(p)}


def as_json_num(v: float) -> float | None:
    return None if not math.isfinite(v) else v


def momentum_floor(f0: float, beta: float) -> float:
    """ln((f₀ − f*)/TARGET)/ln(1/β): iterations a heavy-ball iteration with momentum β needs
    when its asymptotic rate (√β per step in ‖x − x*‖, β per step in f) is the binding limit.
    The roots of z² − (1 + β − aλ)z + β have product β, so the spectral radius is ≥ √β."""
    return math.log(f0 / TARGET) / math.log(1.0 / beta)


def polyak_heavy_ball(pid: str) -> dict[str, Any]:
    """Heavy ball with the optimal quadratic parameters lr = 4/(√L + √μ)², β = ((√κ−1)/(√κ+1))²
    (Polyak 1964): the rotation-equivariant reference rate, run at the reference angle."""
    spec = PROBLEMS[pid]
    L, mu = spec["L_mu"]
    lr = 4.0 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    beta = ((math.sqrt(L / mu) - 1.0) / (math.sqrt(L / mu) + 1.0)) ** 2
    Ns, firsts = [], []
    for s in range(len(spec["starts"])):
        P, R = rotated(pid, spec["ref_deg"])
        res = METHODS["momentum"][0](
            P,
            x0=R.T @ np.asarray(spec["starts"][s]),
            gtol=GTOL,
            max_iter=spec["certify_horizon"],
            lr=lr,
            beta=beta,
        )
        fs, ks = [st.fun for st in res.trace], [st.k for st in res.trace]
        Ns.append(solved_n(settle_iteration(fs, ks), spec["n_max"]))
        firsts.append(first_hit(fs, ks))
    return {"lr": lr, "beta": beta, "N": Ns, "first": firsts, "median": float(np.median(Ns))}


def _stats(
    arrays: dict[str, np.ndarray], grid: list[dict[str, float]], m: str, a: int, c: int
) -> dict[str, Any]:
    """Summary of the runs at angle a and grid point c (arrays indexed [angle, config, start])."""
    rows = arrays["N"][a, c]
    d: dict[str, Any] = {
        "median": as_json_num(median_inf(rows)),
        "min": as_json_num(float(rows.min())),
        "max": as_json_num(float(rows.max())),
        "solved": int(np.isfinite(rows).sum()),
        "per_start": [as_json_num(float(v)) for v in rows],
        "first_hit_median": as_json_num(median_inf(arrays["first"][a, c])),
        "peak_ratio_max": as_json_num(float(arrays["peak"][a, c].max())),
        "horizon": int(arrays["horizon"][a, c].min()),
        "config": grid[c],
    }
    if m == "lion":
        d["unsolved_frozen"] = int((~np.isfinite(rows) & arrays["frozen"][a, c]).sum())
    return d


def analyse(raw: dict[tuple[str, int, str, int, int], dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pid, spec in PROBLEMS.items():
        thetas = spec["thetas_deg"]
        n_s = len(spec["starts"])
        ref = int(np.argmin([abs(t - spec["ref_deg"]) for t in thetas]))
        prob_out: dict[str, Any] = {
            "thetas_deg": thetas,
            "phi_deg": [phi_of(pid, t) for t in thetas],
            "h11_at_minimizer": [h11_at_minimizer(pid, t) for t in thetas],
            "ref_index": ref,
            "n_max": spec["n_max"],
            "horizon": spec["horizon"],
            "certify_horizon": spec["certify_horizon"],
            "methods": {},
        }
        f0 = []
        for s in range(n_s):
            P, R = rotated(pid, thetas[ref])
            f0.append(float(P.f(R.T @ np.asarray(spec["starts"][s]))))
        prob_out["f0_starts"] = f0
        for m, grid in GRIDS[pid].items():
            shape = (len(thetas), len(grid), n_s)
            N = np.full(shape, np.inf)
            peak = np.full(shape, np.inf)
            first = np.full(shape, np.inf)
            frozen = np.zeros(shape, dtype=bool)
            horizon = np.zeros(shape)
            diag = {
                "certified_runs": 0,
                "certified_runs_whose_N_changed": 0,
                "runs": 0,
                "first_hit": 0,
                "left_after_first_hit": 0,
                "hit_but_unsolved": 0,
                "stopped_at_gtol": 0,
            }
            for a in range(len(thetas)):
                for c in range(len(grid)):
                    for s in range(n_s):
                        r = raw[(pid, a, m, c, s)]
                        diag["runs"] += 1
                        horizon[a, c, s] = r["horizon"]
                        if "N_sweep" in r:
                            diag["certified_runs"] += 1
                            diag["certified_runs_whose_N_changed"] += r["N"] != r["N_sweep"]
                        if r["N"] is not None:
                            N[a, c, s] = r["N"]
                        if r["peak_ratio"] is not None:
                            peak[a, c, s] = r["peak_ratio"]
                        if r["first"] is not None:
                            first[a, c, s] = r["first"]
                            diag["first_hit"] += 1
                            if r["settle"] != r["first"]:
                                diag["left_after_first_hit"] += 1
                            if r["N"] is None:
                                diag["hit_but_unsolved"] += 1
                        if r["converged"]:
                            diag["stopped_at_gtol"] += 1
                        frozen[a, c, s] = bool(r["frozen"])
            tuned_idx = [choose(N[a]) for a in range(len(thetas))]
            fixed_idx = tuned_idx[ref]

            arrays = {"N": N, "first": first, "peak": peak, "frozen": frozen, "horizon": horizon}
            tuned = [_stats(arrays, grid, m, a, i) for a, i in enumerate(tuned_idx)]
            fixed = [_stats(arrays, grid, m, a, fixed_idx) for a in range(len(thetas))]
            edges = [
                {"theta_deg": thetas[a], "keys": edge_keys(grid, i)}
                for a, i in enumerate(tuned_idx)
                if edge_keys(grid, i) and np.isfinite(median_inf(N[a, i]))
            ]
            entry: dict[str, Any] = {
                "grid": grid,
                "N": [
                    [[as_json_num(v) for v in N[a, c]] for c in range(len(grid))]
                    for a in range(len(thetas))
                ],
                "tuned": tuned,
                "fixed": fixed,
                "tuned_best_on_grid_edge": edges,
                "overshoot": {
                    proto: [
                        thetas[a]
                        for a, r in enumerate(rows)
                        if r["median"] is not None
                        and (r["peak_ratio_max"] is None or r["peak_ratio_max"] > OVERSHOOT)
                    ]
                    for proto, rows in (("tuned", tuned), ("fixed", fixed))
                },
                "diagnostics": diag,
            }
            for proto in ("tuned", "fixed"):
                med = [r["median"] if r["median"] is not None else math.inf for r in entry[proto]]
                entry[f"{proto}_spearman_h11"] = spearman(prob_out["h11_at_minimizer"], med)
                entry[f"{proto}_spearman_phi"] = spearman(prob_out["phi_deg"], med)
                lit = [i for i, t in enumerate(thetas) if pid != "quadratic_ill" or i < 7]
                entry[f"{proto}_spearman_theta_literal"] = spearman(
                    [thetas[i] for i in lit], [med[i] for i in lit]
                )
                finite = [v for v in med if math.isfinite(v)]
                # null when some angle is unsolved at the median (the ratio is unbounded).
                entry[f"{proto}_max_over_ref"] = (
                    as_json_num(max(med) / med[ref]) if math.isfinite(med[ref]) else None
                )
                entry[f"{proto}_max_over_min"] = (
                    as_json_num(max(med) / min(finite)) if finite else None
                )
            if pid == "quadratic_ill":
                # S(cfg) = median N at φ = 45° / median N at φ = 0°, for every grid point.
                i45 = int(np.argmax(prob_out["phi_deg"]))
                sens = []
                for c in range(len(grid)):
                    a0, a45 = median_inf(N[ref, c]), median_inf(N[i45, c])
                    sens.append(
                        {
                            "config": grid[c],
                            "N_phi0": as_json_num(a0),
                            "N_phi45": as_json_num(a45),
                            "ratio": as_json_num(a45 / a0) if math.isfinite(a0) else None,
                        }
                    )
                entry["sensitivity_by_config"] = sens
            prob_out["methods"][m] = entry
        # Momentum floor ln(f₀/ε)/ln(1/β) per start, and each method's N at its reference
        # tuned config divided by it (start by start). β: the method's momentum coefficient
        # in EMA form (Adam family β₁ = 0.9, Sophia β₁ = 0.96, Adan 1 − β₁ = 0.98, heavy ball β).
        betas = {
            "momentum": 0.9,
            "adam": 0.9,
            "adamw": 0.9,
            "amsgrad": 0.9,
            "nadam": 0.9,
            "adabelief": 0.9,
            "sophia": 0.96,
            "sophia_h": 0.96,
            "adan": 0.98,
        }
        hb_beta = prob_out["methods"]["heavy_ball"]["tuned"][ref]["config"]["beta"]
        betas["heavy_ball"] = hb_beta
        floor: dict[str, Any] = {}
        for m, b in betas.items():
            fl = [momentum_floor(v, b) for v in f0]
            per = prob_out["methods"][m]["tuned"][ref]["per_start"]
            floor[m] = {
                "beta": b,
                "floor_per_start": fl,
                "floor_median": float(np.median(fl)),
                "N_over_floor_per_start": [
                    None if n is None else n / f for n, f in zip(per, fl, strict=True)
                ],
            }
        prob_out["momentum_floor"] = floor
        if pid == "quadratic_ill":
            prob_out["polyak_heavy_ball"] = polyak_heavy_ball(pid)
        if pid == "rosenbrock":
            gd_lr = prob_out["methods"]["gd"]["tuned"][ref]["config"]["lr"]
            prob_out["h11_along_gd"] = h11_along_gd(pid, thetas, gd_lr)
            for entry in prob_out["methods"].values():
                for proto in ("tuned", "fixed"):
                    med = [
                        r["median"] if r["median"] is not None else math.inf for r in entry[proto]
                    ]
                    entry[f"{proto}_spearman_h11_gd_path"] = spearman(prob_out["h11_along_gd"], med)
        out[pid] = prob_out
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

# Validated categorical slots (dataviz reference palette, light mode) in fixed order; the
# rotation-equivariant controls are neutral grays. Same method → same colour in every figure.
COLORS = {
    "adam": "#2a78d6",
    "adabelief": "#eb6834",
    "lion": "#1baf7a",
    "adan": "#eda100",
    "sophia": "#e87ba4",
    "gd": "#52514e",
    "heavy_ball": "#9a9893",
}
LABELS = {
    "gd": "GD (control)",
    "adam": "Adam",
    "adabelief": "AdaBelief",
    "lion": "Lion",
    "adan": "Adan",
    "sophia": "Sophia",
    "sophia_h": "Sophia-H",
    "momentum": "Heavy ball β=0.9",
    "heavy_ball": "Heavy ball (control)",
    "adamw": "AdamW",
    "amsgrad": "AMSGrad",
    "nadam": "NAdam",
    "rmsprop": "RMSprop",
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
CONTROLS = ("gd", "heavy_ball")


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label_ends(ax: Any, xs: list[float], series: dict[str, list[float]]) -> None:
    """Direct labels at the right end of each line, nudged apart in log space."""
    ends = sorted(((math.log10(v[-1]), m) for m, v in series.items() if math.isfinite(v[-1])))
    placed: list[float] = []
    for y, m in ends:
        y_adj = max(y, placed[-1] + 0.075) if placed else y
        placed.append(y_adj)
        ax.annotate(
            LABELS[m],
            (xs[-1], 10**y),
            xytext=(xs[-1] + 1.2, 10**y_adj),
            fontsize=8,
            color=INK,
            va="center",
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.6},
        )


def figure_angles(summary: dict[str, Any], path: Path) -> None:
    from matplotlib.figure import Figure

    shown = ("gd", "heavy_ball", "adam", "adabelief", "lion", "adan", "sophia")
    fig = Figure(figsize=(10.5, 7.2), layout="constrained", facecolor="white")
    axes = fig.subplots(2, 2)
    for col, pid in enumerate(("quadratic_ill", "rosenbrock")):
        pr = summary[pid]
        if pid == "quadratic_ill":
            x = pr["phi_deg"]
            xlabel = r"misalignment $\varphi$ of the Hessian eigenbasis (degrees)"
        else:
            x = pr["thetas_deg"]
            xlabel = r"rotation angle $\theta$ (degrees)"
        order = np.argsort(x, kind="stable")
        xs = [x[i] for i in order]
        for row, proto in enumerate(("tuned", "fixed")):
            ax = axes[row, col]
            _style(ax)
            series = {}
            notes = []
            for m in shown:
                med = [pr["methods"][m][proto][i]["median"] for i in order]
                ys = [v if v is not None else math.inf for v in med]
                series[m] = ys
                # NaN breaks the line, so no segment is drawn across an unsolved angle.
                ax.plot(
                    xs,
                    [v if math.isfinite(v) else math.nan for v in ys],
                    color=COLORS[m],
                    lw=2.4 if m in CONTROLS else 2.0,
                    ls="--" if m in CONTROLS else "-",
                    marker="o",
                    ms=4.5,
                    mec="white",
                    mew=0.8,
                    zorder=3,
                )
                bad = [i for i, v in enumerate(ys) if not math.isfinite(v)]
                if bad:
                    top = pr["n_max"]
                    ax.plot(
                        [xs[i] for i in bad],
                        [top] * len(bad),
                        ls="none",
                        marker="x",
                        color=COLORS[m],
                        ms=7,
                        mew=2,
                    )
                    notes.append(LABELS[m])
            ax.set_yscale("log")
            ax.set_xlim(min(xs) - 1, max(xs) + 14)
            _label_ends(ax, xs, series)
            ax.set_xlabel(xlabel, fontsize=9, color=INK2)
            ax.set_ylabel(
                r"median iterations $N$ to settle below $10^{-6}$", fontsize=9, color=INK2
            )
            title = {
                "tuned": "hyperparameters re-tuned at every angle",
                "fixed": "tuned at the reference, then fixed",
            }[proto]
            name = "quadratic_ill (κ = 50)" if pid == "quadratic_ill" else "Rosenbrock"
            sub = (
                f"\n× at N_max = {pr['n_max']}: median start does not settle ({', '.join(notes)})"
                if notes
                else ""
            )
            ax.set_title(f"{name}: {title}{sub}", fontsize=10, color=INK, loc="left")
    fig.savefig(path)


def figure_sensitivity(summary: dict[str, Any], path: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    shown = ("adam", "adabelief", "lion", "adan", "sophia")
    pr = summary["quadratic_ill"]
    ref = pr["ref_index"]
    fig = Figure(figsize=(12, 3.3), layout="constrained", facecolor="white")
    axes = fig.subplots(1, len(shown), sharey=True)
    for ax, m in zip(axes, shown, strict=True):
        _style(ax)
        entry = pr["methods"][m]
        sens = entry["sensitivity_by_config"]
        d = None
        if m == "lion":  # one schedule: the decay chosen at the reference angle
            d = entry["fixed"][ref]["config"]["lr_decay"]
            sens = [r for r in sens if r["config"]["lr_decay"] == d]
        pts = [(r["config"]["lr"], r["ratio"]) for r in sens if r["ratio"] is not None]
        if pts:
            ax.plot(*zip(*pts, strict=True), color=COLORS[m], lw=2, marker="o", ms=4.5, mec="white")
        # Settles at φ = 0 but not at φ = 45°: the ratio exceeds N_max/N(0).
        cens = [
            (r["config"]["lr"], pr["n_max"] / r["N_phi0"])
            for r in sens
            if r["N_phi0"] is not None and r["N_phi45"] is None
        ]
        if cens:
            ax.plot(
                *zip(*cens, strict=True),
                ls="none",
                marker="^",
                ms=6,
                mfc="white",
                mec=COLORS[m],
                mew=1.5,
            )
        ref_lr = entry["fixed"][ref]["config"]["lr"]
        ax.axvline(ref_lr, color=INK2, lw=0.8, ls=":")
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        title = LABELS[m] + (f" (decay {d:.4g})" if d is not None else "")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("learning rate", fontsize=9, color=INK2)
    axes[0].set_ylabel(r"$N(\varphi{=}45^\circ)\,/\,N(\varphi{=}0^\circ)$", fontsize=9, color=INK2)
    fig.suptitle(
        "quadratic_ill: slowdown from full misalignment, at each learning rate that settles "
        "at φ = 0\n"
        "dotted: lr tuned at φ = 0;  open triangle: does not settle at φ = 45° within N_max, "
        "so the ratio exceeds N_max / N(φ = 0)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path)


def block_max(ks: list[int], fs: list[float], points: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a trace for plotting by the maximum of f over blocks of iterations, so
    that every excursion above the target stays visible (NaN counts as +∞)."""
    f = np.nan_to_num(np.asarray(fs, dtype=np.float64), nan=np.inf)
    b = max(1, math.ceil(len(f) / points))
    n = math.ceil(len(f) / b) * b
    pad = np.concatenate([f, np.full(n - len(f), -np.inf)])
    k = np.asarray(ks)[::b]
    return k, pad.reshape(-1, b).max(axis=1)  # (n/b,)


def figure_convergence(summary: dict[str, Any], path: Path) -> None:
    from matplotlib.figure import Figure

    shown = ("gd", "adam", "adabelief", "lion", "adan", "sophia")
    fig = Figure(figsize=(12, 5.4), layout="constrained", facecolor="white")
    axes = fig.subplots(2, len(shown))
    for row, pid in enumerate(("quadratic_ill", "rosenbrock")):
        pr = summary[pid]
        ref = pr["ref_index"]
        worst = (
            int(np.argmax(pr["phi_deg"])) if pid == "quadratic_ill" else len(pr["thetas_deg"]) - 1
        )
        for col, m in enumerate(shown):
            ax = axes[row, col]
            _style(ax)
            cfg = pr["methods"][m]["fixed"][ref]["config"]
            for a, colour, lab in ((ref, "#2a78d6", "reference"), (worst, "#eb6834", "rotated")):
                res, _ = run_one(pid, pr["thetas_deg"][a], m, cfg, 0, pr["certify_horizon"])
                k, fv = block_max(
                    [s.k for s in res.trace],
                    [math.inf if s.fun is None else float(s.fun) for s in res.trace],
                )
                ax.plot(k, np.maximum(fv, 1e-300), color=colour, lw=1.0, label=lab)
            ax.axhline(TARGET, color=INK2, lw=0.8, ls=":")
            ax.axvline(pr["n_max"], color=INK2, lw=0.6, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 1e4)
            if row == 0:
                ax.set_title(LABELS[m], fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(
                    ("quadratic_ill" if pid == "quadratic_ill" else "Rosenbrock")
                    + r"   $f(x_k)-f^*$",
                    fontsize=9,
                    color=INK2,
                )
            if row == 1:
                ax.set_xlabel("iteration k", fontsize=9, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        "Start 0, hyperparameters fixed at their reference-tuned values. Blue: reference "
        "orientation (quadratic φ = 0°, Rosenbrock θ = 0°); orange: φ = 45° / θ = 45°.\n"
        "Dotted: target 1e-6; dashed: N_max. Curves: max of f over blocks of iterations, "
        "so every excursion above the target stays visible.",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path)


# Profile line styles: one colour per family, the variants dashed or dotted.
PROFILE_STYLE = {
    "gd": (COLORS["gd"], "-"),
    "momentum": (COLORS["heavy_ball"], ":"),
    "heavy_ball": (COLORS["heavy_ball"], "-"),
    "adam": (COLORS["adam"], "-"),
    "adamw": (COLORS["adam"], "--"),
    "amsgrad": (COLORS["adam"], ":"),
    "nadam": (COLORS["adam"], "-."),
    "rmsprop": (COLORS["gd"], ":"),
    "adabelief": (COLORS["adabelief"], "-"),
    "adan": (COLORS["adan"], "-"),
    "sophia": (COLORS["sophia"], "-"),
    "sophia_h": (COLORS["sophia"], "--"),
    "lion": (COLORS["lion"], "-"),
}


def plot_profile(
    plot: Callable[..., Any], profile: Any, methods: list[str], title: str, path: Path
) -> None:
    """numopt.bench's profile plot on a wider axes, restyled per family, legend outside."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 4.6), layout="constrained", facecolor="white")
    ax = fig.add_subplot()
    plot(profile, None, ax=ax, title=title)
    by_label = {LABELS[m]: m for m in methods}
    for line in ax.get_lines():
        m = by_label.get(str(line.get_label()))
        if m is not None:
            colour, ls = PROFILE_STYLE[m]
            line.set_color(colour)
            line.set_linestyle(cast(Any, ls))
            line.set_linewidth(1.8)
    _style(ax)
    ax.title.set_fontsize(10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.savefig(path)


def figures_profiles(summary: dict[str, Any]) -> dict[str, Any]:
    labels = list(GRIDS["rosenbrock"])  # every method runs on every instance
    rows = []
    for pid in PROBLEMS:
        pr = summary[pid]
        for a in range(len(pr["thetas_deg"])):
            for s in range(len(PROBLEMS[pid]["starts"])):
                row = []
                for m in labels:
                    v = pr["methods"][m]["tuned"][a]["per_start"][s]
                    row.append(math.inf if v is None else max(float(v), 1.0))
                rows.append(row)
    names = [LABELS[m] for m in labels]
    perf = bench.performance_profile_from_costs(rows, names)
    data = bench.data_profile_from_costs(rows, [2] * len(rows), names)
    title = f"{len(rows)} instances (problem × angle × start), settling iterations, tuned per angle"
    plot_profile(
        bench.plot_performance_profile,
        perf,
        labels,
        "Performance profile: " + title,
        FIGURES / "performance_profile.svg",
    )
    plot_profile(
        bench.plot_data_profile,
        data,
        labels,
        "Data profile: " + title,
        FIGURES / "data_profile.svg",
    )
    best = np.min(np.array(rows), axis=1)
    winners: dict[str, dict[str, int]] = {}
    i = 0
    for pid in PROBLEMS:
        count = {n: 0 for n in names}
        for _ in range(len(summary[pid]["thetas_deg"]) * len(PROBLEMS[pid]["starts"])):
            for j, n in enumerate(names):
                if math.isfinite(best[i]) and rows[i][j] == best[i]:
                    count[n] += 1
            i += 1
        winners[pid] = {n: c for n, c in count.items() if c}
    return {
        "labels": names,
        "instances": len(rows),
        "rho_at_1": dict(zip(names, perf.at(1.0).tolist(), strict=True)),
        "rho_at_2": dict(zip(names, perf.at(2.0).tolist(), strict=True)),
        "solved": dict(zip(names, perf.solved.tolist(), strict=True)),
        "fastest_or_tied_by_problem": winners,
    }


# --------------------------------------------------------------------------------------


def n_array(raw: dict[Key, dict[str, Any]], pid: str, m: str) -> tuple[np.ndarray, np.ndarray]:
    """N[angle, config, start] (∞ = unsolved) and the horizon each value was measured at."""
    spec = PROBLEMS[pid]
    shape = (len(spec["thetas_deg"]), len(GRIDS[pid][m]), len(spec["starts"]))
    N = np.full(shape, np.inf)
    H = np.zeros(shape, dtype=np.int64)
    for a, c, s in np.ndindex(*shape):
        r = raw[(pid, a, m, c, s)]
        N[a, c, s] = math.inf if r["N"] is None else r["N"]
        H[a, c, s] = r["horizon"]
    return N, H


def certification_todo(raw: dict[Key, dict[str, Any]]) -> list[tuple[str, str, int, int]]:
    """(problem, method, angle, config) whose runs must be repeated at the certify horizon.

    The settling iteration can only grow with the horizon (the trajectory prefix is the
    same, and a later excursion above the target moves N later or past N_max), so every
    sweep value is a lower bound on the certified one, and so is the selection key of
    choose(). Hence: if the config chosen from the current values is certified, it is the
    exact choice at the certify horizon. Otherwise certify it and choose again. Configs
    with no solved start need no certification (they cannot improve).
    """
    todo = []
    for pid, spec in PROBLEMS.items():
        # A sweep run that is unsolved stays unsolved at a longer horizon only if the sweep
        # horizon exceeds n_max (then an N found later would exceed n_max as well).
        assert spec["horizon"] > spec["n_max"]
        hc = spec["certify_horizon"]
        ref = int(np.argmin([abs(t - spec["ref_deg"]) for t in spec["thetas_deg"]]))
        for m in GRIDS[pid]:
            N, H = n_array(raw, pid, m)
            done = (H >= hc).all(axis=2)  # (angle, config)
            need = np.isfinite(N).any(axis=2) & ~done
            tuned = [choose(N[a]) for a in range(N.shape[0])]
            for a, c in enumerate(tuned):
                if need[a, c]:
                    todo.append((pid, m, a, c))
            c_ref = tuned[ref]
            if done[ref, c_ref]:  # the fixed protocol keeps the (final) reference choice
                todo.extend((pid, m, a, c_ref) for a in range(N.shape[0]) if need[a, c_ref])
    return sorted(set(todo))


def certify(raw: dict[Key, dict[str, Any]], pool: Any) -> dict[str, Any]:
    """Repeat the selected configs at the certify horizon until every selection is certified."""
    rounds, runs, changed = 0, 0, 0
    while todo := certification_todo(raw):
        rounds += 1
        tasks = [
            (pid, a, m, c, s, PROBLEMS[pid]["certify_horizon"])
            for pid, m, a, c in todo
            for s in range(len(PROBLEMS[pid]["starts"]))
        ]
        for key, rec in pool.imap_unordered(_task, tasks, chunksize=1):
            rec["N_sweep"] = raw[key]["N"]
            rec["cpu"] += raw[key]["cpu"]
            changed += rec["N"] != rec["N_sweep"]
            raw[key] = rec
        runs += len(tasks)
    return {"rounds": rounds, "runs": runs, "runs_whose_N_changed": changed}


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    tasks = all_tasks()
    workers = min(os.cpu_count() or 1, 20)
    with get_context("fork").Pool(workers) as pool:
        raw = dict(pool.imap_unordered(_task, tasks, chunksize=4))
        certification = certify(raw, pool)
    t_runs = time.time() - t0
    summary = analyse(raw)
    summary["profiles"] = figures_profiles(summary)
    figure_angles(summary, FIGURES / "iterations_vs_angle.svg")
    figure_sensitivity(summary, FIGURES / "sensitivity_vs_lr.svg")
    figure_convergence(summary, FIGURES / "convergence.svg")
    summary["meta"] = {
        "target": TARGET,
        "metric": "settling iteration N (f < target from N to the horizon), solved iff N <= n_max;"
        " every reported selection is measured at certify_horizon",
        "gtol": GTOL,
        "overshoot_threshold": OVERSHOOT,
        "runs": len(tasks),
        "certification": certification,
        "workers": workers,
        "seconds_runs": round(t_runs, 1),
        "seconds_total": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "problems": {
            pid: {k: v for k, v in spec.items() if k != "starts"} | {"starts": spec["starts"]}
            for pid, spec in PROBLEMS.items()
        },
    }
    cpu: dict[str, float] = {}
    for key, val in raw.items():
        label = f"{key[0]}/{key[2]}"
        cpu[label] = round(cpu.get(label, 0.0) + val["cpu"], 1)
    summary["meta"]["cpu_seconds_by_problem_method"] = cpu
    # Raw counts N[angle][config][start] (null = did not settle by n_max) go to their own file.
    iterations = {
        pid: {m: {"grid": e["grid"], "N": e.pop("N")} for m, e in summary[pid]["methods"].items()}
        for pid in PROBLEMS
    }
    with open(RESULTS / "iterations.json", "w") as fh:
        json.dump(iterations, fh, allow_nan=False)
    with open(RESULTS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, allow_nan=False)
    print(f"{len(tasks)} runs in {t_runs:.1f} s; total {time.time() - t0:.1f} s")
    for pid in PROBLEMS:
        pr = summary[pid]
        print(f"\n{pid}: angles {[round(t, 2) for t in pr['thetas_deg']]}")
        for m, e in pr["methods"].items():
            t = [r["median"] for r in e["tuned"]]
            f = [r["median"] for r in e["fixed"]]
            print(f"  {m:10s} tuned {t}\n  {'':10s} fixed {f}  edge={e['tuned_best_on_grid_edge']}")


if __name__ == "__main__":
    main()
