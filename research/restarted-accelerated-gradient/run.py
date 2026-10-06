"""Experiments: accelerated gradient with adaptive restart (AGD / FISTA) vs. numopt baselines.

Stage 1 — random SPD quadratics f(x) = ½(x − x*)ᵀA(x − x*), n = 50, κ ∈ {10, 10², 10³, 10⁴},
5 instances per κ. Iterations and cost to reach f(x_k) − f* < 1e-8.

Stage 2 — lasso F(x) = ½‖Ax − b‖² + λ‖x‖₁: a 2-D instance (trajectories) and 15 sparse
regression instances with n = 50 (5 data seeds × 3 values of λ). Support identification.

Run:  .venv/bin/python research/restarted-accelerated-gradient/run.py
Writes results/*.json and figures/*.svg. Deterministic (numopt.core.rng.Rng, fixed seeds).
"""

from __future__ import annotations

import os

# Single-threaded BLAS: on a loaded machine a multithreaded BLAS made a 200×200 solve ~2500×
# slower (thread oversubscription); every array here is small.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from method import adaptive_proxgd, fista, ista, make_lasso  # noqa: E402
from reference import lasso_solution  # noqa: E402

from numopt import bench  # noqa: E402
from numopt.core.registry import run as run_method  # noqa: E402
from numopt.core.rng import Rng  # noqa: E402
from numopt.core.types import Problem, Result, to_jsonable  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Categorical palette (dataviz reference palette, light mode; validated: CVD ΔE ≥ 9.1 on
# adjacent pairs). Colors follow the method, never its rank; two extra neutrals for CG/L-BFGS.
C = {
    "AGD-GR": "#2a78d6",
    "AGD-FR": "#eb6834",
    "AGD": "#1baf7a",
    "AGD-GR-BT": "#eda100",
    "AdGD": "#e87ba4",
    "nesterov": "#008300",
    "momentum": "#4a3aa7",
    "GD": "#e34948",
    "GD-opt": "#e34948",  # same hue as GD (same method, other step), solid line
    "CG-PR": "#3d3d3a",
    "L-BFGS": "#8a887f",
}
LS = {
    "AGD-GR": "-",
    "AGD-FR": "--",
    "AGD": ":",
    "AGD-GR-BT": "-.",
    "AdGD": "--",
    "nesterov": "-",
    "momentum": "-.",
    "GD": ":",
    "GD-opt": "-",
    "CG-PR": "-",
    "L-BFGS": "--",
}
INK, MUTED, GRID = "#1a1a19", "#5f5e5a", "#e4e3dc"


def save(fig: Any, name: str) -> None:
    """Write figures/<name>.svg (for the README) and a .png copy."""
    fig.savefig(FIGURES / f"{name}.svg")
    fig.savefig(FIGURES / f"{name}.png", dpi=130)
    plt.close(fig)


def style_axes(ax: Any) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


# ======================================================================================
# Random data from numopt's portable generator
# ======================================================================================


def normals(rng: Rng, shape: tuple[int, ...]) -> np.ndarray:
    """Standard normals, row-major draw order (Box–Muller, one variate per call)."""
    size = int(np.prod(shape))
    return np.array([rng.normal() for _ in range(size)]).reshape(shape)


def random_orthogonal(rng: Rng, n: int) -> np.ndarray:
    """Q from the QR factorization of a Gaussian matrix, columns sign-fixed (Haar measure)."""
    Q, R = np.linalg.qr(normals(rng, (n, n)))
    return Q * np.sign(np.diag(R))


# ======================================================================================
# Stage 1: quadratics
# ======================================================================================

N_QUAD = 50
KAPPAS = (1e1, 1e2, 1e3, 1e4)
N_INST = 5
EPS = 1e-8
# ‖∇f‖ ≤ GTOL ⇒ f − f* ≤ ‖∇f‖²/(2λ_min) < EPS (λ_min = 1), so a method that stops on its
# gradient test has always crossed the target first.
GTOL = 0.999 * math.sqrt(2.0 * EPS)
MAX_ITER = 400_000


def make_quadratic(kappa: float, inst: int) -> Problem:
    """A = Q diag(λ) Qᵀ with λ_i = κ^{i/(n−1)} (log-spaced, λ_min = 1, λ_max = κ exactly)."""
    seed = 10_000 + 100 * round(math.log10(kappa)) + inst
    rng = Rng(seed)
    n = N_QUAD
    Q = random_orthogonal(rng, n)
    lam = kappa ** (np.arange(n) / (n - 1))
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    x_star = normals(rng, (n,))
    x0 = normals(rng, (n,))

    def f(x: np.ndarray) -> float:
        d = x - x_star
        return 0.5 * float(d @ (A @ d))

    return Problem(
        id=f"quad_k{int(kappa)}_{inst}",
        name=f"Random SPD quadratic, κ = {kappa:g}, instance {inst}",
        latex=r"\tfrac12 (x-x^*)^\top A (x-x^*)",
        f=f,
        dim=n,
        domain=(),
        grad=lambda x: A @ (x - x_star),
        hess=lambda x: A,
        x0=x0,
        minima=(x_star,),
        extra={"f_min": 0.0, "L": float(kappa), "mu": 1.0, "seed": seed},
    )


class Recorder:
    """Wrap f and ∇f; remember (n_fev, n_gev) at the first f evaluation with f − f* < EPS."""

    def __init__(self, problem: Problem, f_star: float) -> None:
        self.nf = 0
        self.ng = 0
        self.hit: tuple[int, int] | None = None
        self.f_star = f_star
        f, g = problem.f, problem.grad
        assert g is not None

        def wf(x: Any) -> float:
            self.nf += 1
            v = float(f(x))
            if self.hit is None and v - self.f_star < EPS:
                self.hit = (self.nf, self.ng)
            return v

        def wg(x: Any) -> Any:
            self.ng += 1
            return g(x)

        self.problem = dataclasses.replace(problem, f=wf, grad=wg)


def quad_methods(kappa: float) -> list[tuple[str, Callable[..., Result], dict[str, Any]]]:
    L, mu = kappa, 1.0
    q = math.sqrt(kappa)
    common = {"gtol": GTOL, "max_iter": MAX_ITER}
    reg = lambda mid: lambda P, **kw: run_method(mid, P, **kw)  # noqa: E731
    return [
        ("AGD-GR", fista, {"restart": "gradient", "lr": 1 / L, "backtracking": False, **common}),
        ("AGD-FR", fista, {"restart": "function", "lr": 1 / L, "backtracking": False, **common}),
        ("AGD", fista, {"restart": "none", "lr": 1 / L, "backtracking": False, **common}),
        ("AGD-GR-BT", fista, {"restart": "gradient", "lr": 1.0, "backtracking": True, **common}),
        ("AdGD", adaptive_proxgd, {"lr": 1 / L, **common}),
        ("nesterov", reg("nesterov"), {"lr": 1 / L, "beta": (q - 1) / (q + 1), **common}),
        (
            "momentum",
            reg("momentum"),
            {
                "lr": 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2,
                "beta": ((q - 1) / (q + 1)) ** 2,
                **common,
            },
        ),
        # NOTE: GD uses step 1/L, the step of ISTA/FISTA with g ≡ 0 (the like-for-like
        # baseline). GD-opt uses the optimal fixed step 2/(L + μ) for a quadratic (true κ).
        ("GD", reg("gradient_descent"), {"step_rule": "fixed", "lr": 1 / L, **common}),
        ("GD-opt", reg("gradient_descent"), {"step_rule": "fixed", "lr": 2 / (L + mu), **common}),
        # NOTE: these two test ‖∇f‖∞ ≤ gtol; ‖∇f‖₂ ≤ √n‖∇f‖∞, so gtol/√n keeps the guarantee.
        ("CG-PR", reg("cg_polak_ribiere"), {**common, "gtol": GTOL / math.sqrt(N_QUAD)}),
        ("L-BFGS", reg("lbfgs"), {**common, "gtol": GTOL / math.sqrt(N_QUAD)}),
    ]


def fun(s: Any) -> float:
    """Step.fun as a float (every method here sets it at every step)."""
    assert s.fun is not None
    return float(s.fun)


def first_below(values: list[float], target: float) -> int | None:
    for k, v in enumerate(values):
        if v < target:
            return k
    return None


def stage1() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    for kappa in KAPPAS:
        for inst in range(N_INST):
            P = make_quadratic(kappa, inst)
            assert P.x0 is not None
            f0 = float(P.f(P.x0))
            for label, fn, params in quad_methods(kappa):
                rec = Recorder(P, 0.0)
                t0 = time.perf_counter()
                res = fn(rec.problem, **params)
                dt = time.perf_counter() - t0
                fvals = [fun(s) for s in res.trace]
                k_hit = first_below(fvals, EPS)
                ngev = None if rec.hit is None else rec.hit[1]
                ngev_charged = ngev
                if label == "nesterov" and ngev is not None and k_hit is not None:
                    # NOTE: numopt's nesterov evaluates ∇f at the look-ahead y_{k−1} AND at x_k
                    # (the latter only for its stopping test): K iterates cost ∇f(x_0..x_{K−1})
                    # + ∇f(y_1..y_{K−1}) = 2K − 1. The method itself (Sutskever et al. eq. 3–4)
                    # needs only ∇f(y_0..y_{K−1}) = K, the same count as FISTA/AGD, which tests
                    # convergence at y_k. We charge K; the raw count is kept in ngev_at_target.
                    assert ngev == 2 * k_hit - 1, (ngev, k_hit)
                    ngev_charged = k_hit
                rows.append(
                    {
                        "kappa": kappa,
                        "instance": inst,
                        "method": label,
                        "iters": k_hit,
                        "nfev_at_target": None if rec.hit is None else rec.hit[0],
                        "ngev_at_target": ngev,
                        "ngev_charged": ngev_charged,
                        "f0": f0,
                        "n_iter_total": res.n_iter,
                        "converged": res.converged,
                        "message": res.message,
                        "f_final": res.fun,
                        "n_restart": res.extra.get("n_restart"),
                        "seconds": dt,
                    }
                )
                if kappa == KAPPAS[-1] and inst == 0:
                    curves[label] = {
                        "f": fvals,
                        "restarts": res.extra.get("restarts", []),
                    }
                del res
        print(f"  stage 1: κ = {kappa:g} done", flush=True)
    return {"rows": rows, "curves": curves}


def fit_slope(ks: np.ndarray, its: np.ndarray) -> dict[str, float]:
    """OLS fit log10(iters) = a + p·log10(κ); p with its standard error."""
    X = np.column_stack([np.ones_like(ks), np.log10(ks)])
    y = np.log10(its)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)  # 2×2, well conditioned (κ grid 10..1e4)
    return {
        "slope": float(coef[1]),
        "slope_se": float(math.sqrt(cov[1, 1])),
        "intercept": float(coef[0]),
    }


def bootstrap_slope(ks: np.ndarray, its: np.ndarray, n_boot: int = 2000) -> list[float]:
    """95 % percentile interval of the OLS slope, resampling instances within each κ."""
    rng = Rng(777)
    groups = [np.flatnonzero(ks == k) for k in KAPPAS]
    slopes = []
    for _ in range(n_boot):
        idx = np.concatenate([g[[rng.integers(g.size) for _ in range(g.size)]] for g in groups])
        slopes.append(fit_slope(ks[idx], its[idx])["slope"])
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def summarize_stage1(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["rows"]
    labels = [m[0] for m in quad_methods(10.0)]
    summary: dict[str, Any] = {
        "median_iters": {},
        "median_ngev": {},
        "median_ngev_charged": {},
        "solved": {},
        "slopes": {},
    }
    for lab in labels:
        r = [x for x in rows if x["method"] == lab]
        summary["solved"][lab] = sum(x["iters"] is not None for x in r)
        med_it, med_g, med_gc = {}, {}, {}
        for kappa in KAPPAS:
            rk = [x for x in r if x["kappa"] == kappa and x["iters"] is not None]
            med_it[f"{kappa:g}"] = float(np.median([x["iters"] for x in rk])) if rk else None
            med_g[f"{kappa:g}"] = (
                float(np.median([x["ngev_at_target"] for x in rk])) if rk else None
            )
            med_gc[f"{kappa:g}"] = float(np.median([x["ngev_charged"] for x in rk])) if rk else None
        summary["median_iters"][lab] = med_it
        summary["median_ngev"][lab] = med_g
        summary["median_ngev_charged"][lab] = med_gc
        ok = [x for x in r if x["iters"] is not None]
        if len(ok) == len(r) and len(ok) >= 3:
            ks = np.array([x["kappa"] for x in ok])
            its = np.array([max(x["iters"], 1) for x in ok], float)
            sl: dict[str, Any] = dict(fit_slope(ks, its))
            # The target is absolute while f(x0) grows with κ (log-spaced spectrum, x0 − x*
            # standard normal), so a linear method needs ∝ √κ·log(f(x0)/ε) iterations. Dividing
            # by log(f(x0)/ε) removes that factor from the fitted exponent.
            log_ratio = np.array([math.log(x["f0"] / EPS) for x in ok])
            sl_norm = fit_slope(ks, its / log_ratio)
            med = np.array([float(np.median(its[ks == k])) for k in KAPPAS])
            sl["slope_f0_normalized"] = sl_norm["slope"]
            sl["slope_f0_normalized_se"] = sl_norm["slope_se"]
            sl["slope_medians_only"] = fit_slope(np.array(KAPPAS), med)["slope"]
            sl["slope_bootstrap95"] = bootstrap_slope(ks, its)
            summary["slopes"][lab] = sl
    summary["median_f0"] = {
        f"{k:g}": float(
            np.median([x["f0"] for x in rows if x["kappa"] == k and x["method"] == labels[0]])
        )
        for k in KAPPAS
    }
    # Per-instance ratio AGD-* / nesterov (iterations and gradient evaluations). "ngev" uses
    # the charged count (one ∇f per nesterov iteration, see stage1); "ngev_raw" uses numopt's
    # actual count, which includes the extra ∇f(x_k) of its stopping test.
    ratios: dict[str, Any] = {}
    for other in ("AGD-GR", "AGD-FR", "AGD-GR-BT"):
        it_r, g_r, g_raw = [], [], []
        for kappa in KAPPAS:
            for inst in range(N_INST):
                a = next(
                    x
                    for x in rows
                    if x["method"] == other and x["kappa"] == kappa and x["instance"] == inst
                )
                b = next(
                    x
                    for x in rows
                    if x["method"] == "nesterov" and x["kappa"] == kappa and x["instance"] == inst
                )
                it_r.append((kappa, a["iters"] / b["iters"]))
                g_r.append((kappa, a["ngev_charged"] / b["ngev_charged"]))
                g_raw.append(a["ngev_at_target"] / b["ngev_at_target"])
        ratios[other] = {
            "iters_ratio_max": max(r for _, r in it_r),
            "iters_ratio_median": float(np.median([r for _, r in it_r])),
            "iters_ratio_by_kappa_max": {
                f"{k:g}": max(r for kk, r in it_r if kk == k) for k in KAPPAS
            },
            "ngev_ratio_max": max(r for _, r in g_r),
            "ngev_ratio_median": float(np.median([r for _, r in g_r])),
            "ngev_raw_ratio_max": max(g_raw),
            "ngev_raw_ratio_median": float(np.median(g_raw)),
        }
    summary["ratio_vs_nesterov"] = ratios
    # Restarts per run (AGD-GR) by κ.
    summary["restarts_median"] = {
        lab: {
            f"{k:g}": float(
                np.median([x["n_restart"] for x in rows if x["method"] == lab and x["kappa"] == k])
            )
            for k in KAPPAS
        }
        for lab in ("AGD-GR", "AGD-FR", "AGD-GR-BT")
    }
    return summary


def stage1_profile(
    data: dict[str, Any], *, charged: bool = True
) -> tuple[bench.Profile, list[str]]:
    """Dolan–Moré profile of cost n_fev + n·n_gev to f − f* < EPS (Moré–Wild cost model).

    charged=True counts one ∇f per `nesterov` iteration (the method's own cost; see stage1);
    charged=False uses numopt's raw count, which doubles it.
    """
    rows = data["rows"]
    labels = [
        "AGD-GR",
        "AGD-GR-BT",
        "AGD-FR",
        "AdGD",
        "nesterov",
        "momentum",
        "GD",
        "GD-opt",
        "CG-PR",
        "L-BFGS",
    ]
    key = "ngev_charged" if charged else "ngev_at_target"
    T = np.full((len(KAPPAS) * N_INST, len(labels)), np.inf)
    for p, (kappa, inst) in enumerate((k, i) for k in KAPPAS for i in range(N_INST)):
        for s, lab in enumerate(labels):
            x = next(
                r
                for r in rows
                if r["method"] == lab and r["kappa"] == kappa and r["instance"] == inst
            )
            if x["nfev_at_target"] is not None:
                T[p, s] = x["nfev_at_target"] + N_QUAD * x[key]
    return bench.performance_profile_from_costs(T, labels), labels


# Start values L₀ for the backtracking sensitivity sweep. Beck–Teboulle backtracking only
# increases L (L_k = ηⁱL_{k−1}), so L₀ > L is never corrected, and with η = 2 the accepted
# L_k = 2^j·L₀ depends on where the powers of two from L₀ fall relative to L. √2 shifts that
# phase by half an octave relative to L₀ = 1; "3L" is a start above L.
BT_L0 = ("1e-4", "1e-2", "1", "sqrt2", "1e2", "3L")


def bt_l0_value(tag: str, L: float) -> float:
    return {"sqrt2": math.sqrt(2.0), "3L": 3.0 * L}.get(tag) or float(tag)


def stage1_bt_sweep(data: dict[str, Any]) -> dict[str, Any]:
    """AGD-GR-BT (gradient restart, Beck–Teboulle backtracking, η = 2) for each L₀ in BT_L0."""
    rows = data["rows"]
    out: list[dict[str, Any]] = []
    for kappa in KAPPAS:
        for inst in range(N_INST):
            P = make_quadratic(kappa, inst)
            nest = next(
                r
                for r in rows
                if r["method"] == "nesterov" and r["kappa"] == kappa and r["instance"] == inst
            )
            for tag in BT_L0:
                L0 = bt_l0_value(tag, kappa)
                rec = Recorder(P, 0.0)
                res = fista(
                    rec.problem,
                    restart="gradient",
                    lr=1.0 / L0,
                    backtracking=True,
                    gtol=GTOL,
                    max_iter=MAX_ITER,
                )
                k_hit = first_below([fun(s) for s in res.trace], EPS)
                assert k_hit is not None and rec.hit is not None
                out.append(
                    {
                        "kappa": kappa,
                        "instance": inst,
                        "L0": tag,
                        "L0_value": L0,
                        "iters": k_hit,
                        "nfev_at_target": rec.hit[0],
                        "ngev_at_target": rec.hit[1],
                        "final_L_over_L": float(res.trace[-1].info["L"]) / kappa,
                        "ratio_vs_nesterov": k_hit / nest["iters"],
                    }
                )
        print(f"  stage 1 (L0 sweep): κ = {kappa:g} done", flush=True)
    summ: dict[str, Any] = {}
    for tag in BT_L0:
        rr = [r for r in out if r["L0"] == tag]
        summ[tag] = {
            "median_iters": {
                f"{k:g}": float(np.median([r["iters"] for r in rr if r["kappa"] == k]))
                for k in KAPPAS
            },
            "median_final_L_over_L": {
                f"{k:g}": float(np.median([r["final_L_over_L"] for r in rr if r["kappa"] == k]))
                for k in KAPPAS
            },
            "ratio_vs_nesterov_max": max(r["ratio_vs_nesterov"] for r in rr),
            "ratio_vs_nesterov_median": float(np.median([r["ratio_vs_nesterov"] for r in rr])),
            "slope": fit_slope(
                np.array([r["kappa"] for r in rr]), np.array([r["iters"] for r in rr], float)
            )["slope"],
        }
    below = [r for r in out if r["L0_value"] <= r["kappa"]]
    summ["L0_le_L"] = {
        "n_runs": len(below),
        "ratio_vs_nesterov_max": max(r["ratio_vs_nesterov"] for r in below),
        "ratio_vs_nesterov_median": float(np.median([r["ratio_vs_nesterov"] for r in below])),
        "final_L_over_L_min": min(r["final_L_over_L"] for r in below),
        "final_L_over_L_max": max(r["final_L_over_L"] for r in below),
    }
    return {"rows": out, "summary": summ}


def plot_stage1(data: dict[str, Any], summary: dict[str, Any], profile: bench.Profile) -> None:
    rows = data["rows"]
    # --- Figure 1: scaling of iterations with κ --------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    shown = [
        "GD",
        "GD-opt",
        "AGD",
        "AdGD",
        "AGD-FR",
        "AGD-GR",
        "AGD-GR-BT",
        "nesterov",
        "momentum",
    ]
    ks = np.array(KAPPAS)
    for lab in shown:
        pts = [
            (x["kappa"], x["iters"]) for x in rows if x["method"] == lab and x["iters"] is not None
        ]
        kk = np.array([p[0] for p in pts]) * (1 + 0.04 * (shown.index(lab) - 4))  # jitter
        ax.scatter(kk, [p[1] for p in pts], s=10, color=C[lab], alpha=0.45, linewidths=0)
        med = [summary["median_iters"][lab][f"{k:g}"] for k in KAPPAS]
        sl = summary["slopes"].get(lab)
        tag = f"{lab}  (slope {sl['slope']:.2f})" if sl else lab
        ax.plot(
            ks,
            med,
            color=C[lab],
            linestyle=LS[lab],
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=tag,
        )
    base = summary["median_iters"]["nesterov"]["10"]
    for p, txt in ((0.5, r"$\propto\kappa^{1/2}$"), (1.0, r"$\propto\kappa$")):
        ref = base * 0.5 * (ks / 10.0) ** p
        ax.plot(ks, ref, color=MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.annotate(
            txt,
            (ks[-1], ref[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=MUTED,
            va="center",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"condition number $\kappa = L/\mu$")
    ax.set_ylabel(r"iterations to $f(x_k)-f^* < 10^{-8}$")
    ax.set_title(
        "Random SPD quadratics, n = 50 (5 instances per κ; line = median)", fontsize=10, color=INK
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    save(fig, "quad_scaling")

    # --- Figure 2: convergence curves at κ = 1e4 -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for lab in ["GD", "AGD", "AdGD", "AGD-GR", "nesterov", "momentum"]:  # AGD-FR ≈ AGD-GR here
        f = np.maximum(np.array(data["curves"][lab]["f"]), 1e-17)
        k = np.arange(f.size)
        ax.plot(k[1:], f[1:], color=C[lab], linestyle=LS[lab], linewidth=1.5, label=lab)
    rs = data["curves"]["AGD-GR"]["restarts"]
    fg = np.array(data["curves"]["AGD-GR"]["f"])
    ax.scatter(
        rs,
        np.maximum(fg[rs], 1e-17),
        s=16,
        marker="v",
        color=C["AGD-GR"],
        zorder=5,
        label="AGD-GR restart",
    )
    ax.axhline(EPS, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(
        "target $10^{-8}$",
        (1.2, EPS),
        xytext=(0, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 2e5)
    ax.set_ylim(1e-12, None)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$f(x_k) - f^*$")
    ax.set_title(r"One instance with $\kappa = 10^4$ (n = 50)", fontsize=10, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=2)
    save(fig, "quad_curves")

    # --- Figure 3: performance profile ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for j, lab in enumerate(profile.labels):
        ax.step(
            profile.x,
            profile.y[j],
            where="post",
            color=C[lab],
            linestyle=LS[lab],
            linewidth=1.6,
            label="nesterov (1 ∇f / iter)" if lab == "nesterov" else lab,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, float(profile.x[-1]))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"performance ratio $\alpha$ (cost = n_fev + n·n_gev to $f-f^*<10^{-8}$)")
    ax.set_ylabel(r"$\rho_s(\alpha)$")
    ax.set_title("Performance profile, 20 quadratic instances", fontsize=10, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncols=2)
    save(fig, "quad_profile")


# ======================================================================================
# Stage 2: lasso
# ======================================================================================

PROX_LABELS = ("ISTA", "FISTA", "FISTA-FR", "FISTA-GR", "AdProxGD")
PC = {  # same hues as the corresponding stage-1 methods
    "ISTA": C["GD"],
    "FISTA": C["AGD"],
    "FISTA-FR": C["AGD-FR"],
    "FISTA-GR": C["AGD-GR"],
    "AdProxGD": C["AdGD"],
    "PG (l1 ball)": C["momentum"],
    "FW (l1 ball)": C["CG-PR"],
}
PLS = {
    "ISTA": ":",
    "FISTA": ":",
    "FISTA-FR": "--",
    "FISTA-GR": "-",
    "AdProxGD": "--",
    "PG (l1 ball)": "-.",
    "FW (l1 ball)": "-",
}


def prox_methods(
    L: float, gtol: float, max_iter: int
) -> list[tuple[str, Callable[..., Result], dict[str, Any]]]:
    fixed = {"lr": 1 / L, "backtracking": False, "gtol": gtol, "max_iter": max_iter}
    return [
        ("ISTA", ista, fixed),
        ("FISTA", fista, {"restart": "none", **fixed}),
        ("FISTA-FR", fista, {"restart": "function", **fixed}),
        ("FISTA-GR", fista, {"restart": "gradient", **fixed}),
        ("AdProxGD", adaptive_proxgd, {"lr": 1 / L, "gtol": gtol, "max_iter": max_iter}),
    ]


def support_stats(xs: list[np.ndarray], S_star: frozenset[int]) -> dict[str, Any]:
    """first: first k with supp(x_k) = S*; final: first k after which it never changes again
    (None if the last iterate's support differs); changes: support changes after `first`;
    err: |supp(x_k) Δ S*| for every k."""
    err = [len(frozenset(np.flatnonzero(x).tolist()) ^ S_star) for x in xs]
    first = next((k for k, e in enumerate(err) if e == 0), None)
    final = None
    if err[-1] == 0:
        final = len(err) - 1
        while final > 0 and err[final - 1] == 0:
            final -= 1
    supp = [frozenset(np.flatnonzero(x).tolist()) for x in xs]
    changes = (
        0 if first is None else sum(supp[k] != supp[k - 1] for k in range(first + 1, len(supp)))
    )
    return {"first": first, "final": final, "changes_after_first": changes, "err": err}


def l1_ball_problem_2d(A: np.ndarray, b: np.ndarray, r: float, x0: np.ndarray) -> Problem:
    """min ½‖Ax − b‖² s.t. ‖x‖₁ ≤ r in 2-D: the polyhedron |x₁| + |x₂| ≤ r (4 facets)."""
    A_ub = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    V = r * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return Problem(
        id="l1ball_2d",
        name="l1-ball least squares (2-D)",
        latex=r"\min \tfrac12\|Ax-b\|^2,\ \|x\|_1\le r",
        f=lambda x: 0.5 * float((A @ x - b) @ (A @ x - b)),
        dim=2,
        domain=(),
        grad=lambda x: A.T @ (A @ x - b),
        x0=x0,
        extra={"projection": "polyhedron", "A_ub": A_ub, "b_ub": np.full(4, r), "vertices": V},
    )


def l1_ball_problem_lifted(A: np.ndarray, b: np.ndarray, r: float) -> Problem:
    """The same problem in z = (u, v) ∈ ℝ²ⁿ, x = u − v: u, v ≥ 0, 1ᵀ(u + v) ≤ r.

    # NOTE: the l1 ball in ℝ⁵⁰ has 2⁵⁰ facets, so numopt's polyhedral projection cannot use it
    # directly; the lifted set has 2n + 1 facets and 2n + 1 vertices (0 and r·e_i).
    """
    n = A.shape[1]
    B = np.hstack([A, -A])
    N = 2 * n
    A_ub = np.vstack([-np.eye(N), np.ones((1, N))])
    b_ub = np.concatenate([np.zeros(N), [r]])
    V = np.vstack([np.zeros(N), r * np.eye(N)])
    return Problem(
        id="l1ball_lifted",
        name="l1-ball least squares (lifted)",
        latex=r"\min \tfrac12\|B z-b\|^2,\ z\ge0,\ 1^\top z\le r",
        f=lambda z: 0.5 * float((B @ z - b) @ (B @ z - b)),
        dim=N,
        domain=(),
        grad=lambda z: B.T @ (B @ z - b),
        x0=np.zeros(N),
        extra={"projection": "polyhedron", "A_ub": A_ub, "b_ub": b_ub, "vertices": V},
    )


LASSO2D_A = np.array([[1.0, 0.95], [0.0, 0.15]])
LASSO2D_B = np.array([1.2, -0.3])
LASSO2D_LAM = 0.2
LASSO2D_X0 = np.array([-1.0, 1.5])


def stage2_2d() -> dict[str, Any]:
    A, b, lam = LASSO2D_A, LASSO2D_B, LASSO2D_LAM
    cert = lasso_solution(A, b, lam)
    P = make_lasso(A, b, lam, id="lasso2d", name="2-D lasso", x0=LASSO2D_X0)
    L = P.extra["L"]
    S_star = frozenset(cert.support)
    out: dict[str, Any] = {
        "A": A,
        "b": b,
        "lam": lam,
        "L": L,
        "kappa_AtA": float(np.linalg.cond(A.T @ A)),
        "x_star": cert.x,
        "F_star": cert.F,
        "kkt_margin": cert.margin,
        "x0": LASSO2D_X0,
        "methods": {},
    }
    paths: dict[str, np.ndarray] = {}
    for lab, fn, params in prox_methods(L, 1e-13, 400):
        res = fn(P, x0=LASSO2D_X0, **params)
        xs = [np.asarray(s.x) for s in res.trace]
        st = support_stats(xs, S_star)
        dist = [float(np.linalg.norm(x - cert.x)) for x in xs]
        out["methods"][lab] = {
            "k_dist_1e-6": first_below(dist, 1e-6),
            "k_F_1e-8": first_below([fun(s) - cert.F for s in res.trace], 1e-8),
            "support_first": st["first"],
            "support_final": st["final"],
            "support_changes_after_first": st["changes_after_first"],
            "n_iter": res.n_iter,
            "n_gev": res.n_gev,
            "converged": res.converged,
            "restarts": res.extra.get("restarts"),
        }
        paths[lab] = np.array(xs)
        out["methods"][lab]["dist"] = dist
    r = float(np.abs(cert.x).sum())
    Pb = l1_ball_problem_2d(A, b, r, LASSO2D_X0)
    for lab, mid, params in (
        ("PG (l1 ball)", "projected_gradient", {"tol": 1e-13, "max_iter": 400}),
        ("FW (l1 ball)", "frank_wolfe", {"tol": 1e-13, "max_iter": 400}),
    ):
        res = run_method(mid, Pb, **params)
        xs = [np.asarray(s.x) for s in res.trace]
        dist = [float(np.linalg.norm(x - cert.x)) for x in xs]
        st = support_stats(xs, S_star)
        out["methods"][lab] = {
            "k_dist_1e-6": first_below(dist, 1e-6),
            "support_first": st["first"],
            "support_final": st["final"],
            "n_iter": res.n_iter,
            "n_gev": res.n_gev,
            "converged": res.converged,
            "message": res.message,
            "dist": dist,
        }
        paths[lab] = np.array(xs)
    out["_paths"] = paths
    out["_problem"] = P
    return out


LASSO_M, LASSO_N, LASSO_S = 25, 50, 5
LASSO_SEEDS = (0, 1, 2, 3, 4)
LASSO_RATIOS = (0.3, 0.1, 0.03)
LASSO_NOISE = 0.05
LASSO_MAX_ITER = 20_000
LASSO_GTOL = 1e-11
PG_MAX_ITER = 400


def make_lasso_instance(seed: int, ratio: float) -> tuple[Any, Any]:
    rng = Rng(20_000 + seed)
    A = normals(rng, (LASSO_M, LASSO_N)) / math.sqrt(LASSO_M)
    perm = rng.permutation(LASSO_N)
    x_true = np.zeros(LASSO_N)
    for i in perm[:LASSO_S]:
        x_true[i] = (1.0 + abs(rng.normal())) * (1.0 if rng.random() < 0.5 else -1.0)
    b = A @ x_true + LASSO_NOISE * normals(rng, (LASSO_M,))
    lam_max = float(np.max(np.abs(A.T @ b)))
    lam = ratio * lam_max
    cert = lasso_solution(A, b, lam)
    P = make_lasso(
        A,
        b,
        lam,
        id=f"lasso50_s{seed}_r{ratio}",
        name="sparse regression lasso",
        x0=np.zeros(LASSO_N),
    )
    return P, cert


def stage2_50() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    example: dict[str, Any] = {}
    for seed in LASSO_SEEDS:
        for ratio in LASSO_RATIOS:
            P, cert = make_lasso_instance(seed, ratio)
            L = P.extra["L"]
            S_star = frozenset(cert.support)
            inst = {
                "seed": seed,
                "ratio": ratio,
                "lam": P.extra["lam"],
                "L": L,
                "support_size": len(S_star),
                "kkt_margin": cert.margin,
                "min_abs_xstar": cert.min_abs,
                "F_star": cert.F,
            }
            for lab, fn, params in prox_methods(L, LASSO_GTOL, LASSO_MAX_ITER):
                res = fn(P, **params)
                xs = [np.asarray(s.x) for s in res.trace]
                gap = [fun(s) - cert.F for s in res.trace]
                st = support_stats(xs, S_star)
                rows.append(
                    {
                        **inst,
                        "method": lab,
                        "support_first": st["first"],
                        "support_final": st["final"],
                        "support_changes_after_first": st["changes_after_first"],
                        "k_F_1e-8": first_below(gap, 1e-8),
                        "final_gap": gap[-1],
                        "final_dist": float(np.linalg.norm(xs[-1] - cert.x)),
                        "n_iter": res.n_iter,
                        "n_gev": res.n_gev,
                        "converged": res.converged,
                        "n_restart": res.extra.get("n_restart"),
                    }
                )
                if seed == 0 and ratio == 0.1:
                    example[lab] = {"gap": gap, "supp_err": st["err"]}
            # Frank–Wolfe on the equivalent l1-ball problem (lifted), every instance.
            A, b = P.extra["A"], P.extra["b"]
            r = float(np.abs(cert.x).sum())
            Pl = l1_ball_problem_lifted(A, b, r)
            res = run_method("frank_wolfe", Pl, tol=1e-13, max_iter=3000)
            n = LASSO_N
            xs = [np.asarray(s.x)[:n] - np.asarray(s.x)[n:] for s in res.trace]
            st = support_stats(xs, S_star)
            rows.append(
                {
                    **inst,
                    "method": "FW (l1 ball)",
                    "support_first": st["first"],
                    "support_final": st["final"],
                    "support_changes_after_first": st["changes_after_first"],
                    "final_dist": float(np.linalg.norm(xs[-1] - cert.x)),
                    "final_nnz": int(np.count_nonzero(xs[-1])),
                    "n_iter": res.n_iter,
                    "n_gev": res.n_gev,
                    "converged": res.converged,
                    "message": res.message,
                }
            )
            if seed == 0 and ratio == 0.1:
                example["FW (l1 ball)"] = {"dist": [float(np.linalg.norm(x - cert.x)) for x in xs]}
                # Projected gradient: one instance only (each Armijo trial solves a 100-variable
                # QP for the polyhedral projection; ~25 ms per iteration here).
                t0 = time.perf_counter()
                res = run_method("projected_gradient", Pl, tol=1e-13, max_iter=PG_MAX_ITER)
                xs = [np.asarray(s.x)[:n] - np.asarray(s.x)[n:] for s in res.trace]
                st = support_stats(xs, S_star)
                rows.append(
                    {
                        **inst,
                        "method": "PG (l1 ball)",
                        "support_first": st["first"],
                        "support_final": st["final"],
                        "support_changes_after_first": st["changes_after_first"],
                        "final_dist": float(np.linalg.norm(xs[-1] - cert.x)),
                        "final_nnz": int(np.count_nonzero(xs[-1])),
                        "n_iter": res.n_iter,
                        "n_gev": res.n_gev,
                        "converged": res.converged,
                        "message": res.message,
                        "seconds": time.perf_counter() - t0,
                    }
                )
                example["PG (l1 ball)"] = {"dist": [float(np.linalg.norm(x - cert.x)) for x in xs]}
        print(f"  stage 2: seed {seed} done", flush=True)
    return {"rows": rows, "example": example}


def summarize_stage2(d50: dict[str, Any]) -> dict[str, Any]:
    rows = d50["rows"]
    out: dict[str, Any] = {}
    insts = sorted({(r["seed"], r["ratio"]) for r in rows})

    def get(lab: str, s: int, ra: float) -> dict[str, Any]:
        return next(r for r in rows if r["method"] == lab and r["seed"] == s and r["ratio"] == ra)

    for lab in PROX_LABELS:
        fin = [get(lab, s, ra)["support_final"] for s, ra in insts]
        ok = [v for v in fin if v is not None]
        out[lab] = {
            "identified": len(ok),
            "median_support_final": float(np.median(ok)) if ok else None,
            "median_support_first": float(
                np.median([get(lab, s, ra)["support_first"] for s, ra in insts])
            ),
            "median_k_F_1e-8": float(np.median([get(lab, s, ra)["k_F_1e-8"] for s, ra in insts])),
            "median_changes_after_first": float(
                np.median([get(lab, s, ra)["support_changes_after_first"] for s, ra in insts])
            ),
            "max_final_gap": max(get(lab, s, ra)["final_gap"] for s, ra in insts),
            "by_ratio_median_support_final": {
                f"{ra:g}": float(np.median([get(lab, s, ra)["support_final"] for s in LASSO_SEEDS]))
                for ra in LASSO_RATIOS
            },
        }
    for lab in PROX_LABELS:
        rr = [get(lab, s, ra) for s, ra in insts]
        lags = [r["support_final"] - r["support_first"] for r in rr]
        out[lab]["n_instances_lag_positive"] = sum(v > 0 for v in lags)
        out[lab]["max_lag_final_minus_first"] = max(lags)
        out[lab]["max_changes_after_first"] = max(r["support_changes_after_first"] for r in rr)
        out[lab]["median_restarts"] = float(np.median([r["n_restart"] or 0 for r in rr]))
        out[lab]["n_converged"] = sum(r["converged"] for r in rr)
        out[lab]["max_final_dist"] = max(r["final_dist"] for r in rr)
    cmp: dict[str, Any] = {}
    for a, bl in (
        ("FISTA", "ISTA"),
        ("FISTA-GR", "ISTA"),
        ("FISTA-FR", "ISTA"),
        ("FISTA-GR", "FISTA"),
        ("FISTA-FR", "FISTA"),
        ("AdProxGD", "ISTA"),
    ):
        ra_ = [get(a, s, r)["support_final"] / get(bl, s, r)["support_final"] for s, r in insts]
        cmp[f"{a}/{bl}"] = {
            "median_ratio_support_final": float(np.median(ra_)),
            "n_later": int(sum(x > 1 for x in ra_)),
            "n_instances": len(ra_),
        }
    out["comparisons"] = cmp
    # Does restart remove FISTA's lag? Count over the instances where FISTA HAS a lag, and
    # list where restart moves the FIRST identification later.
    lag = {
        lab: {
            (s, ra): get(lab, s, ra)["support_final"] - get(lab, s, ra)["support_first"]
            for s, ra in insts
        }
        for lab in PROX_LABELS
    }
    affected = [i for i in insts if lag["FISTA"][i] > 0]
    restart_effect: dict[str, Any] = {"fista_lag_instances": [list(i) for i in affected]}
    for lab in ("FISTA-FR", "FISTA-GR"):
        restart_effect[lab] = {
            "n_lag_removed": sum(lag[lab][i] == 0 for i in affected),
            "n_affected": len(affected),
            "n_lag_free": sum(v == 0 for v in lag[lab].values()),
            "remaining": [
                {
                    "seed": i[0],
                    "ratio": i[1],
                    "lag_fista": lag["FISTA"][i],
                    "lag_restart": lag[lab][i],
                    "first_fista": get("FISTA", *i)["support_first"],
                    "final_fista": get("FISTA", *i)["support_final"],
                    "first_restart": get(lab, *i)["support_first"],
                    "final_restart": get(lab, *i)["support_final"],
                }
                for i in affected
                if lag[lab][i] > 0
            ],
            "new_lag": [list(i) for i in insts if lag["FISTA"][i] == 0 and lag[lab][i] > 0],
            "first_later_than_fista": [
                {
                    "seed": i[0],
                    "ratio": i[1],
                    "first_fista": get("FISTA", *i)["support_first"],
                    "first_restart": get(lab, *i)["support_first"],
                    "final_fista": get("FISTA", *i)["support_final"],
                    "final_restart": get(lab, *i)["support_final"],
                }
                for i in insts
                if get(lab, *i)["support_first"] > get("FISTA", *i)["support_first"]
            ],
        }
    out["restart_effect_on_lag"] = restart_effect
    # "Delay" of FISTA's final identification beyond its first identification, vs ISTA.
    out["lag_final_minus_first"] = {
        lab: float(
            np.median(
                [
                    get(lab, s, ra)["support_final"] - get(lab, s, ra)["support_first"]
                    for s, ra in insts
                ]
            )
        )
        for lab in PROX_LABELS
    }
    fw = [r for r in rows if r["method"] == "FW (l1 ball)"]
    out["FW (l1 ball)"] = {
        "identified": sum(r["support_final"] is not None for r in fw),
        "median_final_dist": float(np.median([r["final_dist"] for r in fw])),
        "median_final_nnz": float(np.median([r["final_nnz"] for r in fw])),
        "min_final_dist": min(r["final_dist"] for r in fw),
        "max_final_dist": max(r["final_dist"] for r in fw),
        "n_iter": fw[0]["n_iter"],
    }
    pg = [r for r in rows if r["method"] == "PG (l1 ball)"]
    out["PG (l1 ball)"] = {
        k: pg[0][k]
        for k in ("final_dist", "final_nnz", "n_iter", "support_final", "message", "seconds")
    }
    return out


def plot_stage2(d2: dict[str, Any], d50: dict[str, Any], s50: dict[str, Any]) -> None:
    # --- Figure 4: 2-D lasso trajectories ----------------------------------------------------
    P = d2["_problem"]
    xs_ = np.linspace(-1.3, 1.6, 300)
    ys_ = np.linspace(-0.9, 1.95, 300)
    X, Y = np.meshgrid(xs_, ys_)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = np.array([X[i, j], Y[i, j]])
            Z[i, j] = P.f(z) + P.g(z)
    Fs = d2["F_star"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), layout="constrained", width_ratios=(1.1, 1)
    )
    levels = Fs + np.geomspace(1e-3, float(Z.max() - Fs), 14)
    ax.contour(X, Y, Z, levels=levels, colors=GRID, linewidths=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    r = float(np.abs(d2["x_star"]).sum())
    ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)))
    ax.annotate(r"$\|x\|_1 \leq \|x^*\|_1$", (-0.62, -0.5), color=MUTED, fontsize=8)
    for lab, lw in (("ISTA", 2.6), ("AdProxGD", 1.4), ("FISTA-GR", 1.4), ("FISTA", 1.2)):
        p = d2["_paths"][lab][:120]
        ax.plot(
            p[:, 0],
            p[:, 1],
            color=PC[lab],
            linestyle=PLS[lab],
            linewidth=lw,
            marker="o",
            markersize=2.4,
            label=lab,
        )
    ax.plot(*d2["x0"], marker="s", color=INK, markersize=5)
    ax.annotate("$x_0$", d2["x0"], xytext=(5, 2), textcoords="offset points", fontsize=9)
    ax.plot(*d2["x_star"], marker="*", color=INK, markersize=11)
    ax.annotate(
        "$x^* = (1, 0)$", d2["x_star"], xytext=(4, 6), textcoords="offset points", fontsize=9
    )
    ax.set_xlim(xs_[0], xs_[-1])
    ax.set_ylim(ys_[0], ys_[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(
        rf"2-D lasso, $\kappa(A^\top A)$ = {d2['kappa_AtA']:.0f}: first 120 iterates",
        fontsize=10,
        color=INK,
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for lab in (*PROX_LABELS, "PG (l1 ball)", "FW (l1 ball)"):
        dist = np.maximum(np.array(d2["methods"][lab]["dist"]), 1e-17)
        ax2.plot(
            np.arange(dist.size), dist, color=PC[lab], linestyle=PLS[lab], linewidth=1.4, label=lab
        )
    ax2.set_yscale("log")
    ax2.set_ylim(1e-14, 10)
    ax2.set_xlim(0, 120)
    ax2.set_xlabel("iteration k")
    ax2.set_ylabel(r"$\|x_k - x^*\|_2$")
    ax2.set_title("Distance to the minimizer", fontsize=10, color=INK)
    style_axes(ax2)
    ax2.legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    save(fig, "lasso2d")

    # --- Figure 5: one n = 50 instance: gap and support error -------------------------------
    ex = d50["example"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.0), layout="constrained", sharex=True)
    kmax = 0
    for lab in PROX_LABELS:
        gap = np.maximum(np.array(ex[lab]["gap"]), 1e-17)
        kmax = max(kmax, gap.size)
        a1.plot(
            np.arange(gap.size), gap, color=PC[lab], linestyle=PLS[lab], linewidth=1.4, label=lab
        )
        a2.plot(
            np.arange(len(ex[lab]["supp_err"])),
            ex[lab]["supp_err"],
            color=PC[lab],
            linestyle=PLS[lab],
            linewidth=1.3,
            label=lab,
        )
    a1.set_yscale("log")
    a1.set_ylim(1e-15, None)
    a1.set_ylabel(r"$F(x_k) - F^*$")
    a1.set_title(
        r"Sparse regression lasso, n = 50, m = 25, seed 0, $\lambda = 0.1\,\lambda_{\max}$",
        fontsize=10,
        color=INK,
    )
    a2.set_ylabel(r"$|\mathrm{supp}(x_k)\ \triangle\ \mathrm{supp}(x^*)|$")
    a2.set_yscale("symlog", linthresh=1)
    a2.set_ylim(0, None)
    a2.set_xlabel("iteration k")
    a2.set_xscale("log")
    for a in (a1, a2):
        style_axes(a)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "lasso50_example")

    # --- Figure 6: final support identification, all 15 instances ---------------------------
    rows = d50["rows"]
    insts = sorted({(r["seed"], r["ratio"]) for r in rows}, key=lambda t: (-t[1], t[0]))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    for j, lab in enumerate(PROX_LABELS):
        ys = []
        for s, ra in insts:
            r = next(x for x in rows if x["method"] == lab and x["seed"] == s and x["ratio"] == ra)
            ys.append(r["support_final"] if r["support_final"] is not None else np.nan)
        xs = np.arange(len(insts)) + (j - 2) * 0.13
        ax.scatter(
            xs, ys, s=22, color=PC[lab], marker="os^Dv"[j], label=lab, linewidths=0, zorder=3
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(insts)))
    ax.set_xticklabels([f"{ra:g}\ns{s}" for s, ra in insts], fontsize=7)
    for b in (4.5, 9.5):
        ax.axvline(b, color=GRID, linewidth=1.0)
    ax.set_xlabel(r"instance ($\lambda/\lambda_{\max}$, data seed)")
    ax.set_ylabel("final support identification k")
    ax.set_title(
        "Iteration after which supp(x_k) = supp(x*) for good (15 instances)", fontsize=10, color=INK
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncols=5, loc="upper left")
    save(fig, "lasso50_identification")


# ======================================================================================


def dump(name: str, obj: Any) -> None:
    (RESULTS / name).write_text(json.dumps(to_jsonable(obj), indent=1, allow_nan=False))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "axes.titleweight": "normal"})
    t0 = time.perf_counter()
    print("stage 1: quadratics", flush=True)
    d1 = stage1()
    s1 = summarize_stage1(d1)
    prof, _ = stage1_profile(d1)
    s1["profile_rho_at_1"] = dict(zip(prof.labels, prof.at(1.0).tolist(), strict=True))
    s1["profile_rho_at_2"] = dict(zip(prof.labels, prof.at(2.0).tolist(), strict=True))
    raw, _ = stage1_profile(d1, charged=False)
    s1["profile_raw_nesterov_rho_at_1"] = dict(zip(raw.labels, raw.at(1.0).tolist(), strict=True))
    s1["profile_raw_nesterov_rho_at_2"] = dict(zip(raw.labels, raw.at(2.0).tolist(), strict=True))
    sweep = stage1_bt_sweep(d1)
    s1["bt_L0_sweep"] = sweep["summary"]
    dump("quadratic_runs.json", {"eps": EPS, "gtol": GTOL, "n": N_QUAD, "rows": d1["rows"]})
    dump("quadratic_bt_sweep.json", {"L0": BT_L0, "rows": sweep["rows"]})
    dump("quadratic_summary.json", s1)
    plot_stage1(d1, s1, prof)
    print(f"  {time.perf_counter() - t0:.1f} s", flush=True)

    print("stage 2: lasso", flush=True)
    d2 = stage2_2d()
    dump(
        "lasso2d.json",
        {k: v for k, v in d2.items() if not k.startswith("_")}
        | {
            "methods": {
                m: {k: v for k, v in d.items() if k != "dist"} for m, d in d2["methods"].items()
            }
        },
    )
    d50 = stage2_50()
    s50 = summarize_stage2(d50)
    dump(
        "lasso50_runs.json",
        {"m": LASSO_M, "n": LASSO_N, "s": LASSO_S, "noise": LASSO_NOISE, "rows": d50["rows"]},
    )
    dump("lasso50_summary.json", s50)
    plot_stage2(d2, d50, s50)
    total = time.perf_counter() - t0
    dump(
        "meta.json",
        {"runtime_seconds": total, "numpy": np.__version__, "python": sys.version.split()[0]},
    )
    print(f"done in {total:.1f} s", flush=True)


if __name__ == "__main__":
    main()
