"""Reproducible experiment: AAA and Floater–Hormann against numopt's polynomial baselines.

Run:  .venv/bin/python research/barycentric-rational-approximation/run.py

Deterministic (no random numbers, no network). Writes results/*.json, results/tables.md and
figures/*.svg next to this file. Runtime on the development machine: about 1 minute.

Q1 (AAA): for f = |x| on [-1, 1], sqrt(x) on [0, 1], tanh(50x) and 1/(1+25x²) on [-1, 1],
record the sup-norm error of every AAA step (type (n, n), n = m - 1) on a dense test grid
that is disjoint from the samples, and compare with numopt's Chebyshev interpolation of
degree n (n + 1 Chebyshev points). Fits of ln E against √n and ln n test the rate.

Q2 (Floater–Hormann): Runge's function on n + 1 equispaced points, n = 10..1280, FH with
d = 3..8 against numopt's Lagrange / barycentric polynomial, not-a-knot spline and PCHIP on
the same samples (and Chebyshev interpolation at n + 1 Chebyshev points as an "if the
nodes were free" reference); a sweep over d with Lebesgue constants.
"""

from __future__ import annotations

import os

# NOTE: one BLAS thread. The SVDs here are small (≤ 20000 × 100); on a loaded many-core
# machine multi-threaded OpenBLAS was measured 50× slower (8.4 s vs 0.17 s per SVD).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib.util
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
from matplotlib.scale import FuncScale

from numopt.core.types import Dataset, Result, to_jsonable
from numopt.interpolation.methods import (
    _barycentric_eval,
    barycentric,
    chebyshev_interpolation,
    cubic_spline_not_a_knot,
    lagrange,
    pchip,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def _load_method() -> Any:
    spec = importlib.util.spec_from_file_location("barycentric_rational_method", HERE / "method.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["barycentric_rational_method"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_method()
EPS = float(np.finfo(np.float64).eps)

# ---- palette (dataviz reference palette, light surface) -------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # 250..700
REF_GRAY = "#9a9893"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "svg.fonttype": "none",
        "svg.hashsalt": "barycentric-rational-approximation",
        "font.family": "DejaVu Sans",
    }
)


def _sup_error(values: np.ndarray, truth: np.ndarray) -> float:
    if not np.all(np.isfinite(values)):
        return math.inf
    return float(np.max(np.abs(values - truth)))


# =======================================================================================
# Q1: AAA vs Chebyshev interpolation
# =======================================================================================

N_TERMS = 100  # AAA max_terms (NST default mmax)
LEVELS = (1e-6, 1e-10, 1e-13)
#: A final pole estimate counts as resolved when |d(λ)| ≤ 1e-8·Σ_j |w_j/(λ - z_j)|.
POLE_RESOLVED_RTOL = 1e-8  # error levels of the "smallest degree to reach" table
CHEB_DEGREES = sorted({int(v) for v in np.rint(np.logspace(0, 3, 31))} | set(range(1, 21)))


def _lg(lo: float, hi: float, n: int) -> np.ndarray:
    return np.logspace(lo, hi, n)


def q1_problems() -> dict[str, dict[str, Any]]:
    """Functions, intervals, AAA sample sets Z and test grids T (T ∩ Z is not required empty,
    but the clustered test points use a different exponent grid than the samples)."""
    return {
        "abs": {
            "label": "|x| on [-1, 1]",
            "f": np.abs,
            "a": -1.0,
            "b": 1.0,
            "samples": {
                "equispaced": np.linspace(-1.0, 1.0, 20_000),
                # |x| at ±10^-15..1 corresponds to sqrt(x) at 10^-30..1 (x = t²)
                "clustered": np.unique(
                    np.concatenate(
                        [np.linspace(-1.0, 1.0, 4000), _lg(-15, 0, 1000), -_lg(-15, 0, 1000)]
                    )
                ),
            },
            "test": np.unique(
                np.concatenate(
                    [
                        np.linspace(-1.0, 1.0, 200_001),
                        _lg(-16.05, 0, 20_001),
                        -_lg(-16.05, 0, 20_001),
                    ]
                )
            ),
            "best_C": math.pi,  # Stahl (1993): E_nn(|x|) ~ 8 exp(-π √n)
            "best_label": r"best: $8e^{-\pi\sqrt{n}}$",
        },
        "sqrt": {
            "label": "√x on [0, 1]",
            "f": np.sqrt,
            "a": 0.0,
            "b": 1.0,
            "samples": {
                "equispaced": np.linspace(0.0, 1.0, 20_000),
                "clustered": np.unique(
                    np.concatenate([np.linspace(0.0, 1.0, 4000), _lg(-30, 0, 2000)])
                ),
            },
            "test": np.unique(
                np.concatenate([np.linspace(0.0, 1.0, 100_001), _lg(-32.05, 0, 30_001)])
            ),
            # type (n, n) for √x on [0, 1] = type (2n, 2n) for |t| on [-1, 1]
            "best_C": math.pi * math.sqrt(2.0),
            "best_label": r"best: $8e^{-\pi\sqrt{2n}}$",
        },
        "tanh50": {
            "label": "tanh(50x) on [-1, 1]",
            "f": lambda x: np.tanh(50.0 * x),
            "a": -1.0,
            "b": 1.0,
            "samples": {"equispaced": np.linspace(-1.0, 1.0, 2000)},
            "test": np.linspace(-1.0, 1.0, 100_001),
            # poles at ±iπ/100: Bernstein ellipse parameter ρ = s + √(1 + s²), s = π/100
            "rho": math.pi / 100 + math.sqrt(1 + (math.pi / 100) ** 2),
        },
        "runge": {
            "label": "1/(1+25x²) on [-1, 1]",
            "f": lambda x: 1.0 / (1.0 + 25.0 * x * x),
            "a": -1.0,
            "b": 1.0,
            "samples": {"equispaced": np.linspace(-1.0, 1.0, 2000)},
            "test": np.linspace(-1.0, 1.0, 100_001),
            "rho": 0.2 + math.sqrt(1.04),  # poles at ±i/5
        },
    }


def run_aaa(
    fn: Callable[[Any], Any], z: np.ndarray, test: np.ndarray, scaling: str
) -> dict[str, Any]:
    t0 = time.perf_counter()
    res: Result = M.aaa((z, fn(z)), max_terms=N_TERMS, scaling=scaling)
    seconds = time.perf_counter() - t0
    truth = fn(test)
    steps = []
    for s in res.trace[1:]:
        info = s.info
        with np.errstate(all="ignore"):
            vals = _barycentric_eval(
                np.asarray(info["support"]),
                np.asarray(info["weights"]),
                np.asarray(info["support_values"]),
                test,
            )
        steps.append(
            {
                "n": int(info["degree"]),
                "sample_error": float(info["sample_error"]),
                "test_error": _sup_error(vals, truth),
                "n_interval_poles": int(info["n_interval_poles"]),
                "n_doublets": int(info["n_doublets"]),
            }
        )
    return {
        "scaling": scaling,
        "M": int(z.size),
        "converged": res.converged,
        "message": res.message,
        "n_support": res.n_iter,
        "seconds": seconds,
        "steps": steps,
        "final_poles": res.extra["poles"],
        "final_residues": res.extra["residues"],
        "final_pole_rel_residual": _pole_rel_residual(res),
        "final_interval_poles": res.extra["interval_poles"],
        "final_support": res.extra["nodes"],
    }


def _pole_rel_residual(res: Result) -> list[float]:
    """|d(λ)| / Σ_j |w_j/(λ - z_j)| at each final pole estimate: ≈ eps for a resolved
    zero of d, large where the estimate is below the solver's absolute resolution."""
    z = np.asarray(res.extra["nodes"])
    w = np.asarray(res.extra["coefficients"])
    lam = np.array([complex(*p) for p in res.extra["poles"]], dtype=complex)
    if lam.size == 0:
        return []
    with np.errstate(all="ignore"):
        c = w[None, :] / (lam[:, None] - z[None, :])
        return [float(v) for v in np.abs(c.sum(axis=1)) / np.abs(c).sum(axis=1)]


def run_chebyshev(
    fn: Callable[[Any], Any], a: float, b: float, test: np.ndarray
) -> list[dict[str, Any]]:
    x0 = np.linspace(a, b, 5)
    ds = Dataset("q1", "q1", x0, fn(x0), f_true=fn, domain=(a, b))
    truth = fn(test)
    u = (2.0 * test - (a + b)) / (b - a)
    out = []
    for n in CHEB_DEGREES:
        res = chebyshev_interpolation(ds, n_nodes=n + 1)
        assert res.extra["source"] == "f_true"
        coef = np.asarray(res.extra["coefficients"])
        vals = np.polynomial.chebyshev.chebval(u, coef)  # independent evaluator (not Clenshaw copy)
        out.append(
            {
                "n": n,
                "test_error": _sup_error(vals, truth),
                "converged": res.converged,
                "coefficients": coef,
            }
        )
    return out


CHEB_SCAN_MAX = 1000  # the reach table scans every degree n = 1..CHEB_SCAN_MAX
CHEB_SCAN_STRIDE = 50  # subset = every 50th test point (lower bound on the test-grid error)


def _cheb_coefficients(fn: Callable[[Any], Any], a: float, b: float, n_nodes: int) -> np.ndarray:
    """c_k = (2/N) Σ_j f(x_j) cos(πk(j + ½)/N), c_0 halved, x_j the roots of T_N on [a, b]:
    numopt's chebyshev_interpolation formula, computed by a DCT-II (O(N log N)) so that the
    scan over every degree is affordable; checked against numopt at the grid degrees."""
    import scipy.fft  # experiment-side only

    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(theta)
    c = np.asarray(scipy.fft.dct(np.asarray(fn(x), dtype=np.float64), type=2)) / n_nodes
    c[0] *= 0.5
    return c


def chebyshev_reach(
    fn: Callable[[Any], Any],
    a: float,
    b: float,
    test: np.ndarray,
    levels: tuple[float, ...],
    numopt_runs: list[dict[str, Any]],
    numopt_coef: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Smallest degree n ≤ CHEB_SCAN_MAX with test-grid error ≤ level, scanning every n.

    For each n the error on a subset of the test grid is computed first; it is a lower
    bound on the full test-grid error, so n is skipped when it exceeds every pending level.
    Otherwise the full grid decides. The DCT coefficients are compared with numopt's at
    every grid degree (max |difference| reported)."""
    truth = fn(test)
    u = (2.0 * test - (a + b)) / (b - a)
    sub, sub_truth = u[::CHEB_SCAN_STRIDE], truth[::CHEB_SCAN_STRIDE]
    first: dict[str, int | None] = {f"{lv:.0e}": None for lv in levels}
    full_evals = 0
    coef_diff = 0.0
    for n in range(1, CHEB_SCAN_MAX + 1):
        c = _cheb_coefficients(fn, a, b, n + 1)
        if n in numopt_coef:
            coef_diff = max(coef_diff, float(np.max(np.abs(c - numopt_coef[n]))))
        pending = [lv for lv in levels if first[f"{lv:.0e}"] is None]
        if not pending:
            continue
        e_sub = _sup_error(np.polynomial.chebyshev.chebval(sub, c), sub_truth)
        if e_sub > max(pending):
            continue
        full_evals += 1
        e = _sup_error(np.polynomial.chebyshev.chebval(u, c), truth)
        for lv in pending:
            if e <= lv:
                first[f"{lv:.0e}"] = n
    grid_first = {}
    cn = np.array([r["n"] for r in numopt_runs])
    ce = np.array([r["test_error"] for r in numopt_runs])
    for lv in levels:
        grid_first[f"{lv:.0e}"] = _first_n_below(cn, ce, lv)
    return {
        "first_n": first,
        "first_grid_n": grid_first,
        "n_max": CHEB_SCAN_MAX,
        "full_grid_evaluations": full_evals,
        "max_coef_diff_vs_numopt": coef_diff,
    }


def _fit1(ns: np.ndarray, errs: np.ndarray, model: str) -> tuple[float, float]:
    g = {"root_exp": np.sqrt(ns), "algebraic": np.log(ns), "geometric": ns.astype(float)}[model]
    A = np.column_stack([np.ones_like(g), -g])
    y = np.log(errs)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = (y - A @ coef) / math.log(10.0)
    return float(coef[1]), float(np.sqrt(np.mean(resid**2)))


def _fit(ns: np.ndarray, errs: np.ndarray, model: str) -> dict[str, float]:
    """Least-squares fit of ln E = α - C·g(n), g = √n ('root_exp'), ln n ('algebraic'),
    n ('geometric'), on the whole window and on its first and second halves. A model that
    describes the data has the same C on both halves; the RMS residual is in decades."""
    c, rms = _fit1(ns, errs, model)
    h = ns.size // 2
    c1, _ = _fit1(ns[: h + 1], errs[: h + 1], model)
    c2, _ = _fit1(ns[h:], errs[h:], model)
    return {
        "C": c,
        "rms_decades": rms,
        "C_first_half": c1,
        "C_second_half": c2,
        "n_lo": int(ns[0]),
        "n_hi": int(ns[-1]),
        "points": int(ns.size),
    }


def _first_n_below(ns: np.ndarray, errs: np.ndarray, level: float) -> int | None:
    hit = np.flatnonzero(errs <= level)
    return int(ns[hit[0]]) if hit.size else None


def q1() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, prob in q1_problems().items():
        fn = prob["f"]
        entry: dict[str, Any] = {"label": prob["label"], "aaa": {}, "fits": {}}
        for sname, z in prob["samples"].items():
            for scaling in ("columns", "none"):
                entry["aaa"][f"{sname}/{scaling}"] = run_aaa(fn, z, prob["test"], scaling)
        cheb_runs = run_chebyshev(fn, prob["a"], prob["b"], prob["test"])
        numopt_coef = {int(c["n"]): np.asarray(c.pop("coefficients")) for c in cheb_runs}
        entry["chebyshev"] = cheb_runs
        entry["chebyshev_reach"] = chebyshev_reach(
            fn, prob["a"], prob["b"], prob["test"], LEVELS, cheb_runs, numopt_coef
        )
        # ---- rate fits ----
        cheb_n = np.array([c["n"] for c in entry["chebyshev"]])
        cheb_e = np.array([c["test_error"] for c in entry["chebyshev"]])
        if key in ("abs", "sqrt"):
            sel = (cheb_n >= 10) & (cheb_e > 1e-15)
            for model in ("algebraic", "root_exp"):
                entry["fits"][f"chebyshev/{model}"] = _fit(cheb_n[sel], cheb_e[sel], model)
            for run_key in ("clustered/columns", "clustered/none", "equispaced/columns"):
                steps = entry["aaa"][run_key]["steps"]
                ns = np.array([s["n"] for s in steps])
                es = np.array([s["test_error"] for s in steps])
                # fit window: n ≥ 6 (after the start-up steps) up to the first step with
                # error ≤ 1e-11, else up to the smallest error (so the 1e-13 rounding
                # floor does not enter the fit)
                below = np.flatnonzero(es <= 1e-11)
                hi = int(below[0]) if below.size else int(np.argmin(es))
                sel = (ns >= 6) & (ns <= ns[hi]) & np.isfinite(es)
                if sel.sum() >= 3:
                    for model in ("algebraic", "root_exp"):
                        entry["fits"][f"aaa {run_key}/{model}"] = _fit(ns[sel], es[sel], model)
                # sensitivity: the root-exponential fit up to the best step (rounding floor
                # included), and from n = 10 up to n = 60 where the run gets that far
                k_best = int(np.argmin(np.where(np.isfinite(es), es, np.inf)))
                for name, lo_n, hi_n in (("to_best", 6, int(ns[k_best])), ("10_60", 10, 60)):
                    sel2 = (ns >= lo_n) & (ns <= hi_n) & np.isfinite(es)
                    if sel2.sum() >= 3 and hi_n <= ns[-1]:
                        entry["fits"][f"aaa {run_key}/root_exp_{name}"] = _fit(
                            ns[sel2], es[sel2], "root_exp"
                        )
            # ratio of the AAA error to the best-approximation asymptotics 8 e^{-C_best √n}
            steps = entry["aaa"]["clustered/columns"]["steps"]
            entry["ratio_to_best"] = {
                str(s["n"]): s["test_error"] / (8.0 * math.exp(-prob["best_C"] * math.sqrt(s["n"])))
                for s in steps
                if s["n"] >= 10 and (s["n"] % 5 == 0 or s is steps[-1])
            }
            entry["best_C"] = prob["best_C"]
        else:
            # window: n ≥ 10 while the error is above the rounding floor (> 1e-12)
            stop = int(np.argmax(cheb_e <= 1e-12)) if np.any(cheb_e <= 1e-12) else cheb_n.size
            sel = (cheb_n >= 10) & (np.arange(cheb_n.size) < stop)
            fit = _fit(cheb_n[sel], cheb_e[sel], "geometric")
            entry["fits"]["chebyshev/geometric"] = {
                **fit,
                "rho_fit": math.exp(fit["C"]),
                "rho_theory": prob["rho"],
            }
        reach: dict[str, Any] = {}
        for level in LEVELS:
            row = {"chebyshev": entry["chebyshev_reach"]["first_n"][f"{level:.0e}"]}
            for run_key, run in entry["aaa"].items():
                ns = np.array([s["n"] for s in run["steps"]])
                es = np.array([s["test_error"] for s in run["steps"]])
                row[f"aaa {run_key}"] = _first_n_below(ns, es, level)
            reach[f"{level:.0e}"] = row
        entry["degree_to_reach"] = reach
        out[key] = entry
        print(f"Q1 {key}: done", flush=True)
    return out


DEPTHS = (10, 15, 20, 25, 30)  # √x clustered samples log-spaced in [10^-depth, 1]


def q1_depth_sweep() -> list[dict[str, Any]]:
    """√x on [0, 1]: Z = 4000 equispaced ∪ 2000 log-spaced points in [10^-depth, 1].

    The test grid is the fixed Q1 √x grid (down to 1e-32) for every depth, so the test
    error estimates the true maximum error on [0, 1], also below the smallest sample."""
    prob = q1_problems()["sqrt"]
    test = prob["test"]
    truth = np.sqrt(test)
    out = []
    for depth in DEPTHS:
        z = np.unique(np.concatenate([np.linspace(0.0, 1.0, 4000), _lg(-depth, 0, 2000)]))
        for scaling in ("columns", "none"):
            res: Result = M.aaa((z, np.sqrt(z)), max_terms=N_TERMS, scaling=scaling)
            errs = []
            for st_ in res.trace[1:]:
                with np.errstate(all="ignore"):
                    vals = _barycentric_eval(
                        np.asarray(st_.info["support"]),
                        np.asarray(st_.info["weights"]),
                        np.asarray(st_.info["support_values"]),
                        test,
                    )
                errs.append(_sup_error(vals, truth))
            k = int(np.argmin(errs))
            out.append(
                {
                    "depth": depth,
                    "scaling": scaling,
                    "M": int(z.size),
                    "m": res.n_iter,
                    "converged": res.converged,
                    "sample_error": res.extra["sample_error"],
                    "test_error": errs[-1],
                    "best_test_error": errs[k],
                    "best_n": int(res.trace[k + 1].info["degree"]),
                    "sqrt_smallest_sample": math.sqrt(10.0**-depth),
                }
            )
        print(f"Q1 depth 1e-{depth}: done", flush=True)
    return out


# =======================================================================================
# Q2: Floater–Hormann on equispaced Runge samples
# =======================================================================================

N_LIST = [10, 20, 40, 80, 160, 320, 640, 1280]
D_LIST = [3, 4, 5, 6, 7, 8]
POLY_MAX_N = 80  # Lagrange form: O(n²) per point; diverged long before n = 80
SWEEP_N = [10, 20, 40, 80, 160]
SWEEP_D_MAX = 30
TABLE2_GRIDS = (101, 201, 501, 1001, 2001, 10_001, 100_001)
#: FH 2007 Table 3: clamped cubic spline on Runge 1/(1+x²), [-5, 5], n + 1 equispaced points.
FH_TABLE3_SPLINE = {
    10: 2.2e-2,
    20: 3.2e-3,
    40: 2.8e-4,
    80: 1.6e-5,
    160: 9.5e-7,
    320: 5.9e-8,
    640: 3.7e-9,
}


def runge(x: Any) -> Any:
    return 1.0 / (1.0 + 25.0 * np.square(x))


def _pp_eval(breaks: np.ndarray, coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Piecewise cubic a + b s + c s² + d s³, s = t - x_i on [x_i, x_{i+1}] (numopt layout)."""
    idx = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, breaks.size - 2)
    s = t - breaks[idx]
    a, b, c, d = coef[idx, 0], coef[idx, 1], coef[idx, 2], coef[idx, 3]
    return a + s * (b + s * (c + s * d))


def _lagrange_eval(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """p(t) = Σ y_j ℓ_j(t), ℓ_j = Π_{m≠j}(t - x_m)/(x_j - x_m) (numopt's Lagrange form)."""
    total = np.zeros_like(t)
    for j in range(x.size):
        lj = np.ones_like(t)
        for m in range(x.size):
            if m != j:
                lj = lj * ((t - x[m]) / (x[j] - x[m]))
        total = total + y[j] * lj
    return total


def q2() -> dict[str, Any]:
    t = np.linspace(-1.0, 1.0, 100_001)
    truth = runge(t)
    rows: list[dict[str, Any]] = []
    for n in N_LIST:
        x = np.linspace(-1.0, 1.0, n + 1)
        y = runge(x)
        row: dict[str, Any] = {"n": n, "h": 2.0 / n}
        for d in D_LIST:
            res = M.floater_hormann((x, y), d=d)
            w = res.extra["coefficients"]
            lam = M.lebesgue_constant(x, w, 30)
            row[f"fh{d}"] = {
                "error": _sup_error(_barycentric_eval(x, w, y, t), truth),
                "lebesgue": lam,
                "floor": EPS * lam * float(np.max(np.abs(y))),
                "converged": res.converged,
            }
        sp = cubic_spline_not_a_knot((x, y))
        row["spline"] = {
            "error": _sup_error(_pp_eval(x, np.asarray(sp.extra["coefficients"]), t), truth),
            "converged": sp.converged,
        }
        pc = pchip((x, y))
        row["pchip"] = {
            "error": _sup_error(_pp_eval(x, np.asarray(pc.extra["coefficients"]), t), truth),
            "converged": pc.converged,
        }
        bc = barycentric((x, y))
        if bc.converged:
            with np.errstate(all="ignore"):
                vals = _barycentric_eval(x, np.asarray(bc.extra["coefficients"]), y, t)
            row["barycentric"] = {"error": _sup_error(vals, truth), "converged": True}
        else:
            row["barycentric"] = {"error": None, "converged": False, "message": bc.message}
        if n <= POLY_MAX_N:
            lg = lagrange((x, y))
            with np.errstate(all="ignore"):
                row["lagrange"] = {
                    "error": _sup_error(_lagrange_eval(x, y, t), truth),
                    "converged": lg.converged,
                }
        ds = Dataset("runge", "runge", x, y, f_true=runge, domain=(-1.0, 1.0))
        ch = chebyshev_interpolation(ds, n_nodes=n + 1)
        u = t  # domain is [-1, 1]
        row["chebyshev_points"] = {
            "error": _sup_error(
                np.polynomial.chebyshev.chebval(u, np.asarray(ch.extra["coefficients"])), truth
            ),
            "converged": ch.converged,
        }
        rows.append(row)
        print(f"Q2 n={n}: done", flush=True)
    # observed orders log2(e_n / e_2n)
    orders: dict[str, list[float | None]] = {}
    for key in [f"fh{d}" for d in D_LIST] + ["spline", "pchip"]:
        errs = [r[key]["error"] for r in rows]
        orders[key] = [
            math.log2(errs[i] / errs[i + 1]) if errs[i] and errs[i + 1] else None
            for i in range(len(errs) - 1)
        ]
    # d sweep
    sweep: dict[str, Any] = {}
    for n in SWEEP_N:
        x = np.linspace(-1.0, 1.0, n + 1)
        y = runge(x)
        ds_ = list(range(0, min(n, SWEEP_D_MAX) + 1))
        errs, lams = [], []
        for d in ds_:
            res = M.floater_hormann((x, y), d=d)
            w = res.extra["coefficients"]
            errs.append(_sup_error(_barycentric_eval(x, w, y, t), truth))
            lams.append(M.lebesgue_constant(x, w, 30))
        k = int(np.argmin(errs))
        sweep[str(n)] = {
            "d": ds_,
            "error": errs,
            "lebesgue": lams,
            "best_d": ds_[k],
            "best_error": errs[k],
        }
    # FH 2007 Table 2 reports 1.3e-15 for n = 160, d = 10. The paper's setting is
    # 1/(1 + x²) on [-5, 5]; its test grid is not stated. Two candidate explanations are
    # measured on equispaced test grids of several sizes: (i) the truncation error, i.e.
    # the same float64 weights and data evaluated in IEEE quad (numpy.longdouble is
    # binary128 on aarch64; on x86 it is 80-bit extended, ~3 digits more), and (ii) the
    # float64 error on a coarser test grid.
    x = np.linspace(-5.0, 5.0, 161)
    y = 1.0 / (1.0 + x * x)
    w = M.floater_hormann((x, y), d=10).extra["coefficients"]
    xq, wq, yq = (np.asarray(v, dtype=np.longdouble) for v in (x, w, y))
    grids = []
    for n_test in TABLE2_GRIDS:
        tt = np.linspace(-5.0, 5.0, n_test)
        e64 = _sup_error(_barycentric_eval(x, w, y, tt), 1.0 / (1.0 + tt * tt))
        inner = ~np.isin(tt, x)  # at a node r = y exactly (the error is the rounding of y)
        tq = tt[inner].astype(np.longdouble)
        e_ld = 0.0
        for start in range(0, tq.size, 20_000):
            tc = tq[start : start + 20_000]
            cq = wq[None, :] / (tc[:, None] - xq[None, :])
            e_ld = max(e_ld, float(np.max(np.abs((cq @ yq) / cq.sum(axis=1) - 1 / (1 + tc * tc)))))
        grids.append(
            {
                "test_points": n_test,
                "shared_with_nodes": int(np.sum(~inner)),
                "float64_error": e64,
                "longdouble_eval_error": e_ld,
            }
        )
    lam10 = M.lebesgue_constant(x, w, 30)
    reconcile = {
        "n": 160,
        "d": 10,
        "interval": [-5.0, 5.0],
        "paper_error": 1.3e-15,
        "grids": grids,
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "lebesgue": lam10,
        "eps_lebesgue": EPS * lam10,
    }
    return {
        "rows": rows,
        "orders": orders,
        "sweep": sweep,
        "reconcile_table2": reconcile,
        "test_points": int(t.size),
    }


# =======================================================================================
# Figures
# =======================================================================================


def _sqrt_axis(ax: Any) -> None:
    ax.set_xscale(FuncScale(ax, (lambda v: np.sqrt(np.maximum(v, 0)), lambda v: np.square(v))))


def _end_label(ax: Any, x: float, y: float, text: str, color: str, dy: float = 0.0) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(4, dy),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
        va="center",
    )
    ax.plot([x], [y], "o", ms=3.5, color=color, zorder=5)


def fig1(q1r: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), constrained_layout=True)
    order = [
        ("abs", axes[0, 0]),
        ("sqrt", axes[0, 1]),
        ("tanh50", axes[1, 0]),
        ("runge", axes[1, 1]),
    ]
    for key, ax in order:
        e = q1r[key]
        cheb = e["chebyshev"]
        cn = np.array([c["n"] for c in cheb])
        ce = np.maximum([c["test_error"] for c in cheb], 1e-17)
        if key in ("abs", "sqrt"):
            _sqrt_axis(ax)
            xmax = 400
            series = [
                ("clustered/columns", "AAA, clustered Z", C_BLUE, "-"),
                ("clustered/none", "AAA unscaled, clustered Z", C_ORANGE, "--"),
                ("equispaced/columns", "AAA, equispaced Z", C_AQUA, ":"),
            ]
            nn = np.linspace(1, xmax, 400)
            mult = 1.0 if key == "abs" else 2.0
            best_label = (
                r"best: $8e^{-\pi\sqrt{n}}$" if key == "abs" else r"best: $8e^{-\pi\sqrt{2n}}$"
            )
            ax.plot(
                nn,
                8 * np.exp(-math.pi * np.sqrt(mult * nn)),
                color=REF_GRAY,
                lw=1.0,
                ls=(0, (2, 2)),
                label=best_label,
            )
            ax.set_xticks([1, 4, 16, 36, 64, 100, 196, 400])
        else:
            ax.set_xscale("log")
            xmax = 1000
            series = [
                ("equispaced/columns", "AAA", C_BLUE, "-"),
                ("equispaced/none", "AAA unscaled", C_ORANGE, "--"),
            ]
        for run_key, label, color, ls in series:
            steps = e["aaa"][run_key]["steps"]
            ns = np.array([s["n"] for s in steps])
            es = np.maximum([s["test_error"] for s in steps], 1e-17)
            mk = "o" if key in ("tanh50", "runge") else None
            ax.plot(ns, es, color=color, ls=ls, lw=1.8, marker=mk, ms=3, label=label)
        sel = cn <= xmax
        ax.plot(
            cn[sel],
            ce[sel],
            color=C_VIOLET,
            lw=1.8,
            marker="o",
            ms=3,
            label="Chebyshev",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e1 if key != "tanh50" else 1e3)
        ax.set_xlim((0, xmax) if key in ("abs", "sqrt") else (0.8, xmax))
        ax.set_title(e["label"], loc="left")
        ax.set_xlabel(
            "degree n (type (n, n) for AAA)"
            + (", √n axis" if key in ("abs", "sqrt") else ", log axis")
        )
        ax.set_ylabel("max error on test grid")
        if key in ("abs", "sqrt"):
            # empty corner: right of the last AAA step, below the best-approximation line
            ax.legend(loc="lower right", fontsize=8)
        else:
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Q1  AAA vs Chebyshev interpolation: error against degree",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.savefig(FIGURES / "fig1_aaa_vs_chebyshev.svg")
    plt.close(fig)


def fig2(q1r: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    # (a) poles of the final AAA approximant of √x (clustered Z, column scaling)
    ax = axes[0]
    run = q1r["sqrt"]["aaa"]["clustered/columns"]
    poles = np.array([complex(*p) for p in run["final_poles"]])
    res = np.array([complex(*r) for r in run["final_residues"]])
    dbl = np.abs(res) < M.DOUBLET_RTOL
    unres = np.array(run["final_pole_rel_residual"]) > POLE_RESOLVED_RTOL
    ang = np.abs(np.angle(poles)) / np.pi
    mag = np.abs(poles)
    ax.scatter(
        mag[~unres],
        ang[~unres],
        s=22,
        color=C_BLUE,
        edgecolor=SURFACE,
        linewidth=0.8,
        label="pole (resolved)",
        zorder=3,
    )
    ax.scatter(
        mag[unres],
        ang[unres],
        s=26,
        marker="x",
        color=REF_GRAY,
        linewidth=1.1,
        label=f"estimate not resolved (rel. residual of d > {POLE_RESOLVED_RTOL:.0e})",
        zorder=3,
    )
    ax.scatter(
        mag[dbl],
        ang[dbl],
        s=48,
        facecolor="none",
        edgecolor=C_ORANGE,
        linewidth=1.2,
        label="|residue| < 1e-13 (doublet)",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("|λ|")
    ax.set_ylabel("|arg λ| / π   (1 = negative real axis)")
    ax.set_title(
        f"(a) √x, final AAA type ({run['n_support'] - 1}, {run['n_support'] - 1}): poles",
        loc="left",
    )
    ax.legend(loc="center right", fontsize=7.5)
    # (b) real poles inside the interval per step
    ax = axes[1]
    cfg = [
        ("abs", "equispaced/columns", "|x|, equispaced Z", C_AQUA, ":"),
        ("abs", "clustered/columns", "|x|, clustered Z", C_BLUE, "-"),
        ("sqrt", "clustered/columns", "√x, clustered Z", C_MAGENTA, "-"),
        ("sqrt", "clustered/none", "√x, clustered Z, no scaling", C_ORANGE, "--"),
    ]
    for key, run_key, label, color, ls in cfg:
        steps = q1r[key]["aaa"][run_key]["steps"]
        ax.step(
            [s["n"] for s in steps],
            [s["n_interval_poles"] for s in steps],
            where="post",
            color=color,
            ls=ls,
            lw=1.6,
            label=label,
        )
    ax.set_xlabel("degree n")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("certified real poles in the interval")
    ax.set_title("(b) certified real poles in the interval, per step", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(FIGURES / "fig2_aaa_poles.svg")
    plt.close(fig)


def fig3(q2r: dict[str, Any]) -> None:
    rows = q2r["rows"]
    ns = np.array([r["n"] for r in rows])
    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    for i, d in enumerate(D_LIST):
        es = np.array([r[f"fh{d}"]["error"] for r in rows])
        ax.plot(
            ns, es, color=BLUE_RAMP[i], lw=1.8, marker="o", ms=3.5, label=f"Floater–Hormann d = {d}"
        )

    def series(key: str) -> tuple[np.ndarray, np.ndarray]:
        pts = [(r["n"], r[key]["error"]) for r in rows if key in r and r[key]["error"] is not None]
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    for key, label, color, ls, mk in [
        ("spline", "not-a-knot cubic spline", C_ORANGE, "-", "s"),
        ("pchip", "PCHIP", C_AQUA, "-", "^"),
        ("lagrange", "polynomial, equispaced (Lagrange form)", C_RED, "-", "D"),
        ("chebyshev_points", "polynomial at Chebyshev points (other samples)", C_VIOLET, "--", "o"),
    ]:
        x_, y_ = series(key)
        ax.plot(
            x_, np.maximum(y_, 1e-17), color=color, ls=ls, marker=mk, ms=3.5, lw=1.6, label=label
        )
    e3 = rows[-1]["fh3"]["error"]
    ref4 = e3 * (ns / ns[-1]) ** (-4.0)
    ax.plot(ns[2:], ref4[2:], color=REF_GRAY, lw=1.0, ls=(0, (2, 2)))
    ax.annotate(
        "slope h⁴",
        (ns[3], ref4[3]),
        xytext=(4, 6),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
    )
    e8 = rows[3]["fh8"]["error"]
    ref9 = e8 * (ns / ns[3]) ** (-9.0)
    ax.plot(ns[2:6], ref9[2:6], color=REF_GRAY, lw=1.0, ls=(0, (2, 2)))
    ax.annotate(
        "slope h⁹",
        (ns[4], ref9[4]),
        xytext=(6, -2),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
    )
    ax.axhline(EPS, color=REF_GRAY, lw=0.8)
    ax.annotate(
        "machine ε",
        (ns[-1], EPS),
        xytext=(-48, -11),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e4)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n  (n + 1 equispaced samples on [-1, 1])")
    ax.set_ylabel("max error on 100 001 test points")
    ax.set_title("Q2  Runge 1/(1+25x²): Floater–Hormann vs polynomial and splines", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.savefig(FIGURES / "fig3_fh_runge.svg")
    plt.close(fig)


def fig4(q2r: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    colors = {"20": C_ORANGE, "40": C_AQUA, "80": C_BLUE, "160": C_VIOLET}
    for n, color in colors.items():
        sw = q2r["sweep"][n]
        d = np.array(sw["d"])
        axes[0].plot(
            d,
            np.maximum(sw["error"], 1e-17),
            color=color,
            lw=1.8,
            marker="o",
            ms=3,
            label=f"n = {n}",
        )
        axes[0].plot(d, EPS * np.array(sw["lebesgue"]), color=color, lw=1.0, ls=(0, (2, 2)))
        axes[1].plot(d, sw["lebesgue"], color=color, lw=1.8, marker="o", ms=3, label=f"n = {n}")
    dd = np.arange(1, 31)
    axes[1].plot(dd, 2.0 ** (dd - 1) * (2 + math.log(160)), color=REF_GRAY, lw=1.0, ls=(0, (2, 2)))
    axes[1].annotate(
        r"upper bound $2^{d-1}(2 + \ln 160)$",
        (13, 2.0**12 * (2 + math.log(160))),
        xytext=(-8, 8),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
        ha="right",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-17, 1e2)
    axes[0].set_xlabel("blending degree d")
    axes[0].set_ylabel("max error")
    axes[0].set_title("(a) error vs d (dashed: ε·Λ rounding floor)", loc="left")
    axes[0].legend(loc="lower right")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("blending degree d")
    axes[1].set_ylabel("Lebesgue constant Λ")
    axes[1].set_title("(b) Lebesgue constant vs d", loc="left")
    axes[1].legend(loc="upper left")
    fig.suptitle(
        "Q2  Choosing d for Runge on n + 1 equispaced points",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.savefig(FIGURES / "fig4_fh_degree_sweep.svg")
    plt.close(fig)


# =======================================================================================
# Tables (markdown, pasted into README.md)
# =======================================================================================


def _e(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "non-finite"
    return f"{v:.1e}"


def tables(q1r: dict[str, Any], q2r: dict[str, Any], depth: list[dict[str, Any]]) -> str:
    out: list[str] = []
    out.append("### Q1 rate fits (ln E = α − C·g(n); RMS residual in decades)\n")
    out.append(
        "| function | approximant | window n | algebraic C (1st half, 2nd half) | RMS | root-exp C (1st half, 2nd half) | RMS |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for key in ("abs", "sqrt"):
        fits = q1r[key]["fits"]
        for name in (
            "chebyshev",
            "aaa clustered/columns",
            "aaa clustered/none",
            "aaa equispaced/columns",
        ):
            a, r = fits.get(f"{name}/algebraic"), fits.get(f"{name}/root_exp")
            if a is None or r is None:
                continue
            out.append(
                f"| {q1r[key]['label']} | {name} | {a['n_lo']}–{a['n_hi']} | {a['C']:.2f} ({a['C_first_half']:.2f}, {a['C_second_half']:.2f}) | {a['rms_decades']:.2f} | {r['C']:.2f} ({r['C_first_half']:.2f}, {r['C_second_half']:.2f}) | {r['rms_decades']:.2f} |"
            )
    out.append("\n### Q1 AAA runs (final step and best step on the test grid)\n")
    out.append(
        "| function | Z | scaling | M | m | converged | final sample err | final test err | best test err (n) | real poles in interval (final) | doublets (final) | unresolved pole estimates (final) | s |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _key, e in q1r.items():
        for rk, run in e["aaa"].items():
            zname, sc = rk.split("/")
            steps = run["steps"]
            te = [s["test_error"] for s in steps]
            k = int(np.argmin(te))
            out.append(
                f"| {e['label']} | {zname} | {sc} | {run['M']} | {run['n_support']} | {run['converged']} | {_e(steps[-1]['sample_error'])} | {_e(steps[-1]['test_error'])} | {_e(te[k])} ({steps[k]['n']}) | {steps[-1]['n_interval_poles']} | {steps[-1]['n_doublets']} | {sum(v > POLE_RESOLVED_RTOL for v in run['final_pole_rel_residual'])} | {run['seconds']:.1f} |"
            )
    out.append(
        f"\nUnresolved pole estimate: relative residual |d(λ)|/Σ|w_j/(λ - z_j)| > {POLE_RESOLVED_RTOL:.0e}. "
        "Real poles in the interval: certified zeros of d (sign changes), positions:"
    )
    for _key, e in q1r.items():
        for rk, run in e["aaa"].items():
            if run["final_interval_poles"]:
                pos = ", ".join(f"{v:.3g}" for v in run["final_interval_poles"][:6])
                more = len(run["final_interval_poles"]) - 6
                out.append(
                    f"* {e['label']}, {rk}: {pos}" + (f" (+{more} more)" if more > 0 else "")
                )
    out.append("\n### Q1 root-exponential fit sensitivity (AAA, clustered Z, columns)\n")
    out.append("| function | window n | root-exp C (1st half, 2nd half) | RMS |")
    out.append("|---|---|---|---|")
    for key in ("abs", "sqrt"):
        fits = q1r[key]["fits"]
        for name in ("root_exp", "root_exp_10_60", "root_exp_to_best"):
            r = fits.get(f"aaa clustered/columns/{name}")
            if r is None:
                continue
            out.append(
                f"| {q1r[key]['label']} | {r['n_lo']}–{r['n_hi']} | {r['C']:.2f} ({r['C_first_half']:.2f}, {r['C_second_half']:.2f}) | {r['rms_decades']:.2f} |"
            )
    out.append(
        "\n### Q1 AAA error / best-approximation asymptotics 8e^{-C√n} (clustered Z, columns)\n"
    )
    for key in ("abs", "sqrt"):
        rt = q1r[key]["ratio_to_best"]
        out.append(
            f"* {q1r[key]['label']} (C = {q1r[key]['best_C']:.4f}): "
            + ", ".join(f"n={n}: {v:.1f}" for n, v in rt.items())
        )
    out.append(
        "\n### Q1 √x clustering depth (Z = 4000 equispaced ∪ 2000 log-spaced in [10^-depth, 1]; test grid down to 1e-32)\n"
    )
    out.append(
        "| depth | scaling | m | converged | final sample err | final test err | best test err (n) | √(smallest sample) |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for r in depth:
        out.append(
            f"| 1e-{r['depth']} | {r['scaling']} | {r['m']} | {r['converged']} | {_e(r['sample_error'])} | {_e(r['test_error'])} | {_e(r['best_test_error'])} ({r['best_n']}) | {_e(r['sqrt_smallest_sample'])} |"
        )
    out.append("\n### Q1 Chebyshev interpolation, test error at selected degrees\n")
    sel = [10, 20, 40, 100, 200, 398, 1000]
    out.append("| function | " + " | ".join(f"n={n}" for n in sel) + " |")
    out.append("|---|" + "---|" * len(sel))
    for _key, e in q1r.items():
        d = {c["n"]: c["test_error"] for c in e["chebyshev"]}
        out.append(f"| {e['label']} | " + " | ".join(_e(d.get(n)) for n in sel) + " |")
    out.append(
        "\n### Q1 smallest degree n with test error ≤ level (— = never, within n ≤ 99 for AAA, n ≤ 1000 for Chebyshev; every n checked)\n"
    )
    out.append(
        "| function | level | Chebyshev | "
        + " | ".join(
            f"AAA {k}"
            for k in (
                "clustered/columns",
                "clustered/none",
                "equispaced/columns",
                "equispaced/none",
            )
        )
        + " |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for _key, e in q1r.items():
        for level, row in e["degree_to_reach"].items():
            cells = [
                row.get(f"aaa {k}", "n/a")
                for k in (
                    "clustered/columns",
                    "clustered/none",
                    "equispaced/columns",
                    "equispaced/none",
                )
            ]
            cells = ["n/a" if c == "n/a" else ("—" if c is None else str(c)) for c in cells]
            ch = row["chebyshev"]
            out.append(
                f"| {e['label']} | {level} | {'—' if ch is None else ch} | "
                + " | ".join(cells)
                + " |"
            )
    out.append("\nChebyshev scan check (every n ≤ 1000; first degree of the log grid in brackets):")
    for _key, e in q1r.items():
        cr = e["chebyshev_reach"]
        cells = ", ".join(
            f"{lv}: {cr['first_n'][lv]} ({cr['first_grid_n'][lv]})" for lv in cr["first_n"]
        )
        out.append(
            f"* {e['label']}: {cells}; DCT vs numopt coefficients max |diff| "
            f"{cr['max_coef_diff_vs_numopt']:.1e}; full-grid evaluations {cr['full_grid_evaluations']}"
        )
    for key in ("tanh50", "runge"):
        g = q1r[key]["fits"]["chebyshev/geometric"]
        out.append(
            f"\nChebyshev geometric fit {key}: ρ_fit = {g['rho_fit']:.4f}, ρ_theory = {g['rho_theory']:.4f}, n = {g['n_lo']}–{g['n_hi']}, RMS = {g['rms_decades']:.2f} decades"
        )
    rows = q2r["rows"]
    out.append("\n### Q2 Runge on n + 1 equispaced points: max error\n")
    cols = [f"fh{d}" for d in D_LIST] + [
        "spline",
        "pchip",
        "barycentric",
        "lagrange",
        "chebyshev_points",
    ]
    out.append("| n | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            if c not in r:
                cells.append("—")
            elif r[c]["error"] is None:
                cells.append("overflow")
            else:
                cells.append(_e(r[c]["error"]))
        out.append(f"| {r['n']} | " + " | ".join(cells) + " |")
    out.append("\n### Q2 observed order log2(e_n / e_2n), n → 2n\n")
    keys = list(q2r["orders"].keys())
    out.append("| n → 2n | " + " | ".join(keys) + " |")
    out.append("|---|" + "---|" * len(keys))
    for i in range(len(rows) - 1):
        cells = [
            ("—" if q2r["orders"][k][i] is None else f"{q2r['orders'][k][i]:.1f}") for k in keys
        ]
        out.append(f"| {rows[i]['n']}→{rows[i + 1]['n']} | " + " | ".join(cells) + " |")
    out.append("\n### Q2 Lebesgue constant Λ (FH, equispaced)\n")
    out.append("| n | " + " | ".join(f"Λ d={d}" for d in D_LIST) + " |")
    out.append("|---|" + "---|" * len(D_LIST))
    for r in rows:
        out.append(
            f"| {r['n']} | " + " | ".join(f"{r[f'fh{d}']['lebesgue']:.1f}" for d in D_LIST) + " |"
        )
    last = rows[-1]
    out.append(f"\n### Q2 rounding floor at n = {last['n']}: error vs ε·Λ·max|f|\n")
    out.append("| d | error | ε·Λ·max|f| | error / (ε·Λ·max|f|) |")
    out.append("|---|---|---|---|")
    for d in D_LIST:
        c = last[f"fh{d}"]
        out.append(f"| {d} | {_e(c['error'])} | {_e(c['floor'])} | {c['error'] / c['floor']:.2f} |")
    sw = q2r["sweep"]["160"]
    dd = np.array(sw["d"][10:])
    slope = float(np.polyfit(dd, np.log2(np.array(sw["lebesgue"][10:])), 1)[0])
    out.append(
        f"\nΛ growth for n = 160, d = 10..30: log2 Λ grows by {slope:.2f} per unit d "
        f"(Λ = {sw['lebesgue'][10]:.3g} at d = 10, {sw['lebesgue'][20]:.3g} at d = 20, {sw['lebesgue'][30]:.3g} at d = 30)"
    )
    rc = q2r["reconcile_table2"]
    out.append(
        f"\n### FH 2007 Table 2, n = {rc['n']}, d = {rc['d']}, 1/(1+x²) on [-5, 5]: paper {_e(rc['paper_error'])}; "
        f"Λ = {rc['lebesgue']:.1f}, ε·Λ = {_e(rc['eps_lebesgue'])}; long double eps {rc['longdouble_eps']:.1e}\n"
    )
    out.append(
        "| equispaced test points | shared with nodes | float64 error | same weights in long double (truncation) |"
    )
    out.append("|---|---|---|---|")
    for g in rc["grids"]:
        out.append(
            f"| {g['test_points']} | {g['shared_with_nodes']} | {_e(g['float64_error'])} | {_e(g['longdouble_eval_error'])} |"
        )
    t3 = q2r["diagnostics"]["fh_table3_spline"]
    out.append("\n### FH 2007 Table 3 spline column (1/(1+x²) on [-5, 5], 100 001 test points)\n")
    out.append("| n | paper (clamped) | clamped (SciPy) | not-a-knot (numopt) |")
    out.append("|---|---|---|---|")
    for r in t3:
        out.append(
            f"| {r['n']} | {_e(r['paper_clamped'])} | {_e(r['clamped_scipy'])} | {_e(r['not_a_knot_numopt'])} |"
        )
    dg = q2r["diagnostics"]
    lw, pt = dg["loewner_sqrt_clustered_m24"], dg["poles_tanh50"]
    out.append(
        f"\nDiagnostics: √x clustered, unscaled AAA at m = 24: column-norm ratio "
        f"{lw['column_norm_ratio']:.1e}, κ(A) = {lw['cond']:.1e}, κ(AD) = {lw['cond_column_scaled']:.1e}. "
        f"tanh(50x) poles ({pt['count']}), relative error vs long-double Newton roots: numpy route "
        f"max {pt['max_rel_err_numpy_route']:.1e} / median {pt['median_rel_err_numpy_route']:.1e}; "
        f"QZ max {pt['max_rel_err_qz']:.1e} / median {pt['median_rel_err_qz']:.1e}. "
        f"Degree-80 equispaced polynomial for Runge in long double: max error "
        f"{dg['poly_runge_n80_longdouble_error']:.3g}."
    )
    out.append("\n### Q2 best d (d ≤ min(n, 30))\n")
    out.append("| n | best d | error | Λ at best d | error at d = 3 | error at d = 8 |")
    out.append("|---|---|---|---|---|---|")
    for n, sw in q2r["sweep"].items():
        k = sw["d"].index(sw["best_d"])
        e3 = sw["error"][3] if len(sw["error"]) > 3 else None
        e8 = sw["error"][8] if len(sw["error"]) > 8 else None
        out.append(
            f"| {n} | {sw['best_d']} | {_e(sw['best_error'])} | {sw['lebesgue'][k]:.1f} | {_e(e3)} | {_e(e8)} |"
        )
    return "\n".join(out) + "\n"


def diagnostics() -> dict[str, Any]:
    """Numbers quoted in method.py NOTEs and in the README discussion.

    * Loewner conditioning for the clustered √x samples, unscaled run, at m = 24.
    * Pole accuracy on tanh(50x) (unscaled AAA): numopt-style poles (shift-invert +
      Newton) and SciPy QZ on the pencil (3.11), both against Newton roots of d(λ)
      computed in long double.
    * The degree-80 equispaced polynomial interpolant of Runge, evaluated in long double
      with long-double nodes (the reference for the float64 Lagrange / barycentric forms).
    """
    import scipy.linalg  # test oracle only (not used by method.py)

    out: dict[str, Any] = {"longdouble_eps": float(np.finfo(np.longdouble).eps)}
    prob = q1_problems()["sqrt"]
    z = prob["samples"]["clustered"]
    f = np.sqrt(z)
    res = M.aaa((z, f), scaling="none", max_terms=24)
    zs = np.asarray(res.extra["nodes"])
    fs = np.asarray(res.extra["values"])
    free = ~np.isin(z, zs)
    loewner = (f[free, None] - fs[None, :]) / (z[free, None] - zs[None, :])
    cn = np.linalg.norm(loewner, axis=0)
    sv = np.linalg.svd(loewner, compute_uv=False)
    sv_scaled = np.linalg.svd(loewner / cn, compute_uv=False)
    out["loewner_sqrt_clustered_m24"] = {
        "column_norm_ratio": float(cn.max() / cn.min()),
        "cond": float(sv[0] / sv[-1]),
        "cond_column_scaled": float(sv_scaled[0] / sv_scaled[-1]),
    }
    x = np.linspace(-1.0, 1.0, 2000)
    r = M.aaa((x, np.tanh(50.0 * x)), scaling="none")
    zz = np.asarray(r.extra["nodes"])
    ww = np.asarray(r.extra["coefficients"])
    ff = np.asarray(r.extra["values"])
    mine, _ = M.barycentric_poles(zz, ww, ff)
    m = zz.size
    B = np.eye(m + 1)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1))
    E[0, 1:] = ww
    E[1:, 0] = 1.0
    np.fill_diagonal(E[1:, 1:], zz)
    qz = scipy.linalg.eigvals(E, B)
    qz = qz[np.isfinite(qz)]
    zq, wq = zz.astype(np.longdouble), ww.astype(np.longdouble)
    err_mine, err_qz = [], []
    for p in mine:
        lam = np.clongdouble(p)
        for _ in range(60):
            lam = lam - np.sum(wq / (lam - zq)) / (-np.sum(wq / (lam - zq) ** 2))
        ref = complex(lam)
        err_mine.append(abs(p - ref) / abs(ref))
        err_qz.append(float(np.min(np.abs(qz - ref))) / abs(ref))
    out["poles_tanh50"] = {
        "count": len(err_mine),
        "max_rel_err_numpy_route": max(err_mine),
        "median_rel_err_numpy_route": float(np.median(err_mine)),
        "max_rel_err_qz": max(err_qz),
        "median_rel_err_qz": float(np.median(err_qz)),
    }
    n = 80
    ld = np.longdouble
    xq = np.linspace(ld(-1), ld(1), n + 1)
    yq = 1 / (1 + 25 * xq * xq)
    wpoly = np.array([(-1) ** j * math.comb(n, j) for j in range(n + 1)], dtype=ld)
    t = np.linspace(-1.0, 1.0, 100_001)
    tq = t.astype(ld)
    inner = ~np.isin(tq, xq)
    c = wpoly[None, :] / (tq[inner, None] - xq[None, :])
    out["poly_runge_n80_longdouble_error"] = float(
        np.max(np.abs((c @ yq) / c.sum(axis=1) - 1 / (1 + 25 * tq[inner] ** 2)))
    )
    # FH 2007 Table 3 (clamped cubic spline, Runge 1/(1+x²) on [-5, 5]) with SciPy's
    # CubicSpline, next to numopt's not-a-knot spline on the same data.
    from scipy.interpolate import CubicSpline

    t5 = np.linspace(-5.0, 5.0, 100_001)
    f5 = 1.0 / (1.0 + t5 * t5)
    dfa = 2.0 * 5.0 / (1.0 + 25.0) ** 2  # f'(-5) = 2·5/26², f'(5) = -f'(-5)
    rows = []
    for n_, pub in FH_TABLE3_SPLINE.items():
        x5 = np.linspace(-5.0, 5.0, n_ + 1)
        y5 = 1.0 / (1.0 + x5 * x5)
        cs = CubicSpline(x5, y5, bc_type=((1, dfa), (1, -dfa)))  # pyright: ignore[reportArgumentType]
        nak = cubic_spline_not_a_knot((x5, y5))
        rows.append(
            {
                "n": n_,
                "paper_clamped": pub,
                "clamped_scipy": _sup_error(cs(t5), f5),
                "not_a_knot_numopt": _sup_error(
                    _pp_eval(x5, np.asarray(nak.extra["coefficients"]), t5), f5
                ),
            }
        )
    out["fh_table3_spline"] = rows
    return out


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    q1r = q1()
    depth = q1_depth_sweep()
    q2r = q2()
    q2r["diagnostics"] = diagnostics()
    meta = {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "eps": EPS}
    (RESULTS / "q1_aaa_vs_chebyshev.json").write_text(
        json.dumps(to_jsonable({"meta": meta, **q1r, "depth_sweep": depth}), indent=1)
    )
    (RESULTS / "q2_floater_hormann_runge.json").write_text(
        json.dumps(to_jsonable({"meta": meta, **q2r}), indent=1)
    )
    (RESULTS / "tables.md").write_text(tables(q1r, q2r, depth))
    fig1(q1r)
    fig2(q1r)
    fig3(q2r)
    fig4(q2r)
    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
