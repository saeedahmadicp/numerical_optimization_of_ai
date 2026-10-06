"""Clenshaw–Curtis vs Gauss: a reproducible study of Trefethen (2008), "Is Gauss quadrature
better than Clenshaw–Curtis?" (SIAM Review 50(1):67–87).

Run:  .venv/bin/python research/clenshaw-curtis-vs-gauss/run.py      (< 1 min, no network)

Experiments (all deterministic; there is no randomness anywhere in this study):
  E1  Reproduce the paper: Fig. 2 (six integrands, n = 1..30), Fig. 3 (|x + ½|^½, log–log),
      Fig. 7 (1/(1 + 16x²) to n = 120) and the kink location of Weideman & Trefethen (2007).
  E2  The research question: the efficiency ratio R(ε) = (n_CC(ε) + 1)/(n_G(ε) + 1) of the number
      of points that Clenshaw–Curtis and Gauss need for a relative error ≤ ε, on an extended
      test set (Trefethen's set + a family 1/(1 + α²x²) whose Bernstein parameter ρ is known +
      oscillatory entire functions + |x|^k).
  E3  Practical comparison with error estimates on a common budget of 4097 evaluations:
      two Gauss–Kronrod baselines (nested Gauss–Kronrod–Patterson with CC's stopping test, and
      QUADPACK's adaptive G10/K21 QAGS via scipy), a non-nested Gauss-doubling construct, and the
      numopt baselines romberg, simpson, adaptive_simpson (and gauss_legendre, reported but kept
      out of the profiles because its cost is cumulative over every rule m = 1..n).
      Dolan–Moré performance profiles and convergence plots.

Outputs: results/*.json and figures/*.svg.
"""

from __future__ import annotations

import os

# NOTE: one BLAS thread. numopt's Golub–Welsch rule calls a dense eigh; on a loaded multi-core
# machine the threaded BLAS made the m ≤ 90 rules 80× slower (4.3 s vs 0.05 s), with equal output.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from functools import cache  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from numopt import bench, problems  # noqa: E402
from numopt.core.registry import run as run_method  # noqa: E402
from numopt.core.types import Problem, Result  # noqa: E402
from numopt.integration.methods import gauss_legendre_rule  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load("research_cc_vs_gauss_method", HERE / "method.py")
gk = _load("research_cc_vs_gauss_baselines", HERE / "baselines.py")

# --------------------------------------------------------------------------------------
# Pre-registered constants (fixed before the first run of E2/E3; not independently
# timestamped: the study folder had no commit history when the first run was made)
# --------------------------------------------------------------------------------------

#: Relative-error targets ε for the efficiency ratio R(ε).
EPS_LEVELS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-13)
#: R ≥ GAP counts as "the factor of 2 is visible"; R ≤ EQUAL as "essentially equal".
GAP, EQUAL = 1.8, 1.25
#: n grid for E1/E2: every n ≤ 256, then every 4th n up to 1024 (n + 1 points).
N_GRID = np.concatenate([np.arange(1, 257), np.arange(260, 1025, 4)])
#: E3: evaluation budget per run (= 2^12 + 1, the point count of CC/Romberg/Simpson at 2^12),
#: tolerances, and success test |I_est − I| ≤ tol·max(1, |I|) (numopt's acceptance scale).
BUDGET = 4097
TOLS = (1e-6, 1e-10, 1e-13)
#: Pole family 1/(1 + α²x²): poles ±i/α, Bernstein parameter ρ = 1/α + √(1 + 1/α²).
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

# Colours: numopt.bench's Okabe–Ito order, so a method has one colour in every figure.
C_GAUSS, C_CC = "#0072B2", "#D55E00"
METHOD_STYLE = {  # label: (colour, marker, linestyle)
    "clenshaw_curtis": ("#D55E00", "s", "-"),
    "gauss_patterson": ("#0072B2", "o", "-"),
    "quadpack_qags": ("#000000", "X", "--"),
    "gauss_doubling": ("#009E73", "^", "--"),
    "romberg": ("#CC79A7", "D", "-."),
    "simpson": ("#E69F00", "v", ":"),
    "adaptive_simpson": ("#56B4E9", "P", "--"),
    "gauss_legendre": ("#8c8c8c", "o", ":"),
}
#: Methods in the performance profiles and the "best" counts. numopt's gauss_legendre is run and
#: reported but kept out: its cost is the cumulative cost of every rule m = 1..n (≈ m²/2) and its
#: test takes the max of three differences, so it measures that implementation, not the rule.
PROFILE_METHODS = [m for m in METHOD_STYLE if m != "gauss_legendre"]
INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#d9d9d9"
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "cc-vs-gauss",  # deterministic SVG ids
    }
)

# --------------------------------------------------------------------------------------
# Integrands on [−1, 1]
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Integrand:
    id: str
    latex: str
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    exact: float
    kind: str  # polynomial | entire | analytic | C-infinity | finite smoothness
    rho: float | None = None  # Bernstein parameter of the largest ellipse of analyticity

    def problem(self) -> Problem:
        f = self.f
        return Problem(
            id=self.id,
            name=self.id,
            latex=self.latex,
            f=lambda x: float(f(np.asarray(x, dtype=np.float64))),
            dim=1,
            domain=(-1.0, 1.0),
            exact=self.exact,
        )


def _exp_inv_sq(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    safe = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 0.0, np.exp(-1.0 / (safe * safe)))


def _pole(alpha: float) -> Integrand:
    y = 1.0 / alpha
    return Integrand(
        id=f"pole_a{alpha:g}",
        latex=rf"$1/(1+{alpha:g}^2x^2)$",
        f=lambda x, a=alpha: 1.0 / (1.0 + (a * x) ** 2),
        exact=2.0 * math.atan(alpha) / alpha,
        kind="analytic",
        rho=y + math.sqrt(1.0 + y * y),
    )


# Trefethen (2008) Fig. 2 (six functions) and Fig. 3 (one function).
TREFETHEN = [
    Integrand("x20", r"$x^{20}$", lambda x: x**20, 2.0 / 21.0, "polynomial"),
    Integrand("exp", r"$e^{x}$", np.exp, 2.0 * math.sinh(1.0), "entire"),
    Integrand(
        "exp_neg_x2",
        r"$e^{-x^2}$",
        lambda x: np.exp(-x * x),
        math.sqrt(math.pi) * math.erf(1.0),
        "entire",
    ),
    Integrand(
        "runge16",
        r"$1/(1+16x^2)$",
        lambda x: 1.0 / (1.0 + 16.0 * x * x),
        0.5 * math.atan(4.0),
        "analytic",
        rho=0.25 + math.sqrt(1.0 + 1.0 / 16.0),
    ),
    Integrand(
        "exp_inv_x2",
        r"$e^{-1/x^2}$",
        _exp_inv_sq,
        2.0 * (math.exp(-1.0) - math.sqrt(math.pi) * math.erfc(1.0)),
        "C-infinity",
    ),
    Integrand("abs_x3", r"$|x|^3$", lambda x: np.abs(x) ** 3, 0.5, "finite smoothness"),
    Integrand(
        "sqrt_abs_x_half",
        r"$|x+\frac{1}{2}|^{1/2}$",
        lambda x: np.sqrt(np.abs(x + 0.5)),
        (2.0 / 3.0) * (0.5**1.5 + 1.5**1.5),
        "finite smoothness",
    ),
]
POLES = [_pole(a) for a in ALPHAS]
EXTRA = [
    Integrand("cos10", r"$\cos 10x$", lambda x: np.cos(10.0 * x), 0.2 * math.sin(10.0), "entire"),
    Integrand("cos50", r"$\cos 50x$", lambda x: np.cos(50.0 * x), 0.04 * math.sin(50.0), "entire"),
    Integrand("abs_x", r"$|x|$", np.abs, 1.0, "finite smoothness"),
    Integrand("abs_x5", r"$|x|^5$", lambda x: np.abs(x) ** 5, 1.0 / 3.0, "finite smoothness"),
]
ALL = TREFETHEN + POLES + EXTRA

# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------

#: numopt's Golub–Welsch rule is used up to this many points (dense eigh, O(m³)).
GW_MAX = 128


def _legendre_newton(m: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """m-point Gauss–Legendre rule by Newton on P_m (three-term recurrence), for m > GW_MAX.

    Start: Tricomi's x_i ≈ cos(π(i − ¼)/(m + ½)); 8 Newton steps; w_i = 2/((1 − x_i²) P_m'(x_i)²).
    Checked in :func:`check_gauss_reference` against numopt's rule and a binary128 refinement.
    """
    i = np.arange(1, m + 1, dtype=np.float64)
    x = np.cos(np.pi * (i - 0.25) / (m + 0.5))  # (m,) decreasing
    dp = np.ones_like(x)
    for _ in range(8):
        p0, p1 = np.ones_like(x), x.copy()
        for k in range(2, m + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = m * (x * p1 - p0) / (x * x - 1.0)
        x = x - p1 / dp
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    return x[::-1].copy(), w[::-1].copy()


@cache
def gauss_rule(m: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes/weights of the m-point Gauss–Legendre rule on [−1, 1] (numopt's for m ≤ GW_MAX)."""
    return gauss_legendre_rule(m) if m <= GW_MAX else _legendre_newton(m)


@cache
def cc_rule(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return cc.clenshaw_curtis_rule(n)


def check_gauss_reference() -> dict[str, Any]:
    """Max |Δw|, |Δx| of the Newton rule vs numopt (m ≤ 128) and vs a binary128 Newton (m > 128)."""
    out: dict[str, Any] = {"vs_numopt_golub_welsch": {}, "vs_binary128_newton": {}}
    for m in (65, 96, 128):
        x1, w1 = gauss_legendre_rule(m)
        x2, w2 = _legendre_newton(m)
        out["vs_numopt_golub_welsch"][m] = {
            "max_dx": float(np.max(np.abs(x1 - x2))),
            "max_dw": float(np.max(np.abs(w1 - w2))),
        }
    ld = np.longdouble
    if np.finfo(ld).eps < 1e-30:  # binary128 available (e.g. aarch64 Linux)
        for m in (257, 513, 1025):
            x, w = _legendre_newton(m)
            xq = x.astype(ld)
            dp = np.ones_like(xq)
            for _ in range(2):
                p0, p1 = np.ones_like(xq), xq.copy()
                for k in range(2, m + 1):
                    p0, p1 = p1, ((2 * k - 1) * xq * p1 - (k - 1) * p0) / k
                dp = m * (xq * p1 - p0) / (xq * xq - 1)
                xq = xq - p1 / dp
            wq = 2 / ((1 - xq * xq) * dp * dp)
            out["vs_binary128_newton"][m] = {
                "max_dx": float(np.max(np.abs(x - xq))),
                "max_dw": float(np.max(np.abs(w - wq))),
            }
            if m == 257:
                # Side check: numopt's Golub–Welsch weights and scipy's roots_legendre, both at
                # m = 256 vs a binary128 Newton refinement of numopt's nodes (absolute / relative).
                from scipy.special import roots_legendre

                m2 = 256
                xg, wg = gauss_legendre_rule(m2)
                _, ws = roots_legendre(m2)
                xq = xg.astype(ld)
                for _ in range(3):
                    p0, p1 = np.ones_like(xq), xq.copy()
                    for k in range(2, m2 + 1):
                        p0, p1 = p1, ((2 * k - 1) * xq * p1 - (k - 1) * p0) / k
                    dp = m2 * (xq * p1 - p0) / (xq * xq - 1)
                    xq = xq - p1 / dp
                wq = 2 / ((1 - xq * xq) * dp * dp)
                out["weights_m256_vs_binary128"] = {
                    name: {
                        "max_abs": float(np.max(np.abs(wv - wq))),
                        "max_rel": float(np.max(np.abs(wv - wq) / wq)),
                    }
                    for name, wv in (("numopt_golub_welsch", wg), ("scipy_roots_legendre", ws))
                }
    return out


def quad(rule: tuple[NDArray[np.float64], NDArray[np.float64]], f: Callable) -> float:
    t, w = rule
    return math.fsum((w * f(t)).tolist())


# --------------------------------------------------------------------------------------
# E1 + E2: error curves and efficiency ratios
# --------------------------------------------------------------------------------------


def error_curves(fn: Integrand, ns: NDArray[np.int64]) -> tuple[list[float], list[float]]:
    """Absolute errors |I − I_n| of Gauss (n + 1 points) and Clenshaw–Curtis (n + 1 points)."""
    eg = [abs(quad(gauss_rule(int(n) + 1), fn.f) - fn.exact) for n in ns]
    ec = [abs(quad(cc_rule(int(n)), fn.f) - fn.exact) for n in ns]
    return eg, ec


def n_needed(ns: NDArray[np.int64], rel_err: NDArray[np.float64], eps: float) -> int | None:
    """Smallest grid n with rel_err(n') ≤ ε for every grid n' ≥ n (None if not reached at n_max)."""
    above = np.flatnonzero(rel_err > eps)
    if above.size == 0:
        return int(ns[0])
    if above[-1] == ns.size - 1:
        return None
    return int(ns[above[-1] + 1])


def l1_norm(fn: Integrand) -> float:
    """‖f‖₁ = ∫₋₁¹|f| by Clenshaw–Curtis with n = 2^16 (a scale only; ≥ 5 correct digits here)."""
    return quad(cc_rule(2**16), lambda t: np.abs(fn.f(t)))


def efficiency(fn: Integrand, eg: list[float], ec: list[float]) -> dict[str, Any]:
    """Points needed for |I − I_n| ≤ ε·‖f‖₁ (persistently), and their ratio R(ε).

    # NOTE: the first run normalized by |I|. For cos 50x, |I| = 0.0105 ≪ ‖f‖₁ ≈ 1.27, so the
    # target 1e-13·|I| ≈ 1e-15 lay at the rounding floor and gave a spurious R = 0.09. Rounding
    # errors scale with Σ|w_k f(x_k)| ≈ ‖f‖₁, so ‖f‖₁ is the scale on which ε is meaningful.

    Quantization (added after review, post hoc): R is a ratio of small integers. If Gauss needs
    m = n_G + 1 points (degree 2m − 1), the CC rule of equal degree has n_CC = 2m − 2 (even n is
    exact to degree n + 1), i.e. 2m − 1 points, so an exact factor 2 in degree gives
    R = R_max(m) = 2 − 1/m, not 2. Hence R ≥ GAP = 1.8 is reachable only when m ≥ 5. Each level
    therefore also reports R_max and the normalized gap

        g = (p_CC − m)/(m − 1),   p_CC = n_CC + 1,

    which is 0 for equal point counts and 1 for the full degree factor (g = (R − 1)/(1 − 1/m)).
    One point more or less changes g by 1/(m − 1): that is its resolution.
    """
    scale = l1_norm(fn)
    rg = np.array(eg) / scale
    rc = np.array(ec) / scale
    rows = []
    for eps in EPS_LEVELS:
        ng, nc = n_needed(N_GRID, rg, eps), n_needed(N_GRID, rc, eps)
        row: dict[str, Any] = {"eps": eps, "n_gauss": ng, "n_cc": nc, "ratio_points": None}
        if ng is not None and nc is not None:
            m = ng + 1
            row.update(
                {
                    "ratio_points": (nc + 1) / m,
                    "ratio_ceiling": 2.0 - 1.0 / m,
                    "gap_test_can_fire": 2.0 - 1.0 / m >= GAP,
                    "g_normalized": (nc + 1 - m) / (m - 1),
                    "g_resolution": 1.0 / (m - 1),
                }
            )
        rows.append(row)
    reached = [r["ratio_points"] for r in rows if r["ratio_points"] is not None]
    if len(reached) >= 2 and min(reached[-2:]) >= GAP:
        verdict = "persistent factor-2 gap"
    elif reached and max(reached) <= EQUAL:
        verdict = "essentially equal"
    elif reached:
        verdict = "intermediate"
    else:
        verdict = "no target reached"
    # Equal-n comparison (added after the first run, see README): median of e_CC(n)/e_G(n) over the
    # grid n ≥ 8 where both normalized errors exceed 1e-13. Meaningful for algebraic convergence
    # (Theorem 5.1: same O(n^{-k-1}) bound); for geometric convergence it grows with n.
    both = (N_GRID >= 8) & (rg > 1e-13) & (rc > 1e-13)
    eq_n = float(np.median(rc[both] / rg[both])) if both.any() else None
    return {
        "id": fn.id,
        "latex": fn.latex,
        "kind": fn.kind,
        "rho": fn.rho,
        "l1_norm": scale,
        "median_error_ratio_equal_n": eq_n,
        "n_range_equal_n": [int(N_GRID[both].min()), int(N_GRID[both].max())]
        if both.any()
        else None,
        "levels": rows,
        "verdict": verdict,
        # The persistence rule needs two reached ε levels: undefined for fewer.
        "persistence_test_defined": len(reached) >= 2,
        # Can the pre-registered persistence rule fire at all (R_max ≥ GAP at its two levels)?
        "persistence_test_can_fire": len(reached) >= 2
        and all([r["gap_test_can_fire"] for r in rows if r["ratio_points"] is not None][-2:]),
        "max_ratio": max(reached) if reached else None,
        "g_levels": [r["g_normalized"] for r in rows if r["ratio_points"] is not None],
    }


def kink_analysis(fn: Integrand, ns: NDArray[np.int64], eg: list[float], ec: list[float]) -> dict:
    """Locate the kink of Clenshaw–Curtis on 1/(1 + 16x²) and measure the rates on each side.

    Weideman & Trefethen (2007), Lemma 4 / Thm. 3: the CC error is the sum of a Gauss-like term
    ∝ ρ^{−2n} and an aliasing term ∝ ρ^{−n} n^{−3}; for this f (ξ = 1.28i) the relative sign of
    the two terms alternates from one n to the next of the same parity, so the error zig-zags
    and has a deep cancellation dip where the two terms are equal: that is their kink, eqs.
    (27)–(28). Measured here:

    * kink: per parity of n, the n minimizing D(n) = log e(n) − ½(log e(n−2) + log e(n+2)) over
      n ≥ 10 with e(n ± 2) > 1e-14 (the deepest dip);
    * rates: r₄(n) = log(e(n)/e(n+4)) / (4 log ρ), the local decay per point in units of log ρ
      over four steps of n, which averages out the alternation (2 before the kink and
      1 + 3/(n log ρ) after it, by Thm. 3); and least-squares slopes of log e / log ρ.

    # NOTE: the first run used "first n after which the two-step rate stays < 1.5"; the zig-zag
    # makes that rate jump between −6 and 9 near the kink, so it reported n = 65/58. The dip
    # criterion above is what eqs. (27)–(28) define (equal magnitudes); see README.
    """
    assert fn.rho is not None
    lrho = math.log(fn.rho)
    idx = {int(n): i for i, n in enumerate(ns)}
    floor = 1e-14
    out: dict[str, Any] = {"rho": fn.rho, "rates": {}, "dips": {}}
    for parity, key in ((0, "n_even"), (1, "n_odd")):
        cand = [
            n
            for n in ns
            if n % 2 == parity
            and n >= 10
            and n - 2 in idx
            and n + 2 in idx
            and ec[idx[n + 2]] > floor
        ]
        dip = {
            int(n): math.log(ec[idx[n]])
            - 0.5 * (math.log(ec[idx[n - 2]]) + math.log(ec[idx[n + 2]]))
            for n in cand
        }
        ranked = sorted(dip, key=lambda n: dip[n])
        out[f"kink_{key}"] = ranked[0]
        out["dips"][key] = {"deepest": [[n, dip[n]] for n in ranked[:3]]}
        r4 = {}
        for name, err in (("cc", ec), ("gauss", eg)):
            r4[name] = {
                int(n): math.log(err[idx[n]] / err[idx[n + 4]]) / (4 * lrho)
                for n in ns
                if n % 2 == parity and n + 4 in idx and err[idx[n + 4]] > floor and n <= 120
            }
        out["rates"][key] = r4
    # The first-run detector, kept for the record (see the NOTE above): per parity, the first
    # n such that the two-step rate r₂(n') = log(e(n')/e(n'+2))/(2 log ρ) is < 1.5 for every later
    # n' with e(n'), e(n'+2) > floor. It gives n = 65 (odd) and 58 (even).
    first_run = {}
    for parity, key in ((1, "kink_n_odd"), (0, "kink_n_even")):
        r2 = {
            int(n): math.log(ec[idx[n]] / ec[idx[n + 2]]) / (2 * lrho)
            for n in ns
            if n % 2 == parity
            and n >= 10
            and n + 2 in idx
            and ec[idx[n]] > floor
            and ec[idx[n + 2]] > floor
        }
        keys = sorted(r2)
        first_run[key] = next((n for n in keys if all(r2[m] < 1.5 for m in keys if m >= n)), None)
    out["first_run_detector"] = {
        "definition": "first n after which the two-step rate stays < 1.5 (per parity); "
        "replaced by the deepest-dip criterion after the first run",
        **first_run,
    }
    # Weideman & Trefethen (2007) §4: kink at N = 54.17 points (N even, n = N − 1 odd) and
    # N = 46.74 points (N odd, n even) for this integrand.
    out["predicted_kink_n_odd"] = 54.17 - 1.0
    out["predicted_kink_n_even"] = 46.74 - 1.0
    # Slopes of log e(n)/log ρ (both parities): Gauss before its floor, CC before and after the kink.
    for name, err, lo, hi in (("gauss", eg, 10, 60), ("cc", ec, 10, 40), ("cc", ec, 60, 80)):
        sel = [(n, err[idx[n]]) for n in range(lo, hi + 1) if n in idx and err[idx[n]] > floor]
        x = np.array([n for n, _ in sel], dtype=float)
        y = np.log([e for _, e in sel]) / lrho
        out[f"slope_{name}_n{lo}_{hi}"] = float(np.polyfit(x, y, 1)[0])
    out["predicted_slope_after_kink_n70"] = -(1.0 + 3.0 / (70 * lrho))
    return out


# --------------------------------------------------------------------------------------
# E3: practical comparison with error estimates
# --------------------------------------------------------------------------------------


def gauss_doubling(problem: Problem, *, tol: float, m0: int = 2, max_levels: int = 10) -> Result:
    """Study construct: Gauss–Legendre on m = m0·2^k points with |G_{m_k} − G_{m_{k−1}}| as the
    error estimate and the same stopping test as ``clenshaw_curtis`` (first k ≥ 2 with
    d_k ≤ tol·max(1, |G|)). Gauss rules are not nested: the cost is Σ_k m_k evaluations."""
    a, b = problem.domain
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    n_fev, prev, est, d = 0, math.nan, math.nan, math.nan
    trace_err: list[tuple[int, float]] = []
    for k in range(max_levels + 1):
        m = m0 * 2**k
        t, w = gauss_rule(m)
        vals = [problem.f(float(mid + half * ti)) for ti in t]
        n_fev += m
        prev, est = est, half * math.fsum((w * np.array(vals)).tolist())
        if problem.exact is not None:
            trace_err.append((n_fev, abs(est - problem.exact)))
        if k >= 1:
            d = abs(est - prev)
        if k >= 2 and d <= tol * max(1.0, abs(est)):
            return Result(
                "gauss_doubling", est, est, True, "passed", k, n_fev, extra={"curve": trace_err}
            )
    return Result(
        "gauss_doubling",
        est,
        est,
        False,
        "max_levels",
        max_levels,
        n_fev,
        extra={"curve": trace_err},
    )


def _gauss_cum_fev(m: int) -> int:
    """Distinct evaluations of numopt gauss_legendre(n=m): Σ_{j≤m} j minus the shared node 0."""
    return m * (m + 1) // 2 - (m + 1) // 2 + 1


GAUSS_M_MAX = max(m for m in range(1, 200) if _gauss_cum_fev(m) <= BUDGET)  # = 90
SIMPSON_LEVELS = 11  # n = 2 → 2·2^11 + 1 = 4097 points
ROMBERG_LEVELS = 12  # 2^12 + 1 = 4097 points


def _first_pass(full: Result, tol: float) -> int | None:
    """First step k ≥ 2 whose err_est passes the bound err_est ≤ tol·max(1, |estimate|).

    For ``gauss_legendre`` (step k = rule with k + 1 points) and ``simpson`` (step k = level k),
    step k of a long run equals the last step of the run with n = k + 1 (levels = k): both
    methods compute their estimates from the prefix only. numopt's ``converged`` is this bound
    *and* (for composite rules) a confirmed asymptotic regime, so a converged run has passed the
    bound; the first converged setting is therefore found by scanning upward from this k.
    """
    for k, s in enumerate(full.trace):
        e, v = s.info["err_est"], s.fun
        if k >= 2 and e is not None and v is not None and e <= tol * max(1.0, abs(v)):
            return k
    return None


def _first_converged(
    method: str,
    problem: Problem,
    tol: float,
    key: str,
    first_pass: int | None,
    stop: int,
    *,
    offset: int = 0,
    **fixed: Any,
) -> Result:
    """The run with the smallest ``key`` value whose own ``converged`` is True (else the run at
    ``stop``): values from first_pass + offset to stop, as numopt itself decides convergence."""
    if first_pass is None or first_pass + offset > stop:
        return run_method(method, problem, tol=tol, **{key: stop}, **fixed)
    res = None
    for v in range(first_pass + offset, stop + 1):
        res = run_method(method, problem, tol=tol, **{key: v}, **fixed)
        if res.converged:
            return res
    assert res is not None
    return res


def practical_run(label: str, problem: Problem, tol: float) -> dict[str, Any]:
    """One method on one problem with its own stopping test; cost = distinct f evaluations."""
    t0 = time.perf_counter()
    if label == "clenshaw_curtis":
        res = cc.clenshaw_curtis(problem, n=2, max_levels=SIMPSON_LEVELS, tol=tol)
        curve = [(s.info["n"] + 1, s.info["error"]) for s in res.trace]
    elif label == "gauss_legendre":
        # The run with n = m contains the runs with n < m as a prefix; take the first m whose
        # documented test passes, and re-run with that n so that n_fev is counted, not inferred.
        full = run_method("gauss_legendre", problem, n=GAUSS_M_MAX, tol=tol)
        assert full.n_fev == _gauss_cum_fev(GAUSS_M_MAX)
        ok = _first_pass(full, tol)
        res = _first_converged("gauss_legendre", problem, tol, "n", ok, GAUSS_M_MAX, offset=1)
        curve = [(_gauss_cum_fev(k + 1), s.info["error"]) for k, s in enumerate(full.trace)]
    elif label == "gauss_doubling":
        res = gauss_doubling(problem, tol=tol)
        curve = res.extra.pop("curve")
    elif label == "gauss_patterson":
        res = gk.gauss_patterson(problem, tol=tol)
        curve = [(s.info["n_points"], s.info["error"]) for s in res.trace]
    elif label == "quadpack_qags":
        res = gk.quadpack_qags(problem, tol=tol)
        curve = []
    elif label == "quadpack_qags_epsabs0":  # sensitivity: the reviewer's purely relative request
        res = gk.quadpack_qags(problem, tol=tol, epsabs=0.0)
        curve = []
    elif label == "romberg":
        res = run_method("romberg", problem, tol=tol, max_levels=ROMBERG_LEVELS)
        curve = [(s.info["n_panels"] + 1, s.info["error"]) for s in res.trace]
    elif label == "simpson":
        full = run_method("simpson", problem, n=2, levels=SIMPSON_LEVELS, tol=tol)
        ok = _first_pass(full, tol)
        res = _first_converged("simpson", problem, tol, "levels", ok, SIMPSON_LEVELS, n=2)
        curve = [(s.info["n_panels"] + 1, s.info["error"]) for s in full.trace]
    elif label == "adaptive_simpson":
        res = run_method("adaptive_simpson", problem, tol=tol, max_iter=100_000)
        curve = []
    else:
        raise ValueError(label)
    exact = float(problem.exact) if problem.exact is not None else math.nan
    err = abs(float(res.x) - exact)
    target = tol * max(1.0, abs(exact))
    return {
        "method": label,
        "problem": problem.id,
        "tol": tol,
        "converged": bool(res.converged),
        "n_fev": int(res.n_fev),
        "error": err,
        "success": bool(res.converged and res.n_fev <= BUDGET and err <= target),
        "false_convergence": bool(res.converged and err > target),
        "over_budget": bool(res.n_fev > BUDGET),
        "seconds": time.perf_counter() - t0,
        "curve": [[int(c), e] for c, e in curve],
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _floor(e: list[float], lo: float = 1e-17) -> list[float]:
    """Errors that are exactly 0 are drawn at ``lo`` (stated in each caption)."""
    return [max(v, lo) for v in e]


def _pair_axes(ax, ns, eg, ec, *, ms: float = 4.0, loglog: bool = False) -> None:
    ax.plot(ns, _floor(eg), "o", ms=ms, color=C_GAUSS, label="Gauss", zorder=3)
    ax.plot(
        ns,
        _floor(ec),
        "o",
        ms=ms + 2.5,
        mfc="none",
        mew=1.1,
        color=C_CC,
        label="Clenshaw–Curtis",
        zorder=2,
    )
    ax.set_yscale("log")
    if loglog:
        ax.set_xscale("log", base=2)


def fig_reproduce_fig2(curves: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), layout="constrained", sharex=True)
    ids = ["x20", "exp", "exp_neg_x2", "runge16", "exp_inv_x2", "abs_x3"]
    lookup = {fn.id: fn for fn in TREFETHEN}
    for ax, fid in zip(axes.flat, ids, strict=True):
        c = curves[fid]
        k = [i for i, n in enumerate(c["n"]) if n <= 30]
        _pair_axes(ax, [c["n"][i] for i in k], [c["gauss"][i] for i in k], [c["cc"][i] for i in k])
        ax.set_title(lookup[fid].latex, loc="left")
        ax.set_ylim(1e-17, 10)
        ax.set_yticks([1e0, 1e-5, 1e-10, 1e-15])
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|I - I_n|$")
    for ax in axes[-1]:
        ax.set_xlabel("n  (n + 1 points)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "Reproduction of Trefethen (2008) Fig. 2  —  errors equal to 0 are drawn at 1e-17",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(FIGURES / "fig2_reproduction.svg", metadata={"Date": None})
    plt.close(fig)


def fig_reproduce_fig3(fig3: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
    ax.plot(fig3["n_gauss"], fig3["err_gauss"], "o", ms=5, color=C_GAUSS, label="Gauss")
    ax.plot(
        fig3["n_cc"],
        fig3["err_cc"],
        "o",
        ms=7.5,
        mfc="none",
        mew=1.1,
        color=C_CC,
        label="Clenshaw–Curtis",
    )
    n = np.array(fig3["n_cc"], dtype=float)
    ref = fig3["err_cc"][4] * (n / n[4]) ** -1.5
    ax.plot(n, ref, color=MUTED, lw=0.9, ls="--", label=r"slope $n^{-3/2}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n  (n + 1 points)")
    ax.set_ylabel(r"$|I - I_n|$")
    ax.set_title(r"Reproduction of Fig. 3: $\int_{-1}^{1}|x+\frac{1}{2}|^{1/2}dx$", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "fig3_reproduction.svg", metadata={"Date": None})
    plt.close(fig)


def fig_kink(curves: dict[str, Any], kink: dict[str, Any]) -> None:
    c = curves["runge16"]
    k = [i for i, n in enumerate(c["n"]) if n <= 120]
    ns = np.array([c["n"][i] for i in k])
    eg, ec = [c["gauss"][i] for i in k], [c["cc"][i] for i in k]
    rho = kink["rho"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.2), layout="constrained", sharex=True, height_ratios=(3, 2)
    )
    _pair_axes(ax, ns, eg, ec, ms=3.2)
    # Slope guides anchored to the data (constants are not predicted here): Gauss through n = 20,
    # the post-kink CC model ρ^{−n} n^{−3} (W–T Thm. 3) through n = 70.
    ia, ib = list(ns).index(20), list(ns).index(70)
    n1 = np.linspace(1, 85, 100)
    ax.plot(
        n1,
        eg[ia] * rho ** (-2 * (n1 - 20)),
        color=C_GAUSS,
        lw=0.8,
        ls="--",
        label=r"slope $\rho^{-2n}$",
    )
    n2 = np.linspace(40, 120, 100)
    ax.plot(
        n2,
        ec[ib] * rho ** (-(n2 - 70)) * (n2 / 70) ** -3,
        color=C_CC,
        lw=0.8,
        ls="--",
        label=r"slope $\rho^{-n}n^{-3}$ (W–T Thm. 3)",
    )
    for x, lab, ha, dx in (
        (kink["predicted_kink_n_odd"], "odd n", "left", 1.0),
        (kink["predicted_kink_n_even"], "even n", "right", -1.0),
    ):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax2.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax2.text(
            x + dx,
            2.72,
            f"W–T kink, {lab}\nn = {x:.2f}",
            fontsize=7.5,
            color=MUTED,
            ha=ha,
            va="center",
        )
    dips = [kink["kink_n_odd"], kink["kink_n_even"]]
    ax.plot(
        dips,
        [ec[list(ns).index(n)] for n in dips],
        "x",
        ms=9,
        mew=1.6,
        color=INK,
        label="deepest CC dip (measured kink)",
    )
    ax.set_ylim(1e-17, 1)
    ax.set_ylabel(r"$|I - I_n|$")
    ax.set_title(
        r"$1/(1+16x^2)$, $\rho = (1+\sqrt{17})/4$: the kink (Trefethen 2008, Fig. 7)", loc="left"
    )
    ax.legend(loc="lower left", fontsize=8)
    for key, mk in (("n_odd", "o"), ("n_even", "s")):
        r = kink["rates"][key]
        xs = sorted(r["cc"])
        ax2.plot(
            xs,
            [r["cc"][n] for n in xs],
            mk,
            ms=3.5,
            mfc="none",
            color=C_CC,
            label=f"CC, {key.replace('n_', 'n ')}",
        )
        xg = sorted(r["gauss"])
        ax2.plot(
            xg,
            [r["gauss"][n] for n in xg],
            mk,
            ms=3,
            color=C_GAUSS,
            label=f"Gauss, {key.replace('n_', 'n ')}",
        )
    ax2.axhline(2, color=MUTED, lw=0.8)
    ax2.axhline(1, color=MUTED, lw=0.8)
    ax2.set_ylim(0, 3.1)
    ax2.set_ylabel(r"rate $r_4(n)$ per point" + "\n" + r"(units of $\log\rho$)")
    ax2.set_xlabel("n  (n + 1 points)")
    ax2.legend(loc="lower left", ncol=2, fontsize=7.5)
    fig.savefig(FIGURES / "fig7_kink.svg", metadata={"Date": None})
    plt.close(fig)


def fig_ratio(eff: list[dict[str, Any]]) -> None:
    """Left: the pre-registered R(ε) with its ceiling 2 − 1/m (grey bar). Right: the normalized
    gap g = (p_CC − m)/(m − 1) with a bar of ± one point (its resolution)."""
    show = (1e-4, 1e-6, 1e-10, 1e-13)
    colors = ("#CC79A7", "#0072B2", "#D55E00", "#009E73")
    markers = ("D", "o", "s", "^")
    order = TREFETHEN + EXTRA + POLES
    by_id = {e["id"]: e for e in eff}
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 6.8), layout="constrained", sharey=True, width_ratios=(1.0, 1.0)
    )
    labels = []
    for row, fn in enumerate(order):
        e = by_id[fn.id]
        tag = f"  (ρ = {fn.rho:.3g})" if fn.rho is not None else ""
        labels.append(f"{fn.latex}{tag}")
        for j, eps in enumerate(show):
            lev = next(r for r in e["levels"] if r["eps"] == eps)
            if lev["ratio_points"] is None:
                continue
            y = row + (j - 1.5) * 0.19
            style = {
                "ms": 5.5,
                "color": colors[j],
                "mfc": colors[j] if j != 2 else "none",
                "mew": 1.2,
            }
            ax.plot(
                [lev["ratio_points"], lev["ratio_ceiling"]], [y, y], color=GRID, lw=2.2, zorder=1
            )
            ax.plot(lev["ratio_ceiling"], y, "|", ms=6, color=MUTED, zorder=2)
            ax.plot(lev["ratio_points"], y, markers[j], label=f"ε = {eps:.0e}", zorder=3, **style)
            g, res = lev["g_normalized"], lev["g_resolution"]
            ax2.plot([g - res, g + res], [y, y], color=colors[j], lw=0.9, alpha=0.45, zorder=1)
            ax2.plot(g, y, markers[j], zorder=3, **style)
    ax.axvline(1, color=MUTED, lw=0.9)
    ax.axvline(2, color=MUTED, lw=0.9, ls="--")
    ax.axvspan(GAP, 2.6, color="#f2e6dc", zorder=0, lw=0)
    ax.text(
        GAP + 0.02,
        -0.9,
        f"R ≥ {GAP}: pre-registered\n'factor 2 visible'",
        fontsize=7.5,
        color=MUTED,
    )
    ax.set_yticks(range(len(order)), labels)
    ax.set_ylim(len(order) - 0.4, -1.9)
    ax.set_xlim(0.6, 2.6)
    ax.set_xlabel(
        "R(ε) = CC points / Gauss points (m)\n"
        "grey bar ends at the ceiling 2 − 1/m,\nwhich an exact factor 2 in degree gives",
        fontsize=8.5,
    )
    ax.set_title("Pre-registered statistic R", loc="left")
    for a in (ax, ax2):
        for y in (len(TREFETHEN) - 0.5, len(TREFETHEN) + len(EXTRA) - 0.5):
            a.axhline(y, color=GRID, lw=1.2)
    ax2.axvline(0, color=MUTED, lw=0.9)
    ax2.axvline(1, color=MUTED, lw=0.9, ls="--")
    ax2.set_xlim(-0.35, 1.45)
    ax2.set_xlabel(
        "g = (p_CC − m)/(m − 1)\n0 = equal points, 1 = full factor 2 in degree;\n"
        "faint bar: ± one point (resolution 1/(m − 1))",
        fontsize=8.5,
    )
    ax2.set_title("Normalized gap g (post hoc)", loc="left")
    handles, labs = ax.get_legend_handles_labels()
    uniq = dict(zip(labs, handles, strict=True))
    ax2.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)
    fig.suptitle(
        "Is the factor of 2 visible?  Target |I − I_n| ≤ ε‖f‖₁ for every larger n; "
        "no marker: ε not reached by n = 1024",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(FIGURES / "efficiency_ratio.svg", metadata={"Date": None})
    plt.close(fig)


def fig_profiles(costs: dict[float, NDArray[np.float64]], labels: list[str]) -> None:
    fig, axes = plt.subplots(1, len(TOLS), figsize=(12.5, 4.1), layout="constrained", sharey=True)
    for ax, tol in zip(axes, TOLS, strict=True):
        prof = bench.performance_profile_from_costs(costs[tol], labels, tau=tol)
        for j, lab in enumerate(labels):
            col, mk, ls = METHOD_STYLE[lab]
            ax.step(prof.x, prof.y[j], where="post", color=col, ls=ls, lw=1.7, label=lab)
            ax.plot(prof.x[-1], prof.y[j][-1], mk, color=col, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r"performance ratio $\alpha$ (f evaluations / best)")
        ax.set_title(f"tol = {tol:.0e}", loc="left")
    axes[0].set_ylabel(r"$\rho_s(\alpha)$  (fraction of problems)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=8)
    fig.suptitle(
        f"Dolan–Moré performance profiles, {costs[TOLS[0]].shape[0]} integrands, budget {BUDGET} evaluations; success = converged and error ≤ tol·max(1,|I|)\n"
        "gauss_patterson stops at 127 points; numopt's gauss_legendre (cumulative cost) is not in the profiles",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(FIGURES / "performance_profiles.svg", metadata={"Date": None})
    plt.close(fig)


def fig_convergence(runs: list[dict[str, Any]], pids: list[str], tol: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6), layout="constrained")
    for ax, pid in zip(axes.flat, pids, strict=True):
        for lab, (col, mk, ls) in METHOD_STYLE.items():
            r = next(
                x for x in runs if x["method"] == lab and x["problem"] == pid and x["tol"] == tol
            )
            if r["curve"]:
                cs = [c for c, _ in r["curve"]]
                es = _floor([e for _, e in r["curve"]])
                ax.plot(cs, es, ls=ls, marker=mk, ms=3.5, lw=1.2, color=col, label=lab)
            if r["converged"]:
                ax.plot(
                    r["n_fev"],
                    max(r["error"], 1e-17),
                    mk,
                    ms=9,
                    mfc="none",
                    mew=1.6,
                    color=col,
                    label=None if r["curve"] else f"{lab} (stop only)",
                )
        ax.axhline(tol, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1.5, 2.2 * BUDGET)
        ax.set_ylim(1e-17, 10)
        ax.set_title(pid, loc="left")
        ax.set_xlabel("cumulative f evaluations")
        ax.set_ylabel(r"$|I - I_{est}|$")
    handles: dict[str, Any] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="outside right center", fontsize=8)
    fig.suptitle(
        f"Error vs cost of each method's sequence; large open marker = its own stopping point at tol = {tol:.0e} (dotted line);\nerrors equal to 0 are drawn at 1e-17",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(FIGURES / "convergence.svg", metadata={"Date": None})
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _rule_on(p: Problem, rule: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    """Σ w_k f(x_k) of a rule on [−1, 1] mapped to p.domain."""
    a, b = (float(v) for v in p.domain)
    t, w = rule
    vals = np.array([p.f(float(0.5 * (a + b) + 0.5 * (b - a) * ti)) for ti in t])
    return 0.5 * (b - a) * math.fsum((w * vals).tolist())


def _calculus_problem(pid: str) -> Problem:
    return problems.get(pid)


def main() -> None:
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ---- reference checks of the Gauss baseline
    gauss_check = check_gauss_reference()

    # ---- E1/E2: error curves on the n grid
    curves: dict[str, Any] = {}
    for fn in ALL:
        eg, ec = error_curves(fn, N_GRID)
        curves[fn.id] = {
            "n": N_GRID.tolist(),
            "gauss": eg,
            "cc": ec,
            "exact": fn.exact,
            "kind": fn.kind,
            "rho": fn.rho,
        }
    eff = [efficiency(fn, curves[fn.id]["gauss"], curves[fn.id]["cc"]) for fn in ALL]

    # ---- paper checkpoints
    runge = next(fn for fn in TREFETHEN if fn.id == "runge16")
    kink = kink_analysis(runge, N_GRID, curves["runge16"]["gauss"], curves["runge16"]["cc"])
    half = next(fn for fn in TREFETHEN if fn.id == "sqrt_abs_x_half")
    n_cc3 = [2**k for k in range(4, 17)]
    n_g3 = [2**k for k in range(4, 11)]
    fig3 = {
        "n_cc": n_cc3,
        "err_cc": [abs(quad(cc_rule(n), half.f) - half.exact) for n in n_cc3],
        "n_gauss": n_g3,
        "err_gauss": [abs(quad(gauss_rule(n + 1), half.f) - half.exact) for n in n_g3],
    }
    for key, ns in (("cc", n_cc3), ("gauss", n_g3)):
        sel = [i for i, n in enumerate(ns) if n >= 256]
        fig3[f"loglog_slope_{key}_n_ge_256"] = float(
            np.polyfit(
                np.log([ns[i] for i in sel]), np.log([fig3[f"err_{key}"][i] for i in sel]), 1
            )[0]
        )
    lookup = {fn.id: fn for fn in ALL}
    # |x + ½|^½: CC has a node at the singularity x = −½ exactly when 3 | n (k/n = 2/3).
    c_half = curves["sqrt_abs_x_half"]
    big = [(n, e / half.exact) for n, e in zip(c_half["n"], c_half["cc"], strict=True) if n >= 256]
    spikes = {
        "median_rel_err_cc_n_ge_256_3_divides_n": float(
            np.median([e for n, e in big if n % 3 == 0])
        ),
        "median_rel_err_cc_n_ge_256_otherwise": float(np.median([e for n, e in big if n % 3])),
        "max_rel_err_cc_n_ge_256": float(max(e for _, e in big)),
        "last_n_with_rel_err_above_1e-4": int(max(n for n, e in big if e > 1e-4)),
    }
    # Direct test of the artifact claim: n_X(1e-4) on the grid without the n that 3 divides.
    keep = np.array([n % 3 != 0 for n in N_GRID])
    scale_half = l1_norm(half)
    for key in ("cc", "gauss"):
        rel = np.array(c_half[key])[keep] / scale_half
        spikes[f"n_needed_1e-4_{key}_without_3_divides_n"] = n_needed(N_GRID[keep], rel, 1e-4)
        spikes[f"n_needed_1e-4_{key}_all_n"] = n_needed(
            N_GRID, np.array(c_half[key]) / scale_half, 1e-4
        )

    def err(fid: str, rule: str, n: int) -> float:
        fn = lookup[fid]
        r = gauss_rule(n + 1) if rule == "gauss" else cc_rule(n)
        return abs(quad(r, fn.f) - fn.exact)

    checkpoints = {
        "clenshaw_curtis_1960_quote": {
            "gauss_32_points": err("sqrt_abs_x_half", "gauss", 31),
            "gauss_64_points": err("sqrt_abs_x_half", "gauss", 63),
            "cc_n63": err("sqrt_abs_x_half", "cc", 63),
            "cc_n64": err("sqrt_abs_x_half", "cc", 64),
            "paper": {"gauss_32_points": 0.00317, "gauss_64_points": 0.00036, "cc": 0.00078},
        },
        "pilot_from_task": {
            "abs_x3_n64": {"gauss": err("abs_x3", "gauss", 64), "cc": err("abs_x3", "cc", 64)},
            "runge16_n64": {"gauss": err("runge16", "gauss", 64), "cc": err("runge16", "cc", 64)},
            "exp_n8": {"gauss": err("exp", "gauss", 8), "cc": err("exp", "cc", 8)},
            "exp_n16": {"gauss": err("exp", "gauss", 16), "cc": err("exp", "cc", 16)},
        },
        "x20_first_exact_n": {
            rule: next(n for n in range(1, 40) if err("x20", rule, n) <= 4 * 2.2e-16)
            for rule in ("gauss", "cc")
        },
        "cos_paper": {
            "gauss_n6": quad(gauss_rule(7), np.cos),
            "cc_n11": quad(cc_rule(11), np.cos),
            "cc_n10": quad(cc_rule(10), np.cos),
            "exact": 2.0 * math.sin(1.0),
        },
        "fig7_runge16_n120": {
            "gauss": err("runge16", "gauss", 120),
            "cc": err("runge16", "cc", 120),
        },
    }

    # ---- E3: practical comparison
    labels = list(METHOD_STYLE)
    jp = [labels.index(m) for m in PROFILE_METHODS]
    calc_ids = [p.id for p in problems.list_problems("calculus")]
    # pole_a4 is 1/(1 + 16x²) = runge16: kept in the ρ ladder of E2, counted once in E3.
    probs = [_calculus_problem(pid) for pid in calc_ids] + [
        fn.problem() for fn in ALL if fn.id != "pole_a4"
    ]
    runs: list[dict[str, Any]] = []
    costs: dict[float, NDArray[np.float64]] = {}
    for tol in TOLS:
        table = np.full((len(probs), len(labels)), np.inf)
        for i, p in enumerate(probs):
            for j, lab in enumerate(labels):
                r = practical_run(lab, p, tol)
                runs.append(r)
                if r["success"]:
                    table[i, j] = r["n_fev"]
        costs[tol] = table
    sens_runs = [practical_run("quadpack_qags_epsabs0", p, tol) for tol in TOLS for p in probs]
    prob_ids = [p.id for p in probs]
    # Accuracy at matched nesting levels, independent of any stopping test: CC with 2^{k+1} + 1
    # points vs Patterson with 2^{k+1} − 1 points, k = 4, 5 (33/31 and 65/63 points). Counted only
    # where both errors exceed 1e-14·max(1, |I|) (above rounding).
    matched: dict[str, Any] = {"levels": {}, "examples": {}}
    for k in (4, 5):
        rows = []
        for p in probs:
            assert p.exact is not None
            exact = float(p.exact)
            e_cc = abs(_rule_on(p, cc_rule(2 ** (k + 1))) - exact)
            e_gp = abs(_rule_on(p, gk.patterson_rules()[k]) - exact)
            rows.append((p.id, e_cc, e_gp, 1e-14 * max(1.0, abs(exact))))
        above = [(pid, a, b) for pid, a, b, fl in rows if a > fl and b > fl]
        matched["levels"][f"cc{2 ** (k + 1) + 1}_vs_gp{2 ** (k + 1) - 1}"] = {
            "n_compared": len(above),
            "cc_more_accurate": [pid for pid, a, b in above if a < b],
            "patterson_more_accurate": [pid for pid, a, b in above if b < a],
        }
        for pid, a, b, _ in rows:
            if pid in ("runge", "runge16", "exp_inv_x2", "cos50", "gaussian", "x20"):
                matched["examples"].setdefault(pid, {})[f"k{k}"] = {"cc": a, "patterson": b}
    summary = {}
    for tol in TOLS:
        rows = {}
        prof = bench.performance_profile_from_costs(costs[tol][:, jp], PROFILE_METHODS, tau=tol)
        for lab in labels:
            rr = [r for r in runs if r["method"] == lab and r["tol"] == tol]
            solved = [r for r in rr if r["success"]]
            in_prof = lab in PROFILE_METHODS
            rows[lab] = {
                "solved": len(solved),
                "of": len(rr),
                "false_convergence": sum(r["false_convergence"] for r in rr),
                "false_convergence_problems": [r["problem"] for r in rr if r["false_convergence"]],
                "over_budget": sum(r["over_budget"] for r in rr),
                "median_fev_solved": float(np.median([r["n_fev"] for r in solved]))
                if solved
                else None,
                "rho_at_1": float(prof.at(1.0)[PROFILE_METHODS.index(lab)]) if in_prof else None,
                "rho_at_2": float(prof.at(2.0)[PROFILE_METHODS.index(lab)]) if in_prof else None,
                "in_profiles": in_prof,
            }
        summary[tol] = rows
    # Pairwise: CC against each Gauss-type baseline on the problems both solve.
    pairwise = {}
    for tol in TOLS:
        T = costs[tol]
        jc = labels.index("clenshaw_curtis")
        out = {}
        for other in (
            "gauss_doubling",
            "gauss_patterson",
            "quadpack_qags",
            "romberg",
            "simpson",
            "adaptive_simpson",
        ):
            jo = labels.index(other)
            both = np.isfinite(T[:, jc]) & np.isfinite(T[:, jo])
            ratio = T[both, jc] / T[both, jo]
            out[other] = {
                "n_both": int(both.sum()),
                "cc_cheaper": int((ratio < 1).sum()),
                "equal": int((ratio == 1).sum()),
                "median_cost_ratio_cc_over_other": float(np.median(ratio)) if ratio.size else None,
            }
        # Nested CC (2^{K+1} + 1 points) vs nested Patterson (2^{K+1} − 1 points): same level K?
        jg = labels.index("gauss_patterson")
        both = np.isfinite(T[:, jc]) & np.isfinite(T[:, jg])
        ids_both = [prob_ids[i] for i in np.flatnonzero(both)]
        k_cc = np.log2(T[both, jc] - 1.0)
        k_gp = np.log2(T[both, jg] + 1.0)
        out["gauss_patterson"].update(
            {
                "same_level": int((k_cc == k_gp).sum()),
                "patterson_earlier": int((k_gp < k_cc).sum()),
                "cc_earlier": int((k_cc < k_gp).sum()),
                "patterson_earlier_problems": [ids_both[i] for i in np.flatnonzero(k_gp < k_cc)],
                "cc_earlier_problems": [ids_both[i] for i in np.flatnonzero(k_cc < k_gp)],
            }
        )
        # Equal cap: gauss_patterson stops at 127 points; count CC successes within 129 points
        # (its rule n = 128), the nearest CC cap, so that both nested schemes get the same reach.
        cc_cap = np.isfinite(T[:, jc]) & (T[:, jc] <= 129)
        out["equal_cap_127_129"] = {
            "cc_solved_within_129": int(cc_cap.sum()),
            "patterson_solved_within_127": int(np.isfinite(T[:, jg]).sum()),
        }
        # Sensitivity of the QUADPACK comparison to the request (epsabs = 0, purely relative).
        sr = [r for r in sens_runs if r["tol"] == tol]
        qs = np.array([r["n_fev"] if r["success"] else np.inf for r in sr])
        both = np.isfinite(T[:, jc]) & np.isfinite(qs)
        ratio = T[both, jc] / qs[both]
        out["quadpack_qags_epsabs0"] = {
            "solved": int(np.isfinite(qs).sum()),
            "false_convergence": int(sum(r["false_convergence"] for r in sr)),
            "n_both": int(both.sum()),
            "cc_cheaper": int((ratio < 1).sum()),
            "median_cost_ratio_cc_over_other": float(np.median(ratio)) if ratio.size else None,
        }
        pairwise[tol] = out

    # ---- write results
    def dump(name: str, obj: Any) -> None:
        (RESULTS / name).write_text(
            json.dumps(obj, indent=1, sort_keys=False, allow_nan=False, default=float) + "\n"
        )

    dump(
        "error_curves.json",
        {"n_grid_note": "n = 1..256, then every 4th n to 1024; n + 1 points", "curves": curves},
    )
    dump(
        "efficiency_ratio.json",
        {"eps_levels": EPS_LEVELS, "gap": GAP, "equal": EQUAL, "functions": eff},
    )
    dump(
        "paper_checks.json",
        {
            "kink_runge16": kink,
            "fig3": fig3,
            "checkpoints": checkpoints,
            "sqrt_abs_x_half_spikes": spikes,
            "gauss_reference_check": gauss_check,
        },
    )
    dump(
        "practical.json",
        {
            "budget": BUDGET,
            "tols": TOLS,
            "methods": labels,
            "problems": [p.id for p in probs],
            "numopt_integration_sha256": hashlib.sha256(
                Path(inspect.getfile(gauss_legendre_rule)).read_bytes()
            ).hexdigest(),
            "settings": {
                "clenshaw_curtis": "n=2, max_levels=11 (≤ 4097 points), stop at first k ≥ 2 with d_k ≤ tol·max(1,|I|)",
                "gauss_legendre": f"numopt defaults except tol; first n ≤ {GAUSS_M_MAX} whose run reports converged (cumulative cost; not in the profiles)",
                "gauss_doubling": "m = 2·2^k ≤ 2048 points, same stopping test as clenshaw_curtis (study construct)",
                "gauss_patterson": "nested Gauss–Kronrod–Patterson 1, 3, 7, …, 127 points (baselines.py), same stopping test as clenshaw_curtis; fails if level 6 (127 points) does not pass",
                "quadpack_qags": "scipy.integrate.quad (QUADPACK QAGS, adaptive G10/K21 + epsilon extrapolation), epsabs = epsrel = tol (request = tol·max(1,|I|)), limit=200; converged = ier == 0",
                "romberg": f"numopt, tol, max_levels={ROMBERG_LEVELS}",
                "simpson": f"numopt, n=2, first levels ≤ {SIMPSON_LEVELS} whose run reports converged",
                "adaptive_simpson": "numopt defaults (min_depth=2, max_depth=30) except tol, max_iter=100000; over-budget counts as failure",
            },
            "summary": summary,
            "pairwise_cc": pairwise,
            "matched_level_accuracy": matched,
            "costs": {
                str(t): [
                    [None if not math.isfinite(v) else int(v) for v in row] for row in costs[t]
                ]
                for t in TOLS
            },
            "runs": [{k: v for k, v in r.items() if k not in ("curve", "seconds")} for r in runs],
            "runs_quadpack_qags_epsabs0": [
                {k: v for k, v in r.items() if k not in ("curve", "seconds")} for r in sens_runs
            ],
        },
    )

    # ---- figures
    fig_reproduce_fig2(curves)
    fig_reproduce_fig3(fig3)
    fig_kink(curves, kink)
    fig_ratio(eff)
    fig_profiles({t: c[:, jp] for t, c in costs.items()}, PROFILE_METHODS)
    fig_convergence(runs, ["exp_0_1", "runge", "sqrt_0_1", "abs_kink"], 1e-10)

    # ---- console summary
    print(f"done in {time.perf_counter() - t_start:.1f} s")
    print("Gauss reference check:", json.dumps(gauss_check))
    print(
        "kink (n odd / n even):",
        kink["kink_n_odd"],
        kink["kink_n_even"],
        "predicted",
        kink["predicted_kink_n_odd"],
        kink["predicted_kink_n_even"],
    )
    print({k: v for k, v in kink.items() if k.startswith("slope")})
    for e in eff:
        rs = " ".join(
            f"{r['ratio_points']:.2f}" if r["ratio_points"] is not None else "  - "
            for r in e["levels"]
        )
        rho = f"{e['rho']:.3f}" if e["rho"] else "  -  "
        print(f"{e['id']:16s} {e['kind']:18s} rho={rho}  R: {rs}  -> {e['verdict']}")
    for tol in TOLS:
        print(f"tol={tol:.0e}")
        for lab, row in summary[tol].items():
            print(
                f"  {lab:17s} solved {row['solved']}/{row['of']}  false-conv {row['false_convergence']}  median fev {row['median_fev_solved']}  rho(1)={row['rho_at_1']}  rho(2)={row['rho_at_2']}"
            )
        print("  pairwise:", pairwise[tol])


if __name__ == "__main__":
    main()
