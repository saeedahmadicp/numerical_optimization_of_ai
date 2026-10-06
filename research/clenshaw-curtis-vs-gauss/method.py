"""Clenshaw–Curtis quadrature with nested doubling (research prototype for ``numopt.integration``).

Problem: I = ∫ₐᵇ f(x) dx for a continuous f on a finite interval [a, b] (``problem.domain`` or
``bracket=(a, b)``, a < b). On the reference interval [−1, 1] the (n + 1)-point
Clenshaw–Curtis rule is

    I_n(f) = Σ_{k=0}^{n} w_k f(x_k),   x_k = cos(kπ/n),   k = 0, …, n          (Trefethen 2008, (2.2)–(2.3))

where the weights w_k are those that integrate the degree-≤ n polynomial interpolant of f at the
Chebyshev extreme points exactly. Equivalently, with the interpolant written as
p_n(x) = Σ_{j=0}^{n} a_j T_j(x),

    I_n(f) = ∫₋₁¹ p_n(x) dx = Σ_{j even} a_j · 2/(1 − j²)                         (Trefethen 2008, §2, ``clenshaw_curtis.m``)

Weights (Waldvogel 2006). The explicit cosine formula is (Waldvogel (2.4)–(2.5))

    w_k = (c_k/n) [1 − Σ_{j=1}^{⌊n/2⌋} b_j/(4j² − 1) · cos(2jkπ/n)],
    b_j = 1 if j = n/2 else 2,   c_k = 1 if k ≡ 0 (mod n) else 2,

which costs O(n²). Waldvogel's Theorem (§5) gives the same weights as one inverse DFT of order n,
w = F_n⁻¹(v + g), with the rational vector v of (3.10) and g of (4.2), and w_n := w_0. This
module uses the DFT form (O(n log n), :func:`clenshaw_curtis_rule`); the test suite checks it
against the cosine formula and against the Chebyshev-moment conditions.

Nested doubling. The nodes for n are a subset of the nodes for 2n (x_k^{(n)} = x_{2k}^{(2n)}),
so the sequence n₀, 2n₀, 4n₀, … costs n_K + 1 evaluations in total, and d_k = |I_{n_k} − I_{n_{k−1}}|
is a free error estimate (Trefethen 2008, §1; Waldvogel 2006, §1). The nodes are computed as
x_k = sin(π(n − 2k)/(2n)), which equals cos(kπ/n) mathematically, is exactly antisymmetric
(x_{n−k} = −x_k, x_{n/2} = 0), and gives bit-identical abscissae on the doubled grid, so the
evaluation cache is exact.

Accuracy (for the README; not used by the code). The rule is exact for polynomials of degree n
(n + 1 when n is even, by symmetry); Gauss–Legendre with n + 1 points is exact to degree 2n + 1.
Theorem 5.2 of Trefethen (2008): on the grid, T_{n+p}(x_k) = T_{n−p}(x_k), so

    I(T_{n+p}) − I_n(T_{n+p}) = 8pn / (n⁴ − 2(p² + 1)n² + (p² − 1)²)   if n ± p is even (else 0),   (5.4)

which is O(n⁻²) or smaller for p ≤ n: aliased coefficients are integrated *almost* correctly.

Stopping test. Step k (k = 0, 1, …, ``max_levels``) uses n_k = n·2^k. For k ≥ 1,
err_est_k = d_k = |I_{n_k} − I_{n_{k−1}}|. The run stops at the first k ≥ 2 with

    d_k ≤ tol · max(1, |I_{n_k}|)

and reports converged=True. d_k estimates the error of the *coarser* rule I_{n_{k−1}}; when the
convergence is geometric the error of I_{n_k} is much smaller, so the test is conservative there.
# NOTE: the test starts at k = 2 (three rules) for the same reason as ``numopt`` Romberg: two rules
# can agree by accident at small n. Known blind spot (true of every sampling rule): an f that
# vanishes at all nodes of the finest grids, e.g. f = 1 − T_{4n₀}(x)², gives I_{n_k} = 0 for
# k = 0, 1, 2 and a false "converged" result.
If ``max_levels`` is reached without passing, converged=False. A non-finite f value or estimate
stops the run with converged=False.

Function values are cached by abscissa, so ``n_fev`` counts distinct points: n_K + 1 for a run
that stops at level K. The end points are evaluated exactly at a and b (not at
(a+b)/2 ± (b−a)/2, which can round outside [a, b]). Sums use ``math.fsum``.

Info keys:
    estimate: float — the current approximation I_{n_k} of I (same as ``Step.fun``).
    error: float | None — |estimate − exact|, or None when the exact value is unknown.
    err_est: float | None — d_k = |I_{n_k} − I_{n_{k−1}}|; None at k = 0.
    n: int — polynomial degree n_k of the rule (n_k + 1 points).
    n_points: int — number of nodes n_k + 1.
    nodes: [[x, f(x)]] — every node of the rule on [a, b], in decreasing x.
    weights: [float] — weights aligned with ``nodes`` (scaled by (b − a)/2);
        estimate = Σ weights[i]·nodes[i][1].
    cheb_coeffs: [float] — Chebyshev coefficients a_0, …, a_{n_k} of the interpolant
        p(x) = Σ a_j T_j(t(x)) with t = (2x − a − b)/(b − a); their decay shows the resolution
        and the aliasing that Trefethen (2008, §5) uses to explain the accuracy of the rule.
    new_nodes: int — number of nodes evaluated for the first time at this step.

References:
    L. N. Trefethen, "Is Gauss quadrature better than Clenshaw–Curtis?", SIAM Review 50(1)
    (2008) 67–87, doi:10.1137/060659831 — eqs. (2.2)–(2.3), Theorem 5.2 / (5.4).
    J. Waldvogel, "Fast construction of the Fejér and Clenshaw–Curtis quadrature rules",
    BIT 46 (2006) 195–202, doi:10.1007/s10543-006-0045-4 — eqs. (2.4)–(2.6), (3.10), (4.2), §5.
    C. W. Clenshaw and A. R. Curtis, "A method for numerical integration on an automatic
    computer", Numer. Math. 2 (1960) 197–205.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from numopt.core.counting import Counted, scalar_problem
from numopt.core.registry import ParamSpec
from numopt.core.types import Problem, Result, Step

#: Upper bound on the degree n·2^max_levels, to keep traces and work bounded (as numopt).
MAX_DEGREE = 2**18

#: ParamSpecs for promotion: ``@register(id="clenshaw_curtis", params=PARAMS["clenshaw_curtis"], **META["clenshaw_curtis"])``.
PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "clenshaw_curtis": (
        ParamSpec(
            "n",
            2,
            kind="int",
            min=1,
            max=64,
            help="Degree of the first rule (n + 1 Chebyshev points); step k uses n·2^k.",
        ),
        ParamSpec(
            "max_levels",
            12,
            kind="int",
            min=2,
            max=16,
            help="Maximum number of doublings; the final rule has n·2^max_levels + 1 points.",
        ),
        ParamSpec(
            "tol",
            1e-10,
            min=1e-15,
            max=1e-1,
            log=True,
            help="Stop when |I_{2m} − I_m| ≤ tol·max(1, |I_{2m}|) (tested from step 2).",
        ),
    )
}

META: dict[str, dict[str, Any]] = {
    "clenshaw_curtis": {
        "family": "integration",
        "name": "Clenshaw–Curtis quadrature",
        "needs": ("f", "interval"),
        "order": "exact for degree ≤ n; ≈ Gauss accuracy unless f is analytic in a large ellipse",
        "summary": (
            "Integrate the polynomial through f at Chebyshev points; doubling n reuses every "
            "old point and gives a free error estimate."
        ),
        "references": (
            "Trefethen, Is Gauss quadrature better than Clenshaw–Curtis?, SIAM Rev. 50 (2008)",
            "Waldvogel, Fast construction of the Fejér and Clenshaw–Curtis rules, BIT 46 (2006)",
        ),
    }
}


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------


def clenshaw_curtis_nodes(n: int) -> NDArray[np.float64]:
    """Chebyshev extreme points t_k = cos(kπ/n), k = 0, …, n (decreasing), on [−1, 1].

    Computed as sin(π(n − 2k)/(2n)): exactly antisymmetric, t_{n/2} = 0 for even n, t_0 = 1,
    t_n = −1, and t_k^{(n)} == t_{2k}^{(2n)} bit for bit (the argument scales by 2 exactly).
    """
    n = _check_int("n", n, 1)
    m = n - 2 * np.arange(n + 1, dtype=np.float64)  # (n+1,) integers, exact in float64
    return np.sin(np.pi * m / (2.0 * n))


def clenshaw_curtis_weights(n: int) -> NDArray[np.float64]:
    """Clenshaw–Curtis weights w_0, …, w_n on [−1, 1] for the nodes cos(kπ/n).

    Waldvogel (2006), §5: w = F_n⁻¹(v + g) (inverse DFT of order n), w_n := w_0, with
        v_k = 2/(1 − 4k²) for 0 ≤ k < ⌊n/2⌋,  v_{⌊n/2⌋} = (n − 3)/(2⌊n/2⌋ − 1) − 1,
        v_{n−k} = v_k                                                     (3.10)
        g_k = −w_0 for 0 ≤ k < ⌊n/2⌋,  g_{⌊n/2⌋} = w_0[(2 − n mod 2)n − 1],  g_{n−k} = g_k
                                                                          (4.2)
        w_0 = 1/(n² − 1 + n mod 2)                                        (2.6)
    built as in Waldvogel's MATLAB ``fejer.m``. n = 1 is the trapezoid rule (1, 1).

    # NOTE: the weights are symmetrized, w_k ← (w_k + w_{n−k})/2, which imposes the exact
    # symmetry of the rule and removes the O(ε) asymmetry of the FFT (as numopt does for Gauss).
    """
    n = _check_int("n", n, 1)
    if n == 1:
        return np.array([1.0, 1.0])
    odd = np.arange(1, n, 2, dtype=np.float64)  # (l,) = 1, 3, …, ≤ n−1
    n_odd = odd.size
    n_rest = n - n_odd
    v0 = np.concatenate([2.0 / odd / (odd - 2.0), [1.0 / odd[-1]], np.zeros(n_rest)])  # (n+1,)
    v2 = -v0[:-1] - v0[:0:-1]  # (n,) = v of (3.10)
    g0 = -np.ones(n)
    g0[n_odd] += n
    g0[n_rest] += n
    g = g0 / (n * n - 1 + n % 2)  # (n,) = g of (4.2)
    w = np.fft.ifft(v2 + g).real  # (n,) = w_0, …, w_{n−1}
    w = np.append(w, w[0])  # (n+1,) periodicity w_n = w_0
    return 0.5 * (w + w[::-1])


def clenshaw_curtis_rule(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes t_k (decreasing) and weights w_k of the (n + 1)-point Clenshaw–Curtis rule on [−1, 1]."""
    return clenshaw_curtis_nodes(n), clenshaw_curtis_weights(n)


def chebyshev_coefficients(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coefficients a_0, …, a_n of the interpolant p = Σ a_j T_j through (cos(kπ/n), values[k]).

    DCT-I by an FFT of the even extension (Trefethen 2008, §2, ``clenshaw_curtis.m``):
    g = Re FFT([f_0, …, f_n, f_{n−1}, …, f_1])/(2n), a_0 = g_0, a_j = 2g_j (0 < j < n), a_n = g_n.
    """
    f = np.asarray(values, dtype=np.float64)  # (n+1,)
    n = f.size - 1
    if n < 1:
        raise ValueError("need at least two values")
    ext = np.concatenate([f, f[-2:0:-1]])  # (2n,) even extension in θ
    g = np.fft.fft(ext).real / (2.0 * n)  # (2n,)
    a = 2.0 * g[: n + 1]
    a[0] = g[0]
    a[n] = g[n]
    return a


def integral_from_coefficients(a: NDArray[np.float64]) -> float:
    """∫₋₁¹ Σ a_j T_j(x) dx = Σ_{j even} a_j · 2/(1 − j²)."""
    j = np.arange(0, a.size, 2, dtype=np.float64)
    return math.fsum((a[::2] * (2.0 / (1.0 - j * j))).tolist())


# --------------------------------------------------------------------------------------
# Helpers (local copies of the numopt.integration conventions, so the study is self-contained)
# --------------------------------------------------------------------------------------


def _check_int(name: str, value: int, lo: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < lo:
        raise ValueError(f"{name} must be an integer ≥ {lo}, got {value!r}")
    return int(value)


def _check_tol(tol: float) -> float:
    if not (math.isfinite(tol) and tol > 0.0):
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    return float(tol)


def _safe_eval(f: Callable[[float], Any], x: float) -> float:
    """f(x) as a float; NaN when the evaluation fails with an arithmetic/domain error."""
    try:
        with np.errstate(all="ignore"):
            return float(f(x))
    except (ArithmeticError, ValueError):
        return math.nan


class _CachedF:
    """Counted evaluation of f with a cache keyed by the abscissa (exact float equality)."""

    def __init__(self, f: Callable[[float], Any]) -> None:
        self.counted = Counted(f)
        self.cache: dict[float, float] = {}

    def __call__(self, x: float) -> float:
        v = self.cache.get(x)
        if v is None:
            v = _safe_eval(self.counted, x)
            self.cache[x] = v
        return v

    @property
    def n(self) -> int:
        return self.counted.n


def _resolve_interval(problem: Problem, bracket: tuple[float, float] | None) -> tuple[float, float]:
    ab = bracket if bracket is not None else problem.domain
    if ab is None or len(ab) != 2:
        raise ValueError(f"{problem.id}: an integration interval (a, b) is required")
    a, b = float(ab[0]), float(ab[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"integration interval must be finite, got ({a}, {b})")
    if not a < b:
        raise ValueError(f"invalid integration interval: need a < b, got ({a}, {b})")
    return a, b


# --------------------------------------------------------------------------------------
# The method
# --------------------------------------------------------------------------------------


def clenshaw_curtis(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 2,
    max_levels: int = 12,
    tol: float = 1e-10,
) -> Result:
    """Clenshaw–Curtis quadrature on nested Chebyshev grids n, 2n, 4n, … (Trefethen 2008, (2.2)–(2.3)).

    Step k applies the (n_k + 1)-point rule I_{n_k} = Σ_j ((b − a)/2)·w_j·f(x_j) with n_k = n·2^k,
    x_j = (a + b)/2 + ((b − a)/2)·cos(jπ/n_k), and the weights of Waldvogel (2006), §5. The nested
    grids reuse every old node, so the run costs n_K + 1 evaluations of f in total.

    Stopping: converged=True at the first k ≥ 2 with |I_{n_k} − I_{n_{k−1}}| ≤ tol·max(1, |I_{n_k}|);
    converged=False after ``max_levels`` doublings without passing, or on a non-finite value.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    n = _check_int("n", n, 1)
    max_levels = _check_int("max_levels", max_levels, 2)
    tol = _check_tol(tol)
    if n * 2**max_levels > MAX_DEGREE:
        raise ValueError(f"n·2^max_levels = {n * 2**max_levels} exceeds the limit {MAX_DEGREE}")
    exact = None if prob.exact is None else float(prob.exact)
    fc = _CachedF(prob.f)
    half, mid = 0.5 * (b - a), 0.5 * (a + b)

    trace: list[Step] = []
    estimate = math.nan
    err_est: float | None = None
    for k in range(max_levels + 1):
        n_k = n * 2**k
        t, w = clenshaw_curtis_rule(n_k)  # (n_k+1,), (n_k+1,)
        x = mid + half * t
        x[0], x[-1] = b, a  # exact end points (t_0 = 1, t_n = −1)
        weights = (half * w).tolist()
        n_before = fc.n
        values = [fc(float(xi)) for xi in x]
        prev = estimate
        estimate = math.fsum(wi * v for wi, v in zip(weights, values, strict=True))
        if k > 0:
            err_est = abs(estimate - prev)
        finite = math.isfinite(estimate) and all(math.isfinite(v) for v in values)
        coeffs = chebyshev_coefficients(np.array(values)).tolist() if finite else []
        points = [[float(xi), v] for xi, v in zip(x, values, strict=True)]
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                info={
                    "estimate": estimate,
                    "error": None if exact is None else abs(estimate - exact),
                    "err_est": err_est,
                    "n": n_k,
                    "n_points": n_k + 1,
                    "nodes": points,
                    "weights": weights,
                    "cheb_coeffs": coeffs,
                    "new_nodes": fc.n - n_before,
                },
            )
        )
        extra = {
            "exact": exact,
            "error": None if exact is None else abs(estimate - exact),
            "err_est": err_est,
            "n_points": n_k + 1,
        }
        if not finite:
            bad = next((p[0] for p in points if not math.isfinite(p[1])), None)
            msg = "the estimate overflowed" if bad is None else f"f is not finite at x = {bad:.6g}"
            return Result(
                "clenshaw_curtis", estimate, estimate, False, msg, k, fc.n, trace=trace, extra=extra
            )
        if k >= 2 and err_est is not None and err_est <= tol * max(1.0, abs(estimate)):
            return Result(
                "clenshaw_curtis",
                estimate,
                estimate,
                True,
                f"|I_{n_k} − I_{n_k // 2}| = {err_est:.3g} ≤ tol·max(1, |I|) with {n_k + 1} points",
                k,
                fc.n,
                trace=trace,
                extra=extra,
            )
    n_final = n * 2**max_levels
    return Result(
        "clenshaw_curtis",
        estimate,
        estimate,
        False,
        f"reached max_levels={max_levels} ({n_final + 1} points) with "
        f"|I_n − I_(n/2)| = {err_est:.3g} > tol·max(1, |I|)",
        max_levels,
        fc.n,
        trace=trace,
        extra={
            "exact": exact,
            "error": None if exact is None else abs(estimate - exact),
            "err_est": err_est,
            "n_points": n_final + 1,
        },
    )
