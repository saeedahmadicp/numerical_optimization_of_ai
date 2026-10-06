"""Numerical integration (quadrature) of I = ∫ₐᵇ f(x) dx.

The interval is ``problem.domain`` unless the keyword ``bracket=(a, b)`` overrides it
(``a < b`` is required). The exact value ``problem.exact`` is used only to report errors.

Method groups:

* **Composite rules** (``left_riemann``, ``right_riemann``, ``midpoint_rule``, ``trapezoid``,
  ``simpson``, ``simpson_38``, ``boole``): Step k is the rule on N_k = n·2^k subintervals of
  width h_k = (b − a)/N_k, k = 0, …, ``levels``. If the rule's error is c·h^p + O(h^{p+1}),
  the Richardson estimate of the error of I_k is d_k/(2^p − 1) with d_k = |I_k − I_{k−1}|
  (Burden & Faires, §4.5; Dahlquist & Björck, §5.2.3). The sweep is not stopped early, so
  the trace always shows every level; ``converged`` is True when ``levels`` ≥ 2, the error
  estimate of the final level is ≤ tol·max(1, |I|), and the asymptotic regime is confirmed
  (below).
* ``romberg``: Step k is row k of the Romberg table (Burden & Faires, Alg. 4.2); converged
  when two consecutive diagonal differences pass and the trapezoid column is confirmed.
* ``gauss_legendre``: Step k is the (k + 1)-point Gauss–Legendre rule, nodes and weights from
  the Golub–Welsch eigenproblem, for k + 1 = 1, …, n; converged needs n ≥ 3.
* ``adaptive_simpson``: Step k is the k-th interval accepted by the Lyness test on two
  levels (the interval and its parent), in order (after a forced bisection to ``min_depth``).
* ``monte_carlo_integration``: Step k uses the first n·2^k uniform samples.
* **Nested rules** (``clenshaw_curtis``, ``gauss_patterson``): Step k is the Clenshaw–Curtis
  rule on n·2^k + 1 Chebyshev points, or the Gauss–Kronrod–Patterson rule on 2^{k+1} − 1
  points. Every node of step k − 1 is reused, and d_k = |I_k − I_{k−1}| is the error
  estimate; converged at the first k ≥ 2 with d_k ≤ tol·max(1, |I_k|) (the stopping test of
  the study research/clenshaw-curtis-vs-gauss, see the section comment).

**Error estimate of a sequence** (composite rules with R = 2^p, Gauss without R). The order
p needs a smooth f. For √x every rule of order p ≥ 2 (midpoint, trapezoid, Simpson, 3/8,
Boole) and Romberg converge only like h^{3/2}, so d_k shrinks by 2^{3/2}, not 2^p, per level
and d_k/(R − 1) underestimates the error (34× for Boole); the Riemann sums keep their O(h),
which is slower than h^{3/2}, and Gauss converges like n^{−3}. At a kink, the
estimates of two levels can coincide (midpoint rule on |x − 0.3|: d_k = 0 while the error
is 0.0025), and Gauss rules converge irregularly. With ρ_k = d_{k−1}/d_k:

* k = 1, or d_k and d_{k−1} both at the rounding level 4ε(A_k + A_{k−1}),
  A_k = Σᵢ|wᵢ f(xᵢ)|: the textbook value d_k/(R − 1) (Gauss: d_k);
* ρ_k ≤ 1 (no contraction): d_k;
* otherwise max(d_k/(R − 1), d_k/(ρ_k − 1), d_{k−1}/(R(R − 1))): the textbook value, the
  geometric tail with the observed ratio (larger when the contraction is slower than R;
  Dahlquist & Björck (2008), §3.4) and the textbook prediction from the previous
  difference (larger when d_k is small by accident); Gauss uses max(d_k, d_k/(ρ_k − 1));
* Gauss: at least max(d_k, d_{k−1}, d_{k−2}).

**Confirmed asymptotic regime** (composite rules, and the trapezoid column of Romberg). Every
estimate above extrapolates the observed differences, so it is valid only when they already
follow I_k = I + C·h_k^p·(1 + o(1)). Then every signed difference δ_k = I_k − I_{k−1} has
the sign of −C and ρ_k settles at a fixed rate. The regime is confirmed when the last four
δ have one sign and the last three ratios ρ are > 1 and agree with each other within a
factor 2, or when the last two d are at the rounding level (the ratio tests of de Boor's
CADRE, 1971). A composite rule therefore needs ``levels`` ≥ 4, or two differences at the
rounding level, to converge.

# NOTE: Burden & Faires use d_k/(R − 1) (and |G_n − G_{n−1}| for Gauss) alone and need no
# confirmation; the added terms only enlarge the estimate, and the confirmation only
# withholds ``converged``. On the library problems, the estimate alone gave no error above
# 10× the estimate, but before the grid resolves f the differences can contract at a
# rate near R by accident while the error grows: Simpson on 1/(1 + 100x²), N = 4, 8, 16, has
# errors 0.094, 0.0095, 0.0129, and the estimate 4.3e-4 passed tol = 1e-3. In a sweep on the
# library, 1/(1 + c·x²) (40 values of c in [10, 1000] on [−1, 1]), exp(−c(x − x_c)²)
# (40 pairs, c in [10, 2000], x_c in [0, 1]), x^α (10 values of α in [0.1, 0.9]) and five
# other integrands (exp(sin 3x), tanh(5(x − 0.3)), log(1.0001 + x²), 1/(1.1 − x),
# sin 50x), with tol = 1e-12, …, 1e-1: without the confirmation the composite rules
# converged 66 900 times, 118 times with an error above 10·tol·max(1, |I|); with it they
# converged 33 718 times, with none. It is still not a rigorous bound (see Preconditions).

**Preconditions.** Every method uses only values of f at its nodes, so no rule can detect a
feature of f that the nodes do not resolve:

1. The grids must resolve the length scale of f. A peak narrower than the node spacing can
   be missed entirely, and a periodic f whose period aliases with the grid gives *equal*
   estimates on coarse grids: cos(32πx) on [0, 1] is 1 at every node of N ≤ 32 dyadic
   subintervals, so the trapezoid rule (n = 4, levels = 2) and Romberg report I = 1 with
   d_k = 0, which is at the rounding level, and converge although I = 0. Gauss (non-nested
   nodes) had no such result in the sweeps.
2. f must be continuous for the Riemann and midpoint sums. At a jump, refinement often
   keeps the same nodes on each side of the jump, so consecutive estimates are equal
   (d_k = 0) at a wrong value. The same holds at a kink for the midpoint rule: for
   f = |x − c| its error is dist(c, grid)², and refinement keeps the same nearest node for
   several levels (n = 13: N = 13, 26, 52 all err by 5.9e-5), so every estimate built from
   them is 0.

Function values are cached by abscissa, so ``n_fev`` counts *distinct* points: the nested
grids of the closed composite rules reuse every old node (N_K + 1 evaluations in all).
Sums are computed with ``math.fsum`` (correctly rounded); when it raises (``inf + (−inf)``,
or a partial sum that overflows) the plain IEEE sum is used. Evaluations that raise an
arithmetic error or return NaN/±inf, and estimates that overflow, stop the method with
``converged=False``.

``Step.x`` and ``Step.fun`` both hold the current estimate of I. Errors are absolute. To
keep a trace at a few MB, a Step stores its nodes, weights and panels (or samples) only up
to ``MAX_DISPLAY`` = 4096 points.

Info keys:
    estimate: float — the current approximation of I (same as ``Step.fun``).
    error: float | None — |estimate − exact|, or None when the exact value is unknown.
    err_est: float | None — a computable estimate of the error (the observed-ratio Richardson
        estimate above for composite rules and Gauss, |R(k,k) − R(k−1,k−1)| for Romberg, the
        Lyness estimate |S₂ − S₁|/15 of the accepted interval for adaptive Simpson, the
        standard error (b − a)·s/√N for Monte Carlo, d_k = |I_k − I_{k−1}| for the nested
        rules); None when no estimate exists yet (k = 0).
    ratio: float | None — the observed ratio ρ_k = d_{k−1}/d_k of successive differences
        (composite rules, Gauss; Romberg: of the trapezoid column R(·, 0)); None for k < 2 or
        d_k = 0. For a composite rule of order p it tends to 2^p for smooth f (4 for the
        Romberg column); log₂ ρ_k is the observed order.
    confirmed: bool — the asymptotic regime is confirmed at this step (composite rules: of
        the sequence I_k; Romberg: of the trapezoid column), see "Confirmed asymptotic
        regime"; False for k < 2.
    h: float — subinterval width (composite rules, Romberg).
    n_panels: int — number of subintervals N_k (composite rules, Romberg).
    panels: [[x_left, x_right]] | None — the basic-rule panels: one per subinterval for
        Riemann, midpoint and trapezoid, two subintervals per Simpson panel, three per
        Simpson-3/8 panel, four per Boole panel (composite rules); None when the rule uses
        more than MAX_DISPLAY nodes.
    nodes: [[x, f(x)]] | None — every node the estimate uses (composite rules, Gauss, nested
        rules, in decreasing x for the nested rules), the new nodes of this row (Romberg), or
        the five nodes of the accepted interval (adaptive Simpson); None when there are more
        than MAX_DISPLAY (composite rules, Romberg, Clenshaw–Curtis).
    weights: [float] | None — quadrature weights aligned with ``nodes`` (composite rules,
        Gauss, nested rules); estimate = Σ weights[i]·nodes[i][1]; None when ``nodes`` is
        None.
    row: [float] — row k of the Romberg table, R(k, 0), …, R(k, k) (Romberg).
    n_points: int — number of Gauss points m (Gauss); number of nodes of the rule
        (Clenshaw–Curtis: n_k + 1; Gauss–Patterson: 2^{k+1} − 1).
    n: int — polynomial degree n_k = n·2^k of the rule (Clenshaw–Curtis).
    cheb_coeffs: [float] | None — Chebyshev coefficients a_0, …, a_{n_k} of the interpolant
        p(x) = Σ a_j T_j(t(x)), t = (2x − a − b)/(b − a), through the nodes; their decay shows
        the resolution of f and the aliasing that Trefethen (2008, §5) uses to explain the
        accuracy of the rule. None when f is not finite at a node or there are more than
        MAX_DISPLAY nodes (Clenshaw–Curtis).
    new_nodes: int — number of nodes evaluated for the first time at this step (nested
        rules).
    interval: [a_i, b_i] | None — the interval accepted at this step; None at k = 0
        (adaptive Simpson). The accepted intervals so far are those of steps 1, …, k, in order
        (``Result.extra["intervals"]`` lists them all); a per-step copy would make the trace
        O(N²) in the number N of accepted intervals.
    pending: [[a_i, b_i]] — intervals still waiting on the stack, next first; at most one per
        depth, so at most max_depth + 1 entries (adaptive Simpson).
    depth: int — bisection depth of the accepted interval (adaptive Simpson).
    tol_local: float — the tolerance the interval had to meet (adaptive Simpson).
    passed: bool | None — the interval passed its own Lyness test |S₂ − S₁| ≤ 15·tol_local;
        None at k = 0 (adaptive Simpson).
    parent_passed: bool | None — its parent passed its Lyness test (True for the root);
        None at k = 0 (adaptive Simpson).
    forced: bool — True when the interval was accepted without ``passed`` and
        ``parent_passed``, only because ``max_depth`` was reached or it could not be bisected
        further (adaptive Simpson).
    samples: [[x, f(x)]] — the samples added at this step, at most MAX_DISPLAY (the first
        ones drawn at this step) (Monte Carlo).
    n_samples: int — total number of samples N_k used by the estimate (Monte Carlo).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ..core.counting import Counted, scalar_problem
from ..core.registry import ParamSpec, register
from ..core.rng import Rng
from ..core.types import Problem, Result, Step

#: Upper bound on subintervals (or samples) per run, to keep traces and work bounded.
MAX_POINTS = 2**18
MAX_SAMPLES = 2**20
#: The largest number of nodes (or samples) one Step stores for display; above it the
#: per-step geometry is omitted (None) or truncated, so that a trace stays a few MB.
MAX_DISPLAY = 4096


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _fsum(terms: Iterable[float]) -> float:
    """Correctly rounded Σ terms (``math.fsum``), with IEEE semantics when it cannot be.

    ``math.fsum`` raises ValueError on ``inf + (−inf)`` and OverflowError when a partial sum
    overflows; then the plain sum gives the IEEE value (NaN or ±inf), which every caller
    detects as non-finite and reports with ``converged=False``.
    """
    values = list(terms)
    try:
        return math.fsum(values)
    except (ValueError, OverflowError):
        return float(sum(values))


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


def _exact(problem: Problem) -> float | None:
    return None if problem.exact is None else float(problem.exact)


def _error(estimate: float, exact: float | None) -> float | None:
    return None if exact is None else abs(estimate - exact)


def _passes(err_est: float | None, estimate: float, tol: float) -> bool:
    """The documented acceptance test err_est ≤ tol·max(1, |estimate|)."""
    return (
        err_est is not None
        and math.isfinite(err_est)
        and math.isfinite(estimate)
        and err_est <= tol * max(1.0, abs(estimate))
    )


_EPS = float(np.finfo(float).eps)


def _sequence_error(
    diffs: list[float], noises: list[float], rate: float | None
) -> tuple[float, float | None]:
    """Error estimate of the newest term of a sequence I_0, I_1, …, and the observed ratio.

    ``diffs`` holds d_j = |I_j − I_{j−1}| for j = 1, …, k and ``noises`` the rounding level
    of each d_j. ``rate`` is R = 2^p for a composite rule of order p (the error shrinks by R
    per halving of h) and None for Gauss (no fixed rate). With ρ = d_{k−1}/d_k:

    * k = 1, or d_k and d_{k−1} both at the rounding level: the textbook value d_k/(R − 1)
      (d_k for Gauss);
    * ρ ≤ 1 (no contraction): d_k;
    * otherwise the largest of d_k/(R − 1) (textbook Richardson, Burden & Faires §4.5),
      d_k/(ρ − 1) (the geometric tail with the observed ratio, which is larger when the
      sequence contracts more slowly than R, Dahlquist & Björck (2008) §3.4), and
      d_{k−1}/(R(R − 1)) (the textbook prediction from the previous difference, which is
      larger when d_k is small by accident, e.g. d_k = 0);
    * Gauss: also at least max(d_k, d_{k−1}, d_{k−2}), because G_m converges irregularly
      for non-smooth f and one difference can be small by accident.
    """
    d = diffs[-1]
    d_prev = diffs[-2] if len(diffs) >= 2 else None
    ratio = d_prev / d if d_prev is not None and d > 0.0 else None
    base = d / (rate - 1.0) if rate is not None else d
    if d_prev is None or (d <= noises[-1] and d_prev <= noises[-2]):
        return base, ratio
    if ratio is not None and ratio <= 1.0:
        est = d
    else:
        candidates = [base]
        if ratio is not None:
            candidates.append(d / (ratio - 1.0))
        if rate is not None:
            candidates.append(d_prev / (rate * (rate - 1.0)))
        est = max(candidates)
    if rate is None:
        est = max(est, *diffs[-3:])
    return est, ratio


#: Two consecutive observed ratios "agree" when they differ by at most this factor.
_RATIO_SPREAD = 2.0
#: Number of observed ratios that must agree to confirm the asymptotic regime.
_N_RATIOS = 3


def _confirmed(deltas: list[float], noises: list[float]) -> bool:
    """True when the newest terms of a sequence I_0, I_1, … are in their asymptotic regime.

    ``deltas`` holds the *signed* differences δ_j = I_j − I_{j−1}, j = 1, …, k, and
    ``noises`` the rounding level of each |δ_j|. If I_j = I + C·h_j^p·(1 + o(1)) with
    h_j = h_0/2^j, then every δ_j has the sign of −C and |δ_{j−1}|/|δ_j| → 2^p. The regime is
    confirmed when either

    * the last two differences are both at the rounding level (nothing more is observable),
      or
    * the last ``_N_RATIOS`` + 1 = 4 differences have one sign and the last three ratios
      ρ_j = |δ_{j−1}|/|δ_j| are all > 1 and each agrees with the previous one within the
      factor ``_RATIO_SPREAD`` = 2.

    The ratios are compared with each other, not with 2^p, so that an order drop (ρ → 2^{3/2}
    for √x) is still confirmed; the estimate d_k/(ρ_k − 1) then accounts for it. This is the
    idea of the ratio tests of de Boor's CADRE (in Rice (ed.), Mathematical Software, 1971)
    and of Dahlquist & Björck (2008), §3.4: Richardson extrapolation needs an observed,
    stable rate.
    """
    if len(deltas) >= 2 and abs(deltas[-1]) <= noises[-1] and abs(deltas[-2]) <= noises[-2]:
        return True
    if len(deltas) < _N_RATIOS + 1:
        return False
    last = deltas[-(_N_RATIOS + 1) :]
    if not (all(d > 0.0 for d in last) or all(d < 0.0 for d in last)):
        return False
    rho = [abs(d0) / abs(d1) for d0, d1 in pairwise(last)]
    return all(r > 1.0 for r in rho) and all(
        1.0 / _RATIO_SPREAD <= r1 / r0 <= _RATIO_SPREAD for r0, r1 in pairwise(rho)
    )


def _check_int(name: str, value: int, lo: int) -> int:
    if int(value) != value or value < lo:
        raise ValueError(f"{name} must be an integer ≥ {lo}, got {value!r}")
    return int(value)


def _check_tol(tol: float) -> float:
    if not (math.isfinite(tol) and tol > 0.0):
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    return float(tol)


def _first_bad(points: list[list[float]]) -> float | None:
    """The first abscissa whose f value is not finite (None if all are finite)."""
    return next((x for x, fx in points if not math.isfinite(fx)), None)


def _nonfinite_result(
    method: str,
    estimate: float,
    x_bad: float | None,
    k: int,
    fc: _CachedF | Counted,
    trace: list[Step],
    **extra: Any,
) -> Result:
    msg = "the estimate overflowed" if x_bad is None else f"f is not finite at x = {x_bad:.6g}"
    return Result(method, estimate, estimate, False, msg, k, fc.n, trace=trace, extra=extra)


# --------------------------------------------------------------------------------------
# Composite Newton–Cotes rules (and Riemann sums)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompositeRule:
    """A composite rule: each panel of ``span`` subintervals contributes
    (num/den)·h·Σⱼ coeffs[j]·f(x_{panel start + j}); ``midpoint`` uses the panel center."""

    span: int
    coeffs: tuple[int, ...]
    num: int
    den: int
    order: int  # p: the error is O(h^p) for smooth f
    degree: int  # exact for polynomials of degree ≤ degree
    midpoint: bool = False


_RULES: dict[str, _CompositeRule] = {
    "left_riemann": _CompositeRule(1, (1, 0), 1, 1, order=1, degree=0),
    "right_riemann": _CompositeRule(1, (0, 1), 1, 1, order=1, degree=0),
    "midpoint_rule": _CompositeRule(1, (1,), 1, 1, order=2, degree=1, midpoint=True),
    "trapezoid": _CompositeRule(1, (1, 1), 1, 2, order=2, degree=1),
    "simpson": _CompositeRule(2, (1, 4, 1), 1, 3, order=4, degree=3),
    "simpson_38": _CompositeRule(3, (1, 3, 3, 1), 3, 8, order=4, degree=3),
    "boole": _CompositeRule(4, (7, 32, 12, 32, 7), 2, 45, order=6, degree=5),
}


def composite_nodes_weights(
    method: str, a: float, b: float, n_sub: int
) -> tuple[list[float], list[float], list[list[float]]]:
    """Nodes, weights and basic-rule panels of a composite rule on ``n_sub`` subintervals.

    Grid nodes are xᵢ = a + i·h for i < N and x_N = b exactly (h = (b − a)/N). Halving h
    reproduces every old node bit for bit, which makes the evaluation cache exact.
    """
    rule = _RULES[method]
    if n_sub % rule.span:
        raise ValueError(f"{method}: the number of subintervals must be a multiple of {rule.span}")
    h = (b - a) / n_sub

    def grid(i: int) -> float:
        return b if i == n_sub else a + i * h

    m = rule.span
    panels = [[grid(p * m), grid((p + 1) * m)] for p in range(n_sub // m)]
    if rule.midpoint:
        nodes = [a + (i + 0.5) * h for i in range(n_sub)]
        return nodes, [h] * n_sub, panels
    counts = np.zeros(n_sub + 1, dtype=np.int64)
    for j, c in enumerate(rule.coeffs):
        if c:
            counts[j : n_sub - m + j + 1 : m] += c
    used = np.flatnonzero(counts)
    nodes = [grid(int(i)) for i in used]
    weights = [h * float(counts[i] * rule.num) / rule.den for i in used]
    return nodes, weights, panels


def _composite(
    method: str,
    problem: Problem | Callable[[float], float],
    bracket: tuple[float, float] | None,
    n: int,
    levels: int,
    tol: float,
) -> Result:
    rule = _RULES[method]
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    n = _check_int("n", n, rule.span)
    if n % rule.span:
        raise ValueError(f"{method}: n must be a multiple of {rule.span}, got {n}")
    levels = _check_int("levels", levels, 0)
    tol = _check_tol(tol)
    if n * 2**levels > MAX_POINTS:
        raise ValueError(f"n·2^levels = {n * 2**levels} exceeds the limit {MAX_POINTS}")
    exact = _exact(prob)
    fc = _CachedF(prob.f)

    trace: list[Step] = []
    estimate = math.nan
    err_est: float | None = None
    ratio: float | None = None
    confirmed = False
    diffs: list[float] = []  # d_j = |I_j − I_{j−1}|
    deltas: list[float] = []  # δ_j = I_j − I_{j−1} (signed)
    noises: list[float] = []  # rounding level of each d_j
    mass_prev = math.nan
    for k in range(levels + 1):
        n_sub = n * 2**k
        h = (b - a) / n_sub
        nodes, weights, panels = composite_nodes_weights(method, a, b, n_sub)
        values = [fc(x) for x in nodes]
        prev = estimate
        estimate = _fsum(w * v for w, v in zip(weights, values, strict=True))
        mass = _fsum(abs(w * v) for w, v in zip(weights, values, strict=True))
        finite = math.isfinite(estimate) and math.isfinite(mass)
        if k > 0 and finite:
            deltas.append(estimate - prev)
            diffs.append(abs(deltas[-1]))
            noises.append(4.0 * _EPS * (mass + mass_prev))
            err_est, ratio = _sequence_error(diffs, noises, 2.0**rule.order)
            confirmed = _confirmed(deltas, noises)
        mass_prev = mass
        points = [[x, v] for x, v in zip(nodes, values, strict=True)]
        shown = len(nodes) <= MAX_DISPLAY
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                step_size=h,
                info={
                    "estimate": estimate,
                    "error": _error(estimate, exact),
                    "err_est": err_est,
                    "ratio": ratio,
                    "confirmed": confirmed,
                    "h": h,
                    "n_panels": n_sub,
                    "panels": panels if shown else None,
                    "nodes": points if shown else None,
                    "weights": weights if shown else None,
                },
            )
        )
        if not finite or not all(math.isfinite(v) for v in values):
            return _nonfinite_result(
                method, estimate, _first_bad(points), k, fc, trace, exact=exact
            )

    extra = {
        "exact": exact,
        "error": _error(estimate, exact),
        "err_est": err_est,
        "order": rule.order,
        "degree": rule.degree,
        "n_panels": n * 2**levels,
    }
    if levels <= 1:
        msg = (
            "levels = 0: a single estimate has no error estimate"
            if levels == 0
            else "levels = 1: one difference cannot confirm the order p; use levels ≥ 2"
        )
        return Result(
            method, estimate, estimate, False, msg, levels, fc.n, trace=trace, extra=extra
        )
    passed = _passes(err_est, estimate, tol)
    converged = passed and confirmed
    if converged:
        msg = f"error estimate {err_est:.3g} ≤ tol·max(1, |I|), asymptotic regime confirmed"
    elif passed:
        msg = (
            f"error estimate {err_est:.3g} ≤ tol·max(1, |I|), but the asymptotic regime is not "
            "confirmed (the last 3 ratios ρ disagree or the differences change sign); "
            "increase levels"
        )
    else:
        msg = f"error estimate {err_est:.3g} > tol·max(1, |I|); increase n or levels"
    return Result(
        method, estimate, estimate, converged, msg, levels, fc.n, trace=trace, extra=extra
    )


#: The largest ``levels`` of a composite rule; with ``n`` ≤ the n_max of _composite_params,
#: every combination in the UI ranges has n·2^levels ≤ MAX_POINTS.
_LEVELS_MAX = 12


def _composite_params(
    n_default: int, n_min: int, n_help: str, n_max: int = MAX_POINTS >> _LEVELS_MAX
) -> tuple[ParamSpec, ...]:
    """ParamSpecs of a composite rule. NOTE: the ranges are joint-consistent with the
    MAX_POINTS check of _composite: n_max·2^_LEVELS_MAX ≤ MAX_POINTS (n ≤ 64 = 2^18/2^12; a
    larger N comes from levels), so no slider position raises."""
    assert n_max << _LEVELS_MAX <= MAX_POINTS
    return (
        ParamSpec("n", n_default, kind="int", min=n_min, max=n_max, help=n_help),
        ParamSpec(
            "levels",
            6,
            kind="int",
            min=0,
            max=_LEVELS_MAX,
            help="Number of halvings: step k uses n·2ᵏ subintervals.",
        ),
        ParamSpec(
            "tol",
            1e-8,
            min=1e-15,
            max=1e-1,
            log=True,
            help="Converged when the final error estimate ≤ tol·max(1, |I|).",
        ),
    )


_N_HELP = "Subintervals at step 0."


@register(
    id="left_riemann",
    family="integration",
    name="Left Riemann sum",
    params=_composite_params(4, 1, _N_HELP),
    needs=("f", "interval"),
    order="O(h); exact for constants",
    summary="Add up rectangles whose height is f at the left end of each subinterval.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §4.3–4.4",),
)
def left_riemann(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite left Riemann sum L_N = h·Σ_{i=0}^{N−1} f(xᵢ), xᵢ = a + i·h.

    Error: I − L_N = (b − a)·h·f′(ξ)/2 for some ξ ∈ (a, b) (f ∈ C¹), so the rule has order
    p = 1 and is exact only for constants (degree 0). Stopping: the fixed sweep
    k = 0, …, levels; converged when the error estimate of the last level (Richardson
    d_K/(2¹ − 1), enlarged by the observed ratio; module docstring) is ≤ tol·max(1, |L|)
    and the asymptotic regime is confirmed (``levels`` ≥ 4, or two differences at the
    rounding level). Precondition: f continuous (module docstring, "Preconditions").
    """
    return _composite("left_riemann", problem, bracket, n, levels, tol)


@register(
    id="right_riemann",
    family="integration",
    name="Right Riemann sum",
    params=_composite_params(4, 1, _N_HELP),
    needs=("f", "interval"),
    order="O(h); exact for constants",
    summary="Add up rectangles whose height is f at the right end of each subinterval.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §4.3–4.4",),
)
def right_riemann(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite right Riemann sum R_N = h·Σ_{i=1}^{N} f(xᵢ).

    Error: I − R_N = −(b − a)·h·f′(ξ)/2, order p = 1, degree 0. Stopping as for
    ``left_riemann``.
    """
    return _composite("right_riemann", problem, bracket, n, levels, tol)


@register(
    id="midpoint_rule",
    family="integration",
    name="Composite midpoint rule",
    params=_composite_params(4, 1, _N_HELP),
    needs=("f", "interval"),
    order="O(h²); exact for degree ≤ 1",
    summary="Rectangles whose height is f at the center of each subinterval.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Theorem 4.6",),
)
def midpoint_rule(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite midpoint rule M_N = h·Σ_{i=0}^{N−1} f(a + (i + ½)h).

    Error (Burden & Faires, Thm. 4.6): I − M_N = (b − a)·h²·f″(μ)/24, order p = 2, exact for
    degree ≤ 1. The midpoints of the halved grid are all new points, so step k costs N_k
    evaluations. Stopping as for ``left_riemann`` with the factor 2² − 1 = 3.

    At a kink the error does not change while the nearest grid node to the kink stays the
    same, so consecutive estimates can be identical and the rule can report convergence with
    a wrong value (module docstring, "Preconditions", item 2).
    """
    return _composite("midpoint_rule", problem, bracket, n, levels, tol)


@register(
    id="trapezoid",
    family="integration",
    name="Composite trapezoidal rule",
    params=_composite_params(4, 1, _N_HELP),
    needs=("f", "interval"),
    order="O(h²); exact for degree ≤ 1",
    summary="Join neighboring points with straight lines and add up the trapezoids.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Theorem 4.5",),
)
def trapezoid(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite trapezoidal rule T_N = h·[f(x₀)/2 + f(x₁) + … + f(x_{N−1}) + f(x_N)/2].

    Error (Burden & Faires, Thm. 4.5): I − T_N = −(b − a)·h²·f″(μ)/12, order p = 2, exact for
    degree ≤ 1. Stopping as for ``left_riemann`` with the factor 2² − 1 = 3.
    """
    return _composite("trapezoid", problem, bracket, n, levels, tol)


@register(
    id="simpson",
    family="integration",
    name="Composite Simpson's rule",
    params=_composite_params(4, 2, "Subintervals at step 0 (must be even)."),
    needs=("f", "interval"),
    order="O(h⁴); exact for degree ≤ 3",
    summary="Fit a parabola through each pair of subintervals and integrate it exactly.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Theorem 4.4, Alg. 4.1",),
)
def simpson(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite Simpson rule S_N = (h/3)·[f₀ + 4f₁ + 2f₂ + 4f₃ + … + 4f_{N−1} + f_N], N even.

    Error (Burden & Faires, Thm. 4.4): I − S_N = −(b − a)·h⁴·f⁽⁴⁾(μ)/180, order p = 4, exact
    for degree ≤ 3. Stopping as for ``left_riemann`` with the factor 2⁴ − 1 = 15.
    """
    return _composite("simpson", problem, bracket, n, levels, tol)


@register(
    id="simpson_38",
    family="integration",
    name="Composite Simpson's 3/8 rule",
    params=_composite_params(3, 3, "Subintervals at step 0 (must be a multiple of 3).", 63),
    needs=("f", "interval"),
    order="O(h⁴); exact for degree ≤ 3",
    summary="Fit a cubic through each group of three subintervals and integrate it exactly.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §4.3 (closed Newton–Cotes, n = 3)",
    ),
)
def simpson_38(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 3,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite Simpson 3/8 rule: (3h/8)·[f₀ + 3f₁ + 3f₂ + 2f₃ + 3f₄ + … + 3f_{N−1} + f_N].

    The basic rule (3h/8)(f₀ + 3f₁ + 3f₂ + f₃) has error −(3/80)h⁵f⁽⁴⁾(ξ) (Burden & Faires,
    §4.3), so the composite rule has order p = 4 and is exact for degree ≤ 3. N must be a
    multiple of 3. Stopping as for ``left_riemann`` with the factor 2⁴ − 1 = 15.
    """
    return _composite("simpson_38", problem, bracket, n, levels, tol)


@register(
    id="boole",
    family="integration",
    name="Composite Boole's rule",
    params=_composite_params(4, 4, "Subintervals at step 0 (must be a multiple of 4)."),
    needs=("f", "interval"),
    order="O(h⁶); exact for degree ≤ 5",
    summary="Fit a quartic through each group of four subintervals and integrate it exactly.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §4.3 (closed Newton–Cotes, n = 4)",
    ),
)
def boole(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 4,
    levels: int = 6,
    tol: float = 1e-8,
) -> Result:
    """Composite Boole rule: (2h/45)·[7f₀ + 32f₁ + 12f₂ + 32f₃ + 14f₄ + … + 32f_{N−1} + 7f_N].

    The basic rule has error −(8/945)h⁷f⁽⁶⁾(ξ) (Burden & Faires, §4.3), so the composite rule
    has order p = 6 and is exact for degree ≤ 5. N must be a multiple of 4. Stopping as for
    ``left_riemann`` with the factor 2⁶ − 1 = 63.
    """
    return _composite("boole", problem, bracket, n, levels, tol)


# --------------------------------------------------------------------------------------
# Romberg integration
# --------------------------------------------------------------------------------------

#: Row k uses 2^k subintervals; 2^18 = MAX_POINTS, the cap of the composite rules.
_ROMBERG_LEVELS_MAX = MAX_POINTS.bit_length() - 1


@register(
    id="romberg",
    family="integration",
    name="Romberg integration",
    params=(
        ParamSpec(
            "tol",
            1e-10,
            min=1e-15,
            max=1e-1,
            log=True,
            help=(
                "Converged when |R(k,k) − R(k−1,k−1)| ≤ tol·max(1, |R(k,k)|) at two consecutive "
                "k ≥ 2 and the trapezoid column is in its asymptotic regime."
            ),
        ),
        ParamSpec(
            "max_levels",
            16,
            kind="int",
            min=3,
            max=_ROMBERG_LEVELS_MAX,
            help="Maximum number of rows − 1 (row k uses 2ᵏ subintervals; tested from k = 3).",
        ),
    ),
    needs=("f", "interval"),
    order="O(h^{2k+2}) in row k for smooth f",
    summary="Extrapolate trapezoid estimates on halved grids to h = 0 (Richardson in h²).",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 4.2",
        "Dahlquist & Björck, Numerical Methods in Scientific Computing I (2008), §5.2.3",
        "de Boor, CADRE: an algorithm for numerical quadrature, in Rice (ed.), "
        "Mathematical Software (1971)",
    ),
)
def romberg(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    tol: float = 1e-10,
    max_levels: int = 16,
) -> Result:
    """Romberg integration (Burden & Faires, Alg. 4.2).

    R(k, 0) is the trapezoid rule with 2^k subintervals, h_k = (b − a)/2^k, computed
    recursively: R(k, 0) = R(k−1, 0)/2 + h_k·Σ_{i=1}^{2^{k−1}} f(a + (2i − 1)h_k). The
    Euler–Maclaurin expansion T(h) = I + c₁h² + c₂h⁴ + … (f smooth) justifies
    R(k, j) = R(k, j−1) + (R(k, j−1) − R(k−1, j−1))/(4^j − 1). R(1, 1) is Simpson's rule and
    R(2, 2) is Boole's rule; R(k, k) is exact for polynomials of degree ≤ 2k + 1.

    Stopping: converged at the first k ≥ 3 at which

    1. |R(k, k) − R(k−1, k−1)| ≤ tol·max(1, |R(k, k)|) holds at k and at k − 1 (k − 1 ≥ 2),
       and
    2. the trapezoid column R(·, 0) is in its asymptotic regime (:func:`_confirmed`: its last
       four differences have one sign and the last three ratios agree within a factor 2, or
       its last two differences are at the rounding level).

    Otherwise ``max_levels`` rows (≤ 18, i.e. ≤ MAX_POINTS + 1 evaluations) end the run with
    ``converged=False`` (always so for ``max_levels`` ≤ 2, which the UI range excludes).

    # NOTE: Burden & Faires and Numerical Recipes' qromb stop at the first k with test 1
    # (qromb from k = 2, because the first two rows can agree by accident, e.g. when f
    # vanishes at a, (a+b)/2 and b). Before the trapezoid rule resolves f, the diagonal can
    # still agree by accident: on 1/(1 + 48.3x²) with tol = 1.3e-3 it stopped at k = 3 with
    # an error of 0.030 (23·tol). Test 2 is the ratio check of de Boor's CADRE (1971), which
    # extrapolates only after the T column shows its rate. In a sweep on the library,
    # 1/(1 + c·x²), exp(−c(x − x_c)²), x^α and five other integrands (module docstring),
    # test 1 alone converged 3 365 times, 124 times with an error above 10·tol·max(1, |I|);
    # tests 1 and 2 converged 2 805 times, with none.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    tol = _check_tol(tol)
    max_levels = _check_int("max_levels", max_levels, 1)
    if 2**max_levels > MAX_POINTS:
        raise ValueError(
            f"max_levels = {max_levels}: 2^max_levels exceeds the limit {MAX_POINTS} "
            f"(use max_levels ≤ {_ROMBERG_LEVELS_MAX})"
        )
    exact = _exact(prob)
    fc = _CachedF(prob.f)

    fa, fb = fc(a), fc(b)
    h = b - a
    row = [0.5 * h * (fa + fb)]
    mass = 0.5 * h * (abs(fa) + abs(fb))  # A_0 = Σ|wᵢ f(xᵢ)| of the trapezoid rule
    trace = [
        Step(
            0,
            row[0],
            row[0],
            step_size=h,
            info={
                "estimate": row[0],
                "error": _error(row[0], exact),
                "err_est": None,
                "ratio": None,
                "confirmed": False,
                "h": h,
                "n_panels": 1,
                "row": list(row),
                "nodes": [[a, fa], [b, fb]],
            },
        )
    ]
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return _nonfinite_result("romberg", row[0], a if not math.isfinite(fa) else b, 0, fc, trace)
    if not (math.isfinite(row[0]) and math.isfinite(mass)):
        return _nonfinite_result("romberg", row[0], None, 0, fc, trace)

    err_est = math.inf
    deltas: list[float] = []  # δ_k = R(k, 0) − R(k−1, 0), the trapezoid column
    noises: list[float] = []  # rounding level 4ε(A_k + A_{k−1}) of each |δ_k|
    passed_prev = False
    passed = confirmed = False
    for k in range(1, max_levels + 1):
        n_sub = 2**k
        h = (b - a) / n_sub
        new_x = [a + (2 * i - 1) * h for i in range(1, n_sub // 2 + 1)]
        new_f = [fc(x) for x in new_x]
        prev, mass_prev = row, mass
        row = [0.5 * prev[0] + h * _fsum(new_f)]
        mass = 0.5 * mass_prev + h * _fsum(abs(v) for v in new_f)
        for j in range(1, k + 1):
            row.append(row[j - 1] + (row[j - 1] - prev[j - 1]) / (4.0**j - 1.0))
        estimate = row[k]
        err_est = abs(estimate - prev[k - 1])
        finite = all(math.isfinite(v) for v in row) and math.isfinite(mass)
        ratio: float | None = None
        if finite:
            deltas.append(row[0] - prev[0])
            noises.append(4.0 * _EPS * (mass + mass_prev))
            if k >= 2 and deltas[-1] != 0.0:
                ratio = abs(deltas[-2]) / abs(deltas[-1])
            confirmed = _confirmed(deltas, noises)
            passed_prev, passed = passed, k >= 2 and _passes(err_est, estimate, tol)
        points = [[x, v] for x, v in zip(new_x, new_f, strict=True)]
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                step_size=h,
                info={
                    "estimate": estimate,
                    "error": _error(estimate, exact),
                    "err_est": err_est,
                    "ratio": ratio,
                    "confirmed": confirmed,
                    "h": h,
                    "n_panels": n_sub,
                    "row": list(row),
                    "nodes": points if len(points) <= MAX_DISPLAY else None,
                },
            )
        )
        if not all(math.isfinite(v) for v in new_f):
            return _nonfinite_result("romberg", estimate, _first_bad(points), k, fc, trace)
        if not finite:
            return _nonfinite_result("romberg", estimate, None, k, fc, trace)
        if passed and passed_prev and confirmed:
            return Result(
                "romberg",
                estimate,
                estimate,
                True,
                f"|R(k,k) − R(k−1,k−1)| = {err_est:.3g} ≤ tol·max(1, |I|) at k = {k - 1} and "
                f"k = {k}, trapezoid column in its asymptotic regime",
                k,
                fc.n,
                trace=trace,
                extra={"exact": exact, "error": _error(estimate, exact), "err_est": err_est},
            )
    estimate = row[-1]
    msg = f"reached max_levels={max_levels} (|R(k,k) − R(k−1,k−1)| = {err_est:.3g}"
    if passed and not confirmed:
        msg += "; the trapezoid column is not in its asymptotic regime"
    msg += ")"
    if max_levels < 3:
        msg += "; the stopping test needs rows k − 1 ≥ 2 and k, use max_levels ≥ 3"
    return Result(
        "romberg",
        estimate,
        estimate,
        False,
        msg,
        max_levels,
        fc.n,
        trace=trace,
        extra={"exact": exact, "error": _error(estimate, exact), "err_est": err_est},
    )


# --------------------------------------------------------------------------------------
# Gauss–Legendre quadrature
# --------------------------------------------------------------------------------------


def gauss_legendre_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes tᵢ and weights wᵢ of the n-point Gauss–Legendre rule on [−1, 1].

    Golub–Welsch (1969): the monic Legendre polynomials satisfy the three-term recurrence
    p_{k+1}(t) = t·p_k(t) − β_k²·p_{k−1}(t) with β_k = k/√(4k² − 1). The nodes are the
    eigenvalues of the symmetric tridiagonal Jacobi matrix J (zero diagonal, off-diagonal
    β₁, …, β_{n−1}); the weights are wᵢ = μ₀·v₁ᵢ² with μ₀ = ∫₋₁¹ dt = 2 and v₁ᵢ the first
    component of the i-th unit eigenvector.

    # NOTE: nodes and weights are symmetrized, tᵢ ← (tᵢ − t_{n+1−i})/2 and
    # wᵢ ← (wᵢ + w_{n+1−i})/2, which imposes the exact symmetry of the rule (and t = 0 for
    # the middle node when n is odd) and removes the O(ε) asymmetry of the eigensolver.
    """
    n = _check_int("n", n, 1)
    k = np.arange(1, n, dtype=np.float64)
    beta = k / np.sqrt(4.0 * k * k - 1.0)  # (n−1,)
    jacobi = np.diag(beta, 1) + np.diag(beta, -1)  # (n, n) symmetric tridiagonal
    theta, vecs = np.linalg.eigh(jacobi)  # ascending eigenvalues, orthonormal eigenvectors
    w = 2.0 * vecs[0, :] ** 2
    t = 0.5 * (theta - theta[::-1])
    w = 0.5 * (w + w[::-1])
    return t, w


@register(
    id="gauss_legendre",
    family="integration",
    name="Gauss–Legendre quadrature",
    params=(
        ParamSpec(
            "n",
            10,
            kind="int",
            min=1,
            max=64,
            help="Number of Gauss points of the final rule; step k uses k + 1 points.",
        ),
        ParamSpec(
            "tol",
            1e-10,
            min=1e-15,
            max=1e-1,
            log=True,
            help="Converged when the error estimate from |G_n − G_{n−1}| ≤ tol·max(1, |G_n|).",
        ),
    ),
    needs=("f", "interval"),
    order="exact for degree ≤ 2n − 1; spectral for analytic f",
    summary="Place n nodes at the roots of the Legendre polynomial so degree 2n − 1 is exact.",
    references=(
        "Golub & Welsch, Calculation of Gauss quadrature rules, Math. Comp. 23 (1969)",
        "Burden & Faires, Numerical Analysis (10th ed.), §4.7",
        "Trefethen, Approximation Theory and Approximation Practice (2013), Ch. 19",
    ),
)
def gauss_legendre(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 10,
    tol: float = 1e-10,
) -> Result:
    """Gauss–Legendre quadrature G_m = Σᵢ ((b − a)/2)·wᵢ·f((a + b)/2 + ((b − a)/2)·tᵢ).

    The m-point rule is exact for polynomials of degree ≤ 2m − 1 (Burden & Faires, Thm. 4.7);
    nodes/weights come from :func:`gauss_legendre_rule`. The trace shows the sequence
    m = 1, …, n (Step k uses m = k + 1 points). Gauss rules are not nested, so the whole
    sequence costs n(n + 1)/2 evaluations (shared nodes, such as t = 0 for odd m, are cached).

    Stopping: the sequence always runs to m = n; converged when n ≥ 3 and the error estimate
    of the module docstring, built from d_m = |G_m − G_{m−1}|, is ≤ tol·max(1, |G_n|).
    d_n alone is pessimistic when G_m converges geometrically (analytic f) but too small for
    the algebraic convergence of a non-smooth f: on √x the error is O(n⁻³) and
    d_n ≈ 3·error/n, which the geometric-tail term d_n/(ρ − 1) corrects.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    n = _check_int("n", n, 1)
    tol = _check_tol(tol)
    exact = _exact(prob)
    fc = _CachedF(prob.f)
    half, mid = 0.5 * (b - a), 0.5 * (a + b)

    trace: list[Step] = []
    estimate = math.nan
    err_est: float | None = None
    ratio: float | None = None
    diffs: list[float] = []  # d_j = |G_{j+1} − G_j|
    noises: list[float] = []  # rounding level of each d_j
    mass_prev = math.nan
    for k in range(n):
        t, w = gauss_legendre_rule(k + 1)
        nodes = [mid + half * float(ti) for ti in t]
        weights = [half * float(wi) for wi in w]
        values = [fc(x) for x in nodes]
        prev = estimate
        estimate = _fsum(wi * v for wi, v in zip(weights, values, strict=True))
        mass = _fsum(abs(wi * v) for wi, v in zip(weights, values, strict=True))
        finite = math.isfinite(estimate) and math.isfinite(mass)
        if k > 0 and finite:
            diffs.append(abs(estimate - prev))
            noises.append(4.0 * _EPS * (mass + mass_prev))
            err_est, ratio = _sequence_error(diffs, noises, None)
        mass_prev = mass
        points = [[x, v] for x, v in zip(nodes, values, strict=True)]
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                info={
                    "estimate": estimate,
                    "error": _error(estimate, exact),
                    "err_est": err_est,
                    "ratio": ratio,
                    "n_points": k + 1,
                    "nodes": points,
                    "weights": weights,
                },
            )
        )
        if not finite or not all(math.isfinite(v) for v in values):
            return _nonfinite_result(
                "gauss_legendre", estimate, _first_bad(points), k, fc, trace, exact=exact
            )
    extra = {"exact": exact, "error": _error(estimate, exact), "err_est": err_est}
    if n <= 2:
        msg = (
            "n = 1: a single rule has no error estimate"
            if n == 1
            else "n = 2: one difference cannot confirm the convergence; use n ≥ 3"
        )
        return Result(
            "gauss_legendre", estimate, estimate, False, msg, n - 1, fc.n, trace=trace, extra=extra
        )
    converged = _passes(err_est, estimate, tol)
    rel = "≤" if converged else ">"
    msg = f"error estimate {err_est:.3g} {rel} tol·max(1, |I|)"
    if not converged:
        msg += "; increase n"
    return Result(
        "gauss_legendre", estimate, estimate, converged, msg, n - 1, fc.n, trace=trace, extra=extra
    )


# --------------------------------------------------------------------------------------
# Adaptive Simpson
# --------------------------------------------------------------------------------------


@register(
    id="adaptive_simpson",
    family="integration",
    name="Adaptive Simpson",
    params=(
        ParamSpec(
            "tol",
            1e-8,
            min=1e-14,
            max=1e-1,
            log=True,
            help="Absolute error target; each half of an interval gets half the tolerance.",
        ),
        ParamSpec(
            "min_depth",
            2,
            kind="int",
            min=0,
            max=10,
            help="Bisect at least this deep before an interval may be accepted (≤ max_depth).",
        ),
        ParamSpec(
            "max_depth",
            30,
            kind="int",
            min=1,
            max=50,
            help="Maximum bisection depth; deeper intervals are accepted and flagged.",
        ),
        ParamSpec(
            "max_iter",
            1000,
            kind="int",
            min=1,
            max=10_000,
            help="Maximum number of accepted intervals (trace steps).",
        ),
    ),
    needs=("f", "interval"),
    order="O(h⁴) locally; work concentrates where f is rough",
    summary="Bisect only the intervals where two Simpson estimates disagree.",
    references=(
        "Lyness, Notes on the adaptive Simpson quadrature routine, J. ACM 16 (1969)",
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 4.3",
    ),
)
def adaptive_simpson(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    tol: float = 1e-8,
    min_depth: int = 2,
    max_depth: int = 30,
    max_iter: int = 1000,
) -> Result:
    """Adaptive Simpson quadrature with the Lyness test, confirmed on two levels.

    For an interval [α, β] with midpoint μ, S₁ = Simpson on [α, β] and S₂ = Simpson on
    [α, μ] plus Simpson on [μ, β]. If f⁽⁴⁾ is nearly constant on [α, β] then
    I − S₂ ≈ (S₂ − S₁)/15. The interval *passes* when |S₂ − S₁| ≤ 15·τ, where τ is its
    tolerance (τ = tol for [a, b], halved on each bisection) (Lyness, 1969). An interval is
    *accepted* when it passes, its parent passed too (the root has no parent and needs only
    its own test), and its depth is ≥ ``min_depth``; it then contributes the
    Richardson-corrected value S₂ + (S₂ − S₁)/15. Otherwise both halves are pushed on a
    stack; the left half is processed first, so accepted intervals appear in left-to-right
    order. Each processed interval costs 2 new evaluations (3 for the start).

    # NOTE: Lyness' routine accepts an interval on its own test. The estimate (S₂ − S₁)/15
    # assumes that f⁽⁴⁾ is nearly constant on the interval, which fails on coarse intervals.
    # For 1/(1 + 100x²) and tol = 1e-3, [0, 0.5] passes (|S₂ − S₁| = 1.6e-3 ≤ 15·2.5e-4),
    # although its error is 6.6e-3 and its parent [0, 1] failed (|S₂ − S₁| = 5.2e-2); the
    # result was wrong by 0.013 = 13·tol. The difference |S₂ − S₁| of the parent is S₁ of
    # the two halves against S₁ of the parent, so "the parent passed" is the Lyness test
    # one level coarser: an accepted interval has passed on two consecutive levels, as the
    # composite rules and Romberg must confirm their rate on several levels. This costs
    # about 2× the evaluations (one more level). ``min_depth`` (default 2: 4 initial
    # intervals, as MATLAB's quadgk starts from 10; Shampine, J. Comput. Appl. Math. 211
    # (2008)) protects the root, which has no parent. In the sweep of the module docstring
    # the one-level rule (min_depth = 2) converged 1 215 times, 9 times with an error above
    # 10·tol·max(1, |I|); this rule converged 1 213 times, with none. Both
    # tests use only values of f on the grid, so a feature that no node of the min_depth
    # grid sees (a peak narrower than (b − a)/2^(min_depth + 2), or a period that aliases
    # with the grid, e.g. cos(32πx) on [0, 1]) is not detected (module docstring,
    # "Preconditions").

    ``Step.fun`` at step k is the sum of the accepted contributions plus the Simpson values
    S₁ of the intervals still pending, i.e. the current estimate of the whole integral.

    Result.extra["intervals"] lists the accepted intervals in order; Step k carries only the
    interval it accepted (``interval``) and the stack (≤ max_depth + 1 entries), so the trace
    grows linearly in the number of steps.

    Stopping: converged when the stack empties and every interval was accepted by the test
    above. ``max_depth`` (an interval at that depth is accepted without the test,
    ``forced``), an interval too small to bisect in floating point, ``max_iter`` accepted
    intervals, a non-finite f, or an overflow of S₁, S₂ or the estimate give
    ``converged=False``.

    # NOTE: min_depth > max_depth is clamped to max_depth (both are UI sliders, and every
    # slider position must be valid); Result.extra["min_depth"] is the value used.
    # NOTE: Burden & Faires (Alg. 4.3) accept with the factor 10 instead of 15 as a safety
    # margin; this implements Lyness' factor 15 as the task specifies.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    tol = _check_tol(tol)
    min_depth = _check_int("min_depth", min_depth, 0)
    max_depth = _check_int("max_depth", max_depth, 1)
    min_depth = min(min_depth, max_depth)
    max_iter = _check_int("max_iter", max_iter, 1)
    exact = _exact(prob)
    fc = _CachedF(prob.f)

    def failed(msg: str, estimate: float, extra: dict[str, Any]) -> Result:
        return Result(
            "adaptive_simpson",
            estimate,
            estimate,
            False,
            msg,
            len(trace) - 1,
            fc.n,
            trace=trace,
            extra=extra,
        )

    m = a + 0.5 * (b - a)
    fa, fm, fb = fc(a), fc(m), fc(b)
    s_whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
    # stack entries: (alpha, beta, f_alpha, f_mid, f_beta, S1, tau, depth, parent_passed)
    stack: list[tuple[float, float, float, float, float, float, float, int, bool]] = [
        (a, b, fa, fm, fb, s_whole, tol, 0, True)
    ]
    trace = [
        Step(
            0,
            s_whole,
            s_whole,
            info={
                "estimate": s_whole,
                "error": _error(s_whole, exact),
                "err_est": None,
                "interval": None,
                "pending": [[a, b]],
                "depth": 0,
                "tol_local": tol,
                "passed": None,
                "parent_passed": None,
                "forced": False,
                "nodes": [[a, fa], [m, fm], [b, fb]],
            },
        )
    ]
    if not all(math.isfinite(v) for v in (fa, fm, fb)):
        bad = _first_bad([[a, fa], [m, fm], [b, fb]])
        return _nonfinite_result("adaptive_simpson", s_whole, bad, 0, fc, trace, exact=exact)
    if not math.isfinite(s_whole):
        return _nonfinite_result("adaptive_simpson", s_whole, None, 0, fc, trace, exact=exact)

    accepted: list[list[float]] = []
    contributions: list[float] = []
    err_total = 0.0
    n_forced = 0
    estimate = s_whole
    while stack:
        alpha, beta, f_al, f_mu, f_be, s1, tau, depth, parent_passed = stack.pop()
        mu = alpha + 0.5 * (beta - alpha)
        # The quarter points are computed exactly as each child computes its midpoint, so the
        # f value passed down belongs to the child's own midpoint bit for bit.
        x_l, x_r = alpha + 0.5 * (mu - alpha), mu + 0.5 * (beta - mu)
        f_l, f_r = fc(x_l), fc(x_r)
        if not (math.isfinite(f_l) and math.isfinite(f_r)):
            bad = x_l if not math.isfinite(f_l) else x_r
            return _nonfinite_result(
                "adaptive_simpson", estimate, bad, len(trace) - 1, fc, trace, exact=exact
            )
        s_left = (mu - alpha) / 6.0 * (f_al + 4.0 * f_l + f_mu)
        s_right = (beta - mu) / 6.0 * (f_mu + 4.0 * f_r + f_be)
        s2 = s_left + s_right
        diff = s2 - s1
        if not (math.isfinite(s_left) and math.isfinite(s_right) and math.isfinite(diff)):
            return _nonfinite_result(
                "adaptive_simpson", estimate, None, len(trace) - 1, fc, trace, exact=exact
            )
        unsplittable = not (alpha < x_l < mu < x_r < beta)
        passed = abs(diff) <= 15.0 * tau
        ok = passed and parent_passed
        if (ok and depth >= min_depth) or depth >= max_depth or unsplittable:
            forced = not ok
            n_forced += forced
            contributions.append(s2 + diff / 15.0)
            accepted.append([alpha, beta])
            err_total += abs(diff) / 15.0
            estimate = _fsum(contributions) + _fsum(s[5] for s in stack)
            trace.append(
                Step(
                    len(trace),
                    estimate,
                    estimate,
                    step_size=beta - alpha,
                    info={
                        "estimate": estimate,
                        "error": _error(estimate, exact),
                        "err_est": abs(diff) / 15.0,
                        "interval": [alpha, beta],
                        "pending": [[s[0], s[1]] for s in reversed(stack)],
                        "depth": depth,
                        "tol_local": tau,
                        "passed": passed,
                        "parent_passed": parent_passed,
                        "forced": forced,
                        "nodes": [[alpha, f_al], [x_l, f_l], [mu, f_mu], [x_r, f_r], [beta, f_be]],
                    },
                )
            )
            extra = {
                "exact": exact,
                "error": _error(estimate, exact),
                "err_est": err_total,
                "min_depth": min_depth,
                "intervals": accepted,
            }
            if not math.isfinite(estimate):
                return failed("the estimate overflowed", estimate, extra)
            if len(accepted) >= max_iter and stack:
                return failed(
                    f"reached max_iter={max_iter} accepted intervals with {len(stack)} pending",
                    estimate,
                    extra,
                )
        else:
            stack.append((mu, beta, f_mu, f_r, f_be, s_right, 0.5 * tau, depth + 1, passed))
            stack.append((alpha, mu, f_al, f_l, f_mu, s_left, 0.5 * tau, depth + 1, passed))

    extra = {
        "exact": exact,
        "error": _error(estimate, exact),
        "err_est": err_total,
        "min_depth": min_depth,
        "n_intervals": len(accepted),
        "n_forced": n_forced,
        "intervals": accepted,
    }
    if n_forced:
        return failed(
            f"{n_forced} interval(s) accepted without passing the Lyness test on two levels "
            f"(max_depth={max_depth} or too small to bisect)",
            estimate,
            extra,
        )
    msg = (
        f"all {len(accepted)} intervals and their parents passed |S₂ − S₁| ≤ 15τ "
        f"(estimated error {err_total:.3g})"
    )
    return Result(
        "adaptive_simpson",
        estimate,
        estimate,
        True,
        msg,
        len(trace) - 1,
        fc.n,
        trace=trace,
        extra=extra,
    )


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------


@register(
    id="monte_carlo_integration",
    family="integration",
    name="Monte Carlo integration",
    params=(
        # NOTE: joint-consistent with the MAX_SAMPLES check: 128·2^13 = 2^20, so no slider
        # position raises; more samples come from levels.
        ParamSpec("n", 100, kind="int", min=1, max=128, help="Samples at step 0."),
        ParamSpec(
            "levels",
            6,
            kind="int",
            min=0,
            max=13,
            help="Number of doublings: step k uses n·2ᵏ samples.",
        ),
        ParamSpec(
            "tol",
            1e-2,
            min=1e-6,
            max=1.0,
            log=True,
            help="Converged when the standard error ≤ tol·max(1, |I|).",
        ),
    ),
    needs=("f", "interval"),
    order="O(N^{-1/2}) in probability",
    summary="Average f at uniformly random points and multiply by the interval length.",
    references=(
        "Owen, Monte Carlo theory, methods and examples (2013), Ch. 2",
        "Welford, Technometrics 4 (1962) (running variance)",
    ),
    deterministic=False,
)
def monte_carlo_integration(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    seed: int = 0,
    n: int = 100,
    levels: int = 6,
    tol: float = 1e-2,
) -> Result:
    """Plain Monte Carlo: I_N = (b − a)·(1/N)·Σᵢ f(Uᵢ), Uᵢ ~ Uniform(a, b) i.i.d.

    E[I_N] = I and sd(I_N) = (b − a)·σ_f/√N. The standard error is estimated by
    (b − a)·s_N/√N with the sample standard deviation s_N (divisor N − 1), accumulated by
    Welford's stable one-pass update. Step k uses N_k = n·2^k samples: the first N_k draws
    of ``Rng(seed)``, drawn in order as Uᵢ = a + (b − a)·u with u = ``rng.random()``.

    Stopping: the sweep always runs all levels; converged when N ≥ 2 and the standard error
    ≤ tol·max(1, |I_N|) (a 1-σ statistical statement, not a guaranteed bound).
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    n = _check_int("n", n, 1)
    levels = _check_int("levels", levels, 0)
    tol = _check_tol(tol)
    if n * 2**levels > MAX_SAMPLES:
        raise ValueError(f"n·2^levels = {n * 2**levels} exceeds the limit {MAX_SAMPLES}")
    exact = _exact(prob)
    rng = Rng(seed)
    fc = Counted(prob.f)
    width = b - a

    count, mean, m2 = 0, 0.0, 0.0
    trace: list[Step] = []
    estimate = math.nan
    std_err: float | None = None
    x_bad: float | None = None  # the first sample with a non-finite f value
    for k in range(levels + 1):
        target = n * 2**k
        samples: list[list[float]] = []
        while count < target:
            x = rng.uniform(a, b)
            fx = _safe_eval(fc, x)
            if len(samples) < MAX_DISPLAY:
                samples.append([x, fx])
            if x_bad is None and not math.isfinite(fx):
                x_bad = x
            count += 1
            delta = fx - mean
            mean += delta / count
            m2 += delta * (fx - mean)
        estimate = width * mean
        std_err = width * math.sqrt(m2 / (count - 1) / count) if count > 1 else None
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                info={
                    "estimate": estimate,
                    "error": _error(estimate, exact),
                    "err_est": std_err,
                    "samples": samples,
                    "n_samples": count,
                },
            )
        )
        if not math.isfinite(estimate):
            return _nonfinite_result(
                "monte_carlo_integration", estimate, x_bad, k, fc, trace, exact=exact
            )
    extra = {
        "exact": exact,
        "error": _error(estimate, exact),
        "err_est": std_err,
        "n_samples": count,
    }
    if std_err is None:
        msg = "one sample: no standard error"
        converged = False
    else:
        converged = _passes(std_err, estimate, tol)
        rel = "≤" if converged else ">"
        msg = f"standard error {std_err:.3g} {rel} tol·max(1, |I|) with N = {count}"
    return Result(
        "monte_carlo_integration",
        estimate,
        estimate,
        converged,
        msg,
        levels,
        fc.n,
        trace=trace,
        extra=extra,
    )


# --------------------------------------------------------------------------------------
# Nested rules with a free error estimate: Clenshaw–Curtis and Gauss–Kronrod–Patterson
# --------------------------------------------------------------------------------------
#
# Both methods apply a sequence of rules whose node sets are nested (every node of rule k is
# a node of rule k + 1), so a run that stops at level K costs only the nodes of rule K, and
# d_k = |I_k − I_{k−1}| is a free error estimate. Both use the stopping test of the study
# research/clenshaw-curtis-vs-gauss: converged at the first k ≥ 2 with
# d_k ≤ tol·max(1, |I_k|).
#
# NOTE: the test starts at k = 2 (three rules) because two rules can agree by accident at
# small n (as for Romberg). d_k estimates the error of the *coarser* rule I_{k−1}: for
# geometric convergence (analytic f) the error of I_k is far smaller, so the test is
# conservative and usually costs one doubling more than needed. For algebraic convergence
# it can underestimate when the sequence oscillates. This is the study's verified test,
# promoted unchanged; it has no confirmation of an asymptotic regime (composite rules) and
# no second pass (Romberg), so it is weaker than those tests. Measured on the library
# (n = 1, …, 64 for Clenshaw–Curtis, 131 tolerances in [1e-14, 1e-1]):
#
# * Clenshaw–Curtis converged 5 620 times on |x − 0.3| (abs_kink), 52 times with an error
#   above 10·tol·max(1, |I|), worst 234× (n = 60, tol = 2e-10: error 4.7e-8 with 1921
#   points). On the other eight library problems: 0 of 66 920.
# * Gauss–Patterson: 0 violations on the library (978 converged runs).
# * Both can stop on a feature that their first three rules do not resolve
#   ("Preconditions", item 1). On exp(−c(x − x_c)²) on [0, 1], c ∈ [10, 1000],
#   Clenshaw–Curtis misses the peak when the 4n + 1 points of step 2 are too coarse:
#   n = 2, c = 1000, x_c = 0.40625, tol = 1e-3 converged at 9 points with an error of
#   0.056. When the spacing π/(8n) of that grid at the center is ≤ the peak's σ = 1/√(2c),
#   no violation occurred (0 of 405 000 converged runs; on 1/(1 + c·x²), n ≥ 2: 0 of
#   41 850). Gauss–Patterson converged 4 of 5 016 times with 13.5–13.8× tol, e.g.
#   c = 1000, x_c = 1/3, tol = 10^−2.5: I_7 = 0.01574 and I_15 = 0.01320 agree within tol,
#   but I = 0.0560.
#
# Requiring the test at two consecutive levels (k − 1 and k, k ≥ 3) removed every abs_kink
# violation in the same sweep (worst 0.3×) at the cost of one more doubling; it is not
# used here because the study's test is the specification.

#: Upper bound on the degree n·2^max_levels of a Clenshaw–Curtis rule (= MAX_POINTS).
_CC_MAX_DEGREE = MAX_POINTS
#: The largest ``max_levels`` of Clenshaw–Curtis; with n ≤ _CC_N_MAX every slider position
#: has n·2^max_levels ≤ _CC_MAX_DEGREE.
_CC_LEVELS_MAX = 12
_CC_N_MAX = _CC_MAX_DEGREE >> _CC_LEVELS_MAX


def clenshaw_curtis_nodes(n: int) -> np.ndarray:
    """Chebyshev extreme points t_k = cos(kπ/n), k = 0, …, n (decreasing), on [−1, 1].

    Computed as sin(π(n − 2k)/(2n)), which equals cos(kπ/n) mathematically. This form is
    exactly antisymmetric (t_{n−k} = −t_k, t_{n/2} = 0 for even n), t_0 = 1, t_n = −1, and
    t_k^{(n)} == t_{2k}^{(2n)} bit for bit: the argument of the doubled grid is
    fl(fl(π·2m)/(4n)) = fl(fl(π·m)/(2n)) because the scalings by 2 are exact. The evaluation
    cache is therefore exact on nested grids.
    """
    n = _check_int("n", n, 1)
    m = n - 2 * np.arange(n + 1, dtype=np.float64)  # (n+1,) integers n, n−2, …, −n, exact
    return np.sin(np.pi * m / (2.0 * n))


def clenshaw_curtis_weights(n: int) -> np.ndarray:
    """Clenshaw–Curtis weights w_0, …, w_n on [−1, 1] for the nodes cos(kπ/n).

    The weights integrate the degree-≤ n interpolant at the Chebyshev extreme points exactly.
    The explicit form (Waldvogel 2006, (2.4)–(2.5)) is
        w_k = (c_k/n)·[1 − Σ_{j=1}^{⌊n/2⌋} b_j/(4j² − 1)·cos(2jkπ/n)],
        b_j = 1 if j = n/2 else 2,  c_k = 1 if k ≡ 0 (mod n) else 2,
    which costs O(n²). This function uses Waldvogel's Theorem (§5): w = F_n⁻¹(v + g), one
    inverse DFT of order n, w_n := w_0, with
        v_k = 2/(1 − 4k²) for 0 ≤ k < ⌊n/2⌋,  v_{⌊n/2⌋} = (n − 3)/(2⌊n/2⌋ − 1) − 1,
        v_{n−k} = v_k                                                      (3.10)
        g_k = −w_0 for 0 ≤ k < ⌊n/2⌋,  g_{⌊n/2⌋} = w_0·[(2 − n mod 2)·n − 1],
        g_{n−k} = g_k                                                      (4.2)
        w_0 = 1/(n² − 1 + n mod 2)                                         (2.6)
    built as in Waldvogel's MATLAB ``fejer.m``; O(n log n). n = 1 is the trapezoid rule (1, 1).

    # NOTE: the weights are symmetrized, w_k ← (w_k + w_{n−k})/2. This imposes the exact
    # symmetry of the rule and removes the O(ε) asymmetry of the FFT (as gauss_legendre_rule).
    """
    n = _check_int("n", n, 1)
    if n == 1:
        return np.array([1.0, 1.0])
    odd = np.arange(1, n, 2, dtype=np.float64)  # (l,) = 1, 3, …, ≤ n − 1
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


def clenshaw_curtis_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes t_k (decreasing) and weights w_k of the (n + 1)-point Clenshaw–Curtis rule on
    [−1, 1]. Exact for polynomials of degree ≤ n (≤ n + 1 for even n, by symmetry)."""
    return clenshaw_curtis_nodes(n), clenshaw_curtis_weights(n)


def chebyshev_coefficients(values: Iterable[float]) -> np.ndarray:
    """Coefficients a_0, …, a_n of the interpolant p = Σ a_j T_j through (cos(kπ/n), values[k]).

    DCT-I by an FFT of the even extension (Trefethen 2008, §2, ``clenshaw_curtis.m``):
    g = Re FFT([f_0, …, f_n, f_{n−1}, …, f_1])/(2n), a_0 = g_0, a_j = 2g_j (0 < j < n),
    a_n = g_n. Then ∫₋₁¹ p = Σ_{j even} a_j·2/(1 − j²), which equals the Clenshaw–Curtis sum.
    """
    f = np.asarray(list(values), dtype=np.float64)  # (n+1,)
    n = f.size - 1
    if n < 1:
        raise ValueError("chebyshev_coefficients needs at least two values")
    ext = np.concatenate([f, f[-2:0:-1]])  # (2n,) even extension in θ
    g = np.fft.fft(ext).real / (2.0 * n)  # (2n,)
    a = 2.0 * g[: n + 1]
    a[0] = g[0]
    a[n] = g[n]
    return a


# Gauss–Kronrod–Patterson rules on [−1, 1]: levels 0, …, 6 with N_k = 2^{k+1} − 1 = 1, 3, 7,
# 15, 31, 63, 127 points. Level 0 is the midpoint rule, level 1 is Gauss G₃, level 2 is the
# Kronrod extension K₇ of G₃, and each later level is the Patterson (1968) extension of the
# previous one: the N_{k−1} + 1 new nodes are chosen so that the interpolatory rule on all
# N_k nodes has the highest degree, 3·2^k − 1 for k ≥ 1.
#
# NOTE: the extension is ill-conditioned (the node polynomial spans 7 orders of magnitude
# on [−1, 1] at 63 points), so the rules cannot be computed reliably in double precision.
# The values below were computed in IEEE binary128 by the study
# research/clenshaw-curtis-vs-gauss (baselines.py: Patterson's linear conditions in the
# Legendre basis, roots by bisection, interpolatory weights by an exact Gauss rule; three
# independent formulations agree to ≤ 3e-28 up to 63 points), and are stored to 17
# significant digits (the nearest double). The sequence stops at 127 points because the
# 255-point extension needs more than binary128.
# NOTE: binary128 is not enough at 127 points (the study's formulations agree there only to
# ~2e-17), and 9 of its level-6 values (the 7 outermost weights, up to 18989 ulps, and the
# nodes t_0, t_2, 1–2 ulps) were not the nearest double. Those entries were replaced by the
# nearest doubles of a 300-digit recomputation (monomial orthogonality system, Newton-refined
# roots, Legendre moment system for the weights; exact to degree 191 with residual 5e-233).
# The test suite checks the table independently: exactness degree (and not one more) in
# 50-digit arithmetic, nesting, symmetry, positivity, and a 300-digit mpmath recomputation of
# every level (all 127 points) that requires each stored value to be the nearest double.
#
# _GKP_NODES holds the 63 positive nodes of the 127-point rule, decreasing. The nodes are
# nested dyadically: with t_0 > t_1 > … > t_126 the full 127-point set, level k uses the
# t_i with (i + 1) divisible by 2^{6−k}. _GKP_WEIGHTS[k] holds the weights of the 2^k
# non-negative nodes of level k, in decreasing order of the node (the last is at t = 0).
_GKP_MAX_LEVEL = 6
_GKP_NODES: tuple[float, ...] = (
    0.9999824303548916,
    0.99987288812035757,
    0.9995987996719107,
    0.99909812496766759,
    0.99831663531840742,
    0.99720625937222196,
    0.99572410469840722,
    0.99383196321275502,
    0.99149572117810614,
    0.98868475754742946,
    0.98537149959852033,
    0.9815311495537401,
    0.97714151463970567,
    0.97218287474858178,
    0.96663785155841653,
    0.96049126870802026,
    0.95373000642576111,
    0.94634285837340293,
    0.93832039777959286,
    0.92965485742974008,
    0.92034002547001237,
    0.91037115695700432,
    0.89974489977694005,
    0.88845923287225703,
    0.87651341448470532,
    0.86390793819369049,
    0.85064449476835025,
    0.83672593816886875,
    0.82215625436498041,
    0.80694053195021764,
    0.79108493379984834,
    0.7745966692414834,
    0.75748396638051363,
    0.73975604435269471,
    0.72142308537009892,
    0.70249620649152711,
    0.68298743109107918,
    0.66290966002478058,
    0.64227664250975947,
    0.62110294673722644,
    0.59940393024224292,
    0.57719571005204584,
    0.55449513263193251,
    0.53131974364437562,
    0.50768775753371664,
    0.48361802694584105,
    0.45913001198983233,
    0.43424374934680254,
    0.40897982122988868,
    0.38335932419873037,
    0.35740383783153218,
    0.33113539325797681,
    0.30457644155671404,
    0.2777498220218243,
    0.2506787303034832,
    0.22338668642896689,
    0.19589750271110015,
    0.16823525155220748,
    0.14042423315256017,
    0.11248894313318662,
    0.08445404008371088,
    0.056344313046592792,
    0.028184648949745695,
)
_GKP_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (2.0,),
    (
        0.55555555555555558,
        0.88888888888888884,
    ),
    (
        0.10465622602646726,
        0.26848808986833345,
        0.40139741477596225,
        0.45091653865847414,
    ),
    (
        0.017001719629940262,
        0.051603282997079739,
        0.092927195315124542,
        0.13441525524378423,
        0.17151190913639139,
        0.20062852937698902,
        0.2191568584015875,
        0.2255104997982067,
    ),
    (
        0.0025447807915618746,
        0.0084345657393211058,
        0.016446049854387811,
        0.025807598096176654,
        0.035957103307129319,
        0.046462893261757988,
        0.056979509494123358,
        0.067207754295990699,
        0.076879620499003529,
        0.085755920049990345,
        0.093627109981264472,
        0.10031427861179558,
        0.10566989358023481,
        0.10957842105592464,
        0.11195687302095346,
        0.11275525672076869,
    ),
    (
        0.00036322148184553065,
        0.001265156556230068,
        0.0025790497946856883,
        0.0042176304415588546,
        0.0061155068221172464,
        0.0082230079572359303,
        0.010498246909621322,
        0.012903800100351265,
        0.015406750466559498,
        0.017978551568128269,
        0.02059423391591271,
        0.02323144663991027,
        0.025869679327214748,
        0.02848975474583355,
        0.031073551111687966,
        0.033603877148207728,
        0.036064432780782571,
        0.03843981024945553,
        0.040715510116944319,
        0.042877960025007732,
        0.044914531653632198,
        0.04681355499062801,
        0.048564330406673198,
        0.050157139305899538,
        0.051583253952048456,
        0.052834946790116522,
        0.053905499335266061,
        0.054789210527962866,
        0.055481404356559363,
        0.05597843651047632,
        0.056277699831254302,
        0.056377628360384714,
    ),
    (
        5.053609520786252e-05,
        0.00018073956444538837,
        0.00037774664632698465,
        0.0006326073193626335,
        0.0009383698485423815,
        0.0012895240826104174,
        0.00168114286542147,
        0.002108815245726633,
        0.0025687649437940202,
        0.003057753410175531,
        0.0035728927835172995,
        0.0041115039786546927,
        0.0046710503721143215,
        0.0052491234548088595,
        0.0058434498758356398,
        0.0064519000501757368,
        0.0070724899954335554,
        0.007703375233279742,
        0.008342838753968157,
        0.0089892757840641362,
        0.009641177729702537,
        0.010297116957956355,
        0.010955733387837901,
        0.011615723319955135,
        0.01227583056008277,
        0.012934839663607374,
        0.013591571009765546,
        0.014244877372916775,
        0.014893641664815181,
        0.015536775555843983,
        0.016173218729577721,
        0.016801938574103864,
        0.017421930159464173,
        0.018032216390391285,
        0.01863184825613879,
        0.019219905124727765,
        0.019795495048097498,
        0.020357755058472159,
        0.020905851445812022,
        0.021438980012503866,
        0.021956366305317825,
        0.022457265826816099,
        0.02294096422938775,
        0.023406777495314005,
        0.02385405210603854,
        0.024282165203336599,
        0.024690524744487678,
        0.025078569652949769,
        0.025445769965464767,
        0.025791626976024228,
        0.026115673376706099,
        0.026417473395058261,
        0.026696622927450359,
        0.02695274966763303,
        0.027185513229624793,
        0.027394605263981433,
        0.027579749566481872,
        0.027740702178279682,
        0.027877251476613702,
        0.02798921825523816,
        0.028076455793817248,
        0.028138849915627151,
        0.028176319033016602,
        0.028188814180192357,
    ),
)


def gauss_patterson_rule(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes t_i (decreasing) and weights w_i of the Gauss–Kronrod–Patterson rule of ``level``
    (0, …, 6; N = 2^{level+1} − 1 points) on [−1, 1]. Exact for polynomials of degree
    ≤ 3·2^level − 1 (level ≥ 1; degree 1 at level 0)."""
    level = _check_int("level", level, 0)
    if level > _GKP_MAX_LEVEL:
        raise ValueError(f"level must be ≤ {_GKP_MAX_LEVEL} (127 points), got {level}")
    stride = 2 ** (_GKP_MAX_LEVEL - level)
    pos = np.array(_GKP_NODES[stride - 1 :: stride], dtype=np.float64)  # (2^level − 1,)
    half = np.array(_GKP_WEIGHTS[level], dtype=np.float64)  # (2^level,), last at t = 0
    t = np.concatenate([pos, [0.0], -pos[::-1]])  # (N,) decreasing, exactly antisymmetric
    w = np.concatenate([half, half[-2::-1]])  # (N,) symmetric
    return t, w


def _nested_quadrature(
    method: str,
    prob: Problem,
    a: float,
    b: float,
    max_levels: int,
    tol: float,
    rule: Callable[[int], tuple[np.ndarray, np.ndarray]],
    closed: bool,
    level_info: Callable[[int, int, list[float]], dict[str, Any]],
) -> Result:
    """The shared driver of the nested rules: Step k applies rule(k) mapped to [a, b].

    I_k = Σᵢ ((b − a)/2)·wᵢ·f((a + b)/2 + ((b − a)/2)·tᵢ). ``closed`` rules have t_0 = 1 and
    t_last = −1, which are mapped to b and a exactly ((a + b)/2 ± (b − a)/2 can round outside
    [a, b]). Converged at the first k ≥ 2 with d_k = |I_k − I_{k−1}| ≤ tol·max(1, |I_k|);
    converged=False after ``max_levels`` or on a non-finite value. ``level_info(k, n_points,
    values)`` adds the method-specific Info keys.
    """
    exact = _exact(prob)
    fc = _CachedF(prob.f)
    half, mid = 0.5 * (b - a), 0.5 * (a + b)

    trace: list[Step] = []
    estimate = math.nan
    err_est: float | None = None
    n_points = 0
    for k in range(max_levels + 1):
        t, w = rule(k)  # (N_k,), (N_k,)
        x = mid + half * t  # (N_k,) the same doubles for a shared t, so the cache is exact
        if closed:
            x[0], x[-1] = b, a
        nodes = x.tolist()
        weights = (half * w).tolist()
        n_points = len(nodes)
        n_before = fc.n
        values = [fc(xi) for xi in nodes]
        prev = estimate
        estimate = _fsum(wi * v for wi, v in zip(weights, values, strict=True))
        if k > 0:
            err_est = abs(estimate - prev)
        finite = math.isfinite(estimate) and all(math.isfinite(v) for v in values)
        shown = n_points <= MAX_DISPLAY
        points = [[xi, v] for xi, v in zip(nodes, values, strict=True)]
        trace.append(
            Step(
                k,
                estimate,
                estimate,
                info={
                    "estimate": estimate,
                    "error": _error(estimate, exact),
                    "err_est": err_est,
                    "n_points": n_points,
                    "nodes": points if shown else None,
                    "weights": weights if shown else None,
                    "new_nodes": fc.n - n_before,
                    **level_info(k, n_points, values if finite and shown else []),
                },
            )
        )
        if not finite:
            return _nonfinite_result(
                method,
                estimate,
                _first_bad(points),
                k,
                fc,
                trace,
                exact=exact,
                error=_error(estimate, exact),
                err_est=err_est,
                n_points=n_points,
            )
        if k >= 2 and _passes(err_est, estimate, tol):
            return Result(
                method,
                estimate,
                estimate,
                True,
                f"|I_k − I_(k−1)| = {err_est:.3g} ≤ tol·max(1, |I|) with {n_points} points",
                k,
                fc.n,
                trace=trace,
                extra={
                    "exact": exact,
                    "error": _error(estimate, exact),
                    "err_est": err_est,
                    "n_points": n_points,
                },
            )
    return Result(
        method,
        estimate,
        estimate,
        False,
        f"reached max_levels={max_levels} ({n_points} points) with "
        f"|I_k − I_(k−1)| = {err_est:.3g} > tol·max(1, |I|)",
        max_levels,
        fc.n,
        trace=trace,
        extra={
            "exact": exact,
            "error": _error(estimate, exact),
            "err_est": err_est,
            "n_points": n_points,
        },
    )


_NESTED_TOL_HELP = "Converged at the first k ≥ 2 with |I_k − I_(k−1)| ≤ tol·max(1, |I_k|)."


@register(
    id="clenshaw_curtis",
    family="integration",
    name="Clenshaw–Curtis quadrature",
    params=(
        ParamSpec(
            "n",
            2,
            kind="int",
            min=1,
            max=_CC_N_MAX,
            help="Degree of the first rule (n + 1 Chebyshev points); step k uses n·2ᵏ.",
        ),
        ParamSpec(
            "max_levels",
            12,
            kind="int",
            min=2,
            max=_CC_LEVELS_MAX,
            help="Maximum number of doublings L; the last rule has n·2ᴸ + 1 points.",
        ),
        ParamSpec("tol", 1e-10, min=1e-15, max=1e-1, log=True, help=_NESTED_TOL_HELP),
    ),
    needs=("f", "interval"),
    order="exact for degree ≤ n; ≈ Gauss accuracy unless f is analytic in a large ellipse",
    summary=(
        "Integrate the polynomial through f at Chebyshev points; doubling n reuses every "
        "old point and gives a free error estimate."
    ),
    references=(
        "Clenshaw & Curtis, A method for numerical integration on an automatic computer, "
        "Numer. Math. 2 (1960)",
        "Trefethen, Is Gauss quadrature better than Clenshaw–Curtis?, SIAM Rev. 50 (2008), "
        "eqs. (2.2)–(2.3), Thm. 5.2",
        "Waldvogel, Fast construction of the Fejér and Clenshaw–Curtis quadrature rules, "
        "BIT 46 (2006), §5, eqs. (2.6), (3.10), (4.2)",
    ),
)
def clenshaw_curtis(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    n: int = 2,
    max_levels: int = 12,
    tol: float = 1e-10,
) -> Result:
    """Clenshaw–Curtis quadrature on nested Chebyshev grids n, 2n, 4n, … (Trefethen 2008,
    (2.2)–(2.3)).

    Step k applies the (n_k + 1)-point rule I_{n_k} = Σ_j ((b − a)/2)·w_j·f(x_j) with
    n_k = n·2^k, x_j = (a + b)/2 + ((b − a)/2)·cos(jπ/n_k), and the weights of Waldvogel
    (2006), §5 (:func:`clenshaw_curtis_weights`). The rule integrates the degree-n_k
    interpolant of f at the Chebyshev extreme points exactly. It is exact for degree ≤ n_k;
    by Trefethen's Theorem 5.2 the aliased Chebyshev modes T_{n+p} are integrated with an
    error of only 8pn/(n⁴ − 2(p² + 1)n² + (p² − 1)²) (n ± p even, (5.4)), which is why its
    accuracy is close to Gauss with the same number of points unless f is analytic in a
    large Bernstein ellipse. The nested grids reuse every old node, so a run that stops at
    level K costs n_K + 1 evaluations.

    Stopping: converged at the first k ≥ 2 with |I_{n_k} − I_{n_{k−1}}| ≤ tol·max(1, |I_{n_k}|)
    (see the section comment above for its limits); converged=False after ``max_levels``
    doublings without passing, or on a non-finite value (a closed rule evaluates f at a
    and b).

    Known limits (measured in the section comment above): the test can accept an error
    above 10·tol on a non-smooth f (abs_kink, up to 234×) and can miss a peak narrower than
    the spacing of the 4n + 1 points of step 2. An f that vanishes at every node of the first three grids, e.g.
    f = 1 − T_{4n}(t)² on [−1, 1], gives I_{n_k} = 0 for k = 0, 1, 2 and a false
    "converged" result (true of every sampling rule).
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    n = _check_int("n", n, 1)
    max_levels = _check_int("max_levels", max_levels, 2)
    tol = _check_tol(tol)
    if n * 2**max_levels > _CC_MAX_DEGREE:
        raise ValueError(f"n·2^max_levels = {n * 2**max_levels} exceeds the limit {_CC_MAX_DEGREE}")

    def info(k: int, n_points: int, values: list[float]) -> dict[str, Any]:
        coeffs = chebyshev_coefficients(values).tolist() if values else None
        return {"n": n * 2**k, "cheb_coeffs": coeffs}

    return _nested_quadrature(
        "clenshaw_curtis",
        prob,
        a,
        b,
        max_levels,
        tol,
        lambda k: clenshaw_curtis_rule(n * 2**k),
        True,
        info,
    )


@register(
    id="gauss_patterson",
    family="integration",
    name="Gauss–Kronrod–Patterson quadrature",
    params=(
        ParamSpec(
            "max_levels",
            _GKP_MAX_LEVEL,
            kind="int",
            min=2,
            max=_GKP_MAX_LEVEL,
            help="Last level; step k uses 2ᵏ⁺¹ − 1 points (1, 3, 7, …, 127).",
        ),
        ParamSpec("tol", 1e-10, min=1e-15, max=1e-1, log=True, help=_NESTED_TOL_HELP),
    ),
    needs=("f", "interval"),
    order="the N-point rule is exact for degree ≤ (3N + 1)/2 (N ≥ 3)",
    summary=(
        "Gauss's 3 points, then repeatedly add the optimal new points between the old ones: "
        "every old value is reused and each level gives a free error estimate."
    ),
    references=(
        "Patterson, The optimum addition of points to quadrature formulae, Math. Comp. 22 (1968)",
        "Kronrod, Nodes and weights of quadrature formulas (1965)",
        "Gautschi, Orthogonal Polynomials: Computation and Approximation (2004), §3.1.2",
    ),
)
def gauss_patterson(
    problem: Problem | Callable[[float], float],
    *,
    bracket: tuple[float, float] | None = None,
    max_levels: int = _GKP_MAX_LEVEL,
    tol: float = 1e-10,
) -> Result:
    """Non-adaptive nested Gauss–Kronrod–Patterson quadrature (Patterson 1968).

    Step k applies the N_k = 2^{k+1} − 1 point rule (:func:`gauss_patterson_rule`) on
    [a, b]: the midpoint rule, Gauss G₃, Kronrod K₇, then the Patterson extensions with 15,
    31, 63 and 127 points. Given the symmetric node set X of level k − 1 (n points), the
    N_{k−1} + 1 new nodes are the roots of a polynomial q such that the node polynomial
    Π = ∏_{x∈X}(x − x_j)·q satisfies ∫₋₁¹ Π(x)·p(x) dx = 0 for every p of degree ≤ n
    (Gautschi 2004, §3.1.2), and the weights are interpolatory. Level k ≥ 1 is exact for
    degree ≤ 3·2^k − 1, against 2^{k+2} − 3 for Gauss with the same points. Every node is
    reused, so a run that stops at level K costs N_K evaluations: the cost structure of
    nested Clenshaw–Curtis (2^{k+1} + 1 points at the same level with n = 2), with a
    higher-degree rule. It is an open rule (no evaluation at a or b).

    Stopping: the same test as ``clenshaw_curtis``: converged at the first k ≥ 2 with
    |I_k − I_{k−1}| ≤ tol·max(1, |I_k|); converged=False after ``max_levels`` (at most 6,
    127 points: the stored rules end there) or on a non-finite value.

    Known limit (section comment above): two coarse rules can agree by accident before the
    rules resolve f (13.8× tol on a narrow peak at 15 points).

    # NOTE: this is the non-adaptive use of the sequence (as in the study, and as in
    # QUADPACK's QNG, which uses the 10–21–43–87 Patterson sequence on the whole interval).
    """
    prob = scalar_problem(problem)
    a, b = _resolve_interval(prob, bracket)
    max_levels = _check_int("max_levels", max_levels, 2)
    if max_levels > _GKP_MAX_LEVEL:
        raise ValueError(f"max_levels must be ≤ {_GKP_MAX_LEVEL} (127 points), got {max_levels}")
    tol = _check_tol(tol)
    return _nested_quadrature(
        "gauss_patterson",
        prob,
        a,
        b,
        max_levels,
        tol,
        gauss_patterson_rule,
        False,
        lambda k, n_points, values: {},
    )


#: Parity fixtures for the web app: (method_id, problem_id, params). Traces stay < 300 steps.
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("left_riemann", "exp_0_1", {"n": 4, "levels": 5}),
    ("right_riemann", "sin_0_pi", {"n": 4, "levels": 5}),
    ("midpoint_rule", "gaussian", {"n": 4, "levels": 5}),
    ("trapezoid", "runge", {"n": 4, "levels": 6}),
    ("trapezoid", "abs_kink", {"n": 2, "levels": 6}),
    ("simpson", "exp_0_1", {"n": 2, "levels": 5}),
    ("simpson", "sqrt_0_1", {"n": 2, "levels": 6}),
    ("simpson_38", "arctan_deriv", {"n": 3, "levels": 4}),
    ("boole", "gaussian", {"n": 4, "levels": 4}),
    ("romberg", "exp_0_1", {"tol": 1e-12, "max_levels": 10}),
    ("romberg", "sqrt_0_1", {"tol": 1e-10, "max_levels": 8}),
    ("gauss_legendre", "runge", {"n": 12}),
    ("gauss_legendre", "poly3", {"n": 4}),
    ("adaptive_simpson", "sqrt_0_1", {"tol": 1e-6}),
    ("adaptive_simpson", "runge", {"tol": 1e-6}),
    ("monte_carlo_integration", "sin_0_pi", {"seed": 1, "n": 50, "levels": 5}),
    ("clenshaw_curtis", "runge", {"tol": 1e-10}),
    ("clenshaw_curtis", "sqrt_0_1", {"tol": 1e-8}),
    ("clenshaw_curtis", "abs_kink", {"n": 3, "max_levels": 8, "tol": 1e-6}),
    ("gauss_patterson", "gaussian", {"tol": 1e-10}),
    ("gauss_patterson", "exp_0_1", {"tol": 1e-13}),
    ("gauss_patterson", "runge", {"tol": 1e-10}),
]
