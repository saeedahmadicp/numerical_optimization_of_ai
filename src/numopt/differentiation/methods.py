"""Numerical differentiation: estimate f′(x0) (or f″(x0)) from values of f.

Every method sweeps the step sizes h_k = h0/2^k, k = 0, …, ``levels``, and records one Step
per h. The sweep is never stopped early, so the trace always shows the V-shaped error curve:
for large h the *truncation* error C·h^p of the formula dominates; for small h the
*round-off* error of the cancelling differences, about ε·|f|/h^q, dominates
(Burden & Faires, §4.1; Sauer, Numerical Analysis, §5.1.2).

Each formula is D(h) = (1/h^q)·Σᵢ cᵢ·f(x0 + oᵢh) with stencil offsets oᵢ, weights cᵢ, the
derivative order q (1 or 2) and the truncation order p:

==========================  =======================  ============================  ===  ===
method                      offsets oᵢ               weights cᵢ                    q    p
==========================  =======================  ============================  ===  ===
forward_difference          0, 1                     −1, 1                         1    1
backward_difference         −1, 0                    −1, 1                         1    1
central_difference          −1, 1                    −1/2, 1/2                     1    2
five_point_stencil          −2, −1, 1, 2             1/12, −8/12, 8/12, −1/12      1    4
second_derivative_central   −1, 0, 1                 1, −2, 1                      2    2
==========================  =======================  ============================  ===  ===

``richardson_extrapolation`` extrapolates central differences to h = 0, and ``complex_step``
uses Im f(x0 + ih)/h, which has no cancellation at all.

Error estimates at level k (all absolute, computed without the exact derivative):

* ``err_est`` (truncation): the Richardson estimate |D_k − D_{k−1}|/(2^p − 1), valid while the
  error is C·h^p + O(h^{p+1}) (None at k = 0);
* ``roundoff``: the rounding model ε·Σᵢ|cᵢ|·(|f(xᵢ)| + [oᵢ ≠ 0]·|xᵢ|·g)/h^q. The first term
  is a relative error ε in each computed f value; the second is the error caused by rounding
  the abscissa fl(x0 + oᵢh), which moves f by about |f′|·ε|xᵢ| (Gill, Murray & Wright,
  Practical Optimization, §8.6). g estimates |f′(x0)|: g is the median (the mean of the two
  middle values for an even count) of the slopes s_j of the non-collapsed levels j ≤ k with a
  finite s_j, where s_j = |D_j| for the first-derivative formulas (for Richardson: |D(j, 0)|)
  and s_j = |f(x0 + h_j) − f(x0 − h_j)|/(2h_j) for f″. The median is robust to a wrong
  slope at a large h (aliasing) or a tiny h (noise). Without the second term the model fails
  where f(x0) ≈ 0. The model does not see rounding inside the evaluation of f beyond a
  relative ε (cancellation in f's own formula, e.g. exp(x) − 1 or 1 − cos x near 0, whose
  absolute error is ε, not ε|f|); ``roundoff_obs`` below measures it from the trace.
* ``roundoff_obs`` (ν_k, observed round-off): round-off grows like h^{−q} as h shrinks, while
  truncation shrinks, so a later level i whose difference Δ_i = |D_i − D_{i−1}| *grew*
  (Δ_i > Δ_{i−1}) shows round-off of at least about Δ_i/(1 + 2^{−q}) at h_i (|Δ_i| ≤
  |r_i| + |r_{i−1}| with |r_{i−1}| ≈ 2^{−q}|r_i|), which is (h_i/h_k)^q times that at h_k:
  ν_k = max_{i > k, Δ grew} Δ_i/(1 + 2^{−q})·2^{−q(i−k)} (0 when no later level grew; for
  Richardson Δ_i is the diagonal difference).

A level is *collapsed* when rounding merges two of its abscissae (fl(x0 + oᵢh) not strictly
increasing in oᵢ, x0 included), e.g. h < ulp(x0)/2. Its estimate is shown, but it has no
err_est or roundoff and is never selected, and the next level has no err_est.

**Result.** A level is *usable* when k ≥ 1, it is finite and not collapsed, and it has
err_est and roundoff. Its local bound is b_k = err_est_k + max(roundoff_k, ν_k) (the
step-selection quantity of Ridders' method, Numerical Recipes 3rd ed., §5.7, with the
observed round-off). Only levels in the truncation regime may confirm another level (the
safeguard of NR's ``dfridr``, which stops once the error grows): the *truncation run* is the
longest run of consecutive usable levels with strictly decreasing Δ_k (the first on ties),
and the last *confirming* level c is the first usable level after that run whose observed
round-off exceeds 4× the model, max(ν_c, own_c) > 4·roundoff_c (the model is broken, so
the levels after c are noise that b cannot bound), or the last level if there is none.
own_c = Δ_c/(1 + 2^{−q}) when Δ_c grew and the run has ≥ 3 levels, else 0: rounding can
start with one jump (to f values that are bitwise equal) after which nothing grows again,
but after a shorter run such a jump is no evidence (a stencil that leaves a kink jumps to
the exact slope of the linear branch). For ``richardson_extrapolation`` the run and c use
the base column D(k, 0) (see there). Then

    B_k = max(b_k, max_{k < j ≤ c, j usable} (|D_k − D_j| − b_j)).

A level is *confirmed* when at least one such j exists, so its second term was tested.
``Result.x`` (= ``Result.fun``) is D_k* with k* = argmin B_k over the confirmed levels (the
first on ties), or over all usable levels when none is confirmed. ``converged`` is True when
k* is confirmed and B_k* ≤ tol·max(1, |D_k*|).

Preconditions: (1) the sweep must reach steps below the length scale of f. When every h is
beyond that scale (h0 ≫ the width of a peak, or h near whole periods of a periodic f), the
estimates can agree with each other at a wrong value (e.g. sin(10x) at x0 = π/2 with
h = 10, 5, 2.5, 1.25 gives 0.0506, 0.0525, 0.0529, 0.0531, but f′ = −10); no rule that uses
only these values can detect it. (2) For an f that cancels inside its own formula, h0 must
be above the round-off regime: when the whole sweep is noise, a run of levels whose f values
are bitwise equal (D_k = 0 at consecutive k) agrees with itself and nothing in the trace
contradicts it.

# NOTE: Ridders' rule minimizes b_k alone. Two estimates can agree by chance (aliasing:
# h0 = 2π for sin gives D_0 = D_1 = 0; noise at tiny h), which makes b_k ≈ 0 at a wrong
# value. The second term is a lower bound on the error at k whenever the bound at j is valid
# (|D_k − f′| ≥ |D_k − D_j| − |D_j − f′|), so it rejects such levels using the later,
# smaller steps. On a sweep of 11 880 runs it removed all 498 false "converged" results of
# the plain rule and lost no correct one.
# NOTE: the second term needs a *valid* b_j. Where f cancels internally (exp(x) − 1 at
# x0 = 1e-9), the deep levels give D_j = 0 with err_est_j = 0 and a tiny model roundoff_j,
# so every correct level got B ≈ |f′| and the D = 0 level was selected. Hence ν (the observed
# round-off) in b, and the cut at c (no noise level may confirm). On 8 000 random runs on
# such functions this cut the false "converged" results (error > 10·tol·max(1, |D|)) from
# 235 to 32; on 20 040 runs on the library and aliasing sweeps it added no false result and
# kept every converged run except 6 whose old bound was too small (the abscissa term of the
# model used aliased slopes in g). The 32 left mostly violate precondition 2; the rest are
# forward, backward and five-point runs where one jump to bitwise-equal f values follows a
# short run (0.3 % of the converged runs with h0 ≥ 1e-3); the central, second-derivative and
# Richardson formulas had none.

The exact derivative (``problem.grad`` / ``problem.hess``) is used only for the ``error``
column and is not counted as an evaluation, because the methods do not use it. Function
values are cached by abscissa, so ``n_fev`` counts distinct points (the five-point stencil
reuses x0 ± 2h_k = x0 ± h_{k−1}). Evaluations that raise an arithmetic error give NaN.

``Step.x`` and ``Step.fun`` both hold the estimate D_k; ``Step.step_size`` is h_k.

Info keys:
    h: float — the step h_k = h0/2^k.
    estimate: float — D(h_k) (for Richardson: the diagonal entry D(k, k)); NaN (JSON null) when
        it is not finite (a stencil value is ±inf or NaN).
    error: float | None — |estimate − exact derivative|, or None if it is unknown.
    err_est: float | None — the truncation-error estimate |D_k − D_{k−1}|/(2^p − 1); for
        ``richardson_extrapolation`` |D(k, k) − D(k−1, k−1)| (None at k = 0, after a non-finite
        level, after a collapsed level, at a collapsed level, or at the first row after a table
        restart).
    roundoff: float | None — the rounding-error model above (None if non-finite or collapsed).
    roundoff_obs: float | None — the observed round-off ν_k above (None if the level is not
        usable).
    confirms: bool — True when the level is usable and may confirm earlier levels (k ≤ c).
    collapsed: bool — True when rounding merged two abscissae of the stencil (see above);
        always False for ``complex_step``, whose abscissa x0 + ih is exact.
    stencil: [[x, f(x)]] — the points the formula uses at this level (for the secant/chord
        picture); for ``complex_step`` the single point [x0, Re f(x0 + ih)].
    weights: [float] — the coefficients cᵢ/h^q aligned with ``stencil``:
        estimate = Σ weights[i]·stencil[i][1] (for ``complex_step``: [1/h], applied to Im f).
    x0: float — the point of differentiation.
    row: [float] — row k of the Richardson table D(k, 0), …, D(k, j) (``richardson_extrapolation``).
    imag: float — Im f(x0 + ih) (``complex_step``).

Result.extra keys:
    exact: the exact derivative or None; error: |D_k* − exact| or None;
    k_best / h_best / bound_best: the selected level k*, its step and its bound B_k*;
    k_trunc_end: the last level of the truncation run (None without usable levels);
    k_confirm_last: c, the last level that may confirm (None without usable levels);
    h_opt: the textbook optimal step (see each method) or None; h_opt_rule: its formula;
    h_min_error: the h with the smallest actual error (None when the exact value is unknown);
    order: p; derivative: q.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ..core.counting import Counted, scalar_problem, start_scalar
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step

_EPS = float(np.finfo(float).eps)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


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


@dataclass(frozen=True)
class _Stencil:
    """D(h) = Σᵢ ints[i]·f(x0 + offsets[i]·h) / (den·h^q); cᵢ = ints[i]/den."""

    offsets: tuple[int, ...]
    ints: tuple[int, ...]  # integer weights: ints[i]·f is exact for ±1, ±2, ±8
    den: int  # common denominator
    q: int  # derivative order
    p: int  # truncation order


_STENCILS: dict[str, _Stencil] = {
    "forward_difference": _Stencil((0, 1), (-1, 1), 1, q=1, p=1),
    "backward_difference": _Stencil((-1, 0), (-1, 1), 1, q=1, p=1),
    "central_difference": _Stencil((-1, 1), (-1, 1), 2, q=1, p=2),
    "five_point_stencil": _Stencil((-2, -1, 1, 2), (1, -8, 8, -1), 12, q=1, p=4),
    "second_derivative_central": _Stencil((-1, 0, 1), (1, -2, 1), 1, q=2, p=2),
}


def _fsum(terms: Iterable[float]) -> float:
    """Correctly rounded Σ terms (``math.fsum``), with IEEE semantics when it cannot be.

    ``math.fsum`` raises on ``inf + (−inf)`` and when a finite sum overflows; then the plain
    sum gives the IEEE value (NaN or ±inf), which the callers detect as non-finite.
    """
    values = list(terms)
    try:
        return math.fsum(values)
    except (ValueError, OverflowError):
        return float(sum(values))


def _start(prob: Problem, x0: float | None) -> float:
    """The point of differentiation; it must be finite."""
    x = start_scalar(prob, x0)
    if not math.isfinite(x):
        raise ValueError(f"x0 must be finite, got {x!r}")
    return x


def _collapsed(x0: float, h: float, offsets: Iterable[int]) -> bool:
    """True when rounding merges two abscissae: fl(x0 + oᵢh), with x0 itself included, are
    not strictly increasing in the offset oᵢ. This happens when h is below about half the
    spacing of the floats at x0, e.g. 1 + 1e-17 == 1."""
    grid = [x0 + o * h for o in sorted({0, *offsets})]
    return any(right <= left for left, right in pairwise(grid))


def _validate(h0: float, levels: int, tol: float) -> tuple[float, int, float]:
    h0 = float(h0)
    if not (math.isfinite(h0) and h0 > 0.0):
        raise ValueError(f"h0 must be a positive finite number, got {h0!r}")
    if int(levels) != levels or levels < 0:
        raise ValueError(f"levels must be an integer ≥ 0, got {levels!r}")
    if not (math.isfinite(tol) and tol > 0.0):
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    return h0, int(levels), float(tol)


def _exact_derivative(prob: Problem, x0: float, order: int) -> float | None:
    """The exact f′(x0) or f″(x0) from the problem, for the error column only (not counted)."""
    fn = prob.grad if order == 1 else prob.hess
    if fn is None:
        return None
    with np.errstate(all="ignore"):
        value = float(fn(x0))
    return value if math.isfinite(value) else None


def _error(estimate: float, exact: float | None) -> float | None:
    if exact is None or not math.isfinite(estimate):
        return None
    return abs(estimate - exact)


def _finite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _h_opt(x0: float, constant: float, power: float) -> float:
    """h_opt = constant·ε^power·max(1, |x0|) (N&W §8.1 scaling by max(1, |x0|))."""
    return constant * _EPS**power * max(1.0, abs(x0))


@dataclass
class _Level:
    """Per-level numbers that feed the step selection, plus the data its Step displays."""

    h: float
    estimate: float  # NaN when not finite
    diff: float | None  # Δ_k = |D_k − D_{k−1}| (None exactly when err_est is None)
    err_est: float | None
    roundoff: float | None
    collapsed: bool
    stencil: list[list[float]]
    weights: list[float]
    more: dict[str, Any]  # method-specific info keys
    # Richardson only: (Δ, roundoff) of the base column D(k, 0), which classifies the regime;
    # None means (diff, roundoff). See richardson_extrapolation.
    base: tuple[float, float] | None = None


#: NOTE: the observed round-off may exceed the a-priori model by this factor before the
#: model counts as broken. Rounding errors of a correctly modeled f stay below about 1× the
#: model (it adds the worst cases); on the cancellation sweep in the module docstring 2, 4
#: and 16 gave the same 32 false results, and 4 leaves a margin on both sides.
_MODEL_BROKEN = 4.0
#: The truncation run must have at least this many levels (two successive decreases) before
#: one jump right after it counts as round-off. NOTE: a shorter run is no trend; e.g. at a
#: kink the stencil leaves the kink with one jump to the exact slope of the linear branch.
_MIN_RUN = 3


@dataclass(frozen=True)
class _Selection:
    """The output of :func:`_select`: B_k of the usable levels and the regime boundaries."""

    bounds: dict[int, float]
    confirmed: set[int]
    nu: list[float | None]  # observed round-off ν_k (None when not usable)
    trunc_end: int | None  # last level of the truncation run
    confirm_last: int | None  # c: the last level that may confirm another


def _observed_roundoff(diffs: list[float | None], q: int) -> list[float]:
    """ν_k = max_{i > k, Δ_i > Δ_{i−1}} Δ_i/(1 + 2^−q)·2^{−q(i−k)} (module docstring), by one
    backward pass; a None difference neither grows nor is grown past."""
    shrink = 2.0**q
    nu = [0.0] * len(diffs)
    carry = 0.0
    for k in range(len(diffs) - 1, -1, -1):
        nu[k] = carry
        d, d_prev = diffs[k], diffs[k - 1] if k > 0 else None
        if d is not None and d_prev is not None and d > d_prev:
            carry = max(carry, d / (1.0 + 1.0 / shrink))
        carry /= shrink
    return nu


def _select(levels: list[_Level], q: int) -> _Selection:
    """B_k for every usable level (module docstring, "Result")."""
    n = len(levels)
    usable = [
        not lev.collapsed
        and lev.err_est is not None
        and lev.diff is not None
        and lev.roundoff is not None
        and _finite(lev.estimate, lev.err_est + lev.roundoff)
        for lev in levels
    ]
    diffs = [lev.diff if ok else None for lev, ok in zip(levels, usable, strict=True)]
    nu_all = _observed_roundoff(diffs, q)
    nu: list[float | None] = [v if ok else None for v, ok in zip(nu_all, usable, strict=True)]
    # The regime signals: the estimate itself, or Richardson's base column.
    reg = [
        (lev.base if lev.base is not None else (lev.diff, lev.roundoff)) if ok else (None, None)
        for lev, ok in zip(levels, usable, strict=True)
    ]
    reg_diff = [d for d, _ in reg]
    reg_nu = _observed_roundoff(reg_diff, q)

    # The truncation run: the longest run of usable levels with strictly decreasing Δ.
    trunc_end: int | None = None
    best_len = 0
    start: int | None = None
    for k in range(n):
        d = reg_diff[k]
        if d is None:
            start = None
            continue
        d_prev = reg_diff[k - 1] if k > 0 else None
        if start is None or d_prev is None or not d < d_prev:
            start = k
        if k - start + 1 > best_len:
            best_len, trunc_end = k - start + 1, k
    if trunc_end is None:
        return _Selection({}, set(), nu, None, None)

    # c: the first usable level after the run whose observed round-off, its own growth
    # included, breaks the model (rounding can start with one jump, e.g. to f values that
    # are bitwise equal, after which no difference grows again).
    confirm_last = n - 1
    shrink = 2.0**q
    for j in range(trunc_end + 1, n):
        d, d_prev, model = reg_diff[j], reg_diff[j - 1], reg[j][1]
        grew = d is not None and d_prev is not None and d > d_prev
        own = d / (1.0 + 1.0 / shrink) if grew and d is not None and best_len >= _MIN_RUN else 0
        if model is not None and max(reg_nu[j], own) > _MODEL_BROKEN * model:
            confirm_last = j
            break

    local: list[float | None] = [None] * n  # b_k = err_est_k + max(roundoff_k, ν_k)
    for k in range(n):
        lev, nu_k = levels[k], nu[k]
        if lev.err_est is not None and lev.roundoff is not None and nu_k is not None:
            local[k] = lev.err_est + max(lev.roundoff, nu_k)
    bounds: dict[int, float] = {}
    confirmed: set[int] = set()
    for k, b_k in enumerate(local):
        if b_k is None:
            continue
        # Consistency with every later confirming level j: if both local bounds held,
        # |D_k − D_j| ≤ |e_k| + |e_j| ≤ |e_k| + b_j, so |e_k| ≥ |D_k − D_j| − b_j.
        bound = b_k
        for j in range(k + 1, confirm_last + 1):
            b_j = local[j]
            if b_j is not None:
                bound = max(bound, abs(levels[k].estimate - levels[j].estimate) - b_j)
                confirmed.add(k)
        bounds[k] = bound
    return _Selection(bounds, confirmed, nu, trunc_end, confirm_last)


def _finish(
    method: str,
    x0: float,
    levels: list[_Level],
    fc: _CachedF | Counted,
    tol: float,
    exact: float | None,
    order: int,
    derivative: int,
    h_opt: float | None,
    h_opt_rule: str,
) -> Result:
    """Select k* = argmin_k B_k (see the module docstring) and build the trace and Result."""
    sel = _select(levels, derivative)
    bounds, confirmed = sel.bounds, sel.confirmed
    # Prefer confirmed levels; an unconfirmed level is chosen only when none is confirmed.
    pool = [k for k in bounds if k in confirmed] or list(bounds)
    best_k = min(pool, key=lambda k: (bounds[k], k)) if pool else None
    best_bound = bounds[best_k] if best_k is not None else math.inf
    last = sel.confirm_last
    trace = [
        _step(
            k,
            lev,
            x0,
            exact,
            roundoff_obs=sel.nu[k],
            confirms=sel.nu[k] is not None and last is not None and k <= last,
        )
        for k, lev in enumerate(levels)
    ]

    errors = [(_error(lev.estimate, exact), lev.h) for lev in levels]
    known = [(e, h) for e, h in errors if e is not None]
    h_min_error = min(known)[1] if known else None
    n_nonfinite = sum(not math.isfinite(lev.estimate) for lev in levels)

    if best_k is None:
        finite_levels = [lev for lev in levels if math.isfinite(lev.estimate)]
        estimate = finite_levels[-1].estimate if finite_levels else math.nan
        if len(levels) == 1:
            msg = "levels = 0: a single estimate has no error estimate"
        elif not finite_levels:
            msg = "every estimate is non-finite (f is not finite on the stencils)"
        else:
            msg = (
                "no level has an error estimate (it needs two consecutive finite levels "
                "whose abscissae do not collapse)"
            )
        extra = {
            "exact": exact,
            "error": _error(estimate, exact),
            "k_best": None,
            "h_best": None,
            "bound_best": None,
            "k_trunc_end": sel.trunc_end,
            "k_confirm_last": sel.confirm_last,
            "h_opt": h_opt,
            "h_opt_rule": h_opt_rule,
            "h_min_error": h_min_error,
            "order": order,
            "derivative": derivative,
        }
        return Result(
            method, estimate, estimate, False, msg, len(levels) - 1, fc.n, trace=trace, extra=extra
        )

    best = levels[best_k]
    within = best_bound <= tol * max(1.0, abs(best.estimate))
    converged = within and best_k in confirmed
    rel = "≤" if within else ">"
    msg = (
        f"best h = {best.h:.3g} (level {best_k}): error bound {best_bound:.3g} "
        f"{rel} tol·max(1, |D|)"
    )
    if best_k not in confirmed:
        if last is not None and last < len(levels) - 1:
            msg += (
                f"; no later level confirms this bound (the levels after {last} are "
                "dominated by round-off: increase h0)"
            )
        else:
            msg += "; no later level confirms this bound (increase levels)"
    if n_nonfinite:
        msg += f"; {n_nonfinite} level(s) with non-finite values were skipped"
    extra = {
        "exact": exact,
        "error": _error(best.estimate, exact),
        "k_best": best_k,
        "h_best": best.h,
        "bound_best": best_bound,
        "k_trunc_end": sel.trunc_end,
        "k_confirm_last": sel.confirm_last,
        "h_opt": h_opt,
        "h_opt_rule": h_opt_rule,
        "h_min_error": h_min_error,
        "order": order,
        "derivative": derivative,
    }
    return Result(
        method,
        best.estimate,
        best.estimate,
        converged,
        msg,
        len(levels) - 1,
        fc.n,
        trace=trace,
        extra=extra,
    )


def _step(
    k: int,
    lev: _Level,
    x0: float,
    exact: float | None,
    *,
    roundoff_obs: float | None,
    confirms: bool,
) -> Step:
    return Step(
        k,
        lev.estimate,
        lev.estimate,
        step_size=lev.h,
        info={
            "h": lev.h,
            "estimate": lev.estimate,
            "error": _error(lev.estimate, exact),
            "err_est": lev.err_est,
            "roundoff": lev.roundoff,
            "roundoff_obs": roundoff_obs,
            "confirms": confirms,
            "collapsed": lev.collapsed,
            "stencil": [list(pt) for pt in lev.stencil],
            "weights": list(lev.weights),
            "x0": x0,
            **lev.more,
        },
    )


def _diff_and_err_est(
    estimate: float, prev: _Level | None, factor: float
) -> tuple[float | None, float | None]:
    """(Δ_k, err_est_k) = (|D_k − D_{k−1}|, Δ_k/factor), or (None, None) without a usable
    previous level (none, collapsed, or non-finite)."""
    if prev is None or prev.collapsed or not _finite(estimate, prev.estimate):
        return None, None
    diff = abs(estimate - prev.estimate)
    return diff, diff / factor


# --------------------------------------------------------------------------------------
# Fixed-stencil finite differences
# --------------------------------------------------------------------------------------


_H_OPT: dict[str, tuple[float, float, str]] = {
    # (constant, power of ε, formula). Minimizing truncation + round-off bounds with
    # |f| ≈ |f^{(m)}| near x0 (Burden & Faires §4.1; Sauer §5.1.2):
    "forward_difference": (2.0, 1 / 2, "2·√ε·max(1,|x0|)"),  # h·M/2 + 2ε|f|/h
    "backward_difference": (2.0, 1 / 2, "2·√ε·max(1,|x0|)"),
    "central_difference": (3.0 ** (1 / 3), 1 / 3, "(3ε)^(1/3)·max(1,|x0|)"),  # h²M/6 + ε|f|/h
    "five_point_stencil": (
        11.25 ** (1 / 5),
        1 / 5,
        "(45ε/4)^(1/5)·max(1,|x0|)",
    ),  # h⁴M/30 + 1.5ε|f|/h
    "second_derivative_central": (
        48.0**0.25,
        1 / 4,
        "(48ε)^(1/4)·max(1,|x0|)",
    ),  # h²M/12 + 4ε|f|/h²
}


def _finite_difference(
    method: str,
    problem: Problem | Callable[[float], float],
    x0: float | None,
    h0: float,
    levels: int,
    tol: float,
) -> Result:
    st = _STENCILS[method]
    prob = scalar_problem(problem)
    x = _start(prob, x0)
    h0, levels, tol = _validate(h0, levels, tol)
    exact = _exact_derivative(prob, x, st.q)
    fc = _CachedF(prob.f)
    factor = 2.0**st.p - 1.0

    rows: list[_Level] = []
    slopes: list[float] = []  # |f′| estimates of the levels so far (for the abscissa term)
    for k in range(levels + 1):
        h = h0 / 2.0**k
        xs = [x + o * h for o in st.offsets]
        fs = [fc(xi) for xi in xs]
        scale = st.den * h**st.q
        # fsum gives the correctly rounded numerator of the computed f values.
        estimate = _fsum(c * fv for c, fv in zip(st.ints, fs, strict=True)) / scale
        if not math.isfinite(estimate):
            estimate = math.nan  # documented: a non-finite estimate is NaN (JSON null)
        collapsed = _collapsed(x, h, st.offsets)
        diff: float | None = None
        err_est: float | None = None
        roundoff: float | None = None
        if not collapsed:
            # NOTE: at a collapsed level the formula divides by h but the abscissae moved by 0
            # or by whole ulps of x0, so neither the Richardson estimate nor the round-off
            # model describes the error; the level is shown but cannot be selected.
            if st.q == 1:
                slope = abs(estimate)
            else:  # f″ stencil (x0 − h, x0, x0 + h): slope from the outer points
                slope = abs(fs[2] - fs[0]) / (2.0 * h)
            if math.isfinite(slope):
                slopes.append(slope)
            slope = statistics.median(slopes) if slopes else math.nan
            model = (
                _EPS
                * _fsum(
                    abs(c) * (abs(fv) + (abs(xi) * slope if o else 0.0))
                    for c, fv, xi, o in zip(st.ints, fs, xs, st.offsets, strict=True)
                )
                / scale
            )
            roundoff = model if math.isfinite(model) else None
            diff, err_est = _diff_and_err_est(estimate, rows[-1] if rows else None, factor)
        rows.append(
            _Level(
                h,
                estimate,
                diff,
                err_est,
                roundoff,
                collapsed,
                stencil=[[xi, fv] for xi, fv in zip(xs, fs, strict=True)],
                weights=[c / scale for c in st.ints],
                more={},
            )
        )
    const, power, rule = _H_OPT[method]
    return _finish(method, x, rows, fc, tol, exact, st.p, st.q, _h_opt(x, const, power), rule)


def _fd_params(levels_default: int = 30) -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            "h0", 0.1, min=1e-12, max=10.0, log=True, help="Largest step; level k uses h₀/2ᵏ."
        ),
        ParamSpec(
            "levels",
            levels_default,
            kind="int",
            min=0,
            max=60,
            help="Number of halvings of h (the sweep always runs to the end).",
        ),
        ParamSpec(
            "tol",
            1e-6,
            min=1e-15,
            max=1e-1,
            log=True,
            help="Converged when the best level's error bound ≤ tol·max(1, |D|).",
        ),
    )


@register(
    id="forward_difference",
    family="differentiation",
    name="Forward difference",
    params=_fd_params(),
    needs=("f", "x0"),
    order="O(h)",
    summary="Slope of the secant from x₀ to x₀ + h.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Eq. (4.1)",),
)
def forward_difference(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Forward difference D(h) = (f(x0 + h) − f(x0))/h (Burden & Faires, Eq. 4.1).

    Taylor: D(h) − f′(x0) = (h/2)·f″(ξ), so p = 1. Round-off ≈ 2ε|f|/h. Minimizing
    h·M/2 + 2ε|f|/h gives h_opt = 2√(ε|f|/M) ≈ 2√ε·max(1, |x0|) when |f| ≈ |f″| = M.
    f(x0) is evaluated once and reused at every level.

    Stopping: the sweep k = 0, …, levels always runs; Result.x is the estimate at the level
    with the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).
    """
    return _finite_difference("forward_difference", problem, x0, h0, levels, tol)


@register(
    id="backward_difference",
    family="differentiation",
    name="Backward difference",
    params=_fd_params(),
    needs=("f", "x0"),
    order="O(h)",
    summary="Slope of the secant from x₀ − h to x₀.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §4.1 (h < 0 in Eq. 4.1)",),
)
def backward_difference(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Backward difference D(h) = (f(x0) − f(x0 − h))/h.

    Taylor: D(h) − f′(x0) = −(h/2)·f″(ξ), so p = 1; h_opt as for the forward difference.

    Stopping: the sweep k = 0, …, levels always runs; Result.x is the estimate at the level
    with the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).
    """
    return _finite_difference("backward_difference", problem, x0, h0, levels, tol)


@register(
    id="central_difference",
    family="differentiation",
    name="Central difference",
    params=_fd_params(),
    needs=("f", "x0"),
    order="O(h²)",
    summary="Slope of the chord from x₀ − h to x₀ + h; the odd error terms cancel.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Eq. (4.5) (three-point midpoint)",
    ),
)
def central_difference(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Central difference D(h) = (f(x0 + h) − f(x0 − h))/(2h) (Burden & Faires, Eq. 4.5).

    Taylor: D(h) − f′(x0) = (h²/6)·f‴(ξ), so p = 2. Round-off ≈ ε|f|/h. Minimizing
    h²M/6 + ε|f|/h gives h_opt = (3ε|f|/M)^{1/3} ≈ (3ε)^{1/3}·max(1, |x0|).

    Stopping: the sweep k = 0, …, levels always runs; Result.x is the estimate at the level
    with the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).
    """
    return _finite_difference("central_difference", problem, x0, h0, levels, tol)


@register(
    id="five_point_stencil",
    family="differentiation",
    name="Five-point stencil",
    params=_fd_params(),
    needs=("f", "x0"),
    order="O(h⁴)",
    summary="Combine four neighbors so the error terms up to h³ cancel.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Eq. (4.6) (five-point midpoint)",),
)
def five_point_stencil(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Five-point midpoint formula (Burden & Faires, Eq. 4.6):
    D(h) = [f(x0 − 2h) − 8f(x0 − h) + 8f(x0 + h) − f(x0 + 2h)]/(12h).

    Taylor: D(h) − f′(x0) = −(h⁴/30)·f⁽⁵⁾(ξ), so p = 4. Round-off ≈ (18/12)·ε|f|/h.
    Minimizing h⁴M/30 + 1.5ε|f|/h gives h_opt = (45ε/4)^{1/5}·max(1, |x0|).
    Since x0 ± 2h_k = x0 ± h_{k−1}, each level after the first costs only 2 new evaluations.

    Stopping: the sweep k = 0, …, levels always runs; Result.x is the estimate at the level
    with the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).
    """
    return _finite_difference("five_point_stencil", problem, x0, h0, levels, tol)


@register(
    id="second_derivative_central",
    family="differentiation",
    name="Second derivative (central)",
    params=_fd_params(),
    needs=("f", "x0"),
    order="O(h²)",
    summary="Estimate f″(x₀) from the curvature of three equally spaced points.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Eq. (4.9)",),
)
def second_derivative_central(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Second-derivative midpoint formula D₂(h) = [f(x0 − h) − 2f(x0) + f(x0 + h)]/h²
    (Burden & Faires, Eq. 4.9).

    Taylor: D₂(h) − f″(x0) = (h²/12)·f⁽⁴⁾(ξ), so p = 2; the error column compares with
    ``problem.hess``. Round-off ≈ 4ε|f|/h², which grows like h⁻², so the V is steep.
    Minimizing h²M/12 + 4ε|f|/h² gives h_opt = (48ε)^{1/4}·max(1, |x0|).

    Stopping: the sweep k = 0, …, levels always runs; Result.x is the estimate at the level
    with the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).
    """
    return _finite_difference("second_derivative_central", problem, x0, h0, levels, tol)


# --------------------------------------------------------------------------------------
# Richardson extrapolation of central differences
# --------------------------------------------------------------------------------------


@register(
    id="richardson_extrapolation",
    family="differentiation",
    name="Richardson extrapolation",
    params=_fd_params(levels_default=20),
    needs=("f", "x0"),
    order="O(h^{2k+2}) on row k",
    summary="Extrapolate central differences on halved steps to h = 0.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §4.2",
        "Press et al., Numerical Recipes (3rd ed.), §5.7",
    ),
)
def richardson_extrapolation(
    problem: Problem | Callable[[float], float],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 20,
    tol: float = 1e-6,
) -> Result:
    """Richardson extrapolation of the central difference (Burden & Faires, §4.2).

    D(k, 0) = (f(x0 + h_k) − f(x0 − h_k))/(2h_k) has the even expansion
    f′(x0) + c₁h² + c₂h⁴ + …, so D(k, j) = D(k, j−1) + (D(k, j−1) − D(k−1, j−1))/(4^j − 1)
    eliminates one more term per column; D(k, k) is the value at h = 0 of the polynomial in h²
    through (h_i², D(i, 0)), i = 0, …, k. Step k reports D(k, k) and the whole row.

    err_est_k = |D(k, k) − D(k−1, k−1)| (Burden & Faires' stopping quantity). The round-off
    model is carried through the table exactly as the linear combination acts on bounds:
    ρ(k, 0) = ε(|f(x0 + h)| + |f(x0 − h)| + (|x0 + h| + |x0 − h|)·g)/(2h) (the model of the
    module docstring, g the median slope) and ρ(k, j) = (4^j ρ(k, j−1) + ρ(k−1, j−1))/(4^j − 1).
    h_opt is None: the best h depends on the column.

    The regime signals of the step selection (the truncation run and the cut c, module
    docstring) come from the base column D(k, 0), a plain central difference, with
    Δ⁰_k = |D(k, 0) − D(k−1, 0)| and roundoff ρ(k, 0); the bound b_k uses the diagonal.
    # NOTE: when f cancels internally, the base column becomes exactly 0 at tiny h and the
    # diagonal, which extrapolates those zeros, converges super-linearly to a wrong value; its
    # differences would form the longest decreasing run and hide the round-off regime.

    Stopping: the sweep k = 0, …, levels always runs; Result.x is D(k*, k*) at the level with
    the smallest bound B_k (module docstring), and converged means B_k* ≤ tol·max(1, |D|).

    # NOTE: if a level has a non-finite value or collapsed abscissae, the table restarts at
    # the next usable level (that level becomes a new row 0), instead of propagating NaN or a
    # meaningless difference quotient to every later row.
    """
    prob = scalar_problem(problem)
    x = _start(prob, x0)
    h0, levels, tol = _validate(h0, levels, tol)
    exact = _exact_derivative(prob, x, 1)
    fc = _CachedF(prob.f)

    rows: list[_Level] = []
    table: list[float] = []  # previous row D(k−1, ·); empty after a restart
    rho_row: list[float] = []
    slopes: list[float] = []  # |D(j, 0)| of the levels so far (for the abscissa term)
    for k in range(levels + 1):
        h = h0 / 2.0**k
        xp, xm = x + h, x - h
        fp, fm = fc(xp), fc(xm)
        base = (fp - fm) / (2.0 * h)
        collapsed = _collapsed(x, h, (-1, 1))
        if not collapsed and math.isfinite(base):
            slopes.append(abs(base))
        slope = statistics.median(slopes) if slopes else math.nan
        rho0 = _EPS * (abs(fp) + abs(fm) + (abs(xp) + abs(xm)) * slope) / (2.0 * h)
        diff: float | None = None
        err_est: float | None = None
        roundoff: float | None = None
        base_signals: tuple[float, float] | None = None
        if collapsed or not _finite(base, rho0):
            row = [base]
            table, rho_row = [], []
            estimate = base if collapsed and math.isfinite(base) else math.nan
        else:
            row, rho = [base], [rho0]
            for j in range(1, len(table) + 1):
                c = 4.0**j
                row.append(row[j - 1] + (row[j - 1] - table[j - 1]) / (c - 1.0))
                rho.append((c * rho[j - 1] + rho_row[j - 1]) / (c - 1.0))
            estimate, roundoff = row[-1], rho[-1]
            if table:
                diff = err_est = abs(row[-1] - table[-1])
                base_signals = (abs(base - table[0]), rho0)
            table, rho_row = row, rho
        rows.append(
            _Level(
                h,
                estimate,
                diff,
                err_est,
                roundoff,
                collapsed,
                stencil=[[xm, fm], [xp, fp]],
                weights=[-0.5 / h, 0.5 / h],
                more={"row": row},
                base=base_signals,
            )
        )
    return _finish(
        "richardson_extrapolation",
        x,
        rows,
        fc,
        tol,
        exact,
        2,
        1,
        None,
        "none (depends on the column)",
    )


# --------------------------------------------------------------------------------------
# Complex step
# --------------------------------------------------------------------------------------


@register(
    id="complex_step",
    family="differentiation",
    name="Complex-step derivative",
    params=(
        ParamSpec(
            "h0", 0.1, min=1e-30, max=10.0, log=True, help="Largest step; level k uses h₀/2ᵏ."
        ),
        ParamSpec(
            "levels",
            30,
            kind="int",
            min=0,
            max=60,
            help="Number of halvings of h (the sweep always runs to the end).",
        ),
        ParamSpec(
            "tol",
            1e-6,
            min=1e-15,
            max=1e-1,
            log=True,
            help="Converged when the best level's error bound ≤ tol·max(1, |D|).",
        ),
    ),
    needs=("f", "x0", "complex"),
    order="O(h²), no subtractive cancellation",
    summary="Take Im f(x₀ + ih)/h: no difference of nearly equal numbers, so h can be tiny.",
    references=(
        "Squire & Trapp, SIAM Review 40 (1998)",
        "Martins, Sturdza & Alonso, ACM TOMS 29 (2003)",
    ),
)
def complex_step(
    problem: Problem | Callable[[Any], Any],
    *,
    x0: float | None = None,
    h0: float = 0.1,
    levels: int = 30,
    tol: float = 1e-6,
) -> Result:
    """Complex-step derivative D(h) = Im f(x0 + ih)/h (Squire & Trapp, 1998).

    For f real-analytic, f(x0 + ih) = f(x0) + ih f′(x0) − (h²/2) f″(x0) − i(h³/6) f‴(x0) + …,
    so D(h) − f′(x0) = −(h²/6)·f‴(x0) + O(h⁴) (p = 2). No two nearly equal numbers are
    subtracted and the argument x0 + ih is exact, so the round-off is ≈ ε|D| for every h (the
    model used here): the error curve has no V, only the truncation branch and then a floor
    at ε. Any h ≤ h_opt = √(6ε)·max(1, |x0|) gives full accuracy when |f‴| ≈ |f′|.

    f must accept complex arguments and be complex-analytic near x0 (written with NumPy
    ufuncs, not ``math``); otherwise a ``ValueError`` is raised.
    """
    prob = scalar_problem(problem)
    x = _start(prob, x0)
    h0, levels, tol = _validate(h0, levels, tol)
    exact = _exact_derivative(prob, x, 1)
    fc = Counted(prob.f)

    rows: list[_Level] = []
    for k in range(levels + 1):
        h = h0 / 2.0**k
        z = complex(x, h)
        try:
            with np.errstate(all="ignore"):
                value = fc(z)
        except TypeError as exc:
            raise ValueError(f"complex_step: f must accept complex input ({exc})") from exc
        except (ArithmeticError, ValueError):
            value = complex(math.nan, math.nan)
        if not np.iscomplexobj(value):
            raise ValueError(
                "complex_step: f returned a real value for complex input; write f with NumPy "
                "ufuncs so that it is complex-analytic (e.g. np.sin, not math.sin or abs)"
            )
        w = complex(value)
        estimate = w.imag / h
        if not math.isfinite(estimate):
            estimate = math.nan
        roundoff = _EPS * abs(estimate)
        diff, err_est = _diff_and_err_est(estimate, rows[-1] if rows else None, 3.0)
        rows.append(
            _Level(
                h,
                estimate,
                diff,
                err_est,
                roundoff if math.isfinite(roundoff) else None,
                False,
                stencil=[[x, w.real]],
                weights=[1.0 / h],
                more={"imag": w.imag},
            )
        )
    return _finish(
        "complex_step",
        x,
        rows,
        fc,
        tol,
        exact,
        2,
        1,
        _h_opt(x, math.sqrt(6.0), 0.5),
        "√(6ε)·max(1,|x0|)",
    )


#: Parity fixtures for the web app: (method_id, problem_id, params). Traces stay < 300 steps.
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("forward_difference", "exp_0_1", {"h0": 0.1, "levels": 30}),
    ("backward_difference", "sin_0_pi", {"h0": 0.1, "levels": 24}),
    ("central_difference", "exp_0_1", {"h0": 0.1, "levels": 30}),
    ("central_difference", "abs_kink", {"h0": 0.2, "levels": 12}),
    ("five_point_stencil", "runge", {"h0": 0.1, "levels": 24}),
    ("second_derivative_central", "gaussian", {"h0": 0.5, "levels": 20}),
    ("richardson_extrapolation", "sin_0_pi", {"h0": 0.5, "levels": 12}),
    ("complex_step", "arctan_deriv", {"h0": 0.1, "levels": 40}),
]
