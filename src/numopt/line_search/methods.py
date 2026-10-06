"""Line searches: choose a step length α > 0 along a descent direction p.

All searches work on the one-dimensional restriction

    φ(α) = f(x + α p),        φ'(α) = ∇f(x + α p)ᵀ p,        φ'(0) < 0,

and accept a step by one of these tests (Nocedal & Wright (2006), §3.1):

* Armijo / sufficient decrease (eq. 3.4):   φ(α) ≤ φ(0) + c₁ α φ'(0)
* curvature (eq. 3.6b):                      φ'(α) ≥ c₂ φ'(0)
* strong curvature (eq. 3.7b):               |φ'(α)| ≤ c₂ |φ'(0)|
* Goldstein (eq. 3.11):  φ(0) + (1 - c) α φ'(0) ≤ φ(α) ≤ φ(0) + c α φ'(0)

Kinds (``search(kind, ...)`` and the registered demo methods share these ids):

``backtracking``     Armijo backtracking, N&W Alg. 3.1: α ← ρα until eq. (3.4) holds.
``strong_wolfe``     N&W Alg. 3.5 (bracketing) + Alg. 3.6 (zoom) for the strong Wolfe conditions
                     (3.7); zoom trials come from cubic (eq. 3.59) or quadratic (eq. 3.58)
                     interpolation, clamped into the safeguard interval, with bisection when
                     the bracket stalls (Moré & Thuente (1994), §4).
``weak_wolfe``       Lewis & Overton (2013) bisection/doubling search for the weak Wolfe
                     conditions (3.6).
``goldstein``        The same bisection/doubling bracket for the Goldstein conditions (3.11).
``exact_quadratic``  α = -∇fᵀp / pᵀ∇²f p, the exact minimizer of the quadratic model along p
                     (N&W eq. 5.6); exact when f is quadratic. Accepted when f does not
                     increase, or when the rise of f is within a rounding bound ``f_err`` that
                     the caller supplies and ∇f confirms the line minimizer (see
                     :func:`_exact_quadratic`).

Offset invariance: every test compares differences of f values, φ(α) - φ(0) ≤ c₁αφ'(0), never
the sum φ(0) + c₁αφ'(0), whose rounding grows with |φ(0)|. So f and f + C give the same trials
and the same verdict whenever f + C is computed exactly, and no tolerance is relative to |f|.

Rounding guard: a step α > 0 with fl(x + αp) = x does not move x, so φ(α) = φ(0) exactly. It
fails the Armijo test in exact arithmetic (c₁αφ'(0) < 0), but the computed test can hold when
c₁αφ'(0) underflows to zero, and the Goldstein lower test holds for it. Such a step is never
accepted: ``backtracking`` stops with a failure without evaluating it (every shorter step is
fixed too), and the expanding searches (``strong_wolfe`` in its bracketing phase, ``weak_wolfe``,
``goldstein``) treat it as too short without evaluating it. A strong Wolfe zoom trial that does
not move x is evaluated and fails the Armijo test.

``search`` is the helper that every n-D descent method calls. The five registered methods
(family ``line_search``) are didactic demos: each runs ONE line search from ``x0`` along the
steepest-descent or Newton direction and records one :class:`Step` per trial step. Like the
other n-D families, a demo accepts a bare callable f or a Problem without derivatives: ∇f is
then a central difference (``numopt.core.diff.gradient``, 2n f evaluations counted in ``n_fev``,
one gradient in ``n_gev``) and ∇²f a central difference of ∇f (``numopt.core.diff.hessian``, 2n
gradient evaluations counted in ``n_gev``, one Hessian in ``n_hev``).

Trace of a demo method: ``k = 0`` is α = 0 (the start); ``k ≥ 1`` is the k-th trial step, with
``Step.x = x0 + α p``, ``Step.fun = φ(α)``, ``Step.step_size = α`` and ``Step.grad_norm =
‖∇f(x0 + α p)‖`` when the gradient was evaluated at that trial (else ``None``). ``converged``
is ``True`` only when the last trial satisfies the kind's acceptance conditions.

Info keys (every step of every kind). A float that is NaN is exported as JSON null, so every
float below can be null in the exported trace:
    alpha: float — the trial step length α (0 at k = 0).
    phi: float | None — φ(α) = f(x0 + α p); null when f is not finite there (NaN).
    dphi: float | None — φ'(α) = ∇f(x0 + α p)ᵀp; None when the search did not evaluate the
        gradient at this trial, null when the gradient is not finite.
    phi0: float | None — φ(0) = f(x0); null only when f(x0) is NaN (the demo then fails).
    dphi0: float | None — φ'(0) = ∇f(x0)ᵀp: < 0 whenever a search runs. On a demo that fails
        before the first trial it is the computed value (0 at a stationary x0, ≥ 0 for a
        non-descent Newton direction); null when p could not be computed (singular Hessian,
        non-finite f(x0) or ∇f(x0)), and null or "±inf" when p is not finite.
    c1: float — Armijo constant c₁ (for ``goldstein``: the Goldstein constant c).
    c2: float | None — curvature constant c₂ (Wolfe kinds), else None.
    direction: [n] — the search direction p (all zeros when p could not be computed; entries
        can be null / "inf" / "-inf" when p is not finite).
    phase: str — "start" (k = 0), "backtrack", "expand" (no upper bound known yet),
        "zoom" (strong Wolfe Alg. 3.6), "bisect" (weak Wolfe / Goldstein with a finite
        bracket) or "exact".
    interval: [lo, hi] | None — the interval from which this trial was chosen
        (``hi`` may be +inf, exported as "inf"); None for "start", "backtrack" and "exact".
    accepted: bool — True only on the accepted (last) trial of a successful search.
    conditions: {name: bool | None} — the kind's tests evaluated at this trial (None when the
        needed φ'(α) was not computed):
        backtracking:    armijo
        strong_wolfe:    armijo, curvature, strong_curvature
        weak_wolfe:      armijo, curvature
        goldstein:       armijo (upper Goldstein test, with c), goldstein_lower
        exact_quadratic: decrease (φ(α) ≤ φ(0)), strong_curvature (|φ'(α)| ≤ 0.1·|φ'(0)|:
                         α is close to a stationary point of φ; None when φ'(α) was not
                         evaluated, i.e. when decrease held) and armijo (a diagnostic only:
                         the exact step need not satisfy it on a non-quadratic f). The demo
                         has no rounding bound (f_err = 0), so a trial is accepted exactly
                         when decrease holds.
Info keys (kind-specific):
    rho: float — backtracking contraction factor ρ (``backtracking`` only).
    alpha_lo, alpha_hi: float | None — the zoom end points of Alg. 3.6 (α_lo has the lowest
        φ among Armijo steps; not ordered); None outside the zoom phase (``strong_wolfe`` only).
    interp: str | None — how a zoom trial was chosen: "cubic" or "quadratic" (the interpolant's
        minimizer lies in the safeguard interval), "cubic_clamped" or "quadratic_clamped"
        (the minimizer was moved to the nearest end of the safeguard interval) or "bisection"
        (no interpolant minimizer, or the bracket stalled); None outside the zoom phase
        (``strong_wolfe`` only).
    pHp: float | None — the curvature pᵀ∇²f(x0)p of the quadratic model; null when the demo
        failed before it was computed (``exact_quadratic`` only).
    model_phi: float | None — the quadratic model q(α) = φ(0) + α φ'(0) + ½ α² pᵀ∇²f p at α;
        None when pᵀ∇²f p is not finite (``exact_quadratic`` only).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, Vector, as_vector

Kind = Literal["backtracking", "strong_wolfe", "weak_wolfe", "goldstein", "exact_quadratic"]
KINDS: tuple[str, ...] = Kind.__args__  # type: ignore[attr-defined]

#: Default Armijo constant c₁ (N&W p. 33: "c₁ is chosen to be quite small, say 1e-4").
C1_DEFAULT = 1e-4
#: Default Goldstein constant c ∈ (0, ½).
C_GOLDSTEIN_DEFAULT = 0.25
#: Default curvature constant c₂ (N&W p. 34: 0.9 for Newton and quasi-Newton directions).
C2_DEFAULT = 0.9

#: Zoom safeguard: an interpolated trial is clamped into [a + δw, b - δw] of the bracket [a, b]
#: of width w (N&W p. 58–59 safeguards). The bracket therefore shrinks by at least the factor
#: 1 - δ per zoom trial.
_SAFEGUARD = 0.1
#: Zoom stall test (Moré & Thuente (1994), ACM TOMS 20:286–307, §4 and MINPACK-2 dcsrch): when
#: the bracket did not shrink by this factor over the last two trials, the next trial is the
#: midpoint. With the clamp above this keeps a worst-case rate of √0.66 ≈ 0.81 per trial.
_STALL = 0.66
#: exact_quadratic: when f rose at the model step, ∇f must confirm the line minimizer by the
#: strong curvature condition (N&W eq. 3.7b) with this c₂. On a quadratic φ'(α) = 0 up to the
#: rounding of ∇f; 0.1 is the c₂ of an accurate line search (N&W p. 34, the CG value).
_EXACT_C2 = 0.1


@dataclass(frozen=True)
class LineSearchResult:
    """Outcome of :func:`search`.

    Attributes:
        alpha: The accepted step (0.0 if the search failed).
        f_new: f(x + alpha p) (equals f(x) on failure).
        g_new: ∇f(x + alpha p) when it was computed (Wolfe kinds, and ``exact_quadratic`` when
            f rose at the model step; ∇f(x) on failure), else None.
        n_fev: f evaluations made inside the search (including f(x) when ``f0`` was not given).
        n_gev: ∇f evaluations made inside the search (including ∇f(x) when ``g0`` was not given).
        success: True when the kind's acceptance conditions hold at ``alpha``.
        trials: Every (alpha, phi(alpha)) evaluated, in order (φ(0) is not a trial).
        message: Why the search stopped.
        n_hev: ∇²f evaluations made inside the search (``exact_quadratic`` with a callable
            ``hess`` only).
    """

    alpha: float
    f_new: float
    g_new: Vector | None
    n_fev: int
    n_gev: int
    success: bool
    trials: list[tuple[float, float]]
    message: str
    # NOTE: n_hev is an addition to the architecture contract (default 0, so readers of the
    # documented fields are unaffected); exact_quadratic evaluates the Hessian and the counts
    # must be exact.
    n_hev: int = 0


# --------------------------------------------------------------------------------------
# The line function φ and the trial record
# --------------------------------------------------------------------------------------


class _Line:
    """φ(α) = f(x + αp) and φ'(α) = ∇f(x + αp)ᵀp with exact call counts."""

    __slots__ = ("f", "grad", "n_fev", "n_gev", "p", "x")

    def __init__(
        self,
        f: Callable[[Vector], Any],
        grad: Callable[[Vector], Any],
        x: Vector,
        p: Vector,
    ) -> None:
        self.f, self.grad, self.x, self.p = f, grad, x, p
        self.n_fev = 0
        self.n_gev = 0

    def point(self, alpha: float) -> Vector:
        return self.x + alpha * self.p

    def moves(self, alpha: float) -> bool:
        """False when fl(x + αp) = x: the step is lost to rounding (always for α = 0).

        fl(x_i + αp_i) is monotone in α, so if α does not move x, no shorter step does.
        """
        return not np.array_equal(self.point(alpha), self.x)

    def phi(self, alpha: float) -> float:
        self.n_fev += 1
        return float(self.f(self.point(alpha)))

    def gradient(self, alpha: float) -> Vector:
        self.n_gev += 1
        return as_vector(self.grad(self.point(alpha)))


@dataclass
class _Trial:
    """One trial step α with what the search learned there."""

    alpha: float
    phi: float
    phase: str
    interval: tuple[float, float] | None
    dphi: float | None = None
    g: Vector | None = None
    alpha_lo: float | None = None
    alpha_hi: float | None = None
    interp: str | None = None


@dataclass
class _Outcome:
    success: bool
    message: str
    trials: list[_Trial] = field(default_factory=list)


# NOTE: the tests below compare the difference φ(α) - φ(0) with the slope term, not φ(α) with
# the sum φ(0) + c·αφ'(0) as N&W print them. The two are equal in exact arithmetic; the sum
# rounds at the scale of |φ(0)|, so its verdict changes when a constant is added to f (audit
# finding), while the difference of two values that are computed exactly does not.


def _armijo(alpha: float, phi: float, phi0: float, dphi0: float, c: float) -> bool:
    """Sufficient decrease, N&W eq. (3.4). A non-finite φ(α) fails (the step is too long)."""
    return math.isfinite(phi) and phi - phi0 <= c * alpha * dphi0


def _goldstein_lower(alpha: float, phi: float, phi0: float, dphi0: float, c: float) -> bool:
    """The lower Goldstein test of N&W eq. (3.11): the step is not too short."""
    return math.isfinite(phi) and phi - phi0 >= (1.0 - c) * alpha * dphi0


def _decrease(phi: float, phi0: float) -> bool:
    """φ(α) ≤ φ(0): the first acceptance test of ``exact_quadratic``."""
    return math.isfinite(phi) and phi <= phi0


def _curvature(dphi: float, dphi0: float, c2: float) -> bool:
    """Weak curvature condition, N&W eq. (3.6b)."""
    return math.isfinite(dphi) and dphi >= c2 * dphi0


def _strong_curvature(dphi: float, dphi0: float, c2: float) -> bool:
    """Strong curvature condition, N&W eq. (3.7b): |φ'(α)| ≤ c₂|φ'(0)| = -c₂ φ'(0)."""
    return math.isfinite(dphi) and abs(dphi) <= -c2 * dphi0


def _conditions(
    kind: str,
    alpha: float,
    phi: float,
    dphi: float | None,
    phi0: float,
    dphi0: float,
    c1: float,
    c2: float,
) -> dict[str, bool | None]:
    """The acceptance tests of ``kind`` at one trial (for Step.info["conditions"])."""
    out: dict[str, bool | None] = {"armijo": _armijo(alpha, phi, phi0, dphi0, c1)}
    if kind in ("strong_wolfe", "weak_wolfe"):
        out["curvature"] = None if dphi is None else _curvature(dphi, dphi0, c2)
        if kind == "strong_wolfe":
            out["strong_curvature"] = None if dphi is None else _strong_curvature(dphi, dphi0, c2)
    elif kind == "goldstein":
        out["goldstein_lower"] = _goldstein_lower(alpha, phi, phi0, dphi0, c1)
    elif kind == "exact_quadratic":
        out["decrease"] = _decrease(phi, phi0)
        out["strong_curvature"] = (
            None if dphi is None else _strong_curvature(dphi, dphi0, _EXACT_C2)
        )
    return out


# --------------------------------------------------------------------------------------
# Interpolation for the zoom (N&W §3.5)
# --------------------------------------------------------------------------------------


def _cubic_minimizer(
    a: float, fa: float, da: float, b: float, fb: float, db: float
) -> float | None:
    """Minimizer of the cubic that interpolates φ, φ' at a and b, N&W eq. (3.59).

    With α_{i-1} = a, α_i = b, h = b - a, eq. (3.59) reads
        d₁ = φ'(a) + φ'(b) - 3 (φ(a) - φ(b)) / (a - b)
        d₂ = sign(b - a) √(d₁² - φ'(a) φ'(b))
        α  = b - (b - a) (φ'(b) + d₂ - d₁) / (φ'(b) - φ'(a) + 2 d₂).
    On t = a + s h the interpolant has P'(a + s h) = A s² + B s + C with A = φ'(a) + φ'(b) + 2d₁,
    -B/2 = d₁ + φ'(a), C = φ'(a) and discriminant B² - 4AC = 4d₂², so the same minimizer is
        s* = (d₁ + φ'(a) + d₂) / A = φ'(a) / (d₁ + φ'(a) - d₂).
    Returns None when the cubic has no strict local minimizer (d₁² - φ'(a)φ'(b) ≤ 0) or the
    formula breaks down.
    """
    # NOTE: eq. (3.59) as printed has a removable 0/0 (e.g. φ' = 0.75 - 3t² on [0, 1], whose
    # minimizer is -0.5) and cancels digits near it. We evaluate the same root with whichever
    # of the two equivalent forms adds terms of equal sign (the stable quadratic formula,
    # Higham (2002) §1.8); on random cubics its error is ~10x smaller than the printed form.
    h = b - a
    if h == 0.0:
        return None
    d1 = da + db + 3.0 * (fa - fb) / h
    disc = d1 * d1 - da * db
    # disc ≤ 0: no strict local minimizer (no stationary point, or an inflection point).
    if not math.isfinite(disc) or disc <= 0.0:
        return None
    d2 = math.copysign(math.sqrt(disc), h)
    m = d1 + da
    if (m >= 0.0) == (d2 >= 0.0):
        num, den = m + d2, da + db + 2.0 * d1
    else:
        num, den = da, m - d2
    if den == 0.0 or not math.isfinite(den):
        return None
    t = a + h * (num / den)
    return t if math.isfinite(t) else None


def _quadratic_minimizer(a: float, fa: float, da: float, b: float, fb: float) -> float | None:
    """Minimizer of the quadratic that interpolates φ(a), φ'(a) and φ(b).

    q(α) = φ(a) + φ'(a)(α - a) + C (α - a)², C = (φ(b) - φ(a) - φ'(a)(b - a)) / (b - a)²,
    minimized at α = a - φ'(a) / (2C) when C > 0. With a = 0 this is N&W eq. (3.58).
    """
    h = b - a
    if h == 0.0:
        return None
    curv = (fb - fa - da * h) / (h * h)
    if not math.isfinite(curv) or curv <= 0.0:
        return None
    t = a - da / (2.0 * curv)
    return t if math.isfinite(t) else None


# --------------------------------------------------------------------------------------
# The searches (each returns every trial; the accepted trial, if any, is the last one)
# --------------------------------------------------------------------------------------


def _backtracking(
    line: _Line, phi0: float, dphi0: float, alpha0: float, c1: float, rho: float, max_iter: int
) -> _Outcome:
    """Backtracking line search, N&W Alg. 3.1: α ← ρα until the Armijo condition holds.

    No gradient evaluations. A non-finite φ(α) fails the Armijo test, so the search also
    backs away from points outside the domain of f. The search fails, without evaluating f,
    once fl(x + αp) = x (α underflowed, or αp is below the rounding of x): φ(α) = φ(0) can then
    pass the computed Armijo test only by rounding, and no shorter step moves x either.
    """
    out = _Outcome(False, "")
    alpha = alpha0
    for _ in range(max_iter):
        if not line.moves(alpha):
            out.message = (
                f"the step underflowed before the Armijo condition held: x + αp = x "
                f"in floating point at α = {alpha:.6g}"
            )
            return out
        val = line.phi(alpha)
        out.trials.append(_Trial(alpha, val, "backtrack", None))
        if _armijo(alpha, val, phi0, dphi0, c1):
            out.success = True
            out.message = f"Armijo condition holds at α = {alpha:.6g}"
            return out
        alpha *= rho
    out.message = (
        f"no step satisfied the Armijo condition within max_iter={max_iter} trials "
        f"(last α = {out.trials[-1].alpha:.3g})"
    )
    return out


def _strong_wolfe(
    line: _Line,
    phi0: float,
    dphi0: float,
    alpha0: float,
    c1: float,
    c2: float,
    max_iter: int,
    alpha_max: float,
) -> _Outcome:
    """Strong Wolfe line search, N&W Alg. 3.5 (bracketing phase) + Alg. 3.6 (zoom).

    Bracketing (Alg. 3.5), α₀ = 0, α₁ = ``alpha0``:
        if φ(α_i) > φ(0) + c₁α_iφ'(0) or [φ(α_i) ≥ φ(α_{i-1}) and i > 1]: zoom(α_{i-1}, α_i)
        if |φ'(α_i)| ≤ -c₂φ'(0): accept α_i
        if φ'(α_i) ≥ 0: zoom(α_i, α_{i-1})
        α_{i+1} = min(2α_i, α_max)
    φ(0) and φ'(0) stay fixed for the whole search (the Armijo reference never moves).
    A step with fl(x + αp) = x is too short and is not evaluated: α doubles (up to α_max).
    """
    out = _Outcome(False, "")
    a_prev, phi_prev, dphi_prev = 0.0, phi0, dphi0
    alpha = alpha0
    i = 1
    while len(out.trials) < max_iter:
        if not line.moves(alpha):
            # NOTE: N&W evaluate every α_i. Here φ(α) = φ(0) exactly, which fails the Armijo
            # test and would zoom into steps that do not move x either; like the bracket
            # searches, we treat the step as too short (α_{i-1} = α carries φ(0), φ'(0)).
            if alpha >= alpha_max:
                out.message = (
                    f"x + alpha p == x in floating point even at alpha_max={alpha_max:.3g}"
                )
                return out
            a_prev = alpha
            alpha = min(2.0 * alpha, alpha_max)
            continue
        val = line.phi(alpha)
        trial = _Trial(alpha, val, "expand", (a_prev, alpha_max))
        out.trials.append(trial)
        if not _armijo(alpha, val, phi0, dphi0, c1) or (i > 1 and val >= phi_prev):
            return _zoom(
                line,
                out,
                phi0,
                dphi0,
                c1,
                c2,
                max_iter,
                (a_prev, phi_prev, dphi_prev),
                alpha,
                val,
                None,
            )
        g = line.gradient(alpha)
        d = float(g @ line.p)
        trial.dphi, trial.g = d, g
        if not math.isfinite(d):
            out.message = f"non-finite gradient at α = {alpha:.6g}"
            return out
        if _strong_curvature(d, dphi0, c2):
            out.success = True
            out.message = f"strong Wolfe conditions hold at α = {alpha:.6g}"
            return out
        if d >= 0.0:
            return _zoom(
                line,
                out,
                phi0,
                dphi0,
                c1,
                c2,
                max_iter,
                (alpha, val, d),
                a_prev,
                phi_prev,
                dphi_prev,
            )
        if alpha >= alpha_max:
            out.message = (
                f"phi still decreases at alpha_max={alpha_max:.3g}; f may be unbounded below "
                "along p"
            )
            return out
        # NOTE: N&W only require α_{i+1} ∈ (α_i, α_max); doubling (as in SciPy) is our choice.
        a_prev, phi_prev, dphi_prev = alpha, val, d
        alpha = min(2.0 * alpha, alpha_max)
        i += 1
    out.message = f"reached max_iter={max_iter} trials in the bracketing phase"
    return out


def _zoom(
    line: _Line,
    out: _Outcome,
    phi0: float,
    dphi0: float,
    c1: float,
    c2: float,
    max_iter: int,
    lo: tuple[float, float, float],
    a_hi: float,
    phi_hi: float,
    dphi_hi: float | None,
) -> _Outcome:
    """Zoom, N&W Alg. 3.6. Invariants: α_lo satisfies Armijo with the lowest φ seen so far,
    φ'(α_lo)(α_hi - α_lo) < 0, and the bracket contains strong Wolfe points.

    Each trial α_j is the minimizer of the cubic (eq. 3.59) through φ, φ' at both end points
    when φ'(α_hi) is known, else of the quadratic through φ(α_lo), φ'(α_lo), φ(α_hi)
    (eq. 3.58 generalized), clamped into [a + δw, b - δw] of the bracket [a, b] of width w.
    The midpoint is used when no interpolant minimizer exists or when the bracket did not
    shrink by the factor 0.66 over the last two trials (Moré & Thuente (1994), §4).
        if φ(α_j) > φ(0) + c₁α_jφ'(0) or φ(α_j) ≥ φ(α_lo): α_hi = α_j
        else: if |φ'(α_j)| ≤ -c₂φ'(0): accept α_j
              if φ'(α_j)(α_hi - α_lo) ≥ 0: α_hi = α_lo
              α_lo = α_j
    """
    # NOTE: N&W Alg. 3.6 uses cubic interpolation; φ'(α_hi) is unknown whenever α_hi failed the
    # Armijo test (Alg. 3.5/3.6 do not evaluate the gradient there), so we use the quadratic
    # interpolant in that case instead of spending a gradient evaluation.
    # NOTE: an estimate outside [a + δw, b - δw] is clamped, not replaced by the midpoint: when
    # α* is near one end the clamped trial shrinks the bracket to δw, where bisection would
    # take ~log2(1/δ) trials. Measured: Moré–Thuente function 2 from α₀ = 1 needs 12 trials
    # instead of 25, and steepest descent on Rosenbrock needs 2.2x fewer trials.
    a_lo, phi_lo, dphi_lo = lo
    widths: list[float] = []  # bracket width before each zoom trial
    while len(out.trials) < max_iter:
        a, b = (a_lo, a_hi) if a_lo < a_hi else (a_hi, a_lo)
        w = b - a
        widths.append(w)
        if dphi_hi is not None and math.isfinite(phi_hi):
            t, how = _cubic_minimizer(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi), "cubic"
        elif math.isfinite(phi_hi):
            t, how = _quadratic_minimizer(a_lo, phi_lo, dphi_lo, a_hi, phi_hi), "quadratic"
        else:
            t, how = None, "bisection"
        stalled = len(widths) >= 3 and w > _STALL * widths[-3]
        if t is not None and not stalled:
            lo_safe, hi_safe = a + _SAFEGUARD * w, b - _SAFEGUARD * w
            if not lo_safe <= t <= hi_safe:
                t, how = min(max(t, lo_safe), hi_safe), f"{how}_clamped"
        if t is None or stalled or not (a < t < b):
            # (a < t < b fails for a clamped t only when δw is below the spacing of floats)
            t, how = a + 0.5 * w, "bisection"
        if not (a < t < b):
            out.message = (
                f"the zoom interval [{a:.6g}, {b:.6g}] collapsed to machine precision "
                "before the strong Wolfe conditions held"
            )
            return out
        val = line.phi(t)
        trial = _Trial(t, val, "zoom", (a, b), alpha_lo=a_lo, alpha_hi=a_hi, interp=how)
        out.trials.append(trial)
        if not _armijo(t, val, phi0, dphi0, c1) or val >= phi_lo:
            a_hi, phi_hi, dphi_hi = t, val, None
            continue
        g = line.gradient(t)
        d = float(g @ line.p)
        trial.dphi, trial.g = d, g
        if not math.isfinite(d):
            out.message = f"non-finite gradient at α = {t:.6g}"
            return out
        if _strong_curvature(d, dphi0, c2):
            out.success = True
            out.message = f"strong Wolfe conditions hold at α = {t:.6g}"
            return out
        if d * (a_hi - a_lo) >= 0.0:
            a_hi, phi_hi, dphi_hi = a_lo, phi_lo, dphi_lo
        a_lo, phi_lo, dphi_lo = t, val, d
    out.message = f"reached max_iter={max_iter} trials before the zoom found a strong Wolfe step"
    return out


def _bisect_double(
    kind: str,
    line: _Line,
    phi0: float,
    dphi0: float,
    alpha0: float,
    c1: float,
    c2: float,
    max_iter: int,
    alpha_max: float,
) -> _Outcome:
    """Bisection/doubling bracket search for the weak Wolfe or the Goldstein conditions.

    Lewis & Overton (2013), weak Wolfe line search; ``lo = 0``, ``hi = +inf``, α = ``alpha0``:
        if the step is too long  (Armijo / upper Goldstein fails): hi = α
        elif the step is too short (curvature / lower Goldstein fails): lo = α
        else accept α
        α = (lo + hi)/2 if hi < inf else min(2·lo, α_max)
    Weak Wolfe evaluates φ'(α) only when the Armijo test passed; Goldstein never needs φ'.
    A step with fl(x + αp) = x is too short and is not evaluated (φ(α) = φ(0) exactly; the
    computed Goldstein tests could both hold by rounding).
    """
    out = _Outcome(False, "")
    name = "weak Wolfe" if kind == "weak_wolfe" else "Goldstein"
    lo, hi = 0.0, math.inf
    alpha = alpha0
    # Steps that do not move x add no trial; they are bounded by the float exponent range
    # (each one doubles α or halves the bracket), so the loop still terminates.
    while len(out.trials) < max_iter:
        if not line.moves(alpha):
            lo = alpha
        else:
            val = line.phi(alpha)
            trial = _Trial(alpha, val, "expand" if math.isinf(hi) else "bisect", (lo, hi))
            out.trials.append(trial)
            if not _armijo(alpha, val, phi0, dphi0, c1):
                hi = alpha
            else:
                if kind == "weak_wolfe":
                    g = line.gradient(alpha)
                    d = float(g @ line.p)
                    trial.dphi, trial.g = d, g
                    if not math.isfinite(d):
                        out.message = f"non-finite gradient at α = {alpha:.6g}"
                        return out
                    too_short = not _curvature(d, dphi0, c2)
                else:
                    too_short = not _goldstein_lower(alpha, val, phi0, dphi0, c1)
                if not too_short:
                    out.success = True
                    out.message = f"{name} conditions hold at α = {alpha:.6g}"
                    return out
                lo = alpha
        if math.isinf(hi):
            if lo >= alpha_max:
                out.message = (
                    f"x + alpha p == x in floating point even at alpha_max={alpha_max:.3g}"
                    if not line.moves(lo)
                    else f"phi still decreases at alpha_max={alpha_max:.3g}; "
                    "f may be unbounded below along p"
                )
                return out
            alpha = min(2.0 * lo, alpha_max)
        else:
            alpha = lo + 0.5 * (hi - lo)
            if not (lo < alpha < hi):
                out.message = (
                    f"the bracket [{lo:.6g}, {hi:.6g}] collapsed to machine precision "
                    f"before the {name} conditions held"
                )
                return out
    out.message = f"reached max_iter={max_iter} trials before the {name} conditions held"
    return out


def _exact_quadratic(
    line: _Line, phi0: float, dphi0: float, pHp: float, f_err: float = 0.0
) -> _Outcome:
    """α = -φ'(0) / pᵀHp, the minimizer of q(α) = φ(0) + αφ'(0) + ½α² pᵀHp (N&W eq. 5.6).

    The step is the exact line minimizer when f is quadratic along p. Success needs pᵀHp > 0
    and one of two tests at α:
        decrease:  φ(α) - φ(0) ≤ 0.
        rounding:  only when f rose: φ'(α) is evaluated, and the step is accepted when
                   |φ'(α)| ≤ 0.1·|φ'(0)| (strong curvature, N&W eq. 3.7b with c₂ = 0.1: ∇f
                   confirms the line minimizer) and the rise φ(α) - φ(0) ≤ 2·f_err, where
                   f_err ≥ 0 is the caller's bound on the absolute rounding error of each
                   computed value of f. A rise within that bound does not show that f increased.
    The gradient test alone cannot decide: φ'(α) = 0 also holds at a local maximum of φ, where f
    really rose (φ(t) = -t + ½t² + 4t³ - 3t⁴ has φ'(1) = 0, φ(1) - φ(0) = ½, and its model step
    is α = 1). And no tolerance computed from the values of f can bound their rounding error:
    for f = ½xᵀAx - bᵀx + c the error scales with the cancelling terms (≈ ε|c|), not with |f|,
    which is pure rounding noise near a minimizer with f* = 0. So the bound must come from the
    caller (for this f, Higham (2002) Thm 3.5 gives f_err = γ_{2n+4}(½|z|ᵀ|A||z| + |b|ᵀ|z| + |c|)
    over z ∈ {x, x + αp}, where the extra ε covers the rounding of x + αp).
    Both tests depend only on differences of f and on ∇f, so f + C gives the same verdict when
    f + C is computed exactly. When both tests fail, the search fails; the message says whether
    ∇f contradicts the model (f is not quadratic along p) or confirms the line minimizer (then
    either the values of f are rounding noise beyond f_err, or α is near a local maximum of φ).
    The Armijo test is reported but not required: the method promises the model minimizer, not
    sufficient decrease (on a quadratic it holds for every c₁ ≤ ½ in exact arithmetic).
    """
    # NOTE: N&W eq. (5.6) has no acceptance test. The rounding test replaces two tolerances
    # relative to |f| (100·n·ε·|f| and Hager & Zhang's 10⁻⁶|f|): they changed the verdict when a
    # constant was added to f, called a real rise of 0.1 "rounding error" at |f| ≥ 10⁵, and
    # rejected pure rounding rises on quadratics with f* = 0 (audit findings). The gradient
    # evaluation of a rise costs one ∇f, which the callers reuse as g_new.
    out = _Outcome(False, "")
    if not (math.isfinite(pHp) and pHp > 0.0):
        out.message = f"pᵀ∇²f p = {pHp:.3g} ≤ 0: the quadratic model has no minimizer along p"
        return out
    alpha = -dphi0 / pHp
    val = line.phi(alpha)
    trial = _Trial(alpha, val, "exact", None)
    out.trials.append(trial)
    if not math.isfinite(val):
        out.message = f"f(x + αp) is not finite at α = {alpha:.6g}"
        return out
    head = f"exact minimizer of the quadratic model along p: α = {alpha:.6g}"
    if _decrease(val, phi0):
        out.success = True
        out.message = head
        return out
    rise = val - phi0
    g = line.gradient(alpha)
    d = float(g @ line.p)
    trial.dphi, trial.g = d, g
    if not math.isfinite(d):
        out.message = f"f increased at α = {alpha:.6g} and the gradient there is not finite"
        return out
    slope = f"|φ′(α)| = {abs(d):.3g}"
    bound = f"{_EXACT_C2:g}·|φ′(0)| = {-_EXACT_C2 * dphi0:.3g}"
    increased = f"f increased by {rise:.3g} (from f(x) = {phi0:.6g}) at the model step"
    if not _strong_curvature(d, dphi0, _EXACT_C2):
        side = "overshoots" if d > 0.0 else "falls short of"
        out.message = (
            f"{increased} α = {alpha:.6g} and {slope} > {bound}: the quadratic model {side} "
            "the line minimizer (f is not quadratic along p, or ∇f is dominated by rounding "
            "error)"
        )
        return out
    if rise <= 2.0 * f_err:
        out.success = True
        out.message = (
            f"{head}; f rose by {rise:.3g} ≤ 2·f_err = {2.0 * f_err:.3g}, within the rounding "
            f"error of f, and {slope} ≤ {bound} confirms the line minimizer"
        )
        return out
    out.message = (
        f"{increased} α = {alpha:.6g} although {slope} ≤ {bound}: the rise exceeds the "
        f"rounding bound 2·f_err = {2.0 * f_err:.3g}, so either α is near a local maximum "
        "of φ (f is not quadratic along p) or the values of f are dominated by rounding "
        "error (pass f_err, a bound on the rounding error of f)"
    )
    return out


# --------------------------------------------------------------------------------------
# Public helper
# --------------------------------------------------------------------------------------


def _default_c1(kind: str, c1: float | None) -> float:
    if c1 is not None:
        return float(c1)
    return C_GOLDSTEIN_DEFAULT if kind == "goldstein" else C1_DEFAULT


def _validate(
    kind: str, c1: float, c2: float, rho: float, alpha0: float, alpha_max: float, max_iter: object
) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown line search kind {kind!r}; expected one of {KINDS}")
    if kind == "goldstein":
        if not 0.0 < c1 < 0.5:
            raise ValueError(f"goldstein needs 0 < c < 1/2 (passed as c1), got {c1}")
    elif not 0.0 < c1 < 1.0:
        raise ValueError(f"need 0 < c1 < 1, got c1={c1}")
    if kind in ("strong_wolfe", "weak_wolfe") and not c1 < c2 < 1.0:
        raise ValueError(f"{kind} needs 0 < c1 < c2 < 1, got c1={c1}, c2={c2}")
    if kind == "backtracking" and not 0.0 < rho < 1.0:
        raise ValueError(f"need 0 < rho < 1, got rho={rho}")
    if kind != "exact_quadratic":
        if not (math.isfinite(alpha0) and alpha0 > 0.0):
            raise ValueError(f"need a finite alpha0 > 0, got {alpha0}")
    if kind in ("strong_wolfe", "weak_wolfe", "goldstein"):
        # An infinite α_max lets the expansion double α to inf, where x + αp is not finite.
        if not math.isfinite(alpha_max):
            raise ValueError(f"need a finite alpha_max, got {alpha_max}")
        if not alpha0 <= alpha_max:
            raise ValueError(
                f"need alpha0 <= alpha_max, got alpha0={alpha0}, alpha_max={alpha_max}"
            )
    # float(inf).is_integer() and float(nan).is_integer() are False: no OverflowError here.
    if not (
        isinstance(max_iter, numbers.Real)
        and float(max_iter).is_integer()
        and float(max_iter) >= 1.0
    ):
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")


def _run(
    kind: str,
    line: _Line,
    phi0: float,
    dphi0: float,
    *,
    alpha0: float,
    pHp: float,
    c1: float,
    c2: float,
    rho: float,
    max_iter: int,
    alpha_max: float,
    f_err: float = 0.0,
) -> _Outcome:
    if kind == "backtracking":
        return _backtracking(line, phi0, dphi0, alpha0, c1, rho, max_iter)
    if kind == "strong_wolfe":
        return _strong_wolfe(line, phi0, dphi0, alpha0, c1, c2, max_iter, alpha_max)
    if kind in ("weak_wolfe", "goldstein"):
        return _bisect_double(kind, line, phi0, dphi0, alpha0, c1, c2, max_iter, alpha_max)
    return _exact_quadratic(line, phi0, dphi0, pHp, f_err)


def search(
    kind: str,
    f: Callable[[Vector], Any],
    grad: Callable[[Vector], Any],
    x: ArrayLike,
    p: ArrayLike,
    *,
    f0: float | None = None,
    g0: ArrayLike | None = None,
    alpha0: float = 1.0,
    hess: Callable[[Vector], Any] | ArrayLike | None = None,
    c1: float | None = None,
    c2: float = C2_DEFAULT,
    rho: float = 0.5,
    max_iter: int = 50,
    alpha_max: float = 1e3,
    f_err: float = 0.0,
) -> LineSearchResult:
    """Find a step length α along the descent direction ``p`` from ``x``.

    Args:
        kind: "backtracking", "strong_wolfe", "weak_wolfe", "goldstein" or "exact_quadratic"
            (see the module docstring for the algorithm and textbook reference of each).
        f, grad: f(x) -> float and ∇f(x) -> (n,). Every call made here is counted in
            ``n_fev`` / ``n_gev`` (if they are already ``Counted``, those counters see the same
            calls; add only one of the two to your totals).
        x, p: The current point and a descent direction, shape (n,).
        f0, g0: f(x) and ∇f(x) if already known (not re-evaluated; φ(0) is never recomputed).
        alpha0: The first trial step (unused by ``exact_quadratic``).
        hess: ∇²f as a callable x -> (n, n) or an (n, n) array; required by ``exact_quadratic``.
            For n = 1 a scalar (or a callable that returns one) is also accepted.
        c1: Armijo constant c₁ ∈ (0, 1); for ``goldstein`` the Goldstein constant c ∈ (0, ½).
            Default 1e-4 (``goldstein``: 0.25).
        c2: Curvature constant c₂ ∈ (c₁, 1) for the Wolfe kinds.
        rho: Backtracking contraction factor ρ ∈ (0, 1).
        max_iter: Maximum number of trial steps (evaluations of φ(α), α > 0).
        alpha_max: Finite upper limit on α for the expanding kinds (strong/weak Wolfe,
            Goldstein).
        f_err: ``exact_quadratic`` only: a bound (≥ 0, absolute) on the rounding error of each
            computed value f(x), f(x + αp). A rise of f by at most 2·f_err at the model step is
            accepted when ∇f confirms the line minimizer. The default 0 accepts no rise. Pass
            a bound built from the magnitudes of the terms of f (e.g. γ_{2n+4}(½|z|ᵀ|A||z| +
            |b|ᵀ|z| + |c|) at z = x and x + αp for ½zᵀAz - bᵀz + c, Higham (2002) Thm 3.5),
            never one relative to |f| alone: f + C must not change the verdict.

    Returns:
        A :class:`LineSearchResult`. ``success`` is True only when the accepted step satisfies
        the kind's conditions (``exact_quadratic``: f did not increase, or the rise is at most
        2·f_err and ∇f confirms the line minimizer, see :func:`_exact_quadratic`); on
        failure ``alpha = 0`` and ``f_new``/``g_new`` refer to x. ``trials`` is empty when
        ``backtracking`` fails because already ``alpha0`` does not move x in floating point.

    Raises:
        ValueError: on invalid parameters (including a non-finite ``alpha_max`` for the
            expanding kinds and a ``max_iter`` that is not a positive integer, such as inf or
            nan), a non-finite f(x) or ∇f(x)ᵀp, when ``p`` is not a
            descent direction (∇f(x)ᵀp ≥ 0), or for ``exact_quadratic`` when ``hess`` is
            missing, pᵀ∇²f p ≤ 0, or f_err is not a finite number ≥ 0.
    """
    # NOTE: the task signature lists c1=1e-4, while the Goldstein search needs c = 0.25 by
    # default; c1=None selects the per-kind default so both statements hold.
    c1v = _default_c1(kind, c1)
    _validate(kind, c1v, float(c2), float(rho), float(alpha0), float(alpha_max), max_iter)
    if kind == "exact_quadratic" and not (
        isinstance(f_err, numbers.Real) and math.isfinite(f_err) and f_err >= 0.0
    ):
        raise ValueError(f"f_err must be a finite number >= 0, got {f_err!r}")
    xv, pv = as_vector(x), as_vector(p)
    if xv.shape != pv.shape:
        raise ValueError(f"x and p must have the same shape, got {xv.shape} and {pv.shape}")
    line = _Line(f, grad, xv, pv)
    phi0 = line.phi(0.0) if f0 is None else float(f0)
    gv = line.gradient(0.0) if g0 is None else as_vector(g0)
    if gv.shape != xv.shape:
        raise ValueError(f"g0 must have shape {xv.shape}, got {gv.shape}")
    dphi0 = float(gv @ pv)
    if not (math.isfinite(phi0) and math.isfinite(dphi0)):
        raise ValueError(f"f(x)={phi0} and ∇f(x)ᵀp={dphi0} must be finite")
    if dphi0 >= 0.0:
        raise ValueError(f"p is not a descent direction: ∇f(x)ᵀp = {dphi0:.6g} ≥ 0")

    n_hev = 0
    pHp = math.nan
    if kind == "exact_quadratic":
        if hess is None:
            raise ValueError("exact_quadratic needs the Hessian (pass hess=...)")
        if callable(hess):
            n_hev = 1
            H = np.asarray(hess(xv), dtype=np.float64)
        else:
            H = np.asarray(hess, dtype=np.float64)
        if xv.size == 1 and H.size == 1:
            # The Problem convention for 1-D problems: hess returns a float, not a (1, 1) array.
            H = H.reshape(1, 1)
        if H.shape != (xv.size, xv.size):
            raise ValueError(f"hess must have shape {(xv.size, xv.size)}, got {H.shape}")
        pHp = float(pv @ (H @ pv))
        if not (math.isfinite(pHp) and pHp > 0.0):
            raise ValueError(f"exact_quadratic needs pᵀ∇²f p > 0, got {pHp:.6g}")

    out = _run(
        kind,
        line,
        phi0,
        dphi0,
        alpha0=float(alpha0),
        pHp=pHp,
        c1=c1v,
        c2=float(c2),
        rho=float(rho),
        max_iter=int(max_iter),
        alpha_max=float(alpha_max),
        f_err=float(f_err),
    )
    trials = [(t.alpha, t.phi) for t in out.trials]
    if out.success:
        last = out.trials[-1]
        return LineSearchResult(
            last.alpha, last.phi, last.g, line.n_fev, line.n_gev, True, trials, out.message, n_hev
        )
    return LineSearchResult(
        0.0, phi0, gv, line.n_fev, line.n_gev, False, trials, out.message, n_hev
    )


# --------------------------------------------------------------------------------------
# Registered demo methods: one line search from x0, one Step per trial
# --------------------------------------------------------------------------------------

P_DIRECTION = ParamSpec(
    "direction",
    "steepest",
    kind="choice",
    choices=("steepest", "newton"),
    help="Search direction p: steepest descent −∇f(x₀) or Newton −∇²f(x₀)⁻¹∇f(x₀).",
)
P_ALPHA0 = ParamSpec(
    "alpha0", 1.0, min=1e-4, max=100.0, log=True, help="First trial step length α₀."
)
P_C1 = ParamSpec(
    "c1",
    C1_DEFAULT,
    min=1e-6,
    max=0.5,
    log=True,
    help="Armijo constant c₁: accept only φ(α) ≤ φ(0) + c₁αφ'(0).",
)
# The slider ranges are disjoint where the searches need an order (c₁ < c₂, α₀ ≤ α_max), so
# every combination of slider values is valid: the Wolfe c₁ stays below the smallest c₂ and
# α_max starts at the largest α₀.
P_C1_WOLFE = ParamSpec(
    "c1",
    C1_DEFAULT,
    min=1e-6,
    max=0.09,
    log=True,
    help="Armijo constant c₁ < c₂: accept only φ(α) ≤ φ(0) + c₁αφ'(0).",
)
P_C2 = ParamSpec(
    "c2",
    C2_DEFAULT,
    min=0.1,
    max=0.999,
    help="Curvature constant c₂ ∈ (c₁, 1) (0.9 for Newton/quasi-Newton, 0.1 for CG).",
)
P_RHO = ParamSpec("rho", 0.5, min=0.05, max=0.95, help="Contraction factor: α ← ρα.")
P_MAX_ITER = ParamSpec(
    "max_iter", 50, kind="int", min=1, max=200, help="Maximum number of trial steps."
)
P_ALPHA_MAX = ParamSpec(
    "alpha_max",
    1e3,
    min=100.0,
    max=1e6,
    log=True,
    help="Largest step length the search may try (≥ α₀).",
)
P_C_GOLDSTEIN = ParamSpec(
    "c1",
    C_GOLDSTEIN_DEFAULT,
    min=0.01,
    max=0.49,
    help="Goldstein constant c ∈ (0, ½): φ(0)+(1-c)αφ'(0) ≤ φ(α) ≤ φ(0)+cαφ'(0).",
)


def _demo(
    kind: str,
    problem: Problem | Callable[..., Any],
    x0: Any,
    direction: str,
    *,
    alpha0: float = 1.0,
    c1: float = C1_DEFAULT,
    c2: float = C2_DEFAULT,
    rho: float = 0.5,
    max_iter: int = 50,
    alpha_max: float = 1e3,
) -> Result:
    """Run one line search of ``kind`` from x0 along p and record one Step per trial."""
    _validate(kind, c1, c2, rho, alpha0, alpha_max, max_iter)
    max_iter = int(max_iter)
    if direction not in ("steepest", "newton"):
        raise ValueError(f"direction must be 'steepest' or 'newton', got {direction!r}")
    prob = vector_problem(problem, x0=x0)
    needs_hess = direction == "newton" or kind == "exact_quadratic"
    x = start_point(prob, x0)
    scalar = prob.dim == 1
    grad_fn, hess_fn = prob.grad, prob.hess

    # 1-D problems take and return floats; adapt them to the vector convention.
    def f_vec(v: Vector) -> float:
        return float(prob.f(float(v[0]) if scalar else v))

    # Missing derivatives are central differences of the counted f and ∇f (as in the
    # unconstrained family), so their f / ∇f evaluations appear in n_fev / n_gev.
    def g_vec(v: Vector) -> Vector:
        if grad_fn is None:
            return diff.gradient(fc, v)
        return as_vector(grad_fn(float(v[0]) if scalar else v))

    def h_mat(v: Vector) -> np.ndarray:
        if hess_fn is None:
            return diff.hessian(gc, v)
        H = np.asarray(hess_fn(float(v[0]) if scalar else v), dtype=np.float64)
        return H.reshape(v.size, v.size)

    def as_x(v: Vector) -> Any:
        return float(v[0]) if scalar else v

    fc, gc, hc = Counted(f_vec), Counted(g_vec), Counted(h_mat)
    phi0 = float(fc(x))
    g0 = gc(x)
    H = hc(x) if needs_hess else None

    # The search direction p, φ'(0) and (exact_quadratic) the model curvature pᵀHp.
    p = np.zeros_like(x)
    dphi0 = math.nan
    pHp = math.nan
    failure: str | None = None
    if not finite(phi0, g0):
        failure = "f(x0) or ∇f(x0) is not finite"
    elif direction == "steepest":
        p = -g0
    else:
        assert H is not None
        try:
            p = -np.linalg.solve(H, g0)
        except np.linalg.LinAlgError:
            failure = "the Hessian at x0 is singular: no Newton direction"
    if failure is None:
        dphi0 = float(g0 @ p)
        if not finite(p, dphi0):
            failure = "the search direction is not finite"
        elif not np.any(g0):
            failure = "∇f(x0) = 0: x0 is a stationary point, so there is no descent direction"
        elif dphi0 >= 0.0:
            failure = f"p is not a descent direction (∇f(x0)ᵀp = {dphi0:.3g} ≥ 0)"
        elif kind == "exact_quadratic":
            assert H is not None
            pHp = float(p @ (H @ p))
            if not (math.isfinite(pHp) and pHp > 0.0):
                failure = (
                    f"pᵀ∇²f(x0)p = {pHp:.3g} ≤ 0: the quadratic model has no minimizer along p"
                )

    c2_info = c2 if kind in ("strong_wolfe", "weak_wolfe") else None
    direction_list = p.tolist()

    def info(t: _Trial | None, accepted: bool) -> dict[str, Any]:
        alpha, phi = (0.0, phi0) if t is None else (t.alpha, t.phi)
        dphi = dphi0 if t is None else t.dphi
        d: dict[str, Any] = {
            "alpha": alpha,
            "phi": phi,
            "dphi": dphi,
            "phi0": phi0,
            "dphi0": dphi0,
            "c1": c1,
            "c2": c2_info,
            "direction": direction_list,
            "phase": "start" if t is None else t.phase,
            "interval": None if t is None or t.interval is None else list(t.interval),
            "accepted": accepted,
            "conditions": _conditions(kind, alpha, phi, dphi, phi0, dphi0, c1, c2),
        }
        if kind == "backtracking":
            d["rho"] = rho
        elif kind == "strong_wolfe":
            d["alpha_lo"] = None if t is None else t.alpha_lo
            d["alpha_hi"] = None if t is None else t.alpha_hi
            d["interp"] = None if t is None else t.interp
        elif kind == "exact_quadratic":
            d["pHp"] = pHp
            d["model_phi"] = (
                phi0 + alpha * dphi0 + 0.5 * alpha * alpha * pHp if math.isfinite(pHp) else None
            )
        return d

    trace = [Step(0, as_x(x), phi0, float(np.linalg.norm(g0)), 0.0, info(None, False))]
    if failure is not None:
        return Result(
            kind,
            as_x(x),
            phi0,
            False,
            failure,
            0,
            fc.n,
            gc.n,
            hc.n,
            trace=trace,
            extra={"alpha": 0.0, "direction": direction_list, "phi0": phi0, "dphi0": dphi0},
        )

    line = _Line(fc, gc, x, p)
    out = _run(
        kind,
        line,
        phi0,
        dphi0,
        alpha0=alpha0,
        pHp=pHp,
        c1=c1,
        c2=c2,
        rho=rho,
        max_iter=max_iter,
        alpha_max=alpha_max,
    )
    n = len(out.trials)
    for k, t in enumerate(out.trials, start=1):
        g_norm = None if t.g is None else float(np.linalg.norm(t.g))
        accepted = out.success and k == n
        trace.append(Step(k, as_x(line.point(t.alpha)), t.phi, g_norm, t.alpha, info(t, accepted)))
    # A failed search can have no trial (backtracking whose α₀ does not move x).
    alpha_star, f_star = (out.trials[-1].alpha, out.trials[-1].phi) if out.success else (0.0, phi0)
    return Result(
        kind,
        as_x(line.point(alpha_star)) if out.success else as_x(x),
        f_star,
        out.success,
        out.message,
        n,
        fc.n,
        gc.n,
        hc.n,
        trace=trace,
        extra={"alpha": alpha_star, "direction": direction_list, "phi0": phi0, "dphi0": dphi0},
    )


_NW = "Nocedal & Wright (2006), Numerical Optimization, 2nd ed."


@register(
    id="backtracking",
    family="line_search",
    name="Backtracking (Armijo)",
    params=(P_DIRECTION, P_ALPHA0, P_C1, P_RHO, P_MAX_ITER),
    needs=("f", "grad"),
    summary="Start with a long step and shrink it by ρ until f decreases enough (Armijo).",
    references=(f"{_NW}, Algorithm 3.1 and eq. (3.4)", "Armijo (1966), Pacific J. Math. 16"),
)
def backtracking(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    direction: str = "steepest",
    alpha0: float = 1.0,
    c1: float = C1_DEFAULT,
    rho: float = 0.5,
    max_iter: int = 50,
) -> Result:
    """Backtracking line search (N&W Alg. 3.1) from x0 along p = -∇f(x0) or the Newton step.

    Trials α₀, ρα₀, ρ²α₀, ...; stops (converged) at the first α with
    φ(α) ≤ φ(0) + c₁αφ'(0) (eq. 3.4). Not converged after ``max_iter`` trials.
    """
    return _demo(
        "backtracking", problem, x0, direction, alpha0=alpha0, c1=c1, rho=rho, max_iter=max_iter
    )


@register(
    id="strong_wolfe",
    family="line_search",
    name="Strong Wolfe (bracket + zoom)",
    params=(P_DIRECTION, P_ALPHA0, P_C1_WOLFE, P_C2, P_MAX_ITER, P_ALPHA_MAX),
    needs=("f", "grad"),
    summary="Expand until a good step is bracketed, then zoom in with cubic interpolation.",
    references=(f"{_NW}, Algorithms 3.5 and 3.6, eqs. (3.7), (3.58), (3.59)",),
)
def strong_wolfe(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    direction: str = "steepest",
    alpha0: float = 1.0,
    c1: float = C1_DEFAULT,
    c2: float = C2_DEFAULT,
    max_iter: int = 50,
    alpha_max: float = 1e3,
) -> Result:
    """Strong Wolfe line search (N&W Alg. 3.5 + zoom Alg. 3.6) from x0 along p.

    Stops (converged) at the first trial with φ(α) ≤ φ(0) + c₁αφ'(0) and
    |φ'(α)| ≤ c₂|φ'(0)| (eq. 3.7). Not converged when ``max_iter`` trials are used, α reaches
    ``alpha_max`` while φ still decreases, or the zoom bracket collapses to rounding level.
    """
    return _demo(
        "strong_wolfe",
        problem,
        x0,
        direction,
        alpha0=alpha0,
        c1=c1,
        c2=c2,
        max_iter=max_iter,
        alpha_max=alpha_max,
    )


@register(
    id="weak_wolfe",
    family="line_search",
    name="Weak Wolfe (bisection)",
    params=(P_DIRECTION, P_ALPHA0, P_C1_WOLFE, P_C2, P_MAX_ITER, P_ALPHA_MAX),
    needs=("f", "grad"),
    summary="Halve a too-long step, double a too-short one, until both Wolfe conditions hold.",
    references=(
        "Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods, "
        "Math. Program. 141:135–163",
        f"{_NW}, eq. (3.6)",
    ),
)
def weak_wolfe(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    direction: str = "steepest",
    alpha0: float = 1.0,
    c1: float = C1_DEFAULT,
    c2: float = C2_DEFAULT,
    max_iter: int = 50,
    alpha_max: float = 1e3,
) -> Result:
    """Weak Wolfe bisection/doubling line search (Lewis & Overton 2013) from x0 along p.

    Stops (converged) at the first trial with φ(α) ≤ φ(0) + c₁αφ'(0) and φ'(α) ≥ c₂φ'(0)
    (eq. 3.6). Not converged after ``max_iter`` trials, at ``alpha_max``, or when the bracket
    collapses to rounding level.
    """
    return _demo(
        "weak_wolfe",
        problem,
        x0,
        direction,
        alpha0=alpha0,
        c1=c1,
        c2=c2,
        max_iter=max_iter,
        alpha_max=alpha_max,
    )


@register(
    id="goldstein",
    family="line_search",
    name="Goldstein (bisection)",
    params=(P_DIRECTION, P_ALPHA0, P_C_GOLDSTEIN, P_MAX_ITER, P_ALPHA_MAX),
    needs=("f", "grad"),
    summary="Keep f(x+αp) between two lines through f(x): not too long, not too short.",
    references=(f"{_NW}, eq. (3.11)", "Goldstein (1965), SIAM J. Control 3"),
)
def goldstein(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    direction: str = "steepest",
    alpha0: float = 1.0,
    c1: float = C_GOLDSTEIN_DEFAULT,
    max_iter: int = 50,
    alpha_max: float = 1e3,
) -> Result:
    """Goldstein line search with a bisection/doubling bracket from x0 along p.

    ``c1`` is the Goldstein constant c ∈ (0, ½). Stops (converged) at the first trial with
    φ(0) + (1-c)αφ'(0) ≤ φ(α) ≤ φ(0) + cαφ'(0) (eq. 3.11). Not converged after ``max_iter``
    trials, at ``alpha_max``, or when the bracket collapses to rounding level.
    """
    return _demo(
        "goldstein",
        problem,
        x0,
        direction,
        alpha0=alpha0,
        c1=c1,
        max_iter=max_iter,
        alpha_max=alpha_max,
    )


@register(
    id="exact_quadratic",
    family="line_search",
    name="Exact step (quadratic model)",
    params=(P_DIRECTION, P_C1),
    needs=("f", "grad", "hess"),
    summary="Jump straight to the minimizer of the local quadratic model along p.",
    references=(f"{_NW}, eq. (5.6)",),
)
def exact_quadratic(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    direction: str = "steepest",
    c1: float = C1_DEFAULT,
) -> Result:
    """Exact step α = -∇f(x0)ᵀp / pᵀ∇²f(x0)p along p (N&W eq. 5.6).

    One trial at the exact minimizer α of the quadratic model, which is the exact line
    minimizer when f is quadratic. Converged when pᵀ∇²f(x0)p > 0 and f(x0 + αp) ≤ f(x0)
    (``conditions["decrease"]``). The demo has no bound on the rounding error of f
    (``search(..., f_err=0)``), so any rise of f is a failure; its message reports the
    gradient test |φ'(α)| ≤ 0.1|φ'(0)| (``conditions["strong_curvature"]``, one more ∇f
    evaluation), which tells a model step that misses the line minimizer from one that ∇f
    confirms. ``c1`` only feeds the reported (diagnostic) Armijo test.
    """
    return _demo("exact_quadratic", problem, x0, direction, c1=c1)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
#: Together they cover every phase and every zoom interpolation branch (the union of
#: ``info["interp"]`` is checked by tests/test_line_search_methods.py).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("backtracking", "rosenbrock", {"x0": [-1.2, 1.0]}),
    ("backtracking", "rosenbrock", {"x0": [0.0, 0.0], "direction": "newton"}),
    # expand only: alpha doubles from a short first step and Alg. 3.5 accepts
    ("strong_wolfe", "quadratic_ill", {"x0": [-2.0, 2.0], "alpha0": 1e-3}),
    # expand twice, then zoom(alpha_1, alpha_2) with an interior alpha_lo; trials by quadratic,
    # quadratic_clamped and bisection (the stall rule: the bracket shrank by less than 0.66)
    ("strong_wolfe", "himmelblau", {"x0": [0.0, 0.0], "alpha0": 0.1, "c2": 0.1}),
    # expand 9 times, phi' > 0 at the last step -> zoom(alpha_9, alpha_8) (alpha_lo > alpha_hi);
    # trials by cubic and cubic_clamped
    ("strong_wolfe", "ackley", {"x0": [2.6, -3.4], "alpha0": 1e-3, "c2": 0.1}),
    ("weak_wolfe", "rosenbrock", {"x0": [-1.2, 1.0]}),
    ("goldstein", "himmelblau", {"x0": [0.0, 0.0]}),
    ("exact_quadratic", "quadratic_ill", {"x0": [-2.0, 2.0]}),
]
