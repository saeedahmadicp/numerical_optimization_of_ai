"""Gradient descent with certified step-size schedules: silver steps, long steps and OGM.

Every method here minimizes an L-smooth convex f: ℝⁿ → ℝ with a *fixed* schedule of
normalized steps h_t, chosen before the run and certified by a performance-estimation (PEP)
analysis. Plain gradient descent with such a schedule is

    x_{t+1} = x_t − (h_t / L) ∇f(x_t),        t = 0, 1, 2, ...                     (GD)

* ``silver_gd`` — the convex Silver Stepsize Schedule (Altschuler & Parrilo, Math. Program.
  2024, eq. 2.1): h_t = 1 + ρ^{ν(t+1)−1}, ρ = 1 + √2, ν = 2-adic valuation. Theorem 1.1 there:
  for n = 2^k − 1, f(x_n) − f* ≤ r_k L‖x₀ − x*‖² with r_k = 1/(1 + √(4ρ^{2k} − 3)) (eq. 1.4).
* ``silver_gd_strongly_convex`` — the κ-aware Silver Stepsize Schedule (Altschuler & Parrilo,
  J. ACM 2025, §3, eqs. 3.1–3.9) for μ-strongly convex f with κ = L/μ. Theorem 1.1 there:
  for n a power of 2, ‖x_n − x*‖² ≤ τ_n ‖x₀ − x*‖², τ_n = ((1 − z_n)/(1 + z_n))² (eq. 3.9).
  The run repeats the n-step schedule ("horizon"), so ‖x_{mn} − x*‖² ≤ τ_n^m ‖x₀ − x*‖².
* ``long_step_gd`` — Grimmer's periodic "straightforward" long-step patterns (Grimmer, SIAM
  J. Optim. 2024, Table 1 and Theorem 2.1): f(x_T) − f* ≤ L D²/(avg(h)·T) + O(1/T²), with D
  the radius of the initial sublevel set. The O(1/T²) constant is not explicit, so this
  method has checkpoints but no computable certificate.
* ``ogm`` — Kim & Fessler's optimized gradient method OGM1 (Math. Program. 2016, Algorithm
  OGM1 in §7.1, bound eq. 6.17): f(x_N) − f* ≤ L‖x₀ − x*‖²/(2θ_N²) ≤ L‖x₀ − x*‖²/(N + 1)²,
  attained exactly by the Huber function of their Theorem 3 (eq. 8.1).

Conventions shared by every method (they follow ``numopt.unconstrained.first_order``):

* **Smoothness constant L.** ``L > 0`` is used as given. ``L = 0`` (the default) means "auto":
  ``problem.extra["L"]`` when the problem states a global constant, else, for a problem
  tagged ``"quadratic"``, λ_max(∇²f(x₀)) (exact for a quadratic; one Hessian evaluation).
  Otherwise a ``ValueError`` asks for L: the certificates hold only for a *global* L.
  ``silver_gd_strongly_convex`` resolves μ the same way (``extra["mu"]``, else λ_min(∇²f(x₀))).
* **Trace.** Step ``k = 0`` holds x₀; step ``k ≥ 1`` holds x_k. ``Step.fun = f(x_k)``,
  ``Step.grad_norm = ‖∇f(x_k)‖₂`` and ``Step.step_size = α_k = h_{k−1}/L`` (``None`` at k = 0).
  Every update is written x_k = x_{k−1} + α_k p_k. ``n_iter == trace[-1].k``.
  f is evaluated only for the trace: no update rule here uses f.
* **Stopping test (converged).** ‖∇f(x_k)‖₂ ≤ ``gtol``, checked at x₀ and after every
  iteration. ``gtol = 0`` runs exactly ``max_iter`` iterations (unless ∇f(x_k) = 0 exactly).
* **Failure (converged=False).** ``max_iter`` reached; *divergence*: f(x_k) or ∇f(x_k) not
  finite, or f(x_k) − f(x₀) > 10¹²·max(1, |f(x₀)|) (as in ``first_order``; the transient
  rises after a long step stay far below this when L is a valid global constant).
* **Counts.** ``n_fev`` counts f evaluations (one per iterate, plus 2n per central-difference
  gradient when the problem has no ``grad``), ``n_gev`` counts gradient evaluations (analytic
  or finite-difference) and ``n_hev`` counts Hessian evaluations (only for L = 0 or μ = 0 on
  a quadratic; one evaluation serves both L and μ).

Info keys (every method, every step):
    grad: [n]               ∇f(x_k).
    direction: [n] | null   p_k with x_k = x_{k−1} + alpha·p_k (null at k = 0).
    alpha: float | null     α_k (= Step.step_size; null at k = 0).
    h: float | null         the normalized step h_{k−1} = L·α_k (null at k = 0).
    checkpoint: bool        True at the iterates the method's theorem speaks about:
                            k = 2^j − 1 (silver_gd), k a multiple of the horizon
                            (silver_gd_strongly_convex), k a multiple of the pattern length
                            (long_step_gd), k = N = max_iter (ogm). False at k = 0.
    bound_f: float | null   at a certified checkpoint, the coefficient c of the proved bound
                            f(x_k) − f* ≤ c·L‖x₀ − x*‖² (r_j, τ_n^m / 2, or 1/(2θ_N²)); null
                            elsewhere and always null for long_step_gd.
Additional info keys:
    silver_gd_strongly_convex:
        bound_dist: float | null  τ_n^m at checkpoint k = m·n: ‖x_k − x*‖² ≤ τ_n^m‖x₀ − x*‖².
    ogm:
        y: [n]              the primary sequence y_k = x_{k−1} − ∇f(x_{k−1})/L (= x₀ at k = 0).
        theta: float        θ_k of the OGM1 recursion.

References:
    J. M. Altschuler and P. A. Parrilo, "Acceleration by stepsize hedging: Multi-step descent
    and the silver stepsize schedule", J. ACM 72(2), 2025. doi:10.1145/3708502.
    J. M. Altschuler and P. A. Parrilo, "Acceleration by stepsize hedging: Silver stepsize
    schedule for smooth convex optimization", Math. Program., 2024.
    doi:10.1007/s10107-024-02164-2.
    B. Grimmer, "Provably faster gradient descent via long steps", SIAM J. Optim. 34(3), 2024.
    arXiv:2307.06324.
    D. Kim and J. A. Fessler, "Optimized first-order methods for smooth convex minimization",
    Math. Program. 159, 2016. doi:10.1007/s10107-015-0949-3.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt.core import diff
from numopt.core.counting import Counted, finite, start_point, vector_problem
from numopt.core.registry import ParamSpec
from numopt.core.types import Problem, Result, Step

Vector = NDArray[np.float64]
Fn = Callable[[Any], float]

#: The silver ratio ρ = 1 + √2.
RHO = 1.0 + math.sqrt(2.0)
#: The rise of f above f(x₀) that counts as divergence (relative to max(1, |f(x₀)|)).
F_DIVERGE = 1e12
#: Largest horizon / schedule length built in memory (2²⁰ steps).
MAX_HORIZON = 2**20
#: Auto horizon: stop doubling once a doubling improves the per-step rate −log(τ_n)/n by
#: less than this fraction (the saturation regime of Part I, Theorem 4.1).
SATURATION_GAIN = 0.01


# --------------------------------------------------------------------------------------
# Schedules and their certified rates
# --------------------------------------------------------------------------------------


def two_adic_valuation(t: int) -> int:
    """ν(t): the exponent of the largest power of 2 dividing t ≥ 1 (ν(1) = 0, ν(4) = 2)."""
    if t < 1:
        raise ValueError(f"ν(t) needs t ≥ 1, got {t}")
    return (t & -t).bit_length() - 1


def silver_step(t: int) -> float:
    """The t-th convex silver step α_t = 1 + ρ^{ν(t+1)−1}, t = 0, 1, ... (Part II, eq. 2.1)."""
    return 1.0 + RHO ** (two_adic_valuation(t + 1) - 1)


def silver_schedule(n: int) -> Vector:
    """The first n convex silver steps [√2, 2, √2, 2 + √2, √2, 2, √2, ...] (Part II, eq. 2.1)."""
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    return np.array([silver_step(t) for t in range(n)], dtype=np.float64)


def silver_rate(k: int) -> float:
    """r_k = 1/(1 + √(4ρ^{2k} − 3)) (Part II, eq. 1.4): f(x_n) − f* ≤ r_k L‖x₀ − x*‖², n = 2^k − 1."""
    if k < 0:
        raise ValueError(f"k must be ≥ 0, got {k}")
    return 1.0 / (1.0 + math.sqrt(4.0 * RHO ** (2 * k) - 3.0))


def _psi(t: float, kappa: float) -> float:
    """ψ(t) = (1 + κt)/(1 + t), the map from normalized to actual steps (Part I, eq. 3.5)."""
    return (1.0 + kappa * t) / (1.0 + t)


def _silver_sc_levels(kappa: float) -> Iterator[tuple[float, float, float]]:
    """Yield (a_n, b_n, log τ_n) for n = 1, 2, 4, 8, ... (Part I, eqs. 3.1–3.4 and 3.9).

    y₁ = z₁ = 1/κ; for n ≥ 2, with ξ = 1 − z_{n/2} and s = ξ + √(1 + ξ²) (eq. 3.2),
    y_n = z_{n/2}/s and z_n = z_{n/2}·s; a_n = ψ(y_n), b_n = ψ(z_n); τ_n = ((1 − z_n)/(1 + z_n))².

    ``# NOTE:`` 1 − z_n is NOT formed by subtraction: z_n → 1 doubly exponentially and
    1 − z_n would cancel to 0 long before τ_n underflows. The recursion is run on w = 1 − z
    in log form, w_n = w² s/(1 + √(1 + w²)) — the exact identity
    1 − (1 − w)(w + √(1 + w²)) = w² (w + √(1 + w²))/(1 + √(1 + w²)) — which has no
    cancellation (verified against 600-digit arithmetic in the tests).
    """
    z = 1.0 / kappa
    log_w = math.log1p(-1.0 / kappa) if kappa > 1.0 else -math.inf  # log(1 − z₁)
    b = _psi(z, kappa)
    yield b, b, _log_tau(log_w)
    while True:
        w = math.exp(log_w)  # may underflow to 0 near convergence; then s = 1 exactly
        s = w + math.hypot(1.0, w)
        y = z / s
        log_w = 2.0 * log_w + math.log(s) - math.log1p(math.hypot(1.0, w))
        # z_n = z_{n/2}·s = 1 − w_n; near 1 the second form is exact to ε/2 (w_n is accurate),
        # while the product accumulates one rounding per level.
        w_new = math.exp(log_w)
        z = 1.0 - w_new if w_new < 0.5 else z * s
        yield _psi(y, kappa), _psi(z, kappa), _log_tau(log_w)


def _log_tau(log_w: float) -> float:
    """log τ = 2 log(w/(2 − w)) from log w (−∞ when w = 0)."""
    if log_w == -math.inf:
        return -math.inf
    return 2.0 * (log_w - math.log(2.0 - math.exp(log_w)))


def _check_kappa(kappa: float) -> None:
    if not (math.isfinite(kappa) and kappa >= 1.0):
        raise ValueError(f"κ = L/μ must be finite and ≥ 1, got {kappa!r}")


def _check_horizon(n: int) -> int:
    n = int(n)
    if n < 1 or n & (n - 1) or n > MAX_HORIZON:
        raise ValueError(f"the horizon must be a power of 2 in [1, {MAX_HORIZON}], got {n}")
    return n


def silver_sc_schedule(kappa: float, n: int) -> tuple[Vector, float]:
    """The κ-aware silver schedule h⁽ⁿ⁾ (n a power of 2) and its rate τ_n (Part I, §3).

    h⁽¹⁾ = [b₁] and h⁽ⁿ⁾ = [h̃⁽ⁿᐟ²⁾, a_n, h̃⁽ⁿᐟ²⁾, b_n] (eq. 3.8), where h̃ drops the last
    entry. Steps are normalized by 1/L (Part I sets M = 1, m = 1/κ). Returns ``(h, τ_n)``
    with ‖x_n − x*‖² ≤ τ_n ‖x₀ − x*‖² (Theorem 1.1); τ_n underflows to 0 for large n.
    """
    _check_kappa(kappa)
    n = _check_horizon(n)
    levels = _silver_sc_levels(kappa)
    _, b, log_tau = next(levels)
    h: list[float] = [b]
    m = 1
    while m < n:
        a, b, log_tau = next(levels)
        body = h[:-1]
        h = [*body, a, *body, b]
        m *= 2
    return np.array(h, dtype=np.float64), math.exp(log_tau)


def silver_sc_rate(kappa: float, n: int) -> float:
    """τ_n of the κ-aware silver schedule (Part I, eq. 3.9), n a power of 2."""
    _check_kappa(kappa)
    n = _check_horizon(n)
    levels = _silver_sc_levels(kappa)
    _, _, log_tau = next(levels)
    for _ in range(n.bit_length() - 1):
        _, _, log_tau = next(levels)
    return math.exp(log_tau)


def silver_sc_auto_horizon(kappa: float) -> int:
    """The smallest n = 2^j after which doubling improves −log(τ_n)/n by < 1 %.

    ``# NOTE:`` Part I's Theorem 1.1 holds for any power-of-2 horizon and fixes no cycling
    block; Theorem 4.1 shows the per-step rate saturates at n* ≍ κ^{log_ρ 2}. This rule picks
    the first horizon in that saturation regime, so cycling it loses < 1 % of the best
    per-step rate while keeping certified checkpoints frequent.
    """
    _check_kappa(kappa)
    levels = _silver_sc_levels(kappa)
    _, _, log_tau = next(levels)
    n = 1
    while n < MAX_HORIZON:
        _, _, log_tau2 = next(levels)
        if log_tau == -math.inf:  # κ = 1: one step of 1/L is exact
            return n
        if log_tau2 / (2 * n) >= (1.0 + SATURATION_GAIN) * log_tau / n:
            return n
        log_tau, n = log_tau2, 2 * n
    return n


def _long(*parts: float | tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for p in parts:
        out.extend(p if isinstance(p, tuple) else (p,))
    return tuple(out)


_B3 = (1.4, 2.0, 1.4)
_B7 = (1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4)

#: Grimmer (2024), Table 1: straightforward patterns (t = 2 uses η = 0.1 in (3 − η, 1.5)).
LONG_STEP_PATTERNS: dict[str, tuple[float, ...]] = {
    "2": (2.9, 1.5),
    "3": (1.5, 4.9, 1.5),
    "7": (1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5),
    "15": _long(_B3, 4.5, _B3, 29.7, _B3, 4.5, _B3),
    "31": _long(_B7, 8.2, _B7, 72.3, _B7, 8.2, _B7),
    "63": _long(_B7, 7.2, _B7, 14.2, _B7, 7.2, _B7, 164.0, _B7, 7.2, _B7, 14.2, _B7, 7.2, _B7),
    "127": _long(
        *[
            p
            for s in (
                7.2,
                12.6,
                7.2,
                23.5,
                7.2,
                12.6,
                7.2,
                370.0,
                7.2,
                12.6,
                7.2,
                23.5,
                7.2,
                12.6,
                7.2,
            )
            for p in (_B7, s)
        ],
        _B7,
    ),
}

#: Grimmer (2024), Table 1: the proved rate f(x_T) − f* ≤ L D²/(c·T) + O(1/T²) has this c.
LONG_STEP_RATES: dict[str, float] = {
    "2": 2.2,
    "3": 2.6333333333333333,
    "7": 3.1999999,
    "15": 3.8599999,
    "31": 4.6032258,
    "63": 5.2253968,
    "127": 5.8346303,
}


def ogm_thetas(N: int) -> Vector:
    """θ₀, ..., θ_N of OGM1 (Kim & Fessler 2016, Algorithm OGM1).

    θ₀ = 1, θ_{i+1} = (1 + √(1 + 4θ_i²))/2 for i ≤ N − 2, θ_N = (1 + √(1 + 8θ_{N−1}²))/2.
    """
    if N < 1:
        raise ValueError(f"N must be ≥ 1, got {N}")
    th = np.empty(N + 1)
    th[0] = 1.0
    for i in range(N):
        c = 8.0 if i == N - 1 else 4.0
        th[i + 1] = 0.5 * (1.0 + math.sqrt(1.0 + c * th[i] ** 2))
    return th


def ogm_rate(N: int) -> float:
    """1/(2θ_N²): f(x_N) − f* ≤ L‖x₀ − x*‖²/(2θ_N²) for OGM1 (Kim & Fessler 2016, eq. 6.17)."""
    return 0.5 / float(ogm_thetas(N)[-1]) ** 2


# --------------------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------------------


class _Oracle:
    """Counted f and ∇f (central differences when the problem has no gradient)."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.f = Counted(problem.f)
        self.n_gev = 0
        self.n_hev = 0
        self._eigs: tuple[Vector, Vector] | None = None  # (x, λ(∇²f(x))) of the last call

    def value(self, x: Vector) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(self.f(x))

    def gradient(self, x: Vector) -> Vector:
        self.n_gev += 1
        with np.errstate(over="ignore", invalid="ignore"):
            if self.problem.grad is not None:
                return np.asarray(self.problem.grad(x), dtype=np.float64).reshape(-1)
            return np.asarray(diff.gradient(self.f, x), dtype=np.float64)

    def hessian_eigs(self, x: Vector) -> Vector:
        if self.problem.hess is None:
            raise ValueError(f"{self.problem.id}: no Hessian to compute L or μ from")
        # NOTE: L and μ come from the same spectrum; cache it so that auto L and auto μ
        # cost one Hessian evaluation (and one eigvalsh), not two.
        if self._eigs is not None and np.array_equal(self._eigs[0], x):
            return self._eigs[1]
        self.n_hev += 1
        H = np.asarray(self.problem.hess(x), dtype=np.float64)
        lam = np.linalg.eigvalsh(0.5 * (H + H.T))
        self._eigs = (np.array(x, copy=True), lam)
        return lam


def _constant(
    name: str, given: float, problem: Problem, oracle: _Oracle, x0: Vector, which: int
) -> tuple[float, str]:
    """Resolve L (which = −1, largest eigenvalue) or μ (which = 0, smallest)."""
    if not (math.isfinite(given) and given >= 0.0):
        raise ValueError(f"{name} must be finite and ≥ 0 (0 = auto), got {given!r}")
    if given > 0.0:
        return float(given), "given"
    stated = problem.extra.get(name) if isinstance(problem.extra, Mapping) else None
    if stated is not None and float(stated) > 0.0:
        return float(stated), f'problem.extra["{name}"]'
    if "quadratic" in problem.tags and problem.hess is not None:
        # NOTE: λ(∇²f(x₀)) is a global constant only because f is quadratic (constant Hessian).
        lam = float(oracle.hessian_eigs(x0)[which])
        if lam > 0.0:
            return lam, "eigenvalue of the (constant) Hessian"
    raise ValueError(
        f"{problem.id}: give {name} > 0 explicitly; the certificates need a global constant "
        f"and the problem states none"
    )


def _check_common(gtol: float, max_iter: int) -> int:
    if not (math.isfinite(gtol) and gtol >= 0.0):
        raise ValueError(f"gtol must be finite and ≥ 0, got {gtol!r}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be an integer ≥ 1, got {max_iter!r}")
    return int(max_iter)


def _setup(problem: Problem | Fn, x0: ArrayLike | None) -> tuple[Problem, Vector]:
    prob = vector_problem(problem, x0=x0)
    return prob, start_point(prob, x0)


def _result(
    method: str,
    trace: list[Step],
    oracle: _Oracle,
    converged: bool,
    message: str,
    extra: dict[str, Any],
) -> Result:
    last = trace[-1]
    return Result(
        method=method,
        x=np.array(last.x, dtype=np.float64),
        fun=last.fun,
        converged=converged,
        message=message,
        n_iter=last.k,
        n_fev=oracle.f.n,
        n_gev=oracle.n_gev,
        n_hev=oracle.n_hev,
        trace=trace,
        extra=extra,
    )


def _status(k: int, fx: float, g: Vector, f0: float, gtol: float) -> tuple[bool, str] | None:
    """(converged, message) when the run must stop at iterate k, else None."""
    if not finite(fx, g):
        return False, f"diverged: f or ∇f is not finite at iteration {k}"
    if fx - f0 > F_DIVERGE * max(1.0, abs(f0)):
        return False, f"diverged: f(x_{k}) − f(x₀) = {fx - f0:.3g} at iteration {k} (L too small?)"
    gn = float(np.linalg.norm(g))
    if gn <= gtol:
        return True, f"‖∇f‖ = {gn:.3g} ≤ gtol = {gtol:.3g} after {k} iterations"
    return None


def _step(
    k: int,
    x: Vector,
    fx: float,
    g: Vector,
    alpha: float | None,
    p: Vector | None,
    h: float | None,
    checkpoint: bool = False,
    bound_f: float | None = None,
    **more: Any,
) -> Step:
    return Step(
        k=k,
        x=x.copy(),
        fun=fx,
        grad_norm=float(np.linalg.norm(g)),
        step_size=alpha,
        info={
            "grad": g.copy(),
            "direction": None if p is None else p.copy(),
            "alpha": alpha,
            "h": h,
            "checkpoint": checkpoint,
            "bound_f": bound_f,
            **more,
        },
    )


def _schedule_gd(
    method: str,
    problem: Problem,
    oracle: _Oracle,
    x: Vector,
    L: float,
    step: Callable[[int], float],
    marks: Callable[[int], dict[str, Any]],
    gtol: float,
    max_iter: int,
    extra: dict[str, Any],
) -> Result:
    """Run (GD) with normalized steps ``step(t)``; ``marks(k)`` gives the checkpoint keys."""
    fx = oracle.value(x)
    g = oracle.gradient(x)
    f0 = fx
    trace = [_step(0, x, fx, g, None, None, None, **marks(0))]
    stop = _status(0, fx, g, f0, gtol)
    if stop is not None:
        return _result(method, trace, oracle, stop[0], stop[1], extra)
    for k in range(1, max_iter + 1):
        h = step(k - 1)
        alpha = h / L
        p = -g
        x = x + alpha * p
        fx = oracle.value(x)
        g = oracle.gradient(x)
        trace.append(_step(k, x, fx, g, alpha, p, h, **marks(k)))
        stop = _status(k, fx, g, f0, gtol)
        if stop is not None:
            return _result(method, trace, oracle, stop[0], stop[1], extra)
    return _result(
        method,
        trace,
        oracle,
        False,
        f"max_iter = {max_iter} reached (‖∇f‖ = {float(np.linalg.norm(g)):.3g} > gtol)",
        extra,
    )


# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------


def silver_gd(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    L: float = 0.0,
    gtol: float = 1e-6,
    max_iter: int = 1023,
) -> Result:
    """Gradient descent with the convex Silver Stepsize Schedule (Altschuler & Parrilo 2024).

    x_{t+1} = x_t − (h_t/L)∇f(x_t) with h_t = 1 + ρ^{ν(t+1)−1} (Part II, eq. 2.1), i.e.
    [√2, 2, √2, 2 + √2, √2, 2, √2, 1 + ρ², ...]. The schedule needs no horizon: the first
    2^k − 1 steps are the same for every longer run.

    Certificate (Part II, Theorem 1.1): for any L-smooth convex f and n = 2^k − 1,
    f(x_n) − f* ≤ r_k L‖x₀ − x*‖², r_k = 1/(1 + √(4ρ^{2k} − 3)) ≈ 1/(2 n^{log₂ρ}),
    log₂ρ ≈ 1.2716. ``info["bound_f"] = r_k`` at those k. Between checkpoints f is not
    monotone: the step 1 + ρ^{k−1} taken right after x_{2^k−1} can raise f a lot.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. The schedule ignores strong convexity: its
    spikes grow without bound, so on a strongly convex f the rate stays polynomial. With the
    defaults it does NOT reach gtol = 1e-6 within 1023 iterations on ``quadratic_bowl``
    (κ = 2: f(x₁₀₂₃) = 5.5e-9, ‖∇f‖ = 2.1e-4; f(x_{2^k−1}) falls by ≈ ρ² ≈ 5.8 per doubling
    of n) nor on ``quadratic_nd`` (κ = 100: f(x₁₀₂₃) = 3.4e-7). Use
    ``silver_gd_strongly_convex`` when μ > 0 is known.
    """
    max_iter = _check_common(gtol, max_iter)
    prob, x = _setup(problem, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, prob, oracle, x, -1)

    def marks(k: int) -> dict[str, Any]:
        j = (k + 1).bit_length() - 1
        cp = k >= 1 and (k + 1) == 1 << j
        return {"checkpoint": cp, "bound_f": silver_rate(j) if cp else None}

    return _schedule_gd(
        "silver_gd",
        prob,
        oracle,
        x,
        L,
        silver_step,
        marks,
        gtol,
        max_iter,
        {"L": L, "L_source": src},
    )


def silver_gd_strongly_convex(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    L: float = 0.0,
    mu: float = 0.0,
    horizon: int = 0,
    gtol: float = 1e-6,
    max_iter: int = 1024,
) -> Result:
    """Gradient descent with the κ-aware Silver Stepsize Schedule (Altschuler & Parrilo, JACM).

    With κ = L/μ, the n-step schedule h⁽ⁿ⁾ (n a power of 2) is built by the recursion of
    Part I, eqs. 3.1–3.8: y₁ = z₁ = 1/κ; y_n z_n = z_{n/2}², z_n − y_n = 2(z_{n/2} − z_{n/2}²);
    a_n = ψ(y_n), b_n = ψ(z_n), ψ(t) = (1 + κt)/(1 + t); h⁽ⁿ⁾ = [h̃⁽ⁿᐟ²⁾, a_n, h̃⁽ⁿᐟ²⁾, b_n].
    Every step lies in (1, (κ + 1)/2], so none is longer than 1/μ.

    ``# NOTE:`` arXiv v1 (the only public version) states 1/κ ≤ y_n (Lemma 3.1), hence
    a_n ≥ 2κ/(κ + 1) (Lemma 3.2), and a₂ = κ/(κ − 1) (Remark 3.3). These do not follow from
    eqs. 3.1–3.2 as printed, which give y₂ = (1/κ)/(ξ + √(1 + ξ²)) < 1/κ (e.g. κ = 10:
    a₂ = 1.384 < 2κ/(κ + 1) = 1.818). We implement eqs. 3.1–3.2. Evidence that this is the
    intended schedule: the exact worst case of ‖x_n − x*‖² (PEP SDP, pep.py) equals τ_n of
    eq. 3.9 for κ ∈ {4, 16, 100}, n ≤ 16, and as κ → ∞ the steps tend to the convex silver
    schedule (Part II, Remark 2.2), e.g. a₂ → 1 + 1/ρ = √2.

    The run repeats h⁽ⁿ⁾ with n = ``horizon`` (0 = :func:`silver_sc_auto_horizon`).
    Certificate (Part I, Theorem 1.1, applied once per block): for any L-smooth, μ-strongly
    convex f, ‖x_{mn} − x*‖² ≤ τ_n^m ‖x₀ − x*‖², τ_n = ((1 − z_n)/(1 + z_n))² (eq. 3.9);
    hence f(x_{mn}) − f* ≤ (τ_n^m/2) L‖x₀ − x*‖². ``# NOTE:`` cycling one block is our choice
    (the paper states the rate for one horizon; repeating it composes the contraction).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. With the defaults it converges on
    ``quadratic_bowl`` (15 iterations), ``quadratic_ill`` (239) and ``quadratic_nd`` (367,
    auto horizon 64).
    """
    max_iter = _check_common(gtol, max_iter)
    if int(horizon) != horizon or horizon < 0:
        raise ValueError(f"horizon must be 0 (auto) or a power of 2, got {horizon!r}")
    prob, x = _setup(problem, x0)
    oracle = _Oracle(prob)
    L, src_L = _constant("L", L, prob, oracle, x, -1)
    mu, src_mu = _constant("mu", mu, prob, oracle, x, 0)
    if mu > L:
        raise ValueError(f"μ = {mu:g} exceeds L = {L:g}")
    kappa = L / mu
    n = silver_sc_auto_horizon(kappa) if horizon == 0 else _check_horizon(int(horizon))
    h, tau = silver_sc_schedule(kappa, n)

    def marks(k: int) -> dict[str, Any]:
        cp = k >= 1 and k % n == 0
        bd = tau ** (k // n) if cp else None
        return {"checkpoint": cp, "bound_f": None if bd is None else 0.5 * bd, "bound_dist": bd}

    return _schedule_gd(
        "silver_gd_strongly_convex",
        prob,
        oracle,
        x,
        L,
        lambda t: float(h[t % n]),
        marks,
        gtol,
        max_iter,
        {
            "L": L,
            "L_source": src_L,
            "mu": mu,
            "mu_source": src_mu,
            "kappa": kappa,
            "horizon": n,
            "tau_horizon": tau,
        },
    )


def long_step_gd(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    L: float = 0.0,
    pattern: str = "7",
    gtol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Gradient descent cycling one of Grimmer's long-step patterns (SIAM J. Optim. 2024).

    x_{k+1} = x_k − (h_{k mod t}/L)∇f(x_k) (Grimmer, eq. 1.3) with the straightforward pattern
    h of length t = ``pattern`` from Table 1, e.g. t = 2: (2.9, 1.5); t = 7: (1.5, 2.2, 1.5,
    12.0, 1.5, 2.2, 1.5). Theorem 2.1 with Table 1: f(x_T) − f* ≤ L D²/(c·T) + O(1/T²),
    D = sup{‖x − x*‖ : f(x) ≤ f(x₀)}, c ≈ avg(h) (``LONG_STEP_RATES``). The O(1/T²) term has no
    explicit constant, so ``bound_f`` is always null; ``checkpoint`` marks T = multiples of t.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. The patterns are tuned for the convex worst
    case, not for curvature equal to L: on an eigenvector of ∇²f with eigenvalue L one period
    of pattern 7 contracts the distance only by Π|1 − h_i| = 0.99. With the defaults it does
    NOT converge on ``quadratic_bowl`` (f = 5.6e-2 after 1000 iterations).
    """
    max_iter = _check_common(gtol, max_iter)
    if pattern not in LONG_STEP_PATTERNS:
        raise ValueError(f"pattern must be one of {tuple(LONG_STEP_PATTERNS)}, got {pattern!r}")
    prob, x = _setup(problem, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, prob, oracle, x, -1)
    h = LONG_STEP_PATTERNS[pattern]
    t = len(h)

    def marks(k: int) -> dict[str, Any]:
        return {"checkpoint": k >= 1 and k % t == 0, "bound_f": None}

    return _schedule_gd(
        "long_step_gd",
        prob,
        oracle,
        x,
        L,
        lambda i: h[i % t],
        marks,
        gtol,
        max_iter,
        {
            "L": L,
            "L_source": src,
            "pattern": list(h),
            "avg_h": float(np.mean(h)),
            "rate_c": LONG_STEP_RATES[pattern],
        },
    )


def ogm(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    L: float = 0.0,
    gtol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Kim & Fessler's optimized gradient method OGM1 (Math. Program. 2016, §7.1).

    With N = ``max_iter``, y₀ = x₀ and θ₀ = 1, for i = 0, ..., N − 1:

        y_{i+1} = x_i − ∇f(x_i)/L,
        θ_{i+1} = (1 + √(1 + 4θ_i²))/2  (i ≤ N − 2),   (1 + √(1 + 8θ_i²))/2  (i = N − 1),
        x_{i+1} = y_{i+1} + ((θ_i − 1)/θ_{i+1})(y_{i+1} − y_i) + (θ_i/θ_{i+1})(y_{i+1} − x_i).

    Certificate (Theorem 2, eq. 6.17): f(x_N) − f* ≤ L‖x₀ − x*‖²/(2θ_N²) ≤ L‖x₀ − x*‖²/(N + 1)²,
    half of Nesterov's bound and tight (Theorem 3). ``bound_f`` is set at k = N only, as the
    last step uses the modified θ_N; a run stopped earlier by ``gtol`` has no certificate.
    The trace holds the secondary sequence x_k (where ∇f is taken); ``info["y"]`` holds y_k.
    ``alpha`` = 1/L and ``direction`` = L(x_k − x_{k−1}).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. OGM1 has no linear rate on strongly convex
    f: with the defaults it does NOT reach gtol = 1e-6 on ``quadratic_bowl`` (f = 5.0e-7,
    ‖∇f‖ = 2.0e-3 at N = 1000).
    """
    N = _check_common(gtol, max_iter)
    prob, x = _setup(problem, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, prob, oracle, x, -1)
    theta = ogm_thetas(N)
    y = x.copy()
    fx = oracle.value(x)
    g = oracle.gradient(x)
    f0 = fx
    extra: dict[str, Any] = {"L": L, "L_source": src, "theta_N": float(theta[-1])}
    trace = [_step(0, x, fx, g, None, None, None, y=y.copy(), theta=1.0)]
    stop = _status(0, fx, g, f0, gtol)
    if stop is not None:
        return _result("ogm", trace, oracle, stop[0], stop[1], extra)
    alpha = 1.0 / L
    for i in range(N):
        y_new = x - alpha * g
        t0, t1 = float(theta[i]), float(theta[i + 1])
        x_new = y_new + ((t0 - 1.0) / t1) * (y_new - y) + (t0 / t1) * (y_new - x)
        p = (x_new - x) / alpha
        x, y = x_new, y_new
        fx = oracle.value(x)
        g = oracle.gradient(x)
        k = i + 1
        last = k == N
        trace.append(
            _step(
                k,
                x,
                fx,
                g,
                alpha,
                p,
                1.0,
                last,
                ogm_rate(N) if last else None,
                y=y.copy(),
                theta=t1,
            )
        )
        stop = _status(k, fx, g, f0, gtol)
        if stop is not None:
            return _result("ogm", trace, oracle, stop[0], stop[1], extra)
    return _result(
        "ogm",
        trace,
        oracle,
        False,
        f"max_iter = N = {N} reached (‖∇f‖ = {float(np.linalg.norm(g)):.3g} > gtol)",
        extra,
    )


# --------------------------------------------------------------------------------------
# Registry metadata (for promotion into numopt.unconstrained)
# --------------------------------------------------------------------------------------

_L = ParamSpec(
    "L",
    0.0,
    min=0.0,
    max=1e6,
    help="global Lipschitz constant of ∇f (0 = auto: problem.extra or the Hessian of a quadratic)",
)
_GTOL = ParamSpec("gtol", 1e-6, min=0.0, max=1e-2, log=True, help="stop when ‖∇f‖ ≤ gtol")

PARAMS: dict[str, list[ParamSpec]] = {
    "silver_gd": [
        _L,
        _GTOL,
        ParamSpec(
            "max_iter",
            1023,
            kind="int",
            min=1,
            max=100_000,
            help="iterations; certified at 2^k − 1",
        ),
    ],
    "silver_gd_strongly_convex": [
        _L,
        ParamSpec(
            "mu",
            0.0,
            min=0.0,
            max=1e6,
            help="strong-convexity constant (0 = auto: problem.extra or the Hessian)",
        ),
        ParamSpec(
            "horizon",
            0,
            kind="int",
            min=0,
            max=MAX_HORIZON,
            help="power-of-2 block length to repeat (0 = auto, near saturation)",
        ),
        _GTOL,
        ParamSpec("max_iter", 1024, kind="int", min=1, max=100_000),
    ],
    "long_step_gd": [
        _L,
        ParamSpec(
            "pattern",
            "7",
            kind="choice",
            choices=tuple(LONG_STEP_PATTERNS),
            help="length of Grimmer's straightforward pattern (Table 1)",
        ),
        _GTOL,
        ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000),
    ],
    "ogm": [
        _L,
        _GTOL,
        ParamSpec(
            "max_iter",
            1000,
            kind="int",
            min=1,
            max=100_000,
            help="the horizon N (the last step uses the modified θ_N)",
        ),
    ],
}

METHODS: dict[str, Callable[..., Result]] = {
    "silver_gd": silver_gd,
    "silver_gd_strongly_convex": silver_gd_strongly_convex,
    "long_step_gd": long_step_gd,
    "ogm": ogm,
}

__all__ = [
    "LONG_STEP_PATTERNS",
    "LONG_STEP_RATES",
    "METHODS",
    "PARAMS",
    "RHO",
    "long_step_gd",
    "ogm",
    "ogm_rate",
    "ogm_thetas",
    "silver_gd",
    "silver_gd_strongly_convex",
    "silver_rate",
    "silver_sc_auto_horizon",
    "silver_sc_rate",
    "silver_sc_schedule",
    "silver_schedule",
    "silver_step",
    "two_adic_valuation",
]
