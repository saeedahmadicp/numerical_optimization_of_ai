"""Accelerated first-order methods: certified step-size schedules, OGM and restarted AGD.

Every method here minimizes an L-smooth convex f: ℝⁿ → ℝ using only f and ∇f. Four methods
are plain gradient descent with normalized steps h_t (or OGM's momentum), fixed before the
run and certified by a performance-estimation (PEP) analysis; the fifth is Nesterov's
accelerated gradient with the O(1/k²) schedule and adaptive restart.

    x_{t+1} = x_t − (h_t / L) ∇f(x_t),        t = 0, 1, 2, ...                     (GD)

``silver_gd``
    The convex silver step-size schedule (Altschuler & Parrilo, Math. Program. 2024, eq. 2.1):
    h_t = 1 + ρ^{ν(t+1)−1}, ρ = 1 + √2, ν = 2-adic valuation. Theorem 1.1 there: for
    n = 2^k − 1, f(x_n) − f* ≤ r_k L‖x₀ − x*‖² with r_k = 1/(1 + √(4ρ^{2k} − 3)) (eq. 1.4).
``silver_gd_strongly_convex``
    The κ-aware silver schedule (Altschuler & Parrilo, J. ACM 2025, §3, eqs. 3.1–3.9) for a
    μ-strongly convex f, κ = L/μ. Theorem 1.1 there: for n a power of 2,
    ‖x_n − x*‖² ≤ τ_n‖x₀ − x*‖², τ_n = ((1 − z_n)/(1 + z_n))² (eq. 3.9). One block of
    n = ``horizon`` steps is repeated, so ‖x_{mn} − x*‖² ≤ τ_n^m‖x₀ − x*‖².
``long_step_gd``
    Grimmer's periodic "straightforward" long-step patterns (SIAM J. Optim. 2024, Table 1 and
    Theorem 2.1): f(x_T) − f* ≤ L D²/(c·T) + O(1/T²), D the radius of the initial sublevel
    set. The O(1/T²) constant is not explicit, so there is no computable certificate.
``ogm``
    Kim & Fessler's optimized gradient method OGM1 (Math. Program. 2016, §7.1, bound
    eq. 6.17): f(x_N) − f* ≤ L‖x₀ − x*‖²/(2θ_N²) ≤ L‖x₀ − x*‖²/(N + 1)², which is tight.
``fista``
    FISTA with g ≡ 0, i.e. Nesterov's accelerated gradient with the t_k schedule (Beck &
    Teboulle 2009, eqs. 4.1–4.3; constant step or backtracking), with the adaptive restart
    tests of O'Donoghue & Candès (2015, §3.2): ``restart="function"`` or ``"gradient"``.

The methods and their verification come from two numopt research studies:
``research/certified-stepsize-schedules`` (silver, long steps, OGM) and
``research/restarted-accelerated-gradient`` (FISTA with restart).

Conventions shared by every method:

* **Start.** ``x0`` (or the problem's default). A bare callable ``f`` needs ``x0``; without
  a ``grad`` the gradient is formed by central differences (``numopt.core.diff.gradient``).
* **Smoothness constant L** (``silver_gd``, ``silver_gd_strongly_convex``, ``long_step_gd``,
  ``ogm``). ``L > 0`` is used as given. ``L = 0`` (the default) means "auto":
  ``problem.extra["L"]`` when the problem states a global constant, else, for a problem tagged
  ``"quadratic"`` with a Hessian, λ_max(∇²f(x₀)) (exact for a quadratic; one Hessian
  evaluation). Otherwise a ``ValueError`` asks for L: the certificates need a *global* L.
  ``silver_gd_strongly_convex`` resolves μ the same way (``extra["mu"]``, else λ_min(∇²f(x₀));
  one Hessian evaluation serves both). ``fista`` takes ``lr`` (= 1/L, or 1/L₀ with
  backtracking) instead.
* **Trace.** Step ``k = 0`` holds x₀; step ``k ≥ 1`` holds the iterate x_k produced by
  iteration k. ``Step.fun = f(x_k)``. Every update is written x_k = x_{k−1} + α_k p_k with
  ``Step.step_size = info["alpha"] = α_k`` and ``info["direction"] = p_k`` (``None`` at k = 0).
  ``Step.grad_norm = ‖∇f(x_k)‖₂`` for the schedule methods and OGM; ``None`` for ``fista``,
  which never evaluates ∇f at x_k (its gradient is taken at y_k: ``info["grad_norm_y"]``).
  ``n_iter == trace[-1].k``.
* **Stopping test (converged).** Schedules and OGM: ‖∇f(x_k)‖₂ ≤ ``gtol``, checked at x₀ and
  after every iteration. ``gtol = 0`` runs exactly ``max_iter`` iterations unless ∇f(x_k) = 0
  exactly (the norm is computed with scaling, so a tiny nonzero gradient never rounds to 0).
  ``fista``: ‖∇f(y_k)‖₂ ≤ ``gtol`` at the point y_k where the gradient was taken, checked after
  x_k is computed (then f(x_k) ≤ f(y_k) − ‖∇f(y_k)‖²/(2L_k)).
* **Failure (converged=False).** ``max_iter`` reached; *divergence*: f or ∇f not finite, or
  f(x_k) − f(x₀) > 10¹²·max(1, |f(x₀)|) (as in ``first_order``; the transient rises after a
  long step stay far below this when L is a valid global constant); ``fista``: backtracking
  finds no acceptable L within ``MAX_BACKTRACK`` trials; ``fista`` *stall*: x_k = x_{k−1} = y_{k+1}
  in floating point with ∇f(y_k) ≠ 0 (the next iteration would repeat this one exactly). A
  non-finite f or ∇f at x₀ also stops
  at once. Invalid parameters raise ``ValueError``.
* **Counts.** ``n_fev`` counts f evaluations (plus 2n per central-difference gradient when the
  problem has no ``grad``), ``n_gev`` counts gradient evaluations and ``n_hev`` Hessian
  evaluations (auto L or μ on a quadratic only). No update rule here uses f except
  ``fista``'s function restart and backtracking; f is otherwise evaluated for the trace.

Info keys (every method, every step):
    direction: [n] | null   p_k with x_k = x_{k−1} + alpha·p_k (null at k = 0).
    alpha: float | null     α_k (= Step.step_size; null at k = 0).

Additional info keys of ``silver_gd``, ``silver_gd_strongly_convex``, ``long_step_gd``, ``ogm``:
    grad: [n]               ∇f(x_k).
    h: float | null         the normalized step h_{k−1} = L·α_k (null at k = 0; 1 for ogm).
    checkpoint: bool        True at the iterates the method's theorem speaks about:
                            k = 2^j − 1 (silver_gd), k a multiple of the horizon
                            (silver_gd_strongly_convex), k a multiple of the pattern length
                            (long_step_gd), k = N = max_iter (ogm). False at k = 0.
    bound_f: float | null   at a certified checkpoint, the coefficient c of the proved bound
                            f(x_k) − f* ≤ c·L‖x₀ − x*‖² (r_j, τ_n^m/2, or 1/(2θ_N²)); null
                            elsewhere and always null for long_step_gd.
    silver_gd_strongly_convex only:
        bound_dist: float | null  τ_n^m at checkpoint k = m·n: ‖x_k − x*‖² ≤ τ_n^m‖x₀ − x*‖²
                            (null elsewhere).
    ogm only:
        y: [n]              the primary sequence y_k = x_{k−1} − ∇f(x_{k−1})/L (= x₀ at k = 0).
        theta: float        θ_k of the OGM1 recursion (θ₀ = 1).

Additional info keys of ``fista``:
    y: [n] | null           y_k, the point where ∇f was evaluated (x_k = y_k − ∇f(y_k)/L_k);
                            null at k = 0.
    grad_y: [n] | null      ∇f(y_k) (null at k = 0).
    grad_norm_y: float | null  ‖∇f(y_k)‖₂, the quantity of the stopping test (null at k = 0).
    beta: float | null      the momentum coefficient that formed y_k,
                            y_k = x_{k−1} + beta·(x_{k−1} − x_{k−2}); 0 after a restart and at
                            k = 1 (null at k = 0).
    t: float                t_{k+1}, the schedule value carried into the next iteration
                            (1 at k = 0 and after a restart).
    restarted: bool         True when the restart test fired at x_k (then y_{k+1} = x_k and
                            t_{k+1} = 1).
    L: float                L_k, the Lipschitz estimate used to produce x_k (1/lr at k = 0).
    trials: [[L, f]]        backtracking trials: each L̄ tried and f(y_k − ∇f(y_k)/L̄), in
                            order ([] with a constant step and at k = 0).

References:
    J. M. Altschuler and P. A. Parrilo, "Acceleration by stepsize hedging: Multi-step descent
    and the silver stepsize schedule", J. ACM 72(2), 2025. doi:10.1145/3708502.
    J. M. Altschuler and P. A. Parrilo, "Acceleration by stepsize hedging: Silver stepsize
    schedule for smooth convex optimization", Math. Program., 2024.
    doi:10.1007/s10107-024-02164-2.
    B. Grimmer, "Provably faster gradient descent via long steps", SIAM J. Optim. 34(3), 2024.
    D. Kim and J. A. Fessler, "Optimized first-order methods for smooth convex minimization",
    Math. Program. 159, 2016. doi:10.1007/s10107-015-0949-3.
    A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear
    inverse problems", SIAM J. Imaging Sci. 2(1), 2009.
    B. O'Donoghue and E. Candès, "Adaptive restart for accelerated gradient schemes",
    Found. Comput. Math. 15, 2015. doi:10.1007/s10208-013-9150-3.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, as_vector

Vector = NDArray[np.float64]
Fn = Callable[[Any], Any]

#: The silver ratio ρ = 1 + √2.
RHO = 1.0 + math.sqrt(2.0)
#: A rise f(x_k) − f(x₀) above F_DIVERGE·max(1, |f(x₀)|) counts as divergence (as in
#: ``numopt.unconstrained.first_order``).
F_DIVERGE = 1e12
#: Largest κ-aware silver horizon (block length) built in memory: 2²⁰ steps.
MAX_HORIZON = 2**20
#: Auto horizon: stop doubling once a doubling improves the per-step rate −log(τ_n)/n by less
#: than this fraction (the saturation regime of Part I, Theorem 4.1).
SATURATION_GAIN = 0.01
#: FISTA backtracking: at most this many trials L̄ = ηⁱL_{k−1} per iteration.
MAX_BACKTRACK = 60
#: FISTA backtracking: relative slack for rounding in the test f(p) ≤ Q_L(p, y).
# NOTE: Beck & Teboulle's test (eq. 2.9 with g ≡ 0) is exact. In floating point f(p) and f(y)
# carry rounding errors of order ε·(sum of |terms| of f), which exceed the decrease being
# tested once ‖p − y‖ is tiny; the exact test then rejects every L̄ and drives L_k → ∞. The
# test accepts a violation up to BT_RTOL·max(|f(y)|, |f(p)|), below what f can resolve.
BT_RTOL = 1e-12
#: Auto μ: λ_min(∇²f) ≤ EIG_RESIDUE·n·ε·λ_max is read as 0 (not strongly convex).
# NOTE: eigvalsh is backward stable, so each computed eigenvalue carries an absolute error of
# order n·ε·‖H‖₂ (Golub & Van Loan, 4th ed., §8.1.2; Higham, 2nd ed., §1.12). A singular
# PSD Hessian then returns λ_min ≈ ±1e-16·λ_max, and a positive residue was taken as μ with
# κ ≈ 1e16. The factor 64 is a safety margin over that bound; below it, μ has no correct digit.
EIG_RESIDUE = 64
_EPS = float(np.finfo(np.float64).eps)
#: Restart schemes of ``fista`` (O'Donoghue & Candès 2015, §3.2).
RESTARTS = ("none", "function", "gradient")


# --------------------------------------------------------------------------------------
# Schedules and their certified rates
# --------------------------------------------------------------------------------------


def two_adic_valuation(t: int) -> int:
    """ν(t): the exponent of the largest power of 2 that divides t ≥ 1 (ν(1) = 0, ν(12) = 2)."""
    if t < 1:
        raise ValueError(f"ν(t) needs t ≥ 1, got {t}")
    return (t & -t).bit_length() - 1


def silver_step(t: int) -> float:
    """The t-th convex silver step h_t = 1 + ρ^{ν(t+1)−1}, t = 0, 1, ... (Part II, eq. 2.1)."""
    return 1.0 + RHO ** (two_adic_valuation(t + 1) - 1)


def silver_schedule(n: int) -> Vector:
    """The first n convex silver steps [√2, 2, √2, 2 + √2, √2, 2, √2, 1 + ρ², ...]."""
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    return np.array([silver_step(t) for t in range(n)], dtype=np.float64)


def silver_rate(k: int) -> float:
    """r_k = 1/(1 + √(4ρ^{2k} − 3)) (Part II, eq. 1.4).

    f(x_n) − f* ≤ r_k L‖x₀ − x*‖² for n = 2^k − 1 (Part II, Theorem 1.1); r₀ = 1/2.
    """
    if k < 0:
        raise ValueError(f"k must be ≥ 0, got {k}")
    return 1.0 / (1.0 + math.sqrt(4.0 * RHO ** (2 * k) - 3.0))


def _psi(t: float, kappa: float) -> float:
    """ψ(t) = (1 + κt)/(1 + t), the map from Part I's normalized to actual steps (eq. 3.5)."""
    return (1.0 + kappa * t) / (1.0 + t)


def _log_tau(log_w: float) -> float:
    """log τ = 2 log(w/(2 − w)) from log w, w = 1 − z (−∞ when w = 0).

    # NOTE: computed as 2(log w − log1p(u)) with u = 1 − w = −expm1(log w), since 2 − w = 1 + u.
    # The direct form log(2 − exp(log w)) rounds exp(log w) to 1 once u < ε (κ ≳ 1e15), which
    # drops the −log(1 + u) term: log τ₁ came out as −2/κ instead of −4/κ.
    """
    if log_w == -math.inf:
        return -math.inf
    return 2.0 * (log_w - math.log1p(-math.expm1(log_w)))


def _silver_sc_levels(kappa: float) -> Iterator[tuple[float, float, float]]:
    """Yield (a_n, b_n, log τ_n) for n = 1, 2, 4, 8, ... (Part I, eqs. 3.1–3.4 and 3.9).

    y₁ = z₁ = 1/κ; for n ≥ 2, with ξ = 1 − z_{n/2} and s = ξ + √(1 + ξ²) (the solution of
    eqs. 3.1–3.2), y_n = z_{n/2}/s and z_n = z_{n/2}·s; a_n = ψ(y_n), b_n = ψ(z_n);
    τ_n = ((1 − z_n)/(1 + z_n))².
    """
    z = 1.0 / kappa
    # log w₁ = log(1 − 1/κ). NOTE: for κ < 2, κ − 1 is exact (Sterbenz) and log1p(−fl(1/κ))
    # would lose log10(1/(κ − 1)) digits to the rounding of 1/κ; for κ ≥ 2, log1p is exact
    # to ε and log(κ − 1) − log(κ) would cancel.
    if kappa == 1.0:
        log_w = -math.inf
    elif kappa < 2.0:
        log_w = math.log((kappa - 1.0) / kappa)
    else:
        log_w = math.log1p(-1.0 / kappa)
    b = _psi(z, kappa)
    yield b, b, _log_tau(log_w)
    while True:
        # NOTE: 1 − z_n is never formed by subtraction: z_n → 1 doubly exponentially, so
        # 1 − z_n would cancel to 0 long before τ_n underflows. The recursion runs on
        # w = 1 − z in log form with the exact identity
        # 1 − (1 − w)(w + r) = w²(w + r)/(1 + r), r = √(1 + w²), and
        # (w + r)/(1 + r) = 1 − u/(1 + r) with u = 1 − w = −expm1(log w). The log1p form has
        # no cancellation for any κ; log(s) − log1p(r) subtracted two numbers ≈ 0.88 to get
        # one of size 1/κ and lost every digit for κ ≳ 1e15 (the tests compare it with
        # 600-digit arithmetic up to κ = 1e18).
        w = math.exp(log_w)  # may underflow to 0 near convergence; then s = 1 exactly
        u = -math.expm1(log_w)  # = 1 − w with full relative accuracy
        r = math.hypot(1.0, w)
        s = w + r
        y = z / s
        log_w = 2.0 * log_w + math.log1p(-u / (1.0 + r))
        # z_n = z_{n/2}·s = 1 − w_n; near 1 the second form is exact to ε/2 (w_n is accurate),
        # while the product accumulates one rounding per level.
        w_new = math.exp(log_w)
        z = 1.0 - w_new if w_new < 0.5 else z * s
        yield _psi(y, kappa), _psi(z, kappa), _log_tau(log_w)


def _check_kappa(kappa: float) -> None:
    if not (math.isfinite(kappa) and kappa >= 1.0):
        raise ValueError(f"κ = L/μ must be finite and ≥ 1, got {kappa!r}")


def _check_horizon(n: object) -> int:
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, float, np.integer, np.floating))
        or not math.isfinite(n)
        or n != math.floor(n)
    ):
        raise ValueError(f"the horizon must be a power of 2 in [1, {MAX_HORIZON}], got {n!r}")
    m = int(n)
    if m < 1 or m & (m - 1) or m > MAX_HORIZON:
        raise ValueError(f"the horizon must be a power of 2 in [1, {MAX_HORIZON}], got {n!r}")
    return m


def silver_sc_schedule(kappa: float, n: int) -> tuple[Vector, float]:
    """The κ-aware silver schedule h⁽ⁿ⁾ (n a power of 2) and its rate τ_n (Part I, §3).

    h⁽¹⁾ = [b₁] and h⁽ⁿ⁾ = [h̃⁽ⁿᐟ²⁾, a_n, h̃⁽ⁿᐟ²⁾, b_n] (eq. 3.8), where h̃ drops the last entry.
    The steps are normalized by 1/L (Part I sets M = 1, m = 1/κ). Returns ``(h, τ_n)`` with
    ‖x_n − x*‖² ≤ τ_n‖x₀ − x*‖² (Theorem 1.1); τ_n underflows to 0 for large n.
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
    """The smallest n = 2^j after which a doubling improves −log(τ_n)/n by < 1 % (≤ 2²⁰).

    # NOTE: Part I's Theorem 1.1 holds for any power-of-2 horizon and fixes no block to repeat;
    # Theorem 4.1 shows that the per-step rate saturates at n* ≍ κ^{log_ρ 2}. This rule (the
    # study's choice) picks the first horizon in that regime, so repeating it loses < 1 % of
    # the best per-step rate while the certified checkpoints stay frequent. It gives 64 for
    # κ = 100 and 32 for κ = 50.
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


def _repeat(*parts: float | tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for p in parts:
        out.extend(p if isinstance(p, tuple) else (p,))
    return tuple(out)


_B3 = (1.4, 2.0, 1.4)
_B7 = (1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4)
_S127 = (7.2, 12.6, 7.2, 23.5, 7.2, 12.6, 7.2, 370.0, 7.2, 12.6, 7.2, 23.5, 7.2, 12.6, 7.2)

#: Grimmer (2024), Table 1: the straightforward patterns (t = 2 uses η = 0.1 in (3 − η, 1.5)).
LONG_STEP_PATTERNS: dict[str, tuple[float, ...]] = {
    "2": (2.9, 1.5),
    "3": (1.5, 4.9, 1.5),
    "7": (1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5),
    "15": _repeat(_B3, 4.5, _B3, 29.7, _B3, 4.5, _B3),
    "31": _repeat(_B7, 8.2, _B7, 72.3, _B7, 8.2, _B7),
    "63": _repeat(_B7, 7.2, _B7, 14.2, _B7, 7.2, _B7, 164.0, _B7, 7.2, _B7, 14.2, _B7, 7.2, _B7),
    "127": _repeat(*[p for s in _S127 for p in (_B7, s)], _B7),
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
    """θ₀, ..., θ_N of OGM1 (Kim & Fessler 2016, §7.1).

    θ₀ = 1, θ_{i+1} = (1 + √(1 + 4θ_i²))/2 for i ≤ N − 2, θ_N = (1 + √(1 + 8θ_{N−1}²))/2.
    """
    if N < 1:
        raise ValueError(f"N must be ≥ 1, got {N}")
    th = np.empty(N + 1, dtype=np.float64)
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


def _vec(a: Vector) -> list[float]:
    return [float(v) for v in a]


def _norm(v: Vector) -> float:
    """‖v‖₂ with scaling (as in LAPACK's dnrm2): no overflow, and 0 only when v = 0.

    # NOTE: the plain √(vᵀv) underflows to 0 for ‖v‖ ≲ 1e-162; with gtol = 0 the stopping test
    # would then report convergence at a nonzero gradient.
    """
    m = float(np.max(np.abs(v))) if v.size else 0.0
    if m == 0.0 or not math.isfinite(m):
        return m
    u = v / m
    return m * math.sqrt(float(u @ u))


def _check_common(gtol: float, max_iter: object) -> int:
    """Validate gtol ≥ 0 and an integral max_iter ≥ 1 (``1e3`` from the CLI is accepted)."""
    if not (math.isfinite(gtol) and gtol >= 0.0):
        raise ValueError(f"gtol must be a finite number ≥ 0, got {gtol!r}")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, float, np.integer, np.floating))
        or not math.isfinite(max_iter)
        or max_iter != math.floor(max_iter)
        or max_iter < 1
    ):
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    return int(max_iter)


class _Oracle:
    """Counted f, ∇f (central differences when the problem has no gradient) and Hessian."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.f = Counted(problem.f)
        f_counted = self.f
        if problem.grad is not None:
            self.grad = Counted(problem.grad)
        else:
            self.grad = Counted(lambda z: diff.gradient(f_counted, z))
        self.n_hev = 0
        self._eigs: tuple[Vector, Vector] | None = None  # (x, λ(∇²f(x))) of the last call

    def value(self, x: Vector) -> float:
        with np.errstate(all="ignore"):
            return float(self.f(x))

    def gradient(self, x: Vector) -> Vector:
        with np.errstate(all="ignore"):
            return as_vector(self.grad(x))

    def hessian_eigs(self, x: Vector) -> Vector:
        assert self.problem.hess is not None
        # NOTE: L and μ come from the same spectrum; it is cached so that auto L and auto μ
        # cost one Hessian evaluation (and one eigvalsh), not two.
        if self._eigs is not None and np.array_equal(self._eigs[0], x):
            return self._eigs[1]
        self.n_hev += 1
        H = np.asarray(self.problem.hess(x), dtype=np.float64)
        lam = np.asarray(np.linalg.eigvalsh(0.5 * (H + H.T)), dtype=np.float64)
        self._eigs = (x.copy(), lam)
        return lam

    def result(
        self, method: str, trace: list[Step], converged: bool, message: str, extra: dict[str, Any]
    ) -> Result:
        last = trace[-1]
        return Result(
            method,
            np.array(last.x, dtype=np.float64),
            last.fun,
            converged,
            message,
            last.k,
            n_fev=self.f.n,
            n_gev=self.grad.n,
            n_hev=self.n_hev,
            trace=trace,
            extra=extra,
        )


def _constant(
    name: str, given: float, oracle: _Oracle, x0: Vector, which: int
) -> tuple[float, str]:
    """Resolve L (``which = −1``, the largest eigenvalue) or μ (``which = 0``, the smallest)."""
    if not (math.isfinite(given) and given >= 0.0):
        raise ValueError(f"{name} must be finite and ≥ 0 (0 = auto), got {given!r}")
    if given > 0.0:
        return float(given), "given"
    problem = oracle.problem
    stated = problem.extra.get(name) if isinstance(problem.extra, Mapping) else None
    if isinstance(stated, (int, float)) and math.isfinite(stated) and stated > 0.0:
        return float(stated), f'problem.extra["{name}"]'
    if "quadratic" in problem.tags and problem.hess is not None:
        # NOTE: λ(∇²f(x₀)) is a global constant only because f is quadratic (constant Hessian).
        eigs = oracle.hessian_eigs(x0)
        lam = float(eigs[which])
        floor = EIG_RESIDUE * eigs.size * _EPS * float(np.max(np.abs(eigs))) if which == 0 else 0.0
        if lam > floor:
            return lam, "eigenvalue of the (constant) Hessian"
        if which == 0 and lam > 0.0:
            raise ValueError(
                f"{problem.id}: give {name} > 0 explicitly; λ_min(∇²f) = {lam:.3g} is at the "
                f"rounding level of eigvalsh (≤ {EIG_RESIDUE}·n·ε·λ_max = {floor:.3g}), so f is "
                f"not certifiably strongly convex"
            )
    raise ValueError(
        f"{problem.id}: give {name} > 0 explicitly; the certificates need a global constant "
        f"and the problem states none"
    )


def _status(k: int, fx: float, g: Vector, f0: float, gtol: float) -> tuple[bool, str] | None:
    """(converged, message) when the run must stop at iterate x_k, else None."""
    if not (math.isfinite(fx) and bool(np.all(np.isfinite(g)))):
        if k == 0:
            return False, "f(x0) or ∇f(x0) is not finite"
        return False, f"diverged: f(x) or ∇f(x) is not finite at iteration {k}"
    rise_max = F_DIVERGE * max(1.0, abs(f0))
    if fx - f0 > rise_max:
        return False, (
            f"diverged: f(x) − f(x0) = {fx - f0:.3g} > {F_DIVERGE:.0e}·max(1, |f(x0)|) "
            f"= {rise_max:.3g} at iteration {k} (L too small?)"
        )
    gn = _norm(g)
    if gn <= gtol:
        where = "at the start point" if k == 0 else f"after {k} iterations"
        return True, f"‖∇f(x)‖ = {gn:.3g} ≤ gtol {where}"
    return None


def _max_iter_message(max_iter: int, g: Vector) -> str:
    return f"reached max_iter={max_iter} (‖∇f(x)‖ = {_norm(g):.3g} > gtol)"


def _step(
    k: int,
    x: Vector,
    fx: float,
    g: Vector,
    alpha: float | None,
    p: Vector | None,
    h: float | None,
    checkpoint: bool,
    bound_f: float | None,
    **more: Any,
) -> Step:
    return Step(
        k,
        x.copy(),
        fx,
        grad_norm=_norm(g),
        step_size=alpha,
        info={
            "grad": _vec(g),
            "direction": None if p is None else _vec(p),
            "alpha": alpha,
            "h": h,
            "checkpoint": checkpoint,
            "bound_f": bound_f,
            **more,
        },
    )


def _schedule_gd(
    method: str,
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
        return oracle.result(method, trace, stop[0], stop[1], extra)
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
            return oracle.result(method, trace, stop[0], stop[1], extra)
    return oracle.result(method, trace, False, _max_iter_message(max_iter, g), extra)


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

P_L = ParamSpec(
    "L",
    0.0,
    min=0.0,
    max=1e6,
    help='Global Lipschitz constant of ∇f (0 = auto: problem.extra["L"] or the Hessian of a '
    "quadratic).",
)
# NOTE: the study's range for gtol is [0, 1e-2]; the UI range starts at 1e-14 because a
# logarithmic slider needs min > 0. The methods still accept gtol = 0 (run to max_iter).
P_GTOL = ParamSpec(
    "gtol", 1e-6, min=1e-14, max=1e-2, log=True, help="Stop when ‖∇f(x_k)‖₂ ≤ gtol (0 = never)."
)


def _p_max_iter(default: int, help: str = "Iteration limit.") -> ParamSpec:
    return ParamSpec("max_iter", default, kind="int", min=1, max=100_000, help=help)


# --------------------------------------------------------------------------------------
# Certified step-size schedules
# --------------------------------------------------------------------------------------


@register(
    id="silver_gd",
    family="unconstrained",
    name="Silver step-size gradient descent",
    params=(P_L, P_GTOL, _p_max_iter(1023, "Iterations; the certificate holds at k = 2ʲ − 1.")),
    needs=("f", "grad"),
    order="sublinear: f − f⋆ ≤ r_k L‖x₀ − x⋆‖² ≈ L‖x₀ − x⋆‖²/(2n^{1.2716}) at n = 2ᵏ − 1",
    summary="Gradient descent with a fixed, fractal schedule of long and short steps that is "
    "provably faster than any constant step on convex functions.",
    references=(
        "Altschuler & Parrilo (2024), Math. Program., eq. 2.1 (schedule), Theorem 1.1 and "
        "eq. 1.4 (bound)",
        "numopt study research/certified-stepsize-schedules",
    ),
)
def silver_gd(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    L: float = 0.0,
    gtol: float = 1e-6,
    max_iter: int = 1023,
) -> Result:
    """Gradient descent with the convex silver step-size schedule (Altschuler & Parrilo 2024).

    x_{t+1} = x_t − (h_t/L)∇f(x_t) with h_t = 1 + ρ^{ν(t+1)−1} (Part II, eq. 2.1), i.e.
    [√2, 2, √2, 2 + √2, √2, 2, √2, 1 + ρ², ...]. The schedule needs no horizon: the first
    2^k − 1 steps are the same for every longer run.

    Certificate (Part II, Theorem 1.1): for any L-smooth convex f and n = 2^k − 1,
    f(x_n) − f* ≤ r_k L‖x₀ − x*‖², r_k = 1/(1 + √(4ρ^{2k} − 3)) ≈ 1/(2n^{log₂ρ}),
    log₂ρ ≈ 1.2716. ``info["bound_f"] = r_k`` at those k. Between checkpoints f is not
    monotone: the step 1 + ρ^{k−1} taken right after x_{2^k−1} can raise f a lot.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. The schedule ignores strong convexity (its
    spikes grow without bound), so on a strongly convex f the rate stays polynomial. With the
    defaults it does NOT reach gtol = 1e-6 within 1023 iterations on ``quadratic_bowl``
    (κ = 2: f(x₁₀₂₃) = 5.5e-9; f(x_{2^k−1}) falls by ≈ ρ² ≈ 5.8 per doubling of n). Use
    ``silver_gd_strongly_convex`` when μ > 0 is known.
    """
    max_iter = _check_common(gtol, max_iter)
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, oracle, x, -1)

    def marks(k: int) -> dict[str, Any]:
        j = (k + 1).bit_length() - 1
        cp = k >= 1 and (k + 1) == 1 << j
        return {"checkpoint": cp, "bound_f": silver_rate(j) if cp else None}

    extra = {"L": L, "L_source": src}
    return _schedule_gd("silver_gd", oracle, x, L, silver_step, marks, gtol, max_iter, extra)


@register(
    id="silver_gd_strongly_convex",
    family="unconstrained",
    name="Silver step-size gradient descent (strongly convex)",
    params=(
        P_L,
        ParamSpec(
            "mu",
            0.0,
            min=0.0,
            max=1e6,
            help='Strong-convexity constant μ ≤ L (0 = auto: problem.extra["mu"] or the '
            "Hessian of a quadratic).",
        ),
        ParamSpec(
            "horizon",
            0,
            kind="int",
            min=0,
            max=4096,
            help="Block length n to repeat: a power of 2 (0 = auto, near saturation of the rate).",
        ),
        P_GTOL,
        _p_max_iter(1024),
    ),
    needs=("f", "grad"),
    order="linear: ‖x − x⋆‖² shrinks by τ_n per block of n steps, O(κ^{0.7864} log(1/ε)) steps",
    summary="Gradient descent with a κ-aware silver schedule: certified linear convergence "
    "faster than any constant step on strongly convex functions.",
    references=(
        "Altschuler & Parrilo (2025), J. ACM 72(2), §3, eqs. 3.1–3.9 (schedule) and "
        "Theorem 1.1 (rate)",
        "numopt study research/certified-stepsize-schedules",
    ),
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
    """Gradient descent with the κ-aware silver step-size schedule (Altschuler & Parrilo, JACM).

    With κ = L/μ, the n-step schedule h⁽ⁿ⁾ (n a power of 2) comes from Part I, eqs. 3.1–3.8:
    y₁ = z₁ = 1/κ; y_n z_n = z_{n/2}², z_n − y_n = 2(z_{n/2} − z_{n/2}²); a_n = ψ(y_n),
    b_n = ψ(z_n), ψ(t) = (1 + κt)/(1 + t); h⁽ⁿ⁾ = [h̃⁽ⁿᐟ²⁾, a_n, h̃⁽ⁿᐟ²⁾, b_n]. Every step lies
    in (1, (κ + 1)/2], so none is longer than 1/μ.

    # NOTE: arXiv v1 of Part I (the only public version) states 1/κ ≤ y_n (Lemma 3.1), hence
    # a_n ≥ 2κ/(κ + 1) (Lemma 3.2), and a₂ = κ/(κ − 1) (Remark 3.3). These do not follow from
    # eqs. 3.1–3.2 as printed, which give y₂ < 1/κ (κ = 10: a₂ = 1.384 < 1.818). We implement
    # eqs. 3.1–3.2: the study's exact PEP worst case of ‖x_n − x*‖² equals τ_n of eq. 3.9 for
    # κ ∈ {4, 16, 100}, n ≤ 16, and as κ → ∞ the steps tend to the convex silver schedule
    # (Part II, Remark 2.2).

    The run repeats h⁽ⁿ⁾ with n = ``horizon`` (0 = :func:`silver_sc_auto_horizon`).
    Certificate (Part I, Theorem 1.1, applied once per block): for any L-smooth, μ-strongly
    convex f, ‖x_{mn} − x*‖² ≤ τ_n^m‖x₀ − x*‖², τ_n = ((1 − z_n)/(1 + z_n))² (eq. 3.9), hence
    f(x_{mn}) − f* ≤ (τ_n^m/2) L‖x₀ − x*‖².
    # NOTE: repeating one block is the study's choice; the paper states the rate for one
    # horizon, and repeating it composes the contraction.

    ``mu > L`` raises ``ValueError`` (κ < 1 is impossible). Stops (converged) when
    ‖∇f(x_k)‖₂ ≤ ``gtol``. With the defaults it converges on ``quadratic_bowl`` (15
    iterations), ``quadratic_ill`` (239) and ``quadratic_nd`` (367, auto horizon 64).
    """
    max_iter = _check_common(gtol, max_iter)
    if not (
        not isinstance(horizon, bool)
        and isinstance(horizon, (int, float, np.integer, np.floating))
        and horizon == 0
    ):
        _check_horizon(horizon)
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    oracle = _Oracle(prob)
    L, src_L = _constant("L", L, oracle, x, -1)
    mu, src_mu = _constant("mu", mu, oracle, x, 0)
    if mu > L:
        raise ValueError(f"μ = {mu:g} exceeds L = {L:g}")
    kappa = L / mu
    n = silver_sc_auto_horizon(kappa) if horizon == 0 else _check_horizon(horizon)
    h, tau = silver_sc_schedule(kappa, n)

    def marks(k: int) -> dict[str, Any]:
        cp = k >= 1 and k % n == 0
        bd = tau ** (k // n) if cp else None
        return {"checkpoint": cp, "bound_f": None if bd is None else 0.5 * bd, "bound_dist": bd}

    extra = {
        "L": L,
        "L_source": src_L,
        "mu": mu,
        "mu_source": src_mu,
        "kappa": kappa,
        "horizon": n,
        "tau_horizon": tau,
    }
    return _schedule_gd(
        "silver_gd_strongly_convex",
        oracle,
        x,
        L,
        lambda t: float(h[t % n]),
        marks,
        gtol,
        max_iter,
        extra,
    )


@register(
    id="long_step_gd",
    family="unconstrained",
    name="Long-step gradient descent",
    params=(
        P_L,
        ParamSpec(
            "pattern",
            "7",
            kind="choice",
            choices=tuple(LONG_STEP_PATTERNS),
            help="Length t of Grimmer's straightforward step pattern (Table 1), cycled.",
        ),
        P_GTOL,
        _p_max_iter(1000),
    ),
    needs=("f", "grad"),
    order="sublinear: f − f⋆ ≤ L D²/(avg(h)·T) + O(1/T²), avg(h) up to 5.83 (pattern 127)",
    summary="Gradient descent that cycles a short step pattern with one very long step; its "
    "proved O(1/T) constant grows with the average step.",
    references=(
        "Grimmer (2024), SIAM J. Optim. 34(3), eq. 1.3 (method), Table 1 (patterns), "
        "Theorem 2.1 and eq. 1.4 (rate)",
        "numopt study research/certified-stepsize-schedules",
    ),
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
    """Gradient descent that cycles one of Grimmer's long-step patterns (SIAM J. Optim. 2024).

    x_{k+1} = x_k − (h_{k mod t}/L)∇f(x_k) (Grimmer, eq. 1.3) with the straightforward pattern
    h of length t = ``pattern`` from Table 1, e.g. t = 2: (2.9, 1.5); t = 7: (1.5, 2.2, 1.5,
    12.0, 1.5, 2.2, 1.5). Theorem 2.1 with Table 1: f(x_T) − f* ≤ L D²/(c·T) + O(1/T²),
    D = sup{‖x − x*‖ : f(x) ≤ f(x₀)}, c ≈ avg(h) (``LONG_STEP_RATES``). The O(1/T²) term has no
    explicit constant, so ``bound_f`` is always null; ``checkpoint`` marks T = multiples of t.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. The patterns are tuned for the convex worst
    case, not for curvature equal to L: along an eigenvector of ∇²f with eigenvalue L, one
    period of pattern 7 contracts the distance only by Π|1 − h_i| = 0.99. With the defaults it
    does NOT converge on ``quadratic_bowl`` (f = 5.6e-2 after 1000 iterations).
    """
    max_iter = _check_common(gtol, max_iter)
    if pattern not in LONG_STEP_PATTERNS:
        raise ValueError(f"pattern must be one of {tuple(LONG_STEP_PATTERNS)}, got {pattern!r}")
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, oracle, x, -1)
    h = LONG_STEP_PATTERNS[pattern]
    t = len(h)

    def marks(k: int) -> dict[str, Any]:
        return {"checkpoint": k >= 1 and k % t == 0, "bound_f": None}

    extra = {
        "L": L,
        "L_source": src,
        "pattern": list(h),
        "avg_h": float(np.mean(h)),
        "rate_c": LONG_STEP_RATES[pattern],
    }
    return _schedule_gd(
        "long_step_gd", oracle, x, L, lambda i: h[i % t], marks, gtol, max_iter, extra
    )


@register(
    id="ogm",
    family="unconstrained",
    name="Optimized gradient method (OGM)",
    params=(P_L, P_GTOL, _p_max_iter(1000, "The horizon N (the last step uses a modified θ_N).")),
    needs=("f", "grad"),
    order="sublinear: f(x_N) − f⋆ ≤ L‖x₀ − x⋆‖²/(2θ_N²) ≤ L‖x₀ − x⋆‖²/(N + 1)², tight",
    summary="An accelerated gradient method whose worst-case bound is half of Nesterov's: the "
    "best possible for a fixed number of gradient steps.",
    references=(
        "Kim & Fessler (2016), Math. Program. 159, §7.1 (Algorithm OGM1), Theorem 2 and "
        "eq. 6.17 (bound), Theorem 3 (tightness)",
        "numopt study research/certified-stepsize-schedules",
    ),
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
    half of Nesterov's bound, and attained (Theorem 3). ``bound_f`` is set at k = N only, as
    the last step uses the modified θ_N; a run stopped earlier by ``gtol`` has no certificate.
    The trace holds the secondary sequence x_k (where ∇f is taken); ``info["y"]`` holds y_k.
    ``alpha`` = 1/L and ``direction`` = L(x_k − x_{k−1}).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. OGM1 has no linear rate on strongly convex
    f: with the defaults it does NOT reach gtol = 1e-6 on ``quadratic_bowl`` (f = 5.0e-7 at
    N = 1000).
    """
    N = _check_common(gtol, max_iter)
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    oracle = _Oracle(prob)
    L, src = _constant("L", L, oracle, x, -1)
    theta = ogm_thetas(N)
    bound_N = 0.5 / float(theta[-1]) ** 2
    y = x.copy()
    fx = oracle.value(x)
    g = oracle.gradient(x)
    f0 = fx
    extra: dict[str, Any] = {"L": L, "L_source": src, "theta_N": float(theta[-1])}
    trace = [_step(0, x, fx, g, None, None, None, False, None, y=_vec(y), theta=1.0)]
    stop = _status(0, fx, g, f0, gtol)
    if stop is not None:
        return oracle.result("ogm", trace, stop[0], stop[1], extra)
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
                bound_N if last else None,
                y=_vec(y),
                theta=t1,
            )
        )
        stop = _status(k, fx, g, f0, gtol)
        if stop is not None:
            return oracle.result("ogm", trace, stop[0], stop[1], extra)
    return oracle.result("ogm", trace, False, _max_iter_message(N, g), extra)


# --------------------------------------------------------------------------------------
# FISTA / accelerated gradient with adaptive restart
# --------------------------------------------------------------------------------------


def _backtrack(
    oracle: _Oracle, y: Vector, gy: Vector, fy: float, L: float, eta: float
) -> tuple[bool, Vector, float, float, list[list[float]]]:
    """Beck–Teboulle backtracking (§3, eq. 2.9 with g ≡ 0) from the estimate L.

    Tries L̄ = ηⁱL, i = 0, 1, ..., until p = y − ∇f(y)/L̄ satisfies
    f(p) ≤ Q_L̄(p, y) = f(y) + ⟨p − y, ∇f(y)⟩ + (L̄/2)‖p − y‖² (up to the ``BT_RTOL`` slack).
    Returns (found, p, f(p), L̄, trials) with trials = [[L̄, f(p)], ...] in order.
    """
    trials: list[list[float]] = []
    p, fp = y, math.nan
    for _ in range(MAX_BACKTRACK):
        p = y - gy / L
        fp = oracle.value(p)
        trials.append([L, fp])
        d = p - y
        q = fy + float(d @ gy) + 0.5 * L * float(d @ d)
        if math.isfinite(fp) and fp <= q + BT_RTOL * max(abs(fy), abs(fp)):
            return True, p, fp, L, trials
        L *= eta
    return False, p, fp, L, trials


@register(
    id="fista",
    family="unconstrained",
    name="Accelerated gradient with adaptive restart (FISTA)",
    params=(
        ParamSpec(
            "restart",
            "gradient",
            kind="choice",
            choices=RESTARTS,
            help="Adaptive restart test (O'Donoghue & Candès 2015, §3.2): none = plain FISTA; "
            "function: f(x_k) > f(x_{k−1}); gradient: ∇f(y_k)ᵀ(x_k − x_{k−1}) > 0.",
        ),
        ParamSpec(
            "lr",
            1.0,
            min=1e-8,
            max=1e4,
            log=True,
            help="Step 1/L. With backtracking, the first estimate L₀ = 1/lr (it only grows, so "
            "choose L₀ ≤ L).",
        ),
        ParamSpec(
            "backtracking",
            True,
            kind="bool",
            help="Beck–Teboulle backtracking: multiply L by η until f(p) ≤ f(y) − ‖∇f(y)‖²/(2L).",
        ),
        ParamSpec("eta", 2.0, min=1.1, max=10.0, help="Backtracking factor η > 1."),
        ParamSpec(
            "gtol",
            1e-6,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Stop when ‖∇f(y_k)‖₂ ≤ gtol (the gradient at the extrapolated point).",
        ),
        ParamSpec("max_iter", 5000, kind="int", min=1, max=1_000_000, help="Iteration limit."),
    ),
    needs=("f", "grad"),
    order="sublinear O(1/k²) on convex f; with restart, linear O(√κ log(1/ε)) on strongly "
    "convex quadratics without knowing μ",
    summary="Nesterov's accelerated gradient with growing momentum, reset whenever the "
    "momentum starts to work against the descent.",
    references=(
        "Beck & Teboulle (2009), SIAM J. Imaging Sci. 2(1), §4, eqs. 4.1–4.3 (FISTA), "
        "eq. 2.9 and §3 (backtracking), Theorem 4.4 (rate)",
        "O'Donoghue & Candès (2015), Found. Comput. Math. 15, §3.2 (restart tests), "
        "Algorithm 3 (reset), §4.5 (restart interval)",
        "numopt study research/restarted-accelerated-gradient",
    ),
)
def fista(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    restart: str = "gradient",
    lr: float = 1.0,
    backtracking: bool = True,
    eta: float = 2.0,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """FISTA with g ≡ 0 (Nesterov's accelerated gradient) and optional adaptive restart.

    Beck & Teboulle (2009), §4, with the smooth part only: y₁ = x₀, t₁ = 1 and for k ≥ 1

        x_k     = y_k − ∇f(y_k)/L_k                                                   (4.1)
        t_{k+1} = (1 + √(1 + 4t_k²))/2                                                (4.2)
        y_{k+1} = x_k + ((t_k − 1)/t_{k+1})(x_k − x_{k−1})                            (4.3)

    This is O'Donoghue & Candès' Algorithm 1 with q = 0 (written there with θ_k = 1/t_k).

    ``backtracking=False``: L_k ≡ 1/lr, which must be ≥ L(f) for the guarantee
    f(x_k) − f* ≤ 2L_k‖x₀ − x*‖²/(k + 1)² (Theorem 4.4, ``restart="none"``).
    ``backtracking=True``: L_k = ηⁱL_{k−1} with the smallest i ≥ 0 such that
    f(p) ≤ f(y_k) + ⟨p − y_k, ∇f(y_k)⟩ + (L̄/2)‖p − y_k‖², p = y_k − ∇f(y_k)/L̄ (eq. 2.9 with
    g ≡ 0), L₀ = 1/lr. L_k never decreases and L_k ≤ max(L₀, ηL(f)) (Remark 3.2), so a start
    L₀ > L(f) is never corrected: choose L₀ ≤ L(f). The accepted L_k = ηʲL₀ can lie below L(f).

    Adaptive restart (O'Donoghue & Candès 2015, §3.2), tested after x_k is computed:

    * ``"function"``: restart when f(x_k) > f(x_{k−1});
    * ``"gradient"``: restart when ∇f(y_k)ᵀ(x_k − x_{k−1}) > 0, i.e. when the momentum step
      makes an acute angle with the gradient;
    * ``"none"``: plain FISTA.

    # NOTE: a restart sets y_{k+1} = x_k and t_{k+1} = 1: a fresh start from x_k, the reset of
    # O'Donoghue & Candès' Algorithm 3 (fixed restarting, line 3: x⁰ = y⁰ = x_k, θ₀ = 1)
    # applied when a §3.2 test fires. The next two steps then carry no momentum, as at the
    # start; setting only t_k = 1 would give one momentum-free step instead.
    # NOTE: the study's general (proximal) code measures stationarity with the gradient
    # mapping L_k‖y_k − x_k‖ and tests (y_k − x_k)ᵀ(x_k − x_{k−1}) > 0. With g ≡ 0 both equal
    # ‖∇f(y_k)‖ and ∇f(y_k)ᵀ(x_k − x_{k−1})/L_k in exact arithmetic; the package uses ∇f(y_k)
    # directly, because y_k − x_k loses digits once ‖∇f(y_k)‖/L_k ≲ ε‖y_k‖ and could then
    # round to 0 and report convergence at a nonzero gradient.

    Costs per iteration: one ∇f (at y_k) and one f (at x_k, for the trace and the function
    test); backtracking adds f(y_k) (unless y_k = x_{k−1}, whose value is known) and one f per
    extra trial. ``Result.extra`` holds ``n_restart`` and ``restarts`` (the iterations k at
    which a restart fired).

    Stops (converged) when ‖∇f(y_k)‖₂ ≤ ``gtol``; then f(x_k) ≤ f(y_k) − ‖∇f(y_k)‖²/(2L_k).
    """
    max_it = _check_common(gtol, max_iter)
    if not gtol > 0.0:
        raise ValueError(f"gtol must be a finite number > 0, got {gtol!r}")
    if restart not in RESTARTS:
        raise ValueError(f"restart must be one of {RESTARTS}, got {restart!r}")
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be a finite number > 0, got {lr!r}")
    if not (math.isfinite(eta) and eta > 1.0):
        raise ValueError(f"eta must be a finite number > 1, got {eta!r}")
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)  # x_{k−1} at the top of iteration k
    oracle = _Oracle(prob)
    restarts: list[int] = []

    def done(converged: bool, message: str) -> Result:
        extra = {"n_restart": len(restarts), "restarts": list(restarts)}
        return oracle.result("fista", trace, converged, message, extra)

    y = x.copy()  # y_k (y₁ = x₀)
    t = 1.0  # t_k (t₁ = 1)
    gn_last = math.inf
    beta = 0.0  # the coefficient that formed y_k
    L = 1.0 / lr
    fx = oracle.value(x)
    f0 = fx
    trace = [
        Step(
            0,
            x.copy(),
            fx,
            None,
            None,
            {
                "direction": None,
                "alpha": None,
                "y": None,
                "grad_y": None,
                "grad_norm_y": None,
                "beta": None,
                "t": t,
                "restarted": False,
                "L": L,
                "trials": [],
            },
        )
    ]
    if not math.isfinite(fx):
        return done(False, "f(x0) is not finite")

    for k in range(1, max_it + 1):
        gy = oracle.gradient(y)
        if not bool(np.all(np.isfinite(gy))):
            return done(False, f"diverged: ∇f is not finite at y_k (iteration {k})")
        if backtracking:
            # f(y_k) is already known when y_k = x_{k−1} (no momentum): reuse it.
            fy = fx if np.array_equal(y, x) else oracle.value(y)
            found, p, fp, L, trials = _backtrack(oracle, y, gy, fy, L, eta)
            if not found:
                return done(
                    False,
                    f"backtracking failed: no L̄ = ηⁱL_(k−1) with f(p) ≤ Q_L(p, y) in "
                    f"{MAX_BACKTRACK} trials (iteration {k}, last L̄ = {trials[-1][0]:.3e})",
                )
        else:
            trials = []
            p = y - gy / L
            fp = oracle.value(p)
        restarted = False
        if restart == "function":
            restarted = fp > fx
        elif restart == "gradient":
            restarted = float(gy @ (p - x)) > 0.0
        if restarted:
            t_next, beta_next, y_next = 1.0, 0.0, p.copy()
            restarts.append(k)
        else:
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))  # (4.2)
            beta_next = (t - 1.0) / t_next
            y_next = p + beta_next * (p - x)  # (4.3)
        alpha = 1.0 / L
        gn = _norm(gy)
        trace.append(
            Step(
                k,
                p.copy(),
                fp,
                None,
                alpha,
                {
                    "direction": _vec((p - x) / alpha),
                    "alpha": alpha,
                    "y": _vec(y),
                    "grad_y": _vec(gy),
                    "grad_norm_y": gn,
                    "beta": beta,
                    "t": t_next,
                    "restarted": restarted,
                    "L": L,
                    "trials": trials,
                },
            )
        )
        if not (math.isfinite(fp) and bool(np.all(np.isfinite(p)))):
            return done(False, f"diverged: f(x) is not finite at iteration {k}")
        rise_max = F_DIVERGE * max(1.0, abs(f0))
        if fp - f0 > rise_max:
            return done(
                False,
                f"diverged: f(x) − f(x0) = {fp - f0:.3g} > {F_DIVERGE:.0e}·max(1, |f(x0)|) "
                f"= {rise_max:.3g} at iteration {k} (step 1/L too long?)",
            )
        if gn <= gtol:
            return done(True, f"‖∇f(y_k)‖ = {gn:.3g} ≤ gtol after {k} iterations")
        if np.array_equal(p, x) and np.array_equal(y_next, x):
            # NOTE: not in the study's code (which runs on to max_iter here). x_k = x_{k−1} =
            # y_{k+1} with ∇f(y_k) ≠ 0 means ∇f(y_k)/L_k is below the rounding of y_k (with
            # backtracking: L_k grew until the step vanished at the precision floor of f).
            # Iteration k + 1 then repeats iteration k exactly, so the run can never progress.
            return done(
                False,
                f"stalled: x_k = x_(k−1) in floating point at iteration {k} "
                f"(step ‖∇f(y_k)‖/L_k = {gn / L:.3g}; gtol below the attainable precision?)",
            )
        x, fx = p, fp
        y, t, beta = y_next, t_next, beta_next
        gn_last = gn
    return done(False, f"reached max_iter={max_it} (‖∇f(y_k)‖ = {gn_last:.3g} > gtol)")


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("silver_gd", "quadratic_ill", {"max_iter": 255}),
    ("silver_gd_strongly_convex", "quadratic_ill", {}),
    ("long_step_gd", "booth", {"pattern": "15"}),
    ("ogm", "quadratic_ill", {"max_iter": 150}),
    ("fista", "quadratic_ill", {}),
    ("fista", "rosenbrock", {"restart": "function", "max_iter": 300}),
]
