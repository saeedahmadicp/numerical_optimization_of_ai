"""Trust-region methods for min f(x), f: ℝⁿ → ℝ twice differentiable.

At the iterate x_k the methods minimize, approximately, the quadratic model

    m_k(p) = f_k + g_kᵀp + ½ pᵀB_k p     subject to ‖p‖₂ ≤ Δ_k,

with g_k = ∇f(x_k) and B_k = ∇²f(x_k) (which may be indefinite). They share the radius update of
Nocedal & Wright (2006), Alg. 4.1, and differ only in the subproblem solver:

========================  ============================================================
``trust_region_cauchy``    the Cauchy point p^C (N&W Alg. 4.2, eq. 4.11–4.12)
``trust_region_dogleg``    the dogleg step (N&W §4.1, eq. 4.15–4.16); B_k must be positive
                           definite, otherwise the Cauchy point is used (``note`` says so)
``trust_region_steihaug``  Steihaug–Toint truncated CG (N&W Alg. 7.2)
``trust_region_exact``     the global minimizer of the subproblem: Newton's method on the
                           secular equation (N&W Alg. 4.3, Moré & Sorensen 1983) with the
                           hard case (N&W eq. 4.45)
========================  ============================================================

Radius update and step acceptance (N&W Alg. 4.1), with the reduction ratio

    ρ_k = (f(x_k) − f(x_k + p_k)) / (m_k(0) − m_k(p_k)):

    ρ_k < ¼                         →  Δ_{k+1} = ¼ Δ_k
    ρ_k > ¾ and ‖p_k‖ = Δ_k          →  Δ_{k+1} = min(2Δ_k, Δ̂)
    otherwise                        →  Δ_{k+1} = Δ_k
    x_{k+1} = x_k + p_k if ρ_k > η, else x_{k+1} = x_k.

Parameters: Δ₀ = min(``radius0``, Δ̂), Δ̂ = ``max_radius``, η = ``eta`` ∈ [0, ¼) (default 0.15,
the value SciPy uses; N&W require η ∈ [0, ¼)). Alg. 4.1 takes Δ₀ ∈ (0, Δ̂); a larger ``radius0``
is clamped to Δ̂ and the ``note`` of Step 0 says so. "‖p_k‖ = Δ_k" is the solver's report that
p_k lies on the boundary (``hits_boundary``), not a floating-point equality test. A non-finite
f(x_k + p_k) counts as ρ_k = −∞ (rejected, radius shrinks).

Stopping test (converged): ‖g_k‖∞ ≤ ``gtol``, checked at every accepted iterate. Failures
(``converged=False``): non-finite f, ∇f or ∇²f at x₀ or at an accepted iterate, the radius
collapsing below ε·max(1, ‖x_k‖) (no step can be accepted), ``max_iter`` iterations, and two
floating-point limits that exact arithmetic does not have:

* ‖g_k‖₂² underflows (below the smallest normal float64 ≈ 2.2e-308, i.e. ‖g_k‖₂ ≲ 1.5e-154)
  while ‖g_k‖∞ > ``gtol``: the model's terms gᵀp and gᵀBg are then not representable (rescale
  f in that case);
* the solver's step predicts no decrease, m_k(0) − m_k(p_k) ≤ 0, which every solver excludes in
  exact arithmetic (each achieves at least the Cauchy decrease ½‖g‖min(Δ, ‖g‖/‖B‖) > 0, N&W
  Lemma 4.3): the model is at its rounding level and ρ would carry no information.

‖g‖₂ is computed with a power-of-2 scaling (exact; equal to ``np.linalg.norm`` whenever no square
underflows), and the solvers form gᵀBg from the scaled ĝ = 2⁻ᵉg, so they do not underflow
earlier than these tests.

Trace: one Step per iteration, accepted or rejected (k = 0 is the start). Step k holds
x_k (= x_{k−1} after a rejection), f(x_k), ‖g_k‖₂ and ``step_size`` = Δ_{k−1}, the radius of the
subproblem solved at this iteration. ``n_iter == trace[-1].k``.

Evaluation counts are exact: one f per iteration (at the trial point), one ∇f and one ∇²f per
accepted point (rejections reuse them). Without ``Problem.grad`` the gradient is formed by central
differences (``numopt.core.diff``; its 2n evaluations of f are included in ``n_fev``); without
``Problem.hess`` the Hessian is formed by central differences of ∇f (its 2n gradient evaluations
are included in ``n_gev``).

A :class:`Problem` with ``dim == 1`` (also the one built from a bare callable with a one-entry x0)
follows the scalar convention of ``core.types``: f, f′ and f″ receive and return floats. It is run
as an n = 1 problem, so ``Result.x``, the trace iterates and the vector info keys have length 1.

Info keys (every method, every step; vectors called "point" are absolute positions, ``step`` is
relative to ``center``):
    center: [n] — x_{k−1}, the point at which the model of this iteration was built (the center
        of the trust region); x₀ at k = 0.
    grad: [n] — ∇f(center), the model gradient g.
    H: [[2]] | None — the model Hessian B = ∇²f(center) when n = 2 (for drawing the model's
        contours), else None.
    radius: float — Δ used for this iteration's subproblem (Δ₀ at k = 0).
    new_radius: float — Δ after the update of Alg. 4.1 (Δ₀ at k = 0).
    step: [n] | None — the model step p (None at k = 0).
    step_norm: float | None — ‖p‖₂.
    hits_boundary: bool | None — the solver placed p on the boundary ‖p‖ = Δ.
    predicted: float | None — the predicted reduction m(0) − m(p) = −gᵀp − ½pᵀBp.
    actual: float | None — the actual reduction f(center) − f(center + p) (None if
        f(center + p) is not finite).
    rho: float | None — the reduction ratio ρ actually used (see the rounding note below; None
        when f(center + p) is not finite or at k = 0).
    accepted: bool | None — ρ > η, so x moved to center + p (None at k = 0).
    trial_point: [n] | None — center + p.
    cauchy_point: [n] | None — center + p^C, the Cauchy point (N&W eq. 4.11–4.12).
    newton_point: [n] | None — center − B⁻¹g when B is positive definite (Cholesky succeeds),
        else None. May lie outside the trust region.
    note: str | None — a remark on this step (e.g. the dogleg fallback to the Cauchy point); at
        k = 0, the clamp Δ₀ = Δ̂ when ``radius0`` > ``max_radius``.
Additional info keys per method (None at k = 0):
    trust_region_cauchy:
        tau: float — p^C = τ p^S with p^S = −Δ g/‖g‖ (N&W eq. 4.12).
    trust_region_dogleg:
        dogleg_path: [[n]×3] | None — the path vertices center, center + p^U, center + p^B with
            p^U = −(gᵀg/gᵀBg) g (N&W eq. 4.15); None when B is not positive definite.
        tau: float | None — the dogleg parameter τ ∈ [0, 2] of p = p̃(τ) (N&W eq. 4.16); None
            on the Cauchy-point fallback.
    trust_region_steihaug:
        cg_path: [[n]...] — center + z_j for the inner CG iterates z₀ = 0, z₁, …, ending at
            center + p.
        cg_iters: int — the number of inner CG iterations (products B d).
        termination: "residual" | "boundary" | "negative_curvature" | "max_iter" — why the inner
            CG stopped (N&W Alg. 7.2).
        cg_tol: float — the inner tolerance ε_k = min(½, √‖g‖)‖g‖ (N&W eq. 7.3 / Thm. 7.2).
    trust_region_exact:
        lambda: float | None — the multiplier λ ≥ 0 with (B + λI)p = −g, B + λI ⪰ 0 and
            λ(Δ − ‖p‖) = 0 (N&W Thm. 4.1); None when the secular equation is not solvable in
            float64 and the Cauchy point is used (``note`` says so).
        lambda_min: float — λ₁, the smallest eigenvalue of B.
        hard_case: bool — the (nearly) hard case of N&W §4.3: g ⊥ eigenspace of λ₁ (up to
            rounding) and λ = −λ₁, so p = p_⊥ + τu includes a multiple of an eigenvector u of
            λ₁ (N&W eq. 4.45).
        lambda_iters: int — Newton (or bisection) iterations on the secular equation.

Rounding note: near a minimizer the predicted reduction falls below the rounding error of f, and
ρ computed from differences of f is noise. Following the practice described by Conn, Gould &
Toint (2000), *Trust-Region Methods*, §17.4.2, ρ is computed as (actual + δ)/(predicted + δ) with
the relative allowance δ = 10³ ε max(|f(x_k)|, |f(x_k + p_k)|) (10³ is the default of Manopt's
``rho_regularization``, Boumal, Mishra, Absil & Sepulchre (2014), JMLR 15, 1455–1459). It changes
ρ by a negligible amount when predicted ≫ δ, and gives ρ ≈ 1 when both reductions are at the
rounding level of f, so the Newton-like steps are still taken. Because δ is relative, a step is
accepted (ρ > η ≥ 0) only if f(x_k + p_k) < f(x_k) + δ, a relative increase below
10³ε ≈ 2.2e-13, whatever the scale of f; and for the Cauchy, dogleg and exact solvers the iterates
do not change (up to rounding) when f is multiplied by a constant c > 0 and ``gtol`` by c.
NOTE: CGT and Manopt use the absolute floor max(1, |f|) instead of |f|. It is not scale
invariant: for |f| ≲ 1e-15 it dominates both reductions for the whole run, ρ ≈ 1 for every step,
and steps that increase f many times over are accepted. The factor 10³ (not 10) is needed because
the rounding error of f is ε·Σ|terms of f|, not ε|f|: at the local minimizer (−1.75, 0.87) of
the three-hump camel, f ≈ 0.30 is a sum of terms of size ≈ 10, and the measured noise of
f(x_k) − f(x_k + p_k) reaches ≈ 50 ε|f|.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step

Array = NDArray[np.float64]

_EPS = float(np.finfo(np.float64).eps)
#: Radius update thresholds of N&W Alg. 4.1.
_RHO_SHRINK = 0.25
_RHO_EXPAND = 0.75
#: Smallest positive normal float64; ‖g‖₂² below it has underflowed (gradual underflow).
_TINY = float(np.finfo(np.float64).tiny)
#: Rounding allowance δ = _ROUNDING · ε · max(|f(x)|, |f(x + p)|) in ρ (see the module docstring);
#: 10³ is Manopt's default ``rho_regularization`` (Boumal et al. 2014).
_ROUNDING = 1e3
#: Relative accuracy |‖p(λ)‖ − Δ| ≤ _LAMBDA_RTOL·Δ of the exact solver's secular equation.
_LAMBDA_RTOL = 1e-12
#: Iteration cap of the secular-equation solver (Newton converges in a handful of steps).
_LAMBDA_MAX_ITER = 100

PARAMS = (
    ParamSpec(
        "gtol",
        1e-8,
        min=1e-14,
        max=1e-2,
        log=True,
        help="Stop when the gradient ‖∇f(x)‖∞ ≤ gtol.",
    ),
    ParamSpec("max_iter", 200, kind="int", min=1, max=100_000, help="Iteration limit."),
    ParamSpec(
        "radius0",
        1.0,
        min=1e-3,
        max=100.0,
        log=True,
        help="Initial trust-region radius Δ₀.",
    ),
    ParamSpec(
        "max_radius",
        100.0,
        min=0.1,
        max=1e4,
        log=True,
        help="Largest radius Δ̂ the update may reach.",
    ),
    ParamSpec(
        "eta",
        0.15,
        min=0.0,
        max=0.24,
        help="Accept the step when ρ = actual/predicted reduction > η (η ∈ [0, ¼)).",
    ),
)


# --------------------------------------------------------------------------------------
# Small dense linear algebra (numpy only; SciPy is not a runtime dependency)
# --------------------------------------------------------------------------------------


def _norm2(v: Array) -> float:
    """‖v‖₂ without underflow or overflow of the squares (Higham (2002), §27.8 / LAPACK dnrm2).

    v is scaled by 2⁻ᵉ with 2^(e−1) ≤ ‖v‖∞ < 2^e before squaring. Scaling by a power of 2 is
    exact, so the result equals ``np.linalg.norm(v)`` bit for bit whenever no square under- or
    overflows, and stays accurate (and > 0 for v ≠ 0) when ``np.linalg.norm`` would return 0.
    """
    vinf = float(np.max(np.abs(v))) if v.size else 0.0
    if not 0.0 < vinf < math.inf:
        return vinf  # 0, inf or nan
    _, e = math.frexp(vinf)
    return math.ldexp(float(np.linalg.norm(np.ldexp(v, -e))), e)


def _pow2_scaled(g: Array) -> tuple[Array, int]:
    """(ĝ, e) with ĝ = 2⁻ᵉg exactly and ½ ≤ ‖ĝ‖∞ < 1; g must be finite and nonzero.

    Quotients of quadratic forms such as gᵀg/gᵀBg are formed from ĝ: their value is unchanged
    (bit for bit, as both forms scale by 4⁻ᵉ exactly) but the forms do not underflow.
    """
    _, e = math.frexp(float(np.max(np.abs(g))))
    return np.ldexp(g, -e), e


def _cholesky(B: Array) -> Array | None:
    """Lower Cholesky factor L of B (B = LLᵀ), or None when B is not positive definite."""
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return None
    return L if bool(np.all(np.isfinite(L))) else None


def _cho_solve(L: Array, b: Array) -> Array:
    """Solve LLᵀx = b by forward and back substitution (Golub & Van Loan, Alg. 3.1.1–3.1.2)."""
    n = b.size
    y = np.empty(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]
    return x


def _newton_step(g: Array, B: Array) -> Array | None:
    """p^B = −B⁻¹g when B is positive definite, else None."""
    L = _cholesky(B)
    if L is None:
        return None
    p = -_cho_solve(L, g)
    return p if bool(np.all(np.isfinite(p))) else None


def _boundary_tau(z: Array, d: Array, delta: float) -> tuple[float, float]:
    """Roots τ₋ ≤ 0 ≤ τ₊ of ‖z + τd‖ = Δ for ‖z‖ ≤ Δ, d ≠ 0.

    The quadratic aτ² + 2bτ + c = 0 with a = dᵀd, b = zᵀd, c = zᵀz − Δ² ≤ 0 is solved with the
    cancellation-free formula (Higham (2002), §1.8): the root whose formula adds terms of equal
    sign first, the other from the product of the roots τ₊τ₋ = c/a.
    """
    a = float(d @ d)
    b = float(z @ d)
    c = min(float(z @ z) - delta * delta, 0.0)
    s = math.sqrt(max(b * b - a * c, 0.0))
    if b <= 0.0:
        tau_plus = (-b + s) / a
        tau_minus = c / (a * tau_plus) if tau_plus > 0.0 else (-b - s) / a
    else:
        tau_minus = (-b - s) / a
        tau_plus = c / (a * tau_minus) if tau_minus < 0.0 else (-b + s) / a
    return tau_minus, tau_plus


# --------------------------------------------------------------------------------------
# Subproblem solvers: each returns p, whether p is on the boundary, and its own info
# --------------------------------------------------------------------------------------


@dataclass
class _Subproblem:
    p: Array
    hits_boundary: bool
    info: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


def _cauchy(g: Array, B: Array, delta: float) -> _Subproblem:
    """Cauchy point, N&W Alg. 4.2: p^C = τ p^S with p^S = −Δ g/‖g‖ (eq. 4.11) and

        τ = 1                                if gᵀBg ≤ 0,
        τ = min(‖g‖³ / (Δ gᵀBg), 1)          otherwise          (eq. 4.12).

    p^C minimizes m along −g inside the trust region; it is on the boundary iff τ = 1.
    The driver never calls a solver with g = 0 (that is the stopping test); p = 0 is returned.

    With ĝ = 2⁻ᵉg (exact), ‖g‖³/(Δ gᵀBg) = 2ᵉ‖ĝ‖³/(Δ ĝᵀBĝ) and p^C = −(τΔ/‖ĝ‖) ĝ, so the
    quadratic form gᵀBg never underflows (it would for ‖g‖ ≲ 1e-154).
    """
    if not np.any(g):
        return _Subproblem(np.zeros_like(g), False, {"tau": 0.0})
    g_hat, e = _pow2_scaled(g)  # (n,), ½ ≤ ‖ĝ‖∞ < 1
    g_hat_norm = float(np.linalg.norm(g_hat))
    curv = float(g_hat @ (B @ g_hat))  # ĝᵀBĝ = gᵀBg/4ᵉ
    if curv <= 0.0:
        tau = 1.0
    else:
        with np.errstate(over="ignore", under="ignore"):
            tau = min(float(np.ldexp(g_hat_norm**3 / curv / delta, e)), 1.0)
    p = -(tau * delta / g_hat_norm) * g_hat
    return _Subproblem(p, tau >= 1.0, {"tau": tau})


def _dogleg(g: Array, B: Array, delta: float) -> _Subproblem:
    """Dogleg step, N&W §4.1. With p^B = −B⁻¹g and p^U = −(gᵀg/gᵀBg) g (eq. 4.15), the path

        p̃(τ) = τ p^U                     for 0 ≤ τ ≤ 1,
        p̃(τ) = p^U + (τ − 1)(p^B − p^U)  for 1 ≤ τ ≤ 2              (eq. 4.16)

    has increasing ‖p̃‖ and decreasing m (N&W Lemma 4.2) when B is positive definite, so the step
    is p^B if ‖p^B‖ ≤ Δ, else the point where the path leaves the trust region. The ratio
    gᵀg/gᵀBg is formed from ĝ = 2⁻ᵉg (exact, same value), so it does not underflow to 0/0.
    """
    if not np.any(g):
        return _Subproblem(np.zeros_like(g), False, {"dogleg_path": None, "tau": 0.0})

    def cauchy_fallback(why: str) -> _Subproblem:
        # NOTE: the dogleg needs B ≻ 0 (Lemma 4.2); otherwise we take the Cauchy point (N&W §4.1
        # suggests it as the safe fallback), which keeps the global convergence of Thm. 4.5.
        sub = _cauchy(g, B, delta)
        return _Subproblem(sub.p, sub.hits_boundary, {"dogleg_path": None, "tau": None}, why)

    pB = _newton_step(g, B)
    if pB is None:
        return cauchy_fallback("∇²f is not positive definite: dogleg undefined, Cauchy point used")
    g_hat, _ = _pow2_scaled(g)
    curv = float(g_hat @ (B @ g_hat))  # ĝᵀBĝ > 0 because B ≻ 0, up to rounding
    pU = -(float(g_hat @ g_hat) / curv) * g if curv > 0.0 else np.full_like(g, np.nan)
    if not _finite(pU):
        # NOTE: B passed the Cholesky test but gᵀBg ≤ 0 in floating point (B is positive definite
        # only at the rounding level along g), so p^U is undefined.
        return cauchy_fallback("gᵀ∇²f g ≤ 0 in floating point: dogleg undefined, Cauchy point used")
    path = [np.zeros_like(g), pU, pB]
    if float(np.linalg.norm(pB)) <= delta:
        return _Subproblem(pB, False, {"dogleg_path": path, "tau": 2.0})
    pU_norm = float(np.linalg.norm(pU))
    if pU_norm >= delta:
        tau = delta / pU_norm
        return _Subproblem(tau * pU, True, {"dogleg_path": path, "tau": tau})
    _, s = _boundary_tau(pU, pB - pU, delta)  # s = τ − 1 ∈ (0, 1)
    s = min(max(s, 0.0), 1.0)
    return _Subproblem(pU + s * (pB - pU), True, {"dogleg_path": path, "tau": 1.0 + s})


def _steihaug(g: Array, B: Array, delta: float) -> _Subproblem:
    """Steihaug–Toint truncated CG, N&W Alg. 7.2 (CG on Bp = −g from z₀ = 0, stopped early).

        r₀ = g, d₀ = −r₀
        for j = 0, 1, …:
            if d_jᵀBd_j ≤ 0: return z_j + τd_j, the boundary point (τ of either sign) with the
                             smaller model value                    ("negative_curvature")
            α_j = r_jᵀr_j / d_jᵀBd_j;  z_{j+1} = z_j + α_j d_j
            if ‖z_{j+1}‖ ≥ Δ: return z_j + τd_j with τ ≥ 0, ‖·‖ = Δ ("boundary")
            r_{j+1} = r_j + α_j B d_j
            if ‖r_{j+1}‖ < ε_k: return z_{j+1}                       ("residual")
            β_{j+1} = r_{j+1}ᵀr_{j+1} / r_jᵀr_j;  d_{j+1} = −r_{j+1} + β_{j+1} d_j

    with ε_k = min(½, √‖g‖)‖g‖ (N&W eq. 7.3, which gives superlinear convergence, Thm. 7.2).
    z_j increases in norm and decreases m (N&W Thm. 7.3), and z₁ is the Cauchy point.
    """
    n = g.size
    z = np.zeros(n)
    if not np.any(g):
        info = {"cg_path": [z, z], "cg_iters": 0, "termination": "residual", "cg_tol": 0.0}
        return _Subproblem(z, False, info)
    gnorm = _norm2(g)
    eps_k = min(0.5, math.sqrt(gnorm)) * gnorm
    # NOTE: the recurrences run on r̂_j = 2⁻ᵉr_j and d̂_j = 2⁻ᵉd_j (r₀ = g, scaled exactly), so
    # r̂ᵀr̂ and d̂ᵀBd̂ do not underflow when ‖g‖ is tiny. α_j = r̂ᵀr̂/d̂ᵀBd̂ is unchanged, the
    # iterate update is z_{j+1} = z_j + (2ᵉα_j) d̂_j, and the residual test compares ‖r̂‖ with
    # 2⁻ᵉε_k. In the normal range every quantity equals the unscaled one bit for bit.
    r, e = _pow2_scaled(g)  # r̂₀ = 2⁻ᵉ g
    eps_hat = min(0.5, math.sqrt(gnorm)) * float(np.linalg.norm(r))  # 2⁻ᵉ ε_k
    d = -r
    rr = float(r @ r)
    path = [z.copy()]

    def finish(p: Array, boundary: bool, why: str, iters: int) -> _Subproblem:
        path.append(p.copy())
        info = {"cg_path": path, "cg_iters": iters, "termination": why, "cg_tol": eps_k}
        return _Subproblem(p, boundary, info)

    # NOTE: in exact arithmetic CG ends after at most n iterations; 2n allows for rounding.
    max_inner = 2 * n
    for j in range(max_inner):
        Bd = B @ d
        dBd = float(d @ Bd)
        if dBd <= 0.0:
            tau_minus, tau_plus = _boundary_tau(z, d, delta)
            candidates = [z + tau_minus * d, z + tau_plus * d]
            values = [float(g @ p + 0.5 * (p @ (B @ p))) for p in candidates]
            # Ties go to τ₊ ≥ 0 (the direction of decrease at z = 0, where gᵀd = −‖g‖² < 0).
            p = candidates[0] if values[0] < values[1] else candidates[1]
            return finish(p, True, "negative_curvature", j + 1)
        alpha = rr / dBd
        with np.errstate(over="ignore", invalid="ignore"):
            z_next = z + float(np.ldexp(alpha, e)) * d
        if not float(np.linalg.norm(z_next)) < delta:  # also when z_next overflowed
            _, tau = _boundary_tau(z, d, delta)
            return finish(z + tau * d, True, "boundary", j + 1)
        r = r + alpha * Bd
        rr_next = float(r @ r)
        z = z_next
        if math.sqrt(rr_next) < eps_hat:
            return finish(z, False, "residual", j + 1)
        path.append(z.copy())
        d = -r + (rr_next / rr) * d
        rr = rr_next
    path.pop()
    return finish(z, False, "max_iter", max_inner)


def _exact(g: Array, B: Array, delta: float) -> _Subproblem:
    """Global minimizer of m(p) = gᵀp + ½pᵀBp on ‖p‖ ≤ Δ (N&W §4.3, Moré & Sorensen 1983).

    p* solves the subproblem iff, for some λ ≥ 0 (N&W Thm. 4.1),
        (B + λI)p* = −g,   λ(Δ − ‖p*‖) = 0,   B + λI ⪰ 0.
    With B = QΛQᵀ (λ₁ ≤ … ≤ λₙ) and γ = Qᵀg, p(λ) = −Σ_j γ_j/(λ_j + λ) q_j (N&W eq. 4.38), and
    ‖p(λ)‖ decreases on (−λ₁, ∞).

    1. Interior: if λ₁ > 0 and ‖p(0)‖ ≤ Δ, then λ = 0 and p* = −B⁻¹g.
    2. Hard case (N&W eq. 4.45): if λ₁ ≤ 0, γ vanishes on the eigenspace E₁ of λ₁ and
       ‖p_⊥‖ ≤ Δ with p_⊥ = −Σ_{λ_j > λ₁} γ_j/(λ_j − λ₁) q_j, then λ = −λ₁ and
       p* = p_⊥ + τ u with τ = √(Δ² − ‖p_⊥‖²) and a unit vector u ∈ E₁ (u = q₁, or the side
       of lower model value when γ vanishes on E₁ only up to rounding).
    3. Otherwise solve ‖p(λ)‖ = Δ for λ > max(0, −λ₁) by Newton's method on the secular equation
       φ(λ) = 1/Δ − 1/‖p(λ)‖ (N&W Alg. 4.3):
           λ ← λ + (‖p‖/‖q‖)² (‖p‖ − Δ)/Δ,    ‖q‖² = pᵀ(B + λI)⁻¹p = Σ_j γ_j²/(λ_j + λ)³.
       Both norms are formed from w_j = γ_j/(λ_j + λ) (|w| = |Qᵀp| = O(Δ) near the root):
       ‖p‖ = ‖w‖ and ‖q‖² = Σ_j w_j²/(λ_j + λ), so they do not underflow when ‖g‖ is tiny
       (γ_j² would for ‖g‖ ≲ 1e-154).
       φ is convex and decreasing on (−λ₁, ∞) (‖p(λ)‖ decreases and 1/‖p(λ)‖ is concave;
       Moré & Sorensen 1983, §3), so its tangent lies below it: a Newton step from any λ lands at
       or left of the root, and the iterates then increase monotonically to it. The bracket [lo, hi] with
       ‖p(lo)‖ > Δ ≥ ‖p(hi)‖ (lo = max(0, −λ₁), hi = ‖g‖/Δ − λ₁ at the start) is kept, and a
       Newton step that leaves it is replaced by the Moré–Sorensen safeguard
       μ = max(√(μ_lo μ_hi), 10⁻³ μ_hi) on the shift μ = λ + λ₁.
    4. Safety net: if the iteration stops before |‖p‖ − Δ| ≤ 10⁻¹²Δ (iteration cap, or a
       bracket collapsed to adjacent floating-point numbers), take λ = hi and
       p* = p(hi) + τ q₁ with ‖p*‖ = Δ and the sign of τ that gives the lower model value (the
       Moré & Sorensen (1983), §3, step for the nearly hard case); then
       (B + λI)p* + g = τ(λ₁ + λ)q₁. Working with μ = λ + λ₁ (below) resolves the root to full
       relative precision, so this branch is not reached in practice; ``note`` reports it.
    """
    # NOTE: N&W Alg. 4.3 factors B + λI = RᵀR by Cholesky at every iteration and Moré & Sorensen
    # estimate the eigenvector of λ₁ with LINPACK's condition estimator. We use one symmetric
    # eigendecomposition instead (n is small here): the Newton update is the same formula, the
    # safeguard interval (−λ₁, ∞) is exact, and q₁ for the (nearly) hard case is available.
    n = g.size
    lam_all, Q = np.linalg.eigh(B)
    gamma = Q.T @ g  # (n,)
    lam1 = float(lam_all[0])
    gnorm = _norm2(g)
    bnorm = float(np.max(np.abs(lam_all)))
    q1 = Q[:, 0].copy()
    # Make the eigenvector's sign reproducible: its largest |entry| is positive.
    if q1[int(np.argmax(np.abs(q1)))] < 0.0:
        q1 = -q1

    def p_of(lam: float) -> Array:
        return -(Q @ (gamma / (lam_all + lam)))

    def info(lam: float, hard: bool, iters: int) -> dict[str, Any]:
        return {"lambda": lam, "lambda_min": lam1, "hard_case": hard, "lambda_iters": iters}

    # 1. Interior solution.
    if lam1 > 0.0:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            p0 = p_of(0.0)
            p0_norm = float(np.linalg.norm(p0))
        if p0_norm <= delta:  # False for a non-finite ‖p(0)‖ (B nearly singular)
            return _Subproblem(p0, False, info(0.0, False, 0))

    # 2. Hard case: eigenvalues within rounding of λ₁ form its eigenspace, and γ "vanishes" there
    # when its norm is at the rounding level of γ = Qᵀg (or of ‖B‖Δ, the scale of ‖Bp‖).
    # NOTE: these two tolerances are our choices; the exact hard case (γ₁ = 0) is measure zero,
    # and the iteration of step 3 on μ = λ + λ₁ resolves the nearly hard case (γ₁ tiny, λ ≈ −λ₁).
    tol = 10.0 * n * _EPS
    if lam1 <= 0.0:
        cluster = lam_all - lam1 <= tol * bnorm
        gamma1 = _norm2(gamma[cluster])
        if gamma1 <= tol * max(bnorm * delta, gnorm):
            coef = np.zeros_like(gamma)
            rest = ~cluster
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                coef[rest] = gamma[rest] / (lam_all[rest] - lam1)
                p_perp = -(Q @ coef)
                pp = float(np.linalg.norm(p_perp))
            if pp <= delta:  # False for a non-finite ‖p_⊥‖ (a gap λ_j − λ₁ near underflow)
                tau = math.sqrt(max(delta * delta - pp * pp, 0.0))
                # For a unit vector u in the eigenspace E₁ of λ₁, p_⊥ ⊥ E₁ and Bu = λ₁u give
                #     m(p_⊥ + τu) = m(p_⊥) + τ uᵀg + ½τ²λ₁,
                # so the lowest model value on the boundary takes u = −P₁g/‖P₁g‖ (P₁ = the
                # projector onto E₁): the residual component of g decides the side exactly,
                # even when τ|uᵀg| is below the rounding level of m.
                # NOTE: when P₁g = 0 both sides give the same value; we take u = q₁ (its
                # largest |entry| positive) so that the step is deterministic.
                if gamma1 > 0.0:
                    u = -(Q[:, cluster] @ gamma[cluster]) / gamma1
                else:
                    u = q1
                return _Subproblem(p_perp + tau * u, True, info(-lam1, True, 0))

    # 3. Newton's method on the secular equation, safeguarded by bisection. The iteration runs
    # on the shift μ = λ + λ₁ (the smallest eigenvalue of B + λI), with d_j = λ_j − λ₁ ≥ 0
    # formed once, so λ_j + λ = d_j + μ is computed without cancellation even when the root is
    # within rounding of −λ₁ (d₁ = 0 exactly).
    d = lam_all - lam1  # (n,), d[0] = 0

    def p_of_mu(mu: float) -> Array:
        return -(Q @ (gamma / (d + mu)))

    lo = max(lam1, 0.0)  # μ at λ = max(0, −λ₁); ‖p‖ > Δ there (or the pole μ = 0)
    hi = gnorm / delta  # μ at λ = ‖g‖/Δ − λ₁, where ‖p‖ ≤ ‖g‖/μ = Δ
    mu = lo if lam1 > 0.0 else hi
    iters = 0
    while iters < _LAMBDA_MAX_ITER:
        # NOTE: when B + λI is nearly singular (μ ≲ 1e-150), ‖p(μ)‖ and ‖q‖² overflow; such a μ
        # is far left of the root, so a non-finite ‖p‖ counts as ‖p‖ > Δ and the non-finite
        # Newton update is replaced by the safeguard below.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w = gamma / (d + mu)  # (n,), −Qᵀp
            p = -(Q @ w)  # p(μ)
            pnorm = _norm2(p)
            q_sq = float(np.sum(w * w / (d + mu)))  # ‖q‖² = pᵀ(B + λI)⁻¹p
            # A non-finite or zero q_sq (B + λI nearly singular) leaves μ_new to the safeguard.
            ok = q_sq > 0.0 and math.isfinite(q_sq)
            mu_new = mu + (pnorm * pnorm / q_sq) * (pnorm - delta) / delta if ok else math.nan
        if abs(pnorm - delta) <= _LAMBDA_RTOL * delta:
            return _Subproblem(p, True, info(mu - lam1, False, iters))
        if pnorm <= delta:
            hi = mu
        else:  # ‖p‖ > Δ, or ‖p‖ not finite
            lo = mu
        # μ_new = hi is allowed: the root is exactly hi when g lies in the eigenspace of λ₁.
        if not (lo < mu_new <= hi):
            # Moré & Sorensen (1983), §3 safeguard: a geometric-mean bisection, which reaches a
            # root many orders of magnitude below hi (the nearly hard case) in a few steps.
            mu_new = max(math.sqrt(lo) * math.sqrt(hi), 1e-3 * hi)  # √(lo·hi), no underflow
        iters += 1
        if not lo < mu_new <= hi or mu_new == mu:
            break  # the bracket has collapsed to adjacent floating-point numbers
        mu = mu_new

    # 4. Safety net: move from p(hi) (‖p(hi)‖ ≤ Δ) along q₁ to the boundary.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        p = p_of_mu(hi)  # hi = 0 (‖g‖/Δ underflowed) puts the pole d₁ + μ = 0 here
        if _finite(p):
            tau_minus, tau_plus = _boundary_tau(p, q1, delta)
            candidates = [p + tau_minus * q1, p + tau_plus * q1]
            values = [float(g @ c + 0.5 * (c @ (B @ c))) for c in candidates]
            p = candidates[0] if values[0] < values[1] else candidates[1]
    if _finite(p):
        note = "secular equation not solved to tolerance; boundary step p(λ) + τq₁ used"
        return _Subproblem(p, True, info(hi - lam1, False, iters), note)
    # NOTE: not in N&W or Moré–Sorensen: when even p(hi) is not representable (‖g‖/Δ below the
    # float range) we return the Cauchy point, which keeps the Cauchy decrease (N&W Lemma 4.3)
    # and so the global convergence of Alg. 4.1; λ is then unknown (NaN → null in JSON).
    sub = _cauchy(g, B, delta)
    note = "secular equation not solvable in float64; Cauchy point used"
    return _Subproblem(sub.p, sub.hits_boundary, info(math.nan, False, iters), note)


_SOLVERS: dict[str, Callable[[Array, Array, float], _Subproblem]] = {
    "trust_region_cauchy": _cauchy,
    "trust_region_dogleg": _dogleg,
    "trust_region_steihaug": _steihaug,
    "trust_region_exact": _exact,
}

#: Method-specific info keys (None at k = 0).
_EXTRA_KEYS: dict[str, tuple[str, ...]] = {
    "trust_region_cauchy": ("tau",),
    "trust_region_dogleg": ("dogleg_path", "tau"),
    "trust_region_steihaug": ("cg_path", "cg_iters", "termination", "cg_tol"),
    "trust_region_exact": ("lambda", "lambda_min", "hard_case", "lambda_iters"),
}


# --------------------------------------------------------------------------------------
# The shared trust-region driver (N&W Alg. 4.1)
# --------------------------------------------------------------------------------------


def _resolve(
    problem: Problem | Callable[[Array], float], x0: ArrayLike | None
) -> tuple[Array, Counted, Counted, Counted]:
    """Return (x0, counted f, counted ∇f, counted ∇²f) for a Problem or a bare f(x).

    The driver works on vectors x ∈ ℝⁿ. A :class:`Problem` with ``dim == 1`` follows the scalar
    convention of ``core.types`` (f, f′, f″ take and return floats), so its callables receive
    ``float(x[0])`` and their results are lifted to shapes (1,) and (1, 1). This includes the
    Problem built from a bare callable with a one-entry x0 (``core.counting.vector_problem`` gives
    it ``dim == 1``), so that a direct call and ``numopt.minimize`` treat it the same way. A
    size-1 array returned by f is accepted as the value f(x).
    """
    prob = problem if isinstance(problem, Problem) else vector_problem(problem, x0=x0)
    scalar = prob.dim == 1
    x = start_point(prob, x0)

    def arg(z: Array) -> Any:
        return float(z[0]) if scalar else z

    f = Counted(lambda z: _value(prob.f(arg(z))))
    if prob.grad is not None:
        g_fn = prob.grad
        grad = Counted(lambda z: np.asarray(g_fn(arg(z)), dtype=np.float64).reshape(-1))
    else:
        grad = Counted(lambda z: diff.gradient(f, z))
    if prob.hess is not None:
        h_fn = prob.hess
        n = x.size
        hess = Counted(lambda z: np.asarray(h_fn(arg(z)), dtype=np.float64).reshape(n, n))
    else:
        hess = Counted(lambda z: diff.hessian(grad, z))
    return x, f, grad, hess


def _value(v: Any) -> float:
    """f(x) as a float; a size-1 array is accepted, a larger one is invalid input."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"f(x) must return a scalar, got an array of shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _finite(*values: Any) -> bool:
    return all(bool(np.all(np.isfinite(v))) for v in values)


def _eval(fn: Counted, x: Array) -> Any:
    """Evaluate without floating-point warnings; callers check the result for finiteness."""
    with np.errstate(all="ignore"):
        return fn(x)


def _vec(v: Array | None) -> list[float] | None:
    return None if v is None else [float(t) for t in v]


def _trust_region(
    method: str,
    problem: Problem | Callable[[Array], float],
    x0: ArrayLike | None,
    gtol: float,
    max_iter: int,
    radius0: float,
    max_radius: float,
    eta: float,
) -> Result:
    if not gtol >= 0.0:
        raise ValueError(f"gtol must be ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not (0.0 < radius0 < math.inf and 0.0 < max_radius < math.inf):
        raise ValueError(
            f"need 0 < radius0 < ∞ and 0 < max_radius < ∞, got radius0={radius0}, "
            f"max_radius={max_radius}"
        )
    if not 0.0 <= eta < 0.25:
        raise ValueError(f"eta must lie in [0, 1/4), got {eta}")
    max_iter = int(max_iter)
    solve = _SOLVERS[method]
    extra_keys = _EXTRA_KEYS[method]

    x, f, grad, hess = _resolve(problem, x0)
    n = x.size
    fx = float(_eval(f, x))
    g: Array = _eval(grad, x)
    B: Array = _eval(hess, x)
    trace: list[Step] = []
    n_rejected = 0

    def done(converged: bool, message: str, k: int) -> Result:
        return Result(
            method,
            x,
            fx,
            converged,
            message,
            k,
            f.n,
            grad.n,
            hess.n,
            trace=trace,
            extra={"n_rejected": n_rejected},
        )

    def base_info(center: Array, gc: Array, Bc: Array, radius: float) -> dict[str, Any]:
        info: dict[str, Any] = {
            "center": _vec(center),
            "grad": _vec(gc) if _finite(gc) else None,
            "H": Bc.tolist() if n == 2 and _finite(Bc) else None,
            "radius": radius,
            "new_radius": radius,
            "step": None,
            "step_norm": None,
            "hits_boundary": None,
            "predicted": None,
            "actual": None,
            "rho": None,
            "accepted": None,
            "trial_point": None,
            "cauchy_point": None,
            "newton_point": None,
            "note": None,
        }
        info.update(dict.fromkeys(extra_keys))
        return info

    # NOTE: N&W Alg. 4.1 takes Δ₀ ∈ (0, Δ̂). The two parameters have independent UI ranges, so a
    # radius0 above max_radius is clamped to Δ̂ (and Step 0 says so) instead of being rejected.
    radius = min(float(radius0), float(max_radius))
    note0 = (
        f"radius0 = {radius0:.3g} > max_radius = {max_radius:.3g}: Δ₀ = Δ̂ used"
        if radius0 > max_radius
        else None
    )
    if not (math.isfinite(fx) and _finite(g, B)) or B.shape != (n, n):
        gn = _norm2(g) if _finite(g) else None
        trace.append(Step(0, x.copy(), fx, gn, None, base_info(x, g, B, radius) | {"note": note0}))
        return done(False, "f, ∇f or ∇²f is not finite (or ∇²f has the wrong shape) at x0", 0)
    B = 0.5 * (B + B.T)
    trace.append(
        Step(0, x.copy(), fx, _norm2(g), None, base_info(x, g, B, radius) | {"note": note0})
    )

    for k in range(1, max_iter + 1):
        ginf = float(np.max(np.abs(g)))
        if ginf <= gtol:
            return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", k - 1)

        sub = solve(g, B, radius)
        p = sub.p
        with np.errstate(all="ignore"):
            predicted = -(float(g @ p) + 0.5 * float(p @ (B @ p)))  # m(0) − m(p)
        if not (predicted > 0.0 and math.isfinite(predicted)):
            # NOTE: every solver gives m(0) − m(p) ≥ ½‖g‖min(Δ, ‖g‖/‖B‖) > 0 in exact
            # arithmetic (N&W Lemma 4.3); a computed value ≤ 0 is rounding, and ρ would be noise.
            return done(
                False,
                f"the model step predicts no decrease (m(0) − m(p) = {predicted:.3g} at iteration "
                f"{k}): the quadratic model is at its rounding level; ‖∇f‖∞ = {ginf:.3g} > gtol",
                k - 1,
            )
        trial = x + p
        f_trial = float(_eval(f, trial))
        if math.isfinite(f_trial):
            actual: float | None = fx - f_trial
            # NOTE: N&W Alg. 4.1 uses ρ = actual/predicted. We add the relative rounding allowance
            # δ = 10³ε·max(|f(x)|, |f(x + p)|) to both (Conn, Gould & Toint (2000), §17.4.2): ρ is
            # unchanged when predicted ≫ δ, ρ ≈ 1 when both reductions are at the rounding level
            # of f, and acceptance (ρ > η ≥ 0) still requires f(x + p) < f(x) + δ. δ has no
            # absolute floor, so it is invariant under f → cf (see the module docstring).
            delta_round = _ROUNDING * _EPS * max(abs(fx), abs(f_trial))
            rho: float | None = (fx - f_trial + delta_round) / (predicted + delta_round)
        else:
            actual, rho = None, None
        rho_value = -math.inf if rho is None else rho

        # Radius update, N&W Alg. 4.1.
        if rho_value < _RHO_SHRINK:
            new_radius = _RHO_SHRINK * radius
        elif rho_value > _RHO_EXPAND and sub.hits_boundary:
            new_radius = min(2.0 * radius, max_radius)
        else:
            new_radius = radius
        accepted = rho_value > eta

        info = base_info(x, g, B, radius)
        cauchy = _cauchy(g, B, radius).p
        newton = _newton_step(g, B)
        info.update(
            {
                "new_radius": new_radius,
                "step": _vec(p),
                "step_norm": float(np.linalg.norm(p)),
                "hits_boundary": sub.hits_boundary,
                "predicted": predicted,
                "actual": actual,
                "rho": rho,
                "accepted": accepted,
                "trial_point": _vec(trial),
                "cauchy_point": _vec(x + cauchy),
                "newton_point": None if newton is None else _vec(x + newton),
                "note": sub.note,
            }
        )
        for key in extra_keys:
            value = sub.info.get(key)
            if key in ("dogleg_path", "cg_path") and value is not None:
                value = [_vec(x + v) for v in value]
            info[key] = value

        if accepted:
            g_new: Array = _eval(grad, trial)
            B_new: Array = _eval(hess, trial)
            if not (_finite(g_new, B_new) and B_new.shape == (n, n)):
                x, fx = trial, f_trial
                gn = _norm2(g_new) if _finite(g_new) else None
                trace.append(Step(k, x.copy(), fx, gn, radius, info))
                return done(False, f"∇f or ∇²f is not finite at the iterate of step {k}", k)
            x, fx, g, B = trial, f_trial, g_new, 0.5 * (B_new + B_new.T)
        else:
            n_rejected += 1
        trace.append(Step(k, x.copy(), fx, _norm2(g), radius, info))
        radius = new_radius
        if radius < _EPS * max(1.0, float(np.linalg.norm(x))):
            return done(
                False,
                f"trust radius collapsed to {radius:.3g} (no acceptable step); "
                f"‖∇f‖∞ = {float(np.max(np.abs(g))):.3g} > gtol",
                k,
            )

    ginf = float(np.max(np.abs(g)))
    if ginf <= gtol:
        return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", max_iter)
    return done(False, f"reached max_iter={max_iter}", max_iter)


# --------------------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------------------

_COMMON_REFS = ("Nocedal & Wright (2006), Numerical Optimization, Alg. 4.1",)


@register(
    id="trust_region_cauchy",
    family="unconstrained",
    name="Trust region (Cauchy point)",
    params=PARAMS,
    needs=("f", "grad", "hess"),
    order="linear (steepest descent with a model-based step length)",
    summary="Minimize the quadratic model along −∇f inside the trust region: the Cauchy point.",
    references=("Nocedal & Wright (2006), Alg. 4.2, eq. 4.11–4.12", *_COMMON_REFS),
)
def trust_region_cauchy(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    radius0: float = 1.0,
    max_radius: float = 100.0,
    eta: float = 0.15,
) -> Result:
    """Trust-region method with the Cauchy point as the step (N&W Alg. 4.1 + Alg. 4.2).

    p_k = τ p^S with p^S = −Δ_k g_k/‖g_k‖ and τ = 1 if g_kᵀB_kg_k ≤ 0, else
    min(‖g_k‖³/(Δ_k g_kᵀB_kg_k), 1) (N&W eq. 4.11–4.12). It is globally convergent (N&W Thm.
    4.5) but, like steepest descent, only linearly convergent: it is the yardstick that the other
    solvers improve on.

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, radius
    collapse, ``max_iter``, or a model at its rounding level (see the module docstring).
    """
    return _trust_region(
        "trust_region_cauchy", problem, x0, gtol, max_iter, radius0, max_radius, eta
    )


@register(
    id="trust_region_dogleg",
    family="unconstrained",
    name="Trust region (dogleg)",
    params=PARAMS,
    needs=("f", "grad", "hess"),
    order="quadratic near a minimizer with ∇²f ≻ 0",
    summary="Follow the two-segment path 0 → steepest-descent minimizer → Newton point "
    "until it leaves the trust region.",
    references=("Nocedal & Wright (2006), §4.1, eq. 4.15–4.16", "Powell (1970)", *_COMMON_REFS),
)
def trust_region_dogleg(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    radius0: float = 1.0,
    max_radius: float = 100.0,
    eta: float = 0.15,
) -> Result:
    """Dogleg trust-region method (N&W §4.1 with Alg. 4.1).

    With B_k ≻ 0: p^B = −B_k⁻¹g_k (by Cholesky), p^U = −(gᵀg/gᵀBg)g, and p_k = p̃(τ) on the
    path 0 → p^U → p^B (eq. 4.16) at ‖p̃(τ)‖ = Δ_k, or p^B when it lies inside. When B_k is not
    positive definite the dogleg is undefined and the Cauchy point is used (``note`` in the step
    info). Near a minimizer with ∇²f ≻ 0 the full Newton step is taken and convergence is
    quadratic.

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, radius
    collapse, ``max_iter``, or a model at its rounding level (see the module docstring).
    """
    return _trust_region(
        "trust_region_dogleg", problem, x0, gtol, max_iter, radius0, max_radius, eta
    )


@register(
    id="trust_region_steihaug",
    family="unconstrained",
    name="Trust region (Steihaug–Toint CG)",
    params=PARAMS,
    needs=("f", "grad", "hess"),
    order="superlinear (forcing term min(½, √‖g‖))",
    summary="Run CG on the Newton equations and stop at the trust-region boundary, at "
    "negative curvature, or when the residual is small.",
    references=(
        "Nocedal & Wright (2006), Alg. 7.2 (CG–Steihaug)",
        "Steihaug (1983), SIAM J. Numer. Anal. 20(3), 626–637",
        "Toint (1981), in Sparse Matrices and Their Uses, 57–88",
        *_COMMON_REFS,
    ),
)
def trust_region_steihaug(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    radius0: float = 1.0,
    max_radius: float = 100.0,
    eta: float = 0.15,
) -> Result:
    """Newton–CG trust-region method with the Steihaug–Toint inner solver (N&W Alg. 7.2).

    The inner CG iteration on B_k p = −g_k starts at z₀ = 0 (its first iterate is the Cauchy
    point) and stops at the boundary, at a direction of non-positive curvature (it then follows
    that direction to the boundary), or when ‖r_j‖ < min(½, √‖g_k‖)‖g_k‖ (N&W eq. 7.3). It needs
    only products B_k d, works with indefinite B_k, and converges superlinearly (N&W Thm. 7.2).

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, radius
    collapse, ``max_iter``, or a model at its rounding level (see the module docstring).
    """
    return _trust_region(
        "trust_region_steihaug", problem, x0, gtol, max_iter, radius0, max_radius, eta
    )


@register(
    id="trust_region_exact",
    family="unconstrained",
    name="Trust region (exact subproblem, Moré–Sorensen)",
    params=PARAMS,
    needs=("f", "grad", "hess"),
    order="quadratic near a minimizer with ∇²f ≻ 0",
    summary="Solve the trust-region subproblem exactly: find λ ≥ 0 with (∇²f + λI)p = −∇f "
    "and ‖p‖ = Δ, including the hard case.",
    references=(
        "Nocedal & Wright (2006), Alg. 4.3, Thm. 4.1, eq. 4.38–4.45",
        "Moré & Sorensen (1983), SIAM J. Sci. Stat. Comput. 4(3), 553–572",
        *_COMMON_REFS,
    ),
)
def trust_region_exact(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    radius0: float = 1.0,
    max_radius: float = 100.0,
    eta: float = 0.15,
) -> Result:
    """Trust-region method with the exact subproblem solution (N&W §4.3 with Alg. 4.1).

    p_k is the global minimizer of m_k on ‖p‖ ≤ Δ_k: the interior Newton step when B_k ≻ 0 and
    ‖B_k⁻¹g_k‖ ≤ Δ_k; otherwise the boundary step (B_k + λI)p = −g_k with λ ≥ max(0, −λ₁) found
    by Newton's method on the secular equation (N&W Alg. 4.3), or the hard-case step
    p_⊥ + τq₁ (N&W eq. 4.45). Because the step uses negative curvature, the limit points satisfy
    the second-order necessary conditions (N&W Thm. 4.8).

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, radius
    collapse, ``max_iter``, or a model at its rounding level (see the module docstring).
    """
    return _trust_region(
        "trust_region_exact", problem, x0, gtol, max_iter, radius0, max_radius, eta
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("trust_region_cauchy", "quadratic_bowl", {}),
    ("trust_region_cauchy", "himmelblau", {}),
    ("trust_region_dogleg", "rosenbrock", {}),
    ("trust_region_dogleg", "himmelblau", {}),
    ("trust_region_steihaug", "rosenbrock", {}),
    ("trust_region_steihaug", "six_hump_camel", {}),
    ("trust_region_exact", "rosenbrock", {}),
    ("trust_region_exact", "himmelblau", {}),
]
