"""Restarted primal-dual hybrid gradient (PDHG) for linear programs, in the style of PDLP.

Problem. Every :class:`~numopt.core.types.LinearProgram` is brought to the equality standard
form (one slack per ``≤`` row; ``max`` becomes ``min −cᵀx``)

    min c̃ᵀz   s.t.   A z = b,   z ≥ 0,      A = [[A_ub, I], [A_eq, 0]],   b = [b_ub; b_eq],

and solved through the saddle point (Applegate et al. 2021, eq. 2, with X = ℝ₊ᴺ, Y = ℝᵐ)

    min_{z ≥ 0} max_y  L(z, y) = c̃ᵀz − yᵀA z + bᵀy.

PDHG (Chambolle & Pock 2011, Algorithm 1 with θ = 1; Applegate et al. 2021, eq. 3)::

    z⁺ = proj_{z ≥ 0}( z − τ (c̃ − Aᵀy) )
    y⁺ = y + σ ( b − A(2z⁺ − z) )

with τ = η/ω, σ = ηω (Applegate et al. 2021, eq. 4), η = 0.9/‖A‖₂, so τσ‖A‖₂² = 0.81 < 1.
ω is the *primal weight*.

``# NOTE:`` the task text writes "ω = τ/σ". We follow the papers' parameterization
τ = η/ω, σ = ηω (so ω² = σ/τ), because PDLP's primal-weight update (Algorithm 3, below) is
stated for that parameterization; with ω = τ/σ the same update rule would be wrong.

Variants (the four stages of the study):

* ``restart="none"`` — plain PDHG (the iterate and its running average are both reported).
* ``restart="fixed"`` — restart from the running average every ``restart_period`` iterations
  (Applegate, Hinder, Lu & Lubin 2023, Algorithm 1 with the fixed-frequency scheme, eq. 29).
* ``restart="adaptive"`` — restart from the running average z̄ⁿ'ᵗ when the normalized duality
  gap falls by the factor β (Applegate et al. 2023, Algorithm 1 with scheme eq. 30):

      ρ_{‖z̄ⁿ'ᵗ − zⁿ'⁰‖}(z̄ⁿ'ᵗ) ≤ β ρ_{‖zⁿ'⁰ − zⁿ⁻¹'⁰‖}(zⁿ'⁰)   (n ≥ 1),     t ≥ τ⁰ = 1   (n = 0),

  with β = e⁻¹ by default (their Remark 5), tested every ``restart_check_every`` iterations.
  The normalized duality gap is ρ_r(z) = (1/r) max{ L(z, ŷ) − L(ẑ, y) : (ẑ, ŷ) ∈ Z, ‖(ẑ, ŷ) − (z, y)‖_ω ≤ r }
  (Applegate et al. 2023, eq. 4a; weighted norm ‖(z, y)‖_ω² = ω‖z‖² + ‖y‖²/ω of
  Applegate et al. 2021, §2). It is computed exactly by :func:`normalized_duality_gap`.
* ``primal_weight`` — ``"unit"`` (ω = 1), ``"balanced"`` (ω = ‖c̃‖₂/‖b‖₂, PDLP's
  InitializePrimalWeight, Applegate et al. 2021, §3.3) or ``"adaptive"`` (that start, then
  PDLP Algorithm 3 at every restart: ω ← exp(θ log(Δy/Δz) + (1−θ) log ω), θ = ½,
  Δz = ‖zⁿ'⁰ − zⁿ⁻¹'⁰‖₂, Δy = ‖yⁿ'⁰ − yⁿ⁻¹'⁰‖₂, skipped when either is ≤ 1e-10).
  ``# NOTE:`` the default is ``"balanced"``, not PDLP's ``"adaptive"``: with a constant step and
  restarts to the average, the update drove ω to ≈1.6e-7 on a random 3×10 LP after the KKT
  error had reached 3e-8, and the error then grew back above 1e-3 (README, Results §4).
* ``precondition="ruiz_pc"`` — 10 Ruiz equilibration passes then one Pock–Chambolle pass with
  α = 1 (Applegate et al. 2021, §3.5): Ã = D₁ A D₂, c̃ ← D₂ c̃, b ← D₁ b, z = D₂ ẑ, y = D₁ ŷ.

``# NOTE:`` deviations from PDLP (Applegate et al. 2021, Algorithm 1), all deliberate so that
the study isolates the effect of restarts:

* constant step η = 0.9/‖A‖₂ (as in the paper's baseline PDHG and the 2023 experiments), not
  the adaptive step of Algorithm 2. ‖A‖₂ is computed exactly by an SVD of the (small) matrix;
  this setup cost is not counted in ``n_matvec``;
* the restart test is the theory scheme (eq. 30 of the 2023 paper, equivalent to PDLP's
  "adaptive restart (theory)" mode with β_sufficient = β_necessary = e⁻¹, Appendix C.2), and
  the restart point is always the average (2023 paper, Algorithm 1 line 10), not PDLP's
  GetRestartCandidate; PDLP's conditions (ii) and (iii) are not used;
* the adaptive restart test is evaluated every ``restart_check_every`` iterations (default 40,
  as in PDLP, Applegate et al. 2021, §3 before Algorithm 1; the 2023 paper uses 30). The
  termination test is evaluated every iteration (it is free, see "Cost"). With
  ``restart_check_every=1`` an epoch can be a single iteration; then Δy/Δz in Algorithm 3 is
  ≈ ω²‖b − Az‖/‖c − Aᵀy‖ (it measures the step sizes, not the distance to optimality) and the
  update becomes log ω⁺ ≈ 1.5 log ω + const, an unstable map. In the study (README) ω falls
  below 1e-6 on 13 of 24 library instances in that setting; an interval of 40 avoids that on
  the benchmark set (but see the ``primal_weight`` NOTE above);
* no presolve and no infeasibility detection: an infeasible or unbounded LP runs to
  ``max_iter`` and returns ``converged=False``.
* the slack (equality standard) form above. PDLP keeps the inequality rows Gx ≥ h with a dual
  projection y_G ≥ 0, and measures the primal residual as ‖(Ax − b, (h − Gx)⁺)‖₂ (Applegate
  et al. 2021, eq. 6b). Here every ``≤`` row gets a slack column, so the constraint matrix gains
  an identity block. This changes ‖A‖₂ (hence η), the Ruiz/Pock–Chambolle scalings, the weighted
  norm of the normalized gap, and the KKT terms (they are measured on the slack form).

Stopping test (Applegate et al. 2021, eq. 6, on the unscaled standard form; λ = c̃ − Aᵀy):

    |c̃ᵀz − bᵀy| ≤ ε (1 + |c̃ᵀz| + |bᵀy|),   ‖A z − b‖₂ ≤ ε (1 + ‖b‖₂),   ‖min(λ, 0)‖₂ ≤ ε (1 + ‖c̃‖₂).

The *relative KKT error* is the largest of the three left/right ratios; the method stops with
``converged=True`` when the error of the current iterate or of the running average is ≤ ``tol``
and returns that point. z ≥ 0 holds for every iterate by the projection.

Cost. One product with A and one with Aᵀ per iteration, plus one of each for the start point:
``n_matvec = 2 + 2·n_iter`` exactly. The KKT error and the normalized gap need A z and Aᵀy only,
and both are kept for the iterate and (by linearity) for the running average, so they cost no
extra products. ``# NOTE:`` the running averages of A z and Aᵀy equal A z̄ and Aᵀȳ in exact
arithmetic; in floating point they differ by O(t·eps) relative, far below ``tol``.

Info keys:
    y: [float] — dual iterate of the standard form (original units), after a restart if one
        happened at this step.
    x_avg: [float] — running average z̄ⁿ'ᵗ of the current epoch, original variables (before
        the restart of this step; equal to ``Step.x`` when ``restarted``).
    kkt_last: float — relative KKT error of the PDHG iterate zⁿ'ᵗ produced at this step.
    kkt_avg: float — relative KKT error of the running average z̄ⁿ'ᵗ.
    kkt: float — min(kkt_last, kkt_avg): the error of the point the method would return.
    primal_residual, dual_residual, gap: float — the three relative terms of ``kkt_last``.
    omega: float — primal weight ω after this step.
    tau, sigma: float — primal and dual step sizes after this step.
    restarted: bool — a restart happened at this step (``Step.x`` is then the average).
    epoch: int — restart counter n after this step.
    epoch_len: int — inner iterations t of the current epoch before this step's restart.
    normalized_gap: float | None — (adaptive) ρ of the running average with radius
        ‖z̄ⁿ'ᵗ − zⁿ'⁰‖_ω at a restart check; None for n = 0, between checks, and for the
        other schemes.
    restart_threshold: float | None — (adaptive, n ≥ 1) β ρ_{‖zⁿ'⁰ − zⁿ⁻¹'⁰‖}(zⁿ'⁰).
    matvecs: int — products with A or Aᵀ so far.

``Step.x`` is the point the next iteration starts from (original variables x = z[:n]),
``Step.fun`` = cᵀx in the original sense, ``Step.step_size`` = τ.

References:
    D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy, "Practical
    large-scale linear programming using primal-dual hybrid gradient", NeurIPS 2021.
    D. Applegate, O. Hinder, H. Lu, M. Lubin, "Faster first-order primal-dual methods for linear
    programming using restarts and sharpness", Math. Program. 201 (2023) 133–184.
    A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex problems with
    applications to imaging", J. Math. Imaging Vis. 40 (2011) 120–145.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt.core.registry import ParamSpec
from numopt.core.types import LinearProgram, Result, Step
from numopt.lp.simplex import check_lp, lp_arrays

Vec = NDArray[np.float64]
Mat = NDArray[np.float64]

#: Step-size factor η·‖A‖₂ (Applegate et al. 2021, §2 baseline; 2023, §7).
ETA_FACTOR = 0.9
#: Primal-weight smoothing θ (Applegate et al. 2021, §3.3).
THETA = 0.5
#: Ruiz passes before the Pock–Chambolle pass (Applegate et al. 2021, §3.5).
RUIZ_ITERS = 10
#: "zero" of InitializePrimalWeight and of Algorithm 3 (Applegate et al. 2021, §3.3).
ZERO = 1e-10
#: First restart length τ⁰ of the adaptive scheme (Applegate et al. 2023, eq. 30: "τ⁰ = 1").
TAU0 = 1

PARAMS: dict[str, list[ParamSpec]] = {
    "restarted_pdhg": [
        ParamSpec(
            "restart",
            "adaptive",
            kind="choice",
            choices=("none", "fixed", "adaptive"),
            help="none: plain PDHG; fixed: restart to the average every restart_period "
            "iterations; adaptive: restart when the normalized duality gap falls by beta.",
        ),
        ParamSpec(
            "restart_period",
            64,
            kind="int",
            min=1,
            max=100_000,
            help="Restart length for restart='fixed'.",
        ),
        ParamSpec(
            "beta",
            math.exp(-1.0),
            min=0.01,
            max=0.99,
            help="Required decay factor of the normalized duality gap (restart='adaptive').",
        ),
        ParamSpec(
            "restart_check_every",
            40,
            kind="int",
            min=1,
            max=10_000,
            help="Evaluate the adaptive restart test every this many iterations (PDLP: 40).",
        ),
        ParamSpec(
            "primal_weight",
            "balanced",
            kind="choice",
            choices=("unit", "balanced", "adaptive"),
            help="unit: ω = 1; balanced: ω = ‖c‖/‖b‖; adaptive: balanced start, then "
            "PDLP's smoothed update at every restart.",
        ),
        ParamSpec(
            "precondition",
            "ruiz_pc",
            kind="choice",
            choices=("none", "ruiz_pc"),
            help="Diagonal preconditioning: 10 Ruiz passes then Pock–Chambolle (α = 1).",
        ),
        ParamSpec(
            "tol",
            1e-8,
            min=1e-12,
            max=1e-2,
            log=True,
            help="Stop when the relative KKT error (gap, primal and dual residual) is ≤ tol.",
        ),
        ParamSpec("max_iter", 20_000, kind="int", min=1, max=1_000_000, help="Iteration limit."),
    ]
}


# --------------------------------------------------------------------------------------
# Problem data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardLP:
    """min cᵀz s.t. A z = b, z ≥ 0; the original variables are z[:n]; ``sign`` = ±1."""

    A: Mat  # (m, N)
    b: Vec  # (m,)
    c: Vec  # (N,)
    n: int
    m_ub: int
    sign: float


def standard_form(lp: LinearProgram) -> StandardLP:
    """Equality standard form with one slack per ``≤`` row (module docstring)."""
    # NOTE: deviation from PDLP, which keeps inequality rows with a dual projection y ≥ 0
    # (Applegate et al. 2021, eq. 6b); the slack form changes ‖A‖₂, the scalings and the KKT.
    c, a_ub, b_ub, a_eq, b_eq = lp_arrays(lp)
    n, m_ub, m_eq = c.size, b_ub.size, b_eq.size
    A = np.zeros((m_ub + m_eq, n + m_ub))
    A[:m_ub, :n] = a_ub
    A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = a_eq
    sign = 1.0 if lp.sense == "min" else -1.0
    c_std = np.concatenate([sign * c, np.zeros(m_ub)])
    return StandardLP(A=A, b=np.concatenate([b_ub, b_eq]), c=c_std, n=n, m_ub=m_ub, sign=sign)


def ruiz_pock_chambolle(A: Mat, ruiz_iters: int = RUIZ_ITERS) -> tuple[Vec, Vec]:
    """Diagonal scalings d₁ (m,), d₂ (N,) with Ã = diag(d₁) A diag(d₂) (Applegate 2021, §3.5).

    Ruiz pass: divide row i by √‖Ã_i,:‖∞ and column j by √‖Ã_:,j‖∞. Pock–Chambolle pass with
    α = 1: divide row i by √‖Ã_i,:‖₁ and column j by √‖Ã_:,j‖₁ (Pock & Chambolle 2011,
    Lemma 2). A zero row or column is left unscaled.
    """
    m, N = A.shape
    d1, d2 = np.ones(m), np.ones(N)
    K = A.copy()
    for _ in range(ruiz_iters):
        r = np.sqrt(np.max(np.abs(K), axis=1, initial=0.0))
        s = np.sqrt(np.max(np.abs(K), axis=0, initial=0.0))
        r[r == 0.0] = 1.0
        s[s == 0.0] = 1.0
        K = K / r[:, None] / s[None, :]
        d1 /= r
        d2 /= s
    r = np.sqrt(np.sum(np.abs(K), axis=1))
    s = np.sqrt(np.sum(np.abs(K), axis=0))
    r[r == 0.0] = 1.0
    s[s == 0.0] = 1.0
    return d1 / r, d2 / s


# --------------------------------------------------------------------------------------
# Normalized duality gap
# --------------------------------------------------------------------------------------


def normalized_duality_gap(
    z: Vec, Az: Vec, ATy: Vec, b: Vec, c: Vec, r: float, omega: float
) -> float:
    """ρ_r(z, y) for min cᵀz s.t. Az = b, z ≥ 0 in the norm ‖(dz, dy)‖_ω² = ω‖dz‖² + ‖dy‖²/ω.

    L(z, ŷ) − L(ẑ, y) is linear in d = (ẑ − z, ŷ − y) with gradient g = (Aᵀy − c, b − Az), so
    ρ_r = (1/r) max{gᵀd : ‖d‖_ω ≤ r, d_z ≥ −z} (Applegate et al. 2023, eq. 54). The maximizer is
    d(t) = (max(−z, t g_z/ω), t ω g_y) for the largest t ≥ 0 with ‖d(t)‖_ω ≤ r (their eqs. 50–52,
    written for the weighted norm). ‖d(t)‖_ω² = S + t² Q is piecewise quadratic between the
    break points t_i = z_i ω / |g_i| (g_i < 0) where component i hits its bound; a sort of the
    break points gives t exactly (O(N log N) instead of the paper's linear-time median search).
    For r = 0 the limit ρ_0 = ‖g restricted to the tangent cone‖ (dual norm) is returned.
    """
    if r < 0.0:
        raise ValueError("radius must be ≥ 0")
    gz = ATy - c  # (N,)
    gy = b - Az  # (m,)
    qz = gz * gz / omega  # (N,) g_i² / W_i
    q_total = float(np.sum(qz) + omega * np.sum(gy * gy))
    clamp = gz < 0.0
    bp = z[clamp] * omega / -gz[clamp]  # break points t_i ≥ 0
    s = omega * z[clamp] ** 2  # W_i l_i²
    q = qz[clamp]
    if r == 0.0:
        return math.sqrt(max(q_total - float(np.sum(q[bp == 0.0])), 0.0))
    order = np.argsort(bp, kind="stable")
    bp, s, q = bp[order], s[order], q[order]
    S = np.concatenate([[0.0], np.cumsum(s)])  # (K+1,) clamped contribution on interval k
    Q = np.maximum(q_total - np.concatenate([[0.0], np.cumsum(q)]), 0.0)  # (K+1,)
    r2 = r * r
    # Norm² at the right end of interval k (bp[k]); the last interval extends to ∞.
    right = S[:-1] + bp**2 * Q[:-1]
    hit = np.nonzero(right >= r2)[0]
    if hit.size:
        k = int(hit[0])
    elif Q[-1] > 0.0:
        k = bp.size
    else:  # ‖d(∞)‖ ≤ r: every direction of ascent is blocked by a bound
        d_z = np.where(clamp, -z, 0.0)
        return float(gz @ d_z) / r
    t = math.sqrt(max(r2 - S[k], 0.0) / Q[k]) if Q[k] > 0.0 else 0.0
    d_z = np.maximum(-z, t * gz / omega)
    d_y = t * omega * gy
    return float(gz @ d_z + gy @ d_y) / r


def weighted_norm(dz: Vec, dy: Vec, omega: float) -> float:
    """‖(dz, dy)‖_ω = √(ω‖dz‖² + ‖dy‖²/ω) (Applegate et al. 2021, §2)."""
    return math.sqrt(omega * float(dz @ dz) + float(dy @ dy) / omega)


def primal_weight_update(dz: float, dy: float, omega: float, theta: float = THETA) -> float:
    """PDLP Algorithm 3: exp(θ log(Δy/Δz) + (1 − θ) log ω) when Δz, Δy > ``ZERO``, else ω."""
    # NOTE: the finiteness guard is ours: an overflowed Δ would make log(Δy/Δz) undefined.
    if dz > ZERO and dy > ZERO and math.isfinite(dz) and math.isfinite(dy):
        return math.exp(theta * math.log(dy / dz) + (1.0 - theta) * math.log(omega))
    return omega


# --------------------------------------------------------------------------------------
# KKT error
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KKT:
    error: float
    primal: float
    dual: float
    gap: float
    pobj: float
    dobj: float


def kkt_error(lp: StandardLP, z: Vec, y: Vec, Az: Vec, ATy: Vec) -> KKT:
    """Relative KKT error of (z, y) for the unscaled standard form (Applegate 2021, eq. 6)."""
    pobj = float(lp.c @ z)
    dobj = float(lp.b @ y)
    primal = float(np.linalg.norm(Az - lp.b)) / (1.0 + float(np.linalg.norm(lp.b)))
    dual = float(np.linalg.norm(np.minimum(lp.c - ATy, 0.0))) / (1.0 + float(np.linalg.norm(lp.c)))
    gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    err = max(primal, dual, gap)
    if not math.isfinite(err):
        err = math.inf
    return KKT(err, primal, dual, gap, pobj, dobj)


# --------------------------------------------------------------------------------------
# The method
# --------------------------------------------------------------------------------------


def _start(lp: StandardLP, x0: ArrayLike | None) -> Vec:
    """z⁰: x0 (negative entries set to 0) and slacks max(b_ub − A_ub x0, 0); zero if None."""
    z = np.zeros(lp.c.size)
    if x0 is None:
        return z
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x.size != lp.n or not np.all(np.isfinite(x)):
        raise ValueError(f"x0 must be a finite vector of length {lp.n}")
    # NOTE: PDHG iterates live in z ≥ 0; a start with negative entries is projected first.
    z[: lp.n] = np.maximum(x, 0.0)
    if lp.m_ub:
        z[lp.n :] = np.maximum(lp.b[: lp.m_ub] - lp.A[: lp.m_ub, : lp.n] @ z[: lp.n], 0.0)
    return z


def restarted_pdhg(
    problem: LinearProgram,
    *,
    x0: ArrayLike | None = None,
    seed: int | None = None,
    restart: str = "adaptive",
    restart_period: int = 64,
    beta: float = math.exp(-1.0),
    restart_check_every: int = 40,
    primal_weight: str = "balanced",
    precondition: str = "ruiz_pc",
    tol: float = 1e-8,
    max_iter: int = 20_000,
) -> Result:
    """Restarted PDHG for LP (Applegate, Hinder, Lu & Lubin 2023, Algorithm 1; PDLP §3).

    Inner step: PDHG (Chambolle & Pock 2011, Algorithm 1, θ = 1; Applegate et al. 2021,
    eq. 3) with τ = η/ω, σ = ηω, η = 0.9/‖Ã‖₂ (eq. 4). Running average z̄ⁿ'ᵗ = (1/t) Σᵢ zⁿ'ⁱ
    (2023 paper, Algorithm 1 line 7). Restart (line 10): zⁿ⁺¹'⁰ ← z̄ⁿ'ᵗ when the scheme of
    ``restart`` fires (eq. 29 fixed, eq. 30 adaptive); then ω is updated (PDLP Algorithm 3)
    when ``primal_weight="adaptive"``.

    Stops with ``converged=True`` when the relative KKT error (PDLP eq. 6, unscaled standard
    form) of the iterate or of the running average is ≤ ``tol``; with ``converged=False`` at
    ``max_iter`` or on a non-finite value. ``seed`` is accepted for the numopt signature and
    unused (the method is deterministic).
    """
    del seed  # deterministic method
    lp = check_lp(problem)
    if restart not in ("none", "fixed", "adaptive"):
        raise ValueError("restart must be 'none', 'fixed' or 'adaptive'")
    if primal_weight not in ("unit", "balanced", "adaptive"):
        raise ValueError("primal_weight must be 'unit', 'balanced' or 'adaptive'")
    if precondition not in ("none", "ruiz_pc"):
        raise ValueError("precondition must be 'none' or 'ruiz_pc'")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if restart_period < 1 or restart_check_every < 1 or max_iter < 1 or not tol > 0.0:
        raise ValueError("restart_period, restart_check_every, max_iter must be ≥ 1, tol > 0")

    std = standard_form(lp)
    if std.A.shape[0] == 0 or not np.any(std.A):
        raise ValueError("restarted_pdhg needs at least one nonzero constraint row")
    if precondition == "ruiz_pc":
        d1, d2 = ruiz_pock_chambolle(std.A)
    else:
        d1, d2 = np.ones(std.A.shape[0]), np.ones(std.A.shape[1])
    A = std.A * d1[:, None] * d2[None, :]  # (m, N) Ã = D₁ A D₂
    b = d1 * std.b  # (m,)
    c = d2 * std.c  # (N,)
    norm_A = float(np.linalg.norm(A, 2))  # NOTE: exact SVD; setup cost not in n_matvec
    eta = ETA_FACTOR / norm_A
    if primal_weight == "unit":
        omega = 1.0
    else:
        nc, nb = float(np.linalg.norm(c)), float(np.linalg.norm(b))
        omega = nc / nb if nc > ZERO and nb > ZERO else 1.0
    tau, sigma = eta / omega, eta * omega

    def unscale(zs: Vec, ys: Vec, Azs: Vec, ATys: Vec) -> tuple[Vec, Vec, Vec, Vec]:
        # z = D₂ ẑ, y = D₁ ŷ, A z = D₁⁻¹ Ã ẑ, Aᵀy = D₂⁻¹ Ãᵀ ŷ.
        return d2 * zs, d1 * ys, Azs / d1, ATys / d2

    def kkt_scaled(zs: Vec, ys: Vec, Azs: Vec, ATys: Vec) -> KKT:
        return kkt_error(std, *unscale(zs, ys, Azs, ATys))

    def orig_x(zs: Vec) -> Vec:
        return d2[: std.n] * zs[: std.n]

    def fun_of(zs: Vec) -> float:
        return float(std.sign * (std.c[: std.n] @ orig_x(zs)))

    # State (scaled space): iterate, its products, epoch start and running sums.
    z = _start(std, x0) / d2
    y = np.zeros(A.shape[0])
    Az, ATy = A @ z, A.T @ y
    matvecs = 2
    z_start, y_start = z.copy(), y.copy()  # zⁿ'⁰
    z_prev: Vec | None = None  # zⁿ⁻¹'⁰
    y_prev: Vec | None = None
    sums = [np.zeros_like(z), np.zeros_like(y), np.zeros_like(Az), np.zeros_like(ATy)]
    t = 0
    epoch = 0
    rho_ref: float | None = None

    k0 = kkt_scaled(z, y, Az, ATy)
    trace: list[Step] = [
        Step(
            k=0,
            x=orig_x(z),
            fun=fun_of(z),
            step_size=None,
            info={
                "y": d1 * y,
                "x_avg": orig_x(z),
                "kkt_last": k0.error,
                "kkt_avg": k0.error,
                "kkt": k0.error,
                "primal_residual": k0.primal,
                "dual_residual": k0.dual,
                "gap": k0.gap,
                "omega": omega,
                "tau": tau,
                "sigma": sigma,
                "restarted": False,
                "epoch": 0,
                "epoch_len": 0,
                "normalized_gap": None,
                "restart_threshold": None,
                "matvecs": matvecs,
            },
        )
    ]
    best = (k0.error, z.copy(), y.copy(), Az.copy(), ATy.copy(), "last")
    status = "max_iter"
    n_restarts = 0
    if k0.error <= tol:
        status = "optimal"
    k = 0
    while status == "max_iter" and k < max_iter:
        k += 1
        # PDHG step (Applegate et al. 2021, eq. 3).
        z_new = np.maximum(z - tau * (c - ATy), 0.0)
        Az_new = A @ z_new
        y_new = y + sigma * (b - (2.0 * Az_new - Az))
        ATy_new = A.T @ y_new
        matvecs += 2
        z, y, Az, ATy = z_new, y_new, Az_new, ATy_new
        t += 1
        for acc, v in zip(sums, (z, y, Az, ATy), strict=True):
            acc += v
        z_avg, y_avg, Az_avg, ATy_avg = (acc / t for acc in sums)

        k_last = kkt_scaled(z, y, Az, ATy)
        k_avg = kkt_scaled(z_avg, y_avg, Az_avg, ATy_avg)
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(y))):
            status = "nonfinite"
        if k_last.error < best[0]:
            best = (k_last.error, z.copy(), y.copy(), Az.copy(), ATy.copy(), "last")
        if k_avg.error < best[0]:
            best = (k_avg.error, z_avg, y_avg, Az_avg, ATy_avg, "average")
        if min(k_last.error, k_avg.error) <= tol:
            status = "optimal"

        # Restart decision (2023 paper, eqs. 29 and 30).
        rho_avg: float | None = None
        threshold: float | None = None
        do_restart = False
        if status == "max_iter":
            if restart == "fixed":
                do_restart = t >= restart_period
            elif restart == "adaptive" and t % restart_check_every == 0:
                if epoch == 0:
                    do_restart = t >= TAU0
                else:
                    r = weighted_norm(z_avg - z_start, y_avg - y_start, omega)
                    rho_avg = normalized_duality_gap(z_avg, Az_avg, ATy_avg, b, c, r, omega)
                    assert rho_ref is not None
                    threshold = beta * rho_ref
                    do_restart = rho_avg <= threshold
        epoch_len = t
        if do_restart:
            z_prev, y_prev = z_start, y_start
            z, y, Az, ATy = z_avg.copy(), y_avg.copy(), Az_avg.copy(), ATy_avg.copy()
            z_start, y_start = z.copy(), y.copy()
            for acc in sums:
                acc[:] = 0.0
            t = 0
            epoch += 1
            n_restarts += 1
            if primal_weight == "adaptive":
                omega = primal_weight_update(
                    float(np.linalg.norm(z_start - z_prev)),
                    float(np.linalg.norm(y_start - y_prev)),
                    omega,
                )
                tau, sigma = eta / omega, eta * omega
            if restart == "adaptive":
                r_ref = weighted_norm(z_start - z_prev, y_start - y_prev, omega)
                rho_ref = normalized_duality_gap(z_start, Az, ATy, b, c, r_ref, omega)

        trace.append(
            Step(
                k=k,
                x=orig_x(z),
                fun=fun_of(z),
                step_size=tau,
                info={
                    "y": d1 * y,
                    "x_avg": orig_x(z_avg),
                    "kkt_last": k_last.error,
                    "kkt_avg": k_avg.error,
                    "kkt": min(k_last.error, k_avg.error),
                    "primal_residual": k_last.primal,
                    "dual_residual": k_last.dual,
                    "gap": k_last.gap,
                    "omega": omega,
                    "tau": tau,
                    "sigma": sigma,
                    "restarted": do_restart,
                    "epoch": epoch,
                    "epoch_len": epoch_len,
                    "normalized_gap": rho_avg,
                    "restart_threshold": threshold,
                    "matvecs": matvecs,
                },
            )
        )

    err, zb, yb, Azb, ATyb, which = best
    kb = kkt_scaled(zb, yb, Azb, ATyb)
    z_out, y_out, _, _ = unscale(zb, yb, Azb, ATyb)
    if status == "optimal":
        message = (
            f"relative KKT error {err:.2e} ≤ tol={tol:g} at the {which} point after "
            f"{k} iterations ({matvecs} products with A or Aᵀ, {n_restarts} restarts)"
        )
    elif status == "nonfinite":
        message = f"stopped: non-finite iterate at iteration {k}"
    else:
        message = (
            f"reached max_iter={max_iter}; best relative KKT error {err:.2e} > tol={tol:g} "
            "(no infeasibility detection: the LP may be infeasible or unbounded)"
        )
    extra: dict[str, Any] = {
        "status": status,
        "y": y_out,
        "output": which,
        "kkt": err,
        "primal_residual": kb.primal,
        "dual_residual": kb.dual,
        "gap": kb.gap,
        "n_matvec": matvecs,
        "n_restarts": n_restarts,
        "omega": omega,
        "eta": eta,
        "norm_A": norm_A,
        "variant": {
            "restart": restart,
            "restart_period": restart_period,
            "beta": beta,
            "primal_weight": primal_weight,
            "precondition": precondition,
        },
    }
    return Result(
        method="restarted_pdhg",
        x=z_out[: std.n],
        fun=float(std.sign * kb.pobj),
        converged=status == "optimal",
        message=message,
        n_iter=k,
        trace=trace,
        extra=extra,
    )
