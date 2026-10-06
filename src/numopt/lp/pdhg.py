"""Restarted primal-dual hybrid gradient (PDHG) for linear programs, in the style of PDLP.

Promoted from the verified study ``research/pdlp-restarted-pdhg`` (same mathematics, same
defaults; the study's README has the experiment behind every default).

Problem. Every :class:`~numopt.core.types.LinearProgram`

    min/max cᵀx   s.t.   A_ub x ≤ b_ub,   A_eq x = b_eq,   x ≥ 0

is solved in the equality standard form of :func:`numopt.lp.interior_point.equality_form`
(one slack per ``≤`` row; ``max`` becomes ``min −cᵀx``)

    min c̃ᵀz   s.t.   A z = b,   z ≥ 0,      A = [[A_ub, I], [A_eq, 0]],   b = [b_ub; b_eq],

through the saddle point (Applegate et al. 2021, eq. 2, with X = ℝ₊ᴺ, Y = ℝᵐ)

    min_{z ≥ 0} max_y  L(z, y) = c̃ᵀz − yᵀA z + bᵀy.

Integrality flags are ignored: an integer program is solved as its LP relaxation.

PDHG step (Chambolle & Pock 2011, Algorithm 1 with θ = 1; Applegate et al. 2021, eq. 3)::

    z⁺ = proj_{z ≥ 0}( z − τ (c̃ − Aᵀy) )
    y⁺ = y + σ ( b − A(2z⁺ − z) )

with τ = η/ω, σ = ηω (Applegate et al. 2021, eq. 4) and η = 0.9/‖A‖₂, so that
τσ‖A‖₂² = 0.81 < 1 (the convergence condition of Chambolle & Pock 2011, Theorem 1). ω is the
*primal weight*.

Restarts (Applegate, Hinder, Lu & Lubin 2023, Algorithm 1). Epoch n starts at zⁿ'⁰ and keeps
the running average z̄ⁿ'ᵗ = (1/t) Σᵢ₌₁ᵗ zⁿ'ⁱ (line 7). A restart sets zⁿ⁺¹'⁰ ← z̄ⁿ'ᵗ (line 10):

* ``restart="none"`` — plain PDHG; the iterate and the average of the whole run are reported.
* ``restart="fixed"`` — restart every ``restart_period`` iterations (2023 paper, eq. 29).
* ``restart="adaptive"`` — restart when the normalized duality gap falls by the factor β
  (2023 paper, eq. 30; β = e⁻¹ by default, their Remark 5):

      ρ_{‖z̄ⁿ'ᵗ − zⁿ'⁰‖}(z̄ⁿ'ᵗ) ≤ β ρ_{‖zⁿ'⁰ − zⁿ⁻¹'⁰‖}(zⁿ'⁰)   (n ≥ 1),     t ≥ τ⁰ = 1   (n = 0),

  tested every ``restart_check_every`` iterations. The normalized duality gap is
  ρ_r(z, y) = (1/r) max{L(z, ŷ) − L(ẑ, y) : ẑ ≥ 0, ‖(ẑ − z, ŷ − y)‖_ω ≤ r} (2023 paper,
  eq. 4a) in the weighted norm ‖(dz, dy)‖_ω² = ω‖dz‖² + ‖dy‖²/ω (Applegate et al. 2021, §2).
  :func:`normalized_duality_gap` computes it exactly.

Primal weight ``primal_weight``: ``"unit"`` (ω = 1); ``"balanced"`` (ω = ‖c̃‖₂/‖b‖₂, PDLP's
InitializePrimalWeight, Applegate et al. 2021, §3.3; ω = 1 when either norm is ≤ 1e-10);
``"adaptive"`` (that start, then PDLP Algorithm 3 at every restart:
ω ← exp(θ log(Δy/Δz) + (1 − θ) log ω), θ = ½, Δz = ‖zⁿ'⁰ − zⁿ⁻¹'⁰‖₂, Δy = ‖yⁿ'⁰ − yⁿ⁻¹'⁰‖₂,
skipped when either is ≤ 1e-10).

Preconditioning ``precondition="ruiz_pc"``: 10 Ruiz equilibration passes, then one
Pock–Chambolle pass with α = 1 (Applegate et al. 2021, §3.5; Pock & Chambolle 2011, Lemma 2):
Ã = D₁ A D₂, ĉ = D₂ c̃, b̂ = D₁ b; the method iterates on (ẑ, ŷ) with z = D₂ ẑ, y = D₁ ŷ.
:func:`ruiz_pock_chambolle` gives ‖Ã‖₂ ≤ 1.

Stopping test (Applegate et al. 2021, eq. 6, on the *unscaled* standard form; λ = c̃ − Aᵀy):

    |c̃ᵀz − bᵀy| ≤ ε (1 + |c̃ᵀz| + |bᵀy|),   ‖A z − b‖₂ ≤ ε (1 + ‖b‖₂),   ‖min(λ, 0)‖₂ ≤ ε (1 + ‖c̃‖₂).

The *relative KKT error* is the largest of the three ratios (:func:`kkt_error`). The method
stops with ``converged=True`` when the error of the current iterate or of the running average
is ≤ ``tol``, and returns that point. Every iterate satisfies z ≥ 0 (projection).

Failures (``converged=False``, ``Result.extra["status"]``): ``"max_iter"`` — the iteration
limit; ``Result.x`` is then the iterate or average with the smallest KKT error seen;
``"nonfinite"`` — a non-finite iterate. ``# NOTE:`` there is no infeasibility or
unboundedness detection (PDLP §3 detects them from the iterate differences); an infeasible or
unbounded LP runs to ``max_iter``.

``# NOTE:`` deviations from PDLP (Applegate et al. 2021, Algorithm 1). The study chose each of
them so that it measures the effect of restarts alone; its README gives the evidence.

* Constant step η = 0.9/‖A‖₂ (the paper's baseline PDHG and the 2023 experiments), not the
  adaptive step of Algorithm 2. ‖A‖₂ comes from an exact SVD of the (small) matrix; this setup
  cost is not counted in ``n_matvec``.
* The adaptive test is the theory scheme (2023 paper, eq. 30), i.e. PDLP's "adaptive restart
  (theory)" with β_sufficient = β_necessary = β (PDLP App. C.2); the restart point is always
  the average (2023 paper, Algorithm 1 line 10), not PDLP's GetRestartCandidate, and PDLP's
  restart conditions (ii)–(iii) are not used.
* The adaptive test runs every ``restart_check_every`` iterations (default 40, as PDLP §3);
  the stopping test runs every iteration (it costs no products, see "Cost").
* Default ``primal_weight="balanced"``, not PDLP's ``"adaptive"``. With a constant step and
  restarts to the average, Algorithm 3 drove ω to ≈ 1.6e-7 on a random 3×10 LP after the KKT
  error had reached 3e-8, and the error then grew above 1e-3 (study README, Results §4). With
  ``restart_check_every=1`` an epoch can be one iteration; then Δy/Δz ≈ ω²‖b − Az‖/‖c̃ − Aᵀy‖
  and log ω⁺ ≈ 1.5 log ω + const is an unstable map.
* The slack (equality) standard form. PDLP keeps the rows Gx ≥ h with a dual projection
  y_G ≥ 0 (Applegate et al. 2021, eq. 6b). The slack columns add an identity block to A,
  which changes ‖A‖₂ (hence η), the scalings, the weighted norm and the KKT terms.
* The task statement of the study wrote "ω = τ/σ"; the papers' parameterization τ = η/ω,
  σ = ηω (ω² = σ/τ) is used, because PDLP Algorithm 3 is stated for it.

Cost. One product with A and one with Aᵀ per iteration, plus one of each for the start:
``Result.extra["n_matvec"] = 2 + 2·n_iter`` exactly. A z and Aᵀy are kept for the iterate
and, by linearity, for the running average, so the KKT error and the normalized gap cost no
further products. ``# NOTE:`` the running averages of A z and Aᵀy equal A z̄ and Aᵀȳ in exact
arithmetic; in floating point they differ by O(t·eps) relative, far below ``tol``. The
normalized gap costs O(N log N) per restart check (a sort), not counted. ``n_fev``,
``n_gev`` and ``n_hev`` are 0.

Trace: Step k = 0 is the start (y⁰ = 0); Step k is the state after k PDHG steps.
``Step.x`` is the point the next iteration starts from (original variables x = z[:n]; the
average when this step restarted), ``Step.fun`` = cᵀ``Step.x`` in the original sense,
``Step.step_size`` = the τ that produced this step (``None`` at k = 0).
``n_iter == trace[-1].k``.

Info keys (every step; values in original units, standard-form duals):
    x: [n] — same as ``Step.x``: the primal point in original variables (for the LP plot).
    x_pdhg: [n] — the PDHG iterate zⁿ'ᵗ produced at this step, before any restart
        (original variables; the start x⁰ at k = 0).
    x_avg: [n] — the running average z̄ⁿ'ᵗ of the current epoch, before this step's restart
        (equal to ``x`` when ``restarted``).
    y: [m] — the dual iterate of the standard form (one entry per ``≤`` row, then per ``=``
        row), after this step's restart.
    kkt_last: float — relative KKT error of the PDHG iterate zⁿ'ᵗ.
    kkt_avg: float — relative KKT error of the running average z̄ⁿ'ᵗ.
    kkt: float — min(kkt_last, kkt_avg), the error of the point the method would return.
    primal_residual: float — ‖A z − b‖₂/(1 + ‖b‖₂) of the PDHG iterate.
    dual_residual: float — ‖min(c̃ − Aᵀy, 0)‖₂/(1 + ‖c̃‖₂) of the PDHG iterate.
    gap: float — |c̃ᵀz − bᵀy|/(1 + |c̃ᵀz| + |bᵀy|) of the PDHG iterate.
    omega: float — primal weight ω after this step (it changes only at a restart).
    tau, sigma: float — primal and dual step sizes after this step.
    restarted: bool — a restart happened at this step (``x`` is then the average).
    epoch: int — restart counter n after this step.
    epoch_len: int — inner iterations t of the epoch at this step, before its restart.
    normalized_gap: float | None — (adaptive, n ≥ 1, at a restart check) ρ of the running
        average with radius ‖z̄ⁿ'ᵗ − zⁿ'⁰‖_ω (scaled space); ``None`` otherwise. ω is the
        weight of epoch n, i.e. ``omega`` of the previous step: at a restart, ``omega`` of
        this step is already the updated weight.
    restart_threshold: float | None — (with ``normalized_gap``) β ρ_{‖zⁿ'⁰ − zⁿ⁻¹'⁰‖}(zⁿ'⁰),
        in the same ω (computed at the restart that opened epoch n, after its ω update).
    matvecs: int — products with A or Aᵀ so far (2 + 2k).

References:
    D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy, "Practical
    large-scale linear programming using primal-dual hybrid gradient", NeurIPS 2021
    (Algorithm 1, Algorithm 3, eqs. 2–6, §3.3, §3.5).
    D. Applegate, O. Hinder, H. Lu, M. Lubin, "Faster first-order primal-dual methods for linear
    programming using restarts and sharpness", Math. Program. 201 (2023) 133–184
    (Algorithm 1, eqs. 4a, 29, 30, 50–54).
    A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex problems with
    applications to imaging", J. Math. Imaging Vis. 40 (2011) 120–145 (Algorithm 1).
    T. Pock, A. Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in
    convex optimization", ICCV 2011 (Lemma 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.registry import ParamSpec, register
from ..core.types import LinearProgram, Result, Step
from .interior_point import _norm, equality_form
from .simplex import check_lp

Vec = NDArray[np.float64]
Mat = NDArray[np.float64]

#: Step-size factor η·‖A‖₂ (Applegate et al. 2021, §2 baseline; 2023, §7).
ETA_FACTOR = 0.9
#: Primal-weight smoothing θ of PDLP Algorithm 3 (Applegate et al. 2021, §3.3).
THETA = 0.5
#: Ruiz passes before the Pock–Chambolle pass (Applegate et al. 2021, §3.5).
RUIZ_ITERS = 10
#: The "zero" of InitializePrimalWeight and of Algorithm 3 (Applegate et al. 2021, §3.3).
ZERO = 1e-10
#: First restart length τ⁰ of the adaptive scheme (Applegate et al. 2023, eq. 30: "τ⁰ = 1").
TAU0 = 1

RESTARTS = ("none", "fixed", "adaptive")
PRIMAL_WEIGHTS = ("unit", "balanced", "adaptive")
PRECONDITIONERS = ("none", "ruiz_pc")


# --------------------------------------------------------------------------------------
# Problem data and preconditioning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardLP:
    """min cᵀz s.t. A z = b, z ≥ 0 (unscaled); the original variables are z[:n]."""

    A: Mat  # (m, N)
    b: Vec  # (m,)
    c: Vec  # (N,)  c̃: sign · c on x, 0 on the slacks
    n: int
    m_ub: int
    sign: float  # +1 for min, −1 for max


def standard_form(lp: LinearProgram) -> StandardLP:
    """The equality standard form (module docstring), one slack per ``≤`` row."""
    A, b, c, n, sign = equality_form(lp)
    return StandardLP(A=A, b=b, c=c, n=n, m_ub=A.shape[1] - n, sign=sign)


def ruiz_pock_chambolle(A: Mat, ruiz_iters: int = RUIZ_ITERS) -> tuple[Vec, Vec]:
    """Diagonal scalings d₁ (m,), d₂ (N,) with Ã = diag(d₁) A diag(d₂) (Applegate 2021, §3.5).

    Ruiz pass: divide row i by √‖Ã_i,:‖∞ and column j by √‖Ã_:,j‖∞. Pock–Chambolle pass with
    α = 1: divide row i by √‖Ã_i,:‖₁ and column j by √‖Ã_:,j‖₁ (Pock & Chambolle 2011,
    Lemma 2, which gives ‖Ã‖₂ ≤ 1). A zero row or column is left unscaled.
    """
    m, N = A.shape
    d1, d2 = np.ones(m), np.ones(N)
    K = A.copy()
    for _ in range(ruiz_iters):
        r = np.sqrt(np.max(np.abs(K), axis=1, initial=0.0))  # (m,)
        s = np.sqrt(np.max(np.abs(K), axis=0, initial=0.0))  # (N,)
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
# Normalized duality gap, primal weight, KKT error
# --------------------------------------------------------------------------------------


def weighted_norm(dz: Vec, dy: Vec, omega: float) -> float:
    """‖(dz, dy)‖_ω = √(ω‖dz‖² + ‖dy‖²/ω) (Applegate et al. 2021, §2)."""
    return math.sqrt(omega * float(dz @ dz) + float(dy @ dy) / omega)


def normalized_duality_gap(
    z: Vec, Az: Vec, ATy: Vec, b: Vec, c: Vec, r: float, omega: float
) -> float:
    """ρ_r(z, y) for min cᵀz s.t. Az = b, z ≥ 0 in the norm ‖·‖_ω (2023 paper, eq. 4a).

    L(z, ŷ) − L(ẑ, y) is linear in d = (ẑ − z, ŷ − y) with gradient g = (Aᵀy − c, b − Az),
    so ρ_r = (1/r) max{gᵀd : ‖d‖_ω ≤ r, d_z ≥ −z} (2023 paper, eq. 54). The maximizer is
    d(t) = (max(−z, t g_z/ω), t ω g_y) for the largest t ≥ 0 with ‖d(t)‖_ω ≤ r (their
    eqs. 50–52, written for the weighted norm). ‖d(t)‖_ω² = S + t² Q is piecewise quadratic
    between the break points tᵢ = zᵢ ω / |gᵢ| (gᵢ < 0) where component i reaches its bound;
    a sort of the break points gives t exactly (O(N log N) instead of the paper's linear-time
    median search). For r = 0 the limit ρ₀ (the dual norm of g restricted to the tangent cone
    of z ≥ 0) is returned. ``Az`` and ``ATy`` are the products A z and Aᵀy (y is not needed).
    """
    if not r >= 0.0:
        raise ValueError("radius must be ≥ 0")
    gz = ATy - c  # (N,)
    gy = b - Az  # (m,)
    qz = gz * gz / omega  # (N,) gᵢ²/Wᵢ with W = ω on z
    clamp = gz < 0.0
    # NOTE: Q (the t² coefficient) is accumulated from the free part plus a *reverse*
    # cumulative sum of the clampable terms, not as q_total − cumsum: a sum of nonnegative
    # terms has no cancellation (Higham, 2nd ed., §4.2), and the last interval then has Q = 0
    # exactly when every ascent direction is blocked, so t never overflows to inf.
    q_free = float(np.sum(qz[~clamp]) + omega * np.sum(gy * gy))
    bp = z[clamp] * omega / -gz[clamp]  # break points tᵢ ≥ 0
    s = omega * z[clamp] ** 2  # Wᵢ lᵢ², the clamped contribution
    q = qz[clamp]
    if r == 0.0:
        return math.sqrt(q_free + float(np.sum(q[bp > 0.0])))
    order = np.argsort(bp, kind="stable")
    bp, s, q = bp[order], s[order], q[order]
    S = np.concatenate([[0.0], np.cumsum(s)])  # (K+1,) clamped part on interval k
    Q = q_free + np.concatenate([np.cumsum(q[::-1])[::-1], [0.0]])  # (K+1,) free part
    r2 = r * r
    right = S[:-1] + bp**2 * Q[:-1]  # ‖d‖² at the right end bp[k] of interval k
    hit = np.nonzero(right >= r2)[0]
    if hit.size:
        k = int(hit[0])
    elif Q[-1] > 0.0:
        k = bp.size
    else:  # ‖d(∞)‖_ω ≤ r: every ascent direction is blocked by a bound
        return float(gz[clamp] @ -z[clamp]) / r
    t = math.sqrt(max(r2 - float(S[k]), 0.0) / float(Q[k])) if Q[k] > 0.0 else 0.0
    d_z = np.maximum(-z, t * gz / omega)
    d_y = t * omega * gy
    return float(gz @ d_z + gy @ d_y) / r


def primal_weight_update(dz: float, dy: float, omega: float, theta: float = THETA) -> float:
    """PDLP Algorithm 3: exp(θ log(Δy/Δz) + (1 − θ) log ω) when Δz, Δy > ``ZERO``, else ω."""
    # NOTE: the finiteness guard is not in PDLP: an overflowed Δ makes log(Δy/Δz) undefined.
    if dz > ZERO and dy > ZERO and math.isfinite(dz) and math.isfinite(dy):
        return math.exp(theta * math.log(dy / dz) + (1.0 - theta) * math.log(omega))
    return omega


@dataclass(frozen=True)
class KKT:
    """The relative KKT error and its three terms (Applegate et al. 2021, eq. 6)."""

    error: float
    primal: float
    dual: float
    gap: float
    pobj: float
    dobj: float


def kkt_error(lp: StandardLP, z: Vec, y: Vec, Az: Vec, ATy: Vec) -> KKT:
    """Relative KKT error of (z, y) for the unscaled standard form (module docstring)."""
    # NOTE: the norms use the overflow-free scaling of LAPACK's xNRM2 (``_norm``): with
    # np.linalg.norm, ‖c̃‖₂ = inf for entries near 1e308 made the dual term (finite)/inf = 0
    # and reported a non-optimal start as converged.
    pobj = float(lp.c @ z)
    dobj = float(lp.b @ y)
    primal = _norm(Az - lp.b) / (1.0 + _norm(lp.b))
    dual = _norm(np.minimum(lp.c - ATy, 0.0)) / (1.0 + _norm(lp.c))
    gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    # NOTE: Python's max drops a NaN term (max(1e-9, nan) == 1e-9), so a non-finite term
    # is mapped to an error of inf explicitly.
    terms = (primal, dual, gap)
    err = max(terms) if all(math.isfinite(v) for v in terms) else math.inf
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


@register(
    id="restarted_pdhg",
    family="lp",
    name="Restarted PDHG (PDLP-style)",
    params=(
        ParamSpec(
            "restart",
            "adaptive",
            kind="choice",
            choices=RESTARTS,
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
            choices=PRIMAL_WEIGHTS,
            help="unit: ω = 1; balanced: ω = ‖c‖/‖b‖; adaptive: balanced start, then "
            "PDLP's smoothed update at every restart.",
        ),
        ParamSpec(
            "precondition",
            "ruiz_pc",
            kind="choice",
            choices=PRECONDITIONERS,
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
    ),
    needs=("lp",),
    order="linear with restarts (sharp LPs); O(1/k) for the plain average",
    summary="Take cheap projected primal and dual gradient steps on the Lagrangian, and "
    "restart from the running average whenever the normalized duality gap has fallen enough.",
    references=(
        "Applegate, Hinder, Lu & Lubin (2023), Math. Program. 201:133–184, Algorithm 1, "
        "restart schemes eqs. 29–30, normalized duality gap eq. 4a",
        "Applegate et al. (2021), NeurIPS, PDLP: eqs. 3–4 (PDHG step), eq. 6 (KKT error), "
        "Algorithm 3 (primal weight), §3.5 (preconditioning)",
        "Chambolle & Pock (2011), J. Math. Imaging Vis. 40:120–145, Algorithm 1 (θ = 1)",
        "Pock & Chambolle (2011), ICCV, Lemma 2 (diagonal preconditioning)",
    ),
)
def restarted_pdhg(
    problem: LinearProgram,
    *,
    x0: ArrayLike | None = None,
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
    eq. 3) with τ = η/ω, σ = ηω, η = 0.9/‖Ã‖₂ (eq. 4). Restart to the running average when
    the scheme of ``restart`` fires (2023 paper, eq. 29 fixed, eq. 30 adaptive); then ω is
    updated (PDLP Algorithm 3) when ``primal_weight="adaptive"``.

    Stops with ``converged=True`` when the relative KKT error (PDLP eq. 6, unscaled standard
    form) of the iterate or of the running average is ≤ ``tol``; ``converged=False`` at
    ``max_iter`` or on a non-finite iterate. ``x0`` (optional, default 0) is the primal start
    in original variables; the slacks start at max(b_ub − A_ub x0, 0) and y⁰ = 0.
    """
    lp = check_lp(problem)
    if restart not in RESTARTS:
        raise ValueError(f"restart must be one of {RESTARTS}")
    if primal_weight not in PRIMAL_WEIGHTS:
        raise ValueError(f"primal_weight must be one of {PRIMAL_WEIGHTS}")
    if precondition not in PRECONDITIONERS:
        raise ValueError(f"precondition must be one of {PRECONDITIONERS}")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if restart_period < 1 or restart_check_every < 1 or max_iter < 1:
        raise ValueError("restart_period, restart_check_every and max_iter must be ≥ 1")
    if not tol > 0.0:
        raise ValueError("tol must be > 0")

    # NOTE: overflow/invalid floating-point warnings are silenced: a non-finite iterate is
    # detected and reported (status "nonfinite"), and the contract forbids raising on it.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        std = standard_form(lp)
        if std.A.shape[0] == 0 or not np.any(std.A):
            raise ValueError(f"{lp.id}: restarted_pdhg needs at least one nonzero constraint row")
        if precondition == "ruiz_pc":
            d1, d2 = ruiz_pock_chambolle(std.A)
        else:
            d1, d2 = np.ones(std.A.shape[0]), np.ones(std.A.shape[1])
        A = std.A * d1[:, None] * d2[None, :]  # (m, N) Ã = D₁ A D₂
        b = d1 * std.b  # (m,)
        c = d2 * std.c  # (N,)
        norm_A = float(np.linalg.norm(A, 2))  # NOTE: exact SVD; setup cost not in n_matvec
        eta = ETA_FACTOR / norm_A
        omega = 1.0
        if primal_weight != "unit":
            nc, nb = float(np.linalg.norm(c)), float(np.linalg.norm(b))
            if nc > ZERO and nb > ZERO:
                omega = nc / nb
        tau, sigma = eta / omega, eta * omega

        def unscale(zs: Vec, ys: Vec, Azs: Vec, ATys: Vec) -> tuple[Vec, Vec, Vec, Vec]:
            # z = D₂ ẑ, y = D₁ ŷ, A z = D₁⁻¹ Ã ẑ, Aᵀy = D₂⁻¹ Ãᵀ ŷ.
            return d2 * zs, d1 * ys, Azs / d1, ATys / d2

        def kkt_of(zs: Vec, ys: Vec, Azs: Vec, ATys: Vec) -> KKT:
            return kkt_error(std, *unscale(zs, ys, Azs, ATys))

        def orig_x(zs: Vec) -> Vec:
            return d2[: std.n] * zs[: std.n]

        def fun_of(x: Vec) -> float:
            return float(std.sign * (std.c[: std.n] @ x))

        def make_step(
            k: int, zs: Vec, ys: Vec, z_pdhg: Vec, z_avg: Vec, kl: KKT, ka: KKT, **flags: Any
        ) -> Step:
            x = orig_x(zs)
            return Step(
                k=k,
                x=x,
                fun=fun_of(x),
                step_size=flags.pop("step_size"),
                info={
                    "x": x.copy(),
                    "x_pdhg": orig_x(z_pdhg),
                    "x_avg": orig_x(z_avg),
                    "y": d1 * ys,
                    "kkt_last": kl.error,
                    "kkt_avg": ka.error,
                    "kkt": min(kl.error, ka.error),
                    "primal_residual": kl.primal,
                    "dual_residual": kl.dual,
                    "gap": kl.gap,
                    "omega": omega,
                    "tau": tau,
                    "sigma": sigma,
                    **flags,
                },
            )

        # State in the scaled space: the iterate and its products, the epoch start zⁿ'⁰ and the
        # previous epoch start zⁿ⁻¹'⁰, and running sums of (z, y, Az, Aᵀy) over the epoch.
        z = _start(std, x0) / d2  # (N,)
        y = np.zeros(A.shape[0])  # (m,)
        Az, ATy = A @ z, A.T @ y
        matvecs = 2
        z_start, y_start = z.copy(), y.copy()
        sums = [np.zeros_like(z), np.zeros_like(y), np.zeros_like(Az), np.zeros_like(ATy)]
        t = 0
        epoch = 0
        rho_ref = math.inf  # β ρ of the epoch start; set at the first restart (n ≥ 1)

        k0 = kkt_of(z, y, Az, ATy)
        trace = [
            make_step(
                0,
                z,
                y,
                z,
                z,
                k0,
                k0,
                step_size=None,
                restarted=False,
                epoch=0,
                epoch_len=0,
                normalized_gap=None,
                restart_threshold=None,
                matvecs=matvecs,
            )
        ]
        best = (k0.error, z.copy(), y.copy(), Az.copy(), ATy.copy(), "iterate")
        status = "optimal" if k0.error <= tol else "max_iter"
        k = 0
        while status == "max_iter" and k < max_iter:
            k += 1
            tau_used = tau
            # PDHG step (Applegate et al. 2021, eq. 3); A(2z⁺ − z) = 2Az⁺ − Az by linearity.
            z = np.maximum(z - tau * (c - ATy), 0.0)
            Az_old, Az = Az, A @ z
            y = y + sigma * (b - (2.0 * Az - Az_old))
            ATy = A.T @ y
            matvecs += 2
            t += 1
            for acc, v in zip(sums, (z, y, Az, ATy), strict=True):
                acc += v
            z_avg, y_avg, Az_avg, ATy_avg = (acc / t for acc in sums)

            k_last = kkt_of(z, y, Az, ATy)
            k_avg = kkt_of(z_avg, y_avg, Az_avg, ATy_avg)
            if not (np.all(np.isfinite(z)) and np.all(np.isfinite(y))):
                status = "nonfinite"
            if k_last.error < best[0]:
                best = (k_last.error, z.copy(), y.copy(), Az.copy(), ATy.copy(), "iterate")
            if k_avg.error < best[0]:
                best = (k_avg.error, z_avg, y_avg, Az_avg, ATy_avg, "average")
            if status == "max_iter" and min(k_last.error, k_avg.error) <= tol:
                status = "optimal"

            # Restart decision (2023 paper, eqs. 29 and 30); no restart once the run has stopped.
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
                        threshold = beta * rho_ref
                        do_restart = rho_avg <= threshold
            z_pdhg, epoch_len = z, t
            if do_restart:
                z_prev, y_prev = z_start, y_start
                z, y, Az, ATy = z_avg, y_avg, Az_avg, ATy_avg  # fresh arrays (acc / t)
                z_start, y_start = z.copy(), y.copy()
                for acc in sums:
                    acc[:] = 0.0
                t = 0
                epoch += 1
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
                make_step(
                    k,
                    z,
                    y,
                    z_pdhg,
                    z_avg,
                    k_last,
                    k_avg,
                    step_size=tau_used,
                    restarted=do_restart,
                    epoch=epoch,
                    epoch_len=epoch_len,
                    normalized_gap=rho_avg,
                    restart_threshold=threshold,
                    matvecs=matvecs,
                )
            )

        err, zb, yb, Azb, ATyb, which = best
        kb = kkt_of(zb, yb, Azb, ATyb)
        z_out, y_out, _, _ = unscale(zb, yb, Azb, ATyb)
        if status == "optimal":
            message = (
                f"relative KKT error {err:.2e} ≤ tol={tol:g} at the {which} after {k} iterations "
                f"({matvecs} products with A or Aᵀ, {epoch} restarts)"
            )
        elif status == "nonfinite":
            message = f"stopped: non-finite iterate at iteration {k}"
        else:
            message = (
                f"reached max_iter={max_iter}; best relative KKT error {err:.2e} > tol={tol:g} "
                "(no infeasibility detection: the LP may be infeasible or unbounded)"
            )
        x_out = z_out[: std.n]
        return Result(
            method="restarted_pdhg",
            x=x_out,
            fun=fun_of(x_out),
            converged=status == "optimal",
            message=message,
            n_iter=k,
            trace=trace,
            extra={
                "status": status,
                "y": y_out,
                "output": which,
                "kkt": err,
                "primal_residual": kb.primal,
                "dual_residual": kb.dual,
                "gap": kb.gap,
                "n_matvec": matvecs,
                "n_restarts": epoch,
                "omega": omega,
                "eta": eta,
                "norm_A": norm_A,
            },
        )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("restarted_pdhg", "wyndor", {}),
    ("restarted_pdhg", "diet_2d", {}),
    ("restarted_pdhg", "klee_minty_3", {}),
    ("restarted_pdhg", "degenerate_2d", {"restart": "fixed", "restart_period": 32}),
    ("restarted_pdhg", "transport_small", {"primal_weight": "adaptive"}),
    ("restarted_pdhg", "wyndor", {"restart": "none", "max_iter": 300}),
]
