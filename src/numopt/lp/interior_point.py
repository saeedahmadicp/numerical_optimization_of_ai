"""Interior-point methods for linear programs.

Both methods work on the equality standard form

    min c̃ᵀz   s.t.   A z = b,   z ≥ 0,     A = [[A_ub, I], [A_eq, 0]],  b = [b_ub; b_eq],

with c̃ = c (min) or −c (max) on the original variables and 0 on the slacks; the original
variables are z[:n]. The iterates stay strictly inside the positive orthant, so they approach
the optimal face from its interior; the final x matches a simplex vertex only to the stopping
tolerance.

Scaling. ``# NOTE:`` both methods iterate on a scaled copy of the data (:func:`scaled_form`),
so that every tolerance test below is invariant under a change of units of the rows, of x and
of c (Wright, *Primal-Dual Interior-Point Methods*, 1997, §11.1, "scaling"):

    Â = R A D,   b̂ = R b / β,   ĉ = c̃ / γ,   z = β D ẑ,   λ = γ R λ̂,   s = γ D⁻¹ ŝ,

where R = diag(1/ρᵢ) with ρᵢ = maxⱼ≤ₙ |A_ij| (row equilibration on the original variables;
ρᵢ = |bᵢ| for a row without one, or 1 if bᵢ = 0 too), D = diag(1 on x, ρᵢ on the slack of
row i) keeps the slack columns equal to the identity, β = ‖R b‖∞ and γ = ‖c̃‖∞ (1 when
zero). Without it, the tests
"‖r_b‖/(1+‖b‖) ≤ tol" and "zᵀs/(1+|c̃ᵀz|) ≤ tol" become absolute when b or c is small, and
converged=True can come with a 1% error in x. Linearly dependent equality rows are then
removed (a dependent row whose right-hand side is inconsistent makes the LP infeasible).

Infeasibility and unboundedness. ``# NOTE:`` neither method uses a homogeneous self-dual
embedding, so they cannot *certify* these outcomes exactly. Both stop when an iterate gives an
approximate Farkas certificate in the scaled data (Wright 1997, ch. 9; Todd 2002, Acta Numer.):

* primal infeasibility: y = λ̂ / b̂ᵀλ̂ with b̂ᵀy = 1 and max(Âᵀy) ≤ ε. Then every feasible ẑ
  has 1 = yᵀÂẑ ≤ ε‖ẑ‖₁, so no feasible point with ‖ẑ‖₁ < 1/ε exists;
* dual infeasibility (a ray of descent): d = ẑ / (−ĉᵀẑ) ≥ 0 with ĉᵀd = −1 and ‖Âd‖∞ ≤ ε.
  Then no dual feasible point with ‖λ̂‖₁ < 1/ε exists. If an iterate was also feasible, the
  LP is unbounded. ``# NOTE:`` the ray often appears before the primal residual (or the
  artificial t) is at its tolerance, so the method keeps iterating while that measure still
  falls by the factor 0.9 per iteration. If it stops falling first, the method decides
  feasibility by a second run of the same method on the LP with the zero objective (which can
  only converge or detect infeasibility): ``"unbounded"`` if that run converges,
  ``"infeasible"`` if it detects infeasibility, else ``"infeasible_or_unbounded"``.
  ``Result.extra["feasibility_check"]`` holds the status of that run.

The certificate tolerance is ε = ``_CERT_TOL`` = 1e-8. This replaces a fixed limit on ‖z‖,
which labeled feasible, bounded LPs with large optima as "likely unbounded".

``Step.x`` is x (original variables), ``Step.fun`` = cᵀx in the original sense, and
``Step.step_size`` is the primal step length that produced the iterate (``None`` at k = 0).
Info values are in the original units unless stated otherwise.

Info keys:
    x: [float] — current iterate, original variables (same as ``Step.x``).
    mu: float — ``primal_dual_ipm``: the central-path measure μ = zᵀs / N.
        ``affine_scaling``: the complementarity zᵀr / N with the reduced costs r below.
    primal_residual: float — ‖A z − b‖₂ of the original constraints.
    dual_residual: float — ``primal_dual_ipm``: ‖Aᵀλ + s − c̃‖₂;
        ``affine_scaling``: ‖min(r, 0)‖₂ (violation of dual feasibility r ≥ 0).
    gap: float — zᵀs (``primal_dual_ipm``) or zᵀr (``affine_scaling``).
    alpha_primal, alpha_dual: float | None — (``primal_dual_ipm``) step lengths of the
        corrector step that produced this iterate.
    alpha_aff_primal, alpha_aff_dual: float | None — (``primal_dual_ipm``) predictor
        (affine-scaling) step lengths of that step.
    sigma: float | None — (``primal_dual_ipm``) centering parameter σ = (μ_aff/μ)³.
    x_affine: [float] | None — (``primal_dual_ipm``) predictor point z + α_aff Δz_aff,
        original variables, computed from the previous iterate.
    alpha: float | None — (``affine_scaling``) step length along −Ẑ²r̂ (scaled data) that
        produced the iterate.
    direction: [float] | None — (``affine_scaling``) that step, original variables.
    artificial: float — (``affine_scaling``) value of the big-M artificial variable t of the
        scaled problem (the iterate is feasible for the original LP when t = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import LinearProgram, Result, Step
from .simplex import check_lp, lp_arrays

Matrix = np.ndarray

#: Tolerance ε of the approximate Farkas certificates (scaled data; module docstring).
_CERT_TOL = 1e-8

#: primal_dual_ipm stops ("stalled") after this many consecutive long steps (α_p ≥ ½) after
#: which the primal residual did not fall by half; it must fall by 1 − α_p in exact arithmetic.
_LOST_STEPS = 5

#: A ray of descent whose classification is still open (no feasible iterate yet) is followed
#: while the feasibility measure (relative primal residual, or the artificial t) still falls
#: by this factor per iteration; the ambiguous status is reported once it stops falling.
_PROGRESS = 0.9

#: affine_scaling: the original constraints count as satisfied when the artificial part of
#: the residual, t·‖r̂₀‖, is ≤ this multiple of (1 + ‖b̂‖). At a big-M optimum t·r_t ≤ zᵀr with
#: r_t ≈ big_m, so a feasible LP gives t·‖r̂₀‖ many orders below this bound; an infeasible LP
#: keeps t bounded away from 0 (the phase-1 minimum).
_FEAS_RTOL = 1e-6


def equality_form(lp: LinearProgram) -> tuple[Matrix, np.ndarray, np.ndarray, int, float]:
    """Return ``A, b, c̃, n, sign`` of the equality standard form (unscaled, all rows)."""
    c, a_ub, b_ub, a_eq, b_eq = lp_arrays(lp)
    n, m_ub, m_eq = c.size, b_ub.size, b_eq.size
    A = np.zeros((m_ub + m_eq, n + m_ub))
    A[:m_ub, :n] = a_ub
    A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = a_eq
    b = np.concatenate([b_ub, b_eq])
    sign = 1.0 if lp.sense == "min" else -1.0
    return A, b, np.concatenate([sign * c, np.zeros(m_ub)]), n, sign


def independent_rows(A: Matrix, b: np.ndarray) -> tuple[list[int], bool]:
    """Greedy row selection: keep a row when it raises the rank of the kept rows.

    Returns the kept row indices and ``consistent``: False when a dropped row aᵢ = wᵀA_R has
    bᵢ ≠ wᵀb_R, i.e. the system A z = b has no solution.
    """
    keep: list[int] = []
    consistent = True
    scale = 1.0 + (float(np.max(np.abs(A))) if A.size else 0.0)
    tol = max(A.shape) * np.finfo(np.float64).eps * scale * 1e3
    for i in range(A.shape[0]):
        trial = [*keep, i]
        if np.linalg.matrix_rank(A[trial], tol=tol) == len(trial):
            keep.append(i)
            continue
        w, *_ = np.linalg.lstsq(A[keep].T, A[i], rcond=None)
        if abs(b[i] - w @ b[keep]) > 1e-9 * (
            1.0 + abs(b[i]) + float(np.abs(b[keep]).max(initial=0.0))
        ):
            consistent = False
    return keep, consistent


@dataclass(frozen=True)
class ScaledForm:
    """The scaled equality form min ĉᵀẑ s.t. Â ẑ = b̂, ẑ ≥ 0 (module docstring, "Scaling").

    ``A``, ``b`` hold the linearly independent rows only; ``A_full``, ``b_full`` are the
    unscaled rows of :func:`equality_form`, used to report residuals in the original units.
    """

    A: Matrix  # (m, N) Â, kept rows
    b: np.ndarray  # (m,) b̂, kept rows
    c: np.ndarray  # (N,) ĉ
    n: int
    sign: float
    col: np.ndarray  # (N,) diagonal of D: z = β·col·ẑ, s = γ·ŝ/col
    row: np.ndarray  # (m,) ρᵢ of the kept rows: λ = γ·λ̂/ρ
    beta: float
    gamma: float
    A_full: Matrix
    b_full: np.ndarray
    consistent: bool

    def z(self, z_hat: np.ndarray) -> np.ndarray:
        """Original z = β D ẑ."""
        return self.beta * self.col * z_hat

    def x(self, z_hat: np.ndarray) -> np.ndarray:
        """Original variables x = β ẑ[:n] (D = 1 on x)."""
        return self.beta * z_hat[: self.n]


def scaled_form(lp: LinearProgram) -> ScaledForm:
    """Row-equilibrate, scale x by β and c by γ, and drop dependent rows (module docstring)."""
    A_full, b_full, c_full, n, sign = equality_form(lp)
    m, N = A_full.shape
    m_ub = N - n
    rho = np.max(np.abs(A_full[:, :n]), axis=1) if m else np.zeros(0)
    # A row with no entry on x (0·x ≤ bᵢ or 0·x = bᵢ) is scaled by |bᵢ| so that it does not
    # set β by itself.
    tiny = np.finfo(np.float64).tiny  # a subnormal bᵢ would overflow 1/ρᵢ
    rho = np.where(rho > 0.0, rho, np.where(np.abs(b_full) >= tiny, np.abs(b_full), 1.0))
    col = np.ones(N)
    col[n:] = rho[:m_ub]
    A_hat = A_full * col[None, :] / rho[:, None]  # slack entries ρᵢ/ρᵢ = 1 exactly
    b_row = b_full / rho
    beta = float(np.max(np.abs(b_row))) if m else 0.0
    beta = beta if beta > 0.0 else 1.0
    gamma = float(np.max(np.abs(c_full)))
    gamma = gamma if gamma > 0.0 else 1.0
    b_hat = b_row / beta
    keep, consistent = independent_rows(A_hat, b_hat)
    return ScaledForm(
        A=A_hat[keep],
        b=b_hat[keep],
        c=c_full / gamma,
        n=n,
        sign=sign,
        col=col,
        row=rho[keep],
        beta=beta,
        gamma=gamma,
        A_full=A_full,
        b_full=b_full,
        consistent=consistent,
    )


def _norm(v: np.ndarray) -> float:
    """‖v‖₂ without overflow of the squares: m·‖v/m‖₂ with m = ‖v‖∞ (the scaling device of
    LAPACK's xNRM2). Info values in the original units can exceed 1e154 when a row is tiny."""
    m = float(np.max(np.abs(v), initial=0.0))
    if m == 0.0 or not np.isfinite(m):
        return m
    return m * float(np.linalg.norm(v / m))


def _cholesky(M: Matrix) -> Matrix | None:
    """Cholesky factor of the SPD normal matrix; one retry with a tiny diagonal shift."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        pass
    # NOTE: near a degenerate optimum A D² Aᵀ can lose definiteness to rounding; a shift of
    # 1e-14·max diag (far below the stopping tolerance) restores it (Wright 1997, §11.1).
    shift = 1e-14 * max(1.0, float(np.max(np.diag(M)))) if M.size else 0.0
    try:
        return np.linalg.cholesky(M + shift * np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
        return None


def _chol_solve(L: Matrix, r: np.ndarray) -> np.ndarray:
    """Solve L Lᵀ y = r by forward and back substitution."""
    m = L.shape[0]
    y = np.array(r, dtype=np.float64)
    for i in range(m):
        y[i] = (y[i] - L[i, :i] @ y[:i]) / L[i, i]
    for i in range(m - 1, -1, -1):
        y[i] = (y[i] - L[i + 1 :, i] @ y[i + 1 :]) / L[i, i]
    return y


def _newton(
    A: Matrix,
    L: Matrix,
    z: np.ndarray,
    s: np.ndarray,
    r_b: np.ndarray,
    r_c: np.ndarray,
    r_zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve [0 Aᵀ I; A 0 0; S 0 Z](Δz, Δλ, Δs) = (−r_c, −r_b, r_zs) by the normal equations.

    Eliminating Δs = −r_c − AᵀΔλ and Δz = S⁻¹(r_zs − ZΔs) leaves
    A Z S⁻¹ Aᵀ Δλ = −r_b − A S⁻¹(r_zs + Z r_c), solved with the Cholesky factor L of
    A Z S⁻¹ Aᵀ (Nocedal & Wright §14.2, "normal-equations form").
    """
    dlam = _chol_solve(L, -r_b - A @ ((r_zs + z * r_c) / s))
    ds = -r_c - A.T @ dlam
    dz = (r_zs - z * ds) / s
    return dz, dlam, ds


def _max_step(v: np.ndarray, dv: np.ndarray) -> float:
    """Largest α ∈ [0, 1] with v + α dv ≥ 0 (v > 0)."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _farkas_primal(A: Matrix, b: np.ndarray, lam: np.ndarray) -> bool:
    """True when y = λ/bᵀλ has bᵀy = 1 and max(Aᵀy) ≤ ``_CERT_TOL`` (primal infeasible)."""
    b_lam = float(b @ lam)
    return b_lam > 0.0 and float(np.max(A.T @ lam)) <= _CERT_TOL * b_lam


def _descent_ray(A: Matrix, c: np.ndarray, z: np.ndarray) -> bool:
    """True when d = z/(−cᵀz) ≥ 0 has cᵀd = −1 and ‖A d‖∞ ≤ ``_CERT_TOL`` (dual infeasible)."""
    cz = float(c @ z)
    return cz < 0.0 and float(np.max(np.abs(A @ z), initial=0.0)) <= _CERT_TOL * -cz


_INCONSISTENT = (
    "LP is infeasible: the equality rows are linearly dependent with inconsistent right-hand sides"
)


def _resolve_ray(check: Result, ray_message: str) -> tuple[str, str]:
    """Status and message for an LP with a ray of descent, from a zero-objective run."""
    if check.converged:
        return "unbounded", (
            f"LP is unbounded: {ray_message}, and the LP is feasible (the same method on the "
            "zero objective converged)"
        )
    if check.extra.get("status") == "infeasible":
        return "infeasible", (
            f"LP is infeasible (the same method on the zero objective: {check.message}); "
            f"it also has {ray_message}"
        )
    return "infeasible_or_unbounded", (
        f"LP is unbounded or infeasible: {ray_message}, but feasibility is undecided (the "
        f"same method on the zero objective: {check.message})"
    )


def _zero_objective(lp: LinearProgram) -> LinearProgram:
    return replace(lp, c=np.zeros(np.size(lp.c)), sense="min", integer=())


_RAY = "an approximate ray of descent d (cᵀd = −1, ‖A d‖∞ ≤ 1e-8, scaled data)"
_FARKAS = (
    "LP is infeasible: approximate Farkas certificate y = λ/bᵀλ with bᵀy = 1 and "
    "max(Aᵀy) ≤ 1e-8 (scaled data), so no feasible point has ‖ẑ‖₁ < 1e8"
)


# --------------------------------------------------------------------------------------
# Mehrotra predictor–corrector
# --------------------------------------------------------------------------------------


@register(
    id="primal_dual_ipm",
    family="lp",
    name="Primal–dual interior point (Mehrotra)",
    params=(
        ParamSpec(
            "tol",
            1e-8,
            min=1e-12,
            max=1e-3,
            log=True,
            help="Stop when the relative primal residual, dual residual and gap of the scaled LP are all ≤ tol.",
        ),
        ParamSpec(
            "eta",
            0.99,
            min=0.5,
            max=0.9999,
            help="Fraction of the step to the boundary (α = min(1, η·α_max)).",
        ),
        ParamSpec("max_iter", 100, kind="int", min=1, max=1000, help="Iteration limit."),
    ),
    needs=("lp",),
    order="superlinear in practice (polynomial bound for path-following variants)",
    summary="Follow the central path with Newton steps on the perturbed KKT conditions, using an affine predictor and a centering corrector.",
    references=(
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Algorithm 14.3 and §14.2",
        "Mehrotra (1992), SIAM J. Optim. 2(4):575–601",
        "Wright, Primal-Dual Interior-Point Methods (1997), ch. 9–11",
    ),
)
def primal_dual_ipm(
    problem: LinearProgram,
    *,
    tol: float = 1e-8,
    eta: float = 0.99,
    max_iter: int = 100,
) -> Result:
    """Mehrotra's predictor–corrector primal–dual method (Nocedal & Wright, Algorithm 14.3).

    Runs on the scaled LP min ĉᵀẑ s.t. Â ẑ = b̂, ẑ ≥ 0 (module docstring; hats dropped
    below). KKT conditions: Aᵀλ + s = c, A z = b, ZSe = 0, (z, s) ≥ 0. Each iteration from
    (z, λ, s) with z, s > 0 and μ = zᵀs/N:

    1. Predictor: solve the Newton system [0 Aᵀ I; A 0 0; S 0 Z](Δz, Δλ, Δs) =
       (−r_c, −r_b, −ZSe) with r_b = Az − b, r_c = Aᵀλ + s − c, by the normal equations
       A Z S⁻¹ Aᵀ Δλ = −r_b − A S⁻¹(r_zs + Z r_c) (one Cholesky factor, reused in step 3).
    2. α_aff (primal, dual) = largest steps in [0, 1] keeping z, s ≥ 0;
       μ_aff = (z + α_aff Δz)ᵀ(s + α_aff Δs)/N, σ = (μ_aff/μ)³.
    3. Corrector: same system with third block −ZSe − ΔZ_aff ΔS_aff e + σμe.
    4. α_pri = min(1, η α_max,pri), α_dual = min(1, η α_max,dual) (separate step lengths);
       z ← z + α_pri Δz, (λ, s) ← (λ, s) + α_dual (Δλ, Δs).

    The infeasible starting point is Mehrotra's heuristic (Nocedal & Wright §14.2, "Starting
    point"): z̃ = Aᵀ(AAᵀ)⁻¹b, λ̃ = (AAᵀ)⁻¹Ac, s̃ = c − Aᵀλ̃, shifted to be positive and
    balanced.

    Stops (converged) when ‖r_b‖/(1+‖b‖) ≤ tol, ‖r_c‖/(1+‖c‖) ≤ tol and zᵀs/(1+|cᵀz|) ≤ tol
    on the scaled data. Stops with ``converged=False`` on an approximate Farkas certificate
    (module docstring): ``"infeasible"`` (primal), ``"unbounded"`` (a ray of descent and a
    feasible point) or ``"infeasible_or_unbounded"`` (a ray of descent, feasibility
    undecided); also when steps stall (steps below 1e-12, or 5 long steps in a row that do
    not reduce the primal residual by half although A z = b is linear), on a non-finite value,
    a singular normal matrix, or at max_iter. ``# NOTE:`` after such a breakdown with no
    feasible iterate, a zero-objective run decides feasibility, and the status becomes
    ``"infeasible"`` when that run detects infeasibility (an infeasible-start method without
    a self-dual embedding can stall at a complementary infeasible point).
    """
    lp = check_lp(problem)
    if not 0 < eta < 1:
        raise ValueError("eta must be in (0, 1)")
    if not tol > 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter ≥ 1")
    sf = scaled_form(lp)
    A, b, c = sf.A, sf.b, sf.c
    c_orig = np.asarray(lp.c, dtype=np.float64)
    N = c.size
    norm_b, norm_c = float(np.linalg.norm(b)), float(np.linalg.norm(c))
    bg = sf.beta * sf.gamma  # zᵀs (original) = βγ ẑᵀŝ
    trace: list[Step] = []

    def objective(z_hat: np.ndarray) -> float:
        return float(c_orig @ sf.x(z_hat))

    def base_info(z_hat: np.ndarray) -> dict[str, Any]:
        r = sf.A_full @ sf.z(z_hat) - sf.b_full
        return {
            "x": sf.x(z_hat),
            "primal_residual": _norm(r),
        }

    if not sf.consistent:
        z0 = np.zeros(N)
        trace.append(Step(0, sf.x(z0), objective(z0), info=base_info(z0)))
        return Result(
            "primal_dual_ipm",
            sf.x(z0),
            objective(z0),
            False,
            _INCONSISTENT,
            0,
            trace=trace,
            extra={"status": "infeasible"},
        )

    # Starting point (Nocedal & Wright §14.2).
    L0 = _cholesky(A @ A.T)
    if L0 is None:
        raise ValueError(f"{lp.id}: A has dependent rows that could not be removed")
    z = A.T @ _chol_solve(L0, b)
    lam = _chol_solve(L0, A @ c)
    s = c - A.T @ lam
    z = z + max(-1.5 * float(np.min(z)), 0.0)
    s = s + max(-1.5 * float(np.min(s)), 0.0)
    # Both shifts of N&W (14.42) use the same (ẑ, ŝ): δ̂z = ½ẑᵀŝ/eᵀŝ, δ̂s = ½ẑᵀŝ/eᵀẑ.
    zs = float(z @ s)
    dz_shift = 0.5 * zs / max(float(np.sum(s)), 1e-300)
    ds_shift = 0.5 * zs / max(float(np.sum(z)), 1e-300)
    z, s = z + dz_shift, s + ds_shift
    if not (np.all(z > 0) and np.all(s > 0)):
        # NOTE: the heuristic gives z = 0 or s = 0 only in degenerate cases (e.g. b = 0 and
        # c = 0); fall back to the all-ones start then.
        z, s = np.ones(N), np.ones(N)

    last: dict[str, Any] = {
        "alpha_primal": None,
        "alpha_dual": None,
        "alpha_aff_primal": None,
        "alpha_aff_dual": None,
        "sigma": None,
        "x_affine": None,
    }
    alpha_last: float | None = None
    feasible_seen = False  # some iterate had ‖r_b‖/(1+‖b‖) ≤ tol
    ray_seen = False  # some iterate gave a ray of descent while no iterate was feasible
    rel_p_prev = np.inf
    lost = 0  # consecutive long steps after which r_b did not fall (see _LOST_STEPS)
    status, message = "max_iter", f"reached max_iter={max_iter}"
    k = 0
    while True:
        r_b = A @ z - b
        r_c = A.T @ lam + s - c
        mu = float(z @ s) / N
        rel_p = float(np.linalg.norm(r_b)) / (1.0 + norm_b)
        rel_d = float(np.linalg.norm(r_c)) / (1.0 + norm_c)
        rel_gap = float(z @ s) / (1.0 + abs(float(c @ z)))
        with np.errstate(over="ignore"):  # see the NOTE in affine_scaling's step_info
            dual_res = _norm(sf.gamma * r_c / sf.col)
        info = {
            **base_info(z),
            "mu": bg * mu,
            "dual_residual": dual_res,
            "gap": bg * float(z @ s),
            **last,
        }
        trace.append(Step(k, sf.x(z), objective(z), step_size=alpha_last, info=info))
        if rel_p <= tol and rel_d <= tol and rel_gap <= tol:
            status, message = "optimal", "optimal: relative residuals and gap ≤ tol"
            break
        if _farkas_primal(A, b, lam):
            status, message = "infeasible", _FARKAS
            break
        feasible_seen = feasible_seen or rel_p <= tol
        if _descent_ray(A, c, z):
            if feasible_seen:
                status = "unbounded"
                message = (
                    "LP is unbounded: an iterate was feasible (relative residual ≤ tol) and "
                    "d = z/(−cᵀz) ≥ 0 has cᵀd = −1, ‖A d‖∞ ≤ 1e-8 (scaled data): an "
                    "approximate ray of descent"
                )
                break
            ray_seen = True
            if rel_p > _PROGRESS * rel_p_prev:
                status = "ray"
                break
        # A z = b is linear, so a step α_p gives r_b ← (1 − α_p) r_b exactly; a long step
        # that leaves r_b ≥ ½ r_b means the normal equations lost their accuracy.
        long_step = k > 0 and last["alpha_primal"] >= 0.5
        lost = lost + 1 if long_step and rel_p > tol and rel_p > 0.5 * rel_p_prev else 0
        rel_p_prev = rel_p
        if k > 0 and max(last["alpha_primal"], last["alpha_dual"]) < 1e-12:
            status, message = "stalled", "step lengths below 1e-12: the method stalled"
            break
        if lost >= _LOST_STEPS:
            status = "stalled"
            message = (
                f"the primal residual did not fall in {_LOST_STEPS} long steps (it must fall by "
                "the factor 1 − α): the normal equations A Z S⁻¹ Aᵀ lost their accuracy"
            )
            break
        if k >= max_iter:
            break
        d = z / s  # D² = Z S⁻¹
        L = _cholesky((A * d) @ A.T) if A.shape[0] else np.zeros((0, 0))
        if L is None:
            status, message = "singular", "normal matrix A Z S⁻¹ Aᵀ is not positive definite"
            break

        dz_a, _, ds_a = _newton(A, L, z, s, r_b, r_c, -z * s)
        a_aff_p, a_aff_d = _max_step(z, dz_a), _max_step(s, ds_a)
        mu_aff = float((z + a_aff_p * dz_a) @ (s + a_aff_d * ds_a)) / N
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
        dz, dlam, ds = _newton(A, L, z, s, r_b, r_c, -z * s - dz_a * ds_a + sigma * mu)
        a_p = min(1.0, eta * _max_step(z, dz))
        a_d = min(1.0, eta * _max_step(s, ds))
        x_aff = sf.x(z + a_aff_p * dz_a)
        z = z + a_p * dz
        lam = lam + a_d * dlam
        s = s + a_d * ds
        k += 1
        alpha_last = a_p
        last = {
            "alpha_primal": a_p,
            "alpha_dual": a_d,
            "alpha_aff_primal": a_aff_p,
            "alpha_aff_dual": a_aff_d,
            "sigma": sigma,
            "x_affine": x_aff,
        }
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(lam)) and np.all(np.isfinite(s))):
            status, message = "non_finite", "non-finite iterate"
            trace.append(Step(k, sf.x(z), None, step_size=a_p, info={"x": sf.x(z), **last}))
            break

    extra: dict[str, Any] = {}
    breakdown = status in ("stalled", "singular", "non_finite")
    ray_case = status == "ray" or (ray_seen and (breakdown or status == "max_iter"))
    if ray_case or (breakdown and not feasible_seen and np.any(c != 0.0)):
        # Decide feasibility with a second run on the zero objective (module docstring).
        check = primal_dual_ipm(_zero_objective(lp), tol=tol, eta=eta, max_iter=max_iter)
        extra["feasibility_check"] = check.extra["status"]
        if ray_case:
            status, message = _resolve_ray(check, _RAY)
        elif check.extra["status"] == "infeasible":
            status = "infeasible"
            message = (
                f"LP is infeasible (the same method on the zero objective: {check.message}); "
                f"the run on the LP itself stopped: {message}"
            )
    final = trace[-1]
    return Result(
        method="primal_dual_ipm",
        x=np.asarray(final.x, dtype=np.float64),
        fun=final.fun,
        converged=status == "optimal",
        message=message,
        n_iter=final.k,
        trace=trace,
        extra={
            "status": status,
            "lambda": sf.gamma * lam / sf.row,
            "s": sf.gamma * s / sf.col,
            "z": sf.z(z),
            **extra,
        },
    )


# --------------------------------------------------------------------------------------
# Primal affine scaling
# --------------------------------------------------------------------------------------


def _scaled_reduced_costs(A: Matrix, z: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Return Zr = P Z c, the projection of Z·cost onto null(A Z) (Bertsimas & Tsitsiklis §9.2).

    With w = (A Z² Aᵀ)⁻¹ A Z² c the reduced costs r = c − Aᵀw satisfy Z r = (I − Q Qᵀ) Z c,
    where Q is an orthonormal basis of range(Z Aᵀ) (Householder QR). The projection is applied
    twice ("twice is enough": Parlett, *The Symmetric Eigenvalue Problem*, §6.9).
    # NOTE: the step is −α Z(Zr) with α = β / maxᵢ zᵢrᵢ ≫ 1 near the optimum; a dual estimate
    # from one least-squares solve has an error ∝ ‖Z c‖, which α amplifies into a drift of
    # A z = b (measured up to 1e-4 on an LP without interior). After two projections the
    # error is ∝ ‖Zr‖, so α·‖A Δz‖ stays at rounding level (tests/test_lp_interior_point.py).
    """
    p = z * cost
    if A.shape[0] == 0:
        return p
    Q, _ = np.linalg.qr((A * z).T)  # (N, m), orthonormal columns spanning range(Z Aᵀ)
    for _ in range(2):
        p = p - Q @ (Q.T @ p)
    return p


@register(
    id="affine_scaling",
    family="lp",
    name="Affine scaling (Dikin)",
    params=(
        ParamSpec(
            "beta",
            0.66,
            min=0.05,
            max=0.99,
            help="Step fraction: of the distance to the boundary (long) or of the Dikin ellipsoid radius (short). β ≤ 2/3 guarantees convergence.",
        ),
        ParamSpec(
            "variant",
            "long",
            kind="choice",
            choices=("long", "short"),
            help="long: α = β / maxᵢ zᵢrᵢ (Vanderbei et al.); short: α = β / ‖Zr‖₂ (Dikin).",
        ),
        ParamSpec(
            "big_m",
            1e6,
            min=10.0,
            max=1e9,
            log=True,
            help="Cost of the artificial variable that makes z = 1 an interior start (relative to the scaled costs, ‖ĉ‖∞ = 1).",
        ),
        ParamSpec(
            "tol",
            1e-8,
            min=1e-12,
            max=1e-3,
            log=True,
            help="Stop when r ≥ −tol·(1+‖c‖∞) and zᵀr ≤ tol·(1+|cᵀz|) on the scaled LP.",
        ),
        ParamSpec("max_iter", 500, kind="int", min=1, max=10_000, help="Iteration limit."),
    ),
    needs=("lp",),
    order="linear",
    summary="Rescale so the current point is the center of the orthant, then step along the projected steepest-descent direction.",
    references=(
        "Dikin (1967), Soviet Math. Dokl. 8:674–675",
        "Vanderbei, Meketon & Freedman (1986), Algorithmica 1:395–407",
        "Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §9.2",
        "Hall & Vanderbei (1993), 'Two-thirds is sharp for affine scaling', Oper. Res. Lett. 13:197–201",
    ),
)
def affine_scaling(
    problem: LinearProgram,
    *,
    beta: float = 0.66,
    variant: str = "long",
    big_m: float = 1e6,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> Result:
    """Primal affine-scaling algorithm (Bertsimas & Tsitsiklis 1997, §9.2).

    Runs on the scaled LP min ĉᵀẑ s.t. Â ẑ = b̂, ẑ ≥ 0 (module docstring; hats dropped
    below). Interior start: z = e (all ones) and one artificial variable t = 1 with column
    r₀ = b − A e and cost ``big_m``, so that A z + r₀ t = b holds at (e, 1). With
    Z = diag(z) (t included) each iteration computes

    1. dual estimate w = (A Z² Aᵀ)⁻¹ A Z² c and reduced costs r = c − Aᵀw, computed as
       Z r = (I − QQᵀ) Z c with Q an orthonormal basis of range(Z Aᵀ) ``# NOTE:`` (Householder
       QR, projection applied twice; A Z² Aᵀ is never formed, see
       :func:`_scaled_reduced_costs`);
    2. optimality: stop when r ≥ −tol·(1+‖c‖∞) and zᵀr ≤ tol·(1+|cᵀz|);
    3. unboundedness: if Z²r ≤ 0 (no zᵢrᵢ > 0) the direction −Z²r is a ray of descent;
    4. z ← z − α Z²r with α = β / maxᵢ zᵢrᵢ (``long``, Vanderbei–Meketon–Freedman; β < 1 is
       the fraction of the step to the boundary) or α = β / ‖Zr‖₂ (``short``, Dikin's
       ellipsoid step).

    Stops (converged) at step 2 when also t·‖r₀‖ ≤ 1e-6·(1+‖b‖) (the original constraints
    hold). An optimum with t > 0 means the LP is infeasible or ``big_m`` is too small
    (status ``"infeasible"``). A ray of descent d of the big-M problem, exact (step 3) or an
    approximate certificate (module docstring), is classified by its parts: when ‖A d_z‖∞ ≤ ε
    the original LP has the ray d_z, so the LP is ``"unbounded"`` if the iterate is feasible
    (t ≈ 0); otherwise feasibility is decided by a zero-objective run (module docstring); when
    the ray needs t, the big-M problem is unbounded only through the artificial, and the
    zero-objective run gives ``"infeasible"`` or ``"big_m_too_small"`` (LP feasible; big_m too
    small for the data, or the LP unbounded), or ``"infeasible_or_big_m_too_small"`` when it
    decides nothing. An approximate ray of the LP with t > 0 is
    followed while t still falls by the factor 0.9 per iteration (module docstring), and a
    ray through t while t still decreases.
    Non-finite values and max_iter also give ``converged=False``.
    """
    lp = check_lp(problem)
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    if variant not in ("long", "short"):
        raise ValueError("variant must be 'long' or 'short'")
    if not big_m > 0 or not tol > 0 or max_iter < 1:
        raise ValueError("big_m and tol must be positive and max_iter ≥ 1")
    sf = scaled_form(lp)
    A0, b, c = sf.A, sf.b, sf.c
    c_orig = np.asarray(lp.c, dtype=np.float64)
    N0 = c.size
    r0 = b - A0 @ np.ones(N0)
    A = np.column_stack([A0, r0])  # last column: artificial t
    c_aug = np.concatenate([c, [big_m]])
    col_aug = np.concatenate([sf.col, [1.0]])
    z = np.ones(N0 + 1)
    norm_c = float(np.max(np.abs(c))) if c.size else 0.0
    norm_r0 = float(np.linalg.norm(r0))
    feas_tol = _FEAS_RTOL * (1.0 + float(np.linalg.norm(b)))
    bg = sf.beta * sf.gamma
    trace: list[Step] = []

    def objective(v: np.ndarray) -> float:
        return float(c_orig @ sf.x(v))

    def step_info(v: np.ndarray, r: np.ndarray, alpha: float | None, dx: Any) -> dict[str, Any]:
        # NOTE: r / col can exceed the float range for a row with a tiny ρᵢ; inf is then the
        # correctly rounded value of the original-unit residual (exported as "inf").
        with np.errstate(over="ignore"):
            dual_res = _norm(sf.gamma * np.minimum(r / col_aug, 0.0))
        return {
            "x": sf.x(v),
            "mu": bg * float(v @ r) / v.size,
            "primal_residual": _norm(sf.A_full @ sf.z(v[:N0]) - sf.b_full),
            "dual_residual": dual_res,
            "gap": bg * float(v @ r),
            "alpha": alpha,
            "direction": dx,
            "artificial": float(v[-1]),
        }

    def classify_ray(d: np.ndarray) -> tuple[str, str]:
        """Status and message for a ray d ≥ 0 of the big-M problem with c_augᵀd = −1."""
        if float(np.max(np.abs(A0 @ d[:-1]), initial=0.0)) <= _CERT_TOL:
            if z[-1] * norm_r0 <= feas_tol:
                return "unbounded", (
                    "LP is unbounded: the iterate is feasible (t ≈ 0) and the big-M problem "
                    "has a ray of descent d with ‖A d_z‖∞ ≤ 1e-8 (scaled data), a ray of the LP"
                )
            return "infeasible_or_unbounded", (
                f"LP is unbounded or infeasible: the LP has an approximate ray of descent "
                f"(‖A d_z‖∞ ≤ 1e-8, scaled data), but the artificial t = {z[-1]:.3g} > 0, so "
                "LP feasibility is not established"
            )
        return "infeasible_or_big_m_too_small", (
            f"the big-M problem is unbounded only through the artificial t = {z[-1]:.3g}: "
            "the LP is infeasible, or big_m is too small for the data"
        )

    if not sf.consistent:
        v = np.zeros(N0 + 1)
        trace.append(
            Step(0, sf.x(v), objective(v), info=step_info(v, np.zeros_like(v), None, None))
        )
        return Result(
            "affine_scaling",
            sf.x(v),
            objective(v),
            False,
            _INCONSISTENT,
            0,
            trace=trace,
            extra={"status": "infeasible", "artificial": 0.0},
        )

    alpha: float | None = None
    dx_last: Any = None
    ray_open: tuple[str, str] | None = None  # classification of a ray still being followed
    status, message = "max_iter", f"reached max_iter={max_iter}"
    k = 0
    while True:
        zr = _scaled_reduced_costs(A, z, c_aug)  # zᵢrᵢ
        r = zr / z
        trace.append(
            Step(k, sf.x(z), objective(z), step_size=alpha, info=step_info(z, r, alpha, dx_last))
        )
        gap = float(z @ r)
        dual_ok = float(np.min(r)) >= -tol * (1.0 + norm_c)
        if dual_ok and gap <= tol * (1.0 + abs(float(c @ z[:N0]))):
            if z[-1] * norm_r0 <= feas_tol:
                status, message = "optimal", "optimal: r ≥ −tol and zᵀr ≤ tol (B&T §9.2 step 3)"
            else:
                status = "infeasible"
                message = (
                    f"optimal for the big-M problem but the artificial t = {z[-1]:.3g} > 0: "
                    "the LP is infeasible (or big_m is too small)"
                )
            break
        if float(np.max(zr)) <= 0.0:
            # −Z²r ≥ 0 is an exact ray of the big-M problem with c_augᵀ(−Z²r) = −‖Zr‖² < 0.
            ray = -z * zr
            status, message = classify_ray(ray / -float(c_aug @ ray))
            break
        if k >= max_iter:
            break
        alpha = beta / float(np.max(zr)) if variant == "long" else beta / float(np.linalg.norm(zr))
        dz = -alpha * z * zr  # −α Z² r
        t_prev = float(z[-1])
        z = z + dz
        dx_last = sf.x(dz)
        k += 1
        if not np.all(np.isfinite(z)):
            status, message = "non_finite", "non-finite iterate"
            trace.append(Step(k, sf.x(z), None, step_size=alpha, info={"x": sf.x(z)}))
            break
        ray_open = classify_ray(z / -float(c_aug @ z)) if _descent_ray(A, c_aug, z) else None
        # Finalize a ray when it is decisive or t stopped falling: "unbounded" at once; a ray
        # of the LP with t > 0 once t falls slower than _PROGRESS; a ray through t only once t
        # no longer decreases (while t falls, the big-M cost still drives it to 0).
        finalize = ray_open is not None and (
            ray_open[0] == "unbounded"
            or (ray_open[0] == "infeasible_or_unbounded" and z[-1] > _PROGRESS * t_prev)
            or (ray_open[0] == "infeasible_or_big_m_too_small" and z[-1] >= t_prev)
        )
        if ray_open is not None and finalize:
            status, message = ray_open
            if status == "infeasible_or_unbounded":
                status = "ray"
            zr = _scaled_reduced_costs(A, z, c_aug)
            trace.append(
                Step(
                    k,
                    sf.x(z),
                    objective(z),
                    step_size=alpha,
                    info=step_info(z, zr / z, alpha, dx_last),
                )
            )
            break

    if ray_open is not None and status in ("max_iter", "non_finite"):
        status, message = ray_open[0], f"{ray_open[1]} (then: {message})"
    extra: dict[str, Any] = {}
    if status in ("ray", "infeasible_or_unbounded", "infeasible_or_big_m_too_small"):
        # Decide feasibility with a second run on the zero objective (module docstring); its
        # big-M problem min M·t is bounded, so it ends optimal (t = 0) or infeasible (t > 0).
        check = affine_scaling(
            _zero_objective(lp), beta=beta, variant=variant, big_m=big_m, tol=tol, max_iter=max_iter
        )
        extra["feasibility_check"] = check.extra["status"]
        if status != "infeasible_or_big_m_too_small":
            status, message = _resolve_ray(check, _RAY)
        elif check.extra["status"] == "infeasible":
            status, message = (
                "infeasible",
                (
                    f"LP is infeasible (the same method on the zero objective: {check.message}); "
                    f"before that: {message}"
                ),
            )
        elif check.converged:
            status, message = (
                "big_m_too_small",
                (
                    "the LP is feasible (the same method on the zero objective converged), but the "
                    f"big-M problem is unbounded through the artificial t = {z[-1]:.3g}: big_m is "
                    "too small for the data, or the LP is unbounded"
                ),
            )
    final = trace[-1]
    return Result(
        method="affine_scaling",
        x=np.asarray(final.x, dtype=np.float64),
        fun=final.fun,
        converged=status == "optimal",
        message=message,
        n_iter=final.k,
        trace=trace,
        extra={"status": status, "artificial": float(z[-1]), **extra},
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("primal_dual_ipm", "wyndor", {}),
    ("primal_dual_ipm", "diet_2d", {}),
    ("primal_dual_ipm", "transport_small", {}),
    ("affine_scaling", "wyndor", {}),
    ("affine_scaling", "diet_2d", {"variant": "short", "beta": 0.5}),
    ("affine_scaling", "degenerate_2d", {}),
]
