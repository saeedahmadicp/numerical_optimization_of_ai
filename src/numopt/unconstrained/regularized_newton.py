"""Globalized Newton without a line search: adaptive cubic regularization (ARC) and
gradient-regularized Newton, for min f(x), f: ℝⁿ → ℝ twice differentiable.

Both methods regularize the Newton model instead of searching along a direction.

``arc``         Adaptive Regularisation using Cubics, Cartis, Gould & Toint (2011a), Algorithm 2.1.
                At x_k the step s_k is the *global* minimizer of the cubic model (eq. 1.4)

                    m_k(s) = f_k + g_kᵀs + ½ sᵀB_k s + (σ_k/3)‖s‖³,   B_k = ∇²f(x_k).

                By CGT Theorem 3.1, s* is a global minimizer iff (B_k + λI)s* = −g_k with
                λ = σ_k‖s*‖ and B_k + λI ⪰ 0. With B_k = QΛQᵀ (λ₁ ≤ … ≤ λₙ) and γ = Qᵀg_k, λ is the
                root λ > max(0, −λ₁) of the secular equation (CGT eq. 6.7)

                    φ₁(λ) = 1/‖s(λ)‖ − σ_k/λ = 0,   s(λ) = −Q(Λ + λI)⁻¹γ,

                or, in the hard case (γ vanishes on the eigenspace of λ₁ < 0 and the root does
                not exist), λ = −λ₁ and s* = s(−λ₁) + αu₁ with ‖s*‖ = −λ₁/σ_k (CGT eq. 6.6).
                Step acceptance (eq. 2.4–2.5): ρ_k = (f_k − f(x_k + s_k)) / (f_k − m_k(s_k)) and
                x_{k+1} = x_k + s_k iff ρ_k ≥ η₁. Regularization update (eq. 2.6, with the
                choices of CGT §7):

                    ρ_k > η₂            (very successful)  σ_{k+1} = max(min(σ_k, ‖g_k‖), ε_M)
                    η₁ ≤ ρ_k ≤ η₂       (successful)       σ_{k+1} = σ_k
                    ρ_k < η₁            (unsuccessful)     σ_{k+1} = γ σ_k

                Defaults σ₀ = 1, η₁ = 0.1, η₂ = 0.9, γ = 2 are those of CGT §7. With the exact
                model minimizer, ARC is Q-superlinear near a minimizer with ∇²f ≻ 0 (CGT Part I,
                §4.2), converges to second-order critical points (Part I, §5), and needs
                O(ε^{−3/2}) evaluations to reach ‖g‖ ≤ ε (CGT Part II).

``reg_newton``  Gradient-regularized Newton: x_{k+1} = x_k − (B_k + λ_k I)⁻¹ g_k, one linear solve
                per trial. Three variants (``variant``):

                ``fixed``            Mishchenko (2023), Algorithm 1: λ_k = √(H‖g_k‖). Global
                                     O(1/k²) for convex f under Assumption 1 (Theorem 1), local
                                     superlinear for strongly convex f (Theorem 2). Assumption 1
                                     holds when ∇²f is 2H-Lipschitz, i.e. H = L₂/2.
                ``adan``             Mishchenko (2023), Algorithm 2 (AdaN): H_k starts at
                                     H_{k−1}/4 (H₀ at k = 0) and doubles before every trial,
                                     λ = √(H_k‖g_k‖), until ‖∇f(x₊)‖ ≤ 2λr₊ and
                                     f(x₊) ≤ f(x_k) − (2/3)λr₊², r₊ = ‖x₊ − x_k‖ (Theorem 3).
                ``super_universal``  Doikov, Mishchenko & Nesterov (2024), Algorithm 2 with ψ ≡ 0
                                     and B = I: λ = 4ʲH_k‖g_k‖^α for j = 0, 1, … until
                                     ⟨∇f(x₊), x_k − x₊⟩ ≥ ‖∇f(x₊)‖²/(4λ); then
                                     H_{k+1} = 4^{j_k}H_k/4. α ∈ [2/3, 1].

                ``H`` is H (``fixed``) or the initial H₀ (``adan``, ``super_universal``). The
                theory is for convex f; on nonconvex f, B_k + λI can be indefinite and the step is
                still taken as written (``info["pd"]`` is False), so the method can be attracted
                to a saddle point or a maximizer.

Stopping test (both methods): ‖∇f(x_k)‖∞ ≤ ``gtol``; the run then reports ``converged=True`` only
if ∇²f(x_k) also has no eigenvalue below −tol, i.e. the second-order necessary conditions hold to
the accuracy of ∇²f (N&W Thms 2.3–2.4). This is the convention of ``numopt.unconstrained.newton``
with the same tolerance: tol = n·ε·|λ|_max for an analytic ∇²f, n·ε^{1/3}·max(1, |λ|_max) for a
central-difference ∇²f (|f(x)| added to the max when ∇f is a central difference too). A point with
‖∇f‖∞ ≤ gtol and λ_min < −tol is a saddle point or a maximizer: the run stops there with
``converged=False`` and the message says which. ``Result.extra["lambda_min"]`` is λ_min(∇²f) at the
final point (None when ∇²f is not finite there). # NOTE: ``numopt.unconstrained.trust_region`` uses
the first-order test alone (``converged=True`` at a saddle); here a consumer that reads only
``converged`` never shows a saddle stop as a success. ARC reaches such a stop only from an iterate
with ∇f = 0 at a saddle (CGT Part I §5); ``reg_newton`` has no such guarantee.

Failures (``converged=False``): a stop at a saddle point or maximizer (above); non-finite f, ∇f or
∇²f at x₀ or at an accepted point; ``max_iter``. ``arc``: a model step that predicts no decrease
(the model is at its rounding level), or σ_k growing until a rejected step is below the rounding
level of x_k. ``reg_newton``: B_k + λI numerically singular (``fixed``), or the adaptive search
failing after 100 trials (in the adaptive variants a singular trial is rejected and H grows).
Invalid parameters raise ``ValueError``.

Rounding allowance. Near a minimizer the predicted decrease falls below the rounding error of f.
Like ``numopt.unconstrained.trust_region`` (Conn, Gould & Toint (2000), §17.4.2), ``arc`` computes
ρ_k = (actual + δ)/(predicted + δ) and ``adan`` tests f(x₊) ≤ f(x_k) − (2/3)λr₊² + δ, with the
relative allowance δ = 10³ ε max(|f(x_k)|, |f(x₊)|). # NOTE: not in CGT or Mishchenko; without it,
both methods can reject a correct Newton-like step only because f is evaluated in floating point.

Evaluation counts (exact, with ``Counted``): f, ∇f, ∇²f once at x₀; then ``arc`` evaluates f once
per iteration (at the trial point) and ∇f, ∇²f once per accepted point. ``reg_newton``: ∇²f once
per accepted point; ``fixed`` evaluates f and ∇f once per iteration; ``adan`` evaluates f and ∇f
once per non-singular trial; ``super_universal`` evaluates ∇f once per non-singular trial and f
once per accepted point. # NOTE: ``fixed`` and ``super_universal`` do not need f; it is evaluated
(and counted) for the trace only. Without ``Problem.grad`` (``Problem.hess``), ∇f (∇²f) is a
central difference (``numopt.core.diff``) whose evaluations are counted in ``n_fev`` (``n_gev``).

Linear algebra: one symmetric eigendecomposition of ∇²f per accepted point (``numpy.linalg.eigh``);
every solve with ∇²f + λI is a diagonal solve in its eigenbasis, so no matrix is inverted.

Trace: k = 0 is x₀. ``arc``: one Step per iteration, accepted or rejected (like the trust-region
methods); Step k holds x_k (= x_{k−1} after a rejection). ``reg_newton``: one Step per accepted
iterate; the rejected trials of the adaptive search are in ``info["trials"]``.
``n_iter == trace[-1].k``. ``Step.step_size`` is ‖s‖₂, the length of the trial step of the
iteration (None at k = 0).

Info keys (``arc``; every key except ``center``, ``grad``, ``H``, ``sigma``, ``new_sigma`` is None
at k = 0):
    center: [n]            x_{k−1}, where the model of this iteration was built (x₀ at k = 0).
    grad: [n] | None       ∇f(center) (None if not finite).
    H: [[2]] | None        ∇²f(center) when n = 2 (for model contours), else None.
    sigma: float           σ used for this iteration's model (σ₀ at k = 0).
    new_sigma: float       σ after the update (σ₀ at k = 0).
    step: [n]              the model step s.
    step_norm: float       ‖s‖₂.
    lambda: float | None   the multiplier λ = σ‖s‖ of CGT Thm. 3.1 (None for a Cauchy fallback).
    lambda_min: float      λ₁, the smallest eigenvalue of ∇²f(center).
    hard_case: bool        the step came from the hard case (CGT eq. 6.6).
    lambda_iters: int      iterations on the secular equation.
    predicted: float       f(center) − m(s) > 0.
    actual: float | None   f(center) − f(center + s) (None if f(center + s) is not finite).
    rho: float | None      the ratio ρ with the rounding allowance (None if f(center + s) is not
                           finite).
    accepted: bool         ρ ≥ η₁.
    iteration: "very_successful" | "successful" | "unsuccessful"   the class of CGT eq. 2.6.
    trial_point: [n]       center + s.
    cauchy_point: [n]      center + s^C, the minimizer of m along −g (CGT eq. 2.3).
    newton_point: [n] | None   center − B⁻¹g when B ≻ 0 (Cholesky succeeds), else None.
    note: str | None       a remark (e.g. the Cauchy fallback of the subproblem solver).

Info keys (``reg_newton``; "incoming" keys describe x_{k−1} → x_k and are None ([] for ``trials``)
at k = 0):
    grad: [n] | None       ∇f(x_k) (None if not finite).
    hess: [[n]] | None     ∇²f(x_k) for n ≤ 2, else None.
    hess_eigs: [n] | None  eigenvalues of ∇²f(x_k), ascending (None if ∇²f is not finite).
    direction: [n] | None  incoming: the accepted step s = x_k − x_{k−1}.
    lambda: float | None   incoming: the accepted regularization λ.
    H_reg: float | None    incoming: the accepted H (``fixed``: H; ``adan``: H_k;
                           ``super_universal``: 4^{j_k}H_k).
    trials: [[H, lambda, f, grad_norm, accepted]]   incoming: every trial of the search, in
                           order (f is None when the variant does not evaluate it at that trial;
                           f and grad_norm are None for a singular trial).
    inner_iters: int | None    incoming: the number of trials (n_k of Mishchenko Thm. 3; 1 for
                           ``fixed``).
    pd: bool | None        incoming: B_{k−1} + λI was positive definite (always, for convex f).
    descent: bool | None   incoming: ∇f(x_{k−1})ᵀs < 0.

References:
    C. Cartis, N. I. M. Gould, Ph. L. Toint, "Adaptive cubic regularisation methods for
        unconstrained optimization. Part I", Math. Program. 127 (2011) 245–295 (Alg. 2.1,
        eq. 1.4, 2.3–2.6, Thm. 3.1, §6.1 eq. 6.6–6.7, §7).
    C. Cartis, N. I. M. Gould, Ph. L. Toint, "... Part II: worst-case function- and
        derivative-evaluation complexity", Math. Program. 130 (2011) 295–319.
    Y. Nesterov, B. T. Polyak, "Cubic regularization of Newton method and its global
        performance", Math. Program. 108 (2006) 177–205.
    K. Mishchenko, "Regularized Newton method with global O(1/k²) convergence", SIAM J. Optim.
        33(3) (2023) 1440–1462; numbering from arXiv:2112.02089v3 (Alg. 1, 2; Assumption 1;
        Thm. 1–3).
    N. Doikov, K. Mishchenko, Y. Nesterov, "Super-universal regularized Newton method", SIAM J.
        Optim. 34(1) (2024) 27–56; numbering from arXiv:2208.05888v1 (Alg. 2).

Promoted from ``research/regularized-newton-arc`` (method and verification there).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step

Array = NDArray[np.float64]

#: Machine epsilon of float64.
_EPS = float(np.finfo(np.float64).eps)
#: ε_M of CGT §7: the floor of σ after a very successful iteration.
_SIGMA_MIN = _EPS
#: Rounding allowance δ = _ROUNDING·ε·max(|f(x)|, |f(x₊)|) (same constant as the trust-region code).
_ROUNDING = 1e3
#: Relative accuracy |‖s‖ − λ/σ| ≤ _SECULAR_RTOL·λ/σ of the cubic subproblem's secular equation.
_SECULAR_RTOL = 1e-12
#: Iteration cap of the secular-equation solver (it needs < 60 in the tests).
_SECULAR_MAX_ITER = 200
#: Cap on the trials of the adaptive searches (H grows by 2 or 4 per trial: 2¹⁰⁰ ≈ 1e30).
_MAX_TRIALS = 100

VARIANTS = ("adan", "fixed", "super_universal")

P_GTOL = ParamSpec(
    "gtol",
    1e-8,
    min=1e-14,
    max=1e-2,
    log=True,
    help="Stop when ‖∇f(x)‖∞ ≤ gtol (converged only if ∇²f(x) has no negative eigenvalue).",
)
P_MAX_ITER = ParamSpec("max_iter", 200, kind="int", min=1, max=100_000, help="Iteration limit.")
P_SIGMA0 = ParamSpec(
    "sigma0",
    1.0,
    min=1e-4,
    max=1e4,
    log=True,
    help="Initial cubic regularization σ₀ (like an inverse trust radius).",
)
P_ETA1 = ParamSpec(
    "eta1", 0.1, min=1e-4, max=0.5, help="Accept the step when ρ ≥ η₁ (CGT eq. 2.5)."
)
P_ETA2 = ParamSpec(
    "eta2", 0.9, min=0.5, max=0.999, help="Very successful step when ρ > η₂: σ decreases (eq. 2.6)."
)
P_GAMMA = ParamSpec(
    "gamma", 2.0, min=1.1, max=10.0, help="σ ← γσ after an unsuccessful step (CGT eq. 2.6)."
)
P_VARIANT = ParamSpec(
    "variant",
    "adan",
    kind="choice",
    choices=VARIANTS,
    help="fixed: λ = √(H‖g‖) (Mishchenko Alg. 1); adan: H adapted by doubling (Alg. 2); "
    "super_universal: λ = 4ʲH‖g‖^α (Doikov–Mishchenko–Nesterov Alg. 2).",
)
P_H = ParamSpec(
    "H",
    1.0,
    min=1e-6,
    max=1e6,
    log=True,
    help="fixed: the constant H (∇²f is 2H-Lipschitz); adaptive variants: the initial H₀.",
)
P_ALPHA = ParamSpec(
    "alpha",
    1.0,
    min=2.0 / 3.0,
    max=1.0,
    help="super_universal only: the power α ∈ [2/3, 1] of ‖∇f‖ in λ.",
)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _value(v: Any) -> float:
    arr = np.asarray(v, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"f(x) must return a scalar, got an array of shape {arr.shape}")
    return float(arr.reshape(-1)[0])


@dataclass(frozen=True)
class _Source:
    """Which derivatives are central differences (they set the eigenvalue tolerance)."""

    grad_fd: bool
    hess_fd: bool


def _resolve(
    problem: Problem | Callable[..., Any], x0: ArrayLike | None
) -> tuple[Array, Counted, Counted, Counted, _Source]:
    """(x0, counted f, counted ∇f, counted ∇²f, source); ∇f/∇²f fall back to central differences.

    A dim-1 ``Problem`` follows the scalar convention of ``numopt.core.types`` (its callables
    receive a float); the method itself works on a length-1 vector.
    """
    prob = problem if isinstance(problem, Problem) else vector_problem(problem, x0=x0)
    scalar = prob.dim == 1
    x = start_point(prob, x0)
    n = x.size

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
        hess = Counted(lambda z: np.asarray(h_fn(arg(z)), dtype=np.float64).reshape(n, n))
    else:
        hess = Counted(lambda z: diff.hessian(grad, z))
    return x, f, grad, hess, _Source(prob.grad is None, prob.hess is None)


def _eval(fn: Counted, x: Array) -> Any:
    with np.errstate(all="ignore"):
        return fn(x)


def _vec(v: Array | None) -> list[float] | None:
    return None if v is None else [float(t) for t in v]


def _eig_tol(lam: Array, fx: float, src: _Source) -> float:
    """|λ| ≤ tol cannot be told apart from 0 (the rule of ``numopt.unconstrained.newton``).

    Analytic ∇²f: n·ε·|λ|_max (rounding in the eigendecomposition only). Central-difference ∇²f:
    n·ε^{1/3}·max(1, |λ|_max), with |f(x)| in the max when ∇f is a central difference too (a
    central difference of a central difference has the error ε^{1/3}|f| per entry, N&W §8.1, and a
    symmetric perturbation E moves every eigenvalue by at most ‖E‖₂ ≤ n·max|E_ij|, Weyl).
    """
    n = lam.size
    lam_max = float(np.max(np.abs(lam)))
    if not src.hess_fd:
        return n * _EPS * lam_max
    scale = max(1.0, lam_max, abs(fx) if src.grad_fd and math.isfinite(fx) else 0.0)
    return n * _EPS ** (1.0 / 3.0) * scale


def _second_order(lam: Array, fx: float, src: _Source) -> tuple[bool, str]:
    """(no eigenvalue below −tol, description) at a point that passed the gradient test.

    Classification as in ``numopt.unconstrained.newton`` (N&W Thms 2.3–2.4).
    """
    tol = _eig_tol(lam, fx, src)
    lam_min = float(lam[0])
    lam_top = float(lam[-1])
    source = "the finite-difference ∇²f" if src.hess_fd else "∇²f"
    if lam_min < -tol:
        if lam_top < -tol:
            kind = "a maximizer"
        elif lam_top > tol:
            kind = "a saddle point"
        else:
            # ∇²f ⪯ 0 and singular: the higher-order terms decide, not ∇²f.
            kind = "a saddle point or a maximizer (∇²f ⪯ 0 is singular)"
        return False, (
            f"stopped at {kind}, not a minimizer: {source} has the eigenvalue "
            f"λ_min = {lam_min:.3g} < −tol = {-tol:.3g}"
        )
    if lam_min <= tol:
        return True, (
            f"λ_min({source}) = {lam_min:.3g} is within ±{tol:.3g} of 0: positive semidefinite, "
            "the second-order sufficient condition is not verified"
        )
    return True, f"{source} is positive definite (λ_min = {lam_min:.3g}): a strict local minimizer"


def _check_common(gtol: float, max_iter: int) -> int:
    if not gtol >= 0.0:
        raise ValueError(f"gtol must be ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    return int(max_iter)


# --------------------------------------------------------------------------------------
# The cubic subproblem (CGT Thm. 3.1, §6.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CubicStep:
    """The global minimizer s of gᵀs + ½sᵀBs + (σ/3)‖s‖³ and its multiplier λ = σ‖s‖."""

    s: Array
    lam: float
    lam_min: float
    hard_case: bool
    iters: int
    #: f − m(s) = ½sᵀ(B + λI)s + λ‖s‖²/6 (exact identity at the minimizer; no cancellation).
    predicted: float
    #: False when the secular iteration did not reach its tolerance (then s = s(hi)).
    solved: bool


def cubic_subproblem(g: Array, B: Array, sigma: float) -> CubicStep:
    """Global minimizer of m(s) = gᵀs + ½sᵀBs + (σ/3)‖s‖³ over ℝⁿ (CGT Thm. 3.1, §6.1).

    With B = QΛQᵀ (λ₁ ≤ … ≤ λₙ) and γ = Qᵀg, s(λ) = −Q(Λ + λI)⁻¹γ. The iteration variable t is
    chosen so that λ_j + λ = c_j + t and λ = t + t_off are sums of non-negative terms (no
    cancellation): for λ₁ ≥ 0, t = λ (c = Λ, t_off = 0); for λ₁ < 0, t = μ = λ + λ₁, the smallest
    eigenvalue of B + λI (c_j = λ_j − λ₁ ≥ 0 formed once, t_off = −λ₁ > 0), so the pole at λ = −λ₁
    is t = 0 exactly, as in ``trust_region_exact``.

    1. Hard case (CGT eq. 6.6): λ₁ < 0, γ vanishes (to rounding) on the eigenspace E₁ of λ₁ and
       ‖s_⊥‖ ≤ −λ₁/σ with s_⊥ = s(λ = −λ₁) restricted to E₁^⊥. Then λ = −λ₁ and
       s = s_⊥ + αu, α = √((λ₁/σ)² − ‖s_⊥‖²), u ∈ E₁ a unit vector with uᵀg ≤ 0.
    2. Otherwise λ is the root of φ₁(λ) = 1/‖s(λ)‖ − σ/λ (CGT eq. 6.7) for t ∈ (0, hi], where
       t = 0 is λ = max(0, −λ₁) and hi is the t of the positive root of λ(λ + λ₁) = σ‖g‖: there
       ‖s‖ ≤ ‖g‖/(λ + λ₁) = λ/σ, so φ₁ ≥ 0 at hi. φ₁ is increasing and concave in λ (1/‖s‖ is
       concave, Moré–Sorensen 1983; −σ/λ is concave), so Newton's iterates land at or left of the
       root; a step that leaves the bracket is replaced by t = max(√(lo·hi), 10⁻³hi).
    """
    # NOTE: CGT §6.1 factorize B + λI by Cholesky per iteration; one eigendecomposition (n is
    # small here) gives the same Newton update, the exact interval (−λ₁, ∞) and q₁ for case 1.
    n = g.size
    eigvals, Q = np.linalg.eigh(B)  # (n,), (n, n)
    gamma = Q.T @ g  # (n,)
    lam1 = float(eigvals[0])
    gnorm = float(np.linalg.norm(g))
    bnorm = float(np.max(np.abs(eigvals)))
    if lam1 >= 0.0:
        c, t_off = eigvals, 0.0  # t = λ
    else:
        c, t_off = eigvals - lam1, -lam1  # t = μ = λ + λ₁; c[0] = 0
    c = np.maximum(c, 0.0)  # (n,)

    def finish(t: float, s: Array, hard: bool, iters: int, solved: bool) -> CubicStep:
        lam = t + t_off
        v = Q.T @ s  # (n,) coordinates of s in the eigenbasis
        pred = 0.5 * float(np.sum((c + t) * v * v)) + lam * float(v @ v) / 6.0
        return CubicStep(s, lam, lam1, hard, iters, pred, solved)

    # 1. Hard case.
    tol = 10.0 * n * _EPS
    if lam1 < 0.0:
        cluster = c <= tol * bnorm
        gamma1 = float(np.linalg.norm(gamma[cluster]))
        target = -lam1 / sigma  # ‖s‖ at λ = −λ₁
        if gamma1 <= tol * max(bnorm * target, gnorm):
            coef = np.zeros_like(gamma)
            rest = ~cluster
            coef[rest] = gamma[rest] / c[rest]
            s_perp = -(Q @ coef)
            sp = float(np.linalg.norm(s_perp))
            if sp <= target:
                alpha = math.sqrt(max(target * target - sp * sp, 0.0))
                if gamma1 > 0.0:
                    u = -(Q[:, cluster] @ gamma[cluster]) / gamma1  # uᵀg = −‖P₁g‖ < 0
                else:
                    # NOTE: P₁g = 0: both signs give the same model value; the sign is fixed so
                    # that the largest |entry| of u is positive (deterministic, port-independent).
                    u = Q[:, 0].copy()
                    if u[int(np.argmax(np.abs(u)))] < 0.0:
                        u = -u
                return finish(0.0, s_perp + alpha * u, True, 0, True)

    # 2. Safeguarded Newton on φ₁ in t.
    if gnorm == 0.0:
        return finish(0.0, np.zeros(n), False, 0, True)
    disc = math.sqrt(lam1 * lam1 + 4.0 * sigma * gnorm)
    # Positive root of λ(λ + λ₁) = σ‖g‖ in the variable t, free of cancellation:
    #   λ₁ ≥ 0: t = λ = (−λ₁ + disc)/2 = 2σ‖g‖/(λ₁ + disc);
    #   λ₁ < 0: t = λ + λ₁ = (λ₁ + disc)/2 = 2σ‖g‖/(disc − λ₁).
    hi = 2.0 * sigma * gnorm / (disc + abs(lam1))
    lo = 0.0
    t = hi
    iters = 0
    while iters < _SECULAR_MAX_ITER:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            w = gamma / (c + t)  # (n,), −Qᵀs
            snorm = float(np.linalg.norm(w))
            lam = t + t_off
            target = lam / sigma
            if abs(snorm - target) <= _SECULAR_RTOL * target:
                return finish(t, -(Q @ w), False, iters, True)
            q_sq = float(np.sum(w * w / (c + t)))  # Σ γ²/(λ_j + λ)³
            phi = 1.0 / snorm - sigma / lam
            dphi = q_sq / snorm**3 + sigma / (lam * lam)
            t_new = t - phi / dphi
        if snorm > target:  # φ₁ < 0: the root lies right of t
            lo = t
        else:
            hi = t
        if not (math.isfinite(t_new) and lo < t_new <= hi):
            t_new = max(math.sqrt(lo) * math.sqrt(hi), 1e-3 * hi)
        iters += 1
        if not lo < t_new <= hi or t_new == t:
            break  # the bracket collapsed to adjacent floating-point numbers
        t = t_new
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        s = -(Q @ (gamma / (c + hi)))
    return finish(hi, s, False, iters, False)


def cubic_model(g: Array, B: Array, sigma: float, s: Array) -> float:
    """m(s) − f = gᵀs + ½sᵀBs + (σ/3)‖s‖³ (CGT eq. 1.4)."""
    ns = float(np.linalg.norm(s))
    return float(g @ s) + 0.5 * float(s @ (B @ s)) + sigma * ns**3 / 3.0


def cubic_cauchy(g: Array, B: Array, sigma: float) -> Array:
    """s^C = −t g/‖g‖ minimizing m along −g (CGT eq. 2.3).

    With u = g/‖g‖ and c = uᵀBu, φ(t) = −t‖g‖ + ½ct² + (σ/3)t³ has φ'(t) = 0 at the positive root
    t = 2‖g‖/(c + √(c² + 4σ‖g‖)) (c ≥ 0) or (−c + √(c² + 4σ‖g‖))/(2σ) (c < 0), each free of
    cancellation (Higham (2002), §1.8).
    """
    gnorm = float(np.linalg.norm(g))
    if gnorm == 0.0:
        return np.zeros_like(g)
    u = g / gnorm
    c = float(u @ (B @ u))
    root = math.sqrt(c * c + 4.0 * sigma * gnorm)
    t = 2.0 * gnorm / (c + root) if c >= 0.0 else (root - c) / (2.0 * sigma)
    return -t * u


def _newton_point(B: Array, g: Array) -> Array | None:
    """−B⁻¹g by Cholesky when B ≻ 0 (for the picture only), else None."""
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(all="ignore"):
        y = np.linalg.solve(L, g)
        p = -np.linalg.solve(L.T, y)
    return p if finite(p) else None


# --------------------------------------------------------------------------------------
# ARC (CGT Alg. 2.1)
# --------------------------------------------------------------------------------------


@register(
    id="arc",
    family="unconstrained",
    name="ARC (adaptive cubic regularization)",
    params=(P_GTOL, P_MAX_ITER, P_SIGMA0, P_ETA1, P_ETA2, P_GAMMA),
    needs=("f", "grad", "hess"),
    order="superlinear (near a minimizer with ∇²f ≻ 0); O(ε⁻³ᐟ²) worst-case evaluations",
    summary="Step to the global minimizer of the Newton model plus a cubic penalty (σ/3)‖s‖³, "
    "and adapt σ from how well the model predicted the decrease.",
    references=(
        "Cartis, Gould & Toint (2011), Math. Program. 127, Part I: Alg. 2.1; eq. 1.4, 2.3–2.6; "
        "Thm. 3.1; §6.1 eq. 6.6–6.7; §7 (parameters)",
        "Cartis, Gould & Toint (2011), Math. Program. 130, Part II (O(ε⁻³ᐟ²) complexity)",
        "Nesterov & Polyak (2006), Math. Program. 108 (cubic regularization of Newton's method)",
    ),
)
def arc(
    problem: Problem | Callable[..., Any],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    sigma0: float = 1.0,
    eta1: float = 0.1,
    eta2: float = 0.9,
    gamma: float = 2.0,
) -> Result:
    """Adaptive cubic regularization with the exact model minimizer (CGT (2011a), Alg. 2.1).

    s_k = argmin m_k(s) = f_k + g_kᵀs + ½sᵀ∇²f(x_k)s + (σ_k/3)‖s‖³ (eq. 1.4; Thm. 3.1, §6.1);
    ρ_k = (f_k − f(x_k + s_k))/(f_k − m_k(s_k)) (eq. 2.4); x_{k+1} = x_k + s_k iff ρ_k ≥ η₁ (eq. 2.5);
    σ_{k+1} = max(min(σ_k, ‖g_k‖), ε_M) if ρ_k > η₂, σ_k if η₁ ≤ ρ_k ≤ η₂, γσ_k otherwise (eq. 2.6
    with the choices of §7). The step satisfies the Cauchy condition m_k(s_k) ≤ m_k(s_k^C) (eq. 2.2)
    because it is the global minimizer; if the secular iteration misses its tolerance, the better of
    s(hi) and the Cauchy point is used (``note`` says so).

    Stopping test: ‖∇f(x_k)‖∞ ≤ ``gtol``; converged only if λ_min(∇²f(x_k)) ≥ −tol as well (a
    saddle point or maximizer gives ``converged=False``). See the module docstring.
    """
    max_iter = _check_common(gtol, max_iter)
    if not 0.0 < sigma0 < math.inf:
        raise ValueError(f"sigma0 must be positive and finite, got {sigma0}")
    if not 0.0 < eta1 <= eta2 < 1.0:
        raise ValueError(f"need 0 < eta1 ≤ eta2 < 1, got eta1={eta1}, eta2={eta2}")
    if not 1.0 < gamma < math.inf:
        raise ValueError(f"gamma must be > 1, got {gamma}")

    x, f, grad, hess, src = _resolve(problem, x0)
    n = x.size
    fx = float(_eval(f, x))
    g: Array = np.asarray(_eval(grad, x), dtype=np.float64)
    B: Array = np.asarray(_eval(hess, x), dtype=np.float64)
    sigma = float(sigma0)
    trace: list[Step] = []
    n_rejected = 0

    def base_info(center: Array, gc: Array, Bc: Array) -> dict[str, Any]:
        return {
            "center": _vec(center),
            "grad": _vec(gc) if finite(gc) else None,
            "H": Bc.tolist() if n == 2 and Bc.shape == (2, 2) and finite(Bc) else None,
            "sigma": sigma,
            "new_sigma": sigma,
            "step": None,
            "step_norm": None,
            "lambda": None,
            "lambda_min": None,
            "hard_case": None,
            "lambda_iters": None,
            "predicted": None,
            "actual": None,
            "rho": None,
            "accepted": None,
            "iteration": None,
            "trial_point": None,
            "cauchy_point": None,
            "newton_point": None,
            "note": None,
        }

    def done(converged: bool, message: str, k: int) -> Result:
        lam_min = None
        if finite(B) and B.shape == (n, n):
            eig = np.linalg.eigvalsh(B)
            lam_min = float(eig[0])
            if converged:
                # The second-order part of the stopping test (module docstring).
                converged, why = _second_order(eig, fx, src)
                message += "; " + why
        extra = {"n_rejected": n_rejected, "sigma_final": sigma, "lambda_min": lam_min}
        return Result(
            "arc", x, fx, converged, message, k, f.n, grad.n, hess.n, trace=trace, extra=extra
        )

    if not (math.isfinite(fx) and finite(g, B)) or B.shape != (n, n) or g.shape != (n,):
        gn = float(np.linalg.norm(g)) if finite(g) else None
        fun0 = fx if math.isfinite(fx) else None
        trace.append(Step(0, x.copy(), fun0, gn, None, base_info(x, g, B)))
        return done(False, "f, ∇f or ∇²f is not finite (or has the wrong shape) at x0", 0)
    B = 0.5 * (B + B.T)  # NOTE: symmetrize once; eigh reads one triangle only.
    trace.append(Step(0, x.copy(), fx, float(np.linalg.norm(g)), None, base_info(x, g, B)))

    for k in range(1, max_iter + 1):
        ginf = float(np.max(np.abs(g)))
        if ginf <= gtol:
            return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", k - 1)
        sub = cubic_subproblem(g, B, sigma)
        s, note = sub.s, None
        predicted = sub.predicted
        s_cauchy = cubic_cauchy(g, B, sigma)
        lam: float | None = sub.lam
        if not sub.solved or not finite(s):
            # NOTE: safety net (not reached in the tests): keep the Cauchy condition (CGT eq. 2.2).
            m_s = cubic_model(g, B, sigma, s) if finite(s) else math.inf
            m_c = cubic_model(g, B, sigma, s_cauchy)
            if m_c < m_s:
                s, lam, predicted = s_cauchy, None, -m_c
                note = "secular equation not solved to tolerance; Cauchy point used"
            else:
                predicted = -m_s
                note = "secular equation not solved to tolerance; s(λ_hi) used"
        if not (predicted > 0.0 and math.isfinite(predicted)):
            return done(
                False,
                f"the model step predicts no decrease (f − m(s) = {predicted:.3g} at iteration "
                f"{k}): the cubic model is at its rounding level; ‖∇f‖∞ = {ginf:.3g} > gtol",
                k - 1,
            )
        trial = x + s
        f_trial = float(_eval(f, trial))
        if math.isfinite(f_trial):
            actual: float | None = fx - f_trial
            delta_round = _ROUNDING * _EPS * max(abs(fx), abs(f_trial))
            rho: float | None = (fx - f_trial + delta_round) / (predicted + delta_round)
        else:
            actual, rho = None, None
        rho_value = -math.inf if rho is None else rho
        if rho_value > eta2:
            kind = "very_successful"
            new_sigma = max(min(sigma, float(np.linalg.norm(g))), _SIGMA_MIN)
        elif rho_value >= eta1:
            kind, new_sigma = "successful", sigma
        else:
            kind, new_sigma = "unsuccessful", gamma * sigma
        accepted = rho_value >= eta1
        step_norm = float(np.linalg.norm(s))

        info = base_info(x, g, B)
        newton = _newton_point(B, g)
        info.update(
            {
                "new_sigma": new_sigma,
                "step": _vec(s),
                "step_norm": step_norm,
                "lambda": lam,
                "lambda_min": sub.lam_min,
                "hard_case": sub.hard_case,
                "lambda_iters": sub.iters,
                "predicted": predicted,
                "actual": actual,
                "rho": rho,
                "accepted": accepted,
                "iteration": kind,
                "trial_point": _vec(trial),
                "cauchy_point": _vec(x + s_cauchy),
                "newton_point": None if newton is None else _vec(x + newton),
                "note": note,
            }
        )
        if accepted:
            g_new = np.asarray(_eval(grad, trial), dtype=np.float64)
            B_new = np.asarray(_eval(hess, trial), dtype=np.float64)
            x, fx = trial, f_trial
            if not (finite(g_new, B_new) and B_new.shape == (n, n)):
                g = g_new
                gn = float(np.linalg.norm(g_new)) if finite(g_new) else None
                trace.append(Step(k, x.copy(), fx, gn, step_norm, info))
                B = np.full((n, n), np.nan)
                return done(False, f"∇f or ∇²f is not finite at the iterate of step {k}", k)
            g, B = g_new, 0.5 * (B_new + B_new.T)
        else:
            n_rejected += 1
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), step_norm, info))
        sigma = new_sigma
        if not accepted and step_norm <= _EPS * max(1.0, float(np.linalg.norm(x))):
            return done(
                False,
                f"σ = {sigma:.3g}: the step fell below the rounding level of x (no acceptable "
                f"step); ‖∇f‖∞ = {float(np.max(np.abs(g))):.3g} > gtol",
                k,
            )

    ginf = float(np.max(np.abs(g)))
    if ginf <= gtol:
        return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", max_iter)
    return done(False, f"reached max_iter={max_iter}", max_iter)


# --------------------------------------------------------------------------------------
# Gradient-regularized Newton (Mishchenko 2023, Alg. 1–2; Doikov–Mishchenko–Nesterov 2024, Alg. 2)
# --------------------------------------------------------------------------------------


def _reg_solve(eigvals: Array, Q: Array, g: Array, lam: float) -> tuple[Array | None, bool]:
    """s = −(B + λI)⁻¹g from B = QΛQᵀ; (None, pd) when B + λI is numerically singular."""
    shifted = eigvals + lam  # (n,)
    scale = max(float(np.max(np.abs(eigvals))), lam)
    pd = bool(shifted[0] > 0.0)
    if float(np.min(np.abs(shifted))) <= 10.0 * eigvals.size * _EPS * scale:
        return None, pd
    with np.errstate(over="ignore", invalid="ignore"):
        s = -(Q @ ((Q.T @ g) / shifted))
    return (s if finite(s) else None), pd


@register(
    id="reg_newton",
    family="unconstrained",
    name="Gradient-regularized Newton",
    params=(P_GTOL, P_MAX_ITER, P_VARIANT, P_H, P_ALPHA),
    needs=("f", "grad", "hess"),
    order="superlinear (strongly convex f, Mishchenko Thm. 2); global O(1/k²) for convex f",
    summary="Newton step with ∇²f + λI, where λ = √(H‖∇f‖) shrinks as the gradient vanishes; "
    "H is fixed or adapted, and no line search is needed on convex functions.",
    references=(
        "Mishchenko (2023), SIAM J. Optim. 33(3), arXiv:2112.02089v3: Alg. 1 (fixed), "
        "Alg. 2 (AdaN), Assumption 1, Thm. 1–3",
        "Doikov, Mishchenko & Nesterov (2024), SIAM J. Optim. 34(1), arXiv:2208.05888v1: "
        "Alg. 2 (super-universal, ψ ≡ 0, B = I)",
    ),
)
def reg_newton(
    problem: Problem | Callable[..., Any],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    max_iter: int = 200,
    variant: str = "adan",
    H: float = 1.0,
    alpha: float = 1.0,
) -> Result:
    """Gradient-regularized Newton: x_{k+1} = x_k − (∇²f(x_k) + λ_k I)⁻¹∇f(x_k).

    ``fixed``: λ_k = √(H‖∇f(x_k)‖) (Mishchenko (2023), Alg. 1, Thm. 1–2).
    ``adan``: Mishchenko (2023), Alg. 2: H_k = H_{k−1}/4 (H₀ at k = 0), then repeat H_k ← 2H_k,
    λ = √(H_k‖g_k‖), x₊ = x_k − (∇²f(x_k) + λI)⁻¹g_k, r₊ = ‖x₊ − x_k‖ until ‖∇f(x₊)‖ ≤ 2λr₊ and
    f(x₊) ≤ f(x_k) − (2/3)λr₊² (+ the rounding allowance δ of the module docstring).
    ``super_universal``: Doikov, Mishchenko & Nesterov (2024), Alg. 2 (ψ ≡ 0, B = I): for j = 0, 1, …
    λ = 4ʲH_k‖g_k‖^α until ⟨∇f(x₊), x_k − x₊⟩ ≥ ‖∇f(x₊)‖²/(4λ); H_{k+1} = 4^{j_k}H_k/4.

    The theory is for convex f. # NOTE: on nonconvex f, B + λI can be indefinite; the step is then
    still taken as written (``info["pd"]`` is False). A singular B + λI stops ``fixed``; the
    adaptive variants reject that trial and increase H.

    Stopping test: ‖∇f(x_k)‖∞ ≤ ``gtol``; converged only if λ_min(∇²f(x_k)) ≥ −tol as well (a
    saddle point or maximizer gives ``converged=False``). See the module docstring.
    """
    max_iter = _check_common(gtol, max_iter)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not 0.0 < H < math.inf:
        raise ValueError(f"H must be positive and finite, got {H}")
    # NOTE: 1e-12 slack so that the float 2/3 (the ParamSpec minimum) passes the check.
    if not 2.0 / 3.0 - 1e-12 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [2/3, 1], got {alpha}")

    x, f, grad, hess, src = _resolve(problem, x0)
    n = x.size
    fx = float(_eval(f, x))
    g: Array = np.asarray(_eval(grad, x), dtype=np.float64)
    B: Array = np.asarray(_eval(hess, x), dtype=np.float64)
    trace: list[Step] = []
    H_k = float(H)  # adan: H_{k−1}; super_universal: H_k
    total_trials = 0

    def eig_of(Bm: Array) -> tuple[Array, Array] | None:
        if not (finite(Bm) and Bm.shape == (n, n)):
            return None
        lam_, Q_ = np.linalg.eigh(0.5 * (Bm + Bm.T))  # NOTE: symmetrized; eigh reads one triangle.
        return lam_, Q_

    def state_info(gc: Array, Bm: Array, eig: tuple[Array, Array] | None) -> dict[str, Any]:
        return {
            "grad": _vec(gc) if finite(gc) else None,
            "hess": Bm.tolist() if n <= 2 and Bm.shape == (n, n) and finite(Bm) else None,
            "hess_eigs": None if eig is None else _vec(eig[0]),
        }

    incoming_none: dict[str, Any] = {
        "direction": None,
        "lambda": None,
        "H_reg": None,
        "trials": [],
        "inner_iters": None,
        "pd": None,
        "descent": None,
    }

    def done(converged: bool, message: str, k: int, eig: tuple[Array, Array] | None) -> Result:
        lam_min = None if eig is None else float(eig[0][0])
        if converged and eig is not None:
            # The second-order part of the stopping test (module docstring).
            converged, why = _second_order(eig[0], fx, src)
            message += "; " + why
        extra = {"variant": variant, "n_trials": total_trials, "lambda_min": lam_min}
        return Result(
            "reg_newton",
            x,
            fx,
            converged,
            message,
            k,
            f.n,
            grad.n,
            hess.n,
            trace=trace,
            extra=extra,
        )

    eig = eig_of(B)
    ok0 = math.isfinite(fx) and finite(g) and g.shape == (n,) and eig is not None
    trace.append(
        Step(
            0,
            x.copy(),
            fx if math.isfinite(fx) else None,
            float(np.linalg.norm(g)) if finite(g) else None,
            None,
            state_info(g, B, eig) | incoming_none,
        )
    )
    if not ok0 or eig is None:
        return done(False, "f, ∇f or ∇²f is not finite (or has the wrong shape) at x0", 0, eig)

    for k in range(1, max_iter + 1):
        ginf = float(np.max(np.abs(g)))
        if ginf <= gtol:
            return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", k - 1, eig)
        eigvals, Q = eig
        gnorm = float(np.linalg.norm(g))
        trials: list[list[Any]] = []
        accepted_point: tuple[Array, float, Array, float, float, Array, bool] | None = None
        fail: str | None = None

        if variant == "fixed":
            lam = math.sqrt(H * gnorm)
            s, pd = _reg_solve(eigvals, Q, g, lam)
            if s is None:
                fail = f"∇²f + λI is numerically singular at iteration {k} (λ = {lam:.3g})"
            else:
                xp = x + s
                fp = float(_eval(f, xp))
                gp = np.asarray(_eval(grad, xp), dtype=np.float64)
                gpn = float(np.linalg.norm(gp)) if finite(gp) else math.nan
                trials.append(
                    [
                        H,
                        lam,
                        fp if math.isfinite(fp) else None,
                        gpn if math.isfinite(gpn) else None,
                        True,
                    ]
                )
                accepted_point = (xp, fp, gp, lam, H, s, pd)
        else:
            if variant == "adan":
                # NOTE: Alg. 2, line 3 initializes H_k = H_{k−1}/4 ("start with H₀ if k = 0") and
                # line 5 doubles it before the first trial: the first trial uses 2H₀ at k = 0.
                H_try = H_k if k == 1 else H_k / 4.0
            else:
                H_try = H_k
            for j in range(_MAX_TRIALS):
                if variant == "adan":
                    H_try *= 2.0
                    lam = math.sqrt(H_try * gnorm)
                    H_used = H_try
                else:
                    H_used = 4.0**j * H_k
                    lam = H_used * gnorm**alpha
                s, pd = _reg_solve(eigvals, Q, g, lam)
                if s is None:
                    trials.append([H_used, lam, None, None, False])
                    continue
                xp = x + s
                r = float(np.linalg.norm(s))
                gp = np.asarray(_eval(grad, xp), dtype=np.float64)
                gpn = float(np.linalg.norm(gp)) if finite(gp) else math.nan
                if variant == "adan":
                    fp = float(_eval(f, xp))
                    delta_round = (
                        _ROUNDING * _EPS * max(abs(fx), abs(fp)) if math.isfinite(fp) else 0.0
                    )
                    ok = (
                        math.isfinite(fp)
                        and math.isfinite(gpn)
                        and gpn <= 2.0 * lam * r
                        and fp <= fx - (2.0 / 3.0) * lam * r * r + delta_round
                    )
                    trials.append(
                        [
                            H_used,
                            lam,
                            fp if math.isfinite(fp) else None,
                            gpn if math.isfinite(gpn) else None,
                            ok,
                        ]
                    )
                else:
                    fp = math.nan
                    ok = math.isfinite(gpn) and float(gp @ (-s)) >= gpn * gpn / (4.0 * lam)
                    trials.append([H_used, lam, None, gpn if math.isfinite(gpn) else None, ok])
                if ok:
                    if variant == "super_universal":
                        fp = float(_eval(f, xp))
                        H_k = H_used / 4.0  # H_{k+1} = 4^{j_k} H_k / 4
                    else:
                        H_k = H_used
                    accepted_point = (xp, fp, gp, lam, H_used, s, pd)
                    break
            if accepted_point is None:
                fail = (
                    f"the adaptive search found no acceptable step in {_MAX_TRIALS} trials at "
                    f"iteration {k}; ‖∇f‖∞ = {ginf:.3g} > gtol"
                )
        total_trials += len(trials)

        if accepted_point is None:
            return done(False, fail or "no step", k - 1, eig)
        xp, fp, gp, lam, H_used, s, pd = accepted_point
        descent = bool(float(g @ s) < 0.0)
        x, fx, g = xp, fp, gp
        if math.isfinite(fx) and finite(g):
            B = np.asarray(_eval(hess, x), dtype=np.float64)
            eig = eig_of(B)
        else:
            B, eig = np.full((n, n), np.nan), None
        info = state_info(g, B, eig) | {
            "direction": _vec(s),
            "lambda": lam,
            "H_reg": H_used,
            "trials": trials,
            "inner_iters": len(trials),
            "pd": pd,
            "descent": descent,
        }
        gn = float(np.linalg.norm(g)) if finite(g) else None
        fun = fx if math.isfinite(fx) else None
        trace.append(Step(k, x.copy(), fun, gn, float(np.linalg.norm(s)), info))
        if not (math.isfinite(fx) and finite(g)) or eig is None:
            return done(False, f"f, ∇f or ∇²f is not finite at the iterate of step {k}", k, eig)

    ginf = float(np.max(np.abs(g)))
    if ginf <= gtol:
        return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", max_iter, eig)
    return done(False, f"reached max_iter={max_iter}", max_iter, eig)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("arc", "rosenbrock", {}),
    # x0 = (0, 0) lies near Himmelblau's local maximizer; ARC reaches the minimizer (3, 2).
    ("arc", "himmelblau", {}),
    # A start next to the saddle point (0, 0) of the six-hump camel: ARC escapes it.
    ("arc", "six_hump_camel", {"x0": [0.05, 0.05]}),
    ("reg_newton", "rosenbrock", {}),
    ("reg_newton", "beale", {"variant": "super_universal"}),
    # Nonconvex: the fixed variant is attracted to the local maximizer (−0.2708, −0.9230) and
    # stops there with converged=False (the second-order part of the stopping test).
    ("reg_newton", "himmelblau", {"variant": "fixed", "H": 0.5}),
]
