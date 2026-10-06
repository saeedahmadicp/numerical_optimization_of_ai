"""Anderson acceleration AA(m) with the RNA Tikhonov term, for fixed-point maps and gradient descent.

Fixed-point problem x = g(x) with residual f(x) = g(x) − x. With m_k = min(m, k), the iterates
x_{k−m_k}, …, x_k and residuals F_k = [f_{k−m_k}, …, f_k] ∈ R^{n×(m_k+1)}, AA(m) solves
(Walker & Ni 2011, Alg. AA and eq. (1.1); RNA term from Scieur, d'Aspremont & Bach 2016,
Alg. 2 of arXiv:1606.04133, with the λ scaling of their Alg. 3, step 3)

    c* = argmin_{1ᵀc = 1} ‖F_k c‖₂² + λ' ‖c‖₂²,        λ' = λ ‖F_k‖₂²,

and sets x_{k+1} = Σ_i c*_i [(1 − β) x_i + β g(x_i)]  (β = 1: x_{k+1} = Σ_i c*_i g(x_i)).

The constraint is eliminated as in Walker & Ni's unconstrained least-squares form: with
ΔF = [f_{i+1} − f_i], ΔX = [x_{i+1} − x_i] (n × m_k) and c = e_{m_k} + Dγ, where D is the
(m_k+1) × m_k bidiagonal matrix with D_jj = 1, D_{j+1,j} = −1,

    γ* = argmin_γ ‖ [f_k; √λ' e_{m_k}] − [ΔF; −√λ' D] γ ‖₂,
    x_{k+1} = x_k + β f_k − (ΔX + β ΔF) γ*.

The stacked problem is solved by an SVD-based least-squares solve (``np.linalg.lstsq``), never by
the normal equations (they square κ(ΔF); Higham 2002, ch. 20). For λ = 0 this is exactly the
Walker–Ni unconstrained form. Facts the tests check:

* m_k = 0 (or m = 0) is the damped Picard step x_{k+1} = x_k + β f_k.
* Linear g(x) = Mx + b, m = ∞: x̄_k = Σ c*_i x_i is the k-th GMRES iterate for (I − M)x = b with
  the same x_0, ‖F_k c*‖ is its residual norm and x_{k+1} = g(x̄_k) (Walker & Ni, Thm. 2.2).
* AA is the multisecant method x_{k+1} = x_k − H_k f_k with
  H_k = −βI + (ΔX + βΔF)(ΔFᵀΔF)⁻¹ΔFᵀ, which satisfies H_k ΔF = ΔX (Fang & Saad 2009, Type-II).

Info keys (Step k ≥ 1 describes how x_k was produced from x_{k−1}):
    memory: int               m_{k−1}, the number of residual differences used.
    coefficients: [m+1]       c*, the affine weights of x_{k−1−m}, …, x_{k−1} (sum to 1).
    history: [[n]...]         the iterates x_{k−1−m}, …, x_{k−1} that the weights combine.
    x_bar: [n]                Σ c*_i x_i (the GMRES iterate in the linear case).
    lsq_residual: float       ‖F c*‖₂, the optimal combined residual.
    cond: float               σ_max/σ_min of the stacked least-squares matrix (1.0 when m = 0).
    lam_eff: float            λ' = λ‖F‖₂², the absolute Tikhonov weight.
    direction: [n]            x_k − x_{k−1}.
    residual: [n]             (``anderson`` only, every step) F(x_k).
    residual_norm: float      (``anderson`` only, every step) ‖F(x_k)‖₂.
    alpha: float              (``anderson_gd`` only, every step) the gradient step lr.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from numopt.core import diff
from numopt.core.counting import Counted, finite, start_point, vector_problem
from numopt.core.registry import ParamSpec
from numopt.core.types import Problem, Result, Step, Vector, as_vector

#: Stop with converged=False when ‖x_k‖ > DIVERGENCE_FACTOR · max(1, ‖x_0‖) (as numopt systems).
DIVERGENCE_FACTOR = 1e12
_FP_ERRORS = (ArithmeticError, ValueError)

_COMMON = (
    ParamSpec(
        "m", 5, kind="int", min=0, max=100, help="Memory: number of residual differences kept."
    ),
    ParamSpec("beta", 1.0, min=0.05, max=1.0, help="Mixing (damping) β; β = 1 is undamped AA."),
    ParamSpec(
        "lam",
        0.0,
        min=0.0,
        max=1.0,
        help="RNA Tikhonov weight λ, relative to ‖F‖₂² (0 = plain Anderson).",
    ),
)

#: ParamSpecs for later promotion into the numopt registry.
PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "anderson": (
        *_COMMON,
        ParamSpec("omega", 1.0, min=-5.0, max=5.0, help="g(x) = x + ω F(x); ω = 1 when F = g − x."),
        ParamSpec("ftol", 1e-10, min=1e-15, max=1e-2, log=True, help="Stop when ‖F(x)‖₂ ≤ ftol."),
        ParamSpec("max_iter", 200, kind="int", min=1, max=100_000),
    ),
    "anderson_gd": (
        *_COMMON,
        ParamSpec("lr", 1e-3, min=1e-8, max=10.0, log=True, help="Step α in g(x) = x − α∇f(x)."),
        ParamSpec("gtol", 1e-6, min=1e-14, max=1e-2, log=True, help="Stop when ‖∇f(x)‖₂ ≤ gtol."),
        ParamSpec("max_iter", 5000, kind="int", min=1, max=100_000),
    ),
}

# evaluate(x) -> (f = g(x) − x, trace fun, grad norm or None, info, converged-at-x)
Evaluator = Callable[[Vector], tuple[Vector, float, float | None, dict[str, Any], bool]]


def _check(m: int, beta: float, lam: float, tol: float, max_iter: int) -> None:
    if int(m) != m or m < 0:
        raise ValueError(f"m must be an integer ≥ 0, got {m}")
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be finite and > 0, got {beta}")
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be finite and ≥ 0, got {lam}")
    if not (math.isfinite(tol) and tol >= 0):
        raise ValueError(f"tolerance must be finite and ≥ 0, got {tol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be an integer ≥ 1, got {max_iter}")


def aa_coefficients(F: Vector, lam: float = 0.0) -> tuple[Vector, Vector, float, float]:
    """Return (c, γ, cond, λ') for the residual window F = [f_0, …, f_m] (n × (m+1), m ≥ 1).

    c = argmin_{1ᵀc=1} ‖Fc‖² + λ'‖c‖², λ' = λ‖F‖₂², computed via the difference form above.
    """
    mk = F.shape[1] - 1
    dF = np.diff(F, axis=1)  # (n, mk)
    D = np.eye(mk + 1, mk) - np.eye(mk + 1, mk, k=-1)  # (mk+1, mk): c = e_mk + Dγ
    e = np.zeros(mk + 1)
    e[-1] = 1.0
    lam_eff = lam * float(np.linalg.norm(F, 2)) ** 2 if lam > 0 else 0.0
    if lam_eff > 0:
        r = math.sqrt(lam_eff)
        A = np.vstack([dF, -r * D])
        rhs = np.concatenate([F[:, -1], r * e])
    else:
        A, rhs = dF, F[:, -1]
    # NOTE: lstsq truncates singular values below ε·max(dims)·σ_max (minimum-norm γ); Walker & Ni
    # instead drop the oldest columns of a QR factor when it is ill-conditioned. Both keep
    # the step finite when ΔF is (numerically) rank-deficient.
    gamma, _, _, s = np.linalg.lstsq(A, rhs, rcond=None)
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else math.inf
    return e + D @ gamma, gamma, cond, lam_eff


def _anderson_loop(
    x: Vector,
    evaluate: Evaluator,
    m: int,
    beta: float,
    lam: float,
    max_iter: int,
) -> tuple[list[Step], Vector, bool, str, float | None]:
    """Run AA(m); return (trace, x, converged, message, final fun)."""
    x0_scale = max(1.0, float(np.linalg.norm(x)))
    f, fun, gnorm, info, done = evaluate(x)
    trace = [Step(0, x.copy(), fun, grad_norm=gnorm, info=info)]
    if not finite(f, fun):
        return trace, x, False, "non-finite residual at x0", fun
    if done:
        return trace, x, True, "tolerance met at x0", fun
    X: deque[Vector] = deque(maxlen=m + 1)  # x_{k−m_k}, …, x_k
    R: deque[Vector] = deque(maxlen=m + 1)  # f_{k−m_k}, …, f_k
    for k in range(max_iter):
        X.append(x)
        R.append(f)
        if len(X) == 1:  # m_k = 0: damped Picard step
            c, cond, lam_eff = np.ones(1), 1.0, 0.0
            x_new = x + beta * f
            Xw, Fw = x[:, None], f[:, None]
        else:
            Xw, Fw = np.stack(X, axis=1), np.stack(R, axis=1)  # (n, m_k+1)
            c, gamma, cond, lam_eff = aa_coefficients(Fw, lam)
            dX, dF = np.diff(Xw, axis=1), np.diff(Fw, axis=1)
            x_new = (
                x + beta * f - (dX + beta * dF) @ gamma
            )  # Walker & Ni unconstrained form, damped
        step_info = {
            "memory": len(X) - 1,
            "coefficients": c,
            "history": [xi.copy() for xi in X],
            "x_bar": Xw @ c,
            "lsq_residual": float(np.linalg.norm(Fw @ c)),
            "cond": cond,
            "lam_eff": lam_eff,
            "direction": x_new - x,
        }
        if not finite(x_new):
            return trace, x, False, f"non-finite iterate at k = {k + 1}", fun
        f, fun, gnorm, info, done = evaluate(x_new)
        trace.append(
            Step(
                k + 1,
                x_new.copy(),
                fun,
                grad_norm=gnorm,
                step_size=float(np.linalg.norm(x_new - x)),
                info={**step_info, **info},
            )
        )
        if not finite(f, fun):
            return trace, x_new, False, f"non-finite residual at k = {k + 1}", fun
        if done:
            return trace, x_new, True, "tolerance met", fun
        if float(np.linalg.norm(x_new)) > DIVERGENCE_FACTOR * x0_scale:
            return trace, x_new, False, f"diverged: ‖x‖ > {DIVERGENCE_FACTOR:g}·max(1, ‖x₀‖)", fun
        if np.array_equal(x_new, x):
            return trace, x_new, False, "stalled: x_{k+1} = x_k in floating point", fun
        x = x_new
    return trace, x, False, f"max_iter = {max_iter} reached", fun


def anderson(
    problem: Problem | Callable[[Vector], Vector],
    *,
    x0: ArrayLike | None = None,
    m: int = 5,
    beta: float = 1.0,
    lam: float = 0.0,
    omega: float = 1.0,
    ftol: float = 1e-10,
    max_iter: int = 200,
) -> Result:
    """Anderson acceleration AA(m) for F(x) = 0 through the map g(x) = x + ω F(x).

    Walker & Ni (2011), Alg. AA with the unconstrained least-squares form and mixing
    β; λ > 0 adds the RNA term λ‖F‖₂²‖c‖² (Scieur et al. 2016, Alg. 2, with the scaling of their
    Alg. 3, step 3). With ω = 1 and F = g − x, the map is g itself. AA residuals are f = ωF.
    Each iteration costs one F evaluation (``n_fev``); no Jacobian is used.

    # NOTE: RNA extrapolates Σc_i x_i of a fixed sequence; here the weights are applied to
    # (1 − β)x_i + βg(x_i) and the iteration restarts from the result (online AA form).

    Stops (converged) when ‖F(x_k)‖₂ ≤ ``ftol``. converged=False on a non-finite F or iterate,
    ‖x_k‖ > 1e12·max(1, ‖x₀‖), x_{k+1} = x_k in floating point, or ``max_iter``.
    """
    _check(m, beta, lam, ftol, max_iter)
    if not (math.isfinite(omega) and omega != 0):
        raise ValueError(f"omega must be finite and nonzero, got {omega}")
    if isinstance(problem, Problem):
        prob = problem
    elif callable(problem):
        if x0 is None:
            raise ValueError("a starting point x0 is required for a bare callable F")
        prob = Problem(id="custom", name="custom", latex="F(x)", f=problem, dim=0, domain=())
    else:
        raise TypeError("problem must be a numopt Problem or a callable F(x)")
    x = start_point(prob, x0)
    F = Counted(prob.f)

    def evaluate(z: Vector) -> tuple[Vector, float, float | None, dict[str, Any], bool]:
        try:
            Fz = as_vector(F(z))
        except _FP_ERRORS:
            Fz = np.full(z.size, np.nan)
        r = float(np.linalg.norm(Fz))
        return omega * Fz, r, None, {"residual": Fz, "residual_norm": r}, r <= ftol

    trace, x, ok, msg, fun = _anderson_loop(x, evaluate, int(m), beta, lam, max_iter)
    return Result(
        method="anderson",
        x=x,
        fun=fun,
        converged=ok,
        message=msg,
        n_iter=trace[-1].k,
        n_fev=F.n,
        trace=trace,
        extra={"m": int(m), "beta": beta, "lam": lam, "omega": omega},
    )


def anderson_gd(
    problem: Problem | Callable[[Vector], float],
    *,
    x0: ArrayLike | None = None,
    m: int = 5,
    lr: float = 1e-3,
    beta: float = 1.0,
    lam: float = 0.0,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Anderson-accelerated gradient descent: AA(m) on the map g(x) = x − α∇f(x), α = ``lr``.

    The AA residual is f(x) = −α∇f(x); the update and the RNA term are those of
    :func:`anderson` (Walker & Ni 2011, Alg. AA; Scieur et al. 2016, Alg. 2). AA(m)-GD is a
    multisecant quasi-Newton method (Fang & Saad 2009): H_k ΔF = ΔX with ΔF = −αΔ∇f. No line
    search and no descent safeguard: f(x_k) need not decrease.

    Each iteration costs one gradient (``n_gev``). f is evaluated once per iterate only for the
    trace's ``fun`` (``n_fev`` = n_iter + 1); it does not influence the iterates. Without
    ``problem.grad`` the gradient is a central difference (2n f evaluations, in ``n_fev``).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. converged=False on a non-finite value,
    ‖x_k‖ > 1e12·max(1, ‖x₀‖), x_{k+1} = x_k in floating point, or ``max_iter``.
    """
    _check(m, beta, lam, gtol, max_iter)
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError(f"lr must be finite and > 0, got {lr}")
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    fc = Counted(prob.f)
    gc = Counted(prob.grad if prob.grad is not None else (lambda z: diff.gradient(fc, z)))

    def evaluate(z: Vector) -> tuple[Vector, float, float | None, dict[str, Any], bool]:
        try:
            gz = as_vector(gc(z))
            fz = float(fc(z))
        except _FP_ERRORS:
            gz, fz = np.full(z.size, np.nan), math.nan
        gn = float(np.linalg.norm(gz))
        return -lr * gz, fz, gn, {"alpha": lr}, gn <= gtol

    trace, x, ok, msg, fun = _anderson_loop(x, evaluate, int(m), beta, lam, max_iter)
    return Result(
        method="anderson_gd",
        x=x,
        fun=fun,
        converged=ok,
        message=msg,
        n_iter=trace[-1].k,
        n_fev=fc.n,  # includes the central-difference calls when grad is missing
        n_gev=gc.n if prob.grad is not None else 0,
        trace=trace,
        extra={"m": int(m), "beta": beta, "lam": lam, "lr": lr},
    )
