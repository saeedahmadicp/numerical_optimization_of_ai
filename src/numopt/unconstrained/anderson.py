"""Anderson acceleration of gradient descent, AA(m) (Walker & Ni 2011), with the RNA term.

Gradient descent with the fixed step α is the fixed-point iteration x ← g(x) = x − α∇f(x). AA(m)
accelerates it with the residuals f_i = g(x_i) − x_i = −α∇f(x_i) of the last m_k + 1 iterates,
m_k = min(m, k). With F_k = [f_{k−m_k}, …, f_k] (n × (m_k + 1)) it solves (Walker & Ni (2011),
Alg. AA and eq. (1.1); the Tikhonov term λ' is the RNA regularization of Scieur, d'Aspremont &
Bach (2016), Alg. 2, scaled as in their Alg. 3, step 3)

    c* = argmin_{1ᵀc = 1} ‖F_k c‖₂² + λ'‖c‖₂²,        λ' = λ‖F_k‖₂²,

and moves to x_{k+1} = Σ_i c*_i [(1 − β) x_i + β g(x_i)]  (mixing β; β = 1: Σ_i c*_i g(x_i)).

The constraint is removed as in Walker & Ni's unconstrained least-squares form. With the
difference matrices ΔF = [f_{i+1} − f_i] and ΔX = [x_{i+1} − x_i] (n × m_k) and c = e + Dγ, where
e = (0, …, 0, 1) and D is the (m_k + 1) × m_k bidiagonal matrix with D_jj = 1, D_{j+1,j} = −1,

    γ* = argmin_γ ‖ [f_k; √λ' e] − [ΔF; −√λ' D] γ ‖₂,
    x_{k+1} = x_k + β f_k − (ΔX + β ΔF) γ*.

λ = 0 is plain AA. m_k = 0 (the first iteration, or m = 0) is the damped gradient step
x_{k+1} = x_k − βα∇f(x_k); AA(0) with β = 1 is gradient descent with the fixed step α.

Facts that the tests check:

* **GMRES.** On a strictly convex quadratic, g is affine, and untruncated AA (m ≥ n) is GMRES on
  ∇f(x) = 0 (Walker & Ni (2011), Thm. 2.2): x̄_k = Σ c*_i x_i is the k-th GMRES iterate,
  ‖F_k c*‖ is α times its residual norm, and x_{n+1} = x* in exact arithmetic.
* **Multisecant form.** x_{k+1} = x_k − H_k f_k with H_k = −βI + (ΔX + βΔF)(ΔFᵀΔF)⁻¹ΔFᵀ, which
  satisfies the secant equations H_k ΔF = ΔX (Fang & Saad (2009), Type-II). AA(m)-GD is a
  quasi-Newton method without a line search.

Conventions:

* **Linear algebra.** The stacked least-squares problem is solved by ``numpy.linalg.lstsq`` (SVD,
  minimum-norm solution), never through the normal equations, which square κ(ΔF) (Higham (2002),
  ch. 20). Singular values below ε·max(rows, cols)·σ_max are treated as zero, so a numerically
  rank-deficient ΔF gives a finite step.
* **No descent safeguard.** AA solves ∇f(x) = 0. f(x_k) need not decrease, and every stationary
  point attracts the iteration: saddle points and maximizers as well as minimizers. The study
  ``research/anderson-acceleration`` measured this: plain AA(m)-GD stopped at a saddle point or a
  maximizer from 28–32 of 64 starts on ``himmelblau`` (gradient descent: 0), and at a saddle point
  in 12 of 40 tuned runs on ``rosenbrock_nd``. λ = 10⁻² cut the himmelblau count to 11–17, at a
  cost of 1.9–2.8× more iterations; λ ≤ 10⁻⁴ changed it by at most 4.
* **Stopping test (converged).** ‖∇f(x_k)‖₂ ≤ ``gtol`` *and* ∇²f(x_k) has no eigenvalue below
  −tol_eff: the second-order necessary conditions hold to the accuracy of the test (Nocedal &
  Wright (2006), Thms 2.3–2.4). ∇²f is evaluated only at a point that passes the gradient test.
  tol_eff = max(tol_H, tol_g) combines two error sources:

  - tol_H, the accuracy of ∇²f itself: n·ε·|λ|_max for an analytic ∇²f; for a
    central-difference ∇²f, n·ε^{1/3}·max(1, |λ|_max), with |f(x)| added to the max when ∇f is a
    central difference too (the eigenvalue tolerance of ``numopt.unconstrained.newton``).
  - tol_g = √(‖∇f(x_k)‖₂·|λ|_max), the distance from stationarity. ∇²f is evaluated at x_k, not
    at a stationary point x*, and λ_i(∇²f) moves by up to ρ‖x_k − x*‖ between them (Weyl; ρ the
    Lipschitz constant of ∇²f). Near a non-isolated minimizer (a valley of minimizers such as
    (‖x‖² − 1)²) λ_min(∇²f(x_k)) ≈ −c‖∇f(x_k)‖ < 0 although every point of the valley is a
    global minimizer. tol_g is the ε-second-order stationarity test λ_min ≥ −√(ρε), ε = ‖∇f‖, of
    Nesterov & Polyak (2006) and Jin et al. (2017), with the unknown ρ replaced by |λ|_max.
    Measured (AA(0, 1, 3, 5), α = 0.05, 50 starts each): stops on the valleys of (‖x‖² − 1)² and
    (xy − 1)² have |λ_min| ≤ 4·10⁻⁴·tol_g; the saddle points and maximizers of those problems,
    of ``himmelblau`` (E4) and of ``rosenbrock_nd`` have |λ_min| ≥ 190·tol_g.

  A point with an eigenvalue below −tol_eff stops the run with ``converged=False``; the message
  classifies it as a saddle point (eigenvalues of both signs), a maximizer (∇²f ≺ 0), or "a
  saddle point or a maximizer" (∇²f ⪯ 0 and singular), each to the accuracy ±tol_eff. A ∇²f that
  is positive *semi*definite to that accuracy (λ_min ∈ [−tol_eff, tol_eff]) is reported as
  converged, and the message says which accuracy applies and that the second-order sufficient
  condition is not verified. "Positive definite: a strict local minimizer" needs
  λ_min > tol_eff. The price of tol_g: a saddle point whose negative curvature is weaker than
  √(‖∇f‖·|λ|_max) (for example ≈ 3·10⁻³ at ‖∇f‖ = 10⁻⁶, |λ|_max = 10) is reported as
  converged, with that message.
* **Failures** (``converged=False``): a stationary point that is not a minimizer (above); a
  non-finite f, ∇f, ∇²f, iterate, or residual difference (an overflow; the SVD is then not
  tried); a failed SVD; divergence ‖x_k‖₂ > 10¹²·max(1, ‖x_0‖₂) at a point that fails the
  gradient test; a stall x_{k+1} = x_k in floating point; ``max_iter`` reached. Nothing is
  raised for a numerical failure; ``ValueError`` only for an invalid parameter or start point.
* **Evaluation counts.** One ∇f per iterate (n_gev = n_iter + 1 unless a failure stops the run
  first). f is evaluated once per iterate only for the trace's ``fun``; it does not influence
  the iterates. One ∇²f at a point that passes the gradient test (n_hev ≤ 1). Without an
  analytic gradient, ∇f is a central difference (``numopt.core.diff.gradient``) whose 2n f
  evaluations are counted in ``n_fev``; without an analytic Hessian, ∇²f is a central
  difference of ∇f (``numopt.core.diff.hessian``) whose 2n gradient evaluations are counted in
  ``n_gev``.

Trace: one Step per iterate; k = 0 is x_0. ``Step.step_size`` is ‖x_k − x_{k−1}‖₂ (``None`` at
k = 0). ``n_iter == trace[-1].k``.

Info keys (every step). Keys marked "incoming" describe how x_k was produced from x_{k−1}; they
are ``None`` (``[]`` for ``coefficients`` and ``history``) at k = 0:
    grad: [n]                  ∇f(x_k).
    alpha: float               the gradient step α of g(x) = x − α∇f(x) (constant).
    memory: int | None         incoming: m_{k−1}, the number of residual differences used.
    coefficients: [m+1]        incoming: c*, the affine weights of x_{k−1−m}, …, x_{k−1}
                               (they sum to 1; [1.0] for a gradient step).
    history: [[n]...]          incoming: the iterates x_{k−1−m}, …, x_{k−1} that c* combines.
    x_bar: [n] | None          incoming: Σ c*_i x_i (the GMRES iterate on a quadratic).
    lsq_residual: float | None incoming: ‖F c*‖₂, the optimal combined residual.
    cond: float | None         incoming: σ_max/σ_min of the stacked least-squares matrix (1.0 for
                               a gradient step; ``inf`` when the matrix is singular).
    lam_eff: float | None      incoming: λ' = λ‖F‖₂², the absolute Tikhonov weight.
    direction: [n] | None      incoming: x_k − x_{k−1}.
    hess_eigs: [n] | None      the eigenvalues of ∇²f(x_k) in ascending order, only at a step
                               that passes the gradient test (the second-order test); else None.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, as_vector

Array = NDArray[np.float64]

#: Stop with converged=False when ‖x_k‖₂ > _DIVERGENCE_FACTOR · max(1, ‖x_0‖₂).
_DIVERGENCE_FACTOR = 1e12
#: Machine epsilon of float64.
_EPS = float(np.finfo(np.float64).eps)
#: Relative accuracy of a central-difference Hessian (step h = ε^{1/3}, N&W §8.1).
_FD_HESS_REL = _EPS ** (1.0 / 3.0)
#: Errors that a user's f or ∇f can raise on overflow; they are reported, not raised.
_FP_ERRORS = (ArithmeticError, ValueError)


# --------------------------------------------------------------------------------------
# The AA coefficient problem
# --------------------------------------------------------------------------------------


def _aa_coefficients(F: Array, lam: float) -> tuple[Array, Array, float, float]:
    """Return (c*, γ*, cond, λ') for the residual window F = [f_0, …, f_{m_k}], m_k ≥ 1.

    c* = argmin_{1ᵀc=1} ‖Fc‖₂² + λ'‖c‖₂², λ' = λ‖F‖₂², through the difference form of the
    module docstring: c* = e + Dγ*.
    """
    mk = F.shape[1] - 1
    dF = np.diff(F, axis=1)  # (n, mk)
    D = np.eye(mk + 1, mk) - np.eye(mk + 1, mk, k=-1)  # (mk+1, mk), c = e + Dγ
    e = np.zeros(mk + 1)
    e[-1] = 1.0
    lam_eff = lam * float(np.linalg.norm(F, 2)) ** 2 if lam > 0.0 else 0.0
    if lam_eff > 0.0:
        r = math.sqrt(lam_eff)
        A = np.vstack([dF, -r * D])  # (n + mk + 1, mk)
        rhs = np.concatenate([F[:, -1], r * e])
    else:
        A, rhs = dF, F[:, -1]
    # NOTE: lstsq treats singular values below ε·max(rows, cols)·σ_max as zero and returns the
    # minimum-norm γ*; Walker & Ni instead drop the oldest columns of a QR factor of ΔF when it
    # is ill-conditioned. Both keep the step finite for a numerically rank-deficient ΔF (for
    # example m > n, where ΔF has more columns than rows).
    gamma, _, _, s = np.linalg.lstsq(A, rhs, rcond=None)
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0.0 else math.inf
    return e + D @ gamma, np.asarray(gamma, dtype=np.float64), cond, lam_eff


# --------------------------------------------------------------------------------------
# Second-order test at a point that passes the gradient test
# --------------------------------------------------------------------------------------


def _eig_tol(lam: Array, fx: float, hess_fd: bool, grad_fd: bool) -> float:
    """tol_H of the module docstring: the accuracy of the eigenvalues of ∇²f itself."""
    n = lam.size
    lam_max = float(np.abs(lam).max())
    if not hess_fd:
        return n * _EPS * lam_max
    scale = max(1.0, lam_max, abs(fx) if grad_fd else 0.0)
    return n * _FD_HESS_REL * scale


def _stationarity_tol(lam: Array, gnorm: float) -> float:
    """tol_g = √(‖∇f(x)‖₂·|λ|_max): the eigenvalue shift allowed by the distance from x* (docstring).

    # NOTE: Nesterov & Polyak (2006) and Jin et al. (2017) use √(ρ‖∇f‖) with ρ the Lipschitz
    # constant of ∇²f, which one ∇²f evaluation cannot estimate; |λ|_max stands in for ρ (per
    # unit length of x). tol_g is therefore not invariant to a rescaling of x; tol_H is.
    """
    return math.sqrt(gnorm * float(np.abs(lam).max()))


def _classify(lam: Array, tol_h: float, tol_g: float, hess_fd: bool) -> tuple[bool, str]:
    """(converged, why) from the ascending eigenvalues of ∇²f (N&W Thms 2.3–2.4).

    tol_h: the accuracy of ∇²f; tol_g: the allowance for ‖∇f(x)‖ > 0 (module docstring).
    """
    tol = max(tol_h, tol_g)
    lam_min, lam_top = float(lam[0]), float(lam[-1])
    source = "the finite-difference ∇²f" if hess_fd else "∇²f"
    if lam_min < -tol:
        if lam_top < -tol:
            kind = "a maximizer"
        elif lam_top > tol:
            kind = "a saddle point"
        else:
            # ∇²f ⪯ 0 and singular: −x² + y⁴ and −x² − y⁴ have the same ∇²f at 0.
            # (+ 0.0 turns a signed zero −0 into 0 for the message.)
            kind = (
                f"a saddle point or a maximizer (λ_max = {lam_top + 0.0:.3g}: ∇²f is singular, "
                "so the second-order test cannot decide)"
            )
        return False, (
            f"stopped at {kind}, not a minimizer: {source} has the eigenvalue "
            f"λ_min = {lam_min:.3g} < −{tol:.3g}"
        )
    if lam_min <= tol:
        if abs(lam_min) > tol_h:
            # Within tol_g but not within tol_h: only the distance from stationarity explains it.
            return True, (
                f"{source} has λ_min = {lam_min:.3g}, within √(‖∇f‖·|λ|_max) = {tol_g:.3g} of 0: "
                "∇²f is positive semidefinite to the accuracy of the gradient test, and the "
                "second-order sufficient condition is not verified"
            )
        if hess_fd:
            return True, (
                f"the finite-difference ∇²f has λ_min = {lam_min:.3g}, within its accuracy "
                f"±{tol_h:.3g} of 0: ∇²f is positive semidefinite to that accuracy, and the "
                "second-order sufficient condition is not verified"
            )
        return True, (
            "∇²f is positive semidefinite but numerically singular "
            f"(λ_min = {lam_min:.3g}): the second-order sufficient condition is not verified"
        )
    return True, f"{source} is positive definite (λ_min = {lam_min:.3g}): a strict local minimizer"


# --------------------------------------------------------------------------------------
# The method
# --------------------------------------------------------------------------------------


def _incoming_none() -> dict[str, Any]:
    return {
        "memory": None,
        "coefficients": [],
        "history": [],
        "x_bar": None,
        "lsq_residual": None,
        "cond": None,
        "lam_eff": None,
        "direction": None,
    }


def _validate(m: int, lr: float, beta: float, lam: float, gtol: float, max_iter: int) -> None:
    if int(m) != m or m < 0:
        raise ValueError(f"m must be an integer ≥ 0, got {m}")
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be finite and > 0, got {lr}")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be finite and > 0, got {beta}")
    if not (math.isfinite(lam) and lam >= 0.0):
        raise ValueError(f"lam must be finite and ≥ 0, got {lam}")
    if not (math.isfinite(gtol) and gtol >= 0.0):
        raise ValueError(f"gtol must be finite and ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")


def _run(
    problem: Problem | Callable[..., Any],
    x0: Any,
    m: int,
    lr: float,
    beta: float,
    lam: float,
    gtol: float,
    max_iter: int,
) -> Result:
    """The AA(m)-GD iteration of :func:`anderson_gd` (inputs already validated)."""
    prob = problem if isinstance(problem, Problem) else vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    n = x.size
    fc = Counted(prob.f)
    gc = Counted(prob.grad if prob.grad is not None else (lambda z: diff.gradient(fc, z)))
    hc = Counted(prob.hess if prob.hess is not None else (lambda z: diff.hessian(gc, z)))
    grad_fd, hess_fd = prob.grad is None, prob.hess is None

    def evaluate(z: Array) -> tuple[float, Array]:
        try:
            gz = as_vector(gc(z))
            fz = float(fc(z))
        except _FP_ERRORS:
            return math.nan, np.full(n, np.nan)
        return fz, gz

    def stopping_test(z: Array, fz: float, gz: Array) -> tuple[bool, str, Array | None] | None:
        """None to continue; else (converged, message, ascending eigenvalues of ∇²f or None)."""
        gnorm = float(np.linalg.norm(gz))
        if gnorm > gtol:
            return None
        head = f"‖∇f‖₂ = {gnorm:.3g} ≤ gtol"
        try:
            H = np.asarray(hc(z), dtype=np.float64).reshape(n, n)
        except _FP_ERRORS:
            H = np.full((n, n), np.nan)
        if not finite(H):
            return False, f"{head}, but ∇²f is not finite there", None
        # NOTE: symmetrize ∇²f before eigvalsh, which reads one triangle only; an analytic
        # Hessian can be asymmetric in its last bit.
        eigs = np.asarray(np.linalg.eigvalsh(0.5 * (H + H.T)), dtype=np.float64)
        tol_h = _eig_tol(eigs, fz, hess_fd, grad_fd)
        ok, why = _classify(eigs, tol_h, _stationarity_tol(eigs, gnorm), hess_fd)
        return ok, f"{head}; {why}", eigs

    def result(z: Array, fz: float, converged: bool, message: str, trace: list[Step]) -> Result:
        return Result(
            method="anderson_gd",
            x=z,
            fun=fz,
            converged=converged,
            message=message,
            n_iter=trace[-1].k,
            n_fev=fc.n,
            n_gev=gc.n,
            n_hev=hc.n,
            trace=trace,
            extra={"m": m, "lr": lr, "beta": beta, "lam": lam},
        )

    x0_scale = max(1.0, float(np.linalg.norm(x)))
    fx, g = evaluate(x)
    if not finite(fx, g):
        info = {"grad": g, "alpha": lr, **_incoming_none(), "hess_eigs": None}
        trace = [Step(0, x.copy(), fx, grad_norm=float(np.linalg.norm(g)), info=info)]
        return result(x, fx, False, "f or ∇f is not finite at x0", trace)
    stop = stopping_test(x, fx, g)
    info = {
        "grad": g,
        "alpha": lr,
        **_incoming_none(),
        "hess_eigs": None if stop is None else stop[2],
    }
    trace = [Step(0, x.copy(), fx, grad_norm=float(np.linalg.norm(g)), info=info)]
    if stop is not None:
        return result(x, fx, stop[0], f"at x0: {stop[1]}", trace)

    X: deque[Array] = deque(maxlen=m + 1)  # x_{k−m_k}, …, x_k
    R: deque[Array] = deque(maxlen=m + 1)  # f_{k−m_k}, …, f_k with f_i = −α∇f(x_i)
    for k in range(max_iter):
        f_k = -lr * g
        X.append(x)
        R.append(f_k)
        if len(X) == 1:  # m_k = 0: the damped gradient step
            c, cond, lam_eff = np.ones(1), 1.0, 0.0
            x_new = x + beta * f_k
            Xw, Fw = x[:, None], f_k[:, None]
        else:
            Xw, Fw = np.stack(X, axis=1), np.stack(R, axis=1)  # (n, m_k + 1)
            dX, dF = np.diff(Xw, axis=1), np.diff(Fw, axis=1)  # (n, m_k)
            if not finite(f_k, dX, dF):
                # An overflow in α∇f or in a difference (|entries| near 10³⁰⁸): no SVD is tried.
                return result(
                    x, fx, False, f"non-finite residual difference at iteration {k + 1}", trace
                )
            try:
                c, gamma, cond, lam_eff = _aa_coefficients(Fw, lam)
            except np.linalg.LinAlgError:
                return result(
                    x, fx, False, f"the SVD least-squares solve failed at iteration {k + 1}", trace
                )
            x_new = x + beta * f_k - (dX + beta * dF) @ gamma
        incoming = {
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
            return result(x, fx, False, f"non-finite iterate at iteration {k + 1}", trace)
        fx_new, g_new = evaluate(x_new)
        ok_values = finite(fx_new, g_new)
        stop = stopping_test(x_new, fx_new, g_new) if ok_values else None
        trace.append(
            Step(
                k + 1,
                x_new.copy(),
                fx_new,
                grad_norm=float(np.linalg.norm(g_new)),
                step_size=float(np.linalg.norm(x_new - x)),
                info={
                    "grad": g_new,
                    "alpha": lr,
                    **incoming,
                    "hess_eigs": None if stop is None else stop[2],
                },
            )
        )
        if not ok_values:
            return result(
                x_new, fx_new, False, f"f or ∇f is not finite at iteration {k + 1}", trace
            )
        if stop is not None:
            return result(x_new, fx_new, stop[0], stop[1], trace)
        if float(np.linalg.norm(x_new)) > _DIVERGENCE_FACTOR * x0_scale:
            return result(
                x_new,
                fx_new,
                False,
                f"diverged: ‖x‖₂ > {_DIVERGENCE_FACTOR:g}·max(1, ‖x₀‖₂) at iteration {k + 1}",
                trace,
            )
        if np.array_equal(x_new, x):
            return result(
                x_new, fx_new, False, f"stalled: x_{k + 1} = x_{k} in floating point", trace
            )
        x, fx, g = x_new, fx_new, g_new
    return result(x, fx, False, f"max_iter = {max_iter} reached", trace)


@register(
    id="anderson_gd",
    family="unconstrained",
    name="Anderson-accelerated gradient descent",
    params=(
        ParamSpec(
            "m",
            5,
            kind="int",
            min=0,
            max=50,
            help="Memory: the number of residual differences kept (m = 0 is gradient descent).",
        ),
        ParamSpec(
            "lr",
            1e-3,
            min=1e-6,
            max=1.0,
            log=True,
            help="Gradient step α of the map g(x) = x − α∇f(x).",
        ),
        ParamSpec(
            "beta",
            1.0,
            min=0.05,
            max=1.0,
            help="Mixing β: x⁺ = Σ cᵢ[(1 − β)xᵢ + β g(xᵢ)]; β = 1 is undamped AA.",
        ),
        ParamSpec(
            "lam",
            0.0,
            min=0.0,
            max=0.1,
            help="RNA Tikhonov weight λ, relative to ‖F‖₂² (0 = plain Anderson).",
        ),
        ParamSpec(
            "gtol",
            1e-6,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Stop when ‖∇f(x)‖₂ ≤ gtol (then ∇²f is checked for a minimizer).",
        ),
        ParamSpec("max_iter", 500, kind="int", min=1, max=100_000, help="Iteration limit."),
    ),
    needs=("f", "grad"),
    order="linear (r-linear near a point where g is a contraction)",
    summary=(
        "Combine the last m gradient steps with least-squares weights; fast, but it can stop at "
        "a saddle point."
    ),
    references=(
        "Walker & Ni (2011), SIAM J. Numer. Anal. 49(4), Alg. AA and eq. (1.1); Thm. 2.2 (GMRES)",
        "Anderson (1965), J. ACM 12(4)",
        "Fang & Saad (2009), Numer. Linear Algebra Appl. 16, Type-II multisecant form",
        "Scieur, d'Aspremont & Bach (2016), arXiv:1606.04133, Alg. 2; λ scaling of Alg. 3, step 3",
        "Nocedal & Wright (2006), Thms 2.3–2.4 (second-order test at the stopping point)",
        "Nesterov & Polyak (2006), Math. Program. 108; Jin et al. (2017), ICML "
        "(ε-second-order stationarity, λ_min ≥ −√(ρε))",
    ),
)
def anderson_gd(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    m: int = 5,
    lr: float = 1e-3,
    beta: float = 1.0,
    lam: float = 0.0,
    gtol: float = 1e-6,
    max_iter: int = 500,
) -> Result:
    """Anderson-accelerated gradient descent AA(m) on g(x) = x − α∇f(x), α = ``lr``.

    Walker & Ni (2011), Alg. AA in the unconstrained least-squares form (eq. (1.1)), with mixing
    β; λ > 0 adds the RNA term λ‖F‖₂²‖c‖² (Scieur et al. (2016), Alg. 2, with the scaling of
    their Alg. 3, step 3). No line search and no descent safeguard: f(x_k) need not decrease,
    and the run can stop at a saddle point or a maximizer, which is reported with
    ``converged=False``.

    # NOTE: RNA extrapolates Σ c_i x_i of a fixed sequence; here the RNA weights enter the AA
    # step and the iteration restarts from the result (the online form). Only λ is taken from
    # RNA; its adaptive λ grid and step search (Alg. 3) are not implemented.

    Stopping test: ‖∇f(x_k)‖₂ ≤ ``gtol`` and ∇²f(x_k) has no eigenvalue below
    −max(tol_H, √(‖∇f(x_k)‖₂·|λ|_max)) (module docstring). converged=False at a saddle point or maximizer, on a non-finite value,
    divergence ‖x_k‖₂ > 10¹²·max(1, ‖x_0‖₂), a stall x_{k+1} = x_k, or ``max_iter``.
    """
    _validate(m, lr, beta, lam, gtol, max_iter)
    # NOTE: overflow and invalid-operation warnings are silenced: a non-finite f, ∇f, ∇²f or
    # iterate is detected and reported in the Result (converged=False), never hidden.
    with np.errstate(over="ignore", invalid="ignore"):
        return _run(problem, x0, int(m), lr, beta, lam, gtol, max_iter)


#: Parity fixtures for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    # A strictly convex quadratic in 2-D: AA(m ≥ n) is GMRES and stops at x_{n+1} = x*.
    ("anderson_gd", "quadratic_ill", {"lr": 0.01}),
    # Not rosenbrock: AA is chaotic in its curved valley (a one-ulp change of x0 or of one
    # lstsq result changes the stop from 176 to 160..330 iterations), so another CPU would
    # write another trace. six_hump_camel is nonconvex and insensitive to it.
    ("anderson_gd", "six_hump_camel", {}),
    # The same start: plain AA stops at a saddle point; λ = 10⁻² reaches the minimizer (3, 2).
    ("anderson_gd", "himmelblau", {"x0": [1.0, 1.0], "lr": 0.01}),
    ("anderson_gd", "himmelblau", {"x0": [1.0, 1.0], "lr": 0.01, "lam": 0.01}),
    ("anderson_gd", "beale", {"beta": 0.5}),
    # 10-D: the default start stops at the saddle point of the study (f = 9.606).
    ("anderson_gd", "rosenbrock_nd", {}),
]
