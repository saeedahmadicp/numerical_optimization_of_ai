"""Four Adam successors (AdaBelief, Lion, Adan, Sophia) and a rotation wrapper for problems.

Every optimizer here is *coordinate-wise*: each update applies the same scalar map to every
coordinate of (gradient, state). Such a method is equivariant under permutations and sign
flips of the coordinates, but not under rotations. ``rotate(problem, R)`` builds f(R x), which
has the same Hessian spectrum as f, so a rotation-equivariant method (gradient descent) takes
the same number of iterations on both, while a coordinate-wise method can change.

The methods follow the ``numopt`` method contract (``docs/architecture.md``) and the trace
conventions of ``numopt.unconstrained.first_order``:

* Signature ``fn(problem, *, x0=None, [seed=None], **params) -> Result``. ``problem`` is a
  ``Problem`` or a bare callable ``f`` (then ``x0`` is required, and a missing gradient is a
  central difference, ``numopt.core.diff.gradient``).
* **Trace.** Step ``k = 0`` holds x₀; step ``k ≥ 1`` holds the iterate x_k made by iteration
  k from g_k = ∇f(x_{k−1}). ``Step.fun = f(x_k)``, ``Step.grad_norm = ‖∇f(x_k)‖₂`` and
  ``Step.step_size = α_k``, the learning rate used by iteration k. Every update is written
  x_k = x_{k−1} + α_k p_k. ``n_iter == trace[-1].k``.
* **Stopping test (converged).** ‖∇f(x_k)‖₂ ≤ ``gtol``, checked at x₀ and after every
  iteration.
* **Failure (converged=False).** ``max_iter`` reached; f(x_k) or ∇f(x_k) not finite; or
  f(x_k) − f(x₀) > 1e12·max(1, |f(x₀)|) (divergence, the same test as ``first_order``).
* **Counts.** ``n_fev`` / ``n_gev`` / ``n_hev`` count f, ∇f and ∇²f calls exactly. One f and
  one ∇f per iteration (plus one at x₀). Sophia adds one ∇²f call per Hessian refresh (a
  central-difference Hessian from 2n gradient calls when the problem has no ``hess``).

Info keys (every method, every step):
    grad: [n]               ∇f(x_k).
    direction: [n] | null   p_k with x_k = x_{k−1} + alpha·p_k (null at k = 0).
    alpha: float | null     α_k (= Step.step_size; null at k = 0).

Additional info keys:
    adabelief:
        m, s: [n]             first moment and EMA of (g − m)² (with the +ε of Alg. 2).
        m_hat, s_hat: [n]     bias-corrected moments (zeros at k = 0).
        lr_eff: [n] | null    lr/(√ŝ_k + ε), multiplying m̂_k.
    lion:
        m: [n]                EMA of the gradient after iteration k (β₂ average).
        c: [n] | null         the interpolation β₁m_{k−1} + (1 − β₁)g_k whose sign is the step.
        decay: [n] | null     −λ x_{k−1}, the decoupled weight-decay part of p_k.
    adan:
        m, v, n: [n]          the three EMAs of Alg. 1 (gradient, gradient difference, squared
                              Nesterov gradient) used by iteration k (at k = 0: their values
                              before the first step, m₀ = g₀, v₀ = 0, n₀ = g₀²).
        lr_eff: [n] | null    η/(√n + ε), multiplying m + (1 − β₂)v.
    sophia:
        m: [n]                EMA of the gradient.
        h: [n]                EMA of the diagonal-Hessian estimates.
        h_new: [n] | null     the estimate ĥ computed in this iteration (null when h is reused).
        ratio: [n] | null     m_k / max(γ h_k, ε) before clipping.
        clipped: [bool] | null  True where |ratio| > 1 (that coordinate took a sign step).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt.core import diff
from numopt.core.counting import Counted, start_point, vector_problem
from numopt.core.registry import ParamSpec
from numopt.core.rng import Rng
from numopt.core.types import Problem, Result, Step, Vector, as_vector

Fn = Callable[[Any], Any]

#: A rise f(x_k) − f(x₀) above F_DIVERGE·max(1, |f(x₀)|) counts as divergence (as in
#: ``numopt.unconstrained.first_order``).
F_DIVERGE = 1e12

#: Smallest accepted gtol (‖g‖₂ underflows to 0 below ~1e-154).
GTOL_MIN = 1e-150

ESTIMATORS = ("exact", "hutchinson")


# --------------------------------------------------------------------------------------
# Parameter specs (for later promotion into the registry)
# --------------------------------------------------------------------------------------


def _p_lr(default: float) -> ParamSpec:
    # NOTE: max = 100 covers the learning-rate grids of run.py (up to 1e2). Values far above the
    # distance from x₀ to x* make the first steps overshoot; run.py flags such trajectories.
    return ParamSpec(
        "lr",
        default,
        min=1e-6,
        max=100.0,
        log=True,
        help="Learning rate η (values ≫ ‖x₀ − x*‖ overshoot on the first steps).",
    )


def _p_beta(name: str, default: float, help: str) -> ParamSpec:
    return ParamSpec(name, default, min=0.0, max=0.9999, help=help)


P_GTOL = ParamSpec("gtol", 1e-6, min=1e-14, max=1e-2, log=True, help="Stop when ‖∇f(x)‖₂ ≤ gtol.")
P_MAX_ITER = ParamSpec("max_iter", 5000, kind="int", min=1, max=100_000, help="Iteration limit.")

PARAMS: dict[str, list[ParamSpec]] = {
    "adabelief": [
        _p_lr(0.05),
        _p_beta("beta1", 0.9, "Decay β₁ of the first moment."),
        _p_beta("beta2", 0.999, "Decay β₂ of the 'belief' s = EMA((g − m)²)."),
        ParamSpec("eps", 1e-8, min=1e-16, max=1e-2, log=True, help="ε (added to s and to √ŝ)."),
        P_GTOL,
        P_MAX_ITER,
    ],
    "lion": [
        _p_lr(0.01),
        _p_beta("beta1", 0.9, "Interpolation β₁: the step is sign(β₁m + (1 − β₁)g)."),
        _p_beta("beta2", 0.99, "Decay β₂ of the momentum m."),
        ParamSpec("weight_decay", 0.0, min=0.0, max=1.0, help="Decoupled weight decay λ."),
        ParamSpec(
            "lr_decay",
            1.0,
            min=0.5,
            max=1.0,
            help="Schedule η_k = lr·lr_decay^(k−1); 1 gives a constant learning rate. With "
            "lr_decay < 1 the total travel per coordinate is at most lr/(1 − lr_decay).",
        ),
        P_GTOL,
        P_MAX_ITER,
    ],
    "adan": [
        _p_lr(0.05),
        ParamSpec("beta1", 0.02, min=0.0, max=1.0, help="Weight β₁ of g in m (paper notation)."),
        ParamSpec("beta2", 0.08, min=0.0, max=1.0, help="Weight β₂ of g_k − g_{k−1} in v."),
        ParamSpec("beta3", 0.01, min=0.0, max=1.0, help="Weight β₃ of the new term in n."),
        ParamSpec("eps", 1e-8, min=1e-16, max=1e-2, log=True, help="ε in η/(√n + ε)."),
        P_GTOL,
        P_MAX_ITER,
    ],
    "sophia": [
        _p_lr(0.002),
        _p_beta("beta1", 0.96, "Decay β₁ of the gradient EMA m."),
        _p_beta("beta2", 0.99, "Decay β₂ of the Hessian EMA h."),
        ParamSpec("gamma", 0.01, min=1e-4, max=10.0, log=True, help="Scale γ in max(γh, ε)."),
        ParamSpec("eps", 1e-12, min=1e-16, max=1e-4, log=True, help="Floor ε in max(γh, ε)."),
        ParamSpec("k", 10, kind="int", min=1, max=1000, help="Refresh h every k iterations."),
        ParamSpec(
            "estimator",
            "exact",
            kind="choice",
            choices=ESTIMATORS,
            help="Diagonal-Hessian estimate: exact diag(∇²f) or Hutchinson u ⊙ (∇²f u).",
        ),
        P_GTOL,
        P_MAX_ITER,
    ],
}


# --------------------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------------------


def _vec(a: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in a]


def _norm(v: Vector) -> float:
    with np.errstate(over="ignore", under="ignore"):
        return float(np.linalg.norm(v))


def _check_common(gtol: float, max_iter: object, lr: float) -> int:
    if not (math.isfinite(gtol) and gtol >= GTOL_MIN):
        raise ValueError(f"gtol must be a finite number ≥ {GTOL_MIN:g}, got {gtol}")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, float, np.integer, np.floating))
        or not math.isfinite(max_iter)
        or max_iter != math.floor(max_iter)
        or max_iter < 1
    ):
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be a finite number > 0, got {lr}")
    return int(max_iter)


def _check_range(lo: float, hi: float, closed_hi: bool, **values: float) -> None:
    for name, v in values.items():
        ok = lo <= v <= hi if closed_hi else lo <= v < hi
        if not (math.isfinite(v) and ok):
            bracket = "]" if closed_hi else ")"
            raise ValueError(f"{name} must lie in [{lo}, {hi}{bracket}, got {v}")


class _Run:
    """Counted oracles, the trace, and the start/stop logic shared by the four methods."""

    def __init__(self, name: str, problem: Problem | Fn, x0: ArrayLike | None, gtol: float) -> None:
        prob = vector_problem(problem, x0=x0)
        self.name = name
        self.gtol = gtol
        self.x = start_point(prob, x0)
        self.f = Counted(prob.f)
        f_counted = self.f
        if prob.grad is not None:
            self.grad = Counted(prob.grad)
        else:
            self.grad = Counted(lambda z: diff.gradient(f_counted, z))
        grad_counted = self.grad
        if prob.hess is not None:
            self.hess = Counted(prob.hess)
        else:
            self.hess = Counted(lambda z: diff.hessian(grad_counted, z))
        self.trace: list[Step] = []
        self.fx = math.nan
        self.g = np.full_like(self.x, math.nan)
        self.f_start = math.nan

    def value(self, x: Vector) -> float:
        with np.errstate(all="ignore"):
            return float(self.f(x))

    def gradient(self, x: Vector) -> Vector:
        with np.errstate(all="ignore"):
            return as_vector(self.grad(x))

    def hessian(self, x: Vector) -> NDArray[np.float64]:
        with np.errstate(all="ignore"):
            return np.asarray(self.hess(x), dtype=np.float64)

    def _record(
        self, k: int, alpha: float | None, direction: Vector | None, info: dict[str, Any]
    ) -> None:
        base: dict[str, Any] = {
            "grad": _vec(self.g),
            "direction": None if direction is None else _vec(direction),
            "alpha": alpha,
        }
        self.trace.append(
            Step(
                k,
                self.x.copy(),
                self.fx,
                grad_norm=_norm(self.g),
                step_size=alpha,
                info=base | info,
            )
        )

    def result(self, converged: bool, message: str) -> Result:
        return Result(
            self.name,
            self.x.copy(),
            self.fx,
            converged,
            message,
            self.trace[-1].k,
            self.f.n,
            self.grad.n,
            self.hess.n,
            trace=self.trace,
        )

    def begin(self, **info: Any) -> Result | None:
        self.fx = self.value(self.x)
        self.g = self.gradient(self.x)
        self.f_start = self.fx
        self._record(0, None, None, info)
        if not (math.isfinite(self.fx) and bool(np.all(np.isfinite(self.g)))):
            return self.result(False, "f(x0) or ∇f(x0) is not finite")
        gn = _norm(self.g)
        if gn <= self.gtol:
            return self.result(True, f"‖∇f(x)‖ = {gn:.3g} ≤ gtol at the start point")
        return None

    def step_to(self, k: int, x_new: Vector, alpha: float, p: Vector, **info: Any) -> Result | None:
        """Evaluate f and ∇f at x_new, record step k, apply the stopping/divergence tests."""
        f_new = self.value(x_new)
        g_new = self.gradient(x_new)
        self.x, self.fx, self.g = x_new, f_new, g_new
        self._record(k, alpha, p, info)
        if not (math.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
            return self.result(False, f"diverged: f(x) or ∇f(x) is not finite at iteration {k}")
        rise_max = F_DIVERGE * max(1.0, abs(self.f_start))
        if f_new - self.f_start > rise_max:
            return self.result(
                False,
                f"diverged: f(x) − f(x0) = {f_new - self.f_start:.3g} > "
                f"{F_DIVERGE:.0e}·max(1, |f(x0)|) = {rise_max:.3g}",
            )
        gn = _norm(g_new)
        if gn <= self.gtol:
            return self.result(True, f"‖∇f(x)‖ = {gn:.3g} ≤ gtol")
        return None

    def max_iter(self, max_iter: int) -> Result:
        return self.result(
            False, f"reached max_iter={max_iter} (‖∇f(x)‖ = {_norm(self.g):.3g} > gtol)"
        )


# --------------------------------------------------------------------------------------
# AdaBelief
# --------------------------------------------------------------------------------------


def adabelief(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """AdaBelief (Zhuang et al., NeurIPS 2020, Algorithm 2), with g_t = ∇f(x_{t−1}):

        m_t = β₁m_{t−1} + (1 − β₁)g_t
        s_t = β₂s_{t−1} + (1 − β₂)(g_t − m_t)² + ε
        m̂_t = m_t/(1 − β₁ᵗ),   ŝ_t = s_t/(1 − β₂ᵗ)
        x_t = x_{t−1} − lr · m̂_t/(√ŝ_t + ε),          m₀ = s₀ = 0.

    The only change from Adam is the second moment: Adam's v_t = EMA(g²) becomes the EMA of
    the squared deviation of g_t from its prediction m_t (the "belief"). The projection Π_F of
    Alg. 2 is the identity here (F = ℝⁿ). ``direction`` = −m̂_t/(√ŝ_t + ε), ``alpha`` = lr.

    NOTE: ε enters twice, as in Alg. 2: inside the s recursion and in the denominator, so
    ŝ_t ≥ ε(1 − β₂ᵗ)/((1 − β₂)(1 − β₂ᵗ)) = ε/(1 − β₂). With β₁ = 0 (m = g) the bound is attained
    at every step and AdaBelief is fixed-step gradient descent with step lr/(√(ε/(1 − β₂)) + ε).
    On a deterministic run, once (g_t − m_t)² has stayed far below ε for many multiples of
    1/(1 − β₂) iterations, ŝ_t approaches the same floor and AdaBelief tends to heavy-ball
    momentum with that fixed step, a rotation-equivariant iteration.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``.
    """
    max_iter = _check_common(gtol, max_iter, lr)
    _check_range(0.0, 1.0, False, beta1=beta1, beta2=beta2)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be a finite number > 0, got {eps}")
    run = _Run("adabelief", problem, x0, gtol)
    m = np.zeros_like(run.x)
    s = np.zeros_like(run.x)
    zeros = _vec(m)
    done = run.begin(m=zeros, s=zeros, m_hat=zeros, s_hat=zeros, lr_eff=None)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        with np.errstate(all="ignore"):
            m = beta1 * m + (1.0 - beta1) * g
            s = beta2 * s + (1.0 - beta2) * (g - m) ** 2 + eps
            m_hat = m / (1.0 - beta1**t)
            s_hat = s / (1.0 - beta2**t)
            denom = np.sqrt(s_hat) + eps
            p = -m_hat / denom
        done = run.step_to(
            t,
            run.x + lr * p,
            lr,
            p,
            m=_vec(m),
            s=_vec(s),
            m_hat=_vec(m_hat),
            s_hat=_vec(s_hat),
            lr_eff=_vec(lr / denom),
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Lion
# --------------------------------------------------------------------------------------


def lion(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    lr_decay: float = 1.0,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Lion (Chen et al., NeurIPS 2023, Algorithm 2), with g_t = ∇f(x_{t−1}):

        c_t = β₁m_{t−1} + (1 − β₁)g_t
        x_t = x_{t−1} − η_t (sign(c_t) + λ x_{t−1})
        m_t = β₂m_{t−1} + (1 − β₂)g_t,                 m₀ = 0,

    with η_t = lr · lr_decay^(t−1). sign(0) = 0. ``direction`` = −(sign(c_t) + λx_{t−1}),
    ``alpha`` = η_t. Every coordinate moves by exactly ±η_t (λ = 0), so Lion is a
    sign method: steepest descent in the ℓ∞ norm with momentum.

    NOTE: Alg. 2 writes the learning rate as a schedule η_t (the paper's experiments use
    cosine decay). We use the exponential schedule η_t = lr·lr_decay^(t−1). With a constant
    η (lr_decay = 1, the default) the iterates stay on the lattice x₀ + η ℤⁿ and cannot settle
    closer than about η to x*, so reaching a fixed small tolerance needs lr_decay < 1.
    The price: after iteration K a coordinate can travel at most lr·ρ^K/(1 − ρ) in total
    (ρ = lr_decay), so a run that is still farther than that from x* is frozen by the
    schedule and never reaches it, however many iterations follow.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``.
    """
    max_iter = _check_common(gtol, max_iter, lr)
    _check_range(0.0, 1.0, False, beta1=beta1, beta2=beta2)
    if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
        raise ValueError(f"weight_decay must be a finite number ≥ 0, got {weight_decay}")
    if not (math.isfinite(lr_decay) and 0.0 < lr_decay <= 1.0):
        raise ValueError(f"lr_decay must lie in (0, 1], got {lr_decay}")
    run = _Run("lion", problem, x0, gtol)
    m = np.zeros_like(run.x)
    done = run.begin(m=_vec(m), c=None, decay=None)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        eta = lr * lr_decay ** (t - 1)
        c = beta1 * m + (1.0 - beta1) * g
        decay = -weight_decay * run.x
        p = -np.sign(c) + decay
        m = beta2 * m + (1.0 - beta2) * g
        done = run.step_to(t, run.x + eta * p, eta, p, m=_vec(m), c=_vec(c), decay=_vec(decay))
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Adan
# --------------------------------------------------------------------------------------


def adan(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.05,
    beta1: float = 0.02,
    beta2: float = 0.08,
    beta3: float = 0.01,
    eps: float = 1e-8,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Adan (Xie, Zhou, Li, Lin & Yan, IEEE TPAMI 2024, Algorithm 1), in the paper's notation
    (β weights the *new* term). Paper index j = t − 1, with g_j = ∇f(x_j):

        m_j = (1 − β₁)m_{j−1} + β₁g_j
        v_j = (1 − β₂)v_{j−1} + β₂(g_j − g_{j−1})
        n_j = (1 − β₃)n_{j−1} + β₃[g_j + (1 − β₂)(g_j − g_{j−1})]²
        x_{j+1} = x_j − η/(√n_j + ε) ∘ (m_j + (1 − β₂)v_j),

    initialised as in the paper: m₀ = g₀, v₀ = 0, v₁ = g₁ − g₀, n₀ = g₀². The term
    m + (1 − β₂)v is the Nesterov-type momentum of the paper's reformulation (AGD-II, Sec. 3.1):
    it extrapolates the gradient with the EMA of gradient differences.
    ``direction`` = −(m_j + (1 − β₂)v_j)/(√n_j + ε), ``alpha`` = lr.

    NOTE: the restart condition (Alg. 1, lines 8–11) is not used, as in all of the paper's
    experiments but one, and the weight decay λ_k is 0 (no (1 + λη)⁻¹ factor). As in Alg. 1,
    there is no bias correction (the initialisation m₀ = g₀, n₀ = g₀² removes the zero bias).

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``.
    """
    max_iter = _check_common(gtol, max_iter, lr)
    _check_range(0.0, 1.0, True, beta1=beta1, beta2=beta2, beta3=beta3)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be a finite number > 0, got {eps}")
    run = _Run("adan", problem, x0, gtol)
    zeros = _vec(np.zeros_like(run.x))
    done = run.begin(m=zeros, v=zeros, n=zeros, lr_eff=None)
    if done is not None:
        return done
    g_prev = run.g
    with np.errstate(all="ignore"):
        m, v, n = g_prev.copy(), np.zeros_like(g_prev), g_prev * g_prev
    # Step 0 shows the paper's initialisation, which needs g₀ (known only after begin()).
    s0 = run.trace[0]
    run.trace[0] = dataclasses.replace(s0, info={**s0.info, "m": _vec(m), "n": _vec(n)})
    for t in range(1, max_iter + 1):
        g = run.g
        with np.errstate(all="ignore"):
            if t >= 2:
                diff_g = g - g_prev
                m = (1.0 - beta1) * m + beta1 * g
                v = diff_g if t == 2 else (1.0 - beta2) * v + beta2 * diff_g
                n = (1.0 - beta3) * n + beta3 * (g + (1.0 - beta2) * diff_g) ** 2
            denom = np.sqrt(n) + eps
            p = -(m + (1.0 - beta2) * v) / denom
        g_prev = g
        done = run.step_to(
            t, run.x + lr * p, lr, p, m=_vec(m), v=_vec(v), n=_vec(n), lr_eff=_vec(lr / denom)
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Sophia
# --------------------------------------------------------------------------------------


def sophia(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    seed: int | None = None,
    lr: float = 0.002,
    beta1: float = 0.96,
    beta2: float = 0.99,
    gamma: float = 0.01,
    eps: float = 1e-12,
    k: int = 10,
    estimator: str = "exact",
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Sophia (Liu, Li, Hall, Liang & Ma, ICLR 2024, Algorithm 3 and eq. 6), g_t = ∇f(x_{t−1}):

        m_t = β₁m_{t−1} + (1 − β₁)g_t
        if (t − 1) mod k = 0:  ĥ_t = Estimator(x_{t−1}),  h_t = β₂h_{t−k} + (1 − β₂)ĥ_t
        else:                  h_t = h_{t−1}
        x_t = x_{t−1} − lr · clip(m_t / max(γ h_t, ε), 1),       m₀ = 0, h = 0,

    with clip(z, 1) = max(min(z, 1), −1) coordinate-wise. Estimators:
    ``"exact"``: ĥ = diag(∇²f(x)); ``"hutchinson"`` (Alg. 1): ĥ = u ⊙ (∇²f(x) u) with
    u ~ N(0, I), drawn as u_i = Rng(seed).normal() for i = 0, …, n−1 in order at each refresh
    (E ĥ = diag ∇²f, paper eq. 7). Where h ≤ 0 or the ratio exceeds 1 the coordinate takes
    the sign-momentum step lr·sign(m). ``direction`` = −clip(·), ``alpha`` = lr.

    NOTE: Alg. 3 refreshes h when "t mod k = 1"; we use (t − 1) mod k = 0, which is the same
    for k ≥ 2 and refreshes every step for k = 1 (the paper's test is never true for k = 1).
    The Hessian–vector product of the Hutchinson estimator is formed from the full Hessian
    (one ∇²f call), since numopt problems supply ∇²f, not a Hessian-vector product. The
    weight decay of Alg. 3 (line 12) is 0 here.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``.
    """
    max_iter = _check_common(gtol, max_iter, lr)
    _check_range(0.0, 1.0, False, beta1=beta1, beta2=beta2)
    if not (math.isfinite(gamma) and gamma > 0.0):
        raise ValueError(f"gamma must be a finite number > 0, got {gamma}")
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"eps must be a finite number > 0, got {eps}")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    rng = Rng(0 if seed is None else seed)
    run = _Run("sophia", problem, x0, gtol)
    m = np.zeros_like(run.x)
    h = np.zeros_like(run.x)
    done = run.begin(m=_vec(m), h=_vec(h), h_new=None, ratio=None, clipped=None)
    if done is not None:
        return done
    n = run.x.size
    for t in range(1, max_iter + 1):
        g = run.g
        h_new: list[float] | None = None
        with np.errstate(all="ignore"):
            m = beta1 * m + (1.0 - beta1) * g
            if (t - 1) % k == 0:
                H = run.hessian(run.x)
                if estimator == "exact":
                    h_hat = np.diagonal(H).copy()
                else:
                    u = np.array([rng.normal() for _ in range(n)])
                    h_hat = u * (H @ u)
                h = beta2 * h + (1.0 - beta2) * h_hat
                h_new = _vec(h_hat)
            ratio = m / np.maximum(gamma * h, eps)
            p = -np.clip(ratio, -1.0, 1.0)
        done = run.step_to(
            t,
            run.x + lr * p,
            lr,
            p,
            m=_vec(m),
            h=_vec(h),
            h_new=h_new,
            ratio=_vec(ratio),
            clipped=[bool(c) for c in np.abs(ratio) > 1.0],
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Rotation wrapper
# --------------------------------------------------------------------------------------


def rotation_2d(theta: float) -> NDArray[np.float64]:
    """R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]] (counter-clockwise rotation by θ)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotate(problem: Problem, R: ArrayLike, *, tag: str = "rot") -> Problem:
    """Return the problem g(x) = f(R x) for an orthogonal matrix R (RᵀR = I).

        g(x) = f(Rx),   ∇g(x) = Rᵀ∇f(Rx),   ∇²g(x) = Rᵀ ∇²f(Rx) R,

    so ∇²g(x) is similar to ∇²f(Rx): the Hessian spectrum is unchanged. Minimizers map to
    Rᵀx*, the default start to Rᵀx₀ (so a rotation-equivariant method follows the rotated
    path exactly), and the minimum values are unchanged. ``f`` keeps accepting grids of shape
    ``(n, *grid)``. ``domain`` is the bounding box of the rotated original box.
    ``extra["A"], extra["c"]`` (quadratics, f = ½(x − c)ᵀA(x − c)) become RᵀAR and Rᵀc.
    """
    Rm = np.array(R, dtype=np.float64)
    n = problem.dim
    if Rm.shape != (n, n):
        raise ValueError(f"R must have shape ({n}, {n}), got {Rm.shape}")
    if not np.allclose(Rm.T @ Rm, np.eye(n), rtol=0.0, atol=1e-12):
        raise ValueError("R must be orthogonal (RᵀR = I)")
    f0, g0, h0 = problem.f, problem.grad, problem.hess

    def f(x: ArrayLike) -> Any:
        X = np.asarray(x, dtype=np.float64)
        return f0(np.einsum("ij,j...->i...", Rm, X))  # (n, *grid) → (n, *grid)

    grad = None if g0 is None else (lambda x: Rm.T @ as_vector(g0(Rm @ as_vector(x))))
    hess = None if h0 is None else (lambda x: Rm.T @ np.asarray(h0(Rm @ as_vector(x))) @ Rm)

    def back(p: Any) -> list[float]:
        return [float(v) for v in Rm.T @ as_vector(p)]

    domain: tuple[Any, ...] = problem.domain
    if problem.domain and len(problem.domain) == n:
        lo = np.array([d[0] for d in problem.domain], dtype=np.float64)
        hi = np.array([d[1] for d in problem.domain], dtype=np.float64)
        corners = np.array(np.meshgrid(*zip(lo, hi, strict=True))).reshape(n, -1)  # (n, 2ⁿ)
        rc = Rm.T @ corners
        domain = tuple((float(a), float(b)) for a, b in zip(rc.min(1), rc.max(1), strict=True))
    extra = dict(problem.extra)
    if "A" in extra and "c" in extra:
        A = np.asarray(extra["A"], dtype=np.float64)
        extra["A"] = (Rm.T @ A @ Rm).tolist()
        extra["c"] = back(extra["c"])
    extra["rotation"] = Rm.tolist()
    extra["base"] = problem.id
    return Problem(
        id=f"{problem.id}_{tag}",
        name=f"{problem.name} (rotated)",
        latex=rf"g(x) = f(Rx),\ {problem.latex}",
        f=f,
        dim=n,
        domain=domain,
        grad=grad,
        hess=hess,
        x0=None if problem.x0 is None else back(problem.x0),
        minima=tuple(back(mn) for mn in problem.minima),
        exact=problem.exact,
        description=f"{problem.description} Rotated: g(x) = f(Rx).",
        tags=(*problem.tags, "rotated"),
        extra=extra,
    )


def hessian_11_norm(H: ArrayLike) -> float:
    """‖H‖₁,₁ = Σᵢⱼ |Hᵢⱼ|, the ℓ∞-smoothness surrogate of Xie, Mohamadi & Li (ICLR 2025)."""
    return float(np.abs(np.asarray(H, dtype=np.float64)).sum())


METHODS: dict[str, Callable[..., Result]] = {
    "adabelief": adabelief,
    "lion": lion,
    "adan": adan,
    "sophia": sophia,
}
