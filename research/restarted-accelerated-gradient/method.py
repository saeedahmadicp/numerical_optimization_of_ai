"""Accelerated proximal gradient with adaptive restart (FISTA / AGD), ISTA and AdProxGD.

All three methods minimize a composite objective

    F(x) = f(x) + g(x),    f convex with a Lipschitz gradient (constant L(f)),
                           g convex, lower semicontinuous, with a cheap proximal map

    prox_{s g}(v) = argmin_u { g(u) + ‖u − v‖² / (2s) }.

With g ≡ 0 the proximal map is the identity: ISTA is gradient descent with step 1/L, FISTA
is Nesterov's accelerated gradient (AGD) with the t_k schedule, and AdProxGD is adaptive
gradient descent.

Problems. A :class:`CompositeProblem` is a numopt :class:`~numopt.core.types.Problem` with
two more fields, ``g`` and ``prox``; ``Problem.f`` and ``Problem.grad`` are the smooth part f
and ∇f. A plain ``Problem`` (or a bare callable f with ``x0``) is treated as g ≡ 0.
:func:`make_lasso` builds the lasso F(x) = ½‖Ax − b‖² + λ‖x‖₁ (prox = soft-thresholding).

Methods (signature ``fn(problem, *, x0=None, **params) -> Result``):

* :func:`fista` — FISTA (Beck & Teboulle 2009, §4, eqs. 4.1–4.3; constant step or
  backtracking) with the adaptive restart tests of O'Donoghue & Candès (2015, §3.2 and
  eq. 12): ``restart="function"`` restarts when F(x_k) > F(x_{k−1}); ``restart="gradient"``
  restarts when (y_k − x_k)ᵀ(x_k − x_{k−1}) > 0. ``restart="none"`` is plain FISTA.
* :func:`ista` — the proximal gradient method (Beck & Teboulle 2009, eq. 3.1 / 3.3).
* :func:`adaptive_proxgd` — Malitsky & Mishchenko (2024), Algorithm 3: the step size comes
  from observed gradient differences, with no knowledge of L(f).

Trace. Step k = 0 holds x₀. Step k ≥ 1 holds the iterate x_k produced by iteration k, with
``Step.fun = F(x_k)``, ``Step.step_size`` the proximal step s used to produce x_k (1/L_k for
FISTA/ISTA, α_{k−1} for AdProxGD) and ``Step.grad_norm = None`` (the methods never evaluate
∇f at x_k; the stationarity measure they do have is ``info["grad_map_norm"]``).
``n_iter == trace[-1].k``.

Stopping test (converged). The norm of the gradient mapping at the point z where the
gradient was taken, ‖G_s(z)‖ = ‖z − prox_{sg}(z − s∇f(z))‖ / s ≤ ``gtol``. For g ≡ 0 this is
‖∇f(z)‖. z = y_k for FISTA, z = x_{k−1} for ISTA and AdProxGD. The test is applied after
every iteration (x₀ is never tested on its own: its gradient is first evaluated in
iteration 1, which then reports x₁).

Failure (converged=False). ``max_iter`` reached; F(x_k) or ∇f not finite; a rise
F(x_k) − F(x₀) > 10¹²·max(1, |F(x₀)|) (divergence, e.g. a fixed step larger than 2/L(f));
the backtracking search finds no acceptable L within ``MAX_BACKTRACK`` trials.

Counts. ``n_fev`` counts evaluations of f (the smooth part), ``n_gev`` of ∇f.
``Result.extra`` adds ``n_gval`` (evaluations of g), ``n_prox`` (proximal maps),
``n_restart`` (FISTA restarts) and ``restarts`` (the iterations k at which a restart fired).

Info keys:
    fista:
        y: [n] | null        y_k, the point where ∇f was evaluated (x_k = p_{L_k}(y_k)); null
                             at k = 0.
        beta: float | null   the momentum coefficient that formed y_k
                             (y_k = x_{k−1} + beta·(x_{k−1} − x_{k−2})); 0 after a restart.
        t: float             t_{k+1}, the schedule value carried into the next iteration
                             (1 after a restart, 1 at k = 0).
        restarted: bool      True when the restart test fired at x_k (then y_{k+1} = x_k and
                             t_{k+1} = 1).
        L: float             L_k, the Lipschitz estimate used to produce x_k (1/lr at k = 0).
        grad_map_norm: float | null   ‖G_{1/L_k}(y_k)‖ = L_k‖y_k − x_k‖ (null at k = 0).
        trials: [[L, F]]     backtracking trials: each L̄ tried and F(p_{L̄}(y_k)) there, in
                             order ([] with a constant step and at k = 0).
    ista:
        L, grad_map_norm, trials   as for fista, with y_k = x_{k−1}.
    adaptive_proxgd:
        alpha: float | null  α_{k−1}, the step that produced x_k (null at k = 0).
        theta: float | null  θ_{k−1} = α_{k−1}/α_{k−2} (null for k ≤ 1).
        L_local: float | null  L_{k−1} = ‖∇f(x_{k−1}) − ∇f(x_{k−2})‖/‖x_{k−1} − x_{k−2}‖
                             (null for k ≤ 1).
        grad_map_norm: float | null   ‖x_{k−1} − x_k‖/α_{k−1} (null at k = 0).

References:
    A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear
    inverse problems", SIAM J. Imaging Sci. 2(1) (2009) 183–202.
    B. O'Donoghue and E. Candès, "Adaptive restart for accelerated gradient schemes",
    Found. Comput. Math. 15 (2015) 715–732.
    Y. Malitsky and K. Mishchenko, "Adaptive proximal gradient method for convex
    optimization", NeurIPS 2024 (arXiv:2308.02261v2).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt.core import diff
from numopt.core.counting import Counted, start_point, vector_problem
from numopt.core.registry import ParamSpec
from numopt.core.types import Problem, Result, Step, Vector, as_vector

Fn = Callable[[Any], Any]
ProxFn = Callable[[Vector, float], Vector]

#: A rise F(x_k) − F(x₀) above F_DIVERGE·max(1, |F(x₀)|) counts as divergence (the same rule
#: as numopt.unconstrained.first_order).
F_DIVERGE = 1e12

#: Backtracking: at most this many trials L̄ = ηⁱ L_{k−1} per iteration (η = 2 gives a factor
#: 2^60 ≈ 1e18 above the previous estimate).
MAX_BACKTRACK = 60

#: Backtracking: relative slack for rounding in the test F(p) ≤ Q_L(p, y).
# NOTE: Beck & Teboulle's test (2.9) is exact. In floating point, f(p) and f(y) carry rounding
# errors of order ε·(sum of |terms| of f), which can exceed the decrease being tested once
# ‖p − y‖ is tiny; the exact test then rejects every L̄ and drives L_k → ∞ (a stall). The test
# accepts a violation up to BT_RTOL·max(|f(y)|, |f(p)|), which is below what f can resolve.
BT_RTOL = 1e-12


# ======================================================================================
# Composite problems
# ======================================================================================


def _zero(x: Vector) -> float:
    return 0.0


def _identity(v: Vector, s: float) -> Vector:
    return v.copy()


@dataclass(frozen=True)
class CompositeProblem(Problem):
    """A :class:`Problem` for F = f + g: ``f``/``grad`` are the smooth part, plus g and its prox.

    ``prox(v, s)`` returns prox_{s g}(v) = argmin_u g(u) + ‖u − v‖²/(2s).
    ``extra`` may carry ``"L"`` (a Lipschitz constant of ∇f), ``"F_min"`` (the optimal value
    of F) and ``"x_star"`` (the minimizer).
    """

    g: Callable[[Vector], float] = _zero
    prox: ProxFn = _identity


def soft_threshold(v: Vector, tau: float) -> Vector:
    """S_τ(v) = sign(v)·max(|v| − τ, 0) elementwise: prox_{τ‖·‖₁}(v) (Beck–Teboulle eq. 1.5)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def make_lasso(
    A: ArrayLike,
    b: ArrayLike,
    lam: float,
    *,
    id: str = "lasso",
    name: str = "Lasso",
    x0: ArrayLike | None = None,
    domain: tuple[Any, ...] = (),
    extra: dict[str, Any] | None = None,
) -> CompositeProblem:
    """F(x) = ½‖Ax − b‖² + λ‖x‖₁ with f = ½‖Ax − b‖², g = λ‖x‖₁, prox = S_{λs}.

    ``extra["L"]`` = λ_max(AᵀA) = σ_max(A)², the Lipschitz constant of ∇f.
    """
    A_ = np.array(A, dtype=np.float64, copy=True)  # (m, n)
    b_ = np.array(b, dtype=np.float64, copy=True).reshape(-1)  # (m,)
    if A_.ndim != 2 or A_.shape[0] != b_.size:
        raise ValueError(f"A must be (m, n) with m = len(b); got {A_.shape} and {b_.shape}")
    if not lam > 0.0:
        raise ValueError(f"lam must be > 0, got {lam}")
    n = A_.shape[1]
    L = float(np.linalg.norm(A_, 2) ** 2)

    def f(x: Vector) -> float:
        r = A_ @ x - b_
        return 0.5 * float(r @ r)

    def grad(x: Vector) -> Vector:
        return A_.T @ (A_ @ x - b_)

    def hess(x: Vector) -> Vector:
        return A_.T @ A_

    def g(x: Vector) -> float:
        return lam * float(np.sum(np.abs(x)))

    def prox(v: Vector, s: float) -> Vector:
        return soft_threshold(v, lam * s)

    return CompositeProblem(
        id=id,
        name=name,
        latex=r"\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1",
        f=f,
        dim=n,
        domain=domain,
        grad=grad,
        hess=hess,
        x0=np.zeros(n) if x0 is None else as_vector(x0),
        g=g,
        prox=prox,
        extra={"L": L, "lam": float(lam), "A": A_, "b": b_, **(extra or {})},
    )


# ======================================================================================
# Shared machinery
# ======================================================================================


def _vec(a: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in a]


def _check_common(gtol: float, max_iter: object, lr: float) -> int:
    if not (math.isfinite(gtol) and gtol > 0.0):
        raise ValueError(f"gtol must be a finite number > 0, got {gtol}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, float, np.integer)):
        raise ValueError(f"max_iter must be an integer ≥ 1, got {max_iter!r}")
    if float(max_iter) != int(max_iter) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be an integer ≥ 1, got {max_iter!r}")
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be a finite number > 0, got {lr}")
    return int(max_iter)


class _Oracles:
    """Counted f, ∇f, g and prox of a (composite) problem, and the shared trace/result logic."""

    def __init__(self, name: str, problem: Problem | Fn, x0: ArrayLike | None) -> None:
        prob = vector_problem(problem, x0=x0)
        self.name = name
        self.x0 = start_point(prob, x0)
        self.f = Counted(prob.f)
        f_counted = self.f
        if prob.grad is not None:
            self.grad = Counted(prob.grad)
        else:
            self.grad = Counted(lambda z: diff.gradient(f_counted, z))
        g_fn = prob.g if isinstance(prob, CompositeProblem) else _zero
        prox_fn = prob.prox if isinstance(prob, CompositeProblem) else _identity
        self.g = Counted(g_fn)
        self.prox = Counted(prox_fn)
        self.trace: list[Step] = []
        self.restarts: list[int] = []

    def F(self, x: Vector) -> tuple[float, float]:
        """(f(x), F(x) = f(x) + g(x))."""
        fx = float(self.f(x))
        return fx, fx + float(self.g(x))

    def result(self, x: Vector, Fx: float, converged: bool, message: str) -> Result:
        extra: dict[str, Any] = {
            "n_gval": self.g.n,
            "n_prox": self.prox.n,
        }
        if self.name == "fista":
            extra["n_restart"] = len(self.restarts)
            extra["restarts"] = list(self.restarts)
        return Result(
            method=self.name,
            x=x.copy(),
            fun=Fx,
            converged=converged,
            message=message,
            n_iter=self.trace[-1].k,
            n_fev=self.f.n,
            n_gev=self.grad.n,
            n_hev=0,
            trace=self.trace,
            extra=extra,
        )


def _finite(*values: Any) -> bool:
    return all(bool(np.all(np.isfinite(v))) for v in values)


def _diverged(F_new: float, F0: float) -> bool:
    return F_new - F0 > F_DIVERGE * max(1.0, abs(F0))


def _prox_step(
    o: _Oracles,
    y: Vector,
    gy: Vector,
    fy: float | None,
    L: float,
    backtracking: bool,
    eta: float,
) -> tuple[Vector, float, float, float, list[list[float]], str | None]:
    """p_L(y) = prox_{g/L}(y − ∇f(y)/L), with Beck–Teboulle backtracking (3.2) if requested.

    Returns (p, f(p), F(p), L, trials, error). With backtracking, L is the smallest ηⁱ L
    (i ≥ 0) with f(p) ≤ f(y) + ⟨p − y, ∇f(y)⟩ + (L/2)‖p − y‖² (F(p) ≤ Q_L(p, y), eq. 2.9 with
    g(p) cancelled on both sides).
    """
    trials: list[list[float]] = []
    if not backtracking:
        p = o.prox(y - gy / L, 1.0 / L)
        fp, Fp = o.F(p)
        return p, fp, Fp, L, trials, None
    assert fy is not None
    for _ in range(MAX_BACKTRACK):
        p = o.prox(y - gy / L, 1.0 / L)
        fp, Fp = o.F(p)
        trials.append([L, Fp])
        d = p - y
        q = fy + float(d @ gy) + 0.5 * L * float(d @ d)
        if math.isfinite(fp) and fp <= q + BT_RTOL * max(abs(fy), abs(fp)):
            return p, fp, Fp, L, trials, None
        L *= eta
    return (
        y,
        math.nan,
        math.nan,
        L,
        trials,
        (f"backtracking failed: no L̄ = ηⁱL_(k−1) with F(p) ≤ Q_L(p, y) in {MAX_BACKTRACK} trials"),
    )


# ======================================================================================
# FISTA / AGD with adaptive restart, and ISTA
# ======================================================================================

RESTARTS = ("none", "function", "gradient")

PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "fista": (
        ParamSpec(
            "restart",
            "gradient",
            kind="choice",
            choices=RESTARTS,
            help="Adaptive restart test (O'Donoghue & Candès 2015, §3.2): none = plain FISTA; "
            "function: F(x_k) > F(x_{k−1}); gradient: (y_k − x_k)ᵀ(x_k − x_{k−1}) > 0.",
        ),
        ParamSpec(
            "lr",
            1.0,
            min=1e-8,
            max=1e4,
            log=True,
            help="Step 1/L. With backtracking, the first estimate L₀ = 1/lr (it only grows, so L₀ ≤ L).",
        ),
        ParamSpec(
            "backtracking",
            True,
            kind="bool",
            help="Beck–Teboulle backtracking: multiply L by η until F(p_L(y)) ≤ Q_L(p_L(y), y).",
        ),
        ParamSpec("eta", 2.0, min=1.1, max=10.0, help="Backtracking factor η > 1."),
        ParamSpec(
            "gtol",
            1e-6,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Stop when the gradient-mapping norm L‖y_k − x_k‖ ≤ gtol.",
        ),
        ParamSpec("max_iter", 5000, kind="int", min=1, max=1_000_000),
    ),
    "adaptive_proxgd": (
        ParamSpec(
            "lr",
            1e-3,
            min=1e-10,
            max=1e4,
            log=True,
            help="Initial step α₀ (Malitsky & Mishchenko suggest α₀L₁ ∈ [1/√2, 2], eq. 16).",
        ),
        ParamSpec(
            "gtol",
            1e-6,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Stop when ‖x_{k−1} − x_k‖/α_{k−1} ≤ gtol.",
        ),
        ParamSpec("max_iter", 5000, kind="int", min=1, max=1_000_000),
    ),
}
PARAMS["ista"] = tuple(p for p in PARAMS["fista"] if p.name != "restart")


def _prox_gradient(
    name: str,
    problem: Problem | Fn,
    x0: ArrayLike | None,
    *,
    accelerated: bool,
    restart: str,
    lr: float,
    backtracking: bool,
    eta: float,
    gtol: float,
    max_iter: object,
) -> Result:
    max_it = _check_common(gtol, max_iter, lr)
    if restart not in RESTARTS:
        raise ValueError(f"restart must be one of {RESTARTS}, got {restart!r}")
    if not (math.isfinite(eta) and eta > 1.0):
        raise ValueError(f"eta must be a finite number > 1, got {eta}")
    o = _Oracles(name, problem, x0)
    x = o.x0  # x_{k−1} at the top of iteration k
    y = x.copy()  # y_k   (Step 0: y₁ = x₀)
    t = 1.0  # t_k   (Step 0: t₁ = 1)
    beta = 0.0  # the coefficient that formed y_k
    L = 1.0 / lr
    fx, Fx = o.F(x)
    F0 = Fx
    info0: dict[str, Any] = {"L": L, "grad_map_norm": None, "trials": []}
    if accelerated:
        info0 = {"y": None, "beta": None, "t": t, "restarted": False, **info0}
    o.trace.append(Step(0, x.copy(), Fx, None, None, info0))
    if not math.isfinite(Fx):
        return o.result(x, Fx, False, "F(x0) is not finite")

    for k in range(1, max_it + 1):
        gy = np.asarray(o.grad(y), dtype=np.float64)
        if not _finite(gy):
            return o.result(x, Fx, False, f"diverged: ∇f is not finite at y_k (k={k})")
        fy: float | None = None
        if backtracking:
            # f(y_k) is already known when y_k = x_{k−1} (no momentum): reuse it.
            fy = fx if np.array_equal(y, x) else float(o.f(y))
        p, fp, Fp, L, trials, err = _prox_step(o, y, gy, fy, L, backtracking, eta)
        if err is not None:
            # No new iterate: the trace ends at x_{k−1}; the failed trials go in the message.
            return o.result(x, Fx, False, f"{err} (k={k}, last L̄ = {trials[-1][0]:.3e})")
        gm = L * float(np.linalg.norm(y - p))  # ‖G_{1/L}(y_k)‖
        restarted = False
        if accelerated:
            # Adaptive restart tests (O'Donoghue & Candès 2015, §3.2; eq. 12 for the
            # generalized-gradient form), evaluated at the new iterate x_k = p.
            if restart == "function":
                restarted = Fp > Fx
            elif restart == "gradient":
                restarted = float((y - p) @ (p - x)) > 0.0
            if restarted:
                # NOTE: a restart takes x_k as a new starting point: the reset of O'Donoghue
                # & Candès' Algorithm 3 (fixed restarting), line 3: x⁰ = y⁰ = x_k, θ₀ = 1,
                # applied when a §3.2 test fires; i.e. y_{k+1} = x_k and
                # t_{k+1} = 1, so the next two steps carry no momentum (as at the very start).
                # Setting only t_k = 1 would give one momentum-free step instead.
                t_next, beta_next, y_next = 1.0, 0.0, p.copy()
                o.restarts.append(k)
            else:
                # Beck & Teboulle (4.2)–(4.3).
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                beta_next = (t - 1.0) / t_next
                y_next = p + beta_next * (p - x)
        else:
            t_next, beta_next, y_next = 1.0, 0.0, p.copy()  # ISTA: y_{k+1} = x_k (eq. 3.1)
        o.trace.append(
            Step(
                k,
                p.copy(),
                Fp,
                None,
                1.0 / L,
                _fista_info(accelerated, y, beta, t_next, restarted, L, gm, trials),
            )
        )
        if not _finite(Fp, p):
            return o.result(p, Fp, False, f"diverged: F(x_k) is not finite (k={k})")
        if _diverged(Fp, F0):
            return o.result(
                p, Fp, False, f"diverged: F rose by more than 1e12·max(1, |F(x0)|) (k={k})"
            )
        x, fx, Fx = p, fp, Fp
        y, t, beta = y_next, t_next, beta_next
        if gm <= gtol:
            return o.result(x, Fx, True, f"converged: gradient-mapping norm {gm:.3e} ≤ gtol")
    return o.result(x, Fx, False, f"max_iter = {max_it} reached")


def _fista_info(
    accelerated: bool,
    y: Vector,
    beta: float,
    t_next: float,
    restarted: bool,
    L: float,
    gm: float | None,
    trials: list[list[float]],
) -> dict[str, Any]:
    info: dict[str, Any] = {"L": L, "grad_map_norm": gm, "trials": trials}
    if accelerated:
        info = {"y": _vec(y), "beta": beta, "t": t_next, "restarted": restarted, **info}
    return info


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
    """FISTA (accelerated proximal gradient) with optional adaptive restart.

    Beck & Teboulle (2009), §4, FISTA with constant stepsize (eqs. 4.1–4.3) or with
    backtracking: y₁ = x₀, t₁ = 1 and for k ≥ 1

        x_k     = p_{L_k}(y_k) = prox_{g/L_k}(y_k − ∇f(y_k)/L_k)                       (4.1)
        t_{k+1} = (1 + √(1 + 4t_k²))/2                                                (4.2)
        y_{k+1} = x_k + ((t_k − 1)/t_{k+1})(x_k − x_{k−1})                            (4.3)

    With g ≡ 0 this is Nesterov's accelerated gradient (O'Donoghue & Candès 2015,
    Algorithm 1 with q = 0, written there with θ_k = 1/t_k).

    ``backtracking=False``: L_k ≡ 1/lr, which must be ≥ L(f) for the guarantee
    F(x_k) − F* ≤ 2L‖x₀ − x*‖²/(k + 1)² (Theorem 4.4, restart="none").
    ``backtracking=True``: L_k = ηⁱL_{k−1} with the smallest i ≥ 0 such that
    F(p_{L̄}(y_k)) ≤ Q_{L̄}(p_{L̄}(y_k), y_k) (eq. 2.5/2.9), L₀ = 1/lr; L_k never decreases and
    L_k ≤ max(L₀, ηL(f)) (Remark 3.2). Because L_k never decreases, a start L₀ > L(f) is never
    corrected (every step is then 1/L₀ < 1/L(f)); choose L₀ ≤ L(f). The test only needs the
    quadratic upper bound along the observed steps, so the accepted L_k = ηʲL₀ can lie below
    L(f), and the iteration counts depend on where ηʲL₀ falls relative to L(f).

    Adaptive restart (O'Donoghue & Candès 2015, §3.2, and eq. 12 for the proximal case),
    tested after x_k is computed:

    * ``"function"``: restart when F(x_k) > F(x_{k−1});
    * ``"gradient"``: restart when (y_k − x_k)ᵀ(x_k − x_{k−1}) > 0, i.e. when the generalized
      gradient G(y_k) = L_k(y_k − x_k) makes an acute angle with the last step;
    * ``"none"``: plain FISTA.

    A restart sets y_{k+1} = x_k and t_{k+1} = 1 (a fresh start from x_k: the reset of
    Algorithm 3, "fixed restarting", applied when a §3.2 test fires).
    The restart tests need no extra evaluations: F(x_k) is computed for the trace anyway.

    Costs per iteration: one ∇f (at y_k), one f and one g (at x_k), one prox; backtracking
    adds f(y_k) (unless y_k = x_{k−1}) and one f, g and prox per extra trial.

    Stops (converged) when L_k‖y_k − x_k‖ ≤ ``gtol`` (‖∇f(y_k)‖ when g ≡ 0; then
    f(x_k) ≤ f(y_k) − ‖∇f(y_k)‖²/(2L_k) by the descent lemma).
    """
    return _prox_gradient(
        "fista",
        problem,
        x0,
        accelerated=True,
        restart=restart,
        lr=lr,
        backtracking=backtracking,
        eta=eta,
        gtol=gtol,
        max_iter=max_iter,
    )


def ista(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 1.0,
    backtracking: bool = True,
    eta: float = 2.0,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """ISTA, the proximal gradient method: x_k = p_{L_k}(x_{k−1}) (Beck & Teboulle 2009, eq. 3.1;
    eq. 3.2–3.3 with backtracking). F(x_k) is non-increasing (Remark 3.1) and, with a constant
    step 1/L where L ≥ L(f), F(x_k) − F* ≤ L‖x₀ − x*‖²/(2k) (Theorem 3.1).

    Same parameters, costs and stopping test as :func:`fista` with y_k = x_{k−1}.
    """
    return _prox_gradient(
        "ista",
        problem,
        x0,
        accelerated=False,
        restart="none",
        lr=lr,
        backtracking=backtracking,
        eta=eta,
        gtol=gtol,
        max_iter=max_iter,
    )


# ======================================================================================
# Adaptive proximal gradient (Malitsky & Mishchenko 2024, Algorithm 3)
# ======================================================================================


def adaptive_proxgd(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 1e-3,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Adaptive proximal gradient method (Malitsky & Mishchenko 2024, Algorithm 3).

    Input x⁰, θ₀ = 1/3, α₀ = ``lr``; x¹ = prox_{α₀g}(x⁰ − α₀∇f(x⁰)); for k ≥ 1

        L_k     = ‖∇f(x^k) − ∇f(x^{k−1})‖ / ‖x^k − x^{k−1}‖
        α_k     = min{ √(2/3 + θ_{k−1}) α_{k−1},  α_{k−1} / √[2α²_{k−1}L_k² − 1]₊ }
        x^{k+1} = prox_{α_k g}(x^k − α_k∇f(x^k))
        θ_k     = α_k/α_{k−1},

    with [t]₊ = max(t, 0) and a/√0 = +∞ (the second bound is then inactive). With g ≡ 0 this
    is Algorithm 2 (adaptive gradient descent-2). No Lipschitz constant is needed: each
    iteration evaluates one gradient, and ∇f(x^k) is reused for L_{k+1}.

    # NOTE: the paper suggests choosing α₀ by a line search so that α₀L₁ ∈ [1/√2, 2] (eq. 16);
    # here α₀ = ``lr`` is taken as given (the method converges for every α₀ > 0, Theorem 3).
    # NOTE: when ∇f(x^k) = ∇f(x^{k−1}) with x^k ≠ x^{k−1}, L_k = 0 and only the first bound
    # is active, as the formula prescribes.

    Trace index: Step k holds x^k; ``info["alpha"]`` is α_{k−1}, the step that produced it.
    Stops (converged) when ‖x^{k−1} − x^k‖/α_{k−1} ≤ ``gtol`` (the gradient-mapping norm at
    x^{k−1}; ‖∇f(x^{k−1})‖ when g ≡ 0).
    """
    max_it = _check_common(gtol, max_iter, lr)
    o = _Oracles("adaptive_proxgd", problem, x0)
    x = o.x0
    _, Fx = o.F(x)
    F0 = Fx
    o.trace.append(
        Step(
            0,
            x.copy(),
            Fx,
            None,
            None,
            {"alpha": None, "theta": None, "L_local": None, "grad_map_norm": None},
        )
    )
    if not math.isfinite(Fx):
        return o.result(x, Fx, False, "F(x0) is not finite")
    x_prev: Vector | None = None
    g_prev: Vector | None = None
    alpha_prev = lr  # α_{k−1} in the paper's index once the loop runs (α₀ at k = 1)
    theta_prev = 1.0 / 3.0  # θ₀
    for k in range(1, max_it + 1):
        # Iteration k produces x^k from x^{k−1} (= x) with step α_{k−1}.
        gx = np.asarray(o.grad(x), dtype=np.float64)
        if not _finite(gx):
            return o.result(x, Fx, False, f"diverged: ∇f is not finite at x_(k−1) (k={k})")
        theta: float | None = None
        L_loc: float | None = None
        if x_prev is None or g_prev is None:
            alpha = lr  # x¹ = prox_{α₀g}(x⁰ − α₀∇f(x⁰))
        else:
            dx = float(np.linalg.norm(x - x_prev))
            dg = float(np.linalg.norm(gx - g_prev))
            # dx > 0 here: dx = 0 means the previous gradient-mapping norm was 0 ≤ gtol.
            L_loc = dg / dx
            bound1 = math.sqrt(2.0 / 3.0 + theta_prev) * alpha_prev
            s = 2.0 * alpha_prev**2 * L_loc**2 - 1.0
            bound2 = alpha_prev / math.sqrt(s) if s > 0.0 else math.inf
            alpha = min(bound1, bound2)
            theta = alpha / alpha_prev
        x_new = o.prox(x - alpha * gx, alpha)
        _, F_new = o.F(x_new)
        gm = float(np.linalg.norm(x - x_new)) / alpha
        o.trace.append(
            Step(
                k,
                x_new.copy(),
                F_new,
                None,
                alpha,
                {"alpha": alpha, "theta": theta, "L_local": L_loc, "grad_map_norm": gm},
            )
        )
        if not _finite(F_new, x_new):
            return o.result(x_new, F_new, False, f"diverged: F(x_k) is not finite (k={k})")
        if _diverged(F_new, F0):
            return o.result(
                x_new, F_new, False, f"diverged: F rose by more than 1e12·max(1, |F(x0)|) (k={k})"
            )
        if theta is not None:
            theta_prev = theta
        alpha_prev = alpha
        x_prev, g_prev = x, gx
        x, Fx = x_new, F_new
        if gm <= gtol:
            return o.result(x, Fx, True, f"converged: gradient-mapping norm {gm:.3e} ≤ gtol")
    return o.result(x, Fx, False, f"max_iter = {max_it} reached")


METHODS: dict[str, Callable[..., Result]] = {
    "fista": fista,
    "ista": ista,
    "adaptive_proxgd": adaptive_proxgd,
}
