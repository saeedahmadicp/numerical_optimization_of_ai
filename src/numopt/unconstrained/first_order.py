"""First-order methods for unconstrained minimization of a smooth f: ℝⁿ → ℝ.

Every method here uses only f and ∇f (coordinate descent also uses the diagonal of ∇²f).
They differ in how the step x_k − x_{k−1} is built from the gradients seen so far:

* **Line-search methods**: gradient descent (fixed step, Armijo backtracking, strong Wolfe or
  exact quadratic step) and Barzilai–Borwein (two-point step length).
* **Momentum methods**: Polyak's heavy ball and Nesterov's accelerated gradient.
* **Adaptive methods** (from machine learning, here run with exact full gradients): AdaGrad,
  RMSprop, AdaDelta, Adam, AdaMax, NAdam, AMSGrad and AdamW.
* **Cyclic coordinate descent** with a 1-D Newton step on one coordinate per iteration.

Conventions shared by every method:

* **Start.** ``x0`` (or the problem's default). A bare callable ``f`` needs ``x0``; without
  a ``grad`` the gradient is formed by central differences (``numopt.core.diff.gradient``).
* **Trace.** Step ``k = 0`` holds x₀; step ``k ≥ 1`` holds the iterate x_k produced by
  iteration k. ``Step.fun = f(x_k)``, ``Step.grad_norm = ‖∇f(x_k)‖₂`` and ``Step.step_size``
  is the scalar step length α_k of the update that produced x_k (``None`` at k = 0). Every
  update is written as

      x_k = x_{k−1} + α_k p_k,

  with ``info["alpha"] = α_k ≥ 0`` and ``info["direction"] = p_k``: the search direction for
  line-search methods, the velocity divided by the learning rate for momentum methods, and the
  preconditioned direction for adaptive methods (α_k is then the learning rate).
  ``n_iter == trace[-1].k``.
* **Stopping test (converged).** ‖∇f(x_k)‖₂ ≤ ``gtol``, checked at x₀ and after every
  iteration.
* **Failure (converged=False).** ``max_iter`` reached; *divergence*: f(x_k) or ∇f(x_k) not
  finite, or f(x_k) − f(x₀) > 10¹²·max(1, |f(x₀)|) (a rise relative to the scale of f, so
  the verdict does not depend on the units of f: the transient rises of convergent
  non-monotone methods, e.g. pure BB or heavy ball, stay far below it); a line search that
  finds no acceptable step (gradient descent: reported as divergence when a trial step gave
  f = −∞, i.e. f is unbounded below along −∇f);
  *stall* of a line-search method (gradient descent, Barzilai–Borwein: the accepted step does
  not change x in floating point; coordinate descent: n consecutive iterations leave x
  unchanged), because the iteration would then repeat itself until ``max_iter``;
  *overflow* of the line-search slope ∇f(x)ᵀp (‖∇f‖ ≳ 1e154; methods with a line search).
  A non-finite f or ∇f at x₀ also stops at once.
* **Counts.** ``n_fev`` counts f evaluations (including the 2n per central-difference
  gradient when ``grad`` is missing), ``n_gev`` counts gradient evaluations, and ``n_hev``
  counts Hessian evaluations (``exact_quadratic`` rule and coordinate descent only; without
  ``hess`` these are central differences of ∇f, whose 2n gradient calls are in ``n_gev``).

Adaptive methods with a constant learning rate have no general guarantee of convergence to
a stationary point (Reddi et al. 2018 give convex examples where Adam fails): their
normalized steps need not shrink as ∇f → 0, so depending on lr the iterates converge or
keep oscillating near x*. Each method's docstring states what its default parameters achieve
on the ``quadratic_bowl`` and ``rosenbrock`` problems (gtol = 1e-6); the defaults of the
adaptive methods are learning rates chosen for 2-D demos, not the papers' ML defaults.

Info keys (every method, every step):
    grad: [n]               ∇f(x_k), the gradient at the iterate of this step.
    direction: [n] | null   p_k with x_k = x_{k−1} + alpha·p_k (null at k = 0).
    alpha: float | null     the scalar step length α_k (= Step.step_size; null at k = 0).

Additional info keys per method:
    gradient_descent:
        trials: [[alpha, f]]  every step length tried by the line search and f(x_{k−1} + αp)
                              there, in order ([] for the fixed rule and at k = 0).
        alpha0: float | null  the first trial step of the line search (null for "fixed" and
                              "exact_quadratic").
        pHp: float | null     curvature p_kᵀ∇²f(x_{k−1})p_k ("exact_quadratic" only).
    barzilai_borwein:
        alpha_bb: float | null  the BB trial step length of this iteration (before any
                              nonmonotone backtracking; null at k = 0).
        bb1, bb2: float | null  sᵀs/sᵀy and sᵀy/yᵀy from the previous pair s = x_{k−1} − x_{k−2},
                              y = ∇f(x_{k−1}) − ∇f(x_{k−2}) (null when k ≤ 1 or undefined).
        reset: bool           True when alpha_bb is the reset value 1/‖∇f(x_{k−1})‖₂ (k = 1,
                              or the BB step was undefined, non-positive or outside its bounds).
        f_ref: float | null   the nonmonotone reference max of the last M values of f (null
                              when ``nonmonotone`` is false or at k = 0).
        trials: [[alpha, f]]  every step length tried and f there ([] at k = 0).
    momentum:
        velocity: [n]         v_k = x_k − x_{k−1} (zeros at k = 0).
    nesterov:
        velocity: [n]         v_k = x_k − x_{k−1} (zeros at k = 0).
        lookahead: [n] | null the point x_{k−1} + μv_{k−1} where the gradient was taken.
        grad_lookahead: [n] | null  ∇f at ``lookahead``.
    adagrad:
        v: [n]                sum of squared gradients G_k (the gradient at x_{k−1} included).
        lr_eff: [n] | null    lr/(√G_k + ε): per-coordinate step multiplying ∇f(x_{k−1}).
    rmsprop:
        v: [n]                running mean of squared gradients E[g²]_k.
        lr_eff: [n] | null    lr/(√E[g²]_k + ε), multiplying ∇f(x_{k−1}).
    adadelta:
        v: [n]                E[g²]_k;  u: [n]  E[Δx²]_k (running mean of squared updates).
        lr_eff: [n] | null    RMS[Δx]_{k−1}/RMS[g]_k, multiplying ∇f(x_{k−1}).
    adam, adamw:
        m, v: [n]             first and second moment estimates (biased).
        m_hat, v_hat: [n]     bias-corrected moments (zeros at k = 0).
        lr_eff: [n] | null    lr/(√v̂_k + ε), multiplying m̂_k.
        decay: [n] | null     adamw only: the decoupled weight-decay part −λx_{k−1} of the step.
    adamax:
        m: [n];  u: [n]       first moment and exponentially weighted infinity norm.
        lr_eff: [n] | null    lr/((1 − β₁ᵏ)u_k), multiplying m_k (0 where u_k = 0; capped at
                              the largest finite float where it overflows, u_k ≲ lr·6e-309).
    nadam:
        m, v: [n]             first and second moment estimates (biased).
        m_bar: [n]            Nesterov momentum combination m̄_k (zeros at k = 0).
        v_hat: [n]            bias-corrected second moment.
        lr_eff: [n] | null    lr/(√v̂_k + ε), multiplying m̄_k.
    amsgrad:
        m, v: [n]             first and second moment estimates (no bias correction).
        v_max: [n]            v̂_k = max(v̂_{k−1}, v_k), elementwise.
        lr_eff: [n] | null    lr/√v̂_k, multiplying m_k (0 where v̂_k = 0; capped at the
                              largest finite float where it overflows).
        (v and v_max are the squares of the RMS values the method keeps, see ``amsgrad``; they
        underflow to 0 for |g| ≲ 1e-161, and are capped at the largest finite float for
        |g| ≳ 1e154, although the step is computed without either.)
    coordinate_descent:
        coordinate: int | null  the 0-based coordinate i updated in this iteration.
        sweep: int | null     the 0-based sweep (k − 1) div n.
        curvature: float | null  ∂²f/∂x_i² at x_{k−1}.
        newton: bool | null   True when p_k is the 1-D Newton step, False for the gradient
                              fallback (curvature ≤ 0) or a skipped coordinate (∂f/∂x_i = 0,
                              the slope ∂f/∂x_i·p_i underflows to 0, or x_i + p_i rounds to
                              x_i; then direction = 0 and alpha = 0).
        trials: [[alpha, f]]  backtracking trials along p_k ([] at k = 0 or when skipped).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, Vector, as_vector
from ..line_search.methods import C1_DEFAULT, LineSearchResult, search

Fn = Callable[[Any], Any]

#: A rise f(x_k) − f(x₀) above F_DIVERGE·max(1, |f(x₀)|) counts as divergence.
F_DIVERGE = 1e12

#: Largest finite float; per-coordinate step sizes (``lr_eff``) that overflow are capped here.
FLOAT_MAX = float(np.finfo(np.float64).max)

#: Gradient descent, strong Wolfe rule: the search may expand its first trial step α₀ up to
#: α_max = ALPHA_MAX_RATIO·α₀ (20 doublings, which leaves 30 of the search's 50 trials for
#: the zoom). A ratio, not an absolute bound, so that the cap scales with f.
ALPHA_MAX_RATIO = 2.0**20

#: Barzilai–Borwein: step lengths outside [BB_MIN, BB_MAX] are reset (Raydan 1997 uses
#: ε = 1e-10 for the same bounds on the reciprocal step).
BB_MIN = 1e-10
BB_MAX = 1e10
#: Nonmonotone memory M, sufficient-decrease constant γ and the safeguard interval
#: [σ₁, σ₂] for the backtracking factor (Raydan 1997, §3: M = 10, γ = 1e-4, σ₁ = 0.1, σ₂ = 0.5).
BB_MEMORY = 10
BB_GAMMA = 1e-4
BB_SIGMA1 = 0.1
BB_SIGMA2 = 0.5
#: Maximum trial steps in one Barzilai–Borwein nonmonotone search.
BB_MAX_TRIALS = 50

P_GTOL = ParamSpec("gtol", 1e-6, min=1e-14, max=1e-2, log=True, help="Stop when ‖∇f(x)‖₂ ≤ gtol.")


def _p_max_iter(default: int) -> ParamSpec:
    return ParamSpec("max_iter", default, kind="int", min=1, max=100_000, help="Iteration limit.")


def _p_lr(default: float, help: str = "Learning rate (step length) α.") -> ParamSpec:
    return ParamSpec("lr", default, min=1e-6, max=2.0, log=True, help=help)


def _p_beta(name: str, default: float, help: str) -> ParamSpec:
    return ParamSpec(name, default, min=0.0, max=0.9999, help=help)


def _p_eps(default: float) -> ParamSpec:
    return ParamSpec(
        "eps", default, min=1e-12, max=1e-2, log=True, help="ε in the denominator (> 0)."
    )


# --------------------------------------------------------------------------------------
# Shared machinery
# --------------------------------------------------------------------------------------


def _vec(a: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in a]


def _norm(v: Vector) -> float:
    """‖v‖₂; ``inf`` (without a warning) when ‖v‖² overflows, i.e. ‖v‖ ≳ 1e154."""
    with np.errstate(over="ignore", under="ignore"):
        return float(np.linalg.norm(v))


def _check_positive(**values: float) -> None:
    for name, v in values.items():
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be a finite number > 0, got {v}")


def _check_unit(**values: float) -> None:
    for name, v in values.items():
        if not 0.0 <= v < 1.0:
            raise ValueError(f"{name} must lie in [0, 1), got {v}")


#: Smallest accepted gtol: ‖g‖ = √(gᵀg) underflows to 0 once ‖g‖ < ~1e-154, so a smaller
#: tolerance could report convergence for a nonzero gradient.
GTOL_MIN = 1e-150


def _check_common(gtol: float, max_iter: object) -> int:
    """Validate gtol and max_iter; return max_iter as an ``int``.

    An integral float (``1000.0``, ``1e3`` as the CLI parses it) is accepted and converted;
    a bool, a non-number, a non-integral, non-finite or non-positive value raises ValueError.
    """
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
    return int(max_iter)


class _Run:
    """Counted oracles, the trace and the shared start/stop logic of one method run."""

    def __init__(
        self,
        name: str,
        problem: Problem | Fn,
        x0: ArrayLike | None,
        gtol: float,
        *,
        need_hess: bool = False,
    ) -> None:
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
        self.hess: Counted | None = None
        if need_hess:
            if prob.hess is not None:
                self.hess = Counted(prob.hess)
            else:
                self.hess = Counted(lambda z: diff.hessian(grad_counted, z))
        self.trace: list[Step] = []
        self.fx = math.nan
        self.g = np.full_like(self.x, math.nan)
        self.f_start = math.nan

    # -- evaluation -------------------------------------------------------------------

    def value(self, x: Vector) -> float:
        with np.errstate(all="ignore"):
            return float(self.f(x))

    def gradient(self, x: Vector) -> Vector:
        with np.errstate(all="ignore"):
            return as_vector(self.grad(x))

    def hessian(self, x: Vector) -> NDArray[np.float64]:
        assert self.hess is not None
        with np.errstate(all="ignore"):
            return np.asarray(self.hess(x), dtype=np.float64)

    def line_search(self, kind: str, p: Vector, **opts: Any) -> LineSearchResult:
        """``numopt.line_search.methods.search`` from the current x along p (counted)."""
        with np.errstate(all="ignore"):
            return search(kind, self.f, self.grad, self.x, p, f0=self.fx, g0=self.g, **opts)

    # -- trace and results ------------------------------------------------------------

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
            0 if self.hess is None else self.hess.n,
            trace=self.trace,
        )

    def begin(self, **info: Any) -> Result | None:
        """Evaluate f and ∇f at x₀, record step 0; return a finished Result if already done."""
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

    def advance(
        self,
        k: int,
        x_new: Vector,
        f_new: float,
        g_new: Vector,
        alpha: float,
        direction: Vector,
        **info: Any,
    ) -> Result | None:
        """Move to x_new, record step k and apply the stopping/divergence tests."""
        self.x, self.fx, self.g = x_new, f_new, g_new
        self._record(k, alpha, direction, info)
        if not (math.isfinite(f_new) and bool(np.all(np.isfinite(g_new)))):
            return self.result(False, f"diverged: f(x) or ∇f(x) is not finite at iteration {k}")
        # NOTE: relative to the scale of f (floor 1, so f(x₀) = 0 still has a threshold). An
        # absolute threshold flagged convergent non-monotone runs whose f merely has large units.
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

    def step_to(self, k: int, x_new: Vector, alpha: float, p: Vector, **info: Any) -> Result | None:
        """Evaluate f and ∇f at x_new, then :meth:`advance`."""
        return self.advance(k, x_new, self.value(x_new), self.gradient(x_new), alpha, p, **info)

    def max_iter(self, max_iter: int) -> Result:
        gn = _norm(self.g)
        return self.result(False, f"reached max_iter={max_iter} (‖∇f(x)‖ = {gn:.3g} > gtol)")


def _overflow_message(k: int) -> str:
    return (
        f"overflow at iteration {k}: ∇f(x)ᵀp is not finite (‖∇f(x)‖ is too large), so no "
        "step length can be computed"
    )


def _stall_message(k: int, alpha: float) -> str:
    return (
        f"stalled at iteration {k}: the accepted step α = {alpha:.3g} does not change x "
        "(f cannot be decreased further at this floating-point precision)"
    )


# --------------------------------------------------------------------------------------
# Gradient descent
# --------------------------------------------------------------------------------------

STEP_RULES = ("fixed", "backtracking", "strong_wolfe", "exact_quadratic")


@register(
    id="gradient_descent",
    family="unconstrained",
    name="Gradient descent",
    params=(
        ParamSpec(
            "step_rule",
            "backtracking",
            kind="choice",
            choices=STEP_RULES,
            help="Fixed step lr, Armijo backtracking, strong Wolfe search, or the exact "
            "minimizer of the quadratic model along −∇f.",
        ),
        _p_lr(1e-3, "Fixed step length α (step_rule='fixed' only); stable when α < 2/L."),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="linear, rate ((κ − 1)/(κ + 1))² in f with exact steps on quadratics",
    summary="Step downhill along the negative gradient, with the step length set by a rule.",
    references=(
        "Nocedal & Wright (2006), §3.1–3.3, Alg. 3.1 (backtracking), Alg. 3.5 (strong Wolfe), "
        "eq. 3.26 (exact step), eq. 3.60 (initial trial step), Theorem 3.3",
        "Cauchy (1847), C. R. Acad. Sci. Paris 25, 536–538",
    ),
)
def gradient_descent(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    step_rule: str = "backtracking",
    lr: float = 1e-3,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Gradient (steepest) descent, x_k = x_{k−1} − α_k ∇f(x_{k−1}) (N&W §3.1).

    The direction is p_k = −∇f(x_{k−1}); the step length α_k comes from ``step_rule``:

    * ``"fixed"``: α_k = ``lr``. On an L-smooth f, f decreases when 0 < lr < 2/L.
    * ``"backtracking"``: Armijo backtracking (N&W Alg. 3.1, c₁ = 1e-4, ρ = ½).
    * ``"strong_wolfe"``: N&W Alg. 3.5/3.6 (c₁ = 1e-4, c₂ = 0.9).
    * ``"exact_quadratic"``: α_k = ∇fᵀ∇f / ∇fᵀ∇²f∇f (N&W eq. 3.26), the exact line minimizer
      when f is quadratic. Needs pᵀ∇²f p > 0, else the run stops (converged=False).

    First trial step of the two line searches. Every α below becomes α/c when f is replaced by
    c·f (c > 0), so the iterates do not depend on the units of f (exactly when c is a power
    of 2, else up to rounding); the fixed rule does not have this property:

    * k = 1: α₀ = 1/‖∇f(x₀)‖₂, a trial move of unit length.
    * k ≥ 2: α₀ = max(α_{3.60}, α_{3.61}), the larger of the two N&W guesses (§3.5)

          α_{3.60} = α_{k−1}‖∇f(x_{k−2})‖²/‖∇f(x_{k−1})‖²    (the last first-order decrease
                                                             α‖∇f‖² is assumed again),
          α_{3.61} = 2(f(x_{k−2}) − f(x_{k−1}))/‖∇f(x_{k−1})‖²  (the last decrease of f is
                                                             assumed again, quadratic model).

      A guess that is not finite and positive is skipped (α_{3.61} ≤ 0 when f did not
      decrease in floating point); if both are, α₀ = 1/‖∇f(x_{k−1})‖₂.

    Backtracking can only shrink α₀, so α₀ must be able to grow. After a step that was much
    too short, f fell like its linear model (by ≈ α_{k−1}‖∇f‖²) and ∇f hardly changed: then
    α_{3.60} ≈ α_{k−1} but α_{3.61} ≈ 2α_{k−1}, so the step doubles per iteration instead of
    staying short. On a quadratic with exact steps the two guesses are equal. The strong Wolfe
    search expands α up to α_max = 2²⁰·α₀; if φ still decreases at α_max, the step α_max
    (which satisfies the Armijo condition) is taken.

    On a quadratic with exact steps, f − f* falls at least by the factor ((κ − 1)/(κ + 1))²
    per iteration (N&W Theorem 3.3).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. With the defaults it converges on
    ``quadratic_bowl`` (16 iterations) but NOT on ``rosenbrock``: ∇²f(x*) has κ ≈ 2500, so
    steepest descent needs ≈ 10⁴ iterations; it stops at ``max_iter`` = 5000 with
    f ≈ 5e-7 and ‖∇f‖ ≈ 1e-3. A line search that finds no acceptable step stops the run
    (converged=False), and so does a search that breaks down with an arithmetic exception (step
    lengths at the limit of float64); a trial with f = −∞ is reported as divergence (f unbounded
    below). The default ``lr`` = 1e-3 of the fixed rule is below 2/L on
    ``quadratic_bowl`` and ``rosenbrock`` (stable but slow: neither converges in 5000
    iterations); on ``goldstein_price`` it diverges at the first step.
    """
    if step_rule not in STEP_RULES:
        raise ValueError(f"step_rule must be one of {STEP_RULES}, got {step_rule!r}")
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr)
    run = _Run("gradient_descent", problem, x0, gtol, need_hess=step_rule == "exact_quadratic")
    is_fixed = step_rule == "fixed"
    done = run.begin(trials=[], alpha0=None, pHp=None)
    if done is not None:
        return done
    alpha_prev: float | None = None
    gg_prev = math.nan
    f_prev = math.nan
    for k in range(1, max_iter + 1):
        p = -run.g
        with np.errstate(over="ignore"):
            gg = float(run.g @ run.g)  # = −∇fᵀp, the slope of the line search
        if not is_fixed and not math.isfinite(gg):
            # NOTE: search() requires a finite slope ∇fᵀp; ‖∇f‖ > ~1e154 overflows it.
            return run.result(False, _overflow_message(k))
        if is_fixed:
            x_new = run.x + lr * p
            done = run.step_to(k, x_new, lr, p, trials=[], alpha0=None, pHp=None)
        else:
            pHp: float | None = None
            alpha0: float | None = None
            opts: dict[str, Any] = {}
            if step_rule == "exact_quadratic":
                H = run.hessian(run.x)
                pHp = float(p @ (H @ p))
                if not (math.isfinite(pHp) and pHp > 0.0):
                    return run.result(
                        False,
                        f"exact_quadratic step undefined: pᵀ∇²f p = {pHp:.3g} ≤ 0 "
                        f"at iteration {k} (f is not convex along −∇f)",
                    )
                opts["hess"] = H
            else:
                alpha0 = _gd_alpha0(gg, run.fx, alpha_prev, gg_prev, f_prev)
                opts["alpha0"] = alpha0
                if step_rule == "strong_wolfe":
                    opts["alpha_max"] = min(ALPHA_MAX_RATIO * alpha0, FLOAT_MAX)
            try:
                ls = run.line_search(step_rule, p, **opts)
            except ArithmeticError as exc:
                # NOTE: never raise on numerical breakdown (as in conjugate_gradient._search).
                # The zoom of the strong Wolfe search divides by h², which underflows for
                # brackets narrower than ≈ 1.5e-162 (step lengths α ≲ 1e-154, f scaled by ≈ 2⁴⁹⁶).
                return run.result(
                    False,
                    f"line search broke down at iteration {k} ({type(exc).__name__}: {exc}); "
                    "the step lengths are at the limit of float64 (rescale f)",
                )
            alpha, f_new, g_ls = ls.alpha, ls.f_new, ls.g_new
            if not ls.success:
                last = ls.trials[-1] if ls.trials else None
                # NOTE: N&W Alg. 3.5 fails when φ still decreases at α_max. That step satisfies
                # the Armijo condition (else the search would have zoomed), so we take it
                # instead of stopping: α_max = 2²⁰α₀ is only a cap on one search, and the next
                # α₀ ≥ α_{3.60} grows with it. A truly unbounded f then drives f to −∞, which
                # is reported as divergence. ∇f at α_max is evaluated again (one more n_gev).
                if (
                    step_rule == "strong_wolfe"
                    and last is not None
                    and last[0] == opts["alpha_max"]
                    and math.isfinite(last[1])
                    and last[1] <= run.fx - C1_DEFAULT * last[0] * gg
                ):
                    alpha, f_new, g_ls = last[0], last[1], None
                else:
                    a_inf = next((a for a, phi in ls.trials if phi == -math.inf), None)
                    if a_inf is not None:
                        return run.result(
                            False,
                            f"diverged: f(x + αp) = −∞ at α = {a_inf:.3g} in the line "
                            f"search of iteration {k} (f is unbounded below along −∇f)",
                        )
                    return run.result(False, f"line search failed at iteration {k}: {ls.message}")
            x_new = run.x + alpha * p
            if np.array_equal(x_new, run.x):
                return run.result(False, _stall_message(k, alpha))
            g_new = run.gradient(x_new) if g_ls is None else g_ls
            trials = [[a, phi] for a, phi in ls.trials]
            alpha_prev, gg_prev, f_prev = alpha, gg, run.fx
            done = run.advance(
                k, x_new, f_new, g_new, alpha, p, trials=trials, alpha0=alpha0, pHp=pHp
            )
        if done is not None:
            return done
    return run.max_iter(max_iter)


def _gd_alpha0(
    gg: float, f_cur: float, alpha_prev: float | None, gg_prev: float, f_prev: float
) -> float:
    """First trial step of a gradient-descent line search (see ``gradient_descent``).

    ``gg`` = ‖∇f(x_{k−1})‖² (finite and > 0), ``f_cur`` = f(x_{k−1}); ``alpha_prev``,
    ``gg_prev``, ``f_prev`` describe the previous iteration (``alpha_prev`` is None at k = 1).
    """
    unit = 1.0 / math.sqrt(gg)  # a trial move of unit length
    if alpha_prev is None:
        return unit
    # NOTE: N&W §3.5 offer eq. 3.60 and eq. 3.61 as alternatives; we take the larger one.
    # Eq. 3.60 alone cannot grow a step that was much too short (∇f barely changes), and on
    # Rosenbrock, Beale and quadratic_bowl the maximum needed fewer iterations than eq. 3.61.
    guesses = [
        a
        for a in (
            2.0 * (f_prev - f_cur) / gg,  # N&W eq. 3.61 (φ'(0) = −gg)
            alpha_prev * gg_prev / gg,  # N&W eq. 3.60
        )
        if math.isfinite(a) and a > 0.0
    ]
    return max(guesses) if guesses else unit


# --------------------------------------------------------------------------------------
# Barzilai–Borwein
# --------------------------------------------------------------------------------------

BB_VARIANTS = ("bb1", "bb2")


@register(
    id="barzilai_borwein",
    family="unconstrained",
    name="Barzilai–Borwein",
    params=(
        ParamSpec(
            "variant",
            "bb1",
            kind="choice",
            choices=BB_VARIANTS,
            help="BB1: α = sᵀs/sᵀy (long step); BB2: α = sᵀy/yᵀy (short step).",
        ),
        ParamSpec(
            "nonmonotone",
            True,
            kind="bool",
            help="Grippo–Lampariello–Lucidi nonmonotone backtracking (M = 10); off = pure BB.",
        ),
        P_GTOL,
        _p_max_iter(1000),
    ),
    needs=("f", "grad"),
    order="R-superlinear on 2-D quadratics, R-linear on n-D quadratics; not monotone",
    summary="Gradient steps whose length mimics a secant (quasi-Newton) equation from the last step.",
    references=(
        "Barzilai & Borwein (1988), IMA J. Numer. Anal. 8, 141–148",
        "Raydan (1997), SIAM J. Optim. 7(1), 26–33, Algorithm GBB",
        "Grippo, Lampariello & Lucidi (1986), SIAM J. Numer. Anal. 23(4), 707–716",
    ),
)
def barzilai_borwein(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    variant: str = "bb1",
    nonmonotone: bool = True,
    gtol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Barzilai–Borwein gradient method (Barzilai & Borwein 1988; Raydan 1997).

    x_k = x_{k−1} − α_k ∇f(x_{k−1}) with the two-point step length computed from
    s = x_{k−1} − x_{k−2} and y = ∇f(x_{k−1}) − ∇f(x_{k−2}):

        BB1: α = sᵀs / sᵀy        BB2: α = sᵀy / yᵀy

    (α⁻¹I is the scalar matrix that best fits the secant equation Bs = y in the two
    least-squares senses). On a quadratic with Hessian A both lie in [1/λ_max, 1/λ_min],
    and BB2 ≤ BB1 (Cauchy–Schwarz).

    Safeguard (always on): at k = 1, or when the BB value is undefined, non-positive
    (sᵀy ≤ 0) or outside [1e-10, 1e10], the trial step is reset to 1/‖∇f(x_{k−1})‖₂.

    With ``nonmonotone=True`` the trial step is accepted when (GLL condition, Raydan 1997)

        f(x_{k−1} − α∇f) ≤ max_{0≤j<M} f(x_{k−1−j}) − γ α ‖∇f(x_{k−1})‖²,   M = 10, γ = 1e-4,

    else α ← σα with σ the safeguarded minimizer of the quadratic interpolant of
    φ(α) = f(x_{k−1} − α∇f), clipped to [0.1, 0.5] (0.1 when φ(α) is not finite). With
    ``nonmonotone=False`` the BB step is always taken (pure BB, may diverge on non-convex f).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (7 iterations) and ``rosenbrock`` (71). Pure BB1
    (``nonmonotone=False``) does not converge on ``rosenbrock`` within 1000 iterations.
    """
    # NOTE: Raydan (1997) resets the reciprocal step to a δ that depends on ‖∇f‖; we reset the
    # step to 1/‖∇f‖₂ (a trial move of unit length), which plays the same role.
    if variant not in BB_VARIANTS:
        raise ValueError(f"variant must be one of {BB_VARIANTS}, got {variant!r}")
    max_iter = _check_common(gtol, max_iter)
    run = _Run("barzilai_borwein", problem, x0, gtol)
    done = run.begin(alpha_bb=None, bb1=None, bb2=None, reset=False, f_ref=None, trials=[])
    if done is not None:
        return done
    f_hist = [run.fx]
    x_prev: Vector | None = None
    g_prev: Vector | None = None
    for k in range(1, max_iter + 1):
        with np.errstate(over="ignore"):
            gg = float(run.g @ run.g)
        if not math.isfinite(gg):
            return run.result(False, _overflow_message(k))
        bb1: float | None = None
        bb2: float | None = None
        if x_prev is not None and g_prev is not None:
            s = run.x - x_prev
            y = run.g - g_prev
            bb1, bb2 = _bb_steps(s, y)
        alpha_bb = bb1 if variant == "bb1" else bb2
        reset = alpha_bb is None or not (BB_MIN <= alpha_bb <= BB_MAX)
        if reset:
            alpha_bb = min(BB_MAX, max(BB_MIN, 1.0 / math.sqrt(gg)))
        assert alpha_bb is not None
        p = -run.g
        alpha = alpha_bb
        f_new = run.value(run.x + alpha * p)
        trials = [[alpha, f_new]]
        f_ref: float | None = None
        if nonmonotone:
            f_ref = max(f_hist[-BB_MEMORY:])
            while not (math.isfinite(f_new) and f_new <= f_ref - BB_GAMMA * alpha * gg):
                if len(trials) >= BB_MAX_TRIALS:
                    return run.result(
                        False,
                        f"nonmonotone line search failed at iteration {k}: no acceptable step "
                        f"in {BB_MAX_TRIALS} trials (last α = {alpha:.3g})",
                    )
                alpha *= _bb_sigma(alpha, f_new, run.fx, gg)
                f_new = run.value(run.x + alpha * p)
                trials.append([alpha, f_new])
        x_new = run.x + alpha * p
        if np.array_equal(x_new, run.x):
            return run.result(False, _stall_message(k, alpha))
        x_prev, g_prev = run.x, run.g
        done = run.advance(
            k,
            x_new,
            f_new,
            run.gradient(x_new),
            alpha,
            p,
            alpha_bb=alpha_bb,
            bb1=bb1,
            bb2=bb2,
            reset=reset,
            f_ref=f_ref,
            trials=trials,
        )
        if done is not None:
            return done
        f_hist.append(run.fx)
    return run.max_iter(max_iter)


def _pow2_scaled(v: Vector) -> tuple[Vector, int]:
    """(v·2⁻ᵉ, e) with 2^(e−1) ≤ ‖v‖∞ < 2^e; (v, 0) when v = 0. The scaling is exact."""
    _, e = math.frexp(float(np.max(np.abs(v))))
    return np.ldexp(v, -e), e


def _bb_steps(s: Vector, y: Vector) -> tuple[float | None, float | None]:
    """(BB1, BB2) = (sᵀs/sᵀy, sᵀy/yᵀy) for finite s ≠ 0; (None, None) when sᵀy ≤ 0.

    The inner products are formed from s and y scaled by powers of 2 (Higham (2002), §27.8):
    the scaling is exact, so the quotients equal the unscaled ones bit for bit whenever no
    product under- or overflows, and stay defined when yᵀy or sᵀy would underflow to 0.
    """
    # NOTE: unscaled, yᵀy underflows to 0 for ‖y‖∞ ≲ 1e-162 while sᵀy > 0 is still
    # representable (‖s‖ ≫ ‖y‖), and sᵀy / yᵀy raised ZeroDivisionError. A quotient that
    # overflows becomes +inf, which the caller's range test [BB_MIN, BB_MAX] resets.
    s_hat, e_s = _pow2_scaled(s)
    y_hat, e_y = _pow2_scaled(y)
    sy = float(s_hat @ y_hat)
    if not sy > 0.0:  # also y = 0
        return None, None
    ss, yy = float(s_hat @ s_hat), float(y_hat @ y_hat)  # both in [1/4, n)
    with np.errstate(over="ignore", under="ignore"):
        bb1 = float(np.ldexp(ss / sy, e_s - e_y))
        bb2 = float(np.ldexp(sy / yy, e_s - e_y))
    return bb1, bb2


def _bb_sigma(alpha: float, f_alpha: float, f0: float, gg: float) -> float:
    """Backtracking factor σ ∈ [σ₁, σ₂] from the quadratic interpolant of φ(α) = f(x − αg).

    q(t) = φ(0) + φ'(0)t + c t² with φ'(0) = −gᵀg and c = (φ(α) − φ(0) + gᵀg α)/α²; its
    minimizer is t* = gᵀg/(2c), so σ = t*/α (Raydan 1997, §3; N&W eq. 3.58).
    """
    if not math.isfinite(f_alpha):
        return BB_SIGMA1
    alpha_sq = alpha * alpha
    if alpha_sq == 0.0:  # α < 1.5e-162: c is not representable
        return BB_SIGMA2
    curv = (f_alpha - f0 + gg * alpha) / alpha_sq
    if not (math.isfinite(curv) and curv > 0.0):
        return BB_SIGMA2
    denom = 2.0 * curv * alpha
    if denom == 0.0:  # underflow with c > 0: σ = gᵀg/(2cα) exceeds every float, clip to σ₂
        return BB_SIGMA2
    sigma = gg / denom
    return min(BB_SIGMA2, max(BB_SIGMA1, sigma))


# --------------------------------------------------------------------------------------
# Momentum methods
# --------------------------------------------------------------------------------------


@register(
    id="momentum",
    family="unconstrained",
    name="Heavy-ball momentum (Polyak)",
    params=(
        _p_lr(1e-3),
        _p_beta("beta", 0.9, "Momentum coefficient β ∈ [0, 1)."),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="linear; rate (√κ − 1)/(√κ + 1) on quadratics with optimal lr, β",
    summary="Gradient descent plus a fraction β of the previous step (a heavy ball rolling downhill).",
    references=(
        "Polyak (1964), USSR Comput. Math. Math. Phys. 4(5), 1–17",
        "Sutskever, Martens, Dahl & Hinton (2013), ICML, eq. 1–2",
    ),
)
def momentum(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 1e-3,
    beta: float = 0.9,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Polyak's heavy-ball method (Polyak 1964).

        v_k = β v_{k−1} − lr ∇f(x_{k−1}),   x_k = x_{k−1} + v_k,   v₀ = 0,

    i.e. x_k = x_{k−1} − lr ∇f(x_{k−1}) + β(x_{k−1} − x_{k−2}). On a quadratic with Hessian
    eigenvalues in [μ, L] it converges for 0 < lr·L < 2(1 + β); lr = 4/(√L + √μ)² and
    β = ((√κ − 1)/(√κ + 1))² give the optimal rate (√κ − 1)/(√κ + 1).
    ``direction`` = v_k/lr, ``alpha`` = lr.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (607 iterations) and ``rosenbrock`` (3020).
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr)
    _check_unit(beta=beta)
    run = _Run("momentum", problem, x0, gtol)
    v = np.zeros_like(run.x)
    done = run.begin(velocity=_vec(v))
    if done is not None:
        return done
    for k in range(1, max_iter + 1):
        v = beta * v - lr * run.g
        done = run.step_to(k, run.x + v, lr, v / lr, velocity=_vec(v))
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="nesterov",
    family="unconstrained",
    name="Nesterov accelerated gradient",
    params=(
        _p_lr(1e-3),
        _p_beta("beta", 0.9, "Momentum coefficient μ ∈ [0, 1)."),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="linear on strongly convex f for suitable constant lr, μ: rate 1 − 1/√κ in f with "
    "lr = 1/L, μ = (√κ − 1)/(√κ + 1); the O(1/k²) rate needs a schedule μ_k → 1 (not implemented)",
    summary="Momentum that takes the gradient at the look-ahead point x + μv instead of at x.",
    references=(
        "Sutskever, Martens, Dahl & Hinton (2013), ICML, eq. 3–4",
        "Nesterov (1983), Soviet Math. Dokl. 27(2), 372–376",
        "Nesterov (2004), Introductory Lectures on Convex Optimization, §2.2.1, "
        "constant step scheme III",
    ),
)
def nesterov(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 1e-3,
    beta: float = 0.9,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Nesterov's accelerated gradient in the form of Sutskever et al. (2013), eq. 3–4.

        v_k = μ v_{k−1} − lr ∇f(x_{k−1} + μ v_{k−1}),   x_k = x_{k−1} + v_k,   v₀ = 0,

    with μ = ``beta``. Equivalent to Nesterov's two-sequence form
    y_{k−1} = x_{k−1} + μ(x_{k−1} − x_{k−2}), x_k = y_{k−1} − lr ∇f(y_{k−1}).
    ``direction`` = v_k/lr, ``alpha`` = lr.

    μ is constant. On a σ-strongly convex, L-smooth f (κ = L/σ), lr = 1/L and
    μ = (√κ − 1)/(√κ + 1) give f(x_k) − f* ≤ (f(x₀) − f* + (σ/2)‖x₀ − x*‖²)(1 − 1/√κ)ᵏ
    (Nesterov 2004, §2.2.1, constant step scheme III). The O(1/k²) rate on convex f needs
    the schedule μ_k = (k − 1)/(k + 2), which is not implemented.

    Each iteration evaluates ∇f at the look-ahead point and at x_k (for the stopping test),
    so ``n_gev`` ≈ 2·n_iter (the look-ahead gradient is reused when μv_{k−1} = 0, e.g. at k = 1).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (627 iterations) and ``rosenbrock`` (3067).
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr)
    _check_unit(beta=beta)
    run = _Run("nesterov", problem, x0, gtol)
    v = np.zeros_like(run.x)
    done = run.begin(velocity=_vec(v), lookahead=None, grad_lookahead=None)
    if done is not None:
        return done
    for k in range(1, max_iter + 1):
        y = run.x + beta * v
        gy = run.g if np.array_equal(y, run.x) else run.gradient(y)
        if not bool(np.all(np.isfinite(gy))):
            return run.result(False, f"diverged: ∇f is not finite at the look-ahead point (k={k})")
        v = beta * v - lr * gy
        done = run.step_to(
            k,
            run.x + v,
            lr,
            v / lr,
            velocity=_vec(v),
            lookahead=_vec(y),
            grad_lookahead=_vec(gy),
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Adaptive (per-coordinate) methods
# --------------------------------------------------------------------------------------


def _safe_div(num: Vector, den: Vector) -> Vector:
    """num/den elementwise, with 0 where den == 0 (see the callers for when that happens).

    A quotient that overflows is ±inf without a warning; the callers bound |num/den| or
    report the resulting non-finite iterate as divergence.
    """
    out = np.zeros_like(num)
    with np.errstate(over="ignore", under="ignore"):
        np.divide(num, den, out=out, where=den != 0.0)
    return out


def _step_sizes(scale: float, den: Vector) -> Vector:
    """scale/den elementwise for ``info["lr_eff"]``: 0 where den == 0, capped at FLOAT_MAX.

    The cap only affects the reported value (den ≲ scale·6e-309); the step itself is
    computed from a bounded quotient.
    """
    with np.errstate(over="ignore"):
        return np.minimum(scale * _safe_div(np.ones_like(den), den), FLOAT_MAX)


@register(
    id="adagrad",
    family="unconstrained",
    name="AdaGrad",
    params=(_p_lr(1.0), _p_eps(1e-8), P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad"),
    order="sublinear in general (step sizes shrink like 1/√k)",
    summary="Divide each coordinate's step by the root of the sum of all its past squared gradients.",
    references=("Duchi, Hazan & Singer (2011), JMLR 12, 2121–2159 (diagonal AdaGrad)",),
)
def adagrad(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 1.0,
    eps: float = 1e-8,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Diagonal AdaGrad (Duchi, Hazan & Singer 2011), with g_k = ∇f(x_{k−1}):

        G_k = G_{k−1} + g_k²,    x_k = x_{k−1} − lr · g_k / (√G_k + ε),    G₀ = 0

    (ε is the δ of the paper's H = δI + diag(G)^{1/2}). The per-coordinate step
    lr/(√G_k + ε) never increases. ``direction`` = −g_k/(√G_k + ε), ``alpha`` = lr.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. With the defaults it converges on
    ``quadratic_bowl`` (54 iterations) but NOT on ``rosenbrock``: the accumulated G makes
    the steps too short to follow the curved valley to x*; it stops at ``max_iter`` with
    f ≈ 7e-5 (no lr in [0.005, 1] converges there within 5000 iterations).
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr, eps=eps)
    run = _Run("adagrad", problem, x0, gtol)
    G = np.zeros_like(run.x)
    done = run.begin(v=_vec(G), lr_eff=None)
    if done is not None:
        return done
    for k in range(1, max_iter + 1):
        g = run.g
        G = G + g * g
        denom = np.sqrt(G) + eps
        p = -g / denom
        done = run.step_to(k, run.x + lr * p, lr, p, v=_vec(G), lr_eff=_vec(lr / denom))
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="rmsprop",
    family="unconstrained",
    name="RMSprop",
    params=(
        _p_lr(0.1),
        _p_beta("rho", 0.9, "Decay ρ of the running mean of squared gradients."),
        _p_eps(1e-8),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="no convergence guarantee with a constant lr; depending on lr it converges or "
    "hovers near x⋆",
    summary="AdaGrad with a decaying average of squared gradients, so step sizes do not vanish.",
    references=("Tieleman & Hinton (2012), COURSERA Neural Networks for ML, Lecture 6.5",),
)
def rmsprop(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.1,
    rho: float = 0.9,
    eps: float = 1e-8,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """RMSprop (Tieleman & Hinton 2012), with g_k = ∇f(x_{k−1}):

        E_k = ρ E_{k−1} + (1 − ρ) g_k²,    x_k = x_{k−1} − lr · g_k / (√E_k + ε),    E₀ = 0.

    ``direction`` = −g_k/(√E_k + ε), ``alpha`` = lr.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. With the defaults it converges on
    ``quadratic_bowl`` (62 iterations) but NOT on ``rosenbrock``: the normalized steps
    g/√E do not shrink in the narrow valley, so the iterates keep oscillating across it
    (it stops at ``max_iter``; no lr in [0.005, 1] converges there). With lr = 0.01 it does
    not converge on ``quadratic_bowl`` either (it hovers within ≈ lr of x*).
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr, eps=eps)
    _check_unit(rho=rho)
    run = _Run("rmsprop", problem, x0, gtol)
    E = np.zeros_like(run.x)
    done = run.begin(v=_vec(E), lr_eff=None)
    if done is not None:
        return done
    for k in range(1, max_iter + 1):
        g = run.g
        E = rho * E + (1.0 - rho) * g * g
        denom = np.sqrt(E) + eps
        p = -g / denom
        done = run.step_to(k, run.x + lr * p, lr, p, v=_vec(E), lr_eff=_vec(lr / denom))
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="adadelta",
    family="unconstrained",
    name="AdaDelta",
    params=(
        _p_beta("rho", 0.95, "Decay ρ of both running means."),
        ParamSpec(
            "eps", 1e-6, min=1e-12, max=1e-2, log=True, help="ε inside both RMS terms (> 0)."
        ),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="no convergence guarantee; slow start (first steps ≈ √ε)",
    summary="RMSprop whose learning rate is the RMS of recent updates, so no lr is needed.",
    references=(
        "Zeiler (2012), ADADELTA: an adaptive learning rate method, arXiv:1212.5701, Alg. 1",
    ),
)
def adadelta(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    rho: float = 0.95,
    eps: float = 1e-6,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """AdaDelta (Zeiler 2012, Algorithm 1), with g_k = ∇f(x_{k−1}) and RMS[z] = √(E[z²] + ε):

        E[g²]_k = ρE[g²]_{k−1} + (1 − ρ)g_k²
        Δx_k = −(RMS[Δx]_{k−1} / RMS[g]_k) · g_k
        E[Δx²]_k = ρE[Δx²]_{k−1} + (1 − ρ)Δx_k²
        x_k = x_{k−1} + Δx_k,       E[g²]₀ = E[Δx²]₀ = 0.

    There is no learning rate: ``alpha`` = 1 and ``direction`` = Δx_k.

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. The first steps have length
    ≈ √ε/√(1 − ρ) per coordinate and grow only as fast as the updates do. With the defaults
    it converges on ``quadratic_bowl`` (1010 iterations) but NOT on ``rosenbrock`` (it stops
    at ``max_iter`` far from x*, f ≈ 0.4).
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(eps=eps)
    _check_unit(rho=rho)
    run = _Run("adadelta", problem, x0, gtol)
    Eg = np.zeros_like(run.x)
    Edx = np.zeros_like(run.x)
    done = run.begin(v=_vec(Eg), u=_vec(Edx), lr_eff=None)
    if done is not None:
        return done
    for k in range(1, max_iter + 1):
        g = run.g
        Eg = rho * Eg + (1.0 - rho) * g * g
        lr_eff = np.sqrt(Edx + eps) / np.sqrt(Eg + eps)
        dx = -lr_eff * g
        Edx = rho * Edx + (1.0 - rho) * dx * dx
        done = run.step_to(k, run.x + dx, 1.0, dx, v=_vec(Eg), u=_vec(Edx), lr_eff=_vec(lr_eff))
        if done is not None:
            return done
    return run.max_iter(max_iter)


_ADAM_PARAMS_DOC = (
    _p_beta("beta1", 0.9, "Decay β₁ of the first moment (momentum)."),
    _p_beta("beta2", 0.999, "Decay β₂ of the second moment."),
)


def _adam_like(
    name: str,
    problem: Problem | Fn,
    x0: ArrayLike | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    gtol: float,
    max_iter: int,
) -> Result:
    """Adam (Kingma & Ba 2015, Alg. 1) and AdamW (Loshchilov & Hutter 2019, Alg. 2, η_t = 1)."""
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr, eps=eps)
    _check_unit(beta1=beta1, beta2=beta2)
    if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
        raise ValueError(f"weight_decay must be a finite number ≥ 0, got {weight_decay}")
    is_w = name == "adamw"
    run = _Run(name, problem, x0, gtol)
    m = np.zeros_like(run.x)
    v = np.zeros_like(run.x)
    zeros = _vec(m)
    extra0: dict[str, Any] = {"decay": None} if is_w else {}
    done = run.begin(m=zeros, v=zeros, m_hat=zeros, v_hat=zeros, lr_eff=None, **extra0)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        denom = np.sqrt(v_hat) + eps
        step = -lr * m_hat / denom
        extra: dict[str, Any] = {}
        if is_w:
            decay = -weight_decay * run.x
            step = step + decay
            extra["decay"] = _vec(decay)
        done = run.step_to(
            t,
            run.x + step,
            lr,
            step / lr,
            m=_vec(m),
            v=_vec(v),
            m_hat=_vec(m_hat),
            v_hat=_vec(v_hat),
            lr_eff=_vec(lr / denom),
            **extra,
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="adam",
    family="unconstrained",
    name="Adam",
    params=(_p_lr(0.05), *_ADAM_PARAMS_DOC, _p_eps(1e-8), P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad"),
    order="no convergence guarantee with a constant lr (Reddi et al. 2018)",
    summary="Momentum on the gradient and RMSprop scaling, both with bias-corrected averages.",
    references=("Kingma & Ba (2015), Adam: a method for stochastic optimization, ICLR, Alg. 1",),
)
def adam(
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
    """Adam (Kingma & Ba 2015, Algorithm 1), with g_t = ∇f(x_{t−1}):

        m_t = β₁m_{t−1} + (1 − β₁)g_t,      v_t = β₂v_{t−1} + (1 − β₂)g_t²
        m̂_t = m_t/(1 − β₁ᵗ),                v̂_t = v_t/(1 − β₂ᵗ)
        x_t = x_{t−1} − lr · m̂_t/(√v̂_t + ε),       m₀ = v₀ = 0.

    ``direction`` = −m̂_t/(√v̂_t + ε), ``alpha`` = lr.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (281 iterations) and ``rosenbrock`` (2457).

    NOTE: the paper's default lr is 1e-3; we use 0.05 so that a 2-D demo converges within
    a few thousand iterations (lr = 1e-3 would need ≥ 1000 steps just to travel 1 unit).
    """
    return _adam_like("adam", problem, x0, lr, beta1, beta2, eps, 0.0, gtol, max_iter)


@register(
    id="adamw",
    family="unconstrained",
    name="AdamW",
    params=(
        _p_lr(0.05),
        *_ADAM_PARAMS_DOC,
        _p_eps(1e-8),
        ParamSpec(
            "weight_decay",
            1e-3,
            min=0.0,
            max=0.1,
            help="Decoupled weight decay λ: each step also subtracts λ·x.",
        ),
        P_GTOL,
        _p_max_iter(5000),
    ),
    needs=("f", "grad"),
    order="no convergence guarantee with a constant lr; λ > 0 biases the limit toward 0",
    summary="Adam with weight decay applied directly to x instead of through the gradient.",
    references=("Loshchilov & Hutter (2019), Decoupled weight decay regularization, ICLR, Alg. 2",),
)
def adamw(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-3,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """AdamW (Loshchilov & Hutter 2019, Algorithm 2, line 12 with schedule multiplier η_t = 1):

        m_t, v_t, m̂_t, v̂_t as in Adam (from g_t = ∇f(x_{t−1}), with no λx term), and
        x_t = x_{t−1} − (lr · m̂_t/(√v̂_t + ε) + λ x_{t−1}).

    The decay λx is *decoupled*: it is not part of the gradient, so it is not rescaled by
    1/√v̂. ``direction`` = (x_t − x_{t−1})/lr, ``alpha`` = lr; ``info["decay"]`` = −λx_{t−1}.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol`` (the gradient of f, not of a regularized f).
    With λ = 0 the iterates equal Adam's. With the default λ = 1e-3 it does NOT converge on
    ``quadratic_bowl`` or ``rosenbrock``: the decay and the normalized Adam step do not
    balance at a point, and the iterates oscillate in a band of width ≈ 0.02 around x*
    (‖∇f‖ ≈ 2e-3 on ``quadratic_bowl`` at ``max_iter``). It converges when x* = 0
    (``quadratic_ill``), where the decay vanishes at x*.

    NOTE: as in the paper, λ is not multiplied by lr. ``torch.optim.AdamW`` subtracts
    lr·λ·x instead, so its ``weight_decay`` equals λ/lr here.
    """
    return _adam_like("adamw", problem, x0, lr, beta1, beta2, eps, weight_decay, gtol, max_iter)


@register(
    id="adamax",
    family="unconstrained",
    name="AdaMax",
    params=(_p_lr(0.2), *_ADAM_PARAMS_DOC, P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad"),
    order="no convergence guarantee with a constant lr",
    summary="Adam with the second moment replaced by an exponentially weighted max of |g|.",
    references=(
        "Kingma & Ba (2015), Adam: a method for stochastic optimization, ICLR, §7.1, Alg. 2",
    ),
)
def adamax(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """AdaMax (Kingma & Ba 2015, §7.1, Algorithm 2), with g_t = ∇f(x_{t−1}):

        m_t = β₁m_{t−1} + (1 − β₁)g_t,    u_t = max(β₂u_{t−1}, |g_t|),
        x_t = x_{t−1} − (lr/(1 − β₁ᵗ)) · m_t/u_t,      m₀ = u₀ = 0.

    ``direction`` = −(m_t/u_t)/(1 − β₁ᵗ), ``alpha`` = lr.

    The quotient m_t/u_t is formed first (the paper's order): since u_t ≥ β₂^{t−j}|g_j| for
    every j ≤ t, |m_t|/u_t ≤ (1 − β₁)Σ_{j<t}(β₁/β₂)ʲ when β₂ > 0, so it cannot overflow even
    for a gradient as small as 1e-308, whereas the reciprocal 1/((1 − β₁ᵗ)u_t) would. The
    step is invariant under a rescaling of one coordinate's gradients; gradients in the
    subnormal range (|g| < 2.2e-308) lose relative precision in m_t and u_t.

    Where u_t = 0 every past gradient of that coordinate was 0 (or, with β₂ = 0, the current
    one is, or β₂u_{t−1} underflowed); the step is then 0.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (231 iterations) and ``rosenbrock`` (2914).

    NOTE: the paper's default lr is 2e-3; we use 0.2 for 2-D demos (see ``adam``).
    NOTE: no ε (as in the paper). The case u_t = 0 is a zero step: it is 0/0 when all past
    gradients of the coordinate were 0; when m_t ≠ 0 (β₂ = 0 with g_t,i = 0, or an underflow
    of β₂u_{t−1}) the paper's step is undefined and we also take 0.
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr)
    _check_unit(beta1=beta1, beta2=beta2)
    run = _Run("adamax", problem, x0, gtol)
    m = np.zeros_like(run.x)
    u = np.zeros_like(run.x)
    done = run.begin(m=_vec(m), u=_vec(u), lr_eff=None)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        m = beta1 * m + (1.0 - beta1) * g
        u = np.maximum(beta2 * u, np.abs(g))
        bias = 1.0 - beta1**t
        p = -_safe_div(m, u) / bias  # bounded m/u first; 0 where u = 0
        lr_eff = _step_sizes(lr / bias, u)
        done = run.step_to(t, run.x + lr * p, lr, p, m=_vec(m), u=_vec(u), lr_eff=_vec(lr_eff))
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="nadam",
    family="unconstrained",
    name="NAdam",
    params=(_p_lr(0.02), *_ADAM_PARAMS_DOC, _p_eps(1e-8), P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad"),
    order="no convergence guarantee with a constant lr",
    summary="Adam with Nesterov momentum: the step already uses the next iteration's momentum.",
    references=(
        "Dozat (2016), Incorporating Nesterov momentum into Adam, ICLR Workshop; "
        "Stanford CS229 report, Alg. 8",
    ),
)
def nadam(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """NAdam (Dozat 2016, Algorithm 8 of the CS229 report) with a constant μ_t = β₁:

        ĝ_t = g_t/(1 − β₁ᵗ),          m_t = β₁m_{t−1} + (1 − β₁)g_t,   m̂_t = m_t/(1 − β₁ᵗ⁺¹)
        n_t = β₂n_{t−1} + (1 − β₂)g_t²,  n̂_t = n_t/(1 − β₂ᵗ)
        m̄_t = (1 − β₁)ĝ_t + β₁m̂_t,     x_t = x_{t−1} − lr · m̄_t/(√n̂_t + ε),

    with g_t = ∇f(x_{t−1}) and m₀ = n₀ = 0. ``direction`` = −m̄_t/(√n̂_t + ε), ``alpha`` = lr.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (670 iterations) and ``rosenbrock`` (3714).

    NOTE: the lr default 0.02 is for 2-D demos (see ``adam``).
    NOTE: Dozat uses a warming schedule μ_t = μ(1 − ½·0.96^{t/250}); we keep μ_t = β₁
    constant (so Π_{i≤t} μ_i = β₁ᵗ), matching ``adam``'s parameters.
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr, eps=eps)
    _check_unit(beta1=beta1, beta2=beta2)
    run = _Run("nadam", problem, x0, gtol)
    m = np.zeros_like(run.x)
    n = np.zeros_like(run.x)
    zeros = _vec(m)
    done = run.begin(m=zeros, v=zeros, m_bar=zeros, v_hat=zeros, lr_eff=None)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        g_hat = g / (1.0 - beta1**t)
        m = beta1 * m + (1.0 - beta1) * g
        m_hat = m / (1.0 - beta1 ** (t + 1))
        n = beta2 * n + (1.0 - beta2) * g * g
        n_hat = n / (1.0 - beta2**t)
        m_bar = (1.0 - beta1) * g_hat + beta1 * m_hat
        denom = np.sqrt(n_hat) + eps
        p = -m_bar / denom
        done = run.step_to(
            t,
            run.x + lr * p,
            lr,
            p,
            m=_vec(m),
            v=_vec(n),
            m_bar=_vec(m_bar),
            v_hat=_vec(n_hat),
            lr_eff=_vec(lr / denom),
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


@register(
    id="amsgrad",
    family="unconstrained",
    name="AMSGrad",
    params=(_p_lr(0.1), *_ADAM_PARAMS_DOC, P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad"),
    order="linear near a strongly convex minimizer once v̂ stops growing",
    summary="Adam whose per-coordinate step can only shrink: divide by the running max of v.",
    references=("Reddi, Kale & Kumar (2018), On the convergence of Adam and beyond, ICLR, Alg. 2",),
)
def amsgrad(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """AMSGrad (Reddi, Kale & Kumar 2018, Algorithm 2) with constant α_t = lr and β₁t = β₁:

        m_t = β₁m_{t−1} + (1 − β₁)g_t,   v_t = β₂v_{t−1} + (1 − β₂)g_t²,
        v̂_t = max(v̂_{t−1}, v_t),        x_t = x_{t−1} − lr · m_t/√v̂_t,     m₀ = v₀ = v̂₀ = 0,

    with g_t = ∇f(x_{t−1}); no bias correction and no ε (as in the paper; F = ℝⁿ, so the
    projection is the identity). The per-coordinate step lr/√v̂_t never increases.
    ``direction`` = −m_t/√v̂_t, ``alpha`` = lr. Since √v̂_t ≥ √v_j ≥ √(1 − β₂)|g_j| for every
    j ≤ t, the step obeys |m_t|/√v̂_t ≤ (1 − β₁ᵗ)/√(1 − β₂).

    The method keeps the RMS values r_t = √v_t = hypot(√β₂ r_{t−1}, √(1 − β₂) g_t) and
    r̂_t = max(r̂_{t−1}, r_t) = √v̂_t instead of v_t and v̂_t: g_t² underflows to 0 for
    |g_t| ≲ 1e-161 (and overflows for |g_t| ≳ 1e154), which would freeze a coordinate
    whose paper step is ≈ 3 lr. The step is invariant under a rescaling of one coordinate's
    gradients. Where r̂_t = 0 all past gradients of that coordinate were 0 (or subnormal
    enough to underflow, |g| ≲ 1e-322), and the step is 0.

    Stops (converged) when ‖∇f(x_t)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (272 iterations) and ``rosenbrock`` (2944).

    NOTE: the paper's theory uses α_t = α/√t; constant α_t (used in its experiments with
    tuned α) keeps the demo moving.
    NOTE: v_t and v̂_t are kept as their square roots (see above); ``info["v"]`` and
    ``info["v_max"]`` report r_t² and r̂_t² (capped at the largest finite float), which may
    underflow or overflow although the step does not.
    """
    max_iter = _check_common(gtol, max_iter)
    _check_positive(lr=lr)
    _check_unit(beta1=beta1, beta2=beta2)
    run = _Run("amsgrad", problem, x0, gtol)
    sqrt_b2, sqrt_1mb2 = math.sqrt(beta2), math.sqrt(1.0 - beta2)
    m = np.zeros_like(run.x)
    r = np.zeros_like(run.x)  # r_t = √v_t
    r_max = np.zeros_like(run.x)  # r̂_t = √v̂_t
    zeros = _vec(m)
    done = run.begin(m=zeros, v=zeros, v_max=zeros, lr_eff=None)
    if done is not None:
        return done
    for t in range(1, max_iter + 1):
        g = run.g
        m = beta1 * m + (1.0 - beta1) * g
        r = np.hypot(sqrt_b2 * r, sqrt_1mb2 * g)  # √(β₂v_{t−1} + (1 − β₂)g²) without g²
        r_max = np.maximum(r_max, r)
        p = -_safe_div(m, r_max)  # |m|/r̂ ≤ (1 − β₁ᵗ)/√(1 − β₂); 0 where r̂ = 0
        with np.errstate(under="ignore", over="ignore"):  # reported only
            v, v_max = np.minimum(r * r, FLOAT_MAX), np.minimum(r_max * r_max, FLOAT_MAX)
        done = run.step_to(
            t,
            run.x + lr * p,
            lr,
            p,
            m=_vec(m),
            v=_vec(v),
            v_max=_vec(v_max),
            lr_eff=_vec(_step_sizes(lr, r_max)),
        )
        if done is not None:
            return done
    return run.max_iter(max_iter)


# --------------------------------------------------------------------------------------
# Cyclic coordinate descent
# --------------------------------------------------------------------------------------


@register(
    id="coordinate_descent",
    family="unconstrained",
    name="Cyclic coordinate descent",
    params=(P_GTOL, _p_max_iter(5000)),
    needs=("f", "grad", "hess"),
    order="linear (one Gauss–Seidel sweep per n iterations on quadratics)",
    summary="Minimize along one coordinate axis at a time, cycling through the axes.",
    references=(
        "Wright (2015), Coordinate descent algorithms, Math. Program. 151, 3–34, Alg. 1",
        "Nocedal & Wright (2006), §9.3",
        "Golub & Van Loan (2013), §11.2 (Gauss–Seidel)",
    ),
)
def coordinate_descent(
    problem: Problem | Fn,
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-6,
    max_iter: int = 5000,
) -> Result:
    """Cyclic coordinate descent with a 1-D Newton step (Wright 2015, Alg. 1).

    Iteration k updates the single coordinate i = (k − 1) mod n. With g = ∇f(x_{k−1}) and
    h = ∂²f/∂x_i²(x_{k−1}):

        p_k = −(g_i/h) e_i   if h > 0   (1-D Newton step; the exact minimizer along e_i when
                                          f is quadratic, i.e. one Gauss–Seidel update)
        p_k = −g_i e_i       if h ≤ 0   (gradient fallback; f is not convex along e_i)

    and x_k = x_{k−1} + α_k p_k with α_k from Armijo backtracking (N&W Alg. 3.1, α₀ = 1,
    c₁ = 1e-4, ρ = ½). On a quadratic the Newton step satisfies the Armijo test with α = 1,
    so n iterations are exactly one Gauss–Seidel sweep for Ax = Ac. If g_i = 0 the
    coordinate is skipped (x_k = x_{k−1}, α_k = 0); the same holds when the slope g_i·p_i
    underflows to 0 or when x_i + p_i rounds to x_i. A slope or step that overflows stops the run (converged=False).

    Each iteration evaluates the full gradient (for the stopping test) and the Hessian (only
    its diagonal entry h is used).

    Stops (converged) when ‖∇f(x_k)‖₂ ≤ ``gtol``. Converges with the defaults on
    ``quadratic_bowl`` (16 iterations); on ``rosenbrock`` it does NOT converge within 5000
    iterations (axis steps cannot follow the curved valley; it stops at ``max_iter`` with
    f ≈ 3e-7).
    """
    max_iter = _check_common(gtol, max_iter)
    run = _Run("coordinate_descent", problem, x0, gtol, need_hess=True)
    n = run.x.size
    done = run.begin(coordinate=None, sweep=None, curvature=None, newton=None, trials=[])
    if done is not None:
        return done
    unchanged = 0  # consecutive iterations that left x unchanged
    for k in range(1, max_iter + 1):
        x_old = run.x
        i = (k - 1) % n
        sweep = (k - 1) // n
        gi = float(run.g[i])
        h = float(run.hessian(run.x)[i, i])
        p = np.zeros_like(run.x)
        newton_step = math.isfinite(h) and h > 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            p_i = -gi / h if newton_step else -gi
            slope = gi * p_i  # φ'(0) = ∇fᵀp of the 1-D search
        if gi != 0.0 and not (math.isfinite(p_i) and math.isfinite(slope)):
            return run.result(False, _overflow_message(k))
        # NOTE: a slope that underflows to 0 (|g_i·p_i| < 5e-324), or a full step α = 1 that does
        # not change x_i in floating point, is treated like g_i = 0: backtracking only shrinks α,
        # so no step can change x_i; skip the coordinate (the stall test below then applies).
        if slope < 0.0 and run.x[i] + p_i != run.x[i]:
            p[i] = p_i
        if p[i] == 0.0:
            done = run.advance(
                k,
                run.x.copy(),
                run.fx,
                run.g,
                0.0,
                p,
                coordinate=i,
                sweep=sweep,
                curvature=h,
                newton=False,
                trials=[],
            )
        else:
            ls = run.line_search("backtracking", p, alpha0=1.0)
            if not ls.success:
                return run.result(False, f"line search failed at iteration {k}: {ls.message}")
            x_new = run.x + ls.alpha * p
            done = run.advance(
                k,
                x_new,
                ls.f_new,
                run.gradient(x_new),
                ls.alpha,
                p,
                coordinate=i,
                sweep=sweep,
                curvature=h,
                newton=newton_step,
                trials=[[a, phi] for a, phi in ls.trials],
            )
        if done is not None:
            return done
        unchanged = unchanged + 1 if np.array_equal(run.x, x_old) else 0
        if unchanged >= n:
            return run.result(
                False,
                f"stalled at iteration {k}: {n} consecutive coordinate steps left x unchanged "
                "(f cannot be decreased further at this floating-point precision)",
            )
    return run.max_iter(max_iter)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
# NOTE: 18 cases instead of 3–8 so that each of the 13 methods (and each gradient-descent
# step rule) is replayed at least once; every trace stays below 300 steps.
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("gradient_descent", "quadratic_bowl", {}),
    ("gradient_descent", "quadratic_ill", {"step_rule": "exact_quadratic"}),
    ("gradient_descent", "booth", {"step_rule": "strong_wolfe"}),
    ("gradient_descent", "quadratic_bowl", {"step_rule": "fixed", "lr": 0.2}),
    ("barzilai_borwein", "rosenbrock", {}),
    ("barzilai_borwein", "quadratic_ill", {"variant": "bb2", "nonmonotone": False}),
    ("momentum", "quadratic_ill", {"lr": 0.02, "beta": 0.7, "max_iter": 250}),
    ("nesterov", "quadratic_ill", {"lr": 0.02, "beta": 0.7, "max_iter": 250}),
    ("adagrad", "quadratic_bowl", {"max_iter": 250}),
    ("rmsprop", "himmelblau", {"max_iter": 250}),
    ("adadelta", "quadratic_bowl", {"max_iter": 250}),
    ("adam", "beale", {"lr": 0.2, "max_iter": 290}),
    ("adamw", "quadratic_bowl", {"max_iter": 250}),
    ("adamax", "quadratic_ill", {"max_iter": 290}),
    ("nadam", "quadratic_bowl", {"lr": 0.1, "max_iter": 250}),
    ("amsgrad", "quadratic_ill", {"lr": 0.01, "max_iter": 290}),
    ("coordinate_descent", "booth", {"max_iter": 250}),
    ("coordinate_descent", "himmelblau", {}),
]
