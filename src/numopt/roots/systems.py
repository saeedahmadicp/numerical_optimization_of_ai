"""Newton and Broyden methods for square nonlinear systems F(x) = 0, F: Rⁿ → Rⁿ.

Conventions shared by both methods (family ``"systems"``):

* **Problem.** ``f(x) -> (n,)`` is the residual map F and ``jac(x) -> (n, n)`` its Jacobian
  (row i = ∇Fᵢ). Without ``jac`` a central-difference Jacobian (:func:`numopt.core.diff.jacobian`,
  2n + 1 evaluations of F) is used; those evaluations are counted in ``n_fev`` and
  ``Result.extra["jacobian"] = "finite_difference"``. ``n_gev`` counts calls of ``jac``.
* **Trace.** Step k holds x_k, ``fun = ‖F(x_k)‖₂`` and ``step_size = ‖x_k − x_{k−1}‖₂``.
  Its ``info`` describes how x_k was produced from x_{k−1}. ``n_iter == trace[-1].k``.
* **Stopping test (converged).** ‖F(x_k)‖₂ ≤ ``ftol``, or the full (undamped) step p that
  produced x_k satisfies ‖p‖₂ ≤ ``xtol``·(1 + ‖x_{k−1}‖₂). The step test is relative: far
  from the origin it accepts a root to relative accuracy ``xtol``, as MINPACK's ``hybrd``
  does. For Broyden the step test also needs the secant correction
  ‖F(x_k)‖₂·‖s‖₂/‖y‖₂ ≤ ``xtol``·(1 + ‖x_{k−1}‖₂) (s = x_k − x_{k−1}, y = F(x_k) − F(x_{k−1})),
  because its step −H_k F comes from an approximate inverse Jacobian that can be wrong
  (see :func:`broyden`).
* **Failure (converged=False, never raises).** A singular or non-finite Jacobian
  (cond₂(J) > 1/ε), a non-finite F, divergence ‖x_k‖ > DIVERGENCE_FACTOR·max(1, ‖x₀‖),
  a failed backtracking search, an undefined Broyden update, a Broyden step that does
  not move x while ‖F‖ > ``ftol`` (stall), or ``max_iter``. An ``OverflowError`` /
  ``ValueError`` / ``ZeroDivisionError`` raised by F or J counts as a nan value.
* **Norms** are 2-norms computed with scaling (:func:`_norm`), so they neither overflow
  nor underflow for huge or tiny vectors.

Info keys:
    residual: [n]             F(x_k) (every step, including k = 0).
    residual_norm: float      ‖F(x_k)‖₂ (every step).
    step: [n]                 x_k − x_{k−1} (k ≥ 1).
    jacobian: [[n]]           the matrix that produced x_k: J(x_{k−1}) for Newton, the
                              Broyden approximation B_{k−1} for Broyden (k ≥ 1).
    newton_system only (k ≥ 1):
        newton_step: [n]      the full Newton step p = −J⁻¹F(x_{k−1}) (solved, not inverted).
        alpha: float          the step length taken (1.0 without damping).
        trials: [[alpha, phi]] backtracking trials, φ = ½‖F(x_{k−1} + αp)‖² ([] without damping).
    broyden only (k ≥ 1):
        secant: {s: [n], y: [n]}  s = x_k − x_{k−1}, y = F(x_k) − F(x_{k−1}); the next
                              approximation satisfies the secant equation B_k s = y.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core import diff
from ..core.counting import Counted, finite, start_point
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, Vector, as_vector

#: Machine epsilon of float64.
EPS = sys.float_info.epsilon
#: Divergence is declared when ‖x_k‖ > DIVERGENCE_FACTOR · max(1, ‖x₀‖).
DIVERGENCE_FACTOR = 1e12
#: Armijo constant c₁ for the merit function φ = ½‖F‖² (Nocedal & Wright, §3.1).
ARMIJO_C1 = 1e-4
#: Backtracking halves α at most this many times (α ≥ 2⁻³⁰ ≈ 9.3e-10).
MAX_BACKTRACKS = 30
#: Exceptions of floating-point origin that count as a nan value of F or J.
_FP_ERRORS = (ArithmeticError, ValueError)

SYSTEM_PARAMS = (
    ParamSpec(
        "ftol",
        1e-10,
        min=1e-15,
        max=1e-2,
        log=True,
        help="Stop when ‖F(x)‖₂ ≤ ftol.",
    ),
    ParamSpec(
        "xtol",
        1e-12,
        min=1e-16,
        max=1e-2,
        log=True,
        help="Stop when the full step ‖p‖₂ ≤ xtol·(1 + ‖x‖₂).",
    ),
    ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Iteration limit."),
)


class _System:
    """Counted residual and Jacobian with the finite-difference fallback."""

    def __init__(self, prob: Problem, extra: dict[str, Any]) -> None:
        self.F = Counted(prob.f)
        self.J = Counted(prob.jac) if prob.jac is not None else None
        self.extra = extra
        if self.J is None:
            extra["jacobian"] = "finite_difference"

    def residual(self, x: Vector) -> Vector:
        # NOTE: floating-point exceptions in F (math.sin(inf), float ** overflow, ...) become
        # nan entries, which the non-finite checks report; a method never raises on them.
        try:
            return as_vector(self.F(x))
        except _FP_ERRORS:
            return np.full(x.size, np.nan)

    def jacobian(self, x: Vector, *, exact: bool = True) -> Vector:
        try:
            if exact and self.J is not None:
                return np.array(self.J(x), dtype=np.float64)
            return diff.jacobian(self.F, x)
        except _FP_ERRORS:
            return np.full((x.size, x.size), np.nan)

    @property
    def n_gev(self) -> int:
        return self.J.n if self.J is not None else 0


def _setup(problem: Problem | Callable[[Vector], Vector], x0: Any) -> tuple[Problem, Vector]:
    if isinstance(problem, Problem):
        prob = problem
    else:
        if not callable(problem):
            raise TypeError("problem must be a numopt Problem or a callable F(x)")
        if x0 is None:
            raise ValueError("a starting point x0 is required for a bare callable F")
        prob = Problem(id="custom", name="custom", latex="F(x)", f=problem, dim=0, domain=())
    return prob, start_point(prob, x0)


def _norm(v: Vector) -> float:
    """‖v‖₂ without overflow or underflow: scale by max|vᵢ| first (LAPACK dnrm2 idea,
    Higham 2002, ch. 27). Non-finite entries give inf or nan as usual."""
    m = float(np.max(np.abs(v))) if v.size else 0.0
    if m == 0.0 or not math.isfinite(m):
        return m
    w = v / m
    return m * math.sqrt(float(w @ w))


def _singular(J: Vector) -> str | None:
    """Return a reason when J is not finite or numerically singular (cond₂ > 1/ε)."""
    if not finite(J):
        return "the Jacobian has non-finite entries"
    cond = float(np.linalg.cond(J))
    if not math.isfinite(cond) or cond > 1.0 / EPS:
        return f"the Jacobian is singular to working precision (cond₂ = {cond:.3g})"
    return None


def _check(
    x_new: Vector,
    F_new: Vector,
    p: Vector,
    x_old: Vector,
    x0_norm: float,
    ftol: float,
    xtol: float,
    *,
    secant: tuple[Vector, Vector] | None = None,
) -> tuple[bool, str] | None:
    """Shared tests after a step; returns (converged, message) to stop, else None.

    ``secant=(s, y)`` (Broyden) adds the stall test and the secant-correction condition on
    the step test (see the module docstring).
    """
    if not finite(x_new, F_new):
        return False, "diverged: non-finite x or F(x)"
    r = _norm(F_new)
    if r <= ftol:
        return True, f"‖F(x)‖₂ = {r:.3g} ≤ ftol"
    p_norm = _norm(p)
    tol = xtol * (1.0 + _norm(x_old))
    if secant is None:
        if p_norm <= tol:
            return True, f"step ‖p‖₂ = {p_norm:.3g} ≤ xtol·(1 + ‖x‖₂)"
    else:
        s, y = secant
        s_norm, y_norm = _norm(s), _norm(y)
        if s_norm == 0.0:
            return False, (
                f"stalled: the step ‖p‖₂ = {p_norm:.3g} does not change x in floating point "
                f"while ‖F(x)‖₂ = {r:.3g} > ftol (the Broyden matrix no longer models F)"
            )
        # ‖F(x_k)‖·‖s‖/‖y‖: the correction that the local slope of F along s (y = J̄s
        # exactly, J̄ the mean Jacobian on the segment) asks for. It is ≈ ‖s‖ near a root and
        # large where F is not small, whatever the Broyden matrix says.
        correction = r * (s_norm / y_norm) if y_norm > 0.0 else math.inf
        if p_norm <= tol and correction <= tol:
            return True, (
                f"step ‖p‖₂ = {p_norm:.3g} ≤ xtol·(1 + ‖x‖₂) and the secant correction "
                f"‖F‖‖s‖/‖y‖ = {correction:.3g} ≤ xtol·(1 + ‖x‖₂)"
            )
    bound = DIVERGENCE_FACTOR * max(1.0, x0_norm)
    if _norm(x_new) > bound:
        return False, f"diverged: ‖x‖₂ > {bound:.3g}"
    return None


def _start_step(x: Vector, Fx: Vector) -> Step:
    r = _norm(Fx)
    return Step(0, x.copy(), r, info={"residual": Fx.copy(), "residual_norm": r})


@register(
    id="newton_system",
    family="systems",
    name="Newton's method for systems",
    params=(
        *SYSTEM_PARAMS,
        ParamSpec(
            "damping",
            False,
            kind="bool",
            help="Backtrack along the Newton step until ½‖F‖² decreases enough (Armijo).",
        ),
    ),
    needs=("f", "jac", "x0"),
    order="quadratic (nonsingular root)",
    summary="Solve the linearization J(x)p = −F(x) and step to x + p (optionally damped).",
    references=(
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 11.1",
        "Nocedal & Wright (2006), §11.2 (merit function ½‖F‖², line search) and Alg. 3.1",
        "Dennis & Schnabel, Numerical Methods for Unconstrained Optimization and "
        "Nonlinear Equations (1983), §6.5",
    ),
)
def newton_system(
    problem: Problem | Callable[[Vector], Vector],
    *,
    x0: Any = None,
    ftol: float = 1e-10,
    xtol: float = 1e-12,
    max_iter: int = 100,
    damping: bool = False,
) -> Result:
    """Newton's method for F(x) = 0 (Nocedal & Wright Alg. 11.1).

    Each iteration solves J(x_k) p_k = −F(x_k) with an LU solve (no inverse) and sets
    x_{k+1} = x_k + α_k p_k. Without damping α_k = 1 and convergence is quadratic near a
    root with nonsingular Jacobian.

    With ``damping=True`` α_k is found by backtracking (α = 1, ½, ¼, …) on the merit
    function φ(x) = ½‖F(x)‖² until the Armijo condition
    φ(x_k + αp_k) ≤ (1 − 2c₁α) φ(x_k) holds (N&W §11.2; the Newton step is a descent
    direction for φ with ∇φᵀp = −‖F‖² = −2φ). This globalizes Newton away from the root;
    it can still stall at a local minimizer of ‖F‖ that is not a root, which is reported
    as a failed line search.

    Stops (converged) when ‖F(x_k)‖₂ ≤ ``ftol`` or ‖p_{k−1}‖₂ ≤ ``xtol``·(1 + ‖x_{k−1}‖₂).
    """
    prob, x = _setup(problem, x0)
    extra: dict[str, Any] = {}
    sysm = _System(prob, extra)
    Fx = sysm.residual(x)
    if Fx.size != x.size:
        raise ValueError(
            f"newton_system needs a square system; F has {Fx.size} rows, x has {x.size}"
        )
    x0_norm = _norm(x)
    trace = [_start_step(x, Fx)]

    def result(converged: bool, message: str, k: int) -> Result:
        return Result(
            "newton_system",
            x.copy(),
            _norm(Fx),
            converged,
            message,
            k,
            sysm.F.n,
            n_gev=sysm.n_gev,
            trace=trace,
            extra=extra,
        )

    if not finite(Fx):
        return result(False, "F(x₀) is not finite", 0)
    if _norm(Fx) <= ftol:
        return result(True, f"‖F(x)‖₂ = {_norm(Fx):.3g} ≤ ftol", 0)
    k = 0
    while True:
        if k == max_iter:
            return result(False, f"reached max_iter={max_iter}", k)
        J = sysm.jacobian(x)
        if (why := _singular(J)) is not None:
            return result(False, f"{why} at x = {np.array2string(x, precision=6)}", k)
        p = np.linalg.solve(J, -Fx)
        alpha = 1.0
        trials: list[list[float]] = []
        x_new = x + p
        F_new = sysm.residual(x_new)
        if damping:
            r0 = _norm(Fx)
            phi0 = 0.5 * r0 * r0
            for i in range(MAX_BACKTRACKS + 1):
                r_try = _norm(F_new) if finite(F_new) else math.inf
                phi = 0.5 * r_try * r_try
                trials.append([alpha, phi])
                if phi <= (1.0 - 2.0 * ARMIJO_C1 * alpha) * phi0:
                    break
                if i == MAX_BACKTRACKS:
                    return result(
                        False,
                        "line search failed: no sufficient decrease of ½‖F‖² along the "
                        "Newton step (x may be near a local minimizer of ‖F‖ that is not a root)",
                        k,
                    )
                alpha *= 0.5
                x_new = x + alpha * p
                F_new = sysm.residual(x_new)
        k += 1
        r_new = _norm(F_new)
        trace.append(
            Step(
                k,
                x_new.copy(),
                r_new,
                step_size=_norm(x_new - x),
                info={
                    "residual": F_new.copy(),
                    "residual_norm": r_new,
                    "step": x_new - x,
                    "jacobian": J,
                    "newton_step": p,
                    "alpha": alpha,
                    "trials": trials,
                },
            )
        )
        stop = _check(x_new, F_new, p, x, x0_norm, ftol, xtol)
        x, Fx = x_new, F_new
        if stop is not None:
            return result(stop[0], stop[1], k)


@register(
    id="broyden",
    family="systems",
    name="Broyden's method (good Broyden)",
    params=(
        *SYSTEM_PARAMS,
        ParamSpec(
            "jacobian0",
            "exact",
            kind="choice",
            choices=("exact", "finite_difference"),
            help="Initial Jacobian B₀: the exact J(x₀) or a central-difference estimate.",
        ),
    ),
    needs=("f", "x0"),
    order="superlinear",
    summary="A quasi-Newton method: one Jacobian at the start, then rank-one secant updates.",
    references=(
        "Broyden (1965), Math. Comp. 19(92), 577–593 (method 1, inverse update)",
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), §11.1",
        "Dennis & Schnabel (1983), §8.1",
    ),
)
def broyden(
    problem: Problem | Callable[[Vector], Vector],
    *,
    x0: Any = None,
    ftol: float = 1e-10,
    xtol: float = 1e-12,
    max_iter: int = 100,
    jacobian0: str = "exact",
) -> Result:
    """Broyden's "good" method with the inverse update (Broyden 1965, method 1).

    Starting from B₀ ≈ J(x₀) (exact or central differences) and H₀ = B₀⁻¹:

        p_k = −H_k F(x_k),  x_{k+1} = x_k + p_k,  s = x_{k+1} − x_k,  y = F(x_{k+1}) − F(x_k),
        H_{k+1} = H_k + (s − H_k y) sᵀH_k / (sᵀ H_k y)          (Sherman–Morrison)

    which is the inverse of the least-change secant update
    B_{k+1} = B_k + (y − B_k s)sᵀ/(sᵀs) (N&W §11.1). Only one Jacobian is ever formed;
    each iteration costs one F evaluation and O(n²) work. No line search: the method is
    locally superlinear and can fail from far away.

    Stops (converged) when ‖F(x_k)‖₂ ≤ ``ftol``, or when both ‖p_{k−1}‖₂ and the secant
    correction ‖F(x_k)‖₂·‖s‖₂/‖y‖₂ are ≤ ``xtol``·(1 + ‖x_{k−1}‖₂). Stops (failure) when
    sᵀH_k y = 0 (the update is undefined) or when x_k = x_{k−1} in floating point while
    ‖F(x_k)‖₂ > ``ftol``, besides the shared tests.

    NOTE: Newton's step solves with the exact J, so near a nonsingular root ‖p‖ estimates
    the error. Broyden's step −H_k F uses an approximation that can lose rank: on
    circle_line from (0.6, −0.6) with a finite-difference B₀ the first step went to
    x ≈ −9.6e10, where H_k F was below the float spacing of x, and the plain step test
    reported convergence at ‖F‖ = 1.8e22; on freudenstein_roth from (2, 1.386) the steps
    shrank geometrically at ‖F‖ = 9.3e11 with ‖B − J‖/‖J‖ = 3e10. A step that does not move
    x is therefore a stall, and a small step counts only when the residual agrees: y is the
    exact change of F along s, so ‖F‖‖s‖/‖y‖ is the correction a local secant model asks
    for (≈ ‖s‖ near a root). Dennis & Schnabel (1983, §7.2) likewise treat a small step
    alone as a possible stall, not as convergence.

    NOTE: the inverse form needs the explicit matrix H₀ = B₀⁻¹ (computed as the solve
    B₀H₀ = I); this is the method as specified, and n is small for these problems. B_k is
    also updated (by its own rank-one formula, no inverse) only to report the Jacobian
    approximation in ``info["jacobian"]``.
    """
    if jacobian0 not in ("exact", "finite_difference"):
        raise ValueError(f"jacobian0 must be 'exact' or 'finite_difference', got {jacobian0!r}")
    prob, x = _setup(problem, x0)
    extra: dict[str, Any] = {}
    sysm = _System(prob, extra)
    if jacobian0 == "finite_difference":
        extra["jacobian"] = "finite_difference"
    Fx = sysm.residual(x)
    if Fx.size != x.size:
        raise ValueError(f"broyden needs a square system; F has {Fx.size} rows, x has {x.size}")
    x0_norm = _norm(x)
    trace = [_start_step(x, Fx)]

    def result(converged: bool, message: str, k: int) -> Result:
        return Result(
            "broyden",
            x.copy(),
            _norm(Fx),
            converged,
            message,
            k,
            sysm.F.n,
            n_gev=sysm.n_gev,
            trace=trace,
            extra=extra,
        )

    if not finite(Fx):
        return result(False, "F(x₀) is not finite", 0)
    if _norm(Fx) <= ftol:
        return result(True, f"‖F(x)‖₂ = {_norm(Fx):.3g} ≤ ftol", 0)
    B = sysm.jacobian(x, exact=jacobian0 == "exact")
    if (why := _singular(B)) is not None:
        return result(False, f"initial {why}", 0)
    H = np.linalg.solve(B, np.eye(x.size))
    k = 0
    while True:
        if k == max_iter:
            return result(False, f"reached max_iter={max_iter}", k)
        p = -(H @ Fx)
        x_new = x + p
        F_new = sysm.residual(x_new)
        s = x_new - x  # the step actually taken (differs from p by rounding)
        y = F_new - Fx
        k += 1
        r_new = _norm(F_new)
        trace.append(
            Step(
                k,
                x_new.copy(),
                r_new,
                step_size=_norm(s),
                info={
                    "residual": F_new.copy(),
                    "residual_norm": r_new,
                    "step": s.copy(),
                    "jacobian": B.copy(),
                    "secant": {"s": s.copy(), "y": y.copy()},
                },
            )
        )
        stop = _check(x_new, F_new, p, x, x0_norm, ftol, xtol, secant=(s, y))
        x, Fx = x_new, F_new
        if stop is not None:
            return result(stop[0], stop[1], k)
        Hy = H @ y
        denom = float(s @ Hy)
        if not math.isfinite(denom) or abs(denom) <= EPS * float(_norm(s) * _norm(Hy)):
            return result(False, "Broyden update undefined: sᵀH y = 0 (H would become singular)", k)
        H = H + np.outer(s - Hy, s @ H) / denom
        B = B + np.outer(y - B @ s, s) / float(s @ s)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("newton_system", "circle_line", {}),
    ("newton_system", "rosenbrock_system", {}),
    ("newton_system", "trig_system", {"x0": [-1.0, 1.25]}),  # leaves the domain
    ("newton_system", "trig_system", {"x0": [-1.0, 1.25], "damping": True}),  # stays
    ("newton_system", "freudenstein_roth", {"damping": True}),  # stalls at the ‖F‖ trap
    ("broyden", "intersecting_circles", {}),
    ("broyden", "trig_system", {"jacobian0": "finite_difference"}),
    ("broyden", "circle_line", {}),
]
