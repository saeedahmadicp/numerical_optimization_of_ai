"""Constrained nonlinear optimization: min f(x) subject to c_i(x) ≤ 0 (i ∈ I), c_i(x) = 0 (i ∈ E).

Conventions shared by every method in this module:

* **Lagrangian and multipliers.** ``L(x, λ) = f(x) + Σ_i λ_i c_i(x)`` with ``λ_i ≥ 0`` for
  inequalities (the sign convention of Boyd & Vandenberghe; Nocedal & Wright write
  c_i ≥ 0 and L = f − λᵀc, which flips the sign of c_i, not of λ_i).
* **KKT residual** of a pair (x, λ), in the ∞-norm::

      stationarity    = ‖∇f(x) + Σ_i λ_i ∇c_i(x)‖∞
      kkt_residual    = max(stationarity, max_{i∈I} |λ_i c_i(x)|, max_{i∈I} max(0, −λ_i))
      violation       = max(max_{i∈I} max(0, c_i(x)), max_{i∈E} |c_i(x)|)

  The multiplier-based methods (quadratic_penalty, augmented_lagrangian, log_barrier,
  sqp) report this residual with their own multiplier estimate. The projection-based
  methods have a natural stationarity measure instead that is zero exactly at KKT points of
  a convex feasible set: projected_gradient uses ‖x − P(x − ∇f(x))‖∞ (Bertsekas 1999,
  Prop. 2.1.3) and frank_wolfe uses the Frank–Wolfe gap ∇f(x)ᵀ(x − s) (Jaggi 2013).
* **Stopping test (converged).** ``kkt_residual ≤ tol`` and ``violation ≤ tol`` at an
  iterate. Max-iter, line-search failure, non-finite values, an infeasible QP subproblem or
  a penalty parameter beyond its cap give ``converged=False`` with the reason.
* **Trace.** The outer loop (penalty / multiplier / barrier-parameter updates) and the inner
  iterations are flattened: Step ``k`` is the k-th accepted iterate of any loop, ``k = 0`` is
  the start, and ``max_iter`` bounds the total number of steps. An outer update that does
  not move x does not create a Step; the next Step shows the new outer parameters.
  ``Step.fun = f(x_k)`` (the objective, not the merit function), ``Step.grad_norm =
  ‖∇f(x_k)‖₂`` and ``Step.step_size`` is the accepted line-search step.
* **Counts.** ``n_fev``, ``n_gev``, ``n_hev`` count f, ∇f and ∇²f exactly (finite-difference
  fallbacks for missing derivatives count as the evaluations they make).
  ``Result.extra`` adds ``n_cev`` (constraint values: one per constraint function call),
  ``n_cgev`` (constraint gradients) and ``n_chev`` (constraint Hessians, log_barrier only;
  affine constraints are not evaluated because their Hessian is zero), plus the final
  ``multipliers`` (where the method has them), ``kkt_residual`` and ``violation``.
* **Feasible sets for projection methods.** ``Problem.extra["projection"]`` names an exact
  Euclidean projection: ``"box"`` (``extra["bounds"]``, clipping), ``"disk"``
  (``extra["disk"]``, radial scaling) or ``"polyhedron"`` (``extra["A_ub"], ["b_ub"],
  ["A_eq"], ["b_eq"]``, a strictly convex QP solved by :func:`solve_qp`). Frank–Wolfe also
  needs a compact set: a box, a disk or a polyhedron with ``extra["vertices"]``.

Info keys (every method, every step):
    outer: int              outer-iteration index (penalty stage, multiplier stage, barrier
                            centering stage); 0 for the single-loop methods
                            projected_gradient, frank_wolfe and sqp.
    inner: int              iteration index inside the current outer iteration (equals k for
                            the single-loop methods).
    constraints: [m]        c_i(x_k) in problem order.
    active: [int]           indices i with i ∈ E, or i ∈ I and c_i(x_k) ≥ −1e-6 (ACTIVE_TOL).
    violation: float        constraint violation at x_k (∞-norm, see above).
    kkt_residual: float     the method's stationarity measure at x_k (see above).

Keys that describe the move into x_k are absent at k = 0 ("from" etc.); keys that describe
the state at x_k are present at every step.

Additional info keys per method:
    projected_gradient:
        from: [n]           x_{k−1}.
        gradient: [n]       ∇f(x_{k−1}).
        unprojected: [n]    x_{k−1} − s_k ∇f(x_{k−1}) (the point that was projected).
        s: float            accepted step s_k = s̄ β^m of the Armijo rule along the arc.
        arc: [[n]...]       the projected trial points P(x_{k−1} − s ∇f(x_{k−1})), trial order.
        trials: [[s, f]]    every (s, f(P(x_{k−1} − s∇f))) evaluated, in order.
        projected_from: [n] (k = 0 only, when x0 was infeasible) the given x0 before projection.
    frank_wolfe:
        from: [n]           x_{k−1}.
        vertex: [n]         s_{k−1} = argmin_{s ∈ C} ∇f(x_{k−1})ᵀs (the LMO output).
        direction: [n]      s_{k−1} − x_{k−1}.
        gamma: float        the step γ_{k−1} ∈ (0, 1]: x_k = (1 − γ)x_{k−1} + γ s_{k−1}.
        trials: [[γ, f]]    every (γ, f) evaluated (one entry for the open-loop rule).
        lmo: [n]            s_k, the LMO vertex at x_k (defines the gap at x_k).
        gap: float          Frank–Wolfe gap ∇f(x_k)ᵀ(x_k − s_k) (= kkt_residual).
        projected_from: [n] (k = 0 only, when x0 was infeasible) the given x0.
    quadratic_penalty:
        mu: float           penalty parameter μ of the merit minimized to produce x_k.
        tau: float          inner tolerance τ = max(tol, 1/μ) on ‖∇Q(x; μ)‖∞.
        merit: float        Q(x_k; μ) = f + (μ/2)(Σ_E c_i² + Σ_I max(0, c_i)²).
        multipliers: [m]    λ_i = μ c_i (i ∈ E), μ max(0, c_i) (i ∈ I)  (N&W eq. 17.9).
        stationarity: float ‖∇f + Σ λ_i ∇c_i‖∞ = ‖∇Q(x_k; μ)‖∞.
        from, direction: [n]  x_{k−1} and the BFGS direction p.
        alpha: float        accepted step;  trials: [[α, Q]]  Armijo trials.
    augmented_lagrangian:
        mu: float           penalty parameter μ;  lambda: [m]  outer multipliers λ^k.
        omega, eta: float   inner tolerance ω_k and feasibility target η_k (N&W Alg. 17.4).
        update: str         last outer update: "start" | "multipliers" | "penalty".
        merit: float        L_A(x_k; λ^k, μ) (PHR form, see :func:`augmented_lagrangian`).
        multipliers: [m]    first-order estimate λ̃(x_k): λ_i + μc_i (E), max(0, λ_i + μc_i) (I).
        stationarity: float ‖∇_x L_A(x_k)‖∞.
        from, direction, alpha, trials: as for quadratic_penalty (trials hold L_A).
    log_barrier:
        t: float            barrier parameter t of the centering problem min t f + φ.
        gap: float          duality-gap bound |I|/t (B&V eq. 11.13).
        merit: float        t f(x_k) − Σ_I log(−c_i(x_k)).
        multipliers: [m]    λ_i = −1/(t c_i) (i ∈ I, B&V eq. 11.10); least-squares ν (i ∈ E).
        stationarity: float ‖∇f + Σ λ_i ∇c_i‖∞.
        from, direction: [n]  x_{k−1} and the Newton step Δx.
        alpha: float        accepted step;  trials: [[s, t f + φ]] ("inf" outside the domain).
        newton_decrement: float  λ(x_{k−1})²/2 = ΔxᵀHΔx/2, compared with newton_tol.
        hessian_shift: float  τ added to the barrier Hessian to make it positive definite
                            (0 for convex problems; N&W Alg. 3.3).
        projected_from: [n] (k = 0 only) x0 before projection onto the affine equalities.
    sqp:
        mu: float           ℓ1 merit penalty μ_k (N&W eq. 18.36).
        merit: float        φ₁(x_k; μ) = f + μ(Σ_E |c_i| + Σ_I max(0, c_i)).
        multipliers: [m]    λ_k: the QP multipliers λ̂ of the step into x_k (0 at k = 0).
        stationarity: float ‖∇f(x_k) + Σ λ_{k,i} ∇c_i(x_k)‖∞.
        from, direction: [n]  x_{k−1} and the QP step p.
        alpha: float        accepted step;  trials: [[α, φ₁]].
        working_set: [int]  constraints active in the QP solution at x_{k−1} (E always).
        theta: float        Powell damping factor θ of the BFGS update (1 = undamped,
                            null when the update was skipped because s = 0).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, Vector, as_vector

Matrix = NDArray[np.float64]

#: An inequality with c_i(x) ≥ −ACTIVE_TOL is reported as active in ``info["active"]``.
ACTIVE_TOL = 1e-6
#: Armijo sufficient-decrease constant c₁ (N&W §3.1) for the inner line searches.
_C1 = 1e-4
#: Backtracking factor for the inner line searches.
_SHRINK = 0.5
#: Maximum number of trial steps of one line search (0.5⁶⁰ ≈ 1e-18 of the first trial).
_MAX_TRIALS = 60
#: Smallest trial step of projected_gradient relative to s̄; the trial cap depends on β.
_MIN_TRIAL_RATIO = 1e-18
#: Safety cap on outer iterations that take no inner step (see quadratic_penalty).
_MAX_OUTER = 200
#: Relative test for linear dependence of a new constraint normal in :func:`solve_qp`.
_QP_DEP_TOL = 1e-10
#: Relative feasibility tolerance of :func:`solve_qp`.
_QP_FEAS_TOL = 1e-12
#: Safety factor on the first-order rounding bound (n + q + 2)·ε·scale of the consistency
#: residual of a dependent constraint in :func:`solve_qp` (Higham 2002, §3.1).
_QP_ROUND_FACTOR = 10.0
#: Machine epsilon of float64.
EPS = float(np.finfo(np.float64).eps)
#: Relative resolution of a merit value: changes below 100ε·max(1, |Φ|) are rounding noise.
_RESOLUTION = 100.0 * EPS


def _resolution(phi: float) -> float:
    """The smallest change of a merit value Φ that is not rounding noise."""
    return _RESOLUTION * max(1.0, abs(phi))


def _approx_armijo(slope_new: float, pred: float, c1: float) -> bool:
    """Armijo's test f(z) − f(x) ≤ −c₁·pred evaluated with gradients only.

    ``pred = −∇f(x)ᵀd > 0`` is the predicted decrease along the segment d = z − x and
    ``slope_new = ∇f(z)ᵀd``. The trapezoidal rule f(z) − f(x) ≈ ½(∇f(x) + ∇f(z))ᵀd (exact for
    a quadratic f) turns the test into ∇f(z)ᵀd ≤ (1 − 2c₁)·pred: the sufficient-decrease
    half (2δ − 1)φ'(0) ≥ φ'(1), δ = c₁, of the approximate Wolfe conditions of Hager & Zhang
    (2005), SIAM J. Optim. 16, 170–192, with φ(τ) = f(x + τd). The gradients carry no cancellation, so the test stays exact when
    f(x) − f(z) is below the resolution of f.
    """
    return slope_new <= (1.0 - 2.0 * c1) * pred  # False for NaN


# ======================================================================================
# Evaluation and KKT bookkeeping
# ======================================================================================


class _Model:
    """Counted evaluations of f, ∇f, ∇²f and of the constraints of a Problem."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.n = int(problem.dim)
        self.f = Counted(problem.f)
        self.g = Counted(problem.grad) if problem.grad is not None else None
        self.h = Counted(problem.hess) if problem.hess is not None else None
        self.cons = tuple(problem.constraints)
        self.m = len(self.cons)
        self.is_eq = np.array([c.kind == "eq" for c in self.cons], dtype=bool)
        self.is_in = ~self.is_eq
        affine = problem.extra.get("affine")
        self.affine = (
            tuple(bool(a) for a in affine)
            if affine is not None and len(affine) == self.m
            else (False,) * self.m
        )
        hess = problem.extra.get("constraint_hess")
        self._chess = tuple(hess) if hess is not None and len(hess) == self.m else None
        self.n_cev = 0
        self.n_cgev = 0
        self.n_chev = 0

    def fun(self, x: Vector) -> float:
        return float(self.f(x))

    def grad(self, x: Vector) -> Vector:
        if self.g is None:
            return diff.gradient(self.f, x)
        return as_vector(self.g(x))

    def hess(self, x: Vector) -> Matrix:
        if self.h is None:
            return diff.hessian(self.grad, x)
        return np.asarray(self.h(x), dtype=np.float64).reshape(self.n, self.n)

    def cval(self, x: Vector) -> Vector:
        self.n_cev += self.m
        return np.array([float(c.fun(x)) for c in self.cons], dtype=np.float64)

    def cjac(self, x: Vector) -> Matrix:
        self.n_cgev += self.m
        if self.m == 0:
            return np.zeros((0, self.n))
        return np.array([as_vector(c.grad(x)) for c in self.cons], dtype=np.float64)

    def chess(self, i: int, x: Vector) -> Matrix:
        """∇²c_i(x): from ``extra["constraint_hess"]`` or central differences of ∇c_i."""
        self.n_chev += 1
        if self._chess is not None:
            return np.asarray(self._chess[i](x), dtype=np.float64).reshape(self.n, self.n)
        grad_i = self.cons[i].grad

        def counted_grad(z: Vector) -> Vector:
            self.n_cgev += 1
            return as_vector(grad_i(z))

        return diff.hessian(counted_grad, x)

    def counts(self) -> tuple[int, int, int]:
        return (
            self.f.n,
            self.g.n if self.g is not None else 0,
            self.h.n if self.h is not None else 0,
        )


def _violation(c: Vector, is_eq: NDArray[np.bool_]) -> float:
    """max(max_I max(0, c_i), max_E |c_i|); 0 without constraints."""
    if c.size == 0:
        return 0.0
    return float(np.max(np.where(is_eq, np.abs(c), np.maximum(c, 0.0))))


def _active(c: Vector, is_eq: NDArray[np.bool_]) -> list[int]:
    return [i for i in range(c.size) if bool(is_eq[i]) or c[i] >= -ACTIVE_TOL]


def _kkt(
    gf: Vector, J: Matrix, lam: Vector, c: Vector, is_eq: NDArray[np.bool_]
) -> tuple[float, float]:
    """(stationarity, kkt_residual) of (x, λ); see the module docstring."""
    r = gf + J.T @ lam if lam.size else gf
    stat = float(np.max(np.abs(r))) if r.size else 0.0
    ineq = ~is_eq
    if not ineq.any():
        return stat, stat
    comp = float(np.max(np.abs(lam[ineq] * c[ineq])))
    dual = float(np.max(np.maximum(-lam[ineq], 0.0)))
    return stat, max(stat, comp, dual)


def _state(
    c: Vector, is_eq: NDArray[np.bool_], outer: int, inner: int, kkt: float
) -> dict[str, Any]:
    return {
        "outer": outer,
        "inner": inner,
        "constraints": c.tolist(),
        "active": _active(c, is_eq),
        "violation": _violation(c, is_eq),
        "kkt_residual": kkt,
    }


def _finish(
    method: str,
    model: _Model,
    x: Vector,
    fx: float,
    converged: bool,
    message: str,
    trace: list[Step],
    *,
    kkt: float,
    violation: float,
    multipliers: Vector | None = None,
) -> Result:
    n_fev, n_gev, n_hev = model.counts()
    extra: dict[str, Any] = {
        "n_cev": model.n_cev,
        "n_cgev": model.n_cgev,
        "n_chev": model.n_chev,
        "kkt_residual": kkt,
        "violation": violation,
    }
    if multipliers is not None:
        extra["multipliers"] = multipliers.tolist()
    return Result(
        method,
        x.copy(),
        fx,
        converged,
        message,
        trace[-1].k,
        n_fev,
        n_gev,
        n_hev,
        trace=trace,
        extra=extra,
    )


def _converged_msg(kkt: float, viol: float, tol: float) -> str:
    return f"KKT residual {kkt:.3g} ≤ tol and violation {viol:.3g} ≤ tol = {tol:.3g}"


def _nonfinite_msg(x: Vector) -> str:
    return f"non-finite function, constraint or derivative value at x = {x.tolist()!r}"


def _excess(kkt: float, viol: float, tol: float) -> str:
    """The failing stopping-test quantities, e.g. "violation 3e-07 > tol = 1e-08"."""
    failing = [
        f"{name} {val:.3g}"
        for name, val in (("KKT residual", kkt), ("violation", viol))
        if not val <= tol
    ]
    if not failing:  # defensive: the callers only report a failed test
        return f"KKT residual {kkt:.3g}, violation {viol:.3g}, tol = {tol:.3g}"
    return " and ".join(failing) + f" > tol = {tol:.3g}"


def _stalled_msg(kkt: float, viol: float, tol: float) -> str:
    return (
        f"stalled: the accepted step leaves x unchanged ({_excess(kkt, viol, tol)}); the "
        "merit function cannot resolve further progress in float64"
    )


def _nonfinite_trials_msg(kind: str, step: float, kkt: float) -> str:
    return (
        f"{kind}: f is non-finite at the trial points down to step {step:.3g} "
        f"(KKT residual {kkt:.3g})"
    )


def _null_trial_msg(kkt: float, viol: float, tol: float) -> str:
    return (
        f"stalled: backtracking shrank the step until the trial point equals x in float64 "
        f"({_excess(kkt, viol, tol)}); the decrease of f is below its rounding level"
    )


def _rounding_limit_msg(n_trials: int, step: float, kkt: float, viol: float, tol: float) -> str:
    return (
        f"stalled at the rounding level: after {n_trials} trials (step down to {step:.3g}) "
        f"the predicted decrease of f is below its resolution and the gradient form of the "
        f"Armijo test fails ({_excess(kkt, viol, tol)})"
    )


def _max_iter_msg(max_iter: int, kkt: float, viol: float) -> str:
    return f"reached max_iter={max_iter} (KKT residual {kkt:.3g}, violation {viol:.3g})"


def _tol_param(default: float) -> ParamSpec:
    return ParamSpec(
        "tol",
        default,
        min=1e-14,
        max=1e-2,
        log=True,
        help="Stop when the KKT residual and the constraint violation are both ≤ tol.",
    )


def _max_iter_param(default: int) -> ParamSpec:
    return ParamSpec(
        "max_iter",
        default,
        kind="int",
        min=1,
        max=100_000,
        help="Limit on the total number of iterations (inner and outer steps together).",
    )


# ======================================================================================
# Strictly convex QP: Goldfarb–Idnani dual active-set method
# ======================================================================================


@dataclass(frozen=True)
class QPResult:
    """Solution of min ½xᵀGx + aᵀx s.t. A_eq x = b_eq, A_ub x ≤ b_ub (see :func:`solve_qp`).

    Multipliers follow ``Gx + a + A_eqᵀ lam_eq + A_ubᵀ lam_ub = 0`` with ``lam_ub ≥ 0``.
    ``active`` lists the inequality rows in the final active set.
    """

    ok: bool
    x: Vector
    lam_eq: Vector
    lam_ub: Vector
    active: tuple[int, ...]
    n_iter: int
    message: str


def _gi_directions(L: Matrix, N: Matrix, n_p: Vector) -> tuple[Vector, Vector, bool]:
    """Primal step z = H n⁺ and dual step r = N* n⁺ of Goldfarb & Idnani (1983), §3.

    With G = LLᵀ and the active normals N (n × q): B = L⁻¹N = QR (reduced QR),
    H = L⁻ᵀ(I − QQᵀ)L⁻¹ and N* = R⁻¹QᵀL⁻¹, so with w = L⁻¹n⁺:
    r = R⁻¹Qᵀw and z = L⁻ᵀ(w − QQᵀw). ``dependent`` is True when n⁺ lies in span(N)
    (then z = 0).
    """
    w = np.linalg.solve(L, n_p)
    if N.shape[1] == 0:
        w_perp = w
        r = np.zeros(0)
    else:
        Q, R = np.linalg.qr(np.linalg.solve(L, N))
        qtw = Q.T @ w
        r = np.linalg.solve(R, qtw)
        w_perp = w - Q @ qtw
    norm_w = float(np.linalg.norm(w))
    dependent = norm_w == 0.0 or float(np.linalg.norm(w_perp)) <= _QP_DEP_TOL * norm_w
    z = np.zeros_like(w) if dependent else np.linalg.solve(L.T, w_perp)
    return np.asarray(z, dtype=np.float64), np.asarray(r, dtype=np.float64), dependent


def _dependent_is_consistent(
    n_p: Vector, b_p: float, N: Matrix, b_N: Vector, r: Vector, x: Vector
) -> bool:
    """Is a constraint n_pᵀx ≥ b_p with n_p ≈ N r consistent with the active rows Nᵀx = b_N?

    The slack s_p = n_pᵀx − b_p of a dependent constraint carries the rounding error of the
    whole iterate history (it grows like ε·κ(G)·max‖x‖), so |s_p| alone cannot decide
    consistency. The combination

        δ = s_p − rᵀs_N = (n_p − N r)ᵀx − (b_p − rᵀb_N),      s_N = Nᵀx − b_N,

    cancels that error: for a consistent constraint b_p = rᵀb_N and δ is the dependency
    defect (n_p − N r)ᵀx plus the rounding of the dot products, at most (n + q + 2)·u·scale
    with scale = |b_p| + ‖n_p‖‖x‖ + Σ_j |r_j|(|b_j| + ‖n_j‖‖x‖) (Higham 2002, §3.1:
    |fl(aᵀb) − aᵀb| ≤ γ_n|a|ᵀ|b|). An inconsistent constraint has |δ| ≈ |b_p − rᵀb_N|.

    Consistent when min(|s_p|, |δ|) ≤ 1e-12·(1 + |b_p| + ‖n_p‖‖x‖) + 10(n + q + 2)ε·scale
    + ‖n_p − N r‖‖x‖. The first term is the solver's feasibility tolerance: data that are
    consistent only up to their own (absolute) rounding error, e.g. linearizations
    J_E p = −c_E of duplicated equalities near feasibility, stay consistent.
    """
    q = b_N.size
    n = x.size
    xn = float(np.linalg.norm(x))
    s_p = float(n_p @ x - b_p)
    s_N = N.T @ x - b_N if q else np.zeros(0)
    delta = s_p - float(r @ s_N)
    col_norms = np.linalg.norm(N, axis=0) if q else np.zeros(0)
    n_p_norm = float(np.linalg.norm(n_p))
    scale = abs(b_p) + n_p_norm * xn + float(np.abs(r) @ (np.abs(b_N) + col_norms * xn))
    defect = float(np.linalg.norm(n_p - N @ r)) if q else n_p_norm
    tol = (
        _QP_FEAS_TOL * (1.0 + abs(b_p) + n_p_norm * xn)
        + _QP_ROUND_FACTOR * (n + q + 2) * EPS * scale
        + defect * xn
    )
    return min(abs(s_p), abs(delta)) <= tol


def solve_qp(
    G: ArrayLike,
    a: ArrayLike,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
) -> QPResult:
    """Strictly convex QP by the dual active-set method of Goldfarb & Idnani (1983).

    Solves ``min ½xᵀGx + aᵀx`` s.t. ``A_eq x = b_eq``, ``A_ub x ≤ b_ub`` with G symmetric
    positive definite. The method starts from the unconstrained minimizer −G⁻¹a (which is
    dual feasible), so it needs no feasible starting point — the reason it is used here for
    both the SQP subproblem and the Euclidean projection onto a polyhedron (G = I).

    Each major step adds a violated constraint p (first every equality, then the most
    violated inequality) and moves along the primal direction z = Hn⁺ and the dual direction
    (−r, 1) by t = min(t₁, t₂): t₂ = −s_p(x)/zᵀn⁺ makes constraint p active (full step);
    t₁ = min_{j active, r_j > 0} u_j/r_j is the largest step that keeps the active
    inequality multipliers u ≥ 0 (partial step: constraint j is dropped and the step
    repeated). If z = 0 and no multiplier limits the step, the constraints are inconsistent.
    Equality constraints are oriented so that their slack is ≤ 0 when added and are never
    dropped. Every major step strictly increases the dual objective, so the method ends
    after finitely many steps (Goldfarb & Idnani 1983, Thm. 2).

    A violated constraint whose normal lies in the span of the active normals (z = 0) and
    that no active inequality multiplier can release is inconsistent in exact arithmetic
    only if b_p ≠ rᵀb_N. It is skipped as redundant when the rounding-free consistency
    residual |s_p − rᵀs_N| is within the dot-product rounding bound (see
    :func:`_dependent_is_consistent`), and reported as inconsistent otherwise.

    Stops (ok) when every constraint that was not skipped as redundant is violated by at
    most 1e-12·(1 + |b_i| + ‖a_i‖‖x‖); a skipped one is satisfied to rounding accuracy.
    """
    G_m = np.asarray(G, dtype=np.float64)
    a_v = np.asarray(a, dtype=np.float64).reshape(-1)
    n = a_v.size
    Ae = np.zeros((0, n)) if A_eq is None else np.asarray(A_eq, dtype=np.float64).reshape(-1, n)
    be = np.zeros(0) if b_eq is None else np.asarray(b_eq, dtype=np.float64).reshape(-1)
    Ai = np.zeros((0, n)) if A_ub is None else np.asarray(A_ub, dtype=np.float64).reshape(-1, n)
    bi = np.zeros(0) if b_ub is None else np.asarray(b_ub, dtype=np.float64).reshape(-1)
    me, mi = be.size, bi.size
    if Ae.shape[0] != me or Ai.shape[0] != mi:
        raise ValueError("solve_qp: constraint matrices and right-hand sides disagree in size")

    def fail(x: Vector, it: int, msg: str) -> QPResult:
        return QPResult(False, x, np.zeros(me), np.zeros(mi), (), it, msg)

    try:
        L = np.asarray(np.linalg.cholesky(G_m), dtype=np.float64)
    except np.linalg.LinAlgError:
        return fail(np.zeros(n), 0, "the QP Hessian is not positive definite")
    x = np.asarray(np.linalg.solve(L.T, np.linalg.solve(L, -a_v)), dtype=np.float64)
    # G–I form: n_iᵀx ≥ b_i. Inequality A_ub[j]x ≤ b_ub[j] ↦ n = −A_ub[j], b = −b_ub[j];
    # equality i is stored with an orientation sign σ_i: n = σ_i A_eq[i], b = σ_i b_eq[i].
    act: list[int] = []  # constraint ids: i < me equality i, me + j inequality j
    sgn: list[float] = []
    u = np.zeros(0)
    skipped: set[int] = set()
    row_norm_i = np.linalg.norm(Ai, axis=1) if mi else np.zeros(0)
    cap = 10 * (n + me + mi) + 10
    it = 0

    def normal(cid: int, s: float) -> tuple[Vector, float]:
        if cid < me:
            return s * Ae[cid], s * be[cid]
        return -Ai[cid - me], -bi[cid - me]

    while True:
        # Step 1: choose a violated constraint p (equalities first, in order).
        p = -1
        sp = 1.0
        xnorm = float(np.linalg.norm(x))
        for i in range(me):
            if i not in act and i not in skipped:
                p = i
                sp = -1.0 if Ae[i] @ x - be[i] > 0.0 else 1.0
                break
        if p < 0 and mi:
            slack = bi - Ai @ x  # ≥ 0 when feasible
            tol_i = _QP_FEAS_TOL * (1.0 + np.abs(bi) + row_norm_i * xnorm)
            worst = math.inf
            for j in range(mi):
                cid = me + j
                if (
                    cid not in act
                    and cid not in skipped
                    and slack[j] < -tol_i[j]
                    and slack[j] < worst
                ):
                    p, worst = cid, float(slack[j])
        if p < 0:
            break
        n_p, b_p = normal(p, sp)
        u_plus = np.append(u, 0.0)
        # State before p; a skip of p restores it (see below).
        saved = (list(act), list(sgn), set(skipped))
        x_moved = False
        # Step 2: move until constraint p is active (full step) or blocked (partial step).
        while True:
            it += 1
            if it > cap:
                return fail(x, it, f"QP iteration limit ({cap}) reached")
            pairs = [normal(cid, s) for cid, s in zip(act, sgn, strict=True)]
            N = np.column_stack([nv for nv, _ in pairs]) if act else np.zeros((n, 0))
            b_N = np.array([bv for _, bv in pairs], dtype=np.float64)
            z, r, dependent = _gi_directions(L, N, n_p)
            s_p = float(n_p @ x - b_p)
            t1, k_drop = math.inf, -1
            for idx, cid in enumerate(act):
                if cid >= me and r[idx] > 0.0:
                    ratio = float(u_plus[idx] / r[idx])
                    if ratio < t1:
                        t1, k_drop = ratio, idx
            if (
                dependent
                and math.isinf(t1)
                and not x_moved
                and _dependent_is_consistent(n_p, b_p, N, b_N, r, x)
            ):
                # n_p ∈ span(N) and no active multiplier can be released: in exact arithmetic
                # s_p = rᵀs_N = 0, so p is implied by the active rows and its "violation" is
                # rounding. # NOTE: G–I declare the QP infeasible here; the test on s_p alone
                # reported consistent QPs with cond(G) ≳ 1e3 as infeasible (see
                # _dependent_is_consistent). Only dual steps (x fixed) can precede this
                # point, and they drop rows of the state before p, whose (x, u) is optimal for
                # its active set; that state is restored and p is skipped: an equality for
                # good (equalities are added while every active row is an equality, and none
                # is dropped), an inequality until the next drop.
                act, sgn, skipped = list(saved[0]), list(saved[1]), set(saved[2])
                skipped.add(p)
                break
            t2 = math.inf if dependent else -s_p / float(z @ n_p)
            t = min(t1, t2)
            if not math.isfinite(t):
                return fail(x, it, "the QP constraints are inconsistent (no feasible point)")
            if math.isinf(t2):  # dual step only: drop the blocking inequality
                u_plus[:-1] -= t * r
                u_plus[-1] += t
                del act[k_drop], sgn[k_drop]
                u_plus = np.delete(u_plus, k_drop)
                skipped = {cid for cid in skipped if cid < me}  # implied rows may change
                continue
            x = x + t * z
            x_moved = True
            u_plus[:-1] -= t * r
            u_plus[-1] += t
            if t2 <= t1:  # full step: p joins the active set
                act.append(p)
                sgn.append(sp)
                u = u_plus
                break
            del act[k_drop], sgn[k_drop]  # partial step
            u_plus = np.delete(u_plus, k_drop)
            skipped = {cid for cid in skipped if cid < me}

    lam_eq = np.zeros(me)
    lam_ub = np.zeros(mi)
    for cid, s, ui in zip(act, sgn, u, strict=True):
        if cid < me:
            lam_eq[cid] = -s * ui  # Gx + a = Σ u n ⇒ Gx + a + A_eqᵀ(−σu) = 0
        else:
            lam_ub[cid - me] = ui
    active = tuple(sorted(cid - me for cid in act if cid >= me))
    return QPResult(True, x, lam_eq, lam_ub, active, it, "optimal")


# ======================================================================================
# Simple feasible sets: projection and linear-minimization oracle
# ======================================================================================


class _ProjectionError(RuntimeError):
    pass


class _FeasibleSet:
    """Exact Euclidean projection P_C and linear-minimization oracle of a simple convex set."""

    def __init__(self, problem: Problem, method: str, *, compact: bool = False) -> None:
        extra = problem.extra
        kind = extra.get("projection")
        if kind not in ("box", "disk", "polyhedron"):
            raise ValueError(
                f"{method} needs a convex feasible set with an exact projection "
                f"(Problem.extra['projection'] = 'box' | 'disk' | 'polyhedron'); "
                f"problem {problem.id!r} has none"
            )
        self.kind: str = str(kind)
        n = int(problem.dim)
        self.vertices: Matrix | None = None
        if "vertices" in extra:
            self.vertices = np.asarray(extra["vertices"], dtype=np.float64).reshape(-1, n)
        if self.kind == "box":
            bounds = np.asarray(extra["bounds"], dtype=np.float64).reshape(n, 2)
            self.lo, self.hi = bounds[:, 0].copy(), bounds[:, 1].copy()
        elif self.kind == "disk":
            disk = extra["disk"]
            self.center = np.asarray(disk["center"], dtype=np.float64).reshape(n)
            self.radius = float(disk["radius"])
        else:
            self.A_ub = np.asarray(extra.get("A_ub", []), dtype=np.float64).reshape(-1, n)
            self.b_ub = np.asarray(extra.get("b_ub", []), dtype=np.float64).reshape(-1)
            self.A_eq = np.asarray(extra.get("A_eq", []), dtype=np.float64).reshape(-1, n)
            self.b_eq = np.asarray(extra.get("b_eq", []), dtype=np.float64).reshape(-1)
            self._eye = np.eye(n)
            if compact and self.vertices is None:
                raise ValueError(
                    f"{method} needs a compact feasible set; problem {problem.id!r} is a "
                    "polyhedron without extra['vertices'] (it may be unbounded)"
                )

    def project(self, z: Vector) -> Vector:
        if self.kind == "box":
            return np.clip(z, self.lo, self.hi)
        if self.kind == "disk":
            d = z - self.center
            nd = float(np.linalg.norm(d))
            if nd <= self.radius:
                return z.copy()
            return self.center + d * (self.radius / nd)
        qp = solve_qp(self._eye, -z, self.A_eq, self.b_eq, self.A_ub, self.b_ub)
        if not qp.ok:
            raise _ProjectionError(f"projection onto the polyhedron failed: {qp.message}")
        return qp.x

    def lmo(self, g: Vector, x: Vector) -> Vector:
        """argmin_{s ∈ C} gᵀs (ties: lower bound / first vertex; g = 0 returns x)."""
        if self.kind == "box":
            return np.where(g > 0.0, self.lo, np.where(g < 0.0, self.hi, x))
        if self.kind == "disk":
            ng = float(np.linalg.norm(g))
            return x.copy() if ng == 0.0 else self.center - (self.radius / ng) * g
        assert self.vertices is not None
        return self.vertices[int(np.argmin(self.vertices @ g))].copy()


def _project_start(C: _FeasibleSet, x0: Vector) -> Vector:
    try:
        return C.project(x0)
    except _ProjectionError as exc:
        raise ValueError(f"cannot project x0 onto the feasible set: {exc}") from None


# ======================================================================================
# Projected gradient
# ======================================================================================


@register(
    id="projected_gradient",
    family="constrained",
    name="Projected gradient",
    params=(
        ParamSpec(
            "s_bar",
            1.0,
            min=1e-4,
            max=1e2,
            log=True,
            help="First trial step s̄ of the Armijo rule along the projection arc.",
        ),
        ParamSpec("beta", 0.5, min=0.1, max=0.9, help="Backtracking factor β: s ← βs."),
        ParamSpec(
            "sigma",
            1e-4,
            min=1e-6,
            max=0.5,
            log=True,
            help="Sufficient-decrease constant σ of the Armijo rule.",
        ),
        _tol_param(1e-6),
        _max_iter_param(1000),
    ),
    needs=("f", "grad", "projection"),
    order="linear",
    summary="Take a gradient step, project it back onto the feasible set, and backtrack "
    "along the projection arc until f decreases enough.",
    references=(
        "Bertsekas (1999), Nonlinear Programming, 2nd ed., §2.3.1 (Armijo rule along the "
        "projection arc, eq. 2.43)",
        "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., §16.7",
    ),
)
def projected_gradient(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    s_bar: float = 1.0,
    beta: float = 0.5,
    sigma: float = 1e-4,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Gradient projection with the Armijo rule along the projection arc (Bertsekas 1999, §2.3.1).

    Iteration: x_k(s) = P_C(x_k − s∇f(x_k)); accept the first s = s̄βᵐ (m = 0, 1, ...) with

        f(x_k) − f(x_k(s)) ≥ σ ∇f(x_k)ᵀ(x_k − x_k(s)),

    and set x_{k+1} = x_k(s). The trial points trace the "projection arc" — a piecewise
    smooth path on the boundary of C when the gradient step leaves C. The right-hand side
    is ≥ σ‖x_k − x_k(s)‖²/s by the projection theorem, so every accepted step decreases f.

    C must be a box, a disk or a polyhedron (``Problem.extra["projection"]``; the polyhedral
    projection is the QP min ½‖y − z‖² solved by :func:`solve_qp`). An infeasible x0 is
    projected first (# NOTE: Bertsekas assumes x0 ∈ C).

    # NOTE: when the predicted decrease ∇f(x_k)ᵀ(x_k − x_k(s)) of a trial is below the
    # resolution 100ε·max(1, |f(x_k)|) of f, the comparison of f values is rounding noise:
    # backtracking would shrink s until x_k(s) = x_k and accept that null step. Such a trial
    # is instead accepted when f rises by less than the resolution and the gradient form of
    # the Armijo test, ∇f(x_k(s))ᵀd ≤ (1 − 2σ)∇f(x_k)ᵀ(x_k − x_k(s)), d = x_k(s) − x_k,
    # holds (approximate Wolfe, Hager & Zhang 2005; see :func:`_approx_armijo`). It costs one gradient
    # per such trial (reused when the trial is accepted). The first trial s̄ is not scaled
    # to the curvature, so the plain acceptance of a noise-level increase used by the
    # quasi-Newton line searches of this module would accept unstable steps s > 2/L here.
    # A trial point equal to x_k in float64 stops the method as stalled. The search tries at
    # most 1 + ⌈log(1e-18)/log β⌉ steps (down to s̄·1e-18). Even the gradient test cannot
    # certify descent once the decrease per step (≈ residual²/L) is below ‖∇f‖·ε·‖x‖, the
    # change of f caused by the rounding of the projected point itself; this monotone method
    # then stops "stalled" (a residual floor ≈ 3e-8 on linear_eq_quadratic).

    Stops (converged) when ‖x_k − P_C(x_k − ∇f(x_k))‖∞ ≤ tol (for convex C this is zero
    exactly at stationary points, Bertsekas Prop. 2.1.3) and the violation is ≤ tol.
    """
    method = "projected_gradient"
    prob = vector_problem(problem, x0=x0)
    if not (s_bar > 0.0 and 0.0 < beta < 1.0 and 0.0 < sigma < 1.0):
        raise ValueError("need s_bar > 0, 0 < beta < 1 and 0 < sigma < 1")
    max_trials = 1 + math.ceil(math.log(_MIN_TRIAL_RATIO) / math.log(beta))
    model = _Model(prob)
    C = _FeasibleSet(prob, method)
    x_given = start_point(prob, x0)
    x = _project_start(C, x_given)
    fx = model.fun(x)
    g = model.grad(x)
    c = model.cval(x)
    info0: dict[str, Any] = {}
    if not np.array_equal(x, x_given):
        info0["projected_from"] = x_given.tolist()
    if not finite(fx, g, c):
        trace = [Step(0, x.copy(), fx, info={**_state(c, model.is_eq, 0, 0, math.nan), **info0})]
        return _finish(
            method, model, x, fx, False, _nonfinite_msg(x), trace, kkt=math.nan, violation=math.nan
        )
    try:
        x_unit = C.project(x - g)
    except _ProjectionError as exc:
        trace = [Step(0, x.copy(), fx, info={**_state(c, model.is_eq, 0, 0, math.nan), **info0})]
        return _finish(method, model, x, fx, False, str(exc), trace, kkt=math.nan, violation=0.0)
    res = float(np.max(np.abs(x - x_unit)))
    viol = _violation(c, model.is_eq)
    trace = [
        Step(
            0,
            x.copy(),
            fx,
            float(np.linalg.norm(g)),
            info={**_state(c, model.is_eq, 0, 0, res), **info0},
        )
    ]
    k = 0
    while True:
        if res <= tol and viol <= tol:
            return _finish(
                method, model, x, fx, True, _converged_msg(res, viol, tol), trace,
                kkt=res, violation=viol,
            )  # fmt: skip
        if k == max_iter:
            return _finish(
                method, model, x, fx, False, _max_iter_msg(max_iter, res, viol), trace,
                kkt=res, violation=viol,
            )  # fmt: skip
        # On every failure below, the last accepted iterate (x, fx) is returned: it is
        # trace[-1], and (res, viol) describe it.
        res_f = _resolution(fx)
        s = s_bar
        trials: list[list[float]] = []
        arc: list[list[float]] = []
        accepted = stalled = rounding = False
        z, fz = x, fx
        gz: Vector | None = None  # ∇f(z), evaluated only in the rounding regime
        try:
            for _ in range(max_trials):
                z = x_unit if s == 1.0 else C.project(x - s * g)
                if np.array_equal(z, x):  # s is below the spacing of the floats at x
                    stalled = True
                    break
                fz = model.fun(z)
                trials.append([s, fz])
                arc.append(z.tolist())
                pred = float(g @ (x - z))  # ≥ ‖x − z‖²/s > 0 (projection theorem)
                gz = None
                rounding = pred <= res_f and math.isfinite(fz)
                if not rounding:
                    if fx - fz >= sigma * pred:  # False for NaN
                        accepted = True
                        break
                elif fz <= fx + res_f:  # rounding regime: the gradient test decides (NOTE)
                    gz = model.grad(z)
                    if _approx_armijo(float(gz @ (z - x)), pred, sigma):
                        accepted = True
                        break
                s *= beta
            if stalled or not accepted:
                if not math.isfinite(fz):  # the last evaluated trial (s/β)
                    msg = _nonfinite_trials_msg(
                        "Armijo rule along the projection arc", s / beta, res
                    )
                elif stalled:
                    msg = _null_trial_msg(res, viol, tol)
                elif rounding:
                    msg = _rounding_limit_msg(max_trials, s / beta, res, viol, tol)
                else:
                    msg = (
                        f"Armijo rule along the projection arc: trial limit reached "
                        f"({max_trials} trials, s down to {s / beta:.3g}; KKT residual "
                        f"{res:.3g}); f may be non-finite along the arc or the gradient "
                        "inaccurate"
                    )
                return _finish(method, model, x, fx, False, msg, trace, kkt=res, violation=viol)
            g_new = gz if gz is not None else model.grad(z)
            c_new = model.cval(z)
            if not finite(fz, g_new, c_new):
                return _finish(
                    method, model, x, fx, False, _nonfinite_msg(z), trace, kkt=res, violation=viol
                )
            z_unit = C.project(z - g_new)
        except _ProjectionError as exc:
            return _finish(method, model, x, fx, False, str(exc), trace, kkt=res, violation=viol)
        x_prev, g_prev = x, g
        x, fx, g, c, x_unit = z, fz, g_new, c_new, z_unit
        res = float(np.max(np.abs(x - x_unit)))
        viol = _violation(c, model.is_eq)
        k += 1
        info = {
            **_state(c, model.is_eq, 0, k, res),
            "from": x_prev.tolist(),
            "gradient": g_prev.tolist(),
            "unprojected": (x_prev - s * g_prev).tolist(),
            "s": s,
            "arc": arc,
            "trials": trials,
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), s, info))


# ======================================================================================
# Frank–Wolfe (conditional gradient)
# ======================================================================================


@register(
    id="frank_wolfe",
    family="constrained",
    name="Frank–Wolfe (conditional gradient)",
    params=(
        ParamSpec(
            "step",
            "open_loop",
            kind="choice",
            choices=("open_loop", "armijo"),
            help="γ_k = 2/(k+2) (open loop) or Armijo backtracking from γ = 1.",
        ),
        _tol_param(1e-6),
        _max_iter_param(1000),
    ),
    needs=("f", "grad", "linear minimization oracle"),
    order="sublinear O(1/k)",
    summary="Minimize the linear model of f over the feasible set (a vertex) and move "
    "part of the way toward that vertex; no projection needed.",
    references=(
        "Frank & Wolfe (1956), Naval Res. Logist. Q. 3, 95–110",
        "Jaggi (2013), Revisiting Frank–Wolfe, ICML, Algorithm 1 and eq. (2)",
        "Bertsekas (1999), Nonlinear Programming, 2nd ed., §2.2 (conditional gradient)",
    ),
)
def frank_wolfe(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    step: Literal["open_loop", "armijo"] = "open_loop",
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Frank–Wolfe / conditional gradient method (Jaggi 2013, Algorithm 1).

    Iteration: s_k = argmin_{s ∈ C} ∇f(x_k)ᵀs (the linear-minimization oracle, LMO);
    x_{k+1} = (1 − γ_k)x_k + γ_k s_k with γ_k = 2/(k + 2) (open loop, k = 0, 1, ...), or
    with Armijo backtracking γ = 1, ½, ¼, ... until f(x_k + γd_k) ≤ f(x_k) − 10⁻⁴ γ G_k.
    Iterates stay in C because they are convex combinations of points of C.

    The Frank–Wolfe gap G_k = ∇f(x_k)ᵀ(x_k − s_k) ≥ f(x_k) − f* for convex f (Jaggi 2013,
    eq. 2) and is zero exactly at stationary points. With the open-loop rule
    f(x_k) − f* ≤ 2C_f/(k + 2) (Jaggi Thm. 1); when the solution lies on a face of a
    polytope the iterates zig-zag between vertices and the O(1/k) rate is sharp.

    C must be compact: a box (LMO: the bound opposite to each gradient sign), a disk (LMO:
    c − r∇f/‖∇f‖) or a polyhedron with ``extra["vertices"]`` (LMO: the best vertex). An
    infeasible x0 is projected onto C first (# NOTE: Frank–Wolfe itself needs x0 ∈ C).

    # NOTE: as in projected_gradient, an Armijo trial whose predicted decrease γG_k is below
    # the resolution 100ε·max(1, |f(x_k)|) of f is decided by the gradient form of the test,
    # γ∇f(x_k + γd_k)ᵀd_k ≤ (1 − 2·10⁻⁴)γG_k (approximate Wolfe, Hager & Zhang 2005; see
    # :func:`_approx_armijo`), when f rises by less than the resolution; a trial point equal
    # to x_k in float64 stops the method as stalled.

    Stops (converged) when G_k ≤ tol and the violation is ≤ tol.
    """
    method = "frank_wolfe"
    if step not in ("open_loop", "armijo"):
        raise ValueError(f"unknown step rule {step!r}; expected 'open_loop' or 'armijo'")
    prob = vector_problem(problem, x0=x0)
    model = _Model(prob)
    C = _FeasibleSet(prob, method, compact=True)
    x_given = start_point(prob, x0)
    x = _project_start(C, x_given)
    fx = model.fun(x)
    g = model.grad(x)
    c = model.cval(x)
    info0: dict[str, Any] = {}
    if not np.array_equal(x, x_given):
        info0["projected_from"] = x_given.tolist()
    if not finite(fx, g, c):
        trace = [Step(0, x.copy(), fx, info={**_state(c, model.is_eq, 0, 0, math.nan), **info0})]
        return _finish(
            method, model, x, fx, False, _nonfinite_msg(x), trace, kkt=math.nan, violation=math.nan
        )
    s = C.lmo(g, x)
    gap = float(g @ (x - s))
    viol = _violation(c, model.is_eq)
    info0 = {**_state(c, model.is_eq, 0, 0, gap), "lmo": s.tolist(), "gap": gap, **info0}
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), info=info0)]
    k = 0
    while True:
        if gap <= tol and viol <= tol:
            return _finish(
                method, model, x, fx, True, _converged_msg(gap, viol, tol), trace,
                kkt=gap, violation=viol,
            )  # fmt: skip
        if k == max_iter:
            return _finish(
                method, model, x, fx, False, _max_iter_msg(max_iter, gap, viol), trace,
                kkt=gap, violation=viol,
            )  # fmt: skip
        d = s - x
        trials: list[list[float]] = []
        g_new: Vector | None = None  # ∇f(x_new) when the rounding regime evaluated it
        if step == "open_loop":
            gamma = 2.0 / (k + 2.0)
            x_new = (1.0 - gamma) * x + gamma * s
            f_new = model.fun(x_new)
            trials.append([gamma, f_new])
        else:
            res_f = _resolution(fx)
            gamma = 1.0
            x_new, f_new = x, fx
            accepted = rounding = False
            for _ in range(_MAX_TRIALS):
                x_new = (1.0 - gamma) * x + gamma * s
                if np.array_equal(x_new, x):  # γ below the spacing of the floats at x
                    msg = (
                        _null_trial_msg(gap, viol, tol)
                        if math.isfinite(f_new)
                        else _nonfinite_trials_msg("Armijo backtracking on γ", 2.0 * gamma, gap)
                    )
                    return _finish(method, model, x, fx, False, msg, trace, kkt=gap, violation=viol)
                f_new = model.fun(x_new)
                trials.append([gamma, f_new])
                pred = gamma * gap  # predicted decrease −γ∇f(x)ᵀd > 0
                g_new = None
                rounding = pred <= res_f and math.isfinite(f_new)
                if not rounding:
                    if f_new <= fx - _C1 * pred:  # False for NaN
                        accepted = True
                        break
                elif f_new <= fx + res_f:  # rounding regime (see the NOTE above)
                    g_new = model.grad(x_new)
                    if _approx_armijo(gamma * float(g_new @ d), pred, _C1):
                        accepted = True
                        break
                gamma *= _SHRINK
            if not accepted:
                msg = (
                    _nonfinite_trials_msg("Armijo backtracking on γ", 2.0 * gamma, gap)
                    if not math.isfinite(f_new)
                    else _rounding_limit_msg(_MAX_TRIALS, 2.0 * gamma, gap, viol, tol)
                    if rounding
                    else (
                        f"Armijo backtracking on γ: trial limit reached ({_MAX_TRIALS} trials, "
                        f"γ down to {2.0 * gamma:.3g}; gap {gap:.3g}); f may be non-finite "
                        "along the segment or the gradient inaccurate"
                    )
                )
                return _finish(method, model, x, fx, False, msg, trace, kkt=gap, violation=viol)
        g_new = g_new if g_new is not None else model.grad(x_new)
        c_new = model.cval(x_new)
        if not finite(f_new, g_new, c_new):
            # The last accepted iterate (x, fx) = trace[-1] is returned.
            return _finish(
                method, model, x, fx, False, _nonfinite_msg(x_new), trace, kkt=gap, violation=viol
            )
        x_prev, s_prev = x, s
        x, fx, g, c = x_new, f_new, g_new, c_new
        s = C.lmo(g, x)
        gap = float(g @ (x - s))
        viol = _violation(c, model.is_eq)
        k += 1
        info = {
            **_state(c, model.is_eq, 0, k, gap),
            "from": x_prev.tolist(),
            "vertex": s_prev.tolist(),
            "direction": d.tolist(),
            "gamma": gamma,
            "trials": trials,
            "lmo": s.tolist(),
            "gap": gap,
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), gamma, info))


# ======================================================================================
# Sequential unconstrained minimization: quadratic penalty and augmented Lagrangian
# ======================================================================================


@dataclass
class _Point:
    """An iterate with its objective, constraint values and first derivatives."""

    x: Vector
    f: float
    c: Vector
    g: Vector
    J: Matrix


class _Policy:
    """Outer-loop rules of a sequential unconstrained minimization method."""

    is_eq: NDArray[np.bool_]

    def merit(self, f: float, c: Vector) -> float:
        raise NotImplementedError

    def weights(self, c: Vector) -> Vector:
        """Multiplier estimate w(x) with ∇merit = ∇f + Jᵀw."""
        raise NotImplementedError

    def inner_tol(self) -> float:
        raise NotImplementedError

    def after_inner(self, pt: _Point) -> str | None:
        """Update the outer parameters; return a failure message to stop."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        raise NotImplementedError


def _bfgs_update(H: Matrix, s: Vector, y: Vector, first: bool) -> tuple[Matrix, bool]:
    """Inverse BFGS update (N&W eq. 6.17) with the initial scaling (N&W eq. 6.20).

    # NOTE: the inner line search is Armijo backtracking, which does not guarantee sᵀy > 0;
    # the update is skipped when sᵀy ≤ 1e-12‖s‖‖y‖ (N&W §6.1, "skip the update").
    """
    sy = float(s @ y)
    if not sy > 1e-12 * float(np.linalg.norm(s)) * float(np.linalg.norm(y)):
        return H, False
    n = s.size
    if first:
        H = (sy / float(y @ y)) * np.eye(n)
    rho = 1.0 / sy
    V = np.eye(n) - rho * np.outer(s, y)
    return V @ H @ V.T + rho * np.outer(s, s), True


def _sequential(
    method: str,
    problem: Problem | Callable[..., float],
    x0: Any,
    tol: float,
    max_iter: int,
    make_policy: Callable[[_Model], _Policy],
) -> Result:
    """Shared driver: inner BFGS minimization of the policy's merit, outer updates between.

    Inner iteration (N&W Alg. 6.1 with Armijo backtracking): p = −H∇Φ, α from 1 (the first
    step after a reset tries α = min(1, 1/‖p‖₂), a unit-length step), halve α until
    Φ(x + αp) ≤ Φ(x) + 10⁻⁴α∇Φᵀp. The inverse Hessian H is reset to I at every outer
    iteration because the merit function changes. The inner loop stops when
    ‖∇Φ(x)‖∞ ≤ policy.inner_tol().

    # NOTE: near a solution with a large μ the predicted decrease −α∇Φᵀp of a trial can fall
    # below the resolution 100ε·max(1, |Φ|) of Φ (‖∇Φ‖ = 10⁻⁸ and ‖∇²Φ‖ = 10⁴ give 10⁻²⁰);
    # the Armijo comparison is then decided by rounding and the iteration stalls with tiny
    # steps. Such a trial is accepted when Φ increases by less than the resolution (cf. the
    # approximate Wolfe conditions of Hager & Zhang 2005, SIAM J. Optim. 16, 170–192, which
    # also tolerate an increase of f at the level of its error). An accepted step that
    # leaves x unchanged is an exact fixed point of the iteration (the BFGS update is
    # skipped for s = 0), so the method stops with converged=False instead of repeating it.

    Reported multipliers: ``lam`` is the estimate w(x) of the merit function that produced
    the current x (the last Step's ``multipliers``), or of the current merit when an outer
    update needed no inner step; ``kkt`` is always the residual of (x, lam). The weights of a
    merit with updated outer parameters are not an estimate at an x that minimized the old
    merit (for the penalty method ρμc(x) is off by the factor ρ, N&W eq. 17.9).
    """
    prob = vector_problem(problem, x0=x0)
    model = _Model(prob)
    policy = make_policy(model)
    is_eq = model.is_eq
    x = start_point(prob, x0)
    fx = model.fun(x)
    c = model.cval(x)
    g = model.grad(x)
    J = model.cjac(x)
    pt = _Point(x, fx, c, g, J)
    w = policy.weights(c) if finite(c) else np.full(model.m, math.nan)
    gphi = g + J.T @ w
    phi = policy.merit(fx, c)
    if not finite(fx, c, g, J, gphi):
        state = _state(c, is_eq, 0, 0, math.nan)
        trace = [Step(0, x.copy(), fx, info=state)]
        return _finish(
            method, model, x, fx, False, _nonfinite_msg(x), trace, kkt=math.nan, violation=math.nan
        )
    lam = w
    stat, kkt = _kkt(g, J, lam, c, is_eq)
    viol = _violation(c, is_eq)
    info0 = {
        **_state(c, is_eq, 0, 0, kkt),
        **policy.info(),
        "merit": phi,
        "multipliers": lam.tolist(),
        "stationarity": stat,
    }
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), info=info0)]

    def done(converged: bool, msg: str) -> Result:
        return _finish(
            method, model, pt.x, pt.f, converged, msg, trace,
            kkt=kkt, violation=viol, multipliers=lam,
        )  # fmt: skip

    if kkt <= tol and viol <= tol:
        return done(True, _converged_msg(kkt, viol, tol))
    k = 0
    outer = 0
    idle_outer = 0
    while True:
        tau = policy.inner_tol()
        H = np.eye(model.n)
        n_updates = 0
        inner = 0
        while float(np.max(np.abs(gphi))) > tau:
            if k == max_iter:
                return done(False, _max_iter_msg(max_iter, kkt, viol))
            p = -H @ gphi
            slope = float(gphi @ p)
            if not slope < 0.0:  # cannot happen for H ≻ 0; guard against rounding
                H, n_updates = np.eye(model.n), 0
                p, slope = -gphi, -float(gphi @ gphi)
            alpha = 1.0 if n_updates else min(1.0, 1.0 / float(np.linalg.norm(p)))
            trials: list[list[float]] = []
            accepted = False
            xt, ft, ct, phit = pt.x, pt.f, pt.c, phi
            for _ in range(_MAX_TRIALS):
                xt = pt.x + alpha * p
                ft = model.fun(xt)
                ct = model.cval(xt)
                phit = policy.merit(ft, ct) if finite(ft, ct) else math.inf
                trials.append([alpha, phit])
                if phit <= phi + _C1 * alpha * slope or (
                    -alpha * slope <= _resolution(phi) and phit <= phi + _resolution(phi)
                ):
                    accepted = True
                    break
                alpha *= _SHRINK
            if not accepted:
                return done(False, f"inner line search failed after {_MAX_TRIALS} trials")
            if np.array_equal(xt, pt.x):
                return done(False, _stalled_msg(kkt, viol, tol))
            gt = model.grad(xt)
            Jt = model.cjac(xt)
            wt = policy.weights(ct)
            gphi_t = gt + Jt.T @ wt
            if not finite(gt, Jt, gphi_t):
                return done(False, _nonfinite_msg(xt))
            H, updated = _bfgs_update(H, xt - pt.x, gphi_t - gphi, n_updates == 0)
            n_updates += int(updated)
            x_prev = pt.x
            pt, gphi, phi, lam = _Point(xt, ft, ct, gt, Jt), gphi_t, phit, wt
            k += 1
            inner += 1
            stat, kkt = _kkt(gt, Jt, lam, ct, is_eq)
            viol = _violation(ct, is_eq)
            info = {
                **_state(ct, is_eq, outer, inner, kkt),
                **policy.info(),
                "merit": phit,
                "multipliers": wt.tolist(),
                "stationarity": stat,
                "from": x_prev.tolist(),
                "direction": p.tolist(),
                "alpha": alpha,
                "trials": trials,
            }
            trace.append(Step(k, xt.copy(), ft, float(np.linalg.norm(gt)), alpha, info))
            if kkt <= tol and viol <= tol:
                return done(True, _converged_msg(kkt, viol, tol))
        if inner == 0:
            # The outer parameters changed without a new step: test the current point
            # with the multiplier estimate of the current merit (no Step is added).
            lam = policy.weights(pt.c)
            stat, kkt = _kkt(pt.g, pt.J, lam, pt.c, is_eq)
            if kkt <= tol and viol <= tol:
                return done(True, _converged_msg(kkt, viol, tol))
            idle_outer += 1
            if idle_outer > _MAX_OUTER:
                return done(False, f"{_MAX_OUTER} outer updates in a row without progress")
        else:
            idle_outer = 0
        msg = policy.after_inner(pt)
        if msg is not None:
            return done(False, msg)
        outer += 1
        # The new merit's gradient; the reported (lam, kkt) stay those of the merit that
        # produced pt.x until the next Step or idle test.
        gphi = pt.g + pt.J.T @ policy.weights(pt.c)
        phi = policy.merit(pt.f, pt.c)


class _PenaltyPolicy(_Policy):
    def __init__(self, model: _Model, mu0: float, rho: float, mu_max: float, tol: float) -> None:
        self.is_eq = model.is_eq
        self.mu = mu0
        self.rho = rho
        self.mu_max = mu_max
        self.tol = tol

    def _r(self, c: Vector) -> Vector:
        return np.where(self.is_eq, c, np.maximum(c, 0.0))

    def merit(self, f: float, c: Vector) -> float:
        r = self._r(c)
        return f + 0.5 * self.mu * float(r @ r)

    def weights(self, c: Vector) -> Vector:
        return self.mu * self._r(c)

    def inner_tol(self) -> float:
        return max(self.tol, 1.0 / self.mu)

    def after_inner(self, pt: _Point) -> str | None:
        if self.mu * self.rho > self.mu_max:
            return (
                f"penalty parameter would exceed mu_max = {self.mu_max:.3g} "
                f"(violation {_violation(pt.c, self.is_eq):.3g}; the problem may be infeasible "
                "or need a larger mu_max)"
            )
        self.mu *= self.rho
        return None

    def info(self) -> dict[str, Any]:
        return {"mu": self.mu, "tau": self.inner_tol()}


@register(
    id="quadratic_penalty",
    family="constrained",
    name="Quadratic penalty",
    params=(
        ParamSpec("mu0", 1.0, min=1e-3, max=1e4, log=True, help="Initial penalty parameter μ₀."),
        ParamSpec("rho", 10.0, min=1.5, max=100.0, log=True, help="Growth factor: μ ← ρμ."),
        ParamSpec(
            "mu_max",
            1e10,
            min=1e4,  # = the largest μ₀, so that no UI choice violates μ_max ≥ μ₀
            max=1e15,
            log=True,
            help="Give up when μ would exceed this value.",
        ),
        _tol_param(1e-6),
        _max_iter_param(1000),
    ),
    needs=("f", "grad", "constraints"),
    order="linear in the number of μ updates; violation O(1/μ)",
    summary="Replace the constraints by a quadratic penalty (μ/2)·violation², minimize, "
    "and repeat with a larger μ.",
    references=(
        "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Framework 17.1, eqs. 17.2 and 17.5",
        "Nocedal & Wright (2006), Algorithm 6.1 (BFGS, inner solver)",
    ),
)
def quadratic_penalty(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    mu0: float = 1.0,
    rho: float = 10.0,
    mu_max: float = 1e10,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Result:
    """Quadratic penalty method (N&W Framework 17.1).

    Outer iteration k minimizes

        Q(x; μ_k) = f(x) + (μ_k/2) [Σ_E c_i(x)² + Σ_I max(0, c_i(x))²]      (N&W 17.5)

    from the previous minimizer with BFGS (N&W Alg. 6.1, Armijo backtracking) until
    ‖∇Q(x; μ_k)‖∞ ≤ τ_k = max(tol, 1/μ_k) (Framework 17.1 only asks τ_k → 0), then sets
    μ_{k+1} = ρμ_k. The multiplier estimate λ_i = μ_k c_i (E), μ_k max(0, c_i) (I) makes
    ∇Q = ∇_x L(x, λ) (N&W eq. 17.9), so the KKT residual's stationarity part equals
    ‖∇Q‖∞. The minimizers are infeasible by O(λ*/μ), so the violation reaches tol only
    when μ ≳ ‖λ*‖/tol and the inner problems become ill-conditioned (N&W §17.1).

    Stops (converged) when the KKT residual and the violation are ≤ tol at an iterate.
    Fails when μ would exceed ``mu_max`` or after ``max_iter`` inner steps.
    """
    if not (mu0 > 0.0 and rho > 1.0 and mu_max >= mu0):
        raise ValueError("need mu0 > 0, rho > 1 and mu_max ≥ mu0")
    return _sequential(
        "quadratic_penalty",
        problem,
        x0,
        tol,
        max_iter,
        lambda model: _PenaltyPolicy(model, mu0, rho, mu_max, tol),
    )


class _ALPolicy(_Policy):
    def __init__(
        self, model: _Model, mu0: float, mu_factor: float, mu_max: float, tol: float
    ) -> None:
        self.is_eq = model.is_eq
        self.mu = mu0
        self.mu_factor = mu_factor
        self.mu_max = mu_max
        self.tol = tol
        self.lam = np.zeros(model.m)
        # N&W Alg. 17.4 initial tolerances: ω₀ = 1/μ₀, η₀ = 1/μ₀^0.1.
        self.omega = max(1.0 / mu0, tol)
        self.eta = 1.0 / mu0**0.1
        self.update = "start"

    def _shifted(self, c: Vector) -> Vector:
        return self.lam + self.mu * c

    def merit(self, f: float, c: Vector) -> float:
        lc = self._shifted(c)
        eq = self.is_eq
        val = f + float(self.lam[eq] @ c[eq]) + 0.5 * self.mu * float(c[eq] @ c[eq])
        ineq = ~eq
        plus = np.maximum(lc[ineq], 0.0)
        return val + float(plus @ plus - self.lam[ineq] @ self.lam[ineq]) / (2.0 * self.mu)

    def weights(self, c: Vector) -> Vector:
        lc = self._shifted(c)
        return np.where(self.is_eq, lc, np.maximum(lc, 0.0))

    def inner_tol(self) -> float:
        return self.omega

    def after_inner(self, pt: _Point) -> str | None:
        # Birgin & Martínez (2014) feasibility-complementarity measure V_i = max(c_i, −λ_i/μ)
        # for inequalities (|c_i| for equalities) replaces ‖c(x)‖ of N&W Alg. 17.4.
        V = np.where(self.is_eq, np.abs(pt.c), np.abs(np.maximum(pt.c, -self.lam / self.mu)))
        if V.size == 0 or float(np.max(V)) <= self.eta:
            self.lam = self.weights(pt.c)
            self.eta = self.eta / self.mu**0.9
            self.omega = max(self.omega / self.mu, self.tol)
            self.update = "multipliers"
            return None
        if self.mu * self.mu_factor > self.mu_max:
            return (
                f"penalty parameter would exceed mu_max = {self.mu_max:.3g} "
                f"(violation {_violation(pt.c, self.is_eq):.3g}; the problem may be infeasible)"
            )
        self.mu *= self.mu_factor
        self.eta = 1.0 / self.mu**0.1
        self.omega = max(1.0 / self.mu, self.tol)
        self.update = "penalty"
        return None

    def info(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "lambda": self.lam.tolist(),
            "omega": self.omega,
            "eta": self.eta,
            "update": self.update,
        }


@register(
    id="augmented_lagrangian",
    family="constrained",
    name="Augmented Lagrangian (method of multipliers)",
    params=(
        ParamSpec(
            "mu0",
            10.0,
            min=2.0,  # Alg. 17.4 needs μ₀ > 1: ω ← ω/μ and η ← η/μ^0.9 must tighten
            max=1e4,
            log=True,
            help="Initial penalty parameter μ₀ (> 1).",
        ),
        ParamSpec(
            "mu_factor",
            100.0,
            min=2.0,
            max=1000.0,
            log=True,
            help="Penalty growth μ ← factor·μ when the constraints did not improve enough.",
        ),
        ParamSpec(
            "mu_max",
            1e12,
            min=1e4,  # = the largest μ₀, so that no UI choice violates μ_max ≥ μ₀
            max=1e16,
            log=True,
            help="Give up when μ would exceed this.",
        ),
        _tol_param(1e-8),
        _max_iter_param(1000),
    ),
    needs=("f", "grad", "constraints"),
    order="linear in the outer iterations (rate → 0 as μ grows)",
    summary="Minimize the Lagrangian plus a quadratic penalty, then move the multipliers "
    "toward their optimal values; μ need not go to infinity.",
    references=(
        "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Framework 17.3 and "
        "Algorithm 17.4 (update rules for μ, ω, η)",
        "Rockafellar (1973), Math. Program. 5, 354–373 (PHR inequality form)",
        "Birgin & Martínez (2014), Practical Augmented Lagrangian Methods, SIAM, Alg. 4.1",
    ),
)
def augmented_lagrangian(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    mu0: float = 10.0,
    mu_factor: float = 100.0,
    mu_max: float = 1e12,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> Result:
    """Augmented Lagrangian method (N&W Framework 17.3 with the updates of Alg. 17.4).

    The merit function, in the Powell–Hestenes–Rockafellar form for inequalities:

        L_A(x; λ, μ) = f + Σ_E [λ_i c_i + (μ/2)c_i²] + (1/2μ) Σ_I [max(0, λ_i + μc_i)² − λ_i²]

    is C¹ with ∇L_A = ∇f + Σ λ̃_i ∇c_i, λ̃_i = λ_i + μc_i (E), max(0, λ_i + μc_i) (I).
    (# NOTE: N&W 2nd ed. write c ≥ 0 and L_A = f − λᵀc + (μ/2)‖c‖²; with c ≤ 0 the signs of
    the λ terms flip. N&W treat inequalities with slacks and bounds (Alg. 17.4 is the
    bound-constrained LANCELOT method); the PHR form is the equivalent unconstrained
    formulation obtained by minimizing out the slacks, N&W §17.4.)

    Outer iteration (N&W Alg. 17.4, μ₀ = 10, ω₀ = 1/μ₀, η₀ = 1/μ₀^0.1): minimize L_A with
    BFGS until ‖∇L_A‖∞ ≤ ω_k. If the measure ‖V‖∞, V_i = |c_i| (E) or
    |max(c_i, −λ_i/μ)| (I) (Birgin & Martínez 2014), is ≤ η_k: update the multipliers
    λ ← λ̃(x), η ← η/μ^0.9, ω ← ω/μ; otherwise keep λ and set μ ← factor·μ, η ← 1/μ^0.1,
    ω ← 1/μ. (# NOTE: ω is kept ≥ tol; a tighter inner tolerance is never needed.)
    μ₀ must exceed 1 (ValueError otherwise): μ only grows, and the updates ω ← ω/μ,
    η ← η/μ^0.9 tighten the tolerances only for μ > 1 (and slowly for μ near 1). With μ₀ < 1,
    ω₀ = 1/μ₀ exceeds ‖∇L_A(x₀)‖, x never moves, and every multiplier update loosens ω and η.

    Stops (converged) when the KKT residual of (x_k, λ̃(x_k)) and the violation are ≤ tol.
    Fails when μ would exceed ``mu_max`` or after ``max_iter`` inner steps.
    """
    if not (mu0 > 1.0 and mu_factor > 1.0 and mu_max >= mu0):
        raise ValueError("need mu0 > 1, mu_factor > 1 and mu_max ≥ mu0")
    return _sequential(
        "augmented_lagrangian",
        problem,
        x0,
        tol,
        max_iter,
        lambda model: _ALPolicy(model, mu0, mu_factor, mu_max, tol),
    )


# ======================================================================================
# Log-barrier interior-point method
# ======================================================================================

#: Backtracking constants of B&V Alg. 9.2 (the values of the examples in B&V §11.3.2).
_BARRIER_ALPHA = 0.01
_BARRIER_BETA = 0.5
#: N&W Alg. 3.3: β, the first shift added when the Cholesky factorization fails.
_SHIFT_BETA = 1e-3


def _make_pd(H: Matrix) -> tuple[Matrix, float] | None:
    """H + τI with the smallest τ of the sequence of N&W Alg. 3.3 that admits Cholesky."""
    dmin = float(np.min(np.diag(H)))
    tau = 0.0 if dmin > 0.0 else -dmin + _SHIFT_BETA
    eye = np.eye(H.shape[0])
    for _ in range(200):
        try:
            np.linalg.cholesky(H + tau * eye)
            return H + tau * eye, tau
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, _SHIFT_BETA)
    return None


def _solve_pd(Hm: Matrix, b: Vector) -> Vector:
    """Solve Hm·x = b for an Hm that passed the Cholesky test of _make_pd (LU first).

    # NOTE: near the boundary the barrier Hessian is Jᵀdiag(1/c²)J + tH with 1/c² ≈ 1e32, i.e.
    # positive definite only at the rounding level. Cholesky (in _make_pd) and LU then disagree
    # on some CPUs: LU meets an exactly zero pivot while Cholesky has positive pivots. Hm·x = b
    # is then solved with that Cholesky factor, Hm = LLᵀ (as solve_qp does).
    """
    try:
        return np.asarray(np.linalg.solve(Hm, b), dtype=np.float64)
    except np.linalg.LinAlgError:
        L = np.asarray(np.linalg.cholesky(Hm), dtype=np.float64)
        return np.asarray(np.linalg.solve(L.T, np.linalg.solve(L, b)), dtype=np.float64)


@register(
    id="log_barrier",
    family="constrained",
    name="Log-barrier interior-point method",
    params=(
        ParamSpec("t0", 1.0, min=1e-3, max=1e3, log=True, help="Initial barrier parameter t₀."),
        ParamSpec("mu", 10.0, min=1.5, max=100.0, log=True, help="Barrier growth: t ← μt."),
        _tol_param(1e-6),
        ParamSpec(
            "newton_tol",
            1e-10,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Centering ends when the Newton decrement λ²/2 ≤ newton_tol.",
        ),
        _max_iter_param(500),
    ),
    needs=("f", "grad", "hess", "constraints", "strictly feasible x0"),
    order="outer: gap |I|/t shrinks by μ per stage; inner: Newton (quadratic)",
    summary="Follow the central path: minimize t·f − Σ log(−c_i) with Newton's method "
    "for growing t, staying strictly inside the feasible set.",
    references=(
        "Boyd & Vandenberghe (2004), Convex Optimization, Algorithm 11.1 (barrier method)",
        "Boyd & Vandenberghe (2004), Algorithms 9.5 and 10.1 (Newton's method), 9.2 (backtracking)",
        "Nocedal & Wright (2006), Algorithm 3.3 (Cholesky with added multiple of the identity)",
    ),
)
def log_barrier(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    t0: float = 1.0,
    mu: float = 10.0,
    tol: float = 1e-6,
    newton_tol: float = 1e-10,
    max_iter: int = 500,
) -> Result:
    """Barrier method (Boyd & Vandenberghe, Alg. 11.1) with Newton centering.

    Given a strictly feasible x (c_i(x) < 0, i ∈ I; Ax = b for the affine equalities),
    t := t₀. Repeat: (1) centering — minimize t f(x) + φ(x), φ = −Σ_I log(−c_i(x)),
    subject to Ax = b by Newton's method from the current x (B&V Alg. 10.1): solve

        [∇²F  Aᵀ] [Δx]   [−∇F    ]
        [A    0 ] [w ] = [−(Ax−b)]      F = t f + φ,

    stop when λ²/2 = ΔxᵀHΔx/2 ≤ newton_tol (or at the rounding floor of F, see the NOTE in
    the code), else backtrack s = 1, β, β², ... first until x + sΔx is strictly feasible,
    then until F(x + sΔx) ≤ F(x) + αs∇FᵀΔx (B&V Alg. 9.2, α = 0.01, β = 0.5); (2) stopping
    test; (3) t := μt.

    ∇F = t∇f − Σ_I ∇c_i/c_i and ∇²F = t∇²f + Σ_I ∇c_i∇c_iᵀ/c_i² − Σ_I ∇²c_i/c_i.
    (# NOTE: B&V assume convex f and c_i; for nonconvex problems ∇²F may be indefinite and
    is replaced by ∇²F + τI with τ from N&W Alg. 3.3, reported as ``hessian_shift``.)

    The central point x*(t) gives dual-feasible multipliers λ_i = −1/(t c_i) with duality
    gap |I|/t (B&V eqs. 11.10–11.13); ν for the equalities is the least-squares estimate.
    (# NOTE: B&V stop when |I|/t < ε; the uniform test below includes the complementarity
    |λ_i c_i| = 1/t ≤ tol, i.e. ε = |I|·tol.)

    Equality constraints must be affine (B&V §11.1): every equality i must be declared
    affine by ``Problem.extra["affine"][i] = True`` (as the problem library does), else
    ValueError — with a nonlinear equality the Newton step is not a descent direction of F
    and the method would wander off. An x0 off the affine set is first moved to
    its nearest point x0 − A⁺(Ax0 − b) (# NOTE: B&V require Ax0 = b); x0 must then satisfy
    every inequality strictly (else ValueError).

    Stops (converged) when the KKT residual of (x_k, λ_k) and the violation are ≤ tol.
    """
    method = "log_barrier"
    if not (t0 > 0.0 and mu > 1.0 and newton_tol > 0.0):
        raise ValueError("need t0 > 0, mu > 1 and newton_tol > 0")
    prob = vector_problem(problem, x0=x0)
    model = _Model(prob)
    E, I = model.is_eq, model.is_in
    eq_idx = np.flatnonzero(E)
    in_idx = np.flatnonzero(I)
    not_affine = [int(i) for i in eq_idx if not model.affine[i]]
    if not_affine:
        raise ValueError(
            f"log_barrier needs affine equality constraints (B&V §11.1); equality "
            f"constraint(s) {not_affine} of problem {prob.id!r} are not declared affine "
            "(set Problem.extra['affine'][i] = True for an affine c_i)"
        )
    x_given = start_point(prob, x0)
    x = x_given.copy()
    c = model.cval(x)
    info0: dict[str, Any] = {}
    if eq_idx.size and np.any(c[E] != 0.0):
        A = model.cjac(x)[E]
        x = x - np.asarray(np.linalg.lstsq(A, c[E], rcond=None)[0], dtype=np.float64)
        c = model.cval(x)
        info0["projected_from"] = x_given.tolist()
    if not (finite(c) and np.all(c[I] < 0.0)):
        raise ValueError(
            f"log_barrier needs a strictly feasible x0 (c_i(x0) < 0 for every inequality); "
            f"got c(x0) = {c.tolist()}"
        )
    m_in = int(in_idx.size)
    eye = np.eye(model.n)

    def evaluate(x: Vector) -> tuple[Vector, Matrix, Matrix, list[Matrix]]:
        g = model.grad(x)
        J = model.cjac(x)
        H = model.hess(x)
        CH = [eye * 0.0 if model.affine[i] else model.chess(int(i), x) for i in in_idx]
        return g, J, H, CH

    def barrier(t: float, f: float, c: Vector) -> float:
        return t * f - float(np.sum(np.log(-c[I])))

    def multipliers(t: float, g: Vector, J: Matrix, c: Vector) -> Vector:
        lam = np.zeros(model.m)
        lam[I] = -1.0 / (t * c[I])
        if eq_idx.size:
            r = g + J[I].T @ lam[I]
            lam[E] = np.linalg.lstsq(J[E].T, -r, rcond=None)[0]
        return lam

    fx = model.fun(x)
    g, J, H, CH = evaluate(x)
    t = t0
    if not finite(fx, g, J, H, *CH):
        trace = [Step(0, x.copy(), fx, info={**_state(c, E, 0, 0, math.nan), **info0})]
        return _finish(
            method, model, x, fx, False, _nonfinite_msg(x), trace, kkt=math.nan, violation=math.nan
        )
    lam = multipliers(t, g, J, c)
    stat, kkt = _kkt(g, J, lam, c, E)
    viol = _violation(c, E)
    F = barrier(t, fx, c)
    trace = [
        Step(
            0,
            x.copy(),
            fx,
            float(np.linalg.norm(g)),
            info={
                **_state(c, E, 0, 0, kkt),
                "t": t,
                "gap": m_in / t,
                "merit": F,
                "multipliers": lam.tolist(),
                "stationarity": stat,
                **info0,
            },
        )
    ]

    def done(converged: bool, msg: str) -> Result:
        return _finish(
            method, model, x, fx, converged, msg, trace, kkt=kkt, violation=viol, multipliers=lam
        )

    if kkt <= tol and viol <= tol:
        return done(True, _converged_msg(kkt, viol, tol))
    k = 0
    outer = 0
    inner = 0
    at_floor = False  # centering reached the rounding floor of this t (see the NOTE below)
    while True:
        ci = c[I]
        Ji = J[I]
        dF = t * g - Ji.T @ (1.0 / ci)
        HF = t * H + Ji.T @ (Ji / ci[:, None] ** 2)
        for ch, cv in zip(CH, ci, strict=True):
            HF = HF - ch / cv
        HF = 0.5 * (HF + HF.T)
        made = _make_pd(HF)
        if made is None:
            return done(False, "could not make the barrier Hessian positive definite")
        Hm, shift = made
        try:
            if eq_idx.size:
                A = J[E]
                p = A.shape[0]
                K = np.block([[Hm, A.T], [A, np.zeros((p, p))]])
                sol = np.asarray(np.linalg.solve(K, np.concatenate([-dF, -c[E]])), dtype=np.float64)
                dx = sol[: model.n]
            else:
                dx = _solve_pd(Hm, -dF)
        except np.linalg.LinAlgError:
            if eq_idx.size:
                return done(
                    False, "singular Newton (KKT) system: the equality constraints are dependent"
                )
            return done(False, "singular Newton system: the barrier Hessian is singular in float64")
        decrement = 0.5 * float(dx @ Hm @ dx)
        final_stage = m_in == 0 or 1.0 / t < tol
        if (decrement <= newton_tol or at_floor) and not final_stage:
            # Centering done and the complementarity 1/t is not yet below tol: B&V step 3
            # (stopping test with the current t), then step 4, t := μt.
            at_floor = False
            lam = multipliers(t, g, J, c)
            stat, kkt = _kkt(g, J, lam, c, E)
            if kkt <= tol and viol <= tol:
                return done(True, _converged_msg(kkt, viol, tol))
            t *= mu
            outer += 1
            inner = 0
            F = barrier(t, fx, c)
            continue
        # NOTE: once 1/t < tol, t stays fixed and Newton continues until the KKT test passes
        # (B&V's test |I|/t < ε assumes exact centering; the stationarity part of the KKT
        # residual needs a few more quadratically convergent steps).
        if k == max_iter:
            return done(False, _max_iter_msg(max_iter, kkt, viol))
        slope = float(dF @ dx)
        # With Ax = b, ∇FᵀΔx = −ΔxᵀHΔx < 0 (B&V §10.2.1); only rounding (|slope| at the
        # resolution of F) or an equality that is not affine makes it positive.
        if slope > _resolution(F):
            return done(
                False,
                f"the Newton step is not a descent direction of the barrier function "
                f"(∇FᵀΔx = {slope:.3g} > 0); are the equality constraints affine?",
            )
        # NOTE: when the predicted decrease |∇FᵀΔx| is below the resolution of F
        # (100ε max(1, |F|)), the Armijo comparison is decided by rounding; the full step is
        # then accepted if it is strictly feasible (B&V §9.5.3: in the quadratically
        # convergent phase backtracking always selects s = 1). The gradient is still exact
        # to rounding, so such steps go on reducing the stationarity, and the decrement
        # test λ²/2 ≤ newton_tol decides as usual. A below-resolution step that does not
        # reduce the stationarity marks the rounding floor of this t: before the final stage
        # the centering ends there (t := μt); in the final stage the method stops as stalled.
        below_resolution = -slope <= _resolution(F)  # |slope| ≤ resolution (see the guard)
        s = 1.0
        trials: list[list[float]] = []
        accepted = False
        xt, ft, ct, Ft = x, fx, c, F
        for _ in range(_MAX_TRIALS):
            xt = x + s * dx
            ct = model.cval(xt)
            if not (finite(ct) and np.all(ct[I] < 0.0)):
                trials.append([s, math.inf])
                s *= _BARRIER_BETA
                continue
            ft = model.fun(xt)
            Ft = barrier(t, ft, ct) if finite(ft) else math.inf
            trials.append([s, Ft])
            if math.isfinite(Ft) and (below_resolution or Ft <= F + _BARRIER_ALPHA * s * slope):
                accepted = True
                break
            s *= _BARRIER_BETA
        if not accepted:
            return done(False, f"backtracking line search failed after {_MAX_TRIALS} trials")
        if np.array_equal(xt, x):
            # The accepted step is below the spacing of the floating-point numbers at x (e.g.
            # x0 within rounding of the boundary, where the barrier Hessian is huge): an
            # exact fixed point, as in _sequential and sqp.
            where = f"; max_I c_i(x) = {float(np.max(c[I])):.3g}" if m_in else ""
            return done(False, _stalled_msg(kkt, viol, tol) + where)
        gt, Jt, Ht, CHt = evaluate(xt)
        if not finite(gt, Jt, Ht, *CHt):
            # The last accepted iterate (x, fx) = trace[-1] is returned; (lam, kkt, viol)
            # describe it.
            return done(False, _nonfinite_msg(xt))
        stat_before = stat
        x_prev = x
        x, fx, c, F = xt, ft, ct, Ft
        g, J, H, CH = gt, Jt, Ht, CHt
        k += 1
        inner += 1
        lam = multipliers(t, g, J, c)
        stat, kkt = _kkt(g, J, lam, c, E)
        viol = _violation(c, E)
        info = {
            **_state(c, E, outer, inner, kkt),
            "t": t,
            "gap": m_in / t,
            "merit": F,
            "multipliers": lam.tolist(),
            "stationarity": stat,
            "from": x_prev.tolist(),
            "direction": dx.tolist(),
            "alpha": s,
            "trials": trials,
            "newton_decrement": decrement,
            "hessian_shift": shift,
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), s, info))
        if kkt <= tol and viol <= tol:
            return done(True, _converged_msg(kkt, viol, tol))
        if below_resolution and not stat < stat_before:
            # A full Newton step in the quadratically convergent phase that does not reduce
            # the stationarity: the centering is at the rounding floor of this t. (The KKT
            # residual max(stationarity, 1/t) is not the measure: centering cannot change
            # its 1/t part.) Before the final stage this ends the centering (t := μt);
            # in the final stage nothing can improve the KKT residual any more.
            if not final_stage:
                at_floor = True
                continue
            if m_in:
                why = (
                    f"for t = {t:.3g}: {_excess(kkt, viol, tol)} (near the boundary the "
                    "rounding error of c_i(x) ≈ −1/(t λ_i) is amplified in λ_i = −1/(t c_i) "
                    "and in the barrier Hessian; use a larger tol)"
                )
            else:
                why = (
                    f"{_excess(kkt, viol, tol)} (no inequalities: the full Newton step of the "
                    "equality-constrained problem no longer reduces the KKT residual in "
                    "float64; use a larger tol)"
                )
            return done(False, f"stalled at the rounding level {why}")


# ======================================================================================
# Sequential quadratic programming
# ======================================================================================

#: N&W Alg. 18.3 constants: η (Armijo on the merit function), τ (backtracking factor),
#: ρ in the penalty rule (18.36).
_SQP_ETA = 1e-4
_SQP_TAU = 0.5
_SQP_RHO = 0.5


@register(
    id="sqp",
    family="constrained",
    name="SQP (line search, BFGS)",
    params=(_tol_param(1e-8), _max_iter_param(200)),
    needs=("f", "grad", "constraints"),
    order="superlinear (quasi-Newton)",
    summary="Solve a quadratic model of the Lagrangian subject to linearized constraints, "
    "then step along its solution with an ℓ1 merit-function line search.",
    references=(
        "Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Algorithm 18.3 "
        "(line-search SQP), eqs. 18.11 (QP), 18.36 (penalty update), Procedure 18.2 "
        "(damped BFGS)",
        "Nocedal & Wright (2006), Algorithm 18.1 (multiplier update λₖ₊₁ = λ̂)",
        "Powell (1978), A fast algorithm for nonlinearly constrained optimization "
        "calculations, LNM 630, 144–157",
        "Goldfarb & Idnani (1983), Math. Program. 27, 1–33 (QP subproblem solver)",
    ),
)
def sqp(
    problem: Problem | Callable[..., float],
    *,
    x0: Any = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Result:
    """Line-search SQP with a damped-BFGS Hessian of the Lagrangian (N&W Alg. 18.3).

    At (x_k, λ_k), with B_k ≈ ∇²_xx L (B₀ = I), solve the QP (N&W 18.11)

        min_p ½pᵀB_k p + ∇f_kᵀp   s.t.  ∇c_i(x_k)ᵀp + c_i(x_k) = 0 (E),  ≤ 0 (I)

    by the dual active-set method :func:`solve_qp`, giving p_k and multipliers λ̂. Then:

    * ℓ1 merit φ₁(x; μ) = f(x) + μ‖c(x)⁻‖₁, ‖c⁻‖₁ = Σ_E |c_i| + Σ_I max(0, c_i), with
      μ_k = max(μ_{k−1}, (∇f_kᵀp + ½pᵀB_k p)/((1 − ρ)‖c_k⁻‖₁)), ρ = 0.5 (N&W 18.36, σ = 1),
      so that D(φ₁; p) = ∇f_kᵀp − μ_k‖c_k⁻‖₁ ≤ −ρμ_k‖c_k⁻‖₁ < 0;
    * α = 1, 0.5, 0.25, ... until φ₁(x_k + αp) ≤ φ₁(x_k) + ηαD (η = 10⁻⁴);
    * x_{k+1} = x_k + αp, λ_{k+1} = λ̂;
    * damped BFGS (N&W Procedure 18.2): s = x_{k+1} − x_k,
      y = ∇_x L(x_{k+1}, λ_{k+1}) − ∇_x L(x_k, λ_{k+1}); θ = 1 if sᵀy ≥ 0.2 sᵀBs, else
      0.8 sᵀBs/(sᵀBs − sᵀy); r = θy + (1 − θ)Bs; B ← B − Bs sᵀB/sᵀBs + rrᵀ/sᵀr, which
      keeps B positive definite.

    λ₀ = 0. If the linearized constraints are inconsistent (the QP is infeasible) the
    method stops with converged=False (# NOTE: practical codes switch to an elastic mode).

    # NOTE: Alg. 18.3 damps the multipliers, λ_{k+1} = λ_k + α(λ̂ − λ_k). With short steps
    # α this keeps a poor early estimate (e.g. λ < 0 from the first QP with B₀ = I) for many
    # iterations; y then sees the negative curvature of ∇²L = ∇²f + Σλ_i∇²c_i at every step,
    # Powell's damping shrinks B along s each time, and B degenerates (observed on circle_eq).
    # We take the full multiplier step λ_{k+1} = λ̂ of the local SQP method (N&W Alg. 18.1;
    # Powell 1978), which also uses the QP multipliers in the BFGS update.
    # NOTE: (18.36) constrains μ only when ‖c_k⁻‖₁ > 0, so μ stays at its previous value
    # (μ₀ = 0) at feasible iterates and the first steps from a feasible x0 are judged by f
    # alone; a step may then leave the feasible set by a large amount (quadratic_disk from
    # (−0.5, 0.5) steps to (2, 1), c = 4). This is Alg. 18.3 as written. The common safeguard
    # μ ≥ ‖λ̂‖∞ + δ (exactness of φ₁ needs μ > ‖λ*‖∞, N&W Thm. 17.3) does not prevent it there:
    # λ̂ = 0 in that QP, and even μ = 1.01‖λ*‖∞ accepts the same step.
    # NOTE: as in the other line searches of this module, when the predicted change −αD of
    # a trial is below the merit's resolution 100ε·max(1, |φ₁|), the trial is accepted if φ₁
    # increases by less than that resolution (the comparison is otherwise rounding noise);
    # a step that changes neither x nor λ is a fixed point and stops the method.

    Stops (converged) when the KKT residual of (x_k, λ_k) and the violation are ≤ tol.
    """
    method = "sqp"
    prob = vector_problem(problem, x0=x0)
    model = _Model(prob)
    E = model.is_eq
    I = model.is_in
    x = start_point(prob, x0)
    fx = model.fun(x)
    c = model.cval(x)
    g = model.grad(x)
    J = model.cjac(x)
    lam = np.zeros(model.m)
    mu_merit = 0.0

    def l1(c: Vector) -> float:
        return float(np.sum(np.abs(c[E])) + np.sum(np.maximum(c[I], 0.0)))

    if not finite(fx, c, g, J):
        trace = [Step(0, x.copy(), fx, info=_state(c, E, 0, 0, math.nan))]
        return _finish(
            method, model, x, fx, False, _nonfinite_msg(x), trace, kkt=math.nan, violation=math.nan
        )
    stat, kkt = _kkt(g, J, lam, c, E)
    viol = _violation(c, E)
    phi = fx + mu_merit * l1(c)
    info0 = {
        **_state(c, E, 0, 0, kkt),
        "mu": mu_merit,
        "merit": phi,
        "multipliers": lam.tolist(),
        "stationarity": stat,
    }
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), info=info0)]

    def done(converged: bool, msg: str) -> Result:
        return _finish(
            method, model, x, fx, converged, msg, trace, kkt=kkt, violation=viol, multipliers=lam
        )

    B = np.eye(model.n)
    k = 0
    while True:
        if kkt <= tol and viol <= tol:
            return done(True, _converged_msg(kkt, viol, tol))
        if k == max_iter:
            return done(False, _max_iter_msg(max_iter, kkt, viol))
        qp = solve_qp(B, g, J[E], -c[E], J[I], -c[I])
        if not qp.ok:
            return done(False, f"QP subproblem failed at x = {x.tolist()}: {qp.message}")
        p = qp.x
        lam_hat = np.zeros(model.m)
        lam_hat[E] = qp.lam_eq
        lam_hat[I] = qp.lam_ub
        in_idx = np.flatnonzero(I)
        working = sorted([int(i) for i in np.flatnonzero(E)] + [int(in_idx[j]) for j in qp.active])
        v1 = l1(c)
        gp = float(g @ p)
        pBp = float(p @ B @ p)
        if v1 > 0.0:
            mu_merit = max(mu_merit, (gp + 0.5 * pBp) / ((1.0 - _SQP_RHO) * v1))
        D = gp - mu_merit * v1
        phi = fx + mu_merit * v1
        alpha = 1.0
        trials: list[list[float]] = []
        accepted = False
        xt, ft, ct = x, fx, c
        for _ in range(_MAX_TRIALS):
            xt = x + alpha * p
            ft = model.fun(xt)
            ct = model.cval(xt)
            phit = ft + mu_merit * l1(ct) if finite(ft, ct) else math.inf
            trials.append([alpha, phit])
            if phit <= phi + _SQP_ETA * alpha * D or (
                -alpha * D <= _resolution(phi) and phit <= phi + _resolution(phi)
            ):
                accepted = True
                break
            alpha *= _SQP_TAU
        if not accepted:
            return done(False, f"merit-function line search failed after {_MAX_TRIALS} trials")
        gt = model.grad(xt)
        Jt = model.cjac(xt)
        if not finite(gt, Jt):
            # The last accepted iterate (x, fx) = trace[-1] is returned.
            return done(False, _nonfinite_msg(xt))
        lam_new = lam_hat
        if np.array_equal(xt, x) and np.array_equal(lam_new, lam):
            return done(False, _stalled_msg(kkt, viol, tol))
        s = xt - x
        y = (gt + Jt.T @ lam_new) - (g + J.T @ lam_new)
        Bs = B @ s
        sBs = float(s @ Bs)
        theta: float | None = None
        if sBs > 0.0:
            sy = float(s @ y)
            theta = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
            r = theta * y + (1.0 - theta) * Bs
            B = B - np.outer(Bs, Bs) / sBs + np.outer(r, r) / float(s @ r)
            B = 0.5 * (B + B.T)
        x_prev = x
        x, fx, c, g, J, lam = xt, ft, ct, gt, Jt, lam_new
        k += 1
        stat, kkt = _kkt(g, J, lam, c, E)
        viol = _violation(c, E)
        info = {
            **_state(c, E, 0, k, kkt),
            "mu": mu_merit,
            "merit": fx + mu_merit * l1(c),
            "multipliers": lam.tolist(),
            "stationarity": stat,
            "from": x_prev.tolist(),
            "direction": p.tolist(),
            "alpha": alpha,
            "trials": trials,
            "working_set": working,
            "theta": theta,
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), alpha, info))


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("projected_gradient", "box_quadratic", {}),
    ("projected_gradient", "halfplanes_quadratic", {}),
    ("frank_wolfe", "quadratic_disk", {"step": "armijo"}),
    ("quadratic_penalty", "quadratic_disk", {"tol": 1e-5}),
    ("augmented_lagrangian", "circle_eq", {}),
    ("log_barrier", "hs21", {}),
    ("sqp", "rosenbrock_unit_disk", {}),
    ("sqp", "hs21", {"x0": [-1.0, -1.0]}),
]
