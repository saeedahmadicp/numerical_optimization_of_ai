"""Nonlinear least squares: minimize f(x) = ½‖r(x)‖² with r: ℝⁿ → ℝᵐ and its Jacobian J.

Both methods replace r near x by its linearization r(x + h) ≈ r + J h, i.e. f by the model

    L(h) = ½‖r + J h‖² = f + hᵀg + ½ hᵀJᵀJ h,      g = ∇f = Jᵀr,

(Nocedal & Wright (2006), §10.3; Madsen, Nielsen & Tingleff (2004), eq. 3.7b). Gauss–Newton
minimizes L exactly; Levenberg–Marquardt adds the damping term ½μ‖h‖².

Linear algebra: every subproblem is solved from the thin SVD J = U diag(σ) Vᵀ, computed once per
Jacobian, never from the normal equations (forming JᵀJ squares the condition number; Higham
(2002), §20.4; N&W §10.2):

    Gauss–Newton:         p = −V diag(1/σ) Uᵀ r              (min ‖r + J p‖)
    Levenberg–Marquardt:  h = −V diag(σ/(σ² + μ)) Uᵀ r       ((JᵀJ + μI) h = −g; N&W eq. 10.38)

so the LM step for a new μ costs only O(mn) and κ₂(JᵀJ) = (σ_max/σ_min)² comes for free.

Tolerances: ``gtol`` is absolute (scaling r by c scales ∇f = Jᵀr by c²); ``xtol`` is relative to
‖x‖. Both methods accept steps by comparing values of f, which resolve decreases only down to
the rounding noise of f, so x is pinned to ≈ √(2·noise/λ_min(∇²f)) (the classic √ε limit).
The step test is normwise (MNT Alg. 3.16): it certifies ‖x − x*‖ ≲ xtol·‖x‖, not that f is near
f*. When the parameters differ greatly in scale it can stop while a small coordinate still
matters: e.g. exp_decay_fit from (1, −84), where ∂r/∂a ≈ 1e153, stops with a = 7e-9 (within
1e-10·‖x‖ of the valley a = 0) but f = 8e281. The stop message reports ‖Jᵀr‖∞ so this is visible.

Overflow and underflow: a start or an iterate where f = ½‖r‖² or ∇f = Jᵀr overflows (or the
SVD of J fails) ends the run with ``converged=False``. When f underflows to exactly 0, x attains
the global minimum value of f in floating point, and the run stops as converged.

Evaluation counts: ``n_fev`` counts residual evaluations r(x) (each also gives f), ``n_gev``
counts Jacobian evaluations, ``n_hev`` is 0. Without a ``jac`` the Jacobian is formed by
central differences (``numopt.core.diff.jacobian``) and its 2n + 1 residual evaluations are
included in ``n_fev``.

Info keys (one Step per iteration; k = 0 is the start). Every Step of a method, including the
single Step of a failed start, carries the same key set:
    step: [n] | None — the step computed at this iteration: the full Gauss–Newton step p
        (the iterate moves by alpha·p), or the LM trial step h (taken only if accepted).
        ``None`` at k = 0.
    lambda: float | None — the damping μ used to compute ``step`` (LM; at k = 0 the initial
        μ₀ = τ·max diag(JᵀJ), ``inf`` if it overflows). Always 0.0 for Gauss–Newton, which is
        LM with μ = 0. ``None`` only on an LM failed start where J is not available.
    gain_ratio: float | None — ϱ = (f(x) − f(x + h)) / (L(0) − L(h)), actual over predicted
        reduction for the step h actually tried (h = alpha·p for Gauss–Newton). ``None`` at
        k = 0, when f(x + h) is not finite, or when the predicted reduction L(0) − L(h) is not
        positive (e.g. μ is near overflow, h is subnormal and L(0) − L(h) underflows to 0;
        f(x + h) is then finite).
    residual_norm: float | None — ‖r(x_k)‖₂ at the current iterate (``inf`` when f overflows;
        ``None`` on a failed start where r is not finite).
    jtj_cond: float | None — κ₂(JᵀJ) = (σ_max/σ_min)² at the current iterate; ``inf`` when J
        is numerically rank-deficient (σ_min ≤ max(m, n)·ε·σ_max). ``None`` on a failed start
        where J (or its SVD) is not available.
    accepted: bool | None — whether the step was taken (LM rejects steps with ϱ ≤ 0;
        Gauss–Newton steps are always taken). ``None`` at k = 0.
    nu: float — LM only: Nielsen's growth factor ν after this iteration's update.
    alpha: float | None — Gauss–Newton only: the accepted step length along p (1.0 without
        line search). ``None`` at k = 0.
    trials: [[alpha, f]] — Gauss–Newton only: every step length tried by the Armijo
        backtracking and f(x + alpha p) there, in order (``[]`` at k = 0).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import diff
from ..core.counting import Counted, start_point
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, as_vector

Array = NDArray[np.float64]

# Armijo backtracking constants (Nocedal & Wright, Alg. 3.1): sufficient-decrease constant c₁,
# contraction factor ρ, and the number of contractions before the search gives up (α ≥ 2⁻⁵⁰).
_ARMIJO_C1 = 1e-4
_ARMIJO_RHO = 0.5
_ARMIJO_MAX_BACKTRACKS = 50

_EPS = float(np.finfo(np.float64).eps)

# Relative rounding noise of f = ½‖r‖²: fl(f) is wrong by ≈ ε·Σ|rᵢ|(|rᵢ| + |modelᵢ|) ≤ 100·ε·f
# when |model| ≲ 50|r| (Higham (2002), §3.1). ‖Jp‖²/‖r‖² = (L(0) − L(p))/f is the decrease the
# Gauss–Newton model predicts relative to f; at or below this level comparisons of f cannot
# resolve it, so x is then within √(2·100·ε·f/λ_min) of x* (the module docstring's √ε limit).
_F_NOISE_REL = 100.0 * _EPS

# Default Gauss–Newton ftol ≈ 45ε. It must lie above the rounding noise of f (a few ε·f in
# practice): below that level the Armijo search cannot show the full step's decrease
# f·‖Jp‖²/‖r‖² and stalls with converged=False at x*. It should lie below _F_NOISE_REL: the ftol
# stop leaves ½eᵀJᵀJe ≈ ftol·f, i.e. ‖e‖ ≲ √(2·ftol·f/λ_min) ≈ 0.67·√(2·_F_NOISE_REL·f/λ_min),
# inside the module docstring's √ε limit.
# NOTE: the audited default 1e-15 ≈ 4.5ε was below the rounding noise of f: 6 of 4000 domain
# starts on circle_fit and michaelis_menten stalled at x* with converged=False. With 1e-14,
# 30000 domain starts on exp_decay_fit, circle_fit and michaelis_menten gave no stall.
_GN_FTOL = 1e-14

_TOL_PARAMS = (
    ParamSpec(
        "gtol",
        1e-8,
        min=1e-14,
        max=1e-2,
        log=True,
        help="Stop when the gradient ‖Jᵀr‖∞ ≤ gtol.",
    ),
    ParamSpec(
        "xtol",
        1e-10,
        min=1e-15,
        max=1e-2,
        log=True,
        help="Stop when the step ‖h‖₂ ≤ xtol·(‖x‖₂ + xtol).",
    ),
)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


@dataclass
class _Linearization:
    """r, J, f, g and the thin SVD of J at one point."""

    r: Array  # (m,)
    J: Array  # (m, n)
    f: float
    g: Array  # (n,)
    U: Array  # (m, p), p = min(m, n)
    s: Array  # (p,) singular values, descending
    Vt: Array  # (p, n)

    @property
    def rank_tol(self) -> float:
        """NumPy's ``matrix_rank`` tolerance max(m, n)·ε·σ_max."""
        return max(self.J.shape) * _EPS * float(self.s[0]) if self.s.size else 0.0

    @property
    def rank_deficient(self) -> bool:
        """Numerical rank < n: σ_min ≤ max(m, n)·ε·σ_max (or fewer than n singular values)."""
        n = self.J.shape[1]
        if self.s.size < n or self.s[0] == 0.0:
            return True
        return bool(self.s[-1] <= self.rank_tol)

    @property
    def jtj_cond(self) -> float:
        """κ₂(JᵀJ) = (σ_max/σ_min)², or ``inf`` when J is numerically rank-deficient."""
        if self.rank_deficient:
            return math.inf
        return float((self.s[0] / self.s[-1]) ** 2)

    def gauss_newton_step(self) -> Array:
        """Minimum-norm Gauss–Newton step p = −J⁺r, with σᵢ ≤ ``rank_tol`` treated as 0."""
        keep = self.s > self.rank_tol  # (p,)
        inv_s = np.divide(1.0, self.s, out=np.zeros_like(self.s), where=keep)
        return -(self.Vt.T @ (inv_s * (self.U.T @ self.r)))  # (n,)

    def gauss_newton_ratio(self) -> float:
        """‖Jp‖²/‖r‖² for the Gauss–Newton step p: the fraction of f its model removes.

        Jp = −U_k U_kᵀ r over the kept singular vectors, so the ratio is ‖U_kᵀ r̂‖²/‖r̂‖² with
        r̂ = r/‖r‖∞. The ratio is scale-free: it does not underflow, and it does not divide by
        0 when f = ½‖r‖² underflows. It equals cos²∠(r, range J).
        """
        r_max = float(np.max(np.abs(self.r))) if self.r.size else 0.0
        if r_max == 0.0:
            return 0.0
        r_hat = self.r / r_max  # (m,)
        u_hat = (self.U.T @ r_hat)[self.s > self.rank_tol]  # (rank,)
        return float(u_hat @ u_hat) / float(r_hat @ r_hat)


def _linearize(r: Array, J: Array) -> _Linearization:
    m, n = J.shape
    with np.errstate(all="ignore"):
        g = J.T @ r  # (n,); may overflow, checked by _breakdown
        try:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            p = min(m, n)
            U, s, Vt = np.full((m, p), np.nan), np.full(p, np.nan), np.full((p, n), np.nan)
    return _Linearization(r=r, J=J, f=_half_sq(r), g=g, U=U, s=s, Vt=Vt)


def _breakdown(lin: _Linearization, where: str) -> str:
    """Why the linearization is unusable (overflow of f or g, failed SVD), or ``""``."""
    if not math.isfinite(lin.f):
        r_max = float(np.max(np.abs(lin.r)))
        return f"f = ½‖r‖² overflows {where} (‖r‖∞ = {r_max:.3g})"
    if not _finite(lin.g):
        return f"the gradient ∇f = Jᵀr overflows {where}"
    if not _finite(lin.s):
        return f"the SVD of the Jacobian failed {where} (non-finite singular values)"
    return ""


def _norm2(v: Array) -> float:
    """‖v‖₂ computed as s·‖v/s‖₂ with s = ‖v‖∞ (Higham (2002), §27.5).

    The scaling stops vᵀv from overflowing (‖g‖ near 1e290) or underflowing (a step near
    1e-300, which ``np.linalg.norm`` returns as 0 and which would pass the step test falsely).
    """
    scale = float(np.max(np.abs(v))) if v.size else 0.0
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return scale * float(np.linalg.norm(v / scale))


def _half_sq(r: Array) -> float:
    with np.errstate(over="ignore"):
        return 0.5 * float(np.dot(r, r))


def _resolve(
    problem: Problem | Callable[[Array], ArrayLike], x0: Any
) -> tuple[Array, Counted, Counted]:
    """Return (x0, counted residual, counted Jacobian) for a Problem or a bare r(x)."""
    if isinstance(problem, Problem):
        if problem.residual is None:
            raise ValueError(
                f"{problem.id}: least-squares methods need a residual r(x) (Problem.residual)"
            )
        x = start_point(problem, x0)
        residual, jac = problem.residual, problem.jac
    elif callable(problem):
        if x0 is None:
            raise ValueError("a starting point x0 is required for a bare residual function")
        x = as_vector(x0)
        residual, jac = problem, None
    else:
        raise TypeError("problem must be a numopt Problem or a callable residual r(x)")

    res = Counted(lambda z: np.asarray(residual(z), dtype=np.float64).reshape(-1))
    if jac is None:
        jac_fn = Counted(lambda z: diff.jacobian(res, z))
    else:
        jac_fn = Counted(lambda z: np.atleast_2d(np.asarray(jac(z), dtype=np.float64)))
    return x, res, jac_fn


def _evaluate(fn: Counted, x: Array) -> Array:
    """Evaluate without floating-point warnings; callers check the result for finiteness."""
    with np.errstate(all="ignore"):
        return fn(x)


def _finite(a: Array) -> bool:
    return bool(np.all(np.isfinite(a)))


def _info(
    lin: _Linearization,
    *,
    step: Array | None,
    lam: float,
    gain: float | None,
    accepted: bool | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "step": None if step is None else step.tolist(),
        "lambda": lam,
        "gain_ratio": gain,
        "residual_norm": math.sqrt(2.0 * lin.f),
        "jtj_cond": lin.jtj_cond,
        "accepted": accepted,
        **extra,
    }


def _gtol_met(lin: _Linearization, gtol: float) -> bool:
    return float(np.max(np.abs(lin.g))) <= gtol


def _xtol_met(h: Array, x: Array, xtol: float) -> bool:
    return _norm2(h) <= xtol * (_norm2(x) + xtol)


def _msg_gtol(lin: _Linearization) -> str:
    return f"gradient ‖Jᵀr‖∞ = {float(np.max(np.abs(lin.g))):.3g} ≤ gtol"


def _msg_xtol(h: Array, lin: _Linearization) -> str:
    return (
        f"step ‖h‖ = {_norm2(h):.3g} ≤ xtol·(‖x‖ + xtol) "
        f"(‖Jᵀr‖∞ = {float(np.max(np.abs(lin.g))):.3g})"
    )


_MSG_ZERO_F = "f = ½‖r‖² = 0 in floating point: x attains the global minimum value 0"


def _start(x: Array, res: Counted, jac: Counted) -> tuple[_Linearization | None, Array, str]:
    """Evaluate r and J at x0; return (linearization if J is usable, r, failure message).

    The linearization is returned even when it fails (f or g overflow), so the failed-start
    Step can still show κ₂(JᵀJ); the message is ``""`` exactly when the start is usable.
    """
    r = _evaluate(res, x)
    if not _finite(r):
        return None, r, "residual is not finite at x0"
    J = _evaluate(jac, x)
    if J.shape != (r.size, x.size) or not _finite(J):
        return None, r, "Jacobian is not finite (or has the wrong shape) at x0"
    lin = _linearize(r, J)
    return lin, r, _breakdown(lin, "at x0")


def _failed_start(
    method: str,
    x: Array,
    r: Array,
    lin: _Linearization | None,
    message: str,
    res: Counted,
    jac: Counted,
    **extra: Any,
) -> Result:
    """Result for a start that cannot be iterated; the Step carries the full info key set."""
    f = _half_sq(r) if _finite(r) else math.nan
    info = {
        "step": None,
        "lambda": extra.pop("lam"),
        "gain_ratio": None,
        "residual_norm": math.sqrt(2.0 * f) if not math.isnan(f) else None,
        "jtj_cond": lin.jtj_cond if lin is not None and _finite(lin.s) else None,
        "accepted": None,
        **extra,
    }
    trace = [Step(0, x.copy(), f, info=info)]
    return Result(method, x, f, False, message, 0, res.n, jac.n, 0, trace=trace)


# --------------------------------------------------------------------------------------
# Gauss–Newton
# --------------------------------------------------------------------------------------


@register(
    id="gauss_newton",
    family="least_squares",
    name="Gauss–Newton",
    params=(
        *_TOL_PARAMS,
        ParamSpec(
            "ftol",
            _GN_FTOL,
            min=1e-18,
            max=1e-4,
            log=True,
            help="Stop when the predicted relative reduction ‖Jp‖²/‖r‖² of the full step ≤ ftol.",
        ),
        ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Iteration limit."),
        ParamSpec(
            "line_search",
            "backtracking",
            kind="choice",
            choices=("backtracking", "none"),
            help="Armijo backtracking along the Gauss–Newton step, or the full step.",
        ),
    ),
    needs=("residual", "jac"),
    order="quadratic for zero residual, linear otherwise",
    summary="Linearize the residuals and jump to the least-squares solution of the linear model.",
    references=(
        "Nocedal & Wright (2006), §10.3, eq. 10.23",
        "Björck (1996), Numerical Methods for Least Squares Problems, §9.2",
    ),
)
def gauss_newton(
    problem: Problem | Callable[[Array], ArrayLike],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    xtol: float = 1e-10,
    ftol: float = _GN_FTOL,
    max_iter: int = 100,
    line_search: str = "backtracking",
) -> Result:
    """Gauss–Newton method (Nocedal & Wright (2006), §10.3).

    Iteration: the step p_k minimizes the linear model ‖r_k + J_k p‖ (N&W eq. 10.23,
    J_kᵀJ_k p = −J_kᵀ r_k), computed from the SVD of J_k as p = −V diag(1/σ) Uᵀ r. Then
    x_{k+1} = x_k + α_k p_k, with α_k = 1 (``line_search="none"``) or the first
    α ∈ {1, ½, ¼, …} meeting the Armijo condition f(x + αp) ≤ f(x) + c₁ α ∇fᵀp, c₁ = 10⁻⁴
    (N&W Alg. 3.1). When J has full column rank, ∇fᵀp = −‖Jp‖² < 0, so p is a descent
    direction (N&W eq. 10.24).

    Stopping tests, checked at the top of every iteration and once more after the last one
    (none of them costs an evaluation, so a run capped at ``max_iter`` = N reports the same
    flag as an uncapped run that stops at iteration N); when one passes, the method has
    converged and the pending step is not taken:
        * gradient:  ‖∇f‖∞ = ‖Jᵀr‖∞ ≤ ``gtol``;
        * zero f:    f = ½‖r‖² = 0 in floating point (it underflowed; the global minimum value);
        * step:      ‖p‖₂ ≤ ``xtol``·(‖x‖₂ + ``xtol``)  (Madsen, Nielsen & Tingleff (2004),
          Alg. 3.16; normwise, see the module docstring);
        * reduction: ‖Jp‖²/‖r‖² ≤ ``ftol``, i.e. the linear model predicts that the full step
          lowers f by at most the fraction ``ftol`` (MINPACK's ftol test, Moré, Garbow &
          Hillstrom (1980), User Guide §2.3, applied to the predicted reduction only). It equals
          cos²∠(r, range J), a scale-free stationarity measure, and never fires on
          zero-residual problems (there r ∈ range J and the ratio → 1). It is computed from
          r/‖r‖∞, so it stays defined when f underflows.
    Failures (``converged=False``): J numerically rank-deficient (σ_min ≤ max(m, n)·ε·σ_max;
    the Gauss–Newton step is then not unique), non-finite residual or Jacobian, f or ∇f
    overflowing, the Armijo search not succeeding within 50 halvings, or ``max_iter``
    reached. The line search also
    stops (not converged) when α|∇fᵀp| ≤ ε·f without a decrease, i.e. f can no longer
    certify progress. This needs a predicted reduction ‖Jp‖²/‖r‖² in (``ftol``, rounding
    noise of f), so it can occur only when ``ftol`` is set below the default 1e-14 ≈ 45ε (the
    rounding noise of f is a few ε·f, at most ≈ 100ε·f).
    """
    if line_search not in ("backtracking", "none"):
        raise ValueError(f"line_search must be 'backtracking' or 'none', got {line_search!r}")
    name = "gauss_newton"
    x, res, jac = _resolve(problem, x0)
    lin0, r0, why = _start(x, res, jac)
    if lin0 is None or why:
        return _failed_start(name, x, r0, lin0, why, res, jac, lam=0.0, alpha=None, trials=[])
    lin: _Linearization = lin0

    trace = [
        Step(
            0,
            x.copy(),
            lin.f,
            grad_norm=_norm2(lin.g),
            info=_info(lin, step=None, lam=0.0, gain=None, accepted=None, alpha=None, trials=[]),
        )
    ]

    def done(converged: bool, message: str, n_iter: int) -> Result:
        return Result(name, x, lin.f, converged, message, n_iter, res.n, jac.n, 0, trace=trace)

    # k = max_iter + 1 only runs the stopping tests at the last iterate (no evaluations).
    for k in range(1, max_iter + 2):
        if _gtol_met(lin, gtol):
            return done(True, _msg_gtol(lin), k - 1)
        if lin.f == 0.0:
            return done(True, _MSG_ZERO_F, k - 1)
        if lin.rank_deficient:
            ratio = float(lin.s[-1] / lin.s[0]) if lin.s.size and lin.s[0] > 0.0 else 0.0
            return done(
                False,
                f"Jacobian is numerically rank-deficient (σ_min/σ_max = {ratio:.3g} ≤ "
                "max(m, n)·ε); the Gauss–Newton step is not unique",
                k - 1,
            )
        utr = lin.U.T @ lin.r  # (p,)
        p = -(lin.Vt.T @ (utr / lin.s))  # (n,)
        if _xtol_met(p, x, xtol):
            return done(True, _msg_xtol(p, lin), k - 1)
        # NOTE: the ftol test is not in N&W. It is needed with the line search: comparisons of f
        # cannot certify a decrease below the rounding noise of f (a few ε·f, ≤ 100ε·f), so once
        # ‖Jp‖²/‖r‖² is at that level the Armijo test stalls. On the badly scaled
        # michaelis_menten problem this happens while ‖g‖∞ ≈ 1.7e-4 ≫ gtol (x is then accurate
        # to 3e-10 relative). The default 1e-14 ≈ 45ε (_GN_FTOL) makes it a documented stop.
        reduction = lin.gauss_newton_ratio()  # ‖Jp‖²/‖r‖², scale-free (safe if f underflows)
        if reduction <= ftol:
            return done(
                True, f"predicted relative reduction ‖Jp‖²/‖r‖² = {reduction:.3g} ≤ ftol", k - 1
            )
        if k > max_iter:
            return done(False, f"reached max_iter={max_iter}", max_iter)

        slope = float(lin.g @ p)  # ∇fᵀp = −‖Jp‖² in exact arithmetic
        trials: list[list[float]] = []
        alpha = 1.0
        if line_search == "backtracking":
            if not slope < 0.0:
                return done(False, "Gauss–Newton step is not a descent direction (rounding)", k - 1)
            for _ in range(_ARMIJO_MAX_BACKTRACKS + 1):
                r_new = _evaluate(res, x + alpha * p)
                f_new = _half_sq(r_new) if _finite(r_new) else math.inf
                trials.append([alpha, f_new])
                # NOTE: we also require f_new < f. Armijo with ∇fᵀp < 0 implies it in exact
                # arithmetic; without it, steps below the resolution of f (x + αp == x) are
                # "accepted" with f_new == f and the iteration freezes until max_iter.
                if f_new < lin.f and f_new <= lin.f + _ARMIJO_C1 * alpha * slope:
                    break
                if alpha * -slope <= _EPS * lin.f:
                    # The first-order decrease α|∇fᵀp| is below the rounding level ε·f: no
                    # shorter step can show a decrease, so x is as accurate as f can certify.
                    return done(
                        False,
                        "no measurable decrease along the Gauss–Newton step (the predicted "
                        f"reduction is below the rounding level of f); ‖Jᵀr‖∞ = "
                        f"{float(np.max(np.abs(lin.g))):.3g} > gtol",
                        k - 1,
                    )
                alpha *= _ARMIJO_RHO
            else:
                return done(
                    False,
                    f"Armijo backtracking failed after {_ARMIJO_MAX_BACKTRACKS} halvings",
                    k - 1,
                )
        else:
            r_new = _evaluate(res, x + p)
            f_new = _half_sq(r_new) if _finite(r_new) else math.inf
            trials.append([alpha, f_new])
            if not math.isfinite(f_new):
                return done(False, "residual is not finite at the Gauss–Newton step", k - 1)

        h = alpha * p
        # Predicted reduction L(0) − L(h) = −gᵀh − ½‖Jh‖².
        predicted = -float(lin.g @ h) - 0.5 * float(np.dot(lin.J @ h, lin.J @ h))
        gain = (lin.f - f_new) / predicted if predicted > 0.0 else None
        x_new = x + h
        J_new = _evaluate(jac, x_new)
        if not _finite(J_new):
            return done(False, "Jacobian is not finite at the new iterate", k - 1)
        lin_new = _linearize(r_new, J_new)
        why = _breakdown(lin_new, "at the new iterate")
        if why:
            return done(False, why, k - 1)
        x = x_new
        lin = lin_new
        trace.append(
            Step(
                k,
                x.copy(),
                lin.f,
                grad_norm=_norm2(lin.g),
                step_size=_norm2(h),
                info=_info(
                    lin, step=p, lam=0.0, gain=gain, accepted=True, alpha=alpha, trials=trials
                ),
            )
        )
    raise AssertionError("unreachable: the loop returns at k = max_iter + 1")


# --------------------------------------------------------------------------------------
# Levenberg–Marquardt
# --------------------------------------------------------------------------------------


@register(
    id="levenberg_marquardt",
    family="least_squares",
    name="Levenberg–Marquardt",
    params=(
        *_TOL_PARAMS,
        ParamSpec("max_iter", 200, kind="int", min=1, max=10_000, help="Iteration limit."),
        ParamSpec(
            "tau",
            1e-3,
            min=1e-8,
            max=1e2,
            log=True,
            help="Initial damping μ₀ = tau·max diag(JᵀJ); small trusts Gauss–Newton at once.",
        ),
    ),
    needs=("residual", "jac"),
    # Near a zero-residual solution e_{k+1}/e_k ≈ μ_k/(σ_min² + μ_k), and Nielsen's update lowers
    # μ by at most 3× per step, so the rate is Q-superlinear but not quadratic; quadratic order
    # needs μ_k = O(‖r_k‖) (Yamashita & Fukushima (2001)). Measured on rosenbrock_ls: order
    # estimates 1.49, 1.35, 1.26, 1.21, 1.17 (tests/test_unconstrained_least_squares.py).
    order="superlinear for zero residual (not quadratic: μ falls at most 3× per step), "
    "linear otherwise",
    summary="Gauss–Newton with an adaptive damping term: large μ gives short gradient-like "
    "steps, small μ gives Gauss–Newton steps.",
    references=(
        "Madsen, Nielsen & Tingleff (2004), Methods for Non-Linear Least Squares Problems, Alg. 3.16",
        "Nielsen (1999), Damping parameter in Marquardt's method, IMM-REP-1999-05",
        "Nocedal & Wright (2006), §10.3",
        "Moré (1978), The Levenberg–Marquardt algorithm: implementation and theory",
    ),
)
def levenberg_marquardt(
    problem: Problem | Callable[[Array], ArrayLike],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-8,
    xtol: float = 1e-10,
    max_iter: int = 200,
    tau: float = 1e-3,
) -> Result:
    """Levenberg–Marquardt method with Nielsen's damping update (Madsen, Nielsen & Tingleff
    (2004), Alg. 3.16).

    Iteration: solve (JᵀJ + μI) h = −Jᵀr (from the SVD of J) and compute the gain ratio
    ϱ = (f(x) − f(x + h)) / (L(0) − L(h)) with L(0) − L(h) = ½ hᵀ(μh − g) (MNT eq. 3.14,
    positive whenever g ≠ 0). If ϱ > 0 the step is accepted, x ← x + h, and
    μ ← μ·max(1/3, 1 − (2ϱ − 1)³), ν ← 2; otherwise x is kept and μ ← μ·ν, ν ← 2ν (Nielsen
    1999). μ₀ = τ·max_i (JᵀJ)_ii. Each trial (accepted or rejected) is one iteration.
    LM is the trust-region method with the Gauss–Newton model; μ is the Lagrange multiplier
    of the trust-region constraint (N&W §10.3, Thm. 10.2; Moré 1978). The damping is μI
    (Levenberg's form), so the iterates depend on how the parameters are scaled.

    Stopping tests (MNT Alg. 3.16), checked at the top of every iteration and once more after
    the last one (they cost no evaluations, so a run capped at ``max_iter`` = N reports the
    same flag as an uncapped run that stops at iteration N): converged when
    ‖g‖∞ = ‖Jᵀr‖∞ ≤ ``gtol``, when f = ½‖r‖² underflows to 0 (the global minimum value), or
    when the trial step satisfies ‖h‖₂ ≤ ``xtol``·(‖x‖₂ + ``xtol``) (the step is not taken)
    and h is not small merely because μ is large. The step test certifies convergence only
    if the Gauss–Newton step p = −J⁺r is small too (‖p‖₂ ≤ 2·``xtol``·(‖x‖₂ + ``xtol``)) or
    its model predicts a relative decrease ‖Jp‖²/‖r‖² ≤ 100·ε, at or below the rounding
    noise of f, which comparisons of f cannot resolve. Otherwise, after an accepted trial the
    iteration continues (μ keeps falling); after a rejected trial it stops with
    ``converged=False``. A large μ₀ (large ``tau``) near a minimizer can end that way: the
    damped steps then predict decreases below the rounding noise of f, Nielsen's update
    raises μ on each rejection, and the run reports that the damping blocked it.
    Failures (``converged=False``): non-finite residual or Jacobian at an accepted point,
    f or ∇f overflowing, μ₀ or μ overflowing, a step test blocked by the damping as above, or
    ``max_iter``. A non-finite f(x + h) counts as a rejected step (ϱ undefined), so μ grows;
    if the step test then fires before another step is accepted, the run is reported as *not*
    converged (the step is small only because μ grew, not because x is near a stationary
    point).
    """
    if not tau > 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    name = "levenberg_marquardt"
    x, res, jac = _resolve(problem, x0)
    lin0, r0, why = _start(x, res, jac)
    mu = math.nan
    if lin0 is not None:
        # NOTE: damping uses μI (Levenberg's form, as in MNT Alg. 3.16), not Marquardt's /
        # Moré's scaled form μ·DᵀD with D = diag of column norms; this keeps the textbook method.
        with np.errstate(over="ignore"):
            mu = tau * float(np.max(np.sum(lin0.J * lin0.J, axis=0)))  # τ·max diag(JᵀJ)
        if not why and not math.isfinite(mu):
            why = "the initial damping μ₀ = τ·max diag(JᵀJ) overflows at x0"
    if lin0 is None or why:
        lam0 = mu if lin0 is not None else None
        return _failed_start(name, x, r0, lin0, why, res, jac, lam=lam0, nu=2.0)
    lin: _Linearization = lin0
    nu = 2.0
    blocked = False  # a trial since the last accepted step gave a non-finite residual
    last_rejected = False  # the previous trial was rejected (ϱ ≤ 0 or f(x + h) not finite)
    trace = [
        Step(
            0,
            x.copy(),
            lin.f,
            grad_norm=_norm2(lin.g),
            info=_info(lin, step=None, lam=mu, gain=None, accepted=None, nu=nu),
        )
    ]

    def done(converged: bool, message: str, n_iter: int) -> Result:
        return Result(name, x, lin.f, converged, message, n_iter, res.n, jac.n, 0, trace=trace)

    # k = max_iter + 1 only runs the stopping tests at the last iterate (no evaluations).
    for k in range(1, max_iter + 2):
        if _gtol_met(lin, gtol):
            return done(True, _msg_gtol(lin), k - 1)
        if lin.f == 0.0:
            return done(True, _MSG_ZERO_F, k - 1)
        # h = −V diag(σ/(σ² + μ)) Uᵀ r solves (JᵀJ + μI) h = −Jᵀr (N&W eq. 10.38).
        denom = lin.s * lin.s + mu
        coef = np.divide(lin.s, denom, out=np.zeros_like(denom), where=denom > 0.0)
        h = -(lin.Vt.T @ (coef * (lin.U.T @ lin.r)))  # (n,)
        if _xtol_met(h, x, xtol):
            if blocked:
                return done(
                    False,
                    "trial steps reached points where the residual is not finite; the damped "
                    "step shrank below xtol without meeting gtol",
                    k - 1,
                )
            # NOTE: MNT Alg. 3.16 stops on a small h unconditionally. We first ask why h is
            # small. The stop certifies convergence when the undamped Gauss–Newton step
            # p = −J⁺r is small too (‖p‖ ≤ 2·xtol·(‖x‖ + xtol)), or when its model predicts a
            # relative decrease ‖Jp‖²/‖r‖² ≤ 100·ε, at the rounding noise of f, which comparisons
            # of f cannot resolve (the rounding-limited stop). A looser √ε here let LM stop at
            # k = 0 on michaelis_menten with τ = 1 from x* + 2.5e-3·v (v the soft eigenvector),
            # 730× outside the √ε accuracy bound, although f − f* = 7.3e-6 ≫ 100·ε·f = 1.3e-11.
            # Otherwise h is small only because μ is large: μ₀ is set
            # at x0 and Nielsen's update lowers it by at most 3× per accepted step. Without
            # this check LM on exp_decay_fit from (0.1, −4) stopped "converged" at f = 5.65
            # with ‖g‖∞ = 119 (f* = 0.0071) and ‖Jp‖²/‖r‖² = 1.6e-4. After an accepted trial
            # we take the trial as usual (μ keeps falling); after a rejected one the large μ
            # made the decrease unresolvable, so we stop and report converged=False.
            p_gn = lin.gauss_newton_step()  # (n,)
            gn_ratio = lin.gauss_newton_ratio()
            if _xtol_met(0.5 * p_gn, x, xtol) or gn_ratio <= _F_NOISE_REL:
                return done(True, _msg_xtol(h, lin), k - 1)
            if last_rejected:
                return done(
                    False,
                    f"the damped step ‖h‖ = {_norm2(h):.3g} is below xtol only "
                    f"because the damping μ = {mu:.3g} is large (σ_min² = "
                    f"{float(lin.s[-1]) ** 2:.3g}); the Gauss–Newton model still predicts a "
                    f"relative decrease ‖Jp‖²/‖r‖² = {gn_ratio:.3g} with ‖p‖ = "
                    f"{_norm2(p_gn):.3g}, and ‖Jᵀr‖∞ = "
                    f"{float(np.max(np.abs(lin.g))):.3g} > gtol",
                    k - 1,
                )

        if k > max_iter:
            return done(False, f"reached max_iter={max_iter}", max_iter)

        mu_used = mu
        r_new = _evaluate(res, x + h)
        f_new = _half_sq(r_new) if _finite(r_new) else math.nan
        predicted = 0.5 * float(h @ (mu * h - lin.g))  # L(0) − L(h), MNT eq. 3.14
        gain = (lin.f - f_new) / predicted if predicted > 0.0 and math.isfinite(f_new) else None
        accepted = gain is not None and gain > 0.0
        blocked = (blocked or not math.isfinite(f_new)) and not accepted
        last_rejected = not accepted
        if accepted:
            x_new = x + h
            J_new = _evaluate(jac, x_new)
            if not _finite(J_new):
                return done(False, "Jacobian is not finite at the new iterate", k - 1)
            lin_new = _linearize(r_new, J_new)
            why = _breakdown(lin_new, "at the new iterate")
            if why:
                return done(False, why, k - 1)
            x = x_new
            lin = lin_new
            assert gain is not None
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3)
            nu = 2.0
        else:
            mu *= nu
            nu *= 2.0
        trace.append(
            Step(
                k,
                x.copy(),
                lin.f,
                grad_norm=_norm2(lin.g),
                step_size=_norm2(h) if accepted else 0.0,
                info=_info(lin, step=h, lam=mu_used, gain=gain, accepted=accepted, nu=nu),
            )
        )
        if not math.isfinite(mu) or not math.isfinite(nu):
            return done(False, "damping parameter μ overflowed (no acceptable step found)", k)
    raise AssertionError("unreachable: the loop returns at k = max_iter + 1")


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("gauss_newton", "rosenbrock_ls", {}),
    ("gauss_newton", "exp_decay_fit", {}),
    ("gauss_newton", "michaelis_menten", {}),
    ("gauss_newton", "circle_fit", {"line_search": "none"}),
    ("levenberg_marquardt", "rosenbrock_ls", {}),
    ("levenberg_marquardt", "exp_decay_fit", {}),
    ("levenberg_marquardt", "circle_fit", {}),
    ("levenberg_marquardt", "michaelis_menten", {}),
]
