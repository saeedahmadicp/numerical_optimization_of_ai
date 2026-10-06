"""Nonlinear conjugate gradient (CG) methods for min f(x), f: ℝⁿ → ℝ smooth.

Every method generates search directions

    d₀ = −g₀,        d_k = −g_k + β_k d_{k−1}   (k ≥ 1),        g_k = ∇f(x_k),

and steps x_{k+1} = x_k + α_k d_k with a line search (Nocedal & Wright (2006), Alg. 5.4). The
methods differ only in the formula for β_k. With y_{k−1} = g_k − g_{k−1}:

=====================  ==================================================================
``cg_fletcher_reeves``  β^FR = ‖g_k‖² / ‖g_{k−1}‖²                         (N&W eq. 5.41a)
``cg_polak_ribiere``    β^PR+ = max(g_kᵀy_{k−1} / ‖g_{k−1}‖², 0)           (N&W eq. 5.44, 5.45)
``cg_hestenes_stiefel`` β^HS = g_kᵀy_{k−1} / d_{k−1}ᵀy_{k−1}              (N&W eq. 5.46)
``cg_dai_yuan``         β^DY = ‖g_k‖² / d_{k−1}ᵀy_{k−1}                    (N&W eq. 5.49)
``cg_hager_zhang``      β̄^N = max(β^N, η_k),                              (Hager & Zhang 2005)
                        β^N = (y − 2d‖y‖²/dᵀy)ᵀg_k / dᵀy,  η_k = −1/(‖d‖ min(η, ‖g_{k−1}‖)),
                        with d = d_{k−1}, y = y_{k−1} and η = 0.01
=====================  ==================================================================

Restarts (d_k = −g_k, β_k = 0), tested in this order (N&W §5.2):

1. ``periodic``: n iterations since the last steepest-descent direction (n = dim x).
2. ``powell``: |g_kᵀg_{k−1}| ≥ ν‖g_k‖² with ν = 0.1 (N&W eq. 5.52): successive gradients are
   far from orthogonal, so the conjugacy that CG relies on is lost.
3. ``breakdown``: β_k is undefined (zero, negative or non-finite denominator ‖g_{k−1}‖² or
   d_{k−1}ᵀy_{k−1}; the Wolfe conditions make d_{k−1}ᵀy_{k−1} > 0, so in exact arithmetic this
   cannot occur with ``line_search="strong_wolfe"``), or the CG direction −g_k + β_k d_{k−1} or
   its slope g_kᵀd_k overflows float64.
4. ``not_descent``: the CG direction is not a descent direction, g_kᵀd_k ≥ 0.

Line search (``line_search``):
    ``strong_wolfe`` (default): N&W Alg. 3.5/3.6 with c₁ = 10⁻⁴ and c₂ = 0.1 (N&W §5.2: "c₂ =
    0.1" for nonlinear CG; Lemma 5.6 needs c₂ < ½ for Fletcher–Reeves to give descent). The first
    trial step is α = 1/‖g₀‖₂ at k = 0 (a unit-length step) and
    α = α_{k−1} g_{k−1}ᵀd_{k−1} / g_kᵀd_k afterwards (N&W §3.5, p. 59: the first-order change
    of f equals that of the previous step).
    ``exact_quadratic``: α_k = −g_kᵀd_k / d_kᵀ∇²f(x_k)d_k (N&W eq. 5.6), the exact line
    minimizer when f is quadratic. On a convex quadratic every β formula then equals β^FR and the
    method reproduces linear CG (N&W Alg. 5.2): it terminates in at most n iterations in exact
    arithmetic (N&W Thm. 5.1). It needs the Hessian and stops (not converged) when
    d_kᵀ∇²f d_k ≤ 0, or when the search rejects the step because f increased (f is then not
    quadratic along d_k).

Stopping test (converged): ‖g_k‖∞ ≤ ``gtol``, checked at every iterate. The default
``gtol = 1e-5`` (the value SciPy's CG uses) reflects the attainable accuracy: the line search
accepts a step by comparing values of f, which resolve decreases only down to ≈ ε|f|. Near a
minimizer x* with f(x*) ≠ 0 the decrease of a step falls below that level once
‖g‖ ≲ √(2ε|f(x*)|λ_max(∇²f(x*))) (≈ 8e-6 at the local minimizer (−0.6, −0.4) of
Goldstein–Price, where f = 30), and the search then fails; such a run is reported as not
converged, with ‖∇f‖∞ in the message. Failures (``converged=False``): a non-finite f or ∇f at
x₀ or at a new iterate, a failed line search, d_kᵀ∇²f d_k ≤ 0 with the exact line search,
``max_iter`` iterations, or a gradient whose squared norm ‖g_k‖₂² underflows (below the smallest
normal float64, ≈ 2.2e-308, i.e. ‖g_k‖₂ ≲ 1.5e-154) while ‖g_k‖∞ > ``gtol``, or overflows (above
≈ 1.8e308, i.e. ‖g_k‖₂ ≳ 1.3e154): β_k and the slope g_kᵀd_k that the line search tests are
then not representable (rescale f in that case). A line search that breaks down with an
arithmetic exception (step lengths at the limit of float64, e.g. α ≲ 1e-154) is reported the same
way; no method raises on such a breakdown.
‖g_k‖₂ itself is computed with a power-of-2 scaling (exact, and equal to the unscaled value
whenever no square underflows), so the reported norm is never 0 for g_k ≠ 0.

Trace: Step k holds x_k, f(x_k), ‖g_k‖₂ and ``step_size`` = α_{k−1} (``None`` at k = 0);
``n_iter == trace[-1].k`` = the number of line searches completed. Evaluation counts are exact:
``n_fev`` / ``n_gev`` include the line-search evaluations, ``n_hev`` counts the Hessians of the
exact line search (one per iteration). Without ``Problem.grad`` the gradient is formed by central
differences (``numopt.core.diff``) and its 2n evaluations of f are included in ``n_fev``; without
``Problem.hess`` the exact line search uses a central-difference Hessian, whose 2n gradient
evaluations are included in ``n_gev``.

A :class:`Problem` with ``dim == 1`` (also the one built from a bare callable with a one-entry x0)
follows the scalar convention of ``core.types``: f, f′ and f″ receive and return floats. It is run
as an n = 1 problem, so ``Result.x``, the trace iterates and the vector info keys have length 1.

Info keys (one Step per iterate; k = 0 is the start):
    direction: [n] | None — d_k, the search direction from x_k; None when the run stops at x_k
        before a direction is computed (convergence, non-finite values, gradient under- or
        overflow).
    beta: float | None — β_k used to form d_k (0.0 on a restart); None at k = 0 and when
        ``direction`` is None.
    beta_formula: float | None — the method's raw β formula at x_k, before the PR+ truncation or
        the Hager–Zhang lower bound η_k; None at k = 0, when ``direction`` is None, or when the
        formula is undefined (``restart = "breakdown"``).
    restart: str | None — why d_k = −g_k: "initial" (k = 0), "periodic", "powell", "breakdown"
        or "not_descent"; None when d_k is the CG direction.
    powell_ratio: float | None — |g_kᵀg_{k−1}| / ‖g_k‖², the quantity of Powell's restart test
        (restart when ≥ 0.1); None at k = 0 and when ``direction`` is None.
    descent: float | None — g_kᵀd_k (< 0 for a descent direction); None when ``direction`` is
        None.
    alpha: float | None — α_{k−1}, the accepted step length that produced x_k (None at k = 0).
    trials: [[alpha, f]] — every step length tried by the line search that produced x_k and
        f(x_{k−1} + α d_{k−1}) there, in order ([] at k = 0).
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
from ..core.types import Problem, Result, Step
from ..line_search.methods import LineSearchResult, search

Array = NDArray[np.float64]

#: Armijo constant c₁ (N&W p. 33).
C1 = 1e-4
#: Default curvature constant c₂ for CG (N&W §5.2, p. 125).
C2_DEFAULT = 0.1
#: Powell's restart threshold ν (N&W eq. 5.52).
POWELL_NU = 0.1
#: Hager–Zhang lower-bound constant η (Hager & Zhang 2005, eq. 1.6; CG_DESCENT uses 0.01).
HZ_ETA = 0.01
#: Smallest positive normal float64; ‖g‖₂² below it has underflowed (gradual underflow).
TINY = float(np.finfo(np.float64).tiny)
#: Largest step length the strong Wolfe search may try (the search reports "f may be unbounded
#: below along p" when φ still decreases there).
ALPHA_MAX = 1e10

LINE_SEARCHES = ("strong_wolfe", "exact_quadratic")
RULES = ("fletcher_reeves", "polak_ribiere", "hestenes_stiefel", "dai_yuan", "hager_zhang")

PARAMS = (
    ParamSpec(
        "gtol",
        1e-5,
        min=1e-14,
        max=1e-2,
        log=True,
        help="Stop when the gradient ‖∇f(x)‖∞ ≤ gtol. Below ≈ √(2ε|f⋆|λₘₐₓ) the f-based line "
        "search cannot certify a decrease and the run stops unconverged (10⁻⁵ = SciPy's "
        "default).",
    ),
    ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Iteration limit."),
    ParamSpec(
        "line_search",
        "strong_wolfe",
        kind="choice",
        choices=LINE_SEARCHES,
        help="Strong Wolfe search, or the exact step −gᵀd/dᵀ∇²f d (exact for quadratics).",
    ),
    ParamSpec(
        "c2",
        C2_DEFAULT,
        min=0.01,
        max=0.9,
        help="Strong Wolfe curvature constant c₂ (0.1 for CG; Fletcher–Reeves needs c₂ < ½).",
    ),
)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _resolve(
    problem: Problem | Callable[[Array], float], x0: ArrayLike | None
) -> tuple[Array, Counted, Counted, Counted]:
    """Return (x0, counted f, counted ∇f, counted ∇²f) for a Problem or a bare f(x).

    The driver works on vectors x ∈ ℝⁿ. A :class:`Problem` with ``dim == 1`` follows the scalar
    convention of ``core.types`` (f, f′, f″ take and return floats), so its callables receive
    ``float(x[0])`` and their results are lifted to shapes (1,) and (1, 1). This includes the
    Problem built from a bare callable with a one-entry x0 (``core.counting.vector_problem`` gives
    it ``dim == 1``), so that a direct call and ``numopt.minimize`` treat it the same way. A
    size-1 array returned by f is accepted as the value f(x).
    """
    prob = problem if isinstance(problem, Problem) else vector_problem(problem, x0=x0)
    scalar = prob.dim == 1
    x = start_point(prob, x0)

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
        n = x.size
        hess = Counted(lambda z: np.asarray(h_fn(arg(z)), dtype=np.float64).reshape(n, n))
    else:
        hess = Counted(lambda z: diff.hessian(grad, z))
    return x, f, grad, hess


def _value(v: Any) -> float:
    """f(x) as a float; a size-1 array is accepted, a larger one is invalid input."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"f(x) must return a scalar, got an array of shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _finite(*values: Any) -> bool:
    return all(bool(np.all(np.isfinite(v))) for v in values)


def _norm2(v: Array) -> float:
    """‖v‖₂ without underflow or overflow of the squares (Higham (2002), §27.8 / LAPACK dnrm2).

    v is scaled by 2⁻ᵉ with 2^(e−1) ≤ ‖v‖∞ < 2^e before squaring. Scaling by a power of 2 is
    exact, so the result equals ``np.linalg.norm(v)`` bit for bit whenever no square under- or
    overflows, and stays accurate (and > 0 for v ≠ 0) when ``np.linalg.norm`` would return 0.
    """
    vinf = float(np.max(np.abs(v))) if v.size else 0.0
    if not 0.0 < vinf < math.inf:
        return vinf  # 0, inf or nan
    _, e = math.frexp(vinf)
    return math.ldexp(float(np.linalg.norm(np.ldexp(v, -e))), e)


def _eval(fn: Counted, x: Array) -> Any:
    """Evaluate without floating-point warnings; callers check the result for finiteness."""
    with np.errstate(all="ignore"):
        return fn(x)


def _search(k: int, kind: str, *args: Any, **opts: Any) -> LineSearchResult | str:
    """``line_search.methods.search``, or a failure message when it raises ``ArithmeticError``."""
    # NOTE: the contract forbids raising on numerical breakdown. The shared zoom divides by h²
    # (``_quadratic_minimizer``), which underflows to 0 for brackets narrower than ≈ 1.5e-162; it
    # occurs when the step α‖d‖ is near the spacing of floats at x and α is below ≈ 1e-154
    # (f scaled by ≈ 2⁴⁹⁶). This guard turns any such arithmetic exception into a failed run.
    try:
        return search(kind, *args, **opts)
    except ArithmeticError as exc:
        return (
            f"line search broke down at iteration {k} ({type(exc).__name__}: {exc}); the step "
            "lengths are at the limit of float64 (rescale f)"
        )


def _beta(rule: str, g: Array, g_prev: Array, d_prev: Array) -> tuple[float | None, float | None]:
    """Return (β used, raw β formula) for ``rule``; (None, None) when the formula is undefined.

    ``g`` = g_k, ``g_prev`` = g_{k−1}, ``d_prev`` = d_{k−1}; y = g_k − g_{k−1}.
    """
    y = g - g_prev  # (n,)
    if rule in ("fletcher_reeves", "polak_ribiere"):
        den = float(g_prev @ g_prev)
        num = float(g @ g) if rule == "fletcher_reeves" else float(g @ y)
    else:
        den = float(d_prev @ y)
        if rule == "hestenes_stiefel":
            num = float(g @ y)
        elif rule == "dai_yuan":
            num = float(g @ g)
        else:  # hager_zhang: β^N = (y − 2d‖y‖²/dᵀy)ᵀg / dᵀy  (Hager & Zhang 2005, eq. 1.4)
            num = float(g @ y) - 2.0 * float(y @ y) * float(d_prev @ g) / den if den > 0.0 else 0.0
    if not (den > 0.0 and math.isfinite(den)):
        return None, None
    raw = num / den
    if not math.isfinite(raw):
        return None, None
    if rule == "polak_ribiere":
        return max(raw, 0.0), raw  # PR+ (N&W eq. 5.45)
    if rule == "hager_zhang":
        # η_k = −1/(‖d_{k−1}‖ min(η, ‖g_{k−1}‖)) (Hager & Zhang 2005, eq. 1.6).
        eta_k = -1.0 / (float(np.linalg.norm(d_prev)) * min(HZ_ETA, float(np.linalg.norm(g_prev))))
        return (max(raw, eta_k), raw) if math.isfinite(eta_k) else (raw, raw)
    return raw, raw


def _info(
    *,
    direction: Array | None,
    beta: float | None,
    beta_formula: float | None,
    restart: str | None,
    powell_ratio: float | None,
    descent: float | None,
    alpha: float | None,
    trials: list[list[float]],
) -> dict[str, Any]:
    return {
        "direction": None if direction is None else direction.tolist(),
        "beta": beta,
        "beta_formula": beta_formula,
        "restart": restart,
        "powell_ratio": powell_ratio,
        "descent": descent,
        "alpha": alpha,
        "trials": trials,
    }


# --------------------------------------------------------------------------------------
# The shared nonlinear CG driver (N&W Alg. 5.4 with a choice of β)
# --------------------------------------------------------------------------------------


def _nonlinear_cg(
    method: str,
    rule: str,
    problem: Problem | Callable[[Array], float],
    x0: ArrayLike | None,
    gtol: float,
    max_iter: int,
    line_search: str,
    c2: float,
) -> Result:
    if line_search not in LINE_SEARCHES:
        raise ValueError(f"line_search must be one of {LINE_SEARCHES}, got {line_search!r}")
    if not gtol >= 0.0:
        raise ValueError(f"gtol must be ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not C1 < c2 < 1.0:
        raise ValueError(f"c2 must lie in (c1, 1) = ({C1}, 1), got {c2}")
    max_iter = int(max_iter)

    x, f, grad, hess = _resolve(problem, x0)
    n = x.size
    fx = float(_eval(f, x))
    g: Array = _eval(grad, x) if math.isfinite(fx) else np.full(n, np.nan)
    trace: list[Step] = []
    n_restarts = 0

    def done(converged: bool, message: str, k: int) -> Result:
        return Result(
            method,
            x,
            fx,
            converged,
            message,
            k,
            f.n,
            grad.n,
            hess.n,
            trace=trace,
            extra={"n_restarts": n_restarts},
        )

    if not _finite(fx, g):
        info = _info(
            direction=None,
            beta=None,
            beta_formula=None,
            restart=None,
            powell_ratio=None,
            descent=None,
            alpha=None,
            trials=[],
        )
        gn = _norm2(g) if _finite(g) else None
        trace.append(Step(0, x.copy(), fx, grad_norm=gn, info=info))
        return done(False, "f or ∇f is not finite at x0", 0)

    # State carried between iterations (values at x_{k−1}).
    g_prev: Array | None = None
    d_prev: Array | None = None
    slope_prev = math.nan  # g_{k−1}ᵀd_{k−1}
    alpha_prev: float | None = None
    trials: list[list[float]] = []
    since_restart = 0  # iterations since the last steepest-descent direction
    k = 0
    while True:
        gnorm = _norm2(g)
        ginf = float(np.max(np.abs(g)))
        gnorm_sq = gnorm * gnorm  # ‖g_k‖₂², = −g_kᵀd_k for d_k = −g_k (may over/underflow)
        underflow = gnorm_sq < TINY
        overflow = not math.isfinite(gnorm_sq)
        if ginf <= gtol or underflow or overflow:
            info = _info(
                direction=None,
                beta=None,
                beta_formula=None,
                restart=None,
                powell_ratio=None,
                descent=None,
                alpha=alpha_prev,
                trials=trials,
            )
            trace.append(Step(k, x.copy(), fx, gnorm, alpha_prev, info))
            if ginf <= gtol:
                return done(True, f"gradient ‖∇f‖∞ = {ginf:.3g} ≤ gtol", k)
            # NOTE: not in N&W Alg. 5.4, which assumes exact arithmetic. ‖g_k‖₂² is the
            # denominator of Powell's test and of β^FR/β^PR, and equals −g_kᵀd_k for d_k = −g_k.
            # Below 2.2e-308 or above 1.8e308 none of them is representable, and the line search
            # needs a finite slope g_kᵀd_k < 0, so the iteration cannot continue meaningfully.
            if overflow:
                return done(
                    False,
                    f"‖∇f‖₂ = {gnorm:.3g} is so large that ‖∇f‖₂² overflows: β and the slope ∇fᵀd "
                    "cannot be formed in float64; rescale f",
                    k,
                )
            return done(
                False,
                f"‖∇f‖₂ = {gnorm:.3g} is so small that ‖∇f‖₂² underflows (‖∇f‖∞ = {ginf:.3g} > "
                "gtol): β and the slope ∇fᵀd cannot be formed in float64; rescale f",
                k,
            )

        # ---- search direction d_k ------------------------------------------------------
        beta: float | None = None
        beta_formula: float | None = None
        powell_ratio: float | None = None
        restart: str | None = None
        if g_prev is None or d_prev is None:
            restart = "initial"
            d = -g
        else:
            # Products of large finite numbers may overflow to ±inf; each one is tested below.
            with np.errstate(over="ignore", invalid="ignore"):
                beta, beta_formula = _beta(rule, g, g_prev, d_prev)
                powell_ratio = abs(float(g @ g_prev)) / gnorm_sq  # inf ≥ ν restarts
                d = -g
                if since_restart + 1 >= n:
                    restart = "periodic"
                elif powell_ratio >= POWELL_NU:
                    restart = "powell"
                elif beta is None:
                    restart = "breakdown"
                else:
                    d = -g + beta * d_prev
                    gd = float(g @ d)
                    if not (_finite(d) and math.isfinite(gd)):
                        # NOTE: β_k d_{k−1} or g_kᵀd_k overflowed: the CG direction is not
                        # representable, a breakdown like an undefined β_k.
                        restart = "breakdown"
                        d = -g
                    elif not gd < 0.0:
                        restart = "not_descent"
                        d = -g
            if restart is not None:
                beta = 0.0
                n_restarts += 1
        since_restart = 0 if restart is not None else since_restart + 1
        slope = float(g @ d)  # g_kᵀd_k < 0, finite: −‖g_k‖₂² on a restart, else tested above
        info = _info(
            direction=d,
            beta=beta,
            beta_formula=beta_formula,
            restart=restart,
            powell_ratio=powell_ratio,
            descent=slope,
            alpha=alpha_prev,
            trials=trials,
        )
        trace.append(Step(k, x.copy(), fx, gnorm, alpha_prev, info))
        if k == max_iter:
            return done(False, f"reached max_iter={max_iter}", k)

        # ---- line search along d_k --------------------------------------------------------
        with np.errstate(all="ignore"):
            if line_search == "exact_quadratic":
                H = _eval(hess, x)
                curvature = float(d @ (H @ d))
                if not math.isfinite(curvature):
                    return done(
                        False,
                        f"dᵀ∇²f d = {curvature:.3g} is not finite at iteration {k}: the exact line "
                        "search cannot form the step −gᵀd/dᵀ∇²f d (rescale f)",
                        k,
                    )
                if not curvature > 0.0:
                    return done(
                        False,
                        f"dᵀ∇²f d = {curvature:.3g} ≤ 0 at iteration {k}: the exact line search "
                        "needs positive curvature along d (f is not a convex quadratic)",
                        k,
                    )
                ls = _search(k, "exact_quadratic", f, grad, x, d, f0=fx, g0=g, hess=H)
            else:
                if k == 0:
                    alpha0 = 1.0 / gnorm
                else:
                    assert alpha_prev is not None
                    alpha0 = alpha_prev * slope_prev / slope
                if not (math.isfinite(alpha0) and alpha0 > 0.0):
                    alpha0 = 1.0
                alpha0 = min(alpha0, ALPHA_MAX)
                ls = _search(
                    k,
                    "strong_wolfe",
                    f,
                    grad,
                    x,
                    d,
                    f0=fx,
                    g0=g,
                    alpha0=alpha0,
                    c1=C1,
                    c2=c2,
                    alpha_max=ALPHA_MAX,
                )
        if isinstance(ls, str):
            return done(False, ls, k)
        trials = [[float(a), float(v)] for a, v in ls.trials]
        if not ls.success:
            ginf_now = float(np.max(np.abs(g)))
            return done(
                False,
                f"line search failed at iteration {k} (‖∇f‖∞ = {ginf_now:.3g}): {ls.message}",
                k,
            )

        x_new = x + ls.alpha * d
        f_new = float(ls.f_new)
        g_new = ls.g_new if ls.g_new is not None else _eval(grad, x_new)
        if not _finite(f_new, g_new):
            return done(False, f"f or ∇f is not finite at the new iterate (iteration {k})", k)
        g_prev, d_prev, slope_prev, alpha_prev = g, d, slope, float(ls.alpha)
        x, fx, g = x_new, f_new, np.asarray(g_new, dtype=np.float64)
        k += 1


# --------------------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------------------

_COMMON_REFS = (
    "Nocedal & Wright (2006), Numerical Optimization, Alg. 5.4 and §5.2",
    "Nocedal & Wright (2006), Alg. 3.5–3.6 (strong Wolfe line search)",
)


@register(
    id="cg_fletcher_reeves",
    family="unconstrained",
    name="Conjugate gradient (Fletcher–Reeves)",
    params=PARAMS,
    needs=("f", "grad"),
    order="linear (n-step quadratic with restarts)",
    summary="Steepest descent plus a multiple of the previous direction, β = ‖g_k‖²/‖g_{k−1}‖².",
    references=("Fletcher & Reeves (1964), Comput. J. 7, 149–154", *_COMMON_REFS),
)
def cg_fletcher_reeves(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-5,
    max_iter: int = 1000,
    line_search: str = "strong_wolfe",
    c2: float = C2_DEFAULT,
) -> Result:
    """Fletcher–Reeves nonlinear CG (N&W Alg. 5.4, eq. 5.41a).

    d_k = −g_k + β_k d_{k−1} with β^FR = ‖g_k‖²/‖g_{k−1}‖². With a strong Wolfe search and
    c₂ < ½ every direction is a descent direction (N&W Lemma 5.6), and with restarts the method
    is globally convergent (N&W Thm. 5.7). Restarts: periodic (every n), Powell (|g_kᵀg_{k−1}| ≥
    0.1‖g_k‖²), breakdown, or a non-descent direction (see the module docstring).

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, a failed line
    search, ``max_iter``, or an under- or overflowing ‖∇f‖₂² (see the module docstring).
    """
    return _nonlinear_cg(
        "cg_fletcher_reeves", "fletcher_reeves", problem, x0, gtol, max_iter, line_search, c2
    )


@register(
    id="cg_polak_ribiere",
    family="unconstrained",
    name="Conjugate gradient (Polak–Ribière+)",
    params=PARAMS,
    needs=("f", "grad"),
    order="linear (n-step quadratic with restarts)",
    summary="CG with β = max(g_kᵀ(g_k − g_{k−1})/‖g_{k−1}‖², 0): restarts by itself when stuck.",
    references=(
        "Polak & Ribière (1969), Rev. Française Inform. Rech. Opér. 16, 35–43",
        "Gilbert & Nocedal (1992), SIAM J. Optim. 2(1), 21–42 (PR+)",
        *_COMMON_REFS,
    ),
)
def cg_polak_ribiere(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-5,
    max_iter: int = 1000,
    line_search: str = "strong_wolfe",
    c2: float = C2_DEFAULT,
) -> Result:
    """Polak–Ribière+ nonlinear CG (N&W eq. 5.44–5.45).

    β^PR+ = max(g_kᵀ(g_k − g_{k−1})/‖g_{k−1}‖², 0). When a step makes little progress,
    g_k ≈ g_{k−1}, so β^PR ≈ 0 and the method restarts along −g_k by itself; the truncation at 0
    gives global convergence under the strong Wolfe conditions with a sufficient-descent
    safeguard (Gilbert & Nocedal 1992). The restart rules of the module docstring also apply.

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, a failed line
    search, ``max_iter``, or an under- or overflowing ‖∇f‖₂² (see the module docstring).
    """
    return _nonlinear_cg(
        "cg_polak_ribiere", "polak_ribiere", problem, x0, gtol, max_iter, line_search, c2
    )


@register(
    id="cg_hestenes_stiefel",
    family="unconstrained",
    name="Conjugate gradient (Hestenes–Stiefel)",
    params=PARAMS,
    needs=("f", "grad"),
    order="linear (n-step quadratic with restarts)",
    summary="CG with β = g_kᵀy/d_{k−1}ᵀy, y = g_k − g_{k−1}: d_k is conjugate to d_{k−1} "
    "with respect to the average Hessian.",
    references=(
        "Hestenes & Stiefel (1952), J. Res. Nat. Bur. Standards 49(6), 409–436",
        *_COMMON_REFS,
    ),
)
def cg_hestenes_stiefel(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-5,
    max_iter: int = 1000,
    line_search: str = "strong_wolfe",
    c2: float = C2_DEFAULT,
) -> Result:
    """Hestenes–Stiefel nonlinear CG (N&W eq. 5.46).

    β^HS = g_kᵀy_{k−1}/d_{k−1}ᵀy_{k−1}. It makes d_kᵀḠ d_{k−1} = 0 for the average Hessian
    Ḡ = ∫₀¹ ∇²f(x_{k−1} + tα_{k−1}d_{k−1}) dt (N&W eq. 5.47). β^HS can be negative and the
    direction need not be a descent direction; the restart rules of the module docstring restore
    −g_k in that case.

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, a failed line
    search, ``max_iter``, or an under- or overflowing ‖∇f‖₂² (see the module docstring).
    """
    return _nonlinear_cg(
        "cg_hestenes_stiefel", "hestenes_stiefel", problem, x0, gtol, max_iter, line_search, c2
    )


@register(
    id="cg_dai_yuan",
    family="unconstrained",
    name="Conjugate gradient (Dai–Yuan)",
    params=PARAMS,
    needs=("f", "grad"),
    order="linear (n-step quadratic with restarts)",
    summary="CG with β = ‖g_k‖²/d_{k−1}ᵀy: a descent direction under any Wolfe line search.",
    references=("Dai & Yuan (1999), SIAM J. Optim. 10(1), 177–182", *_COMMON_REFS),
)
def cg_dai_yuan(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-5,
    max_iter: int = 1000,
    line_search: str = "strong_wolfe",
    c2: float = C2_DEFAULT,
) -> Result:
    """Dai–Yuan nonlinear CG (Dai & Yuan 1999; N&W eq. 5.49).

    β^DY = ‖g_k‖²/d_{k−1}ᵀy_{k−1}. Under the (weak) Wolfe conditions d_{k−1}ᵀy_{k−1} > 0 and
    g_kᵀd_k = β^DY g_{k−1}ᵀd_{k−1} < 0, so every direction is a descent direction (Dai & Yuan
    1999, Thm. 2.1). The restart rules of the module docstring also apply.

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, a failed line
    search, ``max_iter``, or an under- or overflowing ‖∇f‖₂² (see the module docstring).
    """
    return _nonlinear_cg("cg_dai_yuan", "dai_yuan", problem, x0, gtol, max_iter, line_search, c2)


@register(
    id="cg_hager_zhang",
    family="unconstrained",
    name="Conjugate gradient (Hager–Zhang)",
    params=PARAMS,
    needs=("f", "grad"),
    order="linear (n-step quadratic with restarts)",
    summary="The CG_DESCENT direction: Hestenes–Stiefel plus a correction that guarantees "
    "gᵀd ≤ −⅞‖g‖² for any line search.",
    references=(
        "Hager & Zhang (2005), SIAM J. Optim. 16(1), 170–192, eq. 1.4–1.6",
        "Hager & Zhang (2006), Pacific J. Optim. 2(1), 35–58 (survey)",
        *_COMMON_REFS,
    ),
)
def cg_hager_zhang(
    problem: Problem | Callable[[Array], float],
    *,
    x0: ArrayLike | None = None,
    gtol: float = 1e-5,
    max_iter: int = 1000,
    line_search: str = "strong_wolfe",
    c2: float = C2_DEFAULT,
) -> Result:
    """Hager–Zhang (CG_DESCENT) nonlinear CG (Hager & Zhang 2005, eq. 1.4–1.6).

    With d = d_{k−1}, y = y_{k−1}:
        β^N = (y − 2d‖y‖²/dᵀy)ᵀg_k / dᵀy,     η_k = −1/(‖d‖ min(η, ‖g_{k−1}‖)),  η = 0.01,
        β̄^N = max(β^N, η_k),                d_k = −g_k + β̄^N d.
    For β^N, g_kᵀd_k ≤ −⅞‖g_k‖² whenever dᵀy ≠ 0 (Hager & Zhang 2005, Thm. 1.1); g_kᵀd_k is
    linear in β and equals −‖g_k‖² at β = 0, so the bound also holds for β̄^N ∈ [β^N, 0].

    Stopping test (converged): ‖∇f(x_k)‖∞ ≤ ``gtol``. Failures: non-finite values, a failed line
    search, ``max_iter``, or an under- or overflowing ‖∇f‖₂² (see the module docstring).
    """
    # NOTE: CG_DESCENT uses its own approximate-Wolfe line search and no Powell or periodic
    # restarts; here the shared strong Wolfe search (c₂ = 0.1) and restart rules of this module
    # are used so that the five β formulas can be compared on equal terms.
    return _nonlinear_cg(
        "cg_hager_zhang", "hager_zhang", problem, x0, gtol, max_iter, line_search, c2
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("cg_fletcher_reeves", "quadratic_ill", {}),
    ("cg_fletcher_reeves", "quadratic_bowl", {"line_search": "exact_quadratic"}),
    ("cg_polak_ribiere", "rosenbrock", {}),
    ("cg_hestenes_stiefel", "beale", {}),
    ("cg_dai_yuan", "himmelblau", {}),
    ("cg_hager_zhang", "rosenbrock", {}),
    ("cg_hager_zhang", "six_hump_camel", {}),
]
