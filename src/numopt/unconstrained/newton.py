"""Newton's method for unconstrained minimization of a twice-differentiable f: ℝⁿ → ℝ.

At the iterate x_k the quadratic model

    m_k(p) = f(x_k) + ∇f(x_k)ᵀp + ½ pᵀ∇²f(x_k) p

has the stationary point given by the Newton equations ∇²f(x_k) p = −∇f(x_k)
(Nocedal & Wright (2006), eq. 3.30; §2.2). The three methods differ in what they do with it:

``pure_newton``      Pure Newton: x_{k+1} = x_k + p_k (α = 1). Quadratic convergence near a
                     minimizer with ∇²f ≻ 0 (N&W Thm 3.5), but no globalization: it is attracted
                     to any stationary point (saddles, maxima) and it can diverge.
``damped_newton``    Line-search Newton: x_{k+1} = x_k + α_k p_k with a line search that tries
                     α = 1 first (N&W §3.3; Boyd & Vandenberghe (2004), Alg. 9.5). When the
                     Newton direction is not a descent direction (indefinite or singular
                     ∇²f), the steepest-descent direction −∇f is used for that iteration.
``modified_newton``  Line-search Newton with Hessian modification, N&W Alg. 3.2: the direction
                     solves (∇²f + τI) p = −∇f, with τ ≥ 0 found by N&W Alg. 3.3 (Cholesky with
                     added multiple of the identity), so p is always a descent direction.

Conventions shared by the three methods:

* **Linear algebra.** No inverse is formed. ``pure_newton`` and ``damped_newton`` solve the Newton
  equations with the spectral decomposition ∇²f = Q Λ Qᵀ (``numpy.linalg.eigh``):
  p = −Q Λ⁻¹ Qᵀ ∇f. The same decomposition gives the eigenvalues for the singularity test and
  for ``info["hess_eigs"]``. ∇²f is *numerically singular* when |λ|_min ≤ tol, with the
  eigenvalue tolerance tol defined below. ``modified_newton`` solves
  with the Cholesky factor L Lᵀ = ∇²f + τI by forward and back substitution (Golub & Van Loan
  (2013), Algs. 3.1.1–3.1.2).
* **Eigenvalue tolerance.** An eigenvalue with |λ| ≤ tol cannot be told apart from 0. tol
  depends on the source of ∇²f:

  - analytic ∇²f: tol = n·ε·|λ|_max (the ``numpy.linalg.matrix_rank`` tolerance; rounding
    in the eigendecomposition only);
  - central-difference ∇²f (``numopt.core.diff.hessian``, step h = ε^{1/3}·max(1, |x_i|)):
    tol = n·ε^{1/3}·max(1, |λ|_max), and max(1, |λ|_max, |f(x)|) when ∇f is a central
    difference too. A central difference of a central difference has the rounding error
    ε|f|/h² = ε^{1/3}|f| per entry (N&W §8.1), and a symmetric perturbation E moves every
    eigenvalue by at most ‖E‖₂ ≤ n·max_ij |E_ij| (Weyl's inequality). The floor 1 covers
    rounding in the intermediate terms of f and ∇f, which the values at x do not show.

  With the analytic tolerance, a central-difference ∇²f at a minimizer with a singular
  Hessian (f = (x + y)², every point of x + y = 0) shows the eigenvalue ±10⁻¹¹ instead of 0,
  which would claim a saddle point or a strict minimizer from noise alone.
* **Descent test** (``damped_newton``): p is accepted as a descent direction when
  cos θ = −∇fᵀp / (‖∇f‖‖p‖) > η = 1e-8 (computed with ∇f and p scaled by their largest
  entries, so that the norms cannot underflow). Zoutendijk's theorem (N&W Thm 3.2, eq. 3.12) needs
  cos θ bounded away from zero; for ∇²f ≻ 0 the Kantorovich bound cos θ ≥ 2√κ/(1 + κ) means
  the test only rejects Newton directions whose Hessian has κ(∇²f) > 4·10¹⁶.
* **Line search** (``damped_newton``, ``modified_newton``): ``numopt.line_search.methods.search``
  with α₀ = 1, default Armijo backtracking (N&W Alg. 3.1, c₁ = 1e-4, ρ = ½); the Wolfe and
  Goldstein searches can be selected.
* **Stopping test (converged).** ‖∇f(x_k)‖∞ ≤ ``gtol`` *and* ∇²f(x_k) has no eigenvalue below
  −tol, i.e. the second-order necessary conditions hold to the accuracy of ∇²f (N&W Thms 2.3,
  2.4). A point with ‖∇f‖∞ ≤ gtol and a clearly negative eigenvalue is not a minimizer; the
  method then stops with ``converged=False`` and classifies the point by λ_max: a maximizer
  when λ_max < −tol (∇²f ≺ 0, N&W Thm 2.4 applied to −f), a saddle point when λ_max > tol
  (∇²f indefinite), and "a saddle point or a maximizer" when |λ_max| ≤ tol (∇²f is negative
  semidefinite and singular, so second-order information cannot decide: −x² + y⁴ and
  −x² − y⁴ have the same ∇²f = diag(−2, 0) at 0). A positive *semi*definite ∇²f at the
  stopping point (λ_min ∈ [−tol, tol]) is reported as converged, and the message says that the
  second-order sufficient condition is not verified (for a finite-difference ∇²f: that λ_min
  lies within the accuracy of the estimate).
* **Failures** (``converged=False``): ``max_iter`` reached; a non-finite f, ∇f or ∇²f; a
  failed line search, or ∇fᵀp = 0 in floating point (∇f so small that the product
  underflows, e.g. with ``gtol = 0``; no line search can start); ``pure_newton`` only: a numerically singular ∇²f, or divergence
  ‖x_k‖∞ > 10⁸·max(1, ‖x_0‖∞) at an iterate that does not pass the stopping test (the stopping
  test runs first, so a minimizer far from the origin is still reported as converged);
  ``modified_newton`` only: 64 Cholesky attempts of Alg. 3.3
  with τ ≥ β failed (τ ≥ β·2⁶³).
* **Evaluation counts.** f, ∇f and ∇²f are evaluated at x_0 and at every accepted iterate
  (so n_hev = n_iter + 1 unless a failure stops the run first); line-search trials add their
  own f (and, for Wolfe searches, ∇f) evaluations, and a Wolfe search's ∇f at the accepted
  step is reused. Without an analytic gradient, ∇f is a central difference
  (``numopt.core.diff.gradient``) whose 2n f evaluations are counted in ``n_fev``; without an
  analytic Hessian, ∇²f is a central difference of ∇f (``numopt.core.diff.hessian``) whose 2n
  gradient evaluations are counted in ``n_gev``.

Trace: one Step per iterate; k = 0 is x_0. ``Step.step_size`` is the step length α_{k−1} that
produced x_k (``None`` at k = 0). ``n_iter == trace[-1].k``.

Info keys (every method, every step). Keys marked "incoming" describe the step x_{k−1} → x_k
and are ``None`` (``[]`` for ``trials``) at k = 0; the others describe the state at x_k:
    grad: [n]                 ∇f(x_k).
    hess: [[n]] | None        ∇²f(x_k) (the curvature of the quadratic model, for an ellipse
                              overlay); only for n ≤ 2, else None.
    hess_eigs: [n]            eigenvalues of ∇²f(x_k), ascending.
    direction: [n] | None     incoming: the search direction p_{k−1}.
    alpha: float | None       incoming: the step length α_{k−1} (1.0 for ``pure_newton``).
    trials: [[alpha, f]]      incoming: every (α, f(x_{k−1} + αp)) the line search evaluated, in
                              order ([] for ``pure_newton``, which takes α = 1 without a search).
    descent: bool | None      incoming: whether ∇f(x_{k−1})ᵀp_{k−1} < 0 (``pure_newton`` can step
                              uphill; for the other methods this is always True).

Additional info keys per method:
    damped_newton:
        direction_type: "newton" | "steepest" | None   incoming: which direction was used
                              ("steepest" when the Newton direction failed the descent test
                              or ∇²f was singular).
    modified_newton:
        tau: float | None     incoming: the shift τ ≥ 0 with ∇²f(x_{k−1}) + τI = LLᵀ
                              (0 when ∇²f was already numerically positive definite).
        chol_attempts: int | None  incoming: Cholesky factorizations tried by Alg. 3.3.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, as_vector
from ..line_search.methods import search

Array = NDArray[np.float64]

#: Machine epsilon of float64.
_EPS = float(np.finfo(np.float64).eps)
#: A finite-difference ∇f with |∇fᵀp| below this many ulps of f cannot make progress.
_FD_STALL = 10.0

#: Relative accuracy ε^{1/3} of a central-difference ∇²f (see ``_eig_tol``).
_FD_HESS_REL = _EPS ** (1.0 / 3.0)

#: Descent test cos θ > η for ``damped_newton`` (see the module docstring).
_DESCENT_COS = 1e-8

#: ``pure_newton`` reports divergence when ‖x_k‖∞ > _DIVERGENCE_FACTOR · max(1, ‖x_0‖∞).
_DIVERGENCE_FACTOR = 1e8

#: N&W Alg. 3.3 gives up after this many failed factorizations with τ ≥ β.
_MAX_SHIFT_DOUBLINGS = 64

LINE_SEARCHES = ("backtracking", "strong_wolfe", "weak_wolfe", "goldstein")

P_GTOL = ParamSpec(
    "gtol",
    1e-8,
    min=1e-14,
    max=1e-2,
    log=True,
    help="Stop when ‖∇f(x)‖∞ ≤ gtol (and ∇²f(x) has no negative eigenvalue).",
)
P_MAX_ITER = ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Iteration limit.")
P_LINE_SEARCH = ParamSpec(
    "line_search",
    "backtracking",
    kind="choice",
    choices=LINE_SEARCHES,
    help="Step-length rule; every search tries α = 1 (the full Newton step) first.",
)
P_BETA = ParamSpec(
    "beta",
    1e-3,
    min=1e-8,
    max=10.0,
    log=True,
    help="Alg. 3.3: smallest nonzero shift τ; τ doubles until ∇²f + τI has a Cholesky factor.",
)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


@dataclass
class _Oracle:
    """Counted f, ∇f and ∇²f (finite-difference fallbacks are counted through f and ∇f)."""

    f: Counted
    grad: Counted
    hess: Counted
    #: ∇f is a central difference of f (no analytic gradient).
    grad_fd: bool
    #: ∇²f is a central difference of ∇f (no analytic Hessian).
    hess_fd: bool


def _resolve(problem: Problem | Callable[..., Any], x0: Any) -> tuple[Array, _Oracle]:
    prob = vector_problem(problem, x0=x0) if not isinstance(problem, Problem) else problem
    x = start_point(prob, x0)
    f = Counted(prob.f)
    if prob.grad is not None:
        grad = Counted(prob.grad)
    else:
        grad = Counted(lambda z: diff.gradient(f, z))
    if prob.hess is not None:
        hess = Counted(prob.hess)
    else:
        hess = Counted(lambda z: diff.hessian(grad, z))
    return x, _Oracle(f, grad, hess, prob.grad is None, prob.hess is None)


def _evaluate_hessian(oracle: _Oracle, x: Array) -> Array:
    n = x.size
    H = np.asarray(oracle.hess(x), dtype=np.float64).reshape(n, n)
    # NOTE: symmetrize ∇²f once: the spectral solve and the Cholesky factorization assume a
    # symmetric matrix and read only one triangle; an analytic Hessian can be asymmetric in
    # its last bit.
    return 0.5 * (H + H.T)


def _eigs(H: Array) -> tuple[Array, Array] | None:
    """Spectral decomposition ∇²f = Q diag(λ) Qᵀ (ascending λ), or None when H is not finite."""
    if not finite(H):
        return None
    lam, Q = np.linalg.eigh(H)
    return lam, Q


def _eig_tol(lam: Array, fx: float, oracle: _Oracle) -> float:
    """The eigenvalue tolerance of the module docstring: |λ| ≤ tol is indistinguishable from 0.

    Analytic ∇²f: n·ε·|λ|_max. Central-difference ∇²f: n·ε^{1/3}·max(1, |λ|_max), with |f(x)|
    added to the max when ∇f is a central difference too.
    """
    n = lam.size
    lam_max = float(np.abs(lam).max())
    if not oracle.hess_fd:
        return n * _EPS * lam_max
    scale = max(1.0, lam_max, abs(fx) if oracle.grad_fd else 0.0)
    return n * _FD_HESS_REL * scale


def _is_singular(lam: Array, tol: float) -> bool:
    """|λ|_min ≤ tol (and ∇²f = 0 is singular for any tol)."""
    absl = np.abs(lam)
    return float(absl.max()) == 0.0 or float(absl.min()) <= tol


def _newton_direction(lam: Array, Q: Array, g: Array) -> Array:
    """p = −Q diag(1/λ) Qᵀ g, the solution of ∇²f p = −∇f (N&W eq. 3.30)."""
    return -(Q @ ((Q.T @ g) / lam))


def _cos_angle(g: Array, p: Array) -> float:
    """cos θ = −gᵀp / (‖g‖‖p‖); NaN when either vector is zero or not finite.

    The vectors are first scaled by their largest entries (cos θ does not change), so that
    ‖g‖ and ‖p‖ do not underflow to 0 when ∇f is tiny but nonzero (for example at gtol = 0).
    """
    g_max = float(np.max(np.abs(g)))
    p_max = float(np.max(np.abs(p)))
    if not (math.isfinite(g_max) and math.isfinite(p_max) and g_max > 0.0 and p_max > 0.0):
        return math.nan
    g_hat, p_hat = g / g_max, p / p_max
    return -float(g_hat @ p_hat) / float(np.linalg.norm(g_hat) * np.linalg.norm(p_hat))


def _second_order(lam: Array, tol: float, fd_hessian: bool) -> tuple[bool, str]:
    """Classify a stationary point by the eigenvalues of ∇²f (N&W Thms 2.3–2.4).

    Eigenvalues with |λ| ≤ ``tol`` count as 0; ``fd_hessian`` says that ∇²f is a central
    difference (the messages then say so).
    """
    lam_min = float(lam[0])
    source = "the finite-difference ∇²f" if fd_hessian else "∇²f"
    if lam_min < -tol:
        lam_top = float(lam[-1])
        if lam_top < -tol:
            # ∇²f ≺ 0: the second-order sufficient condition for a strict local maximizer.
            kind = "a maximizer"
        elif lam_top > tol:
            # Eigenvalues of both signs: f increases along one eigenvector, decreases along another.
            kind = "a saddle point"
        else:
            # ∇²f ⪯ 0 and singular: the higher-order terms decide (−x² ± y⁴ at 0), not ∇²f.
            kind = (
                # (+ 0.0 turns a signed zero −0 into 0 for the message.)
                f"a saddle point or a maximizer (λ_max = {lam_top + 0.0:.3g}: ∇²f is singular, so "
                "the second-order test cannot decide)"
            )
        return False, (
            f"stopped at {kind}, not a minimizer: {source} has the eigenvalue "
            f"λ_min = {lam_min:.3g} < 0"
        )
    if lam_min <= tol:
        if fd_hessian:
            return True, (
                f"the finite-difference ∇²f has λ_min = {lam_min:.3g}, within its accuracy "
                f"±{tol:.3g} of 0: ∇²f is positive semidefinite to that accuracy, and the "
                "second-order sufficient condition is not verified"
            )
        return True, (
            "∇²f is positive semidefinite but numerically singular "
            f"(λ_min = {lam_min:.3g}): the second-order sufficient condition is not verified"
        )
    return True, f"{source} is positive definite (λ_min = {lam_min:.3g}): a strict local minimizer"


def _hess_info(H: Array, lam: Array | None) -> dict[str, Any]:
    return {
        "hess": H.tolist() if H.shape[0] <= 2 else None,
        "hess_eigs": lam.tolist() if lam is not None else [math.nan] * H.shape[0],
    }


def _incoming_none() -> dict[str, Any]:
    return {"direction": None, "alpha": None, "trials": [], "descent": None}


def _result(
    method: str,
    x: Array,
    fx: float,
    converged: bool,
    message: str,
    k: int,
    oracle: _Oracle,
    trace: list[Step],
) -> Result:
    return Result(
        method,
        x,
        fx,
        converged,
        message,
        k,
        n_fev=oracle.f.n,
        n_gev=oracle.grad.n,
        n_hev=oracle.hess.n,
        trace=trace,
    )


def _validate(gtol: float, max_iter: int, line_search: str | None) -> None:
    if not (math.isfinite(gtol) and gtol >= 0.0):
        raise ValueError(f"gtol must be finite and ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if line_search is not None and line_search not in LINE_SEARCHES:
        raise ValueError(f"unknown line_search {line_search!r}; expected one of {LINE_SEARCHES}")


def _start(
    method: str, problem: Problem | Callable[..., Any], x0: Any
) -> tuple[Array, float, Array, Array, tuple[Array, Array] | None, _Oracle]:
    x, oracle = _resolve(problem, x0)
    fx = float(oracle.f(x))
    g = as_vector(oracle.grad(x))
    if not finite(fx, g):
        raise ValueError(f"{method}: f and ∇f must be finite at x0; got f={fx}, ∇f={g}")
    H = _evaluate_hessian(oracle, x)
    return x, fx, g, H, _eigs(H), oracle


def _check_stop(
    g: Array, fx: float, eig: tuple[Array, Array] | None, gtol: float, oracle: _Oracle
) -> tuple[bool, str] | None:
    """The shared stopping test; returns (converged, message) or None to continue."""
    gnorm = float(np.max(np.abs(g)))
    if gnorm > gtol:
        return None
    head = f"‖∇f‖∞ = {gnorm:.3g} ≤ gtol"
    if eig is None:
        return False, f"{head}, but ∇²f is not finite there"
    lam = eig[0]
    ok, why = _second_order(lam, _eig_tol(lam, fx, oracle), oracle.hess_fd)
    return ok, f"{head}; {why}"


def _slope(g: Array, p: Array) -> float:
    """∇fᵀp, the slope φ'(0) of the line search; ±inf or NaN (without a warning) on overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(g @ p)


def _slope_failure(slope: float) -> str:
    """Why no step can be tested when ∇fᵀp is not a finite negative number."""
    if math.isfinite(slope):
        return "∇fᵀp underflowed to 0, so no step can be tested"
    return f"∇fᵀp overflowed to {slope}, so no step can be tested"


def _line_search_failure(k: int, g: Array, p: Array, fx: float, why: str) -> str:
    """Failure message that compares the predicted decrease with the rounding level of f."""
    return (
        f"line search failed at iteration {k}: {why}; predicted decrease "
        f"|∇fᵀp| = {abs(_slope(g, p)):.3g} vs rounding level of f "
        f"ε·max(1, |f|) = {_EPS * max(1.0, abs(fx)):.3g}"
    )


# --------------------------------------------------------------------------------------
# Pure Newton
# --------------------------------------------------------------------------------------


@register(
    id="pure_newton",
    family="unconstrained",
    name="Newton's method",
    params=(P_GTOL, P_MAX_ITER),
    needs=("f", "grad", "hess"),
    order="quadratic (near a minimizer with ∇²f ≻ 0)",
    summary="Jump to the stationary point of the local quadratic model: solve ∇²f p = −∇f.",
    references=(
        "Nocedal & Wright (2006), §2.2 and eq. 3.30; Theorem 3.5",
        "Boyd & Vandenberghe (2004), §9.5",
    ),
)
def pure_newton(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 100,
) -> Result:
    """Pure Newton's method: x_{k+1} = x_k + p_k with ∇²f(x_k) p_k = −∇f(x_k) (N&W eq. 3.30).

    No line search (α = 1) and no Hessian modification. Near a minimizer x* with ∇²f(x*) ≻ 0
    and a Lipschitz Hessian the iterates converge quadratically (N&W Thm 3.5); on a strictly
    convex quadratic one step lands on the minimizer. Far from x*, or where ∇²f is indefinite,
    the step can go uphill, reach a saddle point or a maximizer, or diverge; this is reported
    honestly, never corrected.

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol`` and ∇²f(x_k) has no negative eigenvalue
    (beyond rounding). Stops with ``converged=False`` at a saddle point or maximizer, when
    ∇²f(x_k) is numerically singular (|λ|_min ≤ tol, module docstring), on non-finite values, on
    divergence (‖x_k‖∞ > 10⁸·max(1, ‖x_0‖∞) at an iterate that fails the stopping test), or at
    ``max_iter``.
    """
    _validate(gtol, max_iter, None)
    method = "pure_newton"
    x, fx, g, H, eig, oracle = _start(method, problem, x0)
    x_scale = max(1.0, float(np.max(np.abs(x))))
    lam = eig[0] if eig is not None else None
    info = {"grad": g.tolist(), **_hess_info(H, lam), **_incoming_none()}
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), None, info)]
    k = 0
    while True:
        stop = _check_stop(g, fx, eig, gtol, oracle)
        if stop is not None:
            return _result(method, x, fx, stop[0], stop[1], k, oracle, trace)
        # The divergence test runs only after the stopping test: an iterate far from x_0 that
        # is a minimizer (‖∇f‖∞ ≤ gtol, ∇²f ⪰ 0) has converged, not diverged.
        x_norm = float(np.max(np.abs(x)))
        if x_norm > _DIVERGENCE_FACTOR * x_scale:
            msg = (
                f"the iterates diverge: ‖x_{k}‖∞ = {x_norm:.3g} > "
                f"{_DIVERGENCE_FACTOR:.0e}·max(1, ‖x_0‖∞)"
            )
            return _result(method, x, fx, False, msg, k, oracle, trace)
        if eig is None:
            msg = f"∇²f is not finite at iteration {k}"
            return _result(method, x, fx, False, msg, k, oracle, trace)
        lam, Q = eig
        tol = _eig_tol(lam, fx, oracle)
        if _is_singular(lam, tol):
            source = "the finite-difference ∇²f" if oracle.hess_fd else "∇²f"
            msg = (
                f"{source}(x_{k}) is numerically singular (|λ|_min = "
                f"{float(np.abs(lam).min()):.3g} ≤ tol = {tol:.3g}); the Newton step is undefined"
            )
            return _result(method, x, fx, False, msg, k, oracle, trace)
        if k == max_iter:
            return _result(method, x, fx, False, f"reached max_iter={max_iter}", k, oracle, trace)
        p = _newton_direction(lam, Q, g)
        descent = bool(float(g @ p) < 0.0)
        k += 1
        x = x + p
        fx = float(oracle.f(x))
        g = as_vector(oracle.grad(x))
        H = _evaluate_hessian(oracle, x) if finite(fx, g) else np.full((x.size, x.size), np.nan)
        eig = _eigs(H)
        info = {
            "grad": g.tolist(),
            **_hess_info(H, eig[0] if eig is not None else None),
            "direction": p.tolist(),
            "alpha": 1.0,
            "trials": [],
            "descent": descent,
        }
        gnorm = float(np.linalg.norm(g))
        trace.append(Step(k, x.copy(), fx, gnorm, 1.0, info))
        if not finite(fx, g):
            msg = f"f or ∇f is not finite at iteration {k}: the iterates diverged"
            return _result(method, x, fx, False, msg, k, oracle, trace)


# --------------------------------------------------------------------------------------
# Line-search Newton methods
# --------------------------------------------------------------------------------------


def _cholesky_shift(A: Array, beta: float) -> tuple[Array | None, float, int]:
    """N&W Alg. 3.3, Cholesky with added multiple of the identity.

        τ₀ = 0 if min_i a_ii > 0, else −min_i a_ii + β
        for k = 0, 1, 2, ...: try LLᵀ = A + τ_k I; on success stop; else τ_{k+1} = max(2τ_k, β)

    Returns (L, τ, attempts); L is None after 64 failed attempts with τ ≥ β.
    """
    n = A.shape[0]
    a_min = float(np.min(np.diag(A)))
    tau = 0.0 if a_min > 0.0 else -a_min + beta
    eye = np.eye(n)
    attempts = 0
    doublings = 0
    while True:
        attempts += 1
        try:
            L = np.linalg.cholesky(A + tau * eye)
        except np.linalg.LinAlgError:
            L = None
        if L is not None and finite(L):
            return L, tau, attempts
        if tau >= beta:
            doublings += 1
            if doublings >= _MAX_SHIFT_DOUBLINGS:
                return None, tau, attempts
        tau = max(2.0 * tau, beta)


def _cholesky_solve(L: Array, b: Array) -> Array:
    """Solve L Lᵀ z = b by forward then back substitution (Golub & Van Loan, Algs 3.1.1–3.1.2)."""
    n = b.size
    w = np.empty(n)
    z = np.empty(n)
    # A barely positive definite ∇²f + τI can overflow here; the caller rejects a non-finite p.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(n):
            w[i] = (b[i] - float(L[i, :i] @ w[:i])) / L[i, i]
        for i in range(n - 1, -1, -1):
            z[i] = (w[i] - float(L[i + 1 :, i] @ z[i + 1 :])) / L[i, i]
    return z


def _line_search_newton(
    method: str,
    problem: Problem | Callable[..., Any],
    x0: Any,
    gtol: float,
    max_iter: int,
    line_search: str,
    direction: Callable[
        [Array, Array, tuple[Array, Array], float], tuple[Array, dict[str, Any]] | str
    ],
) -> Result:
    """The loop shared by ``damped_newton`` and ``modified_newton`` (N&W Alg. 3.2 pattern).

    ``direction(H, g, (λ, Q), tol)`` returns (p, extra incoming info) with p a descent
    direction, or a failure message; ``tol`` is the eigenvalue tolerance (:func:`_eig_tol`).
    """
    x, fx, g, H, eig, oracle = _start(method, problem, x0)
    lam = eig[0] if eig is not None else None
    extra_none = {
        "damped_newton": {"direction_type": None},
        "modified_newton": {"tau": None, "chol_attempts": None},
    }[method]
    info = {"grad": g.tolist(), **_hess_info(H, lam), **_incoming_none(), **extra_none}
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), None, info)]
    k = 0
    while True:
        stop = _check_stop(g, fx, eig, gtol, oracle)
        if stop is not None:
            return _result(method, x, fx, stop[0], stop[1], k, oracle, trace)
        if eig is None:
            msg = f"∇²f is not finite at iteration {k}"
            return _result(method, x, fx, False, msg, k, oracle, trace)
        if k == max_iter:
            return _result(method, x, fx, False, f"reached max_iter={max_iter}", k, oracle, trace)
        out = direction(H, g, eig, _eig_tol(eig[0], fx, oracle))
        if isinstance(out, str):
            return _result(method, x, fx, False, out, k, oracle, trace)
        p, extra = out
        slope = _slope(g, p)
        if not (math.isfinite(slope) and slope < 0.0):
            # Every direction passed a descent test (cos θ > η, −∇f, or −‖L⁻¹∇f‖² < 0), so a
            # non-finite ∇fᵀp overflowed and ∇fᵀp ≥ 0 underflowed (∇f near the underflow
            # threshold). search() needs a finite ∇fᵀp < 0, so either one ends the run.
            why = _slope_failure(slope)
            return _result(
                method, x, fx, False, _line_search_failure(k + 1, g, p, fx, why), k, oracle, trace
            )
        # The line search counts its calls through the Counted f and ∇f (oracle totals).
        ls = search(line_search, oracle.f, oracle.grad, x, p, f0=fx, g0=g, alpha0=1.0)
        if not ls.success:
            floor = _FD_STALL * _EPS * max(1.0, abs(fx))
            if oracle.grad_fd and abs(float(g @ p)) <= floor:
                # NOTE: with a finite-difference ∇f, a predicted decrease at the rounding level
                # of f means x is stationary to the accuracy of the gradient estimate: no step
                # can lower f measurably, so ‖∇f‖ ≤ gtol is unattainable, not violated.
                ok, why = _second_order(eig[0], _eig_tol(eig[0], fx, oracle), oracle.hess_fd)
                msg = (
                    f"stationary to the accuracy of the finite-difference ∇f at iteration {k}: "
                    f"predicted decrease |∇fᵀp| = {abs(float(g @ p)):.3g} ≤ {floor:.3g} "
                    f"(the rounding level of f); {why}"
                )
                return _result(method, x, fx, ok, msg, k, oracle, trace)
            msg = _line_search_failure(k + 1, g, p, fx, ls.message)
            return _result(method, x, fx, False, msg, k, oracle, trace)
        k += 1
        x = x + ls.alpha * p
        fx = float(ls.f_new)
        g = as_vector(ls.g_new) if ls.g_new is not None else as_vector(oracle.grad(x))
        H = _evaluate_hessian(oracle, x) if finite(g) else np.full((x.size, x.size), np.nan)
        eig = _eigs(H)
        info = {
            "grad": g.tolist(),
            **_hess_info(H, eig[0] if eig is not None else None),
            "direction": p.tolist(),
            "alpha": ls.alpha,
            "trials": [[a, v] for a, v in ls.trials],
            "descent": True,
            **extra,
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), ls.alpha, info))
        if not finite(fx, g):
            msg = f"f or ∇f is not finite at iteration {k}"
            return _result(method, x, fx, False, msg, k, oracle, trace)


@register(
    id="damped_newton",
    family="unconstrained",
    name="Damped Newton (line search)",
    params=(P_GTOL, P_MAX_ITER, P_LINE_SEARCH),
    needs=("f", "grad", "hess"),
    order="quadratic (near a minimizer with ∇²f ≻ 0, where α = 1 is accepted)",
    summary="Newton direction with a line search; falls back to −∇f where Newton points uphill.",
    references=(
        "Nocedal & Wright (2006), §3.3 (Newton's method with line search), Alg. 3.1",
        "Boyd & Vandenberghe (2004), Alg. 9.5 (damped Newton)",
    ),
)
def damped_newton(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 100,
    line_search: str = "backtracking",
) -> Result:
    """Damped (line-search) Newton: x_{k+1} = x_k + α_k p_k with the search trying α = 1 first.

    p_k is the Newton direction ∇²f(x_k) p_k = −∇f(x_k) (N&W eq. 3.30) when ∇²f(x_k) is not
    numerically singular and p_k passes the descent test cos θ_k > 10⁻⁸ (module docstring).
    Otherwise p_k = −∇f(x_k).

    # NOTE: the fallback to −∇f is this method's documented choice for an indefinite or
    # singular Hessian (Boyd & Vandenberghe's damped Newton assumes a convex f, where it never
    # triggers); ``modified_newton`` is the textbook alternative (N&W §3.4).

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol`` and ∇²f(x_k) has no negative eigenvalue
    (beyond rounding); ``converged=False`` at a saddle point or maximizer, on a failed line
    search, a non-finite value, or at ``max_iter``.
    """
    _validate(gtol, max_iter, line_search)

    def direction(
        H: Array, g: Array, eig: tuple[Array, Array], tol: float
    ) -> tuple[Array, dict[str, Any]] | str:
        lam, Q = eig
        if not _is_singular(lam, tol):
            p = _newton_direction(lam, Q, g)
            if _cos_angle(g, p) > _DESCENT_COS:
                return p, {"direction_type": "newton"}
        return -g, {"direction_type": "steepest"}

    return _line_search_newton("damped_newton", problem, x0, gtol, max_iter, line_search, direction)


@register(
    id="modified_newton",
    family="unconstrained",
    name="Modified Newton (Hessian + τI)",
    params=(P_GTOL, P_MAX_ITER, P_LINE_SEARCH, P_BETA),
    needs=("f", "grad", "hess"),
    order="quadratic (near a minimizer with ∇²f ≻ 0, where τ = 0 and α = 1)",
    summary="Add τI to the Hessian until it is positive definite, then take a Newton step.",
    references=(
        "Nocedal & Wright (2006), Alg. 3.2 (line search Newton with modification)",
        "Nocedal & Wright (2006), Alg. 3.3 (Cholesky with added multiple of the identity)",
    ),
)
def modified_newton(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 100,
    line_search: str = "backtracking",
    beta: float = 1e-3,
) -> Result:
    """Line-search Newton with Hessian modification, N&W Alg. 3.2 with Alg. 3.3.

    B_k = ∇²f(x_k) + τ_k I where τ_k is the first shift of the sequence of Alg. 3.3
    (τ₀ = 0 if min_i (∇²f)_ii > 0, else −min_i (∇²f)_ii + β; then τ ← max(2τ, β)) for which the
    Cholesky factorization B_k = L Lᵀ succeeds. The direction solves L Lᵀ p_k = −∇f(x_k), so
    ∇f(x_k)ᵀp_k = −‖L⁻¹∇f(x_k)‖² < 0: always a descent direction. Near a minimizer with
    ∇²f ≻ 0 the shift is τ = 0 and the method is Newton's method.

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol`` and ∇²f(x_k) has no negative eigenvalue
    (beyond rounding); ``converged=False`` at a saddle point or maximizer (a start with
    ∇f = 0 exactly), on a failed line search or factorization, a non-finite value, or at
    ``max_iter``.
    """
    _validate(gtol, max_iter, line_search)
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be finite and > 0, got {beta}")

    def direction(
        H: Array, g: Array, eig: tuple[Array, Array], tol: float
    ) -> tuple[Array, dict[str, Any]] | str:
        L, tau, attempts = _cholesky_shift(H, beta)
        if L is None:
            return f"Alg. 3.3 found no shift: ∇²f + τI had no Cholesky factor up to τ = {tau:.3g}"
        p = _cholesky_solve(L, -g)
        if not finite(p):
            return f"the modified Newton direction is not finite (τ = {tau:.3g})"
        return p, {"tau": tau, "chol_attempts": attempts}

    return _line_search_newton(
        "modified_newton", problem, x0, gtol, max_iter, line_search, direction
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("pure_newton", "rosenbrock", {}),
    ("pure_newton", "quadratic_ill", {}),
    ("pure_newton", "himmelblau", {}),
    ("damped_newton", "rosenbrock", {}),
    ("damped_newton", "six_hump_camel", {}),
    ("modified_newton", "rosenbrock", {}),
    ("modified_newton", "himmelblau", {}),
    ("modified_newton", "beale", {"line_search": "strong_wolfe"}),
]
