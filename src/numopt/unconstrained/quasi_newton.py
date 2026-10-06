"""Quasi-Newton methods: Newton-like steps from gradients only.

Each method keeps an approximation H_k of the *inverse* Hessian ∇²f(x_k)⁻¹, moves along

    p_k = −H_k ∇f(x_k),      x_{k+1} = x_k + α_k p_k      (α_k from a line search, α₀ = 1),

and then updates H_k with the step and gradient change

    s_k = x_{k+1} − x_k,      y_k = ∇f(x_{k+1}) − ∇f(x_k),

so that H_{k+1} satisfies the secant equation H_{k+1} y_k = s_k (Nocedal & Wright (2006), eq. 6.6
in inverse form). The methods differ in the update (N&W ch. 6–7):

``bfgs``           N&W Alg. 6.1, eq. 6.17:  H⁺ = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ,  ρ = 1/yᵀs.
``dfp``            N&W eq. 6.15:           H⁺ = H − H y yᵀH / (yᵀH y) + s sᵀ / (yᵀs).
``sr1``            N&W eq. 6.25:           H⁺ = H + v vᵀ / (vᵀy),  v = s − H y.
``broyden_class``  N&W eq. 6.32 (the restricted Broyden class, φ ∈ [0, 1]; φ = 0 is BFGS,
                   φ = 1 is DFP), applied in its inverse form (see :func:`broyden_class`).
``lbfgs``          N&W Alg. 7.5 with the two-loop recursion Alg. 7.4: H_k is never formed; it is
                   the BFGS update of H₀ᵏ = γ_k I (eq. 7.20) with the last m pairs (s_i, y_i).

Shared rules (every method):

* **Start.** H₀ = I, so the first step is a steepest-descent step.
* **Initial scaling** (``bfgs``, ``dfp``, ``broyden_class``; N&W eq. 6.20): just before the
  first update that is applied, H₀ is replaced by γ I with γ = yᵀs / yᵀy, then updated.
  ``sr1`` keeps H₀ = I (γ can be negative for SR1, whose pairs need not have yᵀs > 0).
  ``lbfgs`` rescales every iteration (eq. 7.20, γ_k from the newest stored pair).
* **Curvature test** (``bfgs``, ``dfp``, ``broyden_class``, ``lbfgs``): the pair is used only
  when yᵀs > 10⁻¹⁰‖s‖‖y‖ and ρ = 1/yᵀs and γ = yᵀs/yᵀy are finite with yᵀy > 0; otherwise
  the update is skipped and H is kept (N&W §6.1 p. 143 on skipping; with a Wolfe line search
  yᵀs > 0 always holds, eq. 6.7, but backtracking and Goldstein searches do not guarantee
  it). This keeps H symmetric positive definite. The finiteness conditions only matter near
  the underflow threshold, where yᵀy can round to 0 while yᵀs > 0.
* **SR1 skip rule** (N&W eq. 6.26, in the dual form for the inverse update): apply eq. 6.25
  only if |vᵀy| ≥ r‖y‖‖v‖ with r = 10⁻⁸, v = s − Hy ≠ 0, y ≠ 0 and vᵀy ≠ 0; v = 0 means H
  already satisfies the secant equation, so nothing changes. (With y = 0 the inequality
  reads 0 ≥ 0, so vᵀy ≠ 0 must be required separately.)
* **Finite updates.** An update whose H⁺ is not finite (an overflow in a rank-one term) is
  skipped and H is kept.
* **Descent safeguard.** p_k = −H_k∇f is used only when cos θ = −∇fᵀp / (‖∇f‖‖p‖) > η = 10⁻⁸
  (Zoutendijk, N&W Thm 3.2). Otherwise H_k is reset to I (L-BFGS: the memory is cleared) and
  p_k = −∇f(x_k); a reset restarts the method, so the initial scaling is applied again at the
  next update. SR1 can produce an indefinite H_k, so a reset is a normal event there. The
  other methods keep H_k positive definite, and the Kantorovich bound cos θ ≥ 2√κ/(1 + κ),
  κ = κ(H_k), shows that they reset only when H_k is numerically singular (κ ≳ 4·10¹⁶, e.g.
  on a run that follows an asymptotic valley) or has lost definiteness through rounding.
* **Line search.** ``numopt.line_search.methods.search`` with α₀ = 1; strong Wolfe by default
  (N&W Algs. 3.5–3.6, c₁ = 10⁻⁴, c₂ = 0.9). A Wolfe search returns ∇f at the accepted step,
  which is reused.
* **Stopping test (converged).** ‖∇f(x_k)‖∞ ≤ ``gtol``. This is a first-order test: no Hessian
  is available, so a stationary point is not classified (the methods are descent methods and
  rarely stop at a saddle point, but they can).
* **Failures** (``converged=False``): ``max_iter`` reached; a failed line search (for example
  when f is unbounded below along p, or when the requested ``gtol`` is below the precision at
  which f can resolve a decrease); ∇fᵀp = 0 in floating point (∇f so small that the product
  underflows, e.g. with ``gtol = 0``; no line search can start); a non-finite ∇f.
* **Evaluation counts.** f and ∇f at x_0, then every line-search evaluation, plus ∇f at the
  accepted point when the search did not return it. No Hessian evaluations (n_hev = 0).
  Without an analytic gradient, ∇f is a central difference (``numopt.core.diff.gradient``) and
  its 2n f evaluations are counted in ``n_fev``.
* **Linear algebra.** No matrix is inverted or factorized; every update is O(n²) (O(mn) for
  L-BFGS). The updates are written so that H stays exactly symmetric in floating point.

Trace: one Step per iterate; k = 0 is x_0. ``Step.step_size`` is α_{k−1} (``None`` at k = 0).
``n_iter == trace[-1].k``.

Info keys (every method, every step). Keys marked "incoming" describe the step x_{k−1} → x_k
and the update made after it; they are ``None`` (``[]`` for ``trials``, ``False`` for
``reset``) at k = 0:
    grad: [n]                 ∇f(x_k).
    direction: [n] | None     incoming: the search direction p_{k−1}.
    alpha: float | None       incoming: the accepted step length α_{k−1}.
    trials: [[alpha, f]]      incoming: every (α, f(x_{k−1} + αp)) the line search evaluated.
    reset: bool               incoming: True when H was reset to I (memory cleared) before
                              p_{k−1} was computed, because −H∇f failed the descent test.
    s: [n] | None             incoming: s_{k−1} = x_k − x_{k−1}.
    y: [n] | None             incoming: y_{k−1} = ∇f(x_k) − ∇f(x_{k−1}).
    curvature: float | None   incoming: yᵀs.
    rho: float | None         incoming: the coefficient of the applied update: 1/yᵀs for
                              bfgs, dfp, broyden_class and lbfgs; 1/(vᵀy) for sr1;
                              None when the update was skipped.
    update: "applied" | "skipped" | None   incoming: whether (s, y) changed H (lbfgs: whether
                              the pair entered the memory). "skipped" also when ∇f(x_k) is
                              not finite (the run then stops); H is kept unchanged.
    gamma: float | None       incoming: the scaling γ = yᵀs/yᵀy applied to H₀ before this
                              update (eq. 6.20; None when no scaling happened). lbfgs: the γ_k
                              of H₀ᵏ = γ_k I used for the next direction (eq. 7.20; 1.0 with an
                              empty memory).
    H: [[n]] | None           the inverse-Hessian approximation H_k after the update (it gives
                              the next direction p_k = −H_k∇f(x_k); its ellipse
                              {x_k + d : dᵀH_k⁻¹d ≤ 1} is the model's curvature). Only for
                              n ≤ 2, else None. For lbfgs it is the implicit matrix, obtained
                              by applying the two-loop recursion to the unit vectors.

Additional info keys per method:
    sr1:
        denominator: float | None   incoming: vᵀy with v = s − Hy (the SR1 denominator).
    broyden_class:
        phi_inverse: float | None   incoming: Φ, the parameter of the equivalent inverse-form
                              update (Φ = 1 is the BFGS, Φ = 0 the DFP inverse update).
    lbfgs:
        memory: int           number of stored pairs (≤ m) after the update.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import diff
from ..core.counting import Counted, finite, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, as_vector
from ..line_search.methods import search

Array = NDArray[np.float64]

#: Curvature test yᵀs > _CURVATURE_EPS·‖s‖‖y‖ for the BFGS-type updates.
_CURVATURE_EPS = 1e-10
#: SR1 safeguard constant r of N&W eq. 6.26 (N&W suggest r = 10⁻⁸).
_SR1_R = 1e-8
#: Descent test cos θ > η (Zoutendijk, N&W Thm 3.2).
_DESCENT_COS = 1e-8
#: Machine epsilon of float64.
_EPS = float(np.finfo(np.float64).eps)

LINE_SEARCHES = ("strong_wolfe", "weak_wolfe", "backtracking", "goldstein")

P_GTOL = ParamSpec("gtol", 1e-8, min=1e-14, max=1e-2, log=True, help="Stop when ‖∇f(x)‖∞ ≤ gtol.")
P_MAX_ITER = ParamSpec("max_iter", 500, kind="int", min=1, max=100_000, help="Iteration limit.")
P_LINE_SEARCH = ParamSpec(
    "line_search",
    "strong_wolfe",
    kind="choice",
    choices=LINE_SEARCHES,
    help="Step-length rule (tries α = 1 first). Wolfe searches guarantee yᵀs > 0.",
)


# --------------------------------------------------------------------------------------
# Inverse-Hessian approximations
# --------------------------------------------------------------------------------------


class _Approximation:
    """State of one quasi-Newton approximation: direction, reset and update rules."""

    def __init__(self, n: int) -> None:
        self.n = n

    def apply(self, g: Array) -> Array:
        """Return H g."""
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, s: Array, y: Array, sBs: float) -> dict[str, Any]:
        """Update with the pair (s, y); ``sBs`` = sᵀB s for the B = H⁻¹ that produced p.

        Returns the incoming info keys rho, update, gamma (and the method's own keys).
        """
        raise NotImplementedError

    def matrix(self) -> Array:
        """The (explicit or implicit) H as an n×n array (for the n ≤ 2 info overlay)."""
        raise NotImplementedError

    def empty_info(self) -> dict[str, Any]:
        """The method-specific info keys at k = 0."""
        return {}

    def unchanged_info(self) -> dict[str, Any]:
        """The incoming info keys when no update was attempted (a non-finite pair).

        The approximation is kept as it is, so the keys describe its current state.
        """
        return {"rho": None, "update": "skipped", "gamma": None, **self.empty_info()}


def _norm(v: Array) -> float:
    """‖v‖₂ computed as max|v_i|·‖v / max|v_i|‖₂, which does not underflow for a tiny nonzero v.

    ``numpy.linalg.norm`` forms √(vᵀv), which is 0 for v = 1e-200 (the scaling of BLAS dnrm2).
    """
    v_max = float(np.max(np.abs(v)))
    if not math.isfinite(v_max) or v_max == 0.0:
        return v_max
    return v_max * float(np.linalg.norm(v / v_max))


def _curvature_ok(s: Array, y: Array, ys: float) -> bool:
    """yᵀs > 10⁻¹⁰‖s‖‖y‖ (skip test that keeps the BFGS-type updates positive definite).

    The pair must also give a finite ρ = 1/yᵀs and a finite, positive γ = yᵀs/yᵀy (eqs. 6.20,
    7.20). Near the underflow threshold yᵀy can round to 0 while yᵀs > 0 (s = 1, y = 1e-170),
    and γ is then not representable.
    """
    if not (math.isfinite(ys) and ys > 0.0):
        return False
    yy = float(y @ y)
    if not (math.isfinite(yy) and yy > 0.0):
        return False
    if not (math.isfinite(1.0 / ys) and math.isfinite(ys / yy)):
        return False
    return ys > _CURVATURE_EPS * _norm(s) * _norm(y)


class _Dense(_Approximation):
    """An explicit symmetric H (BFGS, DFP, Broyden class, SR1)."""

    #: Apply N&W eq. 6.20 (H₀ = γI) before the first applied update.
    scale_initial = True

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.H = np.eye(n)
        self.scaled = False

    def apply(self, g: Array) -> Array:
        return self.H @ g

    def reset(self) -> None:
        self.H = np.eye(self.n)
        self.scaled = False

    def matrix(self) -> Array:
        return self.H.copy()

    def _maybe_scale(self, s: Array, y: Array, ys: float) -> float | None:
        """N&W eq. 6.20: H₀ ← (yᵀs / yᵀy) I before the first update. Returns γ or None."""
        if not self.scale_initial or self.scaled:
            return None
        gamma = ys / float(y @ y)
        self.H = gamma * np.eye(self.n)
        self.scaled = True
        return gamma

    def update(self, s: Array, y: Array, sBs: float) -> dict[str, Any]:
        ys = float(y @ s)
        if not _curvature_ok(s, y, ys):
            return {"rho": None, "update": "skipped", "gamma": None, **self.skipped_info()}
        gamma = self._maybe_scale(s, y, ys)
        if gamma is not None:
            # NOTE: sᵀBs for the rescaled B₀ = I/γ (the relation Bs = −α∇f holds only for the H
            # that produced the direction, which was replaced by γI).
            sBs = float(s @ s) / gamma
        # Overflow in a rank-one term is detected below (the update is then skipped).
        with np.errstate(over="ignore", invalid="ignore"):
            out = self.formula(s, y, ys, sBs)
        # A non-finite H⁺ (an overflow in one of the rank-one terms) is rejected: H is kept.
        if out is None or not finite(out[0]):
            return {"rho": None, "update": "skipped", "gamma": gamma, **self.skipped_info()}
        self.H, extra = out
        return {"rho": 1.0 / ys, "update": "applied", "gamma": gamma, **extra}

    def formula(
        self, s: Array, y: Array, ys: float, sBs: float
    ) -> tuple[Array, dict[str, Any]] | None:
        """Return (H⁺, extra info) for the pair, or None to skip the update."""
        raise NotImplementedError

    def skipped_info(self) -> dict[str, Any]:
        return {}


class _BFGS(_Dense):
    def formula(
        self, s: Array, y: Array, ys: float, sBs: float
    ) -> tuple[Array, dict[str, Any]] | None:
        # N&W eq. 6.17 expanded: with ρ = 1/yᵀs and u = H y,
        #   (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ = H − ρ(u sᵀ + s uᵀ) + (ρ² yᵀu + ρ) s sᵀ.
        # The expanded form is exactly symmetric in floating point (u sᵀ + s uᵀ is).
        rho = 1.0 / ys
        u = self.H @ y
        H = (
            self.H
            - rho * (np.outer(u, s) + np.outer(s, u))
            + (rho * rho * float(y @ u) + rho) * np.outer(s, s)
        )
        return H, {}


class _DFP(_Dense):
    def formula(
        self, s: Array, y: Array, ys: float, sBs: float
    ) -> tuple[Array, dict[str, Any]] | None:
        # N&W eq. 6.15: H⁺ = H − (Hy)(Hy)ᵀ / (yᵀHy) + s sᵀ / (yᵀs).
        u = self.H @ y
        yHy = float(y @ u)
        if not (math.isfinite(yHy) and yHy > 0.0):
            # Reachable when rounding has made H indefinite, or when yᵀHy underflows to 0.
            return None
        return self.H - np.outer(u, u) / yHy + np.outer(s, s) / ys, {}


class _Broyden(_Dense):
    """The restricted Broyden class N&W eq. 6.32, φ ∈ [0, 1], applied to H.

    The inverse of B⁺(φ) (eq. 6.32) is the inverse-form Broyden update
        H⁺ = H − u uᵀ/(yᵀu) + s sᵀ/(yᵀs) + Φ (yᵀu) w wᵀ,   u = H y,  w = s/(yᵀs) − u/(yᵀu),
    with the dual parameter Φ = (1 − φ) / (1 − φ + φ μ),  μ = (yᵀHy)(sᵀBs) / (yᵀs)² ≥ 1
    (Fletcher (1987), §3.4; Sherman–Morrison–Woodbury). Φ = 1 gives the BFGS and Φ = 0 the DFP
    inverse update, so φ = 0 is BFGS and φ = 1 is DFP, exactly as in eq. 6.32.
    """

    def __init__(self, n: int, phi: float) -> None:
        super().__init__(n)
        self.phi = phi

    def formula(
        self, s: Array, y: Array, ys: float, sBs: float
    ) -> tuple[Array, dict[str, Any]] | None:
        u = self.H @ y
        yHy = float(y @ u)
        if not (math.isfinite(yHy) and yHy > 0.0 and math.isfinite(sBs) and sBs > 0.0):
            return None
        # μ = (yᵀHy/yᵀs)(sᵀBs/yᵀs): the product (yᵀs)² of the plain form underflows to 0 for
        # yᵀs ≲ 1e-162, although yᵀs itself passed the curvature test.
        mu = (yHy / ys) * (sBs / ys)
        if not math.isfinite(mu):
            return None
        big_phi = (1.0 - self.phi) / (1.0 - self.phi + self.phi * mu)
        w = s / ys - u / yHy
        H = self.H - np.outer(u, u) / yHy + np.outer(s, s) / ys + (big_phi * yHy) * np.outer(w, w)
        return H, {"phi_inverse": big_phi}

    def skipped_info(self) -> dict[str, Any]:
        return {"phi_inverse": None}

    def empty_info(self) -> dict[str, Any]:
        return {"phi_inverse": None}


class _SR1(_Dense):
    scale_initial = False

    def update(self, s: Array, y: Array, sBs: float) -> dict[str, Any]:
        v = s - self.H @ y
        den = float(v @ y)
        v_norm = _norm(v)
        y_norm = _norm(y)
        # N&W eq. 6.26 for the inverse update: |vᵀy| ≥ r‖y‖‖v‖. v = 0: the secant equation
        # already holds and eq. 6.25 would be 0/0, so H is kept. The test alone accepts
        # y = 0 (or ‖y‖ rounded to 0), where it reads 0 ≥ 0 with vᵀy = 0: hence vᵀy ≠ 0 and
        # ‖y‖ > 0 are required explicitly.
        skipped = {"rho": None, "update": "skipped", "gamma": None, "denominator": den}
        ok = (
            v_norm > 0.0
            and y_norm > 0.0
            and math.isfinite(den)
            and den != 0.0
            and abs(den) >= _SR1_R * y_norm * v_norm
        )
        if not ok:
            return skipped
        with np.errstate(over="ignore", invalid="ignore"):
            H = self.H + np.outer(v, v) / den
        if not (finite(H) and math.isfinite(1.0 / den)):
            # vᵀy near the underflow threshold: H⁺ or ρ overflows, so H is kept.
            return skipped
        self.H = H
        return {"rho": 1.0 / den, "update": "applied", "gamma": None, "denominator": den}

    def empty_info(self) -> dict[str, Any]:
        return {"denominator": None}


def _two_loop(g: Array, pairs: list[tuple[Array, Array, float]], gamma: float) -> Array:
    """N&W Alg. 7.4: return H_k g for H₀ᵏ = γI and the stored pairs (oldest first)."""
    q = g.copy()
    a = [0.0] * len(pairs)
    for i in range(len(pairs) - 1, -1, -1):
        s, y, rho = pairs[i]
        a[i] = rho * float(s @ q)
        q = q - a[i] * y
    r = gamma * q
    for i, (s, y, rho) in enumerate(pairs):
        beta = rho * float(y @ r)
        r = r + (a[i] - beta) * s
    return r


class _LBFGS(_Approximation):
    """Limited-memory BFGS: the last m pairs and H₀ᵏ = γ_k I (N&W eq. 7.20)."""

    def __init__(self, n: int, m: int) -> None:
        super().__init__(n)
        self.m = m
        self.pairs: list[tuple[Array, Array, float]] = []
        self.gamma = 1.0

    def apply(self, g: Array) -> Array:
        return _two_loop(g, self.pairs, self.gamma)

    def reset(self) -> None:
        self.pairs = []
        self.gamma = 1.0

    def update(self, s: Array, y: Array, sBs: float) -> dict[str, Any]:
        ys = float(y @ s)
        applied = _curvature_ok(s, y, ys)
        if applied:
            if len(self.pairs) == self.m:
                self.pairs.pop(0)
            self.pairs.append((s.copy(), y.copy(), 1.0 / ys))
            # eq. 7.20 with the newest stored pair.
            self.gamma = ys / float(y @ y)
        return {
            "rho": 1.0 / ys if applied else None,
            "update": "applied" if applied else "skipped",
            "gamma": self.gamma,
            "memory": len(self.pairs),
        }

    def matrix(self) -> Array:
        return np.column_stack([self.apply(e) for e in np.eye(self.n)])

    def empty_info(self) -> dict[str, Any]:
        return {"memory": 0}

    def unchanged_info(self) -> dict[str, Any]:
        # The memory and γ_k are kept: they still define the H of the "H" info key.
        return {"rho": None, "update": "skipped", "gamma": self.gamma, "memory": len(self.pairs)}


# --------------------------------------------------------------------------------------
# The shared quasi-Newton loop
# --------------------------------------------------------------------------------------


def _validate(gtol: float, max_iter: int, line_search: str) -> None:
    if not (math.isfinite(gtol) and gtol >= 0.0):
        raise ValueError(f"gtol must be finite and ≥ 0, got {gtol}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if line_search not in LINE_SEARCHES:
        raise ValueError(f"unknown line_search {line_search!r}; expected one of {LINE_SEARCHES}")


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
    """Failure message that compares the predicted decrease with the rounding level of f.

    When |∇fᵀp| is near ε·|f|, no step can be shown to decrease f (the Armijo test compares
    values of f), so ``gtol`` is below the accuracy that f can resolve.
    """
    return (
        f"line search failed at iteration {k} (‖∇f‖∞ = {float(np.max(np.abs(g))):.3g}): {why}; "
        f"predicted decrease |∇fᵀp| = {abs(_slope(g, p)):.3g} vs rounding level of f "
        f"ε·max(1, |f|) = {_EPS * max(1.0, abs(fx)):.3g}"
    )


def _quasi_newton(
    method: str,
    problem: Problem | Callable[..., Any],
    x0: Any,
    gtol: float,
    max_iter: int,
    line_search: str,
    make: Callable[[int], _Approximation],
) -> Result:
    """Generic line-search quasi-Newton iteration (N&W Alg. 6.1 pattern)."""
    _validate(gtol, max_iter, line_search)
    max_iter = int(max_iter)
    prob = vector_problem(problem, x0=x0) if not isinstance(problem, Problem) else problem
    x = start_point(prob, x0)
    f = Counted(prob.f)
    if prob.grad is not None:
        grad = Counted(prob.grad)
    else:
        grad = Counted(lambda z: diff.gradient(f, z))
    n = x.size
    approx = make(n)

    def result(converged: bool, message: str, k: int) -> Result:
        return Result(
            method, x, fx, converged, message, k, n_fev=f.n, n_gev=grad.n, n_hev=0, trace=trace
        )

    def h_info() -> list[list[float]] | None:
        return approx.matrix().tolist() if n <= 2 else None

    fx = float(f(x))
    g = as_vector(grad(x))
    if not finite(fx, g):
        raise ValueError(f"{method}: f and ∇f must be finite at x0; got f={fx}, ∇f={g}")
    info: dict[str, Any] = {
        "grad": g.tolist(),
        "direction": None,
        "alpha": None,
        "trials": [],
        "reset": False,
        "s": None,
        "y": None,
        "curvature": None,
        "rho": None,
        "update": None,
        "gamma": None,
        **approx.empty_info(),
        "H": h_info(),
    }
    trace = [Step(0, x.copy(), fx, float(np.linalg.norm(g)), None, info)]
    k = 0
    while True:
        gnorm = float(np.max(np.abs(g)))
        if gnorm <= gtol:
            return result(True, f"‖∇f‖∞ = {gnorm:.3g} ≤ gtol", k)
        if k == max_iter:
            return result(False, f"reached max_iter={max_iter} (‖∇f‖∞ = {gnorm:.3g})", k)
        p = -approx.apply(g)
        reset = not (_cos_angle(g, p) > _DESCENT_COS)
        if reset:
            approx.reset()
            p = -g
        slope = _slope(g, p)
        if not (math.isfinite(slope) and slope < 0.0):
            # The line search needs a finite ∇fᵀp < 0 in floating point. Here p passed the
            # descent test (or p = −∇f), so a non-finite ∇fᵀp overflowed (‖∇f‖‖p‖ ≳ 1e308)
            # and ∇fᵀp ≥ 0 underflowed: ∇f is near the underflow threshold, far below any
            # decrease that f can resolve.
            why = _slope_failure(slope)
            return result(False, _line_search_failure(k + 1, g, p, fx, why), k)
        ls = search(line_search, f, grad, x, p, f0=fx, g0=g, alpha0=1.0)
        if not ls.success:
            return result(False, _line_search_failure(k + 1, g, p, fx, ls.message), k)
        x_new = x + ls.alpha * p
        g_new = as_vector(ls.g_new) if ls.g_new is not None else as_vector(grad(x_new))
        s = x_new - x
        y = g_new - g
        ys = float(y @ s)
        # NOTE: B_k s_k = −α_k ∇f(x_k) because B_k p_k = −∇f(x_k) (B = H⁻¹ of the matrix that
        # produced p_k), so sᵀBs = −α ∇fᵀs needs no inverse (Broyden class only).
        sBs = -ls.alpha * float(g @ s)
        k += 1
        x, fx, g = x_new, float(ls.f_new), g_new
        if finite(g, s, y):
            upd = approx.update(s, y, sBs)
        else:
            upd = approx.unchanged_info()
        info = {
            "grad": g.tolist(),
            "direction": p.tolist(),
            "alpha": ls.alpha,
            "trials": [[a, v] for a, v in ls.trials],
            "reset": reset,
            "s": s.tolist(),
            "y": y.tolist(),
            "curvature": ys,
            **upd,
            "H": h_info(),
        }
        trace.append(Step(k, x.copy(), fx, float(np.linalg.norm(g)), ls.alpha, info))
        if not finite(g):
            return result(False, f"∇f is not finite at iteration {k}", k)


# --------------------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------------------


@register(
    id="bfgs",
    family="unconstrained",
    name="BFGS",
    params=(P_GTOL, P_MAX_ITER, P_LINE_SEARCH),
    needs=("f", "grad"),
    order="superlinear",
    summary="Build an inverse-Hessian estimate from gradient changes; the most popular quasi-Newton.",
    references=(
        "Nocedal & Wright (2006), Algorithm 6.1, eqs. 6.17 and 6.20",
        "Broyden, Fletcher, Goldfarb & Shanno (1970)",
    ),
)
def bfgs(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 500,
    line_search: str = "strong_wolfe",
) -> Result:
    """BFGS method, N&W Alg. 6.1 (inverse update eq. 6.17, H₀ scaling eq. 6.20).

    p_k = −H_k∇f_k; x_{k+1} = x_k + α_kp_k (strong Wolfe by default); then
    H_{k+1} = (I − ρ_k s_k y_kᵀ) H_k (I − ρ_k y_k s_kᵀ) + ρ_k s_k s_kᵀ, ρ_k = 1/y_kᵀs_k.
    The update is skipped when y_kᵀs_k ≤ 10⁻¹⁰‖s_k‖‖y_k‖ (``info["update"] == "skipped"``).
    Superlinear convergence near a minimizer with ∇²f ≻ 0 (N&W Thm 6.6).

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol``; ``converged=False`` on a failed line
    search, a non-finite gradient, or at ``max_iter``.
    """
    return _quasi_newton("bfgs", problem, x0, gtol, max_iter, line_search, _BFGS)


@register(
    id="dfp",
    family="unconstrained",
    name="DFP",
    params=(P_GTOL, P_MAX_ITER, P_LINE_SEARCH),
    needs=("f", "grad"),
    order="superlinear",
    summary="The first quasi-Newton method: a rank-two update of the inverse Hessian (BFGS's dual).",
    references=(
        "Nocedal & Wright (2006), eq. 6.15",
        "Davidon (1959); Fletcher & Powell (1963)",
    ),
)
def dfp(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 500,
    line_search: str = "strong_wolfe",
) -> Result:
    """Davidon–Fletcher–Powell method: N&W Alg. 6.1 with the DFP inverse update eq. 6.15.

    H_{k+1} = H_k − H_ky_ky_kᵀH_k / (y_kᵀH_ky_k) + s_ks_kᵀ / (y_kᵀs_k), with the same
    curvature test and H₀ scaling (eq. 6.20) as :func:`bfgs`. DFP is much less effective
    than BFGS at correcting a bad Hessian approximation (N&W §6.1, p. 142), so expect many
    more iterations on curved valleys.

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol``; ``converged=False`` on a failed line
    search, a non-finite gradient, or at ``max_iter``.
    """
    return _quasi_newton("dfp", problem, x0, gtol, max_iter, line_search, _DFP)


@register(
    id="sr1",
    family="unconstrained",
    name="SR1 (symmetric rank-one)",
    params=(P_GTOL, P_MAX_ITER, P_LINE_SEARCH),
    needs=("f", "grad"),
    order="superlinear (n+1-step superlinear)",
    summary="A rank-one secant update that can model negative curvature (H may be indefinite).",
    references=("Nocedal & Wright (2006), §6.2, eqs. 6.25–6.26",),
)
def sr1(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 500,
    line_search: str = "strong_wolfe",
) -> Result:
    """Symmetric rank-one method with a line search (N&W §6.2, inverse update eq. 6.25).

    H_{k+1} = H_k + v vᵀ / (vᵀy_k) with v = s_k − H_ky_k, applied only when
    |vᵀy_k| ≥ 10⁻⁸‖y_k‖‖v‖ (eq. 6.26 in dual form). H₀ = I with no scaling.

    # NOTE: N&W present SR1 with a trust region (Alg. 6.2), where an indefinite H is
    # harmless. With a line search, −H_k∇f can be an ascent direction; this implementation
    # then resets H_k to I and takes a steepest-descent step (``info["reset"]``).

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol``; ``converged=False`` on a failed line
    search, a non-finite gradient, or at ``max_iter``.
    """
    return _quasi_newton("sr1", problem, x0, gtol, max_iter, line_search, _SR1)


@register(
    id="broyden_class",
    family="unconstrained",
    name="Broyden class (φ)",
    params=(
        P_GTOL,
        P_MAX_ITER,
        P_LINE_SEARCH,
        ParamSpec(
            "phi",
            0.5,
            min=0.0,
            max=1.0,
            help="Broyden parameter φ ∈ [0, 1]: φ = 0 is BFGS, φ = 1 is DFP.",
        ),
    ),
    needs=("f", "grad"),
    order="superlinear",
    summary="Blend BFGS (φ = 0) and DFP (φ = 1): the restricted Broyden family of updates.",
    references=(
        "Nocedal & Wright (2006), §6.3, eq. 6.32",
        "Fletcher (1987), Practical Methods of Optimization, §3.4",
    ),
)
def broyden_class(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 500,
    line_search: str = "strong_wolfe",
    phi: float = 0.5,
) -> Result:
    """Restricted Broyden class, N&W eq. 6.32 with φ ∈ [0, 1], in inverse form.

    B_{k+1} = B_k − B_ks_ks_kᵀB_k/(s_kᵀB_ks_k) + y_ky_kᵀ/(y_kᵀs_k) + φ (s_kᵀB_ks_k) v_kv_kᵀ,
    v_k = y_k/(y_kᵀs_k) − B_ks_k/(s_kᵀB_ks_k). The method stores H_k = B_k⁻¹ and applies the
    equivalent inverse update with the dual parameter Φ (see ``_Broyden``); B_ks_k = −α_k∇f_k
    gives sᵀBs without forming B. Same curvature test and H₀ scaling (eq. 6.20) as
    :func:`bfgs`; for φ ∈ [0, 1] every applied update keeps H positive definite (N&W p. 150).

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol``; ``converged=False`` on a failed line
    search, a non-finite gradient, or at ``max_iter``.
    """
    if not (math.isfinite(phi) and 0.0 <= phi <= 1.0):
        raise ValueError(f"phi must lie in [0, 1] (the restricted Broyden class), got {phi}")
    phi = float(phi)
    return _quasi_newton(
        "broyden_class",
        problem,
        x0,
        gtol,
        max_iter,
        line_search,
        lambda n: _Broyden(n, phi),
    )


@register(
    id="lbfgs",
    family="unconstrained",
    name="L-BFGS",
    params=(
        P_GTOL,
        P_MAX_ITER,
        P_LINE_SEARCH,
        ParamSpec(
            "m",
            10,
            kind="int",
            min=1,
            max=50,
            help="Memory: number of (s, y) pairs kept (N&W suggest 3 ≤ m ≤ 20).",
        ),
    ),
    needs=("f", "grad"),
    order="linear (fast; superlinear only in the limit m → ∞)",
    summary="BFGS that keeps only the last m step/gradient pairs: O(mn) memory and work.",
    references=(
        "Nocedal & Wright (2006), Algorithms 7.4 (two-loop recursion) and 7.5, eq. 7.20",
        "Liu & Nocedal (1989)",
    ),
)
def lbfgs(
    problem: Problem | Callable[..., Any],
    *,
    x0: Any = None,
    gtol: float = 1e-8,
    max_iter: int = 500,
    line_search: str = "strong_wolfe",
    m: int = 10,
) -> Result:
    """Limited-memory BFGS, N&W Alg. 7.5 with the two-loop recursion Alg. 7.4.

    H_k∇f_k is computed from the newest m pairs (s_i, y_i, ρ_i = 1/y_iᵀs_i) and
    H₀ᵏ = γ_k I with γ_k = sᵀy/yᵀy of the newest stored pair (eq. 7.20; γ = 1 and p = −∇f while
    the memory is empty). A pair enters the memory only when yᵀs > 10⁻¹⁰‖s‖‖y‖; when the
    memory is full the oldest pair is discarded.

    Stops (converged) when ‖∇f(x_k)‖∞ ≤ ``gtol``; ``converged=False`` on a failed line
    search, a non-finite gradient, or at ``max_iter``.
    """
    if int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer, got {m}")
    mem = int(m)
    return _quasi_newton(
        "lbfgs", problem, x0, gtol, max_iter, line_search, lambda n: _LBFGS(n, mem)
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("bfgs", "rosenbrock", {}),
    ("bfgs", "himmelblau", {"line_search": "backtracking"}),
    ("dfp", "quadratic_ill", {}),
    ("sr1", "rosenbrock", {}),
    ("sr1", "six_hump_camel", {}),
    ("broyden_class", "beale", {"phi": 0.5}),
    ("lbfgs", "rosenbrock", {"m": 5}),
    ("lbfgs", "rosenbrock_nd", {}),
]
