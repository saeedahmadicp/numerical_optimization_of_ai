"""Test problems of this study that the numopt library does not have (all 2-D, exact derivatives).

Convex (pure Newton diverges from far starts):
    lse              f(x) = log Σᵢ exp(aᵢᵀx − bᵢ), three rows aᵢ with 0 in the interior of their convex
                     hull: strictly convex, a unique minimizer, ∇²f → 0 far from it.
    sqrt1p           f(x) = √(1 + ‖x‖²): pure Newton maps the radius r to −r³ (diverges for r > 1).
    logistic_ridge   f(w) = (1/m) Σᵢ log(1 + exp(−yᵢ aᵢᵀw)) + (μ/2)‖w‖², μ = 10⁻³, on a separable
                     set of m = 8 points: strongly convex, minimizer far from 0.
    logistic_sep     the same loss with μ = 0: convex, inf f = 0 is NOT attained (separable data), so
                     Assumption 2 of Mishchenko (2023) fails; ‖∇f(w)‖ → 0 only as ‖w‖ → ∞.

Nonconvex:
    quartic_saddle   f(x, y) = x² − y² + y⁴/4: a strict saddle at 0 (∇²f = diag(2, −2)) and minima
                     (0, ±√2) with f = −1. Every start on the line y = 0 lies on the saddle's stable
                     manifold for methods whose steps keep y = 0.

``extra["f_min"]`` is the known minimum value (None for ``logistic_sep``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numopt.core.types import Problem

Array = NDArray[np.float64]

# --------------------------------------------------------------------------------------
# log-sum-exp
# --------------------------------------------------------------------------------------

#: Rows aᵢ; Σ cᵢaᵢ = 0 with c ∝ (1, 1.375, 1.875) > 0, so 0 is interior to conv{aᵢ} and f is
#: bounded below; 1 ∉ range(A), so ∇²f = Aᵀ(diag p − ppᵀ)A ≻ 0 everywhere.
LSE_A = np.array([[1.0, 0.5], [-1.0, 1.0], [0.2, -1.0]])  # (m=3, n=2)
LSE_B = np.array([0.0, 0.5, -0.5])  # (3,)


def _softmax(z: Array) -> Array:
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def _lse_f(x: ArrayLike) -> float:
    z = LSE_A @ np.asarray(x, dtype=np.float64) - LSE_B  # (3,)
    zmax = float(np.max(z))
    return zmax + float(np.log(np.sum(np.exp(z - zmax))))


def _lse_grad(x: ArrayLike) -> Array:
    p = _softmax(LSE_A @ np.asarray(x, dtype=np.float64) - LSE_B)
    return LSE_A.T @ p  # (2,)


def _lse_hess(x: ArrayLike) -> Array:
    p = _softmax(LSE_A @ np.asarray(x, dtype=np.float64) - LSE_B)
    W = np.diag(p) - np.outer(p, p)  # (3, 3)
    return LSE_A.T @ W @ LSE_A  # (2, 2)


# --------------------------------------------------------------------------------------
# sqrt(1 + ||x||^2)
# --------------------------------------------------------------------------------------


def _sqrt_f(x: ArrayLike) -> float:
    v = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(1.0 + v @ v))


def _sqrt_grad(x: ArrayLike) -> Array:
    v = np.asarray(x, dtype=np.float64)
    return v / np.sqrt(1.0 + v @ v)


def _sqrt_hess(x: ArrayLike) -> Array:
    v = np.asarray(x, dtype=np.float64)
    s = float(np.sqrt(1.0 + v @ v))
    return (np.eye(v.size) - np.outer(v, v) / s**2) / s


# --------------------------------------------------------------------------------------
# logistic loss
# --------------------------------------------------------------------------------------

#: 8 points, labels yᵢ = sign(aᵢᵀw°) with w° = (1, 2): separable through the origin, smallest
#: normalized margin min yᵢaᵢᵀw°/‖w°‖ = 0.268.
LOG_A = np.array(
    [
        [1.0, 0.5],
        [0.4, 1.2],
        [-0.5, 0.8],
        [2.0, -0.6],
        [-1.0, -0.3],
        [0.3, -1.0],
        [-1.5, 0.6],
        [0.8, -0.7],
    ]
)  # (8, 2)
LOG_Y = np.sign(LOG_A @ np.array([1.0, 2.0]))  # (8,)
LOG_Z = LOG_Y[:, None] * LOG_A  # (8, 2) rows yᵢaᵢ


def _sigmoid(t: Array) -> Array:
    """σ(t) = 1/(1 + e^{−t}) = ½(1 + tanh(t/2)) (no overflow for any t)."""
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def _make_logistic(mu: float) -> tuple[Any, Any, Any]:
    m = LOG_Z.shape[0]

    def f(w: ArrayLike) -> float:
        v = np.asarray(w, dtype=np.float64)
        t = LOG_Z @ v  # (8,) margins
        return float(np.mean(np.logaddexp(0.0, -t))) + 0.5 * mu * float(v @ v)

    def grad(w: ArrayLike) -> Array:
        v = np.asarray(w, dtype=np.float64)
        t = LOG_Z @ v
        return -(LOG_Z.T @ _sigmoid(-t)) / m + mu * v

    def hess(w: ArrayLike) -> Array:
        v = np.asarray(w, dtype=np.float64)
        t = LOG_Z @ v
        c = _sigmoid(t) * _sigmoid(-t)  # (8,) ℓ''(t)
        return (LOG_Z.T * c) @ LOG_Z / m + mu * np.eye(v.size)

    return f, grad, hess


# --------------------------------------------------------------------------------------
# x^2 - y^2 + y^4/4
# --------------------------------------------------------------------------------------


def _quartic_f(x: ArrayLike) -> float:
    u, v = np.asarray(x, dtype=np.float64)
    return float(u * u - v * v + 0.25 * v**4)


def _quartic_grad(x: ArrayLike) -> Array:
    u, v = np.asarray(x, dtype=np.float64)
    return np.array([2.0 * u, -2.0 * v + v**3])


def _quartic_hess(x: ArrayLike) -> Array:
    _, v = np.asarray(x, dtype=np.float64)
    return np.array([[2.0, 0.0], [0.0, -2.0 + 3.0 * v * v]])


def _minimize_reference(f: Any, grad: Any, hess: Any, x: Array) -> Array:
    """Newton's method from a point already in the quadratic-convergence region (reference only)."""
    for _ in range(100):
        g = grad(x)
        if float(np.max(np.abs(g))) <= 1e-15:
            break
        x = x - np.linalg.solve(hess(x), g)
    return x


def get(pid: str) -> Problem:
    """Return the study problem ``pid``."""
    if pid == "lse":
        x_star = _minimize_reference(_lse_f, _lse_grad, _lse_hess, np.array([0.0, 0.3]))
        return Problem(
            id="lse",
            name="Log-sum-exp (3 terms)",
            latex=r"f(x) = \log\sum_{i=1}^{3} e^{a_i^\top x - b_i}",
            f=_lse_f,
            grad=_lse_grad,
            hess=_lse_hess,
            dim=2,
            domain=((-10.0, 10.0), (-10.0, 10.0)),
            x0=(8.0, 8.0),
            minima=(tuple(float(t) for t in x_star),),
            extra={"f_min": _lse_f(x_star)},
            tags=("convex", "2d"),
        )
    if pid == "sqrt1p":
        return Problem(
            id="sqrt1p",
            name="sqrt(1 + ||x||^2)",
            latex=r"f(x) = \sqrt{1 + \|x\|^2}",
            f=_sqrt_f,
            grad=_sqrt_grad,
            hess=_sqrt_hess,
            dim=2,
            domain=((-10.0, 10.0), (-10.0, 10.0)),
            x0=(3.0, 4.0),
            minima=((0.0, 0.0),),
            extra={"f_min": 1.0},
            tags=("convex", "2d"),
        )
    if pid in ("logistic_ridge", "logistic_sep"):
        mu = 1e-3 if pid == "logistic_ridge" else 0.0
        f, grad, hess = _make_logistic(mu)
        minima: tuple[Any, ...] = ()
        extra: dict[str, Any] = {"f_min": None, "mu": mu}
        if mu > 0.0:
            # Start the reference Newton iteration close to the minimizer (found by a continuation
            # in μ is unnecessary: the damped iteration below reaches the quadratic region).
            w = np.array([5.0, 10.0])
            for _ in range(200):
                g = grad(w)
                if float(np.max(np.abs(g))) <= 1e-15:
                    break
                p = -np.linalg.solve(hess(w), g)
                t = 1.0
                while f(w + t * p) > f(w) + 1e-4 * t * float(g @ p) and t > 1e-12:
                    t *= 0.5
                w = w + t * p
            minima = (tuple(float(t) for t in w),)
            extra["f_min"] = f(w)
        return Problem(
            id=pid,
            name=f"Logistic loss, separable data, mu = {mu:g}",
            latex=r"f(w) = \frac1m\sum_i \log(1 + e^{-y_i a_i^\top w}) + \frac{\mu}{2}\|w\|^2",
            f=f,
            grad=grad,
            hess=hess,
            dim=2,
            domain=((-10.0, 10.0), (-10.0, 10.0)),
            x0=(-5.0, -5.0),
            minima=minima,
            extra=extra,
            tags=("convex", "2d"),
        )
    if pid == "quartic_saddle":
        r2 = float(np.sqrt(2.0))
        return Problem(
            id="quartic_saddle",
            name="x^2 - y^2 + y^4/4",
            latex=r"f(x, y) = x^2 - y^2 + \tfrac14 y^4",
            f=_quartic_f,
            grad=_quartic_grad,
            hess=_quartic_hess,
            dim=2,
            domain=((-3.0, 3.0), (-3.0, 3.0)),
            x0=(2.0, 0.0),
            minima=((0.0, r2), (0.0, -r2)),
            extra={"f_min": -1.0},
            tags=("nonconvex", "saddle", "2d"),
        )
    raise KeyError(f"unknown study problem {pid!r}")


CONVEX = ("lse", "sqrt1p", "logistic_ridge", "logistic_sep")
NONCONVEX_LOCAL = ("quartic_saddle",)
