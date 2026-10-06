"""Nonlinear least-squares problems: minimize f(x) = ½‖r(x)‖² over parameters x ∈ ℝⁿ.

Every problem supplies the residual r: ℝⁿ → ℝᵐ, its Jacobian J = ∂r/∂x (m × n), and the exact
derivatives of f (Nocedal & Wright (2006), eqs. 10.4–10.5):

    ∇f(x)  = J(x)ᵀ r(x),
    ∇²f(x) = J(x)ᵀ J(x) + Σᵢ rᵢ(x) ∇²rᵢ(x)      (Gauss–Newton term + second-order term).

The sign convention is r = model − data, so J is the derivative of the model.

Conventions:
    * ``residual(x)`` and ``f(x)`` accept ``x`` of shape ``(n,)``, or ``(n, *grid)`` for contour
      grids (``residual`` then returns ``(m, *grid)`` and ``f`` returns ``grid``).
    * ``jac(x) -> (m, n)``, ``grad(x) -> (n,)``, ``hess(x) -> (n, n)`` take ``x`` of shape ``(n,)``.
    * ``domain = ((lo_1, hi_1), (lo_2, hi_2))`` is the plotting box in parameter space.
    * ``minima`` lists the known local minimizers, global first (verified in the tests).

Extra keys (``Problem.extra``):
    t, y: [float] — the data abscissae and observations of the curve fits
        (``exp_decay_fit``, ``michaelis_menten``; ``t`` is the substrate concentration there).
    points_x, points_y: [float] — the data points of ``circle_fit``.
    radius: float — the known circle radius of ``circle_fit``.
    model: str — LaTeX of the fitted model.
    true_params: [float] — the parameters used to generate synthetic data.
    noise_std: float, seed: int — the Gaussian noise level and the ``numopt.core.rng.Rng`` seed.
    m: int — the number of residuals.
    minima_f: [float], n_global: int, f_min: float — as in ``problems.unconstrained``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.rng import Rng
from ..core.types import Problem
from .registry import factory

Array = NDArray[np.float64]


def _arr(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def _col(v: Array, x: Array) -> Array:
    """Reshape the data vector ``v`` (m,) to broadcast against parameter grids ``x[k]``."""
    return v.reshape((-1,) + (1,) * (x.ndim - 1))


def _ls_problem(
    *,
    id: str,
    name: str,
    latex: str,
    residual: Callable[[Array], Array],
    jac: Callable[[Array], Array],
    residual_hessians: Callable[[Array], Array],
    domain: Sequence[tuple[float, float]],
    x0: Sequence[float],
    minima: Sequence[Sequence[float]],
    n_global: int,
    m: int,
    description: str,
    tags: tuple[str, ...],
    extra: dict[str, Any],
) -> Problem:
    """Build f = ½‖r‖², ∇f = Jᵀr and ∇²f = JᵀJ + Σ rᵢ∇²rᵢ from the residual pieces.

    ``residual_hessians(x)`` returns the stacked second derivatives ∇²rᵢ, shape (m, n, n).
    """

    def res(x: ArrayLike) -> Array:
        return residual(_arr(x))

    def jacobian(x: ArrayLike) -> Array:
        return jac(_arr(x))

    def f(x: ArrayLike) -> Any:
        r = residual(_arr(x))  # (m, *grid)
        return 0.5 * np.sum(r * r, axis=0)

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return jac(x).T @ residual(x)

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        r = residual(x)  # (m,)
        J = jac(x)  # (m, n)
        H = J.T @ J + np.einsum("i,ijk->jk", r, residual_hessians(x))
        return 0.5 * (H + H.T)

    minima_f = [float(f(np.array(p, dtype=float))) for p in minima]
    return Problem(
        id=id,
        name=name,
        latex=latex,
        f=f,
        dim=len(x0),
        domain=tuple((float(lo), float(hi)) for lo, hi in domain),
        grad=grad,
        hess=hess,
        jac=jacobian,
        residual=res,
        x0=[float(v) for v in x0],
        minima=tuple([float(v) for v in p] for p in minima),
        description=description,
        tags=("least-squares", *tags),
        extra={
            **extra,
            "m": m,
            "minima_f": minima_f,
            "n_global": n_global,
            "f_min": minima_f[0],
        },
    )


# --------------------------------------------------------------------------------------
# Exponential decay  y ≈ a·exp(−b t)
# --------------------------------------------------------------------------------------


@factory("least_squares")
def exp_decay_fit() -> Problem:
    # Data (replayed exactly by the TypeScript port): t_i = 0.3 i for i = 0..14 and
    # y_i = a* exp(−b* t_i) + σ·rng.normal(), drawn in order i = 0..14 from Rng(7).
    seed, sigma, a_true, b_true = 7, 0.05, 2.5, 1.3
    t = 0.3 * np.arange(15, dtype=np.float64)
    rng = Rng(seed)
    y = np.array([a_true * math.exp(-b_true * ti) + sigma * rng.normal() for ti in t])

    def residual(x: Array) -> Array:
        return x[0] * np.exp(-x[1] * _col(t, x)) - _col(y, x)

    def jac(x: Array) -> Array:
        e = np.exp(-x[1] * t)
        return np.column_stack([e, -x[0] * t * e])

    def residual_hessians(x: Array) -> Array:
        # ∂²rᵢ/∂a² = 0, ∂²rᵢ/∂a∂b = −tᵢe_i, ∂²rᵢ/∂b² = a tᵢ² e_i, with e_i = exp(−b tᵢ).
        e = np.exp(-x[1] * t)
        H = np.zeros((t.size, 2, 2))
        H[:, 0, 1] = H[:, 1, 0] = -t * e
        H[:, 1, 1] = x[0] * t**2 * e
        return H

    return _ls_problem(
        id="exp_decay_fit",
        name="Exponential decay fit",
        latex=r"\min_{a,b}\ \tfrac12\sum_{i=1}^{15}\left(a\,e^{-b t_i} - y_i\right)^2",
        residual=residual,
        jac=jac,
        residual_hessians=residual_hessians,
        domain=((0.0, 4.0), (0.0, 3.0)),
        x0=(1.0, 0.3),
        minima=((2.4935921290442926, 1.3348854533574257),),
        n_global=1,
        m=t.size,
        description="Fit y = a·exp(−b t) to 15 noisy samples of 2.5·exp(−1.3 t) (σ = 0.05). "
        "A small-residual problem: Gauss–Newton converges fast near the solution.",
        tags=("curve-fit", "small-residual", "2d"),
        extra={
            "t": t.tolist(),
            "y": y.tolist(),
            "model": r"y = a\,e^{-b t}",
            "true_params": [a_true, b_true],
            "noise_std": sigma,
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------------------
# Rosenbrock as a zero-residual least-squares problem
# --------------------------------------------------------------------------------------


@factory("least_squares")
def rosenbrock_ls() -> Problem:
    def residual(x: Array) -> Array:
        return np.stack([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    def jac(x: Array) -> Array:
        return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])

    def residual_hessians(x: Array) -> Array:
        return np.array([[[-20.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])

    return _ls_problem(
        id="rosenbrock_ls",
        name="Rosenbrock (least squares)",
        latex=r"r(x,y) = \begin{pmatrix} 10\,(y - x^2) \\ 1 - x \end{pmatrix},\ "
        r"f = \tfrac12\|r\|^2",
        residual=residual,
        jac=jac,
        residual_hessians=residual_hessians,
        domain=((-2.0, 2.0), (-1.0, 3.0)),
        x0=(-1.2, 1.0),
        minima=((1.0, 1.0),),
        n_global=1,
        m=2,
        description="Rosenbrock's function written as residuals (MGH problem 1); f is half the "
        "usual Rosenbrock value. J is square with det J = 10, so Gauss–Newton is Newton's "
        "method for r(x) = 0: the full step (no line search) converges in two steps from "
        "(−1.2, 1), via (1, −3.84). With Armijo backtracking the first full step is rejected "
        "(f rises from 12.1 to 1171), and the damped iteration takes 10 steps.",
        tags=("zero-residual", "classic", "2d"),
        extra={"model": r"r = (10(y - x^2),\ 1 - x)"},
    )


# --------------------------------------------------------------------------------------
# Circle fit with known radius: find the center (a, b)
# --------------------------------------------------------------------------------------


@factory("least_squares")
def circle_fit() -> Problem:
    # Data (replayed exactly by the TypeScript port), from Rng(3), for i = 0..11 in order:
    #   θ_i = rng.uniform(π/6, 5π/6), ε_i = σ·rng.normal(),
    #   p_i = c* + (R + ε_i)(cos θ_i, sin θ_i).
    seed, sigma, radius = 3, 0.05, 2.0
    center = (1.0, -0.5)
    rng = Rng(seed)
    pts: list[tuple[float, float]] = []
    for _ in range(12):
        theta = rng.uniform(math.pi / 6.0, 5.0 * math.pi / 6.0)
        rho = radius + sigma * rng.normal()
        pts.append((center[0] + rho * math.cos(theta), center[1] + rho * math.sin(theta)))
    px = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])

    def _offsets(x: Array) -> tuple[Array, Array, Array]:
        dx = px - x[0]
        dy = py - x[1]
        return dx, dy, np.hypot(dx, dy)

    def residual(x: Array) -> Array:
        return np.hypot(_col(px, x) - x[0], _col(py, x) - x[1]) - radius

    def jac(x: Array) -> Array:
        # ∂dᵢ/∂(a, b) = −(pᵢ − c)/dᵢ = −eᵢ (unit vector from the center to the point).
        dx, dy, d = _offsets(x)
        # NOTE: dᵢ = ‖pᵢ − c‖ is not differentiable when the center sits on a data point
        # (dᵢ = 0); we use the zero row there (0 is in the subdifferential of ‖·‖ at 0).
        safe = np.where(d > 0.0, d, 1.0)
        return np.column_stack(
            [np.where(d > 0.0, -dx / safe, 0.0), np.where(d > 0.0, -dy / safe, 0.0)]
        )

    def residual_hessians(x: Array) -> Array:
        # ∇²dᵢ = (I − eᵢeᵢᵀ)/dᵢ (the curvature of the distance function).
        dx, dy, d = _offsets(x)
        safe = np.where(d > 0.0, d, 1.0)
        ex, ey = dx / safe, dy / safe
        H = np.empty((px.size, 2, 2))
        H[:, 0, 0] = (1.0 - ex * ex) / safe
        H[:, 0, 1] = H[:, 1, 0] = -ex * ey / safe
        H[:, 1, 1] = (1.0 - ey * ey) / safe
        # NOTE: at dᵢ = 0 the curvature is unbounded; we return 0 for that point (see jac).
        H[d == 0.0] = 0.0
        return H

    return _ls_problem(
        id="circle_fit",
        name="Circle fit (known radius)",
        latex=r"\min_{a,b}\ \tfrac12\sum_{i=1}^{12}\left(\sqrt{(x_i-a)^2 + (y_i-b)^2} - R\right)^2,\ R = 2",
        residual=residual,
        jac=jac,
        residual_hessians=residual_hessians,
        domain=((-2.0, 4.0), (-3.0, 4.5)),
        x0=(3.0, -2.5),
        minima=((0.9572854075202153, -0.5108428174802644), (1.2991121563858266, 2.742363185576751)),
        n_global=1,
        m=px.size,
        description="Find the center of a circle of known radius 2 from 12 noisy points on a "
        "120° arc. The arc is short, so the mirror image of the true center across the arc "
        "is a second, worse local minimum; which one a method finds depends on x₀.",
        tags=("geometry", "multiple-minima", "2d"),
        extra={
            "points_x": px.tolist(),
            "points_y": py.tolist(),
            "radius": radius,
            "model": r"(x - a)^2 + (y - b)^2 = R^2",
            "true_params": list(center),
            "noise_std": sigma,
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------------------
# Michaelis–Menten enzyme kinetics: Puromycin (treated) data
# --------------------------------------------------------------------------------------

# Bates & Watts, "Nonlinear Regression Analysis and Its Applications" (1988), Appendix A1.3
# (also R's `Puromycin` data, state == "treated"): substrate concentration (ppm) and initial
# velocity (counts/min²). The least-squares estimates are θ̂ = (212.68, 0.06412) with
# residual sum of squares 1195 (Bates & Watts, §2.2).
_PURO_CONC = (0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10)
_PURO_RATE = (76.0, 47.0, 97.0, 107.0, 123.0, 139.0, 159.0, 152.0, 191.0, 201.0, 207.0, 200.0)


@factory("least_squares")
def michaelis_menten() -> Problem:
    S = np.array(_PURO_CONC)
    v = np.array(_PURO_RATE)

    def residual(x: Array) -> Array:
        Sc = _col(S, x)
        return x[0] * Sc / (x[1] + Sc) - _col(v, x)

    def jac(x: Array) -> Array:
        q = 1.0 / (x[1] + S)
        return np.column_stack([S * q, -x[0] * S * q * q])

    def residual_hessians(x: Array) -> Array:
        # ∂²rᵢ/∂V² = 0, ∂²rᵢ/∂V∂K = −Sᵢ/(K+Sᵢ)², ∂²rᵢ/∂K² = 2 V Sᵢ/(K+Sᵢ)³.
        q = 1.0 / (x[1] + S)
        H = np.zeros((S.size, 2, 2))
        H[:, 0, 1] = H[:, 1, 0] = -S * q * q
        H[:, 1, 1] = 2.0 * x[0] * S * q**3
        return H

    return _ls_problem(
        id="michaelis_menten",
        name="Michaelis–Menten (Puromycin)",
        latex=r"\min_{V,K}\ \tfrac12\sum_{i=1}^{12}\left(\frac{V S_i}{K + S_i} - v_i\right)^2",
        residual=residual,
        jac=jac,
        residual_hessians=residual_hessians,
        domain=((150.0, 260.0), (0.02, 0.12)),
        x0=(205.0, 0.08),
        minima=((212.68374314253606, 0.06412128168156707),),
        n_global=1,
        m=S.size,
        description="Fit the Michaelis–Menten law v = V S/(K + S) to the classic Puromycin "
        "(treated) enzyme-kinetics data of Bates & Watts (1988); start (205, 0.08) as in the "
        "book. V and K differ in scale by ~3000×, so JᵀJ is badly conditioned (κ ≈ 1.7·10⁶ at the solution).",
        tags=("curve-fit", "real-data", "badly-scaled", "2d"),
        extra={
            "t": S.tolist(),
            "y": v.tolist(),
            "model": r"v = \frac{V S}{K + S}",
        },
    )
