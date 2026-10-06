"""Unconstrained test problems: smooth f: ℝⁿ → ℝ with exact gradient and Hessian.

Most problems are 2-D so that the web app can draw their contours; ``rosenbrock_nd`` (n = 10)
and ``quadratic_nd`` (n = 20) serve the CLI and the tests. Formulas and known minima follow
Jamil & Yang, "A literature survey of benchmark functions for global optimisation problems",
Int. J. Math. Model. Numer. Optim. 4(2), 2013 (cited as J&Y with the function number), and
Moré, Garbow & Hillstrom, ACM TOMS 7(1), 1981 (cited as MGH).

Conventions:
    * ``f(x)`` accepts ``x`` of shape ``(n,)`` and returns a float, or ``x`` of shape
      ``(n, *grid)`` (e.g. ``np.stack(np.meshgrid(xs, ys))``) and returns an array of shape
      ``grid``, so the visualizer can evaluate whole contour grids at once.
    * ``grad(x) -> (n,)`` and ``hess(x) -> (n, n)`` take ``x`` of shape ``(n,)``.
    * ``domain = ((lo_1, hi_1), ..., (lo_n, hi_n))`` is the plotting box.
    * ``minima`` lists the known local minimizers, **global minimizers first**; every listed
      point is verified in the tests (gradient ≈ 0, Hessian positive definite, value matches).

Extra keys (``Problem.extra``):
    minima_f: [float] — f at each entry of ``minima`` (same order).
    n_global: int — the first ``n_global`` entries of ``minima`` are global minimizers.
    f_min: float — the global minimum value (over the plotting box for ``mccormick``).
    A, c: [[float]], [float] — the matrix and center of the quadratic problems
        (f = ½(x − c)ᵀA(x − c)).
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
_PI = math.pi
_TWO_PI = 2.0 * math.pi


def _arr(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def _problem(
    *,
    id: str,
    name: str,
    latex: str,
    f: Callable[[ArrayLike], Any],
    grad: Callable[[ArrayLike], Array],
    hess: Callable[[ArrayLike], Array],
    domain: Sequence[tuple[float, float]],
    x0: Sequence[float],
    minima: Sequence[Sequence[float]],
    minima_f: Sequence[float],
    n_global: int,
    description: str,
    tags: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> Problem:
    """Assemble a Problem; ``minima_f[0]`` is the global minimum value."""
    if len(minima) != len(minima_f) or not 1 <= n_global <= len(minima):
        raise ValueError(f"{id}: inconsistent minima metadata")
    return Problem(
        id=id,
        name=name,
        latex=latex,
        f=f,
        dim=len(x0),
        domain=tuple((float(lo), float(hi)) for lo, hi in domain),
        grad=grad,
        hess=hess,
        x0=[float(v) for v in x0],
        minima=tuple([float(v) for v in m] for m in minima),
        description=description,
        tags=tags,
        extra={
            "minima_f": [float(v) for v in minima_f],
            "n_global": n_global,
            "f_min": float(minima_f[0]),
            **(extra or {}),
        },
    )


# --------------------------------------------------------------------------------------
# Quadratics: f(x) = ½ (x − c)ᵀ A (x − c), ∇f = A(x − c), ∇²f = A
# --------------------------------------------------------------------------------------


def _quadratic(
    A: Array, c: Array
) -> tuple[Callable[..., Any], Callable[..., Array], Callable[..., Array]]:
    A = A.copy()
    c = c.copy()

    def f(x: ArrayLike) -> Any:
        d = _arr(x) - c.reshape((-1,) + (1,) * (np.ndim(x) - 1))  # (n, *grid)
        return 0.5 * np.einsum("i...,ij,j...->...", d, A, d)

    def grad(x: ArrayLike) -> Array:
        return A @ (_arr(x) - c)

    def hess(x: ArrayLike) -> Array:
        return A.copy()

    return f, grad, hess


@factory("unconstrained")
def quadratic_bowl() -> Problem:
    # Eigenvalues 2 and 4 (cond 2), eigenvectors (1, 1)/√2 and (1, −1)/√2.
    A = np.array([[3.0, 1.0], [1.0, 3.0]])
    c = np.array([1.0, -0.5])
    f, grad, hess = _quadratic(A, c)
    return _problem(
        id="quadratic_bowl",
        name="Quadratic bowl",
        latex=r"f(x,y) = \tfrac12\left(3(x-1)^2 + 2(x-1)(y+\tfrac12) + 3(y+\tfrac12)^2\right)",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-3.0, 3.0), (-3.0, 3.0)),
        x0=(-2.0, 2.0),
        minima=((1.0, -0.5),),
        minima_f=(0.0,),
        n_global=1,
        description="A well-conditioned convex quadratic (Hessian eigenvalues 2 and 4, "
        "condition number 2). Every descent method should converge quickly.",
        tags=("quadratic", "convex", "well-conditioned", "2d"),
        extra={"A": A.tolist(), "c": c.tolist()},
    )


@factory("unconstrained")
def quadratic_ill() -> Problem:
    # A = Q diag(1, 50) Qᵀ with Q = [[0.8, -0.6], [0.6, 0.8]] (a rotation by atan(3/4) ≈ 36.87°);
    # the entries are the exact decimals 18.64, -23.52, 32.36, and cond(A) = 50.
    Q = np.array([[0.8, -0.6], [0.6, 0.8]])
    A = Q @ np.diag([1.0, 50.0]) @ Q.T
    A = 0.5 * (A + A.T)
    c = np.zeros(2)
    f, grad, hess = _quadratic(A, c)
    return _problem(
        id="quadratic_ill",
        name="Ill-conditioned quadratic",
        latex=r"f(x) = \tfrac12 x^\top Q\,\mathrm{diag}(1, 50)\,Q^\top x,\ "
        r"Q = \begin{pmatrix}0.8 & -0.6\\ 0.6 & 0.8\end{pmatrix}",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-3.0, 3.0), (-3.0, 3.0)),
        x0=(-2.0, 2.0),
        minima=((0.0, 0.0),),
        minima_f=(0.0,),
        n_global=1,
        description="A rotated convex quadratic with condition number 50: long, narrow, "
        "tilted elliptic contours. Steepest descent zig-zags; Newton converges in one step.",
        tags=("quadratic", "convex", "ill-conditioned", "2d"),
        extra={"A": A.tolist(), "c": c.tolist()},
    )


@factory("unconstrained")
def quadratic_nd() -> Problem:
    n = 20
    seed = 20
    # Construction (replayed exactly by the TypeScript port; draws from Rng(seed) in order):
    #   1. v_i = rng.normal() for i = 0..n-1      → Householder reflector H = I − 2 v vᵀ / vᵀv
    #   2. c_i = rng.uniform(-2, 2) for i = 0..n-1 → the minimizer
    #   λ_i = 10^(2 i / (n − 1)) (geometric from 1 to 100) and A = H diag(λ) H.
    # H is orthogonal and symmetric, so A is SPD with eigenvalues λ and cond(A) = 100.
    rng = Rng(seed)
    v = np.array([rng.normal() for _ in range(n)])
    c = np.array([rng.uniform(-2.0, 2.0) for _ in range(n)])
    lam = 10.0 ** (2.0 * np.arange(n) / (n - 1))
    H = np.eye(n) - (2.0 / (v @ v)) * np.outer(v, v)
    A = H @ np.diag(lam) @ H
    A = 0.5 * (A + A.T)
    f, grad, hess = _quadratic(A, c)
    return _problem(
        id="quadratic_nd",
        name="Random SPD quadratic (n = 20)",
        latex=r"f(x) = \tfrac12 (x - c)^\top A (x - c),\ A \succ 0,\ \kappa(A) = 100",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-3.0, 3.0),) * n,
        x0=(0.0,) * n,
        minima=(c.tolist(),),
        minima_f=(0.0,),
        n_global=1,
        description="A 20-dimensional SPD quadratic A = H diag(λ) H, with H a Householder "
        "reflector and λ geometric from 1 to 100 (seeded with numopt.core.rng.Rng(20)).",
        tags=("quadratic", "convex", "n-d"),
        extra={"A": A.tolist(), "c": c.tolist(), "eigenvalues": lam.tolist(), "seed": seed},
    )


# --------------------------------------------------------------------------------------
# Classic 2-D test functions
# --------------------------------------------------------------------------------------


def _rosen_f(x: ArrayLike) -> Any:
    x = _arr(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2, axis=0)


def _rosen_grad(x: ArrayLike) -> Array:
    x = _arr(x)
    d = x[1:] - x[:-1] ** 2  # (n-1,)
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * d
    return g


def _rosen_hess(x: ArrayLike) -> Array:
    x = _arr(x)
    n = x.size
    diag = np.zeros(n)
    diag[:-1] = 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
    diag[1:] += 200.0
    H = np.diag(diag)
    off = -400.0 * x[:-1]
    idx = np.arange(n - 1)
    H[idx, idx + 1] = off
    H[idx + 1, idx] = off
    return H


@factory("unconstrained")
def rosenbrock() -> Problem:
    return _problem(
        id="rosenbrock",
        name="Rosenbrock",
        latex=r"f(x,y) = (1-x)^2 + 100\,(y-x^2)^2",
        f=_rosen_f,
        grad=_rosen_grad,
        hess=_rosen_hess,
        domain=((-2.0, 2.0), (-1.0, 3.0)),
        x0=(-1.2, 1.0),
        minima=((1.0, 1.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Rosenbrock's banana valley (MGH problem 1, standard start (−1.2, 1)). "
        "Finding the valley is easy; following its curved floor to (1, 1) is hard.",
        tags=("nonconvex", "valley", "classic", "2d"),
    )


@factory("unconstrained")
def rosenbrock_nd() -> Problem:
    n = 10
    x0 = [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]
    local = [
        -0.993263372856369,
        0.9966060394434352,
        0.9982406113911125,
        0.9989884337678007,
        0.999226153439665,
        0.9990736481633443,
        0.9984541774029088,
        0.9970562516630377,
        0.9941793752280661,
        0.9883926301288679,
    ]
    return _problem(
        id="rosenbrock_nd",
        name="Rosenbrock (n = 10)",
        latex=r"f(x) = \sum_{i=1}^{n-1} \left[100\,(x_{i+1}-x_i^2)^2 + (1-x_i)^2\right],\ n = 10",
        f=_rosen_f,
        grad=_rosen_grad,
        hess=_rosen_hess,
        domain=((-2.0, 2.0),) * n,
        x0=x0,
        minima=([1.0] * n, local),
        minima_f=(0.0, float(_rosen_f(np.array(local)))),
        n_global=1,
        description="The extended Rosenbrock function in 10 dimensions (the chained form used "
        "by scipy.optimize.rosen). Besides the global minimizer (1, …, 1) it has a local "
        "minimizer near (−1, 1, …, 1) (Shang & Qiu 2006 show this for 4 ≤ n ≤ 30).",
        tags=("nonconvex", "valley", "n-d"),
    )


@factory("unconstrained")
def himmelblau() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        return np.array([4.0 * x[0] * a + 2.0 * b, 2.0 * a + 4.0 * x[1] * b])

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        h11 = 12.0 * x[0] ** 2 + 4.0 * x[1] - 42.0
        h12 = 4.0 * (x[0] + x[1])
        h22 = 4.0 * x[0] + 12.0 * x[1] ** 2 - 26.0
        return np.array([[h11, h12], [h12, h22]])

    minima = (
        (3.0, 2.0),
        (-2.805118086952745, 3.131312518250573),
        (-3.779310253377747, -3.283185991286169),
        (3.5844283403304917, -1.8481265269644036),
    )
    return _problem(
        id="himmelblau",
        name="Himmelblau",
        latex=r"f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-5.0, 5.0), (-5.0, 5.0)),
        x0=(0.0, 0.0),
        minima=minima,
        minima_f=(0.0, 0.0, 0.0, 0.0),
        n_global=4,
        description="Himmelblau's function has four global minimizers, all with f = 0, and a "
        "local maximum near (−0.2708, −0.9230). Which minimizer a method finds depends on "
        "the start point.",
        tags=("nonconvex", "multiple-minima", "classic", "2d"),
    )


_BEALE_C = (1.5, 2.25, 2.625)


@factory("unconstrained")
def beale() -> Problem:
    # f = Σ_{i=1}^{3} t_i², t_i = c_i − x + x yⁱ  (MGH problem 5).
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return sum((c - x[0] + x[0] * x[1] ** i) ** 2 for i, c in enumerate(_BEALE_C, start=1))

    def grad(x: ArrayLike) -> Array:
        x0, y = _arr(x)
        g = np.zeros(2)
        for i, c in enumerate(_BEALE_C, start=1):
            t = c - x0 + x0 * y**i
            g += 2.0 * t * np.array([y**i - 1.0, i * x0 * y ** (i - 1)])
        return g

    def hess(x: ArrayLike) -> Array:
        x0, y = _arr(x)
        H = np.zeros((2, 2))
        for i, c in enumerate(_BEALE_C, start=1):
            t = c - x0 + x0 * y**i
            dt = np.array([y**i - 1.0, i * x0 * y ** (i - 1)])
            d2_xy = i * y ** (i - 1)
            d2_yy = i * (i - 1) * x0 * y ** (i - 2) if i >= 2 else 0.0
            H += 2.0 * (np.outer(dt, dt) + t * np.array([[0.0, d2_xy], [d2_xy, d2_yy]]))
        return H

    return _problem(
        id="beale",
        name="Beale",
        latex=r"f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-4.5, 4.5), (-4.5, 4.5)),
        x0=(1.0, 1.0),
        minima=((3.0, 0.5),),
        minima_f=(0.0,),
        n_global=1,
        description="Beale's function (MGH problem 5, standard start (1, 1)): flat plateaus "
        "and steep ridges near the corners of the box; global minimum f(3, 0.5) = 0.",
        tags=("nonconvex", "classic", "2d"),
    )


@factory("unconstrained")
def booth() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return (x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        a = x[0] + 2.0 * x[1] - 7.0
        b = 2.0 * x[0] + x[1] - 5.0
        return np.array([2.0 * a + 4.0 * b, 4.0 * a + 2.0 * b])

    def hess(x: ArrayLike) -> Array:
        return np.array([[10.0, 8.0], [8.0, 10.0]])

    return _problem(
        id="booth",
        name="Booth",
        latex=r"f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-10.0, 10.0), (-10.0, 10.0)),
        x0=(-5.0, -5.0),
        minima=((1.0, 3.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Booth's function (J&Y 22): a convex quadratic with Hessian eigenvalues "
        "2 and 18 (condition number 9).",
        tags=("quadratic", "convex", "2d"),
    )


@factory("unconstrained")
def matyas() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.array([0.52 * x[0] - 0.48 * x[1], 0.52 * x[1] - 0.48 * x[0]])

    def hess(x: ArrayLike) -> Array:
        return np.array([[0.52, -0.48], [-0.48, 0.52]])

    return _problem(
        id="matyas",
        name="Matyas",
        latex=r"f(x,y) = 0.26\,(x^2 + y^2) - 0.48\,xy",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-10.0, 10.0), (-10.0, 10.0)),
        x0=(8.0, 2.0),
        minima=((0.0, 0.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Matyas' function (J&Y 71): a convex quadratic whose Hessian eigenvalues "
        "are 0.04 and 1 (condition number 25), so the valley along y = x is very flat.",
        tags=("quadratic", "convex", "ill-conditioned", "2d"),
    )


@factory("unconstrained")
def three_hump_camel() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        u, v = x[0], x[1]
        return 2.0 * u**2 - 1.05 * u**4 + u**6 / 6.0 + u * v + v**2

    def grad(x: ArrayLike) -> Array:
        u, v = _arr(x)
        return np.array([4.0 * u - 4.2 * u**3 + u**5 + v, u + 2.0 * v])

    def hess(x: ArrayLike) -> Array:
        u, _ = _arr(x)
        return np.array([[4.0 - 12.6 * u**2 + 5.0 * u**4, 1.0], [1.0, 2.0]])

    a, b = 1.747552345830289, -0.8737761729151445
    f_loc = float(f(np.array([a, b])))
    return _problem(
        id="three_hump_camel",
        name="Three-hump camel",
        latex=r"f(x,y) = 2x^2 - 1.05x^4 + \tfrac{x^6}{6} + xy + y^2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-3.0, 3.0), (-3.0, 3.0)),
        x0=(-2.0, 1.5),
        minima=((0.0, 0.0), (a, b), (-a, -b)),
        minima_f=(0.0, f_loc, f_loc),
        n_global=1,
        description="Three-hump camel (J&Y 29): a global minimum at the origin and two "
        "symmetric local minima (f ≈ 0.2986) separated by saddle points.",
        tags=("nonconvex", "multiple-minima", "2d"),
    )


@factory("unconstrained")
def six_hump_camel() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        u, v = x[0], x[1]
        return (4.0 - 2.1 * u**2 + u**4 / 3.0) * u**2 + u * v + (-4.0 + 4.0 * v**2) * v**2

    def grad(x: ArrayLike) -> Array:
        u, v = _arr(x)
        return np.array([8.0 * u - 8.4 * u**3 + 2.0 * u**5 + v, u - 8.0 * v + 16.0 * v**3])

    def hess(x: ArrayLike) -> Array:
        u, v = _arr(x)
        return np.array([[8.0 - 25.2 * u**2 + 10.0 * u**4, 1.0], [1.0, -8.0 + 48.0 * v**2]])

    g1 = (0.08984201310031807, -0.7126564030207396)
    l1 = (-1.7036067149699814, 0.7960835686726251)
    l2 = (1.6071047529201974, 0.5686514548841313)
    minima = (g1, (-g1[0], -g1[1]), l1, (-l1[0], -l1[1]), l2, (-l2[0], -l2[1]))
    return _problem(
        id="six_hump_camel",
        name="Six-hump camel",
        latex=r"f(x,y) = \left(4 - 2.1x^2 + \tfrac{x^4}{3}\right)x^2 + xy + (-4 + 4y^2)\,y^2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-2.0, 2.0), (-1.2, 1.2)),
        x0=(-1.5, -0.5),
        minima=minima,
        minima_f=tuple(float(f(np.array(m))) for m in minima),
        n_global=2,
        description="Six-hump camel (J&Y 30): two global minima f ≈ −1.0316 at "
        "±(0.0898, −0.7127) and four local minima, in point-symmetric pairs.",
        tags=("nonconvex", "multiple-minima", "classic", "2d"),
    )


@factory("unconstrained")
def goldstein_price() -> Problem:
    # f = A·B with A = 1 + u² P, B = 30 + v² Q (J&Y 58), where
    #   u = x + y + 1,  P = 19 − 14x + 3x² − 14y + 6xy + 3y²,
    #   v = 2x − 3y,    Q = 18 − 32x + 12x² + 48y − 36xy + 27y².
    # Product rule: ∇f = B∇A + A∇B, ∇²f = B∇²A + A∇²B + ∇A∇Bᵀ + ∇B∇Aᵀ, with
    #   ∇A = 2uP∇u + u²∇P and ∇²A = 2P∇u∇uᵀ + 2u(∇u∇Pᵀ + ∇P∇uᵀ) + u²∇²P   (∇²u = 0).
    du = np.array([1.0, 1.0])
    dv = np.array([2.0, -3.0])
    d2P = np.array([[6.0, 6.0], [6.0, 6.0]])
    d2Q = np.array([[24.0, -36.0], [-36.0, 54.0]])

    def parts(x: ArrayLike) -> tuple[Any, Any, Any, Any, Any, Any]:
        x = _arr(x)
        p, q = x[0], x[1]
        u = p + q + 1.0
        P = 19.0 - 14.0 * p + 3.0 * p**2 - 14.0 * q + 6.0 * p * q + 3.0 * q**2
        v = 2.0 * p - 3.0 * q
        Q = 18.0 - 32.0 * p + 12.0 * p**2 + 48.0 * q - 36.0 * p * q + 27.0 * q**2
        return p, q, u, P, v, Q

    def f(x: ArrayLike) -> Any:
        _, _, u, P, v, Q = parts(x)
        return (1.0 + u**2 * P) * (30.0 + v**2 * Q)

    def derivs(x: ArrayLike) -> tuple[float, float, Array, Array, Array, Array]:
        p, q, u, P, v, Q = parts(x)
        dP = np.array([-14.0 + 6.0 * p + 6.0 * q, -14.0 + 6.0 * p + 6.0 * q])
        dQ = np.array([-32.0 + 24.0 * p - 36.0 * q, 48.0 - 36.0 * p + 54.0 * q])
        A = 1.0 + u**2 * P
        B = 30.0 + v**2 * Q
        dA = 2.0 * u * P * du + u**2 * dP
        dB = 2.0 * v * Q * dv + v**2 * dQ
        d2A = 2.0 * P * np.outer(du, du) + 2.0 * u * (np.outer(du, dP) + np.outer(dP, du))
        d2A += u**2 * d2P
        d2B = 2.0 * Q * np.outer(dv, dv) + 2.0 * v * (np.outer(dv, dQ) + np.outer(dQ, dv))
        d2B += v**2 * d2Q
        return A, B, dA, dB, d2A, d2B

    def grad(x: ArrayLike) -> Array:
        A, B, dA, dB, _, _ = derivs(x)
        return B * dA + A * dB

    def hess(x: ArrayLike) -> Array:
        A, B, dA, dB, d2A, d2B = derivs(x)
        return B * d2A + A * d2B + np.outer(dA, dB) + np.outer(dB, dA)

    minima = ((0.0, -1.0), (-0.6, -0.4), (1.8, 0.2), (1.2, 0.8))
    return _problem(
        id="goldstein_price",
        name="Goldstein–Price",
        latex=r"f(x,y) = \left[1 + (x+y+1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\right]"
        r"\left[30 + (2x-3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\right]",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=(-1.0, 1.0),
        minima=minima,
        minima_f=(3.0, 30.0, 84.0, 840.0),
        n_global=1,
        description="Goldstein–Price (J&Y 58): global minimum f(0, −1) = 3 and local minima "
        "f = 30, 84, 840; values span six orders of magnitude over the box.",
        tags=("nonconvex", "multiple-minima", "badly-scaled", "2d"),
    )


@factory("unconstrained")
def rastrigin() -> Problem:
    # f = 10 n + Σ (x_i² − 10 cos 2πx_i)   (J&Y does not list it; Törn & Žilinskas 1989).
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return 10.0 * x.shape[0] + np.sum(x**2 - 10.0 * np.cos(_TWO_PI * x), axis=0)

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return 2.0 * x + 20.0 * _PI * np.sin(_TWO_PI * x)

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.diag(2.0 + 40.0 * _PI**2 * np.cos(_TWO_PI * x))

    # Nearest local minima: one coordinate solves x + 10π sin(2πx) = 0 near ±1.
    a = 0.9949586376523347
    near = ((a, 0.0), (-a, 0.0), (0.0, a), (0.0, -a))
    f_near = float(f(np.array([a, 0.0])))
    return _problem(
        id="rastrigin",
        name="Rastrigin",
        latex=r"f(x) = 10n + \sum_{i=1}^{n}\left(x_i^2 - 10\cos 2\pi x_i\right),\ n = 2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-5.12, 5.12), (-5.12, 5.12)),
        x0=(3.3, -2.6),
        minima=((0.0, 0.0), *near),
        minima_f=(0.0, f_near, f_near, f_near, f_near),
        n_global=1,
        description="Rastrigin's function: a paraboloid covered by a regular grid of local "
        "minima near the integer points. Only the four local minima nearest the origin are "
        "listed; local methods stop in whichever basin they start.",
        tags=("nonconvex", "multimodal", "2d"),
    )


@factory("unconstrained")
def ackley() -> Problem:
    # f = −20 exp(−0.2 r) − exp(s) + e + 20 with r = √(Σx_i²/n), s = (1/n) Σ cos 2πx_i (J&Y 1).
    n = 2

    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        r = np.sqrt(np.sum(x**2, axis=0) / n)
        s = np.sum(np.cos(_TWO_PI * x), axis=0) / n
        # NOTE: −20(e^{−0.2r} − 1) − e(e^{s−1} − 1) is the same function, written with expm1 so
        # that f(0) is exactly 0 and f keeps its relative accuracy near the minimum.
        return -20.0 * np.expm1(-0.2 * r) - math.e * np.expm1(s - 1.0)

    def _smooth_part(x: Array) -> tuple[Array, Array]:
        # Gradient and Hessian of −exp(s): ∂/∂x_i = (2π/n) eˢ sin 2πx_i.
        es = math.exp(float(np.sum(np.cos(_TWO_PI * x))) / n)
        sn = np.sin(_TWO_PI * x)
        cs = np.cos(_TWO_PI * x)
        g = (_TWO_PI / n) * es * sn
        H = (_TWO_PI / n) * es * (np.diag(_TWO_PI * cs) - (_TWO_PI / n) * np.outer(sn, sn))
        return g, H

    def _radius(x: Array) -> float:
        # math.hypot scales internally: np.linalg.norm underflows to 0 for |x| ≲ 1e-154 and
        # would send a nonzero x into the r == 0 branch below.
        return math.hypot(*(float(v) for v in x)) / math.sqrt(n)

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        g, _ = _smooth_part(x)
        r = _radius(x)
        if r == 0.0:
            # NOTE: the cone term −20 exp(−0.2 r) is not differentiable at x = 0 (like ‖x‖);
            # 0 is in its subdifferential and is the minimizer, so we return the zero gradient.
            return g
        dr = x / (n * r)  # ∇r
        return g + 4.0 * math.exp(-0.2 * r) * dr

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        _, H = _smooth_part(x)
        r = _radius(x)
        if r == 0.0:
            # NOTE: the cone term has no Hessian at 0 (its curvature is +∞ radially); we return
            # the Hessian of the smooth cosine part only, so callers get a finite matrix.
            return H
        dr = x / (n * r)
        d2r = (np.eye(n) / n - np.outer(dr, dr)) / r  # ∇²r = (I/n − ∇r∇rᵀ)/r
        return H + 4.0 * math.exp(-0.2 * r) * (d2r - 0.2 * np.outer(dr, dr))

    return _problem(
        id="ackley",
        name="Ackley",
        latex=r"f(x) = -20\,e^{-0.2\sqrt{\frac1n\sum x_i^2}} - e^{\frac1n\sum\cos 2\pi x_i} + e + 20",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-5.0, 5.0), (-5.0, 5.0)),
        x0=(2.6, -3.4),
        minima=((0.0, 0.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Ackley's function (J&Y 1): a nearly flat outer region with many local "
        "minima and a deep, narrow funnel at the origin. f has a cone-shaped kink at the "
        "origin; grad(0) returns the zero subgradient.",
        tags=("nonconvex", "multimodal", "nonsmooth-at-minimum", "2d"),
    )


# Stationary points of ½(x⁴ − 16x² + 5x): the roots of 2x³ − 16x + 2.5 = 0.
_ST_GLOBAL = -2.903534027771177
_ST_LOCAL = 2.746802770990837


@factory("unconstrained")
def styblinski_tang() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=0)

    def grad(x: ArrayLike) -> Array:
        x = _arr(x)
        return 2.0 * x**3 - 16.0 * x + 2.5

    def hess(x: ArrayLike) -> Array:
        x = _arr(x)
        return np.diag(6.0 * x**2 - 16.0)

    a, b = _ST_GLOBAL, _ST_LOCAL
    minima = ((a, a), (a, b), (b, a), (b, b))
    return _problem(
        id="styblinski_tang",
        name="Styblinski–Tang",
        latex=r"f(x) = \tfrac12\sum_{i=1}^{n}\left(x_i^4 - 16x_i^2 + 5x_i\right),\ n = 2",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-5.0, 5.0), (-5.0, 5.0)),
        x0=(0.5, 0.5),
        minima=minima,
        minima_f=tuple(float(f(np.array(m))) for m in minima),
        n_global=1,
        description="Styblinski–Tang (J&Y 144): separable; each coordinate has a deep and a "
        "shallow well, giving one global minimum (f ≈ −78.332) and three local minima.",
        tags=("nonconvex", "multiple-minima", "separable", "2d"),
    )


@factory("unconstrained")
def mccormick() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.0

    def grad(x: ArrayLike) -> Array:
        p, q = _arr(x)
        c = math.cos(p + q)
        return np.array([c + 2.0 * (p - q) - 1.5, c - 2.0 * (p - q) + 2.5])

    def hess(x: ArrayLike) -> Array:
        p, q = _arr(x)
        s = math.sin(p + q)
        return np.array([[2.0 - s, -2.0 - s], [-2.0 - s, 2.0 - s]])

    # Stationarity gives x − y = 1 and cos(x + y) = −½; the Hessian is PD iff sin(x + y) < 0, so
    # the minimizers are x + y = σ_k = −2π/3 + 2πk, x = (σ_k + 1)/2, y = (σ_k − 1)/2, and
    # f = σ_k/2 − √3/2. Inside the box only k = 0 (global over the box) and k = 1 lie.
    def point(k: int) -> tuple[float, float]:
        sigma = -2.0 * _PI / 3.0 + _TWO_PI * k
        return (0.5 * (sigma + 1.0), 0.5 * (sigma - 1.0))

    minima = (point(0), point(1))
    return _problem(
        id="mccormick",
        name="McCormick",
        latex=r"f(x,y) = \sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-1.5, 4.0), (-3.0, 4.0)),
        x0=(3.0, -2.0),
        minima=minima,
        minima_f=(
            -math.pi / 3.0 - math.sqrt(3.0) / 2.0,
            2.0 * math.pi / 3.0 - math.sqrt(3.0) / 2.0,
        ),
        n_global=1,
        description="McCormick's function (J&Y 80) on its box [−1.5, 4] × [−3, 4]: global "
        "minimum f(−0.5472, −1.5472) = −π/3 − √3/2 ≈ −1.9132 over the box, and a local "
        "minimum at (2.5944, 1.5944). Unconstrained, f is unbounded below along x − y = 1 "
        "(f = σ/2 + sin σ with σ = x + y → −∞), so the global label holds on the box only.",
        tags=("nonconvex", "bounded-domain", "2d"),
    )


@factory("unconstrained")
def bohachevsky() -> Problem:
    # Bohachevsky function 1 (J&Y 17).
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        return (
            x[0] ** 2
            + 2.0 * x[1] ** 2
            - 0.3 * np.cos(3.0 * _PI * x[0])
            - 0.4 * np.cos(4.0 * _PI * x[1])
            + 0.7
        )

    def grad(x: ArrayLike) -> Array:
        p, q = _arr(x)
        return np.array(
            [
                2.0 * p + 0.9 * _PI * math.sin(3.0 * _PI * p),
                4.0 * q + 1.6 * _PI * math.sin(4.0 * _PI * q),
            ]
        )

    def hess(x: ArrayLike) -> Array:
        p, q = _arr(x)
        return np.diag(
            [
                2.0 + 2.7 * _PI**2 * math.cos(3.0 * _PI * p),
                4.0 + 6.4 * _PI**2 * math.cos(4.0 * _PI * q),
            ]
        )

    return _problem(
        id="bohachevsky",
        name="Bohachevsky",
        latex=r"f(x,y) = x^2 + 2y^2 - 0.3\cos 3\pi x - 0.4\cos 4\pi y + 0.7",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        x0=(0.8, -0.7),
        minima=((0.0, 0.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Bohachevsky function 1 (J&Y 17): a convex bowl with cosine ripples that "
        "create many shallow local minima; global minimum f(0, 0) = 0.",
        tags=("nonconvex", "multimodal", "2d"),
    )


@factory("unconstrained")
def levi13() -> Problem:
    def f(x: ArrayLike) -> Any:
        x = _arr(x)
        p, q = x[0], x[1]
        return (
            np.sin(3.0 * _PI * p) ** 2
            + (p - 1.0) ** 2 * (1.0 + np.sin(3.0 * _PI * q) ** 2)
            + (q - 1.0) ** 2 * (1.0 + np.sin(_TWO_PI * q) ** 2)
        )

    def grad(x: ArrayLike) -> Array:
        p, q = _arr(x)
        s3q = math.sin(3.0 * _PI * q)
        s2q = math.sin(_TWO_PI * q)
        gx = 3.0 * _PI * math.sin(6.0 * _PI * p) + 2.0 * (p - 1.0) * (1.0 + s3q**2)
        gy = (
            3.0 * _PI * (p - 1.0) ** 2 * math.sin(6.0 * _PI * q)
            + 2.0 * (q - 1.0) * (1.0 + s2q**2)
            + _TWO_PI * (q - 1.0) ** 2 * math.sin(4.0 * _PI * q)
        )
        return np.array([gx, gy])

    def hess(x: ArrayLike) -> Array:
        p, q = _arr(x)
        s3q = math.sin(3.0 * _PI * q)
        s2q = math.sin(_TWO_PI * q)
        h11 = 18.0 * _PI**2 * math.cos(6.0 * _PI * p) + 2.0 * (1.0 + s3q**2)
        h12 = 6.0 * _PI * (p - 1.0) * math.sin(6.0 * _PI * q)
        h22 = (
            18.0 * _PI**2 * (p - 1.0) ** 2 * math.cos(6.0 * _PI * q)
            + 2.0 * (1.0 + s2q**2)
            + 8.0 * _PI * (q - 1.0) * math.sin(4.0 * _PI * q)
            + 8.0 * _PI**2 * (q - 1.0) ** 2 * math.cos(4.0 * _PI * q)
        )
        return np.array([[h11, h12], [h12, h22]])

    return _problem(
        id="levi13",
        name="Lévi N.13",
        latex=r"f(x,y) = \sin^2 3\pi x + (x-1)^2(1 + \sin^2 3\pi y) + (y-1)^2(1 + \sin^2 2\pi y)",
        f=f,
        grad=grad,
        hess=hess,
        domain=((-2.0, 4.0), (-2.0, 4.0)),
        x0=(-1.3, 3.1),
        minima=((1.0, 1.0),),
        minima_f=(0.0,),
        n_global=1,
        description="Lévi function N.13: oscillating terms create rows of local minima "
        "around the global minimum f(1, 1) = 0.",
        tags=("nonconvex", "multimodal", "2d"),
    )
