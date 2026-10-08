"""Finite-sum objectives of machine learning (kind ``"stochastic"``).

Every problem is a regularized empirical risk over n samples (aᵢ, yᵢ) with a linear predictor
zᵢ = aᵢᵀw (Bottou, Curtis & Nocedal (2018), eq. 2.3 / §3):

    f(w) = (1/n) Σᵢ fᵢ(w),        fᵢ(w) = φ(aᵢᵀw, yᵢ) + (λ/2)‖w‖²,

    ∇fᵢ(w)  = φ'(aᵢᵀw, yᵢ) aᵢ + λw,
    ∇f(w)   = (1/n) Aᵀ φ'(Aw, y) + λw,
    ∇²f(w)  = (1/n) Aᵀ diag(φ''(Aw, y)) A + λI,

where A ∈ ℝⁿˣᵈ stacks the feature rows aᵢᵀ (``X``). The regularizer is inside every fᵢ, so the
stochastic methods (SAG, SAGA, SVRG) treat it like any other part of the component gradient.

Per-sample losses φ(z, y) (``loss``):

* ``"squared"``:  φ = ½(z − y)²,  φ' = z − y,  φ'' = 1.  (f is ½·MSE.)
* ``"logistic"`` (y ∈ {0, 1}, binary cross-entropy):
  φ = log(1 + eᶻ) − yz,  φ' = σ(z) − y,  φ'' = σ(z)σ(−z).
  Evaluated without overflow as φ = max(z, 0) + log1p(e^{−|z|}) − yz,
  φ' = (1 − y)σ(z) − yσ(−z) (no cancellation for y ∈ {0, 1}), φ'' = e/(1 + e)² with
  e = e^{−|z|}, and σ(z) = 1/(1 + e) for z ≥ 0, e/(1 + e) for z < 0.
* ``"huber"`` (Huber 1964, threshold δ = ``huber_delta``), r = z − y:
  φ = ½r² if |r| ≤ δ else δ(|r| − ½δ),  φ' = clip(r, −δ, δ),  φ'' = 1 if |r| ≤ δ else 0.
  φ'' does not exist at |r| = δ; ``hess`` takes the value 1 there (closed quadratic region).

Mini-batch gradients. ``grad_batch(w, idx)`` returns (1/|B|) Σ_{i∈B} ∇fᵢ(w) for the multiset
B = idx. It sorts ``idx`` first, so the result is bit-for-bit independent of the order of
``idx``, and ``grad(w)`` *is* ``grad_batch(w, range(n))``. A full-batch SGD step therefore
equals a gradient-descent step exactly.

Data. Every data set is drawn only from :class:`numopt.core.rng.Rng` (Mulberry32) in the order
that its builder documents, so the TypeScript port regenerates the identical samples. The arrays
are read-only.

Minimizers. ``minima[0]`` is the unique minimizer: the least-squares solution (``numpy.linalg
.lstsq``, an SVD; the normal equations are never formed) for the squared loss, and damped Newton
iterated to ‖∇f‖ ≈ 1e-16 for the logistic and Huber losses. The tests verify both against SciPy.

Ids (n = 200 samples, d = 2 parameters each, so the loss landscape can be drawn):
    linreg_2d             y ≈ w₀ + w₁x, squared loss, x ~ U(−1, 3)
    logreg_2d             P(y = 1 | x) = σ(w₀ + w₁x), logistic loss + (10⁻²/2)‖w‖²
    ill_conditioned_ls    y ≈ w₀u + w₁(30v), u, v ~ N(0, 1): feature scales 1 and 30
    huber_regression_2d   y ≈ w₀ + w₁x with 10% gross outliers, Huber loss (δ = 1)

Extra keys (``FiniteSumProblem.extra``):
    seed: int             the Rng seed of the data.
    true_w: [d]           the parameters that generated the data.
    noise_std: float      the std of the Gaussian label noise (regression problems).
    f_min: float          f(minima[0]).
    L: float              λ_max of the Hessian bound: (1/n)λ_max(AᵀA)·c + λ, with c = 1
                          (squared, Huber) or ¼ (logistic); f is L-smooth.
    L_max: float          maxᵢ (c‖aᵢ‖² + λ): every fᵢ is L_max-smooth (step-size scale of
                          single-sample SGD/SAG/SAGA).
    mu: float             a global strong-convexity constant (λ_min(AᵀA)/n + λ for the squared
                          loss, λ for the logistic loss, 0 for Huber, which is not strongly convex
                          far from the data).
    model: str            LaTeX of the predictor.
    outliers: [int]       (huber_regression_2d) indices of the corrupted samples.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.rng import Rng
from ..core.types import to_jsonable
from .registry import factory

Array = NDArray[np.float64]
Loss = Literal["squared", "logistic", "huber"]

#: Number of samples of every library problem.
N_SAMPLES = 200


def _frozen(values: ArrayLike) -> Array:
    """A read-only float64 copy of ``values``."""
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.flags.writeable = False
    return arr


# --------------------------------------------------------------------------------------
# Per-sample losses φ(z, y) of the linear predictor z = aᵀw
# --------------------------------------------------------------------------------------


def _sigmoids(z: Array) -> tuple[Array, Array, Array]:
    """Return (σ(z), σ(−z), e^{−|z|}) without overflow."""
    e = np.exp(-np.abs(z))
    big = 1.0 / (1.0 + e)  # σ(|z|)
    small = e / (1.0 + e)  # σ(−|z|)
    pos = z >= 0
    return np.where(pos, big, small), np.where(pos, small, big), e


def _phi(loss: Loss, z: Array, y: Array, delta: float) -> Array:
    """φ(z, y) elementwise."""
    if loss == "squared":
        r = z - y
        return 0.5 * r * r
    if loss == "logistic":
        return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z
    r = np.abs(z - y)
    return np.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta))


def _dphi(loss: Loss, z: Array, y: Array, delta: float) -> Array:
    """∂φ/∂z elementwise."""
    if loss == "squared":
        return z - y
    if loss == "logistic":
        s_pos, s_neg, _ = _sigmoids(z)
        return (1.0 - y) * s_pos - y * s_neg
    return np.clip(z - y, -delta, delta)


def _d2phi(loss: Loss, z: Array, y: Array, delta: float) -> Array:
    """∂²φ/∂z² elementwise (Huber: 1 on the closed region |r| ≤ δ)."""
    if loss == "squared":
        return np.ones_like(z)
    if loss == "logistic":
        _, _, e = _sigmoids(z)
        return e / ((1.0 + e) * (1.0 + e))
    return (np.abs(z - y) <= delta).astype(np.float64)


# --------------------------------------------------------------------------------------
# The problem type
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FiniteSumProblem:
    """f(w) = (1/n) Σᵢ [φ(aᵢᵀw, yᵢ) + (λ/2)‖w‖²] over a data set (X, y).

    Attributes:
        id, name, latex: Registry id, display name and LaTeX of f.
        X: Feature matrix A (n × d), one row aᵢ per sample (read-only).
        y: Targets (n,) — real values, or labels in {0, 1} for the logistic loss (read-only).
        loss: ``"squared"``, ``"logistic"`` or ``"huber"`` (see the module docstring).
        domain: Plotting box ((lo₀, hi₀), (lo₁, hi₁)) in parameter space.
        x0: Default starting weights.
        minima: The minimizer(s); ``minima[0]`` is the unique global minimizer.
        l2: The ridge weight λ ≥ 0.
        huber_delta: The Huber threshold δ > 0 (ignored by the other losses).
        description, tags, extra: UI text, tags, and the keys listed in the module docstring.
    """

    id: str
    name: str
    latex: str
    X: Array
    y: Array
    loss: Loss
    domain: tuple[tuple[float, float], ...]
    x0: list[float]
    minima: tuple[list[float], ...] = ()
    l2: float = 0.0
    huber_delta: float = 1.0
    description: str = ""
    tags: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "stochastic"

    @property
    def dim(self) -> int:
        """Number of parameters d."""
        return int(self.X.shape[1])

    @property
    def n_samples(self) -> int:
        """Number of samples n (components of the finite sum)."""
        return int(self.X.shape[0])

    @property
    def data(self) -> tuple[Array, Array]:
        """The data set ``(X, y)``."""
        return self.X, self.y

    # -- objective -----------------------------------------------------------------------

    def f(self, w: ArrayLike) -> Any:
        """Full average loss f(w).

        ``w`` has shape ``(d,)`` (returns a float) or ``(d, *grid)`` for contour grids
        (returns an array of shape ``grid``).
        """
        w = np.asarray(w, dtype=np.float64)
        z = np.einsum("nd,d...->n...", self.X, w)  # (n, *grid)
        y = self.y.reshape((-1,) + (1,) * (w.ndim - 1))
        val = np.mean(_phi(self.loss, z, y, self.huber_delta), axis=0)
        val = val + 0.5 * self.l2 * np.sum(w * w, axis=0)
        return float(val) if w.ndim == 1 else val

    def _index(self, idx: ArrayLike) -> NDArray[np.intp]:
        ii = np.asarray(idx, dtype=np.intp).reshape(-1)
        if ii.size == 0:
            raise ValueError("a mini-batch needs at least one index")
        if ii.min() < 0 or ii.max() >= self.n_samples:
            raise ValueError(f"sample indices must lie in [0, {self.n_samples})")
        return ii

    def grad_batch(self, w: ArrayLike, idx: ArrayLike) -> Array:
        """Mini-batch gradient (1/|B|) Σ_{i∈B} ∇fᵢ(w) = (1/|B|) A_Bᵀ φ'(A_B w, y_B) + λw.

        ``idx`` is sorted first, so the value does not depend on the order of ``idx``.
        Repeated indices count with their multiplicity.
        """
        w = np.asarray(w, dtype=np.float64)
        ii = np.sort(self._index(idx))
        A = self.X[ii]  # (b, d)
        s = _dphi(self.loss, A @ w, self.y[ii], self.huber_delta)  # (b,)
        return (A.T @ s) / ii.size + self.l2 * w

    def grad(self, w: ArrayLike) -> Array:
        """Full gradient ∇f(w) (identical to ``grad_batch(w, range(n))``)."""
        return self.grad_batch(w, np.arange(self.n_samples))

    def grad_samples(self, w: ArrayLike, idx: ArrayLike) -> Array:
        """Per-sample gradients ∇fᵢ(w) = φ'(aᵢᵀw, yᵢ)aᵢ + λw, one row per entry of ``idx``.

        Rows follow the order of ``idx`` (no sorting). Shape ``(len(idx), d)``.
        """
        w = np.asarray(w, dtype=np.float64)
        ii = self._index(idx)
        A = self.X[ii]  # (b, d)
        s = _dphi(self.loss, A @ w, self.y[ii], self.huber_delta)  # (b,)
        return s[:, None] * A + self.l2 * w[None, :]

    def hess(self, w: ArrayLike) -> Array:
        """∇²f(w) = (1/n) Aᵀ diag(φ''(Aw, y)) A + λI (Huber: φ'' = 1 on |r| ≤ δ)."""
        w = np.asarray(w, dtype=np.float64)
        c = _d2phi(self.loss, self.X @ w, self.y, self.huber_delta)  # (n,)
        H = np.einsum("n,ni,nj->ij", c, self.X, self.X) / self.n_samples
        H = H + self.l2 * np.eye(self.dim)
        return 0.5 * (H + H.T)

    def to_dict(self) -> dict[str, Any]:
        """Metadata and the data set (for plotting the samples and the fitted model)."""
        return to_jsonable(
            {
                "id": self.id,
                "name": self.name,
                "latex": self.latex,
                "kind": self.kind,
                "dim": self.dim,
                "n_samples": self.n_samples,
                "domain": self.domain,
                "x0": self.x0,
                "minima": self.minima,
                "loss": self.loss,
                "l2": self.l2,
                "huber_delta": self.huber_delta,
                "X": self.X,
                "y": self.y,
                "description": self.description,
                "tags": self.tags,
                "extra": dict(self.extra),
            }
        )


# --------------------------------------------------------------------------------------
# Exact minimizers and smoothness constants
# --------------------------------------------------------------------------------------

#: Newton stops once a full step is shorter than this relative size (the iterate is then
#: within a few ulps of the minimizer: Newton converges quadratically near it).
_NEWTON_XTOL = 1e-15
_EPS = float(np.finfo(np.float64).eps)
_NEWTON_MAX_ITER = 100
_ARMIJO_C1 = 1e-4


def _least_squares_minimizer(X: Array, y: Array, l2: float) -> Array:
    """argmin (1/2n)‖Xw − y‖² + (λ/2)‖w‖² via SVD least squares on the stacked system
    [X; √(nλ) I] w ≈ [y; 0] (the normal equations would square κ; Higham 2002, ch. 20)."""
    n, d = X.shape
    if l2 > 0.0:
        X = np.vstack([X, math.sqrt(n * l2) * np.eye(d)])
        y = np.concatenate([y, np.zeros(d)])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(w, dtype=np.float64)


def _newton_minimizer(p: FiniteSumProblem, w: Array) -> Array:
    """Damped Newton (Nocedal & Wright 2006, Alg. 3.2 with Armijo backtracking, Alg. 3.1).

    Stops when a unit Newton step is shorter than 1e-15·(1 + ‖w‖) and takes that last step;
    the Hessians here are positive definite (logistic: λ > 0; Huber: ≥ 2 inliers).
    """
    for _ in range(_NEWTON_MAX_ITER):
        g = p.grad(w)
        step = np.linalg.solve(p.hess(w), -g)
        if np.linalg.norm(step) <= _NEWTON_XTOL * (1.0 + np.linalg.norm(w)):
            # NOTE: take the last tiny step without a line search; at this size the Armijo
            # test compares f values that differ only by rounding.
            return w + step
        f0, slope, alpha = p.f(w), float(g @ step), 1.0
        # NOTE: the Armijo test gets a rounding allowance of 16ε|f(w)| (the approximate-Wolfe
        # idea of Hager & Zhang 2005). Near w* the predicted decrease gᵀs ≈ 1e-19 is below
        # the rounding of f, and the exact test would reject the (correct) Newton step.
        slack = 16.0 * _EPS * abs(f0)
        while p.f(w + alpha * step) > f0 + _ARMIJO_C1 * alpha * slope + slack and alpha > 1e-12:
            alpha *= 0.5
        w = w + alpha * step
    return w


def _constants(X: Array, loss: Loss, l2: float) -> dict[str, float]:
    """Global smoothness constants L, L_max and a strong-convexity constant mu."""
    n = X.shape[0]
    c = 0.25 if loss == "logistic" else 1.0
    eig = np.linalg.eigvalsh(X.T @ X / n)  # ascending
    row_sq = np.einsum("nd,nd->n", X, X)
    mu = {"squared": float(eig[0]) + l2, "logistic": l2, "huber": 0.0}[loss]
    return {"L": c * float(eig[-1]) + l2, "L_max": c * float(row_sq.max()) + l2, "mu": mu}


def _build(
    *,
    id: str,
    name: str,
    latex: str,
    X: Array,
    y: Array,
    loss: Loss,
    l2: float,
    domain: Sequence[tuple[float, float]],
    x0: Sequence[float],
    description: str,
    tags: tuple[str, ...],
    extra: dict[str, Any],
    huber_delta: float = 1.0,
) -> FiniteSumProblem:
    common: dict[str, Any] = {
        "id": id,
        "name": name,
        "latex": latex,
        "X": _frozen(X),
        "y": _frozen(y),
        "loss": loss,
        "domain": tuple((float(lo), float(hi)) for lo, hi in domain),
        "x0": [float(v) for v in x0],
        "l2": float(l2),
        "huber_delta": float(huber_delta),
        "description": description,
        "tags": ("finite-sum", *tags),
    }
    w_ls = _least_squares_minimizer(np.asarray(X), np.asarray(y), l2)
    if loss == "squared":
        w_star = w_ls
    else:
        # Logistic: start at 0 (f is convex; damped Newton is globally convergent).
        # Huber: start at the least-squares fit, which is close to the robust fit.
        start = np.zeros(X.shape[1]) if loss == "logistic" else w_ls
        w_star = _newton_minimizer(FiniteSumProblem(**common, minima=()), start)
    minimum = [float(v) for v in w_star]
    probe = FiniteSumProblem(**common, minima=(minimum,))
    return FiniteSumProblem(
        **common,
        minima=(minimum,),
        extra={**extra, "f_min": probe.f(w_star), **_constants(np.asarray(X), loss, l2)},
    )


def _design_with_intercept(x: Array) -> Array:
    """Rows aᵢ = (1, xᵢ)."""
    return np.column_stack([np.ones_like(x), x])


# --------------------------------------------------------------------------------------
# Library problems
# --------------------------------------------------------------------------------------


@factory("stochastic")
def linreg_2d() -> FiniteSumProblem:
    # Data (Rng(11)), for i = 0..199 in order: x_i = rng.uniform(-1, 3), then
    # y_i = 1 + 2 x_i + rng.normal(0, 0.5).
    seed, sigma, true_w = 11, 0.5, (1.0, 2.0)
    rng = Rng(seed)
    x = np.empty(N_SAMPLES)
    y = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        x[i] = rng.uniform(-1.0, 3.0)
        y[i] = true_w[0] + true_w[1] * x[i] + rng.normal(0.0, sigma)
    return _build(
        id="linreg_2d",
        name="Linear regression (2 parameters)",
        latex=r"f(w) = \frac{1}{2n}\sum_{i=1}^{n} (w_0 + w_1 x_i - y_i)^2",
        X=_design_with_intercept(x),
        y=y,
        loss="squared",
        l2=0.0,
        domain=((-3.0, 4.0), (-1.5, 4.5)),
        x0=(-2.0, -1.0),
        description="Fit a line to 200 noisy points. The loss is a quadratic bowl; the "
        "intercept and the slope are correlated because x is not centered (κ ≈ 6).",
        tags=("regression", "quadratic", "convex"),
        extra={
            "seed": seed,
            "true_w": list(true_w),
            "noise_std": sigma,
            "model": r"\hat y = w_0 + w_1 x",
        },
    )


@factory("stochastic")
def logreg_2d() -> FiniteSumProblem:
    # Data (Rng(12)), for i = 0..199 in order: x_i = rng.uniform(-3, 3), then
    # y_i = 1 if rng.random() < σ(0.5 + 2.5 x_i) else 0.
    seed, true_w, l2 = 12, (0.5, 2.5), 1e-2
    rng = Rng(seed)
    x = np.empty(N_SAMPLES)
    y = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        x[i] = rng.uniform(-3.0, 3.0)
        prob = 1.0 / (1.0 + math.exp(-(true_w[0] + true_w[1] * x[i])))
        y[i] = 1.0 if rng.random() < prob else 0.0
    return _build(
        id="logreg_2d",
        name="Logistic regression (2 parameters)",
        latex=r"f(w) = \frac{1}{n}\sum_{i=1}^{n} \left[\log(1 + e^{z_i}) - y_i z_i\right]"
        r" + \frac{\lambda}{2}\|w\|^2,\ z_i = w_0 + w_1 x_i,\ \lambda = 10^{-2}",
        X=_design_with_intercept(x),
        y=y,
        loss="logistic",
        l2=l2,
        domain=((-4.0, 4.0), (-2.0, 6.0)),
        x0=(-3.0, -1.0),
        description="Classify 200 points on a line into two nearly separable classes. "
        "The ridge term λ = 10⁻² makes the minimizer unique; far from it the loss is "
        "almost linear, near it almost quadratic.",
        tags=("classification", "convex", "strongly-convex"),
        extra={"seed": seed, "true_w": list(true_w), "model": r"P(y=1) = \sigma(w_0 + w_1 x)"},
    )


@factory("stochastic")
def ill_conditioned_ls() -> FiniteSumProblem:
    # Data (Rng(13)), for i = 0..199 in order: u_i = rng.normal(), v_i = rng.normal(),
    # a_i = (u_i, 30 v_i), y_i = a_i·(1, 0.5) + rng.normal(0, 0.5).
    seed, sigma, true_w, scale = 13, 0.5, (1.0, 0.5), 30.0
    rng = Rng(seed)
    X = np.empty((N_SAMPLES, 2))
    y = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        X[i, 0] = rng.normal()
        X[i, 1] = scale * rng.normal()
        y[i] = true_w[0] * X[i, 0] + true_w[1] * X[i, 1] + rng.normal(0.0, sigma)
    return _build(
        id="ill_conditioned_ls",
        name="Ill-conditioned least squares",
        latex=r"f(w) = \frac{1}{2n}\sum_{i=1}^{n} (w_0 u_i + 30\,w_1 v_i - y_i)^2",
        X=X,
        y=y,
        loss="squared",
        l2=0.0,
        domain=((-2.0, 4.0), (-1.0, 2.0)),
        x0=(-1.5, 1.5),
        description="Two features on scales 1 and 30: the Hessian has κ ≈ 1070, a long "
        "narrow valley. Plain SGD must use a tiny step; per-coordinate methods "
        "(AdaGrad, RMSProp, Adam) rescale the axes.",
        tags=("regression", "quadratic", "ill-conditioned"),
        extra={
            "seed": seed,
            "true_w": list(true_w),
            "noise_std": sigma,
            "model": r"\hat y = w_0 u + w_1 (30 v)",
        },
    )


@factory("stochastic")
def huber_regression_2d() -> FiniteSumProblem:
    # Data (Rng(14)), for i = 0..199 in order: x_i = rng.uniform(-1, 3),
    # e_i = rng.normal(0, 0.3), u_i = rng.random(), o_i = rng.uniform(5, 15) (always drawn);
    # y_i = 1 + 2 x_i + e_i, plus o_i when u_i < 0.1 (an outlier).
    seed, sigma, true_w, delta, p_out = 14, 0.3, (1.0, 2.0), 1.0, 0.1
    rng = Rng(seed)
    x = np.empty(N_SAMPLES)
    y = np.empty(N_SAMPLES)
    outliers: list[int] = []
    for i in range(N_SAMPLES):
        x[i] = rng.uniform(-1.0, 3.0)
        e = rng.normal(0.0, sigma)
        u = rng.random()
        o = rng.uniform(5.0, 15.0)
        y[i] = true_w[0] + true_w[1] * x[i] + e
        if u < p_out:
            y[i] += o
            outliers.append(i)
    return _build(
        id="huber_regression_2d",
        name="Huber regression (2 parameters)",
        latex=r"f(w) = \frac{1}{n}\sum_{i=1}^{n} h_\delta(w_0 + w_1 x_i - y_i),\ "
        r"h_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ "
        r"\delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases},\ \delta = 1",
        X=_design_with_intercept(x),
        y=y,
        loss="huber",
        l2=0.0,
        huber_delta=delta,
        domain=((-3.0, 5.0), (-1.5, 4.5)),
        x0=(-2.0, -1.0),
        description="A line fit with 10% gross outliers. The Huber loss is quadratic for "
        "small residuals and linear for large ones, so the outliers pull the fit far less "
        "than in least squares. f is convex and C¹ but its Hessian jumps.",
        tags=("regression", "robust", "convex", "nonsmooth-hessian"),
        extra={
            "seed": seed,
            "true_w": list(true_w),
            "noise_std": sigma,
            "outliers": outliers,
            "model": r"\hat y = w_0 + w_1 x",
        },
    )
