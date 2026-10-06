"""Datasets ``(x_i, y_i)`` for the interpolation and regression families.

Every dataset is a :class:`~numopt.core.types.Dataset`. When the data come from a known
function, ``f_true`` is that function (vectorized: it accepts a float or an array) and the
interpolation methods report their maximum error against it. For the noisy datasets,
``f_true`` is the noise-free regression function: E[y | x] for the additive-noise sets,
and the median of y | x for ``exponential_growth`` (multiplicative log-normal noise:
E[y | x] = f_true(x)·exp(σ²/2), 0.125% above f_true for σ = 0.05). Chebyshev
interpolation resamples ``f_true`` only when the data are its exact samples (the four
noise-free sets); on the noisy sets it interpolates the data like every other method.

Noise is drawn ONLY from :class:`numopt.core.rng.Rng` (Mulberry32), one ``normal`` draw per
point in increasing index order, so the TypeScript port can regenerate the identical data.
The arrays are read-only so that no method can mutate the shared library copy.

Ids:
    runge_equispaced    Runge function 1/(1+25x²), 11 equispaced nodes on [-1, 1]
    runge_chebyshev     Runge function at the 11 Chebyshev nodes of the first kind
    sine_samples        sin x at 9 equispaced nodes on [0, 2π]
    step_data           unit step H(x) at 12 equispaced nodes on [-1, 1] (no node at 0)
    noisy_linear        y = 2 + 0.5x + N(0, 0.5²), 20 points on [0, 10]
    noisy_quadratic     y = 1 - x + 0.5x² + N(0, 0.4²), 25 points on [-2, 4]
    anscombe_1          Anscombe's quartet, data set I (Anscombe 1973)
    outliers_linear     y = 1 + 2x + N(0, 0.3²), 20 points on [0, 10], two gross outliers
    exponential_growth  y = 2 exp(0.3x + N(0, 0.05²)), 11 points on [0, 10]
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.rng import Rng
from ..core.types import Dataset, Vector
from .registry import factory


def _frozen(values: Any) -> Vector:
    """A read-only float64 copy of ``values``."""
    arr = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    arr.flags.writeable = False
    return arr


def _normal_noise(seed: int, n: int, std: float) -> Vector:
    """``n`` draws of N(0, std²) from Mulberry32 seeded with ``seed``, in index order."""
    rng = Rng(seed)
    return np.array([rng.normal(0.0, std) for _ in range(n)], dtype=np.float64)


def runge(x: Any) -> Any:
    """Runge's function f(x) = 1 / (1 + 25 x²)."""
    return 1.0 / (1.0 + 25.0 * np.square(x))


def chebyshev_nodes_first_kind(n: int, a: float = -1.0, b: float = 1.0) -> Vector:
    """The n roots of T_n mapped to [a, b], in increasing order.

    u_j = cos(π(2j + 1)/(2n)), j = 0..n-1 (Burden & Faires, Numerical Analysis, 10th ed.,
    §8.3, Theorem 8.9); x_j = (a + b)/2 + (b - a)/2 · u_j. Returned ascending.
    """
    if n < 1:
        raise ValueError("need at least one Chebyshev node")
    j = np.arange(n - 1, -1, -1, dtype=np.float64)  # descending j gives ascending cos
    u = np.cos(np.pi * (2.0 * j + 1.0) / (2.0 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * u


def _dataset(
    id: str,
    name: str,
    x: Any,
    y: Any,
    f_true: Callable[[Any], Any] | None,
    latex: str,
    domain: tuple[float, float],
    description: str,
) -> Dataset:
    return Dataset(
        id=id,
        name=name,
        x=_frozen(x),
        y=_frozen(y),
        f_true=f_true,
        latex=latex,
        domain=domain,
        description=description,
    )


# --------------------------------------------------------------------------------------
# Interpolation datasets (noise-free samples of a known function)
# --------------------------------------------------------------------------------------


@factory("data")
def build_runge_equispaced() -> Dataset:
    x = np.linspace(-1.0, 1.0, 11)
    return _dataset(
        "runge_equispaced",
        "Runge function, 11 equispaced nodes",
        x,
        runge(x),
        runge,
        r"f(x) = \frac{1}{1 + 25x^2}",
        (-1.0, 1.0),
        "Runge's example: the degree-10 interpolant on equispaced nodes oscillates wildly "
        "near ±1 (max error ≈ 1.9), and the error grows with the degree.",
    )


@factory("data")
def build_runge_chebyshev() -> Dataset:
    x = chebyshev_nodes_first_kind(11)
    return _dataset(
        "runge_chebyshev",
        "Runge function, 11 Chebyshev nodes",
        x,
        runge(x),
        runge,
        r"f(x) = \frac{1}{1 + 25x^2},\ x_j = \cos\frac{(2j+1)\pi}{22}",
        (-1.0, 1.0),
        "The same function sampled at the roots of T₁₁: the node polynomial is minimal in "
        "the max norm, and the interpolation error converges as the degree grows.",
    )


@factory("data")
def build_sine_samples() -> Dataset:
    x = np.linspace(0.0, 2.0 * math.pi, 9)
    return _dataset(
        "sine_samples",
        "sin x, 9 equispaced samples",
        x,
        np.sin(x),
        np.sin,
        r"f(x) = \sin x",
        (0.0, 2.0 * math.pi),
        "A smooth, analytic function: every interpolant converges quickly.",
    )


def unit_step(x: Any) -> Any:
    """Heaviside step H(x) = 1 for x ≥ 0, else 0."""
    return np.where(np.asarray(x) >= 0.0, 1.0, 0.0)


@factory("data")
def build_step_data() -> Dataset:
    x = np.linspace(-1.0, 1.0, 12)  # an even count: no node at the jump x = 0
    return _dataset(
        "step_data",
        "Unit step (discontinuous)",
        x,
        unit_step(x),
        unit_step,
        r"H(x) = \begin{cases} 0 & x < 0 \\ 1 & x \ge 0 \end{cases}",
        (-1.0, 1.0),
        "Monotone data with a jump: polynomials and cubic splines overshoot (Gibbs-like), "
        "while PCHIP and the linear spline stay monotone. The jump has height 1, so the "
        "sup-norm error of any continuous interpolant is at least 1/2; the 200-point error "
        "grid has no point at x = 0, so the reported max errors of the splines are a "
        "little lower (about 0.46–0.48).",
    )


# --------------------------------------------------------------------------------------
# Regression datasets
# --------------------------------------------------------------------------------------


def _line(beta0: float, beta1: float) -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        return beta0 + beta1 * np.asarray(x, dtype=np.float64)

    return f


def quadratic_truth(x: Any) -> Any:
    """E[y | x] for noisy_quadratic: 1 - x + 0.5 x²."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 - x + 0.5 * x * x


def exponential_truth(x: Any) -> Any:
    """Median growth curve for exponential_growth: 2 exp(0.3 x)."""
    return 2.0 * np.exp(0.3 * np.asarray(x, dtype=np.float64))


NOISY_LINEAR_SEED = 42
NOISY_QUADRATIC_SEED = 7
OUTLIERS_LINEAR_SEED = 11
EXPONENTIAL_GROWTH_SEED = 3

#: Indices and additive offsets of the two gross outliers in ``outliers_linear``.
OUTLIER_OFFSETS: tuple[tuple[int, float], ...] = ((5, 15.0), (16, -20.0))


@factory("data")
def build_noisy_linear() -> Dataset:
    x = np.linspace(0.0, 10.0, 20)
    truth = _line(2.0, 0.5)
    y = truth(x) + _normal_noise(NOISY_LINEAR_SEED, x.size, 0.5)
    return _dataset(
        "noisy_linear",
        "Noisy line",
        x,
        y,
        truth,
        r"y = 2 + 0.5x + \varepsilon,\ \varepsilon \sim N(0, 0.5^2)",
        (0.0, 10.0),
        "Twenty points on a line with Gaussian noise (Mulberry32 seed 42, Box–Muller).",
    )


@factory("data")
def build_noisy_quadratic() -> Dataset:
    x = np.linspace(-2.0, 4.0, 25)
    y = quadratic_truth(x) + _normal_noise(NOISY_QUADRATIC_SEED, x.size, 0.4)
    return _dataset(
        "noisy_quadratic",
        "Noisy parabola",
        x,
        y,
        quadratic_truth,
        r"y = 1 - x + \tfrac12 x^2 + \varepsilon,\ \varepsilon \sim N(0, 0.4^2)",
        (-2.0, 4.0),
        "Twenty-five points on a parabola with Gaussian noise (seed 7): a line underfits, "
        "degree 2 fits, high degrees overfit.",
    )


#: Anscombe (1973), "Graphs in Statistical Analysis", Am. Stat. 27(1), Table 1, set I.
ANSCOMBE_I_X = (10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0)
ANSCOMBE_I_Y = (8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68)


@factory("data")
def build_anscombe_1() -> Dataset:
    return _dataset(
        "anscombe_1",
        "Anscombe's quartet I",
        ANSCOMBE_I_X,
        ANSCOMBE_I_Y,
        None,
        r"\hat y = 3.00 + 0.500\,x,\ R^2 = 0.67",
        (4.0, 14.0),
        "Anscombe (1973), set I, in the published order: the well-behaved member of the "
        "quartet. OLS gives ŷ = 3.0001 + 0.5001x with R² = 0.6665.",
    )


@factory("data")
def build_outliers_linear() -> Dataset:
    x = np.linspace(0.0, 10.0, 20)
    truth = _line(1.0, 2.0)
    y = truth(x) + _normal_noise(OUTLIERS_LINEAR_SEED, x.size, 0.3)
    for i, offset in OUTLIER_OFFSETS:
        y[i] += offset
    return _dataset(
        "outliers_linear",
        "Line with two gross outliers",
        x,
        y,
        truth,
        r"y = 1 + 2x + \varepsilon,\ \varepsilon \sim N(0, 0.3^2)",
        (0.0, 10.0),
        "A clean line (seed 11) with y₅ raised by 15 and y₁₆ lowered by 20: ordinary least "
        "squares is "
        "pulled away, while Huber, LAD and Theil–Sen stay within the noise of the "
        "least-squares line through the 18 clean points.",
    )


@factory("data")
def build_exponential_growth() -> Dataset:
    x = np.linspace(0.0, 10.0, 11)
    y = 2.0 * np.exp(0.3 * x + _normal_noise(EXPONENTIAL_GROWTH_SEED, x.size, 0.05))
    return _dataset(
        "exponential_growth",
        "Exponential growth",
        x,
        y,
        exponential_truth,
        r"y = 2\,e^{0.3x + \varepsilon},\ \varepsilon \sim N(0, 0.05^2)",
        (0.0, 10.0),
        "Multiplicative noise on an exponential (seed 3): a straight line fits badly; "
        "a line fitted to log y recovers log 2 and 0.3.",
    )
