"""Calculus test problems: integrands for quadrature and functions to differentiate.

Every problem is a 1-D :class:`~numopt.core.types.Problem` with

* ``f``      the function, written with NumPy ufuncs so that it accepts floats, NumPy arrays
  **and complex numbers** (the complex-step derivative evaluates ``f(x + i h)``);
* ``grad``   the exact first derivative f′ and ``hess`` the exact second derivative f″;
* ``domain`` the integration interval ``(a, b)``;
* ``exact``  the exact value of ∫ₐᵇ f(x) dx (closed form, see each docstring);
* ``x0``     the default point at which the differentiation methods estimate f′(x0).

The set is chosen to show where the error theory of quadrature and finite differences holds
and where it fails: smooth analytic integrands (``poly3``, ``exp_0_1``, ``sin_0_pi``,
``gaussian``, ``arctan_deriv``), a function with poles near the interval (``runge``), a
singular derivative at an endpoint (``sqrt_0_1``), a fast oscillation (``oscillatory``) and a
kink (``abs_kink``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.types import Problem
from .registry import factory


def _where_real_nonneg(d: Any) -> Any:
    """``|d|`` continued analytically from each side: ``d`` where Re d ≥ 0, else ``−d``.

    ``np.abs`` would return the complex modulus for complex input, which is not analytic and
    breaks the complex-step derivative. This piecewise form is analytic on each side of the
    kink, so ``Im f(x + i h)/h`` returns the one-sided slope ±1 away from the kink.
    """
    return np.where(np.real(d) >= 0, d, -d)[()]


@factory("calculus")
def poly3() -> Problem:
    """f(x) = x³ − 2x + 1 on [0, 2]; ∫ = [x⁴/4 − x² + x]₀² = 4 − 4 + 2 = 2.

    f‴ ≡ 6 and f⁽⁴⁾ ≡ 0: Simpson, Simpson 3/8, Boole and Gauss with n ≥ 2 points are exact,
    and the central-difference error is exactly h² (= h² f‴/6).
    """
    return Problem(
        id="poly3",
        name="Cubic polynomial",
        latex=r"f(x) = x^3 - 2x + 1",
        f=lambda x: x**3 - 2.0 * x + 1.0,
        grad=lambda x: 3.0 * x**2 - 2.0,
        hess=lambda x: 6.0 * x,
        dim=1,
        domain=(0.0, 2.0),
        x0=1.0,
        exact=2.0,
        description="A cubic: rules exact for degree 3 (Simpson, Boole, Gauss n ≥ 2) give it exactly.",
        tags=("smooth", "polynomial"),
    )


@factory("calculus")
def exp_0_1() -> Problem:
    """f(x) = eˣ on [0, 1]; ∫ = e − 1 (computed as ``expm1(1)``)."""
    return Problem(
        id="exp_0_1",
        name="Exponential",
        latex=r"f(x) = e^{x}",
        f=lambda x: np.exp(x),
        grad=lambda x: np.exp(x),
        hess=lambda x: np.exp(x),
        dim=1,
        domain=(0.0, 1.0),
        x0=1.0,
        exact=math.expm1(1.0),
        description="Smooth and positive; every rule shows its textbook convergence order.",
        tags=("smooth", "analytic"),
    )


@factory("calculus")
def sin_0_pi() -> Problem:
    """f(x) = sin x on [0, π]; ∫ = [−cos x]₀^π = 2."""
    return Problem(
        id="sin_0_pi",
        name="Sine on [0, π]",
        latex=r"f(x) = \sin x",
        f=lambda x: np.sin(x),
        grad=lambda x: np.cos(x),
        hess=lambda x: -np.sin(x),
        dim=1,
        domain=(0.0, math.pi),
        x0=math.pi / 3.0,
        exact=2.0,
        description="One arch of the sine; f′(π/3) = 1/2.",
        tags=("smooth", "analytic"),
    )


@factory("calculus")
def runge() -> Problem:
    """Runge's function f(x) = 1/(1 + 25x²) on [−1, 1]; ∫ = (2/5)·arctan 5.

    f′ = −50x/u², f″ = (3750x² − 50)/u³ with u = 1 + 25x². The poles at x = ±i/5 lie close to
    the interval, so polynomial-based rules converge slowly at first.
    """

    def f(x: Any) -> Any:
        return 1.0 / (1.0 + 25.0 * x**2)

    def grad(x: Any) -> Any:
        u = 1.0 + 25.0 * x**2
        return -50.0 * x / u**2

    def hess(x: Any) -> Any:
        u = 1.0 + 25.0 * x**2
        return (3750.0 * x**2 - 50.0) / u**3

    return Problem(
        id="runge",
        name="Runge function",
        latex=r"f(x) = \frac{1}{1 + 25x^2}",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(-1.0, 1.0),
        x0=0.2,
        exact=0.4 * math.atan(5.0),
        description="Complex poles at ±i/5 near the interval slow down polynomial-based rules.",
        tags=("smooth", "analytic", "runge"),
    )


@factory("calculus")
def sqrt_0_1() -> Problem:
    """f(x) = √x on [0, 1]; ∫ = 2/3.

    f′ = 1/(2√x) and f″ = −1/(4 x^{3/2}) are unbounded at x = 0, so the error expansions behind
    the Newton–Cotes orders, Romberg and Gauss do not hold. The panel at 0 has an error of
    order h^{3/2}, so every rule of order ≥ 2 (midpoint, trapezoid, Simpson, Simpson 3/8,
    Boole) and Romberg converge only like h^{3/2}; the left and right Riemann sums keep their
    O(h), which is slower; Gauss–Legendre converges algebraically, like n^{−3} in the number
    of points (observed orders: 1.00, 1.50 and 3.0).
    For x < 0 the real function is undefined (NaN), which a central stencil can hit when
    h > x0.
    """
    return Problem(
        id="sqrt_0_1",
        name="Square root",
        latex=r"f(x) = \sqrt{x}",
        f=lambda x: np.sqrt(x),
        grad=lambda x: 0.5 / np.sqrt(x),
        hess=lambda x: -0.25 / (x * np.sqrt(x)),
        dim=1,
        domain=(0.0, 1.0),
        x0=0.25,
        exact=2.0 / 3.0,
        description=(
            "Singular derivative at x = 0: rules of order ≥ 2 drop to h^{1.5} (Riemann sums keep h)."
        ),
        tags=("singular-derivative",),
        extra={"singular_point": 0.0},
    )


@factory("calculus")
def gaussian() -> Problem:
    """f(x) = e^{−x²} on [−2, 2]; ∫ = √π·erf(2)."""
    return Problem(
        id="gaussian",
        name="Gaussian",
        latex=r"f(x) = e^{-x^2}",
        f=lambda x: np.exp(-(x**2)),
        grad=lambda x: -2.0 * x * np.exp(-(x**2)),
        hess=lambda x: (4.0 * x**2 - 2.0) * np.exp(-(x**2)),
        dim=1,
        domain=(-2.0, 2.0),
        x0=0.5,
        exact=math.sqrt(math.pi) * math.erf(2.0),
        description="The bell curve; the exact integral needs the error function erf.",
        tags=("smooth", "analytic"),
    )


@factory("calculus")
def oscillatory() -> Problem:
    """f(x) = sin(10x) on [0, π]; ∫ = (1 − cos 10π)/10 = 0.

    The interval holds five full periods. f is odd about the midpoint π/2
    (f(π − x) = −f(x)), so every rule whose nodes and weights are symmetric about π/2
    (trapezoid, midpoint, Simpson, Boole, Gauss, Romberg, and here also the Riemann sums,
    because f(0) = f(π) = 0) returns 0 up to rounding at every resolution. Monte Carlo and
    adaptive rules show the effect of the oscillation; f′ = 10 cos 10x has a large scale.
    """
    return Problem(
        id="oscillatory",
        name="Oscillatory sine",
        latex=r"f(x) = \sin(10x)",
        f=lambda x: np.sin(10.0 * x),
        grad=lambda x: 10.0 * np.cos(10.0 * x),
        hess=lambda x: -100.0 * np.sin(10.0 * x),
        dim=1,
        domain=(0.0, math.pi),
        x0=0.5,
        exact=0.0,
        description="Five periods on [0, π]; the exact integral is 0 by symmetry.",
        tags=("smooth", "oscillatory"),
    )


@factory("calculus")
def arctan_deriv() -> Problem:
    """f(x) = 1/(1 + x²) = (arctan x)′ on [0, 1]; ∫ = arctan 1 = π/4.

    f′ = −2x/(1 + x²)², f″ = (6x² − 2)/(1 + x²)³.
    """

    def f(x: Any) -> Any:
        return 1.0 / (1.0 + x**2)

    def grad(x: Any) -> Any:
        return -2.0 * x / (1.0 + x**2) ** 2

    def hess(x: Any) -> Any:
        return (6.0 * x**2 - 2.0) / (1.0 + x**2) ** 3

    return Problem(
        id="arctan_deriv",
        name="Derivative of arctan",
        latex=r"f(x) = \frac{1}{1 + x^2}",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(0.0, 1.0),
        x0=0.5,
        exact=math.pi / 4.0,
        description="Integrates to π/4: the classic way to compute π by quadrature.",
        tags=("smooth", "analytic"),
    )


@factory("calculus")
def abs_kink() -> Problem:
    """f(x) = |x − 0.3| on [0, 1]; ∫ = (0.3² + 0.7²)/2 = 0.29.

    f is not differentiable at the kink x = 0.3. ``grad`` returns sign(x − 0.3) (0 at the
    kink, the midpoint of the subdifferential [−1, 1]) and ``hess`` returns 0 (f″ is a Dirac
    delta at the kink). The kink is not a node of any dyadic grid on [0, 1]: the panel that
    contains it has error O(h²), so Simpson, Boole and Romberg drop to O(h²) (the trapezoid
    and midpoint rules keep O(h²); Gauss with n points errs like O(n⁻²)). The default
    ``x0 = 0.35`` lies 0.05 right of the kink: difference stencils with h > 0.05 that straddle
    the kink return wrong slopes, smaller h return the exact slope 1.
    """
    return Problem(
        id="abs_kink",
        name="Absolute value with a kink",
        latex=r"f(x) = |x - 0.3|",
        f=lambda x: _where_real_nonneg(x - 0.3),
        grad=lambda x: np.sign(x - 0.3),
        hess=lambda x: 0.0 * x,
        dim=1,
        domain=(0.0, 1.0),
        x0=0.35,
        exact=0.29,
        description="A kink at x = 0.3: quadrature loses its order; stencils that straddle it fail.",
        tags=("nonsmooth",),
        extra={"kink": 0.3},
    )
