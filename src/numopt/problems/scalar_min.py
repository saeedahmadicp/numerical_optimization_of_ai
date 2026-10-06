"""One-dimensional minimization problems (kind ``"scalar_min"``).

Every problem is a :class:`~numopt.core.types.Problem` with ``dim == 1`` whose ``f``,
``grad`` (= f') and ``hess`` (= f'') take and return Python floats. Each problem has

* ``bracket = (a, b)``: an interval that contains a local minimizer (the default input of
  the interval methods golden section, Fibonacci, dichotomous, ternary, parabolic
  interpolation and Brent);
* ``x0``: a default start for Newton's method and for minimum bracketing;
* ``minima``: every local minimizer inside ``domain`` that is not a domain end point, the
  global one first (``x`` values only; f there is ``f(x)``);
* ``domain``: the plotting window ``(lo, hi)``.

The tags say which assumptions of the interval methods hold on ``bracket``:
``unimodal`` (exactly one local minimizer in the bracket, so interval elimination is
exact), ``multimodal``, ``nonsmooth``, ``flat``.

Outside its mathematical domain a function returns ``nan`` (it never raises), so methods
can report the breakdown honestly.
"""

from __future__ import annotations

import math

from ..core.types import Problem
from .registry import factory

# --------------------------------------------------------------------------------------
# quadratic_1d
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def quadratic_1d() -> Problem:
    """f(x) = (x - 2)² + 1: the model problem; Newton converges in one step."""

    def f(x: float) -> float:
        return (x - 2.0) ** 2 + 1.0

    def grad(x: float) -> float:
        return 2.0 * (x - 2.0)

    def hess(x: float) -> float:
        return 2.0

    return Problem(
        id="quadratic_1d",
        name="Quadratic",
        latex=r"f(x) = (x - 2)^2 + 1",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(-1.0, 5.0),
        bracket=(0.0, 5.0),
        x0=0.5,
        minima=(2.0,),
        description="A convex parabola with minimizer x⋆ = 2 and f⋆ = 1. Parabolic "
        "interpolation and Newton's method are exact on it.",
        tags=("smooth", "unimodal", "convex"),
    )


# --------------------------------------------------------------------------------------
# quartic_1d
# --------------------------------------------------------------------------------------

# Critical points of f(x) = x⁴ - 4x² + x solve f'(x) = 4x³ - 8x + 1 = 0, i.e. the depressed
# cubic x³ + p x + q = 0 with p = -2, q = 1/4. All three roots are real, and Viète's
# trigonometric formula gives them in closed form:
#   x_k = 2√(-p/3) cos( (1/3) arccos( (3q / 2p) √(-3/p) ) - 2πk/3 ),  k = 0, 1, 2.
_QP, _QQ = -2.0, 0.25
_QUARTIC_CRIT = tuple(
    2.0
    * math.sqrt(-_QP / 3.0)
    * math.cos(
        math.acos(3.0 * _QQ / (2.0 * _QP) * math.sqrt(-3.0 / _QP)) / 3.0 - 2.0 * math.pi * k / 3.0
    )
    for k in range(3)
)
# k = 0: local minimizer ≈ 1.34700, k = 1: local maximizer ≈ 0.12600,
# k = 2: global minimizer ≈ -1.47300.
QUARTIC_GLOBAL_MIN = _QUARTIC_CRIT[2]
QUARTIC_LOCAL_MIN = _QUARTIC_CRIT[0]
QUARTIC_LOCAL_MAX = _QUARTIC_CRIT[1]


@factory("scalar_min")
def quartic_1d() -> Problem:
    """f(x) = x⁴ - 4x² + x: a tilted double well with two local minima."""

    def f(x: float) -> float:
        return x**4 - 4.0 * x**2 + x

    def grad(x: float) -> float:
        return 4.0 * x**3 - 8.0 * x + 1.0

    def hess(x: float) -> float:
        return 12.0 * x**2 - 8.0

    return Problem(
        id="quartic_1d",
        name="Tilted double well",
        latex=r"f(x) = x^4 - 4x^2 + x",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(-2.2, 2.2),
        # The default bracket holds only the *local* minimizer ≈ 1.347: interval methods
        # find the minimizer in the bracket, not the global one at ≈ -1.473.
        bracket=(0.5, 2.0),
        x0=1.0,
        minima=(QUARTIC_GLOBAL_MIN, QUARTIC_LOCAL_MIN),
        description="Two wells: the global minimizer x ≈ −1.4730 (f ≈ −5.4442) and a local "
        "minimizer x ≈ 1.3470 (f ≈ −2.6186), separated by a local maximum at x ≈ 0.1260. "
        "f″ < 0 on |x| < √(2/3), where Newton's step points uphill.",
        tags=("smooth", "multimodal"),
    )


# --------------------------------------------------------------------------------------
# sin_1d
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def sin_1d() -> Problem:
    """f(x) = sin x on [0, 2π]: minimizer 3π/2."""

    def f(x: float) -> float:
        return math.sin(x)

    def grad(x: float) -> float:
        return math.cos(x)

    def hess(x: float) -> float:
        return -math.sin(x)

    return Problem(
        id="sin_1d",
        name="Sine",
        latex=r"f(x) = \sin x",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(0.0, 2.0 * math.pi),
        bracket=(0.0, 2.0 * math.pi),
        x0=4.0,
        minima=(1.5 * math.pi,),
        description="One period of the sine. The interior minimizer is x⋆ = 3π/2 with "
        "f⋆ = −1; f″ = −sin x < 0 on (0, π), where Newton needs its safeguard.",
        tags=("smooth",),
    )


# --------------------------------------------------------------------------------------
# x_log_x
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def x_log_x() -> Problem:
    """f(x) = x ln x on x ≥ 0 (f(0) = 0 by continuity): minimizer 1/e."""

    def f(x: float) -> float:
        if x > 0.0:
            return x * math.log(x)
        if x == 0.0:
            return 0.0  # the continuous extension: lim_{x→0⁺} x ln x = 0
        return math.nan  # outside the domain

    def grad(x: float) -> float:
        if x > 0.0:
            return math.log(x) + 1.0
        if x == 0.0:
            return -math.inf
        return math.nan

    def hess(x: float) -> float:
        if x > 0.0:
            return 1.0 / x
        if x == 0.0:
            return math.inf
        return math.nan

    return Problem(
        id="x_log_x",
        name="x log x",
        latex=r"f(x) = x \ln x",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(0.0, 2.0),
        bracket=(0.0, 2.0),
        x0=0.5,
        minima=(math.exp(-1.0),),
        description="The entropy-type function x ln x, convex on x > 0 with minimizer "
        "x⋆ = 1/e and f⋆ = −1/e. f′ → −∞ at 0; f is undefined (nan) for x < 0, so a long "
        "Newton step from x₀ ≥ 1 leaves the domain.",
        tags=("smooth", "unimodal", "convex", "domain"),
    )


# --------------------------------------------------------------------------------------
# abs_shifted
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def abs_shifted() -> Problem:
    """f(x) = |x - 0.3| + x²/2: convex, non-differentiable at its minimizer x* = 0.3."""
    kink = 0.3

    def f(x: float) -> float:
        return abs(x - kink) + 0.5 * x * x

    def grad(x: float) -> float:
        # The derivative where it exists. At the kink we return the minimum-norm element
        # of the subdifferential [-1, 1] + 0.3, which is 0 (so x* is stationary).
        if x == kink:
            return 0.0
        return math.copysign(1.0, x - kink) + x

    def hess(x: float) -> float:
        # f'' = 1 wherever it exists; the kink is a point mass that no pointwise value shows.
        return 1.0

    return Problem(
        id="abs_shifted",
        name="Shifted absolute value",
        latex=r"f(x) = |x - 0.3| + \tfrac{1}{2}x^2",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(-1.0, 1.0),
        bracket=(-1.0, 1.0),
        x0=-0.5,
        minima=(kink,),
        description="Convex but not differentiable at the minimizer x⋆ = 0.3 (f⋆ = 0.045), "
        "because 0 ∈ ∂f(0.3) = [−0.7, 1.3]. Interval elimination still works; Newton's "
        "method oscillates between −1 and 1 because f′ jumps by 2 at the kink.",
        tags=("nonsmooth", "unimodal", "convex"),
    )


# --------------------------------------------------------------------------------------
# multimodal_1d
# --------------------------------------------------------------------------------------

# Local minimizers of sin x + sin(10x/3) on [2.7, 7.5]: roots of f' = cos x + (10/3) cos(10x/3)
# with f'' > 0, computed with Brent's root finder to |f'| ≤ 1e-15 (global minimizer first).
_MULTIMODAL_MINIMA = (5.145735290256129, 3.387251718444631, 7.0001491168622545)


@factory("scalar_min")
def multimodal_1d() -> Problem:
    """f(x) = sin x + sin(10x/3) on [2.7, 7.5] (a classic global-optimization test)."""
    w = 10.0 / 3.0

    def f(x: float) -> float:
        return math.sin(x) + math.sin(w * x)

    def grad(x: float) -> float:
        return math.cos(x) + w * math.cos(w * x)

    def hess(x: float) -> float:
        return -math.sin(x) - w * w * math.sin(w * x)

    return Problem(
        id="multimodal_1d",
        name="Sum of sines",
        latex=r"f(x) = \sin x + \sin\tfrac{10x}{3}",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(2.7, 7.5),
        bracket=(2.7, 7.5),
        x0=5.0,
        minima=_MULTIMODAL_MINIMA,
        description="Three local minima on [2.7, 7.5]; the global one is x ≈ 5.14574 with "
        "f ≈ −1.89960. f decreases at 2.7 and increases at 7.5, so every interval method "
        "ends at an interior local minimizer, but not necessarily the global one.",
        tags=("smooth", "multimodal"),
    )


# --------------------------------------------------------------------------------------
# flat_valley
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def flat_valley() -> Problem:
    """f(x) = exp(-1/x²) (f(0) = 0): every derivative vanishes at the minimizer."""

    def neg_inv_sq(x: float) -> float:
        # -1/x², computed as -(1/x)² so that a tiny x gives -inf instead of 1/0.
        s = 1.0 / x
        return -(s * s)

    def f(x: float) -> float:
        if x == 0.0:
            return 0.0
        return math.exp(neg_inv_sq(x))

    # The derivatives are written in log form, exp(-1/x² - m ln|x|), so that the tiny
    # exponential factor and the huge power 1/|x|^m never multiply to inf·0 = nan.
    def grad(x: float) -> float:
        if x == 0.0:
            return 0.0
        return math.copysign(2.0 * math.exp(neg_inv_sq(x) - 3.0 * math.log(abs(x))), x)

    def hess(x: float) -> float:
        if x == 0.0:
            return 0.0
        return (4.0 - 6.0 * x * x) * math.exp(neg_inv_sq(x) - 6.0 * math.log(abs(x)))

    return Problem(
        id="flat_valley",
        name="Flat valley",
        latex=r"f(x) = e^{-1/x^2}",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(-1.0, 2.0),
        bracket=(-1.0, 2.0),
        x0=0.5,
        minima=(0.0,),
        description="The minimizer x⋆ = 0 is infinitely flat: f and all its derivatives "
        "vanish there. In double precision f(x) underflows to exactly 0 for |x| < 0.0367, "
        "so no f-comparison can locate x⋆ better than that, and f′(x) < 10⁻⁸ already for "
        "|x| < 0.2. Newton's step x − 2x³/(4 − 6x²) converges sublinearly.",
        tags=("smooth", "unimodal", "flat"),
    )


# --------------------------------------------------------------------------------------
# rational_1d
# --------------------------------------------------------------------------------------


@factory("scalar_min")
def rational_1d() -> Problem:
    """f(x) = (x² - 2x + 3)/(x + 1) on x > -1: minimizer √6 - 1."""

    def f(x: float) -> float:
        d = x + 1.0
        if d == 0.0:
            return math.nan  # the pole
        return (x * x - 2.0 * x + 3.0) / d

    # With u = x + 1: f = u + 6/u - 4, f' = 1 - 6/u², f'' = 12/u³.
    def grad(x: float) -> float:
        d = x + 1.0
        if d == 0.0:
            return math.nan
        return (x * x + 2.0 * x - 5.0) / (d * d)

    def hess(x: float) -> float:
        d = x + 1.0
        if d == 0.0:
            return math.nan
        return 12.0 / (d * d * d)

    return Problem(
        id="rational_1d",
        name="Rational model",
        latex=r"f(x) = \frac{x^2 - 2x + 3}{x + 1}",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(0.0, 6.0),
        bracket=(0.0, 6.0),
        x0=0.5,
        minima=(math.sqrt(6.0) - 1.0,),
        description="A rational function of the kind a Padé or rational least-squares fit "
        "produces. On x > −1 it equals (x + 1) + 6/(x + 1) − 4, which is strictly convex with "
        "minimizer x⋆ = √6 − 1 ≈ 1.4495 and f⋆ = 2√6 − 4 ≈ 0.8990. It is asymmetric: "
        "steep on the left, nearly linear on the right.",
        tags=("smooth", "unimodal", "convex"),
    )


# --------------------------------------------------------------------------------------
# drug_concentration
# --------------------------------------------------------------------------------------

# One-compartment model with first-order absorption (the Bateman function):
#   C(t) = D (e^{-k_e t} - e^{-k_a t}),
# D [mg/L] lumps bioavailability · dose · k_a / (V (k_a - k_e)).
DRUG_D = 10.0  # mg/L
DRUG_KA = 1.0  # absorption rate constant, 1/h
DRUG_KE = 0.2  # elimination rate constant, 1/h
# Peak time: C'(t) = 0  ⇔  k_e e^{-k_e t} = k_a e^{-k_a t}  ⇔  t = ln(k_a/k_e) / (k_a - k_e).
DRUG_T_MAX = math.log(DRUG_KA / DRUG_KE) / (DRUG_KA - DRUG_KE)


@factory("scalar_min")
def drug_concentration() -> Problem:
    """Time of peak plasma concentration: minimize f(t) = -C(t)."""
    D, ka, ke = DRUG_D, DRUG_KA, DRUG_KE

    def f(t: float) -> float:
        return -D * (math.exp(-ke * t) - math.exp(-ka * t))

    def grad(t: float) -> float:
        return -D * (ka * math.exp(-ka * t) - ke * math.exp(-ke * t))

    def hess(t: float) -> float:
        return -D * (ke * ke * math.exp(-ke * t) - ka * ka * math.exp(-ka * t))

    return Problem(
        id="drug_concentration",
        name="Peak drug concentration",
        latex=r"f(t) = -D\,(e^{-k_e t} - e^{-k_a t}),\ D = 10,\ k_a = 1,\ k_e = 0.2",
        f=f,
        grad=grad,
        hess=hess,
        dim=1,
        domain=(0.0, 24.0),
        bracket=(0.0, 12.0),
        x0=1.0,
        minima=(DRUG_T_MAX,),
        description="Plasma concentration after an oral dose (one-compartment model with "
        "first-order absorption). The peak is at t⋆ = ln(kₐ/kₑ)/(kₐ − kₑ) = ln 5/0.8 "
        "≈ 2.0118 h with C ≈ 5.3499 mg/L. −C is convex only for t < 2t⋆ ≈ 4.02 h; beyond "
        "that Newton's step points the wrong way. (A nod to the legacy "
        "'drug_effectiveness' example.)",
        tags=("smooth", "unimodal", "application"),
    )


__all__ = [
    "DRUG_D",
    "DRUG_KA",
    "DRUG_KE",
    "DRUG_T_MAX",
    "QUARTIC_GLOBAL_MIN",
    "QUARTIC_LOCAL_MAX",
    "QUARTIC_LOCAL_MIN",
]
