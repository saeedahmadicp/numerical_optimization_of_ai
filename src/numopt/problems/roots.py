"""Scalar root-finding problems f(x) = 0 (kind ``"roots"``).

Every problem is a 1-D :class:`~numopt.core.types.Problem` with

* ``f``    — the function, ``float -> float``;
* ``grad`` — the exact first derivative f'(x);
* ``hess`` — the exact second derivative f''(x) (Halley's method uses it);
* ``bracket`` — an interval ``(a, b)`` with ``f(a)·f(b) < 0`` for the bracketing methods;
* ``x0`` — a starting point for the open methods (Newton, secant, ...);
* ``roots`` — every real root of f (in increasing order), to full double precision;
* ``domain`` — the plotting window ``(x_min, x_max)``.

Several problems exist to show a failure mode of a method; their ``description`` and
``tags`` say which. Functions are written in their factored or otherwise well-conditioned
form, so f(x) has a small relative error near each root.

Every function follows IEEE semantics far from its domain: overflow gives ±inf and a
non-finite argument gives inf or nan, never an exception. (``math.exp`` and the float
``**`` operator raise ``OverflowError`` and ``math.sin``/``math.cos`` raise ``ValueError``
at ±inf; :func:`_exp`, :func:`_pow`, :func:`_sin` and :func:`_cos` wrap them, and squares
are written as products.) Open methods evaluate f far outside the plotting domain.

The constant roots were computed with ``scipy.optimize.brentq`` at the tightest
tolerance (or in closed form) and are verified in ``tests/test_problems_roots.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..core.types import Problem
from .registry import add

ScalarFn = Callable[[float], float]


def _exp(x: float) -> float:
    """eˣ, or +inf when it overflows (``math.exp`` raises OverflowError for x > 709.78)."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _sin(x: float) -> float:
    """sin x, or nan for a non-finite x (``math.sin`` raises ValueError there)."""
    return math.sin(x) if math.isfinite(x) else math.nan


def _cos(x: float) -> float:
    """cos x, or nan for a non-finite x (``math.cos`` raises ValueError there)."""
    return math.cos(x) if math.isfinite(x) else math.nan


def _pow(x: float, n: int) -> float:
    """xⁿ for an integer n ≥ 0, or ±inf when it overflows (float ``**`` raises instead)."""
    try:
        return x**n
    except OverflowError:
        return -math.inf if x < 0.0 and n % 2 == 1 else math.inf


def _scalar(
    id: str,
    name: str,
    latex: str,
    f: ScalarFn,
    df: ScalarFn,
    d2f: ScalarFn,
    *,
    bracket: tuple[float, float],
    x0: float,
    roots: tuple[float, ...],
    domain: tuple[float, float],
    description: str,
    tags: tuple[str, ...] = (),
    extra: dict[str, float] | None = None,
) -> Problem:
    return add(
        "roots",
        Problem(
            id=id,
            name=name,
            latex=latex,
            f=f,
            grad=df,
            hess=d2f,
            dim=1,
            domain=domain,
            x0=x0,
            bracket=bracket,
            roots=roots,
            description=description,
            tags=tags,
            extra=extra or {},
        ),
    )


# --- x² − 2 ---------------------------------------------------------------------------

SQRT2 = _scalar(
    "sqrt2",
    "Square root of 2",
    r"f(x) = x^2 - 2",
    lambda x: x * x - 2.0,
    lambda x: 2.0 * x,
    lambda x: 2.0,
    bracket=(1.0, 2.0),
    x0=1.0,
    roots=(-math.sqrt(2.0), math.sqrt(2.0)),
    domain=(-2.0, 2.5),
    description="The classic first example: a convex parabola with a simple root at √2.",
    tags=("polynomial", "simple-root", "convex"),
)

# --- Wallis' cubic x³ − 2x − 5 ----------------------------------------------------------

CUBIC = _scalar(
    "cubic",
    "Wallis' cubic",
    r"f(x) = x^3 - 2x - 5",
    lambda x: (x * x - 2.0) * x - 5.0,
    lambda x: 3.0 * x * x - 2.0,
    lambda x: 6.0 * x,
    bracket=(2.0, 3.0),
    x0=2.0,
    roots=(2.0945514815423265,),
    domain=(0.0, 3.5),
    description=(
        "The equation Wallis used (1685) to present Newton's method; one real root near 2.0946."
    ),
    tags=("polynomial", "simple-root", "historical"),
)

# --- cos x − x ---------------------------------------------------------------------------

COS_MINUS_X = _scalar(
    "cos_minus_x",
    "cos x = x",
    r"f(x) = \cos x - x",
    lambda x: _cos(x) - x,
    lambda x: -_sin(x) - 1.0,
    lambda x: -_cos(x),
    bracket=(0.0, 1.0),
    x0=1.0,
    roots=(0.7390851332151607,),
    domain=(-1.0, 2.0),
    description=(
        "The fixed point of cos (the Dottie number). f is decreasing, so the fixed-point "
        "iteration x ← x − λ f(x) needs λ < 0."
    ),
    tags=("transcendental", "simple-root", "fixed-point"),
)

# --- x¹⁰ − 1 -----------------------------------------------------------------------------

X10_MINUS_1 = _scalar(
    "x10_minus_1",
    "x¹⁰ − 1",
    r"f(x) = x^{10} - 1",
    lambda x: _pow(x, 10) - 1.0,
    lambda x: 10.0 * _pow(x, 9),
    lambda x: 90.0 * _pow(x, 8),
    bracket=(0.0, 1.3),
    x0=1.3,
    roots=(-1.0, 1.0),
    domain=(0.0, 1.4),
    description=(
        "Flat on the left and steep on the right of x = 1, so plain regula falsi keeps the "
        "right end fixed and converges slowly; Illinois-type methods fix this. Newton from "
        "x₀ = 0.5 overshoots to x ≈ 51.6."
    ),
    tags=("polynomial", "simple-root", "regula-falsi-hard"),
)

# --- Kepler's equation -------------------------------------------------------------------

KEPLER_E = 0.9
KEPLER_M = 0.3

KEPLER = _scalar(
    "kepler",
    "Kepler's equation (e = 0.9)",
    r"f(E) = E - e\sin E - M,\quad e = 0.9,\ M = 0.3",
    lambda x: x - KEPLER_E * _sin(x) - KEPLER_M,
    lambda x: 1.0 - KEPLER_E * _cos(x),
    lambda x: KEPLER_E * _sin(x),
    bracket=(0.0, math.pi),
    x0=KEPLER_M,
    roots=(1.103517720303087,),
    domain=(0.0, math.pi),
    description=(
        "Eccentric anomaly E of an orbit with eccentricity e = 0.9 at mean anomaly M = 0.3. "
        "With the naive start E₀ = M, f′(E₀) = 1 − e cos M ≈ 0.14 is small and Newton's "
        "first step overshoots to E ≈ 2.2."
    ),
    tags=("transcendental", "simple-root", "astronomy"),
    extra={"e": KEPLER_E, "M": KEPLER_M},
)

# --- (x − 1)²(x + 2) ---------------------------------------------------------------------

DOUBLE_ROOT = _scalar(
    "double_root",
    "Double root",
    r"f(x) = (x-1)^2 (x+2)",
    lambda x: (x - 1.0) * (x - 1.0) * (x + 2.0),
    lambda x: 3.0 * (x - 1.0) * (x + 1.0),
    lambda x: 6.0 * x,
    bracket=(-3.0, 0.0),
    x0=2.0,
    roots=(-2.0, 1.0),
    domain=(-3.0, 2.5),
    description=(
        "f touches zero at the double root x = 1 without a sign change, so no bracket can "
        "isolate it; Newton converges to it only linearly (rate ½). The default bracket "
        "isolates the simple root x = −2."
    ),
    tags=("polynomial", "multiple-root"),
    extra={"multiplicity_at_1": 2.0},
)

# --- arctan x ----------------------------------------------------------------------------

#: Newton on arctan diverges for |x₀| > x_c, where x_c solves (1 + x²)·arctan x = 2x
#: (then the Newton map sends x_c to −x_c: a 2-cycle that separates the two regimes).
ATAN_NEWTON_CRITICAL = 1.3917452002707353

ATAN_NEWTON = _scalar(
    "atan_newton",
    "arctan x (Newton diverges)",
    r"f(x) = \arctan x",
    math.atan,
    lambda x: 1.0 / (1.0 + x * x),
    lambda x: -2.0 * x / ((1.0 + x * x) * (1.0 + x * x)),
    bracket=(-1.5, 2.0),
    x0=1.5,
    roots=(0.0,),
    domain=(-6.0, 6.0),
    description=(
        "Newton's method converges to 0 only for |x₀| < 1.3917452; from x₀ = 1.5 the "
        "iterates alternate in sign and grow without bound."
    ),
    tags=("transcendental", "simple-root", "newton-diverges"),
    extra={"newton_critical_x0": ATAN_NEWTON_CRITICAL},
)

# --- x³ − 2x + 2 -------------------------------------------------------------------------

NEWTON_CYCLE = _scalar(
    "newton_cycle",
    "Newton 2-cycle",
    r"f(x) = x^3 - 2x + 2",
    lambda x: (x * x - 2.0) * x + 2.0,
    lambda x: 3.0 * x * x - 2.0,
    lambda x: 6.0 * x,
    bracket=(-2.0, -1.0),
    x0=0.0,
    roots=(-1.7692923542386316,),
    domain=(-2.5, 2.0),
    description=(
        "Newton from x₀ = 0 cycles 0 → 1 → 0 → … forever and never reaches the only real "
        "root x ≈ −1.769 (the cycle is superattracting because f″(0) = 0)."
    ),
    tags=("polynomial", "simple-root", "newton-cycles"),
)

# --- eˣ − 10 -----------------------------------------------------------------------------

STEEP_EXP = _scalar(
    "steep_exp",
    "Steep exponential",
    r"f(x) = e^x - 10",
    lambda x: _exp(x) - 10.0,
    _exp,
    _exp,
    bracket=(0.0, 6.0),
    x0=4.0,
    roots=(math.log(10.0),),
    domain=(0.0, 6.0),
    description=(
        "Very steep on the right of the bracket (f(6) ≈ 393), so regula falsi keeps the "
        "right end and crawls; Newton from the left (x₀ < 0) jumps far to the right first."
    ),
    tags=("transcendental", "simple-root", "steep"),
)

# --- Wilkinson-type polynomial ∏(x − i), i = 1..5 ------------------------------------------

_W_ROOTS = (1.0, 2.0, 3.0, 4.0, 5.0)


def _w_f(x: float) -> float:
    p = 1.0
    for r in _W_ROOTS:
        p *= x - r
    return p


def _w_df(x: float) -> float:
    # f'(x) = Σ_i ∏_{j≠i} (x − j)   (product rule; accurate near every root)
    total = 0.0
    for i in range(len(_W_ROOTS)):
        p = 1.0
        for j, r in enumerate(_W_ROOTS):
            if j != i:
                p *= x - r
        total += p
    return total


def _w_d2f(x: float) -> float:
    # f''(x) = 2 Σ_{i<j} ∏_{l∉{i,j}} (x − l)
    total = 0.0
    n = len(_W_ROOTS)
    for i in range(n):
        for j in range(i + 1, n):
            p = 1.0
            for m, r in enumerate(_W_ROOTS):
                if m != i and m != j:
                    p *= x - r
            total += p
    return 2.0 * total


WILKINSON5 = _scalar(
    "wilkinson5",
    "Wilkinson polynomial (degree 5)",
    r"f(x) = \prod_{i=1}^{5} (x - i)",
    _w_f,
    _w_df,
    _w_d2f,
    bracket=(0.6, 5.3),
    x0=5.4,
    roots=_W_ROOTS,
    domain=(0.5, 5.5),
    description=(
        "Five simple roots 1, …, 5; the default bracket holds all of them, so each bracketing "
        "method shows which root its own steps lead to. Evaluated in product form."
    ),
    tags=("polynomial", "multiple-roots-in-bracket"),
)

__all__ = [
    "ATAN_NEWTON",
    "ATAN_NEWTON_CRITICAL",
    "COS_MINUS_X",
    "CUBIC",
    "DOUBLE_ROOT",
    "KEPLER",
    "NEWTON_CYCLE",
    "SQRT2",
    "STEEP_EXP",
    "WILKINSON5",
    "X10_MINUS_1",
]
