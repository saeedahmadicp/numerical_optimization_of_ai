# algorithms/__init__.py

from .convex import (
    bisection,
    newton,
    regula_falsi,
    secant,
    fibonacci_search,
    golden_search,
    elimination_search,
    nelder_mead,
    powell_conjugate_direction,
    newton_hessian,
    steepest_descent,
)
from .differentiation import (
    forward_difference,
    backward_difference,
    central_difference,
)
from .integration import trapezoidal, simpson
from .interpolation import lagrange_interpolation, cubic_spline
from .lin_algebra import gauss_seidel, jacobi, gauss_elim, gauss_elim_pivot
from .regression import linear, chebyshev


__all__ = [
    "bisection",
    "newton",
    "regula_falsi",
    "secant",
    "fibonacci_search",
    "golden_search",
    "elimination_search",
    "nelder_mead",
    "powell_conjugate_direction",
    "newton_hessian",
    "steepest_descent",
    "forward_difference",
    "backward_difference",
    "central_difference",
    "trapezoidal",
    "simpson",
    "lagrange_interpolation",
    "cubic_spline",
    "gauss_seidel",
    "jacobi",
    "gauss_elim",
    "gauss_elim_pivot",
    "linear",
    "chebyshev",
]
