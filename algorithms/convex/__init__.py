# algorithms/convex/__init__.py

"""Root-finding methods for nonlinear equations."""

from .bisection import bisection_search
from .elimination import elimination_search
from .fibonacci import fibonacci_search
from .golden_section import golden_section_search
from .nelder_mead import nelder_mead_search
from .newton_hessian import newton_hessian_search
from .newton import newton_search
from .powell import powell_search
from .quasi_newton import bfgs_root_search
from .regula_falsi import regula_falsi_search
from .secant import secant_search
from .steepest_descent import steepest_descent_search


__all__ = [
    "bisection_search",
    "elimination_search",
    "fibonacci_search",
    "golden_section_search",
    "nelder_mead_search",
    "newton_hessian_search",
    "newton_search",
    "powell_search",
    "bfgs_root_search",
    "regula_falsi_search",
    "secant_search",
    "steepest_descent_search",
]
