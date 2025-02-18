# methods/regression/__init__.py

"""Regression methods for curve fitting."""

from .linear import linear, LinearResult
from .chebyshev import chebyshev, ChebyshevResult
from .r_squared import r_squared

__all__ = ["linear", "LinearResult", "chebyshev", "ChebyshevResult", "r_squared"]
