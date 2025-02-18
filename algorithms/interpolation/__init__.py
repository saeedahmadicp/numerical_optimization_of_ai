# algorithms/interpolation/__init__.py

"""Numerical interpolation methods."""

from .lagrange import lagrange_interpolation
from .spline import cubic_spline

__all__ = ["lagrange_interpolation", "cubic_spline"]
