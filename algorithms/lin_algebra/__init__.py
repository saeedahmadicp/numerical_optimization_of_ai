# algorithms/lin_algebra/__init__.py

"""Linear algebra methods for solving systems of equations."""

from .gauss_elim import gauss_elim
from .gauss_elim_pivot import gauss_elim_pivot
from .gauss_seidel import gauss_seidel
from .jacobi import jacobi

__all__ = ["gauss_elim", "gauss_elim_pivot", "gauss_seidel", "jacobi"]
