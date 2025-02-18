# algorithms/interpolation/lagrange.py

"""Lagrange polynomial interpolation method."""

from typing import Callable, List, Union
import numpy as np
from numpy.typing import NDArray

__all__ = ["lagrange_interpolation"]


def lagrange_interpolation(
    x_values: Union[List[float], NDArray[np.float64]],
    y_values: Union[List[float], NDArray[np.float64]],
) -> Callable[[float], float]:
    """Create Lagrange interpolation polynomial from data points.

    Args:
        x_values: x coordinates of data points
        y_values: y coordinates of data points

    Returns:
        Callable that evaluates the interpolating polynomial

    Example:
        >>> x = [0, 1, 2]
        >>> y = [1, 2, 4]
        >>> f = lagrange_interpolation(x, y)
        >>> f(1.5)  # Evaluates polynomial at x = 1.5
        2.875
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)

    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length")

    def basis(j: int, x: float) -> float:
        """Compute jth Lagrange basis polynomial at x."""
        terms = [
            (x - x_array[m]) / (x_array[j] - x_array[m])
            for m in range(len(x_array))
            if m != j
        ]
        return np.prod(terms)

    def interpolate(x: float) -> float:
        """Evaluate interpolating polynomial at x."""
        try:
            return sum(y_array[j] * basis(j, x) for j in range(len(x_array)))
        except Exception as e:
            raise ValueError(f"Error in Lagrange interpolation: {str(e)}")

    return interpolate
