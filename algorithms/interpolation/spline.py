# algorithms/interpolation/spline.py

"""Natural cubic spline interpolation method."""

from typing import Callable, List, Union
import numpy as np
from numpy.typing import NDArray

__all__ = ["cubic_spline"]


def cubic_spline(
    x_values: Union[List[float], NDArray[np.float64]],
    y_values: Union[List[float], NDArray[np.float64]],
) -> Callable[[float], float]:
    """Create natural cubic spline interpolation from data points.

    Args:
        x_values: x coordinates of data points (must be sorted)
        y_values: y coordinates of data points

    Returns:
        Callable that evaluates the cubic spline

    Example:
        >>> x = [0, 1, 2]
        >>> y = [1, 2, 4]
        >>> f = cubic_spline(x, y)
        >>> f(1.5)  # Evaluates spline at x = 1.5
        2.875

    Raises:
        ValueError: If x_values are not sorted or inputs have different lengths
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)

    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length")
    if not np.all(np.diff(x_array) > 0):
        raise ValueError("x_values must be strictly increasing")

    n = len(x_array)
    h = np.diff(x_array)

    # Set up tridiagonal system
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1 : i + 2] = [h[i - 1], 2 * (h[i - 1] + h[i]), h[i]]

    # Right-hand side
    b = np.zeros(n)
    for i in range(1, n - 1):
        b[i] = 3 * (
            (y_array[i + 1] - y_array[i]) / h[i]
            - (y_array[i] - y_array[i - 1]) / h[i - 1]
        )

    # Solve for second derivatives
    c = np.linalg.solve(A, b)

    # Compute polynomial coefficients
    b_coef = [
        (y_array[i + 1] - y_array[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        for i in range(n - 1)
    ]
    d_coef = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n - 1)]

    def interpolate(x: float) -> float:
        """Evaluate cubic spline at x."""
        try:
            if x < x_array[0] or x > x_array[-1]:
                raise ValueError("x value out of range")

            # Find appropriate interval
            i = np.searchsorted(x_array, x) - 1
            i = min(i, n - 2)

            # Compute local coordinate
            dx = x - x_array[i]

            # Evaluate cubic polynomial
            return y_array[i] + b_coef[i] * dx + c[i] * dx**2 + d_coef[i] * dx**3
        except Exception as e:
            raise ValueError(f"Error in spline interpolation: {str(e)}")

    return interpolate
