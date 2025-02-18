# algorithms/regression/linear.py

"""Linear regression using ordinary least squares method."""

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Union
from .r_squared import r_squared


class LinearResult(NamedTuple):
    """Results from linear regression."""

    slope: float
    intercept: float
    y_pred: NDArray[np.float64]
    r_squared: float


def linear(
    x: Union[NDArray[np.float64], np.ndarray], y: Union[NDArray[np.float64], np.ndarray]
) -> LinearResult:
    """Fit linear regression model y = ax + b using least squares.

    Args:
        x: Independent variable values
        y: Dependent variable values

    Returns:
        NamedTuple containing:
            slope: Coefficient a in y = ax + b
            intercept: Coefficient b in y = ax + b
            y_pred: Predicted y values
            r_squared: R² score (coefficient of determination)

    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If input lengths don't match

    Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 3.8, 6.2, 7.8, 9.3])
        >>> result = linear(x, y)
        >>> print(f"y = {result.slope:.2f}x + {result.intercept:.2f}")
        y = 1.84x + 0.37
    """
    # Input validation
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if len(x) != len(y):
        raise ValueError("Input arrays must have same length")

    # Convert to float64 for numerical stability
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)

    # Compute means and deviations
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope and intercept using least squares
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Compute predictions and R²
    y_pred = slope * x + intercept
    r2 = r_squared(y, y_pred)

    return LinearResult(slope, intercept, y_pred, r2)
