# algorithms/regression/chebyshev.py

"""Chebyshev polynomial regression."""

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Union
from .r_squared import r_squared


class ChebyshevResult(NamedTuple):
    """Results from Chebyshev polynomial regression."""

    weights: NDArray[np.float64]
    y_pred: NDArray[np.float64]
    r_squared: float


def chebyshev(
    x: Union[NDArray[np.float64], np.ndarray],
    y: Union[NDArray[np.float64], np.ndarray],
    degree: int,
) -> ChebyshevResult:
    """Fit Chebyshev polynomial regression of specified degree.

    Args:
        x: Independent variable values (must be in [-1, 1])
        y: Dependent variable values
        degree: Degree of polynomial

    Returns:
        NamedTuple containing:
            weights: Coefficients of Chebyshev polynomials
            y_pred: Predicted y values
            r_squared: R² score (coefficient of determination)

    Raises:
        ValueError: If x values outside [-1, 1] or degree < 0

    Example:
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.cos(2*np.pi*x) + 0.1*np.random.randn(100)
        >>> result = chebyshev(x, y, degree=4)
    """
    # Input validation
    if np.any(np.abs(x) > 1):
        raise ValueError("x values must be in [-1, 1]")
    if degree < 0:
        raise ValueError("Degree must be non-negative")

    # Convert to float64 for numerical stability
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = len(x)

    # Compute Chebyshev coefficients
    weights = np.zeros(degree + 1, dtype=np.float64)
    for i in range(degree + 1):
        weights[i] = (2 / m) * np.sum(y * np.cos(i * np.arccos(x)))

    # Compute predictions
    y_pred = predict(x, weights)
    r2 = r_squared(y, y_pred)

    return ChebyshevResult(weights, y_pred, r2)


def predict(
    x: Union[NDArray[np.float64], np.ndarray], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate Chebyshev polynomial at given points.

    Args:
        x: Points at which to evaluate polynomial
        weights: Coefficients of Chebyshev polynomials

    Returns:
        Polynomial values at x
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)

    for i, w in enumerate(weights):
        y += w * np.cos(i * np.arccos(x))

    return y
