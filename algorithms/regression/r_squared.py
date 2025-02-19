# algorithms/regression/r_squared.py

"""Coefficient of determination (R²) calculation."""

import numpy as np
from numpy.typing import NDArray
from typing import Union


def r_squared(
    y_true: Union[NDArray[np.float64], np.ndarray],
    y_pred: Union[NDArray[np.float64], np.ndarray],
) -> float:
    """Calculate R² (coefficient of determination) score.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        R² score between 0 and 1

    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If input lengths don't match

    Example:
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> r2 = r_squared(y_true, y_pred)
    """
    # Input validation
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have same length")

    # Convert to float64 for numerical stability
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Compute R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)
