# utils/midpoint.py

"""Utilities for finding appropriate initial points for root finding methods."""

from typing import Tuple, Callable
import numpy as np


def find_bracket_points(
    f: Callable[[float], float], x_range: Tuple[float, float], num_points: int = 20
) -> Tuple[float, float]:
    """
    Find two points that bracket a root (have opposite signs).

    Args:
        f: Function to find roots for
        x_range: Tuple of (min_x, max_x) to search within
        num_points: Number of points to sample in the range

    Returns:
        Tuple of (x1, x2) where f(x1) and f(x2) have opposite signs

    Raises:
        ValueError: If no bracketing points found in the given range
    """
    x_min, x_max = x_range
    x_points = np.linspace(x_min, x_max, num_points)
    y_values = [f(x) for x in x_points]

    # Look for sign changes
    for i in range(len(y_values) - 1):
        if y_values[i] * y_values[i + 1] <= 0:  # Sign change detected
            return x_points[i], x_points[i + 1]

    raise ValueError(
        f"No root bracketing points found in range [{x_min}, {x_max}]. "
        "Try a different range or increase num_points."
    )


def get_safe_initial_points(
    f: Callable,
    x_range: Tuple[float, float],
    method_name: str,
    x0=None,
) -> Tuple:
    """
    Get appropriate initial point(s) for different root finding methods.

    Args:
        f: Function to find roots for
        x_range: Tuple of (min_x, max_x) to search within
        method_name: Name of the method to get initial points for
        x0: User-provided initial point (optional)

    Returns:
        Tuple of (x0, x1) where x0 and/or x1 are appropriate initial points
        For 2D functions, returns points as numpy arrays
    """
    # Handle 2D functions
    if x0 is not None and isinstance(x0, (list, np.ndarray)) and len(x0) > 1:
        # For 2D functions, if x0 is provided as array/list
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)

        # For methods requiring two points, create a second point nearby
        if method_name in ["secant", "bisection", "regula_falsi"]:
            # Create a second point with a small perturbation
            x1 = x0.copy()
            x1[0] += 0.1  # Small shift in x direction
            return x0, x1
        else:
            # For methods requiring just one point
            return x0, None

    # Original 1D handling
    if method_name in ["secant", "bisection", "regula_falsi"]:
        # Methods that need two points that bracket a root
        if x0 is not None:
            # If user provided an initial point, use it as x0
            # and generate a reasonable x1 (0.1 units away)
            x1 = x0 + 0.1
            return x0, x1
        else:
            # Try to find bracket points automatically
            try:
                return find_bracket_points(f, x_range)
            except ValueError:
                # If no bracket found, use endpoints of range
                return x_range[0], x_range[1]
    else:
        # Methods that only need one point
        if x0 is not None:
            return x0, None
        else:
            # Use midpoint of range as default
            midpoint = (x_range[0] + x_range[1]) / 2
            return midpoint, None
