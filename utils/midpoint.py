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
    f: Callable[[float], float],
    x_range: Tuple[float, float],
    method_name: str,
    x0: float = None,
) -> Tuple[float, float]:
    """
    Get appropriate initial point(s) for different root finding methods.

    Args:
        f: Function to find roots for
        x_range: Tuple of (min_x, max_x) to search within
        method_name: Name of the root finding method
        x0: Optional initial point (for methods that use one point)

    Returns:
        Tuple of (x0, x1) where:
        - For single-point methods: x1 is None
        - For bracketing methods: x0, x1 bracket a root
    """
    # Methods requiring bracketing points
    if method_name in ["bisection", "regula_falsi"]:
        return find_bracket_points(f, x_range)

    # Methods requiring two points but not necessarily bracketing
    elif method_name == "secant":
        if x0 is None:
            x0 = (x_range[0] + x_range[1]) / 2  # Use midpoint if no x0 provided
        x1 = x0 + 0.5  # Default offset for second point
        return x0, x1

    # Methods requiring single point
    else:
        return x0 or (x_range[0] + x_range[1]) / 2, None
