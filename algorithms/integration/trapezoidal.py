# methods/integration/trapezoidal.py

"""Trapezoidal rule for numerical integration."""

from typing import Callable, Any, Union
import numpy as np
from numpy.typing import NDArray

__all__ = ["trapezoidal"]


def trapezoidal(
    f: Callable[..., Union[float, NDArray[np.float64]]],
    a: float,
    b: float,
    n: int,
    *args: Any,
    **kwargs: Any,
) -> float:
    """Approximate definite integral using the trapezoidal rule.

    Args:
        f: Function to integrate
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals
        *args: Additional positional arguments for f
        **kwargs: Additional keyword arguments for f

    Returns:
        Approximation of ∫[a,b] f(x)dx

    Example:
        >>> f = lambda x: x**2
        >>> trapezoidal(f, 0, 1, 100)
        0.3333583333333333  # Approximates ∫[0,1] x²dx = 1/3
    """
    try:
        h = (b - a) / n  # Step size
        x = np.linspace(a, b, n + 1)  # Integration points
        y = f(x, *args, **kwargs)  # Function values

        # Apply trapezoidal rule: h * (f₀/2 + f₁ + f₂ + ... + fₙ₋₁ + fₙ/2)
        return h * (np.sum(y) - (y[0] + y[-1]) / 2)
    except Exception as e:
        raise ValueError(f"Error in trapezoidal integration: {str(e)}")
