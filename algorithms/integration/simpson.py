# methods/integration/simpson.py

"""Simpson's rule for numerical integration."""

from typing import Callable, Any, Union
import numpy as np
from numpy.typing import NDArray

__all__ = ["simpson"]


def simpson(
    f: Callable[..., Union[float, NDArray[np.float64]]],
    a: float,
    b: float,
    n: int,
    *args: Any,
    **kwargs: Any,
) -> float:
    """Approximate definite integral using Simpson's rule.

    Args:
        f: Function to integrate
        a: Lower bound of integration
        b: Upper bound of integration
        n: Number of subintervals (must be even)
        *args: Additional positional arguments for f
        **kwargs: Additional keyword arguments for f

    Returns:
        Approximation of ∫[a,b] f(x)dx

    Example:
        >>> f = lambda x: x**2
        >>> simpson(f, 0, 1, 100)
        0.3333333333333333  # Approximates ∫[0,1] x²dx = 1/3

    Raises:
        ValueError: If n is not even
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even")

    try:
        h = (b - a) / n  # Step size
        x = np.linspace(a, b, n + 1)  # Integration points
        y = f(x, *args, **kwargs)  # Function values

        # Apply Simpson's rule: h/3 * (f₀ + 4f₁ + 2f₂ + 4f₃ + ... + 2fₙ₋₂ + 4fₙ₋₁ + fₙ)
        weights = np.ones(n + 1)
        weights[1:-1:2] = 4  # Odd indices get weight 4
        weights[2:-1:2] = 2  # Even indices get weight 2

        return h / 3 * np.sum(weights * y)
    except Exception as e:
        raise ValueError(f"Error in Simpson integration: {str(e)}")
