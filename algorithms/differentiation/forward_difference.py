# algorithms/differentiation/forward_difference.py

"""Forward difference method for numerical differentiation."""

from typing import Callable, Any, Union
import numpy as np
from numpy.typing import NDArray

__all__ = ["forward_difference"]


def forward_difference(
    f: Callable[..., Union[float, NDArray[np.float64]]],
    x: Union[float, NDArray[np.float64]],
    h: float,
    *args: Any,
    **kwargs: Any,
) -> Union[float, NDArray[np.float64]]:
    """Compute derivative using forward difference method.

    Args:
        f: Function to differentiate
        x: Point(s) at which to evaluate derivative
        h: Step size
        *args: Additional positional arguments for f
        **kwargs: Additional keyword arguments for f

    Returns:
        Approximation of f'(x)

    Example:
        >>> f = lambda x: x**2
        >>> forward_difference(f, 2.0, 0.01)
        4.01  # Approximates f'(2) = 4
    """
    try:
        return (f(x + h, *args, **kwargs) - f(x, *args, **kwargs)) / h
    except Exception as e:
        raise ValueError(f"Error computing forward difference: {str(e)}")
