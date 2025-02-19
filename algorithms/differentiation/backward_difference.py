# algorithms/differentiation/backward_difference.py

"""Backward difference method for numerical differentiation."""

from typing import Callable, Any, Union
import numpy as np
from numpy.typing import NDArray


def backward_difference(
    f: Callable[..., Union[float, NDArray[np.float64]]],
    x: Union[float, NDArray[np.float64]],
    h: float,
    *args: Any,
    **kwargs: Any,
) -> Union[float, NDArray[np.float64]]:
    """Compute derivative using backward difference method.

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
        >>> backward_difference(f, 2.0, 0.01)
        3.99  # Approximates f'(2) = 4
    """
    try:
        return (f(x, *args, **kwargs) - f(x - h, *args, **kwargs)) / h
    except Exception as e:
        raise ValueError(f"Error computing backward difference: {str(e)}")
