# methods/differentiation/central_difference.py

"""Central difference method for numerical differentiation."""

from typing import Callable, Any, Union
import numpy as np
from numpy.typing import NDArray


def central_difference(
    f: Callable[..., Union[float, NDArray[np.float64]]],
    x: Union[float, NDArray[np.float64]],
    h: float,
    *args: Any,
    **kwargs: Any,
) -> Union[float, NDArray[np.float64]]:
    """Compute derivative using central difference method.

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
        >>> central_difference(f, 2.0, 0.01)
        4.0000000001  # Approximates f'(2) = 4
    """
    try:
        return (f(x + h, *args, **kwargs) - f(x - h, *args, **kwargs)) / (2 * h)
    except Exception as e:
        raise ValueError(f"Error computing central difference: {str(e)}")
