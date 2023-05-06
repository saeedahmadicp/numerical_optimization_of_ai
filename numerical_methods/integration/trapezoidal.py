import numpy as np
import inspect

__all__ = ["trapezoidal"]

def trapezoidal(f, a, b, n, *args, **kwargs):
    """
    This function performs the trapezoidal rule.
    param: f: function to integrate
    param: a: lower bound
    param: b: upper bound
    param: n: number of subintervals
    *args: additional positional arguments for the function f
    **kwargs: additional keyword arguments for the function f
    return: approximation of the integral
    """

    # Check the number of arguments that the function f takes
    num_args = len(inspect.signature(f).parameters)

    # Calculate the step size (h is the width of each subinterval)
    h = (b - a) / n

    if num_args == 1 or num_args == 2:
        left = f(a, *args, **kwargs) / 2
        right = f(b, *args, **kwargs) / 2
        middle = np.sum(f(a + h * np.arange(1, n), *args, **kwargs))
    else:
        raise ValueError("The function f must take either 1 or 2 arguments.")

    return h * (left + middle + right)