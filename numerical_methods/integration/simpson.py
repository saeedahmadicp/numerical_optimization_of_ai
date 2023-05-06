import numpy as np
import inspect

__all__ = ["simpson"]

def simpson(f, a, b, n, *args, **kwargs):
    """
    This function performs the Simpson's rule.
    :param f: function to integrate
    :param a: lower bound
    :param b: upper bound
    :param n: number of subintervals
    :param args: additional positional arguments for the function f
    :param kwargs: additional keyword arguments for the function f
    :return: approximation of the integral
    """

    # Check the number of arguments that the function f takes
    num_args = len(inspect.signature(f).parameters)

    # Calculate the step size (h is the width of each subinterval)
    h = (b - a) / n

    # Evaluate the integrand at the left and right endpoints
    if num_args == 1 or num_args == 2:
        left = f(a, *args, **kwargs)
        right = f(b, *args, **kwargs)
    else:
        raise ValueError("The function f must take either 1 or 2 arguments.")

    # Calculate the sum of the odd and even terms
    odd = np.sum(f(a + h * np.arange(1, n, 2), *args, **kwargs))
    even = np.sum(f(a + h * np.arange(2, n, 2), *args, **kwargs))

    return h / 3 * (left + right + 4 * odd + 2 * even)
