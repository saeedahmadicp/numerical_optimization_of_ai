# utils/initial_guess.py

"""Utility for finding initial intervals for root-finding methods."""

import numpy as np

__all__ = ["find_initial_interval"]


def find_initial_interval(f, x0, d0):
    """Find initial interval containing a minimum for unimodal functions.

    Args:
        f: Function to evaluate
        x0: Initial guess
        d0: Initial step size
    Returns:
        [a, b]: Interval containing the minimum
    """
    # Evaluate function at three points
    f_minus, f_0, f_plus = f(x0 - d0), f(x0), f(x0 + d0)

    # Determine search direction based on function values
    if f_minus >= f_0 >= f_plus:
        d, x_minus_1, x1 = d0, x0 - d0, x0 + d0
    elif f_minus <= f_0 <= f_plus:
        d, x_minus_1, x1 = -d0, x0 + d0, x0 - d0
    elif f_minus >= f_0 <= f_plus:
        return [x0 - d0, x0 + d0]
    else:
        raise ValueError("Cannot determine minimum direction")

    # Expand interval until minimum is bracketed
    xk_minus_1, xk, xk_plus_1 = x_minus_1, x0, x1
    while True:
        xk_plus_2 = xk + 2 * d
        if f(xk_plus_2) >= f(xk):
            return [xk_minus_1, xk_plus_2] if d > 0 else [xk_plus_2, xk_minus_1]
        xk_minus_1, xk, xk_plus_1 = xk, xk_plus_1, xk_plus_2


if __name__ == "__main__":
    f = lambda x: x
    x0 = np.random.rand(1)[0]
    d0 = 0.1
    a, b = find_initial_interval(f, x0, d0)
    print(find_initial_interval(f, x0, d0))
    print(f"function value at a: {f(a)}")
    print(f"function value at b: {f(b)}")
