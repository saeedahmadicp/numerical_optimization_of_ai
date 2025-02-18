# methods/root_finding/elimination.py

"""Elimination method for finding roots."""

from typing import Callable, List, Tuple
import random

__all__ = ["elim_step", "elimination_search"]


def elim_step(
    f: Callable[[float], float],
    a: float,
    b: float,
    x1: float,
    x2: float,
) -> Tuple[float, float]:
    """Perform one step of elimination method.

    Args:
        f: Function to minimize
        a: Left endpoint of interval
        b: Right endpoint of interval
        x1: First test point
        x2: Second test point

    Returns:
        New interval endpoints (a', b')
    """
    f1, f2 = f(x1), f(x2)

    if f1 < f2:
        return a, x2
    elif f1 > f2:
        return x1, b
    else:
        return x1, x2


"""
The problem with elimination method is that it is not guaranteed to 
converge to the root as it depends on the random numbers generated.
"""


def elimination_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[float, List[float], int]:
    """Find minimum of function using elimination method.

    Args:
        f: Function to minimize
        a: Left endpoint of interval
        b: Right endpoint of interval
        max_iter: Maximum number of iterations
        tol: Tolerance for interval width

    Returns:
        Tuple of (x_min, errors, iterations) where:
            x_min: Approximate minimizer
            errors: List of function values at each iteration
            iterations: Number of iterations used

    Example:
        >>> f = lambda x: x**2
        >>> x, errs, iters = elimination_search(f, -1, 1)
        >>> abs(x) < 1e-6
        True
    """
    errors = []

    for i in range(max_iter):
        # Generate test points
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)

        # Update interval
        a, b = elim_step(f, a, b, x1, x2)

        # Record error
        x_mid = (a + b) / 2
        errors.append(abs(f(x_mid)))

        # Check convergence
        if abs(b - a) < tol:
            return x_mid, errors, i + 1

        # Check for equal function values
        if abs(f(a) - f(b)) < tol:
            return (a + b) / 2, errors, i + 1

    return (a + b) / 2, errors, max_iter
