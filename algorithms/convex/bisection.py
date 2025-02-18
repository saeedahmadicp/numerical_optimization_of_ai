# algorithms/convex/bisection.py

"""Bisection method for finding roots of continuous functions."""

from typing import Callable, List, Tuple

__all__ = ["bisection"]


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """Find root of f(x) = 0 using bisection method.

    Args:
        f: Function to find root of
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations) where:
            root: Approximate root of f(x) = 0
            errors: List of absolute errors |f(x)|
            iterations: Number of iterations used

    Raises:
        ValueError: If f(a) and f(b) have same sign

    Example:
        >>> f = lambda x: x**2 - 2
        >>> x, errs, iters = bisection(f, 1, 2)
        >>> abs(x - 2**0.5) < 1e-6
        True
    """
    # Validate input interval
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Function must have opposite signs at interval endpoints")

    errors = []
    iterations = 0

    while iterations < max_iter:
        # Compute midpoint
        c = (a + b) / 2
        fc = f(c)
        errors.append(abs(fc))

        # Check convergence
        if abs(fc) <= tol:
            return c, errors, iterations

        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    return (a + b) / 2, errors, iterations
