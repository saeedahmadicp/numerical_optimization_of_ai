# algorithms/convex/secant.py

"""Secant method for finding roots."""

from typing import Callable, List, Tuple


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """Find root of f(x) = 0 using secant method.

    Args:
        f: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations) where:
            root: Approximate root of f(x) = 0
            errors: List of absolute errors |f(x)|
            iterations: Number of iterations used

    Example:
        >>> f = lambda x: x**2 - 2
        >>> x, errs, iters = secant(f, 1, 2)
        >>> abs(x - 2**0.5) < 1e-6
        True
    """
    errors = []
    iterations = 0

    # Initial function values
    f0 = f(x0)
    f1 = f(x1)
    errors.append(abs(f1))

    while iterations < max_iter:
        # Check for zero slope
        if abs(f1 - f0) < 1e-10:
            raise ValueError("Secant slope too close to zero")

        # Compute next approximation
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        errors.append(abs(f2))

        # Check convergence
        if abs(f2) <= tol:
            return x2, errors, iterations

        # Update points
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        iterations += 1

    return x1, errors, iterations
