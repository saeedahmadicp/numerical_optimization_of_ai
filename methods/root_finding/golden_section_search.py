# methods/root_finding/golden_section_search.py

"""Golden section search method for finding minima."""

from typing import Callable, List, Tuple
import math
from .elimination import elim_step

__all__ = ["golden_search"]


def golden_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[float, List[float], int]:
    """Find minimum of function using golden section search.

    Args:
        f: Function to minimize
        a: Left endpoint of interval
        b: Right endpoint of interval
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (x_min, errors, iterations) where:
            x_min: Approximate minimizer
            errors: List of function values at each iteration
            iterations: Number of iterations used

    Example:
        >>> f = lambda x: x**2
        >>> x, errs, iters = golden_search(f, -1, 1)
        >>> abs(x) < 1e-6
        True
    """
    errors = []
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    tau = 1 / phi  # 1/φ ≈ 0.618034

    # Initial points
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)

    for i in range(max_iter):
        # Handle equal function values
        if abs(f(x1) - f(x2)) < 1e-10:
            x2 += 1e-6

        # Update interval
        a, b = elim_step(f, a, b, x1, x2)

        # Record error
        errors.append(abs(f((a + b) / 2)))

        # Check convergence
        if abs(b - a) < tol:
            return a, errors, i + 1

        # Update points
        x1 = a + (1 - tau) * (b - a)
        x2 = a + tau * (b - a)

    return (a + b) / 2, errors, max_iter
