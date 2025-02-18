# methods/root_finding/fibonacci_search.py

"""Fibonacci search method for finding minima."""

from typing import Callable, List, Tuple
from .elimination import elim_step

__all__ = ["fib_generator", "fibonacci_search"]


def fib_generator(n: int) -> List[int]:
    """Generate Fibonacci sequence up to n terms.

    Args:
        n: Number of terms to generate

    Returns:
        List of first n Fibonacci numbers
    """
    if n < 1:
        return []
    elif n == 1:
        return [1]

    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


def fibonacci_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_terms: int,
    tol: float = 1e-6,
) -> Tuple[float, List[float], int]:
    """Find minimum of function using Fibonacci search.

    Args:
        f: Function to minimize
        a: Left endpoint of interval
        b: Right endpoint of interval
        n_terms: Number of Fibonacci terms to use
        tol: Convergence tolerance

    Returns:
        Tuple of (x_min, errors, iterations) where:
            x_min: Approximate minimizer
            errors: List of function values at each iteration
            iterations: Number of iterations used

    Example:
        >>> f = lambda x: x**2
        >>> x, errs, iters = fibonacci_search(f, -1, 1, 10)
        >>> abs(x) < 1e-6
        True
    """
    errors = []
    fib = fib_generator(n_terms + 1)

    # Initial points
    x1 = a + fib[n_terms - 2] / fib[n_terms] * (b - a)
    x2 = a + fib[n_terms - 1] / fib[n_terms] * (b - a)

    for i in range(n_terms):
        # Update interval
        a, b = elim_step(f, a, b, x1, x2)

        # Record error
        errors.append(abs(f((a + b) / 2)))

        # Check convergence
        if abs(b - a) < tol:
            return a, errors, i + 1

        # Check for equal function values
        if abs(f(a) - f(b)) < tol:
            return (a + b) / 2, errors, i + 1

        # Update points
        x1 = a + fib[n_terms - i - 3] / fib[n_terms - i - 1] * (b - a)
        x2 = a + fib[n_terms - i - 2] / fib[n_terms - i - 1] * (b - a)

    return (a + b) / 2, errors, n_terms
