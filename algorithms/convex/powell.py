# algorithms/convex/powell.py

"""Powell's conjugate direction method for optimization."""

from typing import Callable, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

__all__ = ["powell_conjugate_direction"]


def powell_conjugate_direction(
    f: Callable[..., float],
    x0: NDArray[np.float64],
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[NDArray[np.float64], List[float], int, NDArray[np.float64]]:
    """Find minimum using Powell's conjugate direction method.

    Args:
        f: Function to minimize
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        verbose: Whether to print progress

    Returns:
        Tuple of (x_min, errors, iterations, history) where:
            x_min: Approximate minimizer
            errors: List of function values at each iteration
            iterations: Number of iterations used
            history: Array of points visited

    Example:
        >>> def f(x, y): return x**2 + 2*y**2
        >>> x0 = np.array([1.0, 1.0])
        >>> x, errs, iters, _ = powell_conjugate_direction(f, x0)
        >>> np.allclose(x, [0, 0], atol=1e-5)
        True
    """
    n = len(x0)
    x = x0.copy()
    directions = np.eye(n)
    errors = []
    history = [x0.copy()]

    for i in range(max_iter):
        x_prev = x.copy()

        # Minimize along each direction
        for j in range(n):
            # Create line search function
            def line_func(alpha):
                return f(*(x + alpha * directions[j]))

            # Minimize along current direction
            res = minimize_scalar(line_func)
            if not res.success:
                if verbose:
                    print("Line search failed")
                continue

            x = x + res.x * directions[j]
            errors.append(f(*x))
            history.append(x.copy())

        # Update directions
        directions[:-1] = directions[1:]
        directions[-1] = x - x_prev

        # Check convergence
        if np.linalg.norm(x - x_prev) < tol:
            if verbose:
                print(f"Converged in {i + 1} iterations")
            return x, errors, i + 1, np.array(history)

    if verbose:
        print(f"Failed to converge in {max_iter} iterations")
    return x, errors, max_iter, np.array(history)
