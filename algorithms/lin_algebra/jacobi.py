# methods/lin_algebra/jacobi.py

"""Jacobi iterative method for solving linear systems."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

__all__ = ["jacobi"]


def jacobi(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[NDArray[np.float64], bool, int]:
    """Solve linear system Ax = b using Jacobi iteration.

    Args:
        A: Square coefficient matrix
        b: Right-hand side vector
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        verbose: Whether to print convergence messages

    Returns:
        Tuple of (x, converged, iterations) where:
            x: Solution vector
            converged: Whether method converged
            iterations: Number of iterations used

    Example:
        >>> A = np.array([[4, -1], [1, 3]])  # Diagonally dominant
        >>> b = np.array([1, 2])
        >>> x0 = np.zeros(2)
        >>> x, conv, iters = jacobi(A, b, x0)
    """
    # Extract diagonal and remainder
    D = np.diag(A)
    R = A - np.diag(D)
    x = x0.copy()

    # Check diagonal dominance
    if not np.all(2 * np.abs(D) >= np.sum(np.abs(A), axis=1)):
        if verbose:
            print("Warning: Matrix is not diagonally dominant")

    # Iterate
    for k in range(max_iter):
        x_new = (b - R @ x) / D

        if not np.all(np.isfinite(x_new)):
            return x, False, k + 1

        # Check convergence
        if np.linalg.norm(x_new - x, np.inf) < tol:
            if verbose:
                print(f"Converged in {k + 1} iterations")
            return x_new, True, k + 1

        x = x_new

    if verbose:
        print(f"Failed to converge in {max_iter} iterations")
    return x, False, max_iter
