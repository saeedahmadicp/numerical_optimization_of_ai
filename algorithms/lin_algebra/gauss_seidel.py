# algorithms/lin_algebra/gauss_seidel.py

"""Gauss-Seidel iterative method for solving linear systems."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

__all__ = ["gauss_seidel"]


def gauss_seidel(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    x0: NDArray[np.float64],
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[NDArray[np.float64], bool, int]:
    """Solve linear system Ax = b using Gauss-Seidel iteration.

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
        >>> x, conv, iters = gauss_seidel(A, b, x0)
    """
    n = len(b)
    x = x0.copy()

    # Check diagonal dominance
    if not np.all(2 * np.abs(np.diag(A)) >= np.sum(np.abs(A), axis=1)):
        if verbose:
            print("Warning: Matrix is not diagonally dominant")

    # Iterate
    for k in range(max_iter):
        x_old = x.copy()

        # Update each component
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x_old[i + 1 :]) / A[i, i]

            if not np.isfinite(x[i]):
                return x_old, False, k + 1

        # Check convergence
        if np.linalg.norm(x - x_old, np.inf) < tol:
            if verbose:
                print(f"Converged in {k + 1} iterations")
            return x, True, k + 1

    if verbose:
        print(f"Failed to converge in {max_iter} iterations")
    return x, False, max_iter
