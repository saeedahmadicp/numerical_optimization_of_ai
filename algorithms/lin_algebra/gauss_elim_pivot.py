# algorithms/lin_algebra/gauss_elim_pivot.py

"""Gaussian elimination with partial pivoting for solving linear systems."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

__all__ = ["gauss_elim_pivot"]


def gauss_elim_pivot(
    A: NDArray[np.float64], b: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve linear system Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: Square coefficient matrix
        b: Right-hand side vector

    Returns:
        Tuple of (x, An, bn) where:
            x: Solution vector
            An: Coefficient matrix after elimination
            bn: Right-hand side vector after elimination

    Raises:
        ValueError: If A is not square or dimensions don't match
        np.linalg.LinAlgError: If system is singular

    Example:
        >>> A = np.array([[0, 1], [2, 1]])  # First pivot would be zero
        >>> b = np.array([1, 3])
        >>> x, _, _ = gauss_elim_pivot(A, b)
        >>> x  # Should be [1, 1]
        array([1., 1.])
    """
    # Validate inputs
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A must be square")
    if m != len(b):
        raise ValueError("Dimensions of A and b must match")

    # Create working copies
    A = A.astype(np.float64, copy=True)
    b = b.astype(np.float64, copy=True)

    # Forward elimination with pivoting
    for k in range(n - 1):
        # Find pivot
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[pivot_row, k]) < np.finfo(float).eps:
            raise np.linalg.LinAlgError("Matrix is singular")

        # Swap rows if necessary
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        # Eliminate
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1 :] -= factor * A[k, k + 1 :]
            A[i, k] = factor
            b[i] -= factor * b[k]

    # Back substitution
    x = np.zeros(n, dtype=np.float64)
    x[-1] = b[-1] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]

    return x, A.copy(), b.copy()
