# algorithms/lin_algebra/gauss_elim.py

"""Gaussian elimination method for solving linear systems."""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

__all__ = ["gauss_elim"]


def gauss_elim(
    A: NDArray[np.float64], b: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve linear system Ax = b using Gaussian elimination.

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
        >>> A = np.array([[2, 1], [1, -1]])
        >>> b = np.array([3, 1])
        >>> x, _, _ = gauss_elim(A, b)
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

    # Forward elimination
    for k in range(n - 1):
        if abs(A[k, k]) < np.finfo(float).eps:
            raise np.linalg.LinAlgError("Zero pivot encountered")

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
