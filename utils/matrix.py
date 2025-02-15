# utils/matrix.py

"""Matrix utility functions for optimization algorithms."""

import numpy as np
import scipy.linalg
from typing import Union, Tuple, Optional
from numpy.typing import NDArray


def is_positive_definite(A: NDArray[np.float64]) -> bool:
    """Check if matrix A is positive definite using Cholesky decomposition."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def is_symmetric(
    A: NDArray[np.float64], rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """Check if matrix A is symmetric within tolerance."""
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def nearest_positive_definite(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """Find the nearest positive definite matrix to input matrix A."""
    B = (A + A.T) / 2  # Symmetrize
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def condition_number(A: NDArray[np.float64]) -> float:
    """Compute the condition number of matrix A."""
    return np.linalg.cond(A)


def rank(A: NDArray[np.float64], tol: Optional[float] = None) -> int:
    """Compute the rank of matrix A with optional tolerance."""
    return np.linalg.matrix_rank(A, tol=tol)


def null_space(
    A: NDArray[np.float64], rcond: Optional[float] = None
) -> NDArray[np.float64]:
    """Compute an orthonormal basis for the null space of A."""
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(A.shape)
    mask = s > rcond
    null_mask = np.logical_not(mask)
    null_space = vh[null_mask]
    return null_space.T


def solve_linear_system(
    A: NDArray[np.float64], b: NDArray[np.float64], method: str = "direct"
) -> Tuple[NDArray[np.float64], bool]:
    """Solve linear system Ax = b using specified method.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        method: Solution method ('direct', 'lstsq', or 'pinv')

    Returns:
        Tuple of (solution vector, success flag)
    """
    try:
        if method == "direct":
            x = np.linalg.solve(A, b)
        elif method == "lstsq":
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        elif method == "pinv":
            x = np.linalg.pinv(A) @ b
        else:
            raise ValueError(f"Unknown method: {method}")
        return x, True
    except np.linalg.LinAlgError:
        return np.zeros_like(b), False


def matrix_power(A: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Compute matrix power A^n using efficient algorithm."""
    return np.linalg.matrix_power(A, n)


def eigendecomposition(
    A: NDArray[np.float64], sort: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute eigendecomposition of matrix A.

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    eigenvals, eigenvecs = np.linalg.eig(A)
    if sort:
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
    return eigenvals, eigenvecs


def is_diagonally_dominant(A: NDArray[np.float64]) -> bool:
    """Check if matrix A is diagonally dominant."""
    D = np.abs(np.diag(A))
    S = np.sum(np.abs(A), axis=1) - D
    return np.all(D > S)


def decompose_lu(
    A: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]
]:
    """Compute LU decomposition with pivoting.

    Returns:
        Tuple of (P, L, U, piv), where P@A = L@U
    """
    P, L, U = scipy.linalg.lu(A)
    piv = np.argmax(P, axis=1)
    return P, L, U, piv


def decompose_qr(
    A: NDArray[np.float64], mode: str = "reduced"
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute QR decomposition.

    Args:
        A: Input matrix
        mode: 'reduced' or 'complete'

    Returns:
        Tuple of (Q, R)
    """
    return np.linalg.qr(A, mode=mode)


def decompose_svd(
    A: NDArray[np.float64], full_matrices: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Singular Value Decomposition.

    Returns:
        Tuple of (U, S, Vh) where A = U @ diag(S) @ Vh
    """
    return np.linalg.svd(A, full_matrices=full_matrices)


def matrix_norm(A: NDArray[np.float64], ord: Optional[Union[int, str]] = None) -> float:
    """Compute matrix norm.

    Args:
        A: Input matrix
        ord: Order of the norm (None: Frobenius norm)
    """
    return np.linalg.norm(A, ord=ord)


def is_orthogonal(A: NDArray[np.float64], tol: float = 1e-10) -> bool:
    """Check if matrix A is orthogonal."""
    I = np.eye(len(A))
    return np.allclose(A @ A.T, I, atol=tol) and np.allclose(A.T @ A, I, atol=tol)


def projection_matrix(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute orthogonal projection matrix onto column space of A."""
    return A @ np.linalg.pinv(A)


def gram_schmidt(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """Perform Gram-Schmidt orthogonalization."""
    Q = np.zeros_like(A, dtype=np.float64)

    for i in range(A.shape[1]):
        q = A[:, i]
        for j in range(i):
            q = q - np.dot(q, Q[:, j]) * Q[:, j]
        if np.any(q):  # Check for non-zero vector
            q = q / np.linalg.norm(q)
        Q[:, i] = q

    return Q
