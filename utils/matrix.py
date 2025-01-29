# utility file for matrix operations

import numpy as np


def is_positive_definite(A):
    """Return True if A is positive definite.
    >>> A = np.array([[1, 2], [2, -3]])
    >>> print(is_positive_definite(A))  # Output: True
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
