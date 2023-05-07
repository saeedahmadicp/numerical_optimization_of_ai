import numpy as np

def jacobi(A, b, x0, delta, max_it):
    """
    A function implementing the Jacobi iteration method to solve the linear system Ax=b.

    Inputs:
        A: square coefficient matrix
        b: right side vector
        x0: initial guess
        delta: error tolerance for the relative difference between two consecutive iterates
        max_it: maximum number of iterations to be allowed

    Outputs:
        x: numerical solution vector
        iflag: 1 if a numerical solution satisfying the error tolerance is found within max_it iterations, -1 otherwise
        itnum: the number of iterations used to compute x
    """

    iflag = 1
    k = 0
    diagA = np.diag(A)
    A = A - np.diag(diagA)

    while k < max_it:
        k = k + 1
        x = (b - np.dot(A, x0)) / diagA
        relerr = np.linalg.norm(x - x0, np.inf) / (np.linalg.norm(x, np.inf) + np.finfo(float).eps)
        
        # reset the old solution for the next iteration.
        x0 = x
        
        if relerr < delta:
            break

    itnum = k
    if itnum == max_it:
        iflag = -1
    return x, iflag, itnum
