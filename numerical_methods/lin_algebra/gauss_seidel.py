import numpy as np

__all__ = ['gauss_seidel']

def gauss_seidel(A, b, x0, delta, max_it):
    """
    The function employs the Gauss-Seidel iteration method to solve
    the linear system Ax=b.

    Input:
        A: square coefficient matrix
        b: right side vector
        x0: initial guess
        delta: error tolerance for the relative difference between
               two consecutive iterates
        max_it: maximum number of iterations to be allowed

    Output:
        x: numerical solution vector
        iflag: 1 if a numerical solution satisfying the error
                 tolerance is found within max_it iterations
               -1 if the program fails to produce a numerical
                  solution in max_it iterations
        itnum: the number of iterations used to compute x
    """

    n = len(b)
    iflag = 1
    k = 0
    x = x0.copy()

    while k < max_it:
        k += 1

        for i in range(n):
            # update x(i), the ith component of the solution
            x_old = x[i]
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]

            # check for divergence
            if np.isnan(x[i]) or np.isinf(x[i]):
                iflag = -1
                break

            # compute relative error
            relerr = np.abs(x[i] - x_old) / (np.abs(x[i]) + np.finfo(float).eps)
            if relerr > delta:
                break

        # check for convergence
        if relerr <= delta:
            break

    itnum = k
    if itnum == max_it or not np.allclose(A @ x, b):
        iflag = -1
        print('Gauss-Seidel failed to find the correct solution in %d iterations.' % max_it)
        print('The last computed solution is:', x)
    else:
        print('Gauss-Seidel converged to the correct solution in %d iterations.' % itnum)
    
    return x, iflag, itnum