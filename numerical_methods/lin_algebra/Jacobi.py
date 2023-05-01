import numpy as np

def Jacobi(A, b, x0, delta, max_it):
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


if __name__ == "__main__":
    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
    b = np.array([7, -8, 6])
    x0 = np.zeros(3)
    delta = 1e-6
    max_it = 1000

    x, iflag, itnum = Jacobi(A, b, x0, delta, max_it)
    print("The solution vector is:\n", x)

    expected_x = np.linalg.solve(A, b)
    tolerance = 1e-6
    
    if np.allclose(x, expected_x, rtol=tolerance, atol=tolerance):
        print("Test case passed: the solution is correct.")
    else:
        print("Test case failed: the solution is not correct.") 