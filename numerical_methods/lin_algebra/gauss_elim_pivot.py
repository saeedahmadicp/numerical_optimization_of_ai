import numpy as np

def gauss_elim_pivot(A, b):
    """
    This function employs the Gaussian elimination method with partial 
    pivoting to solve the linear system Ax=b.
    
    Input:
        A: coefficient square matrix
        b: right side column vector
    Output:
        x: solution vector
        An, bn: coefficient matrix and right hand side vector after row operations
    """

    # check the order of the matrix and the size of the vector
    m, n = np.shape(A)
    if m != n:
        raise ValueError('The matrix is not square.')
    if m != len(b):
        raise ValueError('The matrix and the vector do not match in size.')
    
    # convert A and b to floats to ensure correct division later on
    A = A.astype(float)
    b = b.astype(float)

    # elimination step
    for k in range(n-1):
        # pivoting: find the maximal element in column k, starting from row k, 
        # then swap that row with row k
        for i in range(k+1, n):
            if abs(A[i,k]) > abs(A[k,k]):
                # swap rows k and i if the leading entry A(k,k) is smaller than A(i,k) in magnitude
                A[[k,i],:] = A[[i,k],:]
                b[[k,i]] = b[[i,k]]
        
        # row elimination.
        for i in range(k+1, n):
            factor = A[i,k] / A[k,k]
            A[i,k+1:n] -= factor*A[k,k+1:n]
            A[i,k] = factor
            b[i] -= factor*b[k]
    
    # back substitution: solve the upper triangular linear system.
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1,n-1]
    for i in range(n-2,-1,-1):
        x[i] = (b[i] - np.dot(A[i, i+1:],x[i+1:])) / A[i,i]
    
    An = A.copy()
    bn = b.copy()
    
    return x, An, bn
