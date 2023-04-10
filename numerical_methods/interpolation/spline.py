import numpy as np

__all__ = ["cubic_spline"]

def cubic_spline(x_values, y_values):
    """
    This function performs cubic spline interpolation.
    param: x_values: list of x values
    param: y_values: list of y values
    return: function that interpolates the given points
    """
    n = len(x_values)
    h = [x_values[i+1] - x_values[i] for i in range(n-1)]
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n-1, n-1] = 1
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
    b = np.zeros(n)
    for i in range(1, n-1):
        b[i] = 3 * ((y_values[i+1] - y_values[i]) / h[i] - (y_values[i] - y_values[i-1]) / h[i-1])
    c = np.linalg.solve(A, b)
    b = [(y_values[i+1] - y_values[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3 for i in range(n-1)]
    d = [(c[i+1] - c[i]) / (3 * h[i]) for i in range(n-1)]
    def _basis(i):
        """
        This function returns the basis function for the ith interval.
        """
        return lambda x: y_values[i] + b[i] * (x - x_values[i]) + c[i] * (x - x_values[i])**2 + d[i] * (x - x_values[i])**3
    
    def _spline(x):
        """
        This function checks if x is in the range of x_values and returns the corresponding spline value.
        """
        if x < x_values[0] or x > x_values[-1]:
            raise ValueError("x value out of range")
        i = 0
        while i < n-1 and x > x_values[i+1]:
            i += 1
        return _basis(i)(x)
    
    return _spline

