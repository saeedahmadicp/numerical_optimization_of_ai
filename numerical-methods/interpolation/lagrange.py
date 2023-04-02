import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_values, y_values):
    """
    This function performs Lagrange interpolation.
    param: x_values: list of x values
    param: y_values: list of y values
    return: function that interpolates the given points
    """

    def _basis(j, x):
        """
        This function returns the basis function for the jth point.
        """
        p = [(x - x_values[m]) / (x_values[j] - x_values[m]) for m in range(k) if m != j]
        return np.prod(p)
    
    k = len(x_values)
    return lambda x: sum(_basis(j, x) * y_values[j] for j in range(k))