import numpy as np

__all__ = ["lagrange_interpolation"]

def lagrange_interpolation(x_values, y_values):
    """
    This function performs Lagrange interpolation.
    :param x_values: list of x values
    :param y_values: list of y values
    :return: function that interpolates the given points
    """

    def basis(j):
        """
        This function returns the basis function for the jth point.
        """
        xj = x_values[j]
        return lambda x: np.prod((x - x_values[m]) / (xj - x_values[m]) for m in range(k) if m != j)

    k = len(x_values)
    basis_functions = [basis(j) for j in range(k)]
    return lambda x: sum(y_values[j] * basis_functions[j](x) for j in range(k))
