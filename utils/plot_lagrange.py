from interpolation.lagrange import lagrange_interpolation
import matplotlib.pyplot as plt
import numpy as np


def lagrange_interpolation_plot(x_values, y_values):
    """
    This function plots the Lagrange interpolation.
    param: x_values: list of x values
    param: y_values: list of y values
    """
    
    x = np.linspace(min(x_values), max(x_values), 100)
    y = [lagrange_interpolation(x_values, y_values)(i) for i in x]
    
    plt.plot(x, y, 'b', x_values, y_values, 'ro')
    plt.show()


if __name__ == "__main__":
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    lagrange_interpolation_plot(x_values, y_values)
