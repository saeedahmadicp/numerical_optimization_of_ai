from interpolation.spline import cubic_spline


def cubic_spline_plot(x_values, y_values):
    """
    This function plots the cubic spline interpolation.
    param: x_values: list of x values
    param: y_values: list of y values
    """
    import matplotlib.pyplot as plt
    from numpy import linspace
    x = linspace(min(x_values), max(x_values), 100)
    y = [cubic_spline(x_values, y_values)(i) for i in x]
    plt.plot(x, y, 'b', x_values, y_values, 'ro')
    plt.show()


if __name__ == "__main__":
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    cubic_spline_plot(x_values, y_values)