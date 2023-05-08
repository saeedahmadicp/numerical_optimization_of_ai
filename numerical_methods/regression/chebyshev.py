import numpy as np
from collections import namedtuple
from r_squared import r_squared

def chebyshev(x, y, n):
    """
    Given the x-values, the y-values, and the degree of the polynomial, this
    function returns the weights of the Chebyshev regression model.

    Parameters:
        x : np.ndarray
            The x-values of the data.
        y : np.ndarray
            The y-values of the data.
        n : int
            The degree of the polynomial.  

    Returns:
        a : np.ndarray
            The weights of the Chebyshev regression model.
        y_pred : np.ndarray
            The predicted y-values of the Chebyshev regression model.
        r_squared : float
            The R-squared value of the Chebyshev regression model.
    """

    # calculating the total number of values
    m = len(x)

    # calculating the weights
    a = np.zeros(n + 1)
    for i in range(n + 1):
        a[i] = (2 / m) * np.sum(y * np.cos(i * np.arccos(x)))

    # calculate the predicted y-values
    y_pred = predict(x, a)

    # calculate the R-squared value
    r2 = r_squared(y, y_pred)

    # create a named tuple to return the results
    Result = namedtuple('Result', ['weights', 'y_pred', 'r_squared'])
    return Result(a, y_pred, r2)


def predict(x, a):
    """
    This function returns the predicted values of y.
    """
    # calculating the predicted values
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(a)):
            y_pred[i] += a[j] * np.cos(j * np.arccos(x[i]))

    return y_pred