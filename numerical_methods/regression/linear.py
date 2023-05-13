import numpy as np
from collections import namedtuple
from .r_squared import r_squared

def linear(x, y):
    """
    Given a set of data points (x, y), this function returns the weights a and b
    of the linear regression model y = ax + b.

    Parameters:
        x : np.ndarray
            The x-coordinates of the data points.
        y : np.ndarray
            The y-coordinates of the data points.

    Returns:
        a : float
            The slope of the linear regression model.
        b : float
            The intercept of the linear regression model.
        y_pred : array_like
            The predicted y-values of the linear regression model.
        r_squared : float
            The R-squared value of the linear regression model.
    """

    # checking the input arrays
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be numpy arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    # calculating the total number of values
    n = len(x)

    # calculate the weights a and b
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    b = (np.sum(y) - a * np.sum(x)) / n

    # calculate the predicted y-values
    y_pred = a * x + b

    # calculate the R-squared value
    r2 = r_squared(y, y_pred)

    # create a named tuple to return the results
    Result = namedtuple('Result', ['slope', 'intercept', 'y_pred', 'r_squared'])
    return Result(a, b, y_pred, r2)