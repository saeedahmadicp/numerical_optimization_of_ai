import numpy as np

def r_squared(y, y_pred):
    """
    Given the actual y-values and the predicted y-values, this function returns
    the R-squared value of the linear regression model.

    Parameters:
        y : np.ndarray
            The actual y-values.

        y_pred : np.ndarray
            The predicted y-values.

    Returns:
        r2 : float
            The R-squared value of the linear regression model.
    """

    # checking the input arrays
    if not isinstance(y, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y and y_pred must be numpy arrays")
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred must have the same length")

    # calculating the mean of y
    y_mean = np.mean(y)

    # calculating the total sum of squares
    ss_tot = np.sum((y - y_mean) ** 2)

    # calculating the residual sum of squares
    ss_res = np.sum((y - y_pred) ** 2)

    # calculating the R-squared value
    r2 = 1 - (ss_res / ss_tot)
    return r2