import numpy as np

def linear_regression(x, y):
    """
    This function returns the coefficients of the linear regression model.
    """
    # Calculating the total number of values
    n = len(x)

    # Calculate the weights a and b
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    b = (np.sum(y) - a * np.sum(x)) / n

    # Return the weights
    return a, b


def predict(x, a, b):
    """
    This function returns the predicted values of y.
    """
    # Return the predicted values
    return a * x + b


def r_squared(y, y_pred):
    """
    This function returns the R-squared value.
    """
    # Calculating the mean of y
    y_mean = np.mean(y)

    # Calculating the total sum of squares
    ss_tot = np.sum((y - y_mean) ** 2)

    # Calculating the residual sum of squares
    ss_res = np.sum((y - y_pred) ** 2)

    # Calculating the R-squared value
    r2 = 1 - (ss_res / ss_tot)

    # Return the R-squared value
    return r2


if __name__ == "__main__":
    """
    This is the main function.
    """
    # Defining the data
    x = np.array([-1, -0.6, -0.2, 0.2, 0.6, 1])
    f = lambda x: np.exp(x)
    y = f(x)

    # Calculating the weights
    a, b = linear_regression(x, y)

    # Calculating the predicted values
    y_pred = predict(x, a, b)

    # Calculating the R-squared value
    r2 = r_squared(y, y_pred)

    # Printing the results
    print("The weights are: a = {0} and b = {1}".format(a, b))
    print("The predicted values are: {0}".format(y_pred))
    print("The R-squared value is: {0}".format(r2))