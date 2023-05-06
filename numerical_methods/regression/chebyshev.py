import numpy as np

def chebyshev_regression(x, y, n):
    """
    This function returns the coefficients of the Chebyshev regression model.
    """
    # Calculating the total number of values
    m = len(x)

    # Calculating the weights
    a = np.zeros(n + 1)
    for i in range(n + 1):
        a[i] = (2 / m) * np.sum(y * np.cos(i * np.arccos(x)))

    # Return the weights
    return a


def predict(x, a):
    """
    This function returns the predicted values of y.
    """
    # Calculating the predicted values
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(a)):
            y_pred[i] += a[j] * np.cos(j * np.arccos(x[i]))

    # Return the predicted values
    return y_pred


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
    a = chebyshev_regression(x, y, 3)

    # Calculating the predicted values
    y_pred = predict(x, a)

    # Printing the R-squared value
    print("R-squared:", r_squared(y, y_pred))

    # Plotting the results
    import matplotlib.pyplot as plt
    plt.scatter(x, y, color="blue")
    plt.plot(x, y_pred, color="red")
    plt.title("Chebyshev Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()