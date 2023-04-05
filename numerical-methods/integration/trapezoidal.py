import numpy as np

def trapezoidal(f, a, b, n):
    """
    This function performs the trapezoidal rule.
    param: f: function to integrate
    param: a: lower bound
    param: b: upper bound
    param: n: number of subintervals
    return: approximation of the integral
    """

    # Calculate the step size (h is the width of each subinterval)
    h = (b - a) / n
    left = f(a) / 2
    right = f(b) / 2
    middle = np.sum(f(a + h * np.arange(1, n)))
    return h * (left + middle + right)



if __name__ == "__main__":
    # Test the function
    f = lambda x: np.exp(x)
    a = 1
    b = 2
    n = 4
    print(trapezoidal(f, a, b, n))