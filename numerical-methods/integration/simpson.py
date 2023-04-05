import numpy as np

def simpson(f: callable, a: float, b: float, n: int) -> float:
    """
    This function performs the Simpson's rule.
    param: f: function to integrate
    param: a: lower bound
    param: b: upper bound
    param: n: number of subintervals
    return: approximation of the integral
    """

    # Calculate the step size (h is the width of each subinterval)
    h = (b - a) / n
    left = f(a)
    right = f(b)

    # Calculate the sum of the odd and even terms
    odd = np.sum(f(a + h * np.arange(1, n, 2)))
    even = np.sum(f(a + h * np.arange(2, n, 2)))

    return h / 3 * (left + right + 4 * odd + 2 * even)


if __name__ == "__main__":
    # Test the function
    f = lambda x: np.exp(x)
    a = 1
    b = 2
    n = 4
    print(simpson(f, a, b, n))