def backward_difference(f, x, h):
    """
    Backward difference method for numerical differentiation.
    :param f: function
    :param x: point
    :param h: step size
    :return: derivative
    """
    return (f(x) - f(x - h)) / h


if __name__ == "__main__":
    f = lambda x: x**2
    x = 2
    h = 0.01
    print(backward_difference(f, x, h))