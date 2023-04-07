def central_difference(f, x, h):
    """
    Central difference method for numerical differentiation.
    :param f: function
    :param x: point
    :param h: step size
    :return: derivative
    """
    return (f(x+h) - f(x-h)) / (2*h)


if __name__ == "__main__":
    f = lambda x: x**2
    x = 2
    h = 0.01
    print(central_difference(f, x, h))