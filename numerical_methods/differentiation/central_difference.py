import inspect

__all__ = ["central_difference"]

def central_difference(f, x, h, *args, **kwargs):
    """
    Central difference method for numerical differentiation.
    :param f: function
    :param x: point
    :param h: step size
    :param args: additional positional arguments for the function f
    :param kwargs: additional keyword arguments for the function f
    :return: derivative
    """
    num_args = len(inspect.signature(f).parameters)
    if num_args == 1:
        return (f(x+h) - f(x-h)) / (2*h)
    elif num_args == 2:
        return (f(x+h, *args, **kwargs) - f(x-h, *args, **kwargs)) / (2*h)
    else:
        raise ValueError("The function f must take either 1 or 2 arguments.")
    