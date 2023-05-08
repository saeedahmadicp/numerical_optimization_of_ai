import inspect

__all__ = ["forward_difference"]

def forward_difference(f, x, h, *args, **kwargs):
    """
    Forward difference method for numerical differentiation.
    :param f: function
    :param x: point
    :param h: step size
    :param args: additional positional arguments for the function f
    :param kwargs: additional keyword arguments for the function f
    :return: derivative
    """
    num_args = len(inspect.signature(f).parameters)
    if num_args == 1:
        return (f(x + h) - f(x)) / h
    elif num_args == 2:
        return (f(x + h, *args, **kwargs) - f(x, *args, **kwargs)) / h
    else:
        raise ValueError("The function f must take either 1 or 2 arguments.")
