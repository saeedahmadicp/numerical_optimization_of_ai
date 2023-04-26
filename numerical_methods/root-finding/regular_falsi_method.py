__all__ = ['regularFalsiMethod']


def regularFalsiMethod(f: callable, a: float, b: float, tol: float, max_iter: int) -> tuple:
    """
    This function finds the root of a function f using the regular falsi method.
    :param: f: function to be evaluated
    :param: a: lower bound of the interval
    :param: b: upper bound of the interval
    :param: tol: tolerance
    :param: max_iter: maximum number of iterations
    :return: x: root of the function, E: list of errors, N: number of iterations
    """    
    if f(a)*f(b) > 0:
        raise ValueError("The function must have opposite signs at the interval endpoints.")
    
    E = []
    N = 0

    while N < max_iter:
        x = b - f(b)*(b-a)/(f(b)-f(a))
        E.append(abs(f(x)))

        if E[-1] <= tol:
            return x, E, N
        elif f(a)*f(x) < 0:
            b = x
        else:
            a = x
        N += 1
        
    return x, E, N