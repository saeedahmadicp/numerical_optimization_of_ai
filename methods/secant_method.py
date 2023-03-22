
__all__ = ['secantMethod']

def secantMethod(f, a, b, tol, max_iter):
    """
    param: f: function to be evaluated
    param: a: lower bound of the interval
    param: b: upper bound of the interval
    param: tol: tolerance
    param: max_iter: maximum number of iterations
    
    return x: root of the function, E: list of errors, N: number of iterations
    
    Description: This function finds the root of a function f using the secant method
    """
    
    if f(a)*f(b) > 0:
        print("The function has the same sign at both ends of the interval")
        return None
    
    E = []
    N = 0
    while N < max_iter:
        x = b - f(b)*(b-a)/(f(b)-f(a))
        E.append(abs(f(x)))
        if abs(f(x)) <= tol:
            return x, E, N
        a = b
        b = x
        N += 1
    return x, E, N