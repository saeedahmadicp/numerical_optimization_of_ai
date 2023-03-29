__all__ = ['bisectionMethod']

def bisectionMethod(f: callable, a: float, b: float, tol: float, max_iter: int) -> tuple:
    """
    This function finds the root of a function f using the bisection method
    :param: f: function to be evaluated
    :param: a: lower bound of the interval
    :param: b: upper bound of the interval
    :param: tol: tolerance
    :param: max_iter: maximum number of iterations
    :return: x: root of the function, E: list of errors, N: number of iterations
    """
    # raise ValueError if initial conditions do not satisfy the bisection method requirement
    if f(a) * f(b) >= 0:
        raise ValueError("The function has the same sign at both ends of the interval")
    
    # initialize error list and iteration counter 
    E = [] 
    N = 0 
    
    # iterate until max iterations or tolerance is met
    while N < max_iter: 
        
        x = (a + b) / 2 
        E.append(abs(f(x)))
        
        # check if tolerance is met
        if E[-1] <= tol: 
            return x, E, N
        
        ## apply elimination step
        elif f(a) * f(x) < 0: 
            b = x
        else:
            a = x
            
        N += 1
    return x, E, N
