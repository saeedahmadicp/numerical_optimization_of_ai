import random

__all__ = ['eliminationStep', 'eliminationSearch']


def eliminationStep(func, a, b , x1, x2, delta=1e-6):
    """
    :param func: function to be evaluated
    :param a: lower bound of the interval
    :param b: upper bound of the interval
    :param x1_ran: random number between a and b
    :param x2_ran: random number between a and b
    :return: x1, x2: the two points that will be eliminated
    """
    
    if func(x1) < func(x2):
        return a, x2
    
    elif func(x1) > func(x2):
        return x1, b 
    
    else:
        return x1, x2
    
  
"""
The problem with elimination method is that it is not guaranteed to converge to the root as it depends on the random numbers generated.
"""  
def eliminationSearchMethod(func, a, b, N, tol=1e-6):
    """
    :param func: function to be evaluated
    :param a: lower bound of the interval
    :param b: upper bound of the interval
    :param N: number of iterations
    :param tol: tolerance
    :return: x: root of the function, E: list of errors, N: number of iterations
    """
    E = []
    for i in range(N):
        
        ## generate two random numbers between a and b
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        
        ## apply elimination step
        a, b = eliminationStep(func, a, b, x1, x2)
        
        ## append error to the list
        E.append(abs(func((a+b)/2)))
        
        ## terminate if the error is less than the tolerance
        if abs(b-a) < tol:
            return (a+b)/2, E, i
        
        
        ## terminate if the function value on x1 is same as on x2
        if func(a) == func(b):
            return (a+b)/2, E, i
        
    return (a+b)/2, E, N


