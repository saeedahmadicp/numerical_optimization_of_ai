import math

from .elimination import eliminationStep

def goldenSearchMethod(f, a, b, N, tol=1e-6):
    """
    :param f: function to be evaluated
    :param a: lower bound of the interval
    :param b: upper bound of the interval
    :param N: number of iterations
    :return: x: root of the function, E: list of errors, N: number of iterations
    """
    E = []
    tai = (-1 + math.sqrt(5)) / 2
    x1 = a + (1-tai) * (b - a)
    x2 = a + tai * (b - a)
    for i in range(N):
        N -= 1
        
        ## if the function value on x1 is same as on x2, then adding a small value to x2 will make it different
        if f(x1) == f(x2):
            x2 += 1e-6
        
        ## apply elimination step
        a, b = eliminationStep(f, a, b, x1, x2)
        
        ## calculate new x1 and x2
        x1 = a + (1-tai) * (b - a)
        x2 = a + tai * (b - a)
        
        ## append error to the list
        E.append(abs(f((a + b) / 2)))
        
        ## check if the error is less than the tolerance
        if abs(b - a) < tol:
            return a, E, i
        
    return (a + b) / 2, E, N


if __name__ == "__main__":
    f = lambda x: abs(x - 0.3)
    a = 0
    b = 1
    N = 5
    x, E, N = goldenSearchMethod(f, a, b, N)
    print(x, E, N)