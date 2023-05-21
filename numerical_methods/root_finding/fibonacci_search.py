from .elimination import elim_step

__all__ = ['fib_generator', 'fibonacci_search']

def fib_generator(n):
    """
    Generates the Fibonacci sequence up to n
    :param n: the number of elements in the sequence
    :return: the Fibonacci sequence
    """
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


def fibonacci_search(f, a, b, N, tol=1e-6):
    """
    :param f: function to be evaluated
    :param a: lower bound of the interval
    :param b: upper bound of the interval
    :param N: number of iterations
    :return: x: root of the function, E: list of errors, N: number of iterations
    """
    E = []
    fib = fib_generator(N + 1)
    x1 = a + fib[N - 2] / fib[N] * (b - a)
    x2 = b - fib[N - 2] / fib[N] * (b - a)
    for i in range(N):
        N -= 1
        
        ## apply elimination step
        a, b = elim_step(f, a, b, x1, x2)
        
        ## terminate if the function value on x1 is same as on x2
        if f(a) == f(b):
            return (a + b) / 2, E, N
        
        
        ## calculate new x1 and x2
        x1 = a + fib[N -  2] / fib[N ] * (b - a)
        x2 = b - fib[N -  2] / fib[N] * (b - a)
        
        ## append error to the list
        E.append(abs(f((a + b) / 2)))
        
        
        ## check if the error is less than the tolerance
        if abs(b - a) < tol:
            return a, E, i
        
    return (a + b) / 2, E, N