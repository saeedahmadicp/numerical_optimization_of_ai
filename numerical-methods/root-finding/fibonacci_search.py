from .elimination_method import eliminationStep

__all__ = ['fibonacciGenerator', 'fibonacciSearchMethod']

def fibonacciGenerator(n):
    """
    Generates the Fibonacci sequence up to n
    :param n: the number of elements in the sequence
    :return: the Fibonacci sequence
    """
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


def fibonacciSearchMethod(f, a, b, N, tol=1e-6):
    """
    :param f: function to be evaluated
    :param a: lower bound of the interval
    :param b: upper bound of the interval
    :param N: number of iterations
    :return: x: root of the function, E: list of errors, N: number of iterations
    """
    E = []
    fib = fibonacciGenerator(N + 1)
    x1 = a + fib[N - 2] / fib[N] * (b - a)
    x2 = b - fib[N - 2] / fib[N] * (b - a)
    for i in range(N):
        N -= 1
        
        ## apply elimination step
        a, b = eliminationStep(f, a, b, x1, x2)
        
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


if __name__ == "__main__":
    f = lambda x: abs(x - 0.3)
    a = 0
    b = 1
    N = 5
    x, E, N = fibonacciSearchMethod(f, a, b, N)
    print(x, E, N)