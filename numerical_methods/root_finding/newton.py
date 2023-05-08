import torch

__all__ = ['newton']


def newton(f: callable, x0: float, tol: float, max_iter: int) -> tuple:
    """
    Find the root of the function f using the Newton-Raphson method.
    :param: f: function to be evaluated
    x0: The initial guess.
    tol: The tolerance.
    max_iter: The maximum number of iterations.
    :return: guess: root of the function, E: list of errors, N: number of iterations
    """
    guess = torch.tensor(x0, requires_grad=True)
    E = []
    N = 0

    while N < max_iter:
        derivative = torch.autograd.grad(f(guess), guess)[0]
        guess = guess - f(guess) / derivative
        E.append(abs(f(guess).detach().numpy()))
        
        if abs(f(guess)) <= tol:
            return guess, E, N
        
        N += 1

    return guess.detach().numpy(), E, N