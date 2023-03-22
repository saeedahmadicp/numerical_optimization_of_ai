import torch


__all__ = ['newtonMethod']


def newtonMethod(f, x0, tol, max_iter):
    """
    param: f: function to be evaluated
    param: x0: initial guess
    param: tol: tolerance
    param: max_iter: maximum number of iterations
    
    return x: root of the function, E: list of errors, N: number of iterations
    
    Description: This function finds the root of a function f using the newton method
    """
    x = torch.tensor(x0, requires_grad=True)
    E = []
    N = 0
    while N < max_iter:
        x = x - f(x)/ torch.autograd.grad(f(x), x)[0]
        E.append(abs(f(x).detach().numpy()))
        if abs(f(x)) <= tol:
            return x, E, N
        N += 1
    return x.detach().numpy() , E, N