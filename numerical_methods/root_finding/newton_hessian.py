import numpy as np
import torch

__all__ = ['newton_hessian']

def newton_hessian(func, variables, tol, max_iters):
    """
    Newton-Hessian method optimisation algorithm
    
    Parameters:
        func (function): the objective function to be optimised
        variables (list): list of variables
        tol (float): tolerance value for termination
        max_iters (int): maximum number of iterations
        
    Returns:
        x (float): the optimised value
        E (list): the list of errors
        N (int): number of iterations
    """
    
    # Error list
    E = []
    
    # history
    history = []
    
    x = np.random.rand(len(variables))
    x = torch.tensor(x, requires_grad=True)
    iters = 0
    
    while iters < max_iters:
        iters += 1
        history.append(x.detach().clone().numpy())
        x_prev = x.clone()
        f = func(*x)
        f.backward()
        
        hessian_func = lambda x: func(*x)
        hessian = torch.autograd.functional.hessian(hessian_func, x)
        x.data = x.data - torch.matmul(torch.inverse(hessian), x.grad)
        x.grad.zero_()
        E.append(abs(func(*x).detach().numpy()))
        
        # Check for convergence
        if torch.norm(x - x_prev) < tol:
            break
        
    return x.detach().numpy(), E, len(E), np.array(history)