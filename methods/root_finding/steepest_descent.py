import numpy as np
import torch 

__all__ = ['steepest_descent']

def steepest_descent(func, variables, alpha, tol, max_iters):
    """
    Steepest descent method optimisation algorithm
    
    Parameters:
        func (function): the objective function to be optimised
        variables (list): list of variables
        alpha (float): learning rate
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
        x.data = x.data - alpha * x.grad
        x.grad.zero_()
        E.append(abs(func(*x).detach().numpy()))
        
        # check for convergence
        if torch.norm(x - x_prev) < tol:
            break
        
    return x.detach().numpy(), E, len(E), np.array(history)