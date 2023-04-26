import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar




def powell_conjugate_direction_method(func, x0, tol, max_iters):
    """
    Powell's conjugate direction method optimisation algorithm
    
    Parameters:
        func (function): the objective function to be optimised
        initial_directions (list): initial directions
        tol (float): tolerance value for termination
        max_iters (int): maximum number of iterations
        
    Returns:
        x (float): the optimised value
        E (list): the list of errors
        N (int): number of iterations
    """
    
    ## Error list
    E = []
    
    n = len(x0)
    f0 = func(*x0)
    directions = np.eye(n)
    iters = 0
    
    while iters < max_iters:
        iters += 1
        x_prev = x0.copy()
        
        # Minimise along each direction
        for i in range(n):
            res = minimize_scalar(lambda alpha: func(*(x0 + alpha * directions[i, :])))
            x0 = x0 + res.x * directions[i, :]
            E.append(abs(func(*x0)))
            
        # Update directions
        for i in range(n - 1):
            directions[i, :] = directions[i + 1, :]
        directions[-1, :] = x0 - x_prev
        
        ## Check for convergence
        if np.linalg.norm(x0 - x_prev) < tol:
            break
        
    return x0, E, len(E)
    
    
if __name__ == '__main__':
    func = lambda x, y: (x - 2) ** 2 + (y - 3) ** 2
    x0 = np.random.rand(2)
    tol = 1e-6
    max_iters = 1000
    x, E, N = powell_conjugate_direction_method(func, x0, tol, max_iters)
    print(f'x = {x}')
    print(f'f(x) = {func(*x)}')
    print(f'N = {N}')
    
    plt.plot(E)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

