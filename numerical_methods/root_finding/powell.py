import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


__all__ = ['powell_conjugate_direction_method']

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
    
    ## history
    history = [x0.copy()]
    
    while iters < max_iters:
        iters += 1
        x_prev = x0.copy()
        history.append(x_prev)
        
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
        
    return x0, E, len(E), np.array(history)
    
    
if __name__ == '__main__':
    func = lambda x, y: x**2 -2*x*y + 2*y**2 - 8*y + 16
    x0 = np.random.rand(2)
    tol = 1e-6
    max_iters = 1000
    x_min, E, N, history = powell_conjugate_direction_method(func, x0, tol, max_iters)
    print(f'x = {x_min}')
    print(f'f(x) = {func(*x_min)}')
    print(f'N = {N}')
    
    ## Define the range of the contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    ### Plot the contour
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 10), cmap='jet')
    plt.colorbar(label='f(x, y)')
    
    
    ## plot the movement of the best point during the iterations
    plt.plot(history[:, 0], history[:, 1], 'k.-')
    plt.plot(x_min[0], x_min[1], 'r*', label='Optimum')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Powell Method, function = x**2 -2*x*y + 2*y**2 - 8*y + 16')
    plt.show()

