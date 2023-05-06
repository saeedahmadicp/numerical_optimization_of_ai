import numpy as np
import torch 
import matplotlib.pyplot as plt


__all__ = ['steepest_descent_method']

def steepest_descent_method(func, variables, alpha, tol, max_iters):
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
    
    ## Error list
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
        
        ## Check for convergence
        if torch.norm(x - x_prev) < tol:
            break
        
    return x.detach().numpy(), E, len(E), np.array(history)


if __name__ == '__main__':
    func = lambda x, y: (x) ** 2 + (y) ** 2
    variables = ['x', 'y']
    alpha = 0.1
    tol = 1e-6
    max_iters = 1000
    x_min, E, N, history = steepest_descent_method(func, variables, alpha, tol, max_iters)
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
    plt.title('Steepest Descent Method, function = x^2 + y^2')
    plt.show()