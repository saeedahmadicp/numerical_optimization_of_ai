import torch
from torch.autograd import Variable

__all__ = ['bfgs', 'dfp', 'sr1']

def bfgs(objective_func, x_init, epsilon=1e-8, max_iter=100000):
    x = Variable(torch.FloatTensor(x_init), requires_grad=True) ## convert the input to a torch variable
    B = torch.eye(len(x_init)) ## initialize B as an identity matrix
    f_x = objective_func(x) ## compute the objective function at x

    for i in range(max_iter):
        grad_x = torch.autograd.grad(f_x, x, allow_unused=True)[0] ## compute the gradient of f_x at x
        if grad_x is None:  ## if grad_x is None, set it to zero
            grad_x = torch.zeros_like(x) ## set grad_x to zero
        p = -torch.matmul(B, grad_x) ## compute the search direction p

        f_x_p = objective_func(x + p) ## compute the objective function at x + p

        alpha = 1.0 ## initialize alpha
        c = 0.5     ## initialize c
        rho = 0.9  ## initialize rho
        while f_x_p > f_x + c * alpha * torch.matmul(grad_x, p): ## if the Armijo condition is not satisfied
            alpha *= rho ## update alpha
            f_x_p = objective_func(x + alpha * p) ## compute the objective function at x + alpha * p

        x_new = x + alpha * p ## update x
        s = x_new - x ## compute s
        y = torch.autograd.grad(objective_func(x_new), x_new)[0] - grad_x ## compute y

        rho = 1 / torch.matmul(y, s) ## compute denominator of BFGS update
        A = torch.eye(len(x_init)) - rho * torch.matmul(s.view(-1, 1), y.view(1, -1))  ## compute the first term of BFGS update
        B = torch.matmul(A, torch.matmul(B, A.t())) + rho * torch.matmul(s.view(-1, 1), s.view(1, -1)) ## compute the BFGS update

        x = x_new
        f_x = f_x_p

        if torch.norm(grad_x) < epsilon:
            break

    return x.data.numpy()



def dfp(objective_func, x_init, epsilon=1e-6, max_iter=100):
    x = Variable(torch.FloatTensor(x_init), requires_grad=True) ## convert the input to a torch variable
    B = torch.eye(len(x_init)) ## initialize B as an identity matrix
    f_x = objective_func(x)  ## compute the objective function at x

    for i in range(max_iter):   ## repeat until the maximum number of iterations is reached
        grad_x = torch.autograd.grad(f_x, x, allow_unused=True)[0] ## compute the gradient of f_x at x
        if grad_x is None: ## if grad_x is None, set it to zero
            grad_x = torch.zeros_like(x)  # 그래디언트가 None인 경우 0으로 설정
        p = -torch.matmul(B, grad_x) ## compute the search direction p

        f_x_p = objective_func(x + p) ## compute the objective function at x + p    

        alpha = 1.0
        c = 0.5
        rho = 0.9
        while f_x_p > f_x + c * alpha * torch.matmul(grad_x, p): ## if the Armijo condition is not satisfied
            alpha *= rho ## update alpha
            f_x_p = objective_func(x + alpha * p)

        x_new = x + alpha * p
        s = x_new - x
        y = torch.autograd.grad(objective_func(x_new), x_new)[0] - grad_x ## compute y

        rho = 1 / torch.matmul(y, s) 
        A = torch.eye(len(x_init)) - rho * torch.matmul(s.view(-1, 1), y.view(1, -1)) ## compute the first term of DFP update
        B = torch.matmul(A.t(), torch.matmul(B, A)) + rho * torch.matmul(s.view(-1, 1), s.view(1, -1)) ## compute the DFP update

        x = x_new
        f_x = f_x_p

        if torch.norm(grad_x) < epsilon:
            break

    return x.data.numpy()

def sr1(objective_func, x_init, epsilon=1e-6, max_iter=100):
    x = Variable(torch.FloatTensor(x_init), requires_grad=True) ## convert the input to a torch variable
    B = torch.eye(len(x_init)) ## initialize B as an identity matrix
    f_x = objective_func(x) ## compute the objective function at x

    for i in range(max_iter):   ## repeat until max_iter
        grad_x = torch.autograd.grad(f_x, x, allow_unused=True)[0] ## compute the gradient of f_x at x
        if grad_x is None:      ## if grad_x is None, set it to zero
            grad_x = torch.zeros_like(x)  # 그래디언트가 None인 경우 0으로 설정
        p = -torch.matmul(B, grad_x) ## compute the search direction p

        f_x_p = objective_func(x + p) ## compute the objective function at x + p

        alpha = 1.0 
        c = 0.5
        rho = 0.9
        while f_x_p > f_x + c * alpha * torch.matmul(grad_x, p): #Armijo's Condition
            alpha *= rho  ## update alpha
            f_x_p = objective_func(x + alpha * p) ## compute the objective function at x + alpha * p

        x_new = x + alpha * p  ## update x 
        s = x_new - x  ## compute s
        y = torch.autograd.grad(objective_func(x_new), x_new, allow_unused=True)[0] - grad_x ## compute y

        y_minus_Bs = y - torch.matmul(B, s) ## compute y - B * s
        rho = 1 / torch.matmul(y_minus_Bs, s) ## compute denominator of SR1 update

        B = B + torch.matmul(y_minus_Bs.view(-1, 1), y_minus_Bs.view(1, -1)) * rho ## compute the SR1 update

        x = x_new
        f_x = f_x_p

        if torch.norm(grad_x) < epsilon:
            break

    return x.data.numpy()