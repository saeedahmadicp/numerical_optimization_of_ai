import numpy as np

__all__ = ["findInitialInterval"]

def findInitialInterval(f, x0, d0):
    """
    :param f: function to be evaluated
    :param x0: initial guess
    :param d0: initial step size
    :return: [a, b]: initial interval
    Description: find initial interval for unimodal functions using the initial guess and initial step size 
    """
    f_minus = f(x0 - d0)
    f_0 = f(x0)
    f_plus = f(x0 + d0)
    
    if f_minus >= f_0 >= f_plus:
        d = d0
        x_minus_1 = x0 - d0
        x1 = x0 + d0
    elif f_minus <= f_0 <= f_plus:
        d = -d0
        x_minus_1 = x0 + d0
        x1 = x0 - d0
    elif f_minus >= f_0 <= f_plus:
        return [x0 - d0, x0 + d0]
    else:
        raise ValueError("Cannot determine the direction of minimum")
    
    xk_minus_1 = x_minus_1
    xk = x0
    xk_plus_1 = x1
    
    while True:
        xk_plus_2 = xk + 2 * d
        
        if f(xk_plus_2) >= f(xk) and d > 0:
            return [xk_minus_1, xk_plus_2]
        elif f(xk_plus_2) >= f(xk) and d < 0:
            return [xk_plus_2, xk_minus_1]
        
        xk_minus_1 = xk
        xk = xk_plus_1
        xk_plus_1 = xk_plus_2


if __name__ == "__main__":
    f = lambda x: x**2
    x0 = np.random.rand(1)[0]
    d0 = 0.1
    print(findInitialInterval(f, x0, d0))