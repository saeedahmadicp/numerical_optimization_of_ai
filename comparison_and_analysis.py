from matplotlib import pyplot as plt
import numpy as np
import torch

from methods import bisectionMethod, newtonMethod, regularFalsiMethod, secantMethod, plotRootFindingMethods

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def func1(x):
    return x**2 - 2

def func2(x):
    return x**3 - 2*x - 5

def func3(x, ):
    return np.exp(x) - 2*x - 1

def func4(x):
    return np.sin(x) - x/2

def func5(x):
    return np.cos(x) - x


## functions for newton method (using torch)
def func1_torch(x):
    return x**2 - 2

def func2_torch(x):
    return x**3 - 2*x - 5

def func3_torch(x):
    return torch.exp(x) - 2*x - 1

def func4_torch(x):
    return torch.sin(x) - x/2

def func5_torch(x):
    return torch.cos(x) - x


## finding interval for each function
def find_a_b(f):
    """
    param: f: function to be evaluated
    return a: function has positive value at a, b: function has negative value at b
    """
    ##generate randomly a and b real values betwee -5, 5
    a = np.random.uniform(-10, 0)
    b = np.random.uniform(0, 10) 
    
    while f(a)*f(b) > 0:
        a = np.random.uniform(-10, 0)
        b = np.random.uniform(0, 10) 
    return (a, b) if a < b else (b, a)
    
    

def main():
    functions_list = [func1, func2, func3, func4, func5]
    functions_list_torch = [func1_torch, func2_torch, func3_torch, func4_torch, func5_torch]
    
    for index, func, func_torch in zip(range(1, 6), functions_list, functions_list_torch):
        (a,b) = find_a_b(func)
        tol = 1e-6
        max_iter = 100
        
        bisection_x, bisection_E, _ = bisectionMethod(func, a, b, tol, max_iter)
        newton_x, newton_E, _ = newtonMethod(func_torch, a, tol, max_iter)
        regular_falsi_x, regular_falsi_E, _ = regularFalsiMethod(func, a, b, tol, max_iter)
        secant_x, secant_E, _ = secantMethod(func, a, b, tol, max_iter)
        
        print(f"Root of function {index+1} using Bisection Method: {bisection_x}")
        print(f"Root of function {index+1} using Newton Raphson Method: {newton_x}")
        print(f"Root of function {index+1} using Regular Falsi Method: {regular_falsi_x}")
        print(f"Root of function {index+1} using Secant Method: {secant_x}")
        
        plotRootFindingMethods(index,
                                bisection_E, "blue",
                                newton_E, "red",
                                regular_falsi_E, "green",
                                secant_E, "orange")
    
if __name__ == "__main__":
    main()
        
        
           

   