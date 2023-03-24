import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from methods import bisectionMethod, newtonMethod, regularFalsiMethod, secantMethod, plotRootFindingMethods


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def func1(x):
    return x ** 2 - 2

def func2(x):
    return x ** 3 - 2 * x - 5

def func3(x):
    return np.exp(x) - 2 * x - 1

def func4(x):
    return np.sin(x) - x / 2

def func5(x):
    return np.cos(x) - x


# functions for Newton's Method using torch
def func1_torch(x):
    return x ** 2 - 2

def func2_torch(x):
    return x ** 3 - 2 * x - 5

def func3_torch(x):
    return torch.exp(x) - 2 * x - 1

def func4_torch(x):
    return torch.sin(x) - x / 2

def func5_torch(x):
    return torch.cos(x) - x


def find_a_b(f):
    """
    Find the interval (a,b) for a given function f 
    where f(a) is positive and f(b) is negative
    """
    a = np.random.uniform(-10, 0)
    b = np.random.uniform(0, 10)
    
    while f(a) * f(b) > 0:
        a = np.random.uniform(-10, 0)
        b = np.random.uniform(0, 10)
        
    return (a, b) if a < b else (b, a)


def main():
    """
    Main function to calculate roots of given functions 
    using different methods and plot the error values
    """
    functions_list = [func1, func2, func3, func4, func5]
    functions_list_torch = [func1_torch, func2_torch, func3_torch, func4_torch, func5_torch]
    
    for index, func, func_torch in zip(range(1, 6), functions_list, functions_list_torch):
        (a, b) = find_a_b(func)
        tol = 1e-6
        max_iter = 100
        
        bisection_x, bisection_E, _ = bisectionMethod(func, a, b, tol, max_iter)
        newton_x, newton_E, _ = newtonMethod(func_torch, a, tol, max_iter)
        regular_falsi_x, regular_falsi_E, _ = regularFalsiMethod(func, a, b, tol, max_iter)
        secant_x, secant_E, _ = secantMethod(func, a, b, tol, max_iter)
        
        print(f"\nRoot of function {index} using Bisection Method: {bisection_x}")
        print(f"Root of function {index} using Newton Raphson Method: {newton_x}")
        print(f"Root of function {index} using Regular Falsi Method: {regular_falsi_x}")
        print(f"Root of function {index} using Secant Method: {secant_x}")
        
        plotRootFindingMethods(index,
                                bisection_E, "blue",
                                newton_E, "red",
                                regular_falsi_E, "green",
                                secant_E, "orange")
        
    plt.show()

if __name__ == "__main__":
    main()