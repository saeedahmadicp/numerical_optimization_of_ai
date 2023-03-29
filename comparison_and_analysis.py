
import numpy as np

from methods import fibonacciSearchMethod, goldenSearchMethod, eliminationSearchMethod
from utils import findInitialInterval, plotRootFindingMethods
from functions import func1, func2, func3, func4, func5


def main():
    """
    Main function to calculate roots of given functions 
    using different methods and plot the error values
    """
    functions_list = [func1, func2, func3, func4, func5]
    

    ## initial guess and initial step size
    x0 = np.random.rand(1)[0] 
    d0 = 0.1
    
    ## root finding methods
    for index, func in enumerate(functions_list):
        (a,b) = findInitialInterval(func, x0, d0)
        tol = 1e-6
        N = 10
        
        ## call methods, EM = elimination method, FM = fibonacci method, GSM = golden section method
        EM_root, EM_E, _ = eliminationSearchMethod(func, a, b, N, tol)
        FM_root, FM_E, _ = fibonacciSearchMethod(func, a, b, N, tol)
        GSM_root, GSM_E, _ = goldenSearchMethod(func, a, b, N, tol)
        

        ## print the guessing roots
        print(f"Elimination method: function index: {index},  root = {EM_root}, solution: {func(EM_root)}")
        print(f"Fibonacci method: function index: {index},  root = {FM_root}, solution: {func(FM_root)}")
        print(f"Golden section method: function index: {index},  root = {GSM_root}, solution: {func(GSM_root)}")
        
        ## build data dictionary for plotting
        data = {
            "Elimination method": {"errors": EM_E, "color": "blue"},
            "Fibonacci method": {"errors": FM_E, "color": "red"},
            "Golden section method": {"errors": GSM_E, "color": "green"}
        }
        
        ## plot the errors
        plotRootFindingMethods(index, data)
    
if __name__ == "__main__":
    main()
        
        
           
