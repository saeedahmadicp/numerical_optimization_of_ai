from matplotlib import pyplot as plt

__all__ = ["plotRootFindingMethods"]

def plotRootFindingMethods(index,
                            bisection_errors, bisection_color,
                            newton_errors, newton_color,
                            regular_falsi_errors, regular_falsi_color,
                            secant_errors, secant_color):
    """
    Description: This function plots the absolute errors vs iterations for each of the root finding methods
    """
    
    ## generating iterations list for each method
    bisections_iters_list = [i for i in range(1, len(bisection_errors)+1)]
    newton_iters_list = [i for i in range(1, len(newton_errors)+1)]
    regular_falsi_iters_list = [i for i in range(1, len(regular_falsi_errors)+1)]
    secant_iters_list = [i for i in range(1, len(secant_errors)+1)]
    
    
    
    Figure = plt.figure(figsize=(10, 10))
    Figure.set_facecolor("white")
    Figure.set_edgecolor("black")
    Figure.set_linewidth(2)
    Figure.set_frameon(True)
    Figure.set_dpi(100)
    
    
    plt.plot(bisections_iters_list, bisection_errors, color=bisection_color, label="Bisection Method") ## Bisection Method
    plt.plot(newton_iters_list, newton_errors, color=newton_color, label="Newton Raphson Method") ## Newton Raphson Method
    plt.plot(secant_iters_list, secant_errors, color=secant_color, label="Secant Method") ## Secant Method
    plt.plot(regular_falsi_iters_list, regular_falsi_errors, color=regular_falsi_color, label="Regular Falsi Method") ## Regular Falsi Method
    
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Absolute Errors", fontsize=15)
    plt.title("Absolute Errors vs Iterations", fontsize=20)
    plt.legend(title="Root finding methods", loc="upper right", fontsize=15)
    
    
    plt.savefig(f'root_finding_methods_{index}.png')
    plt.show()