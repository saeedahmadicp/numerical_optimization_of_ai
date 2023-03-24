from matplotlib import pyplot as plt

__all__ = ["plotRootFindingMethods"]


def plot_root_finding_methods(index, *args):
    """
    Description: This function plots the absolute errors vs iterations for each of the root finding methods.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor('white')
    fig.set_edgecolor('black')
    fig.set_linewidth(2)
    fig.set_frameon(True)

    for error_list, color, label in args:
        iters_list = [i for i in range(1, len(error_list) + 1)]
        ax.plot(iters_list, error_list, color=color, label=label)

    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("Absolute Errors", fontsize=15)
    ax.set_title("Absolute Errors vs Iterations", fontsize=20)
    ax.legend(title="Root finding methods", loc="upper right", fontsize=15)

    plt.savefig(f'root_finding_methods_{index}.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    # Test case
    import numpy as np
    from methods import bisection_method, newton_method, secant_method

    f = lambda x: x**3 - 2*x - 5
    f_prime = lambda x: 3*x**2 - 2

    x0 = 1
    a = 1
    b = 2
    tol = 1e-6
    max_iter = 100

    x_newton, E_newton, N_newton = newton_method.newton_method(f, f_prime, x0, tol, max_iter)
    x_bisection, E_bisection, N_bisection = bisection_method.bisection_method(f, a, b, tol, max_iter)
    x_secant, E_secant, N_secant = secant_method.secant_method(f, x0, tol, max_iter)

    plot_root_finding_methods(1, (E_newton, 'r', 'Newton-Raphson'), (E_bisection, 'b', 'Bisection'), (E_secant, 'g', 'Secant'))
    plot_root_finding_methods(2, (E_newton, 'r', 'Newton-Raphson'), (E_bisection, 'b', 'Bisection'), (E_secant, 'g', 'Secant'))

    print(f'Newton-Raphson method: x = {x_newton}, E = {E_newton}, N = {N_newton}')
    print(f'Bisection method: x = {x_bisection}, E = {E_bisection}, N = {N_bisection}')
    print(f'Secant method: x = {x_secant}, E = {E_secant}, N = {N_secant}')
