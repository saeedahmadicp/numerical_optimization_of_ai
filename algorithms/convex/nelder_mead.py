# algorithms/convex/nelder_mead.py

import numpy as np

__all__ = ["nelder_mead"]


def nelder_mead(func, noOfPoints, iters, tol):
    """
    Nelder-Mead method optimisation algorithm

    Parameters:
        func (function): the objective function to be optimised
        noOfPoints (int): number of points in the simplex
        iters (int): number of iterations
        tol (float): tolerance value for termination

    Returns:
        x (float): the optimised value
        E (list): the list of errors
        N (int): number of iterations
    """

    # generate the initial simplex
    n = noOfPoints - 1
    simplex = np.random.rand(noOfPoints, n)
    x_best = simplex[0, :]
    f_best = func(*x_best)

    # Define the reflection, expansion, contraction and shrinkage coefficients
    alpha = 1
    gamma = 2
    rho = 0.5
    sigma = 0.5

    # Error list
    Errors = []
    x_history = [x_best.copy()]

    # Optimization loop
    for i in range(iters):
        f_values = np.array([func(*x) for x in simplex])
        order = np.argsort(f_values)
        simplex = simplex[order, :]
        f_values = f_values[order]

        x_centroid = np.mean(simplex[:-1, :], axis=0)
        x_reflection = x_centroid + alpha * (x_centroid - simplex[-1, :])
        f_reflection = func(*x_reflection)

        if f_values[0] <= f_reflection < f_values[-2]:
            simplex[-1, :] = x_reflection
        elif f_reflection < f_values[0]:
            x_expansion = x_centroid + gamma * (x_reflection - x_centroid)
            f_expansion = func(*x_expansion)
            if f_expansion < f_reflection:
                simplex[-1, :] = x_expansion
            else:
                simplex[-1, :] = x_reflection
        elif f_reflection >= f_values[-2]:
            if f_reflection < f_values[-1]:
                x_contraction = x_centroid + rho * (x_reflection - x_centroid)
                f_contraction = func(*x_contraction)
            else:
                x_contraction = x_centroid + rho * (simplex[-1, :] - x_centroid)
                f_contraction = func(*x_contraction)

            if f_contraction < np.min(f_values):
                simplex[-1, :] = x_contraction
            else:
                for i in range(1, noOfPoints):
                    simplex[i, :] = sigma * (simplex[i, :] + simplex[0, :])

        x_history.append(simplex[0, :].copy())

        # Termination condition
        if np.std(f_values) < tol:
            break

        # Update the best value
        if f_values[0] < f_best:
            x_best = simplex[0, :]
            f_best = f_values[0]

        # Update the error list
        Errors.append(f_best)

    return x_best, Errors, len(Errors), np.array(x_history)
