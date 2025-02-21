# algorithms/convex/line_search.py

"""Line search methods for convex optimization.

# TODO: Implement the following methods:
- Backtracking line search
- Wolfe-Powell line search
- Goldstein-Armijo line search
- Damped line search
- Interpolation line search

THIS FILE IS NOT COMPLETE!
"""

import numpy as np


def backtracking_line_search(
    f, grad_f, xk, pk, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=100, alpha_min=1e-16
):
    """
    Perform a backtracking line search to choose a step size alpha along the descent direction pk.

    Parameters:
        f       : Callable. The objective function f(x).
        grad_f  : Callable. The gradient of f, which takes x as input.
        xk      : np.array. The current iterate.
        pk      : np.array. The descent direction (should satisfy grad_f(xk).T @ pk < 0).
        alpha_init: float. Initial step size (default is 1.0).
        rho     : float. Contraction factor for reducing step size (default is 0.5).
        c       : float. Armijo condition constant (default is 1e-4).
        max_iter: int. Maximum number of backtracking iterations (default is 100).
        alpha_min: float. Minimum step size (default is 1e-16).

    Returns:
        alpha   : float. A step size that satisfies the Armijo condition.
    """
    alpha = alpha_init

    for _ in range(max_iter):
        if alpha < alpha_min:
            break
        if f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(grad_f(xk).T, pk):
            break
        alpha *= rho

    return alpha
