# algorithms/convex/line_search.py

"""Line search methods for convex optimization."""

import numpy as np


def backtracking_line_search(f, grad_f, xk, pk, alpha_init=1.0, rho=0.5, c=1e-4):
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

    Returns:
        alpha   : float. A step size that satisfies the Armijo condition.
    """
    alpha = alpha_init
    # Armijo condition: f(xk + alpha * pk) <= f(xk) + c * alpha * grad_f(xk).T @ pk
    while f(xk + alpha * pk) > f(xk) + c * alpha * np.dot(grad_f(xk).T, pk):
        alpha *= rho  # Reduce step size by factor rho
    return alpha


# if __name__ == "__main__":
#     # Example usage
#     A = np.array([[3, 0.5], [0.5, 1]])
#     b = np.array([1, 2])

#     def f(x):
#         return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

#     def grad_f(x):
#         return A @ x - b

#     x0 = np.array([0.0, 0.0])
#     pk = -grad_f(x0)  # Steepest descent direction

#     alpha = backtracking_line_search(f, grad_f, x0, pk)
#     x_new = x0 + alpha * pk

#     print("Line Search Example:")
#     print(f"Initial point: {x0}")
#     print(f"Step size alpha: {alpha}")
#     print(f"New point: {x_new}")
#     print(f"Function value decreased from {f(x0):.6f} to {f(x_new):.6f}")
