# algorithms/convex/quasi_newton.py

import numpy as np
from typing import Callable, Tuple, List, Optional
from line_search import backtracking_line_search


def bfgs_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) method.

    Maintains and updates an approximation to the inverse Hessian matrix.

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        x0: Initial point
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations

    Returns:
        Tuple containing:
        - Final iterate x
        - History of iterates
        - History of function values
    """
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial inverse Hessian approximation

    history = [x.copy()]
    f_history = [f(x)]

    for k in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break

        # Compute search direction
        p = -H @ grad

        # Line search
        alpha = backtracking_line_search(f, grad_f, x, p)

        # Update position and gradient
        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        # Compute differences
        s = x_new - x
        y = grad_new - grad

        # BFGS update
        rho = 1.0 / (y @ s)
        if rho > 0:  # Skip update if curvature condition fails
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (
                I - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        x = x_new
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


def lbfgs_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    m: int = 10,  # Number of corrections to store
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    L-BFGS (Limited-memory BFGS) method.

    Uses a limited memory version of BFGS to handle large-scale problems.

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        x0: Initial point
        m: Number of corrections to store
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations

    Returns:
        Tuple containing:
        - Final iterate x
        - History of iterates
        - History of function values
    """

    def two_loop_recursion(
        q: np.ndarray,
        s_list: List[np.ndarray],
        y_list: List[np.ndarray],
        rho_list: List[float],
    ) -> np.ndarray:
        """Two-loop recursion for computing H∇f."""
        alpha_list = []
        a = q.copy()

        # First loop
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            alpha = rho * s @ a
            alpha_list.append(alpha)
            a = a - alpha * y

        # Scaling
        if len(y_list) > 0:
            s = s_list[-1]
            y = y_list[-1]
            gamma = (s @ y) / (y @ y)
            r = gamma * a
        else:
            r = a

        # Second loop
        for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(alpha_list)):
            beta = rho * y @ r
            r = r + (alpha - beta) * s

        return r

    x = x0.copy()
    history = [x.copy()]
    f_history = [f(x)]

    s_list = []  # List of position differences
    y_list = []  # List of gradient differences
    rho_list = []  # List of rho values

    for k in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break

        # Compute search direction using L-BFGS two-loop recursion
        p = -two_loop_recursion(grad, s_list, y_list, rho_list)

        # Line search
        alpha = backtracking_line_search(f, grad_f, x, p)

        # Update position and gradient
        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        # Compute differences
        s = x_new - x
        y = grad_new - grad

        # Update lists
        rho = 1.0 / (y @ s)
        if rho > 0:  # Skip update if curvature condition fails
            if len(s_list) == m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)

        x = x_new
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


# if __name__ == "__main__":
#     # Example usage on Rosenbrock function
#     def rosenbrock(x):
#         return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

#     def rosenbrock_grad(x):
#         return np.array([
#             -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
#             200 * (x[1] - x[0]**2)
#         ])

#     # Initial point
#     x0 = np.array([-1.0, 1.0])

#     # Run both methods
#     print("BFGS Method:")
#     x_bfgs, hist_bfgs, f_hist_bfgs = bfgs_method(rosenbrock, rosenbrock_grad, x0)
#     print(f"Solution: {x_bfgs}")
#     print(f"Iterations: {len(hist_bfgs)-1}")
#     print(f"Final value: {f_hist_bfgs[-1]}\n")

#     print("L-BFGS Method:")
#     x_lbfgs, hist_lbfgs, f_hist_lbfgs = lbfgs_method(rosenbrock, rosenbrock_grad, x0)
#     print(f"Solution: {x_lbfgs}")
#     print(f"Iterations: {len(hist_lbfgs)-1}")
#     print(f"Final value: {f_hist_lbfgs[-1]}")
