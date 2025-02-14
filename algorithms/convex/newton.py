# algorithms/convex/newton.py

"""Newton's method for unconstrained optimization."""

import numpy as np
from typing import Callable, Tuple, List
from line_search import backtracking_line_search


def newton_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    use_line_search: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Newton's method for unconstrained optimization.

    The method iteratively updates:
        x_{k+1} = x_k - α_k [∇²f(x_k)]^{-1} ∇f(x_k)
    where α_k is determined by line search if enabled.

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        hess_f: Hessian function ∇²f(x)
        x0: Initial point
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations
        use_line_search: Whether to use line search for step size

    Returns:
        Tuple containing:
        - Final iterate x
        - History of iterates
        - History of function values
    """
    x = x0.copy()
    history = [x.copy()]
    f_history = [f(x)]

    for k in range(max_iter):
        # Compute gradient and Hessian
        grad = grad_f(x)
        hess = hess_f(x)

        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        try:
            # Compute Newton direction: -H^{-1}g
            newton_dir = -np.linalg.solve(hess, grad)

            # Ensure descent direction
            if np.dot(grad, newton_dir) > 0:
                newton_dir = -grad  # Fall back to steepest descent

            # Determine step size
            if use_line_search:
                alpha = backtracking_line_search(f, grad_f, x, newton_dir)
            else:
                alpha = 1.0

            # Update iterate
            x = x + alpha * newton_dir

        except np.linalg.LinAlgError:
            # If Hessian is singular, fall back to gradient descent
            newton_dir = -grad
            alpha = backtracking_line_search(f, grad_f, x, newton_dir)
            x = x + alpha * newton_dir

        # Record history
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


def damped_newton_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    beta: float = 0.5,  # Damping factor
    min_step: float = 1e-10,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Damped Newton's method for improved robustness.

    Uses Hessian modification when necessary:
        H_k = H_k + βI until H_k is sufficiently positive definite

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        hess_f: Hessian function ∇²f(x)
        x0: Initial point
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations
        beta: Initial damping factor
        min_step: Minimum allowed step size

    Returns:
        Tuple containing:
        - Final iterate x
        - History of iterates
        - History of function values
    """
    x = x0.copy()
    history = [x.copy()]
    f_history = [f(x)]
    n = len(x0)

    for k in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Try to compute Newton direction with damping
        damping = 0.0
        while True:
            try:
                # Add damping to Hessian
                H = hess + damping * np.eye(n)
                # Attempt to solve the system
                newton_dir = -np.linalg.solve(H, grad)

                # Check if descent direction
                if np.dot(grad, newton_dir) < 0:
                    break

                damping = max(2.0 * damping, beta)

            except np.linalg.LinAlgError:
                damping = max(2.0 * damping, beta)

            # Prevent infinite loop
            if damping > 1e6:
                newton_dir = -grad  # Fall back to steepest descent
                break

        # Line search
        alpha = backtracking_line_search(f, grad_f, x, newton_dir)

        # Update iterate
        step = alpha * newton_dir
        if np.linalg.norm(step) < min_step:
            break

        x = x + step

        # Record history
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


# if __name__ == "__main__":
#     # Example usage on Rosenbrock function
#     def rosenbrock(x):
#         return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

#     def rosenbrock_grad(x):
#         return np.array(
#             [
#                 -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
#                 200 * (x[1] - x[0] ** 2),
#             ]
#         )

#     def rosenbrock_hess(x):
#         return np.array(
#             [[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]], [-400 * x[0], 200]]
#         )

#     # Initial point
#     x0 = np.array([-1.0, 1.0])

#     # Run both methods
#     print("Standard Newton's Method:")
#     x_newton, hist_newton, f_hist_newton = newton_method(
#         rosenbrock, rosenbrock_grad, rosenbrock_hess, x0
#     )
#     print(f"Solution: {x_newton}")
#     print(f"Iterations: {len(hist_newton)-1}")
#     print(f"Final value: {f_hist_newton[-1]}\n")

#     print("Damped Newton's Method:")
#     x_damped, hist_damped, f_hist_damped = damped_newton_method(
#         rosenbrock, rosenbrock_grad, rosenbrock_hess, x0
#     )
#     print(f"Solution: {x_damped}")
#     print(f"Iterations: {len(hist_damped)-1}")
#     print(f"Final value: {f_hist_damped[-1]}")
