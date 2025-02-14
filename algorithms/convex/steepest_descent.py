"""Steepest descent method for unconstrained optimization."""

import numpy as np
from typing import Callable, Tuple, List
from line_search import backtracking_line_search


def steepest_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
    alpha_init: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Steepest descent method with backtracking line search.

    The method iteratively updates:
        x_{k+1} = x_k - α_k ∇f(x_k)
    where α_k is determined by line search.

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        x0: Initial point
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations
        alpha_init: Initial step size for line search

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
        # Compute gradient
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        # Use unnormalized gradient for direction
        p = -grad

        # Line search with more conservative parameters
        alpha = backtracking_line_search(
            f,
            grad_f,
            x,
            p,
            alpha_init=alpha_init,  # Keep constant initial step
            rho=0.8,  # Less aggressive step size reduction
            c=1e-4,
            max_iter=50,
            alpha_min=1e-16,
        )

        # Update iterate
        x_new = x + alpha * p

        # Prevent too small steps
        if np.linalg.norm(x_new - x) < tol * 1e-3:
            break

        x = x_new
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


def accelerated_gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    L: float,  # Lipschitz constant of gradient
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Nesterov's accelerated gradient descent.

    Uses momentum to achieve faster convergence:
        y_k = x_k + β_k(x_k - x_{k-1})
        x_{k+1} = y_k - α_k ∇f(y_k)
    where β_k is the momentum parameter.

    Parameters:
        f: Objective function f(x)
        grad_f: Gradient function ∇f(x)
        x0: Initial point
        L: Lipschitz constant of gradient
        tol: Tolerance for gradient norm
        max_iter: Maximum iterations

    Returns:
        Tuple containing:
        - Final iterate x
        - History of iterates
        - History of function values
    """
    x = x0.copy()
    y = x0.copy()
    x_prev = x0.copy()

    history = [x.copy()]
    f_history = [f(x)]

    # Initialize step size and momentum parameter
    alpha = 1 / L
    t = 1.0

    for k in range(max_iter):
        # Compute gradient at extrapolated point
        grad = grad_f(y)
        if np.linalg.norm(grad) < tol:
            break

        # Store current x
        x_prev = x.copy()

        # Update x using gradient at y
        x = y - alpha * grad

        # Update momentum parameter
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2

        # Update y with momentum
        beta = (t - 1) / t_next
        y = x + beta * (x - x_prev)

        # Update t
        t = t_next

        # Record history
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


# if __name__ == "__main__":
#     # Example usage on quadratic function
#     A = np.array([[2.0, 0.5], [0.5, 1.0]])
#     b = np.array([1.0, 2.0])

#     def f(x):
#         return 0.5 * x.T @ A @ x - b.T @ x

#     def grad_f(x):
#         return A @ x - b

#     # Initial point
#     x0 = np.array([0.0, 0.0])

#     # Run both methods
#     print("Steepest Descent:")
#     x_sd, hist_sd, f_hist_sd = steepest_descent(f, grad_f, x0)
#     print(f"Solution: {x_sd}")
#     print(f"Iterations: {len(hist_sd)-1}")
#     print(f"Final value: {f_hist_sd[-1]}\n")

#     # Estimate Lipschitz constant (largest eigenvalue of A)
#     L = np.linalg.eigvals(A).max()

#     print("Accelerated Gradient Descent:")
#     x_agd, hist_agd, f_hist_agd = accelerated_gradient_descent(f, grad_f, x0, L)
#     print(f"Solution: {x_agd}")
#     print(f"Iterations: {len(hist_agd)-1}")
#     print(f"Final value: {f_hist_agd[-1]}")
