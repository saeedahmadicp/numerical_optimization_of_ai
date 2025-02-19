# algorithms/convex/quasi_newton.py

import numpy as np
from typing import Callable, Tuple, List, Optional

from .line_search import backtracking_line_search
from .protocols import BaseRootFinder, RootFinderConfig


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

    # Initialize the inverse Hessian approximation as the identity matrix.
    H = np.eye(n)

    history = [x.copy()]  # Record of all iterates.
    f_history = [f(x)]  # Record of function values at each iterate.

    for k in range(max_iter):
        grad = grad_f(x)
        # Stop if the gradient norm is below tolerance.
        if np.linalg.norm(grad) < tol:
            break

        # Compute the search direction: p = -H * grad.
        p = -H @ grad

        # Perform a line search to determine the step size alpha.
        alpha = backtracking_line_search(f, grad_f, x, p)

        # Update the iterate.
        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        # Compute the difference vectors s and y.
        s = x_new - x
        y = grad_new - grad

        # Compute the curvature factor rho.
        rho = 1.0 / (y @ s)
        if rho > 0:  # Only update if the curvature condition holds.
            I = np.eye(n)
            # Perform the BFGS update on the inverse Hessian approximation.
            H = (I - rho * np.outer(s, y)) @ H @ (
                I - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        # Prepare for the next iteration.
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
        """
        Two-loop recursion for computing H∇f in L-BFGS.

        This routine computes the product of the approximate inverse Hessian
        with the gradient vector q using the stored vectors.
        """
        alpha_list = []
        a = q.copy()

        # First loop: traverse stored vectors in reverse order.
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            alpha = rho * s @ a
            alpha_list.append(alpha)
            a = a - alpha * y

        # Scaling: use the most recent y and s to compute a scaling factor.
        if len(y_list) > 0:
            s = s_list[-1]
            y = y_list[-1]
            gamma = (s @ y) / (y @ y)
            r = gamma * a
        else:
            r = a

        # Second loop: traverse stored vectors in original order.
        for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(alpha_list)):
            beta = rho * y @ r
            r = r + (alpha - beta) * s

        return r

    x = x0.copy()
    history = [x.copy()]
    f_history = [f(x)]

    s_list = []  # List to store differences in x (s = x_{k+1} - x_k)
    y_list = []  # List to store differences in gradients (y = grad_{k+1} - grad_k)
    rho_list = []  # List to store curvature values (rho = 1 / (y^T s))

    for k in range(max_iter):
        grad = grad_f(x)
        # Terminate if the gradient norm is sufficiently small.
        if np.linalg.norm(grad) < tol:
            break

        # Compute search direction using the two-loop recursion.
        p = -two_loop_recursion(grad, s_list, y_list, rho_list)

        # Determine step size using a line search.
        alpha = backtracking_line_search(f, grad_f, x, p)

        # Update iterate and compute new gradient.
        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        # Compute differences for updating the stored corrections.
        s = x_new - x
        y = grad_new - grad

        # Compute curvature and update the correction lists if condition holds.
        rho = 1.0 / (y @ s)
        if rho > 0:
            if len(s_list) == m:
                # If maximum storage is reached, remove the oldest correction.
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)

        # Prepare for the next iteration.
        x = x_new
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history


class BFGSMethod(BaseRootFinder):
    """Implementation of BFGS method for root finding using quasi-Newton updates."""

    def __init__(self, config: RootFinderConfig, x0: float, alpha: float = 0.1):
        """
        Initialize BFGS method.

        Args:
            config: Configuration including function, derivative, and tolerances.
            x0: Initial guess.
            alpha: Initial step size for line search.
        """
        if config.derivative is None:
            raise ValueError("BFGS method requires derivative function")

        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Set the current approximation.
        self.alpha = alpha  # Starting step size.
        self._history: List[float] = []

        # Initialize inverse Hessian approximation (for scalar case, it's a 1x1 matrix).
        self.H = np.array([[1.0]])
        self.prev_grad: Optional[np.ndarray] = None
        self.prev_x: Optional[float] = None

    def _backtracking_line_search(self, p: float) -> float:
        """
        Perform backtracking line search to determine an acceptable step size.

        Args:
            p: Search direction.

        Returns:
            A step size alpha that satisfies the Armijo condition.
        """
        c = 1e-4  # Armijo condition constant.
        rho = 0.5  # Reduction factor.
        alpha = self.alpha

        fx = abs(self.func(self.x))
        # Compute a directional derivative approximation using sign for scalar case.
        grad_fx = np.sign(self.func(self.x)) * self.derivative(self.x)  # type: ignore

        # Reduce alpha until the Armijo condition is satisfied.
        while abs(self.func(self.x + alpha * p)) > fx + c * alpha * grad_fx * p:
            alpha *= rho
            if alpha < 1e-10:
                break

        return alpha

    def step(self) -> float:
        """
        Perform one iteration of the BFGS method.

        Returns:
            float: The current approximation of the root.
        """
        if self._converged:
            return self.x

        # Compute function value and its gradient (adjusted with sign for scalar problems).
        fx = self.func(self.x)
        grad = np.array([np.sign(fx) * self.derivative(self.x)])  # type: ignore

        # Compute search direction: p = -H * grad.
        p = -float(self.H @ grad)

        # Determine step size using backtracking line search.
        alpha = self._backtracking_line_search(p)

        # Save the previous state.
        self.prev_x = self.x
        self.prev_grad = grad

        # Update the current approximation.
        self.x += alpha * p
        self._history.append(self.x)
        self.iterations += 1

        # Compute the new gradient.
        new_fx = self.func(self.x)
        new_grad = np.array([np.sign(new_fx) * self.derivative(self.x)])  # type: ignore

        # BFGS update: compute s and y vectors.
        if self.prev_grad is not None:
            s = np.array([self.x - self.prev_x])
            y = new_grad - self.prev_grad

            rho = 1.0 / float(y @ s)
            if rho > 0:  # Only update if curvature condition holds.
                I = np.eye(1)
                self.H = (I - rho * np.outer(s, y)) @ self.H @ (
                    I - rho * np.outer(y, s)
                ) + rho * np.outer(s, s)

        # Check convergence: based on function value tolerance or iteration count.
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "BFGS Method"


def bfgs_root_search(
    f: RootFinderConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for the root finder.
        x0: Initial guess.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # Create configuration instance.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the BFGSMethod.
    method = BFGSMethod(config, x0)

    errors = []
    # Iterate until convergence.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Example function: f(x) = x^2 - 2, aiming to find sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     def df(x):
#         return 2 * x
#
#     # Using the new protocol-based implementation for BFGS:
#     config = RootFinderConfig(func=f, derivative=df, tol=1e-6)
#     method = BFGSMethod(config, x0=1.5)
#
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
