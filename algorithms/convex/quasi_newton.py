# algorithms/convex/quasi_newton.py

import numpy as np
from typing import Callable, Tuple, List, Optional

from .line_search import backtracking_line_search
from .protocols import BaseNumericalMethod, NumericalMethodConfig


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


class BFGSMethod(BaseNumericalMethod):
    """Implementation of BFGS method for both root-finding and optimization."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: float,
        second_derivative: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize BFGS method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess
            second_derivative: Optional, not used but included for protocol compatibility
        """
        if config.derivative is None:
            raise ValueError("BFGS method requires derivative function")

        super().__init__(config)
        self.x = x0

        # Initialize inverse Hessian approximation as identity
        self.H = 1.0
        self.prev_grad = None
        self.prev_x = None

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _backtracking_line_search(self, direction: float) -> float:
        """
        Perform backtracking line search with Armijo condition.

        Args:
            direction: Search direction

        Returns:
            Step size alpha
        """
        alpha = 1.0
        beta = 0.5  # Reduction factor
        c = 1e-4  # Armijo condition parameter

        fx = self.func(self.x)
        dfx = self.derivative(self.x)  # type: ignore

        # For root-finding, use |f(x)| as objective
        if self.method_type == "root":
            fx = abs(fx)
            dfx = np.sign(fx) * dfx

        while True:
            x_new = self.x + alpha * direction
            f_new = self.func(x_new)
            if self.method_type == "root":
                f_new = abs(f_new)

            # Check Armijo condition
            if f_new <= fx + c * alpha * dfx * direction:
                break

            alpha *= beta
            if alpha < 1e-10:
                break

        return alpha

    def step(self) -> float:
        """
        Perform one iteration of BFGS method.

        Returns:
            float: Current approximation
        """
        if self._converged:
            return self.x

        x_old = self.x
        fx = self.func(self.x)
        dfx = self.derivative(self.x)  # type: ignore

        # For root-finding, treat |f(x)| as objective function
        if self.method_type == "root":
            grad = np.sign(fx) * dfx
        else:
            grad = dfx

        details = {
            "f(x)": fx,
            "f'(x)": dfx,
            "H": self.H,
        }

        # Compute search direction
        direction = -self.H * grad

        # Perform line search
        alpha = self._backtracking_line_search(direction)
        step = alpha * direction

        # Update position
        self.x += step

        # Compute new gradient
        fx_new = self.func(self.x)
        dfx_new = self.derivative(self.x)  # type: ignore

        if self.method_type == "root":
            grad_new = np.sign(fx_new) * dfx_new
        else:
            grad_new = dfx_new

        # BFGS update
        if self.prev_grad is not None:
            s = step  # x_new - x_old
            y = grad_new - grad  # grad_new - grad_old

            # Only update if curvature condition is satisfied
            rho = 1.0 / (y * s)
            if rho > 0:
                self.H = (1.0 + rho * y * y) * self.H

        # Store current values for next iteration
        self.prev_grad = grad_new
        self.prev_x = self.x

        details["step"] = step
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence
        if self.method_type == "root":
            if abs(fx) <= self.tol or self.iterations >= self.max_iter:
                self._converged = True
        else:
            grad_norm = abs(dfx_new)
            if grad_norm <= self.tol or self.iterations >= self.max_iter:
                self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "BFGS Method"


def bfgs_search(
    f: NumericalMethodConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: str = "root",
) -> Tuple[float, List[float], int]:
    """Legacy wrapper for backward compatibility."""
    if callable(f):
        h = 1e-7

        def derivative(x: float) -> float:
            return (f(x + h) - f(x)) / h

        config = NumericalMethodConfig(
            func=f,
            method_type=method_type,
            derivative=derivative,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        config = f

    method = BFGSMethod(config, x0)
    errors = []
    prev_x = x0

    while not method.has_converged():
        x = method.step()
        if x != prev_x:  # Only record error if x changed
            errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
