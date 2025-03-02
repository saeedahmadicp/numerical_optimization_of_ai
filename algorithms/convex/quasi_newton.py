# algorithms/convex/quasi_newton.py

"""
Quasi-Newton methods for optimization and root-finding.

This module implements Quasi-Newton methods, specifically the BFGS (Broyden-Fletcher-Goldfarb-Shanno)
and L-BFGS (Limited-memory BFGS) methods. These methods approximate the Hessian matrix or its
inverse without explicitly computing second derivatives, making them efficient for large-scale
optimization problems while maintaining superlinear convergence rates.

Mathematical Basis:
----------------
For optimization (finding x where ∇f(x) = 0):
    1. Start with an initial point x₀ and an initial Hessian approximation H₀ (often identity matrix)
    2. At each iteration k:
       a. Compute search direction pₖ = -Hₖ∇f(xₖ)
       b. Determine step length αₖ using line search
       c. Update position: xₖ₊₁ = xₖ + αₖpₖ
       d. Compute difference vectors:
          sₖ = xₖ₊₁ - xₖ (position difference)
          yₖ = ∇f(xₖ₊₁) - ∇f(xₖ) (gradient difference)
       e. Update Hessian approximation using BFGS formula:
          Hₖ₊₁ = (I - ρₖsₖyₖᵀ)Hₖ(I - ρₖyₖsₖᵀ) + ρₖsₖsₖᵀ
          where ρₖ = 1/(yₖᵀsₖ)

For root-finding (finding x where f(x) = 0):
    1. Use Newton-like iterations: xₖ₊₁ = xₖ - f(xₖ)/f'(xₖ)
    2. Approximate the Jacobian matrix inverse using BFGS updates
    3. Step direction is calculated as -f(xₖ)/f'(xₖ) for scalar problems

Convergence Properties:
--------------------
- Superlinear convergence rate (faster than first-order methods)
- Does not require the computation or storage of the Hessian matrix
- Maintains a positive definite approximation of the Hessian inverse
- More robust than Newton's method when far from the solution
- Generally requires fewer function evaluations than gradient descent
- L-BFGS variant uses limited memory for large-scale problems

The implementation includes:
- Support for both scalar and vector optimization/root-finding problems
- Multiple line search methods (backtracking, Wolfe, strong Wolfe, Goldstein)
- Safeguards to ensure positive definiteness of Hessian approximation
- Robust convergence criteria based on gradient norm and function improvement
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Union

from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)
from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


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
    """
    Implementation of BFGS (Broyden-Fletcher-Goldfarb-Shanno) method for numerical optimization and root-finding.

    BFGS is a quasi-Newton method that approximates the Hessian matrix using rank-one updates
    based on gradient evaluations. This avoids the expensive computation and storage of the
    true Hessian matrix while maintaining superlinear convergence properties.

    Mathematical basis:
    - For optimization: Uses quasi-Newton updates to approximate the inverse Hessian matrix
      and determine descent directions.
    - For root-finding: Uses a Newton-like approach with approximate Jacobian updates.

    Convergence properties:
    - Superlinear convergence rate for smooth functions
    - Self-correcting: Hessian approximation improves as iterations progress
    - Positive definiteness of Hessian approximation is maintained
    - Minimal storage requirements compared to full Newton methods
    - Robust performance even with inexact line searches

    Implementation features:
    - Works with both scalar and vector inputs
    - Implements multiple line search options for step size control
    - Uses safeguards to ensure the Hessian approximation stays positive definite
    - Adapts the approach based on whether solving optimization or root-finding problem
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: Union[float, np.ndarray],
        second_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize BFGS method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess (scalar or numpy array)
            second_derivative: Optional, not used but included for protocol compatibility
        """
        if config.derivative is None:
            raise ValueError("BFGS method requires derivative function")

        super().__init__(config)

        # Handle both scalar and array inputs
        is_scalar = isinstance(x0, (int, float))
        self.is_scalar = is_scalar

        if is_scalar:
            self.x = float(x0)
            # For scalar case, the "Hessian" is just a 1x1 matrix (a scalar)
            self.H = np.array([[1.0]], dtype=float)
        else:
            self.x = np.array(x0, dtype=float)
            # Initialize inverse Hessian approximation as identity matrix
            n = len(x0)
            self.H = np.eye(n, dtype=float)

        # Limit maximum step size to prevent overflow
        self.max_step_size = 10.0

        # Minimum step size threshold for convergence
        self.min_step_size = 1e-14

        # For tracking convergence
        self.prev_x = self.x
        self.stuck_iterations = 0

    def step(self) -> Union[float, np.ndarray]:
        """Perform one iteration of BFGS method."""
        if self._converged:
            return self.x

        x_old = self.x if self.is_scalar else self.x.copy()
        fx_old = self.func(x_old)
        dfx_old = self.derivative(x_old)

        # Compute search direction: H * g
        direction = self.compute_descent_direction(x_old)

        # Normalize direction if too large for vector case
        if not self.is_scalar:
            direction_norm = np.linalg.norm(direction)
            if direction_norm > self.max_step_size:
                direction = self.max_step_size * direction / direction_norm

        # Line search
        alpha = self.compute_step_length(x_old, direction)
        step = alpha * direction

        # Update position
        self.x = x_old + step
        fx_new = self.func(self.x)
        dfx_new = self.derivative(self.x)

        # Compute difference vectors
        s = self.x - x_old  # Position difference
        y = dfx_new - dfx_old  # Gradient difference

        # BFGS update with safeguards
        if self.is_scalar:
            # For scalar case, use simple formula
            sy = s * y
            if sy > 0:  # Only update if curvature condition is satisfied
                # Handle scalar case directly
                self.H = np.array(
                    [
                        [
                            self.H[0, 0]
                            + (y * y) / sy
                            - (self.H[0, 0] * y * y * self.H[0, 0])
                            / (y * self.H[0, 0] * y)
                        ]
                    ]
                )
        else:
            # Vector case
            sy = np.dot(s, y)
            if sy > 0:  # Only update if curvature condition is satisfied
                rho = 1.0 / sy
                I = np.eye(len(self.x))

                # Update Hessian approximation using BFGS formula
                self.H = (I - rho * np.outer(s, y)) @ self.H @ (
                    I - rho * np.outer(y, s)
                ) + rho * np.outer(s, s)

                # Add regularization if needed
                eigvals = np.linalg.eigvals(self.H)
                if np.any(eigvals <= 0):
                    self.H += 1e-6 * I

        # Store iteration details
        details = {
            "direction": str(direction),
            "step_size": alpha,
            "f_old": fx_old,
            "f_new": fx_new,
            "f(x)": fx_new,  # Add this for compatibility with tests
            "f'(x)": str(dfx_new),
            "gradient_old": str(dfx_old),
            "gradient_new": str(dfx_new),
            "H": str(self.H),
            "step": str(step),
            "line_search_method": self.step_length_method or "backtracking",
        }
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence with better criteria
        if self.is_scalar:
            if self.method_type == "root":
                # For root-finding, measure convergence by how close f(x) is to zero
                error = abs(fx_new)
                self.error = error
                if error <= self.tol or self.iterations >= self.max_iter:
                    self._converged = True
            else:
                # For optimization
                grad_norm = abs(dfx_new)
                x_scale = max(1.0, abs(self.x))
                rel_step = abs(step) / (abs(x_old) + 1e-10)

                f_scale = max(1.0, abs(fx_new))
                rel_improvement = abs(fx_new - fx_old) / (abs(fx_old) + 1e-10)

                if self.iterations >= 3 and (
                    (grad_norm <= self.tol * f_scale)  # Gradient norm small enough
                    or (
                        rel_step <= self.tol and rel_improvement <= self.tol
                    )  # Step and improvement small
                    or self.iterations >= self.max_iter  # Max iterations reached
                ):
                    self._converged = True
        else:
            # Vector case
            grad_norm = np.linalg.norm(dfx_new)
            x_scale = max(1.0, np.linalg.norm(self.x))
            rel_step = np.linalg.norm(step) / (np.linalg.norm(x_old) + 1e-10)

            f_scale = max(1.0, abs(fx_new))
            rel_improvement = abs(fx_new - fx_old) / (abs(fx_old) + 1e-10)

            if self.iterations >= 3 and (  # Minimum iterations requirement
                (grad_norm <= self.tol * f_scale)  # Gradient norm small enough
                or (
                    rel_step <= self.tol and rel_improvement <= self.tol
                )  # Step and improvement small
                or self.iterations >= self.max_iter  # Max iterations reached
            ):
                self._converged = True

        return self.x

    def get_current_x(self) -> Union[float, np.ndarray]:
        """Get current x value."""
        return self.x

    def compute_descent_direction(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute the BFGS descent direction using the approximated inverse Hessian.

        Args:
            x: Current point (scalar or numpy array)

        Returns:
            Direction vector (scalar or numpy array)
        """
        if self.is_scalar:
            # For scalar case, convert to array for matrix multiplication
            gradient = np.array([self.derivative(x)])

            # For root-finding, we want f(x) = 0, so the goal is to find x where f(x) = 0
            # For optimization, we want f'(x) = 0, so the goal is to find x where f'(x) = 0
            if self.method_type == "root":
                # For root-finding, use Jacobian method: Newton's method for solving f(x) = 0
                # The direction is -f(x)/f'(x)
                f_val = self.func(x)
                if abs(self.derivative(x)) < 1e-10:
                    direction = (
                        -np.sign(f_val) * 0.1
                    )  # Safe direction if derivative is too small
                else:
                    direction = -f_val / self.derivative(x)
            else:
                # For optimization, use the BFGS descent direction
                direction = -float(self.H @ gradient)

            return direction
        else:
            gradient = self.derivative(x)
            direction = -self.H @ gradient
            return direction

    def compute_step_length(
        self, x: Union[float, np.ndarray], direction: Union[float, np.ndarray]
    ) -> float:
        """
        Compute the step length using the specified line search method.

        Args:
            x: Current point
            direction: Descent direction

        Returns:
            float: Step length (alpha)
        """
        # If direction is too small, return zero step size
        if self.is_scalar:
            if abs(direction) < self.min_step_size:
                return 0.0
        else:
            if np.linalg.norm(direction) < self.min_step_size:
                return 0.0

        # For root-finding, we typically use a fixed step size of 1.0 (full Newton step)
        # unless line search is explicitly enabled
        if self.method_type == "root":
            # Check if line search is enabled via step_length_method
            if not self.step_length_method or self.step_length_method == "fixed":
                # Use full Newton step or the value specified in step_length_params
                params = self.step_length_params or {}
                return params.get("step_size", 1.0)

        # For optimization, use the specified line search method
        method = self.step_length_method or "backtracking"
        params = self.step_length_params or {}

        # Create a wrapper for gradient function
        if self.derivative is None:
            raise ValueError("Derivative function is required for line search")

        grad_f = self.derivative

        # Dispatch to appropriate line search method
        if method == "fixed":
            return params.get("step_size", 1.0)

        elif method == "backtracking":
            return backtracking_line_search(
                self.func,
                grad_f,
                x,
                direction,
                alpha_init=params.get("alpha_init", 1.0),
                rho=params.get("rho", 0.5),
                c=params.get("c", 1e-4),
                max_iter=params.get("max_iter", 100),
                alpha_min=params.get("alpha_min", 1e-16),
            )

        elif method == "wolfe":
            return wolfe_line_search(
                self.func,
                grad_f,
                x,
                direction,
                alpha_init=params.get("alpha_init", 1.0),
                c1=params.get("c1", 1e-4),
                c2=params.get("c2", 0.9),
                max_iter=params.get("max_iter", 25),
                zoom_max_iter=params.get("zoom_max_iter", 10),
                alpha_min=params.get("alpha_min", 1e-16),
            )

        elif method == "strong_wolfe":
            return strong_wolfe_line_search(
                self.func,
                grad_f,
                x,
                direction,
                alpha_init=params.get("alpha_init", 1.0),
                c1=params.get("c1", 1e-4),
                c2=params.get("c2", 0.1),
                max_iter=params.get("max_iter", 25),
                zoom_max_iter=params.get("zoom_max_iter", 10),
                alpha_min=params.get("alpha_min", 1e-16),
            )

        elif method == "goldstein":
            return goldstein_line_search(
                self.func,
                grad_f,
                x,
                direction,
                alpha_init=params.get("alpha_init", 1.0),
                c=params.get("c", 0.1),
                max_iter=params.get("max_iter", 100),
                alpha_min=params.get("alpha_min", 1e-16),
                alpha_max=params.get("alpha_max", 1e10),
            )

        # Default to full step if method is not recognized
        return 1.0

    def get_error(self) -> float:
        """
        Get the current error estimate.

        For root-finding: |f(x)|
        For optimization: |f'(x)| or derivative norm

        Returns:
            float: Error estimate
        """
        if self.method_type == "root":
            # For root-finding, error is how close f(x) is to zero
            return abs(self.func(self.x))
        else:
            # For optimization, error is how close gradient is to zero
            return (
                abs(self.derivative(self.x))
                if self.is_scalar
                else np.linalg.norm(self.derivative(self.x))
            )

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For BFGS method, the theoretical rate is superlinear (between linear and quadratic)
        for well-behaved functions near the solution.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # For superlinear convergence, we expect error_{n+1} ≈ C * error_n^p where 1 < p < 2
        # Estimate as error_{n+1} / error_n^1.5 as a compromise
        rate1 = (
            recent_errors[-1] / (recent_errors[-2] ** 1.5)
            if recent_errors[-2] > 0
            else 0
        )
        rate2 = (
            recent_errors[-2] / (recent_errors[-3] ** 1.5)
            if recent_errors[-3] > 0
            else 0
        )

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        return f"BFGS {'Root-Finding' if self.method_type == 'root' else 'Optimization'} Method"


def bfgs_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "root",
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
        errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
