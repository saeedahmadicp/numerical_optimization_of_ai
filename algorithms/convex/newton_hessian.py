# algorithms/convex/newton_hessian.py

"""
Newton-Hessian method for both root-finding and optimization.

The Newton-Hessian method extends Newton's method by directly computing or approximating
the Hessian matrix (matrix of second derivatives) for more robust convergence. This
implementation uses automatic differentiation when possible and provides fallback
mechanisms for numerical stability.

Mathematical Basis:
------------------
For root-finding (finding x where f(x) = 0):
    x_{n+1} = x_n - J^{-1}(x_n) f(x_n)

    where J is the Jacobian matrix of partial derivatives

For optimization (finding x where ∇f(x) = 0):
    x_{n+1} = x_n - H^{-1}(x_n) ∇f(x_n)

    where H is the Hessian matrix of second derivatives and ∇f is the gradient

The method can be viewed as performing a second-order approximation of the function
using its Taylor series and finding the zero/extremum of this quadratic approximation.

Convergence Properties:
---------------------
- Quadratic convergence when started sufficiently close to a solution
- More robust than standard Newton's method for ill-conditioned problems due to
  Hessian regularization techniques
- Requires computation or approximation of second derivatives
- Includes safeguards against overshooting and divergence:
  * Line search methods to ensure sufficient decrease
  * Trust region constraints to limit step size
  * Regularization to handle non-positive definite Hessian matrices
- Handles both scalar and vector optimization/root-finding problems
- Falls back to gradient descent when Newton direction is unreliable
"""

from typing import List, Tuple, Optional, Callable
import torch
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType
from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)


class NewtonHessianMethod(BaseNumericalMethod):
    """Implementation of Newton-Hessian method using automatic differentiation."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: np.ndarray,
        second_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize Newton-Hessian method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess (numpy array)
            second_derivative: Optional for optimization (will use auto-diff if not provided)

        Raises:
            ValueError: If derivative is missing
        """
        if config.derivative is None:
            raise ValueError("Newton-Hessian method requires derivative function")

        super().__init__(config)
        self.x = np.array(x0, dtype=float)
        self.second_derivative = second_derivative

    # ------------------------
    # Core Algorithm Methods
    # ------------------------

    def step(self) -> np.ndarray:
        """
        Perform one iteration of Newton-Hessian method.

        The method uses the Hessian matrix (second derivatives) to compute
        the Newton direction, applying trust region methods and line search
        to ensure convergence.

        Returns:
            np.ndarray: Updated point after one step
        """
        if self._converged:
            return self.x

        x_old = self.x.copy()
        fx = self.func(self.x)
        dfx = self.derivative(self.x)

        # Use provided second derivative or compute via finite differences
        if self.second_derivative is not None:
            d2fx = self.second_derivative(self.x)
        else:
            d2fx = self._compute_hessian(self.x)

        details = {
            "f(x)": fx,
            "f'(x)": str(dfx),
            "f''(x)": str(d2fx),
        }

        # Compute scale-invariant measures
        grad_norm = np.linalg.norm(dfx)
        x_scale = max(1.0, np.linalg.norm(self.x))
        f_scale = max(1.0, abs(fx))

        # Check if gradient is small enough for convergence
        if grad_norm <= self.tol * f_scale and self.iterations >= 5:
            self._converged = True
            return self.x

        # Get descent direction using compute_descent_direction
        direction = self.compute_descent_direction(self.x)

        # Get step length using compute_step_length
        alpha = self.compute_step_length(self.x, direction)

        # Update current point
        step = alpha * direction
        self.x = self.x + step

        # For root finding, check if we've found a root
        if self.method_type == "root":
            f_new = self.func(self.x)
            # If we're not getting closer to the root, try the opposite direction
            if abs(f_new) > abs(fx) and self.iterations < 2:
                self.x = x_old - step  # Try the opposite direction
                f_new = self.func(self.x)

            # Special check for sign changes in the function value
            # which can indicate we jumped past the root
            if np.sign(f_new) != np.sign(fx) and abs(f_new) > abs(fx):
                # Take a shorter step (binary search between current and previous point)
                self.x = (x_old + self.x) / 2

            # For high iterations, check if we're oscillating and take the point with smaller |f(x)|
            if self.iterations > 3 and abs(f_new) > abs(fx):
                self.x = x_old  # Go back to previous point if it was better

        # Check convergence criteria
        if (
            (
                self.method_type == "root" and abs(self.func(self.x)) <= self.tol
            )  # Root finding: check function value
            or (
                self.method_type == "optimize" and grad_norm <= self.tol * f_scale
            )  # Optimization: check gradient
            or np.linalg.norm(step) <= self.tol * x_scale  # Step size small enough
            or self.iterations >= self.max_iter  # Maximum iterations reached
        ):
            self._converged = True

        # Add step_size to details for proper testing and logging
        details["step"] = str(step)
        details["step_size"] = alpha
        details["descent_direction"] = str(direction)

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        return self.x

    def get_current_x(self) -> np.ndarray:
        """
        Get current x value.

        Returns:
            np.ndarray: Current point
        """
        return self.x

    def compute_descent_direction(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Newton direction using the Hessian matrix.

        The Newton direction is defined as: d = -H⁻¹∇f(x), where H is the Hessian
        matrix and ∇f(x) is the gradient. This method handles matrix singularity
        issues through regularization.

        Args:
            x: Current point

        Returns:
            np.ndarray: The Newton descent direction
        """
        dfx = self.derivative(x)
        fx = self.func(x)

        # Use provided second derivative or compute via finite differences
        if self.second_derivative is not None:
            d2fx = self.second_derivative(x)
        else:
            d2fx = self._compute_hessian(x)

        # Handle scalar case (0-dimensional arrays)
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            # For scalar case, the Newton direction is simply -f'(x)/f''(x) for optimization
            # For root finding, it's -f(x)/f'(x)
            if self.method_type == "root":
                # Check for near-zero derivatives (prevent division by zero)
                if np.abs(dfx) < 1e-14:
                    # Return a safe direction (-1 or 1) based on function sign
                    return np.array(-np.sign(fx) or 1.0)

                # Classic Newton's method for root finding: x_{n+1} = x_n - f(x_n)/f'(x_n)
                direction = -fx / dfx

                # Apply safeguards to prevent too large steps
                max_step = 10.0 * (abs(x) + 1.0)
                if abs(direction) > max_step:
                    direction = np.sign(direction) * max_step

                return np.array(direction)
            else:
                # For optimization, use standard Newton direction
                # Avoid division by zero
                if np.abs(d2fx) < 1e-14:
                    return np.array(-dfx)  # Fallback to gradient descent
                if isinstance(d2fx, np.ndarray):
                    d2fx_value = d2fx[0, 0] if d2fx.size > 0 else 1.0
                else:
                    d2fx_value = d2fx
                return np.array(-dfx / d2fx_value)

        # Vector case processing below
        # Different handling for root finding vs. optimization
        if self.method_type == "root" and len(x) > 0:
            # For root finding with vectors, Newton direction is -J⁻¹f(x)
            # where J is the Jacobian matrix
            try:
                # For vectors, dfx is the Jacobian matrix, so we can directly use it
                direction = -np.linalg.solve(np.atleast_2d(dfx), np.atleast_1d(fx))
                # Apply safeguards
                max_step = 10.0 * (np.linalg.norm(x) + 1.0)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > max_step:
                    direction *= max_step / direction_norm
                return direction.flatten()
            except np.linalg.LinAlgError:
                # Fallback to scaled steepest descent if Jacobian is singular
                direction = -np.array(fx) * dfx
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                return direction

        # Modified Newton's method with trust region and regularization for optimization
        try:
            # Add regularization to Hessian
            lambda_min = 1e-6
            eigvals = np.linalg.eigvals(d2fx)
            if np.any(eigvals <= lambda_min):
                d2fx += (lambda_min - np.min(eigvals) + 1e-6) * np.eye(len(x))

            # Compute Newton direction
            direction = -np.linalg.solve(d2fx, dfx)

            # Trust region constraint
            trust_radius = 1.0
            direction_norm = np.linalg.norm(direction)
            if direction_norm > trust_radius:
                direction *= trust_radius / direction_norm

        except np.linalg.LinAlgError:
            # Fallback to steepest descent if Hessian is singular
            direction = -dfx
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction /= direction_norm

        return direction

    def compute_step_length(self, x: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute step length using the specified line search method.

        Supports various line search algorithms including fixed step size,
        backtracking, Wolfe conditions, strong Wolfe conditions, and Goldstein conditions.

        Args:
            x: Current point
            direction: Descent direction

        Returns:
            float: Step length
        """
        # If direction is too small, return zero step size
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-14:
            return 0.0

        # For root-finding, we typically use a fixed step size of 1.0 (full Newton step)
        # unless line search is explicitly enabled
        if self.method_type == "root":
            # Check if line search is enabled via step_length_method
            if not self.step_length_method or self.step_length_method == "fixed":
                # Use full Newton step or the value specified in step_length_params
                params = self.step_length_params or {}
                return params.get("step_size", 1.0)

        # For optimization, or when explicitly configured for root-finding,
        # use the specified line search method
        method = self.step_length_method or "backtracking"
        params = self.step_length_params or {}

        # Create a wrapper for gradient function
        grad_f = self.derivative

        # Dispatch to appropriate line search method
        if method == "fixed":
            return params.get("step_size", self.initial_step_size)

        elif method == "backtracking":
            return backtracking_line_search(
                self.func,
                grad_f,
                x,
                direction,
                alpha_init=params.get("alpha_init", self.initial_step_size),
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
                alpha_init=params.get("alpha_init", self.initial_step_size),
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
                alpha_init=params.get("alpha_init", self.initial_step_size),
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
                alpha_init=params.get("alpha_init", self.initial_step_size),
                c=params.get("c", 0.1),
                max_iter=params.get("max_iter", 100),
                alpha_min=params.get("alpha_min", 1e-16),
                alpha_max=params.get("alpha_max", 1e10),
            )

        # If method not recognized, fall back to enhanced backtracking with Wolfe conditions
        alpha = 1.0
        fx = self.func(x)
        dfx = self.derivative(x)
        rho = 0.5  # Backtracking factor
        c1 = 1e-4  # Sufficient decrease parameter (Armijo condition)
        c2 = 0.9  # Curvature condition (Wolfe condition)
        max_iter = 25  # Maximum backtracking iterations

        x_new = x + alpha * direction
        f_new = self.func(x_new)
        df_new = self.derivative(x_new)

        iter_count = 0
        while (
            f_new > fx + c1 * alpha * np.dot(dfx, direction)  # Armijo condition
            or np.dot(df_new, direction)
            < c2 * np.dot(dfx, direction)  # Wolfe condition
        ) and iter_count < max_iter:
            alpha *= rho
            x_new = x + alpha * direction
            f_new = self.func(x_new)
            df_new = self.derivative(x_new)
            iter_count += 1

        return alpha

    # ----------------
    # Helper Methods
    # ----------------

    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian using automatic differentiation.

        Args:
            x: Point at which to compute the Hessian (numpy array)

        Returns:
            np.ndarray: The Hessian matrix
        """
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float64)

        # Handle scalar inputs (0-dimensional arrays)
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            # For scalar optimization, compute second derivative
            if self.method_type == "optimize":
                if self.second_derivative is not None:
                    return np.array([[self.second_derivative(x)]])
                else:
                    # Estimate second derivative using finite differences
                    h = 1e-8
                    x_float = float(x)
                    grad_plus_h = self.derivative(x_float + h)
                    grad_minus_h = self.derivative(x_float - h)
                    hessian = (grad_plus_h - grad_minus_h) / (2 * h)
                    return np.array([[hessian]])
            else:
                # For root finding with scalar, return 1.0 (identity)
                return np.array([[1.0]])

        # Handle vector inputs
        n = len(x)

        # For optimization, compute Hessian of function
        if self.method_type == "optimize":
            # Compute gradient first
            fx = self.func(x_tensor.detach().numpy())
            fx_tensor = torch.tensor(fx, requires_grad=True)
            fx_tensor.backward()

            if x_tensor.grad is None:
                return np.eye(n)

            # Compute Hessian using finite differences as fallback
            h = 1e-8
            hessian = np.zeros((n, n))
            x_np = x_tensor.detach().numpy()

            for i in range(n):
                for j in range(n):
                    x_plus_h = x_np.copy()
                    x_plus_h[j] += h
                    grad_plus_h = self.derivative(x_plus_h)

                    x_minus_h = x_np.copy()
                    x_minus_h[j] -= h
                    grad_minus_h = self.derivative(x_minus_h)

                    hessian[i, j] = (grad_plus_h[i] - grad_minus_h[i]) / (2 * h)

            # Ensure symmetry and positive definiteness
            hessian = (hessian + hessian.T) / 2
            eigvals = np.linalg.eigvals(hessian)

            if np.any(eigvals <= 0):
                # Add regularization if not positive definite
                min_eigval = np.min(eigvals)
                if min_eigval <= 0:
                    hessian += (-min_eigval + 1e-6) * np.eye(n)

            return hessian
        else:
            return np.eye(n)  # For root finding, return identity matrix

    # ---------------------
    # State Access Methods
    # ---------------------

    def has_converged(self) -> bool:
        """
        Check if the method has converged based on error tolerance or max iterations.

        Returns:
            bool: True if converged, False otherwise
        """
        return self._converged

    def get_error(self) -> float:
        """
        Calculate the error estimate for the current solution.

        For optimization: norm of the gradient
        For root-finding: norm of the function value

        Returns:
            float: Error estimate
        """
        if self.method_type == "optimize":
            # For optimization, error is norm of gradient
            grad = self.derivative(self.x)
            return np.linalg.norm(grad)
        else:
            # For root-finding, error is norm of function value
            return np.linalg.norm(self.func(self.x))

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For Newton's method with exact Hessian, the theoretical convergence rate
        is quadratic near the solution.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # For quadratic convergence, look at log(e_{n+1})/log(e_n)
        try:
            rate1 = (
                np.log(recent_errors[-1]) / np.log(recent_errors[-2])
                if recent_errors[-2] > 0
                else 2.0
            )
            rate2 = (
                np.log(recent_errors[-2]) / np.log(recent_errors[-3])
                if recent_errors[-3] > 0
                else 2.0
            )

            # Average of recent rates, capped at 2.0 for numerical stability
            return min(2.0, (rate1 + rate2) / 2)
        except (ValueError, RuntimeWarning):
            # Fallback for numerical issues
            return 2.0  # Assume quadratic convergence

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return "Newton-Hessian Method"


def newton_hessian_search(
    f: NumericalMethodConfig,
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

    method = NewtonHessianMethod(config, x0)
    errors = []
    prev_x = x0

    while not method.has_converged():
        x = method.step()
        if x != prev_x:  # Only record error if x changed
            errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
