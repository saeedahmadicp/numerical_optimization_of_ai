# algorithms/convex/newton_hessian.py

"""Newton-Hessian method for both root-finding and optimization."""

from typing import List, Tuple, Optional, Callable
import torch
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig


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

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _compute_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian using automatic differentiation.

        Args:
            x: Point at which to compute the Hessian (numpy array)

        Returns:
            np.ndarray: The Hessian matrix
        """
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float64)
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

    def step(self) -> np.ndarray:
        """Perform one iteration of Newton-Hessian method."""
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

        # Modified Newton's method with trust region and regularization
        try:
            # Add regularization to Hessian
            lambda_min = 1e-6
            eigvals = np.linalg.eigvals(d2fx)
            if np.any(eigvals <= lambda_min):
                d2fx += (lambda_min - np.min(eigvals) + 1e-6) * np.eye(len(self.x))

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

        # Enhanced backtracking line search
        alpha = 1.0
        rho = 0.5  # Backtracking factor
        c1 = 1e-4  # Sufficient decrease parameter (Armijo condition)
        c2 = 0.9  # Curvature condition (Wolfe condition)
        max_iter = 25  # Maximum backtracking iterations

        x_new = self.x + alpha * direction
        f_new = self.func(x_new)
        df_new = self.derivative(x_new)

        iter_count = 0
        while (
            f_new > fx + c1 * alpha * np.dot(dfx, direction)  # Armijo condition
            or np.dot(df_new, direction) < c2 * np.dot(dfx, direction)
        ) and iter_count < max_iter:  # Wolfe condition
            alpha *= rho
            x_new = self.x + alpha * direction
            f_new = self.func(x_new)
            df_new = self.derivative(x_new)
            iter_count += 1

        # Update current point
        step = alpha * direction
        self.x = x_new

        # Check convergence criteria
        if (
            grad_norm <= self.tol * f_scale  # Gradient norm small enough
            or np.linalg.norm(step) <= self.tol * x_scale  # Step size small enough
            or self.iterations >= self.max_iter  # Maximum iterations reached
            or iter_count >= max_iter  # Line search failed to improve
        ):
            self._converged = True

        details["step"] = str(step)
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        return self.x

    @property
    def name(self) -> str:
        return "Newton-Hessian Method"


def newton_hessian_search(
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

    method = NewtonHessianMethod(config, x0)
    errors = []
    prev_x = x0

    while not method.has_converged():
        x = method.step()
        if x != prev_x:  # Only record error if x changed
            errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
