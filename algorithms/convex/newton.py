# algorithms/convex/newton.py

"""Newton's method for both root-finding and optimization."""

from typing import List, Tuple, Optional, Callable
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class NewtonMethod(BaseNumericalMethod):
    """Implementation of Newton's method for both root-finding and optimization."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: float,
        second_derivative: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize Newton's method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess
            second_derivative: Required for optimization mode

        Raises:
            ValueError: If derivative is missing, or if second_derivative is missing in optimization mode
        """
        if config.derivative is None:
            raise ValueError("Newton's method requires derivative function")

        if config.method_type == "optimize" and second_derivative is None:
            raise ValueError(
                "Newton's method requires second derivative for optimization"
            )

        super().__init__(config)
        self.x = x0
        self.second_derivative = second_derivative

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of Newton's method.

        For root-finding: x_{n+1} = x_n - f(x_n)/f'(x_n)
        For optimization: x_{n+1} = x_n - f'(x_n)/f''(x_n)

        Returns:
            float: Current approximation
        """
        if self._converged:
            return self.x

        x_old = self.x

        if self.method_type == "root":
            # Root-finding mode
            fx = self.func(self.x)
            dfx = self.derivative(self.x)  # type: ignore

            # Avoid division by zero
            if abs(dfx) < 1e-10:
                self._converged = True
                return self.x

            # Newton step for root-finding
            step = -fx / dfx

            details = {
                "f(x)": fx,
                "f'(x)": dfx,
                "step": step,
            }

            # Update approximation
            self.x = self.x + step

            # Check convergence for root-finding
            if abs(fx) <= self.tol or self.iterations >= self.max_iter:
                self._converged = True

        else:  # optimization mode
            # Evaluate derivatives
            dfx = self.derivative(self.x)  # type: ignore
            d2fx = self.second_derivative(self.x)  # type: ignore
            fx = self.func(self.x)  # Also evaluate function for monitoring progress

            details = {
                "f(x)": fx,
                "f'(x)": dfx,
                "f''(x)": d2fx,
            }

            # Compute scale-invariant measures for vectors
            grad_norm = np.linalg.norm(dfx)
            x_norm = np.linalg.norm(self.x)
            x_scale = max(1.0, x_norm)
            f_scale = max(1.0, abs(fx))

            # Early convergence check with normalized measures
            if grad_norm <= self.tol * f_scale:
                self._converged = True
                return self.x

            # Compute search direction
            if isinstance(d2fx, np.ndarray):  # Matrix case
                try:
                    # Newton direction: H^(-1) * g
                    direction = -np.linalg.solve(d2fx, dfx)

                    # Ensure descent direction
                    if np.dot(direction, dfx) > 0:
                        # If not descent, use negative gradient
                        direction = -dfx
                except np.linalg.LinAlgError:
                    # Fallback to gradient descent if Hessian is singular
                    direction = -dfx
            else:  # Scalar case
                if abs(d2fx) < 1e-10:
                    direction = -dfx
                else:
                    direction = -dfx / d2fx

            # Normalize direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm > x_scale:
                direction = direction * (x_scale / direction_norm)

            # Backtracking line search
            alpha = 1.0
            beta = 0.5  # Reduction factor
            c = 0.1  # Sufficient decrease parameter
            x_new = self.x + alpha * direction
            f_new = self.func(x_new)

            # Armijo condition
            while f_new > fx + c * alpha * np.dot(dfx, direction):
                alpha *= beta
                if alpha < 1e-10:
                    break
                x_new = self.x + alpha * direction
                f_new = self.func(x_new)

            step = alpha * direction
            details["step"] = step
            self.x = x_new

            # Check convergence using normalized criteria
            if (
                grad_norm <= self.tol * f_scale  # Normalized gradient small enough
                or np.linalg.norm(step)
                <= self.tol * x_scale  # Relative step size small enough
                or self.iterations >= self.max_iter
            ):
                self._converged = True

        # Store iteration data and increment counter
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        return self.x

    @property
    def name(self) -> str:
        return "Newton's Method"


def newton_search(
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

        def second_derivative(x: float) -> float:
            return (f(x + h) + f(x - h) - 2 * f(x)) / (h * h)

        config = NumericalMethodConfig(
            func=f,
            method_type=method_type,
            derivative=derivative,
            tol=tol,
            max_iter=max_iter,
        )
        method = NewtonMethod(
            config, x0, second_derivative if method_type == "optimize" else None
        )
    else:
        config = f
        method = NewtonMethod(config, x0)

    errors = []
    prev_x = x0

    while not method.has_converged():
        x = method.step()
        if x != prev_x:  # Only record error if x changed
            errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
