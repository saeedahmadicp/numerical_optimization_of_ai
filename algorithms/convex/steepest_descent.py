# algorithms/convex/steepest_descent.py

"""Steepest descent method for function minimization."""

from typing import List, Tuple, Union, Dict, Any, Callable
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig
from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)


class SteepestDescentMethod(BaseNumericalMethod):
    """Steepest descent method with configurable line search."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: np.ndarray,
    ):
        """Initialize the method.

        Args:
            config: Configuration object with function, derivatives, and parameters
            x0: Initial point
        """
        super().__init__(config)

        # Check method type
        if config.method_type != "optimize":
            raise ValueError(
                "Steepest descent can only be used for optimization problems"
            )

        # Ensure we have a derivative function or scalar case for numerical approximation
        if (
            config.derivative is None
            and not isinstance(x0, (int, float))
            and not (isinstance(x0, np.ndarray) and x0.size == 1)
        ):
            raise ValueError(
                "Steepest descent requires derivative function for vector inputs"
            )

        self.x = np.asarray(x0, dtype=float)
        self._converged = False
        self.iterations = 0

    def compute_descent_direction(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute the steepest descent direction at the current point.

        For steepest descent, the direction is always -∇f(x).

        Args:
            x: Current point

        Returns:
            Union[float, np.ndarray]: Negative gradient direction
        """
        if self.derivative is not None:
            grad = self.derivative(x)
        else:
            # Handle numeric approximation for scalar or 1D array
            if isinstance(x, (int, float)) or (
                isinstance(x, np.ndarray) and x.size == 1
            ):
                if isinstance(x, np.ndarray):
                    x_val = float(x.item())  # Convert 1D array to scalar
                else:
                    x_val = float(x)

                grad = self.estimate_derivative(x_val)
                # If x was an array, convert result back to array
                if isinstance(x, np.ndarray):
                    grad = np.array([grad])
            else:
                raise ValueError("Derivative function is required for vector inputs")

        # Return negative gradient as descent direction
        p = -grad

        # Normalize search direction for better scaling
        if isinstance(p, np.ndarray) and p.size > 1:
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-10:
                p = p / p_norm

        return p

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
        # Use step_length_method from configuration
        method = self.step_length_method or "backtracking"

        # Create a wrapper for gradient function
        if self.derivative is not None:
            grad_f = self.derivative
        else:
            # Create a wrapper for numerical gradient
            def grad_f(point):
                if isinstance(point, (int, float)):
                    return self.estimate_derivative(point)
                elif isinstance(point, np.ndarray) and point.size == 1:
                    return np.array([self.estimate_derivative(float(point.item()))])
                else:
                    raise ValueError(
                        "Derivative function is required for vector inputs"
                    )

        # Extract parameters from step_length_params or use defaults
        params = self.step_length_params or {}

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

        else:
            # Default to backtracking if method is unrecognized
            return backtracking_line_search(
                self.func, grad_f, x, direction, alpha_init=self.initial_step_size
            )

    def step(self) -> np.ndarray:
        """Perform one iteration of steepest descent.

        Returns:
            np.ndarray: New point
        """
        # Compute descent direction using the protocol method
        p = self.compute_descent_direction(self.x)

        # Find step size using the protocol method
        alpha = self.compute_step_length(self.x, p)

        # Get detailed information for history
        if self.derivative is not None:
            grad = self.derivative(self.x)
        else:
            if isinstance(self.x, np.ndarray) and self.x.size == 1:
                grad = self.estimate_derivative(float(self.x.item()))
            elif isinstance(self.x, (int, float)):
                grad = self.estimate_derivative(self.x)
            else:
                grad = "N/A"  # Not available for vector case with no derivative

        # Store details for visualization
        self.add_iteration(
            self.x,
            self.x + alpha * p,
            {
                "gradient": str(grad),
                "search_direction": str(p),
                "step_size": alpha,
                "line_search_method": self.step_length_method or "backtracking",
            },
        )

        # Update position
        self.x = self.x + alpha * p
        self.iterations += 1

        # Check convergence
        error = self.get_error()
        self._converged = error < self.tol or self.iterations >= self.max_iter

        return self.x

    def get_current_x(self) -> np.ndarray:
        """Get current point."""
        return self.x

    @property
    def name(self) -> str:
        line_search_name = self.step_length_method or "backtracking"
        return f"Steepest Descent with {line_search_name.replace('_', ' ').title()} Line Search"


def steepest_descent_search(
    f: Union[NumericalMethodConfig, Callable],
    x0: float,
    step_length_method: str = "backtracking",
    step_length_params: Dict[str, Any] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for optimization
        x0: Initial guess
        step_length_method: Method to use for line search
        step_length_params: Parameters for the line search method
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (minimum, errors, iterations)
    """
    if callable(f):
        config = NumericalMethodConfig(
            func=f,
            method_type="optimize",
            tol=tol,
            max_iter=max_iter,
            step_length_method=step_length_method,
            step_length_params=step_length_params or {},
            descent_direction_method="steepest_descent",
            # If scalar input, no need for explicit derivative function
            derivative=None,
        )
    else:
        config = f

    method = SteepestDescentMethod(config, x0)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
