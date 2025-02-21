# algorithms/convex/powell.py

"""Powell's conjugate direction method for derivative-free optimization."""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize_scalar

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class PowellMethod(BaseNumericalMethod):
    """Implementation of Powell's method for derivative-free optimization."""

    def __init__(self, config: NumericalMethodConfig, x0: np.ndarray):
        """
        Initialize Powell's method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial guess (numpy array)

        Raises:
            ValueError: If method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError("Powell's method can only be used for optimization")

        config.use_derivative_free = True  # Ensure derivative-free mode
        super().__init__(config)

        self.x = np.array(x0, dtype=float)
        # Initialize direction as unit vector for each dimension
        self.direction = np.eye(len(x0))[0]  # Start with first basis vector
        self.prev_x: Optional[np.ndarray] = None
        self.prev_fx: Optional[float] = None

    def get_current_x(self) -> np.ndarray:
        """Get current x value."""
        return self.x

    def _line_search(self, x: np.ndarray, direction: np.ndarray) -> float:
        """
        Perform line search along the given direction.

        Args:
            x: Current point (numpy array)
            direction: Search direction (numpy array)

        Returns:
            float: Optimal step size (alpha)
        """

        def objective(alpha: float) -> float:
            """Objective function for the line search."""
            return self.func(x + alpha * direction)

        result = minimize_scalar(objective)
        return result.x if result.success else 0.0

    def step(self) -> np.ndarray:
        """
        Perform one iteration of Powell's method.

        Returns:
            np.ndarray: Current approximation of the minimum
        """
        if self._converged:
            return self.x

        n = len(self.x)
        x_old = self.x.copy()
        fx_old = self.func(x_old)

        # Store best point
        x_best = self.x.copy()
        f_best = fx_old

        # Initialize set of directions as the basis vectors
        directions = np.eye(n)

        # Perform line minimization in each direction
        for i in range(n):
            self.direction = directions[i]
            alpha = self._line_search(self.x, self.direction)
            self.x = self.x + alpha * self.direction

            # Update best point if we found a better one
            f_new = self.func(self.x)
            if f_new < f_best:
                x_best = self.x.copy()
                f_best = f_new

        # Compute new direction as x_new - x_old
        new_direction = self.x - x_old
        direction_norm = np.linalg.norm(new_direction)

        # If the new direction is significant, try line search in this direction
        if direction_norm > 1e-10:
            self.direction = new_direction / direction_norm
            alpha = self._line_search(self.x, self.direction)
            self.x = self.x + alpha * self.direction
            f_new = self.func(self.x)

            if f_new < f_best:
                x_best = self.x.copy()
                f_best = f_new

        # Update to best point found
        self.x = x_best

        # Store iteration details
        details = {
            "x_old": str(x_old),
            "x_new": str(self.x),
            "f_old": fx_old,
            "f_new": f_best,
            "improvement": fx_old - f_best,
        }

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence using relative improvement and gradient (if available)
        rel_improvement = abs(f_best - fx_old) / (abs(fx_old) + 1e-10)

        # Use gradient for convergence check if available
        if not self.use_derivative_free and self.derivative is not None:
            grad_norm = np.linalg.norm(self.derivative(self.x))
            grad_converged = grad_norm < self.tol
        else:
            grad_converged = False

        # Check various convergence criteria
        if (
            rel_improvement < self.tol  # Function value not improving
            or np.linalg.norm(self.x - x_old)
            < self.tol * (1 + np.linalg.norm(self.x))  # Small step
            or grad_converged  # Gradient small (if available)
            or self.iterations >= self.max_iter  # Max iterations reached
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Powell's Method"


def powell_search(
    f: NumericalMethodConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for optimization
        x0: Initial guess
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (minimum, errors, iterations)
    """
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type="optimize", tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = PowellMethod(config, x0)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
