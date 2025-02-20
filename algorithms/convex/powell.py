# algorithms/convex/powell.py

"""Powell's conjugate direction method for derivative-free optimization."""

from typing import List, Tuple, Optional
from scipy.optimize import minimize_scalar

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class PowellMethod(BaseNumericalMethod):
    """Implementation of Powell's method for derivative-free optimization."""

    def __init__(self, config: NumericalMethodConfig, x0: float):
        """
        Initialize Powell's method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial guess

        Raises:
            ValueError: If method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError("Powell's method can only be used for optimization")

        config.use_derivative_free = True  # Ensure derivative-free mode
        super().__init__(config)

        self.x = x0
        self.direction = 1.0
        self.prev_x: Optional[float] = None
        self.prev_fx: Optional[float] = None

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _line_search(self, x: float, direction: float) -> float:
        """
        Perform line search along the given direction.

        Args:
            x: Current point
            direction: Search direction

        Returns:
            float: Optimal step size (alpha)
        """

        def objective(alpha: float) -> float:
            """Objective function for the line search."""
            return self.func(x + alpha * direction)

        result = minimize_scalar(objective)
        return result.x if result.success else 0.0

    def step(self) -> float:
        """
        Perform one iteration of Powell's method.

        Returns:
            float: Current approximation of the minimum
        """
        if self._converged:
            return self.x

        x_old = self.x
        fx_old = self.func(x_old)

        # Store current point and function value
        self.prev_x = self.x
        self.prev_fx = fx_old

        # Perform line search
        alpha = self._line_search(self.x, self.direction)

        # Update current approximation
        self.x += alpha * self.direction

        # Update the direction
        if self.prev_x is not None:
            new_direction = self.x - self.prev_x
            if abs(new_direction) > 1e-10:
                new_direction /= abs(new_direction)
            self.direction = new_direction

        details = {
            "alpha": alpha,
            "direction": self.direction,
            "prev_x": self.prev_x,
            "prev_fx": self.prev_fx,
            "line_search": {
                "start": x_old,
                "step_size": alpha,
                "direction": self.direction,
            },
        }

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence using relative improvement
        fx_new = self.func(self.x)
        rel_improvement = abs(fx_new - fx_old) / (abs(fx_old) + 1e-10)

        if (
            rel_improvement < self.tol
            or (self.prev_x is not None and abs(self.x - self.prev_x) < self.tol)
            or self.iterations >= self.max_iter
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
