# algorithms/convex/powell.py

"""Powell's conjugate direction method for root finding."""

from typing import List, Tuple, Optional
from scipy.optimize import minimize_scalar

from .protocols import BaseRootFinder, RootFinderConfig


class PowellMethod(BaseRootFinder):
    """Implementation of Powell's method."""

    def __init__(self, config: RootFinderConfig, x0: float):
        """
        Initialize Powell's method.

        Args:
            config: Configuration including function and tolerances.
            x0: Initial guess.
        """
        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Set the current approximation to the initial guess.
        self._history: List[float] = []  # To record the history of approximations.

        # For the 1D case, initialize the search direction as 1.0.
        self.direction = 1.0
        # Keep track of the previous point for updating the direction.
        self.prev_x: Optional[float] = None

    def _line_search(self, x: float, direction: float) -> float:
        """
        Perform line search along the given direction.

        Args:
            x: Current point.
            direction: Search direction.

        Returns:
            Optimal step size (alpha) found along the direction.
        """

        def objective(alpha: float) -> float:
            """Objective function for the line search; returns the absolute function value."""
            return abs(self.func(x + alpha * direction))

        # Use scipy's minimize_scalar to perform the line search along the given direction.
        result = minimize_scalar(objective)
        # If the line search failed, return 0.0 as the step size.
        if not result.success:
            return 0.0

        # Return the optimal step size.
        return result.x

    def step(self) -> float:
        """
        Perform one iteration of Powell's method.

        Returns:
            float: Current approximation of the root.
        """
        # If the method has already converged, return the current approximation.
        if self._converged:
            return self.x

        # Store the current point as the previous point.
        self.prev_x = self.x

        # Perform a line search along the current direction to determine the step size.
        alpha = self._line_search(self.x, self.direction)

        # Update the current approximation using the step size and direction.
        self.x += alpha * self.direction
        # Record the new approximation in the history.
        self._history.append(self.x)
        # Increment the iteration count.
        self.iterations += 1

        # Update the direction using the difference between the current and previous points.
        if self.prev_x is not None:
            # New direction is the difference (current - previous).
            self.direction = self.x - self.prev_x
            # Normalize the direction for consistency.
            if abs(self.direction) > 1e-10:
                self.direction /= abs(self.direction)

        # Check for convergence based on the function value, change in x, or iteration count.
        fx = self.func(self.x)
        if (
            abs(fx) <= self.tol
            or (self.prev_x is not None and abs(self.x - self.prev_x) < self.tol)
            or self.iterations >= self.max_iter
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Powell's Method"


def powell_search(
    f: RootFinderConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding.
        x0: Initial guess.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration instance with provided parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate Powell's method with the given initial guess.
    method = PowellMethod(config, x0)

    errors = []  # List to store error values per iteration.
    # Iterate until convergence is achieved.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define the function: f(x) = x^2 - 2, aiming to find sqrt(2).
#     def f(x):
#         return x**2 - 2
#
#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6)
#     method = PowellMethod(config, x0=1.5)
#
#     # Iterate until convergence, printing the progress.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper:
#     root, errors, iters = powell_search(f, 1.5)
#     print(f"Found root (legacy): {root}")
