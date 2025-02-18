# algorithms/convex/elimination.py

"""Elimination method for finding roots."""

from typing import List, Tuple
import random

from .protocols import BaseRootFinder, RootFinderConfig


class EliminationMethod(BaseRootFinder):
    """Implementation of the elimination method."""

    def __init__(self, config: RootFinderConfig, a: float, b: float):
        """
        Initialize the elimination method.

        Args:
            config: Configuration including function, tolerances, and iteration limits.
            a: Left endpoint of the interval.
            b: Right endpoint of the interval.
        """
        # Initialize the base class to set up common attributes (e.g., function, tolerance).
        super().__init__(config)
        self.a = a  # Set the left endpoint of the search interval.
        self.b = b  # Set the right endpoint of the search interval.
        self.x = (a + b) / 2  # Start with the midpoint as the initial approximation.
        self._history: List[float] = []  # Record the history of approximations.

    def _elim_step(self, x1: float, x2: float) -> Tuple[float, float]:
        """
        Perform one step of the elimination method.

        Given two randomly generated test points, compare the absolute function values
        and decide which part of the interval is more promising.

        Args:
            x1: First test point.
            x2: Second test point.

        Returns:
            Tuple containing new interval endpoints (a', b').
        """
        # Evaluate the function at both test points and take their absolute values.
        f1, f2 = abs(self.func(x1)), abs(self.func(x2))

        # If the function value at x1 is lower, retain the left endpoint and use x2 as the new right.
        if f1 < f2:
            return self.a, x2
        # If the function value at x2 is lower, use x1 as the new left endpoint and retain the right.
        elif f1 > f2:
            return x1, self.b
        else:
            # If both have equal absolute values, narrow the interval to between x1 and x2.
            return x1, x2

    def step(self) -> float:
        """
        Perform one iteration of the elimination method.

        Returns:
            float: Current approximation of the root.
        """
        # If already converged, return the current approximation without further processing.
        if self._converged:
            return self.x

        # Randomly generate two test points within the current interval.
        x1 = random.uniform(self.a, self.b)
        x2 = random.uniform(self.a, self.b)

        # Use the elimination strategy to update the interval based on the test points.
        self.a, self.b = self._elim_step(x1, x2)

        # Compute the new approximation as the midpoint of the updated interval.
        self.x = (self.a + self.b) / 2
        # Record the current approximation in the history.
        self._history.append(self.x)
        # Increment the iteration counter.
        self.iterations += 1

        # Check convergence by considering either:
        # - The absolute error is within tolerance,
        # - The interval width is less than the tolerance,
        # - Or the maximum number of iterations has been reached.
        error = self.get_error()
        if (
            error <= self.tol
            or abs(self.b - self.a) < self.tol
            or self.iterations >= self.max_iter
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Elimination Method"


def elimination_search(
    f: RootFinderConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) to be used.
        a: Left endpoint of interval.
        b: Right endpoint of interval.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (x_min, errors, iterations) where:
         - x_min is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the total number of iterations performed.
    """
    # Create a configuration instance using provided function, tolerance, and iteration limits.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the elimination method with the specified interval.
    method = EliminationMethod(config, a, b)

    errors = []  # List to record error values for each iteration.
    # Iterate until the method converges.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Example function: f(x) = x^2 - 2, looking for sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6, max_iter=100)
#     method = EliminationMethod(config, a=1, b=2)
#
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper for backward compatibility:
#     x_min, errors, iters = elimination_search(f, 1, 2)
#     print(f"Found root (legacy): {x_min}")
