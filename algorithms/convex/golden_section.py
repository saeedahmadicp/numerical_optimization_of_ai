# algorithms/convex/golden_section.py

"""Golden section search method for root finding."""

from typing import List, Tuple
import math

from .protocols import BaseRootFinder, RootFinderConfig


class GoldenSectionMethod(BaseRootFinder):
    """Implementation of golden section method."""

    def __init__(self, config: RootFinderConfig, a: float, b: float):
        """
        Initialize golden section method.

        Args:
            config: Configuration including function and tolerances.
            a: Left endpoint of interval.
            b: Right endpoint of interval.
        """
        # Initialize common attributes from the base class.
        super().__init__(config)

        # Store the endpoints of the interval.
        self.a = a
        self.b = b
        # Set current approximation to the midpoint of the interval.
        self.x = (a + b) / 2
        # Initialize history of approximations.
        self._history: List[float] = []

        # Calculate golden ratio constants.
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio (~1.618034)
        self.tau = 1 / self.phi  # Inverse golden ratio (~0.618034)

        # Initialize test points within the interval using the golden ratio.
        self.x1 = a + (1 - self.tau) * (b - a)
        self.x2 = a + self.tau * (b - a)
        # Evaluate the function at these test points (using absolute value for root finding).
        self.f1 = abs(self.func(self.x1))
        self.f2 = abs(self.func(self.x2))

    def step(self) -> float:
        """
        Perform one iteration of golden section method.

        Returns:
            float: Current approximation of the root.
        """
        # If convergence has already been achieved, return the current approximation.
        if self._converged:
            return self.x

        # Handle the rare case when the function values are nearly equal (to avoid numerical issues).
        if abs(self.f1 - self.f2) < 1e-10:
            self.x2 += 1e-6  # Small perturbation to break tie.
            self.f2 = abs(self.func(self.x2))

        # Update the interval based on comparing function values at test points.
        if self.f1 < self.f2:
            # If the left test point is better, move the right endpoint to x2.
            self.b = self.x2
            # Shift x1 to the right.
            self.x2 = self.x1
            self.f2 = self.f1
            # Compute a new left test point.
            self.x1 = self.a + (1 - self.tau) * (self.b - self.a)
            self.f1 = abs(self.func(self.x1))
        else:
            # Otherwise, if the right test point is better, move the left endpoint to x1.
            self.a = self.x1
            # Shift x2 to the left.
            self.x1 = self.x2
            self.f1 = self.f2
            # Compute a new right test point.
            self.x2 = self.a + self.tau * (self.b - self.a)
            self.f2 = abs(self.func(self.x2))

        # Update the current approximation as the midpoint of the updated interval.
        self.x = (self.a + self.b) / 2
        # Record the new approximation.
        self._history.append(self.x)
        # Increment the iteration counter.
        self.iterations += 1

        # Check for convergence: if the interval width is within tolerance or max iterations reached.
        if abs(self.b - self.a) < self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Golden Section Method"


def golden_section_search(
    f: RootFinderConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding.
        a: Left endpoint of interval.
        b: Right endpoint of interval.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration instance from the given parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the GoldenSectionMethod with the interval [a, b].
    method = GoldenSectionMethod(config, a, b)

    errors = []  # List to store error values for each iteration.
    # Iterate until the method converges.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and the number of iterations.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Example usage:
#     def f(x):
#         return x**2 - 2  # Function: f(x) = x^2 - 2, finding sqrt(2)
#
#     # Using the new protocol-based implementation.
#     config = RootFinderConfig(func=f, tol=1e-6)
#     method = GoldenSectionMethod(config, a=1.0, b=2.0)
#
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper.
#     root, errors, iters = golden_section_search(f, 1.0, 2.0)
#     print(f"Found root (legacy): {root}")
