# algorithms/convex/regula_falsi.py

"""Regula falsi (false position) method for finding roots."""

from typing import List, Tuple

from .protocols import BaseRootFinder, RootFinderConfig


class RegulaFalsiMethod(BaseRootFinder):
    """Implementation of regula falsi method."""

    def __init__(self, config: RootFinderConfig, a: float, b: float):
        """
        Initialize regula falsi method.

        Args:
            config: Configuration including function and tolerances.
            a: Left endpoint of interval.
            b: Right endpoint of interval.

        Raises:
            ValueError: If f(a) and f(b) have the same sign.
        """
        # Initialize common attributes from the base class.
        super().__init__(config)

        # Evaluate the function at the endpoints.
        self.a = a
        self.b = b
        self.fa = self.func(a)
        self.fb = self.func(b)

        # Verify that the function values at a and b have opposite signs.
        if self.fa * self.fb >= 0:
            raise ValueError("Function must have opposite signs at interval endpoints")

        # Initialize current approximation as the weighted average (false position).
        self.x = self._weighted_average()

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _weighted_average(self) -> float:
        """
        Compute the weighted average of the endpoints based on their function values.

        Returns:
            float: The weighted average, i.e. the false position.
        """
        # Formula: x = (b * f(a) - a * f(b)) / (f(a) - f(b))
        return (self.b * self.fa - self.a * self.fb) / (self.fa - self.fb)

    def step(self) -> float:
        """
        Perform one iteration of the regula falsi method.

        Returns:
            float: Current approximation of the root.
        """
        # If the method has already converged, return the current approximation.
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Compute a new approximation using the false position formula.
        self.x = self._weighted_average()
        fx = self.func(self.x)

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "f(a)": self.fa,
            "f(b)": self.fb,
            "f(x)": fx,
            "weighted_avg": self.x,
        }

        # Update the bracketing interval based on the sign of f(x):
        # If the function changes sign between a and x, update b; otherwise update a.
        if self.fa * fx < 0:
            self.b = self.x
            self.fb = fx
            details["updated_end"] = "b"
        else:
            self.a = self.x
            self.fa = fx
            details["updated_end"] = "a"

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        # Increment the iteration count.
        self.iterations += 1

        # Check for convergence: if the function value is within tolerance or if max iterations reached.
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Regula Falsi Method"


def regula_falsi_search(
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
    # Create a configuration instance with the given parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the RegulaFalsiMethod with endpoints a and b.
    method = RegulaFalsiMethod(config, a, b)

    errors = []  # List to record error values for each iteration.
    # Iterate until the method converges.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, the error history, and the iteration count.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define function f(x) = x^2 - 2, aiming to find sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6)
#     method = RegulaFalsiMethod(config, a=1.0, b=2.0)
#
#     # Iterate until convergence, printing progress.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper:
#     root, errors, iters = regula_falsi_search(f, 1.0, 2.0)
#     print(f"Found root (legacy): {root}")
