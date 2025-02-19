# algorithms/convex/secant.py

"""Secant method for finding roots."""

from typing import List, Tuple, Optional

from .protocols import BaseRootFinder, RootFinderConfig


class SecantMethod(BaseRootFinder):
    """Implementation of secant method."""

    def __init__(self, config: RootFinderConfig, x0: float, x1: float):
        """
        Initialize secant method.

        Args:
            config: Configuration including function and tolerances.
            x0: First initial guess.
            x1: Second initial guess.
        """
        # Initialize common attributes from the base class.
        super().__init__(config)

        # Store the two initial guesses and compute their function values.
        self.x0: Optional[float] = x0
        self.x1: Optional[float] = x1
        self.f0 = self.func(x0)
        self.f1 = self.func(x1)

        # Set current approximation to the second guess.
        self.x = x1

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of secant method.

        Returns:
            float: Current approximation of the root.
        """
        # If convergence has been reached, return the current approximation.
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Ensure both previous points are available.
        if self.x0 is None or self.x1 is None:
            self._converged = True
            return self.x

        # Check for near-zero denominator to avoid division by zero.
        if abs(self.f1 - self.f0) < 1e-10:
            self._converged = True
            return self.x

        # Compute the next approximation using the secant formula:
        # x2 = x1 - f(x1)*(x1 - x0) / (f(x1) - f(x0))
        x2 = self.x1 - self.f1 * (self.x1 - self.x0) / (self.f1 - self.f0)
        f2 = self.func(x2)

        # Store iteration details
        details = {
            "x0": self.x0,
            "x1": self.x1,
            "f(x0)": self.f0,
            "f(x1)": self.f1,
            "f(x2)": f2,
            "denominator": self.f1 - self.f0,
            "step": x2 - self.x1,
        }

        # Update stored values: shift x1->x0 and x2 becomes new x1.
        self.x0 = self.x1
        self.f0 = self.f1
        self.x1 = x2
        self.f1 = f2
        self.x = x2  # Update the current approximation.

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        self.iterations += 1

        # Check convergence based on the function value at new approximation or iteration count.
        if abs(f2) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Secant Method"


def secant_search(
    f: RootFinderConfig,
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding.
        x0: First initial guess.
        x1: Second initial guess.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration instance using provided parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the SecantMethod with the given initial guesses.
    method = SecantMethod(config, x0, x1)

    errors = []  # List to record error values at each iteration.
    # Run iterations until convergence is reached.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and the number of iterations.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define function f(x) = x^2 - 2, aiming to find sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6)
#     method = SecantMethod(config, x0=1.0, x1=2.0)
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
#     root, errors, iters = secant_search(f, 1.0, 2.0)
#     print(f"Found root (legacy): {root}")
