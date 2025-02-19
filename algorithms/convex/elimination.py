# algorithms/convex/elimination.py

"""Elimination method for finding roots."""

from typing import List, Tuple

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
        
        Compare function values to determine which region likely contains the root.
        """
        # Evaluate the function at both test points
        f1, f2 = self.func(x1), self.func(x2)
        
        # Compare actual values instead of absolute values
        # This helps determine which side of the root we're on
        if f1 * f2 <= 0:  # If points straddle the root
            return (x1, x2) if x1 < x2 else (x2, x1)
        elif abs(f1) < abs(f2):
            return (self.a, x2) if x1 < x2 else (x1, self.b)
        else:
            return (x1, self.b) if x1 < x2 else (self.a, x2)

    def step(self) -> float:
        """
        Perform one iteration of the elimination method.

        Returns:
            float: Current approximation of the root.
        """
        # If already converged, return the current approximation without further processing.
        if self._converged:
            return self.x

        # Generate test points with better distribution
        interval_third = (self.b - self.a) / 3
        x1 = self.a + interval_third
        x2 = self.b - interval_third
        
        # Use the elimination strategy to update the interval
        self.a, self.b = self._elim_step(x1, x2)

        # Compute the new approximation as the midpoint of the updated interval.
        self.x = (self.a + self.b) / 2
        # Record the current approximation in the history.
        self._history.append(self.x)
        # Increment the iteration counter.
        self.iterations += 1

        # Update convergence check to include function value
        error = abs(self.func(self.x))
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

#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6, max_iter=100)
#     method = EliminationMethod(config, a=1, b=2)

#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")

#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")

#     # Or using the legacy wrapper for backward compatibility:
#     x_min, errors, iters = elimination_search(f, 1, 2)
#     print(f"Found root (legacy): {x_min}")
