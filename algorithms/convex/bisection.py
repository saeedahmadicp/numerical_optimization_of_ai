# algorithms/convex/bisection.py

"""Bisection method for finding roots of continuous functions."""

from typing import List, Tuple

from .protocols import BaseRootFinder, RootFinderConfig


class BisectionMethod(BaseRootFinder):
    """Implementation of the bisection method."""

    def __init__(self, config: RootFinderConfig, a: float, b: float):
        """
        Initialize the bisection method.

        Args:
            config: Configuration including function and tolerances
            a: Left endpoint of interval
            b: Right endpoint of interval

        Raises:
            ValueError: If f(a) and f(b) have same sign, as a root is not guaranteed.
        """
        # Call the base class initializer to set up configuration and common attributes.
        super().__init__(config)

        # Evaluate the function at both endpoints to ensure they have opposite signs.
        fa, fb = self.func(a), self.func(b)
        if fa * fb >= 0:
            raise ValueError("Function must have opposite signs at interval endpoints")

        # Initialize the interval endpoints.
        self.a = a
        self.b = b
        # Set the initial approximation to the midpoint of the interval.
        self.x = (a + b) / 2
        # Keep a history of approximations.
        self._history: List[float] = []

    def step(self) -> float:
        """Perform one iteration of the bisection method.

        Returns:
            float: Current approximation of the root
        """
        # If the method has already converged, simply return the current approximation.
        if self._converged:
            return self.x

        # Compute the midpoint of the current interval.
        c = (self.a + self.b) / 2
        fc = self.func(c)  # Evaluate the function at the midpoint.

        # Determine in which sub-interval the root lies by checking sign change.
        if self.func(self.a) * fc < 0:
            # Root is between self.a and c, so update the right endpoint.
            self.b = c
        else:
            # Otherwise, root must be between c and self.b, so update the left endpoint.
            self.a = c

        # Update the current approximation to the new midpoint.
        self.x = (self.a + self.b) / 2
        # Record the new approximation in the history.
        self._history.append(self.x)
        # Increment the iteration counter.
        self.iterations += 1

        # Check convergence: if the function value at c is small enough or max iterations reached.
        if abs(fc) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Bisection Method"


def bisection_search(
    f: RootFinderConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function, depending on legacy usage)
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations)
    """
    # Create a configuration instance using provided function, tolerance, and iteration limit.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the BisectionMethod with the provided interval.
    method = BisectionMethod(config, a, b)

    errors = []  # List to store error values over iterations.
    # Loop until the method converges.
    while not method.has_converged():
        method.step()  # Take one bisection step.
        errors.append(method.get_error())  # Record the error after each step.

    # Return the final approximation, the error history, and the number of iterations performed.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Example usage
#     def f(x):
#         return x**2 - 2  # Function for which we want to find the root (sqrt(2))
#
#     # Using the new protocol-based implementation
#     config = RootFinderConfig(func=f, tol=1e-6, max_iter=100)
#     method = BisectionMethod(config, a=1, b=2)
#
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper for backward compatibility
#     root, errors, iters = bisection_search(f, 1, 2)
#     assert abs(root - 2**0.5) < 1e-6  # Ensure the computed root is close to sqrt(2)
