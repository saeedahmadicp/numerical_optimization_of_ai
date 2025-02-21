# algorithms/convex/fibonacci.py

"""Fibonacci search method for finding roots."""

from typing import List, Tuple

from .protocols import BaseRootFinder, RootFinderConfig


def fib_generator(n: int) -> List[int]:
    """
    Generate Fibonacci sequence up to n terms.

    Args:
        n: Number of terms to generate

    Returns:
        List of first n Fibonacci numbers
    """
    # Return an empty list if no terms are requested.
    if n < 1:
        return []
    # If only one term is requested, return [1].
    elif n == 1:
        return [1]

    # Initialize the sequence with the first two Fibonacci numbers.
    fib = [1, 1]
    # Generate remaining terms using the recurrence relation.
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


class FibonacciMethod(BaseRootFinder):
    """
    Implementation of the Fibonacci search method.

    This method uses Fibonacci ratios to successively narrow down the interval where a
    root is located.
    """

    def __init__(self, config: RootFinderConfig, a: float, b: float, n_terms: int = 30):
        """
        Initialize the Fibonacci search method.

        Args:
            config: Configuration including function, tolerances, etc.
            a: Left endpoint of interval.
            b: Right endpoint of interval.
            n_terms: Number of Fibonacci terms to use in the search.
        """
        # Initialize the base class to set up common attributes.
        super().__init__(config)
        self.a = a  # Left endpoint of the interval.
        self.b = b  # Right endpoint of the interval.
        self.x = (a + b) / 2  # Initial approximation set as the midpoint.

        # Generate the Fibonacci sequence needed for the search.
        self.fib = fib_generator(n_terms + 1)
        self.n_terms = n_terms
        self.current_term = n_terms  # This will be decreased with each iteration.

        # Initialize test points using Fibonacci ratios.
        # These ratios determine how the interval is divided.
        self.x1 = a + self.fib[self.n_terms - 2] / self.fib[self.n_terms] * (b - a)
        self.x2 = a + self.fib[self.n_terms - 1] / self.fib[self.n_terms] * (b - a)

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _update_points(self):
        """
        Update test points using Fibonacci ratios.

        This helper method recalculates x1 and x2 based on the current interval [a, b]
        and the remaining Fibonacci numbers.
        """
        self.x1 = self.a + self.fib[self.current_term - 3] / self.fib[
            self.current_term - 1
        ] * (self.b - self.a)
        self.x2 = self.a + self.fib[self.current_term - 2] / self.fib[
            self.current_term - 1
        ] * (self.b - self.a)

    def step(self) -> float:
        """
        Perform one iteration of the Fibonacci search method.

        Returns:
            float: Current approximation of the root.
        """
        # If convergence has been achieved, return the current approximation.
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Evaluate the function at the two test points.
        f1, f2 = abs(self.func(self.x1)), abs(self.func(self.x2))

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "x1": self.x1,
            "x2": self.x2,
            "f(x1)": f1,
            "f(x2)": f2,
            "fib_term": self.current_term,
        }

        # Update the search interval based on which test point gives a smaller function value.
        if f1 < f2:
            # If f1 is lower, the new interval becomes [a, x2].
            self.b = self.x2
            # Shift x2 to x1, and compute a new x1.
            self.x2 = self.x1
            self.x1 = self.a + self.fib[self.current_term - 3] / self.fib[
                self.current_term - 1
            ] * (self.b - self.a)
        else:
            # Otherwise, the new interval becomes [x1, b].
            self.a = self.x1
            # Shift x1 to x2, and compute a new x2.
            self.x1 = self.x2
            self.x2 = self.a + self.fib[self.current_term - 2] / self.fib[
                self.current_term - 1
            ] * (self.b - self.a)

        # Update the current approximation to be the midpoint of the new interval.
        self.x = (self.a + self.b) / 2

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        # Increment the iteration counter.
        self.iterations += 1
        # Decrement the current Fibonacci term index.
        self.current_term -= 1

        # Check for convergence based on error, interval width, iteration count, or Fibonacci term exhaustion.
        error = self.get_error()
        if (
            error <= self.tol
            or abs(self.b - self.a) < self.tol
            or self.iterations >= self.max_iter
            or self.current_term < 3
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Fibonacci Method"


def fibonacci_search(
    f: RootFinderConfig,
    a: float,
    b: float,
    n_terms: int = 30,
    tol: float = 1e-6,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for the root finder.
        a: Left endpoint of the interval.
        b: Right endpoint of the interval.
        n_terms: Number of Fibonacci terms to use.
        tol: Error tolerance.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values for each iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration object using the provided function, tolerance, etc.
    config = RootFinderConfig(func=f, tol=tol)
    # Instantiate the FibonacciMethod with the specified parameters.
    method = FibonacciMethod(config, a, b, n_terms)

    errors = []  # Initialize a list to record errors for each iteration.
    # Continue iterating until convergence is reached.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, the error history, and the iteration count.
    return method.x, errors, method.iterations
