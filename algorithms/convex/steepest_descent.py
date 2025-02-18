# algorithms/convex/steepest_descent.py

"""Steepest descent method for root finding."""

from typing import List, Tuple
import numpy as np

from .protocols import BaseRootFinder, RootFinderConfig


class SteepestDescentMethod(BaseRootFinder):
    """Implementation of steepest descent method."""

    def __init__(self, config: RootFinderConfig, x0: float, alpha: float = 0.1):
        """
        Initialize steepest descent method.

        Args:
            config: Configuration including function, derivative, and tolerances.
            x0: Initial guess.
            alpha: Learning rate (step size).

        Raises:
            ValueError: If derivative function is not provided in config.
        """
        # Ensure the derivative is provided, as it is needed for the gradient.
        if config.derivative is None:
            raise ValueError("Steepest descent method requires derivative function")

        # Initialize common attributes using the base class.
        super().__init__(config)
        self.x = x0  # Current approximation of the root.
        self.alpha = alpha  # Base learning rate (initial step size).
        self._history: List[float] = []  # To record the history of approximations.

    def _backtracking_line_search(self, p: float) -> float:
        """
        Perform backtracking line search to find a suitable step size.

        Args:
            p: Search direction (typically the negative gradient).

        Returns:
            A step size that satisfies the Armijo condition.
        """
        c = 1e-4  # Armijo condition parameter.
        rho = 0.5  # Factor to reduce the step size.
        alpha = self.alpha  # Start with the initial learning rate.

        fx = abs(self.func(self.x))
        # For scalar problems, adjust the derivative with the sign of f(x).
        grad_fx = np.sign(self.func(self.x)) * self.derivative(self.x)  # type: ignore

        # Backtracking: reduce alpha until the Armijo condition is met.
        while abs(self.func(self.x + alpha * p)) > fx + c * alpha * grad_fx * p:
            alpha *= rho
            if alpha < 1e-10:  # Prevent alpha from becoming too small.
                break

        return alpha

    def step(self) -> float:
        """
        Perform one iteration of the steepest descent method.

        Returns:
            float: Current approximation of the root.
        """
        # If the method has already converged, return the current approximation.
        if self._converged:
            return self.x

        # Evaluate the function at the current approximation.
        fx = self.func(self.x)
        # Compute the gradient using the derivative and adjust sign for root finding.
        grad = np.sign(fx) * self.derivative(self.x)  # type: ignore

        # The search direction is the negative gradient.
        p = -grad

        # Determine an appropriate step size using backtracking line search.
        alpha = self._backtracking_line_search(p)

        # Update the current approximation using the step size and search direction.
        self.x += alpha * p
        self._history.append(self.x)  # Record the new approximation.
        self.iterations += 1  # Increment iteration count.

        # Check for convergence:
        # If the absolute function value is within tolerance or maximum iterations reached.
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Steepest Descent Method"


def steepest_descent_search(
    f: RootFinderConfig,
    x0: float,
    alpha: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for the root finder.
        x0: Initial guess.
        alpha: Learning rate.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations), where:
         - root is the final approximation,
         - errors is a list of error values for each iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration instance from the provided parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the steepest descent method with the configuration and initial guess.
    method = SteepestDescentMethod(config, x0, alpha)

    errors = []  # List to store error values per iteration.
    # Run iterations until convergence.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define the function: f(x) = x^2 - 2, to find sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     # Define its derivative: f'(x) = 2x
#     def df(x):
#         return 2 * x
#
#     # Setup configuration with function, derivative, and tolerance.
#     config = RootFinderConfig(func=f, derivative=df, tol=1e-6)
#     # Instantiate the method with an initial guess and learning rate.
#     method = SteepestDescentMethod(config, x0=1.5, alpha=0.1)
#
#     # Iterate until convergence.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
