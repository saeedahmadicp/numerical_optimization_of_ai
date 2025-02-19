# algorithms/convex/newton.py

"""Newton-Raphson method for finding roots of differentiable functions."""

from typing import List, Tuple

from .protocols import BaseRootFinder, RootFinderConfig


class NewtonMethod(BaseRootFinder):
    """Implementation of Newton's method."""

    def __init__(self, config: RootFinderConfig, x0: float):
        """
        Initialize Newton's method.

        Args:
            config: Configuration including function, derivative, tolerances, etc.
            x0: Initial guess for the root.

        Raises:
            ValueError: If derivative function is not provided in config.
        """
        # Ensure a derivative is provided, as Newton's method requires it.
        if config.derivative is None:
            raise ValueError("Newton's method requires derivative function")

        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Set the current approximation to the initial guess.
        self._history: List[float] = []  # Record the history of approximations.

    def step(self) -> float:
        """
        Perform one iteration of Newton's method.

        Returns:
            float: The current approximation of the root.
        """
        # If convergence has already been reached, return the current approximation.
        if self._converged:
            return self.x

        # Evaluate the function at the current approximation.
        fx = self.func(self.x)
        # Evaluate the derivative at the current approximation.
        dfx = self.derivative(self.x)  # type: ignore  # Already ensured derivative exists in __init__

        # Avoid division by zero: if derivative is nearly zero, stop iterations.
        if abs(dfx) < 1e-10:
            self._converged = True
            return self.x

        # Update the approximation using the Newton-Raphson formula.
        self.x = self.x - fx / dfx
        # Record the new approximation.
        self._history.append(self.x)
        # Increment the iteration counter.
        self.iterations += 1

        # Check convergence: if function value is within tolerance or maximum iterations reached.
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Newton's Method"


def newton_search(
    f: RootFinderConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or callable) for root finding.
        x0: Initial guess.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations).
    """
    # If f is a callable (old style), create a derivative function using finite differences
    if callable(f):

        def derivative(x: float, h: float = 1e-7) -> float:
            return (f(x + h) - f(x)) / h

        config = RootFinderConfig(
            func=f, derivative=derivative, tol=tol, max_iter=max_iter
        )
    else:
        # f is already a RootFinderConfig
        config = f

    method = NewtonMethod(config, x0)

    errors = []
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define the function for which to find the root (e.g., x^2 - 2 for sqrt(2))
#     def f(x):
#         return x**2 - 2

#     # Define its derivative (2x)
#     def df(x):
#         return 2 * x

#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, derivative=df, tol=1e-6)
#     method = NewtonMethod(config, x0=1.5)

#     # Run iterations until convergence
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")

#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")

#     # Alternatively, using the legacy wrapper:
#     root, errors, iters = newton_search(f, 1.5)
#     print(f"Found root (legacy): {root}")
