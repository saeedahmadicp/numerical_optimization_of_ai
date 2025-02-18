# algorithms/convex/newton_hessian.py

"""Newton-Hessian method for root finding."""

from typing import List, Tuple
import torch  # Used for automatic differentiation

from .protocols import BaseRootFinder, RootFinderConfig


class NewtonHessianMethod(BaseRootFinder):
    """Implementation of Newton-Hessian method."""

    def __init__(self, config: RootFinderConfig, x0: float):
        """
        Initialize Newton-Hessian method.

        Args:
            config: Configuration including function, derivative, and tolerances.
            x0: Initial guess for the root.

        Raises:
            ValueError: If derivative function is not provided in config.
        """
        # Ensure that a derivative is provided, as this method requires it.
        if config.derivative is None:
            raise ValueError("Newton-Hessian method requires derivative function")

        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Current approximation of the root.
        self._history: List[float] = []  # Record history of approximations.

        # Convert the initial guess to a torch tensor for automatic differentiation.
        self.x_tensor = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def _compute_hessian(self) -> float:
        """
        Compute the Hessian of |f(x)| using automatic differentiation.

        Returns:
            float: The Hessian value (second derivative) at the current point.
        """
        # Convert the current approximation to a tensor with gradient tracking.
        x = torch.tensor(self.x, requires_grad=True, dtype=torch.float64)

        # Compute the function value at the current point.
        fx = self.func(float(x))
        # Use absolute value to ensure non-negative measurement for the Hessian.
        fx_abs = abs(fx)

        # Compute the first derivative by creating a tensor for fx_abs and backpropagating.
        fx_tensor = torch.tensor(fx_abs, requires_grad=True)
        fx_tensor.backward()
        grad = x.grad  # This holds the first derivative.

        # If gradient computation failed, return a default value (avoiding division by zero later).
        if grad is None:
            return 1.0

        # Compute the second derivative by backpropagating through the gradient.
        grad.backward()
        hess = x.grad  # This now holds the Hessian (second derivative).

        # If Hessian computation fails, return a default value.
        if hess is None:
            return 1.0

        # Return the Hessian value as a float.
        return float(hess)

    def step(self) -> float:
        """
        Perform one iteration of Newton-Hessian method.

        Returns:
            float: Current approximation of the root.
        """
        # If convergence has been reached, return the current approximation.
        if self._converged:
            return self.x

        # Evaluate the function and its derivative at the current approximation.
        fx = self.func(self.x)
        dfx = self.derivative(self.x)  # type: ignore  # Already ensured derivative exists in __init__

        # Compute the Hessian using automatic differentiation.
        hess = self._compute_hessian()

        # Check for near-zero Hessian to avoid division by zero.
        if abs(hess) < 1e-10:
            self._converged = True
            return self.x

        # Perform the Newton-Hessian update: x = x - f(x)' / Hessian.
        try:
            self.x = self.x - dfx / hess
            self._history.append(self.x)  # Record the new approximation.
            self.iterations += 1  # Increment iteration count.
        except RuntimeError:
            # If an error occurs during update, mark convergence and return the current value.
            self._converged = True
            return self.x

        # Check for convergence based on function value tolerance or max iterations.
        if abs(fx) <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Newton-Hessian Method"


def newton_hessian_search(
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
    # Create a configuration instance with provided function, tolerance, and max iterations.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the Newton-Hessian method with the configuration and initial guess.
    method = NewtonHessianMethod(config, x0)

    errors = []  # List to record error values at each iteration.
    # Continue iterating until convergence.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define function f(x) = x^2 - 2, aiming to find sqrt(2)
#     def f(x):
#         return x**2 - 2
#
#     # Define its derivative f'(x) = 2x
#     def df(x):
#         return 2 * x
#
#     # Use the new protocol-based implementation.
#     config = RootFinderConfig(func=f, derivative=df, tol=1e-6)
#     method = NewtonHessianMethod(config, x0=1.5)
#
#     # Iterate until convergence, printing progress.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
