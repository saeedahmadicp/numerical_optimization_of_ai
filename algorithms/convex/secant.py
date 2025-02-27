# algorithms/convex/secant.py

"""
Secant method for both root-finding and optimization.

The secant method approximates the derivative using finite differences
between two consecutive iterations, making it suitable when analytical
derivatives are unavailable or expensive to compute.

For root-finding, it approximates Newton's method as:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

For optimization, it applies the same formula to the derivative approximation:
    x_{n+1} = x_n - f'(x_n) * (x_n - x_{n-1}) / (f'(x_n) - f'(x_{n-1}))

where f'(x) is approximated using finite differences.
"""

from typing import List, Tuple, Optional, Callable, Union

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class SecantMethod(BaseNumericalMethod):
    """
    Implementation of secant method for root finding and optimization.

    For root-finding, the method approximates the derivative using finite differences
    to find zeros of a function.

    For optimization, the method approximates the second derivative using finite differences
    of the first derivative to find extrema of a function.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: float,
        x1: float,
        derivative: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize secant method.

        Args:
            config: Configuration including function and tolerances
            x0: First initial guess
            x1: Second initial guess
            derivative: Derivative function (required for optimization mode)

        Raises:
            ValueError: If method_type is 'optimize' but no derivative is provided
        """
        if config.method_type == "optimize" and derivative is None:
            raise ValueError(
                "Secant method for optimization requires derivative function"
            )

        super().__init__(config)

        self.x0: Optional[float] = x0
        self.x1: Optional[float] = x1
        self.derivative = derivative

        # For root-finding mode
        if self.method_type == "root":
            self.f0 = self.func(x0)
            self.f1 = self.func(x1)
        # For optimization mode
        else:
            if self.derivative is None:
                raise ValueError(
                    "Derivative function is required for optimization mode"
                )
            self.f0 = self.derivative(x0)
            self.f1 = self.derivative(x1)

        self.x = x1

        # For finite difference approximation in optimization mode when no derivative is provided
        self.h = 1e-7  # Step size for finite difference

        # Record initial state for iteration history
        if self.method_type == "optimize":
            initial_details = {
                "x0": self.x0,
                "x1": self.x1,
                "f(x0)": self.f0,
                "f(x1)": self.f1,
                "denominator": None,
                "step": None,
                "func(x0)": self.func(self.x0),
                "func(x1)": self.func(self.x1),
            }
            self.add_iteration(self.x0, self.x1, initial_details)

    def _approx_derivative(self, x: float) -> float:
        """
        Approximate the derivative at point x using finite differences.

        Args:
            x: Point at which to approximate the derivative

        Returns:
            float: Approximated derivative value
        """
        h = self.h
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of secant method.

        Returns:
            float: Current approximation of the root or extremum
        """
        if self._converged:
            return self.x

        x_old = self.x

        if self.x0 is None or self.x1 is None:
            self._converged = True
            return self.x

        if abs(self.f1 - self.f0) < 1e-10:
            self._converged = True
            return self.x

        # Calculate the next approximation
        x2 = self.x1 - self.f1 * (self.x1 - self.x0) / (self.f1 - self.f0)

        # Apply damping factor for optimization to improve convergence
        if self.method_type == "optimize":
            # Simple damping to improve convergence
            damping = 0.8
            x2 = self.x1 + damping * (x2 - self.x1)

            # Safeguard step size for better convergence
            max_step = 2.0 * abs(self.x1 - self.x0)
            if abs(x2 - self.x1) > max_step:
                x2 = self.x1 + (max_step if x2 > self.x1 else -max_step)

        # Calculate function value at new point
        if self.method_type == "root":
            f2 = self.func(x2)
        else:  # optimization mode
            f2 = self.derivative(x2) if self.derivative else self._approx_derivative(x2)

        # Record details for iteration history
        details = {
            "x0": self.x0,
            "x1": self.x1,
            "f(x0)": self.f0,
            "f(x1)": self.f1,
            "f(x2)": f2,
            "denominator": self.f1 - self.f0,
            "step": x2 - self.x1,
        }

        # If in optimization mode, also record function values
        if self.method_type == "optimize":
            details["func(x0)"] = self.func(self.x0)
            details["func(x1)"] = self.func(self.x1)
            details["func(x2)"] = self.func(x2)

        # Update for next iteration
        self.x0 = self.x1
        self.f0 = self.f1
        self.x1 = x2
        self.f1 = f2
        self.x = x2

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # For optimization, use stricter convergence criteria based on both derivative and function value
        if self.method_type == "optimize":
            func_change = abs(details["func(x2)"] - details["func(x1)"])
            if (
                func_change < 1e-10 and self.get_error() < self.tol
            ) or self.iterations >= self.max_iter:
                self._converged = True
        else:
            if self.get_error() <= self.tol or self.iterations >= self.max_iter:
                self._converged = True

        return self.x

    def get_error(self) -> float:
        """
        Calculate the error estimate for the current approximation.

        For root-finding: |f(x)|
        For optimization: |f'(x)|

        Returns:
            float: Error estimate
        """
        if self.method_type == "root":
            return abs(self.func(self.x))
        else:  # optimization mode
            if self.derivative:
                return abs(self.derivative(self.x))
            else:
                return abs(self._approx_derivative(self.x))

    @property
    def name(self) -> str:
        """Return the name of the method."""
        if self.method_type == "root":
            return "Secant Method (Root-Finding)"
        else:
            return "Secant Method (Optimization)"


def secant_search(
    f: Union[NumericalMethodConfig, Callable[[float], float]],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: str = "root",
    derivative: Optional[Callable[[float], float]] = None,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function)
        x0: First initial guess
        x1: Second initial guess
        tol: Error tolerance
        max_iter: Maximum number of iterations
        method_type: Type of problem ("root" or "optimize")
        derivative: Derivative function (required for optimization)

    Returns:
        Tuple of (solution, errors, iterations)
    """
    # If f is a callable rather than a config, create a config
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type=method_type, tol=tol, max_iter=max_iter
        )
    else:
        config = f

    # Create secant method instance
    method = SecantMethod(config, x0, x1, derivative=derivative)
    errors = []

    # Run until convergence
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
