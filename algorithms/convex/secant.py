# algorithms/convex/secant.py

"""
Secant method for both root-finding and optimization.

The secant method approximates the derivative using finite differences
between two consecutive iterations, making it suitable when analytical
derivatives are unavailable or expensive to compute.

Mathematical Basis:
----------------
For root-finding (finding x where f(x) = 0):
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

For optimization (finding x where f'(x) = 0):
    x_{n+1} = x_n - f'(x_n) * (x_n - x_{n-1}) / (f'(x_n) - f'(x_{n-1}))

where f'(x) is approximated using finite differences if not provided.

Convergence Properties:
--------------------
- Superlinear convergence with order approximately 1.618 (golden ratio)
- Faster than bisection and fixed-point iteration, but slower than Newton's method
- Does not require analytical derivatives
- More robust than Newton's method in some cases
- May fail when consecutive function evaluations are too close
- Requires two initial points instead of one
"""

from typing import List, Tuple, Optional, Callable, Union

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType
from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)


class SecantMethod(BaseNumericalMethod):
    """
    Implementation of secant method for root finding and optimization.

    The secant method uses finite differences between consecutive iterations
    to approximate derivatives, making it useful when analytical derivatives
    are unavailable or expensive to compute.

    Mathematical basis:
    - For root-finding: Uses secant approximation to find zeros of a function
    - For optimization: Uses secant approximation of the second derivative to find extrema

    Convergence properties:
    - Superlinear convergence with order approximately 1.618 (golden ratio)
    - Does not require analytical derivatives like Newton's method
    - Requires two initial points instead of one
    - May have stability issues if function values at consecutive points are too close

    Implementation features:
    - Handles both root-finding and optimization problems
    - Includes safeguards for avoiding division by near-zero values
    - Uses damping in optimization mode for better stability
    - Can approximate derivatives with finite differences when not provided
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
        self.x = x1  # Current point is the second initial guess

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

        # For finite difference approximation in optimization mode when no derivative is provided
        self.h = 1e-7  # Step size for finite difference

        # Safeguards for step size
        self.max_step_size = 2.0
        self.min_step_size = 1e-14

        # Damping factor for optimization to improve convergence
        self.damping = 0.8

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

        # Compute the descent direction using secant approximation
        direction = self.compute_descent_direction(self.x)

        # Compute step length with possible damping
        step_length = self.compute_step_length(self.x, direction)

        # Calculate the next approximation
        x2 = self.x1 + step_length * direction

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
            "direction": direction,
            "step_length": step_length,
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

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def compute_descent_direction(self, x: float) -> float:
        """
        Compute the secant descent direction.

        For root-finding: direction = -(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1})) * f(x_n)
        For optimization: direction = -(x_n - x_{n-1}) / (f'(x_n) - f'(x_{n-1})) * f'(x_n)

        Args:
            x: Current point (ignored, using stored x0 and x1)

        Returns:
            float: Secant direction
        """
        # Check if denominator is close to zero
        if abs(self.f1 - self.f0) < 1e-10:
            # If function values are very close, use a small default step
            if self.method_type == "root":
                f_val = self.func(self.x1)
                return -0.01 * (1.0 + abs(self.x1)) * (1.0 if f_val >= 0 else -1.0)
            else:
                df_val = (
                    self.derivative(self.x1)
                    if self.derivative
                    else self._approx_derivative(self.x1)
                )
                return -0.01 * (1.0 + abs(self.x1)) * (1.0 if df_val >= 0 else -1.0)

        # Calculate secant direction
        if self.method_type == "root":
            # Classical secant formula for root-finding
            direction = -(self.x1 - self.x0) / (self.f1 - self.f0) * self.f1
        else:
            # Secant formula for optimization (minimizing f)
            direction = -(self.x1 - self.x0) / (self.f1 - self.f0) * self.f1

        # Limit step size if too large
        if abs(direction) > self.max_step_size:
            direction = self.max_step_size * (1.0 if direction >= 0 else -1.0)

        return direction

    def compute_step_length(self, x: float, direction: float) -> float:
        """
        Compute the step length for the given direction.

        Args:
            x: Current point
            direction: Descent direction

        Returns:
            float: Step length
        """
        # If direction is too small, return zero
        if abs(direction) < self.min_step_size:
            return 0.0

        # For root-finding, typically use full step
        if self.method_type == "root":
            return 1.0

        # For optimization, apply damping factor
        if self.method_type == "optimize":
            # Use step length methods if specified
            if self.step_length_method:
                method = self.step_length_method
                params = self.step_length_params or {}

                if self.derivative is None:
                    raise ValueError("Derivative function is required for line search")

                # Use imported line search methods
                if method == "backtracking":
                    return backtracking_line_search(
                        self.func, self.derivative, x, direction, **(params or {})
                    )
                elif method == "wolfe":
                    return wolfe_line_search(
                        self.func, self.derivative, x, direction, **(params or {})
                    )
                elif method == "strong_wolfe":
                    return strong_wolfe_line_search(
                        self.func, self.derivative, x, direction, **(params or {})
                    )
                elif method == "goldstein":
                    return goldstein_line_search(
                        self.func, self.derivative, x, direction, **(params or {})
                    )
                elif method == "fixed":
                    return params.get("step_size", self.damping)

            # Default: use damping factor
            return self.damping

        return 1.0  # Default to full step

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

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For secant method, the theoretical rate is approximately 1.618 (golden ratio).

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # For secant method, we expect error_{n+1} ≈ C * error_n^φ
        # where φ ≈ 1.618 (golden ratio)
        phi = 1.618
        rate1 = (
            recent_errors[-1] / (recent_errors[-2] ** phi)
            if recent_errors[-2] > 0
            else 0
        )
        rate2 = (
            recent_errors[-2] / (recent_errors[-3] ** phi)
            if recent_errors[-3] > 0
            else 0
        )

        # Return average of recent rates
        return (rate1 + rate2) / 2

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
    method_type: MethodType = "root",
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
