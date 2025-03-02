# algorithms/convex/powell_quadratic.py

"""
Powell's Quadratic Interpolation Method for optimization and root-finding.

This is a 1D numerical method that fits a quadratic polynomial to three points
and uses it to estimate roots or optima of a function. Unlike gradient-based methods,
it relies only on function evaluations, making it suitable for cases where derivatives
are unavailable or expensive to compute. The method maintains a bracket of three points
and iteratively refines the estimate by replacing one of these points in each iteration.

Mathematical Basis:
----------------
For optimization (finding minimum of function f):
    1. Maintain a bracket of three points x₁ < x₂ < x₃
    2. Fit a quadratic polynomial p(x) = ax² + bx + c through the points (x₁,f(x₁)), (x₂,f(x₂)), (x₃,f(x₃))
    3. Calculate the minimum of p(x) at x = -b/(2a)
    4. Evaluate f at this minimum and update the bracket by replacing one point
    5. Repeat until convergence criteria are met

For root-finding (finding x where f(x) = 0):
    1. Maintain a bracket of three points, ideally with a sign change
    2. Fit a quadratic polynomial through the points
    3. Calculate where p(x) = 0 (apply quadratic formula)
    4. Evaluate f at this root estimate and update the bracket
    5. If updating creates a bracket with a sign change, convergence is accelerated

Convergence Properties:
--------------------
- Superlinear convergence for smooth functions (faster than linear methods)
- For optimization: More efficient than pure bracketing methods like golden section search
- For root-finding: Comparable to secant method but potentially more robust with bracketing
- Not dependent on derivatives, making it suitable for noisy or non-differentiable functions
- Safeguards against failure of quadratic fitting with fallback strategies
- May struggle with functions that are far from quadratic in shape

The implementation includes:
- Robust bracketing to ensure convergence
- Fallback mechanisms when quadratic fitting fails
- Support for both optimization and root-finding with appropriate error metrics
- Compatible with line search methods for optimization problems
- Appropriate termination criteria based on function value, bracket width, and iterations
"""

from typing import List, Tuple, Optional, Callable, Union
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType
from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)


class PowellMethod(BaseNumericalMethod):
    """
    Implementation of Powell's Quadratic Interpolation Method.

    This method fits a quadratic through three points and uses that to estimate
    the minimum (for optimization) or root (for root-finding) of a function.

    Mathematical guarantees:
    - For optimization tasks: Converges superlinearly on smooth functions
      with a well-defined minimum in the interval.
    - For root-finding tasks: Similar convergence properties to the secant method
      when the function is nearly quadratic near the root.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        a: float,
        b: float,
        c: Optional[float] = None,
        record_initial_state: bool = False,
    ):
        """
        Initialize Powell's Quadratic Interpolation Method.

        Args:
            config: Configuration including function and tolerances
            a: First point in the bracket
            b: Second point in the bracket
            c: Third point (if None, calculated as (a+b)/2)
            record_initial_state: Whether to record the initial state in history

        Raises:
            ValueError: If a >= b (invalid interval)
        """
        # Validate input
        if a >= b:
            raise ValueError(f"Invalid interval: [a={a}, b={b}]. Must have a < b.")

        # Initialize the base class
        super().__init__(config)

        # Set initial bracket points
        self.a = a
        self.b = b
        self.c = c if c is not None else (a + b) / 2

        # Ensure all three points are distinct and ordered
        if self.a > self.c:
            self.a, self.c = self.c, self.a
        if self.a > self.b:
            self.a, self.b = self.b, self.a
        if self.b > self.c:
            self.b, self.c = self.c, self.b

        # Evaluate function at bracket points
        self.fa = self.func(self.a)
        self.fb = self.func(self.b)
        self.fc = self.func(self.c)

        # For root-finding, check if there's a sign change in the interval
        if self.method_type == "root":
            # For root-finding, verify sign change (optional but recommended)
            if (
                self.fa * self.fc > 0
                and abs(self.fa) > self.tol
                and abs(self.fc) > self.tol
            ):
                # Only warn if neither endpoint is already close to a root
                print(
                    f"Warning: Function may not have a root in [{a}, {c}] as f({a})·f({c}) > 0"
                )

        # Set current point as the middle point initially
        self.x = self.b

        # Optionally record initial state
        if record_initial_state:
            initial_details = {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "f(a)": self.fa,
                "f(b)": self.fb,
                "f(c)": self.fc,
                "bracket_width": self.c - self.a,
                "method_type": self.method_type,
            }
            self.add_iteration(x_old=a, x_new=self.x, details=initial_details)

    def step(self) -> float:
        """
        Perform one iteration of Powell's method.

        Each iteration:
        1. Fits a quadratic through the three points
        2. Finds the minimum/root of the quadratic
        3. Updates the bracket with the new point
        4. Checks convergence criteria

        Returns:
            float: Current best approximation
        """
        # If already converged, return current approximation
        if self._converged:
            return self.get_current_x()

        # Store old x value for history
        x_old = self.get_current_x()

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "f(a)": self.fa,
            "f(b)": self.fb,
            "f(c)": self.fc,
            "bracket_width": self.c - self.a,
            "method_type": self.method_type,
        }

        # Compute descent direction and step length
        direction = self.compute_descent_direction(self.x)
        step_length = self.compute_step_length(self.x, direction)

        # Calculate the new point u
        u = self.x + step_length * direction

        # Evaluate function at the new point
        fu = self.func(u)
        details["u"] = u
        details["f(u)"] = fu
        details["direction"] = direction
        details["step_length"] = step_length

        # Add line search info for details
        line_search_method = self.step_length_method or "quadratic_fit"
        line_search_info = {
            "method": line_search_method,
            "initial_alpha": 1.0,
            "final_alpha": step_length,
            "f_old": self.func(self.x),
            "f_new": fu,
            "success": (
                fu < self.fb
                if self.method_type == "optimize"
                else abs(fu) < abs(self.fb)
            ),
        }
        details["line_search"] = line_search_info

        # Update the bracketing points
        if self.method_type == "optimize":
            # For optimization: maintain a bracket around the minimum
            if fu < self.fb:
                # If the new point is better than b, replace b
                if u < self.b:
                    self.c = self.b
                    self.fc = self.fb
                else:
                    self.a = self.b
                    self.fa = self.fb
                self.b = u
                self.fb = fu
            else:
                # If the new point is worse than b, replace a or c
                if u < self.b:
                    self.a = u
                    self.fa = fu
                else:
                    self.c = u
                    self.fc = fu
        else:
            # For root-finding: select the interval with a sign change
            if np.sign(fu) == np.sign(self.fa):
                # Replace a with u
                self.a = u
                self.fa = fu
            elif np.sign(fu) == np.sign(self.fc):
                # Replace c with u
                self.c = u
                self.fc = fu
            else:
                # If fu is close to zero or has different sign than both a and c
                # Replace the point with largest absolute function value
                if abs(self.fa) > abs(self.fc):
                    self.a = u
                    self.fa = fu
                else:
                    self.c = u
                    self.fc = fu

        # Ensure b is the middle point
        self._reorder_points()

        # Update the current approximation
        self.x = self.get_current_x()

        # Store iteration data
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence criteria
        bracket_width = self.c - self.a
        error = self.get_error()

        if (
            error <= self.tol  # Error within tolerance
            or bracket_width <= self.tol  # Bracket sufficiently small
            or self.iterations >= self.max_iter  # Max iterations reached
        ):
            self._converged = True

            # Add convergence reason to the last iteration
            last_iteration = self._history[-1]
            if error <= self.tol:
                last_iteration.details["convergence_reason"] = (
                    "function value within tolerance"
                )
            elif bracket_width <= self.tol:
                last_iteration.details["convergence_reason"] = (
                    "bracket width within tolerance"
                )
            else:
                last_iteration.details["convergence_reason"] = (
                    "maximum iterations reached"
                )

        return self.x

    def get_current_x(self) -> float:
        """
        Get current best approximation.

        For root-finding: Returns the point with smallest |f(x)|
        For optimization: Returns the point with smallest f(x)

        Returns:
            float: Current best approximation
        """
        if self.method_type == "root":
            # For root-finding, return point with smallest absolute function value
            points = [
                (self.a, abs(self.fa)),
                (self.b, abs(self.fb)),
                (self.c, abs(self.fc)),
            ]
            return min(points, key=lambda p: p[1])[0]
        else:
            # For optimization, return point with smallest function value
            points = [(self.a, self.fa), (self.b, self.fb), (self.c, self.fc)]
            return min(points, key=lambda p: p[1])[0]

    def compute_descent_direction(self, x: float) -> float:
        """
        Compute the descent direction based on the quadratic fit.

        For optimization:
        - Direction points toward the minimum of the fitted quadratic

        For root-finding:
        - Direction points toward the root of the fitted quadratic

        Args:
            x: Current point (typically in the bracket [a, c])

        Returns:
            float: Direction for the next step
        """
        # Get the quadratic fit point (minimum for optimization, root for root-finding)
        u = self._fit_quadratic()

        # If fitting failed, use a fallback strategy
        if u is None:
            if self.method_type == "optimize":
                # For optimization, move toward the point with lowest function value
                points = [(self.a, self.fa), (self.b, self.fb), (self.c, self.fc)]
                best_point = min(points, key=lambda p: p[1])[0]
                return best_point - x
            else:
                # For root-finding, use a secant-like approach if possible
                if abs(self.fb) > self.tol and abs(self.fc) > self.tol:
                    # Use secant direction between points with opposite signs if available
                    if self.fa * self.fb < 0:
                        return (self.a * self.fb - self.b * self.fa) / (
                            self.fb - self.fa
                        ) - x
                    elif self.fb * self.fc < 0:
                        return (self.b * self.fc - self.c * self.fb) / (
                            self.fc - self.fb
                        ) - x

                # Fallback to moving toward midpoint
                return (self.a + self.c) / 2 - x

        # Return direction from current x to the quadratic-suggested point
        return u - x

    def compute_step_length(self, x: float, direction: float) -> float:
        """
        Compute step length along the given direction.

        For Powell's method, we typically use the full step to the quadratic fit point,
        but for optimization problems we can also use standard line search methods.

        Args:
            x: Current point
            direction: Descent direction

        Returns:
            float: Step length
        """
        # If direction is very small, return 0
        if abs(direction) < 1e-10:
            return 0.0

        # Default step length is 1.0 (full step to the quadratic fit point)
        # This is appropriate for the pure Powell method
        if not self.step_length_method or self.step_length_method == "quadratic_fit":
            return 1.0

        # Only use other line search methods for optimization problems
        if self.method_type == "optimize":
            # For optimization, we can use standard line search methods
            method = self.step_length_method
            params = self.step_length_params or {}

            # Ensure we have a gradient function for line search methods
            grad_f = lambda x: self._estimate_gradient(x)

            # Dispatch to appropriate line search method
            if method == "fixed":
                return params.get("step_size", 1.0)

            elif method == "backtracking":
                return backtracking_line_search(
                    self.func,
                    grad_f,
                    x,
                    direction,
                    alpha_init=params.get("alpha_init", 1.0),
                    rho=params.get("rho", 0.5),
                    c=params.get("c", 1e-4),
                    max_iter=params.get("max_iter", 100),
                    alpha_min=params.get("alpha_min", 1e-16),
                )

            elif method == "wolfe":
                return wolfe_line_search(
                    self.func,
                    grad_f,
                    x,
                    direction,
                    alpha_init=params.get("alpha_init", 1.0),
                    c1=params.get("c1", 1e-4),
                    c2=params.get("c2", 0.9),
                    max_iter=params.get("max_iter", 25),
                    zoom_max_iter=params.get("zoom_max_iter", 10),
                    alpha_min=params.get("alpha_min", 1e-16),
                )

            elif method == "strong_wolfe":
                return strong_wolfe_line_search(
                    self.func,
                    grad_f,
                    x,
                    direction,
                    alpha_init=params.get("alpha_init", 1.0),
                    c1=params.get("c1", 1e-4),
                    c2=params.get("c2", 0.1),
                    max_iter=params.get("max_iter", 25),
                    zoom_max_iter=params.get("zoom_max_iter", 10),
                    alpha_min=params.get("alpha_min", 1e-16),
                )

            elif method == "goldstein":
                return goldstein_line_search(
                    self.func,
                    grad_f,
                    x,
                    direction,
                    alpha_init=params.get("alpha_init", 1.0),
                    c=params.get("c", 0.1),
                    max_iter=params.get("max_iter", 100),
                    alpha_min=params.get("alpha_min", 1e-16),
                    alpha_max=params.get("alpha_max", 1e10),
                )

        # If no standard method matched or not in optimization mode, use full step
        return 1.0

    def _fit_quadratic(self) -> Optional[float]:
        """
        Fit a quadratic through the three points and find its minimum or root.

        For optimization: Find the minimum of the quadratic
        For root-finding: Find where the quadratic equals zero

        Returns:
            Optional[float]: Position of the minimum/root, or None if calculation fails
        """
        # Points must be distinct to avoid division by zero
        if (
            abs(self.a - self.b) < 1e-10
            or abs(self.b - self.c) < 1e-10
            or abs(self.a - self.c) < 1e-10
        ):
            return None

        # For optimization: find the minimum of the quadratic
        if self.method_type == "optimize":
            # Compute coefficients for the quadratic fit
            denom = (self.a - self.b) * (self.a - self.c) * (self.b - self.c)
            A = (
                self.c * (self.fb - self.fa)
                + self.b * (self.fa - self.fc)
                + self.a * (self.fc - self.fb)
            ) / denom

            # If A <= 0, the quadratic has no minimum
            if abs(A) < 1e-10 or A < 0:
                return None

            B = (
                self.c**2 * (self.fa - self.fb)
                + self.b**2 * (self.fc - self.fa)
                + self.a**2 * (self.fb - self.fc)
            ) / denom

            # Calculate the minimum of the quadratic: -B/(2A)
            return -B / (2 * A)

        # For root-finding: find where the quadratic equals zero
        else:
            # Fit a quadratic: p(x) = A(x-b)(x-c) + B(x-a)(x-c) + C(x-a)(x-b)
            # where p(a)=fa, p(b)=fb, p(c)=fc
            A = self.fa / ((self.a - self.b) * (self.a - self.c))
            B = self.fb / ((self.b - self.a) * (self.b - self.c))
            C = self.fc / ((self.c - self.a) * (self.c - self.b))

            # Convert to standard form: p(x) = ax² + bx + c
            a = A + B + C
            b = -(A * (self.b + self.c) + B * (self.a + self.c) + C * (self.a + self.b))
            c = A * self.b * self.c + B * self.a * self.c + C * self.a * self.b

            # Solve the quadratic using quadratic formula
            # The discriminant must be positive for real roots
            discriminant = b**2 - 4 * a * c

            if discriminant < 0 or abs(a) < 1e-10:
                return None

            # Calculate both roots
            r1 = (-b + np.sqrt(discriminant)) / (2 * a)
            r2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Choose the root within our interval [a, c]
            if self.a <= r1 <= self.c:
                return r1
            elif self.a <= r2 <= self.c:
                return r2
            else:
                # If neither root is in the interval, return the closest one
                if abs(r1 - self.b) < abs(r2 - self.b):
                    return r1
                else:
                    return r2

    def _reorder_points(self):
        """
        Reorder points to ensure a < b < c.

        For optimization: Also ensures f(b) <= f(a) and f(b) <= f(c)
        """
        # Create list of points and sort them
        points = [(self.a, self.fa), (self.b, self.fb), (self.c, self.fc)]
        points.sort(key=lambda p: p[0])  # Sort by x-coordinate

        # Unpack sorted points
        self.a, self.fa = points[0]
        self.b, self.fb = points[1]
        self.c, self.fc = points[2]

        if self.method_type == "optimize":
            # For optimization, ensure b is the point with the lowest function value
            best_point = min(points, key=lambda p: p[1])

            # If the best point isn't already b, rearrange to make it b
            if best_point[0] != self.b:
                if best_point[0] == self.a:
                    # Shift points left
                    self.c, self.fc = self.b, self.fb
                    self.b, self.fb = self.a, self.fa
                else:  # best_point[0] == self.c
                    # Shift points right
                    self.a, self.fa = self.b, self.fb
                    self.b, self.fb = self.c, self.fc

    def _estimate_gradient(self, x: float) -> float:
        """
        Estimate gradient at point x using finite differences.

        This is used for the line search methods.

        Args:
            x: Point at which to estimate gradient

        Returns:
            float: Estimated gradient
        """
        if self.derivative is not None:
            return self.derivative(x)
        else:
            return self.estimate_derivative(x)

    def get_error(self) -> float:
        """
        Get error at current point based on method type.

        For root-finding:
            - Error = |f(x)|, which measures how close we are to f(x) = 0

        For optimization:
            - Error estimate is based on bracket width and function value change
            - In practice, gradient or function differences could be used

        Returns:
            float: Current error estimate
        """
        x = self.get_current_x()

        if self.method_type == "root":
            # For root-finding, error is how close f(x) is to zero
            return abs(self.func(x))
        else:
            # For optimization, we can use bracket width as error measure
            # or we can use the base class method which will estimate gradient
            if self.derivative is not None:
                return abs(self.derivative(x))  # If derivative available
            else:
                # Default to gradient estimation from base class
                return super().get_error()

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For Powell's method, the theoretical rate is superlinear
        for smooth functions.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in errors):
            return 0.0  # Exact convergence

        # Estimate rates using consecutive pairs of errors
        rates = []
        for i in range(len(errors) - 1):
            if errors[i] > 0:
                rate = errors[i + 1] / errors[i]
                rates.append(rate)

        # Return mean of observed rates
        return sum(rates) / len(rates) if rates else None

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return f"Powell's Quadratic Interpolation {'Root-Finding' if self.method_type == 'root' else 'Optimization'} Method"


def powell_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    a: float,
    b: float,
    c: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "optimize",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for Powell's Quadratic Interpolation Method.

    This function provides a simpler interface to Powell's method for
    users who don't need the full object-oriented functionality.

    Mathematical guarantee:
    - For optimization: Converges superlinearly on smooth functions with a minimum
    - For root-finding: Similar convergence to the secant method for smooth functions

    Args:
        f: Function or configuration (if function, it's wrapped in a config)
        a: Left endpoint of interval
        b: Right endpoint of interval
        c: Optional third point (if None, calculated as (a+b)/2)
        tol: Error tolerance
        max_iter: Maximum number of iterations
        method_type: Type of problem ("root" or "optimize")

    Returns:
        Tuple of (solution, errors, iterations)
    """
    # If f is a function rather than a config, create a config
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type=method_type, tol=tol, max_iter=max_iter
        )
    else:
        config = f

    # Create and run the method
    method = PowellMethod(config, a, b, c)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.get_current_x(), errors, method.iterations
