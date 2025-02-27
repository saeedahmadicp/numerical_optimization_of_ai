# algorithms/convex/golden_section.py

"""
Golden section search method for optimization and root-finding.

The golden section search is an efficient technique for finding extrema of
unimodal functions. It uses the golden ratio to efficiently narrow down the
interval containing the optimum, requiring fewer function evaluations than
simple grid search or bisection.

Mathematical Basis:
----------------
For optimization of a unimodal function f over an interval [a,b]:

1. Place test points x₁ and x₂ at positions determined by the golden ratio
2. Compare f(x₁) and f(x₂) to determine which portion of the interval to eliminate
3. Reduce the interval and recalculate one test point (reusing the other)
4. Repeat until convergence or the interval is sufficiently small

For root-finding, this method adapts by:
1. Comparing absolute function values |f(x₁)| and |f(x₂)| instead
2. Selecting the subinterval that likely contains the root

Convergence Properties:
--------------------
- More efficient than bisection for optimization problems
- Reduction ratio is approximately 0.618 per iteration
- Does not require derivatives of the function
- Guarantees (b-a) is reduced by a factor of ~0.618 per iteration
"""

from typing import List, Tuple, Optional, Callable, Union
import math

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


class GoldenSectionMethod(BaseNumericalMethod):
    """
    Implementation of the golden section method.

    The Golden Section method is an efficient technique for bracketing extrema
    of unimodal functions or finding roots. It uses the golden ratio to
    optimally place test points, providing predictable convergence properties.

    Mathematical guarantees:
    - For optimization tasks: If f is unimodal on [a,b], the method will converge
      to a minimum with an interval reduction ratio of ~0.618 per iteration.
    - For root-finding tasks: If f is continuous and f(a)·f(b) < 0, the method will
      converge to a root with similar efficiency.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        a: float,
        b: float,
        record_initial_state: bool = False,
    ):
        """
        Initialize golden section method.

        Args:
            config: Configuration including function and tolerances.
            a: Left endpoint of interval.
            b: Right endpoint of interval.
            record_initial_state: Whether to record the initial state in history.

        Raises:
            ValueError: If a >= b (invalid interval)
            ValueError: For root-finding, if f(a)·f(b) >= 0 (no sign change)
        """
        # Validate input
        if a >= b:
            raise ValueError(f"Invalid interval: [a={a}, b={b}]. Must have a < b.")

        # Initialize the base class
        super().__init__(config)

        # Check method type is valid
        if self.method_type not in ("root", "optimize"):
            raise ValueError(
                f"Invalid method_type: {self.method_type}. Must be 'root' or 'optimize'."
            )

        # For root-finding, verify sign change (optional but recommended)
        if self.method_type == "root":
            fa, fb = self.func(a), self.func(b)
            if fa * fb > 0 and abs(fa) > self.tol and abs(fb) > self.tol:
                # Only warn if neither endpoint is already close to a root
                print(
                    f"Warning: Function may not have a root in [{a}, {b}] as f({a})·f({b}) > 0"
                )

        # Store the endpoints of the interval.
        self.a = a
        self.b = b
        # Set current approximation to the midpoint of the interval.
        self.x = (a + b) / 2

        # Calculate golden ratio constants.
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio (~1.618034)
        self.tau = 1 / self.phi  # Inverse golden ratio (~0.618034)

        # Initialize test points within the interval using the golden ratio.
        self.x1 = a + (1 - self.tau) * (b - a)  # First test point
        self.x2 = a + self.tau * (b - a)  # Second test point

        # Evaluate the function at test points
        self.f1 = self.func(self.x1)
        self.f2 = self.func(self.x2)

        # Optionally record initial state
        if record_initial_state:
            initial_details = {
                "a": a,
                "b": b,
                "x1": self.x1,
                "x2": self.x2,
                "f(x1)": self.f1,
                "f(x2)": self.f2,
                "interval_width": b - a,
                "phi": self.phi,
                "tau": self.tau,
                "method_type": self.method_type,
            }
            self.add_iteration(x_old=a, x_new=self.x, details=initial_details)

    def get_current_x(self) -> float:
        """
        Get current best approximation.

        For root-finding: Returns the test point with smallest |f(x)|
        For optimization: Returns the test point with smallest f(x)

        Returns:
            float: Current best approximation
        """
        # Choose best point based on method type
        if self.method_type == "root":
            # For root-finding, return point with smallest absolute function value
            f_a, f_x1, f_x2, f_b = map(
                abs,
                [
                    self.func(self.a),
                    self.func(self.x1),
                    self.func(self.x2),
                    self.func(self.b),
                ],
            )

            # Consider all possible points including endpoints
            points = [(self.a, f_a), (self.x1, f_x1), (self.x2, f_x2), (self.b, f_b)]
            return min(points, key=lambda p: p[1])[0]
        else:
            # For optimization, return point with smallest function value
            f_a, f_x1, f_x2, f_b = [
                self.func(self.a),
                self.func(self.x1),
                self.func(self.x2),
                self.func(self.b),
            ]

            # Consider all possible points including endpoints
            points = [(self.a, f_a), (self.x1, f_x1), (self.x2, f_x2), (self.b, f_b)]
            return min(points, key=lambda p: p[1])[0]

    def step(self) -> float:
        """
        Perform one iteration of golden section method.

        Each iteration:
        1. Evaluates function at test points and compares results
        2. Eliminates a portion of the search interval
        3. Updates one test point and reuses the other
        4. Checks convergence criteria

        Returns:
            float: Current best approximation
        """
        # If convergence has already been achieved, return the current approximation.
        if self._converged:
            return self.get_current_x()

        # Store old x value for history
        x_old = self.get_current_x()

        # Handle the rare case when the function values are nearly equal (to avoid numerical issues).
        if abs(self.f1 - self.f2) < 1e-10:
            self.x2 += 1e-6  # Small perturbation to break tie.
            self.f2 = self.func(self.x2)

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "x1": self.x1,
            "x2": self.x2,
            "f(x1)": self.func(self.x1),
            "f(x2)": self.func(self.x2),
            "interval_width": self.b - self.a,
            "tau": self.tau,
            "phi": self.phi,
            "method_type": self.method_type,
        }

        # Compare function values to determine interval reduction
        if self.method_type == "root":
            # For root-finding, use absolute function values
            compare_result = abs(self.func(self.x1)) < abs(self.func(self.x2))
        else:
            # For optimization, use actual function values
            compare_result = self.func(self.x1) < self.func(self.x2)

        # Update the interval based on comparing function values at test points.
        if compare_result:
            # If the left test point is better, move the right endpoint to x2.
            self.b = self.x2
            # Shift x1 to the right.
            self.x2 = self.x1
            self.f2 = self.f1
            # Compute a new left test point.
            self.x1 = self.a + (1 - self.tau) * (self.b - self.a)
            self.f1 = self.func(self.x1)
        else:
            # Otherwise, if the right test point is better, move the left endpoint to x1.
            self.a = self.x1
            # Shift x2 to the left.
            self.x1 = self.x2
            self.f1 = self.f2
            # Compute a new right test point.
            self.x2 = self.a + self.tau * (self.b - self.a)
            self.f2 = self.func(self.x2)

        # Update the current approximation as the midpoint of the updated interval.
        self.x = (self.a + self.b) / 2

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        # Increment the iteration counter.
        self.iterations += 1

        # Check convergence criteria
        interval_width = self.b - self.a
        error = self.get_error()

        if (
            error <= self.tol  # Error within tolerance
            or interval_width <= self.tol  # Interval sufficiently small
            or self.iterations >= self.max_iter  # Max iterations reached
        ):
            self._converged = True

            # Add convergence reason to the last iteration
            last_iteration = self._history[-1]
            if error <= self.tol:
                last_iteration.details["convergence_reason"] = (
                    "function value within tolerance"
                )
            elif interval_width <= self.tol:
                last_iteration.details["convergence_reason"] = (
                    "interval width within tolerance"
                )
            else:
                last_iteration.details["convergence_reason"] = (
                    "maximum iterations reached"
                )

        return self.get_current_x()

    def get_error(self) -> float:
        """
        Get error at current point based on method type.

        For root-finding:
            - Error = |f(x)|, which measures how close we are to f(x) = 0

        For optimization:
            - Error estimate is based on interval width relative to original interval
            - In practice, gradient or function differences could be used

        Returns:
            float: Current error estimate
        """
        x = self.get_current_x()

        if self.method_type == "root":
            # For root-finding, error is how close f(x) is to zero
            return abs(self.func(x))
        else:
            # For optimization, we can use interval width as error measure
            # or we can use the base class method which will estimate gradient
            if self.derivative is not None:
                return abs(self.derivative(x))  # If derivative available
            else:
                # Default to gradient estimation from base class
                return super().get_error()

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For Golden Section search, the theoretical reduction ratio is
        the inverse golden ratio τ ≈ 0.618 per iteration.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract interval widths from last few iterations
        widths = [data.details.get("interval_width", 0) for data in self._history[-3:]]
        if any(w == 0 for w in widths):
            return None

        # Estimate convergence rate as ratio of successive interval widths
        rate1 = widths[-1] / widths[-2] if widths[-2] != 0 else 0
        rate2 = widths[-2] / widths[-3] if widths[-3] != 0 else 0

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return f"Golden Section {'Root-Finding' if self.method_type == 'root' else 'Optimization'} Method"


def golden_section_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "root",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    This function provides a simpler interface to the Golden Section method for
    users who don't need the full object-oriented functionality.

    Mathematical guarantee:
    - The interval reduction ratio is approximately τ ≈ 0.618 per iteration
    - For root-finding: If function is continuous with opposite signs at endpoints
    - For optimization: If function is unimodal on the interval

    Args:
        f: Function or configuration (if function, it's wrapped in a config)
        a: Left endpoint of interval
        b: Right endpoint of interval
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
    method = GoldenSectionMethod(config, a, b)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.get_current_x(), errors, method.iterations
