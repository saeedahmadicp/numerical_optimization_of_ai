# algorithms/convex/regula_falsi.py

"""
Regula falsi (false position) method for finding roots.

The regula falsi method combines the robustness of bisection with the speed of
secant method by using linear interpolation between interval endpoints where the
function changes sign, always maintaining a bracket around the root.

Mathematical Basis:
----------------
The false position formula computes the next approximation as:
    x_{n+1} = (b_n * f(a_n) - a_n * f(b_n)) / (f(a_n) - f(b_n))

where [a_n, b_n] is the current bracketing interval with f(a_n) and f(b_n) having opposite signs.
Unlike bisection which simply uses the midpoint, this formula weights the next point
based on function values, giving faster convergence near the root.

Convergence Properties:
--------------------
- Guaranteed convergence if a continuous function changes sign over an interval
- Linear convergence, faster than bisection but slower than secant method
- Never loses bracketing of the root, unlike secant method
- May converge slowly when the function is nearly flat on one side of the root
- Modified versions exist to address slow convergence in certain cases
"""

from typing import List, Tuple, Optional, Union, Callable

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


class RegulaFalsiMethod(BaseNumericalMethod):
    """
    Implementation of regula falsi (false position) method for root finding.

    Regula falsi combines bisection's robustness with secant method's efficiency
    by using linear interpolation between interval endpoints while maintaining
    a bracket around the root at all times.

    Mathematical basis:
    - Uses weighted average based on function values at interval endpoints
    - Maintains a bracketing interval where the function changes sign
    - Updates one endpoint per iteration based on the sign of f(x)

    Convergence properties:
    - Guaranteed convergence for continuous functions
    - Linear convergence rate, but often faster than bisection
    - May exhibit "stalling" where one endpoint remains fixed
    - Always brackets the root, ensuring reliability

    Implementation features:
    - Simple implementation with minimal parameters required
    - Robustness against difficult functions with irregular behavior
    - Works on any continuous function that changes sign on the interval
    - Can only be used for root-finding, not optimization
    """

    def __init__(self, config: NumericalMethodConfig, a: float, b: float):
        """
        Initialize regula falsi method.

        Args:
            config: Configuration including function and tolerances
            a: Left endpoint of interval
            b: Right endpoint of interval

        Raises:
            ValueError: If f(a) and f(b) have same sign, or if method_type is not 'root'
        """
        if config.method_type != "root":
            raise ValueError("Regula falsi method can only be used for root finding")

        super().__init__(config)

        self.a = a
        self.b = b
        self.fa = self.func(a)
        self.fb = self.func(b)

        if self.fa * self.fb >= 0:
            raise ValueError("Function must have opposite signs at interval endpoints")

        # Initial position will be computed using the weighted average
        self.x = self._compute_next_x()

        # For tracking convergence
        self.prev_x = None
        self.iterations_without_progress = 0

        # For modified regula falsi (Illinois algorithm)
        self.use_modified = True  # Enable modified method to speed up convergence
        self.modified_factor = 0.5  # Scaling factor for Illinois algorithm

    def step(self) -> float:
        """
        Perform one iteration of the regula falsi method.

        Returns:
            float: Current approximation of the root
        """
        if self._converged:
            return self.x

        x_old = self.x
        self.prev_x = x_old

        # Compute the descent direction and next position
        self.x = self._compute_next_x()
        fx = self.func(self.x)

        details = {
            "a": self.a,
            "b": self.b,
            "f(a)": self.fa,
            "f(b)": self.fb,
            "f(x)": fx,
            "weighted_avg": self.x,
        }

        # Update the brackets
        updated_end = ""
        if self.fa * fx < 0:
            # Root is between a and x
            self.b = self.x
            self.fb = fx
            updated_end = "b"

            # Illinois algorithm: if we're updating the same endpoint repeatedly
            # Scale down the function value at the fixed endpoint
            if (
                self.use_modified
                and updated_end == "b"
                and self.iterations_without_progress > 2
            ):
                self.fa *= self.modified_factor
                details["modified_fa"] = self.fa
                self.iterations_without_progress = 0
        else:
            # Root is between x and b
            self.a = self.x
            self.fa = fx
            updated_end = "a"

            # Illinois algorithm for the other endpoint
            if (
                self.use_modified
                and updated_end == "a"
                and self.iterations_without_progress > 2
            ):
                self.fb *= self.modified_factor
                details["modified_fb"] = self.fb
                self.iterations_without_progress = 0

        details["updated_end"] = updated_end

        # Track if we're making progress
        if self.prev_x is not None and abs(self.x - self.prev_x) < 1e-10 * abs(self.x):
            self.iterations_without_progress += 1
        else:
            self.iterations_without_progress = 0

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        if self.get_error() <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def compute_descent_direction(self, x: float) -> float:
        """
        Compute the descent direction at point x.

        For regula falsi, this is not explicitly used as the method
        calculates the next point directly using weighted average.

        Args:
            x: Current point

        Returns:
            float: Direction towards the root (always 0 for regula falsi)
        """
        # Regula falsi doesn't use a descent direction in the same way
        # as other methods. It computes the next point directly.
        return 0.0

    def compute_step_length(self, x: float, direction: float) -> float:
        """
        Compute step length for the direction.

        For regula falsi, this is not explicitly used as the method
        calculates the next point directly.

        Args:
            x: Current point
            direction: Descent direction (unused)

        Returns:
            float: Step length (always 0 for regula falsi)
        """
        # Regula falsi doesn't use step length in the same way as other methods
        return 0.0

    def _compute_next_x(self) -> float:
        """
        Compute the next x value using the weighted average formula.

        Returns:
            float: Next approximation
        """
        # Regula falsi formula: weighted average based on function values
        return (self.b * self.fa - self.a * self.fb) / (self.fa - self.fb)

    def get_error(self) -> float:
        """
        Calculate the error estimate for the current approximation.

        For root-finding: |f(x)|

        Returns:
            float: Error estimate
        """
        return abs(self.func(self.x))

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For regula falsi, the convergence rate is typically linear, but
        depends on the function's properties and implementation details.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # For linear convergence, we expect error_{n+1} ≈ C * error_n
        rate1 = recent_errors[-1] / recent_errors[-2] if recent_errors[-2] > 0 else 0
        rate2 = recent_errors[-2] / recent_errors[-3] if recent_errors[-3] > 0 else 0

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        """Return the name of the method."""
        variant = "Modified " if self.use_modified else ""
        return f"{variant}Regula Falsi Method"


def regula_falsi_search(
    f: Union[NumericalMethodConfig, Callable[[float], float]],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "root",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations)
    """
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type=method_type, tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = RegulaFalsiMethod(config, a, b)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
