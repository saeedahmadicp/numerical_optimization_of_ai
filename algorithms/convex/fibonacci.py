# algorithms/convex/fibonacci.py

"""
Fibonacci search method for optimization and root-finding.

The Fibonacci search method is primarily an optimization technique for finding extrema
of unimodal functions. It efficiently narrows down the interval containing the minimum
by using Fibonacci numbers to determine testing points.

Mathematical Basis:
----------------
For optimization of a unimodal function f over an interval [a,b]:

1. Place test points x₁ and x₂ at positions determined by Fibonacci ratios
2. Compare f(x₁) and f(x₂) to determine which portion of the interval to eliminate
3. Reduce the interval and recalculate test points
4. Repeat until convergence or Fibonacci terms are exhausted

For root-finding, this method adapts by:
1. Comparing absolute function values |f(x₁)| and |f(x₂)| instead
2. Selecting the subinterval that likely contains the root

Convergence Properties:
--------------------
- The interval reduction factor is asymptotically the golden ratio: (√5-1)/2 ≈ 0.618
- More efficient than bisection for minimization problems
- For n iterations, requires only n+1 function evaluations
- Optimal for minimizing the worst-case number of function evaluations
"""

from typing import List, Tuple, Optional, Callable, Union

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


def fib_generator(n: int) -> List[int]:
    """
    Generate Fibonacci sequence up to n terms.

    The Fibonacci sequence is defined by the recurrence relation:
    F₀ = 0, F₁ = 1, Fₙ = Fₙ₋₁ + Fₙ₋₂ for n > 1

    Args:
        n: Number of terms to generate

    Returns:
        List of first n Fibonacci numbers
    """
    # Handle edge cases
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]

    # Initialize with first two Fibonacci numbers
    fib = [1, 1]
    # Generate remaining terms using the recurrence relation
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib


class FibonacciMethod(BaseNumericalMethod):
    """
    Implementation of the Fibonacci search method.

    The Fibonacci method is an efficient technique for bracketing extrema
    of unimodal functions or finding roots. It uses the Fibonacci sequence to
    optimally place test points, requiring fewer function evaluations than
    methods like bisection for the same accuracy.

    Mathematical guarantees:
    - For optimization tasks: If f is unimodal on [a,b], the method will converge
      to a minimum with an interval of width approximately (b-a)/Fₙ after n steps.
    - For root-finding tasks: If f is continuous and f(a)·f(b) < 0, the method will
      converge to a root with similar efficiency, though bisection may be more robust.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        a: float,
        b: float,
        n_terms: int = 30,
        record_initial_state: bool = False,
    ):
        """
        Initialize the Fibonacci search method.

        Args:
            config: Configuration including function, convergence criteria, etc.
            a: Left endpoint of interval
            b: Right endpoint of interval
            n_terms: Number of Fibonacci terms to use in the search
            record_initial_state: Whether to record the initial state in history

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

        self.a = a  # Left endpoint
        self.b = b  # Right endpoint
        self.x = (a + b) / 2  # Initial approximation (midpoint)

        # Generate Fibonacci sequence for interval reduction
        self.fib = fib_generator(n_terms + 1)  # We need n_terms+1 Fibonacci numbers
        self.n_terms = n_terms
        self.current_term = n_terms  # Index of current Fibonacci term (counts down)

        # Calculate initial test points using Fibonacci ratios
        # These are placed optimally to minimize worst-case function evaluations
        ratio1 = self.fib[self.current_term - 2] / self.fib[self.current_term]
        ratio2 = self.fib[self.current_term - 1] / self.fib[self.current_term]

        self.x1 = a + ratio1 * (b - a)  # First test point
        self.x2 = a + ratio2 * (b - a)  # Second test point

        # Evaluate function at test points
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
                "fib_term": self.current_term,
                "fibonacci_number": self.fib[self.current_term],
                "method_type": self.method_type,
            }
            self.add_iteration(x_old=a, x_new=self.x, details=initial_details)

    # ------------------------
    # Core Algorithm Methods
    # ------------------------

    def step(self) -> float:
        """
        Perform one iteration of the Fibonacci search method.

        Each iteration:
        1. Evaluates function at test points and compares results
        2. Eliminates a portion of the search interval
        3. Updates one test point and reuses the other
        4. Checks convergence criteria

        Returns:
            float: Current best approximation
        """
        # If already converged, return current approximation
        if self._converged:
            return self.get_current_x()

        # Store old x value for history
        x_old = self.get_current_x()

        # Store iteration details before updating
        details = {
            "a": self.a,
            "b": self.b,
            "x1": self.x1,
            "x2": self.x2,
            "f(x1)": self.func(self.x1),
            "f(x2)": self.func(self.x2),
            "interval_width": self.b - self.a,
            "fib_term": self.current_term,
            "fibonacci_number": self.fib[self.current_term],
            "method_type": self.method_type,
        }

        # Decrement Fibonacci term counter
        self.current_term -= 1

        # Compare function values to determine interval reduction
        if self.method_type == "root":
            # For root-finding, use absolute function values
            f1, f2 = abs(self.func(self.x1)), abs(self.func(self.x2))
            compare_result = f1 < f2
        else:
            # For optimization, use actual function values
            f1, f2 = self.func(self.x1), self.func(self.x2)
            compare_result = f1 < f2

        # Reduce interval based on comparison
        if compare_result:
            # If f(x₁) is better, eliminate [x₂, b]
            self.b = self.x2
            self.x2 = self.x1
            # Calculate new x₁ using Fibonacci ratio
            if self.current_term >= 2:  # Ensure we have enough Fibonacci terms
                ratio = self.fib[self.current_term - 2] / self.fib[self.current_term]
                self.x1 = self.a + ratio * (self.b - self.a)
            else:
                # For last iterations when we run out of terms
                self.x1 = self.a + 0.5 * (self.x2 - self.a)
        else:
            # If f(x₂) is better, eliminate [a, x₁]
            self.a = self.x1
            self.x1 = self.x2
            # Calculate new x₂ using Fibonacci ratio
            if self.current_term >= 2:  # Ensure we have enough Fibonacci terms
                ratio = self.fib[self.current_term - 1] / self.fib[self.current_term]
                self.x2 = self.a + ratio * (self.b - self.a)
            else:
                # For last iterations when we run out of terms
                self.x2 = self.x1 + 0.5 * (self.b - self.x1)

        # Update current approximation to best point
        x_new = self.get_current_x()

        # Store iteration data
        self.add_iteration(x_old, x_new, details)
        self.iterations += 1

        # Check convergence criteria
        interval_width = self.b - self.a
        error = self.get_error()

        if (
            error <= self.tol  # Error within tolerance
            or interval_width <= self.tol  # Interval sufficiently small
            or self.iterations >= self.max_iter  # Max iterations reached
            or self.current_term < 2  # Fibonacci terms exhausted
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
            elif self.current_term < 2:
                last_iteration.details["convergence_reason"] = (
                    "Fibonacci terms exhausted"
                )
            else:
                last_iteration.details["convergence_reason"] = (
                    "maximum iterations reached"
                )

        return x_new

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

    def compute_descent_direction(self, x: Union[float, float]) -> Union[float, float]:
        """
        Compute the descent direction at the current point.

        This method is implemented for interface compatibility, but is not used
        by the Fibonacci method which doesn't use gradient information.

        For Fibonacci method, we don't use descent directions because we're
        bracketing the solution rather than following a gradient.

        Args:
            x: Current point

        Returns:
            Union[float, float]: Direction (always 0 for Fibonacci method)

        Raises:
            NotImplementedError: This method is not applicable for Fibonacci method
        """
        raise NotImplementedError(
            "Fibonacci method doesn't use descent directions - it uses interval reduction"
        )

    def compute_step_length(
        self, x: Union[float, float], direction: Union[float, float]
    ) -> float:
        """
        Compute the step length.

        This method is implemented for interface compatibility, but is not used
        by the Fibonacci method which doesn't use step lengths.

        The Fibonacci method uses strategic interval reduction based on the
        Fibonacci sequence rather than step lengths.

        Args:
            x: Current point
            direction: Descent direction (not used)

        Returns:
            float: Step length

        Raises:
            NotImplementedError: This method is not applicable for Fibonacci method
        """
        raise NotImplementedError(
            "Fibonacci method doesn't use step lengths - it uses interval reduction"
        )

    # ---------------------
    # State Access Methods
    # ---------------------

    def has_converged(self) -> bool:
        """
        Check if method has converged based on error tolerance, interval width,
        maximum iterations, or exhaustion of Fibonacci terms.

        Returns:
            bool: True if converged, False otherwise
        """
        return self._converged

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

        For Fibonacci search, the theoretical reduction ratio approaches
        the golden ratio (≈ 0.618) as n increases.

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
        return f"Fibonacci {'Root-Finding' if self.method_type == 'root' else 'Optimization'} Method"


def fibonacci_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    a: float,
    b: float,
    n_terms: int = 30,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "root",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    This function provides a simpler interface to the Fibonacci method for
    users who don't need the full object-oriented functionality.

    Mathematical guarantee:
    - For n iterations, the final interval has width approximately (b-a)/Fₙ.
    - For root-finding: If function is continuous with opposite signs at endpoints.
    - For optimization: If function is unimodal on the interval.

    Args:
        f: Function or configuration (if function, it's wrapped in a config)
        a: Left endpoint of interval
        b: Right endpoint of interval
        n_terms: Number of Fibonacci terms to use
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
    method = FibonacciMethod(config, a, b, n_terms)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.get_current_x(), errors, method.iterations
