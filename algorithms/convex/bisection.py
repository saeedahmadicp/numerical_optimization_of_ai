# algorithms/convex/bisection.py

"""
Bisection method for finding roots of continuous functions.

The bisection method is one of the simplest and most robust root-finding
algorithms. It is based on the Intermediate Value Theorem from calculus,
which states that if a continuous function f(x) has values of opposite sign
at the endpoints of an interval [a,b], then f must have at least one root
in that interval.

Mathematical Basis:
----------------
Given a continuous function f and an interval [a,b] such that f(a)·f(b) < 0:

1. Compute the midpoint c = (a + b) / 2
2. Evaluate f(c)
3. If f(c) = 0 (or |f(c)| < tol), c is the root
4. If f(a)·f(c) < 0, the root is in [a,c], so set b = c
5. Otherwise, the root is in [c,b], so set a = c
6. Repeat until convergence

Convergence Properties:
--------------------
- The method always converges for continuous functions when f(a)·f(b) < 0
- The error is halved in each iteration: |x_n - root| ≤ (b-a)/2^n
- Linear convergence rate with convergence factor of 1/2
- Does not require derivatives of the function
"""

from typing import List, Tuple, Optional, Callable, Union
import math

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class BisectionMethod(BaseNumericalMethod):
    """
    Implementation of the bisection method for root finding.

    The bisection method iteratively narrows down an interval [a,b] where
    f(a) and f(b) have opposite signs. The midpoint of the interval is
    computed at each step, and the interval is updated to maintain the
    property that the function changes sign within it.

    Mathematical guarantee:
    The error after n iterations is at most (b-a)/2^n, where [a,b] is the
    initial interval. This provides a predictable convergence rate, making
    it reliable though not as fast as methods like Newton's method.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        a: float,
        b: float,
        record_initial_state: bool = False,
    ):
        """
        Initialize the bisection method.

        Args:
            config: Configuration including function and tolerances
            a: Left endpoint of interval
            b: Right endpoint of interval
            record_initial_state: Whether to record the initial state in the iteration history

        Raises:
            ValueError: If f(a) and f(b) have same sign, or if method_type is not 'root'
            ValueError: If a >= b (invalid interval)
        """
        if config.method_type != "root":
            raise ValueError("Bisection method can only be used for root finding")

        if a >= b:
            raise ValueError(f"Invalid interval: [a={a}, b={b}]. Must have a < b.")

        # Call the base class initializer
        super().__init__(config)

        # Evaluate the function at both endpoints
        fa, fb = self.func(a), self.func(b)

        # Check if either endpoint is already a root
        if abs(fa) < self.tol:
            self.a = self.b = a
            self.x = a
            self._converged = True
            return

        if abs(fb) < self.tol:
            self.a = self.b = b
            self.x = b
            self._converged = True
            return

        # Ensure opposite signs for the bisection method to work
        if fa * fb >= 0:
            raise ValueError(
                f"Function must have opposite signs at interval endpoints: "
                f"f({a}) = {fa}, f({b}) = {fb}"
            )

        self.a = a
        self.b = b
        self.x = (a + b) / 2

        # Calculate theoretical maximum iterations needed for the given tolerance
        # Based on: (b-a)/2^n < tol => n > log2((b-a)/tol)
        self.theoretical_max_iter = math.ceil(math.log2((b - a) / self.tol))

        # Optionally record the initial state in the history
        if record_initial_state:
            initial_details = {
                "a": a,
                "b": b,
                "f(a)": fa,
                "f(b)": fb,
                "interval_width": b - a,
                "theoretical_max_iter": self.theoretical_max_iter,
            }
            self.add_iteration(x_old=a, x_new=self.x, details=initial_details)

    def get_current_x(self) -> float:
        """
        Get current x value (midpoint of the current interval).

        Returns:
            float: Current approximation of the root
        """
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of the bisection method.

        Each iteration:
        1. Computes the midpoint of the current interval [a,b]
        2. Evaluates the function at the midpoint
        3. Updates the interval to maintain opposite signs at endpoints
        4. Checks convergence criteria

        Guarantee: The width of the interval is halved with each iteration,
        ensuring the error decreases by a factor of 2 each time.

        Returns:
            float: Current approximation of the root
        """
        # If already converged, return current approximation
        if self._converged:
            return self.x

        # Save previous approximation for history
        x_old = self.x

        # Compute the midpoint
        c = (self.a + self.b) / 2
        fc = self.func(c)

        # Check if midpoint is a root (within tolerance)
        if abs(fc) < self.tol:
            self.x = c
            self._converged = True

            # Store iteration details
            details = {
                "a": self.a,
                "b": self.b,
                "f(a)": self.func(self.a),
                "f(b)": self.func(self.b),
                "f(c)": fc,
                "interval_width": self.b - self.a,
                "convergence_reason": "f(x) within tolerance",
            }

            self.add_iteration(x_old, self.x, details)
            self.iterations += 1
            return self.x

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "f(a)": self.func(self.a),
            "f(b)": self.func(self.b),
            "f(c)": fc,
            "interval_width": self.b - self.a,
            "error_bound": (self.b - self.a) / 2,  # Theoretical error bound
        }

        # Update interval based on sign
        fa = self.func(self.a)
        if fa * fc < 0:
            self.b = c  # Root is in left half
        else:
            self.a = c  # Root is in right half

        # Update current approximation (midpoint of new interval)
        self.x = (self.a + self.b) / 2

        # Store iteration data
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence based on:
        # 1. Function value at midpoint being close to zero
        # 2. Interval width being sufficiently small
        # 3. Maximum iterations reached
        interval_width = self.b - self.a
        if (
            self.get_error() <= self.tol
            or interval_width <= self.tol
            or self.iterations >= min(self.max_iter, self.theoretical_max_iter)
        ):

            self._converged = True

            # Add convergence reason to the last iteration's details
            last_iteration = self._history[-1]
            if self.get_error() <= self.tol:
                last_iteration.details["convergence_reason"] = "f(x) within tolerance"
            elif interval_width <= self.tol:
                last_iteration.details["convergence_reason"] = (
                    "interval width within tolerance"
                )
            else:
                last_iteration.details["convergence_reason"] = (
                    "maximum iterations reached"
                )

        return self.x

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For bisection, the theoretical rate is linear with factor 1/2.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # Estimate convergence rate as |e_{n+1}/e_n|
        rate1 = recent_errors[-1] / recent_errors[-2] if recent_errors[-2] != 0 else 0
        rate2 = recent_errors[-2] / recent_errors[-3] if recent_errors[-3] != 0 else 0

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return "Bisection Method"


def bisection_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    This function provides a simpler interface to the bisection method for
    users who don't need the full object-oriented functionality.

    Mathematical guarantee:
    If f is continuous and f(a)·f(b) < 0, the method will converge to a root.
    The error after n iterations is bounded by (b-a)/2^n.

    Args:
        f: Function or configuration (if function, it's wrapped in a config)
        a: Left endpoint of interval
        b: Right endpoint of interval
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations)
    """
    # If f is a function rather than a config, create a config
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type="root", tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = BisectionMethod(config, a, b)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
