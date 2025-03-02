# algorithms/convex/elimination.py

"""Elimination method for finding roots."""

from typing import List, Tuple, Optional, Union

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


class EliminationMethod(BaseNumericalMethod):
    """Implementation of the elimination method."""

    def __init__(self, config: NumericalMethodConfig, a: float, b: float):
        """
        Initialize the elimination method.

        Args:
            config: Configuration including function, tolerances, and iteration limits.
            a: Left endpoint of the interval.
            b: Right endpoint of the interval.
        """
        # Initialize the base class to set up common attributes (e.g., function, tolerance).
        super().__init__(config)
        self.a = a  # Set the left endpoint of the search interval.
        self.b = b  # Set the right endpoint of the search interval.
        self.x = (a + b) / 2  # Start with the midpoint as the initial approximation.

    # ------------------------
    # Core Algorithm Methods
    # ------------------------

    def step(self) -> float:
        """
        Perform one iteration of the elimination method.

        Returns:
            float: Current approximation of the root.
        """
        # If already converged, return the current approximation without further processing.
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Generate test points with better distribution
        interval_third = (self.b - self.a) / 3
        x1 = self.a + interval_third
        x2 = self.b - interval_third

        # Store function values for details
        f1, f2 = self.func(x1), self.func(x2)

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "x1": x1,
            "x2": x2,
            "f(x1)": f1,
            "f(x2)": f2,
        }

        # Use the elimination strategy to update the interval
        self.a, self.b = self._elim_step(x1, x2)

        # Compute the new approximation as the midpoint of the updated interval.
        self.x = (self.a + self.b) / 2

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        # Increment the iteration counter.
        self.iterations += 1

        # Update convergence check to include function value
        error = abs(self.func(self.x))
        if (
            error <= self.tol
            or abs(self.b - self.a) < self.tol
            or self.iterations >= self.max_iter
        ):
            self._converged = True

        return self.x

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def compute_descent_direction(self, x: Union[float, float]) -> Union[float, float]:
        """
        Compute the descent direction at the current point.

        This method is implemented for interface compatibility, but is not used
        by the elimination method which doesn't use gradient information.

        For elimination method, we don't use descent directions because we're
        using interval reduction rather than following a gradient.

        Args:
            x: Current point

        Returns:
            Union[float, float]: Direction (always 0 for elimination)

        Raises:
            NotImplementedError: This method is not applicable for elimination method
        """
        raise NotImplementedError(
            "Elimination method doesn't use descent directions - it uses interval reduction"
        )

    def compute_step_length(
        self, x: Union[float, float], direction: Union[float, float]
    ) -> float:
        """
        Compute the step length.

        This method is implemented for interface compatibility, but is not used
        by the elimination method which doesn't use step lengths.

        The elimination method uses interval reduction to find roots.

        Args:
            x: Current point
            direction: Descent direction (not used)

        Returns:
            float: Step length

        Raises:
            NotImplementedError: This method is not applicable for elimination method
        """
        raise NotImplementedError(
            "Elimination method doesn't use step lengths - it uses interval reduction"
        )

    # ----------------
    # Helper Methods
    # ----------------

    def _elim_step(self, x1: float, x2: float) -> Tuple[float, float]:
        """
        Perform one step of the elimination method.

        Compare function values to determine which region likely contains the root.
        """
        # Evaluate the function at both test points
        f1, f2 = self.func(x1), self.func(x2)

        # Compare actual values instead of absolute values
        # This helps determine which side of the root we're on
        if f1 * f2 <= 0:  # If points straddle the root
            return (x1, x2) if x1 < x2 else (x2, x1)
        elif abs(f1) < abs(f2):
            return (self.a, x2) if x1 < x2 else (x1, self.b)
        else:
            return (x1, self.b) if x1 < x2 else (self.a, x2)

    # ---------------------
    # State Access Methods
    # ---------------------

    def get_error(self) -> float:
        """
        Calculate the error estimate for the current solution.

        For root-finding, the error is |f(x)|.

        Returns:
            float: Error estimate
        """
        return abs(self.func(self.x))

    def has_converged(self) -> bool:
        """
        Check if method has converged based on error tolerance or max iterations.

        Returns:
            bool: True if converged, False otherwise
        """
        return self._converged

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        The elimination method typically reduces the interval by 1/3 in each iteration.

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
        return "Elimination Method"


def elimination_search(
    f: NumericalMethodConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "root",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) to be used.
        a: Left endpoint of interval.
        b: Right endpoint of interval.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (x_min, errors, iterations) where:
         - x_min is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the total number of iterations performed.
    """
    # Create a configuration instance using provided function, tolerance, and iteration limits.
    config = NumericalMethodConfig(
        func=f, method_type=method_type, tol=tol, max_iter=max_iter
    )
    # Instantiate the elimination method with the specified interval.
    method = EliminationMethod(config, a, b)

    errors = []  # List to record error values for each iteration.
    # Iterate until the method converges.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return the final approximation, error history, and iteration count.
    return method.x, errors, method.iterations
