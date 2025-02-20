# algorithms/convex/bisection.py

"""Bisection method for finding roots of continuous functions."""

from typing import List, Tuple

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class BisectionMethod(BaseNumericalMethod):
    """Implementation of the bisection method for root finding."""

    def __init__(self, config: NumericalMethodConfig, a: float, b: float):
        """
        Initialize the bisection method.

        Args:
            config: Configuration including function and tolerances
            a: Left endpoint of interval
            b: Right endpoint of interval

        Raises:
            ValueError: If f(a) and f(b) have same sign, or if method_type is not 'root'
        """
        if config.method_type != "root":
            raise ValueError("Bisection method can only be used for root finding")

        # Call the base class initializer
        super().__init__(config)

        # Evaluate the function at both endpoints
        fa, fb = self.func(a), self.func(b)
        if fa * fb >= 0:
            raise ValueError("Function must have opposite signs at interval endpoints")

        self.a = a
        self.b = b
        self.x = (a + b) / 2

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def step(self) -> float:
        """Perform one iteration of the bisection method.

        Returns:
            float: Current approximation of the root
        """
        if self._converged:
            return self.x

        x_old = self.x

        # Compute the midpoint
        c = (self.a + self.b) / 2
        fc = self.func(c)

        # Store iteration details
        details = {
            "a": self.a,
            "b": self.b,
            "f(a)": self.func(self.a),
            "f(b)": self.func(self.b),
            "f(c)": fc,
        }

        # Update interval based on sign
        if self.func(self.a) * fc < 0:
            self.b = c
        else:
            self.a = c

        # Update current approximation
        self.x = c

        # Store iteration data
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence
        if self.get_error() <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Bisection Method"


def bisection_search(
    f: NumericalMethodConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function, depending on legacy usage)
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
