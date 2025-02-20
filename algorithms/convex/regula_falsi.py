# algorithms/convex/regula_falsi.py

"""Regula falsi (false position) method for finding roots."""

from typing import List, Tuple

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class RegulaFalsiMethod(BaseNumericalMethod):
    """Implementation of regula falsi method for root finding."""

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

        self.x = self._weighted_average()

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _weighted_average(self) -> float:
        """
        Compute the weighted average of the endpoints based on their function values.

        Returns:
            float: The weighted average (false position)
        """
        return (self.b * self.fa - self.a * self.fb) / (self.fa - self.fb)

    def step(self) -> float:
        """
        Perform one iteration of the regula falsi method.

        Returns:
            float: Current approximation of the root
        """
        if self._converged:
            return self.x

        x_old = self.x
        self.x = self._weighted_average()
        fx = self.func(self.x)

        details = {
            "a": self.a,
            "b": self.b,
            "f(a)": self.fa,
            "f(b)": self.fb,
            "f(x)": fx,
            "weighted_avg": self.x,
        }

        if self.fa * fx < 0:
            self.b = self.x
            self.fb = fx
            details["updated_end"] = "b"
        else:
            self.a = self.x
            self.fa = fx
            details["updated_end"] = "a"

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        if self.get_error() <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Regula Falsi Method"


def regula_falsi_search(
    f: NumericalMethodConfig,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
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
            func=f, method_type="root", tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = RegulaFalsiMethod(config, a, b)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
