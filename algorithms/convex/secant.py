# algorithms/convex/secant.py

"""Secant method for finding roots."""

from typing import List, Tuple, Optional

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class SecantMethod(BaseNumericalMethod):
    """Implementation of secant method for root finding."""

    def __init__(self, config: NumericalMethodConfig, x0: float, x1: float):
        """
        Initialize secant method.

        Args:
            config: Configuration including function and tolerances
            x0: First initial guess
            x1: Second initial guess

        Raises:
            ValueError: If method_type is not 'root'
        """
        if config.method_type != "root":
            raise ValueError("Secant method can only be used for root finding")

        super().__init__(config)

        self.x0: Optional[float] = x0
        self.x1: Optional[float] = x1
        self.f0 = self.func(x0)
        self.f1 = self.func(x1)
        self.x = x1

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of secant method.

        Returns:
            float: Current approximation of the root
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

        x2 = self.x1 - self.f1 * (self.x1 - self.x0) / (self.f1 - self.f0)
        f2 = self.func(x2)

        details = {
            "x0": self.x0,
            "x1": self.x1,
            "f(x0)": self.f0,
            "f(x1)": self.f1,
            "f(x2)": f2,
            "denominator": self.f1 - self.f0,
            "step": x2 - self.x1,
        }

        self.x0 = self.x1
        self.f0 = self.f1
        self.x1 = x2
        self.f1 = f2
        self.x = x2

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        if self.get_error() <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Secant Method"


def secant_search(
    f: NumericalMethodConfig,
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding
        x0: First initial guess
        x1: Second initial guess
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

    method = SecantMethod(config, x0, x1)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
