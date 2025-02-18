# algorithms/convex/protocols.py

"""Protocols for root-finding methods."""

from typing import Protocol, runtime_checkable, Optional, Callable, Tuple
from dataclasses import dataclass


@runtime_checkable
class RootFinder(Protocol):
    """Protocol that all root finding implementations must follow."""

    def step(self) -> float:
        """
        Perform one iteration of the method.

        Returns:
            float: Current approximation of the root
        """
        ...

    def get_error(self) -> float:
        """
        Get current error estimate.

        Returns:
            float: Current error estimate (typically |f(x)|)
        """
        ...

    def has_converged(self) -> bool:
        """
        Check if method has converged.

        Returns:
            bool: True if converged, False otherwise
        """
        ...

    @property
    def name(self) -> str:
        """Name of the method for display purposes."""
        ...


@dataclass
class RootFinderConfig:
    """Configuration for root finding methods.

    Attributes:
        func: Function whose root we want to find
        x_range: Range to plot function (xmin, xmax)
        tol: Error tolerance
        max_iter: Maximum number of iterations
        derivative: Optional derivative function (for methods that use it)
    """

    func: Callable[[float], float]
    x_range: Tuple[float, float] = (-10, 10)
    tol: float = 1e-6
    max_iter: int = 100
    derivative: Optional[Callable[[float], float]] = None


# Base class that implements common functionality
class BaseRootFinder:
    """Base class for root finding methods."""

    def __init__(self, config: RootFinderConfig):
        """
        Initialize the root finder.

        Args:
            config: Configuration for the root finder
        """
        self.func = config.func
        self.tol = config.tol
        self.max_iter = config.max_iter
        self.derivative = config.derivative
        self._converged = False
        self.iterations = 0

    def has_converged(self) -> bool:
        """Check if method has converged."""
        return self._converged

    def get_error(self) -> float:
        """Get absolute error at current point."""
        return abs(self.func(self.x))

    @property
    def name(self) -> str:
        """Get method name."""
        return self.__class__.__name__


# # Example of how a method would be implemented using these protocols:
# class NewtonMethod(BaseRootFinder):
#     """Newton's method implementation."""

#     def __init__(self, config: RootFinderConfig, x0: float):
#         """
#         Initialize Newton's method.

#         Args:
#             config: Root finder configuration
#             x0: Initial guess
#         """
#         if config.derivative is None:
#             raise ValueError("Newton's method requires derivative function")
#         super().__init__(config)
#         self.x = x0

#     def step(self) -> float:
#         """Perform one Newton iteration."""
#         if not self._converged:
#             fx = self.func(self.x)
#             dfx = self.derivative(self.x)

#             if abs(dfx) < 1e-10:  # Avoid division by zero
#                 self._converged = True
#                 return self.x

#             self.x = self.x - fx / dfx
#             self.iterations += 1

#             if abs(fx) < self.tol or self.iterations >= self.max_iter:
#                 self._converged = True

#         return self.x

#     @property
#     def name(self) -> str:
#         return "Newton's Method"
