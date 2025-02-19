# algorithms/convex/protocols.py

"""Protocols for root-finding methods."""

from typing import (
    Protocol,
    runtime_checkable,
    Optional,
    Callable,
    Tuple,
    List,
    Dict,
    Any,
)
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
class IterationData:
    """Data structure to store details of each iteration."""

    iteration: int
    x_old: float
    x_new: float
    f_old: float
    f_new: float
    error: float
    details: Dict[str, Any]  # Method-specific details (e.g., derivative for Newton)


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
        self._history: List[IterationData] = []

    def has_converged(self) -> bool:
        """Check if method has converged."""
        return self._converged

    def get_error(self) -> float:
        """Get absolute error at current point."""
        return abs(self.func(self.get_current_x()))

    def get_current_x(self) -> float:
        """Get current x value."""
        raise NotImplementedError

    def add_iteration(
        self, x_old: float, x_new: float, details: Dict[str, Any]
    ) -> None:
        """Store iteration data."""
        f_old = self.func(x_old)
        f_new = self.func(x_new)
        error = abs(f_new)

        data = IterationData(
            iteration=self.iterations,
            x_old=x_old,
            x_new=x_new,
            f_old=f_old,
            f_new=f_new,
            error=error,
            details=details,
        )
        self._history.append(data)

    def get_iteration_history(self) -> List[IterationData]:
        """Get complete iteration history."""
        return self._history

    def get_last_iteration(self) -> Optional[IterationData]:
        """Get data from last iteration."""
        return self._history[-1] if self._history else None

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
