# algorithms/convex/protocols.py

"""Protocols for root-finding and optimization methods."""

from typing import (
    Protocol,
    runtime_checkable,
    Optional,
    Callable,
    Tuple,
    List,
    Dict,
    Any,
    Literal,
)
from dataclasses import dataclass

# Define valid method types
MethodType = Literal["root", "optimize"]


@runtime_checkable
class NumericalMethod(Protocol):
    """Protocol that all numerical methods must follow."""

    def step(self) -> float:
        """
        Perform one iteration of the method.

        Returns:
            float: Current approximation (root for root-finding, minimum for optimization)
        """
        ...

    def get_error(self) -> float:
        """
        Get current error estimate.

        Returns:
            float: Current error estimate (|f(x)| for root-finding, |f'(x)| for optimization)
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
    details: Dict[str, Any]  # Method-specific details


@dataclass
class NumericalMethodConfig:
    """Configuration for numerical methods.

    Attributes:
        func: Function to process (find root or minimize)
        method_type: Type of method ("root" or "optimize")
        x_range: Range to plot function (xmin, xmax)
        tol: Error tolerance
        max_iter: Maximum number of iterations
        derivative: Optional derivative function (for derivative-based methods)
        hessian: Optional hessian function (for second-order methods)
        use_derivative_free: bool = False  # Whether to use derivative-free approach
        finite_diff_step: float = 1e-7  # Step size for finite differences
    """

    func: Callable[[float], float]
    method_type: MethodType
    x_range: Tuple[float, float] = (-10, 10)
    tol: float = 1e-6
    max_iter: int = 100
    derivative: Optional[Callable[[float], float]] = None
    hessian: Optional[Callable[[float], float]] = None
    use_derivative_free: bool = False
    finite_diff_step: float = 1e-7


class BaseNumericalMethod:
    """Base class for numerical methods."""

    def __init__(self, config: NumericalMethodConfig):
        """
        Initialize the numerical method.

        Args:
            config: Configuration for the method
        """
        self.func = config.func
        self.method_type = config.method_type
        self.tol = config.tol
        self.max_iter = config.max_iter
        self.derivative = config.derivative
        self.hessian = config.hessian
        self.use_derivative_free = config.use_derivative_free
        self.finite_diff_step = config.finite_diff_step
        self._converged = False
        self.iterations = 0
        self._history: List[IterationData] = []

    def has_converged(self) -> bool:
        """Check if method has converged."""
        return self._converged

    def estimate_derivative(self, x: float) -> float:
        """
        Estimate derivative using central finite differences.

        Args:
            x: Point at which to estimate derivative

        Returns:
            float: Estimated derivative
        """
        h = self.finite_diff_step
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def estimate_gradient_norm(self, x: float) -> float:
        """
        Estimate gradient norm for derivative-free methods.
        Uses coordinate-wise finite differences.

        Args:
            x: Current point

        Returns:
            float: Estimated gradient norm
        """
        grad = self.estimate_derivative(x)
        return abs(grad)

    def get_error(self) -> float:
        """
        Get error at current point.

        For root: |f(x)|
        For optimize:
            - With derivative: |f'(x)|
            - Without derivative: Uses finite differences or other method-specific
              convergence criteria
        """
        x = self.get_current_x()
        if self.method_type == "root":
            return abs(self.func(x))
        else:  # optimize
            if self.use_derivative_free:
                # For derivative-free methods, use method-specific error
                # Default to gradient norm estimation
                return self.estimate_gradient_norm(x)
            elif self.derivative is not None:
                # Use actual derivative if available
                return abs(self.derivative(x))
            else:
                # Fallback to finite differences
                return abs(self.estimate_derivative(x))

    def get_current_x(self) -> float:
        """Get current x value."""
        raise NotImplementedError

    def add_iteration(
        self, x_old: float, x_new: float, details: Dict[str, Any]
    ) -> None:
        """Store iteration data."""
        f_old = self.func(x_old)
        f_new = self.func(x_new)
        error = self.get_error()

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
