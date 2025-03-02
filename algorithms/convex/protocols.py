# algorithms/convex/protocols.py

"""
Protocols for root-finding and optimization methods.

This module defines the interfaces and base classes for numerical methods used in
root-finding and optimization problems. It provides a consistent API that all
numerical methods must implement, along with common utilities and data structures.

Numerical methods follow an iterative approach where each step improves the approximation
until convergence criteria are met. This modular design allows for easy implementation
of new methods while ensuring consistent behavior across the library.
"""

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
    TypeVar,
    Union,
)
from dataclasses import dataclass
import numpy as np

# Define valid method types
MethodType = Literal["root", "optimize"]

# Define valid step length methods
StepLengthMethod = Literal[
    "fixed",
    "backtracking",
    "wolfe",
    "strong_wolfe",
    "goldstein",
    "barzilai_borwein",
    "diminishing",
    "polyak",
    "trust_region",
]

# Define valid descent direction methods
DescentDirectionMethod = Literal[
    "steepest_descent",
    "newton",
    "bfgs",
    "l_bfgs",
    "dfp",
    "sr1",
    "conjugate_gradient",
    "momentum",
    "nesterov",
    "coordinate_descent",
    "frank_wolfe",
    "subgradient",
    "proximal_gradient",
]

# Type variable for scalar functions
T = TypeVar("T", float, np.ndarray)


@runtime_checkable
class NumericalMethod(Protocol):
    """
    Protocol that all numerical methods must follow.

    This protocol defines the interface for all numerical methods, whether they are
    used for root-finding or optimization. Each method must implement:
    - A step function that performs one iteration
    - Error estimation functionality
    - Convergence checking
    - Method identification
    """

    def step(self) -> float:
        """
        Perform one iteration of the method.

        Each step should improve the approximation toward the solution according to
        the specific algorithm's approach.

        Returns:
            float: Current approximation (root for root-finding, minimum for optimization)
        """
        ...

    def get_current_x(self) -> float:
        """
        Get current x value (current approximation).

        Returns:
            float: Current approximation
        """
        ...

    def get_error(self) -> float:
        """
        Get current error estimate.

        For root-finding methods, error is typically |f(x)|, which measures how close
        the current point is to being a root.

        For optimization methods, error is typically |f'(x)|, which measures how close
        the current point is to being a stationary point.

        Returns:
            float: Current error estimate
        """
        ...

    def has_converged(self) -> bool:
        """
        Check if method has converged to a solution.

        Convergence is typically determined by comparing the error estimate against
        a specified tolerance, or by reaching a maximum number of iterations.

        Returns:
            bool: True if converged, False otherwise
        """
        ...

    @property
    def name(self) -> str:
        """
        Name of the method for display purposes.

        Returns:
            str: Human-readable name of the method
        """
        ...


@dataclass
class IterationData:
    """
    Data structure to store details of each iteration of a numerical method.

    This class captures the state transitions during iterations and provides
    a consistent way to analyze and visualize the progress of numerical methods.

    Attributes:
        iteration: Iteration number (0-indexed)
        x_old: Previous approximation value
        x_new: Current approximation value
        f_old: Function value at previous approximation
        f_new: Function value at current approximation
        error: Current error estimate
        details: Method-specific details (e.g., interval bounds, derivatives)
    """

    iteration: int
    x_old: float
    x_new: float
    f_old: float
    f_new: float
    error: float
    details: Dict[str, Any]


@dataclass
class NumericalMethodConfig:
    """
    Configuration for numerical methods.

    This class encapsulates all parameters needed to initialize and run a numerical
    method, including the target function, convergence criteria, and optional
    derivative information.

    Mathematical context:
    - For root-finding: We seek x such that f(x) = 0
    - For optimization: We seek x such that f'(x) = 0 (for minimization)

    Attributes:
        func: Function to process (find root or minimize)
        method_type: Type of method ("root" or "optimize")
        x_range: Range to plot function (xmin, xmax)
        tol: Error tolerance for convergence criteria
        max_iter: Maximum number of iterations before stopping
        derivative: Optional derivative function for gradient-based methods
        hessian: Optional hessian function for second-order methods
        use_derivative_free: Whether to use derivative-free approach even if derivative is provided
        finite_diff_step: Step size for finite differences when approximating derivatives
        is_2d: Whether the function is multivariate (2D visualization support)
        step_length_method: Method to use for determining step length in optimization
        descent_direction_method: Method to use for determining descent direction in optimization
        step_length_params: Additional parameters for the step length method
        descent_direction_params: Additional parameters for the descent direction method
        initial_step_size: Initial step size for line search methods
    """

    func: Callable[[Union[float, np.ndarray]], float]
    method_type: MethodType
    x_range: Tuple[float, float] = (-10, 10)
    tol: float = 1e-6
    max_iter: int = 100
    derivative: Optional[
        Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
    ] = None
    hessian: Optional[
        Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
    ] = None
    use_derivative_free: bool = False
    finite_diff_step: float = 1e-7
    is_2d: bool = False
    step_length_method: Optional[StepLengthMethod] = None
    descent_direction_method: Optional[DescentDirectionMethod] = None
    step_length_params: Dict[str, Any] = None
    descent_direction_params: Dict[str, Any] = None
    initial_step_size: float = 1.0


class BaseNumericalMethod:
    """
    Base class for numerical methods.

    This class provides common functionality shared by all numerical methods,
    including error calculation, derivative estimation, and iteration tracking.
    Specific methods should inherit from this class and implement the specialized
    step() and get_current_x() methods.

    Mathematical foundation:
    - Root-finding methods seek x where f(x) = 0
    - Optimization methods seek x where f'(x) = 0 (for minimization)
    """

    def __init__(self, config: NumericalMethodConfig):
        """
        Initialize the numerical method with the given configuration.

        Args:
            config: Configuration object containing function, tolerance, and other parameters
        """
        self.func = config.func
        self.method_type = config.method_type
        self.tol = config.tol
        self.max_iter = config.max_iter
        self.derivative = config.derivative
        self.hessian = config.hessian
        self.use_derivative_free = config.use_derivative_free
        self.finite_diff_step = config.finite_diff_step
        self.step_length_method = config.step_length_method
        self.descent_direction_method = config.descent_direction_method
        self.step_length_params = config.step_length_params or {}
        self.descent_direction_params = config.descent_direction_params or {}
        self.initial_step_size = config.initial_step_size
        self._converged = False
        self.iterations = 0
        self._history: List[IterationData] = []

    # ----------------------------------------------------------
    # Core Algorithm Methods - Must be implemented by subclasses
    # ----------------------------------------------------------

    def step(self) -> float:
        """
        Perform one iteration of the method.

        Each step should improve the approximation toward the solution according to
        the specific algorithm's approach.

        Returns:
            float: Current approximation (root for root-finding, minimum for optimization)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement step()")

    def get_current_x(self) -> float:
        """
        Get current x value (current approximation).

        This method must be implemented by subclasses to provide the
        current approximation.

        Returns:
            float: Current approximation

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_current_x()")

    def compute_descent_direction(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute the descent direction at the current point.

        The descent direction depends on the specified method:
        - steepest_descent: -∇f(x)
        - newton: -H(x)^(-1)∇f(x)
        - bfgs, l_bfgs, dfp, sr1: Quasi-Newton approximations
        - conjugate_gradient: Conjugate directions
        - etc.

        This method should be implemented by concrete optimization classes
        to provide the appropriate descent direction based on the
        descent_direction_method specified in the configuration.

        Args:
            x: Current point

        Returns:
            Union[float, np.ndarray]: Descent direction at x
        """
        raise NotImplementedError(
            "Subclasses must implement compute_descent_direction()"
        )

    def compute_step_length(
        self, x: Union[float, np.ndarray], direction: Union[float, np.ndarray]
    ) -> float:
        """
        Compute the step length using the specified line search method.

        The step length is determined based on the current point, descent direction,
        and the specified line search method.

        Args:
            x: Current point
            direction: Descent direction

        Returns:
            float: Step length (alpha)
        """
        raise NotImplementedError("Subclasses must implement compute_step_length()")

    # ---------------------
    # State Access Methods
    # ---------------------

    def get_error(self) -> float:
        """
        Get error at current point based on method type.

        For root-finding:
            - Error = |f(x)|, which measures how close we are to f(x) = 0

        For optimization:
            - With derivative: Error = |f'(x)|, measuring proximity to stationary point
            - Without derivative: Estimate gradient norm using finite differences

        Returns:
            float: Current error estimate
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
                derivative_value = self.derivative(x)
                # Handle both scalar and vector derivatives
                if isinstance(derivative_value, (float, int)):
                    return abs(derivative_value)
                else:
                    # For vector derivatives, return the norm
                    return float(np.linalg.norm(derivative_value))
            else:
                # Fallback to finite differences
                return abs(self.estimate_derivative(x))

    def has_converged(self) -> bool:
        """
        Check if method has converged based on error tolerance or max iterations.

        Returns:
            bool: True if converged, False otherwise
        """
        return self._converged

    @property
    def name(self) -> str:
        """
        Get method name based on class name.

        Returns:
            str: Name of the method
        """
        return self.__class__.__name__

    # -------------------------
    # History Tracking Methods
    # -------------------------

    def add_iteration(
        self, x_old: float, x_new: float, details: Dict[str, Any]
    ) -> None:
        """
        Store iteration data for analysis and visualization.

        This method captures the state before and after each iteration,
        allowing for detailed tracking of the method's progress.

        Args:
            x_old: Previous approximation
            x_new: Current approximation
            details: Method-specific details
        """
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
        """
        Get complete iteration history for analysis.

        Returns:
            List[IterationData]: List of data from all iterations
        """
        return self._history

    def get_last_iteration(self) -> Optional[IterationData]:
        """
        Get data from last iteration.

        Returns:
            Optional[IterationData]: Data from last iteration or None if no iterations
        """
        return self._history[-1] if self._history else None

    # ----------------
    # Utility Methods
    # ----------------

    def estimate_derivative(self, x: float) -> float:
        """
        Estimate derivative using central finite differences.

        The central difference approximation is given by:
        f'(x) ≈ [f(x + h) - f(x - h)] / (2h)

        This provides O(h²) accuracy compared to O(h) for forward/backward differences.

        Args:
            x: Point at which to estimate derivative

        Returns:
            float: Estimated derivative f'(x)
        """
        h = self.finite_diff_step
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def estimate_gradient_norm(self, x: float) -> float:
        """
        Estimate gradient norm for derivative-free methods.

        For single-variable functions, this is simply |f'(x)|.
        For multivariate functions, this would compute ||∇f(x)||.

        Args:
            x: Current point

        Returns:
            float: Estimated gradient norm
        """
        grad = self.estimate_derivative(x)
        return abs(grad)
