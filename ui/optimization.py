# ui/optimization.py

"""
Shared optimization components for the UI.

This module provides a unified interface for various optimization algorithms
implemented in the numerical_optimization_of_ai library. It maps friendly method
names to their corresponding class implementations and provides a standardized
function to run optimization tasks while handling common edge cases.

Mathematical context:
- Optimization seeks to find x* such that f(x*) is minimized
- Different methods have different convergence properties and requirements
- Some methods use gradient information (first derivatives)
- Some methods use Hessian information (second derivatives)
- Some methods are derivative-free
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any, Type
from enum import Enum, auto

from algorithms.convex.protocols import NumericalMethodConfig, BaseNumericalMethod
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.nelder_mead import NelderMeadMethod
from algorithms.convex.powell_quadratic import PowellMethod
from algorithms.convex.powell_conjugate import PowellConjugateMethod
from algorithms.convex.steepest_descent import SteepestDescentMethod
from algorithms.convex.golden_section import GoldenSectionMethod
from algorithms.convex.fibonacci import FibonacciMethod
from algorithms.convex.bisection import BisectionMethod
from algorithms.convex.elimination import EliminationMethod
from algorithms.convex.regula_falsi import RegulaFalsiMethod
from algorithms.convex.secant import SecantMethod


class OptimizationStatus(Enum):
    """Enum representing the status of optimization."""

    SUCCESS = auto()
    MAX_ITERATIONS_REACHED = auto()
    SINGULAR_HESSIAN = auto()
    FUNCTION_EVALUATION_ERROR = auto()
    GRADIENT_EVALUATION_ERROR = auto()
    HESSIAN_EVALUATION_ERROR = auto()
    UNKNOWN_ERROR = auto()


# Map method names to their classes and properties
METHOD_MAP: Dict[str, Type[BaseNumericalMethod]] = {
    "Newton's Method": NewtonMethod,
    "BFGS": BFGSMethod,
    "Steepest Descent": SteepestDescentMethod,
    "Nelder-Mead": NelderMeadMethod,
    "Powell's Quadratic": PowellMethod,
    "Powell's Conjugate": PowellConjugateMethod,
    "Golden Section": GoldenSectionMethod,
    "Fibonacci": FibonacciMethod,
    "Bisection": BisectionMethod,
    "Elimination": EliminationMethod,
    "Regula Falsi": RegulaFalsiMethod,
    "Secant": SecantMethod,
}

# Categorize methods by their derivative requirements
GRADIENT_REQUIRED_METHODS = {
    "Newton's Method",
    "BFGS",
    "Steepest Descent",
}

HESSIAN_REQUIRED_METHODS = {
    "Newton's Method",
}

DERIVATIVE_FREE_METHODS = {
    "Nelder-Mead",
    "Powell's Quadratic",
    "Powell's Conjugate",
    "Golden Section",
    "Fibonacci",
    "Bisection",
    "Elimination",
    "Regula Falsi",
    "Secant",
}


def run_optimization(
    method_class: Type[BaseNumericalMethod],
    func: Callable[[np.ndarray], float],
    grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    x0: Union[float, List[float], np.ndarray] = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    bounds: Optional[List[Tuple[float, float]]] = None,
    step_length_method: str = "backtracking",
    step_length_params: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run optimization using the specified method and return results.

    This function provides a unified interface to various optimization methods,
    handling parameter validation, error cases, and consistent result reporting.

    Args:
        method_class: The optimization method class to use
        func: The objective function to minimize
        grad: The gradient function (first derivative) if available
        hessian: The Hessian function (second derivative) if available
        x0: Initial point for optimization (scalar or vector)
        tol: Tolerance for convergence criteria
        max_iter: Maximum number of iterations
        bounds: List of (min, max) bounds for each dimension
        step_length_method: Method for determining step size
        step_length_params: Parameters for the step length method
        callback: Optional callback function called after each iteration

    Returns:
        Dictionary containing optimization results:
        - x: The solution point
        - fun: Function value at the solution
        - success: Whether optimization succeeded
        - status: Detailed status of the optimization
        - message: Human-readable status message
        - nit: Number of iterations performed
        - path: List of points visited during optimization
        - function_values: Function values at each point in the path
        - gradient_norm: Norm of the gradient at the solution (if available)
    """
    # Ensure x0 is a numpy array
    if not isinstance(x0, np.ndarray):
        if isinstance(x0, (list, tuple)):
            x0_array = np.array(x0, dtype=float)
        else:
            x0_array = np.array([x0], dtype=float)
    else:
        x0_array = x0.copy()

    # Set default bounds if none provided
    if bounds is None:
        # Set wide default bounds
        dim = x0_array.size
        bounds = [(-1e6, 1e6)] * dim

    # Ensure bounds is a list of tuples, one for each dimension
    if len(bounds) != x0_array.size:
        raise ValueError(f"Expected {x0_array.size} bounds but got {len(bounds)}")

    # Ensure initial point is within bounds
    for i, (lower, upper) in enumerate(bounds):
        if x0_array[i] < lower or x0_array[i] > upper:
            x0_array[i] = np.clip(x0_array[i], lower, upper)
            print(
                f"Warning: Initial point component {i} was outside bounds. Clipped to {x0_array[i]}."
            )

    # Create x range for visualization from the first dimension
    x_range = bounds[0] if bounds else (-10, 10)
    y_range = bounds[1] if len(bounds) > 1 else (-10, 10)

    # Set up step length parameters
    if step_length_params is None:
        step_length_params = {}

    # Create configuration
    config = NumericalMethodConfig(
        func=func,
        derivative=grad,
        hessian=hessian,
        method_type="optimize",
        tol=tol,
        max_iter=max_iter,
        x_range=x_range,
        is_2d=len(bounds) > 1,
        step_length_method=step_length_method,
        step_length_params=step_length_params,
    )

    # Initialize method
    try:
        if method_class == NewtonMethod and hessian is not None:
            method = method_class(config, x0_array, second_derivative=hessian)
        else:
            method = method_class(config, x0_array)
    except Exception as e:
        return {
            "x": x0_array,
            "fun": func(x0_array),
            "success": False,
            "status": OptimizationStatus.UNKNOWN_ERROR,
            "message": f"Failed to initialize optimization method: {str(e)}",
            "nit": 0,
            "path": [x0_array],
            "function_values": [func(x0_array)],
            "gradient_norm": np.nan,
        }

    # Run optimization
    path = [x0_array.copy()]
    function_values = [func(x0_array)]
    status = OptimizationStatus.SUCCESS
    error_message = ""

    while not method.has_converged() and len(method.get_iteration_history()) < max_iter:
        try:
            method.step()
            current_x = method.get_current_x()

            # Add bounds checking for stability
            clipped = False
            for i, (lower, upper) in enumerate(bounds):
                if current_x[i] < lower or current_x[i] > upper:
                    current_x[i] = np.clip(current_x[i], lower, upper)
                    clipped = True

            if clipped:
                # If we had to clip, update the method's internal state
                if hasattr(method, "_current_x"):
                    method._current_x = current_x

            path.append(current_x.copy())
            function_values.append(func(current_x))

            # Call optional callback with current state
            if callback is not None:
                callback_info = {
                    "iteration": len(path) - 1,
                    "x": current_x.copy(),
                    "fun": function_values[-1],
                    "path": path.copy(),
                    "function_values": function_values.copy(),
                }
                callback(callback_info)

        except np.linalg.LinAlgError as e:
            status = OptimizationStatus.SINGULAR_HESSIAN
            error_message = f"Singular matrix encountered: {str(e)}"
            break
        except ValueError as e:
            # Function evaluation error often raises ValueError
            status = OptimizationStatus.FUNCTION_EVALUATION_ERROR
            error_message = f"Function evaluation error: {str(e)}"
            break
        except Exception as e:
            status = OptimizationStatus.UNKNOWN_ERROR
            error_message = f"Optimization error: {str(e)}"
            break

    # Check if we reached max iterations
    if not method.has_converged() and status == OptimizationStatus.SUCCESS:
        status = OptimizationStatus.MAX_ITERATIONS_REACHED
        error_message = "Maximum iterations reached without convergence"

    # Get final results
    x_final = method.get_current_x()

    # Ensure final point is within bounds
    for i, (lower, upper) in enumerate(bounds):
        if x_final[i] < lower or x_final[i] > upper:
            x_final[i] = np.clip(x_final[i], lower, upper)

    f_final = func(x_final)
    success = method.has_converged()
    iterations = len(method.get_iteration_history())

    # Compute gradient norm at solution if gradient function is available
    gradient_norm = np.nan
    if grad is not None:
        try:
            gradient = grad(x_final)
            gradient_norm = np.linalg.norm(gradient)
        except Exception:
            pass

    # Format appropriate message
    if success:
        message = "Optimization successful"
    else:
        message = error_message if error_message else "Optimization failed"

    return {
        "x": x_final,
        "fun": f_final,
        "success": success,
        "status": status,
        "message": message,
        "nit": iterations,
        "path": path,
        "function_values": function_values,
        "gradient_norm": gradient_norm,
    }


def get_method_info(method_name: str) -> Dict[str, Any]:
    """
    Get information about an optimization method.

    Args:
        method_name: Name of the optimization method

    Returns:
        Dictionary containing information about the method:
        - requires_gradient: Whether the method requires a gradient function
        - requires_hessian: Whether the method requires a Hessian function
        - is_derivative_free: Whether the method is derivative-free
        - description: Brief description of the method
    """
    requires_gradient = method_name in GRADIENT_REQUIRED_METHODS
    requires_hessian = method_name in HESSIAN_REQUIRED_METHODS
    is_derivative_free = method_name in DERIVATIVE_FREE_METHODS

    descriptions = {
        "Newton's Method": "Second-order method with quadratic convergence near minimum",
        "BFGS": "Quasi-Newton method that approximates the Hessian",
        "DFP": "Quasi-Newton method with Davidon-Fletcher-Powell update",
        "SR1": "Quasi-Newton method with Symmetric Rank 1 update",
        "Steepest Descent": "First-order method using negative gradient direction",
        "Nelder-Mead": "Derivative-free method using simplex operations",
        "Powell's Quadratic": "Derivative-free method using quadratic interpolation",
        "Powell's Conjugate": "Derivative-free method using conjugate directions",
        "Golden Section": "Derivative-free method for univariate optimization",
        "Fibonacci": "Derivative-free method using Fibonacci sequence for interval reduction",
        "Bisection": "Derivative-free method for finding roots by repeatedly halving an interval",
        "Elimination": "Derivative-free method for finding global minima by systematic interval reduction",
        "Regula Falsi": "Hybrid method combining bisection with linear interpolation for root-finding",
        "Secant": "Root-finding method that approximates derivatives using two points",
    }

    description = descriptions.get(method_name, "No description available")

    return {
        "requires_gradient": requires_gradient,
        "requires_hessian": requires_hessian,
        "is_derivative_free": is_derivative_free,
        "description": description,
    }
