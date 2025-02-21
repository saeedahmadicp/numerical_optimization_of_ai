# ui/optimization.py

"""Shared optimization components for the UI."""

import numpy as np
from algorithms.convex.protocols import NumericalMethodConfig
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.nelder_mead import NelderMeadMethod
from algorithms.convex.powell import PowellMethod
from algorithms.convex.steepest_descent import SteepestDescentMethod


# Map method names to their classes
METHOD_MAP = {
    "Newton's Method": NewtonMethod,
    "BFGS": BFGSMethod,
    "Steepest Descent": SteepestDescentMethod,
    "Nelder-Mead": NelderMeadMethod,
    "Powell's Method": PowellMethod,
}


def run_optimization(
    method_class, func, grad, hessian, x0, tol, max_iter, bounds
) -> dict:
    """Run optimization using the specified method and return results."""
    # Create configuration
    config = NumericalMethodConfig(
        func=func,
        derivative=grad,
        method_type="optimize",
        tol=tol,
        max_iter=max_iter,
        x_range=bounds[0],  # Use x1 bounds for range
        is_2d=True,
    )

    # Initialize method
    x0_array = np.array(x0)

    # Only pass second_derivative to Newton's Method
    if method_class == NewtonMethod:
        method = method_class(config, x0_array, second_derivative=hessian)
    else:
        method = method_class(config, x0_array)

    # Run optimization
    path = [x0_array]
    function_values = [func(x0_array)]

    while not method.has_converged() and len(method.get_iteration_history()) < max_iter:
        try:
            method.step()
            current_x = method.get_current_x()
            # Add bounds checking for stability
            current_x = np.clip(
                current_x, [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]]
            )
            path.append(current_x.copy())
            function_values.append(func(current_x))
        except np.linalg.LinAlgError:
            # Handle potential singular Hessian
            break
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            break

    # Get final results
    x_final = method.get_current_x()
    # Ensure final point is within bounds
    x_final = np.clip(
        x_final, [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]]
    )
    f_final = func(x_final)
    success = method.has_converged()
    iterations = len(method.get_iteration_history())

    return {
        "x": x_final,
        "fun": f_final,
        "success": success,
        "message": (
            "Optimization successful" if success else "Maximum iterations reached"
        ),
        "nit": iterations,
        "path": path,
        "function_values": function_values,
    }
