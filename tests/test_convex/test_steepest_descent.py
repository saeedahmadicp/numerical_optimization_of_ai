# tests/test_convex/test_steepest_descent.py

import pytest
import math
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.steepest_descent import (
    SteepestDescentMethod,
    steepest_descent_search,
)
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_optimization():
    """Test finding minimum of x^2"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, method_type="optimize", derivative=df, step_length_method="backtracking"
    )
    method = SteepestDescentMethod(config, x0=1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-6  # Should find minimum at x=0
    assert method.iterations < 100


def test_invalid_method_type():
    """Test that initialization fails when method_type is not 'optimize'"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="root", derivative=df)
    with pytest.raises(ValueError, match="can only be used for optimization"):
        SteepestDescentMethod(config, x0=1.0)


def test_missing_derivative():
    """Test that initialization fails when derivative is not provided for vector inputs"""

    def f(x):
        return x**2

    # Use a vector input rather than a scalar, which should trigger the error
    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="requires derivative function"):
        SteepestDescentMethod(config, x0=np.array([1.0, 2.0]))  # 2D vector


def test_quadratic_function():
    """Test optimization of quadratic function"""

    def f(x):
        return 2 * x**2 + 4 * x + 1

    def df(x):
        return 4 * x + 4

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = SteepestDescentMethod(config, x0=1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x + 1) < 1e-6  # Minimum at x=-1
    assert abs(f(x) - (-1)) < 1e-6  # Minimum value is -1


def test_line_search():
    """Test that line search produces decrease in function value"""

    def f(x):
        return x**4  # Steeper function to test line search

    def df(x):
        return 4 * x**3

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = SteepestDescentMethod(config, x0=1.0)

    x_old = method.get_current_x()
    f_old = f(x_old)

    method.step()

    x_new = method.get_current_x()
    f_new = f(x_new)

    assert f_new < f_old, "Line search should decrease function value"


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, method_type="optimize", derivative=df, step_length_method="backtracking"
    )
    method = SteepestDescentMethod(config, x0=1.0)

    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert errors[-1] < errors[0]

    # Check that details contain the expected keys
    for data in history:
        assert "gradient" in data.details
        assert "search_direction" in data.details
        assert "step_size" in data.details
        assert "line_search_method" in data.details  # Updated from "line_search"


def test_legacy_wrapper():
    """Test the backward-compatible steepest_descent_search function with new parameters"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    # Test with default parameters
    minimum1, errors1, iters1 = steepest_descent_search(f, x0=1.0)
    assert abs(minimum1) < 1e-6

    # Test with custom step length method and parameters
    minimum2, errors2, iters2 = steepest_descent_search(
        f,
        x0=1.0,
        step_length_method="strong_wolfe",
        step_length_params={"c1": 1e-4, "c2": 0.1},
    )
    assert abs(minimum2) < 1e-6


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        # Simple quadratic (easy to optimize)
        (
            lambda x: x**2,
            lambda x: 2 * x,
            1.0,
            1e-4,
            "backtracking",
            "quadratic",
        ),
        # Scaled quadratic (still easy but different scale)
        (
            lambda x: 0.5 * x**2,
            lambda x: x,
            1.0,
            1e-4,
            "wolfe",
            "scaled quadratic",
        ),
        # Linear + quadratic (minimum at -2)
        (
            lambda x: x**2 + 4 * x,
            lambda x: 2 * x + 4,
            0.0,
            1e-4,
            "strong_wolfe",
            "linear-quadratic",
        ),
    ]

    for func, deriv, x0, tol, step_method, name in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            derivative=deriv,
            tol=tol,
            max_iter=1000,
            step_length_method=step_method,
        )
        method = SteepestDescentMethod(config, x0=x0)

        while not method.has_converged():
            x = method.step()

        grad_norm = abs(deriv(x))
        assert grad_norm < tol * 1.1, (  # Allow 10% tolerance buffer
            f"Function '{name}' with {step_method} did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  Gradient norm: {grad_norm}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )


def test_challenging_functions():
    """Test method behavior with more challenging functions using different step methods"""
    step_methods = ["backtracking", "wolfe", "strong_wolfe", "goldstein"]

    for step_method in step_methods:

        def f(x):
            return math.exp(x**2)

        def df(x):
            return 2 * x * math.exp(x**2)

        config = NumericalMethodConfig(
            func=f,
            method_type="optimize",
            derivative=df,
            tol=1e-3,
            max_iter=2000,
            step_length_method=step_method,
            initial_step_size=0.01,  # Smaller initial step
        )
        method = SteepestDescentMethod(config, x0=0.5)

        while not method.has_converged():
            x = method.step()

        # For challenging functions, verify that:
        # 1. We're close to the minimum (x=0)
        assert abs(x) < 0.1, f"Not close enough to minimum with {step_method}. x={x}"
        # 2. Function value has decreased significantly
        assert f(x) < f(0.5), f"Function value did not decrease with {step_method}"
        # 3. We haven't exceeded max iterations
        assert (
            method.iterations < 2000
        ), f"Too many iterations with {step_method}: {method.iterations}"


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        derivative=df,
        max_iter=5,
        step_length_method="backtracking",
    )
    method = SteepestDescentMethod(config, x0=1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, method_type="optimize", derivative=df, step_length_method="backtracking"
    )
    method = SteepestDescentMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()


def test_line_search_methods():
    """Test different line search methods"""

    def f(x):
        return x**4  # Steeper function to test line search

    def df(x):
        return 4 * x**3

    # Test each line search method
    line_search_methods = ["backtracking", "wolfe", "strong_wolfe", "goldstein"]
    results = {}

    for method_name in line_search_methods:
        config = NumericalMethodConfig(
            func=f,
            method_type="optimize",
            derivative=df,
            step_length_method=method_name,
            initial_step_size=1.0,
        )
        method = SteepestDescentMethod(config, x0=2.0)

        # Perform several iterations
        for _ in range(10):
            if not method.has_converged():
                method.step()

        results[method_name] = {
            "final_x": method.get_current_x(),
            "function_value": f(method.get_current_x()),
            "iterations": method.iterations,
            "error": method.get_error(),
        }

    # All methods should decrease the function value
    for method_name, result in results.items():
        assert result["function_value"] < f(
            2.0
        ), f"{method_name} failed to decrease function value"
        assert result["error"] < 1.0, f"{method_name} did not reduce gradient norm"


def test_step_length_params():
    """Test customization of step length parameters"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    # Test with default parameters
    config_default = NumericalMethodConfig(
        func=f, method_type="optimize", derivative=df, step_length_method="backtracking"
    )
    method_default = SteepestDescentMethod(config_default, x0=5.0)

    # Test with custom parameters (smaller rho = more aggressive step reduction)
    config_custom = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        derivative=df,
        step_length_method="backtracking",
        step_length_params={"rho": 0.2, "c": 0.01},
    )
    method_custom = SteepestDescentMethod(config_custom, x0=5.0)

    # Run both methods for the same number of iterations
    for _ in range(3):
        method_default.step()
        method_custom.step()

    # The custom method with smaller rho should take smaller steps
    # resulting in different final positions
    assert method_default.get_current_x() != method_custom.get_current_x()

    # But both should decrease the function value
    assert f(method_default.get_current_x()) < f(5.0)
    assert f(method_custom.get_current_x()) < f(5.0)


def test_fixed_step_size():
    """Test fixed step size method"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        derivative=df,
        step_length_method="fixed",
        step_length_params={"step_size": 0.1},
    )
    method = SteepestDescentMethod(config, x0=1.0)

    x_old = method.get_current_x()
    method.step()
    x_new = method.get_current_x()

    # With fixed step size, we expect the step to be exactly 0.1 in the appropriate direction
    expected_step = 0.1 * 2  # step_size * |gradient at x=1|
    assert abs(x_old - x_new - expected_step) < 1e-10

    # Should still converge to minimum
    while not method.has_converged():
        method.step()

    assert abs(method.get_current_x()) < 1e-6


def test_numerical_derivatives():
    """Test that method works with numerical derivatives"""

    def f(x):
        return x**2

    # No derivative provided
    config = NumericalMethodConfig(
        func=f, method_type="optimize", step_length_method="backtracking"
    )
    method = SteepestDescentMethod(config, x0=1.0)

    # Should be able to compute descent direction and step length
    p = method.compute_descent_direction(1.0)
    alpha = method.compute_step_length(1.0, p)

    assert p < 0  # Should be negative for x > 0
    assert alpha > 0  # Step size should be positive

    # Run a full optimization
    while not method.has_converged():
        method.step()

    assert (
        abs(method.get_current_x()) < 1e-4
    )  # Less precise due to numerical derivatives
