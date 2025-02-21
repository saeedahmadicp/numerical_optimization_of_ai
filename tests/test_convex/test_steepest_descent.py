# tests/test_convex/test_steepest_descent.py

import pytest
import math
import sys
from pathlib import Path

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

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
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
    """Test that initialization fails when derivative is not provided"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="requires derivative function"):
        SteepestDescentMethod(config, x0=1.0)


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

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
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
        assert "line_search" in data.details


def test_legacy_wrapper():
    """Test the backward-compatible steepest_descent_search function"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    minimum, errors, iters = steepest_descent_search(config, x0=1.0)

    assert abs(minimum) < 1e-6
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        # Simple quadratic (easy to optimize)
        (
            lambda x: x**2,
            lambda x: 2 * x,
            1.0,
            1e-4,
            "quadratic",
        ),
        # Scaled quadratic (still easy but different scale)
        (
            lambda x: 0.5 * x**2,
            lambda x: x,
            1.0,
            1e-4,
            "scaled quadratic",
        ),
        # Linear + quadratic (minimum at -2)
        (
            lambda x: x**2 + 4 * x,
            lambda x: 2 * x + 4,
            0.0,
            1e-4,
            "linear-quadratic",
        ),
    ]

    for func, deriv, x0, tol, name in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            derivative=deriv,
            tol=tol,
            max_iter=1000,
        )
        method = SteepestDescentMethod(config, x0=x0, alpha=0.1)

        while not method.has_converged():
            x = method.step()

        grad_norm = abs(deriv(x))
        assert grad_norm < tol * 1.1, (  # Allow 10% tolerance buffer
            f"Function '{name}' did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  Gradient norm: {grad_norm}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )


def test_challenging_functions():
    """Test method behavior with more challenging functions"""

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
    )
    method = SteepestDescentMethod(config, x0=0.5, alpha=0.01)  # Smaller initial step

    while not method.has_converged():
        x = method.step()

    # For challenging functions, verify that:
    # 1. We're close to the minimum (x=0)
    assert abs(x) < 0.1, f"Not close enough to minimum. x={x}"
    # 2. Function value has decreased significantly
    assert f(x) < f(0.5), "Function value did not decrease"
    # 3. We haven't exceeded max iterations
    assert method.iterations < 2000, f"Too many iterations: {method.iterations}"


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, method_type="optimize", derivative=df, max_iter=5
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

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = SteepestDescentMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()
