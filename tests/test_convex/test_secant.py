# tests/test_convex/test_secant.py

import pytest
import math
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.secant import SecantMethod, secant_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = SecantMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_basic_optimization():
    """Test finding minimum of x^2 using secant method for optimization"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = SecantMethod(config, 1.0, 0.5, derivative=df)

    while not method.has_converged():
        x = method.step()

    # Minimum should be at x=0
    assert abs(x) < 1e-6
    assert method.iterations < 20


def test_optimization_without_derivative():
    """Test optimization using finite difference approximation of derivative"""

    def f(x):
        return (x - 2) ** 2 + 1  # Minimum at x=2

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)

    # Create method with a derivative approximation
    h = 1e-7

    def approx_df(x):
        return (f(x + h) - f(x - h)) / (2 * h)

    method = SecantMethod(config, 0.0, 1.0, derivative=approx_df)

    while not method.has_converged():
        x = method.step()

    assert abs(x - 2.0) < 1e-5, f"Expected x≈2.0, got {x}"
    assert method.iterations < 30


def test_missing_derivative_for_optimization():
    """Test that initialization fails when no derivative is provided for optimization"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="requires derivative function"):
        SecantMethod(config, 1.0, 0.5)


def test_optimization_error_calculation():
    """Test that error is calculated as |f'(x)| for optimization"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-6)
    method = SecantMethod(config, 1.0, 0.5, derivative=df)

    # Run a few steps
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    # Error should be |f'(x)| = |2*x|
    x = method.get_current_x()
    expected_error = abs(2 * x)
    actual_error = method.get_error()

    assert (
        abs(actual_error - expected_error) < 1e-10
    ), f"Error calculation incorrect: expected |f'({x})| = {expected_error}, got {actual_error}"


def test_invalid_method_type():
    """Test that initialization succeeds with optimization method type if derivative is provided"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = SecantMethod(config, 1, 2, derivative=df)

    # Should not raise an error when derivative is provided
    assert method.method_type == "optimize"


def test_near_zero_denominator():
    """Test handling of near-zero denominator"""

    def f(x):
        return x**3  # Has root at x=0

    config = NumericalMethodConfig(func=f, method_type="root")
    method = SecantMethod(config, 1e-10, -1e-10)  # Very close points near root

    x = method.step()
    assert method.has_converged()  # Should detect near-zero denominator and stop


def test_exact_root():
    """Test when one initial guess is the root"""

    def f(x):
        return x - 2  # Linear function with root at x=2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = SecantMethod(config, 2, 2.1)  # x0 is exact root

    while not method.has_converged():
        x = method.step()

    assert abs(x - 2) < 1e-6
    assert abs(f(x)) < 1e-6


def test_root_finding_convergence_rate():
    """Test that secant method converges superlinearly for root-finding"""

    def f(x):
        return x**3 - x - 2  # Cubic function

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    method = SecantMethod(config, 1, 2)

    iterations = 0
    while not method.has_converged():
        method.step()
        iterations += 1

    # Secant method should converge faster than bisection
    assert iterations < 20


def test_optimization_convergence_rate():
    """Test that secant method converges for optimization problems"""

    def f(x):
        return (x - 3) ** 2  # Quadratic with minimum at x=3

    def df(x):
        return 2 * (x - 3)  # Linear derivative

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-8)
    method = SecantMethod(config, 1, 2, derivative=df)

    iterations = 0
    while not method.has_converged():
        method.step()
        iterations += 1

    # Should find the minimum at x=3
    assert abs(method.get_current_x() - 3) < 1e-6
    # Should converge in reasonable number of iterations
    assert iterations < 20


def test_root_finding_iteration_history():
    """Test that iteration history is properly recorded for root-finding"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = SecantMethod(config, 1, 3)

    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error generally decreases
    errors = [data.error for data in history]
    assert errors[-1] < errors[0]

    # Check that details contain the expected keys
    for data in history:
        assert "x0" in data.details
        assert "x1" in data.details
        assert "step" in data.details
        assert "denominator" in data.details


def test_optimization_iteration_history():
    """Test that iteration history is properly recorded for optimization"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = SecantMethod(config, 1, 0.5, derivative=df)

    # Record initial iteration + at least one more
    method.step()

    history = method.get_iteration_history()
    # Should have at least the initial iteration plus one step
    assert len(history) >= 2

    # Check that details contain the expected keys for optimization
    for i, data in enumerate(history):
        # Skip checking initial iteration which has some None values
        if i == 0:
            continue
        assert "x0" in data.details
        assert "x1" in data.details
        assert "step" in data.details
        assert "denominator" in data.details
        assert "func(x0)" in data.details
        assert "func(x1)" in data.details
        assert "func(x2)" in data.details


def test_legacy_wrapper_root_finding():
    """Test the backward-compatible secant_search function for root finding"""

    def f(x):
        return x**2 - 2

    root, errors, iters = secant_search(f, 1, 2)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_legacy_wrapper_optimization():
    """Test the backward-compatible secant_search function for optimization"""

    def f(x):
        return (x - 3) ** 2  # Minimum at x=3

    def df(x):
        return 2 * (x - 3)

    minimum, errors, iters = secant_search(
        f, 1, 2, method_type="optimize", derivative=df
    )

    assert abs(minimum - 3) < 1e-6
    assert len(errors) == iters


def test_different_root_finding_functions():
    """Test method works with different types of functions for root-finding"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1, 2),  # Exponential
        (lambda x: math.sin(x), 3, 4),  # Trigonometric
    ]

    for func, x0, x1 in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = SecantMethod(config, x0, x1)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_different_optimization_functions():
    """Test method works with different types of functions for optimization"""
    test_cases = [
        # f(x), df(x), x0, x1, expected_minimum
        (lambda x: x**2, lambda x: 2 * x, 1, 0.5, 0),  # Quadratic
        (lambda x: (x - 3) ** 2, lambda x: 2 * (x - 3), 1, 2, 3),  # Shifted quadratic
        (lambda x: x**4, lambda x: 4 * x**3, 1, 0.5, 0),  # Quartic
    ]

    for func, deriv, x0, x1, expected_min in test_cases:
        config = NumericalMethodConfig(func=func, method_type="optimize", tol=1e-4)
        method = SecantMethod(config, x0, x1, derivative=deriv)

        # Use more iterations to ensure convergence
        method.max_iter = 50

        while not method.has_converged():
            x = method.step()

        # Allow a larger tolerance for quartic functions which converge more slowly
        if func(0) == 0 and func(1) == 1:  # This is the quartic function x^4
            assert (
                abs(x - expected_min) < 5e-2
            ), f"Got {x}, expected close to {expected_min}"
        else:
            assert (
                abs(x - expected_min) < 1e-3
            ), f"Got {x}, expected close to {expected_min}"


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = SecantMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = SecantMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_name_property():
    """Test that the name property returns the correct name for each method type"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    # Root-finding
    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = SecantMethod(config_root, 1, 2)
    assert method_root.name == "Secant Method (Root-Finding)"

    # Optimization
    config_opt = NumericalMethodConfig(func=f, method_type="optimize")
    method_opt = SecantMethod(config_opt, 1, 2, derivative=df)
    assert method_opt.name == "Secant Method (Optimization)"


def test_challenging_optimization():
    """Test secant method on a more challenging optimization problem"""

    def f(x):
        return math.exp(-x) + x**2  # Function with minimum near x=0.5

    def df(x):
        return -math.exp(-x) + 2 * x

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-6)
    method = SecantMethod(config, 0, 1, derivative=df)

    # Allow more iterations for challenging functions
    method.max_iter = 30

    while not method.has_converged():
        method.step()

    # The actual minimum is around x=0.35-0.5
    min_x = method.get_current_x()
    assert 0.3 < min_x < 0.6, f"Expected minimum near x=0.35-0.5, got {min_x}"
    assert abs(df(min_x)) < 1e-5, f"Derivative not close to zero: {df(min_x)}"


def test_compare_with_analytical_minimum():
    """Test that secant method finds the same minimum as analytical calculation"""

    def f(x):
        return 2 * x**2 - 8 * x + 9  # Minimum at x=2

    def df(x):
        return 4 * x - 8

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-6)
    method = SecantMethod(config, 0, 1, derivative=df)

    while not method.has_converged():
        method.step()

    # Analytically, minimum is at x=2
    assert abs(method.get_current_x() - 2.0) < 1e-6
