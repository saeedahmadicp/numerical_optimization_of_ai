# tests/test_convex/test_secant.py

import pytest
import math
import sys
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


def test_invalid_method_type():
    """Test that initialization fails when method_type is not 'root'"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="can only be used for root finding"):
        SecantMethod(config, 1, 2)


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


def test_convergence_rate():
    """Test that secant method converges superlinearly"""

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


def test_iteration_history():
    """Test that iteration history is properly recorded"""

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


def test_legacy_wrapper():
    """Test the backward-compatible secant_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = secant_search(f, 1, 2)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
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
