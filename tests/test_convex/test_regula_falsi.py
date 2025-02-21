# tests/test_convex/test_regula_falsi.py

import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.regula_falsi import RegulaFalsiMethod, regula_falsi_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 2)

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
        RegulaFalsiMethod(config, 1, 2)


def test_invalid_interval():
    """Test that initialization fails when f(a) and f(b) have same sign"""

    def f(x):
        return x**2 + 1  # Always positive

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="must have opposite signs"):
        RegulaFalsiMethod(config, 1, 2)


def test_exact_root():
    """Test when one endpoint is close to the root"""

    def f(x):
        return x - 2  # Linear function with root at x=2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = RegulaFalsiMethod(config, 1.999, 2.001)

    while not method.has_converged():
        x = method.step()

    assert abs(x - 2) < 1e-6
    assert abs(f(x)) < 1e-6


def test_convergence_rate():
    """Test that regula falsi converges faster than bisection for some functions"""

    def f(x):
        return x**3 - x - 2  # Cubic function

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    method = RegulaFalsiMethod(config, 1, 2)

    iterations = 0
    while not method.has_converged():
        method.step()
        iterations += 1

    # Regula falsi should converge in fewer iterations than bisection would need
    assert iterations < 30  # Bisection typically needs more


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 1, 3)

    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    # Check that details contain the expected keys
    for data in history:
        assert "a" in data.details
        assert "b" in data.details
        assert "updated_end" in data.details


def test_legacy_wrapper():
    """Test the backward-compatible regula_falsi_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = regula_falsi_search(f, 1, 2)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1, 2),  # Exponential
        (lambda x: math.sin(x), 3, 4),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = RegulaFalsiMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_weighted_average():
    """Test the weighted average calculation"""

    def f(x):
        return x - 1  # Linear function with root at x=1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = RegulaFalsiMethod(config, 0, 2)

    # For a linear function, regula falsi should find the root in one step
    x = method.step()
    assert abs(x - 1) < 1e-10


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = RegulaFalsiMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5
