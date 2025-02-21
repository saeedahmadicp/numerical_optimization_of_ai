# tests/test_convex/test_bisection.py

import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.bisection import BisectionMethod, bisection_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 1, 2)

    # Run until convergence
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
        BisectionMethod(config, 1, 2)


def test_invalid_interval():
    """Test that initialization fails when f(a) and f(b) have same sign"""

    def f(x):
        return x**2 + 1  # Always positive

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="must have opposite signs"):
        BisectionMethod(config, 1, 2)


def test_exact_root():
    """Test when one endpoint is close to the root"""

    def f(x):
        return x - 2  # Linear function with root at x=2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6, max_iter=100)
    method = BisectionMethod(config, 1.999, 2.001)  # Even tighter interval around x=2

    # Run until convergence or max iterations
    x = method.step()
    for _ in range(20):  # Ensure enough iterations for convergence
        if method.has_converged():
            break
        x = method.step()

    # Verify we found the root
    assert method.has_converged(), "Method did not converge"
    assert abs(f(x)) < 1e-6, f"Function value {f(x)} not within tolerance"
    assert abs(x - 2) < 1e-6, f"x value {x} not close enough to root"


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    method = BisectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(f(x)) <= 1e-8


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))


def test_legacy_wrapper():
    """Test the backward-compatible bisection_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = bisection_search(f, 1, 2, tol=1e-6)

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = BisectionMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (
            lambda x: math.exp(x) - 4,
            1.3,
            1.4,
        ),  # Exponential: tighter interval around ln(4)
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric: tighter interval around pi
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = BisectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()
