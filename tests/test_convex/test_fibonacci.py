# tests/test_convex/test_fibonacci.py

import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.fibonacci import (
    FibonacciMethod,
    fibonacci_search,
    fib_generator,
)
from algorithms.convex.protocols import NumericalMethodConfig


def test_fib_generator():
    """Test the Fibonacci sequence generator"""
    # Test empty sequence
    assert fib_generator(0) == []
    # Test single term
    assert fib_generator(1) == [1]
    # Test multiple terms
    assert fib_generator(5) == [1, 1, 2, 3, 5]
    # Test longer sequence
    fib = fib_generator(10)
    assert len(fib) == 10
    assert fib[-1] == 55  # 10th Fibonacci number


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_fibonacci_points():
    """Test that test points are properly placed using Fibonacci ratios"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 0, 3, n_terms=10)

    # Check initial points
    assert 0 < method.x1 < method.x2 < 3
    # Check points are properly ordered
    assert method.a < method.x1 < method.x2 < method.b


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = FibonacciMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(f(x)) <= 1e-6


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Verify history contains Fibonacci terms
    for data in history:
        assert "fib_term" in data.details


def test_legacy_wrapper():
    """Test the backward-compatible fibonacci_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = fibonacci_search(
        f,
        1,
        2,
        n_terms=30,
        tol=1e-5,
        method_type="root",  # Ensure method_type is specified
    )

    assert abs(root - math.sqrt(2)) < 1e-5  # Relaxed tolerance
    assert len(errors) == iters


def test_fibonacci_exhaustion():
    """Test convergence when Fibonacci terms are exhausted"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 1, 2, n_terms=5)  # Small number of terms

    while not method.has_converged():
        x = method.step()

    # Should converge when current_term < 3
    assert method.current_term < 3
    assert method.has_converged()


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1.3, 1.4),  # Exponential
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = FibonacciMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_interval_reduction():
    """Test that interval is properly reduced using Fibonacci ratios"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
    method = FibonacciMethod(config, 0, 2)

    initial_width = method.b - method.a
    x = method.step()

    # Check interval reduction
    assert method.b - method.a < initial_width
    # Check new point is within bounds
    assert method.a <= x <= method.b


def test_n_terms_validation():
    """Test handling of different n_terms values"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")

    # Test with minimum viable terms
    method = FibonacciMethod(config, 1, 2, n_terms=3)
    assert len(method.fib) == 4  # n_terms + 1

    # Test with larger number of terms
    method = FibonacciMethod(config, 1, 2, n_terms=30)
    assert len(method.fib) == 31
    assert method.fib[-1] > method.fib[-2]  # Verify sequence is increasing


def test_optimization_mode():
    """Test that the method works for optimization problems"""

    def f(x):
        # Simple quadratic function with minimum at x=2
        return (x - 2) ** 2

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)
    method = FibonacciMethod(config, 0, 4)

    while not method.has_converged():
        x = method.step()

    # The minimum should be close to x=2
    assert abs(x - 2) < 1e-4

    # Function value should be close to zero at minimum
    assert f(x) < 1e-8
