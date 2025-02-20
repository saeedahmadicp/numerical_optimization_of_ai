# tests/test_convex/test_elimination.py

import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.elimination import EliminationMethod, elimination_search
from algorithms.convex.protocols import RootFinderConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f)
    method = EliminationMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_elimination_step():
    """Test the elimination step logic"""

    def f(x):
        return x**2 - 4  # Roots at x = ±2

    config = RootFinderConfig(func=f)
    method = EliminationMethod(config, 0, 3)

    # Take one step and verify interval reduction
    x = method.step()

    # Interval should be reduced
    assert method.b - method.a < 3
    # New point should be between endpoints
    assert method.a < x < method.b


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = RootFinderConfig(func=f, tol=1e-8)
    method = EliminationMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(f(x)) <= 1e-8


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = RootFinderConfig(func=f)
    method = EliminationMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error generally decreases
    errors = [data.error for data in history]
    assert errors[-1] < errors[0]  # Final error should be less than initial


def test_legacy_wrapper():
    """Test the backward-compatible elimination_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = elimination_search(f, 1, 2, tol=1e-6)

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f, max_iter=5)
    method = EliminationMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1.3, 1.4),  # Exponential
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = RootFinderConfig(func=func, tol=1e-4)
        method = EliminationMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f)
    method = EliminationMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_interval_reduction():
    """Test that interval is properly reduced"""

    def f(x):
        return x - 1  # Simple linear function with root at x=1

    config = RootFinderConfig(func=f)
    method = EliminationMethod(config, 0, 2)

    initial_width = 2  # b - a = 2 - 0

    # Take a few steps
    for _ in range(3):
        method.step()

    final_width = method.b - method.a
    assert final_width < initial_width


def test_convergence_with_interval():
    """Test convergence based on interval width"""

    def f(x):
        return x**3 - x - 2

    config = RootFinderConfig(func=f, tol=1e-6)
    method = EliminationMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    # Check both function value and interval width
    assert abs(f(x)) < 1e-6 or (method.b - method.a) < 1e-6
