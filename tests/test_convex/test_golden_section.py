# tests/test_convex/test_golden_section.py

import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.golden_section import GoldenSectionMethod, golden_section_search
from algorithms.convex.protocols import RootFinderConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_golden_ratio_constants():
    """Test that golden ratio constants are correctly initialized"""

    def f(x):
        return x - 1

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 0, 2)

    # Check phi (golden ratio)
    assert abs(method.phi - (1 + math.sqrt(5)) / 2) < 1e-10
    # Check tau (inverse golden ratio)
    assert abs(method.tau - 1 / method.phi) < 1e-10
    # Verify relationship
    assert abs(method.phi * method.tau - 1) < 1e-10


def test_test_points_placement():
    """Test that test points are properly placed using golden ratio"""

    def f(x):
        return x**2 - 4

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 0, 3)

    # Check initial points
    assert 0 < method.x1 < method.x2 < 3
    # Verify golden ratio relationships
    ratio1 = (method.x1 - method.a) / (method.b - method.a)
    ratio2 = (method.x2 - method.a) / (method.b - method.a)
    assert abs(ratio1 - (1 - method.tau)) < 1e-10
    assert abs(ratio2 - method.tau) < 1e-10


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = RootFinderConfig(func=f, tol=1e-6)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    # Check interval width convergence
    assert abs(method.b - method.a) < method.tol


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Verify history contains golden ratio details
    for data in history:
        assert "tau" in data.details
        assert abs(data.details["tau"] - method.tau) < 1e-10


def test_legacy_wrapper():
    """Test the backward-compatible golden_section_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = golden_section_search(f, 1, 2, tol=1e-6)

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_numerical_stability():
    """Test handling of nearly equal function values"""

    def f(x):
        return x**2  # Function with minimum at x=0

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, -1, 1)

    # Force nearly equal function values
    method.f1 = method.f2 = 1e-10

    # Should not raise any errors
    x = method.step()
    assert -1 <= x <= 1


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (lambda x: math.exp(x) - 4, 1.3, 1.4),  # Exponential
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric
    ]

    for func, a, b in test_cases:
        config = RootFinderConfig(func=func, tol=1e-4)
        method = GoldenSectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_interval_reduction():
    """Test that interval is properly reduced using golden ratio"""

    def f(x):
        return x - 1

    config = RootFinderConfig(func=f)
    method = GoldenSectionMethod(config, 0, 2)

    initial_width = method.b - method.a
    x = method.step()

    # Check interval reduction
    assert method.b - method.a < initial_width
    # Check new point is within bounds
    assert method.a <= x <= method.b
    # Verify reduction ratio approximately follows golden ratio
    reduction_ratio = (method.b - method.a) / initial_width
    assert abs(reduction_ratio - method.tau) < 0.1


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = RootFinderConfig(func=f, max_iter=5)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5
