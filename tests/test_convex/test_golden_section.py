# tests/test_convex/test_golden_section.py

import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.golden_section import GoldenSectionMethod, golden_section_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_golden_ratio_constants():
    """Test that golden ratio constants are correctly initialized"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
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

    config = NumericalMethodConfig(func=f, method_type="root")
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

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    # Check convergence to root
    assert abs(f(x)) < 1e-6


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
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

    root, errors, iters = golden_section_search(f, 1, 2, tol=1e-6, method_type="root")

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_numerical_stability():
    """Test handling of nearly equal function values"""

    def f(x):
        return x**2  # Function with minimum at x=0

    config = NumericalMethodConfig(func=f, method_type="optimize")
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
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = GoldenSectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_interval_reduction():
    """Test that interval is properly reduced using golden ratio"""

    def f(x):
        return x - 1

    config = NumericalMethodConfig(func=f, method_type="root")
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

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = GoldenSectionMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_basic_optimization():
    """Test finding the minimum of a simple quadratic function"""

    def f(x):
        # Simple quadratic function with minimum at x=2
        return (x - 2) ** 2

    config = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)
    method = GoldenSectionMethod(config, 0, 4)

    while not method.has_converged():
        x = method.step()

    # The minimum should be close to x=2
    assert abs(x - 2) < 1e-4

    # Function value should be close to zero at minimum
    assert f(x) < 1e-8


def test_optimization_vs_root_finding():
    """Test that the method behaves differently in optimization vs root-finding modes"""

    def f(x):
        # Function with a root at x=2 and minimum at x=1.5
        return (x - 2) * (x - 1)

    # Test in optimization mode (should find minimum at x=1.5)
    config_opt = NumericalMethodConfig(func=f, method_type="optimize", tol=1e-5)
    method_opt = GoldenSectionMethod(config_opt, 0, 3)

    while not method_opt.has_converged():
        x_opt = method_opt.step()

    # Test in root-finding mode (should find root at x=2)
    config_root = NumericalMethodConfig(func=f, method_type="root", tol=1e-5)
    method_root = GoldenSectionMethod(config_root, 1.5, 3)

    while not method_root.has_converged():
        x_root = method_root.step()

    # Check optimization mode found the minimum close to x=1.5
    assert abs(x_opt - 1.5) < 1e-4

    # Check root-finding mode found the root close to x=2
    assert abs(x_root - 2) < 1e-4


def test_convergence_rate():
    """Test that convergence rate estimation works"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = GoldenSectionMethod(config, 1, 2)

    # Run enough iterations to get convergence rate data
    for _ in range(5):
        method.step()

    rate = method.get_convergence_rate()

    # Rate should be close to the inverse golden ratio (0.618)
    if rate is not None:  # Will be None if not enough iterations
        assert 0.5 < rate < 0.7  # Allow some flexibility
