# tests/test_convex/test_powell_quadratic.py

import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.powell_quadratic import PowellMethod, powell_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_optimization():
    """Test finding minimum of x^2"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, -1.0, 1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-4  # Should find minimum at x=0
    assert method.iterations < 100


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = PowellMethod(config, 1.0, 2.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-4  # Should find root at x=sqrt(2)
    assert method.iterations < 100


def test_quadratic_fit():
    """Test that quadratic fitting works correctly"""

    def f(x):
        return x**2 - 4 * x + 3  # Minimum at x=2, roots at x=1 and x=3

    # Test optimization mode
    config_opt = NumericalMethodConfig(func=f, method_type="optimize")
    method_opt = PowellMethod(config_opt, 0.0, 4.0)

    # Manually set points to make quadratic fit predictable
    method_opt.a, method_opt.b, method_opt.c = 1.0, 2.0, 3.0
    method_opt.fa, method_opt.fb, method_opt.fc = f(1.0), f(2.0), f(3.0)

    # The exact minimum should be at x=2
    u = method_opt._fit_quadratic()
    assert abs(u - 2.0) < 1e-4

    # Test root-finding mode
    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = PowellMethod(config_root, 0.0, 4.0)

    # Manually set points to make quadratic fit predictable
    method_root.a, method_root.b, method_root.c = 0.5, 2.0, 3.5
    method_root.fa, method_root.fb, method_root.fc = f(0.5), f(2.0), f(3.5)

    # Should find a root close to x=1 or x=3
    u = method_root._fit_quadratic()
    assert abs(u - 1.0) < 0.5 or abs(u - 3.0) < 0.5


def test_quadratic_function():
    """Test optimization of quadratic function"""

    def f(x):
        return 2 * x**2 + 4 * x + 1

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, -2.0, 0.0)  # Bracket containing minimum at x=-1

    while not method.has_converged():
        x = method.step()

    assert abs(x + 1) < 1e-4  # Minimum at x=-1
    assert abs(f(x) - (-1)) < 1e-4  # Minimum value is -1


def test_bracket_update():
    """Test that bracketing points are properly updated"""

    def f(x):
        return x**2  # Minimum at x=0

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, -1.0, 1.0)

    # Record initial bracket
    initial_a, initial_c = method.a, method.c

    # Perform one step
    method.step()

    # Bracket should be reduced
    assert method.c - method.a < initial_c - initial_a

    # Test root-finding bracket update
    def g(x):
        return x**2 - 1  # Roots at x=-1 and x=1

    config_root = NumericalMethodConfig(func=g, method_type="root")
    method_root = PowellMethod(config_root, 0.0, 2.0)

    # Record initial bracket
    initial_a, initial_c = method_root.a, method_root.c

    # Perform one step
    method_root.step()

    # Bracket should be reduced
    assert method_root.c - method_root.a < initial_c - initial_a


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(
        config, -2.0, 2.0
    )  # Start with wider bracket for better testing

    # Run for a few iterations or until convergence
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    history = method.get_iteration_history()
    assert len(history) > 0, "Should have at least one iteration"

    # Check that we make progress in each iteration
    for data in history:
        assert (
            "bracket_width" in data.details
        ), "Missing bracket_width in iteration details"
        if "u" in data.details:
            assert "f(u)" in data.details, "Missing f(u) in iteration details"

    # Verify overall progress towards minimum
    assert (
        abs(method.get_current_x()) < 2.0
    ), "Should make progress towards minimum at x=0"


def test_legacy_wrapper():
    """Test the backward-compatible powell_search function"""

    def f(x):
        return x**2

    # Test optimization mode
    min_point, errors, iters = powell_search(f, -1.0, 1.0, method_type="optimize")
    assert abs(min_point) < 1e-4  # Should find minimum at x=0
    assert len(errors) == iters

    # Test root-finding mode
    def g(x):
        return x**2 - 2

    root, errors, iters = powell_search(g, 1.0, 2.0, method_type="root")
    assert abs(root - math.sqrt(2)) < 1e-4  # Should find root at x=sqrt(2)
    assert len(errors) == iters


def test_method_type_validation():
    """Test that method works with both method types"""

    def f(x):
        return x**2 - 1  # Roots at x=-1, x=1; minimum at x=0

    # Test both method types
    config_opt = NumericalMethodConfig(func=f, method_type="optimize")
    method_opt = PowellMethod(config_opt, -1.5, 1.5)

    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = PowellMethod(config_root, 0.5, 1.5)

    # Run both methods
    while not method_opt.has_converged():
        x_opt = method_opt.step()

    while not method_root.has_converged():
        x_root = method_root.step()

    # Verify different results for different method types
    assert abs(x_opt) < 0.5, f"Optimization should find minimum near x=0, got {x_opt}"
    assert (
        abs(x_root - 1.0) < 0.1
    ), f"Root-finding should find root near x=1, got {x_root}"


def test_different_functions():
    """Test method works with different types of functions"""
    opt_test_cases = [
        # Simple quadratic
        (lambda x: x**2, -1.0, 1.0, 0.0, 1e-4, "quadratic"),
        # Scaled quadratic
        (lambda x: 0.5 * x**2, -1.0, 1.0, 0.0, 1e-4, "scaled quadratic"),
        # Linear + quadratic (minimum at -2)
        (lambda x: x**2 + 4 * x, -3.0, -1.0, -2.0, 1e-4, "linear-quadratic"),
    ]

    for func, a, b, true_min, tol, name in opt_test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            tol=tol,
            max_iter=100,
        )
        method = PowellMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(x - true_min) < tol * 10, (  # Allow larger tolerance
            f"Function '{name}' did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  True minimum: {true_min}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )

    # Test for root-finding
    root_test_cases = [
        # Simple polynomial with root at x=2
        (lambda x: x**2 - 4, 1.0, 3.0, 2.0, 1e-4, "quadratic root"),
        # Exponential with root at ln(4)
        (lambda x: math.exp(x) - 4, 1.0, 2.0, math.log(4), 1e-4, "exponential"),
        # Trigonometric with root at pi
        (lambda x: math.sin(x), 3.0, 3.5, math.pi, 1e-4, "sine"),
    ]

    for func, a, b, true_root, tol, name in root_test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="root",
            tol=tol,
            max_iter=100,
        )
        method = PowellMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(x - true_root) < tol * 10, (  # Allow larger tolerance
            f"Function '{name}' did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  True root: {true_root}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize", max_iter=5)
    method = PowellMethod(config, -1.0, 1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, -1.0, 1.0)

    x = method.step()
    assert x == method.get_current_x()

    # Test for root-finding as well
    def g(x):
        return x**2 - 2

    config_root = NumericalMethodConfig(func=g, method_type="root")
    method_root = PowellMethod(config_root, 1.0, 2.0)

    x_root = method_root.step()
    assert x_root == method_root.get_current_x()


def test_convergence_rate():
    """Test that convergence rate estimation works"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, -1.0, 1.0)

    # Run enough iterations to get convergence rate data
    for _ in range(5):
        if method.has_converged():
            break
        method.step()

    rate = method.get_convergence_rate()

    # Powell's quadratic interpolation should converge quickly
    # but the rate will vary depending on the function
    if rate is not None:  # Will be None if not enough iterations
        assert rate >= 0, "Convergence rate should be non-negative"
