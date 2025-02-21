# tests/test_convex/test_powell.py

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.powell import PowellMethod, powell_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_optimization():
    """Test finding minimum of x^2"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, x0=1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-4  # Should find minimum at x=0
    assert method.iterations < 100


def test_invalid_method_type():
    """Test that initialization fails when method_type is not 'optimize'"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="can only be used for optimization"):
        PowellMethod(config, x0=1.0)


def test_quadratic_function():
    """Test optimization of quadratic function"""

    def f(x):
        return 2 * x**2 + 4 * x + 1

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, x0=1.0)

    while not method.has_converged():
        x = method.step()

    assert abs(x + 1) < 1e-4  # Minimum at x=-1
    assert abs(f(x) - (-1)) < 1e-4  # Minimum value is -1


def test_line_search():
    """Test that line search produces decrease in function value"""

    def f(x):
        return x**4  # Steeper function to test line search

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, x0=1.0)

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

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(
        config, x0=2.0
    )  # Start further from minimum for better testing

    # Run for a few iterations or until convergence
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    history = method.get_iteration_history()
    assert len(history) > 0, "Should have at least one iteration"

    # Check that we make progress in each iteration
    for data in history:
        assert data.f_new <= data.f_old, (
            f"Function value should not increase within iteration.\n"
            f"Old value: {data.f_old}, New value: {data.f_new}"
        )

    # Verify overall progress from start to end
    first_iter = history[0]
    last_iter = history[-1]
    assert last_iter.f_new < first_iter.f_old, (
        f"Should make progress from start to end.\n"
        f"Starting value: {first_iter.f_old}, Final value: {last_iter.f_new}"
    )

    # Check that details contain the expected keys
    for data in history:
        assert "alpha" in data.details, "Missing alpha in iteration details"
        assert "direction" in data.details, "Missing direction in iteration details"
        assert "prev_x" in data.details, "Missing prev_x in iteration details"
        assert "prev_fx" in data.details, "Missing prev_fx in iteration details"
        assert "line_search" in data.details, "Missing line_search in iteration details"

        # Check line_search details
        line_search = data.details["line_search"]
        assert "start" in line_search, "Missing start in line_search details"
        assert "step_size" in line_search, "Missing step_size in line_search details"
        assert "direction" in line_search, "Missing direction in line_search details"

    # Verify the convergence
    final_x = method.get_current_x()
    assert abs(final_x) < 1e-4, f"Should converge to minimum at x=0, got {final_x}"


def test_legacy_wrapper():
    """Test the backward-compatible powell_search function"""

    def f(x):
        return x**2

    minimum, errors, iters = powell_search(f, x0=1.0)
    assert abs(minimum) < 1e-4
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        # Simple quadratic
        (lambda x: x**2, 1.0, 0.0, 1e-4, "quadratic"),
        # Scaled quadratic
        (lambda x: 0.5 * x**2, 1.0, 0.0, 1e-4, "scaled quadratic"),
        # Linear + quadratic (minimum at -2)
        (lambda x: x**2 + 4 * x, 0.0, -2.0, 1e-4, "linear-quadratic"),
    ]

    for func, x0, true_min, tol, name in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            tol=tol,
            max_iter=1000,
        )
        method = PowellMethod(config, x0=x0)

        while not method.has_converged():
            x = method.step()

        assert (
            abs(x - true_min) < tol * 10
        ), (  # Allow larger tolerance for derivative-free
            f"Function '{name}' did not converge properly:\n"
            f"  Final x: {x}\n"
            f"  True minimum: {true_min}\n"
            f"  Tolerance: {tol}\n"
            f"  Iterations: {method.iterations}"
        )


def test_challenging_functions():
    """Test method behavior with more challenging functions"""

    def f(x):
        return abs(x) ** 1.5  # Non-smooth at minimum

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        tol=1e-3,
        max_iter=2000,
    )
    method = PowellMethod(config, x0=0.5)

    while not method.has_converged():
        x = method.step()

    # For challenging functions, verify that:
    # 1. We're close to the minimum (x=0)
    assert abs(x) < 0.1, f"Not close enough to minimum. x={x}"
    # 2. Function value has decreased
    assert f(x) < f(0.5), "Function value did not decrease"
    # 3. We haven't exceeded max iterations
    assert method.iterations < 2000, f"Too many iterations: {method.iterations}"


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize", max_iter=5)
    method = PowellMethod(config, x0=1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = PowellMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()
