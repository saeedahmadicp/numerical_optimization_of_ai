# tests/test_convex/test_nelder_mead.py

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.nelder_mead import NelderMeadMethod, nelder_mead_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_optimization():
    """Test finding minimum of x^2"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

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
        NelderMeadMethod(config, x0=1.0)


def test_quadratic_function():
    """Test optimization of quadratic function"""

    def f(x):
        return 2 * x**2 + 4 * x + 1

    config = NumericalMethodConfig(
        func=f,
        method_type="optimize",
        tol=1e-5,  # Adjusted tolerance
        max_iter=500,  # More iterations allowed
    )
    method = NelderMeadMethod(
        config, x0=0.0, delta=0.05
    )  # Start closer, smaller simplex

    while not method.has_converged():
        x = method.step()

    assert abs(x + 1) < 1e-4  # Minimum at x=-1
    assert abs(f(x) - (-1)) < 1e-4  # Minimum value is -1


def test_simplex_operations():
    """Test that simplex operations (reflection, expansion, etc.) work correctly"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0, delta=0.5)

    # Do one step and check the details
    method.step()
    history = method.get_iteration_history()
    details = history[0].details

    # Check that all operations are recorded
    assert "simplex_points" in details
    assert "f_values" in details
    assert "reflection" in details
    assert "action" in details


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=2.0)  # Start further from minimum

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

    # Verify overall progress
    first_iter = history[0]
    last_iter = history[-1]
    assert last_iter.f_new < first_iter.f_old, (
        f"Should make progress from start to end.\n"
        f"Starting value: {first_iter.f_old}, Final value: {last_iter.f_new}"
    )


def test_legacy_wrapper():
    """Test the backward-compatible nelder_mead_search function"""

    def f(x):
        return x**2

    minimum, errors, iters = nelder_mead_search(f, x0=1.0)
    assert abs(minimum) < 1e-4
    assert len(errors) == iters


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        # Simple quadratic (easy to optimize)
        (lambda x: x**2, 0.5, 0.0, 1e-4, 0.05, "quadratic"),
        # Scaled quadratic (still easy)
        (lambda x: 0.5 * x**2, 0.5, 0.0, 1e-4, 0.05, "scaled quadratic"),
        # Linear + quadratic (minimum at -2)
        (lambda x: x**2 + 4 * x, -1.5, -2.0, 1e-3, 0.1, "linear-quadratic"),
    ]

    for func, x0, true_min, tol, delta, name in test_cases:
        config = NumericalMethodConfig(
            func=func,
            method_type="optimize",
            tol=tol,
            max_iter=1000,
        )
        method = NelderMeadMethod(config, x0=x0, delta=delta)

        while not method.has_converged():
            x = method.step()

        assert method.iterations >= 5, f"Too few iterations for {name}"
        assert abs(x - true_min) < tol * 20, (
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
    method = NelderMeadMethod(config, x0=0.5)

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
    method = NelderMeadMethod(config, x0=1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="optimize")
    method = NelderMeadMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()
