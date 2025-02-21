# tests/test_convex/test_newton_hessian.py

import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.newton_hessian import NewtonHessianMethod, newton_hessian_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_root_finding():
    """Test basic root finding with x^2 - 2"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root", tol=1e-6)
    method = NewtonHessianMethod(config, x0=1.5)

    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 10


def test_optimization():
    """Test basic optimization with x^2"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    def d2f(x):
        return 2.0

    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6
    )
    method = NewtonHessianMethod(config, x0=1.0, second_derivative=d2f)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-6  # Should find minimum at x=0
    assert method.iterations < 10


def test_auto_diff_optimization():
    """Test optimization using automatic differentiation for Hessian"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6
    )
    method = NewtonHessianMethod(config, x0=1.0)  # No second_derivative provided

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-6
    assert method.iterations < 10


def test_different_functions():
    """Test method works with different types of functions"""
    root_cases = [
        # Simple quadratic
        (lambda x: x**2 - 4, lambda x: 2 * x, None, 2.5, 2.0, 1e-6, "sqrt(4)"),
        # Exponential
        (
            lambda x: math.exp(x) - 2,
            lambda x: math.exp(x),
            None,
            1.0,
            math.log(2),
            1e-6,
            "log(2)",
        ),
    ]

    opt_cases = [
        # Quadratic
        (lambda x: x**2, lambda x: 2 * x, lambda x: 2.0, 1.0, 0.0, 1e-6, "quadratic"),
        # Quartic
        (
            lambda x: x**4,
            lambda x: 4 * x**3,
            lambda x: 12 * x**2,
            0.5,
            0.0,
            1e-4,
            "quartic",
        ),
    ]

    # Test root finding
    for func, deriv, _, x0, true_root, tol, name in root_cases:
        config = NumericalMethodConfig(
            func=func,
            derivative=deriv,
            method_type="root",
            tol=tol,
        )
        method = NewtonHessianMethod(config, x0=x0)

        while not method.has_converged():
            x = method.step()

        assert abs(x - true_root) < tol, (
            f"Function '{name}' did not find root properly:\n"
            f"  Found: {x}\n"
            f"  Expected: {true_root}\n"
            f"  Error: {abs(x - true_root)}"
        )

    # Test optimization
    for func, deriv, d2f, x0, true_min, tol, name in opt_cases:
        config = NumericalMethodConfig(
            func=func,
            derivative=deriv,
            method_type="optimize",
            tol=tol,
        )
        method = NewtonHessianMethod(config, x0=x0, second_derivative=d2f)

        # Store initial values
        f0 = func(x0)
        df0 = abs(deriv(x0))

        while not method.has_converged():
            x = method.step()

        # Final values
        fx = func(x)
        dfx = abs(deriv(x))

        # Check convergence based on function type
        if name == "quartic":
            # For quartic, check relative improvements
            rel_improvement = (f0 - fx) / f0
            rel_grad_improvement = (df0 - dfx) / df0

            assert rel_improvement > 0.9, (
                f"Function value not decreased enough:\n"
                f"  Initial: {f0}\n"
                f"  Final: {fx}\n"
                f"  Relative improvement: {rel_improvement:.2%}"
            )
            assert rel_grad_improvement > 0.9, (
                f"Gradient not decreased enough:\n"
                f"  Initial: {df0}\n"
                f"  Final: {dfx}\n"
                f"  Relative improvement: {rel_grad_improvement:.2%}"
            )
        else:
            assert abs(x - true_min) < tol, (
                f"Function '{name}' did not find minimum properly:\n"
                f"  Found: {x}\n"
                f"  Expected: {true_min}\n"
                f"  Error: {abs(x - true_min)}"
            )


def test_missing_derivative():
    """Test that initialization fails when derivative is missing"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="requires derivative"):
        NewtonHessianMethod(config, x0=1.0)


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root")
    method = NewtonHessianMethod(config, x0=2.0)

    # Run for a few iterations
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    history = method.get_iteration_history()
    assert len(history) > 0

    # Check that details contain expected keys
    for data in history:
        assert "f(x)" in data.details
        assert "f'(x)" in data.details
        assert "step" in data.details

    # Verify progress
    first_iter = history[0]
    last_iter = history[-1]
    assert abs(last_iter.f_new) < abs(first_iter.f_old)


def test_legacy_wrapper():
    """Test the backward-compatible newton_hessian_search function"""

    def f(x):
        return x**2 - 2

    # Test root finding
    root, errors, iters = newton_hessian_search(f, x0=1.5)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters

    # Test optimization
    minimum, errors, iters = newton_hessian_search(f, x0=1.0, method_type="optimize")
    assert abs(minimum) < 1e-6
    assert len(errors) == iters
