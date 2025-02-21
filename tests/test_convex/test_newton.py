# tests/test_convex/test_newton.py

import pytest
import math
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.newton import NewtonMethod, newton_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_root_finding():
    """Test basic root finding with x^2 - 2"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root", tol=1e-6)
    method = NewtonMethod(config, x0=1.5)

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
    method = NewtonMethod(config, x0=1.0, second_derivative=d2f)

    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-6  # Should find minimum at x=0
    assert method.iterations < 10


def test_missing_derivative():
    """Test that initialization fails when derivative is missing"""

    def f(x):
        return x**2

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="requires derivative"):
        NewtonMethod(config, x0=1.0)


def test_missing_second_derivative():
    """Test that initialization fails when second derivative is missing in optimization mode"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="optimize")
    with pytest.raises(ValueError, match="requires second derivative"):
        NewtonMethod(config, x0=1.0)


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root")
    method = NewtonMethod(config, x0=2.0)

    # Run for a few iterations
    for _ in range(3):
        if method.has_converged():
            break
        method.step()

    history = method.get_iteration_history()
    assert len(history) > 0, "Should have at least one iteration"

    # Check that details contain the expected keys
    for data in history:
        assert "f(x)" in data.details
        assert "f'(x)" in data.details
        assert "step" in data.details

    # Verify progress
    first_iter = history[0]
    last_iter = history[-1]
    assert abs(last_iter.f_new) < abs(
        first_iter.f_old
    ), "Should make progress towards root"


def test_optimization_history():
    """Test iteration history for optimization mode"""

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    def d2f(x):
        return 2.0

    config = NumericalMethodConfig(func=f, derivative=df, method_type="optimize")
    method = NewtonMethod(config, x0=2.0, second_derivative=d2f)

    method.step()
    history = method.get_iteration_history()
    details = history[0].details

    assert "f'(x)" in details
    assert "f''(x)" in details
    assert "step" in details


def test_legacy_wrapper_root():
    """Test the backward-compatible newton_search function for root finding"""

    def f(x):
        return x**2 - 2

    root, errors, iters = newton_search(f, x0=1.5)
    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_legacy_wrapper_optimize():
    """Test the backward-compatible newton_search function for optimization"""

    def f(x):
        return x**2

    minimum, errors, iters = newton_search(f, x0=1.0, method_type="optimize")
    assert abs(minimum) < 1e-6
    assert len(errors) == iters


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
        # Quadratic (well-behaved)
        (lambda x: x**2, lambda x: 2 * x, lambda x: 2.0, 1.0, 0.0, 1e-6, "quadratic"),
        # Quartic (more challenging near minimum)
        (
            lambda x: x**4,  # Function
            lambda x: 4 * x**3,  # First derivative
            lambda x: 12 * x**2,  # Second derivative
            0.5,  # Start further from minimum for meaningful improvement
            0.0,  # True minimum
            1e-2,  # Much looser tolerance for quartic
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
            max_iter=100,
        )
        method = NewtonMethod(config, x0=x0)

        while not method.has_converged():
            x = method.step()

        assert abs(x - true_root) < tol, (
            f"Function '{name}' did not find root properly:\n"
            f"  Found: {x}\n"
            f"  Expected: {true_root}\n"
            f"  Error: {abs(x - true_root)}"
        )

    # Test optimization with function-specific checks
    for func, deriv, d2f, x0, true_min, tol, name in opt_cases:
        config = NumericalMethodConfig(
            func=func,
            derivative=deriv,
            method_type="optimize",
            tol=tol,
            max_iter=200,
        )
        method = NewtonMethod(config, x0=x0, second_derivative=d2f)

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
            # For quartic, check both relative improvements
            rel_improvement = (f0 - fx) / f0  # Relative improvement in function value
            rel_grad_improvement = (df0 - dfx) / df0  # Relative improvement in gradient

            assert rel_improvement > 0.9, (  # Should improve by at least 90%
                f"Function value not decreased enough:\n"
                f"  Initial: {f0}\n"
                f"  Final: {fx}\n"
                f"  Relative improvement: {rel_improvement:.2%}"
            )
            assert rel_grad_improvement > 0.9, (  # Gradient should decrease by 90%
                f"Gradient not decreased enough:\n"
                f"  Initial: {df0}\n"
                f"  Final: {dfx}\n"
                f"  Relative improvement: {rel_grad_improvement:.2%}"
            )
        else:
            # For well-behaved functions, use standard tolerance
            assert abs(x - true_min) < tol, (
                f"Function '{name}' did not find minimum properly:\n"
                f"  Found: {x}\n"
                f"  Expected: {true_min}\n"
                f"  Error: {abs(x - true_min)}"
            )


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="root", max_iter=5
    )
    method = NewtonMethod(config, x0=1.0)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root")
    method = NewtonMethod(config, x0=1.0)

    x = method.step()
    assert x == method.get_current_x()


def test_rosenbrock_optimization():
    """Test optimization of Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2"""

    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def df(x):
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
        dy = 200 * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    def d2f(x):
        # Hessian matrix
        h11 = 2 - 400 * x[1] + 1200 * x[0] ** 2
        h12 = -400 * x[0]
        h21 = -400 * x[0]
        h22 = 200
        return np.array([[h11, h12], [h21, h22]])

    # Test cases with different initial points
    test_cases = [
        (np.array([0.0, 0.0]), "far"),  # Far from minimum
        (np.array([0.8, 0.6]), "near"),  # Near minimum
    ]

    for x0, case_name in test_cases:
        config = NumericalMethodConfig(
            func=f,
            derivative=df,
            method_type="optimize",
            tol=1e-5,
            max_iter=100,
        )
        method = NewtonMethod(config, x0, second_derivative=d2f)

        # Store initial values
        f0 = f(x0)
        df0 = np.linalg.norm(df(x0))

        history = []
        while not method.has_converged():
            x = method.step()
            history.append(
                {"x": x.copy(), "f": f(x), "grad_norm": np.linalg.norm(df(x))}
            )

        x_final = x
        f_final = f(x_final)
        df_final = np.linalg.norm(df(x_final))

        # Check convergence to known minimum [1,1]
        assert np.allclose(x_final, [1.0, 1.0], rtol=1e-4), (
            f"Failed to find minimum from {case_name} start:\n"
            f"  Found: {x_final}\n"
            f"  Expected: [1.0, 1.0]\n"
            f"  Error: {np.linalg.norm(x_final - [1.0, 1.0])}"
        )

        # Check gradient norm meets tolerance
        assert df_final < 1e-5, (
            f"Gradient norm not small enough from {case_name} start:\n"
            f"  Found: {df_final}\n"
            f"  Expected: < 1e-5"
        )

        # Verify monotonic decrease in function value
        for i in range(1, len(history)):
            assert (
                history[i]["f"] <= history[i - 1]["f"]
            ), f"Function value increased at iteration {i}"


def test_scaled_quadratic():
    """Test optimization of scaled quadratic f(x,y) = x^2 + ay^2"""

    def make_functions(a):
        def f(x):
            return x[0] ** 2 + a * x[1] ** 2

        def df(x):
            return np.array([2 * x[0], 2 * a * x[1]])

        def d2f(x):
            return np.array([[2.0, 0.0], [0.0, 2 * a]])

        return f, df, d2f

    # Test cases with different scalings and starting points
    test_cases = [
        (1, np.array([2.0, 2.0]), "well-scaled"),
        (1, np.array([-5.0, 5.0]), "well-scaled"),
        (1, np.array([7.0, 8.0]), "well-scaled"),
        (100, np.array([2.0, 2.0]), "poorly-scaled"),
        (100, np.array([-5.0, 5.0]), "poorly-scaled"),
        (100, np.array([7.0, 8.0]), "poorly-scaled"),
    ]

    for a, x0, case_name in test_cases:
        f, df, d2f = make_functions(a)

        config = NumericalMethodConfig(
            func=f,
            derivative=df,
            method_type="optimize",
            tol=1e-6,
            max_iter=100,
        )
        method = NewtonMethod(config, x0, second_derivative=d2f)

        iterations = 0
        while not method.has_converged():
            x = method.step()
            iterations += 1

        # Newton's method should converge in very few iterations
        assert iterations <= 3, (
            f"Newton method took too many iterations for {case_name} case:\n"
            f"  a = {a}\n"
            f"  x0 = {x0}\n"
            f"  iterations = {iterations}"
        )

        # Check convergence to minimum at origin
        assert np.allclose(x, [0.0, 0.0], rtol=1e-5), (
            f"Failed to find minimum for {case_name} case:\n"
            f"  a = {a}\n"
            f"  Found: {x}\n"
            f"  Expected: [0.0, 0.0]"
        )
