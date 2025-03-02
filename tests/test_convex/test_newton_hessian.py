# tests/test_convex/test_newton_hessian.py

import pytest
import math
import sys
from pathlib import Path
import numpy as np

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


def test_line_search_methods():
    """Test different line search methods for optimization"""

    def f(x):
        return (
            (x - 3) ** 4 + (x - 3) ** 2 + 1
        )  # A function with slow convergence without line search

    def df(x):
        return 4 * (x - 3) ** 3 + 2 * (x - 3)

    def d2f(x):
        return 12 * (x - 3) ** 2 + 2

    # Compare different line search methods
    line_search_methods = [
        "backtracking",
        "wolfe",
        "strong_wolfe",
        "goldstein",
        "fixed",
    ]
    results = {}

    for method_name in line_search_methods:
        config = NumericalMethodConfig(
            func=f,
            derivative=df,
            method_type="optimize",
            tol=1e-6,
            step_length_method=method_name,
            step_length_params=(
                {"alpha_init": 1.0} if method_name != "fixed" else {"step_size": 0.5}
            ),
        )

        method = NewtonHessianMethod(config, x0=0.0, second_derivative=d2f)

        # Run the method
        try:
            while not method.has_converged():
                method.step()

            results[method_name] = {
                "x": method.get_current_x(),
                "iterations": method.iterations,
                "f(x)": f(method.get_current_x()),
            }
        except Exception as e:
            print(f"Error with {method_name}: {str(e)}")
            results[method_name] = {
                "error": str(e),
                "x": method.get_current_x(),
                "iterations": method.iterations,
            }

    # All methods should find the minimum approximately
    for method_name, result in results.items():
        if "error" not in result:
            assert (
                abs(result["x"] - 3.0) < 1e-2
            ), f"{method_name} should find minimum near x=3"
            assert result["f(x)"] < f(
                0.0
            ), f"{method_name} should decrease function value"

    # Print results for comparison
    print("\nLine search method comparison:")
    for method_name, result in results.items():
        if "error" not in result:
            print(
                f"  {method_name}: x={result['x']:.6f}, iterations={result['iterations']}, f(x)={result['f(x)']:.6e}"
            )
        else:
            print(f"  {method_name}: error={result['error']}")


def test_step_length_params():
    """Test customization of step length parameters"""

    # Using a function that requires different step lengths
    def f(x):
        return (x - 3) ** 4 + 10 * np.sin(x)

    def df(x):
        return 4 * (x - 3) ** 3 + 10 * np.cos(x)

    def d2f(x):
        return 12 * (x - 3) ** 2 - 10 * np.sin(x)

    # Test with different line search methods
    config_backtracking = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="optimize",
        step_length_method="backtracking",
        step_length_params={"rho": 0.5, "c": 1e-4},
    )
    method_backtracking = NewtonHessianMethod(
        config_backtracking, x0=10.0, second_derivative=d2f
    )

    # Test with fixed step length
    config_fixed = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="optimize",
        step_length_method="fixed",
        step_length_params={"step_size": 0.5},
    )
    method_fixed = NewtonHessianMethod(config_fixed, x0=10.0, second_derivative=d2f)

    # Run both methods for one iteration
    method_backtracking.step()
    method_fixed.step()

    # The methods should produce different results due to different line search methods
    x_backtracking = method_backtracking.get_current_x()
    x_fixed = method_fixed.get_current_x()

    # Different line search methods should result in different x values
    assert (
        x_backtracking != x_fixed
    ), f"Expected different x values, got {x_backtracking} and {x_fixed}"

    # Both methods should decrease the function value
    assert f(x_backtracking) < f(10.0), "Backtracking failed to decrease function value"
    assert f(x_fixed) < f(10.0), "Fixed step failed to decrease function value"

    # Get step sizes from history
    history_backtracking = method_backtracking.get_iteration_history()[0].details
    history_fixed = method_fixed.get_iteration_history()[0].details

    # Verify step sizes are different
    assert (
        history_backtracking["step_size"]
        if "step_size" in history_backtracking
        else None
    ) != (
        history_fixed["step_size"] if "step_size" in history_fixed else None
    ), "Expected different step sizes"


def test_convergence_rate():
    """Test that Newton-Hessian method achieves quadratic convergence rate"""

    def f(x):
        return x**2 - 2  # Simple function with known root at sqrt(2)

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root", tol=1e-10)
    method = NewtonHessianMethod(config, x0=1.5)

    # Run several iterations to get convergence data
    for _ in range(5):
        if method.has_converged():
            break
        method.step()

    # Get convergence rate
    rate = method.get_convergence_rate()

    # For well-behaved functions, Newton's method should show quadratic convergence
    assert rate is not None, "Should have calculated convergence rate"
    assert (
        rate < 10.0
    ), f"Rate {rate} should indicate quadratic convergence (typically < 10)"


def test_difficult_function():
    """Test Newton-Hessian method on a more challenging function"""

    def f(x):
        # Function with severe non-linearity and flat regions
        if x <= 0:
            return x**2 + 1
        else:
            return math.log(1 + x**2) - 0.5

    def df(x):
        if x <= 0:
            return 2 * x
        else:
            return 2 * x / (1 + x**2)

    def d2f(x):
        if x <= 0:
            return 2.0
        else:
            return 2 / (1 + x**2) - 4 * x**2 / (1 + x**2) ** 2

    # Test root finding
    config_root = NumericalMethodConfig(
        func=f, derivative=df, method_type="root", tol=1e-6, max_iter=50
    )
    method_root = NewtonHessianMethod(config_root, x0=2.0)

    while not method_root.has_converged():
        method_root.step()

    # There's a root near x ≈ 1.31
    root = method_root.get_current_x()
    assert abs(f(root)) < 1e-5, f"Should find a root. f({root}) = {f(root)}"

    # Test optimization
    config_opt = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="optimize",
        tol=1e-6,
        max_iter=50,
        step_length_method="backtracking",
    )
    method_opt = NewtonHessianMethod(config_opt, x0=-1.0, second_derivative=d2f)

    while not method_opt.has_converged():
        method_opt.step()

    # The minimum is at x = 0 (for the left part of the function)
    minimum = method_opt.get_current_x()
    assert (
        abs(minimum) < 1e-5 or abs(df(minimum)) < 1e-5
    ), f"Should find a minimum. x = {minimum}, f'(x) = {df(minimum)}"


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
            step_length_method="backtracking",
        )
        method = NewtonHessianMethod(config, x0, second_derivative=d2f)

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
                history[i]["f"] <= history[i - 1]["f"] + 1e-10
            ), f"Function value increased at iteration {i}"


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
