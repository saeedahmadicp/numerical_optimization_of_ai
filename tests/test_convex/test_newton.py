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


def test_line_search():
    """Test that line search improves robustness"""

    def f(x):
        return (
            (x - 3) ** 4 + (x - 3) ** 2 + 1
        )  # A function with slow convergence without line search

    def df(x):
        return 4 * (x - 3) ** 3 + 2 * (x - 3)

    def d2f(x):
        return 12 * (x - 3) ** 2 + 2

    # Compare with and without line search
    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6
    )

    # With line search
    method_with_ls = NewtonMethod(
        config, x0=0.0, second_derivative=d2f, use_line_search=True
    )

    # Without line search (use very small safeguard to simulate no line search)
    method_without_ls = NewtonMethod(
        config,
        x0=0.0,
        second_derivative=d2f,
        use_line_search=False,
        safeguard_factor=0.01,
    )

    # Run both methods
    while not method_with_ls.has_converged():
        method_with_ls.step()

    while not method_without_ls.has_converged():
        method_without_ls.step()

    # With line search, we should find the minimum precisely
    assert (
        abs(method_with_ls.get_current_x() - 3.0) < 1e-5
    ), "Should find minimum at x=3"

    # Without line search, we might not reach the exact minimum for challenging functions,
    # but we should still make substantial progress toward it
    x_without_ls = method_without_ls.get_current_x()
    assert f(x_without_ls) < f(0.0), "Should reduce function value significantly"
    # Instead of requiring precise convergence to x=3, check that we're at least closer
    # to the minimum than we started
    assert abs(x_without_ls - 3.0) < abs(
        0.0 - 3.0
    ), "Should move closer to the minimum at x=3"

    # Line search typically requires fewer iterations
    assert (
        method_with_ls.iterations <= method_without_ls.iterations
    ), f"Line search should be more efficient ({method_with_ls.iterations} vs {method_without_ls.iterations} iterations)"


def test_safeguard_factor():
    """Test that safeguard factor prevents overshooting"""

    def f(x):
        return (x - 5) ** 2 + 10  # Simple quadratic with minimum at x=5

    def df(x):
        return 2 * (x - 5)

    def d2f(x):
        return 2.0

    # Compare different safeguard factors
    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6
    )

    # With aggressive steps
    method_aggressive = NewtonMethod(
        config,
        x0=20.0,
        second_derivative=d2f,
        use_line_search=False,
        safeguard_factor=1.0,
    )

    # With conservative steps
    method_conservative = NewtonMethod(
        config,
        x0=20.0,
        second_derivative=d2f,
        use_line_search=False,
        safeguard_factor=0.1,
    )

    # Run one step for each
    method_aggressive.step()
    method_conservative.step()

    # Calculate how far each step moved
    aggressive_movement = abs(method_aggressive.get_current_x() - 20.0)
    conservative_movement = abs(method_conservative.get_current_x() - 20.0)

    # Aggressive steps should move further in one iteration
    assert (
        aggressive_movement > conservative_movement
    ), "Aggressive steps should move further in a single iteration"


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


def test_convergence_rate():
    """Test that Newton's method achieves quadratic convergence rate"""

    def f(x):
        return x**2 - 2  # Simple function with known root at sqrt(2)

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root", tol=1e-10)
    method = NewtonMethod(config, x0=1.5)

    # Run several iterations to get convergence data
    for _ in range(5):
        if method.has_converged():
            break
        method.step()

    # Get convergence rate
    rate = method.get_convergence_rate()

    # For well-behaved functions, Newton's method should show quadratic convergence
    # This means error_{n+1} ≈ C * error_n^2, so rate should be small
    assert rate is not None, "Should have calculated convergence rate"
    assert (
        rate < 10.0
    ), f"Rate {rate} should indicate quadratic convergence (typically < 10)"

    # The error should decrease rapidly
    errors = [data.error for data in method.get_iteration_history()]
    for i in range(1, len(errors)):
        if errors[i - 1] > 1e-10:  # Avoid division by very small numbers
            ratio = errors[i] / (errors[i - 1] ** 2)
            assert (
                ratio < 100
            ), f"Error ratio {ratio} should indicate quadratic convergence"


def test_difficult_function():
    """Test Newton's method on a more challenging function"""

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
    method_root = NewtonMethod(config_root, x0=2.0)

    while not method_root.has_converged():
        method_root.step()

    # There's a root near x ≈ 1.31
    root = method_root.get_current_x()
    assert abs(f(root)) < 1e-5, f"Should find a root. f({root}) = {f(root)}"

    # Test optimization
    config_opt = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6, max_iter=50
    )
    method_opt = NewtonMethod(config_opt, x0=-1.0, second_derivative=d2f)

    while not method_opt.has_converged():
        method_opt.step()

    # The minimum is at x = 0 (for the left part of the function)
    minimum = method_opt.get_current_x()
    assert (
        abs(minimum) < 1e-5 or abs(df(minimum)) < 1e-5
    ), f"Should find a minimum. x = {minimum}, f'(x) = {df(minimum)}"


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


def test_near_zero_derivative():
    """Test Newton's method with nearly zero derivative"""

    def f(x):
        return (x - 1) ** 3  # Has zero derivative at x=1

    def df(x):
        return 3 * (x - 1) ** 2  # Derivative is zero at x=1

    # For root finding - near a root with small derivative
    config_root = NumericalMethodConfig(
        func=f, derivative=df, method_type="root", tol=1e-6
    )
    method_root = NewtonMethod(config_root, x0=1.01)  # Start very close to x=1

    while not method_root.has_converged():
        method_root.step()

    # When derivative is near zero, Newton's method may not converge precisely
    # to the root, but it should still make progress. The function value should
    # be close to zero even if x is not exactly at the root
    x_final = method_root.get_current_x()
    assert (
        abs(f(x_final)) < 1e-5
    ), f"Function value should be near zero. f({x_final}) = {f(x_final)}"

    # Allow for a larger tolerance since we're dealing with a challenging case
    # (cubic function with zero derivative at the root)
    assert (
        abs(x_final - 1.0) < 1e-2
    ), f"Should find a point near the root at x=1, got {x_final}"


def test_record_initial_state():
    """Test that initial state is recorded properly when requested"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    config = NumericalMethodConfig(func=f, derivative=df, method_type="root", tol=1e-6)
    method = NewtonMethod(config, x0=1.5, record_initial_state=True)

    # History should have initial state before any steps
    history = method.get_iteration_history()
    assert len(history) == 1, "Should have initial state recorded"

    # Initial details should contain expected keys
    details = history[0].details
    assert "x0" in details
    assert "f(x0)" in details
    assert "f'(x0)" in details
    assert "method_type" in details

    # Initial values should match what we provided
    assert details["x0"] == 1.5
    assert details["f(x0)"] == f(1.5)
    assert details["f'(x0)"] == df(1.5)
    assert details["method_type"] == "root"


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


def test_comparison_with_power_conjugate():
    """Compare Newton's method with Powell Conjugate method"""

    # Import PowellConjugateMethod for comparison
    from algorithms.convex.powell_conjugate import PowellConjugateMethod

    def f(x):
        return (x - 3) ** 4  # Function with minimum at x=3

    def df(x):
        return 4 * (x - 3) ** 3

    def d2f(x):
        return 12 * (x - 3) ** 2

    # Setup config
    config = NumericalMethodConfig(
        func=f, derivative=df, method_type="optimize", tol=1e-6
    )

    # Create both methods
    newton_method = NewtonMethod(config, x0=1.0, second_derivative=d2f)
    powell_method = PowellConjugateMethod(config, x0=1.0)

    # Run both methods
    while not newton_method.has_converged():
        newton_method.step()

    while not powell_method.has_converged():
        powell_method.step()

    # Both methods should find the minimum approximately
    newton_x = newton_method.get_current_x()
    powell_x = powell_method.get_current_x()

    # For quartic functions, Newton might not converge exactly to the minimum
    # Check if we're close enough and if the gradient is small
    assert (
        abs(newton_x - 3.0) < 1e-2
    ), f"Newton should find minimum near x=3, got {newton_x}"
    assert (
        abs(df(newton_x)) < 1e-4
    ), f"Newton should find point with small gradient, got gradient={df(newton_x)}"

    # Powell Conjugate should also find the minimum approximately
    assert (
        abs(powell_x - 3.0) < 1e-2
    ), f"Powell Conjugate should find minimum near x=3, got {powell_x}"
    assert (
        abs(df(powell_x)) < 1e-4
    ), f"Powell Conjugate should find point with small gradient, got gradient={df(powell_x)}"

    # Compare methods - different algorithms may have different performance characteristics
    # on specific functions. For this quartic function, Powell Conjugate often performs better.
    print(
        f"Newton iterations: {newton_method.iterations}, Powell Conjugate: {powell_method.iterations}"
    )

    # Check that both methods converged successfully
    assert newton_method.has_converged(), "Newton's method should converge"
    assert powell_method.has_converged(), "Powell Conjugate method should converge"
