# tests/test_convex/test_bisection.py

import pytest
import math
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.bisection import BisectionMethod, bisection_search
from algorithms.convex.protocols import NumericalMethodConfig


def test_basic_root_finding():
    """Test finding sqrt(2) using x^2 - 2 = 0"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 1, 2)

    # Run until convergence
    while not method.has_converged():
        x = method.step()

    assert abs(x - math.sqrt(2)) < 1e-6
    assert method.iterations < 100


def test_basic_optimization():
    """Test finding minimum of x^2 using bisection method."""

    def f(x):
        return x**2  # Minimum at x=0

    def df(x):
        return 2 * x  # Derivative is zero at x=0

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = BisectionMethod(
        config, -1, 1
    )  # Derivative changes sign from negative to positive

    # Run until convergence
    while not method.has_converged():
        x = method.step()

    assert abs(x) < 1e-6  # Should find minimum at x=0
    assert method.iterations < 100


def test_invalid_method_type():
    """Test that initialization fails when method_type is not valid or derivative is missing."""

    def f(x):
        return x**2 - 2

    # Test with invalid method type
    config = NumericalMethodConfig(func=f, method_type="invalid")
    with pytest.raises(ValueError):
        BisectionMethod(config, 1, 2)

    # Test optimization without derivative
    config = NumericalMethodConfig(func=f, method_type="optimize")
    with pytest.raises(ValueError, match="requires a derivative function"):
        BisectionMethod(config, 1, 2)


def test_optimization_with_derivative():
    """Test optimization when derivative is provided."""

    def f(x):
        return (x - 3) ** 2 + 1  # Minimum at x=3

    def df(x):
        return 2 * (x - 3)  # Derivative is zero at x=3

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    # Derivative is negative at x=2 and positive at x=4
    method = BisectionMethod(config, 2, 4)

    # Run until convergence
    while not method.has_converged():
        x = method.step()

    assert abs(x - 3) < 1e-6  # Should find minimum at x=3
    assert method.iterations < 100


def test_invalid_interval():
    """Test that initialization fails when f(a) and f(b) have same sign"""

    def f(x):
        return x**2 + 1  # Always positive

    config = NumericalMethodConfig(func=f, method_type="root")
    with pytest.raises(ValueError, match="must have opposite signs"):
        BisectionMethod(config, 1, 2)

    # Test for optimization mode
    def df(x):
        return 2 * x  # Always positive for x > 0

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    with pytest.raises(ValueError, match="must have opposite signs"):
        BisectionMethod(config, 1, 2)  # Both derivatives are positive


def test_exact_root():
    """Test when one endpoint is close to the root"""

    def f(x):
        return x - 2  # Linear function with root at x=2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-6, max_iter=100)
    method = BisectionMethod(config, 1.999, 2.001)  # Even tighter interval around x=2

    # Run until convergence or max iterations
    x = method.step()
    for _ in range(20):  # Ensure enough iterations for convergence
        if method.has_converged():
            break
        x = method.step()

    # Verify we found the root
    assert method.has_converged(), "Method did not converge"
    assert abs(f(x)) < 1e-6, f"Function value {f(x)} not within tolerance"
    assert abs(x - 2) < 1e-6, f"x value {x} not close enough to root"


def test_exact_minimum():
    """Test when one endpoint is close to the minimum"""

    def f(x):
        return (x - 2) ** 2  # Quadratic with minimum at x=2

    def df(x):
        return 2 * (x - 2)  # Derivative is zero at x=2

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-6, derivative=df
    )
    method = BisectionMethod(config, 1.999, 2.001)  # Tight interval around x=2

    # Run until convergence
    x = method.step()
    for _ in range(20):  # Ensure enough iterations for convergence
        if method.has_converged():
            break
        x = method.step()

    # Verify we found the minimum
    assert method.has_converged(), "Method did not converge"
    assert abs(df(x)) < 1e-6, f"Derivative value {df(x)} not within tolerance"
    assert abs(x - 2) < 1e-6, f"x value {x} not close enough to minimum"


def test_convergence_criteria():
    """Test that method converges within tolerance"""

    def f(x):
        return x**3 - x - 2

    config = NumericalMethodConfig(func=f, method_type="root", tol=1e-8)
    method = BisectionMethod(config, 1, 2)

    while not method.has_converged():
        x = method.step()

    assert abs(f(x)) <= 1e-8


def test_optimization_convergence_criteria():
    """Test that optimization converges within tolerance"""

    def f(x):
        return x**4 - 2 * x**2  # Minima at x=-1 and x=1

    def df(x):
        return 4 * x**3 - 4 * x  # Derivative is zero at x=0, -1, 1

    config = NumericalMethodConfig(
        func=f, method_type="optimize", tol=1e-8, derivative=df
    )
    method = BisectionMethod(config, 0.5, 1.5)  # Bracket the minimum at x=1

    while not method.has_converged():
        x = method.step()

    assert abs(df(x)) <= 1e-8  # Derivative should be close to zero
    assert abs(x - 1) <= 1e-6  # Should converge to x=1


def test_iteration_history():
    """Test that iteration history is properly recorded"""

    def f(x):
        return x**2 - 4

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 0, 3)

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))


def test_optimization_iteration_history():
    """Test that iteration history is properly recorded for optimization"""

    def f(x):
        return x**2  # Minimum at x=0

    def df(x):
        return 2 * x  # Derivative is zero at x=0

    config = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method = BisectionMethod(config, -1, 1)  # Bracket the minimum

    # Perform a few steps
    for _ in range(3):
        method.step()

    history = method.get_iteration_history()
    assert len(history) == 3

    # Check that error decreases
    errors = [data.error for data in history]
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    # Check that details contain method_type and derivative values
    for data in history:
        assert "method_type" in data.details
        assert data.details["method_type"] == "optimize"
        assert "f'(a)" in data.details
        assert "f'(b)" in data.details
        if "f'(c)" in data.details:
            assert (
                "f(c)" in data.details
            )  # Should include function value for optimization


def test_legacy_wrapper():
    """Test the backward-compatible bisection_search function"""

    def f(x):
        return x**2 - 2

    root, errors, iters = bisection_search(f, 1, 2, tol=1e-6)

    assert abs(root - math.sqrt(2)) < 1e-6
    assert len(errors) == iters


def test_legacy_wrapper_optimization():
    """Test the backward-compatible bisection_search function for optimization"""

    def f(x):
        return x**2  # Minimum at x=0

    def df(x):
        return 2 * x  # Derivative is zero at x=0

    minimum, errors, iters = bisection_search(
        f, -1, 1, tol=1e-6, method_type="optimize", derivative=df
    )

    assert abs(minimum) < 1e-6  # Should find minimum at x=0
    assert len(errors) == iters


def test_max_iterations():
    """Test that method respects maximum iterations"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root", max_iter=5)
    method = BisectionMethod(config, 1, 2)

    while not method.has_converged():
        method.step()

    assert method.iterations <= 5


def test_different_functions():
    """Test method works with different types of functions"""
    test_cases = [
        (lambda x: x**3 - x - 2, 1, 2),  # Cubic
        (
            lambda x: math.exp(x) - 4,
            1.3,
            1.4,
        ),  # Exponential: tighter interval around ln(4)
        (lambda x: math.sin(x), 3.1, 3.2),  # Trigonometric: tighter interval around pi
    ]

    for func, a, b in test_cases:
        config = NumericalMethodConfig(func=func, method_type="root", tol=1e-4)
        method = BisectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(func(x)) < 1e-4


def test_different_optimization_functions():
    """Test method works with different types of functions for optimization"""
    test_cases = [
        # f(x), df(x), a, b, expected_minimum
        (lambda x: x**2, lambda x: 2 * x, -1, 1, 0),  # Quadratic
        (lambda x: (x - 2) ** 2, lambda x: 2 * (x - 2), 1, 3, 2),  # Shifted quadratic
        (
            lambda x: math.sin(x),
            lambda x: math.cos(x),
            1.5,
            4.7,
            math.pi / 2,
        ),  # Sin function, first min at π/2
    ]

    for func, deriv, a, b, expected_min in test_cases:
        config = NumericalMethodConfig(
            func=func, method_type="optimize", derivative=deriv, tol=1e-4
        )
        method = BisectionMethod(config, a, b)

        while not method.has_converged():
            x = method.step()

        assert abs(x - expected_min) < 1e-4, f"Expected {expected_min}, got {x}"
        assert abs(deriv(x)) < 1e-4, f"Derivative not close to zero: {deriv(x)}"


def test_get_current_x():
    """Test that get_current_x returns the latest approximation"""

    def f(x):
        return x**2 - 2

    config = NumericalMethodConfig(func=f, method_type="root")
    method = BisectionMethod(config, 1, 2)

    x = method.step()
    assert x == method.get_current_x()


def test_name_property():
    """Test that the name property returns the correct name based on method_type"""

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    # Root finding
    config_root = NumericalMethodConfig(func=f, method_type="root")
    method_root = BisectionMethod(config_root, 1, 2)
    assert method_root.name == "Bisection Method (Root-Finding)"

    # Optimization
    config_opt = NumericalMethodConfig(func=f, method_type="optimize", derivative=df)
    method_opt = BisectionMethod(config_opt, -1, 1)
    assert method_opt.name == "Bisection Method (Optimization)"


def test_error_calculation():
    """Test that error is calculated correctly for both methods"""

    # Root finding: error = |f(x)|
    def f1(x):
        return x**2 - 2

    config_root = NumericalMethodConfig(func=f1, method_type="root")
    method_root = BisectionMethod(config_root, 1, 2)
    method_root.step()
    x = method_root.get_current_x()
    assert abs(method_root.get_error() - abs(f1(x))) < 1e-10

    # Optimization: error = |f'(x)|
    def f2(x):
        return x**2

    def df2(x):
        return 2 * x

    config_opt = NumericalMethodConfig(func=f2, method_type="optimize", derivative=df2)
    method_opt = BisectionMethod(config_opt, -1, 1)
    method_opt.step()
    x = method_opt.get_current_x()
    assert abs(method_opt.get_error() - abs(df2(x))) < 1e-10
