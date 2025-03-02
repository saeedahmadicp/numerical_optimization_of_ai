# tests/test_convex/test_line_search.py

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.convex.line_search import (
    backtracking_line_search,
    wolfe_line_search,
    strong_wolfe_line_search,
    goldstein_line_search,
)


# Test helper functions
def setup_quadratic_function(A=None, b=None, c=0.0):
    """
    Create a simple quadratic function f(x) = 0.5 * x^T A x + b^T x + c
    with its gradient.

    Parameters:
        A : numpy array, shape (n, n). Default is identity matrix.
        b : numpy array, shape (n,). Default is zeros.
        c : float. Default is 0.

    Returns:
        f : function that computes the quadratic function value
        grad_f : function that computes the gradient of f
    """
    if A is None:
        A = np.eye(2)  # Default to 2x2 identity matrix

    if b is None:
        b = np.zeros(A.shape[0])

    def f(x):
        return 0.5 * x.T @ A @ x + b.T @ x + c

    def grad_f(x):
        return A @ x + b

    return f, grad_f


def setup_rosenbrock_function(a=1.0, b=100.0):
    """
    Create the Rosenbrock function f(x) = (a - x[0])^2 + b(x[1] - x[0]^2)^2
    with its gradient.

    Parameters:
        a, b : parameters of the Rosenbrock function

    Returns:
        f : function that computes the Rosenbrock function value
        grad_f : function that computes the gradient of f
    """

    def f(x):
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad_f(x):
        grad = np.zeros(2)
        grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
        grad[1] = 2 * b * (x[1] - x[0] ** 2)
        return grad

    return f, grad_f


def setup_exponential_function():
    """
    Create the function f(x) = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1)
    with its gradient.

    Returns:
        f : function that computes the exponential function value
        grad_f : function that computes the gradient of f
    """

    def f(x):
        return (
            np.exp(x[0] + 3 * x[1] - 0.1)
            + np.exp(x[0] - 3 * x[1] - 0.1)
            + np.exp(-x[0] - 0.1)
        )

    def grad_f(x):
        grad = np.zeros(2)
        grad[0] = (
            np.exp(x[0] + 3 * x[1] - 0.1)
            + np.exp(x[0] - 3 * x[1] - 0.1)
            - np.exp(-x[0] - 0.1)
        )
        grad[1] = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
        return grad

    return f, grad_f


# Test cases
def test_backtracking_line_search_quadratic():
    """Test backtracking line search on a simple quadratic function."""
    # Setup a quadratic function
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point and descent direction
    x0 = np.array([1.0, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run backtracking line search
    alpha = backtracking_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Armijo condition is satisfied
    c = 1e-4
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    armijo_rhs = f_x0 + c * alpha * np.dot(grad_f(x0), p)

    assert f_new <= armijo_rhs


def test_backtracking_line_search_rosenbrock():
    """Test backtracking line search on the Rosenbrock function."""
    # Setup Rosenbrock function
    f, grad_f = setup_rosenbrock_function()

    # Starting point and descent direction
    x0 = np.array([-1.2, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run backtracking line search
    alpha = backtracking_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Armijo condition is satisfied
    c = 1e-4
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    armijo_rhs = f_x0 + c * alpha * np.dot(grad_f(x0), p)

    assert f_new <= armijo_rhs


def test_wolfe_line_search_quadratic():
    """Test Wolfe line search on a simple quadratic function."""
    # Setup a quadratic function
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point and descent direction
    x0 = np.array([1.0, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Wolfe line search
    alpha = wolfe_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Wolfe conditions are satisfied
    c1 = 1e-4
    c2 = 0.9

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)
    g_new = grad_f(x0 + alpha * p)

    directional_derivative_0 = np.dot(g_x0, p)
    directional_derivative_alpha = np.dot(g_new, p)

    # Check sufficient decrease (Armijo condition)
    armijo_rhs = f_x0 + c1 * alpha * directional_derivative_0
    assert f_new <= armijo_rhs

    # Check curvature condition
    assert directional_derivative_alpha >= c2 * directional_derivative_0


def test_wolfe_line_search_rosenbrock():
    """Test Wolfe line search on the Rosenbrock function."""
    # Setup Rosenbrock function
    f, grad_f = setup_rosenbrock_function()

    # Starting point and descent direction
    x0 = np.array([-1.2, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Wolfe line search
    alpha = wolfe_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Wolfe conditions are satisfied
    c1 = 1e-4
    c2 = 0.9

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)
    g_new = grad_f(x0 + alpha * p)

    directional_derivative_0 = np.dot(g_x0, p)
    directional_derivative_alpha = np.dot(g_new, p)

    # Check sufficient decrease (Armijo condition)
    armijo_rhs = f_x0 + c1 * alpha * directional_derivative_0
    assert f_new <= armijo_rhs

    # Sometimes the curvature condition may not be satisfied exactly
    # due to numerical precision, so we use a small tolerance
    tol = 1e-6
    assert directional_derivative_alpha + tol >= c2 * directional_derivative_0


def test_strong_wolfe_line_search_quadratic():
    """Test Strong Wolfe line search on a simple quadratic function."""
    # Setup a quadratic function
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point and descent direction
    x0 = np.array([1.0, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Strong Wolfe line search
    alpha = strong_wolfe_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Strong Wolfe conditions are satisfied
    c1 = 1e-4
    c2 = 0.1

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)
    g_new = grad_f(x0 + alpha * p)

    directional_derivative_0 = np.dot(g_x0, p)
    directional_derivative_alpha = np.dot(g_new, p)

    # Check sufficient decrease (Armijo condition)
    armijo_rhs = f_x0 + c1 * alpha * directional_derivative_0
    assert f_new <= armijo_rhs

    # Check strong curvature condition
    assert abs(directional_derivative_alpha) <= c2 * abs(directional_derivative_0)


def test_strong_wolfe_line_search_exponential():
    """Test Strong Wolfe line search on an exponential function."""
    # Setup exponential function
    f, grad_f = setup_exponential_function()

    # Starting point and descent direction
    x0 = np.array([0.0, 0.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Strong Wolfe line search
    alpha = strong_wolfe_line_search(f, grad_f, x0, p)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Strong Wolfe conditions are satisfied
    c1 = 1e-4
    c2 = 0.1

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)
    g_new = grad_f(x0 + alpha * p)

    directional_derivative_0 = np.dot(g_x0, p)
    directional_derivative_alpha = np.dot(g_new, p)

    # Check sufficient decrease (Armijo condition)
    armijo_rhs = f_x0 + c1 * alpha * directional_derivative_0
    assert f_new <= armijo_rhs

    # Check strong curvature condition with a small tolerance for numerical precision
    tol = 1e-6
    assert abs(directional_derivative_alpha) <= c2 * abs(directional_derivative_0) + tol


def test_goldstein_line_search_quadratic():
    """Test Goldstein line search on a simple quadratic function."""
    # Setup a quadratic function
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point and descent direction
    x0 = np.array([1.0, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Goldstein line search
    alpha = goldstein_line_search(f, grad_f, x0, p, c=0.1)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Goldstein conditions are satisfied
    c = 0.1

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)

    directional_derivative_0 = np.dot(g_x0, p)

    # Check lower bound (sufficient decrease)
    lower_bound = f_x0 + c * alpha * directional_derivative_0

    # Check upper bound (prevent too small steps)
    upper_bound = f_x0 + (1 - c) * alpha * directional_derivative_0

    # For quadratic functions, both conditions should be satisfied exactly
    assert f_new <= lower_bound
    assert f_new >= upper_bound


def test_goldstein_line_search_rosenbrock():
    """Test Goldstein line search on the Rosenbrock function."""
    # Setup Rosenbrock function
    f, grad_f = setup_rosenbrock_function()

    # Starting point and descent direction
    x0 = np.array([-1.2, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run Goldstein line search
    alpha = goldstein_line_search(f, grad_f, x0, p, c=0.1)

    # Check if the step size is positive
    assert alpha > 0

    # Check if the Goldstein conditions are satisfied
    c = 0.1

    # Calculate function and gradient values
    f_x0 = f(x0)
    f_new = f(x0 + alpha * p)
    g_x0 = grad_f(x0)

    directional_derivative_0 = np.dot(g_x0, p)

    # Check lower bound (sufficient decrease)
    lower_bound = f_x0 + c * alpha * directional_derivative_0

    # Check upper bound (prevent too small steps)
    upper_bound = f_x0 + (1 - c) * alpha * directional_derivative_0

    # Due to numerical precision and the complex nature of Rosenbrock function,
    # we allow a small tolerance
    tol = 1e-8
    assert f_new <= lower_bound + tol
    assert f_new >= upper_bound - tol


def test_comparison_of_methods():
    """Compare all line search methods on the same problem."""
    # Setup a quadratic function
    A = np.array([[4.0, 0.0], [0.0, 2.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point and descent direction
    x0 = np.array([1.0, 1.0])
    p = -grad_f(x0)  # Steepest descent direction

    # Run all line search methods
    alpha_backtracking = backtracking_line_search(f, grad_f, x0, p)
    alpha_wolfe = wolfe_line_search(f, grad_f, x0, p)
    alpha_strong_wolfe = strong_wolfe_line_search(f, grad_f, x0, p)
    alpha_goldstein = goldstein_line_search(f, grad_f, x0, p)

    # All step sizes should be positive
    assert alpha_backtracking > 0
    assert alpha_wolfe > 0
    assert alpha_strong_wolfe > 0
    assert alpha_goldstein > 0

    # Calculate function values after each step
    f_backtracking = f(x0 + alpha_backtracking * p)
    f_wolfe = f(x0 + alpha_wolfe * p)
    f_strong_wolfe = f(x0 + alpha_strong_wolfe * p)
    f_goldstein = f(x0 + alpha_goldstein * p)

    # All methods should decrease the function value
    f_x0 = f(x0)
    assert f_backtracking < f_x0
    assert f_wolfe < f_x0
    assert f_strong_wolfe < f_x0
    assert f_goldstein < f_x0

    # Print step sizes and function values for comparison
    print(f"\nComparison of line search methods on quadratic function:")
    print(f"Initial function value: {f_x0}")
    print(f"Backtracking: alpha = {alpha_backtracking}, f = {f_backtracking}")
    print(f"Wolfe: alpha = {alpha_wolfe}, f = {f_wolfe}")
    print(f"Strong Wolfe: alpha = {alpha_strong_wolfe}, f = {f_strong_wolfe}")
    print(f"Goldstein: alpha = {alpha_goldstein}, f = {f_goldstein}")


def test_non_descent_direction_handling():
    """Test how line search methods handle non-descent directions."""
    # Setup a quadratic function
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    f, grad_f = setup_quadratic_function(A)

    # Starting point
    x0 = np.array([1.0, 1.0])

    # Create a non-descent direction (opposite of steepest descent)
    p = grad_f(x0)  # This points in the direction of increase

    # Each method should return a very small step size or handle the issue
    alpha_backtracking = backtracking_line_search(f, grad_f, x0, p, alpha_min=1e-8)
    alpha_wolfe = wolfe_line_search(f, grad_f, x0, p, alpha_min=1e-8)
    alpha_strong_wolfe = strong_wolfe_line_search(f, grad_f, x0, p, alpha_min=1e-8)
    alpha_goldstein = goldstein_line_search(f, grad_f, x0, p, alpha_min=1e-8)

    # All methods should return the minimum alpha or a very small step size
    assert alpha_backtracking <= 1e-6
    assert alpha_wolfe <= 1e-6
    assert alpha_strong_wolfe <= 1e-6
    assert alpha_goldstein <= 1e-6

    print(f"\nHandling non-descent directions:")
    print(f"Backtracking: alpha = {alpha_backtracking}")
    print(f"Wolfe: alpha = {alpha_wolfe}")
    print(f"Strong Wolfe: alpha = {alpha_strong_wolfe}")
    print(f"Goldstein: alpha = {alpha_goldstein}")


if __name__ == "__main__":
    # Run the tests
    test_backtracking_line_search_quadratic()
    test_backtracking_line_search_rosenbrock()
    test_wolfe_line_search_quadratic()
    test_wolfe_line_search_rosenbrock()
    test_strong_wolfe_line_search_quadratic()
    test_strong_wolfe_line_search_exponential()
    test_goldstein_line_search_quadratic()
    test_goldstein_line_search_rosenbrock()
    test_comparison_of_methods()
    test_non_descent_direction_handling()

    print("All tests passed!")
