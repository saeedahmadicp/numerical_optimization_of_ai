# methods/root_finding/newton.py

"""Newton-Raphson method for finding roots of differentiable functions."""

import torch  # type: ignore
from typing import Callable, List, Tuple

__all__ = ["newton"]


def newton(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """Find root of f(x) = 0 using Newton-Raphson method.

    Args:
        f: Function to find root of (must be differentiable)
        x0: Initial guess
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (root, errors, iterations) where:
            root: Approximate root of f(x) = 0
            errors: List of absolute errors |f(x)|
            iterations: Number of iterations used

    Example:
        >>> f = lambda x: x**2 - 2
        >>> x, errs, iters = newton(f, 1.5)
        >>> abs(x - 2**0.5) < 1e-6
        True
    """
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float64)
    errors = []
    iterations = 0

    while iterations < max_iter:
        # Compute function value and derivative
        fx = f(x)
        fx.backward()
        derivative = x.grad

        # Check for zero derivative
        if abs(derivative) < torch.finfo(torch.float64).eps:
            raise ValueError("Derivative too close to zero")

        # Update x
        with torch.no_grad():
            x -= fx / derivative
        x.requires_grad_(True)
        x.grad = None

        # Record error and check convergence
        fx = f(x)
        error = abs(fx.item())
        errors.append(error)

        if error <= tol:
            return float(x), errors, iterations

        iterations += 1

    return float(x), errors, iterations
