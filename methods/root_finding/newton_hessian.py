# methods/root_finding/newton_hessian.py

"""Newton-Hessian method for optimization."""

from typing import Callable, List, Tuple
import numpy as np
import torch  # type: ignore
from numpy.typing import NDArray

__all__ = ["newton_hessian"]


def newton_hessian(
    f: Callable[..., torch.Tensor],
    x0: NDArray[np.float64],
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[NDArray[np.float64], List[float], int, NDArray[np.float64]]:
    """Find minimum using Newton-Hessian method.

    Args:
        f: Function to minimize (must be twice differentiable)
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        verbose: Whether to print progress

    Returns:
        Tuple of (x_min, errors, iterations, history) where:
            x_min: Approximate minimizer
            errors: List of function values at each iteration
            iterations: Number of iterations used
            history: Array of points visited

    Example:
        >>> def f(x, y): return x**2 + 2*y**2
        >>> x0 = np.array([1.0, 1.0])
        >>> x, errs, iters, _ = newton_hessian(f, x0)
        >>> np.allclose(x, [0, 0], atol=1e-5)
        True
    """
    # Convert initial point to tensor
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float64)
    errors = []
    history = [x0.copy()]

    for i in range(max_iter):
        # Compute gradient and Hessian
        fx = f(*x)
        fx.backward(create_graph=True)
        grad = x.grad.clone()

        # Compute Hessian using autograd
        hessian = torch.autograd.functional.hessian(lambda x: f(*x), x)

        try:
            # Update point using Newton step
            with torch.no_grad():
                x_prev = x.clone()
                x -= torch.linalg.solve(hessian, grad)
            x.requires_grad_(True)
            x.grad = None

        except RuntimeError:
            if verbose:
                print("Singular Hessian encountered")
            return x_prev.detach().numpy(), errors, i + 1, np.array(history)

        # Record progress
        history.append(x.detach().numpy())
        errors.append(float(fx.detach()))

        # Check convergence
        if torch.norm(x - x_prev) < tol:
            if verbose:
                print(f"Converged in {i + 1} iterations")
            return x.detach().numpy(), errors, i + 1, np.array(history)

    if verbose:
        print(f"Failed to converge in {max_iter} iterations")
    return x.detach().numpy(), errors, max_iter, np.array(history)
