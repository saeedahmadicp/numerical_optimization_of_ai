# algorithms/convex/steepest_descent.py

"""Steepest descent method for function minimization."""

from typing import List, Tuple
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class SteepestDescentMethod(BaseNumericalMethod):
    """Steepest descent method with backtracking line search."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: np.ndarray,
        alpha: float = 0.1,
        beta: float = 0.8,
        c: float = 0.0001,
    ):
        """Initialize the method.

        Args:
            config: Configuration object
            x0: Initial point
            alpha: Initial step size for line search
            beta: Step size reduction factor
            c: Sufficient decrease parameter
        """
        super().__init__(config)
        self.x = np.array(x0, dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self._converged = False
        self.iterations = 0

    def _backtracking_line_search(self, p: np.ndarray) -> float:
        """Backtracking line search to find step size.

        Args:
            p: Search direction

        Returns:
            float: Step size
        """
        alpha = self.alpha
        fx = self.func(self.x)
        grad_fx = self.derivative(self.x)

        # For vector case, use dot product for directional derivative
        directional_derivative = np.dot(grad_fx, p)

        while True:
            x_new = self.x + alpha * p
            fx_new = self.func(x_new)

            # Armijo condition
            if fx_new <= fx + self.c * alpha * directional_derivative:
                break

            alpha *= self.beta

            if alpha < 1e-10:  # Prevent too small steps
                break

        return alpha

    def step(self) -> np.ndarray:
        """Perform one iteration of steepest descent.

        Returns:
            np.ndarray: New point
        """
        # Get gradient at current point
        grad = self.derivative(self.x)

        # Search direction is negative gradient
        p = -grad

        # Normalize search direction for better scaling
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-10:
            p = p / p_norm

        # Find step size using line search
        alpha = self._backtracking_line_search(p)

        # Store details for visualization
        self.add_iteration(
            self.x,
            self.x + alpha * p,
            {
                "gradient": str(grad),
                "search_direction": str(p),
                "step_size": alpha,
                "line_search": {
                    "initial_alpha": self.alpha,
                    "final_alpha": alpha,
                },
            },
        )

        # Update position
        self.x = self.x + alpha * p
        self.iterations += 1

        # Check convergence
        grad_norm = np.linalg.norm(self.derivative(self.x))
        self._converged = grad_norm < self.tol or self.iterations >= self.max_iter

        return self.x

    def get_current_x(self) -> np.ndarray:
        """Get current point."""
        return self.x

    def get_error(self) -> float:
        """Get error estimate (gradient norm)."""
        return float(np.linalg.norm(self.derivative(self.x)))

    @property
    def name(self) -> str:
        return "Steepest Descent Method"


def steepest_descent_search(
    f: NumericalMethodConfig,
    x0: float,
    alpha: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for optimization
        x0: Initial guess
        alpha: Learning rate
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (minimum, errors, iterations)
    """
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type="optimize", tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = SteepestDescentMethod(config, x0, alpha)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
