# algorithms/convex/steepest_descent.py

"""Steepest descent method for function minimization."""

from typing import List, Tuple

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class SteepestDescentMethod(BaseNumericalMethod):
    """Implementation of steepest descent method for optimization."""

    def __init__(self, config: NumericalMethodConfig, x0: float, alpha: float = 0.1):
        """
        Initialize steepest descent method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess
            alpha: Learning rate (step size)

        Raises:
            ValueError: If derivative function is not provided or method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError(
                "Steepest descent method can only be used for optimization"
            )

        if config.derivative is None:
            raise ValueError("Steepest descent method requires derivative function")

        super().__init__(config)
        self.x = x0
        self.alpha = alpha

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _backtracking_line_search(self, p: float) -> float:
        """
        Perform backtracking line search to find a suitable step size.

        Args:
            p: Search direction (negative gradient)

        Returns:
            float: Step size that satisfies the Armijo condition
        """
        c = 1e-4  # Armijo condition parameter
        rho = 0.5  # Step size reduction factor
        alpha = self.alpha

        fx = self.func(self.x)
        grad_fx = self.derivative(self.x)  # type: ignore

        while self.func(self.x + alpha * p) > fx + c * alpha * grad_fx * p:
            alpha *= rho
            if alpha < 1e-10:
                break

        return alpha

    def step(self) -> float:
        """
        Perform one iteration of steepest descent.

        Returns:
            float: Current approximation of the minimum
        """
        if self._converged:
            return self.x

        x_old = self.x
        grad = self.derivative(self.x)  # type: ignore
        p = -grad  # Search direction is negative gradient

        alpha = self._backtracking_line_search(p)

        details = {
            "gradient": grad,
            "search_direction": p,
            "step_size": alpha,
            "line_search": {
                "initial_alpha": self.alpha,
                "final_alpha": alpha,
            },
        }

        self.x += alpha * p

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        if self.get_error() <= self.tol or self.iterations >= self.max_iter:
            self._converged = True

        return self.x

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
