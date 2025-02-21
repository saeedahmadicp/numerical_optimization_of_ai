# algorithms/convex/newton_hessian.py

"""Newton-Hessian method for both root-finding and optimization."""

from typing import List, Tuple, Optional, Callable
import torch

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class NewtonHessianMethod(BaseNumericalMethod):
    """Implementation of Newton-Hessian method using automatic differentiation."""

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: float,
        second_derivative: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize Newton-Hessian method.

        Args:
            config: Configuration including function, derivative, and tolerances
            x0: Initial guess
            second_derivative: Optional for optimization (will use auto-diff if not provided)

        Raises:
            ValueError: If derivative is missing
        """
        if config.derivative is None:
            raise ValueError("Newton-Hessian method requires derivative function")

        super().__init__(config)
        self.x = x0
        self.second_derivative = second_derivative

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _compute_hessian(self, x: float) -> float:
        """
        Compute the Hessian using automatic differentiation.

        Args:
            x: Point at which to compute the Hessian

        Returns:
            float: The Hessian value
        """
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float64)

        # For optimization, compute Hessian of function
        if self.method_type == "optimize":
            fx = self.func(float(x_tensor))
            fx_tensor = torch.tensor(fx, requires_grad=True)
            fx_tensor.backward()

            if x_tensor.grad is None:
                return 1.0

            grad = x_tensor.grad.clone()
            x_tensor.grad.zero_()

            grad.backward()
            hess = x_tensor.grad

            return float(hess) if hess is not None else 1.0

        # For root-finding, compute Hessian of derivative
        else:
            dfx = self.derivative(float(x_tensor))  # type: ignore
            dfx_tensor = torch.tensor(dfx, requires_grad=True)
            dfx_tensor.backward()

            if x_tensor.grad is None:
                return 1.0

            return float(x_tensor.grad) if x_tensor.grad is not None else 1.0

    def step(self) -> float:
        """
        Perform one iteration of Newton-Hessian method.

        Returns:
            float: Current approximation
        """
        if self._converged:
            return self.x

        x_old = self.x
        fx = self.func(self.x)
        dfx = self.derivative(self.x)  # type: ignore

        # Use provided second derivative or compute via auto-diff
        if self.second_derivative is not None:
            d2fx = self.second_derivative(self.x)
        else:
            d2fx = self._compute_hessian(self.x)

        details = {
            "f(x)": fx,
            "f'(x)": dfx,
            "f''(x)": d2fx,
        }

        if self.method_type == "root":
            # Root-finding mode
            if abs(dfx) < 1e-10:
                self._converged = True
                return self.x

            step = -fx / dfx
            self.x = self.x + step

            if abs(fx) <= self.tol or self.iterations >= self.max_iter:
                self._converged = True

        else:  # optimization mode
            # Compute scale-invariant measures
            grad_norm = abs(dfx)
            x_scale = max(1.0, abs(self.x))
            f_scale = max(1.0, abs(fx))

            # Early convergence check
            if grad_norm <= self.tol * f_scale:
                self._converged = True
                return self.x

            # Compute search direction
            if abs(d2fx) < 1e-10:
                direction = -dfx
            else:
                direction = -dfx / d2fx

            # Normalize direction
            if abs(direction) > x_scale:
                direction = direction * (x_scale / abs(direction))

            # Backtracking line search
            alpha = 1.0
            beta = 0.5
            c = 0.1
            x_new = self.x + alpha * direction
            f_new = self.func(x_new)

            while f_new > fx + c * alpha * dfx * direction:
                alpha *= beta
                if alpha < 1e-10:
                    break
                x_new = self.x + alpha * direction
                f_new = self.func(x_new)

            step = alpha * direction
            self.x = x_new

            if (
                grad_norm <= self.tol * f_scale
                or abs(step) <= self.tol * x_scale
                or self.iterations >= self.max_iter
            ):
                self._converged = True

        details["step"] = step
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        return self.x

    @property
    def name(self) -> str:
        return "Newton-Hessian Method"


def newton_hessian_search(
    f: NumericalMethodConfig,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: str = "root",
) -> Tuple[float, List[float], int]:
    """Legacy wrapper for backward compatibility."""
    if callable(f):
        h = 1e-7

        def derivative(x: float) -> float:
            return (f(x + h) - f(x)) / h

        config = NumericalMethodConfig(
            func=f,
            method_type=method_type,
            derivative=derivative,
            tol=tol,
            max_iter=max_iter,
        )
    else:
        config = f

    method = NewtonHessianMethod(config, x0)
    errors = []
    prev_x = x0

    while not method.has_converged():
        x = method.step()
        if x != prev_x:  # Only record error if x changed
            errors.append(method.get_error())
        prev_x = x

    return method.x, errors, method.iterations
