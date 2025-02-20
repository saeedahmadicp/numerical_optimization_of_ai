# algorithms/convex/nelder_mead.py

"""Nelder-Mead simplex method for derivative-free optimization."""

from typing import List, Tuple
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class NelderMeadMethod(BaseNumericalMethod):
    """Implementation of Nelder-Mead method for derivative-free optimization."""

    def __init__(self, config: NumericalMethodConfig, x0: float, delta: float = 0.1):
        """
        Initialize Nelder-Mead method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial guess
            delta: Initial simplex size

        Raises:
            ValueError: If method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError("Nelder-Mead method can only be used for optimization")

        config.use_derivative_free = True  # Ensure derivative-free mode
        super().__init__(config)

        self.x = x0
        # Create initial simplex
        self.simplex = np.array([x0, x0 + delta])
        self.f_values = np.array([self.func(x) for x in self.simplex])

        # Standard Nelder-Mead parameters
        self.alpha = 1.0  # Reflection
        self.gamma = 2.0  # Expansion
        self.rho = 0.5  # Contraction
        self.sigma = 0.5  # Shrink

        self.min_iterations = 5  # Add minimum iterations requirement
        self.scale = abs(self.func(x0)) + 1.0  # Function scale estimate

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _update_simplex(self) -> None:
        """Sort simplex points based on function values."""
        order = np.argsort(self.f_values)
        self.simplex = self.simplex[order]
        self.f_values = self.f_values[order]

    def step(self) -> float:
        """
        Perform one iteration of Nelder-Mead method.

        Returns:
            float: Current approximation of the minimum
        """
        if self._converged:
            return self.x

        x_old = self.x
        f_old = self.func(x_old)

        # Sort simplex
        self._update_simplex()

        x0 = self.simplex[0]  # Best point
        xn = self.simplex[-1]  # Worst point
        f0 = self.f_values[0]  # Best value
        fn = self.f_values[-1]  # Worst value

        # Reflection
        xr = x0 + self.alpha * (x0 - xn)
        fr = self.func(xr)

        details = {
            "simplex_points": self.simplex.tolist(),
            "f_values": self.f_values.tolist(),
            "reflection": xr,
            "f(reflection)": fr,
            "best_point": x0,
            "worst_point": xn,
        }

        if fr < f0:
            # Expansion
            xe = x0 + self.gamma * (xr - x0)
            fe = self.func(xe)
            details.update({"expansion": xe, "f(expansion)": fe})

            if fe < fr:
                self.simplex[-1] = xe
                self.f_values[-1] = fe
                details["action"] = "expansion"
            else:
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
        else:
            if fr < fn:
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
            else:
                # Contraction
                xc = x0 + self.rho * (xn - x0)
                fc = self.func(xc)
                details.update({"contraction": xc, "f(contraction)": fc})

                if fc < fn:
                    self.simplex[-1] = xc
                    self.f_values[-1] = fc
                    details["action"] = "contraction"
                else:
                    # Shrink
                    self.simplex[-1] = x0 + self.sigma * (xn - x0)
                    self.f_values[-1] = self.func(self.simplex[-1])
                    details["action"] = "shrink"

        # Update current best point
        self.x = self.simplex[0]

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Update convergence checks
        spread = np.std(self.f_values)
        x_spread = np.std(self.simplex)

        # Scale-invariant convergence criteria
        rel_f_spread = spread / (abs(self.f_values[0]) + 1e-10)
        rel_x_spread = x_spread / (abs(self.simplex[0]) + 1e-10)

        if self.iterations >= self.min_iterations and (  # Ensure minimum iterations
            (
                rel_f_spread < self.tol and rel_x_spread < self.tol
            )  # Relative convergence
            or self.iterations >= self.max_iter  # Max iterations
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Nelder-Mead Method"


def nelder_mead_search(
    f: NumericalMethodConfig,
    x0: float,
    delta: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for optimization
        x0: Initial guess
        delta: Initial simplex size
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

    method = NelderMeadMethod(config, x0, delta)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.x, errors, method.iterations
