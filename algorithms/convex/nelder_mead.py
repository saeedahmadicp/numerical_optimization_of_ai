# algorithms/convex/nelder_mead.py

"""Nelder-Mead simplex method for derivative-free optimization."""

from typing import List, Tuple
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig


class NelderMeadMethod(BaseNumericalMethod):
    """Implementation of Nelder-Mead method for derivative-free optimization."""

    def __init__(
        self, config: NumericalMethodConfig, x0: np.ndarray, delta: float = 0.1
    ):
        """
        Initialize Nelder-Mead method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial guess (numpy array)
            delta: Initial simplex size

        Raises:
            ValueError: If method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError("Nelder-Mead method can only be used for optimization")

        config.use_derivative_free = True  # Ensure derivative-free mode
        super().__init__(config)

        self.x = np.array(x0, dtype=float)
        # Create initial simplex for n-dimensional case
        n = len(x0)
        self.simplex = np.zeros((n + 1, n))
        self.simplex[0] = x0
        for i in range(n):
            vertex = x0.copy()
            vertex[i] += delta
            self.simplex[i + 1] = vertex
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

    def step(self) -> np.ndarray:
        """Perform one iteration of Nelder-Mead method."""
        if self._converged:
            return self.x

        x_old = self.x.copy()
        f_old = self.func(x_old)

        # Sort simplex
        self._update_simplex()

        # Get best and worst points
        x0 = self.simplex[0]  # Best point
        xn = self.simplex[-1]  # Worst point
        f0 = self.f_values[0]  # Best value
        fn = self.f_values[-1]  # Worst value

        # Calculate centroid of all points except worst
        xc = np.mean(self.simplex[:-1], axis=0)

        # Reflection
        xr = xc + self.alpha * (xc - xn)
        fr = self.func(xr)

        details = {
            "simplex_points": self.simplex.tolist(),
            "f_values": self.f_values.tolist(),
            "centroid": xc.tolist(),
            "reflection": xr.tolist(),
            "f(reflection)": fr,
        }

        if fr < f0:
            # Expansion
            xe = xc + self.gamma * (xr - xc)
            fe = self.func(xe)
            details.update({"expansion": xe.tolist(), "f(expansion)": fe})

            if fe < fr:
                self.simplex[-1] = xe
                self.f_values[-1] = fe
                details["action"] = "expansion"
            else:
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
        else:
            if fr < self.f_values[-2]:  # Better than second worst
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
            else:
                # Contraction
                if fr < fn:  # Outside contraction
                    xk = xc + self.rho * (xr - xc)
                    fk = self.func(xk)
                else:  # Inside contraction
                    xk = xc - self.rho * (xr - xc)
                    fk = self.func(xk)

                details.update({"contraction": xk.tolist(), "f(contraction)": fk})

                if fk < min(fr, fn):
                    self.simplex[-1] = xk
                    self.f_values[-1] = fk
                    details["action"] = "contraction"
                else:
                    # Shrink: update all points except best
                    details["action"] = "shrink"
                    for i in range(1, len(self.simplex)):
                        self.simplex[i] = x0 + self.sigma * (self.simplex[i] - x0)
                        self.f_values[i] = self.func(self.simplex[i])

        # Update current best point
        self.x = self.simplex[0].copy()

        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Update convergence checks with better criteria
        f_spread = np.max(self.f_values) - np.min(self.f_values)
        x_spread = np.max(
            [np.linalg.norm(x - self.simplex[0]) for x in self.simplex[1:]]
        )

        # Scale-invariant convergence criteria
        rel_f_spread = f_spread / (abs(self.f_values[0]) + 1e-10)
        rel_x_spread = x_spread / (np.linalg.norm(self.simplex[0]) + 1e-10)

        if self.iterations >= self.min_iterations and (
            (rel_f_spread < self.tol and rel_x_spread < self.tol)
            or self.iterations >= self.max_iter
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
