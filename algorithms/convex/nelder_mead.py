# algorithms/convex/nelder_mead.py

"""
Nelder-Mead simplex method for derivative-free optimization.

The Nelder-Mead method is a direct search optimization technique that doesn't
require derivatives, making it suitable for non-smooth or noisy functions. It
operates by maintaining a simplex (a generalization of a triangle in n dimensions)
and iteratively improving it through reflection, expansion, contraction, and
shrinkage operations.

Mathematical Basis:
----------------
1. Maintain a simplex with n+1 vertices for n-dimensional optimization
2. In each iteration:
   a. Identify the worst point and reflect it through the centroid of other points
   b. If the reflected point is promising, try expansion
   c. If the reflected point is not good, try contraction
   d. If contraction fails, shrink the simplex toward the best point
3. Continue until convergence criteria are met

Convergence Properties:
--------------------
- Robust for problems with noise or discontinuities where derivatives are unavailable
- Generally linear convergence, but can stall on degenerate problems
- Provable convergence for strictly convex functions in 1D and 2D
- No general convergence proof for higher dimensions
- Heuristically effective in practice for many problems
"""

from typing import List, Tuple, Optional, Callable, Union
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


class NelderMeadMethod(BaseNumericalMethod):
    """
    Implementation of Nelder-Mead method for derivative-free optimization.

    The Nelder-Mead simplex method performs a direct search without requiring derivatives,
    making it suitable for non-smooth functions or when gradient information is unavailable.
    It maintains a simplex (a geometric figure with n+1 vertices in n dimensions) and
    systematically transforms it to approach an optimum.

    Mathematical guarantees:
    - For 1D and 2D strictly convex functions, convergence to a minimum is guaranteed
    - For higher dimensions, no general convergence guarantees, but empirically effective

    Implementation features:
    - Adaptive to function topology through reflection, expansion, contraction, and shrink operations
    - Robust against non-smooth functions and noise
    - Automatically scales simplex to function behavior
    - Handles any number of dimensions (limited by computational resources)
    - Uses scale-invariant convergence criteria for better stopping behavior
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: Union[float, np.ndarray],
        delta: float = 0.1,
    ):
        """
        Initialize Nelder-Mead method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial guess (scalar float or numpy array)
            delta: Initial simplex size

        Raises:
            ValueError: If method_type is not 'optimize'
        """
        if config.method_type != "optimize":
            raise ValueError("Nelder-Mead method can only be used for optimization")

        config.use_derivative_free = True  # Ensure derivative-free mode
        super().__init__(config)

        # Handle scalar case by converting to numpy array
        if isinstance(x0, (int, float)):
            self.x = np.array([float(x0)])
            self.is_scalar_problem = True
        else:
            self.x = np.array(x0, dtype=float)
            self.is_scalar_problem = False

        # Get dimensionality
        n = len(self.x)

        # Create initial simplex for n-dimensional case
        self.simplex = np.zeros((n + 1, n))
        self.simplex[0] = self.x
        for i in range(n):
            vertex = self.x.copy()
            vertex[i] += delta
            self.simplex[i + 1] = vertex
        self.f_values = np.array(
            [self.func(self._maybe_scalar(x)) for x in self.simplex]
        )

        # Standard Nelder-Mead parameters
        self.alpha = 1.0  # Reflection
        self.gamma = 2.0  # Expansion
        self.rho = 0.5  # Contraction
        self.sigma = 0.5  # Shrink

        self.min_iterations = 5  # Add minimum iterations requirement
        self.scale = (
            abs(self.func(self._maybe_scalar(self.x))) + 1.0
        )  # Function scale estimate

        # Store initial sizes for convergence rate estimation
        self.initial_size = self._calculate_simplex_size()
        self.prev_sizes = [self.initial_size]

    def _maybe_scalar(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Convert 1D array to scalar if this is a scalar problem.

        Args:
            x: Array input

        Returns:
            float or np.ndarray: Scalar value if is_scalar_problem, otherwise unchanged array
        """
        if self.is_scalar_problem and len(x) == 1:
            return float(x[0])
        return x

    def get_current_x(self) -> Union[float, np.ndarray]:
        """
        Get current best point in the simplex.

        Returns:
            float or np.ndarray: Current best approximation of the minimum
        """
        return self._maybe_scalar(self.x)

    def compute_descent_direction(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the descent direction at point x.

        For Nelder-Mead, this is not explicitly used as the method is derivative-free
        and operates on simplex transformations rather than computing directions.
        Included for protocol consistency.

        Args:
            x: Current point

        Returns:
            np.ndarray: Zero vector (placeholder implementation)
        """
        # Nelder-Mead doesn't use a descent direction in the same way as gradient-based methods
        # It implicitly determines direction through simplex operations
        if self.is_scalar_problem and np.isscalar(x):
            x = np.array([x])
        return np.zeros_like(x)  # Placeholder return for protocol consistency

    def compute_step_length(
        self, x: Union[float, np.ndarray], direction: np.ndarray
    ) -> float:
        """
        Compute step length for the direction.

        For Nelder-Mead, this is not explicitly used as the method operates through
        simplex transformations rather than step computations.
        Included for protocol consistency.

        Args:
            x: Current point
            direction: Descent direction (unused)

        Returns:
            float: Zero (placeholder implementation)
        """
        # Nelder-Mead doesn't use step length in traditional sense
        return 0.0  # Placeholder return for protocol consistency

    def _update_simplex(self) -> None:
        """Sort simplex points based on function values."""
        order = np.argsort(self.f_values)
        self.simplex = self.simplex[order]
        self.f_values = self.f_values[order]

    def _calculate_simplex_size(self) -> float:
        """
        Calculate the size of the simplex.

        Returns:
            float: Maximum distance between any vertex and the best vertex
        """
        if len(self.simplex) <= 1:
            return 0.0

        best_point = self.simplex[0]
        return np.max([np.linalg.norm(x - best_point) for x in self.simplex[1:]])

    def step(self) -> Union[float, np.ndarray]:
        """
        Perform one iteration of Nelder-Mead method.

        Each iteration applies one or more of the simplex operations:
        1. Reflection: reflect worst point through centroid of remaining points
        2. Expansion: if reflection yields improvement, try expanding further
        3. Contraction: if reflection doesn't help, try contracting instead
        4. Shrink: if all else fails, shrink the entire simplex toward best point

        Returns:
            float or np.ndarray: Current best approximation of the minimum
        """
        if self._converged:
            return self.get_current_x()

        x_old = self.x.copy()
        f_old = self.func(self._maybe_scalar(x_old))

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
        fr = self.func(self._maybe_scalar(xr))

        details = {
            "simplex_points": self.simplex.tolist(),
            "f_values": self.f_values.tolist(),
            "centroid": xc.tolist(),
            "reflection": xr.tolist(),
            "f(reflection)": fr,
            "simplex_size": self._calculate_simplex_size(),
        }

        if fr < f0:
            # Expansion
            xe = xc + self.gamma * (xr - xc)
            fe = self.func(self._maybe_scalar(xe))
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
                    fk = self.func(self._maybe_scalar(xk))
                else:  # Inside contraction
                    xk = xc - self.rho * (xr - xc)
                    fk = self.func(self._maybe_scalar(xk))

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
                        self.f_values[i] = self.func(
                            self._maybe_scalar(self.simplex[i])
                        )

        # Update current best point
        self.x = self.simplex[0].copy()

        # Track simplex size for convergence rate estimation
        current_size = self._calculate_simplex_size()
        self.prev_sizes.append(current_size)
        if len(self.prev_sizes) > 3:
            self.prev_sizes.pop(0)

        # For history, convert x_old to scalar if needed for display
        if self.is_scalar_problem:
            x_old_display = float(x_old[0])
            x_new_display = float(self.x[0])
        else:
            x_old_display = x_old
            x_new_display = self.x

        self.add_iteration(x_old_display, x_new_display, details)
        self.iterations += 1

        # Update convergence checks with better criteria
        f_spread = np.max(self.f_values) - np.min(self.f_values)
        x_spread = current_size

        # Scale-invariant convergence criteria
        rel_f_spread = f_spread / (abs(self.f_values[0]) + 1e-10)
        rel_x_spread = x_spread / (np.linalg.norm(self.simplex[0]) + 1e-10)

        if self.iterations >= self.min_iterations and (
            (rel_f_spread < self.tol and rel_x_spread < self.tol)
            or self.iterations >= self.max_iter
        ):
            self._converged = True
            # Add convergence reason to the last iteration
            last_iteration = self._history[-1]
            if self.iterations >= self.max_iter:
                last_iteration.details["convergence_reason"] = (
                    "maximum iterations reached"
                )
            else:
                last_iteration.details["convergence_reason"] = (
                    "simplex sufficiently small"
                )

        return self.get_current_x()

    def get_error(self) -> float:
        """
        Get error estimate for current approximation.

        For Nelder-Mead, we use the size of the simplex as an error estimate,
        normalized by the initial size for scale-invariance.

        Returns:
            float: Current error estimate
        """
        current_size = self._calculate_simplex_size()

        # For scale invariance, normalize by initial size
        if self.initial_size > 0:
            return current_size / self.initial_size
        return current_size

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate the observed convergence rate of the method.

        For Nelder-Mead, this estimates the rate at which the simplex size decreases.
        The theoretical rate depends on the function properties and dimensionality.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self.prev_sizes) < 3:
            return None

        # Avoid division by zero
        if self.prev_sizes[-2] == 0 or self.prev_sizes[-3] == 0:
            return None

        # Calculate rate based on simplex size reduction
        rate1 = self.prev_sizes[-1] / self.prev_sizes[-2]
        rate2 = self.prev_sizes[-2] / self.prev_sizes[-3]

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return "Nelder-Mead Method"


def nelder_mead_search(
    f: Union[NumericalMethodConfig, Callable],
    x0: Union[float, np.ndarray],
    delta: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "optimize",
) -> Tuple[Union[float, np.ndarray], List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    This function provides a simpler interface to the Nelder-Mead method for
    users who don't need the full object-oriented functionality.

    Mathematical guarantee:
    - For 1D and 2D strictly convex functions, will converge to a minimum
    - For higher dimensions, empirically effective but no theoretical guarantees

    Args:
        f: Function configuration or callable for optimization
        x0: Initial guess (scalar or numpy array)
        delta: Initial simplex size
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (minimum point, errors, iterations)
    """
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type=method_type, tol=tol, max_iter=max_iter
        )
    else:
        config = f

    method = NelderMeadMethod(config, x0, delta)
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.get_current_x(), errors, method.iterations
