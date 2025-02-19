# algorithms/convex/nelder_mead.py

from typing import List, Tuple
import numpy as np

from .protocols import BaseRootFinder, RootFinderConfig


class NelderMeadMethod(BaseRootFinder):
    """Implementation of Nelder-Mead method."""

    def __init__(self, config: RootFinderConfig, x0: float, delta: float = 0.1):
        """
        Initialize Nelder-Mead method.

        Args:
            config: Configuration including function and tolerances.
            x0: Initial guess.
            delta: Initial simplex size (used to generate the initial simplex).
        """
        # Initialize common attributes from the base class.
        super().__init__(config)
        self.x = x0  # Set the current approximation.

        # For the 1D case, create an initial simplex consisting of two points:
        # the initial guess and a second point offset by delta.
        self.simplex = np.array([x0, x0 + delta])
        # Evaluate the function at each simplex point (using absolute values for root finding).
        self.f_values = np.array([abs(self.func(x)) for x in self.simplex])

        # Nelder-Mead parameters:
        self.alpha = 1.0  # Reflection coefficient.
        self.gamma = 2.0  # Expansion coefficient.
        self.rho = 0.5  # Contraction coefficient.
        self.sigma = 0.5  # Shrink coefficient.

    def get_current_x(self) -> float:
        """Get current x value."""
        return self.x

    def _update_simplex(self) -> None:
        """
        Update simplex by sorting points based on their function values.

        This ensures that the best (lowest f-value) point is always first.
        """
        order = np.argsort(self.f_values)
        self.simplex = self.simplex[order]
        self.f_values = self.f_values[order]

    def _try_point(self, x: float) -> float:
        """
        Evaluate the function at a given point and return its absolute value.

        Args:
            x: Point to evaluate.

        Returns:
            Absolute value of the function at x.
        """
        return abs(self.func(x))

    def step(self) -> float:
        """
        Perform one iteration of the Nelder-Mead method.

        Returns:
            float: Current approximation of the root.
        """
        # If the method has already converged, return the current approximation.
        if self._converged:
            return self.x

        # Store old x value
        x_old = self.x

        # Update (sort) the simplex based on current function values.
        self._update_simplex()

        # For the 1D case, the best point is the first element.
        # Let x0 be the best point, and xn the worst point.
        x0 = self.simplex[0]
        xn = self.simplex[-1]

        # Reflection: reflect the worst point across the best point.
        xr = x0 + self.alpha * (x0 - xn)
        fr = self._try_point(xr)

        # Store details for this iteration
        details = {
            "simplex_points": self.simplex.tolist(),
            "f_values": self.f_values.tolist(),
            "reflection": xr,
            "f(reflection)": fr,
            "best_point": x0,
            "worst_point": xn,
        }

        if fr < self.f_values[0]:
            # Expansion: if reflection is even better than the best,
            # try expanding further.
            xe = x0 + self.gamma * (xr - x0)
            fe = self._try_point(xe)
            details.update({"expansion": xe, "f(expansion)": fe})

            if fe < fr:
                # Accept the expansion point if it's better.
                self.simplex[-1] = xe
                self.f_values[-1] = fe
                details["action"] = "expansion"
            else:
                # Otherwise, accept the reflection.
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
        else:
            if fr < self.f_values[-1]:
                # If reflection is better than the worst, accept it.
                self.simplex[-1] = xr
                self.f_values[-1] = fr
                details["action"] = "reflection"
            else:
                # Contraction: if reflection is not better,
                # contract the simplex towards the best point.
                xc = x0 + self.rho * (xn - x0)
                fc = self._try_point(xc)
                details.update({"contraction": xc, "f(contraction)": fc})

                if fc < self.f_values[-1]:
                    self.simplex[-1] = xc
                    self.f_values[-1] = fc
                    details["action"] = "contraction"
                else:
                    # Shrink: if contraction fails, shrink the entire simplex.
                    self.simplex[-1] = x0 + self.sigma * (xn - x0)
                    self.f_values[-1] = self._try_point(self.simplex[-1])
                    details["action"] = "shrink"

        # Update current approximation to the best point of the simplex.
        self.x = self.simplex[0]

        # Store iteration data
        self.add_iteration(x_old, self.x, details)

        # Increment iteration count.
        self.iterations += 1

        # Check convergence:
        # Convergence if best function value is within tolerance,
        # or if the standard deviation of function values is below tolerance,
        # or if maximum iterations are reached.
        if (
            self.f_values[0] <= self.tol
            or np.std(self.f_values) < self.tol
            or self.iterations >= self.max_iter
        ):
            self._converged = True

        return self.x

    @property
    def name(self) -> str:
        return "Nelder-Mead Method"


def nelder_mead_search(
    f: RootFinderConfig,
    x0: float,
    delta: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    Args:
        f: Function configuration (or function) for root finding.
        x0: Initial guess.
        delta: Initial simplex size.
        tol: Error tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (root, errors, iterations) where:
         - root is the final approximation,
         - errors is a list of error values per iteration,
         - iterations is the number of iterations performed.
    """
    # Create a configuration instance with provided parameters.
    config = RootFinderConfig(func=f, tol=tol, max_iter=max_iter)
    # Instantiate the Nelder-Mead method with the initial guess and simplex size.
    method = NelderMeadMethod(config, x0, delta)

    errors = []  # List to record error values.
    # Iterate until convergence is reached.
    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    # Return final approximation, error history, and number of iterations.
    return method.x, errors, method.iterations


# if __name__ == "__main__":
#     # Define the function f(x) = x^2 - 2 (finding sqrt(2))
#     def f(x):
#         return x**2 - 2
#
#     # Using the new protocol-based implementation:
#     config = RootFinderConfig(func=f, tol=1e-6)
#     method = NelderMeadMethod(config, x0=1.5, delta=0.1)
#
#     # Iterate until convergence, printing the progress.
#     while not method.has_converged():
#         x = method.step()
#         print(f"x = {x:.6f}, error = {method.get_error():.6f}")
#
#     print(f"\nFound root: {x}")
#     print(f"Iterations: {method.iterations}")
#     print(f"Final error: {method.get_error():.6e}")
#
#     # Or using the legacy wrapper:
#     root, errors, iters = nelder_mead_search(f, 1.5)
#     print(f"Found root (legacy): {root}")
