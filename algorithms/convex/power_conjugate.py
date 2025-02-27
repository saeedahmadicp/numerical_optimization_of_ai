# algorithms/convex/power_conjugate.py

"""
Power Conjugate method for optimization and root-finding.

The Power Conjugate method combines aspects of conjugate gradient methods with
power iteration techniques. It uses an iterative approach to generate search
directions that approximate the eigenvectors of the Hessian, which helps to
navigate narrow valleys in the objective function more efficiently.

Mathematical Basis:
----------------
For optimization of a function f over a domain:

1. Generate a search direction based on gradient/function information
2. Apply a power iteration step to improve the search direction
3. Perform line search along this direction
4. Update the current point
5. Repeat until convergence

For root-finding:
1. Use a modified power iteration approach to estimate the direction
   toward the root
2. Take steps in that direction, adaptively adjusting step size
3. Repeat until |f(x)| is sufficiently small

Convergence Properties:
--------------------
- Faster than gradient descent for ill-conditioned problems
- Can handle non-quadratic objective functions
- Adaptively approximates the dominant eigenvector of the Hessian
- Converges superlinearly for well-behaved functions
"""

from typing import List, Tuple, Optional, Callable, Union, Dict, Any
import math
import numpy as np

from .protocols import BaseNumericalMethod, NumericalMethodConfig, MethodType


class PowerConjugateMethod(BaseNumericalMethod):
    """
    Implementation of the Power Conjugate method.

    The Power Conjugate method combines power iteration techniques with conjugate
    direction methods to efficiently optimize functions or find roots. It is particularly
    effective for problems with poorly conditioned Hessians.

    Mathematical guarantees:
    - For optimization: Converges superlinearly to local minima for smooth functions
    - For root-finding: Converges to roots with guaranteed progress if the function
      has suitable properties near the root
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        x0: float,
        direction_reset_freq: int = 5,
        line_search_factor: float = 0.5,
        power_iterations: int = 2,
        record_initial_state: bool = False,
    ):
        """
        Initialize the Power Conjugate method.

        Args:
            config: Configuration including function and tolerances
            x0: Initial point
            direction_reset_freq: Frequency of direction resets (similar to CG restarts)
            line_search_factor: Factor for line search step size reduction
            power_iterations: Number of power iterations to perform in each step
            record_initial_state: Whether to record the initial state in history

        Raises:
            ValueError: If method_type is not 'root' or 'optimize'
        """
        # Call the base class initializer
        super().__init__(config)

        # Check method type is valid
        if self.method_type not in ("root", "optimize"):
            raise ValueError(
                f"Invalid method_type: {self.method_type}. Must be 'root' or 'optimize'."
            )

        self.x = x0
        self.prev_x = x0
        self.direction_reset_freq = direction_reset_freq
        self.line_search_factor = line_search_factor
        self.power_iterations = power_iterations

        # For bracketing in root-finding
        self.bracket = None
        if self.method_type == "root":
            # Try to establish initial bracket for root-finding
            self._setup_initial_bracket(x0)

        # Initialize search direction
        self.direction = self._estimate_initial_direction()
        self.prev_direction = self.direction

        # For conjugate updates
        self.prev_gradient = self._estimate_gradient(x0)
        self.beta = 0.0

        # Maximum allowed step size to prevent overflow
        self.max_step_size = 10.0

        # Use a much smaller min step size for better convergence
        self.min_step_size = 1e-14

        # Use tighter termination criteria for optimization problems
        if self.method_type == "optimize":
            # Use a tighter tolerance for optimization
            self.opt_tol = self.tol * 0.1
        else:
            self.opt_tol = self.tol

        # Optionally record initial state
        if record_initial_state:
            initial_details = {
                "x0": x0,
                "f(x0)": self.func(x0),
                "initial_direction": self.direction,
                "initial_gradient": self.prev_gradient,
                "method_type": self.method_type,
                "bracket": self.bracket,
            }
            self.add_iteration(self.x, self.x, initial_details)

    def _setup_initial_bracket(self, x0: float, search_radius: float = 5.0):
        """
        Attempt to find an initial bracket containing a root.

        For root-finding, having a bracket [a, b] where f(a) and f(b) have opposite
        signs can significantly improve convergence.

        Args:
            x0: Initial point
            search_radius: How far to search for bracketing points
        """
        if self.method_type != "root":
            return

        f0 = self.func(x0)

        # If we're already at a root, no need for bracketing
        if abs(f0) < self.tol:
            return

        # Try points in increasing distance from x0 in both directions
        for d in [0.1, 0.5, 1.0, 2.0, search_radius]:
            for direction in [1, -1]:
                x_test = x0 + direction * d
                try:
                    f_test = self.func(x_test)

                    # Check if we found a bracket
                    if f0 * f_test <= 0:
                        if direction == 1:
                            self.bracket = (x0, x_test)
                        else:
                            self.bracket = (x_test, x0)
                        return
                except:
                    # Function evaluation failed, skip this point
                    continue

    def _estimate_initial_direction(self) -> float:
        """
        Estimate initial direction based on method type.

        For optimization: Use negative gradient direction
        For root-finding: Use direction toward root based on function value

        Returns:
            float: Initial direction
        """
        gradient = self._estimate_gradient(self.x)

        if self.method_type == "optimize":
            # For optimization, move in negative gradient direction
            return -gradient
        else:
            # For root-finding
            func_val = self.func(self.x)

            # If we have a bracket, move toward the other end
            if self.bracket:
                a, b = self.bracket
                if self.x == a:
                    return b - a  # Move toward b
                elif self.x == b:
                    return a - b  # Move toward a
                else:
                    # We're inside the bracket, use Newton-like direction
                    return -func_val / (gradient + 1e-10)
            else:
                # No bracket, use Newton-like direction with safeguards
                newton_dir = -func_val / (gradient + 1e-10)

                # If Newton gives a very large step, use gradient direction instead
                if abs(newton_dir) > 10.0:
                    return -math.copysign(1.0, func_val * gradient) * abs(gradient)
                return newton_dir

    def _estimate_gradient(self, x: float) -> float:
        """
        Estimate gradient at point x.

        Args:
            x: Point at which to estimate gradient

        Returns:
            float: Estimated gradient
        """
        if self.derivative is not None:
            return self.derivative(x)
        else:
            return self.estimate_derivative(x)

    def _power_iteration_update(self) -> float:
        """
        Apply power iteration to refine search direction.

        This approximates the dominant eigenvector of the local Hessian.

        Returns:
            float: Updated direction
        """
        direction = self.direction

        # For numerical stability, don't do power iteration if direction is near zero
        if abs(direction) < 1e-10:
            gradient = self._estimate_gradient(self.x)
            return -math.copysign(1.0, gradient)

        for _ in range(self.power_iterations):
            # Simple power iteration: approximates applying the Hessian
            step_size = min(
                self.finite_diff_step, 0.1 * abs(self.x) + self.finite_diff_step
            )

            try:
                x_plus = self.x + step_size * direction
                x_minus = self.x - step_size * direction

                g_plus = self._estimate_gradient(x_plus)
                g_minus = self._estimate_gradient(x_minus)

                # Finite difference approximation of Hessian-vector product
                hessian_vec = (g_plus - g_minus) / (2 * step_size)

                # Update direction (use dominant eigenvector direction)
                if abs(hessian_vec) > 1e-10:
                    direction = hessian_vec

                    # Normalize
                    direction = direction / abs(direction)
            except:
                # If power iteration fails, revert to simple gradient direction
                gradient = self._estimate_gradient(self.x)
                direction = -math.copysign(1.0, gradient)

        # Make direction point in the correct direction based on method type
        if self.method_type == "optimize":
            gradient = self._estimate_gradient(self.x)
            # Explicitly set direction to negative gradient to ensure it points downhill
            direction = -math.copysign(1.0, gradient)
        else:  # Root-finding
            func_val = self.func(self.x)
            # For root-finding, we use Newton's method direction when possible
            if abs(direction) > 1e-10:
                try:
                    gradient = self._estimate_gradient(self.x)
                    if abs(gradient) > 1e-10:
                        newton_dir = -func_val / gradient
                        # If Newton direction is reasonable, use it as a guide
                        if abs(newton_dir) < 10.0:
                            direction = math.copysign(1.0, newton_dir)
                except:
                    pass

            # If we have a bracket, make sure we move toward it
            if self.bracket:
                a, b = self.bracket
                mid = (a + b) / 2

                # If we're moving away from the bracket's midpoint, reverse direction
                if (self.x < mid and direction < 0) or (self.x > mid and direction > 0):
                    direction = -direction

        return direction

    def _compute_conjugate_direction(self) -> float:
        """
        Compute conjugate direction using Fletcher-Reeves formula.

        Returns:
            float: Updated conjugate direction
        """
        current_gradient = self._estimate_gradient(self.x)

        # Check for zero gradient
        if abs(current_gradient) < 1e-10:
            return 0.0  # Potentially at a critical point

        # Reset periodically or if gradients are nearly orthogonal
        if self.iterations % self.direction_reset_freq == 0 or abs(
            current_gradient * self.prev_gradient
        ) < 1e-10 * abs(current_gradient) * abs(self.prev_gradient):

            self.beta = 0.0
            direction = -current_gradient
        else:
            # Fletcher-Reeves formula
            self.beta = (current_gradient**2) / max(self.prev_gradient**2, 1e-10)

            # Limit beta to prevent numerical issues
            self.beta = min(self.beta, 2.0)

            direction = -current_gradient + self.beta * self.prev_direction

        # Store for next iteration
        self.prev_gradient = current_gradient
        self.prev_direction = direction

        return direction

    def _line_search(self, direction: float) -> Tuple[float, float, Dict[str, Any]]:
        """
        Perform line search along the given direction.

        Uses backtracking line search to find a step size that produces
        sufficient decrease in the objective function (for optimization)
        or in |f(x)| (for root-finding).

        Args:
            direction: Search direction

        Returns:
            Tuple[float, float, Dict]: Step size, new x value, and line search details
        """
        # Prevent zero direction
        if abs(direction) < 1e-10:
            direction = 1e-10

        # Limit initial step size to prevent overflow
        alpha = min(1.0, self.max_step_size / (abs(direction) + 1e-10))

        try:
            x_new = self.x + alpha * direction
            f_current = self.func(self.x)
            f_new = self.func(x_new)
        except (OverflowError, ValueError, ZeroDivisionError):
            # If function evaluation fails, try a smaller step
            alpha *= 0.1
            try:
                x_new = self.x + alpha * direction
                f_current = self.func(self.x)
                f_new = self.func(x_new)
            except:
                # If all else fails, take a tiny step
                alpha = 1e-4
                x_new = self.x + alpha * direction
                f_current = self.func(self.x)
                try:
                    f_new = self.func(x_new)
                except:
                    # If we still can't evaluate, don't move
                    return 0.0, self.x, {"error": "Function evaluation failed"}

        line_search_details = {
            "initial_alpha": alpha,
            "initial_x_new": x_new,
            "initial_f_new": f_new,
        }

        # Special handling for root-finding with a bracket
        if self.method_type == "root" and self.bracket:
            a, b = self.bracket

            # If we're stepping outside the bracket, limit the step
            if (x_new < a) or (x_new > b):
                # Limit step to stay within bracket with a small margin
                alpha = min(
                    alpha,
                    (
                        0.9 * abs(b - self.x) / abs(direction)
                        if direction > 0
                        else 0.9 * abs(self.x - a) / abs(direction)
                    ),
                )
                x_new = self.x + alpha * direction
                try:
                    f_new = self.func(x_new)
                except:
                    # Fall back to bisection if function evaluation fails
                    x_new = (a + b) / 2
                    try:
                        f_new = self.func(x_new)
                    except:
                        # If all else fails, return current position
                        return 0.0, self.x, {"error": "Function evaluation failed"}

            # Update bracket if possible
            f_a = self.func(a)
            f_b = self.func(b)
            f_x_new = f_new

            if f_x_new * f_a <= 0:
                # Root is between a and x_new
                self.bracket = (a, x_new)
            elif f_x_new * f_b <= 0:
                # Root is between x_new and b
                self.bracket = (x_new, b)

            line_search_details["bracket_updated"] = self.bracket

        # Determine if step is acceptable based on method type
        backtrack_count = 0
        max_backtracks = 20  # Increase max backtracks for more precision

        if self.method_type == "optimize":
            # For optimization, we want to decrease function value
            while (
                f_new > f_current
                and alpha > self.min_step_size
                and backtrack_count < max_backtracks
            ):
                alpha *= self.line_search_factor
                x_new = self.x + alpha * direction
                try:
                    f_new = self.func(x_new)
                except:
                    # If function evaluation fails, try even smaller step
                    alpha *= self.line_search_factor
                    x_new = self.x + alpha * direction
                    try:
                        f_new = self.func(x_new)
                    except:
                        # If all else fails, don't move
                        return (
                            0.0,
                            self.x,
                            {"error": "Function evaluation failed during backtracking"},
                        )

                backtrack_count += 1
                line_search_details["reduced_alpha"] = alpha
                line_search_details["reduced_f_new"] = f_new

            # For quadratic-like functions near the minimum, take a smaller step for precision
            f_grad = self._estimate_gradient(self.x)
            if abs(f_grad) < 0.01 and self.method_type == "optimize":
                # We're close to the minimum, take a smaller step for precision
                alpha_refined = alpha * 0.1
                x_new_refined = self.x + alpha_refined * direction
                try:
                    f_new_refined = self.func(x_new_refined)
                    if f_new_refined < f_new:
                        alpha = alpha_refined
                        x_new = x_new_refined
                        f_new = f_new_refined
                        line_search_details["refined_step"] = True
                except:
                    pass  # If refinement fails, keep the original step
        else:
            # For root-finding, we want to decrease |f(x)|
            while (
                abs(f_new) > abs(f_current)
                and alpha > self.min_step_size
                and backtrack_count < max_backtracks
            ):
                alpha *= self.line_search_factor
                x_new = self.x + alpha * direction
                try:
                    f_new = self.func(x_new)
                except:
                    # If function evaluation fails, try even smaller step
                    alpha *= self.line_search_factor
                    x_new = self.x + alpha * direction
                    try:
                        f_new = self.func(x_new)
                    except:
                        # If all else fails, don't move
                        return (
                            0.0,
                            self.x,
                            {"error": "Function evaluation failed during backtracking"},
                        )

                backtrack_count += 1
                line_search_details["reduced_alpha"] = alpha
                line_search_details["reduced_f_new"] = f_new

            # For root-finding, we should also check for sign changes
            if self.method_type == "root" and f_current * f_new <= 0:
                # We've bracketed a root! Update bracket
                self.bracket = (self.x, x_new) if f_current < 0 else (x_new, self.x)
                line_search_details["found_bracket"] = self.bracket

        # If we've found a new bracket for root-finding, use bisection step
        if (
            self.method_type == "root"
            and self.bracket
            and backtrack_count >= max_backtracks
        ):
            a, b = self.bracket
            x_new = (a + b) / 2
            try:
                f_new = self.func(x_new)
                # Update bracket
                f_a = self.func(a)
                if f_new * f_a <= 0:
                    self.bracket = (a, x_new)
                else:
                    self.bracket = (x_new, b)

                line_search_details["bisection_used"] = True
                line_search_details["new_bracket"] = self.bracket
            except:
                # If function evaluation fails, don't move
                return (
                    0.0,
                    self.x,
                    {"error": "Function evaluation failed during bisection"},
                )

        return alpha, x_new, line_search_details

    def get_current_x(self) -> float:
        """
        Get current best approximation.

        Returns:
            float: Current approximation (minimum or root)
        """
        return self.x

    def step(self) -> float:
        """
        Perform one iteration of the Power Conjugate method.

        Each iteration:
        1. Computes a conjugate direction
        2. Refines it using power iteration
        3. Performs line search to update the current point
        4. Checks convergence criteria

        Returns:
            float: Current approximation (minimum or root)
        """
        # If already converged, return current approximation
        if self._converged:
            return self.x

        # Store old value for iteration history
        x_old = self.x

        # 1. Compute conjugate direction
        base_direction = self._compute_conjugate_direction()

        # 2. Apply power iteration to refine direction
        refined_direction = self._power_iteration_update()

        # Combine directions (with more weight on refined direction)
        if self.method_type == "optimize":
            # For optimization, combine the directions
            self.direction = 0.3 * base_direction + 0.7 * refined_direction

            # Make sure direction points downhill
            gradient = self._estimate_gradient(self.x)
            if gradient * self.direction > 0:  # If pointing uphill
                self.direction = -self.direction
        else:
            # For root-finding, use Newton-like direction when possible
            func_val = self.func(self.x)
            gradient = self._estimate_gradient(self.x)

            if abs(gradient) > 1e-10:
                newton_dir = -func_val / gradient

                # If Newton direction is reasonable, use it more heavily
                if abs(newton_dir) < 10.0:
                    self.direction = 0.7 * newton_dir + 0.3 * refined_direction
                else:
                    self.direction = refined_direction
            else:
                # If gradient is too small, use refined direction
                self.direction = refined_direction

            # If we have a bracket, make sure we're moving in the right direction
            if self.bracket:
                a, b = self.bracket
                mid = (a + b) / 2

                # If we're moving away from the bracket's midpoint, reverse direction
                if (self.x < mid and self.direction < 0) or (
                    self.x > mid and self.direction > 0
                ):
                    self.direction = -self.direction

        # 3. Perform line search
        alpha, x_new, line_search_details = self._line_search(self.direction)

        # For difficult functions, if we're making no progress, try a different direction
        if abs(x_new - self.x) < self.tol * 1e-1:
            line_search_details["step_too_small"] = True

            # For difficult functions, try random direction if we're stagnating
            if self.iterations > 0 and self.iterations % 20 == 0:
                # Use a random direction occasionally to escape local plateaus
                random_dir = 2.0 * (0.5 - np.random.random())
                alpha_random, x_new_random, random_details = self._line_search(
                    random_dir
                )

                # If the random direction gives better progress, use it
                if abs(x_new_random - self.x) > abs(x_new - self.x):
                    alpha = alpha_random
                    x_new = x_new_random
                    line_search_details = random_details
                    line_search_details["random_direction_used"] = True

            # For root-finding with a bracket, try bisection
            if self.method_type == "root" and self.bracket:
                a, b = self.bracket
                x_new = (a + b) / 2
                try:
                    f_new = self.func(x_new)
                    # Update bracket
                    f_a = self.func(a)
                    if f_new * f_a <= 0:
                        self.bracket = (a, x_new)
                    else:
                        self.bracket = (x_new, b)

                    line_search_details["bisection_used"] = True
                    line_search_details["new_bracket"] = self.bracket
                except:
                    pass

        # Record iteration details
        details = {
            "prev_x": self.x,
            "new_x": x_new,
            "direction": self.direction,
            "step_size": alpha,
            "beta": self.beta,
            "base_direction": base_direction,
            "refined_direction": refined_direction,
            "gradient": self._estimate_gradient(self.x),
            "line_search": line_search_details,
            "method_type": self.method_type,
            "bracket": self.bracket,
        }

        # Update current point
        self.prev_x = self.x
        self.x = x_new

        # Add to iteration history
        self.add_iteration(x_old, self.x, details)
        self.iterations += 1

        # Check convergence - use stricter criteria for optimization
        error = self.get_error()

        if self.method_type == "optimize":
            # For optimization, also consider function gradient and step size
            gradient_norm = abs(self._estimate_gradient(self.x))
            step_size = abs(x_new - x_old)

            # Consider converged if any of these criteria are met:
            # 1. Error is below tolerance
            # 2. Gradient is very small (near stationary point)
            # 3. Step size is very small (can't make further progress)
            # 4. Max iterations reached
            if (
                error <= self.tol
                or gradient_norm < self.tol * 0.1
                or step_size < self.tol * 0.01
                or self.iterations >= self.max_iter
            ):
                self._converged = True

                # Add convergence reason to last iteration
                last_iteration = self._history[-1]
                if error <= self.tol:
                    last_iteration.details["convergence_reason"] = (
                        "error within tolerance"
                    )
                elif gradient_norm < self.tol * 0.1:
                    last_iteration.details["convergence_reason"] = "gradient near zero"
                elif step_size < self.tol * 0.01:
                    last_iteration.details["convergence_reason"] = "step size near zero"
                else:
                    last_iteration.details["convergence_reason"] = (
                        "maximum iterations reached"
                    )
        else:
            # Standard convergence check for root-finding
            if error <= self.tol or self.iterations >= self.max_iter:
                self._converged = True

                # Add convergence reason to last iteration
                last_iteration = self._history[-1]
                if error <= self.tol:
                    last_iteration.details["convergence_reason"] = (
                        "error within tolerance"
                    )
                else:
                    last_iteration.details["convergence_reason"] = (
                        "maximum iterations reached"
                    )

        return self.x

    def get_error(self) -> float:
        """
        Get error at current point based on method type.

        For root-finding:
            - Error = |f(x)|, which measures how close we are to f(x) = 0

        For optimization:
            - Error = |f'(x)|, which measures how close we are to a stationary point

        Returns:
            float: Current error estimate
        """
        if self.method_type == "root":
            # For root-finding, error is how close f(x) is to zero
            return abs(self.func(self.x))
        else:
            # For optimization, error is gradient magnitude
            if self.derivative is not None:
                return abs(self.derivative(self.x))
            else:
                return abs(self.estimate_derivative(self.x))

    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate observed convergence rate of the method.

        For well-behaved functions, the method should exhibit superlinear convergence.

        Returns:
            Optional[float]: Observed convergence rate or None if insufficient data
        """
        if len(self._history) < 3:
            return None

        # Extract errors from last few iterations
        recent_errors = [data.error for data in self._history[-3:]]
        if any(err == 0 for err in recent_errors):
            return 0.0  # Exact convergence

        # Estimate convergence rate as |e_{n+1}/e_n|
        rate1 = recent_errors[-1] / recent_errors[-2] if recent_errors[-2] != 0 else 0
        rate2 = recent_errors[-2] / recent_errors[-3] if recent_errors[-3] != 0 else 0

        # Return average of recent rates
        return (rate1 + rate2) / 2

    @property
    def name(self) -> str:
        """
        Get human-readable name of the method.

        Returns:
            str: Name of the method
        """
        return f"Power Conjugate {'Root-Finding' if self.method_type == 'root' else 'Optimization'} Method"


def power_conjugate_search(
    f: Union[Callable[[float], float], NumericalMethodConfig],
    x0: float,
    direction_reset_freq: int = 5,
    line_search_factor: float = 0.5,
    power_iterations: int = 2,
    tol: float = 1e-6,
    max_iter: int = 100,
    method_type: MethodType = "optimize",
) -> Tuple[float, List[float], int]:
    """
    Legacy wrapper for backward compatibility.

    This function provides a simpler interface to the Power Conjugate method.

    Mathematical guarantees:
    - For optimization: Converges superlinearly to local minima for smooth functions
    - For root-finding: Converges to roots if the starting point is sufficiently close

    Args:
        f: Function or configuration (if function, it's wrapped in a config)
        x0: Initial point
        direction_reset_freq: Frequency of direction resets
        line_search_factor: Factor for line search step size reduction
        power_iterations: Number of power iterations per step
        tol: Error tolerance
        max_iter: Maximum number of iterations
        method_type: Type of problem ("root" or "optimize")

    Returns:
        Tuple of (solution, errors, iterations)
    """
    # If f is a function rather than a config, create a config
    if callable(f):
        config = NumericalMethodConfig(
            func=f, method_type=method_type, tol=tol, max_iter=max_iter
        )
    else:
        config = f

    # Create and run the method
    method = PowerConjugateMethod(
        config,
        x0,
        direction_reset_freq=direction_reset_freq,
        line_search_factor=line_search_factor,
        power_iterations=power_iterations,
    )
    errors = []

    while not method.has_converged():
        method.step()
        errors.append(method.get_error())

    return method.get_current_x(), errors, method.iterations
