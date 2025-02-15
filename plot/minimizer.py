# plot/minimizer.py

"""Visualize optimization methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Function(Protocol):
    """Protocol for objective functions."""

    def __call__(self, x: np.ndarray) -> float: ...


class Gradient(Protocol):
    """Protocol for gradient functions."""

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class Hessian(Protocol):
    """Protocol for Hessian functions."""

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class OptimizationProblem:
    """Configuration for optimization problem.

    Attributes:
        func: Objective function to minimize
        grad: Gradient of objective function
        hess: Hessian of objective function (optional)
        x0: Initial guess
        bounds: Plot bounds for visualization (x_min, x_max, y_min, y_max)
        tolerance: Convergence tolerance
        max_iter: Maximum iterations
    """

    func: Function
    grad: Gradient
    hess: Optional[Hessian] = None
    x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5)
    tolerance: float = 1e-6
    max_iter: int = 100


@dataclass
class VisualizationConfig:
    """Configuration for visualization.

    Attributes:
        figsize: Figure size in inches
        n_contours: Number of contour lines
        interval: Animation interval in milliseconds
        colormap: Colormap for contour plot
        show_path: Whether to show optimization path
        show_gradient: Whether to show gradient arrows
        path_color: Color for optimization path
        marker_size: Size of current point marker
    """

    figsize: Tuple[int, int] = (10, 8)
    n_contours: int = 50
    interval: int = 100
    colormap: str = "viridis"
    show_path: bool = True
    show_gradient: bool = True
    path_color: str = "red"
    marker_size: int = 100


class OptimizationMethod(ABC):
    """Base class for optimization methods."""

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.current_x = problem.x0
        self.path = [problem.x0]
        self.converged = False

    @abstractmethod
    def step(self) -> np.ndarray:
        """Perform one optimization step."""
        pass

    def has_converged(self) -> bool:
        """Check if optimization has converged."""
        return self.converged


class GradientDescent(OptimizationMethod):
    """Gradient descent optimization."""

    def __init__(self, problem: OptimizationProblem, learning_rate: float = 0.1):
        super().__init__(problem)
        self.learning_rate = learning_rate

    def step(self) -> np.ndarray:
        grad = self.problem.grad(self.current_x)
        self.current_x = self.current_x - self.learning_rate * grad
        self.path.append(self.current_x.copy())

        if np.linalg.norm(grad) < self.problem.tolerance:
            self.converged = True

        return self.current_x


class NewtonMethod(OptimizationMethod):
    """Newton's method optimization."""

    def step(self) -> np.ndarray:
        grad = self.problem.grad(self.current_x)
        hess = self.problem.hess(self.current_x)

        try:
            delta = np.linalg.solve(hess, -grad)
            self.current_x = self.current_x + delta
            self.path.append(self.current_x.copy())

            if np.linalg.norm(grad) < self.problem.tolerance:
                self.converged = True
        except np.linalg.LinAlgError:
            self.converged = True  # Stop if Hessian is singular

        return self.current_x


class OptimizationVisualizer:
    """Real-time visualization of optimization methods."""

    def __init__(
        self, problem: OptimizationProblem, config: Optional[VisualizationConfig] = None
    ):
        self.problem = problem
        self.config = config or VisualizationConfig()

        # Create figure and subplots
        self.fig = plt.figure(figsize=self.config.figsize)
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1])

        # Main contour plot
        self.ax_contour = self.fig.add_subplot(gs[0, :])
        # Error plot
        self.ax_error = self.fig.add_subplot(gs[1, 0])
        # Function value plot
        self.ax_value = self.fig.add_subplot(gs[1, 1])

        self.setup_plots()
        self.quiver = None
        self.errors = []
        self.values = []
        self.iterations = []

    def setup_plots(self):
        """Setup all plots."""
        # Setup contour plot
        x_min, x_max, y_min, y_max = self.problem.bounds
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.problem.func(np.array([xi, yi])) for xi in x] for yi in y])

        # Plot contour
        contour = self.ax_contour.contour(
            X, Y, Z, levels=self.config.n_contours, cmap=self.config.colormap
        )
        self.fig.colorbar(contour, ax=self.ax_contour)
        self.ax_contour.set_xlabel("x₁")
        self.ax_contour.set_ylabel("x₂")
        self.ax_contour.set_title("Optimization Progress")

        # Setup error plot
        (self.error_line,) = self.ax_error.plot([], [], "r-")
        self.ax_error.set_xlabel("Iteration")
        self.ax_error.set_ylabel("||∇f||")
        self.ax_error.set_yscale("log")
        self.ax_error.grid(True)

        # Setup value plot
        (self.value_line,) = self.ax_value.plot([], [], "b-")
        self.ax_value.set_xlabel("Iteration")
        self.ax_value.set_ylabel("f(x)")
        self.ax_value.grid(True)

        plt.tight_layout()

    def _update_gradient_arrow(self, x: np.ndarray, grad: np.ndarray):
        """Update gradient arrow with safety checks."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:  # Only draw if gradient is significant
            # Remove previous arrow if it exists
            if self.quiver is not None:
                self.quiver.remove()

            # Normalize gradient and scale for visualization
            scale_factor = 0.5  # Adjust this to change arrow length
            normalized_grad = -grad * scale_factor / grad_norm

            self.quiver = self.ax_contour.quiver(
                x[0],
                x[1],
                normalized_grad[0],
                normalized_grad[1],
                color="blue",
                alpha=0.5,
                scale=1.0,
                scale_units="xy",
                angles="xy",
                width=0.02,
            )

    def optimize(self, method: OptimizationMethod):
        """Run optimization with animation."""
        # Initialize plots
        (path_line,) = self.ax_contour.plot(
            [], [], color=self.config.path_color, marker="o", markersize=8
        )
        current_point = self.ax_contour.scatter(
            [], [], color="red", s=self.config.marker_size, marker="*"
        )

        def init():
            """Initialize animation."""
            path_line.set_data([], [])
            current_point.set_offsets(np.c_[[], []])
            self.error_line.set_data([], [])
            self.value_line.set_data([], [])
            return path_line, current_point, self.error_line, self.value_line

        def update(frame):
            """Update animation."""
            if not method.has_converged():
                # Perform optimization step
                x = method.step()
                path = np.array(method.path)

                # Update path and current point
                if self.config.show_path:
                    path_line.set_data(path[:, 0], path[:, 1])
                current_point.set_offsets([x[0], x[1]])

                # Update gradient arrow
                if self.config.show_gradient and frame % 5 == 0:
                    grad = self.problem.grad(x)
                    self._update_gradient_arrow(x, grad)

                # Update error and value plots
                grad_norm = np.linalg.norm(self.problem.grad(x))
                func_value = self.problem.func(x)

                self.errors.append(grad_norm)
                self.values.append(func_value)
                self.iterations.append(frame)

                self.error_line.set_data(self.iterations, self.errors)
                self.value_line.set_data(self.iterations, self.values)

                # Adjust plot limits
                self.ax_error.relim()
                self.ax_error.autoscale_view()
                self.ax_value.relim()
                self.ax_value.autoscale_view()

                # Update title with current values
                self.ax_contour.set_title(
                    f"Optimization Progress\n"
                    f"f(x) = {func_value:.6f}, ||∇f|| = {grad_norm:.6f}"
                )

            return path_line, current_point, self.error_line, self.value_line

        anim = FuncAnimation(
            self.fig,
            update,
            frames=self.problem.max_iter,
            interval=self.config.interval,
            init_func=init,
            blit=True,
            repeat=False,
        )
        plt.show()


# # Example usage:
# if __name__ == "__main__":
#     # Example: Himmelblau's function - a challenging optimization problem
#     # f(x,y) = (x² + y - 11)² + (x + y² - 7)²
#     # Has four local minima at:
#     # (3.0, 2.0), (-2.805118, 3.131312),
#     # (-3.779310, -3.283186), (3.584428, -1.848126)
#     def himmelblau(x):
#         return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

#     def himmelblau_grad(x):
#         return np.array(
#             [
#                 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
#                 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7),
#             ]
#         )

#     def himmelblau_hess(x):
#         return np.array(
#             [
#                 [12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
#                 [4 * x[0] + 4 * x[1], 4 * x[0] ** 2 + 12 * x[1] ** 2 - 26],
#             ]
#         )

#     # Setup multiple starting points to show different convergence paths
#     starting_points = [
#         np.array([-4.0, 4.0]),
#         np.array([4.0, 4.0]),
#         np.array([-4.0, -4.0]),
#         np.array([4.0, -4.0]),
#         np.array([0.0, 0.0]),
#     ]

#     # Configure visualization
#     config = VisualizationConfig(
#         interval=50,  # Faster animation
#         show_gradient=True,
#         show_path=True,
#         n_contours=50,  # More contour lines
#         colormap="viridis",
#         figsize=(12, 10),  # Larger figure
#     )

#     # Try different optimization methods from different starting points
#     for x0 in starting_points:
#         # Setup optimization problem
#         problem = OptimizationProblem(
#             func=himmelblau,
#             grad=himmelblau_grad,
#             hess=himmelblau_hess,
#             x0=x0,
#             bounds=(-6, 6, -6, 6),  # Wider view
#             tolerance=1e-8,  # Stricter convergence
#             max_iter=200,  # More iterations
#         )

#         # Create visualizer
#         visualizer = OptimizationVisualizer(problem, config)

#         # Run with Newton's method
#         print(f"\nNewton's Method from x0 = {x0}")
#         method = NewtonMethod(problem)
#         visualizer.optimize(method)

#         # Run with gradient descent (with different learning rates)
#         print(f"\nGradient Descent from x0 = {x0}")
#         method = GradientDescent(problem, learning_rate=0.01)
#         visualizer.optimize(method)

#         # Try a more aggressive learning rate
#         print(f"\nGradient Descent (faster) from x0 = {x0}")
#         method = GradientDescent(problem, learning_rate=0.05)
#         visualizer.optimize(method)

#     # Also try a more exotic function: Rastrigin function
#     # Global minimum at (0,0) but many local minima
#     def rastrigin(x):
#         A = 10
#         n = len(x)
#         return A * n + sum(x**2 - A * np.cos(2 * np.pi * x))

#     def rastrigin_grad(x):
#         A = 10
#         return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)

#     def rastrigin_hess(x):
#         A = 10
#         return np.diag(2 + 4 * A * np.pi**2 * np.cos(2 * np.pi * x))

#     # Try Rastrigin function
#     problem = OptimizationProblem(
#         func=rastrigin,
#         grad=rastrigin_grad,
#         hess=rastrigin_hess,
#         x0=np.array([3.0, 3.0]),
#         bounds=(-5.12, 5.12, -5.12, 5.12),  # Standard bounds for Rastrigin
#         tolerance=1e-8,
#         max_iter=300,
#     )

#     # Configure visualization for Rastrigin
#     config = VisualizationConfig(
#         interval=30,
#         show_gradient=True,
#         show_path=True,
#         n_contours=100,  # More contours to show the complex landscape
#         colormap="coolwarm",  # Different colormap to highlight valleys
#         figsize=(12, 10),
#     )

#     visualizer = OptimizationVisualizer(problem, config)

#     # Try both methods
#     print("\nRastrigin function optimization")
#     method = NewtonMethod(problem)
#     visualizer.optimize(method)

#     method = GradientDescent(problem, learning_rate=0.01)
#     visualizer.optimize(method)
