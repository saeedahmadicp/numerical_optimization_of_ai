# plot/optimizer_viz.py

"""
Visualization utilities for comparing optimization methods.

This module provides interactive and comparative visualizations for various
optimization algorithms. It supports both 1D and 2D optimization problems,
with features for visualizing:
- Function surfaces and contours
- Optimization paths
- Convergence behavior
- Error trends
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from algorithms.convex.protocols import RootFinder, RootFinderConfig


# Configuration for visualization options like figure size, animation speed, styles, etc.
@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: Tuple[int, int] = (15, 8)  # Default figure size
    animation_interval: int = 500  # Default animation interval in milliseconds
    show_convergence: bool = True  # Default to show convergence plot
    show_error: bool = True  # Default to show error plot
    show_contour: bool = True  # Default to show contour for 2D
    style: str = "default"  # Use Matplotlib's default style
    palette: str = "viridis"  # Default colormap
    point_size: int = 100  # Default point size
    dpi: int = 100  # Default DPI for figure resolution
    show_legend: bool = True  # Default to show legend
    grid_alpha: float = 0.3  # Default grid transparency
    title: str = "Optimization Methods Comparison"
    background_color: str = "#FFFFFF"  # White background


class OptimizationVisualizer:
    """Visualizer for comparing optimization methods.

    Handles both 1D and 2D optimization problems with appropriate
    visualizations for each case.
    """

    def __init__(
        self,
        problem: RootFinderConfig,
        methods: List[RootFinder],
        config: Optional[VisualizationConfig] = None,
    ):
        """Initialize the visualizer with optimization configuration.

        Args:
            problem: Configuration containing the objective function
            methods: List of optimization methods to compare
            config: Optional visualization configuration
        """
        self.problem = problem
        self.methods = methods
        self.config = config or VisualizationConfig()

        # Set up the style
        plt.style.use(self.config.style)
        plt.rcParams["figure.dpi"] = self.config.dpi
        plt.rcParams["figure.facecolor"] = self.config.background_color
        plt.rcParams["axes.facecolor"] = self.config.background_color
        plt.rcParams["grid.alpha"] = self.config.grid_alpha

        # Create figure and determine if problem is 1D or 2D
        self.is_2d = self._check_if_2d()
        self.setup_plots()

        # Initialize storage for optimization history
        self.histories = {id(method): [] for method in methods}
        self.errors = {id(method): [] for method in methods}

        # Generate color scheme for methods using specified palette
        n_colors = len(methods)
        self.colors = plt.cm.get_cmap(self.config.palette)(np.linspace(0, 1, n_colors))

        # Initial plot setup
        self.plot_function()
        plt.tight_layout()

    def _check_if_2d(self) -> bool:
        """Check if the optimization problem is 2D."""
        try:
            # Try calling with a 2D point
            test_point = np.array([0.0, 0.0])
            self.problem.func(test_point)
            # If it works and x0 is 2D, then it's a 2D problem
            return len(self.methods[0].x) == 2
        except:
            return False

    def setup_plots(self):
        """Setup the subplot layout based on problem dimensionality."""
        self.fig = plt.figure(figsize=self.config.figsize)

        if self.is_2d:
            # 2D optimization layout
            gs = plt.GridSpec(2, 2)
            self.ax_main = self.fig.add_subplot(gs[0, :], projection="3d")
            if self.config.show_contour:
                self.ax_contour = self.fig.add_subplot(gs[1, 0])
            self.ax_error = self.fig.add_subplot(gs[1, 1])
        else:
            # 1D optimization layout
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_error = self.fig.add_subplot(gs[1])

        self.fig.suptitle(self.config.title)

    def plot_function(self):
        """Plot the objective function."""
        x = np.linspace(*self.problem.x_range, 100)

        if self.is_2d:
            self._plot_2d_function(x)
        else:
            self._plot_1d_function(x)

    def _plot_1d_function(self, x: np.ndarray):
        """Plot 1D objective function."""
        y = [self.problem.func(xi) for xi in x]
        self.ax_main.plot(x, y, "b-", alpha=0.5)
        self.ax_main.grid(True)
        self.ax_main.set_xlabel("x")
        self.ax_main.set_ylabel("f(x)")

    def _plot_2d_function(self, x: np.ndarray):
        """Plot 2D objective function with surface and contour plots."""
        y = x
        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                [self.problem.func(np.array([xi, yi])) for xi, yi in zip(x_row, y_row)]
                for x_row, y_row in zip(X, Y)
            ]
        )

        # Surface plot
        self.ax_main.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        self.ax_main.set_xlabel("x")
        self.ax_main.set_ylabel("y")
        self.ax_main.set_zlabel("f(x, y)")

        # Contour plot
        if self.config.show_contour:
            self.ax_contour.contour(X, Y, Z, levels=20)
            self.ax_contour.set_xlabel("x")
            self.ax_contour.set_ylabel("y")

    def run_comparison(self):
        """Run and visualize the optimization methods."""
        for iteration in range(self.problem.max_iter):
            # Update each method
            for method, color in zip(self.methods, self.colors):
                if not method.has_converged():
                    x_new = method.step()
                    self.histories[id(method)].append(x_new)
                    self.errors[id(method)].append(self.problem.func(x_new))

                    # Plot current point
                    if self.is_2d:
                        self._plot_2d_point(x_new, color, method)
                    else:
                        self._plot_1d_point(x_new, color, method)

            # Update error plot
            self._update_error_plot()

            # Update display
            plt.pause(self.config.animation_interval / 1000)

            # Check convergence
            if all(method.has_converged() for method in self.methods):
                break

        plt.ioff()
        plt.show()

    def _plot_1d_point(self, x: float, color: str, method: RootFinder):
        """Plot current point for 1D optimization."""
        self.ax_main.scatter(
            x,
            self.problem.func(x),
            color=color,
            label=method.__class__.__name__,
            s=self.config.point_size,
        )
        if self.config.show_legend:
            self.ax_main.legend()

    def _plot_2d_point(self, x: np.ndarray, color: str, method: RootFinder):
        """Plot current point for 2D optimization."""
        z = self.problem.func(x)
        self.ax_main.scatter(
            x[0],
            x[1],
            z,
            color=color,
            label=method.__class__.__name__,
            s=self.config.point_size,
        )
        if self.config.show_contour:
            self.ax_contour.scatter(x[0], x[1], color=color, s=self.config.point_size)
        if self.config.show_legend:
            self.ax_main.legend()

    def _update_error_plot(self):
        """Update the error plot showing optimization progress."""
        self.ax_error.clear()
        for method, error_history, color in zip(
            self.methods, self.errors.values(), self.colors
        ):
            if error_history:
                self.ax_error.semilogy(
                    error_history, color=color, label=method.__class__.__name__
                )
        self.ax_error.set_xlabel("Iteration")
        self.ax_error.set_ylabel("f(x)")
        self.ax_error.grid(True)
        self.ax_error.legend()
