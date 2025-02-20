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
import seaborn as sns

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig


# Configuration for visualization options like figure size, animation speed, styles, etc.
@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: Tuple[int, int] = (15, 8)  # Default figure size
    animation_interval: int = 50  # Faster updates for optimization
    show_convergence: bool = True  # Default to show convergence plot
    show_error: bool = True  # Default to show error plot
    show_contour: bool = True  # Default to show contour for 2D
    style: str = "white"  # Options: darkgrid, whitegrid, dark, white, ticks
    context: str = "talk"  # Options: paper, notebook, talk, poster
    palette: str = "viridis"  # Default colormap
    point_size: int = 100  # Default point size
    dpi: int = 100  # Default DPI for figure resolution
    show_legend: bool = True  # Default to show legend
    grid_alpha: float = 0.3  # Default grid transparency
    title: str = "Optimization Methods Comparison"
    background_color: str = "#FFFFFF"  # White background
    verbose: bool = False  # Enable verbose output


class OptimizationVisualizer:
    """Visualizer for comparing optimization methods.

    Handles both 1D and 2D optimization problems with appropriate
    visualizations for each case.
    """

    def __init__(
        self,
        problem: NumericalMethodConfig,
        methods: List[BaseNumericalMethod],
        config: Optional[VisualizationConfig] = None,
    ):
        """Initialize the visualizer with optimization configuration.

        Args:
            problem: Configuration containing the objective function
            methods: List of optimization methods to compare
            config: Optional visualization configuration

        Note:
            Expects problem.method_type to be "optimize"
        """
        if problem.method_type != "optimize":
            raise ValueError("OptimizationVisualizer requires method_type='optimize'")

        self.problem = problem
        self.methods = methods
        self.config = config or VisualizationConfig()

        # Set up the style using seaborn
        sns.set_style(self.config.style)
        sns.set_context(self.config.context)

        # Enable interactive mode for live updating of the plots
        plt.ion()

        # Create the main figure with the specified size, dpi, and background color
        self.fig = plt.figure(
            figsize=self.config.figsize,
            dpi=self.config.dpi,
            facecolor=self.config.background_color,
        )

        # Create figure and determine if problem is 1D or 2D
        self.is_2d = self._check_if_2d()
        self.setup_plots()

        # Initialize method states and visualization elements
        self._initialize_visualization()

    def _initialize_visualization(self):
        """Initialize visualization state for all methods."""
        n_colors = len(self.methods)
        self.colors = self._get_colors(n_colors)

        self.method_states = {}
        self.error_lines = {}

        for i, method in enumerate(self.methods):
            method_id = id(method)
            color = self.colors[i]
            name = method.__class__.__name__

            # Create visualization elements
            if self.is_2d:
                main_line = self.ax_main.plot(
                    [],
                    [],
                    [],
                    "o-",
                    color=color,
                    label=name,
                    linewidth=2,
                    markersize=8,
                    alpha=0.8,
                    zorder=2,
                )[0]
                contour_line = (
                    self.ax_contour.plot(
                        [], [], "o-", color=color, linewidth=2, markersize=8, zorder=2
                    )[0]
                    if self.config.show_contour
                    else None
                )
            else:
                main_line = self.ax_main.plot(
                    [],
                    [],
                    "o-",
                    color=color,
                    label=name,
                    linewidth=2,
                    markersize=8,
                    zorder=2,
                )[0]
                contour_line = None

            error_line = self.ax_error.plot(
                [], [], "-", color=color, label=name, linewidth=2
            )[0]

            self.method_states[method_id] = {
                "method": method,
                "line": main_line,
                "contour_line": contour_line,
                "color": color,
                "points": [],
                "errors": [],
            }
            self.error_lines[method_id] = error_line

        # Setup legends and labels
        if self.config.show_legend:
            self.ax_main.legend(loc="upper right")
            self.ax_error.legend(loc="upper right")

        # Configure error plot
        self.ax_error.set_xlabel("Iteration")
        self.ax_error.set_ylabel("Error")
        self.ax_error.grid(True, alpha=0.3)
        self.ax_error.set_yscale("log")  # Use log scale for error plot
        self.ax_error.set_xlim(-1, self.problem.max_iter)  # Set x-axis limits
        self.ax_error.set_ylim(1e-8, 1e2)  # Set reasonable y-axis limits for error

        # Plot the objective function
        self.plot_function()
        plt.tight_layout()

    def _get_colors(self, n_colors: int) -> List:
        """Get color scheme for methods."""
        if self.config.palette in plt.colormaps:
            cmap = plt.colormaps[self.config.palette]
            return [
                cmap(i / (n_colors - 1) if n_colors > 1 else 0.5)
                for i in range(n_colors)
            ]
        return sns.color_palette(self.config.palette, n_colors=n_colors)

    def _check_if_2d(self) -> bool:
        """Check if the optimization problem is 2D."""
        try:
            # Try calling with a 2D point
            test_point = np.array([0.0, 0.0])
            self.problem.func(test_point)
            # If it works and x0 is 2D, then it's a 2D problem
            return len(self.methods[0].get_current_x()) == 2
        except:
            return False

    def setup_plots(self):
        """Setup the subplot layout based on problem dimensionality."""
        if self.is_2d:
            gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
            self.ax_main = self.fig.add_subplot(gs[0, :], projection="3d")
            self.ax_contour = self.fig.add_subplot(gs[1, 0])
            self.ax_error = self.fig.add_subplot(gs[1, 1])
        else:
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_error = self.fig.add_subplot(gs[1])

        self.fig.suptitle(self.config.title)

    def plot_function(self):
        """Plot the objective function landscape."""
        x = np.linspace(*self.problem.x_range, 200)
        if self.is_2d:
            self._plot_2d_function(x)
        else:
            self._plot_1d_function()

    def _plot_1d_function(self):
        """Plot 1D optimization landscape."""
        # Use more points and wider range for smooth visualization
        x_min, x_max = self.problem.x_range
        x_range = x_max - x_min
        # Extend range by 20% on each side
        x_plot_min = x_min - 0.2 * x_range
        x_plot_max = x_max + 0.2 * x_range
        x = np.linspace(x_plot_min, x_plot_max, 1000)

        # Compute function values
        y = np.array([self.problem.func(np.array([xi])) for xi in x])

        # Plot the function as a static line
        self.function_line = self.ax_main.plot(
            x, y, "-", color="gray", alpha=0.5, label="f(x)", zorder=1, linewidth=2
        )[0]

        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel("x")
        self.ax_main.set_ylabel("f(x)")

        # Fix scientific notation and scaling
        self.ax_main.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")

        # Set view limits with padding
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-10)  # Avoid zero range
        y_plot_min = y_min - 0.1 * y_range
        y_plot_max = y_max + 0.1 * y_range

        self.ax_main.set_xlim(x_plot_min, x_plot_max)
        self.ax_main.set_ylim(y_plot_min, y_plot_max)

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

        # Surface plot - store reference to surface
        self.surface = self.ax_main.plot_surface(
            X, Y, Z, cmap="viridis", alpha=0.3, zorder=1
        )
        self.ax_main.set_xlabel("x")
        self.ax_main.set_ylabel("y")
        self.ax_main.set_zlabel("f(x, y)")

        # Contour plot - store reference to contours
        if self.config.show_contour:
            self.contours = self.ax_contour.contour(
                X, Y, Z, levels=20, colors="gray", alpha=0.5, zorder=1
            )
            self.ax_contour.set_xlabel("x")
            self.ax_contour.set_ylabel("y")

    def run_comparison(self):
        """Run and visualize the optimization methods."""
        plt.ion()

        # Generate test points and find global minimum
        if self.problem.is_2d:
            x = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            y = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            X, Y = np.meshgrid(x, y)
            x_test = np.vstack((X.ravel(), Y.ravel())).T
            f_min = np.min([self.problem.func(x) for x in x_test])
        else:
            x_test = np.linspace(self.problem.x_range[0], self.problem.x_range[1], 100)
            f_min = np.min([self.problem.func(np.array([x])) for x in x_test])

        for iteration in range(self.problem.max_iter):
            any_updated = False

            for method_id, state in self.method_states.items():
                method = state["method"]
                if not method.has_converged():
                    any_updated = True
                    x_new = method.step()
                    f_new = self.problem.func(x_new)

                    # Get error as scalar (gradient norm for optimization)
                    error = float(np.linalg.norm(method.derivative(x_new)))

                    state["points"].append(x_new)
                    state["errors"].append(error)

                    # Update optimization path
                    points = np.array(state["points"])
                    values = np.array([self.problem.func(p) for p in points])

                    if self.is_2d:
                        # Update 3D path and contour
                        state["line"].set_data(points[:, 0], points[:, 1])
                        state["line"].set_3d_properties(values)
                        if state["contour_line"]:
                            state["contour_line"].set_data(points[:, 0], points[:, 1])
                    else:
                        # Update 1D path
                        state["line"].set_data(points.ravel(), values)

                    # Update error plot with iteration numbers
                    iterations = np.arange(len(state["errors"]))
                    errors = np.array(state["errors"])
                    self.error_lines[method_id].set_data(iterations, errors)

            if any_updated:
                # Update error plot limits
                if any(
                    len(state["errors"]) > 0 for state in self.method_states.values()
                ):
                    all_errors = np.concatenate(
                        [
                            state["errors"]
                            for state in self.method_states.values()
                            if len(state["errors"]) > 0
                        ]
                    )
                    error_min, error_max = np.min(all_errors), np.max(all_errors)
                    self.ax_error.set_ylim(
                        max(error_min * 0.1, 1e-10),  # Avoid zero on log scale
                        error_max * 10,
                    )

                self.fig.canvas.draw()
                plt.pause(0.01)

            if all(method.has_converged() for method in self.methods):
                break

        plt.ioff()
        plt.show()

    def _update_plot_limits(self):
        """Update axis limits to show all data."""
        if self.is_2d:
            all_points = np.concatenate(
                [
                    state["points"]
                    for state in self.method_states.values()
                    if len(state["points"]) > 0
                ]
            )
            all_values = [
                v for state in self.method_states.values() for v in state["errors"]
            ]

            self.ax_main.set_xlim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))
            self.ax_main.set_ylim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
            self.ax_main.set_zlim(np.min(all_values), np.max(all_values))

            if self.config.show_contour:
                self.ax_contour.set_xlim(
                    np.min(all_points[:, 0]), np.max(all_points[:, 0])
                )
                self.ax_contour.set_ylim(
                    np.min(all_points[:, 1]), np.max(all_points[:, 1])
                )
        else:
            # For 1D case, keep x limits fixed as set in _plot_1d_function
            # Only update error plot limits
            if any(len(state["errors"]) > 0 for state in self.method_states.values()):
                self.ax_error.relim()
                self.ax_error.autoscale_view()
