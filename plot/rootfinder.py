# plot/rootfinder.py

"""
Visualization utilities for comparing root finding methods.

This module provides interactive and comparative visualizations for various
root finding algorithms. It is designed to work with any root finding implementation
that follows the defined protocol.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

from algorithms.convex.protocols import RootFinder, RootFinderConfig


# Configuration for visualization options like figure size, animation speed, styles, etc.
@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: Tuple[int, int] = (15, 8)  # Default figure size
    animation_interval: int = 500  # Default animation interval in milliseconds
    show_convergence: bool = True  # Default to show convergence plot
    show_error: bool = True  # Default to show error plot
    style: str = "darkgrid"  # (darkgrid, whitegrid, dark, white, ticks)
    context: str = "notebook"  # (paper, notebook, talk, poster)
    palette: str = "husl"  # (husl, hls, viridis, magma, etc.)
    point_size: int = 100  # Default point size
    dpi: int = 100  # Default DPI for figure resolution
    show_legend: bool = True  # Default to show legend
    grid_alpha: float = 0.3  # Default grid transparency
    title: str = "Root Finding Methods Comparison"
    background_color: str = "#2E3440"  # Nord theme dark background
    verbose: bool = False  # Add this line


class RootFindingVisualizer:
    """Visualizer for comparing root finding methods."""

    def __init__(
        self,
        problem: RootFinderConfig,
        methods: List[RootFinder],
        config: Optional[VisualizationConfig] = None,
    ):
        # Store the problem, methods, and visualization configuration
        self.problem = problem
        self.methods = methods
        self.config = config or VisualizationConfig()

        # Apply Seaborn styling based on the provided configuration
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

        # Setup the layout and styling for subplots
        self.setup_plots()

        # Initialize dictionaries to store method histories and error estimates over iterations
        self.histories: Dict[str, List[float]] = {method.name: [] for method in methods}
        self.errors: Dict[str, List[float]] = {method.name: [] for method in methods}

        # Determine a color for each method from the configured palette
        n_colors = len(methods)
        if self.config.palette in plt.colormaps:
            colors = plt.colormaps[self.config.palette]
            # Handle single method case: choose the middle color of the colormap
            if n_colors == 1:
                color_list = [colors(0.5)]
            else:
                color_list = [colors(i / (n_colors - 1)) for i in range(n_colors)]
        else:
            # Use Seaborn's palette function if not using a matplotlib colormap
            color_list = sns.color_palette(self.config.palette, n_colors=n_colors)

        # Initialize state for each method: store the method, its associated line on the plot, and its color.
        self.method_states = {}
        for i, method in enumerate(self.methods):
            color = color_list[i]
            # Plot an initial empty line for each method on the function plot
            (line,) = self.ax_func.plot(
                [], [], "o-", color=color, label=method.name, linewidth=2, markersize=8
            )
            self.method_states[method.name] = {
                "method": method,
                "line": line,
                "color": color,
            }

        # Optionally display a legend on the function plot
        if self.config.show_legend:
            self.ax_func.legend()

        # Setup convergence plot lines if that axis exists
        if self.ax_conv:
            self.conv_lines = {
                name: self.ax_conv.plot([], [], label=name, color=state["color"])[0]
                for name, state in self.method_states.items()
            }

        # Setup error plot lines if that axis exists
        if self.ax_error:
            self.error_lines = {
                name: self.ax_error.plot([], [], label=name, color=state["color"])[0]
                for name, state in self.method_states.items()
            }

        # Adjust layout to prevent overlapping and update the canvas
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setup_plots(self):
        """Setup the subplot layout with enhanced styling."""
        # If both convergence and error plots are enabled, create a 2x2 grid layout
        if self.config.show_convergence and self.config.show_error:
            gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
            self.ax_func = self.fig.add_subplot(gs[0, :])  # Top row for function plot
            self.ax_conv = self.fig.add_subplot(gs[1, 0])  # Bottom left for convergence
            self.ax_error = self.fig.add_subplot(gs[1, 1])  # Bottom right for error
        else:
            # If not, use a single plot
            self.ax_func = self.fig.add_subplot(111)
            self.ax_conv = None
            self.ax_error = None

        # Style each axis with the specified background and text colors
        for ax in [self.ax_func, self.ax_conv, self.ax_error]:
            if ax is not None:
                ax.set_facecolor(self.config.background_color)
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")

        # Plot the target function with a gradient line to enhance visualization.
        x = np.linspace(*self.problem.x_range, 1000)
        # Evaluate the function at each x value
        y = [self.problem.func(xi) for xi in x]

        # Create a custom colormap using Nord theme blues
        colors = ["#81A1C1", "#88C0D0", "#8FBCBB"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        # Create segments for the function line to apply the gradient color mapping
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(y), max(y))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.array(y))
        self.ax_func.add_collection(lc)

        # Add a horizontal line at y=0 (the x-axis) to mark the root level.
        self.ax_func.axhline(y=0, color="#BF616A", linestyle="--", alpha=0.5)
        # Enable grid lines with the specified transparency
        self.ax_func.grid(True, alpha=self.config.grid_alpha, color="gray")
        self.ax_func.set_title("Root Finding Methods Comparison", pad=20)

        # Setup the convergence plot's title, labels, and grid (if it exists)
        if self.ax_conv:
            self.ax_conv.set_title("Convergence Plot")
            self.ax_conv.set_xlabel("Iteration")
            self.ax_conv.set_ylabel("x value")
            self.ax_conv.grid(True, alpha=self.config.grid_alpha, color="gray")

        # Setup the error plot with title, labels, and a logarithmic y-scale for error values.
        if self.ax_error:
            self.ax_error.set_title("Error Plot")
            self.ax_error.set_xlabel("Iteration")
            self.ax_error.set_ylabel("|f(x)|")
            self.ax_error.set_yscale("log")
            self.ax_error.grid(True, alpha=self.config.grid_alpha, color="gray")

        # Set the overall figure background color
        self.fig.patch.set_facecolor(self.config.background_color)

    def run_comparison(self):
        """Run comparison in real-time."""
        all_converged = False
        iteration = 0

        while not all_converged and iteration < self.problem.max_iter:
            all_converged = True

            for name, state in self.method_states.items():
                method = state["method"]
                if not method.has_converged():
                    all_converged = False

                    # Get current x value before step
                    x_old = method.get_current_x()

                    # Perform one step
                    x_new = method.step()

                    # Get iteration data
                    iter_data = method.get_last_iteration()

                    # Update visualization
                    self.histories[name].append(x_new)
                    self.errors[name].append(
                        iter_data.error if iter_data else abs(self.problem.func(x_new))
                    )

                    # Update the function plot with the latest approximation point
                    state["line"].set_data([x_new], [self.problem.func(x_new)])

                    # Update the convergence plot with the full history of approximations
                    if self.ax_conv:
                        self.conv_lines[name].set_data(
                            range(len(self.histories[name])), self.histories[name]
                        )
                        self.ax_conv.relim()
                        self.ax_conv.autoscale_view()

                    # Update the error plot with the error history
                    if self.ax_error:
                        self.error_lines[name].set_data(
                            range(len(self.errors[name])), self.errors[name]
                        )
                        self.ax_error.relim()
                        self.ax_error.autoscale_view()

                    # Redraw and pause to show animation
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.pause(
                        self.config.animation_interval / 1000
                    )  # Convert to seconds

                if all_converged:
                    break

            iteration += 1

        # Once finished, keep the plot visible
        plt.show()
