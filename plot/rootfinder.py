# plot/rootfinder.py

"""
Visualization utilities for comparing root finding methods.

This module provides interactive and comparative visualizations for various
root finding algorithms. It is designed to work with any root finding implementation
that follows the defined protocol.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Protocol

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


class Function(Protocol):
    """Protocol for functions that can be root-found."""

    def __call__(self, x: float) -> float: ...


class RootFindingMethod(Protocol):
    """Protocol that root finding implementations must follow."""

    def step(self) -> float:
        """Perform one iteration of the method.

        Returns:
            float: Current approximation of the root
        """
        ...

    def get_error(self) -> float:
        """Get current error estimate.

        Returns:
            float: Current error estimate
        """
        ...

    def has_converged(self) -> bool:
        """Check if method has converged.

        Returns:
            bool: True if converged, False otherwise
        """
        ...

    @property
    def name(self) -> str:
        """Name of the method for display purposes."""
        ...


@dataclass
class RootFindingProblem:
    """Configuration for a root finding problem.

    Attributes:
        func: Function whose root we want to find
        x_range: Range to plot function (xmin, xmax)
        tolerance: Convergence tolerance
        max_iter: Maximum iterations
    """

    func: Function
    x_range: Tuple[float, float] = (-10, 10)
    tolerance: float = 1e-6
    max_iter: int = 100


@dataclass
class VisualizationConfig:
    """Configuration for visualization.

    Attributes:
        figsize: Figure size in inches
        animation_interval: Milliseconds between animation frames
        show_convergence: Whether to show convergence plot
        show_error: Whether to show error plot
        style: Seaborn style theme ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette: Color palette to use
        point_size: Size of points in visualization
        dpi: Dots per inch for figure resolution
        show_legend: Whether to show legend
        grid_alpha: Transparency of grid lines
        background_color: Background color for the plots
    """

    figsize: Tuple[int, int] = (15, 8)
    animation_interval: int = 500
    show_convergence: bool = True
    show_error: bool = True
    style: str = "darkgrid"
    context: str = "notebook"
    palette: str = "husl"  # Can be 'husl', 'hls', 'viridis', 'magma', etc.
    point_size: int = 100
    dpi: int = 100
    show_legend: bool = True
    grid_alpha: float = 0.3
    background_color: str = "#2E3440"  # Nord theme dark background


class RootFindingVisualizer:
    """Visualizer for comparing root finding methods."""

    def __init__(
        self,
        problem: RootFindingProblem,
        methods: List[RootFindingMethod],
        config: Optional[VisualizationConfig] = None,
    ):
        self.problem = problem
        self.methods = methods
        self.config = config or VisualizationConfig()

        # Set the style
        sns.set_style(self.config.style)
        sns.set_context(self.config.context)

        # Enable interactive mode
        plt.ion()

        # Initialize figure and axes with dark theme
        self.fig = plt.figure(
            figsize=self.config.figsize,
            dpi=self.config.dpi,
            facecolor=self.config.background_color,
        )

        self.setup_plots()

        # Store method histories
        self.histories: Dict[str, List[float]] = {method.name: [] for method in methods}
        self.errors: Dict[str, List[float]] = {method.name: [] for method in methods}

        # Get color palette
        n_colors = len(methods)
        if self.config.palette in plt.colormaps:
            colors = plt.colormaps[self.config.palette]
            # Fix division by zero when only one method
            if n_colors == 1:
                color_list = [colors(0.5)]  # Use middle color for single method
            else:
                color_list = [colors(i / (n_colors - 1)) for i in range(n_colors)]
        else:
            color_list = sns.color_palette(self.config.palette, n_colors=n_colors)

        # Initialize method states
        self.method_states = {}
        for i, method in enumerate(self.methods):
            color = color_list[i]
            (line,) = self.ax_func.plot(
                [], [], "o-", color=color, label=method.name, linewidth=2, markersize=8
            )
            self.method_states[method.name] = {
                "method": method,
                "line": line,
                "color": color,
            }

        if self.config.show_legend:
            self.ax_func.legend()

        # Initialize convergence and error lines
        if self.ax_conv:
            self.conv_lines = {
                name: self.ax_conv.plot([], [], label=name, color=state["color"])[0]
                for name, state in self.method_states.items()
            }

        if self.ax_error:
            self.error_lines = {
                name: self.ax_error.plot([], [], label=name, color=state["color"])[0]
                for name, state in self.method_states.items()
            }

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def setup_plots(self):
        """Setup the subplot layout with enhanced styling."""
        if self.config.show_convergence and self.config.show_error:
            gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
            self.ax_func = self.fig.add_subplot(gs[0, :])
            self.ax_conv = self.fig.add_subplot(gs[1, 0])
            self.ax_error = self.fig.add_subplot(gs[1, 1])
        else:
            self.ax_func = self.fig.add_subplot(111)
            self.ax_conv = None
            self.ax_error = None

        # Style all axes
        for ax in [self.ax_func, self.ax_conv, self.ax_error]:
            if ax is not None:
                ax.set_facecolor(self.config.background_color)
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")

        # Plot the function with gradient
        x = np.linspace(*self.problem.x_range, 1000)
        y = [self.problem.func(xi) for xi in x]

        # Create custom colormap for function plot
        colors = ["#81A1C1", "#88C0D0", "#8FBCBB"]  # Nord theme blues
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        # Plot function with gradient
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(y), max(y))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.array(y))
        self.ax_func.add_collection(lc)

        # Add zero line
        self.ax_func.axhline(y=0, color="#BF616A", linestyle="--", alpha=0.5)
        self.ax_func.grid(True, alpha=self.config.grid_alpha, color="gray")
        self.ax_func.set_title("Root Finding Methods Comparison", pad=20)

        # Style convergence plot
        if self.ax_conv:
            self.ax_conv.set_title("Convergence Plot")
            self.ax_conv.set_xlabel("Iteration")
            self.ax_conv.set_ylabel("x value")
            self.ax_conv.grid(True, alpha=self.config.grid_alpha, color="gray")

        # Style error plot
        if self.ax_error:
            self.ax_error.set_title("Error Plot")
            self.ax_error.set_xlabel("Iteration")
            self.ax_error.set_ylabel("|f(x)|")
            self.ax_error.set_yscale("log")
            self.ax_error.grid(True, alpha=self.config.grid_alpha, color="gray")

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

                    # Perform one step
                    x_new = method.step()
                    error = method.get_error()

                    # Update histories
                    self.histories[name].append(x_new)
                    self.errors[name].append(error)

                    # Update function plot
                    state["line"].set_data([x_new], [self.problem.func(x_new)])

                    # Update convergence plot
                    if self.ax_conv:
                        self.conv_lines[name].set_data(
                            range(len(self.histories[name])), self.histories[name]
                        )
                        self.ax_conv.relim()
                        self.ax_conv.autoscale_view()

                    # Update error plot
                    if self.ax_error:
                        self.error_lines[name].set_data(
                            range(len(self.errors[name])), self.errors[name]
                        )
                        self.ax_error.relim()
                        self.ax_error.autoscale_view()

            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.config.animation_interval / 1000)  # Convert to seconds

            iteration += 1

        # Disable interactive mode after completion
        plt.ioff()
        plt.show()


# Example usage showing how to implement a method:
class NewtonMethod:
    """Example implementation of Newton's method."""

    def __init__(
        self,
        func: Function,
        derivative: Callable[[float], float],
        x0: float,
        tolerance: float = 1e-6,
    ):
        self.func = func
        self.derivative = derivative
        self.x = x0
        self.tolerance = tolerance
        self._converged = False

    def step(self) -> float:
        if not self._converged:
            self.x = self.x - self.func(self.x) / self.derivative(self.x)
            if abs(self.func(self.x)) < self.tolerance:
                self._converged = True
        return self.x

    def get_error(self) -> float:
        return abs(self.func(self.x))

    def has_converged(self) -> bool:
        return self._converged

    @property
    def name(self) -> str:
        return "Newton's Method"


if __name__ == "__main__":
    # Example usage with the new interface
    def f(x):
        return x**3 - 2 * x - 5

    def df(x):
        return 3 * x**2 - 2

    problem = RootFindingProblem(
        func=f,
        x_range=(-3, 3),
        tolerance=1e-6,
        max_iter=50,
    )

    # Configure visualization with beautiful styling
    config = VisualizationConfig(
        figsize=(12, 8),
        animation_interval=100,  # Faster updates
        show_convergence=True,
        show_error=True,
        style="darkgrid",
        context="notebook",
        palette="magma",  # Changed from colormap to palette
        point_size=100,
        dpi=120,
        background_color="#2E3440",  # Nord theme dark background
    )

    # Create method instances
    methods = [
        NewtonMethod(f, df, x0=2.0),
        # Add other method implementations here
    ]

    # Create and run visualizer
    visualizer = RootFindingVisualizer(problem, methods, config)
    visualizer.run_comparison()
