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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig


# Configuration for visualization options like figure size, animation speed, styles, etc.
@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    figsize: Tuple[int, int] = (15, 8)  # Default figure size
    animation_interval: int = 1  # Reduced from 50 to 1ms
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

        # Enhanced styling setup
        plt.style.use("seaborn-v0_8-white")
        sns.set_context("notebook", font_scale=1.2)

        # Custom style configurations
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.2,
                "axes.linewidth": 1.5,
                "axes.labelsize": 12,
                "lines.linewidth": 2.5,
                "lines.markersize": 8,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "legend.framealpha": 0.9,
                "legend.edgecolor": "0.8",
                "figure.titlesize": 16,
                "figure.titleweight": "bold",
                "figure.constrained_layout.use": True,  # Use constrained_layout instead
            }
        )

        # Enable interactive mode for live updating
        plt.ion()

        # Create main figure with enhanced size and DPI
        self.fig = plt.figure(
            figsize=self.config.figsize,
            dpi=120,  # Higher DPI for sharper rendering
            constrained_layout=True,  # Use constrained_layout here
        )

        self.is_2d = self._check_if_2d()
        self.setup_plots()
        self._initialize_visualization()

    def _initialize_visualization(self):
        """Initialize visualization state with enhanced styling."""
        # Get colors from a more vibrant palette
        n_colors = len(self.methods)
        self.colors = sns.color_palette("husl", n_colors)  # More vibrant colors

        self.method_states = {}
        self.error_lines = {}

        for i, method in enumerate(self.methods):
            method_id = id(method)
            color = self.colors[i]
            name = method.__class__.__name__

            # Enhanced line styling
            line_style = dict(
                linewidth=2.5,
                markersize=8,
                markeredgewidth=2,
                markeredgecolor="white",
                alpha=0.9,
                zorder=5,
            )

            if hasattr(self, "is_3d_parameter") and self.is_3d_parameter:
                # For 3D parameter case, create three lines for parameter evolution
                param_lines = []
                param_names = ["param_1", "param_2", "param_3"]
                param_styles = ["-", "--", ":"]
                for param_name, style in zip(param_names, param_styles):
                    line = self.ax_main.plot(
                        [],
                        [],
                        style,
                        color=color,
                        label=f"{name} {param_name}",
                        **line_style,
                    )[0]
                    param_lines.append(line)

                # Create line for model fit
                fit_line = self.ax_contour.plot(
                    [],
                    [],
                    "-",
                    color=color,
                    label=f"{name} Fit",
                    alpha=0.7,
                    linewidth=2,
                )[0]

                main_lines = param_lines
                contour_line = fit_line
            elif self.is_2d:
                # Original 2D case
                main_line = self.ax_main.plot(
                    [], [], [], "o-", color=color, label=name, **line_style
                )[0]
                contour_line = (
                    self.ax_contour.plot([], [], "o-", color=color, **line_style)[0]
                    if self.config.show_contour
                    else None
                )
                main_lines = [main_line]
            else:
                # Original 1D case
                main_line = self.ax_main.plot(
                    [], [], "o-", color=color, label=name, **line_style
                )[0]
                contour_line = None
                main_lines = [main_line]

            # Enhanced convergence metrics styling
            grad_norm_line = self.ax_convergence.plot(
                [],
                [],
                "-",
                color=color,
                label=f"{name} (Grad Norm)",
                linewidth=2.5,
                alpha=0.9,
            )[0]
            differences_line = self.ax_convergence.plot(
                [],
                [],
                "--",
                color=color,
                label=f"{name} (Step Size)",
                linewidth=2.5,
                alpha=0.7,
            )[0]

            self.method_states[method_id] = {
                "method": method,
                "lines": main_lines,
                "contour_line": contour_line,
                "color": color,
                "points": [],
                "errors": [],
            }
            self.error_lines[method_id] = {
                "grad_norm": grad_norm_line,
                "differences": differences_line,
            }

        # Enhanced legend styling
        if self.config.show_legend:
            legend_style = dict(
                framealpha=0.9, edgecolor="0.8", fancybox=True, shadow=True
            )
            if hasattr(self, "is_3d_parameter") and self.is_3d_parameter:
                self.ax_main.legend(loc="upper right", **legend_style)
            self.ax_convergence.legend(loc="upper right", **legend_style)

        # Enhanced convergence plot styling
        self.ax_convergence.set_xlabel("Iteration")
        self.ax_convergence.set_ylabel("Convergence Metrics")
        self.ax_convergence.grid(True, alpha=0.2)
        self.ax_convergence.set_yscale("log")
        self.ax_convergence.set_xlim(-1, self.problem.max_iter)
        self.ax_convergence.set_ylim(1e-8, 1e2)

        # Add tolerance line
        self.ax_convergence.axhline(
            y=self.problem.tol,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Tolerance: {self.problem.tol:.0e}",
        )

        # Plot function
        self.plot_function()

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
        """Check if the optimization problem is 2D or 3D parameter case."""
        # First check if it's a 3D parameter optimization problem
        if hasattr(self.problem.func, "is_3d"):
            self.is_3d_parameter = True
            return False

        try:
            # Try calling with a 2D point
            test_point = np.array([0.0, 0.0])
            self.problem.func(test_point)
            # If it works and x0 is 2D, then it's a 2D problem
            return len(self.methods[0].get_current_x()) == 2
        except:
            # If 2D fails, it's likely 1D
            self.is_3d_parameter = False
            return False

    def setup_plots(self):
        """Setup the subplot layout based on problem dimensionality."""
        # Create 2x2 GridSpec with equal sizing for all cases
        gs = GridSpec(2, 2, figure=self.fig)

        if hasattr(self, "is_3d_parameter") and self.is_3d_parameter:
            # Top left: Parameter evolution plot
            self.ax_main = self.fig.add_subplot(gs[0, 0])
            self.ax_main.set_title("Parameter Evolution", fontsize=10, pad=5)
            self.ax_main.set_xlabel("Iteration", fontsize=10)
            self.ax_main.set_ylabel("Parameter Value", fontsize=10)

            # Top right: Function value progress
            self.ax_progress = self.fig.add_subplot(gs[0, 1])
            self.ax_progress.set_title("Function Value Progress", fontsize=10, pad=5)
            self.ax_progress.set_xlabel("Iteration", fontsize=10)
            self.ax_progress.set_ylabel("$f(x_k)$", fontsize=10)

            # Bottom left: Model fit plot
            self.ax_contour = self.fig.add_subplot(gs[1, 0])
            self.ax_contour.set_title("Model Fit", fontsize=10, pad=5)
            self.ax_contour.set_xlabel("Input", fontsize=10)
            self.ax_contour.set_ylabel("Output", fontsize=10)

            # Bottom right: Convergence metrics
            self.ax_convergence = self.fig.add_subplot(gs[1, 1])
            self.ax_convergence.set_title("Convergence Metrics", fontsize=10, pad=5)
            self.ax_convergence.set_xlabel("Iteration", fontsize=10)
            self.ax_convergence.set_ylabel("Metric Value", fontsize=10)
            self.ax_convergence.set_yscale("log")

        elif self.is_2d:
            # Original 2D setup
            self.ax_main = self.fig.add_subplot(gs[0, 0], projection="3d")
            self.ax_main.set_title("Surface Plot", fontsize=10, pad=5)

            self.ax_progress = self.fig.add_subplot(gs[0, 1])
            self.ax_progress.set_title("Function Value Progress", fontsize=10, pad=5)

            self.ax_contour = self.fig.add_subplot(gs[1, 0])
            self.ax_contour.set_title("Gradient Field & Contours", fontsize=10, pad=5)

            self.ax_convergence = self.fig.add_subplot(gs[1, 1])
            self.ax_convergence.set_title("Convergence Metrics", fontsize=10, pad=5)
        else:
            # Original 1D setup
            gs = GridSpec(2, 1, height_ratios=[2, 1], figure=self.fig)
            self.ax_main = self.fig.add_subplot(gs[0])
            self.ax_convergence = self.fig.add_subplot(gs[1])

        # Common styling for all axes
        for ax in [self.ax_main, self.ax_progress, self.ax_convergence]:
            if hasattr(ax, "grid"):  # 3D axes don't have grid
                ax.grid(True, linestyle="--", alpha=0.2)

        if hasattr(self, "ax_contour"):
            self.ax_contour.grid(True, linestyle="--", alpha=0.2)

    def plot_function(self):
        """Plot the objective function landscape."""
        if hasattr(self, "is_3d_parameter") and self.is_3d_parameter:
            self._plot_3d_parameter_optimization()
        elif self.is_2d:
            x = np.linspace(*self.problem.x_range, 200)
            self._plot_2d_function(x)
        else:
            self._plot_1d_function()

    def _plot_1d_function(self):
        """Plot 1D optimization landscape with enhanced styling."""
        x_min, x_max = self.problem.x_range
        x_range = x_max - x_min
        x_plot_min = x_min - 0.2 * x_range
        x_plot_max = x_max + 0.2 * x_range
        x = np.linspace(x_plot_min, x_plot_max, 1000)

        # Compute function values
        y = np.array([self.problem.func(np.array([xi])) for xi in x])

        # Create custom color gradient
        colors = ["#FF6B6B", "#4ECDC4"]  # Modern color scheme
        n_colors = 256
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        # Plot the function with gradient fill
        self.ax_main.fill_between(x, y, alpha=0.2, color=colors[0], zorder=1)
        self.function_line = self.ax_main.plot(
            x,
            y,
            color=colors[1],
            alpha=0.8,
            label="f(x)",
            zorder=2,
            linewidth=2.5,
            path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()],
        )[0]

        # Enhanced styling
        self.ax_main.grid(True, linestyle="--", alpha=0.3)
        self.ax_main.set_xlabel("x", fontsize=12, fontweight="bold")
        self.ax_main.set_ylabel("f(x)", fontsize=12, fontweight="bold")

        # Add subtle background color
        self.ax_main.set_facecolor("#f8f9fa")

        # Fix scientific notation and scaling
        self.ax_main.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")

        # Set view limits with padding
        y_min, y_max = np.min(y), np.max(y)
        y_range = max(y_max - y_min, 1e-10)
        y_plot_min = y_min - 0.1 * y_range
        y_plot_max = y_max + 0.1 * y_range

        self.ax_main.set_xlim(x_plot_min, x_plot_max)
        self.ax_main.set_ylim(y_plot_min, y_plot_max)

        # Add optimal point annotation if known
        if hasattr(self.problem, "optimal_value"):
            opt_x = self.problem.optimal_point[0]
            opt_y = self.problem.optimal_value
            self.ax_main.plot(opt_x, opt_y, "r*", markersize=15, label="Global Minimum")
            self.ax_main.annotate(
                f"Minimum: ({opt_x:.2f}, {opt_y:.2f})",
                (opt_x, opt_y),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    def _plot_2d_function(self, x: np.ndarray):
        """Plot 2D objective function with enhanced professional visualization."""
        y = x
        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                [self.problem.func(np.array([xi, yi])) for xi, yi in zip(x_row, y_row)]
                for x_row, y_row in zip(X, Y)
            ]
        )

        # Calculate gradients for quiver plot
        dx = np.zeros_like(Z)
        dy = np.zeros_like(Z)
        grad_magnitude = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                grad = self.problem.derivative(np.array([X[i, j], Y[i, j]]))
                dx[i, j] = grad[0]
                dy[i, j] = grad[1]
                grad_magnitude[i, j] = np.linalg.norm(grad)

        # Create professional colormap with perceptually uniform gradients
        colors = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"]  # Blues
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

        # 1. Enhanced 3D surface plot (top left)
        self.surface = self.ax_main.plot_surface(
            X,
            Y,
            Z,
            cmap=custom_cmap,
            alpha=0.9,
            antialiased=True,
            linewidth=0.5,
            rcount=200,
            ccount=200,
        )
        self.ax_main.set_title("Surface Plot", fontsize=10, pad=5)

        # Add colorbar for surface plot
        colorbar = self.fig.colorbar(
            self.surface,
            ax=self.ax_main,
            pad=0.1,
            format=mtick.ScalarFormatter(useMathText=True),
        )
        colorbar.set_label("Function Value", fontsize=10)

        # 2. Path Progress Plot (top right)
        self.ax_progress.set_title("Function Value Progress", fontsize=10, pad=5)
        self.ax_progress.set_xlabel("Step", fontsize=10)
        self.ax_progress.set_ylabel("$f(x_k)$", fontsize=10)  # Mathematical notation
        self.ax_progress.grid(True, linestyle="--", alpha=0.2)
        self.ax_progress.set_yscale("log")

        # Add reference lines for better context
        if hasattr(self.problem, "optimal_value"):
            opt_val = self.problem.optimal_value
            self.ax_progress.axhline(
                y=opt_val,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Global Minimum: {opt_val:.2e}",
            )

        # 3. Enhanced Gradient Field (bottom left)
        if self.config.show_contour:
            # Add filled contours with better color scheme
            levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 20)
            contourf = self.ax_contour.contourf(
                X,
                Y,
                Z,
                levels=levels,
                cmap="Blues",
                alpha=0.2,
            )

            # Add contour lines
            contours = self.ax_contour.contour(
                X,
                Y,
                Z,
                levels=levels,
                colors="k",
                alpha=0.3,
                linewidths=0.5,
            )

            # Add gradient field with better visibility
            skip = 8
            magnitude_norm = plt.Normalize(grad_magnitude.min(), grad_magnitude.max())
            quiver = self.ax_contour.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                dx[::skip, ::skip],
                dy[::skip, ::skip],
                grad_magnitude[::skip, ::skip],
                cmap="Reds",
                alpha=0.8,
                norm=magnitude_norm,
                scale=50,
                width=0.003,
            )

            self.ax_contour.set_title("Gradient Field & Contours", fontsize=10, pad=5)
            self.ax_contour.set_xlabel("$x_1$", fontsize=10)
            self.ax_contour.set_ylabel("$x_2$", fontsize=10)
            self.ax_contour.set_aspect("equal")

            # Add colorbar for gradient magnitude
            quiver_cbar = self.fig.colorbar(
                quiver,
                ax=self.ax_contour,
                format=mtick.ScalarFormatter(useMathText=True),
            )
            quiver_cbar.set_label("Gradient Magnitude", fontsize=8)

        # 4. Error plot (bottom right)
        self.ax_convergence.set_title("Gradient Norm Convergence", fontsize=10, pad=5)
        self.ax_convergence.set_xlabel("Iteration", fontsize=10)
        self.ax_convergence.set_ylabel(
            "$\\| \\nabla f(x_k) \\|$", fontsize=10
        )  # Gradient norm notation
        self.ax_convergence.grid(True, linestyle="--", alpha=0.2)
        self.ax_convergence.set_yscale("log")

        # Add reference line for tolerance
        self.ax_convergence.axhline(
            y=self.problem.tol,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Tolerance: {self.problem.tol:.0e}",
        )

        # Common styling
        for ax in [self.ax_progress, self.ax_contour]:
            ax.grid(True, linestyle="--", alpha=0.2)

    def _plot_3d_parameter_optimization(self):
        """Plot setup for 3D parameter optimization problems."""
        # Initialize parameter evolution plot with proper ranges
        self.ax_main.set_xlim(-1, 10)  # Will be updated during animation
        self.ax_main.set_ylim(0, 200)  # Adjust based on parameter ranges
        self.ax_main.grid(True, alpha=0.2)

        # Initialize progress plot
        self.ax_progress.set_yscale("log")
        self.ax_progress.set_ylim(1e-1, 1e4)
        self.ax_progress.set_xlim(-1, 10)  # Will be updated during animation
        self.ax_progress.grid(True, alpha=0.2)

        # Initialize model fit plot
        self.ax_contour.grid(True, alpha=0.2)
        if hasattr(self.problem.func, "data"):
            data = self.problem.func.data
            self.ax_contour.scatter(
                data[:, 0],
                data[:, 1],
                color="red",
                label="Observed Data",
                zorder=5,
                alpha=0.7,
            )
            self.ax_contour.set_xscale("log")
            self.ax_contour.set_xlim(min(data[:, 0]) * 0.5, max(data[:, 0]) * 2)
            self.ax_contour.set_ylim(0, max(data[:, 1]) * 1.2)
        self.ax_contour.legend()

        # Initialize convergence plot
        self.ax_convergence.set_yscale("log")
        self.ax_convergence.set_ylim(1e-10, 1e2)
        self.ax_convergence.set_xlim(-1, 10)  # Will be updated during animation
        self.ax_convergence.grid(True, alpha=0.2)
        self.ax_convergence.legend()

    def plot_3d_parameter_comparison(self):
        """Plot comparison for 3D parameter optimization."""
        # Get any observed data points if available
        data = None
        if hasattr(self.problem.func, "data"):
            data = self.problem.func.data

        # Pre-compute data for all methods
        animation_data = {}
        max_iters = 0
        for method_id, state in self.method_states.items():
            history = state["method"].get_iteration_history()

            # Extract parameter values, function values, and gradients
            params = np.array([iter_data.x_new for iter_data in history])
            f_values = np.array([iter_data.f_new for iter_data in history])
            gradients = np.array([iter_data.error for iter_data in history])

            # Calculate consecutive iterate differences
            differences = np.array(
                [
                    np.linalg.norm(params[i] - params[i - 1])
                    for i in range(1, len(params))
                ]
            )

            animation_data[method_id] = {
                "params": params,
                "f_values": f_values,
                "gradients": gradients,
                "differences": differences,
                "converged_at": (
                    len(history) if state["method"].has_converged() else None
                ),
            }
            max_iters = max(max_iters, len(params))

        # Update x-axis limits based on actual number of iterations
        self.ax_main.set_xlim(-1, max_iters + 2)
        self.ax_progress.set_xlim(-1, max_iters + 2)
        self.ax_convergence.set_xlim(-1, max_iters + 2)

        # Set up progress lines dictionary to avoid recreating lines each frame
        progress_lines = {}
        for method_id, state in self.method_states.items():
            (progress_lines[method_id],) = self.ax_progress.plot(
                [], [], color=state["color"], label=state["method"].__class__.__name__
            )
        self.ax_progress.legend()

        # Initialize model fit plot with data points
        if hasattr(self.problem.func, "data"):
            data = self.problem.func.data
            # Plot data points but don't add to legend yet
            self.ax_contour.scatter(
                data[:, 0], data[:, 1], color="red", zorder=5, alpha=0.7
            )
            self.ax_contour.set_xscale("log")
            self.ax_contour.set_xlim(min(data[:, 0]) * 0.5, max(data[:, 0]) * 2)
            self.ax_contour.set_ylim(0, max(data[:, 1]) * 1.2)
            self.ax_contour.grid(True, alpha=0.2)

            # Create a proxy artist for the data points
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    label="Observed Data",
                    markersize=8,
                    alpha=0.7,
                )
            ]

            # Add method fit lines to legend elements
            for state in self.method_states.values():
                legend_elements.append(state["contour_line"])

            # Create single legend with all elements
            self.ax_contour.legend(handles=legend_elements)

        def update(frame):
            frame_idx = min(frame, max_iters - 1)
            lines_to_update = []

            for method_id, state in self.method_states.items():
                data = animation_data[method_id]

                # If method has converged and we're past convergence point, skip updates
                if (
                    data["converged_at"] is not None
                    and frame_idx >= data["converged_at"]
                ):
                    continue

                if frame_idx >= len(data["params"]):
                    continue

                # Update parameter evolution lines
                for i, line in enumerate(state["lines"]):
                    x_data = range(frame_idx + 1)
                    y_data = data["params"][: frame_idx + 1, i]
                    line.set_data(x_data, y_data)
                    lines_to_update.append(line)

                # Update progress plot
                progress_lines[method_id].set_data(
                    range(frame_idx + 1), data["f_values"][: frame_idx + 1]
                )
                lines_to_update.append(progress_lines[method_id])

                # Update model fit plot
                if frame_idx > 0 and hasattr(self.problem.func, "get_fit"):
                    x_fit, y_fit = self.problem.func.get_fit(data["params"][frame_idx])
                    state["contour_line"].set_data(x_fit, y_fit)
                    lines_to_update.append(state["contour_line"])

                # Update convergence metrics
                grad_norms = [
                    np.linalg.norm(g) for g in data["gradients"][: frame_idx + 1]
                ]
                self.error_lines[method_id]["grad_norm"].set_data(
                    range(frame_idx + 1), grad_norms
                )
                lines_to_update.append(self.error_lines[method_id]["grad_norm"])

                if frame_idx > 0:
                    self.error_lines[method_id]["differences"].set_data(
                        range(1, frame_idx + 1), data["differences"][:frame_idx]
                    )
                    lines_to_update.append(self.error_lines[method_id]["differences"])

            return lines_to_update

        # Create animation
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            self.fig,
            update,
            frames=max_iters + 1,  # One frame per iteration
            interval=200,  # Slower animation for better visibility
            blit=True,
            repeat=False,  # Don't repeat the animation
        )

        # Make sure all plots are visible
        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure plots are rendered
        plt.show(block=True)

    def run_comparison(self):
        """Run and visualize the optimization methods with smooth animation."""
        if hasattr(self, "is_3d_parameter") and self.is_3d_parameter:
            self.plot_3d_parameter_comparison()
            return

        # Pre-compute ALL data needed for visualization
        animation_data = {}
        max_iters = 0
        for method_id, state in self.method_states.items():
            history = state["method"].get_iteration_history()
            points = np.array([iter_data.x_new for iter_data in history])
            values = np.array([iter_data.f_new for iter_data in history])
            errors = np.array(
                [float(np.linalg.norm(iter_data.error)) for iter_data in history]
            )

            # Calculate consecutive iterate differences
            differences = np.array(
                [
                    np.linalg.norm(points[i] - points[i - 1])
                    for i in range(1, len(points))
                ]
            )

            animation_data[method_id] = {
                "points": points,
                "values": values,
                "errors": errors,
                "differences": differences,
            }
            max_iters = max(max_iters, len(points))

        # Configure function value progress plot
        self.ax_progress.clear()
        self.ax_progress.set_title("Function Value Progress", fontsize=10, pad=5)
        self.ax_progress.set_xlabel("Iteration", fontsize=10)
        self.ax_progress.set_ylabel("$f(x_k)$", fontsize=10)
        self.ax_progress.grid(True, linestyle="--", alpha=0.2)
        self.ax_progress.set_yscale("log")

        # Compute better y-limits for progress plot
        all_values = np.concatenate(
            [data["values"] for data in animation_data.values()]
        )
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        if min_val > 0:  # For log scale, ensure positive values
            self.ax_progress.set_ylim(min_val * 0.1, max_val * 2)
        else:
            self.ax_progress.set_ylim(1e-15, max_val * 2)
        self.ax_progress.set_xlim(-2, max_iters + 2)

        # Configure combined convergence plot
        self.ax_convergence.clear()
        self.ax_convergence.set_title("Convergence Metrics", fontsize=10, pad=5)
        self.ax_convergence.set_xlabel("Iteration", fontsize=10)
        self.ax_convergence.grid(True, linestyle="--", alpha=0.2)
        self.ax_convergence.set_yscale("log")

        # Add tolerance line
        self.ax_convergence.axhline(
            y=self.problem.tol,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Tolerance: {self.problem.tol:.0e}",
        )

        # Set y-axis limits for convergence plot
        all_errors = np.concatenate(
            [data["errors"] for data in animation_data.values()]
        )
        all_differences = np.concatenate(
            [data["differences"] for data in animation_data.values()]
        )
        min_val = min(np.min(all_errors), np.min(all_differences))
        max_val = max(np.max(all_errors), np.max(all_differences))
        self.ax_convergence.set_ylim(min_val * 0.1, max_val * 2)
        self.ax_convergence.set_xlim(-2, max_iters + 2)

        # Create line objects for animation
        lines = {}
        for method_id, state in self.method_states.items():
            color = state["color"]
            label = state["method"].__class__.__name__

            if self.is_2d:
                lines[method_id] = {
                    "3d": self.ax_main.plot(
                        [],
                        [],
                        [],
                        "o-",
                        color=color,
                        label=label,
                        alpha=0.7,
                        markersize=4,
                        linewidth=1.5,
                    )[0],
                    "contour": self.ax_contour.plot(
                        [],
                        [],
                        "o-",
                        color=color,
                        alpha=0.7,
                        markersize=4,
                        linewidth=1.5,
                    )[0],
                    "progress": self.ax_progress.plot(
                        [], [], "-", color=color, label=label, linewidth=2
                    )[0],
                    "grad_norm": self.ax_convergence.plot(
                        [],
                        [],
                        "-",
                        color=color,
                        label=f"{label} (Grad Norm)",
                        linewidth=2,
                    )[0],
                    "differences": self.ax_convergence.plot(
                        [],
                        [],
                        "--",
                        color=color,
                        label=f"{label} (Step Size)",
                        linewidth=2,
                        alpha=0.7,
                    )[0],
                }

        # Add legends with better positioning
        if self.config.show_legend:
            self.ax_progress.legend(loc="upper right", bbox_to_anchor=(1, 1))
            self.ax_convergence.legend(loc="upper right", bbox_to_anchor=(1, 1))

        def update(frame):
            points_to_show = int((frame / 60) * max_iters)

            for method_id, data in animation_data.items():
                n_points = min(points_to_show, len(data["points"]))
                if n_points == 0:
                    continue

                method_lines = lines[method_id]
                if self.is_2d:
                    # Update 3D and contour lines
                    method_lines["3d"].set_data(
                        data["points"][:n_points, 0], data["points"][:n_points, 1]
                    )
                    method_lines["3d"].set_3d_properties(data["values"][:n_points])

                    method_lines["contour"].set_data(
                        data["points"][:n_points, 0], data["points"][:n_points, 1]
                    )

                    # Update progress line
                    method_lines["progress"].set_data(
                        range(n_points), data["values"][:n_points]
                    )

                    # Update convergence metrics
                    method_lines["grad_norm"].set_data(
                        range(n_points), data["errors"][:n_points]
                    )
                    if n_points > 1:  # Need at least 2 points for differences
                        method_lines["differences"].set_data(
                            range(1, n_points), data["differences"][: n_points - 1]
                        )

            return [
                line
                for method_lines in lines.values()
                for line in method_lines.values()
            ]

        # Create animation with adjusted timing
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            self.fig,
            update,
            frames=60,
            interval=100,  # Increased interval for smoother animation
            blit=True,
            repeat=False,
        )

        plt.show(block=True)

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
                self.ax_convergence.relim()
                self.ax_convergence.autoscale_view()
