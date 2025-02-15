# plot/convexity.py

"""Visualization utilities for convex functions and their properties.

This module provides a flexible framework for plotting convex functions and visualizing
their various properties including inequalities, first/second order conditions, and
function compositions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any, Protocol, TypeVar

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Type variables for generic functions
T = TypeVar("T")
S = TypeVar("S")


class Function(Protocol[T, S]):
    """Protocol for callable functions with type hints."""

    def __call__(self, x: T) -> S: ...


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior.

    Attributes:
        figsize: Figure size in inches (width, height)
        points: Number of points for smooth curve plotting
        alpha: Transparency value for fill colors
        test_points: Points to use for convexity testing
        grid: Whether to show grid lines
        latex_font: Font size for LaTeX annotations
        dpi: Dots per inch for figure resolution
        style: Matplotlib style sheet to use
    """

    figsize: Tuple[int, int] = (10, 6)
    points: int = 100
    alpha: float = 0.5
    test_points: List[float] = (-1, 0, 1)
    grid: bool = True
    latex_font: int = 10
    dpi: int = 100
    style: str = "default"

    def __post_init__(self):
        """Apply plot style settings after initialization."""
        plt.style.use(self.style)


class PlotBase(ABC):
    """Abstract base class for plotting functionality.

    This class provides common functionality for all plotting classes,
    including plot setup and configuration management.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize with optional configuration.

        Args:
            config: Plot configuration settings
        """
        self.config = config or PlotConfig()

    @abstractmethod
    def plot(self) -> Figure:
        """Generate the plot.

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        pass

    def _setup_plot(
        self, title: str, subplot_spec: Optional[Any] = None
    ) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """Setup basic plot with common decorations.

        Args:
            title: Plot title
            subplot_spec: Subplot specification for complex layouts

        Returns:
            tuple: (figure, axes) where axes might be single or multiple
        """
        if subplot_spec:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(subplot_spec)
        else:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = plt.gca()

        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.set_title(title)

        return fig, ax

    def _add_latex_annotation(
        self,
        ax: Axes,
        text: str,
        position: Tuple[float, float],
        is_annotation: bool = False,
        **kwargs,
    ) -> None:
        """Add LaTeX annotation to plot.

        Args:
            ax: Axes to add annotation to
            text: LaTeX text to add
            position: Position (x, y) for annotation
            is_annotation: Whether to use annotate instead of text
            **kwargs: Additional arguments for annotation/text
        """
        default_kwargs = {
            "fontsize": self.config.latex_font,
            "bbox": dict(facecolor="white", alpha=0.8),
        }

        if is_annotation:
            # For annotate method
            default_kwargs.update(
                {
                    "xytext": kwargs.pop("xytext", (10, 10)),
                    "textcoords": kwargs.pop("textcoords", "offset points"),
                }
            )
            kwargs = {**default_kwargs, **kwargs}
            ax.annotate(text, position, **kwargs)
        else:
            # For text method
            kwargs = {**default_kwargs, **kwargs}
            ax.text(*position, text, **kwargs)


class ConvexInequalityPlot(PlotBase):
    """Visualizes the convex inequality: f(αx+(1-α)y) ≤ αf(x)+(1-α)f(y)."""

    def __init__(
        self,
        f: Function[float, float],
        x: float,
        y: float,
        config: Optional[PlotConfig] = None,
    ):
        """
        Args:
            f: Convex function to plot
            x: First x-coordinate
            y: Second x-coordinate
            config: Plot configuration
        """
        super().__init__(config)
        self.f = f
        self.x = x
        self.y = y

    def plot(self) -> Figure:
        """Generate the convex inequality visualization."""
        # Setup plot
        fig, ax = self._setup_plot("Visualization of Convex Function Inequality")

        # Generate points for smooth curve
        x_range = np.linspace(
            min(self.x, self.y) - 0.5, max(self.x, self.y) + 0.5, self.config.points
        )
        y_range = [self.f(xi) for xi in x_range]

        # Calculate convex combination point
        z = self.config.alpha * self.x + (1 - self.config.alpha) * self.y

        # Calculate function values
        fx = self.f(self.x)
        fy = self.f(self.y)
        fz = self.f(z)
        f_linear = self.config.alpha * fx + (1 - self.config.alpha) * fy

        # Plot function curve and line segment
        ax.plot(x_range, y_range, "b-", label="f(x)")
        ax.plot([self.x, self.y], [fx, fy], "r--", label="Linear interpolation")

        # Plot points
        ax.plot(self.x, fx, "ro")
        ax.plot(self.y, fy, "ro")
        ax.plot(z, fz, "go")
        ax.plot(z, f_linear, "mo")

        # Add vertical line showing difference
        ax.vlines(z, fz, f_linear, colors="g", linestyles="dotted")

        # Add annotations
        self._add_point_annotations(ax, z, fx, fy, fz, f_linear)

        # Customize plot
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(["f(x)", "Linear interpolation"])

        return fig

    def _add_point_annotations(
        self, ax: Axes, z: float, fx: float, fy: float, fz: float, f_linear: float
    ) -> None:
        """Add all annotations to the plot."""
        # Point annotations
        self._add_latex_annotation(ax, f"$(x, f(x))$", (self.x, fx), is_annotation=True)
        self._add_latex_annotation(ax, f"$(y, f(y))$", (self.y, fy), is_annotation=True)

        # x-axis labels
        self._add_latex_annotation(
            ax, "$x$", (self.x, 0), is_annotation=True, xytext=(0, -20), ha="center"
        )
        self._add_latex_annotation(
            ax, "$y$", (self.y, 0), is_annotation=True, xytext=(0, -20), ha="center"
        )

        # Middle point annotations
        self._add_latex_annotation(
            ax,
            r"$\alpha x + (1-\alpha)y$",
            (z, 0),
            is_annotation=True,
            xytext=(0, -20),
            ha="center",
        )

        # Function value annotations
        self._add_latex_annotation(
            ax, r"$f(\alpha x + (1-\alpha)y)$", (z, fz), is_annotation=True
        )
        self._add_latex_annotation(
            ax, r"$\alpha f(x) + (1-\alpha)f(y)$", (z, f_linear), is_annotation=True
        )


class FirstOrderConvexityPlot(PlotBase):
    """Visualizes the first-order condition for convexity: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)."""

    def __init__(
        self,
        f: Function[float, float],
        df: Function[float, float],
        x: float,
        y: float,
        config: Optional[PlotConfig] = None,
    ):
        """
        Args:
            f: Convex function to plot
            df: Derivative (gradient) of f
            x: First x-coordinate
            y: Second x-coordinate
            config: Plot configuration
        """
        super().__init__(config)
        self.f = f
        self.df = df
        self.x = x
        self.y = y

    def plot(self) -> Figure:
        """Generate the first-order condition visualization."""
        # Setup plot
        fig, ax = self._setup_plot("First-Order Condition for Convexity")

        # Generate points for smooth curve
        x_range = np.linspace(
            min(self.x, self.y) - 0.5, max(self.x, self.y) + 0.5, self.config.points
        )

        # Calculate function values and derivatives
        y_range = [self.f(xi) for xi in x_range]
        fx, fy = self.f(self.x), self.f(self.y)
        dfx, dfy = self.df(self.x), self.df(self.y)

        # Create and plot tangent lines
        tangent_x = [self._tangent_line(t, self.x, fx, dfx) for t in x_range]
        tangent_y = [self._tangent_line(t, self.y, fy, dfy) for t in x_range]

        # Plot function and tangent lines
        ax.plot(x_range, y_range, "b-", label="f(x)")
        ax.plot(x_range, tangent_x, "g--", label="Tangent at x")
        ax.plot(x_range, tangent_y, "r--", label="Tangent at y")

        # Plot points
        ax.plot(self.x, fx, "go")
        ax.plot(self.y, fy, "ro")

        # Add annotations
        self._add_annotations(ax, fx, fy)

        # Customize plot
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        return fig

    def _tangent_line(self, t: float, x0: float, fx0: float, dfx0: float) -> float:
        """Calculate tangent line value at point t."""
        return fx0 + dfx0 * (t - x0)

    def _add_annotations(self, ax: Axes, fx: float, fy: float) -> None:
        """Add all annotations to the plot."""
        # Point annotations
        self._add_latex_annotation(ax, "$(x, f(x))$", (self.x, fx), is_annotation=True)
        self._add_latex_annotation(ax, "$(y, f(y))$", (self.y, fy), is_annotation=True)

        # Gradient vectors annotations
        self._add_latex_annotation(
            ax,
            r"$\nabla f(x)$",
            (self.x, fx),
            is_annotation=True,
            xytext=(30, -20),
            arrowprops=dict(arrowstyle="->"),
        )
        self._add_latex_annotation(
            ax,
            r"$\nabla f(y)$",
            (self.y, fy),
            is_annotation=True,
            xytext=(30, -20),
            arrowprops=dict(arrowstyle="->"),
        )

        # First order condition inequality
        self._add_latex_annotation(
            ax,
            r"$f(y) \geq f(x) + \nabla f(x)^T(y-x)$",
            (0.05, 0.95),
            transform=ax.transAxes,
        )


class SecondOrderConvexityPlot(PlotBase):
    """Visualizes the second-order condition: g''(t) = dᵀ∇²f(x + td)d ≥ 0."""

    def __init__(
        self,
        f: Function[np.ndarray, float],
        df: Function[np.ndarray, np.ndarray],
        d2f: Function[np.ndarray, np.ndarray],
        x: np.ndarray,
        directions: Optional[List[np.ndarray]] = None,
        t_range: Tuple[float, float] = (-1, 1),
        config: Optional[PlotConfig] = None,
    ):
        """
        Args:
            f: Convex function to plot (takes vector input)
            df: Gradient of f (takes vector input)
            d2f: Hessian of f (takes vector input)
            x: Base point
            directions: List of direction vectors to plot
            t_range: Range for parameter t
            config: Plot configuration
        """
        super().__init__(config)
        self.f = f
        self.df = df
        self.d2f = d2f
        self.x = x
        self.t_range = t_range
        self.directions = directions or [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

    def plot(self) -> Figure:
        """Generate the second-order condition visualization."""
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0], projection="3d")  # 3D plot for f(x)
        ax2 = fig.add_subplot(gs[0, 1])  # 2D plot for g(t)
        ax3 = fig.add_subplot(gs[0, 2])  # 2D plot for g''(t)

        self._plot_3d_surface(ax1)
        self._plot_directional_slices(ax2)
        self._plot_second_derivatives(ax3)

        plt.tight_layout()
        return fig

    def _plot_3d_surface(self, ax: Axes) -> None:
        """Plot 3D surface of f(x) with direction vectors."""
        # Create grid for 3D plot
        x1 = np.linspace(self.x[0] - 1, self.x[0] + 1, self.config.points)
        x2 = np.linspace(self.x[1] - 1, self.x[1] + 1, self.config.points)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        # Compute function values
        for i in range(self.config.points):
            for j in range(self.config.points):
                Z[i, j] = self.f(np.array([X1[i, j], X2[i, j]]))

        # Plot surface
        ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)

        # Plot base point
        ax.scatter(
            [self.x[0]],
            [self.x[1]],
            [self.f(self.x)],
            color="red",
            s=100,
            label="Base point x",
        )

        # Plot direction vectors
        t = np.linspace(self.t_range[0], self.t_range[1], self.config.points)
        colors = ["b", "g", "r"]

        for d, color in zip(self.directions, colors):
            direction_points = np.array([self.x + ti * d for ti in t])
            z_points = np.array([self.f(p) for p in direction_points])
            ax.plot(
                direction_points[:, 0],
                direction_points[:, 1],
                z_points,
                color=color,
                label=f"d = [{d[0]:.1f}, {d[1]:.1f}]",
            )

        ax.set_title("f(x)")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("f(x)")
        ax.legend()

    def _plot_directional_slices(self, ax: Axes) -> None:
        """Plot g(t) = f(x + td) for each direction."""
        t = np.linspace(self.t_range[0], self.t_range[1], self.config.points)
        colors = ["b", "g", "r"]

        for d, color in zip(self.directions, colors):
            g_t = np.array([self.f(self.x + ti * d) for ti in t])
            ax.plot(t, g_t, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]")

        ax.plot(0, self.f(self.x), "ko")
        ax.grid(True)
        ax.legend()
        ax.set_title("g(t) = f(x + td)")
        ax.set_xlabel("t")
        ax.set_ylabel("g(t)")

    def _plot_second_derivatives(self, ax: Axes) -> None:
        """Plot g''(t) = dᵀ∇²f(x + td)d for each direction."""
        t = np.linspace(self.t_range[0], self.t_range[1], self.config.points)
        colors = ["b", "g", "r"]

        for d, color in zip(self.directions, colors):
            g_double_prime = np.array([d.T @ self.d2f(self.x + ti * d) @ d for ti in t])
            ax.plot(
                t, g_double_prime, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]"
            )

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True)
        ax.legend()
        ax.set_title(r"g′′(t) = dᵀ∇²f(x + td)d ≥ 0")
        ax.set_xlabel("t")
        ax.set_ylabel("g′′(t)")


class ConvexitySlicePlot(PlotBase):
    """Visualizes how g(t) = f(x + td) relates to f(x) and shows its convexity near t=0."""

    def __init__(
        self,
        f: Function[np.ndarray, float],
        x: np.ndarray,
        directions: Optional[List[np.ndarray]] = None,
        t_range: Tuple[float, float] = (-1, 1),
        config: Optional[PlotConfig] = None,
    ):
        """
        Args:
            f: Convex function to plot (takes vector input)
            x: Base point
            directions: List of direction vectors to plot
            t_range: Range for parameter t
            config: Plot configuration
        """
        super().__init__(config)
        self.f = f
        self.x = x
        self.t_range = t_range
        self.directions = directions or [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

    def plot(self) -> Figure:
        """Generate the convexity slice visualization."""
        # Create figure with 2 subplots side by side
        fig = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1])

        # Create subplots
        ax1 = fig.add_subplot(gs[0], projection="3d")  # 3D plot for f(x)
        ax2 = fig.add_subplot(gs[1])  # 2D plot for g(t)

        self._plot_3d_surface(ax1)
        self._plot_directional_slices(ax2)

        # Add explanation text
        self._add_latex_annotation(
            ax2,
            r"For each direction d, g(t) = f(x + td) is convex near t = 0"
            + "\nPoints show g(λt₁ + (1-λ)t₂) ≤ λg(t₁) + (1-λ)g(t₂)",
            (0.05, 0.95),
            transform=ax2.transAxes,
        )

        plt.tight_layout()
        return fig

    def _plot_3d_surface(self, ax: Axes) -> None:
        """Plot 3D surface of f(x) with direction vectors."""
        # Create grid for 3D plot
        x1 = np.linspace(self.x[0] - 2, self.x[0] + 2, self.config.points)
        x2 = np.linspace(self.x[1] - 2, self.x[1] + 2, self.config.points)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        # Compute function values
        for i in range(self.config.points):
            for j in range(self.config.points):
                Z[i, j] = self.f(np.array([X1[i, j], X2[i, j]]))

        # Plot surface
        ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)

        # Plot base point
        ax.scatter(
            [self.x[0]],
            [self.x[1]],
            [self.f(self.x)],
            color="red",
            s=100,
            label="Base point x",
        )

        # Plot direction vectors and slices
        t = np.linspace(self.t_range[0], self.t_range[1], self.config.points)
        colors = ["b", "g", "r"]

        for d, color in zip(self.directions, colors):
            direction_points = np.array([self.x + ti * d for ti in t])
            z_points = np.array([self.f(p) for p in direction_points])
            ax.plot(
                direction_points[:, 0],
                direction_points[:, 1],
                z_points,
                color=color,
                linewidth=2,
                label=f"d = [{d[0]:.1f}, {d[1]:.1f}]",
            )

        ax.set_title("f(x) with directional slices")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("f(x)")
        ax.legend()

    def _plot_directional_slices(self, ax: Axes) -> None:
        """Plot g(t) = f(x + td) for each direction with convexity test points."""
        t = np.linspace(self.t_range[0], self.t_range[1], self.config.points)
        colors = ["b", "g", "r"]

        for d, color in zip(self.directions, colors):
            # Plot g(t)
            g_t = np.array([self.f(self.x + ti * d) for ti in t])
            ax.plot(t, g_t, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]")

            # Add test points to show convexity
            t_points = [-0.5, 0, 0.5]
            g_points = [self.f(self.x + tp * d) for tp in t_points]
            ax.plot(t_points, g_points, f"{color}o")

            # Add line segment connecting test points
            ax.plot([-0.5, 0.5], [g_points[0], g_points[2]], f"{color}--", alpha=0.5)

        # Add vertical lines
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=-0.5, color="k", linestyle=":", alpha=0.2)
        ax.axvline(x=0.5, color="k", linestyle=":", alpha=0.2)

        ax.grid(True)
        ax.legend()
        ax.set_title("g(t) = f(x + td)")
        ax.set_xlabel("t")
        ax.set_ylabel("g(t)")


class CompositionPlot(PlotBase):
    """Visualizes convexity preservation under function composition f(g(x))."""

    def __init__(
        self,
        f: Function[float, float],
        g: Function[float, float],
        x_range: Tuple[float, float] = (-2, 2),
        config: Optional[PlotConfig] = None,
    ):
        """
        Args:
            f: Outer function
            g: Inner function
            x_range: Range for x-axis
            config: Plot configuration
        """
        super().__init__(config)
        self.f = f
        self.g = g
        self.x_range = x_range

    def plot(self) -> Figure:
        """Generate the composition visualization."""
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Generate x values
        x = np.linspace(self.x_range[0], self.x_range[1], self.config.points)

        # Compute function values
        g_x = np.array([self.g(xi) for xi in x])
        fg_x = np.array([self.f(self.g(xi)) for xi in x])

        self._plot_inner_function(ax1, x, g_x)
        self._plot_outer_function(ax2, g_x)
        self._plot_composition(ax3, x, fg_x)

        plt.tight_layout()
        return fig

    def _plot_inner_function(self, ax: Axes, x: np.ndarray, g_x: np.ndarray) -> None:
        """Plot inner function g(x) with convexity test."""
        ax.plot(x, g_x, "b-", label="g(x)")
        self._add_convexity_test(ax, x, self.g)

        ax.grid(True)
        ax.legend()
        ax.set_title("Inner function g(x)")
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")

    def _plot_outer_function(self, ax: Axes, g_x: np.ndarray) -> None:
        """Plot outer function f(y)."""
        y_range = np.linspace(min(g_x) - 0.5, max(g_x) + 0.5, self.config.points)
        f_y = np.array([self.f(yi) for yi in y_range])

        ax.plot(y_range, f_y, "r-", label="f(y)")
        ax.grid(True)
        ax.legend()
        ax.set_title("Outer function f(y)")
        ax.set_xlabel("y")
        ax.set_ylabel("f(y)")

    def _plot_composition(self, ax: Axes, x: np.ndarray, fg_x: np.ndarray) -> None:
        """Plot composition f(g(x)) with convexity test."""
        ax.plot(x, fg_x, "g-", label="f(g(x))")
        self._add_convexity_test(ax, x, lambda x: self.f(self.g(x)))

        ax.grid(True)
        ax.legend()
        ax.set_title("Composition f(g(x))")
        ax.set_xlabel("x")
        ax.set_ylabel("f(g(x))")

    def _add_convexity_test(
        self, ax: Axes, x: np.ndarray, func: Function[float, float]
    ) -> None:
        """Add convexity test points and interpolation."""
        # Test points
        test_points = [-1, 0, 1]
        y_vals = [func(x) for x in test_points]

        # Compute midpoint and interpolation
        alpha = 0.5
        mid_x = alpha * test_points[0] + (1 - alpha) * test_points[2]
        mid_y = func(mid_x)
        interp_y = alpha * y_vals[0] + (1 - alpha) * y_vals[2]

        # Plot test points and interpolation
        ax.plot(test_points, y_vals, "ko", label="Test points")
        ax.plot(mid_x, mid_y, "go", label="Actual")
        ax.plot(mid_x, interp_y, "ro", label="Interpolation")
        ax.plot(
            [test_points[0], test_points[2]], [y_vals[0], y_vals[2]], "r--", alpha=0.5
        )


class CompositionExamplesPlot(PlotBase):
    """Visualizes examples of convex and non-convex function compositions."""

    def plot(self) -> Figure:
        """Generate the composition examples visualization."""
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Example 1: f(t) = t², g(x) = x² (convex case)
        f1 = lambda t: t**2  # convex and non-decreasing for t ≥ 0
        g1 = lambda x: x**2  # convex
        fg1 = lambda x: (x**2) ** 2  # composition is convex

        # Example 2: f(t) = -t, g(x) = x² (non-convex case)
        f2 = lambda t: -t  # affine but decreasing
        g2 = lambda x: x**2  # convex
        fg2 = lambda x: -(x**2)  # composition is concave

        self._plot_example(axes[0], f1, g1, fg1, "Convex Case")
        self._plot_example(axes[1], f2, g2, fg2, "Non-convex Case")

        # Add explanation text
        self._add_latex_annotation(
            axes[0, 0],
            "Example 1: f₁(g₁(x)) is convex because f₁ is convex & non-decreasing and g₁ is convex",
            (0.05, 0.95),
            transform=axes[0, 0].transAxes,
        )
        self._plot_example(axes[1, 0], f2, g2, fg2, "Non-convex Case")
        self._add_latex_annotation(
            axes[1, 0],
            "Example 2: f₂(g₂(x)) is not convex because f₂ is decreasing",
            (0.05, 0.95),
            transform=axes[1, 0].transAxes,
        )

        plt.tight_layout()
        return fig

    def _plot_example(self, axes: np.ndarray, f, g, fg, title: str) -> None:
        """Plot one row of composition example."""
        x = np.linspace(-2, 2, self.config.points)

        # Plot g(x)
        axes[0].plot(x, [g(xi) for xi in x], "b-")
        axes[0].set_title(f"g(x) = x²\n({title})")
        axes[0].grid(True)

        # Plot f(y)
        y = np.linspace(-2, 2, self.config.points)
        axes[1].plot(y, [f(yi) for yi in y], "r-")
        axes[1].set_title(f"f(t)\n({title})")
        axes[1].grid(True)

        # Plot f(g(x))
        axes[2].plot(x, [fg(xi) for xi in x], "g-")
        axes[2].set_title(f"f(g(x))\n({title})")
        axes[2].grid(True)


class ConvexityPlotter:
    """A class for visualizing various aspects of convex functions."""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()

    def plot_convex_inequality(
        self, f: Function[float, float], x: float, y: float
    ) -> Figure:
        """Plot convex inequality visualization."""
        plotter = ConvexInequalityPlot(f, x, y, self.config)
        return plotter.plot()

    def plot_first_order_convexity(
        self, f: Function[float, float], df: Function[float, float], x: float, y: float
    ) -> Figure:
        """Plot first-order condition visualization."""
        plotter = FirstOrderConvexityPlot(f, df, x, y, self.config)
        return plotter.plot()

    def plot_second_order_convexity(
        self,
        f: Function[np.ndarray, float],
        df: Function[np.ndarray, np.ndarray],
        d2f: Function[np.ndarray, np.ndarray],
        x: np.ndarray,
        directions: Optional[List[np.ndarray]] = None,
        t_range: Tuple[float, float] = (-1, 1),
    ) -> Figure:
        """Plot second-order condition visualization."""
        plotter = SecondOrderConvexityPlot(
            f, df, d2f, x, directions, t_range, self.config
        )
        return plotter.plot()

    def plot_convexity_slice(
        self,
        f: Function[np.ndarray, float],
        x: np.ndarray,
        directions: Optional[List[np.ndarray]] = None,
        t_range: Tuple[float, float] = (-1, 1),
    ) -> Figure:
        """Plot convexity slice visualization."""
        plotter = ConvexitySlicePlot(f, x, directions, t_range, self.config)
        return plotter.plot()

    def plot_composition_convexity(
        self,
        f: Function[float, float],
        g: Function[float, float],
        x_range: Tuple[float, float] = (-2, 2),
    ) -> Figure:
        """Plot composition convexity visualization."""
        plotter = CompositionPlot(f, g, x_range, self.config)
        return plotter.plot()

    def plot_composition_examples(
        self, x_range: Tuple[float, float] = (-2, 2)
    ) -> Figure:
        """Plot composition examples visualization."""
        plotter = CompositionExamplesPlot(self.config)
        return plotter.plot()


# # Example usage:
# if __name__ == "__main__":
#     # Configure plotting parameters
#     config = PlotConfig(
#         figsize=(12, 8),
#         points=200,
#         alpha=0.3,
#         test_points=[-1, 0, 1],
#         grid=True,
#         latex_font=12,
#     )

#     # Create plotter instance with configuration
#     plotter = ConvexityPlotter(config)

#     # Example with quadratic function
#     f = lambda x: x**2
#     df = lambda x: 2 * x
#     d2f = lambda x: np.array([[2.0]])

#     # Plot various aspects of convexity
#     plotter.plot_convex_inequality(f, x=-1, y=2)
#     plt.show()

#     plotter.plot_first_order_convexity(f, df, x=-1, y=2)
#     plt.show()

#     # Example with composition
#     g = lambda x: x**2
#     plotter.plot_composition_convexity(f, g)
#     plt.show()
