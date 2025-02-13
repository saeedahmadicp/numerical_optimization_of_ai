# plot/function.py

"""Utility for plotting mathematical functions with various styles and options."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
from enum import Enum
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PlotStyle(Enum):
    """Available plotting styles."""

    NORMAL = "normal"  # Regular 2D plot
    FILLED = "filled"  # Plot with filled area below
    GRADIENT = "gradient"  # Plot with gradient coloring
    SCATTER = "scatter"  # Scatter plot of points
    STEM = "stem"  # Stem plot


def latex_function(latex_str):
    """Decorator to add LaTeX representation to a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.latex_str = latex_str
        return wrapper

    return decorator


def piecewise_function(pieces, latex_str):
    """Decorator for piecewise functions.

    Args:
        pieces: List of tuples (condition_func, value_func)
        latex_str: LaTeX representation of the piecewise function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(x):
            # Convert to numpy array if not already
            x_arr = np.asarray(x)
            result = np.zeros_like(x_arr, dtype=float)

            # Apply each piece
            for condition_func, value_func in pieces:
                mask = condition_func(x_arr)
                result[mask] = value_func(x_arr[mask])

            return result

        wrapper.latex_str = latex_str
        return wrapper

    return decorator


@dataclass
class FunctionPlotConfig:
    """Configuration for function plotting.

    Attributes:
        figsize: Figure size in inches (width, height)
        style: Plot style to use
        color: Line/fill color
        alpha: Transparency for fills
        grid: Whether to show grid
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        points: Number of points to plot
        linewidth: Width of the plotted line
        show_zeros: Whether to mark x-axis crossings
        show_extrema: Whether to mark local extrema
        dpi: Dots per inch for figure
    """

    figsize: Tuple[int, int] = (10, 6)
    style: PlotStyle = PlotStyle.NORMAL
    color: str = "blue"
    alpha: float = 0.3
    grid: bool = True
    title: str = "Function Plot"
    xlabel: str = "x"
    ylabel: str = "f(x)"
    points: int = 1000
    linewidth: float = 2.0
    show_zeros: bool = False
    show_extrema: bool = False
    dpi: int = 100


class FunctionPlotter:
    """Class for plotting mathematical functions."""

    def __init__(self, config: Optional[FunctionPlotConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or FunctionPlotConfig()

    def plot(
        self,
        func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        x_range: Tuple[float, float],
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot a function over the specified range.

        Args:
            func: Function to plot
            x_range: Tuple of (x_min, x_max)
            ax: Optional matplotlib axes to plot on

        Returns:
            tuple: (figure, axes) of the plot
        """
        # Create figure and axes if not provided
        if ax is None:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = plt.gca()
        else:
            fig = ax.figure

        # Generate x values
        x = np.linspace(x_range[0], x_range[1], self.config.points)
        y = func(x)

        # Plot based on style
        if self.config.style == PlotStyle.NORMAL:
            ax.plot(x, y, color=self.config.color, linewidth=self.config.linewidth)

        elif self.config.style == PlotStyle.FILLED:
            ax.fill_between(x, y, 0, color=self.config.color, alpha=self.config.alpha)
            ax.plot(x, y, color=self.config.color, linewidth=self.config.linewidth)

        elif self.config.style == PlotStyle.GRADIENT:
            points = ax.scatter(x, y, c=y, cmap="viridis", s=self.config.linewidth * 20)
            fig.colorbar(points, ax=ax)

        elif self.config.style == PlotStyle.SCATTER:
            ax.scatter(
                x,
                y,
                color=self.config.color,
                s=self.config.linewidth * 20,
                alpha=self.config.alpha,
            )

        elif self.config.style == PlotStyle.STEM:
            # Create stem plot
            markerline, stemlines, baseline = ax.stem(
                x, y, linefmt="-", markerfmt="o", basefmt="k-", label=self.config.title
            )

            # Set colors for all components
            plt.setp(markerline, color=self.config.color)  # Markers
            plt.setp(stemlines, color=self.config.color)  # Vertical lines
            plt.setp(baseline, color="black")  # Baseline

        # Add optional features
        if self.config.show_zeros:
            self._mark_zeros(ax, x, y)

        if self.config.show_extrema:
            self._mark_extrema(ax, x, y)

        # Customize plot
        if self.config.grid:
            ax.grid(True, alpha=0.3)

        ax.set_title(self.config.title)
        ax.set_xlabel(self.config.xlabel)
        ax.set_ylabel(self.config.ylabel)

        return fig, ax

    def _mark_zeros(self, ax: Axes, x: np.ndarray, y: np.ndarray):
        """Mark x-axis crossings."""
        # Find where y changes sign
        zero_crossings = np.where(np.diff(np.signbit(y)))[0]
        for idx in zero_crossings:
            # Linear interpolation to find more accurate zero
            x_zero = x[idx] - y[idx] * (x[idx + 1] - x[idx]) / (y[idx + 1] - y[idx])
            ax.plot(x_zero, 0, "ro", label="Zero" if idx == zero_crossings[0] else "")

    def _mark_extrema(self, ax: Axes, x: np.ndarray, y: np.ndarray):
        """Mark local extrema."""
        # Find local maxima and minima
        dy = np.diff(y)
        maxima = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
        minima = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1

        # Plot maxima
        if len(maxima) > 0:
            ax.plot(x[maxima], y[maxima], "g^", label="Local Max")

        # Plot minima
        if len(minima) > 0:
            ax.plot(x[minima], y[minima], "rv", label="Local Min")


class PiecewiseFunction:
    """Class for defining piecewise functions."""

    def __init__(self, pieces, latex_str):
        """
        Args:
            pieces: List of tuples (condition_func, value_func)
            latex_str: LaTeX representation of the function
        """
        self.pieces = pieces
        self.latex_str = latex_str

    def __call__(self, x):
        x_arr = np.asarray(x)
        result = np.zeros_like(x_arr, dtype=float)

        for condition_func, value_func in self.pieces:
            mask = condition_func(x_arr)
            result[mask] = value_func(x_arr[mask])

        return result


# Example usage:
if __name__ == "__main__":
    # Step 1: Define the piecewise function
    f = PiecewiseFunction(
        pieces=[
            (lambda x: x != 0, lambda x: x**2 * np.sin(1 / x) + 2 * x**2),
            (lambda x: x == 0, lambda x: 0),
        ],
        # Simplified LaTeX formatting
        latex_str=r"x^2 \sin\left(\frac{1}{x}\right) + 2x^2 \text{ if } x \neq 0, \quad 0 \text{ if } x = 0",
    )

    # Step 2: Configure the plot
    config = FunctionPlotConfig(
        style=PlotStyle.NORMAL,
        figsize=(12, 8),
        color="blue",
        grid=True,
        title=r"$f(x) = " + f.latex_str + "$",  # Simple title formatting
        points=2000,  # More points for better resolution
        linewidth=1.5,
        show_zeros=True,
        show_extrema=True,
    )

    # Step 3: Create plotter and show
    plotter = FunctionPlotter(config)
    fig, ax = plotter.plot(f, x_range=(-1, 1))
    plt.legend()
    plt.show()
