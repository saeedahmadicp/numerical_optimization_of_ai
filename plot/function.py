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

    NORMAL = "normal"
    FILLED = "filled"
    GRADIENT = "gradient"
    SCATTER = "scatter"
    STEM = "stem"


@dataclass
class FunctionPlotConfig:
    """Configuration for function plotting."""

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


def latex_function(latex_str):
    """Decorator to add LaTeX representation to a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.latex_str = latex_str
        return wrapper

    return decorator


class PiecewiseFunction:
    """Class for defining piecewise functions."""

    def __init__(self, pieces, latex_str):
        self.pieces = pieces
        self.latex_str = latex_str

    def __call__(self, x):
        x_arr = np.asarray(x)
        result = np.zeros_like(x_arr, dtype=float)
        for condition_func, value_func in self.pieces:
            mask = condition_func(x_arr)
            result[mask] = value_func(x_arr[mask])
        return result


class FunctionPlotter:
    """Class for plotting mathematical functions."""

    def __init__(self, config: Optional[FunctionPlotConfig] = None):
        self.config = config or FunctionPlotConfig()

    def plot(
        self,
        func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        x_range: Tuple[float, float],
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot a function over the specified range."""
        if ax is None:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = plt.gca()
        else:
            fig = ax.figure

        x = np.linspace(x_range[0], x_range[1], self.config.points)
        y = func(x)

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
            markerline, stemlines, baseline = ax.stem(
                x, y, linefmt="-", markerfmt="o", basefmt="k-"
            )
            plt.setp(markerline, color=self.config.color)
            plt.setp(stemlines, color=self.config.color)
            plt.setp(baseline, color="black")

        if self.config.show_zeros:
            self._mark_zeros(ax, x, y)
        if self.config.show_extrema:
            self._mark_extrema(ax, x, y)

        if self.config.grid:
            ax.grid(True, alpha=0.3)

        ax.set_title(self.config.title)
        ax.set_xlabel(self.config.xlabel)
        ax.set_ylabel(self.config.ylabel)

        return fig, ax

    def _mark_zeros(self, ax: Axes, x: np.ndarray, y: np.ndarray):
        zero_crossings = np.where(np.diff(np.signbit(y)))[0]
        for idx in zero_crossings:
            x_zero = x[idx] - y[idx] * (x[idx + 1] - x[idx]) / (y[idx + 1] - y[idx])
            ax.plot(x_zero, 0, "ro", label="Zero" if idx == zero_crossings[0] else "")

    def _mark_extrema(self, ax: Axes, x: np.ndarray, y: np.ndarray):
        dy = np.diff(y)
        maxima = np.where((dy[:-1] > 0) & (dy[1:] < 0))[0] + 1
        minima = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1

        if len(maxima) > 0:
            ax.plot(x[maxima], y[maxima], "g^", label="Local Max")
        if len(minima) > 0:
            ax.plot(x[minima], y[minima], "rv", label="Local Min")

    def plot_interactive(
        self,
        func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        x_range: Tuple[float, float],
    ) -> Figure:
        """Create an interactive plot with mouse scroll zoom and drag pan capabilities."""
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        self.plot(func, x_range, ax=ax)

        y = func(np.array(x_range))
        y_min, y_max = min(y), max(y)
        y_padding = (y_max - y_min) * 0.1

        ax.set_xlim(x_range)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        self._pan_data = None

        def zoom_factory(ax, base_scale=1.1):
            def zoom_fun(event):
                if event.inaxes != ax:
                    return
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                xdata = event.xdata
                ydata = event.ydata

                if event.button == "up":
                    scale_factor = 1 / base_scale
                elif event.button == "down":
                    scale_factor = base_scale
                else:
                    scale_factor = 1

                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

                relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

                ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
                ax.set_ylim(
                    [ydata - new_height * (1 - rely), ydata + new_height * rely]
                )
                plt.draw()

            fig.canvas.mpl_connect("scroll_event", zoom_fun)
            return zoom_fun

        def pan_factory(ax):
            def on_mouse_down(event):
                if event.inaxes != ax or event.button != 1:
                    return
                self._pan_data = {
                    "x": event.xdata,
                    "y": event.ydata,
                    "xlim": ax.get_xlim(),
                    "ylim": ax.get_ylim(),
                }
                fig.canvas.get_tk_widget().config(cursor="fleur")

            def on_mouse_up(event):
                if self._pan_data is not None:
                    fig.canvas.get_tk_widget().config(cursor="arrow")
                    self._pan_data = None

            def on_mouse_move(event):
                if (
                    self._pan_data is None
                    or event.inaxes != ax
                    or event.xdata is None
                    or event.ydata is None
                ):
                    return

                dx = event.xdata - self._pan_data["x"]
                dy = event.ydata - self._pan_data["y"]
                xlim = self._pan_data["xlim"]
                ylim = self._pan_data["ylim"]

                ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
                ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
                plt.draw()

            fig.canvas.mpl_connect("button_press_event", on_mouse_down)
            fig.canvas.mpl_connect("button_release_event", on_mouse_up)
            fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

        zoom_factory(ax)
        pan_factory(ax)
        return fig


if __name__ == "__main__":
    f = PiecewiseFunction(
        pieces=[
            (lambda x: x != 0, lambda x: x**2 * np.sin(1 / x) + 2 * x**2),
            (lambda x: x == 0, lambda x: 0),
        ],
        latex_str=r"x^2 \sin\left(\frac{1}{x}\right) + 2x^2 \text{ if } x \neq 0, \quad 0 \text{ if } x = 0",
    )

    config = FunctionPlotConfig(
        style=PlotStyle.NORMAL,
        figsize=(12, 8),
        color="blue",
        grid=True,
        title=r"$f(x) = " + f.latex_str + "$",
        points=2000,
        linewidth=1.5,
        show_zeros=True,
        show_extrema=True,
    )

    plotter = FunctionPlotter(config)
    fig = plotter.plot_interactive(f, x_range=(-1, 1))
    plt.show()
