# plot/function_viz.py

"""
Function Visualization Module
----------------------------
This module provides utilities for visualizing mathematical functions in 1D, 2D, and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, List, Optional, Union, Dict, Any
import plotly.graph_objects as go


class FunctionVisualizer:
    """Class for visualizing mathematical functions with various dimensionality."""

    def __init__(self, theme: str = "default"):
        """
        Initialize the function visualizer.

        Args:
            theme: Visual theme for plots ('default', 'dark', 'light', 'seaborn')
        """
        self.theme = theme
        self._setup_theme()

    def _setup_theme(self):
        """Setup the visual theme for plots."""
        if self.theme == "dark":
            plt.style.use("dark_background")
        elif self.theme == "light":
            plt.style.use("default")
        elif self.theme == "seaborn":
            sns.set_theme()

    def plot_1d(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        num_points: int = 1000,
        title: str = "Function Visualization",
        xlabel: str = "x",
        ylabel: str = "f(x)",
        show_grid: bool = True,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        critical_points: Optional[List[float]] = None,
        interactive: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot a 1D function.

        Args:
            func: The function to visualize, should take numpy arrays as input
            x_range: Tuple containing (min_x, max_x)
            num_points: Number of points to evaluate function at
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            show_grid: Whether to show grid lines
            plot_kwargs: Additional keyword arguments for plot function
            critical_points: List of x-values to highlight (e.g., extrema, inflection points)
            interactive: Use plotly for interactive visualization
            ax: Optional matplotlib axes to plot on

        Returns:
            The matplotlib figure object
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = func(x)

        if plot_kwargs is None:
            plot_kwargs = {"linewidth": 2.5, "color": "#1f77b4"}

        if interactive:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=3, color="royalblue"),
                    name="f(x)",
                )
            )

            if critical_points:
                critical_y = [func(np.array([cp]))[0] for cp in critical_points]
                fig.add_trace(
                    go.Scatter(
                        x=critical_points,
                        y=critical_y,
                        mode="markers",
                        marker=dict(size=10, color="red"),
                        name="Critical Points",
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly_white",
                hovermode="closest",
            )

            fig.show()
            return fig

        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.figure

            ax.plot(x, y, **plot_kwargs)

            if critical_points:
                critical_y = [func(np.array([cp]))[0] for cp in critical_points]
                ax.scatter(
                    critical_points,
                    critical_y,
                    color="red",
                    s=50,
                    zorder=5,
                    label="Critical Points",
                )

            ax.set_title(title, fontsize=15)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            if show_grid:
                ax.grid(True, alpha=0.3)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if critical_points:
                ax.legend()

            plt.tight_layout()
            return fig

    def plot_2d(
        self,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_points: int = 100,
        title: str = "2D Function Visualization",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "f(x,y)",
        plot_type: str = "contour",
        colormap: str = "viridis",
        critical_points: Optional[List[Tuple[float, float]]] = None,
        interactive: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot a 2D function.

        Args:
            func: The function to visualize, should take (x, y) numpy arrays as input
            x_range: Tuple containing (min_x, max_x)
            y_range: Tuple containing (min_y, max_y)
            num_points: Number of points to evaluate function at in each dimension
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for z-axis (function values)
            plot_type: Type of plot ('contour', 'surface', 'heatmap')
            colormap: Colormap for the plot
            critical_points: List of (x,y) points to highlight (e.g., extrema)
            interactive: Use plotly for interactive visualization
            ax: Optional matplotlib axes to plot on

        Returns:
            The matplotlib figure object
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        if interactive:
            if plot_type == "surface":
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colormap)])

                if critical_points:
                    cp_x, cp_y = zip(*critical_points)
                    cp_z = [
                        func(np.array([cpx]), np.array([cpy]))[0][0]
                        for cpx, cpy in zip(cp_x, cp_y)
                    ]
                    fig.add_trace(
                        go.Scatter3d(
                            x=cp_x,
                            y=cp_y,
                            z=cp_z,
                            mode="markers",
                            marker=dict(size=5, color="red"),
                            name="Critical Points",
                        )
                    )

                camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        zaxis_title=zlabel,
                        camera=camera,
                    ),
                    template="plotly_white",
                )

            else:  # contour or heatmap
                if plot_type == "contour":
                    fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale=colormap))
                else:  # heatmap
                    fig = go.Figure(data=go.Heatmap(z=Z, x=x, y=y, colorscale=colormap))

                if critical_points:
                    cp_x, cp_y = zip(*critical_points)
                    fig.add_trace(
                        go.Scatter(
                            x=cp_x,
                            y=cp_y,
                            mode="markers",
                            marker=dict(size=10, color="red", symbol="x"),
                            name="Critical Points",
                        )
                    )

                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    template="plotly_white",
                )

            fig.show()
            return fig

        else:
            if plot_type == "surface":
                if ax is None:
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection="3d")
                else:
                    fig = ax.figure

                surf = ax.plot_surface(
                    X, Y, Z, cmap=colormap, alpha=0.8, linewidth=0, antialiased=True
                )

                if critical_points:
                    cp_x, cp_y = zip(*critical_points)
                    cp_z = [
                        func(np.array([cpx]), np.array([cpy]))[0][0]
                        for cpx, cpy in zip(cp_x, cp_y)
                    ]
                    ax.scatter(
                        cp_x, cp_y, cp_z, color="red", s=50, label="Critical Points"
                    )

                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_zlabel(zlabel, fontsize=12)

                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

            else:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10, 8))
                else:
                    fig = ax.figure

                if plot_type == "contour":
                    contour = ax.contourf(X, Y, Z, 20, cmap=colormap)
                    ax.contour(X, Y, Z, 20, colors="k", alpha=0.3, linewidths=0.5)
                else:  # heatmap
                    contour = ax.imshow(
                        Z,
                        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                        aspect="auto",
                        origin="lower",
                        cmap=colormap,
                    )

                if critical_points:
                    cp_x, cp_y = zip(*critical_points)
                    ax.scatter(
                        cp_x,
                        cp_y,
                        color="red",
                        s=50,
                        marker="x",
                        label="Critical Points",
                    )

                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)

                fig.colorbar(contour, ax=ax)

            ax.set_title(title, fontsize=15)

            if critical_points:
                ax.legend()

            plt.tight_layout()
            return fig

    def plot_3d(
        self,
        func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        num_points: int = 20,
        title: str = "3D Function Visualization",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "z",
        colormap: str = "viridis",
        plot_type: str = "vector_field",
        interactive: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Visualize a 3D function.

        Args:
            func: The function to visualize, should take (x, y, z) numpy arrays and return vectors
            x_range: Tuple containing (min_x, max_x)
            y_range: Tuple containing (min_y, max_y)
            z_range: Tuple containing (min_z, max_z)
            num_points: Number of points to evaluate function at in each dimension
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for z-axis
            colormap: Colormap for the plot
            plot_type: Type of plot ('vector_field', 'streamlines', 'isosurface')
            interactive: Use plotly for interactive visualization

        Returns:
            The matplotlib or plotly figure object
        """
        # Use fewer points for 3D visualization to avoid overcrowding
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        z = np.linspace(z_range[0], z_range[1], num_points)

        X, Y, Z = np.meshgrid(x, y, z)
        U, V, W = func(X, Y, Z)

        # Calculate vector magnitudes for color mapping
        magnitude = np.sqrt(U**2 + V**2 + W**2)

        if interactive:
            fig = go.Figure()

            if plot_type == "vector_field":
                # Create a subset of points to make the visualization clearer
                skip = (
                    slice(None, None, 2),
                    slice(None, None, 2),
                    slice(None, None, 2),
                )

                fig.add_trace(
                    go.Cone(
                        x=X[skip].flatten(),
                        y=Y[skip].flatten(),
                        z=Z[skip].flatten(),
                        u=U[skip].flatten(),
                        v=V[skip].flatten(),
                        w=W[skip].flatten(),
                        colorscale=colormap,
                        colorbar=dict(title="Magnitude"),
                        sizemode="absolute",
                        sizeref=0.15,
                    )
                )

                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        zaxis_title=zlabel,
                        aspectratio=dict(x=1, y=1, z=1),
                    ),
                    template="plotly_white",
                )

            fig.show()
            return fig

        else:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            if plot_type == "vector_field":
                # Use a subset of points for clarity
                skip = (
                    slice(None, None, 2),
                    slice(None, None, 2),
                    slice(None, None, 2),
                )

                ax.quiver(
                    X[skip],
                    Y[skip],
                    Z[skip],
                    U[skip],
                    V[skip],
                    W[skip],
                    length=0.1,
                    normalize=True,
                    color=plt.cm.get_cmap(colormap)(magnitude[skip] / magnitude.max()),
                    alpha=0.8,
                )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_zlabel(zlabel, fontsize=12)
            ax.set_title(title, fontsize=15)

            plt.tight_layout()
            return fig

    def plot_optimization_path(
        self,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        path: List[Tuple[float, float]],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_points: int = 100,
        title: str = "Optimization Path",
        xlabel: str = "x",
        ylabel: str = "y",
        zlabel: str = "f(x,y)",
        plot_type: str = "contour",
        colormap: str = "viridis",
        interactive: bool = True,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Visualize the path of an optimization algorithm on a 2D function.

        Args:
            func: The function being optimized
            path: List of (x,y) coordinates representing the optimization path
            x_range: Tuple containing (min_x, max_x)
            y_range: Tuple containing (min_y, max_y)
            num_points: Number of points to evaluate function at in each dimension
            title: Plot title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for z-axis (function values)
            plot_type: Type of plot ('contour', 'surface')
            colormap: Colormap for the plot
            interactive: Use plotly for interactive visualization

        Returns:
            The matplotlib or plotly figure object
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        path_x, path_y = zip(*path)
        path_z = [
            float(func(np.array([px]), np.array([py])))
            for px, py in zip(path_x, path_y)
        ]

        if interactive:
            if plot_type == "surface":
                fig = go.Figure()

                # Add surface plot
                fig.add_trace(
                    go.Surface(z=Z, x=X, y=Y, colorscale=colormap, opacity=0.8)
                )

                # Add optimization path
                fig.add_trace(
                    go.Scatter3d(
                        x=path_x,
                        y=path_y,
                        z=path_z,
                        mode="lines+markers",
                        line=dict(color="red", width=4),
                        marker=dict(
                            size=5,
                            color=list(range(len(path_x))),
                            colorscale="Plasma",
                            symbol="circle",
                        ),
                        name="Optimization Path",
                    )
                )

                # Add start and end points
                fig.add_trace(
                    go.Scatter3d(
                        x=[path_x[0]],
                        y=[path_y[0]],
                        z=[path_z[0]],
                        mode="markers",
                        marker=dict(size=8, color="green"),
                        name="Start Point",
                    )
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[path_x[-1]],
                        y=[path_y[-1]],
                        z=[path_z[-1]],
                        mode="markers",
                        marker=dict(size=8, color="red"),
                        name="End Point",
                    )
                )

                camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        zaxis_title=zlabel,
                        camera=camera,
                    ),
                    template="plotly_white",
                )

            else:  # contour
                fig = go.Figure()

                # Add contour plot
                fig.add_trace(
                    go.Contour(
                        z=Z,
                        x=x,
                        y=y,
                        colorscale=colormap,
                        contours=dict(
                            showlabels=True, labelfont=dict(size=12, color="white")
                        ),
                    )
                )

                # Add optimization path
                fig.add_trace(
                    go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines+markers",
                        line=dict(color="red", width=3),
                        marker=dict(
                            size=8,
                            color=list(range(len(path_x))),
                            colorscale="Plasma",
                            symbol="circle",
                        ),
                        name="Optimization Path",
                    )
                )

                # Add start and end points
                fig.add_trace(
                    go.Scatter(
                        x=[path_x[0]],
                        y=[path_y[0]],
                        mode="markers",
                        marker=dict(size=12, color="green", symbol="star"),
                        name="Start Point",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[path_x[-1]],
                        y=[path_y[-1]],
                        mode="markers",
                        marker=dict(size=12, color="red", symbol="star"),
                        name="End Point",
                    )
                )

                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    template="plotly_white",
                )

            fig.show()
            return fig

        else:
            if plot_type == "surface":
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Plot the surface
                surf = ax.plot_surface(
                    X, Y, Z, cmap=colormap, alpha=0.7, linewidth=0, antialiased=True
                )

                # Plot the optimization path
                ax.plot(path_x, path_y, path_z, "r-", linewidth=2, label="Path")
                ax.scatter(
                    path_x, path_y, path_z, c=range(len(path_x)), cmap="plasma", s=40
                )

                # Mark start and end points
                ax.scatter(
                    [path_x[0]],
                    [path_y[0]],
                    [path_z[0]],
                    color="green",
                    s=100,
                    label="Start",
                )
                ax.scatter(
                    [path_x[-1]],
                    [path_y[-1]],
                    [path_z[-1]],
                    color="red",
                    s=100,
                    label="End",
                )

                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_zlabel(zlabel, fontsize=12)
                ax.set_title(title, fontsize=15)
                ax.legend()

                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

            else:  # contour
                fig, ax = plt.subplots(figsize=(10, 8))

                # Plot the contour
                contour = ax.contourf(X, Y, Z, 20, cmap=colormap)
                ax.contour(X, Y, Z, 20, colors="k", alpha=0.3, linewidths=0.5)

                # Plot the optimization path
                ax.plot(path_x, path_y, "r-", linewidth=2, label="Path")
                ax.scatter(path_x, path_y, c=range(len(path_x)), cmap="plasma", s=40)

                # Mark start and end points
                ax.scatter(
                    [path_x[0]], [path_y[0]], color="green", s=100, label="Start"
                )
                ax.scatter([path_x[-1]], [path_y[-1]], color="red", s=100, label="End")

                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=15)
                ax.legend()

                fig.colorbar(contour, ax=ax)

            plt.tight_layout()
            return fig


# # EXAMPLE USAGE - UNCOMMENT TO RUN (OPENS UP BROWSER)
# def example():
#     """Demonstrate usage of the FunctionVisualizer class."""
#     visualizer = FunctionVisualizer(theme="seaborn")

#     # 1D Example
#     def func_1d(x):
#         return np.sin(x) * np.exp(-0.1 * x)

#     fig1 = visualizer.plot_1d(
#         func=func_1d,
#         x_range=(-2 * np.pi, 6 * np.pi),
#         title="Damped Sine Wave",
#         critical_points=[0, np.pi, 2 * np.pi, 3 * np.pi],
#         interactive=True,
#     )

#     # 2D Example
#     def func_2d(x, y):
#         return np.sin(np.sqrt(x**2 + y**2)) / (np.sqrt(x**2 + y**2) + 0.1)

#     fig2 = visualizer.plot_2d(
#         func=func_2d,
#         x_range=(-10, 10),
#         y_range=(-10, 10),
#         plot_type="surface",
#         title="2D Sinc Function",
#         interactive=True,
#     )

#     # 3D Vector Field Example
#     def func_3d(x, y, z):
#         u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
#         v = np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
#         w = np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
#         return u, v, w

#     fig3 = visualizer.plot_3d(
#         func=func_3d,
#         x_range=(-1, 1),
#         y_range=(-1, 1),
#         z_range=(-1, 1),
#         title="3D Vector Field",
#         interactive=True,
#     )

#     # Optimization Path Example
#     def rosenbrock(x, y):
#         return (1 - x) ** 2 + 100 * (y - x**2) ** 2

#     # Simulated optimization path
#     path = [(np.random.uniform(-2, 2), np.random.uniform(-1, 3)) for _ in range(10)]
#     path.sort(key=lambda p: float(rosenbrock(np.array([p[0]]), np.array([p[1]]))))

#     fig4 = visualizer.plot_optimization_path(
#         func=rosenbrock,
#         path=path,
#         x_range=(-2, 2),
#         y_range=(-1, 3),
#         title="Rosenbrock Function Optimization",
#         plot_type="contour",
#         interactive=True,
#     )

#     return fig1, fig2, fig3, fig4


# if __name__ == "__main__":
#     example()
