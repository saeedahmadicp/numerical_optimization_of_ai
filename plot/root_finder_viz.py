# plot/root_finder_viz.py

"""
Root Finding Visualization Module.

This module provides visualization capabilities for root-finding algorithms,
allowing for interactive exploration of algorithmic behavior, convergence
properties, and comparative analysis of different methods.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.

    This dataclass encapsulates all parameters related to visualization appearance,
    layout, and behavior.

    Attributes:
        show_convergence: Whether to show convergence plot
        show_error: Whether to show error plot
        style: Plot style
        context: Plot context
        palette: Color palette for different methods
        point_size: Size of points in scatter plots
        dpi: Dots per inch for saved figures
        show_legend: Whether to display the legend
        grid_alpha: Transparency of grid lines
        title: Main title for the visualization
        background_color: Background color for plots
        animation_duration: Duration for each animation frame (ms)
        animation_transition: Transition time between frames (ms)
    """

    show_convergence: bool = True
    show_error: bool = True
    style: str = "white"
    context: str = "talk"
    palette: str = "viridis"
    point_size: int = 8
    dpi: int = 100
    show_legend: bool = True
    grid_alpha: float = 0.2
    title: str = "Root Finding Methods Comparison"
    background_color: str = "#FFFFFF"
    animation_duration: int = 800  # ms per frame
    animation_transition: int = 300  # ms for transition


class RootFindingVisualizer:
    """
    Visualizer for root-finding algorithms.

    This class provides functionality for visualizing the behavior of different
    root-finding algorithms, with support for 1D, 2D, and 3D visualizations,
    animations of the iterative process, and comparative analysis.
    """

    def __init__(
        self,
        config: NumericalMethodConfig,
        methods: List[BaseNumericalMethod],
        vis_config: VisualizationConfig,
    ):
        """
        Initialize the visualizer with configuration and methods.

        Args:
            config: Configuration for the numerical methods
            methods: List of root-finding methods to visualize
            vis_config: Visualization configuration
        """
        self.config = config
        self.methods = methods
        self.vis_config = vis_config

        # Check if we're dealing with multivariate functions
        self.dimensions = self._detect_dimensions()

        # Generate a list of distinct colors for methods
        self.colors = self._generate_colors(len(methods))

        # Define a list of marker symbols for methods to ensure they don't overlap completely
        self.marker_symbols = [
            "circle",
            "square",
            "diamond",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "pentagon",
            "hexagon",
            "star",
            "hexagram",
            "star-triangle-up",
            "star-triangle-down",
            "star-square",
            "star-diamond",
            "diamond-tall",
            "diamond-wide",
            "hourglass",
            "bowtie",
            "circle-cross",
        ]

        # Ensure we have enough symbols (cycle if needed)
        if len(methods) > len(self.marker_symbols):
            # Repeat the marker symbols if we have more methods than symbols
            repeat_count = (len(methods) // len(self.marker_symbols)) + 1
            self.marker_symbols = self.marker_symbols * repeat_count

        # Define size variation based on method index (between 8 and 14)
        self.marker_sizes = [
            self.vis_config.point_size + (i % 4) * 2 for i in range(len(methods))
        ]

        # Store for computed data
        self.all_data = []
        self.root_estimates = []

    def _detect_dimensions(self) -> int:
        """
        Detect the dimensionality of the problem based on the methods.

        Returns:
            int: Number of dimensions (1, 2, or 3)
        """
        # Try to call the function with a sample point to check dimensionality
        try:
            # Check the first method for its current x
            x0 = self.methods[0].get_current_x()

            if isinstance(x0, np.ndarray):
                return len(x0)
            else:
                return 1
        except:
            # Default to 1D if we can't detect
            return 1

    def _generate_colors(self, n_colors: int) -> List[str]:
        """
        Generate a list of distinct colors for the methods.

        Args:
            n_colors: Number of colors needed

        Returns:
            List[str]: List of color strings
        """
        if n_colors <= 10:
            # Use qualitative colors for better distinction when few methods
            return px.colors.qualitative.D3[:n_colors]
        else:
            # Use a colorscale for many methods
            return [
                px.colors.sample_colorscale(
                    px.colors.sequential.Viridis, i / (n_colors - 1)
                )[0]
                for i in range(n_colors)
            ]

    def _prepare_data(self) -> None:
        """
        Prepare data from the methods for visualization.

        This method extracts iteration history from each method and converts
        it to a format suitable for plotting.
        """
        self.all_data = []
        self.root_estimates = []

        for i, method in enumerate(self.methods):
            # Run the method until convergence if it hasn't been run yet
            if not method.get_iteration_history():
                while not method.has_converged():
                    method.step()

            # Get the iteration history
            history = method.get_iteration_history()

            # Extract the final root estimate
            root = method.get_current_x()
            self.root_estimates.append(root)

            # Convert history to a DataFrame for easier manipulation
            method_data = []
            for iter_data in history:
                data_point = {
                    "Method": method.name,
                    "Iteration": iter_data.iteration,
                    "x_old": iter_data.x_old,
                    "x_new": iter_data.x_new,
                    "f_old": iter_data.f_old,
                    "f_new": iter_data.f_new,
                    "Error": iter_data.error,
                    "Color": self.colors[i],
                }

                # Add method-specific details
                for key, value in iter_data.details.items():
                    if isinstance(value, (int, float, str, bool)):
                        data_point[key] = value

                method_data.append(data_point)

            self.all_data.append(pd.DataFrame(method_data))

    def _create_function_space(self, fig, row=1, col=1) -> None:
        """
        Create the function space visualization based on dimensionality.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        if self.dimensions == 1:
            # 1D function visualization
            x_min, x_max = self.config.x_range
            x = np.linspace(x_min, x_max, 1000)

            # Calculate function values
            try:
                y = [self.config.func(xi) for xi in x]

                # Plot the function
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name="f(x)",
                        line=dict(color="black", width=2.5),
                        legendgroup="function",
                        hoverinfo="text",
                        hovertemplate="x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                # Add a horizontal line at y=0 to show the roots
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[0, 0],
                        mode="lines",
                        name="y=0",
                        line=dict(color="gray", width=1, dash="dash"),
                        legendgroup="function",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Set axis labels with improved styling
                fig.update_xaxes(
                    title_text="x",
                    title_font=dict(size=14),
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    title_text="f(x)",
                    title_font=dict(size=14),
                    row=row,
                    col=col,
                )

            except Exception as e:
                print(f"Error plotting function: {e}")

        elif self.dimensions == 2:
            # 2D function visualization (surface plot or contour)
            x_min, x_max = self.config.x_range
            y_min, y_max = self.config.x_range  # Assuming same range for y

            # Create a grid of points
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Compute function values
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                    except:
                        Z[i, j] = np.nan

            # Create contour plot
            fig.add_trace(
                go.Contour(
                    z=Z,
                    x=x_grid,
                    y=y_grid,
                    colorscale="Viridis",
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color="white"),
                    ),
                    colorbar=dict(
                        title="f(x,y)",
                        thickness=15,
                        len=0.6,
                    ),
                    name="f(x,y)",
                    legendgroup="function",
                ),
                row=row,
                col=col,
            )

            # Add contour line for z=0 to show roots
            fig.add_trace(
                go.Contour(
                    z=Z,
                    x=x_grid,
                    y=y_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        showlabels=False,
                        coloring="lines",
                    ),
                    line=dict(color="white", width=3),
                    showscale=False,
                    name="f(x,y)=0",
                    legendgroup="function",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Set axis labels with improved styling
            fig.update_xaxes(
                title_text="x",
                title_font=dict(size=14),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="y",
                title_font=dict(size=14),
                row=row,
                col=col,
            )

        elif self.dimensions == 3:
            # 3D function - represented as isosurfaces or slices
            # For 3D, we'll create a simpler visualization as it's more complex

            fig.update_layout(
                scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"
                )
            )

            # NOTE: Detailed 3D visualization would be added here
            # For now, we'll add a placeholder message
            fig.add_annotation(
                text="3D visualization not fully implemented",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )

    def _add_initial_points(self, fig, row=1, col=1) -> List[Dict]:
        """
        Add initial points for all methods to the visualization.

        Args:
            fig: The plotly figure to add points to
            row: Row in the subplot grid
            col: Column in the subplot grid

        Returns:
            List of trace indices for initial points
        """
        point_traces = []

        for i, method in enumerate(self.methods):
            if not self.all_data[i].empty:
                # Get initial point
                initial_data = self.all_data[i].iloc[0]

                if self.dimensions == 1:
                    x0 = initial_data["x_old"]
                    y0 = self.config.func(x0)

                    trace = fig.add_trace(
                        go.Scatter(
                            x=[x0],
                            y=[y0],
                            mode="markers",
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i]
                                + 2,  # Slightly larger for initial points
                                symbol=self.marker_symbols[i],
                                line=dict(color="black", width=1),
                            ),
                            name=method.name,
                            hoverinfo="text",
                            hovertext=f"Initial: x={x0:.6f}, f(x)={y0:.6f}",
                            legendgroup=method.name,  # Group legends by method name
                            showlegend=True,  # Show only one legend entry per method
                        ),
                        row=row,
                        col=col,
                    )
                    point_traces.append(trace)

                elif self.dimensions == 2:
                    x0, y0 = initial_data["x_old"]
                    z0 = self.config.func(np.array([x0, y0]))

                    trace = fig.add_trace(
                        go.Scatter(
                            x=[x0],
                            y=[y0],
                            mode="markers",
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i]
                                + 2,  # Slightly larger for initial points
                                symbol=self.marker_symbols[i],
                                line=dict(color="black", width=1),
                            ),
                            name=method.name,
                            hoverinfo="text",
                            hovertext=f"Initial: x={x0:.4f}, y={y0:.4f}, f(x,y)={z0:.6f}",
                            legendgroup=method.name,  # Group legends by method name
                            showlegend=True,  # Show only one legend entry per method
                        ),
                        row=row,
                        col=col,
                    )
                    point_traces.append(trace)

                elif self.dimensions == 3:
                    # For 3D, we'd add a point to the 3D scene
                    # This would be implemented as needed
                    pass

        return point_traces

    def _create_convergence_plot(self, fig, row=1, col=1) -> None:
        """
        Create a plot showing the convergence of x values.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        for i, method in enumerate(self.methods):
            if not self.all_data[i].empty:
                df = self.all_data[i]

                # For 1D functions
                if self.dimensions == 1:
                    fig.add_trace(
                        go.Scatter(
                            x=df["Iteration"],
                            y=df["x_new"],
                            mode="lines+markers",
                            name=method.name,
                            line=dict(color=self.colors[i], width=2),
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i],
                                symbol=self.marker_symbols[i],
                            ),
                            legendgroup=method.name,  # Group legends by method name
                            showlegend=False,  # Don't show duplicate legend entries
                        ),
                        row=row,
                        col=col,
                    )

                # For 2D or 3D functions, we would show the norm of the position vector
                else:
                    # Calculate norms for multidimensional x values
                    norms = []
                    for j in range(len(df)):
                        if isinstance(df["x_new"].iloc[j], np.ndarray):
                            norms.append(np.linalg.norm(df["x_new"].iloc[j]))
                        else:
                            # Handle case where x_new might be stored as a string or other format
                            try:
                                norms.append(float(df["x_new"].iloc[j]))
                            except:
                                norms.append(np.nan)

                    fig.add_trace(
                        go.Scatter(
                            x=df["Iteration"],
                            y=norms,
                            mode="lines+markers",
                            name=f"{method.name} |x|",
                            line=dict(color=self.colors[i], width=2),
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i],
                                symbol=self.marker_symbols[i],
                            ),
                            legendgroup=method.name,  # Group legends by method name
                            showlegend=False,  # Don't show duplicate legend entries
                        ),
                        row=row,
                        col=col,
                    )

        # Set axis labels
        fig.update_xaxes(title_text="Iteration", row=row, col=col)
        if self.dimensions == 1:
            fig.update_yaxes(title_text="x value", row=row, col=col)
        else:
            fig.update_yaxes(title_text="||x||", row=row, col=col)

    def _create_error_plot(self, fig, row=1, col=1) -> None:
        """
        Create a plot showing the error over iterations.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        for i, method in enumerate(self.methods):
            if not self.all_data[i].empty:
                df = self.all_data[i]

                # Plot error on a log scale
                fig.add_trace(
                    go.Scatter(
                        x=df["Iteration"],
                        y=df["Error"],
                        mode="lines+markers",
                        name=method.name,
                        line=dict(color=self.colors[i], width=2),
                        marker=dict(
                            color=self.colors[i],
                            size=self.marker_sizes[i],
                            symbol=self.marker_symbols[i],
                        ),
                        legendgroup=method.name,  # Group legends by method name
                        showlegend=False,  # Don't show duplicate legend entries
                    ),
                    row=row,
                    col=col,
                )

        # Set axis labels and log scale for y-axis
        fig.update_xaxes(title_text="Iteration", row=row, col=col)
        fig.update_yaxes(title_text="Error (|f(x)|)", type="log", row=row, col=col)

    def _create_animation_frames(self) -> Tuple[List[go.Frame], List[Dict]]:
        """
        Create animation frames showing the progression of root-finding.

        Returns:
            Tuple containing animation frames and slider steps
        """
        frames = []
        slider_steps = []

        # Determine the maximum number of iterations across all methods
        max_iterations = max(
            [df["Iteration"].max() if not df.empty else 0 for df in self.all_data]
        )

        # Create frames for each iteration
        for iteration in range(int(max_iterations) + 1):
            frame_data = []

            # First, include the function curve and y=0 line in each frame so they don't disappear
            if self.dimensions == 1:
                # Add the function curve
                x_min, x_max = self.config.x_range
                x = np.linspace(x_min, x_max, 1000)
                try:
                    y = [self.config.func(xi) for xi in x]
                    frame_data.append(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name="f(x)",
                            line=dict(color="black", width=2.5),
                            legendgroup="function",
                            showlegend=True,
                            hoverinfo="text",
                            hovertemplate="x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>",
                            xaxis="x",  # Explicitly set to first x-axis
                            yaxis="y",  # Explicitly set to first y-axis
                        )
                    )

                    # Add y=0 line
                    frame_data.append(
                        go.Scatter(
                            x=[x_min, x_max],
                            y=[0, 0],
                            mode="lines",
                            name="y=0",
                            line=dict(color="gray", width=1, dash="dash"),
                            legendgroup="function",
                            showlegend=False,
                            xaxis="x",  # Explicitly set to first x-axis
                            yaxis="y",  # Explicitly set to first y-axis
                        )
                    )
                except:
                    pass
            elif self.dimensions == 2:
                # For 2D functions, add the contour plot to each frame
                x_min, x_max = self.config.x_range
                y_min, y_max = self.config.x_range  # Assuming same range for y

                # Create a grid of points
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_grid, y_grid)

                # Compute function values
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                        except:
                            Z[i, j] = np.nan

                # Add main contour plot
                frame_data.append(
                    go.Contour(
                        z=Z,
                        x=x_grid,
                        y=y_grid,
                        colorscale="Viridis",
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=12, color="white"),
                        ),
                        colorbar=dict(
                            title="f(x,y)",
                            thickness=15,
                            len=0.6,
                        ),
                        name="f(x,y)",
                        legendgroup="function",
                        xaxis="x",  # Explicitly set to first x-axis
                        yaxis="y",  # Explicitly set to first y-axis
                    )
                )

                # Add contour line for z=0 to show roots
                frame_data.append(
                    go.Contour(
                        z=Z,
                        x=x_grid,
                        y=y_grid,
                        contours=dict(
                            start=0,
                            end=0,
                            showlabels=False,
                            coloring="lines",
                        ),
                        line=dict(color="white", width=3),
                        showscale=False,
                        name="f(x,y)=0",
                        legendgroup="function",
                        showlegend=False,
                        xaxis="x",  # Explicitly set to first x-axis
                        yaxis="y",  # Explicitly set to first y-axis
                    )
                )
            elif self.dimensions == 3:
                # For 3D functions, add the 3D visualization to each frame
                # (code for 3D visualization would need to be included here)
                pass

            # Keep track of which methods are shown in this frame
            methods_in_current_frame = set()

            # Add method-specific points for this iteration
            for i, method in enumerate(self.methods):
                df = self.all_data[i]
                if df.empty:
                    continue

                # Add current point to function plot
                iter_data = df[df["Iteration"] == iteration]

                # If we have data for this iteration
                if not iter_data.empty:
                    methods_in_current_frame.add(i)
                    if self.dimensions == 1:
                        x_val = iter_data["x_new"].iloc[0]
                        y_val = iter_data["f_new"].iloc[0]

                        # Keep the showlegend setting for the first point of each method to maintain the legend
                        frame_data.append(
                            go.Scatter(
                                x=[x_val],
                                y=[y_val],
                                mode="markers",
                                marker=dict(
                                    color=self.colors[i],
                                    size=self.marker_sizes[i],
                                    symbol=self.marker_symbols[i],
                                ),
                                name=method.name,
                                hoverinfo="text",
                                hovertext=f"{method.name}: x={x_val:.6f}, f(x)={y_val:.6f}",
                                legendgroup=method.name,
                                showlegend=True,  # Always show legend for this trace
                                xaxis="x",  # Explicitly set to first x-axis
                                yaxis="y",  # Explicitly set to first y-axis
                            )
                        )

                    elif self.dimensions == 2:
                        try:
                            x_val, y_val = iter_data["x_new"].iloc[0]
                            z_val = self.config.func(np.array([x_val, y_val]))

                            frame_data.append(
                                go.Scatter(
                                    x=[x_val],
                                    y=[y_val],
                                    mode="markers",
                                    marker=dict(
                                        color=self.colors[i],
                                        size=self.marker_sizes[i],
                                        symbol=self.marker_symbols[i],
                                    ),
                                    name=method.name,
                                    hoverinfo="text",
                                    hovertext=f"{method.name}: x={x_val:.4f}, y={y_val:.4f}, f(x,y)={z_val:.6f}",
                                    legendgroup=method.name,
                                    showlegend=True,  # Always show legend for this trace
                                    xaxis="x",  # Explicitly set to first x-axis
                                    yaxis="y",  # Explicitly set to first y-axis
                                )
                            )
                        except:
                            # Handle the case where x_new might not be an array
                            pass
                # If this method has already converged, we still need to show its last point to keep legend entry
                elif iteration > df["Iteration"].max():
                    methods_in_current_frame.add(i)
                    # Get the last data point for this method
                    last_point = df.iloc[-1]

                    if self.dimensions == 1:
                        x_val = last_point["x_new"]
                        y_val = last_point["f_new"]

                        # Add the last point with a special marker to indicate it has converged
                        frame_data.append(
                            go.Scatter(
                                x=[x_val],
                                y=[y_val],
                                mode="markers",
                                marker=dict(
                                    color=self.colors[i],
                                    size=self.marker_sizes[i],
                                    symbol=self.marker_symbols[i],
                                    opacity=0.7,  # Slightly transparent to indicate it's not changing
                                ),
                                name=method.name,
                                hoverinfo="text",
                                hovertext=f"{method.name} (converged): x={x_val:.6f}, f(x)={y_val:.6f}",
                                legendgroup=method.name,
                                showlegend=True,  # Important: always show in legend
                                xaxis="x",  # Explicitly set to first x-axis
                                yaxis="y",  # Explicitly set to first y-axis
                            )
                        )
                    elif self.dimensions == 2:
                        try:
                            x_val, y_val = last_point["x_new"]
                            z_val = self.config.func(np.array([x_val, y_val]))

                            frame_data.append(
                                go.Scatter(
                                    x=[x_val],
                                    y=[y_val],
                                    mode="markers",
                                    marker=dict(
                                        color=self.colors[i],
                                        size=self.marker_sizes[i],
                                        symbol=self.marker_symbols[i],
                                        opacity=0.7,
                                    ),
                                    name=method.name,
                                    hoverinfo="text",
                                    hovertext=f"{method.name} (converged): x={x_val:.4f}, y={y_val:.4f}, f(x,y)={z_val:.6f}",
                                    legendgroup=method.name,
                                    showlegend=True,  # Important: always show in legend
                                    xaxis="x",  # Explicitly set to first x-axis
                                    yaxis="y",  # Explicitly set to first y-axis
                                )
                            )
                        except:
                            pass

                # For convergence and error plots, include only data up to current iteration
                filtered_df = df[df["Iteration"] <= iteration]
                if not filtered_df.empty:
                    # Add convergence plot trace - progressively showing data
                    if self.dimensions == 1:
                        # Convergence plot (row 1, col 2)
                        convergence_trace = go.Scatter(
                            x=filtered_df["Iteration"],
                            y=filtered_df["x_new"],
                            mode="lines+markers",
                            name=method.name,
                            line=dict(color=self.colors[i], width=2),
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i],
                                symbol=self.marker_symbols[i],
                            ),
                            legendgroup=method.name,
                            showlegend=False,  # Keep as False to avoid duplicate legend entries
                            xaxis="x2",  # Reference to second x-axis
                            yaxis="y2",  # Reference to second y-axis
                        )
                        frame_data.append(convergence_trace)
                    else:
                        # For 2D/3D, calculate norms
                        norms = []
                        for j in range(len(filtered_df)):
                            if isinstance(filtered_df["x_new"].iloc[j], np.ndarray):
                                norms.append(
                                    np.linalg.norm(filtered_df["x_new"].iloc[j])
                                )
                            else:
                                try:
                                    norms.append(float(filtered_df["x_new"].iloc[j]))
                                except:
                                    norms.append(np.nan)

                        # Convergence plot (row 1, col 2)
                        convergence_trace = go.Scatter(
                            x=filtered_df["Iteration"],
                            y=norms,
                            mode="lines+markers",
                            name=f"{method.name} |x|",
                            line=dict(color=self.colors[i], width=2),
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i],
                                symbol=self.marker_symbols[i],
                            ),
                            legendgroup=method.name,
                            showlegend=False,  # Keep as False to avoid duplicate legend entries
                            xaxis="x2",  # Reference to second x-axis
                            yaxis="y2",  # Reference to second y-axis
                        )
                        frame_data.append(convergence_trace)

                    # Error plot (row 2, col 2)
                    error_trace = go.Scatter(
                        x=filtered_df["Iteration"],
                        y=filtered_df["Error"],
                        mode="lines+markers",
                        name=method.name,
                        line=dict(color=self.colors[i], width=2),
                        marker=dict(
                            color=self.colors[i],
                            size=self.marker_sizes[i],
                            symbol=self.marker_symbols[i],
                        ),
                        legendgroup=method.name,
                        showlegend=False,  # Keep as False to avoid duplicate legend entries
                        xaxis="x3",  # Reference to third x-axis
                        yaxis="y3",  # Reference to third y-axis
                    )
                    frame_data.append(error_trace)

            # Create frame
            frame = go.Frame(
                data=frame_data,
                name=f"iteration_{iteration}",
                traces=list(range(len(frame_data))),
            )
            frames.append(frame)

            # Create slider step
            slider_step = {
                "args": [
                    [f"iteration_{iteration}"],
                    {
                        "frame": {
                            "duration": self.vis_config.animation_duration,
                            "redraw": True,
                        },
                        "mode": "immediate",
                        "transition": {
                            "duration": self.vis_config.animation_transition
                        },
                    },
                ],
                "label": str(iteration),
                "method": "animate",
            }
            slider_steps.append(slider_step)

        return frames, slider_steps

    def run_comparison(self, show_plots: bool = True) -> Optional[go.Figure]:
        """
        Run a comparison of all configured methods and visualize the results.

        Args:
            show_plots: Whether to display the plots

        Returns:
            The plotly figure object if show_plots is False
        """
        # Prepare data from all methods
        self._prepare_data()

        # Create a figure with a 2-column layout: 50% for function plot, 50% for convergence/error plots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"rowspan": 2, "colspan": 1}, {"rowspan": 1, "colspan": 1}],
                [None, {"rowspan": 1, "colspan": 1}],
            ],
            column_widths=[0.5, 0.5],  # Equal 50-50 split
            row_heights=[0.5, 0.5],
            subplot_titles=[
                "<b>Function Plot</b>",  # Bold titles
                "<b>Convergence of x Values</b>",
                "",  # Empty for the None cell
                "<b>Error Rate</b>",
            ],
            vertical_spacing=0.15,  # Increased spacing between rows
            horizontal_spacing=0.08,  # Increased spacing between columns
        )

        # Add function visualization in the first column (spans both rows)
        self._create_function_space(fig, row=1, col=1)

        # Add initial points
        self._add_initial_points(fig, row=1, col=1)

        # Add convergence plot in the top of the second column
        self._create_convergence_plot(fig, row=1, col=2)

        # Add error plot in the bottom of the second column
        self._create_error_plot(fig, row=2, col=2)

        # Add summary text to function plot
        summary_text = (
            f"<b>Function:</b> {self.vis_config.title.split(':')[-1].strip()}<br>"
        )

        # Add information for each method
        for i, method in enumerate(self.methods):
            if not self.all_data[i].empty:
                df = self.all_data[i]
                final_point = df.iloc[-1]

                # Format based on dimensionality
                if self.dimensions == 1:
                    root_val = f"{final_point['x_new']:.6f}"
                else:
                    # Format vector roots
                    root_val = str([f"{x:.6f}" for x in final_point["x_new"]])

                # Add method details
                summary_text += (
                    f"<span style='color:{self.colors[i]}'>{method.name}:</span> "
                )
                summary_text += f"Root={root_val}, f(Root)={final_point['f_new']:.2e}, "
                summary_text += f"Iter={int(final_point['Iteration'])+1}<br>"

        # Add the annotation to the function plot
        fig.add_annotation(
            text=summary_text,
            xref="x domain",
            yref="y domain",
            x=0.05,  # Position in the top left
            y=0.95,
            showarrow=False,
            font=dict(size=10, family="Arial"),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            borderpad=4,
            row=1,
            col=1,
        )

        # Create animation frames and slider steps
        frames, slider_steps = self._create_animation_frames()

        # Add frames to the figure
        fig.frames = frames

        # Add animation controls
        if frames:
            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [
                                    None,
                                    {
                                        "frame": {
                                            "duration": self.vis_config.animation_duration,
                                            "redraw": True,
                                        },
                                        "fromcurrent": True,
                                        "transition": {
                                            "duration": self.vis_config.animation_transition
                                        },
                                    },
                                ],
                                "label": "Play",
                                "method": "animate",
                            },
                            {
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": "Pause",
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top",
                    }
                ],
                sliders=[
                    {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 16},
                            "prefix": "Iteration: ",
                            "visible": True,
                            "xanchor": "right",
                        },
                        "transition": {
                            "duration": self.vis_config.animation_transition
                        },
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": slider_steps,
                    }
                ],
            )

        # Add borders to subplots for better distinction
        # The specific subplot layout is: 2x2 grid with function plot in (1,1) spanning 2 rows,
        # convergence plot in (1,2), and error plot in (2,2)

        # Map of used subplot positions to their axis indices
        subplot_axes = {
            (1, 1): 1,  # function plot
            (1, 2): 2,  # convergence plot
            (2, 2): 3,  # error plot
        }

        # Add borders and grids to each used subplot
        for pos, axis_idx in subplot_axes.items():
            row, col = pos
            # Only process existing subplots
            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor="lightgray",
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(211,211,211,0.3)",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                showline=True,
                linewidth=1,
                linecolor="lightgray",
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(211,211,211,0.3)",
                row=row,
                col=col,
            )

        # Update subplot title font sizes
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=14, family="Arial", color="#000000")

        # Update layout to use the full browser window
        fig.update_layout(
            autosize=True,
            paper_bgcolor=self.vis_config.background_color,
            plot_bgcolor=self.vis_config.background_color,
            showlegend=self.vis_config.show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                # Make the legend more compact and readable
                itemsizing="constant",
                itemwidth=30,
                font=dict(size=10),
                tracegroupgap=2,
            ),
            title={
                "text": f"<b>{self.vis_config.title}</b>",  # Bold title
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 24, "family": "Arial"},
            },
            margin=dict(l=70, r=70, t=100, b=100),
            height=800,  # Default height that will scale with browser
        )

        # Show the figure if requested
        if show_plots:
            config = {
                "displayModeBar": True,
                "responsive": True,
                "scrollZoom": True,
            }
            fig.show(config=config)
            return None
        else:
            return fig

    def create_3d_visualization(self) -> Optional[go.Figure]:
        """
        Create a specialized 3D visualization for 2D functions (surface plot with paths).

        Returns:
            The plotly figure object
        """
        if self.dimensions != 2:
            print(
                f"3D visualization only available for 2D functions, not {self.dimensions}D"
            )
            return None

        # Prepare data if not already done
        if not self.all_data:
            self._prepare_data()

        # Create a 3D figure
        fig = go.Figure()

        # Generate surface plot of the function
        x_min, x_max = self.config.x_range
        y_min, y_max = self.config.x_range  # Assuming same range for y

        # Create a grid of points
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Compute function values
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                except:
                    Z[i, j] = np.nan

        # Add surface plot
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="f(x,y)"),
                name="f(x,y)",
            )
        )

        # Add z=0 plane to show where roots would be
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=np.zeros_like(Z),
                colorscale=[[0, "rgba(255,255,255,0.3)"], [1, "rgba(255,255,255,0.3)"]],
                showscale=False,
                name="z=0 plane",
            )
        )

        # Add paths for each method
        for i, method in enumerate(self.methods):
            df = self.all_data[i]

            if not df.empty:
                # Extract x and y coordinates from each iteration
                x_vals = []
                y_vals = []
                z_vals = []

                for j in range(len(df)):
                    x_new = df["x_new"].iloc[j]
                    if isinstance(x_new, np.ndarray) and len(x_new) == 2:
                        x, y = x_new
                        z = self.config.func(np.array([x, y]))

                        x_vals.append(x)
                        y_vals.append(y)
                        z_vals.append(z)

                # Add path as a scatter3d trace
                fig.add_trace(
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode="lines+markers",
                        name=method.name,
                        line=dict(color=self.colors[i], width=5),
                        marker=dict(
                            color=self.colors[i],
                            size=self.marker_sizes[i],
                            symbol=(
                                self.marker_symbols[i][
                                    : self.marker_symbols[i].find("-")
                                ]
                                if "-" in self.marker_symbols[i]
                                else self.marker_symbols[i]
                            ),
                        ),
                    )
                )

                # Add final point with a larger marker
                if x_vals:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_vals[-1]],
                            y=[y_vals[-1]],
                            z=[z_vals[-1]],
                            mode="markers",
                            name=f"{method.name} final",
                            marker=dict(
                                color=self.colors[i],
                                size=self.marker_sizes[i] + 4,
                                symbol=(
                                    self.marker_symbols[i][
                                        : self.marker_symbols[i].find("-")
                                    ]
                                    if "-" in self.marker_symbols[i]
                                    else self.marker_symbols[i]
                                ),
                                line=dict(color="black", width=1),
                            ),
                        )
                    )

        # Update layout for better 3D visualization
        fig.update_layout(
            autosize=True,
            title=f"3D Visualization of Root-Finding Methods",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x,y)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
            ),
            showlegend=self.vis_config.show_legend,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        # Enable better controls for 3D visualization
        config = {
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": True,
        }

        return fig

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table with performance metrics for all methods.

        Returns:
            DataFrame with method performance metrics
        """
        # Prepare data if not already done
        if not self.all_data:
            self._prepare_data()

        # Prepare summary data
        summary_data = []

        for i, method in enumerate(self.methods):
            df = self.all_data[i]

            if not df.empty:
                # Extract final point
                final_point = df.iloc[-1]

                # Calculate convergence metrics
                iterations = int(final_point["Iteration"]) + 1
                error = final_point["Error"]
                converged = method.has_converged()

                # Get convergence rate if available
                conv_rate = None
                if hasattr(method, "get_convergence_rate"):
                    try:
                        conv_rate = method.get_convergence_rate()
                    except:
                        pass

                # Format the root value based on dimensionality
                if self.dimensions == 1:
                    root_val = f"{final_point['x_new']:.6f}"
                else:
                    # Format vector roots
                    root_val = str([f"{x:.6f}" for x in final_point["x_new"]])

                # Add to summary data
                summary_data.append(
                    {
                        "Method": method.name,
                        "Root": root_val,
                        "f(Root)": f"{final_point['f_new']:.2e}",
                        "Error": f"{error:.2e}",
                        "Iterations": iterations,
                        "Converged": converged,
                        "Convergence Rate": (
                            f"{conv_rate:.2f}" if conv_rate is not None else "N/A"
                        ),
                    }
                )

        return pd.DataFrame(summary_data)

    def save_visualization(self, filename: str, format: str = "html") -> None:
        """
        Save the visualization to a file.

        Args:
            filename: Name of the file to save to
            format: Format to save in ('html', 'png', 'jpg', 'svg', or 'pdf')
        """
        # Create the visualization if not already done
        fig = self.run_comparison(show_plots=False)

        if fig is not None:
            if format.lower() == "html":
                fig.write_html(
                    f"{filename}.html", include_plotlyjs="cdn", full_html=True
                )
            elif format.lower() in ["png", "jpg", "jpeg", "svg", "pdf"]:
                fig.write_image(f"{filename}.{format}", scale=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"Visualization saved to {filename}.{format}")
