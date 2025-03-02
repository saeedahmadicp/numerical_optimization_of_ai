# plot/base.py

"""
Base Visualization Module for Numerical Methods.

This module provides the core visualization capabilities for numerical algorithms,
with a common structure that can be extended for specific use cases like
root-finding, optimization, interpolation, etc.
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
        show_top_right_plot: Whether to show top right plot (e.g., convergence)
        show_bottom_right_plot: Whether to show bottom right plot (e.g., error)
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

    show_top_right_plot: bool = True
    show_bottom_right_plot: bool = True
    style: str = "white"
    context: str = "talk"
    palette: str = "viridis"
    point_size: int = 8
    dpi: int = 100
    show_legend: bool = True
    grid_alpha: float = 0.2
    title: str = "Numerical Methods Comparison"
    background_color: str = "#FFFFFF"
    animation_duration: int = 800  # ms per frame
    animation_transition: int = 300  # ms for transition


class NumericalMethodVisualizer:
    """
    Base visualizer for numerical methods.

    This class provides the core functionality for visualizing the behavior of different
    numerical algorithms, with a common layout structure and animation framework.
    Specific visualization details are implemented in subclasses.
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
            methods: List of numerical methods to visualize
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
        self.final_points = []

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
        it to a format suitable for plotting. This is a base implementation
        that should be extended by subclasses.
        """
        self.all_data = []
        self.final_points = []

        for i, method in enumerate(self.methods):
            # Run the method until convergence if it hasn't been run yet
            if not method.get_iteration_history():
                while not method.has_converged():
                    method.step()

            # Get the iteration history
            history = method.get_iteration_history()

            # Extract the final point
            final_point = method.get_current_x()
            self.final_points.append(final_point)

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
        Create the function space visualization.

        This is a placeholder method that should be implemented by subclasses.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        raise NotImplementedError("Subclasses must implement _create_function_space")

    def _add_initial_points(self, fig, row=1, col=1) -> List[Dict]:
        """
        Add initial points for all methods to the visualization.

        This is a placeholder method that should be implemented by subclasses.

        Args:
            fig: The plotly figure to add points to
            row: Row in the subplot grid
            col: Column in the subplot grid

        Returns:
            List of trace indices for initial points
        """
        raise NotImplementedError("Subclasses must implement _add_initial_points")

    def _create_convergence_plot(self, fig, row=1, col=1) -> None:
        """
        Create the convergence plot.

        This is a placeholder method that should be implemented by subclasses.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        raise NotImplementedError("Subclasses must implement _create_convergence_plot")

    def _create_error_plot(self, fig, row=1, col=1) -> None:
        """
        Create the error plot.

        This is a placeholder method that should be implemented by subclasses.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        raise NotImplementedError("Subclasses must implement _create_error_plot")

    def _create_animation_frames(self) -> Tuple[List[go.Frame], List[Dict]]:
        """
        Create animation frames showing the progression of the numerical method.

        This is a placeholder method that should be implemented by subclasses.

        Returns:
            Tuple containing animation frames and slider steps
        """
        raise NotImplementedError("Subclasses must implement _create_animation_frames")

    def _add_summary_info(self, fig, row=1, col=1) -> None:
        """
        Add summary information to the visualization.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        # This can be implemented by subclasses or used as-is
        # Basic implementation that can be overridden
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
                    final_val = f"{final_point['x_new']:.6f}"
                else:
                    # Format vector final points
                    final_val = str([f"{x:.6f}" for x in final_point["x_new"]])

                # Add method details
                summary_text += (
                    f"<span style='color:{self.colors[i]}'>{method.name}:</span> "
                )
                summary_text += (
                    f"Result={final_val}, f(Result)={final_point['f_new']:.2e}, "
                )
                summary_text += f"Iter={int(final_point['Iteration'])+1}<br>"

        # Add the annotation to the plot
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
            row=row,
            col=col,
        )

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

        # Create a figure with a 2-column layout: 50% for function plot, 50% for metrics plots
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
                "<b>Top Metric</b>",  # Updated by subclass
                "",  # Empty for the None cell
                "<b>Bottom Metric</b>",  # Updated by subclass
            ],
            vertical_spacing=0.15,  # Increased spacing between rows
            horizontal_spacing=0.08,  # Increased spacing between columns
        )

        # Add function visualization in the first column (spans both rows)
        self._create_function_space(fig, row=1, col=1)

        # Add initial points
        self._add_initial_points(fig, row=1, col=1)

        # Add convergence plot
        self._create_convergence_plot(fig, row=1, col=2)

        # Add error plot
        self._create_error_plot(fig, row=2, col=2)

        # Add summary text to function plot
        self._add_summary_info(fig, row=1, col=1)

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
        # Map of used subplot positions to their axis indices
        subplot_axes = {
            (1, 1): 1,  # function plot
            (1, 2): 2,  # top right plot
            (2, 2): 3,  # bottom right plot
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
                gridcolor=f"rgba(211,211,211,{self.vis_config.grid_alpha})",
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
                gridcolor=f"rgba(211,211,211,{self.vis_config.grid_alpha})",
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

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table with performance metrics for all methods.

        This is a placeholder method that should be overridden by subclasses.

        Returns:
            DataFrame with method performance metrics
        """
        raise NotImplementedError("Subclasses must implement generate_summary_table")

    def create_3d_visualization(self) -> Optional[go.Figure]:
        """
        Create a specialized 3D visualization.

        This is a placeholder method that can be overridden by subclasses.

        Returns:
            The plotly figure object, or None if 3D visualization is not applicable
        """
        return None
