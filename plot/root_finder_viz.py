# plot/root_finder_viz.py

"""Root Finding Visualization Module."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from .base import VisualizationConfig, NumericalMethodVisualizer
from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig


class RootFindingVisualizer(NumericalMethodVisualizer):
    """
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
        super().__init__(config, methods, vis_config)

    def _create_function_space(self, fig, row=1, col=1) -> None:
        """
        Create the function space visualization based on dimensionality.

        Args:
            fig: The plotly figure to add the visualization to
            row: Row in the subplot grid
            col: Column in the subplot grid
        """
        if self.dimensions == 1:
            x_min, x_max = self.config.x_range
            x = np.linspace(x_min, x_max, 1000)
            try:
                y = [self.config.func(xi) for xi in x]
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
            x_min, x_max = self.config.x_range
            y_min, y_max = self.config.x_range
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                    except:
                        Z[i, j] = np.nan
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
            fig.update_layout(
                scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"
                )
            )
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
                                size=self.marker_sizes[i] + 2,
                                symbol=self.marker_symbols[i],
                                line=dict(color="black", width=1),
                            ),
                            name=method.name,
                            hoverinfo="text",
                            hovertext=f"Initial: x={x0:.6f}, f(x)={y0:.6f}",
                            legendgroup=method.name,
                            showlegend=True,
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
                                size=self.marker_sizes[i] + 2,
                                symbol=self.marker_symbols[i],
                                line=dict(color="black", width=1),
                            ),
                            name=method.name,
                            hoverinfo="text",
                            hovertext=f"Initial: x={x0:.4f}, y={y0:.4f}, f(x,y)={z0:.6f}",
                            legendgroup=method.name,
                            showlegend=True,
                        ),
                        row=row,
                        col=col,
                    )
                    point_traces.append(trace)
                elif self.dimensions == 3:
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
                            legendgroup=method.name,
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    norms = []
                    for j in range(len(df)):
                        if isinstance(df["x_new"].iloc[j], np.ndarray):
                            norms.append(np.linalg.norm(df["x_new"].iloc[j]))
                        else:
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
                            legendgroup=method.name,
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
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
                        legendgroup=method.name,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
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
        max_iterations = max(
            [df["Iteration"].max() if not df.empty else 0 for df in self.all_data]
        )
        for iteration in range(int(max_iterations) + 1):
            frame_data = []
            if self.dimensions == 1:
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
                            xaxis="x",
                            yaxis="y",
                        )
                    )
                    frame_data.append(
                        go.Scatter(
                            x=[x_min, x_max],
                            y=[0, 0],
                            mode="lines",
                            name="y=0",
                            line=dict(color="gray", width=1, dash="dash"),
                            legendgroup="function",
                            showlegend=False,
                            xaxis="x",
                            yaxis="y",
                        )
                    )
                except:
                    pass
            elif self.dimensions == 2:
                x_min, x_max = self.config.x_range
                y_min, y_max = self.config.x_range
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                        except:
                            Z[i, j] = np.nan
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
                        xaxis="x",
                        yaxis="y",
                    )
                )
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
                        xaxis="x",
                        yaxis="y",
                    )
                )
            elif self.dimensions == 3:
                pass

            for i, method in enumerate(self.methods):
                df = self.all_data[i]
                if df.empty:
                    continue
                iter_data = df[df["Iteration"] == iteration]
                if not iter_data.empty:
                    if self.dimensions == 1:
                        x_val = iter_data["x_new"].iloc[0]
                        y_val = iter_data["f_new"].iloc[0]
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
                                showlegend=True,
                                xaxis="x",
                                yaxis="y",
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
                                    showlegend=True,
                                    xaxis="x",
                                    yaxis="y",
                                )
                            )
                        except:
                            pass
                elif iteration > df["Iteration"].max():
                    last_point = df.iloc[-1]
                    if self.dimensions == 1:
                        x_val = last_point["x_new"]
                        y_val = last_point["f_new"]
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
                                hovertext=f"{method.name} (converged): x={x_val:.6f}, f(x)={y_val:.6f}",
                                legendgroup=method.name,
                                showlegend=True,
                                xaxis="x",
                                yaxis="y",
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
                                    showlegend=True,
                                    xaxis="x",
                                    yaxis="y",
                                )
                            )
                        except:
                            pass

                filtered_df = df[df["Iteration"] <= iteration]
                if not filtered_df.empty:
                    if self.dimensions == 1:
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
                            showlegend=False,
                            xaxis="x2",
                            yaxis="y2",
                        )
                        frame_data.append(convergence_trace)
                    else:
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
                            showlegend=False,
                            xaxis="x2",
                            yaxis="y2",
                        )
                        frame_data.append(convergence_trace)

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
                        showlegend=False,
                        xaxis="x3",
                        yaxis="y3",
                    )
                    frame_data.append(error_trace)

            frame = go.Frame(
                data=frame_data,
                name=f"iteration_{iteration}",
                traces=list(range(len(frame_data))),
            )
            frames.append(frame)
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

        if not self.all_data:
            self._prepare_data()

        fig = go.Figure()
        x_min, x_max = self.config.x_range
        y_min, y_max = self.config.x_range
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = self.config.func(np.array([X[i, j], Y[i, j]]))
                except:
                    Z[i, j] = np.nan
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
        for i, method in enumerate(self.methods):
            df = self.all_data[i]
            if not df.empty:
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
        if not self.all_data:
            self._prepare_data()

        summary_data = []
        for i, method in enumerate(self.methods):
            df = self.all_data[i]
            if not df.empty:
                final_point = df.iloc[-1]
                iterations = int(final_point["Iteration"]) + 1
                error = final_point["Error"]
                converged = method.has_converged()
                conv_rate = None
                if hasattr(method, "get_convergence_rate"):
                    try:
                        conv_rate = method.get_convergence_rate()
                    except:
                        pass
                if self.dimensions == 1:
                    root_val = f"{final_point['x_new']:.6f}"
                else:
                    root_val = str([f"{x:.6f}" for x in final_point["x_new"]])
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
