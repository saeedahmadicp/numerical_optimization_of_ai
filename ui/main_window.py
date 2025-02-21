# ui/main_window.py

"""Main window for the numerical methods UI."""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QTextEdit,
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import numpy as np
from sympy import sympify, symbols, diff, lambdify
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import traceback

from .optimization import METHOD_MAP, run_optimization


class FunctionInput(QGroupBox):
    """Custom widget for function input with validation."""

    def __init__(self, parent=None):
        super().__init__("Function Definition", parent)
        self.setup_ui()
        self.update_derivative_requirements("Newton's Method")  # Default method

    def setup_ui(self):
        layout = QVBoxLayout()

        # Function input
        self.func_input = QLineEdit()
        # Himmelblau function as default
        self.func_input.setText("(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2")
        self.func_input.setPlaceholderText("e.g., x1**2 + x2**2")
        layout.addWidget(QLabel("f(x) ="))
        layout.addWidget(self.func_input)

        # Variable inputs and initial guess
        var_layout = QHBoxLayout()
        self.var_inputs = []
        self.initial_guess_inputs = []

        for i in range(2):
            var_group = QGroupBox(f"x{i+1} settings")
            var_layout_inner = QVBoxLayout()

            # Range inputs
            range_layout = QHBoxLayout()
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1000, 1000)
            min_spin.setValue(-5)

            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1000, 1000)
            max_spin.setValue(5)

            range_layout.addWidget(QLabel("Min:"))
            range_layout.addWidget(min_spin)
            range_layout.addWidget(QLabel("Max:"))
            range_layout.addWidget(max_spin)
            var_layout_inner.addLayout(range_layout)

            # Initial guess input
            guess_layout = QHBoxLayout()
            guess_spin = QDoubleSpinBox()
            guess_spin.setRange(-1000, 1000)
            guess_spin.setValue(0)  # Default to 0
            guess_layout.addWidget(QLabel("Initial:"))
            guess_layout.addWidget(guess_spin)
            var_layout_inner.addLayout(guess_layout)

            var_group.setLayout(var_layout_inner)
            var_layout.addWidget(var_group)
            self.var_inputs.append((min_spin, max_spin))
            self.initial_guess_inputs.append(guess_spin)

        layout.addLayout(var_layout)

        # Derivatives
        deriv_group = QGroupBox("Derivatives")
        deriv_layout = QVBoxLayout()

        self.show_deriv = QCheckBox("Show first derivative")
        self.show_second_deriv = QCheckBox("Show second derivative")
        self.show_second_deriv.setChecked(
            True
        )  # Enable second derivative for Newton's method

        deriv_layout.addWidget(self.show_deriv)
        deriv_layout.addWidget(self.show_second_deriv)

        deriv_group.setLayout(deriv_layout)
        layout.addWidget(deriv_group)

        self.setLayout(layout)

    def update_derivative_requirements(self, method_name):
        """Update derivative checkboxes based on method requirements."""
        # Reset styles
        self.show_deriv.setStyleSheet("")
        self.show_second_deriv.setStyleSheet("")

        # Define requirements for each method
        requirements = {
            "Newton's Method": {"first": True, "second": True},
            "BFGS": {"first": True, "second": False},
            "Steepest Descent": {"first": True, "second": False},
            "Nelder-Mead": {"first": False, "second": False},
            "Powell's Method": {"first": False, "second": False},
        }

        if method_name in requirements:
            req = requirements[method_name]

            # First derivative
            if req["first"]:
                self.show_deriv.setChecked(True)
                self.show_deriv.setStyleSheet(
                    """
                    QCheckBox {
                        color: #ef5350;
                        font-weight: bold;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #ef5350;
                        border-color: #ef5350;
                    }
                """
                )

            # Second derivative
            if req["second"]:
                self.show_second_deriv.setChecked(True)
                self.show_second_deriv.setStyleSheet(
                    """
                    QCheckBox {
                        color: #ef5350;
                        font-weight: bold;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #ef5350;
                        border-color: #ef5350;
                    }
                """
                )

    def get_initial_guess(self):
        """Get the initial guess values."""
        return np.array([spin.value() for spin in self.initial_guess_inputs])


class MethodSelector(QGroupBox):
    """Widget for selecting and configuring numerical methods."""

    def __init__(self, parent=None):
        super().__init__("Method Selection", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Method selection
        self.method_combo = QComboBox()
        methods = [
            "Newton's Method",
            "BFGS",
            "Steepest Descent",
            "Nelder-Mead",
            "Powell's Method",
        ]
        self.method_combo.addItems(methods)
        self.method_combo.setCurrentText(
            "Newton's Method"
        )  # Set Newton's method as default
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_combo)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        # Tolerance
        tol_layout = QHBoxLayout()
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-6, 1.0)  # Changed minimum to 1e-6
        self.tol_spin.setValue(1e-4)  # Default tolerance of 1e-4
        self.tol_spin.setDecimals(6)  # Show 6 decimal places
        self.tol_spin.setSingleStep(1e-4)  # Step by 1e-4
        tol_layout.addWidget(QLabel("Tolerance:"))
        tol_layout.addWidget(self.tol_spin)
        param_layout.addLayout(tol_layout)

        # Max iterations
        iter_layout = QHBoxLayout()
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(100)
        iter_layout.addWidget(QLabel("Max iterations:"))
        iter_layout.addWidget(self.max_iter_spin)
        param_layout.addLayout(iter_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Main window for the numerical methods UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Numerical Methods Visualizer")
        self.optimization_result = None
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_frames = []
        self.current_frame = 0
        self.plot_colors = None  # Initialize plot_colors as None
        self.setup_ui()

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Add function input widget
        self.func_input = FunctionInput()
        left_layout.addWidget(self.func_input)

        # Add method selector widget
        self.method_selector = MethodSelector()
        left_layout.addWidget(self.method_selector)

        # Connect method change to derivative requirements update
        self.method_selector.method_combo.currentTextChanged.connect(
            self.func_input.update_derivative_requirements
        )

        # Add solve button
        button_layout = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        button_layout.addWidget(self.solve_btn)
        left_layout.addLayout(button_layout)

        # Add results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        left_layout.addWidget(QLabel("Results:"))
        left_layout.addWidget(self.results_text)

        # Add left panel to main layout
        layout.addWidget(left_panel, stretch=1)

        # Right panel for visualization
        right_panel = QTabWidget()

        # Add visualization tabs
        self.surface_plot = FigureCanvasQTAgg(Figure())
        self.convergence_plot = FigureCanvasQTAgg(Figure())
        self.surface_ax = self.surface_plot.figure.add_subplot(111, projection="3d")
        self.convergence_ax = self.convergence_plot.figure.add_subplot(111)

        right_panel.addTab(self.surface_plot, "Surface Plot")
        right_panel.addTab(self.convergence_plot, "Convergence")

        # Add right panel to main layout
        layout.addWidget(right_panel, stretch=2)

        # Connect signal
        self.solve_btn.clicked.connect(self.solve)

        # Set window properties
        self.setMinimumSize(1200, 800)
        self.apply_styling()

    def apply_styling(self):
        """Apply modern styling to the UI."""
        # Set color scheme
        palette = QPalette()

        # Professional color palette
        colors = {
            "background": "#1a1f2b",  # Dark navy background
            "surface": "#242935",  # Lighter navy for surfaces
            "primary": "#4a90e2",  # Professional blue
            "secondary": "#5c6bc0",  # Indigo accent
            "accent": "#00b8d4",  # Cyan accent
            "success": "#66bb6a",  # Green for success states
            "warning": "#ffa726",  # Orange for warnings
            "error": "#ef5350",  # Red for errors
            "text": "#ffffff",  # White text
            "text_secondary": "#b3e5fc",  # Light blue secondary text
            "border": "#2f3646",  # Dark border color
            "hover": "#3d8bd4",  # Hover state color
        }

        # Main colors
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["background"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["background"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["primary"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["text"]))

        self.setPalette(palette)

        # Set fonts
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # Style buttons and widgets
        style = f"""
            QMainWindow {{
                background-color: {colors['background']};
            }}
            QPushButton {{
                background-color: {colors['primary']};
                color: {colors['text']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['secondary']};
            }}
            QGroupBox {{
                border: 2px solid {colors['border']};
                border-radius: 6px;
                margin-top: 1em;
                padding-top: 1em;
                color: {colors['text']};
                font-weight: bold;
                background-color: {colors['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {colors['accent']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 6px;
                selection-background-color: {colors['primary']};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 2px solid {colors['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {colors['text']};
                width: 0;
                height: 0;
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['surface']};
                color: {colors['text']};
                selection-background-color: {colors['primary']};
                selection-color: {colors['text']};
            }}
            QCheckBox {{
                color: {colors['text']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {colors['border']};
                border-radius: 4px;
                background-color: {colors['surface']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['accent']};
                border-color: {colors['accent']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['primary']};
            }}
            QTextEdit {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 8px;
                selection-background-color: {colors['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
                background-color: {colors['surface']};
            }}
            QTabBar::tab {{
                background-color: {colors['background']};
                color: {colors['text']};
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors['primary']};
            }}
            QTabBar::tab:hover {{
                background-color: {colors['hover']};
            }}
            QScrollBar:vertical {{
                background-color: {colors['background']};
                width: 12px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {colors['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background-color: {colors['background']};
                height: 12px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {colors['border']};
                border-radius: 6px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {colors['primary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
        """
        self.setStyleSheet(style)

        # Update matplotlib style for plots
        self.plot_colors = {  # Store as instance variable
            "background": colors["background"],
            "surface": colors["surface"],
            "text": colors["text"],
            "grid": colors["border"],
            "accent": colors["accent"],
        }

        # Surface plot styling
        self.surface_ax.set_facecolor(self.plot_colors["background"])
        self.surface_ax.grid(
            True, color=self.plot_colors["grid"], alpha=0.2, linestyle="--"
        )
        self.surface_ax.xaxis.label.set_color(self.plot_colors["text"])
        self.surface_ax.yaxis.label.set_color(self.plot_colors["text"])
        self.surface_ax.zaxis.label.set_color(self.plot_colors["text"])
        self.surface_ax.tick_params(colors=self.plot_colors["text"])
        self.surface_plot.figure.patch.set_facecolor(self.plot_colors["background"])

        # Set 3D plot pane colors
        self.surface_ax.xaxis.set_pane_color(
            (
                *[
                    int(self.plot_colors["surface"][i : i + 2], 16) / 255
                    for i in (1, 3, 5)
                ],
                0.95,
            )
        )
        self.surface_ax.yaxis.set_pane_color(
            (
                *[
                    int(self.plot_colors["surface"][i : i + 2], 16) / 255
                    for i in (1, 3, 5)
                ],
                0.95,
            )
        )
        self.surface_ax.zaxis.set_pane_color(
            (
                *[
                    int(self.plot_colors["surface"][i : i + 2], 16) / 255
                    for i in (1, 3, 5)
                ],
                0.95,
            )
        )

        # Convergence plot styling
        self.convergence_ax.set_facecolor(self.plot_colors["background"])
        self.convergence_ax.grid(
            True, color=self.plot_colors["grid"], alpha=0.2, linestyle="--"
        )
        self.convergence_ax.xaxis.label.set_color(self.plot_colors["text"])
        self.convergence_ax.yaxis.label.set_color(self.plot_colors["text"])
        self.convergence_ax.tick_params(colors=self.plot_colors["text"])
        self.convergence_plot.figure.patch.set_facecolor(self.plot_colors["background"])

    def parse_function(self):
        """Parse and validate the input function."""
        try:
            func_str = self.func_input.func_input.text()
            x1, x2 = symbols("x1 x2")

            # Parse the function
            expr = sympify(func_str)

            # Check if function uses only allowed variables
            used_symbols = expr.free_symbols
            if not used_symbols.issubset({x1, x2}):
                raise ValueError("Function must use only x1 and x2 variables")

            # Create lambda functions with array input wrappers
            raw_func = lambdify([x1, x2], expr, "numpy")

            # Vectorized wrapper for optimization
            def func(x):
                if isinstance(x, np.ndarray):
                    if len(x.shape) == 1:  # Single point
                        return float(raw_func(x[0], x[1]))
                    else:  # Multiple points or meshgrid
                        return raw_func(x[..., 0], x[..., 1])
                return raw_func(x[0], x[1])  # Handle list/tuple

            # Calculate derivatives
            dx1 = diff(expr, x1)
            dx2 = diff(expr, x2)
            raw_grad = lambdify([x1, x2], [dx1, dx2], "numpy")

            def grad(x):
                if isinstance(x, np.ndarray) and len(x.shape) == 1:
                    return np.array(raw_grad(x[0], x[1]))
                return np.array(raw_grad(x[0], x[1]))

            # Calculate second derivatives if needed
            if self.func_input.show_second_deriv.isChecked():
                dx1x1 = diff(dx1, x1)
                dx1x2 = diff(dx1, x2)
                dx2x2 = diff(dx2, x2)
                raw_hessian = lambdify(
                    [x1, x2], [[dx1x1, dx1x2], [dx1x2, dx2x2]], "numpy"
                )

                def hessian(x):
                    if isinstance(x, np.ndarray) and len(x.shape) == 1:
                        return np.array(raw_hessian(x[0], x[1]))
                    return np.array(raw_hessian(x[0], x[1]))

            else:
                hessian = None

            # Store raw function for plotting
            self.raw_func = raw_func
            return func, grad, hessian

        except Exception as e:
            raise ValueError(f"Invalid function: {str(e)}")

    def get_bounds(self):
        """Get the variable bounds from inputs."""
        bounds = []
        for min_spin, max_spin in self.func_input.var_inputs:
            min_val = min_spin.value()
            max_val = max_spin.value()
            if min_val >= max_val:
                raise ValueError("Invalid bounds: min must be less than max")
            bounds.append((min_val, max_val))
        return bounds

    def update_surface_plot(self, func=None):
        """Update the surface plot with the current function."""
        self.surface_ax.clear()
        self.surface_plot.figure.clear()  # Clear the entire figure to prevent legend duplication
        self.surface_ax = self.surface_plot.figure.add_subplot(111, projection="3d")

        # Reapply styling after clearing
        self.surface_ax.set_facecolor(self.plot_colors["background"])
        self.surface_ax.tick_params(colors=self.plot_colors["text"])
        self.surface_plot.figure.patch.set_facecolor(self.plot_colors["background"])

        # Use raw function for plotting surface and wrapped function for optimization path
        plot_func = self.raw_func if func is None else func
        opt_func = func if func is not None else (lambda x: self.raw_func(x[0], x[1]))

        bounds = self.get_bounds()
        x1_min, x1_max = bounds[0]
        x2_min, x2_max = bounds[1]

        # Create a finer mesh for smoother plotting
        x1 = np.linspace(x1_min, x1_max, 150)  # Increased resolution
        x2 = np.linspace(x2_min, x2_max, 150)
        X1, X2 = np.meshgrid(x1, x2)
        Z = plot_func(X1, X2)  # Use raw function for surface plot

        # Add contour plot at the bottom with enhanced visibility
        offset = np.min(Z) - 0.2 * (np.max(Z) - np.min(Z))  # Increased offset
        contours = self.surface_ax.contour(
            X1,
            X2,
            Z,
            zdir="z",
            offset=offset,
            levels=30,  # Increased number of levels
            cmap="viridis",
            alpha=0.7,  # Increased opacity
            linewidths=2,  # Thicker contour lines
        )

        # Plot the surface with enhanced styling
        surf = self.surface_ax.plot_surface(
            X1,
            X2,
            Z,
            cmap="viridis",
            alpha=0.8,
            linewidth=0.5,
            antialiased=True,
            rstride=2,
            cstride=2,
        )

        # Add color bar with better positioning
        cbar = self.surface_plot.figure.colorbar(
            surf, shrink=0.8, aspect=15, pad=0.1  # Adjusted aspect ratio
        )
        cbar.ax.tick_params(colors=self.plot_colors["text"])
        cbar.set_label("Function Value", color=self.plot_colors["text"], fontsize=10)

        # Initialize empty lists for legend handles and labels
        legend_handles = []
        legend_labels = []

        if self.optimization_result is not None:
            path = np.array(self.optimization_result["path"])
            z_path = np.array(
                [opt_func(p) for p in path]
            )  # Use wrapped function for path

            # Plot optimization path with gradient color
            path_points = self.surface_ax.scatter(
                path[:, 0],
                path[:, 1],
                z_path,
                c=range(len(path)),
                cmap="Reds",
                s=50,
                alpha=1.0,
            )
            legend_handles.append(path_points)
            legend_labels.append("Optimization path")

            # Add lines connecting the points
            for i in range(len(path) - 1):
                self.surface_ax.plot(
                    [path[i, 0], path[i + 1, 0]],
                    [path[i, 1], path[i + 1, 1]],
                    [z_path[i], z_path[i + 1]],
                    color="red",
                    alpha=0.3,
                    linewidth=1,
                )

            # Highlight start and end points
            start_point = self.surface_ax.scatter(
                path[0, 0],
                path[0, 1],
                z_path[0],
                color="green",
                s=150,  # Increased size
                marker="^",
            )
            end_point = self.surface_ax.scatter(
                path[-1, 0],
                path[-1, 1],
                z_path[-1],
                color="red",
                s=150,  # Increased size
                marker="*",
            )
            legend_handles.extend([start_point, end_point])
            legend_labels.extend(["Start", "End"])

        # Enhance the appearance
        self.surface_ax.set_xlabel("x1", labelpad=10, color=self.plot_colors["text"])
        self.surface_ax.set_ylabel("x2", labelpad=10, color=self.plot_colors["text"])
        self.surface_ax.set_zlabel(
            "f(x1, x2)", labelpad=10, color=self.plot_colors["text"]
        )

        # Set better viewing angle
        self.surface_ax.view_init(elev=25, azim=45)  # Adjusted viewing angle

        # Add grid with custom styling
        self.surface_ax.grid(
            True, linestyle="--", alpha=0.3, color=self.plot_colors["grid"]
        )

        # Customize axis appearance
        for axis in [
            self.surface_ax.xaxis,
            self.surface_ax.yaxis,
            self.surface_ax.zaxis,
        ]:
            axis.pane.fill = False
            axis.pane.set_edgecolor(self.plot_colors["grid"])
            axis.pane.set_alpha(0.3)
            axis.label.set_color(self.plot_colors["text"])

        # Add legend only once with custom styling
        if legend_handles:
            self.surface_ax.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                facecolor=self.plot_colors["surface"],
                edgecolor=self.plot_colors["grid"],
                labelcolor=self.plot_colors["text"],
            )

        # Adjust layout to prevent clipping
        self.surface_plot.figure.tight_layout()
        self.surface_plot.draw()

    def update_convergence_plot(self):
        """Update the convergence plot with optimization history."""
        if self.optimization_result is None:
            return

        self.convergence_ax.clear()

        iterations = range(len(self.optimization_result["function_values"]))
        values = self.optimization_result["function_values"]

        self.convergence_ax.plot(iterations, values, "b-", label="Function value")
        self.convergence_ax.set_xlabel("Iteration")
        self.convergence_ax.set_ylabel("Function Value")
        self.convergence_ax.set_yscale("log")
        self.convergence_ax.grid(True)
        self.convergence_ax.legend()

        self.convergence_plot.draw()

    def update_animation(self):
        """Update animation frame for optimization visualization."""
        if not self.animation_frames or self.current_frame >= len(
            self.animation_frames
        ):
            self.animation_timer.stop()
            return

        frame = self.animation_frames[self.current_frame]
        self.surface_ax.clear()
        self.update_surface_plot()

        # Plot current position
        self.surface_ax.scatter(
            frame[0],
            frame[1],
            self.optimization_result["function_values"][self.current_frame],
            color="red",
            s=100,
            marker="*",
        )

        self.surface_plot.draw()
        self.current_frame += 1

    def solve(self):
        """Handle solve button click."""
        try:
            # Get function and parameters
            func, grad, hessian = self.parse_function()
            method = self.method_selector.method_combo.currentText()
            tol = self.method_selector.tol_spin.value()
            max_iter = self.method_selector.max_iter_spin.value()
            bounds = self.get_bounds()

            # Get initial guess from user input instead of using center of bounds
            x0 = self.func_input.get_initial_guess()

            # Validate initial guess is within bounds
            for i, (lower, upper) in enumerate(bounds):
                if not (lower <= x0[i] <= upper):
                    raise ValueError(f"Initial guess for x{i+1} must be within bounds")

            # Get method class and run optimization
            if method not in METHOD_MAP:
                raise ValueError(f"Unknown method: {method}")

            method_class = METHOD_MAP[method]
            result = run_optimization(
                method_class=method_class,
                func=func,
                grad=grad,
                hessian=hessian,
                x0=x0,
                tol=tol,
                max_iter=max_iter,
                bounds=bounds,
            )

            # Store result and update display
            self.optimization_result = result

            # Update results text
            status = "Success" if result["success"] else "Failed"
            message = (
                f"Optimization {status}\n"
                f"Final value: {result['fun']:.6f}\n"
                f"Solution: x1={result['x'][0]:.6f}, x2={result['x'][1]:.6f}\n"
                f"Iterations: {result['nit']}\n"
                f"Message: {result['message']}"
            )
            self.results_text.setText(message)

            # Update plots
            self.update_surface_plot()
            self.update_convergence_plot()

            # Start animation
            self.animation_frames = result["path"]
            self.current_frame = 0
            self.animation_timer.start(100)  # Update every 100ms

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            traceback.print_exc()

    def visualize(self):
        """Deprecated - functionality merged into solve()."""
        pass
