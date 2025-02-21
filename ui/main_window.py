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
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QImage
import numpy as np
from sympy import sympify, symbols, diff, lambdify
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import traceback
import matplotlib.pyplot as plt
from matplotlib import patheffects
import re
from io import BytesIO

from .optimization import METHOD_MAP, run_optimization


class FunctionInput(QGroupBox):
    """Custom widget for function input with validation."""

    def __init__(self, parent=None):
        super().__init__("Function Definition", parent)
        self._latex_update_timer = QTimer()
        self._latex_update_timer.setSingleShot(True)
        self._latex_update_timer.timeout.connect(self._update_latex_display)
        self.setup_ui()
        self.update_derivative_requirements("Newton's Method")  # Default method

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)  # Add spacing between elements

        # Function input section
        input_layout = QHBoxLayout()
        func_label = QLabel("f(x) =")
        func_label.setStyleSheet("color: #00ffff; font-weight: bold;")
        input_layout.addWidget(func_label)

        self.func_input = QLineEdit()
        # Himmelblau function as default
        self.func_input.setText("(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2")
        self.func_input.setPlaceholderText(
            "Python-compatible function, e.g., x1**2 + x2**2"
        )
        self.func_input.textChanged.connect(self._schedule_latex_update)
        input_layout.addWidget(self.func_input)
        layout.addLayout(input_layout)

        # LaTeX display
        self.latex_display = QLabel()
        self.latex_display.setStyleSheet(
            """
            QLabel {
                background-color: #242935;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                padding: 20px;
                min-height: 120px;
                margin: 10px 0;
            }
        """
        )
        self.latex_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.latex_display)

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

        # Initial LaTeX render
        self._update_latex_display()

    def _schedule_latex_update(self):
        """Schedule a LaTeX update after a short delay to prevent excessive updates."""
        self._latex_update_timer.start(500)  # 500ms delay

    def _update_latex_display(self):
        """Update the LaTeX display with the current function."""
        try:
            # Get the function text and convert Python notation to LaTeX
            func_text = self.func_input.text()

            # Basic replacements for common mathematical notation
            latex_text = func_text

            # Handle parentheses groups first
            def replace_in_parentheses(match):
                content = match.group(1)
                # Handle powers inside parentheses
                content = re.sub(r"\*\*(\d+)", r"^{\1}", content)
                return f"({content})"

            # Replace contents in parentheses
            latex_text = re.sub(r"\(([^()]+)\)", replace_in_parentheses, latex_text)

            # Handle remaining powers
            latex_text = re.sub(r"\*\*(\d+)", r"^{\1}", latex_text)

            # Handle multiplication
            latex_text = re.sub(r"(?<=\d)\*", r" \\cdot ", latex_text)  # Number * ...
            latex_text = re.sub(r"\*(?=\d)", r" \\cdot ", latex_text)  # ... * Number
            latex_text = re.sub(r"(?<=[x])\*", r" \\cdot ", latex_text)  # x * ...
            latex_text = re.sub(r"\*(?=[x])", r" \\cdot ", latex_text)  # ... * x
            latex_text = re.sub(r"\*", r" \\cdot ", latex_text)  # Remaining *

            # Handle variables with subscripts
            latex_text = re.sub(r"x1", r"x_1", latex_text)
            latex_text = re.sub(r"x2", r"x_2", latex_text)

            # Remove extra spaces around operators
            latex_text = re.sub(r"\s*([+\-])\s*", r" \1 ", latex_text)
            latex_text = re.sub(r"\s*\\cdot\s*", r" \\cdot ", latex_text)

            # Create a figure with fixed size
            width = self.latex_display.width() / 80  # Adjusted for better scaling
            height = 2.0  # Increased height
            fig = Figure(figsize=(width, height))
            fig.patch.set_facecolor("#242935")

            # Create axes that fill the figure
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_facecolor("#242935")

            # Render the LaTeX equation
            eq = ax.text(
                0.5,
                0.5,
                f"$f(x) = {latex_text}$",
                color="#00ffff",
                fontsize=24,  # Significantly increased font size
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="bold",
            )  # Made text bold

            # Add a stronger glow effect
            eq.set_path_effects(
                [
                    patheffects.withStroke(linewidth=4, foreground="#242935"),
                    patheffects.Normal(),
                    patheffects.withStroke(
                        linewidth=2, foreground="#00ffff", alpha=0.3
                    ),  # Added subtle cyan glow
                ]
            )

            # Convert matplotlib figure to QPixmap with higher DPI
            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                facecolor="#242935",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.3,
                dpi=200,
            )  # Increased DPI and padding
            buf.seek(0)

            # Create QImage from buffer
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)

            # Set a fixed height for the label while maintaining aspect ratio
            fixed_height = 120  # Increased height
            scaled_pixmap = pixmap.scaledToHeight(
                fixed_height, Qt.TransformationMode.SmoothTransformation
            )

            # Update the label
            self.latex_display.setPixmap(scaled_pixmap)

            # Clean up
            plt.close(fig)
            buf.close()

        except Exception as e:
            # If there's an error, show a simple message
            self.latex_display.setText("Invalid expression")
            print(f"LaTeX rendering error: {str(e)}")
            traceback.print_exc()

    def resizeEvent(self, event):
        """Handle resize events to update the LaTeX display size."""
        super().resizeEvent(event)
        # Only update if we have a valid pixmap
        if not self.latex_display.pixmap().isNull():
            self._update_latex_display()

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
        self.setWindowTitle(
            "Numerical Methods Visualizer (use full screen - recommended)"
        )
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
        left_layout.setSpacing(20)  # Add spacing between major sections
        left_layout.setContentsMargins(10, 10, 10, 10)  # Add margins around the panel

        # Add function input widget
        self.func_input = FunctionInput()
        left_layout.addWidget(self.func_input)

        # Add vertical spacing
        left_layout.addSpacing(10)

        # Add method selector widget
        self.method_selector = MethodSelector()
        left_layout.addWidget(self.method_selector)

        # Add vertical spacing
        left_layout.addSpacing(10)

        # Connect method change to derivative requirements update
        self.method_selector.method_combo.currentTextChanged.connect(
            self.func_input.update_derivative_requirements
        )

        # Add solve button with spacing
        button_layout = QHBoxLayout()
        button_layout.addSpacing(10)
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setFixedHeight(40)  # Make button taller
        button_layout.addWidget(self.solve_btn)
        button_layout.addSpacing(10)
        left_layout.addLayout(button_layout)

        # Add vertical spacing
        left_layout.addSpacing(20)

        # Add results section with header
        results_header = QLabel("Results:")
        results_header.setStyleSheet(
            "color: #00ffff; font-weight: bold; font-size: 12px;"
        )
        left_layout.addWidget(results_header)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)  # Set minimum height
        self.results_text.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px;
                background-color: #242935;
                color: #ffffff;
            }
        """
        )
        left_layout.addWidget(self.results_text)

        # Add stretch at the bottom to push everything up
        left_layout.addStretch()

        # Add left panel to main layout
        layout.addWidget(left_panel, stretch=1)

        # Right panel for visualization
        right_panel = QTabWidget()
        layout.addWidget(right_panel, stretch=2)

        # Add visualization tabs
        self.surface_plot = FigureCanvasQTAgg(Figure())
        self.convergence_plot = FigureCanvasQTAgg(Figure())
        self.surface_ax = self.surface_plot.figure.add_subplot(111, projection="3d")
        self.convergence_ax = self.convergence_plot.figure.add_subplot(111)

        right_panel.addTab(self.surface_plot, "Surface Plot")
        right_panel.addTab(self.convergence_plot, "Convergence")

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

        # Reduce mesh resolution for better performance
        x1 = np.linspace(x1_min, x1_max, 50)  # Reduced from 150 to 50
        x2 = np.linspace(x2_min, x2_max, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = plot_func(X1, X2)

        # Add simplified contour plot with more vibrant colors
        offset = np.min(Z) - 0.2 * (np.max(Z) - np.min(Z))
        contours = self.surface_ax.contour(
            X1,
            X2,
            Z,
            zdir="z",
            offset=offset,
            levels=15,
            cmap="plasma",  # Match surface colormap
            alpha=0.7,  # Increased visibility
            linewidths=1.5,  # Slightly thicker lines
        )

        # Plot surface with optimized parameters
        surf = self.surface_ax.plot_surface(
            X1,
            X2,
            Z,
            cmap="plasma",  # Changed from viridis to plasma for more vibrant colors
            alpha=0.8,
            linewidth=0,
            antialiased=True,
            rcount=30,
            ccount=30,
        )

        # Enhanced colorbar
        cbar = self.surface_plot.figure.colorbar(surf, shrink=0.8, aspect=15, pad=0.1)
        cbar.ax.tick_params(colors=self.plot_colors["text"], labelsize=10)
        cbar.set_label(
            "Function Value", color="#00ffff", fontsize=12, weight="bold"
        )  # Bright cyan

        # Initialize empty lists for legend handles and labels
        legend_handles = []
        legend_labels = []

        if self.optimization_result is not None:
            path = np.array(self.optimization_result["path"])
            z_path = np.array([opt_func(p) for p in path])

            # Plot optimization path with enhanced visibility
            path_points = self.surface_ax.scatter(
                path[:, 0],
                path[:, 1],
                z_path,
                color="#ffffff",  # Fixed white color instead of colormap
                s=60,  # Increased point size
                alpha=1.0,  # Full opacity
                zorder=100,  # Ensure points are drawn on top
            )
            legend_handles.append(path_points)
            legend_labels.append("Optimization path")

            # Add path lines with gradient color
            for i in range(len(path) - 1):
                # Calculate color based on progress
                progress = i / (len(path) - 1)
                color = plt.cm.magma(progress)
                self.surface_ax.plot3D(
                    [path[i, 0], path[i + 1, 0]],
                    [path[i, 1], path[i + 1, 1]],
                    [z_path[i], z_path[i + 1]],
                    color="#ffffff",  # White color for path lines
                    linewidth=3,  # Thicker lines
                    alpha=0.9,  # More visible
                    zorder=99,  # Draw lines below points but above surface
                )

            # Highlight start and end points with more prominent markers
            start_point = self.surface_ax.scatter(
                path[0, 0],
                path[0, 1],
                z_path[0],
                color="#ffff00",  # Bright yellow
                s=200,  # Larger marker
                marker="^",
                linewidth=2,
                edgecolor="white",
                zorder=101,  # Ensure start point is on top
                label="Start",
            )
            end_point = self.surface_ax.scatter(
                path[-1, 0],
                path[-1, 1],
                z_path[-1],
                color="#ffffff",  # Bright white
                s=200,  # Larger marker
                marker="*",
                linewidth=2,
                edgecolor="#ffff00",  # Yellow edge for contrast
                zorder=101,  # Ensure end point is on top
                label="End",
            )
            legend_handles.extend([start_point, end_point])
            legend_labels.extend(["Start", "End"])

        # Enhanced axis labels
        self.surface_ax.set_xlabel(
            "x1", labelpad=10, color="#00ffff", fontsize=12, weight="bold"
        )  # Bright cyan
        self.surface_ax.set_ylabel(
            "x2", labelpad=10, color="#00ffff", fontsize=12, weight="bold"
        )  # Bright cyan
        self.surface_ax.set_zlabel(
            "f(x1, x2)", labelpad=10, color="#00ffff", fontsize=12, weight="bold"
        )  # Bright cyan

        # Enhanced tick labels
        self.surface_ax.tick_params(colors="#00ffff", labelsize=10)  # Bright cyan

        # Enhanced grid
        self.surface_ax.grid(
            True, linestyle="--", alpha=0.3, color="#4a4a4a"
        )  # Brighter grid

        # Enhanced axis panes
        for axis in [
            self.surface_ax.xaxis,
            self.surface_ax.yaxis,
            self.surface_ax.zaxis,
        ]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("#4a4a4a")  # Brighter pane edges
            axis.pane.set_alpha(0.3)
            axis.label.set_color("#00ffff")  # Bright cyan

        # Enhanced legend with better visibility
        if legend_handles:
            legend = self.surface_ax.legend(
                legend_handles,
                legend_labels,
                loc="upper center",  # Position at the top center
                bbox_to_anchor=(0.5, 1.15),  # Move above the plot
                facecolor=self.plot_colors["surface"],
                edgecolor="#4a4a4a",  # Brighter edge
                labelcolor="#ffffff",  # Changed to bright white for better visibility
                fontsize=12,  # Larger font
                framealpha=0.8,  # More opaque background
                borderpad=2,  # More padding
                handletextpad=2,  # More space between handles and text
                markerscale=1.5,  # Larger legend markers
                ncol=3,  # Display items in 3 columns for better horizontal layout
            )
            # Add a bright border around legend text
            for text in legend.get_texts():
                text.set_path_effects(
                    [
                        patheffects.withStroke(
                            linewidth=3, foreground=self.plot_colors["background"]
                        )
                    ]
                )

        # Adjust layout to accommodate legend at the top
        self.surface_plot.figure.subplots_adjust(
            top=0.85
        )  # Make room for legend at top

        # Enable mouse rotation with instant feedback
        self.surface_plot.figure.canvas.draw_idle()
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
