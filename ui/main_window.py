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
    QGroupBox,
    QTextEdit,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QImage
import numpy as np
from sympy import sympify, symbols, diff, lambdify
import traceback
import re
from io import BytesIO

# Plotly imports for interactive visualizations
import plotly.graph_objects as go
from plotly.offline import plot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

# Matplotlib imports needed for LaTeX rendering
from matplotlib.figure import Figure
from matplotlib import patheffects
import matplotlib.pyplot as plt

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
        layout.setSpacing(10)  # Reduced spacing between elements

        # Function input section with updated styling
        input_layout = QHBoxLayout()
        func_label = QLabel("f(x) =")
        func_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 11px;")
        input_layout.addWidget(func_label)

        self.func_input = QLineEdit()
        # Himmelblau function as default
        self.func_input.setText("(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2")
        self.func_input.setPlaceholderText(
            "Python-compatible function, e.g., x1**2 + x2**2"
        )
        self.func_input.textChanged.connect(self._schedule_latex_update)
        self.func_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                selection-background-color: #4a90e2;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
        """
        )
        input_layout.addWidget(self.func_input)
        layout.addLayout(input_layout)

        # LaTeX display with updated styling
        self.latex_display = QLabel()
        self.latex_display.setStyleSheet(
            """
            QLabel {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 10px;
                min-height: 80px;
                margin: 5px 0;
            }
        """
        )
        self.latex_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.latex_display)

        # Variable inputs and initial guess with updated styling
        var_layout = QHBoxLayout()
        var_layout.setSpacing(10)  # Reduced spacing between variable sections
        self.var_inputs = []
        self.initial_guess_inputs = []

        for i in range(2):
            var_group = QGroupBox(f"x{i+1} settings")
            var_group.setStyleSheet(
                """
                QGroupBox {
                    font-weight: bold;
                    color: #ffffff;
                    font-size: 10px;
                }
            """
            )
            var_layout_inner = QVBoxLayout()
            var_layout_inner.setSpacing(5)  # Reduced spacing between elements

            # Range inputs with updated styling
            range_layout = QHBoxLayout()
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1000, 1000)
            min_spin.setValue(-5)
            min_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1000, 1000)
            max_spin.setValue(5)
            max_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            # Updated labels
            min_label = QLabel("Min:")
            min_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")
            max_label = QLabel("Max:")
            max_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

            range_layout.addWidget(min_label)
            range_layout.addWidget(min_spin)
            range_layout.addWidget(max_label)
            range_layout.addWidget(max_spin)
            var_layout_inner.addLayout(range_layout)

            # Initial guess input with updated styling
            guess_layout = QHBoxLayout()
            guess_spin = QDoubleSpinBox()
            guess_spin.setRange(-1000, 1000)
            guess_spin.setValue(0)  # Default to 0
            guess_spin.setStyleSheet(
                """
                QDoubleSpinBox {
                    padding-right: 10px;
                    background-color: #222222;
                    color: #ffffff;
                    border: 1px solid #333333;
                    border-radius: 3px;
                }
            """
            )

            # Updated label
            initial_label = QLabel("Initial:")
            initial_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

            guess_layout.addWidget(initial_label)
            guess_layout.addWidget(guess_spin)
            var_layout_inner.addLayout(guess_layout)

            var_group.setLayout(var_layout_inner)
            var_layout.addWidget(var_group)
            self.var_inputs.append((min_spin, max_spin))
            self.initial_guess_inputs.append(guess_spin)

        layout.addLayout(var_layout)

        # Derivatives with updated styling
        deriv_group = QGroupBox("Derivatives")
        deriv_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                font-size: 10px;
            }
        """
        )
        deriv_layout = QVBoxLayout()
        deriv_layout.setSpacing(5)  # Reduced spacing

        # Create more compact derivatives section
        self.show_deriv = QCheckBox("Show first derivative")
        self.show_second_deriv = QCheckBox("Show second derivative")
        self.show_second_deriv.setChecked(
            True
        )  # Enable second derivative for Newton's method

        # Add updated styling to checkboxes
        checkbox_style = """
            QCheckBox {
                color: #b3b3b3;
                font-size: 10px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 2px;
                border: 1px solid #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
        """
        self.show_deriv.setStyleSheet(checkbox_style)
        self.show_second_deriv.setStyleSheet(checkbox_style)

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
            height = 1.0  # Reduced height from 2.0 to 1.0 for more compact display
            fig = Figure(figsize=(width, height))
            fig.patch.set_facecolor("#242935")

            # Create axes that fill the figure
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_facecolor("#242935")

            # Render the LaTeX equation with sharper font
            eq = ax.text(
                0.5,
                0.5,
                f"$f(x) = {latex_text}$",
                color="#00ffff",
                fontsize=22,  # Slightly reduced font size for compactness
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="bold",
                family="serif",  # Use serif font for sharper appearance
            )

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

            # Convert matplotlib figure to QPixmap with higher DPI for sharper rendering
            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                facecolor="#242935",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,  # Reduced padding
                dpi=300,  # Increased DPI from 200 to 300 for sharper text
            )
            buf.seek(0)

            # Create QImage from buffer
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)

            # Set a fixed height for the label while maintaining aspect ratio
            fixed_height = 80  # Reduced height from 120 to 80
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

        # Base checkbox style
        checkbox_style = """
            QCheckBox {
                color: #b3b3b3;
                font-size: 10px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border-radius: 2px;
                border: 1px solid #333333;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
        """

        if method_name in requirements:
            req = requirements[method_name]

            # First derivative
            if req["first"]:
                self.show_deriv.setChecked(True)
                self.show_deriv.setStyleSheet(
                    checkbox_style
                    + """
                    QCheckBox {
                        color: #4a90e2;
                        font-weight: bold;
                    }
                """
                )
            else:
                # Apply the base style
                self.show_deriv.setStyleSheet(checkbox_style)

            # Second derivative
            if req["second"]:
                self.show_second_deriv.setChecked(True)
                self.show_second_deriv.setStyleSheet(
                    checkbox_style
                    + """
                    QCheckBox {
                        color: #4a90e2;
                        font-weight: bold;
                    }
                """
                )
            else:
                # Apply the base style
                self.show_second_deriv.setStyleSheet(checkbox_style)

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
        layout.setSpacing(10)  # Reduced spacing

        # Method selection with updated styling
        method_label = QLabel("Method:")
        method_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")
        layout.addWidget(method_label)

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

        # Updated combobox styling
        self.method_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 5px;
                padding-right: 15px;
                font-size: 10px;
                min-height: 20px;
            }
            QComboBox:focus {
                border: 1px solid #4a90e2;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left: none;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                width: 0;
                height: 0;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 3px;
                selection-background-color: #333333;
                selection-color: #ffffff;
                padding: 2px;
            }
        """
        )
        layout.addWidget(self.method_combo)

        # Parameters with updated styling
        param_group = QGroupBox("Parameters")
        param_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                font-size: 10px;
            }
        """
        )
        param_layout = QVBoxLayout()
        param_layout.setSpacing(10)  # Reduced spacing

        # Tolerance with updated styling
        tol_layout = QHBoxLayout()
        tol_label = QLabel("Tolerance:")
        tol_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-6, 1.0)  # Changed minimum to 1e-6
        self.tol_spin.setValue(1e-4)  # Default tolerance of 1e-4
        self.tol_spin.setDecimals(6)  # Show 6 decimal places
        self.tol_spin.setSingleStep(1e-4)  # Step by 1e-4

        # Updated spinbox styling
        self.tol_spin.setStyleSheet(
            """
            QDoubleSpinBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 3px;
                padding-right: 12px;
                min-width: 100px;
                font-size: 10px;
            }
            QDoubleSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #333333;
                width: 12px;
                border-radius: 2px;
                margin: 1px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4a90e2;
            }
        """
        )

        tol_layout.addWidget(tol_label)
        tol_layout.addWidget(self.tol_spin)
        param_layout.addLayout(tol_layout)

        # Max iterations with updated styling
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Max iterations:")
        iter_label.setStyleSheet("color: #b3b3b3; font-size: 10px;")

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10000)
        self.max_iter_spin.setValue(100)

        # Updated spinbox styling
        self.max_iter_spin.setStyleSheet(
            """
            QSpinBox {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 3px;
                padding-right: 12px;
                min-width: 100px;
                font-size: 10px;
            }
            QSpinBox:focus {
                border: 1px solid #4a90e2;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #333333;
                width: 12px;
                border-radius: 2px;
                margin: 1px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #4a90e2;
            }
        """
        )

        iter_layout.addWidget(iter_label)
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
        self.animation_in_progress = False

        # Initialize plot_colors before setup_ui to avoid errors
        self.plot_colors = {
            "background": "#000000",  # Pure black for consistent background
            "surface": "#000000",  # Pure black for surface
            "input_bg": "#222222",  # Input background
            "panel_header": "#111111",  # Panel header background
            "panel_highlight": "#333333",  # Panel highlight
            "text": "#ffffff",  # White text
            "text_secondary": "#b3e5fc",  # Light blue secondary text
            "border": "#2f3646",  # Dark border color
            "primary": "#4a90e2",  # Professional blue
            "secondary": "#5c6bc0",  # Indigo accent
            "accent": "#00ffff",  # Cyan accent
            "gradient_start": "#3a7bd5",  # Gradient start
            "gradient_end": "#00d2ff",  # Gradient end
            "success": "#66bb6a",  # Green for success states
            "warning": "#ffa726",  # Orange for warnings
            "error": "#ef5350",  # Red for errors
        }

        self.setup_ui()

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel for inputs - Enhanced styling
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")  # Add object name for specific styling
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)  # Reduced spacing between major sections
        left_layout.setContentsMargins(
            10, 10, 10, 10
        )  # Reduced margins for more compact look

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

        # Add solve button with updated styling
        button_layout = QHBoxLayout()
        button_layout.addSpacing(5)
        self.solve_btn = QPushButton("SOLVE")
        self.solve_btn.setFixedHeight(35)  # Make button more compact
        self.solve_btn.setObjectName(
            "solveButton"
        )  # Add object name for specific styling
        self.solve_btn.setStyleSheet(
            """
            #solveButton {
                background-color: #4a90e2;
                color: #ffffff;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
            }
            #solveButton:hover {
                background-color: #2c7be5;
            }
            #solveButton:pressed {
                background-color: #1a5eb8;
                padding-top: 2px;
            }
        """
        )
        button_layout.addWidget(self.solve_btn)
        button_layout.addSpacing(5)
        left_layout.addLayout(button_layout)

        # Add results section with enhanced header
        results_header = QLabel("RESULTS")
        results_header.setObjectName(
            "resultsHeader"
        )  # Add object name for specific styling
        results_header.setStyleSheet(
            f"""
            #resultsHeader {{
                color: {self.plot_colors['text']};
                font-weight: bold;
                font-size: 12px;
                background-color: {self.plot_colors['panel_header']};
                padding: 3px 8px;
                border-radius: 2px;
                letter-spacing: 1px;
                margin-top: 5px;
            }}
            """
        )
        results_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(results_header)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        # Remove minimum height to allow results to expand to fill available space
        self.results_text.setStyleSheet(
            f"""
            QTextEdit {{
                border: 1px solid {self.plot_colors['panel_highlight']};
                border-radius: 3px;
                padding: 5px;
                background-color: {self.plot_colors['input_bg']};
                color: {self.plot_colors['text']};
            }}
            """
        )
        # Set the results text to expand to fill available space (stretch=1)
        left_layout.addWidget(self.results_text, stretch=1)

        # Add left panel to main layout with specific styling
        left_panel.setStyleSheet(
            f"""
            #leftPanel {{
                background-color: {self.plot_colors['left_panel_bg'] if 'left_panel_bg' in self.plot_colors else '#000000'};
                border-right: 1px solid {self.plot_colors['panel_highlight']};
                border-radius: 0px;
            }}
            """
        )
        layout.addWidget(left_panel, stretch=1)

        # Right panel for visualization - with enhanced styling
        right_panel = QTabWidget()
        right_panel.setObjectName("rightPanel")  # Add object name for specific styling
        layout.addWidget(right_panel, stretch=2)

        # Add visualization tabs - replace Matplotlib with Plotly
        self.surface_plot = QWebEngineView()
        self.surface_plot.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptEnabled, True
        )
        self.surface_plot.setMinimumHeight(500)  # Increased minimum height
        # Set default background color to black to prevent white flash
        self.surface_plot.setStyleSheet("background-color: #000000;")

        # Initialize with a blank black page to prevent white flash
        blank_black_html = """
        <html><head><style>
        body, html {
            background-color: #000000;
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
        }
        </style></head><body></body></html>
        """
        self.surface_plot.setHtml(blank_black_html)

        self.convergence_plot = QWebEngineView()
        self.convergence_plot.page().settings().setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptEnabled, True
        )
        self.convergence_plot.setMinimumHeight(500)  # Increased minimum height
        # Set default background color to black to prevent white flash
        self.convergence_plot.setStyleSheet("background-color: #000000;")

        # Initialize with a blank black page to prevent white flash
        self.convergence_plot.setHtml(blank_black_html)

        right_panel.addTab(self.surface_plot, "Surface Plot")
        right_panel.addTab(self.convergence_plot, "Convergence")

        # Apply specific styling to the right panel
        right_panel.setStyleSheet(
            f"""
            #rightPanel::pane {{
                border: none;
                background-color: #000000;
            }}
            #rightPanel > QTabBar::tab {{
                background-color: #000000;
                padding: 12px 25px;
                margin-right: 5px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                letter-spacing: 1px;
                font-size: 12px;
            }}
            #rightPanel > QTabBar::tab:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 {self.plot_colors['gradient_start']}, 
                                          stop:1 {self.plot_colors['gradient_end']});
                color: white;
            }}
            """
        )

        # Connect signal
        self.solve_btn.clicked.connect(self.solve)

        # Set window properties
        self.setMinimumSize(1200, 800)
        self.apply_styling()

    def apply_styling(self):
        """Apply modern styling to the UI."""
        # Set color scheme
        palette = QPalette()

        # Enhanced professional color palette - updated for left panel
        colors = {
            "background": "#000000",  # Changed to pure black to match left panel
            "surface": "#000000",  # Changed to pure black
            "left_panel_bg": "#000000",  # Pure black for left panel
            "primary": "#4a90e2",  # Professional blue
            "secondary": "#5c6bc0",  # Indigo accent
            "accent": "#00ffff",  # Cyan accent
            "gradient_start": "#3a7bd5",  # Gradient start
            "gradient_end": "#00d2ff",  # Gradient end
            "success": "#66bb6a",  # Green for success states
            "warning": "#ffa726",  # Orange for warnings
            "error": "#ef5350",  # Red for errors
            "text": "#ffffff",  # White text
            "text_secondary": "#b3b3b3",  # Light gray secondary text
            "border": "#222222",  # Dark border color
            "hover": "#3d8bd4",  # Hover state color
            "panel_header": "#111111",  # Panel header background
            "panel_highlight": "#333333",  # Panel highlight
            "input_bg": "#222222",  # Input background
        }

        # Update plot_colors with the black background
        self.plot_colors.update(
            {
                "background": "#000000",  # Changed to pure black
                "surface": "#000000",  # Changed to pure black
                "left_panel_bg": "#000000",  # Pure black
            }
        )

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

        # Set fonts with improved sharpness
        font = QFont("Segoe UI", 10)
        font.setHintingPreference(
            QFont.HintingPreference.PreferFullHinting
        )  # Add hinting for sharper rendering
        self.setFont(font)

        # Enhanced style for more professional and compact left panel with sharper fonts
        style = f"""
            QMainWindow {{
                background-color: {colors['background']};
            }}
            QPushButton {{
                background-color: {colors['panel_highlight']};
                color: {colors['text']};
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 0.5px;  /* Reduced for sharper text */
            }}
            QPushButton:hover {{
                background-color: {colors['primary']};
            }}
            QPushButton:pressed {{
                background-color: {colors['secondary']};
                padding-top: 9px;
                padding-bottom: 7px;
            }}
            QGroupBox {{
                border: 1px solid {colors['panel_highlight']};
                border-radius: 4px;
                margin-top: 1.2em;
                padding-top: 1em;
                padding-bottom: 0.5em;
                padding-left: 0.5em;
                padding-right: 0.5em;
                color: {colors['text']};
                font-weight: bold;
                background-color: {colors['left_panel_bg']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 5px;
                color: {colors['text']};
                background-color: {colors['panel_header']};
                border-radius: 2px;
                font-size: 11px;
                letter-spacing: 0.5px;  /* Reduced for sharper text */
            }}
            QLabel {{
                color: {colors['text_secondary']};
                font-weight: 400;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['panel_highlight']};
                border-radius: 3px;
                padding: 5px;
                selection-background-color: {colors['primary']};
                font-weight: 400;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {colors['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 5px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {colors['text']};
                width: 0;
                height: 0;
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                selection-background-color: {colors['panel_highlight']};
                selection-color: {colors['text']};
                border-radius: 2px;
                padding: 2px;
            }}
            QCheckBox {{
                color: {colors['text_secondary']};
                spacing: 5px;
                font-weight: 400;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {colors['panel_highlight']};
                border-radius: 2px;
                background-color: {colors['input_bg']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['primary']};
            }}
            QTextEdit {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['panel_highlight']};
                border-radius: 3px;
                padding: 5px;
                selection-background-color: {colors['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {colors['panel_highlight']};
                border-radius: 3px;
                background-color: {colors['surface']};
            }}
            QTabBar::tab {{
                background-color: {colors['background']};
                color: {colors['text']};
                padding: 8px 15px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                margin-right: 1px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {colors['primary']};
            }}
            QTabBar::tab:hover {{
                background-color: {colors['hover']};
            }}
            QScrollBar:vertical {{
                background-color: {colors['left_panel_bg']};
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors['panel_highlight']};
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {colors['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background-color: {colors['left_panel_bg']};
                height: 10px;
                margin: 0;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {colors['panel_highlight']};
                border-radius: 5px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {colors['primary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
            QWebEngineView {{
                background-color: {colors['background']};
                border: 1px solid {colors['panel_highlight']};
                border-radius: 3px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {colors['panel_highlight']};
                border-radius: 2px;
                margin: 1px;
            }}
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: {colors['primary']};
            }}
        """
        self.setStyleSheet(style)

        # Update plot_colors with the current color palette
        self.plot_colors.update(
            {
                "background": colors["background"],
                "surface": colors["surface"],
                "text": colors["text"],
                "text_secondary": colors["text_secondary"],
                "grid": colors["border"],
                "accent": colors["accent"],
                "primary": colors["primary"],
                "secondary": colors["secondary"],
                "success": colors["success"],
                "warning": colors["warning"],
                "error": colors["error"],
                "gradient_start": colors["gradient_start"],
                "gradient_end": colors["gradient_end"],
                "panel_header": colors["panel_header"],
                "panel_highlight": colors["panel_highlight"],
                "input_bg": colors["input_bg"],
                "left_panel_bg": colors["left_panel_bg"],
            }
        )

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
        """Update the surface plot with the current function using Plotly."""
        # Use raw function for plotting surface and wrapped function for optimization path
        plot_func = self.raw_func if func is None else func
        opt_func = func if func is not None else (lambda x: self.raw_func(x[0], x[1]))

        bounds = self.get_bounds()
        x1_min, x1_max = bounds[0]
        x2_min, x2_max = bounds[1]

        # Create mesh for surface
        resolution = 50  # Reduced from original 150 for better performance
        x1 = np.linspace(x1_min, x1_max, resolution)
        x2 = np.linspace(x2_min, x2_max, resolution)
        X1, X2 = np.meshgrid(x1, x2)
        Z = plot_func(X1, X2)

        # Create figure with better sizing
        fig = go.Figure()

        # Add surface with enhanced styling
        surface = go.Surface(
            x=X1,
            y=X2,
            z=Z,
            colorscale="Plasma",
            opacity=0.9,
            showscale=True,
            lighting=dict(
                ambient=0.6, diffuse=0.8, fresnel=0.2, roughness=0.9, specular=1.0
            ),
            colorbar=dict(
                title=dict(text="Function Value", font=dict(color="#00ffff", size=14)),
                tickfont=dict(color="#ffffff"),
                thickness=20,
                len=0.8,
                outlinewidth=0,
            ),
            contours=dict(
                x=dict(show=True, width=2, color="#444444"),
                y=dict(show=True, width=2, color="#444444"),
                z=dict(show=True, width=2, color="#444444"),
            ),
            hoverinfo="x+y+z",
        )
        fig.add_trace(surface)

        # Add contour at bottom for better effect
        contour = go.Contour(
            x=x1,
            y=x2,
            z=Z.min() * np.ones_like(Z),
            contours=dict(start=Z.min(), end=Z.max(), size=(Z.max() - Z.min()) / 15),
            colorscale="Plasma",
            showscale=False,
            opacity=0.7,
            line=dict(width=2),
            hoverinfo="none",
        )

        # Add path and points if optimization has been run
        if self.optimization_result is not None:
            path = np.array(self.optimization_result["path"])
            z_path = np.array([opt_func(p) for p in path])

            # Add path line with gradient color
            path_points = go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=z_path,
                mode="lines+markers",
                line=dict(color="#ffffff", width=5),
                marker=dict(
                    size=4,
                    color=np.arange(len(path)),  # Color points by steps
                    colorscale="Turbo",
                    opacity=0.8,
                    symbol="circle",
                ),
                name="Optimization Path",
                hovertemplate="x1: %{x:.4f}<br>x2: %{y:.4f}<br>f(x): %{z:.4f}<br>Step: %{marker.color}<extra></extra>",
            )
            fig.add_trace(path_points)

            # Add special markers for start and end points with improved contrast colors
            fig.add_trace(
                go.Scatter3d(
                    x=[path[0, 0]],
                    y=[path[0, 1]],
                    z=[z_path[0]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="#ff3399",  # Bright pink for better contrast
                        opacity=1.0,
                        symbol="diamond",
                        line=dict(
                            color="#ffffff", width=1
                        ),  # White border for definition
                    ),
                    name="Start",
                    hovertemplate="Start Point<br>x1: %{x:.4f}<br>x2: %{y:.4f}<br>f(x): %{z:.4f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[path[-1, 0]],
                    y=[path[-1, 1]],
                    z=[z_path[-1]],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color="#00ff99",  # Bright mint green for high contrast
                        opacity=1.0,
                        symbol="cross",  # Use 'cross' instead of 'star' which is not supported
                        line=dict(color="#ffffff", width=1.5),  # White border
                    ),
                    name="Solution",
                    hovertemplate="Final Solution<br>x1: %{x:.4f}<br>x2: %{y:.4f}<br>f(x): %{z:.4f}<extra></extra>",
                )
            )

        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text="Objective Function Surface Plot",
                font=dict(size=18, color="#00ffff"),
                x=0.5,
                y=0.97,
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="x1", font=dict(size=14, color="#00ffff")),
                    showbackground=True,
                    backgroundcolor="#000000",  # Changed to pure black
                    gridcolor="#4a4a4a",
                    zerolinecolor="#5e5e5e",
                    showspikes=False,
                    ticks="outside",
                    tickfont=dict(color="#ffffff"),
                ),
                yaxis=dict(
                    title=dict(text="x2", font=dict(size=14, color="#00ffff")),
                    showbackground=True,
                    backgroundcolor="#000000",  # Changed to pure black
                    gridcolor="#4a4a4a",
                    zerolinecolor="#5e5e5e",
                    showspikes=False,
                    ticks="outside",
                    tickfont=dict(color="#ffffff"),
                ),
                zaxis=dict(
                    title=dict(text="f(x1, x2)", font=dict(size=14, color="#00ffff")),
                    showbackground=True,
                    backgroundcolor="#000000",  # Changed to pure black
                    gridcolor="#4a4a4a",
                    zerolinecolor="#5e5e5e",
                    showspikes=False,
                    ticks="outside",
                    tickfont=dict(color="#ffffff"),
                ),
                aspectratio=dict(x=1, y=1, z=0.8),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1), up=dict(x=0, y=0, z=1)),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=12, color="#ffffff"),
                bgcolor="rgba(0, 0, 0, 0.7)",  # Changed to black with opacity
                bordercolor="#4a4a4a",
                borderwidth=1,
            ),
            paper_bgcolor="#000000",  # Changed to pure black
            plot_bgcolor="#000000",  # Changed to pure black
            template="plotly_dark",
            uirevision="true",  # Keep camera position on updates
        )

        # Add annotations for a more professional look
        if self.optimization_result is not None:
            # Use a 3D text instead of annotation since annotations don't support z-axis
            fig.add_trace(
                go.Scatter3d(
                    x=[path[-1, 0]],
                    y=[path[-1, 1]],
                    z=[z_path[-1] + (Z.max() - Z.min()) * 0.1],
                    mode="text",
                    text=[f"Minimum: {z_path[-1]:.6f}"],
                    textposition="top center",
                    textfont=dict(color="#00ffff", size=12),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        # Convert to HTML and display in the QWebEngineView
        html_str = plot(
            fig, output_type="div", include_plotlyjs="cdn", config={"responsive": True}
        )

        # Set HTML content in the WebEngineView
        self.surface_plot.setHtml(html_str)

    def update_convergence_plot(self):
        """Update the convergence plot with optimization history using Plotly."""
        if self.optimization_result is None:
            return

        iterations = list(range(len(self.optimization_result["function_values"])))
        values = self.optimization_result["function_values"]

        # Create a more visually appealing figure
        fig = go.Figure()

        # Add line chart with gradient color and markers
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=values,
                mode="lines+markers",
                line=dict(
                    color="#4a90e2",
                    width=3,
                    shape="spline",  # Smooth lines
                    smoothing=1.3,
                    dash="solid",
                ),
                marker=dict(
                    size=8,
                    color=iterations,  # Already converted to list
                    colorscale="Turbo",
                    line=dict(color="#ffffff", width=1),
                    opacity=0.8,
                ),
                name="Function Value",
                hovertemplate="Iteration: %{x}<br>Value: %{y:.6f}<extra></extra>",
            )
        )

        # Add markers for key points
        if len(values) > 0:
            # Mark starting point
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[values[0]],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=14,
                        color="#ffff00",
                        line=dict(color="#ffffff", width=1),
                    ),
                    name="Start",
                    hovertemplate="Start<br>Value: %{y:.6f}<extra></extra>",
                )
            )

            # Mark final point
            fig.add_trace(
                go.Scatter(
                    x=[len(values) - 1],
                    y=[values[-1]],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=16,
                        color="#00ffff",
                        line=dict(color="#ffffff", width=1),
                    ),
                    name="Final",
                    hovertemplate="Final<br>Value: %{y:.6f}<extra></extra>",
                )
            )

            # Mark minimum point if different from final
            min_idx = np.argmin(values)
            if min_idx != len(values) - 1:
                fig.add_trace(
                    go.Scatter(
                        x=[min_idx],
                        y=[values[min_idx]],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=12,
                            color="#66bb6a",
                            line=dict(color="#ffffff", width=1),
                        ),
                        name="Minimum",
                        hovertemplate="Minimum<br>Iteration: %{x}<br>Value: %{y:.6f}<extra></extra>",
                    )
                )

        # Add horizontal and vertical indicator lines for final value
        if len(values) > 1:
            final_value = values[-1]
            fig.add_trace(
                go.Scatter(
                    x=[0, len(iterations) - 1],
                    y=[final_value, final_value],
                    mode="lines",
                    line=dict(color="#ffffff", width=1, dash="dash"),
                    name="Final Value",
                    hoverinfo="none",
                )
            )

            # Add trend indicator annotations
            if values[0] > values[-1]:
                percent_reduction = (values[0] - values[-1]) / values[0] * 100
                fig.add_annotation(
                    x=len(iterations) * 0.5,
                    y=final_value,
                    text=f"↓ {percent_reduction:.1f}% reduction",
                    showarrow=False,
                    font=dict(size=12, color="#00ffff"),
                    bgcolor="rgba(0, 0, 0, 0.7)",  # Changed to black with opacity
                    bordercolor="#4a4a4a",
                    borderwidth=1,
                    yshift=10,
                )

        # Enhanced layout
        fig.update_layout(
            title=dict(
                text="Convergence History",
                font=dict(size=18, color="#00ffff"),
                x=0.5,
                y=0.95,  # Move title up slightly to make room for legend
            ),
            xaxis=dict(
                title=dict(
                    text="Iteration",
                    font=dict(size=14, color="#ffffff"),
                ),
                showgrid=True,
                gridcolor="#444444",
                griddash="dot",
                zeroline=True,
                zerolinecolor="#5e5e5e",
                zerolinewidth=1.5,
                showline=True,
                linecolor="#ffffff",
                tickfont=dict(color="#ffffff"),
                showspikes=True,
                spikethickness=1,
                spikecolor="#ffffff",
                spikemode="across",
            ),
            yaxis=dict(
                title=dict(
                    text="Function Value",
                    font=dict(size=14, color="#ffffff"),
                ),
                type=(
                    "log" if min(values) > 0 else "linear"
                ),  # Use log scale if all values are positive
                showgrid=True,
                gridcolor="#444444",
                griddash="dot",
                zeroline=True,
                zerolinecolor="#5e5e5e",
                zerolinewidth=1.5,
                showline=True,
                linecolor="#ffffff",
                tickfont=dict(color="#ffffff"),
                showspikes=True,
                spikethickness=1,
                spikecolor="#ffffff",
                spikemode="across",
            ),
            legend=dict(
                orientation="h",
                yanchor="top",  # Change from "bottom" to "top"
                y=0.88,  # Position below the title (was 1.02)
                xanchor="center",
                x=0.5,
                font=dict(size=12, color="#ffffff"),
                bgcolor="rgba(0, 0, 0, 0.7)",  # Changed to black with opacity
                bordercolor="#4a4a4a",
                borderwidth=1,
            ),
            hovermode="closest",
            paper_bgcolor="#000000",  # Changed to pure black
            plot_bgcolor="#000000",  # Changed to pure black
            margin=dict(
                l=20, r=20, t=80, b=20
            ),  # Increase top margin to accommodate title and legend
        )

        # Add visual comparison between start and end
        if len(values) > 1:
            # Add gradient shading to show progress
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=values,
                    fill="tozeroy",
                    fillcolor="rgba(0, 184, 212, 0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

            # Add annotation for total iterations
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Total iterations: {len(iterations)}",
                showarrow=False,
                font=dict(size=12, color="#ffffff"),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor="#4a4a4a",
                borderwidth=1,
                align="left",
            )

        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74, 74, 74, 0.3)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(74, 74, 74, 0.3)")

        # Configure interactive features
        config = {
            "displayModeBar": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "convergence_plot",
                "height": 800,
                "width": 1200,
                "scale": 2,
            },
        }

        # Convert to HTML and display in the QWebEngineView
        html_str = plot(fig, output_type="div", include_plotlyjs="cdn", config=config)

        # Set HTML content in the WebEngineView
        self.convergence_plot.setHtml(html_str)

    def update_animation(self):
        """Update animation frame for optimization visualization using Plotly."""
        if not self.animation_frames or self.current_frame >= len(
            self.animation_frames
        ):
            self.animation_timer.stop()
            return

        try:
            # Instead of redrawing the entire plot on each frame (which causes blinking),
            # update only the path points to show the current progress
            if self.current_frame == 0:
                # For the first frame, update the entire plot
                self.update_surface_plot()
            else:
                # Get the current frame data for path point animation
                current_path = self.animation_frames[: self.current_frame + 1]

                # Update just the path points without redrawing the whole plot
                # This requires modifying the existing plot instead of regenerating it
                path_update = {}
                if (
                    hasattr(self, "optimization_result")
                    and self.optimization_result is not None
                ):
                    func = lambda x: self.raw_func(x[0], x[1])
                    path = np.array(current_path)
                    z_path = np.array([func(p) for p in path])

                    # Update just the last point location - this is much less disruptive
                    if (
                        self.current_frame % 5 == 0
                    ):  # Only update display every 5 frames to reduce flicker
                        # We would update the plot here with Plotly's partial updates
                        # but QWebEngineView doesn't support direct JS execution
                        # So we'll just indicate progress in a non-blinking way
                        progress_percent = min(
                            100,
                            int(
                                (self.current_frame / len(self.animation_frames)) * 100
                            ),
                        )
                        if progress_percent % 10 == 0:  # Only print every 10%
                            print(f"Animation progress: {progress_percent}%")

            # Increment the frame counter for next update
            self.current_frame += 1

        except Exception as e:
            print(f"Animation error: {str(e)}")
            self.animation_timer.stop()

    def solve(self):
        """Handle solve button click."""
        try:
            # Show loading message in results area before starting computation
            loading_html = f"""
            <style>
                .loading-container {{
                    text-align: center;
                    padding: 20px;
                    margin-top: 20px;
                }}
                .loading-spinner {{
                    display: inline-block;
                    width: 40px;
                    height: 40px;
                    border: 4px solid rgba(74, 144, 226, 0.3);
                    border-radius: 50%;
                    border-top: 4px solid {self.plot_colors['primary']};
                    animation: spin 1s linear infinite;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .loading-text {{
                    margin-top: 15px;
                    color: {self.plot_colors['text']};
                    font-weight: bold;
                }}
                .loading-details {{
                    margin-top: 10px;
                    color: {self.plot_colors['text_secondary']};
                    font-size: 0.9em;
                }}
            </style>
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Calculating optimization...</div>
                <div class="loading-details">This may take a few moments depending on the complexity of the function and selected method</div>
            </div>
            """
            self.results_text.setHtml(loading_html)

            # Process events to update the UI before continuing
            from PyQt6.QtCore import QCoreApplication

            QCoreApplication.processEvents()

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

            # Update results text with enhanced formatting and more details
            status = "Success" if result["success"] else "Failed"
            status_color = "#66bb6a" if result["success"] else "#ef5350"  # Green or red

            html_result = f"""
            <style>
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #4a4a4a; }}
                th {{ background-color: #242935; color: #00ffff; font-weight: bold; }}
                .status-badge {{ 
                    display: inline-block; 
                    padding: 4px 8px; 
                    border-radius: 4px; 
                    background-color: {status_color}; 
                    color: white;
                    font-weight: bold;
                }}
                .highlight {{ color: #00ffff; font-weight: bold; }}
                .section-header {{ 
                    margin-top: 15px; 
                    margin-bottom: 5px; 
                    color: #4a90e2; 
                    font-weight: bold; 
                    border-bottom: 1px solid #4a90e2;
                    padding-bottom: 3px;
                }}
            </style>
            
            <div class="section-header">Optimization Result</div>
            <div><span class="status-badge">{status}</span> after {result['nit']} iterations</div>
            
            <div class="section-header">Solution</div>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>x<sub>1</sub></td>
                    <td class="highlight">{result['x'][0]:.6f}</td>
                </tr>
                <tr>
                    <td>x<sub>2</sub></td>
                    <td class="highlight">{result['x'][1]:.6f}</td>
                </tr>
                <tr>
                    <td>Function Value</td>
                    <td class="highlight">{result['fun']:.8f}</td>
                </tr>
            </table>
            
            <div class="section-header">Method Details</div>
            <table>
                <tr>
                    <td>Method</td>
                    <td>{method}</td>
                </tr>
                <tr>
                    <td>Iterations</td>
                    <td>{result['nit']}</td>
                </tr>
                <tr>
                    <td>Tolerance</td>
                    <td>{tol:.8f}</td>
                </tr>
                <tr>
                    <td>Message</td>
                    <td>{result['message']}</td>
                </tr>
            </table>
            
            <div class="section-header">Performance</div>
            <table>
                <tr>
                    <td>Function Evaluations</td>
                    <td>{result.get('nfev', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Gradient Evaluations</td>
                    <td>{result.get('njev', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Initial Value</td>
                    <td>{result['function_values'][0]:.6f}</td>
                </tr>
                <tr>
                    <td>Final Value</td>
                    <td>{result['fun']:.6f}</td>
                </tr>
                <tr>
                    <td>Improvement</td>
                    <td>{((result['function_values'][0] - result['fun']) / result['function_values'][0] * 100):.2f}%</td>
                </tr>
            </table>
            """

            self.results_text.setHtml(html_result)

            # Now update plots (show a loading message in the plots too)
            self.surface_plot.setHtml(
                f"""
            <html>
            <head>
                <style>
                    body, html {{
                        background-color: #000000 !important;
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        width: 100%;
                        overflow: hidden;
                    }}
                    .loading-container {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        z-index: 1000;
                        background-color: #000000;
                    }}
                    .loading-content {{
                        text-align: center;
                    }}
                    .loading-title {{
                        margin-bottom: 20px;
                        font-size: 20px;
                        color: {self.plot_colors['accent']};
                        font-family: 'Segoe UI', Arial, sans-serif;
                    }}
                    .loading-spinner {{
                        border: 5px solid rgba(74, 144, 226, 0.3);
                        border-radius: 50%;
                        border-top: 5px solid {self.plot_colors['primary']};
                        width: 50px;
                        height: 50px;
                        margin: 0 auto;
                        animation: spin 1s linear infinite;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                </style>
            </head>
            <body>
                <div class="loading-container">
                    <div class="loading-content">
                        <div class="loading-title">Generating Surface Plot</div>
                        <div class="loading-spinner"></div>
                    </div>
                </div>
            </body>
            </html>
            """
            )

            self.convergence_plot.setHtml(
                f"""
            <html>
            <head>
                <style>
                    body, html {{
                        background-color: #000000 !important;
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        width: 100%;
                        overflow: hidden;
                    }}
                    .loading-container {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        z-index: 1000;
                        background-color: #000000;
                    }}
                    .loading-content {{
                        text-align: center;
                    }}
                    .loading-title {{
                        margin-bottom: 20px;
                        font-size: 20px;
                        color: {self.plot_colors['accent']};
                        font-family: 'Segoe UI', Arial, sans-serif;
                    }}
                    .loading-spinner {{
                        border: 5px solid rgba(74, 144, 226, 0.3);
                        border-radius: 50%;
                        border-top: 5px solid {self.plot_colors['primary']};
                        width: 50px;
                        height: 50px;
                        margin: 0 auto;
                        animation: spin 1s linear infinite;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                </style>
            </head>
            <body>
                <div class="loading-container">
                    <div class="loading-content">
                        <div class="loading-title">Generating Convergence Plot</div>
                        <div class="loading-spinner"></div>
                    </div>
                </div>
            </body>
            </html>
            """
            )

            # Process events to update the UI
            QCoreApplication.processEvents()

            # Update plots
            self.update_surface_plot()
            self.update_convergence_plot()

            # Process events again after plots are ready
            QCoreApplication.processEvents()

            # Instead of animating all steps for large iteration counts,
            # sample a reasonable number of frames to prevent flickering
            max_animation_frames = 100  # Maximum frames to show in animation

            if len(result["path"]) > max_animation_frames:
                # Sample the path to get a reasonable number of frames
                indices = np.linspace(
                    0, len(result["path"]) - 1, max_animation_frames, dtype=int
                )
                self.animation_frames = [result["path"][i] for i in indices]
            else:
                self.animation_frames = result["path"]

            # Disable animation for very large iteration counts to prevent blinking
            if len(result["path"]) > 1000:
                # Just show the final result without animation
                self.animation_timer.stop()
                self.current_frame = 0
                print(
                    f"Animation disabled for large iteration count ({len(result['path'])} iterations)"
                )
            else:
                # Start animation for smaller iteration counts
                self.current_frame = 0
                self.animation_timer.start(
                    200
                )  # Slower update rate (200ms instead of 100ms)

        except Exception as e:
            # Enhanced error display
            error_html = f"""
            <style>
                .error-box {{ 
                    background-color: rgba(239, 83, 80, 0.1); 
                    padding: 10px; 
                    border-left: 4px solid #ef5350; 
                    margin: 10px 0;
                    border-radius: 4px;
                }}
                .error-title {{
                    color: #ef5350;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .error-message {{
                    color: #ffffff;
                }}
                .error-details {{
                    font-family: monospace;
                    background-color: #1a1f2b;
                    padding: 8px;
                    margin-top: 10px;
                    border-radius: 4px;
                    max-height: 200px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }}
            </style>
            
            <div class="error-box">
                <div class="error-title">Error</div>
                <div class="error-message">{str(e)}</div>
                <div class="error-details">{traceback.format_exc()}</div>
            </div>
            """
            self.results_text.setHtml(error_html)
            traceback.print_exc()

    def visualize(self):
        """Deprecated - functionality merged into solve()."""
        pass
