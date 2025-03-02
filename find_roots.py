#!/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path
import pandas as pd
import os
import numpy as np

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.newton_hessian import NewtonHessianMethod
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.secant import SecantMethod
from algorithms.convex.bisection import BisectionMethod
from algorithms.convex.regula_falsi import RegulaFalsiMethod
from plot.root_finder_viz import RootFindingVisualizer, VisualizationConfig
from utils.funcs import get_test_function, FUNCTION_MAP
from utils.midpoint import get_safe_initial_points

# Map method names to their classes
METHOD_MAP: Dict[str, Type[BaseNumericalMethod]] = {
    # Root-finding only methods
    "bisection": BisectionMethod,
    "regula_falsi": RegulaFalsiMethod,
    "secant": SecantMethod,
    # Methods that can do both root-finding and optimization
    "newton": NewtonMethod,
    "newton_hessian": NewtonHessianMethod,
    "bfgs": BFGSMethod,
}

# Group methods by capability
ROOT_FINDING_ONLY = {"bisection", "regula_falsi", "secant"}
DUAL_METHODS = {"newton", "newton_hessian", "bfgs"}

# Default ranges for different function types
DEFAULT_RANGES = {
    "quadratic": (-3, 3),
    "cubic": (-2, 3),
    "quartic": (-3, 3),
    "exponential": (-2, 2),
    "logarithmic": (0.1, 4),
    "exp_linear": (-1, 2),
    "sinusoidal": (0, 2 * 3.14159),
    "cosine": (0, 2),
    "tangent": (-1.5, 1.5),
    "trig_polynomial": (0, 6),
    "exp_sine": (0, 2),
    "stiff": (0, 1),
    "multiple_roots": (-3, 2),
    "ill_conditioned": (0, 2),
    "2d_himmelblau": (-6, 6),
    "2d_rastrigin": (-5, 5),
}


def load_config_file(config_path: Path) -> dict:
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration values

    Raises:
        ValueError: If file format is not supported or file can't be read
    """
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError("Configuration file must be .yaml, .yml, or .json")
    except Exception as e:
        raise ValueError(f"Error reading configuration file: {e}")


def determine_x_range(
    function_name: str,
    x0_values: List[float],
    specified_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Determine the appropriate x-range for visualization.

    Args:
        function_name: Name of the function to be used
        x0_values: Initial point(s) for the methods
        specified_range: User-specified range, if any

    Returns:
        Tuple of (x_min, x_max) for visualization
    """
    # If range is explicitly specified, use it
    if specified_range is not None:
        return specified_range

    # Check if it's a 2D function
    is_2d_function = function_name.startswith("2d_")

    # Use default range for the function type
    x_range = DEFAULT_RANGES.get(function_name, (-2, 2))

    # For 2D functions, we don't adjust the range based on initial points
    if is_2d_function:
        return x_range

    # For 1D functions, expand range if initial points are outside
    min_x, max_x = x_range
    padding = 0.5  # Add padding to ensure points aren't right at the edge

    for x0 in x0_values:
        if x0 < min_x:
            min_x = x0 - padding
        if x0 > max_x:
            max_x = x0 + padding

    return (min_x, max_x)


def create_method(
    method_name: str,
    config: NumericalMethodConfig,
    x0: Union[float, np.ndarray],
    x1: Optional[Union[float, np.ndarray]] = None,
) -> BaseNumericalMethod:
    """Create a numerical method instance.

    Args:
        method_name: Name of the method to create
        config: Configuration for the method
        x0: Initial point
        x1: Second point (for methods that need it)

    Returns:
        Initialized method instance
    """
    method_class = METHOD_MAP[method_name]

    try:
        # Methods that need two points
        if method_name in ["secant", "bisection", "regula_falsi"]:
            return method_class(config, x0, x1, record_initial_state=True)
        # Dual-capable methods don't pass second_derivative since we're in root-finding mode
        elif method_name == "bfgs":  # BFGS doesn't support record_initial_state
            return method_class(config, x0)
        else:
            return method_class(config, x0, record_initial_state=True)
    except TypeError as e:
        # Fallback if record_initial_state is not supported
        if "record_initial_state" in str(e):
            if method_name in ["secant", "bisection", "regula_falsi"]:
                return method_class(config, x0, x1)
            else:
                return method_class(config, x0)
        else:
            # Re-raise if it's a different TypeError
            raise


def run_methods(
    function_name: str,
    method_names: List[str],
    x0_values: List[float],
    tol: float = 1e-6,
    max_iter: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    step_length_method: Optional[str] = None,
    step_length_params: Optional[Dict[str, Any]] = None,
    visualize: bool = True,
    save_viz: Optional[str] = None,
    viz_format: str = "html",
    viz_3d: bool = False,
    save_data: Optional[Path] = None,
) -> Tuple[List[BaseNumericalMethod], pd.DataFrame]:
    """Run the selected numerical methods and return results.

    Args:
        function_name: Name of the function to use
        method_names: List of method names to run
        x0_values: Initial point(s)
        tol: Error tolerance
        max_iter: Maximum iterations
        x_range: Range for visualization
        step_length_method: Method for determining step length
        step_length_params: Parameters for step length method
        visualize: Whether to visualize results
        save_viz: Path to save visualizations (without extension)
        viz_format: Format for saving visualizations
        viz_3d: Whether to create 3D visualization for 2D functions
        save_data: Directory to save iteration history

    Returns:
        Tuple of (methods, summary_table)
    """
    # Get function and derivatives
    f, df = get_test_function(function_name)

    # Check if we're using a 2D function
    is_2d_function = function_name.startswith("2d_")

    # Determine visualization range
    x_range = determine_x_range(function_name, x0_values, x_range)

    # Create configuration
    config = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="root",  # Always root-finding mode for now
        tol=tol,
        max_iter=max_iter,
        x_range=x_range,
        step_length_method=step_length_method,
        step_length_params=step_length_params,
    )

    # Initialize methods
    methods: List[BaseNumericalMethod] = []
    for method_name in method_names:
        # Prepare initial points based on function dimensionality
        if is_2d_function:
            # For 2D functions, we need an array of [x, y]
            if len(x0_values) >= 2:
                x0_point = np.array(x0_values[:2])
            else:
                # If only one coordinate is provided, use [x0, x0]
                x0_point = np.array([x0_values[0], x0_values[0]])
        else:
            # For 1D functions, use the first x0
            x0_point = x0_values[0] if x0_values else None

        # Get appropriate initial points
        x0, x1 = get_safe_initial_points(
            f=config.func,
            x_range=config.x_range,
            method_name=method_name,
            x0=x0_point,
        )

        # Create and add the method
        method = create_method(method_name, config, x0, x1)
        methods.append(method)

    # Run all methods until convergence
    for method in methods:
        while not method.has_converged():
            method.step()

    # Generate summary table
    summary_data = []
    for method in methods:
        history = method.get_iteration_history()
        if not history:
            continue

        last_iter = history[-1]
        summary_data.append(
            {
                "Method": method.name,
                "Root/Minimum": (
                    f"{last_iter.x_new:.6f}"
                    if isinstance(last_iter.x_new, (float, int))
                    else str(last_iter.x_new)
                ),
                "Function Value": f"{last_iter.f_new:.6e}",
                "Iterations": method.iterations,
                "Error": f"{last_iter.error:.6e}",
                "Converged": method.has_converged(),
            }
        )

    summary_table = pd.DataFrame(summary_data)

    # Save iteration history if requested
    if save_data:
        save_iteration_history(methods, function_name, save_data)

    # Create and handle visualizations if requested
    if visualize:
        visualize_results(methods, config, function_name, save_viz, viz_format, viz_3d)

    return methods, summary_table


def save_iteration_history(
    methods: List[BaseNumericalMethod], function_name: str, save_dir: Path
):
    """Save iteration history to Excel files.

    Args:
        methods: List of method instances
        function_name: Name of the function
        save_dir: Directory to save files
    """
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on function
    filename = f"{function_name}_root_finding_history.xlsx"
    filepath = save_dir / filename

    # Create Excel writer
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for method in methods:
            history = method.get_iteration_history()
            if not history:
                continue

            # Prepare data for DataFrame
            data = []
            for iter_data in history:
                row = {
                    "Iteration": iter_data.iteration,
                    "x_old": (
                        f"{iter_data.x_old:.8f}"
                        if isinstance(iter_data.x_old, (float, int))
                        else str(iter_data.x_old)
                    ),
                    "f(x_old)": f"{iter_data.f_old:.8e}",
                    "x_new": (
                        f"{iter_data.x_new:.8f}"
                        if isinstance(iter_data.x_new, (float, int))
                        else str(iter_data.x_new)
                    ),
                    "f(x_new)": f"{iter_data.f_new:.8e}",
                    "|error|": f"{iter_data.error:.2e}",
                }

                # Add method-specific details
                for key, value in iter_data.details.items():
                    if isinstance(value, float):
                        row[key] = f"{value:.6e}"
                    else:
                        row[key] = str(value)
                data.append(row)

            # Create DataFrame and save to Excel sheet
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=method.name, index=False)

    print(f"Saved root-finding history to {filepath}")


def visualize_results(
    methods: List[BaseNumericalMethod],
    config: NumericalMethodConfig,
    function_name: str,
    save_viz: Optional[str] = None,
    viz_format: str = "html",
    viz_3d: bool = False,
):
    """Create and display visualizations of the results.

    Args:
        methods: List of method instances
        config: Method configuration
        function_name: Name of the function
        save_viz: Path to save visualizations (without extension)
        viz_format: Format for saving visualizations
        viz_3d: Whether to create 3D visualization for 2D functions
    """
    # Create visualization configuration
    vis_config = VisualizationConfig(
        style="white",
        context="talk",
        palette="viridis",
        point_size=10,
        dpi=100,
        show_legend=True,
        grid_alpha=0.2,
        title=f"Root Finding Methods Comparison: {function_name.capitalize()}",
        background_color="#FFFFFF",
        animation_duration=800,
        animation_transition=300,
    )

    # Create visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)

    # Generate and display interactive visualization
    visualizer.run_comparison()

    # Create 3D visualization if requested and if function is 2D
    fig_3d = None
    if viz_3d:
        fig_3d = visualizer.create_3d_visualization()
        if fig_3d:
            fig_3d.show()
        else:
            print("\nNote: 3D visualization is only available for 2D functions.")

    # Save visualizations if requested
    if save_viz:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_viz)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save main visualization
        visualizer.save_visualization(save_viz, format=viz_format)

        # Save 3D visualization if created
        if viz_3d and fig_3d:
            viz_3d_path = f"{save_viz}_3d"
            if viz_format == "html":
                fig_3d.write_html(f"{viz_3d_path}.html")
            else:
                fig_3d.write_image(f"{viz_3d_path}.{viz_format}", scale=2)
            print(f"3D visualization saved to {viz_3d_path}.{viz_format}")


def parse_args():
    """Parse and validate command-line arguments.

    Returns:
        Parsed and validated arguments
    """
    parser = argparse.ArgumentParser(
        description="Root Finding Method Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all methods
  python find_roots.py --all --function quadratic --x0 1.5
  
  # Compare specific methods
  python find_roots.py --methods newton secant --function quadratic --x0 1.5
  
  # Test on a challenging function
  python find_roots.py --methods newton bfgs --function multiple_roots --x0 0.5
  
  # Save visualizations in different formats
  python find_roots.py --methods newton secant --function cubic --x0 1.5 --save-viz results/cubic_comparison
  
  # Create 3D visualization for 2D functions
  python find_roots.py --methods newton bfgs --function 2d_example --x0 0.5 0.5 --viz-3d
  
  # Run without visualization
  python find_roots.py --methods newton --function quadratic --x0 1.5 --no-viz
""",
    )

    # Method selection group
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_MAP.keys()),
        help="Root finding methods to compare",
    )
    method_group.add_argument(
        "--all",
        action="store_true",
        help="Use all available root finding methods",
    )

    # Function selection
    parser.add_argument(
        "--function",
        choices=list(FUNCTION_MAP.keys()),
        default="quadratic",
        help="Test function to use",
    )

    # Initial points/intervals
    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",
        default=[1.5],
        help="Initial point(s) for the methods",
    )

    # Optional second point for methods that need it
    parser.add_argument(
        "--x1",
        type=float,
        nargs="+",
        help="Second point(s) for methods that need it (e.g., secant)",
    )

    # Tolerance and maximum iterations
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Error tolerance",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of iterations",
    )

    # X-axis range
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        help="X-axis range for visualization (default: auto-selected based on function)",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (JSON or YAML)",
    )

    # Add the save argument
    parser.add_argument(
        "--save",
        type=Path,
        help="Directory to save iteration history CSV files",
    )

    # Visualization options
    visualization_group = parser.add_argument_group("Visualization Options")

    visualization_group.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization (output to console only)",
    )

    visualization_group.add_argument(
        "--save-viz",
        type=str,
        help="Path to save visualizations (without extension)",
    )

    visualization_group.add_argument(
        "--viz-format",
        type=str,
        choices=["html", "png", "jpg", "svg", "pdf"],
        default="html",
        help="Format for saving visualizations",
    )

    visualization_group.add_argument(
        "--viz-3d",
        action="store_true",
        help="Create 3D visualization for 2D functions",
    )

    # Step length options
    step_length_group = parser.add_argument_group("Step Length Options")

    step_length_group.add_argument(
        "--step-length-method",
        choices=["fixed", "backtracking", "wolfe", "strong_wolfe", "goldstein"],
        help="Method to use for determining step length",
    )

    step_length_group.add_argument(
        "--step-length-params",
        type=json.loads,
        help='JSON string with step length parameters, e.g., \'{"alpha_init": 1.0, "rho": 0.5}\'',
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        try:
            config = load_config_file(args.config)
            # Update args with config file values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    parser.error(f"Unknown configuration option: {key}")
        except ValueError as e:
            parser.error(str(e))

    # If neither --methods nor --all is specified, default to newton
    if not args.methods and not args.all:
        args.methods = ["newton"]
    # If --all is specified, use all methods
    elif args.all:
        args.methods = list(METHOD_MAP.keys())

    return args


def main():
    """Main function to run root-finding methods."""
    # Parse command-line arguments
    args = parse_args()

    # Run the methods and get results
    methods, summary_table = run_methods(
        function_name=args.function,
        method_names=args.methods,
        x0_values=args.x0,
        tol=args.tol,
        max_iter=args.max_iter,
        x_range=args.xrange if hasattr(args, "xrange") else None,
        step_length_method=(
            args.step_length_method if hasattr(args, "step_length_method") else None
        ),
        step_length_params=(
            args.step_length_params if hasattr(args, "step_length_params") else None
        ),
        visualize=not args.no_viz,
        save_viz=args.save_viz,
        viz_format=args.viz_format,
        viz_3d=args.viz_3d,
        save_data=args.save,
    )

    # Print summary table to console
    print("\nRoot-Finding Results Summary:")
    print("-" * 50)
    print(summary_table.to_string(index=False))

    # If visualization is enabled, show plots
    if not args.no_viz:
        plt.ioff()
        plt.show(block=True)


if __name__ == "__main__":
    main()


# # Example commands:
# # basic root finding without visualization
# python find_roots.py --methods newton --function quadratic --x0 1.5 --no-viz

# # compare two methods with visualization
# python find_roots.py --methods newton secant --function cubic --x0 1.5

# # run with custom step length method
# python find_roots.py --methods newton --function quadratic --x0 1.5 --step-length-method wolfe

# # run with custom step length method and save visualization
# python find_roots.py --methods newton --function quadratic --x0 1.5 --step-length-method wolfe --save-viz results/quadratic_comparison

# # run with custom step length method and save visualization
# python find_roots.py --methods newton --function quadratic --x0 1.5 --step-length-method wolfe --save-viz results/quadratic_comparison

# # run with 3d visualization
# python find_roots.py --methods newton --function 2d_himmelblau --x0 1.0 1.0 --viz-3d

# run with 3d visualization and save visualization
# python find_roots.py --methods newton --function 2d_himmelblau --x0 1.0 1.0 --viz-3d --save-viz results/2d_himmelblau_comparison

# # Run the JSON config test
# python find_roots.py --config configs/root_finding.json

# # Run the YAML config test
# python find_roots.py --config configs/root_finding.yaml
