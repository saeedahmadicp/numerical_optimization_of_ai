#!/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List
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


def main():
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

    # Tolerance
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Error tolerance",
    )

    # Maximum iterations
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

    # Add visualization saving options
    parser.add_argument(
        "--save-viz",
        type=str,
        help="Path to save visualizations (without extension)",
    )

    parser.add_argument(
        "--viz-format",
        type=str,
        choices=["html", "png", "jpg", "svg", "pdf"],
        default="html",
        help="Format for saving visualizations",
    )

    # Add 3D visualization option
    parser.add_argument(
        "--viz-3d",
        action="store_true",
        help="Create 3D visualization for 2D functions",
    )

    # Add option to show/hide convergence and error plots
    parser.add_argument(
        "--show-convergence",
        action="store_true",
        default=True,
        help="Show convergence plot",
    )

    parser.add_argument(
        "--show-error",
        action="store_true",
        default=True,
        help="Show error plot",
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        if not args.config.exists():
            parser.error(f"Configuration file not found: {args.config}")

        try:
            with open(args.config) as f:
                if args.config.suffix.lower() in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                elif args.config.suffix.lower() == ".json":
                    config = json.load(f)
                else:
                    parser.error("Configuration file must be .yaml, .yml, or .json")

                # Update args with config file values
                for key, value in config.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                    else:
                        parser.error(f"Unknown configuration option: {key}")
        except Exception as e:
            parser.error(f"Error reading configuration file: {e}")

    # If neither --methods nor --all is specified, default to newton
    if not args.methods and not args.all:
        args.methods = ["newton"]
    # If --all is specified, use all methods
    elif args.all:
        args.methods = list(METHOD_MAP.keys())

    # Get function and derivatives
    f, df = get_test_function(args.function)
    d2f = None  # Second derivative only needed for some methods

    # Check if we're using a 2D function
    is_2d_function = args.function.startswith("2d_")

    # Use default range if not specified
    if args.xrange is None:
        if is_2d_function:
            # Default range for 2D functions
            args.xrange = (-5, 5)
        else:
            args.xrange = DEFAULT_RANGES.get(args.function, (-2, 2))

            # Check if initial guess is outside the default range and expand if needed
            if not is_2d_function:  # Only for 1D functions
                for x0_value in args.x0:
                    # Add padding to ensure points aren't right at the edge
                    padding = 0.5
                    if x0_value < args.xrange[0]:
                        args.xrange = (x0_value - padding, args.xrange[1])
                    if x0_value > args.xrange[1]:
                        args.xrange = (args.xrange[0], x0_value + padding)

    # Create configuration
    config = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="root",  # Always root-finding mode
        tol=args.tol,
        max_iter=args.max_iter,
        x_range=args.xrange,
    )

    # Initialize methods
    methods: List[BaseNumericalMethod] = []
    for method_name in args.methods:
        method_class = METHOD_MAP[method_name]

        # Prepare initial points based on function dimensionality
        if is_2d_function:
            # For 2D functions, we need an array of [x, y]
            if len(args.x0) >= 2:
                x0_point = np.array(args.x0[:2])
            else:
                # If only one coordinate is provided, use [x0, x0]
                x0_point = np.array([args.x0[0], args.x0[0]])
        else:
            # For 1D functions, use the first x0
            x0_point = args.x0[0] if args.x0 else None

        # Get appropriate initial points
        x0, x1 = get_safe_initial_points(
            f=config.func,
            x_range=config.x_range,
            method_name=method_name,
            x0=x0_point,
        )

        # Initialize the method with appropriate parameters
        try:
            if method_name in ROOT_FINDING_ONLY:
                # Methods that need two points
                if method_name in ["secant", "bisection", "regula_falsi"]:
                    methods.append(
                        method_class(config, x0, x1, record_initial_state=True)
                    )
                else:
                    methods.append(method_class(config, x0, record_initial_state=True))
            else:
                # Dual-capable methods - don't pass second_derivative since we're in root-finding mode
                # For BFGS and potentially other methods that don't accept record_initial_state
                if method_name == "bfgs":
                    methods.append(method_class(config, x0))
                else:
                    methods.append(method_class(config, x0, record_initial_state=True))
        except TypeError as e:
            # Fallback if record_initial_state is not supported
            if "record_initial_state" in str(e):
                if method_name in ["secant", "bisection", "regula_falsi"]:
                    methods.append(method_class(config, x0, x1))
                else:
                    methods.append(method_class(config, x0))
            else:
                # Re-raise if it's a different TypeError
                raise

    # Create visualization configuration
    vis_config = VisualizationConfig(
        show_convergence=args.show_convergence,
        show_error=args.show_error,
        style="white",
        context="talk",
        palette="viridis",
        point_size=10,
        dpi=100,
        show_legend=True,
        grid_alpha=0.2,
        title=f"Root Finding Methods Comparison: {args.function.capitalize()}",
        background_color="#FFFFFF",
        animation_duration=800,
        animation_transition=300,
    )

    # Create visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)

    # Run comparison (show interactive visualization)
    visualizer.run_comparison()

    # Generate and display summary table
    summary_table = visualizer.generate_summary_table()
    print("\nRoot-Finding Results Summary:")
    print("-" * 50)
    print(summary_table.to_string(index=False))

    # Create 3D visualization if requested and if function is 2D
    if args.viz_3d:
        fig_3d = visualizer.create_3d_visualization()
        if fig_3d:
            fig_3d.show()
        else:
            print("\nNote: 3D visualization is only available for 2D functions.")

    # Save visualizations if requested
    if args.save_viz:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_viz)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save main visualization
        visualizer.save_visualization(args.save_viz, format=args.viz_format)

        # Save 3D visualization if created
        if (
            args.viz_3d
            and hasattr(visualizer, "dimensions")
            and visualizer.dimensions == 2
        ):
            viz_3d_path = f"{args.save_viz}_3d"
            fig_3d = visualizer.create_3d_visualization()
            if fig_3d:
                if args.viz_format == "html":
                    fig_3d.write_html(f"{viz_3d_path}.html")
                else:
                    fig_3d.write_image(f"{viz_3d_path}.{args.viz_format}", scale=2)
                print(f"3D visualization saved to {viz_3d_path}.{args.viz_format}")

    # Save iteration history to Excel if requested
    if args.save:
        # Create directory if it doesn't exist
        args.save.mkdir(parents=True, exist_ok=True)

        # Generate filename based on function
        filename = f"{args.function}_root_finding_history.xlsx"
        filepath = args.save / filename

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

    # Show visualizations (if they were created)
    plt.ioff()
    plt.show(block=True)


if __name__ == "__main__":
    main()


# Example commands:
# Compare all methods:
# python find_roots.py --all --function quadratic --x0 1.5 --save results/

# Compare root-finding only methods:
# python find_roots.py --methods bisection secant regula_falsi --function quadratic --x0 1.5 --save output/

# Compare dual-capable methods:
# python find_roots.py --methods newton newton_hessian bfgs --function multiple_roots --x0 0.5 --save data/

# Save interactive visualizations:
# python find_roots.py --methods newton secant --function cubic --x0 1.5 --save-viz results/cubic_comparison

# Create 3D visualizations for 2D functions:
# python find_roots.py --methods newton bfgs --function 2d_example --x0 0.5 0.5 --viz-3d

# Config file examples
# # Basic example with YAML
# python find_roots.py --config configs/root_finding.yaml

# # Basic example with JSON
# python find_roots.py --config configs/root_finding.json

# # Override config file values with command line
# python find_roots.py --config configs/root_finding.yaml --tol 1.0e-8

# python find_roots.py --function 2d_himmelblau --methods newton --x0 1.0 1.0 --viz-3d

# python find_roots.py --function 2d_rastrigin --methods newton --x0 0.5 0.5 --viz-3d

# python find_roots.py --function 2d_himmelblau --methods newton bfgs --x0 3.0 2.0 --viz-3d
