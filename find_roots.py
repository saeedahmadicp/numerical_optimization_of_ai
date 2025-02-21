#!/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List
from tabulate import tabulate
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path
import pandas as pd
from openpyxl import Workbook

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

    # Use default range if not specified
    if args.xrange is None:
        args.xrange = DEFAULT_RANGES.get(args.function, (-2, 2))

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

        # Get appropriate initial points
        x0, x1 = get_safe_initial_points(
            f=config.func,
            x_range=config.x_range,
            method_name=method_name,
            x0=args.x0[0] if args.x0 else None,
        )

        if method_name in ROOT_FINDING_ONLY:
            # Methods that need two points
            if method_name in ["secant", "bisection", "regula_falsi"]:
                methods.append(method_class(config, x0, x1))
            else:
                methods.append(method_class(config, x0))
        else:
            # Dual-capable methods - don't pass second_derivative since we're in root-finding mode
            methods.append(method_class(config, x0))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
        style="white",
        context="talk",
        palette="viridis",
        point_size=100,
        dpi=100,
        show_legend=True,
        grid_alpha=0.2,
        title="Root Finding Methods Comparison",
        background_color="#FFFFFF",
    )

    # Create and run visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)
    visualizer.run_comparison()

    # Show iteration tables or save to Excel
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
                        "x_old": f"{iter_data.x_old:.8f}",
                        "f(x_old)": f"{iter_data.f_old:.8e}",
                        "x_new": f"{iter_data.x_new:.8f}",
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

            print(f"Saved all iteration histories to {filepath}")
    else:
        # If --save is not specified, print tables as before
        for method in methods:
            history = method.get_iteration_history()
            if not history:
                continue

            # Prepare data for DataFrame
            data = []
            for iter_data in history:
                row = {
                    "Iteration": iter_data.iteration,
                    "x_old": f"{iter_data.x_old:.8f}",
                    "f(x_old)": f"{iter_data.f_old:.8e}",
                    "x_new": f"{iter_data.x_new:.8f}",
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

            # Create DataFrame and print table
            df = pd.DataFrame(data)
            print(f"\n{method.name} Iteration History:")
            print(tabulate(df.values.tolist(), headers=df.columns, floatfmt=".8f"))
            print()

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


# # Config file examples
# # Basic example with YAML
# python find_roots.py --config configs/root_finding.yaml

# # Basic example with JSON
# python find_roots.py --config configs/root_finding.json

# # Override config file values with command line
# python find_roots.py --config configs/root_finding.yaml --tol 1.0e-8

# # Note: The test cases in root_finding_tests.yaml would need additional
# # functionality in find_roots.py to handle multiple test cases
