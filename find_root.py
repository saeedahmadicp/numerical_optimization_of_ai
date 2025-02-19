#!/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List
from tabulate import tabulate

from algorithms.convex.protocols import BaseRootFinder, RootFinderConfig
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.newton_hessian import NewtonHessianMethod
from algorithms.convex.secant import SecantMethod
from algorithms.convex.bisection import BisectionMethod
from algorithms.convex.regula_falsi import RegulaFalsiMethod
from plot.rootfinder import RootFindingVisualizer, VisualizationConfig
from utils.funcs import get_test_function, FUNCTION_MAP
from utils.midpoint import get_safe_initial_points

# Map method names to their classes - only root finding methods
METHOD_MAP: Dict[str, Type[BaseRootFinder]] = {
    "newton": NewtonMethod,
    "newton_hessian": NewtonHessianMethod,
    "secant": SecantMethod,
    "bisection": BisectionMethod,
    "regula_falsi": RegulaFalsiMethod,
}

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
  python find_root.py --all --function quadratic --x0 1.5
  
  # Compare specific methods
  python find_root.py --methods newton secant --function quadratic --x0 1.5
  
  # Test on a challenging function
  python find_root.py --methods newton --function multiple_roots --x0 0.5
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

    # Show detailed iteration table at the end
    parser.add_argument(
        "--show-table",
        "-t",
        action="store_true",
        help="Show detailed iteration table at the end",
    )

    args = parser.parse_args()

    # If neither --methods nor --all is specified, default to newton
    if not args.methods and not args.all:
        args.methods = ["newton"]
    # If --all is specified, use all methods
    elif args.all:
        args.methods = list(METHOD_MAP.keys())

    # Get function and derivative for root finding
    f, df = get_test_function(args.function)

    # Use default range if not specified
    if args.xrange is None:
        args.xrange = DEFAULT_RANGES.get(args.function, (-2, 2))

    # Create configuration
    config = RootFinderConfig(
        func=f,
        derivative=df,
        tol=args.tol,
        max_iter=args.max_iter,
        x_range=args.xrange,
    )

    # Initialize methods
    methods: List[BaseRootFinder] = []
    for method_name in args.methods:
        method_class = METHOD_MAP[method_name]

        # Get appropriate initial points for this method
        x0, x1 = get_safe_initial_points(
            f=config.func,
            x_range=config.x_range,
            method_name=method_name,
            x0=args.x0[0] if args.x0 else None,
        )

        if method_name in ["secant", "bisection", "regula_falsi"]:
            methods.append(method_class(config, x0, x1))
        else:
            methods.append(method_class(config, x0))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
        style="darkgrid",
        context="notebook",
        palette="husl",
        point_size=100,
        dpi=100,
        show_legend=True,
        grid_alpha=0.3,
        title="Root Finding Methods Comparison",
        background_color="#2E3440",
    )

    # Create and run visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)
    visualizer.run_comparison()

    # Show iteration tables
    for method in methods:
        history = method.get_iteration_history()
        if not history:
            continue

        print(f"\n{method.name} Iteration History:")
        table_data = []
        for iter_data in history:
            row = [
                iter_data.iteration,
                f"{iter_data.x_old:.8f}",
                f"{iter_data.f_old:.8e}",
                f"{iter_data.x_new:.8f}",
                f"{iter_data.f_new:.8e}",
                f"{iter_data.error:.2e}",
            ]
            # Add method-specific details
            for key, value in iter_data.details.items():
                if isinstance(value, float):
                    row.append(f"{value:.6e}")
                else:
                    row.append(str(value))
            table_data.append(row)

        # Create headers based on method details
        headers = ["Iter", "x_old", "f(x_old)", "x_new", "f(x_new)", "|error|"]
        if history[0].details:
            headers.extend(history[0].details.keys())

        print(tabulate(table_data, headers=headers, floatfmt=".8f"))
        print()


if __name__ == "__main__":
    main()


# # Use all available methods
# python find_root.py --all --function quadratic --x0 1.5

# # Test the stiff function
# python find_root.py --methods newton --function stiff --x0 0.5

# # Test the multiple roots function
# python find_root.py --methods newton --function multiple_roots --x0 0.9

# # Test the ill-conditioned function
# python find_root.py --methods newton --function ill_conditioned --x0 0.5

# # Test the trig polynomial
# python find_root.py --methods newton --function trig_polynomial --x0 1.0
