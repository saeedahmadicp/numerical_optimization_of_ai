#!/usr/bin/env python3

"""Command-line interface for running and visualizing optimization methods."""

import argparse
from typing import Dict, Type, List
import numpy as np
import json
import yaml
from pathlib import Path

from algorithms.convex.protocols import BaseNumericalMethod, NumericalMethodConfig
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.newton_hessian import NewtonHessianMethod
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.nelder_mead import NelderMeadMethod
from algorithms.convex.powell import PowellMethod
from algorithms.convex.steepest_descent import SteepestDescentMethod
from plot.optimizer_viz import OptimizationVisualizer, VisualizationConfig
from utils.funcs import get_minimization_function, MINIMIZATION_MAP

# Map method names to their classes
METHOD_MAP: Dict[str, Type[BaseNumericalMethod]] = {
    # Optimization only methods
    "steepest": SteepestDescentMethod,
    "powell": PowellMethod,
    "nelder_mead": NelderMeadMethod,
    # Methods that can do both optimization and root-finding
    "newton": NewtonMethod,
    "newton_hessian": NewtonHessianMethod,
    "bfgs": BFGSMethod,
}

# Group methods by capability
OPTIMIZATION_ONLY = {"steepest", "powell", "nelder_mead"}
DUAL_METHODS = {"newton", "newton_hessian", "bfgs"}

# Default ranges for different function types
DEFAULT_RANGES = {
    "quadratic": (-3, 3),
    "rosenbrock": (-2, 2),
    "himmelblau": (-5, 5),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-5, 5),
    "sphere": (-5, 5),
    "booth": (-10, 10),
    "beale": (-4.5, 4.5),
    "matyas": (-10, 10),
}


def main():
    parser = argparse.ArgumentParser(
        description="Optimization Methods Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare gradient-based methods
  python minimize.py --methods steepest bfgs --function quadratic --x0 1.5

  # Test on Rosenbrock function
  python minimize.py --methods bfgs newton --function rosenbrock --x0 -1.0 -1.0

  # Compare all methods
  python minimize.py --all --function himmelblau --x0 1.0 1.0
""",
    )

    # Method selection group
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_MAP.keys()),
        help="Optimization methods to compare",
    )
    method_group.add_argument(
        "--all",
        action="store_true",
        help="Use all available optimization methods",
    )

    # Function and parameter arguments
    parser.add_argument(
        "--function",
        choices=list(MINIMIZATION_MAP.keys()),
        default="quadratic",
        help="Test function to minimize",
    )
    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",
        default=[1.5],
        help="Initial point coordinates",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        help="X-axis range for visualization",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (JSON or YAML)",
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

    # If neither --methods nor --all is specified, default to bfgs
    if not args.methods and not args.all:
        args.methods = ["bfgs"]
    # If --all is specified, use all methods
    elif args.all:
        args.methods = list(METHOD_MAP.keys())

    # Get function and derivatives
    f, df, d2f = get_minimization_function(args.function, with_second_derivative=True)

    # Use appropriate range for the function
    if args.xrange is None:
        if args.function == "quadratic":
            args.xrange = (-4, 4)  # Wider range for quadratic
        else:
            args.xrange = (-2, 2)  # Default range

    # Create configuration
    config = NumericalMethodConfig(
        func=f,
        derivative=df,
        method_type="optimize",
        tol=args.tol,
        max_iter=args.max_iter,
        x_range=args.xrange,
    )

    # Initialize methods
    methods: List[BaseNumericalMethod] = []
    for method_name in args.methods:
        method_class = METHOD_MAP[method_name]
        x0 = np.array(args.x0)

        if method_name in OPTIMIZATION_ONLY:
            methods.append(method_class(config, x0))
        else:
            # Dual-capable methods - pass second_derivative for methods that need it
            if method_name in ["newton", "newton_hessian"]:
                methods.append(method_class(config, x0, second_derivative=d2f))
            else:
                methods.append(method_class(config, x0))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
        show_contour=True,
        style="white",
        context="talk",
        palette="viridis",
    )

    # Create and run visualizer
    visualizer = OptimizationVisualizer(config, methods, vis_config)
    visualizer.run_comparison()


if __name__ == "__main__":
    main()


# Example commands:
# Compare all methods:
# python minimize.py --all --function quadratic --x0 1.5

# Compare gradient-based methods:
# python minimize.py --methods steepest bfgs newton --function quadratic --x0 1.5

# Test on 2D functions:
# python minimize.py --methods bfgs newton --function rosenbrock --x0 -1.0 -1.0
# python minimize.py --methods bfgs nelder_mead --function himmelblau --x0 1.0 1.0

# Using config files:
# python minimize.py --config configs/optimization.yaml
# python minimize.py --config configs/optimization.json
