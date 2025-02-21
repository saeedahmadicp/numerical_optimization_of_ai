#!/usr/bin/env python3

"""Command-line interface for running and visualizing optimization methods."""

import argparse
from typing import Dict, Type, List
import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

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
        default=1e-5,
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
    parser.add_argument(
        "--fast", action="store_true", help="Enable fast animation mode"
    )
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

    # If neither --methods nor --all is specified, default to bfgs
    if not args.methods and not args.all:
        args.methods = ["bfgs"]
    # If --all is specified, use all methods
    elif args.all:
        args.methods = list(METHOD_MAP.keys())

    # Get function and derivatives
    f, df, d2f = get_minimization_function(args.function, with_second_derivative=True)

    # Determine if function is 2D based on number of initial coordinates
    is_2d = len(args.x0) == 2

    # Validate function and dimensions match
    if is_2d and args.function not in [
        "rosenbrock",
        "himmelblau",
        "rastrigin",
        "ackley",
        "beale",
        "booth",
    ]:
        parser.error(f"Function '{args.function}' is not a 2D function")
    elif not is_2d and args.function in [
        "rosenbrock",
        "himmelblau",
        "rastrigin",
        "ackley",
        "beale",
        "booth",
    ]:
        parser.error(
            f"Function '{args.function}' requires 2D input (--x0 requires two values)"
        )

    # Use appropriate range for the function
    if args.xrange is None:
        if args.function in DEFAULT_RANGES:
            args.xrange = DEFAULT_RANGES[args.function]
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
        is_2d=is_2d,
    )

    # Initialize methods
    methods: List[BaseNumericalMethod] = []
    for method_name in args.methods:
        method_class = METHOD_MAP[method_name]
        x0 = np.array(args.x0, dtype=float)

        if method_name in OPTIMIZATION_ONLY:
            methods.append(method_class(config, x0))
        else:
            if method_name in ["newton", "newton_hessian"]:
                methods.append(method_class(config, x0, second_derivative=d2f))
            else:
                methods.append(method_class(config, x0))

    # First, run all optimizations to completion
    print("\nRunning optimizations...")
    for method in methods:
        while (
            not method.has_converged()
            and len(method.get_iteration_history()) < config.max_iter
        ):
            method.step()

        # Print immediate feedback about convergence
        print(
            f"{method.name}: {'Converged' if method.has_converged() else 'Did not converge'}"
        )

    # Print final results summary
    print("\nOptimization Results Summary:")
    print("-" * 50)
    for method in methods:
        x_final = method.get_current_x()
        f_final = method.func(x_final)
        grad_final = np.linalg.norm(method.derivative(x_final))
        iterations = len(method.get_iteration_history())

        print(f"\n{method.name}:")
        print(f"  Iterations: {iterations}")
        if len(x_final) == 1:
            print(f"  Final x: {x_final[0]:.8f}")
        else:
            print(f"  Final x: [{', '.join(f'{x:.8f}' for x in x_final)}]")
        print(f"  Final f(x): {f_final:.8e}")
        print(f"  Final |∇f(x)|: {grad_final:.2e}")
        print(f"  Converged: {method.has_converged()}")

    # Save results if requested
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        filename = f"{args.function}_optimization_history.xlsx"
        filepath = args.save / filename

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for method in methods:
                history = method.get_iteration_history()
                if not history:
                    continue

                data = []
                for iter_data in history:
                    # Format x_old and x_new based on dimensionality
                    if len(iter_data.x_old) == 1:
                        x_old_str = f"{iter_data.x_old[0]:.8f}"
                        x_new_str = f"{iter_data.x_new[0]:.8f}"
                    else:
                        x_old_str = (
                            f"[{', '.join(f'{x:.8f}' for x in iter_data.x_old)}]"
                        )
                        x_new_str = (
                            f"[{', '.join(f'{x:.8f}' for x in iter_data.x_new)}]"
                        )

                    # Convert function values and error to float
                    f_old = (
                        float(iter_data.f_old)
                        if isinstance(iter_data.f_old, np.ndarray)
                        else iter_data.f_old
                    )
                    f_new = (
                        float(iter_data.f_new)
                        if isinstance(iter_data.f_new, np.ndarray)
                        else iter_data.f_new
                    )

                    error = float(np.linalg.norm(iter_data.error))

                    row = {
                        "Iteration": iter_data.iteration,
                        "x_old": x_old_str,
                        "f(x_old)": f"{f_old:.8e}",
                        "x_new": x_new_str,
                        "f(x_new)": f"{f_new:.8e}",
                        "|f'(x)|": f"{error:.2e}",
                    }

                    # Add method-specific details
                    for key, value in iter_data.details.items():
                        if isinstance(value, (float, np.floating)):
                            row[key] = f"{float(value):.6e}"
                        elif isinstance(value, np.ndarray):
                            row[key] = (
                                f"{np.array2string(value, precision=6, separator=', ')}"
                            )
                        else:
                            row[key] = str(value)
                    data.append(row)

                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=method.name, index=False)

            print(f"\nSaved optimization history to {filepath}")

    # Create visualization configuration with fast animation
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
        show_contour=True if is_2d else False,
        style="white",
        context="talk",
        palette="viridis",
        animation_interval=1,  # Always use fast mode for replay
    )

    # Now create and run visualizer with pre-computed results
    print("\nGenerating visualization...")
    visualizer = OptimizationVisualizer(config, methods, vis_config)
    visualizer.run_comparison()

    plt.ioff()
    plt.show(block=True)


if __name__ == "__main__":
    main()


# Example commands:
# Compare all methods:
# python minimize.py --all --function quadratic --x0 1.5 --save results/

# Compare gradient-based methods:
# python minimize.py --methods steepest bfgs newton --function quadratic --x0 1.5 --save output/

# Test on 2D functions:
# python minimize.py --methods bfgs newton --function rosenbrock --x0 -1.0 -1.0 --save data/
# python minimize.py --methods bfgs nelder_mead --function himmelblau --x0 1.0 1.0 --save data/

# Using config files:
# python minimize.py --config configs/optimization.yaml
# python minimize.py --config configs/optimization.json
