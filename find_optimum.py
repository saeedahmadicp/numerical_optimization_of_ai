#!/usr/bin/env python3

"""Command-line interface for running and visualizing optimization methods."""

import argparse
from typing import Dict, Type, List
import numpy as np

from algorithms.convex.protocols import BaseRootFinder, RootFinderConfig
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.nelder_mead import NelderMeadMethod
from algorithms.convex.powell import PowellMethod
from algorithms.convex.steepest_descent import SteepestDescentMethod
from plot.optimizer import OptimizationVisualizer, VisualizationConfig
from utils.funcs import get_minimization_function, MINIMIZATION_MAP, MINIMIZATION_RANGES

# Map method names to their classes - only optimization methods
METHOD_MAP: Dict[str, Type[BaseRootFinder]] = {
    "steepest": SteepestDescentMethod,
    "bfgs": BFGSMethod,
    "powell": PowellMethod,
    "nelder_mead": NelderMeadMethod,
}


def main():
    parser = argparse.ArgumentParser(
        description="Optimization Methods Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare gradient-based methods on quadratic function
  python find_optimum.py --methods steepest bfgs --function quadratic --x0 1.5

  # Test BFGS on Rosenbrock function
  python find_optimum.py --methods bfgs --function rosenbrock --x0 -1.0 -1.0

  # Compare multiple methods on Himmelblau function
  python find_optimum.py --methods bfgs nelder_mead powell --function himmelblau --x0 1.0 1.0

  # Test convergence on Rastrigin function
  python find_optimum.py --methods nelder_mead bfgs --function rastrigin --tol 1e-10
""",
    )

    # Method selection
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_MAP.keys()),
        default=["bfgs"],
        help="Optimization methods to compare",
    )

    # Function selection
    parser.add_argument(
        "--function",
        choices=list(MINIMIZATION_MAP.keys()),  # Use optimization functions
        default="quadratic",
        help="Test function to use",
    )

    # Initial point(s)
    parser.add_argument(
        "--x0",
        type=float,
        nargs="+",
        default=[1.5],
        help="Initial point coordinates (1 value for 1D, 2 values for 2D)",
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

    args = parser.parse_args()

    # Get function and derivative for optimization
    f, df = get_minimization_function(args.function)

    # Use default range if not specified
    if args.xrange is None:
        args.xrange = MINIMIZATION_RANGES.get(args.function, (-2, 2))

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
        # Convert x0 to numpy array for multi-dimensional functions
        x0 = np.array(args.x0)
        methods.append(method_class(config, x0))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
        show_contour=True,  # For 2D functions
        title="Optimization Methods Comparison",
    )

    # Create and run visualizer
    visualizer = OptimizationVisualizer(config, methods, vis_config)
    visualizer.run_comparison()


if __name__ == "__main__":
    main()


# # 1D optimization - will use 1D plot
# python find_optimum.py --methods bfgs steepest --function quadratic --x0 1.5

# # 2D optimization - will use 2D plot
# python find_optimum.py --methods bfgs nelder_mead --function rosenbrock --x0 -1.0 -1.0
