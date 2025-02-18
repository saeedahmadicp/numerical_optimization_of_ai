# !/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List
import numpy as np

from algorithms.convex.protocols import BaseRootFinder, RootFinderConfig
from algorithms.convex.newton import NewtonMethod
from algorithms.convex.quasi_newton import BFGSMethod
from algorithms.convex.secant import SecantMethod
from algorithms.convex.bisection import BisectionMethod
from algorithms.convex.regula_falsi import RegulaFalsiMethod
from algorithms.convex.nelder_mead import NelderMeadMethod
from algorithms.convex.powell import PowellMethod
from algorithms.convex.steepest_descent import SteepestDescentMethod
from algorithms.convex.newton_hessian import NewtonHessianMethod
from plot.rootfinder import RootFindingVisualizer, VisualizationConfig

# Map method names to their classes
METHOD_MAP: Dict[str, Type[BaseRootFinder]] = {
    "newton": NewtonMethod,
    "bfgs": BFGSMethod,
    "secant": SecantMethod,
    "bisection": BisectionMethod,
    "regula_falsi": RegulaFalsiMethod,
    "nelder_mead": NelderMeadMethod,
    "powell": PowellMethod,
    "steepest": SteepestDescentMethod,
    "newton_hessian": NewtonHessianMethod,
}


def create_test_function(func_name: str):
    """Create test function and its derivative."""
    if func_name == "quadratic":
        return lambda x: x**2 - 2, lambda x: 2 * x
    elif func_name == "cubic":
        return lambda x: x**3 - x - 2, lambda x: 3 * x**2 - 1
    elif func_name == "sin":
        return lambda x: np.sin(x) - 0.5, lambda x: np.cos(x)
    else:  # default to quadratic
        return lambda x: x**2 - 2, lambda x: 2 * x


def main():
    parser = argparse.ArgumentParser(description="Root Finding Method Visualizer")

    # Method selection
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_MAP.keys()),
        default=["newton"],
        help="Root finding methods to compare",
    )

    # Function selection
    parser.add_argument(
        "--function",
        choices=["quadratic", "cubic", "sin"],
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

    # Optional second point for methods that need it (like secant)
    parser.add_argument(
        "--x1",
        type=float,
        nargs="+",
        help="Second point(s) for methods that need it (e.g., secant)",
    )

    # Tolerance
    parser.add_argument("--tol", type=float, default=1e-6, help="Error tolerance")

    # Maximum iterations
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum number of iterations"
    )

    # X-axis range
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=[-2, 2],
        help="X-axis range for visualization",
    )

    args = parser.parse_args()

    # Create function and derivative
    f, df = create_test_function(args.function)

    # Create configuration
    config = RootFinderConfig(
        func=f, derivative=df, tol=args.tol, max_iter=args.max_iter, x_range=args.xrange
    )

    # Initialize methods
    methods: List[BaseRootFinder] = []
    for method_name in args.methods:
        method_class = METHOD_MAP[method_name]

        if method_name in ["secant", "bisection", "regula_falsi"]:
            # Methods requiring two points
            x1 = args.x1[0] if args.x1 else args.x0[0] + 0.5
            methods.append(method_class(config, args.x0[0], x1))
        else:
            # Methods requiring one point
            methods.append(method_class(config, args.x0[0]))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8), show_convergence=True, show_error=True
    )

    # Create and run visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)
    visualizer.run_comparison()


if __name__ == "__main__":
    main()


# python run.py --methods secant --function quadratic --x0 1.5 --x1 2.0 --tol 1e-6 --max-iter 100 --xrange -2 2
