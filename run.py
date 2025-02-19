#!/usr/bin/env python3

"""Command-line interface for running and visualizing root-finding methods."""

import argparse
from typing import Dict, Type, List

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
from utils.funcs import get_test_function, AVAILABLE_FUNCTIONS

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
  # Compare Newton and BFGS on a simple quadratic
  python run.py --methods newton bfgs --function quadratic --x0 1.5
  
  # Test secant method on a challenging function
  python run.py --methods secant --function multiple_roots --x0 0.5 --x1 1.5
  
  # Compare multiple methods on trigonometric function
  python run.py --methods newton secant bisection --function sinusoidal
  
  # Test convergence on ill-conditioned function
  python run.py --methods newton_hessian bfgs --function ill_conditioned --tol 1e-10
""",
    )

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
        choices=AVAILABLE_FUNCTIONS,
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

    args = parser.parse_args()

    # Get function and derivative
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

        if method_name in ["secant", "bisection", "regula_falsi"]:
            # Methods requiring two points
            x1 = args.x1[0] if args.x1 else args.x0[0] + 0.5
            methods.append(method_class(config, args.x0[0], x1))
        else:
            # Methods requiring one point
            methods.append(method_class(config, args.x0[0]))

    # Create visualization configuration
    vis_config = VisualizationConfig(
        figsize=(12, 8),
        show_convergence=True,
        show_error=True,
    )

    # Create and run visualizer
    visualizer = RootFindingVisualizer(config, methods, vis_config)
    visualizer.run_comparison()


if __name__ == "__main__":
    main()


# python run.py --methods secant --function quadratic --x0 1.5 --x1 2.0 --tol 1e-6 --max-iter 100 --xrange -2 2
