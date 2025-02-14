"""Run script to compare optimization methods on the Rosenbrock function."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

from steepest_descent import steepest_descent
from newton import newton_method


# Global optimization parameters
PARAMS = {
    "tol": 1e-3,  # Gradient norm tolerance
    "max_iter": 10000,  # Maximum iterations
    "alpha_init": 0.01,  # Initial step size
    "x0": np.array([-1.0, 1.0]),  # Starting point
    "a": 400.0,  # Rosenbrock parameter
}


def rosenbrock(a: float = 100) -> Tuple[
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Create Rosenbrock function and its derivatives with parameter a.
    f(x) = (1-x₁)² + a(x₂-x₁²)²

    Returns:
        Tuple of (function, gradient, hessian)
    """

    def f(x: np.ndarray) -> float:
        return (1 - x[0]) ** 2 + a * (x[1] - x[0] ** 2) ** 2

    def grad_f(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                -2 * (1 - x[0]) - 4 * a * x[0] * (x[1] - x[0] ** 2),
                2 * a * (x[1] - x[0] ** 2),
            ]
        )

    def hess_f(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [2 - 4 * a * x[1] + 12 * a * x[0] ** 2, -4 * a * x[0]],
                [-4 * a * x[0], 2 * a],
            ]
        )

    return f, grad_f, hess_f


def plot_contours(
    f: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    histories: dict,
):
    """Plot contours of function with optimization trajectories."""
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi])) for xi in x] for yi in y])

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))

    colors = ["b", "r"]
    markers = ["o", "s"]

    for (name, history), color, marker in zip(histories.items(), colors, markers):
        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], f"{color}-", label=f"{name} path")
        plt.plot(
            history[0, 0], history[0, 1], f"{color}{marker}", label=f"{name} start"
        )
        plt.plot(history[-1, 0], history[-1, 1], f"{color}*", label=f"{name} end")

    plt.plot(1, 1, "g*", markersize=15, label="Global minimum")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Optimization Trajectories on Rosenbrock Function")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_convergence(f: Callable, histories: dict):
    """Plot convergence of function values."""
    plt.figure(figsize=(10, 6))

    for (name, history), color in zip(histories.items(), ["b", "r"]):
        values = [f(x) for x in history]
        plt.semilogy(values, f"{color}-", label=name)

    plt.xlabel("Iteration")
    plt.ylabel("f(x) (log scale)")
    plt.title("Convergence of Function Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_results(
    name: str, x: np.ndarray, f: Callable, history: list, x_star: np.ndarray
):
    """Print optimization results in a consistent format."""
    print(f"\n{name} Results:")
    print(f"Solution: {x}")
    print(f"Function value: {f(x):.2e}")
    print(f"Iterations: {len(history)-1}")
    print(f"Distance to optimum: {np.linalg.norm(x - x_star):.2e}")


def main():
    # Create Rosenbrock function
    f, grad_f, hess_f = rosenbrock(PARAMS["a"])
    x_star = np.array([1.0, 1.0])  # Known optimum

    # Run optimization methods
    print("\nRunning optimizations...")

    # Steepest Descent
    x_sd, hist_sd, f_hist_sd = steepest_descent(
        f,
        grad_f,
        PARAMS["x0"],
        tol=PARAMS["tol"],
        max_iter=PARAMS["max_iter"],
        alpha_init=PARAMS["alpha_init"],
    )
    print_results("Steepest Descent", x_sd, f, hist_sd, x_star)

    # Newton's Method
    x_nt, hist_nt, f_hist_nt = newton_method(
        f,
        grad_f,
        hess_f,
        PARAMS["x0"],
        tol=PARAMS["tol"],
    )
    print_results("Newton's Method", x_nt, f, hist_nt, x_star)

    # Plot results
    histories = {"Steepest Descent": hist_sd, "Newton's Method": hist_nt}

    print("\nGenerating plots...")
    # Contour plot with optimization paths
    plot_contours(f, (-2, 2), (-1, 3), histories)

    # Convergence plot
    plot_convergence(f, histories)


if __name__ == "__main__":
    main()
