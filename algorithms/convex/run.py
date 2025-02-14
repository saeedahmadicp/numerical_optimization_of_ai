"""Run script to compare optimization methods on the Rosenbrock function."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

from steepest_descent import steepest_descent
from newton import newton_method


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


def main():
    # Create Rosenbrock function
    a = 100  # Standard value
    f, grad_f, hess_f = rosenbrock(a)

    # Initial point
    x0 = np.array([-1.0, 1.0])

    # Run optimization methods with stricter tolerance
    print("Running Steepest Descent...")
    x_sd, hist_sd, f_hist_sd = steepest_descent(
        f,
        grad_f,
        x0,
        tol=1e-8,
        max_iter=10000,
        alpha_init=0.01,  # Smaller but constant initial step
    )
    print(f"Solution: {x_sd}")
    print(f"Function value: {f(x_sd)}")
    print(f"Iterations: {len(hist_sd)-1}\n")

    print("Running Newton's Method...")
    x_nt, hist_nt, f_hist_nt = newton_method(
        f, grad_f, hess_f, x0, tol=1e-8  # Stricter tolerance
    )
    print(f"Solution: {x_nt}")
    print(f"Function value: {f(x_nt)}")
    print(f"Iterations: {len(hist_nt)-1}\n")

    # Plot results
    histories = {"Steepest Descent": hist_sd, "Newton's Method": hist_nt}

    # Contour plot with optimization paths
    plot_contours(f, (-2, 2), (-1, 3), histories)

    # Convergence plot
    plot_convergence(f, histories)

    # Print distance to optimum
    x_star = np.array([1.0, 1.0])
    print("\nDistance to optimum:")
    print(f"Steepest Descent: {np.linalg.norm(x_sd - x_star)}")
    print(f"Newton's Method: {np.linalg.norm(x_nt - x_star)}")


if __name__ == "__main__":
    main()
