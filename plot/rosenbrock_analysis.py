import numpy as np
from function import FunctionPlotter, FunctionPlotConfig, PlotStyle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def analyze_rosenbrock():
    """Analyze and visualize the Rosenbrock function properties."""

    # Define Rosenbrock function and its derivatives
    def f(x1, x2):
        """Rosenbrock function: f(x) = 100(x₂-x₁²)² + (1-x₁)²"""
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2

    def grad_f(x):
        """Gradient of Rosenbrock function"""
        x1, x2 = x
        return np.array([-400 * x1 * (x2 - x1**2) - 2 * (1 - x1), 200 * (x2 - x1**2)])

    def hess_f(x):
        """Hessian of Rosenbrock function"""
        x1, x2 = x
        return np.array([[-400 * (x2 - 3 * x1**2) + 2, -400 * x1], [-400 * x1, 200]])

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1.2, 1])

    # 1. 3D Surface plot (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-1, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    surf = ax1.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)
    ax1.set_title("Rosenbrock Function Surface")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_zlabel("f(x₁,x₂)")

    # Mark the minimizer
    ax1.scatter([1], [1], [f(1, 1)], color="red", s=100, label="Minimizer (1,1)")

    # 2. Contour plot with gradient field (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    levels = np.logspace(-1, 3, 20)
    contours = ax2.contour(X1, X2, Z, levels=levels, cmap="viridis")
    ax2.clabel(contours, inline=True, fontsize=8)

    # Add gradient field
    x1_grid = np.linspace(-2, 2, 20)
    x2_grid = np.linspace(-1, 3, 20)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    U = np.zeros_like(X1_grid)
    V = np.zeros_like(X2_grid)

    for i in range(len(x1_grid)):
        for j in range(len(x2_grid)):
            grad = grad_f(np.array([X1_grid[i, j], X2_grid[i, j]]))
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            U[i, j] = -grad[0]  # Negative gradient for descent direction
            V[i, j] = -grad[1]

    ax2.quiver(X1_grid, X2_grid, U, V, alpha=0.3)
    ax2.plot(1, 1, "r*", markersize=15, label="Minimizer (1,1)")

    ax2.set_title("Contours and Gradient Field")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.legend()

    # 3. Hessian analysis at minimizer (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")

    # Compute Hessian at minimizer
    H_star = hess_f(np.array([1, 1]))
    eigenvals = np.linalg.eigvals(H_star)

    hessian_text = [
        "Hessian Analysis at x* = (1,1):",
        "",
        "H(x*) = ",
        f"⎡ {H_star[0,0]:>8.1f}  {H_star[0,1]:>8.1f} ⎤",
        f"⎣ {H_star[1,0]:>8.1f}  {H_star[1,1]:>8.1f} ⎦",
        "",
        "Eigenvalues:",
        f"λ₁ = {eigenvals[0]:.2f}",
        f"λ₂ = {eigenvals[1]:.2f}",
        "",
        "Since both eigenvalues are positive,",
        "H(x*) is positive definite.",
        "Therefore, (1,1) is a strict local minimizer.",
    ]

    y_pos = 0.95
    for i, line in enumerate(hessian_text):
        ax3.text(
            0.05,
            y_pos,
            line,
            va="top",
            fontsize=10,
            fontfamily="monospace" if 2 <= i <= 4 else None,
        )
        y_pos -= 0.08

    # 4. Function properties (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    properties = [
        "Rosenbrock Function Properties:",
        "",
        "f(x₁,x₂) = 100(x₂-x₁²)² + (1-x₁)²",
        "",
        "• Global minimizer: x* = (1,1)",
        "• Minimum value: f(x*) = 0",
        "",
        "• Function is non-convex",
        "• Has a narrow, parabolic valley",
        "• Valley is easy to find, hard to traverse",
        "• Often used to test optimization algorithms",
    ]

    y_pos = 0.95
    for i, line in enumerate(properties):
        ax4.text(0.05, y_pos, line, va="top", fontsize=10)
        y_pos -= 0.09

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_rosenbrock()
