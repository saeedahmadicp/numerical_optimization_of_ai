import numpy as np
from function import FunctionPlotter, FunctionPlotConfig, PlotStyle
import matplotlib.pyplot as plt


def visualize_taylors_theorem():
    """Visualize Taylor's Theorem using the road trip analogy."""

    # Define a simple nonlinear function and its derivatives
    def f(x):
        return np.sin(x) + 0.5 * x**2

    def df(x):
        return np.cos(x) + x

    def d2f(x):
        return -np.sin(x) + 1

    # Points for visualization
    x0 = 1.0  # Base point
    p = 1.5  # Step

    # Calculate t* using mean value theorem
    t = np.linspace(0, 1, 100)
    grad_values = [
        df(x0 + t_i * p) * p for t_i in t
    ]  # Include p in gradient calculation
    avg_value = np.trapz(grad_values, t)
    t_star = np.interp(avg_value, grad_values, t)
    x_star = x0 + t_star * p

    # Create compact figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.2, wspace=0.25)

    # Common style settings
    STYLE = {
        "grid.alpha": 0.2,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.4,
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(STYLE)

    # 1. The "Road" - Path from x to x+p (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    x_range = np.linspace(x0 - 0.3, x0 + p + 0.3, 100)

    ax1.plot(x_range, [f(x) for x in x_range], "b-", label="f(x)", alpha=0.2)
    ax1.plot(x0, f(x0), "ko", markersize=6, label="x")
    ax1.plot(x0 + p, f(x0 + p), "go", markersize=6, label="x+p")
    ax1.plot(x_star, f(x_star), "mo", markersize=6, label="x+t*p")

    ax1.set_title("Path from x to x+p", pad=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=2, columnspacing=1)

    # 2. The "Speed" profile (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, grad_values, "b-", label="Speed")
    ax2.axhline(y=avg_value, color="r", linestyle="--", label="Avg")
    ax2.plot(t_star, avg_value, "mo", markersize=6, label=f"t*={t_star:.3f}")
    ax2.fill_between(t, grad_values, alpha=0.1)

    ax2.set_title("Speed Profile", pad=5)
    ax2.set_xlabel("t")
    ax2.set_ylabel("∇f(x+tp)ᵀp")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=2, columnspacing=1)

    # 3. The accumulated change (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    distances = [np.trapz(grad_values[: i + 1], t[: i + 1]) for i in range(len(t))]

    ax3.plot(t, distances, "b-", label="Δf(t)")
    ax3.plot(1, distances[-1], "go", markersize=6, label="f(x+p)-f(x)")

    ax3.set_title("Accumulated Change", pad=5)
    ax3.set_xlabel("t")
    ax3.set_ylabel("Change in f")
    ax3.legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1, columnspacing=1)

    # 4. Annotations (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    explanation = [
        "Road Trip Analogy:",
        "• Journey: x → x+p",
        "• Speed = ∇f(x+tp)ᵀp",
        f"• At t* = {t_star:.3f}:",
        "  speed = average speed",
        "",
        "Mean Value Theorem:",
        r"$f(x+p) - f(x) = \nabla f(x+t^*p)^Tp$",
        "Total Δf = Speed(t*) × Length",
    ]

    y_pos = 0.95
    for i, line in enumerate(explanation):
        if i == 0 or i == 6:
            ax4.text(0.05, y_pos, line, va="top", fontweight="bold", fontsize=9)
        else:
            ax4.text(0.05, y_pos, line, va="top", fontsize=9)
        y_pos -= 0.11

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.show()


if __name__ == "__main__":
    visualize_taylors_theorem()
