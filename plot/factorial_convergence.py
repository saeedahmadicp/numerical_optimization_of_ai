import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


def analyze_factorial_convergence():
    """Visualize convergence properties of the sequence x_k = 1/k!"""

    # Generate sequence and convergence ratios
    def generate_sequence(k_max):
        x = np.zeros(k_max)
        errors = np.zeros(k_max)
        superlinear_ratios = np.zeros(k_max - 1)
        quadratic_ratios = np.zeros(k_max - 1)

        for k in range(k_max):
            x[k] = 1 / factorial(k + 1)
            errors[k] = x[k]  # error = |x_k - 0|
            if k > 0:
                # Superlinear ratio: |x_{k+1}|/|x_k|
                superlinear_ratios[k - 1] = errors[k] / errors[k - 1]
                # Quadratic ratio: |x_{k+1}|/|x_k|^2
                quadratic_ratios[k - 1] = errors[k] / (errors[k - 1] ** 2)

        return x, errors, superlinear_ratios, quadratic_ratios

    k_max = 15
    x, errors, superlinear_ratios, quadratic_ratios = generate_sequence(k_max)
    k_range = np.arange(1, k_max + 1)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Common style settings
    plt.style.use("seaborn-v0_8-whitegrid")
    STYLE = {
        "grid.alpha": 0.2,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
    plt.rcParams.update(STYLE)

    # 1. Sequence convergence (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(k_range, x, "bo-", label="x_k = 1/k!")
    ax1.axhline(y=0, color="r", linestyle="--", label="Limit (x* = 0)")

    ax1.set_title("Sequence Convergence (Log Scale)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("x_k")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Superlinear ratio (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(k_range[1:], superlinear_ratios, "bo-", label="|x_{k+1}|/|x_k|")
    ax2.axhline(y=1, color="r", linestyle="--", label="Linear convergence threshold")

    ax2.set_title("Q-superlinear Convergence Ratio (Log Scale)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Quadratic ratio (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(k_range[1:], quadratic_ratios, "bo-", label="|x_{k+1}|/|x_k|²")

    ax3.set_title("Q-quadratic Ratio (Log Scale)")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Ratio")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Explanation (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    explanation = [
        "Convergence Analysis of x_k = 1/k!:",
        "",
        "Q-superlinear convergence:",
        "lim(|x_{k+1}|/|x_k|) = lim(1/(k+1)) = 0",
        "✓ Sequence converges Q-superlinearly",
        "",
        "Q-quadratic convergence:",
        "|x_{k+1}|/|x_k|² = k!/(k+1)",
        "lim(k!/(k+1)) = ∞",
        "✗ Sequence does not converge Q-quadratically",
        "",
        "Properties:",
        "• Very rapid convergence to 0",
        "• Superlinear but not quadratic",
        "• Ratio grows unboundedly",
    ]

    y_pos = 0.95
    for i, line in enumerate(explanation):
        if i == 0 or i == 2 or i == 6:
            ax4.text(0.05, y_pos, line, va="top", fontweight="bold")
        else:
            ax4.text(0.05, y_pos, line, va="top")
        y_pos -= 0.07

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_factorial_convergence()
