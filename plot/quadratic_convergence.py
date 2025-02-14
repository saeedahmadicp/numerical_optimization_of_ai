import numpy as np
import matplotlib.pyplot as plt


def analyze_quadratic_convergence():
    """Visualize Q-quadratic convergence of the sequence x_k = 1 + (0.5)^(2^k)."""

    # Generate sequence and errors
    def generate_sequence(k_max):
        x = np.zeros(k_max)
        errors = np.zeros(k_max)
        ratios = np.zeros(k_max - 1)

        for k in range(k_max):
            x[k] = 1 + 0.5 ** (2**k)
            errors[k] = 0.5 ** (2**k)  # error = x_k - 1
            if k > 0:
                ratios[k - 1] = errors[k] / (errors[k - 1] ** 2)

        return x, errors, ratios

    k_max = 10
    x, errors, ratios = generate_sequence(k_max)
    k_range = np.arange(k_max)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Common style settings
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Sequence convergence (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(k_range, x, "bo-", label="x_k")
    ax1.axhline(y=1, color="r", linestyle="--", label="Limit (x* = 1)")

    ax1.set_title("Sequence Convergence")
    ax1.set_xlabel("k")
    ax1.set_ylabel("x_k")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error decay (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(k_range, errors, "bo-", label="|x_k - x*|")

    ax2.set_title("Error Decay (Log Scale)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Q-quadratic ratio (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(k_range[1:], ratios, "bo-", label="Ratio")
    ax3.axhline(y=1, color="r", linestyle="--", label="M = 1")

    ax3.set_title("Q-quadratic Convergence Ratio")
    ax3.set_xlabel("k")
    ax3.set_ylabel("|x_{k+1} - x*| / |x_k - x*|²")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Explanation (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    explanation = [
        "Q-quadratic Convergence Analysis:",
        "",
        "Sequence: x_k = 1 + (0.5)^(2^k)",
        "Error: e_k = x_k - 1 = (0.5)^(2^k)",
        "",
        "Q-quadratic convergence ratio:",
        "e_{k+1}/e_k² = 1 for all k",
        "",
        "Properties:",
        "• Converges to x* = 1",
        "• Error decreases quadratically",
        "• Constant ratio M = 1",
        "• Very rapid convergence",
    ]

    y_pos = 0.95
    for i, line in enumerate(explanation):
        if i == 0:
            ax4.text(0.05, y_pos, line, va="top", fontweight="bold")
        else:
            ax4.text(0.05, y_pos, line, va="top")
        y_pos -= 0.08

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_quadratic_convergence()
