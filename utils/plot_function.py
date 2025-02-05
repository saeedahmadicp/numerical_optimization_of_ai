# utils/plot_function.py

import numpy as np
import matplotlib.pyplot as plt


def plot_convex_inequality(f, x, y, alpha=0.5, points=100):
    """
    Visualize the convex inequality: f(αx+(1-α)y) ≤ αf(x)+(1-α)f(y)

    Parameters:
    -----------
    f : callable
        The convex function to plot
    x : float
        First x-coordinate
    y : float
        Second x-coordinate
    alpha : float, optional
        The mixing parameter α between 0 and 1 (default: 0.5)
    points : int, optional
        Number of points for smooth curve plotting (default: 100)
    """
    # Generate points for smooth curve
    x_range = np.linspace(min(x, y) - 0.5, max(x, y) + 0.5, points)
    y_range = [f(xi) for xi in x_range]

    # Calculate the convex combination point
    z = alpha * x + (1 - alpha) * y

    # Calculate function values
    fx = f(x)
    fy = f(y)
    fz = f(z)

    # Calculate the linear interpolation value
    f_linear = alpha * fx + (1 - alpha) * fy

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the function curve
    plt.plot(x_range, y_range, "b-", label="f(x)")

    # Plot the line segment connecting (x,f(x)) and (y,f(y))
    plt.plot([x, y], [fx, fy], "r--", label="Linear interpolation")

    # Plot points
    plt.plot(x, fx, "ro")
    plt.plot(y, fy, "ro")
    plt.plot(z, fz, "go")
    plt.plot(z, f_linear, "mo")

    # Add vertical line to show the difference
    plt.vlines(z, fz, f_linear, colors="g", linestyles="dotted")

    # Add annotations with LaTeX
    # Point annotations
    plt.annotate(f"$(x, f(x))$", (x, fx), xytext=(10, 10), textcoords="offset points")
    plt.annotate(f"$(y, f(y))$", (y, fy), xytext=(10, 10), textcoords="offset points")

    # x-axis labels
    plt.annotate(
        "$x$", (x, 0), xytext=(0, -20), textcoords="offset points", ha="center"
    )
    plt.annotate(
        "$y$", (y, 0), xytext=(0, -20), textcoords="offset points", ha="center"
    )

    # Middle point annotations
    plt.annotate(
        r"$\alpha x + (1-\alpha)y$",
        (z, 0),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
    )

    # Function value annotations
    plt.annotate(
        r"$f(\alpha x + (1-\alpha)y)$",
        (z, fz),
        xytext=(10, 0),
        textcoords="offset points",
    )
    plt.annotate(
        r"$\alpha f(x) + (1-\alpha)f(y)$",
        (z, f_linear),
        xytext=(10, 0),
        textcoords="offset points",
    )

    # Customize the plot
    plt.grid(True)
    plt.title("Visualization of Convex Function Inequality")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # Add legend without duplicate point entries
    plt.legend(["f(x)", "Linear interpolation"])

    return plt


def plot_first_order_convexity(f, df, x, y, points=100):
    """
    Visualize the first-order condition proof of convexity:
    f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)

    Parameters:
    -----------
    f : callable
        The convex function to plot
    df : callable
        The gradient (derivative) of f
    x : float
        First x-coordinate
    y : float
        Second x-coordinate
    points : int
        Number of points for smooth curve plotting
    """
    # Generate points for smooth curve
    x_range = np.linspace(min(x, y) - 0.5, max(x, y) + 0.5, points)
    y_range = [f(xi) for xi in x_range]

    # Calculate function values and gradients
    fx = f(x)
    fy = f(y)
    dfx = df(x)
    dfy = df(y)

    # Create tangent lines
    def tangent_line_x(t):
        return fx + dfx * (t - x)

    def tangent_line_y(t):
        return fy + dfy * (t - y)

    tangent_x = [tangent_line_x(t) for t in x_range]
    tangent_y = [tangent_line_y(t) for t in x_range]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot function and tangent lines
    plt.plot(x_range, y_range, "b-", label="f(x)")
    plt.plot(x_range, tangent_x, "g--", label="Tangent at x")
    plt.plot(x_range, tangent_y, "r--", label="Tangent at y")

    # Plot points
    plt.plot(x, fx, "go")
    plt.plot(y, fy, "ro")

    # Add annotations
    plt.annotate("$(x, f(x))$", (x, fx), xytext=(10, 10), textcoords="offset points")
    plt.annotate("$(y, f(y))$", (y, fy), xytext=(10, 10), textcoords="offset points")

    # Add gradient vectors annotations
    plt.annotate(
        r"$\nabla f(x)$",
        (x, fx),
        xytext=(30, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )
    plt.annotate(
        r"$\nabla f(y)$",
        (y, fy),
        xytext=(30, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    # Add first order condition inequality
    plt.text(
        0.05,
        0.95,
        r"$f(y) \geq f(x) + \nabla f(x)^T(y-x)$",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.grid(True)
    plt.legend()
    plt.title("First-Order Condition for Convexity")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    return plt


def plot_second_order_convexity(
    f, df, d2f, x, directions=None, t_range=(-1, 1), points=100
):
    """
    Visualize the second-order condition for convexity:
    g(t) = f(x + td) is convex ⟹ g''(t) = dᵀ∇²f(x + td)d ≥ 0

    Parameters:
    -----------
    f : callable
        The convex function to plot (should take vector input)
    df : callable
        The gradient of f (should take vector input)
    d2f : callable
        The Hessian of f (should take vector input)
    x : numpy.ndarray
        Base point
    directions : list of numpy.ndarray, optional
        List of direction vectors to plot
    t_range : tuple, optional
        Range for parameter t
    points : int, optional
        Number of points for smooth curve plotting
    """
    if directions is None:
        # Default directions: unit vectors and their combination
        directions = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")  # 3D plot for f(x)
    ax2 = fig.add_subplot(gs[0, 1])  # 2D plot for g(t)
    ax3 = fig.add_subplot(gs[0, 2])  # 2D plot for g''(t)

    # Create grid for 3D plot
    x1 = np.linspace(x[0] - 1, x[0] + 1, points)
    x2 = np.linspace(x[1] - 1, x[1] + 1, points)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    # Compute function values for 3D plot
    for i in range(points):
        for j in range(points):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Plot 3D surface
    surf = ax1.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)
    ax1.set_title("f(x)")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_zlabel("f(x)")

    # Plot base point in 3D
    ax1.scatter([x[0]], [x[1]], [f(x)], color="red", s=100, label="Base point x")

    # Plot directions in 3D
    colors = ["b", "g", "r"]
    t = np.linspace(t_range[0], t_range[1], points)

    for d, color in zip(directions, colors):
        # Plot direction vectors in 3D
        direction_points = np.array([x + ti * d for ti in t])
        z_points = np.array([f(p) for p in direction_points])
        ax1.plot(
            direction_points[:, 0],
            direction_points[:, 1],
            z_points,
            color=color,
            label=f"d = [{d[0]:.1f}, {d[1]:.1f}]",
        )

        # Compute and plot g(t)
        g_t = np.array([f(x + ti * d) for ti in t])
        ax2.plot(t, g_t, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]")

        # Compute and plot g''(t)
        g_double_prime = np.array([d.T @ d2f(x + ti * d) @ d for ti in t])
        ax3.plot(t, g_double_prime, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]")

    # Mark base point in g(t)
    ax2.plot(0, f(x), "ko")

    # Add zero line in g''(t)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Customize plots
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("g(t) = f(x + td)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("g(t)")

    ax3.grid(True)
    ax3.legend()
    ax3.set_title(r"g′′(t) = dᵀ∇²f(x + td)d ≥ 0")
    ax3.set_xlabel("t")
    ax3.set_ylabel("g′′(t)")

    # Add text explaining the second-order condition
    plt.figtext(
        0.02,
        0.02,
        r"Second-order condition: $\nabla^2f(x)$ is positive semidefinite"
        + r" $\Leftrightarrow$ $d^T\nabla^2f(x)d \geq 0$ for all directions d",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return plt


def plot_convexity_slice(f, x, directions=None, t_range=(-1, 1), points=100):
    """
    Visualize how g(t) = f(x + td) relates to f(x) and show its convexity near t=0

    Parameters:
    -----------
    f : callable
        The convex function to plot (should take vector input)
    x : numpy.ndarray
        Base point
    directions : list of numpy.ndarray, optional
        List of direction vectors to plot
    t_range : tuple, optional
        Range for parameter t to show domain
    points : int, optional
        Number of points for smooth curve plotting
    """
    if directions is None:
        # Default directions: unit vectors and their combination
        directions = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1])

    # 3D plot for f(x)
    ax1 = fig.add_subplot(gs[0], projection="3d")

    # 2D plot for g(t)
    ax2 = fig.add_subplot(gs[1])

    # Create grid for 3D plot
    x1 = np.linspace(x[0] - 2, x[0] + 2, points)
    x2 = np.linspace(x[1] - 2, x[1] + 2, points)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    # Compute function values for 3D plot
    for i in range(points):
        for j in range(points):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Plot 3D surface
    surf = ax1.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.8)

    # Plot base point in 3D
    ax1.scatter([x[0]], [x[1]], [f(x)], color="red", s=100, label="Base point x")

    # Plot directions and slices
    colors = ["b", "g", "r"]
    t = np.linspace(t_range[0], t_range[1], points)

    for d, color in zip(directions, colors):
        # Plot direction vectors in 3D
        direction_points = np.array([x + ti * d for ti in t])
        z_points = np.array([f(p) for p in direction_points])

        # Plot the slice curve on the surface
        ax1.plot(
            direction_points[:, 0],
            direction_points[:, 1],
            z_points,
            color=color,
            linewidth=2,
            label=f"d = [{d[0]:.1f}, {d[1]:.1f}]",
        )

        # Plot g(t) = f(x + td)
        g_t = np.array([f(x + ti * d) for ti in t])
        ax2.plot(t, g_t, f"{color}-", label=f"d = [{d[0]:.1f}, {d[1]:.1f}]")

        # Add points at t = -0.5, 0, 0.5 to show convexity
        t_points = [-0.5, 0, 0.5]
        g_points = [f(x + tp * d) for tp in t_points]
        ax2.plot(t_points, g_points, f"{color}o")

        # Add a line segment connecting t = -0.5 and t = 0.5 points
        ax2.plot([-0.5, 0.5], [g_points[0], g_points[2]], f"{color}--", alpha=0.5)

    # Customize 3D plot
    ax1.set_title("f(x) with directional slices")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_zlabel("f(x)")
    ax1.legend()

    # Customize g(t) plot
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("g(t) = f(x + td)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("g(t)")

    # Add vertical line at t=0
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax2.axvline(x=-0.5, color="k", linestyle=":", alpha=0.2)
    ax2.axvline(x=0.5, color="k", linestyle=":", alpha=0.2)

    # Add text explaining the relationship
    plt.figtext(
        0.02,
        0.02,
        r"For each direction d, g(t) = f(x + td) is convex near t = 0"
        + "\nPoints show g(λt₁ + (1-λ)t₂) ≤ λg(t₁) + (1-λ)g(t₂)",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    return plt


def plot_composition_convexity(f, g, x_range=(-2, 2), points=100):
    """
    Visualize convexity preservation under function composition f(g(x))

    Parameters:
    -----------
    f : callable
        Outer function
    g : callable
        Inner function
    x_range : tuple
        Range for x-axis
    points : int
        Number of points for smooth curve plotting
    """
    x = np.linspace(x_range[0], x_range[1], points)

    # Compute function values
    g_x = np.array([g(xi) for xi in x])
    fg_x = np.array([f(g(xi)) for xi in x])

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot g(x)
    ax1.plot(x, g_x, "b-", label="g(x)")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Inner function g(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("g(x)")

    # Plot f(y)
    y_range = np.linspace(min(g_x) - 0.5, max(g_x) + 0.5, points)
    f_y = np.array([f(yi) for yi in y_range])
    ax2.plot(y_range, f_y, "r-", label="f(y)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("Outer function f(y)")
    ax2.set_xlabel("y")
    ax2.set_ylabel("f(y)")

    # Plot composition f(g(x))
    ax3.plot(x, fg_x, "g-", label="f(g(x))")
    ax3.grid(True)
    ax3.legend()
    ax3.set_title("Composition f(g(x))")
    ax3.set_xlabel("x")
    ax3.set_ylabel("f(g(x))")

    # Add convexity test points
    test_points = [-1, 0, 1]
    alpha = 0.5

    # Test convexity for g
    g_test = [g(x) for x in test_points]
    g_mid = g(alpha * test_points[0] + (1 - alpha) * test_points[2])
    g_interp = alpha * g_test[0] + (1 - alpha) * g_test[2]

    # Test convexity for f∘g
    fg_test = [f(g(x)) for x in test_points]
    fg_mid = f(g(alpha * test_points[0] + (1 - alpha) * test_points[2]))
    fg_interp = alpha * fg_test[0] + (1 - alpha) * fg_test[2]

    # Plot test points and interpolation
    for ax, y_vals, mid, interp in [
        (ax1, g_test, g_mid, g_interp),
        (ax3, fg_test, fg_mid, fg_interp),
    ]:
        # Plot test points
        ax.plot(test_points, y_vals, "ko")
        # Plot midpoint
        ax.plot(0, mid, "go")
        # Plot interpolation point
        ax.plot(0, interp, "ro")
        # Plot interpolation line
        ax.plot(
            [test_points[0], test_points[2]], [y_vals[0], y_vals[2]], "r--", alpha=0.5
        )

    plt.tight_layout()
    return plt


def plot_composition_examples(x_range=(-2, 2), points=100):
    """
    Visualize two examples of function composition and convexity:
    1. A working case where f(g(x)) is convex
    2. A case where f(g(x)) is not convex
    """
    x = np.linspace(x_range[0], x_range[1], points)

    # Create figure with 2x3 subplots in a compact layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Example 1: f(t) = t², g(x) = x² (convex case)
    def f1(t):
        return t**2  # convex and non-decreasing for t ≥ 0

    def g1(x):
        return x**2  # convex

    def fg1(x):
        return (x**2) ** 2  # composition is convex

    # Example 2: f(t) = -t, g(x) = x² (non-convex case)
    def f2(t):
        return -t  # affine but decreasing

    def g2(x):
        return x**2  # convex

    def fg2(x):
        return -(x**2)  # composition is concave

    # Plot first example
    y_range1 = np.linspace(0, max(g1(x)), points)  # ensure domain of f1 is non-negative
    axes[0, 0].plot(x, g1(x), "b-")
    axes[0, 0].set_title("g₁(x) = x²\n(convex)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(y_range1, f1(y_range1), "r-")
    axes[0, 1].set_title("f₁(t) = t²\n(convex & non-decreasing)")
    axes[0, 1].grid(True)

    axes[0, 2].plot(x, fg1(x), "g-")
    axes[0, 2].set_title("f₁(g₁(x)) = (x²)²\n(convex)")
    axes[0, 2].grid(True)

    # Plot second example
    y_range2 = np.linspace(min(g2(x)), max(g2(x)), points)
    axes[1, 0].plot(x, g2(x), "b-")
    axes[1, 0].set_title("g₂(x) = x²\n(convex)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(y_range2, f2(y_range2), "r-")
    axes[1, 1].set_title("f₂(t) = -t\n(affine but decreasing)")
    axes[1, 1].grid(True)

    axes[1, 2].plot(x, fg2(x), "g-")
    axes[1, 2].set_title("f₂(g₂(x)) = -x²\n(concave)")
    axes[1, 2].grid(True)

    # Add convexity test points and interpolation
    test_points = [-1, 0, 1]
    alpha = 0.5
    mid_x = alpha * test_points[0] + (1 - alpha) * test_points[2]

    for i, (g, f, fg) in enumerate([(g1, f1, fg1), (g2, f2, fg2)]):
        # Test points for g
        g_test = [g(x) for x in test_points]
        g_mid = g(mid_x)
        g_interp = alpha * g_test[0] + (1 - alpha) * g_test[2]

        # Test points for f∘g
        fg_test = [fg(x) for x in test_points]
        fg_mid = fg(mid_x)
        fg_interp = alpha * fg_test[0] + (1 - alpha) * fg_test[2]

        # Plot test points on g and f∘g
        for ax, y_vals, mid, interp in [
            (axes[i, 0], g_test, g_mid, g_interp),
            (axes[i, 2], fg_test, fg_mid, fg_interp),
        ]:
            ax.plot(test_points, y_vals, "ko", label="Test points")
            ax.plot(mid_x, mid, "go", label="Actual")
            ax.plot(mid_x, interp, "ro", label="Interpolation")
            ax.plot(
                [test_points[0], test_points[2]],
                [y_vals[0], y_vals[2]],
                "r--",
                alpha=0.5,
            )

    plt.tight_layout()

    # Add explanation text
    plt.figtext(
        0.02,
        0.98,
        "Example 1: f₁(g₁(x)) is convex because f₁ is convex & non-decreasing and g₁ is convex",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.figtext(
        0.02,
        0.48,
        "Example 2: f₂(g₂(x)) is not convex because f₂ is decreasing",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    return plt


# Example usage:
if __name__ == "__main__":
    plot = plot_composition_examples()
    plt.show()
