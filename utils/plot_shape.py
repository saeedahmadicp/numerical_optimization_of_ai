# utils/plot_shape.py

"""Visualization utilities for convex and non-convex shapes."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot_unit_ball():
    """Plot unit ball as example of convex set."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Unit Ball Boundary")
    plt.fill(x, y, alpha=0.3, label="Unit Ball")
    _add_plot_decorations("Unit Ball (Convex Set)")


def plot_convex_polygon():
    """Plot example of convex polygon."""
    vertices = np.array([[0, 0], [1, 0], [0.8, 0.8], [0.2, 1], [-0.5, 0.5], [-0.5, 0]])
    _plot_polygon(vertices, "Convex Polygon")


def plot_non_convex_polygon():
    """Plot example of non-convex polygon with line showing non-convexity."""
    vertices = np.array(
        [[0, 0], [1, 0], [0.8, 0.8], [0.4, 0.2], [0.2, 1], [-0.5, 0.5], [-0.5, 0]]
    )
    _plot_polygon(vertices, "Non-Convex Polygon", show_line=True)


def plot_convex_quadratic():
    """Plot convex quadratic function."""
    _plot_quadratic(lambda x: x**2, "$f(x) = x^2$ (Convex)")


def plot_non_convex_quadratic():
    """Plot non-convex quadratic function."""
    _plot_quadratic(lambda x: -(x**2), "$f(x) = -x^2$ (Non-Convex)")


def _plot_polygon(vertices, title, show_line=False):
    """Helper to plot polygon with consistent styling."""
    plt.figure(figsize=(6, 6))
    plt.gca().add_patch(Polygon(vertices, closed=True, alpha=0.3, label=title))
    plt.plot(vertices[:, 0], vertices[:, 1], "o-", label="Vertices")

    if show_line:
        p1, p2 = [0.8, 0.8], [-0.5, 0]
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2, label="Line Outside"
        )
        plt.scatter(*p1, color="red", label="Point 1")
        plt.scatter(*p2, color="red", label="Point 2")

    _add_plot_decorations(title)


def _plot_quadratic(f, label):
    """Helper to plot quadratic functions with consistent styling."""
    x = np.linspace(-2, 2, 100)
    plt.figure(figsize=(6, 6))
    plt.plot(x, f(x), label=label)
    _add_plot_decorations(label.split("(")[1][:-1])


def _add_plot_decorations(title):
    """Add common plot decorations."""
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()
