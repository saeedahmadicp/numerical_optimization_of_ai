# plot/shape.py

"""Visualization utilities for convex and non-convex shapes."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt


class Shape(ABC):
    """Abstract base class for shapes."""

    def __init__(self, title: str, figsize: Tuple[int, int] = (6, 6)):
        self.title = title
        self.figsize = figsize

    @abstractmethod
    def generate_points(self) -> Tuple[np.ndarray, ...]:
        """Generate points that define the shape."""
        pass

    @abstractmethod
    def plot(self) -> plt.Figure:
        """Plot the shape."""
        pass

    def _setup_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Setup basic plot with common decorations."""
        fig = plt.figure(figsize=self.figsize)
        ax = plt.gca()

        # Add common decorations
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(self.title)
        ax.axis("equal")
        ax.grid(True)

        return fig, ax


class ConvexShape(Shape):
    """Base class for convex shapes with convexity verification."""

    def verify_convexity(
        self, points: List[Tuple[float, float]], num_tests: int = 1000
    ) -> bool:
        """
        Verify if shape is convex by testing random pairs of points.

        Args:
            points: List of points defining the shape
            num_tests: Number of random tests to perform

        Returns:
            bool: True if all tested points lie within the shape
        """
        points = np.array(points)
        for _ in range(num_tests):
            # Select two random points
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx1], points[idx2]

            # Test a random point on the line segment
            t = np.random.random()
            test_point = t * p1 + (1 - t) * p2

            # Check if point lies within shape
            if not self.contains_point(test_point):
                return False
        return True

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point lies within the shape."""
        pass


class Ball(ConvexShape):
    """Unit ball centered at origin."""

    def __init__(self, radius: float = 1.0, center: Tuple[float, float] = (0, 0)):
        super().__init__(title=f"Ball (radius={radius})")
        self.radius = radius
        self.center = np.array(center)

    def generate_points(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return x, y

    def contains_point(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) <= self.radius

    def plot(self) -> plt.Figure:
        fig, ax = self._setup_plot()
        x, y = self.generate_points()
        ax.plot(x, y, label="Boundary")
        ax.fill(x, y, alpha=0.3, label="Interior")
        ax.legend()
        return fig


class Polygon(ConvexShape):
    """Polygon defined by vertices."""

    def __init__(
        self, vertices: List[Tuple[float, float]], title: Optional[str] = None
    ):
        super().__init__(title=title or "Polygon")
        self.vertices = np.array(vertices)

    def generate_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices[:, 0], self.vertices[:, 1]

    def contains_point(self, point: np.ndarray) -> bool:
        # Implementation of point-in-polygon test
        # Using ray casting algorithm
        x, y = point
        n = len(self.vertices)
        inside = False

        for i in range(n):
            j = (i + 1) % n
            if (self.vertices[i, 1] > y) != (self.vertices[j, 1] > y) and x < (
                self.vertices[j, 0] - self.vertices[i, 0]
            ) * (y - self.vertices[i, 1]) / (
                self.vertices[j, 1] - self.vertices[i, 1]
            ) + self.vertices[
                i, 0
            ]:
                inside = not inside
        return inside

    def plot(self, show_vertices: bool = True) -> plt.Figure:
        fig, ax = self._setup_plot()
        patch = plt.Polygon(self.vertices, alpha=0.3, label="Interior")
        ax.add_patch(patch)

        if show_vertices:
            x, y = self.generate_points()
            ax.plot(x, y, "o-", label="Vertices")

        ax.legend()
        return fig


class FunctionShape(Shape):
    """Shape defined by a function."""

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        x_range: Tuple[float, float],
        title: Optional[str] = None,
    ):
        super().__init__(title=title or "Function")
        self.func = func
        self.x_range = x_range

    def generate_points(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.x_range[0], self.x_range[1], 100)
        y = self.func(x)
        return x, y

    def plot(self) -> plt.Figure:
        fig, ax = self._setup_plot()
        x, y = self.generate_points()
        ax.plot(x, y, label=self.title)
        ax.legend()
        return fig


# # Example usage:
# if __name__ == "__main__":
#     # Plot unit ball
#     ball = Ball(radius=2.0)
#     ball.plot()
#     plt.show()

#     # Plot convex polygon
#     vertices = [(0, 0), (1, 0), (0.8, 0.8), (0.2, 1), (-0.5, 0.5), (-0.5, 0)]
#     poly = Polygon(vertices, "Convex Polygon")
#     poly.plot()
#     plt.show()

#     # Plot quadratic function
#     quad = FunctionShape(lambda x: x**2, x_range=(-2, 2), title="$f(x) = x^2$ (Convex)")
#     quad.plot()
#     plt.show()
