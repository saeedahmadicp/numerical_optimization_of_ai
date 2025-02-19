# utils/funcs.py

"""Common test functions for root finding algorithms."""

import os
import numpy as np
import torch  # type: ignore
from typing import Tuple, Callable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Type aliases for function and derivative pairs
FuncPair = Tuple[Callable[[float], float], Callable[[float], float]]


# Basic Polynomial Functions
def quadratic() -> FuncPair:
    """f(x) = x² - 2, root at ±√2."""
    return (lambda x: x**2 - 2, lambda x: 2 * x)


def cubic() -> FuncPair:
    """f(x) = x³ - x - 2, one real root near x ≈ 1.7693."""
    return (lambda x: x**3 - x - 2, lambda x: 3 * x**2 - 1)


def quartic() -> FuncPair:
    """f(x) = x⁴ - 5x² + 4, roots at ±1, ±2."""
    return (lambda x: x**4 - 5 * x**2 + 4, lambda x: 4 * x**3 - 10 * x)


# Transcendental Functions
def exponential() -> FuncPair:
    """f(x) = e^x - 4, root at ln(4)."""
    return (lambda x: np.exp(x) - 4, lambda x: np.exp(x))


def logarithmic() -> FuncPair:
    """f(x) = ln(x) - 1, root at e."""
    return (lambda x: np.log(x) - 1, lambda x: 1 / x)


def exp_linear() -> FuncPair:
    """f(x) = e^x - 2x - 1, root near x ≈ 0.5671."""
    return (lambda x: np.exp(x) - 2 * x - 1, lambda x: np.exp(x) - 2)


# Trigonometric Functions
def sinusoidal() -> FuncPair:
    """f(x) = sin(x) - 0.5, roots near x ≈ 0.5236, 2.6180."""
    return (lambda x: np.sin(x) - 0.5, lambda x: np.cos(x))


def cosine() -> FuncPair:
    """f(x) = cos(x) - x, root near x ≈ 0.7390."""
    return (lambda x: np.cos(x) - x, lambda x: -np.sin(x) - 1)


def tangent() -> FuncPair:
    """f(x) = tan(x) - x, multiple roots."""
    return (lambda x: np.tan(x) - x, lambda x: 1 / np.cos(x) ** 2 - 1)


# Combined Functions
def trig_polynomial() -> FuncPair:
    """f(x) = x³ - 6x² + 11x - 6 + sin(x), multiple roots."""
    return (
        lambda x: x**3 - 6 * x**2 + 11 * x - 6 + np.sin(x),
        lambda x: 3 * x**2 - 12 * x + 11 + np.cos(x),
    )


def exp_sine() -> FuncPair:
    """f(x) = e^x - sin(x) - 2, root near x ≈ 0.9275."""
    return (lambda x: np.exp(x) - np.sin(x) - 2, lambda x: np.exp(x) - np.cos(x))


# Challenging Functions
def stiff() -> FuncPair:
    """f(x) = 1000x³ - 2000x² + 1000x - 0.001, highly stiff system."""
    return (
        lambda x: 1000 * x**3 - 2000 * x**2 + 1000 * x - 0.001,
        lambda x: 3000 * x**2 - 4000 * x + 1000,
    )


def multiple_roots() -> FuncPair:
    """f(x) = (x-1)²(x+2), roots at x = 1 (double root) and x = -2."""
    return (
        lambda x: (x - 1) ** 2 * (x + 2),
        lambda x: (x - 1) ** 2 + 2 * (x - 1) * (x + 2),
    )


def ill_conditioned() -> FuncPair:
    """f(x) = x¹⁰ - 1, roots at e^(2πik/10), k=0,1,...,9."""
    return (lambda x: x**10 - 1, lambda x: 10 * x**9)


# PyTorch-based functions for automatic differentiation
def quadratic_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x² - 2 using PyTorch."""
    return x**2 - 2


def cubic_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x³ - x - 2 using PyTorch."""
    return x**3 - x - 2


def exp_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = e^x - 2x - 1 using PyTorch."""
    return torch.exp(x) - 2 * x - 1


def sin_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x) - x/2 using PyTorch."""
    return torch.sin(x) - x / 2


def cos_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = cos(x) - x using PyTorch."""
    return torch.cos(x) - x


# Map function names to their implementations
FUNCTION_MAP = {
    "quadratic": quadratic(),
    "cubic": cubic(),
    "quartic": quartic(),
    "exponential": exponential(),
    "logarithmic": logarithmic(),
    "exp_linear": exp_linear(),
    "sinusoidal": sinusoidal(),
    "cosine": cosine(),
    "tangent": tangent(),
    "trig_polynomial": trig_polynomial(),
    "exp_sine": exp_sine(),
    "stiff": stiff(),
    "multiple_roots": multiple_roots(),
    "ill_conditioned": ill_conditioned(),
}

# List of all available test functions
AVAILABLE_FUNCTIONS = list(FUNCTION_MAP.keys())

# PyTorch-based functions
TORCH_FUNCTIONS = {
    "quadratic": quadratic_torch,
    "cubic": cubic_torch,
    "exp_linear": exp_torch,
    "sinusoidal": sin_torch,
    "cosine": cos_torch,
}


def quadratic_min() -> FuncPair:
    """f(x) = x², minimum at x = 0."""
    return (lambda x: x**2, lambda x: 2 * x)


def cubic_min() -> FuncPair:
    """f(x) = x³ + x, minimum at x ≈ -0.577."""
    return (lambda x: x**3 + x, lambda x: 3 * x**2 + 1)


def quartic_min() -> FuncPair:
    """f(x) = x⁴ - 2x² + 1, minima at x = ±1."""
    return (lambda x: x**4 - 2 * x**2 + 1, lambda x: 4 * x**3 - 4 * x)


def rosenbrock() -> FuncPair:
    """Rosenbrock function (banana function): f(x,y) = (1-x)² + 100(y-x²)², minimum at (1,1)."""
    return (
        lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
        lambda x: np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        ),
    )


def himmelblau() -> FuncPair:
    """Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)², 4 local minima."""
    return (
        lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
        lambda x: np.array(
            [
                4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
                2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7),
            ]
        ),
    )


def rastrigin() -> FuncPair:
    """Rastrigin function: f(x) = 10n + Σ(x_i² - 10cos(2πx_i)), global minimum at origin."""

    def f(x):
        return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

    def df(x):
        return np.array([2 * xi + 20 * np.pi * np.sin(2 * np.pi * xi) for xi in x])

    return (f, df)


def ackley() -> FuncPair:
    """Ackley function: Complex multimodal function with global minimum at origin."""

    def f(x):
        n = len(x)
        sum_sq = sum(xi**2 for xi in x)
        sum_cos = sum(np.cos(2 * np.pi * xi) for xi in x)
        return (
            -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e
        )

    def df(x):
        n = len(x)
        sum_sq = sum(xi**2 for xi in x)
        term1 = 4 * np.exp(-0.2 * np.sqrt(sum_sq / n)) / np.sqrt(n * sum_sq)
        return np.array(
            [
                term1 * xi
                + 2
                * np.pi
                * np.exp(np.cos(2 * np.pi * xi) / n)
                * np.sin(2 * np.pi * xi)
                / n
                for xi in x
            ]
        )

    return (f, df)


def beale() -> FuncPair:
    """Beale function: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)², min at (3,0.5)."""

    def f(x):
        return (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )

    def df(x):
        t1 = 1.5 - x[0] + x[0] * x[1]
        t2 = 2.25 - x[0] + x[0] * x[1] ** 2
        t3 = 2.625 - x[0] + x[0] * x[1] ** 3
        return np.array(
            [
                2 * t1 * (-1 + x[1])
                + 2 * t2 * (-1 + x[1] ** 2)
                + 2 * t3 * (-1 + x[1] ** 3),
                2 * t1 * x[0]
                + 2 * t2 * 2 * x[0] * x[1]
                + 2 * t3 * 3 * x[0] * x[1] ** 2,
            ]
        )

    return (f, df)


def booth() -> FuncPair:
    """Booth function: f(x,y) = (x+2y-7)² + (2x+y-5)², minimum at (1,3)."""
    return (
        lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
        lambda x: np.array(
            [
                2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5),
                4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5),
            ]
        ),
    )


# Map function names to their minimization implementations
MINIMIZATION_MAP = {
    "quadratic": quadratic_min(),
    "cubic": cubic_min(),
    "quartic": quartic_min(),
    "rosenbrock": rosenbrock(),
    "himmelblau": himmelblau(),
    "rastrigin": rastrigin(),
    "ackley": ackley(),
    "beale": beale(),
    "booth": booth(),
}

# Default ranges for minimization functions
MINIMIZATION_RANGES = {
    "quadratic": (-3, 3),
    "cubic": (-2, 2),
    "quartic": (-2, 2),
    "rosenbrock": (-2, 2),
    "himmelblau": (-5, 5),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-5, 5),
    "beale": (-4.5, 4.5),
    "booth": (-10, 10),
}


def get_minimization_function(name: str) -> FuncPair:
    """Get a minimization test function and its derivative by name."""
    return MINIMIZATION_MAP[name]


def get_test_function(name: str) -> FuncPair:
    """Get a test function and its derivative by name."""
    return FUNCTION_MAP[name]


def get_torch_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a PyTorch-based test function by name."""
    return TORCH_FUNCTIONS[name]


# # Example usage in doctest format:
# """
# >>> f, df = get_test_function("quadratic")
# >>> f(2.0)  # Should be close to 2
# 2.0
# >>> df(2.0)  # Derivative at x=2
# 4.0

# >>> f, df = get_test_function("exponential")
# >>> abs(f(np.log(4)))  # Should be close to 0
# 0.0
# """

# if __name__ == "__main__":
#     import doctest

#     doctest.testmod()
