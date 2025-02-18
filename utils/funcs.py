# utils/funcs.py

"""Common test functions for root finding algorithms."""

import os
import numpy as np
import torch  # type: ignore
from typing import Tuple, Callable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Type aliases for function and derivative pairs
FuncPair = Tuple[Callable[[float], float], Callable[[float], float]]

def get_test_function(name: str) -> FuncPair:
    """Get a test function and its derivative by name."""
    return FUNCTION_MAP[name]

# Basic Polynomial Functions
def quadratic() -> FuncPair:
    """f(x) = x² - 2, root at ±√2."""
    return (
        lambda x: x**2 - 2,
        lambda x: 2 * x
    )

def cubic() -> FuncPair:
    """f(x) = x³ - x - 2, one real root near x ≈ 1.7693."""
    return (
        lambda x: x**3 - x - 2,
        lambda x: 3 * x**2 - 1
    )

def quartic() -> FuncPair:
    """f(x) = x⁴ - 5x² + 4, roots at ±1, ±2."""
    return (
        lambda x: x**4 - 5*x**2 + 4,
        lambda x: 4*x**3 - 10*x
    )

# Transcendental Functions
def exponential() -> FuncPair:
    """f(x) = e^x - 4, root at ln(4)."""
    return (
        lambda x: np.exp(x) - 4,
        lambda x: np.exp(x)
    )

def logarithmic() -> FuncPair:
    """f(x) = ln(x) - 1, root at e."""
    return (
        lambda x: np.log(x) - 1,
        lambda x: 1/x
    )

def exp_linear() -> FuncPair:
    """f(x) = e^x - 2x - 1, root near x ≈ 0.5671."""
    return (
        lambda x: np.exp(x) - 2*x - 1,
        lambda x: np.exp(x) - 2
    )

# Trigonometric Functions
def sinusoidal() -> FuncPair:
    """f(x) = sin(x) - 0.5, roots near x ≈ 0.5236, 2.6180."""
    return (
        lambda x: np.sin(x) - 0.5,
        lambda x: np.cos(x)
    )

def cosine() -> FuncPair:
    """f(x) = cos(x) - x, root near x ≈ 0.7390."""
    return (
        lambda x: np.cos(x) - x,
        lambda x: -np.sin(x) - 1
    )

def tangent() -> FuncPair:
    """f(x) = tan(x) - x, multiple roots."""
    return (
        lambda x: np.tan(x) - x,
        lambda x: 1/np.cos(x)**2 - 1
    )

# Combined Functions
def trig_polynomial() -> FuncPair:
    """f(x) = x³ - 6x² + 11x - 6 + sin(x), multiple roots."""
    return (
        lambda x: x**3 - 6*x**2 + 11*x - 6 + np.sin(x),
        lambda x: 3*x**2 - 12*x + 11 + np.cos(x)
    )

def exp_sine() -> FuncPair:
    """f(x) = e^x - sin(x) - 2, root near x ≈ 0.9275."""
    return (
        lambda x: np.exp(x) - np.sin(x) - 2,
        lambda x: np.exp(x) - np.cos(x)
    )

# Challenging Functions
def stiff() -> FuncPair:
    """f(x) = 1000x³ - 2000x² + 1000x - 0.001, highly stiff system."""
    return (
        lambda x: 1000*x**3 - 2000*x**2 + 1000*x - 0.001,
        lambda x: 3000*x**2 - 4000*x + 1000
    )

def multiple_roots() -> FuncPair:
    """f(x) = (x-1)²(x+2), roots at x = 1 (double root) and x = -2."""
    return (
        lambda x: (x-1)**2 * (x+2),
        lambda x: (x-1)**2 + 2*(x-1)*(x+2)
    )

def ill_conditioned() -> FuncPair:
    """f(x) = x¹⁰ - 1, roots at e^(2πik/10), k=0,1,...,9."""
    return (
        lambda x: x**10 - 1,
        lambda x: 10*x**9
    )

# PyTorch-based functions for automatic differentiation
def quadratic_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x² - 2 using PyTorch."""
    return x**2 - 2

def cubic_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x³ - x - 2 using PyTorch."""
    return x**3 - x - 2

def exp_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = e^x - 2x - 1 using PyTorch."""
    return torch.exp(x) - 2*x - 1

def sin_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x) - x/2 using PyTorch."""
    return torch.sin(x) - x/2

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

def get_torch_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a PyTorch-based test function by name."""
    return TORCH_FUNCTIONS[name]

# Example usage in doctest format:
"""
>>> f, df = get_test_function("quadratic")
>>> f(2.0)  # Should be close to 2
2.0
>>> df(2.0)  # Derivative at x=2
4.0

>>> f, df = get_test_function("exponential")
>>> abs(f(np.log(4)))  # Should be close to 0
0.0
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod()
