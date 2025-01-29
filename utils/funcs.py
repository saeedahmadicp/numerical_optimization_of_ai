# utils/funcs.py

"""Common test functions for optimization algorithms."""

import os
import numpy as np
import torch  # type: ignore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = [
    "func1",
    "func2",
    "func3",
    "func4",
    "func5",
    "func1_torch",
    "func2_torch",
    "func3_torch",
    "func4_torch",
    "func5_torch",
]


# NumPy-based test functions
def func1(x):
    return abs(x + np.cos(x))


def func2(x):
    return abs(x - 0.3)


def func3(x):
    return 2 * np.exp(x) - 2**x - 3 * x


def func4(x):
    return abs(x - 0.5)


def func5(x):
    return abs(x**3 + x**2 + x)


# PyTorch-based test functions for gradient-based methods
def func1_torch(x):
    return x**2 - 2  # Root at ±√2


def func2_torch(x):
    return x**3 - 2 * x - 5  # Cubic equation


def func3_torch(x):
    return torch.exp(x) - 2 * x - 1  # Transcendental equation


def func4_torch(x):
    return torch.sin(x) - x / 2  # Trigonometric equation


def func5_torch(x):
    return torch.cos(x) - x  # Fixed point equation
