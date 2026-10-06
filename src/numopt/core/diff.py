"""Finite-difference fallbacks for derivatives that a problem does not supply.

Step sizes follow Nocedal & Wright (2006), §8.1: h ≈ √ε·max(1,|x|) for forward
differences and h ≈ ε^{1/3}·max(1,|x|) for central differences.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_EPS = np.finfo(float).eps
_H_CENTRAL = _EPS ** (1.0 / 3.0)


def derivative(f: Callable[[float], float], x: float) -> float:
    """Central-difference f'(x)."""
    h = _H_CENTRAL * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second_derivative(f: Callable[[float], float], x: float) -> float:
    """Central-difference f''(x) with h ≈ ε^{1/4} (balances truncation and rounding)."""
    h = _EPS**0.25 * max(1.0, abs(x))
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def gradient(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """Central-difference gradient."""
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    for i in range(x.size):
        h = _H_CENTRAL * max(1.0, abs(x[i]))
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2.0 * h)
    return g


def hessian(grad: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Symmetrized central-difference Hessian from a gradient."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.empty((n, n))
    for i in range(n):
        h = _H_CENTRAL * max(1.0, abs(x[i]))
        e = np.zeros_like(x)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2.0 * h)
    return 0.5 * (H + H.T)


def jacobian(F: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian of a vector function."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(F(x), dtype=float)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = _H_CENTRAL * max(1.0, abs(x[i]))
        e = np.zeros_like(x)
        e[i] = h
        J[:, i] = (np.asarray(F(x + e)) - np.asarray(F(x - e))) / (2.0 * h)
    return J
