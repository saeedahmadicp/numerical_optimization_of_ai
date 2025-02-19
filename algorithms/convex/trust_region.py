# algorithms/convex/trust_region.py

"""Trust region methods for convex optimization.

TODO: Implement the following methods:
- Dogleg method
- Subspace trust region method
- Conjugate gradient trust region method
- Preconditioned conjugate gradient trust region method
- Preconditioned conjugate gradient trust region method

THIS FILE IS NOT COMPLETE!
"""

import numpy as np
from typing import Callable, Tuple, List


def dogleg_trust_region(grad: np.ndarray, hess: np.ndarray, delta: float) -> np.ndarray:
    """
    Solve the trust region subproblem using the Dogleg method.

    Parameters:
        grad  : The gradient vector at the current iterate.
        hess  : The Hessian matrix at the current iterate.
        delta : The trust region radius.

    Returns:
        The computed step within the trust region.
    """
    # Try Newton step
    try:
        pB = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        pB = -grad

    # Compute Cauchy point
    grad_norm_sq = np.dot(grad, grad)
    if grad_norm_sq == 0:
        return np.zeros_like(grad)

    curvature = np.dot(grad, hess @ grad)
    tau = 1.0 if curvature <= 0 else grad_norm_sq / curvature
    pU = -tau * grad

    # Check trust region cases
    if np.linalg.norm(pB) <= delta:
        return pB
    if np.linalg.norm(pU) >= delta:
        return (delta / np.linalg.norm(pU)) * pU

    # Solve for dogleg path
    p_diff = pB - pU
    a = np.dot(p_diff, p_diff)
    b = 2 * np.dot(pU, p_diff)
    c = np.dot(pU, pU) - delta**2
    tau_dl = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return pU + tau_dl * p_diff


def trust_region_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    delta0: float = 1.0,
    eta: float = 0.15,
    max_iter: int = 100,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Trust Region method for unconstrained minimization.

    Parameters:
        f        : The objective function
        grad_f   : The gradient function
        hess_f   : The Hessian function
        x0       : Initial guess
        delta0   : Initial trust region radius
        eta      : Acceptance threshold
        max_iter : Maximum iterations

    Returns:
        Tuple of (final iterate, history of iterates)
    """
    x = x0.copy()
    delta = delta0
    history = [x.copy()]

    for _ in range(max_iter):
        gk = grad_f(x)
        if np.linalg.norm(gk) < 1e-6:
            break

        Hk = hess_f(x)
        p = dogleg_trust_region(gk, Hk, delta)

        actual_reduction = f(x) - f(x + p)
        predicted_reduction = -(np.dot(gk, p) + 0.5 * np.dot(p, Hk @ p))

        rho = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0

        # Update trust region radius
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) >= delta:
            delta = min(2 * delta, 100.0)

        # Accept or reject step
        if rho > eta:
            x = x + p
            history.append(x.copy())

    return x, history


# if __name__ == "__main__":
#     # Example usage
#     A = np.array([[3, 0.5], [0.5, 1]])
#     b = np.array([1, 2])

#     def f(x):
#         return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

#     def grad_f(x):
#         return A @ x - b

#     def hess_f(x):
#         return A

#     x0 = np.array([0.0, 0.0])
#     x_final, history = trust_region_method(f, grad_f, hess_f, x0)

#     print("Trust Region Example:")
#     print(f"Initial point: {x0}")
#     print(f"Final point: {x_final}")
#     print(f"Function value decreased from {f(x0):.6f} to {f(x_final):.6f}")
#     print(f"Number of iterations: {len(history)-1}")
