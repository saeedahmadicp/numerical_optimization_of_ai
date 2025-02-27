# algorithms/convex/line_search.py

"""Line search methods for convex optimization.

This module implements various line search methods for determining step sizes in
optimization algorithms. These methods ensure sufficient decrease in the objective
function while maintaining reasonable step sizes.

Line search methods find a step size α that minimizes f(x + α*p) where:
- f is the objective function
- x is the current point
- p is the search direction (typically a descent direction)
- α is the step size to be determined

The implemented methods include:
- Backtracking line search (simple Armijo condition)
- Wolfe-Powell line search (Wolfe conditions)
- Strong Wolfe-Powell line search (strong Wolfe conditions)
- Goldstein-Armijo line search (Goldstein conditions)
"""

import numpy as np


def backtracking_line_search(
    f, grad_f, xk, pk, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=100, alpha_min=1e-16
):
    """
    Perform a backtracking line search to choose a step size alpha along the descent direction pk.

    This implements the Armijo condition:

    f(xk + α*pk) ≤ f(xk) + c*α*∇f(xk)ᵀpk

    where c ∈ (0, 1) is a small constant (typically 1e-4).

    The backtracking approach starts with a relatively large step size and reduces it
    by a factor of rho until the Armijo condition is satisfied.

    Parameters:
        f       : Callable. The objective function f(x).
        grad_f  : Callable. The gradient of f, which takes x as input.
        xk      : np.array. The current iterate.
        pk      : np.array. The descent direction (should satisfy grad_f(xk).T @ pk < 0).
        alpha_init: float. Initial step size (default is 1.0).
        rho     : float. Contraction factor for reducing step size (default is 0.5).
        c       : float. Armijo condition constant (default is 1e-4).
        max_iter: int. Maximum number of backtracking iterations (default is 100).
        alpha_min: float. Minimum step size (default is 1e-16).

    Returns:
        alpha   : float. A step size that satisfies the Armijo condition.
    """
    alpha = alpha_init
    fk = f(xk)
    gk = grad_f(xk)
    directional_derivative = np.dot(gk, pk)

    for _ in range(max_iter):
        if alpha < alpha_min:
            break
        if f(xk + alpha * pk) <= fk + c * alpha * directional_derivative:
            break
        alpha *= rho

    return alpha


def wolfe_line_search(
    f,
    grad_f,
    xk,
    pk,
    alpha_init=1.0,
    c1=1e-4,
    c2=0.9,
    max_iter=25,
    zoom_max_iter=10,
    alpha_min=1e-16,
):
    """
    Perform a line search that satisfies the Wolfe conditions.

    The Wolfe conditions consist of:

    1. Sufficient decrease (Armijo) condition:
       f(xk + α*pk) ≤ f(xk) + c1*α*∇f(xk)ᵀpk

    2. Curvature condition:
       ∇f(xk + α*pk)ᵀpk ≥ c2*∇f(xk)ᵀpk

    where:
    - c1 and c2 are constants with 0 < c1 < c2 < 1
    - c1 is typically small (e.g., 1e-4)
    - c2 is typically large (e.g., 0.9)

    Parameters:
        f          : Callable. The objective function f(x).
        grad_f     : Callable. The gradient of f, which takes x as input.
        xk         : np.array. The current iterate.
        pk         : np.array. The descent direction.
        alpha_init : float. Initial step size.
        c1         : float. Parameter for the sufficient decrease condition (0 < c1 < c2 < 1).
        c2         : float. Parameter for the curvature condition (0 < c1 < c2 < 1).
        max_iter   : int. Maximum number of iterations.
        zoom_max_iter: int. Maximum iterations for the zoom procedure.
        alpha_min  : float. Minimum step size.

    Returns:
        alpha      : float. Step size satisfying the Wolfe conditions.
    """

    def phi(alpha):
        return f(xk + alpha * pk)

    def dphi(alpha):
        return np.dot(grad_f(xk + alpha * pk), pk)

    alpha_0 = 0.0
    alpha_1 = alpha_init

    phi_0 = phi(alpha_0)
    dphi_0 = dphi(alpha_0)

    # If the directional derivative is positive, the direction is not a descent direction
    if dphi_0 >= 0:
        print("Warning: Direction is not a descent direction")
        return alpha_min

    phi_1 = phi(alpha_1)

    i = 0
    while i < max_iter:
        # Check if current step size violates sufficient decrease condition
        if phi_1 > phi_0 + c1 * alpha_1 * dphi_0 or (i > 0 and phi_1 >= phi(alpha_0)):
            # Use zoom to find a step size between alpha_0 and alpha_1
            return _zoom(
                f,
                grad_f,
                xk,
                pk,
                alpha_0,
                alpha_1,
                phi,
                dphi,
                phi_0,
                dphi_0,
                c1,
                c2,
                zoom_max_iter,
            )

        dphi_1 = dphi(alpha_1)

        # Check if curvature condition is satisfied
        if abs(dphi_1) <= -c2 * dphi_0:
            return alpha_1

        # If the slope is positive, use zoom to find a step size between alpha_0 and alpha_1
        if dphi_1 >= 0:
            return _zoom(
                f,
                grad_f,
                xk,
                pk,
                alpha_1,
                alpha_0,
                phi,
                dphi,
                phi_0,
                dphi_0,
                c1,
                c2,
                zoom_max_iter,
            )

        # Update alpha_0 and alpha_1
        alpha_0 = alpha_1
        alpha_1 = 2.0 * alpha_1  # Simple heuristic: double the step size

        phi_0 = phi_1
        phi_1 = phi(alpha_1)

        i += 1

    # If we reach maximum iterations, return the last alpha
    return alpha_1


def strong_wolfe_line_search(
    f,
    grad_f,
    xk,
    pk,
    alpha_init=1.0,
    c1=1e-4,
    c2=0.1,
    max_iter=25,
    zoom_max_iter=10,
    alpha_min=1e-16,
):
    """
    Perform a line search that satisfies the strong Wolfe conditions.

    The strong Wolfe conditions consist of:

    1. Sufficient decrease (Armijo) condition:
       f(xk + α*pk) ≤ f(xk) + c1*α*∇f(xk)ᵀpk

    2. Strong curvature condition:
       |∇f(xk + α*pk)ᵀpk| ≤ c2|∇f(xk)ᵀpk|

    The strong Wolfe conditions differ from the standard Wolfe conditions
    in that the curvature condition uses the absolute value of the directional
    derivative, which prevents the step size from being too large.

    Parameters:
        f          : Callable. The objective function f(x).
        grad_f     : Callable. The gradient of f, which takes x as input.
        xk         : np.array. The current iterate.
        pk         : np.array. The descent direction.
        alpha_init : float. Initial step size.
        c1         : float. Parameter for the sufficient decrease condition (0 < c1 < c2 < 1).
        c2         : float. Parameter for the curvature condition (0 < c1 < c2 < 1).
                     For strong Wolfe, c2 is typically smaller (e.g., 0.1).
        max_iter   : int. Maximum number of iterations.
        zoom_max_iter: int. Maximum iterations for the zoom procedure.
        alpha_min  : float. Minimum step size.

    Returns:
        alpha      : float. Step size satisfying the strong Wolfe conditions.
    """

    def phi(alpha):
        return f(xk + alpha * pk)

    def dphi(alpha):
        return np.dot(grad_f(xk + alpha * pk), pk)

    alpha_0 = 0.0
    alpha_1 = alpha_init

    phi_0 = phi(alpha_0)
    dphi_0 = dphi(alpha_0)

    # If the directional derivative is positive, the direction is not a descent direction
    if dphi_0 >= 0:
        print("Warning: Direction is not a descent direction")
        return alpha_min

    phi_1 = phi(alpha_1)

    i = 0
    while i < max_iter:
        # Check if current step size violates sufficient decrease condition
        if phi_1 > phi_0 + c1 * alpha_1 * dphi_0 or (i > 0 and phi_1 >= phi(alpha_0)):
            # Use zoom to find a step size between alpha_0 and alpha_1
            return _zoom_strong(
                f,
                grad_f,
                xk,
                pk,
                alpha_0,
                alpha_1,
                phi,
                dphi,
                phi_0,
                dphi_0,
                c1,
                c2,
                zoom_max_iter,
            )

        dphi_1 = dphi(alpha_1)

        # Check if strong curvature condition is satisfied
        if abs(dphi_1) <= c2 * abs(dphi_0):
            return alpha_1

        # If the slope is positive, use zoom to find a step size between alpha_0 and alpha_1
        if dphi_1 >= 0:
            return _zoom_strong(
                f,
                grad_f,
                xk,
                pk,
                alpha_1,
                alpha_0,
                phi,
                dphi,
                phi_0,
                dphi_0,
                c1,
                c2,
                zoom_max_iter,
            )

        # Update alpha_0 and alpha_1
        alpha_0 = alpha_1
        alpha_1 = 2.0 * alpha_1  # Simple heuristic: double the step size

        phi_0 = phi_1
        phi_1 = phi(alpha_1)

        i += 1

    # If we reach maximum iterations, return the last alpha
    return alpha_1


def goldstein_line_search(
    f,
    grad_f,
    xk,
    pk,
    alpha_init=1.0,
    c=0.1,
    max_iter=100,
    alpha_min=1e-16,
    alpha_max=1e10,
):
    """
    Perform a line search that satisfies the Goldstein conditions.

    The Goldstein conditions provide both an upper and lower bound on the step size:

    1. Lower bound (sufficient decrease):
       f(xk + α*pk) ≤ f(xk) + c*α*∇f(xk)ᵀpk

    2. Upper bound (prevent too small steps):
       f(xk + α*pk) ≥ f(xk) + (1-c)*α*∇f(xk)ᵀpk

    where c ∈ (0, 0.5) is a constant.

    The Goldstein conditions are symmetric around the sufficient decrease line,
    which helps avoid excessively small steps while still ensuring descent.

    Parameters:
        f          : Callable. The objective function f(x).
        grad_f     : Callable. The gradient of f, which takes x as input.
        xk         : np.array. The current iterate.
        pk         : np.array. The descent direction.
        alpha_init : float. Initial step size.
        c          : float. Parameter for Goldstein conditions (0 < c < 0.5).
        max_iter   : int. Maximum number of iterations.
        alpha_min  : float. Minimum step size.
        alpha_max  : float. Maximum step size.

    Returns:
        alpha      : float. Step size satisfying the Goldstein conditions.
    """
    fk = f(xk)
    gk = grad_f(xk)
    directional_derivative = np.dot(gk, pk)

    # If the directional derivative is positive, the direction is not a descent direction
    if directional_derivative >= 0:
        print("Warning: Direction is not a descent direction")
        return alpha_min

    alpha_lo = alpha_min
    alpha_hi = alpha_max
    alpha = alpha_init

    for _ in range(max_iter):
        fk_new = f(xk + alpha * pk)

        # Check lower bound (sufficient decrease)
        lower_bound = fk + c * alpha * directional_derivative

        # Check upper bound (prevent too small steps)
        upper_bound = fk + (1 - c) * alpha * directional_derivative

        # If both conditions are satisfied, return the step size
        if fk_new <= lower_bound and fk_new >= upper_bound:
            return alpha

        # If lower bound is violated, reduce alpha
        if fk_new > lower_bound:
            alpha_hi = alpha

        # If upper bound is violated, increase alpha
        elif fk_new < upper_bound:
            alpha_lo = alpha

        # Bisection step
        if alpha_hi < alpha_max:
            alpha = 0.5 * (alpha_lo + alpha_hi)
        else:
            alpha = 2.0 * alpha_lo

        # Check if alpha is too small
        if alpha < alpha_min:
            print("Warning: Step size too small in Goldstein line search")
            return alpha_min

    # If max iterations reached, return the current alpha
    return alpha


def _zoom(
    f, grad_f, xk, pk, alpha_lo, alpha_hi, phi, dphi, phi_0, dphi_0, c1, c2, max_iter
):
    """
    Helper function for Wolfe line search to zoom in on a good step size.

    This function finds a step size between alpha_lo and alpha_hi that satisfies
    the Wolfe conditions.

    Parameters:
        f, grad_f  : The objective function and its gradient.
        xk, pk     : Current point and descent direction.
        alpha_lo, alpha_hi: Low and high bounds for the step size.
        phi, dphi  : Functions to compute f(xk + alpha*pk) and its derivative.
        phi_0, dphi_0: Values at alpha=0.
        c1, c2     : Parameters for the Wolfe conditions.
        max_iter   : Maximum number of iterations.

    Returns:
        alpha      : Step size satisfying the Wolfe conditions.
    """
    for i in range(max_iter):
        # Bisection or quadratic interpolation could be used here
        alpha = 0.5 * (alpha_lo + alpha_hi)

        phi_alpha = phi(alpha)

        # Check sufficient decrease condition
        if phi_alpha > phi_0 + c1 * alpha * dphi_0 or phi_alpha >= phi(alpha_lo):
            alpha_hi = alpha
        else:
            dphi_alpha = dphi(alpha)

            # Check curvature condition
            if abs(dphi_alpha) <= -c2 * dphi_0:
                return alpha

            # Update interval based on sign of derivative
            if dphi_alpha * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha

    # Return the last value if max iterations reached
    return alpha


def _zoom_strong(
    f, grad_f, xk, pk, alpha_lo, alpha_hi, phi, dphi, phi_0, dphi_0, c1, c2, max_iter
):
    """
    Helper function for Strong Wolfe line search to zoom in on a good step size.

    This function finds a step size between alpha_lo and alpha_hi that satisfies
    the strong Wolfe conditions.

    Parameters:
        f, grad_f  : The objective function and its gradient.
        xk, pk     : Current point and descent direction.
        alpha_lo, alpha_hi: Low and high bounds for the step size.
        phi, dphi  : Functions to compute f(xk + alpha*pk) and its derivative.
        phi_0, dphi_0: Values at alpha=0.
        c1, c2     : Parameters for the strong Wolfe conditions.
        max_iter   : Maximum number of iterations.

    Returns:
        alpha      : Step size satisfying the strong Wolfe conditions.
    """
    for i in range(max_iter):
        # Bisection or quadratic interpolation could be used here
        alpha = 0.5 * (alpha_lo + alpha_hi)

        phi_alpha = phi(alpha)

        # Check sufficient decrease condition
        if phi_alpha > phi_0 + c1 * alpha * dphi_0 or phi_alpha >= phi(alpha_lo):
            alpha_hi = alpha
        else:
            dphi_alpha = dphi(alpha)

            # Check strong curvature condition
            if abs(dphi_alpha) <= c2 * abs(dphi_0):
                return alpha

            # Update interval based on sign of derivative
            if dphi_alpha * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha

    # Return the last value if max iterations reached
    return alpha
