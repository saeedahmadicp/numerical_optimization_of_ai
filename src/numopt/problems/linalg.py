"""Linear systems ``A x = b`` for the direct and iterative solvers of :mod:`numopt.linalg`.

Each system is a :class:`~numopt.core.types.LinearSystem`. ``solution`` is the exact solution of
the system as written in exact arithmetic (``None`` when the system has no unique solution).

Tags describe the structural properties the solvers depend on, so the web app can tell a student
which methods apply. Every tag is verified by ``tests/test_problems_linalg.py``:

* ``spd``                    symmetric positive definite (Cholesky, CG, steepest descent apply)
* ``symmetric``              ``A = Aᵀ``
* ``nonsymmetric``           ``A ≠ Aᵀ`` (GMRES, LU, QR; not CG/Cholesky)
* ``tridiagonal``            nonzero entries only on the three central diagonals (Thomas applies)
* ``diagonally_dominant``    strictly row diagonally dominant (Jacobi and Gauss–Seidel converge)
* ``ill_conditioned``        nonsingular with κ₂(A) ≥ 1e5
* ``needs_pivoting``         a zero leading pivot: elimination without row swaps breaks down
* ``jacobi_diverges``        ρ(T_J) > 1 while ρ(T_GS) < 1
* ``singular``               ``det A = 0``
* ``2d``                     2 × 2: each equation is a line, so iterates can be drawn in the plane
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..core.types import LinearSystem
from .registry import factory


def _frozen(values: object) -> np.ndarray:
    """A read-only float64 copy, so no method can mutate the shared problem data."""
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


@factory("linalg")
def spd_2x2() -> LinearSystem:
    # Shewchuk (1994), "An Introduction to the Conjugate Gradient Method Without the Agonizing
    # Pain", eq. (4): the quadratic form ½xᵀAx − bᵀx has elliptical contours, λ = 2 and 7.
    return LinearSystem(
        id="spd_2x2",
        name="2×2 SPD system (Shewchuk)",
        A=_frozen([[3.0, 2.0], [2.0, 6.0]]),
        b=_frozen([2.0, -8.0]),
        solution=_frozen([2.0, -2.0]),
        description=(
            "Shewchuk's 2×2 example: eigenvalues 2 and 7 (κ₂ = 3.5). Its solution minimizes "
            "φ(x) = ½xᵀAx − bᵀx, so iterates can be drawn on the elliptical contours of φ. "
            "ρ(T_J) = √2/3 ≈ 0.471, ρ(T_GS) = 2/9."
        ),
        tags=("spd", "symmetric", "tridiagonal", "diagonally_dominant", "2d"),
    )


@factory("linalg")
def diag_dominant_3() -> LinearSystem:
    return LinearSystem(
        id="diag_dominant_3",
        name="3×3 diagonally dominant system",
        A=_frozen([[4.0, -1.0, 1.0], [-1.0, 4.0, -2.0], [1.0, -2.0, 4.0]]),
        b=_frozen([12.0, -1.0, 5.0]),
        solution=_frozen([3.0, 1.0, 1.0]),
        description=(
            "Strictly diagonally dominant and SPD (κ₂ ≈ 3.37). Every method in the family "
            "applies; ρ(T_J) ≈ 0.683 and ρ(T_GS) ≈ 0.177."
        ),
        tags=("spd", "symmetric", "diagonally_dominant"),
    )


@factory("linalg")
def poisson_1d_10() -> LinearSystem:
    # -u'' = f on (0, 1), u(0) = u(1) = 0, central differences with h = 1/11, scaled by h²:
    # tridiag(-1, 2, -1) u = h² f. With h² f ≡ 1 the discrete solution is u_i = i(11 - i)/2
    # (a quadratic, whose second difference is exactly -1).
    n = 10
    A = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    i = np.arange(1, n + 1, dtype=np.float64)
    return LinearSystem(
        id="poisson_1d_10",
        name="1-D Poisson matrix, n = 10",
        A=_frozen(A),
        b=_frozen(np.ones(n)),
        solution=_frozen(i * (n + 1 - i) / 2.0),
        description=(
            "The scaled second-difference matrix tridiag(−1, 2, −1) of −u″ = 1 on (0, 1), "
            "h = 1/11. Eigenvalues 2 − 2cos(jπ/11), κ₂ ≈ 48.4. ρ(T_J) = cos(π/11) ≈ 0.959, "
            "so Jacobi is slow; SOR with ω* = 2/(1 + sin(π/11)) ≈ 1.560 is much faster."
        ),
        tags=("spd", "symmetric", "tridiagonal"),
    )


@factory("linalg")
def hilbert_5() -> LinearSystem:
    n = 5
    H = [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]
    # b = H·1 summed in exact rational arithmetic, then rounded once to float64.
    b = [float(sum(row, Fraction(0))) for row in H]
    return LinearSystem(
        id="hilbert_5",
        name="Hilbert matrix, n = 5",
        A=_frozen([[float(h) for h in row] for row in H]),
        b=_frozen(b),
        solution=_frozen(np.ones(n)),
        description=(
            "H_ij = 1/(i + j − 1): SPD but κ₂ ≈ 4.77e5, so about 5–6 of the 16 significant "
            "digits are lost. `solution` is the exact solution of the exact rational system; "
            "the exact solution of the rounded float system differs from it by about 2e-12 "
            "(max-norm; the bound κ·u ≈ 5e-11)."
        ),
        tags=("spd", "symmetric", "ill_conditioned"),
    )


@factory("linalg")
def needs_pivoting() -> LinearSystem:
    return LinearSystem(
        id="needs_pivoting",
        name="Zero leading pivot",
        A=_frozen([[0.0, 2.0, 1.0], [1.0, -2.0, -3.0], [-1.0, 1.0, 2.0]]),
        b=_frozen([-8.0, 0.0, 3.0]),
        solution=_frozen([-4.0, -5.0, 2.0]),
        description=(
            "det A = 1, but a₁₁ = 0: Gaussian elimination without row interchanges breaks down "
            "at the first stage, while partial pivoting solves it. The zero diagonal entry also "
            "makes Jacobi and Gauss–Seidel undefined."
        ),
        tags=("nonsymmetric", "needs_pivoting"),
    )


@factory("linalg")
def nearly_singular() -> LinearSystem:
    # delta = 2^-30 makes 1 + delta and 2 + delta exactly representable, so the stored float
    # system has exactly the solution (1, 1).
    delta = 2.0**-30
    return LinearSystem(
        id="nearly_singular",
        name="Two nearly parallel lines",
        A=_frozen([[1.0, 1.0], [1.0, 1.0 + delta]]),
        b=_frozen([2.0, 2.0 + delta]),
        solution=_frozen([1.0, 1.0]),
        description=(
            "x + y = 2 and x + (1 + δ)y = 2 + δ with δ = 2⁻³⁰: two lines that meet at (1, 1) "
            "at an angle of about 4.7e-10 rad (≈ δ/2). κ₂ ≈ 4.3e9, so a relative change of 1e-10 in b "
            "can move the solution by O(1). ρ(T_J) ≈ 1 − 4.7e-10."
        ),
        tags=("spd", "symmetric", "tridiagonal", "ill_conditioned", "2d"),
    )


@factory("linalg")
def jacobi_diverges() -> LinearSystem:
    return LinearSystem(
        id="jacobi_diverges",
        name="Jacobi diverges, Gauss–Seidel converges",
        A=_frozen([[2.0, -1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 2.0]]),
        b=_frozen([-1.0, 4.0, -5.0]),
        solution=_frozen([1.0, 2.0, -1.0]),
        description=(
            "A classic splitting example: ρ(T_J) = √5/2 ≈ 1.118 > 1, so Jacobi "
            "diverges, but ρ(T_GS) = 1/2, so Gauss–Seidel converges. Not diagonally dominant."
        ),
        tags=("nonsymmetric", "jacobi_diverges"),
    )


@factory("linalg")
def nonsymmetric_4() -> LinearSystem:
    return LinearSystem(
        id="nonsymmetric_4",
        name="4×4 nonsymmetric system",
        A=_frozen(
            [
                [4.0, 1.0, 0.0, 2.0],
                [-1.0, 5.0, 2.0, 0.0],
                [0.0, -2.0, 6.0, 1.0],
                [3.0, 0.0, -1.0, 7.0],
            ]
        ),
        b=_frozen([8.0, 7.0, -9.0, 11.0]),
        solution=_frozen([1.0, 2.0, -1.0, 1.0]),
        description=(
            "Nonsymmetric, strictly row diagonally dominant, κ₂ ≈ 3.15, with a complex pair of "
            "eigenvalues 5.5 ± 2.1i. CG does not apply; GMRES, LU and QR do."
        ),
        tags=("nonsymmetric", "diagonally_dominant"),
    )


@factory("linalg")
def singular_3() -> LinearSystem:
    return LinearSystem(
        id="singular_3",
        name="Singular 3×3 system",
        A=_frozen([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        b=_frozen([6.0, 15.0, 24.0]),
        solution=None,
        description=(
            "rank A = 2 with null space span{(1, −2, 1)}. b = A·(1, 1, 1) is consistent, so "
            "there are infinitely many solutions (1, 1, 1) + t(1, −2, 1) and no unique one. "
            "Direct methods must report a zero pivot; the minimum-norm solution is (1, 1, 1)."
        ),
        tags=("nonsymmetric", "singular"),
    )
