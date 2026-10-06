"""The linear-system library: every stored solution, tag and quoted number is checked."""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numopt import problems
from numopt.core.types import LinearSystem

EPS = np.finfo(np.float64).eps
REQUIRED = (
    "spd_2x2",
    "diag_dominant_3",
    "poisson_1d_10",
    "hilbert_5",
    "needs_pivoting",
    "nearly_singular",
    "jacobi_diverges",
    "nonsymmetric_4",
    "singular_3",
)
ALL = problems.list_problems("linalg")


def _splitting_radii(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    rho_j = max(abs(np.linalg.eigvals(-np.linalg.solve(D, L + U))))
    rho_gs = max(abs(np.linalg.eigvals(-np.linalg.solve(D + L, U))))
    return rho_j, rho_gs


def test_required_ids_are_registered_as_linalg():
    for pid in REQUIRED:
        assert problems.kind_of(pid) == "linalg"
        assert isinstance(problems.get(pid), LinearSystem)


@pytest.mark.parametrize("p", ALL, ids=lambda p: p.id)
def test_shapes_finite_readonly_and_json(p):
    n = p.b.size
    assert p.A.shape == (n, n)
    assert np.all(np.isfinite(p.A)) and np.all(np.isfinite(p.b))
    with pytest.raises(ValueError):
        p.A[0, 0] = 1.0  # read-only: a method cannot corrupt the shared library
    json.dumps(p.to_dict(), allow_nan=False)
    assert p.description and p.tags


@pytest.mark.parametrize("p", [p for p in ALL if p.solution is not None], ids=lambda p: p.id)
def test_stored_solution_solves_the_system(p):
    A, b, x = p.A, p.b, p.solution
    # Backward error of the stored solution: exact data are exact (0); Hilbert's rounded data
    # leave O(u) per entry, so ‖Ax − b‖∞ ≤ n·u·(‖A‖∞‖x‖∞ + ‖b‖∞).
    bound = (
        b.size
        * EPS
        * (np.linalg.norm(A, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf))
    )
    assert np.linalg.norm(A @ x - b, np.inf) <= bound
    # Forward error against an independent solver: ≤ κ∞·(backward error) with a factor 10 slack.
    kappa = np.linalg.cond(A, np.inf)
    x_np = np.linalg.solve(A, b)
    assert np.linalg.norm(x_np - x, np.inf) <= 10 * b.size * kappa * EPS * np.linalg.norm(x, np.inf)


@pytest.mark.parametrize("p", ALL, ids=lambda p: p.id)
def test_tags_are_true(p):
    A = p.A
    n = p.b.size
    sym = np.array_equal(A, A.T)
    tags = set(p.tags)
    assert ("symmetric" in tags) == sym
    assert ("nonsymmetric" in tags) == (not sym)
    if "spd" in tags:
        assert sym and np.linalg.eigvalsh(A)[0] > 0
    i, j = np.indices(A.shape)
    assert ("tridiagonal" in tags) == bool(np.all(A[np.abs(i - j) > 1] == 0))
    off = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    assert ("diagonally_dominant" in tags) == bool(np.all(np.abs(np.diag(A)) > off))
    singular = np.linalg.matrix_rank(A) < n
    assert ("singular" in tags) == singular
    assert ("ill_conditioned" in tags) == (not singular and np.linalg.cond(A) >= 1e5)
    assert ("2d" in tags) == (n == 2)
    if "needs_pivoting" in tags:
        assert A[0, 0] == 0 and np.linalg.matrix_rank(A) == n
    if "jacobi_diverges" in tags:
        rho_j, rho_gs = _splitting_radii(A)
        assert rho_j > 1 > rho_gs


def test_quoted_spectral_facts():
    # spd_2x2: eigenvalues 2 and 7; ρ_J = √2/3, ρ_GS = 2/9 (ρ_GS = ρ_J² for 2×2).
    A = problems.get("spd_2x2").A
    assert_allclose(np.linalg.eigvalsh(A), [2.0, 7.0], rtol=1e-14)
    assert_allclose(_splitting_radii(A), [math.sqrt(2) / 3, 2 / 9], rtol=1e-13)
    # jacobi_diverges: ρ_J = √5/2, ρ_GS = 1/2.
    assert_allclose(
        _splitting_radii(problems.get("jacobi_diverges").A), [math.sqrt(5) / 2, 0.5], rtol=1e-13
    )
    # poisson_1d_10: eigenvalues 2 − 2cos(jπ/11) and ρ_J = cos(π/11).
    P = problems.get("poisson_1d_10").A
    lam = 2 - 2 * np.cos(np.arange(1, 11) * np.pi / 11)
    assert_allclose(np.linalg.eigvalsh(P), np.sort(lam), rtol=1e-13)
    assert_allclose(_splitting_radii(P)[0], math.cos(math.pi / 11), rtol=1e-13)
    # hilbert_5 and nearly_singular condition numbers as quoted.
    assert_allclose(np.linalg.cond(problems.get("hilbert_5").A), 4.766e5, rtol=1e-3)
    assert_allclose(np.linalg.cond(problems.get("nearly_singular").A), 4.295e9, rtol=1e-3)


def test_quoted_geometry_and_rounding_facts():
    # nearly_singular: the angle between the lines is the angle between their normals
    # (1, 1) and (1, 1 + δ): tan θ = δ/(2 + δ), so θ ≈ δ/2 ≈ 4.66e-10 (the description says 4.7e-10).
    p = problems.get("nearly_singular")
    n1, n2 = p.A[0], p.A[1]
    theta = math.atan2(abs(n1[0] * n2[1] - n1[1] * n2[0]), float(n1 @ n2))
    assert_allclose(theta, 2.0**-31, rtol=1e-9)
    assert "4.7e-10" in p.description
    # hilbert_5: solve the *stored* float64 system exactly in rational arithmetic. Its solution
    # differs from (1, …, 1) by ≈ 1.78e-12 in the max-norm, below the bound κ₂·u ≈ 5.3e-11.
    h = problems.get("hilbert_5")
    n = h.b.size
    M = [
        [Fraction(float(v)) for v in row] + [Fraction(float(bi))]
        for row, bi in zip(h.A, h.b, strict=True)
    ]
    for c in range(n):  # exact Gaussian elimination (all pivots of an SPD matrix are positive)
        for i in range(c + 1, n):
            m = M[i][c] / M[c][c]
            M[i] = [a - m * q for a, q in zip(M[i], M[c], strict=True)]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum((M[i][j] * x[j] for j in range(i + 1, n)), Fraction(0))) / M[i][i]
    dev = max(abs(float(xi - 1)) for xi in x)
    assert 1e-12 <= dev <= 3e-12
    assert dev <= np.linalg.cond(h.A) * EPS / 2
    assert "2e-12" in h.description and "5e-11" in h.description


def test_singular_3_is_consistent_with_the_stated_null_space():
    p = problems.get("singular_3")
    assert p.solution is None
    assert np.linalg.matrix_rank(p.A) == 2
    assert np.array_equal(p.A @ np.array([1.0, -2.0, 1.0]), np.zeros(3))
    assert np.array_equal(p.A @ np.ones(3), p.b)  # consistent
    # (1, 1, 1) ⟂ null space, so it is the minimum-norm solution.
    assert_allclose(np.linalg.pinv(p.A) @ p.b, np.ones(3), rtol=1e-12)
