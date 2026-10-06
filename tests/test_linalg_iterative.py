"""Iterative solvers: oracles (SciPy CG/GMRES, NumPy solve, matrix splitting), theory checks
(spectral radius, Young's ω*, Kahan, minimal residual, energy decrease), failure paths, contract.

Error tolerances follow from the stopping test: ‖b − A x‖₂ ≤ tol·‖b‖₂ implies
‖x − x*‖₂ ≤ ‖A⁻¹‖₂ · ‖b − A x‖₂, which is checked with the true residual (no free constant).
"""

import json
import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse.linalg as spl
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import numopt
from numopt import problems
from numopt.core.rng import Rng

EPS = np.finfo(np.float64).eps
PROPS = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SPD_KRYLOV = ("steepest_descent_linear", "conjugate_gradient_linear", "preconditioned_cg")
STATIONARY = ("jacobi", "gauss_seidel", "sor")


def error_bound_holds(p, x):
    """‖x − x*‖₂ ≤ ‖A⁻¹‖₂‖b − Ax‖₂ + (error of the reference solve, ≈ κ·u·‖x*‖₂)."""
    x_star = np.linalg.solve(p.A, p.b)
    inv_norm = 1.0 / np.linalg.svd(p.A, compute_uv=False)[-1]
    r = np.linalg.norm(p.b - p.A @ x)
    slack = 10 * np.linalg.cond(p.A) * EPS * np.linalg.norm(x_star)
    return np.linalg.norm(x - x_star) <= inv_norm * r * (1 + 1e-8) + slack


# Problems on which theory guarantees convergence for each method.
CONVERGES = {
    "jacobi": ["spd_2x2", "diag_dominant_3", "nonsymmetric_4"],  # strictly diagonally dominant
    "gauss_seidel": [
        "spd_2x2",
        "diag_dominant_3",
        "poisson_1d_10",
        "nonsymmetric_4",
        "jacobi_diverges",
    ],
    "sor": ["spd_2x2", "diag_dominant_3", "poisson_1d_10"],  # SPD, 0 < ω < 2 (Ostrowski–Reich)
    "steepest_descent_linear": ["spd_2x2", "diag_dominant_3", "poisson_1d_10"],
    "conjugate_gradient_linear": ["spd_2x2", "diag_dominant_3", "poisson_1d_10", "hilbert_5"],
    "preconditioned_cg": ["spd_2x2", "diag_dominant_3", "poisson_1d_10", "hilbert_5"],
    "gmres": [p.id for p in problems.list_problems("linalg") if p.solution is not None],
}
CASES = [(m, pid) for m, pids in CONVERGES.items() for pid in pids]


@pytest.mark.parametrize(("method", "pid"), CASES, ids=[f"{m}-{p}" for m, p in CASES])
def test_converges_where_theory_says_so(method, pid):
    p = problems.get(pid)
    res = numopt.run(method, p, tol=1e-10)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    assert res.n_iter == res.trace[-1].k
    assert np.linalg.norm(p.b - p.A @ res.x) <= 1e-10 * np.linalg.norm(p.b)
    assert error_bound_holds(p, res.x)
    # Trace: residual info agrees with the iterate (recurrences may differ by rounding only).
    for s in res.trace:
        r_true = p.b - p.A @ np.asarray(s.x)
        assert abs(s.info["residual_norm"] - np.linalg.norm(r_true)) <= 1e-8 * np.linalg.norm(p.b)
        assert s.info["relative_residual"] == pytest.approx(
            s.info["residual_norm"] / np.linalg.norm(p.b)
        )
    if method in SPD_KRYLOV or method == "gmres":
        assert res.extra["n_matvec"] >= res.n_iter + 1


def test_fixture_cases_are_valid_and_short():
    from numopt.linalg.iterative import FIXTURE_CASES

    assert 3 <= len(FIXTURE_CASES) <= 8
    assert {m for m, _, _ in FIXTURE_CASES} == set(CONVERGES)
    for method, pid, params in FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) < 300
        json.dumps(res.to_dict(), allow_nan=False)


# --------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ["spd_2x2", "diag_dominant_3", "poisson_1d_10", "hilbert_5"])
def test_cg_iterates_match_scipy(pid):
    p = problems.get(pid)
    iterates = []
    spl.cg(
        p.A,
        p.b,
        x0=np.zeros(p.b.size),
        rtol=1e-10,
        atol=0.0,
        callback=lambda xk: iterates.append(xk.copy()),
    )
    res = numopt.run("conjugate_gradient_linear", p, tol=1e-10)
    ours = [s.x for s in res.trace[1:]]
    assert len(ours) == len(iterates)
    # Same Hestenes–Stiefel recurrence; only the summation order of dot products may differ.
    for a, b in zip(ours, iterates, strict=True):
        assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))


@pytest.mark.parametrize("pid", ["spd_2x2", "diag_dominant_3", "poisson_1d_10", "hilbert_5"])
def test_pcg_matches_scipy_cg_with_jacobi_preconditioner(pid):
    p = problems.get(pid)
    n = p.b.size
    iterates = []
    x_s, info = spl.cg(
        p.A,
        p.b,
        x0=np.zeros(n),
        rtol=1e-10,
        atol=0.0,
        M=np.diag(1.0 / np.diag(p.A)),  # SciPy's M approximates A⁻¹: M = D⁻¹
        callback=lambda xk: iterates.append(xk.copy()),
    )
    assert info == 0
    res = numopt.run("preconditioned_cg", p, tol=1e-10)
    assert res.converged, res.message
    ours = [s.x for s in res.trace[1:]]
    assert len(ours) == len(iterates)
    if pid == "hilbert_5":
        # κ₂(D^{-1/2}AD^{-1/2}) ≈ 2.1e5: in exact arithmetic PCG ends at k = n = 5, but rounding
        # destroys conjugacy and delays convergence (Meurant & Strakoš 2006, Acta Numerica). An
        # exact rational PCG shows that both codes are ≈ 3e-3 away from the exact x₅ (z = r/d here,
        # z = D⁻¹r as a matrix product in SciPy), so only the final solutions are comparable.
        assert error_bound_holds(p, res.x) and error_bound_holds(p, x_s)
        return
    for a, b in zip(ours, iterates, strict=True):
        assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))


@pytest.mark.parametrize(
    ("pid", "restart"),
    [("nonsymmetric_4", 2), ("nonsymmetric_4", 4), ("jacobi_diverges", 2), ("poisson_1d_10", 10)],
)
def test_gmres_residual_history_matches_scipy(pid, restart):
    # NOTE: poisson_1d_10 with restart = 3 (159 inner steps) is checked against the exact-arithmetic
    # oracle in test_gmres_cycles_match_exact_arithmetic instead. b = 𝟙 is reflection-symmetric,
    # so exact GMRES(3) never excites the antisymmetric eigenvectors; a rounding error of 1e-15
    # in that direction grows to ≈ 2e-8 over 40 cycles and changes the tail of the history by
    # 40 %. SciPy itself moves from 159 to 162 steps (history differences 47 %) when b gets an
    # antisymmetric perturbation of 1e-15, and whether a run keeps exact bitwise symmetry depends
    # on the BLAS rounding of A·v on strided vectors. A 1e-8 match along that trajectory is not a
    # property of GMRES.
    p = problems.get(pid)
    hist = []
    x_s, _ = spl.gmres(
        p.A,
        p.b,
        x0=np.zeros(p.b.size),
        rtol=1e-10,
        atol=0.0,
        restart=restart,
        maxiter=200,
        callback=lambda v: hist.append(v),
        callback_type="pr_norm",
    )
    res = numopt.run("gmres", p, restart=restart, tol=1e-10)
    rel = [s.info["relative_residual"] for s in res.trace[1:]]
    assert len(rel) == len(hist)
    # Both report |g_{j+1}|/‖b‖ of the same Givens-rotated least-squares problem. Restarts begin
    # from slightly different x (rounding), so near the floor the difference is ≈ u‖A‖‖x‖/‖b‖.
    assert_allclose(rel, hist, rtol=1e-8, atol=1e-13)
    assert_allclose(res.x, x_s, rtol=0, atol=1e-12 * np.max(np.abs(x_s)))


def _exact_min_residuals(A: np.ndarray, b: np.ndarray, x: np.ndarray, m: int) -> list[float]:
    """min over z ∈ K_j(A, r) of ‖r − A z‖₂, j = 1 … m, r = b − A x, in exact rational arithmetic.

    The minimizer solves the normal equations (AK_j)ᵀ(AK_j) c = (AK_j)ᵀ r with the monomial
    Krylov basis K_j = [r, A r, …, A^{j−1} r]; in rationals this is exact whatever the conditioning.
    """
    n = b.size
    Af = [[Fraction(float(v)) for v in row] for row in A]

    def mv(v: list[Fraction]) -> list[Fraction]:
        return [sum((Af[i][k] * v[k] for k in range(n)), Fraction(0)) for i in range(n)]

    def dot(u: list[Fraction], v: list[Fraction]) -> Fraction:
        return sum((a * c for a, c in zip(u, v, strict=True)), Fraction(0))

    Ax = mv([Fraction(float(v)) for v in x])
    r = [Fraction(float(bi)) - axi for bi, axi in zip(b, Ax, strict=True)]
    K = [r]
    out = []
    for _ in range(m):
        AK = [mv(v) for v in K]
        j = len(K)
        M = [[dot(u, v) for v in AK] + [dot(u, r)] for u in AK]  # [G | rhs]
        for col in range(j):  # Gaussian elimination, exact
            piv = next(i for i in range(col, j) if M[i][col] != 0)
            M[col], M[piv] = M[piv], M[col]
            for i in range(col + 1, j):
                f = M[i][col] / M[col][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[col], strict=True)]
        c = [Fraction(0)] * j
        for i in reversed(range(j)):
            c[i] = (M[i][j] - sum((M[i][t] * c[t] for t in range(i + 1, j)), Fraction(0))) / M[i][i]
        res = [
            ri - sum((ct * akv[i] for ct, akv in zip(c, AK, strict=True)), Fraction(0))
            for i, ri in enumerate(r)
        ]
        out.append(math.sqrt(float(dot(res, res))))
        K.append(mv(K[-1]))
    return out


@pytest.mark.parametrize(("pid", "restart"), [("poisson_1d_10", 3), ("nonsymmetric_4", 2)])
def test_gmres_cycles_match_exact_arithmetic(pid, restart):
    # Oracle independent of the trajectory: from each cycle start x_c (read from the trace), the
    # Givens value |g_{j+1}| of inner step j must equal the exact minimum of ‖b − A x‖₂ over
    # x_c + K_j(A, r_c). The code starts from fl(b − A x_c), whose error is at most
    # γ_{n+1}(‖b‖₂ + ‖A‖_F‖x_c‖₂) (Higham 2002, §3.5); the minimum residual moves by about that
    # much, plus the O(u‖r_c‖) rounding of Arnoldi and the rotations.
    p = problems.get(pid)
    n = p.b.size
    res = numopt.run("gmres", p, restart=restart, tol=1e-10)
    assert res.converged, res.message
    a_norm = np.linalg.norm(p.A)
    x_c = np.zeros(n)
    for cycle in range(res.extra["cycles"]):
        steps = [s for s in res.trace[1:] if s.info["cycle"] == cycle]
        exact = _exact_min_residuals(p.A, p.b, x_c, len(steps))
        atol = 10 * n * EPS * (np.linalg.norm(p.b) + a_norm * np.linalg.norm(x_c))
        assert_allclose([s.info["residual_norm"] for s in steps], exact, rtol=1e-12, atol=atol)
        x_c = np.asarray(steps[-1].x)


def test_gmres_on_consistent_singular_system_finds_minimum_norm_solution():
    # b ∈ R(A) and N(A) ∩ R(A) = {0}: from x₀ = 0 the Krylov space lies in R(A) = N(A)^⊥, so
    # GMRES converges to the minimum-norm solution pinv(A) b = (1, 1, 1).
    p = problems.get("singular_3")
    res = numopt.run("gmres", p)
    assert res.converged
    assert_allclose(res.x, np.linalg.pinv(p.A) @ p.b, rtol=1e-10)


@pytest.mark.parametrize("method", STATIONARY)
@pytest.mark.parametrize("pid", ["diag_dominant_3", "nonsymmetric_4", "jacobi_diverges"])
def test_stationary_sweeps_equal_matrix_splitting(method, pid):
    # Independent formulation: x_{k+1} = (D + ωL)⁻¹(ωb + ((1 − ω)D − ωU) x_k), ω = 1 for GS;
    # Jacobi: x_{k+1} = D⁻¹(b − (L + U) x_k).
    p = problems.get(pid)
    A, b = p.A, p.b
    D, Lo, Up = np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)
    omega = {"jacobi": None, "gauss_seidel": 1.0, "sor": 1.1}[method]
    params = {"omega": omega} if method == "sor" else {}
    res = numopt.run(method, p, max_iter=12, tol=1e-15, **params)
    x = np.zeros(b.size)
    for s in res.trace[1:]:
        if omega is None:
            x = np.linalg.solve(D, b - (Lo + Up) @ x)
        else:
            x = np.linalg.solve(D + omega * Lo, omega * b + ((1 - omega) * D - omega * Up) @ x)
        assert_allclose(s.x, x, rtol=1e-12, atol=1e-13)
        if method != "jacobi":
            sweep = np.asarray(s.info["sweep"])
            assert sweep.shape == (b.size + 1, b.size)
            assert_allclose(sweep[-1], s.x, rtol=0, atol=0)
            # One coordinate changes per sub-step: the 2-D staircase.
            assert all(np.count_nonzero(sweep[i + 1] - sweep[i]) <= 1 for i in range(b.size))


# --------------------------------------------------------------------------------------
# Theory: spectral radius, asymptotic rate, Young, Kahan
# --------------------------------------------------------------------------------------


def test_spectral_radii_match_closed_forms():
    jd = problems.get("jacobi_diverges")
    assert numopt.run("jacobi", jd, max_iter=1).extra["spectral_radius"] == pytest.approx(
        math.sqrt(5) / 2, rel=1e-13
    )
    assert numopt.run("gauss_seidel", jd, max_iter=1).extra["spectral_radius"] == pytest.approx(
        0.5, rel=1e-12
    )
    spd = problems.get("spd_2x2")
    res = numopt.run("jacobi", spd)
    assert res.trace[0].info["spectral_radius"] == pytest.approx(math.sqrt(2) / 3, rel=1e-13)
    assert "spectral_radius" not in res.trace[1].info  # k = 0 only


@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel"])
def test_asymptotic_rate_equals_spectral_radius(method):
    res = numopt.run(method, problems.get("poisson_1d_10"), tol=1e-14, max_iter=3000)
    r = [float(s.info["residual_norm"]) for s in res.trace]
    k0, k1 = 100, 300  # well above the rounding floor (GS reaches tol 1e-14 at k ≈ 390)
    rate = (r[k1] / r[k0]) ** (1 / (k1 - k0))
    # The subdominant modes decay like (λ₂/λ₁)^k; after 100 sweeps they are below 1e-3 relative.
    assert rate == pytest.approx(res.extra["spectral_radius"], rel=1e-3)


def test_young_optimal_omega_on_poisson():
    p = problems.get("poisson_1d_10")
    w_star = 2 / (1 + math.sin(math.pi / 11))  # ρ_J = cos(π/11) ⇒ √(1 − ρ_J²) = sin(π/11)
    res = numopt.run("sor", p, omega=w_star)
    assert res.extra["omega_opt"] == pytest.approx(w_star, rel=1e-12)
    assert res.extra["spectral_radius_opt"] == pytest.approx(w_star - 1, rel=1e-12)
    # At ω* the eigenvalue ω* − 1 is defective (a 2×2 Jordan block), so the computed ρ(G_ω*) is
    # only accurate to O(√u) ≈ 1e-8 (Wilkinson): the tolerance reflects that, not a modelling gap.
    assert res.extra["spectral_radius"] == pytest.approx(w_star - 1, abs=1e-6)
    gs = numopt.run("gauss_seidel", p)
    assert res.converged and gs.converged and res.n_iter < gs.n_iter / 3


def test_optimal_omega_not_claimed_without_theory():
    res = numopt.run("sor", problems.get("diag_dominant_3"), omega=1.1)
    assert res.extra["omega_opt"] is None and "tridiagonal" in res.extra["omega_opt_note"]
    res = numopt.run("sor", problems.get("spd_2x2"), omega=1.0)  # ω = 1 still reports ω*
    rho_j = math.sqrt(2) / 3
    assert res.extra["omega_opt"] == pytest.approx(2 / (1 + math.sqrt(1 - rho_j**2)), rel=1e-13)


def test_cg_terminates_in_n_steps_and_beats_steepest_descent():
    p = problems.get("poisson_1d_10")
    cg = numopt.run("conjugate_gradient_linear", p)
    sd = numopt.run("steepest_descent_linear", p)
    assert cg.converged and cg.n_iter <= p.b.size
    assert sd.converged and sd.n_iter > 10 * cg.n_iter
    kappa = np.linalg.cond(p.A)
    assert cg.trace[0].info["condition_number"] == pytest.approx(kappa, rel=1e-10)
    assert cg.extra["rate_bound"] == pytest.approx(
        (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1), rel=1e-10
    )
    assert sd.extra["rate_bound"] == pytest.approx((kappa - 1) / (kappa + 1), rel=1e-10)


def test_preconditioned_condition_number():
    p = problems.get("hilbert_5")
    res = numopt.run("preconditioned_cg", p)
    s = np.sqrt(np.diag(p.A))
    expected = np.linalg.cond(p.A / np.outer(s, s))
    assert res.extra["preconditioned_condition_number"] == pytest.approx(expected, rel=1e-6)
    assert res.extra["preconditioned_condition_number"] < res.extra["condition_number"]


def test_steepest_descent_residuals_are_orthogonal_on_spd_2x2():
    # Exact line search ⇒ r_{k+1} ⟂ r_k: the zig-zag path (Shewchuk 1994, §4).
    res = numopt.run("steepest_descent_linear", problems.get("spd_2x2"))
    rs = [np.asarray(s.info["residual"]) for s in res.trace]
    for a, b in pairwise(rs):
        assert abs(a @ b) <= 1e-12 * np.linalg.norm(a) * np.linalg.norm(b) + 1e-300
    for k, s in enumerate(res.trace[1:], start=1):
        assert_allclose(
            s.x,
            np.asarray(res.trace[k - 1].x) + s.info["alpha"] * rs[k - 1],
            rtol=1e-14,
            atol=1e-15,
        )


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


def test_jacobi_diverges_and_says_why():
    res = numopt.run("jacobi", problems.get("jacobi_diverges"), max_iter=25)
    assert_valid_result(res, max_iter=25)
    assert not res.converged and "max_iter" in res.message and "diverges" in res.message
    norms = [float(s.info["residual_norm"]) for s in res.trace]
    # ρ(G_J) = √5/2 ⇒ growth ≈ ρ^25 ≈ 16.3 (the dominant pair ±i√5/2 has equal modulus).
    assert norms[-1] / norms[0] == pytest.approx((math.sqrt(5) / 2) ** 25, rel=0.25)


def test_divergence_to_overflow_is_reported_without_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a method must not emit RuntimeWarnings
        res = numopt.run("jacobi", problems.get("hilbert_5"), max_iter=5000)
    assert_valid_result(res, max_iter=5000)
    assert not res.converged and "non-finite" in res.message and "ρ(G) = 3.444" in res.message


@pytest.mark.parametrize("method", STATIONARY)
def test_zero_diagonal_stops_at_k0(method):
    res = numopt.run(method, problems.get("needs_pivoting"))
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0 and "a_00 = 0" in res.message


@pytest.mark.parametrize("method", [*STATIONARY, *SPD_KRYLOV, "gmres"])
def test_max_iter_is_reported(method):
    p = problems.get("poisson_1d_10")
    res = numopt.run(method, p, max_iter=2, tol=1e-14)
    assert_valid_result(res, max_iter=2)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 2 and len(res.trace) == 3


@pytest.mark.parametrize("method", SPD_KRYLOV)
def test_spd_methods_reject_nonsymmetric_and_detect_indefinite(method):
    res = numopt.run(method, problems.get("nonsymmetric_4"))
    assert_valid_result(res)
    assert not res.converged and "not symmetric" in res.message and res.n_iter == 0
    indefinite = (np.diag([1.0, -1.0]), np.array([1.0, 1.0]))  # r₀ᵀA r₀ = 0
    res = numopt.run(method, indefinite)
    assert_valid_result(res)
    assert not res.converged and "not positive definite" in res.message


def test_pcg_rejects_nonpositive_diagonal():
    res = numopt.run("preconditioned_cg", (np.array([[0.0, 1.0], [1.0, 0.0]]), np.ones(2)))
    assert not res.converged and "Jacobi preconditioner" in res.message


def test_invalid_parameters_raise():
    p = problems.get("spd_2x2")
    for omega in (0.0, 2.0, -1.0, 2.5):
        with pytest.raises(ValueError, match="omega"):
            numopt.run("sor", p, omega=omega)
    with pytest.raises(ValueError, match="restart"):
        numopt.run("gmres", p, restart=0)
    with pytest.raises(ValueError, match="x0"):
        numopt.run("jacobi", p, x0=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="outside"):
        numopt.run("conjugate_gradient_linear", (1e-200 * np.eye(2), np.ones(2)))


ALL_ITERATIVE = (*STATIONARY, *SPD_KRYLOV, "gmres")


@pytest.mark.parametrize("method", ALL_ITERATIVE)
@pytest.mark.parametrize("bad", [0, -5, 2.5, True, None])
def test_max_iter_below_one_or_not_integer_raises(method, bad):
    # Regression (audit): max_iter = −5 gave n_iter = −5 with trace[-1].k = 0 (Jacobi), and
    # max_iter = 0 ran one GMRES step (2 Steps > max_iter + 1).
    with pytest.raises(ValueError, match="max_iter"):
        numopt.run(method, problems.get("spd_2x2"), max_iter=bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_gmres_restart_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="restart"):
        numopt.run("gmres", problems.get("spd_2x2"), restart=bad)


@pytest.mark.parametrize("method", ALL_ITERATIVE)
def test_max_iter_one_is_honoured(method):
    res = numopt.run(method, problems.get("poisson_1d_10"), max_iter=1)
    assert_valid_result(res, max_iter=1)
    assert res.n_iter == res.trace[-1].k == 1


@pytest.mark.parametrize("method", [*STATIONARY, "gmres"])
@pytest.mark.parametrize("scale", [1e308, 1.5e308])
def test_norm_above_float_max_is_rejected_with_rescale_advice(method, scale):
    # Regression (audit): A = 1e308·I₄, b = 1e308·𝟙 are finite but ‖A‖_F, ‖b‖₂ ≈ 2e308 overflow;
    # the methods stopped at k = 0 with "‖r‖/‖b‖ = nan". Scaling by 2⁻¹⁰ is exact and solves it.
    n = 4 if scale == 1e308 else 2
    A, b = scale * np.eye(n), scale * np.ones(n)
    with pytest.raises(ValueError, match="rescale"):
        numopt.run(method, (A, b))
    res = numopt.run(method, (A / 2.0**10, b / 2.0**10))
    assert res.converged, res.message
    # Stopping test ⇒ ‖x − 𝟙‖₂ ≤ ‖A⁻¹‖₂·tol·‖b‖₂ = tol·√n (A, b scaled alike).
    assert np.linalg.norm(res.x - np.ones(n)) <= 1e-10 * np.sqrt(n) * (1 + 1e-12)


@pytest.mark.parametrize("method", SPD_KRYLOV)
def test_huge_x0_gives_no_false_breakdown(method):
    # Regression (audit): with x₀ = 1e160·𝟙, r₀ ≈ 1e161 and r₀ᵀr₀ overflowed to ∞, α = ∞/∞ = NaN,
    # and the methods reported "non-finite iterate at k = 1" with x = NaN. With scaled inner
    # products the first step lands near x* = (2, −2); fl(x₀ + α₀p₀) then carries an absolute
    # error of about u·1e160 ≈ 1e144 that a recurrence-residual method cannot see, so the honest
    # verdict is the residual gap (GMRES and Jacobi restart from the true residual and converge).
    p = problems.get("spd_2x2")
    res = numopt.run(method, p, x0=[1e160, 1e160])
    assert_valid_result(res)
    assert all(np.all(np.isfinite(s.x)) for s in res.trace)
    assert "non-finite" not in res.message and "positive definite" not in res.message
    assert not res.converged and "true residual" in res.message
    assert np.linalg.norm(res.x - p.solution) <= 1e3 * EPS * 1e160  # rounding of the first step
    # Oracle for the first step: α₀ = r₀ᵀz₀ / z₀ᵀA z₀ (z₀ = r₀, or D⁻¹r₀ for PCG) in exact
    # rational arithmetic on the float r₀ of the k = 0 Step; r₀ᵀr₀ ≈ 1e322 is not a double.
    r0 = [Fraction(float(v)) for v in res.trace[0].info["residual"]]
    d = [Fraction(float(v)) for v in np.diag(p.A)] if method == "preconditioned_cg" else [1, 1]
    z0 = [ri / di for ri, di in zip(r0, d, strict=True)]
    A = [[Fraction(float(v)) for v in row] for row in p.A]
    Az0 = [sum(A[i][j] * z0[j] for j in range(2)) for i in range(2)]
    alpha0 = sum(ri * zi for ri, zi in zip(r0, z0, strict=True)) / sum(
        zi * ai for zi, ai in zip(z0, Az0, strict=True)
    )
    assert_allclose(res.trace[1].info["alpha"], float(alpha0), rtol=10 * EPS)
    for other in ("gmres", "jacobi"):
        ref = numopt.run(other, p, x0=[1e160, 1e160])
        assert ref.converged and ref.x == pytest.approx(p.solution, abs=1e-8)


@pytest.mark.parametrize("method", [*SPD_KRYLOV, "gmres"])
def test_overflowing_initial_residual_stops_at_k0(method):
    res = numopt.run(method, problems.get("spd_2x2"), x0=[1e308, 1e308])  # A x₀ ≈ 5e308·𝟙
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0 and "overflows" in res.message


@pytest.mark.parametrize(
    ("pid", "restart"), [("nonsymmetric_4", 4), ("diag_dominant_3", 3), ("hilbert_5", 20)]
)
def test_gmres_shows_no_basis_vector_beyond_dimension_n(pid, restart):
    # Regression (audit): at krylov_dim = n, h_{n+1,n} ≈ 1e-14 is rounding noise and w/h was shown as
    # a "basis vector" with max|V_nᵀv| = 0.86. ℝⁿ holds no (n+1)-th orthonormal vector.
    p = problems.get(pid)
    n = p.b.size
    res = numopt.run("gmres", p, restart=restart, tol=1e-15)
    assert_valid_result(res)
    cycle0 = [s for s in res.trace[1:] if s.info["cycle"] == 0]
    assert any(s.info["krylov_dim"] == n for s in cycle0)
    for s in res.trace[1:]:
        if s.info["krylov_dim"] == n:
            assert s.info["basis_vector"] is None
    # One MGS pass loses orthogonality as the residual falls (Greenbaum, Rozložník & Strakoš 1997,
    # BIT 37; ≈ 6e-8 on hilbert_5, κ₂ ≈ 4.8e5); with the reorthogonalization pass the shown vectors
    # v₁ … v_n are orthonormal to rounding level on every problem (the defect showed a noise vector
    # with max|V_nᵀv| = 0.86).
    V = np.column_stack(
        [res.trace[0].info["basis_vector"]]
        + [s.info["basis_vector"] for s in cycle0 if s.info["basis_vector"] is not None]
    )
    assert V.shape[1] == n
    assert_allclose(V.T @ V, np.eye(n), rtol=0, atol=10 * n * EPS)


# Audit regression: a start residual fl(b − A x₀) dominated by rounding noise (a large x₀) made an
# Arnoldi step narrowly miss the happy-breakdown test; with one MGS pass, v_{j+1} was a normalized
# noise vector far from orthogonal to V_j, |r_jj| fell below τ = nε‖A‖_F, and GMRES reported
# "A is singular" on well-conditioned matrices (the 2 × 2 cases above hide this: there j + 1 = n).
def test_gmres_no_false_breakdown_on_poisson_from_huge_x0():
    # κ₂(A) = 48: before the fix, converged=False at k = 6 with relative error 1.5e144.
    p = problems.get("poisson_1d_10")
    n = p.b.size
    res = numopt.run("gmres", p, x0=np.full(n, 1e160))
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    assert error_bound_holds(p, res.x)
    # Root cause: the basis of the first cycle (r₀ is pure rounding noise relative to x₀) is
    # orthonormal to working precision.
    cycle0 = [s for s in res.trace[1:] if s.info["cycle"] == 0]
    V = np.column_stack(
        [res.trace[0].info["basis_vector"]]
        + [s.info["basis_vector"] for s in cycle0 if s.info["basis_vector"] is not None]
    )
    assert V.shape[1] >= 5
    assert_allclose(V.T @ V, np.eye(V.shape[1]), rtol=0, atol=10 * n * EPS)


def test_gmres_no_false_breakdown_on_diag_1122_from_large_x0():
    # The audit's shrunk case: A = diag(1, 1, 2, 2), κ = 2; before the fix "breakdown at k = 4 …
    # A is singular" although restarting from the returned x converged in one step.
    A = np.diag([1.0, 1.0, 2.0, 2.0])
    b = np.array([1.2969153998005238, -0.345672529464465, 0.8545842348534083, -0.4889690638420449])
    x0 = np.array([1760667.296993123, 199217.983013857, -382002.2921434805, 2552424.025371081])
    res = numopt.run("gmres", (A, b), x0=x0)
    assert_valid_result(res)
    assert res.converged, res.message
    assert np.linalg.norm(b - A @ res.x) <= 1e-10 * np.linalg.norm(b)
    assert_allclose(res.x, b / np.diag(A), rtol=1e-9)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_gmres_no_false_breakdown_on_identity_from_random_large_x0(n):
    # The audit found 104 of 500 failures for A = I₃ and x₀ ≈ 1e6 (Mulberry32 keeps this portable).
    rng = Rng(7)
    A = np.eye(n)
    for _ in range(200):
        b = np.array([rng.normal() for _ in range(n)])
        x0 = 1e6 * np.array([rng.normal() for _ in range(n)])
        res = numopt.run("gmres", (A, b), x0=x0)
        assert res.converged, res.message
        assert_allclose(res.x, b, rtol=0, atol=1e-10 * np.linalg.norm(b))


@pytest.mark.parametrize(
    ("pid", "restart", "scale"),
    [
        ("nonsymmetric_4", 20, 0.0),
        ("nonsymmetric_4", 2, 0.0),
        ("poisson_1d_10", 20, 0.0),
        ("poisson_1d_10", 3, 1e3),
        ("hilbert_5", 20, 1.0),
        ("singular_3", 20, 0.0),
        ("jacobi_diverges", 1, 1e2),
    ],
)
def test_gmres_residual_vector_equals_b_minus_ax(pid, restart, scale):
    # Audit regression: info["residual"] was formed before v_{j+1} was stored, so it dropped the
    # component along v_{j+1} (on nonsymmetric_4: ‖r_info‖ = 2.5 vs ‖b − Ax‖ = 6.7 at k = 1).
    p = problems.get(pid)
    n = p.b.size
    res = numopt.run("gmres", p, restart=restart, x0=scale * np.linspace(1.0, -1.0, n))
    assert_valid_result(res)
    _assert_gmres_residuals_consistent(p.A, p.b, res)


def _assert_gmres_residuals_consistent(A, b, res):
    """info["residual"] = b − A x_k and ‖info["residual"]‖ = info["residual_norm"] at every Step.

    Both sides carry rounding errors of size γ_{n+1}(‖b‖ + ‖A‖_F max‖x‖) (the start residual of
    the cycle, the product A x_k, and the O(u) Arnoldi relation A V_j = V_{j+1} H̄_j).
    """
    n = b.size
    x_max = max(float(np.linalg.norm(s.x)) for s in res.trace)
    atol = 10 * n * EPS * (np.linalg.norm(b) + np.linalg.norm(A) * x_max)
    for s in res.trace:
        r_info = np.asarray(s.info["residual"])
        assert_allclose(r_info, b - A @ np.asarray(s.x), rtol=0, atol=atol)
        assert abs(np.linalg.norm(r_info) - s.info["residual_norm"]) <= atol


ALL_ITERATIVE = (*STATIONARY, *SPD_KRYLOV, "gmres")
B_ZERO_MATRIX = np.array([[3.0, 2.0], [2.0, 6.0]])  # SPD, strictly diagonally dominant, κ₂ = 3.5


@pytest.mark.parametrize("method", ALL_ITERATIVE)
@pytest.mark.parametrize("s", [1.0, 1e-6, 1e-12, 1e12])
def test_zero_rhs_uses_a_scale_invariant_test(method, s):
    # Audit regression: for b = 0 the test was ‖r‖ ≤ tol, so with A = 1e-12·[[3, 2], [2, 6]] every
    # method accepted x₀ = (1, 1) at k = 0 (the solution is 0). Now ‖r_k‖ ≤ tol·‖r₀‖, and
    # ‖A x‖ ≤ tol‖A x₀‖ gives ‖x‖ ≤ κ₂·tol·‖x₀‖.
    A = s * B_ZERO_MATRIX
    x0 = np.array([1.0, 1.0])
    res = numopt.run(method, (A, np.zeros(2)), x0=x0, tol=1e-10, max_iter=1000)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    assert res.n_iter >= 1
    assert np.linalg.norm(res.x) <= 3.5 * 1e-10 * np.linalg.norm(x0) * (1 + 1e-8)
    assert "‖b‖" not in res.message
    r0 = np.linalg.norm(A @ x0)
    for st_ in res.trace:
        assert st_.info["relative_residual"] == pytest.approx(st_.info["residual_norm"] / r0)


@pytest.mark.parametrize("method", ALL_ITERATIVE)
def test_zero_rhs_trace_is_invariant_under_power_of_two_scaling(method):
    # Multiplying A by 2⁻⁴⁰ is exact; with b = 0 every quantity of the iteration scales exactly,
    # so the iterates, the verdict and n_iter must be bit-for-bit those of the unscaled run.
    x0 = np.array([1.0, 1.0])
    ref = numopt.run(method, (B_ZERO_MATRIX, np.zeros(2)), x0=x0)
    scaled = numopt.run(method, (2.0**-40 * B_ZERO_MATRIX, np.zeros(2)), x0=x0)
    assert ref.converged and scaled.converged
    assert ref.n_iter == scaled.n_iter
    for a, c in zip(ref.trace, scaled.trace, strict=True):
        assert np.array_equal(a.x, c.x)


@pytest.mark.parametrize(
    "method", ["steepest_descent_linear", "conjugate_gradient_linear", "gmres"]
)
@pytest.mark.parametrize("s", [1.0, 2.0**60])
def test_zero_rhs_with_x0_in_the_null_space_stops_at_k0(method, s):
    # b = 0 and fl(A x₀) = 0: x₀ solves A x = 0 to working precision (d = ‖A‖_F‖x₀‖₂ is used, so
    # the rounding-level test u‖A‖_F‖x₀‖ ≤ tol·d holds at every scale).
    A = s * np.diag([1.0, 0.0])
    res = numopt.run(method, (A, np.zeros(2)), x0=[0.0, 5.0])
    assert_valid_result(res)
    assert res.converged and res.n_iter == 0, res.message
    assert np.array_equal(res.x, [0.0, 5.0])


def test_zero_rhs_failure_message_names_the_initial_residual():
    # Jacobi diverges on jacobi_diverges (ρ(G_J) > 1): with b = 0 the verdict must say ‖r‖/‖r₀‖.
    p = problems.get("jacobi_diverges")
    res = numopt.run("jacobi", (p.A, np.zeros(p.b.size)), x0=np.ones(p.b.size), max_iter=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged
    assert "‖r‖/‖r₀‖" in res.message and "‖b‖" not in res.message


def test_zero_rhs_and_exact_start():
    p = problems.get("spd_2x2")
    for method in CONVERGES:
        res = numopt.run(method, (p.A, np.zeros(2)))
        assert res.converged and res.n_iter == 0 and np.array_equal(res.x, np.zeros(2))
        res = numopt.run(method, p, x0=p.solution)
        assert res.converged and res.n_iter == 0


# --------------------------------------------------------------------------------------
# Hypothesis properties
# --------------------------------------------------------------------------------------

# Entries in {0} ∪ ±[1e-6, 1]: tiny nonzero values would only test underflow, not the methods.
entries = st.one_of(st.just(0.0), st.floats(1e-6, 1.0), st.floats(-1.0, -1e-6))


@st.composite
def spd_systems(draw, max_n=6):
    n = draw(st.integers(1, max_n))
    B = draw(hnp.arrays(np.float64, (n, n), elements=entries))
    b = draw(hnp.arrays(np.float64, (n,), elements=entries))
    A = B.T @ B + 0.5 * np.eye(n)  # λ_min ≥ 0.5, κ ≤ 2n² + 1
    A = 0.5 * (A + A.T)
    return A, b


@st.composite
def general_systems(draw, max_n=6):
    n = draw(st.integers(1, max_n))
    B = draw(hnp.arrays(np.float64, (n, n), elements=entries))
    b = draw(hnp.arrays(np.float64, (n,), elements=entries))
    return B + 2.0 * n * np.eye(n), b  # σ_min ≥ 2n − n = n > 0


@PROPS
@given(spd_systems())
def test_cg_energy_decreases_and_directions_are_conjugate(system):
    A, b = system
    res = numopt.run("conjugate_gradient_linear", (A, b))
    assert res.converged
    assert res.n_iter <= b.size + 1
    phis = [s.info["phi"] for s in res.trace]
    scale = 1e-12 * (1 + max(abs(v) for v in phis))
    assert all(p1 <= p0 + scale for p0, p1 in pairwise(phis))
    # Consecutive directions are A-conjugate. p_kᵀA p_{k+1} = p_kᵀA r_{k+1} + β_{k+1} p_kᵀA p_k
    # vanishes in exact arithmetic; the computed r_{k+1} = r_k − α_k A p_k carries an absolute
    # error of about u(‖r_k‖ + |α_k|‖A p_k‖), and β_{k+1} p_kᵀA p_k a relative error of a few u.
    n = b.size
    for k in range(len(res.trace) - 1):
        p0 = np.asarray(res.trace[k].info["direction"])
        p1 = np.asarray(res.trace[k + 1].info["direction"])
        r0 = np.asarray(res.trace[k].info["residual"])
        alpha, beta = res.trace[k + 1].info["alpha"], res.trace[k + 1].info["beta"]
        Ap0 = A @ p0
        bound = (
            100
            * n
            * EPS
            * (
                np.linalg.norm(Ap0) * (np.linalg.norm(r0) + abs(alpha) * np.linalg.norm(Ap0))
                + abs(beta) * abs(p0 @ Ap0)
            )
        )
        assert abs(p0 @ A @ p1) <= bound + 1e-300


@PROPS
@given(spd_systems())
def test_steepest_descent_strictly_decreases_energy(system):
    A, b = system
    # spd_systems gives λ ∈ [0.5, n² + 0.5], so κ ≤ 2n² + 1 ≤ 73. With x₀ = 0,
    # ‖r_k‖/‖b‖ ≤ √κ ((κ − 1)/(κ + 1))^k (N&W eq. 3.29), which is ≤ 1e-8 for k ≥ 750.
    res = numopt.run("steepest_descent_linear", (A, b), max_iter=1000, tol=1e-8)
    phis = [s.info["phi"] for s in res.trace]
    scale = 1e-12 * (1 + max(abs(v) for v in phis))
    assert all(p1 <= p0 + scale for p0, p1 in pairwise(phis))
    assert res.converged, res.message


@PROPS
@given(general_systems(), st.integers(1, 6))
def test_gmres_residual_is_monotone_and_converges(system, restart):
    A, b = system
    res = numopt.run("gmres", (A, b), restart=restart)
    assert_valid_result(res)
    assert res.converged, res.message
    norms = [s.fun for s in res.trace]
    bnorm = np.linalg.norm(b)
    assert all(n1 <= n0 + 1e-12 * bnorm for n0, n1 in pairwise(norms))
    assert np.linalg.norm(b - A @ res.x) <= 1e-10 * max(bnorm, 1.0)
    _assert_gmres_residuals_consistent(A, b, res)


@PROPS
@given(
    general_systems(),
    st.integers(1, 6),
    st.floats(0.0, 12.0),
    hnp.arrays(np.float64, 6, elements=entries),  # no subnormals: they only test underflow
)
def test_gmres_converges_from_a_large_x0_without_false_breakdown(system, restart, log_scale, d):
    # σ_min(A) ≥ n ≫ τ = nε‖A‖_F, so a "singular" verdict is always false here. x₀ up to 1e12
    # makes the start residual fl(b − A x₀) rounding noise relative to x₀ (audit regression).
    A, b = system
    n = b.size
    x0 = 10.0**log_scale * d[:n]
    res = numopt.run("gmres", (A, b), x0=x0, restart=restart)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    assert "singular" not in res.message
    # Stopping test (d = ‖b‖₂, or ‖r₀‖₂ when b = 0) and the residual vector of every Step.
    d_scale = np.linalg.norm(b) if np.any(b) else np.linalg.norm(b - A @ x0)
    assert np.linalg.norm(b - A @ res.x) <= 1e-10 * d_scale * (1 + 1e-8) + 1e-300
    _assert_gmres_residuals_consistent(A, b, res)


@PROPS
@given(spd_systems(), st.floats(0.05, 1.95))
def test_sor_on_spd_converges_and_respects_kahan(system, omega):
    A, b = system
    res = numopt.run("sor", (A, b), omega=omega, max_iter=5000)
    rho = res.extra["spectral_radius"]
    assert rho >= abs(omega - 1) - 1e-10  # Kahan: ρ(G_ω) ≥ |ω − 1|
    assert rho < 1  # Ostrowski–Reich for SPD and 0 < ω < 2
    if rho <= 0.9:  # 0.9^5000 ≈ 1e-229: any transient growth of a non-normal G_ω has died out
        assert res.converged, res.message


@PROPS
@given(general_systems())
def test_jacobi_and_gs_converge_on_strictly_dominant(system):
    A, b = system
    for method in ("jacobi", "gauss_seidel"):
        res = numopt.run(method, (A, b))
        assert res.extra["spectral_radius"] < 1
        assert res.converged, res.message


@PROPS
@given(spd_systems())
def test_pcg_equals_cg_on_the_jacobi_scaled_system(system):
    # Two equivalent formulations (N&W §5.1, "transformed system"): PCG with M = D on A x = b
    # produces x_k = D^{-1/2} y_k, where y_k are the CG iterates on Â y = b̂ with
    # Â = D^{-1/2} A D^{-1/2}, b̂ = D^{-1/2} b. The iterates are equal in exact arithmetic.
    # In float64, while ‖r_k‖/‖b‖ ≥ 1e-6 the two differ by propagated rounding, O(n·κ₂(Â)·u)‖x*‖.
    # Near the step where exact CG terminates (a repeated eigenvalue, residual at the rounding
    # floor) no such bound holds: finite-precision CG acts like exact CG on a matrix whose
    # eigenvalues are split into clusters of width O(u‖A‖) (Greenbaum 1989, Linear Algebra
    # Appl. 113). On the shrunk example A = 5·𝟙𝟙ᵀ + …, λ = 0.5 (×4), SciPy's PCG and 200
    # u-perturbed runs differ from the exact rational x₄ by up to 2e-12 ≈ 110·n·κ̂·u·‖x*‖, so
    # there the final iterates are each checked against x* by the residual bound instead.
    A, b = system
    n = b.size
    s = np.sqrt(np.diag(A))
    A_hat = A / np.outer(s, s)
    pcg = numopt.run("preconditioned_cg", (A, b))
    cg = numopt.run("conjugate_gradient_linear", (A_hat, b / s))
    assert pcg.converged and cg.converged, (pcg.message, cg.message)
    x_star = np.linalg.solve(A, b)
    tol = 100 * n * np.linalg.cond(A_hat) * EPS * np.linalg.norm(x_star)
    for sp, sc in zip(pcg.trace, cg.trace, strict=False):
        if sp.info["relative_residual"] < 1e-6:
            break
        assert np.linalg.norm(np.asarray(sp.x) - np.asarray(sc.x) / s) <= tol + 1e-300
    # Both final solutions: ‖x − x*‖₂ ≤ ‖A⁻¹‖₂‖b − Ax‖₂ + (error of the reference, ≈ κ·u·‖x*‖).
    inv_norm = 1.0 / np.linalg.eigvalsh(A)[0]
    slack = 10 * np.linalg.cond(A) * EPS * np.linalg.norm(x_star)
    for x in (pcg.x, cg.x / s):
        r = np.linalg.norm(b - A @ x)
        assert np.linalg.norm(x - x_star) <= inv_norm * r * (1 + 1e-8) + slack + 1e-300
    # PCG minimizes the same φ(x) = ½xᵀAx − bᵀx over growing spaces: φ never increases.
    phis = [st_.info["phi"] for st_ in pcg.trace]
    scale = 1e-12 * (1 + max(abs(v) for v in phis))
    assert all(p1 <= p0 + scale for p0, p1 in pairwise(phis))


@PROPS
@given(spd_systems(), st.floats(1e100, 1e200), st.sampled_from(SPD_KRYLOV))
def test_spd_methods_report_no_false_breakdown_for_a_huge_x0(system, scale, method):
    # With ‖x₀‖ ≈ 1e100 … 1e200, r₀ᵀr₀ ≈ 1e200 … 1e400 can overflow; the scaled inner products keep
    # α, β finite, so an SPD system never reports pᵀAp ≤ 0 or a non-finite iterate.
    A, b = system
    n = b.size
    x0 = scale * np.linspace(1.0, -0.5, n)
    res = numopt.run(method, (A, b), x0=x0, max_iter=50)
    assert_valid_result(res, max_iter=50)
    assert "non-finite" not in res.message and "positive definite" not in res.message
    assert all(np.all(np.isfinite(s.x)) for s in res.trace)
