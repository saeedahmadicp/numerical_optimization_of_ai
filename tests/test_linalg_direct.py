"""Direct solvers: oracle comparisons (NumPy/SciPy), failure paths, trace contract, invariants.

Tolerances are set from unit roundoff u and the condition number before running:
* backward error η = ‖b − Ax‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) ≤ 6n³·ρ·u for elimination with growth
  factor ρ (Higham 2002, Ch. 9: ‖ΔA‖∞ ≤ 2n²γ_{3n}ρ‖A‖∞, γ_{3n} ≈ 3nu); QR and Cholesky need no ρ;
* forward error vs numpy.linalg.solve ≤ 20·n·κ∞·u·‖x‖∞ (both solutions carry about κ·u error).
"""

import json

import numpy as np
import pytest
import scipy.linalg as sl
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

import numopt
from numopt import problems
from numopt.linalg.direct import BACKWARD_ERROR_FACTOR, estimate_inv_norm1, pivot_tolerance

EPS = np.finfo(np.float64).eps
ELIMINATION = (
    "gaussian_elimination_pivoting",
    "gauss_jordan",
    "lu_decomposition",
    "qr_householder",
)
ALL_METHODS = ("gaussian_elimination", *ELIMINATION, "cholesky", "thomas")
LINALG = problems.list_problems("linalg")
PROPS = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def applicable(method, p):
    tags = set(p.tags)
    if "singular" in tags:
        return False
    if method == "gaussian_elimination":
        return "needs_pivoting" not in tags
    if method == "cholesky":
        return "spd" in tags
    if method == "thomas":
        return "tridiagonal" in tags
    return True


CASES = [(m, p) for m in ALL_METHODS for p in LINALG if applicable(m, p)]


def pow2_scale(M):
    """A power of 2 near max|m_ij|: dividing by it is exact and keeps LAPACK (SVD, eigvalsh) away
    from underflow/overflow on the tiny or huge matrices that Hypothesis generates."""
    amax = float(np.max(np.abs(M), initial=0.0))
    return 2.0 ** int(np.frexp(amax)[1]) if amax > 0 else 1.0


def sigma_min(A):
    s = pow2_scale(A)
    return float(np.linalg.svd(A / s, compute_uv=False)[-1]) * s


def lambda_min(S):
    s = pow2_scale(S)
    return float(np.linalg.eigvalsh(S / s)[0]) * s


def norm2_bound(M):
    """‖M‖₂ ≤ n·max|m_ij| (no SVD needed)."""
    return M.shape[0] * float(np.max(np.abs(M), initial=0.0))


def backward_error(A, b, x):
    """Normwise backward error η (Rigal–Gaches; Higham 2002, Thm. 7.1); 0 when r = 0."""
    num = np.linalg.norm(b - A @ x, np.inf)
    if num == 0.0:
        return 0.0
    return num / (np.linalg.norm(A, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf))


# --------------------------------------------------------------------------------------
# Library problems: convergence, oracle, contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "p"), CASES, ids=[f"{m}-{p.id}" for m, p in CASES])
def test_solves_library_problem_like_numpy(method, p):
    res = numopt.run(method, p)
    assert_valid_result(res)
    assert res.converged, res.message
    n = p.b.size
    x_np = np.linalg.solve(p.A, p.b)
    kappa = np.linalg.cond(p.A, np.inf)
    assert np.linalg.norm(res.x - x_np, np.inf) <= 20 * n * kappa * EPS * np.linalg.norm(
        x_np, np.inf
    )
    rho = res.extra.get("growth_factor") or 1.0
    eta = backward_error(p.A, p.b, res.x)
    assert eta <= 6 * n**3 * rho * EPS
    # The certificate behind converged=True is the oracle's η∞ (to the log-domain rounding).
    assert_allclose(res.extra["backward_error"], eta, rtol=1e-12, atol=0)
    assert eta <= res.extra["backward_error_bound"]
    assert res.fun == pytest.approx(np.linalg.norm(p.b - p.A @ res.x), abs=1e-300, rel=1e-12)
    # Trace: k = 0 start, one Step per pivot column, solution only on the last Step.
    last = res.trace[-1]
    assert np.array_equal(last.x, res.x)
    assert all(s.x is None for s in res.trace[:-1])
    assert res.trace[0].info["phase"] == "start"
    assert [s.k for s in res.trace] == list(range(len(res.trace)))
    expected_len = n + 1 if method == "gauss_jordan" else n + 2
    assert len(res.trace) == expected_len
    # Hager–Higham estimate: a lower bound of κ₁, and exact on every library matrix.
    k1 = np.linalg.cond(p.A, 1)
    assert res.extra["cond_estimate"] <= k1 * (1 + 1e-6)
    assert_allclose(res.extra["cond_estimate"], k1, rtol=1e-10 + 10 * k1 * EPS)


@pytest.mark.parametrize("p", [p for p in LINALG if "singular" not in p.tags], ids=lambda p: p.id)
def test_lu_factors_match_scipy(p):
    res = numopt.run("lu_decomposition", p)
    P, L, U = (res.extra[k] for k in ("P", "L", "U"))
    Ps, Ls, Us = sl.lu(p.A)  # pyright: ignore[reportAssignmentType]  (A = Ps L U, Ps = Pᵀ)
    assert np.array_equal(P, Ps.T)
    scale = np.max(np.abs(p.A))
    # Both factorizations have backward error O(nε)·|A| (checked below), but the factors
    # themselves move by up to κ(A) times that (the forward error; LAPACK's blocked, FMA
    # kernels and numopt's loops round differently: 7e-15 in L of hilbert_5 on x86-64).
    fwd = 10 * p.b.size * EPS * max(1.0, float(np.linalg.cond(p.A, 1)))
    assert_allclose(L, Ls, rtol=0, atol=fwd)
    assert_allclose(U, Us, rtol=0, atol=fwd * scale)
    assert_allclose(P @ p.A, L @ U, rtol=0, atol=10 * p.b.size * EPS * scale)
    assert np.all(np.abs(L) <= 1.0)  # partial pivoting


@pytest.mark.parametrize("p", [p for p in LINALG if "spd" in p.tags], ids=lambda p: p.id)
def test_cholesky_factor_matches_scipy(p):
    res = numopt.run("cholesky", p)
    L = res.extra["L"]
    scale = np.sqrt(np.max(np.abs(p.A)))
    assert_allclose(L, sl.cholesky(p.A, lower=True), rtol=0, atol=10 * p.b.size * EPS * scale)
    assert np.array_equal(L, np.tril(L)) and np.all(np.diag(L) > 0)


@pytest.mark.parametrize("p", [p for p in LINALG if "singular" not in p.tags], ids=lambda p: p.id)
def test_householder_qr_matches_lapack(p):
    res = numopt.run("qr_householder", p)
    Q, R = res.extra["Q"], res.extra["R"]
    Qs, Rs = sl.qr(p.A)  # pyright: ignore[reportAssignmentType]  (LAPACK geqrf signs)
    n = p.b.size
    scale = np.linalg.norm(p.A, 2)
    assert_allclose(R, Rs, rtol=0, atol=20 * n * EPS * scale)
    assert_allclose(Q, Qs, rtol=0, atol=20 * n * EPS * np.linalg.cond(p.A))
    assert_allclose(Q.T @ Q, np.eye(n), rtol=0, atol=10 * n * EPS)
    for s in res.trace[1:-1]:
        v = np.asarray(s.info["householder_vector"])
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-14) or not np.any(v)


@pytest.mark.parametrize("pid", ["poisson_1d_10", "spd_2x2", "nearly_singular"])
def test_thomas_matches_solve_banded(pid):
    p = problems.get(pid)
    n = p.b.size
    ab = np.zeros((3, n))
    ab[0, 1:] = np.diag(p.A, 1)
    ab[1] = np.diag(p.A)
    ab[2, :-1] = np.diag(p.A, -1)
    x_ref = sl.solve_banded((1, 1), ab, p.b)
    res = numopt.run("thomas", p)
    kappa = np.linalg.cond(p.A, np.inf)
    assert_allclose(res.x, x_ref, rtol=0, atol=20 * n * kappa * EPS * np.max(np.abs(x_ref)))
    assert_allclose(
        res.extra["L"] @ res.extra["U"], p.A, rtol=0, atol=10 * EPS * np.max(np.abs(p.A))
    )


def test_poisson_thomas_reproduces_closed_form():
    # u_i = i(11 − i)/2 exactly; Thomas on tridiag(−1, 2, −1) has no growth.
    res = numopt.run("thomas", problems.get("poisson_1d_10"))
    i = np.arange(1, 11)
    assert_allclose(res.x, i * (11 - i) / 2, rtol=1e-14)


# --------------------------------------------------------------------------------------
# Elimination geometry in the trace
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gaussian_elimination", "gaussian_elimination_pivoting"])
def test_stage_k_clears_column_below_pivot(method):
    p = problems.get("nonsymmetric_4")
    res = numopt.run(method, p)
    n = p.b.size
    for s in res.trace[1 : n + 1]:
        c = s.info["pivot"][1]
        M = np.asarray(s.info["matrix"])
        assert M.shape == (n, n + 1)
        assert np.all(np.tril(M[:, : c + 1], -1) == 0.0)
        m = np.asarray(s.info["multipliers"])
        assert np.all(m[: c + 1] == 0.0)
        if method == "gaussian_elimination_pivoting":
            assert np.all(np.abs(m) <= 1.0)
    # The augmented system after elimination is equivalent: same solution.
    M = np.asarray(res.trace[-1].info["matrix"])
    assert_allclose(np.linalg.solve(M[:, :n], M[:, n]), p.solution, rtol=1e-13)


def test_pivoting_swaps_the_zero_pivot_away():
    res = numopt.run("gaussian_elimination_pivoting", problems.get("needs_pivoting"))
    stage1 = res.trace[1].info
    # Column 0 is (0, 1, −1): the first entry of maximal modulus is in row 1.
    assert stage1["row_swap"] == [0, 1]
    assert stage1["pivot_value"] == 1.0


def test_gauss_jordan_ends_with_identity():
    p = problems.get("diag_dominant_3")
    res = numopt.run("gauss_jordan", p)
    M = np.asarray(res.trace[-1].info["matrix"])
    assert_allclose(M[:, :3], np.eye(3), rtol=0, atol=1e-15)
    assert_allclose(M[:, 3], p.solution, rtol=1e-14)


def test_cholesky_trace_reconstructs_schur_complements():
    p = problems.get("hilbert_5")
    res = numopt.run("cholesky", p)
    for s in res.trace[1:-1]:
        c = s.info["pivot"][1]
        L = np.asarray(s.info["L"])
        W = np.asarray(s.info["matrix"])
        # A = L_{:, :c+1} L_{:, :c+1}ᵀ + (Schur complement in the trailing block).
        assert_allclose(L[:, : c + 1] @ L[:, : c + 1].T + W, p.A, rtol=0, atol=10 * 5 * EPS)


def test_fixture_cases_are_valid_and_short():
    from numopt.linalg.direct import FIXTURE_CASES

    assert 3 <= len(FIXTURE_CASES) <= 8
    assert {m for m, _, _ in FIXTURE_CASES} == set(ALL_METHODS)
    for method, pid, params in FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) < 300
        json.dumps(res.to_dict(), allow_nan=False)


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


def test_no_pivoting_breaks_down_on_zero_leading_pivot():
    res = numopt.run("gaussian_elimination", problems.get("needs_pivoting"))
    assert_valid_result(res)
    assert not res.converged and "zero pivot" in res.message and "stage 1" in res.message
    assert res.x is None and res.trace[-1].info["zero_pivot"] is True
    assert res.extra["cond_estimate"] == float("inf")


@pytest.mark.parametrize("method", ["gaussian_elimination", *ELIMINATION])
def test_singular_matrix_is_reported_not_solved(method):
    res = numopt.run(method, problems.get("singular_3"))
    assert_valid_result(res)
    assert not res.converged
    assert res.x is None
    assert "stage 3" in res.message  # the last pivot is checked too (legacy bug)


def test_cholesky_rejects_nonsymmetric_and_indefinite():
    res = numopt.run("cholesky", problems.get("nonsymmetric_4"))
    assert_valid_result(res)
    assert not res.converged and "not symmetric" in res.message and res.n_iter == 0
    indefinite = (np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([1.0, 1.0]))  # λ = 3, −1
    res = numopt.run("cholesky", indefinite)
    assert_valid_result(res)
    assert not res.converged and "not positive definite" in res.message
    assert res.trace[-1].info["pivot_value"] == pytest.approx(-3.0)  # 1 − 2²/1


@pytest.mark.parametrize("diag", [(1.0, 1e-17), (1e17, 1.0)])
def test_cholesky_tiny_positive_pivot_is_not_called_indefinite(diag):
    # Regression (audit): diag(1, 1e-17) is SPD (SciPy factors it), but its second pivot is
    # below τ = nε‖A‖_F. The method refuses it as numerically singular, not as indefinite.
    A = np.diag(diag)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    sl.cholesky(A, lower=True)  # succeeds: A is SPD in exact arithmetic
    res = numopt.run("cholesky", (A, np.ones(2)))
    assert_valid_result(res)
    assert not res.converged and res.x is None
    assert "not numerically positive definite" in res.message and "stage 2" in res.message
    assert "A is not positive definite" not in res.message
    assert 0 < res.trace[-1].info["pivot_value"] <= pivot_tolerance(A)


def test_thomas_zero_pivot_and_non_tridiagonal():
    res = numopt.run("thomas", (np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.0, 2.0])))
    assert_valid_result(res)
    assert not res.converged and "does not pivot" in res.message
    with pytest.raises(ValueError, match="tridiagonal"):
        numopt.run("thomas", problems.get("diag_dominant_3"))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_invalid_input_raises(method):
    with pytest.raises(ValueError):
        numopt.run(method, (np.ones((2, 3)), np.ones(2)))
    with pytest.raises(ValueError):
        numopt.run(method, (np.eye(2), np.ones(3)))
    with pytest.raises(ValueError):
        numopt.run(method, (np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2)))
    with pytest.raises(TypeError):
        numopt.run(method, "not a system")


def test_inputs_are_not_mutated():
    A = np.array([[0.0, 2.0, 1.0], [1.0, -2.0, -3.0], [-1.0, 1.0, 2.0]])
    b = np.array([-8.0, 0.0, 3.0])
    A0, b0 = A.copy(), b.copy()
    for method in (
        "gaussian_elimination_pivoting",
        "gauss_jordan",
        "lu_decomposition",
        "qr_householder",
    ):
        numopt.run(method, (A, b))
    assert np.array_equal(A, A0) and np.array_equal(b, b0)


def test_n_equals_one():
    for method in ALL_METHODS:
        res = numopt.run(method, (np.array([[4.0]]), np.array([2.0])))
        assert res.converged and res.x == pytest.approx([0.5])
        assert res.extra["cond_estimate"] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e308, 1.5e308])
@pytest.mark.parametrize("method", ALL_METHODS)
def test_norm_above_float_max_is_solved_not_called_singular(method, scale):
    # Regression (audit): ‖A‖_F = 2e308 overflows for A = 1e308·I₄ (κ = 1). With τ = n·ε·‖A‖_F
    # computed as ∞, every pivot counted as zero and A was called "singular to working precision".
    n = 4 if scale == 1e308 else 2
    A, b = scale * np.eye(n), scale * np.ones(n)
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.linalg.norm(A)) and np.all(np.isfinite(A))
    tau = pivot_tolerance(A)
    assert_allclose(tau, n * EPS * scale * np.sqrt(n), rtol=1e-14)
    res = numopt.run(method, (A, b))
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, np.linalg.solve(A, b), rtol=0, atol=4 * EPS)
    assert_allclose(res.extra["cond_estimate"], 1.0, rtol=1e-14)


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_elimination",
        "gaussian_elimination_pivoting",
        "gauss_jordan",
        "lu_decomposition",
        "cholesky",
    ],
)
def test_full_matrix_with_overflowing_norms_matches_numpy(method):
    # A = 1e308·(I + 0.3(𝟙𝟙ᵀ − I)), n = 4: ‖A‖_F ≈ 2.2e308 and ‖A‖₁ = 1.9e308 overflow, every entry
    # and every intermediate of elimination is finite, and A is SPD (λ = 1.9, 0.7) with κ₁ ≈ 3.6.
    # (Householder QR is not run: its reflector x₁ + ‖x‖ ≈ 2.1e308 itself exceeds the range.)
    n = 4
    A = 1e308 * (np.eye(n) + 0.3 * (np.ones((n, n)) - np.eye(n)))
    b = A @ np.array([1.0, -1.0, 1.0, -1.0])
    with np.errstate(over="ignore"):
        assert np.all(np.isfinite(b)) and not np.isfinite(np.linalg.norm(A, 1))
    res = numopt.run(method, (A, b))
    assert_valid_result(res)
    assert res.converged, res.message
    A_s = A / 2.0**1023  # exact: an oracle on representable norms
    kappa = np.linalg.cond(A_s, np.inf)
    x_np = np.linalg.solve(A, b)
    assert_allclose(res.x, x_np, rtol=0, atol=20 * n * kappa * EPS * np.max(np.abs(x_np)))
    assert_allclose(res.extra["cond_estimate"], np.linalg.cond(A_s, 1), rtol=1e-12)


# Thomas on δ = 1e-14 and 1e-8 happens to return x = (1, 1) exactly (η∞ = 0), which the
# certificate rightly accepts; δ = 1e-15 gives x₀ = 0.89 as for elimination.
@pytest.mark.parametrize(
    ("method", "delta"),
    [
        ("gaussian_elimination", 1e-15),
        ("gaussian_elimination", 1e-14),
        ("gaussian_elimination", 1e-8),
        ("thomas", 1e-15),
    ],
)
def test_small_pivot_without_pivoting_is_reported_unstable(method, delta):
    # Regression (audit): A = [[δ, 1], [1, 1]] (κ₂ ≈ 2.6) passes the pivot test |δ| > τ ≈ 7.7e-16,
    # but the growth factor 1/δ ruins x: for δ = 1e-15, x₁ = 0.888 instead of 1 (η∞ ≈ 0.03).
    A = np.array([[delta, 1.0], [1.0, 1.0]])
    b = np.array([1.0 + delta, 2.0])
    assert abs(delta) > pivot_tolerance(A)
    res = numopt.run(method, (A, b))
    assert_valid_result(res)
    assert not res.converged and "unstable" in res.message and "pivot" in res.message
    assert res.x is not None and np.all(np.isfinite(res.x))  # x is kept for the student
    eta = backward_error(A, b, res.x)
    assert eta > 30 * 2 * EPS
    assert_allclose(res.extra["backward_error"], eta, rtol=1e-12)
    if method == "gaussian_elimination":
        assert res.extra["growth_factor"] == pytest.approx(1.0 / delta, rel=1e-6)
    # Partial pivoting on the same system is backward stable and certified.
    ref = numopt.run("gaussian_elimination_pivoting", (A, b))
    assert ref.converged, ref.message
    assert_allclose(ref.x, np.linalg.solve(A, b), rtol=0, atol=10 * EPS)


def test_wilkinson_growth_defeats_partial_pivoting_but_not_qr():
    # Wilkinson's matrix (Higham 2002, §9.4): partial pivoting has growth ρ = 2^{n−1}. For n = 60,
    # ρ = 5.8e17 and the computed x has η∞ ≈ 0.05, which the pivot test alone accepted.
    n = 60
    W = np.eye(n) - np.tril(np.ones((n, n)), -1)
    W[:, -1] = 1.0
    b = W @ np.ones(n)
    for method in ("gaussian_elimination_pivoting", "lu_decomposition", "gauss_jordan"):
        res = numopt.run(method, (W, b))
        assert_valid_result(res)
        assert not res.converged and "unstable" in res.message, (method, res.message)
        assert res.extra["growth_factor"] == pytest.approx(2.0 ** (n - 1))
    qr = numopt.run("qr_householder", (W, b))
    assert qr.converged, qr.message
    assert backward_error(W, b, qr.x) <= BACKWARD_ERROR_FACTOR * n * EPS
    assert_allclose(qr.x, np.ones(n), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# Hypothesis properties
# --------------------------------------------------------------------------------------

entries = st.floats(-10, 10, allow_nan=False, allow_infinity=False, allow_subnormal=False)


@st.composite
def square_systems(draw, max_n=6):
    n = draw(st.integers(1, max_n))
    A = draw(hnp.arrays(np.float64, (n, n), elements=entries))
    b = draw(hnp.arrays(np.float64, (n,), elements=entries))
    return A, b


@PROPS
@given(square_systems())
@pytest.mark.parametrize(
    "method", ["gaussian_elimination_pivoting", "lu_decomposition", "gauss_jordan"]
)
def test_partial_pivoting_is_backward_stable_or_honestly_singular(method, system):
    A, b = system
    n = b.size
    res = numopt.run(method, (A, b))
    assert_valid_result(res)
    unstable = not res.converged and "unstable" in res.message
    if res.converged or unstable:
        # The certificate agrees with the oracle η∞ (to the log-domain rounding), in both cases.
        eta = backward_error(A, b, res.x)
        assert_allclose(res.extra["backward_error"], eta, rtol=1e-12, atol=0)
        bound = res.extra["backward_error_bound"]
        assert (eta <= bound * (1 + 1e-9)) if res.converged else (eta > bound * (1 - 1e-9))
        if unstable and method != "gauss_jordan":
            # Growth ≤ 2^{n−1} = 32 here, so an uncertified x still obeys the a-priori bound.
            assert eta <= 6 * n**3 * max(res.extra["growth_factor"], 1.0) * EPS
    if unstable:
        return
    if res.converged:
        rho = max(res.extra["growth_factor"], 1.0)
        if method == "gauss_jordan":
            # Gauss–Jordan is forward stable but not backward stable (Peters & Wilkinson 1975):
            # compare the forward error with that of an LAPACK solve, both O(n³ρκu).
            x_np = np.linalg.solve(A, b)
            if not np.any(x_np):
                assert not np.any(res.x)  # b = 0 gives x = 0 exactly
                return
            kappa = np.linalg.cond(A, np.inf)
            bound = 12 * n**3 * rho * kappa * EPS * np.linalg.norm(x_np, np.inf)
            assert np.linalg.norm(res.x - x_np, np.inf) <= bound
        else:
            assert rho <= 2.0 ** (n - 1) * (1 + 1e-12)  # Wilkinson's bound for partial pivoting
            assert backward_error(A, b, res.x) <= 6 * n**3 * rho * EPS
    elif "non-finite" in res.message:
        # Entries ≤ 10 bound the growth by 32·10, so only a solution beyond the float64 range
        # can overflow (for example A = [[2.2e-308]], b = [4]).
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        assert not np.all(np.isfinite(x_np)) or np.max(np.abs(x_np)) > 1e300
    else:
        # A pivot |u_kk| ≤ τ = n·u·‖A‖_F. With |l_ij| ≤ 1, min|u_kk| ≥ σ_min(U) ≥ σ_min(A)/‖L‖₂ and
        # ‖L‖₂ ≤ n, so σ_min(A) ≤ n·τ up to rounding in the computed pivot (factor 10 slack).
        assert "zero pivot" in res.message
        assert sigma_min(A) <= 10 * n * pivot_tolerance(A) + 10 * n**3 * EPS * norm2_bound(A)


@PROPS
@given(square_systems())
def test_householder_qr_orthogonal_and_reconstructs(system):
    A, b = system
    n = b.size
    res = numopt.run("qr_householder", (A, b))
    assert_valid_result(res)
    if not res.converged and "non-finite" in res.message:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        assert not np.all(np.isfinite(x_np)) or np.max(np.abs(x_np)) > 1e300
        return
    if not res.converged:
        assert "zero diagonal" in res.message
        smin = sigma_min(A)
        # |r_kk| ≥ σ_min(R) = σ_min(A) for an orthogonal Q, so a zero pivot means σ_min ≤ τ.
        assert smin <= pivot_tolerance(A) + 10 * n**2 * EPS * norm2_bound(A)
        return
    Q, R = res.extra["Q"], res.extra["R"]
    assert_allclose(Q.T @ Q, np.eye(n), rtol=0, atol=10 * n * EPS)
    assert_allclose(Q @ R, A, rtol=0, atol=10 * n**2 * EPS * norm2_bound(A))
    # Householder QR solve: ‖ΔA‖_F ≤ n·γ_cn‖A‖_F (Higham 2002, Ch. 19); ‖·‖_F ≤ √n‖·‖∞.
    assert backward_error(A, b, res.x) <= 10 * n**2.5 * EPS


@PROPS
@given(square_systems())
def test_cholesky_succeeds_iff_positive_definite(system):
    B, b = system
    n = b.size
    S = 0.5 * (B + B.T)
    res = numopt.run("cholesky", (S, b))
    assert_valid_result(res)
    lam_min = lambda_min(S)
    slack = 10 * n**2 * EPS * norm2_bound(S)
    if res.converged:
        # Backward stable: L Lᵀ = S + ΔS with ‖ΔS‖ = O(n u ‖S‖), and S + ΔS is SPD.
        assert lam_min >= -slack
        L = res.extra["L"]
        assert_allclose(L @ L.T, S, rtol=0, atol=slack)
    elif "non-finite" in res.message:
        # The factorization succeeded but x = S⁻¹b is beyond the float64 range.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            x_np = np.linalg.lstsq(S, b, rcond=None)[0]
        assert not np.all(np.isfinite(x_np)) or np.max(np.abs(x_np)) > 1e300
    else:
        # A Cholesky pivot d_k = 1/[(S_{1:k,1:k})⁻¹]_kk ≥ λ_min(S) when S is SPD, so d_k ≤ τ
        # implies λ_min(S) ≤ τ (plus rounding).
        assert lam_min <= pivot_tolerance(S) + slack
        d = res.trace[-1].info["pivot_value"]
        if d > 0:
            # 0 < d ≤ τ is a numerical-rank verdict; it must not claim that S is indefinite.
            assert "not numerically positive definite" in res.message
            assert "A is not positive definite" not in res.message
        else:
            assert "A is not positive definite" in res.message


@PROPS
@given(square_systems(max_n=8))
def test_thomas_on_diagonally_dominant_tridiagonal_matches_lapack(system):
    R, b = system
    n = b.size
    T = np.triu(np.tril(R, 1), -1)
    off = np.sum(np.abs(T), axis=1) - np.abs(np.diag(T))
    T = T + np.diag(np.where(np.diag(T) >= 0, 1.0, -1.0) * (off + 1.0))  # strictly dominant
    res = numopt.run("thomas", (T, b))
    assert res.converged
    x_ref = np.linalg.solve(T, b)
    kappa = np.linalg.cond(T, np.inf)
    assert_allclose(
        res.x, x_ref, rtol=0, atol=20 * n * kappa * EPS * max(np.max(np.abs(x_ref)), 1e-300)
    )


@PROPS
@given(square_systems())
def test_hager_estimate_is_a_lower_bound(system):
    A, b = system
    n = b.size
    A = A + np.diag(np.sum(np.abs(A), axis=1) + 1.0)  # strictly dominant: nonsingular, κ moderate
    inv1 = np.linalg.norm(np.linalg.inv(A), 1)  # test-only oracle; the method never forms A⁻¹
    est = estimate_inv_norm1(lambda v: np.linalg.solve(A, v), lambda v: np.linalg.solve(A.T, v), n)
    assert est <= inv1 * (1 + 1e-10)


def test_final_step_reports_the_final_permutation():
    p = problems.get("needs_pivoting")
    res = numopt.run("gaussian_elimination_pivoting", p)
    perm = res.extra["perm"]
    assert np.array_equal(res.trace[-1].info["perm"], perm)
    assert not np.array_equal(perm, np.arange(3))  # a swap happened
    assert_allclose(p.A[perm], res.extra["L"] @ res.extra["U"], rtol=0, atol=1e-14)


@PROPS
@given(square_systems())
@pytest.mark.parametrize("method", ALL_METHODS)
def test_stability_certificate_matches_oracle_backward_error(method, system):
    # converged=True ⇔ η∞ ≤ bound, with η∞ recomputed by the NumPy oracle (backward_error above).
    A, b = system
    if method == "thomas":
        A = np.triu(np.tril(A, 1), -1)
    elif method == "cholesky":
        A = 0.5 * (A + A.T)
    res = numopt.run(method, (A, b))
    assert_valid_result(res)
    if "backward_error" not in res.extra:  # a breakdown before x existed
        assert not res.converged and res.x is None
        return
    eta = backward_error(A, b, res.x)
    assert_allclose(res.extra["backward_error"], eta, rtol=1e-12, atol=0)
    bound = res.extra["backward_error_bound"]
    kappa = res.extra["cond_estimate"] if method == "gauss_jordan" else 1.0
    assert bound == BACKWARD_ERROR_FACTOR * b.size * EPS * max(1.0, kappa)
    assert res.converged == (res.extra["backward_error"] <= bound)
    if not res.converged:
        assert "unstable" in res.message and np.all(np.isfinite(res.x))
