"""Tests for numopt.unconstrained.quasi_newton: BFGS, DFP, SR1, Broyden class, L-BFGS.

Oracles: SciPy ``minimize`` (BFGS, L-BFGS-B) for minimizers and iteration counts; the direct
(B-form) Broyden-class update N&W eq. 6.32, inverted with ``scipy.linalg.solve``, for every
inverse update; the explicit BFGS product-form recursion for the L-BFGS two-loop recursion;
the secant equation; positive definiteness; and the superlinear rate on Rosenbrock.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.optimize import minimize as sp_minimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.unconstrained import quasi_newton as qn_mod
from numopt.unconstrained.quasi_newton import _BFGS, _DFP, _LBFGS, _SR1, _Broyden, _two_loop

METHODS = ("bfgs", "dfp", "sr1", "broyden_class", "lbfgs")
BFGS_TYPE = ("bfgs", "dfp", "broyden_class")  # explicit H, positive definite updates
EXPLICIT_H = ("bfgs", "dfp", "sr1", "broyden_class")
EPS = float(np.finfo(float).eps)


def _nearest_min(problem: Problem, x: Any) -> np.ndarray:
    mins = [np.asarray(m, dtype=float) for m in problem.minima]
    return min(mins, key=lambda m: float(np.linalg.norm(m - np.asarray(x))))


def _start_in_domain(prob: Problem, u: float, v: float) -> list[float]:
    (x_lo, x_hi), (y_lo, y_hi) = prob.domain[0], prob.domain[1]
    return [x_lo + u * (x_hi - x_lo), y_lo + v * (y_hi - y_lo)]


def _direct_broyden(B: np.ndarray, s: np.ndarray, y: np.ndarray, phi: float) -> np.ndarray:
    """N&W eq. 6.32 (B form), written independently of the implementation."""
    Bs = B @ s
    sBs = float(s @ Bs)
    ys = float(y @ s)
    v = y / ys - Bs / sBs
    return B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / ys + phi * sBs * np.outer(v, v)


def _term_size(B: np.ndarray, s: np.ndarray, y: np.ndarray, phi: float) -> np.ndarray:
    """Entrywise size of the terms of eq. 6.32 (B form; the H form with s ↔ y and φ ↦ Φ):
    |B| + |Bs||Bs|ᵀ/sᵀBs + |y||y|ᵀ/|yᵀs| + φ·sᵀBs·aaᵀ with a = |y|/|yᵀs| + |Bs|/sᵀBs (the
    size of v = y/yᵀs − Bs/sᵀBs before it cancels). A sum's rounding error is ε times this."""
    Bs = np.abs(B @ s)
    sBs = float(s @ (B @ s))
    ys = abs(float(y @ s))
    a = np.abs(y) / ys + Bs / sBs
    return (
        np.abs(B) + np.outer(Bs, Bs) / sBs + np.outer(np.abs(y), np.abs(y)) / ys
        + phi * sBs * np.outer(a, a)
    )  # fmt: skip


def _bfgs_product(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """N&W eq. 6.17 in its product form (the implementation uses the expanded form)."""
    rho = 1.0 / float(y @ s)
    eye = np.eye(s.size)
    return (eye - rho * np.outer(s, y)) @ H @ (eye - rho * np.outer(y, s)) + rho * np.outer(s, s)


# --------------------------------------------------------------------------------------
# Convergence and the SciPy oracle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "pid", ["rosenbrock", "himmelblau", "beale", "quadratic_bowl", "quadratic_ill", "booth"]
)
def test_converges_to_a_known_minimum(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=500)
    assert res.converged, res.message
    assert np.max(np.abs(prob.grad(res.x))) <= 1e-8
    # ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)) with λ_min ≥ 0.3 on these problems.
    assert_allclose(res.x, _nearest_min(prob, res.x), rtol=0, atol=1e-7)


@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "himmelblau", "quadratic_bowl"])
def test_bfgs_matches_scipy_bfgs(pid: str) -> None:
    prob = problems.get(pid)
    ref = sp_minimize(prob.f, prob.x0, jac=prob.grad, method="BFGS", options={"gtol": 1e-8})
    assert ref.success
    for method in ("bfgs", "broyden_class"):
        res = numopt.run(method, prob)
        assert res.converged
        assert_allclose(res.x, ref.x, rtol=0, atol=1e-7)
        # Same algorithm family; SciPy has no eq. 6.20 scaling and a different first step.
        assert res.n_iter <= 2 * ref.nit + 5


@pytest.mark.parametrize("pid", ["rosenbrock", "rosenbrock_nd", "quadratic_nd", "beale"])
def test_lbfgs_matches_scipy_lbfgsb(pid: str) -> None:
    prob = problems.get(pid)
    ref = sp_minimize(
        prob.f,
        prob.x0,
        jac=prob.grad,
        method="L-BFGS-B",
        options={"gtol": 1e-9, "ftol": 1e-15, "maxcor": 10},
    )
    assert ref.success
    res = numopt.run("lbfgs", prob, m=10)
    assert res.converged
    assert_allclose(res.x, ref.x, rtol=0, atol=1e-6)
    assert res.n_iter <= 2 * ref.nit + 5


def _superlinear_tail(e: list[float]) -> bool:
    """Finite-sequence evidence that e_{k+1}/e_k → 0 (superlinear convergence).

    A linear sequence e_k = C r^k has the constant ratio r, whatever r is. The test asks for
    (a) a ratio below 1e-2 in the last three steps, and (b) a fall of the ratio by a factor of
    at least 10 within the last six steps: the last three ratios have a minimum ≤ 0.1 × the
    maximum of the three before them. (b) rejects every constant ratio; the ratios need not
    fall monotonically (they do not for DFP and SR1 on Rosenbrock).
    """
    ratios = [b / a for a, b in itertools.pairwise(e) if a > 0.0]
    if len(ratios) < 6:
        return False
    tail, before = ratios[-3:], ratios[-6:-3]
    return min(tail) <= 1e-2 and min(tail) <= 0.1 * max(before)


@pytest.mark.parametrize("rate", [0.5, 0.09, 1e-2, 1e-3])
def test_superlinear_criterion_rejects_linear_convergence(rate: float) -> None:
    # Audit regression: the old criterion (max of the last 3 ratios ≤ 0.1) accepted the
    # linear sequence 0.09^k. Every geometric sequence must fail the criterion.
    n = int(np.ceil(np.log(1e-12) / np.log(rate))) + 1
    e = [rate**k for k in range(n + 1)]
    assert e[-1] <= 1e-12
    assert not _superlinear_tail(e)
    # A sequence with e_{k+1} = e_k^1.5 (superlinear, order 1.5) passes.
    e_super = [0.5 ** (1.5**k) for k in range(12)]
    assert _superlinear_tail([v for v in e_super if v >= 1e-300])


@pytest.mark.parametrize("method", ["bfgs", "dfp", "sr1", "broyden_class"])
def test_superlinear_convergence_on_rosenbrock(method: str) -> None:
    # N&W Thm 6.6 (BFGS), and the superlinear theory of §6.2–6.4 for the others: the error
    # ratio e_{k+1}/e_k → 0. A linearly convergent method keeps a ratio bounded away from 0.
    res = numopt.run(method, problems.get("rosenbrock"), gtol=1e-12)
    assert res.converged
    e = [float(np.linalg.norm(np.asarray(s.x) - 1.0)) for s in res.trace]
    assert e[-1] <= 1e-12
    assert _superlinear_tail(e), [b / a for a, b in itertools.pairwise(e[-7:])]


# --------------------------------------------------------------------------------------
# The updates against independent formulas
# --------------------------------------------------------------------------------------


def _check_against_direct_update(res: Any, phi: float) -> None:
    """Each applied update equals the inverse of N&W eq. 6.32 with this φ."""
    n_checked = 0
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        if cur.info["update"] != "applied" or cur.info["reset"]:
            continue
        H_old = np.asarray(prev.info["H"])
        if cur.info["gamma"] is not None:  # eq. 6.20 replaced H₀ by γI before the update
            H_old = cur.info["gamma"] * np.eye(2)
        s, y = np.asarray(cur.info["s"]), np.asarray(cur.info["y"])
        B_old = scipy.linalg.solve(H_old, np.eye(2), assume_a="pos")
        B_new = _direct_broyden(B_old, s, y, phi)
        H_ref = scipy.linalg.solve(B_new, np.eye(2), assume_a="pos")
        H_new = np.asarray(cur.info["H"])
        kappa = np.linalg.cond(H_ref) * np.linalg.cond(H_old)
        # The reference: two solves (κ(H_old)κ(H_new)·ε) around the B-form update, whose terms
        # cancel: its rounding error is ε times the size of the terms, not of B_new. The
        # implementation's inverse form (dual parameter Φ) has error ε times the size of its
        # own terms. The tolerance is the sum of the two, with a factor 100.
        phi_inv = cur.info.get("phi_inverse")
        phi_inv = 1.0 - phi if phi_inv is None else phi_inv  # bfgs: Φ = 1, dfp: Φ = 0
        cancel_b = _term_size(B_old, s, y, phi).max() / np.abs(B_new).max()
        ref_err = kappa * cancel_b * np.abs(H_ref).max()
        impl_err = _term_size(H_old, y, s, phi_inv).max()  # dual: s and y swap roles
        assert_allclose(H_new, H_ref, rtol=0, atol=100 * EPS * (ref_err + impl_err))
        n_checked += 1
    assert n_checked >= 3


@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "himmelblau"])
@pytest.mark.parametrize(
    ("method", "phi"), [("bfgs", 0.0), ("dfp", 1.0), ("broyden_class", 0.3), ("broyden_class", 0.8)]
)
def test_inverse_updates_match_direct_broyden_formula(method: str, phi: float, pid: str) -> None:
    params = {"phi": phi} if method == "broyden_class" else {}
    res = numopt.run(method, problems.get(pid), **params)
    _check_against_direct_update(res, phi)


@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "himmelblau", "six_hump_camel"])
def test_broyden_endpoints_reproduce_bfgs_and_dfp(pid: str) -> None:
    prob = problems.get(pid)
    for method, phi in (("bfgs", 0.0), ("dfp", 1.0)):
        ref = numopt.run(method, prob)
        res = numopt.run("broyden_class", prob, phi=phi)
        assert res.n_iter == ref.n_iter
        assert res.converged == ref.converged
        # Different but equivalent formulas (BFGS: expanded eq. 6.17; Broyden: inverse form
        # with Φ = 1): rounding differences of order ε, amplified along the trajectory.
        for a, b in zip(res.trace[:10], ref.trace[:10], strict=True):
            assert_allclose(a.x, b.x, rtol=1e-12, atol=1e-14)
        assert_allclose(res.x, ref.x, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "beale", "himmelblau", "six_hump_camel"])
def test_secant_equation_after_every_applied_update(method: str, pid: str) -> None:
    res = numopt.run(method, problems.get(pid))
    for step in res.trace[1:]:
        if step.info["update"] != "applied":
            continue
        H = np.asarray(step.info["H"])
        s, y = np.asarray(step.info["s"]), np.asarray(step.info["y"])
        # H₊y = s holds to ~κ(H)·ε (relative to ‖s‖).
        bound = 50 * np.linalg.cond(H) * EPS * np.linalg.norm(s)
        assert np.linalg.norm(H @ y - s) <= bound


@given(
    st.sampled_from(BFGS_TYPE),
    st.sampled_from(["rosenbrock", "himmelblau", "beale", "six_hump_camel", "ackley"]),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.sampled_from(["strong_wolfe", "weak_wolfe", "backtracking"]),
)
@settings(max_examples=150, deadline=None)
def test_bfgs_type_updates_stay_symmetric_positive_definite(
    method: str, pid: str, u: float, v: float, phi: float, ls: str
) -> None:
    prob = problems.get(pid)
    params: dict[str, Any] = {"line_search": ls, "max_iter": 80}
    if method == "broyden_class":
        params["phi"] = phi
    res = numopt.run(method, prob, x0=_start_in_domain(prob, u, v), **params)
    assert_valid_result(res, max_iter=80)
    for prev, step in zip([None, *res.trace], res.trace, strict=False):
        H = np.asarray(step.info["H"])
        assert np.array_equal(H, H.T)
        eig = np.linalg.eigvalsh(H)
        # Positive definite in exact arithmetic; in floating point to rounding: on Ackley's
        # kink the curvature along the path grows without bound and λ_min(H) → 0.
        assert eig[0] >= -4 * EPS * eig[-1] and eig[-1] > 0.0
        if prev is not None:
            if step.info["reset"]:
                # Kantorovich: cos θ ≥ 2√κ/(1 + κ) > 1e-8 unless κ(H) > 4e16 (or rounding).
                eig = np.linalg.eigvalsh(np.asarray(prev.info["H"]))
                assert eig[-1] / eig[0] > 1e15
            ys = step.info["curvature"]
            if step.info["update"] == "applied":
                assert ys > 0.0 and step.info["rho"] == pytest.approx(1.0 / ys)


# --------------------------------------------------------------------------------------
# L-BFGS two-loop recursion
# --------------------------------------------------------------------------------------


@given(
    st.integers(1, 6),
    st.integers(1, 8),
    arrays(np.float64, (8, 2, 6), elements=st.floats(-1, 1)),
    arrays(np.float64, 6, elements=st.floats(-1, 1)),
    st.floats(0.01, 10.0),
)
@settings(max_examples=1000, deadline=None)
def test_two_loop_equals_explicit_bfgs_recursion(
    n: int, m: int, raw: np.ndarray, g: np.ndarray, gamma: float
) -> None:
    pairs = []
    H = gamma * np.eye(n)
    for i in range(m):
        s = raw[i, 0, :n]
        y = raw[i, 1, :n] + s  # bias towards yᵀs > 0
        ys = float(y @ s)
        if not ys > 1e-3 * np.linalg.norm(s) * np.linalg.norm(y) or ys < 1e-6:
            continue
        pairs.append((s, y, 1.0 / ys))
        H = _bfgs_product(H, s, y)
    assume(pairs)
    r = _two_loop(g[:n], pairs, gamma)
    kappa = np.linalg.cond(H)
    assume(kappa < 1e8)
    scale = max(np.abs(H).max() * np.abs(g[:n]).max(), 1e-300)
    assert_allclose(r, H @ g[:n], rtol=0, atol=100 * len(pairs) * kappa * EPS * scale)


@pytest.mark.parametrize("pid", ["rosenbrock", "himmelblau"])
@pytest.mark.parametrize("m", [1, 3, 10])
def test_lbfgs_matrix_info_is_the_limited_memory_bfgs_matrix(pid: str, m: int) -> None:
    res = numopt.run("lbfgs", problems.get(pid), m=m)
    assert res.converged
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for step in res.trace[1:]:
        s, y = np.asarray(step.info["s"]), np.asarray(step.info["y"])
        if step.info["reset"]:
            pairs = []
        if step.info["update"] == "applied":
            pairs = [*pairs, (s, y)][-m:]
        assert step.info["memory"] == len(pairs)
        s_new, y_new = pairs[-1]
        gamma = float(s_new @ y_new) / float(y_new @ y_new)  # N&W eq. 7.20
        assert step.info["gamma"] == pytest.approx(gamma, rel=1e-14)
        H = gamma * np.eye(2)
        for s_i, y_i in pairs:
            H = _bfgs_product(H, s_i, y_i)
        kappa = np.linalg.cond(H)
        assert_allclose(step.info["H"], H, rtol=0, atol=100 * m * kappa * EPS * np.abs(H).max())


def test_lbfgs_with_memory_one_differs_from_full_memory() -> None:
    prob = problems.get("rosenbrock_nd")
    one = numopt.run("lbfgs", prob, m=1)
    ten = numopt.run("lbfgs", prob, m=10)
    assert one.converged and ten.converged
    assert all(s.info["memory"] <= 1 for s in one.trace)
    assert max(s.info["memory"] for s in ten.trace) == 10
    assert ten.n_iter < one.n_iter


# --------------------------------------------------------------------------------------
# Skip rules, SR1 safeguards and resets
# --------------------------------------------------------------------------------------


def test_bfgs_skips_updates_with_negative_curvature() -> None:
    # With backtracking (no curvature condition) the Ackley run meets yᵀs ≤ 0 twice.
    res = numopt.run("bfgs", problems.get("ackley"), line_search="backtracking")
    assert_valid_result(res)
    assert res.converged
    skipped = [s for s in res.trace if s.info["update"] == "skipped"]
    assert len(skipped) >= 1
    by_k = {s.k: s for s in res.trace}
    for step in skipped:
        s, y = np.asarray(step.info["s"]), np.asarray(step.info["y"])
        assert step.info["curvature"] <= 1e-10 * np.linalg.norm(s) * np.linalg.norm(y)
        assert step.info["rho"] is None
        # H is kept unchanged.
        assert np.array_equal(step.info["H"], by_k[step.k - 1].info["H"])


def test_lbfgs_with_backtracking_stalls_in_negative_curvature() -> None:
    # Why N&W insist on Wolfe steps for BFGS-type methods: Armijo-only steps in a region
    # of negative curvature give yᵀs < 0, every pair is skipped and the direction stops
    # improving. The run must say that it did not converge.
    res = numopt.run("lbfgs", problems.get("rosenbrock"), line_search="backtracking")
    assert_valid_result(res, max_iter=500)
    assert not res.converged and "max_iter" in res.message
    assert sum(s.info["update"] == "skipped" for s in res.trace) > 100


def test_sr1_update_rules() -> None:
    upd = _SR1(2)
    H0 = upd.H.copy()
    # v = s − Hy = 0: the secant equation already holds; nothing changes.
    info = upd.update(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.0)
    assert info["update"] == "skipped" and np.array_equal(upd.H, H0)
    # vᵀy = 0 with v ≠ 0 (eq. 6.26 rejects it): with H = I, s = (2, 0), y = (1, 1) gives
    # v = s − y = (1, −1) and vᵀy = 0.
    s, y = np.array([2.0, 0.0]), np.array([1.0, 1.0])
    info = upd.update(s, y, 0.0)
    assert info["update"] == "skipped" and info["denominator"] == 0.0
    assert np.array_equal(upd.H, H0)
    # A regular pair: applied, and the secant equation holds afterwards.
    s, y = np.array([1.0, 0.5]), np.array([3.0, -1.0])
    info = upd.update(s, y, 0.0)
    assert info["update"] == "applied"
    assert_allclose(upd.H @ y, s, rtol=1e-14)
    assert np.array_equal(upd.H, upd.H.T)


def test_sr1_skips_a_zero_gradient_change() -> None:
    # Audit regression: y = 0 with v = s − Hy = s ≠ 0. Eq. 6.26 reads 0 ≥ r·0·‖v‖ and holds,
    # but vᵀy = 0, so eq. 6.25 would divide by zero. The pair must be skipped.
    upd = _SR1(2)
    H0 = upd.H.copy()
    info = upd.update(np.array([1.0, 0.0]), np.zeros(2), 0.0)
    assert info["update"] == "skipped" and info["rho"] is None
    assert info["denominator"] == 0.0
    assert np.array_equal(upd.H, H0)
    # vᵀy = 1e-310 is nonzero and passes eq. 6.26, but 1/(vᵀy) overflows: skipped, H kept.
    info = upd.update(np.array([1.0, 0.0]), np.array([1e-310, 0.0]), 0.0)
    assert info["update"] == "skipped" and np.array_equal(upd.H, H0)
    # A tiny but representable vᵀy (1e-200) is applied, and the secant equation holds.
    s, y = np.array([1.0, 0.0]), np.array([1e-200, 0.0])
    info = upd.update(s, y, 0.0)
    assert info["update"] == "applied" and np.all(np.isfinite(upd.H))
    assert_allclose(upd.H @ y, s, rtol=1e-14)


@pytest.mark.parametrize(
    ("pid", "x0"),
    [
        ("goldstein_price", None),
        ("beale", [3.4982937603749944, 3.746915414760627]),
    ],
)
def test_sr1_with_backtracking_does_not_raise_on_a_zero_pair(pid: str, x0: Any) -> None:
    # Audit regression: these runs reach a pair with y = 0 (s ≈ 1e-18 at the precision floor
    # of f) and used to raise ZeroDivisionError in the SR1 update. From this Beale start SR1
    # follows Beale's flat valley y → 1, x → −∞ and stops there with a failed line search, or
    # (on CPUs that round the last bits of f differently) is still walking down the valley at
    # max_iter. The update rule itself is tested from given values in
    # test_sr1_skips_a_zero_gradient_change.
    prob = problems.get(pid)
    res = numopt.run("sr1", prob, x0=x0, line_search="backtracking")
    assert_valid_result(res, max_iter=500)
    if not res.converged:
        assert (
            "line search failed" in res.message and "rounding level" in res.message
        ) or res.message.startswith("reached max_iter=500"), res.message
    if pid == "goldstein_price":
        assert_allclose(res.x, _nearest_min(prob, res.x), rtol=0, atol=1e-5)
    pairs = [s for s in res.trace[1:] if not np.any(s.info["y"])]
    assert all(s.info["update"] == "skipped" for s in pairs)
    for s in res.trace:
        assert s.info["H"] is not None and np.all(np.isfinite(s.info["H"]))


def test_sr1_on_a_linear_function_fails_without_raising() -> None:
    # f = x + y: y = 0 for every pair (∇f is constant), and f is unbounded below.
    res = numopt.run("sr1", lambda x: float(x[0] + x[1]), x0=[0.0, 0.0], line_search="backtracking")
    assert_valid_result(res)
    assert not res.converged and "line search failed" in res.message


@pytest.mark.parametrize("make", [_BFGS, _DFP, lambda n: _Broyden(n, 0.5), lambda n: _LBFGS(n, 5)])
def test_pair_with_underflowing_yty_is_skipped(make: Any) -> None:
    # Audit regression: yᵀs = 1e-170 passes yᵀs > 10⁻¹⁰‖s‖‖y‖ = 1e-180, but yᵀy = 1e-340
    # underflows to 0, so γ = yᵀs/yᵀy (eqs. 6.20, 7.20) was a division by zero.
    approx = make(1)
    H0 = approx.matrix()
    info = approx.update(np.array([1.0]), np.array([1e-170]), 1.0)
    assert info["update"] == "skipped" and info["rho"] is None
    assert np.array_equal(approx.matrix(), H0)


def test_broyden_mu_does_not_underflow() -> None:
    # Audit regression: s = y = 1e-100 (n = 1). (yᵀs)² = 1e-400 underflows to 0, which made
    # μ = yᵀHy·sᵀBs/(yᵀs)² a division by zero. In the factored form μ = 1, so with γ = 1 and
    # φ = 0.5 the dual parameter is Φ = 0.5/(0.5 + 0.5) = 0.5 and H⁺ = 1 (secant: H⁺y = s).
    approx = _Broyden(1, 0.5)
    info = approx.update(np.array([1e-100]), np.array([1e-100]), 0.0)
    assert info["update"] == "applied" and info["gamma"] == 1.0
    assert info["phi_inverse"] == 0.5
    assert_allclose(approx.H, [[1.0]], rtol=1e-15)


def test_sr1_resets_to_steepest_descent_when_not_descending() -> None:
    res = numopt.run("sr1", problems.get("rosenbrock"))
    assert_valid_result(res)
    assert res.converged
    resets = [s for s in res.trace if s.info["reset"]]
    assert resets, "SR1 on Rosenbrock produces indefinite H; the reset path must be used"
    by_k = {s.k: s for s in res.trace}
    for step in resets:
        # The reset direction is exactly −∇f at the previous iterate.
        assert np.array_equal(
            np.asarray(step.info["direction"]), -np.asarray(by_k[step.k - 1].info["grad"])
        )


@given(
    st.sampled_from(METHODS),
    st.sampled_from(["rosenbrock", "himmelblau", "six_hump_camel", "beale", "rastrigin"]),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.sampled_from(["strong_wolfe", "weak_wolfe", "backtracking", "goldstein"]),
)
@settings(max_examples=200, deadline=None)
def test_every_step_is_a_descent_step_with_sufficient_decrease(
    method: str, pid: str, u: float, v: float, ls: str
) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob, x0=_start_in_domain(prob, u, v), line_search=ls, max_iter=60)
    assert_valid_result(res, max_iter=60)
    c1 = 0.25 if ls == "goldstein" else 1e-4
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        p = np.asarray(cur.info["direction"])
        g_prev = np.asarray(prev.info["grad"])
        slope = float(g_prev @ p)
        assert -slope > 1e-8 * np.linalg.norm(g_prev) * np.linalg.norm(p)
        assert cur.step_size is not None and cur.fun is not None and prev.fun is not None
        assert cur.fun <= prev.fun + c1 * cur.step_size * slope
        if cur.info["reset"]:
            assert np.array_equal(p, -g_prev)


# --------------------------------------------------------------------------------------
# Counts, fallbacks, failure paths and the contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("ls", ["strong_wolfe", "backtracking"])
def test_evaluation_counts_are_exact(method: str, ls: str) -> None:
    base = problems.get("rosenbrock")
    calls = {"f": 0, "g": 0}

    def f(x: np.ndarray) -> float:
        calls["f"] += 1
        return base.f(x)

    def g(x: np.ndarray) -> np.ndarray:
        calls["g"] += 1
        return base.grad(x)

    res = numopt.minimize(f, x0=base.x0, grad=g, method=method, line_search=ls, max_iter=60)
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], 0)
    trials = sum(len(s.info["trials"]) for s in res.trace)
    assert res.n_fev == 1 + trials
    if ls == "backtracking":
        assert res.n_gev == 1 + res.n_iter  # one gradient per accepted point


def test_finite_difference_gradient_is_counted() -> None:
    base = problems.get("rosenbrock")
    calls = {"f": 0}

    def f(x: np.ndarray) -> float:
        calls["f"] += 1
        return base.f(x)

    res = numopt.minimize(f, x0=base.x0, method="bfgs", gtol=1e-6)
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, 1.0], atol=1e-5)
    assert res.n_fev == calls["f"]
    assert res.n_fev >= 4 * res.n_gev  # each central-difference gradient costs 2n = 4 values


def _nan_gradient(x: np.ndarray) -> np.ndarray:
    """∇(x² + 10y²), but NaN for x < 0.05 (a gradient routine that fails in a region)."""
    if x[0] < 0.05:
        return np.array([np.nan, np.nan])
    return np.array([2.0 * x[0], 20.0 * x[1]])


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_gradient_keeps_the_approximation_state(method: str) -> None:
    # Audit regression: when ∇f(x_k) is not finite no update is attempted, and the last
    # Step.info must describe the kept approximation (L-BFGS: memory and γ_k unchanged, not
    # an empty memory).
    res = numopt.minimize(
        lambda x: float(x[0] ** 2 + 10.0 * x[1] ** 2),
        x0=[3.0, 1.0],
        grad=_nan_gradient,
        method=method,
        line_search="backtracking",
    )
    assert_valid_result(res)
    assert not res.converged and "not finite" in res.message
    assert res.n_iter >= 2
    prev, last = res.trace[-2].info, res.trace[-1].info
    assert not last["reset"]
    assert last["update"] == "skipped" and last["rho"] is None
    assert np.array_equal(last["H"], prev["H"])  # H is unchanged
    if method == "lbfgs":
        assert last["memory"] == prev["memory"] >= 1
        assert last["gamma"] == prev["gamma"] != 1.0
    else:
        assert last["gamma"] is None
    for key in ("denominator", "phi_inverse"):
        if key in last:
            assert last[key] is None


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_is_reported(method: str) -> None:
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4


@pytest.mark.parametrize("method", METHODS)
def test_unbounded_below_fails_in_the_line_search(method: str) -> None:
    res = numopt.minimize(
        lambda x: float(x[0] + 2.0 * x[1]),
        x0=[0.0, 0.0],
        grad=lambda x: np.array([1.0, 2.0]),
        method=method,
    )
    assert_valid_result(res)
    assert not res.converged
    assert "line search failed" in res.message and "unbounded" in res.message


def test_precision_floor_is_reported_honestly() -> None:
    # Bohachevsky has O(1) constant terms: at ‖∇f‖∞ ≈ 5e-8 the predicted decrease is below
    # ε·|f|, so no step can pass the Armijo test and gtol = 1e-8 is unattainable. SciPy's
    # BFGS calls the same situation "precision loss".
    res = numopt.run("bfgs", problems.get("bohachevsky"))
    assert_valid_result(res)
    assert not res.converged
    assert "line search failed" in res.message and "rounding level" in res.message
    assert np.max(np.abs(res.trace[-1].info["grad"])) < 1e-6
    loose = numopt.run("bfgs", problems.get("bohachevsky"), gtol=1e-6)
    assert loose.converged


def _quartic(n: int = 1) -> Problem:
    """f(x) = Σ x_i⁴: the minimizer 0 has ∇²f = 0, so the iterates approach it only linearly."""
    return Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: float(np.sum(x**4)),
        grad=lambda x: 4.0 * x**3,
        dim=n,
        domain=(),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("ls", ["strong_wolfe", "weak_wolfe", "backtracking", "goldstein"])
@pytest.mark.parametrize("x0", [[1.0], [1.0, -0.7]])
def test_gtol_zero_never_raises(method: str, ls: str, x0: list[float]) -> None:
    # Audit regression: with gtol = 0, ∇f reaches the underflow range, where ∇fᵀp, yᵀy and
    # (yᵀs)² round to 0. The runs raised ValueError (line search) or ZeroDivisionError (γ in
    # L-BFGS, μ in the Broyden class). The contract: never raise on numerical breakdown.
    prob = _quartic(len(x0))
    res = numopt.run(method, prob, x0=x0, gtol=0.0, max_iter=5000, line_search=ls)
    assert_valid_result(res, max_iter=5000)
    if res.converged:
        # Honest flag: gtol = 0 is met only by ∇f = 0 exactly.
        assert not np.any(res.trace[-1].info["grad"])
    else:
        # Either ∇f reaches the precision floor of f, or the slow linear approach to the
        # degenerate minimizer of Σx⁴ uses up max_iter.
        assert ("line search failed" in res.message and "rounding level" in res.message) or (
            "max_iter" in res.message
        )
    for s in res.trace:
        if s.info["H"] is not None:
            assert np.all(np.isfinite(s.info["H"]))


@pytest.mark.parametrize("method", METHODS)
def test_subnormal_gradient_stops_before_the_line_search(method: str) -> None:
    # x0 = 1e-107: ∇f = 4e-321 is subnormal; −∇f passes the (scaled) descent test, but
    # ∇fᵀp = −‖∇f‖² underflows to 0, and the line search needs ∇fᵀp < 0.
    res = numopt.run(method, _quartic(), x0=[1e-107], gtol=0.0)
    assert_valid_result(res)
    assert not res.converged and res.n_iter == 0
    assert "∇fᵀp underflowed to 0" in res.message and "rounding level" in res.message


@pytest.mark.parametrize("method", METHODS)
def test_invalid_input_raises(method: str) -> None:
    prob = problems.get("rosenbrock")
    with pytest.raises(ValueError):
        numopt.run(method, prob, line_search="exact_quadratic")
    with pytest.raises(ValueError):
        numopt.run(method, prob, gtol=float("nan"))
    with pytest.raises(ValueError):
        numopt.run(method, prob, max_iter=0)
    with pytest.raises(ValueError):
        numopt.run(method, prob, x0=[1.0])
    with pytest.raises(ValueError):
        numopt.minimize(lambda x: float("inf"), x0=[0.0, 0.0], method=method)


def test_invalid_method_parameters_raise() -> None:
    prob = problems.get("rosenbrock")
    for phi in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            numopt.run("broyden_class", prob, phi=phi)
    for m in (0, 2.5):
        with pytest.raises(ValueError):
            numopt.run("lbfgs", prob, m=m)


_COMMON_KEYS = {
    "grad",
    "direction",
    "alpha",
    "trials",
    "reset",
    "s",
    "y",
    "curvature",
    "rho",
    "update",
    "gamma",
    "H",
}
_EXTRA_KEYS = {
    "bfgs": set(),
    "dfp": set(),
    "sr1": {"denominator"},
    "broyden_class": {"phi_inverse"},
    "lbfgs": {"memory"},
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "quadratic_nd"])
def test_info_keys_and_trace_contract(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=500)
    assert res.n_iter == res.trace[-1].k
    for step in res.trace:
        assert set(step.info) == _COMMON_KEYS | _EXTRA_KEYS[method]
        assert (step.info["H"] is None) == (prob.dim > 2)
        assert step.grad_norm == pytest.approx(float(np.linalg.norm(prob.grad(step.x))))
        assert_allclose(step.info["grad"], prob.grad(step.x), rtol=1e-15, atol=0)
        assert step.fun == pytest.approx(prob.f(step.x), rel=1e-15, abs=1e-300)
    first = res.trace[0]
    assert first.info["direction"] is None and first.info["trials"] == []
    if prob.dim == 2:
        assert np.array_equal(first.info["H"], np.eye(2))
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        p = np.asarray(cur.info["direction"])
        assert np.array_equal(cur.x, np.asarray(prev.x) + cur.step_size * p)
        assert_allclose(cur.info["s"], np.asarray(cur.x) - np.asarray(prev.x), rtol=0, atol=0)
        assert cur.info["trials"][-1] == [cur.step_size, cur.fun]


def test_initial_scaling_eq_6_20_happens_once() -> None:
    res = numopt.run("bfgs", problems.get("rosenbrock"))
    scaled = [s for s in res.trace if s.info["gamma"] is not None]
    assert len(scaled) == 1 and scaled[0].k == 1
    s, y = np.asarray(scaled[0].info["s"]), np.asarray(scaled[0].info["y"])
    assert scaled[0].info["gamma"] == pytest.approx(float(y @ s) / float(y @ y), rel=1e-15)
    # SR1 is never scaled.
    assert all(s.info["gamma"] is None for s in numopt.run("sr1", problems.get("rosenbrock")).trace)


@pytest.mark.parametrize(("method", "pid", "params"), qn_mod.FIXTURE_CASES)
def test_fixture_cases_run(method: str, pid: str, params: dict[str, Any]) -> None:
    res = numopt.run(method, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 300


# --------------------------------------------------------------------------------------
# A non-finite slope ∇fᵀp is a failed run, not an exception (web-port regression)
# --------------------------------------------------------------------------------------


def _concave_bowl() -> Problem:
    """f = −(x² + y²): unbounded below, so the iterates grow until ∇fᵀp overflows to −inf."""
    return Problem(
        id="concave_bowl",
        name="concave bowl",
        latex="",
        f=lambda x: float(-(x[0] ** 2 + x[1] ** 2)),
        grad=lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: -2.0 * np.eye(2),
        dim=2,
        domain=((-2.0, 2.0), (-2.0, 2.0)),
        x0=np.array([1.0, 0.5]),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the user f overflows, not the method
@pytest.mark.parametrize("line_search", ["backtracking", "strong_wolfe"])
@pytest.mark.parametrize("method", METHODS)
def test_non_finite_slope_is_a_failed_run(method: str, line_search: str) -> None:
    """Regression: `if not float(g @ p) < 0.0` let ∇fᵀp = −inf through, and search() raised
    ValueError. The run must end with converged=False and a finite last iterate."""
    res = numopt.run(method, _concave_bowl(), line_search=line_search, max_iter=2000)
    assert_valid_result(res)
    assert not res.converged
    assert res.fun is not None and np.all(np.isfinite(res.x)) and math.isfinite(res.fun)
    if line_search == "backtracking":  # α = 1 is accepted until ‖x‖ ≈ 1e154
        assert "∇fᵀp overflowed to -inf, so no step can be tested" in res.message, res.message
        assert res.message.startswith(f"line search failed at iteration {res.n_iter + 1}")
        grad_norm = res.trace[-1].grad_norm
        assert grad_norm is not None and grad_norm > 1e153
    else:  # strong Wolfe: φ still decreases at α_max on the first search
        assert "phi still decreases at alpha_max" in res.message, res.message


def test_slope_helpers() -> None:
    big = np.array([1e200, 1e200])
    assert qn_mod._slope(big, -big) == -math.inf  # no RuntimeWarning (errstate)
    assert qn_mod._slope_failure(-math.inf) == "∇fᵀp overflowed to -inf, so no step can be tested"
    assert qn_mod._slope_failure(math.nan) == "∇fᵀp overflowed to nan, so no step can be tested"
    assert qn_mod._slope_failure(0.0) == "∇fᵀp underflowed to 0, so no step can be tested"
    assert qn_mod._slope(np.array([1.0, 2.0]), np.array([-3.0, 0.5])) == -2.0
