"""Tests for numopt.unconstrained.accelerated: silver steps, long steps, OGM and restarted FISTA.

Oracles (none of them calls the module's iteration code):
* exact identities from the papers: Lemma 2.3 (sum of silver steps), the footnote-2 one-liner
  and eq. 1.3 of Part II, eq. 1.4, eq. 3.6 of Part I, Table 1 of Grimmer, eq. 6.17 of OGM1;
* a 600-digit Decimal implementation of Part I's recursion (eqs. 3.1–3.2, 3.9), written from
  the paper without the cancellation-free rewrite of the module;
* the closed form x_n − x* = Π_t (I − h_t A/L)(x₀ − x*) on quadratics (eigenbasis);
* Kim & Fessler's Theorem 3: OGM1 attains its bound exactly on the Huber function (eq. 8.1);
* hand computations of Beck & Teboulle (4.1)–(4.3), and O'Donoghue & Candès' θ-form
  (Algorithm 1, q = 0) with the §3.2 restart tests and the reset of Algorithm 3;
* Hypothesis property tests (1000 examples each): every certificate on random L-smooth convex
  functions with a known minimizer, FISTA's Theorem 4.4, the restart tests' definitions and
  the backtracking bounds of Remark 3.2;
* the restarted-AGD study's key property: iterations grow like √κ without knowing μ, within
  2× of ``nesterov`` tuned with the true κ.

Random test data come from ``numopt.core.rng.Rng`` only.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import math
from decimal import Decimal, getcontext
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.rng import Rng
from numopt.core.types import Problem
from numopt.unconstrained import accelerated as M

RHO = 1.0 + math.sqrt(2.0)
EPS = float(np.finfo(np.float64).eps)
PROPERTY = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])

SCHEDULES = ("silver_gd", "silver_gd_strongly_convex", "long_step_gd", "ogm")
METHODS = (*SCHEDULES, "fista")
RESTART_MODES = ("none", "gradient", "function")


# --------------------------------------------------------------------------------------
# Test data (Rng only)
# --------------------------------------------------------------------------------------


def _normals(rng: Rng, *shape: int) -> np.ndarray:
    return np.array([rng.normal() for _ in range(math.prod(shape))]).reshape(shape)


def _orthogonal(rng: Rng, n: int) -> np.ndarray:
    Q, R = np.linalg.qr(_normals(rng, n, n))
    return Q * np.sign(np.diag(R))


def _quadratic(A: np.ndarray, c: np.ndarray, x0: np.ndarray, *, tags: tuple[str, ...] = ()):
    """f(x) = ½(x − c)ᵀA(x − c), f* = 0 at x* = c."""
    return Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: 0.5 * float((x - c) @ (A @ (x - c))),
        grad=lambda x: A @ (x - c),
        hess=lambda x: A,
        dim=c.size,
        domain=(),
        x0=x0.tolist(),
        tags=tags,
    )


def _fun(step: Any) -> float:
    """Step.fun as a float (the methods here always set it)."""
    assert step.fun is not None
    return float(step.fun)


def _log_spectrum_quadratic(kappa: float, n: int, seed: int) -> tuple[Problem, np.ndarray]:
    """λ_i = κ^{i/(n−1)} (λ_min = 1, λ_max = κ exactly), Haar-like eigenvectors."""
    rng = Rng(seed)
    Q = _orthogonal(rng, n)
    lam = kappa ** (np.arange(n) / (n - 1))
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    c = _normals(rng, n)
    x0 = _normals(rng, n)
    return _quadratic(A, c, x0), c


# --------------------------------------------------------------------------------------
# Schedules: identities from the papers
# --------------------------------------------------------------------------------------


def test_two_adic_valuation() -> None:
    assert [M.two_adic_valuation(t) for t in range(1, 13)] == [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2]
    with pytest.raises(ValueError):
        M.two_adic_valuation(0)


def test_silver_schedule_matches_eq_2_1_footnote_2_and_eq_1_3() -> None:
    s2 = math.sqrt(2.0)
    expected = [s2, 2.0, s2, 1 + RHO, s2, 2.0, s2, 1 + RHO**2]
    assert_allclose(M.silver_schedule(8), expected, rtol=1e-15)
    # Part II, footnote 2: [1+rho**((k & -k).bit_length()-2) for k in range(1,64)]
    one_liner = [1 + RHO ** ((k & -k).bit_length() - 2) for k in range(1, 64)]
    assert_array_equal(M.silver_schedule(63), one_liner)
    # eq. 1.3: h_{2n+1} = [h_n, 1 + ρ^{k−1}, h_n] for n = 2^k − 1, h_1 = [√2]
    h = [s2]
    for k in range(1, 10):
        assert_allclose(M.silver_schedule(len(h)), h, rtol=1e-15)
        h = [*h, 1 + RHO ** (k - 1), *h]
    with pytest.raises(ValueError):
        M.silver_schedule(-1)


@pytest.mark.parametrize("k", range(1, 16))
def test_lemma_2_3_sum_of_silver_steps(k: int) -> None:
    # Part II, Lemma 2.3: Σ_{t < 2^k − 1} h_t = ρ^k − 1
    assert_allclose(M.silver_schedule(2**k - 1).sum(), RHO**k - 1, rtol=1e-13)


def test_silver_rate_eq_1_4() -> None:
    assert M.silver_rate(0) == 0.5  # f(x₀) − f* ≤ L R²/2 (smoothness)
    for k in range(1, 40):
        n = 2**k - 1
        r = M.silver_rate(k)
        assert r == pytest.approx(1 / (1 + math.sqrt(4 * RHO ** (2 * k) - 3)), rel=1e-15)
        assert r <= 1 / (2 * RHO ** math.log2(n)) * (1 + 1e-15)  # eq. 1.4
        if n >= 15:  # between OGM's 1/(n+1)² and the tight constant-step 1/(4n+2)
            assert 1 / (n + 1) ** 2 < r < 1 / (4 * n + 2)
    with pytest.raises(ValueError):
        M.silver_rate(-1)


def _sc_decimal(kappa: float, j: int) -> tuple[list[Decimal], list[Decimal], Decimal]:
    """Part I eqs. 3.1–3.2, 3.4, 3.9 in 600-digit arithmetic, straight from the paper.

    1 − z_n is formed by subtraction here; 600 digits keep it exact for the tested levels.
    """
    getcontext().prec = 600
    k = Decimal(kappa)
    z = Decimal(1) / k

    def psi(t: Decimal) -> Decimal:
        return (1 + k * t) / (1 + t)

    a_list, b_list = [psi(z)], [psi(z)]
    for _ in range(j):
        xi = 1 - z
        root = (1 + xi * xi).sqrt()
        y, z = z / (xi + root), z * (xi + root)
        # eqs. 3.1–3.2: y z = z_{n/2}², z − y = 2(z_{n/2} − z_{n/2}²)
        a_list.append(psi(y))
        b_list.append(psi(z))
    return a_list, b_list, ((1 - z) / (1 + z)) ** 2


@pytest.mark.parametrize("kappa", [1.5, 4.0, 16.0, 100.0, 1e4, 1e8, 1e15, 1e16, 1e18])
def test_kappa_schedule_against_600_digit_recursion(kappa: float) -> None:
    levels = M._silver_sc_levels(kappa)
    for j in range(0, 9):
        n = 2**j
        h, tau = M.silver_sc_schedule(kappa, n)
        a, b, tau_ref = _sc_decimal(kappa, j)
        # log τ_n is what the certificate uses (bound_dist = τ_n^m); for κ ≳ 1e15 it is of size
        # 1/κ while τ_n rounds to 1, so only log τ_n shows a lost −log(1 + z) term.
        # NOTE: 16ε covers one rounding per level of the recursion (measured: ≤ 2.5ε).
        log_tau = next(levels)[2]
        assert log_tau == pytest.approx(float(tau_ref.ln()), rel=16 * EPS)
        assert h[-1] == pytest.approx(float(b[j]), rel=1e-13)
        for i in range(1, n):  # entry i (1-based, i < n) is a at level lsb(i) + 1 (eq. 3.8)
            lsb = (i & -i).bit_length() - 1
            assert h[i - 1] == pytest.approx(float(a[lsb + 1]), rel=1e-13)
        # NOTE: τ_n depends on (1 − 1/κ)^{2^j}, so the input rounding is amplified 2^j-fold;
        # 2^j·64ε is that amplification, not a loosened tolerance.
        assert tau == pytest.approx(float(tau_ref), rel=2**j * 64 * EPS, abs=1e-300)
        assert M.silver_sc_rate(kappa, n) == tau


@pytest.mark.parametrize("kappa", [2.0, 10.0, 100.0, 1e3, 1e6])
def test_kappa_schedule_bounds_and_eq_3_6(kappa: float) -> None:
    hm, am = 2 * kappa / (kappa + 1), (kappa + 1) / 2
    h1, tau1 = M.silver_sc_schedule(kappa, 1)
    assert h1[0] == pytest.approx(hm, rel=1e-15)  # eq. 3.6: a₁ = b₁ = ψ(1/κ) = HM(1, κ)
    assert tau1 == pytest.approx(((kappa - 1) / (kappa + 1)) ** 2, rel=1e-13)
    prev = tau1
    for j in range(1, 14):
        h, tau = M.silver_sc_schedule(kappa, 2**j)
        assert np.all(h > 1.0) and np.all(h <= am * (1 + 8 * EPS))  # 8ε: rounding of ψ
        assert h[-1] >= hm * (1 - 1e-15)
        assert tau <= prev**2 * (1 + 1e-12) + 1e-300  # τ_{2n} ≤ τ_n²
        prev = tau


def test_kappa_schedule_limits() -> None:
    # Part II, Remark 2.2: as μ → 0 the κ-aware schedule tends to the convex one.
    for j in range(1, 8):
        h, _ = M.silver_sc_schedule(1e12, 2**j)
        assert_allclose(h[:-1], M.silver_schedule(2**j - 1), rtol=1e-9)
    h, tau = M.silver_sc_schedule(1.0, 4)  # κ = 1: one step of 1/L is exact
    assert_allclose(h, 1.0)
    assert tau == 0.0
    assert M.silver_sc_auto_horizon(1.0) == 1


def test_auto_horizon_is_at_saturation() -> None:
    for kappa in (4.0, 50.0, 100.0, 1e4):
        n = M.silver_sc_auto_horizon(kappa)

        def rate(m: int, kappa: float = kappa) -> float:
            return -math.log(M.silver_sc_rate(kappa, m)) / m

        assert rate(2 * n) < 1.01 * rate(n)
        if n > 1:
            assert rate(n) >= 1.01 * rate(n // 2)
    assert M.silver_sc_auto_horizon(100.0) == 64
    assert M.silver_sc_auto_horizon(50.0) == 32


def test_auto_horizon_does_not_decrease_with_kappa() -> None:
    # A longer horizon saturates later (Part I, Theorem 4.1: n* ≍ κ^{log_ρ 2}). Before the
    # cancellation-free log τ recursion the rule fell from 2²⁰ to 1, 2 or 4 for κ ≥ 1e16.
    kappas = sorted({1.0, 1.68e16, *(10.0 ** (e / 100) for e in range(0, 3001))})
    horizons = [M.silver_sc_auto_horizon(k) for k in kappas]
    assert all(m <= n for m, n in itertools.pairwise(horizons))
    assert horizons[-1] == M.MAX_HORIZON
    for kappa in (1e8, 1e15, 1e16, 1.68e16, 1e18, 1e30):
        assert M.silver_sc_auto_horizon(kappa) == M.MAX_HORIZON


@PROPERTY
@given(e1=st.floats(0.0, 30.0), e2=st.floats(0.0, 30.0), j=st.integers(0, 20))
def test_property_kappa_rate_and_horizon(e1: float, e2: float, j: int) -> None:
    lo, hi = sorted((10.0**e1, 10.0**e2))
    assert M.silver_sc_auto_horizon(lo) <= M.silver_sc_auto_horizon(hi)
    # log τ_n against the 600-digit recursion of the paper, for κ up to 1e30
    *_, (_, _, log_tau) = itertools.islice(M._silver_sc_levels(hi), j + 1)
    _, _, tau_ref = _sc_decimal(hi, j)
    if hi == 1.0:  # one step of 1/L is exact
        assert log_tau == -math.inf
        return
    if tau_ref == 0 or tau_ref.ln() < -1000:  # 1 − z_n is below 600 digits there
        return
    assert log_tau == pytest.approx(float(tau_ref.ln()), rel=32 * EPS)  # ≤ 21 levels of ε


def test_long_step_patterns_match_grimmer_table_1() -> None:
    for key, h in M.LONG_STEP_PATTERNS.items():
        assert len(h) == int(key)
        # Table 1: the proved coefficient is avg(h), rounded down in the 7th digit for t ≥ 7.
        assert 0.0 <= float(np.mean(h)) - M.LONG_STEP_RATES[key] < 2e-5
    assert M.LONG_STEP_PATTERNS["2"] == (2.9, 1.5)
    assert M.LONG_STEP_PATTERNS["7"] == (1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5)
    assert max(M.LONG_STEP_PATTERNS["127"]) == 370.0
    for key in ("3", "7", "15", "31", "63", "127"):  # palindromes, longest step in the middle
        h = M.LONG_STEP_PATTERNS[key]
        assert h == h[::-1] and h[len(h) // 2] == max(h)


def test_ogm_thetas_and_eq_6_17() -> None:
    th = M.ogm_thetas(5)
    assert th[0] == 1.0
    for i in range(4):
        assert th[i + 1] == pytest.approx((1 + math.sqrt(1 + 4 * th[i] ** 2)) / 2, rel=1e-15)
    assert th[5] == pytest.approx((1 + math.sqrt(1 + 8 * th[4] ** 2)) / 2, rel=1e-15)
    for N in range(1, 200):
        b = M.ogm_rate(N)
        assert b <= 1 / ((N + 1) * (N + 1 + math.sqrt(2))) * (1 + 1e-14)  # eq. 6.17
    with pytest.raises(ValueError):
        M.ogm_thetas(0)


# --------------------------------------------------------------------------------------
# Iterates: hand computations and independent formulations
# --------------------------------------------------------------------------------------


def test_hand_computed_first_steps() -> None:
    p = problems.get("quadratic_bowl")  # Hessian eigenvalues 2 and 4 → L = 4 (auto)
    x0 = np.array(p.x0, dtype=float)
    g0 = np.asarray(p.grad(x0))
    r = numopt.run("silver_gd", p, max_iter=2, gtol=0.0)
    assert r.extra == {"L": pytest.approx(4.0, rel=1e-14), "L_source": r.extra["L_source"]}
    assert_allclose(r.trace[1].x, x0 - math.sqrt(2) / 4 * g0, rtol=1e-15)
    assert r.trace[1].step_size == pytest.approx(math.sqrt(2) / 4, rel=1e-15)
    assert r.trace[2].info["h"] == 2.0
    r = numopt.run("long_step_gd", p, pattern="2", max_iter=1, gtol=0.0)
    assert_allclose(r.trace[1].x, x0 - 2.9 / 4 * g0, rtol=1e-15)
    # OGM1, θ₀ = 1: x₁ = y₁ + (1/θ₁)(y₁ − x₀) = x₀ − (1 + 1/θ₁)∇f(x₀)/L, θ₁ = golden ratio
    r = numopt.run("ogm", p, max_iter=3, gtol=0.0)
    th1 = (1 + math.sqrt(5)) / 2
    assert_allclose(r.trace[1].x, x0 - (1 + 1 / th1) * g0 / 4, rtol=1e-14)
    assert_allclose(r.trace[1].info["y"], x0 - g0 / 4, rtol=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_iterates_equal_the_matrix_polynomial_on_quadratics(seed: int) -> None:
    # Independent formulation: x_n − x* = Π_t (I − h_t A/L)(x₀ − x*) in the eigenbasis of A.
    rng = Rng(seed)
    n = 6
    Q = _orthogonal(rng, n)
    lam = np.geomspace(0.01, 1.0, n) * 7.0
    A = 0.5 * ((Q * lam) @ Q.T + ((Q * lam) @ Q.T).T)
    c, x0 = _normals(rng, n), _normals(rng, n)
    L = 7.0
    P = _quadratic(A, c, x0)
    cases = [
        (numopt.run("silver_gd", P, L=L, max_iter=31, gtol=0.0), M.silver_schedule(31)),
        (
            numopt.run("long_step_gd", P, L=L, pattern="7", max_iter=21, gtol=0.0),
            np.tile(M.LONG_STEP_PATTERNS["7"], 3),
        ),
        (
            numopt.run(
                "silver_gd_strongly_convex", P, L=L, mu=0.07, horizon=8, max_iter=24, gtol=0.0
            ),
            np.tile(M.silver_sc_schedule(100.0, 8)[0], 3),
        ),
    ]
    for res, h in cases:
        coeff = np.prod(1.0 - np.outer(h, lam) / L, axis=0)
        x_ref = c + Q @ (coeff * (Q.T @ (x0 - c)))
        # NOTE: rtol 1e-9, not 1e-12: the products contain factors |1 − hλ/L| up to ~30 (long
        # steps) whose partial products amplify rounding by ~10³ before they shrink.
        assert_allclose(res.x, x_ref, rtol=1e-9, atol=1e-12 * float(np.linalg.norm(x0 - c)))


@pytest.mark.parametrize("N", [1, 2, 3, 7, 20, 100])
@pytest.mark.parametrize("R", [0.1, 1.0, 37.0])
def test_ogm_attains_its_bound_on_the_kim_fessler_huber_function(N: int, R: float) -> None:
    # Kim & Fessler (2016), Theorem 3, eq. 8.1: f(x_N) − f* = L R²/(2θ_N²) exactly.
    L = 3.0
    th = float(M.ogm_thetas(N)[-1])
    thr = R / th**2

    def f(x: np.ndarray) -> float:
        r = float(np.linalg.norm(x))
        return L * R / th**2 * r - L * R**2 / (2 * th**4) if r >= thr else 0.5 * L * r * r

    def g(x: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(x))
        return L * R / th**2 * x / r if r >= thr else L * x

    p = Problem(id="hub", name="hub", latex="", f=f, grad=g, dim=3, domain=(), x0=[0, 0, 0])
    nu = np.array([2.0, -1.0, 2.0]) / 3.0
    res = numopt.run("ogm", p, x0=R * nu, L=L, max_iter=N, gtol=0.0)
    assert res.trace[-1].info["bound_f"] == pytest.approx(M.ogm_rate(N), rel=1e-15)
    assert res.fun == pytest.approx(L * R**2 / (2 * th**2), rel=1e-10)


def test_fista_first_three_steps_match_beck_teboulle() -> None:
    P, _ = _log_spectrum_quadratic(50.0, 5, seed=3)
    L = 50.0
    res = numopt.run("fista", P, restart="none", lr=1 / L, backtracking=False, max_iter=3)
    x0 = np.asarray(P.x0)
    grad = P.grad
    assert grad is not None

    def step(y: np.ndarray) -> np.ndarray:
        return y - grad(y) / L

    x1 = step(x0)  # y₁ = x₀
    t2 = (1 + math.sqrt(5)) / 2  # t₁ = 1, so β = (t₁ − 1)/t₂ = 0
    x2 = step(x1)
    t3 = (1 + math.sqrt(1 + 4 * t2**2)) / 2
    y3 = x2 + ((t2 - 1) / t3) * (x2 - x1)
    x3 = step(y3)
    for k, xk in enumerate((x0, x1, x2, x3)):
        assert_allclose(res.trace[k].x, xk, rtol=1e-13, atol=1e-15)
    assert res.trace[2].info["beta"] == 0.0
    assert res.trace[3].info["beta"] == pytest.approx((t2 - 1) / t3, rel=1e-15)
    assert_allclose(res.trace[3].info["y"], y3, rtol=1e-13, atol=1e-15)
    assert_allclose(res.trace[3].info["grad_y"], grad(y3), rtol=1e-13, atol=1e-13)


def _odc_algorithm1(P: Problem, L: float, iters: int, restart: str) -> tuple[list, list]:
    """O'Donoghue & Candès (2015), Algorithm 1 with q = 0 and step 1/L, θ-form, with the §3.2
    restart tests; a restart applies Algorithm 3's reset (x⁰ ← x^k, y⁰ ← x^k, θ₀ ← 1)."""
    assert P.grad is not None
    x_prev = np.asarray(P.x0, dtype=float)
    y = x_prev.copy()
    theta = 1.0
    xs, restarts = [x_prev.copy()], []
    f_prev = P.f(x_prev)
    for k in range(1, iters + 1):
        gy = P.grad(y)
        x = y - gy / L
        if restart == "gradient":
            fired = float(gy @ (x - x_prev)) > 0.0
        elif restart == "function":
            fired = P.f(x) > f_prev
        else:
            fired = False
        if fired:
            restarts.append(k)
            theta, y = 1.0, x.copy()
        else:
            # θ_{k+1}² = (1 − θ_{k+1})θ_k² (q = 0): the positive root; β = θ(1 − θ)/(θ² + θ⁺).
            theta_next = 0.5 * (-(theta**2) + math.sqrt(theta**4 + 4 * theta**2))
            beta = theta * (1 - theta) / (theta**2 + theta_next)
            y = x + beta * (x - x_prev)
            theta = theta_next
        f_prev = P.f(x)
        x_prev = x
        xs.append(x.copy())
    return xs, restarts


@pytest.mark.parametrize("restart", ["none", "gradient", "function"])
def test_fista_matches_the_odonoghue_candes_theta_form(restart: str) -> None:
    P, _ = _log_spectrum_quadratic(1e3, 12, seed=1)
    L = 1e3
    xs_ref, r_ref = _odc_algorithm1(P, L, 150, restart)
    res = numopt.run(
        "fista", P, restart=restart, lr=1 / L, backtracking=False, gtol=1e-14, max_iter=150
    )
    assert res.n_iter == 150
    assert res.extra["restarts"] == r_ref
    assert res.extra["n_restart"] == len(r_ref)
    assert [s.k for s in res.trace if s.info["restarted"]] == r_ref
    if restart != "none":
        assert len(r_ref) >= 1
    # NOTE: the two forms compute β with different roundings (t-form vs θ-form); the gap stays
    # at the rounding level of the iterates (‖x‖ ~ 5).
    for k in range(0, 151, 10):
        assert_allclose(res.trace[k].x, xs_ref[k], rtol=1e-9, atol=1e-11)


def test_fista_exact_step_on_an_isotropic_quadratic() -> None:
    c = np.array([1.0, -2.0, 0.5])
    P = _quadratic(4.0 * np.eye(3), c, np.zeros(3))
    res = numopt.run("fista", P, lr=0.25, backtracking=False)
    # x₁ = c exactly; ∇f(y₂) = ∇f(x₁) = 0 then stops at k = 2.
    assert res.converged and res.n_iter == 2
    assert_array_equal(res.x, c)


# --------------------------------------------------------------------------------------
# Certificates and guarantees on random instances (Hypothesis)
# --------------------------------------------------------------------------------------


def _huber(r: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.where(np.abs(r) <= d, 0.5 * r * r, d * np.abs(r) - 0.5 * d * d)


@st.composite
def convex_instances(draw: st.DrawFn, strongly: bool = False) -> dict[str, Any]:
    """f(x) = Σ_j c_j hub_{δ_j}(a_jᵀ(x − x*)) + ½(x − x*)ᵀD(x − x*), f* = 0 at x*.

    hub'' ≤ 1, so L = Σ c_j‖a_j‖² + λ_max(D) is a valid global constant (μ = λ_min(D)).
    """
    rng = Rng(draw(st.integers(0, 2**32 - 1)))
    d = draw(st.integers(1, 4))
    m = draw(st.integers(0, 4))
    scale = 10.0 ** draw(st.integers(-3, 3))  # x scale
    fscale = 10.0 ** draw(st.integers(-3, 3))  # f scale
    a = _normals(rng, m, d)
    c = fscale * np.array([rng.uniform(0.1, 1.0) for _ in range(m)])
    delta = scale * np.array([10.0 ** rng.uniform(-3, 1) for _ in range(m)])
    rank = d if strongly else draw(st.integers(0, d))
    B = _normals(rng, d, rank)
    D = fscale * (B @ B.T + (1e-2 * np.eye(d) if strongly else 0.0))
    xs = scale * _normals(rng, d)
    x0 = xs + scale * _normals(rng, d) * 10.0 ** rng.uniform(-1, 1)
    ev = np.linalg.eigvalsh(D)
    L = float(np.sum(c * np.sum(a * a, axis=1)) + ev[-1])
    if L <= 0.0:
        L = fscale  # f ≡ 0 on this draw: any L is valid
    slack = draw(st.sampled_from([1.0, 1.0, 1.3, 3.0]))

    def f(x: np.ndarray) -> float:
        dx = x - xs
        return float(np.sum(c * _huber(a @ dx, delta)) + 0.5 * dx @ D @ dx)

    def g(x: np.ndarray) -> np.ndarray:
        dx = x - xs
        return a.T @ (c * np.clip(a @ dx, -delta, delta)) + D @ dx

    p = Problem(id="rand", name="rand", latex="", f=f, grad=g, dim=d, domain=(), x0=x0.tolist())
    return {
        "p": p,
        "x0": x0,
        "xs": xs,
        "L": L * slack,
        "mu": float(ev[0]),
        "R2": float(np.sum((x0 - xs) ** 2)),
    }


@PROPERTY
@given(convex_instances())
def test_property_silver_envelope(inst: dict[str, Any]) -> None:
    res = numopt.run("silver_gd", inst["p"], x0=inst["x0"], L=inst["L"], max_iter=63, gtol=0.0)
    LR2 = inst["L"] * inst["R2"]
    for k in range(1, 7):
        n = 2**k - 1
        if n > res.n_iter:  # stopped at an exact stationary point (f = f*)
            assert res.converged and res.fun <= 1e-12 * LR2
            break
        step = res.trace[n]
        assert step.info["checkpoint"] and step.info["bound_f"] == M.silver_rate(k)
        # NOTE: absolute slack 1e-12·L R² covers the rounding of f near f* = 0.
        assert step.fun <= M.silver_rate(k) * LR2 * (1 + 1e-12) + 1e-12 * LR2


@PROPERTY
@given(convex_instances(), st.integers(1, 40))
def test_property_ogm_bound(inst: dict[str, Any], N: int) -> None:
    res = numopt.run("ogm", inst["p"], x0=inst["x0"], L=inst["L"], max_iter=N, gtol=0.0)
    LR2 = inst["L"] * inst["R2"]
    if res.n_iter == N:
        assert res.trace[-1].info["bound_f"] == M.ogm_rate(N)
        assert res.fun <= M.ogm_rate(N) * LR2 * (1 + 1e-12) + 1e-12 * LR2
    else:  # only an exact stationary point stops a gtol = 0 run early
        assert res.converged and res.trace[-1].grad_norm == 0.0


@PROPERTY
@given(convex_instances(strongly=True), st.sampled_from([1, 2, 4, 8, 16, 0]))
def test_property_kappa_aware_distance_bound(inst: dict[str, Any], horizon: int) -> None:
    res = numopt.run(
        "silver_gd_strongly_convex",
        inst["p"],
        x0=inst["x0"],
        L=inst["L"],
        mu=inst["mu"],
        horizon=horizon,
        max_iter=64,
        gtol=0.0,
    )
    n, tau = res.extra["horizon"], res.extra["tau_horizon"]
    # NOTE: each coordinate of an iterate carries a few ulps of max(|x₀|, |x*|), so distances
    # are compared with an additive δ.
    scale = max(1e-300, float(np.max(np.abs(np.concatenate([inst["x0"], inst["xs"]])))))
    delta = 64 * EPS * math.sqrt(inst["p"].dim) * scale
    for s in res.trace[1:]:
        if s.info["checkpoint"]:
            assert s.info["bound_dist"] == pytest.approx(tau ** (s.k // n), rel=1e-12)
            d = float(np.linalg.norm(np.asarray(s.x) - inst["xs"]))
            assert d <= math.sqrt(s.info["bound_dist"] * inst["R2"]) * (1 + 1e-9) + delta


quad_inputs = st.tuples(st.integers(2, 6), st.floats(1.0, 1e4), st.integers(0, 2**32 - 1))


@PROPERTY
@given(quad_inputs)
def test_property_fista_theorem_4_4(args: tuple[int, float, int]) -> None:
    n, kappa, seed = args
    P, xs = _log_spectrum_quadratic(kappa, n, seed)
    x0 = np.asarray(P.x0)
    res = numopt.run(
        "fista", P, restart="none", lr=1 / kappa, backtracking=False, gtol=1e-14, max_iter=60
    )
    R2 = float((x0 - xs) @ (x0 - xs))
    for s in res.trace[1:]:
        assert _fun(s) <= 2 * kappa * R2 / (s.k + 1) ** 2 * (1 + 1e-10) + 1e-12  # f* = 0


@PROPERTY
@given(quad_inputs, st.sampled_from(RESTART_MODES))
def test_property_restart_tests_and_backtracking(
    args: tuple[int, float, int], restart: str
) -> None:
    n, kappa, seed = args
    P, xs = _log_spectrum_quadratic(kappa, n, seed)
    assert P.grad is not None
    res = numopt.run(
        "fista", P, restart=restart, lr=1 / kappa, backtracking=False, gtol=1e-14, max_iter=40
    )
    tr = res.trace
    for k in range(1, len(tr)):
        x_k, x_km1 = np.asarray(tr[k].x), np.asarray(tr[k - 1].x)
        y_k, g_k = np.asarray(tr[k].info["y"]), np.asarray(tr[k].info["grad_y"])
        assert_array_equal(g_k, P.grad(y_k))
        if restart == "function":
            expect = _fun(tr[k]) > _fun(tr[k - 1])
        elif restart == "gradient":
            expect = float(g_k @ (x_k - x_km1)) > 0.0
        else:
            expect = False
        assert tr[k].info["restarted"] == expect
        if expect:
            assert tr[k].info["t"] == 1.0
            if k + 1 < len(tr):  # the next step is momentum-free and starts from x_k
                assert tr[k + 1].info["beta"] == 0.0
                assert_array_equal(tr[k + 1].info["y"], x_k)
    # Backtracking (Beck & Teboulle, Remark 3.2): L_k is non-decreasing, L_k ≤ max(L₀, ηL).
    L0 = kappa / 37.0
    res = numopt.run(
        "fista", P, restart=restart, lr=1 / L0, backtracking=True, eta=2.0, gtol=1e-14, max_iter=40
    )
    Ls = [s.info["L"] for s in res.trace]
    assert all(b >= a for a, b in itertools.pairwise(Ls))
    assert max(Ls) <= max(L0, 2.0 * kappa) * (1 + 1e-12)
    for s in res.trace[1:]:  # every accepted step satisfies the sufficient decrease (2.9)
        y = np.asarray(s.info["y"])
        gy = np.asarray(s.info["grad_y"])
        assert s.fun <= P.f(y) - float(gy @ gy) / (2 * s.info["L"]) + 1e-12 * max(1.0, abs(P.f(y)))
    if restart == "none":  # Theorem 4.4 with the backtracking constant α = η
        x0 = np.asarray(P.x0)
        R2 = float((x0 - xs) @ (x0 - xs))
        for s in res.trace[1:]:
            assert _fun(s) <= 2 * 2.0 * kappa * R2 / (s.k + 1) ** 2 * (1 + 1e-10) + 1e-12


# --------------------------------------------------------------------------------------
# The studies' key properties
# --------------------------------------------------------------------------------------


def _first_below(res: Any, target: float) -> int:
    return next(s.k for s in res.trace if s.fun < target)


@pytest.mark.parametrize("seed", range(3))
def test_restarted_agd_scales_like_sqrt_kappa_without_mu(seed: int) -> None:
    # Study: iterations to f − f* < 1e-8 grow like κ^0.549 for gradient restart, κ^0.997 for GD,
    # and stay within 2× of `nesterov` tuned with the true κ (max 1.51× there).
    iters: dict[str, list[int]] = {"gr": [], "fr": [], "none": [], "nest": []}
    for kappa in (1e2, 1e4):
        P, _ = _log_spectrum_quadratic(kappa, 30, seed)
        # gtol = 1e-4 < √(2μ·1e-8) (μ = 1) guarantees f − f* < 1e-8 when a run stops
        # (f(x) − f* ≤ ‖∇f‖²/(2μ); for fista f(x_k) ≤ f(y_k)), so every run reaches the target.
        kw = {"lr": 1 / kappa, "backtracking": False, "gtol": 1e-4, "max_iter": 100_000}
        for key, restart in (("gr", "gradient"), ("fr", "function"), ("none", "none")):
            res = numopt.run("fista", P, restart=restart, **kw)
            assert res.converged
            iters[key].append(_first_below(res, 1e-8))
        beta = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
        res = numopt.run("nesterov", P, lr=1 / kappa, beta=beta, gtol=1e-4, max_iter=100_000)
        assert res.converged
        iters["nest"].append(_first_below(res, 1e-8))
    slope = {k: math.log10(v[1] / v[0]) / 2 for k, v in iters.items()}
    assert 0.4 <= slope["gr"] <= 0.62 and 0.4 <= slope["fr"] <= 0.62, slope
    # Without restart the momentum overshoots: at κ = 1e4 plain FISTA needs ≥ 5× the
    # iterations of gradient restart (study: 12×; here 8–11×).
    assert iters["none"][1] >= 5 * iters["gr"][1], iters
    for a, b in zip(iters["gr"], iters["nest"], strict=True):
        assert a <= 2.0 * b
    # GD with step 1/L: κ¹ scaling, so at κ = 1e4 it needs far more than 10× restarted AGD.
    P, _ = _log_spectrum_quadratic(1e4, 30, seed)
    budget = 10 * iters["gr"][1]
    gd = numopt.run("gradient_descent", P, lr=1e-4, step_rule="fixed", gtol=1e-4, max_iter=budget)
    assert gd.fun is not None and gd.fun > 1e-8


@pytest.mark.parametrize("seed", [0, 3])
def test_silver_beats_constant_steps_on_a_merely_convex_problem(seed: int) -> None:
    # Study: on decay_quadratic (λ_j = 1/j², L = 1, n = 200) convex silver is ahead of GD 1/L
    # at every checkpoint from n = 7 on, by 45× at n = 4095 (here 30–61× over 5 starts).
    n = 200
    lam = 1.0 / np.arange(1, n + 1) ** 2
    x0 = _normals(Rng(seed), n)
    P = _quadratic(np.diag(lam), np.zeros(n), x0, tags=("quadratic",))
    sil = numopt.run("silver_gd", P, max_iter=4095, gtol=0.0)
    gd = numopt.run("gradient_descent", P, lr=1.0, step_rule="fixed", gtol=1e-150, max_iter=4095)
    assert sil.extra["L"] == pytest.approx(1.0, rel=1e-14)
    ratios = [gd.trace[2**k - 1].fun / sil.trace[2**k - 1].fun for k in range(3, 13)]
    assert min(ratios) > 1.0, ratios  # ahead at every checkpoint n = 7, ..., 4095
    assert all(b >= a for a, b in itertools.pairwise(ratios)), ratios  # and the lead grows
    assert ratios[-1] >= 20.0, ratios


def test_convex_silver_rate_is_polynomial_on_a_strongly_convex_problem() -> None:
    # f(x_{2^k−1}) falls by ≈ ρ² per doubling of n on quadratic_bowl (κ = 2).
    r = numopt.run("silver_gd", problems.get("quadratic_bowl"), max_iter=2047, gtol=0.0)
    ratios = [r.trace[2**k - 1].fun / r.trace[2 ** (k + 1) - 1].fun for k in range(6, 10)]
    assert_allclose(ratios, RHO**2, rtol=0.02)


# --------------------------------------------------------------------------------------
# Contract, registry, counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mid", METHODS)
def test_registry_metadata(mid: str) -> None:
    spec = get_method(mid)
    assert spec.family == "unconstrained"
    assert spec.references and spec.summary and spec.order
    assert any("research/" in ref for ref in spec.references)
    sig = inspect.signature(spec.fn).parameters
    assert set(sig) - {"problem", "x0"} == {p.name for p in spec.params}
    for prm in spec.params:
        assert sig[prm.name].default == prm.default
        if prm.kind in ("float", "int"):
            assert prm.min is not None and prm.max is not None
            assert prm.min <= prm.default <= prm.max
            if prm.log:
                assert prm.min > 0.0
        if prm.kind == "choice":
            assert prm.default in prm.choices
    assert spec.defaults()["max_iter"] <= 5000


def _corner_values(mid: str) -> list[dict[str, Any]]:
    out = []
    for prm in get_method(mid).params:
        if prm.name in ("L", "mu", "horizon", "max_iter"):
            continue  # L/μ/horizon corners are checked separately; max_iter by fixture size
        if prm.kind == "choice":
            out += [{prm.name: c} for c in prm.choices]
        elif prm.kind == "bool":
            out += [{prm.name: v} for v in (False, True)]
        else:
            out += [{prm.name: prm.min}, {prm.name: prm.max}]
    return out


@pytest.mark.parametrize("mid", METHODS)
def test_every_param_spec_corner_is_accepted(mid: str) -> None:
    for kw in _corner_values(mid):
        res = numopt.run(mid, problems.get("quadratic_ill"), max_iter=50, **kw)
        assert_valid_result(res, max_iter=50)
    for horizon in (0, 1, 4096):
        res = numopt.run(
            "silver_gd_strongly_convex", problems.get("quadratic_ill"), horizon=horizon, max_iter=50
        )
        assert_valid_result(res, max_iter=50)


@pytest.mark.parametrize("mid", SCHEDULES)
@pytest.mark.parametrize("pid", ["quadratic_bowl", "quadratic_ill", "quadratic_nd", "booth"])
def test_schedule_contract_and_counts(mid: str, pid: str) -> None:
    res = numopt.run(mid, problems.get(pid), max_iter=200)
    assert_valid_result(res, max_iter=200)
    assert res.n_iter == res.trace[-1].k == len(res.trace) - 1
    assert res.n_fev == res.n_gev == res.n_iter + 1
    assert res.n_hev == 1  # auto L (and μ) from one Hessian evaluation
    keys = {"grad", "direction", "alpha", "h", "checkpoint", "bound_f"}
    keys |= {"bound_dist"} if mid == "silver_gd_strongly_convex" else set()
    keys |= {"y", "theta"} if mid == "ogm" else set()
    for s in res.trace:
        assert set(s.info) == keys
        assert s.grad_norm == pytest.approx(float(np.linalg.norm(s.info["grad"])), rel=1e-14)
        assert s.step_size == s.info["alpha"]
    assert res.trace[0].info["direction"] is None and not res.trace[0].info["checkpoint"]
    for a, b in itertools.pairwise(res.trace):
        assert b.info["h"] == pytest.approx(b.info["alpha"] * res.extra["L"], rel=1e-14)
        x_rec = np.asarray(a.x) + b.info["alpha"] * np.asarray(b.info["direction"])
        assert_allclose(b.x, x_rec, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("restart", RESTART_MODES)
@pytest.mark.parametrize("pid", ["quadratic_bowl", "quadratic_ill", "rosenbrock", "himmelblau"])
def test_fista_contract(restart: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run("fista", prob, restart=restart, max_iter=5000)
    assert_valid_result(res, max_iter=5000)
    assert res.n_iter == res.trace[-1].k
    keys = {"direction", "alpha", "y", "grad_y", "grad_norm_y", "beta", "t", "restarted", "L"}
    for s in res.trace:
        assert set(s.info) == keys | {"trials"}
        assert s.grad_norm is None
    for a, b in itertools.pairwise(res.trace):
        x_rec = np.asarray(a.x) + b.info["alpha"] * np.asarray(b.info["direction"])
        assert_allclose(b.x, x_rec, rtol=1e-12, atol=1e-12)
        assert b.step_size == b.info["alpha"] == 1.0 / b.info["L"]
    if restart != "none" or pid != "rosenbrock":  # plain FISTA is slow in the valley
        assert res.converged, res.message
        assert res.trace[-1].info["grad_norm_y"] <= 1e-6
        mins = [np.asarray(m, dtype=float) for m in prob.minima]
        err = min(float(np.linalg.norm(res.x - m)) for m in mins)
        assert err <= 1e-5


@pytest.mark.parametrize("backtracking", [False, True])
@pytest.mark.parametrize("restart", RESTART_MODES)
def test_fista_counts_are_exact(restart: str, backtracking: bool) -> None:
    prob = problems.get("rosenbrock")
    calls = {"f": 0, "g": 0}

    def f(x: np.ndarray) -> float:
        calls["f"] += 1
        return prob.f(x)

    def grad(x: np.ndarray) -> np.ndarray:
        calls["g"] += 1
        assert prob.grad is not None
        return prob.grad(x)

    P = dataclasses.replace(prob, f=f, grad=grad)
    lr = 1e-3 if not backtracking else 1.0
    res = numopt.run("fista", P, restart=restart, lr=lr, backtracking=backtracking, max_iter=400)
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls["f"], calls["g"], 0)
    assert res.n_gev == res.n_iter
    trials = sum(len(s.info["trials"]) for s in res.trace)
    if backtracking:
        # f at x₀, every trial, and f(y_k) whenever y_k ≠ x_{k−1}
        extra_fy = sum(1 for a, b in itertools.pairwise(res.trace) if b.info["y"] != list(a.x))
        assert res.n_fev == 1 + trials + extra_fy
    else:
        assert trials == 0 and res.n_fev == 1 + res.n_iter


def test_finite_difference_gradient_is_counted() -> None:
    def f(x: np.ndarray) -> float:
        return float((x[0] - 1.0) ** 2 + 3.0 * (x[1] + 2.0) ** 2)

    res = numopt.run(
        "silver_gd_strongly_convex", f, x0=[0.0, 0.0], L=6.0, mu=2.0, max_iter=300, gtol=1e-7
    )
    assert res.converged
    assert_allclose(res.x, [1.0, -2.0], atol=1e-7)
    assert res.n_fev == (res.n_iter + 1) * (1 + 2 * 2)  # f per iterate + 2n per FD gradient
    res = numopt.run("fista", f, x0=[0.0, 0.0], gtol=1e-7)
    assert res.converged
    assert_allclose(res.x, [1.0, -2.0], atol=1e-6)
    assert res.n_fev > res.n_iter + 4 * res.n_gev - 1


def test_auto_mu_rejects_a_singular_quadratic() -> None:
    # λ = {0, ½, 1, 2, 3}: eigvalsh returns λ_min ≈ 1e-16 (rounding), which was taken as μ
    # with κ ≈ 1e16; the run then used h ≈ 2/L and made almost no progress in 1024 steps.
    rng = Rng(3)
    Q = _orthogonal(rng, 5)
    A = Q @ np.diag([0.0, 0.5, 1.0, 2.0, 3.0]) @ Q.T
    A = 0.5 * (A + A.T)
    P = _quadratic(A, np.ones(5), _normals(rng, 5), tags=("quadratic",))
    with pytest.raises(ValueError, match=r"give mu > 0 explicitly.*not certifiably"):
        numopt.run("silver_gd_strongly_convex", P)
    # rank-deficient least squares ½‖Xx − y‖² (column 5 = column 1 + column 2)
    X = _normals(rng, 10, 5)
    X[:, 4] = X[:, 0] + X[:, 1]
    lam = np.linalg.eigvalsh(X.T @ X)
    assert 0.0 < abs(lam[0]) < 64 * 5 * EPS * lam[-1]  # the regime of the defect
    P = _quadratic(X.T @ X, np.ones(5), _normals(rng, 5), tags=("quadratic",))
    with pytest.raises(ValueError, match=r"give mu > 0 explicitly"):
        numopt.run("silver_gd_strongly_convex", P)
    res = numopt.run("silver_gd", P, max_iter=8, gtol=0.0)  # auto L is unaffected
    assert res.extra["L"] == pytest.approx(lam[-1], rel=1e-12)


def test_auto_mu_keeps_a_resolvable_small_eigenvalue() -> None:
    # κ = 1e9 is ill-conditioned but λ_min is 1e4× above the rounding floor 64·n·ε·λ_max.
    rng = Rng(4)
    Q = _orthogonal(rng, 4)
    lam = np.array([1e-9, 1e-3, 0.1, 1.0])
    P = _quadratic(Q @ np.diag(lam) @ Q.T, np.zeros(4), _normals(rng, 4), tags=("quadratic",))
    res = numopt.run("silver_gd_strongly_convex", P, max_iter=8, gtol=0.0)
    assert res.extra["mu_source"] == "eigenvalue of the (constant) Hessian"
    assert res.extra["mu"] == pytest.approx(1e-9, rel=1e-6)  # abs error ≲ n·ε·λ_max


def test_constants_from_problem_extra() -> None:
    fs = problems.get("logreg_2d")
    P = Problem(
        id="logreg",
        name="logreg",
        latex="",
        f=fs.f,
        grad=fs.grad,
        hess=fs.hess,
        dim=2,
        domain=fs.domain,
        x0=list(fs.x0),
        extra={"L": fs.extra["L"], "mu": fs.extra["mu"]},
    )
    for mid, kw in [("silver_gd", {}), ("silver_gd_strongly_convex", {}), ("ogm", {})]:
        res = numopt.run(mid, P, max_iter=500, gtol=1e-8, **kw)
        assert res.converged, (mid, res.message)
        assert res.extra["L_source"] == 'problem.extra["L"]' and res.n_hev == 0
        assert_allclose(res.x, fs.minima[0], atol=1e-7)


def test_checkpoints_and_bounds_in_info() -> None:
    p = problems.get("quadratic_nd")
    r = numopt.run("silver_gd", p, max_iter=40, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [1, 3, 7, 15, 31]
    assert [s.info["bound_f"] for s in r.trace if s.info["checkpoint"]] == [
        M.silver_rate(k) for k in range(1, 6)
    ]
    r = numopt.run("silver_gd_strongly_convex", p, horizon=16, max_iter=50, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [16, 32, 48]
    tau = M.silver_sc_rate(100.0, 16)
    assert r.extra["kappa"] == pytest.approx(100.0, rel=1e-12)
    assert r.trace[32].info["bound_dist"] == pytest.approx(tau**2, rel=1e-10)
    assert r.trace[32].info["bound_f"] == pytest.approx(tau**2 / 2, rel=1e-10)
    r = numopt.run("ogm", p, max_iter=10, gtol=0.0)
    assert [s.k for s in r.trace if s.info["bound_f"] is not None] == [10]
    r = numopt.run("long_step_gd", p, pattern="3", max_iter=10, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [3, 6, 9]
    assert all(s.info["bound_f"] is None for s in r.trace)


def test_documented_default_behaviour() -> None:
    p = problems.get("quadratic_bowl")
    r = numopt.run("silver_gd_strongly_convex", p)
    assert r.converged and r.n_iter == 15
    assert_allclose(r.x, [1.0, -0.5], atol=1e-6)
    for mid, f_end in (("silver_gd", 5.5e-9), ("long_step_gd", 5.6e-2), ("ogm", 5.0e-7)):
        r = numopt.run(mid, p)
        assert not r.converged and "max_iter" in r.message
        assert r.fun == pytest.approx(f_end, rel=0.02)
    assert numopt.run("silver_gd_strongly_convex", problems.get("quadratic_ill")).n_iter == 239
    r = numopt.run("silver_gd_strongly_convex", problems.get("quadratic_nd"))
    assert r.n_iter == 367 and r.extra["horizon"] == 64


@pytest.mark.parametrize(("mid", "pid", "params"), M.FIXTURE_CASES)
def test_fixture_cases_run(mid: str, pid: str, params: dict[str, Any]) -> None:
    res = numopt.run(mid, problems.get(pid), **params)
    assert_valid_result(res)
    assert len(res.trace) < 400
    json.dumps(res.to_dict(), allow_nan=False)


def test_fixture_cases_cover_every_method() -> None:
    assert 3 <= len(M.FIXTURE_CASES) <= 6
    assert {c[0] for c in M.FIXTURE_CASES} == set(METHODS)
    all_ids = {p.id for p in problems.list_problems("unconstrained")}
    assert {c[1] for c in M.FIXTURE_CASES} <= all_ids


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


def test_max_iter_is_reported() -> None:
    p = problems.get("quadratic_nd")
    for mid in METHODS:
        res = numopt.run(mid, p, max_iter=5, gtol=1e-12)
        assert not res.converged and "max_iter" in res.message
        assert res.n_iter == 5 and len(res.trace) == 6


def test_divergence_is_reported() -> None:
    p = problems.get("quadratic_nd")  # L = 100
    for mid in SCHEDULES:  # L ten times too small: |1 − hλ/L| > 1 on the top eigenvector
        res = numopt.run(mid, p, L=10.0, max_iter=5000)
        assert not res.converged and "diverged" in res.message, mid
        assert_valid_result(res)
    res = numopt.run("fista", p, lr=3.0 / 100.0, backtracking=False, max_iter=10_000)
    assert not res.converged and "diverged" in res.message
    assert_valid_result(res)


def _const_problem(f: Any, grad: Any) -> Problem:
    return Problem(id="c", name="c", latex="", f=f, grad=grad, dim=2, domain=(), x0=[1.0, 2.0])


def test_non_finite_start_stops_at_once() -> None:
    P = _const_problem(lambda x: math.nan, lambda x: np.asarray(x))
    for mid in METHODS:
        kw = {"fista": {}, "silver_gd_strongly_convex": {"L": 1.0, "mu": 1.0}}.get(mid, {"L": 1.0})
        res = numopt.run(mid, P, **kw)
        assert not res.converged and res.n_iter == 0 and "not finite" in res.message
        assert_valid_result(res)


def test_fista_backtracking_failure_is_reported() -> None:
    # f is finite only at x₀ and ∇f is huge: every trial point x₀ − ∇f/L̄ (L̄ ≤ 2⁶⁰) moves
    # and gives f = inf, so no L̄ is accepted.
    x0 = np.array([1.0, 2.0])
    P = _const_problem(lambda x: 0.0 if np.array_equal(x, x0) else math.inf, lambda x: 1e30 * x)
    res = numopt.run("fista", P, lr=1.0)
    assert not res.converged and "backtracking failed" in res.message
    assert res.n_iter == 0 and res.n_fev == 1 + M.MAX_BACKTRACK  # f(y₁) = f(x₀) is reused
    assert_valid_result(res)


def test_fista_reports_a_stall_at_the_precision_floor() -> None:
    # Same f with a modest ∇f: backtracking grows L̄ until ∇f/L̄ < ulp(x₀); then x₁ = x₀ = y₂ and
    # every later iteration would repeat iteration 1.
    x0 = np.array([1.0, 2.0])
    P = _const_problem(lambda x: 0.0 if np.array_equal(x, x0) else math.inf, lambda x: x)
    res = numopt.run("fista", P, lr=1.0)
    assert not res.converged and "stalled" in res.message and res.n_iter == 1
    assert_array_equal(res.x, x0)
    assert_valid_result(res)
    # Without backtracking, a step far below ulp(x) also stalls (and does not spin to max_iter).
    P = _const_problem(lambda x: float(x @ x), lambda x: np.array([1e-20, 0.0]))
    res = numopt.run("fista", P, lr=1.0, backtracking=False, gtol=1e-30)
    assert not res.converged and "stalled" in res.message and res.n_iter == 1


def test_gtol_zero_does_not_report_convergence_at_a_tiny_nonzero_gradient() -> None:
    # √(gᵀg) underflows to 0 for ‖g‖ ~ 1e-170; the scaled norm keeps it nonzero.
    g_tiny = np.array([1e-170, -2e-170])
    assert float(np.sqrt(g_tiny @ g_tiny)) == 0.0
    assert M._norm(g_tiny) == pytest.approx(math.sqrt(5.0) * 1e-170, rel=1e-15)
    assert M._norm(np.zeros(3)) == 0.0
    P = _const_problem(lambda x: 0.0, lambda x: g_tiny.copy())
    res = numopt.run("silver_gd", P, L=1.0, gtol=0.0, max_iter=3)
    assert not res.converged and res.n_iter == 3
    res = numopt.run("silver_gd", P, L=1.0, gtol=1e-160, max_iter=3)
    assert res.converged and res.n_iter == 0


def test_rosenbrock_needs_an_explicit_L() -> None:
    for mid in SCHEDULES:
        with pytest.raises(ValueError, match="give L"):
            numopt.run(mid, problems.get("rosenbrock"))
    res = numopt.run("ogm", problems.get("rosenbrock"), L=2000.0, max_iter=20)
    assert res.extra["L_source"] == "given"
    with pytest.raises(ValueError, match="give mu"):
        numopt.run("silver_gd_strongly_convex", lambda x: float(x @ x), x0=[1.0, 1.0], L=2.0)


@pytest.mark.parametrize(
    ("mid", "kw"),
    [
        ("silver_gd", {"L": -1.0}),
        ("silver_gd", {"L": math.inf}),
        ("silver_gd", {"gtol": -1.0}),
        ("silver_gd", {"max_iter": 0}),
        ("silver_gd", {"max_iter": 2.5}),
        ("silver_gd", {"max_iter": True}),
        ("long_step_gd", {"pattern": "5"}),
        ("silver_gd_strongly_convex", {"horizon": 3}),
        ("silver_gd_strongly_convex", {"horizon": 2**21}),
        ("silver_gd_strongly_convex", {"horizon": -1}),
        ("silver_gd_strongly_convex", {"L": 1.0, "mu": 2.0}),
        ("ogm", {"L": math.nan}),
        ("fista", {"restart": "sometimes"}),
        ("fista", {"eta": 1.0}),
        ("fista", {"lr": -1.0}),
        ("fista", {"gtol": 0.0}),
        ("fista", {"max_iter": 0}),
    ],
)
def test_invalid_input_raises(mid: str, kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        numopt.run(mid, problems.get("quadratic_bowl"), **kw)


def test_inputs_are_not_mutated() -> None:
    x0 = np.array([-2.0, 2.0])
    keep = x0.copy()
    for mid in METHODS:
        numopt.run(mid, problems.get("quadratic_bowl"), x0=x0, max_iter=20)
        assert_array_equal(x0, keep)
