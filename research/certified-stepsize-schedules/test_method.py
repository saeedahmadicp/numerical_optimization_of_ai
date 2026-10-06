"""Tests for the certified step-size schedules (method.py).

Oracles, by kind:
* exact identities from the papers (Lemma 2.3 sum of silver steps, footnote 2 one-liner,
  Lemma 3.2 bounds, eq. 3.6 special values, eq. 1.4 inequality, Table 1 averages);
* a 600-digit Decimal implementation of Part I's recursion (eqs. 3.1–3.2, 3.9), written from
  the paper's formulas without the cancellation-free rewrite used in method.py;
* the closed form x_n − x* = Π_t (I − h_t A/L)(x₀ − x*) on quadratics (eigendecomposition);
* Kim & Fessler's Theorem 3: OGM1 attains its bound exactly on the Huber function (8.1);
* Hypothesis property tests: every certificate holds on random L-smooth convex functions
  with a known minimizer (Huber terms + PSD quadratic);
* the PEP SDP (pep.py, needs cvxpy; skipped otherwise): exact worst cases.

Run: .venv/bin/python -m pytest research/certified-stepsize-schedules -q
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from numopt import problems
from numopt.core.types import Problem

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Unique module names: other research folders also have a method.py / test_method.py.
M = _load("certified_stepsize_method", HERE / "method.py")
assert_valid_result = _load(
    "numopt_tests_conftest", ROOT / "tests" / "conftest.py"
).assert_valid_result

RHO = 1.0 + math.sqrt(2.0)
EPS = float(np.finfo(np.float64).eps)
PROPERTY = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------------------------
# Schedules: identities from the papers
# --------------------------------------------------------------------------------------


def test_two_adic_valuation() -> None:
    assert [M.two_adic_valuation(t) for t in range(1, 13)] == [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2]
    with pytest.raises(ValueError):
        M.two_adic_valuation(0)


def test_silver_schedule_first_steps() -> None:
    # Part II, eq. 2.1: α_t = 1 + ρ^{ν(t+1)−1} = [√2, 2, √2, 1+ρ, √2, 2, √2, 1+ρ², ...]
    s2 = math.sqrt(2.0)
    expected = [s2, 2.0, s2, 1 + RHO, s2, 2.0, s2, 1 + RHO**2]
    assert_allclose(M.silver_schedule(8), expected, rtol=1e-15)


def test_silver_schedule_matches_paper_one_liner_and_recursion() -> None:
    # footnote 2 of Part II: [1+rho**((k & -k).bit_length()-2) for k in range(1,64)]
    one_liner = [1 + RHO ** ((k & -k).bit_length() - 2) for k in range(1, 64)]
    assert_allclose(M.silver_schedule(63), one_liner, rtol=0, atol=0)
    # eq. 1.3: h_{2n+1} = [h_n, 1 + ρ^{k−1}, h_n] for n = 2^k − 1, with h_1 = [√2]
    h = [math.sqrt(2.0)]
    for k in range(1, 10):
        assert len(h) == 2**k - 1
        assert_allclose(M.silver_schedule(len(h)), h, rtol=1e-15)
        h = [*h, 1 + RHO ** (k - 1), *h]


@pytest.mark.parametrize("k", range(1, 16))
def test_lemma_2_3_sum_of_silver_steps(k: int) -> None:
    # Part II, Lemma 2.3: Σ_{t<2^k−1} α_t = ρ^k − 1
    assert_allclose(M.silver_schedule(2**k - 1).sum(), RHO**k - 1, rtol=1e-13)


def test_silver_rate_values_and_eq_1_4() -> None:
    assert M.silver_rate(0) == 0.5  # f(x₀) − f* ≤ L R²/2 (smoothness)
    for k in range(1, 40):
        n = 2**k - 1
        r = M.silver_rate(k)
        assert r == pytest.approx(1 / (1 + math.sqrt(4 * RHO ** (2 * k) - 3)), rel=1e-15)
        # eq. 1.4: r_k ≤ 1/(2ρ^{log₂ n})
        assert r <= 1 / (2 * RHO ** math.log2(n)) * (1 + 1e-15)
        # between OGM's 1/(n+1)² and the tight constant-step bound 1/(4n+2) — the latter only
        # from n = 15 on (r₃ = 0.0344 > 1/30 at n = 7)
        if n >= 15:
            assert 1 / (n + 1) ** 2 < r < 1 / (4 * n + 2)


def _sc_decimal(kappa: float, j: int) -> tuple[list[Decimal], list[Decimal], Decimal]:
    """Part I eqs. 3.1–3.2, 3.4, 3.9 in 600-digit arithmetic, straight from the paper.

    600 digits: 1 − z_n is formed by subtraction here, and τ_n reaches ~1e-181 at κ = 1.5, n = 256.
    """
    getcontext().prec = 600
    k = Decimal(kappa)
    z = Decimal(1) / k
    a_list, b_list = [], []

    def psi(t: Decimal) -> Decimal:
        return (1 + k * t) / (1 + t)

    a_list.append(psi(z))
    b_list.append(psi(z))
    for _ in range(j):
        xi = 1 - z
        root = (1 + xi * xi).sqrt()
        y, z = z / (xi + root), z * (xi + root)
        a_list.append(psi(y))
        b_list.append(psi(z))
    tau = ((1 - z) / (1 + z)) ** 2
    return a_list, b_list, tau


@pytest.mark.parametrize("kappa", [1.5, 4.0, 16.0, 100.0, 1e4, 1e8])
def test_kappa_schedule_against_high_precision_recursion(kappa: float) -> None:
    for j in range(0, 9):
        n = 2**j
        h, tau = M.silver_sc_schedule(kappa, n)
        a, b, tau_ref = _sc_decimal(kappa, j)
        # h⁽ⁿ⁾ = [h̃⁽ⁿᐟ²⁾, a_n, h̃⁽ⁿᐟ²⁾, b_n]: a_n at index n/2 − 1, b_n last (eq. 3.8)
        assert h[-1] == pytest.approx(float(b[j]), rel=1e-13)
        if j >= 1:
            assert h[n // 2 - 1] == pytest.approx(float(a[j]), rel=1e-13)
            # h⁽²⁾ = [a₂, b₂], h⁽⁴⁾ = [a₂, a₄, a₂, b₄], ...: entry i (1-based, i < n) is
            # a_{2B(i)}, B(i) = lowest power of 2 in i, i.e. level lsb(i) + 1
            for i in range(1, n):
                lsb = (i & -i).bit_length() - 1
                assert h[i - 1] == pytest.approx(float(a[lsb + 1]), rel=1e-13)
        # NOTE: the relative error of τ_n doubles per level (τ_n depends on (1 − 1/κ)^{2^j});
        # 2^j·64ε is that amplification of the input rounding, not a loose tolerance.
        assert tau == pytest.approx(float(tau_ref), rel=2**j * 64 * EPS, abs=1e-300)
        assert M.silver_sc_rate(kappa, n) == tau


@pytest.mark.parametrize("kappa", [2.0, 10.0, 100.0, 1e3, 1e6])
def test_kappa_schedule_bounds_and_special_values(kappa: float) -> None:
    hm, am = 2 * kappa / (kappa + 1), (kappa + 1) / 2
    h1, tau1 = M.silver_sc_schedule(kappa, 1)
    assert h1[0] == pytest.approx(hm, rel=1e-15)  # eq. 3.6: a₁ = b₁ = ψ(1/κ) = HM(1, κ)
    assert tau1 == pytest.approx(((kappa - 1) / (kappa + 1)) ** 2, rel=1e-13)
    prev_tau = tau1
    h = h1
    for j in range(1, 14):
        h, tau = M.silver_sc_schedule(kappa, 2**j)
        # (3.7) for b_n; a_n only satisfies 1 < a_n (see the NOTE in silver_gd_strongly_convex)
        assert np.all(h > 1.0) and np.all(h <= am * (1 + 8 * EPS))  # 8ε: rounding of ψ
        assert h[-1] >= hm * (1 - 1e-15)
        # a_n < b_n for κ > 1 in exact arithmetic; both round to (κ+1)/2 once saturated
        assert h[2 ** (j - 1) - 1] <= h[-1]
        assert tau <= prev_tau**2 * (1 + 1e-12) + 1e-300  # rate monotonicity τ_{2n} ≤ τ_n²
        prev_tau = tau
    if M.silver_sc_auto_horizon(kappa) <= 2**11:  # saturated by n = 2^13
        assert h[-1] == pytest.approx(am, rel=1e-6)  # b_n → AM(1, κ)


def test_kappa_schedule_limit_is_convex_silver() -> None:
    # Part II, Remark 2.2: as μ → 0 the κ-aware schedule tends to the convex one.
    for j in range(1, 8):
        h, _ = M.silver_sc_schedule(1e12, 2**j)
        assert_allclose(h[:-1], M.silver_schedule(2**j - 1), rtol=1e-9)


def test_kappa_one_is_exact_in_one_step() -> None:
    h, tau = M.silver_sc_schedule(1.0, 4)
    assert_allclose(h, 1.0)
    assert tau == 0.0
    assert M.silver_sc_auto_horizon(1.0) == 1


def test_auto_horizon_is_near_saturation() -> None:
    for kappa in (4.0, 50.0, 100.0, 1e4):
        n = M.silver_sc_auto_horizon(kappa)
        e = lambda m, kappa=kappa: -math.log(M.silver_sc_rate(kappa, m)) / m  # noqa: E731
        assert e(2 * n) < 1.01 * e(n)  # saturated: doubling gains < 1 %
        if n > 1:
            assert e(n) >= 1.01 * e(n // 2)  # and the previous doubling still gained ≥ 1 %
    assert M.silver_sc_auto_horizon(100.0) == 64


def test_long_step_patterns_match_table_1() -> None:
    for key, h in M.LONG_STEP_PATTERNS.items():
        assert len(h) == int(key)
        avg = float(np.mean(h))
        c = M.LONG_STEP_RATES[key]
        # Table 1: the proved coefficient is avg(h), rounded down for t ≥ 7
        assert 0.0 <= avg - c < 2e-5
    assert M.LONG_STEP_PATTERNS["2"] == (2.9, 1.5)
    assert max(M.LONG_STEP_PATTERNS["127"]) == 370.0
    # the t = 2^m − 1 patterns are palindromes with the longest step in the middle
    for key in ("3", "7", "15", "31", "63", "127"):
        h = M.LONG_STEP_PATTERNS[key]
        assert h == h[::-1] and h[len(h) // 2] == max(h)


def test_ogm_thetas_and_bound_eq_6_17() -> None:
    th = M.ogm_thetas(5)
    assert th[0] == 1.0
    for i in range(4):
        assert th[i + 1] == pytest.approx((1 + math.sqrt(1 + 4 * th[i] ** 2)) / 2, rel=1e-15)
    assert th[5] == pytest.approx((1 + math.sqrt(1 + 8 * th[4] ** 2)) / 2, rel=1e-15)
    for N in range(1, 200):
        b = M.ogm_rate(N)
        assert b <= 1 / ((N + 1) * (N + 1 + math.sqrt(2))) * (1 + 1e-14)  # eq. 6.17
        assert b <= 1 / (N + 1) ** 2


# --------------------------------------------------------------------------------------
# Iterates
# --------------------------------------------------------------------------------------


def _quadratic(A: np.ndarray, c: np.ndarray, x0: np.ndarray) -> Problem:
    return Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: 0.5 * float((x - c) @ A @ (x - c)),
        grad=lambda x: A @ (x - c),
        hess=lambda x: A,
        dim=len(c),
        domain=(),
        x0=x0.tolist(),
        tags=("quadratic",),
    )


def test_hand_computed_first_steps() -> None:
    p = problems.get("quadratic_bowl")  # Hessian eigenvalues 2 and 4 → L = 4 (auto)
    x0 = np.array(p.x0, dtype=float)
    g0 = np.asarray(p.grad(x0))
    r = M.silver_gd(p, max_iter=2, gtol=0.0)
    assert r.extra["L"] == pytest.approx(4.0, rel=1e-14)
    assert_allclose(r.trace[1].x, x0 - math.sqrt(2) / 4 * g0, rtol=1e-15)
    assert r.trace[1].step_size == pytest.approx(math.sqrt(2) / 4, rel=1e-15)
    r = M.long_step_gd(p, pattern="2", max_iter=1, gtol=0.0)
    assert_allclose(r.trace[1].x, x0 - 2.9 / 4 * g0, rtol=1e-15)
    # OGM1 with θ₀ = 1: x₁ = y₁ + (1/θ₁)(y₁ − x₀) = x₀ − (1 + 1/θ₁)∇f(x₀)/L, θ₁ for N ≥ 2
    r = M.ogm(p, max_iter=3, gtol=0.0)
    th1 = (1 + math.sqrt(5)) / 2
    assert_allclose(r.trace[1].x, x0 - (1 + 1 / th1) * g0 / 4, rtol=1e-14)
    assert_allclose(r.trace[1].info["y"], x0 - g0 / 4, rtol=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_iterates_equal_matrix_polynomial_on_quadratics(seed: int) -> None:
    # Independent formulation: x_n − x* = Π_t (I − h_t A/L)(x₀ − x*) evaluated in the
    # eigenbasis of A (no gradient calls).
    rng = np.random.default_rng(seed)
    n = 6
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.geomspace(0.01, 1.0, n) * 7.0
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    c = rng.standard_normal(n)
    x0 = rng.standard_normal(n)
    L = 7.0
    for res, h in [
        (M.silver_gd(_quadratic(A, c, x0), L=L, max_iter=31, gtol=0.0), M.silver_schedule(31)),
        (
            M.long_step_gd(_quadratic(A, c, x0), L=L, pattern="7", max_iter=21, gtol=0.0),
            np.tile(M.LONG_STEP_PATTERNS["7"], 3),
        ),
        (
            M.silver_gd_strongly_convex(
                _quadratic(A, c, x0), L=L, mu=0.07, horizon=8, max_iter=24, gtol=0.0
            ),
            np.tile(M.silver_sc_schedule(100.0, 8)[0], 3),
        ),
    ]:
        coeff = np.prod(1.0 - np.outer(h, lam) / L, axis=0)  # (n,)
        x_ref = c + Q @ (coeff * (Q.T @ (x0 - c)))
        # NOTE: rtol 1e-9, not 1e-12: the products contain factors |1 − h λ/L| up to ~30
        # (long steps), whose partial products amplify rounding by ~10³ before shrinking.
        assert_allclose(res.x, x_ref, rtol=1e-9, atol=1e-12 * np.linalg.norm(x0 - c))


@pytest.mark.parametrize("N", [1, 2, 3, 7, 20, 100])
@pytest.mark.parametrize("R", [0.1, 1.0, 37.0])
def test_ogm_attains_its_bound_on_the_kim_fessler_huber_function(N: int, R: float) -> None:
    # Kim & Fessler (2016), Theorem 3, eq. 8.1: f(x_N) − f* = L R²/(2θ_N²) exactly.
    L = 3.0
    th = M.ogm_thetas(N)[-1]
    thr = R / th**2

    def f(x: np.ndarray) -> float:
        r = float(np.linalg.norm(x))
        return L * R / th**2 * r - L * R**2 / (2 * th**4) if r >= thr else 0.5 * L * r * r

    def g(x: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(x))
        return L * R / th**2 * x / r if r >= thr else L * x

    p = Problem(id="hub", name="hub", latex="", f=f, grad=g, dim=3, domain=(), x0=[0, 0, 0])
    nu = np.array([2.0, -1.0, 2.0]) / 3.0
    res = M.ogm(p, x0=R * nu, L=L, max_iter=N, gtol=0.0)
    assert res.trace[-1].info["bound_f"] == pytest.approx(M.ogm_rate(N), rel=1e-15)
    assert res.fun == pytest.approx(L * R**2 / (2 * th**2), rel=1e-10)


# --------------------------------------------------------------------------------------
# Certificates on random smooth convex functions (Hypothesis)
# --------------------------------------------------------------------------------------


def _huber(r: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.where(np.abs(r) <= d, 0.5 * r * r, d * np.abs(r) - 0.5 * d * d)


def _dhuber(r: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.clip(r, -d, d)


@st.composite
def convex_instances(draw: st.DrawFn, strongly: bool = False) -> dict:
    """f(x) = Σ_j c_j hub_{δ_j}(a_jᵀ(x − x*)) + ½(x − x*)ᵀD(x − x*), f* = 0 at x*.

    hub'' ≤ 1, so L = Σ c_j‖a_j‖² + λ_max(D) is a valid global constant (μ = λ_min(D)).
    """
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    d = draw(st.integers(1, 4))
    m = draw(st.integers(0, 4))
    scale = 10.0 ** draw(st.integers(-3, 3))  # x scale
    fscale = 10.0 ** draw(st.integers(-3, 3))  # f scale
    a = rng.standard_normal((m, d))
    c = fscale * rng.uniform(0.1, 1.0, m)
    delta = scale * 10.0 ** rng.uniform(-3, 1, m)
    rank = d if strongly else draw(st.integers(0, d))
    B = rng.standard_normal((d, rank))
    D = fscale * (B @ B.T + (1e-2 * np.eye(d) if strongly else 0.0))
    xs = scale * rng.standard_normal(d)
    x0 = xs + scale * rng.standard_normal(d) * 10.0 ** rng.uniform(-1, 1)
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
        return a.T @ (c * _dhuber(a @ dx, delta)) + D @ dx

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
def test_property_silver_envelope(inst: dict) -> None:
    res = M.silver_gd(inst["p"], x0=inst["x0"], L=inst["L"], max_iter=63, gtol=0.0)
    for k in range(1, 7):
        n = 2**k - 1
        if n > res.n_iter:  # stopped at an exact stationary point (f = f*)
            assert res.converged and res.fun <= 1e-12 * inst["L"] * inst["R2"]
            break
        step = res.trace[n]
        assert step.info["checkpoint"] and step.info["bound_f"] == M.silver_rate(k)
        env = M.silver_rate(k) * inst["L"] * inst["R2"]
        # NOTE: absolute slack 1e-12·L R² covers rounding of f near f* = 0 (f ~ L R² scale).
        assert step.fun <= env * (1 + 1e-12) + 1e-12 * inst["L"] * inst["R2"]


@PROPERTY
@given(convex_instances(), st.integers(1, 40))
def test_property_ogm_bound(inst: dict, N: int) -> None:
    res = M.ogm(inst["p"], x0=inst["x0"], L=inst["L"], max_iter=N, gtol=0.0)
    if res.n_iter == N:
        bound = M.ogm_rate(N) * inst["L"] * inst["R2"]
        assert res.fun <= bound * (1 + 1e-12) + 1e-12 * inst["L"] * inst["R2"]


@PROPERTY
@given(convex_instances(strongly=True), st.sampled_from([1, 2, 4, 8, 16, 0]))
def test_property_kappa_aware_distance_bound(inst: dict, horizon: int) -> None:
    res = M.silver_gd_strongly_convex(
        inst["p"], x0=inst["x0"], L=inst["L"], mu=inst["mu"], horizon=horizon, max_iter=64, gtol=0.0
    )
    n, tau = res.extra["horizon"], res.extra["tau_horizon"]
    # NOTE: rounding of the iterates: each coordinate carries an error of a few ulps of the
    # scale max(|x₀|, |x*|), so distances are compared with an additive δ (not squared).
    scale = max(1e-300, float(np.max(np.abs(np.concatenate([inst["x0"], inst["xs"]])))))
    delta = 64 * EPS * math.sqrt(inst["p"].dim) * scale
    for s in res.trace[1:]:
        if s.info["checkpoint"]:
            d = float(np.linalg.norm(np.asarray(s.x) - inst["xs"]))
            assert s.info["bound_dist"] == pytest.approx(tau ** (s.k // n), rel=1e-12)
            assert d <= math.sqrt(s.info["bound_dist"] * inst["R2"]) * (1 + 1e-9) + delta


# --------------------------------------------------------------------------------------
# Contract, counts, failure paths
# --------------------------------------------------------------------------------------

ALL = [
    ("silver_gd", {}),
    ("silver_gd_strongly_convex", {}),
    ("long_step_gd", {"pattern": "15"}),
    ("ogm", {}),
]


@pytest.mark.parametrize(("mid", "kw"), ALL)
@pytest.mark.parametrize("pid", ["quadratic_bowl", "quadratic_ill", "quadratic_nd"])
def test_contract_and_counts(mid: str, kw: dict, pid: str) -> None:
    p = problems.get(pid)
    res = M.METHODS[mid](p, max_iter=200, **kw)
    assert_valid_result(res, max_iter=200)
    assert res.n_iter == res.trace[-1].k == len(res.trace) - 1
    assert res.n_fev == res.n_gev == res.n_iter + 1
    assert res.n_hev == 1  # auto L (and μ) from one Hessian evaluation
    for s in res.trace:
        assert {"grad", "direction", "alpha", "h", "checkpoint", "bound_f"} <= set(s.info)
        assert s.grad_norm == pytest.approx(float(np.linalg.norm(s.info["grad"])), rel=1e-15)
    for a, b in zip(res.trace[:-1], res.trace[1:], strict=True):
        assert_allclose(
            b.x,
            np.asarray(a.x) + b.info["alpha"] * np.asarray(b.info["direction"]),
            rtol=1e-12,
            atol=1e-12,
        )
    json.dumps(res.to_dict(), allow_nan=False)


def test_documented_default_behaviour() -> None:
    # the docstrings' claims, with the defaults
    p = problems.get("quadratic_bowl")
    r = M.silver_gd_strongly_convex(p)
    assert r.converged and r.n_iter == 15
    assert_allclose(r.x, [1.0, -0.5], atol=1e-6)
    for fn, f_end in ((M.silver_gd, 5.5e-9), (M.long_step_gd, 5.6e-2), (M.ogm, 5.0e-7)):
        r = fn(p)
        assert not r.converged and "max_iter" in r.message
        assert r.fun == pytest.approx(f_end, rel=0.02)
    assert M.silver_gd_strongly_convex(problems.get("quadratic_ill")).n_iter == 239
    assert M.silver_gd_strongly_convex(problems.get("quadratic_nd")).n_iter == 367
    # convex silver on a strongly convex quadratic: f(x_{2^k−1}) falls by ≈ ρ² per doubling
    r = M.silver_gd(p, max_iter=2047, gtol=0.0)
    ratios = [r.trace[2**k - 1].fun / r.trace[2 ** (k + 1) - 1].fun for k in range(6, 10)]
    assert_allclose(ratios, RHO**2, rtol=0.02)


def test_checkpoints_and_bounds_in_info() -> None:
    p = problems.get("quadratic_nd")
    r = M.silver_gd(p, max_iter=40, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [1, 3, 7, 15, 31]
    assert [s.info["bound_f"] for s in r.trace if s.info["checkpoint"]] == [
        M.silver_rate(k) for k in range(1, 6)
    ]
    r = M.silver_gd_strongly_convex(p, horizon=16, max_iter=50, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [16, 32, 48]
    tau = M.silver_sc_rate(100.0, 16)
    assert r.extra["kappa"] == pytest.approx(100.0, rel=1e-12)
    assert r.trace[32].info["bound_dist"] == pytest.approx(tau**2, rel=1e-10)
    assert r.trace[32].info["bound_f"] == pytest.approx(tau**2 / 2, rel=1e-10)
    r = M.ogm(p, max_iter=10, gtol=0.0)
    assert [s.k for s in r.trace if s.info["bound_f"] is not None] == [10]
    r = M.long_step_gd(p, pattern="3", max_iter=10, gtol=0.0)
    assert [s.k for s in r.trace if s.info["checkpoint"]] == [3, 6, 9]
    assert all(s.info["bound_f"] is None for s in r.trace)


def test_max_iter_and_divergence_are_reported() -> None:
    p = problems.get("quadratic_nd")
    r = M.silver_gd(p, max_iter=5, gtol=1e-12)
    assert not r.converged and "max_iter" in r.message and len(r.trace) == 6
    # L ten times too small: steps 10× too long → |1 − hλ/L| > 1 → divergence
    r = M.long_step_gd(p, L=10.0, pattern="7", max_iter=2000)
    assert not r.converged and "diverged" in r.message
    r = M.ogm(p, L=1.0, max_iter=2000)
    assert not r.converged and "diverged" in r.message


def test_logreg_full_batch_with_stated_constants() -> None:
    fs = problems.get("logreg_2d")
    p = Problem(
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
    for mid, kw in ALL:
        res = M.METHODS[mid](p, max_iter=500, gtol=1e-8, **kw)
        assert res.converged, (mid, res.message)
        assert res.extra["L_source"] == 'problem.extra["L"]'
        assert_allclose(res.x, fs.minima[0], atol=1e-7)


def test_bare_callable_uses_finite_differences() -> None:
    f = lambda x: (x[0] - 1.0) ** 2 + 3.0 * (x[1] + 2.0) ** 2  # noqa: E731
    res = M.silver_gd_strongly_convex(f, x0=[0.0, 0.0], L=6.0, mu=2.0, max_iter=300, gtol=1e-7)
    assert res.converged
    assert_allclose(res.x, [1.0, -2.0], atol=1e-7)
    assert res.n_fev == (res.n_iter + 1) * (1 + 2 * 2)  # f per iterate + 2n per FD gradient
    with pytest.raises(ValueError, match="give L"):
        M.silver_gd(f, x0=[0.0, 0.0])
    with pytest.raises(ValueError, match="give mu"):
        M.silver_gd_strongly_convex(f, x0=[0.0, 0.0], L=6.0)


@pytest.mark.parametrize(
    ("fn", "kw"),
    [
        (M.silver_gd, {"L": -1.0}),
        (M.silver_gd, {"gtol": -1.0}),
        (M.silver_gd, {"max_iter": 0}),
        (M.long_step_gd, {"pattern": "5"}),
        (M.silver_gd_strongly_convex, {"horizon": 3}),
        (M.silver_gd_strongly_convex, {"L": 1.0, "mu": 2.0}),
        (M.ogm, {"L": math.nan}),
    ],
)
def test_invalid_input_raises(fn, kw) -> None:
    with pytest.raises(ValueError):
        fn(problems.get("quadratic_bowl"), **kw)


def test_rosenbrock_needs_an_explicit_L() -> None:
    with pytest.raises(ValueError, match="give L"):
        M.silver_gd(problems.get("rosenbrock"))


# --------------------------------------------------------------------------------------
# PEP oracle (exact worst case via SDP; needs cvxpy)
# --------------------------------------------------------------------------------------


def test_pep_worst_cases() -> None:
    pytest.importorskip("cvxpy")
    pep = _load("certified_stepsize_pep", HERE / "pep.py")
    # validates the oracle: one step of size h has worst case max(1/(4h+2), (1−h)²/2)
    h = math.sqrt(2.0)
    assert pep.worst_case_gd([h]) == pytest.approx(max(1 / (4 * h + 2), (1 - h) ** 2 / 2), rel=1e-5)
    assert pep.worst_case_gd([1.0] * 3) == pytest.approx(1 / 14, rel=1e-5)  # Drori–Teboulle
    for k in (1, 2, 3):  # silver: worst case ≤ r_k (Part II, Theorem 1.1)
        assert pep.worst_case_gd(M.silver_schedule(2**k - 1)) <= M.silver_rate(k) * (1 + 1e-6)
    for N in (1, 3, 5):  # OGM1 bound is tight (Kim & Fessler 2016, Theorem 3)
        assert pep.worst_case_ogm(N) == pytest.approx(M.ogm_rate(N), rel=1e-5)
    for n in (1, 2, 4):  # κ-aware silver: worst case of ‖x_n − x*‖² equals τ_n
        hs, tau = M.silver_sc_schedule(16.0, n)
        assert pep.worst_case_gd(hs, mu=1 / 16, measure="dist") == pytest.approx(tau, rel=1e-5)
