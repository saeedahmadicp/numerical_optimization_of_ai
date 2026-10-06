"""Tests for method.py: FISTA/AGD with adaptive restart, ISTA and AdProxGD.

Oracles (none of them calls method.py's iteration code):
* hand-computed first iterations of Beck & Teboulle (4.1)–(4.3) and Malitsky–Mishchenko Alg. 3;
* O'Donoghue & Candès' θ-form (Algorithm 1 with q = 0; at a §3.2 restart, the reset of
  Algorithm 3, "fixed restarting"), coded here;
* exact minimizers: x* = A⁻¹b for quadratics, the KKT-certified lasso solution (reference.py);
* the papers' guarantees as properties: Theorem 4.4 (FISTA), Theorem 3.1 and Remark 3.1 (ISTA),
  Remark 3.2 (backtracking bounds), and the restart tests' definitions.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import scipy.optimize
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("rag_method", HERE / "method.py")
REF = _load("rag_reference", HERE / "reference.py")
assert_valid_result = _load(
    "rag_numopt_conftest", REPO / "tests" / "conftest.py"
).assert_valid_result

from numopt import problems  # noqa: E402
from numopt.core.types import Problem  # noqa: E402

PROPS = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def quadratic(lams: np.ndarray, Q: np.ndarray, x_star: np.ndarray, x0: np.ndarray) -> Problem:
    A = (Q * lams) @ Q.T
    return Problem(
        id="q",
        name="q",
        latex="",
        f=lambda x: 0.5 * float((x - x_star) @ (A @ (x - x_star))),
        dim=lams.size,
        domain=(),
        grad=lambda x: A @ (x - x_star),
        x0=x0,
    )


def diag_quadratic(kappa: float, n: int, seed: int = 0) -> tuple[Problem, np.ndarray]:
    rng = np.random.default_rng(seed)  # test data only; methods never draw randomness
    lams = kappa ** (np.arange(n) / (n - 1))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    x_star = rng.standard_normal(n)
    return quadratic(lams, Q, x_star, rng.standard_normal(n)), x_star


def random_lasso(seed: int, m: int = 20, n: int = 10, ratio: float = 0.2):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / math.sqrt(m)
    x_true = np.zeros(n)
    x_true[: max(1, n // 4)] = rng.standard_normal(max(1, n // 4)) + 2.0
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    lam = ratio * float(np.max(np.abs(A.T @ b)))
    return A, b, lam


def soft(v: np.ndarray, tau: float) -> np.ndarray:
    out = np.zeros_like(v)
    out[v > tau] = v[v > tau] - tau
    out[v < -tau] = v[v < -tau] + tau
    return out


# --------------------------------------------------------------------------------------
# The proximal map
# --------------------------------------------------------------------------------------


@PROPS
@given(
    v=st.floats(-1e3, 1e3, allow_nan=False),
    tau=st.floats(1e-6, 1e2, allow_nan=False),
)
def test_soft_threshold_is_the_prox_of_l1(v: float, tau: float) -> None:
    """S_τ(v) = argmin_u τ|u| + (u − v)²/2, checked against a bounded scalar minimizer."""
    got = float(M.soft_threshold(np.array([v]), tau)[0])
    res: Any = scipy.optimize.minimize_scalar(
        lambda u: tau * abs(u) + 0.5 * (u - v) ** 2,
        bounds=(min(v, 0.0) - 1.0, max(v, 0.0) + 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    # NOTE: atol 1e-7·(1 + |v|): Brent's bounded search stops at xatol on a kinked function.
    assert_allclose(got, float(res.x), rtol=0, atol=1e-7 * (1 + abs(v)))
    assert got == float(soft(np.array([v]), tau)[0])


# --------------------------------------------------------------------------------------
# FISTA: hand-computed steps and an independent formulation
# --------------------------------------------------------------------------------------


def test_fista_first_three_steps_match_beck_teboulle() -> None:
    A, b, lam = random_lasso(3)
    P = M.make_lasso(A, b, lam, x0=np.full(A.shape[1], 0.5))
    L = P.extra["L"]
    res = M.fista(P, restart="none", lr=1 / L, backtracking=False, gtol=1e-14, max_iter=3)
    x0 = np.full(A.shape[1], 0.5)

    def p_L(y: np.ndarray) -> np.ndarray:
        return soft(y - A.T @ (A @ y - b) / L, lam / L)

    x1 = p_L(x0)  # y₁ = x₀
    t2 = (1 + math.sqrt(5)) / 2  # t₁ = 1
    x2 = p_L(x1 + 0.0 * (x1 - x0))  # (t₁ − 1)/t₂ = 0
    t3 = (1 + math.sqrt(1 + 4 * t2**2)) / 2
    y3 = x2 + ((t2 - 1) / t3) * (x2 - x1)
    x3 = p_L(y3)
    for k, xk in enumerate((x0, x1, x2, x3)):
        assert_allclose(res.trace[k].x, xk, rtol=1e-13, atol=1e-15)
    assert res.trace[3].info["beta"] == pytest.approx((t2 - 1) / t3, rel=1e-15)
    assert_allclose(res.trace[3].info["y"], y3, rtol=1e-13, atol=1e-15)
    assert res.trace[2].info["beta"] == 0.0


def odc_algorithm1(
    f, grad, x0: np.ndarray, L: float, iters: int, restart: str
) -> tuple[list, list]:
    """O'Donoghue & Candès (2015), Algorithm 1 with q = 0 and t_k = 1/L, with the §3.2
    restart tests; a restart applies the reset of Algorithm 3, fixed restarting
    (x⁰ ← x^k, y⁰ ← x^k, θ₀ ← 1)."""
    x_prev = x0.copy()
    y = x0.copy()
    theta = 1.0
    xs = [x0.copy()]
    restarts = []
    f_prev = f(x0)
    for k in range(1, iters + 1):
        gy = grad(y)
        x = y - gy / L
        fired = False
        if restart == "gradient":
            fired = float(gy @ (x - x_prev)) > 0.0  # ∇f(y^{k−1})ᵀ(x^k − x^{k−1}) > 0
        elif restart == "function":
            fired = f(x) > f_prev
        if fired:
            restarts.append(k)
            theta, y = 1.0, x.copy()
        else:
            # θ_{k+1}² = (1 − θ_{k+1})θ_k²  (q = 0): the positive root.
            theta_next = 0.5 * (-(theta**2) + math.sqrt(theta**4 + 4 * theta**2))
            beta = theta * (1 - theta) / (theta**2 + theta_next)
            y = x + beta * (x - x_prev)
            theta = theta_next
        f_prev = f(x)
        x_prev = x
        xs.append(x.copy())
    return xs, restarts


@pytest.mark.parametrize("restart", ["none", "gradient", "function"])
def test_agd_matches_odonoghue_candes_theta_form(restart: str) -> None:
    P, _ = diag_quadratic(1e3, 12, seed=1)
    L = 1e3
    x0 = np.asarray(P.x0)
    xs_ref, r_ref = odc_algorithm1(P.f, P.grad, x0, L, 150, restart)
    res = M.fista(P, restart=restart, lr=1 / L, backtracking=False, gtol=1e-14, max_iter=150)
    assert res.n_iter == 150
    assert res.extra["restarts"] == r_ref
    # NOTE: the two forms compute β with different roundings; differences stay at the
    # rounding level of the iterates (‖x‖ ~ 5), far below the tolerance.
    for k in range(0, 151, 10):
        assert_allclose(res.trace[k].x, xs_ref[k], rtol=1e-9, atol=1e-11)
    if restart != "none":
        assert len(r_ref) >= 1


# --------------------------------------------------------------------------------------
# Guarantees from the papers (property tests)
# --------------------------------------------------------------------------------------

quad_inputs = st.tuples(
    st.integers(2, 6),  # n
    st.floats(1.0, 1e4),  # κ
    st.integers(0, 2**31 - 1),  # seed
)


@PROPS
@given(quad_inputs)
def test_fista_obeys_theorem_4_4_on_quadratics(args) -> None:
    n, kappa, seed = args
    P, x_star = diag_quadratic(kappa, n, seed)
    x0 = np.asarray(P.x0)
    L = kappa
    res = M.fista(P, restart="none", lr=1 / L, backtracking=False, gtol=1e-14, max_iter=60)
    R2 = float((x0 - x_star) @ (x0 - x_star))
    for s in res.trace[1:]:
        bound = 2 * L * R2 / (s.k + 1) ** 2
        assert s.fun <= bound * (1 + 1e-10) + 1e-12  # f* = 0


@PROPS
@given(quad_inputs)
def test_ista_obeys_theorem_3_1_and_is_monotone(args) -> None:
    n, kappa, seed = args
    P, x_star = diag_quadratic(kappa, n, seed)
    x0 = np.asarray(P.x0)
    res = M.ista(P, lr=1 / kappa, backtracking=False, gtol=1e-14, max_iter=60)
    R2 = float((x0 - x_star) @ (x0 - x_star))
    F = [s.fun for s in res.trace]
    for k in range(1, len(F)):
        assert F[k] <= kappa * R2 / (2 * k) * (1 + 1e-10) + 1e-12
        assert F[k] <= F[k - 1] * (1 + 1e-14) + 1e-300  # Remark 3.1


lasso_inputs = st.tuples(
    st.integers(0, 2**31 - 1),
    st.integers(2, 8),  # n
    st.floats(0.05, 0.9),  # λ/λ_max
)


@PROPS
@given(lasso_inputs, st.sampled_from(["none", "gradient", "function"]))
def test_lasso_bounds_restart_tests_and_backtracking(args, restart: str) -> None:
    seed, n, ratio = args
    A, b, lam = random_lasso(seed, m=n + 3, n=n, ratio=ratio)
    try:
        cert = REF.lasso_solution(A, b, lam)
    except RuntimeError:
        assume(False)
        return
    x0 = np.ones(n)
    P = M.make_lasso(A, b, lam, x0=x0)
    L = P.extra["L"]
    R2 = float((x0 - cert.x) @ (x0 - cert.x))
    tol = 1e-9 * (1 + abs(cert.F))

    # Theorem 4.4 (no restart, constant step).
    res = M.fista(P, restart="none", lr=1 / L, backtracking=False, gtol=1e-14, max_iter=40)
    for s in res.trace[1:]:
        assert s.fun - cert.F <= 2 * L * R2 / (s.k + 1) ** 2 + tol

    # Restart flags follow their definitions exactly; a restart zeroes the next momentum.
    res = M.fista(P, restart=restart, lr=1 / L, backtracking=False, gtol=1e-14, max_iter=40)
    tr = res.trace
    for k in range(1, len(tr)):
        x_k, x_km1 = np.asarray(tr[k].x), np.asarray(tr[k - 1].x)
        y_k = np.asarray(tr[k].info["y"])
        if restart == "function":
            expect = tr[k].fun > tr[k - 1].fun
        elif restart == "gradient":
            expect = float((y_k - x_k) @ (x_k - x_km1)) > 0.0
        else:
            expect = False
        assert tr[k].info["restarted"] == expect
        if expect:
            assert tr[k].info["t"] == 1.0
            if k + 1 < len(tr):
                assert tr[k + 1].info["beta"] == 0.0
                assert_array_equal(tr[k + 1].info["y"], x_k)
        assert F_of(P, x_k) == tr[k].fun

    # Backtracking (Remark 3.2): L_k non-decreasing and L_k ≤ max(L₀, ηL(f)).
    L0 = L / 37.0
    res = M.fista(
        P, restart=restart, lr=1 / L0, backtracking=True, eta=2.0, gtol=1e-14, max_iter=40
    )
    Ls = [s.info["L"] for s in res.trace]
    assert all(b_ >= a for a, b_ in itertools.pairwise(Ls))
    assert max(Ls) <= max(L0, 2.0 * L) * (1 + 1e-12)
    if restart == "none":  # Theorem 4.4 with α = η
        for s in res.trace[1:]:
            assert s.fun - cert.F <= 2 * 2.0 * L * R2 / (s.k + 1) ** 2 + tol


def F_of(P, x: np.ndarray) -> float:
    return float(P.f(x)) + float(P.g(x))


# --------------------------------------------------------------------------------------
# Convergence to independently known minimizers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kw"),
    [
        ("fista", {"restart": "gradient"}),
        ("fista", {"restart": "function"}),
        ("fista", {"restart": "none"}),
        ("ista", {}),
        ("adaptive_proxgd", {}),
    ],
)
def test_lasso_n50_reaches_certified_minimizer(method: str, kw: dict) -> None:
    rng = np.random.default_rng(7)
    m, n = 25, 50
    A = rng.standard_normal((m, n)) / math.sqrt(m)
    x_true = np.zeros(n)
    x_true[[3, 11, 20, 33, 47]] = [2.0, -1.5, 1.2, 1.8, -2.2]
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    cert = REF.lasso_solution(A, b, lam)
    P = M.make_lasso(A, b, lam)
    L = P.extra["L"]
    params = {"lr": 1 / L, "gtol": 1e-10, "max_iter": 20_000}
    if method != "adaptive_proxgd":
        params["backtracking"] = False
    res = M.METHODS[method](P, **params, **kw)
    assert res.converged, res.message
    assert_valid_result(res, max_iter=20_000)
    # NOTE: ‖G‖ ≤ 1e-10 and the restricted strong convexity μ_S = λ_min(A_SᵀA_S) bound the
    # distance by ‖G‖/μ_S ≈ 1e-9; tolerance 1e-8.
    assert_allclose(res.x, cert.x, rtol=0, atol=1e-8)
    assert tuple(np.flatnonzero(res.x)) == cert.support
    assert res.fun - cert.F <= 1e-12


def test_agd_gradient_restart_on_ill_conditioned_quadratic() -> None:
    """κ = 1e4: restarted AGD reaches ‖∇f(y_k)‖ ≤ 1e-8 in far fewer iterations than κ."""
    P, x_star = diag_quadratic(1e4, 30, seed=2)
    res = M.fista(P, restart="gradient", lr=1e-4, backtracking=False, gtol=1e-8, max_iter=10_000)
    assert res.converged
    assert res.n_iter < 3000  # O'Donoghue & Candès §4.5: O(√κ log(1/ε)); GD would need ~2e5
    assert_allclose(res.x, x_star, rtol=0, atol=1e-7)
    # One gradient per iteration and one f per iterate (k = 0 included): exact counts.
    assert res.n_gev == res.n_iter
    assert res.n_fev == res.n_iter + 1


def test_exact_step_on_isotropic_quadratic() -> None:
    """f = (L/2)‖x − c‖²: with lr = 1/L the first step lands on c exactly."""
    c = np.array([1.0, -2.0, 0.5])
    P = Problem(
        id="iso",
        name="iso",
        latex="",
        f=lambda x: 2.0 * float((x - c) @ (x - c)),
        dim=3,
        domain=(),
        grad=lambda x: 4.0 * (x - c),
        x0=np.zeros(3),
    )
    for fn in (M.fista, M.ista):
        res = fn(P, lr=0.25, backtracking=False)
        assert res.converged and res.n_iter == 2
        assert_array_equal(res.x, c)


# --------------------------------------------------------------------------------------
# AdProxGD (Malitsky & Mishchenko 2024, Algorithm 3)
# --------------------------------------------------------------------------------------


def test_adaptive_proxgd_first_steps_match_algorithm_3() -> None:
    A, b, lam = random_lasso(11)
    n = A.shape[1]
    x0 = np.full(n, 0.3)
    P = M.make_lasso(A, b, lam, x0=x0)
    a0 = 0.7 / P.extra["L"]
    res = M.adaptive_proxgd(P, lr=a0, gtol=1e-14, max_iter=3)

    def grad(x):
        return A.T @ (A @ x - b)

    x1 = soft(x0 - a0 * grad(x0), lam * a0)
    L1 = np.linalg.norm(grad(x1) - grad(x0)) / np.linalg.norm(x1 - x0)
    s = 2 * a0**2 * L1**2 - 1
    a1 = min(math.sqrt(2 / 3 + 1 / 3) * a0, a0 / math.sqrt(s) if s > 0 else math.inf)
    x2 = soft(x1 - a1 * grad(x1), lam * a1)
    th1 = a1 / a0
    L2 = np.linalg.norm(grad(x2) - grad(x1)) / np.linalg.norm(x2 - x1)
    s = 2 * a1**2 * L2**2 - 1
    a2 = min(math.sqrt(2 / 3 + th1) * a1, a1 / math.sqrt(s) if s > 0 else math.inf)
    x3 = soft(x2 - a2 * grad(x2), lam * a2)
    for k, xk in enumerate((x0, x1, x2, x3)):
        assert_allclose(res.trace[k].x, xk, rtol=1e-13, atol=1e-15)
    assert res.trace[2].info["alpha"] == pytest.approx(a1, rel=1e-14)
    assert res.trace[3].info["theta"] == pytest.approx(a2 / a1, rel=1e-14)
    assert res.trace[2].info["L_local"] == pytest.approx(L1, rel=1e-14)


@PROPS
@given(quad_inputs, st.floats(1e-3, 1e2))
def test_adaptive_proxgd_step_rule_invariants(args, a0_scale: float) -> None:
    """α_k ≤ √(2/3 + θ_{k−1})·α_{k−1} and α_k²(L_k² − 1/(2α²_{k−1})) ≤ ½ (eq. 30) for all k."""
    n, kappa, seed = args
    P, _ = diag_quadratic(kappa, n, seed)
    res = M.adaptive_proxgd(P, lr=a0_scale / kappa, gtol=1e-14, max_iter=40)
    tr = res.trace
    theta_prev = 1 / 3
    for k in range(2, len(tr)):
        a, a_prev, Lk = tr[k].info["alpha"], tr[k - 1].info["alpha"], tr[k].info["L_local"]
        assert a <= math.sqrt(2 / 3 + theta_prev) * a_prev * (1 + 1e-14)
        assert a**2 * (Lk**2 - 1 / (2 * a_prev**2)) <= 0.5 * (1 + 1e-10)
        theta_prev = tr[k].info["theta"]
    # On a convex quadratic the method converges for any α₀ > 0 (Theorem 3): f decreases
    # from f(x₀) by the end of the budget.
    assert tr[-1].fun < tr[0].fun


# --------------------------------------------------------------------------------------
# Contract (tests/conftest.py checks), failure paths, counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fista", "ista", "adaptive_proxgd"])
@pytest.mark.parametrize("pid", ["quadratic_bowl", "rosenbrock"])
def test_contract_on_library_problems(name: str, pid: str) -> None:
    P = problems.get(pid)
    lr = 1e-3 if name == "adaptive_proxgd" else 1.0
    res = M.METHODS[name](P, lr=lr, max_iter=20_000)
    assert_valid_result(res, max_iter=20_000)
    assert res.n_iter == res.trace[-1].k
    json.dumps(res.to_dict(), allow_nan=False)
    if pid == "quadratic_bowl" or name != "ista":
        assert res.converged, (name, pid, res.message)
        assert_allclose(res.x, np.asarray(P.minima[0]), atol=1e-4)


@pytest.mark.parametrize("name", ["fista", "ista", "adaptive_proxgd"])
def test_max_iter_path(name: str) -> None:
    P, _ = diag_quadratic(1e3, 5)
    res = M.METHODS[name](
        P,
        lr=1e-3,
        gtol=1e-14,
        max_iter=7,
        **({} if name == "adaptive_proxgd" else {"backtracking": False}),
    )
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 7 and len(res.trace) == 8
    assert_valid_result(res, max_iter=7)


@pytest.mark.parametrize("name", ["fista", "ista"])
def test_divergence_with_a_step_above_two_over_L(name: str) -> None:
    P, _ = diag_quadratic(1e2, 5)
    res = M.METHODS[name](P, lr=3.0 / 1e2, backtracking=False, max_iter=10_000)
    assert not res.converged and "diverged" in res.message
    assert_valid_result(res)


def test_nonfinite_start_and_invalid_parameters() -> None:
    P = Problem(
        id="nan",
        name="nan",
        latex="",
        f=lambda x: float("nan"),
        dim=2,
        domain=(),
        grad=lambda x: x,
        x0=np.zeros(2),
    )
    for fn in M.METHODS.values():
        res = fn(P)
        assert not res.converged and res.n_iter == 0
        assert_valid_result(res)
    Q, _ = diag_quadratic(10.0, 3)
    with pytest.raises(ValueError):
        M.fista(Q, restart="sometimes")
    with pytest.raises(ValueError):
        M.fista(Q, eta=1.0)
    with pytest.raises(ValueError):
        M.ista(Q, lr=-1.0)
    with pytest.raises(ValueError):
        M.adaptive_proxgd(Q, max_iter=0)


@pytest.mark.parametrize("backtracking", [False, True])
@pytest.mark.parametrize("name", ["fista", "ista", "adaptive_proxgd"])
def test_counts_are_exact(name: str, backtracking: bool) -> None:
    A, b, lam = random_lasso(5)
    P = M.make_lasso(A, b, lam)
    calls = {"f": 0, "g": 0}

    def f(x):
        calls["f"] += 1
        return P.f(x)

    def grad(x):
        calls["g"] += 1
        return P.grad(x)

    import dataclasses

    Pc = dataclasses.replace(P, f=f, grad=grad)
    kw = {} if name == "adaptive_proxgd" else {"backtracking": backtracking}
    res = M.METHODS[name](Pc, lr=1.0, gtol=1e-9, max_iter=3000, **kw)
    assert res.n_fev == calls["f"] and res.n_gev == calls["g"]
    assert res.n_gev == res.n_iter
    assert res.extra["n_prox"] == res.n_iter + sum(
        len(s.info.get("trials", [])) - 1 for s in res.trace if s.info.get("trials")
    )


def test_bare_callable_uses_finite_difference_gradient() -> None:
    res = M.fista(lambda x: float((x[0] - 1) ** 2 + 4 * (x[1] + 2) ** 2), x0=[0.0, 0.0], gtol=1e-7)
    assert res.converged
    assert_allclose(res.x, [1.0, -2.0], atol=1e-6)
    assert res.n_fev > res.n_iter  # central differences counted as f evaluations
