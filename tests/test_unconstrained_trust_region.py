"""Tests for numopt.unconstrained.trust_region (Cauchy, dogleg, Steihaug–Toint, exact)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import warnings
from typing import Any, cast

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.optimize import minimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.unconstrained import trust_region as tr

METHODS = (
    "trust_region_cauchy",
    "trust_region_dogleg",
    "trust_region_steihaug",
    "trust_region_exact",
)
COMMON_KEYS = {
    "center",
    "grad",
    "H",
    "radius",
    "new_radius",
    "step",
    "step_norm",
    "hits_boundary",
    "predicted",
    "actual",
    "rho",
    "accepted",
    "trial_point",
    "cauchy_point",
    "newton_point",
    "note",
}
EXTRA_KEYS = {
    "trust_region_cauchy": {"tau"},
    "trust_region_dogleg": {"dogleg_path", "tau"},
    "trust_region_steihaug": {"cg_path", "cg_iters", "termination", "cg_tol"},
    "trust_region_exact": {"lambda", "lambda_min", "hard_case", "lambda_iters"},
}
SOLVERS = {
    "cauchy": tr._cauchy,
    "dogleg": tr._dogleg,
    "steihaug": tr._steihaug,
    "exact": tr._exact,
}
EPS = float(np.finfo(float).eps)
#: The relative rounding allowance δ = ROUNDING·ε·max(|f(x)|, |f(x + p)|) of the module docstring.
ROUNDING = 1e3


def _model(g: np.ndarray, B: np.ndarray, p: np.ndarray) -> float:
    return float(g @ p + 0.5 * p @ B @ p)


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


# --------------------------------------------------------------------------------------
# Convergence and the contract
# --------------------------------------------------------------------------------------

CONVERGENCE_CASES = [
    ("trust_region_cauchy", "quadratic_bowl"),
    ("trust_region_cauchy", "himmelblau"),
    ("trust_region_cauchy", "six_hump_camel"),
    *[(m, pid) for m in METHODS[1:] for pid in ("rosenbrock", "himmelblau", "beale", "booth")],
    ("trust_region_steihaug", "rosenbrock_nd"),
    ("trust_region_exact", "rosenbrock_nd"),
    ("trust_region_dogleg", "quadratic_nd"),
    ("trust_region_exact", "quadratic_nd"),
    ("trust_region_steihaug", "quadratic_nd"),
]


@pytest.mark.parametrize(("method", "pid"), CONVERGENCE_CASES)
def test_converges_to_a_known_minimizer(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    assert float(np.max(np.abs(prob.grad(res.x)))) <= 1e-8
    dist = min(float(np.linalg.norm(res.x - np.asarray(m))) for m in prob.minima)
    # NOTE: ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)) ≤ √n·1e-8/λ_min; λ_min ≥ 0.4 on these problems.
    assert dist <= 1e-7
    assert res.n_iter == res.trace[-1].k


@pytest.mark.parametrize("method", METHODS)
def test_info_keys_documented_and_json(method: str) -> None:
    res = numopt.run(method, problems.get("himmelblau"))
    keys = COMMON_KEYS | EXTRA_KEYS[method]
    for s in res.trace:
        assert set(s.info) == keys
    json.dumps(res.to_dict(), allow_nan=False)
    assert tr.__doc__ is not None
    for key in keys:
        assert f"    {key}:" in tr.__doc__ or f" {key}:" in tr.__doc__
    first = res.trace[0].info
    assert first["step"] is None and first["accepted"] is None
    assert first["radius"] == first["new_radius"] == 1.0
    assert all(first[k] is None for k in EXTRA_KEYS[method])
    # H is given for 2-D problems only.
    assert np.asarray(first["H"]).shape == (2, 2)
    res_nd = numopt.run(method, problems.get("quadratic_nd"), max_iter=2)
    assert all(s.info["H"] is None for s in res_nd.trace)


# --------------------------------------------------------------------------------------
# SciPy oracles
# --------------------------------------------------------------------------------------

SCIPY_PATH_CASES = [
    ("trust_region_dogleg", "dogleg", "rosenbrock"),
    ("trust_region_dogleg", "dogleg", "three_hump_camel"),
    ("trust_region_dogleg", "dogleg", "booth"),
    ("trust_region_steihaug", "trust-ncg", "rosenbrock"),
    ("trust_region_steihaug", "trust-ncg", "beale"),
    ("trust_region_steihaug", "trust-ncg", "himmelblau"),
    ("trust_region_steihaug", "trust-ncg", "goldstein_price"),
    ("trust_region_steihaug", "trust-ncg", "rosenbrock_nd"),
]


@pytest.mark.parametrize(("method", "scipy_method", "pid"), SCIPY_PATH_CASES)
def test_iterates_match_scipy(method: str, scipy_method: str, pid: str) -> None:
    """SciPy's dogleg and trust-ncg implement the same subproblem solvers and the same radius
    update (η = 0.15, ¼/¾ thresholds); with Δ₀ = 1 and Δ̂ = 1000 every iterate must agree."""
    prob = problems.get(pid)
    xs: list[np.ndarray] = []
    ref = minimize(
        prob.f,
        prob.x0,
        jac=prob.grad,
        hess=prob.hess,
        method=scipy_method,
        options={"gtol": 1e-8, "initial_trust_radius": 1.0, "max_trust_radius": 1000.0},
        callback=lambda xk: xs.append(np.array(xk)),
    )
    assert ref.success
    res = numopt.run(method, prob, max_radius=1000.0, eta=0.15)
    assert res.converged, res.message
    assert res.n_iter == ref.nit
    mine = np.array([s.x for s in res.trace[1:]])
    # NOTE: the two codes evaluate the same formulas in a different order; the paths agree to
    # rounding amplified by the conditioning of B (observed ≤ 2e-11 on rosenbrock_nd).
    assert_allclose(mine, np.array(xs), rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("pid", "scipy_gtol"),
    [
        ("rosenbrock", 1e-10),
        ("beale", 1e-10),
        ("himmelblau", 1e-10),
        ("levi13", 1e-10),
        # NOTE: on rosenbrock_nd (local minimizer, f ≈ 3.99) SciPy's ratio test stalls at
        # ‖∇f‖∞ ≈ 1.1e-9 ("A bad approximation caused failure to predict improvement") because
        # its predicted reductions fall below the rounding level of f; we ask it for 1e-8. Our
        # ρ carries the rounding allowance of the module docstring and reaches 1e-10.
        ("rosenbrock_nd", 1e-8),
    ],
)
def test_exact_minimizer_matches_scipy_trust_exact(pid: str, scipy_gtol: float) -> None:
    prob = problems.get(pid)
    ref = minimize(
        prob.f,
        prob.x0,
        jac=prob.grad,
        hess=prob.hess,
        method="trust-exact",
        options={"gtol": scipy_gtol, "initial_trust_radius": 1.0, "max_trust_radius": 1000.0},
    )
    assert ref.success
    res = numopt.run("trust_region_exact", prob, max_radius=1000.0, gtol=1e-10)
    assert res.converged, res.message
    # NOTE: SciPy solves each subproblem only approximately (k_easy = 0.1), so the paths differ;
    # both end within ‖∇f‖/λ_min ≲ 1.1e-9/0.5 of the same minimizer (λ_min ≥ 0.5 here).
    assert_allclose(res.x, ref.x, rtol=0, atol=5e-9)


def test_dogleg_handles_indefinite_hessian_unlike_scipy() -> None:
    """At the origin of Himmelblau ∇²f ≺ 0: SciPy's dogleg stops with an error, ours uses the
    Cauchy point (N&W §4.1) and converges."""
    prob = problems.get("himmelblau")
    ref = minimize(prob.f, prob.x0, jac=prob.grad, hess=prob.hess, method="dogleg")
    assert not ref.success
    res = numopt.run("trust_region_dogleg", prob)
    assert res.converged, res.message
    first = res.trace[1].info
    assert first["note"] is not None and "Cauchy point" in first["note"]
    assert first["dogleg_path"] is None and first["newton_point"] is None
    assert_allclose(first["trial_point"], first["cauchy_point"], rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# Subproblem solvers: optimality certificates and textbook properties
# --------------------------------------------------------------------------------------


def _random_instance(n: int, seed: int, *, scale_b: float, scale_g: float, delta: float):
    rng = np.random.default_rng(seed)
    B = _sym(rng.standard_normal((n, n))) * scale_b
    g = rng.standard_normal(n) * scale_g
    return g, B, delta


def _check_kkt(g: np.ndarray, B: np.ndarray, delta: float, sub: Any) -> None:
    """N&W Thm. 4.1: p solves the subproblem iff (B + λI)p = −g, λ ≥ 0, λ(Δ − ‖p‖) = 0, B + λI ⪰ 0."""
    n = g.size
    p, lam = sub.p, sub.info["lambda"]
    eig = np.linalg.eigvalsh(B)
    scale = max(float(np.max(np.abs(eig))) * delta, float(np.linalg.norm(g)))
    pn = float(np.linalg.norm(p))
    # NOTE: tolerances: the residual of an eigendecomposition-based solve is ≈ nε·scale; ‖p‖ is
    # placed on the boundary to the solver's tolerance 1e-12·Δ.
    assert lam >= 0.0
    assert float(np.linalg.norm((B + lam * np.eye(n)) @ p + g)) <= 1e-12 * scale
    assert pn <= delta * (1.0 + 2e-12)
    assert lam * abs(delta - pn) <= 2e-12 * scale
    assert float(np.min(eig)) + lam >= -1e-13 * max(float(np.max(np.abs(eig))), 1e-300)
    assert sub.hits_boundary == (lam > 0.0 or sub.info["hard_case"])


@settings(max_examples=1500, deadline=None)
@given(
    n=st.integers(1, 6),
    seed=st.integers(0, 2**31 - 1),
    log_b=st.floats(-2.0, 2.0),
    log_g=st.floats(-3.0, 2.0),
    log_delta=st.floats(-2.0, 2.0),
    near_hard=st.sampled_from([None, 0.0, 1e-15, 1e-10, 1e-5]),
)
def test_exact_solver_kkt_certificate(
    n: int,
    seed: int,
    log_b: float,
    log_g: float,
    log_delta: float,
    near_hard: float | None,
) -> None:
    g, B, delta = _random_instance(
        n, seed, scale_b=10.0**log_b, scale_g=10.0**log_g, delta=10.0**log_delta
    )
    if near_hard is not None:
        # Remove (most of) g's component along the eigenvector of λ₁: the (nearly) hard case.
        _, Q = np.linalg.eigh(B)
        g = g - (1.0 - near_hard) * float(Q[:, 0] @ g) * Q[:, 0]
    if not np.any(g):
        return
    sub = tr._exact(g, B, delta)
    assert sub.note is None
    _check_kkt(g, B, delta, sub)
    # The global minimizer is never worse than the other solvers' feasible steps.
    m_exact = _model(g, B, sub.p)
    for name in ("cauchy", "dogleg", "steihaug"):
        q = SOLVERS[name](g, B, delta).p
        assert float(np.linalg.norm(q)) <= delta * (1.0 + 1e-12)
        assert m_exact <= _model(g, B, q) + 1e-12 * abs(_model(g, B, q)) + 1e-300


@settings(max_examples=300, deadline=None)
@given(
    g=arrays(np.float64, 2, elements=st.floats(-10.0, 10.0)),
    b=arrays(np.float64, 3, elements=st.floats(-10.0, 10.0)),
    delta=st.floats(0.05, 5.0),
)
def test_exact_solver_beats_brute_force_in_2d(g: np.ndarray, b: np.ndarray, delta: float) -> None:
    """Independent oracle: the minimum of m over a fine circle ‖p‖ = Δ and the interior Newton
    point (the only interior candidate) bounds the exact solver's model value from above."""
    if float(np.linalg.norm(g)) < 1e-6:
        return
    B = np.array([[b[0], b[1]], [b[1], b[2]]])
    theta = np.linspace(0.0, 2.0 * math.pi, 40001)
    P = delta * np.stack([np.cos(theta), np.sin(theta)])  # (2, m)
    values = g @ P + 0.5 * np.einsum("im,ij,jm->m", P, B, P)
    best = float(values.min())
    if np.all(np.linalg.eigvalsh(B) > 0.0):
        with np.errstate(over="ignore"):  # B may be nearly singular (λ ≈ 1e-166): ‖pn‖ = inf
            pn = -np.linalg.solve(B, g)
            pn_norm = float(np.linalg.norm(pn))
        if pn_norm <= delta:
            best = min(best, _model(g, B, pn))
    m_exact = _model(g, B, tr._exact(g, B, delta).p)
    assert m_exact <= best + 1e-12 * (1.0 + abs(best))
    # The grid misses the boundary optimum by at most ½‖∇²_θ m‖(Δθ/2)² ≈ 1e-7 here.
    assert m_exact >= best - 1e-6 * (1.0 + abs(best))


def test_exact_solver_hard_case() -> None:
    """N&W §4.3: B = diag(−2, 1, 3), g ⊥ e₁, ‖p_⊥(λ = 2)‖ = ‖(0, −1/3, −1/5)‖ < Δ = 1."""
    B = np.diag([-2.0, 1.0, 3.0])
    g = np.array([0.0, 1.0, 1.0])
    sub = tr._exact(g, B, 1.0)
    assert sub.info["hard_case"] and sub.info["lambda"] == 2.0 and sub.hits_boundary
    p_perp = np.array([0.0, -1.0 / 3.0, -1.0 / 5.0])
    tau = math.sqrt(1.0 - float(p_perp @ p_perp))
    assert_allclose(sub.p, p_perp + tau * np.array([1.0, 0.0, 0.0]), rtol=0, atol=1e-15)
    _check_kkt(g, B, 1.0, sub)
    # Nearly hard case: a tiny component along e₁ picks the sign of τ that lowers m, and the
    # solution tends to the hard-case solution as the component vanishes.
    for eps_g in (1e-14, 1e-10, -1e-10, 1e-6):
        g2 = np.array([eps_g, 1.0, 1.0])
        sub2 = tr._exact(g2, B, 1.0)
        _check_kkt(g2, B, 1.0, sub2)
        assert sub2.p[0] * eps_g < 0.0
        assert abs(abs(sub2.p[0]) - tau) <= 10.0 * abs(eps_g)
    # A two-dimensional eigenspace of λ₁ = −2: g has a rounding-level residual (1, 1)·1e-15 in
    # it. m(p_⊥ + τu) = m(p_⊥) + τuᵀg + ½τ²λ₁ is lowest for u = −(1, 1, 0)/√2.
    B3 = np.diag([-2.0, -2.0, 1.0])
    g3 = np.array([1e-15, 1e-15, 1.0])
    sub3 = tr._exact(g3, B3, 1.0)
    assert sub3.info["hard_case"] and sub3.info["lambda"] == 2.0
    tau3 = math.sqrt(1.0 - 1.0 / 9.0)
    expected = np.array([-tau3 / math.sqrt(2.0), -tau3 / math.sqrt(2.0), -1.0 / 3.0])
    assert_allclose(sub3.p, expected, rtol=0, atol=1e-15)
    _check_kkt(g3, B3, 1.0, sub3)


def test_exact_solver_interior_and_large_radius() -> None:
    B = np.array([[3.0, 1.0], [1.0, 2.0]])
    g = np.array([1.0, -1.0])
    sub = tr._exact(g, B, 10.0)
    assert sub.info["lambda"] == 0.0 and not sub.hits_boundary
    assert_allclose(sub.p, -np.linalg.solve(B, g), rtol=1e-14)


def test_exact_solver_safety_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the λ iteration capped at one step the fallback p(λ) + τq₁ must still be a feasible
    boundary step with residual τ(λ + λ₁)q₁ and a note."""
    monkeypatch.setattr(tr, "_LAMBDA_MAX_ITER", 1)
    B = np.diag([-2.0, 1.0, 3.0])
    g = np.array([1e-3, 1.0, 1.0])
    sub = tr._exact(g, B, 1.0)
    assert sub.note is not None and sub.hits_boundary
    assert float(np.linalg.norm(sub.p)) == pytest.approx(1.0, rel=1e-14)
    lam = sub.info["lambda"]
    resid = (B + lam * np.eye(3)) @ sub.p + g
    assert abs(resid[1]) <= 1e-14 and abs(resid[2]) <= 1e-14  # only the q₁ = e₁ component
    assert lam >= 2.0


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(1, 6),
    seed=st.integers(0, 2**31 - 1),
    log_b=st.floats(-2.0, 2.0),
    log_delta=st.floats(-2.0, 2.0),
    solver=st.sampled_from(sorted(SOLVERS)),
)
def test_cauchy_decrease_and_feasibility(
    n: int, seed: int, log_b: float, log_delta: float, solver: str
) -> None:
    """N&W Lemma 4.3: every solver achieves m(0) − m(p) ≥ ½‖g‖ min(Δ, ‖g‖/‖B‖), and ‖p‖ ≤ Δ."""
    g, B, delta = _random_instance(n, seed, scale_b=10.0**log_b, scale_g=1.0, delta=10.0**log_delta)
    sub = SOLVERS[solver](g, B, delta)
    gn = float(np.linalg.norm(g))
    bn = float(np.linalg.norm(B, 2))
    bound = 0.5 * gn * min(delta, gn / bn if bn > 0.0 else math.inf)
    pred = -_model(g, B, sub.p)
    assert pred >= bound * (1.0 - 1e-12)
    assert float(np.linalg.norm(sub.p)) <= delta * (1.0 + 1e-12)
    if sub.hits_boundary:
        assert float(np.linalg.norm(sub.p)) == pytest.approx(delta, rel=1e-12)


def test_cauchy_point_closed_form() -> None:
    g = np.array([3.0, -4.0])  # ‖g‖ = 5
    B = np.diag([1.0, 2.0])  # gᵀBg = 9 + 32 = 41
    sub = tr._cauchy(g, B, 1.0)  # τ = min(125/41, 1) = 1 → boundary
    assert sub.hits_boundary and sub.info["tau"] == 1.0
    assert_allclose(sub.p, -g / 5.0, rtol=1e-15)
    sub = tr._cauchy(g, B, 10.0)  # τ = 125/410 → interior minimizer along −g: −(25/41) g
    assert not sub.hits_boundary
    assert_allclose(sub.p, -(25.0 / 41.0) * g, rtol=1e-15)
    sub = tr._cauchy(g, -B, 2.0)  # negative curvature along g → boundary
    assert sub.hits_boundary and float(np.linalg.norm(sub.p)) == pytest.approx(2.0)


def test_dogleg_regimes_and_lemma_4_2() -> None:
    B = np.array([[4.0, 1.0], [1.0, 1.5]])
    g = np.array([2.0, 1.0])
    pB = -np.linalg.solve(B, g)
    pU = -(g @ g) / (g @ B @ g) * g
    big = tr._dogleg(g, B, 10.0)  # Newton point inside
    assert not big.hits_boundary and big.info["tau"] == 2.0
    assert_allclose(big.p, pB, rtol=1e-14)
    small = tr._dogleg(g, B, 0.1 * float(np.linalg.norm(pU)))  # first segment
    assert small.hits_boundary and 0.0 < small.info["tau"] <= 1.0
    assert_allclose(small.p, small.info["tau"] * pU, rtol=1e-14)
    mid_delta = 0.5 * (float(np.linalg.norm(pU)) + float(np.linalg.norm(pB)))
    mid = tr._dogleg(g, B, mid_delta)  # second segment
    assert mid.hits_boundary and 1.0 < mid.info["tau"] < 2.0
    s = mid.info["tau"] - 1.0
    assert_allclose(mid.p, pU + s * (pB - pU), rtol=1e-14)
    assert float(np.linalg.norm(mid.p)) == pytest.approx(mid_delta, rel=1e-14)
    # Lemma 4.2: along the path ‖p̃(τ)‖ increases and m(p̃(τ)) decreases.
    taus = np.linspace(0.0, 2.0, 201)
    path = [t * pU if t <= 1.0 else pU + (t - 1.0) * (pB - pU) for t in taus]
    norms = [float(np.linalg.norm(p)) for p in path]
    values = [_model(g, B, p) for p in path]
    assert all(b >= a for a, b in itertools.pairwise(norms))
    assert all(b <= a + 1e-15 for a, b in itertools.pairwise(values))


def test_steihaug_properties() -> None:
    rng = np.random.default_rng(3)
    M = rng.standard_normal((6, 6))
    B = M @ M.T + 0.5 * np.eye(6)
    g = rng.standard_normal(6)
    # Large radius: stops on the residual test ‖Bp + g‖ < min(½, √‖g‖)‖g‖ (N&W eq. 7.3).
    sub = tr._steihaug(g, B, 1e6)
    assert sub.info["termination"] == "residual" and not sub.hits_boundary
    assert float(np.linalg.norm(B @ sub.p + g)) < sub.info["cg_tol"]
    # The first CG iterate z₁ is the Cauchy point (N&W §7.1, after Alg. 7.2).
    cauchy = tr._cauchy(g, B, 1e6).p
    assert_allclose(sub.info["cg_path"][1], cauchy, rtol=1e-13)
    # The iterates increase in norm (N&W Thm. 7.3 / Steihaug 1983).
    norms = [float(np.linalg.norm(z)) for z in sub.info["cg_path"]]
    assert all(b > a for a, b in itertools.pairwise(norms))
    # Small radius: boundary.
    sub = tr._steihaug(g, B, 1e-2)
    assert sub.info["termination"] == "boundary" and sub.hits_boundary
    # Negative curvature in the first direction: the step goes to the boundary along −g.
    sub = tr._steihaug(np.array([1.0, 2.0]), -np.eye(2), 0.5)
    assert sub.info["termination"] == "negative_curvature"
    assert_allclose(sub.p, -0.5 * np.array([1.0, 2.0]) / math.sqrt(5.0), rtol=1e-15)


def test_boundary_tau_is_stable() -> None:
    z = np.array([0.3, -0.1])
    d = np.array([1e-8, 2e-8])
    t_minus, t_plus = tr._boundary_tau(z, d, 1.0)
    for t in (t_minus, t_plus):
        assert float(np.linalg.norm(z + t * d)) == pytest.approx(1.0, rel=1e-15)
    assert t_minus < 0.0 < t_plus


# --------------------------------------------------------------------------------------
# The driver: Alg. 4.1 bookkeeping, exact counts, monotonicity
# --------------------------------------------------------------------------------------


def _check_driver_trace(res: Any, prob: Problem, eta: float, max_radius: float) -> None:
    assert prob.grad is not None and prob.hess is not None
    for prev, cur in itertools.pairwise(res.trace):
        info = cur.info
        center = np.asarray(info["center"])
        assert_allclose(center, prev.x, rtol=0, atol=0)
        assert info["radius"] == prev.info["new_radius"] == cur.step_size
        step = np.asarray(info["step"])
        assert info["step_norm"] <= info["radius"] * (1.0 + 1e-12)
        g, B = prob.grad(center), prob.hess(center)
        pred = -(float(g @ step) + 0.5 * float(step @ B @ step))
        assert info["predicted"] == pytest.approx(pred, rel=1e-9, abs=1e-300)
        assert info["predicted"] > 0.0
        rho = info["rho"]
        rho_v = -math.inf if rho is None else rho
        radius = info["radius"]
        if rho_v < 0.25:
            expected = 0.25 * radius
        elif rho_v > 0.75 and info["hits_boundary"]:
            expected = min(2.0 * radius, max_radius)
        else:
            expected = radius
        assert info["new_radius"] == expected
        assert info["accepted"] == (rho_v > eta)
        if rho is not None:
            # ρ = (actual + δ)/(predicted + δ) with the relative allowance δ (module docstring).
            f_trial = float(prob.f(np.asarray(info["trial_point"])))
            assert info["actual"] == prev.fun - f_trial
            delta = ROUNDING * EPS * max(abs(prev.fun), abs(f_trial))
            assert rho == pytest.approx(
                (info["actual"] + delta) / (info["predicted"] + delta), rel=1e-15
            )
        if info["accepted"]:
            assert_allclose(cur.x, center + step, rtol=0, atol=0)
            # ρ > η ≥ 0 allows at most a relative rounding-level increase (no absolute floor).
            assert cur.fun < prev.fun + ROUNDING * EPS * max(abs(prev.fun), abs(cur.fun))
        else:
            assert_allclose(cur.x, prev.x, rtol=0, atol=0)
            assert cur.fun == prev.fun


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["rosenbrock", "himmelblau", "six_hump_camel", "beale"])
def test_radius_update_follows_algorithm_4_1(method: str, pid: str) -> None:
    prob = problems.get(pid)
    res = numopt.run(method, prob, max_iter=150)
    assert_valid_result(res, max_iter=150)
    _check_driver_trace(res, prob, eta=0.15, max_radius=100.0)


@settings(max_examples=120, deadline=None)
@given(
    x=st.floats(-4.5, 4.5),
    y=st.floats(-4.5, 4.5),
    method=st.sampled_from(METHODS[1:]),
    eta=st.sampled_from([0.0, 0.1, 0.2]),
    radius0=st.floats(0.05, 3.0),
)
def test_random_starts_himmelblau(
    x: float, y: float, method: str, eta: float, radius0: float
) -> None:
    prob = problems.get("himmelblau")
    res = numopt.run(method, prob, x0=[x, y], eta=eta, radius0=radius0)
    assert_valid_result(res, max_iter=200)
    _check_driver_trace(res, prob, eta=eta, max_radius=100.0)
    assert res.converged, res.message
    # Exact and Steihaug use negative curvature; all three reach a stationary point.
    assert float(np.max(np.abs(prob.grad(res.x)))) <= 1e-8


def test_exact_steps_satisfy_kkt_along_a_run() -> None:
    prob = problems.get("himmelblau")
    res = numopt.run("trust_region_exact", prob)
    for s in res.trace[1:]:
        info = s.info
        center = np.asarray(info["center"])
        g, B = prob.grad(center), prob.hess(center)
        sub = tr._Subproblem(np.asarray(info["step"]), info["hits_boundary"], dict(info))
        _check_kkt(g, B, info["radius"], sub)


def _counting(prob: Problem) -> tuple[Problem, dict[str, int]]:
    counts = {"f": 0, "g": 0, "h": 0}

    def f(x: Any) -> Any:
        counts["f"] += 1
        return prob.f(x)

    def g(x: Any) -> Any:
        counts["g"] += 1
        assert prob.grad is not None
        return prob.grad(x)

    def h(x: Any) -> Any:
        counts["h"] += 1
        assert prob.hess is not None
        return prob.hess(x)

    return dataclasses.replace(prob, f=f, grad=g, hess=h), counts


@pytest.mark.parametrize("method", METHODS)
def test_evaluation_counts_are_exact(method: str) -> None:
    prob, counts = _counting(problems.get("himmelblau"))
    res = numopt.run(method, prob)
    assert res.converged, res.message
    assert (res.n_fev, res.n_gev, res.n_hev) == (counts["f"], counts["g"], counts["h"])
    accepted = sum(bool(s.info["accepted"]) for s in res.trace[1:])
    assert res.n_fev == 1 + res.n_iter
    assert res.n_gev == res.n_hev == 1 + accepted
    assert res.extra["n_rejected"] == res.n_iter - accepted


def test_geometry_keys() -> None:
    prob = problems.get("rosenbrock")
    res = numopt.run("trust_region_dogleg", prob)
    for s in res.trace[1:]:
        info = s.info
        center = np.asarray(info["center"])
        g, B = prob.grad(center), prob.hess(center)
        assert_allclose(info["grad"], g, rtol=0, atol=0)
        assert_allclose(info["H"], B, rtol=0, atol=0)
        assert_allclose(info["trial_point"], center + np.asarray(info["step"]), rtol=0, atol=0)
        assert_allclose(
            np.asarray(info["cauchy_point"]) - center,
            tr._cauchy(g, B, info["radius"]).p,
            atol=1e-15,
        )
        if info["newton_point"] is not None:
            assert_allclose(
                np.asarray(info["newton_point"]) - center, -np.linalg.solve(B, g), rtol=1e-10
            )
            path = np.asarray(info["dogleg_path"])
            assert path.shape == (3, 2)
            assert_allclose(path[0], center, atol=0)
            assert_allclose(path[2], info["newton_point"], atol=1e-14)


def test_steihaug_negative_curvature_at_himmelblau_origin() -> None:
    res = numopt.run("trust_region_steihaug", problems.get("himmelblau"))
    info = res.trace[1].info
    assert info["termination"] == "negative_curvature" and info["hits_boundary"]
    assert info["cg_path"][0] == info["center"]
    assert_allclose(info["cg_path"][-1], info["trial_point"], atol=1e-15)


def test_exact_reports_negative_lambda_min_at_himmelblau_origin() -> None:
    res = numopt.run("trust_region_exact", problems.get("himmelblau"))
    info = res.trace[1].info
    assert info["lambda_min"] < 0.0 and info["lambda"] > -info["lambda_min"]


def test_bare_callable_uses_finite_differences() -> None:
    """f = (x − 1)² + 4(y + ½)² + x⁴/10: ∂f/∂x = 0 ⇔ x³ + 5x − 5 = 0 (one real root ≈ 0.8688)."""
    res = numopt.minimize(
        lambda x: (x[0] - 1.0) ** 2 + 4.0 * (x[1] + 0.5) ** 2 + 0.1 * x[0] ** 4,
        x0=[3.0, 1.0],
        method="trust_region_exact",
        gtol=1e-6,
    )
    assert_valid_result(res)
    assert res.converged, res.message
    roots = np.roots([1.0, 0.0, 5.0, -5.0])
    x_star = float(roots[np.abs(roots.imag) < 1e-12].real[0])
    # ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f) ≤ √2·1e-6/2 (∇²f = diag(2 + 1.2x², 8)), plus the central-
    # difference error ≈ ε^{2/3} of the gradient.
    assert_allclose(res.x, [x_star, -0.5], rtol=0, atol=1e-6)
    assert res.n_fev > 4 * res.n_gev  # every FD gradient costs 2n = 4 evaluations of f


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


def test_cauchy_is_slow_on_rosenbrock() -> None:
    """The Cauchy point is steepest descent with a model step: it legitimately fails to reach
    gtol on Rosenbrock's curved valley within 200 iterations."""
    res = numopt.run("trust_region_cauchy", problems.get("rosenbrock"))
    assert_valid_result(res, max_iter=200)
    assert not res.converged and "max_iter" in res.message
    assert res.fun < problems.get("rosenbrock").f(np.array([-1.2, 1.0]))


@pytest.mark.parametrize("method", METHODS)
def test_max_iter(method: str) -> None:
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=3)
    assert_valid_result(res, max_iter=3)
    assert not res.converged and "max_iter" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4


def test_nonfinite_start() -> None:
    prob = dataclasses.replace(problems.get("rosenbrock"), f=lambda x: math.inf)
    res = numopt.run("trust_region_dogleg", prob)
    assert_valid_result(res)
    assert not res.converged and "not finite" in res.message and res.n_iter == 0


@pytest.mark.parametrize("method", METHODS)
def test_nonfinite_trial_is_rejected_and_radius_shrinks(method: str) -> None:
    """Himmelblau with f = +inf outside the disk ‖x‖ < 4.5. From the origin (where ∇²f ≺ 0)
    with Δ₀ = 5 the first step reaches ‖p‖ = 5 (the boundary of the region): it is rejected
    (ρ = −∞), Δ shrinks to 5/4, and the run then converges to (3, 2) inside the disk."""
    base = problems.get("himmelblau")

    def f(x: Any) -> float:
        return math.inf if float(np.linalg.norm(x)) >= 4.5 else float(base.f(x))

    prob = dataclasses.replace(base, f=f)
    res = numopt.run(method, prob, radius0=5.0)
    assert_valid_result(res)
    assert res.converged, res.message
    assert_allclose(res.x, [3.0, 2.0], rtol=0, atol=1e-7)
    first = res.trace[1]
    assert first.info["step_norm"] == pytest.approx(5.0, rel=1e-12)
    assert_allclose(first.x, [0.0, 0.0], rtol=0, atol=0)
    rejected = [s for s in res.trace[1:] if s.info["actual"] is None]
    assert rejected and rejected[0] is first
    assert all(s.info["rho"] is None and not s.info["accepted"] for s in rejected)
    assert all(s.info["new_radius"] == 0.25 * s.info["radius"] for s in rejected)
    _check_driver_trace(res, base, eta=0.15, max_radius=100.0)


def test_radius_collapse_is_reported() -> None:
    """f is +inf everywhere except x₀: every trial is rejected until Δ < ε·max(1, ‖x‖)."""
    base = problems.get("quadratic_bowl")
    x0 = np.array([-2.0, 2.0])

    def f(x: Any) -> float:
        return float(base.f(x)) if np.array_equal(x, x0) else math.inf

    prob = dataclasses.replace(base, f=f)
    res = numopt.run("trust_region_steihaug", prob)
    assert_valid_result(res, max_iter=200)
    assert not res.converged and "collapsed" in res.message
    assert res.extra["n_rejected"] == res.n_iter


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": 0.25},
        {"eta": -0.1},
        {"radius0": 0.0},
        {"radius0": math.inf},
        {"max_radius": 0.0},
        {"max_radius": math.inf},
        {"max_iter": 0},
        {"gtol": -1.0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        numopt.run("trust_region_exact", problems.get("rosenbrock"), **kwargs)


@pytest.mark.parametrize(("radius0", "max_radius"), [(1.0, 0.5), (50.0, 10.0)])
def test_radius0_above_max_radius_is_clamped(radius0: float, max_radius: float) -> None:
    """Regression: both values lie inside their ParamSpec ranges, so the UI can produce them;
    Alg. 4.1 needs Δ₀ ≤ Δ̂, and the driver clamps Δ₀ = Δ̂ (Step 0 says so) instead of raising."""
    specs = {p.name: p for p in numopt.core.registry.get_method("trust_region_dogleg").params}
    for name, value in (("radius0", radius0), ("max_radius", max_radius)):
        lo, hi = specs[name].min, specs[name].max
        assert lo is not None and hi is not None and lo <= value <= hi
    prob = problems.get("rosenbrock")
    res = numopt.run("trust_region_dogleg", prob, radius0=radius0, max_radius=max_radius)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    first = res.trace[0].info
    assert first["radius"] == first["new_radius"] == max_radius
    assert first["note"] is not None and "max_radius" in first["note"]
    assert all(s.info["radius"] <= max_radius for s in res.trace)
    _check_driver_trace(res, prob, eta=0.15, max_radius=max_radius)
    # The clamped run is the run started with radius0 = max_radius.
    same = numopt.run("trust_region_dogleg", prob, radius0=max_radius, max_radius=max_radius)
    assert [s.x.tolist() for s in same.trace] == [s.x.tolist() for s in res.trace]
    assert same.trace[0].info["note"] is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", ["matyas", "quadratic_ill"])
def test_gtol_zero_stops_at_the_rounding_level_of_the_model(method: str, pid: str) -> None:
    """Regression: with gtol = 0 the iterates reach ‖g‖ ≈ 1e-160, where gᵀg and gᵀBg underflow.
    Before, the dogleg raised ZeroDivisionError, the exact solver emitted a RuntimeWarning and
    reported a false radius collapse, and Cauchy/Steihaug returned p = 0 (‖g‖₂ computed as 0)
    with ρ = 1 and idled until max_iter. Now each run ends cleanly when the model's predicted
    reduction m(0) − m(p) underflows: converged=False, iterate at the float floor."""
    prob = problems.get(pid)
    assert prob.grad is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run(method, prob, gtol=0.0, max_iter=2000)
    assert_valid_result(res, max_iter=2000)
    assert res.n_iter < 2000
    ginf = float(np.max(np.abs(prob.grad(res.x))))
    if res.converged:  # the exact solver lands on ∇f = 0 exactly on quadratic_ill: gtol = 0 met
        assert ginf == 0.0
    else:
        assert "predicts no decrease" in res.message, res.message
        assert 0.0 < ginf < 1e-150
    assert res.n_fev == 1 + res.n_iter  # the stop happens before f is evaluated at a trial
    _check_driver_trace(res, prob, eta=0.15, max_radius=100.0)


def _scaled(prob: Problem, c: float) -> Problem:
    """c·f with exact derivatives c·∇f, c·∇²f."""
    assert prob.grad is not None and prob.hess is not None
    g, h = prob.grad, prob.hess
    return dataclasses.replace(
        prob,
        f=lambda x: c * float(prob.f(x)),
        grad=lambda x: c * np.asarray(g(x)),
        hess=lambda x: c * np.asarray(h(x)),
    )


@pytest.mark.parametrize("method", METHODS)
def test_tiny_scale_quadratic_is_solved(method: str) -> None:
    """f = c‖x‖², c = 2⁻⁵⁶⁶ ≈ 1.6e-170, x₀ = (3, −1): ‖g₀‖₂² underflows but f, g, B and the step
    are representable. Before, the dogleg raised ZeroDivisionError, the exact solver warned, and
    Cauchy/Steihaug never moved (200 iterations at x₀). The scaled solvers move to x* = 0."""
    c = 2.0**-566
    prob = Problem(
        id="tiny",
        name="tiny",
        latex="",
        f=lambda x: c * float(x @ x),
        dim=2,
        domain=(),
        grad=lambda x: 2.0 * c * x,
        hess=lambda x: 2.0 * c * np.eye(2),
        x0=[3.0, -1.0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run(method, prob, gtol=0.0)
    assert_valid_result(res, max_iter=200)
    assert res.trace[0].grad_norm == pytest.approx(2.0 * c * math.sqrt(10.0), rel=2 * EPS)
    assert res.n_iter <= 10
    if method == "trust_region_dogleg":
        # Cholesky of 2c·I and the back-substitution leave x ≈ 4e-81, where gᵀp underflows.
        assert not res.converged and "predicts no decrease" in res.message
        assert float(np.max(np.abs(res.x))) < 1e-75
    else:
        assert res.converged and np.array_equal(res.x, [0.0, 0.0]), res.message
    _check_driver_trace(res, prob, eta=0.15, max_radius=100.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("c", [1e-20, 2.0**-66, 1e-16])
def test_accepted_steps_never_increase_f_beyond_rounding(method: str, c: float) -> None:
    """Regression for the absolute rounding floor δ = 10ε·max(1, |f|): on 1e-20·Rosenbrock it
    dominated both reductions for the whole run, so ρ ≈ 1 and steps that multiplied f by 38–47
    were accepted (ρ reported 0.999 where actual/predicted = −62). With δ relative to |f|, an
    accepted step raises f by less than 10³ε·|f|, and the run converges like the unscaled one."""
    base = problems.get("rosenbrock")
    prob = _scaled(base, c)
    res = numopt.run(method, prob, gtol=c * 1e-8, max_iter=200)
    assert_valid_result(res, max_iter=200)
    _check_driver_trace(res, prob, eta=0.15, max_radius=100.0)
    for prev, cur in itertools.pairwise(res.trace):
        info = cur.info
        f_prev = float(prev.fun) if prev.fun is not None else math.nan
        if info["accepted"] and info["predicted"] > 1e3 * ROUNDING * EPS * abs(f_prev):
            # Above the rounding level ρ is the textbook ratio, to 1e-3 relative.
            assert info["rho"] == pytest.approx(info["actual"] / info["predicted"], rel=1e-3)
    ref = numopt.run(method, base, max_iter=200)
    assert res.converged == ref.converged
    if method != "trust_region_steihaug":  # Steihaug's forcing term √‖g‖ is not scale invariant
        assert res.n_iter == ref.n_iter


@settings(max_examples=1000, deadline=None)
@given(
    e=st.integers(-200, 200),
    method=st.sampled_from(["trust_region_cauchy", "trust_region_dogleg", "trust_region_exact"]),
    pid=st.sampled_from(["rosenbrock", "himmelblau", "beale", "three_hump_camel", "booth"]),
    max_iter=st.integers(1, 60),
)
def test_iterates_are_invariant_under_power_of_two_scaling(
    e: int, method: str, pid: str, max_iter: int
) -> None:
    """Scale invariance (module docstring): for f → 2ᵉf with gtol → 2ᵉgtol the Cauchy, dogleg and
    exact iterations are the same, step for step. Fails for any absolute constant in the radius
    logic (the old δ = 10ε·max(1, |f|)) or in the solvers (underflowing gᵀBg, γ²)."""
    base = problems.get(pid)
    c = math.ldexp(1.0, e)
    ref = numopt.run(method, base, max_iter=max_iter)
    res = numopt.run(method, _scaled(base, c), gtol=c * 1e-8, max_iter=max_iter)
    assert res.n_iter == ref.n_iter and res.converged == ref.converged
    assert [s.info["accepted"] for s in res.trace] == [s.info["accepted"] for s in ref.trace]
    # NOTE: the iterates agree to rounding, not bit for bit: Cholesky (dogleg) takes √(2ᵉb), which
    # is not 2^(e/2)√b in floating point for odd e, and LAPACK's eigh (exact) rescales tiny or
    # huge matrices by non-powers of 2 (κ-amplified over ≤ 60 steps; measured ≤ 1e-13).
    X, X_ref = np.array([s.x for s in res.trace]), np.array([s.x for s in ref.trace])
    assert_allclose(X, X_ref, rtol=0, atol=1e-11)
    # f is compared relative to f(x₀): at a zero-residual minimizer f(x_k) is rounding noise.
    F, F_ref = np.array([s.fun for s in res.trace]) / c, np.array([s.fun for s in ref.trace])
    assert_allclose(F, F_ref, rtol=1e-10, atol=1e-12 * abs(F_ref[0]))


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(1, 6),
    seed=st.integers(0, 2**31 - 1),
    log_b=st.floats(-2.0, 2.0),
    log_delta=st.floats(-2.0, 2.0),
    e=st.sampled_from([-600, -520, -300, 0, 300]),
    solver=st.sampled_from(sorted(SOLVERS)),
)
def test_solvers_at_extreme_scales(
    n: int, seed: int, log_b: float, log_delta: float, e: int, solver: str
) -> None:
    """Every solver applied to (cg, cB), c = 2ᵉ, returns a feasible step with the Cauchy decrease
    of the unscaled model (N&W Lemma 4.3), without a warning; at e = −600 gᵀg and gᵀBg underflow
    (before: Cauchy/Steihaug p = 0, dogleg ZeroDivisionError, exact RuntimeWarning). Cauchy,
    dogleg and exact solve a scale-invariant problem and return the unscaled step."""
    g, B, delta = _random_instance(n, seed, scale_b=10.0**log_b, scale_g=1.0, delta=10.0**log_delta)
    c = math.ldexp(1.0, e)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sub = SOLVERS[solver](c * g, c * B, delta)
    p = sub.p
    assert np.all(np.isfinite(p))
    assert float(np.linalg.norm(p)) <= delta * (1.0 + 1e-12)
    gn = float(np.linalg.norm(g))
    bn = float(np.linalg.norm(B, 2))
    bound = 0.5 * gn * min(delta, gn / bn if bn > 0.0 else math.inf)
    assert -_model(g, B, p) >= bound * (1.0 - 1e-12)  # m_c = c·m: decrease of the unscaled model
    if solver != "steihaug":  # Steihaug's ε_k = min(½, √‖g‖)‖g‖ is not homogeneous in g
        # NOTE: exact equality for Cauchy and dogleg; LAPACK's eigh rescales tiny matrices by a
        # non-power of 2, so the exact step agrees to rounding (measured ≤ 3e-14·Δ).
        tol = 0.0 if solver != "exact" else 1e-11 * delta
        assert_allclose(p, SOLVERS[solver](g, B, delta).p, rtol=0, atol=tol)


def test_dogleg_falls_back_when_curvature_rounds_to_nonpositive() -> None:
    """B = vvᵀ + 1e-16·I passes Cholesky, but with g ⊥ v (to rounding) the computed gᵀBg ≤ 0, so
    p^U = −(gᵀg/gᵀBg)g is undefined; the dogleg takes the Cauchy point and says why."""
    v = np.array([0.7604508913778566, -0.480369070543063])
    B = np.array(
        [[0.5782855581973767, -0.36529708788482473], [-0.36529708788482473, 0.23075444393440622]]
    )
    g = np.array([0.48036907141779395, 0.760450891761848])
    assert tr._newton_step(g, B) is not None  # Cholesky succeeds
    gh, _ = tr._pow2_scaled(g)
    assert float(gh @ (B @ gh)) <= 0.0
    assert abs(float(g @ v)) < 1e-8
    sub = tr._dogleg(g, B, 1.0)
    assert sub.note is not None and "floating point" in sub.note
    assert sub.info == {"dogleg_path": None, "tau": None}
    assert_allclose(sub.p, tr._cauchy(g, B, 1.0).p, rtol=0, atol=0)


def test_exact_solver_cauchy_fallback_when_secular_equation_is_unrepresentable() -> None:
    """Defensive branch: when even p(hi) is not finite (hi = ‖g‖/Δ underflowed to 0, the pole of
    p(μ)), the solver returns the Cauchy point with λ = NaN (null in JSON) and a note, without a
    RuntimeWarning. Before, p(0) was formed outside np.errstate (a RuntimeWarning) and the NaN
    step made the driver report a false radius collapse."""
    g = np.array([5e-324, 5e-324])  # ‖g‖/Δ = 7e-324/10 underflows to 0
    B = np.diag([-1e-315, 1e-315])  # λ₁ < 0; γ₁ is not negligible against ‖B‖Δ: no hard case
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sub = tr._exact(g, B, 10.0)
    assert sub.note is not None and "Cauchy point" in sub.note
    assert math.isnan(sub.info["lambda"])
    assert_allclose(sub.p, tr._cauchy(g, B, 10.0).p, rtol=0, atol=0)
    assert float(np.linalg.norm(sub.p)) <= 10.0


@settings(max_examples=1000, deadline=None)
@given(
    v=arrays(np.float64, st.integers(1, 8), elements=st.floats(-1.0, 1.0)),
    e=st.integers(-1070, 1020),
)
def test_scaled_norm(v: np.ndarray, e: int) -> None:
    """_norm2 equals math.hypot (which scales) to a few ulp over the whole float range, and equals
    np.linalg.norm bit for bit whenever no square under- or overflows."""
    with np.errstate(under="ignore", over="ignore"):
        w = np.ldexp(v, e)
    if not np.all(np.isfinite(w)):
        return
    ref = math.hypot(*w)
    got = tr._norm2(w)
    assert got == pytest.approx(ref, rel=4 * EPS, abs=0.0) or (ref == 0.0 and got == 0.0)
    nz = np.abs(w[w != 0.0])
    if nz.size and float(nz.min()) >= 1.5e-154 and float(nz.max()) <= 1e153:
        assert got == float(np.linalg.norm(w))


def test_fixture_cases_are_valid() -> None:
    for method, pid, params in tr.FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert res.converged, (method, pid, res.message)
        assert len(res.trace) < 300
    assert {m for m, _, _ in tr.FIXTURE_CASES} == set(METHODS)


# --------------------------------------------------------------------------------------
# Regression: dim = 1 Problems follow the scalar convention (floats in, floats out)
# --------------------------------------------------------------------------------------


def _scalar_only_problem(a: float, b: float, x0: float) -> Problem:
    """f(x) = a(x − b)² + 1 with callables that accept only floats (the dim = 1 convention)."""

    def f(x: float) -> float:
        assert isinstance(x, float), type(x)
        return a * (x - b) ** 2 + 1.0

    def grad(x: float) -> float:
        assert isinstance(x, float), type(x)
        return 2.0 * a * (x - b)

    def hess(x: float) -> float:
        assert isinstance(x, float), type(x)
        return 2.0 * a

    return Problem(
        id="q1", name="q1", latex="", f=f, grad=grad, hess=hess, dim=1, domain=(-10, 10), x0=x0
    )


@pytest.mark.parametrize("method", METHODS)
def test_library_problem_with_dim_1(method: str) -> None:
    """Regression: problems.get("quadratic_1d") (dim = 1, f = (x − 2)² + 1) raised TypeError
    ("only 0-dimensional arrays can be converted to Python scalars"): the driver passed an
    array of shape (1,) to callables that follow the float convention of core.types."""
    prob = problems.get("quadratic_1d")
    res = numopt.run(method, prob, gtol=1e-10)
    assert_valid_result(res)
    assert res.converged, res.message
    assert np.shape(res.x) == (1,)
    # Oracle: the closed-form minimizer x* = 2, f* = 1; |x − x*| = |f′(x)|/2 ≤ gtol/2.
    assert_allclose(res.x, [2.0], rtol=0, atol=0.5e-10)
    assert res.fun == pytest.approx(1.0, rel=0, abs=1e-15)
    assert all(len(s.info["center"]) == 1 and s.info["H"] is None for s in res.trace)  # n ≠ 2


@pytest.mark.parametrize("method", METHODS)
def test_scalar_only_callables_with_dim_1(method: str) -> None:
    """f(x) = cos x + x²/10 written with math.* (rejects arrays); from x₀ = 1 the model steps
    reach the local minimizer x* ≈ 2.5957 where f′(x) = −sin x + x/5 = 0 (oracle: SciPy
    brentq). f″(x*) ≈ 1.05, so |x − x*| ≲ gtol/1.05."""
    from scipy.optimize import brentq

    prob = Problem(
        id="cos",
        name="cos",
        latex="",
        f=lambda x: math.cos(x) + 0.1 * x * x,
        grad=lambda x: -math.sin(x) + 0.2 * x,
        hess=lambda x: -math.cos(x) + 0.2,
        dim=1,
        domain=(-5.0, 5.0),
        x0=1.0,
    )
    res = numopt.run(method, prob, gtol=1e-9)
    assert_valid_result(res)
    assert res.converged, res.message
    x_star = cast(float, brentq(lambda x: -math.sin(x) + 0.2 * x, 2.0, 3.0, xtol=1e-15))
    assert_allclose(res.x, [x_star], rtol=0, atol=1e-9)


@settings(max_examples=1000, deadline=None)
@given(
    a=st.floats(1e-2, 1e2),
    b=st.floats(-10.0, 10.0),
    x0=st.floats(-10.0, 10.0),
    method=st.sampled_from(METHODS),
)
def test_dim_1_quadratic_property(a: float, b: float, x0: float, method: str) -> None:
    """On f(x) = a(x − b)² + 1 (dim = 1, float-only callables) every method converges, and
    strong convexity gives |x − b| = |f′(x)|/(2a) ≤ gtol/(2a). The quadratic model is exact, so
    every step with ρ computed is accepted and f never increases along the trace."""
    gtol = 1e-8
    res = numopt.run(method, _scalar_only_problem(a, b, x0), gtol=gtol)
    assert res.converged, res.message
    assert abs(float(res.x[0]) - b) <= gtol / (2.0 * a)
    funs = [s.fun for s in res.trace if s.fun is not None]
    assert all(f1 <= f0 for f0, f1 in itertools.pairwise(funs))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("e", [496, 512, 600])
def test_huge_objective_scale_never_raises(method: str, e: int) -> None:
    """The CG driver raised at these scales (see the CG tests); the trust-region driver must
    also end with a Result: converged=True only when ‖∇f‖∞ ≤ gtol holds."""
    base = problems.get("rosenbrock")
    f_fn, g_fn, h_fn = base.f, base.grad, base.hess
    assert g_fn is not None and h_fn is not None
    c = 2.0**e
    prob = dataclasses.replace(
        base,
        f=lambda x: c * float(f_fn(x)),
        grad=lambda x: c * np.asarray(g_fn(x)),
        hess=lambda x: c * np.asarray(h_fn(x)),
    )
    res = numopt.run(method, prob)
    assert_valid_result(res, max_iter=200)
    assert prob.grad is not None
    if res.converged:
        assert float(np.max(np.abs(np.asarray(prob.grad(res.x))))) <= 1e-5
