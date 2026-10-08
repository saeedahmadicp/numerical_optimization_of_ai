"""Tests for method.py: AdaBelief, Lion, Adan, Sophia and the rotation wrapper.

Oracles used (none is a rewrite of the code under test):
* hand-computed first (and second) steps from the papers' update formulas;
* reductions to independent implementations: AdaBelief(β₁ = 0) is numopt's fixed-step
  gradient descent; Adan(β₁ = β₂ = β₃ = 1) and Lion(β₁ = 0) are both sign descent;
  Sophia(γ → 0) and Lion(β₁ = β₂) are both signum (sign of the gradient EMA);
* one-step Newton of Sophia (k = 1, β₁ = β₂ = 0, lr = γ) on an axis-aligned quadratic;
* invariants: signed-permutation equivariance (Hypothesis, 1000 examples), |Δx| = η for
  Lion, |Δx| ≤ lr for Sophia, unbiased Hutchinson estimates, the rotated problem's
  derivatives against central differences and its unchanged Hessian spectrum.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from method import (
    METHODS,
    PARAMS,
    adabelief,
    adan,
    hessian_11_norm,
    lion,
    rotate,
    rotation_2d,
    sophia,
)
from numpy.testing import assert_allclose, assert_array_equal

from numopt import problems
from numopt.core.types import Problem, Result
from numopt.unconstrained.first_order import gradient_descent

QUAD = problems.get("quadratic_ill")
ROSEN = problems.get("rosenbrock")
PHI0 = math.atan2(0.6, 0.8)  # quadratic_ill = Q diag(1, 50) Qᵀ with Q = R(PHI0)


def assert_valid_result(res: Result, *, max_iter: int | None = None) -> None:
    """The contract checks of tests/conftest.py (copied: this folder is outside tests/)."""
    assert res.trace, "trace must not be empty"
    assert res.trace[0].k == 0, "trace must start at k = 0"
    ks = [s.k for s in res.trace]
    assert ks == sorted(ks), "trace indices must increase"
    if max_iter is not None:
        assert res.n_iter <= max_iter
        assert len(res.trace) <= max_iter + 1
    assert res.n_iter >= 0
    assert isinstance(res.converged, bool)
    assert res.message
    json.dumps(res.to_dict(), allow_nan=False)


def grad(P: Problem, x: Any) -> np.ndarray:
    assert P.grad is not None
    return np.asarray(P.grad(np.asarray(x)))


def hess(P: Problem, x: Any) -> np.ndarray:
    assert P.hess is not None
    return np.asarray(P.hess(np.asarray(x)))


def xs(res: Result) -> np.ndarray:
    return np.array([s.x for s in res.trace])  # (K + 1, n)


# Configurations that converge (‖∇f‖ ≤ 1e-6) on both test problems within 3000 iterations.
CONFIGS: dict[str, tuple[Callable[..., Result], dict[str, Any]]] = {
    "adabelief": (adabelief, {"lr": 0.3}),
    # A sign method meets a tight gtol only if the schedule is matched to the problem: with
    # lr_decay = 0.995 Lion freezes at ‖∇f‖ ≈ 9e-6 on rosenbrock (total travel left ≈ η/(1 − ρ)).
    "lion": (lion, {"lr": 0.03, "lr_decay": 0.996}),
    "adan": (adan, {"lr": 0.1}),
    "sophia": (sophia, {"lr": 0.03}),
    "sophia_h": (sophia, {"lr": 0.03, "estimator": "hutchinson", "seed": 3}),
}
INFO_KEYS = {
    "adabelief": {"m", "s", "m_hat", "s_hat", "lr_eff"},
    "lion": {"m", "c", "decay"},
    "adan": {"m", "v", "n", "lr_eff"},
    "sophia": {"m", "h", "h_new", "ratio", "clipped"},
    "sophia_h": {"m", "h", "h_new", "ratio", "clipped"},
}


# --------------------------------------------------------------------------------------
# Contract, counts and convergence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("problem", [QUAD, ROSEN], ids=["quadratic_ill", "rosenbrock"])
def test_contract_counts_and_convergence(name: str, problem: Problem) -> None:
    fn, kw = CONFIGS[name]
    res = fn(problem, gtol=1e-6, max_iter=3000, **kw)
    assert_valid_result(res, max_iter=3000)
    assert res.converged, res.message
    assert res.n_iter == res.trace[-1].k == len(res.trace) - 1
    assert np.linalg.norm(grad(problem, res.x)) <= 1e-6
    assert res.fun == pytest.approx(0.0, abs=1e-10)
    # One f and one ∇f per iteration plus the start; Sophia: one ∇²f per refresh (k = 10).
    assert res.n_fev == res.n_gev == res.n_iter + 1
    expected_hev = math.ceil(res.n_iter / 10) if name.startswith("sophia") else 0
    assert res.n_hev == expected_hev
    for prev, step in zip(res.trace, res.trace[1:], strict=False):
        info = step.info
        assert INFO_KEYS[name] | {"grad", "direction", "alpha"} <= set(info)
        assert info["alpha"] == step.step_size
        assert_allclose(
            np.asarray(step.x) - np.asarray(prev.x),
            info["alpha"] * np.asarray(info["direction"]),
            rtol=1e-12,
            atol=1e-15,
        )
        assert_allclose(info["grad"], grad(problem, step.x), rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_max_iter_is_reported_as_not_converged(name: str) -> None:
    fn, kw = CONFIGS[name]
    res = fn(ROSEN, max_iter=5, **kw)
    assert_valid_result(res, max_iter=5)
    assert not res.converged
    assert res.n_iter == 5
    assert "max_iter" in res.message


@pytest.mark.parametrize("name", list(CONFIGS))
def test_start_at_the_minimizer_converges_at_once(name: str) -> None:
    fn, kw = CONFIGS[name]
    res = fn(ROSEN, x0=[1.0, 1.0], **kw)
    assert res.converged and res.n_iter == 0 and len(res.trace) == 1


@pytest.mark.parametrize("name", list(CONFIGS))
def test_divergence_to_a_non_finite_value_is_detected(name: str) -> None:
    # f = √x₀ + x₁² decreases toward x₀ = 0 and is NaN beyond it. Every method's first step in
    # x₀ is at least lr = 0.1 (a sign-like step: AdaBelief lr/β₁, Adan and Lion lr, Sophia lr
    # because ∂²f/∂x₀² < 0 there), so from x₀ = 0.05 the iterate leaves the domain.
    sqrt_problem = Problem(
        id="sqrt",
        name="sqrt",
        latex="",
        f=lambda x: math.sqrt(x[0]) + x[1] ** 2 if x[0] >= 0 else math.nan,
        dim=2,
        domain=(),
        grad=lambda x: np.array([0.5 / math.sqrt(x[0]) if x[0] > 0 else math.nan, 2 * x[1]]),
        hess=lambda x: np.array([[-0.25 * x[0] ** -1.5 if x[0] > 0 else math.nan, 0], [0, 2]]),
    )
    fn, kw = CONFIGS[name]
    res = fn(sqrt_problem, x0=[0.05, 0.3], **(kw | {"lr": 0.1}))
    assert_valid_result(res)
    assert not res.converged
    assert res.message.startswith("diverged: f(x) or ∇f(x) is not finite")


@pytest.mark.parametrize("name", list(CONFIGS))
def test_non_finite_start(name: str) -> None:
    fn, kw = CONFIGS[name]
    res = fn(lambda x: math.nan, x0=[1.0, 2.0], **kw)
    assert not res.converged and "not finite" in res.message and res.n_iter == 0


def test_bare_callable_uses_finite_differences() -> None:
    f = lambda x: (x[0] - 1.0) ** 2 + 4.0 * (x[1] + 0.5) ** 2  # noqa: E731
    res = adan(f, x0=[3.0, 1.0], lr=0.1, max_iter=3000, gtol=1e-6)
    assert res.converged, res.message
    assert_allclose(res.x, [1.0, -0.5], atol=1e-6)
    assert res.n_gev == res.n_iter + 1
    assert res.n_fev == (res.n_iter + 1) * (1 + 2 * 2)  # f + central differences (2n)
    res = sophia(f, x0=[3.0, 1.0], lr=0.03, max_iter=3000, gtol=1e-6)
    assert res.converged and res.n_hev == math.ceil(res.n_iter / 10)


@pytest.mark.parametrize(
    ("fn", "kw"),
    [
        (adabelief, {"lr": 0.0}),
        (adabelief, {"beta1": 1.0}),
        (adabelief, {"eps": 0.0}),
        (lion, {"lr_decay": 1.5}),
        (lion, {"lr_decay": 0.0}),
        (lion, {"weight_decay": -1.0}),
        (adan, {"beta3": 1.5}),
        (sophia, {"k": 0}),
        (sophia, {"estimator": "gnb"}),
        (sophia, {"gamma": 0.0}),
        (sophia, {"max_iter": 0}),
        (sophia, {"gtol": math.nan}),
    ],
)
def test_invalid_input_raises(fn: Callable[..., Result], kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fn(QUAD, **kw)


def test_params_dict_matches_signatures() -> None:
    import inspect

    for name, fn in METHODS.items():
        sig = inspect.signature(fn)
        declared = {p.name for p in PARAMS[name]}
        keywords = set(sig.parameters) - {"problem", "x0", "seed"}
        assert declared == keywords, name
        for p in PARAMS[name]:
            assert sig.parameters[p.name].default == p.default, (name, p.name)


# --------------------------------------------------------------------------------------
# Hand-computed steps (the papers' formulas evaluated by hand)
# --------------------------------------------------------------------------------------

X0 = np.array([-2.0, 2.0])
G0 = QUAD.grad(X0)  # A x0 = [-84.32, 111.76]


def test_adabelief_first_step() -> None:
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    res = adabelief(QUAD, x0=X0, lr=lr, max_iter=1)
    # m₁ = (1 − β₁)g, s₁ = (1 − β₂)(g − m₁)² + ε = (1 − β₂)β₁²g² + ε, m̂₁ = g,
    # ŝ₁ = β₁²g² + ε/(1 − β₂); x₁ = x₀ − lr·g/(√ŝ₁ + ε) ≈ x₀ − (lr/β₁)·sign(g).
    s_hat = b1**2 * G0**2 + eps / (1 - b2)
    assert_allclose(res.trace[1].x, X0 - lr * G0 / (np.sqrt(s_hat) + eps), rtol=1e-14)
    assert_allclose(res.trace[1].info["s"], (1 - b2) * b1**2 * G0**2 + eps, rtol=1e-14)


def test_lion_first_two_steps() -> None:
    lr, decay, b1, b2 = 0.1, 0.5, 0.9, 0.99
    res = lion(QUAD, x0=X0, lr=lr, lr_decay=decay, weight_decay=0.2, max_iter=2)
    # t = 1: c₁ = (1 − β₁)g₀; x₁ = x₀ − lr(sign(g₀) + λx₀); m₁ = (1 − β₂)g₀.
    x1 = X0 - lr * (np.sign(G0) + 0.2 * X0)
    assert_allclose(res.trace[1].x, x1, rtol=1e-15)
    m1 = (1 - b2) * G0
    # t = 2: η₂ = lr·decay; c₂ = β₁m₁ + (1 − β₁)g₁.
    g1 = QUAD.grad(x1)
    c2 = b1 * m1 + (1 - b1) * g1
    x2 = x1 - lr * decay * (np.sign(c2) + 0.2 * x1)
    assert_allclose(res.trace[2].x, x2, rtol=1e-15)
    assert res.trace[2].step_size == lr * decay
    assert_allclose(res.trace[2].info["m"], b2 * m1 + (1 - b2) * g1, rtol=1e-15)


def test_adan_first_two_steps() -> None:
    lr, b1, b2, b3, eps = 0.05, 0.02, 0.08, 0.01, 1e-8
    res = adan(QUAD, x0=X0, lr=lr, max_iter=2)
    # j = 0: m₀ = g₀, v₀ = 0, n₀ = g₀²; x₁ = x₀ − lr·g₀/(|g₀| + ε).
    x1 = X0 - lr * G0 / (np.abs(G0) + eps)
    assert_allclose(res.trace[1].x, x1, rtol=1e-15)
    # j = 1: m₁ = (1 − β₁)g₀ + β₁g₁, v₁ = g₁ − g₀ (the paper's initialisation),
    # n₁ = (1 − β₃)g₀² + β₃(g₁ + (1 − β₂)(g₁ − g₀))².
    g1 = QUAD.grad(x1)
    m1 = (1 - b1) * G0 + b1 * g1
    v1 = g1 - G0
    n1 = (1 - b3) * G0**2 + b3 * (g1 + (1 - b2) * (g1 - G0)) ** 2
    x2 = x1 - lr * (m1 + (1 - b2) * v1) / (np.sqrt(n1) + eps)
    assert_allclose(res.trace[2].x, x2, rtol=1e-14)
    assert_allclose(res.trace[0].info["n"], G0**2, rtol=1e-15)


def test_sophia_first_step_exact_hessian() -> None:
    lr, b1, b2, gamma = 0.01, 0.96, 0.99, 0.01
    # Start where g = (0.02, 5): ratio = (1 − β₁)g/(γ(1 − β₂)H_ii) = 400·g/H_ii is ≈ 0.43 in
    # coordinate 0 (not clipped) and ≈ 62 in coordinate 1 (clipped).
    x0 = np.linalg.solve(np.asarray(QUAD.extra["A"]), [0.02, 5.0])
    res = sophia(QUAD, x0=x0, lr=lr, max_iter=1)
    g = QUAD.grad(x0)
    h1 = (1 - b2) * np.diag(QUAD.hess(x0))  # h₁ = β₂·0 + (1 − β₂)·diag(∇²f)
    ratio = (1 - b1) * g / np.maximum(gamma * h1, 1e-12)
    assert_allclose(res.trace[1].x, x0 - lr * np.clip(ratio, -1, 1), rtol=1e-14)
    assert_allclose(res.trace[1].info["ratio"], ratio, rtol=1e-14)
    assert res.trace[1].info["clipped"] == [bool(c) for c in np.abs(ratio) > 1]
    assert res.trace[1].info["clipped"] == [False, True]  # both regimes are exercised


def test_sophia_is_one_newton_step_on_an_aligned_quadratic() -> None:
    # Rotating quadratic_ill by PHI0 aligns it: g(x) = ½xᵀdiag(1, 50)x.
    aligned = rotate(QUAD, rotation_2d(PHI0))
    assert_allclose(aligned.extra["A"], np.diag([1.0, 50.0]), atol=1e-13)
    x0 = np.array([0.7, -0.4])
    kw = {"lr": 1.0, "gamma": 1.0, "beta1": 0.0, "beta2": 0.0, "k": 1, "max_iter": 5}
    res = sophia(aligned, x0=x0, **kw)
    # m = g, h = diag(∇²g) = ∇²g, ratio = H⁻¹g = x (|x| ≤ 1: no clipping) → x₁ = 0. The rotated
    # A is diag(1, 50) only to rounding (its off-diagonal is ~1e-14, CPU-dependent), and
    # x₁ = −D⁻¹(A − D)x₀, so |x₁| ≤ ‖A − diag(1, 50)‖·‖x₀‖ ≤ 1e-13 (asserted above).
    assert_allclose(res.trace[1].x, [0.0, 0.0], atol=1e-13)
    assert res.converged and res.n_iter == 1
    # Misaligned by 45°: the same step is the Jacobi step x₁ = x₀ − D⁻¹Ax₀, not Newton's.
    tilted = rotate(aligned, rotation_2d(math.pi / 4))
    A = np.asarray(tilted.extra["A"])
    res45 = sophia(tilted, x0=x0, **kw)
    assert_allclose(res45.trace[1].x, x0 - (A @ x0) / np.diag(A), rtol=1e-13)
    assert np.linalg.norm(res45.trace[1].x) > 0.1


# --------------------------------------------------------------------------------------
# Reductions to independent implementations (equivalence oracles)
# --------------------------------------------------------------------------------------


def test_adabelief_without_momentum_is_gradient_descent() -> None:
    # β₁ = 0: m = g, g − m = 0, so ŝ_t = ε/(1 − β₂) for all t, and AdaBelief is fixed-step
    # gradient descent with step lr/(√(ε/(1 − β₂)) + ε) (oracle: numopt's implementation).
    lr, b2, eps = 1e-4, 0.999, 1e-8
    step = lr / (math.sqrt(eps / (1 - b2)) + eps)
    ab = adabelief(QUAD, lr=lr, beta1=0.0, beta2=b2, eps=eps, max_iter=300, gtol=1e-150)
    gd = gradient_descent(QUAD, step_rule="fixed", lr=step, max_iter=300, gtol=1e-150)
    assert ab.n_iter == gd.n_iter == 300
    # NOTE: rtol 1e-12, not 1e-15: ŝ_t = s_t/(1 − β₂ᵗ) is recomputed each step and differs
    # from ε/(1 − β₂) by a few ulps; the contraction keeps the accumulated error ~t·ε_mach.
    assert_allclose(xs(ab), xs(gd), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("problem", [QUAD, ROSEN], ids=["quadratic_ill", "rosenbrock"])
def test_adan_with_unit_betas_is_lion_without_momentum(problem: Problem) -> None:
    # Adan(β₁ = β₂ = β₃ = 1): m = g, (1 − β₂)v = 0, n = g², step −lr·g/|g| = −lr·sign(g).
    # Lion(β₁ = 0): c = g, step −lr·sign(g). Both are sign descent.
    a = adan(problem, lr=0.01, beta1=1.0, beta2=1.0, beta3=1.0, eps=1e-300, max_iter=200)
    b = lion(problem, lr=0.01, beta1=0.0, max_iter=200)
    assert_array_equal(xs(a), xs(b))


@pytest.mark.parametrize("problem", [QUAD, ROSEN], ids=["quadratic_ill", "rosenbrock"])
def test_sophia_with_tiny_gamma_is_lion_with_equal_betas(problem: Problem) -> None:
    # γ → 0: every coordinate is clipped, the step is −lr·sign(m_t) with m_t the EMA of g.
    # Lion(β₁ = β₂ = β): c_t = βm_{t−1} + (1 − β)g_t = m_t. Both are signum.
    beta = 0.9
    a = sophia(problem, lr=0.01, beta1=beta, gamma=1e-30, eps=1e-300, max_iter=200)
    b = lion(problem, lr=0.01, beta1=beta, beta2=beta, max_iter=200)
    assert all(all(s.info["clipped"]) for s in a.trace[1:])
    assert_array_equal(xs(a), xs(b))


# --------------------------------------------------------------------------------------
# Properties (Hypothesis)
# --------------------------------------------------------------------------------------

SIGNED_PERMUTATIONS = [
    np.array(m, dtype=float)
    for m in (
        [[1, 0], [0, 1]],
        [[-1, 0], [0, 1]],
        [[1, 0], [0, -1]],
        [[-1, 0], [0, -1]],
        [[0, 1], [1, 0]],
        [[0, -1], [1, 0]],
        [[0, 1], [-1, 0]],
        [[0, -1], [-1, 0]],
    )
]
coord = st.floats(-2.0, 2.0, allow_nan=False)


@settings(max_examples=1000, deadline=None)
@given(
    s=st.integers(0, 7),
    theta=st.floats(0.0, 2 * math.pi),
    x0=st.tuples(coord, coord),
    name=st.sampled_from(["adabelief", "lion", "adan", "sophia"]),
)
def test_coordinatewise_methods_are_signed_permutation_equivariant(
    s: int, theta: float, x0: tuple[float, float], name: str
) -> None:
    # Theorem 2.2 of Xie et al. (2025) for these methods: with S a signed permutation,
    # running on f(Sx) from Sᵀx₀ gives the iterates Sᵀx_k exactly.
    base = rotate(QUAD, rotation_2d(theta))  # a random orientation of the problem
    S = SIGNED_PERMUTATIONS[s]
    fn, kw = CONFIGS[name]
    r1 = fn(base, x0=np.array(x0), max_iter=40, **kw)
    r2 = fn(rotate(base, S), x0=S.T @ np.array(x0), max_iter=40, **kw)
    assert r1.n_iter == r2.n_iter
    assert_allclose(xs(r2), xs(r1) @ S, rtol=1e-13, atol=1e-13)  # rows: (Sᵀx)ᵀ = xᵀS


@pytest.mark.parametrize("name", list(CONFIGS))
def test_coordinatewise_methods_are_not_rotation_equivariant(name: str) -> None:
    R = rotation_2d(math.pi / 8)
    fn, kw = CONFIGS[name]
    r1 = fn(QUAD, max_iter=60, **kw)
    r2 = fn(rotate(QUAD, R), x0=R.T @ np.asarray(QUAD.x0), max_iter=60, **kw)
    assert np.max(np.abs(xs(r2) - xs(r1) @ R)) > 1e-3


def test_gradient_descent_is_rotation_equivariant_on_rotated_problems() -> None:
    # The control of the study (and a check of rotate): GD on f(Rx) from Rᵀx₀ follows Rᵀx_k.
    for theta in np.linspace(0.0, math.pi / 2, 7):
        R = rotation_2d(theta)
        r1 = gradient_descent(ROSEN, step_rule="fixed", lr=1e-3, max_iter=2000)
        r2 = gradient_descent(
            rotate(ROSEN, R),
            x0=R.T @ np.asarray(ROSEN.x0),
            step_rule="fixed",
            lr=1e-3,
            max_iter=2000,
        )
        assert_allclose(xs(r2), xs(r1) @ R, rtol=1e-9, atol=1e-11)


@settings(max_examples=1000, deadline=None)
@given(
    lr=st.floats(1e-4, 1.0),
    decay=st.floats(0.9, 1.0),
    x0=st.tuples(coord, coord),
    theta=st.floats(0.0, math.pi),
)
def test_lion_moves_every_coordinate_by_exactly_eta(
    lr: float, decay: float, x0: tuple[float, float], theta: float
) -> None:
    res = lion(
        rotate(ROSEN, rotation_2d(theta)),
        x0=np.array(x0),
        lr=lr,
        lr_decay=decay,
        max_iter=30,
        gtol=1e-150,
    )
    for prev, step in zip(res.trace, res.trace[1:], strict=False):
        eta = lr * decay ** (step.k - 1)
        assert step.step_size == eta
        nonzero = np.asarray(step.info["c"]) != 0.0
        x_new, x_old = np.asarray(step.x), np.asarray(prev.x)
        d = np.abs(x_new - x_old)[nonzero]
        # x ± η is rounded at the scale of x: allow 2 ulp(x), nothing relative to η.
        ulp = 2 * np.spacing(np.maximum(np.abs(x_new), np.abs(x_old)))[nonzero]
        assert np.all(np.abs(d - eta) <= ulp)


@settings(max_examples=1000, deadline=None)
@given(
    lr=st.floats(1e-4, 0.5),
    gamma=st.floats(1e-4, 10.0),
    x0=st.tuples(coord, coord),
    theta=st.floats(0.0, math.pi),
    estimator=st.sampled_from(["exact", "hutchinson"]),
)
def test_sophia_steps_are_bounded_by_lr(
    lr: float, gamma: float, x0: tuple[float, float], theta: float, estimator: str
) -> None:
    # Clipping bounds every coordinate step by lr (paper, Sec. 2.2), also where the
    # diagonal Hessian of Rosenbrock is negative.
    res = sophia(
        rotate(ROSEN, rotation_2d(theta)),
        x0=np.array(x0),
        lr=lr,
        gamma=gamma,
        estimator=estimator,
        k=3,
        max_iter=30,
        gtol=1e-150,
    )
    for prev, step in zip(res.trace, res.trace[1:], strict=False):
        x_new, x_old = np.asarray(step.x), np.asarray(prev.x)
        ulp = 2 * np.spacing(np.maximum(np.abs(x_new), np.abs(x_old)))
        assert np.all(np.abs(x_new - x_old) <= lr + ulp)


def test_hutchinson_estimates_are_unbiased_and_seeded() -> None:
    # ĥ = u ⊙ (Hu), u ~ N(0, I): E ĥ = diag H (paper eq. 7); Var ĥ_i = 2H_ii² + Σ_{j≠i} H_ij².
    P = rotate(QUAD, rotation_2d(0.3))
    H = np.asarray(P.extra["A"])
    res = sophia(P, lr=1e-12, k=1, estimator="hutchinson", seed=11, max_iter=4000, gtol=1e-150)
    est = np.array([s.info["h_new"] for s in res.trace[1:]])  # (4000, 2)
    var = 2 * np.diag(H) ** 2 + (H**2).sum(1) - np.diag(H) ** 2
    se = np.sqrt(var / est.shape[0])
    assert np.all(np.abs(est.mean(0) - np.diag(H)) < 4 * se)
    again = sophia(P, lr=1e-12, k=1, estimator="hutchinson", seed=11, max_iter=50, gtol=1e-150)
    assert_array_equal(est[:50], [s.info["h_new"] for s in again.trace[1:]])
    other = sophia(P, lr=1e-12, k=1, estimator="hutchinson", seed=12, max_iter=50, gtol=1e-150)
    assert not np.array_equal(est[:50], [s.info["h_new"] for s in other.trace[1:]])


# --------------------------------------------------------------------------------------
# The rotation wrapper
# --------------------------------------------------------------------------------------


def _fd_grad(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    e = np.eye(x.size) * h
    return np.array([(f(x + e[i]) - f(x - e[i])) / (2 * h) for i in range(x.size)])


@settings(max_examples=1000, deadline=None)
@given(theta=st.floats(-math.pi, math.pi), x=st.tuples(coord, coord), which=st.booleans())
def test_rotate_derivatives_and_spectrum(theta: float, x: tuple[float, float], which: bool) -> None:
    base = ROSEN if which else QUAD
    R = rotation_2d(theta)
    P = rotate(base, R)
    z = np.array(x)
    assert P.f(z) == pytest.approx(base.f(R @ z), rel=1e-14, abs=1e-14)
    # Central differences in f64: truncation O(h²·f''') + rounding O(ε|f|/h) → ~1e-6 rel.
    g = grad(P, z)
    assert_allclose(g, _fd_grad(P.f, z), rtol=1e-5, atol=1e-5)
    Hfd = np.column_stack([(grad(P, z + h) - grad(P, z - h)) / 2e-6 for h in np.eye(2) * 1e-6])
    assert_allclose(hess(P, z), Hfd, rtol=1e-5, atol=1e-4)
    # Same spectrum as ∇²f at the rotated point (similarity transform).
    assert_allclose(
        np.linalg.eigvalsh(hess(P, z)),
        np.linalg.eigvalsh(hess(base, R @ z)),
        rtol=1e-12,
        atol=1e-10,
    )


def test_rotate_metadata_and_grids() -> None:
    R = rotation_2d(0.4)
    P = rotate(ROSEN, R)
    assert_allclose(P.minima[0], R.T @ np.array([1.0, 1.0]), rtol=1e-15)
    assert P.f(np.asarray(P.minima[0])) == pytest.approx(0.0, abs=1e-28)
    assert_allclose(P.x0, R.T @ np.asarray(ROSEN.x0), rtol=1e-15)
    grid = np.stack(np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1, 3, 4)))  # (2, 4, 5)
    vals = P.f(grid)
    assert vals.shape == (4, 5)
    assert vals[2, 3] == pytest.approx(P.f(grid[:, 2, 3]), rel=1e-14)
    Q = rotate(QUAD, R)
    A = np.asarray(Q.extra["A"])
    assert_allclose(hess(Q, np.zeros(2)), A, rtol=1e-14)
    assert hessian_11_norm(np.diag([1.0, 50.0])) == 51.0
    with pytest.raises(ValueError):
        rotate(QUAD, [[1.0, 1.0], [0.0, 1.0]])


def test_hessian_11_norm_of_rotated_quadratic() -> None:
    # ‖H(φ)‖₁,₁ = tr H + (λ_max − λ_min)|sin 2φ| for a 2×2 SPD H with eigenbasis angle φ.
    aligned = rotate(QUAD, rotation_2d(PHI0))
    for phi in np.linspace(0, math.pi / 4, 7):
        P = rotate(aligned, rotation_2d(phi))
        assert hessian_11_norm(P.extra["A"]) == pytest.approx(51 + 49 * abs(math.sin(2 * phi)))


# --------------------------------------------------------------------------------------
# The experiment's metric (run.py): settling iteration, grid-edge check, frozen Lion
# --------------------------------------------------------------------------------------

import run as experiment  # noqa: E402

fval = st.one_of(
    st.none(),
    st.just(math.nan),
    st.just(math.inf),
    st.floats(0.0, 1e-5, allow_nan=False),
)


def _settle_brute(fs: list[float | None], target: float) -> int | None:
    """Definition, verbatim: the smallest i with f_j < target for every j ≥ i."""
    for i in range(len(fs)):
        if all(f is not None and f < target for f in fs[i:]):
            return i
    return None


@settings(max_examples=1000, deadline=None)
@given(fs=st.lists(fval, min_size=1, max_size=40))
def test_settle_iteration_matches_its_definition(fs: list[float | None]) -> None:
    ks = list(range(len(fs)))
    assert experiment.settle_iteration(fs, ks, 1e-6) == _settle_brute(fs, 1e-6)
    hit = experiment.first_hit(fs, ks, 1e-6)
    brute_hit = next((k for k, f in enumerate(fs) if f is not None and f < 1e-6), None)
    assert hit == brute_hit
    settle = experiment.settle_iteration(fs, ks, 1e-6)
    if settle is not None:
        assert hit is not None and hit <= settle  # a settled run has hit the target first


def test_settle_iteration_rejects_passing_crossings() -> None:
    fs = [1.0, 1e-7, 1e-3, 1e-8, 1e-9]  # hits at k = 1, leaves at k = 2, settles at k = 3
    ks = [0, 1, 2, 3, 4]
    assert experiment.first_hit(fs, ks) == 1
    assert experiment.settle_iteration(fs, ks) == 3
    assert experiment.settle_iteration([1.0, 1e-7, 1e-3], [0, 1, 2]) is None  # ends above
    assert experiment.solved_n(3, n_max=3) == 3
    assert experiment.solved_n(4, n_max=3) is None
    assert experiment.solved_n(None, n_max=3) is None


def test_rmsprop_crossing_is_not_counted_as_solved() -> None:
    # The reviewer's case: RMSprop, lr = 100, quadratic_ill at φ = 0 first rises to f ≈ 2.5e6,
    # crosses the target at k = 7 and never settles. The first-hit count would report 7.
    a = next(i for i, t in enumerate(experiment.QUAD_THETAS) if abs(t - math.degrees(PHI0)) < 1e-9)
    c = next(
        i for i, g in enumerate(experiment.GRIDS["quadratic_ill"]["rmsprop"]) if g["lr"] == 100
    )
    _, rec = experiment._task(("quadratic_ill", a, "rmsprop", c, 0, 10000))
    assert rec["first"] is not None and rec["first"] < 20
    assert rec["N"] is None
    assert rec["peak_ratio"] > 1e4
    # A monotone method settles at its first hit.
    c = next(i for i, g in enumerate(experiment.GRIDS["quadratic_ill"]["gd"]) if g["lr"] == 0.01)
    _, rec = experiment._task(("quadratic_ill", a, "gd", c, 0, 10000))
    assert rec["N"] == rec["first"] == rec["settle"] and rec["peak_ratio"] == 1.0


def test_edge_keys_checks_every_tuned_hyperparameter() -> None:
    grid = experiment._product([0.1, 1.0, 10.0], "lr_decay", [0.9, 0.95, 0.99])
    i = grid.index({"lr": 1.0, "lr_decay": 0.9})
    assert experiment.edge_keys(grid, i) == ["lr_decay"]
    assert experiment.edge_keys(grid, grid.index({"lr": 10.0, "lr_decay": 0.99})) == [
        "lr",
        "lr_decay",
    ]
    assert experiment.edge_keys(grid, grid.index({"lr": 1.0, "lr_decay": 0.95})) == []
    assert experiment.edge_keys([{"lr": 1.0}], 0) == []  # a single value has no edge


@settings(max_examples=200, deadline=None)
@given(
    lr=st.floats(1e-3, 1.0),
    rho=st.floats(0.5, 0.99),
    K=st.integers(1, 60),
)
def test_lion_remaining_travel_bound(lr: float, rho: float, K: int) -> None:
    # After K iterations a coordinate can move at most lr·ρ^K/(1 − ρ) more (λ = 0). Run 400
    # further iterations and check the bound that lion_frozen relies on.
    x0 = np.array([2.0, -1.5])
    a = lion(QUAD, x0=x0, lr=lr, lr_decay=rho, max_iter=K, gtol=1e-150)
    b = lion(QUAD, x0=x0, lr=lr, lr_decay=rho, max_iter=K + 400, gtol=1e-150)
    if b.n_iter < K + 400 or a.n_iter < K:
        return  # stopped at the gradient floor
    assert_array_equal(b.trace[K].x, a.x)  # same path
    travel = np.max(np.abs(np.asarray(b.x) - np.asarray(a.x)))
    # Each of the 400 updates x ← x + η p rounds once (|error| ≤ ε/2·|x|), and the subtraction
    # above rounds once more, so the floating-point travel can exceed the exact one by that much.
    scale = max(float(np.max(np.abs(s.x))) for s in b.trace[K:])
    rounding = 401 * np.finfo(float).eps * scale
    assert travel <= lr * rho**K / (1 - rho) * (1 + 1e-12) + rounding


def test_lion_frozen_detects_an_exhausted_schedule() -> None:
    P = QUAD
    x0 = np.array([2.0, -1.5])
    cfg = {"lr": 0.01, "lr_decay": 0.9}  # total travel 0.1 < distance 2 to x* = 0
    res = lion(P, x0=x0, max_iter=200, gtol=1e-150, **cfg)
    assert experiment.lion_frozen(res, P, cfg)
    cfg = {"lr": 0.3, "lr_decay": 0.99}  # ample travel left after 50 iterations
    res = lion(P, x0=x0, max_iter=50, gtol=1e-150, **cfg)
    assert not experiment.lion_frozen(res, P, cfg)
    assert not experiment.lion_frozen(res, P, {"lr": 0.3, "lr_decay": 1.0})


def test_choose_prefers_fewest_unsolved_then_median() -> None:
    inf = math.inf
    N = np.array([[10.0, inf, 10.0, 10.0], [50.0, 60.0, 70.0, 80.0], [40.0, 60.0, 70.0, 85.0]])
    assert experiment.choose(N) == 2  # rows 1, 2 solve all; equal medians, row 2 smaller mean


@settings(max_examples=1000, deadline=None)
@given(fs=st.lists(fval, min_size=2, max_size=40), cut=st.integers(1, 39), n_max=st.integers(0, 40))
def test_settling_iteration_is_monotone_in_the_horizon(
    fs: list[float | None], cut: int, n_max: int
) -> None:
    # The lazy certification in run.py relies on this: a longer horizon (the same trajectory
    # with more iterations) never gives a smaller N and never turns an unsolved run solved.
    cut = min(cut, len(fs) - 1)
    ks = list(range(len(fs)))
    short = experiment.solved_n(experiment.settle_iteration(fs[:cut], ks[:cut]), n_max)
    full = experiment.solved_n(experiment.settle_iteration(fs, ks), n_max)
    if short is None:
        assert full is None or full >= cut
    elif full is not None:
        assert full >= short


def test_block_max_keeps_every_excursion() -> None:
    fs = [1.0] + [1e-9] * 999
    fs[537] = 5e-3  # one spike
    k, f = experiment.block_max(list(range(1000)), fs, points=10)
    assert len(k) == len(f) == 10
    assert f.max() == 1.0 and np.sort(f)[-2] == 5e-3
    assert f[5] == 5e-3 and k[5] == 500
