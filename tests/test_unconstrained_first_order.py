"""Tests for numopt.unconstrained.first_order.

Oracles: (1) exact rational arithmetic (``fractions.Fraction``) for the methods whose iterates
are rational on a rational quadratic, including one first step computed by hand;
(2) independent straightforward re-implementations of the papers' update rules (5 steps);
(3) mathematically equivalent formulations (heavy ball as a two-step recurrence, Nesterov's
two-sequence form, the closed form (I − lr A)ᵏ of fixed-step gradient descent, Gauss–Seidel
as a matrix splitting); (4) SciPy's BFGS minimizers; (5) Hypothesis invariants (Kantorovich
bound, gradient orthogonality, Rayleigh-quotient bounds of BB steps, Armijo/GLL decrease,
the Cauchy–Schwarz bound on Adam's step, monotone AdaGrad/AMSGrad step sizes).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import linalg, optimize

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import Problem, Result
from numopt.unconstrained import first_order as fo

METHODS = (
    "gradient_descent",
    "barzilai_borwein",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adadelta",
    "adam",
    "adamax",
    "nadam",
    "amsgrad",
    "adamw",
    "coordinate_descent",
)

#: Documented info keys per method (in addition to grad, direction, alpha).
INFO_KEYS: dict[str, set[str]] = {
    "gradient_descent": {"trials", "alpha0", "pHp"},
    "barzilai_borwein": {"alpha_bb", "bb1", "bb2", "reset", "f_ref", "trials"},
    "momentum": {"velocity"},
    "nesterov": {"velocity", "lookahead", "grad_lookahead"},
    "adagrad": {"v", "lr_eff"},
    "rmsprop": {"v", "lr_eff"},
    "adadelta": {"v", "u", "lr_eff"},
    "adam": {"m", "v", "m_hat", "v_hat", "lr_eff"},
    "adamw": {"m", "v", "m_hat", "v_hat", "lr_eff", "decay"},
    "adamax": {"m", "u", "lr_eff"},
    "nadam": {"m", "v", "m_bar", "v_hat", "lr_eff"},
    "amsgrad": {"m", "v", "v_max", "lr_eff"},
    "coordinate_descent": {"coordinate", "sweep", "curvature", "newton", "trials"},
}

#: Documented behavior of the defaults (gtol = 1e-6): converged or stops at max_iter.
DEFAULT_CONVERGES = {
    "quadratic_bowl": set(METHODS) - {"adamw"},
    "rosenbrock": {
        "barzilai_borwein",
        "momentum",
        "nesterov",
        "adam",
        "adamax",
        "nadam",
        "amsgrad",
    },
}

EPS = float(np.finfo(float).eps)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class Calls:
    """Independent evaluation counters."""

    def __init__(self) -> None:
        self.f = 0
        self.g = 0
        self.h = 0


def counting(p: Problem) -> tuple[Problem, Calls]:
    """A copy of p whose f, grad and hess count their calls."""
    c = Calls()

    def f(x: Any) -> Any:
        c.f += 1
        return p.f(x)

    def grad(x: Any) -> Any:
        c.g += 1
        assert p.grad is not None
        return p.grad(x)

    def hess(x: Any) -> Any:
        c.h += 1
        assert p.hess is not None
        return p.hess(x)

    return dataclasses.replace(p, f=f, grad=grad, hess=hess), c


def quadratic(A: np.ndarray, c: np.ndarray, x0: np.ndarray) -> Problem:
    """f(x) = ½(x − c)ᵀA(x − c) with exact derivatives."""
    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float)

    def f(x: Any) -> float:
        d = np.asarray(x, dtype=float) - c
        return 0.5 * float(d @ (A @ d))

    return Problem(
        id="quad",
        name="quad",
        latex="",
        f=f,
        dim=c.size,
        domain=(),
        grad=lambda x: A @ (np.asarray(x, dtype=float) - c),
        hess=lambda x: A.copy(),
        x0=tuple(x0),
        minima=(tuple(c),),
    )


def spd2(lam1: float, kappa: float, theta: float) -> np.ndarray:
    """2×2 SPD matrix Q diag(λ₁, κλ₁) Qᵀ with Q the rotation by θ."""
    Q = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    A = Q @ np.diag([lam1, kappa * lam1]) @ Q.T
    return 0.5 * (A + A.T)


def xs(res: Result) -> np.ndarray:
    return np.array([s.x for s in res.trace], dtype=float)


def check_contract(res: Result, p: Problem, calls: Calls, method: str, max_iter: int) -> None:
    """Every contract the module docstring promises, checked step by step."""
    assert_valid_result(res, max_iter=max_iter)
    assert res.method == method
    assert res.n_iter == res.trace[-1].k
    assert [s.k for s in res.trace] == list(range(len(res.trace)))
    assert (res.n_fev, res.n_gev, res.n_hev) == (calls.f, calls.g, calls.h)
    np.testing.assert_array_equal(res.x, res.trace[-1].x)
    assert res.fun == res.trace[-1].fun
    keys = {"grad", "direction", "alpha"} | INFO_KEYS[method]
    for s in res.trace:
        assert set(s.info) == keys, (method, set(s.info) ^ keys)
        g = np.array(s.info["grad"])
        assert s.grad_norm == float(np.linalg.norm(g))
        assert s.step_size == s.info["alpha"]
        if s.fun is not None and math.isfinite(s.fun):
            assert s.fun == float(p.f(np.asarray(s.x)))
            np.testing.assert_array_equal(g, p.grad(np.asarray(s.x)))  # type: ignore[misc]
    assert res.trace[0].info["direction"] is None and res.trace[0].info["alpha"] is None
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        p_k = np.array(cur.info["direction"])
        alpha = cur.info["alpha"]
        assert alpha >= 0.0
        # x_k = x_{k−1} + α_k p_k: p_k = step/α and the sum each add one rounding.
        step = alpha * p_k
        err = np.abs(np.asarray(cur.x) - (np.asarray(prev.x) + step))
        assert np.all(err <= 4 * EPS * (np.abs(prev.x) + np.abs(step)))
    if res.converged:
        assert "gtol" in res.message


# --------------------------------------------------------------------------------------
# Registry and fixtures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_registered_with_consistent_params(method):
    spec = get_method(method)
    assert spec.family == "unconstrained"
    assert spec.references and spec.summary
    sig = inspect.signature(spec.fn).parameters
    assert set(sig) - {"problem", "x0"} == {p.name for p in spec.params}
    for prm in spec.params:
        assert sig[prm.name].default == prm.default
        if prm.kind in ("float", "int"):
            assert prm.min is not None and prm.max is not None
            assert prm.min <= prm.default <= prm.max
    assert spec.defaults()["max_iter"] <= 5000


def test_fixture_cases_cover_every_method():
    cases = fo.FIXTURE_CASES
    assert {m for m, _, _ in cases} == set(METHODS)
    for method, pid, params in cases:
        p = problems.get(pid)
        res = numopt.run(method, p, **params)
        assert_valid_result(res)
        assert len(res.trace) < 300


# --------------------------------------------------------------------------------------
# Defaults on the two reference problems (convergence or honest non-convergence)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", sorted(DEFAULT_CONVERGES))
@pytest.mark.parametrize("method", METHODS)
def test_defaults_on_reference_problems(method, pid):
    p0 = problems.get(pid)
    p, calls = counting(p0)
    res = numopt.run(method, p)
    max_iter = get_method(method).defaults()["max_iter"]
    check_contract(res, p0, calls, method, max_iter)
    x_star = np.array(p0.minima[0])
    if method in DEFAULT_CONVERGES[pid]:
        assert res.converged, res.message
        gn = res.trace[-1].grad_norm
        assert gn is not None and gn <= 1e-6
        # ‖x − x*‖ ≤ ‖∇f‖/λ_min(∇²f(x*)) to first order; λ_min = 2 (bowl), 0.399 (rosenbrock).
        lam_min = float(np.linalg.eigvalsh(p0.hess(x_star)).min())  # type: ignore[misc]
        assert np.linalg.norm(res.x - x_star) <= 2e-6 / lam_min
    else:
        assert not res.converged
        assert "max_iter" in res.message
        assert res.n_iter == max_iter
        f_start = res.trace[0].fun
        assert res.fun is not None and f_start is not None
        assert res.fun < f_start  # it made progress, it did not diverge


# --------------------------------------------------------------------------------------
# Oracle: SciPy BFGS minimizers
# --------------------------------------------------------------------------------------

#: (method, problem) pairs that reach gtol = 1e-9 within 20000 iterations from the default x0.
BFGS_CASES = [
    (m, pid)
    for m, pids in {
        "gradient_descent": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "barzilai_borwein": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "momentum": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "nesterov": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "adagrad": "quadratic_bowl quadratic_ill booth beale himmelblau",
        "rmsprop": "quadratic_bowl quadratic_ill himmelblau",
        "adadelta": "quadratic_bowl quadratic_ill",
        "adam": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "adamax": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "nadam": "quadratic_bowl quadratic_ill booth beale himmelblau",
        "amsgrad": "quadratic_bowl quadratic_ill booth beale himmelblau rosenbrock",
        "adamw": "quadratic_ill",
        "coordinate_descent": "quadratic_bowl quadratic_ill booth beale rosenbrock",
    }.items()
    for pid in pids.split()
]


@pytest.mark.parametrize(("method", "pid"), BFGS_CASES)
def test_minimizer_agrees_with_scipy_bfgs(method, pid):
    p = problems.get(pid)
    x0 = np.array(p.x0, dtype=float)
    ref = optimize.minimize(p.f, x0, jac=p.grad, method="BFGS", options={"gtol": 1e-12})
    gtol = 1e-9
    res = numopt.run(method, p, gtol=gtol, max_iter=20_000)
    assert res.converged, res.message
    # Both points are stationary: ‖x − x_ref‖ ≤ (‖∇f(x)‖ + ‖∇f(x_ref)‖)/λ_min to first order.
    # NOTE: factor 2 covers the second-order term and ‖∇f(x_ref)‖ ≤ 1e-12.
    lam_min = float(np.linalg.eigvalsh(p.hess(ref.x)).min())  # type: ignore[misc]
    assert np.linalg.norm(res.x - ref.x) <= 2 * gtol / lam_min


def test_coordinate_descent_himmelblau_reaches_another_local_minimum():
    """From (0, 0) BFGS goes to (3, 2), coordinate descent to a different listed minimum."""
    p = problems.get("himmelblau")
    res = numopt.run("coordinate_descent", p, gtol=1e-9)
    assert res.converged
    dists = [np.linalg.norm(res.x - np.array(m)) for m in p.minima]
    i = int(np.argmin(dists))
    lam_min = float(np.linalg.eigvalsh(p.hess(np.array(p.minima[i]))).min())  # type: ignore[misc]
    assert dists[i] <= 2e-9 / lam_min
    assert i != 0


# --------------------------------------------------------------------------------------
# Oracle: exact rational arithmetic on quadratic_bowl (A = [[3, 1], [1, 3]], c = (1, −½))
# --------------------------------------------------------------------------------------

A_Q = [[Fraction(3), Fraction(1)], [Fraction(1), Fraction(3)]]
C_Q = [Fraction(1), Fraction(-1, 2)]
X0_Q = [Fraction(-2), Fraction(2)]

Vec = list[Fraction]


def q_grad(x: Vec) -> Vec:
    d = [x[0] - C_Q[0], x[1] - C_Q[1]]
    return [A_Q[0][0] * d[0] + A_Q[0][1] * d[1], A_Q[1][0] * d[0] + A_Q[1][1] * d[1]]


def dot(u: Vec, v: Vec) -> Fraction:
    return u[0] * v[0] + u[1] * v[1]


def a_times(v: Vec) -> Vec:
    return [A_Q[0][0] * v[0] + A_Q[0][1] * v[1], A_Q[1][0] * v[0] + A_Q[1][1] * v[1]]


def axpy(a: Fraction, x: Vec, y: Vec) -> Vec:
    return [y[0] + a * x[0], y[1] + a * x[1]]


def assert_iterates(res: Result, ref: list[Vec], n: int = 5) -> None:
    assert res.n_iter >= n
    mine = xs(res)[: n + 1]
    exact = np.array([[float(v) for v in x] for x in ref[: n + 1]])
    # NOTE: 5 steps on a quadratic with κ = 2: a few roundings per step, error ≲ 10ε·‖x‖.
    np.testing.assert_allclose(mine, exact, rtol=1e-13, atol=1e-14)


def test_hand_computed_first_cauchy_step():
    """By hand: g₀ = A(x₀ − c) = (−13/2, 9/2), gᵀg = 125/2, gᵀAg = 129, α = 125/258,
    x₁ = x₀ − αg₀ = (593/516, −31/172)."""
    g0 = q_grad(X0_Q)
    assert g0 == [Fraction(-13, 2), Fraction(9, 2)]
    alpha = dot(g0, g0) / dot(g0, a_times(g0))
    assert alpha == Fraction(125, 258)
    x1 = axpy(-alpha, g0, X0_Q)
    assert x1 == [Fraction(593, 516), Fraction(-31, 172)]
    res = numopt.run(
        "gradient_descent", problems.get("quadratic_bowl"), step_rule="exact_quadratic"
    )
    # NOTE: x₁ = x₀ − αg₀ cancels digits (2 − 2.18…), so bound the absolute error by the
    # magnitude of the terms: |err| ≤ 4ε(|x₀| + |αg₀|).
    terms = np.abs([float(v) for v in X0_Q]) + np.abs([float(alpha * v) for v in g0])
    err = np.abs(np.asarray(res.trace[1].x) - np.array([593 / 516, -31 / 172]))
    assert np.all(err <= 4 * EPS * terms)
    assert res.trace[1].info["alpha"] == pytest.approx(125 / 258, rel=2 * EPS)


def test_gradient_descent_exact_steps_are_rational_cauchy_steps():
    x = list(X0_Q)
    ref = [x]
    for _ in range(5):
        g = q_grad(x)
        x = axpy(-dot(g, g) / dot(g, a_times(g)), g, x)
        ref.append(x)
    res = numopt.run(
        "gradient_descent", problems.get("quadratic_bowl"), step_rule="exact_quadratic", gtol=1e-14
    )
    assert_iterates(res, ref)


def test_gradient_descent_fixed_step_rational():
    lr = Fraction(1, 5)
    x = list(X0_Q)
    ref = [x]
    for _ in range(5):
        x = axpy(-lr, q_grad(x), x)
        ref.append(x)
    res = numopt.run("gradient_descent", problems.get("quadratic_bowl"), step_rule="fixed", lr=0.2)
    assert_iterates(res, ref)


@pytest.mark.parametrize("variant", ["bb1", "bb2"])
def test_barzilai_borwein_pure_rational(variant):
    """Pure BB. The first step 1/‖g₀‖ is irrational, so the exact recurrence starts from the
    method's float x₀ and x₁ (exact binary fractions) and is compared from k = 2 on."""
    res = numopt.run(
        "barzilai_borwein",
        problems.get("quadratic_bowl"),
        variant=variant,
        nonmonotone=False,
        gtol=1e-14,
    )
    x_prev = [Fraction(float(v)) for v in res.trace[0].x]
    x = [Fraction(float(v)) for v in res.trace[1].x]
    ref = [x_prev, x]
    for _ in range(4):
        s = [x[0] - x_prev[0], x[1] - x_prev[1]]
        y = a_times(s)
        alpha = dot(s, s) / dot(s, y) if variant == "bb1" else dot(s, y) / dot(y, y)
        x_prev, x = x, axpy(-alpha, q_grad(x), x)
        ref.append(x)
    assert_iterates(res, ref)
    g0 = np.array(res.trace[0].info["grad"])
    assert res.trace[1].info["alpha"] == pytest.approx(1 / np.linalg.norm(g0), rel=EPS)
    assert res.trace[1].info["reset"] is True


def test_momentum_and_nesterov_rational():
    lr, beta = Fraction(1, 10), Fraction(1, 2)
    for method in ("momentum", "nesterov"):
        x, v = list(X0_Q), [Fraction(0), Fraction(0)]
        ref = [x]
        for _ in range(5):
            look = axpy(beta, v, x) if method == "nesterov" else x
            g = q_grad(look)
            v = [beta * v[0] - lr * g[0], beta * v[1] - lr * g[1]]
            x = [x[0] + v[0], x[1] + v[1]]
            ref.append(x)
        res = numopt.run(method, problems.get("quadratic_bowl"), lr=0.1, beta=0.5)
        assert_iterates(res, ref)


def test_coordinate_descent_rational_gauss_seidel():
    x = list(X0_Q)
    ref = [x]
    for k in range(5):
        i = k % 2
        g = q_grad(x)
        x = list(x)
        x[i] = x[i] - g[i] / A_Q[i][i]
        ref.append(x)
    res = numopt.run("coordinate_descent", problems.get("quadratic_bowl"), gtol=1e-14)
    assert_iterates(res, ref)
    assert all(s.info["alpha"] == 1.0 for s in res.trace[1:6])


# --------------------------------------------------------------------------------------
# Oracle: independent re-implementations of the adaptive methods (5 steps, rosenbrock)
# --------------------------------------------------------------------------------------


def ref_adaptive(method: str, grad: Any, x: np.ndarray, n: int, **h: float) -> np.ndarray:
    """Straight transcription of each paper's pseudocode, written independently."""
    out = [x.copy()]
    d = x.size
    s1, s2, s3 = np.zeros(d), np.zeros(d), np.zeros(d)
    for t in range(1, n + 1):
        g = grad(x)
        if method == "adagrad":  # Duchi et al. 2011
            s1 += g**2
            x = x - h["lr"] * g / (np.sqrt(s1) + h["eps"])
        elif method == "rmsprop":  # Tieleman & Hinton 2012
            s1 = h["rho"] * s1 + (1 - h["rho"]) * g**2
            x = x - h["lr"] * g / (np.sqrt(s1) + h["eps"])
        elif method == "adadelta":  # Zeiler 2012, Alg. 1
            s1 = h["rho"] * s1 + (1 - h["rho"]) * g**2
            dx = -np.sqrt(s2 + h["eps"]) / np.sqrt(s1 + h["eps"]) * g
            s2 = h["rho"] * s2 + (1 - h["rho"]) * dx**2
            x = x + dx
        elif method in ("adam", "adamw"):  # Kingma & Ba 2015 Alg. 1; L&H 2019 Alg. 2
            s1 = h["beta1"] * s1 + (1 - h["beta1"]) * g
            s2 = h["beta2"] * s2 + (1 - h["beta2"]) * g**2
            mh = s1 / (1 - h["beta1"] ** t)
            vh = s2 / (1 - h["beta2"] ** t)
            x = x - (h["lr"] * mh / (np.sqrt(vh) + h["eps"]) + h.get("weight_decay", 0.0) * x)
        elif method == "adamax":  # Kingma & Ba 2015 Alg. 2
            s1 = h["beta1"] * s1 + (1 - h["beta1"]) * g
            s2 = np.maximum(h["beta2"] * s2, np.abs(g))
            x = x - (h["lr"] / (1 - h["beta1"] ** t)) * s1 / s2
        elif method == "nadam":  # Dozat 2016 (CS229 Alg. 8), μ_t = β₁
            b1 = h["beta1"]
            g_hat = g / (1 - b1**t)
            s1 = b1 * s1 + (1 - b1) * g
            m_hat = s1 / (1 - b1 ** (t + 1))
            s2 = h["beta2"] * s2 + (1 - h["beta2"]) * g**2
            n_hat = s2 / (1 - h["beta2"] ** t)
            m_bar = (1 - b1) * g_hat + b1 * m_hat
            x = x - h["lr"] * m_bar / (np.sqrt(n_hat) + h["eps"])
        elif method == "amsgrad":  # Reddi et al. 2018 Alg. 2
            s1 = h["beta1"] * s1 + (1 - h["beta1"]) * g
            s2 = h["beta2"] * s2 + (1 - h["beta2"]) * g**2
            s3 = np.maximum(s3, s2)
            x = x - h["lr"] * s1 / np.sqrt(s3)
        out.append(x.copy())
    return np.array(out)


ADAPTIVE_HYPER = {
    "adagrad": {"lr": 0.3, "eps": 1e-8},
    "rmsprop": {"lr": 0.01, "rho": 0.9, "eps": 1e-8},
    "adadelta": {"rho": 0.95, "eps": 1e-6},
    "adam": {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "adamw": {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
    "adamax": {"lr": 0.2, "beta1": 0.8, "beta2": 0.99},
    "nadam": {"lr": 0.02, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "amsgrad": {"lr": 0.1, "beta1": 0.9, "beta2": 0.99},
}


@pytest.mark.parametrize("method", sorted(ADAPTIVE_HYPER))
def test_adaptive_methods_match_paper_pseudocode(method):
    p = problems.get("rosenbrock")
    h = ADAPTIVE_HYPER[method]
    ref = ref_adaptive(method, p.grad, np.array(p.x0, dtype=float), 5, **h)
    res = numopt.run(method, p, max_iter=5, **h)
    # NOTE: same formulas evaluated in a different order: a few roundings per step.
    np.testing.assert_allclose(xs(res), ref, rtol=1e-13, atol=1e-15)


def test_adam_first_step_is_signed_lr():
    """m̂₁ = g₁ and v̂₁ = g₁², so x₁ = x₀ − lr·g/(|g| + ε) ≈ x₀ − lr·sign(g)."""
    p = problems.get("rosenbrock")
    res = numopt.run("adam", p, lr=0.05, max_iter=1)
    g0 = p.grad(np.array(p.x0))  # type: ignore[misc]
    x1 = np.array(p.x0) - 0.05 * g0 / (np.abs(g0) + 1e-8)
    np.testing.assert_allclose(res.trace[1].x, x1, rtol=4 * EPS)
    np.testing.assert_allclose(res.trace[1].x, np.array(p.x0) - 0.05 * np.sign(g0), rtol=1e-9)


def test_adamw_without_decay_equals_adam():
    p = problems.get("beale")
    a = numopt.run("adam", p, max_iter=200)
    w = numopt.run("adamw", p, weight_decay=0.0, max_iter=200)
    np.testing.assert_array_equal(xs(a), xs(w))
    assert (a.converged, a.n_iter) == (w.converged, w.n_iter)
    assert all(s.info["decay"] == [0.0, 0.0] for s in w.trace[1:])


# --------------------------------------------------------------------------------------
# Oracle: equivalent formulations
# --------------------------------------------------------------------------------------


def test_fixed_gradient_descent_closed_form():
    """On f = ½(x − c)ᵀA(x − c): x_k − c = (I − lr·A)ᵏ(x₀ − c)."""
    p = problems.get("quadratic_ill")
    A, c = np.array(p.extra["A"]), np.array(p.extra["c"])
    lr = 0.03
    res = numopt.run("gradient_descent", p, step_rule="fixed", lr=lr, max_iter=40)
    M = np.eye(2) - lr * A
    for s in res.trace:
        ref = c + np.linalg.matrix_power(M, s.k) @ (np.array(p.x0) - c)
        np.testing.assert_allclose(s.x, ref, rtol=1e-12, atol=1e-14)


def test_heavy_ball_equals_two_step_recurrence():
    """x_{k+1} = x_k − lr∇f(x_k) + β(x_k − x_{k−1}), x_{−1} = x₀ (Polyak 1964)."""
    p = problems.get("himmelblau")
    lr, beta = 0.005, 0.8
    res = numopt.run("momentum", p, lr=lr, beta=beta, max_iter=60)
    x_prev = x = np.array(p.x0, dtype=float)
    ref = [x]
    for _ in range(res.n_iter):
        x_prev, x = x, x - lr * p.grad(x) + beta * (x - x_prev)  # type: ignore[misc]
        ref.append(x)
    np.testing.assert_allclose(xs(res), np.array(ref), rtol=1e-11, atol=1e-12)
    vel = np.array([s.info["velocity"] for s in res.trace[1:]])
    np.testing.assert_allclose(vel, np.diff(xs(res), axis=0), rtol=1e-12, atol=1e-14)


def test_nesterov_equals_two_sequence_form():
    """y_k = x_k + μ(x_k − x_{k−1}),  x_{k+1} = y_k − lr∇f(y_k)  (Nesterov 1983)."""
    p = problems.get("beale")
    lr, mu = 0.01, 0.9
    res = numopt.run("nesterov", p, lr=lr, beta=mu, max_iter=80)
    x_prev = x = np.array(p.x0, dtype=float)
    ref = [x]
    for _ in range(res.n_iter):
        y = x + mu * (x - x_prev)
        x_prev, x = x, y - lr * p.grad(y)  # type: ignore[misc]
        ref.append(x)
    np.testing.assert_allclose(xs(res), np.array(ref), rtol=1e-11, atol=1e-12)
    for s in res.trace[1:]:
        np.testing.assert_array_equal(
            s.info["grad_lookahead"], p.grad(np.array(s.info["lookahead"]))
        )  # type: ignore[misc]


def test_nesterov_gradient_count():
    p0 = problems.get("quadratic_bowl")
    p, calls = counting(p0)
    res = numopt.run("nesterov", p, max_iter=50)
    # 1 at x₀; k = 1 reuses ∇f(x₀) (v₀ = 0); then 2 per iteration.
    assert res.n_gev == calls.g == 2 * res.n_iter


def test_coordinate_descent_sweep_is_gauss_seidel_splitting():
    """n coordinate steps on a quadratic = one Gauss–Seidel sweep x ← (D + L)⁻¹(b − Ux), b = Ac."""
    p = problems.get("quadratic_nd")
    A, c = np.array(p.extra["A"]), np.array(p.extra["c"])
    n = c.size
    b = A @ c
    DL, U = np.tril(A), np.triu(A, 1)
    res = numopt.run("coordinate_descent", p, max_iter=5 * n)
    x = np.array(p.x0, dtype=float)
    for sweep in range(5):
        x = linalg.solve_triangular(DL, b - U @ x, lower=True)
        np.testing.assert_allclose(res.trace[(sweep + 1) * n].x, x, rtol=1e-11, atol=1e-12)
    assert all(s.info["newton"] and s.info["alpha"] == 1.0 for s in res.trace[1:])


# --------------------------------------------------------------------------------------
# Hypothesis invariants
# --------------------------------------------------------------------------------------

quad_params = st.tuples(
    st.floats(0.5, 2.0),  # λ₁
    st.floats(1.0, 100.0),  # κ
    st.floats(0.0, math.pi),  # rotation
    st.tuples(st.floats(-2, 2), st.floats(-2, 2)),  # c
    st.tuples(st.floats(-5, 5), st.floats(-5, 5)),  # x₀
)


@settings(max_examples=1000, deadline=None)
@given(quad_params)
def test_exact_steps_kantorovich_and_orthogonality(params):
    lam1, kappa, theta, c, x0 = params
    c, x0 = np.array(c), np.array(x0)
    assume(np.linalg.norm(x0 - c) > 0.1)
    A = spd2(lam1, kappa, theta)
    lam = np.linalg.eigvalsh(A)
    k_true = lam[-1] / lam[0]
    p = quadratic(A, c, x0)
    # Theorem 3.3 bounds the count: ≤ ln(f₀/f_end)/ln(1/r) ≈ 1150 for κ = 100.
    res = numopt.run("gradient_descent", p, step_rule="exact_quadratic", gtol=1e-9, max_iter=5000)
    assert res.converged
    rate = ((k_true - 1) / (k_true + 1)) ** 2
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        # N&W Theorem 3.3 (f* = 0). NOTE: f is computed from d = x − c with absolute error
        # ~ε(‖x‖ + ‖c‖), i.e. f has absolute error ~ λ_max·‖d‖·ε(‖x‖ + ‖c‖).
        d = np.linalg.norm(np.array(prev.x) - c)
        noise = 8 * lam[-1] * d * EPS * (np.linalg.norm(prev.x) + np.linalg.norm(c) + 1)
        assert cur.fun <= rate * prev.fun * (1 + 1e-12) + noise
        g0, g1 = np.array(prev.info["grad"]), np.array(cur.info["grad"])
        # Exact line minimization ⇒ g_{k+1} ⊥ g_k. NOTE: g_{k+1} = A(x − c) carries absolute
        # error ~λ_max·ε(‖x‖ + ‖c‖); the cosine bound scales with that over ‖g_{k+1}‖.
        err = 16 * lam[-1] * EPS * (np.linalg.norm(cur.x) + np.linalg.norm(c) + 1)
        assert abs(g0 @ g1) <= np.linalg.norm(g0) * (1e-12 * np.linalg.norm(g1) + err)


@settings(max_examples=1000, deadline=None)
@given(quad_params, st.sampled_from(["bb1", "bb2"]), st.booleans())
def test_bb_steps_lie_in_inverse_spectrum(params, variant, nonmonotone):
    lam1, kappa, theta, c, x0 = params
    c, x0 = np.array(c), np.array(x0)
    assume(np.linalg.norm(x0 - c) > 0.1)
    A = spd2(lam1, kappa, theta)
    lam = np.linalg.eigvalsh(A)
    p = quadratic(A, c, x0)
    res = numopt.run("barzilai_borwein", p, variant=variant, nonmonotone=nonmonotone, gtol=1e-6)
    assert_valid_result(res, max_iter=1000)
    assert res.converged, res.message
    for k, s in enumerate(res.trace[2:], start=2):
        bb1, bb2 = s.info["bb1"], s.info["bb2"]
        assert bb1 is not None and bb2 is not None
        assert s.info["reset"] is False
        # sᵀy/sᵀs and yᵀy/sᵀy are Rayleigh quotients of A and A² (y = As) ⇒ both steps lie in
        # [1/λ_max, 1/λ_min]; BB2 ≤ BB1 by Cauchy–Schwarz. NOTE: y = g_k − g_{k−1} loses
        # ε·‖g‖/‖y‖ relative accuracy to cancellation; 1e-6 covers ‖s‖ down to the stop.
        assert 1 / lam[-1] * (1 - 1e-6) <= bb2 <= bb1 * (1 + 1e-12)
        assert bb1 <= 1 / lam[0] * (1 + 1e-6)
        # BB1 is the exact (Cauchy) step of two iterations back (lagged steepest descent).
        g = np.array(res.trace[k - 2].info["grad"])
        if np.linalg.norm(g) > 1e-4:
            assert bb1 == pytest.approx((g @ g) / (g @ A @ g), rel=1e-6)


SMOOTH_2D = ["rosenbrock", "himmelblau", "beale", "booth", "quadratic_ill", "six_hump_camel"]
start_points = st.tuples(st.sampled_from(SMOOTH_2D), st.floats(0.0, 1.0), st.floats(0.0, 1.0))


def point_in_domain(pid: str, u: float, v: float) -> tuple[Problem, np.ndarray]:
    p = problems.get(pid)
    (a0, b0), (a1, b1) = p.domain
    return p, np.array([a0 + u * (b0 - a0), a1 + v * (b1 - a1)])


@settings(max_examples=1000, deadline=None)
@given(start_points, st.sampled_from(["backtracking", "strong_wolfe"]))
def test_gradient_descent_line_search_armijo_decrease(start, rule):
    p, x0 = point_in_domain(*start)
    res = numopt.run("gradient_descent", p, x0=x0, step_rule=rule, max_iter=40)
    assert_valid_result(res, max_iter=40)
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        g = np.array(prev.info["grad"])
        alpha = cur.info["alpha"]
        assert cur.fun <= prev.fun + 1e-4 * alpha * -(g @ g)  # N&W eq. 3.4, c₁ = 1e-4
        assert cur.info["trials"][-1] == [alpha, cur.fun]


@settings(max_examples=1000, deadline=None)
@given(start_points)
def test_coordinate_descent_one_coordinate_and_armijo(start):
    p, x0 = point_in_domain(*start)
    res = numopt.run("coordinate_descent", p, x0=x0, max_iter=40)
    assert_valid_result(res, max_iter=40)
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        i = cur.info["coordinate"]
        assert i == (cur.k - 1) % 2 and cur.info["sweep"] == (cur.k - 1) // 2
        assert prev.x[1 - i] == cur.x[1 - i]  # the other coordinate is untouched (bitwise)
        g_i = prev.info["grad"][i]
        p_i = cur.info["direction"][i]
        assert cur.fun <= prev.fun + 1e-4 * cur.info["alpha"] * g_i * p_i
        if cur.info["newton"]:
            assert p_i == pytest.approx(-g_i / cur.info["curvature"], rel=EPS)


@settings(max_examples=1000, deadline=None)
@given(start_points)
def test_bb_nonmonotone_gll_condition(start):
    p, x0 = point_in_domain(*start)
    res = numopt.run("barzilai_borwein", p, x0=x0, max_iter=60)
    assert_valid_result(res, max_iter=60)
    fs = [float(s.fun) for s in res.trace if s.fun is not None]
    assert len(fs) == len(res.trace)
    for k, s in enumerate(res.trace[1:], start=1):
        f_ref = max(fs[max(0, k - fo.BB_MEMORY) : k])
        assert s.info["f_ref"] == f_ref
        g = np.array(res.trace[k - 1].info["grad"])
        assert s.fun <= f_ref - fo.BB_GAMMA * s.info["alpha"] * (g @ g)
        if not s.info["reset"]:
            assert s.info["alpha_bb"] == (s.info["bb1"])
        assert s.info["trials"][0][0] == s.info["alpha_bb"]


adam_hyper = st.tuples(
    st.floats(1e-3, 0.5),  # lr
    st.floats(0.0, 0.99),  # β₁
    st.floats(0.5, 0.9999),  # β₂
)


@settings(max_examples=1000, deadline=None)
@given(start_points, adam_hyper)
def test_adam_step_obeys_cauchy_schwarz_bound(start, hyper):
    """|Δx_i| ≤ lr·(1−β₁)/(1−β₁ᵗ)·√((1−β₂ᵗ)/(1−β₂))·√(Σ_{j<t} γʲ), γ = β₁²/β₂.

    Derivation: m̂ = (1−β₁)/(1−β₁ᵗ)·Σβ₁^{t−j}g_j and v̂ = (1−β₂)/(1−β₂ᵗ)·Σβ₂^{t−j}g_j²; write
    β₁^{t−j}g_j = (β₁/√β₂)^{t−j}·√β₂^{t−j}g_j and apply Cauchy–Schwarz (ε ≥ 0 only shrinks Δ).
    """
    lr, b1, b2 = hyper
    p, x0 = point_in_domain(*start)
    res = numopt.run("adam", p, x0=x0, lr=lr, beta1=b1, beta2=b2, max_iter=30)
    gamma = b1 * b1 / b2
    for t, (prev, cur) in enumerate(zip(res.trace, res.trace[1:], strict=False), start=1):
        s_t = sum(gamma**j for j in range(t))
        bound = lr * (1 - b1) / (1 - b1**t) * math.sqrt((1 - b2**t) / (1 - b2) * s_t)
        step = np.abs(np.array(cur.x) - np.array(prev.x))
        assert np.all(step <= bound * (1 + 1e-12) + 4 * EPS * np.abs(prev.x))


@settings(max_examples=1000, deadline=None)
@given(start_points, adam_hyper)
def test_adamax_step_bound(start, hyper):
    """|m_t|/u_t ≤ (1−β₁)Σ_{j<t}(β₁/β₂)ʲ because u_t ≥ β₂^{t−j}|g_j| for every j ≤ t."""
    lr, b1, b2 = hyper
    p, x0 = point_in_domain(*start)
    res = numopt.run("adamax", p, x0=x0, lr=lr, beta1=b1, beta2=b2, max_iter=30)
    r = b1 / b2
    for t, (prev, cur) in enumerate(zip(res.trace, res.trace[1:], strict=False), start=1):
        bound = lr / (1 - b1**t) * (1 - b1) * sum(r**j for j in range(t))
        step = np.abs(np.array(cur.x) - np.array(prev.x))
        assert np.all(step <= bound * (1 + 1e-12) + 4 * EPS * np.abs(prev.x))


@settings(max_examples=1000, deadline=None)
@given(start_points, adam_hyper)
def test_amsgrad_and_adagrad_step_sizes_never_increase(start, hyper):
    lr, b1, b2 = hyper
    p, x0 = point_in_domain(*start)
    ams = numopt.run("amsgrad", p, x0=x0, lr=lr, beta1=b1, beta2=b2, max_iter=30)
    ada = numopt.run("adagrad", p, x0=x0, lr=lr, max_iter=30)
    for res, acc in ((ams, "v_max"), (ada, "v")):
        assert_valid_result(res, max_iter=30)
        for prev, cur in zip(res.trace, res.trace[1:], strict=False):
            assert np.all(np.array(cur.info[acc]) >= np.array(prev.info[acc]))
            if prev.info["lr_eff"] is not None:
                lp, lc = np.array(prev.info["lr_eff"]), np.array(cur.info["lr_eff"])
                assert np.all((lc <= lp) | (lp == 0.0))
    for s in ams.trace:
        assert np.all(np.array(s.info["v_max"]) >= np.array(s.info["v"]))
    # √v̂_t ≥ √v_j ≥ √(1 − β₂)|g_j| for j ≤ t ⇒ |m_t|/√v̂_t ≤ (1 − β₁ᵗ)/√(1 − β₂).
    for t, (prev, cur) in enumerate(zip(ams.trace, ams.trace[1:], strict=False), start=1):
        bound = lr * (1 - b1**t) / math.sqrt(1 - b2)
        step = np.abs(np.array(cur.x) - np.array(prev.x))
        assert np.all(step <= bound * (1 + 1e-12) + 4 * EPS * np.abs(prev.x))


@pytest.mark.parametrize("method", METHODS)
@settings(max_examples=1000, deadline=None)
@given(start=start_points)
def test_contract_from_random_start(method, start):
    p0, x0 = point_in_domain(*start)
    p, calls = counting(p0)
    res = numopt.run(method, p, x0=x0, max_iter=15)
    check_contract(res, p0, calls, method, 15)


# --------------------------------------------------------------------------------------
# Failure paths and edge cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_max_iter_is_reported(method):
    res = numopt.run(method, problems.get("rosenbrock"), max_iter=3)
    assert not res.converged and "max_iter=3" in res.message
    assert res.n_iter == 3 and len(res.trace) == 4
    assert_valid_result(res, max_iter=3)


@pytest.mark.parametrize("method", METHODS)
def test_start_at_minimizer_converges_immediately(method):
    p = problems.get("quadratic_bowl")
    res = numopt.run(method, p, x0=[1.0, -0.5])
    assert res.converged and res.n_iter == 0 and len(res.trace) == 1
    assert res.n_fev == 1 and res.n_gev == 1 and res.n_hev == 0


def test_divergence_large_fixed_step():
    """quadratic_ill has λ_max = 50, so lr = 1 multiplies that component by −49 per step."""
    res = numopt.run("gradient_descent", problems.get("quadratic_ill"), step_rule="fixed", lr=1.0)
    assert not res.converged and res.message.startswith("diverged")
    f0 = res.trace[0].fun
    assert res.fun is not None and f0 is not None and res.n_iter <= 5
    assert res.fun - f0 > fo.F_DIVERGE * max(1.0, abs(f0))
    # The test fires at the first iterate that passes the threshold, not later.
    prev = res.trace[-2].fun
    assert prev is not None and prev - f0 <= fo.F_DIVERGE * max(1.0, abs(f0))
    assert_valid_result(res)


def test_divergence_heavy_ball():
    res = numopt.run("momentum", problems.get("quadratic_ill"), lr=0.1, beta=0.9)
    assert not res.converged and res.message.startswith("diverged")
    assert_valid_result(res)


def scaled_quadratic(c: float, diag: list[float], x0: list[float]) -> Problem:
    """f = c·½ Σ dᵢxᵢ² (minimizer 0): the same problem in other units of f when c changes."""
    d = np.asarray(diag, dtype=float)
    return Problem(
        "scaled",
        "scaled",
        "",
        lambda x: float(c * 0.5 * np.sum(d * np.asarray(x, dtype=float) ** 2)),
        d.size,
        (),
        grad=lambda x: c * d * np.asarray(x, dtype=float),
        hess=lambda x: c * np.diag(d),
        x0=tuple(x0),
        minima=(tuple(0.0 for _ in x0),),
    )


@pytest.mark.parametrize("c", [1.0, 1e5, 1e-5])
def test_pure_bb_with_large_f_scale_is_not_flagged_as_diverged(c):
    """Regression: pure BB1 is invariant under f → c·f (the reset step 1/‖g‖ moves x by one
    unit, and the BB steps scale by 1/c). From (1000, 1), f rises 32-fold before it
    converges; with c = 1e5, f(x₀) = 5e10 and that rise passed the old absolute test
    f > 1e12, which reported 'diverged' at iteration 9."""
    ref = numopt.run(
        "barzilai_borwein", scaled_quadratic(1.0, [1, 1000], [1000, 1]), nonmonotone=False
    )
    res = numopt.run(
        "barzilai_borwein",
        scaled_quadratic(c, [1, 1000], [1000, 1]),
        nonmonotone=False,
        gtol=c * 1e-6,
    )
    assert res.converged, res.message
    assert res.n_iter == ref.n_iter == 13
    fs = np.array([s.fun for s in res.trace], dtype=float)
    assert fs.max() / fs[0] > 30  # the transient rise is really there
    # NOTE: c·g is not exact in binary, so the twins differ by roundings that BB's
    # step-length recurrence amplifies; 1e-9 of ‖x₀‖ ≈ 1e-6 covers it.
    np.testing.assert_allclose(xs(res), xs(ref), rtol=0, atol=1e-6)
    assert_valid_result(res, max_iter=1000)


def test_heavy_ball_with_large_f_scale_is_not_flagged_as_diverged():
    """Regression: heavy ball on f = c·½·3.7x² with lr = 1/c, β = 0.9 (lr·L = 3.7 <
    2(1 + β) = 3.8, so it converges). From f(x₀) = 2e11 the first step raises f by a factor 7;
    the old absolute test (f > 1e12) reported 'diverged' at k = 1. The twin with c = 1e-6
    takes the same iterates."""
    x0 = math.sqrt(2e11 / 1.85)
    runs = [
        numopt.run(
            "momentum", scaled_quadratic(c * 3.7, [1.0], [x0]), lr=1 / c, beta=0.9, gtol=c * 1e-6
        )
        for c in (1.0, 1e-6)
    ]
    for res in runs:
        assert res.converged, res.message
        assert res.n_iter == 517
        assert_valid_result(res)
    f0, f1 = runs[0].trace[0].fun, runs[0].trace[1].fun
    assert f0 is not None and f1 is not None and f1 > 7 * f0
    assert f0 == pytest.approx(2e11, rel=1e-12)
    # NOTE: 1/c and c·3.7 round differently; 517 contracting steps keep that at ~1e-14·x₀.
    np.testing.assert_allclose(xs(runs[0]), xs(runs[1]), rtol=0, atol=1e-12 * x0)


def test_divergence_is_detected_at_a_large_f_scale():
    """The relative test still fires when the units of f are large: f(x₀) = 7.5e19, and
    lr·cλ₂ = 50 multiplies x₂ by −49 per step."""
    p = scaled_quadratic(1e18, [1.0, 50.0], [10.0, 1.0])
    res = numopt.run("gradient_descent", p, step_rule="fixed", lr=1.0 / 1e18)
    assert not res.converged and res.message.startswith("diverged: f(x) − f(x0)")
    f0 = res.trace[0].fun
    assert f0 is not None and res.fun is not None and res.fun - f0 > fo.F_DIVERGE * f0
    assert_valid_result(res)


@settings(max_examples=1000, deadline=None)
@given(
    quad_params,
    st.floats(-6.0, 7.0),  # log10 of the f scale c (BB steps 1/(cλ) stay in [1e-10, 1e10])
)
def test_pure_bb_converges_at_any_f_scale(params, log_c):
    lam1, kappa, theta, xc, x0 = params
    xc, x0 = np.array(xc), np.array(x0)
    assume(np.linalg.norm(x0 - xc) > 0.1)
    c = 10.0**log_c
    p = quadratic(c * spd2(lam1, kappa, theta), xc, x0)
    gtol = max(fo.GTOL_MIN, 1e-8 * float(np.linalg.norm(p.grad(x0))))  # type: ignore[misc]
    res = numopt.run("barzilai_borwein", p, nonmonotone=False, gtol=gtol)
    assert res.converged, res.message
    assert_valid_result(res, max_iter=1000)


@settings(max_examples=1000, deadline=None)
@given(
    quad_params,
    st.floats(0.0, 0.95),  # β
    st.floats(0.05, 0.95),  # lr·λ_max as a fraction of the stability limit 2(1 + β)
    st.floats(-10.0, 20.0),  # log10 of the f scale c
)
def test_convergent_heavy_ball_is_never_flagged_as_diverged(params, beta, eta, log_c):
    """Polyak: 0 < lr·λ < 2(1 + β) for every eigenvalue ⇒ convergence; so a 'diverged'
    verdict at any scale of f is false. lr = η·2(1 + β)/(cλ_max)."""
    lam1, kappa, theta, xc, x0 = params
    xc, x0 = np.array(xc), np.array(x0)
    assume(np.linalg.norm(x0 - xc) > 0.1)
    c = 10.0**log_c
    A = c * spd2(lam1, kappa, theta)
    lr = eta * 2 * (1 + beta) / float(np.linalg.eigvalsh(A).max())
    res = numopt.run("momentum", quadratic(A, xc, x0), lr=lr, beta=beta, max_iter=200)
    assert not res.message.startswith("diverged"), res.message
    assert_valid_result(res, max_iter=200)


def _blowup() -> Problem:
    """f = ‖x‖² inside the ball of radius 10, NaN outside."""

    def f(x: Any) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ x) if np.linalg.norm(x) < 10 else math.nan

    return Problem(
        "blowup", "blowup", "", f, 2, (), grad=lambda x: 2 * np.asarray(x), x0=(1.0, 1.0)
    )


@pytest.mark.parametrize("method", ["gradient_descent", "momentum", "nesterov"])
def test_non_finite_values_stop_the_run(method):
    # f = ‖x‖² (λ = 2), lr = 1.6: GD factor 1 − 3.2 = −2.2; heavy ball needs lr·λ < 2(1 + β) = 3;
    # Nesterov needs lr·λ < 2(1 + β)/(1 + 2β) = 1.5. All three grow until f is NaN.
    params: dict[str, Any] = {"lr": 1.6}
    if method == "gradient_descent":
        params["step_rule"] = "fixed"
    else:
        params["beta"] = 0.5
    res = numopt.run(method, _blowup(), **params)
    assert not res.converged and "not finite" in res.message
    assert_valid_result(res)


def test_non_finite_start():
    p = dataclasses.replace(_blowup(), x0=(20.0, 0.0))
    res = numopt.run("adam", p)
    assert not res.converged and "x0" in res.message and res.n_iter == 0
    assert_valid_result(res)


def test_exact_quadratic_reports_negative_curvature():
    res = numopt.run("gradient_descent", problems.get("himmelblau"), step_rule="exact_quadratic")
    assert not res.converged and "pᵀ∇²f p" in res.message
    assert res.n_iter == 0 and res.n_hev == 1
    assert_valid_result(res)


def _wrong_gradient(pid: str) -> Problem:
    """The supplied gradient has the wrong sign, so −'∇f' is an ascent direction."""
    p = problems.get(pid)
    return dataclasses.replace(p, grad=lambda x: -p.grad(x))  # type: ignore[misc]


@pytest.mark.parametrize("rule", ["backtracking", "strong_wolfe"])
def test_line_search_failure_is_reported(rule):
    res = numopt.run("gradient_descent", _wrong_gradient("quadratic_bowl"), step_rule=rule)
    assert not res.converged and "line search failed" in res.message
    assert res.n_iter == 0
    assert_valid_result(res)


def test_bb_stall_is_reported():
    """With an ascent 'gradient' the GLL test accepts only steps below the spacing of x."""
    res = numopt.run("barzilai_borwein", _wrong_gradient("quadratic_bowl"))
    assert not res.converged and res.message.startswith("stalled")
    assert res.n_iter == 0
    assert_valid_result(res)


def test_bb_nonmonotone_failure_is_reported():
    """f is NaN everywhere except at x₀ = 0 and the supplied gradient is (1, 1): every trial
    point −α(1, 1), α = 0.1ʲ/√2 for j < 50, is a nonzero float, so all 50 trials fail."""

    def f(x: Any) -> float:
        return 0.0 if not np.any(x) else math.nan

    p = Problem("nan", "nan", "", f, 2, (), grad=lambda x: np.ones(2), x0=(0.0, 0.0))
    res = numopt.run("barzilai_borwein", p)
    assert not res.converged and "nonmonotone line search failed" in res.message
    assert res.n_fev == 1 + fo.BB_MAX_TRIALS
    assert_valid_result(res)


def test_coordinate_descent_failure_is_reported():
    res = numopt.run("coordinate_descent", _wrong_gradient("quadratic_bowl"))
    assert not res.converged and "line search failed" in res.message
    assert_valid_result(res)


def _steep() -> Problem:
    """f = 1e-300·x₁ but the supplied gradient is (1e160, 0): ‖∇f‖² overflows to inf."""
    return Problem(
        "steep",
        "steep",
        "",
        lambda x: float(1e-300 * x[0]),
        2,
        (),
        grad=lambda x: np.array([1e160, 0.0]),
        hess=lambda x: np.eye(2),
        x0=(0.0, 0.0),
    )


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("gradient_descent", {}),
        ("gradient_descent", {"step_rule": "strong_wolfe"}),
        ("gradient_descent", {"step_rule": "exact_quadratic"}),
        ("barzilai_borwein", {}),
        ("coordinate_descent", {}),
    ],
)
def test_slope_overflow_is_reported_not_raised(method, params):
    """search() rejects a non-finite slope ∇fᵀp; the methods must stop with a message instead."""
    with np.errstate(all="raise"):
        res = numopt.run(method, _steep(), **params)
    assert not res.converged and res.message.startswith("overflow")
    assert res.n_iter == 0
    assert res.trace[0].grad_norm == math.inf
    assert_valid_result(res)


def test_coordinate_descent_skips_a_coordinate_whose_slope_underflows():
    """∂f/∂x₁ = 1e-170 ≠ 0, but g₁·p₁ = −1e-340 underflows to 0: skip it, do not raise."""
    p = Problem(
        "tiny",
        "tiny",
        "",
        lambda x: float(1e-170 * x[0] + (x[1] - 1.0) ** 2),
        2,
        (),
        grad=lambda x: np.array([1e-170, 2.0 * (x[1] - 1.0)]),
        hess=lambda x: np.diag([1.0, 2.0]),
        x0=(0.0, 0.0),
    )
    res = numopt.run("coordinate_descent", p, gtol=fo.GTOL_MIN)
    s1 = res.trace[1]
    assert s1.info["coordinate"] == 0 and s1.info["alpha"] == 0.0
    assert s1.info["direction"] == [0.0, 0.0] and s1.info["newton"] is False
    np.testing.assert_array_equal(s1.x, [0.0, 0.0])
    # k = 2: the exact Newton step on x₂ leaves ‖∇f‖ = 1e-170 ≤ gtol = 1e-150.
    assert res.converged and res.n_iter == 2
    np.testing.assert_array_equal(res.x, [0.0, 1.0])
    assert_valid_result(res)


def test_coordinate_descent_newton_step_below_spacing_of_x_is_skipped():
    """f = ½(x₁ − 10¹⁷ + 3)² + 2x₂² from (10¹⁷, 3). The spacing of floats at 10¹⁷ is 16, so the
    Newton step −3 on x₁ rounds away: the coordinate is skipped (no failed line search), x₂ is
    solved exactly, and then n = 2 unchanged iterations in a row report a stall."""

    def grad(x: Any) -> np.ndarray:
        return np.array([(x[0] - 1e17) + 3.0, 4.0 * x[1]])

    p = Problem(
        "far",
        "far",
        "",
        lambda x: float(0.5 * ((x[0] - 1e17) + 3.0) ** 2 + 2.0 * x[1] ** 2),
        2,
        (),
        grad=grad,
        hess=lambda x: np.diag([1.0, 4.0]),
        x0=(1e17, 3.0),
    )
    res = numopt.run("coordinate_descent", p)
    assert [s.info["alpha"] for s in res.trace[1:]] == [0.0, 1.0, 0.0, 0.0]
    assert res.trace[1].info["direction"] == [0.0, 0.0]
    np.testing.assert_array_equal(res.x, [1e17, 0.0])
    assert not res.converged and res.message.startswith("stalled") and res.n_iter == 4
    assert_valid_result(res)


def _linear_in_x2(c: float) -> Problem:
    """f = (x₁ − 1)² + c·x₂: the second gradient component is the constant c."""
    return Problem(
        "lin",
        "lin",
        "",
        lambda x: float((x[0] - 1.0) ** 2 + c * x[1]),
        2,
        (),
        grad=lambda x: np.array([2.0 * (x[0] - 1.0), c]),
        x0=(0.0, 0.0),
    )


def _x2_closed_form(method: str, n: int, lr: float, b1: float, b2: float) -> np.ndarray:
    """x₂ after t = 0..n steps for a constant gradient c > 0 (independent of c).

    m_t = c(1 − β₁ᵗ). AdaMax: u_t = c, step lr(1 − β₁ᵗ)/(1 − β₁ᵗ) = lr. AMSGrad:
    v_t = c²(1 − β₂ᵗ) increases, so v̂_t = v_t and the step is lr(1 − β₁ᵗ)/√(1 − β₂ᵗ)."""
    t = np.arange(1, n + 1, dtype=float)
    if method == "adamax":
        steps = np.full(n, lr)
    else:
        steps = lr * (1 - b1**t) / np.sqrt(1 - b2**t)
    return -np.concatenate([[0.0], np.cumsum(steps)])


@pytest.mark.parametrize("method", ["adamax", "amsgrad"])
@pytest.mark.parametrize("c", [1.0, 1e-162, 1e-300, 1e-308, 1e-310, 1e160])
def test_epsilon_free_methods_handle_tiny_and_huge_gradients(method, c):
    """Regression: AdaMax formed 1/((1 − β₁ᵗ)u), which overflows for u < 5.6e-309 (x₂ = −inf,
    a false 'diverged'); AMSGrad formed g², which underflows for |g| ≲ 1e-161 (x₂ frozen
    although the paper step is ≈ 3·lr) and overflows for |g| ≳ 1.3e154 (x₂ frozen again).
    Both steps are invariant under g₂ → c·g₂, so x₂ must follow the closed form at every c,
    without a floating-point warning, and every info value must stay finite (strict JSON)."""
    n = 5
    with warnings.catch_warnings(), np.errstate(over="raise", divide="raise", invalid="raise"):
        warnings.simplefilter("error")
        res = numopt.run(method, _linear_in_x2(c), max_iter=n)
    assert res.n_iter == n and "max_iter" in res.message
    h = {"lr": 0.2 if method == "adamax" else 0.1, "b1": 0.9, "b2": 0.999}
    ref = _x2_closed_form(method, n, h["lr"], h["b1"], h["b2"])
    # NOTE: a few roundings per step (8tε), plus, for subnormal c, the absolute rounding of
    # m and u at the smallest subnormal 4.9e-324, relative to c.
    rtol = 8 * n * EPS + 64 * 4.9e-324 / c
    np.testing.assert_allclose(xs(res)[:, 1], ref, rtol=rtol, atol=0)
    for s in res.trace[1:]:
        assert all(math.isfinite(v) for v in s.info["lr_eff"])
    assert_valid_result(res, max_iter=n)


@pytest.mark.parametrize("method", ["adamax", "amsgrad"])
def test_epsilon_free_methods_zero_gradient_coordinate_does_not_move(method):
    """g₂ ≡ 0: u = v̂ = 0 and m = 0 (the 0/0 case is a zero step, not NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = numopt.run(method, _linear_in_x2(0.0), max_iter=5)
    for s in res.trace[1:]:
        assert s.x[1] == 0.0 and s.info["direction"][1] == 0.0 and s.info["lr_eff"][1] == 0.0
    assert res.trace[-1].x[0] != 0.0  # the other coordinate moves
    assert_valid_result(res, max_iter=5)


@pytest.mark.parametrize("method", METHODS)
def test_integral_float_max_iter_is_accepted(method):
    """'--set max_iter=1e3' reaches a method as a float; an integral float means that int."""
    p = problems.get("rosenbrock")
    a = numopt.run(method, p, max_iter=3)
    b = numopt.run(method, p, max_iter=3.0)
    np.testing.assert_array_equal(xs(a), xs(b))
    assert (a.converged, a.n_iter, a.message) == (b.converged, b.n_iter, b.message)
    assert "max_iter=3 " in b.message


@pytest.mark.parametrize("bad", [math.inf, math.nan, 2.5, True, "10", None, 0.0, -1])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_max_iter_raises_value_error(method, bad):
    with pytest.raises(ValueError, match="max_iter"):
        numopt.run(method, problems.get("quadratic_bowl"), max_iter=bad)


@pytest.mark.parametrize(
    ("rule", "null"),
    [("fixed", True), ("exact_quadratic", True), ("backtracking", False), ("strong_wolfe", False)],
)
def test_gradient_descent_alpha0_is_null_without_a_trial_step(rule, null):
    res = numopt.run(
        "gradient_descent", problems.get("quadratic_ill"), step_rule=rule, lr=0.01, max_iter=4
    )
    assert res.trace[0].info["alpha0"] is None
    assert all((s.info["alpha0"] is None) == null for s in res.trace[1:])


# --------------------------------------------------------------------------------------
# Gradient descent: the line searches do not depend on the units of f (audit regressions)
# --------------------------------------------------------------------------------------


def _scaled(p: Problem, c: float) -> Problem:
    """c·f, with ∇f and ∇²f scaled to match."""
    f, g, h = p.f, p.grad, p.hess
    assert g is not None and h is not None
    return dataclasses.replace(
        p,
        f=lambda x: c * f(x),
        grad=lambda x: c * np.asarray(g(x)),
        hess=lambda x: c * np.asarray(h(x)),
    )


def _steps(res: Result) -> np.ndarray:
    return np.array([s.step_size for s in res.trace[1:]], dtype=float)


@pytest.mark.parametrize("c", [1e3, 1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("rule", ["backtracking", "strong_wolfe"])
def test_gradient_descent_line_searches_are_invariant_under_f_scaling(rule, c):
    """Regression: with α₀ = 1 at k = 1 and the absolute α_max = 1e3, c·quadratic_bowl
    (gtol = c·1e-6) took 13/2515/1e5+ backtracking iterations for c = 1/1e-4/1e-6, and strong
    Wolfe failed at k = 1 for c = 1e-6. With α₀ = 1/‖∇f(x₀)‖ and eq. 3.60/3.61 every α scales
    with 1/c, so the run is the same for every c."""
    p = problems.get("quadratic_bowl")
    ref = numopt.run("gradient_descent", p, step_rule=rule)
    pc, calls = counting(_scaled(p, c))
    res = numopt.run("gradient_descent", pc, step_rule=rule, gtol=c * 1e-6)
    check_contract(res, pc, calls, "gradient_descent", 5000)
    assert ref.converged, ref.message
    assert res.converged, res.message
    assert (res.n_iter, res.n_fev, res.n_gev) == (ref.n_iter, ref.n_fev, ref.n_gev)
    # NOTE: c is not a power of 2, so every product with c rounds, and α_{3.61} divides the
    # difference f(x_{k−2}) − f(x_{k−1}), whose relative rounding error grows like ε|f|/|Δf| as
    # the decrease shrinks. Measured: max |Δx| = 5.7e-12, max relative |Δα| = 5.9e-6.
    np.testing.assert_allclose(xs(res), xs(ref), rtol=0, atol=1e-10)
    np.testing.assert_allclose(c * _steps(res), _steps(ref), rtol=1e-4)


@settings(max_examples=1000, deadline=None)
@given(quad_params, st.integers(-60, 30), st.sampled_from(["backtracking", "strong_wolfe"]))
def test_gradient_descent_trace_is_bitwise_invariant_under_power_of_two_scaling(params, m, rule):
    """f → 2ᵐf scales f, ∇f and every α exactly (no rounding, no under/overflow here), so the
    iterates must be bitwise identical and every α must be exactly 2⁻ᵐ times the original."""
    lam1, kappa, theta, c, x0 = params
    c, x0 = np.array(c), np.array(x0)
    # NOTE: exactness needs normal floats: Hypothesis also draws subnormal-sized θ, c and x₀
    # (e.g. θ = 1.1e-308 makes A₁₂ subnormal, and 2ᵐ·A₁₂ then rounds). Exclude tiny nonzeros.
    entries = np.concatenate([[theta], c, x0, x0 - c])
    assume(bool(np.all((entries == 0.0) | (np.abs(entries) >= 1e-100))))
    p = quadratic(spd2(lam1, kappa, theta), c, x0)
    s = 2.0**m
    ref = numopt.run("gradient_descent", p, step_rule=rule, gtol=1e-6, max_iter=300)
    res = numopt.run("gradient_descent", _scaled(p, s), step_rule=rule, gtol=s * 1e-6, max_iter=300)
    assert_valid_result(res, max_iter=300)
    assert (res.n_iter, res.converged, res.n_fev, res.n_gev) == (
        ref.n_iter,
        ref.converged,
        ref.n_fev,
        ref.n_gev,
    )
    np.testing.assert_array_equal(xs(res), xs(ref))
    np.testing.assert_array_equal(s * _steps(res), _steps(ref))


@pytest.mark.parametrize("rule", ["backtracking", "strong_wolfe"])
def test_gradient_descent_small_curvature_quadratic(rule):
    """Audit repro: f = 1e-5((x₁ − 3)² + 2(x₂ + 1)²), κ = 2, x₀ = 0. Before the fix strong Wolfe
    failed at k = 1 ('phi still decreases at alpha_max=1e+03', but the curvature condition
    needs α ≥ 0.1/λ ≈ 2.5e3), and backtracking took α = 1 (Cauchy step ≈ 3e4) until max_iter."""
    a = 1e-5
    x_star = np.array([3.0, -1.0])
    p = quadratic(np.diag([2 * a, 4 * a]), x_star, np.zeros(2))
    unit = quadratic(np.diag([2.0, 4.0]), x_star, np.zeros(2))
    assert p.grad is not None
    gtol = 1e-6 * float(np.linalg.norm(p.grad(np.zeros(2))))
    res = numopt.run("gradient_descent", p, step_rule=rule, gtol=gtol)
    assert_valid_result(res, max_iter=5000)
    assert res.converged, res.message
    # The same run as on the problem with a = 1 (f → f/a).
    assert res.n_iter == numopt.run("gradient_descent", unit, step_rule=rule, gtol=gtol / a).n_iter
    # ‖x − x*‖ ≤ ‖A⁻¹∇f(x)‖ ≤ gtol/λ_min, λ_min = 2a; SciPy CG (strong Wolfe search) agrees.
    assert np.linalg.norm(res.x - x_star) <= gtol / (2 * a)
    ref = optimize.minimize(p.f, np.zeros(2), jac=p.grad, method="CG", options={"gtol": 1e-14})
    assert np.linalg.norm(res.x - ref.x) <= 2 * gtol / (2 * a)


def test_gradient_descent_far_start_on_flat_quadratic():
    """x* is ≈ 1.4e8 away and λ_min = 2e-20: the first trial α₀ = 1/‖∇f(x₀)‖ (a unit move) is
    ≈ 1e8 times shorter than the Cauchy step. Strong Wolfe expands to α_max = 2²⁰α₀ while φ
    still decreases and takes that step (it satisfies the Armijo condition); backtracking can
    only shrink α₀, so it relies on the doubling of α_{3.61} after a too-short step (with eq.
    3.60 alone α would stay ≈ α₀ while ∇f hardly changes)."""
    a = 1e-20
    x_star = np.array([1e8, -1e8])
    p = quadratic(np.diag([2 * a, 4 * a]), x_star, np.zeros(2))
    assert p.grad is not None
    g0 = np.asarray(p.grad(np.zeros(2)))
    gtol = 1e-6 * float(np.linalg.norm(g0))
    sw = numopt.run("gradient_descent", p, step_rule="strong_wolfe", gtol=gtol)
    assert_valid_result(sw, max_iter=5000)
    assert sw.converged, sw.message
    first = sw.trace[1]
    alpha = first.info["alpha"]
    assert alpha == fo.ALPHA_MAX_RATIO * first.info["alpha0"] == first.info["trials"][-1][0]
    assert first.fun is not None and sw.trace[0].fun is not None
    assert first.fun <= sw.trace[0].fun - 1e-4 * alpha * float(g0 @ g0)  # N&W eq. 3.4
    bt = numopt.run("gradient_descent", p, step_rule="backtracking", gtol=gtol)
    assert_valid_result(bt, max_iter=5000)
    assert bt.converged, bt.message
    alphas = _steps(bt)
    assert alphas.max() >= 1e7 * alphas[0]
    for res in (sw, bt):
        assert np.linalg.norm(res.x - x_star) <= gtol / (2 * a)  # gtol/λ_min


@pytest.mark.parametrize("rule", ["backtracking", "strong_wolfe"])
def test_gradient_descent_unbounded_linear_function_is_reported(rule):
    """f = −x₁ − 2x₂ is unbounded below. Backtracking accepts every α₀; f falls exactly like
    its linear model, so α_{3.61} doubles the step each iteration until a trial overflows to
    f = −∞, which is reported as divergence (not as a line-search failure)."""
    p = Problem(
        "linear",
        "linear",
        "",
        lambda x: float(-x[0] - 2.0 * x[1]),
        2,
        (),
        grad=lambda x: np.array([-1.0, -2.0]),
        x0=(0.0, 0.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = numopt.run("gradient_descent", p, step_rule=rule)
    assert_valid_result(res, max_iter=5000)
    assert not res.converged
    assert res.message.startswith("diverged") and "unbounded below" in res.message
    assert res.fun is not None and res.fun < -1e300
    if rule == "backtracking":
        alphas = _steps(res)[:40]
        # NOTE: Δf = α‖g‖² up to the rounding of f(x) = −x₁ − 2x₂ (a few ε relative).
        np.testing.assert_allclose(alphas[1:] / alphas[:-1], 2.0, rtol=1e-12)


def test_rmsprop_converges_or_hovers_depending_on_lr():
    """The registry's order string: with lr = 0.1 RMSprop converges on quadratic_bowl; with
    lr = 0.01 the normalized steps do not shrink and x stays within ≈ lr/2 of x*."""
    p = problems.get("quadratic_bowl")
    x_star = np.array(p.minima[0])
    assert numopt.run("rmsprop", p, lr=0.1).converged
    res = numopt.run("rmsprop", p, lr=0.01)
    assert not res.converged and "max_iter" in res.message
    tail = xs(res)[-200:] - x_star
    assert np.abs(tail).max() <= 0.01
    assert min(float(s.grad_norm or 0.0) for s in res.trace[-200:]) > 1e-2
    assert "depending on lr" in get_method("rmsprop").order


@settings(max_examples=1000, deadline=None)
@given(quad_params)
def test_nesterov_constant_momentum_rate(params):
    """Nesterov (2004), §2.2.1, constant step scheme III: lr = 1/L, μ = (√κ − 1)/(√κ + 1) ⇒
    f(x_k) − f* ≤ (f(x₀) − f* + (σ/2)‖x₀ − x*‖²)(1 − 1/√κ)ᵏ (the registry's order string)."""
    lam1, kappa, theta, xc, x0 = params
    xc, x0 = np.array(xc), np.array(x0)
    assume(np.linalg.norm(x0 - xc) > 0.1)
    A = spd2(lam1, kappa, theta)
    sigma, L = (float(v) for v in np.linalg.eigvalsh(A))
    k_true = L / sigma
    mu = (math.sqrt(k_true) - 1) / (math.sqrt(k_true) + 1)
    p = quadratic(A, xc, x0)
    res = numopt.run("nesterov", p, lr=1 / L, beta=mu, gtol=1e-9, max_iter=300)
    C = float(p.f(x0)) + 0.5 * sigma * float((x0 - xc) @ (x0 - xc))
    for s in res.trace:
        assert s.fun is not None
        # NOTE: f = ½dᵀAd from d = x − c has absolute error ~ L·‖d‖·ε(‖x‖ + ‖c‖).
        d = np.linalg.norm(np.array(s.x) - xc)
        noise = 8 * L * d * EPS * (np.linalg.norm(s.x) + np.linalg.norm(xc) + 1)
        assert s.fun <= C * (1 - 1 / math.sqrt(k_true)) ** s.k * (1 + 1e-12) + noise


def test_bare_callable_uses_central_differences():
    calls = []

    def f(x: Any) -> float:
        calls.append(1)
        return float((x[0] - 1) ** 2 + 4 * (x[1] + 2) ** 2)

    res = numopt.minimize(f, x0=[3.0, 1.0], method="barzilai_borwein")
    assert res.converged
    np.testing.assert_allclose(res.x, [1.0, -2.0], atol=1e-6)
    # f(x₀), one f per trial step, and 2n = 4 evaluations of f per central-difference gradient.
    n_trials = sum(len(s.info["trials"]) for s in res.trace[1:])
    assert res.n_fev == len(calls) == 1 + n_trials + 4 * res.n_gev


def test_coordinate_descent_finite_difference_hessian():
    def f(x: Any) -> float:
        return float(3 * x[0] ** 2 + 2 * x[0] * x[1] + 2 * x[1] ** 2)

    res = numopt.minimize(f, x0=[1.0, 1.0], method="coordinate_descent", gtol=1e-8)
    assert res.converged
    np.testing.assert_allclose(res.x, [0.0, 0.0], atol=1e-8)
    assert res.n_hev == res.n_iter


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("gradient_descent", {"step_rule": "newton"}),
        ("gradient_descent", {"lr": 0.0}),
        ("gradient_descent", {"gtol": -1.0}),
        ("barzilai_borwein", {"gtol": 0.0}),
        ("gradient_descent", {"max_iter": 0}),
        ("barzilai_borwein", {"variant": "bb3"}),
        ("momentum", {"beta": 1.0}),
        ("nesterov", {"lr": -0.1}),
        ("adagrad", {"eps": 0.0}),
        ("rmsprop", {"rho": 1.0}),
        ("adadelta", {"rho": -0.1}),
        ("adam", {"beta2": 1.0}),
        ("adamw", {"weight_decay": -1e-3}),
        ("adamax", {"beta1": 1.5}),
        ("nadam", {"eps": math.nan}),
        ("amsgrad", {"lr": math.inf}),
    ],
)
def test_invalid_parameters_raise(method, params):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("quadratic_bowl"), **params)


def test_bare_callable_needs_x0():
    with pytest.raises(ValueError):
        numopt.run("adam", lambda x: float(x @ x))


def test_inputs_are_not_mutated():
    x0 = np.array([-1.2, 1.0])
    for method in METHODS:
        numopt.run(method, problems.get("rosenbrock"), x0=x0, max_iter=5)
        np.testing.assert_array_equal(x0, [-1.2, 1.0])


# --------------------------------------------------------------------------------------
# Numerical breakdown never raises (web-port regressions)
# --------------------------------------------------------------------------------------


def _tiny_gradient_problem(c: float) -> Problem:
    """f(x) = a·Σxᵢ + ½c‖x‖² with a = 1e-148: ‖∇f‖ ≈ 1.4e-148 > GTOL_MIN, and each BB step moves
    ∇f by y ≈ c·1e-138 per entry, so yᵀy underflows to 0 for c ≲ 1e-24 while sᵀy > 0."""
    a = 1e-148
    return Problem(
        id="tiny_gradient",
        name="tiny gradient",
        latex="",
        f=lambda x: float(a * np.sum(x) + 0.5 * c * float(np.dot(x, x))),
        grad=lambda x: a + c * np.asarray(x, dtype=float),
        dim=2,
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        x0=np.zeros(2),
    )


def _exact_bb(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(sᵀs/sᵀy, sᵀy/yᵀy) in exact rational arithmetic, rounded once to float."""
    sf, yf = [Fraction(float(v)) for v in s], [Fraction(float(v)) for v in y]
    ss = sum(u * u for u in sf)
    sy = sum(u * v for u, v in zip(sf, yf, strict=True))
    yy = sum(v * v for v in yf)
    return float(ss / sy), float(sy / yy)


@pytest.mark.parametrize("variant", ["bb1", "bb2"])
@pytest.mark.parametrize("c", [1e-24, 1e-25, 3e-25])
def test_barzilai_borwein_yty_underflow_does_not_raise(variant, c):
    """Regression: yᵀy underflowed to 0 with sᵀy > 0 and sᵀy / yᵀy raised ZeroDivisionError.
    The recorded BB values must equal the exact rational quotients from the trace's s and y."""
    p = _tiny_gradient_problem(c)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = numopt.run("barzilai_borwein", p, variant=variant, gtol=1e-150, max_iter=6)
    assert_valid_result(res, max_iter=6)
    checked = 0
    for prev2, prev, step in zip(res.trace, res.trace[1:], res.trace[2:], strict=False):
        s = prev.x - prev2.x
        y = np.asarray(prev.info["grad"]) - np.asarray(prev2.info["grad"])
        assert float(np.dot(y, y)) == 0.0  # the unscaled yᵀy underflows in this regime
        if step.info["bb1"] is None:
            continue
        bb1, bb2 = _exact_bb(s, y)
        # 4 roundings (two inner products of 2 terms, a quotient): 8ε is a safe bound.
        assert step.info["bb1"] == pytest.approx(bb1, rel=8 * EPS)
        assert step.info["bb2"] == pytest.approx(bb2, rel=8 * EPS)
        checked += 1
    assert checked >= 1


@settings(max_examples=1000, deadline=None)
@given(
    n=st.integers(1, 4),
    seed=st.integers(0, 2**32 - 1),
    e_s=st.integers(-1000, 1000),
    e_y=st.integers(-1000, 1000),
)
def test_bb_steps_scale_exactly_and_match_the_exact_quotients(n, seed, e_s, e_y):
    """_bb_steps(2^e_s·s, 2^e_y·y) = 2^(e_s−e_y)·_bb_steps(s, y) for any exponents (the power-of-2
    scaling is exact), and with all sᵢyᵢ > 0 (no cancellation) both quotients are within 8ε of
    the exact rational values. Unscaled, the products under- or overflow for |e| ≳ 500."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)
    y = s * rng.uniform(0.1, 10.0, n)  # sᵢyᵢ > 0
    bb1, bb2 = fo._bb_steps(s, y)
    assert bb1 is not None and bb2 is not None
    ref1, ref2 = _exact_bb(s, y)
    assert bb1 == pytest.approx(ref1, rel=8 * EPS)
    assert bb2 == pytest.approx(ref2, rel=8 * EPS)
    # Unscaled formula on well-scaled data: bit-for-bit equal (fixtures do not change).
    sy, ss, yy = float(s @ y), float(s @ s), float(y @ y)
    assert (bb1, bb2) == (ss / sy, sy / yy)
    big1, big2 = fo._bb_steps(np.ldexp(s, e_s), np.ldexp(y, e_y))
    with np.errstate(over="ignore", under="ignore"):
        want1 = float(np.ldexp(bb1, e_s - e_y))
        want2 = float(np.ldexp(bb2, e_s - e_y))
    if 1e-300 < abs(want1) < 1e300 and 1e-300 < abs(want2) < 1e300:
        assert (big1, big2) == (want1, want2)
    else:  # the quotient leaves the normal range: overflow gives +inf, never an exception
        assert big1 is not None and big2 is not None and big1 >= 0.0 and big2 >= 0.0


def test_bb_steps_undefined_cases():
    assert fo._bb_steps(np.array([1.0, 0.0]), np.zeros(2)) == (None, None)  # y = 0
    assert fo._bb_steps(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == (None, None)  # sᵀy < 0
    assert fo._bb_steps(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == (None, None)  # sᵀy = 0


@settings(max_examples=1000, deadline=None)
@given(
    alpha=st.floats(5e-324, 1e10),
    f_alpha=st.floats(-1e300, 1e300) | st.just(math.inf) | st.just(math.nan),
    f0=st.floats(-1e300, 1e300),
    gg=st.floats(1e-300, 1e300),
)
def test_bb_sigma_is_always_in_the_safeguard_interval(alpha, f_alpha, f0, gg):
    """Regression: α² or 2cα could underflow to 0 and raise ZeroDivisionError. σ must always
    lie in [σ₁, σ₂] = [0.1, 0.5] and nothing may raise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sigma = fo._bb_sigma(alpha, f_alpha, f0, gg)
    assert fo.BB_SIGMA1 <= sigma <= fo.BB_SIGMA2


def test_bb_sigma_interpolant_minimizer():
    """φ(α) = 1 − α + α² (gᵀg = 1, c = 1): t* = 1/2, so σ = t*/α = 1/(2α), clipped to [0.1, 0.5]."""
    for alpha, want in [(2.0, 0.25), (1.25, 0.4), (10.0, 0.1), (0.5, 0.5), (1e-200, 0.5)]:
        assert fo._bb_sigma(alpha, 1.0 - alpha + alpha * alpha, 1.0, 1.0) == want


@pytest.mark.parametrize("pid", ["beale", "booth", "rosenbrock"])
def test_gradient_descent_strong_wolfe_zoom_breakdown_is_reported(pid):
    """Regression: with f scaled by 2⁴⁹⁶ (as in the conjugate-gradient audit) the zoom's
    brackets fall below ≈ 1.5e-162, where h² underflows, and on beale ZeroDivisionError escaped
    from the strong Wolfe search. The run must end with a message and exact counts."""
    p = problems.get(pid)
    pc, calls = counting(_scaled(p, 2.0**496))
    res = numopt.run("gradient_descent", pc, step_rule="strong_wolfe", max_iter=5000)
    check_contract(res, pc, calls, "gradient_descent", 5000)
    if pid == "beale":
        assert not res.converged
        assert "broke down" in res.message and "ZeroDivisionError" in res.message, res.message
        assert "rescale f" in res.message
