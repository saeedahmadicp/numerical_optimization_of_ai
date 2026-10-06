"""Tests for the finite-sum problem library (numopt.problems.stochastic).

Oracles:
    * loss values: SciPy's ``special.log_expit`` (logistic) and ``special.huber`` (Huber), and
      the plain formula ½r² (squared loss);
    * derivatives: the 5-point central difference with h = ε^{1/5}·max(|w|, 1). Its truncation
      error h⁴|f⁽⁵⁾|/30 and rounding error ε|f|/h are both ≤ 1e-9·(scale) on these problems,
      so a 1e-7 relative tolerance has two orders of margin;
    * minimizers: ``scipy.linalg.lstsq`` with the QR-based gelsy driver (numopt uses NumPy's
      SVD-based gelsd) and ``scipy.optimize.minimize`` (trust-exact);
    * data: the documented Rng draw order, replayed here independently.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as sla
from scipy import optimize, special

from numopt import problems
from numopt.core.rng import Rng
from numopt.problems.stochastic import FiniteSumProblem

REQUIRED = ("linreg_2d", "logreg_2d", "ill_conditioned_ls", "huber_regression_2d")
ALL = [p.id for p in problems.list_problems("stochastic")]
EPS = float(np.finfo(float).eps)
_H5 = EPS**0.2


def _get(pid: str) -> FiniteSumProblem:
    p = problems.get(pid)
    assert isinstance(p, FiniteSumProblem)
    return p


def _fd5(fun, x: np.ndarray, h_rel: float = _H5) -> np.ndarray:
    cols = []
    for i in range(x.size):
        h = h_rel * max(abs(x[i]), 1.0)
        e = np.zeros_like(x)
        e[i] = h
        d = (
            -np.asarray(fun(x + 2 * e))
            + 8 * np.asarray(fun(x + e))
            - 8 * np.asarray(fun(x - e))
            + np.asarray(fun(x - 2 * e))
        )
        cols.append(d / (12 * h))
    return np.stack(cols, axis=-1)


def _phi_oracle(p: FiniteSumProblem, z: np.ndarray) -> np.ndarray:
    """Per-sample losses from SciPy / the textbook formula (independent of numopt)."""
    y = np.asarray(p.y)
    if p.loss == "squared":
        return 0.5 * (z - y) ** 2
    if p.loss == "logistic":
        # −[y log σ(z) + (1 − y) log σ(−z)]
        return -(y * special.log_expit(z) + (1 - y) * special.log_expit(-z))
    return special.huber(p.huber_delta, z - y)


def _f_oracle(p: FiniteSumProblem, w: np.ndarray) -> float:
    z = np.asarray(p.X) @ w
    return float(np.mean(_phi_oracle(p, z)) + 0.5 * p.l2 * w @ w)


def _grad_floor(p: FiniteSumProblem, w: np.ndarray) -> float:
    """Worst-case rounding bound ~4γ_n·mean|terms| of the averaged gradient (Higham 2002, ch. 4)."""
    terms = np.abs(p.grad_samples(w, np.arange(p.n_samples)) - p.l2 * w)  # |φ'ᵢ aᵢ|, (n, d)
    n = p.n_samples
    return 4 * n * EPS * (float(terms.sum(axis=0).max()) / n + p.l2 * float(np.abs(w).max()))


def _away_from_kinks(p: FiniteSumProblem, w: np.ndarray, margin: float) -> bool:
    """Huber: no residual within ``margin`` of ±δ (the Hessian is constant around w)."""
    if p.loss != "huber":
        return True
    r = np.abs(np.asarray(p.X) @ w - np.asarray(p.y))
    return bool(np.min(np.abs(r - p.huber_delta)) > margin)


# --------------------------------------------------------------------------------------
# Metadata and data
# --------------------------------------------------------------------------------------


def test_required_ids_and_metadata():
    for pid in REQUIRED:
        p = _get(pid)
        assert problems.kind_of(pid) == "stochastic" and p.kind == "stochastic"
        assert p.dim == 2 and p.n_samples == 200
        X, y = p.data
        assert X.shape == (200, 2) and y.shape == (200,)
        assert not X.flags.writeable and not y.flags.writeable
        assert len(p.x0) == 2 and len(p.domain) == 2 and len(p.minima) == 1
        for (lo, hi), v, m in zip(p.domain, p.x0, p.minima[0], strict=True):
            assert lo < v < hi and lo < m < hi, "x0 and the minimizer must lie in the plot box"
        assert "finite-sum" in p.tags
        for key in ("seed", "true_w", "f_min", "L", "L_max", "mu", "model"):
            assert key in p.extra
        d = p.to_dict()
        json.dumps(d, allow_nan=False)
        assert d["kind"] == "stochastic" and d["n_samples"] == 200
        assert len(d["X"]) == 200 and len(d["y"]) == 200 and d["loss"] == p.loss
    assert _get("logreg_2d").l2 == 1e-2
    assert set(np.unique(_get("logreg_2d").y)) == {0.0, 1.0}


def test_data_follows_documented_rng_order():
    # linreg_2d: x = U(-1, 3), y = 1 + 2x + N(0, 0.5²), interleaved per sample, Rng(11).
    rng = Rng(11)
    x, y = [], []
    for _ in range(200):
        xi = rng.uniform(-1, 3)
        x.append(xi)
        y.append(1 + 2 * xi + rng.normal(0, 0.5))
    p = _get("linreg_2d")
    assert_array_equal(p.X[:, 0], 1.0)
    assert_array_equal(p.X[:, 1], x)
    assert_array_equal(p.y, y)

    # logreg_2d: x = U(-3, 3), y = [random() < σ(0.5 + 2.5x)], Rng(12).
    rng = Rng(12)
    x, y = [], []
    for _ in range(200):
        xi = rng.uniform(-3, 3)
        x.append(xi)
        y.append(float(rng.random() < 1 / (1 + math.exp(-(0.5 + 2.5 * xi)))))
    p = _get("logreg_2d")
    assert_array_equal(p.X[:, 1], x)
    assert_array_equal(p.y, y)

    # ill_conditioned_ls: u, 30v, y = u + 0.5·30v + N(0, 0.5²), Rng(13).
    rng = Rng(13)
    rows, y = [], []
    for _ in range(200):
        u, v = rng.normal(), 30 * rng.normal()
        rows.append((u, v))
        y.append(1.0 * u + 0.5 * v + rng.normal(0, 0.5))
    p = _get("ill_conditioned_ls")
    assert_array_equal(p.X, rows)
    assert_array_equal(p.y, y)
    col_scale = np.std(p.X, axis=0)
    assert 20 < col_scale[1] / col_scale[0] < 45

    # huber_regression_2d: x, e, u, o always drawn; outlier iff u < 0.1, Rng(14).
    rng = Rng(14)
    x, y, out = [], [], []
    for i in range(200):
        xi, e, u, o = rng.uniform(-1, 3), rng.normal(0, 0.3), rng.random(), rng.uniform(5, 15)
        x.append(xi)
        y.append(1 + 2 * xi + e + (o if u < 0.1 else 0.0))
        if u < 0.1:
            out.append(i)
    p = _get("huber_regression_2d")
    assert_array_equal(p.X[:, 1], x)
    assert_array_equal(p.y, y)
    assert p.extra["outliers"] == out and 10 <= len(out) <= 35


# --------------------------------------------------------------------------------------
# Values and derivatives
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ALL)
def test_loss_matches_scipy_oracle(pid):
    p = _get(pid)
    for w in (np.array(p.x0), np.array(p.minima[0]), np.array([0.3, -0.7])):
        assert_allclose(p.f(w), _f_oracle(p, w), rtol=1e-13, atol=0)
    assert_allclose(p.extra["f_min"], p.f(np.array(p.minima[0])), rtol=0, atol=0)


@pytest.mark.parametrize("pid", ALL)
def test_derivatives_against_finite_differences(pid):
    p = _get(pid)
    lo = np.array([d[0] for d in p.domain])
    hi = np.array([d[1] for d in p.domain])
    pts = [np.array(p.x0), np.array(p.minima[0])]
    pts += [lo + (hi - lo) * np.array(t) for t in ((0.2, 0.7), (0.9, 0.1), (0.55, 0.45))]
    # NOTE: Huber is piecewise quadratic, so inside one piece the stencil has no truncation
    # error at all; h = 1e-5·max(|w|, 1) keeps the rounding error ε|f|/h ≈ 1e-10 and makes the
    # stencil reach (2h‖aᵢ‖₁ ≈ 4e-4 in the residuals) small enough to avoid the dense kinks.
    h_rel = 1e-5 if p.loss == "huber" else _H5
    checked = 0
    for w in pts:
        H = p.hess(w)
        assert_array_equal(H, H.T)
        # A Huber kink |rᵢ| = δ inside the stencil reach breaks the smoothness that the FD
        # error bound needs, so such points are skipped (and at least 3 must remain).
        reach = 2 * h_rel * max(np.abs(w).max(), 1.0) * np.abs(np.asarray(p.X)).sum(axis=1).max()
        if not _away_from_kinks(p, w, margin=reach):
            continue
        checked += 1
        g = p.grad(w)
        scale = 1 + abs(p.f(w)) + np.abs(g).max()
        assert_allclose(g, _fd5(p.f, w, h_rel), rtol=1e-7, atol=1e-7 * scale)
        Hfd = _fd5(p.grad, w, h_rel)
        assert_allclose(H, Hfd, rtol=1e-7, atol=1e-7 * (1 + np.abs(H).max()))
    assert checked >= 3


@pytest.mark.parametrize("pid", ALL)
def test_per_sample_gradients(pid):
    p = _get(pid)
    w = np.array(p.x0) * 0.5 + 0.1
    idx = np.array([5, 0, 199, 5, 17])
    G = p.grad_samples(w, idx)
    assert G.shape == (5, 2)
    A = np.asarray(p.X)
    for row, i in zip(G, idx, strict=True):

        def fi(v, i=i):
            z = np.atleast_1d(A[i] @ v)
            sub = FiniteSumProblem(
                id="one", name="", latex="", X=A[i : i + 1], y=np.asarray(p.y)[i : i + 1],
                loss=p.loss, domain=p.domain, x0=p.x0, l2=p.l2, huber_delta=p.huber_delta,
            )  # fmt: skip
            return float(_phi_oracle(sub, z)[0] + 0.5 * p.l2 * v @ v)

        assert_allclose(row, _fd5(fi, w), rtol=1e-7, atol=1e-7 * (1 + np.abs(row).max()))
    # The mini-batch gradient is the mean of the rows (duplicates counted twice).
    assert_allclose(p.grad_batch(w, idx), G.mean(axis=0), rtol=1e-14, atol=1e-14)
    # The full gradient is the mini-batch gradient over all samples, bit for bit.
    assert_array_equal(p.grad(w), p.grad_batch(w, np.arange(p.n_samples)))


@settings(max_examples=300, deadline=None)
@given(
    st.sampled_from(ALL),
    st.permutations(list(range(200))),
    st.integers(1, 200),
    arrays(np.float64, 2, elements=st.floats(-5, 5)),
)
def test_grad_batch_is_order_independent(pid, perm, b, w):
    p = _get(pid)
    idx = np.array(perm[:b])
    assert_array_equal(p.grad_batch(w, idx), p.grad_batch(w, idx[::-1]))
    assert_array_equal(p.grad_batch(w, idx), p.grad_batch(w, np.sort(idx)))


def test_grad_batch_rejects_bad_indices():
    p = _get("linreg_2d")
    w = np.zeros(2)
    for bad in ([], [200], [-1]):
        with pytest.raises(ValueError):
            p.grad_batch(w, bad)
        with pytest.raises(ValueError):
            p.grad_samples(w, bad)


@pytest.mark.parametrize("pid", ALL)
def test_contour_grid_evaluation(pid):
    p = _get(pid)
    W = np.stack(np.meshgrid(np.linspace(*p.domain[0], 5), np.linspace(*p.domain[1], 4)))
    F = p.f(W)
    assert F.shape == (4, 5)
    for i in range(4):
        for j in range(5):
            assert_allclose(F[i, j], p.f(W[:, i, j]), rtol=1e-14, atol=0)


def test_logistic_is_overflow_free():
    p = _get("logreg_2d")
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow to 0 is fine
        for w in (np.array([800.0, 800.0]), np.array([-800.0, 900.0])):
            fw = p.f(w)
            assert math.isfinite(fw)
            assert_allclose(fw, _f_oracle(p, w), rtol=1e-13)
            assert np.all(np.isfinite(p.grad(w))) and np.all(np.isfinite(p.hess(w)))


# --------------------------------------------------------------------------------------
# Minimizers and constants
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ["linreg_2d", "ill_conditioned_ls"])
def test_least_squares_minimizer_matches_scipy(pid):
    p = _get(pid)
    w_ref, *_ = sla.lstsq(np.asarray(p.X), np.asarray(p.y), lapack_driver="gelsy")
    kappa = np.linalg.cond(np.asarray(p.X))
    # NOTE: two backward-stable LS solvers agree to ~κ(A)·ε for small residuals (Higham 2002,
    # ch. 20);
    # κ(A) ≈ 33 for ill_conditioned_ls, so 1e-12 leaves a margin of ~100.
    assert_allclose(p.minima[0], w_ref, rtol=1e-12, atol=1e-12)
    assert kappa < 100
    w = np.array(p.minima[0])
    assert np.linalg.norm(p.grad(w)) <= _grad_floor(p, w)


@pytest.mark.parametrize("pid", ["logreg_2d", "huber_regression_2d"])
def test_newton_minimizer_matches_scipy(pid):
    p = _get(pid)
    w = np.array(p.minima[0])
    assert np.linalg.norm(p.grad(w)) <= _grad_floor(p, w)
    ref = optimize.minimize(
        lambda v: _f_oracle(p, v), np.zeros(2), jac=p.grad, hess=p.hess, method="trust-exact",
        options={"gtol": 1e-12},
    )  # fmt: skip
    # trust-exact stops at ‖∇f‖ ≈ 1e-11 ("precision loss" in f), so polish its answer with
    # MINPACK's hybrd on ∇f(w) = 0, which does not look at f.
    root = optimize.root(p.grad, ref.x, jac=p.hess, method="hybr", tol=1e-15)
    lam_min = np.linalg.eigvalsh(p.hess(w))[0]
    assert lam_min > 0.05
    # Two points whose gradients are at the rounding floor differ by ≲ 2·floor/λ_min.
    tol = 2 * (_grad_floor(p, w) + _grad_floor(p, root.x)) / lam_min
    assert_allclose(w, root.x, rtol=0, atol=tol)
    assert_allclose(w, ref.x, rtol=0, atol=1e-8)  # the unpolished oracle, at its own accuracy
    assert p.f(w) <= _f_oracle(p, ref.x) + 1e-15


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(ALL), arrays(np.float64, 2, elements=st.floats(-10, 10)))
def test_minimum_is_global(pid, w):
    # All four objectives are convex, so the stationary point is the global minimum.
    p = _get(pid)
    assert p.f(w) >= p.extra["f_min"] - 1e-15 * (1 + abs(p.extra["f_min"]))


@settings(max_examples=1000, deadline=None)
@given(st.sampled_from(ALL), arrays(np.float64, 2, elements=st.floats(-10, 10)))
def test_smoothness_and_convexity_constants(pid, w):
    p = _get(pid)
    lam = np.linalg.eigvalsh(p.hess(w))
    tol = 1e-12 * p.extra["L"]
    assert lam[-1] <= p.extra["L"] + tol
    assert lam[0] >= p.extra["mu"] - tol
    # Every component fᵢ is L_max-smooth: its curvature along aᵢ is φ''·‖aᵢ‖² + λ.
    A = np.asarray(p.X)
    c = 0.25 if p.loss == "logistic" else 1.0
    assert np.max(c * np.einsum("nd,nd->n", A, A) + p.l2) == pytest.approx(p.extra["L_max"])


def test_huber_fit_resists_outliers():
    p = _get("huber_regression_2d")
    w_huber = np.array(p.minima[0])
    w_ols, *_ = np.linalg.lstsq(np.asarray(p.X), np.asarray(p.y), rcond=None)
    true_w = np.array(p.extra["true_w"])
    assert np.linalg.norm(w_huber - true_w) < 0.25 * np.linalg.norm(w_ols - true_w)


def test_condition_numbers_are_as_documented():
    kappa = {pid: np.linalg.cond(_get(pid).hess(np.array(_get(pid).minima[0]))) for pid in ALL}
    assert 4 < kappa["linreg_2d"] < 8
    assert 1000 < kappa["ill_conditioned_ls"] < 1100  # measured 1065.7
    # Every "κ ≈ N" claim in a description must match the measured κ(∇²f(w*)) to 5% (the
    # claims are rounded to 2–3 significant digits). Regression: the description said 900.
    stated = {}
    for pid in ALL:
        m = re.search(r"κ ≈ ([0-9.]+)", _get(pid).description)
        if m is not None:
            stated[pid] = float(m.group(1))
    assert {"linreg_2d", "ill_conditioned_ls"} <= set(stated)
    for pid, value in stated.items():
        assert value == pytest.approx(kappa[pid], rel=0.05), pid
