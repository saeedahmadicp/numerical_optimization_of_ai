"""Tests for numopt.regression.methods.

Oracles: scipy.stats.linregress / theilslopes, numpy.polyfit / linalg.lstsq,
scipy.linalg.solve (ridge closed form), scipy.optimize.linprog (LAD and minimax LPs),
and the first-order optimality condition of the convex Huber objective.
"""

import itertools
import json
import math
import warnings
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import linalg, optimize, stats

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.regression import methods as rm

ALL = (
    "linear_regression",
    "polynomial_regression",
    "ridge_regression",
    "huber_regression",
    "lad_regression",
    "theil_sen",
    "chebyshev_minimax_line",
)
DATA_IDS = (
    "runge_equispaced",
    "runge_chebyshev",
    "sine_samples",
    "step_data",
    "noisy_linear",
    "noisy_quadratic",
    "anscombe_1",
    "outliers_linear",
    "exponential_growth",
)
EPS = np.finfo(float).eps
ETA = np.finfo(float).smallest_subnormal  # underflow term of the standard model


@st.composite
def xy_data(draw, m_min=3, m_max=30, distinct=True):
    """x: integers/4 in [-10, 10] (well-conditioned designs); y: floats in [-100, 100]."""
    m = draw(st.integers(m_min, m_max))
    ints = draw(st.lists(st.integers(-40, 40), min_size=m, max_size=m, unique=distinct))
    x = np.array(ints, dtype=float) / 4.0
    y = np.array(draw(st.lists(st.floats(-100, 100), min_size=m, max_size=m)))
    return x, y


def _lp_lad(x, y):
    m = x.size
    design = np.c_[np.ones(m), x]
    c = np.r_[0.0, 0.0, np.ones(2 * m)]
    a_eq = np.c_[design, np.eye(m), -np.eye(m)]
    bounds = [(None, None)] * 2 + [(0, None)] * (2 * m)
    res = optimize.linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    assert res.status == 0
    return res.x[:2], res.fun


def _lp_minimax(x, y):
    m = x.size
    design = np.c_[np.ones(m), x]
    a_ub = np.r_[np.c_[design, -np.ones(m)], np.c_[-design, -np.ones(m)]]
    res = optimize.linprog(
        [0.0, 0.0, 1.0],
        A_ub=a_ub,
        b_ub=np.r_[y, -y],
        bounds=[(None, None), (None, None), (0, None)],
        method="highs",
    )
    assert res.status == 0
    return res.x[:2], res.fun


def _brute_minimax(x, y):
    """Exact discrete minimax error: max over all 3-point references of the levelled |h|.

    For a Haar space on a finite set the minimax error equals the largest levelled error
    over all (n+1)-point references (Cheney 1966, Ch. 2). For x_i < x_j < x_k the weights
    w = (x_k - x_j, -(x_k - x_i), x_j - x_i) annihilate lines, so
    |h| = |Σ w_r y_r| / (2(x_k - x_i)). O(m³), no solver tolerance (unlike an LP).
    """
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    best = 0.0
    for i, j, k in itertools.combinations(range(xs.size), 3):
        w = (xs[k] - xs[j], -(xs[k] - xs[i]), xs[j] - xs[i])
        num = w[0] * ys[i] + w[1] * ys[j] + w[2] * ys[k]
        best = max(best, abs(num) / (2.0 * (xs[k] - xs[i])))
    return best


# --------------------------------------------------------------------------------------
# Contract, fixtures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
@pytest.mark.parametrize("mid", ALL)
def test_contract_on_every_dataset(mid, pid):
    d = problems.get(pid)
    res = numopt.run(mid, d)
    assert_valid_result(res)
    assert res.converged, res.message
    beta = np.asarray(res.x)
    extra = res.extra
    assert_allclose(extra["coefficients"], beta, rtol=0, atol=0)
    assert_allclose(extra["fitted"], np.polyval(beta[::-1], d.x), rtol=1e-12, atol=1e-12)
    assert_allclose(extra["residuals"], d.y - np.asarray(extra["fitted"]), rtol=0, atol=1e-12)
    assert extra["rss"] == pytest.approx(float(np.sum(np.asarray(extra["residuals"]) ** 2)))
    assert extra["rmse"] == pytest.approx(math.sqrt(extra["rss"] / d.x.size))
    assert len(extra["eval"]["x"]) == len(extra["eval"]["y"]) == rm.N_GRID
    assert res.n_iter == res.trace[-1].k and res.n_fev == 0


def test_fixture_cases_cover_every_method():
    assert {m for m, _, _ in rm.FIXTURE_CASES} == set(ALL)
    for mid, pid, params in rm.FIXTURE_CASES:
        res = numopt.run(mid, problems.get(pid), **params)
        assert res.converged and len(res.trace) < 300
        json.dumps(res.to_dict(), allow_nan=False)


@pytest.mark.parametrize("mid", ALL)
def test_invalid_input(mid):
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0, 1.0, 2.0], [1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        numopt.run(mid, ([0.0, 1.0, 2.0], [1.0, 2.0]))
    with pytest.raises(TypeError):
        numopt.run(mid, 42)


def test_invalid_parameters():
    d = problems.get("noisy_linear")
    with pytest.raises(ValueError):
        numopt.run("linear_regression", d, solver="cramer")
    with pytest.raises(ValueError):
        numopt.run("polynomial_regression", d, degree=-1)
    with pytest.raises(ValueError):
        numopt.run("ridge_regression", d, lam=-1.0)
    with pytest.raises(ValueError):
        numopt.run("huber_regression", d, delta=0.0)
    with pytest.raises(ValueError):
        numopt.run("lad_regression", d, eps=0.0)
    with pytest.raises(ValueError):
        numopt.run("theil_sen", ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]))  # all x equal
    with pytest.raises(ValueError):
        numopt.run("chebyshev_minimax_line", ([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]))  # Haar fails
    with pytest.raises(ValueError):
        numopt.run("huber_regression", ([2.0, 2.0, 2.0], [0.0, 1.0, 2.0]))


# --------------------------------------------------------------------------------------
# Ordinary least squares
# --------------------------------------------------------------------------------------


def test_anscombe_published_fit():
    # Anscombe (1973) set I: ŷ = 3.0001 + 0.5001 x, R² = 0.6665, se(β₁) = 0.1179.
    res = numopt.run("linear_regression", problems.get("anscombe_1"))
    assert_allclose(res.x, [3.0000909, 0.5000909], atol=5e-7)
    assert res.extra["r_squared"] == pytest.approx(0.66654, abs=5e-5)
    assert res.extra["std_errors"][1] == pytest.approx(0.1179, abs=5e-5)


@pytest.mark.parametrize("solver", ["qr", "normal_equations", "svd"])
@pytest.mark.parametrize("pid", ["noisy_linear", "anscombe_1", "outliers_linear", "sine_samples"])
def test_linear_regression_matches_linregress(solver, pid):
    d = problems.get(pid)
    res = numopt.run("linear_regression", d, solver=solver)
    lr: Any = stats.linregress(d.x, d.y)
    # κ₂(scaled X) < 10 here: 1e-12 relative is ~10³ ulps of headroom (normal equations
    # lose 2·log10 κ digits, still far below).
    assert_allclose(res.x, [lr.intercept, lr.slope], rtol=1e-12, atol=1e-12)
    assert res.extra["r_squared"] == pytest.approx(lr.rvalue**2, rel=1e-12)
    se = [lr.intercept_stderr, lr.stderr]
    assert_allclose(res.extra["std_errors"], se, rtol=1e-11)
    m = d.x.size
    assert res.extra["adj_r_squared"] == pytest.approx(
        1 - (1 - lr.rvalue**2) * (m - 1) / (m - 2), rel=1e-12
    )
    info = res.trace[0].info
    assert info["solver"] == solver and info["rank"] == 2
    if solver == "normal_equations":
        assert info["cond_gram"] == pytest.approx(info["cond"] ** 2)


@given(xy_data())
@settings(max_examples=1000, deadline=None)
def test_ols_solvers_agree_with_lstsq(xy):
    x, y = xy
    design = np.c_[np.ones_like(x), x]
    ref = np.linalg.lstsq(design, y, rcond=None)[0]
    # Forward error of β: ~κ·u for QR/SVD, ~κ²·u for the normal equations (Higham 2002,
    # §20.4), relative to ‖β‖ + ‖y‖. κ₂ of the equilibrated [1, x] is NOT bounded by ~25
    # on this grid: clustered x (Hypothesis: x = [8.75, 8.5, 9], y = [0, 0, 2]) give
    # κ₂ = 86, κ₂² = 7.4e3, and a normal-equations error 3.0e-11 on |β₀| = 34.
    scale = 1.0 + float(np.max(np.abs(ref))) + float(np.max(np.abs(y)))
    for solver in ("qr", "normal_equations", "svd"):
        res = numopt.run("linear_regression", (x, y), solver=solver)
        assert res.converged
        cond = res.trace[0].info["cond"]
        amplification = cond**2 if solver == "normal_equations" else cond
        tol = max(1e-11 * (1.0 + float(np.max(np.abs(y)))), 10 * amplification * EPS * scale)
        assert_allclose(res.x, ref, rtol=0, atol=tol)
        r = np.asarray(res.extra["residuals"])
        # Normal equations Xᵀr = 0 (orthogonality of the residual).
        assert np.max(np.abs(design.T @ r)) <= 1e-9 * (1.0 + float(np.max(np.abs(y)))) * x.size
        r2 = res.extra["r_squared"]
        if r2 is not None:
            # OLS with intercept: R² ∈ [0, 1] in exact arithmetic. The computed residuals
            # carry |δr| ≤ δ = c·κ·u·‖y‖∞ (forward error of β plus evaluation; c ≤ 14
            # measured against exact rational residuals on near-constant y, c = 32 here),
            # so |δRSS| ≤ 2δ√(m·RSS) + mδ² and, with RSS ≤ TSS, R² moves by at most
            # (2δ√(m·TSS) + mδ²)/TSS.
            m, tss = x.size, res.extra["tss"]
            delta = 32 * res.trace[0].info["cond"] * (EPS / 2) * float(np.max(np.abs(y)))
            slack = (2 * delta * math.sqrt(m * tss) + m * delta**2) / tss
            assert -slack <= r2 <= 1 + slack


def test_rank_deficient_design():
    x = np.full(5, 2.0)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for solver in ("qr", "normal_equations"):
        res = numopt.run("linear_regression", (x, y), solver=solver)
        assert not res.converged and "rank" in res.message
        assert_valid_result(res)
    # The SVD solver returns a solution, but it is not unique: converged=False, as for
    # polynomial_regression (it used to say converged=True).
    res = numopt.run("linear_regression", (x, y), solver="svd")
    assert not res.converged and res.trace[0].info["rank"] == 1 and "not unique" in res.message
    assert_valid_result(res)
    # Minimum ‖Sβ‖₂, S = diag(column norms): lstsq of the equilibrated design, mapped back.
    design = np.c_[np.ones(5), x]
    s = np.linalg.norm(design, axis=0)
    ref = np.linalg.lstsq(design / s, y, rcond=None)[0] / s
    assert_allclose(res.x, ref, rtol=1e-13)
    assert_allclose(res.extra["fitted"], np.full(5, y.mean()), rtol=1e-14)  # any LS fit
    assert res.extra["std_errors"] is None
    res = numopt.run(
        "polynomial_regression", ([0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0]), degree=2
    )
    assert not res.converged and "rank" in res.message


def test_constant_y_has_no_r_squared():
    res = numopt.run("linear_regression", ([0.0, 1.0, 2.0], [5.0, 5.0, 5.0]))
    assert res.converged and res.extra["r_squared"] is None and res.extra["adj_r_squared"] is None
    assert_allclose(res.x, [5.0, 0.0], atol=1e-14)
    # Hypothesis counterexample: fl(mean(0.2, 0.2, 0.2)) ≠ 0.2, so TSS ≈ 9e-33 > 0 and
    # 1 - RSS/TSS was -38. Constant data have no R², for every method and solver.
    x, y = [0.5, 0.75, 0.25], [0.2, 0.2, 0.2]
    for mid in ALL:
        res = numopt.run(mid, (x, y))
        assert res.extra["r_squared"] is None and res.extra["adj_r_squared"] is None, mid
    for solver in ("qr", "normal_equations", "svd"):
        res = numopt.run("linear_regression", (x, y), solver=solver)
        assert res.extra["r_squared"] is None
        assert_allclose(res.x, [0.2, 0.0], atol=1e-15)


# --------------------------------------------------------------------------------------
# Polynomial regression
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", ["noisy_quadratic", "anscombe_1", "exponential_growth"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_polynomial_matches_polyfit(pid, degree):
    d = problems.get(pid)
    res = numopt.run("polynomial_regression", d, degree=degree)
    ref, cov = np.polyfit(d.x, d.y, degree, cov=True)
    cond = res.trace[0].info["cond"]
    # Both solve the column-scaled LS problem; agreement to ~κ·u relative.
    tol = 100 * cond * EPS * max(1.0, float(np.max(np.abs(ref))))
    assert_allclose(res.x, ref[::-1], rtol=0, atol=tol)
    # polyfit forms (AᵀA)⁻¹ (error ~κ²u); ours uses the R factor (error ~κu).
    se_ref = np.sqrt(np.diag(cov))[::-1]
    assert_allclose(res.extra["std_errors"], se_ref, rtol=100 * cond**2 * EPS)
    assert res.extra["degree"] == degree


def test_polynomial_on_noisy_quadratic_recovers_truth_and_interpolates_at_full_degree():
    d = problems.get("noisy_quadratic")
    res = numopt.run("polynomial_regression", d, degree=2)
    assert_allclose(res.x, [1.0, -1.0, 0.5], atol=0.2)  # within the noise
    assert res.extra["r_squared"] > 0.95
    # m = d + 1 points: the fit interpolates, R² = 1 and σ̂ is undefined.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    res = numopt.run("polynomial_regression", (x, x**3 - x), degree=3)
    assert_allclose(res.x, [0.0, -1.0, 0.0, 1.0], atol=1e-12)
    assert res.extra["sigma"] is None and res.extra["std_errors"] is None
    assert res.extra["r_squared"] == pytest.approx(1.0)


@given(xy_data(m_min=6), st.integers(0, 4))
@settings(max_examples=1000, deadline=None)
def test_polynomial_matches_lstsq_property(xy, degree):
    x, y = xy
    v = np.vander(x, degree + 1, increasing=True)
    scale = np.linalg.norm(v, axis=0)
    ref = np.linalg.lstsq(v / scale, y, rcond=None)[0] / scale
    res = numopt.run("polynomial_regression", (x, y), degree=degree)
    assert res.converged
    cond = res.trace[0].info["cond"]
    # Forward error of a backward-stable LS solve: ~κ·u relative to ‖y‖/σ_min scale.
    tol = 100 * cond * EPS * (1.0 + float(np.max(np.abs(ref))) + float(np.max(np.abs(y))))
    assert_allclose(res.x, ref, rtol=0, atol=tol)


# --------------------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------------------


def _ridge_closed_form(x, y, lam, degree):
    feats = np.vander(x, degree + 1, increasing=True)[:, 1:]
    xbar = feats.mean(axis=0)
    xc = feats - xbar
    yc = y - y.mean()
    slopes = linalg.solve(xc.T @ xc + lam * np.eye(degree), xc.T @ yc, assume_a="pos")
    return np.r_[y.mean() - xbar @ slopes, slopes]


@pytest.mark.parametrize(
    ("lam", "degree"), [(0.0, 1), (0.5, 1), (10.0, 2), (10.0, 6), (1e3, 3), (1e4, 1)]
)
def test_ridge_matches_closed_form(lam, degree):
    d = problems.get("noisy_quadratic")
    res = numopt.run("ridge_regression", d, lam=lam, degree=degree)
    assert res.converged
    ref = _ridge_closed_form(d.x, d.y, lam, degree)
    # The closed form squares κ (≈ 10⁴·... for degree 6): compare to ~κ²u relative.
    cond = res.trace[0].info["cond"]
    tol = 10 * cond**2 * EPS * (1.0 + float(np.max(np.abs(ref))) + float(np.max(np.abs(d.y))))
    assert_allclose(res.x, ref, rtol=0, atol=tol)
    assert res.fun == pytest.approx(res.extra["rss"] + lam * float(np.sum(ref[1:] ** 2)), rel=1e-9)
    assert res.extra["std_errors"] is None


def test_ridge_zero_lambda_is_ols_and_degree_zero_is_mean():
    d = problems.get("noisy_linear")
    a = numopt.run("ridge_regression", d, lam=0.0)
    b = numopt.run("linear_regression", d)
    assert_allclose(a.x, b.x, rtol=1e-12)
    c = numopt.run("ridge_regression", d, lam=5.0, degree=0)
    assert_allclose(c.x, [d.y.mean()], rtol=1e-15)
    res = numopt.run("ridge_regression", ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]), lam=0.0)
    assert not res.converged and "rank" in res.message
    assert_valid_result(res)
    res = numopt.run("ridge_regression", ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]), lam=1.0)
    assert res.converged and res.x[1] == pytest.approx(0.0, abs=1e-15)


@given(xy_data(m_min=5), st.floats(0.0, 100.0), st.floats(0.0, 100.0), st.integers(1, 3))
@settings(max_examples=1000, deadline=None)
def test_ridge_shrinks_monotonically(xy, lam1, lam2, degree):
    # ‖β_{1:}(λ)‖₂ is non-increasing in λ (ESL §3.4.1; the SVD form of the solution).
    x, y = xy
    lo, hi = sorted((lam1, lam2))
    b_lo = numopt.run("ridge_regression", (x, y), lam=lo, degree=degree)
    b_hi = numopt.run("ridge_regression", (x, y), lam=hi, degree=degree)
    assume(b_lo.converged and b_hi.converged)
    n_lo = float(np.linalg.norm(np.asarray(b_lo.x)[1:]))
    n_hi = float(np.linalg.norm(np.asarray(b_hi.x)[1:]))
    assert n_hi <= n_lo * (1 + 1e-9) + 1e-12
    ref = _ridge_closed_form(x, y, hi, degree)
    cond = b_hi.trace[0].info["cond"]
    # The intercept ȳ - x̄ᵀβ carries rounding of size u·‖y‖∞, so ‖y‖∞ is part of the scale.
    scale = 1 + np.max(np.abs(ref)) + np.max(np.abs(y))
    assert_allclose(b_hi.x, ref, rtol=0, atol=10 * cond**2 * EPS * scale)


# --------------------------------------------------------------------------------------
# Huber
# --------------------------------------------------------------------------------------


def _huber_gradient(x, y, beta, scale, delta):
    """∇_β Σ ρ(r_i/σ) = -(1/σ) Σ ψ(u_i) [1, x_i], ψ(u) = clip(u, -δ, δ)."""
    u = (y - beta[0] - beta[1] * x) / scale
    psi = np.clip(u, -delta, delta)
    return -np.array([psi.sum(), (psi * x).sum()]) / scale


@pytest.mark.parametrize("pid", ["outliers_linear", "noisy_linear", "anscombe_1"])
def test_huber_optimality_and_oracle(pid):
    d = problems.get(pid)
    res = numopt.run("huber_regression", d)
    assert res.converged
    scale, delta = res.extra["scale"], res.extra["delta"]
    g = _huber_gradient(d.x, d.y, np.asarray(res.x), scale, delta)
    # KKT of the convex objective: ∇F = 0 up to the IRLS stopping tolerance.
    assert np.max(np.abs(g)) < 1e-7 * d.x.size
    # No point does better (independent minimizer of the same convex function).
    f_obj = lambda b: float(np.sum(rm._huber_rho((d.y - b[0] - b[1] * d.x) / scale, delta)))  # noqa: E731
    opt = optimize.minimize(
        f_obj,
        np.zeros(2),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000},
    )
    assert res.fun <= opt.fun + 1e-10
    assert res.fun == pytest.approx(f_obj(np.asarray(res.x)), rel=1e-14)


def test_robust_methods_resist_the_outliers():
    d = problems.get("outliers_linear")
    ols = numopt.run("linear_regression", d)
    assert abs(ols.x[1] - 2.0) > 0.3  # pulled away by the two outliers
    for mid in ("huber_regression", "lad_regression", "theil_sen"):
        res = numopt.run(mid, d)
        assert abs(res.x[1] - 2.0) < 0.1 and abs(res.x[0] - 1.0) < 0.5, (mid, res.x)
    # Huber (95% Gaussian efficiency) must agree with OLS on the 18 clean points to within
    # one standard error of that fit. With σ̂ from the OLS residuals (the old scale,
    # tilted by the outliers: σ̂ = 2.64 vs noise 0.3) the slope was 5 standard errors off.
    clean = np.ones(d.x.size, dtype=bool)
    clean[[i for i, _ in problems.data.OUTLIER_OFFSETS]] = False
    lr: Any = stats.linregress(d.x[clean], d.y[clean])
    hub = numopt.run("huber_regression", d)
    assert abs(hub.x[0] - lr.intercept) < lr.intercept_stderr
    assert abs(hub.x[1] - lr.slope) < lr.stderr
    # The dataset description: all three stay within the noise of that line (2 s.e.;
    # LAD and Theil–Sen are less efficient than Huber for Gaussian noise).
    for mid in ("lad_regression", "theil_sen"):
        res = numopt.run(mid, d)
        assert abs(res.x[0] - lr.intercept) < 2 * lr.intercept_stderr, mid
        assert abs(res.x[1] - lr.slope) < 2 * lr.stderr, mid
    w = np.asarray(numopt.run("huber_regression", d).extra["weights"])
    outliers = [i for i, _ in problems.data.OUTLIER_OFFSETS]
    # w = δσ̂/|r| for the outliers (|r| ≈ 15–20 ≫ δσ̂); the clean points keep weight 1.
    assert np.all(w[outliers] < 0.3) and np.median(w) == 1.0


@given(xy_data(m_min=4), st.floats(0.5, 3.0))
@settings(max_examples=1000, deadline=None)
def test_huber_objective_decreases_monotonically(xy, delta):
    # IRLS = majorize–minimize for ψ(u)/u non-increasing (Maronna et al., Ch. 4): F(β_k) ↓.
    x, y = xy
    res = numopt.run("huber_regression", (x, y), delta=delta)
    assert_valid_result(res, max_iter=100)
    f = [float(s.fun) for s in res.trace if s.fun is not None]
    assert len(f) == len(res.trace)
    for a, b in itertools.pairwise(f):
        assert b <= a + 1e-12 * max(1.0, abs(a))
    if res.converged and res.extra["scale"] > 0:
        g = _huber_gradient(x, y, np.asarray(res.x), res.extra["scale"], delta)
        hess_scale = (1 + np.abs(x).max()) ** 2 * x.size / res.extra["scale"] ** 2
        assert np.max(np.abs(g)) <= 1e-7 * hess_scale * (1 + np.max(np.abs(res.x)))


def test_huber_scale_is_the_normalized_mad_of_the_l1_residuals():
    # Maronna, Martin & Yohai (2006), Ch. 4: σ̂ = Med(non-null |r_i(β̂_L1)|)/Φ⁻¹(3/4).
    # Oracle: the L1 line from the linear program (HiGHS), its 2 null residuals dropped.
    d = problems.get("outliers_linear")
    res = numopt.run("huber_regression", d)
    beta_l1, _ = _lp_lad(d.x, d.y)
    r_l1 = np.sort(np.abs(d.y - beta_l1[0] - beta_l1[1] * d.x))
    assert np.max(r_l1[:2]) < 1e-9  # an L1 line passes through p = 2 points
    sigma_ref = float(np.median(r_l1[2:])) / stats.norm.ppf(0.75)
    # NOTE: rtol 1e-4, not ~1e-12: the preliminary L1 fit is LAD-IRLS with the floor
    # ε = 1e-6·RMS(OLS residuals) ≈ 3e-6, whose residuals differ from the exact L1
    # residuals by O(ε) (measured relative difference of σ̂: 2.6e-6).
    assert res.extra["scale"] == pytest.approx(sigma_ref, rel=1e-4)
    assert 0.2 < res.extra["scale"] < 0.45  # the clean noise level is 0.3


def test_huber_max_iter_and_exact_fit():
    d = problems.get("outliers_linear")
    res = numopt.run("huber_regression", d, max_iter=2)
    assert not res.converged and "max_iter" in res.message and res.n_iter == 2
    assert_valid_result(res, max_iter=2)
    res = numopt.run("huber_regression", ([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0]))
    assert res.converged and "exactly" in res.message
    assert_allclose(res.x, [1.0, 2.0], atol=1e-13)


# --------------------------------------------------------------------------------------
# LAD
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
def test_lad_matches_linear_program(pid):
    d = problems.get(pid)
    res = numopt.run("lad_regression", d)
    assert res.converged
    _, f_lp = _lp_lad(d.x, d.y)
    eps = 1e-6
    # IRLS minimizes the ε-smoothed objective: Σ|r| ≤ LP optimum + m·ε/2.
    assert f_lp - 1e-9 <= res.fun <= f_lp + d.x.size * eps / 2 + 1e-9


@given(xy_data(m_min=4, m_max=20))
@settings(max_examples=1000, deadline=None)
def test_lad_smoothed_objective_decreases_and_is_near_optimal(xy):
    x, y = xy
    eps = 1e-6
    res = numopt.run("lad_regression", (x, y), eps=eps)
    assert_valid_result(res, max_iter=500)
    # Majorize–minimize: S_ε(β_k) is non-increasing on every run.
    s = [step.info["smoothed_objective"] for step in res.trace]
    for a, b in itertools.pairwise(s):
        assert b <= a + 1e-12 * max(1.0, abs(a))
    _, f_lp = _lp_lad(x, y)
    assert res.fun >= f_lp - 1e-9 * max(1.0, f_lp)  # nothing beats the LP optimum
    if res.converged:
        # |r| ≤ ρ_ε(r) ≤ |r| + ε/2, so the limit is within m·ε/2 of the optimum; the
        # stopping test (‖Δβ‖ ≤ 1e-10·(1+‖β‖)) adds O(1e-10·‖β‖·Σ|x_i|/(1 - rate)).
        assert res.fun <= f_lp + x.size * eps / 2 + 1e-6 * (1 + np.abs(y).max())
    else:
        # IRLS for L1 converges linearly with a rate near 1 at degenerate optima
        # (~4% of these random cases need > 500 iterations): reported, not hidden.
        assert "max_iter" in res.message


def test_lad_max_iter():
    res = numopt.run("lad_regression", problems.get("outliers_linear"), max_iter=3)
    assert not res.converged and "max_iter" in res.message and res.n_iter == 3
    assert_valid_result(res, max_iter=3)


# --------------------------------------------------------------------------------------
# Theil–Sen
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
def test_theil_sen_matches_scipy(pid):
    d = problems.get(pid)
    res = numopt.run("theil_sen", d)
    ref: Any = stats.theilslopes(d.y, d.x, method="joint")
    assert_allclose(res.x, [ref.intercept, ref.slope], rtol=1e-13, atol=1e-13)
    m = d.x.size
    assert res.extra["n_pairs"] == m * (m - 1) // 2


@given(xy_data(m_min=2, m_max=25, distinct=False))
@settings(max_examples=1000, deadline=None)
def test_theil_sen_property(xy):
    x, y = xy
    assume(np.unique(x).size >= 2)
    res = numopt.run("theil_sen", (x, y))
    ref: Any = stats.theilslopes(y, x, method="joint")
    assert_allclose(res.x, [ref.intercept, ref.slope], rtol=1e-12, atol=1e-12)
    slopes = np.asarray(res.extra["slopes"])
    assert np.all(np.diff(slopes) >= 0)
    # Half the pairwise slopes lie on each side of the median.
    assert np.sum(slopes <= res.x[1]) >= slopes.size / 2
    assert np.sum(slopes >= res.x[1]) >= slopes.size / 2


# --------------------------------------------------------------------------------------
# Minimax line
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
def test_minimax_matches_linear_program(pid):
    d = problems.get(pid)
    res = numopt.run("chebyshev_minimax_line", d)
    assert res.converged
    beta_lp, f_lp = _lp_minimax(d.x, d.y)
    assert res.fun == pytest.approx(f_lp, rel=1e-9, abs=1e-12)
    assert_allclose(res.x, beta_lp, rtol=1e-7, atol=1e-7)  # unique under the Haar condition


@given(xy_data(m_min=3, m_max=25))
@settings(max_examples=1000, deadline=None)
def test_minimax_ascent_and_equioscillation(xy):
    x, y = xy
    res = numopt.run("chebyshev_minimax_line", (x, y))
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    levels = [abs(s.info["level"]) for s in res.trace]
    assert all(b > a for a, b in itertools.pairwise(levels))  # strict ascent
    r = np.asarray(res.extra["residuals"])
    ref = res.extra["reference"]
    assert x[ref[0]] < x[ref[1]] < x[ref[2]]
    h = res.extra["level"]
    scale = 1e-9 * max(1.0, float(np.max(np.abs(y))))
    # Alternation theorem: |r| = max|r| on the reference with alternating signs.
    assert_allclose(r[ref], [h, -h, h], rtol=0, atol=scale)
    assert np.max(np.abs(r)) <= abs(h) + scale
    # Exact oracle (an LP is not: HiGHS's feasibility tolerance 1e-7 returned 0 for
    # x = [0, 0.25, 0.5], y = [0, 0, 5.96e-8], whose minimax error is 1.49e-8).
    # Rounding: the stopping test admits 8·(eps·S + η), the oracle ~4·(eps·‖y‖∞ + η).
    f_ref = _brute_minimax(x, y)
    s = float(np.max(np.abs(y) + abs(res.x[0]) + np.abs(res.x[1] * x)))
    assert res.fun is not None
    assert abs(res.fun - f_ref) <= 1e-12 * f_ref + 16 * (EPS * s + ETA)


def test_minimax_max_iter():
    d = problems.get("outliers_linear")
    assert numopt.run("chebyshev_minimax_line", d).n_iter >= 2
    res = numopt.run("chebyshev_minimax_line", d, max_iter=1)
    assert not res.converged and "max_iter" in res.message
    assert_valid_result(res, max_iter=1)
    assert res.trace[-1].info["entering"] is None


# --------------------------------------------------------------------------------------
# Audit regressions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
def test_polynomial_degree_beyond_the_data_gives_the_minimum_norm_fit(pid):
    # Every degree in the ParamSpec range must give a curve: with m < d + 1 points the
    # LS solution is not unique, so the minimum-norm one is returned with converged=False.
    d = problems.get(pid)
    m = d.x.size
    spec = get_method("polynomial_regression").params[0]
    assert spec.max is not None
    for degree in range(m, int(spec.max) + 1):
        res = numopt.run("polynomial_regression", d, degree=degree)
        assert_valid_result(res)
        assert not res.converged and "minimum-norm" in res.message, res.message
        info = res.trace[0].info
        assert info["solver"] == "svd" and info["rank"] == m and info["cond"] == math.inf
        # Oracle: LAPACK least squares (gelsd) on the equilibrated design, min ‖Sβ‖₂.
        v = np.vander(d.x, degree + 1, increasing=True)
        s = np.linalg.norm(v, axis=0)
        gamma, _, rank, sv = np.linalg.lstsq(v / s, d.y, rcond=None)
        assert rank == m
        kappa = sv[0] / sv[rank - 1]  # condition of the row-space problem: ≤ 2.2e9 here
        assert_allclose(res.x * s, gamma, rtol=0, atol=100 * kappa * EPS * np.max(np.abs(gamma)))
        # Distinct x and rank m: the minimum-norm solution interpolates the data.
        r = np.asarray(res.extra["residuals"])
        assert np.max(np.abs(r)) <= 100 * kappa * EPS * float(np.max(np.abs(d.y)))


def test_polynomial_underdetermined_tiny_case():
    res = numopt.run("polynomial_regression", ([0.0, 1.0], [1.0, 2.0]), degree=2)
    assert not res.converged and res.trace[0].info["rank"] == 2
    ref = np.linalg.lstsq(np.vander([0.0, 1.0], 3, increasing=True), [1.0, 2.0], rcond=None)[0]
    assert_allclose(res.x, ref, rtol=1e-14, atol=1e-15)
    assert res.extra["sigma"] is None and res.extra["std_errors"] is None


def test_overflow_is_reported_not_raised():
    # The slope 1e200/1e-150 exceeds the float range. Old behaviour: Huber returned
    # converged=True with β = [1e200, inf], Theil–Sen converged=True with [nan, inf], and
    # LAD raised LinAlgError. Every method must now return, without RuntimeWarnings.
    x = 1e-150 * np.linspace(0.0, 1.0, 12)
    y = 1e200 * np.array([1, 3, 2, 5, 4, 6, 8, 7, 9, 11, 10, 12.0])
    for mid in ALL:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = numopt.run(mid, (x, y))
        assert_valid_result(res)
        if mid == "ridge_regression":
            # The penalty bounds the slope: a finite, correct ridge solution.
            assert res.converged and np.all(np.isfinite(res.x))
        else:
            assert not res.converged, (mid, res.message)
    huber = numopt.run("huber_regression", (x, y))
    assert "exactly" not in huber.message


@pytest.mark.parametrize("offset", [1e3, 1e6, 1e9, 1.7e9, -1e12])
def test_minimax_converges_with_a_large_x_offset(offset):
    # Old behaviour: x = 1e6 + i gave "no ascent" and converged=False (β₀ and β₁x cancel,
    # so the residuals carry eps·|β₁x| ≫ tol·‖y‖∞). The same data at x = i converged.
    t = np.arange(10.0)
    y = 1 + 2 * t + np.array([0.3, -0.5, 0.1, 0.8, -0.2, 0.4, -0.9, 0.2, 0.6, -0.1])
    x = offset + t  # exact: t and offset are integers below 2^53
    res = numopt.run("chebyshev_minimax_line", (x, y))
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    _, f_lp = _lp_minimax(t, y)  # the shifted problem has the same optimal value
    s = float(np.max(np.abs(y) + abs(res.x[0]) + np.abs(res.x[1] * x)))
    assert abs(res.fun - f_lp) <= 1e-9 * f_lp + 16 * (EPS * s + ETA)
    assert res.trace[-1].info["entering"] is None


@given(xy_data(m_min=3, m_max=25), st.sampled_from([1e2, 1e4, 1e6, 1e8, -1e10]))
@settings(max_examples=1000, deadline=None)
def test_minimax_offset_property(xy, offset):
    t, y = xy
    x = offset + t  # exact: t is a multiple of 1/4 and |x| < 2^51
    res = numopt.run("chebyshev_minimax_line", (x, y))
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    assert res.trace[-1].info["entering"] is None
    f_ref = _brute_minimax(t, y)  # the shift does not change the minimax error
    s = float(np.max(np.abs(y) + abs(res.x[0]) + np.abs(res.x[1] * x)))
    assert res.fun is not None
    assert abs(res.fun - f_ref) <= 1e-12 * f_ref + 16 * (EPS * s + ETA)


def test_minimax_subnormal_data_converge():
    # Hypothesis counterexample: 8·eps·S underflows to 0 for subnormal data, so the gap
    # 5e-324 could never pass and the exchange stopped on "no ascent".
    x = 100.0 + np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 5e-324])
    res = numopt.run("chebyshev_minimax_line", (x, y))
    assert_valid_result(res, max_iter=100)
    assert res.converged, res.message
    assert res.fun is not None and res.fun <= 5e-324


def test_minimax_rejected_exchange_has_no_entering_point(monkeypatch):
    # Without the rounding term the large-offset data stop on "no ascent"; the last Step
    # must not announce an exchange that never happens (it used to say entering = 6).
    monkeypatch.setattr(rm, "_MINIMAX_ROUNDING_EPS", 0.0)
    t = np.arange(10.0)
    y = 1 + 2 * t + np.array([0.3, -0.5, 0.1, 0.8, -0.2, 0.4, -0.9, 0.2, 0.6, -0.1])
    res = numopt.run("chebyshev_minimax_line", (1e6 + t, y), tol=1e-15)
    assert not res.converged and "no ascent" in res.message
    assert res.trace[-1].info["entering"] is None
    assert all(s.info["entering"] is not None for s in res.trace[:-1])
    assert_valid_result(res, max_iter=100)


# --------------------------------------------------------------------------------------
# Large x offsets (audit: data on a Unix-time axis)
# --------------------------------------------------------------------------------------


def _offset_line_data():
    t = np.arange(20.0)
    return t, 3.0 + 0.5 * t + np.sin(t)


@pytest.mark.parametrize("offset", [1e4, 1e8, 1.7e9, -1e10])
def test_qr_and_svd_with_a_large_x_offset(offset):
    # Audit: solver='svd' factored the raw [1, x]; for x = 1.7e9 + i its rank rule saw
    # rank 1 and returned slope 4.6e-9 (true 0.475) with converged=True.
    t, y = _offset_line_data()
    x = offset + t  # exact: integers below 2^53
    lr: Any = stats.linregress(t, y)  # the shift changes only the intercept
    for solver in ("qr", "svd"):
        res = numopt.run("linear_regression", (x, y), solver=solver)
        assert_valid_result(res)
        info = res.trace[0].info
        assert res.converged and info["rank"] == 2, (solver, res.message)
        # Forward error of the slope ≲ κ₂·u (Higham 2002, §20.1); measured ≤ 0.13·κ·u.
        assert res.x[1] == pytest.approx(lr.slope, rel=4 * info["cond"] * EPS)
        assert res.x[0] + res.x[1] * offset == pytest.approx(lr.intercept, rel=1e-6)
        assert res.extra["r_squared"] == pytest.approx(lr.rvalue**2, rel=1e-6)


@pytest.mark.parametrize(("offset", "converged"), [(1e4, True), (1e8, False), (1e9, False)])
def test_normal_equations_report_lost_digits(offset, converged):
    # Audit: Cholesky succeeded, so converged=True, although κ₂(AᵀA) = 1.2e15 (x = 1e8 + i)
    # left the slope 12% wrong (0.95 wrong at 1e9).
    t, y = _offset_line_data()
    res = numopt.run("linear_regression", (offset + t, y), solver="normal_equations")
    assert_valid_result(res)
    assert res.converged == converged, res.message
    info = res.trace[0].info
    assert (info["cond_gram"] * EPS < rm.NE_MAX_FORWARD_BOUND) == converged
    assert "digits lost" in res.message
    if not converged:
        assert "no correct digits" in res.message and "solver='qr'" in res.message
        assert np.all(np.isfinite(res.x))  # the wrong line is shown, not hidden


@given(xy_data(), st.sampled_from([1e2, 1e5, 1e7, 1e9, -1e10]))
@settings(max_examples=1000, deadline=None)
def test_ols_flags_follow_the_error_bounds_under_x_offsets(xy, offset):
    t, y = xy
    x = offset + t  # exact: t is a multiple of 1/4 and |x| < 2^51
    ref = np.linalg.lstsq(np.c_[np.ones_like(t), t], y, rcond=None)[0]  # unshifted
    for solver in ("qr", "svd", "normal_equations"):
        res = numopt.run("linear_regression", (x, y), solver=solver)
        assert_valid_result(res)
        info = res.trace[0].info
        assert info["rank"] == 2  # distinct x never look rank deficient after scaling
        amplification = info["cond"] ** 2 if solver == "normal_equations" else info["cond"]
        if solver == "normal_equations" and "Cholesky of AᵀA failed" not in res.message:
            assert res.converged == (amplification * EPS < rm.NE_MAX_FORWARD_BOUND)
        else:
            assert res.converged == (solver != "normal_equations"), res.message
        if res.converged:
            # Slope forward error ≲ c·amplification·u relative to the data scale.
            scale = 1.0 + abs(float(ref[1])) + float(np.max(np.abs(y))) / (1 + np.ptp(t))
            assert abs(res.x[1] - ref[1]) <= 1e-11 * scale + 10 * amplification * EPS * scale


@pytest.mark.parametrize("offset", [1e6, 1e8, 1e10, -1e12])
@pytest.mark.parametrize("mid", ["huber_regression", "lad_regression"])
def test_irls_lines_converge_with_a_large_x_offset(mid, offset):
    # Audit: with x = 1e8 + i Huber stopped at max_iter=100 (β₀ oscillated by 0.031 =
    # ε·κ·|β₀| in the uncentred solves) with the slope correct to 9 digits.
    t, y = _offset_line_data()
    y[4] += 10.0  # one gross outlier
    base = numopt.run(mid, (t, y))
    res = numopt.run(mid, (offset + t, y))
    assert_valid_result(res)
    assert base.converged and res.converged, res.message
    assert res.n_iter <= base.n_iter + 2
    assert res.x[1] == pytest.approx(base.x[1], rel=1e-8)
    # β₀ + β₁·offset cancels: its rounding is ~ε·|β₁·offset| (1e-4 at offset 1e12).
    cancel = 8 * EPS * abs(res.x[1] * offset)
    assert res.x[0] + res.x[1] * offset == pytest.approx(base.x[0], abs=1e-6 + cancel)
    assert res.fun == pytest.approx(base.fun, rel=1e-9)


@given(
    xy_data(m_min=4, m_max=20),
    st.sampled_from([1e4, 1e8, -1e10, 1e12]),
    st.sampled_from(["huber_regression", "lad_regression"]),
)
@settings(max_examples=1000, deadline=None)
def test_irls_flag_and_objective_are_invariant_under_x_offsets(xy, offset, mid):
    # The IRLS iterates live in the centred coefficients θ, which a shift of x does not
    # change, so the stopping test and its outcome must not depend on the shift.
    t, y = xy
    base = numopt.run(mid, (t, y))
    res = numopt.run(mid, (offset + t, y))
    assert_valid_result(res)
    assert res.converged == base.converged, (base.message, res.message)
    assert abs(res.n_iter - base.n_iter) <= 1
    assert res.extra["x_center"] == pytest.approx(offset + float(np.mean(t)), abs=1e-3)
    if res.converged and base.fun is not None and res.fun is not None:
        # The objective is a sum of m terms with slope ≤ 1/σ̂ (Huber: ρ' ≤ δ) in the
        # residuals; only the centring of x differs (rounding ε·|offset|).
        assert res.fun == pytest.approx(base.fun, rel=1e-6, abs=1e-6)
