"""Tests for numopt.interpolation.rational: AAA and Floater–Hormann.

Oracles: scipy.interpolate.AAA and FloaterHormannInterpolator (SciPy is a test-only
dependency), QZ on the arrowhead pencil (scipy.linalg.eigvals), Newton roots of d(λ) in
extended precision (numpy.longdouble), the published integer weights and Table 1 / Table 2
of Floater & Hormann (2007), hand-computed steps and residues, and the theorems the
methods rest on (NST 2018 Proposition 3.1; FH 2007 Theorems 1–2; Bos et al. 2012).
The research study research/barycentric-rational-approximation verified the same math.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import warnings
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import assume as hypothesis_assume
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import AAA, FloaterHormannInterpolator

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.interpolation import rational as rm
from numopt.interpolation.methods import N_GRID, _barycentric_eval, barycentric

METHODS = ("aaa", "floater_hormann")
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
EPS = float(np.finfo(np.float64).eps)
#: numpy.longdouble is IEEE quad on aarch64 (eps 1.9e-34); x86 has 80-bit (eps 1.1e-19).
QUAD = float(np.finfo(np.longdouble).eps) < 1e-30
PROPERTY = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _eval(res: Any, t: np.ndarray) -> np.ndarray:
    """The final rational function of ``res`` at ``t``, evaluated in chunks of 4096 points."""
    z = np.asarray(res.extra["nodes"], dtype=float)
    w = np.asarray(res.extra["coefficients"], dtype=float)
    f = np.asarray(res.extra["values"], dtype=float)
    out = np.empty(t.size)
    with np.errstate(all="ignore"):
        for s in range(0, t.size, 4096):
            out[s : s + 4096] = _barycentric_eval(z, w, f, t[s : s + 4096])
    return out


def _lebesgue_constant(x: np.ndarray, w: np.ndarray, points_per_gap: int) -> float:
    """Λ = max_t Σ_k |w_k/(t - x_k)| / |Σ_k w_k/(t - x_k)| on interior points of each gap
    (Bos, De Marchi, Hormann & Klein 2012, eq. (2)); the Lebesgue function is 1 at nodes."""
    frac = (np.arange(1, points_per_gap + 1) / (points_per_gap + 1))[None, :]  # (1, P)
    t = (x[:-1, None] + frac * np.diff(x)[:, None]).reshape(-1)  # (n·P,)
    best = 1.0
    for s in range(0, t.size, 4096):
        c = w[None, :] / (t[s : s + 4096, None] - x[None, :])  # (T, n+1)
        best = max(best, float(np.max(np.sum(np.abs(c), axis=1) / np.abs(np.sum(c, axis=1)))))
    return best


def _tanh_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 2000)
    return x, np.tanh(50.0 * x)


def _clustered_sqrt() -> np.ndarray:
    return np.unique(np.concatenate([np.linspace(0, 1, 1500), np.logspace(-30, 0, 600)]))


def _scipy_aaa(x: np.ndarray, y: np.ndarray, **kw: Any) -> AAA:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return AAA(x, y, clean_up=False, **kw)


def _pencil_poles(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """QZ on the arrowhead pencil E v = λ B v (NST 2018, eq. (3.11))."""
    m = z.size
    b_mat = np.eye(m + 1)
    b_mat[0, 0] = 0.0
    e_mat = np.zeros((m + 1, m + 1))
    e_mat[0, 1:] = w
    e_mat[1:, 0] = 1.0
    np.fill_diagonal(e_mat[1:, 1:], z)
    p = scipy.linalg.eigvals(e_mat, b_mat)
    return p[np.isfinite(p)]


def _quad_root(lam: complex, z: np.ndarray, w: np.ndarray) -> complex:
    """Newton on d(λ) = Σ w_j/(λ - z_j) in numpy.longdouble (reference root)."""
    zq, wq = z.astype(np.longdouble), w.astype(np.longdouble)
    q = np.clongdouble(lam)
    for _ in range(60):
        q = q - np.sum(wq / (q - zq)) / (-np.sum(wq / (q - zq) ** 2))
    return complex(q)


def _weights_with_roots(z: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """Weights w with d(t) = Σ_j w_j/(t - z_j) = Π_i (t - ρ_i) / Π_j (t - z_j).

    Partial fractions: w_j = Π_i (z_j - ρ_i) / Π_{k≠j} (z_j - z_k) (needs #roots < m)."""
    w = np.empty(z.size)
    for j in range(z.size):
        num = np.prod(z[j] - roots) if roots.size else 1.0
        w[j] = float(np.real(num)) / np.prod(np.delete(z[j] - z, j))
    return w


# --------------------------------------------------------------------------------------
# Contract, registry, fixtures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", DATA_IDS)
@pytest.mark.parametrize("mid", METHODS)
def test_contract_on_every_dataset(mid: str, pid: str) -> None:
    ds = problems.get(pid)
    res = numopt.run(mid, ds)
    assert_valid_result(res, max_iter=100 if mid == "aaa" else None)
    assert res.converged, res.message
    assert res.method == mid
    assert res.n_iter == res.trace[-1].k
    assert res.n_fev == 0
    extra = res.extra
    for key in (
        "kind",
        "coefficients",
        "nodes",
        "values",
        "domain",
        "eval",
        "max_error",
        "node_residual",
    ):
        assert key in extra
    assert extra["kind"] == mid
    assert len(extra["eval"]["x"]) == len(extra["eval"]["y"]) == N_GRID
    assert_array_equal(res.x, extra["coefficients"])
    if ds.f_true is None:
        assert res.fun is None and extra["eval"]["f_true"] is None
    else:
        t = np.asarray(extra["eval"]["x"])
        assert res.fun == pytest.approx(float(np.max(np.abs(extra["eval"]["y"] - ds.f_true(t)))))
    # both evaluate the barycentric formula, which returns f_j exactly at a support point
    assert extra["node_residual"] == 0.0
    if mid == "aaa":
        m = res.n_iter
        assert len(res.trace) == m + 1 == extra["nodes"].size + 1
        assert len(extra["errors"]) == m + 1 and len(extra["sigma_min"]) == m
        for s in res.trace:
            assert len(s.info["curve"]) == N_GRID
            assert len(s.info["weights"]) == len(s.info["support"]) == s.k
        assert_allclose(res.trace[-1].info["curve"], extra["eval"]["y"], rtol=0, atol=0)
    else:
        assert res.n_iter == ds.x.size - 1 and extra["d"] == 3
        assert_array_equal(extra["nodes"], np.sort(ds.x))


def test_registry_metadata_matches_signatures() -> None:
    for mid in METHODS:
        spec = get_method(mid)
        assert spec.family == "interpolation" and spec.needs == ("data",)
        assert spec.references and spec.summary and spec.deterministic
        sig = inspect.signature(spec.fn)
        kwargs = {n: p.default for n, p in sig.parameters.items() if n != "problem"}
        assert {p.name: p.default for p in spec.params} == kwargs
        for p in spec.params:
            if p.kind in ("int", "float"):
                assert p.min is not None and p.max is not None and p.min <= p.default <= p.max


def test_fd_param_range_is_valid_on_every_dataset() -> None:
    """Every value of the d control (0..max) is accepted on every built-in dataset."""
    d_max = int(next(p for p in get_method("floater_hormann").params if p.name == "d").max or 0)
    for pid in DATA_IDS:
        assert problems.get(pid).x.size - 1 >= d_max
    res = numopt.run("floater_hormann", problems.get("sine_samples"), d=d_max)
    assert res.converged


def test_fixture_cases() -> None:
    cases = rm.FIXTURE_CASES
    assert 3 <= len(cases) <= 6
    assert {m for m, _, _ in cases} == set(METHODS)
    for mid, pid, params in cases:
        res = numopt.run(mid, problems.get(pid), **params)
        assert res.converged, (mid, pid, res.message)
        assert len(res.trace) < 400
        json.dumps(res.to_dict(), allow_nan=False)
        if mid == "aaa":
            # w is unique only while the Loewner matrix has ≥ m - 1 rows (see FIXTURE_CASES)
            assert 2 * res.n_iter - problems.get(pid).x.size <= 1
    from numopt.interpolation import FIXTURE_CASES as family_cases

    for case in cases:
        assert case in family_cases


def test_accepts_xy_pair_and_does_not_mutate_inputs() -> None:
    x = np.array([0.3, -1.0, 0.9, -0.2, 0.5])
    y = np.cos(2 * x)
    x0, y0 = x.copy(), y.copy()
    for mid in METHODS:
        params = {"d": 2} if mid == "floater_hormann" else {}
        r1 = numopt.run(mid, (x, y), **params)
        r2 = numopt.run(mid, (list(x), list(y)), **params)
        assert_array_equal(x, x0)
        assert_array_equal(y, y0)
        assert_array_equal(r1.x, r2.x)
        assert r1.fun is None


# --------------------------------------------------------------------------------------
# AAA: hand-computed steps and SciPy oracle
# --------------------------------------------------------------------------------------


def test_aaa_first_step_by_hand() -> None:
    """Step 1: j = argmax |F - mean F|, one support point, r ≡ f_j (w = [1])."""
    x = np.array([-1.0, -0.2, 0.3, 0.9, 1.5])
    y = np.array([0.5, 2.0, -3.0, 1.0, 0.25])
    res = rm.aaa((x, y), max_terms=1)
    s0, s1 = res.trace
    assert s0.info["sample_error"] == float(np.max(np.abs(y - y.mean())))  # = |-3 - 0.15|
    assert s1.info["node_index"] == 2
    assert s1.info["node"] == [0.3, -3.0]
    assert_array_equal(s1.info["weights"], [1.0])
    assert s1.info["sample_error"] == float(np.max(np.abs(y + 3.0)))
    assert s1.info["degree"] == 0
    assert not res.converged and res.n_iter == 1
    assert "max_terms" in res.message


def test_aaa_second_step_by_hand() -> None:
    """Step 2 on 4 samples: A^(2) is 2×2, w ∝ the null vector of the hand-built Loewner
    matrix; r(t) = (w₁f₁/(t-z₁) + w₂f₂/(t-z₂)) / (w₁/(t-z₁) + w₂/(t-z₂))."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 4.0, 10.0])
    res = rm.aaa((x, y), max_terms=2, scaling="none")
    # mean = 3.75: errors 3.75, 2.75, 0.25, 6.25 → z₁ = 3; then |y - 10| largest at x = 0
    assert [s.info["node_index"] for s in res.trace[1:]] == [3, 0]
    zs, fs = np.array([3.0, 0.0]), np.array([10.0, 0.0])
    rows = np.array([1, 2])
    loewner = (y[rows, None] - fs[None, :]) / (x[rows, None] - zs[None, :])  # (2, 2)
    _, _, vh = np.linalg.svd(loewner)
    w_ref = vh[-1] * np.sign(vh[-1][np.argmax(np.abs(vh[-1]))])
    assert_allclose(res.trace[2].info["weights"], w_ref, rtol=1e-14, atol=1e-15)
    cauchy = 1.0 / (x[rows, None] - zs[None, :])  # (2, 2)
    r = (cauchy @ (w_ref * fs)) / (cauchy @ w_ref)
    assert res.trace[2].info["sample_error"] == pytest.approx(
        float(np.max(np.abs(r - y[rows]))), rel=1e-12
    )


def test_aaa_runge_recovered_with_exact_poles_and_residues() -> None:
    """1/(1+25x²) = 1/(25(x - i/5)(x + i/5)) is type (0, 2): 3 support points recover it;
    poles ±i/5 with residues ∓i/10 (hand-computed: 1/(25·(±2i/5)))."""
    res = numopt.run("aaa", problems.get("runge_equispaced"))
    assert res.converged and res.n_iter == 3
    assert res.fun is not None and res.fun < 1e-14
    poles = np.array([complex(*p) for p in res.extra["poles"]])
    resid = np.array([complex(*r) for r in res.extra["residues"]])
    order = np.argsort(np.imag(poles))
    assert_allclose(poles[order], [-0.2j, 0.2j], rtol=0, atol=1e-14)
    assert_allclose(resid[order], [0.1j, -0.1j], rtol=0, atol=1e-14)
    assert res.extra["n_interval_poles"] == 0 and res.extra["n_doublets"] == 0


def test_aaa_plain_matches_scipy_support_sequence() -> None:
    """scaling='none' is NST Fig. 4.1; SciPy AAA (rtol=1e-13) runs the same iteration."""
    # Data without exact ties: on odd or saturating data (tanh(50x)) many samples tie for the
    # largest error, which numopt gives to the first index (TIE_RTOL) and SciPy's argmax to
    # whichever rounds largest.
    x = np.linspace(-1.0, 1.0, 2000)
    y = np.arctan(20.0 * (x - 0.1))
    mine = rm.aaa((x, y), scaling="none")
    ref = _scipy_aaa(x, y, rtol=1e-13)
    seq = [s.info["node_index"] for s in mine.trace[1:]]
    ref_seq = [int(np.flatnonzero(x == z)[0]) for z in ref.support_points]
    # The sequences agree while the sample error is far above rounding; at step 29
    # (error 9e-12) a near-tie is decided by rounding.
    assert seq[:28] == ref_seq[:28]
    # NOTE: rtol 1e-5 — the sample error max|F - N/D| is a difference of O(1) numbers, so
    # its absolute accuracy is ~1e-15·κ; at the 1e-10 level that is ~1e-6 relative.
    assert_allclose(mine.extra["errors"][1:25], ref.errors[:24], rtol=1e-5, atol=1e-14)
    t = np.linspace(-1.0, 1.0, 10_001)
    assert_allclose(_eval(mine, t), ref(t), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("fn", "z"),
    [
        (np.sqrt, _clustered_sqrt()),
        (lambda x: np.tanh(50 * x), np.linspace(-1, 1, 2000)),
    ],
)
def test_aaa_column_scaling_matches_scipy_accuracy(fn: Any, z: np.ndarray) -> None:
    """scaling='columns' reaches the accuracy of SciPy's (column-scaled) AAA."""
    y = fn(z)
    mine = rm.aaa((z, y))
    ref = _scipy_aaa(z, y, rtol=1e-13)
    t = np.unique(
        np.concatenate(
            [np.linspace(z[0], z[-1], 50_001), z[0] + np.logspace(-31, 0, 5001) * (z[-1] - z[0])]
        )
    )
    assert mine.converged
    assert float(np.max(np.abs(_eval(mine, t) - fn(t)))) < 1e-12
    assert float(np.max(np.abs(ref(t) - fn(t)))) < 1e-12
    assert_allclose(_eval(mine, t), ref(t), rtol=0, atol=2e-12)


# --------------------------------------------------------------------------------------
# AAA: properties
# --------------------------------------------------------------------------------------


def test_aaa_unscaled_breaks_down_on_clustered_sqrt() -> None:
    """The study's key finding: Fig. 4.1 without column scaling stalls on clustered √x
    samples (best sample error > 1e-10), while the scaled iteration converges."""
    z = _clustered_sqrt()
    plain = rm.aaa((z, np.sqrt(z)), scaling="none", max_terms=60)
    scaled = rm.aaa((z, np.sqrt(z)), scaling="columns")
    assert not plain.converged and plain.n_iter == 60
    assert min(plain.extra["errors"]) > 1e-10
    assert scaled.converged and scaled.extra["sample_error"] <= 1e-13


def test_aaa_sigma_min_monotone_without_scaling() -> None:
    """NST 2018, Proposition 3.1: σ_min(A^(m)) is non-increasing in m."""
    x, y = _tanh_data()
    sig = np.asarray(rm.aaa((x, y), scaling="none").extra["sigma_min"])
    # NOTE: slack 1e-12·σ_1 — each σ_min has absolute error ~eps·‖A^(m)‖.
    assert np.all(np.diff(sig) <= 1e-12 * sig[0])


def test_aaa_interpolates_support_points_and_unit_weights() -> None:
    x, y = _tanh_data()
    res = rm.aaa((x, y))
    w, z, f = res.extra["coefficients"], res.extra["nodes"], res.extra["values"]
    assert np.linalg.norm(w) == pytest.approx(1.0, rel=1e-14)
    assert_array_equal(_eval(res, z), f)
    for s in res.trace[1:]:
        ws = np.asarray(s.info["weights"])
        assert np.linalg.norm(ws) == pytest.approx(1.0, rel=1e-14)
        assert ws[np.argmax(np.abs(ws))] > 0.0  # fixed sign
    # NST 2018: tanh(50x) on 2000 points needs about two dozen support points.
    assert res.converged and 20 <= res.n_iter <= 30
    assert res.extra["n_interval_poles"] == 0


@pytest.mark.skipif(not QUAD, reason="needs IEEE quad numpy.longdouble (aarch64)")
def test_aaa_poles_match_qz_and_quad_newton() -> None:
    x, y = _tanh_data()
    res = rm.aaa((x, y), scaling="none")
    z, w, f = res.extra["nodes"], res.extra["coefficients"], res.extra["values"]
    poles, residues = rm.barycentric_poles(z, w, f)
    qz = _pencil_poles(z, w)
    assert poles.size == qz.size == z.size - 1
    for p in poles:
        exact = _quad_root(p, z, w)
        # measured: max 6.3e-10 relative (QZ itself 2.8e-9) for these far, ill-conditioned poles
        assert abs(p - exact) <= 1e-8 * abs(exact)
        assert np.min(np.abs(qz - exact)) <= 1e-7 * abs(exact)
    # tanh(50x) has poles at ±iπ(2k+1)/100 with residue 1/50
    near = np.argsort(np.abs(poles))[:2]
    assert_allclose(np.sort(np.abs(np.imag(poles[near]))), [np.pi / 100] * 2, rtol=1e-10)
    assert np.all(np.abs(np.real(poles[near])) < 1e-12)
    assert_allclose(np.abs(residues[near]), [0.02, 0.02], rtol=1e-9)


def _assert_conjugate_closed(poles: np.ndarray, rtol: float) -> None:
    """Real data → real w → the poles come in conjugate pairs (or are real)."""
    for p in poles:
        assert np.min(np.abs(poles - np.conj(p))) <= rtol * max(abs(p), 1.0), poles


@pytest.mark.parametrize("scaling", ["columns", "none"])
@pytest.mark.parametrize("power", [1, 2, 3])
def test_aaa_polynomial_data_has_no_poles(power: int, scaling: str) -> None:
    """f = x^p on symmetric points: r = f exactly, so d = c/Π(t - z_j) has no finite zero.

    The pencil (3.11) then has m + 1 infinite eigenvalues, not two (Σ_j w_j z_j^k = 0 for
    k ≤ m - 2). Regression: the extra infinite eigenvalues used to come out as spurious
    poles of size 1e5..1e9 with residues ~1e16 (QZ itself gives finite ones for x³)."""
    x = np.linspace(-1.0, 1.0, 21)
    res = rm.aaa((x, x**power), scaling=scaling)
    assert res.converged and res.n_iter == power + 1
    assert res.extra["poles"] == [] and res.extra["residues"] == []
    assert res.extra["n_doublets"] == 0 and res.extra["n_interval_poles"] == 0
    assert res.trace[-1].info["poles"] == []
    # SciPy agrees for x and x² (for x³ its QZ reports one spurious pole at ~-3.6e14)
    if power < 3:
        assert _scipy_aaa(x, x**power, rtol=1e-13).poles().size == 0


@pytest.mark.parametrize("scaling", ["columns", "none"])
def test_aaa_sine_samples_poles_match_qz_every_step(scaling: str) -> None:
    """Fixture case aaa/sine_samples: at steps 2 and 4 Σ_j w_j = 0 to rounding (odd data
    about π), so the pencil has a third infinite eigenvalue. Every step's pole list must be
    the finite QZ eigenvalues of (3.11) and closed under conjugation."""
    res = numopt.run("aaa", problems.get("sine_samples"), scaling=scaling)
    counts = []
    for s in res.trace[1:]:
        z, w = np.asarray(s.info["support"]), np.asarray(s.info["weights"])
        poles = np.array([complex(*q) for q in s.info["poles"]])
        # A moment Σ_j w_j = δ that is 0 to rounding puts a QZ eigenvalue near −|z|Σ|w_j|/δ, at
        # 1e15 or at ∞ depending on the CPU's rounding of w; numopt counts it as infinite
        # (barycentric_poles: such a pole lies beyond ~1e13·|z|). The poles of this data are of
        # size 2.7..6.3, so QZ eigenvalues beyond 1e12·max|z| are the infinite ones.
        qz = _pencil_poles(z, w)
        qz = qz[np.abs(qz) <= 1e12 * np.max(np.abs(z))]
        assert poles.size == qz.size, (s.k, poles, qz)
        counts.append(poles.size)
        # measured: max 3.0e-15 relative (poles of size 2.7..6.3, well conditioned)
        for p in poles:
            assert np.min(np.abs(qz - p)) <= 1e-12 * abs(p)
        _assert_conjugate_closed(poles, 1e-12)
        res_mag = np.abs([complex(*r) for r in s.info["residues"]])
        assert np.all(res_mag < 1e3)  # the spurious poles had |residue| ≈ 1e17
    assert counts[1] == 0 and counts[3] == 2  # m = 2: none; m = 4: 2, not 3


@PROPERTY
@given(
    m=st.integers(2, 10),
    data=st.data(),
)
def test_barycentric_poles_count_is_degree_of_numerator(m: int, data: st.DataObject) -> None:
    """Oracle: partial fractions. For roots ρ_1..ρ_q (q ≤ m - 1, real or conjugate pairs),
    w_j = Π_i (z_j - ρ_i)/Π_{k≠j}(z_j - z_k) gives d(t) = Π_i (t - ρ_i)/Π_j (t - z_j), so r
    has exactly the q poles ρ_i; the pencil has m + 1 - q infinite eigenvalues."""
    z = np.linspace(-1.0, 1.0, m) + data.draw(
        st.lists(st.floats(-0.3, 0.3), min_size=m, max_size=m)
    ) * np.full(m, 1.0 / m)
    n_pairs = data.draw(st.integers(0, (m - 1) // 2))
    n_real = data.draw(st.integers(0, m - 1 - 2 * n_pairs))
    real = np.asarray(data.draw(st.lists(st.floats(-3.0, 3.0), min_size=n_real, max_size=n_real)))
    re = np.asarray(data.draw(st.lists(st.floats(-3.0, 3.0), min_size=n_pairs, max_size=n_pairs)))
    im = np.asarray(data.draw(st.lists(st.floats(0.2, 2.0), min_size=n_pairs, max_size=n_pairs)))
    roots = np.concatenate([real, re + 1j * im, re - 1j * im])
    if roots.size:
        # well-separated simple roots away from the support points (conditioned poles)
        sep = np.abs(roots[:, None] - roots[None, :]) + 10.0 * np.eye(roots.size)
        hypothesis_assume(np.min(sep) >= 0.3)
        hypothesis_assume(np.min(np.abs(roots[:, None] - z[None, :])) >= 0.1)
    w = _weights_with_roots(z, roots)
    poles, residues = rm.barycentric_poles(z, w, np.ones(m))
    assert poles.size == residues.size == roots.size, (z, roots, poles)
    for rho in roots:
        # NOTE: tolerance 4·m·eps·κ_ρ, not a fixed rtol: a relative perturbation δ of the
        # w_j (rounding in their 2m-factor products, then in the solver) moves the simple
        # root ρ of d by ≤ δ·κ_ρ, κ_ρ = Σ_j |w_j/(ρ - z_j)| / |d'(ρ)| (first-order
        # perturbation of a simple root). κ_ρ reaches ~1e9 for 9 roots near -3 with m = 10
        # (QZ is off by 2.6e-7 there too). Measured over 50000 random cases of this
        # strategy: max error 0.60·m·eps·κ_ρ.
        kappa = np.sum(np.abs(w / (rho - z))) / abs(np.sum(w / (rho - z) ** 2))
        assert np.min(np.abs(poles - rho)) <= 4.0 * m * EPS * kappa, (rho, poles, kappa)
    # one pole per root (a bijection), so the pole set is conjugate-closed like the roots
    nearest = [int(np.argmin(np.abs(poles - rho))) for rho in roots]
    assert len(set(nearest)) == roots.size


def test_vanishing_moments_counts_degree_drop() -> None:
    """Hand-checked: w = (1, -2, 1) on s = (-1, 0, 1) has Σw = 0, Σws = 0, so q = 2
    (the polynomial-weights of 3 equispaced points: d = 2/((t+1)t(t-1)), no zeros);
    w = (1, -1) on (-1, 1): Σw = 0, q = 1 (= m - 1); w = (1, 1): q = 0."""
    assert rm._vanishing_moments(np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0])) == 2
    assert rm._vanishing_moments(np.array([-1.0, 1.0]), np.array([1.0, -1.0])) == 1
    assert rm._vanishing_moments(np.array([-1.0, 1.0]), np.array([1.0, 1.0])) == 0
    assert rm._vanishing_moments(np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.5, -1.0])) == 0
    # one rounding above the threshold is not zero: Σw = 1e-10 relative
    assert rm._vanishing_moments(np.array([-1.0, 1.0]), np.array([1.0, -1.0 + 1e-10])) == 0


@PROPERTY
@given(
    k=st.integers(1, 4),
    base=st.floats(1.2, 1.6),
    gaps=st.lists(st.floats(0.3, 0.6), min_size=3, max_size=3),
    signs=st.lists(st.sampled_from([-1.0, 1.0]), min_size=4, max_size=4),
    amps=st.lists(st.floats(0.2, 2.0), min_size=4, max_size=4),
    c=st.floats(-1.0, 1.0),
    scaling=st.sampled_from(["columns", "none"]),
)
def test_aaa_recovers_rational_functions(
    k: int,
    base: float,
    gaps: list[float],
    signs: list[float],
    amps: list[float],
    c: float,
    scaling: str,
) -> None:
    """A type (k, k) rational f is recovered with m ≤ k + 1 support points (NST 2018 §3),
    and the computed poles are its poles."""
    mags = base + np.concatenate([[0.0], np.cumsum(gaps)])[:k]  # 1.2 ≤ |p| ≤ 3.4, gaps ≥ 0.3
    p_true = np.asarray(signs[:k]) * mags
    a = np.asarray(amps[:k])

    def f(t: np.ndarray) -> np.ndarray:
        return c + np.sum(a[None, :] / (t[:, None] - p_true[None, :]), axis=1)

    x = np.linspace(-1.0, 1.0, 300)
    y = f(x)
    res = rm.aaa((x, y), scaling=scaling)
    assert res.converged
    assert res.n_iter <= k + 1
    t = np.linspace(-1.0, 1.0, 2001)
    assert_allclose(_eval(res, t), f(t), rtol=0, atol=1e-12 * np.max(np.abs(y)))
    assert res.extra["n_interval_poles"] == 0  # every pole of f lies outside [-1, 1]
    if res.n_iter == k + 1:
        got = np.array([complex(*q) for q in res.extra["poles"]])
        # NOTE: 1e-5 relative, not ~1e-13: a pole far from [-1, 1] is fixed by the data
        # only weakly (its influence decays geometrically with the distance), so it is
        # ill-conditioned. Measured: 1.3e-6 for the pole at -4.97 of a type (4, 4) f.
        for pt in p_true:
            assert np.min(np.abs(got - pt)) <= 1e-5 * abs(pt)


@PROPERTY
@given(
    a=st.floats(0.1, 10.0) | st.floats(-10.0, -0.1),
    b=st.floats(-5.0, 5.0),
    n=st.integers(20, 120),
)
def test_aaa_affine_in_f(a: float, b: float, n: int) -> None:
    """NST 2018, Proposition 3.1 (iii): AAA(a f + b) = a·AAA(f) + b with the same support
    points. Checked on the first 4 steps, where the greedy choice has no near-ties."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.exp(x) * np.sin(3 * x)
    r1 = rm.aaa((x, y), max_terms=4, scaling="none")
    r2 = rm.aaa((x, a * y + b), max_terms=4, scaling="none")
    assert [s.info["node_index"] for s in r1.trace[1:]] == [
        s.info["node_index"] for s in r2.trace[1:]
    ]
    t = np.linspace(-1.0, 1.0, 201)
    # NOTE: rtol 1e-8 — the Loewner SVD of a·f + b is solved anew; w agrees to
    # ~eps·κ(A^(m)), and κ reaches ~1e6 at step 4.
    assert_allclose(_eval(r2, t), a * _eval(r1, t) + b, rtol=1e-8, atol=1e-8 * (abs(a) + abs(b)))


# --------------------------------------------------------------------------------------
# Certified real poles
# --------------------------------------------------------------------------------------


@PROPERTY
@given(
    m=st.integers(2, 9),
    jitter=st.lists(st.floats(-0.3, 0.3), min_size=9, max_size=9),
    picks=st.lists(st.tuples(st.integers(0, 9), st.floats(0.05, 0.95)), min_size=0, max_size=8),
    pairs=st.lists(st.tuples(st.floats(-1.4, 1.4), st.floats(0.05, 1.0)), min_size=0, max_size=4),
)
def test_real_denominator_roots_counts_known_zeros(
    m: int,
    jitter: list[float],
    picks: list[tuple[int, float]],
    pairs: list[tuple[float, float]],
) -> None:
    """d with prescribed zeros: every real zero in [a, b] is bracketed, nothing else is.

    Real zeros are placed in the gaps of [a, b] = [-1.5, 1.5] cut by the support points
    (at most two per gap, at least 10% of the gap apart); complex pairs have |Im| ≥ 0.05."""
    z = np.sort(np.linspace(-1.0, 1.0, m) + np.asarray(jitter[:m]) * (1.0 / max(m - 1, 1)))
    ends = np.concatenate([[-1.5], z, [1.5]])
    real: list[float] = []
    used: dict[int, list[float]] = {}
    for gap, frac in picks:
        g = gap % (ends.size - 1)
        fr = used.setdefault(g, [])
        if len(fr) >= 2 or any(abs(frac - q) < 0.1 for q in fr):
            continue
        fr.append(frac)
        real.append(float(ends[g] + frac * (ends[g + 1] - ends[g])))
    cplx: list[complex] = []
    for re, im in pairs:
        if len(real) + len(cplx) + 2 <= m - 1:
            cplx += [complex(re, im), complex(re, -im)]
    real = real[: m - 1 - len(cplx)]
    w = _weights_with_roots(z, np.array(real + cplx, dtype=complex))
    br = rm.real_denominator_roots(z, w, -1.5, 1.5)
    assert br.shape == (len(real), 2)
    assert np.all(br[:, 0] <= br[:, 1])
    gamma = 4.0 * (m + 2) * EPS  # the rounding bound of _signed_d
    for rho in real:
        # NOTE: 1e-9 slack — the rounded weights move each zero by ~κ·eps.
        assert np.any((br[:, 0] - 1e-9 <= rho) & (rho <= br[:, 1] + 1e-9)), (rho, br)
        # The bracket shrinks to the band where |d| is below γ·Σ|w_j/(t - z_j)|; to first
        # order its half-width is γ·S(ρ)/|d'(ρ)|.
        k = int(np.argmin(np.abs(0.5 * (br[:, 0] + br[:, 1]) - rho)))
        c_rho = w / (rho - z)
        band = gamma * float(np.sum(np.abs(c_rho))) / abs(float(np.sum(c_rho / (rho - z))))
        assert br[k, 1] - br[k, 0] <= 4.0 * band + 4.0 * np.spacing(abs(rho)), (rho, br[k])


def test_real_denominator_roots_far_below_unit_scale() -> None:
    """A real pole at -4e-17 between support points ±1e-15: the eigenvalue route resolves
    poles only to ~eps absolute, the sign-change certificate resolves this one."""
    z = np.array([-1.0, -0.3, -1e-15, 1e-15, 0.4, 1.0])
    rho = np.array([-4e-17, 0.7, complex(-0.5, 1e-16), complex(-0.5, -1e-16)])
    br = rm.real_denominator_roots(z, _weights_with_roots(z, rho), -1.0, 1.0)
    mid = 0.5 * (br[:, 0] + br[:, 1])
    far = np.abs(mid + 0.5) > 1e-6  # -0.5 ± 1e-16i may or may not come out real
    assert_allclose(np.sort(mid[far]), [-4e-17, 0.7], rtol=1e-8)


def test_aaa_reports_certified_real_pole_on_step_data() -> None:
    """Honest flag: on the unit step AAA fits the 12 samples (converged=True) but r has
    real poles in [-1, 1]; the message says how many and d changes sign across each pole.

    At the last step 7 support points leave 5 rows, so the Loewner matrix (5 × 7) has a null
    space of dimension ≥ 2 (σ_min = 0) and every unit null vector interpolates the samples. The
    one the SVD returns depends on the CPU's rounding, and so does the number of real poles
    (1 on aarch64, 3 on x86-64): the test checks the count against the weights returned."""
    res = numopt.run("aaa", problems.get("step_data"))
    assert res.converged
    n = res.extra["n_interval_poles"]
    assert n >= 1 and len(res.extra["interval_poles"]) == n
    assert f"{n} certified real pole" in res.message
    assert res.fun is not None and res.fun > 1.0  # a pole sits between grid samples
    z, w = res.extra["nodes"], res.extra["coefficients"]
    for p in res.extra["interval_poles"]:
        t = np.array([p - 1e-9, p + 1e-9])
        d = np.sum(w[None, :] / (t[:, None] - z[None, :]), axis=1)
        assert np.sign(d[0]) == -np.sign(d[1])


@pytest.mark.skipif(not QUAD, reason="needs IEEE quad numpy.longdouble (aarch64)")
def test_interval_pole_count_matches_long_double_scan() -> None:
    """AAA on clustered |x| samples, 50 support points: the certified count equals an
    independent extended-precision sign scan of d on every gap."""
    z_all = np.unique(
        np.concatenate(
            [np.linspace(-1.0, 1.0, 4000), np.logspace(-15, 0, 1000), -np.logspace(-15, 0, 1000)]
        )
    )
    res = rm.aaa((z_all, np.abs(z_all)), max_terms=50)
    z = np.asarray(res.extra["nodes"])
    w = np.asarray(res.extra["coefficients"])
    order = np.argsort(z)
    zq, wq = z[order].astype(np.longdouble), w[order].astype(np.longdouble)
    ends = np.concatenate([[-1.0], z[order][(z[order] > -1.0) & (z[order] < 1.0)], [1.0]])
    frac = np.unique(np.concatenate([np.logspace(-30, -0.302, 300), np.linspace(0, 1, 1001)[1:-1]]))
    n_sign = 0
    for lo, hi in itertools.pairwise(ends):
        t = np.unique(np.concatenate([lo + frac * (hi - lo), hi - frac * (hi - lo)]))
        t = t[(t > lo) & (t < hi)].astype(np.longdouble)
        sg = np.sign((wq[None, :] / (t[:, None] - zq[None, :])).sum(axis=1))
        sg = sg[sg != 0]
        n_sign += int(np.sum(sg[1:] != sg[:-1]))
    assert not res.converged and res.n_iter == 50
    assert res.extra["n_interval_poles"] == n_sign >= 1
    # Schneider–Werner: equal signs of neighbouring weights force a zero in that gap
    ws = w[order]
    assert n_sign >= int(np.sum(np.sign(ws[1:]) == np.sign(ws[:-1])))
    for p in res.extra["interval_poles"]:
        tq = np.array([p * (1 - 1e-6), p * (1 + 1e-6)], dtype=np.longdouble)
        dq = (wq[None, :] / (tq[:, None] - zq[None, :])).sum(axis=1)
        assert np.sign(dq[0]) != np.sign(dq[1]), p


# --------------------------------------------------------------------------------------
# AAA: failure paths and edge cases
# --------------------------------------------------------------------------------------


def test_aaa_max_terms_failure() -> None:
    x = np.linspace(-1.0, 1.0, 400)
    res = rm.aaa((x, np.abs(x)), max_terms=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and res.n_iter == 5
    assert res.message.startswith("max_terms = 5 reached")
    assert res.extra["sample_error"] > 1e-13


def test_aaa_invalid_input_raises() -> None:
    x = np.linspace(-1.0, 1.0, 10)
    with pytest.raises(ValueError):
        rm.aaa(([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        rm.aaa((x, x), max_terms=0)
    with pytest.raises(ValueError):
        rm.aaa((x, x), max_terms=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rm.aaa((x, x), tol=-1.0)
    with pytest.raises(ValueError):
        rm.aaa((x, x), tol=math.nan)
    with pytest.raises(ValueError):
        rm.aaa((x, x), scaling="rows")
    with pytest.raises(ValueError):
        rm.aaa((x, np.where(x > 0, np.nan, x)))


def test_aaa_svd_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_svd(*args: Any, **kwargs: Any) -> Any:
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(rm.np.linalg, "svd", broken_svd)
    res = rm.aaa(problems.get("sine_samples"))
    assert not res.converged and "SVD" in res.message
    assert res.n_iter == 0
    json.dumps(res.to_dict(), allow_nan=False)


def test_aaa_edge_cases() -> None:
    # constant data: one support point, exact
    r = rm.aaa((np.linspace(0, 1, 5), np.full(5, 2.5)))
    assert r.converged and r.n_iter == 1
    # zero data
    r = rm.aaa((np.linspace(0, 1, 5), np.zeros(5)))
    assert r.converged and r.n_iter == 1
    # single sample
    r = rm.aaa(([0.3], [1.0]))
    assert r.converged and r.n_iter == 1
    assert_valid_result(r)
    # every sample becomes a support point (tol = 0)
    r = rm.aaa((np.array([0.0, 0.3, 0.5, 1.0]), np.array([1.0, -1.0, 2.0, 0.0])), tol=0.0)
    assert r.converged and r.extra["sample_error"] == 0.0
    assert_valid_result(r)


# --------------------------------------------------------------------------------------
# Floater–Hormann
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "delta"),
    [
        (0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (1, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]),
        (2, [1, 3, 4, 4, 4, 4, 4, 4, 4, 3, 1]),
        (3, [1, 4, 7, 8, 8, 8, 8, 8, 7, 4, 1]),
        (4, [1, 5, 11, 15, 16, 16, 16, 15, 11, 5, 1]),
    ],
)
def test_fh_equispaced_integer_weights(d: int, delta: list[int]) -> None:
    """FH 2007 §4: for equispaced nodes d!·h^d·|w_k| are the integers listed there."""
    x = np.linspace(-1.0, 1.0, 11)
    w, windows = rm.floater_hormann_weights(x, d)  # already multiplied by h^d
    assert_allclose(np.abs(w) * math.factorial(d), delta, rtol=1e-13)
    signs = np.sign(w)
    assert np.all(signs[1:] == -signs[:-1])  # weights alternate (Schneider–Werner)
    assert signs[d] > 0  # w_k carries (-1)^{k-d}
    n = x.size - 1
    assert windows == [(max(0, k - d), min(k, n - d)) for k in range(n + 1)]


def test_fh_trace_steps() -> None:
    res = numopt.run("floater_hormann", problems.get("runge_equispaced"), d=3)
    w = res.extra["coefficients"]
    for s in res.trace:
        assert s.info["node_index"] == s.k
        assert s.info["weight"] == w[s.k]
        assert_array_equal(s.x, w[: s.k + 1])
        assert ("curve" in s.info) == (s.k == res.n_iter)
    assert res.trace[-1].fun == res.fun


def test_fh_reproduces_published_runge_tables() -> None:
    """FH 2007 Table 1 (1/(1+x²) on [-5, 5], d = 3) and Table 2 (best d), 2 digits."""
    t = np.linspace(-5.0, 5.0, 100_001)
    ft = 1.0 / (1.0 + t**2)
    table1 = {10: 6.9e-02, 20: 2.8e-03, 40: 4.3e-06, 80: 5.1e-08, 160: 3.0e-09, 320: 1.8e-10}
    table2 = {10: (0, 3.6e-02), 20: (1, 1.5e-03), 40: (3, 4.3e-06), 80: (7, 2.0e-10)}
    cases = [(n, 3, e) for n, e in table1.items()] + [(n, d, e) for n, (d, e) in table2.items()]
    for n, d, published in cases:
        x = np.linspace(-5.0, 5.0, n + 1)
        res = rm.floater_hormann((x, 1.0 / (1.0 + x**2)), d=d)
        e = float(np.max(np.abs(_eval(res, t) - ft)))
        # NOTE: 5.1 % — the table prints 2 significant digits
        assert abs(e - published) <= 0.051 * published, (n, d, e, published)


@PROPERTY
@given(n=st.integers(1, 40), d=st.integers(0, 12), seed=st.integers(0, 2**32 - 1))
def test_fh_matches_scipy(n: int, d: int, seed: int) -> None:
    d = min(d, n)
    rng = np.random.default_rng(seed)  # test data only; the method is deterministic
    x = np.sort(rng.uniform(-1.0, 1.0, n + 1))
    if np.min(np.diff(x)) < 1e-3:
        x = np.sort(np.linspace(-1.0, 1.0, n + 1) + 0.3 * rng.uniform(-1, 1, n + 1) / (n + 1))
    y = np.cos(4 * x) + x
    res = rm.floater_hormann((x, y), d=d)
    assert res.converged
    t = np.linspace(x[0], x[-1], 157)
    ref = FloaterHormannInterpolator(x, y, d=d)(t)
    # NOTE: atol scaled by the Lebesgue constant: both evaluate the same barycentric
    # formula with weights that differ by the common factor h^d and by rounding.
    lam = _lebesgue_constant(x, res.extra["coefficients"], 20)
    assert_allclose(_eval(res, t), ref, rtol=0, atol=50 * EPS * lam * np.max(np.abs(y)) + 1e-14)


@PROPERTY
@given(n=st.integers(1, 30), d=st.integers(0, 10), seed=st.integers(0, 2**32 - 1))
def test_fh_reproduces_polynomials_of_degree_d(n: int, d: int, seed: int) -> None:
    """r blends degree-d interpolants with blend weights that sum to 1, so it reproduces
    every polynomial of degree ≤ d (FH 2007, eq. (4))."""
    d = min(d, n)
    rng = np.random.default_rng(seed)
    x = np.sort(np.concatenate([[-1.0, 1.0], rng.uniform(-1.0, 1.0, n - 1)]))
    if np.min(np.diff(x)) < 1e-2:
        x = np.linspace(-1.0, 1.0, n + 1)
    coef = rng.standard_normal(d + 1)
    y = np.polynomial.polynomial.polyval(x, coef)
    res = rm.floater_hormann((x, y), d=d)
    t = np.linspace(-1.0, 1.0, 101)
    exact = np.polynomial.polynomial.polyval(t, coef)
    lam = _lebesgue_constant(x, res.extra["coefficients"], 20)
    scale = max(float(np.max(np.abs(exact))), float(np.max(np.abs(y))))
    # NOTE: forward error of the second barycentric form ≲ (3n + 4)·eps·Λ·max|y|
    # (Higham 2004, Thm 3.1, with Λ the Lebesgue constant); 100(d + 1) covers n ≤ 30.
    assert_allclose(_eval(res, t), exact, rtol=0, atol=100 * EPS * lam * scale * (d + 1))


@PROPERTY
@given(n=st.integers(1, 30), d=st.integers(0, 10), seed=st.integers(0, 2**32 - 1))
def test_fh_has_no_real_poles(n: int, d: int, seed: int) -> None:
    """FH 2007, Theorem 1: Σ w_k/(x - x_k) keeps the sign of w_k on each gap
    (x_k, x_{k+1}), so it never vanishes on [x_0, x_n]; the certified root finder agrees."""
    d = min(d, n)
    rng = np.random.default_rng(seed)
    x = np.sort(np.concatenate([[-1.0, 1.0], rng.uniform(-1.0, 1.0, n - 1)]))
    if np.min(np.diff(x)) < 1e-3:
        x = np.linspace(-1.0, 1.0, n + 1)
    w, _ = rm.floater_hormann_weights(x, d)
    frac = np.linspace(0.001, 0.999, 60)
    for k in range(n):
        t = x[k] + frac * (x[k + 1] - x[k])
        den = np.sum(w[None, :] / (t[:, None] - x[None, :]), axis=1)
        assert np.all(np.sign(den) == np.sign(w[k]))
    assert rm.real_denominator_roots(x, w, -1.0, 1.0).shape == (0, 2)


def test_fh_d_equals_n_is_the_polynomial_and_d0_is_berrut() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = 1.0 / (1.0 + 25.0 * x**2)
    t = np.linspace(-1.0, 1.0, 301)
    fh = rm.floater_hormann((x, y), d=8)
    poly = barycentric((x, y))
    assert_allclose(_eval(fh, t), _eval(poly, t), rtol=1e-13, atol=1e-14)
    berrut = rm.floater_hormann((x, y), d=0)
    assert_array_equal(berrut.extra["coefficients"], (-1.0) ** np.arange(9))


def test_fh_convergence_order_on_smooth_function() -> None:
    """FH 2007, Theorem 2: error O(h^{d+1}); for sin x with d = 4 the observed order > 4.5."""
    t = np.linspace(-5.0, 5.0, 20_001)
    errs = []
    for n in (40, 80, 160):
        x = np.linspace(-5.0, 5.0, n + 1)
        res = rm.floater_hormann((x, np.sin(x)), d=4)
        errs.append(float(np.max(np.abs(_eval(res, t) - np.sin(t)))))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 4.5), orders


@settings(max_examples=1000, deadline=None)
@given(n=st.integers(2, 120), d=st.integers(1, 6))
def test_fh_lebesgue_constant_within_bos_bounds(n: int, d: int) -> None:
    """Bos, De Marchi, Hormann & Klein (2012), Theorems 1–2, equispaced nodes, n ≥ 2d:
    (1/2^{d+2}) C(2d+1, d) ln(n/d - 1) ≤ Λ_n ≤ 2^{d-1}(2 + ln n)."""
    n = max(n, 2 * d)
    x = np.linspace(0.0, 1.0, n + 1)
    w, _ = rm.floater_hormann_weights(x, d)
    lam = _lebesgue_constant(x, w, 30)
    upper = 2.0 ** (d - 1) * (2.0 + math.log(n))
    lower = math.comb(2 * d + 1, d) / 2.0 ** (d + 2) * math.log(n / d - 1.0) if n > 2 * d else 0.0
    assert lower * (1 - 1e-9) <= lam <= upper * (1 + 1e-12)


def test_fh_overflow_is_reported() -> None:
    """Nodes 1e-200 apart with d = 3: each weight factor h/|x_k - x_j| ≈ 1e199, so w_0
    overflows; the result is converged=False, not an exception."""
    x = np.array([0.0, 1e-200, 2e-200, 3e-200, 1.0])
    res = rm.floater_hormann((x, np.cos(x)), d=3)
    assert not res.converged
    assert "non-finite" in res.message


def test_fh_invalid_input_and_single_node() -> None:
    x = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        rm.floater_hormann((x, x), d=5)
    with pytest.raises(ValueError):
        rm.floater_hormann((x, x), d=-1)
    with pytest.raises(ValueError):
        rm.floater_hormann((x, x), d=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rm.floater_hormann((x, x), d=True)
    with pytest.raises(ValueError):
        rm.floater_hormann(([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]), d=1)
    r = rm.floater_hormann(([0.5], [2.0]), d=0)
    assert r.converged and r.n_iter == 0
    assert_valid_result(r)
