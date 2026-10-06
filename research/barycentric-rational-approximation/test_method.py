"""Tests for AAA and Floater–Hormann (research/barycentric-rational-approximation/method.py).

Oracles: scipy.interpolate.AAA / FloaterHormannInterpolator (SciPy is a test-only
dependency), the QZ solution of the arrowhead pencil (scipy.linalg.eigvals), Newton roots
of d(λ) in IEEE quad precision (numpy.longdouble on this aarch64 box), the published
tables of Floater & Hormann (2007), and closed-form properties stated in the papers.
"""

from __future__ import annotations

import os

# NOTE: one BLAS thread, as in run.py. On a loaded many-core machine the multi-threaded
# OpenBLAS SVDs made two AAA tests 25-50 times slower (36 s against 1.5 s).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import importlib.util
import itertools
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import scipy.linalg
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import AAA, FloaterHormannInterpolator

from numopt.interpolation.methods import _barycentric_eval, barycentric

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("barycentric_rational_method", HERE / "method.py")
CONFTEST = _load("numopt_tests_conftest", ROOT / "tests" / "conftest.py")
assert_valid_result = CONFTEST.assert_valid_result

EPS = float(np.finfo(np.float64).eps)
PROPERTY = settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _tanh_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 2000)
    return x, np.tanh(50.0 * x)


def _scipy_aaa(x: np.ndarray, y: np.ndarray, **kw: Any) -> AAA:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return AAA(x, y, clean_up=False, **kw)


def _pencil_poles(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """QZ on the arrowhead pencil (NST 2018, eq. (3.11))."""
    m = z.size
    B = np.eye(m + 1)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1))
    E[0, 1:] = w
    E[1:, 0] = 1.0
    np.fill_diagonal(E[1:, 1:], z)
    p = scipy.linalg.eigvals(E, B)
    return p[np.isfinite(p)]


def _quad_root(lam: complex, z: np.ndarray, w: np.ndarray) -> complex:
    """Newton on d(λ) = Σ w_j/(λ - z_j) in IEEE quad precision (reference root)."""
    zq, wq = z.astype(np.longdouble), w.astype(np.longdouble)
    l = np.clongdouble(lam)
    for _ in range(60):
        l = l - np.sum(wq / (l - zq)) / (-np.sum(wq / (l - zq) ** 2))
    return complex(l)


# --------------------------------------------------------------------------------------
# AAA
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", ["columns", "none"])
def test_aaa_contract_and_convergence(scaling: str) -> None:
    x, y = _tanh_data()
    res = M.aaa((x, y), scaling=scaling)
    assert_valid_result(res, max_iter=100)
    assert res.converged
    assert res.extra["sample_error"] <= 1e-13 * np.max(np.abs(y))
    assert res.n_iter == len(res.trace) - 1 == res.extra["nodes"].size
    # NST 2018: tanh(50x) on 2000 points needs about two dozen support points.
    assert 20 <= res.n_iter <= 30
    assert res.extra["n_interval_poles"] == 0


def test_aaa_contract_on_numopt_dataset() -> None:
    from numopt import problems

    ds = problems.get("runge_equispaced")
    res = M.aaa(ds)
    assert_valid_result(res)
    assert res.converged
    # 1/(1+25x²) is rational of type (0, 2): AAA needs only 3 support points.
    assert res.n_iter == 3
    assert res.fun is not None and res.fun < 1e-12


def test_aaa_first_step_by_hand() -> None:
    """Step 1: j = argmax |F - mean F|, one support point, r ≡ f_j (w = [1])."""
    x = np.array([-1.0, -0.2, 0.3, 0.9, 1.5])
    y = np.array([0.5, 2.0, -3.0, 1.0, 0.25])
    res = M.aaa((x, y), max_terms=1)
    j = int(np.argmax(np.abs(y - y.mean())))  # = 2 (|-3 - 0.15| largest)
    s1 = res.trace[1]
    assert s1.info["node_index"] == j == 2
    assert_array_equal(s1.info["weights"], [1.0])
    assert s1.info["sample_error"] == pytest.approx(float(np.max(np.abs(y - y[j]))), rel=0, abs=0)
    assert res.trace[0].info["sample_error"] == float(np.max(np.abs(y - y.mean())))
    assert not res.converged and res.n_iter == 1
    assert "max_terms" in res.message


def test_aaa_plain_matches_scipy_support_sequence() -> None:
    """scaling='none' is Fig. 4.1; SciPy AAA (rtol=1e-13) runs the same iteration."""
    x, y = _tanh_data()
    mine = M.aaa((x, y), scaling="none")
    ref = _scipy_aaa(x, y, rtol=1e-13)
    seq = [s.info["node_index"] for s in mine.trace[1:]]
    ref_seq = [int(np.flatnonzero(x == z)[0]) for z in ref.support_points]
    # The sequences agree while the sample error is far above rounding; at step 24
    # (error 5.6e-13) a near-tie between samples 822 and 823 is decided by rounding.
    assert seq[:23] == ref_seq[:23]
    e_mine = np.asarray(mine.extra["errors"][1:24])
    # NOTE: rtol 1e-5 — the sample error max|F - N/D| is a difference of O(1) numbers,
    # so its absolute accuracy is ~1e-15·κ; at the 1e-10 level that is ~1e-6 relative.
    assert_allclose(e_mine, ref.errors[:23], rtol=1e-5, atol=1e-14)
    t = np.linspace(-1.0, 1.0, 10_001)
    assert_allclose(M.aaa_eval(mine, t), ref(t), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("fn", "z"),
    [
        (np.sqrt, np.unique(np.concatenate([np.linspace(0, 1, 1500), np.logspace(-30, 0, 600)]))),
        (lambda x: np.tanh(50 * x), np.linspace(-1, 1, 2000)),
    ],
)
def test_aaa_column_scaling_matches_scipy_accuracy(fn, z) -> None:
    """scaling='columns' reaches the same accuracy as SciPy's (column-scaled) AAA."""
    y = fn(z)
    mine = M.aaa((z, y))
    ref = _scipy_aaa(z, y, rtol=1e-13)
    t = np.unique(
        np.concatenate(
            [np.linspace(z[0], z[-1], 50_001), z[0] + np.logspace(-31, 0, 5001) * (z[-1] - z[0])]
        )
    )
    e_mine = float(np.max(np.abs(M.aaa_eval(mine, t) - fn(t))))
    e_ref = float(np.max(np.abs(ref(t) - fn(t))))
    assert mine.converged
    assert e_mine < 1e-12 and e_ref < 1e-12
    assert_allclose(M.aaa_eval(mine, t), ref(t), rtol=0, atol=2e-12)


def test_aaa_unscaled_breaks_down_on_clustered_sqrt() -> None:
    """The measured reason for the column scaling: Fig. 4.1 stalls on this sample set."""
    z = np.unique(np.concatenate([np.linspace(0, 1, 1500), np.logspace(-30, 0, 600)]))
    plain = M.aaa((z, np.sqrt(z)), scaling="none", max_terms=60)
    scaled = M.aaa((z, np.sqrt(z)), scaling="columns")
    assert not plain.converged
    assert scaled.converged
    assert min(plain.extra["errors"]) > 1e-10


def test_aaa_sigma_min_monotone_without_scaling() -> None:
    """NST 2018, Proposition 3.1: σ_min(A^(m)) is non-increasing in m."""
    x, y = _tanh_data()
    sig = np.asarray(M.aaa((x, y), scaling="none").extra["sigma_min"])
    # NOTE: slack 1e-12·σ_1 — each σ_min is computed with absolute error ~eps·‖A^(m)‖.
    assert np.all(np.diff(sig) <= 1e-12 * sig[0])


def test_aaa_interpolates_support_points_and_unit_weights() -> None:
    x, y = _tanh_data()
    res = M.aaa((x, y))
    w, z, f = res.extra["coefficients"], res.extra["nodes"], res.extra["values"]
    assert np.linalg.norm(w) == pytest.approx(1.0, rel=1e-14)
    assert_array_equal(M.aaa_eval(res, z), f)
    assert res.extra["node_residual"] == 0.0
    for s in res.trace[1:]:
        assert np.linalg.norm(s.info["weights"]) == pytest.approx(1.0, rel=1e-14)


def test_aaa_poles_match_qz_and_quad_newton() -> None:
    x, y = _tanh_data()
    res = M.aaa((x, y), scaling="none")
    z, w, f = res.extra["nodes"], res.extra["coefficients"], res.extra["values"]
    poles, residues = M.barycentric_poles(z, w, f)
    qz = _pencil_poles(z, w)
    assert poles.size == qz.size == z.size - 1
    for p in poles:
        exact = _quad_root(p, z, w)
        # measured: max 6.3e-10 relative (QZ itself: 2.8e-9) for these far, ill-conditioned poles
        assert abs(p - exact) <= 1e-8 * abs(exact)
        assert np.min(np.abs(qz - exact)) <= 1e-7 * abs(exact)
    # tanh(50x) has poles at ±iπ(2k+1)/100; AAA finds the nearest pair to ~13 digits.
    near = poles[np.argsort(np.abs(poles))[:2]]
    assert_allclose(np.sort(np.abs(near.imag)), [np.pi / 100] * 2, rtol=1e-10)
    assert np.all(np.abs(near.real) < 1e-12)
    # residue of tanh(50x) at a pole is 1/50
    assert_allclose(np.abs(residues[np.argsort(np.abs(poles))[:2]]), [0.02, 0.02], rtol=1e-9)


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
def test_aaa_recovers_rational_functions(k, base, gaps, signs, amps, c, scaling) -> None:
    """A type (k, k) rational f is recovered with m ≤ k + 1 support points (NST 2018 §3),
    and the computed poles are its poles."""
    mags = base + np.concatenate([[0.0], np.cumsum(gaps)])[:k]  # 1.2 ≤ |p| ≤ 3.4, gaps ≥ 0.3
    p_true = np.asarray(signs[:k]) * mags
    a = np.asarray(amps[:k])

    def f(t: np.ndarray) -> np.ndarray:
        return c + np.sum(a[None, :] / (t[:, None] - p_true[None, :]), axis=1)

    x = np.linspace(-1.0, 1.0, 300)
    y = f(x)
    res = M.aaa((x, y), scaling=scaling)
    assert res.converged
    assert res.n_iter <= k + 1
    t = np.linspace(-1.0, 1.0, 2001)
    assert_allclose(M.aaa_eval(res, t), f(t), rtol=0, atol=1e-12 * np.max(np.abs(y)))
    if res.n_iter == k + 1:
        got = np.array([complex(*q) for q in res.extra["poles"]])
        # NOTE: 1e-5 relative, not ~1e-13: a pole far from [-1, 1] is fixed by the data
        # only weakly (its influence decays geometrically with the distance), so its
        # condition number is large. Measured: 1.3e-6 for the pole at -4.97 of a type (4, 4)
        # f, while r matched f to 1e-12 (checked above).
        for pt in p_true:
            assert np.min(np.abs(got - pt)) <= 1e-5 * abs(pt)


@PROPERTY
@given(
    a=st.floats(0.1, 10.0) | st.floats(-10.0, -0.1),
    b=st.floats(-5.0, 5.0),
    n=st.integers(20, 120),
)
def test_aaa_affine_in_f(a, b, n) -> None:
    """NST Prop. 3.1 (affineness in f): AAA(a f + b) = a·AAA(f) + b, same support points.

    Checked on the first 4 steps, where the greedy choice has no near-ties."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.exp(x) * np.sin(3 * x)
    r1 = M.aaa((x, y), max_terms=4, scaling="none")
    r2 = M.aaa((x, a * y + b), max_terms=4, scaling="none")
    s1 = [s.info["node_index"] for s in r1.trace[1:]]
    s2 = [s.info["node_index"] for s in r2.trace[1:]]
    assert s1 == s2
    t = np.linspace(-1.0, 1.0, 201)
    assert_allclose(
        M.aaa_eval(r2, t), a * M.aaa_eval(r1, t) + b, rtol=1e-8, atol=1e-8 * (abs(a) + abs(b))
    )


def test_aaa_max_terms_failure_and_invalid_input() -> None:
    x = np.linspace(-1.0, 1.0, 400)
    res = M.aaa((x, np.abs(x)), max_terms=5)
    assert_valid_result(res, max_iter=5)
    assert not res.converged and res.n_iter == 5
    with pytest.raises(ValueError):
        M.aaa(([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        M.aaa((x, x), max_terms=0)
    with pytest.raises(ValueError):
        M.aaa((x, x), tol=-1.0)
    with pytest.raises(ValueError):
        M.aaa((x, x), scaling="rows")


def test_aaa_edge_cases() -> None:
    # constant data: one support point, exact
    r = M.aaa((np.linspace(0, 1, 5), np.full(5, 2.5)))
    assert r.converged and r.n_iter == 1
    # zero data
    r = M.aaa((np.linspace(0, 1, 5), np.zeros(5)))
    assert r.converged and r.n_iter == 1
    # single sample
    r = M.aaa(([0.3], [1.0]))
    assert r.converged and r.n_iter == 1
    assert_valid_result(r)
    # every sample becomes a support point (M = 4 < max_terms, tol = 0)
    r = M.aaa((np.array([0.0, 0.3, 0.5, 1.0]), np.array([1.0, -1.0, 2.0, 0.0])), tol=0.0)
    assert r.converged and r.extra["sample_error"] == 0.0
    assert_valid_result(r)


def _weights_with_roots(z: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """Weights w with d(t) = Σ_j w_j/(t - z_j) = Π_i (t - ρ_i) / Π_j (t - z_j).

    Partial fractions: w_j = Π_i (z_j - ρ_i) / Π_{k≠j} (z_j - z_k) (needs #roots < m)."""
    w = np.empty(z.size)
    for j in range(z.size):
        num = np.prod(z[j] - roots) if roots.size else 1.0
        den = np.prod(np.delete(z[j] - z, j))
        w[j] = float(np.real(num)) / den
    return w


@PROPERTY
@given(
    m=st.integers(2, 9),
    jitter=st.lists(st.floats(-0.3, 0.3), min_size=9, max_size=9),
    picks=st.lists(
        st.tuples(st.integers(0, 9), st.floats(0.05, 0.95)),
        min_size=0,
        max_size=8,
    ),
    pairs=st.lists(st.tuples(st.floats(-1.4, 1.4), st.floats(0.05, 1.0)), min_size=0, max_size=4),
)
def test_real_denominator_roots_counts_known_zeros(m, jitter, picks, pairs) -> None:
    """d with prescribed zeros: every real zero in [a, b] is bracketed, nothing else is.

    Real zeros are placed in the gaps of [a, b] = [-1.5, 1.5] cut by the support points
    (at most two per gap, at least 10% of the gap apart); complex pairs have |Im| ≥ 0.05."""
    z = np.linspace(-1.0, 1.0, m) + np.asarray(jitter[:m]) * (2.0 / max(m - 1, 1)) * 0.5
    z = np.sort(z)
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
    roots = np.array(real + cplx, dtype=complex)
    w = _weights_with_roots(z, roots)
    br = M.real_denominator_roots(z, w, -1.5, 1.5)
    assert br.shape == (len(real), 2)
    for rho in real:
        # NOTE: 1e-9 slack — the rounded weights move each zero by ~κ·eps.
        assert np.any((br[:, 0] - 1e-9 <= rho) & (rho <= br[:, 1] + 1e-9)), (rho, br)
    # The bracket shrinks to the band where |d| is below its rounding bound γ·Σ|w_j/(t-z_j)|,
    # γ = 4(m + 2)·eps (method.py); to first order its half-width is γ·S(ρ)/|d'(ρ)|.
    gamma = 4.0 * (m + 2) * EPS
    for rho in real:
        k = int(np.argmin(np.abs(0.5 * (br[:, 0] + br[:, 1]) - rho)))
        c = w / (rho - z)
        band = gamma * float(np.sum(np.abs(c))) / abs(float(np.sum(c / (rho - z))))
        assert br[k, 1] - br[k, 0] <= 4.0 * band + 4.0 * np.spacing(abs(rho)), (rho, br[k], band)


def test_real_denominator_roots_far_below_unit_scale() -> None:
    """A real pole at -4e-17 between support points ±1e-15 (the |x| case the eigenvalue
    route cannot resolve, whose absolute resolution is ~eps)."""
    z = np.array([-1.0, -0.3, -1e-15, 1e-15, 0.4, 1.0])
    rho = np.array([-4e-17, 0.7, complex(-0.5, 1e-16), complex(-0.5, -1e-16)])
    w = _weights_with_roots(z, rho)
    br = M.real_denominator_roots(z, w, -1.0, 1.0)
    # -0.5 ± 1e-16 i: the computed d may or may not have this pair real; count only the
    # two zeros away from it
    mid = 0.5 * (br[:, 0] + br[:, 1])
    far = np.abs(mid + 0.5) > 1e-6
    assert_allclose(np.sort(mid[far]), [-4e-17, 0.7], rtol=1e-8)


def test_interval_pole_count_matches_long_double_scan() -> None:
    """AAA on clustered |x| samples (Q1), 50 support points: the certified count equals an
    independent long-double sign scan of d on every gap (the old relative test on the
    eigenvalue estimates reported 0 here)."""
    z_all = np.unique(
        np.concatenate(
            [np.linspace(-1.0, 1.0, 4000), np.logspace(-15, 0, 1000), -np.logspace(-15, 0, 1000)]
        )
    )
    res = M.aaa((z_all, np.abs(z_all)), max_terms=50)
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
    assert res.n_iter == 50
    assert res.extra["n_interval_poles"] == n_sign >= 1
    # Schneider–Werner: equal signs of neighbouring weights force a zero in that gap
    ws = w[order]
    assert n_sign >= int(np.sum(np.sign(ws[1:]) == np.sign(ws[:-1])))
    pole = res.extra["interval_poles"]
    assert len(pole) == n_sign
    # each reported position is a sign change of d in long double (±1e-6 relative)
    for p in pole:
        tq = np.array([p * (1 - 1e-6), p * (1 + 1e-6)], dtype=np.longdouble)
        dq = (wq[None, :] / (tq[:, None] - zq[None, :])).sum(axis=1)
        assert np.sign(dq[0]) != np.sign(dq[1]), p


# --------------------------------------------------------------------------------------
# Floater–Hormann
# --------------------------------------------------------------------------------------


def test_fh_contract() -> None:
    from numopt import problems

    for pid in ("runge_equispaced", "sine_samples", "step_data"):
        res = M.floater_hormann(problems.get(pid), d=3)
        assert_valid_result(res)
        assert res.converged
        assert res.n_iter == problems.get(pid).x.size - 1


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
    """FH 2007 §4: for equispaced nodes d!·h^d·|w_k| are the integers δ_k listed there."""
    x = np.linspace(-1.0, 1.0, 11)
    w, _ = M.floater_hormann_weights(x, d)  # already multiplied by h^d (see the NOTE)
    assert_allclose(np.abs(w) * math.factorial(d), delta, rtol=1e-13)
    signs = np.sign(w)
    assert np.all(signs[1:] == -signs[:-1])  # weights alternate (Schneider–Werner)
    assert signs[d] > 0  # w_k carries (-1)^{k-d}


def test_fh_reproduces_published_runge_table() -> None:
    """FH 2007 Table 1 (Runge 1/(1+x²) on [-5, 5], d = 3) and Table 2 (best d)."""
    table1 = {
        10: 6.9e-02,
        20: 2.8e-03,
        40: 4.3e-06,
        80: 5.1e-08,
        160: 3.0e-09,
        320: 1.8e-10,
        640: 1.1e-11,
    }
    t = np.linspace(-5.0, 5.0, 100_001)
    ft = 1.0 / (1.0 + t**2)
    for n, err in table1.items():
        x = np.linspace(-5.0, 5.0, n + 1)
        res = M.floater_hormann((x, 1.0 / (1.0 + x**2)), d=3)
        e = float(
            np.max(
                np.abs(_barycentric_eval(x, res.extra["coefficients"], res.extra["values"], t) - ft)
            )
        )
        # the table prints 2 significant digits
        assert abs(e - err) <= 0.051 * err, (n, e, err)
    table2 = {10: (0, 3.6e-02), 20: (1, 1.5e-03), 40: (3, 4.3e-06), 80: (7, 2.0e-10)}
    for n, (d_best, err) in table2.items():
        x = np.linspace(-5.0, 5.0, n + 1)
        res = M.floater_hormann((x, 1.0 / (1.0 + x**2)), d=d_best)
        e = float(
            np.max(
                np.abs(_barycentric_eval(x, res.extra["coefficients"], res.extra["values"], t) - ft)
            )
        )
        assert abs(e - err) <= 0.051 * err, (n, d_best, e, err)
    # Table 2, n = 160, d = 10: 1.3e-15 (test grid not stated in the paper). On [-5, 5] with
    # this 100 001-point grid: the same float64 weights and data evaluated in quad
    # precision (truncation error) give 3.3e-16, and float64 gives 6.0e-14, the rounding
    # floor eps·Λ·max|f| (Λ = 424). A coarse float64 grid gives 7.8e-16 (101 points) or
    # 8.1e-15 (201 points), so the published value fits either explanation
    # (run.py, results/tables.md); only the bounds below are tested.
    x = np.linspace(-5.0, 5.0, 161)
    y = 1.0 / (1.0 + x**2)
    res = M.floater_hormann((x, y), d=10)
    w = res.extra["coefficients"]
    inner = ~np.isin(t, x)
    tq, xq, wq, yq = (a.astype(np.longdouble) for a in (t[inner], x, w, y))
    cq = wq[None, :] / (tq[:, None] - xq[None, :])
    e_quad = float(np.max(np.abs((cq @ yq) / cq.sum(axis=1) - 1.0 / (1.0 + tq**2))))
    assert e_quad <= 1.3e-15
    e64 = float(np.max(np.abs(_barycentric_eval(x, w, y, t) - ft)))
    lam = M.lebesgue_constant(x, w, 50)
    assert e64 <= 2.0 * EPS * lam * float(np.max(np.abs(y)))


@PROPERTY
@given(
    n=st.integers(1, 40),
    d=st.integers(0, 12),
    seed=st.integers(0, 2**32 - 1),
)
def test_fh_matches_scipy(n, d, seed) -> None:
    d = min(d, n)
    rng = np.random.default_rng(seed)  # test data only; the method itself is deterministic
    x = np.sort(rng.uniform(-1.0, 1.0, n + 1))
    if np.min(np.diff(x)) < 1e-3:
        x = np.linspace(-1.0, 1.0, n + 1) + 0.3 * (rng.uniform(-1, 1, n + 1) / (n + 1))
        x.sort()
    y = np.cos(4 * x) + x
    res = M.floater_hormann((x, y), d=d)
    t = np.linspace(x[0], x[-1], 157)  # interpolation range only (no extrapolation)
    ref = FloaterHormannInterpolator(x, y, d=d)(t)
    mine = _barycentric_eval(x, res.extra["coefficients"], y, t)
    # NOTE: atol scaled by the Lebesgue constant: both evaluate the same barycentric
    # formula with weights that differ by a common factor h^d and rounding.
    lam = M.lebesgue_constant(x, res.extra["coefficients"], 20)
    assert_allclose(mine, ref, rtol=0, atol=50 * EPS * lam * np.max(np.abs(y)) + 1e-14)


@PROPERTY
@given(n=st.integers(1, 30), d=st.integers(0, 10), seed=st.integers(0, 2**32 - 1))
def test_fh_reproduces_polynomials_of_degree_d(n, d, seed) -> None:
    """r is a blend of degree-d interpolants with blend weights summing to 1, so it
    reproduces every polynomial of degree ≤ d (FH 2007, eq. (4))."""
    d = min(d, n)
    rng = np.random.default_rng(seed)
    x = np.sort(np.concatenate([[-1.0, 1.0], rng.uniform(-1.0, 1.0, n - 1)]))
    if np.min(np.diff(x)) < 1e-2:
        x = np.linspace(-1.0, 1.0, n + 1)
    coef = rng.standard_normal(d + 1)
    y = np.polynomial.polynomial.polyval(x, coef)
    res = M.floater_hormann((x, y), d=d)
    t = np.linspace(-1.0, 1.0, 101)
    lam = M.lebesgue_constant(x, res.extra["coefficients"], 20)
    scale = float(np.max(np.abs(np.polynomial.polynomial.polyval(t, coef))))
    assert_allclose(
        _barycentric_eval(x, res.extra["coefficients"], y, t),
        np.polynomial.polynomial.polyval(t, coef),
        rtol=0,
        atol=100 * EPS * lam * max(scale, float(np.max(np.abs(y)))) * (d + 1),
    )


@PROPERTY
@given(n=st.integers(1, 30), d=st.integers(0, 10), seed=st.integers(0, 2**32 - 1))
def test_fh_has_no_real_poles(n, d, seed) -> None:
    """FH 2007, Theorem 1: the denominator Σ w_k/(x - x_k) keeps the sign of w_k on
    each gap (x_k, x_{k+1}), so it never vanishes on [x_0, x_n]."""
    d = min(d, n)
    rng = np.random.default_rng(seed)
    x = np.sort(np.concatenate([[-1.0, 1.0], rng.uniform(-1.0, 1.0, n - 1)]))
    if np.min(np.diff(x)) < 1e-3:
        x = np.linspace(-1.0, 1.0, n + 1)
    w, _ = M.floater_hormann_weights(x, d)
    frac = np.linspace(0.001, 0.999, 60)
    for k in range(n):
        t = x[k] + frac * (x[k + 1] - x[k])
        den = np.sum(w[None, :] / (t[:, None] - x[None, :]), axis=1)
        assert np.all(np.sign(den) == np.sign(w[k]))


def test_fh_d_equals_n_is_the_polynomial() -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = 1.0 / (1.0 + 25.0 * x**2)
    fh = M.floater_hormann((x, y), d=8)
    poly = barycentric((x, y))
    t = np.linspace(-1.0, 1.0, 301)
    assert_allclose(
        _barycentric_eval(x, fh.extra["coefficients"], y, t),
        _barycentric_eval(x, poly.extra["coefficients"], y, t),
        rtol=1e-13,
        atol=1e-14,
    )


def test_fh_convergence_order_on_smooth_function() -> None:
    """FH 2007, Theorem 2: error O(h^{d+1}); observed orders for sin x, d = 4 ≈ 5."""
    errs = []
    t = np.linspace(-5.0, 5.0, 20_001)
    for n in (40, 80, 160):
        x = np.linspace(-5.0, 5.0, n + 1)
        res = M.floater_hormann((x, np.sin(x)), d=4)
        errs.append(
            float(
                np.max(
                    np.abs(
                        _barycentric_eval(x, res.extra["coefficients"], np.sin(x), t) - np.sin(t)
                    )
                )
            )
        )
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 4.5), orders


def test_fh_invalid_input() -> None:
    x = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        M.floater_hormann((x, x), d=5)
    with pytest.raises(ValueError):
        M.floater_hormann((x, x), d=-1)
    with pytest.raises(ValueError):
        M.floater_hormann((x, x), d=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        M.floater_hormann(([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]), d=1)
    r = M.floater_hormann(([0.5], [2.0]), d=0)
    assert r.converged and r.n_iter == 0


# --------------------------------------------------------------------------------------
# Lebesgue constant
# --------------------------------------------------------------------------------------


def test_lebesgue_constant_polynomial_equispaced() -> None:
    """Equispaced polynomial interpolation, n = 10: Λ_10 = 29.89 (Trefethen, ATAP, Ch. 15)."""
    x = np.linspace(-1.0, 1.0, 11)
    w = barycentric((x, x)).extra["coefficients"]
    assert M.lebesgue_constant(x, w, 400) == pytest.approx(29.890, abs=0.01)


@settings(max_examples=1000, deadline=None)
@given(n=st.integers(2, 120), d=st.integers(1, 6))
def test_lebesgue_constant_bos_bounds(n, d) -> None:
    """Bos, De Marchi, Hormann & Klein (2012), Theorems 1–2, equispaced nodes, n ≥ 2d:
    (1/2^{d+2}) C(2d+1, d) ln(n/d - 1) ≤ Λ_n ≤ 2^{d-1}(2 + ln n)."""
    if n < 2 * d:
        n = 2 * d
    x = np.linspace(0.0, 1.0, n + 1)
    w, _ = M.floater_hormann_weights(x, d)
    lam = M.lebesgue_constant(x, w, 30)
    upper = 2.0 ** (d - 1) * (2.0 + math.log(n))
    lower = math.comb(2 * d + 1, d) / 2.0 ** (d + 2) * math.log(n / d - 1.0) if n > 2 * d else 0.0
    assert lower * (1 - 1e-9) <= lam <= upper * (1 + 1e-12)
