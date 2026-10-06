"""Tests for the ``data`` problem kind (datasets for interpolation and regression)."""

import json
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy import stats

from numopt import problems
from numopt.core.rng import Rng
from numopt.core.types import Dataset
from numopt.interpolation.methods import NODE_RTOL
from numopt.problems import data as data_mod

REQUIRED = (
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
NOISE_FREE = ("runge_equispaced", "runge_chebyshev", "sine_samples", "step_data")


@pytest.mark.parametrize("pid", REQUIRED)
def test_dataset_contract(pid):
    d = problems.get(pid)
    assert isinstance(d, Dataset)
    assert problems.kind_of(pid) == "data"
    assert d.x.ndim == d.y.ndim == 1 and d.x.size == d.y.size >= 2
    assert d.x.dtype == d.y.dtype == np.float64
    assert np.all(np.isfinite(d.x)) and np.all(np.isfinite(d.y))
    assert np.unique(d.x).size == d.x.size, "nodes must be distinct"
    assert d.domain is not None
    a, b = d.domain
    assert a < b and a <= d.x.min() and d.x.max() <= b
    assert d.name and d.latex and d.description
    # Read-only: no method can corrupt the shared library copy.
    assert not d.x.flags.writeable and not d.y.flags.writeable
    with pytest.raises(ValueError):
        d.x[0] = 1.0
    json.dumps(d.to_dict(), allow_nan=False)


def test_every_data_problem_is_listed():
    ids = {p.id for p in problems.list_problems("data")}
    assert set(REQUIRED) <= ids


@pytest.mark.parametrize("pid", NOISE_FREE)
def test_noise_free_values_equal_f_true(pid):
    d = problems.get(pid)
    assert d.f_true is not None
    assert_allclose(d.y, d.f_true(d.x), rtol=0, atol=0)
    # Also works on scalars (methods evaluate f_true point by point).
    assert float(d.f_true(float(d.x[3]))) == d.y[3]


def test_runge_nodes():
    eq = problems.get("runge_equispaced")
    assert_allclose(eq.x, np.linspace(-1, 1, 11), rtol=0, atol=0)
    ch = problems.get("runge_chebyshev")
    assert ch.x.size == 11 and np.all(np.diff(ch.x) > 0)
    # The nodes are the roots of T_11(x) = cos(11 arccos x) ...
    assert_allclose(np.cos(11 * np.arccos(ch.x)), 0.0, atol=1e-14)
    # ... and symmetric about 0, with the middle node at 0 (11 is odd).
    assert_allclose(ch.x, -ch.x[::-1], atol=1e-15)
    assert abs(ch.x[5]) < 1e-15
    assert_allclose(eq.y, 1.0 / (1.0 + 25.0 * eq.x**2), rtol=1e-15)


@given(st.integers(1, 60), st.floats(-10, 10), st.floats(0.1, 10))
@settings(max_examples=1000, deadline=None)
def test_chebyshev_nodes_are_roots_of_T_n(n, a, width):
    b = a + width
    x = data_mod.chebyshev_nodes_first_kind(n, a, b)
    assert x.size == n and np.all(np.diff(x) > 0) and a < x[0] and x[-1] < b
    u = (2 * x - a - b) / (b - a)
    # T_n(u_j) = cos(n arccos u_j) = cos((2j+1)π/2) = 0. Error budget: rounding x to
    # [a, b] and mapping back perturbs u by ~eps·(1 + (|a|+|b|)/(b-a)), and
    # |T_n'(u_j)| = n/sin θ_j ≤ 2n²/π at the outermost node.
    eps = np.finfo(float).eps
    atol = 4 * eps * n * n * (1 + (abs(a) + abs(b)) / (b - a))
    assert_allclose(np.cos(n * np.arccos(np.clip(u, -1, 1))), 0.0, atol=atol)


def test_chebyshev_nodes_rejects_zero():
    with pytest.raises(ValueError):
        data_mod.chebyshev_nodes_first_kind(0)


def test_step_data_has_no_node_at_the_jump():
    d = problems.get("step_data")
    assert 0.0 not in d.x
    assert set(np.unique(d.y)) == {0.0, 1.0}
    assert np.all(np.diff(d.y) >= 0)


def test_anscombe_published_statistics():
    # Anscombe (1973): mean x = 9, var x = 11, mean y = 7.50, var y = 4.127,
    # regression line y = 3.00 + 0.500 x, correlation 0.816.
    d = problems.get("anscombe_1")
    assert d.x.size == 11 and d.f_true is None
    assert math.isclose(d.x.mean(), 9.0, abs_tol=1e-12)
    assert math.isclose(d.x.var(ddof=1), 11.0, abs_tol=1e-12)
    assert math.isclose(d.y.mean(), 7.50, abs_tol=5e-3)
    assert math.isclose(d.y.var(ddof=1), 4.127, abs_tol=5e-3)
    lr: Any = stats.linregress(d.x, d.y)
    assert math.isclose(lr.intercept, 3.00, abs_tol=5e-3)
    assert math.isclose(lr.slope, 0.500, abs_tol=5e-4)
    assert math.isclose(lr.rvalue, 0.816, abs_tol=5e-4)


@pytest.mark.parametrize(
    ("pid", "builder", "seed", "std"),
    [
        ("noisy_linear", data_mod.build_noisy_linear, data_mod.NOISY_LINEAR_SEED, 0.5),
        ("noisy_quadratic", data_mod.build_noisy_quadratic, data_mod.NOISY_QUADRATIC_SEED, 0.4),
        ("outliers_linear", data_mod.build_outliers_linear, data_mod.OUTLIERS_LINEAR_SEED, 0.3),
    ],
)
def test_noise_is_seeded_mulberry32(pid, builder, seed, std):
    d = problems.get(pid)
    again = builder()
    assert_allclose(again.y, d.y, rtol=0, atol=0)  # deterministic rebuild
    rng = Rng(seed)
    noise = np.array([rng.normal(0.0, std) for _ in range(d.x.size)])
    expected = d.f_true(d.x) + noise
    for i, offset in data_mod.OUTLIER_OFFSETS if pid == "outliers_linear" else ():
        expected[i] += offset
    assert_allclose(d.y, expected, rtol=0, atol=1e-15)


def test_exponential_growth_noise():
    d = problems.get("exponential_growth")
    rng = Rng(data_mod.EXPONENTIAL_GROWTH_SEED)
    eps = np.array([rng.normal(0.0, 0.05) for _ in range(d.x.size)])
    assert_allclose(np.log(d.y), np.log(2.0) + 0.3 * d.x + eps, rtol=0, atol=1e-14)
    # A line fitted to log y recovers log 2 and 0.3 to within the noise level.
    slope, intercept = np.polyfit(d.x, np.log(d.y), 1)
    assert abs(slope - 0.3) < 0.02 and abs(intercept - math.log(2.0)) < 0.1


def test_outliers_are_exactly_two_gross_points():
    d = problems.get("outliers_linear")
    resid = d.y - d.f_true(d.x)
    gross = np.flatnonzero(np.abs(resid) > 5 * 0.3)
    assert gross.tolist() == [i for i, _ in data_mod.OUTLIER_OFFSETS]
    clean = np.delete(resid, gross)
    assert np.max(np.abs(clean)) < 4 * 0.3


def test_noise_levels_are_plausible():
    # Sample standard deviation of the Gaussian noise within a factor 2 of the nominal σ.
    for pid, std in (("noisy_linear", 0.5), ("noisy_quadratic", 0.4)):
        d = problems.get(pid)
        s = float(np.std(d.y - d.f_true(d.x), ddof=1))
        assert std / 2 < s < 2 * std


def test_exponential_growth_truth_is_the_median_not_the_mean():
    # y = f_true(x)·exp(ε), ε ~ N(0, σ²): f_true is the median of y | x, and the mean is
    # f_true·exp(σ²/2) (log-normal). The module docstring used to call f_true E[y | x].
    d = problems.get("exponential_growth")
    rng = Rng(data_mod.EXPONENTIAL_GROWTH_SEED)
    eps = np.array([rng.normal(0.0, 0.05) for _ in range(d.x.size)])
    assert d.f_true is data_mod.exponential_truth
    assert_allclose(d.y / d.f_true(d.x), np.exp(eps), rtol=1e-14)  # median of exp(ε) is 1
    mean_factor = float(stats.lognorm(s=0.05).mean())
    assert mean_factor == pytest.approx(math.exp(0.05**2 / 2), rel=1e-15)
    assert mean_factor - 1 == pytest.approx(0.00125, rel=1e-3)
    doc = data_mod.__doc__ or ""
    assert "median of y | x for ``exponential_growth``" in doc


def test_outliers_linear_description_matches_the_robust_fits():
    d = problems.get("outliers_linear")
    assert "18 clean points" in d.description and "recover the line" not in d.description


@pytest.mark.parametrize(
    "pid", ["noisy_linear", "noisy_quadratic", "outliers_linear", "exponential_growth"]
)
def test_noisy_data_are_not_samples_of_f_true(pid):
    # chebyshev_interpolation resamples f_true only when max|f_true(x_i) - y_i| is within
    # its node tolerance; the noise must be far above it on every noisy set.
    d = problems.get(pid)
    assert d.f_true is not None
    deviation = float(np.max(np.abs(d.f_true(d.x) - d.y)))
    assert deviation > 1e3 * NODE_RTOL * float(np.max(np.abs(d.y)))
