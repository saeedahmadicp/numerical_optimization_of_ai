"""Tests for the Gauss-type baselines (``baselines.py``) and the E2 / kink helpers of ``run.py``.

Oracles (none of them reuses the extension code under test):
* SciPy's hard-coded QUADPACK tables G7/K15 and G10/K21, read through the behaviour of SciPy's
  private ``_quadrature_gk15/_gk21`` (one-hot integrands), not by parsing its source;
* numopt's Golub–Welsch Gauss rule (level 1 of the Patterson sequence is G₃);
* exactness on random Legendre series up to the theoretical degree 3n + 2 (1000 draws), with the
  closed form ∫P_0 = 2, ∫P_j = 0 (j ≥ 1);
* QUADPACK's own ``neval`` for the evaluation count; Weideman–Trefethen / README values for the
  kink detectors; a hand-built error sequence for the quantization fields of R.

Tolerances are set before running: rules are stored to 17 significant digits (exact double
round-trip) and weights are positive, so Σ w|p| ≤ 2·max|p| and the rounding of a rule sum is
≈ N·ε·max|p| ≤ 127·2.2e-16 ≈ 3e-14; hence 1e-13·max(1, Σ|a_j|) for the exactness checks.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from numopt import problems
from numopt.integration.methods import gauss_legendre_rule

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
EPS = float(np.finfo(float).eps)
BINARY128 = np.finfo(np.longdouble).nmant >= 112


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gk = _load("research_cc_vs_gauss_baselines", _HERE / "baselines.py")
run = _load("research_cc_vs_gauss_run", _HERE / "run.py")
assert_valid_result = _load(
    "research_cc_vs_gauss_conftest_b", _ROOT / "tests" / "conftest.py"
).assert_valid_result

RULES = gk.patterson_rules()
CALCULUS = [p.id for p in problems.list_problems("calculus")]


def scipy_kronrod_table(fn) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and Kronrod weights of a SciPy GK rule on [−1, 1] via one-hot integrands."""
    from scipy.integrate import _quad_vec

    xs: list[float] = []

    def f(x: float) -> np.ndarray:
        e = np.zeros(32)
        e[len(xs)] = 1.0
        xs.append(x)
        return e

    v, _, _ = getattr(_quad_vec, fn)(-1.0, 1.0, f, np.linalg.norm)
    return np.array(xs), v[: len(xs)]


def legendre_moment(j: int) -> float:
    return 2.0 if j == 0 else 0.0


# --------------------------------------------------------------------------------------
# The extension algorithm against independent tables
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not BINARY128, reason="needs binary128 longdouble")
@pytest.mark.parametrize(("m", "fn"), [(7, "_quadrature_gk15"), (10, "_quadrature_gk21")])
def test_extension_of_gauss_reproduces_quadpack_kronrod(m, fn):
    """The Kronrod extension of G_m is the special case 'old = Gauss nodes' of the routine."""
    x_ref, v_ref = scipy_kronrod_table(fn)
    g, _ = gk.gauss_legendre_ld(m)
    nodes = np.sort(np.concatenate([g, gk.kronrod_patterson_extension(g)]))[::-1]
    w = gk.interpolatory_weights(nodes)
    order = np.argsort(-x_ref)
    # QUADPACK tables carry 33 digits; ours are binary128: agreement to double rounding.
    assert_allclose(nodes.astype(float), x_ref[order], rtol=0, atol=2 * EPS)
    assert_allclose(w.astype(float), v_ref[order], rtol=0, atol=2 * EPS)


def test_level_one_is_gauss_three_and_level_zero_is_midpoint():
    t, w = gauss_legendre_rule(3)
    x1, w1 = RULES[1]
    assert_allclose(x1, np.sort(t)[::-1], rtol=0, atol=EPS)
    assert_allclose(w1, w[np.argsort(-t)], rtol=0, atol=2 * EPS)
    assert_array_equal(RULES[0][0], [0.0])
    assert_array_equal(RULES[0][1], [2.0])


@pytest.mark.skipif(not BINARY128, reason="needs binary128 longdouble")
def test_stored_rules_equal_a_fresh_binary128_computation():
    fresh = gk.compute_patterson_rules()
    assert len(fresh) == len(RULES) == gk.MAX_LEVEL + 1
    for (x, w), (xs, ws) in zip(fresh, RULES, strict=True):
        assert_array_equal(x.astype(float), xs)
        assert_array_equal(w.astype(float), ws)


@pytest.mark.skipif(not BINARY128, reason="needs binary128 longdouble")
def test_extension_beyond_127_points_is_refused_not_silently_wrong():
    """The 127 → 255 extension loses the interlacing in binary128 (module NOTE): it must raise."""
    x127 = gk.compute_patterson_rules()[-1][0]
    with pytest.raises(RuntimeError, match="interlacing"):
        gk.kronrod_patterson_extension(x127)


# --------------------------------------------------------------------------------------
# The stored rules in double precision
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", range(len(RULES)))
def test_rules_are_nested_symmetric_positive(level):
    x, w = RULES[level]
    assert x.size == 2 ** (level + 1) - 1
    assert (np.diff(x) < 0).all() and (np.abs(x) < 1).all()
    assert_array_equal(x, -x[::-1])
    assert_array_equal(w, w[::-1])
    assert (w > 0).all()
    assert abs(math.fsum(w.tolist()) - 2.0) <= 8 * EPS
    if level > 0:  # bit-exact nesting: the cache in gauss_patterson relies on it
        assert set(RULES[level - 1][0].tolist()) <= set(x.tolist())


@pytest.mark.parametrize("level", range(1, len(RULES)))
def test_exact_to_degree_3n_plus_2_on_legendre_polynomials(level):
    """N = 2n + 1 points (n old) are exact through degree 3n + 2 (odd degrees by symmetry)."""
    x, w = RULES[level]
    n = (x.size - 1) // 2
    P = np.polynomial.legendre.legvander(x, 3 * n + 3).T  # (3n+4, N)
    moments = P @ w
    for j in range(3 * n + 3):
        assert abs(moments[j] - legendre_moment(j)) <= 1e-13, j
    # and not one degree higher: the defect must be separated from the 1e-13 bound above
    # (measured: 1.32, 0.371, 0.0207, 4.6e-5, 8.8e-11 for 3 … 63 points).
    # NOTE: at 127 points the defect on P_192 is 3.2e-21 (binary128), below double resolution.
    if level <= 5:
        assert abs(moments[3 * n + 3]) > 1e-11


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(1, len(RULES) - 1),
    st.lists(st.floats(-1.0, 1.0, allow_subnormal=False), min_size=200, max_size=200),
)
def test_random_legendre_series_integrated_exactly(level, raw):
    x, w = RULES[level]
    n = (x.size - 1) // 2
    a = np.array(raw[: 3 * n + 3])  # degree ≤ 3n + 2
    values = np.polynomial.legendre.legval(x, a)
    assert abs(math.fsum((w * values).tolist()) - 2.0 * a[0]) <= 1e-13 * max(
        1.0, float(np.abs(a).sum())
    )


# --------------------------------------------------------------------------------------
# gauss_patterson: contract, counts, stopping test, failure paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", CALCULUS)
def test_gauss_patterson_contract_and_nested_count(pid):
    res = gk.gauss_patterson(problems.get(pid), tol=1e-8)
    assert_valid_result(res, max_iter=gk.MAX_LEVEL)
    assert res.n_fev == res.trace[-1].info["n_points"]  # nesting: every old node is reused
    assert sum(s.info["new_nodes"] for s in res.trace) == res.n_fev
    tol = 1e-8
    passes = [
        k >= 2 and s.info["err_est"] <= tol * max(1.0, abs(s.fun)) for k, s in enumerate(res.trace)
    ]
    assert not any(passes[:-1])
    assert res.converged == passes[-1]


@pytest.mark.parametrize("tol", [1e-6, 1e-10, 1e-13])
def test_gauss_patterson_meets_tolerance_on_entire_integrands(tol):
    for pid in ("exp_0_1", "sin_0_pi", "gaussian"):
        p = problems.get(pid)
        res = gk.gauss_patterson(p, tol=tol)
        assert res.converged
        assert abs(res.x - p.exact) <= tol * max(1.0, abs(p.exact))


def test_gauss_patterson_reports_failure_at_the_largest_rule():
    res = gk.gauss_patterson(problems.get("abs_kink"), tol=1e-13)
    assert not res.converged and "127 points" in res.message
    assert res.n_iter == gk.MAX_LEVEL and res.n_fev == 127


def test_gauss_patterson_nonfinite_value_fails():
    res = gk.gauss_patterson(lambda x: 1.0 / x, tol=1e-8)  # midpoint 0 at level 0, on [−1, 1]
    assert not res.converged and "non-finite" in res.message
    assert_valid_result(res)


@pytest.mark.parametrize("kwargs", [{"tol": 0.0}, {"tol": math.nan}, {"tol": 1e-8, "max_level": 7}])
def test_gauss_patterson_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        gk.gauss_patterson(np.exp, **kwargs)


# --------------------------------------------------------------------------------------
# quadpack_qags
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", CALCULUS)
def test_quadpack_counts_equal_neval_and_contract(pid):
    res = gk.quadpack_qags(problems.get(pid), tol=1e-10)
    assert_valid_result(res)
    assert res.n_fev == res.extra["neval"]
    assert res.n_fev % 21 == 0  # G10/K21 on every subinterval


def test_quadpack_request_is_tol_times_max_one_abs_I():
    """epsabs = tol makes the request max(tol, tol·|I|); cos 50x has |I| = 0.0105 < 1, where the
    purely relative request epsabs = 0 asks for more and costs at least as much."""
    p = next(fn for fn in run.ALL if fn.id == "cos50").problem()
    a = gk.quadpack_qags(p, tol=1e-10)
    b = gk.quadpack_qags(p, tol=1e-10, epsabs=0.0)
    assert a.converged and b.converged
    assert a.n_fev <= b.n_fev
    assert a.extra["error"] <= 1e-10 * max(1.0, abs(p.exact))


def test_quadpack_failure_is_reported():
    res = gk.quadpack_qags(problems.get("abs_kink"), tol=1e-13, limit=1)
    assert not res.converged and res.message.startswith("QUADPACK:")


# --------------------------------------------------------------------------------------
# run.py helpers: quantization of R, the two kink detectors, the 3 | n artifact
# --------------------------------------------------------------------------------------


def test_efficiency_ceiling_and_normalized_gap_by_hand():
    """Gauss exact from n = 10 (11 points), CC from n = 20 (21 points): R = 21/11 = 2 − 1/11 is
    the ceiling, g = (21 − 11)/(11 − 1) = 1 (the full degree factor)."""
    fn = run.Integrand("one", "1", lambda x: np.ones_like(x), 2.0, "polynomial")
    eg = [1.0 if n < 10 else 0.0 for n in run.N_GRID]
    ec = [1.0 if n < 20 else 0.0 for n in run.N_GRID]
    out = run.efficiency(fn, eg, ec)
    for lev in out["levels"]:
        assert (lev["n_gauss"], lev["n_cc"]) == (10, 20)
        assert lev["ratio_points"] == lev["ratio_ceiling"] == 21 / 11
        assert lev["g_normalized"] == 1.0 and lev["g_resolution"] == 0.1
        assert lev["gap_test_can_fire"]
    assert out["verdict"] == "persistent factor-2 gap" and out["persistence_test_defined"]


def test_gap_test_cannot_fire_below_five_gauss_points():
    for m in range(2, 12):
        assert (2 - 1 / m >= run.GAP) == (m >= 5)


@pytest.fixture(scope="module")
def runge16_curves():
    fn = next(f for f in run.TREFETHEN if f.id == "runge16")
    ns = np.arange(1, 141)
    eg, ec = run.error_curves(fn, ns)
    return fn, ns, eg, ec


def test_kink_detectors_new_and_first_run(runge16_curves):
    """Deepest dip: 53 / 46 (W–T eqs. 27–28: n = 53.17 / 45.74). First-run rate rule: 65 / 58."""
    fn, ns, eg, ec = runge16_curves
    k = run.kink_analysis(fn, ns, eg, ec)
    assert (k["kink_n_odd"], k["kink_n_even"]) == (53, 46)
    fr = k["first_run_detector"]
    assert (fr["kink_n_odd"], fr["kink_n_even"]) == (65, 58)


def test_sqrt_abs_x_half_without_three_dividing_n():
    """Without the n that 3 divides (a node on the singularity), CC needs n = 194 for 1e-4."""
    fn = next(f for f in run.TREFETHEN if f.id == "sqrt_abs_x_half")
    rel = np.array([abs(run.quad(run.cc_rule(int(n)), fn.f) - fn.exact) for n in run.N_GRID])
    rel /= run.l1_norm(fn)
    keep = run.N_GRID % 3 != 0
    assert run.n_needed(run.N_GRID[keep], rel[keep], 1e-4) == 194
    assert run.n_needed(run.N_GRID, rel, 1e-4) == 544
