"""Tests for the Clenshaw–Curtis prototype (``method.py``).

Oracles (none of them reuses the DFT construction under test):
* Waldvogel's explicit cosine formula (2.4)–(2.5), summed with ``math.fsum`` in O(n²);
* the moment conditions Σ_k w_k T_j(x_k) = ∫T_j, j = 0..n, solved as a linear system;
* Trefethen's Theorem 5.2, eq. (5.4), a closed form for the error on T_{n+p};
* closed-form integrals and the paper's printed values.

Tolerances are set from ε = 2.2e-16 before running: weights are O(1/n) and sums of ≤ n terms,
so 1e-15 absolute for the weights; integrals of O(1) quantities get 1e-14 (a few ε·Σ|w f|),
and T_m evaluated through arccos carries an extra error ≈ m·ε, hence 50·m·ε for (5.4).
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


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load("research_cc_vs_gauss_method", _HERE / "method.py")
# The same contract checks as the package tests (tests/conftest.py), loaded by path.
assert_valid_result = _load(
    "research_cc_vs_gauss_conftest", _ROOT / "tests" / "conftest.py"
).assert_valid_result

CALCULUS = [p.id for p in problems.list_problems("calculus")]


# --------------------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------------------


def weights_cosine_formula(n: int) -> np.ndarray:
    """Waldvogel (2006) eq. (2.4)–(2.5): w_k = (c_k/n)[1 − Σ_j b_j/(4j² − 1)·cos(2jkπ/n)]."""
    w = np.empty(n + 1)
    for k in range(n + 1):
        c_k = 1.0 if k % n == 0 else 2.0
        s = math.fsum(
            (1.0 if 2 * j == n else 2.0) / (4 * j * j - 1) * math.cos(2 * j * k * math.pi / n)
            for j in range(1, n // 2 + 1)
        )
        w[k] = c_k / n * (1.0 - s)
    return w


def chebyshev_integral(j: int) -> float:
    """∫₋₁¹ T_j(x) dx = 2/(1 − j²) for even j, 0 for odd j."""
    return 0.0 if j % 2 else 2.0 / (1.0 - j * j)


def cheb_T(j: int):
    return lambda x: np.cos(j * np.arccos(np.clip(x, -1.0, 1.0)))


def rule_sum(f, n: int) -> float:
    t, w = cc.clenshaw_curtis_rule(n)
    return math.fsum((w * f(t)).tolist())


def eq_5_4(n: int, p: int) -> float:
    """Trefethen (2008) eq. (5.4): I(T_{n+p}) − I_n(T_{n+p})."""
    if (n + p) % 2:
        return 0.0
    return 8.0 * p * n / (n**4 - 2 * (p * p + 1) * n**2 + (p * p - 1) ** 2)


# --------------------------------------------------------------------------------------
# The rule: weights and nodes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [*range(1, 65), 100, 127, 128, 255, 256, 1000, 1024])
def test_dft_weights_match_cosine_formula(n):
    assert_allclose(cc.clenshaw_curtis_weights(n), weights_cosine_formula(n), rtol=0, atol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 20, 32])
def test_weights_solve_the_moment_system(n):
    """Independent oracle: Σ_k w_k T_j(x_k) = ∫T_j for j = 0..n defines w uniquely."""
    t = np.cos(np.arange(n + 1) * np.pi / n)
    A = np.cos(np.outer(np.arange(n + 1), np.arange(n + 1)) * np.pi / n)  # A[j, k] = T_j(x_k)
    moments = np.array([chebyshev_integral(j) for j in range(n + 1)])
    w = np.linalg.solve(A, moments)
    # NOTE: cond(A) ≤ 2n here (a scaled DCT-I matrix), so ≈ log10(2n) digits are lost: 1e-14.
    assert np.linalg.cond(A) < 2 * n + 2
    assert_allclose(cc.clenshaw_curtis_weights(n), w, rtol=0, atol=1e-14)
    assert_allclose(cc.clenshaw_curtis_nodes(n), t, rtol=0, atol=2 * EPS)


def test_small_rules_by_hand():
    """n = 1 is the trapezoid rule, n = 2 Simpson's rule, n = 3 has w₀ = 1/9 (eq. 2.6)."""
    assert_array_equal(cc.clenshaw_curtis_weights(1), [1.0, 1.0])
    assert_allclose(cc.clenshaw_curtis_weights(2), [1 / 3, 4 / 3, 1 / 3], rtol=2 * EPS)
    assert_allclose(cc.clenshaw_curtis_weights(3), [1 / 9, 8 / 9, 8 / 9, 1 / 9], rtol=2 * EPS)
    assert_allclose(cc.clenshaw_curtis_nodes(3), [1.0, 0.5, -0.5, -1.0], rtol=0, atol=EPS)


@settings(max_examples=1000, deadline=None)
@given(st.integers(1, 2000))
def test_weights_positive_symmetric_sum_to_two(n):
    """Imhof (1963): CC weights are positive; Σ w = 2; w_k = w_{n−k}; w_0 = 1/(n² − 1 + n mod 2)."""
    w = cc.clenshaw_curtis_weights(n)
    assert (w > 0).all()
    assert_array_equal(w, w[::-1])
    assert abs(math.fsum(w.tolist()) - 2.0) <= 8 * EPS
    # NOTE: the inverse DFT has an absolute error ≈ ε·log₂n·(1/n) (outputs are O(1/n)); the tiny
    # end weight w₀ ≈ 1/n² therefore has a *relative* error up to ≈ 1e-12 at n ≈ 1000.
    assert abs(w[0] - 1.0 / (n * n - 1 + n % 2)) <= 16 * EPS * math.log2(n + 1) / n


@settings(max_examples=1000, deadline=None)
@given(st.integers(1, 4096))
def test_nodes_are_nested_bit_for_bit_and_antisymmetric(n):
    t, t2 = cc.clenshaw_curtis_nodes(n), cc.clenshaw_curtis_nodes(2 * n)
    assert_array_equal(t, t2[::2])
    assert_array_equal(t, -t[::-1])
    assert t[0] == 1.0 and t[-1] == -1.0
    assert (np.diff(t) < 0).all()


# --------------------------------------------------------------------------------------
# The rule: exactness and Theorem 5.2
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(1, 64),
    st.lists(st.floats(-1.0, 1.0, allow_subnormal=False), min_size=66, max_size=66),
)
def test_exact_for_degree_n_and_n_plus_1_when_n_even(n, raw):
    """Exact for every p = Σ_{j≤d} a_j T_j with d = n (n odd) or n + 1 (n even, by symmetry)."""
    d = n + 1 if n % 2 == 0 else n
    a = np.array(raw[: d + 1])
    t, w = cc.clenshaw_curtis_rule(n)
    values = np.polynomial.chebyshev.chebval(t, a)  # (n+1,)
    exact = math.fsum(a[j] * chebyshev_integral(j) for j in range(d + 1))
    assert abs(math.fsum((w * values).tolist()) - exact) <= 1e-14 * max(1.0, float(np.abs(a).sum()))


@pytest.mark.parametrize("n", [2, 4, 10, 50])
def test_not_exact_one_even_degree_higher(n):
    """For even n, T_{n+2} is integrated with the error 16n/(n⁴ − 10n² + 9) (eq. 5.4, p = 2);
    negative for n = 2 (−32/15), positive for n ≥ 4."""
    err = chebyshev_integral(n + 2) - rule_sum(cheb_T(n + 2), n)
    assert abs(err) > 1e-5
    assert_allclose(err, 16.0 * n / (n**4 - 10 * n**2 + 9), rtol=1e-11)


@settings(max_examples=1000, deadline=None)
@given(st.data())
def test_theorem_5_2_aliasing_error(data):
    """Trefethen (2008) Thm. 5.2: T_{n+p} = T_{n−p} on the grid, error given by (5.4)."""
    n = data.draw(st.integers(2, 200))
    p = data.draw(st.integers(0, n))
    m = n + p
    err = chebyshev_integral(m) - rule_sum(cheb_T(m), n)
    assert abs(err - eq_5_4(n, p)) <= 50 * m * EPS
    t = cc.clenshaw_curtis_nodes(n)
    assert_allclose(cheb_T(m)(t), cheb_T(n - p)(t), rtol=0, atol=50 * m * EPS)


def test_paper_n50_aliasing_numbers():
    """§5: with n = 50, CC errors on T52, T60, T70, T80, T90 are 'about 0.0001, 0.0006, 0.002,
    0.006, 0.02'; Gauss (51 points) integrates up to T101 exactly but errs by ≈ 1.6 on T102.

    # NOTE: eq. (5.4) gives 1.29e-4, 6.95e-4, 1.82e-3, 4.70e-3, 2.00e-2: the prose value 0.006 for
    # T80 is 1.28× the formula's 0.0047, so the prose numbers are checked within a factor 1.5 and
    # the formula exactly (test above). The paper prints the T102 Gauss error as −1.6; with
    # error := I − I_n (the paper's own (5.4) convention) we get +1.563.
    """
    n = 50
    errs = [chebyshev_integral(j) - rule_sum(cheb_T(j), n) for j in (52, 60, 70, 80, 90)]
    for got, prose in zip(errs, (1e-4, 6e-4, 2e-3, 6e-3, 2e-2), strict=True):
        assert prose / 1.5 <= got <= prose * 1.5
    t, w = gauss_legendre_rule(n + 1)
    g101 = chebyshev_integral(101) - math.fsum((w * cheb_T(101)(t)).tolist())
    g102 = chebyshev_integral(102) - math.fsum((w * cheb_T(102)(t)).tolist())
    assert abs(g101) < 1e-13
    assert 1.5 < g102 < 1.6


def test_coefficient_formula_equals_weight_formula():
    """Two formulations of I_n: Σ w_k f(x_k) (2.2) and Σ_{j even} a_j·2/(1 − j²) (Trefethen's code)."""
    for f in (np.exp, np.cos, lambda x: 1.0 / (1.0 + 16.0 * x * x), lambda x: np.abs(x) ** 3):
        for n in (1, 2, 3, 7, 16, 64, 255, 1024):
            t, w = cc.clenshaw_curtis_rule(n)
            via_w = math.fsum((w * f(t)).tolist())
            via_a = cc.integral_from_coefficients(cc.chebyshev_coefficients(f(t)))
            assert abs(via_w - via_a) <= 1e-14


@pytest.mark.parametrize("n", [1, 2, 5, 16])
def test_chebyshev_coefficients_recover_T_j_and_show_aliasing(n):
    t = cc.clenshaw_curtis_nodes(n)
    for j in range(n + 1):
        assert_allclose(
            cc.chebyshev_coefficients(cheb_T(j)(t)), np.eye(n + 1)[j], rtol=0, atol=1e-14
        )
    for p in range(1, n + 1):  # T_{n+p} aliases to T_{n−p} (eq. 5.2)
        assert_allclose(
            cc.chebyshev_coefficients(cheb_T(n + p)(t)), np.eye(n + 1)[n - p], rtol=0, atol=1e-13
        )


def test_paper_printed_values():
    """§2: clenshaw_curtis(@cos, 11) prints 1.68294196961579 (15 digits); n = 10 does not.
    §3: x²⁰ is exact for n ≥ 20 (not 19). §3 / Clenshaw–Curtis (1960): ∫|x + ½|^½ with n = 64
    errs by 0.00078."""
    assert f"{rule_sum(np.cos, 11):.15g}" == "1.68294196961579"
    assert f"{rule_sum(np.cos, 10):.15g}" != "1.68294196961579"
    assert abs(rule_sum(lambda x: x**20, 20) - 2 / 21) <= 2 * EPS
    assert abs(rule_sum(lambda x: x**20, 19) - 2 / 21) > 1e-9
    exact = (2 / 3) * (0.5**1.5 + 1.5**1.5)
    assert f"{abs(rule_sum(lambda x: np.sqrt(np.abs(x + 0.5)), 64) - exact):.2g}" == "0.00078"


# --------------------------------------------------------------------------------------
# The method: contract, first step, counts, convergence, failure paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pid", CALCULUS)
def test_contract_on_every_calculus_problem(pid):
    res = cc.clenshaw_curtis(problems.get(pid))
    assert_valid_result(res, max_iter=12)
    assert res.n_iter == res.trace[-1].k
    assert res.n_fev == res.trace[-1].info["n"] + 1  # nesting: every old node is reused
    assert sum(s.info["new_nodes"] for s in res.trace) == res.n_fev
    for s in res.trace:
        info = s.info
        assert info["estimate"] == s.fun == s.x
        assert len(info["nodes"]) == len(info["weights"]) == info["n_points"] == info["n"] + 1
        assert abs(
            math.fsum(w * v for w, (_, v) in zip(info["weights"], info["nodes"], strict=True))
            - s.fun
        ) <= 1e-14 * max(1.0, abs(s.fun))
        xs = [x for x, _ in info["nodes"]]
        a, b = problems.get(pid).domain
        assert xs[0] == b and xs[-1] == a and all(a <= x <= b for x in xs)


def test_first_step_is_simpson_by_hand():
    """Step 0 with n = 2 on [0, 1]: (1/6)(f(0) + 4f(½) + f(1)); n = 1 gives the trapezoid rule."""
    res = cc.clenshaw_curtis(problems.get("exp_0_1"), n=2)
    assert_allclose(res.trace[0].fun, (1.0 + 4.0 * math.exp(0.5) + math.e) / 6.0, rtol=1e-15)
    assert res.trace[0].info["err_est"] is None
    res1 = cc.clenshaw_curtis(problems.get("exp_0_1"), n=1)
    assert_allclose(res1.trace[0].fun, (1.0 + math.e) / 2.0, rtol=1e-15)
    assert_allclose(res1.trace[1].fun, (1.0 + 4.0 * math.exp(0.5) + math.e) / 6.0, rtol=1e-15)


@pytest.mark.parametrize("pid", CALCULUS)
def test_err_est_is_the_documented_difference_and_stop_is_the_first_pass(pid):
    tol = 1e-10
    res = cc.clenshaw_curtis(problems.get(pid), tol=tol)
    for k in range(1, len(res.trace)):
        assert res.trace[k].info["err_est"] == abs(res.trace[k].fun - res.trace[k - 1].fun)
    passes = [
        k >= 2 and s.info["err_est"] <= tol * max(1.0, abs(s.fun)) for k, s in enumerate(res.trace)
    ]
    assert res.converged == passes[-1]
    assert not any(passes[:-1])


@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "runge", "gaussian", "arctan_deriv"])
@pytest.mark.parametrize("tol", [1e-6, 1e-10, 1e-13])
def test_converges_and_meets_tolerance_on_analytic_integrands(pid, tol):
    p = problems.get(pid)
    res = cc.clenshaw_curtis(p, tol=tol)
    assert res.converged
    assert abs(res.x - p.exact) <= tol * max(1.0, abs(p.exact))


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")  # kink/sqrt problems
def test_matches_scipy_quad_on_every_calculus_problem():
    from scipy.integrate import quad

    for pid in CALCULUS:
        p = problems.get(pid)
        res = cc.clenshaw_curtis(p, max_levels=14, tol=1e-12)
        ref, _ = quad(p.f, *p.domain, epsabs=1e-14, epsrel=1e-14, limit=500)
        assert abs(res.x - ref) <= 1e-8 * max(1.0, abs(ref)), pid


def test_convergence_on_sqrt_is_algebraic_but_honest():
    """√x on [0, 1]: algebraic convergence; the run still stops only when d_k ≤ tol."""
    p = problems.get("sqrt_0_1")
    res = cc.clenshaw_curtis(p, tol=1e-8)
    assert res.converged
    assert res.extra["error"] <= res.extra["err_est"]


def test_max_levels_reached_reports_failure():
    res = cc.clenshaw_curtis(problems.get("abs_kink"), max_levels=4, tol=1e-12)
    assert not res.converged
    assert "max_levels=4" in res.message
    assert res.n_iter == 4 and len(res.trace) == 5
    assert res.n_fev == 2 * 2**4 + 1


def test_nonfinite_integrand_reports_failure():
    res = cc.clenshaw_curtis(lambda x: 1.0 / x, bracket=(-1.0, 1.0))  # node x = 0 at step 0
    assert not res.converged
    assert "not finite" in res.message
    assert_valid_result(res)


def test_endpoint_singularity_is_evaluated_at_the_exact_end_point():
    res = cc.clenshaw_curtis(lambda x: math.sqrt(x), bracket=(0.0, 1.0), tol=1e-6)
    assert res.converged
    assert abs(res.x - 2.0 / 3.0) < 1e-6


def test_documented_blind_spot_false_convergence():
    """f = 1 − T₈(x)² vanishes at every node of n = 2, 4, 8: I_n = 0, d = 0, wrong 'converged'."""
    t8 = cheb_T(8)
    res = cc.clenshaw_curtis(lambda x: 1.0 - t8(x) ** 2, n=2)
    assert res.converged and res.n_iter == 2
    assert all(abs(s.fun) < 1e-14 for s in res.trace)
    # true value: ∫(1 − T₈²) = 1 − ∫T₁₆/2 = 1 − 1/(1 − 256) ≈ 1.0039
    assert abs(res.x - (1.0 - 1.0 / (1.0 - 256.0))) > 1.0


def test_bracket_overrides_domain_and_callable_has_no_error():
    res = cc.clenshaw_curtis(np.exp, bracket=(0.0, 2.0))
    assert res.converged and res.extra["error"] is None
    assert_allclose(res.x, math.expm1(2.0), rtol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": 2.5},
        {"max_levels": 1},
        {"tol": 0.0},
        {"tol": math.nan},
        {"n": 64, "max_levels": 16},
        {"bracket": (1.0, 0.0)},
        {"bracket": (0.0, math.inf)},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        cc.clenshaw_curtis(np.exp, **kwargs)


def test_params_cover_every_keyword():
    import inspect

    sig = inspect.signature(cc.clenshaw_curtis)
    keywords = {k for k, v in sig.parameters.items() if v.kind is inspect.Parameter.KEYWORD_ONLY}
    assert keywords - {"bracket"} == {p.name for p in cc.PARAMS["clenshaw_curtis"]}
    for p in cc.PARAMS["clenshaw_curtis"]:
        assert sig.parameters[p.name].default == p.default
