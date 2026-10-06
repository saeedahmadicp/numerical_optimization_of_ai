"""Tests of the nested rules promoted from research/clenshaw-curtis-vs-gauss: Clenshaw–Curtis and
Gauss–Kronrod–Patterson (``numopt.integration.methods``).

Oracles (none reuses the construction under test):
* Clenshaw–Curtis weights: Waldvogel's explicit cosine formula (2.4)–(2.5), summed with
  ``math.fsum`` in O(n²), and the Chebyshev moment system Σ_k w_k T_j(x_k) = ∫T_j;
* Trefethen (2008), Theorem 5.2, eq. (5.4): a closed form for the error on T_{n+p};
* Patterson rules: Gauss G₃ from Golub–Welsch, and an mpmath recomputation (300 digits, all
  levels up to 127 points) of the Kronrod–Patterson extensions from their defining
  orthogonality conditions; exactness degree
  checked in 50-digit arithmetic with a rounding bound derived from the stored doubles;
* closed-form integrals, ``scipy.integrate.quad``, and the study's reported numbers.

Tolerances are set from ε = 2.2e-16 before running: CC weights are O(1/n) and sums of ≤ n terms,
so 1e-15 absolute; integrals of O(1) quantities get 1e-14 (a few ε·Σ|w f|); T_m evaluated through
arccos carries an extra error ≈ m·ε, hence 50·m·ε for (5.4). A stored Patterson value is the
nearest double, so it is within half an ulp of the true value.
"""

import math
from itertools import pairwise

import mpmath
import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import Problem
from numopt.integration import FIXTURE_CASES as PACKAGE_FIXTURES
from numopt.integration import methods as M

EPS = float(np.finfo(float).eps)
CALCULUS = [p.id for p in problems.list_problems("calculus")]
NESTED = ("clenshaw_curtis", "gauss_patterson")
COMMON_KEYS = {"estimate", "error", "err_est", "n_points", "nodes", "weights", "new_nodes"}
INFO_KEYS = {
    "clenshaw_curtis": COMMON_KEYS | {"n", "cheb_coeffs"},
    "gauss_patterson": COMMON_KEYS,
}
#: The study's matched-level numbers (README §4.3, ``matched_level_accuracy``).
STUDY_RUNGE_CC65, STUDY_RUNGE_GKP63 = 2.9e-11, 3.1e-10
STUDY_COS50_CC65, STUDY_COS50_GKP63 = 1.1e-9, 3.1e-16


def chebyshev_integral(j: int) -> float:
    """∫₋₁¹ T_j(x) dx = 2/(1 − j²) for even j, 0 for odd j."""
    return 0.0 if j % 2 else 2.0 / (1.0 - j * j)


def cheb_T(m: int):
    return lambda t: np.cos(m * np.arccos(np.clip(t, -1.0, 1.0)))


def cc_sum(f, n: int) -> float:
    t, w = M.clenshaw_curtis_rule(n)
    return math.fsum((w * f(t)).tolist())


def eq_5_4(n: int, p: int) -> float:
    """Trefethen (2008) eq. (5.4): I(T_{n+p}) − I_n(T_{n+p})."""
    if (n + p) % 2:
        return 0.0
    return 8.0 * p * n / (n**4 - 2 * (p * p + 1) * n**2 + (p * p - 1) ** 2)


def weights_cosine_formula(n: int) -> np.ndarray:
    """Waldvogel (2006) (2.4)–(2.5), O(n²): w_k = (c_k/n)[1 − Σ_j b_j/(4j² − 1)·cos(2jkπ/n)]."""
    w = np.empty(n + 1)
    for k in range(n + 1):
        terms = [1.0]
        for j in range(1, n // 2 + 1):
            b = 1.0 if 2 * j == n else 2.0
            terms.append(-b / (4.0 * j * j - 1.0) * math.cos(2.0 * j * k * math.pi / n))
        c = 1.0 if k % n == 0 else 2.0
        w[k] = c / n * math.fsum(terms)
    return w


def _ex(prob: Problem) -> float:
    assert prob.exact is not None
    return prob.exact


def _fun(step) -> float:
    assert step.fun is not None
    return step.fun


def _problem(f, a: float, b: float, exact: float, pid: str = "custom") -> Problem:
    return Problem(id=pid, name=pid, latex="f", f=f, dim=1, domain=(a, b), exact=exact)


def _runge_family(c: float) -> Problem:
    """1/(1 + c·x²) on [−1, 1], ∫ = 2·arctan(√c)/√c."""
    return _problem(
        lambda x: 1.0 / (1.0 + c * x * x), -1.0, 1.0, 2.0 * math.atan(math.sqrt(c)) / math.sqrt(c)
    )


def _peak_family(c: float, xc: float) -> Problem:
    """exp(−c(x − x_c)²) on [0, 1], ∫ = √π/(2√c)·(erf(√c(1 − x_c)) + erf(√c·x_c))."""
    s = math.sqrt(c)
    exact = math.sqrt(math.pi) / (2.0 * s) * (math.erf(s * (1.0 - xc)) + math.erf(s * xc))
    return _problem(lambda x: math.exp(-c * (x - xc) ** 2), 0.0, 1.0, exact)


def _cos50() -> Problem:
    return _problem(lambda x: math.cos(50.0 * x), -1.0, 1.0, 2.0 * math.sin(50.0) / 50.0)


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", NESTED)
@pytest.mark.parametrize("pid", CALCULUS)
def test_contract_on_every_problem(method, pid):
    p = problems.get(pid)
    a, b = p.domain
    res = numopt.run(method, p)
    assert_valid_result(res)
    assert res.n_iter == res.trace[-1].k
    assert sum(s.info["new_nodes"] for s in res.trace) == res.n_fev
    for step in res.trace:
        info = step.info
        assert set(info) == INFO_KEYS[method]
        assert step.fun == info["estimate"] == step.x
        if info["nodes"] is None:  # the display cap of the module
            assert info["n_points"] > M.MAX_DISPLAY
            assert info["weights"] is None and info.get("cheb_coeffs") is None
            continue
        xs = [x for x, _ in info["nodes"]]
        assert len(xs) == len(info["weights"]) == info["n_points"]
        assert all(x0 > x1 for x0, x1 in pairwise(xs)), "nodes are in decreasing x"
        terms = [w * fx for w, (_, fx) in zip(info["weights"], info["nodes"], strict=True)]
        assert math.fsum(terms) == info["estimate"]
        if method == "clenshaw_curtis":
            assert xs[0] == b and xs[-1] == a, "closed rule: exact end points"
            assert info["n_points"] == info["n"] + 1
        else:
            assert a < xs[-1] and xs[0] < b, "open rule: no evaluation at a or b"
    assert res.extra["error"] == abs(res.x - p.exact)
    if res.converged:
        assert math.isfinite(res.x)


def test_promoted_fixture_cases_are_exported_and_small():
    mine = [c for c in M.FIXTURE_CASES if c[0] in NESTED]
    assert {c[0] for c in mine} == set(NESTED)
    assert 3 <= len(mine) <= 6
    for case in mine:
        assert case in PACKAGE_FIXTURES
        method, pid, params = case
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) < 400
    # the fixtures show both outcomes: convergence and the 127-point cap of Patterson
    outcomes = {numopt.run(m, problems.get(p), **kw).converged for m, p, kw in mine}
    assert outcomes == {True, False}


@pytest.mark.parametrize("method", NESTED)
def test_params_cover_every_keyword_and_every_slider_corner_runs(method):
    spec = get_method(method)
    names = {p.name for p in spec.params}
    kwonly = set(spec.fn.__kwdefaults__ or {})
    assert kwonly == names | {"bracket"}
    assert spec.family == "integration" and spec.references and spec.summary
    for p in spec.params:
        assert kwonly and spec.fn.__kwdefaults__ is not None
        assert spec.fn.__kwdefaults__[p.name] == p.default
    ints = [p for p in spec.params if p.kind == "int"]
    corners: list[dict[str, int]] = [{}]
    for p in ints:
        assert p.min is not None and p.max is not None
        corners = [{**c, p.name: v} for c in corners for v in (int(p.min), int(p.max))]
    for c in corners:  # no slider position may raise (joint-consistent ranges)
        res = numopt.run(method, problems.get("exp_0_1"), tol=1e-1, **c)
        assert_valid_result(res)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("clenshaw_curtis", {"n": 0}),
        ("clenshaw_curtis", {"n": 2.5}),
        ("clenshaw_curtis", {"max_levels": 1}),
        ("clenshaw_curtis", {"n": 128, "max_levels": 12}),
        ("clenshaw_curtis", {"tol": 0.0}),
        ("clenshaw_curtis", {"tol": math.nan}),
        ("clenshaw_curtis", {"bracket": (1.0, 0.0)}),
        ("gauss_patterson", {"max_levels": 1}),
        ("gauss_patterson", {"max_levels": 7}),
        ("gauss_patterson", {"tol": -1.0}),
        ("gauss_patterson", {"bracket": (0.0, math.inf)}),
    ],
)
def test_invalid_parameters_raise(method, kwargs):
    with pytest.raises(ValueError):
        numopt.run(method, problems.get("exp_0_1"), **kwargs)


# --------------------------------------------------------------------------------------
# Clenshaw–Curtis rule: weights, nodes, exactness, Theorem 5.2
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [*range(1, 65), 100, 127, 128, 255, 256, 1000, 1024])
def test_dft_weights_match_cosine_formula(n):
    assert_allclose(M.clenshaw_curtis_weights(n), weights_cosine_formula(n), rtol=0, atol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 20, 32])
def test_weights_solve_the_moment_system(n):
    """Σ_k w_k T_j(x_k) = ∫T_j for j = 0..n defines w uniquely."""
    A = np.cos(np.outer(np.arange(n + 1), np.arange(n + 1)) * np.pi / n)  # (n+1, n+1) T_j(x_k)
    moments = np.array([chebyshev_integral(j) for j in range(n + 1)])
    w = np.linalg.solve(A, moments)
    # NOTE: cond(A) ≤ 2n (a scaled DCT-I matrix), so ≈ log10(2n) digits are lost: 1e-14.
    assert np.linalg.cond(A) < 2 * n + 2
    assert_allclose(M.clenshaw_curtis_weights(n), w, rtol=0, atol=1e-14)
    t = np.cos(np.arange(n + 1) * np.pi / n)
    assert_allclose(M.clenshaw_curtis_nodes(n), t, rtol=0, atol=2 * EPS)


def test_small_rules_by_hand():
    """n = 1 is the trapezoid rule, n = 2 Simpson's rule, n = 3 has w₀ = 1/9 (eq. 2.6)."""
    assert_array_equal(M.clenshaw_curtis_weights(1), [1.0, 1.0])
    assert_allclose(M.clenshaw_curtis_weights(2), [1 / 3, 4 / 3, 1 / 3], rtol=2 * EPS)
    assert_allclose(M.clenshaw_curtis_weights(3), [1 / 9, 8 / 9, 8 / 9, 1 / 9], rtol=2 * EPS)
    assert_allclose(M.clenshaw_curtis_nodes(3), [1.0, 0.5, -0.5, -1.0], rtol=0, atol=EPS)


@settings(max_examples=1000, deadline=None)
@given(st.integers(1, 2000))
def test_cc_weights_positive_symmetric_sum_to_two(n):
    """Imhof (1963): CC weights are positive; Σ w = 2; w_k = w_{n−k}; w_0 = 1/(n² − 1 + n mod 2)."""
    w = M.clenshaw_curtis_weights(n)
    assert (w > 0).all()
    assert_array_equal(w, w[::-1])
    assert abs(math.fsum(w.tolist()) - 2.0) <= 8 * EPS
    # NOTE: the inverse DFT has an absolute error ≈ ε·log₂n/n (outputs are O(1/n)), so the tiny
    # end weight w₀ ≈ 1/n² has a relative error up to ≈ 1e-12 at n ≈ 1000.
    assert abs(w[0] - 1.0 / (n * n - 1 + n % 2)) <= 16 * EPS * math.log2(n + 1) / n


@settings(max_examples=1000, deadline=None)
@given(st.integers(1, 4096))
def test_cc_nodes_nested_bit_for_bit_and_antisymmetric(n):
    t, t2 = M.clenshaw_curtis_nodes(n), M.clenshaw_curtis_nodes(2 * n)
    assert_array_equal(t, t2[::2])
    assert_array_equal(t, -t[::-1])
    assert t[0] == 1.0 and t[-1] == -1.0
    assert (np.diff(t) < 0).all()


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(1, 64),
    st.lists(st.floats(-1.0, 1.0, allow_subnormal=False), min_size=66, max_size=66),
)
def test_cc_exact_for_degree_n_and_n_plus_1_when_n_even(n, raw):
    """Exact for every p = Σ_{j≤d} a_j T_j with d = n (n odd) or n + 1 (n even, by symmetry)."""
    d = n + 1 if n % 2 == 0 else n
    a = np.array(raw[: d + 1])
    t, w = M.clenshaw_curtis_rule(n)
    values = np.polynomial.chebyshev.chebval(t, a)  # (n+1,)
    exact = math.fsum(a[j] * chebyshev_integral(j) for j in range(d + 1))
    assert abs(math.fsum((w * values).tolist()) - exact) <= 1e-14 * max(1.0, float(np.abs(a).sum()))


@pytest.mark.parametrize("n", [2, 4, 10, 50])
def test_cc_not_exact_one_even_degree_higher(n):
    """For even n, T_{n+2} is integrated with the error 16n/(n⁴ − 10n² + 9) (eq. 5.4, p = 2)."""
    err = chebyshev_integral(n + 2) - cc_sum(cheb_T(n + 2), n)
    assert abs(err) > 1e-5
    assert_allclose(err, 16.0 * n / (n**4 - 10 * n**2 + 9), rtol=1e-11)


@settings(max_examples=1000, deadline=None)
@given(st.data())
def test_theorem_5_2_aliasing_error(data):
    """Trefethen (2008) Thm. 5.2: T_{n+p} = T_{n−p} on the grid; the error is given by (5.4)."""
    n = data.draw(st.integers(2, 200))
    p = data.draw(st.integers(0, n))
    m = n + p
    err = chebyshev_integral(m) - cc_sum(cheb_T(m), n)
    assert abs(err - eq_5_4(n, p)) <= 50 * m * EPS
    t = M.clenshaw_curtis_nodes(n)
    assert_allclose(cheb_T(m)(t), cheb_T(n - p)(t), rtol=0, atol=50 * m * EPS)


def test_paper_printed_value_and_x20():
    """Trefethen (2008) §2: clenshaw_curtis(@cos, 11) prints 1.68294196961579; §3: x²⁰ is exact
    for n ≥ 20 (not 19)."""
    assert f"{cc_sum(np.cos, 11):.15g}" == "1.68294196961579"
    assert f"{cc_sum(np.cos, 10):.15g}" != "1.68294196961579"
    assert abs(cc_sum(lambda x: x**20, 20) - 2 / 21) <= 2 * EPS
    assert abs(cc_sum(lambda x: x**20, 19) - 2 / 21) > 1e-9


@pytest.mark.parametrize("pid", ["exp_0_1", "runge", "gaussian", "sqrt_0_1", "abs_kink"])
def test_cheb_coeffs_integrate_to_the_estimate(pid):
    """Two formulations of I_n: Σ w_k f(x_k) (2.2) and ((b − a)/2)·Σ_{j even} a_j·2/(1 − j²)."""
    p = problems.get(pid)
    a, b = p.domain
    res = numopt.run("clenshaw_curtis", p, max_levels=6, tol=1e-300)
    for step in res.trace:
        c = step.info["cheb_coeffs"]
        assert len(c) == step.info["n"] + 1
        via_a = 0.5 * (b - a) * math.fsum(c[j] * chebyshev_integral(j) for j in range(0, len(c), 2))
        assert abs(via_a - _fun(step)) <= 1e-14 * max(1.0, abs(_fun(step)))


def test_chebyshev_coefficients_recover_T_j_and_aliasing():
    for n in (1, 2, 5, 16):
        t = M.clenshaw_curtis_nodes(n)
        for j in range(n + 1):
            assert_allclose(M.chebyshev_coefficients(cheb_T(j)(t)), np.eye(n + 1)[j], atol=1e-14)
        for p in range(1, n + 1):  # T_{n+p} aliases to T_{n−p} (Trefethen 2008, (5.2))
            got = M.chebyshev_coefficients(cheb_T(n + p)(t))
            assert_allclose(got, np.eye(n + 1)[n - p], atol=1e-13)


# --------------------------------------------------------------------------------------
# Gauss–Kronrod–Patterson rules: the stored table against independent computations
# --------------------------------------------------------------------------------------

LEVELS = range(M._GKP_MAX_LEVEL + 1)


def _gkp_degree(level: int) -> int:
    return 1 if level == 0 else 3 * 2**level - 1


#: Working precision of the Patterson recomputation. The monomial orthogonality system at 63 old
#: nodes loses ~70 digits; 300 digits leave > 200 (the recomputed 127-point rule integrates
#: P_1..P_191 to ≤ 1e-232, asserted below).
MP_DPS = 300


def _mp_patterson_extension(old: list, guess: np.ndarray) -> list:
    """The n + 1 new nodes of the optimal (Kronrod–Patterson) extension of the symmetric set
    ``old`` (n points, n odd), from the definition: q monic of degree n + 1 with
    ∫₋₁¹ ω(x)·q(x)·x^j dx = 0 for j = 0..n, ω = ∏(x − x_i). By symmetry q has the parity of
    n + 1 and only the odd j give conditions: a square system in the monomial basis, solved at
    ``MP_DPS`` digits.

    The roots of q are found by Newton's method started from ``guess`` (the stored doubles of
    the new nodes). The guesses only select which root each iteration reaches: q has degree
    n + 1, so n + 1 converged, distinct, real roots in (−1, 1) are all of its roots, and the
    result does not depend on the stored values."""
    n = len(old)
    omega = [mpmath.mpf(1)]  # ascending coefficients of ω
    for x in old:
        omega = [mpmath.mpf(0), *omega]
        for i in range(len(omega) - 1):
            omega[i] -= x * omega[i + 1]

    def moment(m: int):
        return mpmath.mpf(0) if m % 2 else mpmath.mpf(2) / (m + 1)

    def inner(i: int, j: int):  # ∫ ω(x)·x^{i+j} dx
        return mpmath.fsum(c * moment(l + i + j) for l, c in enumerate(omega))

    unknown = [i for i in range(n + 1) if i % 2 == (n + 1) % 2]
    rows = [j for j in range(n + 1) if j % 2 == 1]
    A = mpmath.matrix([[inner(i, j) for i in unknown] for j in rows])
    rhs = mpmath.matrix([-inner(n + 1, j) for j in rows])
    c = mpmath.lu_solve(A, rhs)
    q = [mpmath.mpf(0)] * (n + 2)  # ascending coefficients of q
    q[n + 1] = mpmath.mpf(1)
    for ci, i in zip(c, unknown, strict=True):
        q[i] = ci
    dq = [i * q[i] for i in range(1, n + 2)]
    step_tol = mpmath.mpf(10) ** -(MP_DPS - 50)
    roots = []
    for g in guess:
        x = mpmath.mpf(float(g))
        for _ in range(60):
            dx = mpmath.polyval(q[::-1], x) / mpmath.polyval(dq[::-1], x)
            x -= dx
            if abs(dx) < step_tol:
                break
        else:
            raise AssertionError(f"Newton did not converge from {g!r}")
        roots.append(x)
    roots.sort(reverse=True)
    assert len(roots) == n + 1 and -1 < roots[-1] and roots[0] < 1
    assert all(a - b > mpmath.mpf(10) ** -10 for a, b in pairwise(roots)), "repeated root"
    return roots


def _mp_interpolatory_weights(nodes: list) -> list:
    """w with Σ_k w_k P_j(x_k) = ∫P_j = 2·δ_{j0} for j = 0..N − 1 (Legendre moment system; the
    Legendre basis is far better conditioned than the monomial Vandermonde at 127 points)."""
    N = len(nodes)
    P = []  # P[j][k] = P_j(x_k)
    for j in range(N):
        if j == 0:
            P.append([mpmath.mpf(1)] * N)
        elif j == 1:
            P.append(list(nodes))
        else:
            P.append([((2 * j - 1) * x * a - (j - 1) * b) / j
                      for x, a, b in zip(nodes, P[-1], P[-2], strict=True)])  # fmt: skip
    mom = mpmath.matrix([mpmath.mpf(2)] + [mpmath.mpf(0)] * (N - 1))
    return list(mpmath.lu_solve(mpmath.matrix(P), mom))


def _mp_legendre_residuals(nodes: list, weights: list, deg: int) -> list:
    """|Σ w_k P_j(x_k) − ∫P_j| for j = 0..deg + 1."""
    out, p_prev, p = [], [mpmath.mpf(1)] * len(nodes), list(nodes)
    out.append(abs(mpmath.fsum(weights) - 2))
    for j in range(1, deg + 2):
        if j >= 2:
            p_prev, p = p, [((2 * j - 1) * x * a - (j - 1) * b) / j
                            for x, a, b in zip(nodes, p, p_prev, strict=True)]  # fmt: skip
        out.append(abs(mpmath.fsum(w * v for w, v in zip(weights, p, strict=True))))
    return out


def test_patterson_table_matches_mpmath_recomputation_at_every_level():
    """Levels 0–6 (up to 127 points) recomputed from the definition at 300 digits. The
    recomputed rule is certified (exact to degree 3·2^k − 1 with residual ≤ 1e-200, and not
    exact one degree higher), and every stored node and weight is the double nearest to it
    (within half an ulp). Regression: binary128 left the 7 outer level-6 weights up to 18989
    ulps off and the nodes t_0, t_2 1–2 ulps off; the old version of this test stopped at
    level 4."""
    with mpmath.workdps(MP_DPS):
        nodes = [mpmath.mpf(0)]
        for level in LEVELS:
            t, w = M.gauss_patterson_rule(level)
            if level > 0:  # the new nodes are t[0::2]; t[1::2] is level − 1
                nodes = sorted([*nodes, *_mp_patterson_extension(nodes, t[0::2])], reverse=True)
            weights = _mp_interpolatory_weights(nodes)
            deg = _gkp_degree(level)
            res = _mp_legendre_residuals(nodes, weights, deg)
            assert max(res[: deg + 1]) <= mpmath.mpf(10) ** -200, level
            assert res[deg + 1] > mpmath.mpf(10) ** -30, level
            assert len(t) == len(nodes) == 2 ** (level + 1) - 1
            for i, (got, want) in enumerate([*zip(t, nodes, strict=True),
                                             *zip(w, weights, strict=True)]):  # fmt: skip
                half_ulp = 0.5 * float(np.spacing(abs(float(want)))) if want != 0 else 0.0
                assert abs(mpmath.mpf(float(got)) - want) <= half_ulp * (1 + 1e-9), (level, i, got)


def test_patterson_level_0_and_1_by_hand_and_against_golub_welsch():
    t0, w0 = M.gauss_patterson_rule(0)
    assert_array_equal(t0, [0.0])
    assert_array_equal(w0, [2.0])
    t1, w1 = M.gauss_patterson_rule(1)
    assert_allclose(t1, [math.sqrt(0.6), 0.0, -math.sqrt(0.6)], rtol=0, atol=EPS)
    assert_allclose(w1, [5 / 9, 8 / 9, 5 / 9], rtol=0, atol=EPS)
    tg, wg = M.gauss_legendre_rule(3)  # ascending
    assert_allclose(t1, tg[::-1], rtol=0, atol=2 * EPS)
    assert_allclose(w1, wg[::-1], rtol=0, atol=2 * EPS)


@pytest.mark.parametrize("level", LEVELS)
def test_patterson_rule_nested_symmetric_positive(level):
    t, w = M.gauss_patterson_rule(level)
    assert t.size == w.size == 2 ** (level + 1) - 1
    assert (np.diff(t) < 0).all() and -1.0 < t[-1] and t[0] < 1.0
    assert_array_equal(t, -t[::-1])
    assert_array_equal(w, w[::-1])
    assert (w > 0).all()
    assert abs(math.fsum(w.tolist()) - 2.0) <= 4 * EPS
    if level > 0:  # every node of level − 1 is a node of this level, bit for bit
        t_prev, _ = M.gauss_patterson_rule(level - 1)
        assert_array_equal(t[1::2], t_prev)


@pytest.mark.parametrize("level", LEVELS)
def test_patterson_exact_to_its_degree_and_not_one_more_in_50_digits(level):
    """Σ wᵢ P_j(xᵢ) = ∫P_j (2 for j = 0, else 0) for j ≤ 3·2^level − 1, evaluated at 50 digits with
    the stored doubles; the residual must lie within the effect of rounding the true rule to
    doubles, |δw|·|P_j| + |w|·|P_j'|·|δx| with |δ| ≤ half an ulp. P_{deg+1} must fail by more
    than 100× that bound (levels 0–5).

    NOTE: at level 6 the residual on P_192 is 2.6e-18 with the stored doubles (1.6e-21 for the
    exact rule), below the rounding bound 1.8e-15 of the stored doubles, so the sharpness of
    degree 191 is not observable in double precision; the exactness up to 191 is checked here,
    and the sharpness is checked at 300 digits in the recomputation test."""
    t, w = M.gauss_patterson_rule(level)
    deg = _gkp_degree(level)
    with mpmath.workdps(50):
        xs = [mpmath.mpf(float(x)) for x in t]
        ws = [mpmath.mpf(float(v)) for v in w]
        hx = [0.5 * float(np.spacing(abs(float(x)))) if x != 0 else 0.0 for x in t]
        hw = [0.5 * float(np.spacing(float(v))) for v in w]
        p_prev, p = [mpmath.mpf(1)] * len(xs), list(xs)  # P_0, P_1
        for j in range(1, deg + 2):
            if j >= 2:
                p_prev, p = p, [((2 * j - 1) * x * pj - (j - 1) * pp) / j for x, pj, pp in
                                zip(xs, p, p_prev, strict=True)]  # fmt: skip
            dp = [j * (x * pj - pp) / (x * x - 1) for x, pj, pp in zip(xs, p, p_prev, strict=True)]
            residual = abs(mpmath.fsum(wi * pj for wi, pj in zip(ws, p, strict=True)))
            bound = mpmath.fsum(
                a * abs(pj) + abs(wi) * abs(d) * b
                for a, pj, wi, d, b in zip(hw, p, ws, dp, hx, strict=True)
            )
            if j <= deg:
                assert residual <= 1.01 * bound + mpmath.mpf(10) ** -40, (level, j)
            elif level < M._GKP_MAX_LEVEL:
                assert residual > 100 * bound, (level, j)


@settings(max_examples=1000, deadline=None)
@given(
    level=st.integers(0, M._GKP_MAX_LEVEL),
    raw=st.lists(st.floats(-1.0, 1.0, allow_subnormal=False), min_size=192, max_size=192),
)
def test_patterson_exact_for_random_polynomials_in_double(level, raw):
    """p = Σ_{j≤deg} a_j P_j has ∫p = 2a_0. NOTE: the 50-digit test above bounds the rounding
    effect of the stored table by max_j B_j ≈ 1.9e-15 at level 6 (|P_j'| ~ j² near ±1), plus the
    legval rounding ≈ deg·ε·Σ|a|, hence 2e-13·Σ|a|."""
    deg = _gkp_degree(level)
    a = np.array(raw[: deg + 1])
    t, w = M.gauss_patterson_rule(level)
    got = math.fsum((w * np.polynomial.legendre.legval(t, a)).tolist())
    assert abs(got - 2.0 * a[0]) <= 2e-13 * max(1.0, float(np.abs(a).sum()))


# --------------------------------------------------------------------------------------
# The methods: estimates, counts, stopping test
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", NESTED)
@pytest.mark.parametrize("pid", CALCULUS)
def test_err_est_is_the_documented_difference_and_stop_is_the_first_pass(method, pid):
    p = problems.get(pid)
    tol = 1e-8
    res = numopt.run(method, p, tol=tol)
    est = [_fun(s) for s in res.trace]
    assert res.trace[0].info["err_est"] is None
    for k in range(1, len(est)):
        assert res.trace[k].info["err_est"] == abs(est[k] - est[k - 1])
    passes = [
        k >= 2 and abs(est[k] - est[k - 1]) <= tol * max(1.0, abs(est[k])) for k in range(len(est))
    ]
    if res.converged:
        assert passes[-1] and not any(passes[:-1])
    else:
        assert not any(passes) and res.n_iter == get_method(method).defaults()["max_levels"]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("pid", ["exp_0_1", "runge", "sqrt_0_1", "abs_kink"])
def test_cc_costs_n_K_plus_1_evaluations(n, pid):
    res = numopt.run("clenshaw_curtis", problems.get(pid), n=n, max_levels=8, tol=1e-9)
    K = res.n_iter
    assert res.n_fev == n * 2**K + 1
    assert [s.info["new_nodes"] for s in res.trace] == [n + 1] + [n * 2 ** (k - 1) for k in
                                                                    range(1, K + 1)]  # fmt: skip


@pytest.mark.parametrize("pid", ["exp_0_1", "runge", "sqrt_0_1", "gaussian"])
def test_patterson_costs_N_K_evaluations(pid):
    res = numopt.run("gauss_patterson", problems.get(pid), tol=1e-12)
    K = res.n_iter
    assert res.n_fev == 2 ** (K + 1) - 1
    assert [s.info["new_nodes"] for s in res.trace] == [2**k for k in range(K + 1)]


# quad warns that 1e-14 is at its roundoff level; its reference is still accurate to ~1e-15
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "arctan_deriv", "gaussian", "runge"])
def test_converged_results_match_quad(pid):
    p = problems.get(pid)
    a, b = p.domain
    ref, _ = integrate.quad(p.f, a, b, epsabs=1e-14, epsrel=1e-14, limit=200)
    cc = numopt.run("clenshaw_curtis", p, tol=1e-12)
    assert cc.converged
    assert abs(cc.x - ref) <= 1e-12 * max(1.0, abs(ref))
    gkp = numopt.run("gauss_patterson", p, tol=1e-12)
    if pid != "runge":  # Runge needs more than 127 Patterson points for 1e-12 (see below)
        assert gkp.converged
        assert abs(gkp.x - ref) <= 1e-12 * max(1.0, abs(ref))


def test_bracket_overrides_domain_and_a_callable_has_no_error():
    res = numopt.run("clenshaw_curtis", math.exp, bracket=(0.0, 2.0), tol=1e-12)
    assert res.converged and abs(res.x - math.expm1(2.0)) <= 1e-13
    assert res.extra["error"] is None and res.trace[-1].info["error"] is None
    p = problems.get("exp_0_1")
    gkp = numopt.run("gauss_patterson", p, bracket=(0.0, 2.0), tol=1e-12)
    assert gkp.converged and abs(gkp.x - math.expm1(2.0)) <= 1e-13


def test_first_steps_are_trapezoid_simpson_and_midpoint_by_hand():
    """CC with n = 1: I_1 = trapezoid, I_2 = Simpson. Patterson level 0 = midpoint rule."""
    p = problems.get("exp_0_1")
    cc = numopt.run("clenshaw_curtis", p, n=1, max_levels=2, tol=1e-300)
    e = math.e
    assert abs(_fun(cc.trace[0]) - 0.5 * (1.0 + e)) <= 2 * EPS
    assert abs(_fun(cc.trace[1]) - (1.0 + 4.0 * math.exp(0.5) + e) / 6.0) <= 4 * EPS
    gkp = numopt.run("gauss_patterson", p, tol=1e-300, max_levels=2)
    assert gkp.trace[0].fun == math.exp(0.5)


def test_x20_shows_the_degree_of_each_rule():
    """x²⁰ on [−1, 1]: CC (n = 2) is exact from n_k = 32 (degree ≥ 20) but not at 16; Patterson
    from 15 points (degree 23) but not at 7 (degree 11).

    Hand value at n_k = 16: x²⁰ = 2^−19·Σ' C(20, i)·T_{20−2i}; only T_20 (p = 4) and T_18 (p = 2)
    exceed the degree 16, so by (5.4) I − I_16 = 2^−19·(eq_5_4(16, 4) + 20·eq_5_4(16, 2)) ≈ 1.7e-7.
    The terms are O(0.1) and the difference is 1.7e-7, so ≈ 20ε/1.7e-7 ≈ 3e-8 relative: rel 1e-7."""
    p = _problem(lambda x: x**20, -1.0, 1.0, 2.0 / 21.0)
    cc = numopt.run("clenshaw_curtis", p, n=2, max_levels=4, tol=1e-300)
    errs = [s.info["error"] for s in cc.trace]
    hand = 2.0**-19 * (eq_5_4(16, 4) + 20.0 * eq_5_4(16, 2))
    assert errs[3] == pytest.approx(hand, rel=1e-7)
    assert errs[4] <= 2 * EPS
    gkp = numopt.run("gauss_patterson", p, max_levels=3, tol=1e-300)
    errs = [s.info["error"] for s in gkp.trace]
    assert errs[2] > 1e-4 and errs[3] <= 2 * EPS


# --------------------------------------------------------------------------------------
# The study's key property: the rule matters only at the margin, in the direction of ρ
# --------------------------------------------------------------------------------------


def test_matched_level_accuracy_reproduces_the_study():
    """README §4.3: at level 5 (65 CC points against 63 Patterson points), CC is more accurate on
    Runge 1/(1 + 25x²) (ρ ≈ 1.22, small: 2.9e-11 against 3.1e-10) and Patterson is more
    accurate on the entire cos 50x (3.1e-16 against 1.1e-9). The factor 2 of Gauss over CC
    appears only for f analytic in a large Bernstein ellipse (Trefethen 2008)."""
    tiny = 1e-300  # run every level
    runge = problems.get("runge")
    cc = numopt.run("clenshaw_curtis", runge, n=2, max_levels=5, tol=tiny).trace[5].info
    gkp = numopt.run("gauss_patterson", runge, max_levels=5, tol=tiny).trace[5].info
    assert cc["n_points"] == 65 and gkp["n_points"] == 63
    assert cc["error"] < gkp["error"]
    assert STUDY_RUNGE_CC65 / 1.5 <= cc["error"] <= STUDY_RUNGE_CC65 * 1.5
    assert STUDY_RUNGE_GKP63 / 1.5 <= gkp["error"] <= STUDY_RUNGE_GKP63 * 1.5
    cos50 = _cos50()
    cc = numopt.run("clenshaw_curtis", cos50, n=2, max_levels=5, tol=tiny).trace[5].info
    gkp = numopt.run("gauss_patterson", cos50, max_levels=5, tol=tiny).trace[5].info
    assert gkp["error"] < cc["error"]
    assert STUDY_COS50_CC65 / 1.5 <= cc["error"] <= STUDY_COS50_CC65 * 1.5
    assert gkp["error"] <= 4 * EPS


@pytest.mark.parametrize("pid", ["exp_0_1", "sin_0_pi", "arctan_deriv", "oscillatory", "poly3"])
def test_both_nested_schemes_stop_at_the_same_level_patterson_two_points_cheaper(pid):
    """README §4.3: the two nested schemes stop at the same doubling level on most integrands;
    at the same level Patterson uses 2^{k+1} − 1 points against 2^{k+1} + 1."""
    p = problems.get(pid)
    cc = numopt.run("clenshaw_curtis", p, n=2, tol=1e-10)
    gkp = numopt.run("gauss_patterson", p, tol=1e-10)
    assert cc.converged and gkp.converged
    assert cc.n_iter == gkp.n_iter
    assert cc.n_fev - gkp.n_fev == 2


def test_patterson_stops_at_127_points_where_cc_continues():
    """README §4.3: Patterson fails more often only because its sequence stops at 127 points."""
    runge = problems.get("runge")
    gkp = numopt.run("gauss_patterson", runge, tol=1e-10)
    cc = numopt.run("clenshaw_curtis", runge, tol=1e-10)
    assert not gkp.converged and gkp.n_fev == 127 and "max_levels=6" in gkp.message
    assert cc.converged and cc.n_fev == 129


# --------------------------------------------------------------------------------------
# Honesty of ``converged`` and the documented blind spots of the stopping test
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(
    family=st.sampled_from(["runge", "peak"]),
    log_c=st.floats(1.0, 3.0),
    xc=st.floats(0.0, 1.0),
    n=st.integers(2, 64),
    log_tol=st.floats(-12, -1),
)
def test_cc_converged_is_honest_on_parametric_families(family, log_c, xc, n, log_tol):
    """converged ⇒ error ≤ 10·tol·max(1, |I|) on 1/(1 + c·x²) (n ≥ 2) and on exp(−c(x − x_c)²)
    when step 2 resolves the peak: its centre spacing π/(8n) ≤ σ = 1/√(2c), i.e.
    n ≥ π√(2c)/8 (c ∈ [10, 1000], the families of the package's honesty tests). Coarser grids
    can miss the peak entirely (next tests): the resolution precondition of every sampling
    rule, which the study's test does not guard."""
    c = 10.0**log_c
    if family == "runge":
        p = _runge_family(c)
    else:
        p = _peak_family(c, xc)
        n = max(n, math.ceil(math.pi * math.sqrt(2.0 * c) / 8.0))
    tol = 10.0**log_tol
    res = numopt.run("clenshaw_curtis", p, n=n, tol=tol)
    if res.converged:
        slack = 16 * EPS * max(1.0, abs(_ex(p)))
        assert res.extra["error"] <= 10 * tol * max(1.0, abs(res.x)) + slack


@settings(max_examples=1000, deadline=None)
@given(pid=st.sampled_from(CALCULUS), log_tol=st.floats(-14, -1), n=st.integers(1, 16))
def test_converged_is_honest_on_the_library(pid, log_tol, n):
    """converged ⇒ error ≤ 10·tol·max(1, |I|) on the library, except Clenshaw–Curtis on the
    kink |x − 0.3| (abs_kink), the documented limit of the study's test (next test)."""
    p = problems.get(pid)
    tol = 10.0**log_tol
    slack = 16 * EPS * max(1.0, abs(_ex(p)))
    runs = [numopt.run("gauss_patterson", p, tol=tol)]
    if pid != "abs_kink":
        runs.append(numopt.run("clenshaw_curtis", p, n=n, tol=tol))
    for res in runs:
        if res.converged:
            assert res.extra["error"] <= 10 * tol * max(1.0, abs(res.x)) + slack


def test_blind_spot_cc_accepts_a_large_error_at_a_kink():
    """Characterization (section comment): at the kink of |x − 0.3| the CC sequence converges
    algebraically and oscillates, and d_k estimates the coarser rule. With n = 60 and
    tol = 2e-10 it stops at k = 5 (1921 points) with an error of 4.7e-8 = 234·tol. The study
    reported the milder case tol = 1e-6 (error 2.6e-6)."""
    p = problems.get("abs_kink")
    res = numopt.run("clenshaw_curtis", p, n=60, tol=2e-10)
    assert res.converged and res.n_iter == 5 and res.n_fev == 1921
    assert res.extra["error"] == pytest.approx(4.6853e-8, rel=1e-4)
    assert res.extra["error"] > 200 * 2e-10
    study = numopt.run("clenshaw_curtis", p, n=2, tol=1e-6)
    assert study.converged and 1e-6 < study.extra["error"] < 10 * 1e-6


def test_blind_spot_cc_misses_a_peak_that_step_2_does_not_resolve():
    """Characterization (section comment): exp(−1000(x − x_c)²) on [0, 1] (σ = 0.022) is ≈ 0 at
    every node of the first three grids when they are coarse, so I_0 ≈ I_1 ≈ I_2 ≈ 0 and the
    test passes at k = 2 although I = 0.056: n = 1 (2, 3, 5 points) with x_c = 1/3, and n = 2
    (3, 5, 9 points) with x_c = 0.40625. With n = 18 (π/(8n) ≤ σ) the peak is resolved."""
    p = _peak_family(1000.0, 1.0 / 3.0)
    res = numopt.run("clenshaw_curtis", p, n=1, tol=1e-8)
    assert res.converged and res.n_iter == 2 and res.n_fev == 5
    assert res.extra["error"] == pytest.approx(0.05605, rel=1e-3)
    q = _peak_family(1000.0, 0.40625)
    res = numopt.run("clenshaw_curtis", q, n=2, tol=1e-3)
    assert res.converged and res.n_iter == 2 and res.n_fev == 9
    assert res.extra["error"] == pytest.approx(0.05601, rel=1e-3)
    for prob in (p, q):
        ok = numopt.run("clenshaw_curtis", prob, n=18, tol=1e-8)
        assert ok.converged and ok.extra["error"] <= 1e-8 * max(1.0, abs(_ex(prob)))


def test_blind_spot_patterson_accidental_agreement_before_resolution():
    """Characterization (method docstring): on exp(−1000(x − 1/3)²) the 7- and 15-point rules
    give 0.01574 and 0.01320, which agree within tol = 10^−2.5 although I = 0.0560: converged
    with an error of 13.5·tol. The study's test has no confirmation of an asymptotic regime."""
    p = _peak_family(1000.0, 1.0 / 3.0)
    tol = 10.0**-2.5
    res = numopt.run("gauss_patterson", p, tol=tol)
    assert res.converged and res.n_iter == 3 and res.n_fev == 15
    assert res.extra["error"] == pytest.approx(0.042847, rel=1e-4)
    assert res.extra["error"] > 10 * tol


def test_blind_spot_function_vanishing_on_the_first_three_grids():
    """f = 1 − T_{4n}(x)² is 0 at every node of n, 2n, 4n: I_0 = I_1 = I_2 = 0 and the run
    reports convergence although ∫ = 1 + 1/(4m² − 1) with m = 4n (method docstring)."""
    n = 2
    m = 4 * n
    p = _problem(lambda x: 1.0 - math.cos(m * math.acos(max(-1.0, min(1.0, x)))) ** 2, -1.0, 1.0,
                 1.0 + 1.0 / (4 * m * m - 1))  # fmt: skip
    res = numopt.run("clenshaw_curtis", p, n=n, tol=1e-10)
    assert res.converged and res.n_iter == 2
    assert all(abs(_fun(s)) <= 1e-14 for s in res.trace)
    assert res.extra["error"] > 0.9


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


def test_max_levels_reached_reports_failure():
    res = numopt.run("clenshaw_curtis", problems.get("abs_kink"), max_levels=4, tol=1e-12)
    assert not res.converged and res.n_iter == 4 and len(res.trace) == 5
    assert "max_levels=4" in res.message and res.n_fev == 2 * 2**4 + 1
    assert_valid_result(res)


def test_nonfinite_integrand_reports_failure():
    # closed CC evaluates f at a = 0, where 1/x is infinite: stop at step 0
    def inv(x):
        return 1.0 / x if x != 0.0 else math.inf

    res = numopt.run("clenshaw_curtis", inv, bracket=(0.0, 1.0))
    assert not res.converged and res.n_iter == 0
    assert "x = 0" in res.message
    assert res.trace[0].info["cheb_coeffs"] is None
    assert_valid_result(res)
    # an f that raises a domain error is reported as non-finite (NaN), not raised
    res = numopt.run("gauss_patterson", lambda x: math.log(x - 0.5), bracket=(0.0, 1.0))
    assert not res.converged and res.n_iter == 0 and "x = 0.5" in res.message
    assert_valid_result(res)


def test_open_patterson_rule_handles_an_end_point_singularity():
    """1/√x on [0, 1] (∫ = 2): CC fails at x = 0; Patterson never evaluates the end points, and
    its error decreases with the level (slowly: the singularity is integrable but not smooth)."""
    p = _problem(lambda x: 1.0 / math.sqrt(x) if x > 0 else math.inf, 0.0, 1.0, 2.0)
    cc = numopt.run("clenshaw_curtis", p)
    assert not cc.converged and cc.n_iter == 0
    gkp = numopt.run("gauss_patterson", p, tol=1e-12)
    assert not gkp.converged and gkp.n_iter == 6
    errs = [s.info["error"] for s in gkp.trace]
    assert all(e1 < e0 for e0, e1 in pairwise(errs)) and errs[-1] < 0.05
