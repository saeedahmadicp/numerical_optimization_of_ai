"""Tests of the scalar root-finding problem library (kind "roots")."""

from __future__ import annotations

import json
import math
import sys
from itertools import pairwise
from typing import cast

import numpy as np
import pytest
from scipy import optimize

import numopt
from numopt import problems
from numopt.core.types import Problem
from numopt.problems.roots import ATAN_NEWTON_CRITICAL

EPS = sys.float_info.epsilon
REQUIRED = (
    "sqrt2",
    "cubic",
    "cos_minus_x",
    "x10_minus_1",
    "kepler",
    "double_root",
    "atan_newton",
    "newton_cycle",
    "steep_exp",
    "wilkinson5",
)
ALL = [p.id for p in problems.list_problems("roots")]


def test_required_ids_are_registered_as_roots():
    for pid in REQUIRED:
        assert problems.kind_of(pid) == "roots"
        p = problems.get(pid)
        assert isinstance(p, Problem) and p.dim == 1
        assert p.grad is not None and p.hess is not None
        assert p.bracket is not None and p.x0 is not None and p.roots
    assert set(REQUIRED) <= set(ALL)


@pytest.mark.parametrize("pid", ALL)
def test_metadata_is_consistent_and_json(pid):
    p = problems.get(pid)
    lo, hi = p.domain
    a, b = p.bracket
    assert lo <= a < b <= hi
    assert lo <= p.x0 <= hi
    assert list(p.roots) == sorted(p.roots)
    json.dumps(p.to_dict(), allow_nan=False)


def _points(p: Problem, n: int = 41) -> np.ndarray:
    lo, hi = p.domain
    return np.linspace(lo, hi, n)[1:-1]


@pytest.mark.parametrize("pid", ALL)
def test_first_derivative_matches_central_difference(pid):
    p = problems.get(pid)
    for x in _points(p):
        h = 1e-6 * max(1.0, abs(x))
        fd = (p.f(x + h) - p.f(x - h)) / (2 * h)
        # NOTE: central-difference truncation O(h²f''') plus rounding O(ε|f|/h) ≈ 1e-9·scale.
        scale = max(1.0, abs(p.grad(x)), abs(p.f(x)))
        assert abs(p.grad(x) - fd) <= 1e-7 * scale, (x, p.grad(x), fd)


@pytest.mark.parametrize("pid", ALL)
def test_second_derivative_matches_central_difference(pid):
    p = problems.get(pid)
    for x in _points(p):
        h = 1e-6 * max(1.0, abs(x))
        fd = (p.grad(x + h) - p.grad(x - h)) / (2 * h)
        scale = max(1.0, abs(p.hess(x)), abs(p.grad(x)))
        assert abs(p.hess(x) - fd) <= 1e-7 * scale, (x, p.hess(x), fd)


@pytest.mark.parametrize("pid", ALL)
def test_listed_roots_are_roots_to_full_precision(pid):
    p = problems.get(pid)
    for r in p.roots:
        if p.grad(r) == 0.0:  # multiple root: no sign change, check exact values
            assert p.f(r) == 0.0
            continue
        # A tight bracket around r and brentq at its tightest tolerance (oracle).
        d = 1e-6 * max(1.0, abs(r))
        assert p.f(r - d) * p.f(r + d) < 0
        ref = float(
            cast(float, optimize.brentq(p.f, r - d, r + d, xtol=1e-300, rtol=np.float64(4 * EPS)))
        )
        assert abs(r - ref) <= 4 * EPS * abs(r) + 1e-300, (r, ref)


@pytest.mark.parametrize("pid", ALL)
def test_roots_list_is_complete_inside_the_domain(pid):
    p = problems.get(pid)
    lo, hi = p.domain
    xs = np.linspace(lo, hi, 20001)
    fs = np.array([p.f(x) for x in xs])
    changes = int(np.sum(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)) + int(np.sum(fs == 0.0))
    simple_inside = [r for r in p.roots if lo < r < hi and p.grad(r) != 0.0]
    multiple_inside = [r for r in p.roots if lo < r < hi and p.grad(r) == 0.0]
    # A double root shows no sign change; it is counted only if a grid point hits it.
    assert len(simple_inside) <= changes <= len(simple_inside) + len(multiple_inside)


@pytest.mark.parametrize("pid", ALL)
def test_default_bracket_has_a_sign_change_and_contains_a_root(pid):
    p = problems.get(pid)
    a, b = p.bracket
    assert p.f(a) * p.f(b) < 0
    assert any(a < r < b for r in p.roots)


def test_polynomial_roots_match_numpy_roots():
    cases = {
        "sqrt2": [1, 0, -2],
        "cubic": [1, 0, -2, -5],
        "double_root": [1, 0, -3, 2],
        "newton_cycle": [1, 0, -2, 2],
        "wilkinson5": np.poly([1, 2, 3, 4, 5]),
        "x10_minus_1": [1, *([0] * 9), -1],
    }
    for pid, coeffs in cases.items():
        z = np.roots(coeffs)
        real = np.sort(z[np.abs(z.imag) < 1e-6].real)
        listed = np.array(problems.get(pid).roots)
        # Distinct real roots (np.roots splits the double root into a close pair).
        distinct = [real[0]] + [v for u, v in pairwise(real) if v - u > 1e-6]
        np.testing.assert_allclose(listed, distinct, rtol=1e-7, atol=1e-7)


def test_newton_cycle_map_sends_zero_to_one_and_back():
    p = problems.get("newton_cycle")

    def newton_map(x: float) -> float:
        return x - p.f(x) / p.grad(x)

    assert newton_map(0.0) == 1.0
    assert newton_map(1.0) == 0.0
    assert p.hess(0.0) == 0.0  # superattracting 2-cycle


def test_atan_newton_critical_start():
    x = ATAN_NEWTON_CRITICAL
    # (1 + x²) arctan x = 2x, so the Newton map sends x_c to −x_c.
    assert abs((1 + x * x) * math.atan(x) - 2 * x) <= 8 * EPS
    p = problems.get("atan_newton")
    n = x - p.f(x) / p.grad(x)
    assert abs(n + x) <= 1e-14
    assert numopt.run("newton", p, x0=x - 1e-3).converged
    assert not numopt.run("newton", p, x0=x + 1e-3).converged
    assert abs(p.x0) > x  # the default start demonstrates divergence


def test_double_root_has_multiplicity_two():
    p = problems.get("double_root")
    assert p.f(1.0) == 0.0 and p.grad(1.0) == 0.0 and p.hess(1.0) != 0.0
    assert p.f(-2.0) == 0.0 and p.grad(-2.0) != 0.0


def test_kepler_and_x10_descriptions_hold():
    k = problems.get("kepler")
    assert k.x0 == k.extra["M"]
    assert abs(k.grad(k.x0) - (1 - 0.9 * math.cos(0.3))) < 1e-15
    first = k.x0 - k.f(k.x0) / k.grad(k.x0)
    assert 2.1 < first < 2.3  # Newton's first step overshoots to E ≈ 2.2
    p = problems.get("x10_minus_1")
    x1 = 0.5 - p.f(0.5) / p.grad(0.5)
    assert abs(x1 - 51.65) < 0.01


@pytest.mark.parametrize("pid", ALL)
def test_functions_never_raise_far_from_the_domain(pid):
    """Open methods evaluate f far away: overflow gives ±inf, a non-finite x gives inf or
    nan, never an exception (math.exp and x**10 used to raise OverflowError)."""
    p = problems.get(pid)
    for x in (1e3, -1e3, 1e40, -1e40, 1e300, -1e300, math.inf, -math.inf, math.nan):
        for fn in (p.f, p.grad, p.hess):
            assert isinstance(fn(x), float)


def test_overflow_gives_infinity_with_the_right_sign():
    steep = problems.get("steep_exp")
    assert steep.f(1e3) == steep.grad(1e3) == math.inf and steep.f(-1e3) == -10.0
    p = problems.get("x10_minus_1")
    assert p.f(1e40) == p.f(-1e40) == math.inf
    assert p.grad(1e40) == math.inf and p.grad(-1e40) == -math.inf
    assert p.hess(-1e40) == math.inf
    assert problems.get("double_root").f(1e300) == math.inf
    assert problems.get("double_root").f(-1e300) == -math.inf
