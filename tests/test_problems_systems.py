"""Tests of the nonlinear-systems problem library (kind "systems")."""

from __future__ import annotations

import json
import math
from typing import cast

import numpy as np
import pytest
from scipy import optimize

from numopt import problems
from numopt.core.types import Problem

REQUIRED = (
    "circle_line",
    "rosenbrock_system",
    "freudenstein_roth",
    "trig_system",
    "intersecting_circles",
)
ALL = [p.id for p in problems.list_problems("systems")]


def _grid(p: Problem, n: int = 9) -> list[np.ndarray]:
    (x0, x1), (y0, y1) = p.domain
    return [np.array([x, y]) for x in np.linspace(x0, x1, n) for y in np.linspace(y0, y1, n)]


def test_required_ids_are_registered_as_systems():
    for pid in REQUIRED:
        assert problems.kind_of(pid) == "systems"
        p = problems.get(pid)
        assert isinstance(p, Problem) and p.dim == 2 and p.jac is not None
    assert set(REQUIRED) <= set(ALL)


@pytest.mark.parametrize("pid", ALL)
def test_shapes_metadata_and_json(pid):
    p = problems.get(pid)
    (x0, x1), (y0, y1) = p.domain
    assert x0 < x1 and y0 < y1
    assert x0 <= p.x0[0] <= x1 and y0 <= p.x0[1] <= y1
    x = np.array(p.x0, dtype=float)
    assert np.asarray(p.f(x)).shape == (2,)
    assert np.asarray(p.jac(x)).shape == (2, 2)
    json.dumps(p.to_dict(), allow_nan=False)


@pytest.mark.parametrize("pid", ALL)
def test_jacobian_matches_central_difference(pid):
    p = problems.get(pid)
    for x in _grid(p):
        J = np.asarray(p.jac(x))
        fd = np.empty((2, 2))
        for j in range(2):
            h = 1e-6 * max(1.0, abs(x[j]))
            e = np.zeros(2)
            e[j] = h
            fd[:, j] = (np.asarray(p.f(x + e)) - np.asarray(p.f(x - e))) / (2 * h)
        # NOTE: truncation O(h²) times third derivatives (≤ ~1e3 here) plus rounding ε|F|/h.
        np.testing.assert_allclose(J, fd, rtol=1e-7, atol=1e-6 * max(1.0, np.abs(J).max()))


@pytest.mark.parametrize("pid", ALL)
def test_listed_roots_are_roots(pid):
    p = problems.get(pid)
    for r in p.roots:
        r = np.array(r)
        scale = 1.0 + np.abs(np.asarray(p.jac(r))).max() * np.abs(r).max()
        assert np.linalg.norm(p.f(r)) <= 16 * np.finfo(float).eps * scale
        # Oracle: fsolve started near the root returns to it.
        sol = np.asarray(cast(np.ndarray, optimize.fsolve(p.f, r + 1e-3, fprime=p.jac, xtol=1e-12)))
        np.testing.assert_allclose(sol, r, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("pid", ALL)
def test_roots_list_is_complete_inside_the_domain(pid):
    """Every root that SciPy's hybrid method finds from a grid of starts is listed."""
    p = problems.get(pid)
    (x0, x1), (y0, y1) = p.domain
    listed = [np.array(r) for r in p.roots]
    for start in _grid(p, 15):
        sol = optimize.root(p.f, start, jac=p.jac, method="hybr", options={"xtol": 1e-13})
        if not sol.success or np.linalg.norm(p.f(sol.x)) > 1e-10:
            continue
        if not (x0 <= sol.x[0] <= x1 and y0 <= sol.x[1] <= y1):
            continue
        assert any(np.allclose(sol.x, r, atol=1e-7) for r in listed), (start, sol.x)


def test_jacobian_singular_sets_from_the_descriptions():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(-3, 3, size=2)
        v = np.array([x, y])
        det = lambda pid, v=v: float(np.linalg.det(problems.get(pid).jac(v)))  # noqa: E731
        assert math.isclose(det("circle_line"), 2 * (x + y), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(det("trig_system"), math.sin(x + y), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            det("intersecting_circles"), 4 * (1 - x - 2 * y), rel_tol=1e-12, abs_tol=1e-12
        )
        assert math.isclose(
            det("freudenstein_roth"), 6 * y * y - 8 * y - 12, rel_tol=1e-12, abs_tol=1e-9
        )
        assert math.isclose(
            det("rosenbrock_system"), 80000 * (x * x - y + 0.005), rel_tol=1e-9, abs_tol=1e-6
        )


def test_freudenstein_roth_has_the_documented_local_minimum():
    p = problems.get("freudenstein_roth")
    res = optimize.least_squares(p.f, [12.0, -1.0], jac=p.jac, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    np.testing.assert_allclose(res.x, [11.4128, -0.896805], rtol=1e-4)
    # Moré, Garbow & Hillstrom (1981): Σ fᵢ² = 48.9842... at the local minimizer.
    assert abs(float(np.sum(res.fun**2)) - 48.9842) < 1e-3
    assert np.linalg.norm(p.f(np.array([5.0, 4.0]))) == 0.0


def test_rosenbrock_system_is_the_gradient():
    p = problems.get("rosenbrock_system")

    def f(v):
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    for v in _grid(p, 5):
        g = np.asarray(optimize.approx_fprime(v, f, 1e-7))
        np.testing.assert_allclose(p.f(v), g, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("pid", [p.id for p in problems.list_problems("systems")])
def test_systems_never_raise_far_from_the_domain(pid):
    """F and J return arrays (±inf / nan entries allowed) for huge or non-finite x, also
    for plain Python lists (the float ** operator would raise OverflowError)."""
    p = problems.get(pid)
    with np.errstate(over="ignore", invalid="ignore"):
        for x in ([1e200, -1e200], [math.inf, 0.0], [math.nan, 1.0], [-1e300, 1e300]):
            for arg in (x, np.array(x)):
                assert np.shape(p.f(arg)) == (2,)
                assert np.shape(p.jac(arg)) == (2, 2)
