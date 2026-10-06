"""The LP / ILP problem library agrees with SciPy (linprog / milp) and has unique optima."""

import itertools
import json

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from numopt import problems
from numopt.core.types import LinearProgram

LP_IDS = [p.id for p in problems.list_problems("lp")]
REQUIRED = {
    "wyndor",
    "diet_2d",
    "degenerate_2d",
    "unbounded_2d",
    "infeasible_2d",
    "klee_minty_3",
    "transport_small",
    "ilp_knapsack_like_2d",
    "ilp_3var",
}


def _linprog(lp: LinearProgram, extra_eq=None):
    sign = 1.0 if lp.sense == "min" else -1.0
    a_eq, b_eq = lp.A_eq, lp.b_eq
    if extra_eq is not None:
        row, val = extra_eq
        a_eq = row[None, :] if a_eq is None else np.vstack([a_eq, row])
        b_eq = np.array([val]) if b_eq is None else np.append(b_eq, val)
    return sign, linprog(
        sign * lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, A_eq=a_eq, b_eq=b_eq, method="highs"
    )


def test_required_ids_present():
    assert REQUIRED <= set(LP_IDS)
    for pid in LP_IDS:
        assert isinstance(problems.get(pid), LinearProgram)
        assert problems.kind_of(pid) == "lp"


@pytest.mark.parametrize("pid", LP_IDS)
def test_problem_metadata(pid):
    lp = problems.get(pid)
    n = lp.c.size
    if lp.A_ub is not None:
        assert lp.A_ub.shape == (lp.b_ub.size, n)
    if lp.A_eq is not None:
        assert lp.A_eq.shape == (lp.b_eq.size, n)
    if n == 2:
        assert len(lp.domain) == 2 and all(lo < hi for lo, hi in lp.domain)
        if lp.optimum is not None:
            assert all(lo <= v <= hi for v, (lo, hi) in zip(lp.optimum, lp.domain, strict=True))
    assert lp.description
    json.dumps(lp.to_dict(), allow_nan=False)


@pytest.mark.parametrize("pid", [p for p in LP_IDS if not problems.get(p).integer])
def test_lp_optimum_matches_linprog_and_is_unique(pid):
    lp = problems.get(pid)
    sign, res = _linprog(lp)
    if lp.optimum is None:
        expected = {"unbounded_2d": 3, "infeasible_2d": 2}[pid]
        assert res.status == expected
        return
    assert res.status == 0
    np.testing.assert_allclose(sign * res.fun, lp.optimal_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(lp.c @ lp.optimum, lp.optimal_value, rtol=1e-12)
    # Uniqueness: on the optimal face {cᵀx = opt}, every coordinate has min = max = optimum.
    for j in range(lp.c.size):
        for direction in (1.0, -1.0):
            obj = np.zeros(lp.c.size)
            obj[j] = direction
            r = linprog(
                obj,
                A_ub=lp.A_ub,
                b_ub=lp.b_ub,
                A_eq=np.vstack([lp.A_eq, lp.c]) if lp.A_eq is not None else lp.c[None, :],
                b_eq=np.append(lp.b_eq, lp.optimal_value)
                if lp.b_eq is not None
                else [lp.optimal_value],
                method="highs",
            )
            assert r.status == 0
            assert abs(r.x[j] - lp.optimum[j]) <= 1e-7 * (1 + abs(lp.optimum[j]))


@pytest.mark.parametrize("pid", [p for p in LP_IDS if problems.get(p).integer])
def test_ilp_optimum_matches_milp_and_brute_force(pid):
    lp = problems.get(pid)
    sign = 1.0 if lp.sense == "min" else -1.0
    res = milp(
        sign * lp.c,
        constraints=LinearConstraint(lp.A_ub, -np.inf, lp.b_ub),
        integrality=np.ones(lp.c.size),
        bounds=Bounds(0, np.inf),
    )
    assert res.status == 0
    np.testing.assert_allclose(sign * res.fun, lp.optimal_value, rtol=1e-12)
    # Brute force over the box 0..10 (both problems bound every variable by < 10).
    best, argbest = -np.inf, []
    for pt in itertools.product(range(11), repeat=lp.c.size):
        x = np.array(pt, dtype=float)
        if np.all(lp.A_ub @ x <= lp.b_ub + 1e-12):
            v = -sign * (lp.c @ x)
            if v > best + 1e-12:
                best, argbest = v, [x]
            elif abs(v - best) <= 1e-12:
                argbest.append(x)
    assert len(argbest) == 1, "integer optimum must be unique"
    np.testing.assert_array_equal(argbest[0], lp.optimum)


def test_relaxation_values_in_descriptions():
    for pid, x_lp, v_lp in [
        ("ilp_knapsack_like_2d", [3.75, 2.25], 41.25),
        ("ilp_3var", [1.2, 2.2, 0.8], 13.8),
    ]:
        lp = problems.get(pid)
        sign, res = _linprog(lp)
        np.testing.assert_allclose(res.x, x_lp, atol=1e-10)
        np.testing.assert_allclose(sign * res.fun, v_lp, rtol=1e-12)
