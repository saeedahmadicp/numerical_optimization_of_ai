"""Branch and bound and Gomory cuts: oracle comparison with scipy.optimize.milp and brute
force, validity of every Gomory cut, B&B tree structure, failure paths."""

from __future__ import annotations

import itertools
import re
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.optimize import Bounds, LinearConstraint, milp

import numopt
from numopt import problems
from numopt.core.registry import get_method
from numopt.core.types import LinearProgram
from numopt.lp import integer as integer_module

ILPS = ("ilp_knapsack_like_2d", "ilp_3var")
STRATEGIES = ("best_bound", "depth_first")


def milp_oracle(lp: LinearProgram):
    sign = 1.0 if lp.sense == "min" else -1.0
    cons = []
    # SciPy's stubs type lb/ub as float, but arrays are accepted.
    b_ub: Any = lp.b_ub
    b_eq: Any = lp.b_eq
    if lp.A_ub is not None:
        cons.append(LinearConstraint(lp.A_ub, -np.inf, b_ub))
    if lp.A_eq is not None:
        cons.append(LinearConstraint(lp.A_eq, b_eq, b_eq))
    integrality = np.ones(lp.c.size) if not lp.integer else np.array(lp.integer, dtype=float)
    r = milp(sign * lp.c, constraints=cons, integrality=integrality, bounds=Bounds(0, np.inf))
    return r.status, (None if r.status != 0 else sign * r.fun)


def milp_x(lp: LinearProgram) -> np.ndarray:
    """milp's optimal x of a pure ILP with only ≤ rows."""
    sign = 1.0 if lp.sense == "min" else -1.0
    b_ub: Any = lp.b_ub
    r = milp(
        sign * lp.c,
        constraints=[LinearConstraint(lp.A_ub, -np.inf, b_ub)],
        integrality=np.ones(lp.c.size),
        bounds=Bounds(0, np.inf),
    )
    return np.asarray(r.x)


def integer_points(lp: LinearProgram, hi: int):
    """All integer points of the box [0, hi]ⁿ that satisfy the constraints."""
    for pt in itertools.product(range(hi + 1), repeat=lp.c.size):
        x = np.array(pt, dtype=float)
        ok = lp.A_ub is None or lp.b_ub is None or bool(np.all(lp.A_ub @ x <= lp.b_ub + 1e-9))
        if lp.A_eq is not None and lp.b_eq is not None:
            ok = ok and bool(np.allclose(lp.A_eq @ x, lp.b_eq))
        if ok:
            yield x


@pytest.mark.parametrize("pid", ILPS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_branch_and_bound_matches_milp(pid, strategy):
    lp = problems.get(pid)
    res = numopt.run("branch_and_bound", lp, strategy=strategy)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    _, value = milp_oracle(lp)
    assert res.fun == pytest.approx(value, abs=1e-9)
    np.testing.assert_array_equal(res.x, lp.optimum)
    final = res.trace[-1].info
    assert final["gap"] == pytest.approx(0.0, abs=1e-9)
    assert final["incumbent_value"] == pytest.approx(value)


def test_winston_tree():
    # Winston §9.3: 7 subproblems. Root LP (3.75, 2.25): both variables are 0.25 from an
    # integer, so the tie goes to x1 (smallest index), as in Winston's tree.
    res = numopt.run("branch_and_bound", problems.get("ilp_knapsack_like_2d"))
    tree = res.extra["tree"]
    assert len(tree) == 7
    root = tree[0]
    assert root["parent"] is None and root["lp_value"] == pytest.approx(41.25)
    np.testing.assert_allclose(root["lp_x"], [3.75, 2.25])
    for node in tree[1:]:
        parent = tree[node["parent"]]
        assert parent["status"] == "branched"
        assert node["depth"] == parent["depth"] + 1
        j, op, v = node["branch"]
        lo, hi = node["bounds"][j]
        assert (op == "<=" and hi == v) or (op == ">=" and lo == v)
        # Children's LP values never beat the parent's (max problem).
        if node["lp_value"] is not None:
            assert node["lp_value"] <= parent["lp_value"] + 1e-9
    statuses = {n["status"] for n in tree}
    assert statuses <= {"branched", "integer", "infeasible", "pruned_bound"}
    assert "open" not in statuses


@pytest.mark.parametrize("pid", ILPS)
def test_bound_and_incumbent_are_monotone(pid):
    res = numopt.run("branch_and_bound", problems.get(pid))
    inc = [s.info["incumbent_value"] for s in res.trace if s.info["incumbent_value"] is not None]
    bounds = [s.info["best_bound"] for s in res.trace if s.info["best_bound"] is not None]
    assert all(b >= a for a, b in pairwise(inc))  # max: incumbent improves
    assert all(b <= a + 1e-9 for a, b in pairwise(bounds))  # bound tightens
    assert all(bd >= iv - 1e-9 for bd, iv in zip(bounds[-len(inc) :], inc, strict=False))


@pytest.mark.parametrize("pid", ILPS)
def test_gomory_matches_milp_and_cuts_are_valid(pid):
    lp = problems.get(pid)
    res = numopt.run("gomory_cuts", lp)
    assert_valid_result(res, max_iter=50)
    assert res.converged, res.message
    _, value = milp_oracle(lp)
    assert res.fun == pytest.approx(value, abs=1e-9)
    np.testing.assert_array_equal(res.x, lp.optimum)
    points = list(integer_points(lp, 10))
    for prev, cur in zip(res.trace, res.trace[1:], strict=False):
        cut = cur.info["cut"]
        coef, rhs = np.asarray(cut["coef"]), cut["rhs"]
        # Valid: no integer feasible point is cut off ...
        assert all(coef @ x <= rhs + 1e-9 for x in points)
        # ... and the previous LP vertex is cut off by f0 (in z-space) > 0.
        assert coef @ np.asarray(prev.x) > rhs + 1e-9
        # The new vertex satisfies every cut that still has a row in the tableau.
        for q in cur.info["cuts"]:
            if q["active"]:
                assert np.asarray(q["coef"]) @ np.asarray(cur.x) <= q["rhs"] + 1e-9
    assert len(res.extra["cuts"]) == res.n_iter


def test_gomory_first_cut_on_knapsack():
    # Source row: largest fractional rhs. LP vertex (3.75, 2.25); first cut 3x1 + 2x2 ≤ 15.
    res = numopt.run("gomory_cuts", problems.get("ilp_knapsack_like_2d"))
    cut = res.trace[1].info["cut"]
    np.testing.assert_allclose(cut["coef"], [3.0, 2.0], atol=1e-12)
    assert cut["rhs"] == pytest.approx(15.0)
    assert cut["f0"] == pytest.approx(0.75)


# --------------------------------------------------------------------------------------
# Failure paths and input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_iter", (1, 2, 3))
def test_node_and_cut_limits(max_iter):
    """max_iter counts the nodes after the root: Step 0 (root) + max_iter Steps, as the help
    text and the message say (audit: the message claimed max_iter nodes for max_iter + 1)."""
    res = numopt.run("branch_and_bound", problems.get("ilp_knapsack_like_2d"), max_iter=max_iter)
    assert_valid_result(res, max_iter=max_iter)
    assert not res.converged and res.extra["status"] == "max_iter" and "max_iter" in res.message
    assert len(res.trace) == max_iter + 1 and res.n_iter == max_iter
    assert f"the root and {max_iter} more nodes" in res.message
    g = numopt.run("gomory_cuts", problems.get("ilp_3var"), max_iter=1)
    assert_valid_result(g, max_iter=1)
    assert not g.converged and g.extra["status"] == "max_iter"


def test_integer_infeasible_but_lp_feasible():
    # 2x1 + 2x2 = 1 has LP solutions but no integer solution.
    lp = LinearProgram(
        "p",
        "p",
        np.array([1.0, 1.0]),
        A_eq=np.array([[2.0, 2.0]]),
        b_eq=np.array([1.0]),
        integer=(True, True),
    )
    for strategy in STRATEGIES:
        res = numopt.run("branch_and_bound", lp, strategy=strategy)
        assert_valid_result(res)
        assert not res.converged and res.extra["status"] == "infeasible"
    g = numopt.run("gomory_cuts", lp)
    assert_valid_result(g)
    assert not g.converged and g.extra["status"] == "infeasible"


def test_relaxation_failures():
    for pid, status in (("unbounded_2d", "unbounded"), ("infeasible_2d", "infeasible")):
        for method in ("branch_and_bound", "gomory_cuts"):
            res = numopt.run(method, problems.get(pid))
            assert_valid_result(res)
            assert not res.converged
            assert res.extra["status"] == status


def test_mixed_integer_branch_and_bound():
    lp = LinearProgram(
        "mip",
        "mip",
        np.array([8.0, 5.0]),
        A_ub=np.array([[1.0, 1.0], [9.0, 5.0]]),
        b_ub=np.array([6.0, 45.0]),
        sense="max",
        integer=(True, False),
    )
    res = numopt.run("branch_and_bound", lp)
    assert res.converged
    _, value = milp_oracle(lp)
    assert res.fun == pytest.approx(value, abs=1e-9)
    with pytest.raises(ValueError, match="pure ILP"):
        numopt.run("gomory_cuts", lp)


# --------------------------------------------------------------------------------------
# Gomory in exact arithmetic (audit: float tableaux removed the optimum)
# --------------------------------------------------------------------------------------


def ilp(c, a, b, sense: Literal["min", "max"] = "max") -> LinearProgram:
    return LinearProgram(
        "g",
        "g",
        np.array(c, dtype=float),
        A_ub=np.array(a, dtype=float),
        b_ub=np.array(b, dtype=float),
        sense=sense,
    )


def assert_cuts_keep(res, x_star):
    """Every cut is satisfied by the integer optimum x* (validity)."""
    for q in res.extra["cuts"]:
        assert np.asarray(q["coef"]) @ x_star <= q["rhs"] + 1e-9 * (1 + abs(q["rhs"]))


@pytest.mark.parametrize(
    ("c", "a", "b", "x_star", "value"),
    [
        # Float drift: after ~30 cuts the cuts removed (1, 2) and "infeasible" was reported.
        ([6, 19], [[6, 27], [24, 25]], [61, 135], [1, 2], 44.0),
        # Snapped coefficient f = 0.99962 → 0 made the next cut remove (18, 30) at int_tol=1e-3.
        ([18, 14], [[3, 86], [93, 4]], [2660, 1807], [18, 30], 744.0),
    ],
)
def test_gomory_audit_counterexamples(c, a, b, x_star, value):
    lp = ilp(c, a, b)
    status, ref = milp_oracle(lp)
    assert status == 0 and ref == pytest.approx(value)
    res = numopt.run("gomory_cuts", lp, max_iter=200)
    assert_valid_result(res, max_iter=200)
    assert res.converged, res.message
    np.testing.assert_array_equal(res.x, x_star)
    assert res.fun == value
    assert_cuts_keep(res, np.array(x_star, dtype=float))


def test_gomory_cut_coefficients_are_exact_fractional_parts():
    """f = ā − ⌊ā⌋ with no snapping. On the second counterexample the optimal tableau row of
    x2 is x2 + (31/2662) s1 − (1/2662) s2 = 80653/2662 (exact, by hand), so the first cut has
    f(ā_{x2,s2}) = 2661/2662; snapping within int_tol = 1e-3 replaced it by 0."""
    res = numopt.run("gomory_cuts", ilp([18, 14], [[3, 86], [93, 4]], [2660, 1807]), max_iter=1)
    cut = res.trace[1].info["cut"]
    info0 = res.trace[0].info
    T = np.asarray(info0["tableau"])
    row = T[cut["source_row"]]
    for j in info0["nonbasis"]:
        assert cut["f"][j] == pytest.approx(row[j] - np.floor(row[j]), abs=1e-15)
    s1, s2 = info0["col_labels"].index("s1"), info0["col_labels"].index("s2")
    assert cut["source_var"] == "x2"
    assert cut["f"][s1] == 31 / 2662 and cut["f"][s2] == 2661 / 2662
    assert cut["f0"] == (80653 % 2662) / 2662


def test_gomory_deletes_basic_cut_rows_beyond_the_cap():
    """53 cuts > 50: rows of cuts with basic slack are deleted, x* is still found."""
    res = numopt.run("gomory_cuts", ilp([18, 14], [[3, 86], [93, 4]], [2660, 1807]), max_iter=200)
    assert res.converged
    flags = [q["active"] for q in res.extra["cuts"]]
    assert len(flags) > integer_module._MAX_CUT_ROWS and not all(flags)
    final = res.trace[-1].info
    assert sum(1 for lab in final["col_labels"] if lab.startswith("g")) == sum(flags)
    assert_cuts_keep(res, np.array([18.0, 30.0]))


@st.composite
def two_digit_ilps(draw):
    """Bounded feasible pure ILPs with coefficients up to 99 (the audit's failing regime)."""
    n = draw(st.integers(2, 3))
    m = draw(st.integers(1, 3))
    a = draw(hnp.arrays(np.float64, (m, n), elements=st.integers(-99, 99)))
    x_feas = draw(hnp.arrays(np.float64, (n,), elements=st.integers(0, 9)))
    slack = draw(hnp.arrays(np.float64, (m,), elements=st.integers(0, 99)))
    box = float(draw(st.integers(int(x_feas.sum()), 60)))
    a = np.vstack([a, np.ones((1, n))])  # Σx ≤ box keeps the ILP bounded
    b = np.append(a[:-1] @ x_feas + slack, box)  # x_feas is feasible
    c = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-30, 30)))
    sense = draw(st.sampled_from(["min", "max"]))
    return LinearProgram("t", "t", c, A_ub=a, b_ub=b, sense=sense, integer=(True,) * n)


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(two_digit_ilps())
def test_random_gomory_two_digit_data(lp):
    status, best = milp_oracle(lp)
    assert status == 0 and best is not None  # feasible and bounded by construction
    res = numopt.run("gomory_cuts", lp, max_iter=200)
    assert_valid_result(res, max_iter=200)
    # Never a false "infeasible": the ILP has the integer point x_feas.
    assert res.extra["status"] in ("optimal", "max_iter"), res.message
    # Every cut keeps the integer optimum (milp's x, integral to HiGHS's 1e-6, rounded).
    x_star = np.round(milp_x(lp))
    assert np.all(lp.A_ub @ x_star <= lp.b_ub) and lp.c @ x_star == pytest.approx(best)
    assert_cuts_keep(res, x_star)
    if res.converged:
        assert res.fun == pytest.approx(best, abs=1e-9 * (1 + abs(best)))
        x = np.asarray(res.x)
        assert np.all(x == np.round(x)) and np.all(lp.A_ub @ x <= lp.b_ub)


def test_gomory_has_no_integrality_tolerance():
    """Exact arithmetic decides integrality exactly; the old int_tol (which also snapped cut
    coefficients) is gone from the registry and from the signature."""
    spec = get_method("gomory_cuts")
    assert [p.name for p in spec.params] == ["max_iter"]
    with pytest.raises(TypeError, match="int_tol"):
        numopt.run("gomory_cuts", problems.get("ilp_3var"), int_tol=1e-3)


def test_gomory_rejects_fractional_data():
    lp = LinearProgram(
        "f",
        "f",
        np.array([1.0, 1.0]),
        A_ub=np.array([[0.5, 1.0]]),
        b_ub=np.array([2.0]),
        sense="max",
    )
    with pytest.raises(ValueError, match="integer constraint data"):
        numopt.run("gomory_cuts", lp)
    with pytest.raises(ValueError):
        numopt.run("branch_and_bound", lp, strategy="breadth_first")


# --------------------------------------------------------------------------------------
# Hypothesis: random bounded ILPs against milp and brute force
# --------------------------------------------------------------------------------------


@st.composite
def random_ilps(draw):
    n = draw(st.integers(1, 3))
    m = draw(st.integers(1, 3))
    # Nonzero coefficients (Hypothesis favours zeros, which make relaxations integral); a
    # negative entry or right-hand side still appears (≥ rows, infeasible instances).
    a = draw(
        hnp.arrays(np.float64, (m, n), elements=st.sampled_from([-2, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    )
    b = draw(hnp.arrays(np.float64, (m,), elements=st.integers(-3, 40)))
    # A box row Σx ≤ 6 keeps every ILP bounded and brute force cheap.
    a = np.vstack([a, np.ones((1, n))])
    b = np.append(b, 6.0)
    # The objective pushes into the constraints (gains for max, negative costs for min);
    # otherwise the origin is optimal and the relaxation is trivially integral.
    sense = draw(st.sampled_from(["min", "max"]))
    gains = draw(hnp.arrays(np.float64, (n,), elements=st.integers(-1, 9)))
    c = gains if sense == "max" else -gains
    eq = draw(st.booleans()) and n >= 2
    if eq:  # one equality row → also LP-feasible but integer-infeasible instances
        a_eq = draw(hnp.arrays(np.float64, (1, n), elements=st.integers(1, 5)))
        b_eq = np.array([float(draw(st.integers(1, 12)))])
        return LinearProgram(
            "r", "r", c, A_ub=a, b_ub=b, A_eq=a_eq, b_eq=b_eq, sense=sense, integer=(True,) * n
        )
    return LinearProgram("r", "r", c, A_ub=a, b_ub=b, sense=sense, integer=(True,) * n)


def brute_force(lp: LinearProgram):
    sign = 1.0 if lp.sense == "min" else -1.0
    vals = [sign * (lp.c @ x) for x in integer_points(lp, 6)]
    return None if not vals else sign * min(vals)


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_ilps(), st.sampled_from(STRATEGIES))
def test_random_branch_and_bound(lp, strategy):
    res = numopt.run("branch_and_bound", lp, strategy=strategy, max_iter=1000)
    assert_valid_result(res, max_iter=1000)
    # Exact oracle: the row Σx ≤ 6 puts every feasible point in the enumerated box.
    # (milp is not used here: HiGHS returns status 4 on some integer-infeasible instances.)
    best = brute_force(lp)
    if best is None:
        assert not res.converged and res.extra["status"] == "infeasible"
        return
    assert res.converged, res.message
    assert res.fun == pytest.approx(best, abs=1e-9)
    x = np.asarray(res.x)
    assert np.all(lp.A_ub @ x <= lp.b_ub + 1e-9) and np.all(x == np.round(x))


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_ilps())
def test_random_gomory(lp):
    res = numopt.run("gomory_cuts", lp, max_iter=200)
    assert_valid_result(res, max_iter=200)
    best = brute_force(lp)
    points = list(integer_points(lp, 6))
    for s in res.trace[1:]:
        cut = s.info["cut"]
        assert all(np.asarray(cut["coef"]) @ x <= cut["rhs"] + 1e-7 for x in points)
    if best is None:
        assert not res.converged and res.extra["status"] == "infeasible"
        return
    if res.extra["status"] == "max_iter":  # finiteness needs lexicographic rules
        assert not res.converged
        return
    assert res.converged, res.message
    assert res.fun == pytest.approx(best, abs=1e-7)


def documented_info_keys(module) -> set[str]:
    """Names listed in the module docstring's "Info keys:" section."""
    doc = module.__doc__ or ""
    section = doc.split("Info keys", 1)[1]
    keys: set[str] = set()
    for line in section.splitlines():
        m = re.match(r"^\s{4}([a-z_]+(?:, [a-z_]+)*)(?::| —)", line)
        if m:
            keys.update(k.strip() for k in m.group(1).split(","))
    return keys


def test_every_info_key_is_documented():
    documented = documented_info_keys(integer_module)
    for method in ("branch_and_bound", "gomory_cuts"):
        for pid in (*ILPS, "infeasible_2d"):
            res = numopt.run(method, problems.get(pid))
            for s in res.trace:
                assert set(s.info) <= documented, (method, set(s.info) - documented)
