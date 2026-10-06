"""0-1 knapsack methods: DP, greedy (Ext-Greedy), and branch-and-bound.

Oracles: brute force over all 2ⁿ subsets (n ≤ 12) and SciPy's MILP solver.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import pytest
from conftest import assert_valid_result
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import Bounds, LinearConstraint, milp

import numopt
from numopt import problems
from numopt.combinatorial import knapsack
from numopt.combinatorial.knapsack import dantzig_bound, ratio_order
from numopt.problems.combinatorial import KnapsackInstance

METHODS = ("knapsack_dp", "knapsack_greedy", "knapsack_branch_bound")
IDS = ("knapsack_10", "knapsack_20", "knapsack_greedy_trap")


def brute_force(values, weights, capacity) -> int:
    n = len(values)
    X = (np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
    W = X @ np.asarray(weights, dtype=np.int64)
    V = X @ np.asarray(values, dtype=np.int64)
    return int(np.max(np.where(W <= capacity, V, -1)))


def milp_value(inst: KnapsackInstance) -> float:
    v = np.asarray(inst.values, dtype=float)
    res = milp(
        -v,
        constraints=LinearConstraint(
            np.asarray(inst.weights, dtype=float)[None, :], -np.inf, inst.capacity
        ),
        integrality=np.ones_like(v),
        bounds=Bounds(0, 1),
    )
    return -float(num(res.fun))


def num(value: float | None) -> float:
    """Narrow an optional objective value (pyright) and fail loudly if it is missing."""
    assert value is not None
    return value


def make(values, weights, capacity) -> KnapsackInstance:
    return KnapsackInstance("custom", "custom", tuple(values), tuple(weights), capacity)


def check_selection(res, inst: KnapsackInstance) -> None:
    x = res.x
    assert len(x) == inst.n and set(x) <= {0, 1}
    weight = sum(w for w, xi in zip(inst.weights, x, strict=True) if xi)
    value = sum(v for v, xi in zip(inst.values, x, strict=True) if xi)
    assert weight <= inst.capacity, "selection must be feasible"
    assert num(res.fun) == value and res.extra["weight"] == weight
    assert res.extra["items"] == [i for i, xi in enumerate(x) if xi]


@st.composite
def instances(draw, max_n: int = 12) -> KnapsackInstance:
    n = draw(st.integers(1, max_n))
    weights = draw(st.lists(st.integers(1, 30), min_size=n, max_size=n))
    values = draw(st.lists(st.integers(0, 50), min_size=n, max_size=n))
    capacity = draw(st.integers(0, sum(weights) + 5))
    return make(values, weights, capacity)


# --------------------------------------------------------------------------------------
# Contract and library instances
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pid", IDS)
def test_contract_and_feasibility(method, pid):
    inst = problems.get(pid)
    res = numopt.run(method, inst)
    assert_valid_result(res)
    check_selection(res, inst)
    assert res.trace[-1].k == res.n_iter
    assert num(res.fun) <= inst.optimum_value <= res.extra["lp_bound"] + 1e-12


@pytest.mark.parametrize("method", ("knapsack_dp", "knapsack_branch_bound"))
@pytest.mark.parametrize("pid", IDS)
def test_exact_methods_reach_optimum_and_match_milp(method, pid):
    inst = problems.get(pid)
    res = numopt.run(method, inst)
    assert res.converged
    assert num(res.fun) == inst.optimum_value
    assert num(res.fun) == pytest.approx(milp_value(inst), abs=1e-6)


def test_greedy_values_on_library():
    assert num(numopt.run("knapsack_greedy", problems.get("knapsack_10")).fun) == 172
    assert num(numopt.run("knapsack_greedy", problems.get("knapsack_20")).fun) == 397
    trap = problems.get("knapsack_greedy_trap")
    plain = numopt.run("knapsack_greedy", trap, single_item_fix=False)
    fixed = numopt.run("knapsack_greedy", trap)
    assert num(plain.fun) == 53 and plain.x == [1, 1, 0]
    assert num(fixed.fun) == 53 and fixed.extra["best_single_item"] in (1, 2)
    assert num(plain.fun) < 0.55 * trap.optimum_value
    assert "no approximation guarantee" in plain.message and "½" in fixed.message


def test_single_item_fix_rescues_the_classic_trap():
    # v/w: item 0 has ratio 2, item 1 ratio 1; greedy packs item 0 and item 1 no longer fits.
    inst = make((2, 50), (1, 50), 50)
    plain = numopt.run("knapsack_greedy", inst, single_item_fix=False)
    fixed = numopt.run("knapsack_greedy", inst, single_item_fix=True)
    assert num(plain.fun) == 2 and num(fixed.fun) == 50 and fixed.x == [0, 1]
    assert fixed.trace[-1].info["phase"] == "single_item_fix" and fixed.trace[-1].info["taken"]


# --------------------------------------------------------------------------------------
# DP trace semantics
# --------------------------------------------------------------------------------------


def test_dp_table_rows_follow_the_bellman_recursion():
    inst = problems.get("knapsack_10")
    res = numopt.run("knapsack_dp", inst)
    C = inst.capacity
    rows = [np.asarray(s.info["table_row"]) for s in res.trace]
    assert len(rows) == inst.n + 1 and all(r.shape == (C + 1,) for r in rows)
    assert not rows[0].any()
    # Independent recomputation of z_j(d) straight from the recursion, cell by cell.
    prev = [0] * (C + 1)
    for j in range(inst.n):
        w, v = inst.weights[j], inst.values[j]
        cur = [max(prev[d], prev[d - w] + v) if d >= w else prev[d] for d in range(C + 1)]
        take = [d >= w and prev[d - w] + v > prev[d] for d in range(C + 1)]
        step = res.trace[j + 1]
        assert step.info["table_row"] == cur and step.info["take"] == take
        assert step.info["item"] == j and num(step.fun) == cur[C]
        # x at row j is optimal for the first j+1 items and uses only those items.
        assert sum(step.x[j + 1 :]) == 0
        assert sum(inst.values[i] for i in range(inst.n) if step.x[i]) == cur[C]
        prev = cur
    for r in rows:
        assert np.all(np.diff(r) >= 0), "z_j(d) is non-decreasing in d"
    assert res.n_fev == inst.n * (C + 1)


def test_dp_refuses_huge_tables():
    with pytest.raises(ValueError, match="cells"):
        numopt.run("knapsack_dp", make((1, 2), (1, 1), 10_000_000))


# --------------------------------------------------------------------------------------
# Branch and bound: tree, pruning, failure path
# --------------------------------------------------------------------------------------


def nodes_of(res) -> list[dict[str, Any]]:
    return [node for s in res.trace for node in s.info["nodes"]]


@pytest.mark.parametrize("record_every", (1, 4, 1000))
def test_branch_bound_tree_is_complete_and_consistent(record_every):
    inst = problems.get("knapsack_20")
    res = numopt.run("knapsack_branch_bound", inst, record_every=record_every)
    assert res.converged and num(res.fun) == inst.optimum_value
    nodes = nodes_of(res)
    assert [nd["id"] for nd in nodes] == list(range(res.n_iter + 1))
    assert res.n_fev == len(nodes) == res.extra["nodes"]
    by_id = {nd["id"]: nd for nd in nodes}
    order = ratio_order(inst.values, inst.weights)
    for nd in nodes[1:]:
        par = by_id[nd["parent"]]
        assert par["id"] < nd["id"] and par["status"] == "branched"
        assert nd["depth"] == par["depth"] + 1 and nd["item"] == order[par["depth"]]
        dv = inst.values[nd["item"]] if nd["decision"] else 0
        dw = inst.weights[nd["item"]] if nd["decision"] else 0
        assert (nd["value"], nd["weight"]) == (par["value"] + dv, par["weight"] + dw)
        assert nd["weight"] <= inst.capacity
        assert nd["bound"] <= par["bound"] + 1e-9, "bounds tighten down the tree"
    # Running incumbent never decreases and pruned nodes could not beat it.
    best = 0
    for nd in nodes:
        if nd["incumbent"]:
            assert nd["value"] > best
            best = nd["value"]
        if nd["status"] == "pruned":
            assert int(np.floor(nd["bound"] + 1e-9)) <= best
    assert best == inst.optimum_value
    assert all(len(s.info["nodes"]) >= 1 for s in res.trace)


def test_branch_bound_root_bound_is_dantzig():
    inst = problems.get("knapsack_10")
    res = numopt.run("knapsack_branch_bound", inst)
    root = res.trace[0].info["nodes"][0]
    assert root["parent"] is None and root["decision"] is None
    # Independent LP relaxation value from SciPy (continuous x in [0, 1]).
    v = np.asarray(inst.values, dtype=float)
    lp = milp(
        -v,
        constraints=LinearConstraint(
            np.asarray(inst.weights, dtype=float)[None, :], -np.inf, inst.capacity
        ),
        bounds=Bounds(0, 1),
    )
    np.testing.assert_allclose(root["bound"], -lp.fun, rtol=1e-12)
    np.testing.assert_allclose(res.extra["lp_bound"], -lp.fun, rtol=1e-12)


def test_branch_bound_node_limit_is_reported():
    inst = problems.get("knapsack_20")
    res = numopt.run("knapsack_branch_bound", inst, max_nodes=5)
    assert_valid_result(res)
    assert not res.converged and "max_nodes=5" in res.message
    assert res.n_fev == 5 and res.n_iter == 4
    assert num(res.fun) <= inst.optimum_value <= res.extra["upper_bound"]
    assert res.extra["upper_bound"] > num(res.fun), "an unproven stop must leave a gap"


# Regression (audit): when max_nodes stopped the search with every open node dominated
# (U₁ ≤ z), the method reported "incumbent 407, upper bound 407" but converged=False.


def test_branch_bound_node_limit_with_dominated_open_nodes_is_a_proof():
    inst = problems.get("knapsack_20")
    res = numopt.run("knapsack_branch_bound", inst, max_nodes=24)
    assert_valid_result(res)
    assert res.converged and "all dominated" in res.message and "max_nodes=24" in res.message
    assert num(res.fun) == inst.optimum_value == res.extra["upper_bound"]
    assert res.trace[-1].info["open_nodes"] > 0 and res.n_fev == 24
    check_selection(res, inst)


@settings(max_examples=1000, deadline=None)
@given(instances(), st.integers(1, 60))
def test_branch_bound_node_limit_flag_is_exact(inst, max_nodes):
    # converged ⇔ the incumbent is certified: it then equals brute force; otherwise the
    # reported upper bound is a valid bound strictly above the incumbent.
    opt = brute_force(inst.values, inst.weights, inst.capacity)
    res = knapsack.knapsack_branch_bound(inst, max_nodes=max_nodes)
    check_selection(res, inst)
    assert num(res.fun) <= opt <= res.extra["upper_bound"]
    if res.converged:
        assert num(res.fun) == opt == res.extra["upper_bound"]
    else:
        assert res.extra["upper_bound"] > num(res.fun)


# --------------------------------------------------------------------------------------
# Properties against brute force (Hypothesis)
# --------------------------------------------------------------------------------------


@settings(max_examples=1000, deadline=None)
@given(instances())
def test_dp_and_branch_bound_equal_brute_force(inst):
    opt = brute_force(inst.values, inst.weights, inst.capacity)
    dp = knapsack.knapsack_dp(inst)
    bb = knapsack.knapsack_branch_bound(inst)
    assert num(dp.fun) == opt and num(bb.fun) == opt and bb.converged
    check_selection(dp, inst)
    check_selection(bb, inst)


@settings(max_examples=1000, deadline=None)
@given(instances())
def test_greedy_bounds_and_lp_bound(inst):
    opt = brute_force(inst.values, inst.weights, inst.capacity)
    plain = knapsack.knapsack_greedy(inst, single_item_fix=False)
    ext = knapsack.knapsack_greedy(inst, single_item_fix=True)
    check_selection(plain, inst)
    check_selection(ext, inst)
    assert num(plain.fun) <= num(ext.fun) <= opt
    assert 2 * num(ext.fun) >= opt, "Ext-Greedy is a ½-approximation"
    lp, lp_floor = dantzig_bound(
        inst.values, inst.weights, ratio_order(inst.values, inst.weights), 0, inst.capacity
    )
    assert opt <= lp_floor <= lp < lp_floor + 1
    assert lp - num(ext.fun) <= max(inst.values) + 1e-9  # z_greedy ≥ U − v_max (split item)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 50), st.integers(1, 30)), min_size=1, max_size=15))
def test_ratio_order_is_exact_and_stable(items):
    values = tuple(v for v, _ in items)
    weights = tuple(w for _, w in items)
    order = ratio_order(values, weights)
    assert sorted(order) == list(range(len(items)))
    for a, b in itertools.pairwise(order):
        lhs, rhs = values[a] * weights[b], values[b] * weights[a]
        assert lhs > rhs or (lhs == rhs and a < b)


def exact_brute_force(values, weights, capacity) -> int:
    """Brute force in exact Python integers (no int64 wrap-around)."""
    best = 0
    for x in itertools.product((0, 1), repeat=len(values)):
        if sum(w * xi for w, xi in zip(weights, x, strict=True)) <= capacity:
            best = max(best, sum(v * xi for v, xi in zip(values, x, strict=True)))
    return best


def selected_value(inst: KnapsackInstance, x) -> int:
    return sum(v for v, xi in zip(inst.values, x, strict=True) if xi)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((2**62, 2**62), 2**63),  # int64 addition used to wrap: z* reported as 2⁶²
        ((2**62, 2**62 - 1), 2**63 - 1),  # Σ vᵢ = 2⁶³ − 1: the largest int64 table
        ((2**64, 1), 2**64 + 1),  # used to raise OverflowError
    ],
)
def test_dp_is_exact_beyond_the_int64_range(values, expected):
    inst = make(values, (1, 1), 2)
    res = knapsack.knapsack_dp(inst)
    assert_valid_result(res)
    assert res.converged and res.x == [1, 1]
    assert selected_value(inst, res.x) == expected
    assert res.trace[-1].info["table_row"][-1] == expected
    assert str(expected) in res.message
    json.dumps(res.to_dict(), allow_nan=False)


@settings(max_examples=1000, deadline=None)
@given(
    st.integers(1, 8).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(0, 2**70), min_size=n, max_size=n),
            st.lists(st.integers(1, 30), min_size=n, max_size=n),
        )
    ),
    st.integers(0, 250),
)
def test_dp_and_branch_bound_equal_brute_force_for_huge_values(items, capacity):
    values, weights = items
    inst = make(values, weights, capacity)
    opt = exact_brute_force(values, weights, capacity)
    for method in ("knapsack_dp", "knapsack_branch_bound"):
        res = numopt.run(method, inst)
        assert res.converged
        assert sum(w for w, xi in zip(weights, res.x, strict=True) if xi) <= capacity
        assert selected_value(inst, res.x) == opt, method


@pytest.mark.parametrize("method", METHODS)
def test_total_value_beyond_float_range_raises(method):
    with pytest.raises(ValueError, match="float range"):
        numopt.run(method, make((2**1024, 1), (1, 1), 2))


# --------------------------------------------------------------------------------------
# Input validation and fixtures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("values", "weights", "capacity"),
    [
        ((1, 2), (1,), 5),  # length mismatch
        ((), (), 5),  # empty
        ((1, 2), (0, 1), 5),  # zero weight
        ((1, -2), (1, 1), 5),  # negative value
        ((1, 2), (1.5, 1), 5),  # non-integer weight
        ((1, 2), (1, 1), -1),  # negative capacity
        ((True, 2), (1, 1), 3),  # bool is not an integer value
    ],
)
def test_invalid_instances_raise(method, values, weights, capacity):
    with pytest.raises(ValueError):
        numopt.run(method, make(values, weights, capacity))


@pytest.mark.parametrize("method", METHODS)
def test_non_instance_is_rejected(method):
    with pytest.raises(TypeError):
        numopt.run(method, [(1, 2)])


def test_zero_capacity_and_nothing_fits():
    inst = make((5, 7), (3, 4), 2)
    for method in METHODS:
        res = numopt.run(method, inst)
        assert num(res.fun) == 0 and res.x == [0, 0] and res.converged


def test_fixture_cases_are_valid_and_small():
    assert len(knapsack.FIXTURE_CASES) >= 3
    assert {m for m, _, _ in knapsack.FIXTURE_CASES} == set(METHODS)
    for method, pid, params in knapsack.FIXTURE_CASES:
        res = numopt.run(method, problems.get(pid), **params)
        assert_valid_result(res)
        assert len(res.trace) <= 400
        json.dumps({"params": params, "result": res.to_dict()}, allow_nan=False)
