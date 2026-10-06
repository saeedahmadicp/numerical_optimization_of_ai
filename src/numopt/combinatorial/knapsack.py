"""The 0-1 knapsack problem: dynamic programming, greedy, and branch-and-bound.

Problem (Martello & Toth 1990, §2.1): given n items with integer values vᵢ ≥ 0 and integer
weights wᵢ ≥ 1, and an integer capacity C ≥ 0,

    maximize  z = Σᵢ vᵢ xᵢ   subject to   Σᵢ wᵢ xᵢ ≤ C,   xᵢ ∈ {0, 1}.

All three methods take a :class:`~numopt.problems.combinatorial.KnapsackInstance` and return a
:class:`Result` with ``x`` = the 0/1 selection vector (list of n ints, original item order) and
``fun`` = its total value (a maximization: larger is better). ``extra`` holds ``weight`` (total
weight of ``x``), ``items`` (selected item indices), the Dantzig LP bound ``lp_bound`` and,
when the instance knows it, ``optimum_value``. Greedy adds ``greedy_value`` and
``best_single_item``; branch-and-bound adds ``nodes`` and ``upper_bound`` (= z when converged,
else the best U₁ bound among open nodes).

The ratio order used by greedy and branch-and-bound sorts items by vᵢ/wᵢ, largest first. Ratios
are compared exactly by cross-multiplication (vᵢwⱼ > vⱼwᵢ) and ties keep the smaller index, so
the order is identical in every implementation.

Evaluation counts: ``n_fev`` counts elementary objective evaluations — table cells z_j(d) for
DP, items examined for greedy, and bounded nodes for branch-and-bound.

Info keys:
    knapsack_dp:
        item: int | None — index of the item added in this row (None at k = 0).
        weight: int | None, value: int | None — w and v of that item.
        capacity: int — C.
        table_row: [C + 1] ints — z_k(d) for d = 0, …, C (optimum with the first k items).
        take: [C + 1] bools — True where z_k(d) > z_{k−1}(d), i.e. item k−1 is packed.
    knapsack_greedy:
        order: [n] ints — items sorted by value/weight ratio (largest first).
        phase: "start" | "greedy" | "single_item_fix".
        item: int | None — item examined in this step (the best single item in the fix step).
        ratio: float | None — its value/weight ratio.
        fits: bool | None — whether it fits in the residual capacity.
        taken: bool | None — whether the step put it into the knapsack.
        residual: int — capacity left after this step.
        value: int, weight: int — totals of the current selection.
    knapsack_branch_bound:
        order: [n] ints — the ratio order; depth d decides item order[d].
        nodes: [node] — every node bounded since the previous recorded step (one node when
            record_every = 1). A node is {id, parent, depth, item, decision, value, weight,
            bound, status, incumbent}: ``id`` is the pop order (= k), ``item``/``decision`` the
            branching that created it (None at the root), ``bound`` the Dantzig bound of the
            subtree, ``status`` ∈ {"branched", "pruned", "leaf"}, ``incumbent`` True when the
            node's value improved the best solution. The tree up to step k is the union of
            ``nodes`` over steps 0, …, k.
        best_value: int — incumbent value z after this step.
        open_nodes: int — nodes waiting on the depth-first stack.
"""

from __future__ import annotations

import functools
import sys
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import Result, Step
from ..problems.combinatorial import KnapsackInstance

#: Refuse DP tables with more cells than this (memory and trace size).
_MAX_DP_CELLS = 5_000_000
#: Largest table entry an int64 DP row can hold; every entry z_j(d) is at most Σᵢ vᵢ.
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validate(problem: Any) -> KnapsackInstance:
    """Return ``problem`` after checking it is a well-formed 0-1 knapsack instance."""
    if not isinstance(problem, KnapsackInstance):
        raise TypeError("problem must be a numopt KnapsackInstance")
    values, weights, capacity = problem.values, problem.weights, problem.capacity
    if len(values) != len(weights):
        raise ValueError(f"{problem.id}: {len(values)} values but {len(weights)} weights")
    if len(values) == 0:
        raise ValueError(f"{problem.id}: the instance has no items")

    def is_int(a: Any) -> bool:
        return isinstance(a, int | np.integer) and not isinstance(a, bool | np.bool_)

    if not all(is_int(v) and v >= 0 for v in values):
        raise ValueError(f"{problem.id}: values must be non-negative integers")
    if not all(is_int(w) and w >= 1 for w in weights):
        raise ValueError(f"{problem.id}: weights must be positive integers")
    if not (is_int(capacity) and capacity >= 0):
        raise ValueError(f"{problem.id}: capacity must be a non-negative integer")
    # Result.fun and the LP bound are floats; a larger total value cannot be represented.
    if sum(int(v) for v in values) > sys.float_info.max:
        raise ValueError(f"{problem.id}: the total value Σ vᵢ exceeds the float range")
    return problem


def ratio_order(values: tuple[int, ...], weights: tuple[int, ...]) -> list[int]:
    """Item indices sorted by vᵢ/wᵢ descending; exact cross-multiplication, ties by index."""

    def cmp(i: int, j: int) -> int:
        lhs, rhs = int(values[i]) * int(weights[j]), int(values[j]) * int(weights[i])
        if lhs != rhs:
            return -1 if lhs > rhs else 1
        return -1 if i < j else (1 if i > j else 0)

    return sorted(range(len(values)), key=functools.cmp_to_key(cmp))


def dantzig_bound(
    values: tuple[int, ...], weights: tuple[int, ...], order: list[int], start: int, residual: int
) -> tuple[float, int]:
    """Dantzig's LP bound on the best value from items ``order[start:]`` in capacity ``residual``.

    Fill items in ratio order until the split item s no longer fits, then add the fraction
    ``residual · v_s / w_s`` (Dantzig 1957; Martello & Toth 1990, §2.2.1). Returns the
    exact LP bound and its floor U₁ (Martello & Toth's bound for integer values), the latter
    computed in exact integer arithmetic.
    """
    total = 0
    for pos in range(start, len(order)):
        i = order[pos]
        w = int(weights[i])
        if w <= residual:
            residual -= w
            total += int(values[i])
        else:
            num = residual * int(values[i])
            return total + num / w, total + num // w
    return float(total), total


def _selection_result(
    method: str,
    inst: KnapsackInstance,
    x: list[int],
    converged: bool,
    message: str,
    n_iter: int,
    n_fev: int,
    trace: list[Step],
    **extra: Any,
) -> Result:
    value = sum(int(v) for v, xi in zip(inst.values, x, strict=True) if xi)
    weight = sum(int(w) for w, xi in zip(inst.weights, x, strict=True) if xi)
    lp, _ = dantzig_bound(
        inst.values, inst.weights, ratio_order(inst.values, inst.weights), 0, int(inst.capacity)
    )
    info: dict[str, Any] = {
        "weight": weight,
        "items": [i for i, xi in enumerate(x) if xi],
        "lp_bound": lp,
        **extra,
    }
    if inst.optimum_value is not None:
        info["optimum_value"] = inst.optimum_value
    return Result(
        method, x, float(value), converged, message, n_iter, n_fev, trace=trace, extra=info
    )


# --------------------------------------------------------------------------------------
# Dynamic programming
# --------------------------------------------------------------------------------------


@register(
    id="knapsack_dp",
    family="combinatorial",
    name="Knapsack dynamic programming",
    params=(),
    needs=("knapsack",),
    order="exact, O(n·C) time and memory (pseudo-polynomial)",
    summary="Fill a table of the best value for every capacity, one item at a time.",
    references=(
        "Bellman (1957), Dynamic Programming, Ch. 1",
        "Martello & Toth (1990), Knapsack Problems, §2.6",
        "Kellerer, Pferschy & Pisinger (2004), Knapsack Problems, §2.3",
    ),
)
def knapsack_dp(problem: KnapsackInstance) -> Result:
    """Bellman dynamic programming for the 0-1 knapsack problem.

    With z_0(d) = 0 for d = 0, …, C, the recursion over items j = 1, …, n is
    (Kellerer et al. 2004, §2.3)

        z_j(d) = z_{j−1}(d)                                   if d < w_j,
        z_j(d) = max(z_{j−1}(d), z_{j−1}(d − w_j) + v_j)      if d ≥ w_j,

    and z* = z_n(C). The decision table take_j(d) = [z_{j−1}(d − w_j) + v_j > z_{j−1}(d)]
    is stored, and an optimal x is recovered by backtracking from (n, C). Ties keep the item
    out, so the reported solution is the one that packs an item only when it strictly helps.

    Trace: k = 0 is the zero row; step k = j is the table row after item j − 1 (0-based) with
    ``x`` = the optimal selection using the first j items at capacity C. Stopping test: the
    table is complete after n rows; the method is exact, so ``converged`` is always True.
    Raises ``ValueError`` when (n + 1)(C + 1) exceeds 5·10⁶ cells. The table is int64 when
    Σ vᵢ < 2⁶³ and exact Python integers otherwise, so the arithmetic never overflows.
    """
    inst = _validate(problem)
    n, C = inst.n, int(inst.capacity)
    if (n + 1) * (C + 1) > _MAX_DP_CELLS:
        raise ValueError(
            f"{inst.id}: DP table of {(n + 1) * (C + 1)} cells exceeds {_MAX_DP_CELLS}"
        )
    values = [int(v) for v in inst.values]
    weights = [int(w) for w in inst.weights]

    # NOTE: every entry z_j(d) and every candidate z_{j−1}(d − w_j) + v_j is the value of a
    # feasible selection, so it is ≤ Σ vᵢ. An int64 table is exact when Σ vᵢ < 2⁶³; beyond that
    # int64 addition wraps silently, so the table falls back to exact Python ints (dtype=object).
    dtype: type = np.int64 if sum(values) <= _INT64_MAX else object
    z = np.zeros(C + 1, dtype=dtype)  # z_j(d), d = 0..C
    take = np.zeros((n, C + 1), dtype=bool)  # take[j, d]: item j packed in z_{j+1}(d)

    def backtrack(rows: int) -> list[int]:
        x = [0] * n
        d = C
        for j in range(rows - 1, -1, -1):
            if take[j, d]:
                x[j] = 1
                d -= weights[j]
        return x

    trace = [
        Step(
            0,
            [0] * n,
            0.0,
            info={
                "item": None,
                "weight": None,
                "value": None,
                "capacity": C,
                "table_row": z.tolist(),
                "take": [False] * (C + 1),
            },
        )
    ]
    for j in range(n):
        w, v = weights[j], values[j]
        if w <= C:
            candidate = z[: C + 1 - w] + v  # z_{j}(d − w) + v for d = w..C
            take[j, w:] = candidate > z[w:]
            z = z.copy()
            z[w:] = np.where(take[j, w:], candidate, z[w:])
        trace.append(
            Step(
                j + 1,
                backtrack(j + 1),
                float(z[C]),
                info={
                    "item": j,
                    "weight": w,
                    "value": v,
                    "capacity": C,
                    "table_row": z.tolist(),
                    "take": take[j].tolist(),
                },
            )
        )
    x = backtrack(n)
    return _selection_result(
        "knapsack_dp",
        inst,
        x,
        True,
        f"DP table complete ({n} × {C + 1}): z⋆ = zₙ(C) = {int(z[C])}",
        n,
        n * (C + 1),
        trace,
    )


# --------------------------------------------------------------------------------------
# Greedy
# --------------------------------------------------------------------------------------


@register(
    id="knapsack_greedy",
    family="combinatorial",
    name="Greedy by value/weight ratio",
    params=(
        ParamSpec(
            "single_item_fix",
            True,
            kind="bool",
            help="Also try the most valuable single item and keep the better solution "
            "(Ext-Greedy, value ≥ ½ z⋆).",
        ),
    ),
    needs=("knapsack",),
    order="heuristic, O(n log n); ½-approximation with the single-item fix",
    summary="Pack items in order of value per unit weight while they fit.",
    references=(
        "Kellerer, Pferschy & Pisinger (2004), Knapsack Problems, §2.1 (Greedy, Ext-Greedy)",
        "Martello & Toth (1990), Knapsack Problems, §2.4",
    ),
)
def knapsack_greedy(problem: KnapsackInstance, *, single_item_fix: bool = True) -> Result:
    """Greedy for the 0-1 knapsack, optionally with the best-single-item fix (Ext-Greedy).

    Sort items by vᵢ/wᵢ (largest first) and scan them once, packing every item that still fits
    (the scan continues past the first item that does not fit; Kellerer et al. 2004, §2.1).
    Plain greedy can be arbitrarily bad (one tiny high-ratio item can block a huge one). With
    ``single_item_fix`` the result is the better of the greedy solution and the most valuable
    item that fits on its own; this Ext-Greedy has performance ratio ½: z ≥ z*/2.

    Trace: k = 0 is the empty knapsack; step k examines the k-th item of the ratio order; the
    fix, when enabled, is one more step. Stopping test: every item has been examined. The
    method is a heuristic, so ``converged`` (= the pass completed) is not a claim of optimality.
    """
    inst = _validate(problem)
    n, C = inst.n, int(inst.capacity)
    values = [int(v) for v in inst.values]
    weights = [int(w) for w in inst.weights]
    order = ratio_order(inst.values, inst.weights)

    x = [0] * n
    residual, value, weight = C, 0, 0
    trace = [
        Step(
            0,
            list(x),
            0.0,
            info={
                "order": order,
                "phase": "start",
                "item": None,
                "ratio": None,
                "fits": None,
                "taken": None,
                "residual": residual,
                "value": 0,
                "weight": 0,
            },
        )
    ]
    for k, i in enumerate(order, start=1):
        fits = weights[i] <= residual
        if fits:
            x[i] = 1
            residual -= weights[i]
            value += values[i]
            weight += weights[i]
        trace.append(
            Step(
                k,
                list(x),
                float(value),
                info={
                    "order": order,
                    "phase": "greedy",
                    "item": i,
                    "ratio": values[i] / weights[i],
                    "fits": fits,
                    "taken": fits,
                    "residual": residual,
                    "value": value,
                    "weight": weight,
                },
            )
        )
    greedy_value = value
    n_iter, n_fev = n, n
    best_single: int | None = None
    if single_item_fix:
        # The most valuable item that fits alone (ties: smallest index).
        for i in range(n):
            if weights[i] <= C and (best_single is None or values[i] > values[best_single]):
                best_single = i
        n_fev += n
        n_iter += 1
        switched = best_single is not None and values[best_single] > greedy_value
        if switched and best_single is not None:
            x = [0] * n
            x[best_single] = 1
            value, weight, residual = (
                values[best_single],
                weights[best_single],
                C - weights[best_single],
            )
        trace.append(
            Step(
                n_iter,
                list(x),
                float(value),
                info={
                    "order": order,
                    "phase": "single_item_fix",
                    "item": best_single,
                    "ratio": None
                    if best_single is None
                    else values[best_single] / weights[best_single],
                    "fits": best_single is not None,
                    "taken": switched,
                    "residual": residual,
                    "value": value,
                    "weight": weight,
                },
            )
        )
        msg = (
            f"Ext-Greedy complete: greedy value {greedy_value}, best single item "
            f"{0 if best_single is None else values[best_single]}; kept {value} "
            "(heuristic: guaranteed ≥ ½ of the optimum)"
        )
    else:
        msg = f"greedy pass complete: value {value} (heuristic: no approximation guarantee)"
    return _selection_result(
        "knapsack_greedy",
        inst,
        x,
        True,
        msg,
        n_iter,
        n_fev,
        trace,
        greedy_value=greedy_value,
        best_single_item=best_single,
    )


# --------------------------------------------------------------------------------------
# Branch and bound
# --------------------------------------------------------------------------------------


@register(
    id="knapsack_branch_bound",
    family="combinatorial",
    name="Branch and bound (depth-first, Dantzig bound)",
    params=(
        ParamSpec(
            "max_nodes",
            100_000,
            kind="int",
            min=1,
            max=10_000_000,
            help="Stop (not converged) after bounding this many nodes.",
        ),
        ParamSpec(
            "record_every",
            1,
            kind="int",
            min=1,
            max=10_000,
            help="Record one trace step every this many nodes (the first and last are always recorded).",
        ),
    ),
    needs=("knapsack",),
    order="exact, exponential worst case",
    summary="Search the include/exclude tree depth-first and cut subtrees whose LP bound "
    "cannot beat the best solution found.",
    references=(
        "Horowitz & Sahni (1974), J. ACM 21(2), 277–292",
        "Martello & Toth (1990), Knapsack Problems, §2.5.1 (Horowitz–Sahni) and §2.2.1 (bound U₁)",
        "Kellerer, Pferschy & Pisinger (2004), Knapsack Problems, §2.4",
    ),
)
def knapsack_branch_bound(
    problem: KnapsackInstance, *, max_nodes: int = 100_000, record_every: int = 1
) -> Result:
    """Depth-first branch-and-bound with the Dantzig (LP-relaxation) bound.

    Items are taken in ratio order; a node at depth d has fixed x for items order[0..d−1] and
    holds (value V, weight W). Its bound is U = V + Dantzig bound of items order[d:] in capacity
    C − W, and since values are integers the node is pruned when U₁ = ⌊U⌋ ≤ z (Martello & Toth
    1990, §2.2.1). The include child (if it fits) is explored before the exclude child, so the
    first dive is the greedy solution, as in the Horowitz–Sahni forward move (Martello & Toth
    1990, §2.5.1). Every node's partial selection is feasible (the undecided items are left
    out), so a node with V > z becomes the incumbent. Bounds are computed when a node is
    popped, so they use the latest incumbent.

    Stopping test: the stack is empty — every subtree was explored or pruned, so the incumbent
    is optimal (``converged=True``). When ``max_nodes`` is reached first, the open nodes are
    bounded and ``extra["upper_bound"]`` = max(z, max U₁ over open nodes). If that equals z,
    every open subtree is dominated and the incumbent is proven optimal (``converged=True``);
    otherwise ``converged=False``.
    """
    inst = _validate(problem)
    if int(max_nodes) < 1 or int(record_every) < 1:
        raise ValueError("max_nodes and record_every must be ≥ 1")
    n, C = inst.n, int(inst.capacity)
    values = [int(v) for v in inst.values]
    weights = [int(w) for w in inst.weights]
    order = ratio_order(inst.values, inst.weights)

    # Stack entries: (depth, value, weight, chosen items, parent id, item, decision).
    stack: list[tuple[int, int, int, tuple[int, ...], int | None, int | None, int | None]] = [
        (0, 0, 0, (), None, None, None)
    ]
    best_value = 0
    best_items: tuple[int, ...] = ()
    trace: list[Step] = []
    pending: list[dict[str, Any]] = []
    k = -1

    def selection(items: tuple[int, ...]) -> list[int]:
        x = [0] * n
        for i in items:
            x[i] = 1
        return x

    while stack and k + 1 < max_nodes:
        depth, value, weight, items, parent, item, decision = stack.pop()
        k += 1
        rest, rest_floor = dantzig_bound(inst.values, inst.weights, order, depth, C - weight)
        bound, bound_floor = value + rest, value + rest_floor  # U = V + LP bound of the rest
        incumbent = value > best_value
        if incumbent:
            best_value, best_items = value, items
        if depth == n:
            status = "leaf"
        elif bound_floor <= best_value:
            status = "pruned"
        else:
            status = "branched"
            j = order[depth]
            stack.append((depth + 1, value, weight, items, k, j, 0))
            if weights[j] <= C - weight:
                stack.append(
                    (depth + 1, value + values[j], weight + weights[j], (*items, j), k, j, 1)
                )
        pending.append(
            {
                "id": k,
                "parent": parent,
                "depth": depth,
                "item": item,
                "decision": decision,
                "value": value,
                "weight": weight,
                "bound": bound,
                "status": status,
                "incumbent": incumbent,
            }
        )
        if k % record_every == 0 or not stack:
            trace.append(_bb_step(k, selection(best_items), best_value, order, pending, len(stack)))
            pending = []
    if pending:
        trace.append(_bb_step(k, selection(best_items), best_value, order, pending, len(stack)))

    x = selection(best_items)
    if not stack:
        return _selection_result(
            "knapsack_branch_bound",
            inst,
            x,
            True,
            f"search tree exhausted after {k + 1} nodes: the incumbent z = {best_value} is optimal",
            k,
            k + 1,
            trace,
            nodes=k + 1,
            upper_bound=float(best_value),
        )
    open_bound = max(
        v + dantzig_bound(inst.values, inst.weights, order, d, C - w)[1] for d, v, w, *_ in stack
    )
    upper = max(best_value, open_bound)
    # U₁ is a valid integer upper bound on every open subtree, so U₁ ≤ z for all open nodes
    # proves that none of them can beat the incumbent: it is optimal although nodes remain.
    proven = upper <= best_value
    message = (
        f"reached max_nodes={max_nodes} with {len(stack)} open nodes, all dominated "
        f"(U₁ ≤ z): the incumbent z = {best_value} is optimal"
        if proven
        else f"reached max_nodes={max_nodes} with {len(stack)} open nodes: incumbent "
        f"{best_value}, upper bound {upper}"
    )
    return _selection_result(
        "knapsack_branch_bound",
        inst,
        x,
        proven,
        message,
        k,
        k + 1,
        trace,
        nodes=k + 1,
        upper_bound=float(upper),
    )


def _bb_step(
    k: int,
    x: list[int],
    best_value: int,
    order: list[int],
    nodes: list[dict[str, Any]],
    n_open: int,
) -> Step:
    return Step(
        k,
        x,
        float(best_value),
        info={"order": order, "nodes": nodes, "best_value": best_value, "open_nodes": n_open},
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("knapsack_dp", "knapsack_10", {}),
    ("knapsack_dp", "knapsack_greedy_trap", {}),
    ("knapsack_greedy", "knapsack_20", {}),
    ("knapsack_greedy", "knapsack_greedy_trap", {"single_item_fix": False}),
    ("knapsack_greedy", "knapsack_greedy_trap", {"single_item_fix": True}),
    ("knapsack_branch_bound", "knapsack_10", {}),
    ("knapsack_branch_bound", "knapsack_20", {}),
    ("knapsack_branch_bound", "knapsack_greedy_trap", {}),
]
