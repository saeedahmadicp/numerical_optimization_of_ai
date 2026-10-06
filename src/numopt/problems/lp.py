"""Linear and integer programming test problems (kind ``"lp"``).

Every problem is a :class:`~numopt.core.types.LinearProgram`

    min/max cᵀx   s.t.   A_ub x ≤ b_ub,   A_eq x = b_eq,   x ≥ 0,

optionally with integrality flags. For 2-variable problems ``domain`` is the plotting box
``((x1_lo, x1_hi), (x2_lo, x2_hi))``; for larger problems it is empty. ``optimum`` and
``optimal_value`` hold the (unique) optimal solution and value in the original sense; they
are ``None`` for unbounded and infeasible problems (the description says which). All values
were checked against ``scipy.optimize.linprog(method="highs")`` / ``milp`` in
``tests/test_problems_lp.py``.
"""

from __future__ import annotations

import numpy as np

from ..core.types import LinearProgram
from .registry import factory


def _a(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _v(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


@factory("lp")
def wyndor() -> LinearProgram:
    return LinearProgram(
        id="wyndor",
        name="Wyndor Glass Co.",
        c=_v([3.0, 5.0]),
        A_ub=_a([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]),
        b_ub=_v([4.0, 12.0, 18.0]),
        sense="max",
        optimum=_v([2.0, 6.0]),
        optimal_value=36.0,
        description=(
            "Hillier & Lieberman, Introduction to Operations Research, §3.1: "
            "max 3x₁ + 5x₂ s.t. x₁ ≤ 4, 2x₂ ≤ 12, 3x₁ + 2x₂ ≤ 18. The origin is feasible."
        ),
        domain=((0.0, 7.0), (0.0, 10.0)),
    )


@factory("lp")
def diet_2d() -> LinearProgram:
    return LinearProgram(
        id="diet_2d",
        name="Two-food diet",
        c=_v([2.0, 3.0]),
        A_ub=_a([[-1.0, -1.0], [-1.0, -3.0], [-2.0, -1.0]]),
        b_ub=_v([-4.0, -6.0, -5.0]),
        sense="min",
        optimum=_v([3.0, 1.0]),
        optimal_value=9.0,
        description=(
            "min 2x₁ + 3x₂ s.t. x₁ + x₂ ≥ 4, x₁ + 3x₂ ≥ 6, 2x₁ + x₂ ≥ 5 (written as ≤ rows "
            "with negative right-hand sides). The origin is infeasible, so the primal simplex "
            "needs phase 1 or big-M; the slack basis is dual feasible (c ≥ 0), so the dual "
            "simplex starts directly."
        ),
        domain=((0.0, 7.0), (0.0, 6.0)),
    )


@factory("lp")
def degenerate_2d() -> LinearProgram:
    return LinearProgram(
        id="degenerate_2d",
        name="Degenerate vertex",
        c=_v([2.0, 1.0]),
        A_ub=_a([[1.0, 1.0], [1.0, -1.0], [1.0, 0.0]]),
        b_ub=_v([4.0, 2.0, 2.0]),
        sense="max",
        optimum=_v([2.0, 2.0]),
        optimal_value=6.0,
        description=(
            "max 2x₁ + x₂ s.t. x₁ + x₂ ≤ 4, x₁ − x₂ ≤ 2, x₁ ≤ 2. Three constraints meet at "
            "(2, 0), so the ratio test ties there; the simplex then makes a degenerate pivot "
            "(step length 0, basis changes, vertex does not) before it moves to (2, 2). "
            "Cycling needs m ≥ 2 rows and n − m ≥ 3 nonbasic columns in standard form "
            "(Marshall & Suurballe 1969); with 2 variables n − m = 2, so a 2-variable LP "
            "cannot cycle. See beale_cycling for a problem that does."
        ),
        domain=((0.0, 5.0), (0.0, 5.0)),
    )


@factory("lp")
def beale_cycling() -> LinearProgram:
    return LinearProgram(
        id="beale_cycling",
        name="Beale's cycling example",
        c=_v([10.0, -57.0, -9.0, -24.0]),
        A_ub=_a([[0.5, -5.5, -2.5, 9.0], [0.5, -1.5, -0.5, 1.0], [1.0, 0.0, 0.0, 0.0]]),
        b_ub=_v([0.0, 0.0, 1.0]),
        sense="max",
        optimum=_v([1.0, 0.0, 1.0, 0.0]),
        optimal_value=1.0,
        description=(
            "Chvátal, Linear Programming (1983), p. 31 (after Beale 1955). With the "
            "largest-coefficient (Dantzig) entering rule and smallest-subscript tie-breaking "
            "in the ratio test the simplex method cycles through 6 degenerate bases; Bland's "
            "rule terminates."
        ),
    )


@factory("lp")
def unbounded_2d() -> LinearProgram:
    return LinearProgram(
        id="unbounded_2d",
        name="Unbounded LP",
        c=_v([1.0, 1.0]),
        A_ub=_a([[-1.0, 1.0], [1.0, -2.0]]),
        b_ub=_v([1.0, 2.0]),
        sense="max",
        description=(
            "Unbounded: max x₁ + x₂ s.t. −x₁ + x₂ ≤ 1, x₁ − 2x₂ ≤ 2. The feasible region "
            "contains the ray (4, 2) + t·(1, 1), t ≥ 0, along which the objective grows "
            "without bound."
        ),
        domain=((0.0, 8.0), (0.0, 8.0)),
    )


@factory("lp")
def infeasible_2d() -> LinearProgram:
    return LinearProgram(
        id="infeasible_2d",
        name="Infeasible LP",
        c=_v([3.0, 2.0]),
        A_ub=_a([[1.0, 1.0], [-1.0, -1.0]]),
        b_ub=_v([2.0, -4.0]),
        sense="max",
        description=(
            "Infeasible: x₁ + x₂ ≤ 2 and x₁ + x₂ ≥ 4 cannot both hold. Phase 1 ends with a "
            "positive sum of artificial variables (minimum 2)."
        ),
        domain=((0.0, 5.0), (0.0, 5.0)),
    )


@factory("lp")
def klee_minty_3() -> LinearProgram:
    return LinearProgram(
        id="klee_minty_3",
        name="Klee–Minty cube (n = 3)",
        c=_v([100.0, 10.0, 1.0]),
        A_ub=_a([[1.0, 0.0, 0.0], [20.0, 1.0, 0.0], [200.0, 20.0, 1.0]]),
        b_ub=_v([1.0, 100.0, 10000.0]),
        sense="max",
        optimum=_v([0.0, 0.0, 10000.0]),
        optimal_value=10000.0,
        description=(
            "Klee & Minty (1972) in Chvátal's form (Linear Programming, ch. 4): maximize "
            r"$\sum_{j=1}^{n} 10^{n-j} x_j$ subject to "
            r"$2\sum_{j=1}^{i-1} 10^{i-j} x_j + x_i \le 100^{i-1}$ for $i = 1, \dots, n$ and "
            r"$\mathbf{x} \ge 0$. The Dantzig rule visits all $2^n = 8$ vertices (7 pivots) "
            "before it reaches the optimum."
        ),
    )


@factory("lp")
def transport_small() -> LinearProgram:
    # Variables x_ij (supplier i ∈ {1, 2} → customer j ∈ {1, 2, 3}), ordered x11, x12, x13,
    # x21, x22, x23. Supplies are capacities (≤); demands must be met exactly (=).
    cost = _a([[8.0, 6.0, 10.0], [9.0, 12.0, 13.0]])
    supply_rows = _a([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    demand_rows = _a([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
    return LinearProgram(
        id="transport_small",
        name="Small transportation problem",
        c=cost.reshape(-1),
        A_ub=supply_rows,
        b_ub=_v([25.0, 30.0]),
        A_eq=demand_rows,
        b_eq=_v([10.0, 25.0, 15.0]),
        sense="min",
        optimum=_v([0.0, 25.0, 0.0, 10.0, 0.0, 15.0]),
        optimal_value=435.0,
        description=(
            "Two suppliers (capacities 25, 30) and three customers (demands 10, 25, 15, met "
            "exactly) with unit costs [[8, 6, 10], [9, 12, 13]]. Six variables x₁₁…x₂₃, "
            "two ≤ rows and three equality rows."
        ),
    )


@factory("lp")
def ilp_knapsack_like_2d() -> LinearProgram:
    return LinearProgram(
        id="ilp_knapsack_like_2d",
        name="Two-variable integer program",
        c=_v([8.0, 5.0]),
        A_ub=_a([[1.0, 1.0], [9.0, 5.0]]),
        b_ub=_v([6.0, 45.0]),
        sense="max",
        integer=(True, True),
        optimum=_v([5.0, 0.0]),
        optimal_value=40.0,
        description=(
            "Winston, Operations Research (4th ed.), §9.3: max 8x₁ + 5x₂ s.t. x₁ + x₂ ≤ 6, "
            "9x₁ + 5x₂ ≤ 45, x integer. The LP relaxation optimum is (3.75, 2.25) with value "
            "41.25; the integer optimum is (5, 0) with value 40."
        ),
        domain=((0.0, 6.5), (0.0, 6.5)),
    )


@factory("lp")
def ilp_3var() -> LinearProgram:
    return LinearProgram(
        id="ilp_3var",
        name="Three-variable integer program",
        c=_v([4.0, 3.0, 3.0]),
        A_ub=_a([[4.0, 2.0, 1.0], [3.0, 4.0, 2.0], [2.0, 1.0, 3.0]]),
        b_ub=_v([10.0, 14.0, 7.0]),
        sense="max",
        integer=(True, True, True),
        optimum=_v([1.0, 2.0, 1.0]),
        optimal_value=13.0,
        description=(
            "max 4x₁ + 3x₂ + 3x₃ s.t. 4x₁ + 2x₂ + x₃ ≤ 10, 3x₁ + 4x₂ + 2x₃ ≤ 14, "
            "2x₁ + x₂ + 3x₃ ≤ 7, x integer. LP relaxation: (1.2, 2.2, 0.8), value 13.8; "
            "integer optimum (1, 2, 1), value 13."
        ),
    )
