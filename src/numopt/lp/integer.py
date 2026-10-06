"""Integer linear programming: LP-based branch-and-bound and Gomory fractional cuts.

Both methods solve ``min/max cᵀx s.t. A_ub x ≤ b_ub, A_eq x = b_eq, x ≥ 0, x_j ∈ ℤ`` for the
variables flagged in ``LinearProgram.integer`` (an empty tuple means *all* variables are
integer). ``branch_and_bound`` solves its LP relaxations with this package's floating-point
tableau simplex (:func:`numopt.lp.simplex.solve_two_phase`, Bland's rule so that no
relaxation can cycle). ``gomory_cuts`` runs the whole method, LP relaxation included, in exact
rational arithmetic (see its docstring for why).

``Result.x`` is the best integer solution found (``branch_and_bound`` rounds the integer
variables to the nearest integer; ``# NOTE:`` its LP returns them within ``int_tol`` of an
integer); ``Result.fun`` = cᵀx in the original sense; ``Result.extra["status"]`` is
``"optimal"``, ``"infeasible"``, ``"unbounded"`` (LP relaxation unbounded), ``"max_iter"`` or
``"lp_failure"`` (an LP relaxation hit its pivot limit).

Info keys (``branch_and_bound``; one Step per node taken from the open list):
    node: int — id of the node processed in this Step (0 = root).
    tree: [{id, parent, depth, branch, bounds, lp_value, lp_x, status}] — every node created so
        far. ``branch`` = [j, "<=" | ">=", v] (the bound added to the parent, None at the root);
        ``bounds`` = [[lo, hi], ...] per variable (hi = None when unbounded); ``lp_value`` and
        ``lp_x`` are the node's LP optimum in the original sense (None until solved / if
        infeasible); ``status`` ∈ {"open", "branched", "integer", "infeasible", "pruned_bound",
        "unbounded"}.
    bounds: [[lo, hi]] — variable bounds of the processed node (the sub-box it searches).
    lp_x: [float] | None — LP relaxation optimum at the node (same as ``Step.x``).
    branch_var: int | None — variable branched on (the most fractional one), if any.
    incumbent: [float] | None — best integer solution so far.
    incumbent_value: float | None — its objective value cᵀx (original sense).
    best_bound: float | None — best objective any unexplored node could still reach
        (original sense; equals ``incumbent_value`` when the search is complete).
    gap: float | None — |incumbent_value − best_bound| / (1 + |incumbent_value|).

Info keys (``gomory_cuts``; one Step per cut round, Step 0 = LP relaxation optimum):
    tableau, row_labels, col_labels, basis, nonbasis — the optimal tableau after the round
        (same layout as :mod:`numopt.lp.simplex`; cut slacks are named ``g1, g2, ...``; the
        exact rational entries rounded to float). On a failed relaxation: its final tableau.
    vertex: [float] — LP optimum x of the original variables (same as ``Step.x``).
    cuts: [{coef, rhs, active}] — every cut added so far, as coefᵀx ≤ rhs in the original
        variables; ``active`` is False once the cut's slack became basic and its row was
        deleted from the tableau (the vertex may later violate an inactive cut).
    cut: {coef, rhs, source_row, source_var, f0, f} | None — the cut that produced this Step:
        ``source_row`` is the tableau row of the previous optimal tableau it was read from,
        ``source_var`` the label of that row's basic variable, ``f0`` the fractional part of
        its right-hand side and ``f`` the fractional parts of the row (one per column).
    dual_pivots: int — dual simplex pivots needed to reoptimize after the cut.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import LinearProgram, Result, Step
from .simplex import StandardForm, check_lp, lp_arrays, solve_two_phase, standard_form

#: Pivot limit for one LP relaxation (far above what the library problems need).
_LP_MAX_PIVOTS = 5000
_LP_TOL = 1e-9
#: gomory_cuts: when the tableau holds more cut columns than this, the rows and columns of the
#: cuts whose slack is basic are deleted (see the gomory_cuts docstring).
_MAX_CUT_ROWS = 50


def _integer_mask(lp: LinearProgram) -> np.ndarray:
    n = int(np.size(lp.c))
    if not lp.integer:
        return np.ones(n, dtype=bool)
    if len(lp.integer) != n:
        raise ValueError(f"{lp.id}: integer flags must have one entry per variable")
    return np.array(lp.integer, dtype=bool)


def _frac_dist(v: float) -> float:
    """Distance from v to the nearest integer."""
    return abs(v - round(v))


# --------------------------------------------------------------------------------------
# Branch and bound
# --------------------------------------------------------------------------------------


def _node_lp(lp: LinearProgram, lo: np.ndarray, hi: np.ndarray) -> LinearProgram:
    """The relaxation with the node bounds lo ≤ x ≤ hi appended as ≤ rows."""
    c, a_ub, b_ub, _, _ = lp_arrays(lp)
    n = c.size
    rows, rhs = [a_ub], [b_ub]
    for j in range(n):
        if math.isfinite(hi[j]):
            e = np.zeros((1, n))
            e[0, j] = 1.0
            rows.append(e)
            rhs.append(np.array([hi[j]]))
        if lo[j] > 0:
            e = np.zeros((1, n))
            e[0, j] = -1.0
            rows.append(e)
            rhs.append(np.array([-lo[j]]))
    return replace(lp, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), integer=())


@register(
    id="branch_and_bound",
    family="lp",
    name="Branch and bound",
    params=(
        ParamSpec(
            "strategy",
            "best_bound",
            kind="choice",
            choices=("best_bound", "depth_first"),
            help="Node selection: best LP bound first, or deepest node first.",
        ),
        ParamSpec(
            "int_tol",
            1e-6,
            min=1e-10,
            max=1e-2,
            log=True,
            help="A value within int_tol of an integer counts as integral.",
        ),
        ParamSpec(
            "max_iter",
            200,
            kind="int",
            min=1,
            max=10_000,
            help="Maximum number of nodes processed after the root (Step 0 is the root).",
        ),
    ),
    needs=("lp", "integer"),
    order="finite (exponential worst case)",
    summary="Solve LP relaxations, split on a fractional variable, and discard sub-problems whose bound cannot beat the best integer solution.",
    references=(
        "Land & Doig (1960), Econometrica 28(3):497–520",
        "Wolsey, Integer Programming (1998), §7.3–7.4",
        "Winston, Operations Research (4th ed.), §9.3",
    ),
)
def branch_and_bound(
    problem: LinearProgram,
    *,
    strategy: str = "best_bound",
    int_tol: float = 1e-6,
    max_iter: int = 200,
) -> Result:
    """LP-based branch and bound (Land & Doig 1960; Wolsey 1998, §7.3).

    A node is the LP relaxation with extra bounds lo ≤ x ≤ hi. Processing a node:

    1. If its parent's LP value cannot beat the incumbent, prune it (no LP solve).
    2. Solve the LP (two-phase simplex, Bland's rule). Infeasible → prune.
       Not better than the incumbent → prune by bound.
    3. If every integer variable is within ``int_tol`` of an integer, the solution is a new
       incumbent (prune by integrality).
    4. Otherwise branch on the most fractional variable xⱼ = v (fractional part closest to
       ½, ties: smallest j): children with xⱼ ≤ ⌊v⌋ (created first) and xⱼ ≥ ⌈v⌉.

    Node selection: ``best_bound`` takes the open node with the best parent LP value (ties:
    creation order); ``depth_first`` takes the deepest open node (ties: creation order, so
    the ≤ child is explored first). ``# NOTE:`` each relaxation is solved from scratch, not
    warm-started from the parent's tableau with the dual simplex.

    Stops (converged) when no open node is left and an incumbent exists: then it is optimal.
    No incumbent → infeasible; an unbounded root relaxation → ``"unbounded"`` (the ILP is
    unbounded or infeasible); node limit → ``"max_iter"``; all with ``converged=False``. The
    node limit follows the trace contract: Step 0 is the root and ``max_iter`` counts the nodes
    processed after it, so at most ``max_iter + 1`` nodes are processed.
    """
    lp = check_lp(problem)
    if strategy not in ("best_bound", "depth_first"):
        raise ValueError("strategy must be 'best_bound' or 'depth_first'")
    if not 0 < int_tol < 0.5 or max_iter < 1:
        raise ValueError("int_tol must be in (0, 0.5) and max_iter ≥ 1")
    is_int = _integer_mask(lp)
    c = np.asarray(lp.c, dtype=np.float64)
    n = c.size
    sign = 1.0 if lp.sense == "min" else -1.0  # min-form value = sign · cᵀx

    tree: list[dict[str, Any]] = []
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    parent_bound: list[float] = []  # min-form LP value of the parent (−inf at the root)

    def new_node(
        parent: int | None, branch: list[Any] | None, lo: np.ndarray, hi: np.ndarray
    ) -> None:
        nid = len(tree)
        tree.append(
            {
                "id": nid,
                "parent": parent,
                "depth": 0 if parent is None else tree[parent]["depth"] + 1,
                "branch": branch,
                "bounds": [
                    [float(lo[j]), None if math.isinf(hi[j]) else float(hi[j])] for j in range(n)
                ],
                "lp_value": None,
                "lp_x": None,
                "status": "open",
            }
        )
        bounds.append((lo, hi))
        parent_bound.append(-math.inf if parent is None else float(sign * tree[parent]["lp_value"]))

    new_node(None, None, np.zeros(n), np.full(n, math.inf))
    inc_x: np.ndarray | None = None
    inc_val = math.inf  # min form
    trace: list[Step] = []
    total_pivots = 0
    status: str | None = None
    lp_message = ""

    def prune_tol(v: float) -> float:
        return 1e-9 * (1.0 + abs(v))

    def open_nodes() -> list[int]:
        return [t["id"] for t in tree if t["status"] == "open"]

    def best_open_bound() -> float:
        vals = [parent_bound[i] for i in open_nodes()]
        return min([inc_val, *vals])

    k = 0
    while True:
        candidates = open_nodes()
        if not candidates:
            status = "optimal" if inc_x is not None else "infeasible"
            break
        if k > max_iter:
            status = "max_iter"
            break
        if strategy == "best_bound":
            nid = min(candidates, key=lambda i: (parent_bound[i], i))
        else:
            nid = min(candidates, key=lambda i: (-tree[i]["depth"], i))
        node = tree[nid]
        lo, hi = bounds[nid]
        lp_x: np.ndarray | None = None
        branch_var: int | None = None
        if parent_bound[nid] >= inc_val - prune_tol(inc_val):
            node["status"] = "pruned_bound"
        else:
            out = solve_two_phase(
                standard_form(_node_lp(lp, lo, hi)),
                rule="bland",
                tol=_LP_TOL,
                max_iter=_LP_MAX_PIVOTS,
                record=False,
            )
            total_pivots += out.pivots
            if out.status == "infeasible":
                node["status"] = "infeasible"
            elif out.status == "unbounded":
                node["status"] = "unbounded"
                status = "unbounded"
            elif out.status != "optimal":
                node["status"] = "infeasible"
                status = "lp_failure"
                lp_message = out.message
            else:
                lp_x = out.x.copy()
                val = float(sign * (c @ lp_x))
                node["lp_value"] = float(c @ lp_x)
                node["lp_x"] = lp_x.tolist()
                if val >= inc_val - prune_tol(inc_val):
                    node["status"] = "pruned_bound"
                else:
                    dist = np.array([_frac_dist(v) for v in lp_x])
                    frac = np.where(is_int, dist, 0.0)
                    if np.all(frac <= int_tol):
                        node["status"] = "integer"
                        inc_x = np.where(is_int, np.round(lp_x), lp_x)
                        inc_val = float(sign * (c @ inc_x))
                    else:
                        node["status"] = "branched"
                        # Most fractional: distance to nearest integer closest to ½.
                        score = np.where(frac > int_tol, frac, -1.0)
                        branch_var = int(np.flatnonzero(score >= score.max() - 1e-12)[0])
                        v = float(lp_x[branch_var])
                        hi_dn, lo_up = hi.copy(), lo.copy()
                        hi_dn[branch_var] = math.floor(v)
                        lo_up[branch_var] = math.ceil(v)
                        new_node(nid, [branch_var, "<=", float(math.floor(v))], lo.copy(), hi_dn)
                        new_node(nid, [branch_var, ">=", float(math.ceil(v))], lo_up, hi.copy())
        bb = best_open_bound()
        inc_orig = None if inc_x is None else float(c @ inc_x)
        bb_orig = None if math.isinf(bb) else float(sign * bb)
        gap = (
            None
            if inc_orig is None or bb_orig is None
            else abs(inc_orig - bb_orig) / (1.0 + abs(inc_orig))
        )
        trace.append(
            Step(
                k,
                lp_x,
                None if lp_x is None else float(c @ lp_x),
                info={
                    "node": nid,
                    "tree": [dict(t) for t in tree],
                    "bounds": node["bounds"],
                    "lp_x": lp_x,
                    "branch_var": branch_var,
                    "incumbent": inc_x,
                    "incumbent_value": inc_orig,
                    "best_bound": bb_orig,
                    "gap": gap,
                },
            )
        )
        if status in ("unbounded", "lp_failure"):
            break
        k += 1

    messages = {
        "optimal": f"optimal: search tree exhausted after {len(trace)} nodes",
        "infeasible": "ILP is infeasible: every node was infeasible or pruned without an integer solution",
        "unbounded": "LP relaxation is unbounded: the ILP is unbounded or infeasible",
        "max_iter": (
            f"reached max_iter={max_iter}: processed the root and {max_iter} more nodes "
            "with open nodes left"
        ),
    }
    message = messages.get(status or "", "")
    if status == "lp_failure":
        message = f"an LP relaxation failed: {lp_message}"
    x_out = inc_x if inc_x is not None else np.full(n, np.nan)
    return Result(
        method="branch_and_bound",
        x=x_out,
        fun=None if inc_x is None else float(c @ inc_x),
        converged=status == "optimal",
        message=message,
        n_iter=trace[-1].k,
        trace=trace,
        extra={
            "status": status,
            "nodes": len(tree),
            "lp_pivots": total_pivots,
            "tree": [dict(t) for t in tree],
        },
    )


# --------------------------------------------------------------------------------------
# Gomory fractional cuts (exact rational arithmetic)
# --------------------------------------------------------------------------------------


class _QTableau:
    """A simplex tableau over ℚ, same layout as :class:`numopt.lp.simplex.Tableau`.

    ``T[0] = [c̄ | −z̃]`` (one objective row) and ``T[1 + i] = [B⁻¹A | x_B]`` for constraint
    row i, every entry a :class:`fractions.Fraction`; ``basis[i]`` is the basic column of row i.
    """

    def __init__(self, T: list[list[Fraction]], basis: list[int], labels: list[str]) -> None:
        self.T = T
        self.basis = basis
        self.labels = labels

    @property
    def m(self) -> int:
        return len(self.T) - 1

    @property
    def N(self) -> int:
        return len(self.T[0]) - 1

    def pivot(self, i: int, j: int) -> None:
        """Gauss–Jordan pivot on constraint row ``i`` and column ``j`` (exact)."""
        r = 1 + i
        piv = self.T[r][j]
        prow = [v / piv for v in self.T[r]]
        self.T[r] = prow
        for k, row in enumerate(self.T):
            f = row[j]
            if k != r and f != 0:
                self.T[k] = [a - f * p for a, p in zip(row, prow, strict=True)]
        self.basis[i] = j

    def set_objective(self, cost: list[Fraction]) -> None:
        """Row 0 = [c − (B⁻¹A)ᵀc_B | −c_Bᵀx_B] for the column costs ``cost``."""
        row = [*cost, Fraction(0)]
        for i, jb in enumerate(self.basis):
            cb = cost[jb]
            if cb != 0:
                row = [a - cb * t for a, t in zip(row, self.T[1 + i], strict=True)]
        self.T[0] = row

    def values(self) -> list[Fraction]:
        """Values of all N columns at the current basic solution."""
        z = [Fraction(0)] * self.N
        for i, j in enumerate(self.basis):
            z[j] = self.T[1 + i][-1]
        return z

    def snapshot(self) -> dict[str, Any]:
        """The tableau keys of :mod:`numopt.lp.simplex` (entries rounded to float)."""
        basic = set(self.basis)
        return {
            "tableau": [[float(v) for v in row] for row in self.T],
            "row_labels": ["z"] + [self.labels[j] for j in self.basis],
            "col_labels": [*self.labels, "rhs"],
            "basis": list(self.basis),
            "nonbasis": [j for j in range(self.N) if j not in basic],
        }


def _q_primal(tab: _QTableau, budget: int) -> tuple[str, int]:
    """Primal simplex with Bland's rule (Bland 1977) until optimal / unbounded / budget.

    Entering: the smallest column with c̄ⱼ < 0; leaving: the minimum ratio x_B,i / ā_ij over
    ā_ij > 0, ties to the smallest basic index. In exact arithmetic this cannot cycle.
    """
    pivots = 0
    while True:
        obj = tab.T[0]
        j = next((j for j in range(tab.N) if obj[j] < 0), None)
        if j is None:
            return "optimal", pivots
        rows = [i for i in range(tab.m) if tab.T[1 + i][j] > 0]
        if not rows:
            return "unbounded", pivots
        if pivots >= budget:
            return "max_iter", pivots
        col = j
        i = min(rows, key=lambda i: (tab.T[1 + i][-1] / tab.T[1 + i][col], tab.basis[i]))
        tab.pivot(i, col)
        pivots += 1


def _q_dual(tab: _QTableau, budget: int) -> tuple[str, int]:
    """Dual simplex with Bland's rule (Bertsimas & Tsitsiklis §4.5) until primal feasible.

    Leaving: the row with x_B,r < 0 whose basic variable has the smallest index; entering: the
    minimum ratio c̄ⱼ / |ā_rⱼ| over ā_rⱼ < 0, ties to the smallest column. No such column means
    row r reads Σ ā_rⱼ zⱼ = x_B,r < 0 with every ā_rⱼ ≥ 0, so the LP is infeasible.
    """
    pivots = 0
    while True:
        neg = [i for i in range(tab.m) if tab.T[1 + i][-1] < 0]
        if not neg:
            return "optimal", pivots
        r = min(neg, key=lambda i: tab.basis[i])
        row = tab.T[1 + r]
        cols = [j for j in range(tab.N) if row[j] < 0]
        if not cols:
            return "infeasible", pivots
        if pivots >= budget:
            return "max_iter", pivots
        j = min(cols, key=lambda j: (tab.T[0][j] / -row[j], j))
        tab.pivot(r, j)
        pivots += 1


def _q_two_phase(sf: StandardForm, budget: int) -> tuple[str, _QTableau, int]:
    """Two-phase simplex over ℚ (Bertsimas & Tsitsiklis §3.5) on an integer standard form.

    Phase 1 minimizes the sum of the artificials of the rows without a unit column; a zero-level
    artificial left in the basis is pivoted out on any nonzero entry of a real column, or its
    row is deleted when it has none (a redundant constraint). Phase 2 uses the costs ``sf.c``
    (``Fraction(float)`` is the exact binary value). Returns ``(status, tableau, pivots)``.
    """
    m, N = sf.A.shape
    art_rows = [i for i in range(m) if sf.unit_col[i] is None]
    n_art = len(art_rows)
    T: list[list[Fraction]] = [[Fraction(0)] * (N + n_art + 1)]
    basis: list[int] = []
    for i in range(m):
        row = [Fraction(int(v)) for v in sf.A[i]] + [Fraction(0)] * n_art + [Fraction(int(sf.b[i]))]
        uc = sf.unit_col[i]
        if uc is None:
            k = art_rows.index(i)
            row[N + k] = Fraction(1)
            basis.append(N + k)
        else:
            basis.append(uc)
        T.append(row)
    tab = _QTableau(T, basis, sf.labels + [f"a{i + 1}" for i in art_rows])
    pivots = 0
    if n_art:
        tab.set_objective([Fraction(0)] * N + [Fraction(1)] * n_art)
        status, p = _q_primal(tab, budget)
        pivots += p
        if status == "max_iter":
            return status, tab, pivots
        if tab.T[0][-1] != 0:  # row 0 holds −w; w > 0 means no feasible point
            return "infeasible", tab, pivots
        i = 0
        while i < tab.m:
            if tab.basis[i] < N:
                i += 1
                continue
            j = next((j for j in range(N) if tab.T[1 + i][j] != 0), None)
            if j is None:
                del tab.T[1 + i]
                del tab.basis[i]
            else:
                tab.pivot(i, j)
                pivots += 1
                i += 1
        tab.T = [row[:N] + row[-1:] for row in tab.T]
        tab.labels = tab.labels[:N]
    tab.set_objective([Fraction(float(v)) for v in sf.c])
    status, p = _q_primal(tab, budget - pivots)
    return status, tab, pivots + p


def _frac(v: Fraction) -> Fraction:
    """Fractional part f(v) = v − ⌊v⌋ ∈ [0, 1), exact."""
    return v - math.floor(v)


def _is_integer_array(arr: np.ndarray) -> bool:
    return bool(np.all(arr == np.round(arr))) and bool(np.all(np.abs(arr) < 2.0**53))


@register(
    id="gomory_cuts",
    family="lp",
    name="Gomory fractional cuts",
    params=(
        ParamSpec("max_iter", 50, kind="int", min=1, max=1000, help="Maximum number of cuts."),
    ),
    needs=("lp", "integer"),
    order="finite with lexicographic rules (Gomory 1958); slow in practice",
    summary="Read a valid inequality off a fractional row of the optimal tableau, add it, and reoptimize with the dual simplex.",
    references=(
        "Gomory (1958), Bull. AMS 64:275–278",
        "Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §11.1",
        "Wolsey, Integer Programming (1998), §8.6",
    ),
)
def gomory_cuts(problem: LinearProgram, *, max_iter: int = 50) -> Result:
    """Gomory's fractional cutting-plane algorithm for pure integer programs.

    Requires every variable integer and integer data (A, b), so that every slack variable
    is integer as well. From an optimal tableau row with fractional right-hand side,
    x_B,r + Σ_{j∈N} ā_rⱼ zⱼ = b̄_r, the Gomory fractional cut (Bertsimas & Tsitsiklis §11.1,
    Wolsey §8.6)

        Σ_{j∈N} f(ā_rⱼ) zⱼ ≥ f(b̄_r),    f(v) = v − ⌊v⌋,

    holds for every integer feasible point and is violated by the current vertex (where
    z_N = 0 and f(b̄_r) > 0). It is appended as the row −Σ f(ā_rⱼ) zⱼ + g = −f(b̄_r) with a new
    basic slack g ≥ 0, which is integer at every integer point because
    g = −x_B,r − Σ ⌊ā_rⱼ⌋ zⱼ + ⌊b̄_r⌋. The tableau stays dual feasible, so the dual simplex
    (Bland's rule) reoptimizes. The source row has the largest f(b̄_r) (ties: lowest row).

    ``# NOTE:`` the LP relaxation, every cut and every dual simplex pivot are computed in exact
    rational arithmetic (:class:`fractions.Fraction`), and the tableau is rounded to float
    only for display. The cut validity argument needs the exact fractional parts f(ā_rⱼ):
    rounding one coefficient makes g non-integer at integer points, and the cuts that later
    rows of g give can then remove the integer optimum. Floating-point tableaux drift by
    ~1e-9 after about 30 cuts, which is enough to do this. With exact arithmetic the
    integrality test needs no tolerance.

    ``# NOTE:`` when the tableau holds more than 50 cut columns after a reoptimization, the row
    and column of every cut whose slack g is basic are deleted. Column g is then a unit
    vector, so the remaining rows are the tableau of the LP without that cut, and the basis
    stays primal and dual feasible (same vertex). The deleted cut stays valid but is no longer
    enforced. Without the deletion the tableau grows by one row per cut and exact arithmetic
    takes minutes for 1000 cuts; deleting at every round instead makes some instances
    regenerate the same cuts for ever (measured on random ILPs).

    ``# NOTE:`` Gomory's finiteness proof needs lexicographic dual simplex rules; without
    them the method stops at ``max_iter`` cuts if it does not finish.

    Stops (converged) when every basic value of the LP optimum is an integer. An infeasible or
    unbounded relaxation, a cut that makes the LP infeasible (then the ILP is infeasible: the
    cuts are valid and the dual simplex infeasibility is exact), and the cut limit give
    ``converged=False``. Raises ValueError for non-integer data or a mixed program.
    """
    lp = check_lp(problem)
    if max_iter < 1:
        raise ValueError("max_iter must be ≥ 1")
    if not np.all(_integer_mask(lp)):
        raise ValueError(f"{lp.id}: gomory_cuts needs every variable integer (pure ILP)")
    _, a_ub, b_ub, a_eq, b_eq = lp_arrays(lp)
    for arr in (a_ub, b_ub, a_eq, b_eq):
        if arr.size and not _is_integer_array(arr):
            raise ValueError(f"{lp.id}: gomory_cuts needs integer constraint data A and b")
    sf = standard_form(lp)
    c = np.asarray(lp.c, dtype=np.float64)
    c_q = [Fraction(float(v)) for v in c]
    n = c.size
    lp_status, tab, lp_pivots = _q_two_phase(sf, _LP_MAX_PIVOTS)
    # Column j of the standard form in the original variables: expr_const[j] + expr_coef[j]ᵀx.
    expr_const = [Fraction(int(v)) for v in sf.expr_const]
    expr_coef = [[Fraction(int(v)) for v in row] for row in sf.expr_coef]
    cuts: list[dict[str, Any]] = []
    active: list[bool] = []  # active[q]: cut q still has a row in the tableau
    trace: list[Step] = []
    total_dual = 0
    n_std = tab.N  # columns ≥ n_std are cut slacks g1, g2, ...

    def drop_basic_cuts() -> None:
        """Delete the row and column of every cut whose slack is basic (docstring NOTE)."""
        for i in sorted((i for i, j in enumerate(tab.basis) if j >= n_std), reverse=True):
            j = tab.basis[i]
            active[int(tab.labels[j][1:]) - 1] = False
            del tab.T[1 + i]
            del tab.basis[i]
            for t in tab.T:
                del t[j]
            tab.basis = [jb - 1 if jb > j else jb for jb in tab.basis]
            del tab.labels[j], expr_const[j], expr_coef[j]

    def emit(k: int, cut: dict[str, Any] | None, dual_pivots: int) -> None:
        z = tab.values()[:n]
        x = np.array([float(v) for v in z])
        fun = float(sum((cj * zj for cj, zj in zip(c_q, z, strict=True)), Fraction(0)))
        trace.append(
            Step(
                k,
                x,
                fun,
                info={
                    **tab.snapshot(),
                    "vertex": x,
                    "cuts": [
                        {"coef": q["coef"], "rhs": q["rhs"], "active": a}
                        for q, a in zip(cuts, active, strict=True)
                    ],
                    "cut": cut,
                    "dual_pivots": dual_pivots,
                },
            )
        )

    emit(0, None, 0)
    if lp_status != "optimal":
        status = lp_status if lp_status in ("infeasible", "unbounded") else "lp_failure"
        messages = {
            "infeasible": "LP relaxation is infeasible: phase-1 minimum of the artificial sum > 0",
            "unbounded": "LP relaxation is unbounded: the ILP is unbounded or infeasible",
        }
        message = messages.get(status, f"LP relaxation not solved: {lp_status}")
        return Result(
            "gomory_cuts",
            trace[0].x,
            trace[0].fun,
            False,
            message,
            0,
            trace=trace,
            extra={"status": status, "cuts": [], "dual_pivots": 0, "lp_pivots": lp_pivots},
        )

    status, message = "max_iter", f"reached max_iter={max_iter} cuts"
    k = 0
    while True:
        f_rhs = [_frac(tab.T[1 + i][-1]) for i in range(tab.m)]
        if all(f == 0 for f in f_rhs):
            status, message = "optimal", "optimal: every basic value of the LP optimum is integral"
            break
        if k >= max_iter:
            break
        r = max(range(tab.m), key=lambda i: (f_rhs[i], -i))
        f0 = f_rhs[r]
        row = tab.T[1 + r]
        basic = set(tab.basis)
        f = [Fraction(0) if j in basic else _frac(row[j]) for j in range(tab.N)]
        # Cut in the original variables: Σ f_j (e0_j + e_jᵀx) ≥ f0  ⇔  coefᵀx ≤ rhs.
        coef_x = [
            -sum((f[j] * expr_coef[j][i] for j in range(tab.N)), Fraction(0)) for i in range(n)
        ]
        rhs_x = sum((f[j] * expr_const[j] for j in range(tab.N)), Fraction(0)) - f0
        cut = {
            "coef": [float(v) for v in coef_x],
            "rhs": float(rhs_x),
            "source_row": 1 + r,
            "source_var": tab.labels[tab.basis[r]],
            "f0": float(f0),
            "f": [float(v) for v in f],
        }
        # The slack g = Σ f_j z_j − f0 = rhs_x − coef_xᵀx in the original variables.
        expr_const.append(rhs_x)
        expr_coef.append([-v for v in coef_x])
        # Append column g and the row −f·z + g = −f0 (g is basic in it).
        tab.T = [[*t[:-1], Fraction(0), t[-1]] for t in tab.T]
        tab.T.append([*(-v for v in f), Fraction(1), -f0])
        tab.labels = [*tab.labels, f"g{len(cuts) + 1}"]
        tab.basis = [*tab.basis, tab.N - 1]
        cuts.append(cut)
        active.append(True)
        st, dual_pivots = _q_dual(tab, _LP_MAX_PIVOTS)
        total_dual += dual_pivots
        if st == "optimal" and tab.N - n_std > _MAX_CUT_ROWS:
            drop_basic_cuts()
        k += 1
        emit(k, cut, dual_pivots)
        if st == "infeasible":
            status = "infeasible"
            message = (
                "ILP is infeasible: after a valid cut the LP relaxation is infeasible "
                "(exact dual simplex)"
            )
            break
        if st != "optimal":
            status, message = "lp_failure", f"dual simplex stopped ({st}) after a cut"
            break

    final = trace[-1]
    return Result(
        method="gomory_cuts",
        x=np.asarray(final.x, dtype=np.float64),
        fun=final.fun,
        converged=status == "optimal",
        message=message,
        n_iter=final.k,
        trace=trace,
        extra={
            "status": status,
            "cuts": [{**q, "active": a} for q, a in zip(cuts, active, strict=True)],
            "dual_pivots": total_dual,
            "lp_pivots": lp_pivots,
        },
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("branch_and_bound", "ilp_knapsack_like_2d", {}),
    ("branch_and_bound", "ilp_knapsack_like_2d", {"strategy": "depth_first"}),
    ("branch_and_bound", "ilp_3var", {}),
    ("gomory_cuts", "ilp_knapsack_like_2d", {}),
    ("gomory_cuts", "ilp_3var", {}),
]
