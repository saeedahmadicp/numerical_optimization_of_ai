"""Simplex methods for linear programs.

Every :class:`~numopt.core.types.LinearProgram`

    min/max cᵀx   s.t.   A_ub x ≤ b_ub,   A_eq x = b_eq,   x ≥ 0

is solved in the *minimization standard form*

    min c̃ᵀz   s.t.   A z = b,   z ≥ 0,

with c̃ = c for ``sense="min"`` and c̃ = −c for ``sense="max"`` (the reported objective is
always cᵀx in the original sense). The columns of ``A`` are the original variables
x₁…xₙ followed by one slack sᵢ = b_ub,ᵢ − a_ub,ᵢᵀx ≥ 0 per inequality row. Rows with a
negative right-hand side are multiplied by −1 so that b ≥ 0 (the slack variable keeps its
meaning; it then has coefficient −1 in its row, i.e. it acts as a surplus variable). A row
without a +1 slack column (a flipped row or an equality row) gets an artificial variable
``a<i>`` in phase 1 / big-M.

Tableau layout (Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, 1997, §3.3,
"full tableau implementation", with the right-hand side stored as the last column)::

    row 0          [ c̄ᵀ     | −z̃  ]     c̄ = c̃ − (B⁻¹A)ᵀc̃_B  (reduced costs),  z̃ = c̃_Bᵀx_B
    rows 1 … m     [ B⁻¹A   | x_B ]

Because row 0 is stored as ``[c̄ | −z̃]``, one Gauss–Jordan pivot updates every row,
row 0 included. The symbolic big-M tableau has two objective rows: row 0 holds the
coefficients of M and row 1 the constant parts (Hillier & Lieberman write the same entries
as "−4M − 3"); the constraint rows start at row 2. A pivot on ``(r, j)`` makes column j a
unit vector; ``# NOTE:`` the pivot column is then set to that unit vector exactly, which
removes rounding residue but does not change the mathematics.

Pivot rules (entering variable among the columns with c̄ⱼ < −tol·σⱼ·γ, see "Pivot
tolerance" below):

* ``dantzig`` — most negative reduced cost (Dantzig 1947).
* ``bland`` — smallest column index (Bland 1977); it never cycles.
* ``steepest_edge`` — most negative c̄ⱼ / ‖ηⱼ‖, where ηⱼ = (−B⁻¹Aⱼ, eⱼ) is the edge
  direction in the full z-space, ‖ηⱼ‖² = 1 + ‖B⁻¹Aⱼ‖² (Goldfarb & Reid 1977; computed
  exactly from the tableau column here, not with the Goldfarb–Reid update formulas).

Ties in the entering choice (within ``tol``) go to the smallest column index. The leaving row
is chosen by the minimum-ratio test θ = min{x_B,i / ūᵢ : ūᵢ > τᵢⱼ}, ū = B⁻¹Aⱼ; ties
(ratios ≤ θ·(1 + tol), see :func:`min_ratio`) go to the basic variable with the smallest
column index for every rule (this is
Bland's leaving rule; with Dantzig entering it is the rule under which Beale's example
cycles). Because every choice depends only on the *set* of basic variables, a repeated basis
means the method would loop for ever; it then stops with ``converged=False`` and a
"cycling detected" message.

Pivot tolerance. ``# NOTE:`` an entry ā_ij of B⁻¹A counts as nonzero when
|ā_ij| > τᵢⱼ = tol·σⱼ/σ_B(i), not when |ā_ij| > tol. Here σₖ = maxᵢ |A_ik|/ρᵢ is the size
of column k after each row is divided by ρᵢ = maxⱼ≤ₙ |A_ij| (its largest entry on the
original variables; 1 for a row without one). Scaling a row of the data or a variable
changes ā_ij by exactly the factor σⱼ/σ_B(i), so the test is invariant under row and column
scaling: it is the absolute test ``tol`` applied to the equilibrated problem. An absolute
test would ignore a legitimate entry such as the 1e-9 of the row 1e-9·x₁ ≤ 1 and report a
bounded LP as unbounded, with a "ray" that is not a recession direction. The same τ is used
by the dual ratio test and by the phase-1 drive-out pivots. For the same reason a column
enters only when c̄ⱼ < −tol·σⱼ·γ, with γ = maxₖ |costₖ| of the costs in that objective row
(γ = 1 for the phase-1 row and the M row of big-M): c̄ⱼ scales like σⱼ under row and column
scaling and like γ under scaling of c.

Phase 1 and big-M on the equilibrated rows. ``# NOTE:`` the artificial aᵢ of row i has the
phase-1 cost (big-M: M-row cost) wᵢ = 1/ρᵢ, not the textbook 1. Row i divided by ρᵢ has the
artificial âᵢ = aᵢ/ρᵢ with cost 1, so this is the textbook phase 1 of the row-equilibrated LP
(Golub & Van Loan, 4th ed., §3.5.2), with the constraint rows displayed unscaled. Here ρᵢ is
the ρᵢ above, or |bᵢ| for a row with no entry on x. With cost 1 the phase-1 reduced costs
change with the row scaling: the row 1e-6·x₁ ≥ 1e-6 gave c̄ⱼ = −1e-6 > −tol, so phase 1
stopped with the artificial at 1e-6 (x₁ = 5e-5 instead of x₁ ≥ 1). The LP is declared
feasible only when every artificial passes the per-row test aᵢ ≤ capᵢ = tol·(bᵢ + Σⱼ|A_ij|zⱼ)
at the phase-1 optimum z (sum over the real columns). That is the Oettli–Prager
componentwise residual test: the row residual is at most tol times the size of the terms
of the row (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Thm 7.3).
The test does not change under row scaling or under column scaling. A test on Σaᵢ against
10·tol·(1 + ‖b‖∞) let the largest row set the tolerance of every row and accepted
violations of 0.05 at tol = 1e-4; a test aᵢ ≤ tol·(ρᵢ + bᵢ) still accepts a large
violation when a column is large (x₄ with coefficient 6e5 makes ρᵢ = 6e5). An accepted
artificial that is still basic is set to 0 before it is driven out. This moves bᵢ by at
most capᵢ, and every row then holds up to that amount. The drive-out pivot is then
degenerate (θ = 0) and keeps x ≥ 0, also on a negative pivot entry. The dual simplex
leaves on x_B,i < −tol·(|B⁻¹|·|b|)ᵢ, the same componentwise test for x_B = B⁻¹b (its
componentwise scale; Higham, 2nd ed., §7.2).

Statuses (``Result.extra["status"]``): ``"optimal"`` (``converged=True``), ``"unbounded"``,
``"infeasible"``, ``"cycling"``, ``"max_iter"``, ``"singular"`` (revised simplex only); all
but ``"optimal"`` give ``converged=False``. ``Result.x`` holds the original variables of the
last basic solution, ``Result.fun`` = cᵀx in the original sense. ``n_iter`` is the number of
pivots; ``n_fev`` is 0 (no function evaluations).

Trace semantics: Step k shows the state after k pivots. The keys ``entering``, ``leaving``,
``pivot_row`` and ``ratio_test`` describe the pivot *chosen on this tableau* (the one that
produces Step k + 1); they are ``None`` on the final Step when no pivot follows.
``Step.step_size`` is the step length θ of the pivot that *produced* this Step (``None``
for k = 0). ``Step.x`` is the vertex (original variables), ``Step.fun`` = cᵀx at the vertex
(in phase 1 the vertex can violate the constraints that still have positive artificials).

Info keys:
    tableau: [[float]] — full tableau, shape (n_obj + m) × (N + 1), layout above
        (``revised_simplex``: B⁻¹[A | b] and its reduced-cost row, for display only).
    row_labels: [str] — ``"z"`` (``"zM"``, ``"z"`` for big-M), then the basic variable name of
        each constraint row.
    col_labels: [str] — variable names (``x1…xn``, ``s1…`` slack of inequality row i,
        ``a1…`` artificial of row i, ``g1…`` Gomory cut slacks), then ``"rhs"``.
    basis: [int] — column index of the basic variable of each constraint row.
    nonbasis: [int] — column indices of the nonbasic variables, increasing.
    entering: int | None — column chosen to enter on this tableau.
    leaving: int | None — column index of the variable chosen to leave.
    pivot_row: int | None — tableau row index of the pivot (includes the objective rows).
    ratio_test: [float | None] — one entry per constraint row: x_B,i / ūᵢ for ūᵢ > τᵢⱼ,
        else ``None`` (``dual_simplex``: one entry per column, |c̄ⱼ / ā_rⱼ| for ā_rⱼ < −τ_rⱼ;
        τ is the scaled pivot tolerance of the module docstring).
    vertex: [float] — original variables x of the current basic solution.
    objective: float — cᵀx in the original sense (same as ``Step.fun``).
    phase: int — 1 = feasibility phase of the two-phase method, 2 = optimization phase
        (also the single phase of ``simplex``, ``big_m`` and ``dual_simplex``).
    infeasibility: float — Σ aᵢ/ρᵢ, the phase-1 objective (big-M: the M-part of the
        objective), i.e. the sum of the artificials of the equilibrated rows (0 otherwise).
    drive_out: bool — True when the chosen pivot removes a zero-level artificial variable from
        the basis after phase 1 (Bertsimas & Tsitsiklis §3.5); the level is set to exactly 0
        first (see "Phase 1 and big-M"), so the pivot has θ = 0.
    removed_rows: [int] — standard-form rows dropped as redundant at the start of phase 2
        (present on the first phase-2 Step of the two-phase methods only).
    cycling: bool — present (True) on the final Step when its basis repeated an earlier one.
    ray: [float] — (unbounded only) direction d of the original variables such that
        vertex + t·d is feasible for all t ≥ 0 and cᵀd improves the objective.
    primal_infeasibility: float — (``dual_simplex``) Σ max(0, −x_B,i) (unscaled).
    duals: [float] — (``revised_simplex``) simplex multipliers y = B⁻ᵀc̃_B of the min form.
    reduced_costs: [float] — (``revised_simplex``) c̄ = c̃ − Aᵀy, one entry per column.
    direction: [float] | None — (``revised_simplex``) ū = B⁻¹A_q for the entering column q.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import LinearProgram, Result, Step

Matrix = np.ndarray
_PIVOT_RULES = ("dantzig", "bland", "steepest_edge")

PIVOT_PARAMS = (
    ParamSpec(
        "pivot_rule",
        "dantzig",
        kind="choice",
        choices=_PIVOT_RULES,
        help="Entering-variable rule: most negative reduced cost, smallest index, or steepest edge.",
    ),
    ParamSpec(
        "tol",
        1e-9,
        min=1e-14,
        max=1e-4,
        log=True,
        help="Optimality / pivot tolerance on the equilibrated problem: c̄ⱼ < −tol·σⱼ·γ enters; ūᵢ > tol·σⱼ/σ_B(i) takes part in the ratio test.",
    ),
    ParamSpec("max_iter", 200, kind="int", min=1, max=10_000, help="Maximum number of pivots."),
)

REFERENCES = (
    "Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §3.3 (full tableau) and §3.5",
    "Chvátal, Linear Programming (1983), ch. 2–3",
    "Bland (1977), Math. Oper. Res. 2(2):103–107",
)


# --------------------------------------------------------------------------------------
# Standard form
# --------------------------------------------------------------------------------------


@dataclass
class StandardForm:
    """``min c̃ᵀz s.t. A z = b, z ≥ 0`` built from a LinearProgram.

    ``expr_const[j] + expr_coef[j] @ x`` expresses column j in the original variables x
    (x_j itself, or a slack b_i − a_iᵀx); Gomory cuts use it to map cuts back to x.
    """

    A: Matrix  # (m, N)
    b: np.ndarray  # (m,)
    c: np.ndarray  # (N,)  min-form costs
    labels: list[str]  # (N,)
    n: int  # number of original variables
    sign: float  # +1 for min, −1 for max: cᵀx = sign · c̃ᵀz
    unit_col: list[int | None]  # per row: a column with +1 in this row and 0 elsewhere
    expr_const: np.ndarray  # (N,)
    expr_coef: Matrix  # (N, n)


def column_scales(A: Matrix, n: int) -> np.ndarray:
    """σₖ = maxᵢ |A_ik| / ρᵢ with ρᵢ = max_{j<n} |A_ij| (row equilibration on the original
    variables; ρᵢ = 1 for a row with no such entry); σₖ = 1 for a zero column.

    The pivot tolerance of an entry ā_ij of B⁻¹A is tol·σⱼ/σ_B(i) (module docstring).
    """
    m, N = A.shape
    if m == 0:
        return np.ones(N)
    rho = np.max(np.abs(A[:, :n]), axis=1)
    rho = np.where(rho > 0.0, rho, 1.0)
    sigma = np.max(np.abs(A) / rho[:, None], axis=0)
    return np.where(sigma > 0.0, sigma, 1.0)


def row_scales(sf: StandardForm) -> np.ndarray:
    """ρᵢ = max_{j<n} |A_ij| per standard-form row; |bᵢ| for a row with no entry on x (1 if
    bᵢ = 0 too). Same ρ as :func:`numopt.lp.interior_point.scaled_form`."""
    if sf.b.size == 0:
        return np.ones(0)
    rho = np.max(np.abs(sf.A[:, : sf.n]), axis=1)
    tiny = np.finfo(np.float64).tiny  # a subnormal bᵢ would overflow 1/ρᵢ
    return np.where(rho > 0.0, rho, np.where(np.abs(sf.b) >= tiny, np.abs(sf.b), 1.0))


def artificial_weights(sf: StandardForm, n_cols: int) -> np.ndarray:
    """Phase-1 / M-row cost per column of :func:`_initial_tableau`: wᵢ = 1/ρᵢ on the
    artificial of row i (column N + k for the k-th row without a unit column), else 0."""
    N = sf.A.shape[1]
    rho = row_scales(sf)
    w = np.zeros(n_cols)
    art_rows = [i for i in range(sf.b.size) if sf.unit_col[i] is None]
    for k, i in enumerate(art_rows):
        w[N + k] = 1.0 / rho[i]
    return w


def artificial_caps(sf: StandardForm, z: np.ndarray, n_cols: int, tol: float) -> np.ndarray:
    """Accepted level capᵢ = tol·(bᵢ + Σⱼ |A_ij|·|zⱼ|) of the artificial of row i at the
    point z of the N real columns (0 on other columns); module docstring, "Phase 1"."""
    N = sf.A.shape[1]
    cap = np.zeros(n_cols)
    art_rows = [i for i in range(sf.b.size) if sf.unit_col[i] is None]
    for k, i in enumerate(art_rows):
        cap[N + k] = tol * (abs(sf.b[i]) + float(np.abs(sf.A[i]) @ np.abs(z[:N])))
    return cap


def artificial_violation(
    basis: Sequence[int], x_b: np.ndarray, cap: np.ndarray, artificial: set[int]
) -> tuple[int, float] | None:
    """First basic artificial whose level exceeds its accepted level, as (column, level)."""
    for i, j in enumerate(basis):
        if j in artificial and x_b[i] > cap[j]:
            return j, float(x_b[i])
    return None


def check_lp(lp: Any) -> LinearProgram:
    """Validate shapes and finiteness; return the LinearProgram (raise on invalid input)."""
    if not isinstance(lp, LinearProgram):
        raise TypeError("problem must be a numopt LinearProgram")
    c = np.asarray(lp.c, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError(f"{lp.id}: c must be a non-empty vector")
    n = c.size
    for name_a, name_b in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        a, b = getattr(lp, name_a), getattr(lp, name_b)
        if (a is None) != (b is None):
            raise ValueError(f"{lp.id}: {name_a} and {name_b} must be given together")
        if a is None:
            continue
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != n or b.shape != (a.shape[0],):
            raise ValueError(f"{lp.id}: {name_a} must be (m, {n}) and {name_b} must be (m,)")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError(f"{lp.id}: {name_a}/{name_b} contain non-finite values")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"{lp.id}: c contains non-finite values")
    if lp.sense not in ("min", "max"):
        raise ValueError(f"{lp.id}: sense must be 'min' or 'max'")
    return lp


def lp_arrays(
    lp: LinearProgram,
) -> tuple[np.ndarray, Matrix, np.ndarray, Matrix, np.ndarray]:
    """Return ``c, A_ub, b_ub, A_eq, b_eq`` as float arrays (empty (0, n) when absent)."""
    c = np.asarray(lp.c, dtype=np.float64).copy()
    n = c.size
    a_ub = np.zeros((0, n)) if lp.A_ub is None else np.asarray(lp.A_ub, dtype=np.float64)
    b_ub = np.zeros(0) if lp.b_ub is None else np.asarray(lp.b_ub, dtype=np.float64)
    a_eq = np.zeros((0, n)) if lp.A_eq is None else np.asarray(lp.A_eq, dtype=np.float64)
    b_eq = np.zeros(0) if lp.b_eq is None else np.asarray(lp.b_eq, dtype=np.float64)
    return c, a_ub.reshape(-1, n), b_ub.reshape(-1), a_eq.reshape(-1, n), b_eq.reshape(-1)


def standard_form(lp: LinearProgram) -> StandardForm:
    """Build the b ≥ 0 standard form (slack per ≤ row, rows with b < 0 multiplied by −1)."""
    c, a_ub, b_ub, a_eq, b_eq = lp_arrays(lp)
    n, m_ub, m_eq = c.size, b_ub.size, b_eq.size
    m, big_n = m_ub + m_eq, n + m_ub
    A = np.zeros((m, big_n))
    A[:m_ub, :n] = a_ub
    A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = a_eq
    b = np.concatenate([b_ub, b_eq])
    flip = b < 0
    A[flip] *= -1.0
    b[flip] *= -1.0
    sign = 1.0 if lp.sense == "min" else -1.0
    c_std = np.concatenate([sign * c, np.zeros(m_ub)])
    unit_col: list[int | None] = [None if (i >= m_ub or flip[i]) else n + i for i in range(m)]
    expr_const = np.concatenate([np.zeros(n), b_ub])
    expr_coef = np.vstack([np.eye(n), -a_ub])
    labels = [f"x{j + 1}" for j in range(n)] + [f"s{i + 1}" for i in range(m_ub)]
    return StandardForm(A, b, c_std, labels, n, sign, unit_col, expr_const, expr_coef)


# --------------------------------------------------------------------------------------
# Tableau engine
# --------------------------------------------------------------------------------------


@dataclass
class Tableau:
    """A simplex tableau: ``T`` has ``n_obj`` objective rows, then one row per constraint."""

    T: Matrix
    basis: list[int]
    labels: list[str]
    n_obj: int = 1
    artificial: set[int] = field(default_factory=set)
    #: σ per column (:func:`column_scales`); None means σ = 1 (an absolute pivot tolerance).
    scale: np.ndarray | None = None
    #: γ per objective row: maxⱼ |costⱼ| of the costs installed in that row (1 if all zero).
    cost_scale: tuple[float, ...] = (1.0, 1.0)
    #: Phase-1 / M-row cost per column: wᵢ = 1/ρᵢ on the artificial of row i, else 0.
    art_cost: np.ndarray | None = None

    @property
    def m(self) -> int:
        return self.T.shape[0] - self.n_obj

    @property
    def N(self) -> int:
        return self.T.shape[1] - 1

    def rows(self) -> Matrix:
        return self.T[self.n_obj :]

    def values(self) -> np.ndarray:
        """Values of all N columns at the current basic solution."""
        z = np.zeros(self.N)
        z[self.basis] = self.T[self.n_obj :, -1]
        return z

    def cost_tol(self, row: int, tol: float) -> np.ndarray:
        """Entering threshold tol·σⱼ·γ for every column j of objective ``row``."""
        sigma = np.ones(self.N) if self.scale is None else self.scale[: self.N]
        return tol * sigma * self.cost_scale[row]

    def pivot_tol(self, j: int, tol: float) -> np.ndarray:
        """τᵢⱼ = tol·σⱼ/σ_B(i) for every constraint row i (module docstring, "Pivot tolerance")."""
        if self.scale is None:
            return np.full(self.m, tol)
        return tol * self.scale[j] / self.scale[self.basis]

    def pivot(self, i: int, j: int) -> None:
        """Gauss–Jordan pivot on constraint row ``i`` (0-based) and column ``j``."""
        r = self.n_obj + i
        self.T[r] /= self.T[r, j]
        col = self.T[:, j].copy()
        col[r] = 0.0
        self.T -= np.outer(col, self.T[r])
        # NOTE: the pivot column is a unit vector in exact arithmetic; store it exactly.
        self.T[:, j] = 0.0
        self.T[r, j] = 1.0
        self.basis[i] = j

    def snapshot(self) -> dict[str, Any]:
        obj_labels = ["z"] if self.n_obj == 1 else ["zM", "z"]
        basic = set(self.basis)
        return {
            "tableau": self.T.copy(),
            "row_labels": obj_labels + [self.labels[j] for j in self.basis],
            "col_labels": [*self.labels, "rhs"],
            "basis": list(self.basis),
            "nonbasis": [j for j in range(self.N) if j not in basic],
        }


def choose_entering(tab: Tableau, rule: str, tol: float, allowed: np.ndarray) -> int | None:
    """Entering column by ``rule`` among ``allowed`` columns with c̄ⱼ < −tol·σⱼ·γ (None: optimal).

    For the symbolic big-M tableau (two objective rows) the reduced cost c̄ⱼ = Mⱼ·M + cⱼ is
    compared lexicographically: columns with Mⱼ < −τⱼ come first; only when there are none
    do columns with |Mⱼ| ≤ τⱼ and cⱼ < −τ'ⱼ compete (τ, τ' the thresholds of the two rows).
    """
    N = tab.N
    if tab.n_obj == 2:
        d_m, d_c = tab.T[0, :N], tab.T[1, :N]
        thr_m, thr_c = tab.cost_tol(0, tol), tab.cost_tol(1, tol)
        eligible = allowed & (d_m < -thr_m)
        d = d_m
        if not eligible.any():
            eligible = allowed & (np.abs(d_m) <= thr_m) & (d_c < -thr_c)
            d = d_c
    else:
        d = tab.T[0, :N]
        eligible = allowed & (d < -tab.cost_tol(0, tol))
    cand = np.flatnonzero(eligible)
    if cand.size == 0:
        return None
    if rule == "bland":
        return int(cand[0])
    score = d[cand]
    if rule == "steepest_edge":
        score = score / np.sqrt(1.0 + np.sum(tab.rows()[:, cand] ** 2, axis=0))
    best = float(score.min())
    ties = cand[score <= best + tol * (1.0 + abs(best))]
    return int(ties[0])


def ratio_test(tab: Tableau, j: int, tol: float) -> tuple[int | None, list[float | None], float]:
    """Minimum-ratio test on column ``j``: (row, ratios, θ); row is None when unbounded."""
    col, rhs = tab.rows()[:, j], tab.rows()[:, -1]
    leave, theta, ratios = min_ratio(rhs, col, tol, tab.basis, tab.pivot_tol(j, tol))
    return leave, ratios, theta


def min_ratio(
    x_b: np.ndarray,
    u: np.ndarray,
    tol: float,
    basis: Sequence[int],
    pivot_tol: np.ndarray,
) -> tuple[int | None, float, list[float | None]]:
    """θ = min{x_B,i / uᵢ : uᵢ > τᵢ}; ties go to the smallest basic index.

    ``pivot_tol`` holds τᵢ = tol·σⱼ/σ_B(i) per row. A ratio rᵢ ties with θ when
    rᵢ ≤ θ·(1 + tol). ``# NOTE:`` the tie window is relative: choosing a tied row k instead
    of the minimizing row i sets x_B,i to x_B,i − r_k·uᵢ ≥ −tol·x_B,i, at most tol times its
    value before the pivot, in any units. The absolute window rᵢ ≤ θ + tol·(1 + θ) is in the
    units of the entering variable: for the slack of a row scaled by 1e-9 every ratio below
    1e-9 was a tie, Bland's rule chose a row with twice the minimum ratio, and a basic
    variable became negative (row violated by 1.5 after unscaling). Only exact zeros tie with
    θ = 0. Returns ``(row, θ, ratios)`` with ``row = None`` (θ = ∞) when no uᵢ > τᵢ.
    """
    mask = u > pivot_tol
    ratios = np.full(u.size, np.nan)
    # NOTE: a basic value of −1e-17 (rounding) is treated as 0, so θ ≥ 0.
    ratios[mask] = np.maximum(x_b[mask], 0.0) / u[mask]
    listed = [None if np.isnan(v) else float(v) for v in ratios]
    if not mask.any():
        return None, float("inf"), listed
    theta = float(np.min(ratios[mask]))
    ties = np.flatnonzero(mask & (ratios <= theta * (1.0 + tol)))
    leave = min((int(i) for i in ties), key=lambda i: basis[i])
    return leave, theta, listed


def ray_direction(tab: Tableau, j: int, n: int) -> np.ndarray:
    """Original-variable part of the edge direction η (η_B = −B⁻¹Aⱼ, ηⱼ = 1)."""
    eta = np.zeros(tab.N)
    eta[j] = 1.0
    eta[tab.basis] = -tab.rows()[:, j]
    return eta[:n]


class Recorder:
    """Collects Steps; ``k`` is the number of pivots done so far."""

    def __init__(self, sf: StandardForm, max_iter: int, record: bool = True) -> None:
        self.sf = sf
        self.max_iter = max_iter
        self.record = record
        self.k = 0
        self.trace: list[Step] = []
        self.last_theta: float | None = None
        self.unbounded_col: int | None = None
        self.ray: np.ndarray | None = None

    def objective(self, x: np.ndarray) -> float:
        return float(self.sf.sign * (self.sf.c[: self.sf.n] @ x))

    def emit(self, tab: Tableau, phase: int, **info: Any) -> None:
        if not self.record:
            return
        z = tab.values()
        x = z[: self.sf.n].copy()
        fun = self.objective(x)
        art = sorted(tab.artificial)
        payload = {
            **tab.snapshot(),
            "entering": None,
            "leaving": None,
            "pivot_row": None,
            "ratio_test": None,
            "vertex": x,
            "objective": fun,
            "phase": phase,
            "infeasibility": (
                float(z[art] @ tab.art_cost[art]) if art and tab.art_cost is not None else 0.0
            ),
        }
        payload.update(info)
        self.trace.append(Step(self.k, x, fun, step_size=self.last_theta, info=payload))


def _pivot_info(tab: Tableau, j: int, i: int | None, ratios: Sequence[float | None]) -> dict:
    return {
        "entering": j,
        "leaving": None if i is None else tab.basis[i],
        "pivot_row": None if i is None else tab.n_obj + i,
        "ratio_test": list(ratios),
    }


def primal_loop(
    tab: Tableau,
    rec: Recorder,
    *,
    rule: str,
    tol: float,
    phase: int,
    allowed: np.ndarray,
    emit_optimal: bool = True,
    first_info: dict[str, Any] | None = None,
    after_pivot: Callable[[], None] | None = None,
) -> str:
    """Run primal simplex pivots until optimal / unbounded / cycling / max_iter.

    ``first_info`` is merged into the first Step this call emits (phase-switch details);
    ``after_pivot`` is called after every pivot (big-M recomputes its M row there).
    """
    seen: set[frozenset[int]] = set()
    pending = dict(first_info or {})

    def emit(**info: Any) -> None:
        rec.emit(tab, phase, **info, **pending)
        pending.clear()

    while True:
        key = frozenset(tab.basis)
        if key in seen:
            emit(cycling=True)
            return "cycling"
        seen.add(key)
        j = choose_entering(tab, rule, tol, allowed)
        if j is None:
            if emit_optimal:
                emit()
            return "optimal"
        i, ratios, theta = ratio_test(tab, j, tol)
        info = _pivot_info(tab, j, i, ratios)
        if i is None:
            rec.unbounded_col = j
            rec.ray = ray_direction(tab, j, rec.sf.n)
            emit(**info, ray=rec.ray)
            return "unbounded"
        emit(**info)
        if rec.k >= rec.max_iter:
            return "max_iter"
        tab.pivot(i, j)
        if after_pivot is not None:
            after_pivot()
        rec.k += 1
        rec.last_theta = theta


def cost_scale(cost: np.ndarray) -> float:
    """γ = maxⱼ |costⱼ| (1 when every cost is 0)."""
    g = float(np.max(np.abs(cost))) if cost.size else 0.0
    return g if g > 0.0 else 1.0


def _objective_row(tab: Tableau, cost: np.ndarray) -> np.ndarray:
    """Row ``[c̄ | −z̃]`` for costs ``cost`` (length N) and the current basis."""
    rows = tab.rows()
    c_b = cost[tab.basis]
    row = np.empty(tab.N + 1)
    row[:-1] = cost - c_b @ rows[:, :-1]
    row[-1] = -(c_b @ rows[:, -1])
    row[tab.basis] = 0.0  # NOTE: exact zeros for basic columns (c̄_B = 0 by definition)
    return row


@dataclass
class LPOutcome:
    """Result of an LP solve by the tableau engine (used by the integer methods too)."""

    status: str
    message: str
    tab: Tableau | None
    x: np.ndarray
    value: float | None  # cᵀx in the original sense (None unless optimal)
    pivots: int
    trace: list[Step]
    sf: StandardForm
    ray: np.ndarray | None = None


def _initial_tableau(sf: StandardForm, n_obj: int = 1) -> Tableau:
    """Tableau with the slack / artificial starting basis (artificials appended last)."""
    m, N = sf.A.shape
    art_rows = [i for i in range(m) if sf.unit_col[i] is None]
    n_art = len(art_rows)
    labels = sf.labels + [f"a{i + 1}" for i in art_rows]
    A = np.zeros((m, N + n_art))
    A[:, :N] = sf.A
    basis: list[int] = []
    for i in range(m):
        uc = sf.unit_col[i]
        basis.append(uc if uc is not None else N + art_rows.index(i))
    for k, i in enumerate(art_rows):
        A[i, N + k] = 1.0
    T = np.zeros((n_obj + m, N + n_art + 1))
    T[n_obj:, :-1] = A
    T[n_obj:, -1] = sf.b
    tab = Tableau(T, basis, labels, n_obj, set(range(N, N + n_art)), column_scales(A, sf.n))
    tab.art_cost = artificial_weights(sf, N + n_art)
    return tab


def _drive_out_column(rel: np.ndarray, tol: float) -> int:
    """Drive-out pivot column: the largest |ā_ij|/τᵢⱼ, ties (relative, within tol) to the
    smallest index, so that rounding does not decide between equal entries."""
    best = float(np.max(rel))
    return int(np.flatnonzero(rel >= best / (1.0 + tol))[0])


def _infeasible_message(labels: Sequence[str], col: int, level: float, cap: float) -> str:
    return (
        f"LP is infeasible: the minimum of Σ aᵢ/ρᵢ leaves the artificial {labels[col]} at "
        f"{level:.6g} > tol·(bᵢ + Σⱼ|A_ij|zⱼ) = {cap:.3g} (its row is violated)"
    )


def solve_two_phase(
    sf: StandardForm,
    *,
    rule: str,
    tol: float,
    max_iter: int,
    record: bool = True,
) -> LPOutcome:
    """Two-phase tableau simplex (Bertsimas & Tsitsiklis §3.5) on a b ≥ 0 standard form."""
    rec = Recorder(sf, max_iter, record)
    tab = _initial_tableau(sf)
    N = sf.A.shape[1]
    n = sf.n

    def done(status: str, message: str, ray: np.ndarray | None = None) -> LPOutcome:
        x = tab.values()[:n]
        value = rec.objective(x) if status == "optimal" else None
        return LPOutcome(status, message, tab, x, value, rec.k, rec.trace, sf, ray)

    if tab.artificial:
        # Phase 1: min Σ aᵢ/ρᵢ. Row 0 = [−Σ_{artificial rows} Aᵢ/ρᵢ, 0 | −Σ bᵢ/ρᵢ].
        w_cost = artificial_weights(sf, tab.N)
        tab.T[0] = _objective_row(tab, w_cost)
        tab.cost_scale = (1.0,)
        allowed = np.ones(tab.N, dtype=bool)
        status = primal_loop(
            tab, rec, rule=rule, tol=tol, phase=1, allowed=allowed, emit_optimal=False
        )
        if status == "max_iter":
            return done("max_iter", f"reached max_iter={max_iter} pivots in phase 1")
        if status == "cycling":
            return done("cycling", "cycling detected in phase 1: a basis repeated")
        # Phase 1 is bounded below by 0, so "unbounded" cannot occur here.
        cap = artificial_caps(sf, tab.values(), tab.N, tol)
        bad = artificial_violation(tab.basis, tab.rows()[:, -1], cap, tab.artificial)
        if bad is not None:
            rec.emit(tab, 1)
            return done("infeasible", _infeasible_message(tab.labels, bad[0], bad[1], cap[bad[0]]))
        # Drive zero-level artificials out of the basis; drop redundant rows.
        removed: list[int] = []
        art_rows = [r for r in range(sf.b.size) if sf.unit_col[r] is None]
        i = 0
        while i < tab.m:
            if tab.basis[i] not in tab.artificial:
                i += 1
                continue
            # NOTE: the level is ≤ capᵢ (accepted above); set it to exactly 0. This
            # changes b of the artificial's row by the level and no other basic value
            # (B⁻¹ of the artificial's column is the unit vector of row i), so the pivot
            # below is degenerate (θ = 0) and keeps x ≥ 0 on a negative pivot entry too.
            level = float(tab.T[tab.n_obj + i, -1])
            tab.T[tab.n_obj + i, -1] = 0.0
            tab.T[0, -1] += w_cost[tab.basis[i]] * level
            # Largest entry relative to its pivot tolerance τᵢⱼ (module docstring).
            row = tab.rows()[i, :N]
            ratio = np.ones(N) if tab.scale is None else tab.scale[:N] / tab.scale[tab.basis[i]]
            rel = np.abs(row) / (tol * ratio)
            j = _drive_out_column(rel, tol)
            if rel[j] > 1.0:
                rec.emit(
                    tab,
                    1,
                    entering=j,
                    leaving=tab.basis[i],
                    pivot_row=tab.n_obj + i,
                    ratio_test=None,
                    drive_out=True,
                )
                if rec.k >= rec.max_iter:
                    return done("max_iter", f"reached max_iter={max_iter} pivots in phase 1")
                tab.pivot(i, j)
                rec.k += 1
                rec.last_theta = 0.0
                i += 1
            else:
                # Row i of B⁻¹A is zero on the real columns: the constraint of this
                # artificial (row art_rows[k] of the standard form) is redundant.
                removed.append(art_rows[tab.basis[i] - N])
                tab.T = np.delete(tab.T, tab.n_obj + i, axis=0)
                del tab.basis[i]
        # Phase 2 tableau: delete the artificial columns, install the true objective row.
        keep = [j for j in range(tab.N) if j not in tab.artificial]
        tab.T = np.concatenate([tab.T[:, keep], tab.T[:, -1:]], axis=1)
        tab.labels = [tab.labels[j] for j in keep]
        if tab.scale is not None:
            tab.scale = tab.scale[keep]
        tab.artificial = set()
        tab.T[0] = _objective_row(tab, sf.c)
        tab.cost_scale = (cost_scale(sf.c),)
        removed_info: dict[str, Any] = {"removed_rows": removed}
    else:
        tab.T[0] = _objective_row(tab, sf.c)
        tab.cost_scale = (cost_scale(sf.c),)
        removed_info = {}

    allowed = np.ones(tab.N, dtype=bool)
    status = primal_loop(
        tab, rec, rule=rule, tol=tol, phase=2, allowed=allowed, first_info=removed_info
    )
    return _finish(status, tab, rec, done, max_iter)


def _finish(status: str, tab: Tableau, rec: Recorder, done: Any, max_iter: int) -> LPOutcome:
    if status == "optimal":
        return done("optimal", "optimal: every reduced cost c̄ⱼ ≥ −tol")
    if status == "unbounded":
        name = tab.labels[rec.unbounded_col] if rec.unbounded_col is not None else "?"
        return done(
            "unbounded",
            f"LP is unbounded: column {name} has c̄ < 0 and no positive entry, "
            "so the objective improves without limit along the ray",
            rec.ray,
        )
    if status == "cycling":
        return done("cycling", "cycling detected: a basis repeated (use pivot_rule='bland')")
    return done("max_iter", f"reached max_iter={max_iter} pivots")


def _to_result(method: str, out: LPOutcome) -> Result:
    extra: dict[str, Any] = {"status": out.status, "pivots": out.pivots}
    if out.tab is not None:
        extra["basis"] = list(out.tab.basis)
    if out.ray is not None:
        extra["ray"] = out.ray
    fun = out.value if out.value is not None else (out.trace[-1].fun if out.trace else None)
    return Result(
        method=method,
        x=out.x,
        fun=fun,
        converged=out.status == "optimal",
        message=out.message,
        n_iter=out.pivots,
        trace=out.trace,
        extra=extra,
    )


def _check_params(rule: str, tol: float, max_iter: int, rules: Sequence[str]) -> None:
    if rule not in rules:
        raise ValueError(f"pivot_rule must be one of {tuple(rules)}, got {rule!r}")
    if not tol > 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be ≥ 1")


# --------------------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------------------


@register(
    id="simplex",
    family="lp",
    name="Primal simplex (tableau)",
    params=PIVOT_PARAMS,
    needs=("lp",),
    order="finite (vertex to adjacent vertex)",
    summary="Start at the origin and pivot to a better adjacent vertex until no reduced cost is negative.",
    references=(*REFERENCES, "Dantzig (1947); Hillier & Lieberman, §4.3–4.4"),
)
def simplex(
    problem: LinearProgram,
    *,
    pivot_rule: str = "dantzig",
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Result:
    """Primal simplex method, full-tableau implementation, for ``A_ub x ≤ b_ub`` with b ≥ 0.

    The slack basis B = I (the origin x = 0) is feasible because b ≥ 0, so no phase 1 is
    needed. Each iteration (Bertsimas & Tsitsiklis 1997, §3.3, "an iteration of the full
    tableau implementation"): pick an entering column j with c̄ⱼ < 0 by ``pivot_rule``; if
    ū = B⁻¹Aⱼ ≤ 0 the LP is unbounded; otherwise the minimum-ratio row leaves and a pivot on
    (row, j) moves to the adjacent vertex.

    Stops (converged) when every reduced cost c̄ⱼ ≥ −tol. Unbounded, cycling and max_iter
    stop with ``converged=False``. Raises ValueError when the problem has equality rows or
    a negative b_ub entry (use ``two_phase_simplex`` or ``big_m``).
    """
    lp = check_lp(problem)
    _check_params(pivot_rule, tol, max_iter, _PIVOT_RULES)
    if lp.A_eq is not None and np.size(lp.A_eq) > 0:
        raise ValueError(
            f"{lp.id}: simplex needs A_ub x ≤ b_ub only; use two_phase_simplex or big_m"
        )
    if lp.b_ub is not None and np.any(np.asarray(lp.b_ub, dtype=float) < 0):
        raise ValueError(
            f"{lp.id}: simplex needs b_ub ≥ 0 (origin feasible); "
            "use two_phase_simplex, big_m or dual_simplex"
        )
    sf = standard_form(lp)
    return _to_result("simplex", solve_two_phase(sf, rule=pivot_rule, tol=tol, max_iter=max_iter))


@register(
    id="two_phase_simplex",
    family="lp",
    name="Two-phase simplex",
    params=PIVOT_PARAMS,
    needs=("lp",),
    order="finite (vertex to adjacent vertex)",
    summary="Phase 1 minimizes the sum of artificial variables to find a vertex; phase 2 optimizes from it.",
    references=(*REFERENCES, "Dantzig, Orden & Wolfe (1955)"),
)
def two_phase_simplex(
    problem: LinearProgram,
    *,
    pivot_rule: str = "dantzig",
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Result:
    """Two-phase simplex method (Bertsimas & Tsitsiklis 1997, §3.5).

    Phase 1 solves min Σ aᵢ/ρᵢ s.t. A z + a = b, (z, a) ≥ 0 from the basis of slacks and
    artificials (``# NOTE:`` weights 1/ρᵢ instead of 1: the textbook phase 1 of the
    row-equilibrated LP, module docstring). If an artificial of its optimum z exceeds
    tol·(bᵢ + Σⱼ|A_ij|zⱼ) the LP is infeasible. Otherwise every artificial still in the
    basis is set to 0 and driven out by a degenerate pivot on a nonzero entry of its row in
    a real column; if the row has no such entry the constraint is linearly dependent on the
    others and is removed. Phase 2 deletes the artificial columns, installs c̄ = c̃ − (B⁻¹A)ᵀc̃_B in
    row 0 and continues as ``simplex``.

    Stops (converged) when phase 2 has every c̄ⱼ ≥ −tol·σⱼ·γ. Infeasible, unbounded, cycling
    and max_iter (counted over both phases) give ``converged=False``.
    """
    lp = check_lp(problem)
    _check_params(pivot_rule, tol, max_iter, _PIVOT_RULES)
    sf = standard_form(lp)
    return _to_result(
        "two_phase_simplex", solve_two_phase(sf, rule=pivot_rule, tol=tol, max_iter=max_iter)
    )


@register(
    id="big_m",
    family="lp",
    name="Big-M simplex",
    params=PIVOT_PARAMS,
    needs=("lp",),
    order="finite (vertex to adjacent vertex)",
    summary="Give each artificial variable a huge cost M so that the simplex drives them to zero while it optimizes.",
    references=(*REFERENCES, "Hillier & Lieberman, Introduction to Operations Research, §4.6"),
)
def big_m(
    problem: LinearProgram,
    *,
    pivot_rule: str = "dantzig",
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Result:
    """Big-M method with M kept symbolic (Hillier & Lieberman §4.6).

    Solves min c̃ᵀz + M·Σaᵢ s.t. A z + a = b, (z, a) ≥ 0 from the slack/artificial basis,
    where M is "larger than any number it is compared with". Each reduced cost is the pair
    c̄ⱼ = Mⱼ·M + cⱼ, kept in two objective rows (``zM`` and ``z``) and compared
    lexicographically. ``# NOTE:`` a numeric M (say 1e6) would put cancellation errors of
    size M·ε into every reduced cost and can give a wrong answer when M is too small; the
    symbolic M gives the textbook pivots without either problem.

    ``# NOTE:`` the artificial of row i has the M-row cost 1/ρᵢ (any positive weights give a
    big-M method; these make the M part the phase-1 objective of the row-equilibrated LP,
    module docstring). The M row is recomputed from these costs after every pivot instead
    of by the Gauss–Jordan update: once no artificial is basic its real entries are then
    exactly 0. The rounding residue of the update (1e-14 at tol = 1e-14, 7e-9 on badly
    scaled data) blocked every column with a negative cⱼ and gave a false "optimal".

    Stops (converged) when no column has Mⱼ < −tol·σⱼ, or |Mⱼ| ≤ tol·σⱼ and
    cⱼ < −tol·σⱼ·γ, and every artificial is at most tol·(bᵢ + Σⱼ|A_ij|zⱼ). An optimum with
    a larger artificial means the LP is infeasible. Unbounded, cycling and max_iter give
    ``converged=False``.
    """
    lp = check_lp(problem)
    _check_params(pivot_rule, tol, max_iter, _PIVOT_RULES)
    sf = standard_form(lp)
    rec = Recorder(sf, max_iter)
    tab = _initial_tableau(sf, n_obj=2)
    n_total = tab.N
    cost_m = artificial_weights(sf, n_total)
    cost_c = np.zeros(n_total)
    cost_c[: sf.c.size] = sf.c
    tab.T[0] = _objective_row(tab, cost_m)
    tab.T[1] = _objective_row(tab, cost_c)
    tab.cost_scale = (1.0, cost_scale(cost_c))
    allowed = np.ones(n_total, dtype=bool)

    def refresh_m_row() -> None:
        # [Mⱼ | −Σ wᵢaᵢ] = cost_M − cost_M[B]ᵀ B⁻¹[A | b]: exact once cost_M[B] = 0.
        tab.T[0] = _objective_row(tab, cost_m)

    status = primal_loop(
        tab, rec, rule=pivot_rule, tol=tol, phase=2, allowed=allowed, after_pivot=refresh_m_row
    )

    def done(st: str, message: str, ray: np.ndarray | None = None) -> LPOutcome:
        x = tab.values()[: sf.n]
        value = rec.objective(x) if st == "optimal" else None
        return LPOutcome(st, message, tab, x, value, rec.k, rec.trace, sf, ray)

    # The M-part (Σ aᵢ/ρᵢ) is at its minimum whenever no column has Mⱼ < −tol·σⱼ.
    cap = artificial_caps(sf, tab.values(), n_total, tol)
    bad = (
        artificial_violation(tab.basis, tab.rows()[:, -1], cap, tab.artificial)
        if status in ("optimal", "unbounded")
        else None
    )
    if bad is not None:
        out = done("infeasible", _infeasible_message(tab.labels, bad[0], bad[1], cap[bad[0]]))
    else:
        out = _finish(status, tab, rec, done, max_iter)
    return _to_result("big_m", out)


@register(
    id="dual_simplex",
    family="lp",
    name="Dual simplex (tableau)",
    params=(
        ParamSpec(
            "pivot_rule",
            "dantzig",
            kind="choice",
            choices=("dantzig", "bland"),
            help="Leaving row: most negative basic value, or smallest basic index (Bland).",
        ),
        *PIVOT_PARAMS[1:],
    ),
    needs=("lp",),
    order="finite (dual vertex to adjacent dual vertex)",
    summary="Keep every reduced cost ≥ 0 and pivot out negative basic variables until the basis is also primal feasible.",
    references=(
        "Lemke (1954)",
        "Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §4.5",
        "Chvátal, Linear Programming (1983), ch. 10",
    ),
)
def dual_simplex(
    problem: LinearProgram,
    *,
    pivot_rule: str = "dantzig",
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Result:
    """Dual simplex method, full-tableau implementation (Bertsimas & Tsitsiklis §4.5).

    Needs a dual-feasible slack basis: every row is written A x + s = b with b of any sign
    (equality rows aᵀx = β become aᵀx ≤ β and −aᵀx ≤ −β ``# NOTE:`` two degenerate rows instead
    of one), and the min-form costs must satisfy c̃ ≥ 0 so that c̄ = c̃ ≥ 0.

    Iteration: pick a row r with x_B,r < −tol·(|B⁻¹|·|b|)_r (``dantzig``: most negative
    x_B,r; ``bland``: smallest basic index); B⁻¹ is the slack block of the tableau.
    ``# NOTE:`` the threshold is the componentwise scale of x_B = B⁻¹b, which does not
    change under row or column scaling. The absolute x_B,r < −tol accepted the row
    1e-6·x₁ ≥ 1e-6 violated by 1e-6 (x₁ = 5e-5 at tol = 1e-5). If ā_rⱼ ≥ −τ_rⱼ for every j
    the LP is infeasible (row r reads Σ ā_rⱼ zⱼ = x_B,r < 0 with z ≥ 0). Otherwise the
    entering column minimizes c̄ⱼ / |ā_rⱼ| over ā_rⱼ < −τ_rⱼ (ties: smallest index), which
    keeps c̄ ≥ 0.

    Stops (converged) when every x_B,i ≥ −tol·(|B⁻¹|·|b|)ᵢ: the basis is primal and dual
    feasible, hence optimal. Infeasible, cycling and max_iter give ``converged=False``. Raises
    ValueError when the slack basis is not dual feasible: some c̃ⱼ < −tol·σⱼ·γ, the entering
    threshold of the primal methods (γ = maxⱼ |c̃ⱼ|), so that a negative cost of any scale is
    rejected and only a rounding-level one is treated as 0.
    """
    lp = check_lp(problem)
    _check_params(pivot_rule, tol, max_iter, ("dantzig", "bland"))
    sf = _inequality_form(lp)
    tab = _slack_tableau(sf)
    c_min = sf.c[: sf.n]
    if np.any(c_min < -tab.cost_tol(0, tol)[: sf.n]):
        raise ValueError(
            f"{lp.id}: dual_simplex needs a dual-feasible slack basis (min-form costs c̃ ≥ 0); "
            "use two_phase_simplex or big_m"
        )
    rec = Recorder(sf, max_iter)
    status = dual_loop(tab, rec, rule=pivot_rule, tol=tol, b=sf.b)

    def done(st: str, message: str, ray: np.ndarray | None = None) -> LPOutcome:
        x = tab.values()[: sf.n]
        value = rec.objective(x) if st == "optimal" else None
        return LPOutcome(st, message, tab, x, value, rec.k, rec.trace, sf, ray)

    return _to_result("dual_simplex", _dual_finish(status, tab, done, max_iter))


def _inequality_form(lp: LinearProgram) -> StandardForm:
    """A x + s = b with one slack per row, b of any sign, A_eq split into ≤ and ≥ rows."""
    c, a_ub, b_ub, a_eq, b_eq = lp_arrays(lp)
    a = np.vstack([a_ub, a_eq, -a_eq])
    b = np.concatenate([b_ub, b_eq, -b_eq])
    n, m = c.size, b.size
    A = np.hstack([a, np.eye(m)])
    sign = 1.0 if lp.sense == "min" else -1.0
    labels = [f"x{j + 1}" for j in range(n)] + [f"s{i + 1}" for i in range(m)]
    return StandardForm(
        A=A,
        b=b,
        c=np.concatenate([sign * c, np.zeros(m)]),
        labels=labels,
        n=n,
        sign=sign,
        unit_col=[n + i for i in range(m)],
        expr_const=np.concatenate([np.zeros(n), b]),
        expr_coef=np.vstack([np.eye(n), -a]),
    )


def _slack_tableau(sf: StandardForm) -> Tableau:
    m, N = sf.A.shape
    T = np.zeros((1 + m, N + 1))
    T[1:, :-1] = sf.A
    T[1:, -1] = sf.b
    basis = list(range(sf.n, sf.n + m))  # slack basis B = I
    tab = Tableau(T, basis, list(sf.labels), scale=column_scales(sf.A, sf.n))
    tab.T[0] = _objective_row(tab, sf.c)
    tab.cost_scale = (cost_scale(sf.c),)
    return tab


def dual_ratio_test(tab: Tableau, r: int, tol: float) -> tuple[int | None, list[float | None]]:
    """Entering column for leaving row r: argmin c̄ⱼ/|ā_rⱼ| over ā_rⱼ < −τ_rⱼ.

    A ratio tᵢ ties with the minimum t when tᵢ ≤ t·(1 + tol) (relative, as in
    :func:`min_ratio`): entering a tied column k sets c̄ⱼ of the minimizing column to at
    least −tol·c̄ⱼ.
    """
    row = tab.rows()[r, :-1]
    d = tab.T[0, :-1]
    if tab.scale is None:
        mask = row < -tol
    else:
        mask = row < -tol * tab.scale / tab.scale[tab.basis[r]]
    mask[tab.basis] = False
    ratios = np.full(tab.N, np.nan)
    # NOTE: c̄ⱼ = −1e-17 (rounding) is treated as 0, so the ratio is ≥ 0.
    ratios[mask] = np.maximum(d[mask], 0.0) / -row[mask]
    listed = [None if np.isnan(v) else float(v) for v in ratios]
    if not mask.any():
        return None, listed
    best = float(np.min(ratios[mask]))
    ties = np.flatnonzero(mask & (ratios <= best * (1.0 + tol)))
    return int(ties[0]), listed


def dual_loop(
    tab: Tableau, rec: Recorder, *, rule: str, tol: float, b: np.ndarray, phase: int = 2
) -> str:
    """Dual simplex pivots until primal feasible / infeasible / cycling / max_iter.

    ``tab`` must have one slack column per row in its last m columns (identity at the start),
    so that those columns of the constraint rows hold B⁻¹; ``b`` is the right-hand side.
    """
    seen: set[frozenset[int]] = set()
    while True:
        rows = tab.rows()
        rhs = rows[:, -1]
        p_inf = float(np.sum(np.maximum(-rhs, 0.0)))
        key = frozenset(tab.basis)
        if key in seen:
            rec.emit(tab, phase, cycling=True, primal_infeasibility=p_inf)
            return "cycling"
        seen.add(key)
        # Leave test x_B,i < −tol·(|B⁻¹|·|b|)ᵢ (dual_simplex docstring).
        b_inv = rows[:, tab.N - tab.m : tab.N]
        x_scale = np.abs(b_inv) @ np.abs(b)
        neg = [i for i in range(tab.m) if rhs[i] < -tol * x_scale[i]]
        if not neg:
            rec.emit(tab, phase, primal_infeasibility=p_inf)
            return "optimal"
        if rule == "bland":
            r = min(neg, key=lambda i: tab.basis[i])
        else:
            worst = min(float(rhs[i]) for i in neg)
            ties = [i for i in neg if rhs[i] <= worst + tol * (1.0 + abs(worst))]
            r = min(ties, key=lambda i: tab.basis[i])
        j, ratios = dual_ratio_test(tab, r, tol)
        info = {
            "entering": j,
            "leaving": tab.basis[r],
            "pivot_row": tab.n_obj + r,
            "ratio_test": ratios,
            "primal_infeasibility": p_inf,
        }
        rec.emit(tab, phase, **info)
        if j is None:
            return "infeasible"
        if rec.k >= rec.max_iter:
            return "max_iter"
        theta = float(tab.T[0, j] / -tab.rows()[r, j])
        tab.pivot(r, j)
        rec.k += 1
        rec.last_theta = theta


def _dual_finish(status: str, tab: Tableau, done: Any, max_iter: int) -> LPOutcome:
    if status == "optimal":
        return done(
            "optimal", "optimal: every basic value x_B,i ≥ −tol·(|B⁻¹||b|)ᵢ and every c̄ⱼ ≥ 0"
        )
    if status == "infeasible":
        return done(
            "infeasible",
            "LP is infeasible: a row has a negative basic value and no negative entry "
            "(the dual is unbounded)",
        )
    if status == "cycling":
        return done("cycling", "cycling detected: a basis repeated (use pivot_rule='bland')")
    return done("max_iter", f"reached max_iter={max_iter} pivots")


# --------------------------------------------------------------------------------------
# Revised simplex
# --------------------------------------------------------------------------------------


LUFactor = tuple[Matrix, np.ndarray, np.ndarray]


def lu_factor(B: Matrix) -> LUFactor | None:
    """LU with partial pivoting of the row-equilibrated basis, (D B)[p] = L U.

    D = diag(1 / maxⱼ |B_ij|) (row equilibration, Golub & Van Loan, 4th ed., §3.5.2), then
    Gaussian elimination with partial pivoting (Alg. 3.4.1). Returns ``(LU, p, d)`` with the
    unit lower factor below the diagonal and d = diag(D), or ``None`` when B has a zero row
    or a pivot is ≤ m·ε (numerically singular; ‖D B‖∞-rows are 1). ``# NOTE:`` without D the
    singularity test m·ε·‖B‖∞ depends on the row scaling: B = [[1, 0], [1e8, 1]] (a 1e8
    multiple of a constraint row) was reported singular.
    """
    m = B.shape[0]
    row_max = np.max(np.abs(B), axis=1) if B.size else np.ones(m)
    if np.any(row_max == 0.0):
        return None
    d = 1.0 / row_max
    LU = np.array(B * d[:, None], dtype=np.float64, copy=True)
    p = np.arange(m)
    small = m * np.finfo(np.float64).eps
    for k in range(m):
        piv = k + int(np.argmax(np.abs(LU[k:, k])))
        if abs(LU[piv, k]) <= small:
            return None
        if piv != k:
            LU[[k, piv]] = LU[[piv, k]]
            p[[k, piv]] = p[[piv, k]]
        LU[k + 1 :, k] /= LU[k, k]
        LU[k + 1 :, k + 1 :] -= np.outer(LU[k + 1 :, k], LU[k, k + 1 :])
    return LU, p, d


def lu_solve(factor: LUFactor, rhs: np.ndarray, *, trans: bool = False) -> np.ndarray:
    """Solve B y = rhs (or Bᵀ y = rhs) with the factor (D B)[p] = L U of :func:`lu_factor`.

    B y = rhs ⇔ (D B) y = D rhs;  Bᵀ y = rhs ⇔ (D B)ᵀ w = rhs with y = D w.
    """
    LU, p, d = factor
    m = LU.shape[0]
    if not trans:
        y = (d * np.asarray(rhs, dtype=np.float64))[p]
        for i in range(m):  # L y = P rhs (unit diagonal)
            y[i] -= LU[i, :i] @ y[:i]
        for i in range(m - 1, -1, -1):  # U x = y
            y[i] = (y[i] - LU[i, i + 1 :] @ y[i + 1 :]) / LU[i, i]
        return y
    # Bᵀ = (Pᵀ L U)ᵀ = Uᵀ Lᵀ P: solve Uᵀ v = rhs, Lᵀ w = v, then y = Pᵀ w (y[p] = w).
    v = np.array(rhs, dtype=np.float64)
    for i in range(m):
        v[i] = (v[i] - LU[:i, i] @ v[:i]) / LU[i, i]
    for i in range(m - 1, -1, -1):
        v[i] -= LU[i + 1 :, i] @ v[i + 1 :]
    y = np.empty(m)
    y[p] = v
    return d * y


@register(
    id="revised_simplex",
    family="lp",
    name="Revised simplex (LU)",
    params=PIVOT_PARAMS,
    needs=("lp",),
    order="finite (vertex to adjacent vertex)",
    summary="The simplex method without a tableau: solve with the basis matrix B for x_B, the duals and one column.",
    references=(
        "Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §3.3 (revised simplex)",
        "Chvátal, Linear Programming (1983), ch. 7",
        "Golub & Van Loan, Matrix Computations (4th ed.), Alg. 3.4.1",
    ),
)
def revised_simplex(
    problem: LinearProgram,
    *,
    pivot_rule: str = "dantzig",
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Result:
    """Revised simplex method with an LU factorization of the basis matrix, two phases.

    Iteration (Bertsimas & Tsitsiklis §3.3; Chvátal ch. 7), with B = A[:, basis] factored
    once as P B = L U and the factor reused for every solve of the iteration:

    1. x_B = B⁻¹b  (solve B x_B = b),
    2. y = B⁻ᵀc_B  (BTRAN: solve Bᵀ y = c_B),
    3. c̄ = c − Aᵀy  (pricing); stop if c̄ⱼ ≥ −tol for all j,
    4. ū = B⁻¹A_q  (FTRAN) for the entering column q; unbounded if ū ≤ tol,
    5. minimum-ratio test on x_B / ū, then replace the leaving column of B by A_q.

    ``# NOTE:`` B is refactored from scratch every iteration (no Forrest–Tomlin or
    product-form update); this costs O(m³) per iteration but is simple and stable.
    ``steepest_edge`` needs ‖B⁻¹Aⱼ‖ for every nonbasic j, computed here by one FTRAN per
    column. Phase 1 / drive-out / redundant rows follow ``two_phase_simplex``; with the same
    ``pivot_rule`` both methods visit the same bases. The ``tableau`` info key is
    B⁻¹[A | b] with its reduced-cost row, formed for display only.

    Stops (converged) when every reduced cost c̄ⱼ ≥ −tol in phase 2. Infeasible, unbounded,
    cycling, a numerically singular basis and max_iter give ``converged=False``.
    """
    lp = check_lp(problem)
    _check_params(pivot_rule, tol, max_iter, _PIVOT_RULES)
    sf = standard_form(lp)
    return _to_result("revised_simplex", _revised(sf, pivot_rule, tol, max_iter))


class _RevisedState:
    """Revised-simplex data: A (with artificial columns), b, the basis and its LU factor."""

    def __init__(self, sf: StandardForm) -> None:
        m, N = sf.A.shape
        self.art_rows = [i for i in range(m) if sf.unit_col[i] is None]
        n_art = len(self.art_rows)
        self.A = np.zeros((m, N + n_art))
        self.A[:, :N] = sf.A
        for k, i in enumerate(self.art_rows):
            self.A[i, N + k] = 1.0
        self.b = sf.b.copy()
        self.labels = sf.labels + [f"a{i + 1}" for i in self.art_rows]
        self.artificial = set(range(N, N + n_art))
        self.scale = column_scales(self.A, sf.n)
        self.art_cost = artificial_weights(sf, N + n_art)
        self.basis: list[int] = [
            uc if uc is not None else N + self.art_rows.index(i) for i, uc in enumerate(sf.unit_col)
        ]

    @property
    def N(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]


def _revised(sf: StandardForm, rule: str, tol: float, max_iter: int) -> LPOutcome:
    st = _RevisedState(sf)
    rec = Recorder(sf, max_iter)
    n = sf.n

    def tableau_view(factor: Any, cost: np.ndarray) -> Tableau:
        rows = np.column_stack([lu_solve(factor, col) for col in st.A.T] + [lu_solve(factor, st.b)])
        T = np.vstack([np.zeros(st.N + 1), rows])
        tab = Tableau(T, list(st.basis), list(st.labels), 1, set(st.artificial))
        tab.art_cost = st.art_cost[: st.N]
        tab.T[0] = _objective_row(tab, cost)
        return tab

    def done(
        status: str, message: str, ray: np.ndarray | None = None, x_b: Any = None
    ) -> LPOutcome:
        z = np.zeros(st.N)
        if x_b is not None:
            z[st.basis] = x_b
        x = z[:n]
        value = rec.objective(x) if status == "optimal" else None
        return LPOutcome(status, message, None, x, value, rec.k, rec.trace, sf, ray)

    def run_phase(
        cost: np.ndarray, phase: int, emit_optimal: bool, first_info: dict
    ) -> tuple[str, Any]:
        seen: set[frozenset[int]] = set()
        extra_info = dict(first_info)
        # γ = 1 for the phase-1 row (its costs are 1 on the equilibrated rows).
        gamma = 1.0 if phase == 1 else cost_scale(cost)
        while True:
            factor = (
                lu_factor(st.A[:, st.basis])
                if st.m
                else (np.zeros((0, 0)), np.zeros(0, int), np.ones(0))
            )
            if factor is None:
                return "singular", None
            x_b = lu_solve(factor, st.b) if st.m else np.zeros(0)
            y = lu_solve(factor, cost[st.basis], trans=True) if st.m else np.zeros(0)
            d = cost - st.A.T @ y
            d[st.basis] = 0.0  # NOTE: exact zeros for basic columns
            tab = tableau_view(factor, cost) if st.m else _empty_tab(cost, st)
            key = frozenset(st.basis)
            common = {"duals": y, "reduced_costs": d, "direction": None, **extra_info}
            extra_info = {}
            if key in seen:
                rec.emit(tab, phase, cycling=True, **common)
                return "cycling", x_b
            seen.add(key)
            allowed = np.ones(st.N, dtype=bool)
            q = _revised_entering(d, rule, tol, allowed, st, factor, gamma)
            if q is None:
                if emit_optimal:
                    rec.emit(tab, phase, **common)
                return "optimal", x_b
            u = lu_solve(factor, st.A[:, q]) if st.m else np.zeros(0)
            common["direction"] = u
            r, theta, ratios = min_ratio(
                x_b, u, tol, st.basis, tol * st.scale[q] / st.scale[st.basis]
            )
            if r is None:
                eta = np.zeros(st.N)
                eta[q] = 1.0
                eta[st.basis] = -u
                rec.emit(tab, phase, **_pivot_info(tab, q, None, ratios), ray=eta[:n], **common)
                return "unbounded", x_b
            rec.emit(tab, phase, **_pivot_info(tab, q, r, ratios), **common)
            if rec.k >= rec.max_iter:
                return "max_iter", x_b
            st.basis[r] = q
            rec.k += 1
            rec.last_theta = theta

    first_info: dict[str, Any] = {}
    if st.artificial:
        # Phase 1: min Σ aᵢ/ρᵢ (module docstring, "Phase 1 and big-M").
        w_cost = artificial_weights(sf, st.N)
        status, x_b = run_phase(w_cost, 1, False, {})
        if status in ("singular", "max_iter", "cycling"):
            return done(status, _revised_message(status, max_iter, 1), x_b=x_b)
        z = np.zeros(st.N)
        z[st.basis] = x_b
        cap = artificial_caps(sf, z, st.N, tol)
        bad = artificial_violation(st.basis, x_b, cap, st.artificial)
        if bad is not None:
            factor = lu_factor(st.A[:, st.basis])
            assert factor is not None
            rec.emit(tableau_view(factor, w_cost), 1)
            return done(
                "infeasible",
                _infeasible_message(st.labels, bad[0], bad[1], cap[bad[0]]),
                x_b=x_b,
            )
        removed: list[int] = []
        orig_row = list(range(st.m))
        N_real = sf.A.shape[1]
        i = 0
        while i < st.m:
            if st.basis[i] not in st.artificial:
                i += 1
                continue
            # NOTE: set the accepted level to exactly 0 by moving b of the artificial's row
            # (as in solve_two_phase): B·eᵢ is that artificial's unit column, so only x_B,i
            # changes and the drive-out pivot is degenerate.
            art_row = int(np.argmax(st.A[:, st.basis[i]]))
            st.b[art_row] -= x_b[i]
            x_b[i] = 0.0
            factor = lu_factor(st.A[:, st.basis])
            if factor is None:
                return done("singular", _revised_message("singular", max_iter, 1), x_b=x_b)
            e = np.zeros(st.m)
            e[i] = 1.0
            binv_row = lu_solve(factor, e, trans=True)  # πᵀ = row i of B⁻¹
            row = binv_row @ st.A[:, :N_real]
            rel = np.abs(row) / (tol * st.scale[:N_real] / st.scale[st.basis[i]])
            j = _drive_out_column(rel, tol)
            if rel[j] > 1.0:
                tab = tableau_view(factor, w_cost)
                rec.emit(
                    tab,
                    1,
                    entering=j,
                    leaving=st.basis[i],
                    pivot_row=1 + i,
                    ratio_test=None,
                    drive_out=True,
                )
                if rec.k >= rec.max_iter:
                    return done("max_iter", _revised_message("max_iter", max_iter, 1), x_b=x_b)
                st.basis[i] = j
                rec.k += 1
                rec.last_theta = 0.0
                i += 1
            else:
                # Row i of B⁻¹A is zero on the real columns: πᵀA = 0 with π_art_row = 1, so
                # constraint art_row is a combination of the others. Deleting it with the
                # artificial (its unit column) leaves a nonsingular basis.
                removed.append(orig_row.pop(art_row))
                st.A = np.delete(st.A, art_row, axis=0)
                st.b = np.delete(st.b, art_row)
                x_b = np.delete(x_b, i)
                del st.basis[i]
        keep = list(range(N_real))
        st.A = st.A[:, keep]
        st.scale = st.scale[keep]
        st.labels = st.labels[:N_real]
        st.artificial = set()
        first_info = {"removed_rows": removed}
    status, x_b = run_phase(sf.c.copy(), 2, True, first_info)
    if status == "optimal":
        return done("optimal", "optimal: every reduced cost c̄ⱼ ≥ −tol", x_b=x_b)
    if status == "unbounded":
        ray = np.asarray(rec.trace[-1].info["ray"])
        name = st.labels[rec.trace[-1].info["entering"]]
        return done(
            "unbounded",
            f"LP is unbounded: column {name} has c̄ < 0 and B⁻¹A_q ≤ 0, "
            "so the objective improves without limit along the ray",
            ray,
            x_b=x_b,
        )
    return done(status, _revised_message(status, max_iter, 2), x_b=x_b)


def _empty_tab(cost: np.ndarray, st: _RevisedState) -> Tableau:
    T = np.zeros((1, st.N + 1))
    T[0, :-1] = cost
    return Tableau(T, [], list(st.labels), 1, set(st.artificial))


def _revised_entering(
    d: np.ndarray,
    rule: str,
    tol: float,
    allowed: np.ndarray,
    st: _RevisedState,
    factor: Any,
    gamma: float,
) -> int | None:
    """Entering column among c̄ⱼ < −tol·σⱼ·γ (same rule as :func:`choose_entering`)."""
    cand = np.flatnonzero(allowed & (d < -tol * st.scale * gamma))
    if cand.size == 0:
        return None
    if rule == "bland":
        return int(cand[0])
    score = d[cand]
    if rule == "steepest_edge":
        norms = np.array(
            [
                np.sqrt(1.0 + np.sum(lu_solve(factor, st.A[:, j]) ** 2)) if st.m else 1.0
                for j in cand
            ]
        )
        score = score / norms
    best = float(score.min())
    ties = cand[score <= best + tol * (1.0 + abs(best))]
    return int(ties[0])


def _revised_message(status: str, max_iter: int, phase: int) -> str:
    if status == "singular":
        return f"basis matrix became numerically singular in phase {phase}"
    if status == "cycling":
        return f"cycling detected in phase {phase}: a basis repeated (use pivot_rule='bland')"
    return f"reached max_iter={max_iter} pivots"


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("simplex", "wyndor", {}),
    ("simplex", "klee_minty_3", {}),
    ("simplex", "degenerate_2d", {}),
    ("two_phase_simplex", "diet_2d", {}),
    ("two_phase_simplex", "unbounded_2d", {"pivot_rule": "bland"}),
    ("big_m", "infeasible_2d", {}),
    ("dual_simplex", "diet_2d", {"pivot_rule": "bland"}),
    ("revised_simplex", "transport_small", {"pivot_rule": "steepest_edge"}),
]
