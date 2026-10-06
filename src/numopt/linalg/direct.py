"""Direct solvers for a square linear system A x = b.

A direct method transforms ``[A | b]`` (or factors ``A``) in a finite number of *stages*, one per
pivot column, and then solves triangular systems. The trace has one Step per stage so that the
visualizer can animate the elimination:

* ``k = 0``           the initial state (phase ``"start"``),
* ``k = 1 … n``       stage ``k`` works on pivot column ``c = k − 1`` (phase ``"eliminate"``),
* ``k = n + 1``       the triangular solve that produces ``x`` (phase ``"back_substitution"`` or
  ``"solve"``). Gauss–Jordan has no such step: its last stage already shows ``[I | x]``.

``Step.x`` and ``Step.fun`` are ``None`` until the solution exists; on the final step ``Step.fun``
is the residual norm ‖b − A x‖₂ of the computed solution.

A pivot ``p`` counts as zero when ``|p| ≤ τ = n·ε·‖A‖_F`` (ε = 2⁻⁵², the machine epsilon).

# NOTE: the textbooks test ``p = 0``. In floating point an exactly singular matrix usually leaves a
# pivot of size O(ε‖A‖) instead of 0 (for ``singular_3``, |u₃₃| ≈ 1.6e-16), so an exact test misses
# it. τ is the rank threshold of ``numpy.linalg.matrix_rank`` with ‖A‖_F ≥ σ_max in place of σ_max.
# It is a heuristic: a system with a pivot just above τ is solved, and its cond_estimate tells how
# many digits the solution can have lost (about log10 κ).

A breakdown (a zero pivot, a matrix that is not SPD for Cholesky, a non-finite value) stops the
method with ``converged=False``, ``x=None`` and a message that names the stage.

A finished solve is then certified a posteriori by its normwise backward error (Rigal & Gaches
1967; Higham 2002, Thm 7.1)

    η∞ = ‖b − Ax‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞)  ≤  30·n·ε,

the smallest relative perturbation of A and b of which x is the exact solution. ``converged=True``
means that every pivot passed the test above, ``x`` is finite and η∞ passed. Otherwise the method
returns ``converged=False`` with the computed ``x`` and a message that says the solve was unstable.

# NOTE: the textbooks state no such test; a pivot test alone does not certify x. Elimination
# without pivoting on [[1e-15, 1], [1, 1]] passes every pivot test (τ ≈ 7.7e-16) but has growth
# factor 1e15 and returns x with an 11 % error (η∞ ≈ 0.03); partial pivoting on Wilkinson's
# 60 × 60 growth matrix (ρ = 2⁵⁹) returns η∞ ≈ 5e-2. A backward-stable solve has η∞ = O(nu): the
# measured maximum of η∞/(nε) is about 1 for QR, Cholesky and partial pivoting on random matrices,
# and 30 is the pass threshold that the LAPACK test suite applies to its residual ratios. A
# residual computed in floating point carries an error of at most γ_{n+1}(|A||x| + |b|), so
# η∞ ≤ 30nε is always resolvable.

# NOTE: Gauss–Jordan elimination is forward stable but not backward stable (Peters & Wilkinson
# 1975; Higham 2002, §14.4): its residual can legitimately be of the size κ(A)·u·‖A‖‖x‖. Its test is
# η∞ ≤ 30·n·ε·max(1, κ̂₁) with the estimate κ̂₁ = cond_estimate.

Accepted problems: a :class:`~numopt.core.types.LinearSystem` or a pair ``(A, b)``.

Result.extra (all methods):
    cond_estimate: float, an estimate of κ₁(A) = ‖A‖₁‖A⁻¹‖₁ from Hager's (1984) method with
        Higham's (1988) extra test vector, as in LAPACK ``xGECON``; it uses only solves with the
        computed factors (never A⁻¹) and is a lower bound of κ₁(A). ``inf`` after a breakdown.
    residual_norm: float, ‖b − A x‖₂ (``None`` after a breakdown).
    backward_error: float, η∞ of the computed x (absent after a breakdown).
    backward_error_bound: float, the threshold of the η∞ test: 30·n·ε, times max(1, κ̂₁) for
        Gauss–Jordan (absent after a breakdown).
    growth_factor: float, ρ = max_{i,j,k} |a_ij^(k)| / max_{i,j} |a_ij| over every stage
        (Higham 2002, Ch. 9); elimination methods only (GE, GEPP, Gauss–Jordan, LU). For
        Gauss–Jordan the pivot row enters before it is divided by the pivot.
    Factors (method-specific): ``L``, ``U``, ``P``, ``perm`` (LU), ``L`` (Cholesky),
        ``Q``, ``R``, ``householder_vectors`` (QR), ``L``, ``U``, ``c_prime``, ``d_prime`` (Thomas).

Info keys:
    phase: str, ``"start"``, ``"eliminate"``, ``"back_substitution"`` or ``"solve"``.
    matrix: [[n + 1] × n] the augmented matrix [A | b] after the stage (GE, GEPP, Gauss–Jordan,
        Thomas; for QR it is [R | Qᵀb] in progress). For LU and Cholesky it is the n × n working
        matrix: LU keeps the finished rows of U above the active Schur complement; Cholesky keeps
        the active Schur complement and zeros in the finished rows and columns. On the final
        ``"solve"`` step it is U (LU) or Lᵀ (Cholesky).
    pivot: [row, col] the pivot position in ``matrix`` (after any row swap); ``None`` at k = 0 and
        on the solve step.
    pivot_value: float, the pivot before the stage used it: the diagonal entry for elimination
        methods, d = a_kk − Σ l_kj² for Cholesky, the denominator w_k for Thomas, r_kk for QR.
    row_swap: [c, p] when rows c and p were exchanged before the stage, else ``None``.
    multipliers: [n] the factor m_i of the row operation row_i ← row_i − m_i·row_pivot at this
        stage, 0 for rows that the stage does not change (GE, GEPP, Gauss–Jordan, LU, Thomas).
    zero_pivot: bool, ``True`` on the stage where the method broke down.
    L: [[n] × n] the lower-triangular factor so far (LU: unit diagonal; Cholesky).
    perm: [n] row permutation so far: row i of P·A is row perm[i] of A (GEPP, Gauss–Jordan, LU).
    column: [n] the new column of L (Cholesky).
    householder_vector: [n] the unit vector v of H = I − 2vvᵀ, zero above the pivot row; all
        zeros when no reflection is needed (QR).
    c_prime, d_prime: [n] the modified super-diagonal and right-hand side so far (Thomas).
    y: [n] the intermediate solution of the forward substitution L y = P b (LU, Cholesky).
    residual: [n] r = b − A x on the final step.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import numpy as np

from ..core.registry import register
from ..core.types import LinearSystem, Result, Step, Vector

#: Machine epsilon ε = 2⁻⁵² of IEEE double precision (twice the unit roundoff u = 2⁻⁵³).
EPS = float(np.finfo(np.float64).eps)

SystemLike = LinearSystem | tuple[Any, Any] | Sequence[Any]

_F = TypeVar("_F", bound=Callable[..., Result])


def quiet(fn: _F) -> _F:
    """Run ``fn`` with NumPy overflow/invalid warnings off: methods detect non-finite values
    themselves and report them in the Result, and they must not print."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Result:
        with np.errstate(over="ignore", invalid="ignore"):
            return fn(*args, **kwargs)

    return cast(_F, wrapper)


# --------------------------------------------------------------------------------------
# Shared helpers (also used by numopt.linalg.iterative)
# --------------------------------------------------------------------------------------


def resolve_system(problem: SystemLike) -> tuple[np.ndarray, Vector]:
    """Return fresh float64 copies ``(A, b)`` of a LinearSystem or an ``(A, b)`` pair."""
    if isinstance(problem, LinearSystem):
        A_raw, b_raw = problem.A, problem.b
    elif isinstance(problem, (tuple, list)) and len(problem) == 2:
        A_raw, b_raw = problem
    else:
        raise TypeError("problem must be a numopt LinearSystem or a pair (A, b)")
    A = np.array(A_raw, dtype=np.float64, copy=True)
    b = np.array(b_raw, dtype=np.float64, copy=True).reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError(f"A must be a non-empty square matrix, got shape {A.shape}")
    if b.size != A.shape[0]:
        raise ValueError(f"b has {b.size} entries, expected {A.shape[0]}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("A and b must contain only finite values")
    return A, b


def _norm2_parts(v: np.ndarray) -> tuple[float, float]:
    """(s, t) with ‖v‖₂ = s·t, s = max|v_i| and 1 ≤ t ≤ √(size); (0, 0) for a zero v."""
    w = np.abs(np.asarray(v, dtype=np.float64)).reshape(-1)
    s = float(np.max(w, initial=0.0))
    if s == 0.0 or not np.isfinite(s):
        return s, 1.0
    return s, float(np.sqrt(np.sum((w / s) ** 2)))


def norm2(v: np.ndarray) -> float:
    """‖v‖₂ (Frobenius norm for a matrix) without overflow or underflow of the squares.

    # NOTE: ``numpy.linalg.norm`` forms Σv_i² unscaled, so it returns 0 for entries ≲ 1e-162 and
    # ∞ for entries ≳ 1e154. Scaling by s = max|v_i| first, as LAPACK ``xNRM2`` does, avoids both.
    # The result is ``inf`` only when the norm itself exceeds the largest double (≈ 1.8e308), for
    # example ‖1e308·𝟙₄‖₂ = 2e308; callers that need a quantity of the size ε·‖v‖ use
    # :func:`pivot_tolerance`, which stays finite in that case.
    """
    s, t = _norm2_parts(v)
    return s * t


def pivot_tolerance(A: np.ndarray) -> float:
    """τ = n·ε·‖A‖_F: a pivot with |p| ≤ τ is treated as zero (see the module docstring).

    # NOTE: when ‖A‖_F overflows (finite entries with ‖A‖_F > 1.8e308, e.g. 1e308·I₄), τ is formed
    # as (n·ε·s)·‖A/s‖_F with s = max|a_ij|, which is finite (≈ 1.8e293 for 1e308·I₄). Without this,
    # τ = ∞ and every pivot of a perfectly conditioned matrix counts as zero.
    """
    n = A.shape[0]
    s, t = _norm2_parts(A)
    a_norm = s * t
    if np.isfinite(a_norm):
        return n * EPS * a_norm
    return (n * EPS * s) * t


def is_symmetric(A: np.ndarray) -> bool:
    """``True`` when max |a_ij − a_ji| ≤ τ (exactly symmetric data always passes)."""
    return float(np.max(np.abs(A - A.T))) <= pivot_tolerance(A)


def forward_substitution(L: np.ndarray, y: Vector, *, unit: bool = False) -> Vector:
    """Solve L x = y for lower-triangular L (Golub & Van Loan 2013, Alg. 3.1.1, row version)."""
    n = y.size
    x = np.zeros(n)
    for i in range(n):
        s = y[i] - L[i, :i] @ x[:i]
        x[i] = s if unit else s / L[i, i]
    return x


def back_substitution(U: np.ndarray, y: Vector) -> Vector:
    """Solve U x = y for upper-triangular U (Golub & Van Loan 2013, Alg. 3.1.2, row version)."""
    n = y.size
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def estimate_inv_norm1(
    solve: Callable[[Vector], Vector], solve_t: Callable[[Vector], Vector], n: int
) -> float:
    """Lower-bound estimate of ‖A⁻¹‖₁ from solves with A and Aᵀ only.

    Hager (1984), "Condition estimates", SIAM J. Sci. Stat. Comput. 5(2), as refined by Higham
    (1988), ACM TOMS 14(4) (LAPACK ``xLACN2``; Higham 2002, Ch. 15). With B = A⁻¹:

        x = (1/n)·𝟙;  repeat ≤ 5 times:  y = Bx, ξ = sign(y), z = Bᵀξ, j = argmax |z_j|;
        stop when ‖z‖_∞ ≤ zᵀx (Hager's local-maximum test) or j repeats, else x = e_j.

    As in LAPACK, the test is skipped after the first solve, so x = e_j is always tried once.

    The result ‖y‖₁ = ‖Bx‖₁ with ‖x‖₁ = 1 is a lower bound of ‖B‖₁. Higham's extra test vector
    x̃_i = (−1)^i (1 + i/(n − 1)), i = 0 … n−1, adds the bound 2‖Bx̃‖₁/(3n) = ‖Bx̃‖₁/‖x̃‖₁.
    """
    x = np.full(n, 1.0 / n)
    est = 0.0
    j_prev = -1
    for it in range(5):
        y = solve(x)
        est = max(est, float(np.sum(np.abs(y))))
        xi = np.where(y >= 0.0, 1.0, -1.0)
        z = solve_t(xi)
        j = int(np.argmax(np.abs(z)))
        if (it > 0 and float(np.abs(z[j])) <= float(z @ x)) or j == j_prev:
            break
        x = np.zeros(n)
        x[j] = 1.0
        j_prev = j
    if n > 1:
        i = np.arange(n, dtype=np.float64)
        x_alt = np.where(i % 2 == 0, 1.0, -1.0) * (1.0 + i / (n - 1))
        est = max(est, 2.0 * float(np.sum(np.abs(solve(x_alt)))) / (3.0 * n))
    return est


def cond_estimate(
    A: np.ndarray, solve: Callable[[Vector], Vector], solve_t: Callable[[Vector], Vector]
) -> float:
    """κ₁(A) ≈ ‖A‖₁ · est(‖A⁻¹‖₁); ``inf`` if the estimate is not finite.

    # NOTE: when ‖A‖₁ overflows (finite entries, column sum > 1.8e308) the product is formed as
    # ‖A/s‖₁·(s·est) with s = max|a_ij|, since est ≈ ‖A⁻¹‖₁ is then of the size 1/s.
    """
    inv_est = estimate_inv_norm1(solve, solve_t, A.shape[0])
    a1 = float(np.linalg.norm(A, 1))
    if np.isfinite(a1):
        est = a1 * inv_est
    else:
        s = float(np.max(np.abs(A)))
        est = float(np.linalg.norm(A / s, 1)) * (s * inv_est)
    return est if np.isfinite(est) else float("inf")


def backward_error(A: np.ndarray, b: Vector, x: Vector, r: Vector) -> float:
    """Normwise backward error η∞ = ‖r‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) of x, with r = b − A x.

    η∞ is the smallest η such that (A + ΔA) x = b + Δb with ‖ΔA‖∞ ≤ η‖A‖∞ and ‖Δb‖∞ ≤ η‖b‖∞
    (Rigal & Gaches 1967; Higham 2002, Thm 7.1). It is 0 when r = 0 and ``inf`` when r is not
    finite or the denominator is 0 while r ≠ 0.

    # NOTE: evaluated in the log domain, log η = log‖r‖ − logaddexp(log‖A‖ + log‖x‖, log‖b‖), with
    # ‖A‖∞ = s·‖A/s‖∞ (s = max|a_ij|), so that ‖A‖∞‖x‖∞ cannot overflow for finite data. The
    # logarithms add a relative error of about 700ε to η, which no test threshold can see.
    """
    if not np.all(np.isfinite(r)):
        return float("inf")
    r_inf = float(np.max(np.abs(r), initial=0.0))
    if r_inf == 0.0:
        return 0.0
    W = np.abs(A)
    s = float(np.max(W))
    with np.errstate(divide="ignore"):
        log_a = np.log(s) + np.log(float(np.max(np.sum(W / s, axis=1)))) if s > 0 else -np.inf
        log_x = np.log(float(np.max(np.abs(x), initial=0.0)))
        log_b = np.log(float(np.max(np.abs(b), initial=0.0)))
    log_den = float(np.logaddexp(log_a + log_x, log_b))
    if log_den == -np.inf:
        return float("inf")
    return float(np.exp(np.log(r_inf) - log_den))


#: The a-posteriori stability test passes when η∞ ≤ BACKWARD_ERROR_FACTOR·n·ε (module docstring).
BACKWARD_ERROR_FACTOR = 30.0


def _lu_solvers(
    L: np.ndarray, U: np.ndarray, perm: np.ndarray
) -> tuple[Callable[[Vector], Vector], Callable[[Vector], Vector]]:
    """Solves with A and Aᵀ from P·A = L·U (row i of P·A is row perm[i] of A)."""

    def solve(v: Vector) -> Vector:
        return back_substitution(U, forward_substitution(L, v[perm], unit=True))

    def solve_t(v: Vector) -> Vector:
        # A = Pᵀ L U, so Aᵀ = Uᵀ Lᵀ P and Aᵀ z = v ⇔ Uᵀ w = v, then Lᵀ (P z) = w.
        w = forward_substitution(U.T, v)
        pz = back_substitution(L.T, w)
        z = np.empty_like(pz)
        z[perm] = pz
        return z

    return solve, solve_t


def _growth(current: float, M: np.ndarray, n: int) -> float:
    """max(current, max |m_ij|) over the coefficient columns of M (an empty M changes nothing)."""
    return max(current, float(np.max(np.abs(M[:, :n]), initial=0.0)))


def _fail(
    method: str,
    message: str,
    trace: list[Step],
    extra: dict[str, Any] | None = None,
) -> Result:
    out: dict[str, Any] = {"cond_estimate": float("inf"), "residual_norm": None}
    out.update(extra or {})
    return Result(method, None, None, False, message, trace[-1].k, trace=trace, extra=out)


def _certify(
    method: str,
    A: np.ndarray,
    b: Vector,
    x: Vector,
    r: Vector,
    trace: list[Step],
    extra: dict[str, Any],
    hint: str,
    kappa: float = 1.0,
) -> Result:
    """Build the Result of a finished solve; converged only when η∞(x) ≤ 30·n·ε·kappa.

    ``kappa`` is 1 for the backward-stable methods and the estimate κ̂₁ for Gauss–Jordan (module
    docstring). On failure x is kept, so the student can see how wrong it is; ``hint`` says
    what to use instead.
    """
    n = b.size
    rnorm = float(extra["residual_norm"])
    eta = backward_error(A, b, x, r)
    bound = BACKWARD_ERROR_FACTOR * n * EPS * max(1.0, kappa)
    out = {**extra, "backward_error": eta, "backward_error_bound": bound}
    if eta <= bound:
        msg = (
            f"factorization completed; residual ‖b − Ax‖₂ = {rnorm:.3g}, "
            f"backward error η∞ = {eta:.3g}"
        )
        return Result(method, x, rnorm, True, msg, trace[-1].k, trace=trace, extra=out)
    if not np.isfinite(eta):
        msg = "the residual b − Ax of the computed x is not finite, so x cannot be certified"
    else:
        rho = extra.get("growth_factor")
        growth = f" (growth factor ρ = {rho:.3g})" if rho is not None else ""
        msg = (
            f"unstable: the backward error η∞ = ‖b − Ax‖∞/(‖A‖∞‖x‖∞ + ‖b‖∞) = {eta:.3g} of the "
            f"computed x exceeds {bound:.3g}{growth}, so x is not the solution of a nearby "
            f"system{hint}"
        )
    return Result(method, x, rnorm, False, msg, trace[-1].k, trace=trace, extra=out)


def _stage_info(
    M: np.ndarray,
    c: int,
    pivot_value: float,
    row_swap: list[int] | None,
    multipliers: np.ndarray,
    zero_pivot: bool,
    **more: Any,
) -> dict[str, Any]:
    return {
        "phase": "eliminate",
        "matrix": M.copy(),
        "pivot": [c, c],
        "pivot_value": pivot_value,
        "row_swap": row_swap,
        "multipliers": multipliers.copy(),
        "zero_pivot": zero_pivot,
        **more,
    }


def _start_info(M: np.ndarray, **more: Any) -> dict[str, Any]:
    return {
        "phase": "start",
        "matrix": M.copy(),
        "pivot": None,
        "pivot_value": None,
        "row_swap": None,
        "multipliers": None,
        "zero_pivot": False,
        **more,
    }


# --------------------------------------------------------------------------------------
# Gaussian elimination (with and without partial pivoting)
# --------------------------------------------------------------------------------------


@quiet
def _gaussian(method: str, problem: SystemLike, pivoting: bool) -> Result:
    A, b = resolve_system(problem)
    n = b.size
    tau = pivot_tolerance(A)
    a_max = float(np.max(np.abs(A)))
    M = np.hstack([A, b[:, None]])  # augmented [A | b], shape (n, n+1)
    L = np.eye(n)  # unit lower-triangular multipliers, rows follow the swaps
    perm = np.arange(n)
    growth = a_max
    extra_perm: dict[str, Any] = {"perm": perm.copy()} if pivoting else {}
    trace = [Step(0, None, None, info=_start_info(M, **extra_perm))]

    for c in range(n):
        row_swap: list[int] | None = None
        if pivoting:
            # Partial pivoting: the first row index of max |m_ic|, i ≥ c (LAPACK idamax rule).
            p = c + int(np.argmax(np.abs(M[c:, c])))
            if p != c:
                M[[c, p]] = M[[p, c]]
                L[[c, p], :c] = L[[p, c], :c]
                perm[[c, p]] = perm[[p, c]]
                row_swap = [c, p]
        pivot = float(M[c, c])
        mult = np.zeros(n)
        more: dict[str, Any] = {"perm": perm.copy()} if pivoting else {}
        if abs(pivot) <= tau:
            trace.append(
                Step(c + 1, None, None, info=_stage_info(M, c, pivot, row_swap, mult, True, **more))
            )
            hint = "" if pivoting else "; partial pivoting (row interchanges) may avoid it"
            what = "A is singular to working precision" if pivoting else "elimination breaks down"
            return _fail(
                method,
                f"zero pivot |a_{c}{c}| = {abs(pivot):.3g} ≤ τ = {tau:.3g} at stage {c + 1}: "
                f"{what}{hint}",
                trace,
                {"growth_factor": growth / a_max if a_max > 0 else None},
            )
        mult[c + 1 :] = M[c + 1 :, c] / pivot
        # Outer-product update of the rows below the pivot (GVL Alg. 3.2.1 / 3.4.1).
        M[c + 1 :, c:] -= np.outer(mult[c + 1 :], M[c, c:])
        # NOTE: the eliminated entries are zero by construction; store an exact 0 instead of the
        # rounding residue a_ic − (a_ic/p)·p, as every textbook does (it holds l_ic in LAPACK).
        M[c + 1 :, c] = 0.0
        L[c + 1 :, c] = mult[c + 1 :]
        growth = _growth(growth, M, n)
        if not np.all(np.isfinite(M)):
            trace.append(
                Step(
                    c + 1, None, None, info=_stage_info(M, c, pivot, row_swap, mult, False, **more)
                )
            )
            return _fail(method, f"non-finite value at stage {c + 1}", trace)
        trace.append(
            Step(c + 1, None, None, info=_stage_info(M, c, pivot, row_swap, mult, False, **more))
        )

    U = np.triu(M[:, :n])
    x = back_substitution(U, M[:, n])
    r = b - A @ x
    rnorm = norm2(r)
    if not np.all(np.isfinite(x)):
        return _fail(method, "non-finite value in back substitution", trace)
    trace.append(
        Step(
            n + 1,
            x,
            rnorm,
            info={
                "phase": "back_substitution",
                "matrix": M.copy(),
                "pivot": None,
                "pivot_value": None,
                "row_swap": None,
                "multipliers": None,
                "zero_pivot": False,
                "residual": r,
                **({"perm": perm.copy()} if pivoting else {}),
            },
        )
    )
    solve, solve_t = _lu_solvers(L, U, perm)
    extra: dict[str, Any] = {
        "cond_estimate": cond_estimate(A, solve, solve_t),
        "residual_norm": rnorm,
        "growth_factor": growth / a_max,
        "L": L,
        "U": U,
    }
    if pivoting:
        extra["perm"] = perm
        hint = ": partial pivoting is unstable on this matrix; use Householder QR"
    else:
        hint = ": a small pivot made the growth large; use partial pivoting"
    return _certify(method, A, b, x, r, trace, extra, hint)


@register(
    id="gaussian_elimination",
    family="linalg",
    name="Gaussian elimination (no pivoting)",
    needs=("A", "b"),
    order="direct, 2n³/3 flops",
    summary="Subtract multiples of each pivot row to zero the column below it, then back-substitute.",
    references=(
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 20.1",
        "Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.2.1 and 3.1.2",
    ),
)
def gaussian_elimination(problem: SystemLike) -> Result:
    """Gaussian elimination without pivoting, then back substitution.

    Stage c = 0 … n−1 (Trefethen & Bau Alg. 20.1, outer-product form GVL Alg. 3.2.1):

        m_ic = a_ic / a_cc,   row_i ← row_i − m_ic·row_c   (i > c)

    applied to the augmented matrix [A | b]; then U x = b̃ by back substitution (GVL Alg. 3.1.2).
    Without row interchanges the method breaks down on a zero pivot even when A is nonsingular
    (``needs_pivoting``) and is unstable when a pivot is small (large growth factor).

    Stopping: none (finite). converged=False when a pivot fails |a_cc| > τ, the last pivot
    included, when a value becomes non-finite, or when the backward error η∞ of x fails the test of
    the module docstring (a small pivot that passed |a_cc| > τ can still give a growth factor of
    1e15 and a wrong x; x is then returned with the message "unstable").
    extra: cond_estimate, growth_factor, backward_error, L, U.
    """
    return _gaussian("gaussian_elimination", problem, pivoting=False)


@register(
    id="gaussian_elimination_pivoting",
    family="linalg",
    name="Gaussian elimination (partial pivoting)",
    needs=("A", "b"),
    order="direct, 2n³/3 flops",
    summary="Before each stage, swap up the row with the largest entry in the pivot column.",
    references=(
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 21.1",
        "Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.4.1",
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.2",
    ),
)
def gaussian_elimination_pivoting(problem: SystemLike) -> Result:
    """Gaussian elimination with partial (row) pivoting, then back substitution.

    Stage c: choose p = argmax_{i ≥ c} |a_ic| (the first index on ties), swap rows c and p of
    [A | b], then eliminate as in :func:`gaussian_elimination`. All multipliers satisfy
    |m_ic| ≤ 1, which bounds the growth factor by 2^{n−1} (Higham 2002, Ch. 9) and makes the
    method backward stable in practice.

    Stopping: none (finite). converged=False when the largest available pivot fails |p| > τ
    (A is singular to working precision) or when η∞ fails the backward-error test (a growth
    factor near 2^{n−1}). extra: cond_estimate, growth_factor, backward_error, L, U, perm with
    A[perm] = L·U.
    """
    return _gaussian("gaussian_elimination_pivoting", problem, pivoting=True)


# --------------------------------------------------------------------------------------
# Gauss–Jordan elimination
# --------------------------------------------------------------------------------------


@register(
    id="gauss_jordan",
    family="linalg",
    name="Gauss–Jordan elimination",
    needs=("A", "b"),
    order="direct, n³ flops",
    summary="Normalize each pivot row and clear its column both below and above: [A | b] → [I | x].",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §6.1 (Gauss–Jordan method)",
        "Higham, Accuracy and Stability of Numerical Algorithms (2nd ed., 2002), Ch. 14",
    ),
)
@quiet
def gauss_jordan(problem: SystemLike) -> Result:
    """Gauss–Jordan elimination with partial pivoting.

    Stage c: choose p = argmax_{i ≥ c} |a_ic|, swap rows c and p, divide row c by the pivot,
    then row_i ← row_i − a_ic·row_c for every i ≠ c. After n stages [A | b] = [I | x], so no back
    substitution is needed; the cost is n³ flops against 2n³/3 for Gaussian elimination.

    # NOTE: partial pivoting is added (Burden & Faires state the method without it) because
    # Gauss–Jordan without pivoting fails on the same zero pivots as plain elimination.

    The pivot search and the updates of the rows below the pivot are those of
    :func:`gaussian_elimination_pivoting`, so the method records the same factors P·A = L·U
    (U row c = the pivot row before it is normalized); they are used for ``cond_estimate``.

    Stopping: none (finite). converged=False when |p| ≤ τ (A singular to working precision) or
    when η∞ > 30·n·ε·max(1, κ̂₁) (module docstring). The last stage step carries x and the
    residual norm. extra: cond_estimate, growth_factor, backward_error.
    """
    method = "gauss_jordan"
    A, b = resolve_system(problem)
    n = b.size
    tau = pivot_tolerance(A)
    a_max = float(np.max(np.abs(A)))
    M = np.hstack([A, b[:, None]])
    L = np.eye(n)
    U = np.zeros((n, n))
    perm = np.arange(n)
    growth = a_max
    trace = [Step(0, None, None, info=_start_info(M, perm=perm.copy()))]

    for c in range(n):
        row_swap: list[int] | None = None
        p = c + int(np.argmax(np.abs(M[c:, c])))
        if p != c:
            M[[c, p]] = M[[p, c]]
            L[[c, p], :c] = L[[p, c], :c]
            perm[[c, p]] = perm[[p, c]]
            row_swap = [c, p]
        pivot = float(M[c, c])
        mult = np.zeros(n)
        if abs(pivot) <= tau:
            trace.append(
                Step(
                    c + 1,
                    None,
                    None,
                    info=_stage_info(M, c, pivot, row_swap, mult, True, perm=perm.copy()),
                )
            )
            return _fail(
                method,
                f"zero pivot |a_{c}{c}| = {abs(pivot):.3g} ≤ τ = {tau:.3g} at stage {c + 1}: "
                "A is singular to working precision",
                trace,
                {"growth_factor": growth / a_max if a_max > 0 else None},
            )
        U[c, c:] = M[c, c:n]
        L[c + 1 :, c] = M[c + 1 :, c] / pivot
        M[c, :] /= pivot
        M[c, c] = 1.0  # NOTE: exact 1 instead of p/p (which is 1 in IEEE arithmetic anyway).
        others = np.arange(n) != c
        mult[others] = M[others, c]
        M[others, :] -= np.outer(mult[others], M[c, :])
        M[others, c] = 0.0  # NOTE: exact zeros, as in _gaussian.
        # NOTE: the growth factor uses the pivot row before it is normalized (U row c) and the
        # other rows after the update; the normalized row a_cj/a_cc is not on the scale of A.
        growth = max(growth, float(np.max(np.abs(U[c, c:]))), _growth(0.0, M[others], n))
        if not np.all(np.isfinite(M)):
            trace.append(
                Step(
                    c + 1,
                    None,
                    None,
                    info=_stage_info(M, c, pivot, row_swap, mult, False, perm=perm.copy()),
                )
            )
            return _fail(method, f"non-finite value at stage {c + 1}", trace)
        x_c: Vector | None = None
        rnorm_c: float | None = None
        more: dict[str, Any] = {"perm": perm.copy()}
        if c == n - 1:
            # The last stage leaves [I | x]: its Step carries the solution and the residual.
            x_c = M[:, n].copy()
            more["residual"] = b - A @ x_c
            rnorm_c = norm2(more["residual"])
        info = _stage_info(M, c, pivot, row_swap, mult, False, **more)
        trace.append(Step(c + 1, x_c, rnorm_c, info=info))

    x = M[:, n].copy()
    r = b - A @ x
    rnorm = norm2(r)
    solve, solve_t = _lu_solvers(L, U, perm)
    kappa = cond_estimate(A, solve, solve_t)
    extra: dict[str, Any] = {
        "cond_estimate": kappa,
        "residual_norm": rnorm,
        "growth_factor": growth / a_max,
        "perm": perm,
    }
    hint = ": Gauss–Jordan is not backward stable; use Householder QR"
    return _certify(method, A, b, x, r, trace, extra, hint, kappa=kappa)


# --------------------------------------------------------------------------------------
# LU factorization
# --------------------------------------------------------------------------------------


@register(
    id="lu_decomposition",
    family="linalg",
    name="LU factorization (Doolittle, partial pivoting)",
    needs=("A", "b"),
    order="direct, 2n³/3 flops + 2n² per solve",
    summary="Factor P·A = L·U once (unit lower L, upper U), then solve L y = P b and U x = y.",
    references=(
        "Golub & Van Loan, Matrix Computations (4th ed., 2013), Alg. 3.4.1",
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.4",
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 21.1",
    ),
)
@quiet
def lu_decomposition(problem: SystemLike) -> Result:
    """Doolittle LU factorization with partial pivoting, P·A = L·U, then two triangular solves.

    Stage c: p = argmax_{i ≥ c} |w_ic|; swap rows c, p of the working matrix W and of the
    finished part of L; l_ic = w_ic / w_cc (i > c); W[c+1:, c:] −= l·W[c, c:]. After n stages
    W = U. Then L y = P b (forward substitution, unit diagonal) and U x = y (back substitution).

    # NOTE: the factors are computed in the right-looking (outer-product) order of GVL Alg. 3.4.1
    # rather than the compact row-by-row Doolittle scheme of Burden & Faires Alg. 6.4. Both give
    # the same unit-lower L and upper U in exact arithmetic; the outer-product order makes each
    # stage a visible rank-1 update of the Schur complement.

    Convention: ``extra["P"]`` satisfies P·A = L·U. ``scipy.linalg.lu`` returns A = P_s·L·U,
    so P_s = Pᵀ.

    Stopping: none (finite). converged=False when |p| ≤ τ (A singular to working precision) or
    when η∞ fails the backward-error test. extra: P, L, U, perm, cond_estimate, growth_factor,
    backward_error.
    """
    method = "lu_decomposition"
    A, b = resolve_system(problem)
    n = b.size
    tau = pivot_tolerance(A)
    a_max = float(np.max(np.abs(A)))
    W = A.copy()
    L = np.eye(n)
    perm = np.arange(n)
    growth = a_max
    trace = [Step(0, None, None, info=_start_info(W, L=L.copy(), perm=perm.copy()))]

    for c in range(n):
        row_swap: list[int] | None = None
        p = c + int(np.argmax(np.abs(W[c:, c])))
        if p != c:
            W[[c, p]] = W[[p, c]]
            L[[c, p], :c] = L[[p, c], :c]
            perm[[c, p]] = perm[[p, c]]
            row_swap = [c, p]
        pivot = float(W[c, c])
        mult = np.zeros(n)
        if abs(pivot) <= tau:
            trace.append(
                Step(
                    c + 1,
                    None,
                    None,
                    info=_stage_info(
                        W, c, pivot, row_swap, mult, True, L=L.copy(), perm=perm.copy()
                    ),
                )
            )
            return _fail(
                method,
                f"zero pivot |u_{c}{c}| = {abs(pivot):.3g} ≤ τ = {tau:.3g} at stage {c + 1}: "
                "A is singular to working precision",
                trace,
                {"growth_factor": growth / a_max if a_max > 0 else None},
            )
        mult[c + 1 :] = W[c + 1 :, c] / pivot
        L[c + 1 :, c] = mult[c + 1 :]
        W[c + 1 :, c:] -= np.outer(mult[c + 1 :], W[c, c:])
        W[c + 1 :, c] = 0.0  # NOTE: exact zeros, as in _gaussian.
        growth = _growth(growth, W, n)
        info = _stage_info(W, c, pivot, row_swap, mult, False, L=L.copy(), perm=perm.copy())
        trace.append(Step(c + 1, None, None, info=info))
        if not np.all(np.isfinite(W)):
            return _fail(method, f"non-finite value at stage {c + 1}", trace)

    U = np.triu(W)
    y = forward_substitution(L, b[perm], unit=True)
    x = back_substitution(U, y)
    if not np.all(np.isfinite(x)):
        return _fail(method, "non-finite value in the triangular solves", trace)
    r = b - A @ x
    rnorm = norm2(r)
    trace.append(
        Step(
            n + 1,
            x,
            rnorm,
            info={
                "phase": "solve",
                "matrix": U.copy(),
                "pivot": None,
                "pivot_value": None,
                "row_swap": None,
                "multipliers": None,
                "zero_pivot": False,
                "L": L.copy(),
                "perm": perm.copy(),
                "y": y,
                "residual": r,
            },
        )
    )
    solve, solve_t = _lu_solvers(L, U, perm)
    P = np.eye(n)[perm]
    extra: dict[str, Any] = {
        "cond_estimate": cond_estimate(A, solve, solve_t),
        "residual_norm": rnorm,
        "growth_factor": growth / a_max,
        "P": P,
        "L": L,
        "U": U,
        "perm": perm,
    }
    hint = ": partial pivoting is unstable on this matrix; use Householder QR"
    return _certify(method, A, b, x, r, trace, extra, hint)


# --------------------------------------------------------------------------------------
# Cholesky
# --------------------------------------------------------------------------------------


@register(
    id="cholesky",
    family="linalg",
    name="Cholesky factorization",
    needs=("A", "b"),
    order="direct, n³/3 flops",
    summary="Factor a symmetric positive definite A = L·Lᵀ, then solve L y = b and Lᵀ x = y.",
    references=(
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 23.1",
        "Golub & Van Loan, Matrix Computations (4th ed., 2013), §4.2 (outer-product Cholesky)",
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.6",
    ),
)
@quiet
def cholesky(problem: SystemLike) -> Result:
    """Outer-product Cholesky factorization A = L·Lᵀ, then two triangular solves.

    Stage c (Trefethen & Bau Alg. 23.1, written for the lower factor):

        d = w_cc;  l_cc = √d;  l_ic = w_ic / l_cc (i > c);  W[c+1:, c+1:] −= l·lᵀ

    where W is the current Schur complement. A symmetric A is positive definite exactly when
    every d is positive (GVL §4.2), so the factorization itself is the SPD test; no
    pivoting is needed, and the method is backward stable (Higham 2002, Ch. 10).

    Fails cleanly (converged=False, k = 0 only) when A is not symmetric, and at stage c when
    d ≤ τ: d ≤ 0 means "A is not positive definite" (the computed Schur complement has a
    nonpositive diagonal entry); 0 < d ≤ τ means "A is not numerically positive definite" (A may
    be SPD, but it is singular or too ill-conditioned to working precision).
    Stopping: none (finite); converged=True also needs the backward-error test of the module
    docstring. extra: L, cond_estimate, backward_error.
    """
    method = "cholesky"
    A, b = resolve_system(problem)
    n = b.size
    tau = pivot_tolerance(A)
    W = A.copy()
    L = np.zeros((n, n))
    trace = [Step(0, None, None, info={**_start_info(W), "L": L.copy(), "column": None})]
    if not is_symmetric(A):
        asym = float(np.max(np.abs(A - A.T)))
        return _fail(
            method,
            f"A is not symmetric (max |a_ij − a_ji| = {asym:.3g}); Cholesky needs an SPD matrix",
            trace,
        )

    for c in range(n):
        d = float(W[c, c])
        if not d > tau:
            info = {**_stage_info(W, c, d, None, np.zeros(n), True), "L": L.copy(), "column": None}
            trace.append(Step(c + 1, None, None, info=info))
            pivot = f"pivot d_{c} = a_{c}{c} − Σ l_{c}j² = {d:.3g}"
            if d > 0.0:
                # NOTE: 0 < d ≤ τ is the numerical-rank heuristic of the module docstring, not a
                # proof of indefiniteness: diag(1, 1e-17) is SPD but its second pivot is ≤ τ.
                why = (
                    f"{pivot} lies in (0, τ = {tau:.3g}] at stage {c + 1}: A is not numerically "
                    "positive definite (singular or too ill-conditioned for working precision)"
                )
            else:
                why = f"A is not positive definite: {pivot} ≤ 0 at stage {c + 1}"
            return _fail(method, why, trace)
        l_cc = float(np.sqrt(d))
        L[c, c] = l_cc
        L[c + 1 :, c] = W[c + 1 :, c] / l_cc
        col = L[c + 1 :, c]
        W[c + 1 :, c + 1 :] -= np.outer(col, col)
        W[c, :] = 0.0
        W[:, c] = 0.0
        info = {
            **_stage_info(W, c, d, None, np.zeros(n), False),
            "multipliers": None,
            "L": L.copy(),
            "column": L[:, c].copy(),
        }
        trace.append(Step(c + 1, None, None, info=info))
        if not np.all(np.isfinite(W)):
            return _fail(method, f"non-finite value at stage {c + 1}", trace)

    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    if not np.all(np.isfinite(x)):
        return _fail(method, "non-finite value in the triangular solves", trace)
    r = b - A @ x
    rnorm = norm2(r)
    trace.append(
        Step(
            n + 1,
            x,
            rnorm,
            info={
                **_start_info(L.T),
                "phase": "solve",
                "L": L.copy(),
                "column": None,
                "y": y,
                "residual": r,
            },
        )
    )

    def solve(v: Vector) -> Vector:
        return back_substitution(L.T, forward_substitution(L, v))

    extra: dict[str, Any] = {
        "cond_estimate": cond_estimate(A, solve, solve),  # A = Aᵀ
        "residual_norm": rnorm,
        "L": L,
    }
    return _certify(method, A, b, x, r, trace, extra, "")


# --------------------------------------------------------------------------------------
# Householder QR
# --------------------------------------------------------------------------------------


@register(
    id="qr_householder",
    family="linalg",
    name="QR factorization (Householder)",
    needs=("A", "b"),
    order="direct, 4n³/3 flops",
    summary="Reflect each column onto the axis to get A = Q·R, then solve R x = Qᵀ b.",
    references=(
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 10.1 and 10.2",
        "Golub & Van Loan, Matrix Computations (4th ed., 2013), §5.1–5.2",
    ),
)
@quiet
def qr_householder(problem: SystemLike) -> Result:
    """Householder QR applied to [A | b], then back substitution R x = Qᵀ b.

    Stage c (Trefethen & Bau Alg. 10.1, with Qᵀb formed implicitly as in Alg. 10.2):

        x = R[c:, c];  v = x + sign(x₁)‖x‖₂ e₁;  v ← v/‖v‖₂;  R[c:, c:] −= 2 v (vᵀ R[c:, c:])

    and the same reflection H = I − 2vvᵀ is applied to the right-hand-side column. The sign
    choice avoids cancellation in v₁ and gives r_cc = −sign(x₁)‖x‖₂ (sign(0) = +1).

    # NOTE: when x[1:] = 0 exactly (always true for the last column) no reflection is applied
    # (H = I), as in LAPACK ``xLARFG`` (τ = 0). Trefethen & Bau would reflect anyway and flip the
    # sign of r_cc. With this choice R equals LAPACK's R (``scipy.linalg.qr``) up to rounding.

    Q = H₀H₁⋯H_{n−1} is formed only for ``extra`` (Trefethen & Bau Alg. 10.3 applied to I).
    QR is backward stable for any A without pivoting (Higham 2002, Ch. 19).

    Stopping: none (finite). converged=False when |r_cc| ≤ τ (A singular to working precision) or
    when η∞ fails the backward-error test. extra: Q, R, householder_vectors, cond_estimate,
    backward_error.
    """
    method = "qr_householder"
    A, b = resolve_system(problem)
    n = b.size
    tau = pivot_tolerance(A)
    M = np.hstack([A, b[:, None]])  # [R | Qᵀb] in progress
    V = np.zeros((n, n))  # row c: the unit Householder vector of stage c (zeros above c)
    trace = [Step(0, None, None, info={**_start_info(M), "householder_vector": None})]

    for c in range(n):
        xcol = M[c:, c].copy()
        sigma = norm2(xcol[1:])
        v = np.zeros(n)
        if sigma > 0.0:
            alpha = float(np.hypot(xcol[0], sigma))  # ‖x‖₂ without overflow
            sign = 1.0 if xcol[0] >= 0.0 else -1.0
            vc = xcol.copy()
            vc[0] += sign * alpha
            vc /= norm2(vc)
            M[c:, c:] -= 2.0 * np.outer(vc, vc @ M[c:, c:])
            # NOTE: exact values for the reflected column: H x = −sign(x₁)‖x‖ e₁.
            M[c, c] = -sign * alpha
            M[c + 1 :, c] = 0.0
            v[c:] = vc
        V[c] = v
        r_cc = float(M[c, c])
        zero = abs(r_cc) <= tau
        info = {
            **_stage_info(M, c, r_cc, None, np.zeros(n), zero),
            "multipliers": None,
            "householder_vector": v,
        }
        trace.append(Step(c + 1, None, None, info=info))
        if zero:
            return _fail(
                method,
                f"zero diagonal |r_{c}{c}| = {abs(r_cc):.3g} ≤ τ = {tau:.3g} at stage {c + 1}: "
                "A is singular to working precision",
                trace,
            )
        if not np.all(np.isfinite(M)):
            return _fail(method, f"non-finite value at stage {c + 1}", trace)

    R = np.triu(M[:, :n])
    qtb = M[:, n].copy()
    x = back_substitution(R, qtb)
    if not np.all(np.isfinite(x)):
        return _fail(method, "non-finite value in back substitution", trace)
    r = b - A @ x
    rnorm = norm2(r)
    trace.append(
        Step(
            n + 1,
            x,
            rnorm,
            info={
                **_start_info(M),
                "phase": "back_substitution",
                "householder_vector": None,
                "residual": r,
            },
        )
    )
    # Q = H_0 H_1 ⋯ H_{n−1} I, applied from the last reflector to the first.
    Q = np.eye(n)
    for c in range(n - 1, -1, -1):
        vc = V[c, c:]
        Q[c:, :] -= 2.0 * np.outer(vc, vc @ Q[c:, :])

    def solve(w: Vector) -> Vector:  # A⁻¹w = R⁻¹ Qᵀ w
        return back_substitution(R, Q.T @ w)

    def solve_t(w: Vector) -> Vector:  # A⁻ᵀw = Q R⁻ᵀ w
        return Q @ forward_substitution(R.T, w)

    extra: dict[str, Any] = {
        "cond_estimate": cond_estimate(A, solve, solve_t),
        "residual_norm": rnorm,
        "Q": Q,
        "R": R,
        "householder_vectors": V,
    }
    return _certify(method, A, b, x, r, trace, extra, "")


# --------------------------------------------------------------------------------------
# Thomas algorithm (tridiagonal)
# --------------------------------------------------------------------------------------


@register(
    id="thomas",
    family="linalg",
    name="Thomas algorithm (tridiagonal)",
    needs=("A", "b"),
    order="direct, 8n flops",
    summary="Gaussian elimination specialized to a tridiagonal matrix: one forward sweep, one back sweep.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 6.7",
        "Higham, Accuracy and Stability of Numerical Algorithms (2nd ed., 2002), Ch. 9",
    ),
)
@quiet
def thomas(problem: SystemLike) -> Result:
    """Thomas algorithm (Crout factorization of a tridiagonal matrix, no pivoting).

    With sub-diagonal a_i (i ≥ 1), diagonal d_i and super-diagonal c_i (i ≤ n−2), stage i is

        w_i = d_i − a_i c′_{i−1},   c′_i = c_i / w_i,   d′_i = (b_i − a_i d′_{i−1}) / w_i

    (w₀ = d₀, d′₀ = b₀/w₀): row i loses its sub-diagonal entry (multiplier a_i, because row i−1
    is already normalized) and is divided by w_i. The back sweep is x_{n−1} = d′_{n−1},
    x_i = d′_i − c′_i x_{i+1}. This is A = L·U with L lower bidiagonal (diagonal w, sub-diagonal
    a) and U unit upper bidiagonal (super-diagonal c′), as in Burden & Faires Alg. 6.7.

    Without pivoting the method can meet a zero w_i; it cannot when A is strictly diagonally
    dominant or SPD (Higham 2002, Ch. 9).

    Raises ValueError when A is not tridiagonal (an entry with |i − j| > 1 is nonzero).
    Stopping: none (finite). converged=False when |w_i| ≤ τ or when η∞ fails the backward-error
    test (a small w_i that passed |w_i| > τ). extra: L, U, c_prime, d_prime, cond_estimate,
    backward_error.
    """
    method = "thomas"
    A, b = resolve_system(problem)
    n = b.size
    i_idx, j_idx = np.indices((n, n))
    if np.any(A[np.abs(i_idx - j_idx) > 1] != 0.0):
        raise ValueError("thomas: A must be tridiagonal (a_ij = 0 for |i − j| > 1)")
    tau = pivot_tolerance(A)
    sub = np.concatenate([[0.0], np.diag(A, -1)])  # a_i, a_0 unused
    diag = np.diag(A).copy()
    sup = np.concatenate([np.diag(A, 1), [0.0]])  # c_i, c_{n−1} unused
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    w = np.zeros(n)
    M = np.hstack([A, b[:, None]])
    trace = [Step(0, None, None, info={**_start_info(M), "c_prime": None, "d_prime": None})]

    for i in range(n):
        mult = np.zeros(n)
        if i == 0:
            wi = float(diag[0])
            rhs = float(b[0])
        else:
            mult[i] = sub[i]
            wi = float(diag[i] - sub[i] * c_prime[i - 1])
            rhs = float(b[i] - sub[i] * d_prime[i - 1])
        if abs(wi) <= tau:
            info = {
                **_stage_info(M, i, wi, None, mult, True),
                "c_prime": c_prime.copy(),
                "d_prime": d_prime.copy(),
            }
            trace.append(Step(i + 1, None, None, info=info))
            return _fail(
                method,
                f"zero pivot |w_{i}| = {abs(wi):.3g} ≤ τ = {tau:.3g} at stage {i + 1}: the "
                "Thomas algorithm does not pivot; use Gaussian elimination with pivoting",
                trace,
            )
        w[i] = wi
        if i < n - 1:
            c_prime[i] = sup[i] / wi
        d_prime[i] = rhs / wi
        # Row i of [A | b] after the stage: (0 … 0, 1, c′_i, 0 … 0 | d′_i).
        if i > 0:
            M[i, i - 1] = 0.0
        M[i, i] = 1.0
        if i < n - 1:
            M[i, i + 1] = c_prime[i]
        M[i, n] = d_prime[i]
        info = {
            **_stage_info(M, i, wi, None, mult, False),
            "c_prime": c_prime.copy(),
            "d_prime": d_prime.copy(),
        }
        trace.append(Step(i + 1, None, None, info=info))
        if not (np.isfinite(c_prime[i]) and np.isfinite(d_prime[i])):
            return _fail(method, f"non-finite value at stage {i + 1}", trace)

    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    if not np.all(np.isfinite(x)):
        return _fail(method, "non-finite value in back substitution", trace)
    r = b - A @ x
    rnorm = norm2(r)
    trace.append(
        Step(
            n + 1,
            x,
            rnorm,
            info={
                **_start_info(M),
                "phase": "back_substitution",
                "c_prime": c_prime.copy(),
                "d_prime": d_prime.copy(),
                "residual": r,
            },
        )
    )
    L = np.diag(w) + np.diag(sub[1:], -1)
    U = np.eye(n) + np.diag(c_prime[: n - 1], 1)

    def solve(v: Vector) -> Vector:
        return back_substitution(U, forward_substitution(L, v))

    def solve_t(v: Vector) -> Vector:  # Aᵀ = Uᵀ Lᵀ
        return back_substitution(L.T, forward_substitution(U.T, v, unit=True))

    extra: dict[str, Any] = {
        "cond_estimate": cond_estimate(A, solve, solve_t),
        "residual_norm": rnorm,
        "L": L,
        "U": U,
        "c_prime": c_prime,
        "d_prime": d_prime,
    }
    hint = ": the Thomas algorithm does not pivot; use Gaussian elimination with pivoting"
    return _certify(method, A, b, x, r, trace, extra, hint)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("gaussian_elimination", "needs_pivoting", {}),
    ("gaussian_elimination", "diag_dominant_3", {}),
    ("gaussian_elimination_pivoting", "needs_pivoting", {}),
    ("gauss_jordan", "diag_dominant_3", {}),
    ("lu_decomposition", "nonsymmetric_4", {}),
    ("cholesky", "hilbert_5", {}),
    ("qr_householder", "nonsymmetric_4", {}),
    ("thomas", "poisson_1d_10", {}),
]
