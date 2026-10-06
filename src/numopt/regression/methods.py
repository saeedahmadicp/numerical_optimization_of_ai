"""Regression of data (x_i, y_i), i = 1..m, on polynomials in x (mostly the line β₀ + β₁x).

Coefficients are always in increasing powers, β = [β₀, β₁, ..., β_d], so the model is
ŷ(x) = Σ_j β_j x^j (``numpy.polyfit`` returns the reverse order).

* ``Result.x``: β.
* ``Result.fun``: the objective that the method minimizes: RSS = Σ r_i² for least squares,
  RSS + λ‖β_{1:}‖² for ridge, Σ ρ_δ(r_i/σ̂) for Huber, Σ |r_i| for LAD, max |r_i| for the
  minimax line. Theil–Sen minimizes no objective; its ``fun`` is the RSS (for comparison).
* ``Result.extra``::

      coefficients:   β (same as Result.x)
      degree:         d (number of parameters p = d + 1)
      fitted:         [m] ŷ_i;  residuals: [m] r_i = y_i - ŷ_i
      rss, tss:       Σ r_i², Σ (y_i - ȳ)²
      r_squared:      1 - RSS/TSS (None when y is constant to working precision,
                      max_i |y_i - ȳ| ≤ m·ε·max_i |y_i|, which includes TSS = 0;
                      negative when the fit is worse than ȳ, which can happen for every
                      method except OLS with an intercept)
      adj_r_squared:  1 - (1 - R²)(m - 1)/(m - p) (None when m ≤ p or R² is None)
      rmse:           sqrt(RSS/m)
      sigma:          residual standard error sqrt(RSS/(m - p)) (None when m ≤ p)
      std_errors:     [p] OLS standard errors σ̂·sqrt([(XᵀX)⁻¹]_jj) for the least-squares
                      methods with full rank and m > p; None for the other methods
      eval:           {x: [200] grid over the domain, y: [200] the model on the grid}

  plus method-specific keys (documented on each method).

Numerics: no normal-equations solve is used except by the explicit ``normal_equations``
choice of :func:`linear_regression` (kept to demonstrate κ(XᵀX) = κ(X)², Higham 2002,
§20.4). Least squares uses Householder QR of the column-equilibrated design (van der
Sluis 1969: unit-norm columns bring κ₂ within a factor √p of its minimum over column
scalings); the scaling does not change the least-squares solution. The rank is the number
of singular values σ_k > max(m, p)·ε·σ_max of the factored (equilibrated) matrix (the rule
of ``numpy.linalg.matrix_rank``). The robust IRLS lines solve in the centered variable
x - x̄ (see :class:`_LineDesign`). ``n_fev`` is 0: the methods read data, they evaluate
no function.

Info keys:
    solver: str           least squares: "qr" | "normal_equations" | "svd"
    rank: int             numerical rank of the matrix that was factored
    cond: float           κ₂ of the matrix that was factored (∞ if singular)
    cond_gram: float      normal_equations: κ₂(AᵀA) = κ₂(A)², the condition number the
                          Cholesky solve actually sees
    lambda: float         ridge: the penalty λ
    weights: [m]          IRLS (huber, lad): weights w_i at the current β, used for the
                          next weighted solve
    scale: float          huber: the fixed robust residual scale σ̂
    delta: float          huber: the threshold δ (in units of σ̂)
    smoothed_objective: float  lad: Σ ρ_ε(r_i), the function IRLS decreases monotonically
    slopes: [≤ m(m-1)/2]  theil_sen: pairwise slopes (y_j - y_i)/(x_j - x_i), x_i ≠ x_j, sorted
    reference: [3]        minimax: indices (into the input order) of the reference points
    level: float          minimax: levelled error h on the reference
                          (y - ŷ = ±h alternating there)
    entering: int | None  minimax: index of the point of maximum deviation that enters
                          the reference next (None when the method stops)
    max_deviation: float  minimax: max_i |r_i| at the current β
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import Dataset, Result, Step, Vector

#: Number of points on the plotting grid in ``extra.eval``.
N_GRID = 200
#: Φ⁻¹(3/4): MAD / Φ⁻¹(3/4) estimates σ for Gaussian data (normalized MAD).
MAD_TO_SIGMA = 0.6744897501960817
_EPS = float(np.finfo(np.float64).eps)
#: Smallest subnormal 2⁻¹⁰⁷⁴: the absolute error term η of the standard model with
#: gradual underflow, fl(x op y) = (x op y)(1 + δ) + η (Higham 2002, eq. (2.8)).
_ETA = float(np.finfo(np.float64).smallest_subnormal)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _quiet(fn: Callable[..., Result]) -> Callable[..., Result]:
    """Run a method with floating-point warnings off; overflow is reported in the Result."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Result:
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class _Data:
    x: Vector
    y: Vector
    a: float
    b: float

    @property
    def m(self) -> int:
        return int(self.x.size)


def _resolve(problem: Dataset | tuple[Any, Any], min_points: int) -> _Data:
    """Accept a :class:`Dataset` or an ``(x, y)`` pair; validate the data."""
    if isinstance(problem, Dataset):
        x_raw, y_raw, domain = problem.x, problem.y, problem.domain
    elif isinstance(problem, tuple | list) and len(problem) == 2:
        x_raw, y_raw, domain = problem[0], problem[1], None
    else:
        raise TypeError("problem must be a numopt Dataset or an (x, y) pair of arrays")
    x = np.array(x_raw, dtype=np.float64, copy=True)
    y = np.array(y_raw, dtype=np.float64, copy=True)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(f"x and y must be 1-D of equal length; got {x.shape} and {y.shape}")
    if x.size < min_points:
        raise ValueError(f"need at least {min_points} data points, got {x.size}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("data contain NaN or infinite values")
    if domain is not None:
        a, b = float(domain[0]), float(domain[1])
    else:
        a, b = float(np.min(x)), float(np.max(x))
    if not a < b:
        a, b = a - 1.0, b + 1.0
    return _Data(x, y, a, b)


def _vandermonde(x: Vector, degree: int) -> Vector:
    """V_{ij} = x_i^j, j = 0..degree (increasing powers), built by repeated multiplication."""
    v = np.empty((x.size, degree + 1), dtype=np.float64)
    v[:, 0] = 1.0
    for j in range(1, degree + 1):
        v[:, j] = v[:, j - 1] * x
    return v


def _polyval(beta: Vector, t: Vector) -> Vector:
    """Horner evaluation of Σ_j β_j t^j."""
    out = np.full_like(t, beta[-1], dtype=np.float64)
    for j in range(beta.size - 2, -1, -1):
        out = beta[j] + t * out
    return out


def _column_scales(a: Vector) -> Vector:
    """2-norms of the columns of ``a`` (1 for a zero column)."""
    s = np.sqrt(np.sum(a * a, axis=0))
    return np.where(s > 0.0, s, 1.0)


def _back_substitute(r: Vector, z: Vector) -> Vector:
    """Solve the upper-triangular system R β = z (Golub & Van Loan, Alg. 3.1.2)."""
    p = z.size
    beta = np.empty(p, dtype=np.float64)
    for i in range(p - 1, -1, -1):
        beta[i] = (z[i] - float(np.dot(r[i, i + 1 :], beta[i + 1 :]))) / r[i, i]
    return beta


def _forward_substitute(lo: Vector, z: Vector) -> Vector:
    """Solve the lower-triangular system L w = z (Golub & Van Loan, Alg. 3.1.1)."""
    p = z.size
    w = np.empty(p, dtype=np.float64)
    for i in range(p):
        w[i] = (z[i] - float(np.dot(lo[i, :i], w[:i]))) / lo[i, i]
    return w


def _rank_cond(sv: Vector, shape: tuple[int, int]) -> tuple[int, float]:
    """Numerical rank (matrix_rank rule) and κ₂ from the singular values ``sv``.

    A wide matrix (fewer rows than columns) has a null space, so κ₂ = σ_max/σ_p = ∞.
    """
    smax = float(sv[0]) if sv.size else 0.0
    tol = max(shape) * _EPS * smax
    rank = int(np.sum(sv > tol))
    full = sv.size == shape[1] and sv.size > 0 and sv[-1] > 0.0
    cond = smax / float(sv[-1]) if full else math.inf
    return rank, cond


def _qr_lstsq(a: Vector, rhs: Vector) -> tuple[Vector, Vector]:
    """Least squares min ‖A β - rhs‖₂ by Householder QR (Golub & Van Loan, Alg. 5.3.2).

    Returns (β, R). The caller has checked full column rank.
    """
    q, r = np.linalg.qr(a, mode="reduced")
    return _back_substitute(r, q.T @ rhs), r


def _inv_diag_from_triangle(r: Vector, upper: bool) -> Vector:
    """diag((RᵀR)⁻¹) = ‖R⁻ᵀ e_j‖² (upper R) or diag((L Lᵀ)⁻¹) = ‖L⁻¹ e_j‖² (lower L).

    One triangular solve per column; no inverse is formed. Uses (AᵀA)⁻¹ = R⁻¹R⁻ᵀ for
    A = QR (Björck, Numerical Methods for Least Squares Problems, 1996, Ch. 2).
    """
    p = r.shape[0]
    out = np.empty(p, dtype=np.float64)
    for j in range(p):
        e = np.zeros(p, dtype=np.float64)
        e[j] = 1.0
        # (RᵀR)⁻¹_jj = ‖R⁻ᵀ e_j‖²: solve Rᵀ z = e_j (lower-triangular).
        z = _forward_substitute(r.T, e) if upper else _forward_substitute(r, e)
        out[j] = float(np.dot(z, z))
    return out


def _goodness(data: _Data, beta: Vector, n_params: int, cov_diag: Vector | None) -> dict[str, Any]:
    """Residual statistics and the plotting curve for a fitted polynomial β."""
    fitted = _polyval(beta, data.x)
    resid = data.y - fitted
    rss = float(np.dot(resid, resid))
    ybar = float(np.mean(data.y))
    dev = data.y - ybar
    tss = float(np.dot(dev, dev))
    m = data.m
    # NOTE: R² is undefined for constant y (TSS = 0). In floating point ȳ is rounded,
    # by up to (m - 1)u‖y‖∞ for any summation order, so constant data give TSS ~ u²‖y‖²
    # instead of 0, and data whose spread is at that level give an R² with no correct
    # digits (e.g. R² = -38 for y = [0.2, 0.2, 0.2]). Both are reported as None.
    flat = float(np.max(np.abs(dev))) <= m * _EPS * float(np.max(np.abs(data.y)))
    r2 = 1.0 - rss / tss if (tss > 0.0 and not flat) else None
    dof = m - n_params
    adj = 1.0 - (1.0 - r2) * (m - 1) / dof if (r2 is not None and dof > 0) else None
    sigma = math.sqrt(rss / dof) if dof > 0 else None
    se = None
    if cov_diag is not None and sigma is not None:
        se = sigma * np.sqrt(cov_diag)
    grid = np.linspace(data.a, data.b, N_GRID)
    return {
        "coefficients": beta.copy(),
        "degree": n_params - 1,
        "fitted": fitted,
        "residuals": resid,
        "rss": rss,
        "tss": tss,
        "r_squared": r2,
        "adj_r_squared": adj,
        "rmse": math.sqrt(rss / m),
        "sigma": sigma,
        "std_errors": se,
        "eval": {"x": grid, "y": _polyval(beta, grid)},
    }


def _fail(method: str, beta: Vector, message: str, trace: list[Step]) -> Result:
    return Result(method, beta, None, False, message, trace[-1].k, 0, trace=trace)


# --------------------------------------------------------------------------------------
# Ordinary least squares
# --------------------------------------------------------------------------------------


def _svd_lstsq(a: Vector, y: Vector) -> tuple[Vector, int, float, Vector | None]:
    """Minimum-‖γ‖ least squares min ‖Aγ - y‖₂ by the SVD (Golub & Van Loan, Thm. 5.5.1).

    γ = Σ_{σ_k > tol} v_k (u_kᵀy)/σ_k; a rank-deficient (or wide, m < p) matrix gives the
    minimum-norm solution, as ``numpy.linalg.lstsq``. Returns (γ, rank, κ₂,
    diag((AᵀA)⁻¹) or None). The callers pass the column-equilibrated design A = X/S (see
    :func:`_ols`): the rank threshold max(m, p)·ε·σ_max is relative to the largest column,
    so on the raw design it is not invariant under a rescaling of x.
    """
    p = a.shape[1]
    u, sv, vt = np.linalg.svd(a, full_matrices=False)  # sv: (min(m, p),)
    rank, cond = _rank_cond(sv, a.shape)
    coef_u = (u[:, :rank].T @ y) / sv[:rank]  # (r,)
    gamma = vt[:rank].T @ coef_u  # (p,)
    cov_diag = np.sum((vt.T / sv) ** 2, axis=1) if rank == p else None  # Σ_k V_jk²/σ_k²
    return gamma, rank, cond, cov_diag


#: The normal-equations solve is reported as not converged when the first-order forward
#: error bound κ₂(AᵀA)·ε of the Cholesky solution (Higham 2002, Thm. 10.4 with §20.4) is
#: at least this: fewer than 2 correct digits of the coefficients are then guaranteed.
NE_MAX_FORWARD_BOUND = 1e-2


def _not_unique_message(data: _Data, p: int, rank: int) -> str:
    return (
        f"rank-deficient design ({p} parameters, {data.m} points, "
        f"{np.unique(data.x).size} distinct x; numerical rank {rank} < {p}): the "
        "least-squares solution is not unique; returned the minimum-norm one (SVD)"
    )


def _ols(method: str, data: _Data, degree: int, solver: str, *, min_norm: bool = False) -> Result:
    """OLS for the polynomial of the given degree; one Step (k = 0).

    Every solver factors the column-equilibrated design A = X/S, S = diag(‖X_{:,j}‖₂)
    (van der Sluis 1969), and maps γ = Sβ back to β. Fewer points than parameters
    (m < p) is a rank-deficient design like any other. A rank-deficient design gives
    ``converged=False`` (the solution is not unique). With ``solver='svd'`` or
    ``min_norm`` it also returns the solution of minimum ‖Sβ‖₂ (SVD); otherwise NaN.
    """
    p = degree + 1
    x_mat = _vandermonde(data.x, degree)
    nan_beta = np.full(p, np.nan)
    info: dict[str, Any] = {"solver": solver}
    cov_diag: Vector | None = None
    converged = True
    # NOTE: equilibrate before every rank decision, the SVD included. The SVD of the raw
    # design is backward stable, but its rank rule σ_k > max(m, p)·ε·σ_max is not
    # invariant under x → c·x + d: for x = 1.7e9 + i (Unix seconds) the raw [1, x] has
    # σ_min/σ_max ≈ 1e-18 and was declared rank 1, which gave a flat line (slope 5e-9
    # instead of 0.475) with converged=True; the equilibrated design has κ₂ ≈ 6e8, rank 2.
    scale = _column_scales(x_mat)
    a = x_mat / scale  # unit-norm columns

    if solver == "svd":
        gamma, rank, cond, cov_gamma = _svd_lstsq(a, data.y)
        beta = gamma / scale
        info.update(rank=rank, cond=cond)
        if rank == p and cov_gamma is not None:  # cov_gamma is None only when rank < p
            cov_diag = cov_gamma / scale**2
            msg = "least-squares solution by SVD"
        else:
            converged = False
            msg = _not_unique_message(data, p, rank)
    else:
        sv = np.linalg.svd(a, compute_uv=False)
        rank, cond = _rank_cond(sv, a.shape)
        info.update(rank=rank, cond=cond)
        if solver == "normal_equations":
            info["cond_gram"] = cond * cond
        if rank < p and min_norm:
            # NOTE: minimum-norm in the equilibrated variables γ = Sβ (S = diag of the
            # column norms), not minimum ‖β‖₂: the raw monomial columns differ in norm by
            # up to ~|x|^d, so the unscaled SVD truncates rows it could resolve (e.g.
            # anscombe_1, d = 11: rank 9 of 11, misses the data by 0.86) while the
            # equilibrated design keeps rank m and interpolates.
            gamma, rank, cond, _ = _svd_lstsq(a, data.y)
            beta = gamma / scale
            info.update(solver="svd", rank=rank, cond=cond)
            converged = False
            msg = _not_unique_message(data, p, rank)
        elif rank < p:
            trace = [Step(0, nan_beta, None, info=info)]
            return _fail(
                method,
                nan_beta,
                f"rank-deficient design (rank {rank} < {p}): the least-squares solution "
                "is not unique (use solver='svd' for the minimum-norm one)",
                trace,
            )
        elif solver == "qr":
            gamma, r = _qr_lstsq(a, data.y)
            cov_diag = _inv_diag_from_triangle(r, upper=True) / scale**2
            msg = "least-squares solution by Householder QR"
            beta = gamma / scale
        else:
            # NOTE: deliberately the normal equations AᵀA γ = Aᵀy (Cholesky), to show
            # the squared condition number; QR is the default and the stable choice.
            gram = a.T @ a
            try:
                chol = np.linalg.cholesky(gram)
            except np.linalg.LinAlgError:
                trace = [Step(0, nan_beta, None, info=info)]
                return _fail(
                    method,
                    nan_beta,
                    f"Cholesky of AᵀA failed (κ(AᵀA) ≈ {cond * cond:.2g}): "
                    "the normal equations are numerically singular",
                    trace,
                )
            w = _forward_substitute(chol, a.T @ data.y)
            gamma = _back_substitute(chol.T, w)
            cov_diag = _inv_diag_from_triangle(chol, upper=False) / scale**2
            beta = gamma / scale
            cond_gram = cond * cond
            bound = cond_gram * _EPS
            lost = f"κ₂(AᵀA) ≈ {cond_gram:.3g}: about {math.log10(cond_gram):.1f} of 16 digits lost"
            if bound >= NE_MAX_FORWARD_BOUND:
                # A successful Cholesky only says AᵀA is numerically positive definite;
                # the forward error of γ can still be O(‖γ‖) (x = 1e8 + i: slope 0.419
                # instead of 0.475, κ₂(AᵀA) = 1.2e15).
                converged = False
                msg = (
                    f"normal equations solved, but {lost} (forward error bound "
                    f"κ₂(AᵀA)·ε = {bound:.2g} ≥ {NE_MAX_FORWARD_BOUND:g}): the coefficients "
                    "may have no correct digits; use solver='qr'"
                )
            else:
                msg = f"least-squares solution by the normal equations (Cholesky); {lost}"

    if not np.all(np.isfinite(beta)):
        trace = [Step(0, beta, None, info=info)]
        return _fail(method, beta, "non-finite coefficients", trace)
    stats = _goodness(data, beta, p, cov_diag if rank == p else None)
    stats.update(rank=info["rank"], cond=info["cond"], solver=info["solver"])
    trace = [Step(0, beta.copy(), stats["rss"], info=info)]
    return Result(method, beta, stats["rss"], converged, msg, 0, 0, trace=trace, extra=stats)


@register(
    id="linear_regression",
    family="regression",
    name="Linear regression (OLS)",
    params=(
        ParamSpec(
            "solver",
            "qr",
            kind="choice",
            choices=("qr", "normal_equations", "svd"),
            help="QR (stable), normal equations (squares κ), or SVD (rank-revealing).",
        ),
    ),
    needs=("data",),
    order="direct",
    summary="Fit the line y = β₀ + β₁x that minimizes the sum of squared vertical residuals.",
    references=(
        "Golub & Van Loan, Matrix Computations (4th ed.), §5.3 (Alg. 5.3.1 normal "
        "equations, Alg. 5.3.2 Householder LS) and §5.5 (SVD, rank deficiency)",
        "Higham, Accuracy and Stability of Numerical Algorithms (2nd ed.), Ch. 20",
        "Montgomery, Peck & Vining, Introduction to Linear Regression Analysis (5th ed.), "
        "Ch. 2–3 (standard errors, R², adjusted R²)",
    ),
)
@_quiet
def linear_regression(problem: Dataset | tuple[Any, Any], *, solver: str = "qr") -> Result:
    """Ordinary least squares for the line, min_β ‖y - Xβ‖₂², X = [1, x].

    Solvers (Golub & Van Loan, §5.3, §5.5): ``qr`` — Householder QR of the equilibrated design,
    β = R⁻¹Qᵀy by back substitution; ``normal_equations`` — Cholesky of AᵀA, forward and
    back substitution (κ(AᵀA) = κ(A)²: ~2·log10 κ(A) digits are lost instead of
    ~log10 κ(A) (Higham, §20.4)); ``svd`` — γ = Σ_{σ_k > tol} v_k (u_kᵀy)/σ_k on the
    equilibrated design, the minimum-‖Sβ‖ solution when the design is rank deficient
    (all x equal). All three factor A = X/S, S = diag(column norms), and the rank is
    decided on A, so a large x offset (x = 1e9 + i) does not change the rank.
    Standard errors: se_j = σ̂ sqrt([(XᵀX)⁻¹]_jj), σ̂² = RSS/(m - 2), computed from the
    triangular factor or the SVD (no inverse is formed).

    Stops after one Step. ``converged=False`` when the design is rank deficient (all x
    equal, or a single point: no unique solution; ``qr``/``normal_equations`` return NaN,
    ``svd`` returns the minimum-norm solution), when Cholesky fails, or when the
    normal-equations forward error bound κ₂(AᵀA)·ε ≥ ``NE_MAX_FORWARD_BOUND`` = 1e-2
    (the message gives the digits lost, log10 κ₂(AᵀA)). Extra keys: ``rank``, ``cond``,
    ``solver``.
    """
    if solver not in ("qr", "normal_equations", "svd"):
        raise ValueError(f"unknown solver {solver!r}")
    data = _resolve(problem, 1)
    return _ols("linear_regression", data, 1, solver)


@register(
    id="polynomial_regression",
    family="regression",
    name="Polynomial regression",
    params=(
        ParamSpec(
            "degree",
            2,
            kind="int",
            min=0,
            max=15,
            help="Polynomial degree d. With fewer than d + 1 distinct x the fit is not unique: "
            "the minimum-norm solution is returned with converged=False.",
        ),
    ),
    needs=("data",),
    order="direct",
    summary="Least-squares fit of a degree-d polynomial; higher d fits the noise.",
    references=(
        "Golub & Van Loan, Matrix Computations (4th ed.), Alg. 5.3.2 (Householder LS)",
        "van der Sluis (1969), Condition numbers and equilibration of matrices, "
        "Numer. Math. 14 (column scaling)",
    ),
)
@_quiet
def polynomial_regression(problem: Dataset | tuple[Any, Any], *, degree: int = 2) -> Result:
    """Least-squares polynomial fit, min_β Σ_i (y_i - Σ_j β_j x_i^j)².

    The Vandermonde matrix is ill-conditioned for high degree; its columns are scaled to
    unit 2-norm and the problem is solved by Householder QR (G&VL Alg. 5.3.2), and the
    reported ``cond`` shows how many digits are at risk (~log10 κ). Same statistics and
    one-Step trace as :func:`linear_regression` with ``solver='qr'``.

    Fewer than d + 1 distinct x values (in particular m < d + 1 points) make the design
    rank deficient: the least-squares solution is not unique. The method then returns the
    solution of minimum ‖Sβ‖₂, S = diag(column norms of X), by the SVD of the equilibrated
    design (G&VL Thm. 5.5.1; it interpolates the data when the x are distinct) with
    ``converged=False``, ``info.solver = "svd"`` and ``cond = ∞`` (a null space exists),
    so that every degree in the ParamSpec range gives a curve.
    Extra keys: ``rank``, ``cond``, ``solver``.
    """
    if degree < 0:
        raise ValueError("degree must be ≥ 0")
    data = _resolve(problem, 1)
    return _ols("polynomial_regression", data, int(degree), "qr", min_norm=True)


# --------------------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------------------


@register(
    id="ridge_regression",
    family="regression",
    name="Ridge regression",
    params=(
        ParamSpec(
            "lam",
            1.0,
            min=1e-8,
            max=1e4,
            log=True,
            help="Penalty λ on Σ_{j≥1} β_j² (λ = 0, plain OLS, is also accepted).",
        ),
        ParamSpec("degree", 1, kind="int", min=0, max=15, help="Polynomial degree d."),
    ),
    needs=("data",),
    order="direct",
    summary="Least squares plus a penalty λ‖β‖² that shrinks the slope coefficients toward 0.",
    references=(
        "Hastie, Tibshirani & Friedman, The Elements of Statistical Learning (2nd ed.), "
        "§3.4.1, eq. (3.41) (intercept not penalized)",
        "Golub & Van Loan, Matrix Computations (4th ed.), §6.1 (Tikhonov regularization "
        "as an augmented least-squares problem)",
    ),
)
@_quiet
def ridge_regression(
    problem: Dataset | tuple[Any, Any], *, lam: float = 1.0, degree: int = 1
) -> Result:
    """Ridge regression (Tikhonov regularization) of a degree-d polynomial.

    Minimizes Σ_i (y_i - β₀ - Σ_{j≥1} β_j x_i^j)² + λ Σ_{j≥1} β_j² (ESL eq. (3.41): the
    intercept is not penalized). Centering the features and y removes β₀
    (β₀ = ȳ - x̄ᵀβ); the slopes then solve the augmented least-squares problem
    min ‖[X_c; √λ I] β - [y_c; 0]‖₂ by Householder QR, which is the closed form
    (X_cᵀX_c + λI)⁻¹X_cᵀy_c without forming X_cᵀX_c (G&VL §6.1). The penalty is on the
    raw monomial coefficients (the features are not standardized), so λ has units.
    λ = 0 is OLS.

    # NOTE: the task names only λ; ``degree`` is added because on 1-D data ridge is
    # only interesting for polynomial features (degree 1 = the shrunken line).

    ``fun`` = RSS + λ‖β_{1:}‖². One Step. ``converged=False`` only for λ = 0 with a
    rank-deficient design. Extra keys: ``lambda``, ``cond`` (of the augmented matrix),
    ``rank``. ``std_errors`` is None (OLS formulas do not apply to a biased estimator).
    """
    if not (lam >= 0.0 and math.isfinite(lam)):
        raise ValueError("lam must be a finite number ≥ 0")
    if degree < 0:
        raise ValueError("degree must be ≥ 0")
    data = _resolve(problem, 1)
    d = int(degree)
    p = d + 1
    ybar = float(np.mean(data.y))
    info: dict[str, Any] = {"lambda": lam}
    if d == 0:
        beta = np.array([ybar])
        info.update(rank=1, cond=1.0)
    else:
        feats = _vandermonde(data.x, d)[:, 1:]  # (m, d): x, x², ..., x^d
        xbar = np.mean(feats, axis=0)  # (d,)
        xc = feats - xbar
        yc = data.y - ybar
        scale = _column_scales(xc)
        # Variables γ = s ⊙ β: ‖X_c β - y_c‖² + λ‖β‖² = ‖(X_c/s) γ - y_c‖² + λ‖γ/s‖².
        aug = np.vstack([xc / scale, np.diag(math.sqrt(lam) / scale)])  # (m + d, d)
        rhs = np.concatenate([yc, np.zeros(d)])
        sv = np.linalg.svd(aug, compute_uv=False)
        rank, cond = _rank_cond(sv, aug.shape)
        info.update(rank=rank, cond=cond)
        if rank < d:
            nan_beta = np.full(p, np.nan)
            return _fail(
                "ridge_regression",
                nan_beta,
                f"λ = 0 and the design is rank deficient (rank {rank} < {d}): no unique solution",
                [Step(0, nan_beta, None, info=info)],
            )
        gamma, _ = _qr_lstsq(aug, rhs)
        slopes = gamma / scale
        beta = np.concatenate([[ybar - float(np.dot(xbar, slopes))], slopes])
    if not np.all(np.isfinite(beta)):
        return _fail("ridge_regression", beta, "non-finite coefficients", [Step(0, beta, None)])
    stats = _goodness(data, beta, p, None)
    objective = stats["rss"] + lam * float(np.dot(beta[1:], beta[1:]))
    stats.update({"lambda": lam, "rank": info["rank"], "cond": info["cond"]})
    trace = [Step(0, beta.copy(), objective, info=info)]
    return Result(
        "ridge_regression",
        beta,
        objective,
        True,
        f"ridge solution for λ = {lam:g} by QR of the augmented system",
        0,
        0,
        trace=trace,
        extra=stats,
    )


# --------------------------------------------------------------------------------------
# Iteratively reweighted least squares (robust lines)
# --------------------------------------------------------------------------------------


def _wls(x_mat: Vector, y: Vector, w: Vector) -> Vector | None:
    """Weighted least squares min Σ w_i (y_i - x_iᵀβ)² by QR of diag(√w) X.

    Returns None when the weighted design is numerically rank deficient or its entries,
    the weights or the right-hand side are not finite (the SVD cannot be computed then).
    """
    sw = np.sqrt(w)
    a = x_mat * sw[:, None]
    rhs = y * sw
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(rhs))):
        return None
    scale = _column_scales(a)
    a = a / scale
    try:
        sv = np.linalg.svd(a, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    rank, _ = _rank_cond(sv, a.shape)
    if rank < x_mat.shape[1]:
        return None
    gamma, _ = _qr_lstsq(a, rhs)
    return gamma / scale


def _rms(r: Vector) -> float:
    """sqrt(mean(r²)) computed as ‖r‖∞·sqrt(mean((r/‖r‖∞)²)), so r² cannot overflow."""
    rmax = float(np.max(np.abs(r)))
    if rmax == 0.0 or not math.isfinite(rmax):
        return rmax
    return rmax * math.sqrt(float(np.mean((r / rmax) ** 2)))


def _huber_rho(u: Vector, delta: float) -> Vector:
    """Huber (1964): ρ(u) = u²/2 for |u| ≤ δ, δ|u| - δ²/2 otherwise."""
    au = np.abs(u)
    return np.where(au <= delta, 0.5 * u * u, delta * au - 0.5 * delta * delta)


def _huber_weights(u: Vector, delta: float) -> Vector:
    """w(u) = ψ(u)/u = min(1, δ/|u|) (w(0) = 1)."""
    au = np.abs(u)
    return np.where(au <= delta, 1.0, delta / np.maximum(au, delta))


def _lad_smoothed(r: Vector, eps: float) -> Vector:
    """ρ_ε(r) = |r| for |r| ≥ ε, r²/(2ε) + ε/2 otherwise (the function LAD-IRLS decreases)."""
    ar = np.abs(r)
    return np.where(ar >= eps, ar, r * r / (2.0 * eps) + 0.5 * eps)


def _change(beta_new: Vector, beta_old: Vector) -> float:
    return float(np.max(np.abs(beta_new - beta_old)))


@dataclass(frozen=True)
class _LineDesign:
    """The centered line design X_c = [1, x - c], c = x̄, of the IRLS fits.

    The weighted solves and the stopping test work in θ = [θ₀, θ₁], ŷ = θ₀ + θ₁(x - c);
    the reported line is β = [θ₀ - θ₁c, θ₁] (an exact reparametrization of the same model).
    # NOTE: the textbook IRLS solves with X = [1, x] and tests ‖Δβ‖. With a large x
    # offset that design has κ₂ ≈ |x̄|/std(x) even after column scaling (x = 1e8 + i:
    # 3.5e7), so each solve returned β₀ with an error ~ε·κ·|β₀| = 0.03 and β oscillated
    # at that level until max_iter, although the slope and the objective had converged.
    # Centring makes the unweighted design orthogonal (κ₂ of the scaled design is 1).
    # The test is on θ, not β: θ does not change when x is shifted, while
    # Δβ₀ = Δθ₀ - Δθ₁c grows with c, so a ‖Δβ‖ test asks for more iterations (66 at
    # offset 0, more than 100 at offset 1e4 for one slowly converging Huber fit).
    # x̄ is computed as c₀ + mean(x - c₀), c₀ the midrange, so that the sum cannot
    # overflow for |x| near the float limit (c₀ is used if it still does).
    """

    mat: Vector  # (m, 2): [1, x - c]
    center: float

    def beta(self, theta: Vector) -> Vector:
        return np.array([theta[0] - theta[1] * self.center, theta[1]])


def _line_design(data: _Data) -> _LineDesign:
    """The centered design for the robust line fits; all-equal x is invalid input."""
    if np.all(data.x == data.x[0]):
        raise ValueError("all x values are equal: the line is not identifiable")
    lo, hi = float(np.min(data.x)), float(np.max(data.x))
    mid = 0.5 * lo + 0.5 * hi  # cannot overflow; |x - mid| ≤ max|x|
    center = mid + float(np.mean(data.x - mid))
    if not math.isfinite(center):
        center = mid
    return _LineDesign(_vandermonde(data.x - center, 1), center)


def _ols_start(design: _LineDesign, y: Vector) -> tuple[Vector, Vector] | None:
    """The OLS line θ and its residuals (the IRLS start); None on a singular or non-finite fit."""
    theta = _wls(design.mat, y, np.ones(y.size))
    if theta is None or not np.all(np.isfinite(theta)):
        return None
    r = y - design.mat @ theta
    return (theta, r) if np.all(np.isfinite(r)) else None


def _start_failed(method: str, data: _Data, design: _LineDesign) -> Result:
    """converged=False when the least-squares start is singular or not finite."""
    theta = _wls(design.mat, data.y, np.ones(data.m))
    bad = np.full(2, np.nan) if theta is None else design.beta(theta)
    return _fail(
        method,
        bad,
        "the least-squares start is singular or not finite (overflow; e.g. the slope "
        "exceeds the floating-point range): IRLS cannot start",
        [Step(0, bad.copy(), None)],
    )


@dataclass(frozen=True)
class _IrlsRun:
    beta: Vector
    theta: Vector
    objective: float
    weights: Vector
    converged: bool
    message: str
    trace: list[Step]


def _lad_irls(
    design: _LineDesign, y: Vector, theta: Vector, *, eps: float, tol: float, max_iter: int
) -> _IrlsRun:
    """LAD-IRLS (Schlossmacher 1973) from the centered line ``theta``; see :func:`lad_regression`."""

    def state(t: Vector) -> tuple[float, float, Vector]:
        r = y - design.mat @ t
        return (
            float(np.sum(np.abs(r))),
            float(np.sum(_lad_smoothed(r, eps))),
            1.0 / np.maximum(np.abs(r), eps),
        )

    beta = design.beta(theta)
    obj, smooth, w = state(theta)
    trace = [Step(0, beta.copy(), obj, info={"weights": w, "smoothed_objective": smooth})]
    converged, msg = False, f"reached max_iter={max_iter}"
    for k in range(1, max_iter + 1):
        new = _wls(design.mat, y, w)
        if new is None:
            msg = "weighted design matrix is rank deficient or not finite"
            break
        change = _change(new, theta)
        theta = new
        beta = design.beta(theta)
        obj, smooth, w = state(theta)
        trace.append(
            Step(
                k,
                beta.copy(),
                obj,
                step_size=change,
                info={"weights": w, "smoothed_objective": smooth},
            )
        )
        if not (np.all(np.isfinite(beta)) and math.isfinite(obj)):
            msg = "non-finite coefficients or objective"
            break
        if change <= tol * (1.0 + float(np.max(np.abs(theta)))):
            converged, msg = True, f"‖Δθ‖∞ = {change:.3g} ≤ tol·(1 + ‖θ‖∞)"
            break
    return _IrlsRun(beta, theta, obj, w, converged, msg, trace)


#: The preliminary L1 fit of :func:`huber_regression` (its scale only): the smoothing
#: floor ε in units of the RMS least-squares residual, the step tolerance and the
#: iteration limit. A non-converged L1 iterate still gives a usable robust scale.
_PRELIM_LAD_EPS = 1e-6
_PRELIM_LAD_TOL = 1e-10
_PRELIM_LAD_MAX_ITER = 500


def _l1_residual_scale(r: Vector, n_params: int) -> float:
    """σ̂ = median of the m - p largest |r_i| / Φ⁻¹(3/4), r the residuals of an L1 fit.

    Maronna, Martin & Yohai (2006), Ch. 4: the normalized median of the non-null
    absolute residuals of a preliminary L1 fit (an L1 fit makes at least p residuals 0).
    # NOTE: IRLS leaves those residuals at ~ε instead of exactly 0, so the p smallest
    # |r_i| are dropped instead of the exact zeros.
    """
    ar = np.sort(np.abs(r))
    if ar.size > n_params:
        ar = ar[n_params:]
    return float(np.median(ar)) / MAD_TO_SIGMA


@register(
    id="huber_regression",
    family="regression",
    name="Huber regression (IRLS)",
    params=(
        ParamSpec(
            "delta",
            1.345,
            min=0.1,
            max=10.0,
            help="Threshold δ in units of σ̂ (1.345 gives 95% efficiency for Gaussian noise).",
        ),
        ParamSpec(
            "tol",
            1e-10,
            min=1e-15,
            max=1e-2,
            log=True,
            help="Stop when ‖Δθ‖∞ ≤ tol·(1 + ‖θ‖∞), θ = the line in the centered variable x − x̄.",
        ),
        ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="IRLS iteration limit."),
    ),
    needs=("data",),
    order="linear",
    summary="Quadratic loss for small residuals, linear for large ones; solved by reweighted least squares.",
    references=(
        "Huber (1964), Robust estimation of a location parameter, Ann. Math. Stat. 35",
        "Holland & Welsch (1977), Robust regression using iteratively reweighted "
        "least-squares, Commun. Stat. A6",
        "Maronna, Martin & Yohai, Robust Statistics (2006), Ch. 4 (regression M-estimates "
        "with a preliminary scale from an L1 fit; IRWLS and its monotone descent)",
    ),
)
@_quiet
def huber_regression(
    problem: Dataset | tuple[Any, Any],
    *,
    delta: float = 1.345,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Result:
    """Huber M-estimate of the line by iteratively reweighted least squares.

    Minimizes F(β) = Σ_i ρ_δ(r_i(β)/σ̂), r = y - β₀ - β₁x, ρ_δ from :func:`_huber_rho`,
    with the residual scale σ̂ fixed in advance (Maronna et al., Ch. 4): σ̂ is the
    normalized median of the non-null absolute residuals of a preliminary L1 (LAD) fit
    (:func:`_l1_residual_scale`; the L1 fit is LAD-IRLS from the OLS line with the floor
    ε = 1e-6·RMS(OLS residuals)). An L1 fit, unlike OLS, is not tilted by the outliers, so
    σ̂ estimates the noise of the clean points.
    IRLS (Holland & Welsch 1977): w_i = min(1, δσ̂/|r_i|), β_{k+1} = argmin Σ w_i r_i².
    Each step minimizes a quadratic majorizer of F, so F decreases monotonically to the
    minimum (Maronna et al., Ch. 4). Start: β₀ = OLS (Step k = 0; F is convex, so the
    start does not change the limit).

    # NOTE: when more than half the points lie on the L1 line to rounding level
    # (σ̂ ≤ 16·eps·‖y‖∞), σ̂ falls back to the RMS least-squares residual; if that is at
    # rounding level too, the OLS line fits every point and is returned (k = 0).

    The weighted solves use the centered design [1, x - x̄] (:class:`_LineDesign`), in the
    coefficients θ = [θ₀, θ₁] of ŷ = θ₀ + θ₁(x - x̄); β = [θ₀ - θ₁x̄, θ₁].
    Stopping test: ‖θ_k - θ_{k-1}‖∞ ≤ tol·(1 + ‖θ_k‖∞) → converged (invariant under a
    shift of x; see the NOTE in :class:`_LineDesign`).
    ``Step.step_size`` = ‖θ_k - θ_{k-1}‖∞. Max-iter, a singular weighted design, a non-finite start, scale,
    coefficient or objective → ``converged=False``. Raises ``ValueError`` when all x are
    equal. Extra keys: ``weights`` (final), ``scale``, ``delta``, ``objective``,
    ``x_center`` (x̄ of the centered variable).
    """
    if not delta > 0.0:
        raise ValueError("delta must be > 0")
    data = _resolve(problem, 2)
    design = _line_design(data)
    start = _ols_start(design, data.y)
    if start is None:
        return _start_failed("huber_regression", data, design)
    theta, r = start
    beta = design.beta(theta)
    # Residuals below ~16 ulps of the data are rounding noise, not a scale.
    noise = 16.0 * _EPS * max(float(np.max(np.abs(data.y))), float(np.finfo(np.float64).tiny))
    rms = _rms(r)
    if rms <= noise:
        stats = _goodness(data, beta, 2, None)
        stats.update(
            weights=np.ones(data.m), scale=0.0, delta=delta, objective=0.0, x_center=design.center
        )
        trace = [
            Step(
                0,
                beta.copy(),
                0.0,
                info={"weights": np.ones(data.m), "scale": 0.0, "delta": delta},
            )
        ]
        return Result(
            "huber_regression",
            beta,
            0.0,
            True,
            "the least-squares line fits every point exactly",
            0,
            0,
            trace=trace,
            extra=stats,
        )
    l1 = _lad_irls(
        design,
        data.y,
        theta,
        eps=_PRELIM_LAD_EPS * rms,
        tol=_PRELIM_LAD_TOL,
        max_iter=_PRELIM_LAD_MAX_ITER,
    )
    r_l1 = data.y - design.mat @ l1.theta
    scale = _l1_residual_scale(r_l1, 2) if np.all(np.isfinite(r_l1)) else math.nan
    if math.isfinite(scale) and scale <= noise:
        scale = rms
    if not (math.isfinite(scale) and math.isfinite(rms)):
        return _fail(
            "huber_regression",
            beta,
            f"the residual scale σ̂ = {scale:.3g} is not finite (overflow)",
            [Step(0, beta.copy(), None)],
        )

    def state(t: Vector) -> tuple[float, Vector]:
        u = (data.y - design.mat @ t) / scale
        return float(np.sum(_huber_rho(u, delta))), _huber_weights(u, delta)

    obj, w = state(theta)
    if not math.isfinite(obj):
        return _fail(
            "huber_regression",
            beta,
            "non-finite objective at the least-squares start (overflow)",
            [Step(0, beta.copy(), None, info={"weights": w, "scale": scale, "delta": delta})],
        )
    trace = [Step(0, beta.copy(), obj, info={"weights": w, "scale": scale, "delta": delta})]
    converged, msg = False, f"reached max_iter={max_iter}"
    for k in range(1, max_iter + 1):
        new = _wls(design.mat, data.y, w)
        if new is None:
            msg = "weighted design matrix is rank deficient or not finite"
            break
        change = _change(new, theta)
        theta = new
        beta = design.beta(theta)
        obj, w = state(theta)
        trace.append(
            Step(
                k,
                beta.copy(),
                obj,
                step_size=change,
                info={"weights": w, "scale": scale, "delta": delta},
            )
        )
        if not (np.all(np.isfinite(beta)) and math.isfinite(obj)):
            msg = "non-finite coefficients or objective"
            break
        if change <= tol * (1.0 + float(np.max(np.abs(theta)))):
            converged, msg = True, f"‖Δθ‖∞ = {change:.3g} ≤ tol·(1 + ‖θ‖∞)"
            break
    stats = _goodness(data, beta, 2, None)
    stats.update(weights=w, scale=scale, delta=delta, objective=obj, x_center=design.center)
    return Result(
        "huber_regression", beta, obj, converged, msg, trace[-1].k, 0, trace=trace, extra=stats
    )


@register(
    id="lad_regression",
    family="regression",
    name="Least absolute deviations (IRLS)",
    params=(
        ParamSpec(
            "eps",
            1e-6,
            min=1e-12,
            max=1e-1,
            log=True,
            help="Weight floor: w_i = 1/max(|r_i|, eps) (in units of y).",
        ),
        ParamSpec(
            "tol",
            1e-10,
            min=1e-15,
            max=1e-2,
            log=True,
            help="Stop when ‖Δθ‖∞ ≤ tol·(1 + ‖θ‖∞), θ = the line in the centered variable x − x̄.",
        ),
        ParamSpec("max_iter", 500, kind="int", min=1, max=10_000, help="IRLS iteration limit."),
    ),
    needs=("data",),
    order="linear",
    summary="Minimize the sum of absolute residuals by repeatedly solving weighted least squares.",
    references=(
        "Schlossmacher (1973), An iterative technique for absolute deviations curve "
        "fitting, JASA 68",
        "Björck, Numerical Methods for Least Squares Problems (1996), Ch. 4 (IRLS for ℓ_p)",
    ),
)
@_quiet
def lad_regression(
    problem: Dataset | tuple[Any, Any],
    *,
    eps: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> Result:
    """Least-absolute-deviations (L1) line by IRLS (Schlossmacher 1973).

    Minimizes Σ_i |y_i - β₀ - β₁x_i|. IRLS: w_i = 1/max(|r_i|, ε), β_{k+1} = argmin Σ w_i r_i².
    With the floor ε, each step minimizes a quadratic majorizer of the smoothed objective
    S_ε(β) = Σ ρ_ε(r_i) (:func:`_lad_smoothed`, a Huber function with threshold ε), so S_ε
    decreases monotonically; since |r| ≤ ρ_ε(r) ≤ |r| + ε/2, the limit is within m·ε/2 of
    the LAD optimum in objective. LAD solutions need not be unique. Start: OLS (k = 0).

    The weighted solves use the centered design [1, x - x̄] (:class:`_LineDesign`), in the
    coefficients θ = [θ₀, θ₁] of ŷ = θ₀ + θ₁(x - x̄); β = [θ₀ - θ₁x̄, θ₁].
    Stopping test: ‖θ_k - θ_{k-1}‖∞ ≤ tol·(1 + ‖θ_k‖∞) → converged (invariant under a
    shift of x). ``Step.step_size`` = ‖θ_k - θ_{k-1}‖∞. ``Step.fun`` =
    Σ|r_i| (the true L1 objective); ``info.smoothed_objective`` = S_ε. Max-iter, a singular
    weighted design, a non-finite start, coefficient or objective → ``converged=False``.
    Raises ``ValueError`` when all x are equal. Extra keys: ``weights`` (final), ``eps``,
    ``objective``, ``x_center`` (x̄ of the centered variable).
    """
    if not eps > 0.0:
        raise ValueError("eps must be > 0")
    data = _resolve(problem, 2)
    design = _line_design(data)
    start = _ols_start(design, data.y)
    if start is None:
        return _start_failed("lad_regression", data, design)
    run = _lad_irls(design, data.y, start[0], eps=eps, tol=tol, max_iter=max_iter)
    stats = _goodness(data, run.beta, 2, None)
    stats.update(weights=run.weights, eps=eps, objective=run.objective, x_center=design.center)
    return Result(
        "lad_regression",
        run.beta,
        run.objective,
        run.converged,
        run.message,
        run.trace[-1].k,
        0,
        trace=run.trace,
        extra=stats,
    )


# --------------------------------------------------------------------------------------
# Theil–Sen
# --------------------------------------------------------------------------------------


@register(
    id="theil_sen",
    family="regression",
    name="Theil–Sen estimator",
    params=(),
    needs=("data",),
    order="direct, O(m² log m)",
    summary="The slope is the median of the slopes through all pairs of points.",
    references=(
        "Theil (1950), A rank-invariant method of linear and polynomial regression "
        "analysis, Indag. Math. 12",
        "Sen (1968), Estimates of the regression coefficient based on Kendall's tau, JASA 63",
    ),
)
@_quiet
def theil_sen(problem: Dataset | tuple[Any, Any]) -> Result:
    """Theil–Sen line: β₁ = median{(y_j - y_i)/(x_j - x_i) : i < j, x_i ≠ x_j} (exact).

    Pairs with equal x are skipped (Sen 1968). The intercept is β₀ = median_i(y_i - β₁x_i)
    (the "joint" intercept of ``scipy.stats.theilslopes(method='joint')``); the median of
    an even count is the mean of the two middle values. Breakdown point ≈ 29%.
    One Step. ``fun`` = RSS (Theil–Sen minimizes no objective). Raises ``ValueError``
    when all x are equal. A non-finite slope or intercept (a pairwise slope or y_i - β₁x_i
    overflows the floating-point range) → ``converged=False``.
    Extra keys: ``n_pairs``, ``slopes`` (sorted).
    """
    data = _resolve(problem, 2)
    i_idx, j_idx = np.triu_indices(data.m, k=1)
    dx = data.x[j_idx] - data.x[i_idx]
    keep = dx != 0.0
    if not np.any(keep):
        raise ValueError("all x values are equal: no slope is defined")
    slopes = np.sort((data.y[j_idx] - data.y[i_idx])[keep] / dx[keep])
    slope = float(np.median(slopes))
    intercept = float(np.median(data.y - slope * data.x))
    beta = np.array([intercept, slope])
    if not np.all(np.isfinite(beta)):
        return _fail(
            "theil_sen",
            beta,
            f"non-finite line (slope {slope:.3g}, intercept {intercept:.3g}): the pairwise "
            "slopes overflow the floating-point range",
            [Step(0, beta.copy(), None, info={"slopes": slopes})],
        )
    stats = _goodness(data, beta, 2, None)
    stats.update(n_pairs=int(slopes.size), slopes=slopes)
    trace = [Step(0, beta.copy(), stats["rss"], info={"slopes": slopes})]
    return Result(
        "theil_sen",
        beta,
        stats["rss"],
        True,
        f"median of {slopes.size} pairwise slopes",
        0,
        0,
        trace=trace,
        extra=stats,
    )


# --------------------------------------------------------------------------------------
# Chebyshev (minimax) line
# --------------------------------------------------------------------------------------


def _levelled_line(x: Vector, y: Vector, ref: list[int]) -> Vector | None:
    """Solve β₀ + β₁x_{r_j} + (-1)^j h = y_{r_j}, j = 0, 1, 2 → [β₀, β₁, h]."""
    a = np.array([[1.0, x[r], (-1.0) ** j] for j, r in enumerate(ref)])
    try:
        sol = np.linalg.solve(a, y[ref])
    except np.linalg.LinAlgError:
        return None
    return sol if np.all(np.isfinite(sol)) else None


def _exchange(ref: list[int], xs: Vector, k: int, sign_k: float, h: float) -> list[int]:
    """Single-point exchange (Stiefel 1959) keeping the residual signs alternating.

    The residual at reference point j is (-1)^j h, so its sign is σ_j = (-1)^j sgn(h)
    (sgn 0 := 1). A new point k replaces the neighbor with the same sign, or shifts the
    reference when it lies outside it with the opposite sign.
    """
    sg = 1.0 if h >= 0.0 else -1.0
    sigma = [sg, -sg, sg]
    p0, p1, p2 = ref
    xk = xs[k]
    if xk < xs[p0]:
        return [k, p1, p2] if sign_k == sigma[0] else [k, p0, p1]
    if xk < xs[p1]:
        return [k, p1, p2] if sign_k == sigma[0] else [p0, k, p2]
    if xk < xs[p2]:
        return [p0, k, p2] if sign_k == sigma[1] else [p0, p1, k]
    return [p0, p1, k] if sign_k == sigma[2] else [p1, p2, k]


#: Rounding level of the minimax stopping test, in units of eps·S + η (see the NOTE in
#: :func:`chebyshev_minimax_line`).
_MINIMAX_ROUNDING_EPS = 8.0


@register(
    id="chebyshev_minimax_line",
    family="regression",
    name="Minimax (Chebyshev) line",
    params=(
        ParamSpec(
            "tol",
            1e-12,
            min=1e-15,
            max=1e-3,
            log=True,
            help="Stop when max|r| − |h| ≤ tol·max(|h|, ‖y‖∞) + the rounding level of r.",
        ),
        ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Exchange limit."),
    ),
    needs=("data",),
    order="finite (exchange)",
    summary="The line that minimizes the largest vertical error; found by exchanging 3 reference points.",
    references=(
        "Stiefel (1959), Über diskrete und lineare Tschebyscheff-Approximationen, "
        "Numer. Math. 1 (exchange algorithm)",
        "Cheney, Introduction to Approximation Theory (1966), Ch. 2 (alternation theorem, "
        "de la Vallée Poussin ascent)",
    ),
)
@_quiet
def chebyshev_minimax_line(
    problem: Dataset | tuple[Any, Any], *, tol: float = 1e-12, max_iter: int = 100
) -> Result:
    """Discrete minimax line min_β max_i |y_i - β₀ - β₁x_i| by Stiefel's exchange algorithm.

    By the alternation theorem the optimum equioscillates on 3 points x_{r0} < x_{r1} < x_{r2}.
    Each iteration solves β₀ + β₁x_{r_j} + (-1)^j h = y_{r_j} (the levelled reference
    line), finds k = argmax |r_k|, stops if |r_k| ≤ |h| (+ tolerance), and otherwise
    exchanges k into the reference (:func:`_exchange`). |h| increases strictly at each
    exchange (de la Vallée Poussin), so the method terminates. Distinct x are required
    (Haar condition); data are processed in x order but indices refer to the input order.
    Start: the reference {first, middle, last} in x order (Step k = 0).

    Stopping test: max_i |r_i| - |h| ≤ tol·max(|h|, ‖y‖∞) + 8·(eps·S + η) → converged,
    with S = max_i (|y_i| + |β₀| + |β₁x_i| + |h|) and η = 2⁻¹⁰⁷⁴ (the underflow term, for
    subnormal data such as y = [0, 0, 0, 0, 5e-324]). Since |h| ≤ (optimal max error) ≤ max|r|
    (de la Vallée Poussin), the returned line is optimal to within that bound.
    # NOTE: the term 8·(eps·S + η) is the rounding level of the computed gap and is not in
    # the textbook test: r_i = y_i - (β₀ + β₁x_i) carries an error up to γ₃·S (Higham
    # 2002, §3.1), and the 3×3 solve adds its backward error (Higham Thm. 9.4); the
    # largest final gap measured over 4000 random lines with x offsets up to 1e12 was
    # 0.79·eps·S. Without it, data with a large x offset (x = 1e6 + i, where β₀ and β₁x
    # cancel) can never meet tol·max(|h|, ‖y‖∞) and stopped with "no ascent".
    ``Step.fun`` = max_i |r_i|. Max-iter, a singular or overflowing reference system, or no
    ascent → ``converged=False``; the last Step then has ``entering = None``.
    Extra keys: ``reference``, ``level``, ``max_deviation``.
    """
    data = _resolve(problem, 3)
    order = np.argsort(data.x, kind="stable")
    xs, ys = data.x[order], data.y[order]
    if np.any(np.diff(xs) == 0.0):
        raise ValueError("the minimax line needs distinct x values (Haar condition)")
    m = data.m
    x_mat = _vandermonde(xs, 1)
    yscale = float(np.max(np.abs(ys)))
    ref = [0, (m - 1) // 2, m - 1]
    trace: list[Step] = []
    beta = np.full(2, np.nan)
    h, dev, last_ref = 0.0, math.inf, list(ref)
    converged, msg = False, f"reached max_iter={max_iter}"
    for k in range(max_iter + 1):
        sol = _levelled_line(xs, ys, ref)
        if sol is None:
            msg = "singular or non-finite (overflow) reference system"
            break
        h_new = float(sol[2])
        if k > 0 and abs(h_new) <= abs(h):
            # Theory: |h| increases strictly at every exchange; equality means rounding.
            msg = "no ascent of the levelled error |h| (rounding): stopped"
            break
        beta, h, last_ref = sol[:2].copy(), h_new, list(ref)
        r = ys - x_mat @ beta
        kmax = int(np.argmax(np.abs(r)))
        dev = float(abs(r[kmax]))
        fitted_scale = np.abs(ys) + abs(float(beta[0])) + np.abs(beta[1] * xs) + abs(h)
        rounding = _MINIMAX_ROUNDING_EPS * (_EPS * float(np.max(fitted_scale)) + _ETA)
        done = dev - abs(h) <= tol * max(abs(h), yscale) + rounding
        stop = done or k == max_iter
        info = {
            "reference": [int(order[i]) for i in ref],
            "level": h,
            "entering": None if stop else int(order[kmax]),
            "max_deviation": dev,
        }
        trace.append(Step(k, beta.copy(), dev, info=info))
        if done:
            converged, msg = True, f"max|r| = {dev:.6g} equals the levelled error |h|"
            break
        if stop:
            break
        ref = _exchange(ref, xs, kmax, 1.0 if r[kmax] > 0.0 else -1.0, h)
    if not trace:
        nan_beta = np.full(2, np.nan)
        return _fail("chebyshev_minimax_line", nan_beta, msg, [Step(0, nan_beta, None, info={})])
    if not converged and trace[-1].info["entering"] is not None:
        # Stopped on no ascent or a singular system: the exchange announced in the last
        # Step was rejected, so nothing enters.
        last = trace[-1]
        trace[-1] = replace(last, info={**last.info, "entering": None})
    stats = _goodness(data, beta, 2, None)
    stats.update(reference=[int(order[i]) for i in last_ref], level=h, max_deviation=dev)
    return Result(
        "chebyshev_minimax_line",
        beta,
        dev,
        converged,
        msg,
        trace[-1].k,
        0,
        trace=trace,
        extra=stats,
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("linear_regression", "anscombe_1", {"solver": "qr"}),
    ("linear_regression", "noisy_linear", {"solver": "normal_equations"}),
    ("polynomial_regression", "noisy_quadratic", {"degree": 2}),
    ("ridge_regression", "noisy_quadratic", {"lam": 10.0, "degree": 6}),
    ("huber_regression", "outliers_linear", {}),
    ("lad_regression", "outliers_linear", {}),
    ("theil_sen", "outliers_linear", {}),
    ("chebyshev_minimax_line", "noisy_linear", {}),
]
