"""Iterative solvers for a square linear system A x = b.

Stationary methods (Jacobi, Gauss–Seidel, SOR) iterate a fixed map x ← G x + c from the
splitting A = D + L + U (diagonal, strictly lower, strictly upper part). They converge for every
start iff the spectral radius ρ(G) < 1 (Burden & Faires, §7.3). Krylov and descent methods
(steepest descent, CG, preconditioned CG, GMRES) pick x_k from x₀ + span{r₀, A r₀, …}.

Residual and stopping test (every method): r = b − A x and

    converged  ⇔  ‖r_k‖₂ ≤ tol · d   and   u·(‖A‖_F‖x_k‖₂ + ‖b‖₂) ≤ tol · d,   d = ‖b‖₂.

# NOTE: when b = 0 the denominator is d = ‖r₀‖₂ (the test ‖r_k‖₂ ≤ tol·‖r₀‖₂; Barrett et al.,
# Templates for the Solution of Linear Systems (1994), §4.2). The absolute test ‖r_k‖₂ ≤ tol is
# not scale-invariant: for A = 1e-12·I, b = 0 it accepted x₀ = (1, 1) at k = 0, although the
# solution is 0. The normwise backward error ‖r‖/(‖A‖_F‖x‖) is scale-invariant but does not go to
# 0 as x_k → 0 for a nonsingular A (it stays ≥ σ_min/‖A‖_F), so it would reject a convergent
# iteration. ‖r_k‖ ≤ tol·‖r₀‖ scales with A, x₀ and accepts x_k → 0 (or x_k → a null vector of a
# singular A). If also r₀ = fl(A x₀) = 0, then d = ‖A‖_F‖x₀‖₂ (x₀ solves A x = 0 to working
# precision) or d = 1 when x₀ = 0 or A = 0 (x₀ is exact), and the method stops at k = 0.

# NOTE: the second condition is not in the textbooks. u·(‖A‖_F‖x‖₂ + ‖b‖₂), u = ε/2, is the
# rounding level of a computed residual fl(b − A x): the normwise backward error
# ‖r‖/(‖A‖‖x‖ + ‖b‖) of a computed x cannot be resolved below about u (Rigal & Gaches 1967;
# Higham 2002, Thm 7.1 and §3.5). When the rounding level exceeds tol·‖b‖, a small computed residual
# is rounding noise and certifies nothing: on a singular or inconsistent system, GMRES can reach
# ‖x‖ ≈ 1e15 with fl(b − A x) = 0 while min_x ‖b − A x‖₂ > 0. The method then stops with
# converged=False and says so.

``tol`` must be ≥ 1e-15 (a ``ValueError`` otherwise): a relative residual below about 9u cannot be
resolved, and with tol = 0 the recurrences of CG and steepest descent run on until rᵀr and pᵀAp
underflow and report a false breakdown. ``max_iter`` (and GMRES ``restart``) must be an integer
≥ 1, and ‖A‖_F, ‖b‖₂ must not overflow (finite data with a norm above ≈ 1.8e308 must be rescaled
by a power of 2); a ``ValueError`` is raised otherwise. When r₀ = b − A x₀ overflows (a huge x₀),
steepest descent, CG, PCG and GMRES stop at k = 0 with converged=False.

The trace has one Step for k = 0 (x₀) and one per iteration; ``Step.fun`` is ‖r_k‖₂ and
``Step.step_size`` is ‖x_k − x_{k−1}‖₂. Methods never raise on divergence or breakdown: they stop
with ``converged=False`` and say why (ρ(G) ≥ 1, a zero diagonal entry, pᵀAp ≤ 0, a non-symmetric
matrix for CG, a non-finite iterate, max_iter).

CG, steepest descent and GMRES update the residual by a recurrence (``Step.fun`` is the recurrence
or Givens value); on exit they recompute the true residual ‖b − A x‖₂ once, ``Result.fun`` is that
true residual, and ``converged=True`` requires that it also passes the test.

Counts: ``n_fev``/``n_gev`` are 0 (there is no objective function). ``extra["n_matvec"]`` counts
the products A·v exactly for steepest descent, CG, PCG and GMRES (the cost unit of a Krylov
method); a stationary sweep costs the same as one product, and its count is ``n_iter`` sweeps
plus ``n_iter + 1`` residual products.

Accepted problems: a :class:`~numopt.core.types.LinearSystem` or a pair ``(A, b)``. ``x0``
defaults to the zero vector.

Info keys:
    residual: [n] r_k = b − A x_k (GMRES: r₀ − A V_j y_j from the Arnoldi relation
        A V_j = V_j H_j + w e_jᵀ, that is V_j(βe₁ − H_j y_j) − y_j w; no product with A).
    residual_norm: float, ‖r_k‖₂ (GMRES: |g_{j+1}| from the Givens rotations).
    relative_residual: float, ‖r_k‖₂ / d with d = ‖b‖₂ (‖r₀‖₂ when b = 0), the quantity in the
        stopping test.
    spectral_radius: float, ρ(G) of the iteration matrix (k = 0 only; Jacobi, Gauss–Seidel, SOR).
    sweep: [[n] × (n + 1)] the points x_{k−1}, then x after each component update in order
        i = 0 … n−1 (Gauss–Seidel and SOR, k ≥ 1): the coordinate-wise staircase in 2-D.
    direction: [n] the search direction from x_k, along which the next step moves
        (steepest descent: r_k; CG: p_k; PCG: p_k).
    alpha: float, the step α_{k−1} that produced x_k = x_{k−1} + α_{k−1} p_{k−1} (``None`` at k = 0).
    beta: float, β_k in p_k = r_k + β_k p_{k−1} (CG) or p_k = z_k + β_k p_{k−1} (PCG)
        (``None`` at k = 0).
    phi: float, φ(x_k) = ½x_kᵀA x_k − bᵀx_k, evaluated as −½x_kᵀ(b + r_k) (no product with A);
        steepest descent, CG and PCG minimize φ, whose minimizer is the solution.
    condition_number: float | None, κ₂(A) = λ_max/λ_min when A is SPD, else ``None`` (k = 0 only;
        steepest descent, CG, PCG).
    preconditioned_condition_number: float | None, κ₂(D^{−1/2} A D^{−1/2}) (PCG, k = 0 only).
    rate_bound: float | None, the contraction factor of the textbook A-norm error bound per
        iteration: (κ − 1)/(κ + 1) for steepest descent, (√κ − 1)/(√κ + 1) for CG and PCG (with the
        preconditioned κ) (k = 0 only).
    preconditioned_residual: [n] z_k = M⁻¹ r_k with M = diag(A) (PCG).
    cycle: int, the GMRES restart cycle (0-based).
    krylov_dim: int, j, the dimension of the Krylov space of the current cycle (GMRES).
    hessenberg: [[j] × (j + 1)] H̄_j, the (j+1) × j Arnoldi Hessenberg matrix (GMRES, before the
        rotations).
    givens: [c, s] the newest Givens rotation (GMRES).
    basis_vector: [n] | None, v_{j+1}, the newest Arnoldi vector (GMRES; ``None`` on breakdown
        and at j = n, where ℝⁿ holds no further basis vector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.registry import ParamSpec, register
from ..core.types import Result, Step, Vector, as_vector
from .direct import (
    EPS,
    SystemLike,
    back_substitution,
    is_symmetric,
    norm2,
    pivot_tolerance,
    quiet,
    resolve_system,
)

#: Smallest admissible tol (also the ParamSpec minimum); see the module docstring.
TOL_MIN = 1e-15
#: Unit roundoff u = ε/2 = 2⁻⁵³.
UNIT_ROUNDOFF = EPS / 2.0

TOL = ParamSpec(
    "tol",
    1e-10,
    min=TOL_MIN,
    max=1e-1,
    log=True,
    help="Stop when ‖b − Ax‖₂ ≤ tol·‖b‖₂ (tol·‖r₀‖₂ when b = 0).",
)
MAX_ITER = ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Iteration limit.")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _start(x0: Any, n: int) -> Vector:
    if x0 is None:
        return np.zeros(n)
    x = as_vector(x0)
    if x.size != n:
        raise ValueError(f"x0 has {x.size} entries, expected {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must be finite")
    return x


def _check_count(name: str, value: Any) -> int:
    """Return ``value`` as an int; raise ValueError unless it is an integer ≥ 1 (ParamSpec min)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer ≥ 1, got {value!r}")
    return int(value)


def _check_norms(A: np.ndarray, b: Vector) -> float:
    """Return ‖A‖_F; raise ValueError when ‖A‖_F or ‖b‖₂ overflows although A and b are finite.

    # NOTE: the stopping test divides by ‖b‖₂ and the rounding level uses ‖A‖_F. For finite data
    # with a norm above the largest double (e.g. 1e308·I₄, b = 1e308·𝟙) both are ∞, ∞/∞ = NaN, and
    # every method would stop at k = 0 with "‖r‖/‖b‖ = nan". Scaling A and b by a power of 2 is
    # exact and leaves x unchanged, so the input is rejected with that advice (the direct methods
    # solve such a system).
    """
    a_norm = norm2(A)
    for name, value in (("‖A‖_F", a_norm), ("‖b‖₂", norm2(b))):
        if not np.isfinite(value):
            raise ValueError(
                f"{name} exceeds the float64 range (≈ 1.8e308): rescale A and b by a power of 2 "
                "(this is exact and does not change x)"
            )
    return a_norm


def _pow2_scale(v: Vector) -> float:
    """A power of 2 within a factor 2 of max|v_i| (1.0 for v = 0): division by it is exact."""
    s = float(np.max(np.abs(v), initial=0.0))
    return float(np.ldexp(1.0, int(np.frexp(s)[1]))) if s > 0.0 else 1.0


def _scaled_dot(u: Vector, v: Vector, s: float) -> float:
    """uᵀv / s² computed as (u/s)ᵀ(v/s): no overflow of uᵀv when u and v are large."""
    return float((u / s) @ (v / s))


def _check_tol(tol: float) -> None:
    """Reject tol < 1e-15 (and NaN): see the module docstring."""
    if not tol >= TOL_MIN:
        raise ValueError(
            f"tol must be ≥ {TOL_MIN:g} (a relative residual below ≈ 9u is rounding noise), "
            f"got {tol}"
        )


def _rounding_level(a_norm: float, x: Vector, b: Vector) -> float:
    """u·(‖A‖_F‖x‖₂ + ‖b‖₂): the size of the rounding error in a computed residual b − A x."""
    return UNIT_ROUNDOFF * (a_norm * norm2(x) + norm2(b))


@dataclass(frozen=True)
class _Scale:
    """The denominator d of the stopping test ‖r_k‖₂ ≤ tol·d and its printed name."""

    value: float
    name: str

    def ratio(self, rnorm: float) -> str:
        return f"‖r‖/{self.name} = {rnorm / self.value:.3g}"


def _scale(A: np.ndarray, b: Vector, x0: Vector, r0: Vector) -> _Scale:
    """d = ‖b‖₂; for b = 0: ‖r₀‖₂, else ‖A‖_F‖x₀‖₂, else 1 (see the module docstring).

    Each fallback is used only when the previous value is 0 or not finite (an overflowed r₀ for a
    huge x₀), so that d is always finite and positive.
    """
    candidates = (
        (norm2(b), "‖b‖"),
        (norm2(r0), "‖r₀‖"),
        (norm2(A) * norm2(x0), "(‖A‖_F‖x₀‖)"),
    )
    for value, name in candidates:
        if value > 0.0 and np.isfinite(value):
            return _Scale(value, name)
    return _Scale(1.0, "1")


def _certify(
    rnorm: float, x: Vector, a_norm: float, b: Vector, tol: float, scale: _Scale, message: str
) -> tuple[bool, str]:
    """For a residual that passed ‖r‖₂ ≤ tol·d: (converged, message).

    converged additionally requires u·(‖A‖_F‖x‖₂ + ‖b‖₂) ≤ tol·d (module docstring); else the
    message says that the residual is below the rounding level and cannot certify x.
    """
    level = _rounding_level(a_norm, x, b)
    if level <= tol * scale.value:
        return True, message
    return False, (
        f"{scale.ratio(rnorm)} ≤ tol, but the rounding level of the computed residual, "
        f"u(‖A‖_F‖x‖ + ‖b‖)/{scale.name} = {level / scale.value:.3g}, exceeds tol "
        f"(‖x‖ = {norm2(x):.3g}): the residual cannot certify x (A is singular, or too "
        "ill-conditioned for this tol)"
    )


def _residual_info(r: Vector, rnorm: float, scale: _Scale) -> dict[str, Any]:
    return {"residual": r.copy(), "residual_norm": rnorm, "relative_residual": rnorm / scale.value}


def _spectral_radius(G: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def _sor_matrix(A: np.ndarray, omega: float) -> np.ndarray:
    """G_ω = (D + ωL)⁻¹((1 − ω)D − ωU), formed by a triangular solve (never an inverse)."""
    D = np.diag(np.diag(A))
    Lo = np.tril(A, -1)
    Up = np.triu(A, 1)
    return np.linalg.solve(D + omega * Lo, (1.0 - omega) * D - omega * Up)


def _jacobi_matrix(A: np.ndarray) -> np.ndarray:
    """G_J = −D⁻¹(L + U) (a row scaling by 1/a_ii)."""
    d = np.diag(A)
    return -(A - np.diag(d)) / d[:, None]


def _is_tridiagonal(A: np.ndarray) -> bool:
    i, j = np.indices(A.shape)
    return bool(np.all(A[np.abs(i - j) > 1] == 0.0))


def _zero_diagonal(
    method: str, A: np.ndarray, x: Vector, r: Vector, scale: _Scale
) -> Result | None:
    d = np.diag(A)
    if np.all(d != 0.0):
        return None
    i = int(np.flatnonzero(d == 0.0)[0])
    rnorm = norm2(r)
    trace = [Step(0, x, rnorm, info=_residual_info(r, rnorm, scale))]
    return Result(
        method,
        x,
        rnorm,
        False,
        f"a_{i}{i} = 0: the iteration divides by the diagonal; reorder the equations",
        0,
        trace=trace,
    )


def _divergence_note(rho: float) -> str:
    return f" (spectral radius ρ(G) = {rho:.4g} ≥ 1: the iteration diverges)" if rho >= 1.0 else ""


# --------------------------------------------------------------------------------------
# Stationary methods: Jacobi, Gauss–Seidel, SOR
# --------------------------------------------------------------------------------------


@quiet
def _stationary(
    method: str,
    problem: SystemLike,
    x0: Any,
    tol: float,
    max_iter: int,
    omega: float | None,
) -> Result:
    """Shared driver. ``omega=None`` is Jacobi; otherwise SOR with that ω (ω = 1: Gauss–Seidel)."""
    _check_tol(tol)
    max_iter = _check_count("max_iter", max_iter)
    A, b = resolve_system(problem)
    a_norm = _check_norms(A, b)
    n = b.size
    x = _start(x0, n)
    r = b - A @ x
    scale = _scale(A, b, x, r)
    failed = _zero_diagonal(method, A, x, r, scale)
    if failed is not None:
        return failed

    d = np.diag(A).copy()
    G = _jacobi_matrix(A) if omega is None else _sor_matrix(A, omega)
    rho = _spectral_radius(G)
    extra: dict[str, Any] = {"spectral_radius": rho}
    if method == "sor" and omega is not None:
        extra.update(_optimal_omega(A, omega))

    rnorm = norm2(r)
    info0 = {**_residual_info(r, rnorm, scale), "spectral_radius": rho}
    trace = [Step(0, x.copy(), rnorm, info=info0)]
    if rnorm <= tol * scale.value:
        ok, msg = _certify(rnorm, x, a_norm, b, tol, scale, "x0 already satisfies the tolerance")
        return Result(method, x, rnorm, ok, msg, 0, trace=trace, extra=extra)

    off = A - np.diag(d)  # L + U
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        info: dict[str, Any] = {}
        if omega is None:
            # Jacobi (B&F Alg. 7.1): x_i ← (b_i − Σ_{j≠i} a_ij x_j^old) / a_ii for all i at once.
            x = (b - off @ x_old) / d
        else:
            # SOR (B&F Alg. 7.3), in place, so x_j for j < i is already new:
            # x_i ← (1 − ω) x_i + ω (b_i − Σ_{j<i} a_ij x_j − Σ_{j>i} a_ij x_j) / a_ii.
            sweep = [x.copy()]
            for i in range(n):
                sigma = A[i, :i] @ x[:i] + A[i, i + 1 :] @ x[i + 1 :]
                x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sigma) / d[i]
                sweep.append(x.copy())
            info["sweep"] = sweep
        r = b - A @ x
        rnorm = norm2(r)
        step = norm2(x - x_old)
        info = {**_residual_info(r, rnorm, scale), **info}
        trace.append(Step(k, x.copy(), rnorm, step_size=step, info=info))
        if not (np.all(np.isfinite(x)) and np.isfinite(rnorm)):
            msg = f"non-finite iterate at k = {k}{_divergence_note(rho)}"
            return Result(method, x, rnorm, False, msg, k, trace=trace, extra=extra)
        if rnorm <= tol * scale.value:
            ok, msg = _certify(rnorm, x, a_norm, b, tol, scale, _converged_msg(rnorm, scale))
            return Result(method, x, rnorm, ok, msg, k, trace=trace, extra=extra)
    msg = f"reached max_iter={max_iter} with {scale.ratio(rnorm)}{_divergence_note(rho)}"
    return Result(method, x, rnorm, False, msg, trace[-1].k, trace=trace, extra=extra)


def _optimal_omega(A: np.ndarray, omega: float) -> dict[str, Any]:
    """Young's optimal relaxation factor, when the theory applies.

    Young (1950); Burden & Faires §7.4; Saad (2003) §4.2: if A is consistently ordered,
    the eigenvalues of the Jacobi matrix G_J are real and ρ_J = ρ(G_J) < 1, then

        ω* = 2 / (1 + √(1 − ρ_J²)),   ρ(G_ω*) = ω* − 1.

    # NOTE: consistent ordering is checked by the sufficient condition "A is tridiagonal" (every
    # tridiagonal matrix with a nonzero diagonal is consistently ordered; Saad §4.2). For other
    # matrices ω* is reported as ``None`` even when it exists.
    """
    eig = np.asarray(np.linalg.eigvals(_jacobi_matrix(A)), dtype=np.complex128)
    rho_j = float(np.max(np.abs(eig)))
    out: dict[str, Any] = {
        "omega": omega,
        "omega_opt": None,
        "spectral_radius_opt": None,
        "spectral_radius_jacobi": rho_j,
    }
    if not _is_tridiagonal(A):
        out["omega_opt_note"] = "A is not tridiagonal: consistent ordering is not verified"
    elif float(np.max(np.abs(np.imag(eig)))) > 1e-12 * max(1.0, rho_j):
        out["omega_opt_note"] = "the Jacobi matrix has complex eigenvalues"
    elif rho_j >= 1.0:
        out["omega_opt_note"] = f"ρ(G_J) = {rho_j:.4g} ≥ 1"
    else:
        w = 2.0 / (1.0 + float(np.sqrt(1.0 - rho_j**2)))
        out["omega_opt"] = w
        out["spectral_radius_opt"] = w - 1.0
        out["omega_opt_note"] = "Young's formula (A tridiagonal, real Jacobi spectrum, ρ_J < 1)"
    return out


@register(
    id="jacobi",
    family="linalg",
    name="Jacobi iteration",
    params=(TOL, MAX_ITER),
    needs=("A", "b"),
    order="linear (rate ρ(G_J))",
    summary="Solve equation i for xᵢ using the previous iterate for every other unknown.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.1",
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1",
    ),
)
def jacobi(
    problem: SystemLike,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Jacobi iteration.

    x_i^(k) = (b_i − Σ_{j≠i} a_ij x_j^(k−1)) / a_ii   (Burden & Faires Alg. 7.1), that is
    x^(k) = G_J x^(k−1) + D⁻¹b with G_J = −D⁻¹(L + U). It converges for every x₀ iff
    ρ(G_J) < 1; strict diagonal dominance is sufficient (B&F §7.3).

    Stopping: ‖r_k‖₂ ≤ tol·d (converged; d = ‖b‖₂, or ‖r₀‖₂ when b = 0), a non-finite iterate,
    or max_iter. A zero diagonal entry stops at k = 0. extra: spectral_radius = ρ(G_J), also in
    the k = 0 Step.
    """
    return _stationary("jacobi", problem, x0, tol, max_iter, None)


@register(
    id="gauss_seidel",
    family="linalg",
    name="Gauss–Seidel iteration",
    params=(TOL, MAX_ITER),
    needs=("A", "b"),
    order="linear (rate ρ(G_GS))",
    summary="Like Jacobi, but each new component is used as soon as it is computed.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.2",
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1",
    ),
)
def gauss_seidel(
    problem: SystemLike,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Gauss–Seidel iteration.

    x_i^(k) = (b_i − Σ_{j<i} a_ij x_j^(k) − Σ_{j>i} a_ij x_j^(k−1)) / a_ii  (B&F Alg. 7.2), that
    is G_GS = −(D + L)⁻¹U. It converges for strictly diagonally dominant A and for every SPD A
    (Ostrowski–Reich; Saad §4.2). Each Step records the coordinate-wise ``sweep``.

    Stopping: ‖r_k‖₂ ≤ tol·d (converged; d = ‖b‖₂, or ‖r₀‖₂ when b = 0), a non-finite iterate,
    or max_iter. A zero diagonal entry stops at k = 0. extra: spectral_radius = ρ(G_GS), also in
    the k = 0 Step.
    """
    return _stationary("gauss_seidel", problem, x0, tol, max_iter, 1.0)


@register(
    id="sor",
    family="linalg",
    name="Successive over-relaxation (SOR)",
    params=(
        ParamSpec(
            "omega",
            1.5,
            min=0.05,
            max=1.95,
            help="Relaxation factor ω ∈ (0, 2); ω = 1 is Gauss–Seidel, ω > 1 over-relaxes.",
        ),
        TOL,
        MAX_ITER,
    ),
    needs=("A", "b"),
    order="linear (rate ρ(G_ω))",
    summary="Gauss–Seidel with each component step stretched by a relaxation factor ω.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 7.3 and §7.4",
        "Young, Iterative Solution of Large Linear Systems (1971)",
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §4.1–4.2",
    ),
)
def sor(
    problem: SystemLike,
    *,
    x0: Any = None,
    omega: float = 1.5,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Successive over-relaxation.

    x_i^(k) = (1 − ω) x_i^(k−1) + ω (b_i − Σ_{j<i} a_ij x_j^(k) − Σ_{j>i} a_ij x_j^(k−1)) / a_ii
    (B&F Alg. 7.3), that is G_ω = (D + ωL)⁻¹((1 − ω)D − ωU). Kahan: ρ(G_ω) ≥ |ω − 1|, so SOR can
    converge only for 0 < ω < 2; for SPD A it converges for every such ω (Ostrowski–Reich).

    Raises ValueError for ω ∉ (0, 2). Stopping: ‖r_k‖₂ ≤ tol·d (converged; d = ‖b‖₂, or ‖r₀‖₂
    when b = 0), a non-finite iterate, or max_iter. A zero diagonal entry stops at k = 0.
    extra: spectral_radius = ρ(G_ω),
    omega, spectral_radius_jacobi, omega_opt_note and, when Young's theory applies (see
    :func:`_optimal_omega`), omega_opt = ω* and spectral_radius_opt = ω* − 1 (else ``None``).
    """
    if not 0.0 < omega < 2.0:
        raise ValueError(f"omega must lie in (0, 2) (Kahan's theorem), got {omega}")
    return _stationary("sor", problem, x0, tol, max_iter, float(omega))


# --------------------------------------------------------------------------------------
# Steepest descent, conjugate gradients, preconditioned CG (symmetric positive definite A)
# --------------------------------------------------------------------------------------


#: Admissible overall scale max|a_ij|, max|b_i| for the methods that form rᵀr and pᵀAp.
SCALE_RANGE = (2.0**-200, 2.0**200)


def _check_scale(A: np.ndarray, b: Vector) -> None:
    """Reject data whose scale makes the inner products rᵀr, pᵀAp underflow or overflow.

    # NOTE: the textbook recurrences of steepest descent and CG divide rᵀr by pᵀAp. These inner
    # products are formed as (u/s)ᵀ(v/s) with a power of 2 s (see :func:`_scaled_dot`), so a large
    # x₀ (hence a large r₀) cannot overflow them. With max|a_ij| or max|b_i| outside
    # [2⁻²⁰⁰, 2²⁰⁰], the products A·p, the iterates and tol·‖b‖ come near the ends of the float64
    # range and the method could report a false breakdown, so such input is rejected; rescaling A
    # or b by a power of 2 is exact and fixes it. Zero data are allowed.
    """
    lo, hi = SCALE_RANGE
    for name, M in (("A", A), ("b", b)):
        s = float(np.max(np.abs(M), initial=0.0))
        if s != 0.0 and not lo <= s <= hi:
            raise ValueError(
                f"max |{name}| = {s:.3g} lies outside [2^-200, 2^200]: rescale the system "
                "(the inner products rᵀr and pᵀAp would underflow or overflow)"
            )


def _overflowed_start(method: str, x: Vector, r: Vector, scale: _Scale) -> Result | None:
    """A Result with converged=False when r₀ = b − A x₀ is not finite (x₀ too large), else None."""
    if np.all(np.isfinite(r)):
        return None
    rnorm = norm2(r)
    trace = [Step(0, x.copy(), rnorm, info=_residual_info(r, rnorm, scale))]
    msg = (
        "the initial residual r₀ = b − A x₀ overflows (‖x₀‖ = "
        f"{norm2(x):.3g}): x₀ is too large for the float64 range"
    )
    return Result(method, x, rnorm, False, msg, 0, trace=trace, extra={"n_matvec": 1})


def _kappa(S: np.ndarray) -> float | None:
    """κ₂(S) = λ_max/λ_min for a symmetric S; ``None`` unless S is positive definite."""
    lam = np.linalg.eigvalsh(S)
    return float(lam[-1] / lam[0]) if lam[0] > 0.0 else None


def _not_symmetric(method: str, A: np.ndarray, x: Vector, r: Vector, scale: _Scale) -> Result:
    rnorm = norm2(r)
    trace = [Step(0, x, rnorm, info=_residual_info(r, rnorm, scale))]
    asym = float(np.max(np.abs(A - A.T)))
    msg = f"A is not symmetric (max |a_ij − a_ji| = {asym:.3g}); {method} needs an SPD matrix"
    return Result(method, x, rnorm, False, msg, 0, trace=trace, extra={"n_matvec": 1})


def _phi(x: Vector, b: Vector, r: Vector) -> float:
    """φ(x) = ½xᵀAx − bᵀx = −½xᵀ(b + r), using A x = b − r."""
    return float(-0.5 * (x @ (b + r)))


def _finish(
    method: str,
    A: np.ndarray,
    b: Vector,
    x: Vector,
    trace: list[Step],
    tol: float,
    scale: _Scale,
    n_matvec: int,
    extra: dict[str, Any],
    passed: bool,
    message: str,
) -> Result:
    """Recompute the true residual once and build the Result.

    ``passed`` says whether the recurrence residual met the tolerance; converged additionally
    requires the true residual ‖b − A x‖₂ ≤ tol·d and the rounding-level test of the module
    docstring. ``Result.fun`` is the true residual ‖b − A x‖₂ (the recurrence value stays in
    ``trace[-1].fun``).
    """
    true_norm = norm2(b - A @ x)
    out = {**extra, "n_matvec": n_matvec + 1, "true_residual_norm": true_norm}
    converged = False
    if passed:
        if bool(np.isfinite(true_norm)) and true_norm <= tol * scale.value:
            converged, message = _certify(true_norm, x, norm2(A), b, tol, scale, message)
        else:
            message = (
                f"the updated residual met tol, but the true residual {scale.ratio(true_norm)} "
                "does not (rounding-error residual gap)"
            )
    last = trace[-1]
    return Result(method, x, true_norm, converged, message, last.k, trace=trace, extra=out)


def _converged_msg(rnorm: float, scale: _Scale) -> str:
    return f"{scale.ratio(rnorm)} ≤ tol"


@register(
    id="steepest_descent_linear",
    family="linalg",
    name="Steepest descent (SPD system)",
    params=(TOL, MAX_ITER),
    needs=("A", "b"),
    order="linear (A-norm rate (κ − 1)/(κ + 1))",
    summary="Minimize ½xᵀAx − bᵀx by exact line searches along the residual r = b − Ax.",
    references=(
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §5.3.1",
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), §3.3, eq. (3.29)",
        "Shewchuk (1994), An Introduction to the Conjugate Gradient Method Without the "
        "Agonizing Pain, §4",
    ),
)
@quiet
def steepest_descent_linear(
    problem: SystemLike,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Steepest descent with exact line search on φ(x) = ½xᵀAx − bᵀx (A SPD).

    r_k = b − A x_k = −∇φ(x_k);  α_k = r_kᵀr_k / r_kᵀA r_k;  x_{k+1} = x_k + α_k r_k;
    r_{k+1} = r_k − α_k A r_k  (Saad §5.3.1). Consecutive residuals are orthogonal, which gives
    the zig-zag path, and ‖e_{k+1}‖_A ≤ ((κ − 1)/(κ + 1)) ‖e_k‖_A (N&W eq. 3.29).

    # NOTE: the residual is updated by the recurrence (one product A·r per iteration) instead of
    # being recomputed as b − A x (two products); the true residual is recomputed once on exit.

    Fails cleanly (k = 0) when A is not symmetric; stops with converged=False when r_kᵀA r_k ≤ 0
    (A is not positive definite), on a non-finite value, or at max_iter.
    Raises ValueError when max|a_ij| or max|b_i| lies outside [2⁻²⁰⁰, 2²⁰⁰] (see SCALE_RANGE).
    Stopping: ‖r_k‖₂ ≤ tol·d (d = ‖b‖₂, or ‖r₀‖₂ when b = 0), confirmed on exit by
    ‖b − A x‖₂ ≤ tol·d.
    extra: n_matvec, true_residual_norm, condition_number, rate_bound.
    """
    method = "steepest_descent_linear"
    _check_tol(tol)
    max_iter = _check_count("max_iter", max_iter)
    A, b = resolve_system(problem)
    _check_scale(A, b)
    x = _start(x0, b.size)
    r = b - A @ x
    n_matvec = 1
    scale = _scale(A, b, x, r)
    if not is_symmetric(A):
        return _not_symmetric(method, A, x, r, scale)
    kappa = _kappa(A)
    rate = (kappa - 1.0) / (kappa + 1.0) if kappa is not None else None
    extra: dict[str, Any] = {"condition_number": kappa, "rate_bound": rate}

    failed = _overflowed_start(method, x, r, scale)
    if failed is not None:
        return failed
    # NOTE: rᵀr and rᵀAr are both divided by the same s² (s = _pow2_scale(r)), which cancels in
    # α = rᵀr / rᵀAr; this is exact, so the iterates are those of the textbook recurrence.
    s_r = _pow2_scale(r)
    rr = _scaled_dot(r, r, s_r)  # r_kᵀr_k / s_r²
    rnorm = norm2(r)
    info0 = {
        **_residual_info(r, rnorm, scale),
        "direction": r.copy(),
        "alpha": None,
        "phi": _phi(x, b, r),
        "condition_number": kappa,
        "rate_bound": rate,
    }
    trace = [Step(0, x.copy(), rnorm, grad_norm=rnorm, info=info0)]
    if rnorm <= tol * scale.value:
        msg = "x0 already satisfies the tolerance"
        return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, True, msg)

    for k in range(1, max_iter + 1):
        q = A @ r
        n_matvec += 1
        rq = _scaled_dot(r, q, s_r)  # r_kᵀA r_k / s_r²
        if not rq > 0.0:
            msg = f"rᵀAr = {rq * s_r**2:.3g} ≤ 0 at k = {k}: A is not positive definite"
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)
        alpha = rr / rq
        step = abs(alpha) * rnorm  # ‖α_{k−1} r_{k−1}‖₂
        x = x + alpha * r
        r = r - alpha * q
        s_r = _pow2_scale(r)
        rr = _scaled_dot(r, r, s_r)
        rnorm = norm2(r)
        info = {
            **_residual_info(r, rnorm, scale),
            "direction": r.copy(),
            "alpha": alpha,
            "phi": _phi(x, b, r),
        }
        trace.append(Step(k, x.copy(), rnorm, grad_norm=rnorm, step_size=step, info=info))
        if not (np.all(np.isfinite(x)) and np.isfinite(rnorm)):
            msg = f"non-finite iterate at k = {k}"
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)
        if rnorm <= tol * scale.value:
            msg = _converged_msg(rnorm, scale)
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, True, msg)
    msg = f"reached max_iter={max_iter} with {scale.ratio(rnorm)}"
    return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)


@quiet
def _cg(
    method: str,
    problem: SystemLike,
    x0: Any,
    tol: float,
    max_iter: int,
    preconditioned: bool,
) -> Result:
    """CG (N&W Alg. 5.2) or Jacobi-preconditioned CG (N&W Alg. 5.3), with r = b − A x.

    # NOTE: rᵀz and pᵀAp are formed as (r/s_r)ᵀ(z/s_r) and (p/s_p)ᵀ(Ap/s_p) with powers of 2 s_r,
    # s_p, and α = (rᵀz/s_r²)/(pᵀAp/s_p²)·(s_r/s_p)², β likewise. Scaling by a power of 2 is exact,
    # so α and β equal the textbook values bit for bit; only the overflow of rᵀr for a large x₀
    # is gone.
    """
    _check_tol(tol)
    max_iter = _check_count("max_iter", max_iter)
    A, b = resolve_system(problem)
    _check_scale(A, b)
    n = b.size
    x = _start(x0, n)
    r = b - A @ x
    n_matvec = 1
    scale = _scale(A, b, x, r)
    if not is_symmetric(A):
        return _not_symmetric(method, A, x, r, scale)
    d = np.diag(A).copy()
    kappa = _kappa(A)
    extra: dict[str, Any] = {"condition_number": kappa}
    info0: dict[str, Any] = {"condition_number": kappa}
    if preconditioned:
        if not np.all(d > 0.0):
            rnorm = norm2(r)
            trace = [Step(0, x, rnorm, info=_residual_info(r, rnorm, scale))]
            i = int(np.flatnonzero(d <= 0.0)[0])
            msg = (
                f"a_{i}{i} = {d[i]:.3g} ≤ 0: the Jacobi preconditioner M = diag(A) is not "
                "positive definite (A is not SPD)"
            )
            return Result(method, x, rnorm, False, msg, 0, trace=trace, extra={"n_matvec": 1})
        s = np.sqrt(d)
        kappa_eff = _kappa(A / np.outer(s, s))  # D^{−1/2} A D^{−1/2}
        extra["preconditioned_condition_number"] = kappa_eff
        info0["preconditioned_condition_number"] = kappa_eff
    else:
        kappa_eff = kappa
    rate = (np.sqrt(kappa_eff) - 1.0) / (np.sqrt(kappa_eff) + 1.0) if kappa_eff else None
    rate = float(rate) if rate is not None else None
    extra["rate_bound"] = rate
    info0["rate_bound"] = rate

    failed = _overflowed_start(method, x, r, scale)
    if failed is not None:
        return failed
    z = r / d if preconditioned else r
    p = z.copy()
    s_r = _pow2_scale(r)
    rz = _scaled_dot(r, z, s_r)  # r_kᵀz_k / s_r²
    rnorm = norm2(r)
    pre: dict[str, Any] = {"preconditioned_residual": z.copy()} if preconditioned else {}
    info0 = {
        **_residual_info(r, rnorm, scale),
        **pre,
        "direction": p.copy(),
        "alpha": None,
        "beta": None,
        "phi": _phi(x, b, r),
        **info0,
    }
    trace = [Step(0, x.copy(), rnorm, grad_norm=rnorm, info=info0)]
    if rnorm <= tol * scale.value:
        msg = "x0 already satisfies the tolerance"
        return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, True, msg)

    for k in range(1, max_iter + 1):
        q = A @ p
        n_matvec += 1
        s_p = _pow2_scale(p)
        pq = _scaled_dot(p, q, s_p)  # p_kᵀA p_k / s_p²
        if not pq > 0.0:
            msg = f"pᵀAp = {pq * s_p**2:.3g} ≤ 0 at k = {k}: A is not positive definite"
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)
        alpha = (rz / pq) * (s_r / s_p) ** 2
        step = abs(alpha) * norm2(p)
        x = x + alpha * p
        r = r - alpha * q
        z = r / d if preconditioned else r
        s_new = _pow2_scale(r)
        rz_new = _scaled_dot(r, z, s_new)
        beta = (rz_new / rz) * (s_new / s_r) ** 2
        p = z + beta * p
        rz, s_r = rz_new, s_new
        rnorm = norm2(r)
        pre = {"preconditioned_residual": z.copy()} if preconditioned else {}
        info = {
            **_residual_info(r, rnorm, scale),
            **pre,
            "direction": p.copy(),
            "alpha": alpha,
            "beta": beta,
            "phi": _phi(x, b, r),
        }
        trace.append(Step(k, x.copy(), rnorm, grad_norm=rnorm, step_size=step, info=info))
        if not (np.all(np.isfinite(x)) and np.isfinite(rnorm)):
            msg = f"non-finite iterate at k = {k}"
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)
        if rnorm <= tol * scale.value:
            msg = _converged_msg(rnorm, scale)
            return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, True, msg)
    msg = f"reached max_iter={max_iter} with {scale.ratio(rnorm)}"
    return _finish(method, A, b, x, trace, tol, scale, n_matvec, extra, False, msg)


@register(
    id="conjugate_gradient_linear",
    family="linalg",
    name="Conjugate gradient (SPD system)",
    params=(TOL, MAX_ITER),
    needs=("A", "b"),
    order="≤ n steps in exact arithmetic; A-norm rate (√κ − 1)/(√κ + 1)",
    summary="Search along A-conjugate directions; each step minimizes ½xᵀAx − bᵀx over a growing Krylov space.",
    references=(
        "Hestenes & Stiefel (1952), J. Res. Nat. Bur. Standards 49(6)",
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 5.2 and eq. (5.36)",
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 38.1",
    ),
)
def conjugate_gradient_linear(
    problem: SystemLike,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Linear conjugate gradient method (Hestenes–Stiefel 1952; N&W Alg. 5.2).

    With r = b − A x (N&W use r = A x − b, the opposite sign):

        p₀ = r₀;  α_k = r_kᵀr_k / p_kᵀA p_k;  x_{k+1} = x_k + α_k p_k;  r_{k+1} = r_k − α_k A p_k;
        β_{k+1} = r_{k+1}ᵀr_{k+1} / r_kᵀr_k;  p_{k+1} = r_{k+1} + β_{k+1} p_k.

    In exact arithmetic the residuals are mutually orthogonal, the directions are A-conjugate,
    x_k minimizes φ over x₀ + K_k(A, r₀), and the method terminates in at most n steps;
    ‖e_k‖_A ≤ 2((√κ − 1)/(√κ + 1))^k ‖e₀‖_A (N&W eq. 5.36). One product A·p per iteration.

    Fails cleanly (k = 0) when A is not symmetric; stops with converged=False when p_kᵀA p_k ≤ 0
    (A is not positive definite), on a non-finite value, or at max_iter.
    Raises ValueError when max|a_ij| or max|b_i| lies outside [2⁻²⁰⁰, 2²⁰⁰] (see SCALE_RANGE).
    Stopping: ‖r_k‖₂ ≤ tol·d (d = ‖b‖₂, or ‖r₀‖₂ when b = 0), confirmed on exit by
    ‖b − A x‖₂ ≤ tol·d.
    extra: n_matvec, true_residual_norm, condition_number, rate_bound.
    """
    return _cg("conjugate_gradient_linear", problem, x0, tol, max_iter, preconditioned=False)


@register(
    id="preconditioned_cg",
    family="linalg",
    name="Preconditioned CG (Jacobi preconditioner)",
    params=(TOL, MAX_ITER),
    needs=("A", "b"),
    order="A-norm rate (√κ̃ − 1)/(√κ̃ + 1), κ̃ = κ₂(D^{-1/2} A D^{-1/2})",
    summary="CG on the diagonally scaled system: residuals are divided by diag(A) before each update.",
    references=(
        "Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 5.3",
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), §9.2",
    ),
)
def preconditioned_cg(
    problem: SystemLike,
    *,
    x0: Any = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Preconditioned conjugate gradient with the Jacobi preconditioner M = diag(A) (N&W Alg. 5.3).

    With r = b − A x and z = M⁻¹r:

        p₀ = z₀;  α_k = r_kᵀz_k / p_kᵀA p_k;  x_{k+1} = x_k + α_k p_k;  r_{k+1} = r_k − α_k A p_k;
        z_{k+1} = M⁻¹ r_{k+1};  β_{k+1} = r_{k+1}ᵀz_{k+1} / r_kᵀz_k;  p_{k+1} = z_{k+1} + β_{k+1} p_k.

    This is CG applied to the SPD system (D^{−1/2} A D^{−1/2}) (D^{1/2} x) = D^{−1/2} b, so its
    rate depends on κ̃ = κ₂(D^{−1/2} A D^{−1/2}) instead of κ₂(A).

    Fails cleanly (k = 0) when A is not symmetric or a diagonal entry is ≤ 0; stops with
    converged=False when p_kᵀA p_k ≤ 0, on a non-finite value, or at max_iter.
    Raises ValueError when max|a_ij| or max|b_i| lies outside [2⁻²⁰⁰, 2²⁰⁰] (see SCALE_RANGE).
    Stopping: ‖r_k‖₂ ≤ tol·d on the unpreconditioned residual (d = ‖b‖₂, or ‖r₀‖₂ when b = 0),
    confirmed on exit by ‖b − A x‖₂ ≤ tol·d. extra: n_matvec, true_residual_norm, condition_number,
    preconditioned_condition_number, rate_bound.
    """
    return _cg("preconditioned_cg", problem, x0, tol, max_iter, preconditioned=True)


# --------------------------------------------------------------------------------------
# GMRES
# --------------------------------------------------------------------------------------


@register(
    id="gmres",
    family="linalg",
    name="GMRES (restarted)",
    params=(
        ParamSpec(
            "restart",
            20,
            kind="int",
            min=1,
            max=500,
            help="Krylov dimension m of a cycle before restarting from the current x (GMRES(m)).",
        ),
        TOL,
        MAX_ITER,
    ),
    needs=("A", "b"),
    order="minimal residual over x₀ + K_j(A, r₀); full GMRES ends in ≤ n steps",
    summary="Minimize ‖b − Ax‖₂ over a growing Krylov space built by Arnoldi; restart every m steps.",
    references=(
        "Saad & Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3)",
        "Saad, Iterative Methods for Sparse Linear Systems (2nd ed., 2003), Alg. 6.2, 6.9, 6.11, §6.5.3",
        "Trefethen & Bau, Numerical Linear Algebra (1997), Alg. 35.1",
        "Brown & Walker (1997), GMRES on (nearly) singular systems, SIAM J. Matrix Anal. Appl. 18(1)",
    ),
)
@quiet
def gmres(
    problem: SystemLike,
    *,
    x0: Any = None,
    restart: int = 20,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Restarted GMRES(m) with modified Gram–Schmidt Arnoldi and Givens rotations.

    Cycle (Saad Alg. 6.9 / 6.11): r₀ = b − A x₀, β = ‖r₀‖₂, v₁ = r₀/β. For j = 1 … m:

        w = A v_j;  h_ij = wᵀv_i, w ← w − h_ij v_i  (i = 1 … j, modified Gram–Schmidt, Alg. 6.2);
        h_{j+1,j} = ‖w‖₂;  v_{j+1} = w / h_{j+1,j}.

    # NOTE: the Gram–Schmidt loop runs twice (the second pass adds its coefficients to h_ij), and
    # w is set to 0 when the second pass shrinks it below ‖w′‖/√2 (Kahan–Parlett "twice is
    # enough": Parlett, The Symmetric Eigenvalue Problem (1998), §6.9; Giraud, Langou, Rozložník &
    # van den Eshof (2005), Numer. Math. 101). One MGS pass loses orthogonality by about
    # ε‖A v_j‖/h_{j+1,j}: when h_{j+1,j} only narrowly exceeds the happy-breakdown level (a start
    # residual r₀ = fl(b − A x₀) that carries rounding noise, e.g. a large x₀), v_{j+1} is a
    # normalized noise vector far from orthogonal to V_j, |r_jj| collapses on a well-conditioned A,
    # and the rank test below reported a false "A is singular" (poisson_1d_10, x₀ = 1e160·𝟙).
    # With the second pass V_{j+1} is orthonormal to working precision.

    The (j+1) × j Hessenberg matrix H̄_j is reduced to upper-triangular R_j by Givens rotations
    Ω_i = [[c_i, s_i], [−s_i, c_i]] with c = h/√(h² + h′²), s = h′/√(h² + h′²) (Saad §6.5.3), applied
    also to g = βe₁. Then |g_{j+1}| = min_y ‖βe₁ − H̄_j y‖₂ = ‖b − A x_j‖₂, with
    x_j = x₀ + V_j y_j and R_j y_j = g_{1:j}. After m steps (or on success) x ← x_m and the next
    cycle starts from the true residual b − A x.

    # NOTE: x_j and r_j = r₀ − A V_j y_j are formed at every inner step (not only at the end of a
    # cycle) so that the visualizer can draw every iterate. r_j uses the Arnoldi relation
    # A V_j = V_j H_j + w e_jᵀ (w = h_{j+1,j} v_{j+1}, before normalization), so
    # r_j = V_j(βe₁ − H_j y_j) − y_j w; this costs O(nj) work and no extra product with A. The
    # cycle length is min(restart, n): a Krylov space has dimension at most n.

    A happy breakdown h_{j+1,j} ≤ ε‖A v_j‖₂ means that K_j is A-invariant and x_j is exact (for a
    nonsingular A); the cycle ends there. The same holds at j = n (dim K_j ≤ n), whatever the
    computed h_{n+1,n}. If the rotated diagonal r_jj = √(h² + h′²) ≤
    τ = n·ε·‖A‖_F, the projected least-squares matrix H̄_j is numerically rank-deficient (A is
    singular, to working precision, on the Krylov space; Brown & Walker 1997): stop with
    converged=False. For a nonsingular A and V_{j+1} orthonormal (to working precision, by the
    reorthogonalization), |r_jj| ≥ σ_min(H̄_j) = σ_min(A V_j) ≥ σ_min(A), so the test fires only
    when σ_min(A) ≲ τ, the rank threshold of the direct methods.

    # NOTE: a test relative to ‖A v_j‖₂ (instead of τ) is not scale-aware: on the singular,
    # inconsistent system singular_3 with b = e₁ it accepts |r_jj|/max|r_ii| ≈ 6e-17, and the
    # solve with that R_j gives ‖x‖ ≈ 1e15.

    Stopping: |g_{j+1}| ≤ tol·d ends a cycle (d = ‖b‖₂, or ‖r₀‖₂ when b = 0); converged=True only
    when the recomputed true residual ‖b − A x‖₂ ≤ tol·d and the rounding-level test of the
    module docstring passes (if the residual passes but the rounding level does not, GMRES stops
    with converged=False; otherwise it restarts). max_iter counts inner steps. ``Result.fun`` is the true residual.
    Raises ValueError unless restart and max_iter are integers ≥ 1, or when ‖A‖_F or ‖b‖₂
    overflows. A non-finite r₀ = b − A x₀ stops at k = 0 with converged=False.
    extra: n_matvec, true_residual_norm, restart, cycles.
    """
    method = "gmres"
    restart = _check_count("restart", restart)
    max_iter = _check_count("max_iter", max_iter)
    _check_tol(tol)
    A, b = resolve_system(problem)
    a_norm = _check_norms(A, b)
    n = b.size
    x = _start(x0, n)
    m = min(restart, n)
    tau = pivot_tolerance(A)  # rank threshold for the rotated diagonal r_jj

    r = b - A @ x
    n_matvec = 1
    scale = _scale(A, b, x, r)
    failed = _overflowed_start(method, x, r, scale)
    if failed is not None:
        return failed
    beta = norm2(r)
    info0 = {
        **_residual_info(r, beta, scale),
        "cycle": 0,
        "krylov_dim": 0,
        "hessenberg": None,
        "givens": None,
        "basis_vector": r / beta if beta > 0.0 else None,
    }
    trace = [Step(0, x.copy(), beta, info=info0)]

    def result(converged: bool, message: str, cycles: int) -> Result:
        extra = {
            "n_matvec": n_matvec,
            "true_residual_norm": beta,
            "restart": m,
            "cycles": cycles,
        }
        # Result.fun is the true residual β = ‖b − A x‖₂ (trace[-1].fun is the Givens value).
        return Result(method, x, beta, converged, message, trace[-1].k, trace=trace, extra=extra)

    if beta <= tol * scale.value:
        return result(
            *_certify(beta, x, a_norm, b, tol, scale, "x0 already satisfies the tolerance"), 0
        )

    k = 0
    cycle = 0
    while True:
        V = np.zeros((n, m + 1))
        V[:, 0] = r / beta
        H = np.zeros((m + 1, m))  # H̄ as built by Arnoldi
        R = np.zeros((m + 1, m))  # H̄ after the Givens rotations
        g = np.zeros(m + 1)
        g[0] = beta
        cs = np.zeros(m)
        sn = np.zeros(m)
        x_start = x.copy()
        x_j = x.copy()
        breakdown = False
        for j in range(m):
            w = A @ V[:, j]
            n_matvec += 1
            w_norm0 = norm2(w)
            w_norm1 = 0.0
            for sweep in range(2):  # modified Gram–Schmidt, then one reorthogonalization pass
                for i in range(j + 1):
                    h_ij = float(w @ V[:, i])
                    H[i, j] += h_ij
                    w = w - h_ij * V[:, i]
                if sweep == 0:
                    w_norm1 = norm2(w)
            h_next = norm2(w)
            if h_next < w_norm1 / np.sqrt(2.0):  # Kahan–Parlett: w ∈ span(V_j) numerically
                w = np.zeros(n)
                h_next = 0.0
            H[j + 1, j] = h_next
            R[: j + 2, j] = H[: j + 2, j]
            for i in range(j):  # apply the earlier rotations to the new column
                t = cs[i] * R[i, j] + sn[i] * R[i + 1, j]
                R[i + 1, j] = -sn[i] * R[i, j] + cs[i] * R[i + 1, j]
                R[i, j] = t
            den = float(np.hypot(R[j, j], R[j + 1, j]))
            k += 1
            if den <= tau:
                x = x_j
                r = b - A @ x
                n_matvec += 1
                beta = norm2(r)
                info = {
                    **_residual_info(r, beta, scale),
                    "cycle": cycle,
                    "krylov_dim": j + 1,
                    "hessenberg": H[: j + 2, : j + 1].copy(),
                    "givens": None,
                    "basis_vector": None,
                }
                trace.append(Step(k, x.copy(), beta, step_size=0.0, info=info))
                return result(
                    False,
                    f"GMRES breakdown at k = {k}: the rotated diagonal |r_jj| = {den:.3g} ≤ "
                    f"τ = {tau:.3g}, so the projected least-squares matrix is singular to "
                    "working precision (A is singular on the Krylov space)",
                    cycle + 1,
                )
            cs[j] = R[j, j] / den
            sn[j] = R[j + 1, j] / den
            R[j, j] = den
            R[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            res_est = abs(float(g[j + 1]))
            y = back_substitution(R[: j + 1, : j + 1], g[: j + 1])
            x_prev = x_j
            x_j = x_start + V[:, : j + 1] @ y
            c = -(H[: j + 1, : j + 1] @ y)  # βe₁ − H_j y_j
            c[0] += beta
            r_j = V[:, : j + 1] @ c - y[j] * w
            # NOTE: K_{j+1} ⊆ ℝⁿ, so for j + 1 = n there is no further Arnoldi vector: in exact
            # arithmetic h_{n+1,n} = 0. The computed h_{n+1,n} is rounding noise (≈ 1e-14 on
            # nonsymmetric_4) that can exceed ε‖A v_j‖, and w/h would be a normalized noise vector
            # far from orthogonal to V_n, so the space is treated as exhausted.
            breakdown = j + 1 == n or H[j + 1, j] <= EPS * w_norm0
            v_next = None if breakdown else w / H[j + 1, j]
            info = {
                **_residual_info(r_j, res_est, scale),
                "cycle": cycle,
                "krylov_dim": j + 1,
                "hessenberg": H[: j + 2, : j + 1].copy(),
                "givens": [float(cs[j]), float(sn[j])],
                "basis_vector": v_next,
            }
            step = norm2(x_j - x_prev)
            trace.append(Step(k, x_j.copy(), res_est, step_size=step, info=info))
            if not np.all(np.isfinite(x_j)):
                x = x_j
                beta = float("nan")
                return result(False, f"non-finite iterate at k = {k}", cycle + 1)
            if res_est <= tol * scale.value or v_next is None or k >= max_iter:
                break
            V[:, j + 1] = v_next
        x = x_j
        r = b - A @ x
        n_matvec += 1
        beta = norm2(r)
        cycle += 1
        if beta <= tol * scale.value:
            msg = f"true residual {scale.ratio(beta)} ≤ tol"
            return result(*_certify(beta, x, a_norm, b, tol, scale, msg), cycle)
        if not np.isfinite(beta):
            return result(False, f"non-finite residual after cycle {cycle}", cycle)
        if k >= max_iter:
            msg = f"reached max_iter={max_iter} with true {scale.ratio(beta)}"
            return result(False, msg, cycle)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("jacobi", "spd_2x2", {}),
    ("jacobi", "jacobi_diverges", {"max_iter": 25}),
    ("gauss_seidel", "jacobi_diverges", {}),
    ("sor", "poisson_1d_10", {"omega": 1.56}),
    ("steepest_descent_linear", "spd_2x2", {}),
    ("conjugate_gradient_linear", "poisson_1d_10", {}),
    # Not hilbert_5: with κ = 4.8e5 the CG iterates lose orthogonality, and rounding moves
    # iterate 5 by 7e-5 between BLAS kernels. spd_2x2 has an unequal diagonal for Jacobi.
    ("preconditioned_cg", "spd_2x2", {}),
    ("gmres", "nonsymmetric_4", {"restart": 2}),
]
