"""Independent reference solutions (SciPy) for the study; never used by the methods themselves.

lasso_solution(A, b, lam): the minimizer of F(x) = ½‖Ax − b‖² + λ‖x‖₁, certified by the KKT
conditions. It does not reuse any code from ``method.py``:

1. Solve the smooth bound-constrained split form min ½‖A(u − v) − b‖² + λ1ᵀ(u + v), u, v ≥ 0
   with SciPy's L-BFGS-B; x̃ = u − v.
2. Polish on the support S of x̃: with s = sign(x̃_S), the lasso optimality conditions on S are
   A_Sᵀ(A_S x_S − b) + λ s = 0, solved with the QR factors of A_S (see ``_polish``).
3. Certify: sign(x_S) = s, and |A_jᵀ r| < λ for every j ∉ S (strict complementarity). These are
   the KKT conditions of the convex problem, so x is a global minimizer; with A_S of full column
   rank and the strict inequality, it is the unique one (Tibshirani 2013, Lemma 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LassoCertificate:
    x: Array  # (n,) the minimizer
    F: float  # F(x)
    support: tuple[int, ...]
    margin: float  # λ − max_{j ∉ S} |A_jᵀ r| (> 0: strict complementarity)
    min_abs: float  # min_{i ∈ S} |x_i|
    stationarity: float  # ‖A_Sᵀ r + λ sign(x_S)‖_∞
    rank_ok: bool  # A_S has full column rank


def lasso_objective(A: Array, b: Array, lam: float, x: Array) -> float:
    r = A @ x - b
    return 0.5 * float(r @ r) + lam * float(np.abs(x).sum())


def _split_lbfgsb(A: Array, b: Array, lam: float) -> Array:
    n = A.shape[1]

    def fun(z: Array) -> tuple[float, Array]:
        x = z[:n] - z[n:]
        r = A @ x - b
        g = A.T @ r
        return 0.5 * float(r @ r) + lam * float(z.sum()), np.concatenate([g + lam, -g + lam])

    res = scipy.optimize.minimize(
        fun,
        np.zeros(2 * n),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (2 * n),
        options={"maxiter": 50_000, "ftol": 1e-16, "gtol": 1e-13, "maxcor": 30},
    )
    return res.x[:n] - res.x[n:]


def _polish(A: Array, b: Array, lam: float, S: NDArray[np.intp], s: Array) -> tuple[Array, bool]:
    """Solve A_Sᵀ(A_S x_S − b) + λ s = 0 for x_S.

    With the thin QR factorization A_S = QR the equation is Rᵀ(R x_S − Qᵀb) = −λ s, so
    R x_S = Qᵀb − λ w with Rᵀw = s: two triangular solves, and A_SᵀA_S is never formed
    (Golub & Van Loan 2013, §5.3: forming it squares the condition number).
    """
    n = A.shape[1]
    x = np.zeros(n)
    if S.size == 0:
        return x, True
    A_S = A[:, S]  # (m, |S|)
    Q, R = cast(tuple[Array, Array], scipy.linalg.qr(A_S, mode="economic"))  # A_S = Q R
    rank_ok = bool(np.min(np.abs(np.diag(R))) > 1e-12 * np.max(np.abs(np.diag(R))))
    w = scipy.linalg.solve_triangular(R, s, trans=1)  # Rᵀ w = s
    x_S = scipy.linalg.solve_triangular(R, Q.T @ b - lam * w)  # R x_S = Qᵀb − λ w
    x[S] = x_S
    return x, rank_ok


def lasso_solution(A: Array, b: Array, lam: float) -> LassoCertificate:
    """Certified lasso minimizer (see the module docstring). Raises if the KKT check fails."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_tilde = _split_lbfgsb(A, b, lam)
    scale = max(float(np.max(np.abs(x_tilde))), 1e-300)
    last_err = ""
    for rel in (1e-6, 1e-8, 1e-4, 1e-10, 1e-3):
        S = np.flatnonzero(np.abs(x_tilde) > rel * scale)
        s = np.sign(x_tilde[S])
        x, rank_ok = _polish(A, b, lam, S, s)
        r = A @ x - b
        c = A.T @ r  # (n,)
        out = np.setdiff1d(np.arange(A.shape[1]), S)
        margin = lam - (float(np.max(np.abs(c[out]))) if out.size else 0.0)
        signs_ok = bool(np.all(np.sign(x[S]) == s))
        stationarity = float(np.max(np.abs(c[S] + lam * s))) if S.size else 0.0
        if signs_ok and margin > 0.0 and rank_ok:
            return LassoCertificate(
                x=x,
                F=lasso_objective(A, b, lam, x),
                support=tuple(int(i) for i in S),
                margin=margin,
                min_abs=float(np.min(np.abs(x[S]))) if S.size else np.inf,
                stationarity=stationarity,
                rank_ok=rank_ok,
            )
        last_err = f"signs_ok={signs_ok}, margin={margin:.3e}, rank_ok={rank_ok}"
    raise RuntimeError(f"lasso KKT certificate failed: {last_err}")
