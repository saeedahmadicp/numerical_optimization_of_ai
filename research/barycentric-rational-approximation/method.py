"""Barycentric rational approximation: AAA (greedy support points) and Floater–Hormann.

Both methods return a rational function in the barycentric form of numopt's
``barycentric`` method (Berrut & Trefethen 2004, eq. (4.2); Nakatsukasa, Sète & Trefethen
2018, eq. (2.1)),

    r(t) = Σ_j [w_j/(t - z_j)] f_j / Σ_j [w_j/(t - z_j)],      r(z_j) = f_j when w_j ≠ 0,

and evaluate it with numopt's ``_barycentric_eval``. Only the weights w differ from the
polynomial case:

* ``aaa``: the support points z_j are chosen greedily from a sample set Z, and w is the
  right singular vector of the smallest singular value of a Loewner matrix (NST 2018,
  Fig. 4.1). r is a type (m-1, m-1) rational approximant, not an interpolant of all data.
* ``floater_hormann``: the support points are all data nodes (sorted), and w is the
  explicit Floater–Hormann blend weight (FH 2007, eq. (18)). r interpolates every node,
  has no real poles (FH 2007, Theorem 1) and converges at O(h^{d+1}) (Theorem 2).

Both accept a :class:`numopt.core.types.Dataset` or an ``(x, y)`` pair, like every numopt
interpolation method; they take no ``x0`` or ``seed`` (they are deterministic and direct).
``Result.fun`` is max_t |r(t) - f_true(t)| on numopt's 200-point plotting grid when the
dataset has ``f_true``, else ``None``. ``n_fev = 0``: the methods use only the data.

Result.extra (common keys, same meaning as in ``numopt.interpolation.methods``)::

    kind:          "aaa" | "floater_hormann"
    coefficients:  [m] the barycentric weights w
    nodes:         [m] the support points z_j (FH: the sorted data nodes)
    values:        [m] the data values f_j at the support points
    domain, eval, max_error, node_residual

AAA adds ``sample_error`` (max over Z of |F - r|, the quantity AAA drives below
tol·max|F|), ``errors`` ([n_iter + 1] sample error after each step, step 0 = the
constant mean(F)), ``sigma_min`` ([n_iter] smallest singular value of each (scaled) Loewner
matrix; non-increasing for ``scaling="none"`` by NST Proposition 3.1), ``scaling``, ``poles``/``residues`` ([[re, im]]) of the final r,
``n_doublets``, ``n_interval_poles`` and ``interval_poles`` (see below). FH adds ``d``.

Info keys:
    node_index: int        AAA: index in Z of the support point added at this step (k ≥ 1);
                           FH: index k (in sorted order) of the node whose weight w_k this
                           step computes
    node: [x, y]           that sample / node
    support: [m]           AAA: the support points z_1..z_m after this step
    support_values: [m]    AAA: f_1..f_m
    weights: [m]           AAA: w after this step (‖w‖₂ = 1, largest |w_j| positive)
    degree: int            AAA: m - 1, the type (m-1, m-1) of r after this step
    sample_error: float    AAA: max_{z ∈ Z} |f(z) - r(z)| after this step
    sigma_min: float       AAA: smallest singular value of the Loewner matrix A^(m)
                           (of A^(m)·D with column scaling; 0 when A^(m) has fewer
                           rows than columns)
    poles: [[re, im]]      AAA: the m-1 finite eigenvalues of the arrowhead pencil (3.11)
    residues: [[re, im]]   AAA: the residue n(λ)/d'(λ) at each pole
    n_doublets: int        AAA: poles with |residue| < DOUBLET_RTOL·max|F| (numerical
                           Froissart doublets, NST 2018 §5)
    n_interval_poles: int  AAA: certified real zeros of the denominator d in [a, b], each
                           proved by a sign change of d (see real_denominator_roots)
    interval_poles: [k]    AAA: their positions (bisected to the rounding limit of d)
    curve: [200]           AAA: r on ``extra.eval.x`` after this step (step 0: mean F);
                           FH: the final interpolant (last step only)
    weight: float          FH: the weight w_k computed at this step
    window: [i_lo, i_hi]   FH: the index range J_k = {i : k - d ≤ i ≤ k, 0 ≤ i ≤ n - d}
                           of the local interpolants p_i that contain node k
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numopt.core.registry import ParamSpec
from numopt.core.types import Dataset, Result, Step, Vector
from numopt.interpolation.methods import (
    _barycentric_eval,
    _broken,
    _finish,
    _Grid,
    _quiet,
    _require_distinct,
    _resolve,
    _sorted,
)

#: NST 2018, Fig. 5.1 flags a pole as a numerical Froissart doublet when |residue| < 1e-13.
DOUBLET_RTOL = 1e-13
_EPS = float(np.finfo(np.float64).eps)
#: Fractions s of a gap [lo, hi] at which real_denominator_roots samples d: lo + s(hi - lo)
#: and hi - s(hi - lo), graded geometrically towards both ends (2⁻¹..2⁻⁶⁰) plus 31
#: equispaced interior points.
_GAP_FRACTIONS = np.unique(
    np.concatenate([2.0 ** -np.arange(1.0, 61.0), np.linspace(0.0, 1.0, 33)[1:-1]])
)
#: Eigenvalue estimates λ with |Im λ| ≤ _HINT_RTOL·|λ| + _HINT_ATOL·max|z| are "near real":
#: their real parts are added to the sampling grid of real_denominator_roots (so two real
#: zeros in one gap are separated). Extra grid points cannot create a false count.
_HINT_RTOL = 1e-4
_HINT_ATOL = 1e3 * _EPS

PARAMS: dict[str, tuple[ParamSpec, ...]] = {
    "aaa": (
        ParamSpec(
            "tol",
            1e-13,
            min=1e-15,
            max=1e-2,
            log=True,
            help="Stop when max_{z∈Z} |f(z) - r(z)| ≤ tol·max|f| (NST 2018 default 1e-13).",
        ),
        ParamSpec(
            "max_terms",
            100,
            kind="int",
            min=1,
            max=200,
            help="Maximum number m of support points; r has type (m-1, m-1) (NST mmax = 100).",
        ),
        ParamSpec(
            "scaling",
            "columns",
            kind="choice",
            choices=("columns", "none"),
            help="'columns': SVD of the Loewner matrix with unit-norm columns (stable when "
            "column norms differ by many orders); 'none': NST 2018 Fig. 4.1 exactly.",
        ),
    ),
    "floater_hormann": (
        ParamSpec(
            "d",
            3,
            kind="int",
            min=0,
            max=30,
            help="Degree of the blended local interpolants (0 ≤ d ≤ n); error O(h^{d+1}).",
        ),
    ),
}

#: Registration metadata for a later promotion into numopt.interpolation (not registered here).
SPECS: dict[str, dict[str, Any]] = {
    "aaa": {
        "name": "AAA rational approximation",
        "needs": ("data",),
        "order": "root-exponential for |x|-type singularities, geometric for analytic f",
        "summary": "Greedily add the worst-fit sample as a support point; choose barycentric "
        "weights by the smallest singular vector of a Loewner matrix.",
        "references": (
            "Nakatsukasa, Sète & Trefethen (2018), The AAA algorithm for rational "
            "approximation, SIAM J. Sci. Comput. 40(3), A1494–A1522: Fig. 4.1, eqs. (3.5)–(3.11)",
        ),
    },
    "floater_hormann": {
        "name": "Floater–Hormann rational interpolation",
        "needs": ("data",),
        "order": "O(h^{d+1}), no real poles",
        "summary": "Blend the local degree-d interpolants; the result is a pole-free barycentric "
        "rational interpolant with explicit weights.",
        "references": (
            "Floater & Hormann (2007), Barycentric rational interpolation with no poles and "
            "high rates of approximation, Numer. Math. 107, 315–331: eqs. (4), (5), (18), "
            "Theorems 1–2",
        ),
    },
}


# --------------------------------------------------------------------------------------
# Poles and residues of a barycentric rational function
# --------------------------------------------------------------------------------------


#: Newton steps that polish each eigenvalue estimate of a pole (see barycentric_poles).
POLE_NEWTON_STEPS = 3


def _polish_poles(poles: np.ndarray, z: Vector, w: Vector) -> np.ndarray:
    """Safeguarded Newton steps λ ← λ - d(λ)/d'(λ) on d(λ) = Σ_j w_j/(λ - z_j).

    A step is taken only if it lowers the relative residual |d(λ)| / Σ_j |w_j/(λ - z_j)|
    and is shorter than half the distance from λ to the nearest other pole estimate (so two
    estimates cannot merge into one root).
    """
    p = poles.copy()
    if p.size == 0:
        return p

    def rel_residual(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = w[None, :] / (lam[:, None] - z[None, :])  # (p, m)
        dval = np.sum(c, axis=1)
        dprime = -np.sum(c / (lam[:, None] - z[None, :]), axis=1)
        return np.abs(dval) / np.sum(np.abs(c), axis=1), dval, dprime

    for _ in range(POLE_NEWTON_STEPS):
        res, dval, dprime = rel_residual(p)
        step = dval / dprime
        gap = np.abs(p[:, None] - p[None, :])
        np.fill_diagonal(gap, np.inf)
        trial = p - step
        res_trial, _, _ = rel_residual(trial)
        ok = (
            np.isfinite(trial)
            & (res_trial < res)
            & (np.abs(step) < 0.5 * np.min(gap, axis=1, initial=np.inf))
        )
        if not ok.any():
            break
        p = np.where(ok, trial, p)
    return p


def barycentric_poles(z: Vector, w: Vector, f: Vector) -> tuple[np.ndarray, np.ndarray]:
    """Poles and residues of r = n/d, n = Σ w_j f_j/(t - z_j), d = Σ w_j/(t - z_j).

    The poles are the finite eigenvalues λ of the (m+1)×(m+1) arrowhead pencil
    (NST 2018, eq. (3.11))::

        E = [[0, wᵀ], [1, diag(z)]],   B = diag(0, 1, ..., 1),   E v = λ B v.

    At least two eigenvalues are infinite; the other m - 1 are the zeros of d.

    # NOTE: numopt has no runtime SciPy (no QZ), so the pencil is solved by one exact
    # shift-and-invert step: with σ not a pole, M = (E - σB)⁻¹B has eigenvalues
    # μ = 1/(λ - σ) (μ = 0 for λ = ∞). B e₀ = 0 makes the first column of M zero, and the
    # block inverse of the arrowhead E - σB gives the trailing m×m block in closed form,
    #     M₂₂ = D⁻¹ - (D⁻¹1)(D⁻¹w)ᵀ/(wᵀD⁻¹1),   D = diag(z_j - σ),
    # whose eigenvector 1 carries the second μ = 0. So λ = σ + 1/μ over the other m - 1
    # eigenvalues of M₂₂. σ is the candidate on a circle around the support points with the
    # smallest ‖M₂₂‖_F. M₂₂ is far from normal: in development the raw estimates for the
    # far poles of tanh(50x) were ~1e3 times less accurate than QZ (7e-6 relative).
    # POLE_NEWTON_STEPS safeguarded Newton steps on the barycentric d(λ) (stable to
    # evaluate) fix this: against long-double Newton roots the polished poles have
    # relative error max 6.3e-10 / median 2.1e-11, QZ max 2.8e-9 / median 3.1e-10
    # (run.py diagnostics, results/tables.md). Tests
    # compare with scipy.linalg.eigvals(E, B) and with the quad-precision roots.
    # NOTE: absolute resolution. λ = σ + 1/μ with |σ| ≈ 1, so λ is resolved only to an
    # absolute error of about eps·|σ| ≈ 1e-16 (plus the eigenvalue error of μ); Newton
    # cannot repair estimates that collapse onto the same rounded value. Poles closer than
    # about 1e-15 to the origin (AAA puts many there for |x| and √x) are therefore noise in
    # position, and a relative "is it real" test on them fails. QZ is no better (it reports
    # "real" poles at 1e-20..1e-14 with relative residual 1e-2 on the √x run). Real poles
    # inside [a, b] are therefore counted by real_denominator_roots, not from this list.

    Residues: res = n(λ)/d'(λ), d'(λ) = -Σ w_j/(λ - z_j)² (simple poles). Returns complex
    arrays (poles, residues) of length m - 1 (fewer if a μ is exactly 0).
    """
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    nz = w != 0.0  # a zero weight removes its support point from r
    z, w, f = z[nz], w[nz], f[nz]
    m = z.size
    if m <= 1:
        return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.complex128)
    center = 0.5 * (float(np.min(z)) + float(np.max(z)))
    radius = float(np.max(np.abs(z - center)))
    if radius == 0.0:
        radius = 1.0
    best: tuple[float, complex, np.ndarray] | None = None
    for theta in (0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6):  # angles in units of π (upper half)
        for scale in (1.0, 2.0):
            sigma = complex(center + scale * radius * np.exp(1j * np.pi * theta))
            inv_d = 1.0 / (z - sigma)  # (m,) D⁻¹ 1
            s = np.sum(w * inv_d)  # wᵀ D⁻¹ 1 = -d(σ)
            if s == 0.0 or not np.isfinite(s):
                continue
            m22 = np.diag(inv_d) - np.outer(inv_d, w * inv_d) / s  # (m, m)
            norm = float(np.linalg.norm(m22))
            if np.isfinite(norm) and (best is None or norm < best[0]):
                best = (norm, sigma, m22)
    if best is None:
        nan = np.full(m - 1, complex(math.nan, math.nan))
        return nan, nan
    _, sigma, m22 = best
    mu = np.linalg.eigvals(m22)  # (m,)
    keep = np.argsort(np.abs(mu))[1:]  # drop the structural μ = 0 (eigenvector 1)
    mu = mu[keep]
    mu = mu[mu != 0.0]
    poles = _polish_poles(sigma + 1.0 / mu, z, w)
    diff = poles[:, None] - z[None, :]  # (p, m)
    numer = (1.0 / diff) @ (w * f)
    dprime = -(1.0 / diff**2) @ w
    return poles, numer / dprime


def _signed_d(t: np.ndarray, z: Vector, w: Vector) -> np.ndarray:
    """Sign of d(t) = Σ_j w_j/(t - z_j) where it is certain, else 0.

    The computed d has absolute error ≤ γ·Σ_j |w_j/(t - z_j)| with γ ≈ (2 + log₂ m)·eps
    (one rounding in t - z_j and in the division, pairwise summation; Higham 2002, §4.2).
    γ = 4(m + 2)·eps is used, so a returned ±1 is the sign of the exact d at t.
    """
    c = w[None, :] / (t[:, None] - z[None, :])  # (T, m)
    dval = np.sum(c, axis=1)
    gamma = 4.0 * (z.size + 2) * _EPS
    sure = np.abs(dval) > gamma * np.sum(np.abs(c), axis=1)
    return np.where(sure, np.sign(dval), 0.0)


#: Bisection steps per bracket in real_denominator_roots (2⁻²⁰⁰⁰ < any float spacing).
_BISECT_MAX = 2000


def real_denominator_roots(
    z: Vector, w: Vector, a: float, b: float, hints: Vector | None = None
) -> np.ndarray:
    """Certified real zeros of d(t) = Σ_j w_j/(t - z_j) in [a, b], as brackets [lo, hi].

    With real support points z and real weights w, d is real and continuous on every gap
    between consecutive support points. Next to a support point the term w_k/(t - z_k)
    dominates: d → sign(w_k)·∞ for t ↓ z_k and d → -sign(w_k)·∞ for t ↑ z_k. So a gap
    (z_k, z_{k+1}) holds an odd number of zeros exactly when w_k and w_{k+1} have the same
    sign (Schneider & Werner 1986; Berrut 1988). This function samples d on each gap of
    [a, b] (graded towards the ends, plus the real parts ``hints`` of near-real pole
    estimates), keeps only the samples whose sign is certain (see ``_signed_d``), adds the
    exact limit signs at the support points, and returns one bracket per sign change,
    shrunk by bisection to the band where the sign of d is no longer certain. Each bracket contains at least
    one zero of d; the count is a lower bound (two zeros closer than the sample spacing,
    and not separated by a hint, give no sign change) with the Schneider–Werner parity in
    every gap. A zero of d is a pole of r unless n vanishes there too.

    # NOTE: this replaces a relative test |Im λ| ≤ √eps·|λ| on the eigenvalue estimates,
    # which missed real poles at |λ| ≈ 4e-17 (their estimates come out as ±1e-16 ± 1e-16i,
    # see barycentric_poles). Bisection works in absolute float64 arithmetic near the zero,
    # so it resolves such a pole to the rounding limit of d.

    Returns an array of shape (k, 2), sorted by lo.
    """
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    nz = w != 0.0
    z, w = z[nz], w[nz]
    order = np.argsort(z)
    z, w = z[order], w[order]
    h = np.empty(0) if hints is None else np.asarray(hints, dtype=np.float64)
    sw = np.sign(w)
    inner = (z > a) & (z < b)
    ends = np.concatenate([[a], z[inner], [b]])  # segment end points
    # limit sign of d just right of / just left of each end point (0: evaluate d there)
    right_sign = np.concatenate([[0.0], sw[inner], [0.0]])
    left_sign = np.concatenate([[0.0], -sw[inner], [0.0]])
    for i, e in ((0, a), (ends.size - 1, b)):
        k = np.flatnonzero(z == e)
        if k.size:  # a or b is a support point
            right_sign[i], left_sign[i] = sw[k[0]], -sw[k[0]]
        else:
            right_sign[i] = left_sign[i] = _signed_d(np.array([e]), z, w)[0]
    brackets: list[tuple[float, float, float]] = []
    for s in range(ends.size - 1):
        lo, hi = float(ends[s]), float(ends[s + 1])
        if not hi > lo:
            continue
        hs = np.sort(h[(h > lo) & (h < hi)])
        t = np.concatenate([lo + _GAP_FRACTIONS * (hi - lo), hi - _GAP_FRACTIONS * (hi - lo), hs])
        if hs.size > 1:
            t = np.concatenate([t, 0.5 * (hs[1:] + hs[:-1])])
        t = np.unique(t)
        t = t[(t > lo) & (t < hi)]
        sg = _signed_d(t, z, w)
        xs = np.concatenate([[lo], t[sg != 0.0], [hi]])
        ss = np.concatenate([[right_sign[s]], sg[sg != 0.0], [left_sign[s + 1]]])
        keep = ss != 0.0
        xs, ss = xs[keep], ss[keep]
        for i in np.flatnonzero(ss[1:] != ss[:-1]):
            brackets.append((float(xs[i]), float(xs[i + 1]), float(ss[i])))
    out = np.empty((len(brackets), 2))
    for i, (lo, hi, s_lo) in enumerate(brackets):
        out[i] = _bisect_certified(lo, hi, s_lo, z, w)
    return out


def _bisect_certified(
    lo: float, hi: float, s_lo: float, z: Vector, w: Vector
) -> tuple[float, float]:
    """Shrink a bracket (sign s_lo at lo, -s_lo at hi) while every probe stays certified.

    A probe whose sign is uncertain lies in the band where |d| is below its rounding error;
    the bracket is then shrunk from both sides towards that band (probes between lo and the
    leftmost uncertain point, and between the rightmost uncertain point and hi), so the
    result is the narrowest bracket whose end signs are certain."""
    u_lo: float | None = None  # leftmost / rightmost uncertain probe inside (lo, hi)
    u_hi: float | None = None
    for _ in range(_BISECT_MAX):
        if u_lo is None or u_hi is None:
            probes = [lo + 0.5 * (hi - lo)]
            limits = [(lo, hi)]
        else:
            probes = [lo + 0.5 * (u_lo - lo), u_hi + 0.5 * (hi - u_hi)]
            limits = [(lo, u_lo), (u_hi, hi)]
        moved = False
        for p, (a_, b_) in zip(probes, limits, strict=True):
            if not (a_ < p < b_ and lo < p < hi):
                continue
            moved = True
            sp = _signed_d(np.array([p]), z, w)[0]
            if sp == s_lo:
                lo = p
            elif sp == -s_lo:
                hi = p
            else:
                u_lo = p if u_lo is None else min(u_lo, p)
                u_hi = p if u_hi is None else max(u_hi, p)
            if u_lo is not None and u_hi is not None and not (lo < u_lo and u_hi < hi):
                u_lo = u_hi = None  # the band left the bracket: plain bisection again
        if not moved:
            break
    return lo, hi


def _pole_summary(
    z: Vector, w: Vector, f: Vector, a: float, b: float, f_scale: float
) -> dict[str, Any]:
    with np.errstate(all="ignore"):
        poles, res = barycentric_poles(z, w, f)
        scale = float(np.max(np.abs(z))) if np.size(z) else 1.0
        near_real = np.abs(np.imag(poles)) <= _HINT_RTOL * np.abs(poles) + _HINT_ATOL * scale
        brackets = real_denominator_roots(z, w, a, b, hints=np.real(poles)[near_real])
    doublets = np.abs(res) < DOUBLET_RTOL * max(f_scale, float(np.finfo(np.float64).tiny))
    return {
        "poles": [[float(p.real), float(p.imag)] for p in poles],
        "residues": [[float(r.real), float(r.imag)] for r in res],
        "n_doublets": int(np.sum(doublets)),
        "n_interval_poles": int(brackets.shape[0]),
        "interval_poles": [float(0.5 * (lo + hi)) for lo, hi in brackets],
    }


# --------------------------------------------------------------------------------------
# AAA
# --------------------------------------------------------------------------------------


@_quiet
def aaa(
    problem: Dataset | tuple[Any, Any],
    *,
    tol: float = 1e-13,
    max_terms: int = 100,
    scaling: str = "columns",
) -> Result:
    """AAA rational approximation (Nakatsukasa, Sète & Trefethen 2018, Fig. 4.1).

    Data: sample points Z = x (distinct, real) and values F = y, M = |Z|. Start with the
    constant R = mean(F). Step m = 1, 2, ...:

    1. pick the sample with the largest error, j = argmax_{i ∈ J} |F_i - R_i|, where J
       indexes the samples that are not yet support points; z_m = Z_j, f_m = F_j;
    2. form the (M-m)×m Loewner matrix A^(m)_{ij} = (F_i - f_j)/(Z_i - z_j), i ∈ J
       (eq. (3.6), built as S_F C - C S_f, eq. (3.9));
    3. w = the right singular vector of the smallest singular value of A^(m), which solves
       min ‖A^(m) w‖ subject to ‖w‖ = 1 (eq. (3.5));
    4. R_i = N_i/D_i on J with N = C(w∘f), D = Cw (eq. (3.10)), R = F at support points.

    Stopping test (Fig. 4.1): max_i |F_i - R_i| ≤ tol·max_i |F_i|, checked after each
    step → ``converged=True``. Reaching ``max_terms`` support points, or a non-finite
    weight/value, gives ``converged=False``. If every sample becomes a support point,
    R = F exactly and the test passes.

    # NOTE: the argmax runs over the non-support samples J (as SciPy does); Fig. 4.1 takes
    # it over all of Z, which is the same whenever some error is nonzero.
    # NOTE: w is normalized so that its largest-|w_j| entry is positive. The SVD sign is
    # arbitrary and the sign cancels in r; the fixed sign makes the trace reproducible.
    # NOTE: scaling="columns" (default) is not in Fig. 4.1. It takes w = D v / ‖D v‖ with
    # D = diag(1/‖a_j‖) (a_j the columns of A^(m)) and v the minimal right singular vector
    # of A^(m) D, i.e. it solves (3.5) with the constraint ‖D⁻¹w‖ = 1 instead of ‖w‖ = 1.
    # Reason (measured): for sqrt(x) on 4000 equispaced ∪ 2000 log-spaced points in
    # [1e-30, 1] the column norms of A^(m) differ by a factor 9.2e13, and at m = 24
    # κ(A^(m)) = 4.7e21 while κ(A^(m) D) = 5.2e8. The SVD resolves w only to an absolute
    # error ≈ eps·‖A‖, which swamps the entries of w on the small columns; the unscaled
    # iteration stalls at type (24, 24) with test error 2.9e-8 and then fills [0, 1] with
    # real poles, while the scaled one reaches 8.4e-14 at type (64, 64) (run.py, Q1). Column equilibration is within a factor √m of the best diagonal
    # scaling for κ₂ (van der Sluis 1969). SciPy's AAA switches to the same scaling once
    # κ(A^(m)) > 1/(3 eps); "none" reproduces Fig. 4.1 (and SciPy before the switch).
    # NOTE: when A^(m) has fewer rows than columns (M - m < m), the last row of the full
    # V* is a null vector of A^(m) (σ_min reported as 0); Fig. 4.1's svd(A, 0) does the same.
    # NOTE: the poles in ``Step.info`` are computed every step (Fig. 4.1 computes them once,
    # at the end) so the visualizer can show Froissart doublets as they appear.

    The trace has a step k = 0 (the constant mean(F)) and one step per support point, so
    ``n_iter`` = m. ``Result.x`` = the final weights w.
    """
    if not tol >= 0.0:
        raise ValueError("tol must be ≥ 0")
    if max_terms < 1:
        raise ValueError("max_terms must be ≥ 1")
    if scaling not in ("none", "columns"):
        raise ValueError("scaling must be 'none' or 'columns'")
    data = _resolve(problem)
    _require_distinct(data.x)
    grid = _Grid(data)
    z_all, f_all = data.x, data.y  # (M,), (M,)
    n_samples = z_all.size
    f_scale = float(np.max(np.abs(f_all)))
    atol = tol * f_scale

    free = np.ones(n_samples, dtype=bool)  # J: samples that are not support points
    support_idx: list[int] = []
    cauchy_cols: list[Vector] = []  # columns 1/(Z - z_j), each (M,)
    r_samples = np.full(n_samples, float(np.mean(f_all)))  # R, (M,)
    err = float(np.max(np.abs(f_all - r_samples)))
    errors = [err]
    sigmas: list[float] = []
    const_curve = np.full(grid.t.size, float(np.mean(f_all)))
    trace: list[Step] = [
        Step(
            0,
            np.empty(0),
            grid.error(const_curve),
            info={
                "support": [],
                "support_values": [],
                "weights": [],
                "degree": 0,
                "sample_error": err,
                "curve": const_curve,
                "poles": [],
                "residues": [],
                "n_doublets": 0,
                "n_interval_poles": 0,
                "interval_poles": [],
            },
        )
    ]
    w = np.empty(0)
    converged = False
    message = ""
    for m in range(1, max_terms + 1):
        cand = np.flatnonzero(free)
        j = int(cand[np.argmax(np.abs(f_all[cand] - r_samples[cand]))])
        support_idx.append(j)
        free[j] = False
        cauchy_cols.append(1.0 / (z_all - z_all[j]))  # inf at row j only; row j leaves J
        zs = z_all[support_idx]  # (m,)
        fs = f_all[support_idx]  # (m,)
        rows = np.flatnonzero(free)
        cmat = np.stack(cauchy_cols, axis=1)[rows]  # C, (M-m, m)
        loewner = f_all[rows, None] * cmat - cmat * fs[None, :]  # S_F C - C S_f, (M-m, m)
        if scaling == "columns":
            col_norm = np.linalg.norm(loewner, axis=0)  # (m,)
            col_norm[~(col_norm > 0.0)] = 1.0
        else:
            col_norm = np.ones(m)
        try:
            if rows.size >= m:
                _, sv, vh = np.linalg.svd(loewner / col_norm, full_matrices=False)
                sigma_min = float(sv[-1])
            else:
                _, sv, vh = np.linalg.svd(loewner / col_norm, full_matrices=True)
                sigma_min = 0.0
        except np.linalg.LinAlgError:
            return _broken("aaa", trace, f"SVD of the Loewner matrix failed at step {m}")
        w = vh[-1] / col_norm  # (m,) real data → real w
        w = w / np.linalg.norm(w)
        w = w * (1.0 if w[int(np.argmax(np.abs(w)))] >= 0.0 else -1.0)
        sigmas.append(sigma_min)
        numer = cmat @ (w * fs)  # N on J
        denom = cmat @ w  # D on J
        r_samples = f_all.copy()
        r_samples[rows] = numer / denom
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(r_samples))):
            return _broken(
                "aaa",
                trace,
                f"non-finite weights or values at step {m} (d(z) = 0 at a sample point)",
            )
        err = float(np.max(np.abs(f_all - r_samples)))
        errors.append(err)
        curve = _barycentric_eval(zs, w, fs, grid.t)
        info: dict[str, Any] = {
            "node_index": j,
            "node": [z_all[j], f_all[j]],
            "support": zs.copy(),
            "support_values": fs.copy(),
            "weights": w.copy(),
            "degree": m - 1,
            "sample_error": err,
            "sigma_min": sigma_min,
            "curve": curve,
        }
        info.update(_pole_summary(zs, w, fs, data.a, data.b, f_scale))
        trace.append(Step(m, w.copy(), grid.error(curve), info=info))
        if err <= atol:
            converged = True
            message = (
                f"max sample error {err:.3g} ≤ tol·max|f| = {atol:.3g} with {m} support "
                f"points (type ({m - 1}, {m - 1}))"
            )
            break
        if not free.any():
            break
    if not converged:
        message = (
            f"max_terms = {max_terms} reached: max sample error {err:.3g} > tol·max|f| = {atol:.3g}"
        )

    zs = z_all[support_idx]
    fs = f_all[support_idx]
    final = trace[-1].info
    curve = np.asarray(final["curve"])
    max_err = grid.error(curve)
    node_res = float(np.max(np.abs(_barycentric_eval(zs, w, fs, zs) - fs)))
    extra: dict[str, Any] = {
        "kind": "aaa",
        "scaling": scaling,
        "coefficients": w.copy(),
        "nodes": zs.copy(),
        "values": fs.copy(),
        "domain": [data.a, data.b],
        "eval": {"x": grid.t, "y": curve, "f_true": grid.truth},
        "max_error": max_err,
        "node_residual": node_res,
        "sample_error": err,
        "errors": errors,
        "sigma_min": sigmas,
        "poles": final["poles"],
        "residues": final["residues"],
        "n_doublets": final["n_doublets"],
        "n_interval_poles": final["n_interval_poles"],
        "interval_poles": final["interval_poles"],
    }
    return Result(
        "aaa",
        w.copy(),
        max_err,
        converged,
        message,
        trace[-1].k,
        0,
        trace=trace,
        extra=extra,
    )


def aaa_eval(res: Result, t: Any) -> Vector:
    """Evaluate the final AAA approximant of ``res`` at ``t`` (numopt barycentric formula)."""
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    with np.errstate(all="ignore"):
        return _barycentric_eval(
            np.asarray(res.extra["nodes"]),
            np.asarray(res.extra["coefficients"]),
            np.asarray(res.extra["values"]),
            tt,
        )


# --------------------------------------------------------------------------------------
# Floater–Hormann
# --------------------------------------------------------------------------------------


def floater_hormann_weights(x: Vector, d: int) -> tuple[Vector, list[tuple[int, int]]]:
    """Floater–Hormann weights for sorted distinct nodes x_0 < ... < x_n (FH 2007, eq. (18)).

    w_k = (-1)^{k-d} Σ_{i ∈ J_k} Π_{j=i, j≠k}^{i+d} 1/|x_k - x_j|,
    J_k = {i ∈ {0, ..., n-d} : k - d ≤ i ≤ k}  (eq. (11)).

    # NOTE: each factor is computed as h/|x_k - x_j| with h = (x_n - x_0)/n, i.e. all
    # weights are multiplied by the common factor h^d > 0. A common factor cancels in the
    # barycentric formula (FH 2007, §4: "a uniform scaling of these weights does not change
    # the interpolant"); it keeps the weights O(1) instead of O(h^{-d}), which overflows
    # for small h and large d. For equispaced nodes w_k·d! is FH's integer weight
    # (-1)^{k-d} Σ_{i∈J_k} C(d, k-i).

    Returns (w, windows) where windows[k] = (min J_k, max J_k).
    """
    n = x.size - 1
    if not 0 <= d <= n:
        raise ValueError(f"need 0 ≤ d ≤ n = {n}; got d = {d}")
    h = (x[-1] - x[0]) / n if n > 0 else 1.0
    w = np.zeros(n + 1, dtype=np.float64)
    windows: list[tuple[int, int]] = []
    for k in range(n + 1):
        i_lo, i_hi = max(0, k - d), min(k, n - d)
        total = 0.0
        for i in range(i_lo, i_hi + 1):
            prod = 1.0
            for j in range(i, i + d + 1):
                if j != k:
                    prod *= h / abs(x[k] - x[j])
            total += prod
        w[k] = (-1.0) ** (k - d) * total
        windows.append((i_lo, i_hi))
    return w, windows


@_quiet
def floater_hormann(problem: Dataset | tuple[Any, Any], *, d: int = 3) -> Result:
    """Floater–Hormann barycentric rational interpolation (Floater & Hormann 2007).

    With sorted nodes x_0 < ... < x_n and p_i the polynomial of degree ≤ d through
    x_i, ..., x_{i+d}, the interpolant is the blend (eqs. (4)–(5))

        r(x) = Σ_{i=0}^{n-d} λ_i(x) p_i(x) / Σ_{i=0}^{n-d} λ_i(x),
        λ_i(x) = (-1)^i / ((x - x_i)···(x - x_{i+d})),

    which equals the barycentric form (1) with the weights (18) (see
    :func:`floater_hormann_weights`). r interpolates every node, has no poles in ℝ for every
    0 ≤ d ≤ n (Theorem 1), reproduces polynomials of degree ≤ d, and for f ∈ C^{d+2}
    satisfies ‖f - r‖ = O(h^{d+1}) (Theorem 2). d = n gives the interpolating polynomial;
    d = 0 gives Berrut's interpolant w_k = (-1)^k.

    Stopping test: as numopt's interpolation family, ``converged=True`` when every weight
    and grid value is finite and max_k |r(x_k) - y_k| ≤ 1e-6·max|y|. Invalid d (d < 0 or
    d > n) and repeated nodes raise ``ValueError``.

    The trace has one step per weight: step k computes w_k (``Step.x`` = w_0..w_k); the
    last step also carries the interpolant ``curve`` and its grid error. ``Result.x`` = w.
    """
    data = _resolve(problem)
    xs, ys = _sorted(data, 1)
    if not isinstance(d, int | np.integer) or isinstance(d, bool):
        raise ValueError("d must be an integer")
    d = int(d)
    w, windows = floater_hormann_weights(xs, d)
    grid = _Grid(data)
    n = xs.size - 1
    trace: list[Step] = []
    for k in range(n + 1):
        info: dict[str, Any] = {
            "node_index": k,
            "node": [xs[k], ys[k]],
            "weight": w[k],
            "window": list(windows[k]),
        }
        fun = None
        if k == n:
            curve = _barycentric_eval(xs, w, ys, grid.t)
            info["curve"] = curve
            fun = grid.error(curve)
        trace.append(Step(k, w[: k + 1].copy(), fun, info=info))
    res = _finish(
        "floater_hormann",
        w.copy(),
        kind="floater_hormann",
        coefficients=w.copy(),
        nodes=xs,
        values=ys,
        grid=grid,
        evaluate=lambda t: _barycentric_eval(xs, w, ys, t),
        trace=trace,
        domain=(data.a, data.b),
        more={"d": d},
    )
    return res


def lebesgue_constant(x: Vector, w: Vector, points_per_gap: int = 50) -> float:
    """Λ = max_t Σ_k |w_k/(t - x_k)| / |Σ_k w_k/(t - x_k)| on a grid inside each gap.

    The Lebesgue function of a barycentric interpolant with weights w on sorted nodes x
    (Bos, De Marchi, Hormann & Klein 2012, eq. (2)); it equals 1 at the nodes, so only
    interior points of each gap [x_k, x_{k+1}] are sampled (``points_per_gap`` each).
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    frac = (np.arange(1, points_per_gap + 1) / (points_per_gap + 1))[None, :]  # (1, P)
    t = (x[:-1, None] + frac * np.diff(x)[:, None]).reshape(-1)  # (n·P,)
    best = 1.0
    for start in range(0, t.size, 4096):
        tc = t[start : start + 4096]
        c = w[None, :] / (tc[:, None] - x[None, :])  # (T, n+1)
        lam = np.sum(np.abs(c), axis=1) / np.abs(np.sum(c, axis=1))
        best = max(best, float(np.max(lam)))
    return best


__all__ = [
    "DOUBLET_RTOL",
    "PARAMS",
    "SPECS",
    "aaa",
    "aaa_eval",
    "barycentric_poles",
    "floater_hormann",
    "floater_hormann_weights",
    "lebesgue_constant",
    "real_denominator_roots",
]
