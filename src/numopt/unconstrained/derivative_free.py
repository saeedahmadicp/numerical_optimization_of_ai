"""Derivative-free local minimization of f: ℝⁿ → ℝ (only values of f are used).

Methods: Nelder–Mead simplex search, Powell's conjugate-direction method, Hooke–Jeeves
pattern search and compass (coordinate) search. They share these conventions:

* **Start.** ``x0`` (or the problem's default). If f(x0) is not finite the method returns at
  once with ``converged=False`` and a one-step trace.
* **dim = 1.** A :class:`Problem` with ``dim == 1`` (also the one built from a bare callable with
  a one-entry x0) follows the scalar convention of ``core.types``: f receives the float x[0]. A
  size-1 array returned by f is accepted as the value f(x). ``Result.x`` and the trace keep
  vectors of length 1.
* **Extreme barrier.** A trial value f(y) that is NaN or +∞ is treated as +∞ (Audet & Dennis
  2006, SIAM J. Optim. 17(1), §1), so such a point is never accepted and the iterate always has
  a finite value. A value −∞ stops the method with ``converged=False`` (f is unbounded below).
* **Trace.** One :class:`Step` for ``k = 0`` (the start) and one per iteration, with
  ``n_iter == trace[-1].k``. ``Step.x`` and ``Step.fun`` are the best point known after the
  iteration and its value. Each method documents what one iteration is.
* **Counts.** ``n_fev`` counts every evaluation of f exactly; ``n_gev = n_hev = 0``.
* **Tolerances.** Absolute tolerances get a rounding floor 2ε|·| (ε = machine epsilon), as in
  Brent (1973), ch. 4, so the tests stay reachable at any scale of x and f. Powell's f test is
  relative (NR §10.7) and therefore depends on the size of f; Powell adds an x test for that
  reason (see its docstring).

Info keys (all methods):
    Nothing is common to every method; the keys per method follow.

Info keys per method:
    nelder_mead (one iteration = one reflect/expand/contract/shrink operation):
        simplex: [[n] × (n+1)]  the vertices after the iteration, ordered best → worst.
        simplex_f: [n+1]        f at those vertices (same order).
        operation: "reflect" | "expand" | "contract_outside" | "contract_inside" | "shrink" |
                                null — the operation of this iteration (null at k = 0).
        centroid: [n] | null    x̄, the centroid of the n best vertices of the old simplex.
        worst: [n] | null       x_{n+1}, the worst vertex of the old simplex (the one the
                                reflection moves away from).
        trials: [{"op": str, "x": [n], "f": float}]  every non-shrink point evaluated in the
                                iteration, in order ("reflect", "expand", "contract_outside",
                                "contract_inside"); after a shrink the new vertices are in
                                ``simplex``.
        size: float             max_i ‖x_i − x_1‖∞, the simplex size of the stopping test.
        f_spread: float         f(x_{n+1}) − f(x_1), the f-spread of the stopping test.
        coefficients: {"rho", "chi", "gamma", "sigma"}  reflection, expansion, contraction
                                and shrink coefficients in use.
    powell (one iteration = one sweep of n line minimizations, plus the extrapolation test):
        directions: [[n] × n]   the direction set used in this sweep (before replacement).
        new_directions: [[n] × n]  the direction set after this iteration (for the next sweep).
        lines: [{"origin": [n], "direction": [n], "alpha": float, "point": [n], "f": float,
                 "trials": [[alpha, f]]}]  each line minimization in order: start, direction,
                                minimizing step α, minimizer origin + α·direction, its value
                                and every (α, f(origin + α·direction)) evaluated by the
                                bracketing and Brent search. n lines, or n + 1 when the new
                                direction was searched.
        extrapolated: [n] | null  x_E = 2x_n − x_0 (null when the sweep met the stopping test).
        f_extrapolated: float | null  f(x_E).
        largest_decrease: float  Δf, the largest decrease along one direction in the sweep.
        largest_index: int      the index of that direction (the one that may be discarded).
        replace_test: float | null  t = 2(f₀ − 2f_n + f_E)(f₀ − f_n − Δf)² − Δf(f₀ − f_E)²
                                (computed only when f_E < f₀; the direction is replaced iff t < 0).
        replaced: int | null    the index of the discarded direction, or null.
    hooke_jeeves (one iteration = one exploratory move, around the base point or around a
    pattern point):
        move: "explore" | "pattern" | null   where the exploratory move started (null at k = 0).
        outcome: "explore_success" | "step_reduced" | "pattern_success" | "pattern_failed" |
                                null (k = 0).
        base: [n]               the base point after the iteration (= Step.x).
        previous_base: [n] | null  the base point before the last successful move (the
                                pattern direction is base − previous_base).
        pattern_point: [n] | null  p = 2·base − previous_base, the start of a pattern move.
        probes: [{"x": [n], "f": float}]  every point evaluated in the iteration, in order.
        step: float             the step h used for the probes of this iteration.
        new_step: float         the step after the iteration (h, or shrink·h).
    compass_search (one iteration = one poll of the 2n compass directions):
        polls: [{"x": [n], "f": float}]  the poll points evaluated, in the order of
                                D⊕ = (e₁, …, e_n, −e₁, …, −e_n); polling stops at the first
                                point with a decrease.
        success: bool | null    True when a poll point decreased f (null at k = 0).
        direction: int | null   index into D⊕ of the successful direction.
        step: float             Δ_k, the step used for the polls.
        new_step: float         Δ_{k+1} (Δ_k on success, shrink·Δ_k otherwise).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.counting import Counted, start_point, vector_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step, Vector

#: Machine epsilon of float64.
EPS = sys.float_info.epsilon

VectorFn = Callable[[Any], float]


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


class _Objective:
    """f with exact counting and the extreme-barrier convention (NaN, +∞ → +∞).

    With ``scalar=True`` (a Problem with ``dim == 1``) f receives the float x[0], following the
    scalar convention of ``core.types``.
    """

    __slots__ = ("counted", "scalar")

    def __init__(self, fn: Callable[..., Any], scalar: bool = False) -> None:
        self.counted = Counted(fn)
        self.scalar = scalar

    @property
    def n(self) -> int:
        return self.counted.n

    def __call__(self, x: Vector) -> float:
        # Pass a copy: a user f that mutates its argument cannot corrupt the method's state.
        arg: Any = float(x[0]) if self.scalar else np.array(x, dtype=np.float64)
        v = _value(self.counted(arg))
        return math.inf if math.isnan(v) else v


def _value(v: Any) -> float:
    """f(x) as a float; a size-1 array is accepted, a larger one is invalid input."""
    # NOTE: float() of a shape-(1,) array is an error from NumPy 2.5 on; a vectorized 1-D
    # library f returns that shape, so the single entry is taken explicitly.
    arr = np.asarray(v, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"f(x) must return a scalar, got an array of shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _setup(problem: Problem | VectorFn, x0: Any) -> tuple[_Objective, Vector]:
    """The counted objective and the start vector.

    The methods work on vectors x ∈ ℝⁿ. A :class:`Problem` with ``dim == 1`` follows the scalar
    convention of ``core.types`` (f takes and returns floats), so f receives ``float(x[0])``.
    This includes the Problem built from a bare callable with a one-entry x0
    (``core.counting.vector_problem`` gives it ``dim == 1``), so that a direct call and
    ``numopt.minimize`` treat it the same way (as in ``conjugate_gradient`` and
    ``trust_region``). A size-1 array returned by f is accepted as the value f(x).
    """
    prob = vector_problem(problem, x0=x0)
    x = start_point(prob, x0)
    if x.size < 1:
        raise ValueError("x0 must have at least one entry")
    return _Objective(prob.f, scalar=prob.dim == 1), x


def _bad_start(method: str, x0: Vector, f0: float, nfev: int) -> Result:
    msg = f"f(x0) is not finite (f = {f0!r}); cannot start"
    return Result(method, x0, f0, False, msg, 0, nfev, trace=[Step(0, x0.copy(), f0)])


def _unbounded(method: str, x: Vector, trace: list[Step], k: int, nfev: int) -> Result:
    msg = f"f = -inf at x = {x.tolist()}: f is unbounded below"
    return Result(method, x, -math.inf, False, msg, k, nfev, trace=trace)


def _max_iter(
    method: str, x: Vector, fx: float, max_iter: int, nfev: int, trace: list[Step]
) -> Result:
    msg = f"reached max_iter={max_iter}"
    return Result(method, x, fx, False, msg, max_iter, nfev, trace=trace)


def _check_positive(**values: float) -> None:
    for name, v in values.items():
        if not v > 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")


def _check_max_iter(max_iter: int) -> None:
    """Reject a ``max_iter`` that the ``k == max_iter`` test can never meet (0, < 0, non-integer)."""
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")


def _check_shrink(shrink: float) -> None:
    if not 0.0 < shrink < 1.0:
        raise ValueError(f"shrink must lie in (0, 1), got {shrink}")


# --------------------------------------------------------------------------------------
# Nelder–Mead
# --------------------------------------------------------------------------------------


class _TrialLog:
    """Evaluate a Nelder–Mead trial point and record it as ``{"op", "x", "f"}``."""

    __slots__ = ("fobj", "trials")

    def __init__(self, fobj: _Objective, trials: list[dict[str, Any]]) -> None:
        self.fobj = fobj
        self.trials = trials

    def __call__(self, op: str, point: Vector) -> float:
        fp = self.fobj(point)
        self.trials.append({"op": op, "x": point, "f": fp})
        return fp


def _nm_coefficients(n: int, adaptive: bool) -> tuple[float, float, float, float]:
    """(ρ, χ, γ, σ): LRWW (1998) standard values, or Gao & Han (2012) eq. (4.1) adaptive ones."""
    if not adaptive:
        return 1.0, 2.0, 0.5, 0.5
    if n < 2:
        raise ValueError("adaptive Nelder–Mead needs n ≥ 2 (its shrink σ = 1 − 1/n is 0 for n = 1)")
    return 1.0, 1.0 + 2.0 / n, 0.75 - 0.5 / n, 1.0 - 1.0 / n


@register(
    id="nelder_mead",
    family="unconstrained",
    name="Nelder–Mead simplex",
    params=(
        ParamSpec(
            "xtol",
            1e-8,
            min=1e-14,
            max=1e-1,
            log=True,
            help="Stop when the simplex size max‖x_i − x_1‖∞ ≤ xtol (and the f-spread test holds).",
        ),
        ParamSpec(
            "ftol",
            1e-8,
            min=1e-14,
            max=1e-1,
            log=True,
            help="Stop when f(worst) − f(best) ≤ ftol (and the size test holds).",
        ),
        ParamSpec(
            "initial_step",
            0.5,
            min=1e-3,
            max=5.0,
            log=True,
            help="Edge length h of the initial simplex x₀, x₀ + h e₁, …, x₀ + h eₙ.",
        ),
        ParamSpec(
            "adaptive",
            False,
            kind="bool",
            help="Use the dimension-dependent coefficients of Gao & Han (2012).",
        ),
        ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Iteration limit."),
    ),
    needs=("f",),
    order="no general rate (may stall on non-stationary points, McKinnon 1998)",
    summary="Move a simplex of n + 1 points by reflecting, expanding, contracting or shrinking it.",
    references=(
        "Nelder & Mead (1965), Comput. J. 7(4), 308–313",
        "Lagarias, Reeds, Wright & Wright (1998), SIAM J. Optim. 9(1), 112–147, §2",
        "Gao & Han (2012), Comput. Optim. Appl. 51(1), 259–277, eq. (4.1)",
    ),
)
def nelder_mead(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    initial_step: float = 0.5,
    adaptive: bool = False,
    max_iter: int = 1000,
) -> Result:
    """Nelder–Mead simplex method in the form of Lagarias, Reeds, Wright & Wright (1998), §2.

    The simplex x_1, …, x_{n+1} is ordered f_1 ≤ … ≤ f_{n+1}; x̄ = (1/n)Σ_{i≤n} x_i is the
    centroid of the n best vertices. One iteration:

    1. Reflect: x_r = x̄ + ρ(x̄ − x_{n+1}). If f_1 ≤ f_r < f_n, accept x_r.
    2. Expand: if f_r < f_1, x_e = x̄ + ρχ(x̄ − x_{n+1}); accept x_e if f_e < f_r, else x_r.
    3. Contract (f_r ≥ f_n):
       outside, if f_r < f_{n+1}: x_c = x̄ + ργ(x̄ − x_{n+1}); accept if f_c ≤ f_r;
       inside, if f_r ≥ f_{n+1}: x_cc = x̄ − γ(x̄ − x_{n+1}); accept if f_cc < f_{n+1};
       otherwise shrink.
    4. Shrink: v_i = x_1 + σ(x_i − x_1), i = 2, …, n + 1 (n evaluations).

    The accepted point replaces x_{n+1}. Ties are broken as in LRWW: a new vertex is placed
    after the old vertices with the same value, and after a shrink x_1 stays first on a tie
    (both follow from a stable sort). Coefficients (ρ, χ, γ, σ) are (1, 2, ½, ½); with
    ``adaptive`` they are Gao & Han's (1, 1 + 2/n, ¾ − 1/(2n), 1 − 1/n), which keep the
    expansions and contractions less extreme in higher dimension.

    The initial simplex is x0 and x0 + h e_i (i = 1, …, n), h = ``initial_step``; it costs
    n + 1 evaluations. An iteration costs 1 or 2 evaluations, or n + 1 or n + 2 with a shrink.

    Stops (converged) when, at the start of an iteration, both
    max_i ‖x_i − x_1‖∞ ≤ xtol + 2ε‖x_1‖∞ and f_{n+1} − f_1 ≤ ftol + 2ε|f_1|
    (the test of SciPy's ``fmin`` with a rounding floor). This certifies only that the simplex
    has collapsed: Nelder–Mead can converge to a non-stationary point (McKinnon 1998).
    Returns the best vertex x_1.
    """
    _check_max_iter(max_iter)
    _check_positive(xtol=xtol, ftol=ftol, initial_step=initial_step)
    fobj, x_start = _setup(problem, x0)
    n = x_start.size
    rho, chi, gamma, sigma = _nm_coefficients(n, adaptive)
    coefficients = {"rho": rho, "chi": chi, "gamma": gamma, "sigma": sigma}

    f_start = fobj(x_start)
    if not math.isfinite(f_start):
        return _bad_start("nelder_mead", x_start, f_start, fobj.n)

    simplex: list[Vector] = [x_start.copy()]
    for i in range(n):
        v = x_start.copy()
        v[i] += initial_step
        simplex.append(v)
    fvals: list[float] = [f_start] + [fobj(v) for v in simplex[1:]]

    def sort() -> None:
        order = sorted(range(n + 1), key=lambda i: fvals[i])  # stable: LRWW tie rules
        simplex[:] = [simplex[i] for i in order]
        fvals[:] = [fvals[i] for i in order]

    def size() -> float:
        return max(float(np.max(np.abs(v - simplex[0]))) for v in simplex[1:])

    def info(
        op: str | None, centroid: Vector | None, worst: Vector | None, trials: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "simplex": [v.copy() for v in simplex],
            "simplex_f": list(fvals),
            "operation": op,
            "centroid": centroid,
            "worst": worst,
            "trials": trials,
            "size": size(),
            "f_spread": fvals[n] - fvals[0],
            "coefficients": coefficients,
        }

    sort()
    trace = [
        Step(0, simplex[0].copy(), fvals[0], step_size=size(), info=info(None, None, None, []))
    ]
    if fvals[0] == -math.inf:
        return _unbounded("nelder_mead", simplex[0].copy(), trace, 0, fobj.n)
    k = 0
    while True:
        tol_x = xtol + 2.0 * EPS * float(np.max(np.abs(simplex[0])))
        tol_f = ftol + 2.0 * EPS * abs(fvals[0])
        sz, spread = size(), fvals[n] - fvals[0]
        if sz <= tol_x and spread <= tol_f:
            msg = f"simplex size {sz:.3g} ≤ {tol_x:.3g} and f-spread {spread:.3g} ≤ {tol_f:.3g}"
            return Result(
                "nelder_mead", simplex[0].copy(), fvals[0], True, msg, k, fobj.n, trace=trace
            )
        if k == max_iter:
            return _max_iter("nelder_mead", simplex[0].copy(), fvals[0], max_iter, fobj.n, trace)
        k += 1

        worst = simplex[n].copy()
        centroid = np.mean(np.stack(simplex[:n]), axis=0)  # (n,)
        trials: list[dict[str, Any]] = []
        trial = _TrialLog(fobj, trials)

        x_r = centroid + rho * (centroid - worst)
        f_r = trial("reflect", x_r)
        new: tuple[Vector, float] | None = None
        if fvals[0] <= f_r < fvals[n - 1]:
            op, new = "reflect", (x_r, f_r)
        elif f_r < fvals[0]:
            x_e = centroid + rho * chi * (centroid - worst)
            f_e = trial("expand", x_e)
            op, new = ("expand", (x_e, f_e)) if f_e < f_r else ("reflect", (x_r, f_r))
        elif f_r < fvals[n]:
            x_c = centroid + rho * gamma * (centroid - worst)
            f_c = trial("contract_outside", x_c)
            op = "contract_outside" if f_c <= f_r else "shrink"
            new = (x_c, f_c) if f_c <= f_r else None
        else:
            x_cc = centroid - gamma * (centroid - worst)
            f_cc = trial("contract_inside", x_cc)
            op = "contract_inside" if f_cc < fvals[n] else "shrink"
            new = (x_cc, f_cc) if f_cc < fvals[n] else None

        if new is not None:
            simplex[n], fvals[n] = new
        else:
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                fvals[i] = fobj(simplex[i])
        sort()
        trace.append(
            Step(
                k,
                simplex[0].copy(),
                fvals[0],
                step_size=size(),
                info=info(op, centroid, worst, trials),
            )
        )
        if fvals[0] == -math.inf:
            return _unbounded("nelder_mead", simplex[0].copy(), trace, k, fobj.n)


# --------------------------------------------------------------------------------------
# One-dimensional minimization along a line: bracketing (NR §10.1) + Brent (NR §10.3)
# --------------------------------------------------------------------------------------

#: Golden-ratio magnification of mnbrak and the limit of a parabolic extrapolation (NR §10.1).
_GOLD = 0.5 * (1.0 + math.sqrt(5.0))
_GLIMIT = 100.0
_TINY = 1e-20
#: Give up bracketing after this many expansions (f is then decreasing along the whole line).
_BRACKET_MAX_ITER = 100
#: Brent: golden-section fraction (3 − √5)/2, relative tolerance (NR3 ``Brent`` default),
#: absolute floor (NR3 ZEPS) and iteration limit (NR3 ITMAX).
_CGOLD = 0.5 * (3.0 - math.sqrt(5.0))
_BRENT_TOL = 3.0e-8
_BRENT_ZEPS = EPS * 1.0e-3
_BRENT_MAX_ITER = 100


@dataclass
class _LineResult:
    alpha: float
    f: float
    trials: list[list[float]] = field(default_factory=list)
    ok: bool = True
    message: str = ""


def _line_minimize(phi: Callable[[float], float], f0: float) -> _LineResult:
    """Minimize φ(α) = f(x + α d) from α = 0 (where φ = f0): mnbrak, then Brent's localmin.

    Bracketing follows ``mnbrak`` (Press et al., NR 3rd ed., §10.1) from the points 0 and 1;
    the minimization is Brent's ``localmin`` (Brent 1973, ch. 5; NR §10.3) with tolerance
    tol·|α| + ZEPS. Returns the best α found (φ(α) ≤ f0 always). ``ok`` is False when f is
    −∞ somewhere or no bracket is found in ``_BRACKET_MAX_ITER`` expansions.
    """
    trials: list[list[float]] = []

    def ev(a: float) -> float:
        v = phi(a)
        trials.append([a, v])
        return v

    def fail(message: str) -> _LineResult:
        # Report the best finite point seen (never −∞) so the caller keeps a finite iterate.
        finite_trials = [[0.0, f0]] + [t for t in trials if math.isfinite(t[1])]
        a_best, f_best = min(finite_trials, key=lambda t: t[1])
        return _LineResult(a_best, f_best, trials, False, message)

    # ---- mnbrak ----
    ax, bx = 0.0, 1.0
    fa, fb = f0, ev(bx)
    if fb == -math.inf:
        return fail("f = -inf on the line: f is unbounded below")
    if fb > fa:
        ax, bx, fa, fb = bx, ax, fb, fa
    cx = bx + _GOLD * (bx - ax)
    fc = ev(cx)
    n_expand = 0
    while fb > fc:
        if fc == -math.inf:
            return fail("f = -inf on the line: f is unbounded below")
        n_expand += 1
        if n_expand > _BRACKET_MAX_ITER:
            return fail("could not bracket a minimum along the line (f keeps decreasing)")
        r = (bx - ax) * (fb - fc)
        q = (bx - cx) * (fb - fa)
        u = bx - ((bx - cx) * q - (bx - ax) * r) / (
            2.0 * math.copysign(max(abs(q - r), _TINY), q - r)
        )
        ulim = bx + _GLIMIT * (cx - bx)
        if not math.isfinite(u):
            # NOTE: a parabola through a +inf (barrier) value is undefined; NR's formula would
            # give NaN here. Fall back to the default golden magnification.
            u = cx + _GOLD * (cx - bx)
            fu = ev(u)
        elif (bx - u) * (u - cx) > 0.0:  # parabolic u between b and c
            fu = ev(u)
            if fu < fc:  # minimum between b and c
                ax, bx, fa, fb = bx, u, fb, fu
                break
            if fu > fb:  # minimum between a and u
                cx, fc = u, fu
                break
            u = cx + _GOLD * (cx - bx)
            fu = ev(u)
        elif (cx - u) * (u - ulim) > 0.0:  # parabolic u between c and its limit
            fu = ev(u)
            if fu < fc:
                bx, cx, u = cx, u, u + _GOLD * (u - cx)
                fb, fc = fc, fu
                fu = ev(u)
        elif (u - ulim) * (ulim - cx) >= 0.0:  # limit parabolic u to its maximum value
            u = ulim
            fu = ev(u)
        else:  # reject parabolic u, use default magnification
            u = cx + _GOLD * (cx - bx)
            fu = ev(u)
        if fu == -math.inf:
            return fail("f = -inf on the line: f is unbounded below")
        ax, bx, cx = bx, cx, u
        fa, fb, fc = fb, fc, fu
    if fc == -math.inf or fb == -math.inf:
        return fail("f = -inf on the line: f is unbounded below")

    # ---- Brent's localmin on [min(a, c), max(a, c)] starting from the bracket's middle b ----
    a, b = (ax, cx) if ax < cx else (cx, ax)
    x = w = v = bx
    fx = fw = fv = fb
    d = e = 0.0
    for _ in range(_BRENT_MAX_ITER):
        xm = 0.5 * (a + b)
        tol1 = _BRENT_TOL * abs(x) + _BRENT_ZEPS
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp, e = e, d
            # NOTE: with a +inf (barrier) value among fx, fw, fv the parabola is undefined
            # (p, q are NaN); NR's tests would then accept it. Treat it as unacceptable.
            acceptable = (
                math.isfinite(p)
                and math.isfinite(q)
                and not (abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x))
            )
            if acceptable:
                golden = False
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = math.copysign(tol1, xm - x)
        if golden:
            e = (a - x) if x >= xm else (b - x)
            d = _CGOLD * e
        u = x + d if abs(d) >= tol1 else x + math.copysign(tol1, d)
        fu = ev(u)
        if fu == -math.inf:
            return fail("f = -inf on the line: f is unbounded below")
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    # NOTE: NR treats ITMAX as an error; we return Brent's best point, which still has
    # φ(x) ≤ φ(0), so the outer method keeps a monotone decrease.
    if fx > f0:  # cannot happen (b has the lowest value of the bracket and f(0) is in it)
        return _LineResult(0.0, f0, trials, True, "no decrease along the line")
    return _LineResult(x, fx, trials, True, "")


# --------------------------------------------------------------------------------------
# Powell's conjugate-direction method
# --------------------------------------------------------------------------------------

#: Absolute floor of Powell's relative stopping test (NR3 ``Powell``: TINY = 1e-25).
_POWELL_TINY = 1.0e-25


@register(
    id="powell",
    family="unconstrained",
    name="Powell (conjugate directions)",
    params=(
        ParamSpec(
            "ftol",
            1e-10,
            min=1e-15,
            max=1e-2,
            log=True,
            help=(
                "Relative f test: a sweep decreases f by ≤ ftol·(|f_old| + |f_new|)/2 (+10⁻²⁵). "
                "Relative, so it depends on the size of f, not only on its variation."
            ),
        ),
        ParamSpec(
            "xtol",
            1e-8,
            min=1e-14,
            max=1e-1,
            log=True,
            help=(
                "Also required: the sweep moved x by ≤ xtol + 2ε‖x‖∞ (∞-norm), "
                "or decreased f only at rounding level (≤ 2ε|f|)."
            ),
        ),
        ParamSpec("max_iter", 200, kind="int", min=1, max=10_000, help="Sweep limit."),
    ),
    needs=("f",),
    order=(
        "no quadratic termination with NR's discarding rule (2n–3n sweeps on random "
        "quadratics); Powell's (1964) basic method: n sweeps"
    ),
    summary="Minimize along n directions in turn, then swap in the net displacement as a new one.",
    references=(
        "Powell (1964), Comput. J. 7(2), 155–162",
        "Press et al., Numerical Recipes (3rd ed.), §10.7 (Powell) with §10.1 (mnbrak), §10.3 (Brent)",
    ),
)
def powell(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    ftol: float = 1e-10,
    xtol: float = 1e-8,
    max_iter: int = 200,
) -> Result:
    """Powell's method with the direction-discarding rule of Numerical Recipes §10.7.

    Directions u_1, …, u_n start as the coordinate vectors e_i. One iteration (sweep), from
    x_0 with f_0 = f(x_0):

    1. For i = 1, …, n: x_i = x_{i−1} + α_i u_i with α_i = argmin_α f(x_{i−1} + α u_i)
       (bracketing + Brent; see ``_line_minimize``). Record the largest single decrease
       Δf = max_i (f_{i−1} − f_i) and its index i_big.
    2. Stop (converged) if both
       (a) 2(f_0 − f_n) ≤ ftol(|f_0| + |f_n|) + 10⁻²⁵   (NR's relative test), and
       (b) ‖x_n − x_0‖∞ ≤ xtol + 2ε‖x_n‖∞, or f_0 − f_n ≤ 2ε|f_n|.
    3. Extrapolate: x_E = 2x_n − x_0, f_E = f(x_E). Keep the old directions if f_E ≥ f_0 or
       t = 2(f_0 − 2f_n + f_E)(f_0 − f_n − Δf)² − Δf(f_0 − f_E)² ≥ 0 (Powell 1964; NR eq.
       10.7.7). Otherwise minimize along u = x_n − x_0 from x_n, discard u_{i_big} (its place
       is taken by u_n) and append u as the new u_n.

    NOTE: test (b) is not in NR. Test (a) is relative, so it is not invariant to a constant
    added to f: for f = c + g with |c| ≫ |g| it accepts an absolute decrease ≈ ftol·|c| per
    sweep and stops far from the minimizer (Rosenbrock + 10⁸: ‖x − x*‖ ≈ 0.1). Test (b) also
    asks that the sweep did not move x (n line minima along independent directions at the
    same point make every directional derivative vanish), or that f could not be decreased
    beyond rounding (the iterate is on the rounding plateau of f, where x cannot be resolved).

    On a strictly convex quadratic with exact line searches the method with replacement of
    the oldest direction terminates in n sweeps (Powell 1964). The discarding rule loses this
    quadratic termination in exchange for protection against linearly dependent directions;
    random SPD quadratics need 2n to 3n sweeps (n = 5, 8, κ ≤ 10³) to reach f ≤ 10⁻¹² f(x0).

    NOTE: NR stores the new direction scaled by its line minimum, α·u; we store u = x_n − x_0,
    Powell's (1964) direction (the scale does not change conjugacy). The line minimum along u
    from x_n is often α ≈ 0 (u is nearly parallel to the last direction that moved x), and then
    α·u is a near-zero vector that replaces a useful direction. Every later line search along
    it sees only rounding-level changes of f at α = 1, 2.618, …, so the set loses a dimension
    and the method stops early. Example: f = ±100 + ½(x − c)ᵀA(x − c), n = 3, κ(A) ≈ 30 stopped
    with ‖x − c‖_A = 2.4e-3 with α·u, and reaches the minimizer with u.
    NOTE: f(x_0) is reused for the first line search (NR's mnbrak re-evaluates it).
    """
    _check_max_iter(max_iter)
    _check_positive(ftol=ftol, xtol=xtol)
    fobj, p = _setup(problem, x0)
    n = p.size
    fret = fobj(p)
    if not math.isfinite(fret):
        return _bad_start("powell", p, fret, fobj.n)
    directions: list[Vector] = [np.eye(n)[i] for i in range(n)]
    info0 = {
        "directions": [d.copy() for d in directions],
        "new_directions": [d.copy() for d in directions],
    }
    info0 |= {"lines": [], "extrapolated": None, "f_extrapolated": None, "largest_decrease": 0.0}
    info0 |= {"largest_index": 0, "replace_test": None, "replaced": None}
    trace = [Step(0, p.copy(), fret, info=info0)]

    def line(origin: Vector, d: Vector, f_origin: float) -> tuple[_LineResult, dict[str, Any]]:
        res = _line_minimize(lambda a: fobj(origin + a * d), f_origin)
        point = origin + res.alpha * d
        rec = {
            "origin": origin.copy(),
            "direction": d.copy(),
            "alpha": res.alpha,
            "point": point,
            "f": res.f,
            "trials": res.trials,
        }
        return res, rec

    k = 0
    while True:
        k += 1
        f0, x_start = fret, p.copy()
        used = [d.copy() for d in directions]
        lines: list[dict[str, Any]] = []
        delta, ibig = 0.0, 0
        info: dict[str, Any] = {"directions": used, "lines": lines, "extrapolated": None}
        info |= {"f_extrapolated": None, "replace_test": None, "replaced": None}

        failure: str | None = None
        unbounded_at: Vector | None = None
        stop_msg: str | None = None
        for i in range(n):
            f_before = fret
            res, rec = line(p, directions[i], fret)
            lines.append(rec)
            p, fret = rec["point"], res.f
            if not res.ok:
                failure = f"line minimization along direction {i} failed: {res.message}"
                break
            if f_before - fret > delta:
                delta, ibig = f_before - fret, i

        f_test = 2.0 * (f0 - fret) <= ftol * (abs(f0) + abs(fret)) + _POWELL_TINY
        sweep_step = float(np.max(np.abs(p - x_start)))
        x_test = sweep_step <= xtol + 2.0 * EPS * float(np.max(np.abs(p)))
        plateau = f0 - fret <= 2.0 * EPS * abs(fret)
        if failure is None and f_test and (x_test or plateau):
            stop_msg = (
                f"sweep decrease 2(f₀ − f_n) = {2.0 * (f0 - fret):.3g} ≤ ftol·(|f₀| + |f_n|) + 1e-25"
                + (
                    f" and sweep step {sweep_step:.3g} ≤ xtol + 2ε‖x‖∞"
                    if x_test
                    else " and f decreased only at rounding level (≤ 2ε|f|)"
                )
            )
        elif failure is None:
            x_e = 2.0 * p - x_start
            f_e = fobj(x_e)
            info["extrapolated"], info["f_extrapolated"] = x_e, f_e
            if f_e == -math.inf:
                unbounded_at = x_e
            elif f_e < f0:
                t = (
                    2.0 * (f0 - 2.0 * fret + f_e) * (f0 - fret - delta) ** 2
                    - delta * (f0 - f_e) ** 2
                )
                info["replace_test"] = t
                if t < 0.0:
                    u = p - x_start
                    res, rec = line(p, u, fret)
                    lines.append(rec)
                    p, fret = rec["point"], res.f
                    if res.ok:
                        directions[ibig] = directions[n - 1]
                        directions[n - 1] = u
                        info["replaced"] = ibig
                    else:
                        failure = f"line minimization along the new direction failed: {res.message}"

        info["new_directions"] = [d.copy() for d in directions]
        info["largest_decrease"] = delta
        info["largest_index"] = ibig
        step = float(np.linalg.norm(p - trace[-1].x))
        trace.append(Step(k, p.copy(), fret, step_size=step, info=info))
        if unbounded_at is not None:
            return _unbounded("powell", unbounded_at, trace, k, fobj.n)
        if failure is not None:
            return Result("powell", p.copy(), fret, False, failure, k, fobj.n, trace=trace)
        if stop_msg is not None:
            return Result("powell", p.copy(), fret, True, stop_msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("powell", p.copy(), fret, max_iter, fobj.n, trace)


# --------------------------------------------------------------------------------------
# Hooke–Jeeves pattern search
# --------------------------------------------------------------------------------------

_STEP_PARAMS = (
    ParamSpec(
        "step",
        0.5,
        min=1e-3,
        max=5.0,
        log=True,
        help="Initial step length along each coordinate.",
    ),
    ParamSpec(
        "shrink",
        0.5,
        min=0.05,
        max=0.95,
        help="Factor that multiplies the step after an unsuccessful iteration.",
    ),
    ParamSpec(
        "xtol",
        1e-8,
        min=1e-14,
        max=1e-1,
        log=True,
        help="Stop when the step falls below xtol.",
    ),
    ParamSpec("max_iter", 1000, kind="int", min=1, max=100_000, help="Iteration limit."),
)


@register(
    id="hooke_jeeves",
    family="unconstrained",
    name="Hooke–Jeeves pattern search",
    params=_STEP_PARAMS,
    needs=("f",),
    order="linear at best; ‖∇f‖ = O(step) at unsuccessful iterations",
    summary="Probe each coordinate, then jump along the direction of the last success.",
    references=(
        "Hooke & Jeeves (1961), J. ACM 8(2), 212–229",
        "Kelley (1999), Iterative Methods for Optimization, SIAM, §8.3",
        "Torczon (1997), SIAM J. Optim. 7(1), 1–25 (convergence)",
    ),
)
def hooke_jeeves(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    step: float = 0.5,
    shrink: float = 0.5,
    xtol: float = 1e-8,
    max_iter: int = 1000,
) -> Result:
    """Hooke–Jeeves pattern search (Hooke & Jeeves 1961; Kelley 1999, §8.3).

    Exploratory move about a point y with step h: for i = 1, …, n, try y + h e_i; if it does
    not lower the current value try y − h e_i; keep whichever lowers it (else keep y).

    One iteration is one exploratory move:

    * ``explore`` (about the base b): if it finds x with f(x) < f(b), the base moves,
      b_old ← b, b ← x, and the next iteration is a pattern move; otherwise h ← shrink·h.
    * ``pattern``: probe the pattern point p = b + (b − b_old) = 2b − b_old, then explore
      about p (its reference value is f(p)). If the result x has f(x) < f(b), then
      b_old ← b, b ← x and the next iteration is another pattern move; otherwise the pattern
      is discarded and the next iteration explores about b with the same h.

    Stops (converged) when the step h ≤ xtol after a reduction. For f ∈ C¹ with Lipschitz
    gradient, an exploratory failure with step h implies ‖∇f(b)‖ = O(h) (Torczon 1997;
    Kolda, Lewis & Torczon 2003, eq. 3.3), so the test is a first-order stationarity test.
    Returns the base point.
    """
    _check_max_iter(max_iter)
    _check_positive(step=step, xtol=xtol)
    _check_shrink(shrink)
    fobj, base = _setup(problem, x0)
    n = base.size
    f_base = fobj(base)
    if not math.isfinite(f_base):
        return _bad_start("hooke_jeeves", base, f_base, fobj.n)
    h = step
    prev_base: Vector | None = None
    mode = "explore"
    info0 = {"move": None, "outcome": None, "base": base.copy(), "previous_base": None}
    info0 |= {"pattern_point": None, "probes": [], "step": h, "new_step": h}
    trace = [Step(0, base.copy(), f_base, step_size=h, info=info0)]

    def explore(y: Vector, fy: float, probes: list[dict[str, Any]]) -> tuple[Vector, float]:
        y = y.copy()
        for i in range(n):
            for sgn in (1.0, -1.0):
                trial = y.copy()
                trial[i] += sgn * h
                ft = fobj(trial)
                probes.append({"x": trial, "f": ft})
                if ft < fy:
                    y, fy = trial, ft
                    break
        return y, fy

    k = 0
    while True:
        k += 1
        probes: list[dict[str, Any]] = []
        h_used = h
        pattern_point: Vector | None = None
        move = mode
        if mode == "pattern":
            assert prev_base is not None
            pattern_point = 2.0 * base - prev_base
            f_p = fobj(pattern_point)
            probes.append({"x": pattern_point.copy(), "f": f_p})
            x, fx = explore(pattern_point, f_p, probes)
            if fx < f_base:
                prev_base, base, f_base = base, x, fx
                outcome = "pattern_success"
            else:
                outcome, mode = "pattern_failed", "explore"
        else:
            x, fx = explore(base, f_base, probes)
            if fx < f_base:
                prev_base, base, f_base = base, x, fx
                outcome, mode = "explore_success", "pattern"
            else:
                h *= shrink
                outcome = "step_reduced"
        info = {
            "move": move,
            "outcome": outcome,
            "base": base.copy(),
            "previous_base": None if prev_base is None else prev_base.copy(),
            "pattern_point": pattern_point,
            "probes": probes,
            "step": h_used,
            "new_step": h,
        }
        trace.append(Step(k, base.copy(), f_base, step_size=h_used, info=info))
        if f_base == -math.inf:
            return _unbounded("hooke_jeeves", base.copy(), trace, k, fobj.n)
        if outcome == "step_reduced" and h <= xtol:
            msg = f"step h = {h:.3g} ≤ xtol after an unsuccessful exploratory move"
            return Result("hooke_jeeves", base.copy(), f_base, True, msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("hooke_jeeves", base.copy(), f_base, max_iter, fobj.n, trace)


# --------------------------------------------------------------------------------------
# Compass search
# --------------------------------------------------------------------------------------


@register(
    id="compass_search",
    family="unconstrained",
    name="Compass search",
    params=_STEP_PARAMS,
    needs=("f",),
    order="linear at best; ‖∇f‖ ≤ √n·M·Δ at unsuccessful iterations",
    summary="Try a step north, south, east and west; move on the first success, else halve the step.",
    references=(
        "Kolda, Lewis & Torczon (2003), SIAM Review 45(3), 385–482, Alg. 3.1 and eq. (3.3)",
    ),
)
def compass_search(
    problem: Problem | VectorFn,
    *,
    x0: Any = None,
    step: float = 0.5,
    shrink: float = 0.5,
    xtol: float = 1e-8,
    max_iter: int = 1000,
) -> Result:
    """Compass search, Kolda, Lewis & Torczon (2003), Algorithm 3.1.

    With D⊕ = (e_1, …, e_n, −e_1, …, −e_n), one iteration polls x_k + Δ_k d for d ∈ D⊕ in
    that order and stops polling at the first d_k with f(x_k + Δ_k d_k) < f(x_k) (simple
    decrease):

    * success: x_{k+1} = x_k + Δ_k d_k, Δ_{k+1} = Δ_k;
    * failure (no poll point decreases f): x_{k+1} = x_k, Δ_{k+1} = shrink·Δ_k (KLT use ½).

    Stops (converged) after an unsuccessful iteration with Δ_{k+1} < xtol (KLT's Δ_tol). If ∇f
    is Lipschitz with constant M, an unsuccessful iteration gives ‖∇f(x_k)‖ ≤ √n·M·Δ_k (KLT
    eq. 3.3), so the test certifies approximate stationarity. Each iteration costs 1 to 2n
    evaluations.
    """
    _check_max_iter(max_iter)
    _check_positive(step=step, xtol=xtol)
    _check_shrink(shrink)
    fobj, x = _setup(problem, x0)
    n = x.size
    fx = fobj(x)
    if not math.isfinite(fx):
        return _bad_start("compass_search", x, fx, fobj.n)
    delta = step
    info0 = {"polls": [], "success": None, "direction": None, "step": delta, "new_step": delta}
    trace = [Step(0, x.copy(), fx, step_size=delta, info=info0)]
    eye = np.eye(n)
    compass = [eye[i] for i in range(n)] + [-eye[i] for i in range(n)]
    k = 0
    while True:
        k += 1
        polls: list[dict[str, Any]] = []
        direction: int | None = None
        for j, d in enumerate(compass):
            y = x + delta * d
            fy = fobj(y)
            polls.append({"x": y, "f": fy})
            if fy < fx:
                x, fx, direction = y, fy, j
                break
        delta_used = delta
        if direction is None:
            delta *= shrink
        info = {
            "polls": polls,
            "success": direction is not None,
            "direction": direction,
            "step": delta_used,
            "new_step": delta,
        }
        trace.append(Step(k, x.copy(), fx, step_size=delta_used, info=info))
        if fx == -math.inf:
            return _unbounded("compass_search", x.copy(), trace, k, fobj.n)
        if direction is None and delta < xtol:
            msg = f"step Δ = {delta:.3g} < xtol after an unsuccessful poll"
            return Result("compass_search", x.copy(), fx, True, msg, k, fobj.n, trace=trace)
        if k == max_iter:
            return _max_iter("compass_search", x.copy(), fx, max_iter, fobj.n, trace)


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("nelder_mead", "rosenbrock", {}),
    ("nelder_mead", "himmelblau", {"adaptive": True}),
    ("powell", "rosenbrock", {}),
    ("powell", "beale", {}),
    ("hooke_jeeves", "booth", {}),
    ("hooke_jeeves", "himmelblau", {"xtol": 1e-6}),
    ("compass_search", "quadratic_bowl", {"xtol": 1e-6}),
    ("compass_search", "six_hump_camel", {"xtol": 1e-6}),
]
