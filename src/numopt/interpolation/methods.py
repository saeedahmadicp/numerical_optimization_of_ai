"""Polynomial and piecewise-polynomial interpolation of data (x_i, y_i), i = 0..n-1.

Every method builds an interpolant p with p(x_i) = y_i and returns it in a representation
that a plot can evaluate without recomputation:

* ``Result.x``: the coefficient vector of that representation (see each method); for the
  piecewise methods it is the (n-1) × 4 matrix of local coefficients flattened row by row.
  Neville's algorithm is a pointwise scheme with no coefficients, so its ``Result.x`` is
  the value p(x*) at the evaluation point.
* ``Result.fun``: max_t |p(t) - f_true(t)| over the 200-point plotting grid when the
  dataset has ``f_true``; otherwise ``None``.
* ``Result.extra``::

      kind:          "lagrange" | "barycentric" | "newton" | "neville" | "piecewise_cubic"
                     | "chebyshev"
      coefficients:  the representation's coefficients (piecewise: [[a, b, c, d]] per segment,
                     S_i(t) = a + b(t - x_i) + c(t - x_i)² + d(t - x_i)³ on [x_i, x_{i+1}])
      nodes:         [n] interpolation nodes in the order the method uses them (piecewise
                     methods and Chebyshev: ascending)
      values:        [n] the data values at ``nodes``
      domain:        [a, b] plotting domain (dataset.domain, else [min x, max x])
      eval:          {x: [200] grid, y: [200] p on the grid, f_true: [200] or None}
      max_error:     same as Result.fun
      node_residual: max_i |p(x_i) - y_i| evaluated with the method's own evaluator
                     (shows round-off, e.g. of the Newton form at high degree)

  plus method-specific keys (documented on each method).

``converged`` is ``True`` when the construction finished, every coefficient and grid value
is finite, and the interpolant reproduces the data: max_i |p(x_i) - y_i| ≤ 1e-6·max(S,
2⁻¹⁰²²) (``NODE_RTOL``; interpolation is a direct method, this is its only test), where
S = max_i |y_i| is the size of the data (the clamped spline adds its end slopes,
S = max(max|y|, max(|f'_a|, |f'_b|)·max h_i)). The test is relative, so the flag does not
depend on the units of y; 2⁻¹⁰²² (the smallest normal number) only keeps the limit
positive for all-zero or subnormal data.
Overflow (e.g. barycentric weights for hundreds of widely spaced nodes), a zero pivot, or
round-off that destroys the interpolation conditions (e.g. the Newton form with hundreds
of nodes) gives ``converged=False`` with a message. Invalid data (NaN, length mismatch, repeated
nodes) raises ``ValueError``. ``n_fev`` counts calls of ``f_true`` that the *algorithm*
makes (only Chebyshev interpolation: its sample check and resampling); the error
diagnostics on the grid are not counted.

The polynomial methods use the nodes in the given order (the order changes the trace,
not the interpolant). The piecewise methods sort the data by x.

Info keys:
    curve: [200]          the current approximant on ``extra.eval.x`` (every polynomial
                          step; the final step of the piecewise methods)
    node_index: int       lagrange/barycentric/newton: index of the node added at this step
    node: [x, y]          lagrange/barycentric/newton: that node
    basis: [200]          lagrange: the basis polynomial ℓ_k on the grid
    new_row: [k+1]        newton: the new row of the divided-difference table,
                          [f[x_k], f[x_{k-1},x_k], ..., f[x_0..x_k]]
    column: [n-k]         neville: column k of the tableau at x*, [Q_{k,k}, ..., Q_{n-1,k}]
    x_eval: float         neville: the evaluation point x*
    term_index: int       chebyshev: index k of the term c_k T_k added at this step
    stage: str            piecewise: "assemble" | "forward_sweep" | "back_substitution" |
                          "secants" | "slopes" | "coefficients"
    h: [n-1]              piecewise: spacings h_i = x_{i+1} - x_i
    secants: [n-1]        piecewise: secant slopes m_i = (y_{i+1} - y_i)/h_i
    boundary: str         cubic splines: the end conditions used
    lower, diag, upper: [n-1], [n], [n-1]   cubic splines: bands of the tridiagonal system
    rhs: [n]              cubic splines: right-hand side of the system
    pivots: [n]           cubic splines: Thomas-algorithm pivots after the forward sweep
    multipliers: [n-1]    cubic splines: elimination multipliers l_i = lower_{i-1}/pivot_{i-1}
    slopes: [n]           cubic splines/pchip: node slopes s_i = S'(x_i)
    limited: [n]          pchip: True where the slope was set by a shape-preserving rule
                          (zero at a local extremum / flat segment, or end-slope limiting)
    alpha, beta: [n-1]    pchip: s_i/m_i and s_{i+1}/m_i per segment (None when m_i = 0);
                          Fritsch–Carlson monotonicity holds when both lie in [0, 3]
    coefficients: [[4]]   piecewise: per-segment local coefficients (final stage)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.counting import Counted
from ..core.registry import ParamSpec, register
from ..core.types import Dataset, Result, Step, Vector

#: Number of points on the plotting grid in ``extra.eval``.
N_GRID = 200
#: converged requires max_i |p(x_i) - y_i| ≤ NODE_RTOL · max(max_i |y_i|, _TINY).
NODE_RTOL = 1e-6
#: Smallest normal float64, 2⁻¹⁰²²: below it the relative precision of y itself degrades
#: (gradual underflow), so the node test is relative to max(max|y|, _TINY).
_TINY = float(np.finfo(np.float64).tiny)


def _node_limit(values: Vector, data_scale: float | None = None) -> float:
    """NODE_RTOL·max(S, 2⁻¹⁰²²): the scale-invariant node-residual limit.

    S is the size of the interpolation data in units of y: max_i |y_i|, or ``data_scale``
    when the method interpolates more than values (the clamped spline's end slopes).

    # NOTE: the limit was NODE_RTOL·max(1, max|y|), an absolute 1e-6 for |y| < 1. The
    # Newton form on 70 equispaced nodes with y = s·cos(3x) misses the data by 14% of
    # max|y| and was reported converged=False for s = 1 but converged=True for s = 1e-6.
    """
    scale = float(np.max(np.abs(values))) if data_scale is None else data_scale
    return NODE_RTOL * max(scale, _TINY)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Data:
    x: Vector
    y: Vector
    f_true: Callable[[Any], Any] | None
    a: float
    b: float

    @property
    def n(self) -> int:
        return int(self.x.size)


def _resolve(problem: Dataset | tuple[Any, Any]) -> _Data:
    """Accept a :class:`Dataset` or an ``(x, y)`` pair; validate the data."""
    if isinstance(problem, Dataset):
        x_raw, y_raw, f_true, domain = problem.x, problem.y, problem.f_true, problem.domain
    elif isinstance(problem, tuple | list) and len(problem) == 2:
        x_raw, y_raw, f_true, domain = problem[0], problem[1], None, None
    else:
        raise TypeError("problem must be a numopt Dataset or an (x, y) pair of arrays")
    x = np.array(x_raw, dtype=np.float64, copy=True)
    y = np.array(y_raw, dtype=np.float64, copy=True)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(f"x and y must be 1-D of equal length; got {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("need at least one data point")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("data contain NaN or infinite values")
    if domain is not None:
        a, b = float(domain[0]), float(domain[1])
    else:
        a, b = float(np.min(x)), float(np.max(x))
    if not a < b:
        a, b = a - 1.0, b + 1.0
    return _Data(x, y, f_true, a, b)


def _quiet(fn: Callable[..., Result]) -> Callable[..., Result]:
    """Run a method with floating-point warnings off; overflow is reported in the Result."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Result:
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)

    return wrapper


def _require_distinct(x: Vector) -> None:
    s = np.sort(x)
    if s.size > 1 and np.any(np.diff(s) == 0.0):
        raise ValueError("interpolation nodes must be distinct")


def _sorted(data: _Data, min_points: int) -> tuple[Vector, Vector]:
    """Data sorted by x (stable), with distinct nodes and at least ``min_points`` points."""
    if data.n < min_points:
        raise ValueError(f"need at least {min_points} data points, got {data.n}")
    order = np.argsort(data.x, kind="stable")
    x, y = data.x[order], data.y[order]
    if np.any(np.diff(x) == 0.0):
        raise ValueError("interpolation nodes must be distinct")
    return x, y


def _sample(f: Callable[[Any], Any], pts: Vector) -> Vector:
    """Evaluate ``f`` point by point (works for scalar-only callables).

    A point where ``f`` raises a math error (e.g. ``math.log(0)``) gives NaN.
    """
    out = np.empty(pts.size, dtype=np.float64)
    for i, t in enumerate(pts):
        try:
            out[i] = float(f(float(t)))
        except (ValueError, ArithmeticError):
            out[i] = math.nan
    return out


class _Grid:
    """The plotting grid, f_true on it, and the max-error functional."""

    def __init__(self, data: _Data) -> None:
        self.t: Vector = np.linspace(data.a, data.b, N_GRID)
        self.truth: Vector | None = None
        if data.f_true is not None:
            with np.errstate(all="ignore"):
                truth = _sample(data.f_true, self.t)
            if np.all(np.isfinite(truth)):
                self.truth = truth

    def error(self, curve: Vector) -> float | None:
        if self.truth is None:
            return None
        if not np.all(np.isfinite(curve)):
            return math.inf
        return float(np.max(np.abs(curve - self.truth)))


def _finish(
    method: str,
    x_result: Any,
    *,
    kind: str,
    coefficients: Any,
    nodes: Vector,
    values: Vector,
    grid: _Grid,
    evaluate: Callable[[Vector], Vector],
    trace: list[Step],
    n_fev: int = 0,
    domain: tuple[float, float],
    more: dict[str, Any] | None = None,
    data_scale: float | None = None,
) -> Result:
    with np.errstate(all="ignore"):
        curve = evaluate(grid.t)
        at_nodes = evaluate(nodes)
    coef_arr = np.asarray(coefficients, dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(coef_arr))
        and np.all(np.isfinite(curve))
        and np.all(np.isfinite(at_nodes))
    )
    residual = float(np.max(np.abs(at_nodes - values))) if finite else math.inf
    limit = _node_limit(values, data_scale)
    ok = finite and residual <= limit
    err = grid.error(curve)
    if ok:
        msg = f"interpolant through {nodes.size} nodes built; max node residual {residual:.3g}"
    elif finite:
        msg = (
            f"the interpolant misses the data by {residual:.3g} at a node (> {limit:.3g}): "
            "round-off has destroyed this representation for these nodes"
        )
    else:
        msg = "non-finite coefficients or values (overflow): the interpolant is unusable"
    extra: dict[str, Any] = {
        "kind": kind,
        "coefficients": coefficients,
        "nodes": nodes,
        "values": values,
        "domain": [domain[0], domain[1]],
        "eval": {"x": grid.t, "y": curve, "f_true": grid.truth},
        "max_error": err,
        "node_residual": residual,
    }
    if more:
        extra.update(more)
    return Result(
        method,
        x_result,
        err,
        ok,
        msg,
        trace[-1].k,
        n_fev,
        trace=trace,
        extra=extra,
    )


def _broken(method: str, trace: list[Step], message: str, n_fev: int = 0) -> Result:
    """converged=False result after a numerical breakdown during construction."""
    last = trace[-1]
    return Result(method, last.x, None, False, message, last.k, n_fev, trace=trace)


# --------------------------------------------------------------------------------------
# Lagrange form
# --------------------------------------------------------------------------------------


def _lagrange_basis(nodes: Vector, j: int, t: Vector) -> Vector:
    """ℓ_j(t) = Π_{m≠j} (t - x_m)/(x_j - x_m), accumulated factor by factor."""
    out = np.ones_like(t)
    for m in range(nodes.size):
        if m != j:
            out = out * ((t - nodes[m]) / (nodes[j] - nodes[m]))
    return out


@register(
    id="lagrange",
    family="interpolation",
    name="Lagrange form",
    params=(),
    needs=("data",),
    order="error O(hⁿ), n nodes at spacing h: f − p = f⁽ⁿ⁾(ξ)·ω(x)/n!, ω(x) = ∏ⱼ(x − xⱼ)",
    summary="Write p as Σ y_j ℓ_j, where the basis polynomial ℓ_j is 1 at x_j and 0 at every other "
    "node; each evaluation costs O(n²).",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §3.1, Theorem 3.2",),
)
@_quiet
def lagrange(problem: Dataset | tuple[Any, Any]) -> Result:
    """Lagrange form of the interpolating polynomial.

    p(t) = Σ_{j=0}^{n-1} y_j ℓ_j(t), ℓ_j(t) = Π_{m≠j} (t - x_m)/(x_j - x_m)
    (Burden & Faires, §3.1, Theorem 3.2). The coefficients in the Lagrange basis are
    the data values, so ``Result.x = y``. Step k adds the term y_k ℓ_k; the partial sums
    are not interpolants until the last step, which is the point of the picture.
    Evaluation costs O(n²) per point; the barycentric form is the O(n) alternative.
    """
    data = _resolve(problem)
    _require_distinct(data.x)
    grid = _Grid(data)
    x, y = data.x, data.y
    curve = np.zeros_like(grid.t)
    trace: list[Step] = []
    for k in range(data.n):
        basis = _lagrange_basis(x, k, grid.t)
        curve = curve + y[k] * basis
        trace.append(
            Step(
                k,
                y[: k + 1].copy(),
                grid.error(curve),
                info={
                    "node_index": k,
                    "node": [x[k], y[k]],
                    "basis": basis,
                    "curve": curve,
                },
            )
        )

    def evaluate(t: Vector) -> Vector:
        total = np.zeros_like(t)
        for j in range(data.n):
            total = total + y[j] * _lagrange_basis(x, j, t)
        return total

    return _finish(
        "lagrange",
        y.copy(),
        kind="lagrange",
        coefficients=y.copy(),
        nodes=x,
        values=y,
        grid=grid,
        evaluate=evaluate,
        trace=trace,
        domain=(data.a, data.b),
    )


# --------------------------------------------------------------------------------------
# Barycentric form (second / "true" form)
# --------------------------------------------------------------------------------------


def _barycentric_eval(nodes: Vector, w: Vector, values: Vector, t: Vector) -> Vector:
    """Second barycentric formula (Berrut & Trefethen 2004, eq. (4.2)).

    p(t) = Σ_j [w_j/(t - x_j)] y_j / Σ_j [w_j/(t - x_j)], and p(x_j) = y_j exactly.
    """
    diff = t[:, None] - nodes[None, :]  # (T, n)
    exact = diff == 0.0
    c = w / np.where(exact, 1.0, diff)  # (T, n): w_j/(t - x_j)
    out = (c @ values) / np.sum(c, axis=1)
    hit = np.any(exact, axis=1)
    out[hit] = values[np.argmax(exact[hit], axis=1)]
    return out


@register(
    id="barycentric",
    family="interpolation",
    name="Barycentric form",
    params=(),
    needs=("data",),
    order="error O(hⁿ), n nodes at spacing h: f − p = f⁽ⁿ⁾(ξ)·ω(x)/n!, ω(x) = ∏ⱼ(x − xⱼ)",
    summary="Precompute one weight per node in O(n²); then evaluate p with a stable O(n) "
    "weighted average.",
    references=(
        "Berrut & Trefethen (2004), Barycentric Lagrange Interpolation, SIAM Review 46(3), "
        "eq. (3.2) weights, eq. (4.2) second form",
        "Higham (2004), The numerical stability of barycentric Lagrange interpolation, "
        "IMA J. Numer. Anal. 24",
    ),
)
@_quiet
def barycentric(problem: Dataset | tuple[Any, Any]) -> Result:
    """Barycentric Lagrange interpolation, second (true) form.

    Weights (Berrut & Trefethen 2004, eq. (3.2)): w_j = 1 / Π_{m≠j} (x_j - x_m).
    They are built incrementally (B&T §3): when node k joins, w_j ← w_j/(x_j - x_k) for
    j < k and w_k = 1/Π_{m<k}(x_k - x_m), so step k shows the interpolant through nodes
    0..k. Evaluation uses the second form, eq. (4.2), with p(x_j) = y_j returned exactly
    at a node. ``Result.x`` = the weights w (unscaled, exactly eq. (3.2)); a common
    factor would cancel in eq. (4.2), so overflow of the weights is the only failure mode
    and is reported as ``converged=False``.
    """
    data = _resolve(problem)
    _require_distinct(data.x)
    grid = _Grid(data)
    x, y = data.x, data.y
    w = np.empty(0, dtype=np.float64)
    trace: list[Step] = []
    for k in range(data.n):
        with np.errstate(all="ignore"):
            w = w / (x[:k] - x[k])
            w_k = 1.0 / float(np.prod(x[k] - x[:k])) if k else 1.0
        w = np.append(w, w_k)
        with np.errstate(all="ignore"):
            curve = _barycentric_eval(x[: k + 1], w, y[: k + 1], grid.t)
        trace.append(
            Step(
                k,
                w.copy(),
                grid.error(curve),
                info={"node_index": k, "node": [x[k], y[k]], "curve": curve},
            )
        )
        if not np.all(np.isfinite(w)) or np.any(w == 0.0):
            return _broken(
                "barycentric",
                trace,
                f"barycentric weights overflowed/underflowed at node {k}: rescale the data",
            )

    return _finish(
        "barycentric",
        w.copy(),
        kind="barycentric",
        coefficients=w.copy(),
        nodes=x,
        values=y,
        grid=grid,
        evaluate=lambda t: _barycentric_eval(x, w, y, t),
        trace=trace,
        domain=(data.a, data.b),
    )


# --------------------------------------------------------------------------------------
# Newton divided differences
# --------------------------------------------------------------------------------------


def _newton_eval(nodes: Vector, coef: Vector, t: Vector) -> Vector:
    """Nested (Horner) evaluation of the Newton form.

    p(t) = a_0 + (t - x_0)(a_1 + (t - x_1)(a_2 + ... + (t - x_{k-1}) a_k)).
    """
    v = np.full_like(t, coef[-1])
    for j in range(coef.size - 2, -1, -1):
        v = coef[j] + (t - nodes[j]) * v
    return v


@register(
    id="newton_divided_differences",
    family="interpolation",
    name="Newton divided differences",
    params=(),
    needs=("data",),
    order="error O(hⁿ), n nodes at spacing h: f − p = f⁽ⁿ⁾(ξ)·ω(x)/n!, ω(x) = ∏ⱼ(x − xⱼ)",
    summary="Add one node at a time; each new node adds one divided difference and one term "
    "(O(n²) table, O(n) per evaluation by nested multiplication).",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §3.3, Alg. 3.2",),
)
@_quiet
def newton_divided_differences(problem: Dataset | tuple[Any, Any]) -> Result:
    """Newton's divided-difference form.

    Table (Burden & Faires, Alg. 3.2): F_{i,0} = y_i and
    F_{i,j} = (F_{i,j-1} - F_{i-1,j-1}) / (x_i - x_{i-j}) = f[x_{i-j}, ..., x_i].
    Alg. 3.2 fills the table row by row, so row k needs only node k and row k-1: step k
    adds node k and its row, and the coefficient a_k = F_{k,k} = f[x_0, ..., x_k].
    p(t) = Σ_k a_k Π_{j<k} (t - x_j) (Newton form, B&F §3.3), evaluated by nested multiplication.
    ``Result.x`` = [a_0, ..., a_{n-1}].
    """
    data = _resolve(problem)
    _require_distinct(data.x)
    grid = _Grid(data)
    x, y = data.x, data.y
    prev = np.empty(0, dtype=np.float64)
    coef = np.empty(0, dtype=np.float64)
    trace: list[Step] = []
    for k in range(data.n):
        row = np.empty(k + 1, dtype=np.float64)
        row[0] = y[k]
        with np.errstate(all="ignore"):
            for j in range(1, k + 1):
                row[j] = (row[j - 1] - prev[j - 1]) / (x[k] - x[k - j])
        coef = np.append(coef, row[k])
        with np.errstate(all="ignore"):
            curve = _newton_eval(x[: k + 1], coef, grid.t)
        trace.append(
            Step(
                k,
                coef.copy(),
                grid.error(curve),
                info={"node_index": k, "node": [x[k], y[k]], "new_row": row, "curve": curve},
            )
        )
        if not np.all(np.isfinite(row)):
            return _broken(
                "newton_divided_differences",
                trace,
                f"non-finite divided difference at node {k} (overflow)",
            )
        prev = row

    return _finish(
        "newton_divided_differences",
        coef.copy(),
        kind="newton",
        coefficients=coef.copy(),
        nodes=x,
        values=y,
        grid=grid,
        evaluate=lambda t: _newton_eval(x, coef, t),
        trace=trace,
        domain=(data.a, data.b),
    )


# --------------------------------------------------------------------------------------
# Neville
# --------------------------------------------------------------------------------------


def _neville_columns(nodes: Vector, values: Vector, t: Vector) -> list[Vector]:
    """All columns of Neville's tableau, vectorized over the evaluation points ``t``.

    Returns ``cols`` with ``cols[j]`` of shape (n - j, len(t)): rows i = j..n-1 hold
    Q_{i,j}(t) = P_{i-j..i}(t), the interpolant through x_{i-j}, ..., x_i.
    """
    n = nodes.size
    q = np.repeat(values[:, None], t.size, axis=1)  # column 0: Q_{i,0} = y_i
    cols = [q.copy()]
    for j in range(1, n):
        # Rows i = j..n-1, using column j-1 (rows i and i-1); descending i keeps it in place.
        for i in range(n - 1, j - 1, -1):
            q[i] = ((t - nodes[i - j]) * q[i] - (t - nodes[i]) * q[i - 1]) / (
                nodes[i] - nodes[i - j]
            )
        cols.append(q[j:].copy())
    return cols


@register(
    id="neville",
    family="interpolation",
    name="Neville's algorithm",
    params=(
        ParamSpec(
            "x_frac",
            0.95,
            min=0.0,
            max=1.0,
            help="Evaluation point x* = a + x_frac·(b − a) inside the plotting domain [a, b].",
        ),
    ),
    needs=("data",),
    order="error O(hⁿ), n nodes at spacing h: f − p = f⁽ⁿ⁾(ξ)·ω(x)/n!, ω(x) = ∏ⱼ(x − xⱼ)",
    summary="Evaluate p(x*) directly by combining interpolants on ever larger sets of nodes "
    "(O(n²) per point).",
    references=("Burden & Faires, Numerical Analysis (10th ed.), §3.2, Alg. 3.1 (Theorem 3.5)",),
)
@_quiet
def neville(problem: Dataset | tuple[Any, Any], *, x_frac: float = 0.95) -> Result:
    """Neville's iterated interpolation at one point x*.

    Burden & Faires, Alg. 3.1: Q_{i,0} = y_i and, for j = 1..n-1, i = j..n-1,
    Q_{i,j} = [(x* - x_{i-j}) Q_{i,j-1} - (x* - x_i) Q_{i-1,j-1}] / (x_i - x_{i-j}),
    so Q_{i,j} = P_{i-j,...,i}(x*) (Theorem 3.5) and Q_{n-1,n-1} = p(x*).
    Step k shows column k; ``Step.x`` = Q_{k,k} = P_{0..k}(x*), the estimate that uses
    the first k+1 nodes, and ``info.curve`` runs the same tableau at every grid point.
    Neville has no coefficient vector: ``Result.x`` = p(x*) and
    ``extra.coefficients`` = [Q_{0,0}, ..., Q_{n-1,n-1}] (the diagonal estimates).
    Extra keys: ``x_eval``, ``value`` (= p(x*)), ``error_at_x_eval`` (|p(x*) - f(x*)| or None).
    """
    if not 0.0 <= x_frac <= 1.0:
        raise ValueError("x_frac must lie in [0, 1]")
    data = _resolve(problem)
    _require_distinct(data.x)
    grid = _Grid(data)
    x, y = data.x, data.y
    x_star = data.a + x_frac * (data.b - data.a)
    pts = np.concatenate(([x_star], grid.t))
    with np.errstate(all="ignore"):
        cols = _neville_columns(x, y, pts)
    trace: list[Step] = []
    diagonal: list[float] = []
    for k, col in enumerate(cols):
        estimate = float(col[0, 0])
        diagonal.append(estimate)
        curve = col[0, 1:]
        trace.append(
            Step(
                k,
                estimate,
                grid.error(curve),
                info={"column": col[:, 0], "x_eval": x_star, "curve": curve},
            )
        )
    value = diagonal[-1]
    err_at = None
    if data.f_true is not None and math.isfinite(value):
        f_star = float(_sample(data.f_true, np.array([x_star]))[0])
        err_at = abs(value - f_star) if math.isfinite(f_star) else None

    def evaluate(t: Vector) -> Vector:
        return _neville_columns(x, y, t)[-1][0]

    return _finish(
        "neville",
        value,
        kind="neville",
        coefficients=np.array(diagonal),
        nodes=x,
        values=y,
        grid=grid,
        evaluate=evaluate,
        trace=trace,
        domain=(data.a, data.b),
        more={"x_eval": x_star, "value": value, "error_at_x_eval": err_at},
    )


# --------------------------------------------------------------------------------------
# Piecewise polynomials
# --------------------------------------------------------------------------------------


def _pp_eval(breaks: Vector, coef: Vector, t: Vector) -> Vector:
    """Evaluate a piecewise cubic in local power form; the end pieces extrapolate.

    The piece index is i = clip(#{breaks ≤ t} - 1, 0, n-2), so t = x_0 uses piece 0 and
    t = x_{n-1} uses piece n-2.
    """
    idx = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, breaks.size - 2)
    dt = t - breaks[idx]
    a, b, c, d = coef[idx, 0], coef[idx, 1], coef[idx, 2], coef[idx, 3]
    return a + dt * (b + dt * (c + dt * d))


def _hermite_coefficients(x: Vector, y: Vector, s: Vector) -> Vector:
    """Local coefficients of the C¹ piecewise cubic with values y and slopes s.

    On [x_i, x_{i+1}] with h_i = x_{i+1} - x_i, m_i = (y_{i+1} - y_i)/h_i:
    a = y_i, b = s_i, c = (3m_i - 2s_i - s_{i+1})/h_i, d = (s_i + s_{i+1} - 2m_i)/h_i²
    (Hermite form; de Boor, A Practical Guide to Splines, rev. ed. 2001, Ch. IV).
    """
    h = np.diff(x)
    m = np.diff(y) / h
    coef = np.empty((x.size - 1, 4), dtype=np.float64)
    coef[:, 0] = y[:-1]
    coef[:, 1] = s[:-1]
    coef[:, 2] = (3.0 * m - 2.0 * s[:-1] - s[1:]) / h
    coef[:, 3] = (s[:-1] + s[1:] - 2.0 * m) / (h * h)
    return coef


def _finish_piecewise(
    method: str,
    data: _Data,
    xs: Vector,
    ys: Vector,
    coef: Vector,
    trace: list[Step],
    grid: _Grid,
    k: int,
    info: dict[str, Any],
    more: dict[str, Any] | None = None,
    data_scale: float | None = None,
) -> Result:
    with np.errstate(all="ignore"):
        curve = _pp_eval(xs, coef, grid.t)
    trace.append(
        Step(
            k,
            coef.reshape(-1).copy(),
            grid.error(curve),
            info={"stage": "coefficients", **info, "coefficients": coef, "curve": curve},
        )
    )
    return _finish(
        method,
        coef.reshape(-1).copy(),
        kind="piecewise_cubic",
        coefficients=coef,
        nodes=xs,
        values=ys,
        grid=grid,
        evaluate=lambda t: _pp_eval(xs, coef, t),
        trace=trace,
        domain=(data.a, data.b),
        more=more,
        data_scale=data_scale,
    )


@register(
    id="linear_spline",
    family="interpolation",
    name="Linear spline",
    params=(),
    needs=("data",),
    order="error O(h²) for C² data",
    summary="Join consecutive data points with straight lines.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §3.5 (piecewise-linear interpolation)",
    ),
)
@_quiet
def linear_spline(problem: Dataset | tuple[Any, Any]) -> Result:
    """Piecewise-linear interpolation.

    S_i(t) = y_i + m_i (t - x_i) on [x_i, x_{i+1}], m_i = (y_{i+1} - y_i)/h_i
    (Burden & Faires, §3.5). One step (k = 0); coefficients use the common 4-column
    layout [a, b, c, d] = [y_i, m_i, 0, 0]. The data are sorted by x; at least 2 points.
    """
    data = _resolve(problem)
    xs, ys = _sorted(data, 2)
    grid = _Grid(data)
    h = np.diff(xs)
    m = np.diff(ys) / h
    coef = np.zeros((xs.size - 1, 4), dtype=np.float64)
    coef[:, 0] = ys[:-1]
    coef[:, 1] = m
    return _finish_piecewise(
        "linear_spline", data, xs, ys, coef, [], grid, 0, {"h": h, "secants": m}
    )


def _thomas(
    lower: Vector, diag: Vector, upper: Vector, rhs: Vector
) -> tuple[Vector, Vector, Vector, Vector | None]:
    """Thomas algorithm: tridiagonal LU without pivoting (Golub & Van Loan, 4th ed., §4.3).

    Returns (multipliers, pivots, modified rhs, solution); the solution is ``None`` when a
    pivot is zero or non-finite. Safe without pivoting for the spline systems here: the
    natural/clamped matrices are strictly diagonally dominant by rows, and the
    not-a-knot first row is eliminated with multiplier exactly 1 (see the spline docstring).
    """
    n = diag.size
    piv = diag.copy()
    z = rhs.copy()
    mult = np.zeros(max(n - 1, 0), dtype=np.float64)
    for i in range(1, n):
        if piv[i - 1] == 0.0 or not math.isfinite(piv[i - 1]):
            return mult, piv, z, None
        mult[i - 1] = lower[i - 1] / piv[i - 1]
        piv[i] = diag[i] - mult[i - 1] * upper[i - 1]
        z[i] = rhs[i] - mult[i - 1] * z[i - 1]
    if piv[n - 1] == 0.0 or not math.isfinite(piv[n - 1]):
        return mult, piv, z, None
    s = np.empty(n, dtype=np.float64)
    s[n - 1] = z[n - 1] / piv[n - 1]
    for i in range(n - 2, -1, -1):
        s[i] = (z[i] - upper[i] * s[i + 1]) / piv[i]
    return mult, piv, z, s


def _spline_system(
    xs: Vector, ys: Vector, bc: str, fprime_a: float, fprime_b: float
) -> tuple[Vector, Vector, Vector, Vector, str]:
    """Tridiagonal system T s = r for the node slopes s_i = S'(x_i).

    Interior rows i = 1..n-2 (C² continuity; de Boor 2001, Ch. IV):
        h_i s_{i-1} + 2(h_{i-1} + h_i) s_i + h_{i-1} s_{i+1} = 3(h_i m_{i-1} + h_{i-1} m_i).
    End rows:
        natural      S''(x_0) = 0:     2 s_0 + s_1 = 3 m_0;  s_{n-2} + 2 s_{n-1} = 3 m_{n-2}
        clamped      S'(x_0) = f'_a:   s_0 = f'_a;           s_{n-1} = f'_b
        not-a-knot   S''' continuous at x_1 and x_{n-2}; combined with the first/last
                     interior row (de Boor 2001, Ch. IV, CUBSPL):
                     h_1 s_0 + (h_0+h_1) s_1 = [h_1(3h_0 + 2h_1) m_0 + h_0² m_1]/(h_0+h_1),
                     mirrored at the right end.
    Special cases (as in SciPy's CubicSpline): not-a-knot with n = 2 is the straight line
    (s_0 = s_1 = m_0); with n = 3 both conditions coincide and the spline is the
    interpolating parabola, written as s_0 + s_1 = 2m_0, interior row, s_1 + s_2 = 2m_1.
    """
    n = xs.size
    h = np.diff(xs)
    m = np.diff(ys) / h
    lower = np.zeros(n - 1, dtype=np.float64)
    diag = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n - 1, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        lower[i - 1] = h[i]
        diag[i] = 2.0 * (h[i - 1] + h[i])
        upper[i] = h[i - 1]
        rhs[i] = 3.0 * (h[i] * m[i - 1] + h[i - 1] * m[i])
    label = bc
    if bc == "natural":
        diag[0], upper[0], rhs[0] = 2.0, 1.0, 3.0 * m[0]
        lower[n - 2], diag[n - 1], rhs[n - 1] = 1.0, 2.0, 3.0 * m[n - 2]
    elif bc == "clamped":
        diag[0], upper[0], rhs[0] = 1.0, 0.0, fprime_a
        lower[n - 2], diag[n - 1], rhs[n - 1] = 0.0, 1.0, fprime_b
        label = f"clamped (S'(x_0) = {fprime_a:g}, S'(x_n) = {fprime_b:g})"
    elif n == 2:  # not-a-knot, two points: the line
        diag[0], upper[0], rhs[0] = 1.0, 0.0, m[0]
        lower[0], diag[1], rhs[1] = 0.0, 1.0, m[0]
        label = "not-a-knot (n = 2: straight line)"
    elif n == 3:  # not-a-knot, three points: the parabola
        diag[0], upper[0], rhs[0] = 1.0, 1.0, 2.0 * m[0]
        lower[1], diag[2], rhs[2] = 1.0, 1.0, 2.0 * m[1]
        label = "not-a-knot (n = 3: parabola)"
    else:
        h0, h1 = h[0], h[1]
        diag[0], upper[0] = h1, h0 + h1
        rhs[0] = (h1 * (3.0 * h0 + 2.0 * h1) * m[0] + h0 * h0 * m[1]) / (h0 + h1)
        g0, g1 = h[n - 2], h[n - 3]
        lower[n - 2], diag[n - 1] = g0 + g1, g1
        rhs[n - 1] = (g1 * (3.0 * g0 + 2.0 * g1) * m[n - 2] + g0 * g0 * m[n - 3]) / (g0 + g1)
    return lower, diag, upper, rhs, label


def _cubic_spline(
    method: str,
    problem: Dataset | tuple[Any, Any],
    bc: str,
    fprime_a: float = 0.0,
    fprime_b: float = 0.0,
) -> Result:
    data = _resolve(problem)
    xs, ys = _sorted(data, 2)
    grid = _Grid(data)
    h = np.diff(xs)
    m = np.diff(ys) / h
    lower, diag, upper, rhs, label = _spline_system(xs, ys, bc, fprime_a, fprime_b)
    trace = [
        Step(
            0,
            rhs.copy(),
            None,
            info={
                "stage": "assemble",
                "boundary": label,
                "h": h,
                "secants": m,
                "lower": lower,
                "diag": diag,
                "upper": upper,
                "rhs": rhs,
            },
        )
    ]
    with np.errstate(all="ignore"):
        mult, piv, z, s = _thomas(lower, diag, upper, rhs)
    trace.append(
        Step(
            1,
            z.copy(),
            None,
            info={"stage": "forward_sweep", "pivots": piv, "multipliers": mult, "rhs": z},
        )
    )
    if s is None or not np.all(np.isfinite(s)):
        return _broken(method, trace, "zero or non-finite pivot in the tridiagonal solve")
    trace.append(Step(2, s.copy(), None, info={"stage": "back_substitution", "slopes": s}))
    coef = _hermite_coefficients(xs, ys, s)
    data_scale = None
    if bc == "clamped":
        # The end slopes are interpolation data too: in units of y they are worth
        # |f'|·h. With y = 0 and f'_a = 20 the spline is not 0, and the one Horner
        # evaluation at the last node leaves a rounding residual (5.7e-18) that no
        # limit relative to max|y| = 0 can accept.
        slope_size = max(abs(fprime_a), abs(fprime_b)) * float(np.max(h))
        data_scale = max(float(np.max(np.abs(ys))), slope_size)
    return _finish_piecewise(
        method,
        data,
        xs,
        ys,
        coef,
        trace,
        grid,
        3,
        {"slopes": s},
        more={"boundary": label, "slopes": s},
        data_scale=data_scale,
    )


_SPLINE_REFS = (
    "de Boor, A Practical Guide to Splines (rev. ed. 2001), Ch. IV (slope form, CUBSPL)",
    "Burden & Faires, Numerical Analysis (10th ed.), §3.5, Theorems 3.11–3.12",
)


@register(
    id="cubic_spline_natural",
    family="interpolation",
    name="Natural cubic spline",
    params=(),
    needs=("data",),
    order="error O(h²) at the ends, O(h⁴) inside",
    summary="The C² piecewise cubic through the data with zero curvature at both ends.",
    references=(*_SPLINE_REFS, "Burden & Faires (10th ed.), Alg. 3.4 (same spline, c-form)"),
)
@_quiet
def cubic_spline_natural(problem: Dataset | tuple[Any, Any]) -> Result:
    """Natural cubic spline: S''(x_0) = S''(x_{n-1}) = 0 (Burden & Faires, Alg. 3.4).

    Unknowns are the node slopes s_i = S'(x_i) (de Boor's slope form); see
    :func:`_spline_system` for every row. Stages (one Step each): k=0 ``assemble``
    (``Step.x`` = right-hand side r), k=1 ``forward_sweep`` of the Thomas algorithm
    (``Step.x`` = modified rhs), k=2 ``back_substitution`` (``Step.x`` = slopes s),
    k=3 ``coefficients`` (``Step.x`` = flattened [[a, b, c, d]] per segment, from
    :func:`_hermite_coefficients`). The data are sorted by x; at least 2 points.
    Extra keys: ``boundary`` (str), ``slopes`` ([n]).
    """
    return _cubic_spline("cubic_spline_natural", problem, "natural")


@register(
    id="cubic_spline_clamped",
    family="interpolation",
    name="Clamped cubic spline",
    params=(
        ParamSpec("fprime_a", 0.0, min=-50.0, max=50.0, help="Imposed end slope S′(x₀)."),
        ParamSpec("fprime_b", 0.0, min=-50.0, max=50.0, help="Imposed end slope S′(xₙ₋₁)."),
    ),
    needs=("data",),
    order="O(h⁴) with exact end slopes",
    summary="The C² piecewise cubic through the data with prescribed slopes at both ends.",
    references=(*_SPLINE_REFS, "Burden & Faires (10th ed.), Alg. 3.5 (same spline, c-form)"),
)
@_quiet
def cubic_spline_clamped(
    problem: Dataset | tuple[Any, Any], *, fprime_a: float = 0.0, fprime_b: float = 0.0
) -> Result:
    """Clamped cubic spline: S'(x_0) = fprime_a, S'(x_{n-1}) = fprime_b (B&F, Alg. 3.5).

    The O(h⁴) accuracy needs the true end slopes; the defaults (0) are horizontal
    tangents. Same slope form and stages as :func:`cubic_spline_natural`: k=0
    ``assemble``, k=1 ``forward_sweep``, k=2 ``back_substitution``, k=3 ``coefficients``.
    Extra keys: ``boundary`` (str), ``slopes`` ([n]).
    """
    if not (math.isfinite(fprime_a) and math.isfinite(fprime_b)):
        raise ValueError("end slopes must be finite")
    return _cubic_spline("cubic_spline_clamped", problem, "clamped", fprime_a, fprime_b)


@register(
    id="cubic_spline_not_a_knot",
    family="interpolation",
    name="Not-a-knot cubic spline",
    params=(),
    needs=("data",),
    order="O(h⁴)",
    summary="The C² cubic spline whose first two and last two pieces are single cubics.",
    references=_SPLINE_REFS,
)
@_quiet
def cubic_spline_not_a_knot(problem: Dataset | tuple[Any, Any]) -> Result:
    """Not-a-knot cubic spline: S''' is continuous at x_1 and x_{n-2} (de Boor 2001, Ch. IV).

    Elimination without pivoting is stable here: eliminating s_0 from row 1 uses the
    multiplier h_1/h_1 = 1 and leaves the pivot h_0 + h_1 > 0; the later rows are
    diagonally dominant and the last pivot stays positive. Same slope form and stages
    as :func:`cubic_spline_natural`. Extra keys: ``boundary`` (str), ``slopes`` ([n]).
    """
    return _cubic_spline("cubic_spline_not_a_knot", problem, "not-a-knot")


# --------------------------------------------------------------------------------------
# PCHIP
# --------------------------------------------------------------------------------------


def _pchip_end_slope(h0: float, h1: float, m0: float, m1: float) -> tuple[float, bool]:
    """One-sided three-point end slope with shape limiting (Moler, NCM Ch. 3, pchiptx).

    d = ((2h_0 + h_1) m_0 - h_0 m_1)/(h_0 + h_1); d = 0 if sign(d) ≠ sign(m_0);
    d = 3 m_0 if sign(m_0) ≠ sign(m_1) and |d| > 3|m_0|. Returns (d, limited).
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0, True
    if np.sign(m0) != np.sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0, True
    return d, False


@register(
    id="pchip",
    family="interpolation",
    name="PCHIP (monotone cubic)",
    params=(),
    needs=("data",),
    order="O(h³) on smooth monotone data",
    summary="A C¹ piecewise cubic whose slopes are chosen so monotone data stay monotone.",
    references=(
        "Fritsch & Carlson (1980), Monotone piecewise cubic interpolation, SIAM J. Numer. "
        "Anal. 17(2) (monotonicity region)",
        "Fritsch & Butland (1984), A method for constructing local monotone piecewise cubic "
        "interpolants, SIAM J. Sci. Stat. Comput. 5(2) (weighted harmonic mean)",
        "Moler, Numerical Computing with MATLAB (2004), Ch. 3 (pchiptx end slopes)",
    ),
)
@_quiet
def pchip(problem: Dataset | tuple[Any, Any]) -> Result:
    """Piecewise cubic Hermite interpolating polynomial (PCHIP).

    Interior slopes (Fritsch & Butland 1984): s_k = 0 if m_{k-1}·m_k ≤ 0 (a local
    extremum or a flat segment), else the weighted harmonic mean
    (w_1 + w_2)/s_k = w_1/m_{k-1} + w_2/m_k, w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}.
    End slopes: :func:`_pchip_end_slope`. With n = 2 both slopes equal m_0 (a line).
    These slopes satisfy 0 ≤ α_k, β_k ≤ 3 (α_k = s_k/m_k, β_k = s_{k+1}/m_k), which is
    inside the Fritsch–Carlson monotonicity region, so the interpolant is monotone on
    every interval where the data are.

    # NOTE: the slope rule is the Fritsch–Butland (1984) harmonic mean used by SLATEC
    # PCHIM, MATLAB pchip and SciPy PchipInterpolator, not the original two-pass
    # Fritsch–Carlson (1980) slope modification; both stay in the FC80 monotonicity region.

    Stages: k=0 ``secants`` (``Step.x`` = m), k=1 ``slopes`` (``Step.x`` = s),
    k=2 ``coefficients`` (``Step.x`` = flattened per-segment coefficients).
    Extra key: ``slopes``.
    """
    data = _resolve(problem)
    xs, ys = _sorted(data, 2)
    grid = _Grid(data)
    n = xs.size
    h = np.diff(xs)
    m = np.diff(ys) / h
    trace = [Step(0, m.copy(), None, info={"stage": "secants", "h": h, "secants": m})]
    s = np.empty(n, dtype=np.float64)
    limited = [False] * n
    if n == 2:
        s[:] = m[0]
    else:
        for k in range(1, n - 1):
            if m[k - 1] == 0.0 or m[k] == 0.0 or np.sign(m[k - 1]) != np.sign(m[k]):
                s[k] = 0.0
                limited[k] = True
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                s[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
        s[0], limited[0] = _pchip_end_slope(h[0], h[1], m[0], m[1])
        s[n - 1], limited[n - 1] = _pchip_end_slope(h[n - 2], h[n - 3], m[n - 2], m[n - 3])
    alpha = [float(s[i] / m[i]) if m[i] != 0.0 else None for i in range(n - 1)]
    beta = [float(s[i + 1] / m[i]) if m[i] != 0.0 else None for i in range(n - 1)]
    trace.append(
        Step(
            1,
            s.copy(),
            None,
            info={"stage": "slopes", "slopes": s, "limited": limited, "alpha": alpha, "beta": beta},
        )
    )
    coef = _hermite_coefficients(xs, ys, s)
    return _finish_piecewise(
        "pchip", data, xs, ys, coef, trace, grid, 2, {"slopes": s}, more={"slopes": s}
    )


# --------------------------------------------------------------------------------------
# Chebyshev interpolation
# --------------------------------------------------------------------------------------


def _clenshaw(coef: Vector, u: Vector) -> Vector:
    """Σ_k c_k T_k(u) by Clenshaw's recurrence (Numerical Recipes, 3rd ed., §5.4 and §5.8).

    b_{N+1} = b_{N+2} = 0, b_k = c_k + 2u b_{k+1} - b_{k+2} (k = N..1), p = c_0 + u b_1 - b_2.
    """
    b1 = np.zeros_like(u)
    b2 = np.zeros_like(u)
    for k in range(coef.size - 1, 0, -1):
        b1, b2 = coef[k] + 2.0 * u * b1 - b2, b1
    return coef[0] + u * b1 - b2


def _to_unit(t: Vector, a: float, b: float) -> Vector:
    """Affine map [a, b] → [-1, 1]: u = (2t - (a + b))/(b - a)."""
    return (2.0 * t - (a + b)) / (b - a)


@register(
    id="chebyshev_interpolation",
    family="interpolation",
    name="Chebyshev interpolation",
    params=(
        ParamSpec(
            "n_nodes",
            0,
            kind="int",
            min=0,
            max=200,
            help="Number of Chebyshev nodes when the data are samples of f_true (0 = the "
            "dataset size). Ignored otherwise (no f_true, or noisy data): the data points "
            "are the nodes.",
        ),
    ),
    needs=("data",),
    order="geometric for analytic f",
    summary="Sample f at the Chebyshev nodes and expand the interpolant in Chebyshev polynomials.",
    references=(
        "Burden & Faires, Numerical Analysis (10th ed.), §8.3 (Chebyshev nodes)",
        "Press et al., Numerical Recipes (3rd ed.), §5.8 (coefficients by discrete "
        "orthogonality) and §5.4 (Clenshaw)",
        "Trefethen, Approximation Theory and Approximation Practice (2013), Ch. 3–4",
    ),
)
@_quiet
def chebyshev_interpolation(problem: Dataset | tuple[Any, Any], *, n_nodes: int = 0) -> Result:
    """Interpolant in the Chebyshev basis, p(t) = Σ_{k=0}^{N-1} c_k T_k(u), u ∈ [-1, 1].

    Resampling mode, used when the data are samples of ``f_true``
    (max_i |f_true(x_i) - y_i| ≤ ``NODE_RTOL``·max(max_i |y_i|, 2⁻¹⁰²²), the tolerance of
    the node test): N = ``n_nodes`` (or the dataset size), nodes u_j = cos(π(j + ½)/N),
    the roots of T_N, mapped affinely to the domain [a, b]; by the discrete orthogonality
    of T_0..T_{N-1} at these nodes, c_k = (2/N) Σ_j f(x_j) cos(πk(j + ½)/N), with c_0
    halved (Numerical Recipes, §5.8).
    Data mode, used otherwise (no ``f_true``, or noisy data such as ``noisy_linear``,
    whose ``f_true`` is the hidden noise-free curve): the nodes are the data, and c solves
    the Chebyshev–Vandermonde system T c = y with T_{jk} = T_k(u_j) (LU with partial
    pivoting), so p goes through the data like every other interpolant.

    # NOTE: the data-mode fallback is not in the textbook method (which needs f at the
    # Chebyshev nodes); it gives the same polynomial as the other polynomial methods,
    # expressed in the Chebyshev basis.
    # NOTE: resampling used to run whenever f_true existed. On the noisy datasets it then
    # interpolated the hidden mean curve: max error 9e-15 against f_true while the curve
    # missed the user's data by up to 0.99 (noisy_linear), and every other interpolant
    # went through the data. The sample check costs n evaluations of f_true.

    Step k adds the term c_k T_k: ``Step.x`` = [c_0, ..., c_k] and ``info.curve`` is the
    truncated series (its convergence shows the coefficient decay). Evaluation by
    Clenshaw's recurrence. ``n_fev`` = number of f_true calls: n for the sample check
    (when f_true exists) plus N in resampling mode. Extra keys: ``source``
    ("f_true" | "data"), ``sample_deviation`` (max_i |f_true(x_i) - y_i|, ∞ if f_true is
    not finite at a data point, None without f_true); ``domain`` gives the map
    u = (2t - a - b)/(b - a). In data mode a nonzero ``n_nodes`` that differs from the
    number of data points is ignored, and the Result message says so.
    """
    if n_nodes < 0:
        raise ValueError("n_nodes must be ≥ 0")
    data = _resolve(problem)
    grid = _Grid(data)
    a, b = data.a, data.b
    n_fev = 0
    deviation: float | None = None
    f = Counted(data.f_true) if data.f_true is not None else None
    if f is not None:
        with np.errstate(all="ignore"):
            f_data = _sample(f, data.x)
        diff = np.abs(f_data - data.y)
        deviation = float(np.max(diff)) if np.all(np.isfinite(diff)) else math.inf
        n_fev = f.n
    if f is not None and deviation is not None and deviation <= _node_limit(data.y):
        source = "f_true"
        n = int(n_nodes) if n_nodes > 0 else data.n
        jj = np.arange(n, dtype=np.float64)
        theta = np.pi * (jj + 0.5) / n  # u_j = cos θ_j, descending in j
        u_nodes = np.cos(theta)
        nodes_desc = 0.5 * (a + b) + 0.5 * (b - a) * u_nodes
        with np.errstate(all="ignore"):
            f_nodes = _sample(f, nodes_desc)
        n_fev = f.n
        if not np.all(np.isfinite(f_nodes)):
            step = Step(0, np.zeros(1), None, info={"term_index": 0, "curve": np.zeros(N_GRID)})
            return _broken(
                "chebyshev_interpolation",
                [step],
                "f_true is not finite at a Chebyshev node",
                n_fev,
            )
        coef = np.empty(n, dtype=np.float64)
        for k in range(n):
            coef[k] = (2.0 / n) * float(np.sum(f_nodes * np.cos(k * theta)))
        coef[0] *= 0.5
        nodes, values = nodes_desc[::-1].copy(), f_nodes[::-1].copy()
    else:
        source = "data"
        _require_distinct(data.x)
        n = data.n
        u_data = _to_unit(data.x, a, b)
        vander = np.empty((n, n), dtype=np.float64)
        vander[:, 0] = 1.0
        if n > 1:
            vander[:, 1] = u_data
        for k in range(2, n):  # T_{k} = 2u T_{k-1} - T_{k-2}
            vander[:, k] = 2.0 * u_data * vander[:, k - 1] - vander[:, k - 2]
        try:
            coef = np.linalg.solve(vander, data.y)
        except np.linalg.LinAlgError:
            step = Step(0, np.zeros(1), None, info={"term_index": 0, "curve": np.zeros(N_GRID)})
            return _broken(
                "chebyshev_interpolation",
                [step],
                "singular Chebyshev–Vandermonde matrix",
                n_fev,
            )
        nodes, values = data.x, data.y

    u_grid = _to_unit(grid.t, a, b)
    trace: list[Step] = []
    for k in range(n):
        with np.errstate(all="ignore"):
            curve = _clenshaw(coef[: k + 1], u_grid)
        trace.append(
            Step(k, coef[: k + 1].copy(), grid.error(curve), info={"term_index": k, "curve": curve})
        )
    result = _finish(
        "chebyshev_interpolation",
        coef.copy(),
        kind="chebyshev",
        coefficients=coef.copy(),
        nodes=nodes,
        values=values,
        grid=grid,
        evaluate=lambda t: _clenshaw(coef, _to_unit(t, a, b)),
        trace=trace,
        n_fev=n_fev,
        domain=(a, b),
        more={"source": source, "sample_deviation": deviation},
    )
    if source == "data" and deviation is not None:
        result.message += (
            f"; the data are not samples of f_true (max |f_true(x_i) - y_i| = "
            f"{deviation:.3g}), so the data points are the nodes"
        )
    if source == "data" and n_nodes > 0 and int(n_nodes) != n:
        why = "without f_true" if deviation is None else "in data mode"
        result.message += (
            f"; n_nodes={int(n_nodes)} ignored: {why} the {n} data points are the nodes"
        )
    return result


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("lagrange", "runge_equispaced", {}),
    ("barycentric", "runge_chebyshev", {}),
    ("newton_divided_differences", "sine_samples", {}),
    ("neville", "runge_equispaced", {"x_frac": 0.95}),
    ("linear_spline", "step_data", {}),
    ("cubic_spline_natural", "sine_samples", {}),
    ("cubic_spline_clamped", "sine_samples", {"fprime_a": 1.0, "fprime_b": 1.0}),
    ("cubic_spline_not_a_knot", "runge_equispaced", {}),
    ("pchip", "step_data", {}),
    ("chebyshev_interpolation", "runge_equispaced", {}),
]
