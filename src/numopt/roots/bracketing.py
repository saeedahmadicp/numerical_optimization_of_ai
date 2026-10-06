"""Bracketing root finders for f(x) = 0 on an interval [a, b] with f(a)·f(b) < 0.

Bracketing methods keep a sign change inside the current interval, so they cannot
diverge. All methods here share these conventions:

* **Start.** ``bracket=(a, b)`` (or the problem's default) with finite ``a < b`` whose
  width b − a does not overflow, f finite at both ends and of opposite signs, and
  ``xtol ≥ 0`` (``xtol > 0`` for ITP); else ``ValueError``. If f(a) or f(b) is exactly
  zero the method returns that end at once.
* **Trace.** Step ``k`` holds the k-th *newly evaluated* estimate ``x_k`` and ``f(x_k)``.
  ``k = 0`` is the first estimate computed from the initial bracket (the first midpoint,
  chord point, ...). ``info["bracket"]`` is the bracket that ``x_k`` was computed from
  (it contains ``x_k`` and a root); ``info["new_bracket"]`` is the bracket after the sign
  test on ``f(x_k)``. ``n_iter == trace[-1].k``.
* **Tolerance.** ``tol(x) = xtol + 2ε|x|`` (Brent 1973, ch. 4), where ε is the machine
  epsilon. The 2ε|x| term makes the test reachable for any root magnitude.
* **Stopping test (converged).** f(x_k) == 0, or |f(x_k)| ≤ ``ftol``, or the new bracket
  has half-width ≤ tol (strictly < for Chandrupatla; for bisection the bracket that
  contains the midpoint x_k), so the root is within 2·tol of the returned x. ITP uses its
  own test (b − a ≤ 2·xtol, see :func:`itp`). No other test ends a run with
  ``converged=True``: every converged run has a sign change within 2·tol of x.
* **Verified step test (false-position family and Ridders).** Their iterates often
  approach the root from one side while the far end of the bracket stays put, so the
  bracket can stay wide although x_k is already accurate. The a posteriori estimate
  |x_k − x_{k−1}|·max(1, ρ/(1 − ρ)) with ρ = |x_k − x_{k−1}|/|x_{k−1} − x_{k−2}| < 1
  (:func:`step_estimate`), or x_k = x_{k−1}, is therefore only a *trigger*: when it is
  ≤ tol, the next point is the probe x_k ± tol toward the far end of the bracket. If f
  changes sign across the probe, the bracket shrinks to width tol and the bracket test
  passes. If it does not, the estimate was misled (one ratio of two irregular steps, a
  slow one-sided crawl, or a chord point that rounds onto a bracket end), and the run
  goes on: the false-position methods bisect once, Ridders takes its own step (which at
  least halves the bracket). This is the minimum step "never step less than tol" of
  Dekker (1969) and Brent (1973, §4.2), used only when the step test claims convergence.
  NOTE: Ford (1995) Alg. 1 and NR zriddr accept the step test without this check; on
  steep or multiple roots that returns points far from the root (e.g. eˣ − 10 on
  (−50, 60) gave x = −50 with ``converged=True``).
* **Failure.** max_iter reached or a non-finite f(x_k) gives ``converged=False``. An
  exception of floating-point origin in f (``OverflowError`` from ``math.exp`` or float
  ``**``, ``ValueError`` from a math domain error, ``ZeroDivisionError``) counts as
  f = nan (:class:`SafeScalar`), so the run stops with a message instead of raising.

Info keys (every method, every step):
    bracket: [lo, hi]       the sign-change interval x_k was computed from (lo < hi).
    new_bracket: [lo, hi]   the sign-change interval after the sign test on f(x_k)
                            ([x_k, x_k] when f(x_k) == 0).
A run that ends before any new point is computed (f == 0 at a bracket end, or an initial
bracket already within tolerance for brent/itp) has a single step 0 at that end whose info
holds only ``bracket``, ``new_bracket`` (and ``best`` for brent/itp). A step that ends the
run on a non-finite f carries ``bracket`` and whatever geometry was already computed.

Additional info keys per method:
    bisection:
        a, b: float          ends of ``bracket``; fa, fb: f at those ends.
    regula_falsi, illinois, pegasus, anderson_bjorck:
        step: "chord" | "verify" | "bisection"   how x_k was produced: the method's chord,
                             the probe x_{k−1} ± tol of a verified step test, or the
                             bisection that follows a failed probe.
        chord: [[x, y], [x, y]] | null  the two chord end points whose line crosses zero at
                             x_k; the first ordinate is the (possibly scaled) value of the
                             retained end, so the drawn chord is the one actually used
                             (null for verify and bisection steps).
        scale: float         factor m applied to the retained end's ordinate after this
                             step (1.0 = no modification; Illinois 0.5; Pegasus
                             f_k/(f_k + f_{k+1}); Anderson–Björck 1 − f_{k+1}/f_k or 0.5;
                             always 1.0 after verify and bisection steps).
        estimate: float | null  the step-test value |x_k − x_{k−1}|·max(1, ρ/(1 − ρ)) over
                             the chord points since the last verify/bisection step (0.0 when
                             x_k = x_{k−1}; null with fewer than two such points, when ρ ≥ 1,
                             and on verify/bisection steps). A value ≤ tol triggers a probe.
    ridders:
        step: "ridders" | "verify"   Ridders' step, or the probe x_{k−1} ± tol of a
                             verified step test.
        midpoint: float | null  m = (lo + hi)/2;  f_mid: f(m)  (null on verify steps).
        exp_factor: float | null  Q = e^{λ(m − lo)} with f(lo) − 2f(m)Q + f(hi)Q² = 0 (null if
                             f(m) = 0 and on verify steps).
        transformed: [[x, y]×3] | null  (lo, f(lo)), (m, f(m)Q), (hi, f(hi)Q²): collinear
                             points whose line crosses zero at x_k (null if f(m) = 0 and on
                             verify steps).
        estimate: float | null  as for the false-position family, over consecutive Ridders
                             points.
    brent:
        step: "bisection" | "secant" | "inverse_quadratic"   how x_k was produced.
        attempted: "secant" | "inverse_quadratic" | null     interpolation tried this step
                             (differs from ``step`` when Brent rejected it and bisected).
        points: [[x, f(x)]...]  interpolation points of ``attempted`` (2 for secant: a, b;
                             3 for inverse quadratic: a, b, c); [] when none was tried.
        tol: float           Brent's tol = 2ε|b| + xtol at this step (minimum step length).
        best: float          b after the update (the point with the smaller |f| of the
                             new bracket ends; Brent's current estimate).
    chandrupatla:
        step: "bisection" | "inverse_quadratic";  t: float  x_k = x₁ + t(x₂ − x₁).
        points: [[x, f(x)]×3]  (x₁, x₂, x₃) used for the inverse quadratic ([] for bisection).
        xi, phi: float | null  Chandrupatla's validity test 1 − √(1−ξ) < Φ < √ξ.
        best: float          the bracket end with the smaller |f| after the update.
    itp:
        x_half: float        bisection point;  x_f: regula falsi point;  x_t: truncated point.
        delta: float         truncation size max(κ₁(b − a)^κ₂, xtol/2);  r: projection radius
                             (reduced by a few ulps to absorb rounding, see :func:`itp`).
        projected: bool      True when x_t lay outside the radius r and was projected.
        best: float          the bracket end with the smaller |f| after the update.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from ..core.counting import Counted, finite, scalar_problem
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step

#: Machine epsilon of float64.
EPS = sys.float_info.epsilon

COMMON_PARAMS = (
    ParamSpec(
        "xtol",
        1e-10,
        min=1e-15,
        max=1e-2,
        log=True,
        help="Stop when the bracket half-width ≤ xtol + 2ε|x|.",
    ),
    ParamSpec("ftol", 0.0, min=0.0, max=1e-2, help="Also stop when |f(x)| ≤ ftol (0 disables)."),
    ParamSpec("max_iter", 200, kind="int", min=1, max=10_000, help="Iteration limit."),
)

ScalarFn = Callable[[float], float]


def _tol(x: float, xtol: float) -> float:
    """Brent's tolerance ``xtol + 2ε|x|`` (Brent 1973, ch. 4)."""
    return xtol + 2.0 * EPS * abs(x)


def _same_sign(u: float, v: float) -> bool:
    return math.copysign(1.0, u) == math.copysign(1.0, v)


class SafeScalar:
    """Evaluate a scalar function and turn floating-point exceptions into nan.

    Python's ``math`` functions and the float ``**`` operator raise ``OverflowError``,
    ``ValueError`` (math domain error) or ``ZeroDivisionError`` where IEEE arithmetic would
    give ±inf or nan. A method must not raise on numerical breakdown, so such an exception
    becomes ``nan`` (the methods' non-finite checks then stop the run) and its text is kept
    in ``error`` for the message. Any other exception (a bug in f) propagates.
    """

    __slots__ = ("error", "fn")

    def __init__(self, fn: Callable[[float], Any]) -> None:
        self.fn = fn
        self.error: str | None = None

    def __call__(self, x: float) -> float:
        self.error = None
        try:
            return float(self.fn(x))
        except (ArithmeticError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return math.nan


def raised(f: Counted) -> str:
    """Message suffix naming the exception of the last evaluation of ``f`` ("" if none)."""
    inner = f.fn
    if isinstance(inner, SafeScalar) and inner.error is not None:
        return f"; f raised {inner.error}"
    return ""


def _sorted(u: float, v: float) -> list[float]:
    return [u, v] if u <= v else [v, u]


def _resolve_bracket(problem: Problem, bracket: tuple[float, float] | None) -> tuple[float, float]:
    """The validated bracket: finite ends a < b with a finite width b − a.

    A finite width keeps every derived quantity finite: midpoints a + (b − a)/2, half-widths,
    Brent's m = (c − b)/2 and the chord terms all lie within [a, b] or below b − a. An
    infinite end or an overflowing width would make the half-width test read inf ≤ inf
    and return x = ±inf as "converged".
    """
    if bracket is None:
        bracket = problem.bracket
    if bracket is None:
        raise ValueError(f"{problem.id}: a bracket (a, b) is required")
    a, b = float(bracket[0]), float(bracket[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"invalid bracket: the ends must be finite, got ({a}, {b})")
    if not a < b:
        raise ValueError(f"invalid bracket: need a < b, got ({a}, {b})")
    if not math.isfinite(b - a):
        raise ValueError(
            f"invalid bracket: the width b − a of ({a}, {b}) overflows the float range; "
            "use a narrower bracket"
        )
    return a, b


def _check_xtol(xtol: float, *, positive: bool = False) -> None:
    """Raise ValueError unless xtol ≥ 0 (> 0 when ``positive``) and finite."""
    ok = math.isfinite(xtol) and (xtol > 0.0 if positive else xtol >= 0.0)
    if not ok:
        need = "> 0" if positive else "≥ 0"
        raise ValueError(f"xtol must be finite and {need}, got {xtol}")


def _start(
    method: str,
    problem: Problem | ScalarFn,
    bracket: tuple[float, float] | None,
    xtol: float,
) -> tuple[Counted, float, float, float, float, Result | None]:
    """Validate xtol and the bracket, evaluate both ends and check the sign change.

    Returns ``(f, a, b, f(a), f(b), early)``; ``early`` is a finished Result when f is
    exactly zero at an end.
    """
    _check_xtol(xtol, positive=method == "itp")
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    f = Counted(SafeScalar(prob.f))
    fa = float(f(a))
    note = raised(f)
    fb = float(f(b))
    note = note or raised(f)
    if not finite(fa, fb):
        raise ValueError(
            f"f must be finite at the bracket ends; got f({a})={fa}, f({b})={fb}{note}"
        )
    for x in (a, b):
        fx = fa if x == a else fb
        if fx == 0.0:
            info = {"bracket": [a, b], "new_bracket": [x, x]}
            trace = [Step(0, x, 0.0, info=info)]
            msg = "f is exactly zero at a bracket end"
            return f, a, b, fa, fb, Result(method, x, 0.0, True, msg, 0, f.n, trace=trace)
    if _same_sign(fa, fb):
        raise ValueError(f"f(a) and f(b) must have opposite signs; got f({a})={fa}, f({b})={fb}")
    return f, a, b, fa, fb, None


def _stop_reason(fx: float, half_width: float, tol: float, ftol: float) -> str | None:
    """The shared bracketing stopping test; returns the reason, or None to continue."""
    if fx == 0.0:
        return "f(x) is exactly zero"
    if abs(fx) <= ftol:
        return f"|f(x)| = {abs(fx):.3g} ≤ ftol"
    if half_width <= tol:
        return f"bracket half-width {half_width:.3g} ≤ tol = {tol:.3g}"
    return None


def step_estimate(step: float, prev_step: float | None) -> float | None:
    """A posteriori error estimate from the last two steps, or None without evidence.

    With s_k = ``step``, s_{k−1} = ``prev_step`` and the observed rate ρ = s_k/s_{k−1},
    returns s_k·max(1, ρ/(1 − ρ)) when ρ < 1, else None. The factor ρ/(1 − ρ) is the
    a posteriori error bound of a contraction with factor ρ (Burden & Faires §2.2;
    Dahlquist & Björck 2008, ch. 6): for linear convergence with ρ near 1 (regula falsi
    with a fixed end, Newton at a root of multiplicity m where ρ = 1 − 1/m) the error is
    that many times the step. The factor 1 keeps it at least as strict as the classical
    step |x_k − x_{k−1}| when the steps are irregular.

    It is an *estimate* from a single ratio, not a bound: after a large jump ρ is tiny and
    the estimate collapses to s_k. The bracketing methods therefore only use it to
    trigger a verifying probe (see the module docstring).
    """
    if prev_step is None or not step < prev_step:
        return None
    rho = step / prev_step
    return step * max(1.0, rho / (1.0 - rho))


def _estimate(points: list[float]) -> float | None:
    """Step-test value over consecutive points of one rule (0.0 when the last two coincide)."""
    if len(points) < 2:
        return None
    step = abs(points[-1] - points[-2])
    if step == 0.0:
        return 0.0
    prev = abs(points[-2] - points[-3]) if len(points) >= 3 else None
    return step_estimate(step, prev)


def _nonfinite(method: str, f: Counted, x: float, fx: float, k: int, trace: list[Step]) -> Result:
    msg = f"f(x) is not finite at x = {x!r} (f = {fx!r}{raised(f)})"
    return Result(method, x, fx, False, msg, k, f.n, trace=trace)


def _max_iter(
    method: str, x: float, fx: float, max_iter: int, nfev: int, trace: list[Step]
) -> Result:
    return Result(method, x, fx, False, f"reached max_iter={max_iter}", max_iter, nfev, trace=trace)


# --------------------------------------------------------------------------------------
# Bisection
# --------------------------------------------------------------------------------------


@register(
    id="bisection",
    family="roots",
    name="Bisection",
    params=COMMON_PARAMS,
    needs=("f", "bracket"),
    order="linear (rate ½)",
    summary="Halve the bracket and keep the half where f changes sign.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.1",),
)
def bisection(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Bisection method.

    Iteration (Burden & Faires, Alg. 2.1): m = a + (b - a)/2; keep [a, m] if
    f(a)·f(m) < 0, else [m, b]. After k halvings the root is within (b₀ - a₀)/2^{k+1}
    of the midpoint, so convergence is linear with rate ½.

    Stops (converged) when the half-width of the bracket that contains m is
    ≤ xtol + 2ε|m| (then |m − root| ≤ that half-width), when |f(m)| ≤ ``ftol``, or when
    f(m) == 0 exactly.
    """
    f, a, b, fa, fb, early = _start("bisection", problem, bracket, xtol)
    if early is not None:
        return early

    def info(a: float, b: float, fa: float, fb: float, m: float, fm: float) -> dict[str, Any]:
        new = [m, m] if fm == 0.0 else ([a, m] if not _same_sign(fa, fm) else [m, b])
        return {"bracket": [a, b], "new_bracket": new, "a": a, "b": b, "fa": fa, "fb": fb}

    m = a + 0.5 * (b - a)
    fm = float(f(m))
    trace = [Step(0, m, fm, info=info(a, b, fa, fb, m, fm))]
    k = 0
    while True:
        if not finite(fm):
            return _nonfinite("bisection", f, m, fm, k, trace)
        # NOTE: the half-width test uses the bracket that contains m (before the split).
        reason = _stop_reason(fm, 0.5 * (b - a), _tol(m, xtol), ftol)
        if reason is not None:
            return Result("bisection", m, fm, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter("bisection", m, fm, max_iter, f.n, trace)
        k += 1
        if not _same_sign(fa, fm):
            b, fb = m, fm
        else:
            a, fa = m, fm
        m = a + 0.5 * (b - a)
        fm = float(f(m))
        trace.append(Step(k, m, fm, step_size=0.5 * (b - a), info=info(a, b, fa, fb, m, fm)))


# --------------------------------------------------------------------------------------
# False position and its Illinois-type modifications
# --------------------------------------------------------------------------------------


def _chord_point(a: float, fa: float, b: float, fb: float) -> float:
    """Zero b − f_b(b − a)/(f_b − f_a) of the line through (a, fa) and (b, fb), fa·fb < 0.

    f_b − f_a has no cancellation because the signs differ. When the product f_b(b − a) or
    the difference f_b − f_a overflows (|f| and b − a both huge), the same point is formed
    from the weight w = f_b/(f_b − f_a) ∈ [0, 1], with both values halved when their
    difference overflows, so the result stays finite and in [a, b] up to rounding.
    """
    x = b - fb * (b - a) / (fb - fa)
    if math.isfinite(x):
        return x
    den = fb - fa
    w = fb / den if math.isfinite(den) else (0.5 * fb) / (0.5 * fb - 0.5 * fa)
    return b - w * (b - a)


def _inside(x: float, lo: float, hi: float) -> float:
    """x clamped to [lo, hi] and, if it lands on an end, moved one ulp inward (lo < hi).

    Keeps a computed point in the open bracket when rounding pushed it onto or past an end;
    if lo and hi are adjacent floats the end itself is returned.
    """
    if x <= lo:
        x = math.nextafter(lo, hi)
    elif x >= hi:
        x = math.nextafter(hi, lo)
    return min(max(x, lo), hi)


def _chord_root(a: float, fa: float, b: float, fb: float) -> float:
    """:func:`_chord_point` kept inside [a, b] (rounding can push it one ulp outside)."""
    x = _chord_point(a, fa, b, fb)
    lo, hi = (a, b) if a <= b else (b, a)
    return min(max(x, lo), hi)


def _false_position(
    method: str,
    problem: Problem | ScalarFn,
    bracket: tuple[float, float] | None,
    xtol: float,
    ftol: float,
    max_iter: int,
) -> Result:
    """Generic Illinois-type false position (Ford 1995, Algorithm 1).

    State: (a, f_a) = (x_{k−1}, f_{k−1}) — the retained end, its ordinate possibly scaled —
    and (b, f_b) = (x_k, f_k), the latest iterate; f_a and f_b have opposite signs.

        c = b − f_b (b − a)/(f_b − f_a);   f_c = f(c)
        if f_c f_b < 0:  (a, f_a) ← (b, f_b)          (the old iterate becomes the other end)
        else:            f_a ← m · f_a                (a is retained: scale its ordinate)
        (b, f_b) ← (c, f_c)

    with m = 1 (regula falsi), ½ (Illinois), f_b/(f_b + f_c) (Pegasus),
    1 − f_c/f_b or ½ if that is ≤ 0 (Anderson–Björck).

    Safeguard (see the module docstring): when the step test over consecutive chord points
    claims convergence but the bracket is still wider than 2·tol, the next point c is the
    probe b ± tol toward a; if that probe keeps the sign of f(b), the point after it is the
    bisection point of the bracket. Both safeguard points go through the update above with
    m = 1, so (a, b) always brackets the root.
    """
    f, a, b, fa, fb, early = _start(method, problem, bracket, xtol)
    if early is not None:
        return early
    trace: list[Step] = []
    chord_points: list[float] = []  # consecutive chord points since the last safeguard step
    kind = "chord"
    k = 0
    while True:
        old_bracket = _sorted(a, b)
        chord: list[list[float]] | None = None
        if kind == "chord":
            chord = [[a, fa], [b, fb]]
            x = _chord_root(a, fa, b, fb)
        elif kind == "verify":
            x = b + math.copysign(_tol(b, xtol), a - b)
        else:  # bisection after a failed verification
            x = old_bracket[0] + 0.5 * (old_bracket[1] - old_bracket[0])
        fx = float(f(x))
        step = abs(x - trace[-1].x) if trace else None
        info: dict[str, Any] = {"bracket": old_bracket, "step": kind, "chord": chord}
        if not finite(fx):
            trace.append(Step(k, x, fx, step_size=step, info=info))
            return _nonfinite(method, f, x, fx, k, trace)
        scale = 1.0
        if fx != 0.0:
            if not _same_sign(fx, fb):
                a, fa = b, fb
            elif kind == "chord":
                scale = _scale_factor(method, fb, fx)
                fa *= scale
        b, fb = x, fx
        new_bracket = [x, x] if fx == 0.0 else _sorted(a, b)
        estimate: float | None = None
        if kind == "chord":
            chord_points.append(x)
            estimate = _estimate(chord_points)
        info.update(new_bracket=new_bracket, scale=scale, estimate=estimate)
        trace.append(Step(k, x, fx, step_size=step, info=info))

        tol = _tol(x, xtol)
        reason = _stop_reason(fx, 0.5 * (new_bracket[1] - new_bracket[0]), tol, ftol)
        if reason is not None:
            if kind == "verify" and reason.startswith("bracket"):
                reason += " (f changes sign across the probe xₖ₋₁ ± tol)"
            return Result(method, x, fx, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter(method, x, fx, max_iter, f.n, trace)
        if kind == "chord":
            kind = "verify" if estimate is not None and estimate <= tol else "chord"
        else:
            # The probe kept the sign: the step test was misled, so bisect once.
            kind = "bisection" if kind == "verify" else "chord"
            chord_points = []
        k += 1


def _scale_factor(method: str, f_k: float, f_next: float) -> float:
    """Factor m for the retained end when f_k and f_{k+1} have the same sign."""
    if method == "regula_falsi":
        return 1.0
    if method == "illinois":
        return 0.5
    if method == "pegasus":
        return f_k / (f_k + f_next)  # same signs: no cancellation, 0 < m < 1
    # Anderson–Björck
    m = 1.0 - f_next / f_k
    return m if m > 0.0 else 0.5


@register(
    id="regula_falsi",
    family="roots",
    name="Regula falsi (false position)",
    params=COMMON_PARAMS,
    order="linear (one end usually stays fixed)",
    summary="Draw the chord between the bracket ends and keep the sub-bracket where f changes sign.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.5",),
    needs=("f", "bracket"),
)
def regula_falsi(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Regula falsi (method of false position), Burden & Faires Alg. 2.5.

    x_k is the zero of the chord through the current bracket ends,
    x = b − f(b)(b − a)/(f(b) − f(a)); the end with the same sign as f(x_k) is replaced.
    For convex or concave f one end never moves, the bracket does not shrink to zero, and
    convergence is linear with rate 1 − f'(r)(e − r)/f(e) for the fixed end e.

    Stops (converged) on the shared bracketing test only. The a posteriori step test
    |x_k − x_{k−1}|·max(1, ρ/(1 − ρ)) ≤ xtol + 2ε|x_k| (:func:`step_estimate`) triggers the
    probe x_k ± tol, which closes the bracket when x_k is within tol of the root.

    NOTE: Burden & Faires stop on |x_k − x_{k−1}| < TOL. With a fixed end the error is then
    up to ρ/(1 − ρ) times the step (6× the tolerance was observed with ρ ≈ 0.86), and on
    steep f a step test alone accepts points far from the root (eˣ − 10 on (0, 100) gave
    x = 1.5e−39). Here a step test only triggers the verifying probe.
    """
    return _false_position("regula_falsi", problem, bracket, xtol, ftol, max_iter)


@register(
    id="illinois",
    family="roots",
    name="Illinois",
    params=COMMON_PARAMS,
    order="superlinear (≈ 1.442 per evaluation)",
    summary="Regula falsi that halves the retained end's f-value when that end is kept again.",
    references=(
        "Dowell & Jarratt (1971), BIT 11, 168–174",
        "Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1",
    ),
    needs=("f", "bracket"),
)
def illinois(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Illinois method (Dowell & Jarratt 1971) in the form of Ford (1995), Alg. 1.

    Regula falsi, but when f(x_{k+1}) has the same sign as f(x_k) — the other end is kept
    again — that end's stored ordinate is halved (m = ½), which pulls the next chord
    toward it and frees the stuck end. The scaling also applies on the first step when the
    first chord point has the sign of f(b) (the initial pair is (x₀, x₁) = (a, b)).

    Stops (converged) on the shared bracketing test only; the a posteriori step test
    (:func:`step_estimate`) ≤ xtol + 2ε|x_k| triggers a verifying probe x_k ± tol.
    """
    return _false_position("illinois", problem, bracket, xtol, ftol, max_iter)


@register(
    id="pegasus",
    family="roots",
    name="Pegasus",
    params=COMMON_PARAMS,
    order="superlinear (≈ 1.642 per evaluation)",
    summary="Regula falsi that scales the retained end's f-value by f_k/(f_k + f_{k+1}).",
    references=(
        "Dowell & Jarratt (1972), BIT 12, 503–508",
        "Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1",
    ),
    needs=("f", "bracket"),
)
def pegasus(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Pegasus method (Dowell & Jarratt 1972) in the form of Ford (1995), Alg. 1.

    Like Illinois, but the retained end's ordinate is multiplied by
    m = f_k/(f_k + f_{k+1}) ∈ (0, 1) instead of ½.

    Stops (converged) on the shared bracketing test only; the a posteriori step test
    (:func:`step_estimate`) ≤ xtol + 2ε|x_k| triggers a verifying probe x_k ± tol.
    """
    return _false_position("pegasus", problem, bracket, xtol, ftol, max_iter)


@register(
    id="anderson_bjorck",
    family="roots",
    name="Anderson–Björck",
    params=COMMON_PARAMS,
    order="superlinear (≈ 1.7 per evaluation)",
    summary="Regula falsi that scales the retained end's f-value by 1 − f_{k+1}/f_k (or ½).",
    references=(
        "Anderson & Björck (1973), BIT 13, 253–264",
        "Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1",
    ),
    needs=("f", "bracket"),
)
def anderson_bjorck(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Anderson–Björck method (1973) in the form of Ford (1995), Alg. 1.

    The retained end's ordinate is multiplied by m = 1 − f_{k+1}/f_k, the factor that
    makes the parabola through the three last points have the right slope; if m ≤ 0 the
    Illinois value m = ½ is used.

    Stops (converged) on the shared bracketing test only; the a posteriori step test
    (:func:`step_estimate`) ≤ xtol + 2ε|x_k| triggers a verifying probe x_k ± tol.
    """
    return _false_position("anderson_bjorck", problem, bracket, xtol, ftol, max_iter)


# --------------------------------------------------------------------------------------
# Ridders
# --------------------------------------------------------------------------------------


@register(
    id="ridders",
    family="roots",
    name="Ridders",
    params=COMMON_PARAMS,
    needs=("f", "bracket"),
    order="superlinear (quadratic per iteration, √2 per evaluation)",
    summary="Factor out an exponential so the midpoint lies on a line, then take that line's zero.",
    references=(
        "Ridders (1979), IEEE Trans. Circuits Syst. 26(11), 979–980",
        "Press et al., Numerical Recipes (3rd ed.), §9.2.1 (zriddr)",
    ),
)
def ridders(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Ridders' method (Ridders 1979; Numerical Recipes §9.2.1).

    With m = (a + b)/2 and d = m − a, choose Q = e^{λd} > 0 so that the three values
    f(a), f(m)Q, f(b)Q² of h(x) = f(x)e^{λ(x−a)} are collinear:
    f(a) − 2f(m)Q + f(b)Q² = 0. The zero of that line is

        x = m + d · sign(f(a) − f(b)) · f(m) / √(f(m)² − f(a)f(b))      (NR eq. 9.2.4)

    (clamped to [a, b] against rounding). Then f(x) is evaluated and the tightest of
    [m, x], [a, x], [x, b] that keeps the sign change becomes the new bracket. x always moves from m toward the root, so the bracket
    at least halves per iteration. Two evaluations per iteration.

    Stops (converged) on the shared bracketing test only (new half-width ≤ xtol + 2ε|x|,
    f(x) == 0 or |f(x)| ≤ ``ftol``). The iterates often approach the root from one side,
    so the bracket only halves while x_k is already accurate; the a posteriori step test
    over consecutive Ridders points (:func:`step_estimate`) ≤ tol therefore triggers the
    probe x_k ± tol toward the far end. A sign change across it closes the bracket to
    width tol; otherwise Ridders' iteration continues from the bracket [probe, far end].

    NOTE: NR zriddr returns as soon as |x_k − x_{k−1}| ≤ xacc, without evaluating f at the
    new point. On multiple roots two consecutive Ridders points can land close together
    on the same side, far from the root ((x − 0.3)⁵ on (−1, 2) with xtol = 1e−6 gave an
    error of 27·2·tol), so the step test here only triggers the verifying probe. The
    returned x is the last point of the trace.
    """
    f, lo, hi, flo, fhi, early = _start("ridders", problem, bracket, xtol)
    if early is not None:
        return early
    trace: list[Step] = []
    points: list[float] = []  # consecutive Ridders points since the last verify step
    kind = "ridders"
    k = 0
    while True:
        old_bracket = [lo, hi]
        info: dict[str, Any] = {
            "bracket": old_bracket,
            "step": kind,
            "midpoint": None,
            "f_mid": None,
            "exp_factor": None,
            "transformed": None,
        }
        m = fm = math.nan
        if kind == "verify":
            # The last Ridders point is an end of [lo, hi]; probe tol inside, toward the other.
            x_last = trace[-1].x
            x = x_last + math.copysign(_tol(x_last, xtol), 1.0 if x_last == lo else -1.0)
            fx = float(f(x))
        else:
            m = lo + 0.5 * (hi - lo)
            fm = float(f(m))
            info.update(midpoint=m, f_mid=fm)
            if not finite(fm):
                trace.append(Step(k, m, fm, info={"bracket": old_bracket}))
                return _nonfinite("ridders", f, m, fm, k, trace)
            if fm == 0.0:
                # NOTE: the Ridders point equals m exactly when f(m) = 0; skip re-evaluating it.
                x, fx = m, 0.0
            else:
                # √(f(m)² − f(lo)f(hi)) written as hypot to avoid overflow; f(lo)f(hi) < 0.
                s = math.hypot(fm, math.sqrt(abs(flo)) * math.sqrt(abs(fhi)))
                x = m + (m - lo) * math.copysign(1.0, flo - fhi) * fm / s
                # NOTE: |f(m)/s| ≤ 1 puts x in [lo, hi] in exact arithmetic, but rounding of
                # f(m)/s → 1 and of m ± (m − lo) can push it one ulp outside, where f may be
                # undefined (√(x − 0.1) on (0.1, 0.7) gave x = 0.09999999999999998). Clamp,
                # as the chord point of the false-position family is clamped.
                x = min(max(x, lo), hi)
                fx = float(f(x))
                # Q is the positive root of f(hi)Q² − 2f(m)Q + f(lo) = 0. Use the form without
                # cancellation (the product of the two roots is f(lo)/f(hi)).
                if _same_sign(fm, fhi):
                    q = (fm + math.copysign(s, fhi)) / fhi
                else:
                    q = flo / (fm - math.copysign(s, fhi))
                info.update(exp_factor=q, transformed=[[lo, flo], [m, fm * q], [hi, fhi * q * q]])
        step = abs(x - trace[-1].x) if trace else None
        if not finite(fx):
            trace.append(Step(k, x, fx, step_size=step, info=info))
            return _nonfinite("ridders", f, x, fx, k, trace)
        if fx == 0.0:
            lo = hi = x
        elif kind == "ridders" and not _same_sign(fm, fx):
            (lo, flo), (hi, fhi) = sorted(((m, fm), (x, fx)))
        elif not _same_sign(flo, fx):
            hi, fhi = x, fx
        else:
            lo, flo = x, fx
        estimate: float | None = None
        if kind == "ridders":
            points.append(x)
            estimate = _estimate(points)
        info.update(new_bracket=[lo, hi], estimate=estimate)
        trace.append(Step(k, x, fx, step_size=step, info=info))
        tol = _tol(x, xtol)
        reason = _stop_reason(fx, 0.5 * (hi - lo), tol, ftol)
        if reason is not None:
            if kind == "verify" and reason.startswith("bracket"):
                reason += " (f changes sign across the probe xₖ₋₁ ± tol)"
            return Result("ridders", x, fx, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter("ridders", x, fx, max_iter, f.n, trace)
        if kind == "ridders" and estimate is not None and estimate <= tol:
            kind = "verify"
        else:
            if kind == "verify":  # the probe kept the sign: the step test was misled
                points = []
            kind = "ridders"
        k += 1


# --------------------------------------------------------------------------------------
# Brent
# --------------------------------------------------------------------------------------


@register(
    id="brent",
    family="roots",
    name="Brent (zeroin)",
    params=COMMON_PARAMS,
    needs=("f", "bracket"),
    order="superlinear (≈ 1.84 with inverse quadratic steps); never much worse than bisection",
    summary="Inverse quadratic / secant steps, with a bisection fallback whenever they are unsafe.",
    references=(
        "Brent (1973), Algorithms for Minimization without Derivatives, ch. 4, procedure zero",
        "Press et al., Numerical Recipes (3rd ed.), §9.3 (zbrent)",
    ),
)
def brent(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Brent's method, a transcription of procedure ``zero`` (Brent 1973, ch. 4, §4.6).

    State: b is the best estimate (|f(b)| ≤ |f(c)|), c the contrapoint (f(b)·f(c) < 0),
    a the previous b. With m = (c − b)/2 and tol = 2ε|b| + xtol:

    * if the last-but-one step e is large enough and |f(a)| > |f(b)|, try interpolation —
      secant through a, b when a == c, else inverse quadratic through a, b, c — giving the
      step p/q from b;
    * accept it only if 2p < 3mq − |tol·q| (the new point lies within ¾ of the way to c)
      and p < |½ e q| (the step is less than half the step before last); otherwise bisect;
    * never step less than tol.

    After each evaluation, c is reset to a when f(b) and f(c) share a sign, and b, c are
    swapped so that |f(b)| ≤ |f(c)|.

    Stops (converged) when |m| = ½|c − b| ≤ tol, f(b) == 0, or |f(b)| ≤ ``ftol``; returns
    b, which is then within 2·tol of a root. ``Result.x`` is b (``info["best"]``), which
    may be an earlier point than the last evaluated ``trace[-1].x``.
    """
    f, a, b, fa, fb, early = _start("brent", problem, bracket, xtol)
    if early is not None:
        return early
    c, fc = a, fa
    d = e = b - a
    if abs(fc) < abs(fb):
        a, b, c = b, c, b
        fa, fb, fc = fb, fc, fb

    def converged_reason() -> str | None:
        return _stop_reason(fb, 0.5 * abs(c - b), _tol(b, xtol), ftol)

    reason = converged_reason()
    if reason is not None:  # the initial bracket already meets the tolerance
        info0 = {"bracket": _sorted(b, c), "new_bracket": _sorted(b, c), "best": b}
        return Result("brent", b, fb, True, reason, 0, f.n, trace=[Step(0, b, fb, info=info0)])

    trace: list[Step] = []
    k = 0
    while True:
        tol = _tol(b, xtol)
        m = 0.5 * (c - b)
        old_bracket = _sorted(b, c)
        attempted: str | None = None
        points: list[list[float]] = []
        kind = "bisection"
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                attempted = "secant"
                points = [[a, fa], [b, fb]]
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                attempted = "inverse_quadratic"
                points = [[a, fa], [b, fb], [c, fc]]
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            e_old = e
            e = d
            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * e_old * q):
                d = p / q
                kind = attempted
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        step = d if abs(d) > tol else math.copysign(tol, m)
        b = b + step
        fb = float(f(b))
        x_new, f_new = b, fb
        info: dict[str, Any] = {
            "bracket": old_bracket,
            "step": kind,
            "attempted": attempted,
            "points": points,
            "tol": tol,
        }
        if not finite(fb):
            trace.append(Step(k, x_new, f_new, step_size=abs(step), info=info))
            return _nonfinite("brent", f, x_new, f_new, k, trace)
        if (fb > 0.0) == (fc > 0.0):  # Brent's "int": the sign change is between a and b
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):  # Brent's "ext": keep the better point in b
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        info["new_bracket"] = [b, b] if fb == 0.0 else _sorted(b, c)
        info["best"] = b
        trace.append(Step(k, x_new, f_new, step_size=abs(step), info=info))
        reason = converged_reason()
        if reason is not None:
            return Result("brent", b, fb, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter("brent", b, fb, max_iter, f.n, trace)
        k += 1


# --------------------------------------------------------------------------------------
# Chandrupatla
# --------------------------------------------------------------------------------------


@register(
    id="chandrupatla",
    family="roots",
    name="Chandrupatla",
    params=COMMON_PARAMS,
    needs=("f", "bracket"),
    order="superlinear (inverse quadratic when locally valid, else bisection)",
    summary="Inverse quadratic interpolation only where the data say it is valid; bisect otherwise.",
    references=("Chandrupatla (1997), Advances in Engineering Software 28(3), 145–149",),
)
def chandrupatla(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
) -> Result:
    """Chandrupatla's hybrid quadratic/bisection method (Chandrupatla 1997).

    State: x₁ the newest point, x₂ the other bracket end (f(x₁)f(x₂) < 0), x₃ the end
    just discarded. The next point is x₁ + t(x₂ − x₁). With
    ξ = (x₁ − x₂)/(x₃ − x₂) and Φ = (f₁ − f₂)/(f₃ − f₂), inverse quadratic interpolation
    through the three points is used when 1 − √(1 − ξ) < Φ < √ξ (the inverse quadratic is
    then monotone on the bracket):

        t = f₁/(f₁ − f₂) · f₃/(f₃ − f₂) − (x₃ − x₁)/(x₂ − x₁) · f₁/(f₃ − f₁) · f₂/(f₂ − f₃)

    otherwise t = ½. t is clipped to [t_l, 1 − t_l], t_l = tol/|x₂ − x₁|, so every step is
    at least tol away from the bracket ends. The first step is a bisection.
    NOTE: when tol is below the float spacing of the far end (xtol = 0 with a root near 0
    and an end near 1), t = 1 − t_l rounds to 1 and x₁ + t(x₂ − x₁) rounded to a point
    outside the bracket (0.0 for the bracket [2.6e−255, 0.5]); the point is therefore kept
    strictly inside (:func:`_inside`).

    Stops (converged) when |x₂ − x₁| < 2·tol with tol = xtol + 2ε|x_m| (Chandrupatla's
    t_l > ½, his δ = 2·xtol), or f(x_m) == 0, or |f(x_m)| ≤ ``ftol``, where x_m is the end
    with the smaller |f|; returns x_m.
    """
    f, x1, x2, f1, f2, early = _start("chandrupatla", problem, bracket, xtol)
    if early is not None:
        return early
    x3, f3 = x2, f2
    t = 0.5
    pending: dict[str, Any] = {"step": "bisection", "t": t, "points": [], "xi": None, "phi": None}
    trace: list[Step] = []
    k = 0
    while True:
        old_bracket = _sorted(x1, x2)
        xt = _inside(x1 + t * (x2 - x1), *old_bracket)
        ft = float(f(xt))
        info: dict[str, Any] = {"bracket": old_bracket, **pending}
        if not finite(ft):
            trace.append(Step(k, xt, ft, info=info))
            return _nonfinite("chandrupatla", f, xt, ft, k, trace)
        if _same_sign(ft, f1):
            x3, f3 = x1, f1
        else:
            x3, f3 = x2, f2
            x2, f2 = x1, f1
        x1, f1 = xt, ft
        xm, fm = (x1, f1) if abs(f1) < abs(f2) else (x2, f2)
        tol = _tol(xm, xtol)
        dx = abs(x2 - x1)
        info["new_bracket"] = [xt, xt] if ft == 0.0 else _sorted(x1, x2)
        info["best"] = xm
        trace.append(Step(k, xt, ft, info=info))
        reason: str | None = None
        if fm == 0.0:
            reason = "f(x) is exactly zero"
        elif abs(fm) <= ftol:
            reason = f"|f(x)| = {abs(fm):.3g} ≤ ftol"
        elif 0.5 * dx < tol:  # Chandrupatla's strict test t_l > ½ ⇔ |x₂ − x₁| < 2·tol
            reason = f"bracket half-width {0.5 * dx:.3g} < tol = {tol:.3g}"
        if reason is not None:
            return Result("chandrupatla", xm, fm, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter("chandrupatla", xm, fm, max_iter, f.n, trace)
        # Choose the next t.
        xi = (x1 - x2) / (x3 - x2)
        phi = (f1 - f2) / (f3 - f2)
        if 1.0 - math.sqrt(max(0.0, 1.0 - xi)) < phi < math.sqrt(max(0.0, xi)):
            alpha = (x3 - x1) / (x2 - x1)
            t = f1 / (f1 - f2) * f3 / (f3 - f2) - alpha * f1 / (f3 - f1) * f2 / (f2 - f3)
            kind = "inverse_quadratic"
            points = [[x1, f1], [x2, f2], [x3, f3]]
        else:
            t = 0.5
            kind = "bisection"
            points = []
        tl = tol / dx
        t = min(1.0 - tl, max(tl, t))
        pending = {"step": kind, "t": t, "points": points, "xi": xi, "phi": phi}
        k += 1


# --------------------------------------------------------------------------------------
# ITP
# --------------------------------------------------------------------------------------


def itp_n_half(a: float, b: float, eps: float) -> int:
    """n½ = ⌈log₂((b − a)/2ε)⌉ ≥ 0, computed exactly: the least n ≥ 0 with 2ε·2ⁿ ≥ b − a.

    Needs a finite width b − a and ε > 0. The comparison runs on the exact rational
    (b − a)/(2ε) = p/q, so it neither rounds nor overflows (a float ratio overflows for
    b − a ≈ 1e308 and a small ε).
    """
    if not (eps > 0.0 and math.isfinite(eps) and math.isfinite(b - a)):
        raise ValueError(f"itp_n_half needs ε > 0 and a finite width, got ε={eps}, b − a={b - a}")
    ratio = Fraction(b - a) / (2 * Fraction(eps))
    p, q = ratio.numerator, ratio.denominator
    n = max(0, p.bit_length() - q.bit_length())
    while q << n < p:  # 2ⁿ < p/q
        n += 1
    while n > 0 and q << (n - 1) >= p:  # 2ⁿ⁻¹ ≥ p/q
        n -= 1
    return n


@register(
    id="itp",
    family="roots",
    name="ITP (interpolate–truncate–project)",
    params=(
        *COMMON_PARAMS,
        ParamSpec(
            "kappa1",
            0.1,
            min=1e-4,
            max=10.0,
            log=True,
            help="Truncation size κ₁ in δ = κ₁(b − a)^κ₂ (κ₁ > 0).",
        ),
        ParamSpec(
            "kappa2",
            2.0,
            min=1.0,
            max=2.6,
            help="Truncation exponent κ₂ ∈ [1, 1 + φ) (φ = golden ratio).",
        ),
        ParamSpec(
            "n0",
            1,
            kind="int",
            min=0,
            max=20,
            help="Slack: at most n0 more iterations than bisection in the worst case.",
        ),
    ),
    needs=("f", "bracket"),
    order="superlinear on smooth f; worst case ⌈log₂((b−a)/(2·xtol))⌉ + n₀ iterations",
    summary="Regula falsi, nudged toward the midpoint and kept within bisection's worst-case budget.",
    references=("Oliveira & Takahashi (2020), ACM Trans. Math. Softw. 47(1), Art. 5, Alg. 1",),
)
def itp(
    problem: Problem | ScalarFn,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 200,
    kappa1: float = 0.1,
    kappa2: float = 2.0,
    n0: int = 1,
) -> Result:
    """The ITP method (Oliveira & Takahashi 2020, Algorithm 1) with ε = ``xtol``.

    With n½ = ⌈log₂((b₀ − a₀)/2ε)⌉ and n_max = n½ + n₀, iteration j computes

    * interpolation: x_f = regula falsi point, x½ = (a + b)/2;
    * truncation: σ = sign(x½ − x_f), δ = max(κ₁(b − a)^κ₂, ε/2);
      x_t = x_f + σδ if δ ≤ |x½ − x_f| else x½;
    * projection: r = ε·2^{n_max − j} − (b − a)/2;
      x_ITP = x_t if |x_t − x½| ≤ r else x½ − σr;

    then keeps the sub-bracket with the sign change. In exact arithmetic it never needs
    more than n_max iterations (Oliveira & Takahashi 2020, Thm. 2.1; bisection needs n½),
    yet converges superlinearly on smooth f.

    Stops (converged) when b − a ≤ 2ε (the paper's test), f(x) == 0, |f(x)| ≤ ``ftol``,
    or a and b are adjacent floating-point numbers.

    NOTE: the paper returns the midpoint (a + b)/2 of the final bracket (error ≤ ε); we
    return the bracket end with the smaller |f| (error ≤ 2ε) so that ``Result.fun`` is a
    value of f that was actually evaluated and the returned x lies on the trace.
    NOTE: ε is the absolute ``xtol`` (no 2ε_mach|x| term), because the worst-case bound
    needs a fixed ε > 0; ``xtol ≤ 0`` raises ``ValueError`` (the other bracketing methods
    accept xtol = 0 through their 2ε_mach|x| term). A tolerance below the float spacing of
    the root stops on the adjacent-floats test.
    NOTE: the bound rests on b_{j+1} − a_{j+1} ≤ (b_j − a_j)/2 + r_j = ε·2^{n_max − j}, which
    holds with equality once the projection is active, and from then on every later step
    is tight too. In floating point each step adds a rounding error |e_j| ≤ 1.5·ulp(M),
    M = max(|a₀|, |b₀|) (from b − a, a + (b − a)/2 and x½ − σr), and the final width can
    exceed 2ε by up to Σ|e_j|/2^{n−1−j} ≤ 3·ulp(M): a literal transcription needs n_max + 1
    iterations on ~13% of random brackets. The projection radius therefore uses
    ε′ = ε − 2·ulp(M) (or ε/2 if that is larger) in r = ε′·2^{n_max − j} − (b − a)/2, so
    that 2ε′ + 3·ulp(M) ≤ 2ε. n½ is computed exactly (the least n ≥ 0 with 2ε·2ⁿ ≥ b₀ − a₀)
    instead of from a rounded log₂. The bound can still be exceeded when ε′ < ε/2 + 2ulp
    (xtol within a few ulps of the float spacing; the run then stops on adjacent floats)
    or when n₀ = 0 and b₀ − a₀ lies within 4·ulp(M)·2^{n½} below 2ε·2^{n½}.
    """
    if not kappa1 > 0.0:
        raise ValueError(f"kappa1 must be > 0, got {kappa1}")
    if not 1.0 <= kappa2 < 1.0 + 0.5 * (1.0 + math.sqrt(5.0)):
        raise ValueError(f"kappa2 must lie in [1, 1 + φ), got {kappa2}")
    if n0 < 0:
        raise ValueError(f"n0 must be ≥ 0, got {n0}")
    f, a, b, fa, fb, early = _start("itp", problem, bracket, xtol)
    if early is not None:
        return early
    eps = xtol
    n_half = itp_n_half(a, b, eps)
    n_max = n_half + n0
    eps_r = max(eps - 2.0 * math.ulp(max(abs(a), abs(b))), 0.5 * eps)  # ε′ of the radius

    def best() -> tuple[float, float]:
        return (a, fa) if abs(fa) <= abs(fb) else (b, fb)

    if b - a <= 2.0 * eps:
        x, fx = best()
        info0 = {"bracket": [a, b], "new_bracket": [a, b], "best": x}
        msg = f"bracket width {b - a:.3g} ≤ 2·xtol"
        return Result("itp", x, fx, True, msg, 0, f.n, trace=[Step(0, x, fx, info=info0)])

    trace: list[Step] = []
    k = 0
    while True:
        old_bracket = [a, b]
        width = b - a
        x_half = a + 0.5 * width
        # NOTE: ε′ < ε absorbs rounding (see the docstring); r is clipped at 0 (it would
        # turn negative once j > n_max).
        try:
            radius = math.ldexp(eps_r, n_max - k)
        except OverflowError:  # ε′·2^{n_max − j} beyond the float range: no projection
            radius = math.inf
        r = max(0.0, radius - 0.5 * width)
        # NOTE: δ is floored at ε/2. In floating point κ₁(b − a)^κ₂ falls below the spacing
        # of the floats near the root, x_t then rounds to x_f, and every step lands on the
        # same side of the root (the far end never moves; observed on Wallis' cubic: 33
        # iterations instead of 7). The floor keeps the nudge visible to the ε test; the
        # projection step still enforces the n½ + n₀ worst-case bound for any x_t.
        try:
            delta = max(kappa1 * width**kappa2, 0.5 * eps)
        except OverflowError:  # width^κ₂ beyond the float range: no truncation step
            delta = math.inf
        # x_f = (b f(a) − a f(b))/(f(a) − f(b)), clamped to [a, b] against rounding: x_t and
        # the projection stay between x_f and x½, hence inside the bracket.
        x_f = _chord_root(a, fa, b, fb)
        sigma = math.copysign(1.0, x_half - x_f) if x_half != x_f else 0.0
        x_t = x_f + sigma * delta if delta <= abs(x_half - x_f) else x_half
        projected = abs(x_t - x_half) > r
        x = x_half - sigma * r if projected else x_t
        fx = float(f(x))
        info: dict[str, Any] = {
            "bracket": old_bracket,
            "x_half": x_half,
            "x_f": x_f,
            "x_t": x_t,
            "delta": delta,
            "r": r,
            "projected": projected,
        }
        if not finite(fx):
            trace.append(Step(k, x, fx, info=info))
            return _nonfinite("itp", f, x, fx, k, trace)
        if fx == 0.0:
            a = b = x
            fa = fb = 0.0
        elif _same_sign(fx, fa):
            a, fa = x, fx
        else:
            b, fb = x, fx
        xb, fxb = best()
        info["new_bracket"] = [a, b]
        info["best"] = xb
        trace.append(Step(k, x, fx, info=info))
        reason: str | None = None
        if fx == 0.0:
            reason = "f(x) is exactly zero"
        elif abs(fx) <= ftol:
            reason = f"|f(x)| = {abs(fx):.3g} ≤ ftol"
        elif b - a <= 2.0 * eps:
            reason = f"bracket width {b - a:.3g} ≤ 2·xtol"
        elif math.nextafter(a, math.inf) >= b:
            reason = "bracket ends are adjacent floating-point numbers"
        if reason is not None:
            return Result("itp", xb, fxb, True, reason, k, f.n, trace=trace)
        if k == max_iter:
            return _max_iter("itp", xb, fxb, max_iter, f.n, trace)
        k += 1


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("bisection", "sqrt2", {}),
    ("regula_falsi", "x10_minus_1", {}),
    ("illinois", "x10_minus_1", {}),
    ("pegasus", "steep_exp", {}),
    ("anderson_bjorck", "cubic", {}),
    ("ridders", "kepler", {}),
    ("brent", "wilkinson5", {}),
    ("chandrupatla", "cos_minus_x", {}),
    ("itp", "steep_exp", {}),
    # The chord rounds onto x = −50; the verified step test then probes and bisects.
    ("illinois", "steep_exp", {"bracket": (-50.0, 60.0)}),
]
