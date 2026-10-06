"""One-dimensional minimization: interval elimination, interpolation, Newton and bracketing.

Interval-elimination methods (golden section, Fibonacci, dichotomous, ternary) rest on one
fact (Bazaraa, Sherali & Shetty (2006), §8.2, interval-elimination theorem): if f is
strictly quasiconvex (unimodal) on [a, b] and a ≤ x₁ < x₂ ≤ b, then

* f(x₁) ≤ f(x₂)  ⇒  a minimizer lies in [a, x₂]   (we discard (x₂, b]),
* f(x₁) > f(x₂)  ⇒  a minimizer lies in [x₁, b]   (we discard [a, x₁)).

Every method here uses exactly this rule, ties included, so in exact arithmetic the bracket
always contains a minimizer of a unimodal f. The returned ``x`` is the best point of the
current interior pair (or triple), which lies in the final bracket, so ``|x - x*| ≤ b - a``
for unimodal f, up to the accuracy floor below.

Accuracy floor: near a minimizer f(x) ≈ f* + ½ f''(x*)(x - x*)², so points within
h ≈ √(8ε|f*|/f''(x*)) of x* have f-values that differ by less than a few ulps of f*
(h ≈ √ε·(1 + |x*|) for a well-scaled f; 1e-8 to 1e-7 on the library problems).
Comparisons there are decided by rounding, so no method that compares f-values can locate
x* better than about h, whatever ``xtol`` says, and a final bracket narrower than h can
miss x* by up to about h (Press et al., Numerical Recipes, 3rd ed., §10.1; Brent (1973),
Ch. 5). For golden section, Fibonacci, ternary search, parabolic interpolation and Brent
the guarantee is therefore |x - x*| ≲ max(xtol, h). dichotomous_search compares points
only δ apart, so rounding can decide its comparisons far outside h; it stops with
converged=False when that happens (see its docstring).

Info keys (emitted in every Step, including k = 0, by the methods listed):
  all methods except newton_1d:
    bracket: [a, b] -- the interval known to contain a minimizer after this step.
        bracket_minimum: [min, max] of its current search triple; this interval is a true
        bracket only when f(b) ≤ f(a) and f(b) ≤ f(c), i.e. on the final step of a
        converged run (earlier steps are still walking downhill).

  golden_section, fibonacci_search, dichotomous_search, ternary_search:
    interior: [x1, x2] -- the two interior points compared in the next iteration
        (for the last Fibonacci step: λₙ and μₙ = λₙ + ε).
    f_interior: [f(x1), f(x2)].
    evaluated: [x, ...] -- the points evaluated during this step (k = 0: setup points).
    cut: "left" | "right" | null -- the end of the previous bracket that this step
        discarded ("left": [a, x1) removed; "right": (x2, b] removed; null at k = 0).
    ratio: float | null (fibonacci_search only) -- F_{n-k-1}/F_{n-k}: each interior point
        lies this fraction of the bracket width from the far end (the next reduction
        factor); 0.5 marks the last (ε-shifted) step; null at k = 0.
    eps: float | null (fibonacci_search only) -- the shift q - p used in the last step
        (ε, or the spacing of doubles at p when ε is smaller), else null.

  parabolic_interpolation:
    triple: [x_l, x_m, x_r] -- the current three points, x_l < x_m < x_r.
    f_triple: [f(x_l), f(x_m), f(x_r)].
    pattern: bool -- f(x_m) ≤ min(f(x_l), f(x_r)) (the "three-point pattern").
    step: "init" | "parabolic" | "probe" | "golden" | "bisect" -- how the trial point was
        chosen ("probe": the vertex was closer than xtol/2 to x_m; "golden": the fit was
        unusable or its step failed the progress test; see the docstring).
    trial: float | null -- the point evaluated in this step (null at k = 0).
    f_trial: float | null.
    parabola: {center: c, coef: [c0, c1, c2]} | null -- the interpolating parabola
        p(z) = c0 + c1 (z - c) + c2 (z - c)² through the *previous* triple (c = its x_m).
    vertex: float | null -- the vertex of that parabola, c - c1/(2 c2) (also when rejected).
    max_step: float | null -- the progress bound ½·(step before last): a vertex step
        |vertex - x_m| must be smaller to be accepted (null while unbounded, k ≤ 2).

  brent_minimize:
    xwv: [x, w, v] -- best point, second best, previous w (Brent's notation).
    f_xwv: [f(x), f(w), f(v)].
    step: "init" | "parabolic" | "golden".
    trial: float | null -- the point u evaluated in this step.
    f_trial: float | null.
    parabola: {center, coef} | null -- the parabola through the previous (x, w, v) when
        Brent tried a fit and the three points were distinct.
    vertex: float | null -- the vertex x + p/q of the attempted fit (also when rejected).
    tol: float -- Brent's tol1 = rtol·|x| + xtol/3 used in this step.

  newton_1d:
    step: "init" | "newton" | "gradient" -- "gradient" when f''(x) ≤ 0 forced the safeguard.
    g: float -- f'(x) at this iterate.  h: float -- f''(x) at this iterate.
    parabola: {center, coef} | null -- the Taylor model f(c) + f'(c)(z-c) + ½f''(c)(z-c)²
        at the previous iterate c that produced this step.
    alpha: float | null -- the accepted gradient step length (gradient steps only).
    trials: [[alpha, f]] -- backtracking trials of a gradient step ([] otherwise).

  bracket_minimum:
    triple: [a, b, c] -- the current triple in search order (b between a and c).
    f_triple: [f(a), f(b), f(c)].
    step: "init" | "golden" | "parabolic" | "parabolic_far" | "limit" -- how the last
        trial was produced (NR mnbrak branches).
    trials: [[u, f(u)]] -- every point evaluated in this step, in order.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

from ..core import diff
from ..core.counting import Counted, scalar_problem, start_scalar
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step

EPS = sys.float_info.epsilon
SQRT_EPS = math.sqrt(EPS)
#: Golden-section fraction ρ = (3 - √5)/2 = 1 - 1/φ ≈ 0.381966.
RHO = 0.5 * (3.0 - math.sqrt(5.0))
#: NR mnbrak magnification ratio (NR 3rd ed. §10.1 uses this 7-digit value of φ).
GOLD = 1.618034
#: NR mnbrak guard against division by zero in the parabolic extrapolation.
TINY = 1e-20
#: Armijo constant and maximum number of halvings for Newton's gradient safeguard
#: (Nocedal & Wright (2006), Alg. 3.1).
ARMIJO_C1 = 1e-4
MAX_BACKTRACK = 60
#: Factor of the floating-point floor on the bracket width: no interval can be resolved
#: below a few ulps of its end points.
RESOLUTION_ULPS = 4.0
#: Fibonacci search plans for xtol - FIB_SLACK_ULPS·ε·max(|a|, |b|): a budget for the
#: rounding error of the computed final width (see fibonacci_search).
FIB_SLACK_ULPS = 16.0
#: Smallest UI xtol of dichotomous_search: δ = delta_ratio·xtol ≥ 1e-13 must stay above
#: the resolution 4ε·max(|a|, |b|) on every library bracket.
DICHOTOMOUS_XTOL_MIN = 1e-10
#: dichotomous_search treats a comparison as decided by rounding (a "tie") when
#: |f(x₁) - f(x₂)| ≤ TIE_ULPS·ε·max(|f(x₁)|, |f(x₂)|): a few ulps of the larger value.
TIE_ULPS = 4.0
#: Largest Fibonacci number a plan may use (L/F_n is then below every float resolution).
FIB_MAX = 2**1000

Func = Callable[[float], float]
ScalarProblem = Problem | Func


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _resolve_bracket(problem: Problem, bracket: Any) -> tuple[float, float]:
    if bracket is None:
        bracket = problem.bracket
    if bracket is None:
        raise ValueError(f"{problem.id}: a bracket (a, b) is required")
    a, b = float(bracket[0]), float(bracket[1])
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"invalid bracket: end points must be finite, got ({a}, {b})")
    if not a < b:
        raise ValueError(f"invalid bracket: need a < b, got ({a}, {b})")
    if not math.isfinite(b - a):
        raise ValueError(f"invalid bracket: the width b - a overflows, got ({a}, {b})")
    return a, b


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a positive finite number, got {value}")


def _require_max_iter(max_iter: int) -> None:
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")


def _width_tol(xtol: float, a: float, b: float) -> float:
    """The bracket-width stopping threshold: xtol, floored at floating-point resolution."""
    return max(xtol, RESOLUTION_ULPS * EPS * max(abs(a), abs(b)))


def _width_reason(width: float, xtol: float) -> str:
    if width <= xtol:
        return f"bracket width {width:.3g} ≤ xtol"
    return f"bracket width {width:.3g} reached floating-point resolution (xtol={xtol:.3g})"


def _nonfinite(method: str, x: float, fx: float) -> str:
    return f"{method}: f({x:.17g}) = {fx} is not finite; stopped"


def _parabola(
    x0: float, f0: float, x1: float, f1: float, x2: float, f2: float
) -> dict[str, Any] | None:
    """The parabola through three points in centered form about x0, or None.

    p(z) = f0 + c1 (z - x0) + c2 (z - x0)², from Newton divided differences:
    c2 = f[x0, x1, x2], c1 = f[x0, x1] + c2 (x0 - x1).
    """
    if x0 == x1 or x0 == x2 or x1 == x2:
        return None
    d01 = (f1 - f0) / (x1 - x0)
    d02 = (f2 - f0) / (x2 - x0)
    c2 = (d02 - d01) / (x2 - x1)
    c1 = d01 + c2 * (x0 - x1)
    if not (math.isfinite(c1) and math.isfinite(c2)):
        return None
    return {"center": x0, "coef": [f0, c1, c2]}


def _result(
    method: str,
    x: float,
    fx: float,
    converged: bool,
    message: str,
    n_iter: int,
    trace: list[Step],
    *,
    n_fev: int,
    n_gev: int = 0,
    n_hev: int = 0,
    extra: dict[str, Any] | None = None,
) -> Result:
    return Result(
        method,
        x,
        fx,
        converged,
        message,
        n_iter,
        n_fev,
        n_gev,
        n_hev,
        trace=trace,
        extra=extra or {},
    )


def _evaluate_logged(f: Func, log: list[list[float]], z: float) -> float:
    """Evaluate f(z) and append [z, f(z)] to ``log``."""
    fz = f(z)
    log.append([z, fz])
    return fz


def _interval_info(
    a: float,
    b: float,
    x1: float,
    x2: float,
    f1: float,
    f2: float,
    evaluated: list[float],
    cut: str | None,
) -> dict[str, Any]:
    return {
        "bracket": [a, b],
        "interior": [x1, x2],
        "f_interior": [f1, f2],
        "evaluated": evaluated,
        "cut": cut,
    }


XTOL_PARAM = ParamSpec(
    "xtol",
    1e-8,
    min=1e-12,
    max=1e-1,
    log=True,
    help="Stop when the bracket width b − a ≤ xtol. For unimodal f, |x − x⋆| ≤ max(xtol, h): "
    "f-comparisons cannot resolve x⋆ below h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).",
)


def _max_iter_param(default: int) -> ParamSpec:
    return ParamSpec("max_iter", default, kind="int", min=1, max=10_000, help="Iteration limit.")


# --------------------------------------------------------------------------------------
# Golden-section search
# --------------------------------------------------------------------------------------


@register(
    id="golden_section",
    family="scalar",
    name="Golden-section search",
    params=(XTOL_PARAM, _max_iter_param(200)),
    needs=("f", "bracket"),
    order="linear (rate 1/φ ≈ 0.618)",
    summary="Compare f at two golden-ratio points, discard the worse end; one of the old "
    "points is reused, so each step costs one new evaluation.",
    references=(
        "Kiefer (1953), Sequential minimax search for a maximum, Proc. AMS 4, 502–506",
        "Press et al., Numerical Recipes (3rd ed.), §10.2 (routine Golden)",
        "Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2",
    ),
)
def golden_section(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-8,
    max_iter: int = 200,
) -> Result:
    """Golden-section search on [a, b].

    Setup: x₁ = a + ρ(b - a), x₂ = x₁ + ρ(b - x₁) = a + (1 - ρ)(b - a) with
    ρ = (3 - √5)/2, so both interior points sit at golden-ratio positions.
    Iteration (NR §10.2): if f(x₁) ≤ f(x₂) the new bracket is [a, x₂], the old x₁
    becomes the new x₂ and the new x₁ = x₂ - ρ(x₂ - a); otherwise the new bracket is
    [x₁, b], the old x₂ becomes the new x₁ and the new x₂ = x₁ + ρ(b - x₁). Because
    ρ² - 3ρ + 1 = 0, i.e. ρ/(1 - ρ) = 1 - ρ, the reused point is again at a golden
    position, so every iteration costs exactly ONE evaluation and shrinks the width by
    1/φ = 1 - ρ ≈ 0.618.

    Counts: n_fev = 2 + n_iter (two setup evaluations, then one per iteration).

    Stopping test (converged): b - a ≤ max(xtol, 4ε·max(|a|, |b|)); the second term
    only stops a too-small xtol at floating-point resolution. Returns the best evaluated
    point, which lies in [a, b]. A non-finite f-value stops with converged=False.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol)
    _require_max_iter(max_iter)
    f = Counted(prob.f)

    x1 = a + RHO * (b - a)
    x2 = x1 + RHO * (b - x1)
    f1, f2 = f(x1), f(x2)
    xb, fb = (x1, f1) if f1 <= f2 else (x2, f2)
    trace = [
        Step(0, xb, fb, step_size=b - a, info=_interval_info(a, b, x1, x2, f1, f2, [x1, x2], None))
    ]
    for xe, fe in ((x1, f1), (x2, f2)):
        if not math.isfinite(fe):
            return _result(
                "golden_section",
                xe,
                fe,
                False,
                _nonfinite("golden_section", xe, fe),
                0,
                trace,
                n_fev=f.n,
            )

    for k in range(1, max_iter + 1):
        if b - a <= _width_tol(xtol, a, b):
            return _result(
                "golden_section",
                xb,
                fb,
                True,
                _width_reason(b - a, xtol),
                k - 1,
                trace,
                n_fev=f.n,
                extra={"bracket": [a, b]},
            )
        if f1 <= f2:  # minimizer in [a, x2]: discard (x2, b]
            b = x2
            x2, f2 = x1, f1
            x1 = x2 - RHO * (x2 - a)
            f1 = fnew = f(x1)
            xnew, cut = x1, "right"
        else:  # minimizer in [x1, b]: discard [a, x1)
            a = x1
            x1, f1 = x2, f2
            x2 = x1 + RHO * (b - x1)
            f2 = fnew = f(x2)
            xnew, cut = x2, "left"
        xb, fb = (x1, f1) if f1 <= f2 else (x2, f2)
        trace.append(
            Step(k, xb, fb, step_size=b - a, info=_interval_info(a, b, x1, x2, f1, f2, [xnew], cut))
        )
        if not math.isfinite(fnew):
            return _result(
                "golden_section",
                xnew,
                fnew,
                False,
                _nonfinite("golden_section", xnew, fnew),
                k,
                trace,
                n_fev=f.n,
            )

    converged = b - a <= _width_tol(xtol, a, b)
    msg = _width_reason(b - a, xtol) if converged else f"reached max_iter={max_iter}"
    return _result(
        "golden_section",
        xb,
        fb,
        converged,
        msg,
        max_iter,
        trace,
        n_fev=f.n,
        extra={"bracket": [a, b]},
    )


# --------------------------------------------------------------------------------------
# Fibonacci search
# --------------------------------------------------------------------------------------


def fibonacci_numbers(target: float, *, n_max: int | None = None) -> list[int]:
    """F₀ = F₁ = 1, F_{j+1} = F_j + F_{j-1}, extended until n ≥ 3 and F_n ≥ target.

    Uses the indexing of Bazaraa et al. §8.2 (F₀ = F₁ = 1), so F_n = 1, 1, 2, 3, 5, 8, ...
    The list also stops (with F_n < target) at n = ``n_max`` (when given, ≥ 3) or once
    F_n > FIB_MAX, so a target of ``inf`` terminates.
    """
    fib = [1, 1]
    while len(fib) < 4 or (
        fib[-1] < target and (n_max is None or len(fib) <= n_max) and fib[-1] <= FIB_MAX
    ):
        fib.append(fib[-1] + fib[-2])
    return fib


@register(
    id="fibonacci_search",
    family="scalar",
    name="Fibonacci search",
    params=(
        XTOL_PARAM,
        ParamSpec(
            "eps_ratio",
            0.05,
            min=1e-3,
            max=0.5,
            log=True,
            help="Shift ε of the last step as a fraction of the final Fibonacci interval (b₀ − a₀)/Fₙ.",
        ),
        _max_iter_param(200),
    ),
    needs=("f", "bracket"),
    order="linear (optimal for a fixed number of evaluations)",
    summary="Plan n evaluations in advance so the final interval is as small as possible; "
    "points sit at Fibonacci ratios and each step reuses one old point.",
    references=(
        "Kiefer (1953), Sequential minimax search for a maximum, Proc. AMS 4, 502–506",
        "Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (Fibonacci search)",
        "Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (Fibonacci and golden section search)",
    ),
)
def fibonacci_search(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-8,
    eps_ratio: float = 0.05,
    max_iter: int = 200,
) -> Result:
    """Fibonacci search (Bazaraa et al. §8.2) with F₀ = F₁ = 1.

    Planning: with L₁ = b - a, choose the smallest n ≥ 3 with F_n ≥ (1 + r)·L₁/xtol', where
    r = ``eps_ratio``, ε = r·L₁/F_n and xtol' = xtol - s with s = 16ε_mach·max(|a|, |b|)
    (xtol' = xtol·(1 - 16ε_mach) when xtol ≤ 2s). The final interval then has length at
    most L₁/F_n + ε ≤ xtol' in exact arithmetic.

    # NOTE: the margin (xtol' instead of xtol) covers the rounding error of the computed
    # final width. Without it, F_n = (1 + r)L₁/xtol exactly gives a final width 1 ulp
    # above xtol (e.g. quadratic_1d, xtol = 0.1, r = 0.1: F₉ = 55).

    # NOTE: only max_iter + 2 evaluations are allowed, so the plan is capped at
    # n ≤ max_iter + 2 (and at F_n ≤ 2¹⁰⁰⁰, far below every floating-point resolution).
    # A capped plan is still a complete Fibonacci plan (the optimal one for that budget);
    # when its final width exceeds xtol the run reports converged=False.

    Setup: λ = a + (F_{n-2}/F_n) L₁, μ = a + (F_{n-1}/F_n) L₁ (two evaluations).
    Iteration k = 1, ..., n-3 (one evaluation each), with c = F_{n-k-2}/F_{n-k-1}:
    if f(λ) > f(μ) then a ← λ, λ ← μ, μ ← b - c(b - λ); else b ← μ, μ ← λ,
    λ ← a + c(μ - a). In exact arithmetic this is Bazaraa's placement
    μ = a + (F_{n-k-1}/F_{n-k})(b - a), λ = a + (F_{n-k-2}/F_{n-k})(b - a), and the width
    shrinks by F_{n-k}/F_{n-k+1}.

    # NOTE: the new point is placed relative to the reused point (as golden_section does),
    # not at a fixed fraction of [a, b]. Bazaraa's fixed-fraction form ignores the rounding
    # error of the reused point, and that error grows by a factor φ per step relative to
    # the shrinking bracket: on x² over (-1, 2) with xtol = 1e-30 the interior points
    # crossed at k = 103 and x* = 0 left the bracket. In the relative form the offsets
    # (λ - a, μ - λ, b - μ) are only scaled by c or 1 - c, so errors never grow.

    Last iteration k = n-2 (Bazaraa's Steps 2/3 at k = n-2 and Step 5 merged, one
    evaluation): the same reduction leaves the reused point p at the midpoint (ratio
    F₁/F₂ = ½), so the second point would coincide with it; instead evaluate q = p + ε and
    keep [p, b] if f(p) > f(q), else [a, q]. If ε is below the spacing of doubles at p,
    q is the next double after p.

    # NOTE: Bazaraa's Step 5 writes b_n = λ_n in the "else" case; we keep [a, μ_n] (with
    # μ_n = λ_n + ε), which is what the interval-elimination theorem guarantees.

    Counts: n_fev = 2 + n_iter; n_iter = n - 2 for a complete plan.

    Stopping test: the run performs its n - 2 planned iterations; it is converged when the
    final width ≤ max(xtol, 4ε_mach·max(|a|,|b|)). It stops early, before iteration k,
    when the bracket width has reached the floating-point floor 4ε_mach·max(|a|,|b|)
    (converged: no further step can be resolved), or, as guards that the floor test
    normally pre-empts, if rounding broke a < λ < μ < b (converged=False) or no double
    fits between p and b in the last step (the bracket is first reduced by the comparison
    of λ and μ; converged only if its width passes the test). A non-finite f stops with
    converged=False.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol, eps_ratio=eps_ratio)
    if eps_ratio >= 1.0:
        raise ValueError(
            f"eps_ratio must be < 1 (ε must be smaller than the last interval), got {eps_ratio}"
        )
    _require_max_iter(max_iter)
    f = Counted(prob.f)
    method = "fibonacci_search"

    L1 = b - a
    slack = FIB_SLACK_ULPS * EPS * max(abs(a), abs(b))
    # Below 2·slack, xtol is at the resolution of the initial bracket: then either the
    # floor stops the run (x* far from 0) or the final points are tiny and their
    # rounding errors are relative to the width, which the relative margin covers.
    plan_tol = xtol - slack if xtol > 2.0 * slack else xtol * (1.0 - FIB_SLACK_ULPS * EPS)
    target = (1.0 + eps_ratio) * L1 / plan_tol
    fib = fibonacci_numbers(target, n_max=max_iter + 2)
    n = len(fib) - 1
    capped = fib[n] < target
    eps = eps_ratio * L1 / fib[n]
    extra: dict[str, Any] = {"n_planned": n, "eps": eps}

    lam = a + (fib[n - 2] / fib[n]) * L1
    mu = a + (fib[n - 1] / fib[n]) * L1
    flam, fmu = f(lam), f(mu)

    def info(
        evaluated: list[float], cut: str | None, ratio: float | None, shift: float | None
    ) -> dict[str, Any]:
        out = _interval_info(a, b, lam, mu, flam, fmu, evaluated, cut)
        out["ratio"] = ratio
        out["eps"] = shift
        return out

    def best() -> tuple[float, float]:
        return (lam, flam) if flam <= fmu else (mu, fmu)

    def finish(converged: bool, msg: str) -> Result:
        extra["bracket"] = [a, b]
        return _result(method, xb, fb, converged, msg, trace[-1].k, trace, n_fev=f.n, extra=extra)

    xb, fb = best()
    trace = [Step(0, xb, fb, step_size=b - a, info=info([lam, mu], None, None, None))]
    for xe, fe in ((lam, flam), (mu, fmu)):
        if not math.isfinite(fe):
            return _result(
                method, xe, fe, False, _nonfinite(method, xe, fe), 0, trace, n_fev=f.n, extra=extra
            )

    n_iter_planned = n - 2
    for k in range(1, n_iter_planned + 1):
        if b - a <= RESOLUTION_ULPS * EPS * max(abs(a), abs(b)):
            msg = f"{_width_reason(b - a, xtol)} before the planned n = {n} evaluations"
            return finish(True, msg)
        if not a < lam < mu < b:
            msg = f"rounding broke the order a < λ < μ < b at k = {k - 1}; stopped"
            return finish(False, msg)
        if k < n_iter_planned:
            c = fib[n - k - 2] / fib[n - k - 1]
            if flam > fmu:  # discard [a, λ)
                a = lam
                lam, flam = mu, fmu
                mu = b - c * (b - lam)
                fmu = fnew = f(mu)
                xnew, cut = mu, "left"
            else:  # discard (μ, b]
                b = mu
                mu, fmu = lam, flam
                lam = a + c * (mu - a)
                flam = fnew = f(lam)
                xnew, cut = lam, "right"
            xb, fb = best()
            ratio = fib[n - k - 1] / fib[n - k]
            trace.append(Step(k, xb, fb, step_size=b - a, info=info([xnew], cut, ratio, None)))
        else:
            # Last iteration: reduce once more (the reused point p lands on the midpoint),
            # then compare p with the ε-shifted point q = p + ε.
            if flam > fmu:
                a_new, b_new, p, fp = lam, b, mu, fmu
            else:
                a_new, b_new, p, fp = a, mu, lam, flam
            # NOTE: an ε below the spacing of doubles at p would give q = p and no
            # information; any q > p is a valid comparison point, so use the next double.
            shift = eps if p + eps > p else math.nextafter(p, math.inf) - p
            q = p + shift
            if not q < b_new:
                # NOTE: the comparison of λ and μ is already made, so the reduced bracket
                # [a_new, b_new] (≤ 2 ulps of p wide) is valid; judge the width test on it.
                # Testing the previous bracket instead failed by a fraction of an ulp: on
                # (-7, -2) with xtol = 1.001·5/F₇₄ its width was 5 ulps against a floor of
                # 4ε·4.77 ≈ 4.8 ulps, so a run at floating-point resolution was unconverged.
                a, b = a_new, b_new
                msg = (
                    f"no point fits between p and b at width {b - a:.3g} (the bracket after the "
                    "last comparison); stopped"
                )
                return finish(b - a <= _width_tol(xtol, a, b), msg)
            a, b = a_new, b_new
            fq = fnew = f(q)
            xnew = q
            if fp > fq:  # discard [a, p)
                a, cut = p, "left"
            else:  # discard (q, b]
                b, cut = q, "right"
            lam, flam, mu, fmu = p, fp, q, fq
            xb, fb = best()
            trace.append(Step(k, xb, fb, step_size=b - a, info=info([xnew], cut, 0.5, shift)))
        if not math.isfinite(fnew):
            return _result(
                method,
                xnew,
                fnew,
                False,
                _nonfinite(method, xnew, fnew),
                k,
                trace,
                n_fev=f.n,
                extra=extra,
            )

    converged = b - a <= _width_tol(xtol, a, b)
    if converged:
        msg = f"{_width_reason(b - a, xtol)} after the planned n = {n} evaluations"
    elif capped:
        limit = (
            f"max_iter + 2 = {max_iter + 2} evaluations"
            if n == max_iter + 2
            else "a plan with F_n ≤ 2^1000"
        )
        msg = (
            f"xtol = {xtol:.3g} needs more than {limit}; the planned n = {n} point "
            f"Fibonacci plan ends with width {b - a:.3g} > xtol"
        )
    else:
        msg = f"final width {b - a:.3g} > xtol after n = {n} evaluations (rounding)"
    return finish(converged, msg)


# --------------------------------------------------------------------------------------
# Dichotomous search and ternary search (two new evaluations per iteration)
# --------------------------------------------------------------------------------------


def _rounding_tie(f1: float, f2: float) -> bool:
    """True when the comparison of f1 and f2 is decided by rounding (see TIE_ULPS)."""
    return abs(f1 - f2) <= TIE_ULPS * EPS * max(abs(f1), abs(f2))


def _two_point_search(
    method: str,
    prob: Problem,
    a: float,
    b: float,
    place: Callable[[float, float], tuple[float, float]],
    xtol: float,
    max_iter: int,
    *,
    stop_on_tie: bool = False,
) -> Result:
    """Shared loop: evaluate two new interior points per iteration and eliminate one end.

    ``place(a, b)`` returns the interior points (x1, x2) with x1 < x2. The reported x is
    the better point of the current pair (x1 on ties), which lies inside the current
    bracket. (An earlier, eliminated point can have a lower f when f is not unimodal; it is
    not reported, because it is outside the bracket.)

    With ``stop_on_tie`` the loop stops (converged=False) before an elimination whose
    comparison rounding decides (:func:`_rounding_tie`), so no end is discarded on noise.
    """
    f = Counted(prob.f)
    x1, x2 = place(a, b)
    f1, f2 = f(x1), f(x2)
    xb, fb = (x1, f1) if f1 <= f2 else (x2, f2)
    trace = [
        Step(0, xb, fb, step_size=b - a, info=_interval_info(a, b, x1, x2, f1, f2, [x1, x2], None))
    ]
    for xe, fe in ((x1, f1), (x2, f2)):
        if not math.isfinite(fe):
            return _result(method, xe, fe, False, _nonfinite(method, xe, fe), 0, trace, n_fev=f.n)

    for k in range(1, max_iter + 1):
        if b - a <= _width_tol(xtol, a, b):
            return _result(
                method,
                xb,
                fb,
                True,
                _width_reason(b - a, xtol),
                k - 1,
                trace,
                n_fev=f.n,
                extra={"bracket": [a, b]},
            )
        if stop_on_tie and _rounding_tie(f1, f2):
            msg = (
                f"comparison resolution reached at bracket width {b - a:.3g} > xtol: "
                f"f(x1) and f(x2), {x2 - x1:.3g} apart, agree to rounding, so neither end "
                "can be discarded (increase delta_ratio·xtol)"
            )
            return _result(
                method, xb, fb, False, msg, k - 1, trace, n_fev=f.n, extra={"bracket": [a, b]}
            )
        if f1 <= f2:  # minimizer in [a, x2]: discard (x2, b]
            b, cut = x2, "right"
        else:  # minimizer in [x1, b]: discard [a, x1)
            a, cut = x1, "left"
        x1, x2 = place(a, b)
        f1, f2 = f(x1), f(x2)
        xb, fb = (x1, f1) if f1 <= f2 else (x2, f2)
        trace.append(
            Step(
                k, xb, fb, step_size=b - a, info=_interval_info(a, b, x1, x2, f1, f2, [x1, x2], cut)
            )
        )
        for xe, fe in ((x1, f1), (x2, f2)):
            if not math.isfinite(fe):
                return _result(
                    method, xe, fe, False, _nonfinite(method, xe, fe), k, trace, n_fev=f.n
                )

    converged = b - a <= _width_tol(xtol, a, b)
    msg = _width_reason(b - a, xtol) if converged else f"reached max_iter={max_iter}"
    return _result(
        method, xb, fb, converged, msg, max_iter, trace, n_fev=f.n, extra={"bracket": [a, b]}
    )


@register(
    id="dichotomous_search",
    family="scalar",
    name="Dichotomous search",
    params=(
        ParamSpec(
            "xtol",
            1e-6,
            min=DICHOTOMOUS_XTOL_MIN,
            max=1e-1,
            log=True,
            help="Stop when the bracket width b − a ≤ xtol; then |x − x⋆| ≤ max(xtol, h) for "
            "unimodal f, with h ≈ √ε·(1 + |x⋆|). A small δ = delta_ratio·xtol can make rounding "
            "decide a comparison first: the run then stops with converged=False.",
        ),
        ParamSpec(
            "delta_ratio",
            0.1,
            min=1e-3,
            max=0.9,
            log=True,
            help="Separation δ = delta_ratio·xtol of the two points around the midpoint (must be "
            "< xtol and < the bracket width b − a). A smaller δ gives smaller f-differences that "
            "rounding hides sooner.",
        ),
        _max_iter_param(200),
    ),
    needs=("f", "bracket"),
    order="linear (rate ½ per two evaluations)",
    summary="Evaluate f just left and right of the midpoint and keep the half that "
    "contains the lower value; the bracket almost halves each step.",
    references=(
        "Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (dichotomous search)",
    ),
)
def dichotomous_search(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-6,
    delta_ratio: float = 0.1,
    max_iter: int = 200,
) -> Result:
    """Dichotomous search (Bazaraa et al. §8.2).

    Iteration: m = (a + b)/2, x₁ = m - δ/2, x₂ = m + δ/2 with δ = delta_ratio·xtol (the
    "distinguishability constant" 2ε in Bazaraa's notation); keep [a, x₂] if f(x₁) ≤ f(x₂),
    else [x₁, b]. The width obeys L_{k+1} = L_k/2 + δ/2, so L_k → δ < xtol and the search
    ends after about log₂((L₀ - δ)/(xtol - δ)) iterations.

    Precondition: δ < b - a, so the setup pair lies inside [a, b] (Bazaraa's points
    λ, μ = m ∓ ε lie in the interval); then L_k > δ for every k and every pair stays inside
    its bracket. A narrow bracket with δ ≥ b - a raises ValueError: otherwise f would be
    evaluated outside the user's bracket (possibly outside the domain of f) and x could be
    reported outside the final bracket.

    Counts: n_fev = 2(n_iter + 1) (each step, including k = 0, evaluates one new pair).

    Stopping test (converged): b - a ≤ max(xtol, 4ε·max(|a|,|b|)). Returns the better
    point of the last pair. A non-finite f-value stops with converged=False.

    Resolution: the pair compares f at points only δ apart, so f(x₂) - f(x₁) ≈ f''·d·δ
    when the midpoint is at distance d from x*. Rounding decides the comparison once
    d ≲ 4ε|f*|/(f''(x*) δ), which for δ ≪ h = √(8ε|f*|/f''(x*)) is far larger than h. The
    textbook tie rule (keep [a, x₂]) then discards the side that holds x* at random: on
    drug_concentration with xtol = 1e-10 and δ = 1e-13 the search finished "converged" with
    |x - x*| = 3.3e-3 and x* outside the final bracket. So, before each elimination, a
    comparison with |f(x₁) - f(x₂)| ≤ TIE_ULPS·ε·max(|f(x₁)|, |f(x₂)|) stops the run with
    converged=False ("comparison resolution reached"); the current bracket, which still
    holds x*, is returned with the better point of the pair (|x - x*| ≤ b - a).

    # NOTE: the tie stop is not in Bazaraa's method (exact arithmetic: f(x₁) = f(x₂) puts
    # x* in [x₁, x₂]). In floating point an exact tie cannot be told from a rounded one,
    # so it also stops a run whose midpoint is exactly x* by symmetry (x² on (-1, 1)), and
    # a constant f. Comparisons that rounding gets wrong by more than TIE_ULPS ulps are
    # not detected; on the library problems the bracket then still holds x* to within h.
    # A δ ≳ h keeps every rounding-decided comparison inside the floor h: the default
    # xtol = 1e-6 (δ = 1e-7) converges on every unimodal library problem, xtol = 1e-8 stops
    # unconverged on most of them. A δ below 4ε·max(|a|,|b|) raises ValueError (x₁ = x₂).

    # NOTE: the UI range starts at xtol = 1e-10 (not 1e-12 as for the other methods), so
    # every in-range δ ≥ 1e-3·1e-10 = 1e-13 is valid on brackets with max(|a|,|b|) ≤ 112
    # (all library problems); 1e-12 with delta_ratio = 1e-3 raised on 8 of 9 problems.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol, delta_ratio=delta_ratio)
    if delta_ratio >= 1.0:
        raise ValueError(f"delta_ratio must be < 1 (the width converges to δ), got {delta_ratio}")
    _require_max_iter(max_iter)
    half_delta = 0.5 * delta_ratio * xtol
    if 2.0 * half_delta <= RESOLUTION_ULPS * EPS * max(abs(a), abs(b)):
        raise ValueError(
            f"δ = delta_ratio·xtol = {2.0 * half_delta:.3g} is below floating-point resolution on "
            f"[{a}, {b}]: the two points would coincide and every comparison would be a tie"
        )
    if 2.0 * half_delta >= b - a:
        raise ValueError(
            f"δ = delta_ratio·xtol = {2.0 * half_delta:.3g} must be smaller than the bracket width "
            f"b - a = {b - a:.3g}: the points m ± δ/2 would lie outside [{a}, {b}] (reduce xtol "
            "or delta_ratio, or widen the bracket)"
        )

    def place(lo: float, hi: float) -> tuple[float, float]:
        m = lo + 0.5 * (hi - lo)
        # NOTE: the clamp only acts on rounding: δ < hi - lo puts m ± δ/2 inside [lo, hi] in
        # exact arithmetic, but for δ within a few ulps of hi - lo the rounded m ± δ/2 can
        # fall one ulp outside. Any a ≤ x₁ < x₂ ≤ b keeps the elimination rule valid.
        return max(m - half_delta, lo), min(m + half_delta, hi)

    return _two_point_search(
        "dichotomous_search", prob, a, b, place, xtol, max_iter, stop_on_tie=True
    )


@register(
    id="ternary_search",
    family="scalar",
    name="Ternary search",
    params=(XTOL_PARAM, _max_iter_param(200)),
    needs=("f", "bracket"),
    order="linear (rate 2/3 per two evaluations)",
    summary="Split the bracket into thirds, compare f at the two inner points and drop "
    "the outer third next to the worse one.",
    references=(
        "Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (interval elimination)",
    ),
)
def ternary_search(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-8,
    max_iter: int = 200,
) -> Result:
    """Ternary search: interval elimination with points at the thirds.

    Iteration: x₁ = a + (b - a)/3, x₂ = b - (b - a)/3; keep [a, x₂] if f(x₁) ≤ f(x₂),
    else [x₁, b]. No point is reused (neither old point is at a third of the new bracket),
    so each iteration costs two evaluations and multiplies the width by 2/3 — per
    evaluation (2/3)^{1/2} ≈ 0.816, worse than golden section's 0.618.

    Counts: n_fev = 2(n_iter + 1).

    Stopping test (converged): b - a ≤ max(xtol, 4ε·max(|a|,|b|)). Returns the better
    point of the last pair. A non-finite f-value stops with converged=False.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol)
    _require_max_iter(max_iter)

    def place(lo: float, hi: float) -> tuple[float, float]:
        third = (hi - lo) / 3.0
        return lo + third, hi - third

    return _two_point_search("ternary_search", prob, a, b, place, xtol, max_iter)


# --------------------------------------------------------------------------------------
# Successive parabolic interpolation (safeguarded)
# --------------------------------------------------------------------------------------


def parabola_vertex(xl: float, xm: float, xr: float, fl: float, fm: float, fr: float) -> float:
    """Vertex of the parabola through (xl, fl), (xm, fm), (xr, fr).

    NR (3rd ed.) eq. 10.3.1:
    u = xm - ½ [(xm-xl)²(fm-fr) - (xm-xr)²(fm-fl)] / [(xm-xl)(fm-fr) - (xm-xr)(fm-fl)].
    Returns nan when the denominator is zero (collinear points).
    """
    p = (xm - xl) * (fm - fr)
    q = (xm - xr) * (fm - fl)
    den = p - q
    if den == 0.0:
        return math.nan
    return xm - 0.5 * ((xm - xl) * p - (xm - xr) * q) / den


@register(
    id="parabolic_interpolation",
    family="scalar",
    name="Successive parabolic interpolation",
    params=(
        ParamSpec(
            "xtol",
            1e-8,
            min=1e-12,
            max=1e-1,
            log=True,
            help="Stop when xₘ is within xtol/2 of both ends of the bracket (so its width ≤ xtol). "
            "f-comparisons limit the accuracy to h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).",
        ),
        _max_iter_param(200),
    ),
    needs=("f", "bracket"),
    order="linear when an end point stays fixed (the unbracketed variant is superlinear, ≈ 1.324)",
    summary="Fit a parabola through three points that bracket the minimum and jump to its "
    "vertex; fall back to golden-section steps when the fit is unusable or makes too little "
    "progress, and to bisection until the three points bracket the minimum. The setup "
    "evaluates f at both bracket ends, so f must be finite at a and b (choose a bracket "
    "inside the domain; Brent's method evaluates interior points only).",
    references=(
        "Press et al., Numerical Recipes (3rd ed.), §10.3, eq. 10.3.1",
        "Brent (1973), Algorithms for Minimization without Derivatives, Ch. 5 (progress test)",
        "Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (line search by quadratic fit)",
        "Antoniou & Lu, Practical Optimization (2007), Ch. 4 (quadratic interpolation method)",
    ),
)
def parabolic_interpolation(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-8,
    max_iter: int = 200,
) -> Result:
    """Successive parabolic interpolation with a three-point pattern, safeguarded.

    State: x_l < x_m < x_r. Setup evaluates f at a, (a+b)/2 and b (three evaluations).

    Precondition: f is finite at the bracket END points a and b. The textbook method
    (NR §10.3, Luenberger & Ye Ch. 8) works on a triple whose ends carry known f-values, so
    unlike golden section, Fibonacci, dichotomous, ternary search and Brent it evaluates
    f(a) and f(b). A bracket that ends on a pole or a domain limit (e.g. (-1, 6) for
    rational_1d, pole at -1) therefore stops at k = 0 with converged=False and a message
    that names the end point; choose a bracket strictly inside the domain.

    Each iteration evaluates ONE new point u:

    1. If the three-point pattern f(x_m) ≤ min(f(x_l), f(x_r)) fails ("bisect"): by the
       interval-elimination theorem keep [x_l, x_m] if f(x_l) ≤ f(x_r), else [x_m, x_r],
       and let u be the midpoint of the kept interval.
    2. Otherwise ("parabolic"): u = vertex of the parabola through the triple (NR eq.
       10.3.1). Under the pattern the parabola is convex and u ∈ [x_l, x_r]. If the
       points are collinear/flat, rounding puts u outside (x_l, x_r), or the progress
       test |u - x_m| < ½·(step before last) fails, take a "golden" step
       u = x_m ± ρ·(larger segment) instead. The step of an iteration is |vertex - x_m|
       for parabolic and probe steps (the unadjusted vertex step, as in FMM fmin) and
       |u - x_m| otherwise; the first two iterations have no bound.
       If |u - x_m| < δ = tol/2 (tol as below), evaluate the "probe" u = x_m ± δ
       instead, on the vertex's side (the larger segment's side if u = x_m), or on the
       other side if the segment on that side is not longer than δ. If rounding puts
       x_m ± δ on the end point, the probe is the midpoint of that segment.
    3. Replace one point so the pattern holds again (Luenberger & Ye, Ch. 8): if f(u) ≤ f(x_m),
       u becomes the middle point; otherwise u becomes the end point on its side.

    # NOTE: the textbook test "stop when |u - x_m| ≤ xtol" is not used: it is fooled by
    # symmetric data (on x_log_x the second vertex equals x_m = 0.5 exactly, 0.13 away from
    # x* = 1/e). The probe of step 2 (Brent's minimum-step device) always adds information,
    # so the guaranteed width test below is reached instead.

    Stopping test (converged): max(x_m - x_l, x_r - x_m) ≤ δ = tol/2 with
    tol = max(xtol, 4ε·max(|x_l|,|x_r|)), so the bracket width is ≤ tol and, for unimodal
    f, |x - x*| ≤ tol. (While the pattern is being established x_m is the midpoint and the
    test is just "width ≤ tol".) The returned x is the triple's best point. A non-finite f
    stops with converged=False.

    # NOTE: the progress test of step 2 is Brent's (1973, Ch. 5), not part of the textbook
    # method. Without it a vertex step is accepted however little it shrinks the bracket:
    # on exp(5(x - c)) - 5(x - c) the steep far end stays fixed and x_m creeps toward x*
    # (|x_m - x*| = 0.31 after 100 steps on (0, 10)). With it, parabolic steps must halve
    # every two iterations, else golden steps cut the larger segment.

    Convergence: the classical analysis (order ≈ 1.324) uses the three most recent points.
    Keeping a bracketing triple instead often leaves one end point fixed for the whole run
    (e.g. x_l = 0 on drug_concentration), and then the convergence is only linear. On a
    kink with a quadratic and a linear side the method can still need several times more
    evaluations than golden section.

    Counts: n_fev = 3 + n_iter.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol)
    _require_max_iter(max_iter)
    f = Counted(prob.f)
    method = "parabolic_interpolation"

    xl, xm, xr = a, a + 0.5 * (b - a), b
    fl, fm, fr = f(xl), f(xm), f(xr)

    def best() -> tuple[float, float]:
        pts = ((xm, fm), (xl, fl), (xr, fr))
        return min(pts, key=lambda t: t[1])

    def info(
        step: str,
        trial: float | None,
        f_trial: float | None,
        parab: dict[str, Any] | None,
        vertex: float | None,
        max_step: float | None,
    ) -> dict[str, Any]:
        return {
            "bracket": [xl, xr],
            "triple": [xl, xm, xr],
            "f_triple": [fl, fm, fr],
            "pattern": bool(fm <= fl and fm <= fr),
            "step": step,
            "trial": trial,
            "f_trial": f_trial,
            "parabola": parab,
            "vertex": vertex,
            "max_step": max_step,
        }

    # |u - x_m| of the last two iterations (Brent's d and e); inf until two steps exist.
    last_step = step_before_last = math.inf
    trace = [Step(0, xm, fm, step_size=xr - xl, info=info("init", None, None, None, None, None))]
    for xe, fe, where in ((xl, fl, "a"), (xm, fm, ""), (xr, fr, "b")):
        if not math.isfinite(fe):
            msg = _nonfinite(method, xe, fe)
            if where:
                msg = (
                    f"{method}: f({xe:.17g}) = {fe} at the bracket end point {where} is not "
                    "finite; this method evaluates f at both bracket ends, so choose a bracket "
                    "strictly inside the domain of f; stopped"
                )
            return _result(method, xe, fe, False, msg, 0, trace, n_fev=f.n)
    xb, fb = best()
    trace[0] = Step(0, xb, fb, step_size=xr - xl, info=trace[0].info)

    for k in range(1, max_iter + 1):
        width = xr - xl
        delta = 0.5 * _width_tol(xtol, xl, xr)
        if max(xm - xl, xr - xm) <= delta:
            return _result(
                method,
                xb,
                fb,
                True,
                _width_reason(width, xtol),
                k - 1,
                trace,
                n_fev=f.n,
                extra={"bracket": [xl, xr]},
            )
        parab: dict[str, Any] | None = None
        vertex: float | None = None
        max_step = 0.5 * step_before_last
        xm_old = xm
        if not (fm <= fl and fm <= fr):
            step = "bisect"
            if fl <= fr:  # f(x_m) > f(x_l): minimizer in [x_l, x_m]
                xr, fr = xm, fm
            else:  # f(x_m) > f(x_r): minimizer in [x_m, x_r]
                xl, fl = xm, fm
            u = xl + 0.5 * (xr - xl)
            fu = f(u)
            xm, fm = u, fu
        else:
            parab = _parabola(xm, fm, xl, fl, xr, fr)
            u = parabola_vertex(xl, xm, xr, fl, fm, fr)
            if math.isfinite(u) and xl < u < xr and abs(u - xm) < max_step:
                vertex = u
                step = "parabolic"
                if abs(u - xm) < delta:
                    # NOTE: minimum step (Brent 1973, Ch. 5): never evaluate closer than δ to
                    # x_m, so every evaluation can shrink the bracket.
                    step = "probe"
                    right = u > xm if u != xm else xr - xm > xm - xl
                    if right and xr - xm <= delta:
                        right = False
                    elif not right and xm - xl <= delta:
                        right = True
                    u = xm + delta if right else xm - delta
                    if not xl < u < xr:  # rounding: use the segment midpoint
                        u = xm + 0.5 * (xr - xm) if right else xm - 0.5 * (xm - xl)
            else:
                # NOTE: safeguard: collinear/flat triple, rounding put the vertex outside
                # (x_l, x_r), or the vertex step is not less than half the step before
                # last (Brent 1973, Ch. 5: the progress test of localmin); take a
                # golden-section step into the larger segment instead.
                vertex = u if math.isfinite(u) else None
                step = "golden"
                u = xm - RHO * (xm - xl) if xm - xl > xr - xm else xm + RHO * (xr - xm)
            if not xl < u < xr or u == xm:
                msg = "the trial point cannot leave x_m in floating point; xtol is below resolution"
                return _result(
                    method, xb, fb, False, msg, k - 1, trace, n_fev=f.n, extra={"bracket": [xl, xr]}
                )
            fu = f(u)
            if math.isfinite(fu):
                if u < xm:
                    if fu <= fm:
                        xr, fr, xm, fm = xm, fm, u, fu
                    else:
                        xl, fl = u, fu
                elif fu <= fm:
                    xl, fl, xm, fm = xm, fm, u, fu
                else:
                    xr, fr = u, fu
        # NOTE: as in FMM fmin, a probe records the vertex step |vertex - x_m|, not δ:
        # otherwise a run of probes would pass the progress test forever.
        moved = abs(
            (vertex if step in ("parabolic", "probe") and vertex is not None else u) - xm_old
        )
        step_before_last, last_step = last_step, moved
        xb, fb = best()
        trace.append(
            Step(
                k,
                xb,
                fb,
                step_size=xr - xl,
                info=info(
                    step, u, fu, parab, vertex, max_step if math.isfinite(max_step) else None
                ),
            )
        )
        if not math.isfinite(fu):
            return _result(method, u, fu, False, _nonfinite(method, u, fu), k, trace, n_fev=f.n)

    width = xr - xl
    converged = max(xm - xl, xr - xm) <= 0.5 * _width_tol(xtol, xl, xr)
    msg = _width_reason(width, xtol) if converged else f"reached max_iter={max_iter}"
    return _result(
        method, xb, fb, converged, msg, max_iter, trace, n_fev=f.n, extra={"bracket": [xl, xr]}
    )


# --------------------------------------------------------------------------------------
# Brent's method (localmin)
# --------------------------------------------------------------------------------------


@register(
    id="brent_minimize",
    family="scalar",
    name="Brent's method (minimization)",
    params=(
        ParamSpec(
            "xtol",
            1e-8,
            min=1e-12,
            max=1e-1,
            log=True,
            help="Absolute tolerance t: stop when x is within 2(rtol·|x| + xtol/3) of both bracket "
            "ends. f-comparisons limit the accuracy to h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).",
        ),
        ParamSpec(
            "rtol",
            SQRT_EPS,
            min=1e-10,
            max=1e-3,
            log=True,
            help="Relative tolerance (Brent's eps; √ε by default). Below √ε the guarantee "
            "2(rtol·|x| + xtol/3) can fail: f-comparisons resolve x⋆ only to about √ε·(1 + |x⋆|).",
        ),
        _max_iter_param(500),
    ),
    needs=("f", "bracket"),
    order="superlinear (≈ 1.324) for smooth f; guaranteed convergence otherwise",
    summary="Golden-section search that switches to parabolic interpolation whenever the "
    "parabola is trustworthy; the default 1-D minimizer in most libraries. Its accuracy is "
    "2(rtol·|x| + xtol/3), so far from 0 the relative part rtol·|x| dominates xtol.",
    references=(
        "Brent (1973), Algorithms for Minimization without Derivatives, Ch. 5 (localmin)",
        "Forsythe, Malcolm & Moler (1977), Computer Methods for Mathematical Computations, fmin",
        "Press et al., Numerical Recipes (3rd ed.), §10.3 (routine Brent)",
    ),
)
def brent_minimize(
    problem: ScalarProblem,
    *,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-8,
    rtol: float = SQRT_EPS,
    max_iter: int = 500,
) -> Result:
    """Brent's localmin on [a, b] (Brent 1973 Ch. 5; FMM ``fmin``; NR3 §10.3).

    State: a < b (the bracket), x (best point), w (second best), v (previous w), d (last
    step), e (the step before last). Start: x = w = v = a + ρ(b - a), d = e = 0.

    Each iteration, with m = (a + b)/2, tol = rtol·|x| + xtol/3, t2 = 2 tol:

    1. Stop (converged) if |x - m| ≤ t2 - (b - a)/2, i.e. max(x - a, b - x) ≤ 2·tol.
    2. If |e| > tol, fit a parabola through x, w, v: r = (x-w)(f(x)-f(v)),
       q = (x-v)(f(x)-f(w)), p = (x-v)q - (x-w)r, q = 2(q - r); make q ≥ 0 (flip p);
       then e ← d. Accept the parabolic step d = p/q when |p| < |½ q e_old|, i.e. the
       step is less than half the step before last, and x + p/q ∈ (a, b); if u is within
       t2 of a or b, use d = ±tol toward m.
    3. Otherwise take a golden-section step into the larger part: e = (b - x if x < m
       else a - x), d = ρ e.
    4. Never evaluate closer than tol to x: u = x + d if |d| ≥ tol, else x ± tol.
    5. Update (a, b, v, w, x) from f(u) as in FMM fmin.

    Counts: n_fev = 1 + n_iter.

    Stopping test: step 1; max_iter or a non-finite f gives converged=False.

    Accuracy: on convergence x is within 2·tol = 2(rtol·|x| + xtol/3) of both bracket
    ends, so |x - x*| ≤ max(2·tol, h) for unimodal f, where h ≈ √(8ε|f*|/f''(x*)) is the
    accuracy floor of f-comparisons (module docstring). This is NOT xtol: the relative part
    rtol·|x| dominates far from 0. On (1e8, 1e8 + 1) with xtol = 1e-12, 2·tol ≈ 2.98
    exceeds the bracket width, so the method stops at k = 0 after one evaluation, 0.13
    from x* (exactly as SciPy's ``minimize_scalar(method="bounded")``). A smaller rtol
    (down to ~ε) gives an absolute-like test there, because f* = 0 makes h tiny. When
    f* ≠ 0, an rtol below √ε asks for more than comparisons can give: with xtol = 1e-12 and
    rtol = 1e-10 on drug_concentration, 2·tol = 4e-10 but |x - x*| = 1.05e-8 ≈ h/9, and x*
    is outside the final bracket.
    """
    prob = scalar_problem(problem)
    a, b = _resolve_bracket(prob, bracket)
    _require_positive(xtol=xtol, rtol=rtol)
    _require_max_iter(max_iter)
    f = Counted(prob.f)
    method = "brent_minimize"

    x = w = v = a + RHO * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    tol = rtol * abs(x) + xtol / 3.0

    def info(
        step: str,
        u: float | None,
        fu: float | None,
        parab: dict[str, Any] | None,
        vertex: float | None,
    ) -> dict[str, Any]:
        return {
            "bracket": [a, b],
            "xwv": [x, w, v],
            "f_xwv": [fx, fw, fv],
            "step": step,
            "trial": u,
            "f_trial": fu,
            "parabola": parab,
            "vertex": vertex,
            "tol": tol,
        }

    trace = [Step(0, x, fx, step_size=b - a, info=info("init", None, None, None, None))]
    if not math.isfinite(fx):
        return _result(method, x, fx, False, _nonfinite(method, x, fx), 0, trace, n_fev=f.n)

    for k in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        tol = rtol * abs(x) + xtol / 3.0
        t2 = 2.0 * tol
        if abs(x - m) <= t2 - 0.5 * (b - a):
            msg = f"x is within 2·tol = {t2:.3g} of both bracket ends"
            return _result(
                method, x, fx, True, msg, k - 1, trace, n_fev=f.n, extra={"bracket": [a, b]}
            )
        parab: dict[str, Any] | None = None
        vertex: float | None = None
        p = q = r = 0.0
        if abs(e) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
            parab = _parabola(x, fx, w, fw, v, fv)
            if q != 0.0:
                vertex = x + p / q
        if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
            step = "parabolic"
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:  # f must not be evaluated too close to a or b
                d = tol if x <= m else -tol  # FMM: dsign(tol1, xm - x)
        else:
            step = "golden"
            e = b - x if x < m else a - x
            d = RHO * e
        u = x + d if abs(d) >= tol else x + (tol if d >= 0.0 else -tol)  # FMM: dsign(tol1, d)
        x_prev = x
        fu = f(u)
        if not math.isfinite(fu):
            trace.append(
                Step(k, x, fx, step_size=abs(u - x), info=info(step, u, fu, parab, vertex))
            )
            return _result(method, u, fu, False, _nonfinite(method, u, fu), k, trace, n_fev=f.n)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        trace.append(
            Step(k, x, fx, step_size=abs(u - x_prev), info=info(step, u, fu, parab, vertex))
        )

    m = 0.5 * (a + b)
    tol = rtol * abs(x) + xtol / 3.0
    converged = abs(x - m) <= 2.0 * tol - 0.5 * (b - a)
    msg = (
        f"x is within 2·tol = {2.0 * tol:.3g} of both bracket ends"
        if converged
        else f"reached max_iter={max_iter}"
    )
    return _result(
        method, x, fx, converged, msg, max_iter, trace, n_fev=f.n, extra={"bracket": [a, b]}
    )


# --------------------------------------------------------------------------------------
# Newton's method for minimization
# --------------------------------------------------------------------------------------


@register(
    id="newton_1d",
    family="scalar",
    name="Newton's method (1-D minimization)",
    params=(
        ParamSpec(
            "gtol",
            1e-8,
            min=1e-14,
            max=1e-2,
            log=True,
            help="Stop when |f′(x)| ≤ gtol and f″(x) ≥ 0 (necessary conditions only: x can be "
            "an inflection point).",
        ),
        ParamSpec(
            "alpha0",
            1.0,
            min=1e-4,
            max=10.0,
            log=True,
            help="First trial length of the gradient step used when f″(x) ≤ 0.",
        ),
        _max_iter_param(100),
    ),
    needs=("f", "grad", "hess"),
    order="quadratic near a minimizer with f″ > 0",
    summary="Jump to the minimizer of the local quadratic model x − f′/f″; when the "
    "model is not convex (f″ ≤ 0), take a backtracking gradient step instead.",
    references=(
        "Nocedal & Wright, Numerical Optimization (2nd ed.), §3.3 (Newton) and Alg. 3.1 (backtracking)",
        "Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (Newton's method for line search)",
    ),
)
def newton_1d(
    problem: ScalarProblem,
    *,
    x0: float | None = None,
    gtol: float = 1e-8,
    alpha0: float = 1.0,
    max_iter: int = 100,
) -> Result:
    """Newton's method for 1-D minimization with a negative-curvature safeguard.

    Newton step (f''(x_k) > 0): x_{k+1} = x_k - f'(x_k)/f''(x_k), the minimizer of the
    Taylor model m(z) = f(x_k) + f'(x_k)(z - x_k) + ½ f''(x_k)(z - x_k)².

    # NOTE: safeguard (not in the pure method): when f''(x_k) ≤ 0 the model has no
    # minimizer and the Newton step would head for a maximum or an inflection point. We
    # then take a steepest-descent step x_k - α f'(x_k) with Armijo backtracking (N&W
    # Alg. 3.1: α = alpha0, halve until f(x_k - α f') ≤ f(x_k) - c₁ α f'², c₁ = 1e-4) and
    # mark it ``step = "gradient"``.

    Stopping test (converged): |f'(x_k)| ≤ gtol and f''(x_k) ≥ 0, i.e. x_k satisfies the
    first- and second-order NECESSARY conditions to tolerance. That does not make x_k a
    minimizer: near an inflection point of a function with no minimizer the test also
    passes (f = x³ from x0 = 1 stops at x ≈ 3e-5 with converged=True), and a flat
    minimizer (f'' = 0 at x*) can satisfy it far from x*, because f' is tiny on a wide
    region. If |f'| ≤ gtol but f'' < 0, x_k is near a local maximum: stop with
    converged=False. Non-finite values, a failed backtracking search, and max_iter also
    give converged=False.

    Counts: one f, f' and f'' evaluation per iterate, plus the extra f evaluations of the
    backtracking trials. Without an analytic f' or f'', central differences on f are used
    (core.diff) and their f evaluations are counted in n_fev.
    """
    prob = scalar_problem(problem)
    _require_positive(gtol=gtol, alpha0=alpha0)
    _require_max_iter(max_iter)
    x = start_scalar(prob, x0)
    if not math.isfinite(x):
        raise ValueError(f"x0 must be finite, got {x}")
    method = "newton_1d"
    f = Counted(prob.f)
    g_fn: Func = prob.grad if prob.grad is not None else (lambda z: diff.derivative(f, z))
    h_fn: Func = prob.hess if prob.hess is not None else (lambda z: diff.second_derivative(f, z))
    g = Counted(g_fn)
    h = Counted(h_fn)

    def finish(converged: bool, msg: str, n_iter: int) -> Result:
        return _result(
            method,
            x,
            fx,
            converged,
            msg,
            n_iter,
            trace,
            n_fev=f.n,
            n_gev=g.n if prob.grad is not None else 0,
            n_hev=h.n if prob.hess is not None else 0,
        )

    fx, gx, hx = f(x), g(x), h(x)
    trace = [
        Step(
            0,
            x,
            fx,
            grad_norm=abs(gx),
            info={"step": "init", "g": gx, "h": hx, "parabola": None, "alpha": None, "trials": []},
        )
    ]
    if not (math.isfinite(fx) and math.isfinite(gx) and math.isfinite(hx)):
        return finish(False, f"non-finite f, f' or f'' at x0 = {x:.17g}", 0)

    for k in range(1, max_iter + 1):
        if abs(gx) <= gtol:
            if hx >= 0.0:
                return finish(True, f"|f'(x)| = {abs(gx):.3g} ≤ gtol and f''(x) ≥ 0", k - 1)
            msg = f"|f'(x)| ≤ gtol but f''(x) = {hx:.3g} < 0: x is near a local maximum"
            return finish(False, msg, k - 1)
        model = {"center": x, "coef": [fx, gx, 0.5 * hx]}
        trials: list[list[float]] = []
        alpha: float | None = None
        if hx > 0.0:
            step = "newton"
            x_new = x - gx / hx
            f_new = f(x_new)
        else:
            step = "gradient"
            alpha = alpha0
            accepted = False
            x_new, f_new = x, fx
            for _ in range(MAX_BACKTRACK):
                x_new = x - alpha * gx
                f_new = f(x_new)
                trials.append([alpha, f_new])
                if math.isfinite(f_new) and f_new <= fx - ARMIJO_C1 * alpha * gx * gx:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                msg = f"gradient-step backtracking failed after {MAX_BACKTRACK} halvings at x = {x:.17g}"
                return finish(False, msg, k - 1)
        g_new, h_new = g(x_new), h(x_new)
        trace.append(
            Step(
                k,
                x_new,
                f_new,
                grad_norm=abs(g_new),
                step_size=abs(x_new - x),
                info={
                    "step": step,
                    "g": g_new,
                    "h": h_new,
                    "parabola": model,
                    "alpha": alpha,
                    "trials": trials,
                },
            )
        )
        x, fx, gx, hx = x_new, f_new, g_new, h_new
        if not (math.isfinite(fx) and math.isfinite(gx) and math.isfinite(hx)):
            return finish(False, f"non-finite f, f' or f'' at x = {x:.17g} (left the domain?)", k)

    converged = abs(gx) <= gtol and hx >= 0.0
    msg = (
        f"|f'(x)| = {abs(gx):.3g} ≤ gtol and f''(x) ≥ 0"
        if converged
        else f"reached max_iter={max_iter}"
    )
    return finish(converged, msg, max_iter)


# --------------------------------------------------------------------------------------
# Bracketing a minimum (NR mnbrak)
# --------------------------------------------------------------------------------------


@register(
    id="bracket_minimum",
    family="scalar",
    name="Bracketing a minimum",
    params=(
        ParamSpec(
            "step",
            0.1,
            min=1e-6,
            max=10.0,
            log=True,
            help="First step: the second point is x₀ + step.",
        ),
        ParamSpec(
            "grow_limit",
            100.0,
            min=1.5,
            max=1000.0,
            log=True,
            help="A parabolic extrapolation may reach at most uₗᵢₘ = b + grow_limit·(c − b) "
            "(must be > 1, so uₗᵢₘ lies beyond c).",
        ),
        _max_iter_param(50),
    ),
    needs=("f",),
    order="geometric expansion (ratio φ)",
    summary="Walk downhill with steps that grow by the golden ratio (or a parabolic jump) "
    "until f turns up: the last three points bracket a minimum.",
    references=(
        "Press et al., Numerical Recipes (3rd ed.), §10.1 (Bracketmethod::bracket, mnbrak)",
    ),
)
def bracket_minimum(
    problem: ScalarProblem,
    *,
    x0: float | None = None,
    step: float = 0.1,
    grow_limit: float = 100.0,
    max_iter: int = 50,
) -> Result:
    """Find a < b < c (or a > b > c) with f(b) ≤ f(a) and f(b) ≤ f(c) (NR3 §10.1).

    Start: a = x0, b = x0 + step; swap them if f(b) > f(a) so that a → b is downhill;
    c = b + φ(b - a) (φ = GOLD = 1.618034 as in NR). While f(b) > f(c), one iteration:
    try the vertex u of the parabola through (a, b, c), limited to
    u_lim = b + grow_limit·(c - b):

    * u between b and c ("parabolic"): if f(u) < f(c) the bracket is (b, u, c); if
      f(u) > f(b) it is (a, b, u); otherwise u is useless and u = c + φ(c - b) ("golden").
    * u between c and u_lim ("parabolic_far"): if f(u) < f(c), move on with
      (b, c, u) and evaluate a further u = u + φ(u - c).
    * u beyond u_lim ("limit"): u = u_lim.
    * otherwise ("golden"): u = c + φ(c - b).

    Then shift (a, b, c) ← (b, c, u). An iteration evaluates one or two points; every
    evaluation is listed in ``info["trials"]``.

    Stopping test (converged): f(b) ≤ f(c) with a, b, c distinct (a bracket was found);
    converged means "a bracket exists", not "x is a minimizer". max_iter (f may be
    unbounded below) or a non-finite f (e.g. leaving the domain) gives converged=False, and
    so does a triple that rounding collapsed (two equal points: f(b) ≤ f(c) then holds
    trivially). Result.x is the middle point b. Result.extra["triple"] / ["f_triple"] hold
    the last triple (a, b, c) sorted increasingly on every return path, also on failure
    (then it is the current search triple, not a bracket); they are absent only when
    f(x0) or f(x0 + step) is not finite (no triple was formed).

    Invalid input (ValueError): x0 + step == x0 in floating point (the step is lost in
    rounding, as SciPy's ``bracket`` also rejects), or grow_limit ≤ 1 (u_lim = b +
    grow_limit·(c - b) must lie beyond c; at grow_limit = 1 the "limit" step re-evaluates c).
    """
    prob = scalar_problem(problem)
    _require_max_iter(max_iter)
    if not (math.isfinite(step) and step != 0.0):
        raise ValueError(f"step must be a non-zero finite number, got {step}")
    if not (math.isfinite(grow_limit) and grow_limit > 1.0):
        raise ValueError(f"grow_limit must be a finite number > 1, got {grow_limit}")
    xa = start_scalar(prob, x0)
    if not math.isfinite(xa):
        raise ValueError(f"x0 must be finite, got {xa}")
    method = "bracket_minimum"
    f = Counted(prob.f)

    xb = xa + step
    if xb == xa:
        raise ValueError(
            f"step = {step:.3g} is lost in rounding at x0 = {xa:.17g} (x0 + step == x0); "
            "use a larger step"
        )
    fa, fb = f(xa), f(xb)
    init_trials = [[xa, fa], [xb, fb]]
    if not (math.isfinite(fa) and math.isfinite(fb)):
        bad, fbad = (xa, fa) if not math.isfinite(fa) else (xb, fb)
        trace = [
            Step(
                0,
                xa,
                fa,
                info={
                    "bracket": [min(xa, xb), max(xa, xb)],
                    "triple": [xa, xb, xb],
                    "f_triple": [fa, fb, fb],
                    "step": "init",
                    "trials": init_trials,
                },
            )
        ]
        return _result(method, bad, fbad, False, _nonfinite(method, bad, fbad), 0, trace, n_fev=f.n)
    if fb > fa:  # make a → b the downhill direction
        xa, xb, fa, fb = xb, xa, fb, fa
    xc = xb + GOLD * (xb - xa)
    fc = f(xc)
    init_trials.append([xc, fc])

    def info(kind: str, trials: list[list[float]]) -> dict[str, Any]:
        return {
            "bracket": [min(xa, xc), max(xa, xc)],
            "triple": [xa, xb, xc],
            "f_triple": [fa, fb, fc],
            "step": kind,
            "trials": trials,
        }

    def sorted_triple() -> dict[str, Any]:
        triple = sorted([(xa, fa), (xb, fb), (xc, fc)])
        return {"triple": [t[0] for t in triple], "f_triple": [t[1] for t in triple]}

    def done(k: int) -> Result:
        extra = sorted_triple()
        lo, mid, hi = extra["triple"]
        if not lo < mid < hi:
            # NOTE: guard (not in NR): with two equal points f(b) ≤ f(c) holds trivially.
            msg = (
                f"the triple collapsed in floating point ({xa:.17g}, {xb:.17g}, {xc:.17g}): "
                "two points are equal, so f(b) ≤ f(c) is no bracket"
            )
            return _result(method, xb, fb, False, msg, k, trace, n_fev=f.n, extra=extra)
        msg = f"bracket found: f(b) ≤ f(a) and f(b) ≤ f(c) with width {abs(xc - xa):.3g}"
        return _result(method, xb, fb, True, msg, k, trace, n_fev=f.n, extra=extra)

    trace = [Step(0, xb, fb, step_size=abs(xb - xa), info=info("init", init_trials))]
    if not math.isfinite(fc):
        return _result(
            method,
            xc,
            fc,
            False,
            _nonfinite(method, xc, fc),
            0,
            trace,
            n_fev=f.n,
            extra=sorted_triple(),
        )

    for k in range(1, max_iter + 1):
        if fb <= fc:
            return done(k - 1)
        r = (xb - xa) * (fb - fc)
        q = (xb - xc) * (fb - fa)
        qr = q - r
        den = 2.0 * (max(abs(qr), TINY) if qr >= 0.0 else -max(abs(qr), TINY))
        u = xb - ((xb - xc) * q - (xb - xa) * r) / den
        ulim = xb + grow_limit * (xc - xb)
        trials: list[list[float]] = []
        ev = partial(_evaluate_logged, f, trials)
        found = False
        if (xb - u) * (u - xc) > 0.0:  # parabolic u between b and c
            kind = "parabolic"
            fu = ev(u)
            if not math.isfinite(fu):
                pass  # reported below as a non-finite stop
            elif fu < fc:  # minimum between b and c
                xa, xb, fa, fb = xb, u, fb, fu
                found = True
            elif fu > fb:  # minimum between a and u
                xc, fc = u, fu
                found = True
            else:  # the parabola did not help: default magnification
                kind = "golden"
                u = xc + GOLD * (xc - xb)
                fu = ev(u)
        elif (xc - u) * (u - ulim) > 0.0:  # parabolic u between c and its limit
            kind = "parabolic_far"
            fu = ev(u)
            if math.isfinite(fu) and fu < fc:
                xb, xc, u = xc, u, u + GOLD * (u - xc)
                fb, fc = fc, fu
                fu = ev(u)
        elif (u - ulim) * (ulim - xc) >= 0.0:  # limit u to its maximum allowed value
            kind = "limit"
            u = ulim
            fu = ev(u)
        else:  # reject the parabolic u: default magnification
            kind = "golden"
            u = xc + GOLD * (xc - xb)
            fu = ev(u)
        if not math.isfinite(fu):
            trace.append(Step(k, xb, fb, step_size=abs(u - xb), info=info(kind, trials)))
            return _result(
                method,
                u,
                fu,
                False,
                _nonfinite(method, u, fu),
                k,
                trace,
                n_fev=f.n,
                extra=sorted_triple(),
            )
        if not found:
            xa, xb, xc = xb, xc, u
            fa, fb, fc = fb, fc, fu
        trace.append(Step(k, xb, fb, step_size=abs(xc - xa), info=info(kind, trials)))
        if found:
            return done(k)

    if fb <= fc:
        return done(max_iter)
    msg = f"no bracket after max_iter={max_iter} expansions (f may be unbounded below in this direction)"
    return _result(
        method,
        xb,
        fb,
        False,
        msg,
        max_iter,
        trace,
        n_fev=f.n,
        extra=sorted_triple(),
    )


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("golden_section", "sin_1d", {}),
    ("fibonacci_search", "drug_concentration", {"xtol": 1e-6}),
    ("dichotomous_search", "rational_1d", {"xtol": 1e-6}),
    ("ternary_search", "x_log_x", {"xtol": 1e-6}),
    ("parabolic_interpolation", "quartic_1d", {}),
    ("brent_minimize", "multimodal_1d", {}),
    ("newton_1d", "sin_1d", {"x0": 3.0}),
    ("bracket_minimum", "drug_concentration", {}),
]
