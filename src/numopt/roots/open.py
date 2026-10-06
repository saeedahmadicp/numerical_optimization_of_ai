"""Open root finders for f(x) = 0: start from a point, no bracket, no convergence guarantee.

Open methods converge fast near a simple root (Newton quadratically, Halley cubically,
secant with order φ ≈ 1.618) but can diverge, cycle or break down far from it. All
methods here share these conventions:

* **Start.** ``x0`` (or the problem's default). Secant, Müller and inverse quadratic
  interpolation need extra starting points; they use auxiliary points x₀ ± h with
  h = max(``delta``, √ε·|x₀|), which are evaluated but are not trace steps (they appear in
  ``info["auxiliary"]`` of step 0). Step 0 is always the user's x₀.
  NOTE: Burden & Faires take the second start as given. An absolute offset alone rounds
  away for |x₀| ≳ 1e16 (x₀ + 0.1 == x₀) and the run broke down at step 0 with a
  misleading "horizontal chord". The floor √ε·|x₀| (the forward-difference step of
  Nocedal & Wright 2006, §8.1) moves x₀ in about its 8th significant digit; it exceeds the
  default delta = 0.1 only for |x₀| > 6.7e6, so moderate starts keep the plain offset
  (a fully relative delta·|x₀| left narrow domains far from 0). ``delta`` must be finite
  and > 0 (else ``ValueError``); a non-finite f at x₀ ± h ends the run with a message
  that names delta.
* **Trace.** Step ``k`` holds x_k and f(x_k). Its ``info`` describes how x_k was produced
  from the previous points (the tangent at x_{k−1}, the chord, the parabola, ...).
  ``step_size`` is |x_k − x_{k−1}|. ``n_iter == trace[-1].k``.
* **Tolerance.** ``tol(x) = xtol + 2ε|x|`` (ε = machine epsilon).
* **Stopping test (converged).** f(x_k) == 0, or |f(x_k)| ≤ ``ftol``, or
  - *sign change*: f(x_{k−1})·f(x_k) < 0 and |x_k − x_{k−1}| ≤ tol(x_k) — the two iterates
    bracket a root, so x_k is within tol of it; or
  - *a posteriori step test*: s_k·max(1, ρ/(1 − ρ)) ≤ tol(x_k), with the step
    s_k = |x_k − x_{k−1}| and the observed rate ρ = max(s_k/s_{k−1}, s_{k−1}/s_{k−2}) < 1
    (:func:`step_test_estimate`). It needs three steps whose lengths decrease twice in a
    row: one ratio right after a jump is tiny and collapses the estimate to s_k (Halley
    with a finite-difference f'' at a 6-fold root jumped across the root and then stopped
    at 2·tol). For superlinear convergence ρ → 0 and the test is the classical
    s_k ≤ tol. For linear convergence with rate ρ — fixed-point iteration, or any of these
    methods at a root of multiplicity m (Newton: ρ = 1 − 1/m) — the error of x_k is about
    ρ/(1 − ρ)·s_k, e.g. (m − 1)·s_k for Newton, and the factor accounts for it. The test
    also needs the next secant correction through the two newest iterates,
    c = |f(x_k)|·s_k/|f(x_k) − f(x_{k−1})| (a local slope: the two points are within tol),
    to continue the decrease, c < s_k, and its own a posteriori estimate c/(1 − ρ′),
    ρ′ = c/s_k, to be ≤ tol. This rejects a tiny step in a region where f is not small.
    At a root of multiplicity m the correction is only ≈ e_k/m; the factor 1/(1 − ρ′)
    restores e_k when the rate is regular (Newton: ρ′ = 1 − 1/m, so c/(1 − ρ′) = m·c).
    An anomalously short last step (Halley with a finite-difference f'' that is wrong by
    orders of magnitude at a multiple root) has c ≥ s_k and is rejected.
    (Steffensen also requires a short slope probe, see :func:`steffensen`.)
  - *zero step*: x_k = x_{k−1} (the correction rounded to zero) ends the run. It is
    converged when |f(x_k)/m| ≤ tol for a local slope m, or f changes sign between x_k and
    x_k + tol; m is f'(x_{k−1}) for Newton and Halley, Steffensen's slope when its probe
    was ≤ tol, and otherwise the difference quotient over [x_k, x_k + tol] (one extra
    evaluation, recorded in ``info["slope_probe"]``). In that last case the root can lie
    on either side, so a pass also needs a second probe at x_k − tol
    (``info["slope_probe_back"]``): a sign change there, or a difference quotient over
    [x_k − tol, x_k] whose correction is ≤ tol as well. Else the run has stalled.
  NOTE: the textbook test s_k ≤ tol (Burden & Faires Alg. 2.3–2.8) accepts one tiny step:
  Newton on f = 1/x − 1 from x₀ = 1e−11 stopped at x = 2e−11 with f = 5e10, and at a
  root of multiplicity 5 the error was 4·tol. After a jump far out, a chord or parabola
  through the far point can return a tiny or zero step where f is far from 0 (secant on
  x¹⁰ − 1 from −0.63 stopped at f = −0.99; Müller on eˣ − 10 from −6 at f = −10). The
  twice-decreasing-step condition, the factor ρ/(1 − ρ), the local secant correction with its
  own factor 1/(1 − ρ′) and the zero-step judgment remove these false convergences.
  NOTE: these are a posteriori *estimates* that assume a regular (superlinear or linear)
  rate, not bounds. At a root of multiplicity m whose steps are irregular a converged x
  can still lie up to about m·tol from the root (the zero-step judgment with the method's
  own slope uses the Newton correction |f/f'| ≈ e/m). The two-probe zero-step judgment
  without a local slope accepts only |x_k − r| ≤ tol on f = C(x − r)ᵐ, for any m. f is assumed continuous: the sign-change tests cannot tell a
  root from a pole, and they certify a sign change of the *computed* f, so where rounding
  noise dominates f near the root the result is only as accurate as that noise band.
* **Exceptions in f.** ``OverflowError`` / ``ValueError`` / ``ZeroDivisionError`` raised
  by f, f' or f'' count as a nan value (:class:`numopt.roots.bracketing.SafeScalar`).
* **Noise floor.** When x_k returns within tol(x_k) to an earlier iterate x_j, j ≤ k − 2,
  with |f(x_k)| ≥ |f(x_j)|, the run may jitter at the rounding level of f instead of
  cycling. If f changes sign between two iterates that both lie within tol of x_k, a root
  lies between them, hence within tol of x_k: converged ("noise floor"). Else, if every
  iterate x_j … x_k lies within tol of x_k, one extra evaluation at x_k ± tol decides: a
  sign change there is converged, no sign change stops with converged=False and a
  "stagnated" message; if f changes sign among x_j … x_k, it stops with converged=False
  and a "no progress" message that reports the proved distance to a root. Only otherwise
  is it a cycle.
* **Failure (converged=False, never raises).** A non-finite x or f(x); divergence
  |x_k| > DIVERGENCE_FACTOR · max(1, |x₀|); a cycle (x_k returns within tol(x_k) to an
  earlier iterate x_j, j ≤ k − 2, and |f(x_k)| ≥ |f(x_j)|, i.e. no progress over the
  period, no sign change of f within tol of x_k, and some iterate in between farther than
  tol from x_k); a stall (x_k = x_{k−1} exactly while the stopping test fails); a
  method-specific breakdown (f'(x) = 0, a horizontal chord, a parabola with no real
  root, ...); or ``max_iter``. A breakdown right after a step ≤ tol is first judged by a
  sign probe at x_k ± tol: interpolation through points that have converged to the
  rounding level can break down at the root itself, and a sign change there is
  converged. ``Result.extra["cycle_period"]`` is set on a cycle.
* **Derivatives.** Newton and Halley use the problem's f' (``grad``) and f''
  (``hess``). When one is missing they fall back to central differences
  (:mod:`numopt.core.diff`); those f evaluations are counted in ``n_fev`` and
  ``Result.extra["derivatives"] = "finite_difference"``. ``n_gev``/``n_hev`` count calls
  of the supplied f' and f''.

Info keys (step k ≥ 1 unless stated):
    previous: float                    x_{k−1} (all methods).
    newton:
        tangent: {point: [x, y], slope}   tangent at (x_{k−1}, f(x_{k−1})); x_k is its zero.
    secant:
        chord: [[x, y], [x, y]]        (x_{k−2}, f), (x_{k−1}, f); x_k is the chord's zero
                                       (x_{−1} = x₀ + h, the auxiliary point).
    halley:
        hyperbola: {center, alpha, beta, gamma}  osculating hyperbola
                   y(x) = (t + α)/(βt + γ), t = x − center (center = x_{k−1}); it matches
                   f, f', f'' at x_{k−1} and its zero is x_k = center − α.
        derivatives: [f, f', f'']      at x_{k−1}.
    steffensen:
        chord: [[x, y], [x, y]]        (x_{k−1}, f(x_{k−1})) and (z, f(z)), z = x_{k−1} + f(x_{k−1}).
        slope: float                   (f(z) − f(x_{k−1}))/(z − x_{k−1}), the derivative
                                       estimate over the probe actually taken (z − x_{k−1}
                                       equals f(x_{k−1}) up to the rounding of z).
    muller:
        points: [[x, y]×3]             the interpolation points (x_{k−3}, x_{k−2}, x_{k−1});
                                       x_{−2} = x₀ − h and x_{−1} = x₀ + h.
        parabola: {center, a, b, c}    p(x) = a t² + b t + c, t = x − center (center = x_{k−1});
                                       x_k is the root of p nearer to x_{k−1}.
    inverse_quadratic_interpolation:
        points: [[x, y]×3]             the interpolation points (x_{k−3}, x_{k−2}, x_{k−1}).
        inverse_parabola: {a, b, c}    x(y) = a y² + b y + c through the points; x_k = c.
    fixed_point:
        cobweb: [[x, y]×3]             (x_{k−1}, x_{k−1}) → (x_{k−1}, g(x_{k−1})) → (x_k, x_k),
                                       with g(x) = x − λ f(x).
        lam: float                     λ.
    step 0 of secant, muller, inverse_quadratic_interpolation:
        auxiliary: [[x, y]...]         the auxiliary starting points and their f values
                                       (step 0 of the other methods has an empty info).
    last step of a run that ends on x_k = x_{k−1} without a local slope (all methods), and
    of a Steffensen run that ends without a new slope (x_k + f(x_k) rounds to x_k, or
    f(x_k + f(x_k)) = f(x_k)), of a run whose iterates cluster within tol of x_k at the
    noise floor without a sign change, and of a breakdown right after a step ≤ tol:
        slope_probe: [x, y]            (x_k ± tol, f(x_k ± tol)), the extra evaluation that
                                       judges the end of the run (+tol for the zero step;
                                       toward the root predicted by the last slope estimate
                                       for Steffensen and by the last secant otherwise;
                                       always within tol of x_k).
    last step of a run that ends on x_k = x_{k−1} without a local slope, when the
    correction from ``slope_probe`` alone is ≤ tol:
        slope_probe_back: [x, y]       (x_k − tol, f(x_k − tol)), the second probe of the
                                       zero-step judgment (within tol of x_k).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

from ..core import diff
from ..core.counting import Counted, finite, scalar_problem, start_scalar
from ..core.registry import ParamSpec, register
from ..core.types import Problem, Result, Step
from .bracketing import SafeScalar, raised

#: Machine epsilon of float64.
EPS = sys.float_info.epsilon
#: Divergence is declared when |x_k| > DIVERGENCE_FACTOR · max(1, |x₀|).
DIVERGENCE_FACTOR = 1e12

ScalarFn = Callable[[float], float]

OPEN_PARAMS = (
    ParamSpec(
        "xtol",
        1e-10,
        min=1e-15,
        max=1e-2,
        log=True,
        help="Stop when the step |xₖ − xₖ₋₁| ≤ xtol + 2ε|xₖ|.",
    ),
    ParamSpec("ftol", 0.0, min=0.0, max=1e-2, help="Also stop when |f(x)| ≤ ftol (0 disables)."),
    ParamSpec("max_iter", 100, kind="int", min=1, max=10_000, help="Iteration limit."),
)
DELTA_PARAM = ParamSpec(
    "delta",
    0.1,
    min=1e-6,
    max=10.0,
    log=True,
    help="Auxiliary starting points are placed at x₀ ± max(delta, √ε·|x₀|).",
)


def _tol(x: float, xtol: float) -> float:
    return xtol + 2.0 * EPS * abs(x)


#: √ε, the relative floor of the auxiliary offset (Nocedal & Wright 2006, §8.1).
SQRT_EPS = math.sqrt(EPS)


def _offset(x0: float, delta: float) -> float:
    """The auxiliary offset h = max(delta, √ε·|x₀|) (> 0 moves any finite x₀)."""
    if not (delta > 0.0 and math.isfinite(delta)):
        raise ValueError(f"delta must be finite and > 0, got {delta}")
    return max(delta, SQRT_EPS * abs(x0))


def _aux_message(points: list[tuple[float, float]], h: float, delta: float, note: str) -> str:
    """Breakdown message for a non-finite f at an auxiliary start; it names delta."""
    bad = ", ".join(f"f({x!r}) = {fx!r}" for x, fx in points if not finite(fx))
    return (
        f"f is not finite at an auxiliary starting point x₀ ± h, h = max(delta, √ε|x₀|) = "
        f"{h:.3g} ({bad}{note}); a smaller delta (now {delta:g}) keeps it inside the domain of f"
    )


def step_test_estimate(xs: list[float]) -> tuple[float, float] | None:
    """A posteriori error estimate of x_k from its last three steps: ``(estimate, ρ)``.

    With s_k = |x_k − x_{k−1}| and the two observed rates ρ_k = s_k/s_{k−1} and
    ρ_{k−1} = s_{k−1}/s_{k−2}, both < 1, returns s_k·max(1, ρ/(1 − ρ)) with the larger
    rate ρ = max(ρ_k, ρ_{k−1}): the a posteriori bound of a contraction (Burden & Faires
    §2.2; Dahlquist & Björck 2008, ch. 6), evaluated with the slower of two consecutive
    observed rates. None without two consecutive decreases (fewer than three steps, or a
    step that did not shrink). It is an estimate, not a bound.
    """
    if len(xs) < 4:
        return None
    s_k, s_1, s_2 = abs(xs[-1] - xs[-2]), abs(xs[-2] - xs[-3]), abs(xs[-3] - xs[-4])
    if not s_k < s_1 < s_2:
        return None
    rho = max(s_k / s_1, s_1 / s_2)
    return s_k * max(1.0, rho / (1.0 - rho)), rho


def _correction_error(c: float, s: float) -> float:
    """Error estimate of x_k from its next correction c and its last step s.

    With the observed rate ρ′ = c/s < 1 the remaining corrections sum to c/(1 − ρ′), the
    a posteriori bound of a contraction (Burden & Faires §2.2). Returns ``inf`` when
    c ≥ s: the step and the correction then show no contraction.
    """
    if not c < s:
        return math.inf
    return c / (1.0 - c / s)


class _Run:
    """Trace bookkeeping and the shared stopping/divergence/cycle tests."""

    def __init__(
        self, method: str, f: Counted, x0: float, f0: float, xtol: float, ftol: float, max_iter: int
    ) -> None:
        self.method = method
        self.f = f
        self.x0 = x0
        self.xtol = xtol
        self.ftol = ftol
        self.max_iter = max_iter
        self.xs = [x0]
        self.fs = [f0]
        self.trace: list[Step] = []
        self.counters: dict[str, Counted] = {}
        self.extra: dict[str, Any] = {}

    @property
    def k(self) -> int:
        return len(self.xs) - 1

    def result(self, converged: bool, message: str) -> Result:
        return Result(
            self.method,
            self.xs[-1],
            self.fs[-1],
            converged,
            message,
            self.k,
            self.f.n,
            n_gev=self.counters["grad"].n if "grad" in self.counters else 0,
            n_hev=self.counters["hess"].n if "hess" in self.counters else 0,
            trace=self.trace,
            extra=self.extra,
        )

    def start(self, info: dict[str, Any] | None = None) -> Result | None:
        """Record step 0; return a finished Result if x₀ already settles the problem."""
        x, fx = self.xs[0], self.fs[0]
        self.trace.append(Step(0, x, fx, info=info or {}))
        if not finite(x, fx):
            return self.result(False, f"f(x₀) is not finite at x₀ = {x!r}{raised(self.f)}")
        if fx == 0.0:
            return self.result(True, "f(x) is exactly zero")
        if abs(fx) <= self.ftol:
            return self.result(True, f"|f(x)| = {abs(fx):.3g} ≤ ftol")
        return None

    def at_limit(self) -> Result | None:
        if self.k >= self.max_iter:
            return self.result(False, f"reached max_iter={self.max_iter}")
        return None

    def step(
        self,
        x_new: float,
        f_new: float,
        info: dict[str, Any],
        *,
        step_test_ok: bool = True,
        slope: float | None = None,
    ) -> Result | None:
        """Record step k + 1 and apply the shared tests; return a Result to stop.

        ``step_test_ok=False`` lets a method veto the a posteriori step test for this step
        (Steffensen does so while its slope probe is longer than tol); the sign-change
        test still applies. ``slope`` is the method's own local slope at x_{k−1} (f' for
        Newton and Halley), used only when x_k = x_{k−1} (see :meth:`_judge_zero_step`).
        """
        x_old, f_old = self.xs[-1], self.fs[-1]
        info = {"previous": x_old, **info}
        step = abs(x_new - x_old)
        self.xs.append(x_new)
        self.fs.append(f_new)
        self.trace.append(Step(self.k, x_new, f_new, step_size=step, info=info))
        if not finite(x_new, f_new):
            return self.result(
                False,
                f"diverged: non-finite value (x = {x_new!r}, f = {f_new!r}{raised(self.f)})",
            )
        if f_new == 0.0:
            return self.result(True, "f(x) is exactly zero")
        if abs(f_new) <= self.ftol:
            return self.result(True, f"|f(x)| = {abs(f_new):.3g} ≤ ftol")
        tol = _tol(x_new, self.xtol)
        if step <= tol and math.copysign(1.0, f_new) != math.copysign(1.0, f_old):
            return self.result(
                True,
                f"f changes sign between xₖ₋₁ and xₖ, |xₖ − xₖ₋₁| = {step:.3g} ≤ tol "
                f"= {tol:.3g}: a root lies within tol of xₖ",
            )
        if step == 0.0:
            return self._judge_zero_step(x_new, f_new, tol, slope, info)
        test = step_test_estimate(self.xs)
        if step_test_ok and test is not None and test[0] <= tol:
            estimate, rho = test
            # The next secant correction c through the two newest iterates (a local slope,
            # since they are within tol): it rejects a tiny step that a far interpolation
            # point produced in a region where f is not small. At an m-fold root c ≈ e_k/m,
            # so c must also continue the contraction and pass its own a posteriori factor.
            correction = abs(f_new) * (step / abs(f_new - f_old)) if f_new != f_old else math.inf
            error = _correction_error(correction, step)
            if error <= tol:
                return self.result(
                    True,
                    f"step test: |xₖ − xₖ₋₁|·max(1, ρ/(1−ρ)) = {estimate:.3g} ≤ tol = "
                    f"{tol:.3g} (ρ = {rho:.3g}) and the next "
                    f"secant correction c = {correction:.3g} gives "
                    f"c/(1 − c/|xₖ − xₖ₋₁|) = {error:.3g} ≤ tol",
                )
        bound = DIVERGENCE_FACTOR * max(1.0, abs(self.x0))
        if abs(x_new) > bound:
            return self.result(False, f"diverged: |x| = {abs(x_new):.3g} > {bound:.3g}")
        k = self.k
        for j in range(k - 1):  # earlier iterates x_0 … x_{k−2}: period k − j ≥ 2
            if abs(x_new - self.xs[j]) <= tol and abs(f_new) >= abs(self.fs[j]):
                if (done := self._judge_noise_floor(j, tol)) is not None:
                    return done
                self.extra["cycle_period"] = k - j
                return self.result(
                    False,
                    f"cycle of period {k - j} detected: x_{k} returned to x_{j} = {self.xs[j]:.6g} "
                    f"with no decrease in |f|",
                )
        return None

    def _judge_noise_floor(self, j: int, tol: float) -> Result | None:
        """x_k returned within tol to x_j with no decrease in |f|: noise floor or cycle?

        Near a root where the computed f is dominated by rounding (e.g. a polynomial in
        monomial form) the iterates jitter instead of converging; that is not a cycle. Let
        R = min |x_i − x_k| over the iterates x_i (i ≤ k) at which f has the opposite sign
        of f(x_k); a root lies between x_i and x_k, hence within R of x_k (f continuous).

        * R ≤ tol: converged ("noise floor").
        * Else, if every iterate x_j … x_k lies within tol of x_k: one extra evaluation at
          x_k ± tol (toward the root predicted by the secant through x_(k−1), x_k; kept in
          ``info["slope_probe"]``). A sign change there: converged ("noise floor"); else
          stagnated (not converged).
        * Else, if R ≤ max |x_i − x_k| over i = j … k (f changes sign within the period's
          spread): no progress, not converged; the message reports R and does not call it a
          cycle (it may be jitter at the noise floor or a cycle around the root).
        * Else None: the iterates left the tol-ball and came back with no sign change
          around x_k, a genuine cycle.
        """
        k, x_k, f_k = self.k, self.xs[-1], self.fs[-1]
        sign_k = math.copysign(1.0, f_k)
        r_root = min(
            (
                abs(x - x_k)
                for x, fx in zip(self.xs, self.fs, strict=True)
                if math.copysign(1.0, fx) != sign_k
            ),
            default=math.inf,
        )
        if r_root <= tol:
            return self.result(
                True,
                f"noise floor: x_{k} returned within tol = {tol:.3g} of x_{j} with no decrease in "
                f"|f|, and f changes sign between x_k and an iterate {r_root:.3g} away: a root "
                f"lies within tol of x_k",
            )
        spread = max(abs(x - x_k) for x in self.xs[j:])
        if spread <= tol:
            # All iterates sit on one side of the root (e.g. one ulp below it): probe
            # x_k ± tol toward the root predicted by the secant through x_(k−1), x_k.
            if self.sign_probe(self.local_slope()):
                return self.result(
                    True,
                    f"noise floor: x_{j} … x_{k} all lie within tol = {tol:.3g} of x_k with no "
                    f"decrease in |f|, and f changes sign between x_k and x_k ± tol",
                )
            return self.result(
                False,
                f"stagnated: x_{j} … x_{k} all lie within tol = {tol:.3g} of x_k but |f| no "
                f"longer decreases (|f(x_k)| = {abs(f_k):.3g}, likely the rounding level of f) "
                f"and f does not change sign within tol of x_k",
            )
        if r_root <= spread:
            return self.result(
                False,
                f"no progress: x_{k} returned within tol of x_{j} with no decrease in |f|; f "
                f"changes sign among x_{j} … x_{k}, so a root lies within {r_root:.3g} of x_k, "
                f"but that is > tol = {tol:.3g} (jitter at the noise floor of f, or iterates "
                f"oscillating around the root)",
            )
        return None

    def _judge_zero_step(
        self, x: float, fx: float, tol: float, slope: float | None, info: dict[str, Any]
    ) -> Result:
        """x_k = x_{k−1}: the method's correction rounded to zero, so the run ends here.

        Converged when a local slope m places the root within tol, |f(x_k)/m| ≤ tol, or f
        changes sign between x_k and x_k ± tol. The slope is the method's own when it is
        local (f'(x_{k−1}) for Newton and Halley; Steffensen's slope from a probe ≤ tol);
        otherwise one extra evaluation at x_k + tol gives it (``info["slope_probe"]``),
        because an interpolation through a far point can freeze x at a non-root.

        Without a local slope the root may lie on either side of x_k, and a probe that
        points away from it overestimates the slope: at (x − 1)³ the forward correction
        over [x_k, x_k + tol] is e/3.87 and accepted e = 3.8·tol (Hypothesis
        counterexample), at (x − 1)⁶ up to 8·tol. So when the forward correction alone
        would accept, a second probe at x_k − tol (``info["slope_probe_back"]``) must
        show a sign change or a correction |f(x_k)|·tol/|f(x_k − tol) − f(x_k)| ≤ tol
        too. With no sign change at either probe, that requires |f(x_k ± tol)| ≥ 2|f(x_k)|
        on both sides, which on f = C(x − r)ᵐ (any m) holds only for |x_k − r| ≤ tol.
        """
        m: float = math.nan if slope is None else slope
        if finite(m) and m != 0.0:
            correction = abs(fx / m)
            if correction <= tol:
                return self.result(
                    True,
                    f"xₖ = xₖ₋₁ and the Newton correction |f(x)/slope| = {correction:.3g} ≤ tol",
                )
            return self._stalled(x, fx, correction)
        if self.sign_probe(None):
            return self.result(True, "xₖ = xₖ₋₁ and f changes sign between xₖ and xₖ + tol")
        correction = self._probe_correction(x, fx, info["slope_probe"])
        if correction > tol:
            return self._stalled(x, fx, correction)
        if self.sign_probe(None, toward=-1.0, key="slope_probe_back"):
            return self.result(True, "xₖ = xₖ₋₁ and f changes sign between xₖ and xₖ − tol")
        correction = max(correction, self._probe_correction(x, fx, info["slope_probe_back"]))
        if correction <= tol:
            return self.result(
                True,
                f"xₖ = xₖ₋₁ and the Newton corrections |f(x)/slope| over [xₖ − tol, xₖ] and "
                f"[xₖ, xₖ + tol] are both ≤ {correction:.3g} ≤ tol",
            )
        return self._stalled(x, fx, correction)

    @staticmethod
    def _probe_correction(x: float, fx: float, probe: list[float]) -> float:
        """|f(x_k)/m| with the difference quotient m over [x_k, x_p]; inf without a slope."""
        x_p, f_p = probe
        m = (f_p - fx) / (x_p - x) if finite(f_p) else math.nan
        return abs(fx / m) if finite(m) and m != 0.0 else math.inf

    def _stalled(self, x: float, fx: float, correction: float) -> Result:
        return self.result(
            False,
            f"stalled: xₖ = xₖ₋₁ = {x:.6g} but |f(x)| = {abs(fx):.3g} and the Newton "
            f"correction |f(x)/slope| = {correction:.3g} > tol",
        )

    def sign_probe(
        self, slope: float | None, *, toward: float | None = None, key: str = "slope_probe"
    ) -> bool:
        """One extra evaluation at x_p = x_k ± tol: True when f changes sign across it.

        The direction is ``toward`` (±1) when given, else toward the root that ``slope``
        predicts from x_k (−sign(f·slope)); +tol without a usable slope. x_p is kept within
        tol of x_k (one ulp back if the sum rounded outward), so a sign change places a root
        within tol of x_k. The pair (x_p, f(x_p)) is stored in the last step's
        ``info[key]`` (``"slope_probe"`` by default).
        """
        x_k, f_k = self.xs[-1], self.fs[-1]
        tol = _tol(x_k, self.xtol)
        if toward is None:
            toward = -math.copysign(1.0, f_k * slope) if slope and finite(slope) else 1.0
        x_p = x_k + toward * tol
        if abs(x_p - x_k) > tol:
            x_p = math.nextafter(x_p, x_k)
        f_p = float(self.f(x_p))
        info = self.trace[-1].info
        if isinstance(info, dict):
            info[key] = [x_p, f_p]
        return finite(f_p) and math.copysign(1.0, f_p) != math.copysign(1.0, f_k)

    def local_slope(self) -> float | None:
        """The secant slope through x_(k−1), x_k, or None if it does not exist."""
        if len(self.xs) < 2 or self.xs[-1] == self.xs[-2]:
            return None
        return (self.fs[-1] - self.fs[-2]) / (self.xs[-1] - self.xs[-2])

    def breakdown(self, message: str) -> Result:
        """Stop with converged=False, unless the last step was ≤ tol and a sign probe at
        x_k ± tol proves a root within tol (an interpolation through points that have
        converged to the rounding level can break down exactly at the root)."""
        if len(self.xs) >= 2 and finite(self.xs[-1], self.fs[-1]):
            x_k = self.xs[-1]
            if abs(x_k - self.xs[-2]) <= _tol(x_k, self.xtol) and self.sign_probe(
                self.local_slope()
            ):
                return self.result(
                    True,
                    f"{message}; but the last step was ≤ tol and f changes sign between x_k "
                    f"and x_k ± tol: a root lies within tol of x_k",
                )
        return self.result(False, message)


def _first_derivative(prob: Problem, f: Counted, run: _Run) -> Callable[[float], float]:
    if prob.grad is not None:
        g = Counted(SafeScalar(prob.grad))
        run.counters["grad"] = g
        return lambda x: float(g(x))
    run.extra["derivatives"] = "finite_difference"
    return lambda x: float(diff.derivative(f, x))


def _second_derivative(prob: Problem, f: Counted, run: _Run) -> Callable[[float], float]:
    if prob.hess is not None:
        h = Counted(SafeScalar(prob.hess))
        run.counters["hess"] = h
        return lambda x: float(h(x))
    run.extra["derivatives"] = "finite_difference"
    return lambda x: float(diff.second_derivative(f, x))


def _setup(
    method: str,
    problem: Problem | ScalarFn,
    x0: Any,
    xtol: float,
    ftol: float,
    max_iter: int,
) -> tuple[Problem, Counted, _Run]:
    prob = scalar_problem(problem)
    x = start_scalar(prob, x0)
    f = Counted(SafeScalar(prob.f))
    fx = float(f(x))
    return prob, f, _Run(method, f, x, fx, xtol, ftol, max_iter)


# --------------------------------------------------------------------------------------
# Newton
# --------------------------------------------------------------------------------------


@register(
    id="newton",
    family="roots",
    name="Newton–Raphson",
    params=OPEN_PARAMS,
    needs=("f", "grad", "x0"),
    order="quadratic (simple root); linear at a multiple root",
    summary="Follow the tangent line at x_k down to where it crosses zero.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.3",),
)
def newton(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
) -> Result:
    """Newton's method, Burden & Faires Alg. 2.3: x_{k+1} = x_k − f(x_k)/f'(x_k).

    Converges quadratically to a simple root from a close enough start; linearly (rate
    1 − 1/m) at a root of multiplicity m, where the error of x_k is (m − 1) times the last
    step. Breaks down when f'(x_k) = 0.

    Stops (converged) on the shared open-method test (f = 0, |f| ≤ ftol, a sign change
    within tol, or the a posteriori step test, which accounts for the linear rate at a
    multiple root).
    """
    prob, f, run = _setup("newton", problem, x0, xtol, ftol, max_iter)
    fprime = _first_derivative(prob, f, run)
    if (done := run.start()) is not None:
        return done
    while True:
        if (done := run.at_limit()) is not None:
            return done
        x, fx = run.xs[-1], run.fs[-1]
        d = fprime(x)
        if not finite(d):
            return run.breakdown(f"f'(x) is not finite at x = {x!r}")
        if d == 0.0:
            return run.breakdown(f"f'(x) = 0 at x = {x:.6g}: the tangent is horizontal")
        x_new = x - fx / d
        f_new = float(f(x_new))
        info = {"tangent": {"point": [x, fx], "slope": d}}
        if (done := run.step(x_new, f_new, info, slope=d)) is not None:
            return done


# --------------------------------------------------------------------------------------
# Secant
# --------------------------------------------------------------------------------------


@register(
    id="secant",
    family="roots",
    name="Secant",
    params=(*OPEN_PARAMS, DELTA_PARAM),
    needs=("f", "x0"),
    order="superlinear (φ ≈ 1.618)",
    summary="Newton's method with the tangent replaced by the chord through the last two points.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.4",),
)
def secant(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
    delta: float = 0.1,
) -> Result:
    """Secant method, Burden & Faires Alg. 2.4:

        x_{k+1} = x_k − f(x_k)(x_k − x_{k−1})/(f(x_k) − f(x_{k−1})).

    The two starting points are x₋₁ = x₀ + h, h = max(``delta``, √ε|x₀|) (auxiliary), and
    x₀. Breaks down when f(x_k) = f(x_{k−1}) (a horizontal chord).

    Stops (converged) on the shared open-method test.
    """
    _, f, run = _setup("secant", problem, x0, xtol, ftol, max_iter)
    h = _offset(run.x0, delta)
    x_aux = run.x0 + h
    f_aux = float(f(x_aux))
    note = raised(f)
    if (done := run.start({"auxiliary": [[x_aux, f_aux]]})) is not None:
        return done
    if not finite(f_aux):
        return run.breakdown(_aux_message([(x_aux, f_aux)], h, delta, note))
    x_prev, f_prev = x_aux, f_aux
    while True:
        if (done := run.at_limit()) is not None:
            return done
        x, fx = run.xs[-1], run.fs[-1]
        if fx == f_prev:
            return run.breakdown(f"horizontal chord: f(xₖ) = f(xₖ₋₁) = {fx:.6g}")
        x_new = x - fx * (x - x_prev) / (fx - f_prev)
        f_new = float(f(x_new))
        info = {"chord": [[x_prev, f_prev], [x, fx]]}
        if (done := run.step(x_new, f_new, info)) is not None:
            return done
        x_prev, f_prev = x, fx


# --------------------------------------------------------------------------------------
# Halley
# --------------------------------------------------------------------------------------


@register(
    id="halley",
    family="roots",
    name="Halley",
    params=OPEN_PARAMS,
    needs=("f", "grad", "hess", "x0"),
    order="cubic (simple root)",
    summary="Fit the hyperbola that matches f, f′ and f″ at x_k and jump to its zero.",
    references=(
        "Halley (1694); Scavo & Thoo (1995), Amer. Math. Monthly 102(5), 417–426",
        "Press et al., Numerical Recipes (3rd ed.), §9.4, eq. (9.4.7)",
    ),
)
def halley(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
) -> Result:
    """Halley's method:

        x_{k+1} = x_k − 2 f f' / (2 f'² − f f'')      (f, f', f'' at x_k).

    x_{k+1} is the zero of the osculating hyperbola y = (t + α)/(βt + γ), t = x − x_k,
    with γ = 2f'/D, β = −f''/D, α = 2ff'/D, D = 2f'² − ff'' (it matches f, f', f'' at x_k;
    Scavo & Thoo 1995). Cubic convergence at a simple root. Breaks down when f'(x_k) = 0
    (the step would be zero although f ≠ 0) or D = 0.

    Stops (converged) on the shared open-method test.
    """
    prob, f, run = _setup("halley", problem, x0, xtol, ftol, max_iter)
    fprime = _first_derivative(prob, f, run)
    fsecond = _second_derivative(prob, f, run)
    if (done := run.start()) is not None:
        return done
    while True:
        if (done := run.at_limit()) is not None:
            return done
        x, fx = run.xs[-1], run.fs[-1]
        d1 = fprime(x)
        if not finite(d1):
            return run.breakdown(f"f'(x) is not finite at x = {x!r}")
        if d1 == 0.0:
            return run.breakdown(f"f'(x) = 0 at x = {x:.6g}: Halley's step degenerates")
        d2 = fsecond(x)
        den = 2.0 * d1 * d1 - fx * d2
        if not finite(d2, den):
            return run.breakdown(f"f''(x) or 2f'² − ff'' is not finite at x = {x!r}")
        if den == 0.0:
            return run.breakdown(f"2f'² − f·f'' = 0 at x = {x:.6g}: the hyperbola has no zero")
        alpha = 2.0 * fx * d1 / den
        x_new = x - alpha
        f_new = float(f(x_new))
        info = {
            "hyperbola": {"center": x, "alpha": alpha, "beta": -d2 / den, "gamma": 2.0 * d1 / den},
            "derivatives": [fx, d1, d2],
        }
        if (done := run.step(x_new, f_new, info, slope=d1)) is not None:
            return done


# --------------------------------------------------------------------------------------
# Steffensen
# --------------------------------------------------------------------------------------


@register(
    id="steffensen",
    family="roots",
    name="Steffensen",
    params=OPEN_PARAMS,
    needs=("f", "x0"),
    order="quadratic (two f evaluations per step)",
    summary="Newton's method with f′ estimated from the slope between x and x + f(x).",
    references=(
        "Steffensen (1933), Skand. Aktuarietidskr. 16, 64–72",
        "Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.6 with g(x) = x + f(x)",
    ),
)
def steffensen(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
) -> Result:
    """Steffensen's method:

        x_{k+1} = x_k − f(x_k)² / (f(x_k + f(x_k)) − f(x_k)).

    This is Newton's method with f'(x_k) ≈ (f(x_k + f(x_k)) − f(x_k))/f(x_k), and it equals
    Aitken's Δ² extrapolation of the fixed-point iteration g(x) = x + f(x) (Burden & Faires
    Alg. 2.6). Quadratic convergence without derivatives, but the probe step f(x_k) mixes
    units of f and x: the method is not scale invariant and can fail when |f| is large.

    Stops (converged) on the shared open-method test, except that the a posteriori step
    test also requires the probe length |f(x_{k−1})| ≤ tol, i.e. the slope estimate must
    come from an interval no longer than the tolerance. The sign-change test needs no such
    condition: when f is steeply scaled (|f| at one float spacing from the root above tol)
    the iterates settle on the two floats around the root, and the sign change between
    them ends the run (the probe condition alone could never pass). When the probe
    x_k + f(x_k) rounds to x_k (|f| below half the float spacing) or f(x_k + f(x_k)) =
    f(x_k), no new slope exists; the run then stops (:func:`_steffensen_end`). One extra
    evaluation at x_k ± tol (toward the root predicted by the previous slope estimate s)
    proves convergence by a sign change. Without one, x_k is converged only when the
    a posteriori step test of the last three steps holds *and* the Newton correction
    c = |f(x_k)/s| gives c/(1 − c/s_k) ≤ tol; else the run ends with a breakdown message.
    NOTE: |f(x_k)/s| ≤ tol alone accepted x = 1 + 4.1e−6 on (x − 1)³ with xtol = 1e−6:
    at an m-fold root the correction is ≈ e/m, and s was a stale slope from x_{k−1}.
    NOTE: without that extra condition a far start on a steep f (e^x − 10 from x₀ = 4:
    probe 44.6, slope estimate 3·10¹⁹ instead of 55) gives a step below the float spacing
    and a false "converged". With it, such runs stall or crawl and report
    ``converged=False``.
    """
    _, f, run = _setup("steffensen", problem, x0, xtol, ftol, max_iter)
    if (done := run.start()) is not None:
        return done
    last_slope: float | None = None
    while True:
        if (done := run.at_limit()) is not None:
            return done
        x, fx = run.xs[-1], run.fs[-1]
        z = x + fx
        if z == x:
            return _steffensen_end(run, last_slope, "x + f(x) rounds to x")
        fz = float(f(z))
        if not finite(z, fz):
            return run.breakdown(f"f(x + f(x)) is not finite (x + f(x) = {z!r})")
        den = fz - fx
        if den == 0.0:
            return _steffensen_end(run, last_slope, "f(x + f(x)) = f(x) (zero slope estimate)")
        # NOTE: the slope is (f(z) − f(x))/(z − x) over the probe actually taken. z − x is
        # exact (Sterbenz/Fast2Sum) but differs from f(x) whenever x + f(x) rounds; near a
        # root, where |f| is a few ulps of x, the textbook (f(z) − f(x))/f(x) was then off by
        # a large factor and made the steps at a multiple root irregular.
        hz = z - x
        # NOTE: f·(h/den), not f²/den: f² underflows for |f| < 1e-154 (x₀ = 3e-288 on f = x
        # gave a zero step) and overflows for |f| > 1e154.
        x_new = x - fx * (hz / den)
        f_new = float(f(x_new))
        last_slope = den / hz
        info = {"chord": [[x, fx], [z, fz]], "slope": last_slope}
        probe_ok = abs(hz) <= _tol(x_new, xtol)
        slope = last_slope if probe_ok else None
        if (done := run.step(x_new, f_new, info, step_test_ok=probe_ok, slope=slope)) is not None:
            return done


def _steffensen_end(run: _Run, last_slope: float | None, why: str) -> Result:
    """End a Steffensen run that cannot form a new slope at x_k (``why`` says why).

    This happens when the probe x_k + f(x_k) rounds to x_k (|f| below half the float
    spacing) or when f(x_k + f(x_k)) = f(x_k). First a sign probe at x_p = x_k ± tol,
    toward the root predicted by the previous slope estimate s (x_k + tol without one);
    one extra evaluation, kept in the last step's ``info["slope_probe"]``. A sign change
    places a root within tol of x_k. Otherwise x_k is converged only when the a posteriori
    step test of the last three steps holds and the Newton correction c = |f(x_k)/s| gives
    c/(1 − c/s_k) ≤ tol (c alone is ≈ e/m at an m-fold root, and s is the slope at
    x_{k−1}, not at x_k). Else the run breaks down.
    """
    xs, fx = run.xs, run.fs[-1]
    x = xs[-1]
    tol = _tol(x, run.xtol)
    if run.sign_probe(last_slope):
        return run.result(True, f"{why} at x_k, and f changes sign between x_k and x_k ± tol")
    s_k = abs(xs[-1] - xs[-2]) if len(xs) >= 2 else None
    test = step_test_estimate(xs)
    est = test[0] if test is not None else None
    c = abs(fx / last_slope) if last_slope else math.inf
    error = _correction_error(c, s_k) if s_k else math.inf
    if est is not None and est <= tol and error <= tol:
        return run.result(
            True,
            f"{why} at xₖ; the step test |xₖ − xₖ₋₁|·max(1, ρ/(1−ρ)) = {est:.3g} ≤ tol = "
            f"{tol:.3g} and the correction c = |f(x)/slope| = {c:.3g} gives "
            f"c/(1 − c/|xₖ − xₖ₋₁|) = {error:.3g} ≤ tol",
        )
    return run.result(
        False,
        f"{why} at x = {x:.6g} (|f(x)| = {abs(fx):.3g}): no new slope; f keeps its sign at "
        f"x ± tol, and the last steps and slope estimate do not place x within tol",
    )


# --------------------------------------------------------------------------------------
# Müller
# --------------------------------------------------------------------------------------


@register(
    id="muller",
    family="roots",
    name="Müller",
    params=(*OPEN_PARAMS, DELTA_PARAM),
    needs=("f", "x0"),
    order="superlinear (≈ 1.84)",
    summary="Fit a parabola through the last three points and move to its nearer root.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.8",),
)
def muller(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
    delta: float = 0.1,
) -> Result:
    """Müller's method in real arithmetic, Burden & Faires Alg. 2.8.

    Through (p₀, f₀), (p₁, f₁), (p₂, f₂) fit p(x) = a(x − p₂)² + b(x − p₂) + c with
    c = f₂, a = (δ₂ − δ₁)/(h₂ + h₁), b = δ₂ + h₂a, where h_i and δ_i are the last two
    spacings and divided differences. The next point is p₂ − 2c/E with E = b ± √(b² − 4ac)
    choosing the sign that makes |E| largest (the root of p nearer to p₂, computed without
    cancellation). Starting points: p₀ = x₀ − h, p₁ = x₀ + h (auxiliary,
    h = max(``delta``, √ε|x₀|)), p₂ = x₀.

    Real arithmetic only: when b² − 4ac < 0 the parabola has no real root (Müller would
    continue with complex numbers) and the method stops with ``converged=False``.

    Stops (converged) on the shared open-method test (B&F: |h| < TOL).
    """
    _, f, run = _setup("muller", problem, x0, xtol, ftol, max_iter)
    h = _offset(run.x0, delta)
    p0, p1 = run.x0 - h, run.x0 + h
    f0 = float(f(p0))
    note = raised(f)
    f1 = float(f(p1))
    note = note or raised(f)
    if (done := run.start({"auxiliary": [[p0, f0], [p1, f1]]})) is not None:
        return done
    if not finite(f0, f1):
        return run.breakdown(_aux_message([(p0, f0), (p1, f1)], h, delta, note))
    while True:
        if (done := run.at_limit()) is not None:
            return done
        p2, f2 = run.xs[-1], run.fs[-1]
        h1, h2 = p1 - p0, p2 - p1
        if h1 == 0.0 or h2 == 0.0 or h1 + h2 == 0.0:
            return run.breakdown("two interpolation points coincide: the parabola is undefined")
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = d2 + h2 * a
        disc = b * b - 4.0 * f2 * a
        if disc < 0.0:
            return run.breakdown(
                f"the parabola through the last three points has no real root "
                f"(b² − 4ac = {disc:.3g} < 0); real-arithmetic Müller stops here"
            )
        root = math.sqrt(disc)
        e = b + root if abs(b - root) < abs(b + root) else b - root
        if e == 0.0:
            return run.breakdown("the parabola is flat (b = 0 and b² − 4ac = 0)")
        x_new = p2 - 2.0 * f2 / e
        f_new = float(f(x_new))
        info = {
            "points": [[p0, f0], [p1, f1], [p2, f2]],
            "parabola": {"center": p2, "a": a, "b": b, "c": f2},
        }
        if (done := run.step(x_new, f_new, info)) is not None:
            return done
        p0, f0, p1, f1 = p1, f1, p2, f2


# --------------------------------------------------------------------------------------
# Inverse quadratic interpolation
# --------------------------------------------------------------------------------------


@register(
    id="inverse_quadratic_interpolation",
    family="roots",
    name="Inverse quadratic interpolation",
    params=(*OPEN_PARAMS, DELTA_PARAM),
    needs=("f", "x0"),
    order="superlinear (≈ 1.84)",
    summary="Fit x as a quadratic function of y through the last three points; evaluate it at y = 0.",
    references=(
        "Brent (1973), Algorithms for Minimization without Derivatives, §4.3",
        "Süli & Mayers, An Introduction to Numerical Analysis (2003), §1.6",
    ),
)
def inverse_quadratic_interpolation(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
    delta: float = 0.1,
) -> Result:
    """Inverse quadratic interpolation (the interpolation step of Brent's method, unguarded).

    Through (f₀, x₀), (f₁, x₁), (f₂, x₂) the Lagrange polynomial x(y) of degree 2 is
    evaluated at y = 0:

        x_new = x₀ f₁f₂/((f₀−f₁)(f₀−f₂)) + x₁ f₀f₂/((f₁−f₀)(f₁−f₂)) + x₂ f₀f₁/((f₂−f₀)(f₂−f₁)).

    Starting points: x₀ − h, x₀ + h (auxiliary, h = max(``delta``, √ε|x₀|)), x₀. Breaks
    down when two of the three f values are equal (x(y) is then not a function of y).

    Stops (converged) on the shared open-method test.
    """
    _, f, run = _setup("inverse_quadratic_interpolation", problem, x0, xtol, ftol, max_iter)
    h = _offset(run.x0, delta)
    xa, xb = run.x0 - h, run.x0 + h
    fa = float(f(xa))
    note = raised(f)
    fb = float(f(xb))
    note = note or raised(f)
    if (done := run.start({"auxiliary": [[xa, fa], [xb, fb]]})) is not None:
        return done
    if not finite(fa, fb):
        return run.breakdown(_aux_message([(xa, fa), (xb, fb)], h, delta, note))
    while True:
        if (done := run.at_limit()) is not None:
            return done
        xc, fc = run.xs[-1], run.fs[-1]
        if fa == fb or fa == fc or fb == fc:
            return run.breakdown("two interpolation points have equal f values")
        da, db, dc = (fa - fb) * (fa - fc), (fb - fa) * (fb - fc), (fc - fa) * (fc - fb)
        if da == 0.0 or db == 0.0 or dc == 0.0:  # the products underflowed
            return run.breakdown("the f values are too close: the interpolation weights underflow")
        x_new = xa * fb * fc / da + xb * fa * fc / db + xc * fa * fb / dc
        f_new = float(f(x_new))
        # Coefficients of x(y) = a y² + b y + c from divided differences in y (display only).
        dd_ab = (xb - xa) / (fb - fa)
        dd_bc = (xc - xb) / (fc - fb)
        a2 = (dd_bc - dd_ab) / (fc - fa)
        info = {
            "points": [[xa, fa], [xb, fb], [xc, fc]],
            "inverse_parabola": {
                "a": a2,
                "b": dd_ab - a2 * (fa + fb),
                "c": xa - dd_ab * fa + a2 * fa * fb,
            },
        }
        if (done := run.step(x_new, f_new, info)) is not None:
            return done
        xa, fa, xb, fb = xb, fb, xc, fc


# --------------------------------------------------------------------------------------
# Fixed-point iteration
# --------------------------------------------------------------------------------------


@register(
    id="fixed_point",
    family="roots",
    name="Fixed-point iteration",
    params=(
        *OPEN_PARAMS,
        ParamSpec(
            "lam",
            0.1,
            min=-5.0,
            max=5.0,
            help="λ in g(x) = x − λ f(x); converges near r when 0 < λ f′(r) < 2.",
        ),
    ),
    needs=("f", "x0"),
    order="linear (rate |1 − λ f′(r)|)",
    summary="Iterate x ← g(x) = x − λ f(x); a root of f is a fixed point of g.",
    references=("Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.2 and Thm. 2.4",),
)
def fixed_point(
    problem: Problem | ScalarFn,
    *,
    x0: Any = None,
    xtol: float = 1e-10,
    ftol: float = 0.0,
    max_iter: int = 100,
    lam: float = 0.1,
) -> Result:
    """Fixed-point iteration x_{k+1} = g(x_k) with g(x) = x − λ f(x) (B&F Alg. 2.2).

    By the fixed-point theorem (B&F Thm. 2.4) the iteration converges near a root r when
    |g'(r)| = |1 − λ f'(r)| < 1, linearly with that rate (monotonically if g'(r) > 0,
    alternating if g'(r) < 0). λ = 1/f'(r) would give Newton's rate locally. Example:
    λ = −1 on f(x) = cos x − x gives the classic iteration x ← cos x.

    Stops (converged) on the shared open-method test: f(x_k) == 0, |f(x_k)| ≤ ``ftol``, a
    sign change within tol, or the a posteriori estimate
    |x_k − x_{k−1}|·max(1, ρ/(1 − ρ)) ≤ xtol + 2ε|x_k| with the observed rate
    ρ = the larger of the last two step ratios, both < 1 (B&F §2.2; :func:`step_test_estimate`).
    NOTE: B&F Alg. 2.2 stops on |x_k − x_{k−1}| < TOL; for a rate ρ near 1 the error is then
    up to ρ/(1 − ρ) times larger than TOL, so the a posteriori form is used.

    NOTE: the parameter is named ``lam`` because ``lambda`` is a Python keyword.
    """
    _, f, run = _setup("fixed_point", problem, x0, xtol, ftol, max_iter)
    if (done := run.start()) is not None:
        return done
    while True:
        if (done := run.at_limit()) is not None:
            return done
        x, fx = run.xs[-1], run.fs[-1]
        x_new = x - lam * fx
        f_new = float(f(x_new))
        info = {"cobweb": [[x, x], [x, x_new], [x_new, x_new]], "lam": lam}
        if (done := run.step(x_new, f_new, info)) is not None:
            return done


#: Parity fixtures exported for the web app: (method_id, problem_id, params).
FIXTURE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("newton", "cubic", {}),
    ("newton", "newton_cycle", {}),
    ("secant", "kepler", {}),
    ("halley", "wilkinson5", {}),
    ("steffensen", "sqrt2", {}),
    ("muller", "cos_minus_x", {}),
    ("inverse_quadratic_interpolation", "cubic", {}),
    ("fixed_point", "cos_minus_x", {"lam": -1.0}),
]
