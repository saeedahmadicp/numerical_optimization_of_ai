/**
 * The exact integral over a sub-interval [a, b] of a problem's domain, for the lab's
 * draggable interval. The problems carry `exact` only for their whole domain, and the Python
 * methods take the problem's `exact` even when `bracket` overrides the interval, so the lab
 * passes the methods a problem whose `exact` is the value for [a, b].
 *
 * Closed-form antiderivatives where they exist in elementary functions; otherwise (the
 * Gaussian, whose antiderivative needs erf) a composite 20-point Gauss–Legendre rule on 16
 * panels, which is exact to rounding for an entire integrand on a short interval.
 */
import { fsum, gaussLegendreRule } from '../../methods/integration/methods';

type F = (x: number) => number;

const ANTIDERIVATIVE: Record<string, F> = {
  poly3: (x) => x ** 4 / 4 - x ** 2 + x,
  exp_0_1: (x) => Math.exp(x),
  sin_0_pi: (x) => -Math.cos(x),
  runge: (x) => Math.atan(5 * x) / 5,
  sqrt_0_1: (x) => (2 / 3) * x * Math.sqrt(x),
  oscillatory: (x) => -Math.cos(10 * x) / 10,
  arctan_deriv: (x) => Math.atan(x),
  abs_kink: (x) => {
    const d = x - 0.3;
    return 0.5 * d * Math.abs(d);
  },
};

let rule20: [number[], number[]] | null = null;

/** Composite 20-point Gauss–Legendre on `panels` equal panels. */
export function gaussReference(f: F, a: number, b: number, panels = 16): number {
  rule20 ??= gaussLegendreRule(20);
  const [t, w] = rule20;
  const terms: number[] = [];
  const h = (b - a) / panels;
  for (let p = 0; p < panels; p++) {
    const lo = a + p * h;
    const mid = lo + h / 2;
    for (let i = 0; i < t.length; i++) terms.push((h / 2) * w[i] * f(mid + (h / 2) * t[i]));
  }
  return fsum(terms);
}

/**
 * ∫ₐᵇ f for the problem `id`. Returns the problem's own `exact` when [a, b] is its domain, so
 * the default view uses exactly the Python reference value.
 */
export function referenceIntegral(
  problem: { id: string; f: F; domain: readonly [number, number]; exact?: number | null },
  a: number,
  b: number,
): number {
  if (a === problem.domain[0] && b === problem.domain[1] && problem.exact != null)
    return problem.exact;
  const F = ANTIDERIVATIVE[problem.id];
  // abs_kink: ½(x − 0.3)|x − 0.3| is a C¹ antiderivative across the kink.
  if (F) return F(b) - F(a);
  return gaussReference(problem.f, a, b);
}
