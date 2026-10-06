/**
 * The constants the certified schedules and OGM need (L, and μ for the strongly convex
 * schedule) when the problem states none: the extreme eigenvalues of ∇²f over the plotted region,
 * offered by the lab as a value to try (ConstantHint.tsx). They bound the curvature on that
 * region only, not globally, so a certificate computed with them holds while the run stays there.
 */
import type { Problem2D } from '../../core/types';

/** The run failed because the problem gives no L (or μ) and is not a quadratic. */
export const needsConstant = (error: string | undefined): boolean =>
  typeof error === 'string' && /give (L|mu) > 0 explicitly; the certificates need/.test(error);

/** x rounded up to two significant digits (2,431.7 → 2,500; 0.0123 → 0.013). */
export function niceUp(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) return x;
  const e = Math.floor(Math.log10(x)) - 1;
  const p = 10 ** e;
  // Guard against 1.2000000000000002-style rounding after the division.
  return Number((Math.ceil(x / p - 1e-9) * p).toPrecision(2));
}

/**
 * max λ_max(∇²f) and min λ_min(∇²f) over an n × n grid of the problem's domain (null without a
 * Hessian or with non-finite values everywhere).
 */
export function curvatureOnView(
  problem: Problem2D,
  n = 41,
): { lMax: number; muMin: number } | null {
  if (!problem.hess) return null;
  const [[x0, x1], [y0, y1]] = problem.domain;
  let lMax = -Infinity;
  let muMin = Infinity;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const x = x0 + ((x1 - x0) * i) / (n - 1);
      const y = y0 + ((y1 - y0) * j) / (n - 1);
      const H = problem.hess([x, y]);
      const a = H[0][0];
      const c = H[1][1];
      const b = (H[0][1] + H[1][0]) / 2;
      const m = (a + c) / 2;
      const r = Math.hypot((a - c) / 2, b);
      if (!Number.isFinite(m) || !Number.isFinite(r)) continue;
      lMax = Math.max(lMax, m + r);
      muMin = Math.min(muMin, m - r);
    }
  return Number.isFinite(lMax) ? { lMax, muMin } : null;
}
