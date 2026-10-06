/**
 * The step lens, as data: the square (data units) around the focused method's current step and
 * the marks it drew, and whether the step is too small to read on the landscape. A gradient step
 * near a minimizer, or an Adam step of length α, is a few pixels on a view of the whole domain;
 * the lens magnifies it so its geometry can be read.
 */
import type { Step } from '../../core/types';
import type { Geometry, Pt } from './geometry';
import { isVec } from './geometry';

export interface LensBox {
  cx: number;
  cy: number;
  /** Half the side of the square (data units). */
  half: number;
}

/** Points that the step's marks pass through (ellipses and level curves excluded: they may be huge). */
function markPoints(g: Geometry): Pt[] {
  const out: Pt[] = [];
  for (const o of g.overlays) {
    switch (o.kind) {
      case 'arrow':
      case 'segment':
        out.push(o.from, o.to);
        break;
      case 'point':
        out.push(o.at);
        break;
      case 'polygon':
      case 'polyline':
        out.push(...o.points);
        break;
      case 'disk':
        out.push(
          [o.center[0] - o.radius, o.center[1] - o.radius],
          [o.center[0] + o.radius, o.center[1] + o.radius],
        );
        break;
      default:
        break;
    }
  }
  return out;
}

/**
 * The lens square around step g (𝐱_{g−1}, 𝐱_g and every mark), padded so labels fit, or null
 * when there is no step to show.
 */
export function lensBox(g: Geometry, trace: readonly Step[], k: number): LensBox | null {
  if (k < 1 || k >= trace.length) return null;
  const a = trace[k - 1].x;
  const b = trace[k].x;
  if (!isVec(a) || !isVec(b)) return null;
  const pts: Pt[] = [a, b, ...markPoints(g)].filter(
    (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]),
  );
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs),
    x1 = Math.max(...xs),
    y0 = Math.min(...ys),
    y1 = Math.max(...ys);
  let [bx0, bx1, by0, by1] = [x0, x1, y0, y1];
  const size = Math.max(x1 - x0, y1 - y0);
  const scale = Math.max(Math.abs(a[0]), Math.abs(a[1]), 1);
  // Steps at the rounding level of 𝐱 (a converged run's last steps) are not magnified.
  if (!(size > 1e-8 * scale)) return null;
  // Ellipses of the step's own size (the adaptive step ellipse, a model through 𝐱ₖ₋₁) are
  // part of the picture; long thin model ellipses along a valley are not, and are clipped.
  for (const o of g.overlays) {
    if (o.kind !== 'ellipse') continue;
    const r = o.radius ?? 1;
    const det = o.matrix[0][0] * o.matrix[1][1] - o.matrix[0][1] * o.matrix[1][0];
    if (!(det > 0)) continue;
    const ex = r * Math.sqrt(o.matrix[1][1] / det);
    const ey = r * Math.sqrt(o.matrix[0][0] / det);
    if (2 * Math.max(ex, ey) > 6 * size) continue;
    bx0 = Math.min(bx0, o.center[0] - ex);
    bx1 = Math.max(bx1, o.center[0] + ex);
    by0 = Math.min(by0, o.center[1] - ey);
    by1 = Math.max(by1, o.center[1] + ey);
  }
  return { cx: (bx0 + bx1) / 2, cy: (by0 + by1) / 2, half: Math.max(bx1 - bx0, by1 - by0) * 0.62 };
}

/** The lens is shown when the step's square is under this fraction of the domain's span. */
export const LENS_FRACTION = 0.25;

export function needsLens(box: LensBox | null, span: number): boolean {
  return box !== null && 2 * box.half < LENS_FRACTION * span;
}

/** A round length for the scale bar: 1, 2 or 5 × 10ⁿ, at most `max`. */
export function niceLength(max: number): number {
  if (!(max > 0) || !Number.isFinite(max)) return 0;
  const p = 10 ** Math.floor(Math.log10(max));
  for (const m of [5, 2, 1]) if (m * p <= max) return m * p;
  return p;
}
