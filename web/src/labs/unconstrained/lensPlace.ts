/**
 * Where the step lens sits on the landscape: the corner of the plot that covers the least of the
 * drawing. The lens never covers 𝐱₀ or the focused method's current step (𝐱ₖ₋₁ and 𝐱ₖ); after
 * that it avoids the known minimizers, the focused path and the other paths, in that order. It
 * keeps its corner until another corner is clearly better, so it does not jump at every step.
 */

export type Corner = 'tl' | 'tr' | 'bl' | 'br';

/** The lens square (CSS px). */
export const LENS_SIZE = 196;
/**
 * Insets from the plot's edges: the y tick labels are drawn inside the left edge, the x tick
 * labels inside the bottom edge, and the view toolbar (zoom in, zoom out, reset) in the top-right
 * corner, so a lens in that corner sits under it (`toolbar`: 10 px + the 32-px bar + 8 px).
 */
const INSET = { top: 10, left: 44, right: 10, bottom: 30, toolbar: 50 };

/** Ties are broken in this order (the hover readout sits at the top right). */
export const CORNERS: readonly Corner[] = ['tl', 'tr', 'bl', 'br'];

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Top inset of a corner: the top-right lens clears the view toolbar. */
const topOf = (c: Corner) => (c === 'tr' ? INSET.toolbar : INSET.top);

/** The lens box of a corner in a plot of width × height (CSS px), or null if it does not fit. */
export function cornerRect(c: Corner, width: number, height: number): Rect | null {
  const s = LENS_SIZE;
  if (width < s + INSET.left + INSET.right || height < s + topOf(c) + INSET.bottom) return null;
  const x = c === 'tl' || c === 'bl' ? INSET.left : width - INSET.right - s;
  const y = c === 'tl' || c === 'tr' ? topOf(c) : height - INSET.bottom - s;
  return { x, y, w: s, h: s };
}

/** CSS position of the lens for a corner (inside `.plot`). */
export function cornerStyle(c: Corner): Record<string, string> {
  return {
    top: c === 'tl' || c === 'tr' ? `${topOf(c)}px` : 'auto',
    bottom: c === 'bl' || c === 'br' ? `${INSET.bottom}px` : 'auto',
    left: c === 'tl' || c === 'bl' ? `${INSET.left}px` : 'auto',
    right: c === 'tr' || c === 'br' ? `${INSET.right}px` : 'auto',
  };
}

/** A point on the plot (CSS px) and how much covering it costs. */
export interface Mark {
  x: number;
  y: number;
  w: number;
}

/** Covering one of these is never acceptable: 𝐱₀, the focused 𝐱ₖ₋₁ and 𝐱ₖ. */
export const HARD = 1e6;
export const W_MINIMUM = 200;
export const W_FOCUS_PATH = 4;
export const W_PATH = 1;

/**
 * The marks of a polyline (pixel points), with extra samples every ~20 px along long segments so
 * a long step that crosses a corner counts even when neither end is in it.
 */
export function pathMarks(
  px: readonly (readonly [number, number])[],
  w: number,
  maxPts = 600,
): Mark[] {
  const out: Mark[] = [];
  const stride = Math.max(1, Math.ceil(px.length / maxPts));
  for (let i = 0; i < px.length; i += stride) {
    const p = px[i];
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
    out.push({ x: p[0], y: p[1], w });
    const q = px[Math.min(px.length - 1, i + stride)];
    if (!q || q === p || !Number.isFinite(q[0]) || !Number.isFinite(q[1])) continue;
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    const n = Math.min(40, Math.floor(len / 20));
    for (let j = 1; j <= n; j++) {
      const t = j / (n + 1);
      out.push({ x: p[0] + t * (q[0] - p[0]), y: p[1] + t * (q[1] - p[1]), w });
    }
  }
  return out;
}

/** The cost of a lens at rect r: the weight of the marks inside it (with a margin of pad px). */
export function cost(r: Rect, marks: readonly Mark[], pad = 10): number {
  let c = 0;
  for (const m of marks)
    if (m.x >= r.x - pad && m.x <= r.x + r.w + pad && m.y >= r.y - pad && m.y <= r.y + r.h + pad)
      c += m.w;
  return c;
}

/**
 * The best corner for the lens, or null when no corner is free of the hard marks (then the lens
 * is not shown: it must not hide 𝐱₀ or the step it magnifies). `current` is kept unless another
 * corner costs clearly less.
 */
export function pickCorner(
  marks: readonly Mark[],
  width: number,
  height: number,
  current: Corner | null,
): Corner | null {
  const costs = new Map<Corner, number>();
  for (const c of CORNERS) {
    const r = cornerRect(c, width, height);
    if (r) costs.set(c, cost(r, marks));
  }
  let best: Corner | null = null;
  for (const c of CORNERS) {
    const v = costs.get(c);
    if (v === undefined || v >= HARD) continue;
    if (best === null || v < costs.get(best)!) best = c;
  }
  if (best === null) return null;
  const cur = current !== null ? costs.get(current) : undefined;
  if (current !== null && cur !== undefined && cur < HARD) {
    const b = costs.get(best)!;
    if (b >= cur - (2 + 0.2 * cur)) return current;
  }
  return best;
}
