/**
 * Label placement for the step lens (pure, tested).
 *
 * The lens labels its arrows and points itself instead of through the overlay labels, which sit
 * at a fixed offset and collide when several short arrows meet at one point. Each label offers
 * candidate positions (both sides of an arrow's shaft, near and far, then beyond its head; the
 * four sides of a point); they are taken in priority order, and a label takes the first
 * candidate that
 *
 *   - lies inside the lens with a margin,
 *   - stays clear of the reserved corners (the "update k" text and the scale bar),
 *   - does not overlap a label already placed,
 *
 * preferring candidates that cross no drawn segment (the arrows' shafts). A label with no
 * admissible candidate is dropped: an unlabelled mark is better than an unreadable one.
 */

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type Align = 'left' | 'center' | 'right';

export interface Candidate {
  /** Anchor in px (vertical middle of the text). */
  x: number;
  y: number;
  align: Align;
}

export interface LabelRequest {
  /** Width and height of the text box, px. */
  w: number;
  h: number;
  candidates: Candidate[];
}

export interface LayoutOptions {
  width: number;
  height: number;
  margin?: number;
  reserved?: readonly Rect[];
  /** Segments [ax, ay, bx, by] in px that labels should avoid crossing. */
  segments?: readonly (readonly [number, number, number, number])[];
}

export function boxOf(c: Candidate, w: number, h: number, pad = 2): Rect {
  const x0 = c.align === 'left' ? c.x : c.align === 'right' ? c.x - w : c.x - w / 2;
  return { x0: x0 - pad, y0: c.y - h / 2 - pad, x1: x0 + w + pad, y1: c.y + h / 2 + pad };
}

export const overlaps = (a: Rect, b: Rect) =>
  a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** Does the segment (ax, ay)–(bx, by) meet the rectangle? (Liang–Barsky clipping.) */
export function segmentHitsRect(
  [ax, ay, bx, by]: readonly [number, number, number, number],
  r: Rect,
): boolean {
  const dx = bx - ax,
    dy = by - ay;
  let t0 = 0,
    t1 = 1;
  const clip = (p: number, q: number) => {
    if (p === 0) return q >= 0;
    const t = q / p;
    if (p < 0) {
      if (t > t1) return false;
      if (t > t0) t0 = t;
    } else {
      if (t < t0) return false;
      if (t < t1) t1 = t;
    }
    return true;
  };
  return (
    clip(-dx, ax - r.x0) &&
    clip(dx, r.x1 - ax) &&
    clip(-dy, ay - r.y0) &&
    clip(dy, r.y1 - ay) &&
    t0 <= t1
  );
}

/** Place the labels in order; returns the chosen candidate of each, or null (dropped). */
export function placeLabels(reqs: readonly LabelRequest[], o: LayoutOptions): (Candidate | null)[] {
  const m = o.margin ?? 3;
  const placed: Rect[] = [];
  const blocked = [...(o.reserved ?? [])];
  const segs = o.segments ?? [];
  return reqs.map((r) => {
    let best: { c: Candidate; box: Rect; hits: number } | null = null;
    for (const c of r.candidates) {
      const box = boxOf(c, r.w, r.h);
      if (box.x0 < m || box.y0 < m || box.x1 > o.width - m || box.y1 > o.height - m) continue;
      if (blocked.some((b) => overlaps(b, box)) || placed.some((b) => overlaps(b, box))) continue;
      const hits = segs.filter((s) => segmentHitsRect(s, box)).length;
      if (!best || hits < best.hits) best = { c, box, hits };
      if (hits === 0) break;
    }
    if (!best) return null;
    placed.push(best.box);
    return best.c;
  });
}

/** Candidates beside an arrow's shaft (px): left side, right side, near then far, then the head. */
export function arrowCandidates(
  a: readonly [number, number],
  b: readonly [number, number],
): Candidate[] {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  const ux = (b[0] - a[0]) / len,
    uy = (b[1] - a[1]) / len;
  const nx = -uy,
    ny = ux;
  const out: Candidate[] = [];
  // Text extends horizontally: on a side whose normal points right, anchor the text at its left.
  const side = (sx: number): Align => (sx > 0.35 ? 'left' : sx < -0.35 ? 'right' : 'center');
  for (const t of [0.5, 0.3, 0.7])
    for (const d of [10, 18])
      for (const s of [1, -1]) {
        const x = a[0] + (b[0] - a[0]) * t + s * nx * d;
        const y = a[1] + (b[1] - a[1]) * t + s * ny * d;
        out.push({ x, y, align: side(s * nx) });
      }
  out.push({ x: b[0] + ux * 10, y: b[1] + uy * 10, align: side(ux) });
  return out;
}

/** Candidates around a point: right, left, above, below, then the diagonals. */
export function pointCandidates([x, y]: readonly [number, number], r = 4): Candidate[] {
  const g = r + 6;
  return [
    { x: x + g, y, align: 'left' },
    { x: x - g, y, align: 'right' },
    { x, y: y - g - 2, align: 'center' },
    { x, y: y + g + 2, align: 'center' },
    { x: x + g * 0.8, y: y - g, align: 'left' },
    { x: x - g * 0.8, y: y - g, align: 'right' },
    { x: x + g * 0.8, y: y + g, align: 'left' },
    { x: x - g * 0.8, y: y + g, align: 'right' },
  ];
}
