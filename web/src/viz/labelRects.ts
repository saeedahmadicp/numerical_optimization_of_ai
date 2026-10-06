/**
 * A per-canvas registry of the text labels drawn in the current frame, so a label drawn later
 * (an off-view note, a minimizer's coordinates) can move clear of an earlier one instead of
 * printing over it. `useCanvas` clears it before every paint.
 */
export interface LabelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const REGISTRY = new WeakMap<CanvasRenderingContext2D, LabelRect[]>();

/** Forget the labels of the previous frame (called by `useCanvas` before `draw`). */
export function resetLabelRects(ctx: CanvasRenderingContext2D): void {
  REGISTRY.set(ctx, []);
}

/** Record a drawn label's box (CSS px). */
export function addLabelRect(ctx: CanvasRenderingContext2D, r: LabelRect): void {
  const list = REGISTRY.get(ctx);
  if (list) list.push(r);
  else REGISTRY.set(ctx, [r]);
}

/** True when `r` (grown by `pad`) overlaps a label already drawn in this frame. */
export function hitsLabel(ctx: CanvasRenderingContext2D, r: LabelRect, pad = 2): boolean {
  const list = REGISTRY.get(ctx);
  if (!list) return false;
  return list.some(
    (o) =>
      r.x - pad < o.x + o.w &&
      r.x + r.w + pad > o.x &&
      r.y - pad < o.y + o.h &&
      r.y + r.h + pad > o.y,
  );
}

/** The area by which `r` (grown by `pad`) overlaps the labels already drawn in this frame. */
export function labelOverlap(ctx: CanvasRenderingContext2D, r: LabelRect, pad = 2): number {
  const list = REGISTRY.get(ctx);
  if (!list) return 0;
  let area = 0;
  for (const o of list) {
    const w = Math.min(r.x + r.w + pad, o.x + o.w) - Math.max(r.x - pad, o.x);
    const h = Math.min(r.y + r.h + pad, o.y + o.h) - Math.max(r.y - pad, o.y);
    if (w > 0 && h > 0) area += w * h;
  }
  return area;
}
