/**
 * Off-view iterates as chips: one chip per method and edge ("𝐱₇ = (2.84, −1.98)" / "Newton ·
 * 8 iterates below"), laid out along that edge clear of the inset tick labels, the stage legend,
 * the zoom bar and each other. The shared path layer still dashes the steps that leave the view
 * and draws its chevrons; only its per-step labels (which pile up when a run bounces outside the
 * view) are replaced.
 */
import { drawMath, iterateRuns, measureMath, type View2D } from '../../viz';
import { sig } from '../../core/format';
import { LABEL_SIZE, labelFont } from '../../viz/axes';

export type Side = 'below' | 'above' | 'left' | 'right';

export interface OffViewRun {
  points: readonly (readonly [number, number])[];
  /** Last iterate reached by the playhead. */
  reached: number;
  /** Short method name ("Newton"). */
  name: string;
  color: string;
  muted: boolean;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface OffViewChip {
  run: number;
  side: Side;
  /** Iterates of this run outside the view on this side (up to `reached`). */
  count: number;
  /** The latest of them. */
  k: number;
  at: readonly [number, number];
  /** The chip box in CSS px. */
  box: Rect;
}

const PAD_X = 7;
const LINE1 = LABEL_SIZE;
/** Chip height: two lines (the iterate in math, the count in the UI face) at LABEL_SIZE. */
const H = 38;

/** Which side of the rectangle [0, w] × [0, h] a pixel point is on (null: inside). */
export function sideOfPx(px: number, py: number, w: number, h: number): Side | null {
  if (py > h) return 'below';
  if (py < 0) return 'above';
  if (px < 0) return 'left';
  if (px > w) return 'right';
  return null;
}

const overlaps = (a: Rect, b: Rect, gap = 4) =>
  a.x < b.x + b.w + gap && b.x < a.x + a.w + gap && a.y < b.y + b.h + gap && b.y < a.y + a.h + gap;

export const sideWords: Record<Side, string> = {
  below: 'below',
  above: 'above',
  left: 'to the left',
  right: 'to the right',
};

export function chipLine2(name: string, count: number, side: Side): string {
  return count === 1
    ? `${name} · ${sideWords[side]} the view`
    : `${name} · ${count} iterates ${sideWords[side]}`;
}

/**
 * Lay the chips out. `measure(text1, text2)` gives a chip's width; `obstacles` are boxes to keep
 * clear of (legend, zoom bar). Pure, so it is unit-tested without a canvas.
 */
export function layoutOffView(
  runs: readonly OffViewRun[],
  toPx: (x: number, y: number) => [number, number],
  width: number,
  height: number,
  obstacles: readonly Rect[],
  measure: (run: number, k: number, at: readonly [number, number], line2: string) => number,
): OffViewChip[] {
  // Bands the inset axes use: x tick labels along the bottom, y tick labels along the left.
  const bottom = height - 24;
  const left = 40;
  const top = 8;
  const right = width - 8;
  const groups: OffViewChip[] = [];
  runs.forEach((r, ri) => {
    const by = new Map<Side, { count: number; k: number }>();
    for (let k = 1; k <= Math.min(r.reached, r.points.length - 1); k++) {
      const [x, y] = r.points[k];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const [px, py] = toPx(x, y);
      const side = sideOfPx(px, py, width, height);
      if (!side) continue;
      const g = by.get(side);
      if (g) {
        g.count++;
        g.k = k;
      } else by.set(side, { count: 1, k });
    }
    for (const [side, g] of by)
      groups.push({
        run: ri,
        side,
        count: g.count,
        k: g.k,
        at: r.points[g.k],
        box: { x: 0, y: 0, w: 0, h: H },
      });
  });
  // The focused (unmuted) runs first: they get the best spots.
  groups.sort((a, b) => Number(runs[a.run].muted) - Number(runs[b.run].muted));
  const placed: Rect[] = [];
  const out: OffViewChip[] = [];
  for (const g of groups) {
    const r = runs[g.run];
    const w = measure(g.run, g.k, g.at, chipLine2(r.name, g.count, g.side));
    const [qx, qy] = toPx(g.at[0], g.at[1]);
    const clampX = (x: number) => Math.max(left, Math.min(right - w, x));
    const clampY = (y: number) => Math.max(top, Math.min(bottom - H, y));
    // Preferred spot: on the edge, beside (not over) where the step to the iterate leaves the view.
    const besideX = qx + 14 + w <= right ? qx + 14 : qx - 14 - w;
    const besideY = qy + 10 + H <= bottom ? qy + 10 : qy - 10 - H;
    let cand: Rect;
    if (g.side === 'below') cand = { x: clampX(besideX), y: bottom - H, w, h: H };
    else if (g.side === 'above') cand = { x: clampX(besideX), y: top, w, h: H };
    else if (g.side === 'left') cand = { x: left, y: clampY(besideY), w, h: H };
    else cand = { x: right - w, y: clampY(besideY), w, h: H };
    const horizontal = g.side === 'below' || g.side === 'above';
    // Slide along the edge (both ways), then step inward a row, until the chip is clear.
    let found: Rect | null = null;
    for (let row = 0; row < 4 && !found; row++) {
      const inward = row * (H + 4) * (g.side === 'below' || g.side === 'right' ? -1 : 1);
      for (let i = 0; i < 24 && !found; i++) {
        const d = (i % 2 === 0 ? 1 : -1) * Math.ceil(i / 2) * 24;
        const c: Rect = horizontal
          ? { ...cand, x: clampX(cand.x + d), y: cand.y + inward }
          : {
              ...cand,
              y: clampY(cand.y + d),
              x: cand.x + (g.side === 'left' ? 1 : -1) * Math.abs(inward),
            };
        if (c.y < top || c.y + H > bottom || c.x < left || c.x + w > right) continue;
        if (placed.some((p) => overlaps(p, c)) || obstacles.some((o) => overlaps(o, c))) continue;
        found = c;
      }
    }
    if (!found) continue; // no room: the chevrons still show where the path went
    placed.push(found);
    out.push({ ...g, box: found });
  }
  return out;
}

/** Draw the chips (call from the plot's overlay). */
export function drawOffView(
  ctx: CanvasRenderingContext2D,
  view: View2D,
  runs: readonly OffViewRun[],
  obstacles: readonly Rect[],
): void {
  const c = view.colors;
  const font2 = labelFont(c);
  const line1 = (k: number, at: readonly [number, number]) =>
    iterateRuns('x', k, `(${sig(at[0], 3)}, ${sig(at[1], 3)})`);
  ctx.save();
  ctx.font = font2;
  const measure = (_run: number, k: number, at: readonly [number, number], l2: string) => {
    ctx.font = font2;
    return (
      Math.ceil(Math.max(measureMath(ctx, line1(k, at), LINE1), ctx.measureText(l2).width)) +
      2 * PAD_X +
      10
    );
  };
  const chips = layoutOffView(runs, view.toPx, view.width, view.height, obstacles, measure);
  for (const ch of chips) {
    const r = runs[ch.run];
    const { x, y, w, h } = ch.box;
    ctx.globalAlpha = r.muted ? 0.75 : 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 5);
    ctx.fillStyle = c.surface;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = c.grid || c.axis;
    ctx.stroke();
    // Method color bar on the leading edge.
    ctx.fillStyle = r.color;
    ctx.beginPath();
    ctx.roundRect(x + 4, y + 6, 3, h - 12, 1.5);
    ctx.fill();
    drawMath(ctx, line1(ch.k, ch.at), x + PAD_X + 6, y + 16, { size: LINE1, color: c.text });
    ctx.font = font2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = c.text2;
    ctx.fillText(chipLine2(r.name, ch.count, ch.side), x + PAD_X + 6, y + 31);
  }
  ctx.restore();
}
