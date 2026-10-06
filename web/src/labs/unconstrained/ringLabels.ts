/**
 * Labelled rings that sit close together on screen (Anderson's history: iterates that coincide
 * near a stationary point) would print their labels over each other. This keeps a ring's label
 * only when the ring is farther than the label's width from every ring already labelled; the
 * others keep their ring and lose the label. Rings are taken largest first (for AA the ring size is
 * |cᵢ|, so the weights that matter keep their numbers), then newest first.
 */
import type { Overlay2D } from '../../viz/overlays2d';
import { measureMath, type MathRun } from '../../viz/mathText';
import { labelFont } from '../../viz/axes';
import type { ChartColors } from '../../ui/colors';

/** The size drawOverlays2D sets point labels in (its drawLabel default). */
const LABEL_SIZE = 12;

type Pt = readonly [number, number];

export function thinRingLabels(
  ctx: CanvasRenderingContext2D,
  overlays: readonly Overlay2D[],
  toPx: (x: number, y: number) => [number, number],
  colors: Pick<ChartColors, 'fontSans'>,
): readonly Overlay2D[] {
  const rings = overlays
    .map((o, i) => ({ o, i }))
    .filter(
      (e): e is { o: Extract<Overlay2D, { kind: 'point' }>; i: number } =>
        e.o.kind === 'point' && e.o.shape === 'ring' && e.o.label !== undefined,
    );
  if (rings.length < 2) return overlays;
  ctx.save();
  ctx.font = labelFont(colors, LABEL_SIZE - 0.5);
  const sized = rings.map(({ o, i }) => ({
    i,
    at: toPx((o.at as Pt)[0], (o.at as Pt)[1]),
    r: o.radius ?? 0,
    w:
      typeof o.label === 'string'
        ? ctx.measureText(o.label).width
        : measureMath(ctx, o.label as readonly MathRun[], LABEL_SIZE),
  }));
  ctx.restore();
  sized.sort((a, b) => b.r - a.r || b.i - a.i);
  const kept: typeof sized = [];
  const drop = new Set<number>();
  for (const s of sized) {
    const clear = kept.every(
      (q) => Math.hypot(s.at[0] - q.at[0], s.at[1] - q.at[1]) > Math.max(s.w, q.w),
    );
    if (clear) kept.push(s);
    else drop.add(s.i);
  }
  if (drop.size === 0) return overlays;
  return overlays.map((o, i) => (drop.has(i) ? { ...o, label: undefined } : o));
}
