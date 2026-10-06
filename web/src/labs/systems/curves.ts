/**
 * Zero curves Fᵢ(x, y) = 0 drawn as chained polylines, so a dash pattern runs along the curve
 * (marching squares emits one tiny segment per grid cell; dashing those separately reads as solid).
 */
import { implicitSegments, drawMath, type MathRun, type View2D } from '../../viz';

/** Join marching-squares segments `[xa, ya, xb, yb, …]` into polylines (shared endpoints). */
export function chainSegments(segs: readonly number[], tol: number): [number, number][][] {
  const n = segs.length / 4;
  const key = (x: number, y: number) => `${Math.round(x / tol)},${Math.round(y / tol)}`;
  const ends = new Map<string, number[]>(); // endpoint → [seg * 2 + end]
  for (let s = 0; s < n; s++)
    for (let e = 0; e < 2; e++) {
      const k = key(segs[4 * s + 2 * e], segs[4 * s + 2 * e + 1]);
      const list = ends.get(k);
      if (list) list.push(2 * s + e);
      else ends.set(k, [2 * s + e]);
    }
  const used = new Uint8Array(n);
  const pt = (s: number, e: number): [number, number] => [
    segs[4 * s + 2 * e],
    segs[4 * s + 2 * e + 1],
  ];
  /** From the end `e` of segment `s`, follow unused neighbors; returns the points after `s`. */
  const walk = (s: number, e: number): [number, number][] => {
    const out: [number, number][] = [];
    let cs = s,
      ce = e;
    for (;;) {
      const [x, y] = pt(cs, ce);
      const next = (ends.get(key(x, y)) ?? []).find((v) => !used[v >> 1]);
      if (next === undefined) break;
      cs = next >> 1;
      used[cs] = 1;
      ce = 1 - (next & 1); // leave through the other end
      out.push(pt(cs, ce));
    }
    return out;
  };
  const lines: [number, number][][] = [];
  for (let s = 0; s < n; s++) {
    if (used[s]) continue;
    used[s] = 1;
    const fwd = walk(s, 1);
    const back = walk(s, 0);
    lines.push([...back.reverse(), pt(s, 0), pt(s, 1), ...fwd]);
  }
  return lines;
}

interface CachedLines {
  /** The data region that was marched (the view at the time, padded by half a span per side). */
  x: [number, number];
  y: [number, number];
  /** View spans it was marched for (the resolution follows the zoom). */
  span: number;
  lines: [number, number][][];
}

const CACHE = new Map<string, CachedLines[]>();

/**
 * The zero curve of `g` over the visible region, in data coordinates. Cached per curve over a
 * padded region, so panning (and small zooms) reuse it instead of marching again every frame.
 */
function curveLines(g: (x: number, y: number) => number, cacheKey: string, view: View2D) {
  const x0 = view.x.invert(0),
    x1 = view.x.invert(view.width);
  const y0 = view.y.invert(view.height),
    y1 = view.y.invert(0);
  const span = Math.max(x1 - x0, y1 - y0);
  const list = CACHE.get(cacheKey) ?? [];
  const hit = list.find(
    (c) =>
      c.x[0] <= x0 &&
      c.x[1] >= x1 &&
      c.y[0] <= y0 &&
      c.y[1] >= y1 &&
      span / c.span > 0.75 &&
      span / c.span < 1.34,
  );
  if (hit) return hit.lines;
  const px = (x1 - x0) / 2,
    py = (y1 - y0) / 2;
  const X: [number, number] = [x0 - px, x1 + px];
  const Y: [number, number] = [y0 - py, y1 + py];
  // Same cell size as before (about 3 px), over twice the span.
  const nx = 2 * Math.max(48, Math.min(260, Math.round(view.width / 3)));
  const ny = 2 * Math.max(48, Math.min(260, Math.round(view.height / 3)));
  const segs = implicitSegments(g, X, Y, nx, ny, 0);
  const lines = chainSegments(segs, Math.max(X[1] - X[0], Y[1] - Y[0]) * 1e-9);
  list.unshift({ x: X, y: Y, span, lines });
  list.length = Math.min(list.length, 4);
  CACHE.set(cacheKey, list);
  while (CACHE.size > 24) CACHE.delete(CACHE.keys().next().value as string);
  return lines;
}

export interface ZeroCurve {
  g: (x: number, y: number) => number;
  cacheKey: string;
  dash?: number[];
  label: MathRun[];
}

/**
 * Stroke every curve (halo, then ink), then label each at its point nearest the top right, away
 * from the other labels and from `avoid` (data points such as 𝐱₀, the roots, the current iterate).
 */
export function drawZeroCurves(
  ctx: CanvasRenderingContext2D,
  view: View2D,
  curves: readonly ZeroCurve[],
  avoid: readonly (readonly [number, number])[] = [],
) {
  const avoidPx = avoid
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))
    .map((p) => view.toPx(p[0], p[1]));
  const c = view.colors;
  const placed: [number, number][] = [];
  for (const cv of curves) {
    const lines = curveLines(cv.g, cv.cacheKey, view);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const path = new Path2D();
    for (const line of lines) {
      line.forEach(([x, y], i) => {
        const [px, py] = view.toPx(x, y);
        if (i === 0) path.moveTo(px, py);
        else path.lineTo(px, py);
      });
    }
    ctx.strokeStyle = c.halo;
    ctx.globalAlpha = 0.65;
    ctx.lineWidth = 4;
    ctx.stroke(path);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.6;
    ctx.setLineDash(cv.dash ?? []);
    ctx.stroke(path);
    ctx.restore();
    // Direct label: the curve point nearest the top-right corner, away from the other labels.
    let best: [number, number] | null = null,
      score = -Infinity;
    for (const line of lines)
      for (let i = 0; i < line.length; i += 3) {
        const [px, py] = view.toPx(line[i][0], line[i][1]);
        if (px < 70 || px > view.width - 90 || py < 70 || py > view.height - 40) continue;
        if (placed.some(([qx, qy]) => Math.hypot(px - qx, py - qy) < 80)) continue;
        // The label box spans [px + 7, px + 57] × [py − 21, py − 4]; keep 24 px around markers.
        if (
          avoidPx.some(([qx, qy]) => qx > px - 17 && qx < px + 81 && qy > py - 45 && qy < py + 20)
        )
          continue;
        const s = px - py;
        if (s > score) {
          score = s;
          best = [px, py];
        }
      }
    if (best) {
      placed.push(best);
      drawMath(ctx, cv.label, best[0] + 7, best[1] - 7, { size: 13, color: c.text, halo: c.halo });
    }
  }
}
