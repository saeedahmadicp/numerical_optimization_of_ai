/**
 * Declarative geometry for 2-D plots (`<Contour2D overlays={…}>`, or `drawOverlays2D` on any
 * canvas with a data → pixel map). Everything is given in data coordinates.
 *
 * | kind          | draws                                                                    |
 * | ------------- | ------------------------------------------------------------------------ |
 * | `polygon`     | a closed polygon, outlined and/or filled (simplex, polytope, LP region)   |
 * | `polyline`    | an open polyline                                                          |
 * | `segment`     | a line segment                                                            |
 * | `arrow`       | an arrow from → to (gradient, search direction), optional label           |
 * | `disk`        | a circle of radius r (trust region), outlined and/or filled              |
 * | `ellipse`     | {x : (x−c)ᵀM(x−c) = r²} for a symmetric positive definite 2×2 matrix M    |
 * | `implicit`    | the curve g(x) = level, by marching squares over the visible view         |
 * | `constraints` | feasible-set shading: hatch where some gᵢ(x) > 0, boundaries gᵢ(x) = 0    |
 * | `region`      | hatch/tint where `inside(x, y)` is true                                   |
 * | `point`       | a labelled point (filled, hollow, or a × cross)                           |
 * | `text`        | a label (plain text or typeset math runs)                                 |
 *
 * Color: `slot` (0–3) uses a method color; without it, marks are ink (`--color-text`). Region
 * shading uses the neutral `--chart-region`, never a series color.
 *
 * Labels are drawn last, in one pass, with a 3 px surface halo and a collision test against the
 * other labels of the frame and the frame's edges (`labelRects.ts`): a point label tries its side,
 * then the four corners, then the other sides; a label that finds no free place is skipped rather
 * than printed over another. By default the pass runs at the end of `drawOverlays2D` (over this
 * call's marks). To draw labels over the method paths as well, defer it:
 *
 *   const labels = drawOverlays2D(ctx, view, overlays, { labels: 'defer' });
 *   drawPathLayer(ctx, paths, …);
 *   drawOverlayLabels(ctx, view, labels);            // Contour2D does this for its `overlays`
 */
import type { ChartColors } from '../ui/colors';
import { eigSym2 } from '../core/linalg';
import { isoSegments } from './contourField';
import { drawMath, measureMath, type MathRun } from './mathText';
import { addLabelRect, labelOverlap } from './labelRects';
import { LABEL_SIZE, labelFont } from './axes';

export type Pt = readonly [number, number];
type Label = string | readonly MathRun[];

interface Stroke {
  /** Method color slot (0–3); ink when omitted. */
  slot?: number;
  dashed?: boolean;
  /** Line width in CSS px (default 1.5). */
  width?: number;
  /** Opacity of the whole mark (default 1). */
  alpha?: number;
}

export type Overlay2D =
  | ({
      kind: 'polygon';
      points: readonly Pt[];
      fill?: boolean;
      stroke?: boolean;
      label?: Label;
    } & Stroke)
  | ({ kind: 'polyline'; points: readonly Pt[] } & Stroke)
  | ({ kind: 'segment'; from: Pt; to: Pt } & Stroke)
  | ({ kind: 'arrow'; from: Pt; to: Pt; label?: Label } & Stroke)
  | ({ kind: 'disk'; center: Pt; radius: number; fill?: boolean } & Stroke)
  | ({
      kind: 'ellipse';
      center: Pt;
      /** Symmetric positive definite 2×2 matrix M. */
      matrix: readonly (readonly number[])[];
      /** r in (x−c)ᵀM(x−c) = r² (default 1). */
      radius?: number;
      fill?: boolean;
    } & Stroke)
  | ({
      kind: 'implicit';
      g: (x: number, y: number) => number;
      level?: number;
      /** Identifies g for caching (problem id + constraint index). */
      cacheKey: string;
      label?: Label;
    } & Stroke)
  | {
      kind: 'constraints';
      /** Inequalities gᵢ(x) ≤ 0 (feasible when every gᵢ ≤ 0). */
      g: readonly ((x: number, y: number) => number)[];
      /** Equalities hⱼ(x) = 0, drawn as curves. */
      h?: readonly ((x: number, y: number) => number)[];
      cacheKey: string;
      /** Hatch the infeasible side (default true) and draw the boundaries (default true). */
      hatch?: boolean;
      boundary?: boolean;
    }
  | {
      kind: 'region';
      inside: (x: number, y: number) => boolean;
      cacheKey: string;
      hatch?: boolean;
      /** Faint tint under the hatch (default true). */
      tint?: boolean;
    }
  | {
      kind: 'point';
      at: Pt;
      slot?: number;
      shape?: 'dot' | 'ring' | 'cross';
      label?: Label;
      /** Where the label sits (default 'right'). */
      labelSide?: 'right' | 'left' | 'above' | 'below';
      radius?: number;
    }
  | {
      kind: 'text';
      at: Pt;
      text: Label;
      slot?: number;
      align?: 'left' | 'center' | 'right';
      size?: number;
    };

export interface OverlayView {
  toPx: (x: number, y: number) => [number, number];
  /** Inverse map (needed by `implicit`, `constraints`, `region`). */
  toData?: (px: number, py: number) => [number, number];
  width: number;
  height: number;
  dpr: number;
  colors: ChartColors;
}

const color = (c: ChartColors, slot?: number) =>
  slot === undefined ? c.text : c.series[slot % c.series.length];

function strokeStyle(ctx: CanvasRenderingContext2D, o: Stroke, c: ChartColors) {
  ctx.strokeStyle = color(c, o.slot);
  ctx.lineWidth = o.width ?? 1.5;
  ctx.setLineDash(o.dashed ? [5, 4] : []);
  ctx.globalAlpha = o.alpha ?? 1;
}

/** Ellipse {x : (x−c)ᵀM(x−c) = r²} sampled as a closed polyline (null if M is not SPD). */
export function ellipsePoints(
  center: Pt,
  M: readonly (readonly number[])[],
  r = 1,
  n = 96,
): [number, number][] | null {
  const { values, vectors } = eigSym2([
    [M[0][0], M[0][1]],
    [M[1][0], M[1][1]],
  ]);
  if (!(values[0] > 0 && values[1] > 0)) return null;
  const a = r / Math.sqrt(values[0]),
    b = r / Math.sqrt(values[1]);
  const [u, w] = vectors;
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const th = (2 * Math.PI * i) / n;
    const ca = a * Math.cos(th),
      sb = b * Math.sin(th);
    out.push([center[0] + ca * u[0] + sb * w[0], center[1] + ca * u[1] + sb * w[1]]);
  }
  return out;
}

/**
 * Segments of g(x, y) = level over the rectangle [x0, x1] × [y0, y1] on an nx × ny grid, in data
 * coordinates (flat `[xa, ya, xb, yb, …]`). Marching squares (shared with the contour field).
 */
export function implicitSegments(
  g: (x: number, y: number) => number,
  [x0, x1]: readonly [number, number],
  [y0, y1]: readonly [number, number],
  nx = 120,
  ny = 120,
  level = 0,
): number[] {
  const vals = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const y = y0 + ((y1 - y0) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      const v = g(x0 + ((x1 - x0) * i) / (nx - 1), y);
      vals[j * nx + i] = Number.isFinite(v) ? v : NaN;
    }
  }
  const raw: number[] = [];
  isoSegments(vals, nx, ny, level, raw);
  for (let k = 0; k < raw.length; k += 2) {
    raw[k] = x0 + ((x1 - x0) * raw[k]) / (nx - 1);
    raw[k + 1] = y0 + ((y1 - y0) * raw[k + 1]) / (ny - 1);
  }
  return raw;
}

// ── Caches (implicit curves and shaded regions depend only on g, the view and the size) ─────

const SEG_CACHE = new Map<string, number[]>();
const MASK_CACHE = new Map<string, HTMLCanvasElement>();
const CACHE_LIMIT = 48;
function remember<T>(cache: Map<string, T>, key: string, value: T): T {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value as string);
  return value;
}

function viewKey(v: OverlayView): string {
  if (!v.toData) return '';
  const [a, b] = v.toData(0, 0);
  const [c, d] = v.toData(v.width, v.height);
  return [a, b, c, d].map((n) => n.toPrecision(7)).join(',') + `|${v.width}x${v.height}@${v.dpr}`;
}

function curveSegments(
  g: (x: number, y: number) => number,
  key: string,
  v: OverlayView,
  level = 0,
): number[] {
  if (!v.toData) return [];
  const k = `${key}|${level}|${viewKey(v)}`;
  const hit = SEG_CACHE.get(k);
  if (hit) return hit;
  const [x0, y1] = v.toData(0, 0);
  const [x1, y0] = v.toData(v.width, v.height);
  const nx = Math.max(32, Math.min(220, Math.round(v.width / 3)));
  const ny = Math.max(32, Math.min(220, Math.round(v.height / 3)));
  return remember(SEG_CACHE, k, implicitSegments(g, [x0, x1], [y0, y1], nx, ny, level));
}

function hatchPattern(ctx: CanvasRenderingContext2D, ink: string, dpr: number) {
  const s = Math.round(7 * dpr);
  const tile = document.createElement('canvas');
  tile.width = tile.height = s;
  const t = tile.getContext('2d')!;
  t.strokeStyle = ink;
  t.lineWidth = Math.max(1, 0.9 * dpr);
  t.globalAlpha = 0.55;
  t.beginPath();
  // 45° hatch, continuous across tiles.
  t.moveTo(-1, s + 1);
  t.lineTo(s + 1, -1);
  t.moveTo(-1, 1);
  t.lineTo(1, -1);
  t.moveTo(s - 1, s + 1);
  t.lineTo(s + 1, s - 1);
  t.stroke();
  return ctx.createPattern(tile, 'repeat');
}

/** Hatched (and tinted) raster of the set where `shade(x, y)` is true, at device resolution. */
function shadedLayer(
  shade: (x: number, y: number) => boolean,
  key: string,
  v: OverlayView,
  hatch: boolean,
  tint: boolean,
): HTMLCanvasElement | null {
  if (!v.toData || v.width <= 0 || v.height <= 0) return null;
  const k = `${key}|${hatch}|${tint}|${v.colors.mode}|${viewKey(v)}`;
  const hit = MASK_CACHE.get(k);
  if (hit) return hit;
  // Mask on a coarse grid (3 CSS px cells), smoothed when scaled up.
  const cell = 3;
  const mw = Math.max(1, Math.ceil(v.width / cell)),
    mh = Math.max(1, Math.ceil(v.height / cell));
  const mask = document.createElement('canvas');
  mask.width = mw;
  mask.height = mh;
  const mctx = mask.getContext('2d')!;
  const img = mctx.createImageData(mw, mh);
  for (let j = 0; j < mh; j++)
    for (let i = 0; i < mw; i++) {
      const [x, y] = v.toData((i + 0.5) * cell, (j + 0.5) * cell);
      if (shade(x, y)) img.data[(j * mw + i) * 4 + 3] = 255;
    }
  mctx.putImageData(img, 0, 0);

  const out = document.createElement('canvas');
  out.width = Math.round(v.width * v.dpr);
  out.height = Math.round(v.height * v.dpr);
  const o = out.getContext('2d')!;
  if (tint) {
    o.fillStyle = v.colors.region;
    o.globalAlpha = 0.09;
    o.fillRect(0, 0, out.width, out.height);
    o.globalAlpha = 1;
  }
  if (hatch) {
    const pat = hatchPattern(o, v.colors.region, v.dpr);
    if (pat) {
      o.fillStyle = pat;
      o.fillRect(0, 0, out.width, out.height);
    }
  }
  o.globalCompositeOperation = 'destination-in';
  o.imageSmoothingEnabled = true;
  o.drawImage(mask, 0, 0, mw * cell * v.dpr, mh * cell * v.dpr);
  return remember(MASK_CACHE, k, out);
}

type Align = 'left' | 'center' | 'right';
/** A place a label may take: its anchor (vertical middle) and its alignment. */
type LabelSpot = readonly [x: number, y: number, align: Align];

/** A label waiting for the label pass: where it would like to go, in order of preference. */
export interface OverlayLabel {
  label: Label;
  spots: readonly LabelSpot[];
  size: number;
  ink?: string;
  /**
   * Draw even without a free spot (at the least-crowded spot in the frame, else the first one):
   * a `text` overlay the author placed. Point, arrow, polygon and curve labels are skipped.
   */
  keep?: boolean;
}

function labelRect(
  ctx: CanvasRenderingContext2D,
  c: ChartColors,
  label: Label,
  [x, y, align]: LabelSpot,
  size: number,
) {
  ctx.save();
  ctx.font = labelFont(c, Math.max(LABEL_SIZE, size - 0.5));
  const w =
    typeof label === 'string' ? ctx.measureText(label).width : measureMath(ctx, label, size);
  ctx.restore();
  const left = align === 'left' ? x : align === 'center' ? x - w / 2 : x - w;
  return { x: left, y: y - size * 0.7, w, h: size * 1.4 };
}

function paintLabel(
  ctx: CanvasRenderingContext2D,
  c: ChartColors,
  label: Label,
  [x, y, align]: LabelSpot,
  size: number,
  ink?: string,
) {
  if (typeof label === 'string') {
    ctx.save();
    ctx.font = labelFont(c, Math.max(LABEL_SIZE, size - 0.5));
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = c.halo;
    ctx.strokeText(label, x, y);
    ctx.fillStyle = ink ?? c.text2;
    ctx.fillText(label, x, y);
    ctx.restore();
    return;
  }
  drawMath(ctx, label, x, y, {
    size,
    align,
    baseline: 'middle',
    color: ink ?? c.text2,
    halo: c.halo,
  });
}

/** A label that overlaps others by more than this share of its own area is skipped. */
const MAX_SHARED = 0.3;

/**
 * The label pass: draw each pending label at its first spot that stays inside the frame and
 * clear of the labels already drawn this frame, and register its box. With no free spot, the
 * spot with the least overlap is used when the overlap is small, else the label is skipped.
 * Call it after the paths, so no path is drawn over a label.
 */
export function drawOverlayLabels(
  ctx: CanvasRenderingContext2D,
  v: Pick<OverlayView, 'width' | 'height' | 'colors'>,
  labels: readonly OverlayLabel[],
): void {
  const c = v.colors;
  const inFrame = (r: { x: number; y: number; w: number; h: number }) =>
    r.x >= 2 && r.y >= 2 && r.x + r.w <= v.width - 2 && r.y + r.h <= v.height - 2;
  for (const l of labels) {
    let best: { spot: LabelSpot; rect: ReturnType<typeof labelRect>; overlap: number } | null =
      null;
    for (const spot of l.spots) {
      const rect = labelRect(ctx, c, l.label, spot, l.size);
      if (!inFrame(rect)) continue;
      const overlap = labelOverlap(ctx, rect);
      if (!best || overlap < best.overlap) best = { spot, rect, overlap };
      if (overlap === 0) break;
    }
    if (!best || best.overlap > MAX_SHARED * best.rect.w * best.rect.h) {
      if (!l.keep || !l.spots.length) continue;
      best ??= {
        spot: l.spots[0],
        rect: labelRect(ctx, c, l.label, l.spots[0], l.size),
        overlap: 0,
      };
    }
    addLabelRect(ctx, best.rect);
    paintLabel(ctx, c, l.label, best.spot, l.size, l.ink);
  }
}

/** A point label's spots: its side first, then the four corners, then the other sides. */
function pointSpots(px: number, py: number, r: number, side: string): LabelSpot[] {
  const g = r + 6;
  const sides: Record<string, LabelSpot> = {
    right: [px + g, py, 'left'],
    left: [px - g, py, 'right'],
    above: [px, py - g - 4, 'center'],
    below: [px, py + g + 4, 'center'],
  };
  const corners: LabelSpot[] = [
    [px + g - 2, py - g, 'left'],
    [px - g + 2, py - g, 'right'],
    [px + g - 2, py + g, 'left'],
    [px - g + 2, py + g, 'right'],
  ];
  return [
    sides[side] ?? sides.right,
    ...corners,
    ...Object.entries(sides)
      .filter(([k]) => k !== side)
      .map(([, s]) => s),
  ];
}

/** Spots around an anchor: the anchor, then nudged up, down, left and right by `d` px. */
function nudgeSpots([x, y, align]: LabelSpot, d = 14): LabelSpot[] {
  return [
    [x, y, align],
    [x, y - d, align],
    [x, y + d, align],
    [x - d, y, align],
    [x + d, y, align],
  ];
}

function pathOf(ctx: CanvasRenderingContext2D, pts: readonly Pt[], v: OverlayView, close: boolean) {
  ctx.beginPath();
  pts.forEach((p, i) => {
    const [x, y] = v.toPx(p[0], p[1]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (close) ctx.closePath();
}

function arrowHead(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number]) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1) return;
  const ux = dx / len,
    uy = dy / len;
  const s = Math.min(9, len * 0.45);
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - ux * s - uy * s * 0.5, b[1] - uy * s + ux * s * 0.5);
  ctx.lineTo(b[0] - ux * s + uy * s * 0.5, b[1] - uy * s - ux * s * 0.5);
  ctx.closePath();
  ctx.fill();
}

/**
 * Draw overlays in order (shading first is the caller's choice: list regions before marks), then
 * their labels in one collision-tested pass. With `{ labels: 'defer' }` the labels are returned
 * instead, for `drawOverlayLabels` after the paths.
 */
export function drawOverlays2D(
  ctx: CanvasRenderingContext2D,
  v: OverlayView,
  overlays: readonly Overlay2D[],
  options: { labels?: 'now' | 'defer' } = {},
): OverlayLabel[] {
  const c = v.colors;
  const labels: OverlayLabel[] = [];
  const queue = (
    label: Label,
    spots: readonly LabelSpot[],
    size = 12,
    ink?: string,
    keep = false,
  ) => labels.push({ label, spots, size, ink, keep });
  for (const o of overlays) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    switch (o.kind) {
      case 'polygon':
      case 'polyline': {
        if (o.points.length < 2) break;
        const closed = o.kind === 'polygon';
        pathOf(ctx, o.points, v, closed);
        if (o.kind === 'polygon' && o.fill) {
          ctx.fillStyle = color(c, o.slot);
          ctx.globalAlpha = 0.12 * (o.alpha ?? 1);
          ctx.fill();
        }
        if (o.kind === 'polyline' || o.stroke !== false) {
          ctx.strokeStyle = c.halo;
          ctx.lineWidth = (o.width ?? 1.5) + 2.5;
          ctx.globalAlpha = 0.7;
          ctx.setLineDash([]);
          ctx.stroke();
          strokeStyle(ctx, o, c);
          ctx.stroke();
        }
        if (o.kind === 'polygon' && o.label) {
          const cx = o.points.reduce((s, p) => s + p[0], 0) / o.points.length;
          const cy = o.points.reduce((s, p) => s + p[1], 0) / o.points.length;
          const [px, py] = v.toPx(cx, cy);
          queue(o.label, nudgeSpots([px, py, 'center']));
        }
        break;
      }
      case 'segment':
      case 'arrow': {
        const a = v.toPx(o.from[0], o.from[1]);
        const b = v.toPx(o.to[0], o.to[1]);
        ctx.strokeStyle = c.halo;
        ctx.lineWidth = (o.width ?? 1.5) + 2.5;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        strokeStyle(ctx, o, c);
        ctx.stroke();
        if (o.kind === 'arrow') {
          ctx.setLineDash([]);
          ctx.fillStyle = color(c, o.slot);
          arrowHead(ctx, a, b);
          if (o.label) {
            const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
            // Label beside the shaft's midpoint, on its left-hand side.
            const nx = -(b[1] - a[1]) / len,
              ny = (b[0] - a[0]) / len;
            const mx = (a[0] + b[0]) / 2,
              my = (a[1] + b[1]) / 2;
            queue(o.label, [
              [mx + nx * 12, my + ny * 12, 'center'],
              [mx - nx * 12, my - ny * 12, 'center'],
              [mx + nx * 22, my + ny * 22, 'center'],
              [mx - nx * 22, my - ny * 22, 'center'],
              [b[0] + nx * 12, b[1] + ny * 12, 'center'],
            ]);
          }
        }
        break;
      }
      case 'disk':
      case 'ellipse': {
        const pts =
          o.kind === 'disk'
            ? ellipsePoints(
                o.center,
                [
                  [1, 0],
                  [0, 1],
                ],
                o.radius,
              )
            : ellipsePoints(o.center, o.matrix, o.radius ?? 1);
        if (!pts) break;
        pathOf(ctx, pts, v, true);
        if (o.fill) {
          ctx.fillStyle = color(c, o.slot);
          ctx.globalAlpha = 0.1 * (o.alpha ?? 1);
          ctx.fill();
        }
        strokeStyle(ctx, o, c);
        ctx.stroke();
        break;
      }
      case 'implicit': {
        const segs = curveSegments(o.g, o.cacheKey, v, o.level ?? 0);
        strokeCurve(ctx, segs, v, o, c);
        if (o.label && segs.length >= 4) {
          // Label at the curve point nearest the top-right of the view.
          let best = 0,
            score = -Infinity;
          for (let k = 0; k < segs.length; k += 4) {
            const [px, py] = v.toPx(segs[k], segs[k + 1]);
            const s = px - py;
            if (px < v.width - 60 && py > 16 && s > score) {
              score = s;
              best = k;
            }
          }
          const [px, py] = v.toPx(segs[best], segs[best + 1]);
          queue(o.label, nudgeSpots([px + 6, py - 8, 'left']), 12, color(c, o.slot));
        }
        break;
      }
      case 'constraints': {
        const gs = o.g;
        if (o.hatch !== false && gs.length) {
          const layer = shadedLayer(
            (x, y) => gs.some((g) => g(x, y) > 0),
            `${o.cacheKey}|infeasible`,
            v,
            true,
            true,
          );
          if (layer) ctx.drawImage(layer, 0, 0, v.width, v.height);
        }
        if (o.boundary !== false) {
          [...gs, ...(o.h ?? [])].forEach((g, i) =>
            strokeCurve(ctx, curveSegments(g, `${o.cacheKey}|${i}`, v), v, { width: 1.4 }, c),
          );
        }
        break;
      }
      case 'region': {
        const layer = shadedLayer(o.inside, o.cacheKey, v, o.hatch !== false, o.tint !== false);
        if (layer) ctx.drawImage(layer, 0, 0, v.width, v.height);
        break;
      }
      case 'point': {
        const [px, py] = v.toPx(o.at[0], o.at[1]);
        const ink = color(c, o.slot);
        const r = o.radius ?? 4;
        if (o.shape === 'cross') {
          for (const [w, s] of [
            [3.5, c.halo],
            [1.5, ink],
          ] as const) {
            ctx.strokeStyle = s;
            ctx.lineWidth = w;
            ctx.beginPath();
            ctx.moveTo(px - r - 1, py);
            ctx.lineTo(px + r + 1, py);
            ctx.moveTo(px, py - r - 1);
            ctx.lineTo(px, py + r + 1);
            ctx.stroke();
          }
        } else {
          ctx.fillStyle = c.halo;
          ctx.beginPath();
          ctx.arc(px, py, r + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          if (o.shape === 'ring') {
            ctx.strokeStyle = ink;
            ctx.lineWidth = 1.75;
            ctx.stroke();
          } else {
            ctx.fillStyle = ink;
            ctx.fill();
          }
        }
        if (o.label) queue(o.label, pointSpots(px, py, r, o.labelSide ?? 'right'));
        break;
      }
      case 'text': {
        const [px, py] = v.toPx(o.at[0], o.at[1]);
        queue(
          o.text,
          nudgeSpots([px, py, o.align ?? 'left']),
          Math.max(LABEL_SIZE, o.size ?? 12),
          o.slot === undefined ? undefined : color(c, o.slot),
          true,
        );
        break;
      }
    }
    ctx.restore();
  }
  if (options.labels === 'defer') return labels;
  drawOverlayLabels(ctx, v, labels);
  return [];
}

function strokeCurve(
  ctx: CanvasRenderingContext2D,
  segs: readonly number[],
  v: OverlayView,
  o: Stroke,
  c: ChartColors,
) {
  if (!segs.length) return;
  ctx.beginPath();
  for (let k = 0; k < segs.length; k += 4) {
    const a = v.toPx(segs[k], segs[k + 1]);
    const b = v.toPx(segs[k + 2], segs[k + 3]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.strokeStyle = c.halo;
  ctx.lineWidth = (o.width ?? 1.5) + 2;
  ctx.globalAlpha = 0.6;
  ctx.setLineDash([]);
  ctx.stroke();
  strokeStyle(ctx, o, c);
  ctx.stroke();
}
