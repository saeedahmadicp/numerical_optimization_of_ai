/**
 * Canvas helpers shared by the lab previews: views (data → CSS px), a cached contour field, and
 * the chart marks of brand.md §9 (halo under every mark, 𝐱₀ hollow ring, 𝐱⋆ cross, solid end dot).
 */
import { CONTOUR_MAPS, type ChartColors } from '../../../ui/colors';
import { computeField, isoSegments } from '../../../viz/contourField';
import type { CanvasSize } from '../../../viz/useCanvas';
import { easeInOut } from '../../../play/timeline';

export type Box = [[number, number], [number, number]];
export type Pt = readonly [number, number];

export interface View {
  toPx: (x: number, y: number) => [number, number];
  /** Inverse of `toPx` (CSS px → data). */
  toData: (px: number, py: number) => [number, number];
  /** The data box the whole canvas shows. */
  box: Box;
  width: number;
  height: number;
}

/**
 * Equal scale on both axes; `box` fits inside the canvas with `pad` px to spare and the canvas
 * shows whatever lies around it (so a field can fill the whole thumbnail).
 */
export function fitEqual(box: Box, w: number, h: number, pad: number): View {
  const [[x0, x1], [y0, y1]] = box;
  const s = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
  const cx = (x0 + x1) / 2,
    cy = (y0 + y1) / 2;
  return {
    toPx: (x, y) => [w / 2 + (x - cx) * s, h / 2 - (y - cy) * s],
    toData: (px, py) => [cx + (px - w / 2) / s, cy - (py - h / 2) / s],
    box: [
      [cx - w / 2 / s, cx + w / 2 / s],
      [cy - h / 2 / s, cy + h / 2 / s],
    ],
    width: w,
    height: h,
  };
}

/** Independent scales: `xr` and `yr` map onto the rectangle inset by `pad` = [left, top, right, bottom]. */
export function fitAxes(
  xr: [number, number],
  yr: [number, number],
  w: number,
  h: number,
  pad: [number, number, number, number],
): View {
  const [l, t, r, b] = pad;
  const sx = (w - l - r) / (xr[1] - xr[0]);
  const sy = (h - t - b) / (yr[1] - yr[0]);
  return {
    toPx: (x, y) => [l + (x - xr[0]) * sx, h - b - (y - yr[0]) * sy],
    toData: (px, py) => [xr[0] + (px - l) / sx, yr[0] + (h - b - py) / sy],
    box: [
      [xr[0] - l / sx, xr[1] + r / sx],
      [yr[0] - b / sy, yr[1] + t / sy],
    ],
    width: w,
    height: h,
  };
}

// ── Contour field ───────────────────────────────────────────────────────────────────────────

const FIELD_CACHE = new Map<string, HTMLCanvasElement>();

export interface FieldOptions {
  levels?: number;
  fMin?: number;
  scale?: 'auto' | 'linear' | 'log';
  /** Draw the iso-lines (default true). */
  lines?: boolean;
}

/**
 * Filled level bands + iso-lines of f over the view (the lab's `CONTOUR_MAPS`), rasterized once
 * per (key, size, theme) and cached.
 */
export function drawField(
  ctx: CanvasRenderingContext2D,
  key: string,
  f: (x: number, y: number) => number,
  view: View,
  size: CanvasSize,
  c: ChartColors,
  o: FieldOptions = {},
): void {
  const W = Math.max(1, Math.round(view.width * size.dpr));
  const H = Math.max(1, Math.round(view.height * size.dpr));
  const [[x0, x1], [y0, y1]] = view.box;
  const id = `${key}|${W}x${H}|${c.mode}|${x0.toFixed(4)},${y0.toFixed(4)},${x1.toFixed(4)}`;
  let layer = FIELD_CACHE.get(id);
  if (!layer) {
    const nx = 140;
    const ny = Math.max(24, Math.round((nx * H) / W));
    const values = new Float64Array(nx * ny);
    for (let j = 0; j < ny; j++) {
      const y = y0 + ((y1 - y0) * j) / (ny - 1);
      for (let i = 0; i < nx; i++) values[j * nx + i] = f(x0 + ((x1 - x0) * i) / (nx - 1), y);
    }
    const res = computeField({
      id: 0,
      values,
      nx,
      ny,
      width: W,
      height: H,
      levels: o.levels ?? 12,
      scale: o.scale ?? 'auto',
      fMin: o.fMin ?? null,
      lut: CONTOUR_MAPS[c.mode].lut,
    });
    layer = document.createElement('canvas');
    layer.width = W;
    layer.height = H;
    const lc = layer.getContext('2d');
    if (lc) {
      lc.putImageData(new ImageData(new Uint8ClampedArray(res.pixels), W, H), 0, 0);
      if (o.lines !== false) {
        lc.strokeStyle = c.iso;
        lc.lineWidth = Math.max(1, size.dpr * 0.75);
        lc.beginPath();
        const s = res.segments;
        const gx = W / (nx - 1),
          gy = H / (ny - 1);
        for (let k = 0; k < s.length; k += 4) {
          lc.moveTo(s[k] * gx, H - s[k + 1] * gy);
          lc.lineTo(s[k + 2] * gx, H - s[k + 3] * gy);
        }
        lc.stroke();
      }
    }
    if (FIELD_CACHE.size > 48) FIELD_CACHE.delete(FIELD_CACHE.keys().next().value as string);
    FIELD_CACHE.set(id, layer);
  }
  ctx.drawImage(layer, 0, 0, view.width, view.height);
}

/** The zero set g(x, y) = 0 over the view, as line segments in CSS px (marching squares). */
export function zeroSet(g: (x: number, y: number) => number, view: View, n = 160): number[] {
  const [[x0, x1], [y0, y1]] = view.box;
  const ny = Math.max(24, Math.round((n * view.height) / view.width));
  const vals = new Float64Array(n * ny);
  for (let j = 0; j < ny; j++) {
    const y = y0 + ((y1 - y0) * j) / (ny - 1);
    for (let i = 0; i < n; i++) vals[j * n + i] = g(x0 + ((x1 - x0) * i) / (n - 1), y);
  }
  const segs: number[] = [];
  isoSegments(vals, n, ny, 0, segs);
  const sx = view.width / (n - 1),
    sy = view.height / (ny - 1);
  for (let k = 0; k < segs.length; k += 2) {
    segs[k] *= sx;
    segs[k + 1] = view.height - segs[k + 1] * sy;
  }
  return segs;
}

export function strokeSegments(
  ctx: CanvasRenderingContext2D,
  segs: readonly number[],
  color: string,
  width: number,
  dash: number[] = [],
): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  for (let k = 0; k < segs.length; k += 4) {
    ctx.moveTo(segs[k], segs[k + 1]);
    ctx.lineTo(segs[k + 2], segs[k + 3]);
  }
  ctx.stroke();
  ctx.restore();
}

// ── Marks ───────────────────────────────────────────────────────────────────────────────────

/** A polyline in CSS px with a surface-colored halo underneath. */
export function line(
  ctx: CanvasRenderingContext2D,
  pts: readonly Pt[],
  color: string,
  width: number,
  o: { halo?: string; dash?: number[]; alpha?: number } = {},
): void {
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  };
  if (o.halo) {
    ctx.strokeStyle = o.halo;
    ctx.lineWidth = width + 3;
    ctx.globalAlpha = 0.8 * (o.alpha ?? 1);
    path();
    ctx.stroke();
  }
  ctx.globalAlpha = o.alpha ?? 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(o.dash ?? []);
  path();
  ctx.stroke();
  ctx.restore();
}

export function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  color: string,
  halo?: string,
  alpha = 1,
): void {
  ctx.save();
  ctx.globalAlpha = alpha;
  if (halo) {
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, r + 1.75, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export function ring(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  color: string,
  halo?: string,
  width = 1.5,
): void {
  ctx.save();
  if (halo) {
    ctx.strokeStyle = halo;
    ctx.lineWidth = width + 3;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/** 𝐱⋆: a + cross in the text color with a halo. */
export function cross(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  c: ChartColors,
): void {
  const seg = (w: number, color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x - r, y);
    ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r);
    ctx.lineTo(x, y + r);
    ctx.stroke();
  };
  ctx.save();
  ctx.lineCap = 'round';
  seg(4.5, c.halo);
  seg(1.6, c.text);
  ctx.restore();
}

/** f sampled on [a, b] and mapped to px (non-finite samples break the line). */
export function curve(f: (x: number) => number, a: number, b: number, view: View, n = 200): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    const y = f(x);
    if (Number.isFinite(y)) out.push(view.toPx(x, y));
  }
  return out;
}

/** Fill with a color at an alpha (any CSS color). */
export function fillAlpha(
  ctx: CanvasRenderingContext2D,
  color: string,
  alpha: number,
  fill: () => void,
) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  fill();
  ctx.restore();
}

// ── Clocks ──────────────────────────────────────────────────────────────────────────────────

/** Step index at progress u, linear in k (0 → K). */
export const linClock = (u: number, K: number) => Math.max(0, Math.min(1, u)) * K;

/** Step index at progress u, linear in log(k + 1): fast methods and slow ones share one clock. */
export const logClock = (u: number, K: number) => (K + 1) ** Math.max(0, Math.min(1, u)) - 1;

/** Integer step and an eased fraction toward the next one. */
export function stepAt(k: number, last: number): { i: number; f: number } {
  const kk = Math.max(0, Math.min(last, k));
  const i = Math.floor(kk);
  return { i, f: i >= last ? 0 : easeInOut(kk - i) };
}

/** Width scale for hero vs card. */
export const S = (hero: boolean) => (hero ? 1.2 : 1);

/** The bounding box of point sets, grown by `margin` (fraction of the larger side) on every side. */
export function boundsOf(sets: readonly (readonly Pt[])[], margin = 0.08): Box {
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (const s of sets)
    for (const [x, y] of s) {
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
  const m = margin * Math.max(x1 - x0, y1 - y0, 1e-9);
  return [
    [x0 - m, x1 + m],
    [y0 - m, y1 + m],
  ];
}

/** A diagonal hatch over a light tint (brand.md: infeasible sets are hatched + tinted). */
export function hatchPattern(
  ctx: CanvasRenderingContext2D,
  color: string,
  dpr: number,
): CanvasPattern | null {
  const n = Math.round(7 * dpr);
  const tile = document.createElement('canvas');
  tile.width = n;
  tile.height = n;
  const t = tile.getContext('2d');
  if (!t) return null;
  t.fillStyle = color;
  t.globalAlpha = 0.1;
  t.fillRect(0, 0, n, n);
  t.globalAlpha = 0.55;
  t.strokeStyle = color;
  t.lineWidth = Math.max(1, 0.8 * dpr);
  t.beginPath();
  for (const o of [-n, 0, n]) {
    t.moveTo(o, n);
    t.lineTo(o + n, 0);
  }
  t.stroke();
  return ctx.createPattern(tile, 'repeat');
}
