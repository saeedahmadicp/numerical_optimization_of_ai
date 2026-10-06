/**
 * PathLayer — animated iterate paths on any 2-D view.
 *
 * Each path is drawn up to its own local time (`min(t, n − 1)`), with an eased partial segment
 * toward the next iterate, a trail that fades with age, a surface-colored halo so the line reads
 * over any fill, and a current-point marker with a soft glow.
 */
import { lerp, segmentAt } from '../play/timeline';
import { drawMath, iterateRuns, measureMath } from './mathText';
import { addLabelRect, labelOverlap } from './labelRects';
import { labelFont } from './axes';

/** The UI face for the second line of an off-view label (no ChartColors in scope here). */
const SANS = { fontSans: "'Inter Variable', Inter, system-ui, sans-serif" };

export interface PathSpec {
  /** Iterates in data coordinates. */
  points: readonly (readonly [number, number])[];
  color: string;
  /** Method name: used in off-view labels ("Newton, below the view"). */
  label?: string;
  /** Draw small dots at each iterate (default: when the path has ≤ 80 points). */
  dots?: boolean;
  /** Draw arrowheads on the most recent segments. */
  arrows?: boolean;
  /** Dim the whole path (e.g. a method that is not focused). */
  muted?: boolean;
  /** Dash pattern for the line (CSS px), e.g. `[7, 6]` for a path drawn over another one. */
  dash?: number[];
  /** Line width (CSS px, default the layer's `lineWidth`). Paint the slowest run last and thinnest. */
  width?: number;
  /** Progress diamonds at k = 10, 10², 10³, … (default: when the path has > 30 points). */
  milestones?: boolean;
  /**
   * How the run ended, drawn once the head reaches the last iterate: a solid dot when it
   * converged, a hollow ring when it stopped (budget, breakdown). Omit for no end marker.
   */
  end?: 'converged' | 'stopped';
  /** Hollow ring at 𝐱₀ in the path color (default true; off when the plot draws a shared 𝐱₀). */
  start?: boolean;
  /** No glowing head (static figures, thumbnails). */
  quiet?: boolean;
}

/** Pixel rectangle of the visible plot (enables the off-view treatment). */
export interface PxBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface PathLayerOptions {
  /** Global playhead (continuous step index). */
  t: number;
  /** data → CSS pixel. */
  toPx: (x: number, y: number) => [number, number];
  halo: string;
  /** Interpolate between steps with easing (false under reduced motion). */
  ease?: boolean;
  /** Number of recent segments drawn at full strength. */
  trail?: number;
  lineWidth?: number;
  /**
   * The visible plot in CSS px. Steps that leave it are dashed, with an outward chevron where
   * they leave and an inward chevron where the next step comes back (brand.md §9).
   */
  bounds?: PxBounds;
  /**
   * Label the iterate a step jumps to when it leaves the view: `𝐱₂ = (0.76, −3.18)` with the
   * method name and the side ("Newton, below the view"). Needs `bounds`.
   */
  offViewLabels?: boolean;
  /** Label text color (`colors.text2`). */
  textColor?: string;
}

function withAlpha(color: string, a: number): string {
  // Accepts #rrggbb; anything else is returned with globalAlpha handling by the caller.
  if (/^#[0-9a-f]{6}$/i.test(color)) {
    const n = Math.round(Math.max(0, Math.min(1, a)) * 255);
    return color + n.toString(16).padStart(2, '0');
  }
  return color;
}

/** Current interpolated position of a path at time t. */
export function pathPosition(
  points: PathSpec['points'],
  t: number,
  ease = true,
): [number, number] | null {
  if (points.length === 0) return null;
  const { i, u } = segmentAt(t, points.length, ease);
  const p = points[i];
  const q = points[Math.min(i + 1, points.length - 1)];
  return [lerp(p[0], q[0], u), lerp(p[1], q[1], u)];
}

/** k = 10, 100, 1000, … below `n`. */
export function milestoneIndices(n: number): number[] {
  const out: number[] = [];
  for (let k = 10; k < n; k *= 10) out.push(k);
  return out;
}

/**
 * Clip segment a→b to the rectangle (Liang–Barsky). Returns the parameters [u0, u1] ⊂ [0, 1] of
 * the inside part, or null when the segment misses the rectangle.
 */
export function clipSegment(
  a: readonly [number, number],
  b: readonly [number, number],
  r: PxBounds,
): [number, number] | null {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  let u0 = 0,
    u1 = 1;
  const edges: [number, number][] = [
    [-dx, a[0] - r.left],
    [dx, r.right - a[0]],
    [-dy, a[1] - r.top],
    [dy, r.bottom - a[1]],
  ];
  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return null;
      continue;
    }
    const u = q / p;
    if (p < 0) {
      if (u > u1) return null;
      if (u > u0) u0 = u;
    } else {
      if (u < u0) return null;
      if (u < u1) u1 = u;
    }
  }
  return [u0, u1];
}

const inside = (p: readonly [number, number], r: PxBounds) =>
  p[0] >= r.left && p[0] <= r.right && p[1] >= r.top && p[1] <= r.bottom;

/** Where a point lies relative to the view, in words. */
export function sideOf(p: readonly [number, number], r: PxBounds): string {
  if (p[1] > r.bottom) return 'below the view';
  if (p[1] < r.top) return 'above the view';
  if (p[0] < r.left) return 'left of the view';
  return 'right of the view';
}

export function drawPathLayer(
  ctx: CanvasRenderingContext2D,
  paths: readonly PathSpec[],
  o: PathLayerOptions,
): void {
  const { t, toPx, halo, ease = true, trail = 12, lineWidth = 2, bounds } = o;
  // Far-away iterates (diverged runs) are clipped to a generous box so canvas coordinates stay sane.
  const far: PxBounds | undefined = bounds && {
    left: bounds.left - 4000,
    top: bounds.top - 4000,
    right: bounds.right + 4000,
    bottom: bounds.bottom + 4000,
  };
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const geo = paths.map((path) => visibleSegments(path, t, ease, toPx, bounds, far));

  // Pass 1: halos for every path (so a later path's halo never cuts an earlier line).
  paths.forEach((path, pi) => {
    const segs = geo[pi].segs;
    if (!segs.length) return;
    ctx.strokeStyle = halo;
    ctx.lineWidth = (path.width ?? lineWidth) + 2.5;
    ctx.globalAlpha = path.muted ? 0.3 : 0.7;
    ctx.beginPath();
    for (const s of segs) {
      if (s.out) continue;
      ctx.moveTo(s.a[0], s.a[1]);
      ctx.lineTo(s.b[0], s.b[1]);
    }
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  // Pass 2: the colored lines, older segments fading; off-view steps dashed.
  paths.forEach((path, pi) => {
    const { segs, head, reached } = geo[pi];
    const baseAlpha = path.muted ? 0.35 : 1;
    const width = path.width ?? lineWidth;
    const n = segs.length;
    ctx.lineWidth = width;
    // Batch consecutive segments with the same style into one stroke.
    let k = 0;
    while (k < n) {
      const alphaOf = (j: number) => {
        const a = n - 1 - j;
        return a > trail ? 0.78 : 0.78 + 0.22 * (1 - a / Math.max(1, trail));
      };
      const style = (j: number) => `${segs[j].out ? 'd' : 's'}${n - 1 - j > trail ? 'old' : j}`;
      const key = style(k);
      ctx.strokeStyle = withAlpha(path.color, alphaOf(k) * baseAlpha);
      ctx.setLineDash(segs[k].out ? [4, 4] : (path.dash ?? []));
      ctx.beginPath();
      ctx.moveTo(segs[k].a[0], segs[k].a[1]);
      let j = k;
      while (j < n && style(j) === key) {
        if (j > k && (segs[j].a[0] !== segs[j - 1].b[0] || segs[j].a[1] !== segs[j - 1].b[1]))
          ctx.moveTo(segs[j].a[0], segs[j].a[1]);
        ctx.lineTo(segs[j].b[0], segs[j].b[1]);
        j++;
      }
      ctx.stroke();
      k = j;
    }
    ctx.setLineDash([]);

    // Iterate dots (only the reached iterates, not the interpolated head).
    const showDots = path.dots ?? path.points.length <= 80;
    if (showDots) {
      for (let i = 1; i <= reached; i++) {
        const [x, y] = toPx(path.points[i][0], path.points[i][1]);
        if (bounds && !inside([x, y], bounds)) continue;
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(x, y, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = withAlpha(path.color, (i >= reached - trail ? 1 : 0.55) * baseAlpha);
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Progress diamonds at k = 10, 100, 1000, … (second encoding for long runs).
    if (path.milestones ?? path.points.length > 30) {
      for (const i of milestoneIndices(path.points.length)) {
        if (i > reached) break;
        const [x, y] = toPx(path.points[i][0], path.points[i][1]);
        if (bounds && !inside([x, y], bounds)) continue;
        diamond(ctx, x, y, 4.2, withAlpha(path.color, baseAlpha), halo);
      }
    }

    if (path.arrows) {
      for (let i = Math.max(0, n - 3); i < n; i++)
        if (!segs[i].out) drawArrow(ctx, segs[i].a, segs[i].b, path.color);
    }

    // Chevrons where a step leaves / re-enters the view.
    for (const s of segs) {
      if (s.exit) chevron(ctx, s.exit.p, s.exit.dir, withAlpha(path.color, baseAlpha), halo);
      if (s.entry) chevron(ctx, s.entry.p, s.entry.dir, withAlpha(path.color, baseAlpha), halo);
    }

    // Start marker: hollow ring.
    if (path.start !== false && path.points.length) {
      const [sx, sy] = toPx(path.points[0][0], path.points[0][1]);
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = halo;
      ctx.beginPath();
      ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = withAlpha(path.color, baseAlpha);
      ctx.stroke();
    }

    if (!head) return;
    const [hx, hy] = head;
    const atEnd = reached >= path.points.length - 1;
    if (atEnd && path.end) {
      // The run's end: solid when it converged, hollow when it stopped.
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(hx, hy, 5.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(hx, hy, 3.8, 0, Math.PI * 2);
      if (path.end === 'converged') {
        ctx.fillStyle = withAlpha(path.color, baseAlpha);
        ctx.fill();
      } else {
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = withAlpha(path.color, baseAlpha);
        ctx.stroke();
      }
      return;
    }
    if (bounds && !inside(head, bounds)) return;
    // Current point: glow + ringed dot.
    if (!path.muted && !path.quiet) {
      const glow = ctx.createRadialGradient(hx, hy, 0, hx, hy, 16);
      glow.addColorStop(0, withAlpha(path.color, 0.42));
      glow.addColorStop(1, withAlpha(path.color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(hx, hy, 16, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(hx, hy, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = withAlpha(path.color, baseAlpha);
    ctx.beginPath();
    ctx.arc(hx, hy, 4.75, 0, Math.PI * 2);
    ctx.fill();
  });

  // Labels for iterates that left the view (after all lines, so nothing covers them).
  if (bounds && o.offViewLabels) {
    paths.forEach((path, pi) => {
      for (const s of geo[pi].segs) {
        if (!s.exit || s.toIndex === undefined || s.toIndex > geo[pi].reached) continue;
        const q = path.points[s.toIndex];
        const side = sideOf(toPx(q[0], q[1]), bounds);
        drawOffViewLabel(
          ctx,
          s.exit.p,
          bounds,
          s.toIndex,
          q,
          path.label,
          side,
          o.textColor ?? '#555',
          halo,
        );
      }
    });
  }
  ctx.restore();
}

type Pt = readonly [number, number];

/** `n` points evenly spaced along the arc length of a polyline (non-finite points dropped). */
function resample(path: readonly Pt[], n: number): Pt[] {
  const a = path.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (a.length < 2) return a.slice();
  const cum = [0];
  for (let i = 1; i < a.length; i++)
    cum.push(cum[i - 1] + Math.hypot(a[i][0] - a[i - 1][0], a[i][1] - a[i - 1][1]));
  const total = cum[cum.length - 1];
  const out: Pt[] = [];
  let j = 0;
  for (let s = 0; s < n; s++) {
    const d = (total * s) / (n - 1);
    while (j < a.length - 2 && cum[j + 1] < d) j++;
    const len = cum[j + 1] - cum[j];
    const u = len > 0 ? (d - cum[j]) / len : 0;
    out.push([a[j][0] + u * (a[j + 1][0] - a[j][0]), a[j][1] + u * (a[j + 1][1] - a[j][1])]);
  }
  return out;
}

function segDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const L = dx * dx + dy * dy;
  const u = L > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L)) : 0;
  return Math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy));
}

/** Fraction of `a`'s arc length (sampled) that lies within `tol` of the polyline `b`. */
export function pathCoverage(
  a: readonly Pt[],
  b: readonly Pt[],
  tol: number,
  samples = 48,
): number {
  const pts = resample(a, samples);
  if (pts.length === 0 || b.length < 2) return 0;
  let ok = 0;
  for (const p of pts) {
    let best = Infinity;
    for (let k = 0; k + 1 < b.length && best > tol; k++) {
      if (!Number.isFinite(b[k][0]) || !Number.isFinite(b[k + 1][0])) continue;
      best = Math.min(best, segDist(p, b[k], b[k + 1]));
    }
    if (best <= tol) ok++;
  }
  return ok / pts.length;
}

/**
 * Hidden paths: pairs `[i, j]` where path i runs along path j (at least `share` of i's arc
 * length lies within `tol` data units of j), so drawing j over i would hide i. Draw path i on
 * top with a dash so both stay visible, and tell the viewer. Each i is reported once.
 */
export function overlappingPaths(
  paths: readonly (readonly Pt[])[],
  tol: number,
  share = 0.9,
): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < paths.length; i++) {
    if (paths[i].length < 2) continue;
    for (let j = 0; j < paths.length; j++) {
      if (i === j || paths[j].length < 2) continue;
      // Two identical paths: report only the later one as hidden.
      if (j > i && pathCoverage(paths[j], paths[i], tol) >= share) continue;
      if (pathCoverage(paths[i], paths[j], tol) >= share) {
        out.push([i, j]);
        break;
      }
    }
  }
  return out;
}

interface Seg {
  a: [number, number];
  b: [number, number];
  /** The step touches a point outside the view (drawn dashed). */
  out: boolean;
  /** Index of the iterate this step goes to. */
  toIndex?: number;
  exit?: { p: [number, number]; dir: [number, number] };
  entry?: { p: [number, number]; dir: [number, number] };
}

/** Pixel segments up to the playhead, with off-view steps flagged and clipped. */
function visibleSegments(
  path: PathSpec,
  t: number,
  ease: boolean,
  toPx: PathLayerOptions['toPx'],
  bounds?: PxBounds,
  far?: PxBounds,
): { segs: Seg[]; head: [number, number] | null; reached: number } {
  const n = path.points.length;
  if (n === 0) return { segs: [], head: null, reached: -1 };
  const { i, u } = segmentAt(t, n, ease);
  const px: [number, number][] = [];
  for (let k = 0; k <= i; k++) px.push(toPx(path.points[k][0], path.points[k][1]));
  let partial = false;
  if (u > 0 && i + 1 < n) {
    const p = path.points[i],
      q = path.points[i + 1];
    px.push(toPx(lerp(p[0], q[0], u), lerp(p[1], q[1], u)));
    partial = true;
  }
  const finite = (p: [number, number]) => Number.isFinite(p[0]) && Number.isFinite(p[1]);
  const segs: Seg[] = [];
  for (let k = 0; k + 1 < px.length; k++) {
    let a = px[k],
      b = px[k + 1];
    if (!finite(a) || !finite(b)) continue;
    const toIndex = partial && k + 1 === px.length - 1 ? undefined : k + 1;
    if (!bounds || (inside(a, bounds) && inside(b, bounds))) {
      segs.push({ a, b, out: false, toIndex });
      continue;
    }
    // Keep the drawable part inside a generous box (huge coordinates break canvas paths).
    if (far) {
      const c = clipSegment(a, b, far);
      if (!c) continue;
      const a0 = a;
      a = [lerp(a0[0], b[0], c[0]), lerp(a0[1], b[1], c[0])];
      b = [lerp(a0[0], b[0], c[1]), lerp(a0[1], b[1], c[1])];
    }
    const seg: Seg = { a, b, out: true, toIndex };
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const dir: [number, number] = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
    const c = clipSegment(a, b, bounds);
    if (c) {
      const pAt = (w: number): [number, number] => [lerp(a[0], b[0], w), lerp(a[1], b[1], w)];
      if (inside(a, bounds) && !inside(b, bounds)) seg.exit = { p: pAt(c[1]), dir };
      if (!inside(a, bounds) && inside(b, bounds) && toIndex !== undefined)
        seg.entry = { p: pAt(c[0]), dir };
    }
    segs.push(seg);
  }
  const last = px[px.length - 1];
  const reached = Math.min(n - 1, Math.floor(Math.min(t, n - 1) + 1e-9));
  return { segs, head: last && finite(last) ? last : null, reached };
}

function diamond(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  halo: string,
) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = halo;
  ctx.stroke();
  ctx.fillStyle = fill;
  ctx.fill();
}

function chevron(
  ctx: CanvasRenderingContext2D,
  p: [number, number],
  dir: [number, number],
  color: string,
  halo: string,
) {
  const s = 6;
  const [ux, uy] = dir;
  const tip: [number, number] = [p[0] + ux * s * 0.5, p[1] + uy * s * 0.5];
  const l: [number, number] = [tip[0] - ux * s - uy * s * 0.75, tip[1] - uy * s + ux * s * 0.75];
  const r: [number, number] = [tip[0] - ux * s + uy * s * 0.75, tip[1] - uy * s - ux * s * 0.75];
  ctx.save();
  ctx.setLineDash([]);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const [w, c] of [
    [4.5, halo],
    [1.8, color],
  ] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(l[0], l[1]);
    ctx.lineTo(tip[0], tip[1]);
    ctx.lineTo(r[0], r[1]);
    ctx.stroke();
  }
  ctx.restore();
}

const fmtCoord = (v: number) => {
  const a = Math.abs(v);
  const s =
    a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(Number(v.toPrecision(3)));
  return s.replace(/^-/, '−');
};

function drawOffViewLabel(
  ctx: CanvasRenderingContext2D,
  exit: [number, number],
  r: PxBounds,
  k: number,
  q: readonly [number, number],
  name: string | undefined,
  side: string,
  color: string,
  halo: string,
) {
  const value = `(${fmtCoord(q[0])}, ${fmtCoord(q[1])})`;
  const line1 = iterateRuns('x', k, value);
  const w2 = (t: string) => {
    ctx.save();
    ctx.font = labelFont(SANS);
    const w = ctx.measureText(t).width;
    ctx.restore();
    return w;
  };
  const w1 = measureMath(ctx, line1, 12);
  // Where the label can go, beside the exit point and inside the view.
  const place = (w: number, right: boolean, row: number): [number, number] => {
    let x = right ? exit[0] + 8 : exit[0] - 8 - w;
    if (x + w > r.right - 6) x = exit[0] - 8 - w;
    x = Math.max(r.left + 6, x);
    let y = exit[1];
    const down = exit[1] >= r.bottom - 1 ? -1 : 1;
    // The second line's baseline (y + 13) stays above the inset x-tick row (≈ bottom − 24).
    y = Math.min(r.bottom - 40, Math.max(r.top + 14, y + (down < 0 ? -20 : 14) + down * 15 * row));
    // Keep clear of the view toolbar (top right, ≈ 100 × 40 px).
    if (y < r.top + 52 && x + w > r.right - 108) x = Math.max(r.left + 6, r.right - 108 - w);
    return [x, y];
  };
  // Candidates, nearest first: up to two lines in from the edge; with the method name, then
  // without it (shorter); on either side of the exit point. The first one clear of the labels
  // already drawn wins; otherwise the one that overlaps them least.
  const long = name ? `${name}, ${side}` : side;
  let best: { x: number; y: number; line2: string; w: number; cost: number } | null = null;
  search: for (let row = 0; row <= 2; row++)
    for (const line2 of name ? [long, side] : [side])
      for (const right of [true, false]) {
        const w = Math.max(w1, w2(line2));
        const [x, y] = place(w, right, row);
        const overlap = labelOverlap(ctx, { x, y: y - 11, w, h: 28 });
        const cost = overlap + row * 40;
        if (!best || cost < best.cost) best = { x, y, line2, w, cost };
        if (overlap === 0) {
          best = { x, y, line2, w, cost };
          break search;
        }
      }
  const { x, y, line2, w } = best!;
  addLabelRect(ctx, { x, y: y - 11, w, h: 28 });
  drawMath(ctx, line1, x, y, { size: 12, color, halo });
  ctx.save();
  ctx.font = labelFont(SANS);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.strokeStyle = halo;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeText(line2, x, y + 14);
  ctx.fillStyle = color;
  ctx.fillText(line2, x, y + 14);
  ctx.restore();
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  a: [number, number],
  b: [number, number],
  color: string,
) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 14) return;
  const ux = dx / len,
    uy = dy / len;
  const mx = a[0] + dx * 0.55,
    my = a[1] + dy * 0.55;
  const s = 5;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(mx + ux * s, my + uy * s);
  ctx.lineTo(mx - ux * s - uy * s * 0.8, my - uy * s + ux * s * 0.8);
  ctx.lineTo(mx - ux * s + uy * s * 0.8, my - uy * s - ux * s * 0.8);
  ctx.closePath();
  ctx.fill();
}
