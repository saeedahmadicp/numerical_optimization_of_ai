/**
 * Off-view marks for the step geometry (brand §2.4: a step that leaves the view gets a chevron
 * and a label at the exit). The Gauss–Newton point of a far start, its failed Armijo trials, a
 * long LM trial or the μ = 0 end of 𝐡(μ) can all lie outside the plane: each gets an outward
 * chevron where the drawn line leaves the frame and a label with its coordinates — inside the
 * frame, above the x-tick row and right of the y-tick column.
 *
 * Pure canvas code (no React); the exit geometry is exported for the unit tests.
 */
import { sci } from '../../core/format';
import type { ChartColors } from '../../ui/colors';
import {
  drawMath,
  measureMath,
  mathBold as b,
  mathMain as m,
  mathSub as sub,
  type MathRun,
} from '../../viz';
import type { Overlay2D } from '../../viz';
import { labelFont } from '../../viz/axes';
import type { StepMark } from './models';

type Px = [number, number];

export interface PxRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const inside = (p: Px, r: PxRect) =>
  p[0] >= r.left && p[0] <= r.right && p[1] >= r.top && p[1] <= r.bottom;

/**
 * Where a polyline that starts inside `r` first leaves it: the crossing point and the unit
 * direction of the crossing segment. Null when the polyline never leaves (or starts outside).
 */
export function exitPoint(path: readonly Px[], r: PxRect): { p: Px; dir: Px } | null {
  const pts = path.filter((q) => Number.isFinite(q[0]) && Number.isFinite(q[1]));
  if (pts.length < 2 || !inside(pts[0], r)) return null;
  for (let i = 1; i < pts.length; i++) {
    if (inside(pts[i], r)) continue;
    const a = pts[i - 1],
      c = pts[i];
    const dx = c[0] - a[0],
      dy = c[1] - a[1];
    // Largest u ∈ [0, 1] with a + u(c − a) still inside (a is inside).
    let u = 1;
    if (dx > 0) u = Math.min(u, (r.right - a[0]) / dx);
    if (dx < 0) u = Math.min(u, (r.left - a[0]) / dx);
    if (dy > 0) u = Math.min(u, (r.bottom - a[1]) / dy);
    if (dy < 0) u = Math.min(u, (r.top - a[1]) / dy);
    const len = Math.hypot(dx, dy) || 1;
    return { p: [a[0] + u * dx, a[1] + u * dy], dir: [dx / len, dy / len] };
  }
  return null;
}

/** Where a point outside `r` lies, in words ("below the view", "above and left of the view"). */
export function sideWords(p: Px, r: PxRect): string {
  const vert = p[1] > r.bottom ? 'below' : p[1] < r.top ? 'above' : '';
  const hor = p[0] < r.left ? 'left of' : p[0] > r.right ? 'right of' : '';
  if (vert && hor) return `${vert} and ${hor.replace(' of', '')} of the view`;
  if (vert) return `${vert} the view`;
  return `${hor} the view`;
}

/** 3 significant digits, U+2212 minus, `1.2×10⁻⁴` (Unicode superscripts) outside [10⁻³, 10⁵). */
export function fmtCoord(x: number): string {
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return sci(x, 2);
  return String(Number(x.toPrecision(3))).replace(/-/g, '−');
}

function chevron(ctx: CanvasRenderingContext2D, p: Px, dir: Px, color: string, halo: string) {
  const s = 6;
  const [ux, uy] = dir;
  const tip: Px = [p[0] - ux * 3, p[1] - uy * 3];
  const l: Px = [tip[0] - ux * s - uy * s * 0.75, tip[1] - uy * s + ux * s * 0.75];
  const r: Px = [tip[0] - ux * s + uy * s * 0.75, tip[1] - uy * s - ux * s * 0.75];
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

function text(
  ctx: CanvasRenderingContext2D,
  s: string,
  x: number,
  y: number,
  color: string,
  halo: string,
  font: string,
) {
  ctx.save();
  ctx.font = font;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.strokeStyle = halo;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeText(s, x, y);
  ctx.fillStyle = color;
  ctx.fillText(s, x, y);
  ctx.restore();
}

export interface OffViewOptions {
  toPx: (x: number, y: number) => Px;
  width: number;
  height: number;
  colors: ChartColors;
}

/** A method path, for the labels of iterates that left the view. */
export interface OffViewPath {
  points: readonly (readonly [number, number])[];
  name: string;
}

/** The step under the playhead (labels read 𝐱ₖ + 𝐩ₖ, 𝐱ₖ + 𝐡ₖ). */
export interface OffViewStep {
  marks: readonly StepMark[];
  slot: number;
  k: number;
}

/** A label box in CSS px (top-left corner, size). */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
const overlaps = (a: Box, c: Box) =>
  a.x < c.x + c.w && c.x < a.x + a.w && a.y < c.y + c.h && c.y < a.y + a.h;

/**
 * Off-view labels of the parameter plane, drawn by the lab (Contour2D's own exit labels are
 * turned off: they sit on the x-tick row). Two kinds, sharing one set of label boxes so they
 * never overlap:
 *   - iterates already reached that left the view: "𝐱ₖ = (…)" / "Gauss–Newton, below the view";
 *     Contour2D still draws their chevrons;
 *   - the marks of the step under the playhead (GN point and failed trials, LM trial, μ = 0):
 *     a chevron at the exit and "𝐱ₖ + 𝐩ₖ = (…)" / "Gauss–Newton point, below the view · …".
 * Labels stay inside the frame, above the x-tick row and right of the y-tick column. Returns
 * the number of labels drawn.
 */
export function drawOffView(
  ctx: CanvasRenderingContext2D,
  paths: readonly OffViewPath[],
  t: number,
  step: OffViewStep | null,
  o: OffViewOptions,
  /** Boxes already taken by other labels (𝐱⋆, the step's in-view labels). */
  reserved: readonly Box[] = [],
): number {
  const frame: PxRect = { left: 0, top: 0, right: o.width, bottom: o.height };
  // Clear of the y-tick column (left), the x-tick row (bottom) and the zoom bar (bottom right).
  const safe: PxRect = { left: 48, top: 14, right: o.width - 52, bottom: o.height - 26 };
  const ink = o.colors.text2;
  const halo = o.colors.halo;
  const boxes: Box[] = [...reserved];
  const font = labelFont(o.colors);
  /** Line pitch of the label text (12 px type, labelFont). */
  const LINE = 15;
  let drawn = 0;

  const place = (exit: Px, dir: Px, line1: MathRun[], line2: string) => {
    ctx.save();
    ctx.font = font;
    const room = safe.right - safe.left;
    // A text line wider than the safe width breaks at its " · " clauses.
    const lines = ctx.measureText(line2).width <= room ? [line2] : line2.split(' · ');
    const w2 = Math.max(...lines.map((l) => ctx.measureText(l).width));
    ctx.restore();
    const w = Math.max(measureMath(ctx, line1, 12), w2);
    const h = 15 + LINE * lines.length;
    // Beside the exit, pulled inside the safe rectangle; moved by a line pair on a collision.
    let x = exit[0] + 10;
    if (x + w > safe.right) x = exit[0] - 10 - w;
    x = Math.max(safe.left, Math.min(x, safe.right - w));
    const y0 = exit[1] + (dir[1] > 0.5 ? -11 - LINE * lines.length : 18);
    const away = dir[1] > 0.5 ? -1 : 1;
    for (const shift of [0, 30, 60, -30, 90]) {
      const y = Math.max(safe.top + 12, Math.min(y0 + away * shift, safe.bottom - h + 15));
      const box: Box = { x: x - 2, y: y - 12, w: w + 4, h };
      if (boxes.some((c) => overlaps(box, c))) continue;
      boxes.push(box);
      drawMath(ctx, line1, x, y, { size: 12, color: ink, halo });
      lines.forEach((l, n) => text(ctx, l, x, y + LINE * (n + 1), ink, halo, font));
      drawn++;
      return;
    }
  };

  // Iterates already reached that left the view (facts first).
  for (const path of paths) {
    const n = path.points.length;
    const reached = Math.max(0, Math.min(n - 1, Math.floor(t + 1e-9)));
    for (let j = 0; j < reached; j++) {
      const a = o.toPx(path.points[j][0], path.points[j][1]);
      const q = path.points[j + 1];
      const c = o.toPx(q[0], q[1]);
      if (!inside(a, frame) || inside(c, frame)) continue;
      const ex = exitPoint([a, c], frame);
      if (!ex) continue;
      place(
        ex.p,
        ex.dir,
        [b('x'), sub(String(j + 1)), m(` = (${fmtCoord(q[0])}, ${fmtCoord(q[1])})`)],
        `${path.name}, ${sideWords(c, frame)}`,
      );
    }
  }

  if (!step) return drawn;
  const col = o.colors.series[step.slot % o.colors.series.length];
  const kk = String(step.k);
  // Most important first: a label that finds no free place is dropped (its chevron stays).
  const rank = { gn: 0, lm: 1, mu0: 2 } as const;
  for (const mark of [...step.marks].sort((a, c) => rank[a.role] - rank[c.role])) {
    const target = o.toPx(mark.at[0], mark.at[1]);
    if (inside(target, frame)) continue;
    const ex = exitPoint(
      mark.path.map(([x, y]) => o.toPx(x, y)),
      frame,
    );
    if (!ex) continue;
    chevron(ctx, ex.p, ex.dir, col, halo);
    const coords = `(${fmtCoord(mark.at[0])}, ${fmtCoord(mark.at[1])})`;
    const side = sideWords(target, frame);
    if (mark.role === 'gn') {
      const hidden = (mark.trials ?? []).filter(([x, y]) => !inside(o.toPx(x, y), frame)).length;
      place(
        ex.p,
        ex.dir,
        [b('x'), sub(kk), m(' + '), b('p'), sub(kk), m(` = ${coords}`)],
        `Gauss–Newton point, ${side}` +
          (hidden > 0 ? ` · ${hidden} failed trial${hidden === 1 ? '' : 's'} there too` : ''),
      );
    } else if (mark.role === 'lm') {
      place(
        ex.p,
        ex.dir,
        [b('x'), sub(kk), m(' + '), b('h'), sub(kk), m(` = ${coords}`)],
        `${mark.rejected ? 'rejected trial' : 'trial'}, ${side}`,
      );
    } else {
      place(
        ex.p,
        ex.dir,
        [b('x'), sub(kk), m(' + '), b('p'), sub(kk), m(` = ${coords}`)],
        `Gauss–Newton point (μ = 0), ${side}`,
      );
    }
  }
  return drawn;
}

/**
 * The boxes of the labels that `drawOverlays2D` and Contour2D draw (approximate, generous):
 * arrow labels centered 12 px left of the shaft's midpoint, point labels to the right of the
 * point, and the 𝐱⋆ label to the right of each cross.
 */
export function labelBoxes(
  overlays: readonly Overlay2D[],
  stars: readonly Px[],
  toPx: (x: number, y: number) => Px,
): Box[] {
  const out: Box[] = stars.map(([x, y]) => ({ x: x + 8, y: y - 24, w: 140, h: 22 }));
  for (const o of overlays) {
    if (o.kind === 'arrow' && o.label) {
      const a = toPx(o.from[0], o.from[1]),
        c = toPx(o.to[0], o.to[1]);
      const len = Math.hypot(c[0] - a[0], c[1] - a[1]) || 1;
      const cx = (a[0] + c[0]) / 2 - ((c[1] - a[1]) / len) * 12,
        cy = (a[1] + c[1]) / 2 + ((c[0] - a[0]) / len) * 12;
      out.push({ x: cx - 48, y: cy - 12, w: 96, h: 22 });
    } else if (o.kind === 'point' && o.label) {
      const [x, y] = toPx(o.at[0], o.at[1]);
      out.push({ x: x + 4, y: y - 14, w: 90, h: 22 });
    }
  }
  return out;
}
