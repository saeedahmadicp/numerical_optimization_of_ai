/**
 * The search ray on the contour plot: 𝐱₀ + α𝐩 for α ≥ 0, the part of it the φ panel shows,
 * the direction 𝐩 (and −∇f(𝐱₀) when 𝐩 is the Newton direction), the trial points of every
 * method in order, the minimizer of f along the ray and the φ-panel probe.
 */
import type { View2D } from '../../viz';
import { drawMath, mathBold as b, mathMain as m, mathSup as sup, mathVar as v } from '../../viz';
import { easeInOut } from '../../play/timeline';
import { trialOf, type LineFn } from './geometry';
import type { PhiRun } from './PhiPanel';

export interface PlaneScene {
  line: LineFn;
  /** Steepest-descent direction −∇f(𝐱₀) (drawn when 𝐩 differs from it). */
  steepest: number[] | null;
  newton: boolean;
  /** α range of the φ panel and the longest trial (the ray is drawn that far). */
  hi: number;
  rayEnd: number;
  runs: readonly PhiRun[];
  focusId: string | null;
  alphaStar: number | null;
  probe: number | null;
  ease: boolean;
}

type Domain = [[number, number], [number, number]];

/**
 * A view of the plane that frames the search: the box around the segment 𝐱₀ … 𝐱₀ + hi·𝐩 (the
 * stretch the φ panel shows), padded so the segment spans about 60 % of it. Null when the segment
 * is already at least `share` of the problem's domain (the whole domain then frames it well).
 */
export function searchFrame(
  domain: Domain,
  x0: readonly number[],
  p: readonly number[] | null,
  hi: number,
  share = 0.25,
): Domain | null {
  if (!p || !(hi > 0) || !p.every(Number.isFinite)) return null;
  const span = Math.max(domain[0][1] - domain[0][0], domain[1][1] - domain[1][0]);
  const len = hi * Math.hypot(p[0], p[1]);
  if (!(len > 0) || !Number.isFinite(len) || len >= share * span) return null;
  const side = Math.max(len / 0.6, 0.05 * span);
  const cx = x0[0] + 0.5 * hi * p[0];
  const cy = x0[1] + 0.5 * hi * p[1];
  return [
    [cx - side / 2, cx + side / 2],
    [cy - side / 2, cy + side / 2],
  ];
}

/** The first of `spots` at least `gap` px from every point (or null). */
function clearSpot(
  spots: readonly [number, number][],
  points: readonly [number, number][],
  gap: number,
): [number, number] | null {
  for (const q of spots)
    if (points.every((r) => Math.hypot(q[0] - r[0], q[1] - r[1]) >= gap)) return q;
  return null;
}

function arrow(
  ctx: CanvasRenderingContext2D,
  from: [number, number],
  to: [number, number],
  color: string,
  halo: string,
  width = 1.75,
) {
  const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const head = (c: string, w: number) => {
    ctx.strokeStyle = c;
    ctx.fillStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(to[0] - Math.cos(ang) * 5, to[1] - Math.sin(ang) * 5);
    ctx.stroke();
  };
  head(halo, width + 3);
  head(color, width);
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - 9 * Math.cos(ang - 0.42), to[1] - 9 * Math.sin(ang - 0.42));
  ctx.lineTo(to[0] - 9 * Math.cos(ang + 0.42), to[1] - 9 * Math.sin(ang + 0.42));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function dot(
  ctx: CanvasRenderingContext2D,
  [x, y]: [number, number],
  r: number,
  c: string,
  halo: string,
  ring = false,
) {
  ctx.beginPath();
  ctx.arc(x, y, r + 1.75, 0, Math.PI * 2);
  ctx.fillStyle = halo;
  if (!ring) ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (ring) {
    ctx.strokeStyle = halo;
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.strokeStyle = c;
    ctx.lineWidth = 1.75;
    ctx.stroke();
  } else {
    ctx.fillStyle = c;
    ctx.fill();
  }
}

export function drawPlane(ctx: CanvasRenderingContext2D, view: View2D, s: PlaneScene) {
  const { colors: c, toPx } = view;
  const P = (a: number) => {
    const q = s.line.point(a);
    return toPx(q[0], q[1]);
  };
  const o = P(0);
  ctx.save();
  ctx.lineCap = 'round';
  // The whole ray (dashed), then the stretch the φ panel shows (solid).
  const far = P(Math.max(s.rayEnd, s.hi) * 4);
  ctx.setLineDash([4, 5]);
  ctx.strokeStyle = c.text3;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(o[0], o[1]);
  ctx.lineTo(far[0], far[1]);
  ctx.stroke();
  ctx.setLineDash([]);
  const end = P(s.hi);
  for (const [w, col] of [
    [5, c.halo],
    [2, c.text],
  ] as const) {
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(o[0], o[1]);
    ctx.lineTo(end[0], end[1]);
    ctx.stroke();
  }
  // A cap at the end of the φ-panel stretch.
  const ang = Math.atan2(end[1] - o[1], end[0] - o[0]);
  const nx = -Math.sin(ang) * 5,
    ny = Math.cos(ang) * 5;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(end[0] - nx, end[1] - ny);
  ctx.lineTo(end[0] + nx, end[1] + ny);
  ctx.stroke();

  // Directions at 𝐱₀ (fixed screen length).
  const len = Math.min(70, Math.max(40, Math.min(view.width, view.height) * 0.16));
  const unit = (d: readonly number[]) => {
    const a = toPx(s.line.x0[0] + d[0], s.line.x0[1] + d[1]);
    const dx = a[0] - o[0],
      dy = a[1] - o[1];
    const n = Math.hypot(dx, dy) || 1;
    return [o[0] + (dx / n) * len, o[1] + (dy / n) * len] as [number, number];
  };
  if (s.newton && s.steepest && s.steepest.some((x) => x !== 0)) {
    const g = unit(s.steepest);
    arrow(ctx, o, g, c.text3, c.halo, 1.25);
    drawMath(ctx, [m('−∇'), v('f')], g[0] + 6, g[1] - 6, {
      size: 12,
      color: c.text3,
      halo: c.halo,
    });
  }
  // Where the markers sit (the labels keep clear of them).
  const marks: [number, number][] = [];
  for (const r of s.runs)
    for (let k = 1; k < r.trace.length; k++) {
      const a = trialOf(r.trace[k]).alpha;
      if (Number.isFinite(a) && k <= Math.ceil(r.t)) marks.push(P(a));
    }
  if (s.alphaStar !== null) marks.push(P(s.alphaStar));

  const pt = unit(s.line.p);
  arrow(ctx, o, pt, c.text, c.halo);
  {
    // The 𝐩 label: past the tip, or beside the shaft on either side, wherever no marker is.
    const ux = (pt[0] - o[0]) / len,
      uy = (pt[1] - o[1]) / len;
    const mid: [number, number] = [o[0] + ux * len * 0.6, o[1] + uy * len * 0.6];
    const spot = clearSpot(
      [
        [pt[0] + ux * 9, pt[1] + uy * 9 + 4],
        [mid[0] - uy * 11, mid[1] + ux * 11 + 4],
        [mid[0] + uy * 11, mid[1] - ux * 11 + 4],
      ],
      marks,
      11,
    );
    if (spot)
      drawMath(ctx, [b('p')], spot[0], spot[1], {
        size: 13,
        align: 'center',
        color: c.text,
        halo: c.halo,
      });
  }

  // The minimizer of f along the ray (in the φ-panel range).
  if (s.alphaStar !== null) {
    const [px, py] = P(s.alphaStar);
    for (const [w, col] of [
      [3.5, c.halo],
      [1.5, c.text],
    ] as const) {
      ctx.strokeStyle = col;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.moveTo(px - 5, py);
      ctx.lineTo(px + 5, py);
      ctx.moveTo(px, py - 5);
      ctx.lineTo(px, py + 5);
      ctx.stroke();
    }
    // Label beside the ray (either side), clear of the trial points on it.
    const nx2 = Math.sin(ang) * 14,
      ny2 = -Math.cos(ang) * 14;
    const others = marks.filter((q) => Math.hypot(q[0] - px, q[1] - py) > 0.5);
    const spot = clearSpot(
      [
        [px - nx2, py - ny2 + 4],
        [px + nx2, py + ny2 + 4],
      ],
      others,
      10,
    );
    if (spot)
      drawMath(ctx, [v('α'), sup('⋆')], spot[0], spot[1], {
        size: 12,
        align: 'center',
        color: c.text2,
        halo: c.halo,
      });
  }

  // Trials of every method, the focused method on top.
  const order = [...s.runs].sort((a, z) => Number(a.id === s.focusId) - Number(z.id === s.focusId));
  for (const r of order) {
    const col = c.series[r.slot % c.series.length];
    const last = r.trace.length - 1;
    const kf = Math.min(last, Math.floor(r.t + 1e-9));
    const frac = r.t - kf;
    const u = kf < last && frac > 0 ? (s.ease ? easeInOut(frac) : frac) : 0;
    for (let k = 1; k <= kf; k++) {
      const t = trialOf(r.trace[k]);
      if (!Number.isFinite(t.alpha)) continue;
      const cur = k === kf && u === 0;
      const done = t.accepted && k === last;
      ctx.globalAlpha = cur || done || r.id === s.focusId ? 1 : 0.7;
      dot(ctx, P(t.alpha), cur ? 4.25 : 2.75, col, c.halo);
      if (done) dot(ctx, P(t.alpha), 7, col, c.halo, true);
      ctx.globalAlpha = 1;
    }
    if (u > 0) {
      const a0 = kf === 0 ? 0 : trialOf(r.trace[kf]).alpha;
      const a1 = trialOf(r.trace[kf + 1]).alpha;
      dot(ctx, P(a0 + (a1 - a0) * u), 4.25, col, c.halo);
    }
  }
  // Probe from the φ panel.
  if (s.probe !== null) dot(ctx, P(s.probe), 5, c.crosshair, c.halo, true);
  ctx.restore();
}
