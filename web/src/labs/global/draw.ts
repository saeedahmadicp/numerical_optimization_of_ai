/**
 * Canvas layer of the global lab: the geometry of each method's current step, drawn over the
 * contour field (Contour2D `overlay`), under the paths.
 *
 *   annealing        walker 𝐱ₖ, 1σ/2σ proposal ellipse around it, proposal 𝐲 (× when rejected)
 *   particle swarm   particles, velocity vectors 𝐯ᵢ, personal bests 𝐩ᵢ (focused method), 𝐠
 *   DE               population, trial vectors 𝐮ᵢ (segment from the target), and for one target
 *                    the full construction: the donor difference (dashed), the scaled step from
 *                    the base donor to the mutant 𝐯ᵢ, the crossover rectangle spanned by 𝐱ᵢ
 *                    and 𝐯ᵢ, the trial at one of its corners
 *   CMA-ES           1σ and 2σ ellipses of N(𝐦, σ²C) before and after the update, the λ
 *                    samples (the μ selected filled), the mean shift, the evolution path 𝐩_c
 *   basin hopping    hop box, hop 𝐱ₖ → 𝐲, the Nelder–Mead descent 𝐲 → 𝐳, the minima found (□)
 *
 * Every frame between steps interpolates (see geometry.ts `phaseAt`); whole steps are exact.
 */
import type { ChartColors } from '../../ui/colors';
import type { Step, Vector } from '../../core/types';
import { ellipsePoints } from '../../viz/overlays2d';
import { drawMath, b as mb, sub as ms, type MathRun } from '../../viz/mathText';
import { easeInOut } from '../../play/timeline';
import {
  deDonors,
  deHighlight,
  gaussianEllipseMatrix,
  lerp,
  lerpCov,
  lerpPt,
  phaseAt,
  sub,
  trackPoint,
  type Pt,
} from './geometry';

export interface MethodLayer {
  method: string;
  slot: number;
  trace: readonly Step[];
  /** Local continuous time of this method (player.localT(i)). */
  lt: number;
  /** The focused method gets the detailed construction (labels, personal bests, DE target). */
  focused: boolean;
  muted: boolean;
  params: Record<string, unknown>;
}

export interface LayerView {
  toPx: (x: number, y: number) => [number, number];
  width: number;
  height: number;
  colors: ChartColors;
}

export interface DrawOptions {
  ease: boolean;
  /** The search box [[lo, hi], [lo, hi]]. */
  box: readonly (readonly [number, number])[];
}

/** Velocity arrows are drawn to scale up to this length (CSS px), then capped. */
export const MAX_ARROW_PX = 48;

type Ctx = CanvasRenderingContext2D;

/**
 * Set while an unfocused layer is drawn next to two or more others: glyphs skip their halo
 * strokes (half the canvas work for a population; the focused method keeps full contrast).
 */
let lite = false;

// ── primitives ────────────────────────────────────────────────────────────────────────

function dot(ctx: Ctx, [x, y]: Pt, r: number, fill: string, halo: string, haloW = 1.5) {
  if (haloW > 0 && !lite) {
    ctx.beginPath();
    ctx.arc(x, y, r + haloW, 0, Math.PI * 2);
    ctx.fillStyle = halo;
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

function ring(ctx: Ctx, [x, y]: Pt, r: number, stroke: string, halo: string, w = 1.5) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (!lite) {
    ctx.lineWidth = w + 2.5;
    ctx.strokeStyle = halo;
    ctx.stroke();
  }
  ctx.lineWidth = w;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function cross(ctx: Ctx, [x, y]: Pt, r: number, stroke: string, halo: string, w = 1.5) {
  for (const [lw, c] of [
    [w + 2.5, halo],
    [w, stroke],
  ] as const) {
    ctx.beginPath();
    ctx.moveTo(x - r, y - r);
    ctx.lineTo(x + r, y + r);
    ctx.moveTo(x - r, y + r);
    ctx.lineTo(x + r, y - r);
    ctx.lineWidth = lw;
    ctx.strokeStyle = c;
    ctx.stroke();
  }
}

/** A small square: a local minimum that basin hopping found (the problem's known minima are +). */
function square(ctx: Ctx, [x, y]: Pt, r: number, stroke: string, halo: string, w = 1.3) {
  if (!lite) {
    ctx.lineWidth = w + 2.5;
    ctx.strokeStyle = halo;
    ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
  }
  ctx.lineWidth = w;
  ctx.strokeStyle = stroke;
  ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
}

function diamond(ctx: Ctx, [x, y]: Pt, r: number, stroke: string, halo: string, fill?: string) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.lineWidth = 4;
  ctx.strokeStyle = halo;
  ctx.stroke();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

/** Best-so-far glyph: a ring with a center dot (a "target"). */
function bestMark(ctx: Ctx, p: Pt, color: string, halo: string) {
  ring(ctx, p, 6, color, halo, 1.6);
  dot(ctx, p, 1.8, color, halo, 0);
}

function line(ctx: Ctx, a: Pt, b: Pt, color: string, w: number, dash: number[] = []) {
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.lineWidth = w;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.setLineDash([]);
}

function arrow(ctx: Ctx, a: Pt, b: Pt, color: string, w: number, head = 5, dash: number[] = []) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 0.5) return;
  const ux = dx / len,
    uy = dy / len;
  const h = Math.min(head, len * 0.6);
  line(ctx, a, [b[0] - ux * h * 0.6, b[1] - uy * h * 0.6], color, w, dash);
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - ux * h - uy * h * 0.55, b[1] - uy * h + ux * h * 0.55);
  ctx.lineTo(b[0] - ux * h + uy * h * 0.55, b[1] - uy * h - ux * h * 0.55);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function polyline(ctx: Ctx, pts: readonly Pt[], color: string, w: number, dash: number[] = []) {
  if (pts.length < 2) return;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
  ctx.lineWidth = w;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.setLineDash([]);
}

function ellipse(
  ctx: Ctx,
  v: LayerView,
  center: Pt,
  M: number[][] | null,
  r: number,
  stroke: string,
  w: number,
  opts: { dash?: number[]; fill?: string } = {},
) {
  if (!M) return;
  const pts = ellipsePoints(center, M, r, 120);
  if (!pts) return;
  ctx.setLineDash(opts.dash ?? []);
  ctx.beginPath();
  pts.forEach(([x, y], i) => {
    const [px, py] = v.toPx(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  if (opts.fill) {
    ctx.fillStyle = opts.fill;
    ctx.fill();
  }
  ctx.lineWidth = w;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.setLineDash([]);
}

/** A translucent version of a CSS color (hex or rgb) for fills. */
function tint(color: string, a: number): string {
  const c = color.trim();
  if (c.startsWith('#')) {
    const h = c.length === 4 ? [...c.slice(1)].map((x) => x + x).join('') : c.slice(1, 7);
    const n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const [r, g, b] = m[1].split(/[ ,/]+/).filter(Boolean);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return c;
}

const vecPt = (x: unknown): Pt => [(x as Vector)[0], (x as Vector)[1]];
const label = (name: string, index?: string): MathRun[] =>
  index ? [mb(name), ms(index, 'italic')] : [mb(name)];

function mathLabel(ctx: Ctx, v: LayerView, runs: MathRun[], at: Pt, dx = 8, dy = -8) {
  drawMath(ctx, runs, at[0] + dx, at[1] + dy, {
    size: 12,
    align: dx < 0 ? 'right' : 'left',
    color: v.colors.text2,
    halo: v.colors.halo,
  });
}

// ── search box ────────────────────────────────────────────────────────────────────────

/** The search box Ω as a hairline rectangle with its name (CMA-ES and basin hopping may leave it). */
export function drawSearchBox(ctx: Ctx, v: LayerView, box: DrawOptions['box']) {
  const [x0, y0] = v.toPx(box[0][0], box[1][1]);
  const [x1, y1] = v.toPx(box[0][1], box[1][0]);
  ctx.save();
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = v.colors.text3;
  ctx.globalAlpha = 0.8;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();
  // Name the box in its top-right corner (inside the view; the zoom buttons sit bottom-right).
  const lx = Math.min(x1, v.width) - 8;
  const ly = Math.max(y0, 0) + 16;
  if (lx > 24 && ly < v.height - 8)
    drawMath(ctx, [{ t: 'Ω', style: 'main' }], lx, ly, {
      size: 13,
      align: 'right',
      color: v.colors.text3,
      halo: v.colors.halo,
    });
}

// ── tracks ────────────────────────────────────────────────────────────────────────────

/** Segments younger than this many steps fade from strong to faint; older ones stay faint. */
export const TRACK_FADE = 40;

/** Opacity of a track segment of age a (steps behind the playhead). */
export function trackAlpha(age: number): number {
  return age >= TRACK_FADE ? 0.16 : 0.16 + 0.74 * (1 - age / TRACK_FADE);
}

/**
 * Each method's track (the chain of annealing and basin hopping, the swarm's best, the best DE
 * member, the CMA-ES mean) through the last complete step: recent segments strong, older ones
 * faint, so the populations stay the subject. The glyphs of draw*() carry the motion.
 */
export function drawTracks(ctx: Ctx, v: LayerView, layers: readonly MethodLayer[]) {
  const ordered = [...layers].sort((a, b) => Number(a.focused) - Number(b.focused));
  for (const layer of ordered) {
    const K = Math.min(layer.trace.length - 1, Math.floor(layer.lt + 1e-9));
    if (K < 1) continue;
    const pts = layer.trace.slice(0, K + 1).map((s) => {
      const q = trackPoint(layer.method, s);
      return v.toPx(q[0], q[1]);
    });
    const color = v.colors.series[layer.slot];
    const w = layer.focused ? 1.5 : 1.2;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const mute = layer.muted ? 0.4 : 1;
    // Old segments in one faint stroke, then the recent ones one by one.
    const cut = Math.max(0, K - TRACK_FADE);
    if (cut > 0) {
      ctx.globalAlpha = trackAlpha(TRACK_FADE) * mute;
      polyline(ctx, pts.slice(0, cut + 1), color, w);
    }
    for (let j = cut; j < K; j++) {
      const a = pts[j],
        b = pts[j + 1];
      if (a[0] === b[0] && a[1] === b[1]) continue;
      ctx.globalAlpha = trackAlpha(K - 1 - j) * mute;
      line(ctx, a, b, v.colors.halo, w + 2);
      line(ctx, a, b, color, w);
    }
    ctx.restore();
  }
}

// ── per method ────────────────────────────────────────────────────────────────────────

export function drawMethodLayers(
  ctx: Ctx,
  v: LayerView,
  layers: readonly MethodLayer[],
  opts: DrawOptions,
) {
  // Unfocused first, the focused method on top.
  const ordered = [...layers].sort((a, b) => Number(a.focused) - Number(b.focused));
  for (const layer of ordered) {
    if (layer.trace.length === 0) continue;
    lite = !layer.focused && layers.length > 2;
    ctx.save();
    ctx.globalAlpha = layer.muted ? 0.32 : 1;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const { K, p } = phaseAt(layer.lt, layer.trace.length);
    const e = (u: number) => (opts.ease ? easeInOut(u) : u);
    const args = { ctx, v, layer, K, p, e, opts };
    switch (layer.method) {
      case 'simulated_annealing':
        drawAnnealing(args);
        break;
      case 'particle_swarm':
        drawSwarm(args);
        break;
      case 'differential_evolution':
        drawDE(args);
        break;
      case 'cma_es':
        drawCMA(args);
        break;
      case 'basin_hopping':
        drawBasinHopping(args);
        break;
    }
    ctx.restore();
  }
  lite = false;
}

interface Args {
  ctx: Ctx;
  v: LayerView;
  layer: MethodLayer;
  K: number;
  p: number;
  e: (u: number) => number;
  opts: DrawOptions;
}

function drawAnnealing({ ctx, v, layer, K, p, e }: Args) {
  const color = v.colors.series[layer.slot];
  const halo = v.colors.halo;
  const px = (q: Pt) => v.toPx(q[0], q[1]);
  const info = layer.trace[K].info;
  if (K === 0) {
    dot(ctx, px(vecPt(info.current)), 4.5, color, halo);
    return;
  }
  const prev = vecPt(layer.trace[K - 1].info.current);
  const y = vecPt(info.candidate);
  const sd = info.proposal_sd as Vector;
  const accepted = info.accepted === true;
  // Proposal distribution N(𝐱ₖ, diag σₖ²): 1σ (solid) and 2σ (dashed) ellipses.
  const M = [
    [1 / (sd[0] * sd[0]), 0],
    [0, 1 / (sd[1] * sd[1])],
  ];
  ctx.save();
  ctx.globalAlpha *= 0.9;
  ellipse(ctx, v, prev, M, 1, color, 1, { fill: tint(color, 0.1) });
  ellipse(ctx, v, prev, M, 2, color, 0.8, { dash: [3, 4] });
  ctx.restore();
  // The proposal 𝐲 appears, then the walker moves to it (accepted) or stays (rejected).
  const show = sub(p, 0.1, 0.4);
  if (show > 0) {
    ctx.save();
    ctx.globalAlpha *= accepted ? show : show * (p >= 1 ? 0.75 : 1);
    line(ctx, px(prev), px(y), color, 1.2, accepted ? [] : [3, 3]);
    if (accepted) ring(ctx, px(y), 4, color, halo, 1.4);
    else cross(ctx, px(y), 4, color, halo, 1.5);
    if (layer.focused) mathLabel(ctx, v, label('y'), px(y));
    ctx.restore();
  }
  const move = accepted ? e(sub(p, 0.45, 1)) : 0;
  const walker = lerpPt(prev, y, move);
  dot(ctx, px(walker), 4.5, color, halo);
  if (layer.focused) mathLabel(ctx, v, label('x', 'k'), px(walker), -9, 14);
  bestMark(ctx, px(vecPt(info.best)), color, halo);
}

function drawSwarm({ ctx, v, layer, K, p, e }: Args) {
  const color = v.colors.series[layer.slot];
  const halo = v.colors.halo;
  const now = layer.trace[K].info;
  const before = K > 0 ? layer.trace[K - 1].info : now;
  const u = K > 0 ? e(p) : 1;
  const X0 = before.particles as Vector[];
  const X1 = now.particles as Vector[];
  const V0 = before.velocities as Vector[];
  const V1 = now.velocities as Vector[];
  if (layer.focused) {
    ctx.save();
    ctx.globalAlpha *= 0.55;
    for (const pb of now.personal_best as Vector[]) {
      const q = v.toPx(pb[0], pb[1]);
      ring(ctx, q, 2.2, color, halo, 1);
    }
    ctx.restore();
  }
  X1.forEach((x1, i) => {
    const pos = lerpPt(X0[i], x1, u);
    const vel = lerpPt(V0[i], V1[i], u);
    const a = v.toPx(pos[0], pos[1]);
    const b = v.toPx(pos[0] + vel[0], pos[1] + vel[1]);
    let dx = b[0] - a[0],
      dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    if (len > MAX_ARROW_PX) {
      dx *= MAX_ARROW_PX / len;
      dy *= MAX_ARROW_PX / len;
    }
    ctx.save();
    ctx.globalAlpha *= 0.85;
    arrow(ctx, a, [a[0] + dx, a[1] + dy], color, 1.2, 5);
    ctx.restore();
    dot(ctx, a, 3, color, halo, 1.2);
  });
  const g0 = vecPt(before.global_best);
  const g1 = vecPt(now.global_best);
  const g = v.toPx(...lerpPt(g0, g1, u));
  bestMark(ctx, g, color, halo);
  if (layer.focused) mathLabel(ctx, v, label('g'), g, 9, -9);
}

function drawDE({ ctx, v, layer, K, p, e }: Args) {
  const color = v.colors.series[layer.slot];
  const halo = v.colors.halo;
  const px = (q: readonly number[]) => v.toPx(q[0], q[1]);
  const now = layer.trace[K].info;
  if (K === 0) {
    for (const x of now.population as Vector[]) dot(ctx, px(x), 3, color, halo, 1.2);
    bestMark(ctx, px(now.best as Vector), color, halo);
    return;
  }
  const old = layer.trace[K - 1].info.population as Vector[];
  const pop = now.population as Vector[];
  const trials = now.trials as Vector[];
  const mutants = now.mutants as Vector[];
  const accepted = now.accepted as boolean[];
  const appear = sub(p, 0.05, 0.45);
  const move = e(sub(p, 0.5, 1));
  // Trials: a segment from the target, × for a rejected trial.
  ctx.save();
  trials.forEach((u, i) => {
    const a = px(old[i]),
      b = px(u);
    ctx.globalAlpha = (layer.muted ? 0.32 : 1) * appear * (accepted[i] ? 0.7 : 0.4);
    line(ctx, a, b, color, 0.9, accepted[i] ? [] : [2, 3]);
    if (!accepted[i]) cross(ctx, b, 2.4, color, halo, 1);
  });
  ctx.restore();
  // The full construction for one target (focused method).
  if (layer.focused) {
    const i = deHighlight(old, mutants, trials, K % trials.length);
    const x = old[i],
      m = mutants[i],
      u = trials[i];
    ctx.save();
    ctx.globalAlpha *= Math.max(appear, 0.0001);
    const corners = [x, [m[0], x[1]], m, [x[0], m[1]]].map((q) => px(q));
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    corners.forEach((c, j) => (j === 0 ? ctx.moveTo(c[0], c[1]) : ctx.lineTo(c[0], c[1])));
    ctx.closePath();
    ctx.lineWidth = 1;
    ctx.strokeStyle = v.colors.text2;
    ctx.stroke();
    ctx.setLineDash([]);
    // The mutation: the difference vector of two donors (dashed), scaled by F and added to the
    // base donor (solid). The mutant does not depend on the target 𝐱ᵢ.
    const strategy = String(layer.params.strategy ?? 'rand/1/bin');
    const d = deDonors(
      old,
      layer.trace[K - 1].info.best as Vector,
      m,
      i,
      Number(layer.params.F ?? 0.8),
      strategy,
    );
    if (d) {
      const best = strategy === 'best/1/bin';
      arrow(ctx, px(d.minus), px(d.plus), v.colors.text3, 1, 5, [3, 3]);
      arrow(ctx, px(d.base), px(m), v.colors.text2, 1.3, 6);
      // Donors at their old positions (a replaced member has moved on by the end of the step).
      for (const q of [d.base, d.plus, d.minus]) ring(ctx, px(q), 4.5, v.colors.text2, halo, 1.2);
      const xr = (j: number): MathRun[] => [mb('x'), ms('r', 'italic'), ms(String(j))];
      mathLabel(ctx, v, best ? [mb('x'), ms('best')] : xr(1), px(d.base), -8, -8);
      mathLabel(ctx, v, xr(best ? 1 : 2), px(d.plus), 7, -7);
      mathLabel(ctx, v, xr(best ? 2 : 3), px(d.minus), -7, 13);
    }
    diamond(ctx, px(m), 4.5, color, halo, v.colors.surface);
    arrow(ctx, px(x), px(u), color, 1.6, 6);
    ring(ctx, px(x), 5.5, v.colors.text, halo, 1.4);
    mathLabel(ctx, v, label('x', 'i'), px(x), -9, 14);
    mathLabel(ctx, v, label('v', 'i'), px(m));
    mathLabel(ctx, v, label('u', 'i'), px(u), 8, 14);
    ctx.restore();
  }
  pop.forEach((x1, i) => {
    const q = lerpPt(old[i], x1, move);
    dot(ctx, px(q), 3, color, halo, 1.2);
  });
  bestMark(ctx, px(now.best as Vector), color, halo);
}

function drawCMA({ ctx, v, layer, K, p, e }: Args) {
  const color = v.colors.series[layer.slot];
  const halo = v.colors.halo;
  const px = (q: readonly number[]) => v.toPx(q[0], q[1]);
  const now = layer.trace[K].info;
  const mNew = now.mean as Vector;
  const CNew = now.covariance as number[][];
  const sNew = now.sigma as number;
  if (K === 0) {
    const M = gaussianEllipseMatrix(sNew, CNew);
    ellipse(ctx, v, vecPt(mNew), M, 1, color, 1.6, { fill: tint(color, 0.1) });
    ellipse(ctx, v, vecPt(mNew), M, 2, color, 1, { dash: [4, 4] });
    dot(ctx, px(mNew), 3.5, color, halo);
    return;
  }
  const mOld = now.sample_mean as Vector;
  const sOld = now.sample_sigma as number;
  const COld = now.sample_covariance as number[][];
  const X = now.population as Vector[];
  const sel = now.selected as number[];
  const selected = new Set(sel);
  const shoot = e(sub(p, 0, 0.35));
  const pick = sub(p, 0.35, 0.6);
  const update = e(sub(p, 0.6, 1));
  // Sampling distribution (before the update): faint once the update starts.
  const MOld = gaussianEllipseMatrix(sOld, COld);
  ctx.save();
  ctx.globalAlpha *= lerp(1, 0.55, update);
  ellipse(ctx, v, vecPt(mOld), MOld, 1, color, 1.1, {
    dash: update > 0 ? [4, 4] : [],
    fill: tint(color, lerp(0.1, 0.0, update)),
  });
  ellipse(ctx, v, vecPt(mOld), MOld, 2, color, 0.8, { dash: [2, 4] });
  ctx.restore();
  // Updated distribution N(𝐦ₖ₊₁, σₖ₊₁²Cₖ₊₁), morphing in.
  if (update > 0) {
    const mid = lerpPt(mOld, mNew, update);
    const cov = lerpCov(sOld, COld, sNew, CNew, update);
    const M = gaussianEllipseMatrix(1, cov);
    ellipse(ctx, v, mid, M, 1, color, 1.7, { fill: tint(color, 0.1 * update) });
    ellipse(ctx, v, mid, M, 2, color, 1, { dash: [4, 4] });
  }
  // Samples: shoot out from the old mean; the μ best fill in, the rest go hollow.
  X.forEach((x, j) => {
    const q = px(lerpPt(mOld, x, shoot));
    if (selected.has(j) && pick > 0) {
      dot(ctx, q, 3.2, color, halo, 1.2);
    } else {
      ctx.save();
      ctx.globalAlpha *= selected.has(j) ? 1 : lerp(1, 0.6, pick);
      ring(ctx, q, 3, color, halo, 1.2);
      ctx.restore();
    }
  });
  if (layer.focused && pick > 0) {
    // Rank labels 1…μ of the selected samples (𝐲_{i:λ}).
    sel.forEach((j, r) => {
      const q = px(X[j]);
      drawMath(ctx, [{ t: String(r + 1), style: 'main' }], q[0] + 6, q[1] - 5, {
        size: 11,
        color: v.colors.text2,
        halo,
      });
    });
  }
  // Mean shift 𝐦ₖ → 𝐦ₖ₊₁ = 𝐦ₖ + σ⟨𝐲⟩_w.
  if (update > 0) {
    const head = lerpPt(mOld, mNew, update);
    arrow(ctx, px(mOld), px(head), v.colors.text, 1.6, 6);
  }
  dot(ctx, px(lerpPt(mOld, mNew, update)), 3.5, color, halo);
  if (layer.focused) {
    mathLabel(ctx, v, label('m', update >= 1 ? 'k+1' : 'k'), px(lerpPt(mOld, mNew, update)), 9, 14);
    // Evolution path 𝐩_c (scaled by σ): the direction the mean has been drifting.
    if (p >= 1) {
      const pc = now.p_c as Vector;
      const tip: Pt = [mNew[0] + sNew * pc[0], mNew[1] + sNew * pc[1]];
      const a = px(mNew),
        b = px(tip);
      if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 10) {
        arrow(ctx, a, b, color, 1.2, 5, [3, 3]);
        mathLabel(ctx, v, [mb('p'), ms('c', 'italic')], b, 6, -6);
      }
    }
  }
  bestMark(ctx, px(now.best as Vector), color, halo);
}

function drawBasinHopping({ ctx, v, layer, K, p, e, opts }: Args) {
  const color = v.colors.series[layer.slot];
  const halo = v.colors.halo;
  const px = (q: readonly number[]) => v.toPx(q[0], q[1]);
  const now = layer.trace[K].info;
  const minima = (now.minima ?? []) as { x: Vector; f: number; hits: number }[];
  // The minima catalog so far (small squares; hit counts on the focused method).
  ctx.save();
  ctx.globalAlpha *= 0.85;
  for (const m of minima) {
    const q = px(m.x);
    square(ctx, q, 3, color, halo, 1.3);
    if (layer.focused && m.hits > 1)
      drawMath(ctx, [{ t: `×${m.hits}`, style: 'main' }], q[0] + 5, q[1] + 11, {
        size: 11,
        color: v.colors.text3,
        halo,
      });
  }
  ctx.restore();
  const path = (now.local_path as Vector[]).map((q) => px(q));
  const z = now.local_min as Vector;
  if (K === 0) {
    ring(ctx, px(now.start as Vector), 4, color, halo, 1.3);
    polyline(ctx, path, color, 1.3);
    dot(ctx, px(now.current as Vector), 4.5, color, halo);
    bestMark(ctx, px(now.best as Vector), color, halo);
    return;
  }
  const prev = layer.trace[K - 1].info.current as Vector;
  const y = now.start as Vector;
  const accepted = now.accepted === true;
  // Hop box 𝐱ₖ ± s (s = step·(hi − lo)).
  const frac = Number(layer.params.step ?? 0.1);
  const s = opts.box.map(([lo, hi]) => frac * (hi - lo));
  const c0 = px([prev[0] - s[0], prev[1] + s[1]]);
  const c1 = px([prev[0] + s[0], prev[1] - s[1]]);
  ctx.save();
  ctx.globalAlpha *= 0.7;
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.fillStyle = tint(color, 0.07);
  ctx.fillRect(c0[0], c0[1], c1[0] - c0[0], c1[1] - c0[1]);
  ctx.strokeRect(c0[0], c0[1], c1[0] - c0[0], c1[1] - c0[1]);
  ctx.setLineDash([]);
  ctx.restore();
  // The hop.
  const hop = e(sub(p, 0, 0.25));
  if (hop > 0) {
    arrow(ctx, px(prev), px(lerpPt(prev, y, hop)), color, 1.2, 5, [4, 3]);
    if (hop >= 1) ring(ctx, px(y), 3.5, color, halo, 1.3);
    if (layer.focused && hop >= 1) mathLabel(ctx, v, label('y'), px(y));
  }
  // The Nelder–Mead descent 𝐲 → 𝐳, revealed vertex by vertex.
  const slide = sub(p, 0.25, 0.75);
  if (slide > 0 && path.length > 1) {
    const n = Math.max(2, Math.ceil(slide * path.length));
    polyline(ctx, path.slice(0, n), color, 1.4);
  }
  const land = sub(p, 0.75, 1);
  if (land > 0) {
    if (accepted) ring(ctx, px(z), 5, color, halo, 1.5);
    else cross(ctx, px(z), 4, color, halo, 1.5);
    if (layer.focused) mathLabel(ctx, v, label('z'), px(z), 8, 14);
  }
  const walker = accepted ? lerpPt(prev, z, e(land)) : vecPt(prev);
  dot(ctx, px(walker), 4.5, color, halo);
  bestMark(ctx, px(now.best as Vector), color, halo);
}
