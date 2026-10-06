/**
 * Canvas drawing of one TSP panel: cities, the tour of the current step and the geometry of the
 * step that produced it (2-opt / Or-opt edge exchange, the SA proposal and temperature gauge,
 * GA best tours as small multiples, Ant System pheromone as edge opacity, the partial paths of
 * nearest neighbor and Held–Karp). Pure drawing: no React.
 */
import type { Step } from '../../core/types';
import type { ChartColors } from '../../ui/colors';
import { sig, int } from '../../core/format';
import {
  linearTicks,
  drawMath,
  measureMath,
  mathMain,
  mathVar,
  mathSub,
  mathSup,
  type MathRun,
} from '../../viz';
import { labelFont, tickFont } from '../../viz/axes';

export type Pt = readonly [number, number];

export interface TspDrawOptions {
  coords: readonly Pt[];
  domain: [[number, number], [number, number]];
  colors: ChartColors;
  slot: number;
  methodId: string;
  trace: readonly Step[];
  /** Integer step shown. */
  k: number;
  /** Progress towards the next step (0 when paused or under reduced motion). */
  frac: number;
  /** City under the pointer / being dragged / selected by keyboard. */
  hot?: number | null;
  /** Nearest-neighbor start city (ringed). */
  startCity?: number | null;
  /** SA schedule: T₀ and T_min in distance units (for the gauge). */
  schedule?: { t0: number; tMin: number } | null;
  /** Show index labels next to the cities. */
  labels?: boolean;
  /** L⋆ (or the best length found) for the length strip; null when unknown. */
  reference?: number | null;
  /** The reference is the best found, not the optimum. */
  bestFound?: boolean;
}

export interface Mapping {
  toPx: (p: Pt) => [number, number];
  fromPx: (x: number, y: number) => [number, number];
  /** Drawing box of the map (the GA strip sits below it). */
  box: { left: number; top: number; width: number; height: number };
}

/**
 * Equal-scale fit of the domain into a box (y up). `top` aligns the map to the top of the box
 * (a tall panel then keeps its map right under its header) instead of centering it.
 */
export function fitMapping(
  domain: [[number, number], [number, number]],
  box: { left: number; top: number; width: number; height: number },
  align: 'center' | 'top' = 'center',
): Mapping {
  const [[x0, x1], [y0, y1]] = domain;
  const s = Math.max(1e-9, Math.min(box.width / (x1 - x0), box.height / (y1 - y0)));
  const ox = box.left + (box.width - s * (x1 - x0)) / 2;
  const oy = align === 'top' ? box.top : box.top + (box.height - s * (y1 - y0)) / 2;
  const H = s * (y1 - y0);
  return {
    toPx: ([x, y]) => [ox + s * (x - x0), oy + H - s * (y - y0)],
    fromPx: (px, py) => [x0 + (px - ox) / s, y0 + (oy + H - py) / s],
    box,
  };
}

/** Panel geometry: map box, and the GA filmstrip box when there is room. */
export function panelLayout(methodId: string, width: number, height: number) {
  const pad = 10;
  const strip =
    methodId === 'tsp_genetic' && height >= 220 ? Math.min(86, Math.max(60, height * 0.24)) : 0;
  // SA: a band above the map for the temperature gauge, only when the map keeps ≥ 80 px.
  const band =
    methodId === 'tsp_simulated_annealing' && height - 2 * pad - strip - 46 >= 80 ? 46 : 0;
  return {
    band: band ? { left: pad, top: pad, width: width - 2 * pad, height: band } : null,
    map: {
      left: pad,
      top: pad + band,
      width: Math.max(1, width - 2 * pad),
      height: Math.max(1, height - 2 * pad - strip - band),
    },
    strip: strip
      ? { left: pad, top: height - pad - strip + 6, width: width - 2 * pad, height: strip - 6 }
      : null,
  };
}

const alpha = (color: string, a: number) => {
  // Series tokens are hex; fall back to globalAlpha for anything else.
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return color;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

function polyline(
  ctx: CanvasRenderingContext2D,
  pts: readonly [number, number][],
  closed: boolean,
) {
  if (!pts.length) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (closed && pts.length > 2) ctx.closePath();
}

function edgeLine(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number]) {
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
}

const isClosed = (s: Step) => s.info.closed === undefined || s.info.closed === true;

/** Draw the tour of a step (with a surface halo). */
function drawTour(
  ctx: CanvasRenderingContext2D,
  m: Mapping,
  coords: readonly Pt[],
  tour: readonly number[],
  closed: boolean,
  color: string,
  halo: string,
  width = 2,
) {
  const pts = tour.filter((c) => c >= 0 && c < coords.length).map((c) => m.toPx(coords[c]));
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  polyline(ctx, pts, closed);
  ctx.strokeStyle = halo;
  ctx.lineWidth = width + 3;
  ctx.stroke();
  polyline(ctx, pts, closed);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

type Rect = { x0: number; y0: number; x1: number; y1: number };
const overlaps = (a: Rect, b: Rect) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
/** Boxes around the city dots (labels and ticks must not cover them). */
const cityBoxes = (m: Mapping, coords: readonly Pt[], r = 6): Rect[] =>
  coords.map((c) => {
    const [x, y] = m.toPx(c);
    return { x0: x - r, y0: y - r, x1: x + r, y1: y + r };
  });

/** Grid and inset tick labels; a tick label that would touch a city is left out. */
function drawGrid(ctx: CanvasRenderingContext2D, m: Mapping, o: TspDrawOptions): Rect[] {
  const { colors } = o;
  const dots = cityBoxes(m, o.coords, 7);
  const placed: Rect[] = [];
  const free = (r: Rect) => !dots.some((d) => overlaps(d, r));
  const [[x0, x1], [y0, y1]] = o.domain;
  const [ax, ay] = m.toPx([x0, y0]);
  const [bx, by] = m.toPx([x1, y1]);
  const xs = linearTicks(x0, x1, 5),
    ys = linearTicks(y0, y1, 5);
  ctx.save();
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (const x of xs) {
    const [px] = m.toPx([x, y0]);
    ctx.beginPath();
    ctx.moveTo(Math.round(px) + 0.5, by);
    ctx.lineTo(Math.round(px) + 0.5, ay);
    ctx.stroke();
  }
  for (const y of ys) {
    const [, py] = m.toPx([x0, y]);
    ctx.beginPath();
    ctx.moveTo(ax, Math.round(py) + 0.5);
    ctx.lineTo(bx, Math.round(py) + 0.5);
    ctx.stroke();
  }
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.tick;
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'center';
  for (const x of xs.slice(1, -1)) {
    const [px] = m.toPx([x, y0]);
    const t = sig(x, 3).replace('-', '−');
    const w = ctx.measureText(t).width;
    const r = { x0: px - w / 2 - 1, y0: ay - 14, x1: px + w / 2 + 1, y1: ay - 1 };
    if (!free(r)) continue;
    ctx.fillText(t, px, ay - 2);
    placed.push(r);
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (const y of ys.slice(1, -1)) {
    const [, py] = m.toPx([x0, y]);
    const t = sig(y, 3).replace('-', '−');
    const w = ctx.measureText(t).width;
    const r = { x0: ax + 2, y0: py - 6, x1: ax + 4 + w, y1: py + 6 };
    if (!free(r)) continue;
    ctx.fillText(t, ax + 3, py);
    placed.push(r);
  }
  ctx.restore();
  return placed;
}

/**
 * Pheromone τ_ij as edge opacity and width (Ant System), on the scale (τ − τ_min)/(τ_max − τ_min).
 * A uniform field (step 0: τ = τ₀ on every edge) carries no information and is not drawn.
 * Returns false when the field is uniform.
 */
function drawPheromone(
  ctx: CanvasRenderingContext2D,
  m: Mapping,
  coords: readonly Pt[],
  tau: readonly (readonly number[])[],
  tauMin: number,
  tauMax: number,
  color: string,
): boolean {
  if (!(tauMax > 0) || !(tauMin >= 0) || tauMax <= tauMin * (1 + 1e-6)) return false;
  const n = Math.min(coords.length, tau.length);
  const span = tauMax - tauMin;
  ctx.save();
  ctx.lineCap = 'round';
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const r = (tau[i][j] - tauMin) / span;
      if (!(r >= 0.04)) continue;
      edgeLine(ctx, m.toPx(coords[i]), m.toPx(coords[j]));
      ctx.strokeStyle = alpha(color, Math.min(0.75, 0.06 + 0.6 * r * r));
      ctx.lineWidth = 0.75 + 4.5 * r;
      ctx.stroke();
    }
  ctx.restore();
  return true;
}

/** Removed edges (dashed ghosts, fading) and added edges (bold) of a local-search move. */
function drawMove(
  ctx: CanvasRenderingContext2D,
  m: Mapping,
  coords: readonly Pt[],
  move: Record<string, unknown>,
  o: TspDrawOptions,
  color: string,
  accepted: boolean,
) {
  const { colors } = o;
  const ok = ([a, b]: number[]) => a >= 0 && b >= 0 && a < coords.length && b < coords.length;
  const removed = ((move.removed as number[][] | undefined) ?? []).filter(ok);
  const added = ((move.added as number[][] | undefined) ?? []).filter(ok);
  const fade = 1 - 0.55 * o.frac;
  ctx.save();
  ctx.lineCap = 'round';
  // Removed edges: dashed ghosts with a cross at the midpoint.
  for (const [a, b] of accepted ? removed : []) {
    const pa = m.toPx(coords[a]),
      pb = m.toPx(coords[b]);
    edgeLine(ctx, pa, pb);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = colors.text3;
    ctx.globalAlpha = 0.85 * fade;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.setLineDash([]);
    const mx = (pa[0] + pb[0]) / 2,
      my = (pa[1] + pb[1]) / 2;
    ctx.beginPath();
    ctx.moveTo(mx - 3.5, my - 3.5);
    ctx.lineTo(mx + 3.5, my + 3.5);
    ctx.moveTo(mx + 3.5, my - 3.5);
    ctx.lineTo(mx - 3.5, my + 3.5);
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // Added edges: thick, with a halo (rejected SA proposals: dotted, thin).
  for (const [a, b] of added) {
    const pa = m.toPx(coords[a]),
      pb = m.toPx(coords[b]);
    if (accepted) {
      edgeLine(ctx, pa, pb);
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = 8;
      ctx.stroke();
      edgeLine(ctx, pa, pb);
      ctx.strokeStyle = color;
      ctx.lineWidth = 4.2 - 1.6 * o.frac;
      ctx.stroke();
    } else {
      edgeLine(ctx, pa, pb);
      ctx.setLineDash([1.5, 4]);
      ctx.strokeStyle = colors.text2;
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.restore();
}

function drawCities(
  ctx: CanvasRenderingContext2D,
  m: Mapping,
  o: TspDrawOptions,
  visited: Set<number> | null,
  current: number | null,
  color: string,
  ticks: readonly Rect[] = [],
) {
  const { coords, colors } = o;
  ctx.save();
  coords.forEach((c, i) => {
    const [x, y] = m.toPx(c);
    const hollow = visited !== null && !visited.has(i);
    ctx.beginPath();
    ctx.arc(x, y, 4.6, 0, Math.PI * 2);
    ctx.fillStyle = colors.halo;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, hollow ? 2.9 : 3.1, 0, Math.PI * 2);
    if (hollow) {
      ctx.strokeStyle = colors.text2;
      ctx.lineWidth = 1.3;
      ctx.stroke();
    } else {
      ctx.fillStyle = colors.text;
      ctx.fill();
    }
    if (i === current) {
      ctx.beginPath();
      ctx.arc(x, y, 7.5, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (i === o.startCity) {
      ctx.beginPath();
      ctx.rect(x - 6.5, y - 6.5, 13, 13);
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    if (i === o.hot) {
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
  if (o.labels) {
    ctx.font = tickFont(colors);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.lineJoin = 'round';
    // Label placement: try four offsets around the dot; skip a label that still collides.
    const obstacles: Rect[] = [...cityBoxes(m, coords, 4.5), ...ticks];
    const h = 9;
    coords.forEach((c, i) => {
      const [x, y] = m.toPx(c);
      const t = String(i);
      const w = ctx.measureText(t).width;
      const spots: [number, number][] = [
        [x + 5, y - 4], // NE (baseline)
        [x - 5 - w, y - 4], // NW
        [x + 5, y + 4 + h], // SE
        [x - 5 - w, y + 4 + h], // SW
      ];
      for (const [lx, ly] of spots) {
        const r = { x0: lx - 1, y0: ly - h - 1, x1: lx + w + 1, y1: ly + 2 };
        if (obstacles.some((ob, j) => j !== i && overlaps(ob, r))) continue;
        obstacles.push(r);
        ctx.strokeStyle = colors.halo;
        ctx.lineWidth = 3;
        ctx.strokeText(t, lx, ly);
        ctx.fillStyle = colors.text3;
        ctx.fillText(t, lx, ly);
        break;
      }
    });
  }
  ctx.restore();
}

/** Temperature gauge (log scale between T_min and T₀) and the acceptance rate. */
function drawGauge(
  ctx: CanvasRenderingContext2D,
  o: TspDrawOptions,
  step: Step,
  color: string,
  x: number,
  y: number,
  w: number,
) {
  const { colors } = o;
  const T = step.info.temperature as number | null;
  if (!o.schedule || T === null || !(T > 0)) return;
  const { t0, tMin } = o.schedule;
  const u = Math.min(1, Math.max(0, Math.log(T / tMin) / Math.log(t0 / tMin)));
  const rate = step.info.acceptance_rate as number | null;
  ctx.save();
  ctx.font = labelFont(colors);
  ctx.fillStyle = colors.text2;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  const value =
    rate === null ? `T = ${sig(T, 3)}` : `T = ${sig(T, 3)} · accepted ${sig(rate * 100, 3)} %`;
  ctx.font = tickFont(colors);
  const valueW = ctx.measureText(value).width;
  ctx.font = labelFont(colors);
  if (ctx.measureText('temperature').width + valueW + 12 < w) ctx.fillText('temperature', x, y - 3);
  ctx.textAlign = 'right';
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.text;
  ctx.fillText(value, x + w, y - 3);
  // Track + fill (log scale), ticks at T₀ and T_min.
  ctx.fillStyle = colors.grid;
  ctx.fillRect(x, y + 2, w, 5);
  ctx.fillStyle = color;
  ctx.fillRect(x, y + 2, w * u, 5);
  const tickRuns = (name: MathRun[], v: string): MathRun[] => [...name, mathMain(` = ${v}`)];
  const lo = tickRuns([mathVar('T'), mathSub('min')], sig(tMin, 2));
  const hi = tickRuns([mathVar('T'), mathSub('0')], sig(t0, 3));
  drawMath(ctx, lo, x, y + 19, { size: 11.5, color: colors.text3, align: 'left' });
  const loW = measureMath(ctx, lo, 11.5);
  const hiW = measureMath(ctx, hi, 11.5);
  drawMath(ctx, hi, x + w, y + 19, { size: 11.5, color: colors.text3, align: 'right' });
  ctx.font = labelFont(colors);
  ctx.fillStyle = colors.text3;
  ctx.textAlign = 'center';
  if (w - loW - hiW > ctx.measureText('log scale').width + 24)
    ctx.fillText('log scale', x + w / 2, y + 18);
  ctx.restore();
}

/** GA: the best tour at evenly spaced recorded generations up to k (small multiples). */
function drawFilmstrip(
  ctx: CanvasRenderingContext2D,
  o: TspDrawOptions,
  box: { left: number; top: number; width: number; height: number },
  color: string,
) {
  const { colors, trace } = o;
  const count = Math.max(2, Math.min(6, Math.floor(box.width / (box.height * 0.95))));
  const picks: number[] = [];
  for (let s = 0; s < count; s++) {
    const idx = Math.round((s / (count - 1)) * o.k);
    if (!picks.includes(idx)) picks.push(idx);
  }
  const cell = box.width / count;
  ctx.save();
  ctx.strokeStyle = colors.grid;
  ctx.beginPath();
  ctx.moveTo(box.left, Math.round(box.top - 4) + 0.5);
  ctx.lineTo(box.left + box.width, Math.round(box.top - 4) + 0.5);
  ctx.stroke();
  picks.forEach((idx, s) => {
    const step = trace[idx];
    if (!step) return;
    const label = 28;
    const sub = fitMapping(o.domain, {
      left: box.left + s * cell + 3,
      top: box.top + 2,
      width: cell - 6,
      height: box.height - label - 2,
    });
    const tour = step.info.tour as number[];
    drawTour(
      ctx,
      sub,
      o.coords,
      tour,
      true,
      idx === o.k ? color : alpha(color, 0.55),
      colors.halo,
      1.2,
    );
    ctx.font = tickFont(colors);
    ctx.fillStyle = idx === o.k ? colors.text : colors.text3;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const cx = box.left + s * cell + cell / 2;
    ctx.fillText(`g ${int(step.k)}`, cx, box.top + box.height - 13);
    ctx.fillText(sig(step.fun ?? NaN, 4), cx, box.top + box.height);
  });
  ctx.restore();
}

/** Length of the tour (or open path) a step shows: the best-so-far for the Ant System. */
function stepLength(methodId: string, s: Step): number | null {
  if (methodId === 'tsp_ant_colony') {
    const b = s.info.best_length;
    return typeof b === 'number' ? b : s.fun;
  }
  return s.fun;
}

/**
 * The tour length per recorded step under the map (in the free height of a tall panel), with
 * the reference L⋆ dashed and the playhead: the panel then reads as one figure.
 */
function drawLengthStrip(
  ctx: CanvasRenderingContext2D,
  o: TspDrawOptions,
  box: { left: number; top: number; width: number; height: number },
  color: string,
) {
  const { colors, trace } = o;
  const vals = trace.map((s) => stepLength(o.methodId, s));
  const finite = vals.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) return;
  const ref = o.reference ?? null;
  let lo = Math.min(...finite, ref ?? Infinity),
    hi = Math.max(...finite, ref ?? -Infinity);
  if (hi - lo < 1e-9) hi = lo + 1;
  const padY = (hi - lo) * 0.08;
  lo -= padY;
  hi += padY;
  const plot = {
    left: box.left + 40,
    top: box.top + 16,
    width: box.width - 46,
    height: box.height - 30,
  };
  if (plot.width < 60 || plot.height < 24) return;
  const n = Math.max(1, trace.length - 1);
  const X = (i: number) => plot.left + (i / n) * plot.width;
  const Y = (v: number) => plot.top + plot.height - ((v - lo) / (hi - lo)) * plot.height;
  ctx.save();
  // Title.
  ctx.font = labelFont(colors);
  ctx.fillStyle = colors.text3;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  const constructive = o.methodId === 'tsp_nearest_neighbor' || o.methodId === 'tsp_held_karp';
  const title = constructive
    ? 'path length while the tour is built, per recorded step'
    : o.methodId === 'tsp_ant_colony'
      ? 'best tour length so far, per recorded step'
      : 'tour length, per recorded step';
  ctx.fillText(title, box.left, box.top + 8);
  // Frame and y ticks (lo/hi as values).
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    Math.round(plot.left) + 0.5,
    Math.round(plot.top) + 0.5,
    Math.round(plot.width),
    Math.round(plot.height),
  );
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.tick;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(sig(hi - padY, 4), plot.left - 4, Y(hi - padY));
  ctx.fillText(sig(lo + padY, 4), plot.left - 4, Y(lo + padY));
  // Reference.
  if (ref !== null) {
    const y = Math.round(Y(ref)) + 0.5;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = colors.text3;
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.left + plot.width, y);
    ctx.stroke();
    ctx.setLineDash([]);
    const name: MathRun[] = o.bestFound
      ? [mathVar('L'), mathSub('best')]
      : [mathVar('L'), mathSup('⋆')];
    drawMath(ctx, name, plot.left + plot.width - 2, y - 4, {
      size: 11,
      color: colors.text2,
      align: 'right',
      halo: colors.halo,
    });
  }
  // The series: solid up to the playhead, faint after it.
  const line = (from: number, to: number, stroke: string) => {
    ctx.beginPath();
    let started = false;
    for (let i = from; i <= to; i++) {
      const v = vals[i];
      if (v === null || !Number.isFinite(v)) continue;
      if (!started) ctx.moveTo(X(i), Y(v));
      else ctx.lineTo(X(i), Y(v));
      started = true;
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.stroke();
  };
  line(o.k, trace.length - 1, alpha(color, 0.25));
  line(0, o.k, color);
  const v = vals[o.k];
  if (v !== null && Number.isFinite(v)) {
    const closed = isClosed(trace[o.k]);
    ctx.beginPath();
    ctx.arc(X(o.k), Y(v), 3.6, 0, Math.PI * 2);
    ctx.fillStyle = closed ? color : colors.halo;
    ctx.fill();
    ctx.strokeStyle = closed ? colors.halo : color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

/** Draw one panel. Returns the mapping (for hit testing). */
export function drawTspPanel(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  o: TspDrawOptions,
): Mapping {
  const layout = panelLayout(o.methodId, width, height);
  const m = fitMapping(o.domain, layout.map, 'top');
  const color = o.colors.series[o.slot] ?? o.colors.text;
  const step = o.trace[Math.min(o.k, o.trace.length - 1)];
  const ticks = drawGrid(ctx, m, o);
  if (!step) {
    drawCities(ctx, m, o, null, null, color, ticks);
    return m;
  }
  const info = step.info;
  // Ant System: the best tour so far is the result; the iteration-best tour is drawn faint.
  const aco = o.methodId === 'tsp_ant_colony';
  const acoBest = aco ? ((info.best_tour as number[] | undefined) ?? []) : [];
  const tour = aco && acoBest.length ? acoBest : ((info.tour as number[]) ?? []);
  const faint = aco
    ? ((info.tour as number[]) ?? [])
    : ((info.best_tour as number[] | undefined) ?? []);
  const closed = isClosed(step);

  let uniformTau = false;
  if (aco && Array.isArray(info.pheromone))
    uniformTau = !drawPheromone(
      ctx,
      m,
      o.coords,
      info.pheromone as number[][],
      info.tau_min as number,
      info.tau_max as number,
      color,
    );

  // Best tour so far, faint, when the shown tour is not it (SA current vs best, ACO iteration-best).
  if (faint.length && faint.join() !== tour.join()) {
    ctx.save();
    ctx.setLineDash([2, 3]);
    drawTour(ctx, m, o.coords, faint, true, o.colors.text3, 'transparent', 1.2);
    ctx.restore();
  }

  const move = info.move as Record<string, unknown> | null | undefined;
  const accepted = move ? move.accepted !== false : true;
  drawTour(
    ctx,
    m,
    o.coords,
    tour,
    closed,
    color,
    o.colors.halo,
    o.methodId === 'tsp_ant_colony' ? 1.6 : 2,
  );
  if (move) drawMove(ctx, m, o.coords, move, o, color, accepted);

  // Or-opt: ring the moved segment.
  const segment = move?.segment as number[] | undefined;
  if (segment) {
    ctx.save();
    for (const c of segment.filter((c) => c >= 0 && c < o.coords.length)) {
      const [x, y] = m.toPx(o.coords[c]);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
    ctx.restore();
  }

  const constructive = o.methodId === 'tsp_nearest_neighbor' || o.methodId === 'tsp_held_karp';
  const visited = constructive && !closed ? new Set(tour) : null;
  const current = constructive ? (closed ? null : tour[tour.length - 1]) : null;
  drawCities(
    ctx,
    m,
    { ...o, startCity: o.methodId === 'tsp_held_karp' ? 0 : o.startCity },
    visited,
    current ?? null,
    color,
    ticks,
  );

  // Ant System step 0: the pheromone is uniform; say so instead of drawing every edge.
  if (uniformTau && typeof info.tau_max === 'number') {
    const runs: MathRun[] = [
      mathVar('τ'),
      mathSub('ij'),
      mathMain(' = '),
      mathVar('τ'),
      mathSub('0'),
      mathMain(' = '),
      mathVar('m'),
      mathMain('/'),
      mathVar('C'),
      mathSup('nn'),
      mathMain(` = ${sig(info.tau_max, 3)}`),
    ];
    // Below the map when there is room, else over its bottom edge (on a halo).
    const mapBottom = m.toPx([o.domain[0][0], o.domain[1][0]])[1];
    const by = mapBottom + 15 < height - 2 ? mapBottom + 15 : mapBottom - 6;
    const bx = layout.map.left + 4;
    ctx.save();
    ctx.font = labelFont(o.colors);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    const lead = 'uniform pheromone ';
    ctx.strokeStyle = o.colors.halo;
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.strokeText(lead, bx, by);
    ctx.fillStyle = o.colors.text2;
    ctx.fillText(lead, bx, by);
    drawMath(ctx, runs, bx + ctx.measureText(lead).width, by, {
      size: 11.5,
      color: o.colors.text2,
      halo: o.colors.halo,
    });
    ctx.restore();
  }

  if (layout.band) {
    const w = Math.min(260, layout.band.width - 16);
    drawGauge(
      ctx,
      o,
      step,
      color,
      layout.band.left + (layout.band.width - w) / 2,
      layout.band.top + 14,
      w,
    );
  }
  if (layout.strip) drawFilmstrip(ctx, o, layout.strip, color);
  else {
    // A tall panel: the free height under the map holds the length strip.
    const mapBottom = m.toPx([o.domain[0][0], o.domain[1][0]])[1];
    const free = height - 10 - mapBottom - 22;
    if (free >= 90)
      drawLengthStrip(
        ctx,
        o,
        {
          left: layout.map.left,
          top: mapBottom + 26,
          width: layout.map.width,
          height: Math.min(190, free),
        },
        color,
      );
  }
  return m;
}
