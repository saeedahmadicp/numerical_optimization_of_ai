/**
 * The φ panel: the line function φ(α) = f(𝐱₀ + α𝐩) with the geometry of the acceptance tests,
 * its slope φ'(α) with the curvature band, and one lane per method with its acceptable set, its
 * trials in order (hops), and the interval each trial was chosen from. All three share one α
 * axis. The conditions are drawn for the focused method; the trials of every method are drawn.
 */
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { Step } from '../../core/types';
import type { LineSearchKind } from '../../methods/line_search/methods';
import { easeInOut } from '../../play/timeline';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import {
  crisp,
  drawMath,
  linearScale,
  linearTicks,
  mathMain as m,
  mathSub as sub,
  mathSup as sup,
  mathVar as v,
  measureMath,
  useCanvas,
  type MathRun,
  type Scale,
} from '../../viz';
import { LABEL_SIZE, labelFont, tickFont } from '../../viz/axes';
import { tick as fmtTick } from '../../core/format';
import {
  EXACT_C2,
  phiRange,
  quadraticModel,
  shortName,
  slopeRange,
  trialOf,
  zoomInterpolant,
  type LineFn,
  type Samples,
  type Trial,
  type WindowMode,
} from './geometry';
import { num } from './format';
import styles from './LineSearchLab.module.css';

export interface PhiRun {
  id: string;
  name: string;
  slot: number;
  kind: LineSearchKind;
  trace: readonly Step[];
  /** The run's local playhead (continuous). */
  t: number;
  /** α intervals in the window where the method accepts the step. */
  intervals: readonly [number, number][];
}

export interface PhiPanelProps {
  line: LineFn;
  samples: Samples;
  phi0: number;
  dphi0: number;
  hi: number;
  mode: WindowMode;
  runs: readonly PhiRun[];
  focusId: string | null;
  alphaStar: { alpha: number; phi: number } | null;
  ease: boolean;
  probe: number | null;
  onProbe: (alpha: number | null) => void;
  /** Click on the φ or φ' plot: set α₀ of the focused method. */
  onPickAlpha?: (alpha: number) => void;
  /** Click on a lane: focus that method. */
  onFocus?: (id: string) => void;
  /**
   * 'slope': give φ'(α) the larger frame (the phone's φ′ view). Default 'phi': φ(α) large, φ'
   * strip below it when there is room.
   */
  emphasis?: 'phi' | 'slope';
  ariaLabel: string;
}

interface Frame {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface Layout {
  narrow: boolean;
  phi: Frame;
  slope: Frame | null;
  lanes: Frame[];
  axisY: number;
}

const LANE_H = 18;
const LANE_GAP = 4;
/** Phones: the lane name sits in a band of its own above the lane (no arc or outline crosses it). */
const NAME_BAND = Math.ceil(LABEL_SIZE * 1.25);

function layout(w: number, h: number, nLanes: number, emphasis: 'phi' | 'slope' = 'phi'): Layout {
  const narrow = w < 560;
  const left = narrow ? 48 : 108;
  const right = w - 14;
  const axisH = 24;
  const band = narrow ? NAME_BAND : 0;
  const lanesH = nLanes * (band + LANE_H + LANE_GAP);
  const avail = h - 10 - axisH - lanesH - 10;
  const withSlope = emphasis === 'slope' || avail >= 170;
  const slopeH = !withSlope
    ? 0
    : emphasis === 'slope'
      ? Math.max(48, Math.round(avail * 0.62))
      : Math.max(48, Math.round(avail * 0.27));
  const phiH = avail - (withSlope ? slopeH + 10 : 0);
  const phi = { left, right, top: 10, bottom: 10 + phiH };
  const slope = withSlope
    ? { left, right, top: phi.bottom + 10, bottom: phi.bottom + 10 + slopeH }
    : null;
  let y = (slope ?? phi).bottom + 10;
  const lanes: Frame[] = [];
  for (let i = 0; i < nLanes; i++) {
    y += band;
    lanes.push({ left, right, top: y, bottom: y + LANE_H });
    y += LANE_H + LANE_GAP;
  }
  return { narrow, phi, slope, lanes, axisY: y + 2 };
}

/** Text with a halo in the UI face. */
function word(
  ctx: CanvasRenderingContext2D,
  s: string,
  x: number,
  y: number,
  c: ChartColors,
  o: { align?: CanvasTextAlign; color?: string; size?: number; weight?: number } = {},
) {
  ctx.save();
  ctx.font = `${o.weight ?? 500} ${o.size ?? LABEL_SIZE}px ${c.fontSans}`;
  ctx.textAlign = o.align ?? 'left';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = c.halo;
  ctx.strokeText(s, x, y);
  ctx.fillStyle = o.color ?? c.text2;
  ctx.fillText(s, x, y);
  ctx.restore();
}

function mathLabel(
  ctx: CanvasRenderingContext2D,
  runs: readonly MathRun[],
  x: number,
  y: number,
  c: ChartColors,
  align: 'left' | 'right' | 'center' = 'left',
  color = c.text2,
  size = 12.5,
) {
  drawMath(ctx, runs, x, y, { size, align, color, halo: c.halo, baseline: 'middle' });
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  halo: string,
  ring = false,
) {
  ctx.beginPath();
  ctx.arc(x, y, r + 1.75, 0, Math.PI * 2);
  ctx.fillStyle = halo;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (ring) {
    ctx.strokeStyle = fill;
    ctx.lineWidth = 1.75;
    ctx.stroke();
  } else {
    ctx.fillStyle = fill;
    ctx.fill();
  }
}

function strokeFn(
  ctx: CanvasRenderingContext2D,
  g: (a: number) => number,
  from: number,
  to: number,
  x: Scale,
  y: Scale,
  n = 320,
) {
  ctx.beginPath();
  let pen = false;
  const [ylo, yhi] = y.range[0] > y.range[1] ? [y.range[1], y.range[0]] : y.range;
  for (let i = 0; i <= n; i++) {
    const a = from + ((to - from) * i) / n;
    const val = g(a);
    if (!Number.isFinite(val)) {
      pen = false;
      continue;
    }
    // Clamp far-off values so long segments still leave the frame at the right angle.
    const py = Math.max(ylo - 2000, Math.min(yhi + 2000, y(val)));
    if (pen) ctx.lineTo(x(a), py);
    else ctx.moveTo(x(a), py);
    pen = true;
  }
  ctx.stroke();
}

function clip(ctx: CanvasRenderingContext2D, f: Frame) {
  ctx.beginPath();
  ctx.rect(f.left, f.top, f.right - f.left, f.bottom - f.top);
  ctx.clip();
}

/** Where the line a ↦ y0 + s·a leaves the frame (for its label). */
function lineExit(y0: number, s: number, x: Scale, y: Scale, f: Frame): [number, number] {
  const aRight = x.domain[1];
  const yRight = y(y0 + s * aRight);
  if (yRight <= f.bottom - 4 && yRight >= f.top + 4) return [f.right - 4, yRight];
  // Leaves through the bottom (descending lines): α where it hits the lower edge.
  const target = y.invert(yRight > f.bottom ? f.bottom - 4 : f.top + 4);
  const a = (target - y0) / s;
  return [x(a), y(target)];
}

/**
 * A label for the line a ↦ y0 + s·a that does not sit on the line: right-aligned above it when
 * the line leaves through the right edge (raised over the whole label width), or set just right
 * of the point where a descending line leaves through the bottom (the line is out of the frame
 * there). Otherwise, with `fallback`, right-aligned just above the exit point (the label may
 * then touch the line); without it, null.
 */
function lineLabel(
  y0: number,
  s: number,
  w: number,
  x: Scale,
  y: Scale,
  f: Frame,
  fallback = true,
): { x: number; y: number; align: 'left' | 'right' } | null {
  const [ex, ey] = lineExit(y0, s, x, y, f);
  if (ex >= f.right - 5) {
    // The line's highest point under the label is at one of the label's ends.
    const yl = y(y0 + s * x.invert(ex - w));
    const top = Math.min(ey, yl) - 9;
    return { x: ex, y: Math.max(f.top + 10, top), align: 'right' };
  }
  if (ey >= f.bottom - 5) {
    if (ex + 6 + w <= f.right - 2) return { x: ex + 6, y: f.bottom - 22, align: 'left' };
    if (fallback && ex - w > f.left + 40) return { x: ex - 4, y: ey - 10, align: 'right' };
    return null;
  }
  return fallback && ex - w > f.left + 40 ? { x: ex - 4, y: ey + 12, align: 'right' } : null;
}

/** Do two labels (anchors and widths) overlap? */
function overlaps(
  a: { x: number; y: number; align: 'left' | 'right' },
  aw: number,
  b: { x: number; y: number; align: 'left' | 'right' } | null,
  bw: number,
) {
  if (!b) return false;
  const span = (q: { x: number; align: 'left' | 'right' }, w: number) =>
    q.align === 'left' ? [q.x, q.x + w] : [q.x - w, q.x];
  const [a0, a1] = span(a, aw);
  const [b0, b1] = span(b, bw);
  return Math.abs(a.y - b.y) < 15 && a0 < b1 + 6 && b0 < a1 + 6;
}

const ALPHA: MathRun[] = [v('α')];
const PHI_NAME: MathRun[] = [v('φ'), m('('), v('α'), m(')')];
/** The prime of φ′ as KaTeX sets it: a raised script glyph, not U+2032 on the baseline. */
const P = sup('′');
const DPHI_NAME: MathRun[] = [v('φ'), P, m('('), v('α'), m(')')];
const PHI0_RUNS: MathRun[] = [v('φ'), m('(0)')];
const ARMIJO_RUNS: MathRun[] = [v('φ'), m('(0) + '), v('c'), sub('1'), v('αφ'), P, m('(0)')];
const TANGENT_RUNS: MathRun[] = [v('φ'), m('(0) + '), v('αφ'), P, m('(0)')];
const G_UP: MathRun[] = [v('φ'), m('(0) + '), v('cαφ'), P, m('(0)')];
const G_LOW: MathRun[] = [v('φ'), m('(0) + (1 − '), v('c'), m(')'), v('αφ'), P, m('(0)')];
const STRONG_BAND: MathRun[] = [
  m('|'),
  v('φ'),
  P,
  m('| ≤ '),
  v('c'),
  sub('2'),
  m('|'),
  v('φ'),
  P,
  m('(0)|'),
];
/** The exact step's confirmation test when f rose (Python `_EXACT_C2` = 0.1). */
const EXACT_BAND: MathRun[] = [m('|'), v('φ'), P, m('| ≤ 0.1|'), v('φ'), P, m('(0)|')];
const WEAK_BAND: MathRun[] = [v('φ'), P, m(' ≥ '), v('c'), sub('2'), v('φ'), P, m('(0)')];

/** Which step's geometry to show: the latest trial, or the one arriving during a hop. */
function playState(run: PhiRun, ease: boolean) {
  const last = run.trace.length - 1;
  const kf = Math.min(last, Math.floor(run.t + 1e-9));
  const frac = Math.min(1, Math.max(0, run.t - kf));
  const u = kf < last && frac > 0 ? (ease ? easeInOut(frac) : frac) : 0;
  return { kf, u, g: u > 0 ? kf + 1 : kf, fade: u > 0 ? Math.min(1, u * 2.5) : 1 };
}

export function PhiPanel(props: PhiPanelProps) {
  const {
    line,
    samples,
    phi0,
    dphi0,
    hi,
    mode,
    runs,
    focusId,
    alphaStar,
    ease,
    probe,
    onProbe,
    onPickAlpha,
    onFocus,
    emphasis = 'phi',
    ariaLabel,
  } = props;
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [focused, setFocused] = useState(false);
  const focus = runs.find((r) => r.id === focusId) ?? null;

  const trialPhis = useMemo(
    () =>
      runs
        .flatMap((r) => r.trace.slice(1).map((s) => trialOf(s)))
        .filter((t) => t.alpha <= hi)
        .map((t) => t.phi),
    [runs, hi],
  );
  // Without a descent direction there is no well to center on: show the whole range.
  const rangeMode: WindowMode = dphi0 < 0 ? mode : 'all';
  const yPhi = useMemo(
    () => phiRange(samples, phi0, rangeMode, trialPhis),
    [samples, phi0, rangeMode, trialPhis],
  );
  // The curvature constant of the focused method: c₂, or 0.1 for the exact step's rise test.
  const focusC2 = !focus
    ? null
    : focus.kind === 'exact_quadratic'
      ? EXACT_C2
      : trialOf(focus.trace[0]).c2;
  const yD = useMemo(
    () =>
      samples.dphi
        ? slopeRange(samples.dphi, dphi0, focusC2)
        : ([dphi0, -dphi0] as [number, number]),
    [samples, dphi0, focusC2],
  );

  const scales = (w: number, h: number) => {
    const L = layout(w, h, runs.length, emphasis);
    const x = linearScale([-0.025 * hi, hi], [L.phi.left, L.phi.right]);
    const y = linearScale(yPhi, [L.phi.bottom, L.phi.top]);
    const yd = L.slope ? linearScale(yD, [L.slope.bottom, L.slope.top]) : null;
    return { L, x, y, yd };
  };

  const { canvasRef } = useCanvas((ctx, s) => {
    const c = colors;
    const { L, x, y, yd } = scales(s.width, s.height);
    const hair = 1 / s.dpr;
    const fcol = (slot: number) => c.series[slot % c.series.length];
    const xTicks = linearTicks(0, hi, Math.max(2, Math.floor((L.phi.right - L.phi.left) / 90)));
    const tickStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : hi;
    const frames = [L.phi, ...(L.slope ? [L.slope] : [])];
    const fs = focus ? playState(focus, ease) : null;
    const ft: Trial | null =
      focus && fs ? trialOf(focus.trace[Math.min(fs.g, focus.trace.length - 1)]) : null;
    const fc = focus ? fcol(focus.slot) : c.text;
    const head = trialOf(runs[0]?.trace[0] ?? focus?.trace[0] ?? ({ k: 0, info: {} } as Step));
    const c1 = focus ? trialOf(focus.trace[0]).c1 : head.c1;
    // A short φ frame (the phone's φ′ view) keeps the curves and drops the text labels.
    const compact = L.phi.bottom - L.phi.top < 100;

    // ── Grids and frames ──
    ctx.save();
    ctx.strokeStyle = c.grid;
    ctx.lineWidth = hair;
    ctx.beginPath();
    for (const f of [...frames, ...L.lanes])
      for (const a of xTicks) {
        const px = crisp(x(a), s.dpr);
        ctx.moveTo(px, f.top);
        ctx.lineTo(px, f.bottom);
      }
    const yTicks = linearTicks(
      yPhi[0],
      yPhi[1],
      Math.max(2, Math.floor((L.phi.bottom - L.phi.top) / 46)),
    );
    for (const val of yTicks) {
      const py = crisp(y(val), s.dpr);
      ctx.moveTo(L.phi.left, py);
      ctx.lineTo(L.phi.right, py);
    }
    ctx.stroke();
    ctx.strokeStyle = c.axis;
    ctx.beginPath();
    for (const f of frames) {
      ctx.moveTo(crisp(f.left, s.dpr), f.top);
      ctx.lineTo(crisp(f.left, s.dpr), crisp(f.bottom, s.dpr));
      ctx.lineTo(f.right, crisp(f.bottom, s.dpr));
    }
    ctx.stroke();
    // Tick labels.
    ctx.font = tickFont(c);
    ctx.fillStyle = c.tick;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
    for (const val of yTicks) {
      const py = y(val);
      if (py < L.phi.top + 16 || py > L.phi.bottom - 4) continue;
      ctx.fillText(fmtTick(val, yStep), L.phi.left - 7, py);
    }
    if (L.slope && yd) {
      const dt = linearTicks(yD[0], yD[1], 3);
      const dStep = dt.length > 1 ? dt[1] - dt[0] : 1;
      for (const val of dt) {
        const py = yd(val);
        if (py < L.slope.top + 14 || py > L.slope.bottom - 4) continue;
        ctx.fillText(fmtTick(val, dStep), L.slope.left - 7, py);
      }
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const nameW = measureMath(ctx, ALPHA, 14) + 16;
    for (const a of xTicks) {
      const px = x(a);
      if (px > L.phi.right - nameW) continue;
      ctx.fillText(fmtTick(a, tickStep), px, L.axisY);
    }
    drawMath(ctx, ALPHA, L.phi.right - 2, L.axisY + 9, {
      size: 14,
      align: 'right',
      color: c.text2,
      baseline: 'middle',
    });
    ctx.restore();

    // ── φ plot ──
    ctx.save();
    clip(ctx, L.phi);
    // Acceptable intervals of the focused method (vertical bands).
    if (focus) {
      ctx.fillStyle = fc;
      ctx.globalAlpha = c.mode === 'dark' ? 0.16 : 0.1;
      for (const [a, b] of focus.intervals)
        ctx.fillRect(x(a), L.phi.top, x(b) - x(a), L.phi.bottom - L.phi.top);
      ctx.globalAlpha = 1;
    }
    // Goldstein cone.
    if (focus?.kind === 'goldstein') {
      const cc = c1;
      ctx.fillStyle = fc;
      ctx.globalAlpha = c.mode === 'dark' ? 0.14 : 0.09;
      ctx.beginPath();
      ctx.moveTo(x(0), y(phi0));
      ctx.lineTo(x(hi), y(phi0 + cc * hi * dphi0));
      ctx.lineTo(x(hi), y(phi0 + (1 - cc) * hi * dphi0));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    // The tangent φ(0) + αφ'(0): the decrease the slope promises.
    ctx.strokeStyle = c.text3;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    strokeFn(ctx, (a) => phi0 + a * dphi0, 0, hi, x, y, 2);
    ctx.setLineDash([]);
    // Sufficient-decrease line(s) of the focused method. The exact step tests plain decrease,
    // φ(α) ≤ φ(0) (its c₁ feeds only a diagnostic Armijo report), so its line is level.
    const exact = focus?.kind === 'exact_quadratic';
    if (focus) {
      ctx.strokeStyle = fc;
      ctx.lineWidth = 1.5;
      strokeFn(ctx, (a) => phi0 + (exact ? 0 : c1 * a * dphi0), 0, hi, x, y, 2);
      if (focus.kind === 'goldstein')
        strokeFn(ctx, (a) => phi0 + (1 - c1) * a * dphi0, 0, hi, x, y, 2);
    }
    // Interpolant (strong Wolfe zoom) or quadratic model (exact step) of the focused method.
    if (focus && fs && ft) {
      ctx.globalAlpha = fs.fade;
      ctx.strokeStyle = fc;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      if (focus.kind === 'strong_wolfe') {
        const ip = zoomInterpolant(focus.trace, fs.g);
        if (ip) {
          const [a, b] = ip.bracket;
          strokeFn(ctx, ip.value, a, b, x, y, 120);
          ctx.setLineDash([]);
          // Safeguard interval: hairline ticks on the curve's floor.
          ctx.lineWidth = 1;
          const yb = L.phi.bottom - 8;
          ctx.beginPath();
          ctx.moveTo(x(ip.safe[0]), yb);
          ctx.lineTo(x(ip.safe[1]), yb);
          ctx.moveTo(x(ip.safe[0]), yb - 4);
          ctx.lineTo(x(ip.safe[0]), yb + 4);
          ctx.moveTo(x(ip.safe[1]), yb - 4);
          ctx.lineTo(x(ip.safe[1]), yb + 4);
          ctx.stroke();
          if (ip.tStar !== null && ip.how.endsWith('clamped')) {
            // The interpolant's own minimizer, before the safeguard moved it.
            const ty = ip.value(ip.tStar);
            const px = x(ip.tStar),
              py = y(ty);
            ctx.strokeStyle = fc;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px - 4, py - 4);
            ctx.lineTo(px + 4, py + 4);
            ctx.moveTo(px - 4, py + 4);
            ctx.lineTo(px + 4, py - 4);
            ctx.stroke();
          }
        }
      } else if (focus.kind === 'exact_quadratic') {
        const q = quadraticModel(ft);
        if (q) strokeFn(ctx, q, 0, hi, x, y, 200);
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    // φ itself.
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.9;
    ctx.lineJoin = 'round';
    strokeFn(
      ctx,
      (a) => line.phi(a),
      0,
      hi,
      x,
      y,
      Math.max(240, Math.round(L.phi.right - L.phi.left)),
    );
    // The first local minimizer α⋆ of φ: a + cross.
    if (alphaStar && alphaStar.alpha <= hi) {
      const px = x(alphaStar.alpha),
        py = y(alphaStar.phi);
      for (const [w, col] of [
        [3.5, c.halo],
        [1.5, c.text],
      ] as const) {
        ctx.strokeStyle = col;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(px - 5.5, py);
        ctx.lineTo(px + 5.5, py);
        ctx.moveTo(px, py - 5.5);
        ctx.lineTo(px, py + 5.5);
        ctx.stroke();
      }
    }
    // Probe.
    if (probe !== null && probe >= 0 && probe <= hi) {
      ctx.strokeStyle = c.crosshair;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(crisp(x(probe), s.dpr), L.phi.top);
      ctx.lineTo(crisp(x(probe), s.dpr), L.phi.bottom);
      ctx.stroke();
    }
    // Trials of every method (focus last, on top).
    const order = [...runs].sort((a, b) => Number(a.id === focusId) - Number(b.id === focusId));
    const yTop = L.phi.top + 6;
    const toY = (val: number) => Math.max(yTop, Math.min(L.phi.bottom - 2, y(val)));
    for (const r of order) {
      const st = playState(r, ease);
      const col = fcol(r.slot);
      const isF = r.id === focusId;
      for (let k = 1; k <= st.kf; k++) {
        const t = trialOf(r.trace[k]);
        if (!(t.alpha <= hi) || !Number.isFinite(t.phi)) continue;
        const cur = k === st.kf && st.u === 0;
        const above = y(t.phi) < yTop;
        ctx.globalAlpha = cur || isF ? 1 : 0.75;
        if (above) {
          // φ above the view: a chevron at the top edge.
          const px = x(t.alpha);
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.moveTo(px, yTop - 4);
          ctx.lineTo(px - 4, yTop + 2);
          ctx.lineTo(px + 4, yTop + 2);
          ctx.closePath();
          ctx.fill();
        } else {
          const done = t.accepted && k === r.trace.length - 1;
          dot(ctx, x(t.alpha), toY(t.phi), cur ? 4.25 : done ? 4 : 2.75, col, c.halo, false);
          if (done) dot(ctx, x(t.alpha), toY(t.phi), 6.5, col, 'transparent', true);
        }
        ctx.globalAlpha = 1;
      }
      // The hop under way: the marker slides along φ to the next trial.
      if (st.u > 0) {
        const a0 = st.kf === 0 ? 0 : trialOf(r.trace[st.kf]).alpha;
        const a1 = trialOf(r.trace[st.kf + 1]).alpha;
        const a = a0 + (a1 - a0) * st.u;
        if (a <= hi) {
          const val = line.phi(a);
          if (Number.isFinite(val)) dot(ctx, x(a), toY(val), 4.25, col, c.halo);
        }
      }
    }
    // The focused trial: slope segment and label.
    if (focus && fs && ft && fs.g >= 1 && fs.u === 0) {
      if (ft.alpha <= hi && Number.isFinite(ft.phi)) {
        const px = x(ft.alpha),
          py = toY(ft.phi);
        if (ft.dphi !== null && Number.isFinite(ft.dphi) && y(ft.phi) >= yTop) {
          const half = hi * 0.07;
          ctx.strokeStyle = fc;
          ctx.lineWidth = 1.5;
          strokeFn(
            ctx,
            (a) => ft.phi + ft.dphi! * (a - ft.alpha),
            ft.alpha - half,
            ft.alpha + half,
            x,
            y,
            2,
          );
        }
        // Label above the trial (the curve is lowest near accepted steps).
        if (!compact)
          mathLabel(
            ctx,
            [v('α'), sub(String(ft.k)), m(` = ${num(ft.alpha, 4)}`)],
            Math.min(L.phi.right - 50, Math.max(L.phi.left + 50, px)),
            Math.max(L.phi.top + 12, py - 22),
            c,
            'center',
            c.text,
          );
      }
    }
    // φ(0): a hollow ring at the start.
    dot(ctx, x(0), y(phi0), 4, c.text, c.halo, true);
    ctx.restore();

    // Labels of the lines (outside the clip, with halos), and the plot name.
    mathLabel(ctx, PHI_NAME, L.phi.left + 8, L.phi.top + 10, c, 'left', c.text2, 13.5);
    if (!compact) {
      // Line labels beside the point where each line leaves the frame, clear of the line; the
      // tangent's is dropped when it would sit on another label.
      const LS = 11.5;
      const aRuns = exact ? PHI0_RUNS : focus?.kind === 'goldstein' ? G_UP : ARMIJO_RUNS;
      const aw = measureMath(ctx, aRuns, LS);
      const aPos = focus ? lineLabel(phi0, exact ? 0 : c1 * dphi0, aw, x, y, L.phi) : null;
      if (aPos) mathLabel(ctx, aRuns, aPos.x, aPos.y, c, aPos.align, c.text2, LS);
      const gw = measureMath(ctx, G_LOW, LS);
      const gPos =
        focus?.kind === 'goldstein' ? lineLabel(phi0, (1 - c1) * dphi0, gw, x, y, L.phi) : null;
      if (gPos && !overlaps(gPos, gw, aPos, aw))
        mathLabel(ctx, G_LOW, gPos.x, gPos.y, c, gPos.align, c.text2, LS);
      const tw = measureMath(ctx, TANGENT_RUNS, LS);
      const tPos = lineLabel(phi0, dphi0, tw, x, y, L.phi, false);
      if (tPos && !overlaps(tPos, tw, aPos, aw) && !overlaps(tPos, tw, gPos, gw))
        mathLabel(ctx, TANGENT_RUNS, tPos.x, tPos.y, c, tPos.align, c.text3, LS);
    }
    if (!compact && alphaStar && alphaStar.alpha <= hi) {
      const px = x(alphaStar.alpha),
        py = y(alphaStar.phi);
      // Below the cross; below and to the right when the safeguard bar (at the floor of the
      // frame) is under it; beside it when neither has room. φ is above the cross near its
      // minimizer, so below is clear of the curve.
      const STAR: MathRun[] = [v('α'), sup('⋆')];
      const sw = measureMath(ctx, STAR, 12.5);
      const ip = focus?.kind === 'strong_wolfe' && fs ? zoomInterpolant(focus.trace, fs.g) : null;
      const bar = ip ? [x(ip.safe[0]) - 5, x(ip.safe[1]) + 5] : null;
      const barY = L.phi.bottom - 8;
      const hits = (x0: number, x1: number, yMid: number) =>
        !!bar && yMid + 7 > barY - 5 && x0 < bar[1] && bar[0] < x1;
      if (py > L.phi.top && py < L.phi.bottom) {
        if (py + 22 < L.phi.bottom - 1 && !hits(px - sw / 2, px + sw / 2, py + 15))
          mathLabel(ctx, STAR, px, py + 15, c, 'center', c.text2);
        else if (py + 19 < L.phi.bottom - 1 && !hits(px + 6, px + 6 + sw, py + 12))
          mathLabel(ctx, STAR, px + 6, py + 12, c, 'left', c.text2);
        else if (!hits(px + 9, px + 9 + sw, py + 2))
          mathLabel(ctx, STAR, px + 9, py + 2, c, 'left', c.text2);
        // (No room at all: the cross stands alone; the plane names α⋆.)
      }
    }
    // Focused trial off to the right: say where it went.
    if (!compact && focus && fs && ft && fs.g >= 1 && fs.u === 0 && ft.alpha > hi) {
      mathLabel(
        ctx,
        [v('α'), sub(String(ft.k)), m(` = ${num(ft.alpha, 4)} →`)],
        L.phi.right - 6,
        L.phi.top + 28,
        c,
        'right',
        c.text,
      );
    }

    // ── φ' strip ──
    const twoSided = focus?.kind === 'strong_wolfe' || focus?.kind === 'exact_quadratic';
    if (L.slope && yd && samples.dphi) {
      const S = L.slope;
      ctx.save();
      clip(ctx, S);
      if (focus && focusC2 !== null) {
        ctx.fillStyle = fc;
        ctx.globalAlpha = c.mode === 'dark' ? 0.17 : 0.11;
        const band = twoSided
          ? [focusC2 * dphi0, -focusC2 * dphi0]
          : [focusC2 * dphi0, yD[1] + 1e9];
        const top = yd(Math.min(band[1], yD[1] * 10 + 1));
        ctx.fillRect(
          S.left,
          Math.max(S.top, top),
          S.right - S.left,
          Math.min(S.bottom, yd(band[0])) - Math.max(S.top, top),
        );
        ctx.globalAlpha = 1;
        ctx.strokeStyle = fc;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const b of twoSided ? band : [band[0]]) {
          const py = crisp(yd(b), s.dpr);
          ctx.moveTo(S.left, py);
          ctx.lineTo(S.right, py);
        }
        ctx.stroke();
      }
      // Zero line and the reference slope φ'(0).
      ctx.strokeStyle = c.axis;
      ctx.lineWidth = hair;
      ctx.beginPath();
      ctx.moveTo(S.left, crisp(yd(0), s.dpr));
      ctx.lineTo(S.right, crisp(yd(0), s.dpr));
      ctx.stroke();
      ctx.strokeStyle = c.text3;
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(S.left, crisp(yd(dphi0), s.dpr));
      ctx.lineTo(S.right, crisp(yd(dphi0), s.dpr));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < samples.alpha.length; i++) {
        const val = samples.dphi[i];
        if (!Number.isFinite(val)) {
          pen = false;
          continue;
        }
        const py = Math.max(S.top - 50, Math.min(S.bottom + 50, yd(val)));
        if (pen) ctx.lineTo(x(samples.alpha[i]), py);
        else ctx.moveTo(x(samples.alpha[i]), py);
        pen = true;
      }
      ctx.stroke();
      if (probe !== null && probe >= 0 && probe <= hi) {
        ctx.strokeStyle = c.crosshair;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(crisp(x(probe), s.dpr), S.top);
        ctx.lineTo(crisp(x(probe), s.dpr), S.bottom);
        ctx.stroke();
      }
      for (const r of order) {
        const st = playState(r, ease);
        for (let k = 1; k <= st.kf; k++) {
          const t = trialOf(r.trace[k]);
          if (t.dphi === null || !(t.alpha <= hi)) continue;
          const py = Math.max(S.top + 3, Math.min(S.bottom - 3, yd(t.dphi)));
          dot(ctx, x(t.alpha), py, k === st.kf && st.u === 0 ? 3.75 : 2.5, fcol(r.slot), c.halo);
        }
      }
      dot(ctx, x(0), yd(dphi0), 3.5, c.text, c.halo, true);
      ctx.restore();
      mathLabel(ctx, DPHI_NAME, S.left + 8, S.top + 10, c, 'left', c.text2, 13.5);
      if (focus && focusC2 !== null) {
        const runsLabel =
          focus.kind === 'exact_quadratic'
            ? EXACT_BAND
            : focus.kind === 'strong_wolfe'
              ? STRONG_BAND
              : WEAK_BAND;
        const py = yd(twoSided ? -focusC2 * dphi0 : focusC2 * dphi0);
        mathLabel(
          ctx,
          runsLabel,
          S.right - 6,
          Math.max(S.top + 10, Math.min(S.bottom - 10, py - 9)),
          c,
          'right',
          c.text2,
          11.5,
        );
      }
    }

    // ── Lanes: acceptable sets, trials in order, the current interval ──
    runs.forEach((r, i) => {
      const f = L.lanes[i];
      if (!f) return;
      const col = fcol(r.slot);
      const isF = r.id === focusId;
      const mid = (f.top + f.bottom) / 2;
      ctx.save();
      // Lane name (left margin) or inside the lane on phones.
      const nm = shortName(r.name);
      if (!L.narrow) {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(8 + 4, mid, 4, 0, Math.PI * 2);
        ctx.fill();
        word(ctx, nm, f.left - 8, mid, c, {
          align: 'right',
          color: isF ? c.text : c.text2,
          weight: isF ? 600 : 500,
          size: LABEL_SIZE,
        });
      }
      const st = playState(r, ease);
      // Trials past the window are counted at the lane's right end ("+2 →").
      let beyond = 0;
      for (let k = 1; k <= st.kf; k++) if (!(trialOf(r.trace[k]).alpha <= hi)) beyond++;
      const countLabel = `+${beyond} →`;
      ctx.font = labelFont(c);
      const countW = beyond > 0 ? ctx.measureText(countLabel).width + 8 : 0;
      clip(ctx, { ...f, top: L.narrow ? f.top : f.top - 8 });
      ctx.fillStyle = c.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.025)';
      ctx.fillRect(f.left, f.top, f.right - f.left, f.bottom - f.top);
      ctx.fillStyle = col;
      ctx.globalAlpha = isF ? 0.32 : 0.22;
      for (const [a, b] of r.intervals)
        ctx.fillRect(x(a), f.top + 3, Math.max(1, x(b) - x(a)), f.bottom - f.top - 6);
      ctx.globalAlpha = 1;
      if (isF) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        // Inset by 1.5 px: the outline stays inside the lane, clear of the band above it.
        ctx.strokeRect(f.left + 1.5, f.top + 1.5, f.right - f.left - 3, f.bottom - f.top - 3);
      }
      // The interval the (current or arriving) trial was chosen from.
      const gt = trialOf(r.trace[Math.min(st.g, r.trace.length - 1)]);
      if (gt.interval && st.g >= 1) {
        const [lo, up] = gt.interval;
        const xl = x(lo);
        const xr = Number.isFinite(up) && up <= hi ? x(up) : f.right - 1;
        ctx.globalAlpha = st.fade;
        ctx.strokeStyle = c.text;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(xl + 3, f.bottom - 2);
        ctx.lineTo(xl, f.bottom - 2);
        ctx.lineTo(xl, f.top + 2);
        ctx.lineTo(xl + 3, f.top + 2);
        if (Number.isFinite(up) && up <= hi) {
          ctx.moveTo(xr - 3, f.bottom - 2);
          ctx.lineTo(xr, f.bottom - 2);
          ctx.lineTo(xr, f.top + 2);
          ctx.lineTo(xr - 3, f.top + 2);
        }
        ctx.moveTo(xl, f.bottom - 2);
        ctx.lineTo(xr, f.bottom - 2);
        ctx.stroke();
        if (!(Number.isFinite(up) && up <= hi)) {
          ctx.beginPath();
          ctx.moveTo(xr - 5, f.bottom - 5);
          ctx.lineTo(xr, f.bottom - 2);
          ctx.lineTo(xr - 5, f.bottom + 1);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      // Hops between consecutive trials (arcs above the lane's midline), clipped short of the
      // count of the trials past the window.
      ctx.save();
      if (countW > 0)
        clip(ctx, { ...f, top: L.narrow ? f.top : f.top - 8, right: f.right - countW });
      const ax = (a: number) => Math.min(x(Math.min(a, hi * 1.5)), f.right + 30);
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.25;
      let prevA = 0;
      for (let k = 1; k <= st.kf + (st.u > 0 ? 1 : 0); k++) {
        const a = trialOf(r.trace[k]).alpha;
        const partial = k === st.kf + 1 ? st.u : 1;
        const x0 = ax(prevA),
          x1 = ax(a);
        const xm = x0 + (x1 - x0) * partial;
        const hgt = Math.min(LANE_H * 0.9, 4 + Math.abs(x1 - x0) * 0.25);
        ctx.globalAlpha = k >= st.kf ? 0.95 : 0.45;
        ctx.beginPath();
        const steps = 18;
        for (let j = 0; j <= steps; j++) {
          const sj = (j / steps) * partial;
          const px = x0 + (x1 - x0) * sj;
          const py = mid - hgt * 4 * sj * (1 - sj) * 0.5 - 1;
          if (j) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
        }
        ctx.stroke();
        if (partial < 1)
          dot(ctx, xm, mid - hgt * 4 * partial * (1 - partial) * 0.5 - 1, 3.25, col, c.halo);
        prevA = a;
      }
      ctx.restore();
      ctx.globalAlpha = 1;
      // Trial marks.
      for (let k = 1; k <= st.kf; k++) {
        const t = trialOf(r.trace[k]);
        if (!(t.alpha <= hi)) continue;
        const cur = k === st.kf && st.u === 0;
        const done = t.accepted && k === r.trace.length - 1;
        dot(ctx, x(t.alpha), mid, cur ? 3.75 : 2.5, col, c.halo, false);
        if (done) dot(ctx, x(t.alpha), mid, 6, col, 'transparent', true);
      }
      ctx.restore();
      if (beyond > 0)
        word(ctx, countLabel, f.right - 3, mid, c, {
          align: 'right',
          size: LABEL_SIZE,
          color: c.text2,
        });
      if (L.narrow) {
        // The name band above the lane: swatch dot + name, never crossed by the trials.
        const by = f.top - NAME_BAND / 2;
        ctx.save();
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(f.left + 4, by, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        word(ctx, nm, f.left + 12, by, c, {
          size: LABEL_SIZE,
          color: isF ? c.text : c.text2,
          weight: isF ? 600 : 500,
        });
      }
    });
  });

  // ── Pointer and keyboard ──
  const alphaAt = (clientX: number, clientY: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const { L, x } = scales(r.width, r.height);
    const px = clientX - r.left,
      py = clientY - r.top;
    if (px < L.phi.left || px > L.phi.right) return null;
    const band = L.narrow ? NAME_BAND : 2;
    const lane = L.lanes.findIndex((f) => py >= f.top - band && py <= f.bottom + 2);
    return { alpha: Math.max(0, x.invert(px)), lane, inPlot: py <= (L.slope ?? L.phi).bottom };
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'touch') return;
    const hit = alphaAt(e.clientX, e.clientY);
    onProbe(hit && (hit.inPlot || hit.lane >= 0) ? hit.alpha : null);
  };
  const onClick = (e: PointerEvent<HTMLDivElement>) => {
    const hit = alphaAt(e.clientX, e.clientY);
    if (!hit) return;
    if (hit.lane >= 0 && runs[hit.lane]) onFocus?.(runs[hit.lane].id);
    else if (hit.inPlot && hit.alpha > 0) onPickAlpha?.(Number(hit.alpha.toPrecision(3)));
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = hi / (e.shiftKey ? 10 : 100);
    const cur =
      probe ?? (focus ? Math.min(hi, trialOf(focus.trace[focus.trace.length - 1]).alpha) : hi / 2);
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      e.stopPropagation();
      onProbe(Math.max(0, Math.min(hi, cur + (e.key === 'ArrowRight' ? step : -step))));
    } else if (e.key === 'Enter' && probe !== null && probe > 0) {
      e.preventDefault();
      onPickAlpha?.(Number(probe.toPrecision(3)));
    } else if (e.key === 'Escape') onProbe(null);
  };

  const readout =
    probe !== null
      ? `α = ${num(probe, 4)}, φ(α) = ${num(line.phi(probe), 5)}, φ′(α) = ${num(line.dphi(probe), 4)}`
      : '';
  const pick =
    onPickAlpha && focus ? `Click or press Enter to set α₀ of ${shortName(focus.name)}.` : '';

  return (
    <div
      ref={box}
      className={styles.phiBox}
      role="group"
      aria-label={`φ panel. Arrow keys move a probe along α. ${pick}`}
      tabIndex={0}
      data-own-keys=""
      data-plot-focus=""
      onPointerMove={onMove}
      onPointerLeave={() => onProbe(null)}
      onClick={onClick}
      onKeyDown={onKey}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
    >
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
      {probe !== null && (
        <div className={styles.readout} aria-hidden={!focused}>
          {readout}
        </div>
      )}
      <span className="visually-hidden" aria-live="polite">
        {focused ? readout : ''}
      </span>
    </div>
  );
}
