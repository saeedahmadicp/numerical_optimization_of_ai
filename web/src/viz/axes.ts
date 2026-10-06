/**
 * Axes renderer for canvas charts: hairline grid (one device pixel), recessive axis lines and
 * tick labels in a monospaced (tabular) face.
 */
import type { ChartColors } from '../ui/colors';
import { powTen, tick as fmtTick } from '../core/format';
import { crisp, type Scale } from './scales';
import { drawMath, m, measureMath, pow10Runs, v as mathVar, type MathRun } from './mathText';

export interface Frame {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export const frameWidth = (f: Frame) => f.right - f.left;
export const frameHeight = (f: Frame) => f.bottom - f.top;

export interface AxesOptions {
  x: Scale;
  y: Scale;
  frame: Frame;
  colors: ChartColors;
  dpr: number;
  xTicks?: number;
  yTicks?: number;
  grid?: boolean;
  /** Draw axis baselines on the frame's left/bottom edges. */
  baseline?: boolean;
  /** Axis names. A single letter is a variable and is set in KaTeX italic (`x`, `k`). */
  xLabel?: string;
  yLabel?: string;
  /** Axis names as typeset math runs (take precedence over `xLabel`/`yLabel`). */
  xName?: readonly MathRun[];
  yName?: readonly MathRun[];
  /** Integer-only x ticks (iteration counts). */
  xInteger?: boolean;
  /** Draw labels inside the frame (for full-bleed plots like contours). */
  inset?: boolean;
  /** Outline tick labels and axis names with `colors.halo` (default: when `inset`, over fills). */
  halo?: boolean;
  fontSize?: number;
}

/** Tick labels: 11.5 px tabular mono (brand.md §9: the smallest chart type is ≥ 11 px). */
export const TICK_SIZE = 11.5;
/** Labels drawn in the UI face on charts (legends, annotations): 12 px. */
export const LABEL_SIZE = 12;

export function tickFont(colors: Pick<ChartColors, 'fontMono'>, size = TICK_SIZE) {
  return `450 ${size}px ${colors.fontMono || 'ui-monospace, monospace'}`;
}

export function labelFont(colors: Pick<ChartColors, 'fontSans'>, size = LABEL_SIZE) {
  return `500 ${size}px ${colors.fontSans || 'system-ui, sans-serif'}`;
}

/**
 * The power of ten a linear axis factors out of its tick labels, or 0. Ticks such as
 * −6×10⁻⁵ … 6×10⁻⁵ read badly as Unicode superscripts in a monospaced face, so the axis labels
 * them −6 … 6 and states ×10⁻⁵ once, beside its name (as a plotting library does).
 */
export function axisExponent(ticks: readonly number[]): number {
  const big = Math.max(0, ...ticks.map((t) => Math.abs(t)).filter(Number.isFinite));
  if (big === 0) return 0;
  if (big >= 1e5 || big < 1e-3) return Math.floor(Math.log10(big) + 1e-9);
  return 0;
}

function formatTick(scale: Scale, v: number, count: number, exp = 0, integer = false) {
  if (scale.kind === 'log') return powTen(Math.round(Math.log10(v)));
  // Integer axes (iteration counts) print 1, never 1.0, even when the tick step is fractional.
  const step = integer ? Math.max(1, scale.tickStep(count)) : scale.tickStep(count);
  if (exp !== 0) return fmtTick(v / 10 ** exp, step / 10 ** exp);
  return fmtTick(v, step);
}

/** `name ×10ⁿ` (or `×10ⁿ` alone) when a linear axis factors out a power of ten. */
function withFactor(name: MathRun[] | undefined, exp: number): MathRun[] | undefined {
  if (exp === 0) return name;
  const factor: MathRun[] = [m('×'), ...pow10Runs(exp)];
  return name ? [...name, m('  '), ...factor] : factor;
}

/** Draw grid + ticks. Returns the tick values used (for callers that annotate them). */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  o: AxesOptions,
): { xTicks: number[]; yTicks: number[] } {
  const { x, y, frame, colors, dpr, grid = true, baseline = true, inset = false } = o;
  const halo = o.halo ?? inset;
  /** Text with an optional 3-px halo so it stays legible over fills and iso-lines. */
  const text = (s: string, tx: number, ty: number) => {
    if (halo) {
      const fill = ctx.fillStyle;
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.strokeText(s, tx, ty);
      ctx.fillStyle = fill;
    }
    ctx.fillText(s, tx, ty);
  };
  // Inset labels keep clear of the edges (8 px) and of the axis names.
  const edge = inset ? 8 : 2;
  const xCount = o.xTicks ?? Math.max(2, Math.floor(frameWidth(frame) / 90));
  const yCount = o.yTicks ?? Math.max(2, Math.floor(frameHeight(frame) / 48));
  let xt = x.ticks(xCount).filter((v) => x(v) >= frame.left - 0.5 && x(v) <= frame.right + 0.5);
  if (o.xInteger) xt = xt.filter((v) => Number.isInteger(v));
  const yt = y.ticks(yCount).filter((v) => y(v) >= frame.top - 0.5 && y(v) <= frame.bottom + 0.5);
  const hair = 1 / dpr;

  ctx.save();
  if (grid) {
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = hair;
    ctx.beginPath();
    for (const v of xt) {
      const px = crisp(x(v), dpr);
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
    }
    for (const v of yt) {
      const py = crisp(y(v), dpr);
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
  }
  if (baseline) {
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = hair;
    ctx.beginPath();
    const bx = crisp(frame.left, dpr),
      by = crisp(frame.bottom, dpr);
    ctx.moveTo(bx, frame.top);
    ctx.lineTo(bx, by);
    ctx.lineTo(frame.right, by);
    ctx.stroke();
  }

  const fs = Math.max(11, o.fontSize ?? TICK_SIZE);
  const xExp = x.kind === 'log' ? 0 : axisExponent(xt);
  const yExp = y.kind === 'log' ? 0 : axisExponent(yt);
  const xName = withFactor(o.xName ? [...o.xName] : nameRuns(o.xLabel), xExp);
  const yName = withFactor(o.yName ? [...o.yName] : nameRuns(o.yLabel), yExp);
  const nameSize = inset ? 14 : 13;
  const xLabelWidth = xName ? measureName(ctx, xName, nameSize, colors) + (inset ? 22 : 10) : 0;
  ctx.font = tickFont(colors, fs);
  ctx.fillStyle = colors.tick;
  // x labels
  ctx.textAlign = 'center';
  ctx.textBaseline = inset ? 'bottom' : 'top';
  let lastRight = -Infinity;
  for (const v of xt) {
    const px = x(v);
    if (x.kind === 'log') {
      // Powers of ten typeset like KaTeX: 10 upright, the exponent raised, U+2212 minus.
      const runs = pow10Runs(Math.round(Math.log10(v)));
      const w = measureMath(ctx, runs, fs + 1);
      if (px - w / 2 < lastRight + 6) continue;
      if (inset && (px - w / 2 < frame.left + edge || px + w / 2 > frame.right - edge)) continue;
      if (px + w / 2 > frame.right - xLabelWidth + 4 && xLabelWidth > 0) continue;
      drawMath(ctx, runs, px, inset ? frame.bottom - 6 : frame.bottom + fs + 6, {
        size: fs + 1,
        align: 'center',
        color: colors.tick,
        halo: halo ? colors.halo : undefined,
      });
      lastRight = px + w / 2;
      continue;
    }
    const label = formatTick(x, v, xCount, xExp, o.xInteger);
    const w = ctx.measureText(label).width;
    if (px - w / 2 < lastRight + 6) continue;
    if (inset && (px - w / 2 < frame.left + edge || px + w / 2 > frame.right - edge)) continue;
    if (px + w / 2 > frame.right - xLabelWidth + 4 && xLabelWidth > 0) continue;
    text(label, px, inset ? frame.bottom - 5 : frame.bottom + 6);
    lastRight = px + w / 2;
  }
  // y labels
  ctx.textAlign = inset ? 'left' : 'right';
  ctx.textBaseline = 'middle';
  // Inset: skip ticks that would touch the top edge, the y-axis name or the x tick row.
  const yTop = frame.top + (inset ? (yName ? 30 : 12) : 0);
  for (const v of yt) {
    const py = y(v);
    if (inset && (py < yTop || py > frame.bottom - 22)) continue;
    if (y.kind === 'log') {
      drawMath(
        ctx,
        pow10Runs(Math.round(Math.log10(v))),
        inset ? frame.left + 6 : frame.left - 7,
        py,
        {
          size: fs + 1,
          align: inset ? 'left' : 'right',
          baseline: 'middle',
          color: colors.tick,
          halo: halo ? colors.halo : undefined,
        },
      );
      continue;
    }
    text(formatTick(y, v, yCount, yExp), inset ? frame.left + 6 : frame.left - 7, py);
  }
  // Minor decade gridlines on log axes (unlabeled).
  if (y.kind === 'log' && grid) {
    const [lo, hi] = y.domain;
    ctx.strokeStyle = colors.grid;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = hair;
    ctx.beginPath();
    for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
      if (yt.includes(10 ** e)) continue;
      const py = crisp(y(10 ** e), dpr);
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Axis names: math runs (variables in KaTeX italic) or plain words in the UI face.
  const haloColor = halo ? colors.halo : undefined;
  if (xName) {
    // Sits at the right end of the tick row; colliding ticks were skipped above.
    drawName(
      ctx,
      xName,
      inset ? frame.right - 8 : frame.right,
      inset ? frame.bottom - 7 : frame.bottom + 18,
      'right',
      nameSize,
      colors,
      haloColor,
    );
  }
  if (yName) {
    drawName(
      ctx,
      yName,
      inset ? frame.left + 6 : frame.left - 4,
      inset ? frame.top + 20 : frame.top - 9,
      'left',
      nameSize,
      colors,
      haloColor,
    );
  }
  ctx.restore();
  return { xTicks: xt, yTicks: yt };
}

/** A one-letter label is a variable (KaTeX italic); a longer one is a word (Inter). */
function nameRuns(label?: string): MathRun[] | undefined {
  if (!label) return undefined;
  if (label.length === 1) return [mathVar(label)];
  return [{ t: label, style: 'sans' }];
}

/** Words (Inter) sit a little smaller than math, never below the 12-px label size. */
function nameSize(runs: readonly MathRun[], size: number) {
  return runs.every((r) => r.style === 'sans') ? Math.max(LABEL_SIZE, size - 2) : size;
}

function measureName(
  ctx: CanvasRenderingContext2D,
  runs: readonly MathRun[],
  size: number,
  colors: ChartColors,
) {
  void colors;
  return measureMath(ctx, runs, nameSize(runs, size));
}

function drawName(
  ctx: CanvasRenderingContext2D,
  runs: readonly MathRun[],
  x: number,
  y: number,
  align: 'left' | 'right',
  size: number,
  colors: ChartColors,
  halo?: string,
) {
  drawMath(ctx, runs, x, y, { size: nameSize(runs, size), align, color: colors.text2, halo });
}
