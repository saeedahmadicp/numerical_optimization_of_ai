/**
 * The stencil at the current step h, magnified: the window is x0 ± (reach + 1.6)·h and shrinks
 * with h (continuously during playback), so the stencil keeps its place while the curve
 * straightens out — until rounding turns it into a staircase of floating-point values.
 *
 * Drawn in view coordinates (model.ts): u = x − x0 on the x-axis (ticks at multiples of h) and
 * y = f(x) − f(x0) ('f') or f(x) − ℓ(x) with the tangent ℓ ('dev'), where a chord's slope is its
 * error D(h) − f′(x0). Reference marks are ink; method marks wear their slot color.
 */
import { useChartColors } from '../../ui/theme';
import { sci } from '../../core/format';
import type { Step } from '../../core/types';
import { useCanvas } from '../../viz/useCanvas';
import { tickFont } from '../../viz/axes';
import { crisp, linearScale, linearTicks } from '../../viz/scales';
import { drawMath, m as mm, sub, v as mv, type MathRun } from '../../viz/mathText';
import { easeInOut } from '../../play/timeline';
import {
  REACH,
  SECOND_DERIVATIVE,
  stencilGeometry,
  ulp,
  viewY,
  windowHalfWidth,
  type CurveMode,
} from './model';
import styles from './DifferentiationLab.module.css';

export interface StencilRun {
  id: string;
  slot: number;
  trace: readonly Step[];
  /** The run's local playhead (continuous). */
  t: number;
}

export interface StencilViewProps {
  f: (x: number) => number;
  x0: number;
  /** f′(x0) (exact): the tangent ℓ. */
  slope: number;
  /** f″(x0) (exact), for the Taylor parabola of the second-derivative method. */
  curvature: number | null;
  runs: readonly StencilRun[];
  /** Index of the run whose step sets the zoom. */
  focus: number;
  mode: CurveMode;
  /** Ease the zoom between levels (off with reduced motion / while scrubbing). */
  ease: boolean;
  ariaLabel: string;
}

// The left margin holds a signed tick such as −1.5×10⁻¹⁵ at the shared tick size.
const M = { left: 74, right: 14, top: 14, bottom: 30 };
const SAMPLES = 480;

const offsetRuns = (o: number): MathRun[] => {
  if (o === 0) return [mv('x'), sub('0')];
  const sign = o < 0 ? '−' : '+';
  const a = Math.abs(o);
  return [mm(sign + (a === 1 ? '' : String(a))), mv('h')];
};

export function StencilView({
  f,
  x0,
  slope,
  curvature,
  runs,
  focus,
  mode,
  ease,
  ariaLabel,
}: StencilViewProps) {
  const colors = useChartColors();

  const { canvasRef } = useCanvas((ctx, size) => {
    const { width, height, dpr } = size;
    const frame = { left: M.left, top: M.top, right: width - M.right, bottom: height - M.bottom };
    if (frame.right - frame.left < 40 || frame.bottom - frame.top < 40) return;
    const run = runs[focus] ?? runs[0];
    const f0 = f(x0);
    const s = slope;

    // The zoom: h(t) with eased transitions between levels.
    const tFocus = run ? Math.max(0, Math.min(run.t, run.trace.length - 1)) : 0;
    const kA = Math.floor(tFocus);
    const frac = tFocus - kA;
    const e = ease ? easeInOut(frac) : frac;
    const h0 = (run?.trace[0]?.info.h as number | undefined) ?? 0.1;
    const h = h0 * 2 ** -(kA + e);
    const reach = run ? (REACH[run.id] ?? 1) : 1;
    const W = windowHalfWidth(h, reach);

    // f sampled at the representable abscissae x0 + u (rounded), plotted at their true offset.
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i <= SAMPLES; i++) {
      const x = x0 + (-W + (2 * W * i) / SAMPLES);
      const u = x - x0;
      xs.push(u);
      ys.push(viewY(f(x), u, f0, s, mode));
    }

    // Levels on screen per run: the current one and, while zooming, the next (cross-fade).
    const layers = runs.map((r) => {
      const t = Math.max(0, Math.min(r.t, r.trace.length - 1));
      const k = Math.floor(t);
      const fr = t - k;
      const w = ease ? easeInOut(fr) : fr;
      const out: { step: Step; alpha: number }[] = [{ step: r.trace[k], alpha: 1 - w }];
      if (k + 1 < r.trace.length && w > 0.001) out.push({ step: r.trace[k + 1], alpha: w });
      return out.filter((l) => l.step && l.alpha > 0.001);
    });
    const geoms = runs.map((r, i) =>
      layers[i].map((l) => ({ ...l, g: stencilGeometry(r.id, l.step, f0, s, mode) })),
    );

    // y-range: the curve, the focus method's stencil and estimate line, and 0.
    let lo = 0,
      hi = 0;
    const take = (y: number) => {
      if (!Number.isFinite(y)) return;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    };
    ys.forEach(take);
    for (const { g } of geoms[focus] ?? []) {
      g.points.forEach((p) => take(p.y));
      if (g.slope !== null) {
        take(g.slope * W);
        take(-g.slope * W);
      }
    }
    const minSpan = Math.max(4 * ulp(f0), 1e-300);
    if (hi - lo < minSpan) {
      const mid = (hi + lo) / 2;
      lo = mid - minSpan / 2;
      hi = mid + minSpan / 2;
    }
    const pad = (hi - lo) * 0.12;
    lo -= pad;
    hi += pad;

    const xScale = linearScale([-W / h, W / h], [frame.left, frame.right]);
    const yScale = linearScale([lo, hi], [frame.bottom, frame.top]);
    const P = (u: number, y: number): [number, number] => [xScale(u / h), yScale(y)];

    // ── Axes: vertical rules at x0 + o·h, y ticks as offsets ─────────────────────────────
    const hair = 1 / dpr;
    ctx.save();
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = hair;
    ctx.beginPath();
    const oMax = Math.floor(W / h);
    for (let o = -oMax; o <= oMax; o++) {
      const px = crisp(xScale(o), dpr);
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
    }
    const yTicks = linearTicks(lo, hi, Math.max(3, Math.floor((frame.bottom - frame.top) / 46)));
    for (const v of yTicks) {
      const py = crisp(yScale(v), dpr);
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
    ctx.strokeStyle = colors.axis;
    ctx.beginPath();
    ctx.moveTo(crisp(frame.left, dpr), frame.top);
    ctx.lineTo(crisp(frame.left, dpr), crisp(frame.bottom, dpr));
    ctx.lineTo(frame.right, crisp(frame.bottom, dpr));
    ctx.stroke();
    // x labels: x0 + o·h in KaTeX fonts.
    for (let o = -oMax; o <= oMax; o++) {
      drawMath(ctx, offsetRuns(o), xScale(o), frame.bottom + 19, {
        size: 12.5,
        align: 'center',
        color: colors.tick,
      });
    }
    // y labels: signed offsets in scientific notation.
    ctx.font = tickFont(colors);
    ctx.fillStyle = colors.tick;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const step = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
    for (const v of yTicks) {
      const label =
        Math.abs(v) < step * 1e-6 ? '0' : (v > 0 ? '+' : '') + sci(v, 2).replace(/^-/, '−');
      ctx.fillText(label, frame.left - 6, yScale(v));
    }
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();

    // ── Rounding quantum of f near x0: ±½ ulp(f(x0)) ─────────────────────────────────────
    const q = ulp(f0) / 2;
    const bandPx = Math.abs(yScale(-q) - yScale(q));
    if (Number.isFinite(bandPx) && bandPx >= 2) {
      ctx.fillStyle = colors.text;
      ctx.globalAlpha = 0.06;
      ctx.fillRect(frame.left, yScale(q), frame.right - frame.left, bandPx);
      ctx.globalAlpha = 1;
      if (bandPx >= 13)
        drawMath(
          ctx,
          [mm('± ½ ulp '), mv('f'), mm('('), mv('x'), sub('0'), mm(')')],
          frame.right - 8,
          yScale(-q) + 14,
          { size: 11.5, align: 'right', color: colors.text3, halo: colors.halo },
        );
    }

    // ── Reference: the tangent ℓ (and the Taylor parabola for f″) ──────────────────────────
    const focusSecond = run ? SECOND_DERIVATIVE.has(run.id) : false;
    ctx.strokeStyle = colors.text2;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    const tangentY = (u: number) => (mode === 'dev' ? 0 : s * u);
    ctx.moveTo(...P(-W, tangentY(-W)));
    ctx.lineTo(...P(W, tangentY(W)));
    ctx.stroke();
    if (focusSecond && curvature !== null && Number.isFinite(curvature)) {
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const u = -W + (2 * W * i) / 120;
        const y = tangentY(u) + 0.5 * curvature * u * u;
        if (i) ctx.lineTo(...P(u, y));
        else ctx.moveTo(...P(u, y));
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // ── The curve ────────────────────────────────────────────────────────────────────────
    ctx.strokeStyle = colors.text;
    ctx.globalAlpha = 0.88;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < xs.length; i++) {
      const y = ys[i];
      if (!Number.isFinite(y)) {
        pen = false;
        continue;
      }
      const [px, py] = P(xs[i], y);
      if (pen) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // ── Each method's step: estimate line / parabola, chords, stencil points ──────────────
    const order = runs
      .map((_, i) => i)
      .sort((a, b) => (a === focus ? 1 : b === focus ? -1 : a - b));
    for (const i of order) {
      const color = colors.series[runs[i].slot % colors.series.length];
      const strong = i === focus;
      for (const { g, alpha } of geoms[i]) {
        ctx.globalAlpha = alpha * (strong ? 1 : 0.7);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        if (g.slope !== null) {
          ctx.lineWidth = strong ? 2 : 1.5;
          ctx.beginPath();
          ctx.moveTo(...P(-W, g.slope * -W));
          ctx.lineTo(...P(W, g.slope * W));
          ctx.stroke();
        }
        if (g.parabola) {
          const { a, b, c } = g.parabola;
          ctx.lineWidth = strong ? 2 : 1.5;
          ctx.beginPath();
          for (let j = 0; j <= 120; j++) {
            const u = -W + (2 * W * j) / 120;
            const y = a * u * u + b * u + c;
            if (j) ctx.lineTo(...P(u, y));
            else ctx.moveTo(...P(u, y));
          }
          ctx.stroke();
        }
        ctx.lineWidth = 1.25;
        ctx.setLineDash([2, 3]);
        for (const [p1, p2] of g.chords) {
          if (!Number.isFinite(p1.y) || !Number.isFinite(p2.y)) continue;
          ctx.beginPath();
          ctx.moveTo(...P(p1.u, p1.y));
          ctx.lineTo(...P(p2.u, p2.y));
          ctx.stroke();
        }
        ctx.setLineDash([]);
        for (const p of g.points) {
          if (!Number.isFinite(p.y)) continue;
          const [px, py] = P(p.u, p.y);
          ctx.fillStyle = colors.halo;
          ctx.beginPath();
          ctx.arc(px, py, 5.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, 3.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }

    // x0: hollow ring in ink (the start/anchor grammar).
    const [ox, oy] = P(0, 0);
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(ox, oy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(ox, oy, 4.2, 0, Math.PI * 2);
    ctx.stroke();

    // In 'f' mode label the tangent (in 'dev' mode it is the u-axis, named in the panel note).
    if (mode === 'f') {
      const [lx, ly] = P(-W, tangentY(-W));
      drawMath(
        ctx,
        focusSecond
          ? [mm('Taylor parabola')]
          : [mv('ℓ'), mm('('), mv('x'), mm(')'), mm('  tangent')],
        lx + 8,
        Math.max(frame.top + 12, Math.min(frame.bottom - 8, ly + 17)),
        { size: 12, align: 'left', color: colors.text2, halo: colors.halo },
      );
    }
    ctx.restore();
  });

  return (
    <div className={styles.canvasBox} role="img" aria-label={ariaLabel}>
      <canvas ref={canvasRef} />
    </div>
  );
}
