/**
 * Convergence against cost: the bracket width b − a (or |x̂ₖ − x⋆|) on a log axis against the
 * number of evaluations n. On these axes a linear method is a straight line whose slope is its
 * contraction per evaluation, so golden section's (1/φ)ⁿ, ternary search's (2/3)^{n/2} and
 * dichotomous search's 2^{−n/2} are drawn as dashed guides through the first point of each
 * run, and the legend states each run's measured factor θ̂ per evaluation (θ, not ρ: the lab
 * keeps ρ for the golden fraction (3 − √5)/2 ≈ 0.382 of the update rules).
 */
import { useMemo, useRef } from 'react';
import { useChartColors } from '../../ui/theme';
import { Formula } from '../../ui/components/Formula';
import {
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  logExtent,
  logScale,
  mathSans,
  mathVar,
  measureMath,
  SeriesLegend,
  useCanvas,
  type MathRun,
} from '../../viz';
import styles from './ScalarLab.module.css';

export interface RateSeries {
  label: string;
  slot: number;
  /** Cumulative evaluations at each step. */
  n: readonly number[];
  /** Measure at each step (null: not defined, e.g. no bracket). */
  y: readonly (number | null)[];
  end: 'converged' | 'stopped';
  /** Iterations of the run (the legend's count). */
  count: number;
  /** Measured contraction per evaluation (legend). */
  rate: number | null;
}

export interface RateGuide {
  /** Contraction per evaluation. */
  rate: number;
  /** Start of the guide (n₀, y₀). */
  n0: number;
  y0: number;
  label: MathRun[];
}

export interface RateChartProps {
  series: readonly RateSeries[];
  /** Current step of each series. */
  ks: readonly number[];
  /** Continuous step of each series (the head moves smoothly between evaluations). */
  ts: readonly number[];
  guides: readonly RateGuide[];
  yName: MathRun[];
  ariaLabel: string;
  /** Click: the evaluation count under the pointer. */
  onSeekEvaluations?: (n: number) => void;
}

const M = { left: 48, right: 16, top: 24, bottom: 34 };
const ok = (v: number | null | undefined): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v > 0;

export function RateChart({
  series,
  ks,
  ts,
  guides,
  yName,
  ariaLabel,
  onSeekEvaluations,
}: RateChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const maxN = Math.max(4, ...series.map((s) => s.n[s.n.length - 1] ?? 0));
  const yDom = useMemo<[number, number]>(() => {
    const vals = series.flatMap((s) => s.y.filter(ok));
    if (!vals.length) return [1e-10, 1];
    const [lo, hi] = logExtent(vals);
    return [lo, hi];
  }, [series]);
  const empty = !series.some((s) => s.y.some(ok));

  const scales = (w: number, h: number) => ({
    x: linearScale([0, maxN], [M.left, w - M.right]),
    y: logScale(yDom, [h - M.bottom, M.top]),
  });

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      xInteger: true,
      xName: [mathSans('evaluations '), mathVar('n')],
      yName,
    });
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 4, frame.right - frame.left, frame.bottom - frame.top + 4);
    ctx.clip();

    // Guides: y₀ · rateⁿ⁻ⁿ⁰, dashed ink (never a method color), labelled at the right. The
    // labels are drawn last, with a halo, so no series line covers them.
    const guideLabels: { runs: MathRun[]; x: number; y: number }[] = [];
    guides.forEach((g, gi) => {
      const n1 = maxN;
      const y1 = g.y0 * g.rate ** (n1 - g.n0);
      ctx.save();
      ctx.strokeStyle = colors.text3;
      ctx.lineWidth = 1;
      ctx.setLineDash(gi === 0 ? [5, 4] : gi === 1 ? [2, 3] : [8, 3, 2, 3]);
      ctx.beginPath();
      ctx.moveTo(x(g.n0), y(g.y0));
      ctx.lineTo(x(n1), y(y1));
      ctx.stroke();
      ctx.restore();
      // Label where the guide leaves the frame (right edge or bottom).
      let ln = n1,
        ly = y1;
      if (y(y1) > frame.bottom - 4) {
        const nb = g.n0 + Math.log(yDom[0] / g.y0) / Math.log(g.rate);
        ln = Math.min(n1, nb);
        ly = g.y0 * g.rate ** (ln - g.n0);
      }
      guideLabels.push({ runs: g.label, x: x(ln) - 10, y: Math.min(y(ly) - 12, frame.bottom - 6) });
    });

    for (const pass of ['future', 'past'] as const) {
      series.forEach((sr, i) => {
        const color = colors.series[sr.slot % colors.series.length];
        const kNow = ks[i] ?? 0;
        const from = pass === 'past' ? 0 : kNow;
        const to = pass === 'past' ? kNow : sr.y.length - 1;
        ctx.strokeStyle = color;
        ctx.globalAlpha = pass === 'future' ? 0.18 : 1;
        ctx.lineJoin = 'round';
        const path = () => {
          ctx.beginPath();
          let pen = false;
          for (let k = from; k <= to; k++) {
            const v = sr.y[k];
            if (!ok(v)) {
              pen = false;
              continue;
            }
            if (pen) ctx.lineTo(x(sr.n[k]), y(v));
            else ctx.moveTo(x(sr.n[k]), y(v));
            pen = true;
          }
        };
        if (pass === 'past') {
          path();
          ctx.strokeStyle = colors.halo;
          ctx.lineWidth = 4.2;
          ctx.globalAlpha = 0.85;
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = color;
        }
        ctx.lineWidth = pass === 'future' ? 1.25 : 1.9;
        path();
        ctx.stroke();
        // Steps are discrete: a dot per step on short runs.
        if (pass === 'past' && sr.y.length <= 60) {
          ctx.fillStyle = color;
          for (let k = 0; k <= kNow; k++) {
            const v = sr.y[k];
            if (!ok(v)) continue;
            ctx.beginPath();
            ctx.arc(x(sr.n[k]), y(v), 1.7, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
        if (pass === 'past') {
          // Head: between steps it slides along the segment to the next step.
          const t = ts[i] ?? kNow;
          const k0 = Math.min(Math.floor(t), sr.y.length - 1);
          const v0 = sr.y[k0];
          if (!ok(v0)) return;
          let hx = x(sr.n[k0]),
            hy = y(v0);
          const v1 = sr.y[k0 + 1];
          const frac = t - k0;
          if (frac > 0 && ok(v1)) {
            hx += (x(sr.n[k0 + 1]) - hx) * frac;
            hy += (y(v1) - hy) * frac;
          }
          const atEnd = k0 >= sr.y.length - 1;
          ctx.fillStyle = colors.halo;
          ctx.beginPath();
          ctx.arc(hx, hy, 4.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(hx, hy, 3.2, 0, Math.PI * 2);
          if (atEnd && sr.end === 'stopped') {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.6;
            ctx.stroke();
          } else {
            ctx.fillStyle = color;
            ctx.fill();
          }
        }
      });
    }
    for (const l of guideLabels) {
      // A surface-colored plate: the series that follows its guide never shows through the text.
      const w = measureMath(ctx, l.runs, 12);
      ctx.fillStyle = colors.surface;
      ctx.fillRect(l.x - w - 3, l.y - 13, w + 6, 18);
      drawMath(ctx, l.runs, l.x, l.y, {
        size: 12,
        align: 'right',
        color: colors.text2,
        halo: colors.halo,
      });
    }
    ctx.restore();
    // A hairline at n = 0.
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1 / s.dpr;
    ctx.beginPath();
    ctx.moveTo(crisp(x(0), s.dpr), frame.top);
    ctx.lineTo(crisp(x(0), s.dpr), frame.bottom);
    ctx.stroke();
  });

  const onClick = (e: { clientX: number }) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !onSeekEvaluations) return;
    const { x } = scales(r.width, r.height);
    onSeekEvaluations(Math.max(0, Math.round(x.invert(e.clientX - r.left))));
  };

  return (
    <div className={styles.rate}>
      <SeriesLegend
        items={series.map((s) => ({
          slot: s.slot,
          label: s.label,
          count: s.count,
          extra: !s.y.some(ok) ? (
            'no bracket'
          ) : s.rate !== null ? (
            <span title="Measured contraction θ̂ of the bracket width per evaluation">
              <Formula tex="\hat\theta" /> = {s.rate.toFixed(3)} / eval
            </span>
          ) : undefined,
        }))}
      />
      <div
        ref={box}
        className={styles.rateCanvas}
        style={{ cursor: onSeekEvaluations ? 'pointer' : undefined }}
        onClick={onClick}
      >
        <canvas ref={canvasRef} role="img" aria-label={ariaLabel} />
        {empty && (
          <div className={styles.rateEmpty} role="status">
            No run has this measure. Newton’s method keeps no bracket: switch to the error |x̂ₖ −
            x⋆|.
          </div>
        )}
      </div>
    </div>
  );
}
