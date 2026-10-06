/**
 * Training curves on one clock: the error measure against epochs (log y) above, the learning
 * rate ηₜ of every update (log y) below, a shared playhead through both. The x position of a
 * trace step is its update count k divided by the updates per epoch U, so a run that records
 * every r-th update is drawn at the true epoch of each recorded step.
 *
 * For an adaptive method in focus, the lower panel also draws its effective per-coordinate rates
 * η/(√rᵢ + ε) (dashed), which is what the method actually multiplies the gradient by.
 */
import { useRef, type PointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import { useCanvas } from '../../viz/useCanvas';
import { linearScale, logExtent, logScale, crisp } from '../../viz/scales';
import { TICK_SIZE, drawAxes } from '../../viz/axes';
import { SeriesLegend } from '../../viz/SeriesLegend';
import { drawMath, mathSub, mathVar, mathSans, type MathRun } from '../../viz';
import styles from './StochasticLab.module.css';

export interface TrainSeries {
  /** The method's name in the legend (`spec.shortName` only with `title` set to `spec.name`). */
  label: string;
  /** The full name, when `label` is a short name. */
  title?: string;
  /** Updates the run made (the legend's count). */
  count: number;
  slot: number;
  /** Epoch position of each trace step. */
  epochs: readonly number[];
  /** Error measure per trace step (null/≤ 0 are gaps). */
  values: readonly (number | null)[];
  /** η of the update that produced each step (null at k = 0). */
  lr: readonly (number | null)[];
  end: 'converged' | 'stopped';
  muted?: boolean;
}

export interface TrainingChartProps {
  series: readonly TrainSeries[];
  /** Continuous playhead (trace index). */
  t: number;
  yName: readonly MathRun[];
  /** Effective per-coordinate rates of the focused adaptive method. */
  effective?: { slot: number; values: readonly (readonly number[] | null)[] } | null;
  totalEpochs: number;
  /** Draw the learning-rate panel (off for a constant schedule without adaptive rates). */
  showLr?: boolean;
  onSeek?: (index: number) => void;
  ariaLabel: string;
}

const M = { left: 46, right: 30, top: 24, bottom: 30, gap: 26 };

/** The epoch axis's name: "epoch e" (the word in Inter, the variable in KaTeX italic). */
const EPOCH_NAME: MathRun[] = [mathSans('epoch '), mathVar('e')];

/** Position along a series at continuous index t (linear in index). */
function at(values: readonly number[], t: number): number {
  const i = Math.max(0, Math.min(values.length - 1, Math.floor(t)));
  const j = Math.min(values.length - 1, i + 1);
  const u = Math.max(0, Math.min(1, t - i));
  return values[i] + (values[j] - values[i]) * u;
}

export function TrainingChart({
  series,
  t,
  yName,
  effective,
  totalEpochs,
  showLr = true,
  onSeek,
  ariaLabel,
}: TrainingChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const longest = series.reduce<TrainSeries | null>(
    (a, s) => (a === null || s.epochs.length > a.epochs.length ? s : a),
    null,
  );

  const layout = (w: number, h: number) => {
    const inner = h - M.top - M.bottom - M.gap;
    const topH = showLr ? Math.max(60, Math.round(inner * 0.64)) : h - M.top - M.bottom;
    const top = { left: M.left, right: w - M.right, top: M.top, bottom: M.top + topH };
    const bottom = {
      left: M.left,
      right: w - M.right,
      top: top.bottom + M.gap,
      bottom: h - M.bottom,
    };
    const x = linearScale([0, Math.max(1, totalEpochs)], [M.left, w - M.right]);
    return { top, bottom, x };
  };

  const { canvasRef } = useCanvas((ctx, s) => {
    const { top, bottom, x } = layout(s.width, s.height);
    const ok = (v: number | null | undefined): v is number =>
      v !== null && v !== undefined && Number.isFinite(v) && v > 0;
    const yTop = logScale(logExtent(series.flatMap((r) => r.values.filter(ok))), [
      top.bottom,
      top.top,
    ]);
    const lrVals = [
      ...series.flatMap((r) => r.lr.filter(ok)),
      ...(effective?.values.flatMap((v) => (v ? v.filter(ok) : [])) ?? []),
    ];
    const yBot = logScale(logExtent(lrVals), [bottom.bottom, bottom.top]);

    // Top panel: clip away its x tick labels (the lower panel carries the epoch axis).
    ctx.save();
    if (showLr) {
      ctx.beginPath();
      ctx.rect(0, 0, s.width, top.bottom + 1);
      ctx.clip();
    }
    drawAxes(ctx, {
      x,
      y: yTop,
      frame: top,
      colors,
      dpr: s.dpr,
      yName,
      xName: showLr ? undefined : EPOCH_NAME,
      xInteger: true,
    });
    ctx.restore();
    if (showLr)
      drawAxes(ctx, {
        x,
        y: yBot,
        frame: bottom,
        colors,
        dpr: s.dpr,
        xName: EPOCH_NAME,
        yName: [mathVar('η'), mathSub('t', 'italic')],
        xInteger: true,
        yTicks: 2,
      });

    const panel = (
      frame: typeof top,
      y: typeof yTop,
      pick: (r: TrainSeries) => readonly (number | null)[],
      width: number,
    ) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(
        frame.left - 5,
        frame.top - 5,
        frame.right - frame.left + 10,
        frame.bottom - frame.top + 10,
      );
      ctx.clip();
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      const clampY = (v: number) => Math.max(frame.top - 3, Math.min(frame.bottom + 3, y(v)));
      for (const pass of ['future', 'past'] as const) {
        for (const r of series) {
          const vals = pick(r);
          const n = vals.length;
          const kEnd = Math.min(t, n - 1);
          const from = pass === 'past' ? 0 : Math.floor(kEnd);
          const to = pass === 'past' ? Math.floor(kEnd) : n - 1;
          const color = colors.series[r.slot % colors.series.length];
          const trace = () => {
            ctx.beginPath();
            let pen = false;
            for (let k = from; k <= to; k++) {
              const v = vals[k];
              if (!ok(v)) {
                pen = false;
                continue;
              }
              const px = x(r.epochs[k]),
                py = clampY(v);
              if (pen) ctx.lineTo(px, py);
              else ctx.moveTo(px, py);
              pen = true;
            }
          };
          ctx.globalAlpha = pass === 'future' ? 0.16 : r.muted ? 0.45 : 1;
          if (pass === 'past') {
            trace();
            ctx.strokeStyle = colors.halo;
            ctx.lineWidth = width + 2.4;
            ctx.stroke();
          }
          trace();
          ctx.strokeStyle = color;
          ctx.lineWidth = pass === 'future' ? 1.1 : width;
          ctx.stroke();
          ctx.globalAlpha = 1;
          if (pass === 'past') {
            const k0 = Math.floor(kEnd);
            const v = vals[k0];
            if (ok(v)) {
              const px = x(r.epochs[k0]),
                py = clampY(v);
              const atEnd = k0 >= n - 1;
              ctx.fillStyle = colors.halo;
              ctx.beginPath();
              ctx.arc(px, py, 4.4, 0, Math.PI * 2);
              ctx.fill();
              ctx.beginPath();
              ctx.arc(px, py, 2.9, 0, Math.PI * 2);
              if (atEnd && r.end === 'stopped') {
                ctx.fillStyle = colors.surface;
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
              } else {
                ctx.fillStyle = color;
                ctx.fill();
              }
            }
          }
        }
      }
      ctx.restore();
    };
    panel(top, yTop, (r) => r.values, 1.9);
    if (showLr) panel(bottom, yBot, (r) => r.lr, 1.5);

    // Effective per-coordinate rates of the focused adaptive method (dashed), labelled w₀, w₁.
    if (effective && showLr) {
      const color = colors.series[effective.slot % colors.series.length];
      const ref = series.find((r) => r.slot === effective.slot);
      if (ref) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(
          bottom.left,
          bottom.top - 4,
          bottom.right - bottom.left + 30,
          bottom.bottom - bottom.top + 8,
        );
        ctx.clip();
        const kEnd = Math.min(t, effective.values.length - 1);
        for (const coord of [0, 1]) {
          ctx.setLineDash(coord === 0 ? [4, 3] : [1.5, 3]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          let pen = false,
            last: [number, number] | null = null;
          for (let k = 0; k <= Math.floor(kEnd); k++) {
            const v = effective.values[k]?.[coord];
            if (!ok(v)) {
              pen = false;
              continue;
            }
            const px = x(ref.epochs[k]),
              py = yBot(v);
            if (pen) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            pen = true;
            last = [px, py];
          }
          ctx.stroke();
          ctx.setLineDash([]);
          if (last)
            drawMath(ctx, [mathVar('w'), mathSub(String(coord))], last[0] + 5, last[1], {
              size: TICK_SIZE + 1,
              baseline: 'middle',
              color: colors.text2,
              halo: colors.halo,
            });
        }
        ctx.restore();
      }
    }

    // Shared playhead.
    if (longest && Number.isFinite(t)) {
      const px = crisp(x(at(longest.epochs, Math.min(t, longest.epochs.length - 1))), s.dpr);
      ctx.save();
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(px, top.top);
      ctx.lineTo(px, top.bottom);
      if (showLr) {
        ctx.moveTo(px, bottom.top);
        ctx.lineTo(px, bottom.bottom);
      }
      ctx.stroke();
      ctx.restore();
    }
  });

  const seek = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!onSeek || !longest) return;
    const r = e.currentTarget.getBoundingClientRect();
    const { x } = layout(r.width, r.height);
    const ep = x.invert(e.clientX - r.left);
    let best = 0;
    longest.epochs.forEach((v, i) => {
      if (Math.abs(v - ep) < Math.abs(longest.epochs[best] - ep)) best = i;
    });
    onSeek(best);
  };

  return (
    <div className={styles.trainWrap} ref={box}>
      {series.length > 0 && (
        <SeriesLegend
          className={styles.legend}
          items={series.map((r) => ({
            slot: r.slot,
            label: r.label,
            title: r.title,
            count: r.count,
          }))}
        />
      )}
      <canvas
        ref={canvasRef}
        className={styles.trainCanvas}
        role="img"
        aria-label={ariaLabel}
        onPointerDown={seek}
      />
    </div>
  );
}
