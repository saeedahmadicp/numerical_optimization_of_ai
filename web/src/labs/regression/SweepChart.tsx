/**
 * A small model-selection chart: several lines over a parameter (degree d on a linear axis, or
 * λ on a log axis), a marker at the current value, click (or arrow keys) to choose a value.
 * Used for the degree sweep (training / leave-one-out / true error vs d) and the ridge path
 * (the same errors, or the standardized coefficients, vs λ).
 */
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import {
  drawAxes,
  drawMath,
  linearScale,
  logExtent,
  logScale,
  niceDomain,
  useCanvas,
  type MathRun,
  type Scale,
} from '../../viz';
import styles from './RegressionLab.module.css';

export interface SweepLine {
  key: string;
  values: readonly (number | null)[];
  dash?: number[];
  width?: number;
  alpha?: number;
  /** Direct label at the right end of the line (math runs). */
  label?: readonly MathRun[];
  /** Mark the smallest value with a diamond. */
  markMin?: boolean;
  /** The line's value at `marker` when the marker is off the grid (exact, not interpolated). */
  at?: number | null;
}

export interface SweepChartProps {
  xs: readonly number[];
  lines: readonly SweepLine[];
  /** Series color (the method's slot color). */
  color: string;
  logX?: boolean;
  logY?: boolean;
  /** Index into `xs` of the current value (the grid point nearest to it). */
  current: number;
  /**
   * The exact current value when it is not a grid point (ridge λ from the slider), or lies past
   * the grid (a degree beyond the sweep). Dots then use each line's `at`; past the grid an arrow
   * with `markerLabel` sits at the right edge.
   */
  marker?: number;
  markerLabel?: readonly MathRun[];
  onPick?: (index: number) => void;
  /** Arrow keys: step the value itself (default: move to the neighboring grid index). */
  onStep?: (dir: 1 | -1) => void;
  /** Slider semantics when they are not the grid index (value-based). */
  aria?: { min: number; max: number; now: number; text: string };
  xName: readonly MathRun[];
  yName: readonly MathRun[];
  /** Draw the line y = 0 (signed quantities). */
  zeroLine?: boolean;
  ariaLabel: string;
  /** Name of the picked quantity for the keyboard announcement ("degree", "λ"). */
  valueText: (index: number) => string;
}

const M = { left: 48, right: 40, top: 24, bottom: 34 };

export function SweepChart({
  xs,
  lines,
  color,
  logX = false,
  logY = true,
  current,
  marker,
  markerLabel,
  onPick,
  onStep,
  aria,
  xName,
  yName,
  zeroLine = false,
  ariaLabel,
  valueText,
}: SweepChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const xDom = useMemo<[number, number]>(
    () => (logX ? [xs[0], xs[xs.length - 1]] : [xs[0], Math.max(xs[0] + 1, xs[xs.length - 1])]),
    [xs, logX],
  );
  const yDom = useMemo<[number, number]>(() => {
    const all = lines.flatMap((l) =>
      l.values.filter((v): v is number => v !== null && Number.isFinite(v) && (!logY || v > 0)),
    );
    if (all.length === 0) return logY ? [1e-3, 1] : [-1, 1];
    if (logY) {
      const [lo, hi] = logExtent(all);
      // Keep at most 6 decades: a fit that interpolates (error ~1e-14) would flatten the rest.
      return [Math.max(lo, hi / 1e6), hi];
    }
    const lo = Math.min(...all, zeroLine ? 0 : Infinity),
      hi = Math.max(...all, zeroLine ? 0 : -Infinity);
    return niceDomain(lo - 0.04 * (hi - lo || 1), hi + 0.04 * (hi - lo || 1));
  }, [lines, logY, zeroLine]);

  const scales = (w: number, h: number): { x: Scale; y: Scale } => ({
    x: logX ? logScale(xDom, [M.left, w - M.right]) : linearScale(xDom, [M.left, w - M.right]),
    y: logY ? logScale(yDom, [h - M.bottom, M.top]) : linearScale(yDom, [h - M.bottom, M.top]),
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
      xName,
      yName,
      xInteger: !logX,
      xTicks: logX ? 7 : Math.min(8, xs.length),
      yTicks: 4,
    });
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 2, frame.right - frame.left, frame.bottom - frame.top + 4);
    ctx.clip();
    if (zeroLine) {
      ctx.strokeStyle = colors.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(frame.left, Math.round(y(0)) + 0.5);
      ctx.lineTo(frame.right, Math.round(y(0)) + 0.5);
      ctx.stroke();
    }
    // Current value (an exact marker, or the grid point).
    const mx = marker ?? xs[current];
    const offRight = mx > xDom[1] * (1 + 1e-12);
    const offGrid =
      marker !== undefined && Math.abs(marker - xs[current]) > 1e-9 * Math.abs(marker);
    const cx = offRight ? frame.right : x(mx);
    ctx.strokeStyle = colors.playhead;
    ctx.lineWidth = 1;
    ctx.setLineDash(offRight ? [3, 3] : []);
    ctx.beginPath();
    ctx.moveTo(Math.round(cx) + 0.5, frame.top);
    ctx.lineTo(Math.round(cx) + 0.5, frame.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    if (hover !== null && hover !== current) {
      const hx = x(xs[hover]);
      ctx.strokeStyle = colors.grid;
      ctx.beginPath();
      ctx.moveTo(Math.round(hx) + 0.5, frame.top);
      ctx.lineTo(Math.round(hx) + 0.5, frame.bottom);
      ctx.stroke();
    }
    const clampY = (v: number) => Math.max(yDom[0], Math.min(yDom[1], v));
    for (const l of lines) {
      ctx.globalAlpha = l.alpha ?? 1;
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = (l.width ?? 1.75) + 2.5;
      ctx.setLineDash([]);
      const path = () => {
        ctx.beginPath();
        let pen = false;
        l.values.forEach((v, i) => {
          if (v === null || !Number.isFinite(v) || (logY && v <= 0)) {
            pen = false;
            return;
          }
          const px = x(xs[i]),
            py = y(clampY(v));
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        });
      };
      path();
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = l.width ?? 1.75;
      ctx.setLineDash(l.dash ?? []);
      path();
      ctx.stroke();
      ctx.setLineDash([]);
      // Dots at the current value (none past the grid: that fit is not on this chart).
      const v = offRight ? null : offGrid ? l.at : l.values[current];
      if (v !== null && v !== undefined && Number.isFinite(v) && (!logY || v > 0)) {
        ctx.fillStyle = colors.halo;
        ctx.beginPath();
        ctx.arc(cx, y(clampY(v)), 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx, y(clampY(v)), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (l.markMin) {
        let best = -1;
        l.values.forEach((t, i) => {
          if (t !== null && Number.isFinite(t) && (best < 0 || t < (l.values[best] as number)))
            best = i;
        });
        if (best >= 0) {
          const px = x(xs[best]),
            py = y(clampY(l.values[best] as number));
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = colors.halo;
          ctx.fillRect(-5, -5, 10, 10);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(-3.5, -3.5, 7, 7);
          ctx.restore();
        }
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    // A marker past the grid: an arrow at the right edge with its value ("d = 12 ▸").
    if (offRight && markerLabel) {
      const ty = frame.top - 8;
      drawMath(ctx, [...markerLabel, { t: ' ▸', style: 'main' }], frame.right + 4, ty, {
        size: 11.5,
        align: 'right',
        baseline: 'middle',
        color: colors.text,
        halo: colors.halo,
      });
    }
    // Direct labels at the right end of each labelled line (outside the clip).
    const ends = lines
      .flatMap((l) => {
        if (!l.label) return [];
        let i = l.values.length - 1;
        while (
          i >= 0 &&
          !(l.values[i] !== null && Number.isFinite(l.values[i]!) && (!logY || l.values[i]! > 0))
        )
          i--;
        return i < 0 ? [] : [{ l, i, py: y(clampY(l.values[i] as number)) }];
      })
      .sort((a, b) => a.py - b.py);
    // Push labels apart (13 px), then shift the stack back up if it ran past the frame.
    for (let j = 1; j < ends.length; j++)
      if (ends[j].py - ends[j - 1].py < 13) ends[j].py = ends[j - 1].py + 13;
    const over = ends.length ? ends[ends.length - 1].py - (frame.bottom + 4) : 0;
    if (over > 0) for (const e of ends) e.py -= over;
    for (const { l, i, py } of ends)
      drawMath(ctx, l.label!, x(xs[i]) + 6, py, {
        size: 11.5,
        baseline: 'middle',
        color: colors.text2,
        halo: colors.halo,
      });
  });

  const indexAt = (e: PointerEvent<HTMLDivElement>) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const { x } = scales(r.width, r.height);
    const v = x.invert(e.clientX - r.left);
    let best = 0;
    xs.forEach((t, i) => {
      const d = logX ? Math.abs(Math.log(t) - Math.log(v)) : Math.abs(t - v);
      const db = logX ? Math.abs(Math.log(xs[best]) - Math.log(v)) : Math.abs(xs[best] - v);
      if (d < db) best = i;
    });
    return best;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!onPick && !onStep) return;
    const step =
      e.key === 'ArrowRight' || e.key === 'ArrowUp'
        ? 1
        : e.key === 'ArrowLeft' || e.key === 'ArrowDown'
          ? -1
          : 0;
    if (!step) return;
    e.preventDefault();
    e.stopPropagation();
    if (onStep) onStep(step);
    else onPick?.(Math.max(0, Math.min(xs.length - 1, current + step)));
  };

  return (
    <div
      ref={box}
      className={styles.sweep}
      role="slider"
      tabIndex={onPick ? 0 : -1}
      aria-label={ariaLabel}
      aria-valuemin={aria?.min ?? 0}
      aria-valuemax={aria?.max ?? xs.length - 1}
      aria-valuenow={aria?.now ?? current}
      aria-valuetext={aria?.text ?? valueText(current)}
      data-own-keys
      onKeyDown={onKey}
      onPointerMove={(e) => setHover(indexAt(e))}
      onPointerLeave={() => setHover(null)}
      onClick={(e) => {
        const i = indexAt(e as unknown as PointerEvent<HTMLDivElement>);
        if (i !== null) onPick?.(i);
      }}
    >
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
    </div>
  );
}
