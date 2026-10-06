import { useMemo, useRef, useState } from 'react';
import { useChartColors } from '../ui/theme';
import { Swatch } from '../ui/components/MethodChip';
import { SciText } from '../ui/components/Num';
import { int, sci } from '../core/format';
import { useCanvas } from './useCanvas';
import { linearScale, logExtent, logScale, niceDomain, crisp, type Scale } from './scales';
import { drawAxes } from './axes';
import { milestoneIndices } from './PathLayer';
import { drawMath, measureMath, sans, v as mv, type MathRun } from './mathText';
import { slopeTriangle, type SlopeGuide } from './chartMath';
import { SeriesLegend } from './SeriesLegend';
import styles from './viz.module.css';

export interface ConvergenceSeries {
  label: string;
  /** Series slot (0-based) → method color. */
  slot: number;
  /** value[k]; null/NaN/≤0 (log) are gaps. */
  values: readonly (number | null)[];
  /**
   * x-coordinate of each value (default: the iteration index k). Use it for error-vs-h or
   * error-vs-n plots; the playhead then selects an index, not an x.
   */
  x?: readonly number[];
  /** How the run ended: solid end dot (converged) or hollow (stopped). */
  end?: 'converged' | 'stopped';
  /** Shown after the name in the legend (iteration count, tabular). */
  count?: number;
  /**
   * A proven rate to annotate at the end of the curve ("quadratic", "superlinear"), set in
   * Newsreader italic. Only when the method's theory guarantees it.
   */
  rate?: string;
}

export type { SlopeGuide } from './chartMath';

export interface ConvergenceChartProps {
  series: readonly ConvergenceSeries[];
  /** Playhead (continuous index); points after it are drawn faint. Infinity shows all. */
  t: number;
  /** y-axis name (plain text; `yName` for math runs). */
  yLabel: string;
  yName?: readonly MathRun[];
  /** x-axis name (default "k"; on a log axis "iteration k ≥ 1"). */
  xLabel?: string;
  xName?: readonly MathRun[];
  logY?: boolean;
  /** Log x axis: k ≥ 1 only (k = 0 cannot be shown), so rates read as shapes. */
  logX?: boolean;
  xDomain?: [number, number];
  yDomain?: [number, number];
  /** Click on the chart → seek to that index. */
  onSeek?: (k: number) => void;
  /** Hover → index under the pointer (null on leave), e.g. to ghost 𝐱ₖ on the landscape. */
  onHover?: (k: number | null) => void;
  ariaLabel?: string;
  className?: string;
  /** Show the legend row (default: when there are ≥ 2 series). */
  legend?: boolean;
  /** Diamonds at k = 10, 10², … (default: on a log-k axis). */
  milestones?: boolean;
  slopes?: readonly SlopeGuide[];
  /** Small margins and type for figures under 320 px wide (home hero). */
  compact?: boolean;
  /**
   * Integer x ticks (default: on a linear iteration axis). Pass `true` for other counts plotted
   * through `series.x`, `false` for a continuous x.
   */
  xInteger?: boolean;
}

/** Error vs iteration (log-y by default), one series per method, synced to the playhead. */
export function ConvergenceChart({
  series,
  t,
  yLabel,
  yName,
  xLabel,
  xName,
  logY = true,
  logX = false,
  xDomain,
  yDomain,
  onSeek,
  onHover,
  ariaLabel,
  className,
  legend,
  milestones,
  slopes = [],
  compact = false,
  xInteger,
}: ConvergenceChartProps) {
  const colors = useChartColors();
  const [hoverK, setHoverK] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const M = compact
    ? { left: 34, right: 10, top: 20, bottom: 30 }
    : { left: 48, right: 14, top: 26, bottom: 32 };

  const xOf = (s: ConvergenceSeries, k: number) => (s.x ? s.x[k] : k);
  const maxK = Math.max(1, ...series.map((s) => s.values.length - 1));
  const empty = !series.some((s) =>
    s.values.some((v) => v !== null && Number.isFinite(v) && (!logY || v > 0)),
  );
  const yDom = useMemo<[number, number]>(() => {
    if (yDomain) return yDomain;
    const all = series.flatMap((s) =>
      s.values.filter((v): v is number => v !== null && Number.isFinite(v)),
    );
    if (logY) return logExtent(all);
    if (all.length === 0) return [0, 1];
    return niceDomain(Math.min(...all), Math.max(...all));
  }, [series, logY, yDomain]);
  const xDom = useMemo<[number, number]>(() => {
    if (xDomain) return xDomain;
    const xs = series.flatMap((s) => (s.x ? [...s.x] : []));
    if (xs.length) {
      const fin = xs.filter((v) => Number.isFinite(v) && (!logX || v > 0));
      return logX ? logExtent(fin) : niceDomain(Math.min(...fin), Math.max(...fin));
    }
    return logX ? [1, 10 ** Math.max(1, Math.ceil(Math.log10(maxK)))] : [0, maxK];
  }, [series, xDomain, logX, maxK]);

  const scales = (w: number, h: number): { x: Scale; y: Scale } => {
    const x = logX
      ? logScale(xDom, [M.left, w - M.right])
      : linearScale(xDom, [M.left, w - M.right]);
    const y = logY
      ? logScale(yDom, [h - M.bottom, M.top])
      : linearScale(yDom, [h - M.bottom, M.top]);
    return { x, y };
  };
  const showMilestones = milestones ?? logX;
  const xNameRuns: MathRun[] | undefined = xName
    ? [...xName]
    : xLabel !== undefined
      ? xLabel.length === 1
        ? [mv(xLabel)]
        : [sans(xLabel)]
      : logX
        ? [sans('iteration '), mv('k'), { t: ' ≥ 1', style: 'main' }]
        : [mv('k')];

  const { canvasRef, size } = useCanvas((ctx, s) => {
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
      xInteger: xInteger ?? (!logX && !series.some((q) => q.x)),
      xName: xNameRuns,
      yName: yName ?? (yLabel ? [sans(yLabel)] : undefined),
      fontSize: compact ? 11 : undefined,
      xTicks: compact ? 3 : undefined,
      yTicks: compact ? 4 : undefined,
    });
    const clampY = (v: number) => Math.max(frame.top - 4, Math.min(frame.bottom, y(v)));
    const ok = (v: number | null): v is number =>
      v !== null && Number.isFinite(v) && (!logY || v > 0);
    const okX = (xv: number) => Number.isFinite(xv) && (!logX || xv > 0);

    // Slope guides (behind the data).
    if (logX && logY) {
      for (const g of slopes) {
        const [a, b, c] = slopeTriangle(
          g,
          x.domain as [number, number],
          y.domain as [number, number],
        );
        const P = (p: [number, number]) => [x(p[0]), y(p[1])] as const;
        const [ax, ay] = P(a),
          [bx, by] = P(b),
          [cx, cy] = P(c);
        ctx.save();
        ctx.strokeStyle = colors.text2;
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.lineTo(cx, cy);
        ctx.closePath();
        ctx.stroke();
        ctx.globalAlpha = 1;
        drawMath(ctx, [{ t: '1', style: 'main' }], (ax + bx) / 2, by + (cy < by ? 12 : -5), {
          size: 12,
          align: 'center',
          color: colors.text2,
          halo: colors.halo,
        });
        drawMath(
          ctx,
          [{ t: g.label ?? String(Math.abs(g.slope)), style: 'main' }],
          bx + 5,
          (by + cy) / 2,
          {
            size: 12,
            baseline: 'middle',
            color: colors.text2,
            halo: colors.halo,
          },
        );
        ctx.restore();
      }
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(
      frame.left - 6,
      frame.top - 6,
      frame.right - frame.left + 12,
      frame.bottom - frame.top + 12,
    );
    ctx.clip();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const pass of ['future', 'past'] as const) {
      for (const sr of series) {
        const color = colors.series[sr.slot % colors.series.length];
        const kEnd = Math.min(t, sr.values.length - 1);
        ctx.strokeStyle = color;
        ctx.globalAlpha = pass === 'future' ? 0.16 : 1;
        ctx.lineWidth = pass === 'future' ? 1.25 : compact ? 1.75 : 2;
        // Halo under the past line keeps crossings readable.
        const n = sr.values.length;
        const from = pass === 'past' ? 0 : Math.floor(kEnd);
        const to = pass === 'past' ? Math.floor(kEnd) : n - 1;
        const trace = () => {
          ctx.beginPath();
          let pen = false;
          // Long series: skip points closer than 0.75 px (keeps redraws cheap at 20,000 steps).
          let lastPx = -Infinity;
          for (let k = from; k <= to; k++) {
            const v = sr.values[k];
            const xv = xOf(sr, k);
            if (!ok(v) || !okX(xv)) {
              pen = false;
              continue;
            }
            const px = x(xv),
              py = clampY(v);
            if (pen && px - lastPx < 0.75 && k !== to) continue;
            if (pen) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            lastPx = px;
            pen = true;
          }
          // Partial segment to the interpolated playhead.
          if (pass === 'past' && kEnd % 1 > 0 && !sr.x) {
            const k0 = Math.floor(kEnd),
              v0 = sr.values[k0],
              v1 = sr.values[k0 + 1];
            if (ok(v0) && ok(v1) && okX(kEnd)) {
              const u = kEnd - k0;
              const vy = logY
                ? 10 ** (Math.log10(v0) + (Math.log10(v1) - Math.log10(v0)) * u)
                : v0 + (v1 - v0) * u;
              ctx.lineTo(x(kEnd), clampY(vy));
            }
          }
        };
        if (pass === 'past') {
          trace();
          ctx.strokeStyle = colors.halo;
          ctx.lineWidth = (compact ? 1.75 : 2) + 2.4;
          ctx.globalAlpha = 0.85;
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.strokeStyle = color;
          ctx.lineWidth = compact ? 1.75 : 2;
        }
        trace();
        ctx.stroke();
        if (pass === 'past') {
          ctx.globalAlpha = 1;
          // Milestone diamonds at k = 10, 100, ….
          if (showMilestones && !sr.x)
            for (const km of milestoneIndices(sr.values.length)) {
              if (km > kEnd) break;
              const v = sr.values[km];
              if (!ok(v)) continue;
              diamond(ctx, x(km), clampY(v), compact ? 3.4 : 4, color, colors.halo);
            }
          const k0 = Math.floor(kEnd);
          const v = sr.values[k0];
          const xv = xOf(sr, k0);
          if (ok(v) && okX(xv)) {
            const atEnd = k0 >= sr.values.length - 1;
            const px = x(xv),
              py = clampY(v);
            ctx.fillStyle = colors.halo;
            ctx.beginPath();
            ctx.arc(px, py, 4.6, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(px, py, 3.1, 0, Math.PI * 2);
            if (atEnd && sr.end === 'stopped') {
              ctx.strokeStyle = color;
              ctx.lineWidth = 1.6;
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
    ctx.globalAlpha = 1;

    // Rate annotations (Newsreader italic, text-2) once a run has finished drawing.
    for (const sr of series) {
      if (!sr.rate || t < sr.values.length - 1) continue;
      const k = sr.values.length - 1;
      const v = sr.values[k];
      const xv = xOf(sr, k);
      if (!ok(v) || !okX(xv)) continue;
      const px = x(xv),
        py = clampY(v);
      // Above the end dot (clear of the k-axis baseline and the dot), inside the plot frame.
      const size = compact ? 11.5 : 13;
      const runs = [{ t: sr.rate, style: 'serif-italic' as const }];
      const w = measureMath(ctx, runs, size);
      const left = px + 8 + w > frame.right - 4;
      const tx = left ? Math.max(frame.left + 4 + w, px - 6) : Math.max(frame.left + 4, px + 6);
      let ty = py - 14;
      if (ty - size / 2 < frame.top + 2) ty = py + 14;
      ty = Math.min(frame.bottom - size / 2 - 4, Math.max(frame.top + size / 2 + 2, ty));
      drawMath(ctx, runs, tx, ty, {
        size,
        align: left ? 'right' : 'left',
        baseline: 'middle',
        color: colors.text2,
        halo: colors.halo,
      });
    }

    // Playhead (index → x of the first series that has it).
    if (Number.isFinite(t) && !series.some((q) => q.x)) {
      const tk = logX ? Math.max(1, t) : t;
      const px = crisp(x(Math.min(tk, xDom[1])), s.dpr);
      if (px >= frame.left && px <= frame.right) {
        ctx.strokeStyle = colors.playhead;
        ctx.lineWidth = 1 / s.dpr;
        ctx.setLineDash(compact ? [3, 3] : []);
        ctx.beginPath();
        ctx.moveTo(px, frame.top - 4);
        ctx.lineTo(px, frame.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (hoverK !== null) {
      const hx = crisp(x(Math.max(logX ? 1 : 0, hoverK)), s.dpr);
      ctx.strokeStyle = colors.crosshair;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, frame.top);
      ctx.lineTo(hx, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const s2 of series) {
        const v = s2.values[hoverK];
        if (!ok(v ?? null)) continue;
        ctx.fillStyle = colors.series[s2.slot % colors.series.length];
        ctx.beginPath();
        ctx.arc(x(xOf(s2, hoverK)), clampY(v as number), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });

  const kFromEvent = (e: { clientX: number }) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || series.some((q) => q.x)) return null;
    const { x } = scales(r.width, r.height);
    const k = Math.round(x.invert(e.clientX - r.left));
    return Math.max(logX ? 1 : 0, Math.min(maxK, k));
  };

  const showLegend = legend ?? series.length >= 2;
  const tipLeft =
    hoverK !== null && size.width ? scales(size.width, size.height).x(Math.max(1, hoverK)) : 0;

  return (
    <div
      className={className}
      style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0, height: '100%' }}
    >
      {showLegend && (
        <SeriesLegend
          items={series.map((s) => ({ slot: s.slot, label: s.label, count: s.count }))}
        />
      )}
      <div
        ref={box}
        className={styles.chart}
        style={{ flex: 1, cursor: onSeek ? 'pointer' : undefined }}
        role="img"
        aria-label={ariaLabel ?? `${yLabel} versus iteration`}
        onPointerMove={(e) => {
          const k = kFromEvent(e);
          setHoverK(k);
          onHover?.(k);
        }}
        onPointerLeave={() => {
          setHoverK(null);
          onHover?.(null);
        }}
        onClick={(e) => {
          const k = kFromEvent(e);
          if (k !== null) onSeek?.(k);
        }}
      >
        <canvas ref={canvasRef} />
        {empty && (
          <div className={styles.empty} role="status">
            No finite values to plot: every run failed or has no iterations.
          </div>
        )}
        {hoverK !== null && !empty && !compact && (
          <div
            className={styles.tooltip}
            style={{
              left: Math.min(Math.max(8, tipLeft + 12), Math.max(8, size.width - 170)),
              top: 8,
            }}
          >
            <div className={styles.tooltipTitle}>k = {int(hoverK)}</div>
            {series.map((s) => (
              <div key={s.label} className={styles.tooltipRow}>
                <span className={styles.tooltipKey}>
                  <Swatch slot={s.slot} size={8} />
                  {s.label}
                </span>
                <b>
                  <SciText
                    text={hoverK < s.values.length ? sci(s.values[hoverK] ?? null, 3) : '—'}
                  />
                </b>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function diamond(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
  halo: string,
) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = halo;
  ctx.stroke();
  ctx.fillStyle = fill;
  ctx.fill();
}
