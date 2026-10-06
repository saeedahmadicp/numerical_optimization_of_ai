/**
 * Error measure against k on a log axis, one curve per method, on the lab's clock — with the
 * rates the theory predicts drawn as dashed guides in each method's color:
 *   stationary methods  ‖e_k‖ ≈ C·ρ(G)^k: a line of slope log ρ through the last iterate, so the
 *                       curve visibly runs parallel to it once the transient has died out;
 *   CG / PCG            the A-norm bound 2((√κ − 1)/(√κ + 1))^k ‖e₀‖_A (and the line k = n,
 *                       where CG terminates in exact arithmetic);
 *   steepest descent    the A-norm bound ((κ − 1)/(κ + 1))^k ‖e₀‖_A.
 * Direct methods have one value, at their last step (the residual of the computed x).
 */
import { useMemo, useRef, useState } from 'react';
import { int, sci } from '../../core/format';
import { useChartColors } from '../../ui/theme';
import { Formula, Swatch } from '../../ui/components';
import {
  SeriesLegend,
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  logExtent,
  logScale,
  useCanvas,
  type MathRun,
} from '../../viz';
import { TICK_SIZE } from '../../viz/axes';
import styles from './LinalgLab.module.css';

export interface ResidualSeries {
  label: string;
  slot: number;
  values: readonly (number | null)[];
  end: 'converged' | 'stopped';
  count: number;
  muted?: boolean;
  /** TeX shown after the count in the legend (the rate its guide line is drawn with). */
  note?: string;
}

export interface RateGuide {
  slot: number;
  /** Value at k = 0 of the guide line g(k) = v0·rate^k. */
  v0: number;
  rate: number;
  /** k range the guide covers. */
  from: number;
  to: number;
}

export interface ResidualChartProps {
  series: readonly ResidualSeries[];
  guides: readonly RateGuide[];
  /** Vertical marker (k = n for CG's finite termination). */
  marker?: { k: number; label: string } | null;
  t: number;
  yName: readonly MathRun[];
  ariaLabel: string;
  onSeek?: (k: number) => void;
}

const ok = (v: number | null | undefined): v is number =>
  v !== null && v !== undefined && Number.isFinite(v) && v > 0;

export function ResidualChart({
  series,
  guides,
  marker,
  t,
  yName,
  ariaLabel,
  onSeek,
}: ResidualChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hoverK, setHoverK] = useState<number | null>(null);
  const M = { left: 48, right: 16, top: 16, bottom: 32 };
  const maxK = Math.max(1, ...series.map((s) => s.values.length - 1));
  const yDom = useMemo<[number, number]>(() => {
    const all = series.flatMap((s) => s.values.filter(ok));
    if (all.length === 0) return [1e-16, 1];
    const [lo, hi] = logExtent(all);
    return [Math.max(lo, 1e-18), hi];
  }, [series]);
  const empty = !series.some((s) => s.values.some((v) => ok(v) || v === 0));
  const directOnly =
    series.length > 0 && series.every((s) => s.values.filter((v) => v !== null).length <= 1);
  const emptyNow = !series.some((s) => s.values.some((v, k) => k <= t && (ok(v) || v === 0)));

  const scales = (w: number, h: number) => ({
    x: linearScale([0, maxK], [M.left, w - M.right]),
    y: logScale(yDom, [h - M.bottom, M.top]),
  });

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
      xInteger: true,
      // Label every 2–3 decades (3–5 labels), not 2 ticks across 11 decades.
      yTicks: Math.max(3, Math.min(5, Math.round((frame.bottom - frame.top) / 30))),
      xName: [{ t: 'k', style: 'italic' }],
      yName,
      fontSize: 10.5,
    });
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
    const Y = (v: number) => Math.max(frame.top - 4, Math.min(frame.bottom + 4, y(v)));

    // k = n marker (behind everything).
    if (marker && marker.k <= maxK) {
      const px = crisp(x(marker.k), s.dpr);
      ctx.strokeStyle = colors.text3;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      drawMath(ctx, [{ t: marker.label, style: 'serif-italic' }], px + 4, frame.top + 11, {
        size: 12,
        color: colors.text2,
        halo: colors.halo,
      });
    }

    // Rate guides.
    const [ylo, yhi] = yDom;
    for (const g of guides) {
      if (!(g.v0 > 0) || !(g.rate > 0)) continue;
      const color = colors.series[g.slot % colors.series.length];
      const lr = Math.log10(g.rate);
      // Clip the line to the y-range.
      let a = g.from,
        b = Math.min(g.to, maxK);
      const kAt = (v: number) => (Math.log10(v) - Math.log10(g.v0)) / lr;
      if (lr < 0) {
        a = Math.max(a, kAt(yhi * 3));
        b = Math.min(b, kAt(ylo));
      } else if (lr > 0) {
        a = Math.max(a, kAt(ylo));
        b = Math.min(b, kAt(yhi * 3));
      }
      if (!(b > a)) continue;
      const val = (k: number) => g.v0 * 10 ** (lr * k);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x(a), Y(val(a)));
      ctx.lineTo(x(b), Y(val(b)));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    for (const pass of ['future', 'past'] as const) {
      for (const sr of series) {
        const color = colors.series[sr.slot % colors.series.length];
        const n = sr.values.length;
        const kEnd = Math.min(t, n - 1);
        const from = pass === 'past' ? 0 : Math.floor(kEnd);
        const to = pass === 'past' ? Math.floor(kEnd) : n - 1;
        ctx.globalAlpha = (pass === 'future' ? 0.16 : 1) * (sr.muted ? 0.4 : 1);
        ctx.beginPath();
        let pen = false;
        let count = 0;
        for (let k = from; k <= to; k++) {
          const v = sr.values[k];
          if (!ok(v)) {
            pen = false;
            continue;
          }
          if (pen) ctx.lineTo(x(k), Y(v));
          else ctx.moveTo(x(k), Y(v));
          pen = true;
          count++;
        }
        if (pass === 'past' && kEnd % 1 > 0) {
          const k0 = Math.floor(kEnd);
          const v0 = sr.values[k0],
            v1 = sr.values[k0 + 1];
          if (ok(v0) && ok(v1)) {
            const u = kEnd - k0;
            ctx.lineTo(x(kEnd), Y(10 ** (Math.log10(v0) + (Math.log10(v1) - Math.log10(v0)) * u)));
          }
        }
        if (pass === 'past') {
          ctx.strokeStyle = colors.halo;
          ctx.lineWidth = 4.4;
          ctx.stroke();
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = pass === 'future' ? 1.25 : 2;
        ctx.stroke();
        // An exactly zero residual cannot sit on a log axis: a triangle on the floor, "= 0".
        if (pass === 'past')
          for (let k = from; k <= to; k++)
            if (sr.values[k] === 0) {
              const px = x(k),
                py = frame.bottom - 2;
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.moveTo(px - 5, py - 8);
              ctx.lineTo(px + 5, py - 8);
              ctx.lineTo(px, py);
              ctx.closePath();
              ctx.fill();
              const right = px > frame.right - 40;
              drawMath(ctx, [{ t: '= 0', style: 'main' }], right ? px - 8 : px + 7, py - 2, {
                size: TICK_SIZE + 1,
                align: right ? 'right' : 'left',
                color: colors.text2,
                halo: colors.halo,
              });
            }
        // A single value (direct methods): a diamond.
        if (count <= 1 && pass === 'past') {
          for (let k = from; k <= to; k++) {
            const v = sr.values[k];
            if (!ok(v)) continue;
            diamond(ctx, x(k), Y(v), 5, color, colors.halo);
          }
        }
        ctx.globalAlpha = 1;
        if (pass === 'past') {
          const k0 = Math.floor(kEnd);
          const v = sr.values[k0];
          if (ok(v)) {
            const atEnd = k0 >= n - 1;
            ctx.fillStyle = colors.halo;
            ctx.beginPath();
            ctx.arc(x(k0), Y(v), 4.6, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x(k0), Y(v), 3.1, 0, Math.PI * 2);
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

    if (Number.isFinite(t)) {
      const px = crisp(x(Math.min(t, maxK)), s.dpr);
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1 / s.dpr;
      ctx.beginPath();
      ctx.moveTo(px, frame.top - 4);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
    }
    if (hoverK !== null) {
      const hx = crisp(x(hoverK), s.dpr);
      ctx.strokeStyle = colors.crosshair;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, frame.top);
      ctx.lineTo(hx, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  const kFrom = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const { x } = scales(r.width, r.height);
    return Math.max(0, Math.min(maxK, Math.round(x.invert(clientX - r.left))));
  };
  const tipLeft = hoverK !== null && size.width ? scales(size.width, size.height).x(hoverK) : 0;

  return (
    <div className={styles.chartCol}>
      {series.length > 0 && (
        <SeriesLegend
          className={styles.legendPad}
          items={series.map((s) => ({
            slot: s.slot,
            label: s.label,
            count: s.count,
            extra: s.note ? (
              <span className={styles.legendTex} title="Rate the theory predicts (dashed)">
                <span
                  className={styles.legendDash}
                  style={{ borderColor: `var(--series-${(s.slot % 4) + 1})` }}
                  aria-hidden="true"
                />
                <Formula tex={s.note} />
              </span>
            ) : undefined,
          }))}
        />
      )}
      <div
        ref={box}
        className={styles.canvasBox}
        role="img"
        aria-label={ariaLabel}
        style={{ cursor: onSeek ? 'pointer' : undefined }}
        onPointerMove={(e) => setHoverK(kFrom(e.clientX))}
        onPointerLeave={() => setHoverK(null)}
        onClick={(e) => {
          const k = kFrom(e.clientX);
          if (k !== null) onSeek?.(k);
        }}
      >
        <canvas ref={canvasRef} />
        {(empty || (directOnly && emptyNow)) && (
          <div className={styles.chartEmpty} role="status">
            <span>
              {directOnly
                ? 'A direct method has one residual: that of its x, after the last step.'
                : 'No finite values to plot.'}
            </span>
          </div>
        )}
        {hoverK !== null && !empty && (
          <div
            className={styles.tooltip}
            style={{
              left: Math.min(Math.max(8, tipLeft + 12), Math.max(8, size.width - 180)),
              top: 6,
            }}
          >
            <div className={styles.tooltipTitle}>k = {int(hoverK)}</div>
            {series.map((s) => (
              <div key={s.label} className={styles.tooltipRow}>
                <span>
                  <Swatch slot={s.slot} size={8} /> {s.label}
                </span>
                <b>{hoverK < s.values.length ? sci(s.values[hoverK] ?? null, 3) : '—'}</b>
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
