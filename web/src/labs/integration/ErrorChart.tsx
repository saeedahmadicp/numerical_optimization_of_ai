/**
 * Error versus the number of nodes n on log–log axes, one series per method, synced to the
 * playhead. A composite rule of order p is a line of slope −p (h ∝ 1/n): each series carries a
 * dashed guide of its theoretical slope through its last resolved point, named O(hᵖ) in the
 * legend beside the observed order p̂ (a dashed key in the series color), so an
 * order drop (√x: every rule of order ≥ 2 falls to h^{3/2}) shows as a curve leaving its guide.
 * Gauss–Legendre is marked "spectral" (its error curve bends down), Monte Carlo has the N^{−1/2}
 * guide. The method's own error estimate ε̂ can be drawn dotted beside the true error.
 */
import { useRef, useState } from 'react';
import { useChartColors } from '../../ui/theme';
import { Formula, Swatch } from '../../ui/components';
import { sci, sig } from '../../core/format';
import {
  crisp,
  drawAxes,
  drawMath,
  logExtent,
  logScale,
  mathMain,
  mathVar,
  SeriesLegend,
  useCanvas,
  type MathRun,
} from '../../viz';
import { LABEL_SIZE } from '../../viz/axes';
import styles from './IntegrationLab.module.css';

export interface ErrorSeries {
  label: string;
  slot: number;
  /** Nodes used by the estimate of step k. */
  n: readonly (number | null)[];
  /** |I_k − I| (true error). */
  err: readonly (number | null)[];
  /** The method's own error estimate ε̂_k. */
  est?: readonly (number | null)[];
  /** Theoretical order p (guide of slope −p), 'spectral' (Gauss) or 'mc' (slope −1/2). */
  order?: number | 'spectral' | 'mc' | null;
  /** Local continuous playhead of this run. */
  t: number;
  /** Observed order p̂ at the playhead (legend). */
  observed?: number | null;
  /** Iterations of the run (the legend's count). */
  count?: number;
}

export interface ErrorChartProps {
  series: readonly ErrorSeries[];
  /** Rounding level ε·max(|I|, ∫|f|), drawn as a dashed floor. */
  floor: number;
  showEstimate: boolean;
  onSeek?: (seriesIndex: number, k: number) => void;
  ariaLabel: string;
}

const M = { left: 50, right: 46, top: 22, bottom: 34 };
/** TeX of a guide, named in the legend beside its series: O(hᵖ) or N^{−1/2}. */
function guideTex(order: number | 'mc'): string {
  if (order === 'mc') return 'N^{-1/2}';
  return order === 1 ? 'O(h)' : `O(h^{${order}})`;
}

/** A short dashed segment in the series color: the legend key of a guide. */
function GuideKey({ color }: { color: string }) {
  return (
    <svg width="16" height="8" viewBox="0 0 16 8" aria-hidden="true" className={styles.guideKey}>
      <line x1="1" y1="4" x2="15" y2="4" stroke={color} strokeWidth="1.6" strokeDasharray="4 2.5" />
    </svg>
  );
}

export function ErrorChart({ series, floor, showEstimate, onSeek, ariaLabel }: ErrorChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ i: number; k: number } | null>(null);

  const ok = (v: number | null | undefined): v is number =>
    v !== null && v !== undefined && Number.isFinite(v) && v > 0;
  const lo = floor / 10;
  const clampErr = (v: number | null | undefined) =>
    v === null || v === undefined || !Number.isFinite(v) ? null : Math.max(v, lo);

  const allN = series.flatMap((s) => s.n.filter(ok));
  const allE = series.flatMap((s) => [
    ...s.err.map(clampErr).filter(ok),
    ...(showEstimate && s.est ? s.est.map(clampErr).filter(ok) : []),
  ]);
  const xDom = logExtent(
    allN.length ? [Math.max(1, Math.min(...allN)), Math.max(...allN, 10)] : [1, 100],
  );
  // The rounding floor joins the range only when an error comes within three decades of it.
  const nearFloor = allE.some((v) => v < floor * 1e3);
  const yDom = logExtent(allE.length ? (nearFloor ? [...allE, floor] : allE) : [1e-16, 1]);

  const scales = (w: number, h: number) => ({
    x: logScale(xDom, [M.left, w - M.right]),
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
      xName: [mathVar('n')],
      // Every decade: the slopes are read against them (the default thins to every 2nd).
      xTicks: 99,
      yName: [
        mathMain('|'),
        mathVar('I'),
        { t: 'n', style: 'italic', script: 'sub' },
        mathMain(' − '),
        mathVar('I'),
        mathMain('|'),
      ],
    });

    // Rounding floor ε·max(|I|, ∫|f|).
    if (floor > yDom[0] && floor < yDom[1]) {
      const fy = crisp(y(floor), s.dpr);
      ctx.save();
      ctx.strokeStyle = colors.text3;
      ctx.globalAlpha = 0.6;
      ctx.setLineDash([2, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(frame.left, fy);
      ctx.lineTo(frame.right, fy);
      ctx.stroke();
      ctx.restore();
      drawMath(ctx, [{ t: 'rounding level', style: 'sans' }], frame.left + 6, fy - 6, {
        size: LABEL_SIZE,
        align: 'left',
        color: colors.text3,
        halo: colors.halo,
      });
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

    // Guides (behind the data).
    const labels: { runs: MathRun[]; x: number; y: number; align: 'left' | 'right' }[] = [];
    series.forEach((sr) => {
      if (sr.order === null || sr.order === undefined || sr.order === 'spectral') return;
      const p = sr.order === 'mc' ? 0.5 : sr.order;
      const pts = sr.n
        .map((n, k) => [n, sr.err[k]] as const)
        .filter((q): q is readonly [number, number] => ok(q[0]) && ok(q[1]));
      if (pts.length < 2) return;
      // Anchor: the last point clearly above the rounding level (the guide is static in time).
      const resolved = pts.filter((q) => q[1] > 30 * floor);
      const anchor = resolved.length ? resolved[resolved.length - 1] : pts[0];
      // The guide reaches a little past the data on both sides, so it shows even where the
      // curve lies exactly on it (the order is attained).
      const n0 = pts[0][0] / 1.5,
        n1 = pts[pts.length - 1][0] * 1.8;
      const g = (n: number) => anchor[1] * (n / anchor[0]) ** -p;
      ctx.save();
      ctx.strokeStyle = colors.series[sr.slot % colors.series.length];
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1.1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      const steps = 24;
      for (let i = 0; i <= steps; i++) {
        const n = n0 * (n1 / n0) ** (i / steps);
        const v = g(n);
        if (i) ctx.lineTo(x(n), y(v));
        else ctx.moveTo(x(n), y(v));
      }
      ctx.stroke();
      ctx.restore();
    });

    // Series: future faint, past solid with a halo, the current point a dot.
    for (const pass of ['future', 'past'] as const) {
      series.forEach((sr) => {
        const color = colors.series[sr.slot % colors.series.length];
        const kNow = Math.floor(sr.t + 1e-9);
        const draw = (vals: readonly (number | null)[], dotted: boolean) => {
          const from = pass === 'past' ? 0 : kNow;
          const to = pass === 'past' ? kNow : vals.length - 1;
          ctx.beginPath();
          let pen = false;
          for (let k = from; k <= to && k < vals.length; k++) {
            const n = sr.n[k];
            const v = clampErr(vals[k]);
            if (!ok(n) || !ok(v)) {
              pen = false;
              continue;
            }
            if (pen) ctx.lineTo(x(n), y(v));
            else ctx.moveTo(x(n), y(v));
            pen = true;
          }
          ctx.setLineDash(dotted ? [1.5, 3] : []);
          if (pass === 'past' && !dotted) {
            ctx.strokeStyle = colors.halo;
            ctx.lineWidth = 4.4;
            ctx.globalAlpha = 0.85;
            ctx.stroke();
          }
          ctx.strokeStyle = color;
          ctx.globalAlpha = pass === 'future' ? 0.18 : dotted ? 0.9 : 1;
          ctx.lineWidth = pass === 'future' ? 1.25 : dotted ? 1.6 : 2;
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.setLineDash([]);
        };
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        draw(sr.err, false);
        if (showEstimate && sr.est) draw(sr.est, true);
        if (pass !== 'past') return;
        // Point markers for short runs (few levels): one dot per level.
        const kMax = Math.min(kNow, sr.err.length - 1);
        if (sr.err.length <= 40)
          for (let k = 0; k < kMax; k++) {
            const n = sr.n[k],
              v = clampErr(sr.err[k]);
            if (!ok(n) || !ok(v)) continue;
            ctx.fillStyle = colors.halo;
            ctx.beginPath();
            ctx.arc(x(n), y(v), 3.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x(n), y(v), 2, 0, Math.PI * 2);
            ctx.fill();
          }
        const n = sr.n[kMax],
          v = clampErr(sr.err[kMax]);
        if (ok(n) && ok(v)) {
          const px = x(n),
            py = y(v);
          ctx.fillStyle = colors.halo;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(px, py, 3.4, 0, Math.PI * 2);
          ctx.fill();
          // Exact to rounding: a small down-chevron under the floor marker.
          if ((sr.err[kMax] ?? 1) < lo * 1.0001) {
            ctx.strokeStyle = colors.text2;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(px - 3.5, py + 7);
            ctx.lineTo(px, py + 10.5);
            ctx.lineTo(px + 3.5, py + 7);
            ctx.stroke();
          }
        }
        if (sr.order === 'spectral' && kMax >= sr.err.length - 1 && ok(n) && ok(v)) {
          const left = x(n) > frame.right - 80;
          labels.push({
            runs: [{ t: 'spectral', style: 'serif-italic' }],
            x: left ? x(n) - 9 : x(n) + 9,
            y: y(v) + 4,
            align: left ? 'right' : 'left',
          });
        }
      });
    }
    ctx.restore();
    for (const l of labels)
      drawMath(ctx, l.runs, l.x, l.y, {
        size: 12,
        align: l.align,
        color: colors.text2,
        halo: colors.halo,
      });

    if (hover) {
      const sr = series[hover.i];
      const n = sr?.n[hover.k],
        v = clampErr(sr?.err[hover.k]);
      if (ok(n) && ok(v)) {
        ctx.strokeStyle = colors.series[sr.slot % colors.series.length];
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x(n), y(v), 6.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  });

  /** The nearest plotted point within 14 px. */
  const nearest = (clientX: number, clientY: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const { x, y } = scales(r.width, r.height);
    let best: { i: number; k: number; d: number } | null = null;
    series.forEach((sr, i) =>
      sr.err.forEach((e, k) => {
        const n = sr.n[k],
          v = clampErr(e);
        if (!ok(n) || !ok(v)) return;
        const d = Math.hypot(x(n) - (clientX - r.left), y(v) - (clientY - r.top));
        if (d < 14 && (!best || d < best.d)) best = { i, k, d };
      }),
    );
    return best as { i: number; k: number; d: number } | null;
  };

  const hv = hover ? series[hover.i] : null;
  return (
    <div className={styles.errorChart}>
      <SeriesLegend
        className={styles.legend}
        items={series.map((s) => {
          const guide = typeof s.order === 'number' || s.order === 'mc';
          const observed = s.observed !== null && s.observed !== undefined;
          return {
            slot: s.slot,
            label: s.label,
            count: s.count,
            extra:
              guide || observed ? (
                <span className={styles.legendExtra}>
                  {guide && (
                    <span
                      className={styles.legendGuide}
                      title="Dashed guide: the theoretical rate, through the last resolved point"
                    >
                      <GuideKey color={colors.series[s.slot % colors.series.length]} />
                      <Formula tex={guideTex(s.order as number | 'mc')} />
                    </span>
                  )}
                  {observed && (
                    <span
                      className={styles.legendNum}
                      title="Observed order between the last two steps"
                    >
                      p̂ = {sig(s.observed as number, 3)}
                    </span>
                  )}
                </span>
              ) : undefined,
          };
        })}
      />
      <div
        ref={box}
        className={styles.errorCanvas}
        role="img"
        aria-label={ariaLabel}
        style={{ cursor: onSeek ? 'pointer' : undefined }}
        onPointerMove={(e) => {
          const p = nearest(e.clientX, e.clientY);
          setHover(p ? { i: p.i, k: p.k } : null);
        }}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const p = nearest(e.clientX, e.clientY);
          if (p) onSeek?.(p.i, p.k);
        }}
      >
        <canvas ref={canvasRef} />
        {hv && hover && (
          <div className={styles.chartTip}>
            <Swatch slot={hv.slot} size={8} /> {hv.label} · iteration {hover.k} ·{' '}
            <span className={styles.mono}>n = {hv.n[hover.k]}</span> ·{' '}
            <span className={styles.mono}>error {sci(hv.err[hover.k] ?? null, 3)}</span>
            {showEstimate && hv.est && (
              <>
                {' '}
                · <span className={styles.mono}>ε̂ {sci(hv.est[hover.k] ?? null, 3)}</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
