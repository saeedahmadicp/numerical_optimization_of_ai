/**
 * The iterate as a function of its index, x_k(i) against i = 1 … n, for systems larger than
 * 2 × 2 (no plane to draw in). The exact solution x⋆ is the ink reference (hollow circles); each
 * method's current iterate is a polyline in its color, its last few iterates fade behind it.
 * On the Poisson matrix this is the discrete solution u(x) itself, and Jacobi's slow, smooth
 * approach to it is the low-frequency error that classical iterations damp worst.
 *
 * The y-range covers x⋆, x₀ and every iterate within 4× of them. A divergent iterate leaves it:
 * its lines exit through the frame (clipped, never pinned to the edge), each component outside
 * the view is an outward double chevron on the edge, and the largest one is labelled with its
 * true value ("x₂ = −1937  below the view").
 */
import { useMemo } from 'react';
import type { Vector } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import {
  drawAxes,
  drawMath,
  linearScale,
  mathBold,
  mathMain,
  mathSup,
  niceDomain,
  useCanvas,
} from '../../viz';
import { offScaleIndices, offText } from './model';
import styles from './LinalgLab.module.css';

export interface ComponentSeries {
  slot: number;
  label: string;
  /** Iterates x_0 … x_K (null where the method has no iterate yet: direct methods). */
  iterates: readonly (Vector | null)[];
  /** Continuous local playhead. */
  t: number;
  muted?: boolean;
}

export interface ComponentsPlotProps {
  solution: Vector | null;
  series: readonly ComponentSeries[];
  ease: boolean;
  ariaLabel: string;
  /**
   * 'error': the series hold e_k/‖e_k‖∞ (each iterate's error, scaled to unit size), so the
   * shape of the error stays visible however small it gets; x⋆ is then the zero line.
   */
  mode?: 'x' | 'error';
}

const TRAIL = 5;

function lerp(a: Vector, b: Vector, u: number): Vector {
  return a.map((v, i) => v + (b[i] - v) * u);
}

function at(s: ComponentSeries, ease: boolean): { now: Vector | null; k: number } {
  const last = s.iterates.length - 1;
  const t = Math.max(0, Math.min(s.t, last));
  const k = Math.floor(t);
  const a = s.iterates[k];
  if (!a) {
    // Direct methods: the solution appears on their last step only.
    return { now: null, k };
  }
  const b = s.iterates[Math.min(k + 1, last)];
  const u = t - k;
  if (!ease || !b || u <= 0) return { now: a, k };
  const e = u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
  return { now: lerp(a, b, e), k };
}

export function ComponentsPlot({
  solution,
  series,
  ease,
  ariaLabel,
  mode = 'x',
}: ComponentsPlotProps) {
  const colors = useChartColors();
  const n =
    solution?.length ??
    series.find((s) => s.iterates.some(Boolean))?.iterates.find(Boolean)?.length ??
    1;

  // y-range: the solution and the start, widened to every iterate that stays within 4× of them.
  const yDom = useMemo<[number, number]>(() => {
    if (mode === 'error') return [-1.15, 1.15];
    const ref = [...(solution ?? []), 0];
    for (const s of series) {
      const x0 = s.iterates.find(Boolean);
      if (x0) ref.push(...x0);
    }
    const lo0 = Math.min(...ref),
      hi0 = Math.max(...ref);
    const pad = Math.max(1e-9, hi0 - lo0);
    let lo = lo0,
      hi = hi0;
    for (const s of series)
      for (const x of s.iterates)
        if (x)
          for (const v of x)
            if (Number.isFinite(v) && v > lo0 - 4 * pad && v < hi0 + 4 * pad) {
              lo = Math.min(lo, v);
              hi = Math.max(hi, v);
            }
    return niceDomain(lo - 0.06 * (hi - lo || 1), hi + 0.06 * (hi - lo || 1));
  }, [solution, series, mode]);

  const { canvasRef } = useCanvas((ctx, s) => {
    const M = { left: 48, right: 16, top: 30, bottom: 34 };
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    const x = linearScale([0.5, n + 0.5], [frame.left, frame.right]);
    const y = linearScale(yDom, [frame.bottom, frame.top]);
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      xInteger: true,
      xTicks: Math.min(n, 10),
      xName: [mathMain('index '), { t: 'i', style: 'italic' }],
      yName:
        mode === 'error'
          ? [
              { t: 'e', style: 'italic' },
              { t: 'i', style: 'italic', script: 'sub' },
              { t: ' / ‖', style: 'main' },
              { t: 'e', style: 'bold' },
              { t: '‖', style: 'main' },
              { t: '∞', style: 'main', script: 'sub' },
            ]
          : [
              { t: 'x', style: 'italic' },
              { t: 'i', style: 'italic', script: 'sub' },
            ],
    });
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Error mode: x⋆ is the zero line.
    if (mode === 'error') {
      ctx.strokeStyle = colors.text;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(frame.left, y(0));
      ctx.lineTo(frame.right, y(0));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      drawMath(ctx, [mathBold('x'), mathSup('⋆')], frame.right - 6, y(0) - 7, {
        size: 13,
        align: 'right',
        color: colors.text,
        halo: colors.halo,
      });
    }
    // x⋆: hollow ink circles joined by a hairline.
    if (solution && mode === 'x') {
      ctx.strokeStyle = colors.text;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      solution.forEach((v, i) => (i ? ctx.lineTo(x(i + 1), y(v)) : ctx.moveTo(x(i + 1), y(v))));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      for (let i = 0; i < n; i++) {
        ctx.beginPath();
        ctx.arc(x(i + 1), y(solution[i]), 5, 0, Math.PI * 2);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = colors.text;
        ctx.stroke();
      }
      const iMax = solution.indexOf(Math.max(...solution));
      drawMath(ctx, [mathBold('x'), mathSup('⋆')], x(iMax + 1) + 9, y(solution[iMax]) - 9, {
        size: 13,
        color: colors.text,
        halo: colors.halo,
      });
    }

    // Lines are clipped to the frame: a component that leaves the y-range exits through the
    // edge (never pinned to it), and its value is reported by a chevron and a label below.
    const inView = (v: number) => Number.isFinite(v) && v >= yDom[0] && v <= yDom[1];
    const line = (pts: Vector, color: string, width: number, alpha: number, dots: boolean) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(frame.left, frame.top - 1, frame.right - frame.left, frame.bottom - frame.top + 2);
      ctx.clip();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      let pen = false;
      pts.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          pen = false;
          return;
        }
        // Keep the canvas numbers sane for huge values (the clip does the rest).
        const py = Math.max(frame.top - 4000, Math.min(frame.bottom + 4000, y(v)));
        if (pen) ctx.lineTo(x(i + 1), py);
        else ctx.moveTo(x(i + 1), py);
        pen = true;
      });
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = width + 2.4;
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
      if (dots)
        pts.forEach((v, i) => {
          if (!inView(v)) return;
          ctx.beginPath();
          ctx.arc(x(i + 1), y(v), 3.2, 0, Math.PI * 2);
          ctx.fillStyle = colors.halo;
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x(i + 1), y(v), 2.4, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        });
      ctx.restore();
    };

    /** Outward chevrons on the frame edge for every component outside the view. */
    const offScale = (pts: Vector, color: string, alpha: number, labelIt: boolean) => {
      let worst = -1;
      pts.forEach((v, i) => {
        if (inView(v)) return;
        const up = !(v < yDom[0]); // NaN and +∞ point up
        const px = x(i + 1);
        const py = up ? frame.top + 1 : frame.bottom - 1;
        const d = up ? -1 : 1;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 4;
        ctx.strokeStyle = colors.halo;
        const chevron = (off: number) => {
          ctx.beginPath();
          ctx.moveTo(px - 5, py - d * (off + 0));
          ctx.lineTo(px, py + d * (5 - off));
          ctx.lineTo(px + 5, py - d * (off + 0));
        };
        chevron(0);
        ctx.stroke();
        chevron(5);
        ctx.stroke();
        ctx.lineWidth = 1.75;
        ctx.strokeStyle = color;
        chevron(0);
        ctx.stroke();
        chevron(5);
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (worst < 0 || !(Math.abs(pts[worst]) >= Math.abs(v))) worst = i;
      });
      if (!labelIt || worst < 0) return;
      const v = pts[worst];
      const up = !(v < yDom[0]);
      const px = x(worst + 1);
      const right = px < (frame.left + frame.right) / 2;
      drawMath(
        ctx,
        [
          { t: 'x', style: 'italic' },
          { t: String(worst + 1), style: 'main', script: 'sub' },
          { t: ` = ${offText(v)}`, style: 'main' },
          { t: up ? '  above the view' : '  below the view', style: 'serif-italic' },
        ],
        right ? px + 10 : px - 10,
        up ? frame.top + 16 : frame.bottom - 10,
        { size: 12, align: right ? 'left' : 'right', color, halo: colors.halo },
      );
    };

    for (const sr of series) {
      const color = colors.series[sr.slot % colors.series.length];
      const { now, k } = at(sr, ease);
      const fade = sr.muted ? 0.35 : 1;
      // The previous iterates fade out behind the current one.
      const newest = now === sr.iterates[k] ? k - 1 : k;
      for (let j = TRAIL - 1; j >= 0; j--) {
        const old = newest - j >= 0 ? sr.iterates[newest - j] : null;
        if (old) line(old, color, 1, (0.1 + 0.06 * (TRAIL - 1 - j)) * fade, false);
      }
      if (now) {
        line(now, color, 2, fade, n <= 30);
        offScale(now, color, fade, !sr.muted);
      }
    }
    ctx.restore();
  });

  // Say which components of the current iterates are outside the view, with their values.
  const offNote = series
    .map((sr) => {
      const it = sr.iterates[Math.max(0, Math.min(Math.floor(sr.t), sr.iterates.length - 1))];
      if (!it) return '';
      const off = offScaleIndices(it, yDom[0], yDom[1]);
      if (off.length === 0) return '';
      return `${sr.label}: ${off
        .slice(0, 4)
        .map((i) => `x${i + 1} = ${offText(it[i])}`)
        .join(', ')}${off.length > 4 ? ', …' : ''} outside the view`;
    })
    .filter(Boolean)
    .join('; ');

  return (
    <div
      className={styles.canvasBox}
      role="img"
      aria-label={offNote ? `${ariaLabel} ${offNote}.` : ariaLabel}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
