/**
 * Theil–Sen's evidence: the histogram of all pairwise slopes (yⱼ − yᵢ)/(xⱼ − xᵢ), the median
 * (the estimate β₁) and, for comparison, the least-squares slope. Slopes beyond the 2nd/98th
 * percentiles are counted at the edges rather than stretching the axis.
 */
import { useChartColors } from '../../ui/theme';
import { sig } from '../../core/format';
import {
  drawAxes,
  drawMath,
  linearScale,
  mathMain as mm,
  mathSub as sub,
  mathVar as mv,
  niceDomain,
  useCanvas,
} from '../../viz';
import { tickFont } from '../../viz/axes';
import styles from './RegressionLab.module.css';

export interface SlopeStripProps {
  /** Sorted pairwise slopes (Step.info.slopes). */
  slopes: readonly number[];
  median: number;
  color: string;
  /** Least-squares slope of the same data (ink, dashed). */
  olsSlope?: number | null;
  ariaLabel: string;
}

const M = { left: 40, right: 16, top: 30, bottom: 34 };

export function SlopeStrip({ slopes, median, color, olsSlope, ariaLabel }: SlopeStripProps) {
  const colors = useChartColors();
  const n = slopes.length;
  const q = (p: number) => slopes[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))];
  // The bulk: the median ± 4 interquartile ranges (and the least-squares slope).
  const iqr = q(0.75) - q(0.25) || Math.abs(median) * 0.1 || 1;
  const lo0 = Math.min(Math.max(q(0), median - 4 * iqr), median, olsSlope ?? Infinity);
  const hi0 = Math.max(Math.min(q(1), median + 4 * iqr), median, olsSlope ?? -Infinity);
  const span = hi0 - lo0 || Math.max(1, Math.abs(median));
  const xDom = niceDomain(lo0 - 0.06 * span, hi0 + 0.06 * span);
  const BINS = 30;
  const width = (xDom[1] - xDom[0]) / BINS;
  const counts = new Array<number>(BINS).fill(0);
  let below = 0,
    above = 0;
  for (const s of slopes) {
    if (s < xDom[0]) below++;
    else if (s >= xDom[1]) above++;
    else counts[Math.min(BINS - 1, Math.floor((s - xDom[0]) / width))]++;
  }
  const yDom = niceDomain(0, Math.max(1, ...counts) * 1.1);

  const { canvasRef } = useCanvas((ctx, s) => {
    const x = linearScale(xDom, [M.left, s.width - M.right]);
    const y = linearScale(yDom, [s.height - M.bottom, M.top]);
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
      yTicks: 3,
      xTicks: 5,
      xName: [mm('pairwise slope')],
      yName: [mm('pairs')],
    });
    counts.forEach((c, i) => {
      if (!c) return;
      const x0 = x(xDom[0] + i * width) + 0.5,
        x1 = x(xDom[0] + (i + 1) * width) - 0.5;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.32;
      ctx.fillRect(x0, y(c), x1 - x0, frame.bottom - y(c));
      ctx.globalAlpha = 1;
    });
    const vline = (v: number, stroke: string, dash: number[], w: number) => {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = w;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(Math.round(x(v)) + 0.5, frame.top - 4);
      ctx.lineTo(Math.round(x(v)) + 0.5, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    if (olsSlope !== null && olsSlope !== undefined && Number.isFinite(olsSlope)) {
      vline(olsSlope, colors.text3, [3, 3], 1.25);
      drawMath(ctx, [mm('LS '), mv('β'), sub('1')], x(olsSlope), frame.top - 8, {
        size: 11.5,
        align: x(olsSlope) < x(median) ? 'right' : 'left',
        color: colors.text3,
        halo: colors.halo,
      });
    }
    vline(median, color, [], 2);
    drawMath(ctx, [mv('β'), sub('1'), mm(` = med = ${sig(median, 4)}`)], x(median), frame.top - 8, {
      size: 12,
      align:
        olsSlope !== null && olsSlope !== undefined && x(olsSlope) < x(median) ? 'left' : 'right',
      color: colors.text,
      halo: colors.halo,
    });
    ctx.font = tickFont(colors);
    ctx.fillStyle = colors.text3;
    if (below) {
      ctx.textAlign = 'left';
      ctx.fillText(`◂ ${below}`, frame.left + 4, frame.top + 10);
    }
    if (above) {
      ctx.textAlign = 'right';
      ctx.fillText(`${above} ▸`, frame.right - 4, frame.top + 10);
    }
  });

  return (
    <div className={styles.chartBox}>
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
    </div>
  );
}
