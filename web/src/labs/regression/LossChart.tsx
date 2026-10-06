/**
 * The loss each method puts on a residual, with the data's residuals at the playhead placed on
 * it: ρ(r) = r² (least squares), ρ_δ(r/σ̂) (Huber), |r| (LAD), and |r| against the leveled
 * error ±|h| (minimax). Point opacity is the IRLS weight, as on the data plane; the dashed curve
 * is the weight function w(r) on the right-hand axis [0, 1].
 */
import { useChartColors } from '../../ui/theme';
import {
  drawAxes,
  drawMath,
  linearScale,
  mathMain as mm,
  mathSub as sub,
  mathVar as mv,
  niceDomain,
  useCanvas,
  type MathRun,
} from '../../viz';
import { tickFont } from '../../viz/axes';
import { lossFunction, type Kind } from './model';
import styles from './RegressionLab.module.css';

export interface LossChartProps {
  kind: Kind;
  residuals: readonly number[];
  weights: readonly number[] | null;
  color: string;
  /** Huber: δ and σ̂. */
  huber?: { delta: number; scale: number } | null;
  /** Minimax: the leveled error h. */
  level?: number | null;
  /** LAD: median |r| (the weight normalization of the display). */
  ladMedian?: number | null;
  ariaLabel: string;
}

const M = { left: 44, right: 30, top: 24, bottom: 34 };

const LABELS: Record<Kind, MathRun[]> = {
  ols: [mv('ρ'), mm('('), mv('r'), mm(') = '), mv('r'), { t: '2', style: 'main', script: 'sup' }],
  poly: [mv('ρ'), mm('('), mv('r'), mm(') = '), mv('r'), { t: '2', style: 'main', script: 'sup' }],
  ridge: [mv('ρ'), mm('('), mv('r'), mm(') = '), mv('r'), { t: '2', style: 'main', script: 'sup' }],
  huber: [mv('ρ'), sub('δ', 'italic'), mm('('), mv('r'), mm('/'), mv('σ̂'), mm(')')],
  lad: [mv('ρ'), mm('('), mv('r'), mm(') = |'), mv('r'), mm('|')],
  theil: [],
  minimax: [mm('|'), mv('r'), mm('|  vs  ±|'), mv('h'), mm('|')],
};

export function LossChart({
  kind,
  residuals,
  weights,
  color,
  huber,
  level,
  ladMedian,
  ariaLabel,
}: LossChartProps) {
  const colors = useChartColors();
  const rho = lossFunction(kind, huber);
  // Huber: frame the quadratic zone and the bulk of the residuals; larger residuals are
  // counted at the edges (their loss is linear, which is the point of the method).
  const absR = residuals.map(Math.abs).sort((a, b) => a - b);
  const q75 = absR[Math.floor(0.75 * (absR.length - 1))] ?? 0;
  const huberHalf = kind === 'huber' && huber ? huber.delta * huber.scale : 0;
  const rMax =
    huberHalf > 0
      ? Math.min(Math.max(4 * huberHalf, 2.5 * q75), Math.max(1e-12, ...absR) * 1.12)
      : Math.max(1e-12, ...absR, level ? Math.abs(level) : 0) * 1.12;
  const xDom = niceDomain(-rMax, rMax);
  const yMax = Math.max(
    1e-12,
    ...residuals.filter((r) => Math.abs(r) <= xDom[1]).map(rho),
    rho(xDom[1]),
  );
  const yDom = niceDomain(0, yMax * 1.08);
  const weightFn =
    kind === 'huber' && huber
      ? (r: number) => Math.min(1, (huber.delta * huber.scale) / Math.max(Math.abs(r), 1e-300))
      : kind === 'lad' && ladMedian
        ? (r: number) => Math.min(1, ladMedian / Math.max(Math.abs(r), 1e-300))
        : null;

  const { canvasRef } = useCanvas((ctx, s) => {
    const x = linearScale(xDom, [M.left, s.width - M.right]);
    const y = linearScale(yDom, [s.height - M.bottom, M.top]);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    drawAxes(ctx, { x, y, frame, colors, dpr: s.dpr, xLabel: 'r', yTicks: 4, xTicks: 5 });
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 4, frame.right - frame.left, frame.bottom - frame.top + 4);
    ctx.clip();
    // Huber's quadratic zone / the minimax band.
    const half =
      kind === 'huber' && huber
        ? huber.delta * huber.scale
        : kind === 'minimax' && level
          ? Math.abs(level)
          : 0;
    if (half > 0) {
      ctx.fillStyle = color;
      ctx.globalAlpha = colors.mode === 'dark' ? 0.11 : 0.08;
      ctx.fillRect(x(-half), frame.top, x(half) - x(-half), frame.bottom - frame.top);
      ctx.globalAlpha = 0.7;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash(kind === 'minimax' ? [5, 3] : [2, 3]);
      for (const v of [-half, half]) {
        ctx.beginPath();
        ctx.moveTo(Math.round(x(v)) + 0.5, frame.top);
        ctx.lineTo(Math.round(x(v)) + 0.5, frame.bottom);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    // Weight function on the right axis (w ∈ [0, 1] → full height).
    if (weightFn) {
      ctx.strokeStyle = colors.text3;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      const n = 160;
      for (let i = 0; i <= n; i++) {
        const r = xDom[0] + ((xDom[1] - xDom[0]) * i) / n;
        const py = frame.bottom - weightFn(r) * (frame.bottom - frame.top);
        if (i) ctx.lineTo(x(r), py);
        else ctx.moveTo(x(r), py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // The loss.
    ctx.lineJoin = 'round';
    const n = 200;
    const trace = () => {
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const r = xDom[0] + ((xDom[1] - xDom[0]) * i) / n;
        if (i) ctx.lineTo(x(r), y(rho(r)));
        else ctx.moveTo(x(r), y(rho(r)));
      }
    };
    ctx.strokeStyle = colors.halo;
    ctx.lineWidth = 4.5;
    trace();
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    trace();
    ctx.stroke();
    // Residuals on the loss (and a rug on the r axis).
    const worst = kind === 'minimax' ? Math.max(...residuals.map(Math.abs)) : NaN;
    const beyond = [0, 0];
    residuals.forEach((r, i) => {
      if (r < xDom[0] || r > xDom[1]) {
        beyond[r < 0 ? 0 : 1]++;
        return;
      }
      const w = weights?.[i] ?? 1;
      const px = x(r),
        py = y(rho(r));
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = colors.text3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, frame.bottom);
      ctx.lineTo(px, frame.bottom - 5);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 0.16 + 0.84 * Math.max(0, Math.min(1, w));
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.arc(px, py, 3.25, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (kind === 'minimax' && Math.abs(Math.abs(r) - worst) <= 1e-9 * Math.max(1, worst)) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
    ctx.restore();
    // Residuals outside the frame: counted at the edge, on the loss line.
    ctx.font = tickFont(colors);
    ctx.fillStyle = colors.text2;
    if (beyond[0]) {
      ctx.textAlign = 'left';
      ctx.fillText(`◂ ${beyond[0]}`, frame.left + 4, y(rho(xDom[0])) + 14);
    }
    if (beyond[1]) {
      ctx.textAlign = 'right';
      ctx.fillText(`${beyond[1]} ▸`, frame.right - 4, y(rho(xDom[1])) + 14);
    }
    drawMath(ctx, LABELS[kind], frame.left + 8, frame.top + 4, {
      size: 12.5,
      color: colors.text2,
      halo: colors.halo,
    });
    if (weightFn) {
      ctx.save();
      ctx.font = tickFont(colors);
      ctx.fillStyle = colors.tick;
      ctx.textAlign = 'left';
      ctx.fillText('1', frame.right + 5, frame.top + 4);
      ctx.fillText('0', frame.right + 5, frame.bottom);
      ctx.restore();
      drawMath(ctx, [mv('w')], frame.right + 5, (frame.top + frame.bottom) / 2, {
        size: 12,
        baseline: 'middle',
        color: colors.text3,
      });
    }
  });

  return (
    <div className={styles.chartBox}>
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
    </div>
  );
}
