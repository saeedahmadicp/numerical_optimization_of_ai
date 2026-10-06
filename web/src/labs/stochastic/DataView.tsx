/**
 * The data panel: the samples, every method's current model, the focused method's mini-batch
 * and that batch's residual sticks: the per-sample quantities φ'(aᵢᵀ𝐰, yᵢ) its gradient averages.
 * For the squared loss φ' is the residual ŷᵢ − yᵢ (the whole stick); for the logistic loss
 * σ(zᵢ) − yᵢ (the whole stick); for Huber φ' = clip(ŷᵢ − yᵢ, ±δ), so only the first δ of a
 * stick, measured from the model, is drawn solid and the rest is dotted and faint: an outlier
 * pulls no harder than a point δ away.
 *
 *   line       (linreg_2d, huber_regression_2d)  yᵢ vs xᵢ, fitted lines ŷ = w₀ + w₁x
 *   logistic   (logreg_2d)                       labels yᵢ ∈ {0, 1} vs xᵢ (jittered), the curves
 *                                                σ(w₀ + w₁x) and the boundaries x = −w₀/w₁
 *   predicted  (ill_conditioned_ls)              yᵢ vs ŷᵢ = aᵢᵀ𝐰 of the focused method; the
 *                                                perfect fit is the diagonal
 */
import { useMemo } from 'react';
import { useChartColors } from '../../ui/theme';
import { sig } from '../../core/format';
import { useCanvas } from '../../viz/useCanvas';
import { linearScale } from '../../viz/scales';
import { TICK_SIZE, drawAxes } from '../../viz/axes';
import { drawMath, mathMain, mathVar, starRuns, type MathRun } from '../../viz';
import type { FiniteSumProblem } from '../../problems/stochastic';
import { dataMode, type Pt } from './geometry';
import styles from './StochasticLab.module.css';

export interface DataFit {
  label: string;
  slot: number;
  w: Pt | null;
  muted?: boolean;
}

export interface DataViewProps {
  problem: FiniteSumProblem;
  fits: readonly DataFit[];
  /** The focused method: its batch is highlighted (and its ŷ drawn in `predicted` mode). */
  focus: { label: string; slot: number; w: Pt | null; batch: readonly number[] | null } | null;
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** Deterministic vertical jitter for 0/1 labels (golden-ratio sequence), in label units. */
const jitter = (i: number) => (((i * 0.6180339887) % 1) - 0.5) * 0.14;
const M = { left: 40, right: 12, top: 12, bottom: 30 };

function pad([lo, hi]: [number, number], f: number): [number, number] {
  const d = (hi - lo) * f || 1;
  return [lo - d, hi + d];
}
const extent = (v: readonly number[]): [number, number] => [Math.min(...v), Math.max(...v)];

export function DataView({ problem, fits, focus }: DataViewProps) {
  const colors = useChartColors();
  const mode = dataMode(problem);
  const outliers = useMemo(() => new Set(problem.extra.outliers ?? []), [problem]);
  const wStar = problem.minima[0];

  const xd = useMemo<[number, number]>(() => {
    if (mode === 'predicted') return pad(extent(problem.y), 0.08);
    return pad(extent(problem.X.map((r) => r[1])), 0.04);
  }, [problem, mode]);
  const yd = useMemo<[number, number]>(() => {
    if (mode === 'logistic') return [-0.22, 1.22];
    return pad(extent(problem.y), 0.07);
  }, [problem, mode]);

  const predict = (w: Pt, i: number) => problem.X[i][0] * w[0] + problem.X[i][1] * w[1];

  const { canvasRef } = useCanvas((ctx, s) => {
    const x = linearScale(xd, [M.left, s.width - M.right]);
    const y = linearScale(yd, [s.height - M.bottom, M.top]);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    const xName: MathRun[] = mode === 'predicted' ? [mathVar('ŷ')] : [mathVar('x')];
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      xName,
      yName: [mathVar('y')],
      yTicks: mode === 'logistic' ? 3 : 4,
    });
    const color = (slot: number) => colors.series[slot % colors.series.length];
    const n = problem.nSamples;
    const ptY = (i: number) => (mode === 'logistic' ? problem.y[i] + jitter(i) : problem.y[i]);
    const focusW = focus?.w ?? null;
    const ptX = (i: number) =>
      mode === 'predicted' ? (focusW ? predict(focusW, i) : NaN) : problem.X[i][1];

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    const model = (w: Pt) =>
      mode === 'logistic'
        ? (t: number) => sigmoid(w[0] + w[1] * t)
        : (t: number) => w[0] + w[1] * t;
    const curve = (
      f: (t: number) => number,
      stroke: string,
      width: number,
      dash: number[] = [],
    ) => {
      const N = 96;
      const trace = () => {
        ctx.beginPath();
        for (let k = 0; k <= N; k++) {
          const t = xd[0] + ((xd[1] - xd[0]) * k) / N;
          const v = Math.max(yd[0] - 1e3, Math.min(yd[1] + 1e3, f(t)));
          if (k === 0) ctx.moveTo(x(t), y(v));
          else ctx.lineTo(x(t), y(v));
        }
      };
      trace();
      ctx.setLineDash([]);
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = width + 2.5;
      ctx.globalAlpha = 0.75;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.setLineDash(dash);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      trace();
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Reference: the minimizer's model (or the perfect-fit diagonal).
    if (mode === 'predicted') curve((t) => t, colors.text2, 1.25, [5, 4]);
    else curve(model(wStar), colors.text2, 1.25, [5, 4]);

    // Points.
    for (let i = 0; i < n; i++) {
      const px = x(ptX(i)),
        py = y(ptY(i));
      if (!Number.isFinite(px)) continue;
      ctx.globalAlpha = outliers.has(i) ? 0.85 : 0.42;
      if (outliers.has(i)) {
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.arc(px, py, 3.1, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = colors.text;
        ctx.beginPath();
        ctx.arc(px, py, 2.1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // Every method's current model (predicted mode: only the focused one, as points).
    if (mode !== 'predicted') {
      for (const f of fits) {
        if (!f.w) continue;
        ctx.globalAlpha = f.muted ? 0.4 : 1;
        curve(model(f.w), color(f.slot), f.muted ? 1.5 : 2);
        if (mode === 'logistic' && Math.abs(f.w[1]) > 1e-12) {
          const xb = -f.w[0] / f.w[1];
          if (xb > xd[0] && xb < xd[1]) {
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = color(f.slot);
            ctx.lineWidth = 1.25;
            ctx.beginPath();
            ctx.moveTo(x(xb), y(-0.05));
            ctx.lineTo(x(xb), y(1.05));
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
        ctx.globalAlpha = 1;
      }
    }

    // The focused method's mini-batch and its residual sticks (ŷᵢ − yᵢ, or σ(zᵢ) − yᵢ).
    if (focus?.w && focus.batch) {
      const c = color(focus.slot);
      const fw = focus.w;
      const fit = (i: number) => (mode === 'logistic' ? sigmoid(predict(fw, i)) : predict(fw, i));
      ctx.strokeStyle = c;
      ctx.lineWidth = 1.4;
      const huber = problem.loss === 'huber';
      const delta = problem.huberDelta;
      for (const i of focus.batch) {
        const px = x(ptX(i));
        const yi = ptY(i),
          fi = fit(i);
        // The part of the stick the gradient sees: all of it, or δ from the model for Huber.
        const cut = huber && Math.abs(fi - yi) > delta ? fi - Math.sign(fi - yi) * delta : yi;
        ctx.beginPath();
        ctx.moveTo(px, y(fi));
        ctx.lineTo(px, y(cut));
        ctx.stroke();
        if (cut !== yi) {
          ctx.save();
          ctx.globalAlpha = 0.45;
          ctx.lineWidth = 1.1;
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(px, y(cut));
          ctx.lineTo(px, y(yi));
          ctx.stroke();
          ctx.restore();
        }
      }
      for (const i of focus.batch) {
        const px = x(ptX(i)),
          py = y(ptY(i));
        ctx.fillStyle = colors.halo;
        ctx.beginPath();
        ctx.arc(px, py, 5.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = c;
        ctx.beginPath();
        ctx.arc(px, py, 3.7, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();

    // Label the reference curve at its right end.
    const label: MathRun[] =
      mode === 'predicted' ? [mathVar('y'), mathMain(' = '), mathVar('ŷ')] : starRuns('w');
    const tx = frame.right - 6;
    const ty =
      mode === 'predicted'
        ? y(xd[1] - (xd[1] - xd[0]) * 0.04)
        : y(model(wStar)(xd[1] - (xd[1] - xd[0]) * 0.04));
    if (ty > frame.top + 8 && ty < frame.bottom - 4)
      drawMath(ctx, label, tx, ty - 7, {
        size: TICK_SIZE + 1,
        align: 'right',
        color: colors.text2,
        halo: colors.halo,
      });
    if (mode === 'logistic') {
      drawMath(
        ctx,
        [mathVar('P'), mathMain('('), mathVar('y'), mathMain(' = 1)')],
        frame.left + 8,
        frame.top + 12,
        {
          size: TICK_SIZE + 1,
          color: colors.text3,
          halo: colors.halo,
        },
      );
    }
  });

  const fitText = (f: DataFit) =>
    f.w
      ? mode === 'logistic'
        ? `${f.label}: P(y = 1) = σ(${sig(f.w[0], 3)} + ${sig(f.w[1], 3)}x)`
        : mode === 'line'
          ? `${f.label}: ŷ = ${sig(f.w[0], 3)} + ${sig(f.w[1], 3)}x`
          : `${f.label}: ŷ = ${sig(f.w[0], 3)}u + ${sig(f.w[1], 3)}·(30v)`
      : `${f.label}: no iterate`;
  const aria =
    `${problem.nSamples} samples of ${problem.name}` +
    (mode === 'predicted' ? ', observed y against the focused method’s prediction ŷ' : '') +
    '. ' +
    fits.map(fitText).join('; ') +
    (focus?.batch
      ? `. Mini-batch of ${focus.label}: ${focus.batch.length} samples highlighted, with their residuals${problem.loss === 'huber' ? ` (solid up to δ = ${problem.huberDelta}, the part the Huber gradient uses)` : ''}.`
      : '.');

  return (
    <div className={styles.dataWrap}>
      <canvas ref={canvasRef} className={styles.dataCanvas} role="img" aria-label={aria} />
    </div>
  );
}
