/**
 * Damping μₖ (log scale, Levenberg–Marquardt) over the gain ratio ϱₖ (every method), on one
 * k axis and one playhead.
 *
 *   ϱₖ = (f(𝐱ₖ) − f(𝐱ₖ + 𝐡ₖ)) / (L(𝟎) − L(𝐡ₖ))     actual over predicted decrease
 *
 * ϱ ≤ 0 rejects the LM trial (hollow ring) and raises μ by ν; ϱ near 1 means the linear model
 * was exact and lowers μ by up to 3× (Nielsen's update). Gauss–Newton's μ is 0 (undamped), so
 * only its ϱ is drawn. Values outside the ϱ window sit on its edge as a triangle pointing out.
 */
import { useRef, type PointerEvent as RPointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import {
  crisp,
  drawMath,
  linearScale,
  linearTicks,
  logExtent,
  logScale,
  mathSub as sub,
  mathVar as mv,
  pow10Runs,
  useCanvas,
  type Scale,
} from '../../viz';
import { TICK_SIZE, labelFont, tickFont } from '../../viz/axes';
import styles from './LeastSquaresLab.module.css';

export interface DampingSeries {
  name: string;
  slot: number;
  /** μ used at each step (null where undefined; 0 for Gauss–Newton). */
  mu: readonly (number | null)[];
  gain: readonly (number | null)[];
  accepted: readonly (boolean | null)[];
}

export interface DampingChartProps {
  series: readonly DampingSeries[];
  t: number;
  onSeek?: (k: number) => void;
  ariaLabel: string;
}

const RHO: [number, number] = [-0.5, 1.5];
const M = { left: 44, right: 50, top: 10, bottom: 26, gap: 22 };

export function DampingChart({ series, t, onSeek, ariaLabel }: DampingChartProps) {
  const colors = useChartColors();
  const maxK = Math.max(1, ...series.map((s) => s.mu.length - 1));
  const damped = series.filter((s) => s.mu.some((v) => v !== null && v > 0));
  const muDom = logExtent(damped.flatMap((s) => s.mu.filter((v): v is number => v !== null)));
  const xRef = useRef<Scale | null>(null);

  const { canvasRef } = useCanvas((ctx, s) => {
    const c = colors;
    const x = linearScale([0, maxK], [M.left, s.width - M.right]);
    xRef.current = x;
    const h = s.height - M.top - M.bottom - M.gap;
    const topH = Math.round(h * 0.5);
    const muF = { top: M.top, bottom: M.top + topH };
    const rhoF = { top: muF.bottom + M.gap, bottom: s.height - M.bottom };
    const yMu = logScale(muDom, [muF.bottom, muF.top]);
    const yRho = linearScale(RHO, [rhoF.bottom, rhoF.top]);
    const hair = 1 / s.dpr;

    // Grids and ticks.
    ctx.save();
    ctx.strokeStyle = c.grid;
    ctx.lineWidth = hair;
    ctx.beginPath();
    const xt = linearTicks(
      0,
      maxK,
      Math.max(2, Math.floor((s.width - M.left - M.right) / 70)),
    ).filter((v) => Number.isInteger(v));
    for (const v of xt) {
      const px = crisp(x(v), s.dpr);
      for (const f of [muF, rhoF]) {
        ctx.moveTo(px, f.top);
        ctx.lineTo(px, f.bottom);
      }
    }
    const muTicks = yMu.ticks(3);
    if (damped.length)
      for (const v of muTicks) {
        const py = crisp(yMu(v), s.dpr);
        ctx.moveTo(M.left, py);
        ctx.lineTo(s.width - M.right, py);
      }
    for (const v of [-0.5, 0.5, 1.5]) {
      const py = crisp(yRho(v), s.dpr);
      ctx.moveTo(M.left, py);
      ctx.lineTo(s.width - M.right, py);
    }
    ctx.stroke();
    // ϱ ≤ 0: the rejection zone (neutral tint, never a series color).
    ctx.fillStyle = c.region;
    ctx.globalAlpha = 0.35;
    ctx.fillRect(M.left, yRho(0), s.width - M.right - M.left, rhoF.bottom - yRho(0));
    ctx.globalAlpha = 1;
    // ϱ = 0 (accept/reject) and ϱ = 1 (the model is exact).
    ctx.strokeStyle = c.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(M.left, crisp(yRho(0), s.dpr));
    ctx.lineTo(s.width - M.right, crisp(yRho(0), s.dpr));
    ctx.stroke();
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(M.left, crisp(yRho(1), s.dpr));
    ctx.lineTo(s.width - M.right, crisp(yRho(1), s.dpr));
    ctx.stroke();
    ctx.setLineDash([]);
    // Frame baselines.
    ctx.strokeStyle = c.axis;
    ctx.lineWidth = hair;
    ctx.beginPath();
    for (const f of [muF, rhoF]) {
      ctx.moveTo(crisp(M.left, s.dpr), f.top);
      ctx.lineTo(crisp(M.left, s.dpr), crisp(f.bottom, s.dpr));
    }
    ctx.moveTo(M.left, crisp(rhoF.bottom, s.dpr));
    ctx.lineTo(s.width - M.right, crisp(rhoF.bottom, s.dpr));
    ctx.stroke();

    ctx.font = tickFont(c);
    ctx.fillStyle = c.tick;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    // The axis name k sits at the right end of the tick row; ticks under it are skipped.
    for (const v of xt)
      if (x(v) < s.width - M.right - 14) ctx.fillText(String(v), x(v), rhoF.bottom + 5);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of [0, 1]) ctx.fillText(String(v), M.left - 6, yRho(v));
    if (damped.length)
      for (const v of muTicks)
        drawMath(ctx, pow10Runs(Math.round(Math.log10(v))), M.left - 6, yMu(v), {
          size: TICK_SIZE + 1,
          align: 'right',
          baseline: 'middle',
          color: c.tick,
        });
    // Axis names.
    drawMath(ctx, [mv('μ'), sub('k', 'italic')], M.left + 6, muF.top + 12, {
      size: 14,
      color: c.text2,
      halo: c.halo,
    });
    drawMath(ctx, [mv('ϱ'), sub('k', 'italic')], M.left + 6, rhoF.top + 12, {
      size: 14,
      color: c.text2,
      halo: c.halo,
    });
    drawMath(ctx, [mv('k')], s.width - M.right, s.height - 6, {
      size: 13,
      align: 'right',
      color: c.text2,
    });
    // Line labels in the right margin, clear of the data.
    ctx.font = labelFont(c);
    ctx.fillStyle = c.text3;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const lx = s.width - M.right + 6;
    ctx.fillText('exact', lx, yRho(1));
    ctx.fillText('reject', lx, (yRho(0) + rhoF.bottom) / 2);
    if (!damped.length) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = c.text3;
      ctx.font = `400 12px ${c.fontSans || 'system-ui, sans-serif'}`;
      ctx.fillText(
        'Gauss–Newton is undamped (μ = 0); add Levenberg–Marquardt',
        (M.left + s.width - M.right) / 2,
        (muF.top + muF.bottom) / 2,
      );
    }
    ctx.restore();

    // Data: μ lines (LM), ϱ markers (all), faint after the playhead.
    ctx.save();
    for (const pass of ['future', 'past'] as const) {
      for (const sr of series) {
        const col = c.series[sr.slot % c.series.length];
        const n = sr.mu.length;
        const kEnd = Math.min(t, n - 1);
        const inPass = (k: number) => (pass === 'past' ? k <= kEnd + 1e-9 : k > kEnd + 1e-9);
        ctx.globalAlpha = pass === 'future' ? 0.22 : 1;
        if (sr.mu.some((v) => v !== null && v > 0)) {
          ctx.strokeStyle = col;
          ctx.lineWidth = pass === 'future' ? 1.25 : 1.75;
          ctx.lineJoin = 'round';
          ctx.beginPath();
          let pen = false;
          const from = pass === 'past' ? 0 : Math.max(0, Math.floor(kEnd));
          for (let k = from; k < n; k++) {
            if (pass === 'past' && k > kEnd + 1e-9) break;
            const v = sr.mu[k];
            if (v === null || !(v > 0)) {
              pen = false;
              continue;
            }
            const px = x(k),
              py = yMu(v);
            if (pen) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            pen = true;
          }
          ctx.stroke();
          for (let k = 0; k < n; k++) {
            const v = sr.mu[k];
            if (!inPass(k) || v === null || !(v > 0)) continue;
            dot(ctx, x(k), yMu(v), sr.accepted[k] === false ? 'ring' : 'dot', col, c.halo, 2.6);
          }
        }
        for (let k = 0; k < sr.gain.length; k++) {
          if (!inPass(k)) continue;
          const g = sr.gain[k];
          if (g === null || !Number.isFinite(g)) continue;
          const shape = sr.accepted[k] === false ? 'ring' : 'dot';
          if (g < RHO[0] || g > RHO[1]) {
            tri(ctx, x(k), yRho(g < RHO[0] ? RHO[0] : RHO[1]), g < RHO[0] ? 1 : -1, col, c.halo);
          } else dot(ctx, x(k), yRho(g), shape, col, c.halo, 3);
        }
      }
    }
    ctx.restore();
    // Playhead.
    ctx.save();
    ctx.strokeStyle = c.playhead;
    ctx.lineWidth = 1;
    const px = crisp(x(Math.min(t, maxK)), s.dpr);
    ctx.beginPath();
    ctx.moveTo(px, muF.top);
    ctx.lineTo(px, rhoF.bottom);
    ctx.stroke();
    ctx.restore();
  });

  const onClick = (e: RPointerEvent<HTMLCanvasElement>) => {
    const x = xRef.current;
    if (!onSeek || !x) return;
    const r = e.currentTarget.getBoundingClientRect();
    const k = Math.round(x.invert(e.clientX - r.left));
    onSeek(Math.max(0, Math.min(maxK, k)));
  };

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      role="img"
      aria-label={ariaLabel}
      onPointerUp={onClick}
    />
  );
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  shape: 'dot' | 'ring',
  color: string,
  halo: string,
  r: number,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, r + 1.75, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (shape === 'ring') {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
}

/** A small triangle at the window edge pointing outward (dir 1 = down, −1 = up). */
function tri(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  dir: 1 | -1,
  color: string,
  halo: string,
) {
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(x - 4.5, y - dir * 3);
    ctx.lineTo(x + 4.5, y - dir * 3);
    ctx.lineTo(x, y + dir * 4);
    ctx.closePath();
  };
  ctx.strokeStyle = halo;
  ctx.lineWidth = 3;
  path();
  ctx.stroke();
  ctx.fillStyle = color;
  path();
  ctx.fill();
}
