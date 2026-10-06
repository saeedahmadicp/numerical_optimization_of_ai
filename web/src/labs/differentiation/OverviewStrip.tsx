/**
 * f on its whole domain with the point of differentiation x0: drag (or click) to move x0, or
 * focus the strip and use the arrow keys (it is a slider). The magnified window of the stencil
 * view is marked around x0.
 */
import { useRef, type KeyboardEvent, type PointerEvent as RPointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import { sig } from '../../core/format';
import { useCanvas } from '../../viz/useCanvas';
import { linearScale } from '../../viz/scales';
import { drawMath, m as mm, sub, v as mv } from '../../viz/mathText';
import styles from './DifferentiationLab.module.css';

export interface OverviewStripProps {
  f: (x: number) => number;
  /** The drawn range (the problem's domain with some padding). */
  domain: [number, number];
  /** The range x0 can be dragged or stepped in (the problem's domain: f may be undefined outside). */
  limits?: readonly [number, number];
  x0: number;
  /** Half-width of the magnified window (x0 ± w). */
  window: number;
  onChange: (x0: number) => void;
}

const PAD = { left: 12, right: 12, top: 6, bottom: 6 };

export function OverviewStrip({ f, domain, limits, x0, window: w, onChange }: OverviewStripProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [a, b] = domain;
  const [xLo, xHi] = limits ?? domain;

  const { canvasRef } = useCanvas((ctx, size) => {
    const { width, height } = size;
    const xs = linearScale(domain, [PAD.left, width - PAD.right]);
    const vals: number[] = [];
    for (let i = 0; i <= 200; i++) {
      const v = f(a + ((b - a) * i) / 200);
      if (Number.isFinite(v)) vals.push(v);
    }
    let lo = Math.min(...vals),
      hi = Math.max(...vals);
    if (!(hi > lo)) {
      lo -= 1;
      hi += 1;
    }
    const ys = linearScale([lo, hi], [height - PAD.bottom, PAD.top]);
    // Outside the domain x0 cannot go: a faint wash over the padding.
    ctx.fillStyle = colors.text3;
    ctx.globalAlpha = 0.08;
    if (xLo > a) ctx.fillRect(0, 0, xs(xLo), height);
    if (xHi < b) ctx.fillRect(xs(xHi), 0, width - xs(xHi), height);
    ctx.globalAlpha = 1;
    // Window band (at least 3 px wide so it can be seen).
    const wl = xs(x0 - w),
      wr = xs(x0 + w);
    const bw = Math.max(3, wr - wl);
    ctx.fillStyle = colors.accent;
    ctx.globalAlpha = 0.14;
    ctx.fillRect((wl + wr) / 2 - bw / 2, 0, bw, height);
    ctx.globalAlpha = 1;
    // The curve.
    ctx.strokeStyle = colors.text;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i <= 400; i++) {
      const x = a + ((b - a) * i) / 400;
      const v = f(x);
      if (!Number.isFinite(v)) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(xs(x), ys(v));
      else ctx.moveTo(xs(x), ys(v));
      pen = true;
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    // x0 handle: rule + ring.
    const px = xs(x0);
    ctx.strokeStyle = colors.text2;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
    const fy = f(x0);
    const pyCurve = Number.isFinite(fy)
      ? Math.max(PAD.top, Math.min(height - PAD.bottom, ys(fy)))
      : height / 2;
    if (Number.isFinite(fy)) {
      const py = pyCurve;
      ctx.fillStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(px, py, 4.2, 0, Math.PI * 2);
      ctx.stroke();
    }
    const right = px < width - 120;
    // The label sits in the half of the strip the curve leaves free at x0.
    const labelY = pyCurve > height / 2 ? 15 : height - 7;
    drawMath(ctx, [mv('x'), sub('0'), mm(` = ${sig(x0, 4)}`)], px + (right ? 9 : -9), labelY, {
      size: 12.5,
      align: right ? 'left' : 'right',
      color: colors.text,
      halo: colors.halo,
    });
  });

  const xAt = (clientX: number) => {
    const el = box.current;
    if (!el) return x0;
    const r = el.getBoundingClientRect();
    const xs = linearScale(domain, [PAD.left, r.width - PAD.right]);
    const v = xs.invert(clientX - r.left);
    return Number(Math.max(xLo, Math.min(xHi, v)).toPrecision(4));
  };
  const down = (e: RPointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    onChange(xAt(e.clientX));
  };
  const move = (e: RPointerEvent<HTMLDivElement>) => {
    if (dragging.current) onChange(xAt(e.clientX));
  };
  const up = () => {
    dragging.current = false;
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const stepSize = ((xHi - xLo) / 100) * (e.shiftKey ? 10 : 1);
    let next: number | null = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = x0 + stepSize;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = x0 - stepSize;
    else if (e.key === 'Home') next = xLo;
    else if (e.key === 'End') next = xHi;
    if (next === null) return;
    e.preventDefault();
    e.stopPropagation();
    onChange(Number(Math.max(xLo, Math.min(xHi, next)).toPrecision(4)));
  };

  return (
    <div
      ref={box}
      className={styles.strip}
      role="slider"
      tabIndex={0}
      aria-label="Point of differentiation x₀ (drag along the curve)"
      aria-valuemin={xLo}
      aria-valuemax={xHi}
      aria-valuenow={x0}
      aria-valuetext={`x₀ = ${sig(x0, 4)}`}
      aria-orientation="horizontal"
      data-own-keys=""
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onKeyDown={key}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
