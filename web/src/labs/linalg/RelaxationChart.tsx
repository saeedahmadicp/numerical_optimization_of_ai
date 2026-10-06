/**
 * ρ(G_ω) against the relaxation factor ω ∈ (0, 2): the curve that decides SOR's speed. Kahan's
 * lower bound |ω − 1| is dashed, Gauss–Seidel is the point ω = 1, and when Young's theory
 * applies (A tridiagonal, real Jacobi spectrum, ρ_J < 1) the minimum sits at
 * ω⋆ = 2/(1 + √(1 − ρ_J²)) with ρ(G_ω⋆) = ω⋆ − 1. The SOR method's ω is a handle on the curve:
 * drag it (or use the arrow keys) and every view re-runs. Enter, or a drag within 0.005 of ω⋆,
 * sets the exact ω⋆: the minimum is a cusp, and ω rounded to 0.01 misses it.
 * Labels are placed clear of the curve, the marks and each other (first free candidate).
 */
import { useRef, type KeyboardEvent, type PointerEvent } from 'react';
import { useChartColors } from '../../ui/theme';
import {
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  measureMath,
  useCanvas,
  type MathRun,
} from '../../viz';
import { LABEL_SIZE, TICK_SIZE } from '../../viz/axes';
import { nearText } from './texfmt';
import { OMEGA_MAX, OMEGA_MIN, clampOmega, omegaText, snapOmega } from './model';
import styles from './LinalgLab.module.css';

/** Math labels: one step above the tick size, so their subscripts stay legible. */
const MATH_SIZE = TICK_SIZE + 1;

export interface RelaxationChartProps {
  curve: readonly [number, number][];
  rhoJ: number | null;
  rhoGS: number | null;
  omegaOpt: number | null;
  rhoOpt: number | null;
  /** The SOR selection's ω (null when SOR is not selected). */
  omega: number | null;
  /** ρ(G_ω) at that ω, computed exactly (the curve is sampled). */
  rhoOmega: number | null;
  /** Color slot of the SOR method (null when not selected). */
  slot: number | null;
  onOmega: (omega: number) => void;
}

interface Box {
  l: number;
  r: number;
  t: number;
  b: number;
}
const overlaps = (a: Box, b: Box, pad: number) =>
  a.l - pad < b.r && b.l - pad < a.r && a.t - pad < b.b && b.t - pad < a.b;
const inBox = (a: Box, px: number, py: number, pad: number) =>
  px > a.l - pad && px < a.r + pad && py > a.t - pad && py < a.b + pad;

function rhoAt(curve: readonly [number, number][], w: number): number | null {
  for (let i = 1; i < curve.length; i++) {
    const [w0, r0] = curve[i - 1],
      [w1, r1] = curve[i];
    if (w >= w0 && w <= w1) return r0 + ((r1 - r0) * (w - w0)) / (w1 - w0);
  }
  return null;
}

export function RelaxationChart({
  curve,
  rhoJ,
  rhoGS,
  omegaOpt,
  rhoOpt,
  omega,
  rhoOmega,
  slot,
  onOmega,
}: RelaxationChartProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const M = { left: 40, right: 14, top: 22, bottom: 30 };
  const peak = Math.max(1.05, ...curve.map(([, r]) => Math.min(r, 1.6)));
  const yMax = Math.min(1.6, peak * 1.04);
  const scales = (w: number, h: number) => ({
    x: linearScale([0, 2], [M.left, w - M.right]),
    y: linearScale([0, yMax], [h - M.bottom, M.top]),
  });
  const current = omega ?? 1.5;
  const rhoNow = omega !== null && rhoOmega !== null ? rhoOmega : rhoAt(curve, current);

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
      xTicks: 4,
      yTicks: 4,
      xName: [{ t: 'ω', style: 'italic' }],
      yName: [
        { t: 'ρ', style: 'italic' },
        { t: '(', style: 'main' },
        { t: 'G', style: 'italic' },
        { t: 'ω', style: 'italic', script: 'sub' },
        { t: ')', style: 'main' },
      ],
      fontSize: 10,
    });
    // ρ ≥ 1: the iteration diverges.
    if (yMax > 1) {
      const y1 = y(1);
      ctx.fillStyle = colors.region;
      ctx.globalAlpha = 0.5;
      ctx.fillRect(frame.left, frame.top, frame.right - frame.left, y1 - frame.top);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = colors.text3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(frame.left, crisp(y1, s.dpr));
      ctx.lineTo(frame.right, crisp(y1, s.dpr));
      ctx.stroke();
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 2, frame.right - frame.left, frame.bottom - frame.top + 2);
    ctx.clip();
    // Kahan: ρ(G_ω) ≥ |ω − 1|.
    ctx.strokeStyle = colors.text3;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(0), y(1));
    ctx.lineTo(x(1), y(0));
    ctx.lineTo(x(2), y(1));
    ctx.stroke();
    ctx.setLineDash([]);
    // ρ(G_J) reference.
    if (rhoJ !== null && rhoJ < yMax) {
      ctx.strokeStyle = colors.text3;
      ctx.setLineDash([1.5, 3]);
      ctx.beginPath();
      ctx.moveTo(frame.left, y(rhoJ));
      ctx.lineTo(frame.right, y(rhoJ));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // The curve.
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 1.75;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    curve.forEach(([w, r], i) =>
      i
        ? ctx.lineTo(x(w), y(Math.min(r, yMax * 1.2)))
        : ctx.moveTo(x(w), y(Math.min(r, yMax * 1.2))),
    );
    ctx.stroke();
    ctx.restore();

    // ── Marks: Gauss–Seidel's dot, Young's cross, SOR's handle ──────────────────────
    const gsPt = rhoGS !== null && rhoGS < yMax ? { x: x(1), y: y(rhoGS) } : null;
    const optPt = omegaOpt !== null && rhoOpt !== null ? { x: x(omegaOpt), y: y(rhoOpt) } : null;
    const handle =
      omega !== null && slot !== null && rhoNow !== null
        ? { x: crisp(x(omega), s.dpr), y: y(Math.min(rhoNow, yMax)) }
        : null;
    if (gsPt) {
      ctx.fillStyle = colors.text;
      ctx.beginPath();
      ctx.arc(gsPt.x, gsPt.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (optPt) {
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(optPt.x - 5, optPt.y);
      ctx.lineTo(optPt.x + 5, optPt.y);
      ctx.moveTo(optPt.x, optPt.y - 5);
      ctx.lineTo(optPt.x, optPt.y + 5);
      ctx.stroke();
    }
    if (omega !== null && slot !== null) {
      const color = colors.series[slot % colors.series.length];
      const px = crisp(x(omega), s.dpr);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (handle) {
        ctx.fillStyle = colors.halo;
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, 6.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, 4.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ── Labels, placed clear of the curve, the marks and each other ─────────────────
    const curvePts: { x: number; y: number }[] = [];
    for (let i = 0; i < curve.length; i++) {
      const [w1, r1] = curve[i];
      const p1 = { x: x(w1), y: y(Math.min(r1, yMax * 1.2)) };
      if (i > 0) {
        const p0 = curvePts[curvePts.length - 1];
        const steps = Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) / 3);
        for (let q = 1; q < steps; q++)
          curvePts.push({
            x: p0.x + ((p1.x - p0.x) * q) / steps,
            y: p0.y + ((p1.y - p0.y) * q) / steps,
          });
      }
      curvePts.push(p1);
    }
    const marks = [gsPt, optPt, handle].filter(Boolean) as { x: number; y: number }[];
    const placed: Box[] = [];
    const place = (
      runs: MathRun[],
      size: number,
      candidates: readonly [number, number, 'left' | 'right'][],
      color: string,
      avoidCurve = true,
    ): boolean => {
      const w = measureMath(ctx, runs, size);
      for (const [ax, ay, align] of candidates) {
        const box: Box = {
          l: align === 'right' ? ax - w : ax,
          r: align === 'right' ? ax : ax + w,
          t: ay - size * 0.85,
          b: ay + size * 0.3,
        };
        if (box.l < frame.left + 2 || box.r > frame.right - 2) continue;
        if (box.t < frame.top - 2 || box.b > frame.bottom - 1) continue;
        if (placed.some((p) => overlaps(p, box, 2))) continue;
        if (marks.some((m) => inBox(box, m.x, m.y, 6))) continue;
        // Never across the ρ = 1 line that bounds the divergence band.
        if (yMax > 1 && box.t - 1 < y(1) && box.b + 1 > y(1)) continue;
        if (avoidCurve && curvePts.some((p) => inBox(box, p.x, p.y, 2))) continue;
        placed.push(box);
        drawMath(ctx, runs, ax, ay, { size, align, color, halo: colors.halo });
        return true;
      }
      return false;
    };

    if (handle && omega !== null && slot !== null && rhoNow !== null) {
      const atOpt = omegaOpt !== null && Math.abs(omega - omegaOpt) < 1e-9;
      const runs: MathRun[] = [
        { t: 'ω', style: 'italic' },
        ...(atOpt
          ? ([
              { t: ' = ', style: 'main' },
              { t: 'ω', style: 'italic' },
              { t: '⋆', style: 'main', script: 'sup' },
            ] as MathRun[])
          : [{ t: ` = ${omegaText(omega)}`, style: 'main' } as MathRun]),
        { t: ',  ', style: 'main' },
        { t: 'ρ', style: 'italic' },
        { t: ` = ${nearText(rhoNow, (v) => v.toFixed(3))}`, style: 'main' },
      ];
      const { x: hx, y: hy } = handle;
      const placedFree = place(
        runs,
        MATH_SIZE,
        [
          [hx - 10, hy - 10, 'right'],
          [hx + 10, hy - 10, 'left'],
          [hx + 10, hy + 18, 'left'],
          [hx - 10, hy + 18, 'right'],
          [hx - 10, frame.top + 12, 'right'],
          [hx + 10, frame.top + 12, 'left'],
        ],
        colors.text,
      );
      // No spot clear of the curve: the handle's label still shows, over the curve.
      if (!placedFree)
        place(
          runs,
          MATH_SIZE,
          [
            [hx - 10, hy - 10, 'right'],
            [hx + 10, hy - 10, 'left'],
          ],
          colors.text,
          false,
        );
    }
    // Young's ω⋆: dropped when the handle sits on it (its label then reads ω = ω⋆).
    if (
      optPt &&
      omegaOpt !== null &&
      !(handle && Math.hypot(handle.x - optPt.x, handle.y - optPt.y) < 14)
    ) {
      const runs: MathRun[] = [
        { t: 'ω', style: 'italic' },
        { t: '⋆', style: 'main', script: 'sup' },
        { t: ` = ${nearText(omegaOpt, (v) => v.toFixed(3))}`, style: 'main' },
      ];
      place(
        runs,
        MATH_SIZE,
        [
          [optPt.x + 8, optPt.y + 15, 'left'],
          [optPt.x - 8, optPt.y + 15, 'right'],
          [optPt.x + 8, optPt.y - 8, 'left'],
          [optPt.x - 8, optPt.y - 8, 'right'],
        ],
        colors.text2,
      );
    }
    if (gsPt && !(handle && Math.hypot(handle.x - gsPt.x, handle.y - gsPt.y) < 14))
      place(
        [{ t: 'Gauss–Seidel', style: 'sans' }],
        LABEL_SIZE,
        [
          [gsPt.x - 7, gsPt.y - 6, 'right'],
          [gsPt.x - 7, gsPt.y + 14, 'right'],
          [gsPt.x + 7, gsPt.y + 14, 'left'],
          [gsPt.x + 7, gsPt.y - 6, 'left'],
        ],
        colors.text2,
      );
    if (rhoJ !== null && rhoJ < yMax) {
      const yj = y(rhoJ);
      place(
        [
          { t: 'ρ', style: 'italic' },
          { t: '(', style: 'main' },
          { t: 'G', style: 'italic' },
          { t: 'J', style: 'italic', script: 'sub' },
          { t: `) = ${nearText(rhoJ, (v) => v.toFixed(3))}`, style: 'main' },
        ],
        MATH_SIZE,
        [
          [frame.right - 6, yj - 5, 'right'],
          [frame.right - 6, yj + 14, 'right'],
          [frame.left + 6, yj - 5, 'left'],
          [frame.left + 6, yj + 14, 'left'],
          [x(1), yj - 5, 'left'],
          [x(1), yj + 14, 'left'],
        ],
        colors.text2,
      );
    }
    // Kahan's bound |ω − 1| (ω italic; the bars, minus and 1 upright).
    place(
      [
        { t: '|', style: 'main' },
        { t: 'ω', style: 'italic' },
        { t: ' − 1|', style: 'main' },
      ],
      MATH_SIZE,
      [
        [x(0.34) - 4, y(0.66) + 14, 'right'],
        [x(0.5) - 4, y(0.5) + 14, 'right'],
        [x(1.66) + 4, y(0.66) + 14, 'left'],
        [x(1.5) + 4, y(0.5) + 14, 'left'],
      ],
      colors.text3,
    );
    if (yMax > 1) {
      const y1 = y(1);
      const yy = Math.max(frame.top + 11, y1 - 5);
      place(
        [{ t: 'diverges', style: 'serif-italic' }],
        12,
        [
          [frame.left + 6, yy, 'left'],
          [x(1) - 30, yy, 'left'],
          [frame.right - 6, yy, 'right'],
          [x(0.6), yy, 'left'],
          [x(1.3), yy, 'left'],
        ],
        colors.text3,
      );
    }
  });

  const fromPointer = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    const { x } = scales(r.width, r.height);
    const w = x.invert(clientX - r.left);
    // ρ(G_ω) has a square-root cusp at ω⋆: a drag that lands on it gets the exact value
    // (rounding to 0.01 would miss the minimum ρ = ω⋆ − 1).
    onOmega(snapOmega(w, omegaOpt));
  };
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragging.current = true;
    fromPointer(e.clientX);
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) fromPointer(e.clientX);
  };
  const onUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const d = e.shiftKey ? 0.1 : 0.01;
    const map: Record<string, number> = {
      ArrowRight: current + d,
      ArrowUp: current + d,
      ArrowLeft: current - d,
      ArrowDown: current - d,
      Home: OMEGA_MIN,
      End: OMEGA_MAX,
    };
    if (e.key in map) {
      e.preventDefault();
      e.stopPropagation();
      onOmega(clampOmega(map[e.key]));
    } else if (e.key === 'Enter' && omegaOpt !== null) {
      e.preventDefault();
      // The exact ω⋆, not rounded: the minimum of ρ(G_ω) is a cusp.
      onOmega(Math.min(OMEGA_MAX, Math.max(OMEGA_MIN, omegaOpt)));
    }
  };

  return (
    <div
      ref={box}
      className={`${styles.canvasBox} ${styles.slider}`}
      role="slider"
      tabIndex={0}
      data-own-keys=""
      aria-label="Relaxation factor ω of SOR"
      aria-valuemin={OMEGA_MIN}
      aria-valuemax={OMEGA_MAX}
      aria-valuenow={current}
      aria-valuetext={`ω = ${omegaOpt !== null && Math.abs(current - omegaOpt) < 1e-9 ? `${omegaText(current)} (optimal)` : omegaText(current)}, spectral radius ${rhoNow === null ? 'unknown' : nearText(rhoNow, (v) => v.toFixed(3))}${omegaOpt !== null ? `; optimal ω = ${nearText(omegaOpt, (v) => v.toFixed(3))} (Enter)` : ''}`}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={onKey}
    >
      <canvas ref={canvasRef} aria-hidden="true" />
    </div>
  );
}
