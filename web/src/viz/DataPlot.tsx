import {
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as RKeyboardEvent,
  type PointerEvent as RPointerEvent,
} from 'react';
import { useChartColors } from '../ui/theme';
import { sig } from '../core/format';
import { useCanvas, useElementSize } from './useCanvas';
import { linearScale, type Scale } from './scales';
import { dataDomain } from './chartMath';
import { drawAxes } from './axes';
import { adaptiveSample } from './sampling';
import type { MathRun } from './mathText';
import styles from './DataPlot.module.css';

export interface DataPoint {
  x: number;
  y: number;
}

export interface DataCurve {
  f: (x: number) => number;
  /** Method color slot (0–3); ink when omitted. */
  slot?: number;
  label?: string;
  dashed?: boolean;
  /** Restrict the curve to [from, to] (default the x-domain). */
  from?: number;
  to?: number;
}

export interface DataPlotProps {
  points: readonly DataPoint[];
  /** When set, points are draggable (pointer and keyboard) and every move reports the new list. */
  onPointsChange?: (points: DataPoint[]) => void;
  curves?: readonly DataCurve[];
  /** Vertical residual segments from each point to this curve (index into `curves`). */
  residualsTo?: number;
  /** Points drawn emphasized (e.g. outliers a robust fit down-weights). */
  highlight?: readonly number[];
  /** Per-point weight in [0, 1] (robust regression): drawn as the dot's opacity. */
  weights?: readonly number[];
  xDomain?: [number, number];
  yDomain?: [number, number];
  xLabel?: string;
  yLabel?: string;
  xName?: readonly MathRun[];
  yName?: readonly MathRun[];
  ariaLabel: string;
  className?: string;
  /** Keyboard step as a fraction of the domain (default 1 %; Shift = 10×). */
  keyStep?: number;
}

const M = { left: 46, right: 14, top: 22, bottom: 30 };

/**
 * Scatter + curves (fits, interpolants) with optional draggable points. Each draggable point is a
 * real button over the canvas: Tab reaches it, arrow keys move it (Shift for larger steps), and
 * its label states the coordinates.
 */
export function DataPlot({
  points,
  onPointsChange,
  curves = [],
  residualsTo,
  highlight = [],
  weights,
  xDomain,
  yDomain,
  xLabel = 'x',
  yLabel = 'y',
  xName,
  yName,
  ariaLabel,
  className,
  keyStep = 0.01,
}: DataPlotProps) {
  const colors = useChartColors();
  const wrap = useRef<HTMLDivElement>(null);
  const size = useElementSize(wrap);
  // While dragging, the axes stay frozen (a rescaling plot would run away from the pointer).
  const [frozen, setFrozen] = useState<{ x: [number, number]; y: [number, number] } | null>(null);
  const autoX = useMemo(() => xDomain ?? dataDomain(points.map((p) => p.x)), [xDomain, points]);
  const autoY = useMemo(() => {
    if (yDomain) return yDomain;
    const ys = points.map((p) => p.y);
    // Include the curves over the data range so fits stay in view.
    for (const c of curves)
      for (let i = 0; i <= 40; i++) {
        const v = c.f(autoX[0] + ((autoX[1] - autoX[0]) * i) / 40);
        if (Number.isFinite(v)) ys.push(v);
      }
    const d = dataDomain(points.map((p) => p.y));
    const all = dataDomain(ys);
    // Never let a wild curve squash the data: at most 3× the data span.
    const span = d[1] - d[0];
    return [Math.max(all[0], d[0] - span), Math.min(all[1], d[1] + span)] as [number, number];
  }, [yDomain, points, curves, autoX]);
  const xd = frozen?.x ?? autoX;
  const yd = frozen?.y ?? autoY;

  const scales = (w: number, h: number): { x: Scale; y: Scale } => ({
    x: linearScale(xd, [M.left, w - M.right]),
    y: linearScale(yd, [h - M.bottom, M.top]),
  });

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - M.bottom,
    };
    drawAxes(ctx, { x, y, frame, colors, dpr: s.dpr, xLabel, yLabel, xName, yName });
    const color = (slot?: number) =>
      slot === undefined ? colors.text : colors.series[slot % colors.series.length];
    const P = (a: number, b: number): [number, number] => [x(a), y(b)];
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();
    // Residuals under everything.
    const rc = residualsTo !== undefined ? curves[residualsTo] : undefined;
    if (rc) {
      ctx.strokeStyle = color(rc.slot);
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      for (const p of points) {
        const fy = rc.f(p.x);
        if (!Number.isFinite(fy)) continue;
        ctx.moveTo(...P(p.x, p.y));
        ctx.lineTo(...P(p.x, fy));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    for (const c of curves) {
      const from = c.from ?? xd[0],
        to = c.to ?? xd[1];
      const pts = adaptiveSample(c.f, from, to, P);
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = 4.5;
      ctx.setLineDash([]);
      ctx.lineJoin = 'round';
      const trace = () => {
        ctx.beginPath();
        let pen = false;
        for (const [a, b] of pts) {
          const [px, py] = P(a, b);
          if (!Number.isFinite(py) || Math.abs(py) > 1e5) {
            pen = false;
            continue;
          }
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        }
      };
      trace();
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color(c.slot);
      ctx.lineWidth = 2;
      ctx.setLineDash(c.dashed ? [6, 4] : []);
      trace();
      ctx.stroke();
    }
    ctx.setLineDash([]);
    points.forEach((p, i) => {
      const [px, py] = P(p.x, p.y);
      const hi = highlight.includes(i);
      const w = weights?.[i] ?? 1;
      ctx.fillStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(px, py, hi ? 6.5 : 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, hi ? 4.75 : 3.75, 0, Math.PI * 2);
      ctx.globalAlpha = 0.25 + 0.75 * Math.max(0, Math.min(1, w));
      ctx.fillStyle = colors.text;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (hi) {
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
    ctx.restore();
  });

  // ── Dragging ─────────────────────────────────────────────────────────────────────────
  const drag = useRef<{ index: number; id: number } | null>(null);
  const toData = (clientX: number, clientY: number): DataPoint | null => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return null;
    const { x, y } = scales(r.width, r.height);
    const clamp = (v: number, [a, b]: [number, number]) => Math.min(b, Math.max(a, v));
    return {
      x: Number(clamp(x.invert(clientX - r.left), xd).toPrecision(5)),
      y: Number(clamp(y.invert(clientY - r.top), yd).toPrecision(5)),
    };
  };
  const move = (i: number, p: DataPoint) =>
    onPointsChange?.(points.map((q, j) => (j === i ? p : { ...q })));

  const onDown = (i: number) => (e: RPointerEvent<HTMLButtonElement>) => {
    if (!onPointsChange || e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { index: i, id: e.pointerId };
    setFrozen({ x: xd, y: yd });
  };
  const onMove = (e: RPointerEvent<HTMLButtonElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const p = toData(e.clientX, e.clientY);
    if (p) move(d.index, p);
  };
  const onUp = () => {
    drag.current = null;
    setFrozen(null);
  };
  const onKey = (i: number) => (e: RKeyboardEvent<HTMLButtonElement>) => {
    const f = keyStep * (e.shiftKey ? 10 : 1);
    const dx = (xd[1] - xd[0]) * f,
      dy = (yd[1] - yd[0]) * f;
    const p = points[i];
    const next: Record<string, DataPoint> = {
      ArrowLeft: { x: p.x - dx, y: p.y },
      ArrowRight: { x: p.x + dx, y: p.y },
      ArrowUp: { x: p.x, y: p.y + dy },
      ArrowDown: { x: p.x, y: p.y - dy },
    };
    const n = next[e.key];
    if (!n) return;
    e.preventDefault();
    e.stopPropagation();
    move(i, { x: Number(n.x.toPrecision(5)), y: Number(n.y.toPrecision(5)) });
  };

  const { x: sx, y: sy } = scales(size.width, size.height);
  return (
    <div
      ref={wrap}
      className={`${styles.wrap} ${className ?? ''}`}
      role="group"
      aria-label={ariaLabel}
    >
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        role="img"
        aria-label={`${ariaLabel}: ${points.length} points`}
      />
      {onPointsChange &&
        size.width > 0 &&
        points.map((p, i) => (
          <button
            key={i}
            type="button"
            className={styles.handle}
            data-own-keys
            style={{ left: sx(p.x), top: sy(p.y) }}
            aria-label={`Point ${i + 1}: x ${sig(p.x, 4)}, y ${sig(p.y, 4)}. Arrow keys move it.`}
            onPointerDown={onDown(i)}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}
            onKeyDown={onKey(i)}
          />
        ))}
    </div>
  );
}
