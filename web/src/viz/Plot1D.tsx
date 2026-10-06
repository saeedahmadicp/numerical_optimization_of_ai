import { useMemo, useRef, type PointerEvent as RPointerEvent } from 'react';
import { useChartColors } from '../ui/theme';
import type { ChartColors } from '../ui/colors';
import { useCanvas } from './useCanvas';
import { linearScale, logExtent, logScale, niceDomain, crisp, type Scale } from './scales';
import { drawAxes, labelFont, type Frame } from './axes';
import { adaptiveSample } from './sampling';
import { drawMath, type MathRun } from './mathText';
import styles from './viz.module.css';

type Label = string | readonly MathRun[];
type XY = readonly [number, number];

/**
 * Overlays drawn with the curve. `slot` picks a method color (0–3); omit it for ink.
 *
 * | kind       | draws                                                                         |
 * | ---------- | ----------------------------------------------------------------------------- |
 * | `interval` | a shaded x-interval [a, b] with a rule on the axis (bracket), optional label  |
 * | `area`     | the area between a function (default: the plotted f) and y = 0 on [from, to]  |
 * | `segment`  | a segment, optionally with an arrowhead (secant, step)                         |
 * | `arrow`    | an arrow from → to                                                             |
 * | `tangent`  | the tangent line through (x, y) with slope (Newton)                            |
 * | `parabola` | y = a x² + b x + c on [from, to] (model in a line search / Brent)              |
 * | `curve`    | another function y = g(x) (interpolant, Taylor model), dashed or solid         |
 * | `polyline` | points joined by lines; `fill: 'under'` shades down to y = 0 (trapezoids)      |
 * | `vline` / `hline` | a vertical / horizontal marker with an optional label                   |
 * | `point`    | a point (filled, hollow or a + cross) with an optional label                   |
 * | `text`     | a label at a data position                                                     |
 */
export type Overlay1D =
  | {
      kind: 'point';
      x: number;
      y: number;
      slot?: number;
      label?: Label;
      hollow?: boolean;
      shape?: 'dot' | 'ring' | 'cross';
      labelSide?: 'right' | 'left' | 'above' | 'below';
    }
  | {
      kind: 'segment';
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      slot?: number;
      dashed?: boolean;
      arrow?: boolean;
      width?: number;
    }
  | { kind: 'arrow'; from: XY; to: XY; slot?: number; label?: Label; dashed?: boolean }
  | { kind: 'interval'; a: number; b: number; slot?: number; label?: Label }
  | {
      kind: 'area';
      from: number;
      to: number;
      f?: (x: number) => number;
      slot?: number;
      alpha?: number;
    }
  | {
      kind: 'tangent';
      x: number;
      y: number;
      slope: number;
      halfWidth?: number;
      slot?: number;
      label?: Label;
    }
  | {
      kind: 'parabola';
      a: number;
      b: number;
      c: number;
      from: number;
      to: number;
      slot?: number;
      label?: Label;
    }
  | {
      kind: 'curve';
      f: (x: number) => number;
      from?: number;
      to?: number;
      slot?: number;
      dashed?: boolean;
      width?: number;
      label?: Label;
    }
  | {
      kind: 'polyline';
      points: readonly XY[];
      slot?: number;
      dashed?: boolean;
      /** 'under': shade between the polyline and y = 0; 'closed': fill the polygon. */
      fill?: 'none' | 'under' | 'closed';
      /** Draw a dot at each vertex. */
      dots?: boolean;
    }
  | { kind: 'vline'; x: number; slot?: number; dashed?: boolean; label?: Label }
  | { kind: 'hline'; y: number; slot?: number; dashed?: boolean; label?: Label }
  | {
      kind: 'text';
      x: number;
      y: number;
      text: Label;
      slot?: number;
      align?: 'left' | 'center' | 'right';
    };

export interface Plot1DProps {
  f: (x: number) => number;
  domain: [number, number];
  /** Fixed y-range; default: robust range of the sampled curve. */
  yDomain?: [number, number];
  /** Log y axis (positive values only). */
  logY?: boolean;
  overlays?: readonly Overlay1D[];
  /** Axis names; one letter is a variable (KaTeX italic). Use `xName`/`yName` for math runs. */
  xLabel?: string;
  yLabel?: string;
  xName?: readonly MathRun[];
  yName?: readonly MathRun[];
  ariaLabel: string;
  className?: string;
  /** Draw y = 0 as a reference line (root finding). */
  zeroLine?: boolean;
  /** Click → x in data coordinates (e.g. set a start point). Adds a crosshair cursor. */
  onPick?: (x: number) => void;
  /** Hide the curve of f (overlays only, e.g. data). */
  hideCurve?: boolean;
}

const M = { left: 46, right: 14, top: 22, bottom: 28 };

/** A function curve with axes and didactic overlays (brackets, tangents, model parabolas, ...). */
export function Plot1D({
  f,
  domain,
  yDomain,
  logY = false,
  overlays = [],
  xLabel,
  yLabel,
  xName,
  yName,
  ariaLabel,
  className,
  zeroLine,
  onPick,
  hideCurve = false,
}: Plot1DProps) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const autoY = useMemo<[number, number]>(() => {
    if (yDomain) return yDomain;
    const ys: number[] = [];
    for (let i = 0; i <= 400; i++) {
      const v = f(domain[0] + ((domain[1] - domain[0]) * i) / 400);
      if (Number.isFinite(v) && (!logY || v > 0)) ys.push(v);
    }
    if (logY) return logExtent(ys);
    if (!ys.length) return [-1, 1];
    ys.sort((p, q) => p - q);
    const lo = ys[Math.floor(ys.length * 0.01)],
      hi = ys[Math.floor(ys.length * 0.99)];
    const pad = (hi - lo) * 0.08 || 1;
    return niceDomain(lo - pad, hi + pad);
  }, [f, domain, yDomain, logY]);

  const scales = (w: number, h: number) => ({
    x: linearScale(domain, [M.left, w - M.right]),
    y: logY ? logScale(autoY, [h - M.bottom, M.top]) : linearScale(autoY, [h - M.bottom, M.top]),
  });

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame: Frame = {
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

    // Shading first: intervals, areas, filled polylines.
    for (const o of overlays) {
      if (o.kind === 'interval') drawInterval(ctx, o, x, frame, color(o.slot), colors);
      if (o.kind === 'area') {
        const g = o.f ?? f;
        ctx.fillStyle = color(o.slot);
        ctx.globalAlpha = o.alpha ?? 0.14;
        ctx.beginPath();
        ctx.moveTo(...P(o.from, logY ? autoY[0] : 0));
        for (const [a, b] of adaptiveSample(g, o.from, o.to, P)) {
          const [px, py] = P(a, b);
          if (Number.isFinite(py)) ctx.lineTo(px, clampPx(py));
        }
        ctx.lineTo(...P(o.to, logY ? autoY[0] : 0));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (o.kind === 'polyline' && o.fill && o.fill !== 'none' && o.points.length > 1) {
        ctx.fillStyle = color(o.slot);
        ctx.globalAlpha = 0.13;
        ctx.beginPath();
        if (o.fill === 'under') ctx.moveTo(...P(o.points[0][0], 0));
        for (const [a, b] of o.points) ctx.lineTo(...P(a, b));
        if (o.fill === 'under') ctx.lineTo(...P(o.points[o.points.length - 1][0], 0));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
    if (zeroLine && !logY && autoY[0] < 0 && autoY[1] > 0) {
      ctx.strokeStyle = colors.axis;
      ctx.lineWidth = 1 / s.dpr;
      ctx.beginPath();
      ctx.moveTo(frame.left, crisp(y(0), s.dpr));
      ctx.lineTo(frame.right, crisp(y(0), s.dpr));
      ctx.stroke();
    }

    // The curve.
    if (!hideCurve) strokeFunction(ctx, f, domain[0], domain[1], P, colors.text, 1.75, 0.85);

    for (const o of overlays) {
      const c = color(o.slot);
      ctx.strokeStyle = c;
      ctx.fillStyle = c;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      switch (o.kind) {
        case 'segment':
          if (o.dashed) ctx.setLineDash([4, 4]);
          ctx.lineWidth = o.width ?? 1.5;
          ctx.beginPath();
          ctx.moveTo(...P(o.x1, o.y1));
          ctx.lineTo(...P(o.x2, o.y2));
          ctx.stroke();
          if (o.arrow) arrowHead(ctx, P(o.x1, o.y1), P(o.x2, o.y2));
          break;
        case 'arrow': {
          if (o.dashed) ctx.setLineDash([4, 4]);
          const a = P(o.from[0], o.from[1]),
            b = P(o.to[0], o.to[1]);
          ctx.beginPath();
          ctx.moveTo(...a);
          ctx.lineTo(...b);
          ctx.stroke();
          ctx.setLineDash([]);
          arrowHead(ctx, a, b);
          if (o.label)
            label(ctx, o.label, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 10, colors, 'center');
          break;
        }
        case 'tangent': {
          const h = o.halfWidth ?? (domain[1] - domain[0]) * 0.18;
          ctx.beginPath();
          ctx.moveTo(...P(o.x - h, o.y - o.slope * h));
          ctx.lineTo(...P(o.x + h, o.y + o.slope * h));
          ctx.stroke();
          if (o.label) {
            const [lx, ly] = P(o.x + h, o.y + o.slope * h);
            label(ctx, o.label, lx - 4, ly - 10, colors, 'right');
          }
          break;
        }
        case 'parabola': {
          ctx.setLineDash([5, 4]);
          strokeFunction(ctx, (xx) => o.a * xx * xx + o.b * xx + o.c, o.from, o.to, P, c, 1.5, 1);
          if (o.label) {
            const [lx, ly] = P(o.to, o.a * o.to * o.to + o.b * o.to + o.c);
            label(ctx, o.label, lx + 4, ly, colors, 'left');
          }
          break;
        }
        case 'curve': {
          if (o.dashed) ctx.setLineDash([5, 4]);
          const from = o.from ?? domain[0],
            to = o.to ?? domain[1];
          strokeFunction(ctx, o.f, from, to, P, c, o.width ?? 1.75, 1);
          if (o.label) {
            const [lx, ly] = P(to, o.f(to));
            label(ctx, o.label, lx - 6, clampPx(ly) - 10, colors, 'right');
          }
          break;
        }
        case 'polyline': {
          if (o.points.length < 2) break;
          if (o.dashed) ctx.setLineDash([5, 4]);
          ctx.beginPath();
          o.points.forEach(([a, b], i) => (i ? ctx.lineTo(...P(a, b)) : ctx.moveTo(...P(a, b))));
          ctx.stroke();
          if (o.dots) for (const [a, b] of o.points) dot(ctx, P(a, b), c, colors.halo, 3);
          break;
        }
        case 'vline':
        case 'hline':
          if (o.dashed) ctx.setLineDash([3, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          if (o.kind === 'vline') {
            ctx.moveTo(crisp(x(o.x), s.dpr), frame.top);
            ctx.lineTo(crisp(x(o.x), s.dpr), frame.bottom);
          } else {
            ctx.moveTo(frame.left, crisp(y(o.y), s.dpr));
            ctx.lineTo(frame.right, crisp(y(o.y), s.dpr));
          }
          ctx.stroke();
          if (o.label) {
            if (o.kind === 'vline') label(ctx, o.label, x(o.x) + 5, frame.top + 10, colors, 'left');
            else label(ctx, o.label, frame.right - 6, y(o.y) - 9, colors, 'right');
          }
          break;
        default:
          break;
      }
    }
    ctx.setLineDash([]);
    for (const o of overlays) {
      if (o.kind === 'text') {
        const [px, py] = P(o.x, o.y);
        label(
          ctx,
          o.text,
          px,
          py,
          colors,
          o.align ?? 'left',
          o.slot === undefined ? undefined : color(o.slot),
        );
      }
      if (o.kind !== 'point') continue;
      const [px, py] = P(o.x, o.y);
      const shape = o.shape ?? (o.hollow ? 'ring' : 'dot');
      if (shape === 'cross') {
        for (const [w, cc] of [
          [3.5, colors.halo],
          [1.5, color(o.slot)],
        ] as const) {
          ctx.strokeStyle = cc;
          ctx.lineWidth = w;
          ctx.beginPath();
          ctx.moveTo(px - 5.5, py);
          ctx.lineTo(px + 5.5, py);
          ctx.moveTo(px, py - 5.5);
          ctx.lineTo(px, py + 5.5);
          ctx.stroke();
        }
      } else {
        ctx.fillStyle = colors.halo;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, 4.25, 0, Math.PI * 2);
        if (shape === 'ring') {
          ctx.strokeStyle = color(o.slot);
          ctx.lineWidth = 1.75;
          ctx.stroke();
        } else {
          ctx.fillStyle = color(o.slot);
          ctx.fill();
        }
      }
      if (o.label) {
        const side = o.labelSide ?? 'right';
        const [lx, ly, al] =
          side === 'right'
            ? [px + 10, py, 'left' as const]
            : side === 'left'
              ? [px - 10, py, 'right' as const]
              : side === 'above'
                ? [px, py - 14, 'center' as const]
                : [px, py + 16, 'center' as const];
        label(ctx, o.label, lx, ly, colors, al);
      }
    }
    ctx.restore();
  });

  const pick = (e: RPointerEvent<HTMLDivElement>) => {
    const r = box.current?.getBoundingClientRect();
    if (!r || !onPick) return;
    const { x } = scales(r.width, r.height);
    const v = x.invert(e.clientX - r.left);
    if (v >= domain[0] && v <= domain[1]) onPick(Number(v.toPrecision(4)));
  };

  return (
    <div
      ref={box}
      className={`${styles.chart} ${className ?? ''}`}
      role="img"
      aria-label={ariaLabel}
      style={onPick ? { cursor: 'crosshair' } : undefined}
      onPointerUp={onPick ? pick : undefined}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}

const clampPx = (v: number) => Math.max(-1e5, Math.min(1e5, v));

function strokeFunction(
  ctx: CanvasRenderingContext2D,
  g: (x: number) => number,
  from: number,
  to: number,
  P: (a: number, b: number) => [number, number],
  color: string,
  width: number,
  alpha: number,
) {
  const pts = adaptiveSample(g, from, to, P);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let pen = false;
  for (const [a, b] of pts) {
    const [px, py] = P(a, b);
    if (!Number.isFinite(py) || Math.abs(py) > 1e6) {
      pen = false;
      continue;
    }
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
  ctx.restore();
}

function dot(
  ctx: CanvasRenderingContext2D,
  [px, py]: [number, number],
  fill: string,
  halo: string,
  r: number,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(px, py, r + 1.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();
}

function arrowHead(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number]) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1) return;
  const ux = dx / len,
    uy = dy / len,
    s = Math.min(8, len * 0.45);
  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - ux * s - uy * s * 0.5, b[1] - uy * s + ux * s * 0.5);
  ctx.lineTo(b[0] - ux * s + uy * s * 0.5, b[1] - uy * s - ux * s * 0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function label(
  ctx: CanvasRenderingContext2D,
  text: Label,
  x: number,
  y: number,
  colors: ChartColors,
  align: 'left' | 'center' | 'right',
  ink?: string,
) {
  if (typeof text !== 'string') {
    drawMath(ctx, text, x, y, {
      size: 12,
      align,
      baseline: 'middle',
      color: ink ?? colors.text2,
      halo: colors.halo,
    });
    return;
  }
  ctx.save();
  ctx.font = labelFont(colors);
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = colors.halo;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = ink ?? colors.text2;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawInterval(
  ctx: CanvasRenderingContext2D,
  o: Extract<Overlay1D, { kind: 'interval' }>,
  x: Scale,
  frame: { top: number; bottom: number },
  color: string,
  colors: ChartColors,
) {
  const a = x(Math.min(o.a, o.b)),
    b = x(Math.max(o.a, o.b));
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.1;
  ctx.fillRect(a, frame.top, b - a, frame.bottom - frame.top);
  ctx.globalAlpha = 0.6;
  ctx.fillRect(a, frame.bottom - 3, b - a, 3);
  ctx.restore();
  // Text stays in text tokens (never the series color).
  if (o.label) label(ctx, o.label, (a + b) / 2, frame.top + 12, colors, 'center');
}
