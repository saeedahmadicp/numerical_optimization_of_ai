/**
 * The interpolation stage: the data nodes (draggable buttons), f when it is known, and for each
 * method the approximant of its current step with the geometry that step computed —
 * the new node and the term it adds (Lagrange yₖℓₖ, Newton's correction), the previous
 * interpolant as a ghost, Neville's column at x⋆, the Chebyshev samples, and for the splines
 * the secants, the node slopes and the final pieces in alternating tones. AAA: the support
 * point the step chose (z_k), the samples it has not used (hollow) and the certified real poles
 * of r_k (dashed verticals, × on the axis). Floater–Hormann: the window of local polynomials
 * p_i that contain node k, drawn over their own nodes.
 */
import {
  memo,
  useRef,
  useState,
  type KeyboardEvent as RKeyboardEvent,
  type PointerEvent as RPointerEvent,
} from 'react';
import type { Result } from '../../core/types';
import { sig } from '../../core/format';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import { drawAxes, drawMath, linearScale, useCanvas, useElementSize } from '../../viz';
import { mathSub as msub, mathSup as msup, mathVar as mv } from '../../viz';
import type { Scale } from '../../viz';
import {
  MAX_NODES,
  MIN_NODES,
  PLOT_MARGIN,
  aaaIntervalPoles,
  aaaSupport,
  blendWindow,
  localPoly,
  offViewPeaks,
  round6,
  sampled,
  stepFunction,
  stepTerm,
  type ActiveData,
  type Fn,
  type MethodKind,
} from './geometry';
import styles from './InterpolationLab.module.css';

export interface RunView {
  id: string;
  /** Identifies the run's data and parameters (cache key for its sampled curves). */
  key: string;
  slot: number;
  name: string;
  kind: MethodKind;
  result: Result;
  /** The method's own step under the playhead. */
  k: number;
  focus: boolean;
}

export interface InterpPlotProps {
  data: ActiveData;
  runs: readonly RunView[];
  yDomain: [number, number];
  /** Dragging moves a node along f (y = f(x)) instead of freely. */
  snap: boolean;
  /** Nodes moved (`final` false while dragging, true on release / keyboard / add / remove). */
  onNodesChange: (x: number[], y: number[], final: boolean) => void;
  onDragChange: (dragging: boolean) => void;
  /** The browser took the gesture over (a scroll): drop the moves of this drag. */
  onDragCancel: () => void;
  ariaLabel: string;
}

const MARGIN = PLOT_MARGIN;

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);

/**
 * Stroke a function over [lo, hi] as one path (gaps where it is not finite). The samples are
 * cached under `key`, so playback frames between two steps do not re-evaluate the curve. With
 * `poleBreak` (the frame's top and bottom) a step that jumps from above the frame to below it,
 * or back, lifts the pen: a rational function's pole is a gap, not a vertical line.
 */
function strokeFn(
  ctx: CanvasRenderingContext2D,
  key: string,
  g: Fn,
  lo: number,
  hi: number,
  P: (x: number, y: number) => [number, number],
  samples = 601,
  poleBreak?: { top: number; bottom: number },
) {
  const ys = sampled(key, g, lo, hi, samples);
  ctx.beginPath();
  let pen = false;
  let prev = NaN;
  for (let i = 0; i < ys.length; i++) {
    const [px, py0] = P(lo + ((hi - lo) * i) / (ys.length - 1), ys[i]);
    if (!Number.isFinite(py0)) {
      pen = false;
      continue;
    }
    if (
      pen &&
      poleBreak &&
      ((prev < poleBreak.top && py0 > poleBreak.bottom) ||
        (prev > poleBreak.bottom && py0 < poleBreak.top))
    )
      pen = false;
    prev = py0;
    const py = Math.max(-4000, Math.min(4000, py0));
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
}

function haloStroke(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  draw: () => void,
  color: string,
  width: number,
  alpha = 1,
  dash: number[] = [],
) {
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = colors.halo;
  ctx.lineWidth = width + 2.5;
  ctx.globalAlpha = 0.75 * alpha;
  ctx.setLineDash([]);
  draw();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  draw();
  ctx.restore();
}

function dot(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  [px, py]: [number, number],
  r: number,
  fill: string | null,
  stroke?: string,
) {
  ctx.beginPath();
  ctx.arc(px, py, r + 1.75, 0, Math.PI * 2);
  ctx.fillStyle = colors.halo;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function diamond(
  ctx: CanvasRenderingContext2D,
  [px, py]: [number, number],
  r: number,
  color: string,
  halo: string,
) {
  ctx.beginPath();
  ctx.moveTo(px, py - r - 1.5);
  ctx.lineTo(px + r + 1.5, py);
  ctx.lineTo(px, py + r + 1.5);
  ctx.lineTo(px - r - 1.5, py);
  ctx.closePath();
  ctx.fillStyle = halo;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(px, py - r);
  ctx.lineTo(px + r, py);
  ctx.lineTo(px, py + r);
  ctx.lineTo(px - r, py);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

/** A filled (or hollow) chevron at the frame edge pointing out of the view. */
function chevron(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  up: boolean,
  color: string,
  hollow = false,
) {
  const d = up ? -1 : 1;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(px - 5, py - d);
  ctx.lineTo(px, py + d * 5);
  ctx.lineTo(px + 5, py - d);
  ctx.closePath();
  if (hollow) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

/** A haloed monospace value at the frame edge (off-view peaks and estimates). */
function edgeLabel(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  label: string,
  tx: number,
  ty: number,
  align: CanvasTextAlign,
  up: boolean,
) {
  ctx.save();
  ctx.font = `500 11px ${colors.fontMono}`;
  ctx.textAlign = align;
  ctx.textBaseline = up ? 'top' : 'bottom';
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = colors.halo;
  ctx.strokeText(label, tx, ty);
  ctx.fillStyle = colors.text2;
  ctx.fillText(label, tx, ty);
  ctx.restore();
}

/**
 * Whether data node i is used by the run at its step: a node-adding method has used x_0 … x_k;
 * AAA its support points so far; Floater–Hormann the nodes whose weight is computed (sorted
 * nodes 0 … k). Every other method uses every node.
 */
function usedTest(run: RunView, data: ActiveData): (i: number) => boolean {
  if (run.kind === 'nodewise' || run.kind === 'neville') return (i) => i <= run.k;
  if (run.kind === 'aaa') {
    const used = new Set(aaaSupport(run.result, run.k));
    return (i) => used.has(i);
  }
  if (run.kind === 'blend') {
    const sorted = nums(run.result.extra.nodes);
    const done = new Set(sorted.slice(0, run.k + 1));
    return (i) => done.has(data.x[i]);
  }
  return () => true;
}

function InterpPlotImpl({
  data,
  runs,
  yDomain,
  snap,
  onNodesChange,
  onDragChange,
  onDragCancel,
  ariaLabel,
}: InterpPlotProps) {
  const colors = useChartColors();
  const wrap = useRef<HTMLDivElement>(null);
  const size = useElementSize(wrap);
  const [a, b] = data.domain;
  const xPad = 0.03 * (b - a);
  const xd: [number, number] = [a - xPad, b + xPad];

  const scales = (w: number, h: number): { x: Scale; y: Scale } => ({
    x: linearScale(xd, [MARGIN.left, w - MARGIN.right]),
    y: linearScale(yDomain, [h - MARGIN.bottom, MARGIN.top]),
  });

  const focus = runs.find((r) => r.focus) ?? runs[0];

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: MARGIN.left,
      top: MARGIN.top,
      right: s.width - MARGIN.right,
      bottom: s.height - MARGIN.bottom,
    };
    drawAxes(ctx, { x, y, frame, colors, dpr: s.dpr, xLabel: 'x', yLabel: 'y' });
    const P = (u: number, v: number): [number, number] => [x(u), y(v)];
    const series = (slot: number) => colors.series[slot % colors.series.length];

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 6, frame.right - frame.left, frame.bottom - frame.top + 12);
    ctx.clip();

    // f, the function the data sample (context, not a method): ink, dashed.
    if (data.fTrue) {
      const f = data.fTrue;
      haloStroke(
        ctx,
        colors,
        () => strokeFn(ctx, `f|${data.baseId}`, f, a, b, P),
        colors.text3,
        1.3,
        0.9,
        [5, 4],
      );
    }

    // Methods: the others first, the focused one last (on top) with its step geometry.
    const ordered = [...runs].sort((p, q) => Number(p.focus) - Number(q.focus));
    const late: (() => void)[] = [];
    for (const run of ordered) {
      const color = series(run.slot);
      const res = run.result;
      if (!res.trace.length) continue;
      const k = Math.min(run.k, res.trace.length - 1);
      const st = res.trace[k];
      const info = st.info;
      const isFocus = run === focus;

      if (run.kind === 'piecewise' && info.stage !== 'coefficients') {
        // Spline stages before the pieces exist: the secants, then the node slopes.
        const nodes = nums(res.extra.nodes).length ? nums(res.extra.nodes) : sortedNodes(data).x;
        const vals = nums(res.extra.values).length ? nums(res.extra.values) : sortedNodes(data).y;
        haloStroke(
          ctx,
          colors,
          () => {
            ctx.beginPath();
            nodes.forEach((xi, i) =>
              i ? ctx.lineTo(...P(xi, vals[i])) : ctx.moveTo(...P(xi, vals[i])),
            );
            ctx.stroke();
          },
          color,
          1.4,
          info.stage === 'back_substitution' || info.stage === 'slopes' ? 0.45 : 0.9,
          [4, 4],
        );
        const slopes = nums(info.slopes);
        if (slopes.length === nodes.length) {
          const limited = (info.limited as boolean[] | undefined) ?? [];
          nodes.forEach((xi, i) => {
            const [px, py] = P(xi, vals[i]);
            const dx = x(xi + 1) - x(xi),
              dy = y(vals[i] + slopes[i]) - y(vals[i]);
            const len = Math.hypot(dx, dy) || 1;
            const L = 20;
            const ux = (dx / len) * L,
              uy = (dy / len) * L;
            haloStroke(
              ctx,
              colors,
              () => {
                ctx.beginPath();
                ctx.moveTo(px - ux, py - uy);
                ctx.lineTo(px + ux, py + uy);
                ctx.stroke();
              },
              color,
              2.2,
              1,
              limited[i] ? [3, 3] : [],
            );
          });
        }
        continue;
      }

      const g = stepFunction(run.id, res, k, data);
      const key = `${run.key}|${k}`;
      const rational = run.kind === 'aaa' || run.kind === 'blend';
      const poleBreak = rational ? { top: frame.top, bottom: frame.bottom } : undefined;

      // Floater–Hormann: the local polynomials p_i (degree d through d + 1 consecutive sorted
      // nodes) whose window J_k contains node k, each over its own nodes, on a light band.
      if (run.kind === 'blend' && isFocus) {
        const win = blendWindow(res, k);
        const nodes = nums(res.extra.nodes),
          vals = nums(res.extra.values);
        if (win && nodes.length) {
          const left = x(nodes[win.lo]),
            right = x(nodes[Math.min(nodes.length - 1, win.hi + win.d)]);
          ctx.save();
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.07;
          ctx.fillRect(
            Math.min(left, right) - 4,
            frame.top - 6,
            Math.abs(right - left) + 8,
            frame.bottom - frame.top + 12,
          );
          ctx.restore();
          for (let i = win.lo; i <= win.hi; i++) {
            const lo = nodes[i],
              hi = nodes[Math.min(nodes.length - 1, i + win.d)];
            if (!(hi > lo)) continue;
            const p = localPoly(nodes, vals, i, win.d);
            haloStroke(
              ctx,
              colors,
              () => strokeFn(ctx, `${run.key}|p${i}|${win.d}`, p, lo, hi, P, 121),
              color,
              1.3,
              0.75,
              [6, 3],
            );
          }
        }
      }
      if (!g) continue;

      if (isFocus) {
        // The previous approximant (ghost) and the term this step adds.
        if (run.kind !== 'piecewise' && k > 0) {
          const prev = stepFunction(run.id, res, k - 1, data);
          // Same cache key as step k − 1's own curve: the ghost costs nothing during playback.
          if (prev)
            haloStroke(
              ctx,
              colors,
              () => strokeFn(ctx, `${run.key}|${k - 1}`, prev, a, b, P, 601, poleBreak),
              color,
              1.2,
              0.35,
              [2, 3],
            );
        }
        const term = stepTerm(run.id, res, k, data);
        if (term)
          haloStroke(
            ctx,
            colors,
            () => strokeFn(ctx, `${key}|term`, term, a, b, P),
            color,
            1.1,
            0.55,
            [7, 3, 1.5, 3],
          );
      }

      if (run.kind === 'piecewise') {
        // Pieces in alternating tones: one cubic (or line) per interval [xᵢ, xᵢ₊₁].
        const nodes = nums(res.extra.nodes);
        for (let i = 0; i + 1 < nodes.length; i++) {
          const lo = nodes[i],
            hi = nodes[i + 1];
          haloStroke(
            ctx,
            colors,
            () => strokeFn(ctx, `${key}|s${i}`, g, lo, hi, P, 81),
            color,
            isFocus ? 2.4 : 1.8,
            i % 2 ? 0.5 : 1,
          );
        }
        // The end pieces extrapolate (as in Python's evaluator): thin and dashed.
        if (nodes.length) {
          haloStroke(
            ctx,
            colors,
            () => strokeFn(ctx, `${key}|l`, g, a, nodes[0], P, 21),
            color,
            1.2,
            0.6,
            [3, 3],
          );
          haloStroke(
            ctx,
            colors,
            () => strokeFn(ctx, `${key}|r`, g, nodes[nodes.length - 1], b, P, 21),
            color,
            1.2,
            0.6,
            [3, 3],
          );
        }
      } else {
        haloStroke(
          ctx,
          colors,
          () => strokeFn(ctx, key, g, a, b, P, 601, poleBreak),
          color,
          isFocus ? 2.4 : 1.8,
          1,
        );
      }

      // AAA: the certified real poles of r_k in [a, b], a dashed vertical each with a cross on
      // the x-axis (drawn after the curves, so no curve hides them).
      if (run.kind === 'aaa') {
        const poles = aaaIntervalPoles(res, k);
        late.push(() => {
          for (const p of poles) {
            const px = x(p);
            ctx.save();
            ctx.strokeStyle = color;
            ctx.globalAlpha = isFocus ? 0.75 : 0.45;
            ctx.lineWidth = 1.2;
            ctx.setLineDash([3, 4]);
            ctx.beginPath();
            ctx.moveTo(px, frame.top);
            ctx.lineTo(px, frame.bottom);
            ctx.stroke();
            ctx.restore();
            cross(ctx, colors, px, frame.bottom - 6, 4.5, color);
          }
        });
      }

      // Chebyshev in resampling mode: its own sample points (the roots of T_N), drawn after
      // every curve so that no other method hides them.
      if (run.kind === 'chebyshev' && res.extra.source === 'f_true') {
        const cn = nums(res.extra.nodes),
          cv = nums(res.extra.values);
        late.push(() => cn.forEach((xi, i) => diamond(ctx, P(xi, cv[i]), 3.6, color, colors.halo)));
      }

      // Off-view excursions of the curve: a chevron at the edge and the peak value (never
      // silently clipped). Placed first so the Neville guide can leave their labels clear.
      const peakBoxes: { top: number; bottom: number; left: number; right: number }[] = [];
      const peakLabels: (() => void)[] = [];
      if (isFocus || runs.length === 1) {
        const lo = yDomain[0],
          hi = yDomain[1];
        // Next to a certified pole of AAA's r the excursion is unbounded: the pole marker
        // says so, and a peak value there would only be the sample nearest the pole.
        const poles = run.kind === 'aaa' ? aaaIntervalPoles(res, k) : [];
        const nearPole = (px: number) => poles.some((p) => Math.abs(p - px) < 0.02 * (b - a));
        const peaks = offViewPeaks(sampled(key, g, a, b), a, b, lo, hi)
          .filter((pk) => !nearPole(pk.x))
          .sort((p, q) => Math.abs(q.y) - Math.abs(p.y))
          .slice(0, 2);
        ctx.save();
        ctx.font = `500 11px ${colors.fontMono}`;
        for (const pk of peaks) {
          const up = pk.y > hi;
          const px = x(pk.x);
          const label = `${up ? '↑' : '↓'} ${sig(pk.y, 3)}`;
          const right = px > (frame.left + frame.right) / 2;
          const tx = px + (right ? -8 : 8);
          const ty = up ? frame.top + 4 : frame.bottom - 4;
          const w = ctx.measureText(label).width;
          peakBoxes.push({
            left: right ? tx - w - 3 : tx - 3,
            right: right ? tx + 3 : tx + w + 3,
            top: up ? ty - 2 : ty - 15,
            bottom: up ? ty + 15 : ty + 2,
          });
          peakLabels.push(() => {
            chevron(ctx, px, up ? frame.top + 2 : frame.bottom - 2, up, color);
            edgeLabel(ctx, colors, label, tx, ty, right ? 'right' : 'left', up);
          });
        }
        ctx.restore();
      }

      // Neville: the column of estimates at x⋆, one per window of k + 1 consecutive nodes.
      // Estimates beyond the view sit as one chevron per edge with their value, not clamped dots.
      if (run.kind === 'neville' && isFocus) {
        const xs = info.x_eval as number;
        const col = nums(info.column);
        const px = x(xs);
        const above = col
          .map((q, i) => ({ q, i }))
          .filter(({ q }) => Number.isFinite(q) && y(q) < frame.top);
        const below = col
          .map((q, i) => ({ q, i }))
          .filter(({ q }) => Number.isFinite(q) && y(q) > frame.bottom);
        const starTop = below.length > 0 && above.length === 0;
        const starBox = starTop
          ? { top: frame.top, bottom: frame.top + 20 }
          : { top: frame.bottom - 20, bottom: frame.bottom };
        // The guide, interrupted where a label crosses it.
        const gaps = [
          ...peakBoxes.filter((bx) => bx.left <= px + 2 && bx.right >= px - 2),
          ...(above.length ? [{ top: frame.top, bottom: frame.top + 22 }] : []),
          ...(below.length ? [{ top: frame.bottom - 22, bottom: frame.bottom }] : []),
          starBox,
        ].sort((p, q) => p.top - q.top);
        ctx.save();
        ctx.strokeStyle = colors.text3;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        let from = frame.top;
        for (const gp of gaps) {
          if (gp.top > from) {
            ctx.moveTo(px, from);
            ctx.lineTo(px, gp.top);
          }
          from = Math.max(from, gp.bottom);
        }
        if (from < frame.bottom) {
          ctx.moveTo(px, from);
          ctx.lineTo(px, frame.bottom);
        }
        ctx.stroke();
        ctx.restore();
        col.forEach((q, i) => {
          if (!Number.isFinite(q)) return;
          const py = y(q);
          if (py < frame.top || py > frame.bottom) return;
          dot(ctx, colors, [px, py], i === 0 ? 4 : 2.6, i === 0 ? color : colors.halo, color);
        });
        const right = px > (frame.left + frame.right) / 2;
        for (const [set, up] of [
          [above, true],
          [below, false],
        ] as const) {
          if (!set.length) continue;
          const ext = set.reduce((m, e) => (Math.abs(e.q) > Math.abs(m.q) ? e : m), set[0]);
          const diag = set.find((e) => e.i === 0);
          const shown = diag ?? ext;
          const label =
            set.length === 1
              ? `${up ? '↑' : '↓'} ${sig(shown.q, 3)}`
              : `${up ? '↑' : '↓'} ${set.length} estimates${diag ? `, P₀‥ₖ ${sig(diag.q, 3)}` : `, to ${sig(ext.q, 3)}`}`;
          chevron(ctx, px, up ? frame.top + 2 : frame.bottom - 2, up, color, !diag);
          edgeLabel(
            ctx,
            colors,
            label,
            px + (right ? -9 : 9),
            up ? frame.top + 18 : frame.bottom - 18,
            right ? 'right' : 'left',
            up,
          );
        }
        drawMath(
          ctx,
          [mv('x'), msup('⋆')],
          px + (right ? -5 : 5),
          starTop ? frame.top + 14 : frame.bottom - 6,
          {
            size: 13,
            align: right ? 'right' : 'left',
            color: colors.text2,
            halo: colors.halo,
          },
        );
      }

      for (const draw of peakLabels) draw();
    }

    for (const draw of late) draw();

    // Data nodes: filled once the focused method has used them, hollow before (AAA: the
    // samples that are not support points stay hollow).
    const used = focus ? usedTest(focus, data) : () => true;
    data.x.forEach((xi, i) => {
      const p = P(xi, data.y[i]);
      if (used(i)) dot(ctx, colors, p, 3.9, colors.text);
      else dot(ctx, colors, p, 3.4, colors.surface, colors.text2);
    });

    // The node this step added: a ring in the method's color, labelled xₖ (AAA: the support
    // point zₖ; Floater–Hormann: the sorted node xₖ whose weight this step computed).
    const ringNode = focus ? ringOf(focus, data) : null;
    if (focus && ringNode) {
      const { k, nx, ny, name } = ringNode;
      const p = P(nx, ny);
      ctx.save();
      ctx.strokeStyle = colors.series[focus.slot % colors.series.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      const above = p[1] > frame.top + 30;
      const right = p[0] > frame.right - 48;
      drawMath(
        ctx,
        [mv(name), msub(String(k), 'main')],
        right ? p[0] - 10 : p[0] + 9,
        above ? p[1] - 10 : p[1] + 22,
        {
          size: 13,
          align: right ? 'right' : 'left',
          color: colors.text,
          halo: colors.halo,
        },
      );
    }
    ctx.restore();

    // The name of f at the right end of its curve.
    if (data.fTrue) {
      const xa = a + 0.015 * (b - a);
      const ya = data.fTrue(xa);
      if (Number.isFinite(ya) && ya >= yDomain[0] && ya <= yDomain[1])
        drawMath(ctx, [mv('f')], x(xa) - 4, y(ya) + 18, {
          size: 13,
          align: 'right',
          color: colors.text3,
          halo: colors.halo,
        });
    }
  });

  // ── Direct manipulation ──────────────────────────────────────────────────────────────
  const drag = useRef<{ index: number; id: number; moved: boolean } | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const span = b - a;
  const minGap = 1e-4 * span;

  const clampX = (v: number) => Math.min(b, Math.max(a, v));
  const distinct = (xs: readonly number[], v: number, skip: number) =>
    xs.every((xi, j) => j === skip || Math.abs(xi - v) > minGap);
  const yAt = (xv: number, fallback: number) => {
    if (snap && data.fTrue) {
      const v = data.fTrue(xv);
      if (Number.isFinite(v)) return round6(v);
    }
    return fallback;
  };

  const toData = (clientX: number, clientY: number) => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return null;
    const { x, y } = scales(r.width, r.height);
    return { x: x.invert(clientX - r.left), y: y.invert(clientY - r.top) };
  };

  const moveNode = (i: number, xv: number, yv: number, final: boolean) => {
    const nx = round6(clampX(xv));
    if (!distinct(data.x, nx, i)) return;
    const ny = yAt(nx, round6(yv));
    if (nx === data.x[i] && ny === data.y[i]) return; // e.g. ↑/↓ on a node held on f
    const xs = data.x.slice(),
      ys = data.y.slice();
    xs[i] = nx;
    ys[i] = ny;
    onNodesChange(xs, ys, final);
  };

  // A drag starts with the first move, not on pointer-down: a touch that becomes a page scroll
  // (the browser cancels the pointer) leaves the data and the playback untouched.
  const onDown = (i: number) => (e: RPointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    if (e.pointerType === 'mouse') e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { index: i, id: e.pointerId, moved: false };
  };
  const onMove = (e: RPointerEvent<HTMLButtonElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const p = toData(e.clientX, e.clientY);
    if (!p) return;
    if (!d.moved) {
      d.moved = true;
      setDragging(d.index);
      onDragChange(true);
    }
    moveNode(d.index, p.x, p.y, false);
  };
  const onUp = () => {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    if (!d.moved) return;
    setDragging(null);
    onNodesChange(data.x.slice(), data.y.slice(), true);
    onDragChange(false);
  };
  const onCancel = () => {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    if (!d.moved) return;
    setDragging(null);
    onDragCancel();
    onDragChange(false);
  };

  const removeNode = (i: number) => {
    if (data.x.length <= MIN_NODES) return;
    onNodesChange(
      data.x.filter((_, j) => j !== i),
      data.y.filter((_, j) => j !== i),
      true,
    );
  };

  // ── Keyboard: the nodes are one composite widget (a single tab stop, roving tabindex).
  // ←/→ (Home/End) walk the nodes in x order; Enter or Space grabs the focused node, the arrow
  // keys then move it (Shift: ten times further) and Enter, Space or Escape drops it.
  const [active, setActive] = useState(0);
  const [grabbed, setGrabbed] = useState<number | null>(null);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const activeIndex = Math.min(active, data.x.length - 1);
  const grabbedIndex = grabbed !== null && grabbed < data.x.length ? grabbed : null;
  const byX = data.x.map((_, i) => i).sort((i, j) => data.x[i] - data.x[j] || i - j);
  const focusNode = (i: number) => {
    setActive(i);
    buttons.current[i]?.focus();
  };

  const onKey = (i: number) => (e: RKeyboardEvent<HTMLButtonElement>) => {
    const own = () => {
      e.preventDefault();
      e.stopPropagation();
    };
    if (e.key === 'Delete' || e.key === 'Backspace') {
      own();
      setGrabbed(null);
      removeNode(i);
      return;
    }
    if (e.key === 'Enter' || e.key === ' ' || (e.key === 'Escape' && grabbedIndex === i)) {
      own();
      setGrabbed(grabbedIndex === i || e.key === 'Escape' ? null : i);
      return;
    }
    if (grabbedIndex !== i) {
      const pos = byX.indexOf(i);
      const to =
        e.key === 'ArrowLeft' || e.key === 'ArrowDown'
          ? byX[Math.max(0, pos - 1)]
          : e.key === 'ArrowRight' || e.key === 'ArrowUp'
            ? byX[Math.min(byX.length - 1, pos + 1)]
            : e.key === 'Home'
              ? byX[0]
              : e.key === 'End'
                ? byX[byX.length - 1]
                : null;
      if (to === null || to === undefined) return;
      own();
      focusNode(to);
      return;
    }
    const f = e.shiftKey ? 0.1 : 0.01;
    const dx = span * f,
      dy = (yDomain[1] - yDomain[0]) * f;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-dx, 0],
      ArrowRight: [dx, 0],
      ArrowUp: [0, dy],
      ArrowDown: [0, -dy],
    };
    const m = moves[e.key];
    if (!m) return;
    own();
    moveNode(i, data.x[i] + m[0], data.y[i] + m[1], true);
  };

  // A tap or click on empty canvas adds a node there (appended last: the next node a Newton row
  // adds). It is decided on pointer-up, so a scroll gesture that starts on the plot (the browser
  // cancels the pointer) or any drag of more than a few pixels never edits the data.
  const tap = useRef<{ id: number; x: number; y: number; t: number } | null>(null);
  const onCanvasDown = (e: RPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || e.target !== e.currentTarget.querySelector('canvas')) return;
    tap.current = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp };
  };
  const onCanvasUp = (e: RPointerEvent<HTMLDivElement>) => {
    const t0 = tap.current;
    tap.current = null;
    if (!t0 || t0.id !== e.pointerId) return;
    const moved = Math.hypot(e.clientX - t0.x, e.clientY - t0.y);
    const quick = e.pointerType === 'mouse' || e.timeStamp - t0.t < 350;
    if (moved > 6 || !quick) return;
    if (data.x.length >= MAX_NODES) return;
    const p = toData(e.clientX, e.clientY);
    if (!p || p.x < a || p.x > b) return;
    const nx = round6(p.x);
    if (!distinct(data.x, nx, -1)) return;
    onNodesChange([...data.x, nx], [...data.y, yAt(nx, round6(p.y))], true);
  };
  const onCanvasCancel = () => {
    tap.current = null;
  };

  const { x: sx, y: sy } = scales(size.width, size.height);
  const free = !(snap && data.fTrue);
  const n = data.x.length;
  return (
    <div
      ref={wrap}
      className={styles.plot}
      role="group"
      aria-label={ariaLabel}
      onPointerDown={onCanvasDown}
      onPointerUp={onCanvasUp}
      onPointerCancel={onCanvasCancel}
      onPointerLeave={onCanvasCancel}
    >
      <canvas ref={canvasRef} className={styles.canvas} role="img" aria-label={ariaLabel} />
      {size.width > 0 &&
        data.x.map((xi, i) => {
          const top = Math.max(MARGIN.top, Math.min(size.height - MARGIN.bottom, sy(data.y[i])));
          const held = grabbedIndex === i;
          return (
            <button
              key={i}
              ref={(el) => {
                buttons.current[i] = el;
              }}
              type="button"
              className={styles.handle}
              tabIndex={i === activeIndex ? 0 : -1}
              data-own-keys
              data-dragging={dragging === i || held || undefined}
              data-axis={free ? undefined : 'x'}
              aria-pressed={held}
              style={{ left: sx(xi), top }}
              aria-label={`Node ${i} (${byX.indexOf(i) + 1} of ${n} from the left): x ${sig(xi, 4)}, y ${sig(data.y[i], 4)}. ${
                held
                  ? `Grabbed: ${free ? 'arrow keys move it' : 'left and right arrows move it along f'}; Enter drops it.`
                  : 'Left and right arrows go to the neighboring node; Enter grabs it to move it.'
              } Delete removes it.`}
              onFocus={() => setActive(i)}
              onBlur={() => held && setGrabbed(null)}
              onPointerDown={onDown(i)}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerCancel={onCancel}
              onDoubleClick={() => removeNode(i)}
              onKeyDown={onKey(i)}
            />
          );
        })}
    </div>
  );
}

/** The node the focused run's step added, and its label letter (x, or z for AAA's support points). */
function ringOf(
  run: RunView,
  data: ActiveData,
): { k: number; nx: number; ny: number; name: string } | null {
  if (run.kind === 'nodewise' || run.kind === 'neville') {
    const k = Math.min(run.k, data.x.length - 1);
    return { k, nx: data.x[k], ny: data.y[k], name: 'x' };
  }
  if (run.kind === 'aaa' || run.kind === 'blend') {
    const node = nums(run.result.trace[Math.min(run.k, run.result.trace.length - 1)]?.info.node);
    if (node.length !== 2) return null;
    return { k: run.k, nx: node[0], ny: node[1], name: run.kind === 'aaa' ? 'z' : 'x' };
  }
  return null;
}

/** A cross (×) with a halo: a pole on the axis. */
function cross(
  ctx: CanvasRenderingContext2D,
  colors: ChartColors,
  px: number,
  py: number,
  r: number,
  color: string,
) {
  ctx.save();
  ctx.lineCap = 'round';
  for (const [c, w] of [
    [colors.halo, 4.5],
    [color, 2],
  ] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(px - r, py - r);
    ctx.lineTo(px + r, py + r);
    ctx.moveTo(px + r, py - r);
    ctx.lineTo(px - r, py + r);
    ctx.stroke();
  }
  ctx.restore();
}

/** The data sorted by x (the piecewise methods' order), for stages that predate extra. */
function sortedNodes(data: ActiveData): { x: number[]; y: number[] } {
  const order = data.x.map((_, i) => i).sort((i, j) => data.x[i] - data.x[j] || i - j);
  return { x: order.map((i) => data.x[i]), y: order.map((i) => data.y[i]) };
}

/** Redraws only when its inputs change (not on every playback frame of the lab). */
export const InterpPlot = memo(InterpPlotImpl);
