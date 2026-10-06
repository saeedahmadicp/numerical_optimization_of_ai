/**
 * The regression lab's main view: the data plane.
 *
 *   - data points (draggable buttons; click empty space to add, double-click or Delete removes);
 *   - every method's fitted curve at the playhead (β moves straight from β_k to β_{k+1});
 *   - for the focused method: residual sticks rᵢ = yᵢ − ŷ(xᵢ), the residual squares of the
 *     least-squares objective (side |rᵢ|, so area ∝ rᵢ²), robust IRLS weights as point opacity,
 *     the Huber band |r| ≤ δσ̂, the minimax band ŷ ± h with its alternating reference points and
 *     the point that enters next, the points an L1 line passes through, the Theil–Sen
 *     median-slope pair, and the ridge λ path (one fit per decade of λ);
 *   - the noise-free regression function, when the dataset has one.
 */
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as RKeyboardEvent,
  type PointerEvent as RPointerEvent,
} from 'react';
import { useChartColors } from '../../ui/theme';
import { sig } from '../../core/format';
import { polyval } from '../../methods/regression/methods';
import {
  drawAxes,
  drawMath,
  linearScale,
  mathMain as mm,
  mathSub as sub,
  mathVar as mv,
  useCanvas,
  useElementSize,
  type MathRun,
  type Scale,
} from '../../viz';
import { snap, type Pt } from './model';
import styles from './RegressionLab.module.css';

export interface StageCurve {
  key: string;
  slot: number;
  beta: readonly number[];
  focused?: boolean;
  /** Direct label at the right end of the curve. */
  label?: string;
}

export interface StageGeometry {
  slot: number;
  /** β of the focused method at the playhead (residual sticks, squares). */
  beta: readonly number[] | null;
  squares?: boolean;
  /** Point opacity in [0, 1] (robust weights). */
  weights?: readonly number[] | null;
  /** A band |y − ŷ(x)| ≤ half around the focused fit (Huber: δσ̂, minimax: |h|). */
  band?: { half: number; kind: 'huber' | 'minimax' } | null;
  /** Minimax reference points (input order) with the sign of their residual. */
  reference?: { index: number; sign: number }[];
  entering?: number | null;
  /** LAD: points the line passes through (|rᵢ| ≤ ε). */
  rings?: readonly number[];
  /** Theil–Sen: the median-slope pair(s). */
  pairs?: readonly (readonly [number, number])[];
  /** Ridge: fits along the λ path. */
  fan?: readonly { lam: number; beta: readonly number[] }[];
}

export interface DataStageProps {
  points: readonly Pt[];
  /** The problem's plotting domain (x). */
  domain: readonly [number, number];
  curves: readonly StageCurve[];
  geometry: StageGeometry | null;
  truth?: ((x: number) => number) | null;
  editable?: boolean;
  /** Live edits while dragging (null when the drag ends). */
  onDraft?: (pts: Pt[] | null) => void;
  /** A finished edit (drag end, add, remove, keyboard move). */
  onCommit?: (pts: Pt[]) => void;
  minPoints?: number;
  maxPoints?: number;
  ariaLabel: string;
}

const M = { left: 46, right: 16, top: 18, bottom: 32 };

function pad([a, b]: [number, number], f: number): [number, number] {
  const s = b - a || Math.max(1, Math.abs(a));
  return [a - f * s, b + f * s];
}

export function DataStage({
  points,
  domain,
  curves,
  geometry,
  truth,
  editable = true,
  onDraft,
  onCommit,
  minPoints = 3,
  maxPoints = 60,
  ariaLabel,
}: DataStageProps) {
  const colors = useChartColors();
  const wrap = useRef<HTMLDivElement>(null);
  const size = useElementSize(wrap);
  const [frozen, setFrozen] = useState<{ x: [number, number]; y: [number, number] } | null>(null);

  // Axes: the problem's domain and the data, padded; the fits may widen y by at most 45 % of the
  // data span on each side (a wild high-degree fit leaves the frame instead of squashing the data).
  const autoX = useMemo<[number, number]>(() => {
    const xs = points.map((p) => p.x);
    const lo = Math.min(domain[0], ...xs),
      hi = Math.max(domain[1], ...xs);
    return pad([lo, hi], 0.035);
  }, [points, domain]);
  const autoY = useMemo<[number, number]>(() => {
    const ys = points.map((p) => p.y);
    const lo = Math.min(...ys),
      hi = Math.max(...ys);
    const span = hi - lo || Math.max(1, Math.abs(lo));
    let a = lo,
      b = hi;
    const fns: ((x: number) => number)[] = curves.map((c) => (x) => polyval(c.beta, x));
    if (truth) fns.push(truth);
    for (const f of fns)
      for (let i = 0; i <= 48; i++) {
        const v = f(autoX[0] + ((autoX[1] - autoX[0]) * i) / 48);
        if (!Number.isFinite(v)) continue;
        a = Math.min(a, Math.max(v, lo - 0.45 * span));
        b = Math.max(b, Math.min(v, hi + 0.45 * span));
      }
    return pad([a, b], 0.06);
  }, [points, curves, truth, autoX]);
  const xd = frozen?.x ?? autoX;
  const yd = frozen?.y ?? autoY;

  // Direct labels sit in a right margin outside the plot (wide stages only).
  const longest = Math.max(0, ...curves.map((c) => c.label?.length ?? 0));
  const mr = size.width >= 560 && longest > 0 ? 30 + 6.6 * longest : M.right;

  const scales = (w: number, h: number): { x: Scale; y: Scale } => ({
    x: linearScale(xd, [M.left, w - mr]),
    y: linearScale(yd, [h - M.bottom, M.top]),
  });

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - mr,
      bottom: s.height - M.bottom,
    };
    drawAxes(ctx, { x, y, frame, colors, dpr: s.dpr, xLabel: 'x', yLabel: 'y' });
    const P = (a: number, b: number): [number, number] => [x(a), y(b)];
    const series = (slot: number) => colors.series[slot % colors.series.length];
    const sample = (f: (t: number) => number) => {
      const pts: [number, number][] = [];
      const n = Math.max(80, Math.round((frame.right - frame.left) / 2));
      for (let i = 0; i <= n; i++) {
        const t = xd[0] + ((xd[1] - xd[0]) * i) / n;
        pts.push(P(t, f(t)));
      }
      return pts;
    };
    const tracePath = (pts: [number, number][]) => {
      ctx.beginPath();
      let pen = false;
      for (const [px, py] of pts) {
        if (!Number.isFinite(py) || Math.abs(py) > 1e5) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        pen = true;
      }
    };
    const strokeCurve = (
      pts: [number, number][],
      color: string,
      width: number,
      dash: number[] = [],
    ) => {
      ctx.setLineDash([]);
      ctx.strokeStyle = colors.halo;
      ctx.lineWidth = width + 2.5;
      ctx.lineJoin = 'round';
      ctx.globalAlpha = 0.75;
      tracePath(pts);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      tracePath(pts);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();

    const g = geometry;
    const gColor = g ? series(g.slot) : colors.text;
    const fit = g?.beta ?? null;

    // Ridge λ path: one faint fit per decade of λ.
    if (g?.fan?.length) {
      g.fan.forEach((f, i) => {
        ctx.globalAlpha = 0.18 + (0.22 * i) / Math.max(1, g.fan!.length - 1);
        ctx.strokeStyle = gColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        tracePath(sample((t) => polyval(f.beta, t)));
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    // Band around the focused fit: Huber's quadratic zone or the minimax leveled error.
    if (g?.band && fit && g.band.half > 0) {
      const hw = g.band.half;
      const up = sample((t) => polyval(fit, t) + hw);
      const lo = sample((t) => polyval(fit, t) - hw);
      ctx.beginPath();
      up.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      for (let i = lo.length - 1; i >= 0; i--) ctx.lineTo(lo[i][0], lo[i][1]);
      ctx.closePath();
      ctx.fillStyle = gColor;
      ctx.globalAlpha =
        g.band.kind === 'minimax'
          ? colors.mode === 'dark'
            ? 0.06
            : 0.045
          : colors.mode === 'dark'
            ? 0.11
            : 0.08;
      ctx.fill();
      ctx.globalAlpha = 0.7;
      ctx.strokeStyle = gColor;
      ctx.lineWidth = 1;
      ctx.setLineDash(g.band.kind === 'minimax' ? [5, 3] : [2, 3]);
      for (const edge of [up, lo]) {
        tracePath(edge);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // The noise-free regression function.
    if (truth) {
      const pts = sample(truth);
      ctx.globalAlpha = 0.85;
      strokeCurve(pts, colors.text3, 1.25, [1.5, 3.5]);
      ctx.globalAlpha = 1;
    }

    // Residual squares (area ∝ rᵢ²) and sticks of the focused fit.
    if (fit) {
      const sticks: [number, number, number][] = [];
      points.forEach((p) => {
        const fy = polyval(fit, p.x);
        if (Number.isFinite(fy)) sticks.push([p.x, p.y, fy]);
      });
      if (g?.squares) {
        for (const [px0, py0, fy] of sticks) {
          const [sx, sy] = P(px0, py0);
          const [, fyPx] = P(px0, fy);
          const side = Math.abs(fyPx - sy);
          if (side < 0.5) continue;
          const top = Math.min(sy, fyPx);
          const left = sx + side <= frame.right ? sx : sx - side;
          ctx.fillStyle = gColor;
          ctx.globalAlpha = colors.mode === 'dark' ? 0.1 : 0.07;
          ctx.fillRect(left, top, side, side);
          ctx.globalAlpha = 0.32;
          ctx.strokeStyle = gColor;
          ctx.lineWidth = 0.75;
          ctx.strokeRect(left + 0.375, top + 0.375, side - 0.75, side - 0.75);
          ctx.globalAlpha = 1;
        }
      }
      ctx.strokeStyle = gColor;
      ctx.lineWidth = 1.25;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      for (const [px0, py0, fy] of sticks) {
        ctx.moveTo(...P(px0, py0));
        ctx.lineTo(...P(px0, fy));
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Fitted curves: others first, the focused one on top.
    const ordered = [...curves].sort((a, b) => Number(!!a.focused) - Number(!!b.focused));
    for (const c of ordered)
      strokeCurve(
        sample((t) => polyval(c.beta, t)),
        series(c.slot),
        c.focused ? 2.4 : 1.75,
      );

    // Theil–Sen: the median-slope pair(s).
    if (g?.pairs?.length) {
      for (const [i, j] of g.pairs) {
        const a = points[i],
          b = points[j];
        if (!a || !b) continue;
        ctx.strokeStyle = gColor;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([1, 3]);
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(...P(a.x, a.y));
        ctx.lineTo(...P(b.x, b.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineCap = 'butt';
      }
    }

    // Data points: ink dots; opacity = robust weight.
    const ringed = new Set<number>([...(g?.rings ?? []), ...(g?.pairs ?? []).flat()]);
    points.forEach((p, i) => {
      const [px, py] = P(p.x, p.y);
      const w = g?.weights?.[i] ?? 1;
      ctx.fillStyle = colors.halo;
      ctx.beginPath();
      ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 3.75, 0, Math.PI * 2);
      ctx.globalAlpha = 0.16 + 0.84 * Math.max(0, Math.min(1, w));
      ctx.fillStyle = colors.text;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (w < 0.999) {
        // A hairline outline keeps a faded point findable.
        ctx.strokeStyle = colors.text3;
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.arc(px, py, 3.75, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (ringed.has(i)) {
        ctx.strokeStyle = gColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px, py, 7.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    // Minimax: the reference points with the sign of their leveled error, the entering point.
    if (g?.reference?.length) {
      for (const { index, sign } of g.reference) {
        const p = points[index];
        if (!p) continue;
        const [px, py] = P(p.x, p.y);
        ctx.strokeStyle = gColor;
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.arc(px, py, 7.5, 0, Math.PI * 2);
        ctx.stroke();
        const runs: MathRun[] = [mm(sign > 0 ? '+|' : '−|'), mv('h'), mm('|')];
        // Above (+) or below (−) the point; beside it when that would leave the frame.
        const ly = sign > 0 ? py - 13 : py + 22;
        const inside = ly > frame.top + 12 && ly < frame.bottom - 2;
        const left = px > frame.right - 40;
        drawMath(ctx, runs, inside ? px : left ? px - 12 : px + 12, inside ? ly : py + 4, {
          size: 12.5,
          align: inside ? 'center' : left ? 'right' : 'left',
          color: colors.text,
          halo: colors.halo,
        });
      }
    }
    if (g?.entering !== null && g?.entering !== undefined) {
      const p = points[g.entering];
      if (p) {
        const [px, py] = P(p.x, p.y);
        ctx.strokeStyle = gColor;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2.5, 2.5]);
        ctx.beginPath();
        ctx.arc(px, py, 10.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        const right = px < frame.right - 90;
        drawMath(
          ctx,
          [mv('x'), sub(String(g.entering), 'main'), mm(' enters')],
          right ? px + 15 : px - 15,
          py + 4,
          {
            size: 12,
            align: right ? 'left' : 'right',
            color: colors.text2,
            halo: colors.halo,
          },
        );
      }
    }
    ctx.restore();
    // Direct labels at the right end of each curve (pushed apart so they never collide).
    const labels = curves
      .filter((c) => c.label)
      .map((c) => {
        const xr = xd[1];
        const v = polyval(c.beta, xr);
        return { c, py: Number.isFinite(v) ? y(v) : NaN };
      })
      .filter((l) => Number.isFinite(l.py) && mr > M.right)
      .map((l) => ({ ...l, py: Math.max(frame.top + 4, Math.min(frame.bottom - 4, l.py)) }))
      .sort((a, b) => a.py - b.py);
    let lastY = -Infinity;
    ctx.font = `500 11.5px ${colors.fontSans}`;
    for (const l of labels) {
      let py = l.py + 4;
      if (py - lastY < 15) py = lastY + 15;
      lastY = py;
      const text = l.c.label!;
      const tx = frame.right + 22;
      ctx.fillStyle = colors.text2;
      ctx.textAlign = 'left';
      ctx.fillText(text, tx, py);
      ctx.strokeStyle = series(l.c.slot);
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(frame.right + 4, py - 4);
      ctx.lineTo(frame.right + 16, py - 4);
      ctx.stroke();
    }
  });

  // ── Editing ─────────────────────────────────────────────────────────────────────────
  const drag = useRef<{
    index: number;
    id: number;
    pts: Pt[];
    moved: boolean;
    added: boolean;
  } | null>(null);
  const toData = (clientX: number, clientY: number): Pt | null => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return null;
    const { x, y } = scales(r.width, r.height);
    const clamp = (v: number, [a, b]: [number, number]) => Math.min(b, Math.max(a, v));
    return {
      x: snap(clamp(x.invert(clientX - r.left), xd)),
      y: snap(clamp(y.invert(clientY - r.top), yd)),
    };
  };
  const inFrame = (clientX: number, clientY: number) => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return false;
    const px = clientX - r.left,
      py = clientY - r.top;
    return px >= M.left && px <= r.width - mr && py >= M.top && py <= r.height - M.bottom;
  };

  const onPointerDown = (e: RPointerEvent<HTMLDivElement>) => {
    if (!editable || e.button !== 0) return;
    const target = e.target as HTMLElement;
    const idx = target.dataset.index;
    if (idx !== undefined) {
      e.preventDefault();
      wrap.current?.setPointerCapture(e.pointerId);
      drag.current = {
        index: Number(idx),
        id: e.pointerId,
        pts: points.map((p) => ({ ...p })),
        moved: false,
        added: false,
      };
      setFrozen({ x: xd, y: yd });
      return;
    }
    // Empty space adds a point (touch: on release, so a scroll gesture never adds one).
    if (e.pointerType === 'touch' || !inFrame(e.clientX, e.clientY) || points.length >= maxPoints)
      return;
    const p = toData(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    wrap.current?.setPointerCapture(e.pointerId);
    const pts = [...points.map((q) => ({ ...q })), p];
    drag.current = { index: pts.length - 1, id: e.pointerId, pts, moved: false, added: true };
    setFrozen({ x: xd, y: yd });
    onDraft?.(pts);
  };
  const onPointerMove = (e: RPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const p = toData(e.clientX, e.clientY);
    if (!p) return;
    d.pts = d.pts.map((q, j) => (j === d.index ? p : q));
    d.moved = true;
    onDraft?.(d.pts);
  };
  const onPointerUp = (e: RPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (d && d.id === e.pointerId) {
      drag.current = null;
      setFrozen(null);
      onDraft?.(null);
      if (d.moved || d.added) onCommit?.(d.pts);
      return;
    }
    // A tap on empty space (touch) adds a point.
    if (
      editable &&
      e.pointerType === 'touch' &&
      !(e.target as HTMLElement).dataset.index &&
      inFrame(e.clientX, e.clientY) &&
      points.length < maxPoints
    ) {
      const p = toData(e.clientX, e.clientY);
      if (p) onCommit?.([...points, p]);
    }
  };
  // ── Keyboard: one tab stop for the points (roving tabindex) ───────────────────────────
  // Arrows move the selected point (Shift: 10×); Page Up/Down (or [ ]) select the neighbor
  // in x, Home/End the first/last; Delete removes it and the focus moves to its neighbor.
  // A run of key nudges is a drag: the fits follow at their final iterates, and the edit is
  // committed (and replayed) 600 ms after the last key.
  const [sel, setSel] = useState(0);
  const selected = Math.min(sel, points.length - 1);
  const handles = useRef(new Map<number, HTMLButtonElement>());
  const refocus = useRef<number | null>(null);
  const nudge = useRef<{ pts: Pt[]; timer: number } | null>(null);
  useEffect(() => {
    if (refocus.current === null) return;
    handles.current.get(refocus.current)?.focus();
    refocus.current = null;
  });
  const flushNudge = () => {
    const n = nudge.current;
    if (!n) return;
    window.clearTimeout(n.timer);
    nudge.current = null;
    setFrozen(null);
    onDraft?.(null);
    onCommit?.(n.pts);
  };
  const flushRef = useRef(flushNudge);
  useEffect(() => {
    flushRef.current = flushNudge;
  });
  useEffect(() => () => flushRef.current(), []);

  const remove = (i: number, keyboard = false) => {
    if (points.length <= minPoints) return;
    if (keyboard) {
      const to = Math.max(0, i - 1);
      setSel(to);
      refocus.current = to;
    }
    onCommit?.(points.filter((_, j) => j !== i));
  };
  const byX = () =>
    points
      .map((p, i) => ({ p, i }))
      .sort((a, b) => a.p.x - b.p.x || a.i - b.i)
      .map((o) => o.i);
  const choose = (i: number) => {
    flushRef.current();
    setSel(i);
    refocus.current = i;
  };
  const onKey = (i: number) => (e: RKeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      e.stopPropagation();
      if (nudge.current) {
        // Remove from the nudged positions.
        const pts = nudge.current.pts;
        window.clearTimeout(nudge.current.timer);
        nudge.current = null;
        setFrozen(null);
        onDraft?.(null);
        if (pts.length > minPoints) {
          setSel(Math.max(0, i - 1));
          refocus.current = Math.max(0, i - 1);
          onCommit?.(pts.filter((_, j) => j !== i));
        } else onCommit?.(pts);
        return;
      }
      remove(i, true);
      return;
    }
    const order = byX();
    const at = order.indexOf(i);
    const jump: Record<string, number> = {
      PageDown: order[Math.min(order.length - 1, at + 1)],
      ']': order[Math.min(order.length - 1, at + 1)],
      PageUp: order[Math.max(0, at - 1)],
      '[': order[Math.max(0, at - 1)],
      Home: order[0],
      End: order[order.length - 1],
    };
    if (e.key in jump) {
      e.preventDefault();
      e.stopPropagation();
      choose(jump[e.key]);
      return;
    }
    const f = 0.01 * (e.shiftKey ? 10 : 1);
    const dx = (xd[1] - xd[0]) * f,
      dy = (yd[1] - yd[0]) * f;
    const cur = nudge.current?.pts ?? points.map((q) => ({ ...q }));
    const p = cur[i];
    if (!p) return;
    const next: Record<string, Pt> = {
      ArrowLeft: { x: p.x - dx, y: p.y },
      ArrowRight: { x: p.x + dx, y: p.y },
      ArrowUp: { x: p.x, y: p.y + dy },
      ArrowDown: { x: p.x, y: p.y - dy },
    };
    const n = next[e.key];
    if (!n) return;
    e.preventDefault();
    e.stopPropagation();
    const pts = cur.map((q, j) => (j === i ? { x: snap(n.x), y: snap(n.y) } : q));
    if (nudge.current) window.clearTimeout(nudge.current.timer);
    else setFrozen({ x: xd, y: yd });
    nudge.current = { pts, timer: window.setTimeout(() => flushRef.current(), 600) };
    onDraft?.(pts);
  };

  const { x: sx, y: sy } = scales(size.width, size.height);
  const canRemove = points.length > minPoints;
  return (
    <div
      ref={wrap}
      className={styles.stagePlot}
      role="group"
      aria-label={ariaLabel}
      data-editable={editable || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        role="img"
        aria-label={`${ariaLabel} (${points.length} points)`}
      />
      {editable &&
        size.width > 0 &&
        points.map((p, i) => (
          <button
            key={i}
            ref={(el) => {
              if (el) handles.current.set(i, el);
              else handles.current.delete(i);
            }}
            type="button"
            data-index={i}
            data-plot-focus={i === selected ? '' : undefined}
            tabIndex={i === selected ? 0 : -1}
            className={styles.handle}
            data-own-keys
            style={{ left: sx(p.x), top: sy(p.y) }}
            aria-label={`Point ${i} of ${points.length}: x ${sig(p.x, 4)}, y ${sig(p.y, 4)}. Arrow keys move it; Page Up and Page Down select the neighboring point${canRemove ? '; Delete removes it' : ''}.`}
            onFocus={() => setSel(i)}
            onBlur={() => flushRef.current()}
            onDoubleClick={() => remove(i)}
            onKeyDown={onKey(i)}
          />
        ))}
    </div>
  );
}
