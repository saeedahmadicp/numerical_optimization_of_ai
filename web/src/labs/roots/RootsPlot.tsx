/**
 * The roots lab's main view: the graph of f with the geometry of every compared method's
 * current step (bracket and discarded part, chords, tangents, hyperbolas, parabolas, inverse
 * parabolas, Ridders' transformed points, ITP's projection ball, Steffensen's probe), the
 * iterates on the curve and as a rug on the axis, and draggable bracket ends a, b and start x₀.
 *
 * The camera follows the focused method by default: each step is framed (the bracket, or the
 * points the step was built from) and the view zooms geometrically between steps, so a run that
 * gains a digit per step reads as a steady dive toward the root. Zooming or panning by hand
 * stops following; the follow toggle (crosshair) of the view toolbar resumes it.
 *
 * Cobweb mode draws g(x) = x − λ f(x) against y = x with the fixed-point staircase.
 *
 * Built on the shared primitives (useCanvas, drawAxes, adaptiveSample, drawMath); the handles
 * are real sliders (focusable, arrow keys, Shift ×10) positioned over the canvas.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent as RPointerEvent,
} from 'react';
import type { Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import type { ChartColors } from '../../ui/colors';
import { sig } from '../../core/format';
import {
  adaptiveSample,
  crisp,
  drawAxes,
  drawMath,
  linearScale,
  mathMain,
  mathSub,
  mathVar,
  niceDomain,
  useCanvas,
  ViewToolbar,
  useElementSize,
  type MathRun,
  type Scale,
} from '../../viz';
import { cameraAt, dragWindow, headAt, stepGeometry, type Prim, type Pt } from './geometry';
import { ease, HOLD, phase } from './estimate';
import styles from './RootsLab.module.css';

export interface PlotRun {
  methodId: string;
  name: string;
  slot: number;
  trace: readonly Step[];
  /** The run's local playhead (continuous step index). */
  t: number;
  focused: boolean;
  converged: boolean;
  /** Root this run is measured against (x⋆), if any. */
  target: number | null;
}

export interface RootsPlotProps {
  /** f for drawing (fast). */
  f: (x: number) => number;
  /** The problem's plotting window. */
  domain: [number, number];
  /** The view resets when this changes (the problem id). */
  viewKey: string;
  roots: readonly number[];
  runs: readonly PlotRun[];
  /** The bracket of the bracketing methods (null: none selected, no handles). */
  bracket: [number, number] | null;
  x0: number | null;
  onBracket: (b: [number, number], commit: boolean) => void;
  onX0: (x: number, commit: boolean) => void;
  mode: 'f' | 'cobweb';
  /** λ of the fixed-point run (cobweb mode). */
  lam?: number;
  ariaLabel: string;
}

const M = { left: 64, right: 16, top: 26 };
const ROW = 26; // px from the frame bottom to the first handle row (under the tick labels)
const ROW_GAP = 24;

/** The camera moves with the step's transition, not during the hold on step k. */
function cameraT(t: number): number {
  if (!Number.isFinite(t)) return t; // a drag shows every step (t = ∞)
  const k = Math.floor(t);
  const u = t - k;
  return k + (u <= HOLD ? 0 : (u - HOLD) / (1 - HOLD));
}

/** Round a dragged coordinate to about 1/400 of the view width, without float noise. */
function snap(x: number, width: number): number {
  const q = 10 ** Math.floor(Math.log10(width / 400));
  return Number((Math.round(x / q) * q).toPrecision(12));
}

function robustY(
  f: (x: number) => number,
  v: readonly [number, number],
  cobweb: boolean,
): [number, number] {
  if (cobweb) return [v[0], v[1]];
  const ys: number[] = [];
  for (let i = 0; i <= 240; i++) {
    const y = f(v[0] + ((v[1] - v[0]) * i) / 240);
    if (Number.isFinite(y)) ys.push(y);
  }
  if (!ys.length) return [-1, 1];
  ys.sort((a, b) => a - b);
  const lo = Math.min(0, ys[Math.floor(ys.length * 0.02)]);
  const hi = Math.max(0, ys[Math.floor(ys.length * 0.98)]);
  const pad = (hi - lo) * 0.1 || Math.max(Math.abs(lo), 1e-300);
  // niceDomain rounds to ticks; at deep zoom the values are tiny but the rounding is relative.
  return niceDomain(lo - pad, hi + pad);
}

export function RootsPlot({
  f,
  domain,
  viewKey,
  roots,
  runs,
  bracket,
  x0,
  onBracket,
  onX0,
  mode,
  lam = 0.1,
  ariaLabel,
}: RootsPlotProps) {
  const colors = useChartColors();
  const wrap = useRef<HTMLDivElement>(null);
  const size = useElementSize(wrap);

  // ── Camera: follow the focused run, or a view set by hand (zoom, pan) ─────────────────
  const [manual, setManual] = useState<[number, number] | null>(null);
  // While a handle is dragged the camera holds still (the runs re-run under it).
  const [frozen, setFrozen] = useState<[number, number] | null>(null);
  const [prevKey, setPrevKey] = useState(viewKey);
  if (prevKey !== viewKey) {
    setPrevKey(viewKey);
    setManual(null);
    setFrozen(null);
  }
  const lead = runs.find((r) => r.focused) ?? runs[0];
  const follow = manual === null;
  const followView: [number, number] =
    lead && mode === 'f' ? cameraAt(lead.trace, cameraT(lead.t), domain) : [domain[0], domain[1]];
  const view: [number, number] = frozen ?? manual ?? followView;
  const takeOver = () => setManual(view);
  /** The view shows the whole problem (Reset has nothing to do). */
  const wholeView = !follow && view[0] === domain[0] && view[1] === domain[1];

  const g = useMemo(() => (mode === 'cobweb' ? (x: number) => x - lam * f(x) : f), [mode, lam, f]);
  const yDom = robustY(g, view, mode === 'cobweb');
  const twoRows = bracket !== null && x0 !== null && mode !== 'cobweb';
  const hasHandles = mode !== 'cobweb' && (bracket !== null || x0 !== null);
  const bottom = ROW + (twoRows ? ROW_GAP : 0) + 18;
  const frame = {
    left: M.left,
    top: M.top,
    right: Math.max(M.left + 10, size.width - M.right),
    bottom: Math.max(M.top + 10, size.height - bottom),
  };
  const x = linearScale(view, [frame.left, frame.right]);
  const y = linearScale(yDom, [frame.bottom, frame.top]);
  const rowOf = (kind: 'a' | 'b' | 'x0') =>
    frame.bottom + ROW + (twoRows && kind === 'x0' ? ROW_GAP : 0);

  // ── Drawing ─────────────────────────────────────────────────────────────────────────
  const { canvasRef } = useCanvas((ctx, s) => {
    if (s.width < 40 || s.height < 40) return;
    drawAxes(ctx, {
      x,
      y,
      frame,
      colors,
      dpr: s.dpr,
      // With handle rows under the ticks the name moves to the end of y = 0 (drawn below).
      xName: hasHandles ? undefined : [mathVar('x')],
      yName:
        mode === 'cobweb'
          ? [mathVar('g'), mathMain('('), mathVar('x'), mathMain(')')]
          : [mathVar('f'), mathMain('('), mathVar('x'), mathMain(')')],
    });
    const P = (a: number, b: number): [number, number] => [x(a), y(b)];
    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();

    // y = 0 (or y = x in cobweb mode).
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (mode === 'cobweb') {
      ctx.setLineDash([4, 4]);
      ctx.moveTo(...P(view[0], view[0]));
      ctx.lineTo(...P(view[1], view[1]));
    } else {
      ctx.moveTo(frame.left, crisp(y(0), s.dpr));
      ctx.lineTo(frame.right, crisp(y(0), s.dpr));
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (mode === 'cobweb') drawCobweb(ctx, { runs, P, colors });
    else drawSteps(ctx, { runs, f, P, colors, x, y, frame, dpr: s.dpr });

    // The curve of f (or g) above the shading, below the marks.
    strokeFn(ctx, g, view[0], view[1], P, colors.text, 1.75, 0.9);

    if (mode !== 'cobweb') {
      // Roots: + crosses on the axis; x⋆ named for the focused run (left of the cross).
      const shown = roots.filter((r) => r >= view[0] && r <= view[1]);
      const named = lead ? shown.find((r) => r === lead.target) : undefined;
      const star: [number, number] | null = named === undefined ? null : [x(named) - 6, y(0) - 11];
      drawMarks(ctx, { runs, P, colors, x, y, frame, star });
      for (const r of shown) cross(ctx, P(r, 0), colors);
      if (star)
        drawMath(ctx, [mathVar('x'), { t: '⋆', style: 'main', script: 'sup' }], star[0], star[1], {
          size: 13,
          align: 'right',
          color: colors.text2,
          halo: colors.halo,
        });
    }
    if (hasHandles)
      drawMath(ctx, [mathVar('x')], frame.right - 4, Math.min(frame.bottom - 8, y(0) + 15), {
        size: 13,
        align: 'right',
        color: colors.text2,
        halo: colors.halo,
      });
    ctx.restore();

    // Handle guides (outside the clip: they run down to the handle pills).
    const guide = (gx: number, kind: 'a' | 'b' | 'x0') => {
      if (gx < view[0] || gx > view[1]) return;
      ctx.save();
      ctx.strokeStyle = colors.text3;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.setLineDash(kind === 'x0' ? [1, 3] : [2, 3]);
      ctx.beginPath();
      ctx.moveTo(crisp(x(gx), s.dpr), frame.top);
      ctx.lineTo(crisp(x(gx), s.dpr), rowOf(kind) - 12);
      ctx.stroke();
      ctx.restore();
    };
    if (bracket && mode !== 'cobweb') {
      guide(bracket[0], 'a');
      guide(bracket[1], 'b');
    }
    if (x0 !== null) guide(x0, 'x0');
  });

  // ── Interaction ─────────────────────────────────────────────────────────────────────
  const width = view[1] - view[0];
  const toData = useCallback(
    (clientX: number) => {
      const r = wrap.current?.getBoundingClientRect();
      if (!r) return NaN;
      return view[0] + ((clientX - r.left - frame.left) / (frame.right - frame.left)) * width;
    },
    [view, width, frame.left, frame.right],
  );
  const drag = useRef<{
    kind: 'a' | 'b' | 'x0' | 'pan';
    startX: number;
    startView: [number, number];
    moved: boolean;
    /** Handle drags: the handle's value at the press, and data units per pixel. */
    startValue?: number;
    perPx?: number;
  } | null>(null);
  const [dragging, setDragging] = useState<null | 'a' | 'b' | 'x0' | 'pan'>(null);

  const move = (kind: 'a' | 'b' | 'x0', v: number, commit: boolean) => {
    const sv = snap(v, width);
    if (kind === 'x0') onX0(sv, commit);
    else if (bracket) {
      const gap = width * 0.004;
      if (kind === 'a') onBracket([Math.min(sv, bracket[1] - gap), bracket[1]], commit);
      else onBracket([bracket[0], Math.max(sv, bracket[0] + gap)], commit);
    }
  };

  // A handle drags relative to where it was pressed, in a window around the problem (see
  // dragWindow), so a press never makes the value jump and the scale is always usable.
  const dragValue = (d: NonNullable<typeof drag.current>, clientX: number) =>
    (d.startValue as number) + (clientX - d.startX) * (d.perPx as number);
  const onHandleDown =
    (kind: 'a' | 'b' | 'x0', value: number) => (e: RPointerEvent<HTMLElement>) => {
      e.preventDefault();
      e.stopPropagation();
      e.currentTarget.setPointerCapture(e.pointerId);
      const w = dragWindow(view, domain, value);
      drag.current = {
        kind,
        startX: e.clientX,
        startView: w,
        moved: false,
        startValue: value,
        perPx: (w[1] - w[0]) / Math.max(1, frame.right - frame.left),
      };
      setFrozen(w);
      setDragging(kind);
    };
  const onHandleMove = (e: RPointerEvent<HTMLElement>) => {
    const d = drag.current;
    if (!d || d.kind === 'pan') return;
    if (!d.moved && Math.abs(e.clientX - d.startX) < 2) return;
    d.moved = true;
    move(d.kind, dragValue(d, e.clientX), false);
  };
  const onHandleUp = (e: RPointerEvent<HTMLElement>) => {
    const d = drag.current;
    drag.current = null;
    setDragging(null);
    setFrozen(null);
    if (!d || d.kind === 'pan' || !d.moved) return;
    move(d.kind, dragValue(d, e.clientX), true);
  };

  // Background: drag pans the x-range, a click sets x₀ (open methods), double-click resets.
  const onBgDown = (e: RPointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'touch') return; // one finger scrolls the page
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { kind: 'pan', startX: e.clientX, startView: view, moved: false };
  };
  const onBgMove = (e: RPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d || d.kind !== 'pan') return;
    const dx = e.clientX - d.startX;
    if (!d.moved && Math.abs(dx) > 3) {
      d.moved = true;
      setDragging('pan');
    }
    if (!d.moved) return;
    const w = d.startView[1] - d.startView[0];
    const shift = (-dx / (frame.right - frame.left)) * w;
    setManual([d.startView[0] + shift, d.startView[1] + shift]);
  };
  const onBgUp = (e: RPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    setDragging(null);
    if (d?.kind === 'pan' && d.moved) return;
    if (x0 === null) return;
    const v = toData(e.clientX);
    if (v >= view[0] && v <= view[1]) onX0(snap(v, width), true);
  };

  // Wheel zoom around the pointer (a non-passive listener, so the page does not scroll).
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
  });
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.target instanceof HTMLCanvasElement)) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const fx = (e.clientX - r.left - M.left) / Math.max(1, r.width - M.left - M.right);
      const k = Math.exp(Math.max(-0.5, Math.min(0.5, e.deltaY * 0.0015)));
      const [a, b] = viewRef.current;
      const c = a + fx * (b - a);
      setManual([c - (c - a) * k, c + (b - c) * k]);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const zoom = (k: number) => {
    const c = (view[0] + view[1]) / 2;
    setManual([c - (width / 2) * k, c + (width / 2) * k]);
  };

  const handleKey = (kind: 'a' | 'b' | 'x0', value: number) => (e: KeyboardEvent<HTMLElement>) => {
    const d =
      e.key === 'ArrowRight' || e.key === 'ArrowUp'
        ? 1
        : e.key === 'ArrowLeft' || e.key === 'ArrowDown'
          ? -1
          : e.key === 'PageUp'
            ? 10
            : e.key === 'PageDown'
              ? -10
              : 0;
    if (!d) return;
    e.preventDefault();
    e.stopPropagation();
    // Steps of 1/200 of the problem window (Shift: ×10), whatever the zoom.
    const unit = (domain[1] - domain[0]) / 200;
    const next = Number((value + d * unit * (e.shiftKey ? 10 : 1)).toPrecision(12));
    if (kind === 'x0') onX0(next, true);
    else if (bracket) {
      const gap = unit / 4;
      if (kind === 'a') onBracket([Math.min(next, bracket[1] - gap), bracket[1]], true);
      else onBracket([bracket[0], Math.max(next, bracket[0] + gap)], true);
    }
  };

  const handles: { kind: 'a' | 'b' | 'x0'; value: number; label: string; text: string }[] = [];
  if (bracket && mode !== 'cobweb') {
    const fa = f(bracket[0]),
      fb = f(bracket[1]);
    handles.push(
      {
        kind: 'a',
        value: bracket[0],
        label: 'Bracket end a',
        text: `a = ${sig(bracket[0], 5)}, f(a) = ${sig(fa, 3)}`,
      },
      {
        kind: 'b',
        value: bracket[1],
        label: 'Bracket end b',
        text: `b = ${sig(bracket[1], 5)}, f(b) = ${sig(fb, 3)}`,
      },
    );
  }
  if (x0 !== null)
    handles.push({ kind: 'x0', value: x0, label: 'Start point x₀', text: `x₀ = ${sig(x0, 5)}` });
  const signBad = bracket !== null && Math.sign(f(bracket[0])) * Math.sign(f(bracket[1])) > 0;

  return (
    <div
      ref={wrap}
      className={styles.plot}
      role="group"
      aria-label="Graph of f with the bracket ends and the start point as sliders"
      data-dragging={dragging ?? undefined}
    >
      <canvas
        ref={canvasRef}
        className={styles.plotCanvas}
        role="img"
        aria-label={ariaLabel}
        onPointerDown={onBgDown}
        onPointerMove={onBgMove}
        onPointerUp={onBgUp}
        onDoubleClick={() => setManual(null)}
        data-pickable={x0 !== null || undefined}
      />
      {size.width > 0 &&
        handles.map((h) => {
          const px = x(h.value);
          const off = px < frame.left - 1 ? 'left' : px > frame.right + 1 ? 'right' : null;
          const left = Math.max(frame.left, Math.min(size.width - 26, frame.right, px));
          return (
            <div
              key={h.kind}
              role="slider"
              tabIndex={0}
              aria-label={h.label}
              aria-valuenow={h.value}
              aria-valuetext={h.text}
              aria-valuemin={Math.min(domain[0], h.value)}
              aria-valuemax={Math.max(domain[1], h.value)}
              title={`${h.text} — drag, or focus and use the arrow keys`}
              className={styles.handle}
              data-kind={h.kind}
              data-bad={(h.kind !== 'x0' && signBad) || undefined}
              data-off={off ?? undefined}
              style={{ left, top: rowOf(h.kind) }}
              onPointerDown={onHandleDown(h.kind, h.value)}
              onPointerMove={onHandleMove}
              onPointerUp={onHandleUp}
              onPointerCancel={onHandleUp}
              onKeyDown={handleKey(h.kind, h.value)}
            >
              {off === 'left' && <span aria-hidden="true">‹&nbsp;</span>}
              {h.kind === 'x0' ? (
                <>
                  <i>x</i>
                  <sub>0</sub>
                </>
              ) : (
                <i>{h.kind}</i>
              )}
              {off === 'right' && <span aria-hidden="true">&nbsp;›</span>}
            </div>
          );
        })}
      <ViewToolbar
        onZoomIn={() => zoom(0.5)}
        onZoomOut={() => zoom(2)}
        onReset={() => setManual([domain[0], domain[1]])}
        canReset={!wholeView}
        resetLabel="Show the whole problem"
        follow={{ on: follow, onToggle: () => (follow ? takeOver() : setManual(null)) }}
      />
    </div>
  );
}

// ── Canvas helpers ─────────────────────────────────────────────────────────────────────

type ToPx = (a: number, b: number) => [number, number];
type Frame = { left: number; top: number; right: number; bottom: number };

function strokeFn(
  ctx: CanvasRenderingContext2D,
  g: (x: number) => number,
  from: number,
  to: number,
  P: ToPx,
  color: string,
  width: number,
  alpha: number,
) {
  const pts = adaptiveSample(g, from, to, P);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha *= alpha;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let pen = false;
  let lastPy = 0;
  for (const [a, b] of pts) {
    const [px, py] = P(a, b);
    // Lift the pen at non-finite values and at poles (a jump across most of the canvas).
    if (!Number.isFinite(py) || Math.abs(py) > 2e4 || (pen && Math.abs(py - lastPy) > 4000)) {
      pen = false;
      continue;
    }
    if (pen) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    pen = true;
    lastPy = py;
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
  ctx.arc(px, py, r + 1.75, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();
}

function ring(
  ctx: CanvasRenderingContext2D,
  [px, py]: [number, number],
  color: string,
  halo: string,
  r: number,
) {
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(px, py, r + 1.75, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.stroke();
}

function cross(ctx: CanvasRenderingContext2D, [px, py]: [number, number], colors: ChartColors) {
  for (const [w, c] of [
    [3.5, colors.halo],
    [1.5, colors.text],
  ] as const) {
    ctx.strokeStyle = c;
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(px - 5.5, py);
    ctx.lineTo(px + 5.5, py);
    ctx.moveTo(px, py - 5.5);
    ctx.lineTo(px, py + 5.5);
    ctx.stroke();
  }
}

const iterRuns = (k: number | string): MathRun[] => [mathVar('x'), mathSub(String(k))];

function labelRuns(code: string, k: number): MathRun[] | null {
  if (code === 'm') return [mathVar('m')];
  if (code === 'tol') return [mathVar('x'), mathSub(`${k - 1}`), mathMain(' ± tol')];
  if (code === 'x_half') return [mathVar('x'), mathSub('1/2')];
  if (code === 'x_f') return [mathVar('x'), mathSub('f')];
  if (code === 'r') return [mathMain('± '), mathVar('r'), mathSub(String(k))];
  if (code === 'z')
    return [
      mathVar('z'),
      mathMain(' = '),
      mathVar('x'),
      mathMain(' + '),
      mathVar('f'),
      mathMain('('),
      mathVar('x'),
      mathMain(')'),
    ];
  if (code === 'flen')
    return [mathVar('f'), mathMain('('), mathVar('x'), mathSub(`${k - 1}`), mathMain(')')];
  if (code === 'aux') return [mathVar('x'), mathSub('0'), mathMain(' ± '), mathVar('h')];
  // The retained end's stored ordinate: m·f, the true value of f at that end scaled by m.
  if (code.startsWith('scale:')) return [mathMain(code.slice(6)), mathMain(' · '), mathVar('f')];
  return null;
}

interface DrawCtx {
  runs: readonly PlotRun[];
  f: (x: number) => number;
  P: ToPx;
  colors: ChartColors;
  x: Scale;
  y: Scale;
  frame: Frame;
  dpr: number;
}

/** Bracket shading and per-step geometry, cross-faded on the playhead. */
function drawSteps(ctx: CanvasRenderingContext2D, d: DrawCtx) {
  const { runs, f, colors, x, frame } = d;
  const order = [...runs].sort((a, b) => Number(a.focused) - Number(b.focused));
  let lane = 0;
  for (const run of order) {
    const n = run.trace.length;
    if (!n) continue;
    const t = Math.min(run.t, n - 1);
    const k = Math.floor(t);
    const u = t - k;
    const color = colors.series[run.slot % colors.series.length];
    const layers: [Step, number][] =
      k < n - 1 && u > 0
        ? [
            [run.trace[k], 1 - phase(u).fade],
            [run.trace[k + 1], phase(u).fade],
          ]
        : [[run.trace[k], 1]];
    const isBracketing = run.trace[0].info.bracket !== undefined;
    for (const [st, alpha] of layers) {
      if (alpha <= 0.01) continue;
      const geo = stepGeometry(run.methodId, st, f);
      ctx.save();
      const weight = run.focused || runs.length === 1 ? 1 : 0.38;
      ctx.globalAlpha = alpha * weight;
      if (geo.bracket) {
        const [a, b] = geo.bracket;
        const xa = x(a),
          xb = x(b);
        if (run.focused || runs.length === 1) {
          // The bracket, and hatching over the part the sign test discards.
          ctx.fillStyle = color;
          ctx.globalAlpha = alpha * 0.08;
          ctx.fillRect(xa, frame.top, xb - xa, frame.bottom - frame.top);
          if (geo.kept) {
            ctx.globalAlpha = alpha * 0.32;
            for (const [lo, hi] of [
              [a, geo.kept[0]],
              [geo.kept[1], b],
            ]) {
              if (hi - lo <= 0) continue;
              hatch(ctx, x(lo), frame.top, x(hi) - x(lo), frame.bottom - frame.top, color);
            }
          }
          ctx.globalAlpha = alpha;
        }
        // The bracket as a bar on the axis, one lane per method.
        ctx.fillStyle = color;
        const by = frame.bottom - 4 - lane * 5;
        ctx.fillRect(xa, by, Math.max(1.5, xb - xa), 3);
        if (geo.kept && geo.kept[1] > geo.kept[0]) {
          ctx.globalAlpha *= 0.45;
          ctx.fillRect(xa, by, Math.max(1.5, xb - xa), 3);
        }
        ctx.globalAlpha = alpha * weight;
      }
      for (const p of geo.prims) drawPrim(ctx, p, d, color, st.k);
      ctx.restore();
    }
    if (isBracketing) lane++;
  }
}

function hatch(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  w: number,
  h: number,
  color: string,
) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = -h; s < w + h; s += 7) {
    ctx.moveTo(x0 + s, y0 + h);
    ctx.lineTo(x0 + s + h, y0);
  }
  ctx.stroke();
  ctx.restore();
}

function drawPrim(ctx: CanvasRenderingContext2D, p: Prim, d: DrawCtx, color: string, k: number) {
  const { P, colors, y, frame } = d;
  const base = ctx.globalAlpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([]);
  const faint = 'faint' in p && p.faint;
  if (faint) ctx.globalAlpha = base * 0.5;
  const label = (
    code: string | undefined,
    px: number,
    py: number,
    align: 'left' | 'center' | 'right' = 'left',
  ) => {
    if (!code) return;
    const runs = labelRuns(code, k);
    if (runs)
      drawMath(ctx, runs, px, py, {
        size: 12,
        align,
        color: colors.text2,
        halo: colors.halo,
        baseline: 'middle',
      });
  };
  switch (p.t) {
    case 'seg': {
      let [ax, ay] = p.a;
      let [bx, by] = p.b;
      if (p.ext) {
        const dx = bx - ax,
          dy = by - ay;
        ax -= dx * p.ext;
        ay -= dy * p.ext;
        bx += dx * p.ext;
        by += dy * p.ext;
      }
      if (p.dash) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(...P(ax, ay));
      ctx.lineTo(...P(bx, by));
      ctx.stroke();
      break;
    }
    case 'curve':
      if (p.dash) ctx.setLineDash([5, 4]);
      strokeFn(ctx, p.g, p.from, p.to, P, color, 1.5, 1);
      break;
    case 'path': {
      if (p.dash) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      let pen = false;
      for (const [a, b] of p.pts) {
        const [px, py] = P(a, b);
        if (!Number.isFinite(px) || !Number.isFinite(py) || Math.abs(px) > 2e4) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        pen = true;
      }
      ctx.stroke();
      break;
    }
    case 'pt': {
      const q = P(p.p[0], p.p[1]);
      if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) break;
      if (p.shape === 'dot') dot(ctx, q, color, colors.halo, 3);
      else if (p.shape === 'ring') ring(ctx, q, color, colors.halo, 3.6);
      else {
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(q[0] - 4, q[1] - 4);
        ctx.lineTo(q[0] + 4, q[1] + 4);
        ctx.moveTo(q[0] + 4, q[1] - 4);
        ctx.lineTo(q[0] - 4, q[1] + 4);
        ctx.stroke();
      }
      label(p.label, q[0] + 8, q[1] - 9);
      break;
    }
    case 'drop': {
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...P(p.x, 0));
      ctx.lineTo(...P(p.x, p.y));
      ctx.stroke();
      break;
    }
    case 'tick': {
      const [px] = P(p.x, 0);
      const py = y(0);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, py - 6);
      ctx.lineTo(px, py + 6);
      ctx.stroke();
      label(p.label, px, Math.min(frame.bottom - 12, py + 16), 'center');
      break;
    }
    case 'span': {
      const [pa] = P(p.a, 0),
        [pb] = P(p.b, 0);
      const py = y(p.y) - 14;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(pa, py);
      ctx.lineTo(pb, py);
      ctx.moveTo(pa, py - 4);
      ctx.lineTo(pa, py + 4);
      ctx.moveTo(pb, py - 4);
      ctx.lineTo(pb, py + 4);
      ctx.stroke();
      label(p.label, (pa + pb) / 2, py - 10, 'center');
      break;
    }
    case 'link': {
      ctx.setLineDash([1.5, 3]);
      ctx.lineWidth = 1.25;
      const a = P(p.a[0], p.a[1]),
        b = P(p.b[0], p.b[1]);
      ctx.beginPath();
      ctx.moveTo(...a);
      ctx.lineTo(...b);
      ctx.stroke();
      ctx.setLineDash([]);
      label(p.label, b[0] + 8, b[1], 'left');
      break;
    }
  }
  ctx.globalAlpha = base;
  ctx.setLineDash([]);
}

/** Iterates: rug ticks on the axis, dots on the curve, the moving head, off-view chevrons. */
function drawMarks(
  ctx: CanvasRenderingContext2D,
  d: Pick<DrawCtx, 'runs' | 'P' | 'colors' | 'x' | 'y' | 'frame'> & {
    /** Anchor (right end, px) of the x⋆ label, which the head label must not cover. */
    star: [number, number] | null;
  },
) {
  const { runs, P, colors, x, y, frame, star } = d;
  const order = [...runs].sort((a, b) => Number(a.focused) - Number(b.focused));
  let offLabel = 0;
  for (const run of order) {
    const n = run.trace.length;
    if (!n) continue;
    const color = colors.series[run.slot % colors.series.length];
    const t = Math.min(run.t, n - 1);
    const k = Math.floor(t);
    const u = t - k;
    const strong = run.focused || runs.length === 1;
    ctx.save();
    // Rug on the axis and fading dots on the curve for x_0 … x_k.
    for (let j = 0; j <= k; j++) {
      const st = run.trace[j];
      const xj = st.x as number;
      const fj = st.fun;
      if (!Number.isFinite(xj)) continue;
      const age = k - j;
      ctx.globalAlpha = (strong ? 1 : 0.6) * Math.max(0.25, 1 - age * 0.12);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      const px = x(xj);
      const py = y(0);
      ctx.beginPath();
      ctx.moveTo(px, py - 4);
      ctx.lineTo(px, py + 4);
      ctx.stroke();
      if (j < k && fj !== null && Number.isFinite(fj)) dot(ctx, P(xj, fj), color, colors.halo, 2.2);
    }
    // The head: eased along the model from x_k to x_{k+1}, then onto the curve.
    let head: Pt;
    const cur = run.trace[k];
    const travel = k < n - 1 ? phase(u).travel : 0;
    if (travel > 0) head = headAt(run.methodId, cur, run.trace[k + 1], travel);
    else head = [cur.x as number, cur.fun ?? 0];
    const [hx, hy] = P(head[0], head[1]);
    ctx.globalAlpha = 1;
    const last = k === n - 1;
    const inside = hx >= frame.left - 1 && hx <= frame.right + 1;
    if (inside && Number.isFinite(hy)) {
      if (last && !run.converged)
        ring(ctx, [hx, Math.max(frame.top, Math.min(frame.bottom, hy))], color, colors.halo, 4.5);
      else
        dot(
          ctx,
          [hx, Math.max(frame.top, Math.min(frame.bottom, hy))],
          color,
          colors.halo,
          strong ? 4.75 : 4,
        );
      // Name the head: x_k at rest, x_{k+1} once it is on its way there. Next to x⋆ (the
      // label sits up-left of the root's cross) the name moves under the axis.
      if (strong) {
        let ly = Math.max(frame.top + 10, Math.min(frame.bottom - 10, hy)) - 11;
        if (star && Math.abs(hx + 9 - star[0]) < 48 && Math.abs(ly - star[1]) < 16)
          ly = Math.min(frame.bottom - 6, star[1] + 30);
        drawMath(ctx, iterRuns(travel > 0 ? cur.k + 1 : cur.k), hx + 9, ly, {
          size: 13,
          color: colors.text,
          halo: colors.halo,
        });
      }
    } else if (Number.isFinite(head[0])) {
      // Off the view: chevron at the edge, the iterate named and placed.
      const right = hx > frame.right;
      const ex = right ? frame.right - 8 : frame.left + 8;
      // Stacked up from the bottom of the frame (clear of the view toolbar).
      const ey = Math.max(frame.top + 48, frame.bottom - 26 - offLabel * 20);
      offLabel++;
      ctx.fillStyle = color;
      ctx.beginPath();
      const s = right ? 1 : -1;
      ctx.moveTo(ex + s * 5, ey);
      ctx.lineTo(ex - s * 2, ey - 6);
      ctx.lineTo(ex - s * 2, ey + 6);
      ctx.closePath();
      ctx.fill();
      drawMath(
        ctx,
        [
          ...iterRuns(cur.k),
          mathMain(` = ${sig(head[0], 4)}, ${right ? 'right' : 'left'} of the view`),
        ],
        right ? ex - 10 : ex + 10,
        ey,
        {
          size: 12,
          align: right ? 'right' : 'left',
          color: colors.text2,
          halo: colors.halo,
          baseline: 'middle',
        },
      );
    }
    ctx.restore();
  }
}

/** Cobweb of the fixed-point iteration on g(x) = x − λ f(x) against y = x. */
function drawCobweb(
  ctx: CanvasRenderingContext2D,
  d: { runs: readonly PlotRun[]; P: ToPx; colors: ChartColors },
) {
  const run = d.runs.find((r) => r.methodId === 'fixed_point');
  if (!run || !run.trace.length) return;
  const { P, colors } = d;
  const color = colors.series[run.slot % colors.series.length];
  const n = run.trace.length;
  const t = Math.min(run.t, n - 1);
  const k = Math.floor(t);
  const u = t - k;
  const xs = run.trace.map((s) => s.x as number);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(...P(xs[0], xs[0]));
  for (let j = 0; j < k; j++) {
    ctx.lineTo(...P(xs[j], xs[j + 1])); // up/down to g(x_j) = x_{j+1}
    ctx.lineTo(...P(xs[j + 1], xs[j + 1])); // across to the diagonal
  }
  let head: [number, number] = [xs[k], xs[k]];
  if (k < n - 1 && u > 0) {
    const v = ease((u - HOLD) / (1 - HOLD));
    if (v < 0.5) head = [xs[k], xs[k] + (xs[k + 1] - xs[k]) * (v / 0.5)];
    else head = [xs[k] + (xs[k + 1] - xs[k]) * ((v - 0.5) / 0.5), xs[k + 1]];
    if (v >= 0.5) ctx.lineTo(...P(xs[k], xs[k + 1]));
    ctx.lineTo(...P(head[0], head[1]));
  }
  ctx.stroke();
  for (let j = 0; j <= k; j++) dot(ctx, P(xs[j], xs[j]), color, colors.halo, 2.2);
  dot(ctx, P(head[0], head[1]), color, colors.halo, 4.75);
  drawMath(ctx, iterRuns(k), P(head[0], head[1])[0] + 9, P(head[0], head[1])[1] - 11, {
    size: 13,
    color: colors.text,
    halo: colors.halo,
  });
  ctx.restore();
}
