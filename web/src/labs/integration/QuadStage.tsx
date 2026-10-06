/**
 * The quadrature stage: small multiples, one panel per selected method, on a shared x-axis and
 * a shared y-range. Each panel draws f, the area the rule integrates at the current step
 * (geometry.ts), the signed error between f and that area (hatched ╱ where f is above the rule,
 * ╲ where it is below; Gauss: against its interpolant), the nodes, and the integration interval
 * [a, b], whose ends can be dragged. The nested rules (Clenshaw–Curtis, Gauss–Patterson) mark the
 * nodes the previous step already evaluated (filled) apart from the new ones (rings), and
 * Clenshaw–Curtis draws the semicircle whose equally spaced points cast its cosine-spaced nodes. Between two steps the old and new geometry cross-fade (never
 * morph; only the incoming step is hatched); with reduced motion, or while scrubbing, the step
 * changes in one frame. Hatch tiles and hatch paths are cached: they are the costly part of a frame.
 */
import { useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import type { Step } from '../../core/types';
import { useChartColors } from '../../ui/theme';
import { seriesVar, type ChartColors } from '../../ui/colors';
import { Formula, Swatch } from '../../ui/components';
import { int, sci, sig, tick as fmtTick } from '../../core/format';
import {
  adaptiveSample,
  crisp,
  drawMath,
  linearScale,
  linearTicks,
  mathVar,
  niceStep,
  useCanvas,
  type Scale,
} from '../../viz';
import { tickFont } from '../../viz/axes';
import { geometryAt, geometryExtent, RULE_KIND, type Geometry } from './geometry';
import { estimateSymbol } from './cardRule';
import { agreeingDigits } from './quadFormat';
import { Reserve } from './Reserve';
import { widest } from './widest';
import styles from './IntegrationLab.module.css';

export interface StageRun {
  id: string;
  name: string;
  slot: number;
  trace: readonly Step[];
  /** Local continuous playhead of this run. */
  t: number;
  error?: string;
}

export interface QuadStageProps {
  f: (x: number) => number;
  domain: readonly [number, number];
  ab: readonly [number, number];
  onAbChange: (ab: [number, number]) => void;
  runs: readonly StageRun[];
  exact: number | null;
  showError: boolean;
  /** Cross-fade between steps (off under reduced motion). */
  fade: boolean;
  focusId: string | null;
  onFocus: (id: string) => void;
}

const M = { left: 46, right: 14, top: 10 };
const BOTTOM_LAST = 26;
const BOTTOM = 8;

/**
 * Shared y-range: f over the domain, the line y = 0, and every shape the runs draw (`extents`:
 * a rule that overshoots f stays in view), padded. A shape may widen the range by at most half
 * of f's own span, so one wild low-order interpolant cannot flatten f.
 */
function yRange(
  f: (x: number) => number,
  d: readonly [number, number],
  extents: readonly [number, number][] = [],
): [number, number] {
  let lo = 0,
    hi = 0;
  for (let i = 0; i <= 600; i++) {
    const v = f(d[0] + ((d[1] - d[0]) * i) / 600);
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (hi === lo) hi = lo + 1;
  const room = 0.5 * (hi - lo);
  for (const [a, b] of extents) {
    if (Number.isFinite(a)) lo = Math.min(lo, Math.max(a, lo - room));
    if (Number.isFinite(b)) hi = Math.max(hi, Math.min(b, hi + room));
  }
  const pad = (hi - lo) * 0.08;
  return [lo < 0 ? lo - pad : 0, hi > 0 ? hi + pad : 0 + pad];
}

const tiles = new Map<string, HTMLCanvasElement | null>();
const patterns = new WeakMap<CanvasRenderingContext2D, Map<string, CanvasPattern | null>>();

/** A 7-px diagonal hatch pattern in `color` ('/' or '\'), cached per context, color and dpr. */
function hatch(ctx: CanvasRenderingContext2D, color: string, dir: 1 | -1, dpr: number) {
  const key = `${color}|${dir}|${dpr}`;
  let byCtx = patterns.get(ctx);
  if (!byCtx) patterns.set(ctx, (byCtx = new Map()));
  if (byCtx.has(key)) return byCtx.get(key) ?? null;
  if (!tiles.has(key)) tiles.set(key, hatchTile(color, dir, dpr));
  const tile = tiles.get(key);
  const p = tile ? ctx.createPattern(tile, 'repeat') : null;
  p?.setTransform(new DOMMatrix([1 / dpr, 0, 0, 1 / dpr, 0, 0]));
  byCtx.set(key, p);
  return p;
}

function hatchTile(color: string, dir: 1 | -1, dpr: number): HTMLCanvasElement | null {
  const s = Math.round(7 * dpr);
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  if (!g) return null;
  g.strokeStyle = color;
  g.lineWidth = Math.max(1, dpr * 0.9);
  g.beginPath();
  if (dir === 1) {
    g.moveTo(0, s);
    g.lineTo(s, 0);
    g.moveTo(-s / 2, s / 2);
    g.lineTo(s / 2, -s / 2);
    g.moveTo(s / 2, s * 1.5);
    g.lineTo(s * 1.5, s / 2);
  } else {
    g.moveTo(0, 0);
    g.lineTo(s, s);
    g.moveTo(-s / 2, s / 2);
    g.lineTo(s / 2, s * 1.5);
    g.moveTo(s / 2, -s / 2);
    g.lineTo(s * 1.5, s / 2);
  }
  g.stroke();
  return c;
}

const smooth = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

const ROMBERG_DRAWN = ['trapezoid', 'Simpson arches', 'Boole quartics'];

/** Romberg: the column j drawn at row k, its value R(k, j) and that value's error. */
function rombergDrawn(step: Step, exact: number | null) {
  const j = Math.min(step.k, 2);
  const v = (step.info.row as number[] | undefined)?.[j] ?? NaN;
  return { j, v, err: exact === null ? null : Math.abs(v - exact) };
}

/**
 * The rule's own name, for a panel head on a phone: the registry name without "Composite" and
 * without the family word ("Composite trapezoidal rule" → "Trapezoidal", "Gauss–Legendre
 * quadrature" → "Gauss–Legendre"). The button keeps the full name as its accessible name.
 */
function ruleName(name: string): string {
  const s = name
    .replace(/^Composite\s+/, '')
    .replace(/\s+(rule|quadrature|integration)$/, '')
    .trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** One panel's header text (meta) for a step. */
function metaText(
  method: string,
  step: Step | undefined,
  trace: readonly Step[],
  exact: number | null = null,
): string {
  if (!step) return '';
  const info = step.info;
  switch (RULE_KIND[method]) {
    case 'romberg': {
      const N = info.n_panels as number;
      const { j, err } = rombergDrawn(step, exact);
      const panels = `${int(N)} panel${N === 1 ? '' : 's'}`;
      if (step.k <= 2)
        return `row k = ${step.k} · ${ROMBERG_DRAWN[j]} = R(${step.k}, ${step.k}) on ${panels}`;
      return `row k = ${step.k} · drawn: Boole R(${step.k}, 2)${err !== null ? `, error ${sci(err, 2)}` : ''}`;
    }
    case 'gauss':
      return `m = ${info.n_points as number} node${info.n_points === 1 ? '' : 's'} · cell areas w̃ᵢ f(xᵢ)`;
    case 'adaptive': {
      const last = trace.length - 1;
      return step.k === 0
        ? `S₁ on [a, b] · ${last} intervals to accept`
        : `interval ${step.k} of ${last} · depth ${info.depth as number}${info.forced ? ' · forced' : ''}`;
    }
    case 'monte-carlo':
      return `N = ${int(info.n_samples as number)} samples`;
    case 'nested': {
      const { head, nodes, reused, fresh } = nestedCounts(method, step);
      return step.k === 0 ? `${head}${nodes}` : `${head}${nodes}: ${reused} reused, ${fresh} new`;
    }
    default:
      return `N = ${int(info.n_panels as number)} · h = ${sig(step.stepSize, 4)}`;
  }
}

/** Header counts of a nested rule at a step: degree (Clenshaw–Curtis), nodes, reused and new. */
function nestedCounts(method: string, step: Step) {
  const N = step.info.n_points as number;
  const fresh = step.info.new_nodes as number;
  return {
    head: method === 'clenshaw_curtis' ? `n = ${int(step.info.n as number)} · ` : '',
    nodes: `${int(N)} node${N === 1 ? '' : 's'}`,
    reused: int(N - fresh),
    fresh: int(fresh),
  };
}

/**
 * The visible header of a nested rule: a filled dot and the reused count, a ring and the new
 * count. The estimate's symbol beside it (C₃₂, P₃₁) already gives n or N, so the header stays
 * short enough for four panels.
 */
function NestedMeta({ method, step, slot }: { method: string; step: Step; slot: number }) {
  const { head, nodes, reused, fresh } = nestedCounts(method, step);
  if (step.k === 0) return <>{`${head}${nodes}`}</>;
  return (
    <span className={styles.nestedMeta} style={{ ['--_c' as string]: seriesVar(slot) }}>
      <span className={styles.metaDot} aria-hidden="true" />
      {`${reused} reused`}
      <span className={`${styles.metaDot} ${styles.metaRing}`} aria-hidden="true" />
      {`${fresh} new`}
    </span>
  );
}

/** The widest header texts of a run (meta, symbol, estimate, error), cached per trace. */
interface RunWidths {
  meta: string[];
  /** Nested rules: the step whose header is the longest (its dots are not text). */
  metaStep?: Step;
  symbol: string[];
  estimate: string[];
  error: string[];
}
const widthCache = new WeakMap<readonly Step[], { key: string; v: RunWidths }>();

function runWidths(method: string, trace: readonly Step[], exact: number | null): RunWidths {
  const key = `${method}|${exact}`;
  const hit = widthCache.get(trace);
  if (hit && hit.key === key) return hit.v;
  const metas = trace.map((st) => metaText(method, st, trace, exact));
  let longest = 0;
  metas.forEach((t, i) => {
    if (t.length > metas[longest].length) longest = i;
  });
  const v: RunWidths = {
    meta: widest(metas),
    metaStep: trace[longest],
    symbol: widest(trace.map((st) => estimateSymbol(method, st))),
    estimate: widest(
      trace
        .map((st) => agreeingDigits(st.info.estimate as number, exact, 10))
        .map((d) => d.good + d.rest),
    ),
    error: widest(
      trace.map((st) => {
        const e = st.info.error as number | null | undefined;
        return e === null || e === undefined ? '' : sci(e, 2);
      }),
    ),
  };
  widthCache.set(trace, { key, v });
  return v;
}

/** Extents per trace (a trace is immutable; its extent depends on [a, b] only through it). */
const extentCache = new WeakMap<readonly Step[], [number, number]>();

function extentOf(r: StageRun, ab: readonly [number, number], f: (x: number) => number) {
  let e = extentCache.get(r.trace);
  if (!e) extentCache.set(r.trace, (e = geometryExtent(r.id, r.trace, ab, f)));
  return e;
}

export function QuadStage(props: QuadStageProps) {
  const { runs, f, domain, ab } = props;
  const yDom = yRange(
    f,
    domain,
    runs.filter((r) => !r.error).map((r) => extentOf(r, ab, f)),
  );
  return (
    <div className={styles.panels} data-count={runs.length}>
      {runs.map((r, i) => (
        <QuadPanel key={r.id} run={r} last={i === runs.length - 1} yDom={yDom} {...props} />
      ))}
    </div>
  );
}

function QuadPanel({
  run,
  last,
  f,
  domain,
  ab,
  onAbChange,
  exact,
  showError,
  fade,
  focusId,
  onFocus,
  yDom,
}: QuadStageProps & { run: StageRun; last: boolean; yDom: [number, number] }) {
  const colors = useChartColors();
  const box = useRef<HTMLDivElement>(null);
  const cache = useRef(new WeakMap<readonly Step[], Map<string, Geometry | null>>());
  const fCache = useRef<{ key: string; f: QuadStageProps['f']; path: Path2D } | null>(null);
  const [drag, setDrag] = useState<'a' | 'b' | null>(null);
  const [near, setNear] = useState<'a' | 'b' | null>(null);
  const k = Math.min(Math.floor(run.t + 1e-9), run.trace.length - 1);
  const step = run.trace[k];

  const scales = (w: number, h: number) => ({
    x: linearScale([domain[0], domain[1]], [M.left, w - M.right]),
    y: linearScale(yDom, [h - (last ? BOTTOM_LAST : BOTTOM), M.top]),
  });

  const geom = (kk: number, pxPerUnit: number) => {
    let m = cache.current.get(run.trace);
    if (!m) cache.current.set(run.trace, (m = new Map()));
    const key = `${kk}|${Math.round(pxPerUnit)}`;
    if (!m.has(key)) m.set(key, geometryAt(run.id, run.trace, kk, ab, f, pxPerUnit));
    return m.get(key) ?? null;
  };

  const { canvasRef } = useCanvas((ctx, s) => {
    const { x, y } = scales(s.width, s.height);
    const frame = {
      left: M.left,
      top: M.top,
      right: s.width - M.right,
      bottom: s.height - (last ? BOTTOM_LAST : BOTTOM),
    };
    const color = colors.series[run.slot % colors.series.length];
    drawGrid(ctx, x, y, frame, colors, s.dpr, last);

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top - 2, frame.right - frame.left, frame.bottom - frame.top + 2);
    ctx.clip();

    if (step && !run.error) {
      const pxPerUnit = (frame.right - frame.left) / (domain[1] - domain[0]);
      const k0 = Math.min(Math.floor(run.t + 1e-9), run.trace.length - 1);
      const u = fade && k0 + 1 < run.trace.length ? smooth((run.t - k0 - 0.55) / 0.45) : 0;
      const layers: [Geometry | null, number][] =
        u > 0
          ? [
              [geom(k0, pxPerUnit), 1 - u],
              [geom(k0 + 1, pxPerUnit), u],
            ]
          : [[geom(k0, pxPerUnit), 1]];
      layers.forEach(([g, alpha], li) => {
        if (g)
          drawGeometry(ctx, g, {
            x,
            y,
            frame,
            colors,
            color,
            alpha,
            f,
            ab,
            // During a cross-fade only the incoming step is hatched (half the cost, no double hatch).
            showError: showError && li === layers.length - 1,
            dpr: s.dpr,
            method: run.id,
          });
      });
    }

    // Monte Carlo samples lie on f: draw them under the curve so f stays legible.
    const pxPerUnitNow = (frame.right - frame.left) / (domain[1] - domain[0]);
    const gNow = step && !run.error ? geom(k, pxPerUnitNow) : null;
    if (gNow?.samples) drawNodes(ctx, gNow, { x, y, frame, colors, color, method: run.id });

    // f over the whole domain (its path is cached per size: it does not change with the step).
    const fKey = `${s.width}x${s.height}|${domain[0]},${domain[1]}|${yDom[0]},${yDom[1]}`;
    if (!fCache.current || fCache.current.key !== fKey || fCache.current.f !== f) {
      const P = (a: number, b: number): [number, number] => [x(a), y(b)];
      const path = new Path2D();
      let pen = false;
      for (const [a, b] of adaptiveSample(f, domain[0], domain[1], P)) {
        const [px, py] = P(a, b);
        if (!Number.isFinite(py)) {
          pen = false;
          continue;
        }
        if (pen) path.lineTo(px, py);
        else path.moveTo(px, py);
        pen = true;
      }
      fCache.current = { key: fKey, f, path };
    }
    ctx.strokeStyle = colors.text;
    ctx.globalAlpha = 0.88;
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.stroke(fCache.current.path);
    ctx.globalAlpha = 1;

    // Nodes on top of the curve.
    if (gNow && !gNow.samples) drawNodes(ctx, gNow, { x, y, frame, colors, color, method: run.id });

    // The veil outside [a, b] and the interval ends.
    ctx.fillStyle = colors.surface;
    ctx.globalAlpha = 0.62;
    if (ab[0] > domain[0])
      ctx.fillRect(frame.left, frame.top - 2, x(ab[0]) - frame.left, frame.bottom - frame.top + 2);
    if (ab[1] < domain[1])
      ctx.fillRect(x(ab[1]), frame.top - 2, frame.right - x(ab[1]), frame.bottom - frame.top + 2);
    ctx.globalAlpha = 1;
    ctx.restore();
    for (const end of ['a', 'b'] as const) {
      const v = end === 'a' ? ab[0] : ab[1];
      const px = crisp(x(v), s.dpr);
      const hot = drag === end || near === end;
      ctx.strokeStyle = hot ? colors.accent : colors.text2;
      ctx.lineWidth = hot ? 1.5 : 1;
      ctx.setLineDash(hot ? [] : [3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // Grip: a small rounded tab on the axis.
      ctx.fillStyle = hot ? colors.accent : colors.text2;
      const gy = frame.bottom - 9;
      ctx.beginPath();
      ctx.roundRect(px - 3, gy - 7, 6, 14, 3);
      ctx.fill();
      if (last)
        drawMath(ctx, [mathVar(end)], end === 'a' ? px + 7 : px - 7, frame.bottom - 22, {
          size: 13,
          align: end === 'a' ? 'left' : 'right',
          color: hot ? colors.accent : colors.text,
          halo: colors.halo,
        });
    }
  });

  // ── Dragging the interval ends ─────────────────────────────────────────────────────────
  const xAt = (clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return null;
    const { x } = scales(r.width, r.height);
    return { v: x.invert(clientX - r.left), px: clientX - r.left, x };
  };
  const hit = (clientX: number): 'a' | 'b' | null => {
    const p = xAt(clientX);
    if (!p) return null;
    const da = Math.abs(p.x(ab[0]) - p.px),
      db = Math.abs(p.x(ab[1]) - p.px);
    if (Math.min(da, db) > 14) return null;
    return da <= db ? 'a' : 'b';
  };
  const move = (end: 'a' | 'b', clientX: number) => {
    const p = xAt(clientX);
    if (!p) return;
    const w = domain[1] - domain[0];
    const stepv = niceStep(0, w, 200);
    let v = Math.round(p.v / stepv) * stepv;
    v = Number(v.toPrecision(10));
    const gap = w / 40;
    if (end === 'a') {
      v = Math.max(domain[0], Math.min(v, ab[1] - gap));
      if (v !== ab[0]) onAbChange([v, ab[1]]);
    } else {
      v = Math.min(domain[1], Math.max(v, ab[0] + gap));
      if (v !== ab[1]) onAbChange([ab[0], v]);
    }
  };
  const onDown = (e: RPointerEvent<HTMLDivElement>) => {
    const h = hit(e.clientX);
    if (!h) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag(h);
  };
  const onMove = (e: RPointerEvent<HTMLDivElement>) => {
    if (drag) move(drag, e.clientX);
    else if (e.pointerType === 'mouse') setNear(hit(e.clientX));
  };
  const onUp = () => setDrag(null);

  // The widest header texts of the run: the meta, the estimate and the error keep one width
  // for the whole replay, so nothing in the header moves while the numbers change.
  const widths = useMemo(() => runWidths(run.id, run.trace, exact), [run.id, run.trace, exact]);
  const est = step ? (step.info.estimate as number) : null;
  const err = step ? (step.info.error as number | null) : null;
  const digits = est !== null ? agreeingDigits(est, exact, 10) : null;
  const focused = focusId === run.id;
  const drawn =
    step && RULE_KIND[run.id] === 'romberg' && step.k >= 3 ? rombergDrawn(step, exact) : null;
  const label = run.error
    ? `${run.name}: could not run (${run.error}).`
    : step
      ? `${run.name}, step ${k} of ${run.trace.length - 1}: ${metaText(run.id, step, run.trace, exact)}; estimate ${sig(est, 10)}${err !== null ? `, error ${sci(err, 3)}` : ''}.${drawn ? ` The area drawn is R(${step.k}, 2) = ${sig(drawn.v, 10)}${drawn.err !== null ? `, error ${sci(drawn.err, 3)}` : ''}.` : ''} Interval from ${sig(ab[0], 4)} to ${sig(ab[1], 4)}.`
      : run.name;

  return (
    <div className={styles.panel} data-focused={focused || undefined}>
      <div className={styles.panelHead}>
        <button
          type="button"
          className={styles.panelName}
          aria-pressed={focused}
          onClick={() => onFocus(run.id)}
          title={`Show ${run.name} in the details below`}
          aria-label={run.name}
        >
          <Swatch slot={run.slot} size={9} />
          <span className={styles.nameFull}>{run.name}</span>
          <span className={styles.nameShort} aria-hidden="true">
            {ruleName(run.name)}
          </span>
        </button>
        <Reserve
          className={styles.panelMeta}
          widest={
            run.error ? null : widths.metaStep && RULE_KIND[run.id] === 'nested' ? (
              <NestedMeta method={run.id} step={widths.metaStep} slot={run.slot} />
            ) : (
              widths.meta
            )
          }
        >
          {run.error ? (
            'invalid input'
          ) : step && RULE_KIND[run.id] === 'nested' ? (
            <NestedMeta method={run.id} step={step} slot={run.slot} />
          ) : (
            metaText(run.id, step, run.trace, exact)
          )}
        </Reserve>
        {step && digits && !run.error && (
          <span className={styles.panelEst}>
            <Reserve
              className={styles.panelSym}
              align="end"
              widest={widths.symbol.map((tex) => (
                <Formula key={tex} tex={tex} fallback="I" />
              ))}
            >
              <Formula tex={estimateSymbol(run.id, step)} fallback="I" />
            </Reserve>
            <span className={styles.mono}>
              {' = '}
              <Reserve widest={widths.estimate}>
                <span className={styles.good}>{digits.good}</span>
                <span className={styles.rest}>{digits.rest}</span>
              </Reserve>
            </span>
            {widths.error.some(Boolean) && (
              <span className={styles.panelErr}>
                error{' '}
                <Reserve widest={widths.error} className={styles.mono}>
                  {err !== null ? sci(err, 2) : ''}
                </Reserve>
              </span>
            )}
          </span>
        )}
      </div>
      <div
        ref={box}
        className={styles.panelCanvas}
        role="img"
        aria-label={label}
        data-drag={drag ?? near ?? undefined}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onPointerLeave={() => setNear(null)}
      >
        <canvas ref={canvasRef} />
        {run.error && <p className={styles.panelError}>{run.error}</p>}
      </div>
    </div>
  );
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  x: Scale,
  y: Scale,
  frame: { left: number; top: number; right: number; bottom: number },
  colors: ChartColors,
  dpr: number,
  last: boolean,
) {
  const h = frame.bottom - frame.top;
  const yt = linearTicks(y.domain[0], y.domain[1], Math.max(2, Math.floor(h / 34)));
  const xt = linearTicks(
    x.domain[0],
    x.domain[1],
    Math.max(3, Math.floor((frame.right - frame.left) / 80)),
  );
  ctx.save();
  ctx.lineWidth = 1 / dpr;
  ctx.strokeStyle = colors.grid;
  ctx.beginPath();
  for (const v of yt) {
    const py = crisp(y(v), dpr);
    ctx.moveTo(frame.left, py);
    ctx.lineTo(frame.right, py);
  }
  for (const v of xt) {
    const px = crisp(x(v), dpr);
    ctx.moveTo(px, frame.top);
    ctx.lineTo(px, frame.bottom);
  }
  ctx.stroke();
  // y = 0 baseline (the area is measured from it).
  if (y.domain[0] <= 0 && y.domain[1] >= 0) {
    ctx.strokeStyle = colors.axis;
    ctx.beginPath();
    const p0 = crisp(y(0), dpr);
    ctx.moveTo(frame.left, p0);
    ctx.lineTo(frame.right, p0);
    ctx.stroke();
  }
  ctx.font = tickFont(colors);
  ctx.fillStyle = colors.tick;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const ys = y.tickStep(Math.max(2, Math.floor(h / 34)));
  for (const v of yt) {
    const py = y(v);
    if (py < frame.top + 4 || py > frame.bottom - 2) continue;
    ctx.fillText(fmtTick(v, ys), frame.left - 6, py);
  }
  if (last) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xs = x.tickStep(Math.max(3, Math.floor((frame.right - frame.left) / 80)));
    for (const v of xt) ctx.fillText(fmtTick(v, xs), x(v), frame.bottom + 6);
  }
  ctx.restore();
}

interface DrawCtx {
  x: Scale;
  y: Scale;
  frame: { left: number; top: number; right: number; bottom: number };
  colors: ChartColors;
  color: string;
  method: string;
}

function drawGeometry(
  ctx: CanvasRenderingContext2D,
  g: Geometry,
  o: DrawCtx & {
    alpha: number;
    f: (x: number) => number;
    ab: readonly [number, number];
    showError: boolean;
    dpr: number;
  },
) {
  const { x, y, frame, colors, color, alpha } = o;
  const y0 = y(0);
  const kind = RULE_KIND[o.method];
  if (g.tooFine) {
    // More nodes than a step stores for display: shade the area under f and say so.
    ctx.save();
    ctx.globalAlpha = 0.16 * alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x(o.ab[0]), y0);
    for (const [a, b] of adaptiveSample(o.f, o.ab[0], o.ab[1], (a, b) => [x(a), y(b)]))
      ctx.lineTo(x(a), y(b));
    ctx.lineTo(x(o.ab[1]), y0);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    ctx.save();
    ctx.globalAlpha = alpha;
    drawMath(
      ctx,
      [{ t: 'more than 4,096 nodes: too fine to draw panel by panel', style: 'sans' }],
      (frame.left + frame.right) / 2,
      frame.top + 14,
      {
        size: 11,
        align: 'center',
        color: colors.text2,
        halo: colors.halo,
      },
    );
    ctx.restore();
    return;
  }

  const pieces = g.pieces;
  const pxW = (p: (typeof pieces)[number]) => x(p.top[p.top.length - 1][0]) - x(p.top[0][0]);
  const thin = pieces.length > 0 && pieces.reduce((s, p) => s + pxW(p), 0) / pieces.length < 3.5;

  // 1. Fills (pending intervals hatched, the current one stronger).
  ctx.save();
  const pendingHatch = hatch(ctx, color, 1, o.dpr);
  for (const p of pieces) {
    ctx.beginPath();
    ctx.moveTo(x(p.top[0][0]), y0);
    for (const [a, b] of p.top) ctx.lineTo(x(a), y(b));
    ctx.lineTo(x(p.top[p.top.length - 1][0]), y0);
    ctx.closePath();
    if (p.pending && pendingHatch) {
      ctx.globalAlpha = 0.14 * alpha;
      ctx.fillStyle = pendingHatch;
    } else {
      ctx.globalAlpha = (p.current ? 0.36 : kind === 'monte-carlo' ? 0.1 : 0.17) * alpha;
      ctx.fillStyle = color;
    }
    ctx.fill();
  }
  ctx.restore();

  // 2. The signed error between f and the rule (not for Monte Carlo: its rectangle is a mean).
  // Skipped when the pieces are under 3.5 px wide: the hatch would be invisible there anyway.
  if (o.showError && kind !== 'monte-carlo' && pieces.length && !thin) drawErrorHatch(ctx, g, o);

  // 3. Outlines: piece tops and panel edges.
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineJoin = 'round';
  for (const p of pieces) {
    ctx.globalAlpha = (p.pending ? 0.55 : 0.95) * alpha;
    ctx.lineWidth = p.current ? 2 : thin ? 1 : 1.4;
    ctx.setLineDash(p.pending ? [4, 3] : []);
    ctx.beginPath();
    p.top.forEach(([a, b], i) => (i ? ctx.lineTo(x(a), y(b)) : ctx.moveTo(x(a), y(b))));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (!thin) {
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5 * alpha;
    ctx.beginPath();
    for (const p of pieces) {
      for (const end of [p.top[0], p.top[p.top.length - 1]]) {
        ctx.moveTo(crisp(x(end[0]), o.dpr), y0);
        ctx.lineTo(crisp(x(end[0]), o.dpr), y(end[1]));
      }
    }
    ctx.stroke();
  }
  ctx.restore();

  // Gauss: the interpolant of degree m − 1 (Gₘ integrates it exactly).
  if (g.interpolant) {
    ctx.save();
    ctx.globalAlpha = 0.85 * alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    let pen = false;
    const n = Math.max(80, Math.ceil(x(o.ab[1]) - x(o.ab[0])));
    for (let i = 0; i <= n; i++) {
      const xv = o.ab[0] + ((o.ab[1] - o.ab[0]) * i) / n;
      const py = y(g.interpolant(xv));
      if (!Number.isFinite(py) || Math.abs(py) > 1e5) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(x(xv), py);
      else ctx.moveTo(x(xv), py);
      pen = true;
    }
    ctx.stroke();
    ctx.restore();
  }

  // Clenshaw–Curtis: the semicircle construction xⱼ = cos(jπ/n), drawn as a half-ellipse on
  // [a, b] near the axis while the construction is far enough apart to read (n ≤ 16).
  if (g.chebyshev && g.chebyshev <= 16) drawSemicircle(ctx, g.chebyshev, o);

  // Monte Carlo: ± one standard error around the mean height.
  if (g.mean && Number.isFinite(g.mean.y)) {
    ctx.save();
    if (g.mean.se !== null) {
      ctx.globalAlpha = 0.16 * alpha;
      ctx.fillStyle = color;
      const t = y(g.mean.y + g.mean.se),
        b = y(g.mean.y - g.mean.se);
      ctx.fillRect(x(o.ab[0]), t, x(o.ab[1]) - x(o.ab[0]), Math.max(1, b - t));
    }
    ctx.globalAlpha = alpha;
    drawMath(ctx, [mathVar('f\u0304')], x(o.ab[1]) - 8, y(g.mean.y) - 8, {
      size: 13,
      align: 'right',
      color: colors.text,
      halo: colors.halo,
    });
    ctx.restore();
  }
}

/**
 * The points at angles jπ/n, j = 0, …, n, on a half-ellipse over [a, b] (height ~ a fifth of the
 * plot) and their shadows: a dotted drop line from each point to its node xⱼ on the axis. Equal
 * angles, unequal shadows: the nodes crowd towards a and b.
 */
function drawSemicircle(
  ctx: CanvasRenderingContext2D,
  n: number,
  o: DrawCtx & { alpha: number; ab: readonly [number, number]; dpr: number },
) {
  const { x, y, frame, colors, alpha } = o;
  const x0 = x(o.ab[0]),
    x1 = x(o.ab[1]);
  const cx = (x0 + x1) / 2,
    rx = (x1 - x0) / 2;
  const base = Math.min(frame.bottom, Math.max(frame.top, y(0)));
  const ry = Math.min(rx, Math.max(18, (frame.bottom - frame.top) * 0.2));
  ctx.save();
  ctx.globalAlpha = 0.5 * alpha;
  ctx.strokeStyle = colors.text2;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(cx, base, rx, ry, 0, Math.PI, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([1.5, 2.5]);
  ctx.beginPath();
  for (let j = 0; j <= n; j++) {
    const th = (j * Math.PI) / n;
    const px = cx + rx * Math.cos(th);
    ctx.moveTo(crisp(px, o.dpr), base - ry * Math.sin(th));
    ctx.lineTo(crisp(px, o.dpr), base);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = colors.text2;
  for (let j = 0; j <= n; j++) {
    const th = (j * Math.PI) / n;
    ctx.beginPath();
    ctx.arc(cx + rx * Math.cos(th), base - ry * Math.sin(th), 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

interface HatchPaths {
  key: string;
  above: Path2D;
  below: Path2D;
  fDown: Path2D;
  fUp: Path2D;
}
const hatchPaths = new WeakMap<Geometry, HatchPaths>();

/**
 * Hatch where f is above the rule's top (╱, the rule misses area) and below it (╲). The rule's
 * top is the pieces' tops, or `g.errorTop` (Gauss: the interpolant). The four clip and fill paths
 * depend only on the geometry and the scales, so they are cached per geometry and size.
 */
function drawErrorHatch(
  ctx: CanvasRenderingContext2D,
  g: Geometry,
  o: DrawCtx & {
    alpha: number;
    f: (x: number) => number;
    ab: readonly [number, number];
    dpr: number;
  },
) {
  const { x, y, frame } = o;
  const up = hatch(ctx, o.colors.text2, 1, o.dpr);
  const down = hatch(ctx, o.colors.text2, -1, o.dpr);
  if (!up || !down) return;
  const key = `${frame.left},${frame.top},${frame.right},${frame.bottom}|${x.domain.join(',')}|${y.domain.join(',')}`;
  let paths = hatchPaths.get(g);
  if (!paths || paths.key !== key) {
    const tops = g.errorTop ? [g.errorTop] : g.pieces.map((p) => p.top);
    const under = (to: number) => {
      const path = new Path2D();
      for (const t of tops) {
        path.moveTo(x(t[0][0]), y(t[0][1]));
        for (const [a, b] of t) path.lineTo(x(a), y(b));
        path.lineTo(x(t[t.length - 1][0]), to);
        path.lineTo(x(t[0][0]), to);
        path.closePath();
      }
      return path;
    };
    const lo = Math.min(...tops.map((t) => t[0][0]));
    const hi = Math.max(...tops.map((t) => t[t.length - 1][0]));
    const fPts = adaptiveSample(o.f, lo, hi, (a, b) => [x(a), y(b)]).filter(([, b]) =>
      Number.isFinite(b),
    );
    const fRegion = (to: number) => {
      const path = new Path2D();
      path.moveTo(x(lo), to);
      for (const [a, b] of fPts) path.lineTo(x(a), y(b));
      path.lineTo(x(hi), to);
      path.closePath();
      return path;
    };
    paths = {
      key,
      above: under(frame.top - 4),
      below: under(frame.bottom + 4),
      fDown: fRegion(frame.bottom + 4),
      fUp: fRegion(frame.top - 4),
    };
    hatchPaths.set(g, paths);
  }
  ctx.save();
  ctx.globalAlpha = 0.55 * o.alpha;
  ctx.save();
  ctx.clip(paths.above);
  ctx.fillStyle = up;
  ctx.fill(paths.fDown);
  ctx.restore();
  ctx.save();
  ctx.clip(paths.below);
  ctx.fillStyle = down;
  ctx.fill(paths.fUp);
  ctx.restore();
  ctx.restore();
}

function drawNodes(ctx: CanvasRenderingContext2D, g: Geometry, o: DrawCtx) {
  const { x, y, frame, colors, color } = o;
  const dot = (px: number, py: number, r: number, hollow = false) => {
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(px, py, r + 1.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    if (hollow) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
    }
  };
  ctx.save();
  // Monte Carlo samples: older faint, this step's in full color with a rug on the axis.
  if (g.samples) {
    const many = g.samples.length > 1500;
    for (const s of g.samples) {
      if (!Number.isFinite(s.y)) continue;
      ctx.globalAlpha = s.fresh ? (many ? 0.5 : 0.95) : many ? 0.12 : 0.3;
      ctx.fillStyle = s.fresh ? color : colors.text2;
      ctx.beginPath();
      ctx.arc(x(s.x), y(s.y), s.fresh ? (many ? 1.3 : 1.9) : 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 0.8;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const s of g.samples)
      if (s.fresh) {
        ctx.moveTo(x(s.x), frame.bottom);
        ctx.lineTo(x(s.x), frame.bottom - 5);
      }
    ctx.stroke();
    ctx.restore();
    return;
  }
  const nodes = g.nodes;
  const spacing =
    nodes.length > 1 ? (x(nodes[nodes.length - 1].x) - x(nodes[0].x)) / (nodes.length - 1) : 99;
  const maxW = Math.max(...nodes.map((n) => n.weight ?? 0));
  if (RULE_KIND[o.method] === 'nested') {
    drawNestedNodes(ctx, nodes, spacing, o);
  } else if (RULE_KIND[o.method] === 'gauss') {
    // Stems and weight-sized dots: area ∝ wᵢ.
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    for (const n of nodes) {
      ctx.moveTo(x(n.x), y(0));
      ctx.lineTo(x(n.x), y(n.y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    const rMax = Math.max(3, Math.min(8.5, spacing * 0.42));
    for (const n of nodes)
      dot(x(n.x), y(n.y), Math.max(1.8, rMax * Math.sqrt((n.weight ?? 0) / (maxW || 1))));
    // Ticks on the axis at the nodes.
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const n of nodes) {
      ctx.moveTo(x(n.x), y(0) - 4);
      ctx.lineTo(x(n.x), y(0) + 4);
    }
    ctx.stroke();
  } else if (spacing >= 5) {
    const r = spacing >= 14 ? 3 : 2.2;
    for (const n of nodes)
      if (Number.isFinite(n.y)) dot(x(n.x), y(n.y), n.fresh ? r + 1 : r, n.fresh);
  } else {
    // Dense grids: only this step's new nodes, as a rug on the axis.
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const n of nodes)
      if (n.fresh || RULE_KIND[o.method] === 'adaptive') {
        ctx.moveTo(x(n.x), frame.bottom);
        ctx.lineTo(x(n.x), frame.bottom - 4);
      }
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Nodes of a nested rule: a dot of area ∝ weight at (xᵢ, f(xᵢ)) and a tick on the axis. A node the
 * previous step already evaluated is a filled dot with a short grey tick; a new node is a ring
 * with a long tick in the series color, so each step shows which values it pays for. Above 200
 * nodes only the ticks remain (the dots would merge).
 */
function drawNestedNodes(
  ctx: CanvasRenderingContext2D,
  nodes: readonly QNodeLike[],
  spacing: number,
  o: DrawCtx,
) {
  const { x, y, colors, color } = o;
  const y0 = y(0);
  const pts = nodes.filter((n) => Number.isFinite(n.y));
  const minGap = pts.reduce(
    (m, n, i) => (i ? Math.min(m, Math.abs(x(n.x) - x(pts[i - 1].x))) : m),
    Infinity,
  );
  const dots = pts.length <= 200;
  if (dots && pts.length <= 70) {
    // Stems: dotted, from the axis to the node.
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    for (const n of pts) {
      ctx.moveTo(x(n.x), y0);
      ctx.lineTo(x(n.x), y(n.y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // Axis ticks: new nodes long and colored, reused nodes short and grey.
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1.5;
  for (const fresh of [false, true]) {
    ctx.strokeStyle = fresh ? color : colors.text2;
    ctx.beginPath();
    for (const n of pts)
      if (Boolean(n.fresh) === fresh) {
        ctx.moveTo(x(n.x), y0 - (fresh ? 5 : 3));
        ctx.lineTo(x(n.x), y0 + (fresh ? 5 : 3));
      }
    ctx.stroke();
  }
  if (!dots) return;
  const maxW = Math.max(...pts.map((n) => n.weight ?? 0));
  // Radius ∝ √weight, at most 42 % of the mean spacing; the smallest dots stay visible.
  const rMax = Math.max(3, Math.min(7, spacing * 0.42));
  for (const n of pts) {
    const r = Math.max(2, rMax * Math.sqrt((n.weight ?? 0) / (maxW || 1)));
    const px = x(n.x),
      py = y(n.y);
    ctx.fillStyle = colors.halo;
    ctx.beginPath();
    ctx.arc(px, py, r + 1.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    if (n.fresh) {
      // A ring needs room for its hole: at least 2.6 px.
      ctx.arc(px, py, Math.max(2.6, r), 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = minGap < 4 ? 1 : 1.5;
      ctx.stroke();
    } else {
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

type QNodeLike = Geometry['nodes'][number];
