/**
 * Constrained lab: min f(𝐱) s.t. cᵢ(𝐱) ≤ 0, cⱼ(𝐱) = 0 in the plane.
 *
 * The stage draws the contours of f (or of the function the focused method actually minimizes
 * at the current step: the penalty Q(𝐱; μ), the augmented Lagrangian L_A, the barrier
 * f − t⁻¹Σ log(−cᵢ) or the ℓ1 merit φ₁), the infeasible side hatched, every boundary cᵢ = 0
 * labelled, the iterate paths of up to four methods, the geometry of the focused method's
 * current step (projection arc and projection, Frank–Wolfe vertex atoms, quasi-Newton / Newton /
 * QP steps with their trials, the central path, SQP's linearized constraints), and the KKT
 * picture −∇f(𝐱ₖ) vs Σ λᵢ∇cᵢ(𝐱ₖ). Methods: src/methods/constrained/methods.ts.
 */
import './setup';
import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
  type RefObject,
} from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Step } from '../../core/types';
import type { ConstrainedProblem } from '../../problems/constrained';
import { sci, sig, superscript, vec } from '../../core/format';
import type { Vector } from '../../core/types';
import { useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import {
  Formula,
  PlaybackBar,
  SegmentedControl,
  Swatch,
  TabPanel,
  Tabs,
  Toggle,
} from '../../ui/components';
import { MAX_SERIES } from '../../ui/colors';
import {
  Contour2D,
  ConvergenceChart,
  IterationTable,
  drawOverlays2D,
  measureMath,
  useElementSize,
  TableView,
  mathMain as mm,
  mathSub as msub,
  mathSup as msup,
  mathVar as mv,
  mathBold as mb,
  overlappingPaths,
  starRuns,
  type Column,
  type MathRun,
  type Overlay2D,
  type PathSpec,
  type TableColumn,
  type View2D,
} from '../../viz';
import { addLabelRect } from '../../viz/labelRects';
import { LABEL_SIZE, labelFont } from '../../viz/axes';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { useStartWhenDrawn } from '../../play/useStartWhenDrawn';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  StartPointFields,
  useLabRunsState,
  useMethodSelection,
  useProblemState,
  useStartPoint,
  useKAxis,
  KAxisControl,
  focusRuns,
  type LabRun,
} from '../_shell';
import {
  BOOL_CODEC,
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  KEYS,
  LANDSCAPE_CODEC,
  METRIC_CODEC,
  PRESETS,
  VIEW_DOMAIN,
  type Landscape,
  type Metric,
} from './config';
import {
  MERIT_METHODS,
  feasibleOracle,
  feasibleSet,
  geometryIndex,
  drawLinearized,
  kktArrows,
  linearizedConstraints,
  meritLandscape,
  minimizerLabelSide,
  nearestMinimizer,
  stepGeometry,
  texVec,
} from './geometry';
import { compactNum as compact, constraintStatus, filledRule, twoLineLatex } from './rules';
import styles from './ConstrainedLab.module.css';

void preloadKatex();

/** Generic symbol of the function each method minimizes (landscape toggle). */
const MERIT_SYMBOL: Record<string, string> = {
  quadratic_penalty: 'Q(\\mathbf{x};\\mu)',
  augmented_lagrangian: 'L_A',
  log_barrier: 'B_t',
  sqp: '\\phi_1',
};

/** The outer parameter each method varies, for the iteration table. */
/** Step sizes are those of the move into x_k, so they carry the index k − 1 (as in the rules). */
const PARAM_COLUMN: Record<string, { key: string; tex: string } | undefined> = {
  projected_gradient: { key: 's', tex: 's_{k-1}' },
  frank_wolfe: { key: 'gamma', tex: '\\gamma_{k-1}' },
  quadratic_penalty: { key: 'mu', tex: '\\mu' },
  augmented_lagrangian: { key: 'mu', tex: '\\mu' },
  log_barrier: { key: 't', tex: 't' },
  sqp: { key: 'alpha', tex: '\\alpha_{k-1}' },
};

/** ‖c⁺(x)‖∞ (the positive part of the inequalities), defined once (tooltips, the KKT note, the chart). */
const VIOLATION_DEF =
  '‖c⁺(x)‖∞ = max( max over equalities |cᵢ(x)|, max over inequalities max(0, cᵢ(x)) ): the constraint violation (0 when x is feasible).';
const KKT_DEF =
  'rₖ: the stationarity measure the method stops on: the KKT residual (multiplier methods), the projection residual ‖xₖ − P_C(xₖ − ∇f)‖∞ (projected gradient) or the Frank–Wolfe gap.';

const METRIC_LABEL: Record<Metric, { y: string; runs: MathRun[] }> = {
  kkt: { y: 'KKT residual rₖ', runs: [mm('KKT residual '), mv('r'), msub('k')] },
  viol: {
    y: '‖c⁺(xₖ)‖∞',
    runs: [mm('‖'), mv('c'), msup('+'), mm('('), mb('x'), msub('k'), mm(')‖'), msub('∞')],
  },
  gap: {
    y: '|f(xₖ) − f(x⋆)|, x⋆ the listed minimizer nearest the final iterate',
    runs: [
      mm('|'),
      mv('f'),
      mm('('),
      mb('x'),
      msub('k'),
      mm(') − '),
      mv('f'),
      mm('('),
      ...starRuns('x'),
      mm(')|'),
    ],
  },
};

/** f(x_k) in the Iterations table: four significant digits, compact e-notation outside [10⁻³, 10⁴). */
const fShort = (v: unknown) =>
  typeof v === 'number' &&
  Number.isFinite(v) &&
  v !== 0 &&
  (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4)
    ? compact(v)
    : typeof v === 'number'
      ? sig(v, 4)
      : '—';

const headTip = (tip: string, tex: string) => (
  <span title={tip}>
    <Formula tex={tex} />
  </span>
);

/** True below `px` of viewport width (the Iterations table drops its parameter column there). */
function useNarrow(px: number): boolean {
  const query = `(max-width: ${px}px)`;
  const subscribe = useCallback(
    (cb: () => void) => {
      const m = window.matchMedia?.(query);
      m?.addEventListener('change', cb);
      return () => m?.removeEventListener('change', cb);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia?.(query).matches ?? false,
    () => false,
  );
}

function tableColumns(method: string | undefined, narrow: boolean): Column[] {
  const f = (t: string) => <Formula tex={t} />;
  const cols: Column[] = [
    { key: 'k', label: f('k'), value: (s) => s.k, align: 'right', width: '22px' },
    {
      key: 'x',
      label: f('\\mathbf{x}_k'),
      width: 'minmax(104px, 2fr)',
      value: (s) => vec(s.x as number[], 3),
    },
    {
      key: 'fun',
      label: f('f(\\mathbf{x}_k)'),
      align: 'right',
      width: 'minmax(44px, 1fr)',
      value: (s) => fShort(s.fun),
    },
    {
      key: 'viol',
      label: headTip(VIOLATION_DEF, '\\|c^+\\|_\\infty'),
      align: 'right',
      width: 'minmax(44px, 0.8fr)',
      value: (s) => compact(s.info.violation),
    },
    {
      key: 'kkt',
      label: headTip(KKT_DEF, 'r_k'),
      align: 'right',
      width: 'minmax(44px, 0.8fr)',
      value: (s) => compact(s.info.kkt_residual),
    },
  ];
  const pc = method && !narrow ? PARAM_COLUMN[method] : undefined;
  if (pc)
    cols.push({
      key: pc.key,
      label: f(pc.tex),
      align: 'right',
      width: 'minmax(44px, 0.7fr)',
      value: (s) => compact(s.info[pc.key]),
    });
  return cols;
}

// ── Canvas labels: drawn after the paths, with a halo ───────────────────────────────────

/**
 * The labels of `marks` (arrow, point and text labels) as text overlays at the place
 * `drawOverlays2D` gives them, so they can be drawn in a pass of their own after the paths: a
 * path or a dotted linearization never strikes a label through.
 */
function labelOverlays(marks: readonly Overlay2D[], v: View2D): Overlay2D[] {
  const toData = (px: number, py: number): [number, number] => [v.x.invert(px), v.y.invert(py)];
  const out: Overlay2D[] = [];
  for (const o of marks) {
    if (o.kind === 'arrow' && o.label) {
      const a = v.toPx(o.from[0], o.from[1]),
        b = v.toPx(o.to[0], o.to[1]);
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      // Beside the shaft's midpoint, on its left-hand side (as drawOverlays2D).
      const nx = -(b[1] - a[1]) / len,
        ny = (b[0] - a[0]) / len;
      out.push({
        kind: 'text',
        at: toData((a[0] + b[0]) / 2 + nx * 12, (a[1] + b[1]) / 2 + ny * 12),
        text: o.label,
        align: 'center',
      });
    } else if (o.kind === 'point' && o.label) {
      const [px, py] = v.toPx(o.at[0], o.at[1]);
      const gap = (o.radius ?? 4) + 6;
      const side = o.labelSide ?? 'right';
      const [lx, ly, align]: [number, number, 'left' | 'right' | 'center'] =
        side === 'right'
          ? [px + gap, py, 'left']
          : side === 'left'
            ? [px - gap, py, 'right']
            : side === 'above'
              ? [px, py - gap - 4, 'center']
              : [px, py + gap + 4, 'center'];
      out.push({ kind: 'text', at: toData(lx, ly), text: o.label, align });
    } else if (o.kind === 'text') out.push(o);
  }
  return out;
}

/** The marks without the labels `labelOverlays` takes out (drawn under the paths). */
function withoutLabels(marks: readonly Overlay2D[]): Overlay2D[] {
  return marks.flatMap((o): Overlay2D[] =>
    o.kind === 'text'
      ? []
      : (o.kind === 'arrow' || o.kind === 'point') && o.label
        ? [{ ...o, label: undefined }]
        : [o],
  );
}

/** The overlay view of a Contour2D view. */
const overlayView = (v: View2D) => ({
  toPx: v.toPx,
  toData: (px: number, py: number): [number, number] => [v.x.invert(px), v.y.invert(py)],
  width: v.width,
  height: v.height,
  dpr: v.dpr,
  colors: v.colors,
});

/** A label's box in CSS px, as `drawOverlays2D` registers it. */
function labelBox(ctx: CanvasRenderingContext2D, o: Overlay2D & { kind: 'text' }, v: View2D) {
  const size = Math.max(LABEL_SIZE, o.size ?? 12);
  const [px, py] = v.toPx(o.at[0], o.at[1]);
  ctx.save();
  ctx.font = labelFont(v.colors, Math.max(LABEL_SIZE, size - 0.5));
  const w =
    typeof o.text === 'string' ? ctx.measureText(o.text).width : measureMath(ctx, o.text, size);
  ctx.restore();
  const align = o.align ?? 'left';
  const left = align === 'left' ? px : align === 'center' ? px - w / 2 : px - w;
  return { x: left, y: py - size * 0.7, w, h: size * 1.4 };
}

const overlapping = (
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
) => a.x - 2 < b.x + b.w && a.x + a.w + 2 > b.x && a.y - 2 < b.y + b.h && a.y + a.h + 2 > b.y;

/**
 * The step's labels, placed: each label that would print over an earlier one moves to the
 * nearest free row (±1, ±2 label heights), so two labels never overlap.
 */
function placedLabels(
  ctx: CanvasRenderingContext2D,
  marks: readonly Overlay2D[],
  v: View2D,
): { label: Overlay2D; box: { x: number; y: number; w: number; h: number } }[] {
  const out: { label: Overlay2D; box: { x: number; y: number; w: number; h: number } }[] = [];
  for (const o of labelOverlays(marks, v)) {
    if (o.kind !== 'text') continue;
    const box = labelBox(ctx, o, v);
    let dy = 0;
    for (const d of [0, 1, -1, 2, -2].map((t) => t * box.h)) {
      if (!out.some((p) => overlapping({ ...box, y: box.y + d }, p.box))) {
        dy = d;
        break;
      }
    }
    const [px, py] = v.toPx(o.at[0], o.at[1]);
    out.push({
      label: { ...o, at: [v.x.invert(px), v.y.invert(py + dy)] },
      box: { ...box, y: box.y + dy },
    });
  }
  return out;
}

// ── Legend: one size for the whole run ──────────────────────────────────────────────────

/** The KKT arrows' drawing scale: two significant figures in one fixed format. */
function fmtScale(v: number): string {
  if (!Number.isFinite(v) || v <= 0) return '—';
  if (v < 1e-3 || v >= 1e4) return sci(v, 2);
  if (v >= 100) return String(Math.round(v));
  return v.toPrecision(2);
}

/**
 * A fragment as wide as the widest value it takes during the run: the current content and an
 * invisible copy of the widest one share one grid cell, so the legend never re-wraps.
 */
function Fixed({ children, widest }: { children: ReactNode; widest?: ReactNode }) {
  if (!widest) return <>{children}</>;
  return (
    <span className={styles.fixed}>
      <span>{children}</span>
      <span className={styles.ghost} aria-hidden="true">
        {widest}
      </span>
    </span>
  );
}

/**
 * Keep the tallest height `ref` has had as its min-height until `reset` changes (a new run, a
 * new landscape) or its width changes: the plot above it never resizes during playback.
 */
function useKeepHeight(ref: RefObject<HTMLElement | null>, reset: string) {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.minHeight = '';
    let max = 0;
    let width = el.clientWidth;
    const measure = () => {
      if (el.clientWidth !== width) {
        width = el.clientWidth;
        max = 0;
        el.style.minHeight = '';
      }
      const h = el.getBoundingClientRect().height;
      if (h > max + 0.5) {
        max = h;
        el.style.minHeight = `${Math.ceil(max)}px`;
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, reset]);
}

interface ConstraintRow {
  i: number;
  latex: string;
  value: number | null;
  lam: number | null;
  lamStar: number | null;
  status: string;
}

const CONSTRAINT_COLUMNS: TableColumn<ConstraintRow>[] = [
  { key: 'i', tex: 'i', value: (r) => r.i + 1, width: '1.5rem' },
  {
    key: 'c',
    header: 'Constraint',
    align: 'left',
    mono: false,
    value: (r) => (
      <span className={styles.constraintCell}>
        <Formula tex={r.latex} />
        <span className={styles.status} data-status={r.status}>
          {r.status}
        </span>
      </span>
    ),
  },
  {
    key: 'val',
    tex: 'c_i(\\mathbf{x}_k)',
    value: (r) => compact(r.value),
  },
  {
    // One column for the estimate and its reference, so the table fits the panel (≈ 340 px).
    key: 'lam',
    tex: '\\lambda_i\\ (\\lambda_i^\\star)',
    label: 'Multiplier estimate at step k (multiplier at the reference minimizer)',
    value: (r) => (
      <>
        {compact(r.lam)}
        <span className={styles.ref}> ({compact(r.lamStar)})</span>
      </>
    ),
  },
];

/** "Strictly feasible" or the constraints x₀ violates. */
function startStatus(p: ConstrainedProblem, x0: number[]): { ok: boolean; text: string } {
  const bad: string[] = [];
  let boundary = false;
  p.constraints.forEach((c, i) => {
    const v = c.fun(x0);
    if (c.kind === 'eq') {
      if (Math.abs(v) > 1e-12) bad.push(`c${sub(i + 1)} = ${sig(v, 3)}`);
    } else if (v > 0) bad.push(`c${sub(i + 1)} = ${sig(v, 3)} > 0`);
    else if (v === 0) boundary = true;
  });
  if (bad.length) return { ok: false, text: `𝐱₀ is infeasible: ${bad.join(', ')}.` };
  if (boundary) return { ok: true, text: '𝐱₀ lies on the boundary (feasible, not strictly).' };
  return { ok: true, text: '𝐱₀ is feasible.' };
}

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => String(n).replace(/\d/g, (d) => SUBS[Number(d)]);

/**
 * The input errors of the methods, as one typeset sentence. The raw message (the Python text,
 * kept by the port for parity) stays in "Copy Python call" reproductions.
 */
function friendlyError(r: LabRun, p: ConstrainedProblem): string | undefined {
  const e = r.error;
  if (!e) return e;
  if (r.sel.id === 'log_barrier') {
    const got = e.match(/got c\(x0\) = \[([^\]]*)\]/);
    if (got) {
      const c = got[1].split(',').map(Number);
      const bad = c.flatMap((v, i) =>
        p.constraints[i]?.kind === 'ineq' && !(v < 0) ? [`c${sub(i + 1)}(𝐱₀) = ${sig(v, 3)}`] : [],
      );
      if (bad.length)
        return `the log barrier needs every cᵢ(𝐱₀) < 0; here ${bad.join(', ')}. Drag 𝐱₀ inside the feasible set.`;
    }
    if (/affine equality/.test(e))
      return 'the log barrier handles only affine equality constraints (Boyd & Vandenberghe §11.1); this problem has a curved one.';
  }
  return e;
}

export default function ConstrainedLab() {
  const lab = getLab('constrained')!;
  const colors = useChartColors();

  const problems = useMemo(() => listProblems<ConstrainedProblem>('constrained'), []);
  const methods = useMemo(() => listMethods('constrained'), []);
  // The formula preview on two lines (objective, then constraints) so it fits the rail.
  const pickerOptions = useMemo(
    () =>
      problems.map((p) => ({
        ...p,
        latex: twoLineLatex(p.latex),
        // The brand writes the solution mark as ⋆, never * (the Python descriptions use λ*).
        description: p.description.replace(/\*/g, '⋆'),
      })),
    [problems],
  );
  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [x0, setX0] = useStartPoint(problem.x0, 2);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [landscape, setLandscape] = useUrlState<Landscape>(KEYS.landscape, 'f', LANDSCAPE_CODEC);
  const [metric, setMetric] = useUrlState<Metric>(KEYS.metric, 'kkt', METRIC_CODEC);
  const [showKkt, setShowKkt] = useUrlState<boolean>(KEYS.kkt, true, BOOL_CODEC);
  const [tab, setTab] = useState<'method' | 'steps' | 'kkt'>('method');
  const [focusId, setFocusId] = useState<string | null>(null);
  const narrow = useNarrow(420);

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]] }), [x0]);
  const { runs, pending } = useLabRunsState(problem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  // Autoplay waits for the landscape: no iterate moves over a hatched, unfilled stage.
  const player = useTracePlayer(traces, { autoplay: false });
  usePlayerKeyboard(player);

  const focus: LabRun | undefined =
    runs.find((r) => r.sel.id === focusId) ?? runs.find((r) => !r.error) ?? runs[0];
  const fi = focus ? runs.indexOf(focus) : 0;
  const focusTrace = useMemo(() => focus?.result.trace ?? [], [focus]);
  const gk = geometryIndex(player.localT(fi), focusTrace.length);
  const focusStep: Step | undefined = focusTrace[gk];
  const slot = focus?.sel.slot ?? 0;

  const domain = VIEW_DOMAIN[problem.id] ?? problem.domain;
  const span = Math.max(domain[0][1] - domain[0][0], domain[1][1] - domain[1][0]);
  const C = useMemo(() => feasibleOracle(problem), [problem]);

  // ── Landscape: f, or the function the focused method minimizes at this step ─────────────
  // A run that could not start has no step, hence no landscape to show.
  const hasMerit = !!focus && !focus.error && MERIT_METHODS.has(focus.sel.id);
  const merit =
    landscape === 'merit' && focus && hasMerit
      ? meritLandscape(focus.sel.id, focusStep, problem)
      : null;
  const objective = useCallback((x: number, y: number) => problem.f([x, y]), [problem]);
  const field = merit ? merit.f : objective;
  const fieldKey = merit ? `${problem.id}|${focus!.sel.id}|${merit.key}` : problem.id;

  // ── Overlays ────────────────────────────────────────────────────────────────────────────
  const baseOverlays = useMemo(() => feasibleSet(problem, domain), [problem, domain]);
  const projections = useMemo<Overlay2D[]>(
    () =>
      runs.flatMap((r) => {
        const s0 = r.result.trace[0];
        const from = s0?.info.projected_from as number[] | undefined;
        return from
          ? [
              {
                kind: 'segment' as const,
                from: [from[0], from[1]] as const,
                to: [(s0.x as number[])[0], (s0.x as number[])[1]] as const,
                slot: r.sel.slot,
                dashed: true,
                width: 1.3,
              },
            ]
          : [];
      }),
    [runs],
  );
  const geometry = useMemo(
    () =>
      focus && !focus.error
        ? stepGeometry(focus.sel.id, focusTrace, gk, slot, problem, C, domain)
        : [],
    [focus, focusTrace, gk, slot, problem, C, domain],
  );
  const kkt = useMemo(
    () => (showKkt && focusStep ? kktArrows(focusStep, problem, slot, span) : null),
    [showKkt, focusStep, problem, slot, span],
  );
  // Pixels per data unit of the plot (Contour2D fits the domain with equal aspect).
  const plotRef = useRef<HTMLDivElement>(null);
  const plotSize = useElementSize(plotRef);
  useStartWhenDrawn(player, plotRef, traces);
  const ppu =
    plotSize.width > 0
      ? Math.min(
          plotSize.width / (domain[0][1] - domain[0][0]),
          plotSize.height / (domain[1][1] - domain[1][0]),
        )
      : 550 / span;
  // Each run's step index at the playhead (a string key keeps the memo stable between steps).
  const headKey = runs
    .map((r, i) => geometryIndex(player.localT(i), r.result.trace.length))
    .join(',');
  const heads = useMemo(() => headKey.split(',').map(Number), [headKey]);
  // The "𝐱⋆ = (…)" label (one listed minimizer), placed away from the KKT arrows and from the
  // path's approach, drawn last so nothing in the overlays covers it.
  const starLabel = useMemo<Overlay2D[]>(() => {
    if (problem.minima.length !== 1) return [];
    const xs = problem.minima[0] as [number, number];
    // Marks the label must not cover: every path up to the playhead, the KKT arrows, the step.
    const marks: { from: [number, number]; to: [number, number]; weight: number }[] = [];
    runs.forEach((r, ri) => {
      const tr = r.result.trace;
      const w = r === focus ? 1 : 0.7;
      for (let j = 1; j <= Math.min(heads[ri] ?? 0, tr.length - 1); j++) {
        const a = tr[j - 1].x as number[],
          b = tr[j].x as number[];
        marks.push({ from: [a[0], a[1]], to: [b[0], b[1]], weight: w });
      }
    });
    const half = 22 / ppu; // a short label ≈ 44 px wide
    for (const o of [...(kkt?.overlays ?? []), ...geometry]) {
      if (o.kind === 'arrow' || o.kind === 'segment')
        marks.push({
          from: [o.from[0], o.from[1]],
          to: [o.to[0], o.to[1]],
          weight: o.kind === 'arrow' ? 1.5 : 0.8,
        });
      else if (o.kind === 'text')
        marks.push({ from: [o.at[0] - half, o.at[1]], to: [o.at[0] + half, o.at[1]], weight: 1 });
    }
    const side = minimizerLabelSide(xs, domain, marks, ppu);
    const fmt = (v: number) => sig(v, 3);
    return [
      {
        kind: 'point',
        at: xs,
        shape: 'cross',
        radius: 4.5,
        label: starRuns('x', `(${fmt(xs[0])}, ${fmt(xs[1])})`),
        labelSide: side,
      },
    ];
  }, [problem, runs, heads, focus, kkt, geometry, domain, ppu]);
  // The marks of the step, drawn by `drawStep` over the dotted SQP linearizations.
  const marks = useMemo(
    () => [...projections, ...geometry, ...(kkt?.overlays ?? []), ...starLabel],
    [projections, geometry, kkt, starLabel],
  );
  const linearized = useMemo(
    () =>
      focus && !focus.error && focus.sel.id === 'sqp'
        ? linearizedConstraints(focusTrace, gk, problem, domain)
        : [],
    [focus, focusTrace, gk, problem, domain],
  );
  // Under the paths: the dotted linearizations and the step's marks; over them (after the
  // paths, with a halo): every label, so no line strikes one through.
  const bareMarks = useMemo(() => withoutLabels(marks), [marks]);
  const drawStep = useCallback(
    (ctx: CanvasRenderingContext2D, v: View2D) => {
      drawLinearized(ctx, linearized, v.toPx, v.colors.text);
      drawOverlays2D(ctx, overlayView(v), bareMarks);
      // Reserve the boxes of the labels drawn after the paths, so the path layer's off-view
      // notes (drawn before them) keep clear of them.
      for (const { box } of placedLabels(ctx, marks, v)) addLabelRect(ctx, box);
    },
    [linearized, marks, bareMarks],
  );
  const drawLabels = useCallback(
    (ctx: CanvasRenderingContext2D, v: View2D) =>
      drawOverlays2D(
        ctx,
        overlayView(v),
        placedLabels(ctx, marks, v).map((p) => p.label),
      ),
    [marks],
  );
  // The legend's run-wide reserve: which fragments the focused run ever shows, and the widest
  // value of each changing one (step index, arrow scale, merit-function formula).
  const reserve = useMemo(() => {
    const ok = !!focus && !focus.error;
    const last = Math.max(0, focusTrace.length - 1);
    let kktAny = false,
      terms = false,
      scaleCh = 7,
      meritTex = '';
    if (ok)
      for (const s of focusTrace) {
        if (showKkt) {
          const a = kktArrows(s, problem, slot, span);
          if (a) {
            kktAny = true;
            if (a.terms.length > 0) terms = true;
            scaleCh = Math.max(scaleCh, fmtScale(a.scale).length);
          }
        }
        if (landscape === 'merit' && hasMerit) {
          const m = meritLandscape(focus.sel.id, s, problem);
          if (m && m.tex.length > meritTex.length) meritTex = m.tex;
        }
      }
    return {
      last,
      kkt: kktAny,
      terms,
      scaleCh,
      meritTex,
      linearized: ok && focus.sel.id === 'sqp' && last > 0,
    };
  }, [focus, focusTrace, showKkt, problem, slot, span, landscape, hasMerit]);
  const legendRef = useRef<HTMLDivElement>(null);
  useKeepHeight(
    legendRef,
    `${problem.id}|${focus?.sel.id}|${focusTrace.length}|${runs.length}|${landscape}|${showKkt}|${x0.join()}`,
  );
  const hasIneq = problem.constraints.some((c) => c.kind === 'ineq');
  const showLinearized = linearized.length > 0;
  const runError = (r: LabRun) => friendlyError(r, problem);

  // ── Paths ───────────────────────────────────────────────────────────────────────────────
  const overlaps = useMemo(
    () =>
      overlappingPaths(
        runs.map((r) => r.result.trace.map((s) => s.x as [number, number])),
        span * 0.012,
      ),
    [runs, span],
  );
  const paths: PathSpec[] = useMemo(() => {
    const dashed = new Set(overlaps.map(([i]) => i));
    const specs = runs.map((r, i) => ({
      points: r.result.trace.map((s) => s.x as [number, number]),
      color: colors.series[r.sel.slot],
      label: r.method.spec.name,
      muted: focusId !== null && r !== focus,
      dash: dashed.has(i) ? [7, 6] : undefined,
      end: r.result.converged ? ('converged' as const) : ('stopped' as const),
      start: false,
    }));
    const rank = (i: number) => (dashed.has(i) ? 2 : runs[i] === focus ? 1 : 0);
    return specs
      .map((spec, i) => ({ spec, i }))
      .sort((a, b) => rank(a.i) - rank(b.i))
      .map((x) => x.spec);
  }, [runs, colors, focus, focusId, overlaps]);
  const overlapNote =
    overlaps.length > 0
      ? overlaps
          .map(([i, j]) => `${runs[i].method.spec.name} runs along ${runs[j].method.spec.name}`)
          .join('; ') + ' (dashed, on top).'
      : null;

  // ── Convergence ─────────────────────────────────────────────────────────────────────────
  const { series, floor } = useMemo(() => {
    const ok = runs.filter((r) => !r.error);
    const raw = ok.map((r) => {
      // Each run is measured against the minimizer it approaches, not always the global one.
      const fStar =
        problem.extra.minima_f[nearestMinimizer(problem, r.result.trace.at(-1)?.x as Vector)] ??
        problem.extra.f_min;
      return r.result.trace.map((s: Step) =>
        metric === 'kkt'
          ? (s.info.kkt_residual as number)
          : metric === 'viol'
            ? (s.info.violation as number)
            : s.fun === null
              ? null
              : Math.abs(s.fun - fStar),
      );
    });
    // A feasible iterate has violation exactly 0, a gap on a log axis: draw it on a floor one
    // decade below the smallest positive violation, so the curve stays continuous.
    let fl: number | null = null;
    if (metric === 'viol') {
      const pos = raw.flat().filter((v): v is number => typeof v === 'number' && v > 0);
      const lo = pos.length ? Math.min(...pos) : 1e-15;
      fl = 10 ** (Math.floor(Math.log10(Math.max(lo, 1e-300))) - 1);
    }
    return {
      floor: fl,
      series: ok.map((r, i) => ({
        label: r.method.spec.name,
        slot: r.sel.slot,
        end: r.result.converged ? ('converged' as const) : ('stopped' as const),
        count: r.result.nIter,
        values: raw[i].map((v) =>
          typeof v !== 'number' || !Number.isFinite(v)
            ? null
            : v > 0
              ? v
              : fl !== null && v === 0
                ? fl
                : null,
        ),
      })),
    };
  }, [runs, metric, problem]);
  const floorExp = floor !== null ? Math.round(Math.log10(floor)) : 0;
  const yLabel =
    METRIC_LABEL[metric].y +
    (floor !== null ? ` (0, feasible, drawn at 10${superscript(floorExp)})` : '');
  const yName =
    floor !== null
      ? [...METRIC_LABEL[metric].runs, mm(` · 0 at 10${superscript(floorExp)}`)]
      : METRIC_LABEL[metric].runs;
  const [logK, setKAxis] = useKAxis(runs.map((r) => r.result.trace.length));

  // ── KKT table ───────────────────────────────────────────────────────────────────────────
  // References λ⋆, f⋆: the listed minimizer nearest to where the focused run ended (a run can
  // converge to a local minimizer other than the global one).
  const focusEnd = focus?.result.trace.at(-1)?.x as Vector | undefined;
  const refIdx = nearestMinimizer(problem, focusEnd);
  const refX = problem.minima[refIdx] as number[] | undefined;
  const lamStar = problem.extra.multipliers[refIdx];
  const constraintRows: ConstraintRow[] = problem.constraints.map((c, i) => {
    const cv = (focusStep?.info.constraints as number[] | undefined)?.[i] ?? null;
    const lam = (focusStep?.info.multipliers as number[] | undefined)?.[i] ?? null;
    const status = cv === null ? '—' : constraintStatus(c.kind, cv);
    return { i, latex: c.latex, value: cv, lam, lamStar: lamStar?.[i] ?? null, status };
  });

  // ── Interaction ─────────────────────────────────────────────────────────────────────────
  const pickStart = useCallback((p: [number, number]) => setX0(p), [setX0]);
  const minima = problem.minima as [number, number][];
  const start = startStatus(problem, x0);

  const focusMethod = useMemo(() => {
    if (!focus) return undefined;
    const doc = focus.method.doc;
    if (!doc) return focus.method;
    return {
      ...focus.method,
      doc: { ...doc, rule: filledRule(focus.sel.id, doc.rule, focusStep) },
    };
  }, [focus, focusStep]);

  const names = runs.map((r) => r.method.spec.name).join(', ');
  const ariaLabel =
    `Contour plot of ${merit ? 'the merit function of ' + (focus?.method.spec.name ?? '') : problem.name}, ` +
    `infeasible region hatched, with the paths of ${names || 'no method'}.`;

  return (
    <LabShell
      lab={lab}
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
      presets={PRESETS}
      pending={pending}
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={pickerOptions} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection
            title="Methods"
            actions={
              <span className={styles.count}>
                {runs.length} / {MAX_SERIES}
              </span>
            }
          >
            <MethodSlots available={methods} value={selection} onChange={setSelection} />
          </RailSection>
          <RailSection title="Start point">
            <StartPointFields value={x0} onChange={setX0} variable="𝐱₀" />
            <p className={styles.startStatus} data-ok={start.ok || undefined}>
              {start.text}
              {!start.ok &&
                ' Penalty, augmented-Lagrangian and SQP iterates may start anywhere; projection methods project 𝐱₀ onto C first; the log barrier needs every inequality strict at 𝐱₀.'}
            </p>
          </RailSection>
        </>
      }
      stageTitle={
        <>
          <span className={styles.problemName}>{problem.name}</span>
          <RunSummary
            runs={runs.map((r) => ({
              name: r.method.spec.name,
              slot: r.sel.slot,
              result: r.result,
              error: runError(r),
            }))}
          />
        </>
      }
      stageToolbar={
        <div className={styles.toolbar}>
          {hasMerit && (
            <SegmentedControl
              label="Landscape"
              value={landscape}
              onChange={setLandscape}
              options={[
                {
                  value: 'f',
                  label: <Formula tex="f" />,
                  ariaLabel: 'Objective f',
                  tooltip: 'Contours of the objective f',
                },
                {
                  value: 'merit',
                  label: <Formula tex={MERIT_SYMBOL[focus!.sel.id]} />,
                  ariaLabel: 'Merit function of the focused method',
                  tooltip: `Contours of the function ${focus!.method.spec.name} minimizes at this step`,
                },
              ]}
            />
          )}
          <label className={styles.kktToggle} title="KKT arrows: −∇f and Σ λᵢ∇cᵢ at 𝐱ₖ">
            <Toggle checked={showKkt} onChange={setShowKkt} label="KKT arrows" />
            <span aria-hidden="true">KKT</span>
          </label>
        </div>
      }
      stage={
        <div className={styles.stageFill}>
          <div className={styles.plot} ref={plotRef}>
            <Contour2D
              f={field}
              domain={domain}
              cacheKey={fieldKey}
              paths={paths}
              t={player.t}
              ease={!player.reducedMotion}
              minima={minima}
              minimaLabels={false}
              start={[x0[0], x0[1]]}
              overlays={baseOverlays}
              overlay={drawStep}
              overlayAfter={drawLabels}
              onPick={pickStart}
              ariaLabel={ariaLabel}
            />
          </div>
          {/* The key sits under the plot, never over the domain; it keeps one size for the
              run (fragments reserved at their widest, hidden while they do not apply). */}
          <div className={styles.legend} role="note" ref={legendRef}>
            <span>
              Contours of{' '}
              <Fixed widest={merit && reserve.meritTex ? <Formula tex={reserve.meritTex} /> : null}>
                <Formula tex={merit ? merit.tex : 'f(\\mathbf{x})'} />
              </Fixed>
              {!merit &&
                (hasIneq ? (
                  ' · infeasible side hatched'
                ) : (
                  <>
                    {' '}
                    · feasible set: the curve{problem.constraints.length > 1 ? 's' : ''}{' '}
                    <Formula tex="c_i = 0" />
                  </>
                ))}
            </span>
            {reserve.kkt && focus && (
              <span className={styles.hideable} data-off={!kkt || undefined}>
                <span className={styles.inlineSwatch}>
                  <Swatch slot={slot} size={8} />
                </span>
                at{' '}
                <Fixed widest={<Formula tex={`\\mathbf{x}_{${reserve.last}}`} />}>
                  <Formula tex={`\\mathbf{x}_{${gk}}`} />
                </Fixed>
                : <Formula tex="-\nabla f" />
                {reserve.terms && (
                  <span
                    className={styles.hideable}
                    data-off={!(kkt && kkt.terms.length > 0) || undefined}
                  >
                    {' '}
                    vs <Formula tex="\textstyle\sum_i \lambda_i \nabla c_i" />
                  </span>
                )}
                , drawn ×
                <span className={styles.scale} style={{ minWidth: `${reserve.scaleCh}ch` }}>
                  {kkt ? fmtScale(kkt.scale) : ''}
                </span>
              </span>
            )}
            {reserve.linearized && (
              <span className={styles.hideable} data-off={!showLinearized || undefined}>
                <span className={styles.dots} aria-hidden="true" /> linearized constraints{' '}
                <Fixed
                  widest={
                    <Formula
                      tex={`c_i + \\nabla c_i^{\\mathsf T}(\\mathbf{x} - \\mathbf{x}_{${Math.max(0, reserve.last - 1)}}) = 0`}
                    />
                  }
                >
                  <Formula
                    tex={`c_i + \\nabla c_i^{\\mathsf T}(\\mathbf{x} - \\mathbf{x}_{${Math.max(0, gk - 1)}}) = 0`}
                  />
                </Fixed>
              </span>
            )}
            {overlapNote && (
              <span className={styles.wideOnly} title={overlapNote}>
                <span className={styles.dash} aria-hidden="true" /> dashed: path drawn on top of
                another
              </span>
            )}
          </div>
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          height: 264,
          actions: (
            <>
              <KAxisControl logX={logK} onChange={setKAxis} />
              <SegmentedControl
                label="Convergence measure"
                value={metric}
                onChange={setMetric}
                options={[
                  {
                    value: 'kkt',
                    label: 'KKT',
                    ariaLabel: 'KKT residual',
                    tooltip:
                      "Each method's stationarity measure (KKT residual, projection residual or Frank–Wolfe gap)",
                  },
                  {
                    value: 'viol',
                    label: <Formula tex="\|c^+\|" />,
                    ariaLabel: 'Constraint violation',
                    tooltip: VIOLATION_DEF,
                  },
                  {
                    value: 'gap',
                    label: <Formula tex="|f - f^\star|" />,
                    ariaLabel: 'Objective gap',
                    tooltip:
                      '|f(xₖ) − f(x⋆)| with x⋆ the listed minimizer nearest to where each run ends (a run can reach a local minimizer other than the global one).',
                  },
                ]}
              />
            </>
          ),
          content: (
            <div className={styles.chartPad}>
              <ConvergenceChart
                series={series}
                t={player.t}
                logX={logK}
                yLabel={yLabel}
                yName={yName}
                onSeek={(k) => {
                  player.pause();
                  player.seek(k);
                }}
                ariaLabel={`${yLabel} against iteration k for ${names}.`}
              />
            </div>
          ),
        },
        {
          id: 'details',
          label: 'Details',
          grow: true,
          title: (
            <Tabs
              label="Details"
              value={tab}
              onChange={setTab}
              idBase="cdetails"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'kkt', label: 'KKT' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="cdetails" id={tab} focusable={false}>
              {tab === 'method' ? (
                <MethodCard
                  method={focusMethod!}
                  slot={focus.sel.slot}
                  step={focusStep}
                  result={focus.result}
                  error={runError(focus)}
                  call={{ problem: problem.id, options: runOptions, params: focus.sel.params }}
                />
              ) : tab === 'kkt' ? (
                <div className={styles.kktPanel}>
                  <p className={styles.kktLead}>
                    <span className={styles.inlineSwatch}>
                      <Swatch slot={focus.sel.slot} size={8} />
                    </span>{' '}
                    {focus.method.spec.name} at <Formula tex={`\\mathbf{x}_{${gk}}`} />
                    {focusStep && (
                      <>
                        : KKT residual <Formula tex="r_k" /> ={' '}
                        <span className={styles.mono}>
                          {sci(focusStep.info.kkt_residual as number, 2)}
                        </span>
                        , violation <Formula tex="\|c^+\|_\infty" /> ={' '}
                        <span className={styles.mono}>
                          {sci(focusStep.info.violation as number, 2)}
                        </span>
                        {typeof focusStep.info.stationarity === 'number' && (
                          <>
                            , stationarity{' '}
                            <Formula tex="\|\nabla f + \textstyle\sum_i \lambda_i\nabla c_i\|_\infty" />{' '}
                            ={' '}
                            <span className={styles.mono}>
                              {sci(focusStep.info.stationarity, 2)}
                            </span>
                          </>
                        )}
                        .
                      </>
                    )}
                  </p>
                  <TableView
                    columns={CONSTRAINT_COLUMNS}
                    rows={constraintRows}
                    ariaLabel="Constraints, their values and multipliers at the current step"
                  />
                  <p className={styles.kktNote}>
                    Convention: <Formula tex="L = f + \textstyle\sum_i \lambda_i c_i" />,{' '}
                    <Formula tex="\lambda_i \ge 0" /> for <Formula tex="c_i \le 0" />; violation{' '}
                    <Formula tex="\|c^+\|_\infty = \max\big(\max_{E}|c_i|,\ \max_{I}\max(0, c_i)\big)" />
                    ; an inequality is active when <Formula tex="|c_i| \le 10^{-6}" /> (the methods'
                    tolerance).{' '}
                    {refX && (
                      <>
                        <Formula tex="\lambda_i^\star" /> is taken at{' '}
                        <Formula tex={`\\mathbf{x}^\\star = ${texVec(refX, 3)}`} />
                        {problem.minima.length > 1
                          ? ', the listed local minimizer nearest the final iterate. '
                          : '. '}
                      </>
                    )}
                    {focusStep?.info.multipliers
                      ? 'λᵢ is the method’s own estimate at this step.'
                      : 'This method has no multiplier estimate; its residual is the projection residual or the Frank–Wolfe gap.'}
                  </p>
                </div>
              ) : (
                <IterationTable
                  steps={focus.result.trace}
                  columns={tableColumns(focus.sel.id, narrow)}
                  className={styles.iterTable}
                  k={gk}
                  onSelect={(k) => {
                    player.pause();
                    player.seek(k);
                  }}
                  ariaLabel={`${focus.method.spec.name} iterations`}
                />
              )}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}
