/**
 * Stochastic gradients lab: SGD, momentum, AdaGrad, RMSProp, Adam, SVRG, SAGA and SAG on
 * finite sums f(𝐰) = (1/n) Σᵢ fᵢ(𝐰) with n = 200 samples and two parameters.
 *
 * Stage: the landscape of the FULL loss in parameter space with every method's path; for the
 * method in focus, the geometry of the step on screen (geometry.ts): the expected step −η∇f, the
 * exact 2σ ellipse of the mini-batch step, the momentum push, the SVRG snapshot, Nesterov's
 * look-ahead. Insights: the data with each model and the focused method's mini-batch (with its
 * residual sticks), loss and learning rate against epochs on one clock, the MethodCard with the
 * update's numbers filled in, and the iteration table.
 *
 * Sampling settings (batch size, epochs, schedule, tolerance, seed) are shared by every method:
 * one stopping test and one sampling stream per comparison.
 */
import './setup';
import { useCallback, useDeferredValue, useEffect, useMemo, useState } from 'react';
import { defaults, listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { ParamSpec, ParamValue, Step } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { int, sci, sig } from '../../core/format';
import {
  Formula,
  NumberField,
  ParamControls,
  PlaybackBar,
  SegmentedControl,
  Swatch,
  TabPanel,
  Tabs,
  Toggle,
} from '../../ui/components';
import {
  Contour2D,
  IterationTable,
  drawMath,
  measureMath,
  mathBold,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  starRuns,
  type Column,
  type Overlay2D,
  type PathSpec,
  type View2D,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { MAX_SERIES } from '../../ui/colors';
import { getLab } from '../index';
import {
  LabShell,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  coerceParam,
  focusRuns,
  tupleCodec,
  useLabRunsState,
  useMethodSelection,
  useProblemState,
  type LabPreset,
  type MethodSelection,
} from '../_shell';
import type { FiniteSumProblem } from '../../problems/stochastic';
import { DataView } from './DataView';
import { TrainingChart, type TrainSeries } from './TrainingChart';
import { ELLIPSE_MASS, ELLIPSE_R, UPDATE_NOUN, dataMode, epochOf, stepGeometry } from './geometry';
import { stepTex } from './equation';
import { stepOverlays } from './overlays';
import { StepLens } from './StepLens';
import { lossEvaluator } from './lensField';
import { StochasticCard } from './StochasticCard';
import { ScrollFade } from './ScrollFade';
import styles from './StochasticLab.module.css';
import { SeedControl } from '../_shell/SeedControl';
import { StartPointHint } from '../_shell/blocks';
import { useStartOnField } from '../unconstrained/useStartOnField';
import { addLabelRect, hitsLabel, type LabelRect } from '../../viz/labelRects';

type Box = { x0: number; y0: number; x1: number; y1: number };

/** True when the segment a–b passes through the box (Liang–Barsky clipping). */
function segmentHitsBox(a: [number, number], b: [number, number], r: Box): boolean {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  let t0 = 0,
    t1 = 1;
  for (const [p, q] of [
    [-dx, a[0] - r.x0],
    [dx, r.x1 - a[0]],
    [-dy, a[1] - r.y0],
    [dy, r.y1 - a[1]],
  ] as const) {
    if (p === 0) {
      if (q < 0) return false;
      continue;
    }
    const t = q / p;
    if (p < 0) t0 = Math.max(t0, t);
    else t1 = Math.min(t1, t);
    if (t0 > t1) return false;
  }
  return true;
}

/** How many path segments (screen px) pass through the box. */
function crossings(polys: readonly (readonly [number, number][])[], r: Box): number {
  let n = 0;
  for (const p of polys)
    for (let i = 0; i + 1 < p.length; i++) if (segmentHitsBox(p[i], p[i + 1], r)) n++;
  return n;
}

void preloadKatex();

/** Shared sampling settings live in the rail, not in each method's parameters. */
const SHARED = ['batch_size', 'epochs', 'lr_schedule', 'gtol', 'record_every'] as const;

/**
 * The first view: on linear regression with mini-batches of 5 and the same η = 0.1, SGD settles
 * into its noise ball (f − f⋆ ≈ 10⁻³) while SVRG and SAGA converge linearly to ‖∇f‖ ≤ 10⁻⁶.
 */
const DEFAULT: MethodSelection[] = [
  { id: 'sgd', slot: 0, params: { lr: 0.1 } },
  { id: 'svrg', slot: 1, params: { lr: 0.1 } },
  { id: 'saga', slot: 2, params: { lr: 0.1 } },
];
const DEFAULT_SHARED = { b: 5, ep: 20, sch: 'constant', tol: 1e-6, s: 0 } as const;

const PRESETS: LabPreset[] = [
  {
    id: 'variance',
    title: 'Variance reduction beats the noise floor',
    note: 'Same η = 0.1, b = 5: SGD wanders at an O(η) loss floor; SVRG and SAGA converge linearly.',
    problem: 'linreg_2d',
    methods: DEFAULT,
    extra: { b: '5', ep: '20', sch: 'constant', tol: '1e-6', s: '0' },
  },
  {
    id: 'adaptive',
    title: 'Adam and RMSProp round a κ ≈ 1,070 valley',
    note: 'Plain SGD needs η < 2/L ≈ 0.002 and crawls; per-coordinate scaling equalizes the axes.',
    problem: 'ill_conditioned_ls',
    methods: [
      { id: 'sgd', slot: 0, params: { lr: 0.002 } },
      { id: 'stochastic_rmsprop', slot: 1, params: {} },
      { id: 'stochastic_adam', slot: 2, params: {} },
    ],
    extra: { b: '10', ep: '20', sch: 'constant', tol: '1e-6', s: '0' },
  },
  {
    id: 'bias',
    title: 'SAG needs a smaller step than SAGA',
    note: 'At η = 0.1 with reshuffled epochs, SAG’s stale average oscillates (f spikes every few epochs); SAGA, unbiased, converges linearly. Try η = 0.02 for SAG.',
    problem: 'linreg_2d',
    methods: [
      { id: 'saga', slot: 0, params: { lr: 0.1 } },
      { id: 'sag', slot: 1, params: { lr: 0.1 } },
    ],
    extra: { b: '5', ep: '20', sch: 'constant', tol: '1e-6', s: '0' },
  },
  {
    id: 'logistic',
    title: 'A decision boundary settles',
    note: 'Logistic loss, b = 10, 30 epochs: watch σ(w₀ + w₁x) and x = −w₀/w₁ in the data panel.',
    problem: 'logreg_2d',
    methods: [
      { id: 'sgd', slot: 0, params: { lr: 0.5 } },
      { id: 'svrg', slot: 1, params: { lr: 0.5 } },
      { id: 'stochastic_adam', slot: 2, params: { lr: 0.1 } },
    ],
    extra: { b: '10', ep: '30', sch: 'constant', tol: '1e-6', s: '0' },
  },
];

/**
 * The rail's formula for each problem, broken over lines so it fits the rail at full size (the
 * registry's one-line `latex` is the Python metadata and stays as it is).
 */
const RAIL_TEX: Record<string, string> = {
  linreg_2d: String.raw`\begin{aligned} f(\mathbf{w}) &= \frac{1}{2n}\sum_{i=1}^{n} r_i^2 \\ r_i &= w_0 + w_1 x_i - y_i \end{aligned}`,
  logreg_2d: String.raw`\begin{aligned} f(\mathbf{w}) &= \frac{1}{n}\sum_{i=1}^{n} \big[\log(1 + e^{z_i}) - y_i z_i\big] \\ &\quad + \frac{\lambda}{2}\|\mathbf{w}\|^2,\quad \lambda = 10^{-2} \\ z_i &= w_0 + w_1 x_i \end{aligned}`,
  ill_conditioned_ls: String.raw`\begin{aligned} f(\mathbf{w}) &= \frac{1}{2n}\sum_{i=1}^{n} r_i^2 \\ r_i &= w_0 u_i + 30\,w_1 v_i - y_i \end{aligned}`,
  huber_regression_2d: String.raw`\begin{aligned} f(\mathbf{w}) &= \frac{1}{n}\sum_{i=1}^{n} h_\delta(r_i) \\ r_i &= w_0 + w_1 x_i - y_i,\quad \delta = 1 \\ h_\delta(r) &= \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases} \end{aligned}`,
};

type Metric = 'gap' | 'grad';
const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'grad']);
const SCHEDULE_CODEC = codecs.oneOf(['constant', 'step', 'inv_sqrt', 'cosine']);
const X0_CODEC = tupleCodec(2);

const ordinal = (n: number) =>
  n === 1
    ? 'every update'
    : `every ${n}${n % 10 === 2 && n % 100 !== 12 ? 'nd' : n % 10 === 3 && n % 100 !== 13 ? 'rd' : 'th'} update`;

/** Iteration table: k is the update count, plus the epoch (and ηₖ when it changes). */
function tableColumns(U: number, withEta: boolean): Column[] {
  const tex = (t: string) => <Formula tex={t} />;
  const cols: Column[] = [
    { key: 'k', label: tex('k'), value: (s: Step) => int(s.k), align: 'right', width: '38px' },
    {
      key: 'epoch',
      label: 'epoch',
      value: (s: Step) => sig(epochOf(s.k, U), 3),
      align: 'right',
      width: '40px',
    },
    {
      key: 'w',
      label: tex('\\mathbf{w}_k'),
      value: (s: Step) => {
        const x = s.x as number[];
        return `(${sig(x[0], 4)}, ${sig(x[1], 4)})`;
      },
      width: 'minmax(108px, 1.6fr)',
    },
    {
      key: 'fun',
      label: tex('f(\\mathbf{w}_k)'),
      value: (s: Step) => sig(s.fun, 5),
      align: 'right',
      width: 'minmax(58px, 1fr)',
    },
    {
      key: 'grad',
      label: tex('\\|\\nabla f\\|'),
      value: (s: Step) => (s.gradNorm === null ? '—' : sci(s.gradNorm, 2)),
      align: 'right',
      width: 'minmax(56px, 1fr)',
    },
  ];
  if (withEta)
    cols.push({
      key: 'eta',
      label: tex('\\eta_k'),
      value: (s: Step) => (s.stepSize === null ? '—' : sig(s.stepSize, 3)),
      align: 'right',
      width: 'minmax(48px, 0.7fr)',
    });
  return cols;
}

export default function StochasticLab() {
  const lab = getLab('stochastic')!;
  const colors = useChartColors();
  const problems = useMemo(() => listProblems<FiniteSumProblem>('stochastic'), []);
  const pickerProblems = useMemo(
    () =>
      problems.map((p) => ({
        id: p.id,
        name: p.name,
        latex: RAIL_TEX[p.id] ?? p.latex,
        description: p.description,
        tags: p.tags,
      })),
    [problems],
  );
  const methods = useMemo(() => listMethods('stochastic'), []);
  const [problem, setProblem] = useProblemState(problems, 'linreg_2d');
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT);
  const [x0, setX0] = useUrlState<number[]>('x0', problem.x0, X0_CODEC);

  // Shared sampling settings (URL keys b, ep, sch, tol, s).
  const sharedSpecs = useMemo(
    () => (methods[0]?.spec.params ?? []).filter((p) => SHARED.includes(p.name as never)),
    [methods],
  );
  const spec = (name: string) => sharedSpecs.find((p) => p.name === name) as ParamSpec;
  const [bRaw, setB] = useUrlState('b', DEFAULT_SHARED.b as number, codecs.number);
  const [epRaw, setEp] = useUrlState('ep', DEFAULT_SHARED.ep as number, codecs.number);
  const [schedule, setSchedule] = useUrlState('sch', DEFAULT_SHARED.sch as string, SCHEDULE_CODEC);
  const [tolRaw, setTol] = useUrlState('tol', DEFAULT_SHARED.tol as number, codecs.number);
  const [seedRaw, setSeed] = useUrlState('s', DEFAULT_SHARED.s as number, codecs.number);
  const batchSize = sharedSpecs.length
    ? (coerceParam(spec('batch_size'), bRaw) as number)
    : DEFAULT_SHARED.b;
  const epochs = sharedSpecs.length
    ? (coerceParam(spec('epochs'), epRaw) as number)
    : DEFAULT_SHARED.ep;
  const gtol = sharedSpecs.length
    ? (coerceParam(spec('gtol'), tolRaw) as number)
    : DEFAULT_SHARED.tol;
  const seed = Math.max(0, Math.round(seedRaw)) || 0;
  const shared = useMemo(
    () => ({ batch_size: batchSize, epochs, lr_schedule: schedule, gtol }),
    [batchSize, epochs, schedule, gtol],
  );
  const setShared = (name: string, v: ParamValue) => {
    if (name === 'batch_size') setB(Number(v));
    else if (name === 'epochs') setEp(Number(v));
    else if (name === 'lr_schedule') setSchedule(String(v));
    else if (name === 'gtol') setTol(Number(v));
  };

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]], seed, ...shared }), [x0, seed, shared]);
  const { runs, pending } = useLabRunsState(problem, selection, runOptions, { defer: true });
  // The runs come from a deferred render: draw them with the problem and the settings they were
  // computed for (useDeferredValue values in one component update together).
  const shownProblem = useDeferredValue(problem);
  const shownB = useDeferredValue(batchSize);
  const shownEpochs = useDeferredValue(epochs);
  const shownSchedule = useDeferredValue(schedule);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  // Playback starts once the landscape is on screen (useStartOnField.ts), not at mount.
  const player = useTracePlayer(traces, { autoplay: false });
  const onFieldReady = useStartOnField(player, traces, `stochastic:${shownProblem.id}`);
  usePlayerKeyboard(player);
  // Playback updates every frame and would restart the deferred recompute forever: hold the
  // playhead while new runs are computed (the player restarts on the new traces).
  const { pause, playing } = player;
  useEffect(() => {
    if (pending && playing) pause();
  }, [pending, playing, pause]);

  const [focusId, setFocusId] = useState<string | null>(null);
  const [metric, setMetric] = useUrlState<Metric>('y', 'gap', METRIC_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [showGeometry, setShowGeometry] = useState(true);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusStep: Step | undefined = focus?.result.trace[player.localK(focusIndex)];

  const n = shownProblem.nSamples;
  const b = Math.min(shownB, n);
  const U = Math.ceil(n / b);
  const T = shownEpochs * U;
  const every = (focus?.result.extra.record_every as number | undefined) ?? 1;
  const fStar = shownProblem.extra.f_min;

  // ── Landscape ──────────────────────────────────────────────────────────────────────
  // Drawing only: the fast evaluator (lensField.ts) equals problem.f to ~1e-12 but costs no
  // allocation per point (Contour2D samples ~75,000 points on the main thread per view).
  const f2 = useMemo(() => lossEvaluator(shownProblem).f, [shownProblem]);
  const paths: PathSpec[] = useMemo(() => {
    const specs = runs.map((r) => ({
      points: r.result.trace.map((s) => s.x as [number, number]),
      color: colors.series[r.sel.slot],
      label: r.method.spec.name,
      muted: focusId !== null && r !== focus,
      start: false,
      milestones: every === 1 ? undefined : false,
      end: r.result.converged ? ('converged' as const) : ('stopped' as const),
    }));
    // The focused path is drawn last (on top).
    const order = runs
      .map((_, i) => i)
      .sort((a, c) => Number(runs[a] === focus) - Number(runs[c] === focus));
    return order.map((i) => specs[i]);
  }, [runs, colors, focus, focusId, every]);

  const geometry = useMemo(
    () => (focus && focusStep ? stepGeometry(shownProblem, focus.sel.id, focusStep, shownB) : null),
    [shownProblem, focus, focusStep, shownB],
  );

  const span = Math.max(
    shownProblem.domain[0][1] - shownProblem.domain[0][0],
    shownProblem.domain[1][1] - shownProblem.domain[1][0],
  );
  const overlays: Overlay2D[] = useMemo(
    () =>
      geometry && focus && showGeometry
        ? stepOverlays(geometry, focus.sel.slot, {
            labels: false,
            chain: false,
            minLength: span * 0.008,
          })
        : [],
    [geometry, focus, showGeometry, span],
  );

  // Axis names, 𝐰⋆ and 𝐰₀ in this lab's notation (Contour2D's defaults say 𝐱).
  const lensOn = showGeometry && geometry !== null;
  const narrow = useNarrow();
  const wStar = shownProblem.minima[0];
  const drawLabels = useCallback(
    (ctx: CanvasRenderingContext2D, v: View2D) => {
      const c = v.colors;
      const o = { size: 14, color: c.text2, halo: c.halo };
      // Above the inset tick row (so it never runs into the last x tick) and left of the zoom
      // buttons.
      drawMath(ctx, [mathVar('w'), mathSub('0')], v.width - 46, v.height - 26, {
        ...o,
        align: 'right',
      });
      // Right of the inset tick column (its labels are ≤ 4 characters of 10 px mono).
      drawMath(ctx, [mathVar('w'), mathSub('1')], 44, 22, o);
      const [sx, sy] = v.toPx(x0[0], x0[1]);
      ctx.save();
      ctx.lineWidth = 4;
      ctx.strokeStyle = c.halo;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = c.text;
      ctx.stroke();
      ctx.restore();
      drawMath(ctx, [mathBold('w'), mathSub('0')], sx - 9, sy - 8, {
        size: 12,
        align: 'right',
        color: c.text2,
        halo: c.halo,
      });
    },
    [x0],
  );
  // 𝐰⋆'s label, drawn after the paths (Contour2D's overlayAfter) with a 3 px halo so no path
  // covers it. Of the spots around the cross that stay inside the plot and clear of the step
  // lens, it takes the one that crosses the fewest path segments and no label already drawn
  // (path labels are in the canvas's labelRects registry). The whole paths count, not only the
  // part played so far, so the label does not hop while a run plays.
  const drawStarLabel = useCallback(
    (ctx: CanvasRenderingContext2D, v: View2D) => {
      const c = v.colors;
      const [mx, my] = v.toPx(wStar[0], wStar[1]);
      const runs = starRuns('w', `(${sig(wStar[0], 3)}, ${sig(wStar[1], 3)})`);
      const tw = measureMath(ctx, runs, 12);
      // The lens box (CSS: 196 px, 10 px from the right; phones 112 px, 6 px) with a 4-px margin.
      const lens = lensOn
        ? (() => {
            const size = narrow ? 112 : 196;
            const right = narrow ? 6 : 10;
            return { x0: v.width - right - size - 4, y0: 0, x1: v.width, y1: LENS_TOP + size + 4 };
          })()
        : null;
      const polys = paths.map((p) => p.points.map((q) => v.toPx(q[0], q[1])));
      const spots: [number, number, 'left' | 'right' | 'center'][] = [
        [mx + 10, my + 18, 'left'],
        [mx - 10, my + 18, 'right'],
        [mx + 10, my - 10, 'left'],
        [mx - 10, my - 10, 'right'],
        [mx, my - 14, 'center'],
        [mx, my + 24, 'center'],
      ];
      let best: {
        x: number;
        y: number;
        align: 'left' | 'right' | 'center';
        rect: LabelRect;
      } | null = null;
      let bestScore = Infinity;
      for (const [tx, ty, align] of spots) {
        const left = align === 'left' ? tx : align === 'right' ? tx - tw : tx - tw / 2;
        const box = { x0: left - 2, y0: ty - 13, x1: left + tw + 2, y1: ty + 4 };
        if (box.x0 < 4 || box.x1 > v.width - 4 || box.y0 < 4 || box.y1 > v.height - 4) continue;
        if (lens && box.x0 < lens.x1 && lens.x0 < box.x1 && box.y0 < lens.y1 && lens.y0 < box.y1)
          continue;
        const rect = { x: box.x0, y: box.y0, w: box.x1 - box.x0, h: box.y1 - box.y0 };
        // A label overlap outweighs any number of crossed path segments.
        const score = (hitsLabel(ctx, rect) ? 1e6 : 0) + crossings(polys, box);
        if (score < bestScore) {
          bestScore = score;
          best = { x: tx, y: ty, align, rect };
        }
      }
      if (!best) return;
      drawMath(ctx, runs, best.x, best.y, {
        size: 12,
        align: best.align,
        color: c.text2,
        halo: c.halo,
      });
      addLabelRect(ctx, best.rect);
    },
    [wStar, lensOn, narrow, paths],
  );

  // ── Training curves ────────────────────────────────────────────────────────────────
  const series: TrainSeries[] = useMemo(
    () =>
      runs.map((r) => ({
        // The legend shares a 380 px column with the chart: the registry's short name where it
        // has one, with the full name as its title.
        label: r.method.spec.shortName ?? r.method.spec.name,
        title: r.method.spec.shortName ? r.method.spec.name : undefined,
        count: r.result.nIter,
        slot: r.sel.slot,
        epochs: r.result.trace.map((s) => epochOf(s.k, U)),
        values: r.result.trace.map((s) =>
          metric === 'grad' ? s.gradNorm : s.fun === null ? null : Math.max(s.fun - fStar, 1e-16),
        ),
        lr: r.result.trace.map((s) => s.stepSize),
        end: r.result.converged ? 'converged' : 'stopped',
        muted: focusId !== null && r !== focus,
      })),
    [runs, U, metric, fStar, focus, focusId],
  );
  const effective = useMemo(() => {
    if (
      !focus ||
      !['stochastic_adagrad', 'stochastic_rmsprop', 'stochastic_adam'].includes(focus.sel.id)
    )
      return null;
    return {
      slot: focus.sel.slot,
      values: focus.result.trace.map((s) =>
        Array.isArray(s.info.scaled_lr) ? (s.info.scaled_lr as number[]) : null,
      ),
    };
  }, [focus]);

  // ── Data panel ─────────────────────────────────────────────────────────────────────
  const fits = runs.map((r) => {
    const s = r.result.trace[player.localK(runs.indexOf(r))];
    const w =
      s && Array.isArray(s.x) && (s.x as number[]).every(Number.isFinite)
        ? (s.x as [number, number])
        : null;
    return {
      label: r.method.spec.name,
      slot: r.sel.slot,
      w,
      muted: focusId !== null && r !== focus,
    };
  });
  const focusFit = focus ? fits[focusIndex] : null;
  const batch = geometry?.batch ?? null;
  const mode = dataMode(shownProblem);

  // ── Status line under the landscape ────────────────────────────────────────────────
  const eq = focus && focusStep ? stepTex(focus.sel.id, focusStep) : null;
  /** The estimator's name in the rule: 𝐠 (mini-batch gradient) or 𝐯 (variance-reduced). */
  const estimator =
    geometry && (geometry.kind === 'svrg' || geometry.kind === 'table')
      ? '\\mathbf{v}'
      : '\\mathbf{g}';
  const k = focusStep?.k ?? 0;

  const showLr = shownSchedule !== 'constant' || effective !== null;
  const columns = useMemo(() => tableColumns(U, shownSchedule !== 'constant'), [U, shownSchedule]);
  const sharedValues = { ...defaults({ params: sharedSpecs }), ...shared };

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      pending={pending}
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={pickerProblems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection
            title="Methods"
            actions={
              <span className={styles.count}>
                {runs.length} / {MAX_SERIES}
              </span>
            }
          >
            <MethodSlots
              available={methods}
              value={selection}
              onChange={setSelection}
              hiddenParams={SHARED}
            />
          </RailSection>
          <SeedControl value={seed} onChange={setSeed} />
          <RailSection title="Sampling (every method)">
            <ParamControls
              specs={sharedSpecs.filter((p) => p.name !== 'record_every')}
              values={sharedValues}
              onChange={setShared}
              labelPrefix="Sampling"
            />
            <p className={styles.hint}>
              Each epoch draws one permutation of the data, then {int(U)} mini-batches of {b} (
              {int(T)} updates). Converged means{' '}
              <Formula tex={`\\|\\nabla f\\| \\le ${texSci(gtol)}`} /> after an epoch.
            </p>
          </RailSection>
          <RailSection title="Start point">
            <div className={styles.xy}>
              <NumberField
                label="Start w₀"
                prefix="w₀"
                value={x0[0]}
                onChange={(v) => setX0([v, x0[1]])}
              />
              <NumberField
                label="Start w₁"
                prefix="w₁"
                value={x0[1]}
                onChange={(v) => setX0([x0[0], v])}
              />
            </div>
            <p className={styles.hint}>
              <StartPointHint variable="𝐰₀" />
            </p>
          </RailSection>
        </>
      }
      stageTitle={
        <>
          <span className={styles.problemName}>{shownProblem.name}</span>
          <RunSummary
            runs={runs.map((r) => ({
              name: r.method.spec.name,
              slot: r.sel.slot,
              result: r.result,
              error: r.error,
            }))}
            iterationNoun={UPDATE_NOUN}
          />
        </>
      }
      stageToolbar={
        <Toggle checked={showGeometry} onChange={setShowGeometry} label="Step geometry" showLabel />
      }
      stage={
        <div className={styles.stageGrid}>
          <div className={styles.plot}>
            <Contour2D
              f={f2}
              domain={shownProblem.domain}
              cacheKey={`stochastic:${shownProblem.id}`}
              onFieldReady={onFieldReady}
              fMin={fStar}
              paths={paths}
              t={player.t}
              ease={!player.reducedMotion}
              minima={[wStar]}
              minimaLabels={false}
              overlays={overlays}
              overlay={drawLabels}
              overlayAfter={drawStarLabel}
              offViewLabels={false}
              showReadout={false}
              axisLabels={null}
              onPick={(p) => setX0([Number(p[0].toPrecision(4)), Number(p[1].toPrecision(4))])}
              ariaLabel={`Contour plot of the full loss of ${shownProblem.name} in the parameters w₀, w₁, with the paths of ${runs.map((r) => r.method.spec.name).join(', ')}.`}
            />
            {showGeometry && geometry && focus && (
              <StepLens
                problem={shownProblem}
                geometry={geometry}
                slot={focus.sel.slot}
                k={geometry ? (focusStep?.k ?? 0) : 0}
              />
            )}
          </div>
          {focus && (
            <div className={styles.caption}>
              <div className={styles.captionHead}>
                <Swatch slot={focus.sel.slot} size={8} />
                {/* Phones: the registry's short name where it has one (the full name as title). */}
                <span className={styles.captionName} title={focus.method.spec.name}>
                  <span className={focus.method.spec.shortName ? styles.longName : undefined}>
                    {focus.method.spec.name}
                  </span>
                  {focus.method.spec.shortName && (
                    <span className={styles.shortName}>{focus.method.spec.shortName}</span>
                  )}
                </span>
                {/* The counts keep the width of their largest value, so the line never slides. */}
                <span className={styles.captionMeta}>
                  update{' '}
                  <span
                    className={`${styles.mono} ${styles.metaCount}`}
                    style={{ minWidth: `${int(focus.result.nIter).length}ch` }}
                  >
                    {int(k)}
                  </span>{' '}
                  of {int(focus.result.nIter)} · epoch{' '}
                  <span
                    className={`${styles.mono} ${styles.metaEpoch}`}
                    style={{ minWidth: `${sig(shownEpochs, 3).length + 3}ch` }}
                  >
                    {sig(epochOf(k, U), 3)}
                  </span>
                  {every > 1 && (
                    <span className={styles.captionExtra}> · path through {ordinal(every)}</span>
                  )}
                </span>
              </div>
              {eq ? (
                <ScrollFade className={styles.equation}>
                  <Formula tex={eq} display fit />
                </ScrollFade>
              ) : (
                <p className={`${styles.captionNote} ${styles.equationSlot}`}>
                  <Formula tex="\mathbf{w}_0" /> — press Play, or step with →.
                </p>
              )}
              {showGeometry && geometry && (
                <>
                  <p className={styles.captionNote}>
                    {geometry.cov ? (
                      <>
                        Ellipse: {ELLIPSE_R}σ of the exact covariance of the mini-batch step over
                        all <Formula tex={`\\binom{${n}}{${batch?.length ?? b}}`} /> batches (≈{' '}
                        {Math.round(ELLIPSE_MASS * 100)} % of the steps if the step were
                        Gaussian).{' '}
                      </>
                    ) : null}
                    {geometry.mean ? (
                      focus.sel.id === 'sag' ? (
                        <>
                          Dashed: the full-gradient step <Formula tex="-\eta\nabla f" /> (SAG’s
                          direction is biased, so this is not its mean).{' '}
                        </>
                      ) : (
                        <>
                          Dashed: the expected step <Formula tex="-\eta\nabla f" />.{' '}
                        </>
                      )
                    ) : (
                      <>
                        Dashed: <Formula tex="-\nabla f" /> at the step’s length; the per-coordinate
                        scaling turns the step away from it.{' '}
                      </>
                    )}
                    {geometry.noise !== null && (
                      <>
                        Noise <Formula tex={`\\|${estimator}-\\nabla f\\|`} /> ={' '}
                        <span className={styles.mono}>{sig(geometry.noise, 3)}</span>.
                      </>
                    )}
                  </p>
                  <p className={styles.legendNarrow}>
                    {geometry.cov && <>ellipse: {ELLIPSE_R}σ of the step · </>}
                    dashed: <Formula tex={geometry.mean ? '-\\eta\\nabla f' : '-\\nabla f'} />
                    {!geometry.mean && ' direction'}
                    {focus.sel.id === 'sag' && ' (not SAG’s mean)'}
                    {geometry.noise !== null && (
                      <>
                        {' '}
                        · noise <span className={styles.mono}>{sig(geometry.noise, 2)}</span>
                      </>
                    )}
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      }
      playback={
        <PlaybackBar
          player={player}
          slots={runs.map((r) => r.sel.slot)}
          readout={{ label: 'update', value: k, total: focus?.result.nIter ?? 0 }}
        />
      }
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
      insights={[
        {
          id: 'data',
          title:
            mode === 'logistic'
              ? 'Data and classifier'
              : mode === 'predicted'
                ? 'Observed against predicted'
                : 'Data and fit',
          label: 'Data',
          height: 248,
          actions: focus ? (
            <span className={styles.batchNote}>
              <Swatch slot={focus.sel.slot} size={8} />
              {/* Every wording typeset in one cell, the current one visible: one width. */}
              <span className={styles.batchText}>
                {(['start', 'batch', 'unrecorded'] as const).map((v) => {
                  const now = k === 0 ? 'start' : batch ? 'batch' : 'unrecorded';
                  return (
                    <span
                      key={v}
                      data-ghost={v !== now || undefined}
                      aria-hidden={v !== now || undefined}
                    >
                      {v === 'start' ? (
                        <>mini-batches of b = {b}</>
                      ) : v === 'batch' ? (
                        <>
                          mini-batch <Formula tex="B_k" />: {v === now ? batch!.length : b} samples
                        </>
                      ) : (
                        <>b = {b}: indices not recorded</>
                      )}
                    </span>
                  );
                })}
              </span>
            </span>
          ) : undefined,
          content: (
            <DataView
              problem={shownProblem}
              fits={fits}
              focus={
                focus && focusFit
                  ? {
                      label: focus.method.spec.name,
                      slot: focus.sel.slot,
                      w: focusFit.w,
                      batch,
                    }
                  : null
              }
            />
          ),
        },
        {
          id: 'training',
          title: 'Training',
          height: 262,
          actions: (
            <SegmentedControl
              label="Error measure"
              value={metric}
              onChange={setMetric}
              options={[
                { value: 'gap', label: <Formula tex="f - f^\star" />, ariaLabel: 'Objective gap' },
                {
                  value: 'grad',
                  label: <Formula tex="\|\nabla f\|" />,
                  ariaLabel: 'Gradient norm',
                },
              ]}
            />
          ),
          content: (
            <TrainingChart
              series={series}
              t={player.t}
              totalEpochs={shownEpochs}
              effective={effective}
              showLr={showLr}
              yName={
                metric === 'grad'
                  ? [
                      mathMain('‖∇'),
                      mathVar('f'),
                      mathMain('('),
                      mathBold('w'),
                      mathSub('k', 'italic'),
                      mathMain(')‖'),
                    ]
                  : [
                      mathVar('f'),
                      mathMain('('),
                      mathBold('w'),
                      mathSub('k', 'italic'),
                      mathMain(') − '),
                      mathVar('f'),
                      mathSup('⋆'),
                    ]
              }
              onSeek={(i) => {
                player.pause();
                player.seek(i);
              }}
              ariaLabel={`${metric === 'grad' ? 'Full-gradient norm' : 'Loss gap f − f⋆'}${showLr ? ' and learning rate' : ''} against epochs: ${runs
                .map((r) => {
                  const last = r.result.trace[r.result.trace.length - 1];
                  return `${r.method.spec.name} ends at ${last ? sci(metric === 'grad' ? last.gradNorm : (last.fun ?? NaN) - fStar, 2) : '—'}`;
                })
                .join('; ')}.`}
            />
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
              idBase="sgd-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="sgd-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <StochasticCard
                  method={focus.method}
                  slot={focus.sel.slot}
                  step={focusStep}
                  result={focus.result}
                  error={focus.error}
                  call={{
                    problem: shownProblem.id,
                    options: { x0: [x0[0], x0[1]], seed: seed || undefined },
                    params: { ...focus.sel.params, ...shared },
                  }}
                />
              ) : (
                <IterationTable
                  steps={focus.result.trace}
                  columns={columns}
                  k={player.localK(focusIndex)}
                  onSelect={(i) => {
                    player.pause();
                    player.seek(i);
                  }}
                  ariaLabel={`${focus.method.spec.name} updates`}
                />
              )}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}

/** The lens's top edge under the view toolbar (StochasticLab.module.css `.lens`). */
const LENS_TOP = 50;

/** The lens's CSS breakpoint (StochasticLab.module.css: 112 px below 900 px, else 196 px). */
function useNarrow(): boolean {
  const q = '(max-width: 899px)';
  const [narrow, setNarrow] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(q).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return narrow;
}

/** 10⁻⁶ as TeX. */
function texSci(v: number): string {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(v));
  const m = v / 10 ** e;
  return Math.abs(m - 1) < 1e-9 ? `10^{${e}}` : `${Number(m.toPrecision(3))}\\times10^{${e}}`;
}
