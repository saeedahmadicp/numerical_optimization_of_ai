/**
 * Linear & integer programming lab.
 *
 * The stage shows the program's geometry — the feasible polygon (2-D), the polytope (3-D, rotatable)
 * or the iterate as bars (n ≥ 4) — with every method's iterates on one clock, and under it the
 * focused method's working: its tableau with the pivot and the ratio test, its search tree, or its
 * central-path measures, or PDHG's residuals and restarts. The objective vector is draggable on
 * the 2-D plane.
 */
import './setup';
import { useMemo, useState, useSyncExternalStore } from 'react';
import { listMethods, type RegisteredMethod } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { LinearProgram, Step } from '../../core/types';
import { lpLatex, type LPProblem } from '../../problems/lp';
import { codecs, useUrlState } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import {
  Button,
  Formula,
  NumberField,
  PlaybackBar,
  SegmentedControl,
  TabPanel,
  Tabs,
} from '../../ui/components';
import { ConvergenceChart, IterationTable } from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { MAX_SERIES } from '../../ui/colors';
import { sig } from '../../core/format';
import { b as mb, m as mm, sans, sub, sup, v as mv, type MathRun } from '../../viz/mathText';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  useLabRunsState,
  useMethodSelection,
  useProblemState,
  focusRuns,
  type Insight,
  type LabPreset,
  type LabRun,
  type MethodSelection,
} from '../_shell';
import { twoPhaseSimplex } from '../../methods/lp/simplex';
import { branchAndBound } from '../../methods/lp/integer';
import { LPPlane, type PlaneRun } from './LPPlane';
import { LPSolid } from './LPSolid';
import { VectorBars } from './VectorBars';
import { WorkStrip, type WorkRun } from './WorkStrip';
import { ResidualChart, type ResidualRun } from './ResidualChart';
import { PDHG } from './pdhgView';
import { columnsFor } from './columns';
import { liveRule } from './explain';
import styles from './LPLab.module.css';

void preloadKatex();

/** First view: the vertex walk and the interior path on the same polygon, one clock. */
const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'simplex', slot: 0, params: {} },
  { id: 'primal_dual_ipm', slot: 1, params: {} },
];

const PRESETS: LabPreset[] = [
  {
    id: 'klee-minty',
    title: 'Dantzig’s rule visits all 8 corners of the Klee–Minty cube',
    note: 'The most negative reduced cost leads through every vertex (7 pivots); the steepest edge needs one.',
    problem: 'klee_minty_3',
    methods: [
      { id: 'simplex', slot: 0, params: {} },
      { id: 'revised_simplex', slot: 1, params: { pivot_rule: 'steepest_edge' } },
    ],
  },
  {
    id: 'degenerate',
    title: 'A pivot that does not move',
    note: 'Three constraints meet at (2, 0): the ratio test ties, θ = 0, the basis changes and the vertex stays.',
    problem: 'degenerate_2d',
    methods: [{ id: 'simplex', slot: 0, params: {} }],
  },
  {
    id: 'cuts',
    title: 'One Gomory cut against a seven-node tree',
    note: 'Branch and bound splits the box three times; the single cut 3x₁ + 2x₂ ≤ 15 slices the LP vertex off instead.',
    problem: 'ilp_knapsack_like_2d',
    methods: [
      { id: 'branch_and_bound', slot: 0, params: {} },
      { id: 'gomory_cuts', slot: 1, params: {} },
    ],
  },
  {
    id: 'pdhg',
    title: 'PDHG restarts from the average, not the last iterate',
    note: 'Only products with A and Aᵀ: 242 cheap steps and 6 restarts (squares) on Wyndor, against 5 Newton steps for the interior point.',
    problem: 'wyndor',
    methods: [
      { id: 'restarted_pdhg', slot: 0, params: {} },
      { id: 'primal_dual_ipm', slot: 1, params: {} },
    ],
  },
  {
    id: 'cycling',
    title: 'Dantzig’s rule cycles on Beale’s example',
    note: 'Six degenerate pivots return to the starting basis; Bland’s smallest-index rule reaches the optimum.',
    problem: 'beale_cycling',
    methods: [
      { id: 'simplex', slot: 0, params: {} },
      { id: 'revised_simplex', slot: 1, params: { pivot_rule: 'bland' } },
    ],
  },
];

type Metric = 'gap' | 'obj' | 'infeas';
type Tab = 'method' | 'steps';
const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'obj', 'infeas']);
const NO_C: number[] = [];

/** Axis names of the convergence chart, set in math like the measure picker above it. */
/** The convergence x-axis: a simplex pivot or an iteration of the other methods. */
const X_NAME: MathRun[] = [sans('pivot / iteration '), mv('k')];

const Y_NAMES: Record<Metric, MathRun[]> = {
  gap: [mm('|'), mv('z'), sub('k'), mm(' − '), mv('z'), sup('⋆'), mm('|')],
  obj: [mb('c'), sup('⊤'), mb('x'), sub('k')],
  infeas: [mm('‖('), mv('A'), mb('x'), sub('k'), mm(' − '), mb('b'), mm(')'), sub('+'), mm('‖')],
};
const BOUND_GAP_NAME: MathRun[] = [
  mm('|'),
  mv('z̄'),
  sub('k'),
  mm(' − '),
  mv('z'),
  sup('⋆'),
  mm('|'),
];

const MOBILE = '(max-width: 899px)';
const subscribeMedia = (cb: () => void) => {
  const mq = window.matchMedia(MOBILE);
  mq.addEventListener('change', cb);
  return () => mq.removeEventListener('change', cb);
};
const useMobile = () =>
  useSyncExternalStore(
    subscribeMedia,
    () => window.matchMedia(MOBILE).matches,
    () => false,
  );

/** ‖(A𝐱 − 𝐛)₊‖₂ over the inequality rows, equality residuals and 𝐱 ≥ 𝟎. */
function infeasibility(lp: LinearProgram, x: readonly number[]): number {
  let s = 0;
  (lp.aUb ?? []).forEach((a, i) => {
    const r = a.reduce((t, v, j) => t + v * x[j], 0) - (lp.bUb as number[])[i];
    if (r > 0) s += r * r;
  });
  (lp.aEq ?? []).forEach((a, i) => {
    const r = a.reduce((t, v, j) => t + v * x[j], 0) - (lp.bEq as number[])[i];
    s += r * r;
  });
  for (const v of x) if (v < 0) s += v * v;
  return Math.sqrt(s);
}

const isIntegerProgram = (p: LinearProgram) => p.integer.length > 0 && p.integer.some(Boolean);

/** Reference optimum of an LP whose objective the viewer rotated. */
function referenceOptimum(lp: LinearProgram): { x: number[] | null; value: number | null } {
  try {
    const r = isIntegerProgram(lp)
      ? branchAndBound(lp)
      : twoPhaseSimplex(lp, { pivot_rule: 'bland' });
    return r.converged ? { x: r.x as number[], value: r.fun } : { x: null, value: null };
  } catch {
    return { x: null, value: null };
  }
}

export default function LPLab() {
  const lab = getLab('lp')!;
  const mobile = useMobile();
  const problems = useMemo(() => listProblems<LPProblem>('lp'), []);
  const methods = useMemo(() => listMethods('lp'), []);
  const [problem, setProblem] = useProblemState(problems, 'wyndor', ['x0', 'c']);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [cRaw, setC] = useUrlState<number[]>('c', NO_C, codecs.numbers);
  const [metricRaw, setMetric] = useUrlState<Metric>('y', 'gap', METRIC_CODEC);
  const [tab, setTab] = useState<Tab>('method');
  const [focusId, setFocusId] = useState<string | null>(null);

  const n = problem.c.length;
  const cKey = cRaw.length === n && n === 2 ? cRaw.join(',') : '';
  const lp: LinearProgram = useMemo(() => {
    if (!cKey) return problem;
    const c = cKey.split(',').map(Number);
    return { ...problem, id: `${problem.id}~c=${cKey}`, c, optimum: null, optimalValue: null };
  }, [problem, cKey]);
  const reference = useMemo(
    () => (cKey ? referenceOptimum(lp) : { x: problem.optimum, value: problem.optimalValue }),
    [cKey, lp, problem],
  );

  const { runs, pending } = useLabRunsState(lp, selection, {});
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  // The detail follows the chosen method; by default the first one that ran (not one that
  // rejected its input, which has no steps to show).
  const focus: LabRun | undefined =
    runs.find((r) => r.sel.id === focusId) ?? runs.find((r) => !r.error) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = player.localK(focusIndex);
  const focusStep: Step | undefined = focus?.result.trace[focusK];

  const metric: Metric = metricRaw === 'gap' && reference.value === null ? 'obj' : metricRaw;
  const zStar = reference.value;
  const series = useMemo(
    () =>
      runs.map((r) => ({
        label:
          r.sel.id === 'branch_and_bound' && metric === 'gap'
            ? `${r.method.spec.name} · bound gap |z̄ₖ − z⋆|`
            : r.method.spec.name,
        slot: r.sel.slot,
        count: r.result.nIter,
        end: (r.result.converged ? 'converged' : 'stopped') as 'converged' | 'stopped',
        values: r.result.trace.map((s: Step) => {
          // cᵀxₖ is the objective of the step's point (for branch and bound, the node's LP
          // value). The gap of branch and bound is its bound gap |z̄ₖ − z⋆|, the quantity that
          // closes; its legend entry says so.
          if (metric === 'obj') return s.fun;
          if (metric === 'gap') {
            const z =
              r.sel.id === 'branch_and_bound' ? (s.info.best_bound as number | null) : s.fun;
            return z === null || zStar === null ? null : Math.max(Math.abs(z - zStar), 1e-16);
          }
          const x = s.x as number[] | null;
          return x ? Math.max(infeasibility(lp, x), 1e-16) : null;
        }),
      })),
    [runs, metric, zStar, lp],
  );

  const planeRuns: PlaneRun[] = useMemo(
    () =>
      runs.map((r) => ({
        id: r.sel.id,
        slot: r.sel.slot,
        name: r.method.spec.name,
        trace: r.result.trace,
        converged: r.result.converged,
      })),
    [runs],
  );
  const workRun: WorkRun | null = focus
    ? {
        id: focus.sel.id,
        name: focus.method.spec.name,
        slot: focus.sel.slot,
        trace: focus.result.trace,
        params: {
          ...Object.fromEntries(focus.method.spec.params.map((p) => [p.name, p.default])),
          ...focus.sel.params,
        },
        converged: focus.result.converged,
        status: String(focus.result.extra.status ?? ''),
      }
    : null;

  // The MethodCard shows the rule of this step with its numbers.
  const cardMethod: RegisteredMethod | undefined = useMemo(() => {
    if (!focus) return undefined;
    const last = focusK === focus.result.trace.length - 1;
    const rule = liveRule(
      focus.sel.id,
      focusStep,
      focus.result.trace[focusK - 1],
      lp,
      last ? String(focus.result.extra.status ?? '') : null,
    );
    const doc = focus.method.doc;
    return rule && doc ? { ...focus.method, doc: { ...doc, rule } } : focus.method;
  }, [focus, focusStep, focusK, lp]);

  // The picker's formula follows a rotated objective.
  const pickerProblems = useMemo(
    () =>
      cKey
        ? problems.map((p) => (p.id === problem.id ? { ...p, latex: lpLatex(lp) } : p))
        : problems,
    [problems, problem.id, cKey, lp],
  );
  const integer = isIntegerProgram(problem);
  const names = runs.map((r) => r.method.spec.name).join(' and ');
  const optimum = reference.x;
  const domain = problem.domain as [[number, number], [number, number]];

  let figure;
  if (n === 2)
    figure = (
      <LPPlane
        lp={lp}
        domain={domain}
        runs={planeRuns}
        t={player.t}
        localK={player.localK}
        ease={!player.reducedMotion}
        focus={focusIndex}
        optimum={optimum}
        optimalValue={zStar}
        integer={integer}
        onObjectiveChange={(c) => setC(c)}
        ariaLabel={`Feasible set of ${problem.name}, objective level lines, and the iterates of ${names || 'no method'}.`}
      />
    );
  else if (n === 3)
    figure = (
      <LPSolid
        lp={lp}
        runs={planeRuns}
        t={player.t}
        localK={player.localK}
        ease={!player.reducedMotion}
        focus={focusIndex}
        optimum={optimum}
        integer={integer}
        ariaLabel={`Polytope of ${problem.name} with the iterates of ${names || 'no method'}.`}
      />
    );
  else
    figure = (
      <VectorBars
        runs={planeRuns}
        t={player.t}
        ease={!player.reducedMotion}
        optimum={optimum}
        n={n}
      />
    );

  const strip =
    workRun && focusStep ? <WorkStrip run={workRun} k={focusK} lp={lp} compact={mobile} /> : null;

  const convergence: Insight = {
    id: 'convergence',
    title: 'Convergence',
    height: 264,
    actions: (
      <SegmentedControl
        label="Convergence measure"
        value={metric}
        onChange={setMetric}
        options={[
          ...(zStar !== null
            ? [
                {
                  value: 'gap' as const,
                  label: <Formula tex="|z_k - z^\star|" />,
                  ariaLabel: 'Objective gap',
                },
              ]
            : []),
          {
            value: 'obj' as const,
            label: <Formula tex="\mathbf{c}^{\top}\mathbf{x}_k" />,
            ariaLabel: 'Objective value',
          },
          {
            value: 'infeas' as const,
            label: <Formula tex="\|(A\mathbf{x}-\mathbf{b})_+\|" />,
            ariaLabel: 'Infeasibility',
          },
        ]}
      />
    ),
    content: (
      <div className={styles.chartPad}>
        <ConvergenceChart
          series={series}
          t={player.t}
          logY={metric !== 'obj'}
          yLabel={metric === 'gap' ? '|zₖ − z⋆|' : metric === 'obj' ? 'cᵀxₖ' : '‖(Axₖ − b)₊‖'}
          yName={
            metric === 'gap' &&
            runs.length > 0 &&
            runs.every((r) => r.sel.id === 'branch_and_bound')
              ? BOUND_GAP_NAME
              : Y_NAMES[metric]
          }
          legend={
            metric === 'gap' && runs.some((r) => r.sel.id === 'branch_and_bound') ? true : undefined
          }
          xName={X_NAME}
          onSeek={(k) => {
            player.pause();
            player.seek(k);
          }}
          ariaLabel={`Convergence of ${names}`}
        />
      </div>
    ),
  };

  const details: Insight = {
    id: 'details',
    label: 'Details',
    grow: true,
    title: (
      <Tabs
        label="Details"
        value={tab}
        onChange={setTab}
        idBase="lp-details"
        items={[
          { id: 'method', label: 'Method' },
          { id: 'steps', label: 'Iterations' },
        ]}
      />
    ),
    content:
      focus && cardMethod ? (
        <TabPanel idBase="lp-details" id={tab} focusable={false}>
          {tab === 'method' ? (
            <MethodCard
              method={cardMethod}
              slot={focus.sel.slot}
              step={focusStep}
              result={focus.result}
              error={focus.error}
              call={cKey ? undefined : { problem: problem.id, params: focus.sel.params }}
            />
          ) : (
            <IterationTable
              steps={focus.result.trace}
              columns={columnsFor(focus.sel.id)}
              k={focusK}
              onSelect={(k) => {
                player.pause();
                player.seek(k);
              }}
              ariaLabel={`${focus.method.spec.name} iterations`}
            />
          )}
        </TabPanel>
      ) : null,
  };

  // PDHG's residuals: the focused run when it is PDHG, else the PDHG run in the comparison.
  const pdhgRun = focus?.sel.id === PDHG ? focus : runs.find((r) => r.sel.id === PDHG && !r.error);
  const residualRun: ResidualRun | null =
    pdhgRun && !pdhgRun.error
      ? {
          name: pdhgRun.method.spec.name,
          slot: pdhgRun.sel.slot,
          trace: pdhgRun.result.trace,
          converged: pdhgRun.result.converged,
          tol: Number(
            pdhgRun.sel.params.tol ??
              pdhgRun.method.spec.params.find((p) => p.name === 'tol')?.default ??
              1e-8,
          ),
        }
      : null;
  const residuals: Insight | null = residualRun
    ? {
        id: 'residuals',
        title: 'PDHG residuals',
        height: 264,
        content: (
          <div className={styles.chartPad}>
            <ResidualChart
              run={residualRun}
              t={player.t}
              onSeek={(k) => {
                player.pause();
                player.seek(k);
              }}
            />
          </div>
        ),
      }
    : null;
  const charts: Insight[] = residuals ? [convergence, residuals] : [convergence];

  const insights: Insight[] =
    mobile && strip
      ? [
          {
            id: 'work',
            title: 'Step detail',
            wide: true,
            content: <div className={styles.mobileStrip}>{strip}</div>,
          },
          ...charts,
          details,
        ]
      : [...charts, details];

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
            <ProblemPicker problems={pickerProblems} value={problem.id} onChange={setProblem} />
          </RailSection>
          {/* Problem data before Methods: Problem → Objective → Methods (the lab order). */}
          {n === 2 && (
            <RailSection
              title="Objective"
              actions={
                cKey ? (
                  <Button size="sm" variant="ghost" onClick={() => setC(NO_C)}>
                    Reset
                  </Button>
                ) : undefined
              }
            >
              <div className={styles.cFields}>
                <NumberField
                  label="Cost c₁"
                  prefix="c₁"
                  value={lp.c[0]}
                  onChange={(v) => setC([v, lp.c[1]])}
                />
                <NumberField
                  label="Cost c₂"
                  prefix="c₂"
                  value={lp.c[1]}
                  onChange={(v) => setC([lp.c[0], v])}
                />
              </div>
              <p className={styles.hint}>
                {lp.sense === 'max' ? 'Maximize' : 'Minimize'}{' '}
                <Formula tex="\mathbf c^{\top}\mathbf x" />. Drag the arrow{' '}
                <Formula tex={lp.sense === 'max' ? '\\mathbf c' : '-\\mathbf c'} /> on the plot to
                rotate it and watch the optimal vertex move.
              </p>
            </RailSection>
          )}
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
        </>
      }
      stageTitle={
        <>
          <span className={styles.problemName}>{problem.name}</span>
          {zStar !== null && (
            <span className={styles.zstar}>
              <Formula tex={`z^\\star = ${sig(zStar, 6).replace('−', '-')}`} />
            </span>
          )}
          <RunSummary
            runs={runs.map((r) => ({
              name: r.method.spec.name,
              slot: r.sel.slot,
              result: r.result,
              error: r.error,
            }))}
          />
        </>
      }
      stage={
        <div
          className={styles.stageGrid}
          data-bars={
            n > 3 && strip
              ? Array.isArray(focusStep?.info.tableau) || focus?.sel.id === 'branch_and_bound'
                ? 'tableau'
                : 'measures'
              : undefined
          }
        >
          <div className={styles.figure}>{figure}</div>
          {!mobile && strip && <div className={styles.stripWrap}>{strip}</div>}
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={insights}
    />
  );
}
