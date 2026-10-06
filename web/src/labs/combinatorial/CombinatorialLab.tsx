/**
 * Combinatorial lab — the 0-1 knapsack and the Euclidean TSP on the shared LabShell.
 *
 * TSP: one panel per method on the same cities (2-opt / Or-opt edge exchanges, the SA proposal
 * and temperature, GA best tours as small multiples, Ant System pheromone as edge opacity,
 * nearest-neighbor and Held–Karp partial paths); drag a city to edit the instance.
 * Knapsack: the items profile whose area left of C is the LP bound, a knapsack per method, the
 * DP table filling cell by cell with its backtrack path, the branch-and-bound tree; drag C.
 *
 * URL: p (problem), m (methods), s (seed), c (moved cities), C (capacity).
 */
import './setup';
import { useMemo, useState, useSyncExternalStore } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import { codecs, readQuery, useUrlState, type Codec } from '../../app/useUrlState';
import { replaceQuery } from '../../app/router';
import { preloadKatex } from '../../ui/katex';
import {
  Button,
  Formula,
  Icon,
  NumberField,
  PlaybackBar,
  SegmentedControl,
  Select,
  TabPanel,
  Tabs,
  type SelectOption,
} from '../../ui/components';
import {
  ConvergenceChart,
  TableView,
  mathSans,
  mathVar,
  mathSup,
  mathSub,
  mathMain,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { MAX_SERIES } from '../../ui/colors';
import { sig } from '../../core/format';
import type { Step } from '../../core/types';
import type { CombinatorialInstance } from '../../problems/combinatorial';
import { getLab } from '../index';
import {
  LabShell,
  MethodSlots,
  RailSection,
  RunSummary,
  focusRuns,
  useMethodSelection,
} from '../_shell';
import { SeedControl } from '../_shell/SeedControl';
import {
  CITY_EDITS_CODEC,
  applyCityEdits,
  baseId,
  exactReference,
  gapPercent,
  methodKind,
  setCityEdit,
  statusResult,
  statusWording,
  traceStepUnit,
  withBestFound,
  withCapacity,
  type CityEdit,
  type Kind,
} from './model';
import { columns } from './docs';
import { TspStage } from './TspStage';
import { KnapsackStage, type KnapsackView } from './KnapsackStage';
import { LabCard } from './LabCard';
import { DEFAULTS, DEFAULT_PROBLEM, PRESETS } from './presets';
import { useCombinatorialRuns } from './useCombinatorialRuns';
import { StageBoundary } from './StageBoundary';
import styles from './CombinatorialLab.module.css';

void preloadKatex();

const CAPACITY_CODEC: Codec<number | null> = {
  parse: (s) => {
    const n = Number(s);
    return Number.isInteger(n) && n >= 0 ? n : undefined;
  },
  format: (v) => String(v),
};
const SEED_CODEC = codecs.number;
const NO_EDITS: CityEdit[] = [];
/** A preset never inherits moved cities (`c`) or an edited capacity (`C`). */
const PRESET_CLEAR_KEYS = ['c', 'C'] as const;

/** The convergence x-axis: one recorded step of each method's trace. */
const X_NAME = [mathSans('recorded step '), mathVar('k')];

const KIND_LABEL: Record<Kind, string> = {
  knapsack: '0-1 knapsack',
  tsp: 'Traveling salesman',
};

const compactQuery = () => window.matchMedia('(max-width: 899px)');
function useCompact(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const mq = compactQuery();
      mq.addEventListener('change', cb);
      return () => mq.removeEventListener('change', cb);
    },
    () => compactQuery().matches,
    () => false,
  );
}

export default function CombinatorialLab() {
  const lab = getLab('combinatorial')!;
  const compact = useCompact();
  const problems = useMemo(() => listProblems<CombinatorialInstance>('combinatorial'), []);
  const allMethods = useMemo(() => listMethods('combinatorial'), []);

  const [problemId] = useUrlState('p', DEFAULT_PROBLEM);
  const base =
    problems.find((p) => p.id === problemId) ?? problems.find((p) => p.id === DEFAULT_PROBLEM)!;
  const kind: Kind = base.kind;
  const available = useMemo(
    () => allMethods.filter((m) => methodKind(m) === kind),
    [allMethods, kind],
  );

  const [edits, setEdits] = useUrlState<CityEdit[]>('c', NO_EDITS, CITY_EDITS_CODEC);
  const [capacity, setCapacity] = useUrlState<number | null>('C', null, CAPACITY_CODEC);
  const [seed, setSeed] = useUrlState('s', 0, SEED_CODEC);
  const problem: CombinatorialInstance = useMemo(
    () => (base.kind === 'tsp' ? applyCityEdits(base, edits) : withCapacity(base, capacity)),
    [base, edits, capacity],
  );
  const edited = problem.id !== base.id;

  const [selection, setSelection] = useMethodSelection(available, DEFAULTS[kind]);
  const stochastic = selection.some(
    (s) => available.find((m) => m.spec.id === s.id)?.spec.deterministic === false,
  );
  const runOptions = useMemo(() => ({ seed }), [seed]);
  // The runs and the problem they belong to (the previous one while a new run is pending).
  const {
    runs,
    problem: runsProblem,
    pending,
  } = useCombinatorialRuns(problem, selection, runOptions);
  const shownKind: Kind = runsProblem.kind;
  // Draw on the live instance only when the old runs fit it (same library instance, so the
  // same cities or items); otherwise on the instance the runs were computed on.
  const stageProblem: CombinatorialInstance =
    runsProblem.kind === problem.kind &&
    baseId(runsProblem.id) === baseId(problem.id) &&
    problem.kind === 'tsp'
      ? problem
      : runsProblem;
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const [focusId, setFocusId] = useState<string | null>(null);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [kv, setKv] = useState<'items' | 'detail'>('items');
  const [scale, setScale] = useState<'auto' | 'lin' | 'log'>('auto');
  const focus = runs.find((r) => r.sel.id === focusId) ?? runs.find((r) => !r.error) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus ? Math.min(focus.result.trace.length - 1, player.localK(focusIndex)) : 0;
  const focusStep: Step | undefined = focus?.result.trace[Math.max(0, focusK)];

  const exact = useMemo(() => exactReference(runsProblem), [runsProblem]);
  const reference = useMemo(
    () =>
      withBestFound(
        exact,
        shownKind,
        runs.filter((r) => !r.error).map((r) => r.result),
      ),
    [exact, shownKind, runs],
  );
  const bestFound = reference.source === 'best-found';
  const ctx = useMemo(
    () => ({ problem: runsProblem, reference: reference.value, bestFound }),
    [runsProblem, reference.value, bestFound],
  );

  // ── Problem change: one URL update (drop edits; drop methods when the kind changes) ──
  const changeProblem = (id: string) => {
    const next = problems.find((p) => p.id === id);
    const q = readQuery();
    q.delete('c');
    q.delete('C');
    if (next && next.kind !== kind) q.delete('m');
    if (id === DEFAULT_PROBLEM) q.delete('p');
    else q.set('p', id);
    replaceQuery(q);
    setFocusId(null);
  };

  const nnSel = selection.find((s) => s.id === 'tsp_nearest_neighbor');
  const nnStart = nnSel ? Number(nnSel.params.start ?? 0) : null;
  const pickStart = nnSel
    ? (i: number) =>
        setSelection(
          selection.map((s) =>
            s.id === 'tsp_nearest_neighbor' ? { ...s, params: { ...s.params, start: i } } : s,
          ),
        )
    : undefined;

  // ── Convergence: relative gap in percent ─────────────────────────────────────────────
  const series = useMemo(
    () =>
      runs
        .filter((r) => !r.error)
        .map((r) => ({
          label: `${r.method.spec.name} · step = ${traceStepUnit(r.sel.id, r.sel.params)}`,
          slot: r.sel.slot,
          values: r.result.trace.map((s) => gapPercent(shownKind, s, reference.value)),
          end: (r.result.converged ? 'converged' : 'stopped') as 'converged' | 'stopped',
        })),
    [runs, shownKind, reference.value],
  );
  // Log axis by default unless a run reaches the optimum exactly (0 is not on a log axis).
  const reachesZero = series.some((s) => s.values.some((v) => v !== null && v <= 1e-9));
  const logY = scale === 'auto' ? !reachesZero : scale === 'log';
  const refSym = shownKind === 'tsp' ? 'L' : 'z';
  const refTex = bestFound ? `${refSym}_{\\text{best}}` : `${refSym}^\\star`;
  const refNote =
    reference.value === null
      ? 'optimum unknown'
      : bestFound
        ? `the best ${shownKind === 'tsp' ? 'tour' : 'selection'} of these runs. The optimum of this edited instance is unknown (more than 16 cities), so every gap is measured to the best found.`
        : {
            library: 'library optimum, independently checked',
            'held-karp': 'exact: Held–Karp in the browser',
            dp: 'exact: dynamic programming in the browser',
            'best-found': '',
            none: '',
          }[reference.source];
  const refPlain = bestFound ? `${refSym} best` : `${refSym}⋆`;
  const refMath = bestFound ? [mathVar(refSym), mathSub('best')] : [mathVar(refSym), mathSup('⋆')];

  const problemOptions: SelectOption[] = problems.map((p) => ({
    value: p.id,
    label: p.name,
    description: p.description,
    group: KIND_LABEL[p.kind],
    keywords: p.tags.join(' '),
  }));

  const hasDetail = shownKind === 'knapsack' && !!focus && focus.sel.id !== 'knapsack_greedy';
  // Phones: with more than two TSP methods, one panel at a time (the shell's focus picker, in the
  // details header, switches it).
  const tspRuns = compact && runs.length > 2 && focus ? [focus] : runs;
  const view: KnapsackView = compact && hasDetail ? kv : compact ? 'items' : 'both';

  const total = problem.kind === 'knapsack' ? problem.weights.reduce((a, b) => a + b, 0) : 0;
  // Narrow screens: size the stage to the square tour panels (one header line each), so no
  // empty band is left under them.
  const tspCols = tspRuns.length >= 2 ? 2 : 1;
  const tspRows = Math.max(1, Math.ceil(tspRuns.length / tspCols));
  const stageHeightNarrow =
    shownKind === 'tsp' ? `min(${tspRows} * (${100 / tspCols}cqw + 46px), 75dvh)` : undefined;

  return (
    <LabShell
      lab={lab}
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
      presets={PRESETS}
      presetClearKeys={PRESET_CLEAR_KEYS}
      pending={pending}
      stageHeightNarrow={stageHeightNarrow}
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <Select
              label="Problem"
              value={base.id}
              options={problemOptions}
              onChange={changeProblem}
            />
            <div className={styles.formulaBox}>
              <Formula tex={base.latex} display fit />
            </div>
            <p className={styles.desc}>{problem.description}</p>
            {edited && (
              <Button
                size="sm"
                variant="ghost"
                icon="reset"
                onClick={() => (kind === 'tsp' ? setEdits(NO_EDITS) : setCapacity(null))}
              >
                Restore the library instance
              </Button>
            )}
          </RailSection>
          {/* Problem data before Methods: Try this → Problem → Capacity / Cities → Methods → seed. */}
          {kind === 'knapsack' ? (
            <RailSection title="Capacity">
              <NumberField
                label="Capacity C"
                prefix="C"
                value={problem.kind === 'knapsack' ? problem.capacity : 0}
                integer
                min={0}
                max={total}
                onChange={(v) =>
                  setCapacity(
                    base.kind === 'knapsack' && v === base.capacity ? null : Math.round(v),
                  )
                }
              />
              <p className={styles.hint}>
                <Icon name="crosshair" size={13} />
                or drag the line C in the items plot
              </p>
            </RailSection>
          ) : (
            <RailSection title="Cities">
              <p className={styles.hint}>
                <Icon name="crosshair" size={13} />
                Drag a city to move it{nnSel ? '; click one to start nearest neighbor there' : ''}.
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
            <MethodSlots available={available} value={selection} onChange={setSelection} />
          </RailSection>
          {stochastic && <SeedControl value={seed} onChange={setSeed} />}
        </>
      }
      stageTitle={
        <>
          <span className={styles.problemName}>{problem.name}</span>
          <RunSummary
            runs={runs.map((r) => ({
              name: r.method.spec.name,
              slot: r.sel.slot,
              result: statusResult(r.sel.id, r.result),
              error: r.error,
              wording: statusWording(r.sel.id),
            }))}
          />
        </>
      }
      stageToolbar={
        <>
          {compact && hasDetail && (
            <SegmentedControl
              label="Knapsack view"
              value={kv}
              onChange={setKv}
              options={[
                { value: 'items', label: 'Items' },
                { value: 'detail', label: focus?.sel.id === 'knapsack_dp' ? 'Table' : 'Tree' },
              ]}
            />
          )}
        </>
      }
      stage={
        <StageBoundary resetKey={`${runsProblem.id}|${runs.map((r) => r.sel.id).join()}`}>
          {stageProblem.kind === 'tsp' ? (
            <TspStage
              problem={stageProblem}
              runs={tspRuns}
              indexOf={(r) => runs.indexOf(r)}
              player={player}
              reference={reference.value}
              bestFound={bestFound}
              nnStart={nnStart}
              onPickStart={pickStart}
              onMoveCity={(i, x, y) => setEdits(setCityEdit(edits, i, x, y))}
            />
          ) : (
            <KnapsackStage
              problem={stageProblem}
              runs={runs}
              player={player}
              focus={focus}
              focusIndex={focusIndex}
              reference={reference.value}
              view={view}
              onCapacity={(C) =>
                setCapacity(base.kind === 'knapsack' && C === base.capacity ? null : C)
              }
            />
          )}
        </StageBoundary>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: bestFound ? 'Gap to the best found' : 'Gap to the optimum',
          height: 278,
          actions: (
            <SegmentedControl
              label="Gap axis"
              value={logY ? 'log' : 'lin'}
              onChange={(v) => setScale(v)}
              options={[
                { value: 'lin', label: 'linear', ariaLabel: 'Linear gap axis' },
                { value: 'log', label: 'log', ariaLabel: 'Logarithmic gap axis' },
              ]}
            />
          ),
          content: (
            <div className={styles.chartBox}>
              <p className={styles.chartCaption}>
                {reference.value !== null && (
                  <>
                    <Formula
                      tex={`${refTex} = ${sig(reference.value, 7).replace('−', '-')}`}
                    />{' '}
                  </>
                )}
                {reference.value !== null && !bestFound ? `(${refNote})` : refNote}
                {logY && reachesZero
                  ? ` · a gap of 0 (${bestFound ? 'the best found' : 'optimal'}) is not on the log axis`
                  : ''}
              </p>
              <div className={styles.chartArea}>
                <ConvergenceChart
                  series={series}
                  t={player.t}
                  logY={logY}
                  yLabel={
                    shownKind === 'tsp'
                      ? `(L − ${refPlain})/${refPlain} in %`
                      : `(${refPlain} − z)/${refPlain} in %`
                  }
                  yName={
                    shownKind === 'tsp'
                      ? [
                          mathMain('('),
                          mathVar('L'),
                          mathMain(' − '),
                          ...refMath,
                          mathMain(')/'),
                          ...refMath,
                          mathSans(' %'),
                        ]
                      : [
                          mathMain('('),
                          ...refMath,
                          mathMain(' − '),
                          mathVar('z'),
                          mathMain(')/'),
                          ...refMath,
                          mathSans(' %'),
                        ]
                  }
                  xName={X_NAME}
                  onSeek={(k) => {
                    player.pause();
                    player.seek(k);
                  }}
                  ariaLabel={`Relative gap to the ${bestFound ? 'best found' : 'optimum'} per recorded step: ${series.map((s) => `${s.label} ends at ${sig(s.values[s.values.length - 1] ?? NaN, 3)} %`).join('; ')}.`}
                />
              </div>
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
              idBase="comb-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="comb-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <LabCard run={focus} step={focusStep} ctx={ctx} kind={shownKind} seed={seed} />
              ) : (
                <div className={styles.tableBox}>
                  <TableView
                    columns={columns(focus.sel.id, ctx, shownKind)}
                    rows={focus.result.trace}
                    highlight={focusK}
                    futureAfter={focusK}
                    onSelect={(i) => {
                      player.pause();
                      player.seek(i);
                    }}
                    rowKey={(s, i) => `${i}-${s.k}`}
                    maxHeight="100%"
                    ariaLabel={`${focus.method.spec.name} steps`}
                    empty={focus.error ?? 'No steps.'}
                  />
                </div>
              )}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}
