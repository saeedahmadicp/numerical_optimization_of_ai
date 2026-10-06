/**
 * Global lab — stochastic global minimization on multimodal landscapes: simulated annealing,
 * particle swarm, differential evolution, CMA-ES and basin hopping (TS port of
 * numopt.unconstrained.global_), replayed generation by generation over the contour field.
 */
import './setup';
import { useCallback, useMemo, useRef, useState } from 'react';
import { defaults, listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Problem2D, Step } from '../../core/types';
import { useUrlState, codecs, type Codec } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES } from '../../ui/colors';
import { Formula, PlaybackBar, SegmentedControl, TabPanel, Tabs } from '../../ui/components';
import {
  Contour2D,
  ConvergenceChart,
  IterationTable,
  TableView,
  mathBold,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  type MathRun,
  type View2D,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { deRule } from '../../methods/unconstrained/global_';
import { getLab } from '../index';
import {
  KAxisControl,
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  StartPointFields,
  focusRuns,
  useKAxis,
  useLabRunsState,
  useMethodSelection,
  useProblemState,
  useStartPoint,
} from '../_shell';
import { drawMethodLayers, drawSearchBox, drawTracks, type MethodLayer } from './draw';
import {
  TRACK_NAME,
  clampToBox,
  globalMin,
  isGlobalMethod,
  searchScale,
  temperatureOf,
  type Pt,
} from './geometry';
import { iterationColumns, populationView } from './tables';
import { DEFAULT_PROBLEM, DEFAULT_SELECTION, PRESETS, SEED_KEY } from './presets';
import { StepEquation } from './StepEquation';
import { railProblem } from './railTex';
import { StageKey } from './StageKey';
import { SeedControl } from '../_shell/SeedControl';
import { useStartWhenDrawn } from '../../play/useStartWhenDrawn';
import styles from './GlobalLab.module.css';

void preloadKatex();

type Metric = 'gap' | 'spread' | 'temp';
type Tab = 'method' | 'steps' | 'pop';
const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'spread', 'temp']);
const SEED_CODEC: Codec<number> = {
  parse: (s) => {
    const n = Number(s);
    return /^\d+$/.test(s) && Number.isSafeInteger(n) && n < 2 ** 32 ? n : undefined;
  },
  format: (v) => String(v),
};

/** Multimodal problems first (the lab's subject), then the rest of the 2-D library. */
const RANK = (tags: readonly string[] = []) =>
  tags.includes('multimodal') ? 0 : tags.includes('multiple-minima') ? 1 : 2;

const GAP_FLOOR = 1e-16;

const Y_NAME: Record<Metric, MathRun[]> = {
  gap: [
    mathVar('f'),
    mathMain('('),
    mathBold('x'),
    mathSub('best'),
    mathMain(') − '),
    mathVar('f'),
    mathSup('⋆'),
  ],
  spread: [mathMain('search scale (∞-norm)')],
  temp: [mathVar('T'), mathSub('k', 'italic')],
};
const Y_LABEL: Record<Metric, string> = {
  gap: 'f(x_best) − f⋆',
  spread: 'search scale',
  temp: 'temperature T',
};

export default function GlobalLab() {
  const lab = getLab('global')!;

  const problems = useMemo(
    () =>
      listProblems<Problem2D>('unconstrained')
        .filter((p) => p.dim === 2)
        .map((p, i) => ({ p, i }))
        .sort((a, b) => RANK(a.p.tags) - RANK(b.p.tags) || a.i - b.i)
        .map(({ p }) => p),
    [],
  );
  const methods = useMemo(() => listMethods('global'), []);
  const pickerProblems = useMemo(() => problems.map(railProblem), [problems]);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const box = problem.domain;
  const [x0, setX0] = useStartPoint(problem.x0 as number[] | undefined, 2);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [seed, setSeed] = useUrlState<number>(SEED_KEY, 0, SEED_CODEC);
  const [metric, setMetric] = useUrlState<Metric>('y', 'gap', METRIC_CODEC);
  const [tab, setTab] = useState<Tab>('method');
  const [focusId, setFocusId] = useState<string | null>(null);

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]], seed }), [x0, seed]);
  // Not deferred: every run here takes milliseconds, and a deferred render restarts on each
  // animation frame while the player runs (seconds of old paths over the new problem).
  const { runs } = useLabRunsState(problem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces, { autoplay: false });
  usePlayerKeyboard(player);
  const plotRef = useRef<HTMLDivElement>(null);
  useStartWhenDrawn(player, plotRef, traces);
  const [logK, setKAxis] = useKAxis(traces.map((t) => t.length));

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus ? player.localK(focusIndex) : 0;
  const focusStep: Step | undefined = focus?.result.trace[focusK];
  const paramsOf = useCallback(
    (r: (typeof runs)[number]) => ({ ...defaults(r.method.spec), ...r.sel.params }),
    [],
  );
  // One object per run (not per frame): "This step" reads its layout samples once per run.
  const focusParams = useMemo(() => (focus ? paramsOf(focus) : {}), [focus, paramsOf]);

  const fStar = useMemo(() => {
    const known = globalMin(problem);
    if (known !== null) return known;
    const seen = runs.flatMap((r) => r.result.trace.map((s) => s.fun ?? Infinity));
    return Math.min(...seen);
  }, [problem, runs]);

  // ── landscape ─────────────────────────────────────────────────────────────────────────
  const layers: MethodLayer[] = runs
    .filter((r) => isGlobalMethod(r.sel.id) && r.result.trace.length > 0)
    .map((r) => ({
      method: r.sel.id,
      slot: r.sel.slot,
      trace: r.result.trace,
      lt: player.localT(runs.indexOf(r)),
      focused: r === focus,
      muted: focusId !== null && r !== focus,
      params: paramsOf(r),
    }));
  const ease = !player.reducedMotion;
  const overlay = (ctx: CanvasRenderingContext2D, view: View2D) => {
    drawSearchBox(ctx, view, box);
    drawTracks(ctx, view, layers);
    drawMethodLayers(ctx, view, layers, { ease, box });
  };

  const f2 = useCallback((x: number, y: number) => problem.f([x, y]), [problem]);
  // A click picks 𝐱₀; annealing, swarm and DE need it inside the box Ω.
  const pickStart = useCallback((q: [number, number]) => setX0(clampToBox(q, box)), [setX0, box]);
  const minima = (problem.minima ?? []) as [number, number][];

  // ── charts ────────────────────────────────────────────────────────────────────────────
  const series = useMemo(
    () =>
      runs.map((r) => {
        const prm = paramsOf(r);
        return {
          label: r.method.spec.name,
          slot: r.sel.slot,
          count: r.result.nIter,
          end: r.result.converged ? ('converged' as const) : ('stopped' as const),
          values: r.result.trace.map((s: Step) =>
            metric === 'gap'
              ? s.fun === null
                ? null
                : Math.max(s.fun - fStar, GAP_FLOOR)
              : metric === 'spread'
                ? searchScale(r.sel.id, s)
                : temperatureOf(r.sel.id, s, Number(prm.T)),
          ),
        };
      }),
    [runs, metric, fStar, paramsOf],
  );
  const hasTemp = runs.some(
    (r) => r.sel.id === 'simulated_annealing' || r.sel.id === 'basin_hopping',
  );
  const onlyHopping = runs.length > 0 && runs.every((r) => r.sel.id === 'basin_hopping');
  const seek = (k: number) => {
    player.pause();
    player.seek(k);
  };

  // ── details ───────────────────────────────────────────────────────────────────────────
  const pop = focus && focusStep ? populationView(focus.sel.id, focusStep) : null;
  const tabs: { id: Tab; label: string }[] = [
    { id: 'method', label: 'Method' },
    { id: 'steps', label: 'Iterations' },
    ...(pop ? [{ id: 'pop' as const, label: pop.title }] : []),
  ];
  const activeTab: Tab = tabs.some((t) => t.id === tab) ? tab : 'method';

  // The MethodCard shows the DE mutation of the selected strategy only.
  const cardMethod =
    focus && focus.sel.id === 'differential_evolution' && focus.method.doc
      ? {
          ...focus.method,
          doc: {
            ...focus.method.doc,
            rule: deRule(focusParams.strategy === 'best/1/bin' ? 'best/1/bin' : 'rand/1/bin'),
          },
        }
      : focus?.method;

  const changeStart = (v: number[]) => setX0([v[0], v[1]]);
  const trackNote = runs
    .filter((r) => isGlobalMethod(r.sel.id))
    .map((r) => `${r.method.spec.name}: ${TRACK_NAME[r.sel.id as keyof typeof TRACK_NAME]}`)
    .join('; ');

  const ariaLabel =
    `Contour plot of ${problem.name} on the box Ω with ` +
    (runs.length
      ? runs
          .map((r) => {
            const s = r.result.trace[player.localK(runs.indexOf(r))];
            return `${r.method.spec.name} at step ${s?.k ?? 0}, best f = ${s?.fun?.toPrecision(4) ?? '—'}`;
          })
          .join('; ')
      : 'no method selected') +
    '.';

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
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
            <MethodSlots available={methods} value={selection} onChange={setSelection} />
          </RailSection>
          <SeedControl value={seed} onChange={setSeed} />
          <RailSection title="Start point">
            <StartPointFields value={x0} onChange={changeStart} variable="𝐱₀" />
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
              error: r.error,
            }))}
          />
        </>
      }
      stage={
        <div className={styles.stage}>
          <div className={styles.plot} ref={plotRef}>
            <Contour2D
              f={f2}
              domain={problem.domain}
              cacheKey={problem.id}
              fMin={fStar}
              t={player.t}
              ease={ease}
              minima={minima}
              start={[x0[0], x0[1]] as Pt}
              overlay={overlay}
              onPick={pickStart}
              ariaLabel={ariaLabel}
            />
          </div>
          {focus && isGlobalMethod(focus.sel.id) && (
            <div className={styles.keyBar}>
              <StageKey
                method={focus.sel.id}
                name={focus.method.spec.name}
                slot={focus.sel.slot}
                strategy={String(focusParams.strategy ?? '')}
              />
              {trackNote && <p className={styles.trackNote}>Lines follow — {trackNote}.</p>}
            </div>
          )}
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
      insights={[
        {
          id: 'convergence',
          title:
            metric === 'temp'
              ? 'Temperature'
              : metric === 'spread'
                ? 'Search scale'
                : 'Best so far',
          height: 280,
          actions: (
            <div className={styles.chartActions}>
              <KAxisControl logX={logK} onChange={setKAxis} />
              <SegmentedControl
                className={styles.metricSeg}
                label="Chart"
                value={metric}
                onChange={setMetric}
                options={[
                  {
                    value: 'gap',
                    label: <Formula tex="f_{\mathrm{best}} - f^\star" />,
                    ariaLabel: 'Best value found minus the global minimum',
                  },
                  { value: 'spread', label: 'scale', ariaLabel: 'Search scale' },
                  { value: 'temp', label: <Formula tex="T_k" />, ariaLabel: 'Temperature' },
                ]}
              />
            </div>
          ),
          content: (
            <div className={styles.chartPad}>
              {metric === 'temp' && !hasTemp ? (
                <p className={styles.chartEmpty}>
                  Temperature applies to simulated annealing (Tₖ) and basin hopping (a fixed T). Add
                  one of them to see it.
                </p>
              ) : metric === 'spread' && onlyHopping ? (
                <p className={styles.chartEmpty}>
                  Basin hopping has no search scale: its hop size s is fixed. Add a population
                  method or annealing to see one.
                </p>
              ) : (
                <ConvergenceChart
                  series={
                    metric === 'gap'
                      ? series
                      : series.filter((s) => s.values.some((v) => v !== null))
                  }
                  t={player.t}
                  logX={logK}
                  yLabel={Y_LABEL[metric]}
                  yName={Y_NAME[metric]}
                  onSeek={seek}
                  ariaLabel={`${Y_LABEL[metric]} against iteration k for ${runs.map((r) => r.method.spec.name).join(', ')}`}
                />
              )}
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
              value={activeTab}
              onChange={setTab}
              idBase="global-details"
              items={tabs}
            />
          ),
          content: focus ? (
            <div className={styles.details}>
              <div className={styles.detailsBody}>
                <TabPanel idBase="global-details" id={activeTab} focusable={false}>
                  {activeTab === 'method' ? (
                    <div className={styles.methodTab}>
                      <MethodCard
                        live={
                          !focus.error && (
                            <StepEquation
                              method={focus.sel.id}
                              trace={focus.result.trace}
                              k={focusK}
                              params={focusParams}
                              n={2}
                              popSize={Number(focusParams.pop_size ?? 0) || undefined}
                            />
                          )
                        }
                        method={cardMethod!}
                        slot={focus.sel.slot}
                        result={focus.result}
                        error={focus.error}
                        call={{
                          problem: problem.id,
                          options: { x0: [x0[0], x0[1]], seed },
                          params: focus.sel.params,
                        }}
                      />
                    </div>
                  ) : activeTab === 'steps' ? (
                    <IterationTable
                      steps={focus.result.trace}
                      columns={iterationColumns(
                        focus.sel.id,
                        Number(focusParams.pop_size ?? 0) || undefined,
                      )}
                      k={focusK}
                      onSelect={seek}
                      ariaLabel={`${focus.method.spec.name} iterations`}
                    />
                  ) : pop ? (
                    <div className={styles.popTab}>
                      <TableView
                        caption={pop.caption}
                        columns={pop.columns}
                        rows={pop.rows}
                        highlight={pop.highlight}
                        rowKey={(r) => r.i}
                        ariaLabel={`${focus.method.spec.name}: ${pop.title.toLowerCase()} at step ${focusK}`}
                        empty="No members at this step."
                      />
                    </div>
                  ) : null}
                </TabPanel>
              </div>
            </div>
          ) : null,
        },
      ]}
    />
  );
}
