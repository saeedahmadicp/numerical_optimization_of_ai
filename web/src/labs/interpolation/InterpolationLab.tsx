/**
 * Interpolation lab: polynomial forms (Lagrange, barycentric, Newton, Neville, Chebyshev),
 * piecewise cubics (linear, natural / clamped / not-a-knot splines, PCHIP) and rational
 * functions (AAA, Floater–Hormann) through the same nodes. The nodes are draggable; the layout
 * control resamples f at n equispaced or Chebyshev nodes (Runge's phenomenon); the strip under
 * the plot shows the error, the nodal polynomial, the Lagrange basis or AAA's poles in the
 * complex plane on the same x-axis.
 */
import './setup';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { listMethods, type RegisteredMethod } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { DataProblem } from '../../problems/data';
import type { Result, Step } from '../../core/types';
import { sig, sigFixed } from '../../core/format';
import { codecs, useUrlState } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import {
  Button,
  CopyButton,
  Formula,
  PlaybackBar,
  SegmentedControl,
  Slider,
  TabPanel,
  Tabs,
  Toggle,
} from '../../ui/components';
import {
  ConvergenceChart,
  TableView,
  mathMain as mm,
  mathVar as mv,
  type TableColumn,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { MAX_SERIES } from '../../ui/colors';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  useLabRuns,
  useMethodSelection,
  useProblemState,
} from '../_shell';
import {
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  KEYS,
  LAYOUT_CODEC,
  PRESETS,
  STAGE_LABEL,
  STRIP_CODEC,
  SWEEP_N,
  tableTitle,
} from './config';
import {
  MAX_NODES,
  MIN_NODES,
  activeData,
  basisFn,
  detectLayout,
  flattenPoints,
  methodKind,
  sampled,
  stepFunction,
  viewRange,
  widestGapNode,
  type ActiveData,
  type Layout,
} from './geometry';
import { filledRule } from './cardRule';
import { InterpPlot, type RunView } from './InterpPlot';
import { StripPlot, type StripMode } from './StripPlot';
import { MethodTable } from './tables';
import { errorVsN } from './sweep';
import { SweepChart } from './SweepChart';
import { mathHead } from './heads';
import styles from './InterpolationLab.module.css';

void preloadKatex();

const EMPTY: number[] = [];
type DetailTab = 'method' | 'table' | 'steps';
type ChartTab = 'n' | 'k';

const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);
/** Methods that work on the sorted nodes (extra.nodes), not in data order. */
const sortedKind = (id: string) => methodKind(id) === 'piecewise' || methodKind(id) === 'blend';

/** A Python snippet that reproduces the run on the current nodes. */
function pythonData(x: readonly number[], y: readonly number[], methodId: string): string {
  const arr = (v: readonly number[]) => `np.array([${v.map((t) => String(t)).join(', ')}])`;
  return [
    'import numpy as np',
    'import numopt',
    `x = ${arr(x)}`,
    `y = ${arr(y)}`,
    `res = numopt.run(${JSON.stringify(methodId)}, (x, y))`,
    'print(res.converged, res.message)',
  ].join('\n');
}

export default function InterpolationLab() {
  const lab = getLab('interpolation')!;
  const problems = useMemo(() => listProblems<DataProblem>('data'), []);
  const methods = useMemo(() => listMethods('interpolation'), []);

  const [base, setBase] = useProblemState(problems, DEFAULT_PROBLEM);
  const hasF = Boolean(base.fTrue);
  const [layoutRaw, setLayout] = useUrlState<Layout>(KEYS.layout, 'data', LAYOUT_CODEC);
  const layout: Layout = hasF ? layoutRaw : 'data';
  const [nRaw, setN] = useUrlState<number>(KEYS.count, base.x.length, codecs.number);
  const n = Math.min(MAX_NODES, Math.max(MIN_NODES, Math.round(nRaw)));
  const [points, setPoints] = useUrlState<number[]>(KEYS.nodes, EMPTY, codecs.numbers);
  const [snap, setSnap] = useUrlState<boolean>(KEYS.snap, true, codecs.bool);
  const [strip, setStrip] = useUrlState<StripMode>(KEYS.strip, 'error', STRIP_CODEC);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);

  const [draft, setDraft] = useState<number[] | null>(null);
  const [dragging, setDragging] = useState(false);
  const [frozenY, setFrozenY] = useState<[number, number] | null>(null);
  const [tab, setTab] = useState<DetailTab>('method');
  const [chartTab, setChartTab] = useState<ChartTab>(hasF ? 'n' : 'k');
  const [focusId, setFocusId] = useState<string | null>(null);

  const data = useMemo(
    () => activeData(base, layout, n, draft ?? (points.length ? points : null)),
    [base, layout, n, draft, points],
  );

  const runs = useLabRuns(data, selection);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  // While a node is dragged the curves follow it at the final step (no restart per move).
  const player = useTracePlayer(traces, { resetOnChange: !dragging });
  usePlayerKeyboard(player);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus
    ? Math.min(player.localK(focusIndex), Math.max(0, focus.result.trace.length - 1))
    : 0;

  // The y-range: data, f and the final curves (excursions beyond it are labelled, not squashed).
  const autoY = useMemo(() => {
    const ys = [...data.y];
    const ev = (r: (typeof runs)[number]) =>
      r.result.extra.eval as { y?: number[]; f_true?: number[] | null } | undefined;
    for (const r of runs) ys.push(...nums(ev(r)?.f_true));
    return viewRange(
      ys,
      runs.map((r) => nums(ev(r)?.y)),
    );
  }, [data, runs]);
  const yDomain = frozenY ?? autoY;

  const runKey = (r: (typeof runs)[number]) =>
    `${data.id}|${r.sel.id}|${JSON.stringify(r.sel.params)}`;
  // The views change only when a run's own step changes, so the memoized stage and strip redraw
  // per step, not on every playback frame.
  const ks = runs.map((r, i) => Math.min(player.localK(i), Math.max(0, r.result.trace.length - 1)));
  const ksKey = ks.join(',');
  const views: RunView[] = useMemo(
    () =>
      runs.map((r, i) => ({
        id: r.sel.id,
        key: `${data.id}|${r.sel.id}|${JSON.stringify(r.sel.params)}`,
        slot: r.sel.slot,
        name: r.method.spec.name,
        kind: methodKind(r.sel.id),
        result: r.result,
        k: ks[i],
        focus: r === focus,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runs, data.id, focus, ksKey],
  );

  // ── Node editing ─────────────────────────────────────────────────────────────────────
  const onNodesChange = useStableCallback((x: number[], y: number[], final: boolean) => {
    const flat = flattenPoints(x, y);
    if (final) {
      setDraft(null);
      setPoints(flat);
    } else setDraft(flat);
  });
  const onDragChange = useStableCallback((on: boolean) => {
    setDragging(on);
    if (on) {
      setFrozenY(yDomain);
      player.pause();
      player.toEnd();
    } else setFrozenY(null);
  });
  const clearDraft = useCallback(() => setDraft(null), []);
  const resetNodes = () => setPoints(EMPTY);
  const [announce, setAnnounce] = useState('');
  const addNode = () => {
    const p = widestGapNode(data.x, data.y, data.domain, snap ? data.fTrue : undefined);
    if (!p || data.x.length >= MAX_NODES) return;
    onNodesChange([...data.x, p.x], [...data.y, p.y], true);
    setAnnounce(`Node ${data.x.length} added at x = ${sig(p.x, 4)}, y = ${sig(p.y, 4)}.`);
  };
  const chooseLayout = (l: Layout) => {
    setPoints(EMPTY);
    setLayout(l);
  };
  const chooseN = useStableCallback((v: number) => {
    setPoints(EMPTY);
    if (layout === 'data') setLayout('equi');
    setN(v);
  });

  // ── Error vs n (the Runge figure) ────────────────────────────────────────────────────
  const sweepLayout: 'equi' | 'cheb' = layout === 'cheb' ? 'cheb' : 'equi';
  const sweepKey = JSON.stringify(selection.map((s) => [s.id, s.slot, s.params]));
  const sweep = useMemo(
    () => (chartTab === 'n' ? errorVsN(base, sweepLayout, selection, SWEEP_N) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [base, sweepLayout, sweepKey, chartTab],
  );

  // The n of the stage on the error-vs-n axis (when the stage shows that layout).
  const sweepCurrent =
    !data.edited && detectLayout(data.x, data.domain[0], data.domain[1]) === sweepLayout
      ? data.x.length
      : null;
  const ownNodes = sweep.some((s) => s.ownNodes);

  const stepSeries = runs.map((r) => ({
    label: r.method.spec.name,
    slot: r.sel.slot,
    values: r.result.trace.map((s) => (s.fun === null ? null : Math.max(s.fun, 1e-17))),
    end: r.result.converged ? ('converged' as const) : ('stopped' as const),
    count: r.result.trace.length,
  }));

  // ── MethodCard for the focused method ────────────────────────────────────────────────
  const nodes = focus
    ? nums(focus.result.extra.nodes).length
      ? nums(focus.result.extra.nodes)
      : data.x
    : data.x;
  const values = focus
    ? nums(focus.result.extra.values).length
      ? nums(focus.result.extra.values)
      : data.y
    : data.y;
  const card = focus ? cardFor(focus.sel.id, focus.result, focusK, data, runKey(focus)) : null;
  const cardMethod: RegisteredMethod | null =
    focus && card
      ? {
          ...focus.method,
          doc: {
            ...(focus.method.doc ?? { rule: '', intuition: focus.method.spec.summary }),
            rule: filledRule(
              focus.sel.id,
              focus.result,
              focusK,
              sortedKind(focus.sel.id) ? nodes : data.x,
              sortedKind(focus.sel.id) ? values : data.y,
            ),
            quantities: card.quantities,
          },
        }
      : null;
  const libraryData = !data.edited && data.layout === 'data';
  // The Data picker describes what the methods run on: a resampled or edited node set replaces
  // the library dataset's name and note (its node count and error figures no longer apply).
  const pickerProblems = useMemo(
    () =>
      libraryData
        ? problems
        : problems.map((p) =>
            p.id === base.id ? { ...p, name: data.name, description: dataNote(base, data) } : p,
          ),
    [problems, base, data, libraryData],
  );

  // Chebyshev interpolation that resamples f: say so on the stage, under the chips.
  const chebRun = runs.find(
    (r) => r.sel.id === 'chebyshev_interpolation' && r.result.extra.source === 'f_true',
  );
  const chebN = chebRun ? nums(chebRun.result.extra.nodes).length : 0;

  const seekTo = (k: number) => {
    player.pause();
    player.seek(k);
  };

  const basisIndex =
    focus && (methodKind(focus.sel.id) === 'nodewise' || methodKind(focus.sel.id) === 'neville')
      ? Math.min(focusK, data.x.length - 1)
      : null;

  const stripLabel: Record<StripMode, string> = {
    error: `Error |p − f| of each method on a log axis${data.fTrue ? '' : ' (no f known)'}.`,
    omega: `Nodal polynomial ω(x) of the ${data.x.length} nodes against equispaced and Chebyshev nodes.`,
    basis: `Lagrange basis polynomials of the ${data.x.length} nodes and the Lebesgue function.`,
    poles: `Poles of the AAA approximant in the complex plane, with its support points on the real axis.`,
  };

  const layoutNote = data.edited
    ? 'Edited by hand.'
    : layout === 'data'
      ? `The dataset's ${data.x.length} nodes.`
      : layout === 'equi'
        ? `f sampled at ${n} equispaced nodes.`
        : `f sampled at the ${n} roots of T${toSub(n)}.`;

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      focus={
        focus && {
          runs: runs.map((r) => ({ value: r.sel.id, slot: r.sel.slot, name: r.method.spec.name })),
          value: focus.sel.id,
          onChange: setFocusId,
        }
      }
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={pickerProblems} value={base.id} onChange={setBase} />
          </RailSection>
          <RailSection
            title="Nodes"
            actions={
              <span className={styles.count}>
                <span className="visually-hidden">Number of nodes: </span>n = {data.x.length}
              </span>
            }
          >
            <div className={styles.nodeControls}>
              <SegmentedControl
                label="Node layout"
                value={data.edited ? 'data' : layout}
                onChange={chooseLayout}
                fullWidth
                options={[
                  { value: 'data', label: 'Dataset', tooltip: "The dataset's own nodes" },
                  {
                    value: 'equi',
                    label: 'Equispaced',
                    ariaLabel: 'Equispaced nodes',
                    tooltip: hasF ? 'Sample f at n equally spaced nodes' : 'Needs a known f',
                  },
                  {
                    value: 'cheb',
                    label: 'Chebyshev',
                    ariaLabel: 'Chebyshev nodes',
                    tooltip: hasF ? 'Sample f at the n roots of Tₙ' : 'Needs a known f',
                  },
                ]}
              />
              {hasF ? (
                <div className={styles.sliderRow}>
                  <span className={styles.sliderLabel}>
                    <Formula tex="n" /> nodes
                  </span>
                  <Slider
                    label="Number of nodes"
                    value={layout === 'data' ? data.x.length : n}
                    min={MIN_NODES}
                    max={MAX_NODES}
                    integer
                    step={1}
                    onChange={chooseN}
                  />
                  <span className={styles.sliderValue}>
                    {layout === 'data' ? data.x.length : n}
                  </span>
                </div>
              ) : (
                <p className={styles.hint}>No f is known for these data: the nodes are the data.</p>
              )}
              <p className={styles.hint}>{layoutNote}</p>
              {hasF && (
                <label className={styles.toggleRow}>
                  <Toggle checked={snap} onChange={setSnap} label="Keep dragged nodes on f" />
                  <span>
                    Keep dragged nodes on <Formula tex="f" />
                  </span>
                </label>
              )}
              <p className={styles.hint}>
                Drag a node{hasF && snap ? ' along f' : ''}; tap the plot to add one; double-click
                to remove one. Keyboard: Tab to the nodes, ←/→ to pick one, Enter to grab it, the
                arrows to move it, Delete to remove it.
              </p>
              <div className={styles.nodeActions}>
                <Button
                  size="sm"
                  variant="ghost"
                  icon="plus"
                  onClick={addNode}
                  disabled={data.x.length >= MAX_NODES}
                >
                  Add node
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  icon="reset"
                  onClick={resetNodes}
                  disabled={!data.edited}
                >
                  Reset nodes
                </Button>
                {focus && (
                  <CopyButton
                    text={pythonData(data.x, data.y, focus.sel.id)}
                    label="Copy these nodes as a Python call"
                    className={styles.copy}
                  >
                    Copy as Python
                  </CopyButton>
                )}
              </div>
              <p className="visually-hidden" aria-live="polite">
                {announce}
              </p>
            </div>
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
        </>
      }
      stageTitle={
        <>
          <span className={styles.problemName}>{data.name}</span>
          <RunSummary
            runs={runs.map((r) => ({
              name: r.method.spec.name,
              slot: r.sel.slot,
              result: r.result,
              error: r.error,
            }))}
          />
          {chebRun && (
            <span className={styles.titleNote}>
              <span
                aria-hidden="true"
                className={styles.titleDiamond}
                style={{ ['--_c' as string]: `var(--series-${(chebRun.sel.slot % 4) + 1})` }}
              />
              Chebyshev interpolation resamples f at its own {chebN} Chebyshev nodes (roots of T
              {toSub(chebN)}), not at the data nodes.
            </span>
          )}
        </>
      }
      stageToolbar={
        <SegmentedControl
          label="Strip under the plot"
          value={strip}
          onChange={setStrip}
          options={[
            {
              value: 'error',
              label: <Formula tex="|p - f|" />,
              ariaLabel: 'Error',
              tooltip: 'Error |p − f| on a log axis',
            },
            {
              value: 'omega',
              label: <Formula tex="\omega(x)" />,
              ariaLabel: 'Nodal polynomial',
              tooltip: 'Nodal polynomial ∏(x − xᵢ)',
            },
            {
              value: 'basis',
              label: <Formula tex="\ell_j(x)" />,
              ariaLabel: 'Lagrange basis',
              tooltip: 'Lagrange basis and Lebesgue function',
            },
            {
              value: 'poles',
              label: <Formula tex="\operatorname{Im} z" />,
              ariaLabel: 'Poles',
              tooltip: 'Poles of the AAA approximant in the complex plane',
            },
          ]}
        />
      }
      stage={
        <div className={styles.stageGrid}>
          <InterpPlot
            data={data}
            runs={views}
            yDomain={yDomain}
            snap={snap}
            onNodesChange={onNodesChange}
            onDragChange={onDragChange}
            onDragCancel={clearDraft}
            ariaLabel={`${data.name}: ${data.x.length} nodes and the interpolants of ${runs.map((r) => r.method.spec.name).join(', ')}.`}
          />
          <div className={styles.strip}>
            <StripPlot
              mode={strip}
              data={data}
              runs={views}
              basisIndex={basisIndex}
              ariaLabel={stripLabel[strip]}
            />
          </div>
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'error',
          label: 'Error',
          height: 300,
          title: (
            <Tabs
              label="Error chart"
              value={chartTab}
              onChange={setChartTab}
              idBase="interp-chart"
              items={[
                { id: 'n', label: 'Error vs n' },
                { id: 'k', label: 'Per step' },
              ]}
            />
          ),
          content: (
            <TabPanel idBase="interp-chart" id={chartTab} focusable={false}>
              {!data.fTrue && !hasF ? (
                <p className={styles.empty}>
                  No f is known for these data, so the interpolation error cannot be measured.
                </p>
              ) : chartTab === 'n' ? (
                <div className={styles.chartPad}>
                  <SweepChart
                    series={sweep}
                    layout={sweepLayout}
                    current={sweepCurrent}
                    onPick={chooseN}
                    ariaLabel={`Maximum error against the number of ${sweepLayout === 'cheb' ? 'Chebyshev' : 'equispaced'} nodes, n = 3 to 40`}
                  />
                  <p className={styles.caption}>
                    max |p − f| on 200 points, f sampled at n = 3…40{' '}
                    {sweepLayout === 'cheb' ? 'Chebyshev' : 'equispaced'} nodes.
                    {ownNodes &&
                      ' Dashed: Chebyshev interpolation, which samples f at its own n Chebyshev nodes.'}{' '}
                    Click an n to resample.
                  </p>
                </div>
              ) : (
                <div className={styles.chartPad}>
                  <ConvergenceChart
                    series={stepSeries}
                    t={player.t}
                    yLabel="max |pₖ − f|"
                    yName={[
                      mm('max |'),
                      mv('p'),
                      { t: 'k', style: 'italic', script: 'sub' },
                      mm(' − '),
                      mv('f'),
                      mm('|'),
                    ]}
                    onSeek={seekTo}
                    ariaLabel="Maximum error of each step's approximant"
                  />
                  <p className={styles.caption}>
                    Error of the approximant after step k; a spline has one only once its pieces
                    exist.
                  </p>
                </div>
              )}
            </TabPanel>
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
              idBase="interp-details"
              items={[
                { id: 'method', label: 'Method' },
                // One fixed name: a per-method title ("Divided differences", "Tableau") would
                // resize the tab row whenever the focus changes; the pane names its table.
                { id: 'table', label: 'Table' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="interp-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <div className={styles.cardWrap}>
                  <MethodCard
                    method={cardMethod ?? focus.method}
                    slot={focus.sel.slot}
                    step={card?.step}
                    result={focus.result}
                    error={focus.error}
                    call={libraryData ? { problem: base.id, params: focus.sel.params } : undefined}
                  />
                </div>
              ) : tab === 'table' ? (
                <div
                  className={styles.tablePane}
                  tabIndex={0}
                  role="region"
                  aria-label={`${tableTitle(focus.sel.id)} of ${focus.method.spec.name}`}
                >
                  <p className={styles.tableTitle} aria-hidden="true">
                    {tableTitle(focus.sel.id)}
                  </p>
                  <MethodTable
                    methodId={focus.sel.id}
                    result={focus.result}
                    k={focusK}
                    nodes={nodes}
                    values={values}
                    domain={data.domain}
                    onSelect={seekTo}
                  />
                </div>
              ) : (
                <StepsTable
                  methodId={focus.sel.id}
                  steps={focus.result.trace}
                  k={focusK}
                  x={data.x}
                  y={data.y}
                  onSelect={seekTo}
                />
              )}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}

/** A callback with a stable identity that always runs the latest `fn` (for memoized children). */
function useStableCallback<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  useLayoutEffect(() => {
    ref.current = fn;
  });
  return useCallback((...args: A) => ref.current(...args), []);
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
function toSub(n: number): string {
  return String(n)
    .split('')
    .map((d) => SUB[Number(d)])
    .join('');
}

/** The Data picker's note for a resampled or edited node set (replaces the dataset's note). */
function dataNote(base: DataProblem, data: ActiveData): string {
  const [a, b] = data.domain;
  const fn = base.name.split(', ')[0];
  if (data.edited)
    return `${data.x.length} nodes edited by hand${data.fTrue ? `; f is still ${fn}` : ''}. Reset nodes restores the dataset's ${base.x.length}.`;
  const kind =
    data.layout === 'cheb'
      ? `the ${data.x.length} roots of T${toSub(data.x.length)} (Chebyshev nodes)`
      : `${data.x.length} equispaced nodes`;
  return `${fn} sampled at ${kind} on [${sig(a, 3)}, ${sig(b, 3)}]; the dataset's own ${base.x.length} nodes are not used.`;
}

/** max_x |g_k − g_{k−1}| on the plot's 601-point grid (the step's correction), from the cache. */
function correction(
  id: string,
  result: Result,
  k: number,
  data: ActiveData,
  key: string,
): number | null {
  if (k < 1) return null;
  const [a, b] = data.domain;
  const g1 = stepFunction(id, result, k, data),
    g0 = stepFunction(id, result, k - 1, data);
  if (!g1 || !g0) return null;
  const s1 = sampled(`${key}|${k}`, g1, a, b),
    s0 = sampled(`${key}|${k - 1}`, g0, a, b);
  let m = 0;
  for (let i = 0; i < s1.length; i++) {
    const d = Math.abs(s1[i] - s0[i]);
    if (!Number.isFinite(d)) return Infinity;
    if (d > m) m = d;
  }
  return m;
}

/**
 * The MethodCard's "this step" numbers. Node-adding methods: the step is the node it added,
 * so the card's x_k and f(x_k) are that node and its value, then three numbers of the step
 * (a full 3 × 2 grid): the quantity it computed, the size of its correction and the error.
 * Chebyshev and the splines show their numbers in the filled-in rule instead (their steps add
 * a term or a stage, not a node).
 */
function cardFor(
  id: string,
  result: Result,
  k: number,
  data: ActiveData,
  key: string,
): { step?: Step; quantities: { tex: string; key: string }[] } {
  const trace = result.trace;
  const s = trace[k];
  const kind = methodKind(id);
  if (s && kind === 'aaa') {
    // The support point zₖ this step chose, the error AAA drives down on the samples, the
    // smallest singular value of the Loewner matrix, and the error against f.
    const node = nums(s.info.node);
    return {
      step: {
        ...s,
        x: node.length === 2 ? node[0] : (null as unknown as number),
        fun: node.length === 2 ? node[1] : null,
        info: { ...s.info, __err: s.fun },
      },
      quantities: [
        { tex: '\\max_Z|f - r_k|', key: 'info.sample_error' },
        { tex: '\\sigma_{\\min}', key: 'info.sigma_min' },
        { tex: '\\max|r_k - f|', key: 'info.__err' },
      ],
    };
  }
  if (s && kind === 'blend') {
    const win = nums(s.info.window);
    return {
      step: {
        ...s,
        x: nums(s.info.node)[0],
        fun: nums(s.info.node)[1],
        info: {
          ...s.info,
          __win: win.length === 2 ? `{${win[0]}, …, ${win[1]}}` : null,
          __err: s.fun,
        },
      },
      quantities: [
        { tex: 'w_k', key: 'info.weight' },
        { tex: 'J_k', key: 'info.__win' },
        { tex: '\\max|r - f|', key: 'info.__err' },
      ],
    };
  }
  if (!s || (kind !== 'nodewise' && kind !== 'neville')) return { quantities: [] };
  const [a, b] = data.domain;
  let coef: number | null;
  let coefTex: string;
  let delta: number | null;
  let deltaTex = '\\max|p_k{-}p_{k-1}|';
  if (id === 'newton_divided_differences' || id === 'barycentric') {
    coef = nums(s.x)[k] ?? null;
    coefTex = id === 'barycentric' ? 'w_k' : 'f[x_0,\\dots,x_k]';
    delta = correction(id, result, k, data, key);
  } else if (id === 'neville') {
    coef = s.x as number;
    coefTex = 'P_{0..k}(x^\\ast)';
    const prev = trace[k - 1]?.x;
    delta = k > 0 && typeof prev === 'number' ? Math.abs((s.x as number) - prev) : null;
    deltaTex = '|Q_{k,k}{-}Q_{k-1,k-1}|';
  } else {
    // Lagrange: the size of the basis polynomial this step adds, and of the term y_k ℓ_k.
    const lk =
      k < data.x.length ? sampled(`basis|${data.id}|${k}`, basisFn(data.x, k), a, b, 401) : null;
    coef = lk ? lk.reduce((m, v) => Math.max(m, Math.abs(v)), 0) : null;
    coefTex = '\\max|\\ell_k|';
    delta = correction(id, result, k, data, key);
  }
  const quantities = [
    { tex: coefTex, key: 'info.__coef' },
    { tex: deltaTex, key: 'info.__delta' },
    { tex: '\\max|p_k - f|', key: 'info.__err' },
  ];
  return {
    step: {
      ...s,
      x: data.x[k],
      fun: data.y[k],
      info: { ...s.info, __coef: coef, __delta: delta, __err: s.fun },
    },
    quantities,
  };
}

function StepsTable({
  methodId,
  steps,
  k,
  x,
  y,
  onSelect,
}: {
  methodId: string;
  steps: readonly Step[];
  k: number;
  x: readonly number[];
  y: readonly number[];
  onSelect: (k: number) => void;
}) {
  const kind = methodKind(methodId);
  const cell = (v: number | null | undefined) => sigFixed(v ?? null, 5);
  const cols: TableColumn<Step>[] = [
    { key: 'k', ...mathHead('k'), value: (s) => s.k, width: '2.2rem' },
  ];
  if (kind === 'nodewise' || kind === 'neville') {
    cols.push(
      { key: 'x', ...mathHead('x_k'), value: (s) => cell(x[s.k]) },
      { key: 'y', ...mathHead('y_k'), value: (s) => cell(y[s.k]) },
    );
    if (methodId === 'newton_divided_differences')
      cols.push({ key: 'c', ...mathHead('f[x_0..x_k]'), value: (s) => cell(nums(s.x)[s.k]) });
    if (methodId === 'barycentric')
      cols.push({ key: 'w', ...mathHead('w_k'), value: (s) => cell(nums(s.x)[s.k]) });
    if (methodId === 'neville')
      cols.push({ key: 'q', ...mathHead('P_{0..k}(x^\\ast)'), value: (s) => cell(s.x as number) });
  } else if (kind === 'chebyshev') {
    cols.push({ key: 'c', ...mathHead('c_k'), value: (s) => cell(nums(s.x)[s.k]) });
  } else if (kind === 'aaa') {
    cols.push(
      {
        key: 'z',
        ...mathHead('z_k'),
        value: (s) => (nums(s.info.node).length ? cell(nums(s.info.node)[0]) : '—'),
      },
      {
        key: 'se',
        ...mathHead('\\max_Z|f - r_k|', 'max over the samples of |f − r k|'),
        value: (s) => sig(Number(s.info.sample_error), 3),
      },
      // σ_min is on the MethodCard; the count is of the certified real poles in [a, b].
      {
        key: 'np',
        header: 'Poles',
        label: 'certified real poles in [a, b]',
        align: 'right',
        value: (s) => String(s.info.n_interval_poles ?? 0),
      },
    );
  } else if (kind === 'blend') {
    cols.push(
      { key: 'x', ...mathHead('x_k'), value: (s) => cell(nums(s.info.node)[0]) },
      { key: 'y', ...mathHead('y_k'), value: (s) => cell(nums(s.info.node)[1]) },
      { key: 'w', ...mathHead('w_k'), value: (s) => cell(s.info.weight as number) },
    );
  } else {
    cols.push({
      key: 'stage',
      header: 'Stage',
      align: 'left',
      mono: false,
      value: (s) => STAGE_LABEL[s.info.stage as string] ?? String(s.info.stage),
    });
  }
  const errTex =
    kind === 'aaa' ? '\\max|r_k - f|' : kind === 'blend' ? '\\max|r - f|' : '\\max|p_k - f|';
  cols.push({
    key: 'e',
    ...mathHead(errTex),
    value: (s) => (s.fun === null ? '—' : sig(s.fun, 3)),
  });
  return (
    <TableView
      maxHeight="none"
      columns={cols}
      rows={steps}
      highlight={k}
      futureAfter={k}
      onSelect={onSelect}
      ariaLabel="Steps of the focused method"
      rowKey={(s) => s.k}
    />
  );
}
