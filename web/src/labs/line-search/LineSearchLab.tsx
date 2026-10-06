/**
 * Line-search lab: one search from 𝐱₀ along 𝐩, seen twice — on the contour plot as a ray with
 * the trial points, and as the line function φ(α) = f(𝐱₀ + α𝐩) with the Armijo line, the
 * Goldstein cone, the curvature band, the acceptable intervals of every method and its trials in
 * order. Methods: the five registered line-search demos (src/methods/line_search/methods.ts).
 */
import './setup';
import { useCallback, useMemo, useRef, useState } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Problem2D, Step } from '../../core/types';
import type { LineSearchKind } from '../../methods/line_search/methods';
import { useUrlState } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import {
  Button,
  Formula,
  PlaybackBar,
  SegmentedControl,
  TabPanel,
  Tabs,
} from '../../ui/components';
import { MAX_SERIES } from '../../ui/colors';
import {
  Contour2D,
  ConvergenceChart,
  TableView,
  mathMain as m,
  mathSans,
  mathSub as sub,
  mathSup as sup,
  mathVar as v,
  useElementSize,
  type TableColumn,
  type View2D,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  StartPointFields,
  useLabRuns,
  useMethodSelection,
  useProblemState,
  useStartPoint,
} from '../_shell';
import {
  DEFAULT_FOCUS,
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  DIRECTION_CODEC,
  KEYS,
  METRIC_CODEC,
  PRESETS,
  WINDOW_CODEC,
  type Direction,
  type Metric,
} from './config';
import { shortName, trialOf, type WindowMode } from './geometry';
import { buildModel } from './model';
import { fmtNum, num, vecNum } from './format';
import { PhiPanel, type PhiRun } from './PhiPanel';
import { drawPlane, searchFrame } from './plane';
import { TrialCheck } from './TrialCheck';
import styles from './LineSearchLab.module.css';

void preloadKatex();

const P_TEX: Record<Direction, string> = {
  steepest: '\\mathbf p = -\\nabla f(\\mathbf x_0)',
  newton: '\\mathbf p = -\\nabla^2 f(\\mathbf x_0)^{-1}\\,\\nabla f(\\mathbf x_0)',
};

/** The tests each kind records (Step.info.conditions), in the order of its acceptance rule. */
const TESTS: Record<LineSearchKind, { keys: string[]; header: string; label: string }> = {
  backtracking: { keys: ['armijo'], header: 'Armijo', label: 'Armijo' },
  strong_wolfe: {
    keys: ['armijo', 'strong_curvature'],
    header: 'Tests',
    label: 'Armijo, then strong curvature',
  },
  weak_wolfe: { keys: ['armijo', 'curvature'], header: 'Tests', label: 'Armijo, then curvature' },
  goldstein: {
    keys: ['armijo', 'goldstein_lower'],
    header: 'Tests',
    label: 'Goldstein upper line, then lower line',
  },
  exact_quadratic: { keys: ['decrease'], header: 'Decrease', label: 'Decrease' },
};

const mark = (b: boolean | null | undefined) => (b === true ? '✓' : b === false ? '✗' : '–');
const short = (v: number) => (Number.isFinite(v) ? num(v, 3) : '∞');

function tableColumns(kind: LineSearchKind, trace: readonly Step[]): TableColumn<Step>[] {
  const tests = TESTS[kind];
  const slope = trace.some((s) => s.k > 0 && trialOf(s).dphi !== null);
  return [
    { key: 'k', tex: 'k', value: (s) => s.k, width: '2rem' },
    { key: 'alpha', tex: '\\alpha_k', value: (s) => fmtNum(trialOf(s).alpha) },
    { key: 'phi', tex: '\\varphi(\\alpha_k)', value: (s) => num(trialOf(s).phi, 4) },
    ...(slope
      ? [
          {
            key: 'dphi',
            tex: "\\varphi'(\\alpha_k)",
            value: (s: Step) => (s.k === 0 ? num(trialOf(s).dphi0, 4) : num(trialOf(s).dphi, 4)),
          },
        ]
      : []),
    {
      key: 'tests',
      header: <span title={tests.label}>{tests.header}</span>,
      label: tests.label,
      align: 'center' as const,
      value: (s: Step) =>
        s.k === 0
          ? ''
          : tests.keys
              .map((k) => mark((s.info.conditions as Record<string, boolean | null>)[k]))
              .join(' '),
    },
    ...(kind === 'backtracking' || kind === 'exact_quadratic'
      ? []
      : [
          {
            // The phase and the interval share one cell: "zoom [0.1, 0.19]".
            key: 'interval',
            header: (
              <>
                Phase <Formula tex="[\ell,\, u]" />
              </>
            ),
            label: 'Phase, and the interval the trial was chosen from',
            align: 'left' as const,
            mono: false,
            value: (s: Step) => {
              const t = trialOf(s);
              const iv = t.interval;
              return (
                <span className={styles.phaseCell}>
                  <span className={styles.phaseWord}>{t.phase}</span>
                  {iv && (
                    <span className={styles.phaseIv}>
                      [{short(iv[0])}, {short(iv[1])}]
                    </span>
                  )}
                </span>
              );
            },
          },
        ]),
  ];
}

export default function LineSearchLab() {
  const lab = getLab('line-search')!;
  const problems = useMemo(
    () => listProblems<Problem2D>('unconstrained').filter((p) => p.dim === 2),
    [],
  );
  const methods = useMemo(() => listMethods('line_search'), []);
  // The exact step's c₁ feeds only a diagnostic Armijo report (its test is φ(α) ≤ φ(0)), so the
  // rail does not offer it as a control.
  const slotMethods = useMemo(
    () =>
      methods.map((q) =>
        q.spec.id === 'exact_quadratic'
          ? { ...q, spec: { ...q.spec, params: q.spec.params.filter((r) => r.name !== 'c1') } }
          : q,
      ),
    [methods],
  );
  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [x0, setX0] = useStartPoint(problem.x0 as number[] | undefined, 2);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [direction, setDirection] = useUrlState<Direction>(
    KEYS.direction,
    'steepest',
    DIRECTION_CODEC,
  );
  const [mode, setMode] = useUrlState<WindowMode>(KEYS.window, 'near', WINDOW_CODEC);
  const [focusKey, setFocusKey] = useUrlState(KEYS.focus, DEFAULT_FOCUS);
  const [metric, setMetric] = useUrlState<Metric>(KEYS.metric, 'gap', METRIC_CODEC);
  const [tab, setTab] = useState<'method' | 'trials'>('method');
  const [probe, setProbe] = useState<number | null>(null);
  const [narrowView, setNarrowView] = useState<'phi' | 'slope' | 'plane'>('phi');
  const [wholeDomain, setWholeDomain] = useState(false);

  // The direction is one choice for the whole lab: every method searches along the same 𝐩.
  const runSel = useMemo(
    () => selection.map((s) => ({ ...s, params: { ...s.params, direction } })),
    [selection, direction],
  );
  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]] }), [x0]);
  const runs = useLabRuns(problem, runSel, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const focus = runs.find((r) => r.sel.id === focusKey) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus ? player.localK(focusIndex) : 0;

  // ── The line, its window and the acceptable sets (one pure derivation) ──
  const model = useMemo(() => buildModel(problem, x0, runs, mode), [problem, x0, runs, mode]);
  const { line, p, phi0, dphi0, hi, alphaStar, samples, intervals, rayEnd, phiStar } = model;
  const pOk = line !== null;

  const phiRuns: PhiRun[] = runs
    .map((r, i) => ({
      id: r.sel.id,
      name: r.method.spec.name,
      slot: r.sel.slot,
      kind: r.sel.id as LineSearchKind,
      trace: r.result.trace,
      t: player.localT(i),
      intervals: intervals[i] ?? [],
    }))
    .filter((r) => r.trace.length > 0);

  // ── Interactions ──
  const hasAlpha0 = !!focus?.method.spec.params.some((q) => q.name === 'alpha0');
  const pickAlpha = (a: number) => {
    if (!focus) return;
    const spec = focus.method.spec.params.find((q) => q.name === 'alpha0');
    if (!spec) return;
    const val = Math.min(Number(spec.max ?? 100), Math.max(Number(spec.min ?? 1e-4), a));
    setSelection(
      selection.map((s) =>
        s.id === focus.sel.id ? { ...s, params: { ...s.params, alpha0: val } } : s,
      ),
    );
  };
  const pickStart = useCallback((q: [number, number]) => setX0(q), [setX0]);
  const f2 = useCallback((x: number, y: number) => problem.f([x, y]), [problem]);

  const steepest = useMemo(() => {
    const g = problem.grad([x0[0], x0[1]]);
    return [-g[0], -g[1]];
  }, [problem, x0]);
  const scene = {
    steepest,
    newton: direction === 'newton',
    hi,
    rayEnd,
    runs: phiRuns,
    focusId: focus?.sel.id ?? null,
    alphaStar: alphaStar && alphaStar.alpha <= rayEnd ? alphaStar.alpha : null,
    probe,
    ease: !player.reducedMotion,
  };
  const overlay = (ctx: CanvasRenderingContext2D, view: View2D) => {
    if (line) drawPlane(ctx, view, { ...scene, line });
  };

  // ── Convergence: each trial against the first local minimum of φ (or the relative slope) ──
  const series = runs.map((r) => ({
    label: shortName(r.method.spec.name),
    slot: r.sel.slot,
    end: (r.result.converged ? 'converged' : 'stopped') as 'converged' | 'stopped',
    count: r.result.nIter,
    values: r.result.trace.map((s) => {
      const t = trialOf(s);
      if (metric === 'slope') {
        const d = s.k === 0 ? t.dphi0 : t.dphi;
        return d === null || !Number.isFinite(d) ? null : Math.abs(d) / Math.abs(t.dphi0);
      }
      const val = s.k === 0 ? t.phi0 : t.phi;
      if (!Number.isFinite(val) || (s.k === 0 && val <= phiStar)) return null;
      return Math.max(val - phiStar, 1e-16 * Math.max(1, Math.abs(phiStar)));
    }),
  }));

  // ── Stage layout: side by side when wide, stacked when tall, one at a time on phones ──
  const stageRef = useRef<HTMLDivElement>(null);
  const size = useElementSize(stageRef);
  const narrow = size.width > 0 && size.width < 560;
  const wide = !narrow && size.width >= 1.3 * size.height && size.width >= 820;
  const lay = narrow ? 'single' : wide ? 'columns' : 'rows';

  const focusStep = focus?.result.trace[focusK];
  const methodNames = runs.map((r) => shortName(r.method.spec.name)).join(', ');
  const phiAria = line
    ? `φ(α) = f(x₀ + αp) on 0 ≤ α ≤ ${num(hi, 3)}, φ(0) = ${num(phi0, 4)}, φ′(0) = ${num(dphi0, 4)}. ` +
      runs
        .map(
          (r) =>
            `${shortName(r.method.spec.name)}: ${r.result.converged ? `accepted α = ${num(r.result.extra.alpha as number, 4)} after ${r.result.nIter} trials` : 'no step accepted'}`,
        )
        .join('; ') +
      '.'
    : 'No search direction.';

  const failure = !pOk
    ? (runs[0]?.result.message ?? 'No search direction at this start point.')
    : !(dphi0 < 0)
      ? `${direction === 'newton' ? 'The Newton direction' : '𝐩'} is not a descent direction here: φ′(0) = ${num(dphi0, 4)} ≥ 0${direction === 'newton' ? ', because ∇²f(𝐱₀) is not positive definite. Try steepest descent.' : '.'}`
      : !model.searched
        ? runs[0]?.result.message
        : null;

  // A short search would be a few pixels on the whole domain: frame it instead (a button
  // switches between the two views).
  const frame = useMemo(
    () => searchFrame(problem.domain as [[number, number], [number, number]], x0, p, hi),
    [problem.domain, x0, p, hi],
  );
  const framed = frame !== null && !wholeDomain;
  const plane = (
    <div className={styles.planeBox}>
      <Contour2D
        f={f2}
        domain={framed ? frame : problem.domain}
        cacheKey={problem.id}
        fMin={null}
        minima={(problem.minima ?? []) as [number, number][]}
        minimaLabels={false}
        start={[x0[0], x0[1]]}
        overlays={
          Number.isFinite(phi0)
            ? [
                {
                  kind: 'implicit',
                  g: (x, y) => problem.f([x, y]),
                  level: phi0,
                  cacheKey: `${problem.id}:${phi0}`,
                  dashed: true,
                  width: 1,
                  alpha: 0.55,
                },
              ]
            : []
        }
        overlay={overlay}
        onPick={pickStart}
        ariaLabel={`Contour plot of ${problem.name} with the search ray from x₀ = ${vecNum(x0, 3)} and the trial points of ${methodNames}.`}
      />
      {frame && (
        <Button
          size="sm"
          variant="secondary"
          className={styles.frameBtn}
          onClick={() => setWholeDomain(!wholeDomain)}
          aria-pressed={wholeDomain}
          title={
            wholeDomain
              ? 'Zoom the plot to the stretch of the ray the φ panel shows'
              : 'Show the whole domain of the problem'
          }
        >
          {wholeDomain ? 'Fit the search' : 'Whole domain'}
        </Button>
      )}
    </div>
  );
  const phiView =
    line && samples ? (
      <PhiPanel
        line={line}
        samples={samples}
        phi0={phi0}
        dphi0={dphi0}
        hi={hi}
        mode={mode}
        runs={phiRuns}
        focusId={focus?.sel.id ?? null}
        alphaStar={alphaStar}
        ease={!player.reducedMotion}
        probe={probe}
        onProbe={setProbe}
        onPickAlpha={hasAlpha0 ? pickAlpha : undefined}
        onFocus={setFocusKey}
        emphasis={lay === 'single' && narrowView === 'slope' ? 'slope' : 'phi'}
        ariaLabel={phiAria}
      />
    ) : (
      <div className={styles.phiEmpty} />
    );

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      focus={
        focus
          ? {
              runs: runs.map((r) => ({
                value: r.sel.id,
                slot: r.sel.slot,
                name: r.method.spec.name,
              })),
              value: focus.sel.id,
              onChange: setFocusKey,
            }
          : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection title="Search direction">
            <SegmentedControl
              label="Search direction"
              value={direction}
              onChange={setDirection}
              fullWidth
              options={[
                { value: 'steepest', label: 'Steepest descent' },
                { value: 'newton', label: 'Newton' },
              ]}
            />
            <div className={styles.pBox}>
              <Formula tex={P_TEX[direction]} fallback />
              {p && pOk && (
                <dl className={styles.pVals}>
                  <dt>
                    <Formula tex="\mathbf p" />
                  </dt>
                  <dd>{vecNum(p, 4)}</dd>
                  <dt>
                    <Formula tex="\varphi'(0)" />
                  </dt>
                  <dd>{num(dphi0, 4)}</dd>
                </dl>
              )}
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
            <MethodSlots
              available={slotMethods}
              value={selection}
              onChange={setSelection}
              hiddenParams={['direction']}
            />
          </RailSection>
          <RailSection title="Start point">
            <StartPointFields value={x0} onChange={setX0} variable="𝐱₀" />
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
        <div ref={stageRef} className={styles.stageGrid} data-layout={lay}>
          {lay !== 'single' && plane}
          <div className={styles.phiCell}>
            <div className={styles.phiTools}>
              {lay === 'single' && (
                <SegmentedControl
                  label="Panel"
                  value={narrowView}
                  onChange={setNarrowView}
                  options={[
                    { value: 'phi', label: <Formula tex="\varphi" />, ariaLabel: 'φ(α)' },
                    {
                      value: 'slope',
                      label: <Formula tex="\varphi'" />,
                      ariaLabel: 'φ(α) with a large φ′(α)',
                    },
                    { value: 'plane', label: 'Plane' },
                  ]}
                />
              )}
              {lay !== 'single' && (
                <span className={styles.phiTitle}>
                  <Formula tex="\varphi(\alpha) = f(\mathbf x_0 + \alpha\mathbf p)" />
                </span>
              )}
              <span className={styles.toolsEnd}>
                <SegmentedControl
                  label="α range of the φ panel"
                  value={mode}
                  onChange={setMode}
                  options={[
                    {
                      value: 'near',
                      label: lay === 'single' ? 'Near' : 'Near the steps',
                      tooltip: 'α up to 1.6 × the accepted steps',
                    },
                    {
                      value: 'all',
                      label: lay === 'single' ? 'All' : 'All trials',
                      tooltip: 'α up to the longest trial',
                    },
                  ]}
                />
              </span>
            </div>
            <div className={styles.phiBody}>
              {lay === 'single' && narrowView === 'plane' ? plane : phiView}
              {failure && !(lay === 'single' && narrowView === 'plane') && (
                <p className={styles.failure} role="note">
                  {failure}
                </p>
              )}
            </div>
          </div>
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'trial',
          title: 'This trial',
          height: 262,
          content: focus ? (
            <div className={styles.checkPad}>
              <TrialCheck
                name={focus.method.spec.name}
                slot={focus.sel.slot}
                kind={focus.sel.id as LineSearchKind}
                trace={focus.result.trace}
                k={focusK}
                result={focus.result}
              />
            </div>
          ) : null,
        },
        {
          id: 'convergence',
          title: metric === 'slope' ? 'Slope at each trial' : 'Gap to the first minimum of φ',
          height: 230,
          actions: (
            <SegmentedControl
              label="Error measure"
              value={metric}
              onChange={setMetric}
              options={[
                {
                  value: 'gap',
                  label: <Formula tex="\varphi - \varphi^\star" />,
                  ariaLabel: 'Gap to the minimum of φ',
                },
                {
                  value: 'slope',
                  label: <Formula tex="|\varphi'|" />,
                  ariaLabel: 'Relative slope |φ′(α)| / |φ′(0)|',
                },
              ]}
            />
          ),
          content: (
            <div className={styles.chartPad}>
              <ConvergenceChart
                series={series}
                t={player.t}
                xLabel="trial k"
                xName={[mathSans('trial '), v('k')]}
                yLabel={metric === 'slope' ? '|φ′(αₖ)| / |φ′(0)|' : 'φ(αₖ) − φ⋆'}
                yName={
                  metric === 'slope'
                    ? [
                        m('|'),
                        v('φ'),
                        sup('′'),
                        m('('),
                        v('α'),
                        sub('k', 'italic'),
                        m(')| / |'),
                        v('φ'),
                        sup('′'),
                        m('(0)|'),
                      ]
                    : [v('φ'), m('('), v('α'), sub('k', 'italic'), m(') − '), v('φ'), sup('⋆')]
                }
                onSeek={(k) => {
                  player.pause();
                  player.seek(k);
                }}
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
              idBase="ls-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'trials', label: 'Trials' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="ls-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <MethodCard
                  method={focus.method}
                  slot={focus.sel.slot}
                  step={focusStep}
                  result={focus.result}
                  error={focus.error}
                  call={{
                    problem: problem.id,
                    options: { x0: [x0[0], x0[1]] },
                    params: {
                      ...(selection.find((q) => q.id === focus.sel.id)?.params ?? {}),
                      ...(direction === 'newton' ? { direction } : {}),
                    },
                  }}
                />
              ) : (
                <div className={styles.trialsTable}>
                  <TableView
                    columns={tableColumns(focus.sel.id as LineSearchKind, focus.result.trace)}
                    rows={focus.result.trace}
                    highlight={focusK}
                    futureAfter={focusK}
                    onSelect={(k) => {
                      player.pause();
                      player.seek(k);
                    }}
                    ariaLabel={`${focus.method.spec.name} trials`}
                    maxHeight="100%"
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
