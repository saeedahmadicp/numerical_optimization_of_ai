/**
 * Root finding lab: f(x) = 0 with the nine bracketing and seven open methods of
 * `numopt.roots` (TS ports in src/methods/roots, parity-tested against the Python fixtures).
 *
 * Stage: the graph of f with each method's current step drawn from `Step.info` (bracket and
 * the discarded part, chords, tangents, hyperbolas, parabolas, inverse parabolas, …), the
 * bracket ends a, b and the start x₀ as draggable handles, and below it the iterates on a
 * signed log-scale number line of xₖ − x⋆ (one tick per correct digit). Insights: the error
 * |xₖ − x⋆|, the residual |f(xₖ)| or the bracket width per iteration, the MethodCard with the
 * update rule filled in for the current step, and the iteration table.
 */
import './setup';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { listMethods, type RegisteredMethod } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Result } from '../../core/types';
import type { RootProblem } from '../../problems/roots';
import { codecs, useUrlState } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import { sig, sci } from '../../core/format';
import {
  Formula,
  Icon,
  NumberField,
  PlaybackBar,
  SegmentedControl,
  TabPanel,
  Tabs,
} from '../../ui/components';
import { ConvergenceChart, IterationTable, type ConvergenceSeries } from '../../viz';
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
  tupleCodec,
  useLabRuns,
  useMethodSelection,
  useProblemState,
  type MethodSelection,
} from '../_shell';
import { KAxisControl, StartPointHint } from '../_shell/blocks';
import { useKAxis } from '../_shell/labState';
import { BRACKETING } from './geometry';
import { RootsPlot, type PlotRun } from './RootsPlot';
import { NumberLine } from './NumberLine';
import { CARD_QUANTITIES, cardStep, filledRule } from './rules';
import { rootColumns } from './columns';
import { estimateOf, shownStep } from './estimate';
import { DEFAULT_SELECTION, PRESETS } from './presets';
import shellStyles from '../_shell/LabShell.module.css';
import styles from './RootsLab.module.css';

void preloadKatex();

type Metric = 'err' | 'res' | 'width';
type View = 'f' | 'cobweb';
const METRIC_CODEC = codecs.oneOf<Metric>(['err', 'res', 'width']);
const VIEW_CODEC = codecs.oneOf<View>(['f', 'cobweb']);
const BR_CODEC = tupleCodec(2);
const X0_CODEC = tupleCodec(1);

/** Rates the theory guarantees at a simple root (annotated only for converged runs). */
const PROVEN_RATE: Record<string, string> = {
  newton: 'quadratic',
  halley: 'cubic',
  steffensen: 'quadratic',
  secant: 'order φ',
};

/** The root of the problem a run is measured against: the one nearest its final iterate. */
function targetOf(result: Result, roots: readonly number[]): number | null {
  if (!roots.length || !result.trace.length) return null;
  let xf = result.x as number;
  if (typeof xf !== 'number' || !Number.isFinite(xf)) {
    const finite = result.trace.map((s) => s.x as number).filter(Number.isFinite);
    if (!finite.length) return null;
    xf = finite[finite.length - 1];
  }
  return roots.reduce((best, r) => (Math.abs(r - xf) < Math.abs(best - xf) ? r : best));
}

const isBracketing = (m: RegisteredMethod) => BRACKETING.has(m.spec.id);

/** Methods whose estimate is a bracket end (`info.best`), not the trial point x_k. */
const KEEPS_BEST = new Set(['brent', 'chandrupatla', 'itp']);

export default function RootsLab() {
  const lab = getLab('roots')!;
  const problems = useMemo(() => listProblems<RootProblem>('roots'), []);
  const methods = useMemo(() => listMethods('roots'), []);

  const [problem, setProblem] = useProblemState(problems, 'cubic', ['x0', 'br']);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const defaultBr = useMemo(() => [...problem.bracket], [problem]);
  const defaultX0 = useMemo(() => [problem.x0], [problem]);
  const [brUrl, setBrUrl] = useUrlState<number[]>('br', defaultBr, BR_CODEC);
  const [x0Url, setX0Url] = useUrlState<number[]>('x0', defaultX0, X0_CODEC);
  const [metric, setMetric] = useUrlState<Metric>('y', 'err', METRIC_CODEC);
  const [viewUrl, setView] = useUrlState<View>('v', 'f', VIEW_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [focusId, setFocusId] = useState<string | null>(null);
  // The card's rule shrinks to fit its box; measured before the KaTeX fonts arrive it is too
  // narrow and never re-measured (a paused, reduced-motion page). Remount the card once the
  // fonts are ready.
  const [fontsReady, setFontsReady] = useState(false);
  useEffect(() => {
    let live = true;
    document.fonts?.ready.then(() => live && setFontsReady(true));
    return () => {
      live = false;
    };
  }, []);
  // While a handle is dragged the runs follow it live; the URL is written on release.
  const [live, setLive] = useState<{ br?: [number, number]; x0?: number } | null>(null);

  const bracket: [number, number] = live?.br ?? [brUrl[0], brUrl[1]];
  const x0 = live?.x0 ?? x0Url[0];
  const [ba, bb] = bracket;
  const options = useMemo(() => ({ bracket: [ba, bb] as [number, number], x0 }), [ba, bb, x0]);
  const runs = useLabRuns(problem, selection, options);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const hasBracketing = selection.some((s) => BRACKETING.has(s.id));
  const hasOpen = selection.some((s) => !BRACKETING.has(s.id));
  const fixedPoint = runs.find((r) => r.sel.id === 'fixed_point');
  const view: View = viewUrl === 'cobweb' && fixedPoint ? 'cobweb' : 'f';

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const targets = useMemo(
    () => runs.map((r) => targetOf(r.result, problem.roots)),
    [runs, problem],
  );
  // Dragging a bracket end shows step 0 (the bracket being set); dragging x₀ shows the whole
  // trail, so each start's destination is visible while the handle moves.
  const liveT = live?.br ? 0 : Infinity;
  const tOf = (i: number) => (live ? liveT : player.localT(i));

  const onBracket = useCallback(
    (b: [number, number], commit: boolean) => {
      if (commit) {
        setLive(null);
        setBrUrl(b);
      } else setLive({ br: b });
    },
    [setBrUrl],
  );
  const onX0 = useCallback(
    (v: number, commit: boolean) => {
      if (commit) {
        setLive(null);
        setX0Url([v]);
      } else setLive({ x0: v });
    },
    [setX0Url],
  );

  const fa = problem.plot(ba),
    fb = problem.plot(bb);
  const signBad = hasBracketing && Math.sign(fa) * Math.sign(fb) > 0;

  // ── Stage ─────────────────────────────────────────────────────────────────────────────
  const plotRuns: PlotRun[] = runs.map((r, i) => ({
    methodId: r.sel.id,
    name: r.method.spec.name,
    slot: r.sel.slot,
    trace: r.result.trace,
    t: tOf(i),
    focused: r === focus && runs.length > 1,
    converged: r.result.converged,
    target: targets[i],
  }));
  const lineRuns = plotRuns.map((r) => ({ ...r, focused: r.focused }));

  const summary = runs
    .map((r) => {
      const st = r.error ? `could not run (${r.error})` : r.result.message;
      return `${r.method.spec.name}: ${r.result.converged ? 'converged' : 'stopped'} after ${r.result.nIter} iterations at x = ${sig(r.result.x as number, 6)} (${st})`;
    })
    .join('. ');
  const ariaLabel = `Graph of ${problem.name} on [${sig(problem.domain[0], 3)}, ${sig(problem.domain[1], 3)}] with the current step of each method. ${summary}.`;

  // ── Convergence ───────────────────────────────────────────────────────────────────────
  const { series, floorNote } = useMemo(() => {
    let note: string | null = null;
    let zero = false;
    const out: ConvergenceSeries[] = [];
    runs.forEach((r, i) => {
      const target = targets[i];
      const tr = r.result.trace;
      let values: (number | null)[];
      if (metric === 'err') {
        if (target === null) return;
        const floor = Number.EPSILON * Math.max(1, Math.abs(target)) * 0.25;
        values = tr.map((s) => {
          const e = Math.abs(estimateOf(s) - target);
          if (e === 0) zero = true;
          return Number.isFinite(e) ? Math.max(e, floor) : null;
        });
      } else if (metric === 'res') {
        values = tr.map((s) => {
          const xe = estimateOf(s);
          const fe = xe === s.x ? s.fun : problem.f(xe);
          if (fe === 0) note = 'f(xₖ) = 0 exactly is drawn at 10⁻³⁰⁰.';
          return fe === null || !Number.isFinite(fe) ? null : Math.max(Math.abs(fe), 1e-300);
        });
      } else {
        if (!BRACKETING.has(r.sel.id)) return;
        values = tr.map((s) => {
          const b = s.info.new_bracket as number[] | undefined;
          return b && b[1] > b[0] ? b[1] - b[0] : null;
        });
      }
      const simple =
        target !== null &&
        Math.abs(problem.grad(target)) > 1e-8 * Math.max(1, Math.abs(problem.hess(target)));
      out.push({
        label: r.method.spec.name,
        slot: r.sel.slot,
        values,
        end: r.result.converged ? 'converged' : 'stopped',
        count: r.result.nIter,
        rate: metric === 'err' && r.result.converged && simple ? PROVEN_RATE[r.sel.id] : undefined,
      });
    });
    if (zero) note = 'An exact hit of x⋆ is drawn at ¼ ulp.';
    const best = runs.filter((r) => KEEPS_BEST.has(r.sel.id)).map((r) => r.method.spec.name);
    if (best.length && metric !== 'width') {
      const who = best.map((n) => n.replace(/ \(.*\)$/, '')).join(', ');
      const bestNote = `${who}: error of the best bracket end x̂ₖ, the value returned (the last trial point is a tolerance probe).`;
      note = note ? `${bestNote} ${note}` : bestNote;
    }
    return { series: out, floorNote: note as string | null };
  }, [runs, targets, metric, problem]);
  const [logK, setKAxis] = useKAxis(traces.map((t) => t.length));
  const openOnly = metric === 'width' && series.length === 0 && runs.length > 0;

  // ── Method card ───────────────────────────────────────────────────────────────────────
  // The step the figure shows (it switches halfway through the cross-fade, see estimate.ts).
  const focusK = focus ? Math.min(focus.result.trace.length - 1, shownStep(tOf(focusIndex))) : 0;
  const focusStep = focus ? cardStep(focus.sel.id, focus.result.trace, focusK) : undefined;
  const cardMethod = useMemo(() => {
    if (!focus) return null;
    const doc = focus.method.doc;
    if (!doc) return focus.method;
    const rule = filledRule(focus.sel.id, focusStep) ?? doc.rule;
    const quantities = CARD_QUANTITIES[focus.sel.id] ?? doc.quantities;
    return { ...focus.method, doc: { ...doc, rule, quantities } };
  }, [focus, focusStep]);
  const focusBracketing = focus ? isBracketing(focus.method) : false;
  const focusTarget = targets[focusIndex] ?? null;
  const focusBest = focus ? KEEPS_BEST.has(focus.sel.id) : false;
  const columns = useMemo(
    () => rootColumns(focusTarget, focusBracketing, focusBest),
    [focusTarget, focusBracketing, focusBest],
  );

  const setBracketField = (i: 0 | 1) => (v: number) => onBracket(i === 0 ? [v, bb] : [ba, v], true);

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
              onChange: setFocusId,
            }
          : undefined
      }
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          {/* Problem data: the bracket the bracketing methods start from, before the methods
              (the rail order of every lab: problem, its data, methods, start point). */}
          {hasBracketing && (
            <RailSection
              title={
                // The shared section heading; the bracket stays math (caps would print [A, B]).
                <h2 className={shellStyles.sectionTitle}>
                  Bracket{' '}
                  <span className={styles.titleMath}>
                    <Formula tex="[a, b]" />
                  </span>
                </h2>
              }
            >
              <div className={styles.pair}>
                <NumberField
                  label="Bracket end a"
                  prefix="a"
                  value={ba}
                  onChange={setBracketField(0)}
                />
                <NumberField
                  label="Bracket end b"
                  prefix="b"
                  value={bb}
                  onChange={setBracketField(1)}
                />
              </div>
              <p className={styles.signs} data-bad={signBad || undefined}>
                <Formula tex={`f(a) = ${texNum(fa)},\\; f(b) = ${texNum(fb)}`} />
                {signBad ? ' — no sign change' : ''}
              </p>
              {!hasOpen && (
                <p className={styles.hint}>
                  <Icon name="crosshair" size={13} />
                  <span>Drag a or b under the plot to move the bracket</span>
                </p>
              )}
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
            <MethodSlots
              available={methods}
              value={selection}
              onChange={(v: MethodSelection[]) => setSelection(v)}
            />
          </RailSection>
          {hasOpen && (
            <RailSection title="Start point">
              <div className={styles.field}>
                <NumberField
                  label="Start point x0"
                  prefix="x₀"
                  value={x0}
                  onChange={(v) => onX0(v, true)}
                />
              </div>
              {/* The shared start-point hint; with a bracket it also names the ends a, b. */}
              <p className={styles.hint}>
                {hasBracketing ? (
                  <>
                    <Icon name="crosshair" size={13} />
                    <span>Click the plot to set x₀ · drag a, b or x₀ to move them</span>
                  </>
                ) : (
                  <StartPointHint variable="x₀" drag />
                )}
              </p>
            </RailSection>
          )}
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
      stageToolbar={
        fixedPoint ? (
          <SegmentedControl
            label="View"
            value={view}
            onChange={setView}
            options={[
              { value: 'f', label: <Formula tex="f(x)" />, ariaLabel: 'Graph of f' },
              { value: 'cobweb', label: 'Cobweb', ariaLabel: 'Cobweb of g(x) = x − λ f(x)' },
            ]}
          />
        ) : undefined
      }
      stage={
        <div className={styles.stageBody}>
          <div className={styles.plotWrap}>
            <RootsPlot
              f={problem.plot}
              domain={problem.domain}
              viewKey={problem.id}
              roots={problem.roots}
              runs={
                view === 'cobweb' ? plotRuns.filter((r) => r.methodId === 'fixed_point') : plotRuns
              }
              bracket={hasBracketing ? bracket : null}
              x0={hasOpen ? x0 : null}
              onBracket={onBracket}
              onX0={onX0}
              mode={view}
              lam={Number(
                fixedPoint?.result.trace[1]?.info.lam ?? fixedPoint?.sel.params.lam ?? 0.1,
              )}
              ariaLabel={ariaLabel}
            />
            {signBad && (
              <p className={styles.banner} role="status">
                No sign change on [a, b]: f(a) = {sig(fa, 3)} and f(b) = {sig(fb, 3)}. Drag an end
                across a root.
              </p>
            )}
            {view === 'cobweb' && (
              <p className={styles.banner} data-place="bottom" role="note">
                Cobweb of g(x) = x − λ f(x): a root of f is where g meets the diagonal y = x.
              </p>
            )}
          </div>
          {runs.length > 0 && view === 'f' && <NumberLine runs={lineRuns} />}
        </div>
      }
      playback={
        <PlaybackBar
          // The readout names the step the figure shows; while x₀ is dragged that is every step.
          player={
            live?.x0 !== undefined
              ? { ...player, t: player.maxK, k: player.maxK }
              : { ...player, k: Math.min(player.maxK, shownStep(player.t)) }
          }
          slots={runs.map((r) => r.sel.slot)}
        />
      }
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          // Room for the controls on a second header row in a narrow column.
          height: 316,
          actions: (
            // Wraps onto a second row (right-aligned) when the column is too narrow for both.
            <div className={styles.chartActions}>
              <KAxisControl logX={logK} onChange={setKAxis} />
              <SegmentedControl
                label="Convergence measure"
                value={metric}
                onChange={setMetric}
                options={[
                  {
                    value: 'err',
                    label: <Formula tex="|x_k - x^\star|" />,
                    ariaLabel: 'Error to the root',
                  },
                  { value: 'res', label: <Formula tex="|f(x_k)|" />, ariaLabel: 'Residual' },
                  { value: 'width', label: <Formula tex="b - a" />, ariaLabel: 'Bracket width' },
                ]}
              />
            </div>
          ),
          content: (
            <div className={styles.chartPad}>
              <ConvergenceChart
                series={series}
                t={live ? liveT : player.t}
                logX={logK}
                yLabel={metric === 'err' ? '|xₖ − x⋆|' : metric === 'res' ? '|f(xₖ)|' : 'b − a'}
                yName={
                  metric === 'err'
                    ? [
                        { t: '|', style: 'main' },
                        { t: 'x', style: 'italic' },
                        { t: 'k', style: 'italic', script: 'sub' },
                        { t: ' − ', style: 'main' },
                        { t: 'x', style: 'italic' },
                        { t: '⋆', style: 'main', script: 'sup' },
                        { t: '|', style: 'main' },
                      ]
                    : metric === 'res'
                      ? [
                          { t: '|', style: 'main' },
                          { t: 'f', style: 'italic' },
                          { t: '(', style: 'main' },
                          { t: 'x', style: 'italic' },
                          { t: 'k', style: 'italic', script: 'sub' },
                          { t: ')|', style: 'main' },
                        ]
                      : [
                          { t: 'b', style: 'italic' },
                          { t: ' − ', style: 'main' },
                          { t: 'a', style: 'italic' },
                        ]
                }
                onSeek={(k) => {
                  player.pause();
                  player.seek(k);
                }}
                ariaLabel={`Convergence of ${runs.map((r) => r.method.spec.name).join(', ')}`}
              />
              {(floorNote || openOnly) && (
                <p className={styles.chartNote}>
                  {openOnly
                    ? 'Open methods keep no bracket: choose |xₖ − x⋆| or |f(xₖ)|.'
                    : floorNote}
                </p>
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
              value={tab}
              onChange={setTab}
              idBase="roots-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content:
            focus && cardMethod ? (
              <TabPanel idBase="roots-details" id={tab} focusable={false}>
                {tab === 'method' ? (
                  <MethodCard
                    key={fontsReady ? 'fonts' : 'pending'}
                    method={cardMethod}
                    slot={focus.sel.slot}
                    step={focusStep}
                    result={focus.result}
                    error={focus.error}
                    call={{
                      problem: problem.id,
                      options: focusBracketing ? { bracket: [ba, bb] } : { x0 },
                      params: focus.sel.params,
                    }}
                  />
                ) : (
                  <IterationTable
                    steps={focus.result.trace}
                    columns={columns}
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
        },
      ]}
    />
  );
}

/** A value of f for the rail's sign readout (TeX). */
function texNum(v: number): string {
  if (!Number.isFinite(v)) return '\\text{—}';
  const s = Math.abs(v) < 1e-3 && v !== 0 ? sci(v, 2) : sig(v, 3);
  return s
    .replace('−', '-')
    .replace(/×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/, (_, e: string) => `\\times 10^{${supToNum(e)}}`);
}

function supToNum(s: string): string {
  const map: Record<string, string> = {
    '⁻': '-',
    '⁰': '0',
    '¹': '1',
    '²': '2',
    '³': '3',
    '⁴': '4',
    '⁵': '5',
    '⁶': '6',
    '⁷': '7',
    '⁸': '8',
    '⁹': '9',
  };
  return s
    .split('')
    .map((c) => map[c] ?? c)
    .join('');
}
