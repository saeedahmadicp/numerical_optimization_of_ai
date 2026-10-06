/**
 * 1-D minimization lab: golden section, Fibonacci, dichotomous and ternary search, successive
 * parabolic interpolation, Brent's localmin, Newton's method and minimum bracketing — the TS ports
 * of numopt.scalar.methods, replayed step by step on the scalar_min problems.
 *
 * URL state: p (problem), m (methods), br (bracket a,b), x0 (start), v (whole | zoom), y (measure).
 */
import './setup';
import { useMemo, useState } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { ScalarProblem } from '../../problems/scalar_min';
import { codecs, useUrlState, type Codec } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import { sci, sig } from '../../core/format';
import {
  Badge,
  Formula,
  Icon,
  NumberField,
  PlaybackBar,
  SegmentedControl,
  Swatch,
  TabPanel,
  Tabs,
} from '../../ui/components';
import { IterationTable, mathMain, mathVar, mathSub, mathSup, type MathRun } from '../../viz';
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
  URL_KEYS,
  useLabRuns,
  useMethodSelection,
  useProblemState,
} from '../_shell';
import { StartPointHint, StepBlock } from '../_shell/blocks';
import shellStyles from '../_shell/LabShell.module.css';
import {
  GUIDE_LABELS,
  GUIDE_RATES,
  kindOf,
  metricValues,
  nearestMinimizer,
  nextView,
  ratePerEvaluation,
  stepViews,
  usesStart,
  type Metric,
} from './geometry';
import { droppedEnds, isEpsilonStep, stepNote, texNum } from './stepNote';
import { presentProblem } from './problemCopy';
import { columnsFor } from './columns';
import { ScalarStage, type Handle, type StageRun, type ViewMode } from './ScalarStage';
import { RateChart, type RateGuide, type RateSeries } from './RateChart';
import { DEFAULT_PROBLEM, DEFAULT_SELECTION, PRESETS } from './presets';
import styles from './ScalarLab.module.css';

void preloadKatex();

const VIEW_CODEC = codecs.oneOf<ViewMode>(['follow', 'full']);
const METRIC_CODEC = codecs.oneOf<Metric>(['width', 'error']);
/** `br=a,b`: two finite numbers with a < b, else the problem's bracket. */
const BRACKET_CODEC: Codec<number[]> = {
  parse: (s) => {
    const xs = codecs.numbers.parse(s);
    return xs && xs.length === 2 && xs[0] < xs[1] ? xs : undefined;
  },
  format: codecs.numbers.format,
};
const X0_CODEC = codecs.number;

/**
 * Plain-language name of the next step for the stage badge. Elimination steps name the end(s)
 * the next bracket discards (Fibonacci's last, ε-step can discard both).
 */
function nextLabel(
  methodId: string,
  kind: string | null,
  ends: { left: boolean; right: boolean } | null,
  epsilon: boolean,
): string | null {
  if (!kind) return null;
  if (kindOf(methodId) === 'interval') {
    const what =
      ends && ends.left && ends.right
        ? 'drop both ends'
        : (ends ? ends.right : kind === 'right')
          ? 'drop the right end'
          : 'drop the left end';
    return epsilon ? `next: ε-step, ${what}` : `next: ${what}`;
  }
  const words: Record<string, string> = {
    parabolic: 'next: parabolic step',
    probe: 'next: probe δ from the vertex',
    golden: methodId === 'bracket_minimum' ? 'next: expand by φ' : 'next: golden step',
    bisect: 'next: bisect toward the lower end',
    newton: 'next: Newton step',
    gradient: 'next: gradient step (f″ ≤ 0)',
    parabolic_far: 'next: parabolic jump beyond c',
    limit: 'next: jump to the growth limit',
  };
  return words[kind] ?? `next: ${kind}`;
}

export default function ScalarLab() {
  const lab = getLab('scalar')!;
  const problems = useMemo(() => listProblems<ScalarProblem>('scalar_min'), []);
  const methods = useMemo(() => listMethods('scalar'), []);
  const shown = useMemo(() => problems.map(presentProblem), [problems]);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM, [URL_KEYS.start, 'br']);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [brUrl, setBr] = useUrlState<number[]>('br', problem.bracket, BRACKET_CODEC);
  const [x0Url, setX0] = useUrlState<number>(URL_KEYS.start, problem.x0, X0_CODEC);
  const [mode, setMode] = useUrlState<ViewMode>('v', 'follow', VIEW_CODEC);
  const [metricUrl, setMetric] = useUrlState<Metric>('y', 'width', METRIC_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [focusId, setFocusId] = useState<string | null>(null);
  // Live values while a handle is dragged (committed to the URL on release).
  const [draft, setDraft] = useState<{ h: Handle; v: number } | null>(null);

  const bracket: [number, number] = [
    draft?.h === 'a' ? draft.v : brUrl[0],
    draft?.h === 'b' ? draft.v : brUrl[1],
  ];
  const x0 = draft?.h === 'x0' ? draft.v : x0Url;
  const needsBracket = selection.some((s) => !usesStart(s.id));
  const needsStart = selection.some((s) => usesStart(s.id));

  const runOptions = useMemo(
    () => ({ bracket: [bracket[0], bracket[1]] as [number, number], x0 }),
    [bracket[0], bracket[1], x0], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const runs = useLabRuns(problem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const views = useMemo(() => runs.map((r) => stepViews(r.sel.id, r.result.trace)), [runs]);
  // Newton keeps no bracket: with only such runs the chart shows the error instead.
  const metric: Metric = views.some((v) => v[0]?.bracket) ? metricUrl : 'error';
  const stageRuns: StageRun[] = useMemo(
    () =>
      runs.map((r, i) => ({
        id: r.sel.id,
        name: r.method.spec.name,
        slot: r.sel.slot,
        views: views[i],
        result: r.result,
      })),
    [runs, views],
  );

  // The picked method; by default the first one that produced steps (a run that rejected its
  // input has nothing to draw, but stays selectable to read its error).
  const focus =
    runs.find((r) => r.sel.id === focusId) ??
    runs.find((r) => r.result.trace.length > 0) ??
    runs[0];
  const fi = focus ? runs.indexOf(focus) : 0;
  const times = runs.map((_, i) => player.localT(i));
  const kFocus = focus ? player.localK(fi) : 0;
  const focusView = views[fi]?.[kFocus];
  const focusStep = focus?.result.trace[kFocus];
  const next = focus ? nextView(focus.sel.id, focus.result.trace, kFocus) : null;

  // ── Convergence: bracket width (or error) against evaluations ─────────────────────────
  const series: RateSeries[] = useMemo(
    () =>
      runs.map((r, i) => {
        const v = views[i];
        const last = v[v.length - 1];
        const xStar = last ? nearestMinimizer(problem.minima, last.x) : null;
        return {
          label: r.method.spec.name,
          slot: r.sel.slot,
          n: v.map((s) => s.nEval),
          y: metricValues(v, metric, xStar),
          end: r.result.converged ? 'converged' : 'stopped',
          count: r.result.nIter,
          rate:
            metric === 'width' && kindOf(r.sel.id) !== 'bracketing' ? ratePerEvaluation(v) : null,
        };
      }),
    [runs, views, metric, problem],
  );
  const guides: RateGuide[] = useMemo(() => {
    if (metric !== 'width') return [];
    const seen = new Set<string>();
    const out: RateGuide[] = [];
    runs.forEach((r, i) => {
      const g = GUIDE_RATES[r.sel.id];
      const v0 = views[i][0];
      if (!g || !v0?.bracket || seen.has(g.key)) return;
      seen.add(g.key);
      out.push({
        rate: g.rate,
        n0: v0.nEval,
        y0: v0.bracket[1] - v0.bracket[0],
        label: GUIDE_LABELS[g.key],
      });
    });
    return out;
  }, [runs, views, metric]);

  const changeProblem = (id: string) => {
    setDraft(null);
    setProblem(id);
  };
  const onDrag = (h: Handle, v: number, final: boolean) => {
    if (!final) return setDraft({ h, v });
    setDraft(null);
    if (h === 'x0') setX0(v);
    else setBr(h === 'a' ? [v, brUrl[1]] : [brUrl[0], v]);
  };

  const seekTo = (k: number) => {
    player.pause();
    player.seek(k);
  };
  const seekEvaluations = (n: number) => {
    const v = views[fi];
    if (!v?.length) return;
    let k = 0;
    while (k + 1 < v.length && v[k + 1].nEval <= n) k++;
    seekTo(k);
  };

  // ── Text ───────────────────────────────────────────────────────────────────────────────
  const names = runs.map((r) => r.method.spec.name).join(', ');
  // Plain text only (no LaTeX), as in the other labs: a screen reader reads it verbatim.
  const graphOf = `Graph of ${problem.name} on [${sig(problem.domain[0], 3)}, ${sig(problem.domain[1], 3)}]`;
  const stageLabel = focusView
    ? `${graphOf} with ${names}${mode === 'follow' ? ', zoomed to the bracket' : ''}. ${focus!.method.spec.name} at step ${kFocus}: ${
        focusView.bracket
          ? `bracket [${sig(focusView.bracket[0], 6)}, ${sig(focusView.bracket[1], 6)}], width ${sci(
              focusView.bracket[1] - focusView.bracket[0],
              3,
            )}`
          : `x = ${sig(focusView.x, 8)}`
      }, f = ${sig(focusView.f, 6)}.`
    : `${graphOf}.`;
  // Minimum bracketing names its triple (a, b, c): its width is |c − a|, not b − a.
  const withBracket = runs.filter((_, i) => views[i][0]?.bracket);
  const allTriples =
    withBracket.length > 0 && withBracket.every((r) => r.sel.id === 'bracket_minimum');
  const anyTriple = withBracket.some((r) => r.sel.id === 'bracket_minimum');
  const yName: MathRun[] =
    metric === 'width'
      ? allTriples
        ? [mathMain('|'), mathVar('c'), mathMain(' − '), mathVar('a'), mathMain('|')]
        : anyTriple
          ? [mathMain('bracket width')]
          : [mathVar('b'), mathMain(' − '), mathVar('a')]
      : [
          mathMain('|'),
          mathVar('x̂'),
          mathSub('k', 'italic'),
          mathMain(' − '),
          mathVar('x'),
          mathSup('⋆'),
          mathMain('|'),
        ];
  const note = focus && views[fi] ? stepNote(focus.sel.id, focus.result, views[fi], kFocus) : null;
  // The two longest notes of the run, rendered invisibly under the current one: the step block
  // keeps one size (and its formula one scale) while the playback steps through the run.
  const noteSizers = ((): string[] => {
    const v = focus ? views[fi] : undefined;
    if (!focus || !v) return [];
    const all: string[] = [];
    for (let k = 0; k < v.length; k++) {
      const n = stepNote(focus.sel.id, focus.result, v, k);
      if (n) all.push(n);
    }
    return [...new Set(all)].sort((p, q) => q.length - p.length).slice(0, 2);
  })();
  /** One run of the focused method: its notes share one fit and one reserved size. */
  const noteKey = `scalar-note|${focus?.sel.id}|${problem.id}|${bracket.join(',')}|${x0}|${views[fi]?.length ?? 0}`;
  const badgeText = focus
    ? nextLabel(
        focus.sel.id,
        next?.stepKind ?? null,
        focusView?.bracket && next?.nextBracket
          ? droppedEnds(focusView.bracket, next.nextBracket)
          : null,
        isEpsilonStep(focus.sel.id, focus.result.trace, kFocus + 1),
      )
    : null;
  const tripleFocus = focus?.sel.id === 'bracket_minimum';
  const ratio =
    focusView?.bracket && kFocus > 0 && views[fi][kFocus - 1]?.bracket
      ? (focusView.bracket[1] - focusView.bracket[0]) /
        (views[fi][kFocus - 1].bracket![1] - views[fi][kFocus - 1].bracket![0])
      : null;
  /** The step ratio of the badge at step k (the same form at every step of the run). */
  const ratioTexAt = (k: number, value: string) =>
    tripleFocus
      ? `|c_{${k}} - a_{${k}}| / |c_{${k - 1}} - a_{${k - 1}}| = ${value}`
      : `(b_{${k}} - a_{${k}})/(b_{${k - 1}} - a_{${k - 1}}) = ${value}`;
  const ratioTex = ratio === null ? '' : ratioTexAt(kFocus, ratio.toFixed(4));
  // The badge keeps one width over the run: its step slot reserves the last step's k and ratio,
  // and its status slot the longest status of the run (no element moves during playback).
  const focusViews = views[fi];
  const kMax = focusViews ? Math.max(0, focusViews.length - 1) : 0;
  const hasRatio = !!focusViews?.some((v) => v.bracket);
  const longestStatus = ((): string => {
    if (!focus || !focusViews?.length) return '';
    const trace = focus.result.trace;
    let best = focus.result.converged ? '✓ stopping test passed' : '◷ stopped';
    for (let k = 0; k < focusViews.length; k++) {
      const nx = nextView(focus.sel.id, trace, k);
      const b = focusViews[k].bracket;
      const text = nextLabel(
        focus.sel.id,
        nx?.stepKind ?? null,
        b && nx?.nextBracket ? droppedEnds(b, nx.nextBracket) : null,
        isEpsilonStep(focus.sel.id, trace, k + 1),
      );
      if (text && text.length > best.length) best = text;
    }
    return best;
  })();
  // The card's |c − a| for minimum bracketing: Python's step_size is the first step |b − a| at
  // k = 0 (and |u − b| on a non-finite stop), so the card reads the width off the triple.
  const cardStep =
    focusStep && tripleFocus && Array.isArray(focusStep.info.triple)
      ? {
          ...focusStep,
          stepSize: Math.abs(
            (focusStep.info.triple as number[])[2] - (focusStep.info.triple as number[])[0],
          ),
        }
      : focusStep;
  // A bracket that holds no known minimizer: the methods converge to an end point.
  const outside =
    needsBracket && problem.minima.length > 0
      ? problem.minima.every((m) => m < bracket[0] || m > bracket[1])
      : false;
  const nearestStar = outside
    ? problem.minima.reduce((best, m) =>
        Math.min(Math.abs(m - bracket[0]), Math.abs(m - bracket[1])) <
        Math.min(Math.abs(best - bracket[0]), Math.abs(best - bracket[1]))
          ? m
          : best,
      )
    : null;

  // The shared start-point hint ("Click the plot to set x₀ · drag x₀ to move it"); with a
  // bracket it also names the ends a₀, b₀. The zoomed view has no handles.
  const dragHint = (
    <p className={styles.hint}>
      {mode === 'full' && needsStart && !needsBracket ? (
        <StartPointHint variable="x₀" drag />
      ) : (
        <>
          <Icon name="crosshair" size={13} />
          <span>
            {mode === 'full'
              ? needsStart
                ? 'Click the plot to set x₀ · drag a₀, b₀ or x₀ to move them'
                : 'Drag a₀ or b₀ on the plot to move the bracket'
              : 'Switch to “Whole interval” to drag the handles on the plot'}
          </span>
        </>
      )}
    </p>
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
            <ProblemPicker problems={shown} value={problem.id} onChange={changeProblem} />
          </RailSection>
          {/* Problem data: the bracket the bracket methods start from, before the methods
              (the rail order of every lab: problem, its data, methods, start point). */}
          {needsBracket && (
            <RailSection
              title={
                // The shared section heading; the bracket stays math (caps would print [A₀, B₀]).
                <h2 className={shellStyles.sectionTitle}>
                  Bracket{' '}
                  <span className={styles.titleMath}>
                    <Formula tex="[a_0, b_0]" />
                  </span>
                </h2>
              }
            >
              <div className={styles.pair}>
                <NumberField
                  label="Bracket start a"
                  prefix="a₀"
                  value={bracket[0]}
                  onChange={(v) => v < brUrl[1] && setBr([v, brUrl[1]])}
                />
                <NumberField
                  label="Bracket end b"
                  prefix="b₀"
                  value={bracket[1]}
                  onChange={(v) => v > brUrl[0] && setBr([brUrl[0], v])}
                />
              </div>
              {!needsStart && dragHint}
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
          {needsStart && (
            <RailSection title="Start point">
              <NumberField label="Start point x₀" prefix="x₀" value={x0} onChange={setX0} />
              {dragHint}
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
        <div className={styles.toolbar}>
          <SegmentedControl
            label="View"
            value={mode}
            onChange={setMode}
            options={[
              {
                value: 'full',
                label: (
                  <>
                    <span className={styles.long}>Whole interval</span>
                    <span className={styles.short}>Whole</span>
                  </>
                ),
                ariaLabel: 'Whole interval',
              },
              {
                value: 'follow',
                label: (
                  <>
                    <span className={styles.long}>Zoom to bracket</span>
                    <span className={styles.short}>Zoom</span>
                  </>
                ),
                ariaLabel: 'Zoom to the bracket',
              },
            ]}
          />
        </div>
      }
      stage={
        <div className={styles.stageWrap}>
          <ScalarStage
            f={problem.f}
            domain={problem.domain}
            minima={problem.minima}
            runs={stageRuns}
            focus={fi}
            times={times}
            ease={!player.reducedMotion}
            mode={mode}
            bracket={needsBracket ? bracket : null}
            x0={needsStart ? x0 : null}
            onDrag={onDrag}
            ariaLabel={stageLabel}
            notice={
              outside && nearestStar !== null ? (
                <p className={styles.warnNote} role="note">
                  No known minimizer lies in{' '}
                  <Formula
                    tex={`[a_0, b_0] = [${texNum(bracket[0], 4)},\\ ${texNum(bracket[1], 4)}]`}
                  />
                  : the bracket methods converge to an end point, not to{' '}
                  <Formula tex={`x^\\star = ${texNum(nearestStar, 5)}`} />.
                </p>
              ) : undefined
            }
          />
          <div className={styles.overlay}>
            {focus && focusView && (
              <div className={styles.badge} aria-live="off">
                <Swatch slot={focus.sel.slot} size={8} />
                <span className={styles.badgeName}>{focus.method.spec.name}</span>
                <span className={styles.badgeK}>
                  <span>
                    k = {kFocus}
                    {/* k = 0 has no ratio yet: an invisible placeholder keeps the slot. */}
                    {hasRatio && (
                      <span
                        className={styles.badgeRatio}
                        data-hidden={ratio === null || !Number.isFinite(ratio) ? '' : undefined}
                      >
                        {' · '}
                        <Formula tex={ratio !== null ? ratioTex : ratioTexAt(1, '0.0000')} />
                      </span>
                    )}
                  </span>
                  <span aria-hidden="true" data-sizer="">
                    k = {kMax}
                    {hasRatio && kMax > 0 && (
                      <span className={styles.badgeRatio}>
                        {' · '}
                        <Formula tex={ratioTexAt(kMax, '0.0000')} />
                      </span>
                    )}
                  </span>
                </span>
                <span className={styles.badgeStatus}>
                  {badgeText ? (
                    <Badge tone="accent">{badgeText}</Badge>
                  ) : (
                    <Badge tone={focus.result.converged ? 'good' : 'warn'}>
                      {focus.result.converged ? '✓ stopping test passed' : '◷ stopped'}
                    </Badge>
                  )}
                  <span aria-hidden="true" data-sizer="">
                    <Badge tone="accent">{longestStatus}</Badge>
                  </span>
                </span>
              </div>
            )}
          </div>
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          height: 300,
          actions: (
            <SegmentedControl
              label="Convergence measure"
              value={metric}
              onChange={setMetric}
              options={[
                {
                  value: 'width',
                  label: <Formula tex={allTriples ? '|c - a|' : 'b - a'} />,
                  ariaLabel: 'Bracket width',
                },
                {
                  value: 'error',
                  label: <Formula tex="|\hat x_k - x^\star|" />,
                  ariaLabel: 'Distance to the minimizer',
                },
              ]}
            />
          ),
          content: (
            <RateChart
              series={series}
              ks={runs.map((_, i) => player.localK(i))}
              ts={times}
              guides={guides}
              yName={yName}
              onSeekEvaluations={seekEvaluations}
              ariaLabel={`${metric === 'width' ? (allTriples ? 'Bracket width |c − a|' : anyTriple ? 'Bracket width' : 'Bracket width b − a') : 'Distance to the minimizer'} against the number of evaluations, log scale, for ${names}.`}
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
              idBase="scalar-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="scalar-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <MethodCard
                  method={focus.method}
                  slot={focus.sel.slot}
                  step={cardStep}
                  result={focus.result}
                  error={focus.error}
                  live={
                    note ? (
                      <StepBlock
                        k={kFocus}
                        last={kFocus >= kMax}
                        reset={`${focus.sel.id}|${problem.id}|${noteKey}`}
                      >
                        <div className={styles.note}>
                          <Formula
                            tex={`\\begin{aligned}${note}\\end{aligned}`}
                            display
                            fit
                            fitGroup={noteKey}
                          />
                          {noteSizers.map((n) => (
                            <div key={n} aria-hidden="true" data-sizer="">
                              <Formula
                                tex={`\\begin{aligned}${n}\\end{aligned}`}
                                display
                                fit
                                fitGroup={noteKey}
                              />
                            </div>
                          ))}
                        </div>
                      </StepBlock>
                    ) : undefined
                  }
                  call={{
                    problem: problem.id,
                    options: usesStart(focus.sel.id) ? { x0 } : { bracket: runOptions.bracket },
                    params: focus.sel.params,
                  }}
                />
              ) : (
                <IterationTable
                  steps={focus.result.trace}
                  columns={columnsFor(focus.sel.id)}
                  k={kFocus}
                  onSelect={seekTo}
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
