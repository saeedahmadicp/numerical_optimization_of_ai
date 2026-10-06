/**
 * Nonlinear least-squares lab: Gauss–Newton and Levenberg–Marquardt on four fits.
 *
 * The stage is split in two views of the same iteration:
 *   - data space: the data, the model curve (or circle) of every method morphing iterate by
 *     iterate, and the residual sticks of the focused method (for Rosenbrock: the residual plane);
 *   - parameter space: the contours of f = ½‖𝐫‖² with the paths, and the geometry of the step
 *     under the playhead — the Gauss–Newton model's level sets and its minimizer 𝐱ₖ + 𝐩ₖ, the
 *     backtracking trials, the LM trust disk ‖𝐡‖ ≤ ‖𝐡ₖ‖ implied by μₖ, and the curve 𝐡(μ).
 * Insights: convergence, damping μₖ over gain ratio ϱₖ, and the method card with the step's
 * numbers substituted into its rule.
 */
import './setup';
import { useCallback, useMemo, useRef, useState } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Step } from '../../core/types';
import type { LeastSquaresProblem } from '../../problems/least_squares';
import { codecs, useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES } from '../../ui/colors';
import { int, sig } from '../../core/format';
import {
  Formula,
  NumberField,
  PlaybackBar,
  SegmentedControl,
  TabPanel,
  Tabs,
} from '../../ui/components';
import {
  Contour2D,
  ConvergenceChart,
  IterationTable,
  mathBold as b,
  mathMain as mm,
  mathSub as sub,
  mathSup as sup,
  mathVar as mv,
  drawOverlays2D,
  useElementSize,
  type PathSpec,
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
  focusRuns,
  useLabRuns,
  useMethodSelection,
  useProblemState,
  useStartPoint,
} from '../_shell';
import { DataSpacePane, type DataRun } from './DataSpace';
import { StageKey } from './StageKey';
import { DampingChart, type DampingSeries } from './DampingChart';
import { StepRule } from './StepRule';
import { columnsFor } from './columns';
import { describeDamping, roundingNoise } from './text';
import { drawOffView, labelBoxes } from './offview';
import { dataSpaceFor, linearization, paramNames, stepGeometry, type MethodKind } from './models';
import styles from './LeastSquaresLab.module.css';
import { StartPointHint } from '../_shell/blocks';

void preloadKatex();

import { DEFAULT_PROBLEM, DEFAULT_SELECTION, LAB_START, PRESETS } from './presets';

type View = 'split' | 'data' | 'params';
type Metric = 'gap' | 'grad';
const VIEW_CODEC = codecs.oneOf<View>(['split', 'data', 'params']);
const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'grad']);

const SUBSCRIPT_0 = '₀';

const kindOf = (id: string): MethodKind =>
  id === 'gauss_newton' ? 'gauss_newton' : 'levenberg_marquardt';

export default function LeastSquaresLab() {
  const lab = getLab('least-squares')!;
  const colors = useChartColors();
  const problems = useMemo(() => listProblems<LeastSquaresProblem>('least_squares'), []);
  const methods = useMemo(() => listMethods('least_squares'), []);
  // One slot per method: the shell's selection holds each method id once.
  const maxRuns = Math.min(MAX_SERIES, methods.length);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [x0, setX0] = useStartPoint(LAB_START[problem.id] ?? problem.x0, 2);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [view, setView] = useUrlState<View>('v', 'split', VIEW_CODEC);
  const [metric, setMetric] = useUrlState<Metric>('y', 'gap', METRIC_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [focusId, setFocusId] = useState<string | null>(null);

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]] }), [x0]);
  const runs = useLabRuns(problem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus ? player.localK(focusIndex) : 0;
  const focusKind = focus ? kindOf(focus.sel.id) : 'gauss_newton';

  const names = paramNames(problem);
  const space = useMemo(() => dataSpaceFor(problem), [problem]);
  const fMin = problem.extra.f_min;
  const minima = problem.minima as unknown as [number, number][];
  const scaled = problem.id === 'michaelis_menten';

  // ── Parameter space ──────────────────────────────────────────────────────────────
  const paths: PathSpec[] = useMemo(
    () =>
      runs
        .map((r) => ({
          r,
          spec: {
            points: r.result.trace.map((s) => s.x as [number, number]),
            color: colors.series[r.sel.slot],
            label: r.method.spec.name,
            muted: runs.length > 1 && r !== focus,
            start: false,
            end: r.result.converged ? ('converged' as const) : ('stopped' as const),
          } satisfies PathSpec,
        }))
        // The focused method is painted last (on top).
        .sort((a, c) => Number(a.r === focus) - Number(c.r === focus))
        .map((x) => x.spec),
    [runs, colors, focus],
  );

  const geometry = useMemo(
    () =>
      focus && !focus.error
        ? stepGeometry(problem, focus.result.trace, focusK, focus.sel.slot, focusKind)
        : null,
    [problem, focus, focusK, focusKind],
  );
  // The step geometry is drawn in pixel space: a label is kept only when its step is at least
  // 28 px long and it stays clear of the 𝐱⋆ label. Points outside the view — reached iterates
  // and the marks of the current step — get chevrons and labels above the tick row.
  const drawStep = useCallback(
    (ctx: CanvasRenderingContext2D, view: View2D) => {
      const stars = minima.length === 1 ? minima.map(([x, y]) => view.toPx(x, y)) : [];
      const showLabel = (a: [number, number], c: [number, number], at: [number, number]) => {
        const pa = view.toPx(a[0], a[1]),
          pc = view.toPx(c[0], c[1]),
          q = view.toPx(at[0], at[1]);
        if (Math.hypot(pc[0] - pa[0], pc[1] - pa[1]) < 28) return false;
        // 𝐱⋆'s label runs ~130 px to the right of the cross, one line high.
        return !stars.some(
          ([sx, sy]) => q[0] > sx - 70 && q[0] < sx + 150 && Math.abs(q[1] - sy) < 24,
        );
      };
      const g =
        focus && !focus.error
          ? stepGeometry(problem, focus.result.trace, focusK, focus.sel.slot, focusKind, showLabel)
          : null;
      if (g)
        drawOverlays2D(
          ctx,
          {
            toPx: view.toPx,
            toData: (px, py) => [view.x.invert(px), view.y.invert(py)],
            width: view.width,
            height: view.height,
            dpr: view.dpr,
            colors: view.colors,
          },
          g.overlays,
        );
      drawOffView(
        ctx,
        runs.map((r) => ({
          points: r.result.trace.map((st) => st.x as [number, number]),
          name: r.method.spec.name,
        })),
        player.t,
        g && focus ? { marks: g.marks, slot: focus.sel.slot, k: focusK } : null,
        { toPx: view.toPx, width: view.width, height: view.height, colors: view.colors },
        labelBoxes(g?.overlays ?? [], stars, view.toPx),
      );
    },
    [problem, focus, focusK, focusKind, minima, runs, player.t],
  );

  const f2 = useCallback((x: number, y: number) => problem.f([x, y]), [problem]);
  const pickStart = useCallback(
    (p: [number, number]) => setX0([Number(p[0].toPrecision(4)), Number(p[1].toPrecision(4))]),
    [setX0],
  );

  // ── Data space ───────────────────────────────────────────────────────────────────
  const dataRuns: DataRun[] = useMemo(
    () =>
      runs.map((r) => ({
        points: r.result.trace.map((s) => s.x as [number, number]),
        slot: r.sel.slot,
        name: r.method.spec.name,
        focused: r === focus,
      })),
    [runs, focus],
  );
  // Rosenbrock's residual plane: the linear model's prediction for the focused step,
  // 𝐫ₖ + αₖJₖ𝐩ₖ (Gauss–Newton) or 𝐫ₖ + Jₖ𝐡ₖ (Levenberg–Marquardt).
  const prediction = useMemo(() => {
    if (space.kind !== 'residual' || !focus) return null;
    const tr = focus.result.trace;
    const cur = tr[focusK],
      next = tr[focusK + 1];
    const step = next?.info.step as number[] | null | undefined;
    if (!cur || !next || !step) return null;
    const lin = linearization(problem, cur.x as number[]);
    if (!lin) return null;
    const a = focusKind === 'gauss_newton' ? ((next.info.alpha as number | null) ?? 1) : 1;
    const h = [a * step[0], a * step[1]];
    const to = lin.r.map((ri, i) => ri + lin.J[i][0] * h[0] + lin.J[i][1] * h[1]);
    return {
      from: [lin.r[0], lin.r[1]] as const,
      to: [to[0], to[1]] as const,
      k: focusK,
      step:
        focusKind === 'gauss_newton'
          ? a === 1
            ? ('p' as const)
            : ('alpha_p' as const)
          : ('h' as const),
    };
  }, [space, focus, focusK, focusKind, problem]);

  // ── Layout ───────────────────────────────────────────────────────────────────────
  const stageRef = useRef<HTMLDivElement>(null);
  const stageSize = useElementSize(stageRef);
  const compact = stageSize.width > 0 && stageSize.width < 600 && stageSize.height < 640;
  const shown: View = compact && view === 'split' ? 'data' : view;
  const row = stageSize.width > 1.3 * stageSize.height;

  // ── Insights ─────────────────────────────────────────────────────────────────────
  const fStar = useMemo(() => {
    const seen = runs.flatMap((r) => r.result.trace.map((s) => s.fun ?? Infinity));
    return Math.min(fMin, ...seen);
  }, [fMin, runs]);
  const floor = 2.220446049250313e-16 * Math.max(1, Math.abs(fStar));
  const zeroResidual = problem.tags.includes('zero-residual');
  const series = useMemo(
    () =>
      runs.map((r) => ({
        label: r.method.spec.name,
        slot: r.sel.slot,
        count: r.result.nIter,
        end: r.result.converged ? ('converged' as const) : ('stopped' as const),
        // Proven rates only: Gauss–Newton with full steps is Newton's method on a zero-residual
        // problem with square J (quadratic); LM is superlinear there (μ falls ≤ 3× per step).
        rate:
          zeroResidual && r.result.converged && metric === 'gap'
            ? r.sel.id === 'gauss_newton'
              ? 'quadratic'
              : 'superlinear'
            : undefined,
        values: r.result.trace.map((s: Step) =>
          metric === 'grad' ? s.gradNorm : s.fun === null ? null : Math.max(s.fun - fStar, floor),
        ),
      })),
    [runs, metric, fStar, floor, zeroResidual],
  );

  // Textbook indexing throughout the lab: μₖ, 𝐡ₖ, αₖ, ϱₖ belong to the step that leaves 𝐱ₖ,
  // which Python records on Step k + 1 (its info describes the step that produced it).
  const damping: DampingSeries[] = useMemo(
    () =>
      runs.map((r) => {
        const tr = r.result.trace;
        const at = <T,>(k: number, key: string) =>
          (tr[k + 1]?.info[key] as T | null | undefined) ?? null;
        return {
          name: r.method.spec.name,
          slot: r.sel.slot,
          mu: tr.map((_, k) => at<number>(k, 'lambda')),
          gain: tr.map((_, k) => at<number>(k, 'gain_ratio')),
          accepted: tr.map((_, k) => at<boolean>(k, 'accepted')),
        };
      }),
    [runs],
  );
  const tableSteps = useMemo(
    () =>
      focus
        ? focus.result.trace.map((s, k, tr) => ({
            ...s,
            info: { ...s.info, next: tr[k + 1]?.info ?? null },
          }))
        : [],
    [focus],
  );

  const seek = (k: number) => {
    player.pause();
    player.seek(k);
  };

  const pythonX0 = [x0[0], x0[1]];
  const startText = `(${sig(x0[0], 4)}, ${sig(x0[1], 4)})`;
  const resNorm = (s?: Step) => (s ? (s.info.residual_norm as number | null) : null);
  const focusStep = focus?.result.trace[focusK];
  // A component that cancelled to rounding level makes ‖𝐫(𝐱ₖ)‖ noise too (the rank preset).
  const focusNoise =
    !!focusStep &&
    roundingNoise(
      focusStep.x as number[],
      focus?.result.trace[focusK - 1]?.x as number[] | undefined,
    ).length > 0;

  const dataTitle =
    space.kind === 'residual' ? (
      <>
        Residual space <Formula tex="\mathbf{r}(x, y) \in \mathbb{R}^2" />
      </>
    ) : (
      <>
        Data space <Formula tex={space.tex} />
      </>
    );
  const paramTitle = (
    <>
      Parameter space <Formula tex={`f(${names[0]}, ${names[1]}) = \\tfrac12\\|\\mathbf{r}\\|^2`} />
    </>
  );

  // The widest readouts of the run (largest k, longest value), typeset hidden under the live
  // ones so a readout keeps its width while the trace plays.
  const readoutReserve = useMemo(() => {
    const tr = focus?.result.trace ?? [];
    const kMax = Math.max(0, tr.length - 1);
    const longest = (vals: string[]) => vals.reduce((a, v) => (v.length > a.length ? v : a), '');
    const res = longest(
      tr.map((s, k) =>
        roundingNoise(s.x as number[], tr[k - 1]?.x as number[] | undefined).length > 0
          ? '= rounding noise'
          : `= ${sig(resNorm(s), 4)}`,
      ),
    );
    const radius = longest(
      tr.map((_, k) => {
        const h = tr[k + 1]?.info.step as number[] | null | undefined;
        return h ? `= ${sig(Math.hypot(h[0], h[1]), 3)}` : '';
      }),
    );
    return { kMax, res, radius };
  }, [focus]);

  const dataPane = (
    <div className={styles.pane} key="data">
      <div className={styles.paneHead}>
        <span className={styles.paneTitle}>{dataTitle}</span>
        {focus && focusStep && (
          <span className={styles.paneReadout}>
            <Readout
              tex={`\\|\\mathbf{r}(\\mathbf{x}_{${readoutReserve.kMax}})\\|`}
              value={readoutReserve.res}
              ghost
            />
            <Readout
              tex={`\\|\\mathbf{r}(\\mathbf{x}_{${focusK}})\\|`}
              value={focusNoise ? '= rounding noise' : `= ${sig(resNorm(focusStep), 4)}`}
            />
          </span>
        )}
      </div>
      <div className={styles.paneBody}>
        <DataSpacePane
          space={space}
          runs={dataRuns}
          t={player.t}
          ease={!player.reducedMotion}
          prediction={prediction}
          onPick={space.kind === 'circle' ? pickStart : undefined}
          ariaLabel={dataAria(
            space.kind,
            problem.name,
            runs.map((r) => r.method.spec.name),
          )}
        />
      </div>
    </div>
  );

  const paramPane = (
    <div className={styles.pane} key="params">
      <div className={styles.paneHead}>
        <span className={styles.paneTitle}>{paramTitle}</span>
        {geometry?.radius != null && (
          <span className={styles.paneReadout}>
            <Readout
              tex={`\\|\\mathbf{h}_{${readoutReserve.kMax}}\\|`}
              value={readoutReserve.radius}
              ghost
            />
            <Readout tex={`\\|\\mathbf{h}_{${focusK}}\\|`} value={`= ${sig(geometry.radius, 3)}`} />
          </span>
        )}
      </div>
      <div className={styles.paneBody}>
        <Contour2D
          f={f2}
          domain={problem.domain}
          cacheKey={problem.id}
          fMin={fMin}
          paths={paths}
          t={player.t}
          ease={!player.reducedMotion}
          minima={minima}
          start={[x0[0], x0[1]]}
          overlay={drawStep}
          offViewLabels={false}
          onPick={pickStart}
          equalAspect={!scaled}
          axisLabels={names}
          ariaLabel={`Contours of f = ½‖r‖² over (${names.join(', ')}) for ${problem.name}, with the paths of ${runs.map((r) => r.method.spec.name).join(' and ')} from ${startText}.`}
        />
      </div>
      {focus && !focus.error && (
        <div className={styles.keyBar}>
          <StageKey kind={focusKind} name={focus.method.spec.name} slot={focus.sel.slot} />
        </div>
      )}
      {scaled && (
        <p className={styles.caption} role="note">
          <b>Axes not to scale.</b>{' '}
          <Prose
            text={`The plotted $K$ range is ${spanRatioText(problem.domain)} smaller than the $V$ range, so a trust disk $\\|\\mathbf{h}\\| \\le \\Delta$ draws as a thin band: $\\mu I$ damps $K$ as hard as $V$.`}
          />
        </p>
      )}
    </div>
  );

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      stageNotice={
        runs.length === 0
          ? `Add a method to compare — ${maxRuns === 2 ? 'both' : `up to ${maxRuns}`} run on one clock.`
          : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection
            title="Methods"
            actions={
              <span className={styles.count}>
                {runs.length} / {maxRuns}
              </span>
            }
          >
            <MethodSlots available={methods} value={selection} onChange={setSelection} />
          </RailSection>
          <RailSection title="Start point">
            <div className={styles.xy}>
              <NumberField
                label={`Start ${names[0]}`}
                prefix={`${names[0]}${SUBSCRIPT_0}`}
                value={x0[0]}
                onChange={(v) => setX0([v, x0[1]])}
              />
              <NumberField
                label={`Start ${names[1]}`}
                prefix={`${names[1]}${SUBSCRIPT_0}`}
                value={x0[1]}
                onChange={(v) => setX0([x0[0], v])}
              />
            </div>
            <p className={styles.hint}>
              <StartPointHint variable="𝐱₀" />
            </p>
            {/* The circle fit is the one model whose start can also be set in the data plane. */}
            {space.kind === 'circle' && (
              <p className={styles.hintNote}>In the data plane, a click sets the center.</p>
            )}
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
      stageToolbar={
        <SegmentedControl
          label="View"
          value={shown}
          onChange={setView}
          options={
            compact
              ? [
                  { value: 'data', label: 'Data', ariaLabel: 'Data space' },
                  { value: 'params', label: 'Params', ariaLabel: 'Parameter space' },
                ]
              : [
                  { value: 'split', label: 'Split', ariaLabel: 'Data and parameter space' },
                  { value: 'data', label: 'Data', ariaLabel: 'Data space' },
                  { value: 'params', label: 'Params', ariaLabel: 'Parameter space' },
                ]
          }
        />
      }
      stage={
        <div
          ref={stageRef}
          className={styles.split}
          data-layout={shown === 'split' ? (row ? 'row' : 'column') : 'single'}
        >
          {shown !== 'params' && dataPane}
          {shown !== 'data' && paramPane}
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          height: 224,
          actions: (
            <SegmentedControl
              label="Convergence measure"
              value={metric}
              onChange={setMetric}
              options={[
                { value: 'gap', label: <Formula tex="f - f^\star" />, ariaLabel: 'Objective gap' },
                {
                  value: 'grad',
                  label: <Formula tex="\|J^{\mathsf T}\mathbf{r}\|" />,
                  ariaLabel: 'Gradient norm ‖Jᵀr‖',
                },
              ]}
            />
          ),
          content: (
            <div className={styles.chartPad}>
              <ConvergenceChart
                series={series}
                t={player.t}
                yLabel={metric === 'grad' ? '‖∇f(xₖ)‖' : 'f(xₖ) − f⋆'}
                yName={
                  metric === 'grad'
                    ? [mm('‖'), mv('J'), sup('T'), b('r'), mm('‖')]
                    : [mv('f'), mm('('), b('x'), sub('k', 'italic'), mm(') − '), mv('f'), sup('⋆')]
                }
                onSeek={seek}
                ariaLabel={`Convergence of ${runs.map((r) => `${r.method.spec.name} (${int(r.result.nIter)} iterations)`).join(', ')}`}
              />
            </div>
          ),
        },
        {
          id: 'damping',
          title: 'Damping and gain ratio',
          height: 236,
          content: (
            <div className={styles.chartPad}>
              <DampingChart
                series={damping}
                t={player.t}
                onSeek={seek}
                ariaLabel={`Damping μ and gain ratio ϱ per iteration. ${describeDamping(damping)}.`}
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
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="ls-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <div className={styles.methodTab}>
                  {/*
                   * "This step" is the step that leaves 𝐱ₖ (textbook indexing): αₖ, μₖ and ϱₖ
                   * are read from Step k + 1, where Python records them (StepRule.tsx).
                   */}
                  <MethodCard
                    live={
                      !focus.error && (
                        <StepRule
                          kind={focusKind}
                          trace={focus.result.trace}
                          k={focusK}
                          names={names}
                        />
                      )
                    }
                    method={focus.method}
                    slot={focus.sel.slot}
                    step={focusStep}
                    result={focus.result}
                    error={focus.error}
                    call={{
                      problem: problem.id,
                      options: { x0: pythonX0 },
                      params: focus.sel.params,
                    }}
                  />
                </div>
              ) : (
                <IterationTable
                  steps={tableSteps}
                  columns={columnsFor(focusKind)}
                  k={focusK}
                  onSelect={seek}
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

/** "about 1,100×": how many times the second axis range fits in the first. */
function spanRatioText(domain: readonly (readonly [number, number])[]): string {
  const r = (domain[0][1] - domain[0][0]) / (domain[1][1] - domain[1][0]);
  const rounded = Number(r.toPrecision(2));
  return `about ${rounded.toLocaleString('en-US')}×`;
}

function dataAria(kind: string, name: string, methods: string[]): string {
  const who = methods.join(' and ');
  if (kind === 'curve')
    return `${name}: the data points, the model curve of ${who} at the current iterate, and the residuals as vertical sticks.`;
  if (kind === 'circle')
    return `${name}: the data points on the arc, the circle of radius 2 about the current center of ${who}, and the radial residuals.`;
  return `${name}: the residual vector r(x) of ${who} in the plane, moving toward r = 0.`;
}

/** Prose with inline math: `$…$` segments are typeset with KaTeX, the rest is text. */
function Prose({ text }: { text: string }) {
  return (
    <>
      {text
        .split('$')
        .map((part, i) =>
          i % 2 === 1 ? <Formula key={i} tex={part} /> : <span key={i}>{part}</span>,
        )}
    </>
  );
}

/** A pane readout: the quantity's symbol and its value (`ghost`: hidden, reserves the width). */
function Readout({ tex, value, ghost }: { tex: string; value: string; ghost?: boolean }) {
  return (
    <span data-ghost={ghost || undefined} aria-hidden={ghost || undefined}>
      <Formula tex={tex} /> <span className={styles.mono}>{value}</span>
    </span>
  );
}
