/**
 * Regression lab: least squares, ridge and robust fits of a curve to data the viewer can edit.
 *
 * Stage: the data plane (DataStage) — every method's fit at the playhead, and the geometry of
 * the focused method (residual sticks and squares, IRLS weights as opacity, Huber band, minimax
 * equioscillation, L1 contact points, Theil–Sen median pair, ridge λ path).
 * Insights: fit statistics at the playhead (R², RMSE, …); an analysis panel (convergence of the
 * iterative methods, the degree sweep, the ridge path, the loss, the Theil–Sen slopes); the
 * method card with the step's numbers and the iteration table.
 */
import './setup';
import { useCallback, useId, useMemo, useState } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { DataProblem } from '../../problems/data';
import type { Step } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { replaceQuery, splitHash, useHash } from '../../app/router';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES } from '../../ui/colors';
import { MINUS, int, sci, sig, sigFixed } from '../../core/format';
import {
  Badge,
  Button,
  CopyButton,
  Formula,
  Icon,
  PlaybackBar,
  SegmentedControl,
  Slider,
  Swatch,
  TabPanel,
  Tabs,
  Toggle,
  Tooltip,
} from '../../ui/components';
import {
  ConvergenceChart,
  IterationTable,
  TableView,
  mathBold as mb,
  mathMain as mm,
  mathSub as sub,
  mathVar as mv,
  type TableColumn,
} from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  KAxisControl,
  RailSection,
  RunSummary,
  StepBlock,
  labQuery,
  pythonCall,
  useLabRunsState,
  useMethodSelection,
  useKAxis,
  useProblemState,
  type LabRun,
  type RunStatus,
  type RunSummaryItem,
} from '../_shell';
import { DataStage, type StageCurve, type StageGeometry } from './DataStage';
import { SweepChart } from './SweepChart';
import { LossChart } from './LossChart';
import { SlopeStrip } from './SlopeStrip';
import { StageKey } from './StageKey';
import { columnsFor } from './columns';
import { stepQuantities, stepTex, texNum } from './text';
import {
  DEFAULT_DEGREE,
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  PRESETS,
  type RegressionPreset,
} from './presets';
import {
  LAMBDA_GRID,
  betaAt,
  dataKey,
  decodePoints,
  degreeSweep,
  displayWeights,
  encodePoints,
  fitStats,
  floorPoints,
  hasDegree,
  isIterative,
  isLeastSquares,
  kindOf,
  maxSweepDegree,
  medianPairs,
  pythonData,
  relativeGap,
  residuals,
  ridgeFan,
  ridgePath,
  type Kind,
  type Pt,
} from './model';
import styles from './RegressionLab.module.css';
// The shared "Try this" list's styles: the regression list looks and reads the same.
import presetStyles from '../_shell/presets.module.css';

void preloadKatex();

/** Regression datasets first, then the interpolation samples (useful for the degree sweep). */
const PROBLEM_ORDER = [
  'outliers_linear',
  'noisy_linear',
  'noisy_quadratic',
  'anscombe_1',
  'exponential_growth',
  'runge_equispaced',
  'runge_chebyshev',
  'sine_samples',
  'step_data',
];
const DEG_CODEC = codecs.number;
/** Presets shown before "More examples" (as in the shared list). */
const SHOWN_PRESETS = 4;
const BOOL = codecs.bool;
type Analysis = 'conv' | 'degree' | 'lambda' | 'loss' | 'slopes';
type Detail = 'method' | 'steps';
type LamView = 'error' | 'coef';
const ANALYSIS_LABEL: Record<Analysis, string> = {
  conv: 'Convergence',
  degree: 'Degree',
  lambda: 'λ path',
  loss: 'Loss',
  slopes: 'Slopes',
};

export default function RegressionLab() {
  const lab = getLab('regression')!;
  const colors = useChartColors();
  const problems = useMemo(() => {
    const all = listProblems<DataProblem>('data');
    return [...all].sort((a, b) => PROBLEM_ORDER.indexOf(a.id) - PROBLEM_ORDER.indexOf(b.id));
  }, []);
  const methods = useMemo(() => listMethods('regression'), []);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM, ['d', 'f']);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [degRaw, setDegRaw] = useUrlState('deg', DEFAULT_DEGREE, DEG_CODEC);
  const degree = Math.max(0, Math.min(15, Math.round(degRaw)));
  const [dataRaw, setDataRaw] = useUrlState<number[]>('d', [], codecs.numbers);
  const [squares, setSquares] = useUrlState('sq', true, BOOL);
  const [truthOn, setTruthOn] = useUrlState('tr', true, BOOL);
  const [focusId, setFocusId] = useUrlState('f', '', codecs.string);
  const [draft, setDraft] = useState<Pt[] | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [detail, setDetail] = useState<Detail>('method');
  const [lamView, setLamView] = useState<LamView>('error');

  // ── Data: the problem's, an edited copy (URL `d`), or a live drag ─────────────────────
  const base = useMemo<Pt[]>(() => problem.x.map((x, i) => ({ x, y: problem.y[i] })), [problem]);
  const edited = useMemo(() => decodePoints(dataRaw), [dataRaw]);
  const points = draft ?? edited ?? base;
  const isEdited = draft !== null || edited !== null;
  const commit = useCallback((pts: Pt[]) => setDataRaw(encodePoints(pts)), [setDataRaw]);
  const resetData = useCallback(() => setDataRaw([]), [setDataRaw]);

  const runProblem = useMemo(
    () => ({
      id: isEdited ? `${problem.id}~${dataKey(points)}` : problem.id,
      name: problem.name,
      latex: problem.latex,
      description: problem.description,
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      // Edited data are fitted like Python's bare (x, y) pair: the domain is the data's range.
      domain: isEdited ? null : problem.domain,
    }),
    [problem, points, isEdited],
  );

  // The lab's degree slider drives every method that has a degree.
  const effective = useMemo(
    () => selection.map((s) => (hasDegree(s.id) ? { ...s, params: { ...s.params, degree } } : s)),
    [selection, degree],
  );
  const { runs } = useLabRunsState(runProblem, effective, {});
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);
  // While a point is dragged the fits follow it at their final iterate; release replays.
  const t = draft ? Infinity : player.t;
  const ease = !player.reducedMotion && !draft;

  const focus: LabRun | undefined =
    runs.find((r) => r.sel.id === focusId) ??
    runs.find((r) => isIterative(kindOf(r.sel.id))) ??
    runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : -1;
  const focusKind: Kind = focus ? kindOf(focus.sel.id) : 'ols';
  const focusTrace = focus?.result.trace ?? [];
  const focusK = focus ? Math.min(focusTrace.length - 1, Math.floor(Math.min(t, 1e9) + 1e-9)) : 0;
  const focusStep: Step | undefined = focusTrace[Math.max(0, focusK)];
  const focusColor = focus ? colors.series[focus.sel.slot] : colors.text;

  // ── Stage ───────────────────────────────────────────────────────────────────────────
  const curves: StageCurve[] = useMemo(
    () =>
      runs.flatMap((r) => {
        if (r.error) return [];
        // An exchange is a discrete step: the minimax line snaps from β_k to β_{k+1} (its band,
        // reference and residual sticks all belong to β_k). IRLS solves glide between iterates.
        const discrete = kindOf(r.sel.id) === 'minimax';
        const beta = betaAt(r.result.trace, discrete ? Math.floor(t + 1e-9) : t, ease);
        return beta
          ? [
              {
                key: r.sel.id,
                slot: r.sel.slot,
                beta,
                focused: r === focus,
                label: plainName(r.method.spec.name),
              },
            ]
          : [];
      }),
    [runs, t, ease, focus],
  );
  const focusBeta = curves.find((c) => c.key === focus?.sel.id)?.beta ?? null;

  const ridgeDegree = degree;
  const fan = useMemo(
    () => (focusKind === 'ridge' && ridgeDegree >= 1 ? ridgeFan(points, ridgeDegree) : []),
    [focusKind, points, ridgeDegree],
  );

  const ladEps = focus && focusKind === 'lad' ? Number(focus.sel.params.eps ?? 1e-6) : 1e-6;
  const geometry: StageGeometry | null = useMemo(() => {
    if (!focus || focus.error || !focusStep) return null;
    const g: StageGeometry = {
      slot: focus.sel.slot,
      beta: focusBeta,
      squares: squares && isLeastSquares(focusKind),
      weights: displayWeights(focusKind, focusStep),
    };
    const info = focusStep.info;
    const kBeta = Array.isArray(focusStep.x) ? (focusStep.x as number[]) : null;
    if (focusKind === 'huber' && typeof info.scale === 'number' && typeof info.delta === 'number') {
      const half = info.delta * info.scale;
      g.band = { half, kind: 'huber' };
    }
    if (focusKind === 'minimax' && typeof info.level === 'number' && kBeta) {
      const ref = (info.reference as number[]) ?? [];
      const r = residuals(points, kBeta);
      g.band = { half: Math.abs(info.level), kind: 'minimax' };
      g.reference = ref.map((i) => ({ index: i, sign: (r[i] ?? 0) >= 0 ? 1 : -1 }));
      g.entering = typeof info.entering === 'number' ? info.entering : null;
      // The band is drawn around β_k itself (not the eased blend): ±|h| belongs to β_k.
      g.beta = kBeta;
    }
    if (focusKind === 'lad' && kBeta) g.rings = floorPoints(points, kBeta, ladEps);
    if (focusKind === 'theil' && Array.isArray(info.slopes))
      g.pairs = medianPairs(points, info.slopes as number[]);
    if (focusKind === 'ridge') g.fan = fan;
    return g;
  }, [focus, focusStep, focusBeta, focusKind, squares, points, fan, ladEps]);

  const truth = truthOn && problem.fTrue ? problem.fTrue : null;

  // ── Fit statistics at the playhead ──────────────────────────────────────────────────
  interface FitRow {
    run: LabRun;
    stats: ReturnType<typeof fitStats> | null;
    fun: number | null;
  }
  // Statistics of the iterate β_k under the playhead (k = ⌊t⌋), never of the eased blend.
  const fitRows: FitRow[] = runs.map((r) => {
    const k = Math.max(0, Math.min(r.result.trace.length - 1, Math.floor(Math.min(t, 1e9) + 1e-9)));
    const xk = r.error ? null : r.result.trace[k]?.x;
    const beta = Array.isArray(xk) && xk.every(Number.isFinite) ? (xk as number[]) : null;
    return {
      run: r,
      stats: beta ? fitStats(points, beta) : null,
      fun: r.result.trace[k]?.fun ?? null,
    };
  });
  // The stage header: the shared run summary, with R² at the playhead in each run's badge.
  const summaryR2 = new Map<RunSummaryItem, number | null>();
  const summaryRuns: RunSummaryItem[] = fitRows.map(({ run, stats }) => {
    const item = {
      name: run.method.spec.name,
      slot: run.sel.slot,
      result: run.result,
      error: run.error,
    };
    summaryR2.set(item, stats?.r2 ?? null);
    return item;
  });
  const summaryBadge = (run: RunSummaryItem, st: RunStatus) => {
    const r2 = summaryR2.get(run);
    if (run.error || r2 === null || r2 === undefined) return undefined;
    return (
      <>
        <span aria-hidden="true">{st.icon}</span> R² {dec(r2, 3)}
      </>
    );
  };
  const fitColumns: TableColumn<FitRow>[] = [
    {
      key: 'm',
      header: 'Method',
      align: 'left',
      mono: false,
      value: (r) => (
        <span className={styles.fitName}>
          <Swatch slot={r.run.sel.slot} size={9} />
          <span className={styles.fitLabel} title={r.run.method.spec.name}>
            {plainName(r.run.method.spec.name)}
          </span>
        </span>
      ),
    },
    {
      key: 'r2',
      header: <MathHeader tex="R^2" spoken="R squared" />,
      value: (r) => dec(r.stats?.r2, 3),
    },
    {
      key: 'adj',
      header: <MathHeader tex="R^2_{\text{adj}}" spoken="adjusted R squared" />,
      value: (r) => dec(r.stats?.adjR2, 3),
    },
    {
      key: 'rmse',
      header: <MathHeader tex="\text{RMSE}" spoken="RMSE" />,
      value: (r) => fixed(r.stats?.rmse, 4),
    },
    {
      key: 'max',
      header: <MathHeader tex="\max|r_i|" spoken="largest residual" />,
      value: (r) => fixed(r.stats?.maxAbs, 4),
    },
  ];

  // ── Analysis: convergence, degree sweep, ridge path, loss, slopes ─────────────────────
  const iterRuns = runs.filter((r) => !r.error && isIterative(kindOf(r.sel.id)));
  const degreeRuns = runs.filter((r) => hasDegree(r.sel.id));
  const ridgeRun = runs.find((r) => r.sel.id === 'ridge_regression');
  const theilRun = runs.find((r) => r.sel.id === 'theil_sen');
  const tabs: { id: Analysis; label: string }[] = [
    ...(iterRuns.length ? [{ id: 'conv' as const, label: ANALYSIS_LABEL.conv }] : []),
    ...(degreeRuns.length ? [{ id: 'degree' as const, label: ANALYSIS_LABEL.degree }] : []),
    ...(ridgeRun ? [{ id: 'lambda' as const, label: ANALYSIS_LABEL.lambda }] : []),
    ...(focus && focusKind !== 'theil'
      ? [{ id: 'loss' as const, label: ANALYSIS_LABEL.loss }]
      : []),
    ...(theilRun ? [{ id: 'slopes' as const, label: ANALYSIS_LABEL.slopes }] : []),
  ];
  const preferred: Analysis =
    focusKind === 'theil'
      ? 'slopes'
      : focusKind === 'ridge'
        ? 'lambda'
        : focusKind === 'poly'
          ? 'degree'
          : isIterative(focusKind)
            ? 'conv'
            : 'loss';
  const shownAnalysis: Analysis | undefined =
    tabs.find((x) => x.id === analysis)?.id ??
    tabs.find((x) => x.id === preferred)?.id ??
    tabs[0]?.id;

  const domain = problem.domain as [number, number];
  const sweepDomain = useMemo<[number, number]>(() => {
    if (!isEdited) return domain;
    const xs = points.map((p) => p.x);
    return [Math.min(...xs), Math.max(...xs)];
  }, [isEdited, domain, points]);
  const fTrue = problem.fTrue;
  const sweep = useMemo(
    () => (shownAnalysis === 'degree' && !draft ? degreeSweep(points, fTrue, sweepDomain) : null),
    [shownAnalysis, draft, points, fTrue, sweepDomain],
  );
  const lamPath = useMemo(
    () =>
      ridgeRun && !draft && (shownAnalysis === 'lambda' || focusKind === 'ridge')
        ? ridgePath(points, degree, fTrue, sweepDomain)
        : null,
    [ridgeRun, draft, shownAnalysis, focusKind, points, degree, fTrue, sweepDomain],
  );
  const ridgeLam = Number(ridgeRun?.sel.params.lam ?? 1);
  const lamIndex = nearestIndex(LAMBDA_GRID, ridgeLam, true);
  // The path at the run's own λ (a slider or URL value is rarely a grid point): tr H_λ, the dots.
  const lamAt = useMemo(
    () =>
      ridgeRun && !draft && (shownAnalysis === 'lambda' || focusKind === 'ridge')
        ? (ridgePath(points, degree, fTrue, sweepDomain, [ridgeLam])[0] ?? null)
        : null,
    [ridgeRun, draft, shownAnalysis, focusKind, points, degree, fTrue, sweepDomain, ridgeLam],
  );
  const ridgeDf = lamAt?.df ?? null;
  const stepLam = (dir: 1 | -1) => {
    const tol = 1e-9 * ridgeLam;
    const i =
      dir > 0
        ? LAMBDA_GRID.findIndex((v) => v > ridgeLam + tol)
        : LAMBDA_GRID.findLastIndex((v) => v < ridgeLam - tol);
    if (i >= 0) setParam('ridge_regression', 'lam', Number(LAMBDA_GRID[i].toPrecision(3)));
  };
  const sweepMax = maxSweepDegree(points.length);

  const setParam = (id: string, name: string, v: number) =>
    setSelection(
      selection.map((s) => (s.id === id ? { ...s, params: { ...s.params, [name]: v } } : s)),
    );

  const seek = (k: number) => {
    player.pause();
    player.seek(k);
  };

  const convSeries = iterRuns.map((r) => ({
    label: r.method.spec.name,
    slot: r.sel.slot,
    count: r.result.nIter,
    end: r.result.converged ? ('converged' as const) : ('stopped' as const),
    values: relativeGap(r.result),
  }));
  // Linear or log k: the shared control (URL key `kx`), "auto" picks log k for runs ≥ 30× apart.
  const [logK, setKAxis] = useKAxis(iterRuns.map((r) => r.result.trace.length));
  const degColor = degreeRuns[0]
    ? colors.series[
        degreeRuns.find((r) => r.sel.id === 'polynomial_regression')?.sel.slot ??
          degreeRuns[0].sel.slot
      ]
    : colors.text;

  const analysisContent = (() => {
    switch (shownAnalysis) {
      case 'conv':
        return (
          <div className={styles.chartPad}>
            <ConvergenceChart
              series={convSeries}
              t={draft ? Infinity : player.t}
              logX={logK}
              yLabel="(F − F⋆)/F⋆"
              yName={[
                mm('('),
                mv('F'),
                mm('('),
                mb('β'),
                sub('k', 'italic'),
                mm(') − '),
                mv('F'),
                { t: '⋆', style: 'main', script: 'sup' },
                mm(')/'),
                mv('F'),
                { t: '⋆', style: 'main', script: 'sup' },
              ]}
              onSeek={seek}
              ariaLabel={`Relative objective gap of ${iterRuns.map((r) => `${r.method.spec.name} (${int(r.result.nIter)} iterations)`).join(', ')}`}
            />
          </div>
        );
      case 'degree':
        return sweep ? (
          <div className={styles.analysisBody}>
            <SweepChart
              xs={sweep.map((s) => s.at)}
              color={degColor}
              current={Math.min(degree, sweep.length - 1)}
              marker={degree}
              markerLabel={[mv('d'), mm(` = ${degree}`)]}
              onPick={(i) => setDegRaw(sweep[i].at)}
              onStep={(dir) => setDegRaw(Math.max(0, Math.min(15, degree + dir)))}
              aria={{
                min: 0,
                max: 15,
                now: degree,
                text:
                  degree > sweepMax
                    ? `degree ${degree} (beyond the sweep, d ≤ ${sweepMax})`
                    : `degree ${degree}`,
              }}
              logY
              lines={[
                { key: 'train', values: sweep.map((s) => s.train), label: [mm('train')] },
                {
                  key: 'loo',
                  values: sweep.map((s) => s.loo),
                  dash: [6, 4],
                  markMin: true,
                  label: [mm('LOO')],
                },
                ...(fTrue
                  ? [
                      {
                        key: 'truth',
                        values: sweep.map((s) => s.truth),
                        dash: [1.5, 3],
                        width: 2,
                        markMin: true,
                        label: [mm('vs '), mv('f')],
                      },
                    ]
                  : []),
              ]}
              xName={[mm('degree '), mv('d')]}
              yName={[mm('RMS error')]}
              ariaLabel="Polynomial least squares: error against the degree d. Arrow keys change the degree."
              valueText={(i) => `degree ${sweep[i]?.at}`}
            />
            <p className={styles.caption}>
              Least squares of degree <Formula tex="d" />: training RMSE (solid), leave-one-out RMSE{' '}
              <Formula tex="\sqrt{\tfrac1m\sum (r_i/(1-h_{ii}))^2}" /> (dashed)
              {fTrue ? (
                <>
                  , and the RMS distance to the noise-free <Formula tex="f" /> (dotted)
                </>
              ) : null}
              . Diamonds mark the minima; click to set <Formula tex="d" />
              {sweepMax < 15 ? ` (d ≤ m − 2 = ${sweepMax})` : ''}.
            </p>
          </div>
        ) : null;
      case 'lambda':
        return lamPath && ridgeRun ? (
          <div className={styles.analysisBody}>
            <div className={styles.analysisBar}>
              <SegmentedControl
                label="Ridge path view"
                value={lamView}
                onChange={setLamView}
                options={[
                  { value: 'error', label: 'Errors' },
                  { value: 'coef', label: 'Coefficients' },
                ]}
              />
              <span className={styles.readout}>
                <Formula tex="\operatorname{tr} H_\lambda" />{' '}
                <span className={styles.mono}>= {fmt(ridgeDf, 4)}</span>
              </span>
            </div>
            {degree < 1 ? (
              <p className={styles.caption}>
                Degree 0 has no slope to shrink: ridge returns ȳ for every λ.
              </p>
            ) : (
              <SweepChart
                xs={LAMBDA_GRID}
                logX
                logY={lamView === 'error'}
                zeroLine={lamView === 'coef'}
                color={colors.series[ridgeRun.sel.slot]}
                current={lamIndex}
                marker={ridgeLam}
                onStep={stepLam}
                aria={{
                  min: -8,
                  max: 4,
                  now: Math.log10(ridgeLam),
                  text: `λ = ${sig(ridgeLam, 3)}`,
                }}
                onPick={(i) =>
                  setParam('ridge_regression', 'lam', Number(LAMBDA_GRID[i].toPrecision(3)))
                }
                lines={
                  lamView === 'error'
                    ? [
                        {
                          key: 'train',
                          values: lamPath.map((s) => s.train),
                          at: lamAt?.train,
                          label: [mm('train')],
                        },
                        {
                          key: 'loo',
                          values: lamPath.map((s) => s.loo),
                          at: lamAt?.loo,
                          dash: [6, 4],
                          markMin: true,
                          label: [mm('LOO')],
                        },
                        ...(fTrue
                          ? [
                              {
                                key: 'truth',
                                values: lamPath.map((s) => s.truth),
                                at: lamAt?.truth,
                                dash: [1.5, 3],
                                width: 2,
                                markMin: true,
                                label: [mm('vs '), mv('f')],
                              },
                            ]
                          : []),
                      ]
                    : Array.from({ length: degree }, (_, j) => ({
                        key: `g${j}`,
                        values: lamPath.map((s) => s.gamma?.[j] ?? null),
                        at: lamAt?.gamma?.[j] ?? null,
                        width: 1.25,
                        alpha: 0.45 + (0.55 * (j + 1)) / degree,
                        label:
                          j < 3 || j === degree - 1 ? [mv('γ'), sub(String(j + 1))] : undefined,
                      }))
                }
                xName={[mv('λ')]}
                yName={lamView === 'error' ? [mm('RMS error')] : [mv('γ'), sub('j', 'italic')]}
                ariaLabel="Ridge path over λ. Arrow keys change λ."
                valueText={(i) => `λ = ${LAMBDA_GRID[i].toPrecision(3)}`}
              />
            )}
            <p className={styles.caption}>
              {lamView === 'error' ? (
                <>
                  Ridge of degree {degree} on <Formula tex="\lambda \in [10^{-8}, 10^4]" />:
                  training, leave-one-out
                  {fTrue ? ' and true-function' : ''} RMS errors. Click to set{' '}
                  <Formula tex="\lambda" />.
                </>
              ) : (
                <>
                  Standardized slopes{' '}
                  <Formula tex="\gamma_j = \beta_j\,\|\mathbf{x}^j - \overline{x^j}\|_2" /> shrink
                  to 0 as <Formula tex="\lambda" /> grows.
                </>
              )}
            </p>
          </div>
        ) : null;
      case 'loss':
        return focus && !focus.error && focusBeta ? (
          <div className={styles.chartPad}>
            <LossChart
              kind={focusKind}
              residuals={residuals(
                points,
                focusKind === 'minimax' && Array.isArray(focusStep?.x)
                  ? (focusStep!.x as number[])
                  : focusBeta,
              )}
              weights={geometry?.weights ?? null}
              color={focusColor}
              huber={
                focusKind === 'huber' && focusStep
                  ? { delta: Number(focusStep.info.delta), scale: Number(focusStep.info.scale) }
                  : null
              }
              level={focusKind === 'minimax' ? (focusStep?.info.level as number) : null}
              ladMedian={
                focusKind === 'lad' && focusStep
                  ? 1 / medianOf((focusStep.info.weights as number[]) ?? [])
                  : null
              }
              ariaLabel={`Loss of ${focus.method.spec.name} with the residuals at the playhead.`}
            />
          </div>
        ) : null;
      case 'slopes': {
        const s = theilRun?.result.trace[0];
        const slopes = (s?.info.slopes as number[] | undefined) ?? [];
        const ols = runs.find((r) => r.sel.id === 'linear_regression');
        return theilRun && s && slopes.length ? (
          <div className={styles.chartPad}>
            <SlopeStrip
              slopes={slopes}
              median={(s.x as number[])[1]}
              color={colors.series[theilRun.sel.slot]}
              olsSlope={ols && Array.isArray(ols.result.x) ? (ols.result.x as number[])[1] : null}
              ariaLabel={`Histogram of ${slopes.length} pairwise slopes; the median is ${sig((s.x as number[])[1], 4)}.`}
            />
          </div>
        ) : null;
      }
      default:
        return null;
    }
  })();

  // ── Method card + step rule ─────────────────────────────────────────────────────────
  const ruleTex =
    focus && !focus.error
      ? stepTex(focusKind, focusTrace, focusK, { lam: ridgeLam, df: ridgeDf, eps: ladEps })
      : null;
  const quantities =
    focus && !focus.error
      ? stepQuantities(focusKind, focusStep, {
          lam: ridgeLam,
          df: ridgeDf,
          floorCount: geometry?.rings?.length ?? 0,
        })
      : [];
  const focusParams = focus ? (effective.find((s) => s.id === focus.sel.id)?.params ?? {}) : {};
  const pyCall = focus
    ? isEdited
      ? `import numopt\n${pythonData(points)}\nres = ${pythonCall(focus.sel.id, { params: focusParams, specs: focus.method.spec.params }).replace(', f', ', (x, y)')}`
      : undefined
    : undefined;

  // What the focused method draws, with its number (the stage key, after the method's name).
  const stageReadout = (() => {
    if (!focus || focus.error || !focusStep) return null;
    const info = focusStep.info;
    switch (focusKind) {
      case 'huber':
        return typeof info.scale === 'number' && typeof info.delta === 'number'
          ? `|r| \\le \\delta\\hat\\sigma = ${texNum(info.delta * info.scale, 3)}`
          : null;
      case 'minimax':
        return typeof info.level === 'number'
          ? `\\hat y \\pm |h_{${focusK}}| = ${texNum(Math.abs(info.level), 4)}`
          : null;
      case 'lad':
        return `\\#\\{|r_i| \\le \\varepsilon\\} = ${geometry?.rings?.length ?? 0}`;
      case 'theil':
        return Array.isArray(focusStep.x)
          ? `\\beta_1 = \\operatorname{med} = ${texNum((focusStep.x as number[])[1], 4)}`
          : null;
      case 'ridge':
        return `\\lambda = ${texNum(ridgeLam, 3)},\\ d = ${degree}`;
      default:
        return focusStep.fun !== null
          ? `\\textstyle\\sum r_i^2 = ${texNum(focusStep.fun, 4)}`
          : null;
    }
  })();

  // The rail does not depend on the playhead: memoized, so a replay re-renders none of it (the
  // method slots animate their layout, and a per-frame re-render would re-measure them).
  const setSpeed = player.setSpeed;
  const degreeNames = degreeRuns.map((r) => r.method.spec.name).join(' and ');
  const runCount = runs.length;
  const pointCount = points.length;
  const controls = useMemo(
    () => (
      <>
        <RailSection title="Try this">
          <PresetList presets={PRESETS} onApply={(p) => setSpeed(p.speed ?? 1)} />
        </RailSection>
        <RailSection title="Problem">
          <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          <div className={styles.dataRow}>
            <span className={styles.dataCount}>
              <span className={styles.mono}>{pointCount}</span> points
              {isEdited && <Badge tone="accent">edited</Badge>}
            </span>
            {isEdited && (
              <Button size="sm" variant="ghost" icon="reset" onClick={resetData}>
                Restore
              </Button>
            )}
          </div>
          <p className={styles.hint}>
            Drag a point. Click empty space to add one; double-click or press Delete to remove it.
            Keys: arrows move the focused point, Page Up/Down pick its neighbor.
          </p>
        </RailSection>
        <RailSection
          title="Methods"
          actions={
            <span className={styles.count}>
              {runCount} / {MAX_SERIES}
            </span>
          }
        >
          <MethodSlots
            available={methods}
            value={selection}
            onChange={setSelection}
            hiddenParams={['degree']}
          />
        </RailSection>
        {degreeNames && (
          <RailSection title="Model" actions={<span className={styles.mono}>d = {degree}</span>}>
            <div className={styles.degree}>
              <span className={styles.degreeLabel}>
                <Formula tex="d" /> Degree
              </span>
              <Slider
                label="Polynomial degree"
                value={degree}
                min={0}
                max={15}
                integer
                step={1}
                onChange={(v) => setDegRaw(Math.round(v))}
                valueText={`degree ${degree}`}
              />
            </div>
            <p className={styles.hint}>
              One <Formula tex="\hat y(x) = \sum_{j=0}^{d}\beta_j x^j" /> for {degreeNames}.
            </p>
          </RailSection>
        )}
      </>
    ),
    [
      setSpeed,
      problems,
      problem.id,
      setProblem,
      pointCount,
      isEdited,
      resetData,
      degreeNames,
      degree,
      setDegRaw,
      runCount,
      methods,
      selection,
      setSelection,
    ],
  );

  // Tabs and segmented controls animate a layoutId pill: built once per value, not per frame.
  const pickerKey = runs.map((r) => `${r.sel.id}:${r.sel.slot}:${r.method.spec.name}`).join('|');
  const focusSelId = focus?.sel.id ?? '';
  const focusRuns = useMemo(
    () =>
      pickerKey
        ? pickerKey.split('|').map((x) => {
            const [value, slot, ...name] = x.split(':');
            return { value, slot: Number(slot), name: name.join(':') };
          })
        : [],
    [pickerKey],
  );
  const tabsKey = tabs.map((x) => x.id).join(',');
  const analysisTabs = useMemo(
    () =>
      shownAnalysis ? (
        <Tabs
          label="Analysis"
          value={shownAnalysis}
          onChange={setAnalysis}
          idBase="rg-analysis"
          items={tabsKey
            .split(',')
            .map((id) => ({ id: id as Analysis, label: ANALYSIS_LABEL[id as Analysis] }))}
        />
      ) : (
        <h2 className={styles.insightTitle}>Analysis</h2>
      ),
    [shownAnalysis, tabsKey],
  );
  const detailTabs = useMemo(
    () => (
      <Tabs
        label="Details"
        value={detail}
        onChange={setDetail}
        idBase="rg-details"
        items={[
          { id: 'method', label: 'Method' },
          { id: 'steps', label: 'Iterations' },
        ]}
      />
    ),
    [detail],
  );

  const stageAria = `${problem.name}${isEdited ? ' (edited)' : ''}: ${points.length} data points with the fits of ${runs
    .map((r) => r.method.spec.name)
    .join(', ')}${focus ? `; residuals of ${focus.method.spec.name}` : ''}`;

  return (
    <LabShell
      lab={lab}
      focus={focusSelId ? { runs: focusRuns, value: focusSelId, onChange: setFocusId } : undefined}
      stageNotice={
        runs.length === 0 ? 'Add a method to compare — up to four run on one clock.' : undefined
      }
      controls={controls}
      stageTitle={
        <>
          <span className={styles.problemName}>
            {problem.name}
            {isEdited ? ' (edited)' : ''}
          </span>
          <RunSummary runs={summaryRuns} badge={summaryBadge} />
        </>
      }
      stageToolbar={
        <>
          <span className={styles.toolbarToggles}>
            <Tooltip content="The squares of the residuals, drawn for the least-squares methods: their total area is the objective.">
              <span className={styles.toolbarToggle}>
                <Toggle checked={squares} onChange={setSquares} label="Squares" showLabel />
              </span>
            </Tooltip>
            {problem.fTrue && (
              <Tooltip content="The noise-free function the data were drawn from.">
                <span className={styles.toolbarToggle}>
                  <Toggle checked={truthOn} onChange={setTruthOn} label="Noise-free f" showLabel />
                </span>
              </Tooltip>
            )}
          </span>
        </>
      }
      stage={
        <div className={styles.stageRoot}>
          <div className={styles.stageMain}>
            <DataStage
              points={points}
              domain={domain}
              curves={curves}
              geometry={geometry}
              truth={truth}
              onDraft={setDraft}
              onCommit={commit}
              ariaLabel={stageAria}
            />
          </div>
          {/* The key keeps its bar (fixed height) with no method, so the plot never resizes. */}
          <div className={styles.keyBar}>
            {focus && !focus.error && (
              <StageKey
                kind={focusKind}
                name={plainName(focus.method.spec.name)}
                slot={focus.sel.slot}
                squares={!!geometry?.squares}
                fan={!!geometry?.fan?.length}
                truth={!!truth}
                readout={stageReadout}
              />
            )}
          </div>
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'fit',
          title: 'Fit at the playhead',
          // A name longer than the column (9em) takes a second line: 16 px more for its row.
          height:
            64 +
            33 * Math.max(1, runs.length) +
            16 * runs.filter((r) => plainName(r.method.spec.name).length > 15).length,
          content: (
            <div className={styles.fitTable}>
              <TableView
                columns={fitColumns}
                rows={fitRows}
                highlight={focusIndex >= 0 ? focusIndex : null}
                onSelect={(i) => setFocusId(fitRows[i].run.sel.id)}
                rowKey={(r) => r.run.sel.id}
                ariaLabel="Goodness of fit of every method at the playhead"
              />
            </div>
          ),
        },
        {
          id: 'analysis',
          label: 'Analysis',
          height: 272,
          title: analysisTabs,
          actions:
            shownAnalysis === 'conv' ? <KAxisControl logX={logK} onChange={setKAxis} /> : undefined,
          content: shownAnalysis ? (
            <TabPanel idBase="rg-analysis" id={shownAnalysis} focusable={false}>
              {analysisContent}
            </TabPanel>
          ) : null,
        },
        {
          id: 'details',
          label: 'Details',
          grow: true,
          title: detailTabs,
          content: focus ? (
            <TabPanel idBase="rg-details" id={detail} focusable={false}>
              {detail === 'method' ? (
                <div className={styles.methodTab}>
                  <MethodCard
                    method={focus.method}
                    slot={focus.sel.slot}
                    result={focus.result}
                    error={focus.error}
                    live={
                      ruleTex && (
                        <StepBlock
                          k={focusK}
                          last={focusK >= focusTrace.length - 1}
                          label={focusTrace.length <= 1 ? 'The fit' : undefined}
                          reset={`${focus.sel.id}~${focus.sel.slot}~${focusTrace.length}`}
                        >
                          <div className={styles.stepRuleBody}>
                            {/* One fit for the whole run: the size never jumps between steps. */}
                            <Formula
                              tex={ruleTex}
                              display
                              fit
                              fitGroup={`rg-step~${focus.sel.id}~${focusTrace.length}`}
                            />
                          </div>
                          <dl className={styles.live}>
                            {quantities.map((q) => (
                              <div
                                key={q.tex}
                                className={styles.liveCell}
                                data-wide={q.wide || undefined}
                              >
                                <dt>
                                  <Formula tex={q.tex} />
                                </dt>
                                <dd>{q.value}</dd>
                              </div>
                            ))}
                          </dl>
                          {(focusKind === 'huber' || focusKind === 'lad') && (
                            <p className={styles.stepNote}>
                              <Formula tex="\boldsymbol\theta" />: the same coefficients in the
                              centered variable <Formula tex="x - \bar x" />, where the IRLS solves;
                              the stop test is{' '}
                              <Formula tex="\|\Delta\boldsymbol\theta\|_\infty \le \mathrm{tol}\,(1 + \|\boldsymbol\theta\|_\infty)" />
                              .
                            </p>
                          )}
                          {pyCall && (
                            <p className={styles.pyRow}>
                              <CopyButton
                                text={pyCall}
                                label="Copy the Python call for the edited data"
                              >
                                Copy Python call (edited data)
                              </CopyButton>
                            </p>
                          )}
                        </StepBlock>
                      )
                    }
                    call={isEdited ? undefined : { problem: problem.id, params: focusParams }}
                  />
                </div>
              ) : (
                <IterationTable
                  steps={focusTrace}
                  columns={columnsFor(focusKind, degree)}
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

// ── Small pieces ───────────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return sig(v, digits);
}

/** Table cells: fixed significant digits with trailing zeros (columns align on the point). */
function fixed(v: number | null | undefined, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  // An interpolating fit leaves residuals ~1e-9: scientific, not JavaScript's "8.334e-9".
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return sci(v, digits - 1);
  return sigFixed(v, digits).trim();
}

/** Fixed decimals with a true minus (R² columns align on the point: 0.348, −0.074). */
function dec(v: number | null | undefined, digits: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const s = v.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s.replace('-', MINUS);
}

/** A table header in KaTeX with a plain spoken name (the typeset math is hidden from AT). */
function MathHeader({ tex, spoken }: { tex: string; spoken: string }) {
  return (
    <>
      <span aria-hidden="true">
        <Formula tex={tex} />
      </span>
      <span className="visually-hidden">{spoken}</span>
    </>
  );
}

function medianOf(v: readonly number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function nearestIndex(xs: readonly number[], v: number, log: boolean): number {
  let best = 0;
  xs.forEach((t, i) => {
    const d = log ? Math.abs(Math.log(t) - Math.log(v)) : Math.abs(t - v);
    const db = log ? Math.abs(Math.log(xs[best]) - Math.log(v)) : Math.abs(xs[best] - v);
    if (d < db) best = i;
  });
  return best;
}

/**
 * The registry name without its parenthetical ("Huber regression (IRLS)" → "Huber regression"):
 * where space is short (the plot's direct labels, the fit table, the stage chips). The full
 * name stays in the tooltip, the card and every accessible name.
 */
function plainName(name: string): string {
  return name.replace(/\s*\([^)]*\)/g, '').trim();
}

/**
 * "Try this" for the regression presets, in the shared list's form (presets.module.css: titles,
 * the note under the active preset only, four presets before "More examples", `$…$` notes).
 * Only applying and matching differ: a preset also owns `deg`, `f` and the edited data `d`,
 * which the shared `applyPreset` would keep when the preset leaves them unset.
 */
function PresetList({
  presets,
  onApply,
}: {
  presets: readonly RegressionPreset[];
  onApply?: (p: RegressionPreset) => void;
}) {
  const hash = useHash();
  const listId = useId();
  const [more, setMore] = useState(false);
  const cur = new URLSearchParams(splitHash(hash).query);
  const queryOf = (p: RegressionPreset) => {
    const q = labQuery(p);
    if (p.degree !== undefined && p.degree !== DEFAULT_DEGREE) q.set('deg', String(p.degree));
    if (p.focus) q.set('f', p.focus);
    return q;
  };
  const KEYS = ['p', 'm', 'deg', 'f', 'd'];
  const active = (p: RegressionPreset) => {
    const want = queryOf(p);
    return KEYS.every((k) => {
      const a = cur.get(k) ?? (k === 'p' ? DEFAULT_PROBLEM : null);
      const b = want.get(k) ?? (k === 'p' ? DEFAULT_PROBLEM : null);
      return (
        a === b ||
        (k === 'm' && a === null && b === labQuery({ methods: DEFAULT_SELECTION }).get('m'))
      );
    });
  };
  const apply = (p: RegressionPreset) => {
    const next = new URLSearchParams(splitHash(window.location.hash).query);
    for (const k of KEYS) next.delete(k);
    queryOf(p).forEach((v, k) => next.set(k, v));
    replaceQuery(next);
    onApply?.(p);
  };
  const activeIndex = presets.findIndex(active);
  const showAll = more || activeIndex >= SHOWN_PRESETS;
  const shown = showAll ? presets : presets.slice(0, SHOWN_PRESETS);
  return (
    <>
      <ul className={presetStyles.list} id={listId}>
        {shown.map((p, i) => (
          <li key={p.id}>
            <button
              type="button"
              className={presetStyles.preset}
              aria-pressed={i === activeIndex}
              onClick={() => apply(p)}
            >
              <span className={presetStyles.title}>{p.title}</span>
              {typeof p.note === 'string' && i === activeIndex && (
                <span className={presetStyles.note}>
                  <Prose text={p.note} />
                </span>
              )}
            </button>
          </li>
        ))}
      </ul>
      {presets.length > SHOWN_PRESETS && activeIndex < SHOWN_PRESETS && (
        <button
          type="button"
          className={presetStyles.more}
          aria-expanded={showAll}
          aria-controls={listId}
          onClick={() => setMore((m) => !m)}
        >
          {showAll ? 'Fewer examples' : `More examples (${presets.length - SHOWN_PRESETS})`}
          <Icon name="chevronDown" size={12} />
        </button>
      )}
    </>
  );
}

/** Prose with inline math: `$…$` segments are typeset with KaTeX. */
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
