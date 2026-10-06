/**
 * Linear systems lab: A x = b by elimination, factorization and iteration.
 *
 * Main view, by method class:
 *   direct     the augmented matrix stage by stage (pivot, swaps, multipliers, zeros appearing)
 *              with L, U (or Q, R, Cholesky's L, Thomas's c′, d′) building up;
 *   iterative  2 × 2: the two lines in the plane over the level sets of φ, with each method's
 *              step geometry (Jacobi's two solves, the Gauss–Seidel/SOR staircase, CG's tangent
 *              ellipse); n > 2: the iterate x_k(i) against its index i, beside x⋆.
 * Insights: the error against k (log) with the rates the theory predicts (ρ(G)^k, the CG bound,
 * k = n), the ρ(ω) curve of SOR with a draggable ω, and the method card / iteration table.
 */
import './setup';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Matrix, Vector } from '../../core/types';
import type { LinalgProblem } from '../../problems/linalg';
import { codecs, useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES } from '../../ui/colors';
import { sci, sig, vec } from '../../core/format';
import { Formula, PlaybackBar, SegmentedControl, TabPanel, Tabs } from '../../ui/components';
import { IterationTable, type MathRun } from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { describe, oneBased } from './messages';
import { nearTex, nearText } from './texfmt';
import { getLab } from '../index';
import {
  LabShell,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  StartPointFields,
  useLabRunsState,
  useMethodSelection,
  focusRuns,
  useProblemState,
  useStartPoint,
} from '../_shell';
import { DEFAULT_SELECTION, PRESETS } from './presets';
import {
  aNorm2,
  defaultStart,
  isDescent,
  isDirect,
  isStationary,
  relaxationCurve,
  structureHint,
  structureOf,
  displayDescription,
  displayDescriptionPlain,
} from './model';
import { EliminationView } from './EliminationView';
import { PlaneView } from './PlaneView';
import { ComponentsPlot } from './ComponentsPlot';
import { ResidualChart, type RateGuide, type ResidualSeries } from './ResidualChart';
import { RelaxationChart } from './RelaxationChart';
import { LinalgCard } from './LinalgCard';
import { MathProse } from './MathProse';
import { DIRECT_COLUMNS, ITERATIVE_COLUMNS } from './columns';
import styles from './LinalgLab.module.css';

void preloadKatex();

type View = 'auto' | 'matrix' | 'iterates';
type Metric = 'res' | 'err';
type Comp = 'x' | 'e';
const COMP_CODEC = codecs.oneOf<Comp>(['x', 'e']);
const VIEW_CODEC = codecs.oneOf<View>(['auto', 'matrix', 'iterates']);
const METRIC_CODEC = codecs.oneOf<Metric>(['res', 'err']);

const RES_NAME: MathRun[] = [
  { t: '‖', style: 'main' },
  { t: 'r', style: 'bold' },
  { t: 'k', style: 'italic', script: 'sub' },
  { t: '‖ / ', style: 'main' },
  { t: 'd', style: 'italic' },
];
/** ‖x_k − x⋆‖ / ‖x⋆‖ in the A-norm (SPD A) or the 2-norm: the plotted relative error. */
function errName(norm: MathRun): MathRun[] {
  return [
    { t: '‖', style: 'main' },
    { t: 'x', style: 'bold' },
    { t: 'k', style: 'italic', script: 'sub' },
    { t: ' − ', style: 'main' },
    { t: 'x', style: 'bold' },
    { t: '⋆', style: 'main', script: 'sup' },
    { t: '‖', style: 'main' },
    norm,
    { t: ' / ‖', style: 'main' },
    { t: 'x', style: 'bold' },
    { t: '⋆', style: 'main', script: 'sup' },
    { t: '‖', style: 'main' },
    norm,
  ];
}
const ERR_NAME_A = errName({ t: 'A', style: 'italic', script: 'sub' });
const ERR_NAME_2 = errName({ t: '2', style: 'main', script: 'sub' });

const RHO_TEX: Record<string, string> = {
  jacobi: '\\rho(G_J)',
  gauss_seidel: '\\rho(G_{GS})',
  sor: '\\rho(G_\\omega)',
};

export default function LinalgLab() {
  const lab = getLab('linalg')!;
  const colors = useChartColors();
  const problems = useMemo(() => listProblems<LinalgProblem>('linalg'), []);
  const methods = useMemo(() => listMethods('linalg'), []);

  const [problem, setProblem] = useProblemState(problems, 'spd_2x2');
  // The menu lists each description as plain text; the rail sets the current one as prose with
  // inline math (ρ(G_J) typeset as in the structure table below it), so the picker leaves it out.
  const pickerProblems = useMemo(
    () =>
      problems.map((p) => ({
        ...p,
        description: p.id === problem.id ? undefined : displayDescriptionPlain(p),
      })),
    [problems, problem.id],
  );
  const n = problem.n;
  const A = problem.A as Matrix;
  const plane = n === 2;
  const [x0, setX0] = useStartPoint(defaultStart(problem), n);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [viewPref, setViewPref] = useUrlState<View>('v', 'auto', VIEW_CODEC);
  const [metricPref, setMetric] = useUrlState<Metric>('y', 'res', METRIC_CODEC);
  const [compPref, setComp] = useUrlState<Comp>('c', 'x', COMP_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  const [focusId, setFocusId] = useState<string | null>(null);

  const structure = useMemo(() => structureOf(A), [A]);
  const solution = useMemo<Vector | null>(
    () => (problem.solution ? [...problem.solution] : null),
    [problem],
  );

  const runOptions = useMemo(() => (plane ? { x0: [x0[0], x0[1]] } : {}), [plane, x0]);
  const { runs, pending } = useLabRunsState(problem, selection, runOptions, { defer: n > 4 });
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const allDirect = selection.length > 0 && selection.every((s) => isDirect(s.id));
  const player = useTracePlayer(traces, { speedIndex: allDirect ? 1 : 2 });
  usePlayerKeyboard(player);
  // Elimination has few steps, each played in three beats: slow the clock when only direct
  // methods run (and back to 1× when an iteration joins).
  const [prevAllDirect, setPrevAllDirect] = useState(allDirect);
  if (prevAllDirect !== allDirect) {
    setPrevAllDirect(allDirect);
    player.setSpeed(allDirect ? 0.5 : 1);
  }

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  // Runs that produced a trace (a method that rejects the input, e.g. Thomas on a full
  // matrix, has none and is reported in the header and the card instead).
  const directRuns = runs.filter((r) => isDirect(r.sel.id) && r.result.trace.length > 0);
  const iterRuns = runs.filter((r) => !isDirect(r.sel.id) && r.result.trace.length > 0);

  let view: 'matrix' | 'iterates' =
    viewPref === 'auto' ? (focus && isDirect(focus.sel.id) ? 'matrix' : 'iterates') : viewPref;
  if (view === 'matrix' && directRuns.length === 0) view = 'iterates';
  if (view === 'iterates' && iterRuns.length === 0 && !plane && directRuns.length > 0)
    view = 'matrix';
  const matrixRun = focus && directRuns.includes(focus) ? focus : directRuns[0];
  const metric: Metric = solution ? metricPref : 'res';

  // ── Convergence data ──────────────────────────────────────────────────────────────
  const errNorm = useCallback(
    (x: Vector) => {
      if (!solution) return NaN;
      if (structure.spd) return Math.sqrt(Math.max(0, aNorm2(A, x, solution)));
      return Math.hypot(...x.map((v, i) => v - solution[i]));
    },
    [A, solution, structure.spd],
  );
  const xStarNorm = useMemo(
    () => (solution ? errNorm(solution.map(() => 0)) || 1 : 1),
    [solution, errNorm],
  );

  const series: ResidualSeries[] = useMemo(
    () =>
      runs.map((r) => {
        const bNorm = Math.hypot(...problem.b) || 1;
        const values = r.result.trace.map((s) => {
          if (metric === 'err')
            return Array.isArray(s.x) ? errNorm(s.x as number[]) / xStarNorm : null;
          if (isDirect(r.sel.id)) return s.fun === null ? null : s.fun / bNorm;
          return (s.info.relative_residual as number | undefined) ?? null;
        });
        const rho = r.result.extra.spectral_radius as number | undefined;
        const rate = r.result.extra.rate_bound as number | null | undefined;
        const note =
          isStationary(r.sel.id) && rho !== undefined
            ? `${RHO_TEX[r.sel.id]} = ${nearTex(rho, (v) => v.toFixed(3))}`
            : metric === 'err' && isDescent(r.sel.id) && rate
              ? `${r.sel.id === 'steepest_descent_linear' ? '' : '2'}q^k,\\ q = ${nearTex(rate, (v) => v.toFixed(3))}`
              : undefined;
        return {
          label: r.method.spec.name,
          slot: r.sel.slot,
          note,
          values,
          end: r.result.converged ? 'converged' : 'stopped',
          count: r.result.nIter,
          muted: focusId !== null && focusId !== r.sel.id,
        };
      }),
    [runs, metric, errNorm, xStarNorm, problem.b, focusId],
  );

  const guides: RateGuide[] = useMemo(() => {
    const out: RateGuide[] = [];
    runs.forEach((r, i) => {
      const vals = series[i]?.values ?? [];
      if (isStationary(r.sel.id)) {
        const rho = r.result.extra.spectral_radius as number | undefined;
        let K = vals.length - 1;
        while (K > 0 && !(vals[K] !== null && (vals[K] as number) > 0 && Number.isFinite(vals[K])))
          K--;
        if (!rho || !(rho > 0) || K < 3) return;
        const vK = vals[K] as number;
        out.push({
          slot: r.sel.slot,
          v0: vK / rho ** K,
          rate: rho,
          from: Math.max(0, K - Math.max(6, 0.75 * K)),
          to: Math.min(K + Math.max(3, 0.15 * K), K * 2),
        });
      } else if (metric === 'err' && structure.spd && isDescent(r.sel.id)) {
        const rate = r.result.extra.rate_bound as number | null | undefined;
        const e0 = vals[0];
        if (!rate || e0 === null || !(e0 > 0)) return;
        const cg = r.sel.id !== 'steepest_descent_linear';
        out.push({
          slot: r.sel.slot,
          v0: (cg ? 2 : 1) * e0,
          rate,
          from: 0,
          to: Math.max(vals.length - 1, cg ? n + 2 : 4),
        });
      }
    });
    return out;
  }, [runs, series, metric, structure.spd, n]);
  const cgSelected = runs.some(
    (r) => r.sel.id === 'conjugate_gradient_linear' || r.sel.id === 'preconditioned_cg',
  );

  // ── Relaxation ────────────────────────────────────────────────────────────────────
  const showRelaxation = !structure.zeroDiagonal && runs.some((r) => isStationary(r.sel.id));
  const curve = useMemo(
    () =>
      showRelaxation
        ? relaxationCurve(A, structure.omegaOpt !== null ? [structure.omegaOpt] : [])
        : [],
    [A, showRelaxation, structure.omegaOpt],
  );
  const sorSel = selection.find((s) => s.id === 'sor');
  const sorOmega = sorSel ? Number(sorSel.params.omega ?? 1.5) : null;
  const sorRun = runs.find((r) => r.sel.id === 'sor');
  const sorRho = (sorRun?.result.extra.spectral_radius as number | undefined) ?? null;
  const setOmega = useCallback(
    (w: number) =>
      setSelection((prev) => {
        if (prev.some((s) => s.id === 'sor'))
          return prev.map((s) =>
            s.id === 'sor' ? { ...s, params: { ...s.params, omega: w } } : s,
          );
        const used = new Set(prev.map((s) => s.slot));
        const slot = [0, 1, 2, 3].find((q) => !used.has(q));
        if (slot === undefined || prev.length >= MAX_SERIES) return prev;
        return [...prev, { id: 'sor', slot, params: { omega: w } }];
      }),
    [setSelection],
  );

  // ── Views ─────────────────────────────────────────────────────────────────────────
  const pickStart = useCallback(
    (p: [number, number]) => setX0([Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]),
    [setX0],
  );
  const localT = (i: number) => player.localT(i);

  const runNames = runs.map((r) => r.method.spec.name).join(', ');
  let stage: ReactNode;
  if (view === 'matrix' && matrixRun) {
    const i = runs.indexOf(matrixRun);
    stage = (
      <EliminationView
        key={`${problem.id}:${matrixRun.sel.id}`}
        method={matrixRun.sel.id}
        trace={matrixRun.result.trace}
        t={player.localT(i)}
        n={n}
        slot={matrixRun.sel.slot}
        reduced={player.reducedMotion}
        announce={!player.playing}
        converged={matrixRun.result.converged}
        message={matrixRun.error ?? matrixRun.result.message}
        extra={matrixRun.result.extra}
      />
    );
  } else if (plane && solution) {
    stage = (
      <PlaneView
        problem={problem}
        spd={structure.spd}
        solution={solution}
        runs={runs}
        localT={localT}
        t={player.t}
        ease={!player.reducedMotion}
        x0={x0}
        onPick={pickStart}
        focusId={focusId}
        seriesColors={colors.series}
      />
    );
  } else {
    const comp: Comp = solution ? compPref : 'x';
    const scaled = (x: number[]) => {
      const e = x.map((v, i) => v - (solution as number[])[i]);
      const m = Math.max(...e.map(Math.abs));
      return m > 0 && Number.isFinite(m) ? e.map((v) => v / m) : e.map(() => 0);
    };
    stage = (
      <ComponentsPlot
        solution={solution}
        mode={comp === 'e' ? 'error' : 'x'}
        ease={!player.reducedMotion}
        series={iterRuns.map((r) => ({
          slot: r.sel.slot,
          label: r.method.spec.name,
          iterates: r.result.trace.map((s) =>
            Array.isArray(s.x)
              ? comp === 'e'
                ? scaled(s.x as number[])
                : (s.x as number[])
              : null,
          ),
          t: player.localT(runs.indexOf(r)),
          muted: focusId !== null && focusId !== r.sel.id,
        }))}
        ariaLabel={`${comp === 'e' ? 'Shape of the error e_k = x_k − x⋆, scaled to unit maximum,' : 'Components of the iterates'} of ${iterRuns.map((r) => r.method.spec.name).join(', ') || 'no iterative method'} against the index i${solution ? `; x⋆ = ${vec(solution, 3)}` : ''}.`}
      />
    );
  }

  const viewOptions = [
    ...(directRuns.length > 0 ? [{ value: 'matrix' as const, label: 'Elimination' }] : []),
    ...(iterRuns.length > 0 || plane
      ? [{ value: 'iterates' as const, label: plane ? 'Plane' : 'Components' }]
      : []),
  ];

  const focusK = focus ? player.localK(focusIndex) : 0;

  return (
    <LabShell
      focus={
        focus ? { runs: focusRuns(runs), value: focus.sel.id, onChange: setFocusId } : undefined
      }
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
            <p className={styles.desc}>
              <MathProse text={displayDescription(problem)} />
            </p>
            <StructureFacts s={structure} />
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
          <RailSection title="Start point">
            {plane ? (
              <StartPointFields value={x0} onChange={setX0} variable="𝐱₀" />
            ) : (
              <p className={styles.note}>
                The iterations start from <Formula tex="\mathbf{x}_0 = \mathbf{0}" />, as in the
                Python reference; the direct methods need none.
              </p>
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
              // The Python messages in the lab's 1-based notation (the tooltip and the
              // accessible text show the full status).
              result: { ...r.result, message: oneBased(r.result.message) },
              error: r.error && oneBased(r.error),
            }))}
          />
        </>
      }
      stageToolbar={
        <>
          {view === 'iterates' && !plane && solution && (
            <SegmentedControl
              label="Quantity"
              value={compPref}
              onChange={setComp}
              options={[
                {
                  value: 'x',
                  label: <Formula tex="\mathbf{x}_k" />,
                  ariaLabel: 'Iterate',
                  tooltip: 'The iterate x_k beside x⋆',
                },
                {
                  value: 'e',
                  label: <Formula tex="\mathbf{e}_k/\|\mathbf{e}_k\|_\infty" />,
                  ariaLabel: 'Error shape',
                  tooltip:
                    'The error x_k − x⋆ scaled to unit size: the mode the iteration damps slowest',
                },
              ]}
            />
          )}
          {viewOptions.length > 1 && (
            <SegmentedControl
              label="View"
              value={view}
              onChange={(v) => setViewPref(v)}
              options={viewOptions}
            />
          )}
        </>
      }
      stage={stage}
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          height: 260,
          actions: solution ? (
            <SegmentedControl
              label="Error measure"
              value={metric}
              onChange={setMetric}
              options={[
                {
                  value: 'res',
                  label: <Formula tex="\|\mathbf{r}_k\|/d" />,
                  ariaLabel: 'Relative residual',
                  tooltip: 'Relative residual ‖b − Ax_k‖₂ / ‖b‖₂ (the stopping test)',
                },
                {
                  value: 'err',
                  label: (
                    <Formula
                      tex={structure.spd ? '\\|\\mathbf{e}_k\\|_A' : '\\|\\mathbf{e}_k\\|_2'}
                    />
                  ),
                  ariaLabel: 'Relative error',
                  tooltip: `Relative error ‖x_k − x⋆‖${structure.spd ? '_A' : '₂'} / ‖x⋆‖ (the norm the CG bound is stated in)`,
                },
              ]}
            />
          ) : undefined,
          content: (
            <div className={styles.chartPad}>
              <ResidualChart
                series={series}
                guides={guides}
                marker={cgSelected ? { k: n, label: 'k = n' } : null}
                t={player.t}
                yName={metric === 'res' ? RES_NAME : structure.spd ? ERR_NAME_A : ERR_NAME_2}
                ariaLabel={`${metric === 'res' ? 'Relative residual' : 'Relative error'} against iteration k (log scale) for ${runNames}. ${runs
                  .map((r) => `${r.method.spec.name}: ${describe(r.result, r.error).long}`)
                  .join('. ')}.`}
                onSeek={(k) => {
                  player.pause();
                  player.seek(k);
                }}
              />
            </div>
          ),
        },
        ...(showRelaxation
          ? [
              {
                id: 'relaxation',
                title: 'Relaxation',
                height: 220,
                actions: (
                  <span className={styles.panelNote}>{sorSel ? 'drag ω' : 'click to add SOR'}</span>
                ),
                content: (
                  <div className={styles.chartPad}>
                    <RelaxationChart
                      curve={curve}
                      rhoJ={structure.rhoJ}
                      rhoGS={structure.rhoGS}
                      omegaOpt={structure.omegaOpt}
                      rhoOpt={structure.rhoOpt}
                      omega={sorOmega}
                      rhoOmega={sorRho}
                      slot={sorSel ? sorSel.slot : null}
                      onOmega={setOmega}
                    />
                  </div>
                ),
              },
            ]
          : []),
        {
          id: 'details',
          label: 'Details',
          grow: true,
          title: (
            <Tabs
              label="Details"
              value={tab}
              onChange={setTab}
              idBase="linalg-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="linalg-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <LinalgCard
                  method={focus.method}
                  slot={focus.sel.slot}
                  result={focus.result}
                  error={focus.error}
                  trace={focus.result.trace}
                  k={focusK}
                  A={A}
                  n={n}
                  problemId={problem.id}
                  params={focus.sel.params}
                  x0={plane ? x0 : undefined}
                />
              ) : (
                <IterationTable
                  steps={focus.result.trace}
                  columns={isDirect(focus.sel.id) ? DIRECT_COLUMNS : ITERATIVE_COLUMNS}
                  k={focusK}
                  onSelect={(k) => {
                    player.pause();
                    player.seek(k);
                  }}
                  ariaLabel={`${focus.method.spec.name} steps`}
                />
              )}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}

// ── Rail: what the structure of A predicts ────────────────────────────────────────────

function StructureFacts({ s }: { s: ReturnType<typeof structureOf> }) {
  const fmt = (v: number | null) =>
    v === null
      ? '—'
      : !Number.isFinite(v)
        ? '∞'
        : nearText(v, (w) => (Math.abs(w) >= 1e4 ? sci(w, 3) : sig(w, 4)));
  const props = [
    s.spd ? 'symmetric positive definite' : s.symmetric ? 'symmetric, indefinite' : 'nonsymmetric',
    s.tridiagonal ? 'tridiagonal' : null,
    s.diagDominant ? 'strictly diagonally dominant' : null,
    s.zeroDiagonal ? 'a zero on the diagonal' : null,
  ].filter(Boolean);
  return (
    <div className={styles.facts}>
      <p className={styles.factsLine}>
        <Formula tex={`n = ${s.n}`} /> · {props.join(' · ')}
      </p>
      <dl className={styles.factsGrid}>
        <div>
          <dt>
            <Formula tex="\kappa_2(A)" />
          </dt>
          <dd>{fmt(s.kappa)}</dd>
        </div>
        <div>
          <dt>
            <Formula tex="\rho(G_J)" />
          </dt>
          <dd data-bad={s.rhoJ !== null && s.rhoJ >= 1 ? 'true' : undefined}>{fmt(s.rhoJ)}</dd>
        </div>
        <div>
          <dt>
            <Formula tex="\rho(G_{GS})" />
          </dt>
          <dd data-bad={s.rhoGS !== null && s.rhoGS >= 1 ? 'true' : undefined}>{fmt(s.rhoGS)}</dd>
        </div>
        {s.omegaOpt !== null && (
          <div>
            <dt>
              <Formula tex="\omega^\star" />
            </dt>
            <dd>{fmt(s.omegaOpt)}</dd>
          </div>
        )}
      </dl>
      <p className={styles.factsHint}>{structureHint(s)}</p>
    </div>
  );
}
