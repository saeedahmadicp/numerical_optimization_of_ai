/**
 * Quadrature lab: ∫ₐᵇ f(x) dx by composite Newton–Cotes rules, Romberg, Gauss–Legendre, the
 * nested Clenshaw–Curtis and Gauss–Patterson rules, adaptive Simpson and Monte Carlo — the TS port of `numopt.integration.methods`, replayed step by
 * step. The stage draws, for every selected method, the area it integrates at the current step
 * (rectangles, chords, parabolic arches, Gauss weight cells, adaptive intervals, Monte Carlo
 * samples) and the signed error against f; the convergence chart plots the error against the
 * number of nodes on log–log axes with each rule's theoretical slope.
 */
import './setup';
import { useMemo, useState } from 'react';
import { listMethods, type MethodDoc, type RegisteredMethod } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { CalculusProblem } from '../../problems/calculus';
import type { Step } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { preloadKatex } from '../../ui/katex';
import {
  Button,
  Formula,
  NumberField,
  PlaybackBar,
  TabPanel,
  Tabs,
  Toggle,
  Tooltip,
} from '../../ui/components';
import { MAX_SERIES } from '../../ui/colors';
import { int, sci, sig } from '../../core/format';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { getLab } from '../index';
import {
  LabShell,
  MethodCard,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  StepBlock,
  useLabRuns,
  useMethodSelection,
  useProblemState,
  useStartPoint,
  type LabPreset,
  type MethodSelection,
} from '../_shell';
import { QuadStage } from './QuadStage';
import { ErrorChart, type ErrorSeries } from './ErrorChart';
import { RombergTable } from './RombergTable';
import { StepsTable } from './StepsTable';
import { ORDER, RULE_KIND, nodesUsed, observedOrder } from './geometry';
import { estimateSymbol, filledRule } from './cardRule';
import { referenceIntegral } from './reference';
import styles from './IntegrationLab.module.css';
import shellStyles from '../_shell/LabShell.module.css';

void preloadKatex();

/**
 * First view: the Gaussian bell on [−2, 2] with a second-order, a fourth-order and a Gauss
 * rule. Within a few seconds the stage shows trapezoids, parabolic arches and weight cells, and
 * the error chart three shapes: slope −2, slope −4 and the spectral plunge of Gauss. All three
 * converge (Python reference): T₅₁₂ to 7.5×10⁻⁷ at ε = 10⁻⁶, S₁₂₈ to 7.8×10⁻⁹, G₁₆ to 4×10⁻¹⁴ —
 * at nodes 513, 129 and 16, which is the point of the chart.
 */
const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'trapezoid', slot: 0, params: { n: 2, levels: 8, tol: 1e-6 } },
  { id: 'simpson', slot: 1, params: { n: 2, levels: 6 } },
  { id: 'gauss_legendre', slot: 2, params: { n: 16 } },
];

const PRESETS: LabPreset[] = [
  {
    id: 'sqrt',
    title: '√x drags every rule down to order 3/2',
    note: 'f′ is unbounded at 0, so the error has no expansion in h², h⁴, …: trapezoid, Simpson and Boole fall parallel, at slope 3/2.',
    problem: 'sqrt_0_1',
    methods: [
      { id: 'trapezoid', slot: 0, params: { levels: 8 } },
      { id: 'simpson', slot: 1, params: { levels: 8 } },
      { id: 'boole', slot: 3, params: { levels: 8 } },
    ],
  },
  {
    id: 'adaptive',
    title: 'Adaptive Simpson crowds the singularity',
    note: 'Intervals shrink only near x = 0: an error of 10⁻¹¹ from 369 values of f, where composite Simpson reaches 10⁻⁶ with 1,025.',
    problem: 'sqrt_0_1',
    methods: [
      { id: 'adaptive_simpson', slot: 2, params: { tol: 1e-7 } },
      { id: 'simpson', slot: 1, params: { levels: 8 } },
    ],
  },
  {
    id: 'kink',
    title: 'The midpoint rule certifies a wrong value',
    note: 'On |x − 0.3| with N = 13, 26, 52, the grid node x = 4/13 lies 1/130 from the kink at every level, so the midpoint error (1/130)² = 5.9×10⁻⁵ never changes: three equal estimates, “converged”, wrong.',
    problem: 'abs_kink',
    methods: [
      { id: 'midpoint_rule', slot: 0, params: { n: 13, levels: 2 } },
      { id: 'trapezoid', slot: 1, params: { n: 13, levels: 2 } },
    ],
  },
  {
    id: 'nested',
    title: 'Nested rules pay for each value of f once',
    note: 'Every node of one level is a node of the next, so the error estimate is free: on the Gaussian bell Gauss–Patterson converges with 31 values of f and Clenshaw–Curtis with 65. Gauss–Legendre reuses no node from Gₘ to Gₘ₊₁, so certifying G₁₆ costs 129.',
    problem: 'gaussian',
    methods: [
      { id: 'clenshaw_curtis', slot: 0, params: {} },
      { id: 'gauss_patterson', slot: 1, params: {} },
      { id: 'gauss_legendre', slot: 2, params: { n: 16 } },
    ],
  },
  {
    id: 'runge',
    title: 'Runge’s poles slow Gauss to 0.17 digits per node',
    note: 'The poles of 1/(1 + 25x²) at ±i/5 bound the Bernstein ellipse: the error falls like $\\rho^{-2m}$ with $\\rho = 1.22$.',
    problem: 'runge',
    methods: [
      { id: 'gauss_legendre', slot: 2, params: { n: 40 } },
      { id: 'romberg', slot: 3, params: {} },
    ],
  },
];

const BOOL = codecs.bool;
const EPS = Number.EPSILON;

type Tab = 'method' | 'steps' | 'romberg';

/** Sanitize an interval from the URL: inside the domain, a < b; otherwise the whole domain. */
function sanitizeAb(ab: readonly number[], d: readonly [number, number]): [number, number] {
  const a = Math.max(d[0], Math.min(ab[0], d[1])),
    b = Math.max(d[0], Math.min(ab[1], d[1]));
  return a < b ? [a, b] : [d[0], d[1]];
}

/** The live quantities of the focused step (the MethodCard's "this step" grid, quadrature edition). */
function StepReadout({ step, doc, est }: { step: Step; doc?: MethodDoc; est: string }) {
  const show = (v: unknown): string => {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'number')
      return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? sci(v, 3) : sig(v, 6);
    if (Array.isArray(v)) return `[${v.map((x) => sig(x as number, 3)).join(', ')}]`;
    return String(v);
  };
  const cells = [
    { tex: 'k', value: int(step.k) },
    { tex: est, value: sig(step.info.estimate as number, 10) },
    ...(doc?.quantities ?? []).map((q) => ({
      tex: q.tex,
      value: show(q.key === 'stepSize' ? step.stepSize : step.info[q.key.replace(/^info\./, '')]),
    })),
  ];
  return (
    <dl className={styles.readout}>
      {cells.map((c) => (
        <div key={c.tex} className={styles.readoutCell}>
          <dt>
            <Formula tex={c.tex} />
          </dt>
          <dd className={styles.mono}>{c.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function IntegrationLab() {
  const lab = getLab('integration')!;
  const problems = useMemo(() => listProblems<CalculusProblem>('calculus'), []);
  const methods = useMemo(() => listMethods('integration'), []);

  const [problem, setProblem] = useProblemState(problems, 'gaussian');
  const domain = problem.domain;
  const [abRaw, setAbRaw] = useStartPoint(domain, 2);
  const ab = sanitizeAb(abRaw, domain);
  const full = ab[0] === domain[0] && ab[1] === domain[1];
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [showError, setShowError] = useUrlState('err', true, BOOL);
  const [showEst, setShowEst] = useUrlState('est', false, BOOL);
  const [tab, setTab] = useState<Tab>('method');
  const [focusId, setFocusId] = useState<string | null>(null);

  const exact = useMemo(() => referenceIntegral(problem, ab[0], ab[1]), [problem, ab[0], ab[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  // The methods report |estimate − problem.exact|: on a sub-interval, hand them its exact value.
  const runProblem = useMemo(
    () => (full ? problem : { ...problem, exact }),
    [problem, full, exact],
  );
  const runOptions = useMemo(
    () => (full ? {} : { bracket: [ab[0], ab[1]] as [number, number] }),
    [full, ab[0], ab[1]], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const runs = useLabRuns(runProblem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusK = focus ? player.localK(focusIndex) : 0;
  const focusStep = focus?.result.trace[focusK];
  const hasRomberg = runs.some((r) => r.sel.id === 'romberg');
  const romberg = runs.find((r) => r.sel.id === 'romberg');
  const activeTab: Tab = tab === 'romberg' && !hasRomberg ? 'method' : tab;

  const seek = (k: number) => {
    player.pause();
    player.seek(k);
  };

  // Rounding level ε·max(|I|, ∫|f|): the error chart's floor.
  const floor = useMemo(() => {
    let s = 0;
    for (let i = 0; i <= 200; i++) {
      const v = Math.abs(problem.f(ab[0] + ((ab[1] - ab[0]) * i) / 200));
      if (Number.isFinite(v)) s += v;
    }
    return EPS * Math.max(Math.abs(exact), ((ab[1] - ab[0]) * s) / 201, 1e-300);
  }, [problem, ab, exact]);

  const series: ErrorSeries[] = runs.map((r, i) => {
    const tr = r.result.trace;
    const n = tr.map((s) => nodesUsed(r.sel.id, s));
    const err = tr.map((s) => (s.info.error as number | null) ?? null);
    const kk = Math.min(player.localK(i), tr.length - 1);
    const kind = RULE_KIND[r.sel.id];
    return {
      label: r.method.spec.name,
      slot: r.sel.slot,
      count: r.result.nIter,
      n,
      err,
      est: tr.map((s) => (s.info.err_est as number | null) ?? null),
      order:
        ORDER[r.sel.id] ??
        (kind === 'gauss' || kind === 'nested' ? 'spectral' : kind === 'monte-carlo' ? 'mc' : null),
      t: player.localT(i),
      observed:
        kind === 'adaptive' ||
        kind === 'gauss' ||
        kind === 'nested' ||
        kind === 'monte-carlo' ||
        kk < 1 ||
        !((err[kk] ?? 0) > 100 * floor && (err[kk - 1] ?? 0) > 100 * floor)
          ? null
          : observedOrder(n[kk - 1] ?? 0, err[kk - 1] ?? 0, n[kk] ?? 0, err[kk] ?? 0),
    };
  });

  const card = (m: RegisteredMethod): RegisteredMethod => {
    if (!m.doc || !focus) return m;
    const prev = focusK > 0 ? focus.result.trace[focusK - 1] : undefined;
    return {
      ...m,
      doc: {
        ...m.doc,
        rule: filledRule(m.spec.id, m.doc.rule, focusStep, prev, ab),
        quantities: [],
      },
    };
  };

  const stageLabel = runs.map((r) => r.method.spec.name).join(', ');

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
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection
            title={
              // The shared section heading; the interval stays math (caps would print [A, B]).
              <h2 className={shellStyles.sectionTitle}>
                Interval{' '}
                <span className={styles.titleMath}>
                  <Formula tex="[a, b]" />
                </span>
              </h2>
            }
            actions={
              !full && (
                <Button size="sm" variant="ghost" onClick={() => setAbRaw([domain[0], domain[1]])}>
                  Reset
                </Button>
              )
            }
          >
            <div className={styles.abFields}>
              <NumberField
                label="Lower limit a"
                prefix="a"
                value={ab[0]}
                min={domain[0]}
                max={ab[1]}
                onChange={(v) => setAbRaw(sanitizeAb([v, ab[1]], domain))}
              />
              <NumberField
                label="Upper limit b"
                prefix="b"
                value={ab[1]}
                min={ab[0]}
                max={domain[1]}
                onChange={(v) => setAbRaw(sanitizeAb([ab[0], v], domain))}
              />
            </div>
            <p className={styles.exact}>
              <Formula
                tex={`I = \\int_{${sig(ab[0], 4).replace('−', '-')}}^{${sig(ab[1], 4).replace('−', '-')}} f(x)\\,dx`}
              />
              <span className={styles.mono}>= {sig(exact, 15)}</span>
            </p>
            <p className={styles.hint}>Drag the ends of the interval on the plot.</p>
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
        <Tooltip content="Hatched ╱ where f is above the rule, ╲ where it is below; Gauss: against the dashed interpolant, which Gₘ integrates exactly.">
          <span className={styles.toolbarToggle}>
            <Toggle checked={showError} onChange={setShowError} label="Error area" showLabel />
          </span>
        </Tooltip>
      }
      stage={
        runs.length > 0 ? (
          <QuadStage
            f={problem.f}
            domain={domain}
            ab={ab}
            onAbChange={(v) => setAbRaw(v)}
            runs={runs.map((r, i) => ({
              id: r.sel.id,
              name: r.method.spec.name,
              slot: r.sel.slot,
              trace: r.result.trace,
              t: player.localT(i),
              error: r.error,
            }))}
            exact={exact}
            showError={showError}
            fade={player.playing && !player.reducedMotion}
            focusId={focus?.sel.id ?? null}
            onFocus={setFocusId}
          />
        ) : (
          <div />
        )
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: 'Error vs n',
          height: 300,
          actions: (
            <Tooltip content="Each method's own error estimate, dotted beside its true error.">
              <span className={styles.estToggle}>
                <Toggle
                  checked={showEst}
                  onChange={setShowEst}
                  label="Error estimate ε̂"
                  showLabel
                />
              </span>
            </Tooltip>
          ),
          content: (
            <div className={styles.chartPad}>
              <ErrorChart
                series={series}
                floor={floor}
                showEstimate={showEst}
                onSeek={(i, k) => {
                  if (runs[i]) setFocusId(runs[i].sel.id);
                  seek(k);
                }}
                ariaLabel={`Error against the number of nodes, log–log, for ${stageLabel}. ${runs
                  .map((r) => {
                    const last = r.result.trace[r.result.trace.length - 1];
                    return `${r.method.spec.name}: error ${sci((last?.info.error as number) ?? null, 2)} with ${int(nodesUsed(r.sel.id, last ?? ({ k: 0, info: {} } as Step)) ?? 0)} nodes`;
                  })
                  .join('; ')}.`}
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
              value={activeTab}
              onChange={setTab}
              idBase="quad-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
                ...(hasRomberg ? [{ id: 'romberg' as const, label: 'Romberg' }] : []),
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="quad-details" id={activeTab} focusable={false}>
              {activeTab === 'method' ? (
                <div className={styles.methodTab}>
                  <MethodCard
                    live={
                      focusStep && (
                        // Step k − 1 → k: the refinement that produced the estimate under the
                        // playhead (the rule above reads trace[k − 1] → trace[k]).
                        <StepBlock
                          k={focusK - 1}
                          label={focusK === 0 ? 'Start' : undefined}
                          reset={`${focus.sel.id}~${focus.sel.slot}~${focus.result.trace.length}`}
                        >
                          <StepReadout
                            step={focusStep}
                            doc={focus.method.doc}
                            est={estimateSymbol(focus.sel.id, focusStep)}
                          />
                        </StepBlock>
                      )
                    }
                    method={card(focus.method)}
                    slot={focus.sel.slot}
                    result={focus.result}
                    error={focus.error}
                    call={{
                      problem: problem.id,
                      options: runOptions,
                      params: focus.sel.params,
                    }}
                  />
                </div>
              ) : activeTab === 'steps' ? (
                <StepsTable
                  method={focus.sel.id}
                  name={focus.method.spec.name}
                  trace={focus.result.trace}
                  k={focusK}
                  onSelect={seek}
                />
              ) : romberg ? (
                <RombergTable
                  trace={romberg.result.trace}
                  k={player.localK(runs.indexOf(romberg))}
                  exact={exact}
                  onSelect={seek}
                />
              ) : null}
            </TabPanel>
          ) : null,
        },
      ]}
    />
  );
}
