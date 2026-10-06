/**
 * Differentiation lab: estimate f′(x0) (or f″(x0)) from values of f, on a sweep of steps
 * h_k = h0/2^k shared by every method. The stage shows the stencil at the current h, magnified
 * (it zooms with h, so the chords converge to the tangent until rounding turns f into a
 * staircase), and the error against h on log–log axes: the V of truncation (∝ hᵖ) against
 * round-off (∝ ε/hᵠ), with the complex step flat at ε.
 */
import './setup';
import { useCallback, useMemo, useState } from 'react';
import { listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { CalculusProblem } from '../../problems/calculus';
import type { ParamValue, RunOptions, Step } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { sci, sig, sigFixed } from '../../core/format';
import { preloadKatex } from '../../ui/katex';
import {
  Formula,
  NumberField,
  ParamControls,
  PlaybackBar,
  SegmentedControl,
  Swatch,
  TabPanel,
  Tabs,
  Toggle,
} from '../../ui/components';
import { TableView, type TableColumn } from '../../viz';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { MAX_SERIES } from '../../ui/colors';
import { getLab } from '../index';
import {
  LabShell,
  MethodSlots,
  ProblemPicker,
  RailSection,
  RunSummary,
  useLabRuns,
  useMethodSelection,
  useProblemState,
  useStartPoint,
  type LabRun,
  type RunStatus,
  type RunSummaryItem,
  type IterationNoun,
} from '../_shell';
import {
  DEFAULT_H0,
  DEFAULT_LEVELS,
  DEFAULT_PROBLEM,
  DEFAULT_SELECTION,
  DEFAULT_TOL,
  KEYS,
  PRESETS,
  SWEEP_SPECS,
} from './config';
import { REACH, SECOND_DERIVATIVE, errorsOf, hAt, windowHalfWidth, type CurveMode } from './model';
import { StencilView } from './StencilView';
import { ErrorChart } from './ErrorChart';
import { OverviewStrip } from './OverviewStrip';
import { DiffCard } from './DiffCard';
import { Tableau } from './Tableau';
import { useMediaQuery } from './useMediaQuery';
import { Reserve } from './Reserve';
import { widest } from './widest';
import styles from './DifferentiationLab.module.css';

void preloadKatex();

const NUM = codecs.number;
const VIEW_CODEC = codecs.oneOf<CurveMode>(['dev', 'f']);
const SWEEP_HIDDEN = ['h0', 'levels', 'tol'];
const EPS = Number.EPSILON;

type Tab = 'method' | 'levels' | 'tableau';

/** One step of the sweep is a level h_k = h0/2^k: "Converged in 12 levels". */
const LEVEL_NOUN: IterationNoun = ['level', 'levels'];

/**
 * The run's badge in the stage header: the selected step h⋆ in place of the level count, in the
 * shared badge (✓ when a later level confirms its error bound, △ when none does). A run with no
 * selected step keeps the default badge; the full status is the tooltip and accessible text.
 */
function hStarBadge(run: RunSummaryItem, st: RunStatus) {
  const kb = run.result.extra.k_best as number | null | undefined;
  const h =
    typeof kb === 'number' ? (run.result.trace[kb]?.info.h as number | undefined) : undefined;
  if (run.error || typeof h !== 'number') return undefined;
  return (
    <>
      <span aria-hidden="true">{st.icon}</span> h⋆ = {sci(h, 2)}
    </>
  );
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** A number for KaTeX with 4 significant digits (− as a TeX minus). */
const texSig = (v: number) => sig(v, 4).replace('−', '-');

/** The levels table of one run (synced to the playhead). */
function levelColumns(kBest: number | null, second: boolean): TableColumn<Step>[] {
  const d = second ? 'D_2' : 'D';
  const f = second ? "f''" : "f'";
  return [
    { key: 'k', tex: 'k', value: (s) => s.k, width: '2rem' },
    { key: 'h', tex: 'h_k', value: (s) => sci(s.info.h as number, 2) },
    { key: 'D', tex: `${d}(h_k)`, value: (s) => sigFixed(s.info.estimate as number | null, 10) },
    {
      key: 'err',
      tex: `|${d} - ${f}|`,
      value: (s) => {
        const e = s.info.error as number | null;
        return e === null ? '—' : e === 0 ? '0' : sci(e, 2);
      },
    },
    {
      key: 'note',
      header: <span className="visually-hidden">Level status</span>,
      align: 'left',
      mono: false,
      value: (s) =>
        s.k === kBest ? (
          <span className={styles.flagBest} title="The selected level">
            <Formula tex="h^\star" />
          </span>
        ) : s.info.collapsed ? (
          <span className={styles.flag} title="Rounding merged the stencil's abscissae">
            coll.
          </span>
        ) : s.info.roundoff_obs !== null && !s.info.confirms ? (
          <span
            className={styles.flag}
            title="After the cut c: dominated by round-off, may not confirm other levels"
          >
            noise
          </span>
        ) : (
          ''
        ),
    },
  ];
}

const estimateText = (s: Step | undefined) => sigFixed(s?.info.estimate as number | null, 10);
const errText = (s: Step | undefined) => {
  const e = s?.info.error as number | null | undefined;
  return e === null || e === undefined ? '' : e === 0 ? 'exact' : `err ${sci(e, 1)}`;
};
/** The widest estimate and error text of a run (a trace is immutable: cached). */
const widthCache = new WeakMap<readonly Step[], { value: string[]; err: string[] }>();
function readoutWidths(trace: readonly Step[]) {
  let w = widthCache.get(trace);
  if (!w) {
    w = { value: widest(trace.map(estimateText)), err: widest(trace.map(errText)) };
    widthCache.set(trace, w);
  }
  return w;
}

const hCache = new WeakMap<readonly Step[], string[]>();
function hWidestOf(trace: readonly Step[]) {
  let w = hCache.get(trace);
  if (w === undefined)
    hCache.set(trace, (w = widest(trace.map((s) => sci(s.info.h as number, 3)))));
  return w;
}

interface SelectRow {
  run: LabRun;
}

export default function DifferentiationLab() {
  const lab = getLab('differentiation')!;
  const problems = useMemo(() => listProblems<CalculusProblem>('calculus'), []);
  const methods = useMemo(() => listMethods('differentiation'), []);
  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [x0Arr, setX0Arr] = useStartPoint(problem.x0, 1);
  const x0 = x0Arr[0];
  const setX0 = useCallback((v: number) => setX0Arr([v]), [setX0Arr]);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [h0Raw, setH0] = useUrlState(KEYS.h0, DEFAULT_H0, NUM);
  const [levelsRaw, setLevels] = useUrlState(KEYS.levels, DEFAULT_LEVELS, NUM);
  const [tolRaw, setTol] = useUrlState(KEYS.tol, DEFAULT_TOL, NUM);
  const [mode, setMode] = useUrlState<CurveMode>(KEYS.view, 'dev', VIEW_CODEC);
  const [showModel, setShowModel] = useState(true);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('method');
  const [tableShow, setTableShow] = useState<'error' | 'value'>('error');
  const narrow = useMediaQuery('(max-width: 899px)');

  const h0 = h0Raw > 0 ? clamp(h0Raw, 1e-30, 10) : DEFAULT_H0;
  const levels = Math.round(clamp(levelsRaw, 0, 60));
  const tol = tolRaw > 0 ? clamp(tolRaw, 1e-15, 0.1) : DEFAULT_TOL;

  // One sweep for every method: h0, levels and tol override the per-method values.
  const options = useMemo(
    () => ({ x0: [x0], h0, levels, tol }) as unknown as RunOptions,
    [x0, h0, levels, tol],
  );
  const runs = useLabRuns(problem, selection, options);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);
  const seek = useCallback(
    (k: number) => {
      player.pause();
      player.seek(k);
    },
    [player],
  );

  const drawable = runs.map((r, i) => ({ r, i })).filter(({ r }) => r.result.trace.length > 0);
  const focus = drawable.find(({ r }) => r.sel.id === focusId) ?? drawable[0];
  const focusIdx = focus ? drawable.indexOf(focus) : 0;
  const focusK = focus ? player.localK(focus.i) : 0;
  const focusStep = focus?.r.result.trace[focusK];
  const focusSecond = focus ? SECOND_DERIVATIVE.has(focus.r.sel.id) : false;

  const slope = problem.grad(x0);
  const curvature = problem.hess(x0);
  const exactOf = (id: string) => (SECOND_DERIVATIVE.has(id) ? curvature : slope);
  const floor = Number.isFinite(focusSecond ? curvature : slope)
    ? EPS * Math.abs(focusSecond ? curvature : slope) || null
    : null;
  const ease = player.playing && !player.reducedMotion;
  // A paused playhead shows one level: in-between frames exist only while playing.
  const tOf = (i: number) => (player.playing ? player.localT(i) : player.localK(i));

  const stencilRuns = drawable.map(({ r, i }) => ({
    id: r.sel.id,
    slot: r.sel.slot,
    trace: r.result.trace,
    t: tOf(i),
  }));
  const errorRuns = drawable.map(({ r, i }) => ({
    id: r.sel.id,
    name: r.method.spec.name,
    slot: r.sel.slot,
    trace: r.result.trace,
    t: tOf(i),
    kBest: (r.result.extra.k_best as number | null) ?? null,
    hOpt: (r.result.extra.h_opt as number | null) ?? null,
    order: (r.result.extra.order as number) ?? 1,
    derivative: (r.result.extra.derivative as number) ?? 1,
  }));

  const hNow = focus ? hAt(focus.r.result.trace, tOf(focus.i)) : h0;
  const windowW = windowHalfWidth(hNow, focus ? (REACH[focus.r.sel.id] ?? 1) : 1);
  const overviewDomain = useMemo<[number, number]>(() => {
    const [a, b] = problem.domain;
    const pad = (b - a) * 0.15;
    return [a - pad, b + pad];
  }, [problem]);

  const [domLo, domHi] = problem.domain;
  const undefinedAtX0 = !Number.isFinite(problem.f(x0));

  const richardson = drawable.find(({ r }) => r.sel.id === 'richardson_extrapolation');
  const activeTab: Tab = tab === 'tableau' && !richardson ? 'method' : tab;

  // ── Accessible summaries ────────────────────────────────────────────────────────────────
  const stencilLabel = focus
    ? `${mode === 'dev' ? 'f minus its tangent' : 'f'} near x₀ = ${sig(x0, 4)}, magnified to the stencil of ${focus.r.method.spec.name} at h = ${sci(focusStep?.info.h as number, 3)}: estimate ${sig(focusStep?.info.estimate as number | null, 10)}, error ${sci(focusStep?.info.error as number | null, 2)}.`
    : 'No method selected.';
  const errorLabel =
    'Error against step h, log–log. ' +
    drawable
      .map(({ r }) => {
        const es = errorsOf(r.result.trace);
        let best = -1;
        es.forEach((e, k) => {
          if (e !== null && (best < 0 || e < (es[best] as number))) best = k;
        });
        const kb = r.result.extra.k_best as number | null;
        return `${r.method.spec.name}: smallest error ${best >= 0 ? sci(es[best], 2) : '—'} at h = ${best >= 0 ? sci(r.result.trace[best].info.h as number, 2) : '—'}${kb !== null ? `; selected h⋆ = ${sci(r.result.trace[kb].info.h as number, 2)}` : ''}.`;
      })
      .join(' ');

  // ── Panels ──────────────────────────────────────────────────────────────────────────────
  // The widest h of the focused run: the stencil title keeps one width for the whole replay.
  const hWidest = focus ? hWidestOf(focus.r.result.trace) : [];
  const readouts = (
    <ul className={styles.readouts} aria-label="Estimates at this step">
      {drawable.map(({ r, i }) => {
        const s = r.result.trace[player.localK(i)];
        const w = readoutWidths(r.result.trace);
        return (
          <li key={r.sel.id} className={styles.readout}>
            <Swatch slot={r.sel.slot} size={8} />
            <span className={styles.readoutName} aria-hidden="true">
              {r.method.spec.name}
            </span>
            <span className="visually-hidden">{r.method.spec.name}: </span>
            <Reserve widest={w.value} className={styles.readoutValue}>
              {estimateText(s)}
            </Reserve>
            <Reserve widest={w.err} className={styles.readoutErr}>
              {errText(s)}
            </Reserve>
          </li>
        );
      })}
    </ul>
  );

  const stencilPanel = (
    <section className={styles.panel} aria-label="Stencil at the current step">
      <header className={styles.panelHead}>
        <h2 className={styles.panelTitle}>
          Stencil at <Formula tex="h" /> ={' '}
          <Reserve widest={hWidest} className={styles.num}>
            {sci(focusStep?.info.h as number, 3)}
          </Reserve>
          <span className={styles.panelMeta}>
            level{' '}
            <Reserve widest={String(levels)} align="end" className={styles.num}>
              {focusK}
            </Reserve>{' '}
            of <span className={styles.num}>{levels}</span>
          </span>
          {focusStep?.info.collapsed === true && (
            <span
              className={styles.collapsed}
              title="fl(x₀ + oh) are not distinct: rounding merged the stencil's abscissae"
            >
              abscissae collapsed
            </span>
          )}
        </h2>
        <p className={styles.panelNote}>
          {mode === 'dev' ? (
            <>
              <Formula tex="f - \ell" /> with the tangent <Formula tex="\ell" />
              {focusSecond ? (
                <>
                  ; dashed: <Formula tex="\tfrac12 f''(x_0)(x - x_0)^2" />
                </>
              ) : (
                <>: a line&apos;s slope here is its error</>
              )}
            </>
          ) : (
            <>
              <Formula tex="f(x) - f(x_0)" /> and the tangent <Formula tex="\ell" />
            </>
          )}
        </p>
      </header>
      {readouts}
      <div className={styles.panelBody}>
        <StencilView
          f={problem.f}
          x0={x0}
          slope={slope}
          curvature={Number.isFinite(curvature) ? curvature : null}
          runs={stencilRuns}
          focus={focusIdx}
          mode={mode}
          ease={ease}
          ariaLabel={stencilLabel}
        />
      </div>
      <OverviewStrip
        f={problem.f}
        domain={overviewDomain}
        limits={problem.domain}
        x0={x0}
        window={windowW}
        onChange={setX0}
      />
    </section>
  );

  const errorChart = (
    <ErrorChart
      runs={errorRuns}
      focus={focusIdx}
      floor={floor}
      showModel={showModel}
      ease={ease}
      onSeek={seek}
      ariaLabel={errorLabel}
    />
  );
  const modelToggle = (
    <Toggle checked={showModel} onChange={setShowModel} label="Error model" showLabel />
  );

  const errorPanel = (
    <section className={styles.panel} aria-label="Error vs step h">
      <header className={styles.panelHead}>
        <h2 className={styles.panelTitle}>
          Error vs step <Formula tex="h" />
        </h2>
        {modelToggle}
      </header>
      <div className={styles.panelBody}>{errorChart}</div>
    </section>
  );

  // ── Insights ────────────────────────────────────────────────────────────────────────────
  const selectColumns: TableColumn<SelectRow>[] = [
    {
      key: 'm',
      header: 'Method',
      align: 'left',
      mono: false,
      value: ({ run }) => (
        <span className={styles.methodCell}>
          <Swatch slot={run.sel.slot} size={8} />
          {run.method.spec.name}
          {run.error ? (
            <span className={styles.bad} title={run.error} aria-label="invalid input">
              ×
            </span>
          ) : run.result.converged ? (
            <span className={styles.ok} aria-label="converged">
              ✓
            </span>
          ) : (
            <span className={styles.warn} aria-label="not converged" title={run.result.message}>
              ◷
            </span>
          )}
        </span>
      ),
    },
    {
      key: 'h',
      header: (
        <>
          <span aria-hidden="true">
            <Formula tex="h^\star" />
          </span>
          <span className="visually-hidden">Selected step</span>
        </>
      ),
      value: ({ run }) => {
        const kb = run.result.extra.k_best as number | null;
        return kb === null || kb === undefined
          ? '—'
          : sci(run.result.trace[kb].info.h as number, 2);
      },
    },
    {
      key: 'b',
      header: 'Bound',
      value: ({ run }) => sci(run.result.extra.bound_best as number | null, 2),
    },
    {
      key: 'e',
      header: 'Error',
      value: ({ run }) => {
        const e = run.result.extra.error as number | null;
        return e === null || e === undefined ? '—' : e === 0 ? '0' : sci(e, 2);
      },
    },
  ];

  const details = focus ? (
    <TabPanel idBase="diff-details" id={activeTab} focusable={false}>
      {activeTab === 'method' ? (
        <DiffCard
          method={focus.r.method}
          slot={focus.r.sel.slot}
          step={focusStep}
          prev={focusK > 0 ? focus.r.result.trace[focusK - 1] : undefined}
          result={focus.r.result}
          error={focus.r.error}
          exact={exactOf(focus.r.sel.id)}
          levels={levels}
          call={{ problem: problem.id, x0, params: { ...focus.r.sel.params, h0, levels, tol } }}
        />
      ) : activeTab === 'levels' ? (
        <div className={styles.fill}>
          <TableView
            columns={levelColumns(
              (focus.r.result.extra.k_best as number | null) ?? null,
              focusSecond,
            )}
            rows={focus.r.result.trace}
            highlight={focusK}
            futureAfter={focusK}
            onSelect={seek}
            maxHeight="100%"
            className={styles.compact}
            ariaLabel={`${focus.r.method.spec.name} levels`}
          />
        </div>
      ) : richardson ? (
        <div className={styles.fillCol}>
          <div className={styles.tableauBar}>
            <p className={styles.tableauNote}>
              Column <Formula tex="j" /> removes the <Formula tex="h^{2j}" /> term; the diagonal{' '}
              <Formula tex="D(k, k)" /> is the estimate.
            </p>
            <SegmentedControl
              label="Table cells"
              value={tableShow}
              onChange={setTableShow}
              options={[
                { value: 'error', label: 'Error' },
                { value: 'value', label: 'Value' },
              ]}
            />
          </div>
          <Tableau
            trace={richardson.r.result.trace}
            k={player.localK(richardson.i)}
            exact={slope}
            show={tableShow}
            onSelect={seek}
          />
        </div>
      ) : null}
    </TabPanel>
  ) : null;

  const tabs: { id: Tab; label: string }[] = [
    { id: 'method', label: 'Method' },
    { id: 'levels', label: 'Levels' },
    ...(richardson ? [{ id: 'tableau' as const, label: 'Richardson table' }] : []),
  ];

  const insights = [
    ...(narrow
      ? [
          {
            id: 'error',
            label: 'Error vs step h',
            title: (
              <h2 className={styles.insightTitle}>
                Error vs step <Formula tex="h" />
              </h2>
            ),
            height: 340,
            actions: modelToggle,
            content: <div className={styles.chartPad}>{errorChart}</div>,
          },
        ]
      : []),
    {
      id: 'select',
      title: 'Selected step',
      // A row of two lines (a long name wraps) is 46 px.
      height: 64 + 46 * Math.max(1, drawable.length),
      // One column on a phone: the panel takes the height of its table, not a fixed 320 px.
      wide: narrow || undefined,
      content: (
        <div className={narrow ? styles.flow : styles.fill}>
          <TableView
            columns={selectColumns}
            rows={drawable.map(({ r }) => ({ run: r }))}
            rowKey={(row) => row.run.sel.id}
            maxHeight="100%"
            className={styles.compact}
            ariaLabel="Selected step of each method"
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
          idBase="diff-details"
          items={tabs}
        />
      ),
      content: details,
    },
  ];

  const sweepValues: Record<string, ParamValue> = { h0, levels, tol };
  const onSweep = (name: string, v: ParamValue) => {
    if (name === 'h0') setH0(Number(v));
    else if (name === 'levels') setLevels(Number(v));
    else if (name === 'tol') setTol(Number(v));
  };

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      focus={
        focus && {
          runs: drawable.map(({ r }) => ({
            value: r.sel.id,
            slot: r.sel.slot,
            name: r.method.spec.name,
          })),
          value: focus.r.sel.id,
          onChange: setFocusId,
        }
      }
      stageNotice={
        runs.length === 0 ? (
          'Add a method to compare — up to four run on one clock.'
        ) : undefinedAtX0 ? (
          <>
            <Formula tex="f" /> is not defined at <Formula tex={`x_0 = ${texSig(x0)}`} />: every
            stencil needs <Formula tex="f(x_0)" /> or its neighbors. Choose <Formula tex="x_0" /> in
            the domain <Formula tex={`[${texSig(domLo)}, ${texSig(domHi)}]`} />.
          </>
        ) : undefined
      }
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection title="Point">
            <NumberField label="Point of differentiation" prefix="x₀" value={x0} onChange={setX0} />
            <p className={styles.hint}>
              or drag along the curve under the stencil.{' '}
              <span className={styles.exact}>
                <Formula
                  tex={`f'(x_0) = ${Number.isFinite(slope) ? sig(slope, 10).replace('−', '-') : '\\text{—}'}`}
                />
              </span>
            </p>
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
              available={methods}
              value={selection}
              onChange={setSelection}
              hiddenParams={SWEEP_HIDDEN}
            />
          </RailSection>
          <RailSection title="Step sweep">
            <ParamControls
              specs={SWEEP_SPECS}
              values={sweepValues}
              onChange={onSweep}
              labelPrefix="Sweep"
            />
            <p className={styles.hint}>
              <Formula tex={`h_k = h_0 / 2^k,\\ k = 0, \\dots, ${levels}`} /> — down to{' '}
              {sci(h0 / 2 ** levels, 2)}, for every method.
            </p>
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
            iterationNoun={LEVEL_NOUN}
            badge={hStarBadge}
          />
        </>
      }
      stageToolbar={
        <SegmentedControl
          label="Curve in the stencil view"
          value={mode}
          onChange={setMode}
          options={[
            {
              value: 'dev',
              label: <Formula tex="f - \ell" />,
              ariaLabel: 'f minus its tangent ℓ: the error made visible',
            },
            {
              value: 'f',
              label: <Formula tex="f" />,
              ariaLabel: 'f itself, minus the constant f(x₀)',
            },
          ]}
        />
      }
      stage={
        narrow ? (
          <div className={styles.single}>{stencilPanel}</div>
        ) : (
          <div className={styles.split}>
            {stencilPanel}
            {errorPanel}
          </div>
        )
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={insights}
    />
  );
}
