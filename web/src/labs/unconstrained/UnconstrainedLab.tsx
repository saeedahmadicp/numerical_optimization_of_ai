/**
 * Unconstrained lab — the 42 methods of numopt.unconstrained (gradient, certified schedules,
 * momentum and acceleration, adaptive, Newton, quasi-Newton, conjugate gradient, trust region,
 * regularized Newton, derivative-free), TS ports replayed
 * step by step over the contour field of every 2-D test problem. Each step is drawn with the
 * geometry the method used to take it (geometry.ts), explained with its numbers (stepTex.ts),
 * seen from the side when it came from a line search (RayPanel), and tabulated.
 */
import './setup';
import {
  Suspense,
  useCallback,
  useMemo,
  useRef,
  useState,
  type PointerEvent as RPointerEvent,
} from 'react';
import { defaults, listMethods } from '../../core/registry';
import { listProblems } from '../../problems/registry';
import type { Problem2D } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES } from '../../ui/colors';
import {
  Formula,
  PlaybackBar,
  SegmentedControl,
  TabPanel,
  Tabs,
  Toggle,
} from '../../ui/components';
import {
  Contour2D,
  ConvergenceChart,
  IterationTable,
  LazySurface3D,
  overlappingPaths,
  mathBold,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  type MathRun,
  type PathSpec,
  type View2D,
} from '../../viz';
import { drawOverlays2D, type Overlay2D } from '../../viz/overlays2d';
import { useTracePlayer, usePlayerKeyboard } from '../../play/useTracePlayer';
import { getLab } from '../index';
import {
  KAxisControl,
  LabShell,
  MethodCard,
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
  type LabRun,
} from '../_shell';
import {
  DERIVATIVE_FREE,
  INF_NORM_KINDS,
  PROVEN_RATE,
  RAY_KINDS,
  kindOf,
  withLabDoc,
  type Kind,
} from './catalog';
import { ConstantHint } from './ConstantHint';
import { useStartOnField } from './useStartOnField';
import { needsConstant } from './constants';
import { buildGeometry, rayOf, seriesValues, stepAt, type Metric } from './geometry';
import { drawGeometryLayers, type GeometryLayer } from './draw';
import { ROW_HEIGHT, columnsFor } from './columns';
import { DEFAULT_PROBLEM, DEFAULT_SELECTION, KEYS, PRESETS } from './presets';
import { StageKey } from './StageKey';
import { StepEquation } from './StepEquation';
import { RayPanel } from './RayPanel';
import { searchRuleOf } from './search';
import { MethodPicker } from './MethodPicker';
import { railProblem } from './railTex';
import { MathText } from './MathText';
import { StepLens } from './StepLens';
import { lensBox, needsLens } from './lens';
import {
  HARD,
  W_FOCUS_PATH,
  W_MINIMUM,
  W_PATH,
  cornerStyle,
  pathMarks,
  pickCorner,
  type Corner,
  type Mark,
} from './lensPlace';
import styles from './UnconstrainedLab.module.css';

void preloadKatex();

type View = '2d' | '3d';
type Tab = 'method' | 'ray' | 'iter';

// Codecs are module constants so their identity is stable across renders.
const VIEW_CODEC = codecs.oneOf<View>(['2d', '3d']);
const METRIC_CODEC = codecs.oneOf<Metric>(['gap', 'grad', 'dist']);

const K_SUB = mathSub('k', 'italic');
const Y_NAME: Record<Metric, MathRun[]> = {
  gap: [
    mathVar('f'),
    mathMain('('),
    mathBold('x'),
    K_SUB,
    mathMain(') − '),
    mathVar('f'),
    mathSup('⋆'),
  ],
  grad: [
    mathMain('‖∇'),
    mathVar('f'),
    mathMain('('),
    mathBold('x'),
    K_SUB,
    mathMain(')‖'),
    mathSub('2'),
  ],
  dist: [
    mathMain('‖'),
    mathBold('x'),
    K_SUB,
    mathMain(' − '),
    mathBold('x'),
    mathSup('⋆'),
    mathMain('‖'),
  ],
};
const Y_LABEL: Record<Metric, string> = {
  gap: 'f(xₖ) − f⋆',
  grad: '‖∇f(xₖ)‖₂',
  dist: '‖xₖ − x⋆‖',
};
const METRIC_TITLE: Record<Metric, string> = {
  gap: 'Objective gap',
  grad: 'Gradient norm (Euclidean)',
  dist: 'Distance to the minimizer',
};

/** A pointer within this distance (CSS px) of 𝐱₀ drags it instead of panning the view. */
const GRAB_RADIUS = 11;

/** Weight of the geometry of methods that are not focused. */
const BACKGROUND_OPACITY = 0.38;

const paramsOf = (r: LabRun) => ({ ...defaults(r.method.spec), ...r.sel.params });

export default function UnconstrainedLab() {
  const lab = getLab('unconstrained')!;
  const colors = useChartColors();

  const problems = useMemo(
    () => listProblems<Problem2D>('unconstrained').filter((p) => p.dim === 2),
    [],
  );
  const methods = useMemo(() => listMethods('unconstrained'), []);
  const pickerProblems = useMemo(() => problems.map(railProblem), [problems]);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [urlX0, setX0] = useStartPoint(problem.x0 as number[] | undefined, 2);
  // While 𝐱₀ is dragged, the runs follow the pointer; the URL is written on release.
  const [dragX0, setDragX0] = useState<[number, number] | null>(null);
  const x0 = dragX0 ?? urlX0;
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [view, setView] = useUrlState<View>(KEYS.view, '2d', VIEW_CODEC);
  const [metric, setMetric] = useUrlState<Metric>(KEYS.metric, 'gap', METRIC_CODEC);
  const [focusId, setFocusId] = useUrlState<string>(KEYS.focus, '');
  const [lensOn, setLensOn] = useUrlState<boolean>(KEYS.lens, true, codecs.bool);
  const [tab, setTab] = useState<Tab>('method');
  const [hoverK, setHoverK] = useState<number | null>(null);
  // The landscape's current view (for the lens corner and for dragging 𝐱₀), and the lens corner:
  // undefined until measured, null when every corner would hide 𝐱₀ or the step.
  const viewRef = useRef<View2D | null>(null);
  const [corner, setCorner] = useState<Corner | null | undefined>(undefined);
  const cornerRef = useRef<Corner | null | undefined>(undefined);

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]] }), [x0]);
  // Synchronous: four runs of these methods take milliseconds, and a deferred render would be
  // restarted by every playback frame.
  const { runs, pending } = useLabRunsState(problem, selection, runOptions, { defer: false });
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  // Playback starts once the landscape is on screen (useStartOnField.ts), not at mount.
  const player = useTracePlayer(traces, { autoplay: false });
  usePlayerKeyboard(player);
  const onFieldReady = useStartOnField(player, traces, view === '2d' ? problem.id : null);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusKind: Kind | undefined = focus ? kindOf(focus.sel.id) : undefined;
  const focusN = focus?.result.trace.length ?? 0;
  const focusG = focus ? stepAt(player.localT(focusIndex), focusN).g : 0;

  const span = Math.max(
    problem.domain[0][1] - problem.domain[0][0],
    problem.domain[1][1] - problem.domain[1][0],
  );

  // ── landscape ─────────────────────────────────────────────────────────────────────────
  // A path that runs along another one would be hidden under it: it is drawn on top, dashed.
  const overlaps = useMemo(
    () =>
      overlappingPaths(
        runs.map((r) => r.result.trace.map((s) => s.x as [number, number])),
        // Only near-identical paths (two variants that take the same steps), not paths that
        // merely share a valley floor.
        span * 0.002,
      ),
    [runs, span],
  );
  const specs: PathSpec[] = useMemo(() => {
    const dashed = new Set(overlaps.map(([i]) => i));
    return runs.map((r, i) => ({
      points: r.result.trace.map((s) => s.x as [number, number]),
      color: colors.series[r.sel.slot],
      label: r.method.spec.name,
      dash: dashed.has(i) ? [7, 6] : undefined,
      start: false,
      end: r.result.converged ? ('converged' as const) : ('stopped' as const),
    }));
  }, [runs, colors, overlaps]);
  // Draw order: the others, then the focused method, then hidden (dashed) paths on top.
  const rank = (i: number) => (specs[i].dash ? 2 : i === focusIndex ? 1 : 0);
  const paths = specs
    .map((spec, i) => ({ spec, i }))
    .sort((a, b) => rank(a.i) - rank(b.i))
    .map((x) => x.spec);
  const overlapNote =
    overlaps.length > 0
      ? overlaps
          .map(([i, j]) => `${runs[i].method.spec.name} runs along ${runs[j].method.spec.name}`)
          .join('; ') + ' (dashed, drawn on top).'
      : null;

  const ease = !player.reducedMotion;
  const built = runs
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => kindOf(r.sel.id) && r.result.trace.length > 0)
    .map(({ r, i }) => {
      const { g, u } = stepAt(player.localT(i), r.result.trace.length);
      const focused = r === focus;
      const input = {
        kind: kindOf(r.sel.id)!,
        method: r.sel.id,
        trace: r.result.trace,
        g,
        params: paramsOf(r),
        problem,
        slot: r.sel.slot,
        labels: focused,
        span,
      };
      // A method that has finished while others play on steps back entirely (unless focused).
      const done = player.t > r.result.trace.length - 0.5;
      return { focused, done, input, geometry: buildGeometry(input), g, u };
    });
  const lensFor = built.find((b) => b.focused) ?? null;
  const box = focus && lensFor ? lensBox(lensFor.geometry, focus.result.trace, lensFor.g) : null;
  const showLens = view === '2d' && lensOn && needsLens(box, span);
  const lensShown = showLens && corner !== null;
  // The focused method is drawn last, on top; a new step fades in as the head sets off. While
  // the lens magnifies the focused step, its labels are drawn there, not on the landscape.
  const layers: GeometryLayer[] = [...built]
    .sort((a, b) => Number(a.focused) - Number(b.focused))
    .map((b) => ({
      geometry: b.focused && lensShown ? buildGeometry({ ...b.input, labels: false }) : b.geometry,
      opacity:
        (b.focused ? 1 : b.done ? 0 : BACKGROUND_OPACITY) *
        (ease && player.playing ? Math.min(1, b.u / 0.2) : 1),
    }));
  const ghosts: Overlay2D[] =
    hoverK === null
      ? []
      : runs.flatMap((r) => {
          const s = r.result.trace[Math.min(hoverK, r.result.trace.length - 1)];
          return s
            ? [
                {
                  kind: 'point' as const,
                  at: s.x as [number, number],
                  slot: r.sel.slot,
                  shape: 'ring' as const,
                  radius: 5,
                },
              ]
            : [];
        });
  // ── lens placement ────────────────────────────────────────────────────────────────────
  // The lens goes to the corner that hides the least: never 𝐱₀ or the focused step, then as
  // few minimizers and path points as possible (lensPlace.ts). It is measured on the view the
  // landscape draws, so a pan or zoom moves it too.
  const placed = useRef<{ key: string; token: object | null }>({ key: '', token: null });
  const placeToken = useMemo(() => ({ runs, minima: problem.minima }), [runs, problem]);
  const placeLens = (v: View2D) => {
    if (!showLens || !focus) return;
    const a = v.toPx(0, 0);
    const b = v.toPx(1, 1);
    const key = `${a[0]},${a[1]},${b[0]},${b[1]},${v.width},${v.height}|${focusIndex}|${focusG}|${x0[0]},${x0[1]}`;
    if (placed.current.key === key && placed.current.token === placeToken) return;
    placed.current = { key, token: placeToken };
    const P = (p: readonly unknown[]) => v.toPx(p[0] as number, p[1] as number);
    const marks: Mark[] = [];
    const s0 = P(x0);
    marks.push({ x: s0[0], y: s0[1], w: HARD });
    runs.forEach((r, i) =>
      marks.push(
        ...pathMarks(
          r.result.trace.map((s) => P(s.x as number[])),
          i === focusIndex ? W_FOCUS_PATH : W_PATH,
        ),
      ),
    );
    for (const q of (problem.minima ?? []) as number[][]) {
      const m = P(q);
      marks.push({ x: m[0], y: m[1], w: W_MINIMUM });
    }
    const t = focus.result.trace;
    if (focusG >= 1 && t[focusG])
      marks.push(...pathMarks([P(t[focusG - 1].x as number[]), P(t[focusG].x as number[])], HARD));
    const next = pickCorner(marks, v.width, v.height, cornerRef.current ?? null);
    if (next !== cornerRef.current) {
      cornerRef.current = next;
      setCorner(next);
    }
  };

  const overlay = (ctx: CanvasRenderingContext2D, v: View2D) => {
    viewRef.current = v;
    placeLens(v);
    drawGeometryLayers(ctx, v, layers);
    if (ghosts.length)
      drawOverlays2D(
        ctx,
        { toPx: v.toPx, width: v.width, height: v.height, dpr: v.dpr, colors: v.colors },
        ghosts,
      );
  };

  // ── dragging 𝐱₀ ──────────────────────────────────────────────────────────────────────
  // A mouse or pen press within GRAB_RADIUS of 𝐱₀ drags it (the landscape does not see the
  // gesture, so it does not pan); the runs follow once per frame, the URL is written on release.
  // Touch keeps tap-to-place: one finger scrolls the page.
  const drag = useRef<{ id: number; frame: number; next: [number, number] | null } | null>(null);
  const [grab, setGrab] = useState<'near' | 'drag' | null>(null);
  const plotPx = (e: RPointerEvent<HTMLDivElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const nearX0 = (p: [number, number]) => {
    const v = viewRef.current;
    if (!v) return false;
    const q = v.toPx(x0[0], x0[1]);
    return Math.hypot(q[0] - p[0], q[1] - p[1]) <= GRAB_RADIUS;
  };
  const onDragDown = (e: RPointerEvent<HTMLDivElement>) => {
    if (view !== '2d' || e.button !== 0 || e.pointerType === 'touch') return;
    if (!nearX0(plotPx(e))) return;
    e.stopPropagation();
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { id: e.pointerId, frame: 0, next: null };
    setGrab('drag');
    player.pause();
  };
  const onDragMove = (e: RPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) {
      if (e.pointerType !== 'touch' && view === '2d') {
        const near = nearX0(plotPx(e)) ? 'near' : null;
        if (near !== grab) setGrab(near);
      }
      return;
    }
    if (e.pointerId !== d.id) return;
    e.stopPropagation();
    const v = viewRef.current;
    if (!v) return;
    const [px, py] = plotPx(e);
    d.next = [Number(v.x.invert(px).toPrecision(4)), Number(v.y.invert(py).toPrecision(4))];
    if (!d.frame)
      d.frame = requestAnimationFrame(() => {
        d.frame = 0;
        if (d.next) setDragX0(d.next);
      });
  };
  const onDragEnd = (e: RPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    e.stopPropagation();
    cancelAnimationFrame(d.frame);
    drag.current = null;
    setGrab(null);
    setDragX0(null);
    if (d.next) setX0([d.next[0], d.next[1]]);
  };

  const f2 = useCallback((x: number, y: number) => problem.f([x, y]), [problem]);
  const pickStart = useCallback((p: [number, number]) => setX0([p[0], p[1]]), [setX0]);
  const minima = useMemo(() => (problem.minima ?? []) as [number, number][], [problem]);
  const fMin = useMemo(() => {
    const known = minima.map((q) => problem.f([q[0], q[1]]));
    return known.length ? Math.min(...known) : null;
  }, [minima, problem]);

  // ── convergence ───────────────────────────────────────────────────────────────────────
  const baseSeries = useMemo(
    () =>
      runs.map((r) => ({
        label: r.method.spec.name,
        slot: r.sel.slot,
        count: r.result.nIter,
        end: r.result.converged ? ('converged' as const) : ('stopped' as const),
        values: seriesValues(metric, r.result.trace, problem),
      })),
    [runs, metric, problem],
  );
  // The proven rate of the focused method only (labels of curves that end together collide).
  const series = baseSeries.map((sr, i) =>
    i === focusIndex && runs[i].result.converged
      ? { ...sr, rate: PROVEN_RATE[runs[i].sel.id] }
      : sr,
  );
  const [useLogK, setKAxis] = useKAxis(runs.map((r) => r.result.trace.length));
  const seek = (k: number) => {
    player.pause();
    player.seek(k);
  };
  const hasDf = runs.some((r) => DERIVATIVE_FREE.has(kindOf(r.sel.id) ?? 'line'));
  const infKinds = runs.some((r) => INF_NORM_KINDS.has(kindOf(r.sel.id) ?? 'line'));
  const hasFista = runs.some((r) => r.sel.id === 'fista');
  const chartNote =
    metric === 'grad'
      ? [
          infKinds
            ? 'Euclidean norm; Newton, quasi-Newton, CG, trust-region and regularized Newton methods stop on $\\|\\nabla f\\|_\\infty \\le \\mathrm{gtol}$, the others on $\\|\\nabla f\\|_2$.'
            : null,
          hasDf
            ? 'Derivative-free methods never evaluate $\\nabla f$; the lab evaluates it at their iterates.'
            : null,
          hasFista
            ? 'FISTA tests $\\nabla f$ at its extrapolated point $\\mathbf{y}_k$; the curve shows $\\nabla f(\\mathbf{x}_k)$, evaluated by the lab.'
            : null,
        ]
          .filter(Boolean)
          .join(' ') || null
      : `$\\mathbf{x}^\\star$ is the known minimizer nearest each run’s last iterate${metric === 'gap' ? ', $f^\\star = f(\\mathbf{x}^\\star)$' : ''}.`;

  // ── details ───────────────────────────────────────────────────────────────────────────
  const focusRay = focus && focusKind ? rayOf(focusKind, focus.result.trace, focusG) : null;
  const hasRayTab = focusKind !== undefined && RAY_KINDS.has(focusKind);
  const tabs: { id: Tab; label: string }[] = [
    { id: 'method', label: 'Method' },
    ...(hasRayTab ? [{ id: 'ray' as const, label: 'Line search' }] : []),
    { id: 'iter', label: 'Iterations' },
  ];
  const activeTab: Tab = tabs.some((t) => t.id === tab) ? tab : 'method';
  // One object per run (not per frame): "This step" reads its layout samples once per run.
  const focusParams = useMemo(() => (focus ? paramsOf(focus) : {}), [focus]);
  const cardMethod = useMemo(() => (focus ? withLabDoc(focus.method) : null), [focus]);

  const ariaLabel =
    `Contour plot of ${problem.name} with ` +
    runs
      .map((r, i) => {
        const s = r.result.trace[player.localK(i)];
        return `${r.method.spec.name} at step ${s?.k ?? 0}, f = ${s?.fun?.toPrecision(4) ?? '—'}`;
      })
      .join('; ') +
    '.';

  return (
    <LabShell
      lab={lab}
      presets={PRESETS}
      pending={pending}
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
            <MethodPicker available={methods} value={selection} onChange={setSelection} />
          </RailSection>
          <RailSection title="Start point">
            <StartPointFields value={x0} onChange={(v) => setX0([v[0], v[1]])} variable="𝐱₀" drag />
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
        <div className={styles.toolbar}>
          {view === '2d' && (
            <label className={styles.lensToggle}>
              <Toggle checked={lensOn} onChange={setLensOn} label="Step lens" />
              <span aria-hidden="true">Lens</span>
            </label>
          )}
          <SegmentedControl
            label="View"
            value={view}
            onChange={setView}
            options={[
              { value: '2d', label: '2-D' },
              { value: '3d', label: '3-D' },
            ]}
          />
        </div>
      }
      stage={
        <div className={styles.stage}>
          <div className={styles.plot}>
            {view === '3d' ? (
              <Suspense fallback={<div className={styles.loading}>Loading the 3-D view…</div>}>
                <LazySurface3D
                  f={f2}
                  domain={problem.domain}
                  paths={paths}
                  t={player.t}
                  ariaLabel={`${problem.name} as a surface with the paths of ${runs.map((r) => r.method.spec.name).join(', ')}.`}
                />
              </Suspense>
            ) : (
              <div
                className={styles.dragWrap}
                data-grab={grab ?? undefined}
                onPointerDownCapture={onDragDown}
                onPointerMove={onDragMove}
                onPointerUp={onDragEnd}
                onPointerCancel={onDragEnd}
              >
                <Contour2D
                  f={f2}
                  domain={problem.domain}
                  cacheKey={problem.id}
                  onFieldReady={onFieldReady}
                  fMin={fMin}
                  paths={paths}
                  t={player.t}
                  ease={ease}
                  minima={minima}
                  start={[x0[0], x0[1]]}
                  overlay={overlay}
                  onPick={pickStart}
                  ariaLabel={ariaLabel}
                />
              </div>
            )}
            {showLens && corner && focus && lensFor && box && (
              <StepLens
                style={cornerStyle(corner)}
                problem={problem}
                geometry={lensFor.geometry}
                box={box}
                trace={focus.result.trace}
                k={lensFor.g}
                u={lensFor.u}
                ease={ease}
                slot={focus.sel.slot}
                name={focus.method.spec.name}
              />
            )}
          </div>
          {focus && focusKind && (
            <div className={styles.keyBar}>
              <StageKey
                kind={focusKind}
                method={focus.sel.id}
                name={focus.method.spec.name}
                slot={focus.sel.slot}
                params={focusParams}
              />
              {view === '3d' ? (
                <p className={styles.keyNote}>
                  The 3-D view shows the paths; step geometry is drawn in 2-D.
                </p>
              ) : (
                overlapNote && <p className={styles.keyNote}>{overlapNote}</p>
              )}
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
          title: 'Convergence',
          // The header takes two rows where the measures do not fit beside the axis control.
          height: 332,
          actions: (
            <div className={styles.chartActions}>
              <KAxisControl logX={useLogK} onChange={setKAxis} />
              <SegmentedControl
                className={styles.metricSeg}
                label="Convergence measure"
                value={metric}
                onChange={setMetric}
                options={[
                  {
                    value: 'gap',
                    label: <Formula tex="f - f^\star" />,
                    ariaLabel: 'Objective gap f − f⋆',
                  },
                  {
                    value: 'grad',
                    label: <Formula tex="\|\nabla f\|_2" />,
                    ariaLabel: 'Gradient norm ‖∇f‖₂',
                  },
                  {
                    value: 'dist',
                    label: <Formula tex="\|\mathbf{x} - \mathbf{x}^\star\|" />,
                    ariaLabel: 'Distance to the minimizer',
                  },
                ]}
              />
            </div>
          ),
          content: (
            <div className={styles.chartWrap}>
              <div className={styles.chart}>
                <ConvergenceChart
                  series={series}
                  t={player.t}
                  yLabel={Y_LABEL[metric]}
                  yName={Y_NAME[metric]}
                  logX={useLogK}
                  onSeek={seek}
                  onHover={setHoverK}
                  ariaLabel={`${METRIC_TITLE[metric]} against iteration k for ${runs.map((r) => r.method.spec.name).join(', ')}.`}
                />
              </div>
              <p className={styles.chartNote}>{chartNote && <MathText text={chartNote} />}</p>
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
              idBase="unc-details"
              items={tabs}
            />
          ),
          // The method shown here and in the key under the landscape is picked in this panel's
          // header (LabShell `focus`).
          content:
            focus && focusKind ? (
              <TabPanel idBase="unc-details" id={activeTab} focusable={false}>
                {activeTab === 'method' ? (
                  <div className={styles.methodTab}>
                    {cardMethod && (
                      <MethodCard
                        live={
                          focus.error ? (
                            needsConstant(focus.error) && (
                              <ConstantHint
                                problem={problem}
                                method={focus.sel.id}
                                name={focus.method.spec.name}
                                slot={focus.sel.slot}
                                onUse={(p) =>
                                  setSelection(
                                    selection.map((sel) =>
                                      sel.id === focus.sel.id
                                        ? { ...sel, params: { ...sel.params, ...p } }
                                        : sel,
                                    ),
                                  )
                                }
                              />
                            )
                          ) : (
                            <StepEquation
                              kind={focusKind}
                              method={focus.sel.id}
                              trace={focus.result.trace}
                              g={focusG}
                              params={focusParams}
                              problem={problem}
                              slot={focus.sel.slot}
                            />
                          )
                        }
                        method={cardMethod}
                        slot={focus.sel.slot}
                        result={focus.result}
                        error={focus.error}
                        call={{
                          problem: problem.id,
                          options: { x0: [x0[0], x0[1]] },
                          params: focus.sel.params,
                        }}
                      />
                    )}
                  </div>
                ) : activeTab === 'ray' ? (
                  focusRay ? (
                    <RayPanel
                      ray={focusRay}
                      problem={problem}
                      slot={focus.sel.slot}
                      k={focusG}
                      search={searchRuleOf(focus.sel.id, focusParams)}
                      c2={typeof focusParams.c2 === 'number' ? focusParams.c2 : 0.9}
                      fRef={
                        typeof focus.result.trace[focusG]?.info.f_ref === 'number'
                          ? (focus.result.trace[focusG].info.f_ref as number)
                          : null
                      }
                    />
                  ) : (
                    <p className={styles.empty}>
                      {focusG === 0
                        ? 'Step 0 is the start. Step forward to see the first line search.'
                        : 'This step has no line search (it did not move along a direction).'}
                    </p>
                  )
                ) : (
                  <IterationTable
                    steps={focus.result.trace}
                    rowHeight={ROW_HEIGHT}
                    columns={columnsFor(focusKind)}
                    k={focusG}
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
