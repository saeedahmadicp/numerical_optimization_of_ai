/**
 * Nonlinear systems lab: Newton and Broyden on F(𝐱) = 𝟎 in the plane.
 *
 * The stage draws the zero curves F₁ = 0 (solid ink) and F₂ = 0 (dashed ink) over a faint ‖F‖
 * field; their crossings are the roots. For the focused method it draws the geometry of the
 * current step from `Step.info`: the zero lines ℓ₁, ℓ₂ of the linear models
 * Lᵢ(𝐱) = Fᵢ(𝐱ₖ) + ∇Fᵢ(𝐱ₖ)·(𝐱 − 𝐱ₖ) (solid / dashed, like the curves they approximate; they are
 * not tangent to the curves unless 𝐱ₖ lies on them) in the method's color, the full step 𝐩ₖ as an
 * arrow to their crossing, the backtracking trials of damped Newton, and for Broyden the zero lines
 * of the exact J(𝐱ₖ) in gray beside the model's. A toggle replaces the field with the basins of
 * attraction of the focused method (computed in workers).
 */
import './setup';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { listMethods } from '../../core/registry';
import type { Matrix, Step, Vector } from '../../core/types';
import { codecs, useUrlState } from '../../app/useUrlState';
import { useChartColors } from '../../ui/theme';
import { preloadKatex } from '../../ui/katex';
import { MAX_SERIES, oklch, rgbToHex, type ChartColors, type RGB } from '../../ui/colors';
import { Formula, PlaybackBar, SegmentedControl, TabPanel, Tabs } from '../../ui/components';
import { sigFixed, vecFixed, sig } from '../../core/format';
import {
  Contour2D,
  ConvergenceChart,
  IterationTable,
  drawAxes,
  drawOverlays2D,
  mathBold,
  mathMain,
  mathSub,
  mathSup,
  mathVar,
  type Column,
  type Overlay2D,
  type PathSpec,
  suggestLogK,
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
  type MethodSelection,
} from '../_shell';
import { KAxisControl } from '../_shell/blocks';
import { listSystems } from '../../problems/systems';
import {
  geometryIndex,
  linearizationLine,
  modelError,
  nearestRoot,
  rootIndex,
  stepGeometry,
  type StepGeometry,
} from './geometry';
import { NO_ROOT, OTHER_ROOT } from './basins';
import { useBasins, type BasinGrid } from './useBasins';
import { drawZeroCurves, type ZeroCurve } from './curves';
import { StepPanel } from './StepPanel';
import { texPt } from './tex';
import { powerOfTwo } from './format';
import { PRESETS } from './presets';
import { drawOffView, type OffViewRun, type Rect } from './offview';
import styles from './SystemsLab.module.css';

void preloadKatex();

const DEFAULT_PROBLEM = 'intersecting_circles';

/** Newton against Broyden from the problem's start: both converge, at different rates. */
const DEFAULT_SELECTION: MethodSelection[] = [
  { id: 'newton_system', slot: 0, params: {} },
  { id: 'broyden', slot: 1, params: {} },
];

type FieldView = 'norm' | 'basins';
type Metric = 'res' | 'err';
type KAxis = 'auto' | 'lin' | 'log';
const FIELD_CODEC = codecs.oneOf<FieldView>(['norm', 'basins']);
const METRIC_CODEC = codecs.oneOf<Metric>(['res', 'err']);
const KAXIS_CODEC = codecs.oneOf<KAxis>(['auto', 'lin', 'log']);

/** Short names for the stage chips. */
const SHORT: Record<string, string> = { newton_system: 'Newton', broyden: 'Broyden' };

// ── Basin colors ────────────────────────────────────────────────────────────────────────────

/*
 * The basins sit at the light end of the band of the contour fills (light: L ≥ 0.89; dark: L ≤
 * 0.35), so the blue of slot 1 keeps ≥ 3:1 against every cell in both themes (3.2–4.0 light,
 * 3.1–4.5 dark), and the other series keep their halos. The roots differ in hue and in lightness
 * (CVD-safe); "another root" is a warm gray, so it never reads as a third root of the list.
 */
const BASIN_OKLCH: Record<'light' | 'dark', [number, number, number][]> = {
  //        root 1 (mint)          root 2 (cream)         another root (warm gray)
  light: [
    [0.925, 0.055, 150],
    [0.97, 0.07, 95],
    [0.895, 0.012, 60],
  ],
  dark: [
    [0.31, 0.045, 150],
    [0.35, 0.055, 85],
    [0.245, 0.008, 60],
  ],
};
const basinRgb = (label: number, mode: 'light' | 'dark'): RGB => {
  const [L, C, h] = BASIN_OKLCH[mode][label === OTHER_ROOT ? 2 : label % 2];
  return oklch(L, C, h);
};
const basinHex = (label: number, mode: 'light' | 'dark') => rgbToHex(basinRgb(label, mode));

function parseColor(c: string): [number, number, number] {
  const hex = /^#([0-9a-f]{6})/i.exec(c);
  if (hex) {
    const n = parseInt(hex[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const m = /rgba?\(([^)]+)\)/.exec(c);
  if (m) {
    const [r, g, b] = m[1].split(',').map((s) => parseFloat(s));
    return [r, g, b];
  }
  return [128, 128, 128];
}

/** The basin grid as an nx × ny image (top row = y_max), shaded lighter with more iterations. */
function basinImage(grid: BasinGrid, colors: ChartColors): HTMLCanvasElement {
  const { nx, ny, labels, iters } = grid;
  const canvas = document.createElement('canvas');
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(nx, ny);
  const surf = parseColor(colors.surface);
  const lut = [0, 1, OTHER_ROOT].map((l) => basinRgb(l, colors.mode));
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const s = j * nx + i;
      const d = ((ny - 1 - j) * nx + i) * 4;
      const l = labels[s];
      if (l === NO_ROOT) continue;
      const c = l === OTHER_ROOT ? lut[2] : lut[l % 2];
      // More iterations → closer to the surface (classic Newton-fractal shading), within the band.
      const w = 0.45 * Math.min(1, Math.log(1 + iters[s]) / Math.log(41));
      img.data[d] = c[0] * (1 - w) + surf[0] * w;
      img.data[d + 1] = c[1] * (1 - w) + surf[1] * w;
      img.data[d + 2] = c[2] * (1 - w) + surf[2] * w;
      img.data[d + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// ── Fade for step geometry (140 ms cross-fade when the explained step changes) ───────────────

function useFade(key: string, reduced: boolean): number {
  const [fade, setFade] = useState({ key, alpha: 1 });
  if (fade.key !== key) setFade({ key, alpha: reduced ? 1 : 0 });
  const fading = fade.alpha < 1;
  const fadeKey = fade.key;
  useEffect(() => {
    if (!fading) return;
    let raf = 0;
    let start: number | null = null;
    const tick = (now: number) => {
      start ??= now;
      const a = Math.min(1, (now - start) / 140);
      setFade((f) => (f.key === fadeKey ? { ...f, alpha: Math.max(f.alpha, a) } : f));
      if (a < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [fading, fadeKey]);
  return fade.key === key ? fade.alpha : reduced ? 1 : 0;
}

// ── Step overlays ─────────────────────────────────────────────────────────────────────────────

function stepOverlays(
  g: StepGeometry,
  slot: number,
  broyden: boolean,
  exactJ: Matrix | null,
  span: number,
): Overlay2D[] {
  const out: Overlay2D[] = [];
  const L = span * 4;
  // Broyden: the zero lines of the exact linearization (J(𝐱ₖ)) in gray, under the model's.
  if (broyden && exactJ) {
    for (let i = 0; i < 2; i++) {
      const ln = linearizationLine(g.from, g.F[i], exactJ[i], L);
      if (ln)
        out.push({
          kind: 'segment',
          from: ln.from,
          to: ln.to,
          width: 1,
          dashed: true,
          alpha: 0.5,
        });
    }
  }
  for (let i = 0; i < 2; i++) {
    const ln = linearizationLine(g.from, g.F[i], g.M[i], L);
    if (ln)
      out.push({
        kind: 'segment',
        from: ln.from,
        to: ln.to,
        slot,
        width: 1.35,
        dashed: i === 1,
        alpha: 0.9,
      });
  }
  // The full step 𝐩 to the crossing of ℓ₁ and ℓ₂.
  const k = g.j - 1;
  const damped = g.alpha !== 1;
  out.push({
    kind: 'arrow',
    from: g.from,
    to: g.target,
    slot,
    width: 1.6,
    dashed: damped,
    label: [mathBold(broyden ? 's' : 'p'), mathSub(String(k))],
  });
  out.push({ kind: 'point', at: g.target, slot, shape: 'ring', radius: 4.5 });
  // Damped Newton: the trial points 𝐱ₖ + α𝐩ₖ (rejected: crosses; accepted: the path's next iterate).
  if (g.trials.length > 1)
    for (const [a] of g.trials.slice(0, -1).slice(0, 8))
      out.push({
        kind: 'point',
        at: [g.from[0] + a * g.p[0], g.from[1] + a * g.p[1]],
        slot,
        shape: 'cross',
        radius: 3.5,
      });
  return out;
}

// ── Legend over the stage ───────────────────────────────────────────────────────────────────

function LineSample({ dashed, slot, faint }: { dashed?: boolean; slot?: number; faint?: boolean }) {
  return (
    <svg width="26" height="8" aria-hidden="true" className={styles.lineSample}>
      <line
        x1="1"
        y1="4"
        x2="25"
        y2="4"
        stroke={slot === undefined ? 'currentColor' : `var(--series-${slot + 1})`}
        strokeWidth={slot === undefined ? 1.6 : 1.4}
        strokeDasharray={dashed ? (faint ? '3 3' : '5 4') : undefined}
        opacity={faint ? 0.55 : 1}
      />
    </svg>
  );
}

/** ℓ₁ solid over ℓ₂ dashed, as the stage draws them. */
function LinePair({ slot, faint }: { slot?: number; faint?: boolean }) {
  const stroke = slot === undefined ? 'currentColor' : `var(--series-${slot + 1})`;
  return (
    <svg width="26" height="11" aria-hidden="true" className={styles.lineSample}>
      <line
        x1="1"
        y1="2.5"
        x2="25"
        y2="2.5"
        stroke={stroke}
        strokeWidth={faint ? 1 : 1.4}
        opacity={faint ? 0.55 : 1}
        strokeDasharray={faint ? '3 3' : undefined}
      />
      <line
        x1="1"
        y1="8.5"
        x2="25"
        y2="8.5"
        stroke={stroke}
        strokeWidth={faint ? 1 : 1.4}
        opacity={faint ? 0.55 : 1}
        strokeDasharray={faint ? '3 3' : '5 4'}
      />
    </svg>
  );
}

// ── The lab ─────────────────────────────────────────────────────────────────────────────────

export default function SystemsLab() {
  const lab = getLab('systems')!;
  const colors = useChartColors();
  const problems = useMemo(() => listSystems(), []);
  const methods = useMemo(() => listMethods('systems'), []);

  const [problem, setProblem] = useProblemState(problems, DEFAULT_PROBLEM);
  const [x0, setX0] = useStartPoint(problem.x0, 2);
  const [selection, setSelection] = useMethodSelection(methods, DEFAULT_SELECTION);
  const [field, setField] = useUrlState<FieldView>('f', 'norm', FIELD_CODEC);
  const [metric, setMetric] = useUrlState<Metric>('y', 'res', METRIC_CODEC);
  const [kAxis, setKAxis] = useUrlState<KAxis>('kx', 'auto', KAXIS_CODEC);
  const [tab, setTab] = useState<'method' | 'steps'>('method');
  // The method whose geometry (and basins) is drawn; URL-backed so a preset can choose it.
  const [focusId, setFocusId] = useUrlState('fm', '', codecs.string);

  const runOptions = useMemo(() => ({ x0: [x0[0], x0[1]] }), [x0]);
  const runs = useLabRuns(problem, selection, runOptions);
  const traces = useMemo(() => runs.map((r) => r.result.trace), [runs]);
  const player = useTracePlayer(traces);
  usePlayerKeyboard(player);

  const focus = runs.find((r) => r.sel.id === focusId) ?? runs[0];
  const focusIndex = focus ? runs.indexOf(focus) : 0;
  const focusTrace = useMemo(() => focus?.result.trace ?? [], [focus]);
  const focusK = focus ? player.localK(focusIndex) : 0;
  const gIndex = focus ? geometryIndex(player.localT(focusIndex), focusTrace.length) : 0;
  const geometry = useMemo(() => stepGeometry(focusTrace, gIndex), [focusTrace, gIndex]);
  const fade = useFade(`${focus?.sel.id}|${problem.id}|${gIndex}`, player.reducedMotion);

  const [[dx0, dx1], [dy0, dy1]] = problem.domain;
  const span = Math.max(dx1 - dx0, dy1 - dy0);
  const roots = problem.roots as [number, number][];

  // ── Fields ──
  const [F1, F2] = problem.components;
  const normF = useMemo(() => (x: number, y: number) => Math.hypot(F1(x, y), F2(x, y)), [F1, F2]);

  const basins = useBasins(
    field === 'basins',
    problem.id,
    focus ? focus.sel.id : null,
    focus ? focus.sel.params : {},
    problem.domain,
  );
  const basinCanvas = useMemo(
    () => (basins.grid ? basinImage(basins.grid, colors) : null),
    [basins.grid, colors],
  );

  // ── Geometry ──
  const zeroCurves: ZeroCurve[] = useMemo(
    () => [
      {
        g: F1,
        cacheKey: `${problem.id}|F1`,
        label: [mathVar('F'), mathSub('1'), mathMain(' = 0')],
      },
      {
        g: F2,
        cacheKey: `${problem.id}|F2`,
        dash: [6, 4],
        label: [mathVar('F'), mathSub('2'), mathMain(' = 0')],
      },
    ],
    [F1, F2, problem.id],
  );
  const isBroyden = focus?.sel.id === 'broyden';
  const exactJ = useMemo(
    () => (geometry && isBroyden ? problem.jac(geometry.from) : null),
    [geometry, isBroyden, problem],
  );
  const stepMarks = useMemo(
    () =>
      geometry && focus ? stepOverlays(geometry, focus.sel.slot, isBroyden, exactJ, span) : [],
    [geometry, focus, isBroyden, exactJ, span],
  );

  // Markers the zero-curve labels keep clear of: 𝐱₀, the roots and every run's current iterate.
  const curveAvoid: [number, number][] = [
    [x0[0], x0[1]],
    ...roots,
    ...runs.map((r, i) => {
      const tr = r.result.trace;
      const k = Math.min(player.localK(i), tr.length - 1);
      return (tr[k]?.x ?? [NaN, NaN]) as [number, number];
    }),
  ];
  const offRuns: OffViewRun[] = runs.map((r, i) => ({
    points: r.result.trace.map((st) => st.x as [number, number]),
    reached: player.localK(i),
    name: SHORT[r.sel.id] ?? r.method.spec.name,
    color: colors.series[r.sel.slot],
    muted: runs.length > 1 && r !== focus,
  }));
  /** Boxes the off-view chips avoid: the zoom bar (CSS px of the plot). */
  const stageObstacles = (view: View2D): Rect[] => [
    { x: view.width - 48, y: view.height - 130, w: 48, h: 130 },
  ];

  const drawOverlay = (ctx: CanvasRenderingContext2D, view: View2D) => {
    if (field === 'basins' && basinCanvas) {
      const [l, t] = view.toPx(dx0, dy1);
      const [r, b] = view.toPx(dx1, dy0);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(basinCanvas, l, t, r - l, b - t);
      ctx.restore();
      drawAxes(ctx, {
        x: view.x,
        y: view.y,
        frame: { left: 0, top: 0, right: view.width, bottom: view.height },
        colors: view.colors,
        dpr: view.dpr,
        grid: false,
        baseline: false,
        inset: true,
        xLabel: 'x',
        yLabel: 'y',
      });
    }
    drawZeroCurves(ctx, view, zeroCurves, curveAvoid);
    drawOffView(ctx, view, offRuns, stageObstacles(view));
    if (stepMarks.length === 0 || fade <= 0) return;
    const ov = {
      toPx: view.toPx,
      toData: (px: number, py: number): [number, number] => [view.x.invert(px), view.y.invert(py)],
      width: view.width,
      height: view.height,
      dpr: view.dpr,
      colors: view.colors,
    };
    if (fade >= 1) {
      drawOverlays2D(ctx, ov, stepMarks);
      return;
    }
    // Cross-fade: draw the marks (with their halos) off screen, then composite at `fade`.
    const off = document.createElement('canvas');
    off.width = Math.round(view.width * view.dpr);
    off.height = Math.round(view.height * view.dpr);
    const octx = off.getContext('2d');
    if (!octx) return;
    octx.scale(view.dpr, view.dpr);
    drawOverlays2D(octx, ov, stepMarks);
    ctx.save();
    ctx.globalAlpha = fade;
    ctx.drawImage(off, 0, 0, view.width, view.height);
    ctx.restore();
  };

  const paths: PathSpec[] = useMemo(() => {
    const specs = runs.map((r) => ({
      points: r.result.trace.map((s) => s.x as [number, number]),
      color: colors.series[r.sel.slot],
      label: r.method.spec.name,
      muted: runs.length > 1 && r !== focus,
      arrows: true,
      start: false,
      end: r.result.converged ? ('converged' as const) : ('stopped' as const),
    }));
    // The focused method is drawn last (on top).
    return specs
      .map((spec, i) => ({ spec, i }))
      .sort((a, b) => Number(runs[a.i] === focus) - Number(runs[b.i] === focus))
      .map((x) => x.spec);
  }, [runs, colors, focus]);

  // ── Convergence ──
  const series = useMemo(
    () =>
      runs.map((r) => {
        const last = r.result.trace[r.result.trace.length - 1];
        const reached =
          r.result.converged && last && Array.isArray(last.x) ? (last.x as number[]) : null;
        const ref = reached ?? (last ? nearestRoot(last.x as number[], roots) : null);
        const rate = r.result.converged
          ? r.sel.id === 'newton_system'
            ? 'quadratic'
            : 'superlinear'
          : undefined;
        return {
          label: r.method.spec.name,
          slot: r.sel.slot,
          count: r.result.nIter,
          end: r.result.converged ? ('converged' as const) : ('stopped' as const),
          rate,
          values: r.result.trace.map((s: Step) => {
            if (metric === 'res') return s.fun === null ? null : Math.max(s.fun, 1e-16);
            if (!ref) return null;
            const x = s.x as number[];
            return Math.max(Math.hypot(x[0] - ref[0], x[1] - ref[1]), 1e-16);
          }),
        };
      }),
    [runs, metric, roots],
  );
  // A log-k axis when one run is much longer than another (Newton's 7 steps against Broyden's
  // 100 would fill 7 % of a linear axis); the viewer can switch.
  const lengths = runs.map((r) => r.result.trace.length);
  const logK =
    kAxis === 'auto'
      ? suggestLogK(lengths) ||
        (lengths.filter((n) => n > 2).length >= 2 &&
          Math.max(...lengths) >= 5 * Math.min(...lengths.filter((n) => n > 2)))
      : kAxis === 'log';
  const yName =
    metric === 'res'
      ? [
          mathMain('‖'),
          mathVar('F'),
          mathMain('('),
          mathBold('x'),
          mathSub('k', 'italic'),
          mathMain(')‖'),
          mathSub('2'),
        ]
      : [
          mathMain('‖'),
          mathBold('x'),
          mathSub('k', 'italic'),
          mathMain(' − '),
          mathBold('x'),
          mathSup('⋆'),
          mathMain('‖'),
          mathSub('2'),
        ];

  // ── Table ──
  const columns: Column[] = useMemo(() => {
    const tex = (t: string) => <Formula tex={t} />;
    const base: Column[] = [
      { key: 'k', label: tex('k'), value: (s) => s.k, align: 'right', width: '26px' },
      {
        key: 'x',
        label: tex('\\mathbf{x}_k'),
        width: 'minmax(132px, 2fr)',
        value: (s) => vecFixed(s.x as number[], 4),
      },
      {
        key: 'res',
        label: tex('\\|F(\\mathbf{x}_k)\\|'),
        value: (s) => sigFixed(s.fun, 3),
        align: 'right',
        width: 'minmax(50px, 0.7fr)',
      },
      {
        key: 'step',
        label: tex('\\|\\mathbf{s}\\|'),
        value: (s) => sigFixed(s.stepSize, 3),
        align: 'right',
        width: 'minmax(60px, 0.7fr)',
      },
    ];
    if (focus?.sel.id === 'broyden') {
      const trace = focus.result.trace;
      base.push({
        key: 'model',
        // Row k shows the model that produced 𝐱ₖ: B_{k−1} against J(𝐱_{k−1}).
        label: (
          <span title="Model error ‖B_{k−1} − J(x_{k−1})‖_F / ‖J(x_{k−1})‖_F of the matrix that produced row k (B_{k−1}, measured against the exact Jacobian at x_{k−1})">
            model
          </span>
        ),
        value: (s) =>
          s.k === 0
            ? '—'
            : sigFixed(
                modelError(s.info.jacobian as Matrix, problem.jac(trace[s.k - 1].x as Vector)),
                2,
              ),
        align: 'right',
        width: 'minmax(46px, 0.6fr)',
      });
    } else {
      base.push({
        key: 'alpha',
        label: tex('\\alpha'),
        // Backtracking halves α, so it is exactly 2⁻ⁱ: shown as such (compact and exact).
        value: (s) => (s.k === 0 ? '—' : powerOfTwo(s.info.alpha as number)),
        align: 'right',
        width: 'minmax(36px, 0.5fr)',
      });
    }
    return base;
  }, [focus, problem]);

  // ── Text ──
  const rootNames = roots.map((r) => texPt(r, 4));
  const basinLegend = field === 'basins' && (
    <div className={styles.legend} role="note">
      <span className={styles.legendTitle}>
        Basins of {focus?.method.spec.name ?? '—'}
        <span className={styles.legendState}>
          {basins.done
            ? `${basins.grid?.nx} × ${basins.grid?.ny} starts`
            : basins.grid
              ? `refining · ${basins.grid.nx} × ${basins.grid.ny}`
              : 'computing…'}
        </span>
      </span>
      {rootNames.map((t, i) => (
        <span key={t} className={styles.legendItem}>
          <span className={styles.chipSwatch} style={{ background: basinHex(i, colors.mode) }} />
          <Formula tex={`\\to\\ ${t}`} />
        </span>
      ))}
      <span className={styles.legendItem}>
        <span
          className={styles.chipSwatch}
          style={{ background: basinHex(OTHER_ROOT, colors.mode) }}
        />
        another root
      </span>
      <span className={styles.legendItem}>
        <span className={`${styles.chipSwatch} ${styles.chipNone}`} />
        no convergence
      </span>
      <span className={styles.legendNote}>
        Fainter cells took more iterations. Each cell is one run from its center.
      </span>
    </div>
  );

  const curveLegend = (
    <div className={styles.legend} role="note">
      <span className={styles.legendItem}>
        <LineSample />
        <Formula tex="F_1 = 0" />
      </span>
      <span className={styles.legendItem}>
        <LineSample dashed />
        <Formula tex="F_2 = 0" />
      </span>
      {roots.length > 0 && (
        <span className={`${styles.legendItem} ${styles.rootKey}`}>
          <span className={styles.crossSample} aria-hidden="true">
            +
          </span>
          <Formula tex={`\\mathbf{x}^\\star = ${rootNames.join(',\\ ')}`} />
        </span>
      )}
      {focus && geometry && (
        <span className={`${styles.legendItem} ${styles.modelKey}`}>
          <LinePair slot={focus.sel.slot} />
          <Formula tex="\ell_1,\ \ell_2" />
          <span className={styles.legendText}>
            {isBroyden ? (
              <>
                zero lines of the model <Formula tex={`B_{${geometry.j - 1}}`} />
              </>
            ) : (
              <>
                zero lines of the linear models at{' '}
                <Formula tex={`\\mathbf{x}_{${geometry.j - 1}}`} />
              </>
            )}
          </span>
        </span>
      )}
      {focus && geometry && isBroyden && (
        <span className={`${styles.legendItem} ${styles.modelKey}`}>
          <LinePair faint />
          <span className={styles.legendText}>
            zero lines of <Formula tex={`J(\\mathbf{x}_{${geometry.j - 1}})`} />
          </span>
        </span>
      )}
    </div>
  );

  const statusList = runs
    .map((r) => {
      const last = r.result.trace[r.result.trace.length - 1];
      const where =
        last && Array.isArray(last.x)
          ? ` at (${(last.x as number[]).map((v) => sig(v, 4)).join(', ')})`
          : '';
      const which = last ? rootIndex(last.x as number[], roots) : -1;
      return `${r.method.spec.name}: ${r.result.converged ? `converged in ${r.result.nIter} iterations${where}${which < 0 && r.result.converged ? ', a root outside the listed ones' : ''}` : `stopped after ${r.result.nIter} iterations${where}`}`;
    })
    .join('; ');

  const emptyNote = focus
    ? focus.error
      ? `Could not run: ${focus.error}`
      : `No step was taken: ${focus.result.message}.`
    : 'Add a method to compare.';

  const stageNote: ReactNode = field === 'basins' ? basinLegend : curveLegend;

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
      controls={
        <>
          <RailSection title="Problem">
            <ProblemPicker problems={problems} value={problem.id} onChange={setProblem} />
          </RailSection>
          <RailSection
            title="Methods"
            actions={
              <span className={styles.count}>
                {runs.length} / {Math.min(MAX_SERIES, methods.length)}
              </span>
            }
          >
            <MethodSlots available={methods} value={selection} onChange={setSelection} />
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
      stageToolbar={
        <SegmentedControl
          label="Background"
          value={field}
          onChange={setField}
          options={[
            {
              value: 'norm',
              label: <Formula tex="\|F\|" />,
              ariaLabel: 'Residual norm field',
              tooltip: '‖F(x, y)‖₂ as a faint background field',
            },
            {
              value: 'basins',
              label: 'Basins',
              ariaLabel: 'Basins of attraction',
              tooltip: 'Color every start point by the root the focused method reaches',
            },
          ]}
        />
      }
      stage={
        <div className={styles.stageFill}>
          <div className={styles.plot}>
            <Contour2D
              f={normF}
              domain={problem.domain}
              cacheKey={`${problem.id}|normF`}
              fMin={0}
              paths={paths}
              t={player.t}
              ease={!player.reducedMotion}
              minima={roots}
              minimaLabels={false}
              start={[x0[0], x0[1]]}
              overlay={drawOverlay}
              onPick={(p) => setX0(p)}
              offViewLabels={false}
              ariaLabel={`Zero curves F₁ = 0 and F₂ = 0 of ${problem.name} with ${field === 'basins' ? 'the basins of attraction' : 'the residual norm field'}. ${statusList}.`}
            />
          </div>
          {/* The key sits under the plot, never over the domain. */}
          {stageNote}
        </div>
      }
      playback={<PlaybackBar player={player} slots={runs.map((r) => r.sel.slot)} />}
      insights={[
        {
          id: 'convergence',
          title: 'Convergence',
          height: 284,
          actions: (
            <div className={styles.chartActions}>
              <KAxisControl logX={logK} onChange={setKAxis} />
              <SegmentedControl
                label="Error measure"
                value={metric}
                onChange={setMetric}
                options={[
                  {
                    value: 'res',
                    label: <Formula tex="\|F\|" />,
                    ariaLabel: 'Residual norm',
                    tooltip: 'Residual ‖F(xₖ)‖₂',
                  },
                  {
                    value: 'err',
                    label: <Formula tex="\|\mathbf{x}_k - \mathbf{x}^\star\|" />,
                    ariaLabel: 'Distance to the root',
                    tooltip: 'Distance to the root reached (or the nearest listed root)',
                  },
                ]}
              />
            </div>
          ),
          content: (
            <div className={styles.chartPad}>
              <ConvergenceChart
                series={series}
                t={player.t}
                logX={logK}
                yLabel={metric === 'res' ? '‖F(xₖ)‖₂' : '‖xₖ − x⋆‖₂'}
                yName={yName}
                onSeek={(k) => {
                  player.pause();
                  player.seek(k);
                }}
              />
            </div>
          ),
        },
        {
          id: 'step',
          title: 'This step',
          height: 252,
          content: focus ? (
            <StepPanel
              methodId={focus.sel.id}
              name={focus.method.spec.name}
              slot={focus.sel.slot}
              geometry={geometry}
              trace={focusTrace}
              jac={problem.jac}
              emptyNote={emptyNote}
            />
          ) : null,
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
              idBase="systems-details"
              items={[
                { id: 'method', label: 'Method' },
                { id: 'steps', label: 'Iterations' },
              ]}
            />
          ),
          content: focus ? (
            <TabPanel idBase="systems-details" id={tab} focusable={false}>
              {tab === 'method' ? (
                <MethodCard
                  method={focus.method}
                  slot={focus.sel.slot}
                  // No live grid: its generic f(𝐱ₖ) row has no meaning for a system, and the
                  // "This step" panel above shows the step's numbers in the rule itself.
                  result={focus.result}
                  error={focus.error}
                  call={{
                    problem: problem.id,
                    options: { x0: [x0[0], x0[1]] },
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
