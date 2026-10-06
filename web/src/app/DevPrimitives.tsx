/** `#/dev` sections for the viz primitives and the lab-shell APIs (one live example each). */
import { Suspense, useMemo, useState } from 'react';
import {
  defaults,
  getMethod,
  hasMethod,
  listMethods,
  type RegisteredMethod,
} from '../core/registry';
import { getProblem } from '../problems/registry';
import type { Problem2D, Step } from '../core/types';
import { sigFixed } from '../core/format';
import { CopyButton, Menu, Num, Panel, PlaybackBar, Formula } from '../ui/components';
import {
  Contour2D,
  ConvergenceChart,
  DataPlot,
  LazySurface3D,
  MatrixView,
  Plot1D,
  TableView,
  TreeView,
  mathVar,
  type DataPoint,
  type Overlay2D,
  type TreeNode,
} from '../viz';
import { useTracePlayer } from '../play/useTracePlayer';
import { useChartColors } from '../ui/theme';
import { MethodCard, TryThis, type LabPreset } from '../labs/_shell';
import styles from './DevPlayground.module.css';

// ── Plot1D ────────────────────────────────────────────────────────────────────────────

const f1 = (x: number) => Math.cos(3 * x) + 0.3 * x * x - 0.5;
const df1 = (x: number) => -3 * Math.sin(3 * x) + 0.6 * x;

export function Plot1DDemo() {
  const x0 = 1.6;
  return (
    <Panel title="Plot1D — overlays" className={styles.full} flush>
      <div className={styles.plot}>
        <Plot1D
          f={f1}
          domain={[-3, 3]}
          zeroLine
          xLabel="x"
          yName={[mathVar('f'), { t: '(', style: 'main' }, mathVar('x'), { t: ')', style: 'main' }]}
          ariaLabel="A curve with a bracket, a tangent, a model parabola, a secant, trapezoids and labelled points"
          overlays={[
            { kind: 'interval', a: -1.4, b: 0.2, slot: 0, label: '[a, b]' },
            { kind: 'area', from: 2.2, to: 2.9, slot: 3 },
            {
              kind: 'polyline',
              points: [-2.8, -2.4, -2, -1.6].map((x) => [x, f1(x)] as const),
              fill: 'under',
              dots: true,
              slot: 2,
            },
            { kind: 'tangent', x: x0, y: f1(x0), slope: df1(x0), slot: 1, label: 'tangent' },
            {
              kind: 'parabola',
              a: 4,
              b: 0,
              c: -1.45,
              from: -0.6,
              to: 0.6,
              slot: 2,
              label: 'model',
            },
            { kind: 'curve', f: (x) => 0.25 * x, slot: 3, dashed: true, label: 'g(x)' },
            { kind: 'arrow', from: [x0, f1(x0)], to: [x0 - f1(x0) / df1(x0), 0], slot: 1 },
            { kind: 'vline', x: 0, dashed: true, label: 'x = 0' },
            {
              kind: 'point',
              x: x0,
              y: f1(x0),
              slot: 1,
              label: [mathVar('x'), { t: 'k', style: 'italic', script: 'sub' }],
            },
            { kind: 'point', x: -1.4, y: f1(-1.4), slot: 0, shape: 'ring' },
            {
              kind: 'point',
              x: 0.62,
              y: f1(0.62),
              shape: 'cross',
              label: 'x⋆',
              labelSide: 'above',
            },
          ]}
        />
      </div>
    </Panel>
  );
}

// ── Contour2D overlays ────────────────────────────────────────────────────────────────

const quad = (x: number, y: number) => 0.5 * (3 * x * x + 2 * x * y + 2 * y * y) - x - y;
const g1 = (x: number, y: number) => x * x + y * y - 2.2; // disk constraint
const g2 = (x: number, y: number) => y - 0.5 * x - 0.6; // half-plane

export function Contour2DDemo() {
  const colors = useChartColors();
  const overlays: Overlay2D[] = [
    { kind: 'constraints', g: [g1, g2], cacheKey: 'dev-constraints' },
    {
      kind: 'ellipse',
      center: [-1.2, -0.6],
      matrix: [
        [3, 1],
        [1, 2],
      ],
      radius: 0.6,
      slot: 1,
      fill: true,
    },
    { kind: 'disk', center: [0.9, -0.9], radius: 0.35, slot: 2, dashed: true },
    {
      kind: 'polygon',
      points: [
        [-1.6, 0.9],
        [-0.9, 1.3],
        [-1.25, 0.55],
      ],
      fill: true,
      slot: 3,
      label: 'simplex',
    },
    { kind: 'arrow', from: [0.4, 0.4], to: [0.05, -0.1], label: '−∇f', slot: 0 },
    {
      kind: 'implicit',
      g: (x, y) => x * y - 0.5,
      cacheKey: 'dev-hyperbola',
      dashed: true,
      label: 'xy = ½',
    },
    { kind: 'point', at: [1.3, 1.2], shape: 'ring', label: 'a labelled point' },
  ];
  // A path whose third step leaves the view (dashed + chevrons + a label at the exit).
  const path: [number, number][] = [
    [-1.8, 1.6],
    [-0.6, 0.9],
    [0.6, -4.2],
    [0.3, 0.4],
    [0.25, 0.37],
  ];
  return (
    <Panel title="Contour2D — overlays, feasible set, off-view steps" className={styles.full} flush>
      <div className={styles.plot} style={{ height: 380 }}>
        <Contour2D
          f={quad}
          domain={[
            [-2, 2],
            [-1.6, 2],
          ]}
          cacheKey="dev-quad"
          overlays={overlays}
          paths={[
            {
              points: path,
              color: colors.series[0],
              label: 'Newton',
              end: 'converged',
              start: false,
            },
          ]}
          t={99}
          start={path[0]}
          minima={[[0.2, 0.4]]}
          ariaLabel="Quadratic level sets with a hatched infeasible region, an ellipse, a disk, a simplex, an arrow, a hyperbola and a path that leaves the view"
        />
      </div>
    </Panel>
  );
}

// ── ConvergenceChart ──────────────────────────────────────────────────────────────────

export function ConvergenceDemo() {
  const linear = Array.from({ length: 400 }, (_, k) => 0.9 ** k);
  const quadratic = [1, 0.5, 0.125, 7.8e-3, 3.05e-5, 4.66e-10, 1.08e-19];
  const h = Array.from({ length: 12 }, (_, i) => 10 ** (-i * 0.5));
  const err = h.map((x) => Math.abs(x * x * 0.3) + 1e-16 / x);
  return (
    <>
      <Panel title="ConvergenceChart — log-log k, rate notes, milestones" flush>
        <div className={styles.plot} style={{ padding: 12 }}>
          <ConvergenceChart
            series={[
              { label: 'Linear (ρ = 0.9)', slot: 0, values: linear, end: 'converged', count: 399 },
              {
                label: 'Newton',
                slot: 3,
                values: quadratic,
                end: 'converged',
                rate: 'quadratic',
                count: 6,
              },
            ]}
            t={Infinity}
            yLabel="error"
            yName={[
              mathVar('f'),
              { t: '(', style: 'main' },
              { t: 'x', style: 'bold' },
              { t: 'k', style: 'italic', script: 'sub' },
              { t: ') − ', style: 'main' },
              mathVar('f'),
              { t: '⋆', style: 'main', script: 'sup' },
            ]}
            logX
          />
        </div>
      </Panel>
      <Panel title="ConvergenceChart — error vs h with slope guides" flush>
        <div className={styles.plot} style={{ padding: 12 }}>
          <ConvergenceChart
            series={[{ label: 'Central difference', slot: 1, values: err, x: h }]}
            t={Infinity}
            yLabel="error"
            xName={[mathVar('h')]}
            logX
            slopes={[{ slope: 2, at: [1e-3, 1e-9] }]}
            legend
          />
        </div>
      </Panel>
    </>
  );
}

// ── TableView ─────────────────────────────────────────────────────────────────────────

interface Row {
  n: number;
  trap: number;
  simp: number;
}
const ROWS: Row[] = [2, 4, 8, 16, 32, 64].map((n) => ({
  n,
  trap: 0.5 / (n * n),
  simp: 0.2 / n ** 4,
}));

export function TableDemo() {
  const [hi, setHi] = useState(2);
  return (
    <Panel title="TableView — KaTeX headers, tabular numbers" flush>
      <TableView
        rows={ROWS}
        highlight={hi}
        onSelect={setHi}
        ariaLabel="Quadrature errors"
        columns={[
          { key: 'n', tex: 'n', value: (r) => r.n },
          { key: 'h', tex: 'h = \\tfrac{b-a}{n}', value: (r) => sigFixed(1 / r.n, 4) },
          { key: 't', tex: '|E_T|', value: (r) => sigFixed(r.trap, 3) },
          { key: 's', tex: '|E_S|', value: (r) => sigFixed(r.simp, 3) },
          {
            key: 'note',
            header: 'Note',
            align: 'left',
            mono: false,
            value: (r) => (r.n === 2 ? 'coarsest' : ''),
          },
        ]}
      />
    </Panel>
  );
}

// ── TreeView (branch and bound) ───────────────────────────────────────────────────────

const TREE: TreeNode[] = [
  { id: '0', parent: null, label: 'z ≤ 21.5', detail: 'x = (2.5, 3)', status: 'branched', step: 0 },
  {
    id: '1',
    parent: '0',
    label: 'z ≤ 20.0',
    detail: 'x = (2, 3.3)',
    edge: 'x₁ ≤ 2',
    status: 'branched',
    step: 1,
  },
  { id: '2', parent: '0', label: 'infeasible', edge: 'x₁ ≥ 3', status: 'infeasible', step: 2 },
  {
    id: '3',
    parent: '1',
    label: 'z = 19',
    detail: 'x = (2, 3)',
    edge: 'x₂ ≤ 3',
    status: 'incumbent',
    step: 3,
  },
  { id: '4', parent: '1', label: 'z ≤ 18.4', edge: 'x₂ ≥ 4', status: 'pruned', step: 4 },
];

// Module constant: useTracePlayer restarts when the traces array changes identity.
const TREE_STEPS = [TREE.length];

export function TreeDemo() {
  const player = useTracePlayer(TREE_STEPS, { autoplay: false, loop: false });
  const k = player.k;
  return (
    <Panel title="TreeView — branch and bound" flush>
      <div style={{ height: 270 }}>
        <TreeView nodes={TREE} k={k} current={String(k)} ariaLabel="Branch-and-bound tree" />
      </div>
      <PlaybackBar player={player} />
    </Panel>
  );
}

// ── MatrixView with diffs ─────────────────────────────────────────────────────────────

function elimination() {
  let A = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3],
  ];
  const steps: { A: number[][]; pivot: { row: number; col: number } }[] = [
    { A, pivot: { row: 0, col: 0 } },
  ];
  for (let k = 0; k < 2; k++)
    for (let i = k + 1; i < 3; i++) {
      const m = A[i][k] / A[k][k];
      A = A.map((r, ri) => (ri === i ? r.map((v, j) => v - m * A[k][j]) : r.slice()));
      steps.push({ A, pivot: { row: k, col: k } });
    }
  return steps;
}
const ELIM = elimination();
const ELIM_STEPS = [ELIM.length];

export function MatrixDemo() {
  const player = useTracePlayer(ELIM_STEPS, { autoplay: false });
  const step = ELIM[player.k];
  const prev = player.k > 0 ? ELIM[player.k - 1].A : null;
  return (
    <Panel title="MatrixView — pivot, changed cells" flush>
      <div style={{ padding: 16, display: 'grid', placeItems: 'center', minHeight: 180 }}>
        <MatrixView
          matrix={step.A}
          previous={prev}
          pivot={step.pivot}
          rows={[step.pivot.row]}
          separatorBefore={3}
          colLabels={[
            <Formula key="x" tex="x" />,
            <Formula key="y" tex="y" />,
            <Formula key="z" tex="z" />,
            <Formula key="b" tex="b" />,
          ]}
          rowLabels={['r₁', 'r₂', 'r₃']}
          ariaLabel="Augmented matrix"
        />
      </div>
      <PlaybackBar player={player} />
    </Panel>
  );
}

// ── DataPlot ──────────────────────────────────────────────────────────────────────────

function lsq(points: readonly DataPoint[]) {
  const n = points.length;
  const sx = points.reduce((s, p) => s + p.x, 0),
    sy = points.reduce((s, p) => s + p.y, 0);
  const sxx = points.reduce((s, p) => s + p.x * p.x, 0),
    sxy = points.reduce((s, p) => s + p.x * p.y, 0);
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1);
  return { a: (sy - b * sx) / n, b };
}

export function DataDemo() {
  const [pts, setPts] = useState<DataPoint[]>([
    { x: 0.5, y: 1.1 },
    { x: 1.4, y: 1.9 },
    { x: 2.2, y: 2.4 },
    { x: 3.1, y: 3.6 },
    { x: 4, y: 3.9 },
    { x: 4.6, y: 1.2 },
  ]);
  const { a, b } = lsq(pts);
  return (
    <Panel title="DataPlot — drag the points (or Tab + arrows)" flush>
      <div className={styles.plot}>
        <DataPlot
          points={pts}
          onPointsChange={setPts}
          curves={[{ f: (x) => a + b * x, slot: 0, label: 'least squares' }]}
          residualsTo={0}
          highlight={[5]}
          xDomain={[0, 5]}
          yDomain={[0, 5]}
          ariaLabel="Least-squares line through draggable points"
        />
      </div>
      <p className={styles.caption}>
        Fit <Formula tex={`y = ${a.toFixed(2)} + ${b.toFixed(2)}\\,x`} /> — residuals dashed.
      </p>
    </Panel>
  );
}

// ── Surface3D (small) ─────────────────────────────────────────────────────────────────

export function SurfaceDemo() {
  const himmelblau = getProblem<Problem2D>('himmelblau');
  return (
    <Panel title="Surface3D — small, lazy three.js" flush>
      <div className={styles.plot} style={{ height: 220 }}>
        <Suspense fallback={null}>
          <LazySurface3D
            f={(x, y) => himmelblau.f([x, y])}
            domain={himmelblau.domain}
            resolution={64}
            ariaLabel="Himmelblau surface"
          />
        </Suspense>
      </div>
    </Panel>
  );
}

// ── Shell APIs: MethodCard, Try this, copy & menu, numbers ───────────────────────────

const PRESETS: LabPreset[] = [
  {
    id: 'valley',
    title: 'Momentum outruns gradient descent',
    note: 'Same step α; the heavy ball keeps its speed along the valley floor.',
    problem: 'rosenbrock',
    methods: [
      { id: 'gradient_descent', slot: 0, params: {} },
      { id: 'momentum', slot: 1, params: { beta: 0.95 } },
    ],
    start: [-1.2, 1],
  },
  {
    id: 'basin',
    title: 'Four basins of Himmelblau',
    note: 'The start point decides which minimizer you reach.',
    problem: 'himmelblau',
    start: [-0.5, 0.5],
  },
];

export function ShellDemo() {
  const method: RegisteredMethod | null = useMemo(() => {
    const id = hasMethod('bfgs') ? 'bfgs' : (listMethods('unconstrained')[0]?.spec.id ?? null);
    return id ? (getMethod(id) as unknown as RegisteredMethod) : null;
  }, []);
  const problem = getProblem<Problem2D>('rosenbrock');
  const result = useMemo(
    () => (method ? method.fn(problem, { ...defaults(method.spec), x0: [-1.2, 1] }) : null),
    [method, problem],
  );
  const step: Step | undefined = result?.trace[Math.min(12, result.trace.length - 1)];
  return (
    <>
      <Panel title="MethodCard" flush>
        <div style={{ height: 520 }}>
          {method && result && (
            <MethodCard
              method={method}
              slot={2}
              step={step}
              result={result}
              call={{ problem: 'rosenbrock', options: { x0: [-1.2, 1] }, params: {} }}
            />
          )}
        </div>
      </Panel>
      <Panel title="Try this (presets) · copy · menu · numbers">
        <div className={styles.stack}>
          <TryThis presets={PRESETS} />
          <div className={styles.row}>
            <CopyButton text={'numopt.run("bfgs", problems.get("rosenbrock"))'}>
              Copy a call
            </CopyButton>
            <Menu
              label="More actions"
              items={[
                {
                  kind: 'copy',
                  label: 'Copy LaTeX',
                  text: 'x_{k+1} = x_k - \\alpha_k \\nabla f(x_k)',
                },
                { label: 'Do something', onSelect: () => {} },
                { kind: 'link', label: 'Open the methods', href: '#/methods' },
              ]}
            />
          </div>
          <div className={styles.row} style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            <Num value={-0.000012345} /> · <Num value={1234567} mode="int" /> ·{' '}
            <Num value={[-0.2885, 0.0712]} /> · <Num value={null} /> · <Num value={Infinity} />
          </div>
        </div>
      </Panel>
    </>
  );
}
