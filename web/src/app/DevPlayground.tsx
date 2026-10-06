/** `#/dev` — a living catalog of the design system and viz primitives for lab authors. */
import { useState } from 'react';
import {
  Badge,
  Button,
  Formula,
  IconButton,
  Kbd,
  MethodChip,
  NumberField,
  Panel,
  ParamControls,
  SegmentedControl,
  Select,
  Slider,
  Tabs,
  Toggle,
  useToast,
} from '../ui/components';
import { CONTOUR_MAPS, SEQUENTIAL, SERIES } from '../ui/colors';
import { param } from '../core/registry';
import { sci } from '../core/format';
import type { Params, Result } from '../core/types';
import { RunSummary, SeedControl, StartPointHint } from '../labs/_shell';
import '../labs/unconstrained/setup';
import { AppHeader } from './AppHeader';
import {
  ConvergenceDemo,
  Contour2DDemo,
  DataDemo,
  MatrixDemo,
  Plot1DDemo,
  ShellDemo,
  SurfaceDemo,
  TableDemo,
  TreeDemo,
} from './DevPrimitives';
import styles from './DevPlayground.module.css';

const SPECS = [
  param.float('alpha', 1e-3, { min: 1e-6, max: 1, log: true, help: 'Step length.' }),
  param.float('beta', 0.9, { min: 0, max: 0.999, help: 'Momentum.' }),
  param.int('max_iter', 200, { min: 1, max: 5000 }),
  param.bool('nesterov', false),
  param.choice('line_search', 'wolfe', ['wolfe', 'backtracking']),
  param.choice('update', 'bfgs', ['bfgs', 'dfp', 'sr1', 'broyden']),
];

/** A stand-in result for the run-summary demo. */
const demoResult = (nIter: number, converged: boolean, message = ''): Result => ({
  method: 'demo',
  x: 0,
  fun: 0,
  converged,
  message,
  nIter,
  nFev: 0,
  nGev: 0,
  nHev: 0,
  trace: [],
  extra: {},
});
const DEMO_RUNS = [
  {
    name: 'Gradient descent (Armijo backtracking)',
    slot: 0,
    result: demoResult(5000, false, 'reached max_iter=5000'),
  },
  { name: 'BFGS', slot: 1, result: demoResult(38, true) },
  { name: 'Conjugate gradient (Polak–Ribière+)', slot: 2, result: demoResult(110, true) },
  { name: 'Nelder–Mead', slot: 3, result: demoResult(12, false, 'diverged: non-finite value') },
];

export default function DevPlayground() {
  const toast = useToast();
  const [seg, setSeg] = useState<'a' | 'b' | 'c'>('a');
  const [tab, setTab] = useState<'one' | 'two' | 'three'>('one');
  const [sel, setSel] = useState<string>('rosenbrock');
  const [lin, setLin] = useState(0.4);
  const [lg, setLg] = useState(1e-4);
  const [num, setNum] = useState(1.5);
  const [on, setOn] = useState(true);
  const [params, setParams] = useState<Params>({});
  const [seed, setSeed] = useState(0);

  return (
    <div className={styles.page}>
      <AppHeader crumbs={[{ label: 'Design system' }]} />
      <header className={styles.head}>
        <h1 className={styles.title}>Design system</h1>
        <p className={styles.lede}>
          Every component and visualization primitive a lab is built from, live. See web/README.md,
          “How to build a lab”, for the APIs.
        </p>
      </header>
      <main className={styles.wrap}>
        <Panel title="Buttons">
          <div className={styles.stack}>
            <div className={styles.row}>
              <Button variant="primary">Primary</Button>
              <Button>Secondary</Button>
              <Button variant="ghost">Ghost</Button>
              <Button size="sm" icon="plus">
                Small
              </Button>
            </div>
            <div className={styles.row}>
              <IconButton icon="play" label="Play" shortcut="Space" variant="secondary" />
              <IconButton icon="reset" label="Reset" shortcut="R" />
              <Button icon="link" onClick={() => toast('Link copied', 'check')}>
                Toast
              </Button>
              <Kbd>Space</Kbd> <Kbd>←</Kbd> <Kbd>→</Kbd>
            </div>
            <div className={styles.row}>
              <Badge>neutral</Badge>
              <Badge tone="good" dot>
                converged
              </Badge>
              <Badge tone="warn" dot>
                max_iter
              </Badge>
              <Badge tone="bad" dot>
                diverged
              </Badge>
              <Badge tone="accent">new</Badge>
            </div>
            <div className={styles.row}>
              {[0, 1, 2, 3].map((s) => (
                <MethodChip
                  key={s}
                  name={['Gradient descent', 'Momentum', 'Newton', 'BFGS'][s]}
                  slot={s}
                  onRemove={() => {}}
                />
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="Inputs">
          <div className={styles.stack}>
            <SegmentedControl
              label="Demo"
              value={seg}
              onChange={setSeg}
              options={[
                { value: 'a', label: 'Linear' },
                { value: 'b', label: 'Log' },
                { value: 'c', label: 'Auto' },
              ]}
            />
            <Tabs
              label="Demo tabs"
              value={tab}
              onChange={setTab}
              items={[
                { id: 'one', label: 'Method' },
                { id: 'two', label: 'Iterations' },
                { id: 'three', label: 'Notes' },
              ]}
            />
            <Select
              label="Problem"
              value={sel}
              onChange={setSel}
              options={[
                'rosenbrock',
                'himmelblau',
                'beale',
                'booth',
                'matyas',
                'three_hump_camel',
                'goldstein_price',
                'mccormick',
              ].map((v) => ({
                value: v,
                label: v.replace(/_/g, ' '),
                description: 'test problem',
              }))}
            />
            <div className={styles.row}>
              <Slider label="Linear" value={lin} min={0} max={1} onChange={setLin} />
              <NumberField label="Linear value" value={lin} onChange={setLin} width={84} />
            </div>
            <div className={styles.row}>
              <Slider
                label="Log"
                value={lg}
                min={1e-8}
                max={1}
                log
                onChange={setLg}
                valueText={sci(lg)}
              />
              <NumberField label="Log value" value={lg} onChange={setLg} log width={84} />
            </div>
            <div className={styles.row}>
              <NumberField label="x" prefix="x₀" value={num} onChange={setNum} width={110} />
              <Toggle label="Enabled" showLabel checked={on} onChange={setOn} />
            </div>
          </div>
        </Panel>

        <Panel title="ParamControls (from ParamSpec[])">
          <ParamControls
            specs={SPECS}
            values={params}
            onChange={(k, v) => setParams((p) => ({ ...p, [k]: v }))}
          />
        </Panel>

        <Panel title="Lab shell: run summary, start-point hint, seed">
          <div className={styles.stack}>
            {[2, 3].map((n) => (
              <div key={n} className={styles.summaryDemo}>
                <RunSummary runs={DEMO_RUNS.slice(0, n)} />
              </div>
            ))}
            <StartPointHint variable="𝐱₀" drag />
            <div style={{ width: 272 }}>
              <SeedControl value={seed} onChange={setSeed} section={false} />
            </div>
          </div>
        </Panel>

        <Panel title="Color">
          <div className={styles.stack}>
            <div className={styles.swatches}>
              {SERIES.light.map((c, i) => (
                <div
                  key={c}
                  className={styles.swatch}
                  style={{ background: `var(--series-${i + 1})` }}
                >
                  {i + 1}
                </div>
              ))}
            </div>
            <div
              className={styles.ramp}
              style={{
                background: `linear-gradient(90deg, ${CONTOUR_MAPS.light.hex.filter((_, i) => i % 16 === 0).join(',')})`,
              }}
            />
            <div
              className={styles.ramp}
              style={{
                background: `linear-gradient(90deg, ${CONTOUR_MAPS.dark.hex.filter((_, i) => i % 16 === 0).join(',')})`,
              }}
            />
            <div
              className={styles.ramp}
              style={{
                background: `linear-gradient(90deg, ${SEQUENTIAL.hex.filter((_, i) => i % 16 === 0).join(',')})`,
              }}
            />
            <Formula display tex="x_{k+1} = x_k - \alpha_k\, B_k^{-1} \nabla f(x_k)" />
          </div>
        </Panel>

        <Panel title="Type — display, text, numbers, math">
          <div className={styles.stack}>
            <p
              className="serif"
              style={{ fontSize: 'var(--text-3xl)', fontWeight: 500, lineHeight: 1.1 }}
            >
              Unconstrained
            </p>
            <p
              className="serif"
              style={{
                fontSize: 'var(--text-xl)',
                fontStyle: 'italic',
                color: 'var(--color-text-2)',
              }}
            >
              superlinear
            </p>
            <p>
              Step size <Formula tex="\alpha" /> = <span className="mono num">0.002</span> — Inter
              names, KaTeX symbols, mono values.
            </p>
            <Formula
              display
              tex="\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k\, H_k \nabla f(\mathbf{x}_k)"
            />
          </div>
        </Panel>
        <ShellDemo />
        <Plot1DDemo />
        <Contour2DDemo />
        <ConvergenceDemo />
        <TableDemo />
        <TreeDemo />
        <MatrixDemo />
        <DataDemo />
        <SurfaceDemo />
      </main>
    </div>
  );
}
