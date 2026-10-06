import { describe, expect, it } from 'vitest';
import { LABS, buildLabs, getLab, labForFamily, labForProblemKind } from '../src/labs';
import type { LabMeta } from '../src/labs/types';
import { CATALOG } from '../src/app/catalog';
import { buildSearchItems, normalize, searchItems, type SearchLab } from '../src/app/searchIndex';
import { kOfTau, T_END, T_PRE } from '../src/app/heroClock';
import { labHref, labQuery, presetActive, type LabPreset } from '../src/labs/_shell/labState';
import { pyFloat, pythonCall } from '../src/labs/_shell/python';
import { describeResult, evidence } from '../src/labs/_shell/status';
import { param } from '../src/core/registry';
import type { Result } from '../src/core/types';
import { layoutTree } from '../src/viz/treeLayout';
import { ellipsePoints, implicitSegments } from '../src/viz/overlays2d';
import { clipSegment, milestoneIndices, sideOf } from '../src/viz/PathLayer';
import { dataDomain, slopeTriangle, suggestLogK } from '../src/viz/chartMath';
import { readMethods, readProblems, readResearch, countBy } from '../vite/catalog';

const IDS = [
  'roots',
  'systems',
  'scalar',
  'line-search',
  'unconstrained',
  'global',
  'least-squares',
  'stochastic',
  'constrained',
  'lp',
  'combinatorial',
  'linalg',
  'integration',
  'differentiation',
  'interpolation',
  'regression',
];

describe('lab registry (discovered)', () => {
  it('has a meta for every lab, ids matching their folders', () => {
    for (const id of IDS) expect(getLab(id), id).toBeDefined();
    expect(new Set(LABS.map((l) => l.id)).size).toBe(LABS.length);
  });
  it('counts methods from the generated registry', () => {
    const total = LABS.reduce((n, l) => n + l.methodCount, 0);
    expect(total).toBe(CATALOG.methods);
    expect(getLab('unconstrained')!.methodCount).toBe(CATALOG.byFamily.unconstrained);
  });
  it('a lab without index.tsx is planned; with one it is open unless it says preview', () => {
    const meta = (id: string, extra: Partial<LabMeta> = {}): LabMeta => ({
      id,
      title: id,
      group: 'Optimization',
      problem: 'x',
      pitch: '',
      families: ['roots'],
      ...extra,
    });
    const labs = buildLabs(
      {
        './a/meta.ts': meta('a'),
        './b/meta.ts': meta('b'),
        './c/meta.ts': meta('c', { status: 'preview' }),
      },
      {
        './b/index.tsx': async () => ({ default: () => null }),
        './c/index.tsx': async () => ({ default: () => null }),
      },
      { roots: 7 },
    );
    const by = Object.fromEntries(labs.map((l) => [l.id, l]));
    expect(by.a.status).toBe('planned');
    expect(by.a.component).toBeUndefined();
    expect(by.b.status).toBe('open');
    expect(by.c.status).toBe('preview');
    expect(by.b.methodCount).toBe(7);
  });
  it('maps families and problem kinds to labs', () => {
    expect(labForFamily('line_search')?.id).toBe('line-search');
    expect(labForProblemKind('unconstrained')?.id).toBe('unconstrained');
    expect(labForProblemKind('calculus')?.id).toBe('integration');
  });
});

describe('build-time catalog', () => {
  it('reads the generated registry, problems and research notes', () => {
    const m = readMethods();
    expect(m.length).toBe(CATALOG.methods);
    expect(readProblems().length).toBe(CATALOG.problems);
    expect(Object.keys(countBy(m, (x) => x.family)).length).toBe(CATALOG.families);
    expect(readResearch().every((r) => r.id && r.title)).toBe(true);
  });
  it('returns empty data for missing files', () => {
    expect(readMethods('/nonexistent/registry.json')).toEqual([]);
    expect(readProblems('/nonexistent/problems.json')).toEqual([]);
  });
});

describe('search', () => {
  const labs: SearchLab[] = LABS;
  const index = {
    methods: readMethods(),
    problems: readProblems(),
    research: [],
  };
  const items = buildSearchItems(labs, index);
  it('normalizes diacritics, dashes and underscores', () => {
    expect(normalize('Anderson–Björck')).toBe('anderson-bjorck');
    expect(normalize('pure_newton')).toBe('pure newton');
  });
  it('finds a method by name and opens its method page', () => {
    const r = searchItems(items, 'bfgs');
    const m = r.find((x) => x.kind === 'method' && x.id === 'bfgs')!;
    expect(m.href).toBe('#/method/bfgs');
    expect(r[0].kind).toBe('method');
  });
  it('finds Björck without the umlaut, and problems by name', () => {
    expect(searchItems(items, 'bjorck').some((x) => x.id === 'anderson_bjorck')).toBe(true);
    const p = searchItems(items, 'rosenbrock').find(
      (x) => x.kind === 'problem' && x.id === 'rosenbrock',
    );
    expect(p?.href).toBe('#/lab/unconstrained?p=rosenbrock');
  });
  it('requires every token and lists open labs first for an empty query', () => {
    expect(searchItems(items, 'bfgs zzzz')).toEqual([]);
    const empty = searchItems(items, '');
    expect(empty.every((x) => x.kind === 'lab')).toBe(true);
    expect(empty[0].status).toBe('open');
  });
});

describe('lab URL state and presets', () => {
  const preset: LabPreset = {
    id: 'p',
    title: 't',
    problem: 'rosenbrock',
    methods: [
      { id: 'gradient_descent', slot: 0, params: { lr: 0.002 } },
      { id: 'momentum', slot: 1, params: {} },
    ],
    start: [-1.2, 1],
    extra: { v: '3d' },
  };
  it('builds the query a lab reads', () => {
    const q = labQuery(preset);
    expect(q.get('p')).toBe('rosenbrock');
    expect(q.get('m')).toBe('gradient_descent~0~lr=0.002,momentum~1');
    expect(q.get('x0')).toBe('-1.2,1');
    expect(q.get('v')).toBe('3d');
    expect(labHref('unconstrained', { problem: 'beale' })).toBe('#/lab/unconstrained?p=beale');
  });
  it('detects the active preset from the hash', () => {
    const hash = `#/lab/unconstrained?${labQuery(preset).toString()}`;
    expect(presetActive(preset, hash)).toBe(true);
    expect(presetActive(preset, hash.replace('3d', '2d'))).toBe(false);
    expect(
      presetActive({ ...preset, extra: undefined }, hash.replace('x0=-1.2%2C1', 'x0=0%2C0')),
    ).toBe(false);
  });
});

describe('Python call and status text', () => {
  it('formats floats like Python repr', () => {
    expect(pyFloat(1)).toBe('1.0');
    expect(pyFloat(-1.2)).toBe('-1.2');
    expect(pyFloat(1e-8)).toBe('1e-08');
    expect(pyFloat(0.002)).toBe('0.002');
    expect(pyFloat(2.5e-5)).toBe('2.5e-05');
  });
  it('writes only non-default params', () => {
    const specs = [
      param.float('gtol', 1e-8, { min: 1e-12, max: 1 }),
      param.int('max_iter', 500, { min: 1, max: 1e4 }),
    ];
    expect(
      pythonCall('bfgs', {
        problem: 'rosenbrock',
        x0: [-1.2, 1],
        params: { gtol: 1e-10, max_iter: 500 },
        specs,
      }),
    ).toBe('numopt.run("bfgs", problems.get("rosenbrock"), x0=[-1.2, 1.0], gtol=1e-10)');
  });
  it('typesets evidence: numbers, tolerances, budgets', () => {
    expect(evidence('‖∇f‖∞ = 1.31e-11 ≤ gtol', { gtol: 1e-8 })).toBe('‖∇f‖∞ = 1.31×10⁻¹¹ ≤ 10⁻⁸');
    expect(evidence('reached max_iter=5000 (‖∇f(x)‖ = 0.00117 > gtol)')).toBe(
      'reached the 5,000-iteration budget (‖∇f(x)‖ = 0.00117 > gtol)',
    );
  });
  it('puts the status word first with an icon', () => {
    const base: Result = {
      method: 'm',
      x: null,
      fun: null,
      converged: true,
      message: '',
      nIter: 596,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace: [],
      extra: {},
    };
    expect(describeResult(base)).toMatchObject({ short: 'converged · 596 iterations', icon: '✓' });
    expect(
      describeResult({ ...base, converged: false, nIter: 1000, message: 'reached max_iter=1000' }),
    ).toMatchObject({
      short: 'budget · 1,000 iterations',
      icon: '◷',
    });
  });
});

describe('viz geometry', () => {
  it('lays out trees with parents centered over their children', () => {
    const { placed, width, depth } = layoutTree([
      { id: 'r', parent: null },
      { id: 'a', parent: 'r' },
      { id: 'b', parent: 'r' },
      { id: 'a1', parent: 'a' },
      { id: 'a2', parent: 'a' },
    ]);
    expect(width).toBe(3);
    expect(depth).toBe(3);
    expect(placed.get('a')!.col).toBe(0.5);
    expect(placed.get('r')!.col).toBe((0.5 + 2) / 2);
  });
  it('samples the ellipse (x−c)ᵀM(x−c) = r² and rejects indefinite M', () => {
    const M = [
      [3, 1],
      [1, 2],
    ];
    const pts = ellipsePoints([1, -1], M, 0.5)!;
    for (const [x, y] of pts) {
      const dx = x - 1,
        dy = y + 1;
      expect(dx * (3 * dx + dy) + dy * (dx + 2 * dy)).toBeCloseTo(0.25, 10);
    }
    expect(
      ellipsePoints(
        [0, 0],
        [
          [1, 0],
          [0, -1],
        ],
      ),
    ).toBeNull();
  });
  it('traces g(x) = 0 by marching squares', () => {
    const segs = implicitSegments((x, y) => x * x + y * y - 1, [-2, 2], [-2, 2], 81, 81);
    expect(segs.length).toBeGreaterThan(40);
    for (let k = 0; k < segs.length; k += 2)
      expect(Math.hypot(segs[k], segs[k + 1])).toBeCloseTo(1, 2);
  });
  it('clips steps to the view and names the side', () => {
    const r = { left: 0, top: 0, right: 100, bottom: 100 };
    expect(clipSegment([50, 50], [50, 250], r)).toEqual([0, 0.25]);
    expect(clipSegment([200, 0], [300, 0], r)).toBeNull();
    expect(sideOf([50, 300], r)).toBe('below the view');
    expect(milestoneIndices(15_232)).toEqual([10, 100, 1000, 10000]);
  });
  it('slope triangles rise `slope` decades per decade', () => {
    const [a, b, c] = slopeTriangle({ slope: 2, at: [1e-3, 1e-9] }, [1e-6, 1], [1e-16, 1]);
    expect(b[0] / a[0]).toBeCloseTo(10, 10);
    expect(c[1] / b[1]).toBeCloseTo(100, 10);
    expect(suggestLogK([6, 38, 15_231])).toBe(true);
    expect(suggestLogK([300, 600])).toBe(false);
    expect(dataDomain([0.5, 4.6])[0]).toBeLessThanOrEqual(0.5);
  });
  it('hero clock: k is linear for the first step, then linear in log k', () => {
    expect(kOfTau(0)).toBe(0);
    expect(kOfTau(T_PRE / 2)).toBeCloseTo(0.5, 12);
    expect(kOfTau(T_END)).toBeCloseTo(20_000, 6);
    const mid = kOfTau(T_PRE + (T_END - T_PRE) / 2);
    expect(mid).toBeCloseTo(Math.sqrt(20_000), 6);
  });
});
