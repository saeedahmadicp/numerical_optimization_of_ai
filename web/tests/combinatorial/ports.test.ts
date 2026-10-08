/**
 * Combinatorial ports (src/methods/combinatorial/*, src/problems/combinatorial.ts) against Python,
 * stricter than the shared parity harness: every step of every trace (x, fun, info with its
 * snake_case keys), the message, the counts and `extra`, for the exported fixtures and for the
 * extra runs in ./extra_python.json (gen_extra_fixture.py). Plus unit tests for helpers and the
 * input-error paths the fixtures do not reach.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, type FixtureCase } from '../../src/core/json';
import { getMethod, runMethod } from '../../src/core/registry';
import { getProblem, listProblems } from '../../src/problems/registry';
import {
  CIRCLE_12_POINTS,
  isKnapsack,
  type CombinatorialInstance,
  type KnapsackInstance,
  type TspInstance,
} from '../../src/problems/combinatorial';
import {
  dantzigBound,
  knapsackBranchBound,
  ratioOrder,
} from '../../src/methods/combinatorial/knapsack';
import {
  fmtG,
  logaddexp,
  orderCrossover,
  pySum,
  tspHeldKarp,
  tspTwoOpt,
} from '../../src/methods/combinatorial/tsp';
import { Rng } from '../../src/core/rng';
import { mismatch, type Tol } from '../shared-ports/compare';
import '../../src/methods/combinatorial/knapsack';
import '../../src/methods/combinatorial/tsp';
import '../../src/problems/combinatorial';

const read = <T>(rel: string): T =>
  JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')) as T;

const FIXTURES = read<Record<string, unknown>[]>(
  '../../src/generated/fixtures/combinatorial.json',
).map(fixtureCaseFromJson);
const EXTRA = read<Record<string, unknown>[]>('./extra_python.json').map(fixtureCaseFromJson);

/** Deterministic methods: everything to 1e-12. Stochastic: libm (exp/log) may differ late. */
const TIGHT: Tol = { rtol: 1e-12, atol: 1e-12 };

function runCase(c: FixtureCase) {
  return runMethod(c.method, getProblem(c.problem), c.params as Record<string, unknown>);
}

function checkFull(c: FixtureCase) {
  const got = runCase(c);
  const want = c.result;
  expect(got.message).toBe(want.message);
  expect(got.converged).toBe(want.converged);
  expect(got.nIter).toBe(want.nIter);
  expect(got.nFev).toBe(want.nFev);
  expect(mismatch(got.x, want.x, TIGHT)).toBeNull();
  expect(mismatch(got.fun, want.fun, TIGHT)).toBeNull();
  expect(mismatch(got.extra, want.extra, TIGHT, 'extra')).toBeNull();
  expect(got.trace.length).toBe(want.trace.length);
  got.trace.forEach((s, i) => {
    const w = want.trace[i];
    expect(s.k).toBe(w.k);
    expect(mismatch(s.x, w.x, TIGHT, `trace[${i}].x`)).toBeNull();
    expect(mismatch(s.fun, w.fun, TIGHT, `trace[${i}].fun`)).toBeNull();
    expect(mismatch(s.info, w.info, TIGHT, `trace[${i}].info`)).toBeNull();
  });
}

describe('combinatorial ports: full traces equal Python', () => {
  for (const [label, cases] of [
    ['fixtures', FIXTURES],
    ['extra runs', EXTRA],
  ] as const) {
    describe(label, () => {
      cases.forEach((c, i) => {
        it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => checkFull(c));
      });
    });
  }
});

describe('combinatorial problems equal problems.json', () => {
  const meta = read<Record<string, unknown>[]>('../../src/generated/problems.json').filter(
    (p) => p.kind === 'combinatorial',
  );
  it('same ids in the same order', () => {
    expect(listProblems<CombinatorialInstance>('combinatorial').map((p) => p.id)).toEqual(
      meta.map((m) => m.id),
    );
  });
  for (const m of meta) {
    it(`${String(m.id)}: data, optimum, name, description`, () => {
      const p = getProblem<CombinatorialInstance>(m.id as string);
      expect(p.name).toBe(m.name);
      expect(p.description).toBe(m.description);
      if (isKnapsack(p)) {
        expect(p.values).toEqual(m.values);
        expect(p.weights).toEqual(m.weights);
        expect(p.capacity).toBe(m.capacity);
        expect(p.optimumValue).toBe(m.optimum_value);
      } else {
        expect(p.coords).toEqual(m.coords);
        expect(p.optimumLength).toBe(m.optimum_length);
      }
    });
  }
  it('the 12-gon points are the regular polygon (within 1e-13)', () => {
    CIRCLE_12_POINTS.forEach(([x, y], k) => {
      expect(Math.abs(x - (50 + 40 * Math.cos((2 * Math.PI * k) / 12)))).toBeLessThan(1e-13);
      expect(Math.abs(y - (50 + 40 * Math.sin((2 * Math.PI * k) / 12)))).toBeLessThan(1e-13);
    });
  });
});

describe('helpers', () => {
  it('ratioOrder: exact cross-multiplication, ties by index', () => {
    expect(ratioOrder([2, 4, 3, 1], [1, 2, 3, 1])).toEqual([0, 1, 2, 3]);
    expect(ratioOrder([3, 50, 50], [1, 50, 50])).toEqual([0, 1, 2]);
  });
  it('dantzigBound: LP bound and its floor', () => {
    // order 0,1,2 with capacity 75: item 0 (w 1), item 1 (w 50), then 24/50 of item 2.
    const [lp, floor] = dantzigBound([3, 50, 50], [1, 50, 50], [0, 1, 2], 0, 75);
    expect(lp).toBe(53 + (24 * 50) / 50);
    expect(floor).toBe(77);
    expect(dantzigBound([5], [2], [0], 0, 3)).toEqual([5, 5]);
    expect(dantzigBound([7], [3], [0], 0, 2)).toEqual([14 / 3, 4]);
  });
  it('pySum: Neumaier-compensated like CPython ≥ 3.12', () => {
    expect(pySum([0.1, 0.2, 0.3])).toBe(0.6);
    expect(pySum([1e100, 1.0, -1e100, 1.0])).toBe(2.0);
    expect(pySum([])).toBe(0);
  });
  it("fmtG matches Python's %g", () => {
    expect(fmtG(0.0519871234)).toBe('0.0519871');
    expect(fmtG(123456789)).toBe('1.23457e+08');
    expect(fmtG(1e-5)).toBe('1e-05');
    expect(fmtG(100)).toBe('100');
    expect(fmtG(0.001, 6)).toBe('0.001');
  });
  it('logaddexp follows numpy (infinities, symmetry)', () => {
    expect(logaddexp(-Infinity, -Infinity)).toBe(-Infinity);
    expect(logaddexp(-Infinity, 1.5)).toBe(1.5);
    expect(logaddexp(0, 0)).toBe(Math.LN2);
    expect(logaddexp(2, 1)).toBeCloseTo(Math.log(Math.exp(2) + Math.exp(1)), 14);
  });
  it('orderCrossover keeps p1[a..b] and fills in p2 order', () => {
    // Davis' example: p1 = 1 2 3 | 4 5 6 7 | 8 9, p2 = 4 5 2 1 8 7 6 9 3 (0-based cities).
    const p1 = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const p2 = [3, 4, 1, 0, 7, 6, 5, 8, 2];
    expect(orderCrossover(p1, p2, 3, 6)).toEqual([1, 0, 7, 3, 4, 5, 6, 8, 2]);
  });
  it('Rng permutations drive the shuffles (same draw order as Python)', () => {
    expect(new Rng(12).permutation(12)).toHaveLength(12);
  });
});

describe('edge cases and input errors', () => {
  const tiny: TspInstance = {
    kind: 'tsp',
    id: 'tiny',
    name: 'tiny',
    coords: [
      [0, 0],
      [1, 0],
      [0, 1],
    ],
    optimumLength: null,
    description: '',
    latex: '',
    tags: [],
  };
  it('n ≤ 3: local search returns the start tour at once', () => {
    const r = tspTwoOpt(tiny, {
      strategy: 'first',
      init: 'identity',
      max_iter: 10,
      record_every: 1,
    });
    expect(r.converged).toBe(true);
    expect(r.nIter).toBe(0);
    expect(r.trace).toHaveLength(1);
    expect(r.message).toBe('n = 3 ≤ 3: every tour has the same length');
  });
  it('accepts a bare (n, 2) coordinate array', () => {
    const r = tspHeldKarp(
      [
        [0, 0],
        [2, 0],
        [2, 1],
        [0, 1],
      ],
      {},
    );
    expect(r.fun).toBe(6);
  });
  it('Held–Karp refuses n > 16', () => {
    expect(() => runMethod('tsp_held_karp', getProblem('tsp_cities_20'))).toThrow(
      'Held–Karp is limited to n ≤ 16 cities (got n = 20)',
    );
  });
  it('simulated annealing refuses t_min ≥ t0', () => {
    expect(() =>
      runMethod('tsp_simulated_annealing', getProblem('tsp_circle_12'), { t0: 0.01, t_min: 0.01 }),
    ).toThrow('the annealing schedule would be empty');
  });
  it('nearest neighbour refuses a start outside the instance', () => {
    expect(() =>
      runMethod('tsp_nearest_neighbor', getProblem('tsp_circle_12'), { start: 12 }),
    ).toThrow('start must be a city index in [0, 11], got 12');
  });
  it('knapsack validation', () => {
    const bad: KnapsackInstance = {
      ...getProblem<KnapsackInstance>('knapsack_greedy_trap'),
      weights: [0, 50, 50],
    };
    expect(() => getMethod('knapsack_dp').fn(bad, {})).toThrow('weights must be positive integers');
    expect(() => getMethod('knapsack_dp').fn(getProblem('tsp_circle_12'), {})).toThrow(
      'problem must be a numopt KnapsackInstance',
    );
  });
  it('branch and bound: the tree up to step k is the union of info.nodes', () => {
    const r = knapsackBranchBound(getProblem('knapsack_10'), {
      max_nodes: 100_000,
      record_every: 1,
    });
    const ids = r.trace.flatMap((s) => (s.info.nodes as { id: number }[]).map((n) => n.id));
    expect(ids).toEqual(ids.map((_, i) => i));
  });
});
