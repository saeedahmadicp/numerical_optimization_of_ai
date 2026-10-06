/**
 * Stochastic family: the TS port against Python reference data beyond the parity fixtures.
 *
 * - problems: data (X, y) and metadata equal problems.json; f, ∇f, ∇²f, mini-batch and per-sample
 *   gradients equal Python at seven points; ∇f(w⋆) ≈ 0; derivatives match finite differences.
 * - every parity fixture replayed in FULL (all steps, all info keys, counts, message), not only the
 *   first ten iterates;
 * - extra runs (tests/stochastic/fixtures, see gen_stochastic_fixture.py): every method on every
 *   problem, record_every > 1, b > 32, full batches, convergence, divergence, a non-finite start,
 *   every lr schedule; invalid inputs.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { getMethod, runMethod, defaults } from '../../src/core/registry';
import { getProblem, listProblems } from '../../src/problems/registry';
import { reviveNumbers } from '../../src/core/json';
import type { FiniteSumProblem } from '../../src/problems/stochastic';
import { pairwiseSum } from '../../src/problems/stochastic';
import { learningRate, pyG, SCHEDULES } from '../../src/methods/stochastic/methods';
import '../../src/methods/stochastic/methods';
import '../../src/problems/stochastic';
import { mismatch, type Tol } from '../shared-ports/compare';

type Json = Record<string, unknown>;
const read = (rel: string) =>
  reviveNumbers(
    JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')),
  ) as Json;

const REF = read('./fixtures/stochastic_python.json') as {
  problems: Json[];
  schedules: { schedule: string; U: number; T: number; values: number[] }[];
  runs: { method: string; problem: string; params: Json; result: Json }[];
  errors: { method: string; params: Json; error: string }[];
};
const PROBLEMS_JSON = (read('../../src/generated/problems.json') as unknown as Json[]).filter(
  (p) => p.kind === 'stochastic',
);
const FIXTURES = read('../../src/generated/fixtures/stochastic.json') as unknown as {
  method: string;
  problem: string;
  params: Json;
  result: Json;
}[];

/** Iterates and values: libm (`exp`, `log1p`) and BLAS summation order differ in the last bits. */
const ITER: Tol = { rtol: 1e-9, atol: 1e-12 };
const DATA: Tol = { rtol: 1e-14, atol: 1e-15 };

function expectClose(got: unknown, want: unknown, tol: Tol, what = '') {
  const m = mismatch(got, want, tol);
  expect(m, what).toBeNull();
}

const snake = (r: Json) => ({
  method: r.method,
  x: r.x,
  fun: r.fun,
  converged: r.converged,
  message: r.message,
  n_iter: r.nIter,
  n_fev: r.nFev,
  n_gev: r.nGev,
  n_hev: r.nHev,
  extra: r.extra,
  trace: (r.trace as Json[]).map((s) => ({
    k: s.k,
    x: s.x,
    fun: s.fun,
    grad_norm: s.gradNorm,
    step_size: s.stepSize,
    info: s.info,
  })),
});

function run(method: string, problem: string, params: Json) {
  const { spec, fn } = getMethod(method);
  return fn(getProblem(problem), { ...defaults(spec), ...(params as Record<string, never>) });
}

/** Full comparison of a TS result with a Python one; numbers within `tol`, the rest exact. */
function compareRun(got: Json, want: Json, tol: Tol) {
  const g = snake(got) as Json;
  expect(g.n_iter).toBe(want.n_iter);
  expect(g.converged).toBe(want.converged);
  expect(g.message).toBe(want.message);
  expect(g.n_fev).toBe(want.n_fev);
  expect(g.n_gev).toBe(want.n_gev);
  expect(g.extra).toEqual(want.extra);
  const gt = g.trace as Json[],
    wt = want.trace as Json[];
  expect(gt.length).toBe(wt.length);
  gt.forEach((s, i) => {
    const w = wt[i];
    expect(s.k).toBe(w.k);
    expect(Object.keys(s.info as Json).sort()).toEqual(Object.keys(w.info as Json).sort());
    expect((s.info as Json).batch).toEqual((w.info as Json).batch);
    expectClose(s, w, tol, `trace[${i}]`);
  });
  expectClose(g.x, want.x, tol, 'x');
  expectClose(g.fun, want.fun, tol, 'fun');
}

describe('stochastic problems', () => {
  it('registers the four Python ids', () => {
    expect(
      listProblems<FiniteSumProblem>('stochastic')
        .map((p) => p.id)
        .sort(),
    ).toEqual(['huber_regression_2d', 'ill_conditioned_ls', 'linreg_2d', 'logreg_2d']);
  });

  for (const want of PROBLEMS_JSON) {
    it(`${String(want.id)}: data and metadata equal problems.json`, () => {
      const p = getProblem<FiniteSumProblem>(String(want.id));
      expect(p.name).toBe(want.name);
      expect(p.latex).toBe(want.latex);
      expect(p.dim).toBe(want.dim);
      expect(p.nSamples).toBe(want.n_samples);
      expect(p.domain).toEqual(want.domain);
      expect(p.x0).toEqual(want.x0);
      expect(p.minima).toEqual(want.minima);
      expect(p.loss).toBe(want.loss);
      expect(p.l2).toBe(want.l2);
      expect(p.huberDelta).toBe(want.huber_delta);
      expect(p.description).toBe(want.description);
      expect(p.tags).toEqual(want.tags);
      expect(p.extra).toEqual(want.extra);
      expectClose(p.X, want.X, DATA, 'X');
      expectClose(p.y, want.y, DATA, 'y');
    });
  }

  for (const want of REF.problems) {
    const id = String(want.id);
    it(`${id}: f, ∇f, ∇²f, mini-batch and per-sample gradients equal Python`, () => {
      const p = getProblem<FiniteSumProblem>(id);
      const idx = want.idx as number[];
      for (const e of want.evals as Json[]) {
        const w = e.w as number[];
        expectClose(p.f(w), e.f, { rtol: 1e-13, atol: 1e-15 }, 'f');
        expectClose(p.grad(w), e.grad, { rtol: 1e-12, atol: 1e-13 }, 'grad');
        expectClose(p.hess(w), e.hess, { rtol: 1e-12, atol: 1e-13 }, 'hess');
        expectClose(p.gradBatch(w, idx), e.grad_batch, { rtol: 1e-12, atol: 1e-13 }, 'batch');
        expectClose(p.gradSamples(w, idx), e.grad_samples, { rtol: 1e-13, atol: 1e-14 }, 'rows');
      }
    });
    it(`${id}: ∇f(w⋆) ≈ 0, f(w⋆) = f⋆, derivatives match finite differences`, () => {
      const p = getProblem<FiniteSumProblem>(id);
      const ws = p.minima[0];
      const g = p.grad(ws);
      expect(Math.hypot(...g)).toBeLessThan(1e-9);
      expect(p.f(ws)).toBeCloseTo(p.extra.f_min, 13);
      const w = [p.x0[0] + 0.37, p.x0[1] - 0.21];
      const h = 1e-6;
      const gw = p.grad(w);
      for (let j = 0; j < 2; j++) {
        const e = [0, 0];
        e[j] = h;
        const fd = (p.f([w[0] + e[0], w[1] + e[1]]) - p.f([w[0] - e[0], w[1] - e[1]])) / (2 * h);
        expect(Math.abs(fd - gw[j])).toBeLessThan(1e-5 * (1 + Math.abs(gw[j])));
        const gp = p.grad([w[0] + e[0], w[1] + e[1]]);
        const gm = p.grad([w[0] - e[0], w[1] - e[1]]);
        const H = p.hess(w);
        for (let i = 0; i < 2; i++)
          expect(Math.abs((gp[i] - gm[i]) / (2 * h) - H[i][j])).toBeLessThan(
            1e-4 * (1 + Math.abs(H[i][j])),
          );
      }
      // grad is the full mini-batch gradient, independent of the index order.
      const all = Array.from({ length: p.nSamples }, (_, i) => p.nSamples - 1 - i);
      expect(p.gradBatch(w, all)).toEqual(gw);
    });
  }

  it('rejects empty or out-of-range mini-batches', () => {
    const p = getProblem<FiniteSumProblem>('linreg_2d');
    expect(() => p.gradBatch([0, 0], [])).toThrow(/at least one index/);
    expect(() => p.gradBatch([0, 0], [200])).toThrow(/\[0, 200\)/);
    expect(() => p.gradSamples([0, 0], [-1])).toThrow(/\[0, 200\)/);
  });

  it('pairwiseSum equals np.add.reduce bit for bit (blocks of 128, eight accumulators)', () => {
    // Reference values from NumPy 2.5 (np.add.reduce); a naive left-to-right sum differs.
    const a = [1e16, ...new Array<number>(15).fill(1)];
    expect(pairwiseSum(a)).toBe(1.0000000000000014e16);
    expect(a.reduce((s, v) => s + v, 0)).toBe(1e16);
    const b = Array.from({ length: 333 }, (_, i) => Math.sin(i) * 10.0 ** ((i % 7) - 3));
    expect(pairwiseSum(b)).toBeCloseTo(1077.0228838730065, 12);
    expect(pairwiseSum([1, 2, 3])).toBe(6);
    expect(pairwiseSum([])).toBe(0);
  });
});

describe('stochastic methods: learning-rate schedules and messages', () => {
  it('learningRate equals Python for every schedule', () => {
    for (const s of REF.schedules)
      s.values.forEach((v, t) =>
        expect(learningRate(s.schedule, 0.3, t, s.U, s.T)).toBeCloseTo(v, 15),
      );
    expect(() => learningRate('linear', 1, 0, 1, 1)).toThrow(/unknown lr_schedule/);
    expect([...SCHEDULES]).toEqual(['constant', 'step', 'inv_sqrt', 'cosine']);
  });

  it("pyG formats like Python's .3g", () => {
    const cases: [number, string][] = [
      [0.0541, '0.0541'],
      [1e-6, '1e-06'],
      [5.63e-5, '5.63e-05'],
      [7.7e15, '7.7e+15'],
      [4.73e10, '4.73e+10'],
      [11.3, '11.3'],
      [111.4, '111'],
      [1234, '1.23e+03'],
      [0.0001, '0.0001'],
      [0.5, '0.5'],
      [2, '2'],
      [0, '0'],
      [NaN, 'nan'],
      [Infinity, 'inf'],
      [-0.000123456, '-0.000123'],
    ];
    for (const [x, s] of cases) expect(pyG(x)).toBe(s);
  });
});

describe('stochastic methods: every parity fixture, full trace', () => {
  for (const c of FIXTURES) {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)}`, () => {
      compareRun(run(c.method, c.problem, c.params) as unknown as Json, c.result, ITER);
    });
  }
});

describe('stochastic methods: extra Python runs', () => {
  for (const c of REF.runs) {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)}`, () => {
      // Long runs accumulate libm/BLAS rounding; contraction keeps it small.
      compareRun(run(c.method, c.problem, c.params) as unknown as Json, c.result, {
        rtol: 1e-7,
        atol: 1e-10,
      });
    });
  }

  for (const e of REF.errors) {
    it(`${e.method} rejects ${JSON.stringify(e.params)}`, () => {
      const prefix = e.error.split(' got ')[0].replace(/\('constant'.*$/, '');
      // Python NaN is exported as null.
      const params = { ...e.params };
      if (Array.isArray(params.x0)) params.x0 = params.x0.map((v) => (v === null ? NaN : v));
      expect(() => run(e.method, 'linreg_2d', params)).toThrow(prefix);
    });
  }

  it('rejects a problem that is not a finite sum and unknown parameters', () => {
    expect(() => getMethod('sgd').fn({ id: 'x' }, { ...defaults(getMethod('sgd').spec) })).toThrow(
      /FiniteSumProblem/,
    );
    expect(() => runMethod('sgd', getProblem('linreg_2d'), { alpha: 1 })).toThrow(/unknown/);
  });

  it('is reproducible for a seed and changes with it', () => {
    const a = run('sgd', 'linreg_2d', { seed: 5, epochs: 2 });
    const b = run('sgd', 'linreg_2d', { seed: 5, epochs: 2 });
    const c = run('sgd', 'linreg_2d', { seed: 6, epochs: 2 });
    expect(a.trace).toEqual(b.trace);
    expect(a.trace[1].x).not.toEqual(c.trace[1].x);
  });

  it('full-batch SGD is gradient descent exactly', () => {
    const p = getProblem<FiniteSumProblem>('linreg_2d');
    const r = run('sgd', 'linreg_2d', { batch_size: 1000, epochs: 3, lr: 0.1 });
    let w = [...p.x0];
    for (const s of r.trace.slice(1)) {
      const g = p.grad(w);
      w = w.map((wi, i) => wi - 0.1 * g[i]);
      expect(s.x).toEqual(w);
      expect(s.info.batch).toBeNull(); // b = 200 > 32
    }
  });
});
