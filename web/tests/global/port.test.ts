/**
 * Tests for the global-family port (src/methods/unconstrained/global_.ts).
 *
 * 1. Parity with every Python fixture in src/generated/fixtures/global.json, under the harness
 *    rules (first 10 iterates within 1e-8, final f within 1e-6 relative) and, beyond them, the
 *    full trace, nIter, nFev, converged, the message and every Step.info value of the first 10
 *    steps. This file imports only this module, so it runs even while other ports are in flux.
 * 2. The local Nelder–Mead of basin hopping against the `nelder_mead` fixtures.
 * 3. Unit tests for input errors, bounds, determinism and the helpers.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, type FixtureCase } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Problem, Result, Vector } from '../../src/core/types';
import '../../src/problems/unconstrained';
import {
  CHI,
  C_CONSTRICTION,
  basinHopping,
  cmaConstants,
  cmaEs,
  differentialEvolution,
  eighSym,
  formatG,
  nelderMeadLocal,
  particleSwarm,
  simulatedAnnealing,
} from '../../src/methods/unconstrained/global_';

const FIX = fileURLToPath(new URL('../../src/generated/fixtures/', import.meta.url));

function load(family: string): FixtureCase[] {
  const file = `${FIX}${family}.json`;
  if (!existsSync(file)) return [];
  return (JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>[]).map(
    fixtureCaseFromJson,
  );
}

function flat(x: unknown): number[] {
  if (typeof x === 'number') return [x];
  if (typeof x === 'boolean') return [x ? 1 : 0];
  if (x === null) return [NaN];
  if (Array.isArray(x)) return x.flatMap(flat);
  if (typeof x === 'object') return Object.values(x as object).flatMap(flat);
  return [];
}

function maxErr(a: unknown, b: unknown): number {
  const xa = flat(a),
    xb = flat(b);
  if (xa.length !== xb.length) return Infinity;
  let e = 0;
  xa.forEach((v, i) => {
    const w = xb[i];
    if (Number.isNaN(v) && Number.isNaN(w)) return;
    if (!Number.isFinite(v) || !Number.isFinite(w)) {
      if (v !== w) e = Infinity;
      return;
    }
    e = Math.max(e, Math.abs(v - w) / (1e-8 + Math.abs(w)));
  });
  return e;
}

const run = (c: FixtureCase): Result => {
  const { spec, fn } = getMethod(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(c.problem), { ...defaults, ...(c.params as Record<string, never>) });
};

describe('parity with the Python fixtures (global)', () => {
  const cases = load('global');
  it('has fixtures', () => expect(cases.length).toBeGreaterThan(0));
  cases.forEach((c, idx) => {
    it(`${c.method} on ${c.problem} #${idx}`, () => {
      const got = run(c);
      const want = c.result;
      const n = Math.min(10, want.trace.length);
      for (let k = 0; k < n; k++) {
        expect(maxErr(got.trace[k].x, want.trace[k].x)).toBeLessThanOrEqual(1e-8);
        // Every info value of the first steps (null = Python None).
        const keys = Object.keys(want.trace[k].info).sort();
        expect(Object.keys(got.trace[k].info).sort()).toEqual(keys);
        for (const key of keys)
          expect(maxErr(got.trace[k].info[key], want.trace[k].info[key]), key).toBeLessThanOrEqual(
            1e-8,
          );
      }
      expect(Math.abs((got.fun as number) - (want.fun as number))).toBeLessThanOrEqual(
        1e-10 + 1e-6 * Math.abs(want.fun as number),
      );
      // Stronger than the harness: the whole run replays.
      expect(got.nIter).toBe(want.nIter);
      expect(got.nFev).toBe(want.nFev);
      expect(got.converged).toBe(want.converged);
      expect(got.trace.length).toBe(want.trace.length);
      expect(maxErr(got.x, want.x)).toBeLessThanOrEqual(1e-6);
      expect(got.message.split(';')[0].split(':')[0]).toBe(
        want.message.split(';')[0].split(':')[0],
      );
    });
  });
});

describe('local Nelder–Mead of basin hopping = nelder_mead (unconstrained fixtures)', () => {
  const cases = load('unconstrained').filter(
    (c) => c.method === 'nelder_mead' && !(c.params as Record<string, unknown>).adaptive,
  );
  cases.forEach((c, idx) => {
    it(`nelder_mead on ${c.problem} #${idx}`, () => {
      const p = getProblem(c.problem) as Problem<Vector>;
      const prm = { xtol: 1e-8, ftol: 1e-8, initial_step: 0.5, max_iter: 1000, ...c.params } as {
        xtol: number;
        ftol: number;
        initial_step: number;
        max_iter: number;
      };
      const x0 = (prm as { x0?: Vector }).x0 ?? (p.x0 as Vector);
      const got = nelderMeadLocal(
        (x) => p.f(x) as number,
        x0,
        prm.xtol,
        prm.ftol,
        prm.initial_step,
        prm.max_iter,
      );
      const want = c.result;
      expect(got.path.length).toBe(want.trace.length);
      expect(got.nfev).toBe(want.nFev);
      expect(got.converged).toBe(want.converged);
      expect(maxErr(got.x, want.x)).toBeLessThanOrEqual(1e-6);
      for (let k = 0; k < Math.min(10, got.path.length); k++)
        expect(maxErr(got.path[k], want.trace[k].x)).toBeLessThanOrEqual(1e-8);
    });
  });
});

const rastrigin = () => getProblem('rastrigin') as Problem<Vector>;

describe('global methods: contracts', () => {
  const methods = [
    simulatedAnnealing,
    particleSwarm,
    differentialEvolution,
    cmaEs,
    basinHopping,
  ] as const;

  it('one Step per iteration, Step.x = info.best, nIter = last k', () => {
    for (const fn of methods) {
      const r = fn(rastrigin(), { seed: 3, max_iter: 25 });
      expect(r.trace[r.trace.length - 1].k).toBe(r.nIter);
      for (const s of r.trace) {
        expect(s.x).toEqual(s.info.best);
        expect(s.fun).toBe(s.info.best_f);
      }
      // best f never increases
      for (let k = 1; k < r.trace.length; k++)
        expect(r.trace[k].fun as number).toBeLessThanOrEqual(r.trace[k - 1].fun as number);
    }
  });

  it('is deterministic per seed and changes with the seed', () => {
    for (const fn of methods) {
      const a = fn(rastrigin(), { seed: 7, max_iter: 15 });
      const b = fn(rastrigin(), { seed: 7, max_iter: 15 });
      const c = fn(rastrigin(), { seed: 8, max_iter: 15 });
      expect(a.trace.map((s) => s.x)).toEqual(b.trace.map((s) => s.x));
      expect(JSON.stringify(a.trace.map((s) => s.info))).not.toBe(
        JSON.stringify(c.trace.map((s) => s.info)),
      );
    }
  });

  it('keeps annealing, swarm and DE iterates inside the box', () => {
    const lo = -5.12,
      hi = 5.12;
    const inBox = (x: Vector) => x.every((v) => v >= lo && v <= hi);
    const sa = simulatedAnnealing(rastrigin(), { seed: 1, step: 0.5, max_iter: 300 });
    for (const s of sa.trace) expect(inBox(s.info.current as Vector)).toBe(true);
    const ps = particleSwarm(rastrigin(), { seed: 1, w: 1.1, max_iter: 60 });
    for (const s of ps.trace)
      for (const x of s.info.particles as Vector[]) expect(inBox(x)).toBe(true);
    const de = differentialEvolution(rastrigin(), { seed: 1, F: 1.9, max_iter: 40 });
    for (const s of de.trace) {
      for (const x of s.info.population as Vector[]) expect(inBox(x)).toBe(true);
      for (const x of s.info.trials as Vector[]) expect(inBox(x)).toBe(true);
    }
  });

  it('rejects invalid input like Python (ValueError)', () => {
    expect(() => simulatedAnnealing(rastrigin(), { x0: [9, 0] })).toThrow(/outside the search box/);
    expect(() => simulatedAnnealing(rastrigin(), { alpha: 1.0 })).toThrow(/alpha/);
    expect(() => simulatedAnnealing(rastrigin(), { cooling: 'fast' })).toThrow(/cooling/);
    expect(() => particleSwarm(rastrigin(), { n_particles: 1 })).toThrow(/n_particles/);
    expect(() => differentialEvolution(rastrigin(), { pop_size: 3 })).toThrow(/pop_size/);
    expect(() => differentialEvolution(rastrigin(), { CR: 1.5 })).toThrow(/CR/);
    expect(() => cmaEs(rastrigin(), { pop_size: -1 })).toThrow(/pop_size/);
    expect(() => basinHopping(rastrigin(), { patience: 0 })).toThrow(/patience/);
    expect(() => cmaEs(rastrigin(), { max_iter: 0 })).toThrow(/max_iter/);
    expect(() => cmaEs((x: Vector) => x[0] ** 2, { x0: [1] })).toThrow(/search box/);
    expect(() => cmaEs(rastrigin(), { x0: [1, 2, 3] })).toThrow(/3 entries/);
  });

  it('logarithmic cooling does not freeze within the budget (Hajek’s schedule is slow)', () => {
    const r = simulatedAnnealing(rastrigin(), { cooling: 'logarithmic', max_iter: 400 });
    expect(r.converged).toBe(false);
    expect(r.nIter).toBe(400);
    expect(r.message).toMatch(/^reached max_iter=400 \(best f = /);
    const T = r.trace.map((s) => s.info.temperature as number);
    expect(T[1]).toBe(10);
    expect(T[3]).toBeCloseTo((10 * Math.log(2)) / Math.log(4), 14);
  });

  it('counts evaluations exactly (rejected out-of-box proposals cost nothing)', () => {
    const r = simulatedAnnealing(rastrigin(), { seed: 2, step: 1.0, max_iter: 200 });
    const evaluated = r.trace.filter((s) => s.info.inside === true).length;
    expect(r.nFev).toBe(1 + evaluated);
    const p = particleSwarm(rastrigin(), { seed: 2, n_particles: 7, max_iter: 9 });
    expect(p.nFev).toBe(7 * (p.nIter + 1));
    const d = differentialEvolution(rastrigin(), { seed: 2, pop_size: 9, max_iter: 9 });
    expect(d.nFev).toBe(9 * (d.nIter + 1));
    const c = cmaEs(rastrigin(), { seed: 2, max_iter: 9 });
    expect(c.nFev).toBe(1 + cmaConstants(2).lam * c.nIter);
  });

  it('f = −∞ stops with converged = false', () => {
    const p: Problem<Vector> = {
      id: 'cliff',
      name: 'cliff',
      latex: '',
      dim: 2,
      domain: [
        [-1, 1],
        [-1, 1],
      ],
      x0: [0.5, 0.5],
      f: (x) => (x[0] < 0 ? -Infinity : x[0] ** 2 + x[1] ** 2),
    };
    for (const fn of [particleSwarm, differentialEvolution, cmaEs]) {
      const r = fn(p, { seed: 0, max_iter: 50 });
      expect(r.converged).toBe(false);
      expect(r.fun).toBe(-Infinity);
      expect(r.message).toMatch(/unbounded below/);
    }
  });
});

describe('helpers', () => {
  it('constriction constants (Clerc & Kennedy 2002)', () => {
    expect(CHI).toBeCloseTo(0.7298437881283576, 15);
    expect(C_CONSTRICTION).toBeCloseTo(1.496179765663133, 15);
  });

  it('CMA-ES strategy constants for n = 2 (Hansen 2016, Table 1)', () => {
    const c = cmaConstants(2);
    expect(c.lam).toBe(6);
    expect(c.mu).toBe(3);
    expect(c.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15);
    expect(c.weights[0]).toBeGreaterThan(c.weights[1]);
  });

  it('eighSym diagonalizes symmetric matrices (ascending eigenvalues)', () => {
    const A = [
      [4, 1, 0.5],
      [1, 3, -0.2],
      [0.5, -0.2, 1],
    ];
    const { values, vectors: B } = eighSym(A);
    expect(values[0]).toBeLessThanOrEqual(values[1]);
    expect(values[1]).toBeLessThanOrEqual(values[2]);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let l = 0; l < 3; l++) s += B[i][l] * values[l] * B[j][l];
        expect(s).toBeCloseTo(A[i][j], 12);
      }
  });

  it('formatG prints like Python’s .6g / .3g', () => {
    expect(formatG(0.00020027196472527815)).toBe('0.000200272');
    expect(formatG(4.998677355747926)).toBe('4.99868');
    expect(formatG(8.432934350821597e-10)).toBe('8.43293e-10');
    expect(formatG(5.7e-5, 3)).toBe('5.7e-05');
    expect(formatG(0.000985, 3)).toBe('0.000985');
    expect(formatG(1e14, 3)).toBe('1e+14');
    expect(formatG(100, 6)).toBe('100');
    expect(formatG(Infinity)).toBe('inf');
  });
});
