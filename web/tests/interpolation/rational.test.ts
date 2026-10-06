/**
 * The rational interpolation port (src/methods/interpolation/rational.ts) against the Python
 * reference on 82 cases beyond the six parity fixtures: AAA and Floater–Hormann on all nine
 * datasets (both AAA scalings, d = 0, 1, 3, 5, 8), tol / max_terms limits, |x| and tanh on 40
 * points, unsorted data, one and two points, the input errors; and the pole helpers on fixed
 * weight vectors. Every trace step is compared: the chosen support point, the weights, the
 * sample error, σ_min, the poles (as a set), the certified real poles and the messages.
 *
 * Regenerate: .venv/bin/python web/tests/interpolation/gen_rational_oracle.py
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defaults, getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result } from '../../src/core/types';
import {
  barycentricPoles,
  complexEigvals,
  floaterHormannWeights,
  minRightSingular,
  realDenominatorRoots,
  type Complex,
} from '../../src/methods/interpolation/rational';
import { barycentricEval } from '../../src/methods/interpolation/methods';
import '../../src/methods/interpolation/rational';
import '../../src/problems/data';

type Num = number | 'inf' | '-inf' | null;
interface OracleCase {
  name: string;
  method: string;
  problem: { id?: string; x?: number[]; y?: number[] };
  params: Record<string, number | string>;
  error?: string;
  result?: {
    x: number[];
    fun: Num;
    converged: boolean;
    message: string;
    n_iter: number;
    trace: { k: number; x: number[]; fun: Num; info: Record<string, unknown> }[];
    extra: Record<string, unknown>;
  };
}
interface HelperSet {
  z: number[];
  w: number[];
  f: number[];
  poles: Complex[];
  residues: Complex[];
  brackets: [number, number][];
}

const FILE = fileURLToPath(new URL('./fixtures/rational_oracle.json', import.meta.url));
const ORACLE = JSON.parse(readFileSync(FILE, 'utf8')) as { cases: OracleCase[]; helpers: HelperSet[] };

const num = (v: Num): number => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v === null ? NaN : v);
const nums = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : []);

function run(c: OracleCase): Result {
  const m = getMethod(c.method);
  const problem = c.problem.id ? getProblem(c.problem.id) : ([c.problem.x!, c.problem.y!] as const);
  return m.fn(problem, { ...defaults(m.spec), ...(c.params as Record<string, never>) });
}

function closeArr(got: readonly number[], want: readonly number[], rtol: number, atol: number, what: string) {
  expect(got.length, what).toBe(want.length);
  got.forEach((g, i) => {
    const w = want[i];
    if (!Number.isFinite(w)) return expect(g, what).toBe(w);
    expect(Math.abs(g - w), `${what}[${i}]: ${g} vs ${w}`).toBeLessThanOrEqual(atol + rtol * Math.abs(w));
  });
}

/** Every Python pole has a TS pole within rtol·max(1, |p|) (poles are a set; order differs). */
function samePoles(got: readonly (readonly number[])[], want: readonly (readonly number[])[], rtol: number, what: string) {
  expect(got.length, `${what}: pole count`).toBe(want.length);
  const free = got.map(() => true);
  for (const p of want) {
    let best = -1,
      bd = Infinity;
    got.forEach((q, i) => {
      const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (free[i] && d < bd) {
        bd = d;
        best = i;
      }
    });
    expect(bd, `${what}: pole ${p[0]} + ${p[1]}i (nearest TS pole at ${bd})`).toBeLessThanOrEqual(rtol * Math.max(1, Math.hypot(p[0], p[1])));
    free[best] = false;
  }
}

/** Messages with the rounding-level numbers (|v| < 1e-12) masked. */
const mask = (s: string) => s.replace(/\d(\.\d+)?e-(1[2-9]|[2-9]\d)/g, '#');

describe('AAA and Floater–Hormann against the Python oracle', () => {
  for (const c of ORACLE.cases) {
    it(c.name, () => {
      if (c.error) {
        expect(() => run(c)).toThrow(c.error.replace(/^\w+: /, ''));
        return;
      }
      const want = c.result!;
      const got = run(c);
      const fScale = Math.max(...nums(want.extra.values).map(Math.abs), ...(c.problem.y ?? []).map(Math.abs), 1);
      if (c.method === 'floater_hormann') {
        expect(got.converged).toBe(want.converged);
        expect(got.nIter).toBe(want.n_iter);
        expect(got.message).toBe(want.message);
        closeArr(got.x as number[], want.x, 1e-12, 0, 'w');
        expect(got.trace.length).toBe(want.trace.length);
        want.trace.forEach((s, k) => {
          const g = got.trace[k];
          expect(g.info.window).toEqual(s.info.window);
          expect(g.info.node_index).toBe(s.info.node_index);
          closeArr(nums(g.x), s.x, 1e-12, 0, `w at step ${k}`);
        });
        closeArr(nums((got.extra.eval as { y: number[] }).y).filter((_, i) => i % 7 === 0), nums((want.extra.eval as { y: number[] }).y), 1e-10, 1e-12, 'curve');
        expect(got.extra.d).toBe(want.extra.d);
        return;
      }
      // AAA: the greedy choice, the weights and the sample error of every step. A pick that
      // rounding decides (two samples tied for the largest error) may go the other way: the
      // comparison stops there. A tie for the largest |w_j| fixes the sign of w by rounding:
      // w is compared up to sign there (r does not depend on it). A weight that is 0 in exact
      // arithmetic is 0 or rounding noise, and a noise weight can add a real pole: the poles
      // of such steps (step_data) are not compared.
      const ties = (want as unknown as { ties: { pick: number[]; sign: number[]; zero_w: number[]; w_cond: Num[] } }).ties;
      let diverged = false;
      let zeroW = false;
      for (const [k, s] of want.trace.entries()) {
        const g = got.trace[k];
        const at = `${c.name}, step ${k}`;
        expect(g, at).toBeDefined();
        if (g.info.node_index !== s.info.node_index && ties.pick.includes(k)) {
          diverged = true;
          break;
        }
        expect(g.info.node_index, at).toBe(s.info.node_index);
        const gw = nums(g.x);
        const flip = ties.sign.includes(k) && gw.length && Math.sign(gw[0]) !== Math.sign(s.x[0]) ? -1 : 1;
        // The tolerance follows the condition of the minimal singular vector, eps·σ₁/(σ_{m−1} − σ_m).
        const wTol = Math.max(1e-8, 100 * num(ties.w_cond[k]));
        closeArr(gw.map((v) => flip * v), s.x, 0, wTol, `${at}: w (tolerance ${wTol})`);
        expect(Math.abs(Number(g.info.sample_error) - num(s.info.sample_error as Num)), `${at}: sample error`).toBeLessThanOrEqual(1e-9 * fScale + 1e-6 * num(s.info.sample_error as Num));
        if (k > 0) {
          const sw = num(s.info.sigma_min as Num);
          expect(Math.abs(Number(g.info.sigma_min) - sw), `${at}: sigma_min`).toBeLessThanOrEqual(1e-9 + 1e-6 * sw);
        }
        if (ties.zero_w.includes(k)) {
          zeroW = true;
          continue;
        }
        expect(g.info.n_interval_poles, `${at}: real poles`).toBe(s.info.n_interval_poles);
        // Poles move with w: their tolerance follows the condition of w too.
        const pTol = Math.max(1e-6, 30 * num(ties.w_cond[k]));
        closeArr(nums(g.info.interval_poles), nums(s.info.interval_poles), 0, pTol, `${at}: interval poles`);
        samePoles(g.info.poles as number[][], s.info.poles as number[][], pTol, at);
      }
      if (diverged) {
        // Another, equally valid greedy path: it must still end in a valid approximant.
        expect(got.trace.length).toBeGreaterThan(1);
        if (got.converged) expect(Number(got.extra.sample_error)).toBeLessThanOrEqual(Number(c.params.tol ?? 1e-13) * fScale);
        return;
      }
      expect(got.converged).toBe(want.converged);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.trace.length).toBe(want.trace.length);
      if (!zeroW) {
        expect(mask(got.message)).toBe(mask(want.message));
        expect(got.extra.n_doublets).toBe(want.extra.n_doublets);
        expect(got.extra.n_interval_poles).toBe(want.extra.n_interval_poles);
      }
      closeArr(nums(got.extra.errors), nums(want.extra.errors), 1e-6, 1e-9 * fScale, 'errors');
    });
  }
});

describe('pole helpers against the Python oracle', () => {
  ORACLE.helpers.forEach((h, i) => {
    it(`set ${i} (m = ${h.z.length})`, () => {
      const { poles, residues } = barycentricPoles(h.z, h.w, h.f);
      samePoles(poles, h.poles, 1e-9, 'poles');
      samePoles(residues, h.residues, 1e-7, 'residues');
      const br = realDenominatorRoots(h.z, h.w, -1, 1);
      expect(br.length).toBe(h.brackets.length);
      br.forEach(([lo, hi], j) => {
        expect(Math.abs(lo - h.brackets[j][0])).toBeLessThan(1e-12);
        expect(Math.abs(hi - h.brackets[j][1])).toBeLessThan(1e-12);
      });
    });
  });
});

describe('the mathematics of the port', () => {
  it('complex eigenvalues of a non-normal matrix', () => {
    // Upper triangular plus a rotation block: eigenvalues 2, 3 + i, 3 − i, −1 + 2i.
    const re = [
      [2, 5, 1, 0],
      [0, 3, -1, 4],
      [0, 1, 3, 2],
      [0, 0, 0, -1],
    ];
    const im = [
      [0, 1, 0, 3],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 2],
    ];
    const ev = complexEigvals(re, im)!;
    const want: Complex[] = [
      [2, 0],
      [3, 1],
      [3, -1],
      [-1, 2],
    ];
    samePoles(ev, want, 1e-12, 'eigenvalues');
  });

  it('the minimal right singular vector, also of a wide matrix', () => {
    const a = [
      [1, 2, 3],
      [2, 4, 6.000001],
      [1, 0, 1],
      [0, 1, 1],
    ];
    const { sigma, v } = minRightSingular(a, 3)!;
    const av = a.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
    expect(Math.hypot(...av)).toBeCloseTo(sigma, 12);
    expect(Math.hypot(...v)).toBeCloseTo(1, 14);
    const wide = [
      [1, 2, 3],
      [0, 1, 4],
    ];
    const n = minRightSingular(wide, 3)!;
    wide.forEach((r) => expect(Math.abs(r.reduce((s, x, j) => s + x * n.v[j], 0))).toBeLessThan(1e-14));
  });

  it('Floater–Hormann weights: the integer weights on equispaced nodes (FH 2007, §4)', () => {
    const x = Array.from({ length: 9 }, (_, i) => i / 8);
    // d = 3: |w_k|·d! = 1, 4, 7, 8, 8, 8, 7, 4, 1 with alternating signs.
    const { w, windows } = floaterHormannWeights(x, 3);
    expect(w.map((v) => Math.round(v * 6 * 1e9) / 1e9)).toEqual([-1, 4, -7, 8, -8, 8, -7, 4, -1]);
    expect(windows[0]).toEqual([0, 0]);
    expect(windows[4]).toEqual([1, 4]);
    expect(windows[8]).toEqual([5, 5]);
    expect(() => floaterHormannWeights(x, 9)).toThrow('need 0 ≤ d ≤ n = 8');
  });

  it('Floater–Hormann reproduces polynomials of degree ≤ d and has no real poles', () => {
    const x = [0, 0.1, 0.35, 0.4, 0.62, 0.8, 0.97, 1.3, 1.5];
    const p = (t: number) => 1 - 2 * t + 0.5 * t ** 3;
    const m = getMethod('floater_hormann');
    const r = m.fn([x, x.map(p)] as const, { d: 3 });
    const t = Array.from({ length: 50 }, (_, i) => -0.2 + (1.9 * i) / 49);
    const w = r.extra.coefficients as number[];
    barycentricEval(x, w, x.map(p), t).forEach((v, i) => expect(v).toBeCloseTo(p(t[i]), 11));
    // Sign condition: consecutive weights alternate, so d has no real zero (FH 2007, Thm 1).
    expect(realDenominatorRoots(x, w, -10, 10)).toEqual([]);
  });

  it('AAA recovers the Runge function exactly: type (2, 2), poles ±i/5', () => {
    const r = getMethod('aaa').fn(getProblem('runge_equispaced'), { tol: 1e-13, max_terms: 100, scaling: 'columns' });
    expect(r.converged).toBe(true);
    expect(r.nIter).toBe(3);
    samePoles(r.extra.poles as number[][], [
      [0, 0.2],
      [0, -0.2],
    ], 1e-10, 'Runge poles');
    expect(r.fun).toBeLessThan(1e-14);
  });

  it('AAA flags the real poles it puts between noisy samples', () => {
    const r = getMethod('aaa').fn(getProblem('noisy_linear'), { tol: 1e-13, max_terms: 100, scaling: 'columns' });
    const last = r.trace[r.trace.length - 1].info;
    expect(Number(last.n_interval_poles)).toBeGreaterThan(0);
    expect(r.message).toMatch(/warning: \d+ certified real pole\(s\) of r in \[0, 10\]/);
    // Each certified pole is a sign change of the denominator d inside the bracket.
    const z = last.support as number[],
      w = last.weights as number[];
    const d = (t: number) => z.reduce((s, zj, j) => s + w[j] / (t - zj), 0);
    for (const p of last.interval_poles as number[]) {
      const h = 1e-6;
      expect(Math.sign(d(p - h)) * Math.sign(d(p + h))).toBe(-1);
    }
  });
});
