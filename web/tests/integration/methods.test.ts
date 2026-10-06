/**
 * Integration port: parity with the Python fixtures (src/generated/fixtures/integration.json)
 * plus unit tests for what the fixtures do not cover (fsum, Gauss nodes, exactness on
 * polynomials, failure paths, bracket overrides, messages).
 *
 * This file imports only the integration port and the calculus problems, so it does not depend
 * on other families' ports (the global harness tests/parity.test.ts globs every port).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Problem, Result } from '../../src/core/types';
import '../../src/problems/calculus';
import {
  chebyshevCoefficients,
  clenshawCurtisRule,
  compositeNodesWeights,
  confirmedRegime,
  dft,
  fsum,
  fsumExact,
  gaussLegendreRule,
  gaussPattersonRule,
  pyG,
  sequenceError,
} from '../../src/methods/integration/methods';

const FIXTURE = fileURLToPath(
  new URL('../../src/generated/fixtures/integration.json', import.meta.url),
);

function run(id: string, problem: unknown, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

function close(a: number, b: number, rtol: number, atol = rtol) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return expect(a).toBe(b);
  expect(Math.abs(a - b)).toBeLessThanOrEqual(atol + rtol * Math.abs(b));
}

/** Compare two info values (numbers, arrays, booleans, null) at a tolerance. */
function sameInfo(a: unknown, b: unknown, rtol: number): void {
  if (typeof b === 'number' && typeof a === 'number') return close(a, b, rtol, 1e-14);
  if (Array.isArray(b)) {
    expect(Array.isArray(a)).toBe(true);
    expect((a as unknown[]).length).toBe(b.length);
    b.forEach((v, i) => sameInfo((a as unknown[])[i], v, rtol));
    return;
  }
  expect(a).toEqual(b);
}

describe.runIf(existsSync(FIXTURE))('integration parity with Python fixtures', () => {
  const cases = (JSON.parse(readFileSync(FIXTURE, 'utf8')) as Record<string, unknown>[]).map(
    fixtureCaseFromJson,
  );
  it('has every fixture case', () => expect(cases.length).toBe(22));
  cases.forEach((c, idx) => {
    it(`${c.method} on ${c.problem} #${idx}`, () => {
      const { spec } = getMethod(c.method);
      const result = run(c.method, getProblem(c.problem), c.params as Record<string, unknown>);
      const want = c.result;
      const n = Math.min(10, want.trace.length);
      expect(result.trace.length).toBeGreaterThanOrEqual(n);
      for (let k = 0; k < n; k++)
        close(result.trace[k].x as number, want.trace[k].x as number, 1e-8);
      if (spec.deterministic) {
        expect(result.nIter).toBe(want.nIter);
        close(result.x as number, want.x as number, 1e-6, 1e-10);
        expect(result.converged).toBe(want.converged);
        // Beyond the harness: the whole trace, the counts, the message and the info keys.
        expect(result.trace.length).toBe(want.trace.length);
        expect(result.nFev).toBe(want.nFev);
        // Gauss: an error estimate at the rounding level differs in its last digits. Nested
        // rules: the nodes and weights agree bit for bit, but Math.exp and the C library's exp
        // can differ by 1 ulp at a node, so a difference d_k at the rounding level (Patterson on
        // exp_0_1: 0 in Python, 2.2×10⁻¹⁶ here) is noise in both languages.
        const nested = c.method === 'clenshaw_curtis' || c.method === 'gauss_patterson';
        const atRounding =
          Number(want.extra.err_est) <= 1e-15 * Math.max(1, Math.abs(want.x as number));
        if (c.method === 'gauss_legendre' || (nested && atRounding))
          expect(result.message.replace(/[\d.e+-]+ ([≤>])/, '# $1')).toBe(
            want.message.replace(/[\d.e+-]+ ([≤>])/, '# $1'),
          );
        else expect(result.message).toBe(want.message);
        result.trace.forEach((s, k) => {
          const w = want.trace[k];
          close(s.x as number, w.x as number, 1e-12, 1e-14);
          expect(Object.keys(s.info).sort()).toEqual(Object.keys(w.info).sort());
          for (const key of Object.keys(w.info)) {
            // err_est and ratio of Gauss rules depend on O(ε) differences of the eigensolver;
            // once the differences reach the rounding level they are noise in both languages.
            const noisy = key === 'err_est' || key === 'ratio';
            if (noisy && c.method === 'gauss_legendre') {
              const d = k > 0 ? Math.abs((w.x as number) - (want.trace[k - 1].x as number)) : 1;
              if (key === 'ratio' && d < 1e-12) continue;
              if (key === 'err_est' && Number(w.info.err_est) < 1e-12) continue;
            }
            const tol = noisy ? 1e-6 : 1e-9;
            sameInfo(s.info[key], w.info[key], tol);
          }
          if (w.stepSize === null) expect(s.stepSize).toBeNull();
          else close(s.stepSize as number, w.stepSize, 1e-15);
        });
      } else {
        close(result.fun as number, want.fun as number, 1e-6, 1e-10);
        // Mulberry32 + Welford are exact in both languages: the whole trace agrees.
        expect(result.trace.length).toBe(want.trace.length);
        result.trace.forEach((s, k) => {
          close(s.x as number, want.trace[k].x as number, 1e-12, 1e-14);
          expect(s.info.n_samples).toBe(want.trace[k].info.n_samples);
        });
        expect(result.message).toBe(want.message);
      }
    });
  });
});

describe('fsum (port of CPython math.fsum)', () => {
  it('is correctly rounded where the naive sum is not', () => {
    expect(fsum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])).toBe(1.0);
    expect(fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50])).toBe(1e-100);
    expect(fsum([1.0, 1e-16, 1e-16])).toBe(1.0000000000000002);
    expect(fsum([])).toBe(0);
  });
  it('rounds half-even across partials (CPython test case)', () => {
    expect(fsum([2.0 ** 53, -0.5, -(2.0 ** -54)])).toBe(2.0 ** 53 - 1.0);
    expect(fsum([2.0 ** 53, 1.0, 2.0 ** -100])).toBe(2.0 ** 53 + 2.0);
  });
  it('returns null where Python raises, and fsum falls back to the IEEE sum', () => {
    expect(fsumExact([Infinity, -Infinity])).toBeNull();
    expect(fsumExact([1.7e308, 1.7e308])).toBeNull();
    expect(fsum([1.7e308, 1.7e308])).toBe(Infinity);
    expect(fsum([Infinity, 1])).toBe(Infinity);
    expect(fsum([NaN, 1])).toBeNaN();
  });
});

describe('pyG (Python format .Ng)', () => {
  it('matches Python', () => {
    expect(pyG(0.006738670057229332, 3)).toBe('0.00674');
    expect(pyG(1.2345e-5, 3)).toBe('1.23e-05');
    expect(pyG(123456, 3)).toBe('1.23e+05');
    expect(pyG(0.5, 6)).toBe('0.5');
    expect(pyG(-2, 6)).toBe('-2');
    expect(pyG(100, 3)).toBe('100');
    expect(pyG(0.0001, 3)).toBe('0.0001');
  });
});

describe('Gauss–Legendre rule (Golub–Welsch)', () => {
  it('matches the classical nodes and weights', () => {
    const [t2, w2] = gaussLegendreRule(2);
    close(t2[1], 1 / Math.sqrt(3), 1e-15);
    close(t2[0], -1 / Math.sqrt(3), 1e-15);
    close(w2[0], 1, 1e-15);
    const [t3, w3] = gaussLegendreRule(3);
    expect(t3[1]).toBe(0);
    close(t3[2], Math.sqrt(3 / 5), 1e-15);
    close(w3[1], 8 / 9, 1e-15);
    close(w3[0], 5 / 9, 1e-15);
  });
  it('is symmetric, sums to 2 and integrates degree 2m − 1 exactly (m ≤ 64)', () => {
    for (const m of [1, 4, 7, 20, 33, 64]) {
      const [t, w] = gaussLegendreRule(m);
      for (let i = 0; i < m; i++) {
        expect(t[i] + t[m - 1 - i]).toBe(0);
        expect(w[i]).toBe(w[m - 1 - i]);
        if (i > 0) expect(t[i]).toBeGreaterThan(t[i - 1]);
      }
      close(fsum(w), 2, 1e-14);
      // ∫₋₁¹ t^(2j) dt = 2/(2j + 1)
      for (let j = 0; 2 * j <= 2 * m - 1; j++)
        close(fsum(w.map((wi, i) => wi * t[i] ** (2 * j))), 2 / (2 * j + 1), 1e-13);
    }
  });
});

describe('composite nodes and weights', () => {
  it('builds the textbook weight patterns', () => {
    const [x, w] = compositeNodesWeights('simpson', 0, 1, 4);
    expect(x).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(w.map((v) => v * 12)).toEqual([1, 4, 2, 4, 1].map((c) => expect.closeTo(c, 12)));
    const [, wb] = compositeNodesWeights('boole', 0, 1, 8);
    expect(wb.map((v) => (v * 45 * 8) / 2)).toEqual(
      [7, 32, 12, 32, 14, 32, 12, 32, 7].map((c) => expect.closeTo(c, 10)),
    );
    const [xm, wm, pm] = compositeNodesWeights('midpoint_rule', 0, 1, 2);
    expect(xm).toEqual([0.25, 0.75]);
    expect(wm).toEqual([0.5, 0.5]);
    expect(pm).toEqual([
      [0, 0.5],
      [0.5, 1],
    ]);
    expect(() => compositeNodesWeights('simpson_38', 0, 1, 4)).toThrow(/multiple of 3/);
  });
});

describe('error estimate helpers', () => {
  it('uses the textbook value at k = 1 and the geometric tail when contraction is slow', () => {
    const [e1, r1] = sequenceError([0.3], [0], 4);
    expect(e1).toBeCloseTo(0.1, 15);
    expect(r1).toBeNull();
    // ρ = 2 < R = 16: d/(ρ − 1) dominates.
    const [est, ratio] = sequenceError([0.2, 0.1], [0, 0], 16);
    expect(ratio).toBe(2);
    expect(est).toBeCloseTo(0.1, 15);
  });
  it('confirms a clean geometric sequence and rejects a sign change', () => {
    expect(confirmedRegime([-1, -0.25, -0.0625, -0.015625], [0, 0, 0, 0])).toBe(true);
    expect(confirmedRegime([-1, 0.25, -0.0625, -0.015625], [0, 0, 0, 0])).toBe(false);
    expect(confirmedRegime([1e-17, -1e-17], [1e-16, 1e-16])).toBe(true);
  });
});

const poly = (c: number[]): Problem<number> => ({
  id: 'poly',
  name: 'poly',
  latex: '',
  dim: 1,
  domain: [0, 2],
  f: (x: number) => c.reduce((s, ci, i) => s + ci * x ** i, 0),
  exact: c.reduce((s, ci, i) => s + (ci * 2 ** (i + 1)) / (i + 1), 0),
});

describe('exactness on polynomials of the right degree', () => {
  const cases: [string, number][] = [
    ['left_riemann', 0],
    ['right_riemann', 0],
    ['midpoint_rule', 1],
    ['trapezoid', 1],
    ['simpson', 3],
    ['simpson_38', 3],
    ['boole', 5],
  ];
  for (const [id, deg] of cases) {
    it(`${id} is exact for degree ${deg} at level 0`, () => {
      const c = Array.from({ length: deg + 1 }, (_, i) => 1 + 0.5 * i);
      const r = run(id, poly(c), { levels: 0 });
      close(r.x as number, poly(c).exact as number, 1e-14);
      expect(r.converged).toBe(false);
      expect(r.message).toBe('levels = 0: a single estimate has no error estimate');
    });
  }
  it('Gauss with m points is exact for degree 2m − 1', () => {
    const c = [1, -2, 3, 0.5, -0.25, 0.125, 2, -1];
    const r = run('gauss_legendre', poly(c), { n: 4 });
    close(r.trace[3].x as number, poly(c).exact as number, 1e-13);
  });
  it('Romberg R(2,2) is Boole on four subintervals', () => {
    const p = getProblem<Problem<number>>('exp_0_1');
    const r = run('romberg', p, { max_levels: 3 });
    const b = run('boole', p, { n: 4, levels: 0 });
    close((r.trace[2].info.row as number[])[2], b.x as number, 1e-14);
  });
});

describe('failure paths and options', () => {
  it('reports a non-finite f with the abscissa', () => {
    const f = (x: number) => 1 / x;
    const r = run('trapezoid', f, { bracket: [0, 1] });
    expect(r.converged).toBe(false);
    expect(r.message).toBe('f is not finite at x = 0');
    expect(r.nIter).toBe(0);
    const g = run('gauss_legendre', (x: number) => Math.log(x - 0.5), { bracket: [0, 1], n: 5 });
    expect(g.message).toMatch(/^f is not finite at x = /);
  });
  it('rejects invalid input like Python', () => {
    expect(() => run('simpson', getProblem('exp_0_1'), { n: 3 })).toThrow(/multiple of 2/);
    expect(() => run('trapezoid', getProblem('exp_0_1'), { bracket: [1, 0] })).toThrow(/a < b/);
    expect(() => run('trapezoid', getProblem('exp_0_1'), { tol: 0 })).toThrow(/tol/);
    expect(() => run('trapezoid', getProblem('exp_0_1'), { n: 64, levels: 13 })).toThrow(
      /exceeds the limit/,
    );
  });
  it('honours a bracket override (and, like Python, keeps the problem’s exact value)', () => {
    const r = run('simpson', getProblem('exp_0_1'), { bracket: [0, 2], levels: 4 });
    close(r.x as number, Math.exp(2) - 1, 1e-8);
  });
  it('adaptive Simpson clamps min_depth to max_depth and flags forced intervals', () => {
    const r = run('adaptive_simpson', getProblem('sqrt_0_1'), {
      min_depth: 5,
      max_depth: 3,
      tol: 1e-12,
    });
    expect(r.extra.min_depth).toBe(3);
    expect(r.converged).toBe(false);
    expect(r.message).toMatch(/accepted without passing the Lyness test/);
    // Accepted intervals tile [a, b] in order.
    const iv = r.extra.intervals as [number, number][];
    expect(iv[0][0]).toBe(0);
    expect(iv[iv.length - 1][1]).toBe(1);
    for (let i = 1; i < iv.length; i++) expect(iv[i][0]).toBe(iv[i - 1][1]);
  });
  it('adaptive Simpson stops at the interval budget', () => {
    const r = run('adaptive_simpson', getProblem('runge'), { tol: 1e-12, max_iter: 5 });
    expect(r.nIter).toBe(5);
    expect(r.message).toMatch(/^reached max_iter=5 accepted intervals with \d+ pending$/);
  });
  it('Monte Carlo with one sample has no standard error', () => {
    const r = run('monte_carlo_integration', getProblem('sin_0_pi'), { n: 1, levels: 0 });
    expect(r.trace[0].info.err_est).toBeNull();
    expect(r.message).toBe('one sample: no standard error');
  });
  it('Romberg converges on a smooth integrand with the documented message', () => {
    const r = run('romberg', getProblem('arctan_deriv'));
    expect(r.converged).toBe(true);
    close(r.x as number, Math.PI / 4, 1e-12);
    expect(r.message).toMatch(/trapezoid column in its asymptotic regime$/);
  });
});

describe('nested rules: Clenshaw–Curtis and Gauss–Kronrod–Patterson', () => {
  /** Legendre P_j(t) by the three-term recurrence. */
  const legendre = (j: number, t: number) => {
    let p0 = 1,
      p1 = t;
    if (j === 0) return 1;
    for (let k = 1; k < j; k++) [p0, p1] = [p1, ((2 * k + 1) * t * p1 - k * p0) / (k + 1)];
    return p1;
  };
  /**
   * The rule's error on P_j: ∫₋₁¹ P_j = 2 for j = 0, else 0. Unlike tʲ, which is tiny inside
   * [−1, 1] for large j, P_j has size 1 there, so the first degree that fails shows an O(1)-ish
   * error.
   */
  const momentError = (t: number[], w: number[], j: number) =>
    Math.abs(fsum(w.map((wi, i) => wi * legendre(j, t[i]))) - (j === 0 ? 2 : 0));

  it('dft matches the direct sum for mixed lengths (odd factor × power of 2)', () => {
    for (const N of [1, 2, 3, 5, 6, 12, 40, 63, 96]) {
      const re = Array.from({ length: N }, (_, j) => Math.cos(0.7 * j * j + 0.3));
      const im = Array.from({ length: N }, (_, j) => Math.sin(1.3 * j));
      for (const sign of [1, -1] as const) {
        const [R, I] = dft(re, im, sign);
        for (let k = 0; k < N; k++) {
          let sr = 0,
            si = 0;
          for (let j = 0; j < N; j++) {
            const th = (sign * 2 * Math.PI * j * k) / N;
            sr += re[j] * Math.cos(th) - im[j] * Math.sin(th);
            si += re[j] * Math.sin(th) + im[j] * Math.cos(th);
          }
          close(R[k], sr, 0, 1e-12);
          close(I[k], si, 0, 1e-12);
        }
      }
    }
  });

  it('Clenshaw–Curtis: n = 1 is the trapezoid rule and n = 2 is Simpson', () => {
    expect(clenshawCurtisRule(1)).toEqual([
      [1, -1],
      [1, 1],
    ]);
    const [t, w] = clenshawCurtisRule(2);
    expect(t).toEqual([1, 0, -1]);
    w.forEach((wi, i) => close(wi, [1 / 3, 4 / 3, 1 / 3][i], 1e-15));
  });

  it('Clenshaw–Curtis: cosine nodes, symmetric positive weights, exact to degree n (n + 1 if even)', () => {
    for (const n of [3, 4, 7, 16, 33, 64]) {
      const [t, w] = clenshawCurtisRule(n);
      expect(t).toHaveLength(n + 1);
      expect(t[0]).toBe(1);
      expect(t[n]).toBe(-1);
      for (let k = 0; k <= n; k++) {
        close(t[k], Math.cos((k * Math.PI) / n), 0, 4e-16);
        expect(t[k] + t[n - k]).toBe(0);
        expect(w[k]).toBe(w[n - k]);
        expect(w[k]).toBeGreaterThan(0);
      }
      // w₀ = 1/(n² − 1 + n mod 2), Waldvogel (2.6).
      close(w[0], 1 / (n * n - 1 + (n % 2)), 1e-13);
      const deg = n % 2 ? n : n + 1;
      for (let j = 0; j <= deg; j++) expect(momentError(t, w, j)).toBeLessThan(1e-14);
      // Not exact at the first even degree above (odd P_j vanish by symmetry); the error on
      // P_{n+2} decays like n⁻³ (Trefethen 2008, Thm. 5.2): 8.5×10⁻⁶ at n = 64.
      const jBad = deg % 2 ? deg + 1 : deg + 2;
      expect(momentError(t, w, jBad)).toBeGreaterThan(1e-6);
    }
  });

  it('Clenshaw–Curtis grids are nested bit for bit: node k of n is node 2k of 2n', () => {
    for (const n of [1, 3, 5, 12, 64]) {
      const [t1] = clenshawCurtisRule(n);
      const [t2] = clenshawCurtisRule(2 * n);
      t1.forEach((tk, k) => expect(t2[2 * k]).toBe(tk));
    }
  });

  it('Chebyshev coefficients reproduce the samples and integrate to the Clenshaw–Curtis sum', () => {
    const n = 16;
    const [t, w] = clenshawCurtisRule(n);
    const f = t.map((x) => Math.exp(x) * Math.sin(3 * x));
    const a = chebyshevCoefficients(f);
    // p(t_k) = Σ a_j T_j(t_k) = f_k, with T_j(cos θ) = cos jθ.
    t.forEach((_, k) => {
      const p = a.reduce((s, aj, j) => s + aj * Math.cos((j * k * Math.PI) / n), 0);
      close(p, f[k], 0, 1e-14);
    });
    // ∫₋₁¹ T_j = 2/(1 − j²) for even j, 0 for odd j.
    const integral = a.reduce((s, aj, j) => s + (j % 2 ? 0 : (aj * 2) / (1 - j * j)), 0);
    close(integral, fsum(w.map((wi, i) => wi * f[i])), 0, 1e-14);
    expect(() => chebyshevCoefficients([1])).toThrow(/at least two values/);
  });

  it('Gauss–Patterson: 1, 3, 7, …, 127 nested points, exact to degree 3·2^k − 1 and no more', () => {
    let prev: number[] = [];
    for (let k = 0; k <= 6; k++) {
      const [t, w] = gaussPattersonRule(k);
      const N = 2 ** (k + 1) - 1;
      expect(t).toHaveLength(N);
      expect(w).toHaveLength(N);
      close(fsum(w), 2, 1e-15);
      for (let i = 0; i < N; i++) {
        expect(t[i] + t[N - 1 - i]).toBe(0);
        expect(w[i]).toBe(w[N - 1 - i]);
        expect(w[i]).toBeGreaterThan(0);
        if (i > 0) expect(t[i]).toBeLessThan(t[i - 1]);
      }
      // Every old node is a node of the next level (the odd positions), bit for bit.
      prev.forEach((x, i) => expect(t[2 * i + 1]).toBe(x));
      prev = t;
      const deg = k === 0 ? 1 : 3 * 2 ** k - 1;
      for (let j = 0; j <= deg; j++) expect(momentError(t, w, j)).toBeLessThan(1e-14);
      // One degree more fails (0.66 at k = 1, 8.8×10⁻¹¹ at k = 5). At k = 6 the error on P₁₉₂
      // is 7×10⁻¹⁸, below double precision: the Python suite checks that level in 50 digits.
      if (k <= 5) expect(momentError(t, w, deg + 1)).toBeGreaterThan(1e-11);
    }
    // Level 1 is Gauss G₃, level 0 the midpoint rule.
    const [t3, w3] = gaussPattersonRule(1);
    close(t3[0], Math.sqrt(3 / 5), 1e-15);
    close(w3[1], 8 / 9, 1e-15);
    expect(gaussPattersonRule(0)).toEqual([[0], [2]]);
    expect(() => gaussPattersonRule(7)).toThrow(/level must be ≤ 6/);
  });

  it('reuses every old node: new_nodes counts only the new points', () => {
    const cc = run('clenshaw_curtis', getProblem('runge'), { n: 3, max_levels: 4, tol: 1e-15 });
    expect(cc.trace.map((s) => s.info.n_points)).toEqual([4, 7, 13, 25, 49]);
    expect(cc.trace.map((s) => s.info.new_nodes)).toEqual([4, 3, 6, 12, 24]);
    expect(cc.trace.map((s) => s.info.n)).toEqual([3, 6, 12, 24, 48]);
    expect(cc.nFev).toBe(49);
    const gp = run('gauss_patterson', getProblem('runge'), { tol: 1e-15 });
    expect(gp.trace.map((s) => s.info.n_points)).toEqual([1, 3, 7, 15, 31, 63, 127]);
    expect(gp.trace.map((s) => s.info.new_nodes)).toEqual([1, 2, 4, 8, 16, 32, 64]);
    expect(gp.nFev).toBe(127);
    expect(gp.trace[3].info).not.toHaveProperty('cheb_coeffs');
  });

  it('integrates a polynomial of the rule’s degree exactly and stops at k = 2', () => {
    // Degree 9 on [0, 2]: Clenshaw–Curtis n = 3·2^k is exact from k = 2 (n = 12);
    // Patterson's 7 points (degree 11) are exact from k = 2.
    const c = [1, -2, 3, 0.5, -0.25, 0.125, 2, -1, 0.3, 0.05];
    const p = poly(c);
    for (const [id, params] of [
      ['clenshaw_curtis', { n: 3 }],
      ['gauss_patterson', {}],
    ] as const) {
      const r = run(id, p, { ...params, tol: 1e-12 });
      close(r.trace[2].x as number, p.exact as number, 1e-13);
    }
  });

  it('reports a non-finite f, a missed tolerance and invalid input like Python', () => {
    const bad = run('clenshaw_curtis', (x: number) => 1 / x, { bracket: [0, 1] });
    expect(bad.converged).toBe(false);
    expect(bad.message).toBe('f is not finite at x = 0');
    expect(bad.nIter).toBe(0);
    expect(bad.trace[0].info.cheb_coeffs).toBeNull();
    // Patterson is open: it never evaluates f at a or b, so 1/x on [0, 1] is finite at every node.
    const open = run('gauss_patterson', (x: number) => 1 / Math.sqrt(x), {
      bracket: [0, 1],
      max_levels: 3,
    });
    expect(open.converged).toBe(false);
    expect(open.message).toMatch(
      /^reached max_levels=3 \(15 points\) with \|I_k − I_\(k−1\)\| = [\d.e+-]+ > tol·max\(1, \|I\|\)$/,
    );
    expect(open.nIter).toBe(3);
    expect(() => run('clenshaw_curtis', getProblem('exp_0_1'), { n: 65 })).toThrow(
      /exceeds the limit 262144/,
    );
    expect(() => run('clenshaw_curtis', getProblem('exp_0_1'), { max_levels: 1 })).toThrow(
      /max_levels must be an integer ≥ 2/,
    );
    expect(() => run('gauss_patterson', getProblem('exp_0_1'), { max_levels: 7 })).toThrow(
      /max_levels must be ≤ 6 \(127 points\), got 7/,
    );
  });

  it('omits the per-step geometry above 4,096 nodes', () => {
    const r = run('clenshaw_curtis', getProblem('sqrt_0_1'), { n: 64, max_levels: 7, tol: 1e-15 });
    const last = r.trace[r.trace.length - 1];
    expect(last.info.n_points).toBe(8193);
    expect(last.info.nodes).toBeNull();
    expect(last.info.weights).toBeNull();
    expect(last.info.cheb_coeffs).toBeNull();
    // MAX_DISPLAY = 4096: k = 5 (2,049 nodes) is drawn, k = 6 (4,097 nodes) is not.
    expect(r.trace[5].info.nodes).toHaveLength(2049);
    expect(r.trace[5].info.cheb_coeffs).toHaveLength(2049);
    expect(r.trace[6].info.nodes).toBeNull();
  });
});
