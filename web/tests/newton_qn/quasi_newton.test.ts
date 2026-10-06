/**
 * numopt.unconstrained.quasi_newton — TS port checks beyond the parity harness:
 *   - the registered specs equal registry.json;
 *   - every quasi-Newton fixture of unconstrained.json matches step by step (x, fun, ‖∇f‖, α,
 *     every info key incl. s, y, ρ, γ, H), with the same counts and message;
 *   - more Python reference runs (tests/newton_qn/fixtures, from gen_newton_qn_fixture.py): every
 *     method × every line search on seven problems, φ ∈ {0, 0.3, 1}, m ∈ {1, 3, 10},
 *     finite-difference ∇f, n = 3, an unbounded problem, max_iter stops, gtol = 0, ValueErrors;
 *   - focused unit tests (the update formulas, skips, the two-loop recursion).
 */
import { describe, expect, it } from 'vitest';
import type { Vector } from '../../src/core/types';
import {
  bfgs,
  curvatureOk,
  lbfgs,
  scaledNorm,
  twoLoop,
} from '../../src/methods/unconstrained/quasi_newton';
import '../../src/problems';
import {
  caseKey,
  errorOf,
  expectSameResult,
  expectSimilarResult,
  generatedFixtures,
  problemById,
  pythonSpecs,
  referenceCases,
  run,
  tsSpec,
} from './harness';

const IDS = ['bfgs', 'dfp', 'sr1', 'broyden_class', 'lbfgs'] as const;

describe('quasi_newton registry', () => {
  it('registers the five Python methods with identical specs', () => {
    const python = pythonSpecs(IDS);
    expect(python.map((s) => s.id).sort()).toEqual([...IDS].sort());
    for (const want of python) expect(tsSpec(want.id)).toEqual(want);
  });
});

describe('quasi_newton fixtures (unconstrained.json), step by step', () => {
  const cases = generatedFixtures(IDS);
  it('has every Python fixture case', () => expect(cases.length).toBe(8));
  cases.forEach((c, i) => {
    it(`${c.method} on ${c.problem} #${i}`, () => {
      const problem = problemById(c.problem);
      const got = run(c.method, problem, c.params);
      // n = 10 (rosenbrock_nd): BLAS summation order; the parity rule still holds exactly.
      if ((problem as { dim: number }).dim > 2) {
        expectSimilarResult(got, c.result!);
        expect(got.nIter).toBe((c.result as { n_iter: number }).n_iter);
      } else expectSameResult(got, c.result!);
    });
  });
});

/**
 * 2-D runs whose first difference from Python is a last-bit difference of f or ∇f at the SAME x
 * (NumPy's scalar `x ** 2` on himmelblau, a central-difference ∇f), amplified where yᵀs is at the
 * rounding level of ∇f. Every other 2-D run matches every stored step to 1e-9 (most bit for bit).
 */
const NOISY_2D = new Set([
  'dfp himmelblau {"line_search":"backtracking"}',
  'broyden_class himmelblau {"line_search":"backtracking","phi":1}',
  'broyden_class rosen_fonly {"phi":0.3}',
  'lbfgs himmelblau {"line_search":"backtracking","m":1}',
  'lbfgs six_hump_camel {"line_search":"backtracking","m":1}',
]);

describe('quasi_newton reference runs (gen_newton_qn_fixture.py)', () => {
  const cases = referenceCases('quasi_newton');
  it('lists only existing cases as noisy', () => {
    const keys = new Set(cases.map(caseKey));
    for (const k of NOISY_2D) expect(keys.has(k)).toBe(true);
  });
  cases.forEach((c, i) => {
    it(`${caseKey(c)} #${i}`, () => {
      const problem = problemById(c.problem);
      if (c.error !== undefined) {
        const err = errorOf(() => run(c.method, problem, c.params));
        expect(err).toEqual({ name: 'ValueError', message: c.error });
        return;
      }
      const got = run(c.method, problem, c.params);
      if ((problem as { dim: number }).dim > 2 || NOISY_2D.has(caseKey(c)))
        expectSimilarResult(got, c.result!);
      else expectSameResult(got, c.result!);
    });
  });
});

describe('quasi_newton helpers', () => {
  it('scaledNorm does not underflow for a tiny nonzero vector', () => {
    expect(scaledNorm([1e-200, 0])).toBe(1e-200);
    expect(scaledNorm([3, 4])).toBe(5);
    expect(scaledNorm([0, 0])).toBe(0);
    expect(scaledNorm([NaN, 1])).toBeNaN();
  });

  it('curvatureOk: yᵀs > 1e-10‖s‖‖y‖ with finite ρ and γ', () => {
    expect(curvatureOk([1, 0], [1, 0], 1)).toBe(true);
    expect(curvatureOk([1, 0], [-1, 0], -1)).toBe(false);
    // yᵀs > 0 but below the relative threshold
    expect(curvatureOk([1, 0], [1e-12, 1], 1e-12)).toBe(false);
    // yᵀy underflows to 0 while yᵀs > 0: γ is not representable
    expect(curvatureOk([1, 0], [1e-170, 0], 1e-170)).toBe(false);
  });

  it('twoLoop with an empty memory is γ g; with one pair it equals the BFGS update of γI', () => {
    expect(twoLoop([2, -4], [], 0.5)).toEqual([1, -2]);
    const s: Vector = [0.3, -0.1];
    const y: Vector = [0.5, 0.2];
    const ys = s[0] * y[0] + s[1] * y[1];
    const rho = 1 / ys;
    const gamma = ys / (y[0] * y[0] + y[1] * y[1]);
    // H = (I − ρ s yᵀ) γI (I − ρ y sᵀ) + ρ s sᵀ
    const V = [
      [1 - rho * s[0] * y[0], -rho * s[0] * y[1]],
      [-rho * s[1] * y[0], 1 - rho * s[1] * y[1]],
    ];
    const H = [0, 1].map((i) =>
      [0, 1].map((j) => gamma * (V[i][0] * V[j][0] + V[i][1] * V[j][1]) + rho * s[i] * s[j]),
    );
    const g: Vector = [1.5, -0.7];
    const got = twoLoop(g, [[s, y, rho]], gamma);
    const want = [H[0][0] * g[0] + H[0][1] * g[1], H[1][0] * g[0] + H[1][1] * g[1]];
    expect(got[0]).toBeCloseTo(want[0], 14);
    expect(got[1]).toBeCloseTo(want[1], 14);
  });

  it('BFGS keeps H exactly symmetric and satisfies the secant equation', () => {
    const r = bfgs(problemById('rosenbrock') as never, { max_iter: 6 });
    for (const step of r.trace.slice(1)) {
      const H = step.info.H as number[][];
      expect(H[0][1]).toBe(H[1][0]);
      if (step.info.update === 'applied') {
        const s = step.info.s as Vector;
        const y = step.info.y as Vector;
        const Hy = [H[0][0] * y[0] + H[0][1] * y[1], H[1][0] * y[0] + H[1][1] * y[1]];
        expect(Hy[0]).toBeCloseTo(s[0], 10);
        expect(Hy[1]).toBeCloseTo(s[1], 10);
      }
    }
  });

  it('L-BFGS keeps at most m pairs and reports the memory size', () => {
    const r = lbfgs(problemById('rosenbrock') as never, { m: 2, max_iter: 8 });
    const mem = r.trace.map((s) => s.info.memory);
    expect(mem[0]).toBe(0);
    expect(Math.max(...(mem as number[]))).toBe(2);
    expect(r.nHev).toBe(0);
  });

  it('throws the Python ValueError for invalid parameters', () => {
    const rosen = problemById('rosenbrock');
    expect(errorOf(() => run('lbfgs', rosen, { m: 2.5 }))).toEqual({
      name: 'ValueError',
      message: 'm must be a positive integer, got 2.5',
    });
    expect(errorOf(() => run('sr1', rosen, { line_search: 'exact_quadratic' }))?.message).toBe(
      "unknown line_search 'exact_quadratic'; expected one of " +
        "('strong_wolfe', 'weak_wolfe', 'backtracking', 'goldstein')",
    );
  });
});
