/**
 * Shared helpers of the conjugate-gradient / trust-region port tests.
 */
import { expect } from 'vitest';
import { resultFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Result } from '../../src/core/types';
import '../../src/methods/unconstrained/conjugate_gradient';
import '../../src/methods/unconstrained/trust_region';
import { mismatch, type Tol } from '../shared-ports/compare';

export type Raw = Record<string, unknown>;

export const IDS = [
  'cg_dai_yuan',
  'cg_fletcher_reeves',
  'cg_hager_zhang',
  'cg_hestenes_stiefel',
  'cg_polak_ribiere',
  'trust_region_cauchy',
  'trust_region_dogleg',
  'trust_region_exact',
  'trust_region_steihaug',
];

/** Agreement of a whole replay: rounding differences (BLAS dot order, Jacobi vs LAPACK eigh). */
export const STEP_TOL: Tol = { rtol: 1e-9, atol: 1e-12 };

export function run(method: string, problem: unknown, params: Raw): Result {
  const { spec, fn } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

/** Compare a TS Result with a Python result (snake_case JSON) field by field. */
export function expectSameResult(got: Result, wantRaw: Raw, tol: Tol = STEP_TOL) {
  const want = resultFromJson(wantRaw);
  expect(got.method).toBe(want.method);
  expect(got.message).toBe(want.message);
  expect(got.converged).toBe(want.converged);
  expect([got.nIter, got.nFev, got.nGev, got.nHev]).toEqual([
    want.nIter,
    want.nFev,
    want.nGev,
    want.nHev,
  ]);
  expect(got.trace.length).toBe(want.trace.length);
  got.trace.forEach((s, k) => {
    const w = want.trace[k];
    const m = mismatch(
      { k: s.k, x: s.x, fun: s.fun, gradNorm: s.gradNorm, stepSize: s.stepSize, info: s.info },
      { k: w.k, x: w.x, fun: w.fun, gradNorm: w.gradNorm, stepSize: w.stepSize, info: w.info },
      tol,
      `trace[${k}]`,
    );
    expect(m).toBeNull();
  });
  expect(mismatch(got.x, want.x, tol, 'x')).toBeNull();
  expect(mismatch(got.fun, want.fun, tol, 'fun')).toBeNull();
  expect(mismatch(got.extra, want.extra, tol, 'extra')).toBeNull();
}
