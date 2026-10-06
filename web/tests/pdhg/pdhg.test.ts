/**
 * restarted_pdhg against Python on the cases the parity fixtures do not cover
 * (tests/pdhg/fixtures/pdhg_reference.json, from tests/pdhg/gen_pdhg_reference.py): every library
 * LP with every restart scheme, primal weight and preconditioner, a primal start x0, budgets on
 * infeasible and unbounded programs, and the ValueErrors.
 *
 * Per case: the message, status, converged flag and iteration count exactly; the iterations
 * that restart exactly (the lab marks them); the first ten steps with every info key at 1e-8;
 * the final x at 1e-6. Plus unit tests of the helpers (normalized duality gap, scalings, sums).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { LinearProgram, Result } from '../../src/core/types';
import { mismatch } from '../shared-ports/compare';
import {
  normalizedDualityGap,
  npSum,
  pdhgStandardForm,
  primalWeightUpdate,
  restartedPdhg,
  ruizPockChambolle,
  weightedNorm,
} from '../../src/methods/lp/pdhg';
import { singularValues } from '../../src/methods/lp/interior_point';
import '../../src/problems/lp';

interface Case {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  result?: Record<string, unknown> & { trace: Record<string, unknown>[] };
  restarts?: number[];
  n_trace?: number;
  error?: string;
}

const CASES = reviveNumbers<Case[]>(
  JSON.parse(
    readFileSync(fileURLToPath(new URL('./fixtures/pdhg_reference.json', import.meta.url)), 'utf8'),
  ),
);

function run(c: Case): Result {
  const { spec, fn } = getMethod<LinearProgram>(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem<LinearProgram>(c.problem), {
    ...defaults,
    ...(c.params as Record<string, never>),
  });
}

/** A Result as plain JSON with Python's snake_case Step keys (NaN → null). */
function plain(r: Result) {
  const j = JSON.parse(
    JSON.stringify(r, (_, v) => (typeof v === 'number' && Number.isNaN(v) ? null : v)),
  );
  j.trace = j.trace.map((s: Record<string, unknown>) => ({
    k: s.k,
    x: s.x,
    fun: s.fun,
    grad_norm: s.gradNorm,
    step_size: s.stepSize,
    info: s.info,
  }));
  return j;
}

describe('restarted_pdhg against Python (beyond the parity fixtures)', () => {
  it('has reference data', () => expect(CASES.length).toBeGreaterThan(60));

  CASES.forEach((c, idx) => {
    const name = `${c.problem} ${JSON.stringify(c.params)} #${idx}`;
    if (c.error !== undefined) {
      it(`${name} raises`, () => expect(() => run(c)).toThrow(c.error));
      return;
    }
    const want = c.result!;
    it(name, () => {
      const res = run(c);
      const got = plain(res);
      expect(got.message).toBe(want.message);
      expect(got.converged).toBe(want.converged);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.extra.status).toBe((want.extra as Record<string, unknown>).status);
      expect(got.trace.length).toBe(c.n_trace);
      expect(res.trace.filter((s) => s.info.restarted).map((s) => s.k)).toEqual(c.restarts);
      const n = Math.min(10, want.trace.length);
      for (let k = 0; k < n; k++)
        expect(mismatch(got.trace[k], want.trace[k], { rtol: 1e-8, atol: 1e-11 })).toBeNull();
      expect(mismatch(got.x, want.x, { rtol: 1e-6, atol: 1e-9 })).toBeNull();
      const ex = want.extra as Record<string, unknown>;
      for (const key of ['n_matvec', 'n_restarts', 'output'])
        expect(got.extra[key]).toEqual(ex[key]);
      for (const key of ['eta', 'norm_A'])
        expect(mismatch(got.extra[key], ex[key], { rtol: 1e-12, atol: 0 })).toBeNull();
    });
  });
});

describe('restarted_pdhg helpers', () => {
  const wyndor = getProblem<LinearProgram>('wyndor');

  it('npSum follows NumPy pairwise summation (8 accumulators from 8 terms on)', () => {
    const v = [1e16, 1, 1, 1, 1, 1, 1, 1, 1, -1e16];
    // Sequential: 1e16 + 1 rounds back to 1e16 every time, so the sum is 0. Pairwise: seven
    // accumulators hold a 1 each and are added in pairs first, so the ones survive the
    // cancellation (1e16 has a spacing of 2, hence 8, not 9).
    expect(v.reduce((a, b) => a + b, 0)).toBe(0);
    expect(npSum(v)).toBe(8);
    expect(npSum([1, 2, 3])).toBe(6);
    expect(npSum([])).toBe(0);
  });

  it('the Ruiz + Pock–Chambolle scalings give ‖Ã‖₂ ≤ 1 (Pock & Chambolle 2011, Lemma 2)', () => {
    for (const id of ['wyndor', 'diet_2d', 'klee_minty_3', 'transport_small', 'beale_cycling']) {
      const std = pdhgStandardForm(getProblem<LinearProgram>(id));
      const { d1, d2 } = ruizPockChambolle(std.A);
      const At = std.A.map((r, i) => r.map((v, j) => v * d1[i] * d2[j]));
      expect(singularValues(At)[0]).toBeLessThanOrEqual(1 + 1e-12);
      expect(d1.every((v) => v > 0) && d2.every((v) => v > 0)).toBe(true);
    }
  });

  it('the normalized duality gap is 0 at a KKT point and positive elsewhere', () => {
    // min −x s.t. x + s = 1, (x, s) ≥ 0: optimum z = (1, 0), y = −1, so c − Aᵀy = (0, 1) ≥ 0.
    const b = [1],
      c = [-1, 0];
    const at = (z: number[], y: number[], r: number) =>
      normalizedDualityGap(z, [z[0] + z[1]], [y[0], y[0]], b, c, r, 1);
    expect(at([1, 0], [-1], 0.5)).toBeCloseTo(0, 14);
    expect(at([0.5, 0.5], [0], 0.5)).toBeGreaterThan(0);
    // ρ_r is non-increasing in r for a fixed point (2023 paper, Fact 1).
    const r1 = at([0.2, 0.1], [0.3], 0.1),
      r2 = at([0.2, 0.1], [0.3], 1.0);
    expect(r2).toBeLessThanOrEqual(r1 + 1e-15);
    expect(() => at([1, 0], [0], -1)).toThrow('radius must be ≥ 0');
  });

  it('weightedNorm and primalWeightUpdate follow PDLP', () => {
    expect(weightedNorm([3], [4], 1)).toBe(5);
    expect(weightedNorm([1], [2], 4)).toBeCloseTo(Math.sqrt(4 + 1), 15);
    // θ = ½: ω⁺ = √(Δy/Δz · ω).
    expect(primalWeightUpdate(1, 4, 1)).toBeCloseTo(2, 15);
    expect(primalWeightUpdate(0, 4, 0.7)).toBe(0.7);
    expect(primalWeightUpdate(Infinity, 4, 0.7)).toBe(0.7);
  });

  it('keeps z ≥ 0 at every iterate and costs 2 + 2k products', () => {
    const r = restartedPdhg(wyndor);
    for (const s of r.trace) {
      expect((s.info.x_pdhg as number[]).every((v) => v >= 0)).toBe(true);
      expect(s.info.matvecs).toBe(2 + 2 * s.k);
    }
    expect(r.extra.n_matvec).toBe(2 + 2 * r.nIter);
  });

  it('at a restart, x is the average of the epoch', () => {
    const r = restartedPdhg(wyndor);
    const restarts = r.trace.filter((s) => s.info.restarted);
    expect(restarts.length).toBeGreaterThan(0);
    for (const s of restarts) expect(s.info.x).toEqual(s.info.x_avg);
  });

  it('rejects an LP without a nonzero constraint row', () => {
    const lp: LinearProgram = {
      ...wyndor,
      id: 'empty',
      aUb: null,
      bUb: null,
      aEq: [[0, 0]],
      bEq: [0],
    };
    expect(() => restartedPdhg(lp)).toThrow(
      'empty: restarted_pdhg needs at least one nonzero constraint row',
    );
  });
});
