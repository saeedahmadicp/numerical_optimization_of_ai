/**
 * The LP ports against Python on the cases the parity fixtures do not cover
 * (tests/lp/fixtures/lp_reference.json, from tests/lp/gen_lp_reference.py): every pivot rule on
 * every library LP, cycling, phase 1 with equality rows, unbounded and infeasible programs,
 * budgets, the interior-point feasibility checks, the integer methods, and the ValueErrors.
 *
 * Vertex methods (simplex family, branch and bound, Gomory) are compared in full — trace, every
 * info key, messages, extra — at 1e-9 relative. Interior-point methods are compared like the
 * parity harness (first ten steps with every info key at 1e-8, the iteration count, status,
 * message and final x at 1e-6), because their late iterates amplify rounding differences.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resultFromJson, reviveNumbers } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { LinearProgram, Result } from '../../src/core/types';
import { mismatch } from '../shared-ports/compare';
import '../../src/methods/lp/simplex';
import '../../src/methods/lp/interior_point';
import '../../src/methods/lp/integer';
import '../../src/problems/lp';

interface Case {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
}

const CASES = reviveNumbers<Case[]>(
  JSON.parse(
    readFileSync(fileURLToPath(new URL('./fixtures/lp_reference.json', import.meta.url)), 'utf8'),
  ),
);

const VERTEX = new Set([
  'simplex',
  'two_phase_simplex',
  'big_m',
  'dual_simplex',
  'revised_simplex',
  'branch_and_bound',
  'gomory_cuts',
]);

function run(c: Case): Result {
  const { spec, fn } = getMethod<LinearProgram>(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem<LinearProgram>(c.problem), {
    ...defaults,
    ...(c.params as Record<string, never>),
  });
}

/** A Result as plain JSON (NaN → null, like Python's export), so `mismatch` sees the same shapes. */
const plain = (r: Result) =>
  JSON.parse(JSON.stringify(r, (_, v) => (typeof v === 'number' && Number.isNaN(v) ? null : v)));

describe('LP methods against Python (beyond the parity fixtures)', () => {
  it('has reference data', () => expect(CASES.length).toBeGreaterThan(150));

  CASES.forEach((c, idx) => {
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${idx}`;
    if (c.error !== undefined) {
      it(`${name} raises`, () => {
        expect(() => run(c)).toThrow(c.error);
      });
      return;
    }
    const want = resultFromJson(c.result as Record<string, unknown>);
    it(name, () => {
      const got = plain(run(c));
      const ref = JSON.parse(JSON.stringify(want));
      expect(got.message).toBe(ref.message);
      expect(got.converged).toBe(ref.converged);
      expect(got.nIter).toBe(ref.nIter);
      expect(got.extra.status).toBe(ref.extra.status);
      if (VERTEX.has(c.method)) {
        expect(mismatch(got, ref, { rtol: 1e-9, atol: 1e-11 })).toBeNull();
        return;
      }
      expect(got.trace.length).toBe(ref.trace.length);
      const n = Math.min(10, ref.trace.length);
      for (let k = 0; k < n; k++) {
        // Residuals ‖A z − b‖ are cancellations of terms of size ‖x‖: on an unbounded LP the
        // iterates reach 10⁶ and the residual is rounding noise (also near a converged iterate,
        // where it is ~10⁻⁹ with terms of size ‖b‖), so its absolute tolerance scales with x.
        const RES = ['primal_residual', 'dual_residual'];
        const strip = (s: { info: Record<string, unknown> }) => ({
          ...s,
          info: Object.fromEntries(Object.entries(s.info).filter(([key]) => !RES.includes(key))),
        });
        expect(
          mismatch(strip(got.trace[k]), strip(ref.trace[k]), { rtol: 1e-8, atol: 1e-10 }),
        ).toBeNull();
        const xs = (ref.trace[k].x as number[] | null) ?? [];
        const scale = 1 + Math.max(0, ...xs.map(Math.abs));
        for (const key of RES)
          expect(
            mismatch(got.trace[k].info[key], ref.trace[k].info[key], {
              rtol: 1e-8,
              atol: 1e-11 * scale,
            }),
          ).toBeNull();
      }
      expect(mismatch(got.x, ref.x, { rtol: 1e-6, atol: 1e-9 })).toBeNull();
      expect(got.extra.feasibility_check ?? null).toBe(ref.extra.feasibility_check ?? null);
    });
  });
});
