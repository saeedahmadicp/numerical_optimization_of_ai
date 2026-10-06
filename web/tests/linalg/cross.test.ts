/**
 * Cross-check of the linalg ports beyond the parity fixtures: every linalg method on every linalg
 * problem (plus SOR ω variants, GMRES restarts and 2-D start points), from
 * tests/linalg/fixtures/linalg_cross.json (gen_linalg_cross.py runs the Python reference).
 *
 * The ports reproduce NumPy's floating-point kernels (numerics.ts), so the comparison is tight:
 * iterates and residuals within 1e-12 (relative to the size of the trace), the same iteration
 * count, convergence flag, message and Step.info keys.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result, Step } from '../../src/core/types';
import '../../src/methods/linalg/direct';
import '../../src/methods/linalg/iterative';
import '../../src/problems/linalg';

interface RawStep {
  k: number;
  x: unknown;
  fun: number | null;
  step_size: number | null;
  grad_norm: number | null;
  info?: Record<string, unknown>;
}
interface Case {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  error?: string;
  result?: {
    x: unknown;
    fun: number | null;
    converged: boolean;
    message: string;
    n_iter: number;
    extra: Record<string, unknown>;
    trace: RawStep[];
  };
}

const FILE = fileURLToPath(new URL('./fixtures/linalg_cross.json', import.meta.url));
const CASES = reviveNumbers<Case[]>(JSON.parse(readFileSync(FILE, 'utf8')));

function flat(v: unknown): number[] {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return v.flatMap(flat);
  return [];
}

/** Max |a − b| / (1 + max|b|) over two nested arrays (Infinity on a shape mismatch). */
function dist(a: unknown, b: unknown): number {
  const xa = flat(a),
    xb = flat(b);
  if (xa.length !== xb.length) return Infinity;
  const s = 1 + Math.max(0, ...xb.filter(Number.isFinite).map(Math.abs));
  let m = 0;
  xa.forEach((v, i) => {
    const w = xb[i];
    if (!Number.isFinite(v) || !Number.isFinite(w)) {
      if (!(Object.is(v, w) || (Number.isNaN(v) && Number.isNaN(w)))) m = Infinity;
      return;
    }
    m = Math.max(m, Math.abs(v - w) / s);
  });
  return m;
}

function run(c: Case): Result {
  const { spec, fn } = getMethod(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(c.problem), { ...defaults, ...(c.params as Record<string, never>) });
}

/** TS trace thinned exactly like the generator: first 40 and last 3 steps. */
function thin(trace: Step[]): Step[] {
  const last = trace.length - 1;
  return trace.filter((_, i) => i < 40 || i >= last - 2);
}

const TOL = 1e-12;
/** Relative agreement of eigenvalue-based quantities: O(κ·ε), at least 1e-9. */
const eigTol = (v: unknown) => Math.max(1e-9, (typeof v === 'number' ? Math.abs(v) : 1) * 4e-15);

describe('linalg ports vs Python on every method × problem', () => {
  it('has the cross-check fixture', () => expect(CASES.length).toBeGreaterThan(200));

  for (const c of CASES) {
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)}`;
    it(name, () => {
      if (c.error !== undefined) {
        expect(() => run(c)).toThrow();
        return;
      }
      const want = c.result!;
      const got = run(c);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.converged).toBe(want.converged);
      expect(got.message).toBe(want.message);
      expect(dist(got.x, want.x)).toBeLessThanOrEqual(TOL);
      if (want.fun === null) expect(got.fun).toBeNull();
      else expect(dist(got.fun, want.fun)).toBeLessThanOrEqual(TOL);

      const steps = thin(got.trace);
      expect(steps.length).toBe(want.trace.length);
      want.trace.forEach((w, i) => {
        const g = steps[i];
        expect(g.k).toBe(w.k);
        expect(dist(g.x, w.x), `x at k = ${w.k}`).toBeLessThanOrEqual(TOL);
        expect(dist(g.fun, w.fun), `fun at k = ${w.k}`).toBeLessThanOrEqual(TOL);
        expect(dist(g.stepSize, w.step_size), `step at k = ${w.k}`).toBeLessThanOrEqual(TOL);
        expect(dist(g.gradNorm, w.grad_norm), `grad at k = ${w.k}`).toBeLessThanOrEqual(TOL);
        if (!w.info) return;
        expect(Object.keys(g.info).sort(), `info keys at k = ${w.k}`).toEqual(
          Object.keys(w.info).sort(),
        );
        for (const [key, value] of Object.entries(w.info)) {
          const mine = g.info[key];
          if (typeof value === 'string' || typeof value === 'boolean' || value === null)
            expect(mine, `info.${key} at k = ${w.k}`).toEqual(value);
          else {
            // Eigenvalue-based quantities come from a different (backward stable) algorithm, so
            // they agree to O(κ·ε) only.
            const tol = /condition|rate|spectral/.test(key) ? eigTol(value) : TOL;
            expect(dist(mine, value), `info.${key} at k = ${w.k}`).toBeLessThanOrEqual(tol);
          }
        }
      });

      for (const [key, value] of Object.entries(want.extra)) {
        const mine = got.extra[key];
        expect(mine === undefined, `extra.${key} missing`).toBe(false);
        if (typeof value === 'string' || value === null) expect(mine).toEqual(value);
        else {
          const tol = /condition|rate|spectral|omega_opt|cond_estimate/.test(key)
            ? eigTol(value)
            : TOL;
          expect(dist(mine, value), `extra.${key}`).toBeLessThanOrEqual(tol);
        }
      }
    });
  }
});
