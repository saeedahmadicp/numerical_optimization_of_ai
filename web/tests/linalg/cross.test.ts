/**
 * Cross-check of the linalg ports beyond the parity fixtures: every linalg method on every linalg
 * problem (plus SOR ω variants, GMRES restarts and 2-D start points), from
 * tests/linalg/fixtures/linalg_cross.json (gen_linalg_cross.py runs the Python reference).
 *
 * The ports reproduce NumPy's floating-point kernels (numerics.ts), so on the canonical platform
 * (tests/fixtures/platform.ts) the comparison is tight: iterates and residuals within 1e-12
 * (relative to the size of the trace), the same iteration count, convergence flag, message and
 * Step.info keys.
 *
 * On another platform the Python reference rounds differently, and two backward-stable solves of
 * Ax = b differ by up to twice their forward-error bound c·n·κ(A)·ε. The floats are then compared
 * within τ = max(1e-12, 16·n·κ∞(A)·ε) (measured between platforms: 1.5e-11 on hilbert_5,
 * τ = 1.7e-8; 3.4e-7 on nearly_singular, τ = 3e-5), the messages with their rounded numerals
 * masked, the counts exactly. A run driven by rounding is compared by its outcome only (the
 * convergence flag, the message text, the final x within 2·κ∞(A)·tol when it converged): a
 * Krylov run (CG, PCG, GMRES) with more than n iterations, since exact arithmetic ends it by
 * step n (restarts and lost orthogonality decide the rest), and the two runs in ROUNDING_DRIVEN.
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
import { CANONICAL, sameTemplate, sameText } from '../fixtures/platform';

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
const EPS = 2 ** -52;
const KRYLOV = new Set(['conjugate_gradient_linear', 'preconditioned_cg', 'gmres']);
/**
 * Runs whose path differs between platforms although they are not Krylov runs beyond step n
 * (measured on x86-64, generic ARMv8 and Neoverse-N1 OpenBLAS kernels, NumPy 2.4): steepest
 * descent stops after 518 or 532..533 iterations, and SOR with ω = 1.9 overflows to inf at
 * another step.
 */
const ROUNDING_DRIVEN = new Set([
  'steepest_descent_linear on poisson_1d_10 {}',
  'sor on jacobi_diverges {"omega":1.9}',
]);

/** κ∞(A) = ‖A‖∞·‖A⁻¹‖∞ by Gauss–Jordan with partial pivoting; Infinity when A is singular. */
function condInf(A: readonly (readonly number[])[]): number {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) <= 1e-300) return Infinity;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  const normInf = (rows: number[][]) =>
    Math.max(...rows.map((r) => r.reduce((t, v) => t + Math.abs(v), 0)));
  const kappa = normInf(A.map((r) => [...r])) * normInf(M.map((r) => r.slice(n)));
  return Number.isFinite(kappa) && kappa < 1 / EPS ? kappa : Infinity;
}
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
      const { A } = getProblem(c.problem) as unknown as { A: number[][] };
      const kappa = condInf(A);
      const n = A.length;
      if (!CANONICAL && (ROUNDING_DRIVEN.has(name) || (KRYLOV.has(c.method) && want.n_iter > n))) {
        // Outcome only (module comment).
        expect(got.converged).toBe(want.converged);
        expect(sameTemplate(got.message, want.message), `${got.message} vs ${want.message}`).toBe(
          true,
        );
        if (want.converged) {
          const { spec } = getMethod(c.method);
          const tol = Number(
            c.params.tol ?? spec.params.find((q) => q.name === 'tol')?.default ?? 0,
          );
          expect(dist(got.x, want.x)).toBeLessThanOrEqual(Math.max(TOL, 2 * kappa * tol));
        }
        return;
      }
      const tolX = CANONICAL ? TOL : Math.max(TOL, 16 * n * kappa * EPS);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.converged).toBe(want.converged);
      if (CANONICAL) expect(got.message).toBe(want.message);
      else
        expect(sameText(got.message, want.message), `${got.message} vs ${want.message}`).toBe(true);
      expect(dist(got.x, want.x)).toBeLessThanOrEqual(tolX);
      if (want.fun === null) expect(got.fun).toBeNull();
      else expect(dist(got.fun, want.fun)).toBeLessThanOrEqual(tolX);

      const steps = thin(got.trace);
      expect(steps.length).toBe(want.trace.length);
      want.trace.forEach((w, i) => {
        const g = steps[i];
        expect(g.k).toBe(w.k);
        expect(dist(g.x, w.x), `x at k = ${w.k}`).toBeLessThanOrEqual(tolX);
        expect(dist(g.fun, w.fun), `fun at k = ${w.k}`).toBeLessThanOrEqual(tolX);
        expect(dist(g.stepSize, w.step_size), `step at k = ${w.k}`).toBeLessThanOrEqual(tolX);
        expect(dist(g.gradNorm, w.grad_norm), `grad at k = ${w.k}`).toBeLessThanOrEqual(tolX);
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
            const tol = /condition|rate|spectral/.test(key) ? Math.max(eigTol(value), tolX) : tolX;
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
            ? Math.max(eigTol(value), tolX)
            : tolX;
          expect(dist(mine, value), `extra.${key}`).toBeLessThanOrEqual(tol);
        }
      }
    });
  }
});
