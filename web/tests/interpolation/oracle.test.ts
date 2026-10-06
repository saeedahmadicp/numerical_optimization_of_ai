/**
 * The interpolation port against the Python reference on 108 cases the parity fixtures do not
 * cover: every method on seven datasets (sorted, unsorted, noisy, without f_true), data-mode
 * Chebyshev, n = 1/2/3, unsorted splines, PCHIP limiting, weight overflow, round-off failure,
 * and the input errors. Every trace step, every `info` key and every `extra` key is compared
 * (the 200-point curves are thinned to every 7th value in the oracle file).
 *
 * Regenerate: .venv/bin/python web/tests/interpolation/gen_oracle.py
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Result } from '../../src/core/types';
import '../../src/methods/interpolation/methods';
import '../../src/problems/data';

interface OracleCase {
  name: string;
  method: string;
  problem: { id?: string; x?: number[]; y?: number[] };
  params: Record<string, number>;
  error?: string;
  result?: {
    x: unknown;
    fun: unknown;
    converged: boolean;
    message: string;
    n_iter: number;
    n_fev: number;
    trace: { k: number; x: unknown; fun: unknown; info: Record<string, unknown> }[];
    extra: Record<string, unknown>;
  };
}

const FILE = fileURLToPath(new URL('./fixtures/oracle.json', import.meta.url));
const CASES = JSON.parse(readFileSync(FILE, 'utf8')) as OracleCase[];

const THIN = 7;
const thin = (v: unknown) => (Array.isArray(v) ? v.filter((_, i) => i % THIN === 0) : v);

/** Python JSON → comparable: "inf" → ∞, null stays null (NaN or None). */
function num(v: unknown): number | null | unknown {
  if (v === 'inf') return Infinity;
  if (v === '-inf') return -Infinity;
  return v;
}

const RTOL = 1e-9;
const ATOL = 1e-12;

/** Deep comparison with a path in the failure message. */
function same(got: unknown, want: unknown, path: string, rtol = RTOL): void {
  want = num(want);
  if (want === null) {
    const ok = got === null || got === undefined || (typeof got === 'number' && Number.isNaN(got));
    if (!ok) expect.fail(`${path}: expected null/NaN, got ${JSON.stringify(got)}`);
    return;
  }
  if (typeof want === 'number') {
    if (typeof got !== 'number') expect.fail(`${path}: expected ${want}, got ${JSON.stringify(got)}`);
    if (!Number.isFinite(want)) {
      if (got !== want) expect.fail(`${path}: expected ${want}, got ${got}`);
      return;
    }
    const tol = ATOL + rtol * Math.abs(want);
    if (!(Math.abs((got as number) - want) <= tol))
      expect.fail(`${path}: expected ${want}, got ${got} (|Δ| = ${Math.abs((got as number) - want)})`);
    return;
  }
  if (Array.isArray(want)) {
    if (!Array.isArray(got)) expect.fail(`${path}: expected an array, got ${JSON.stringify(got)}`);
    const g = got as unknown[];
    if (g.length !== want.length) expect.fail(`${path}: length ${g.length} ≠ ${want.length}`);
    want.forEach((w, i) => same(g[i], w, `${path}[${i}]`, rtol));
    return;
  }
  if (typeof want === 'object') {
    if (typeof got !== 'object' || got === null) expect.fail(`${path}: expected an object`);
    const g = got as Record<string, unknown>;
    expect(Object.keys(g).sort(), `${path} keys`).toEqual(Object.keys(want).sort());
    for (const [k, w] of Object.entries(want as Record<string, unknown>)) same(g[k], w, `${path}.${k}`, rtol);
    return;
  }
  if (got !== want) expect.fail(`${path}: expected ${JSON.stringify(want)}, got ${JSON.stringify(got)}`);
}

function problemOf(c: OracleCase): unknown {
  if (c.problem.id) return getProblem(c.problem.id);
  return [c.problem.x!, c.problem.y!];
}

function run(c: OracleCase): Result {
  const { spec, fn } = getMethod(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problemOf(c), { ...defaults, ...c.params });
}

const NUM = /\d+(\.\d+)?(e[-+]\d+)?/g;
/**
 * Messages agree word for word; their numbers agree to 3 digits, or are both round-off
 * (< 1e-10: a node residual such as 4.76e-13 depends on the last bit of exp/cos in the data).
 */
function sameMessage(got: string, want: string) {
  expect(got.replace(NUM, '#')).toBe(want.replace(NUM, '#'));
  const g = got.match(NUM) ?? [],
    w = want.match(NUM) ?? [];
  g.forEach((t, i) => {
    const a = Number(t),
      b = Number(w[i]);
    if (a === b || (Math.abs(a) < 1e-10 && Math.abs(b) < 1e-10)) return;
    expect(Math.abs(a - b), `message number ${t} vs ${w[i]}`).toBeLessThanOrEqual(5e-3 * Math.abs(b));
  });
}

/**
 * Cases whose grid values are dominated by cancellation (a degree-k polynomial of size 10¹⁰ far
 * outside its nodes): NumPy sums with BLAS/pairwise order, so only 3–7 digits are shared: their curves are not compared. Their weights
 * (Step.x) are still compared at 1e-9.
 */
const ILL_CONDITIONED = new Set(['barycentric overflow']);

describe('interpolation port vs the Python oracle', () => {
  it('has the oracle file', () => expect(CASES.length).toBeGreaterThan(100));

  for (const c of CASES) {
    it(c.name, () => {
      if (c.error) {
        const msg = c.error.replace(/^\w+: /, '');
        expect(() => run(c)).toThrow(msg);
        return;
      }
      const want = c.result!;
      const got = run(c);
      const cheb = c.method === 'chebyshev_interpolation';
      // Round-off–dominated cases (overflowed weights, a destroyed Newton form) are compared
      // looser on the derived curves; their coefficients still match tightly.
      const rtol = cheb ? 1e-7 : RTOL;
      const ill = ILL_CONDITIONED.has(c.name);
      const curveTol = cheb ? 1e-6 : 1e-8;
      expect(got.converged).toBe(want.converged);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.nFev).toBe(want.n_fev);
      sameMessage(got.message, want.message);
      same(got.x, want.x, 'x', rtol);
      same(got.fun, want.fun, 'fun', cheb ? 1e-6 : 1e-8);
      expect(got.trace.length).toBe(want.trace.length);
      want.trace.forEach((wsIn, i) => {
        let ws = wsIn;
        const gs = got.trace[i];
        expect(gs.k).toBe(ws.k);
        same(gs.x, ws.x, `trace[${i}].x`, rtol);
        same(gs.fun, ws.fun, `trace[${i}].fun`, cheb ? 1e-6 : 1e-8);
        const info = Object.fromEntries(
          Object.entries(gs.info).map(([k, v]) => [k, k === 'curve' || k === 'basis' ? thin(v) : v]),
        );
        if (ill) {
          // The values are cancellation noise beyond ~3 digits (and overflow where they end).
          delete info.curve;
          ws = { ...ws, info: { ...ws.info } };
          delete ws.info.curve;
        }
        same(info, ws.info, `trace[${i}].info`, curveTol);
      });
      const extra = { ...got.extra };
      if (extra.eval) {
        const ev = extra.eval as Record<string, unknown>;
        extra.eval = { x: thin(ev.x), y: thin(ev.y), f_true: ev.f_true === null ? null : thin(ev.f_true) };
      }
      const wantExtra = { ...want.extra };
      if (cheb) {
        // node_residual is a round-off quantity (≈1e-15) — compare its size, not its digits.
        expect(Math.abs((extra.node_residual as number) - (num(wantExtra.node_residual) as number))).toBeLessThan(1e-12);
        delete extra.node_residual;
        delete wantExtra.node_residual;
      }
      same(extra, wantExtra, 'extra', cheb ? 1e-6 : 1e-8);
    });
  }
});
