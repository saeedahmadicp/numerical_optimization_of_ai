/**
 * The regression port against a Python oracle (tests/regression/gen_oracle.py) on cases the
 * parity fixtures do not cover: every method on every regression dataset, the degree sweep on
 * the interpolation datasets, rank-deficient designs (minimum-norm solutions), large x offsets,
 * exact fits, invalid input, messages, info payloads and the `extra` statistics.
 */
import { describe, expect, it } from 'vitest';
import { CANONICAL } from '../fixtures/platform';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import { defaults } from '../../src/core/registry';
import type { Result, Step } from '../../src/core/types';
import '../../src/methods/regression/methods';
import '../../src/problems/data';

interface OracleStep {
  k: number;
  x: (number | null)[];
  fun: number | null;
  step_size: number | null;
  info: Record<string, unknown>;
}
interface OracleCase {
  name: string;
  method: string;
  problem: string | null;
  data: [number[], number[]] | null;
  params: Record<string, number | string>;
  error?: string;
  result?: {
    x: (number | null)[];
    fun: number | null;
    converged: boolean;
    message: string;
    n_iter: number;
    n_fev: number;
    trace_len: number;
    trace: OracleStep[];
    extra: Record<string, unknown>;
  };
}

const FILE = fileURLToPath(new URL('./fixtures/oracle.json', import.meta.url));
const CASES = reviveNumbers<OracleCase[]>(JSON.parse(readFileSync(FILE, 'utf8')));

/** Cases whose coefficients are, by design, not accurate (the point of the case). */
const INACCURATE = new Set(['ols ne offset 1e8']);

function run(c: OracleCase): Result {
  const { spec, fn } = getMethod(c.method);
  // The oracle writes NaN as null.
  const nan = (v: (number | null)[]) => v.map((t) => (t === null ? NaN : t));
  const problem = c.problem ? getProblem(c.problem) : ([nan(c.data![0]), nan(c.data![1])] as const);
  return fn(problem, { ...defaults(spec), ...c.params });
}

function close(a: unknown, b: unknown, rtol: number, atol: number, path = '') {
  if (b === null || b === undefined) {
    // Python None or NaN.
    if (typeof a === 'number') expect(Number.isNaN(a), `${path}: ${a} vs NaN/None`).toBe(true);
    else expect(a ?? null, path).toBeNull();
    return;
  }
  if (typeof b === 'number') {
    expect(typeof a, path).toBe('number');
    const x = a as number;
    if (!Number.isFinite(b)) return expect(x, path).toBe(b);
    expect(Math.abs(x - b), `${path}: ${x} vs ${b}`).toBeLessThanOrEqual(atol + rtol * Math.abs(b));
    return;
  }
  if (Array.isArray(b)) {
    expect(Array.isArray(a), path).toBe(true);
    expect((a as unknown[]).length, `${path}.length`).toBe(b.length);
    b.forEach((v, i) => close((a as unknown[])[i], v, rtol, atol, `${path}[${i}]`));
    return;
  }
  if (typeof b === 'object') {
    const ob = b as Record<string, unknown>;
    for (const k of Object.keys(ob))
      close((a as Record<string, unknown>)[k], ob[k], rtol, atol, `${path}.${k}`);
    return;
  }
  expect(a, path).toEqual(b);
}

/** Scale-aware tolerance for a vector of coefficients. */
const coefTol = (v: readonly (number | null)[]) =>
  1e-9 * Math.max(1, ...v.map((t) => (t === null ? 0 : Math.abs(t))));

function sub13(v: readonly number[]): number[] {
  return v.filter((_, i) => i % 13 === 0);
}

describe('regression port vs the Python oracle', () => {
  it('has the oracle', () => expect(CASES.length).toBeGreaterThan(100));

  for (const c of CASES) {
    it(c.name, () => {
      if (c.error !== undefined) {
        expect(() => run(c)).toThrow(c.error);
        return;
      }
      const want = c.result!;
      const got = run(c);
      expect(got.converged).toBe(want.converged);
      expect(got.nIter).toBe(want.n_iter);
      expect(got.nFev).toBe(0);
      expect(got.trace.length).toBe(want.trace_len);
      // The final ‖Δθ‖∞ of a converged IRLS run is rounding noise (~1e-12): mask its digits.
      const mask = (m: string) => m.replace(/(‖Δθ‖∞ = )[^ ]+/, '$1…');
      expect(mask(got.message)).toBe(mask(want.message));
      if (INACCURATE.has(c.name)) return;

      const xs = got.x as number[];
      // Rank-deficient minimum-norm solutions and high-degree fits are sensitive to rounding.
      const loose = !want.converged || /d=1[0-5]|d=12/.test(c.name);
      // x ≈ 10⁸: β₀ = ȳ − β₁x̄ cancels |β₁x̄| ≈ 5×10⁷, so ~10⁻⁸ absolute is the rounding level.
      const offset = /offset/.test(c.name);
      const rtol = loose ? 1e-6 : 1e-9;
      // Off the canonical platform (tests/fixtures/platform.ts) the dump's QR rounds differently:
      // there β agrees to its forward-error bound 16·κ₂(X)·ε·‖β‖ (κ₂ = 5.8e7 at offset 1e8; the
      // two solutions are 0.12 apart in β₀ ≈ −4.8e7).
      const cond = want.trace[0]?.info.cond;
      const offsetAtol =
        !CANONICAL && typeof cond === 'number'
          ? Math.max(1e-7, 16 * cond * 2 ** -52 * Math.max(...xs.map(Math.abs)))
          : 1e-7;
      const xAtol = offset
        ? offsetAtol
        : loose
          ? 1e-6 * Math.max(1, ...xs.map(Math.abs))
          : coefTol(want.x);
      // Objectives and residual sums near 0 (interpolating fits) are rounding noise in absolute terms.
      const fAtol = offset || loose ? 1e-6 : 1e-12;
      close(xs, want.x, rtol, xAtol);
      close(got.fun, want.fun, offset ? 1e-6 : rtol, fAtol);

      // Trace: the first 12 steps and the last one.
      const steps: Step[] =
        got.trace.length <= 13
          ? got.trace
          : [...got.trace.slice(0, 12), got.trace[got.trace.length - 1]];
      steps.forEach((s, i) => {
        const w = want.trace[i];
        expect(s.k).toBe(w.k);
        close(s.x, w.x, rtol, offset ? offsetAtol : loose ? 1e-6 : coefTol(w.x), `trace[${i}].x`);
        close(s.fun, w.fun, offset ? 1e-6 : rtol, fAtol, `trace[${i}].fun`);
        if (w.step_size !== null)
          close(s.stepSize, w.step_size, 1e-6, offset ? 1e-7 : 1e-10, `trace[${i}].step_size`);
        else expect(s.stepSize).toBeNull();
        expect(Object.keys(s.info).sort()).toEqual(Object.keys(w.info).sort());
        for (const [key, v] of Object.entries(w.info)) {
          if (key === 'cond' || key === 'cond_gram') {
            // κ₂ of a rank-deficient matrix is σ_max/(rounding noise): only "huge" is meaningful.
            if (want.converged) close(s.info[key], v, 1e-6, 0, `info.${key}`);
            else expect(s.info[key] as number).toBeGreaterThan(1e14);
          } else if (key === 'weights') {
            // 1/max(|r|, ε): huge where a residual is ~0, so compare relatively.
            close(s.info[key], v, 1e-5, 1e-9, `info.${key}`);
          } else close(s.info[key], v, offset ? 1e-6 : 1e-9, offset ? 1e-7 : 1e-12, `info.${key}`);
        }
      });

      // Extra statistics.
      const e = got.extra;
      const we = want.extra;
      expect(Object.keys(e).sort()).toEqual(Object.keys(we).sort());
      if (Object.keys(we).length === 0) return;
      const sAtol = offset || loose ? 1e-6 : 1e-12;
      for (const key of ['degree', 'rank', 'solver', 'n_pairs', 'reference']) {
        if (key in we) expect(e[key]).toEqual(we[key]);
      }
      for (const key of [
        'rss',
        'tss',
        'r_squared',
        'adj_r_squared',
        'rmse',
        'sigma',
        'std_errors',
        'objective',
        'scale',
        'level',
        'max_deviation',
        'x_center',
        'lambda',
        'delta',
        'eps',
      ]) {
        if (key in we) close(e[key], we[key], loose || offset ? 1e-6 : 1e-8, sAtol, `extra.${key}`);
      }
      if ('cond' in we) {
        if (want.converged) close(e.cond, we.cond, 1e-6, 0, 'extra.cond');
        else expect(e.cond as number).toBeGreaterThan(1e14);
      }
      for (const key of ['fitted', 'residuals', 'coefficients']) {
        const v = we[key] as number[];
        const atol = offset ? 1e-7 : loose ? 1e-6 : 1e-9 * Math.max(1, ...v.map(Math.abs));
        close(e[key], v, loose || offset ? 1e-6 : 1e-8, atol, `extra.${key}`);
      }
      if ('weights' in we) close(e.weights, we.weights, 1e-5, 1e-9, 'extra.weights');
      if ('slopes' in we) close(e.slopes, we.slopes, 1e-12, 1e-12, 'extra.slopes');
      const ev = e.eval as { x: number[]; y: number[] };
      const wev = we.eval as { x: number[]; y: number[] };
      close(sub13(ev.x), wev.x, 1e-14, 1e-14, 'extra.eval.x');
      const yScale = Math.max(1, ...wev.y.map(Math.abs));
      close(
        sub13(ev.y),
        wev.y,
        loose ? 1e-6 : 1e-8,
        (loose || offset ? 1e-6 : 1e-9) * yScale,
        'extra.eval.y',
      );
    });
  }
});
