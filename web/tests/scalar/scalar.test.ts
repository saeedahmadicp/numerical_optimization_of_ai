/**
 * The 1-D minimization port (src/methods/scalar/methods.ts, src/problems/scalar_min.ts) against
 * Python, beyond the eight parity fixtures:
 *   - the registered specs equal src/generated/registry.json (ids, params, texts, references);
 *   - the problems equal src/generated/problems.json and their f, f′, f″ equal Python's values;
 *   - every method on every problem (defaults) and 32 variations / failure paths reproduce the
 *     WHOLE Python result: every step and every info key, counts, `converged` and the message;
 *   - invalid inputs throw where Python raises ValueError.
 * Reference data: tests/scalar/fixtures/scalar_python.json (gen_scalar_fixture.py).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers, resultFromJson } from '../../src/core/json';
import { getMethod, listMethods } from '../../src/core/registry';
import type { Result } from '../../src/core/types';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { ScalarProblem } from '../../src/problems/scalar_min';
import { fibonacciNumbers, fmtG, nextUp, parabolaThrough } from '../../src/methods/scalar/methods';
import '../../src/methods/scalar/methods';
import '../../src/problems/scalar_min';

const read = <T>(rel: string): T =>
  JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')) as T;

type Raw = Record<string, unknown>;
const FIX = reviveNumbers<{
  problems: {
    id: string;
    values: { x: number; f: number | null; grad: number | null; hess: number | null }[];
  }[];
  runs: { method: string; problem: string; params: Raw; result: Raw }[];
  errors: { method: string; problem: string; params: Raw; error: string }[];
  fibonacci: { target: number; n_max: number | null; fib: number[] }[];
}>(read('./fixtures/scalar_python.json'));
const REGISTRY = read<Raw[]>('../../src/generated/registry.json').filter(
  (m) => m.family === 'scalar',
);
const PROBLEMS = read<Raw[]>('../../src/generated/problems.json').filter(
  (p) => p.kind === 'scalar_min',
);

/** |a − b| ≤ atol + rtol |b|; null (Python NaN/None) matches NaN or null. */
function near(a: unknown, b: unknown, rtol: number, atol = 0, path = ''): void {
  if (b === null || (typeof b === 'number' && Number.isNaN(b))) {
    expect(
      a === null || a === undefined || Number.isNaN(a as number),
      `${path}: ${String(a)} vs NaN/None`,
    ).toBe(true);
    return;
  }
  if (typeof b === 'number') {
    expect(typeof a, path).toBe('number');
    if (!Number.isFinite(b)) return void expect(a, path).toBe(b);
    expect(Math.abs((a as number) - b), `${path}: ${String(a)} vs ${b}`).toBeLessThanOrEqual(
      atol + rtol * Math.abs(b),
    );
    return;
  }
  if (Array.isArray(b)) {
    expect(Array.isArray(a), path).toBe(true);
    expect((a as unknown[]).length, path).toBe(b.length);
    b.forEach((v, i) => near((a as unknown[])[i], v, rtol, atol, `${path}[${i}]`));
    return;
  }
  if (typeof b === 'object') {
    expect(a !== null && typeof a === 'object', path).toBe(true);
    expect(Object.keys(a as Raw).sort(), path).toEqual(Object.keys(b as Raw).sort());
    for (const [k, v] of Object.entries(b as Raw))
      near((a as Raw)[k], v, rtol, atol, `${path}.${k}`);
    return;
  }
  expect(a, path).toEqual(b);
}

function run(method: string, problem: string, params: Raw): Result {
  const { spec, fn } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(getProblem(problem), { ...defaults, ...(params as Record<string, never>) });
}

describe('scalar registry', () => {
  it('registers the Python methods with the same specs', () => {
    const ts = listMethods('scalar');
    expect(ts.map((m) => m.spec.id).sort()).toEqual(REGISTRY.map((m) => m.id as string).sort());
    for (const py of REGISTRY) {
      const { spec } = getMethod(py.id as string);
      expect(spec.name).toBe(py.name);
      expect(spec.needs).toEqual(py.needs);
      expect(spec.order).toBe(py.order);
      expect(spec.summary).toBe(py.summary);
      expect(spec.references).toEqual(py.references);
      expect(spec.deterministic).toBe(py.deterministic ?? true);
      const params = spec.params.map(
        ({ name, default: d, kind, min, max, choices, log, help }) => ({
          name,
          default: d,
          kind,
          min,
          max,
          choices,
          log,
          help,
        }),
      );
      expect(params).toEqual(py.params);
    }
  });
});

describe('scalar_min problems', () => {
  const problems = listProblems<ScalarProblem>('scalar_min');
  it('match problems.json (ids, order, metadata)', () => {
    expect(problems.map((p) => p.id)).toEqual(PROBLEMS.map((p) => p.id));
    for (const meta of PROBLEMS) {
      const p = getProblem<ScalarProblem>(meta.id as string);
      expect(p.name).toBe(meta.name);
      expect(p.latex).toBe(meta.latex);
      expect(p.dim).toBe(meta.dim);
      expect(p.description).toBe(meta.description);
      expect(p.tags).toEqual(meta.tags);
      near(p.domain, meta.domain, 0, 0, `${p.id}.domain`);
      near(p.bracket, meta.bracket, 0, 0, `${p.id}.bracket`);
      near(p.x0, meta.x0, 0, 0, `${p.id}.x0`);
      near(p.minima, meta.minima, 1e-15, 1e-16, `${p.id}.minima`);
    }
  });
  for (const row of FIX.problems) {
    it(`${row.id}: f, f′, f″ equal Python`, () => {
      const p = getProblem<ScalarProblem>(row.id);
      for (const v of row.values) {
        near(p.f(v.x), v.f, 1e-13, 1e-300, `${row.id} f(${v.x})`);
        near(p.grad(v.x), v.grad, 1e-12, 1e-300, `${row.id} f'(${v.x})`);
        near(p.hess(v.x), v.hess, 1e-12, 1e-300, `${row.id} f''(${v.x})`);
      }
    });
  }
});

/**
 * `Math.exp` / `Math.log` (V8's fdlibm ports) and CPython's libm differ by 1 ulp at some points
 * (measured: 5 of 42 sample values of drug_concentration and flat_valley, and points of the
 * x_log_x runs). Comparisons whose f-values differ by a few ulps are decided by rounding (the
 * accuracy floor h ≈ √ε·(1 + |x*|) of the Python module docstring), so once a run's bracket is
 * within a few hundred h of x*, a 1-ulp libm difference can flip a comparison and the two traces
 * part — both still correct to the floor. For runs on these problems every step is compared
 * exactly until the Python bracket width or step falls below FLOOR_WIDTH; after that the result must agree
 * to the accuracy floor (same `converged`, |Δx| within the final width + 10⁻⁷, nIter within 5).
 *
 * These Node results are not the browser's: in Chromium every run here, the lab's default pairs
 * included (brent_minimize on drug_concentration: 14 iterations, as in Python; Node: 16), is
 * bit-identical to Python. tests/scalar/browser-parity.mjs checks that exactly (nIter and every
 * iterate) against a running dev server.
 */
const LIBM_SENSITIVE = new Set(['x_log_x', 'drug_concentration']);
const FLOOR_WIDTH = 1e-5;

function widthOf(info: Record<string, unknown>): number {
  const br = info.bracket as number[] | undefined;
  return br ? br[1] - br[0] : Infinity;
}

describe('scalar methods reproduce every Python step', () => {
  for (const c of FIX.runs) {
    const label = `${c.method} on ${c.problem} ${JSON.stringify(c.params)}`;
    it(label, () => {
      const want = resultFromJson(c.result);
      const got = run(c.method, c.problem, c.params);
      const floorAware = LIBM_SENSITIVE.has(c.problem);
      let n = want.trace.length;
      if (floorAware) {
        // The first step that works at the floor: its bracket or its step is below FLOOR_WIDTH
        // (Brent's bracket stays wide while its steps |u − x| shrink).
        const i = want.trace.findIndex(
          (s) => Math.min(widthOf(s.info), s.stepSize ?? Infinity) < FLOOR_WIDTH,
        );
        if (i >= 0) n = i;
      }
      const exact = n === want.trace.length;
      if (exact) {
        expect(got.nIter).toBe(want.nIter);
        expect(got.message).toBe(want.message);
        expect([got.nFev, got.nGev, got.nHev]).toEqual([want.nFev, want.nGev, want.nHev]);
        expect(got.trace.length).toBe(want.trace.length);
      } else {
        expect(Math.abs(got.nIter - want.nIter)).toBeLessThanOrEqual(5);
        const finalWidth = widthOf(want.trace[want.trace.length - 1].info);
        expect(Math.abs((got.x as number) - (want.x as number))).toBeLessThanOrEqual(
          (Number.isFinite(finalWidth) ? finalWidth : 0) + 1e-7,
        );
      }
      expect(got.converged).toBe(want.converged);
      expect(got.trace.length).toBeGreaterThanOrEqual(Math.min(n, want.trace.length));
      // Every compared step: x, f, step size, |f'| and every info key. Late steps of long runs
      // live near the resolution of f, so steps after the tenth get 1e-7 relative.
      for (let i = 0; i < n; i++) {
        const s = got.trace[i],
          w = want.trace[i];
        const tol = i < 10 ? 1e-10 : 1e-7;
        expect(s.k).toBe(w.k);
        near(s.x, w.x, tol, 1e-12, `k=${i} x`);
        near(s.fun, w.fun, tol, 1e-12, `k=${i} fun`);
        near(s.stepSize, w.stepSize, tol, 1e-12, `k=${i} stepSize`);
        near(s.gradNorm, w.gradNorm, tol, 1e-12, `k=${i} gradNorm`);
        // The fit's slope c₁ at x_m ≈ x* is a difference quotient of nearly equal f-values: a
        // 1-ulp libm difference moves it by ulp(f)/(x spacing), about 10⁻¹¹ here.
        near(s.info, w.info, tol, floorAware ? 1e-9 : 1e-12, `k=${i} info`);
      }
      if (exact) {
        near(got.x, want.x, 1e-9, 1e-12, 'x');
        near(got.fun, want.fun, 1e-9, 1e-12, 'fun');
        near(got.extra, want.extra, 1e-9, 1e-12, 'extra');
      }
    });
  }
});

describe('scalar input errors', () => {
  for (const c of FIX.errors) {
    it(`${c.method} ${JSON.stringify(c.params)} throws`, () => {
      expect(() => run(c.method, c.problem, c.params)).toThrow();
      try {
        run(c.method, c.problem, c.params);
      } catch (e) {
        // Same first words as the Python ValueError.
        const head = c.error.split(/[=:(,]/)[0].trim();
        expect((e as Error).message.startsWith(head)).toBe(true);
      }
    });
  }
});

describe('scalar helpers', () => {
  it('fibonacci_numbers equals Python', () => {
    for (const c of FIX.fibonacci) expect(fibonacciNumbers(c.target, c.n_max)).toEqual(c.fib);
  });
  it('formats numbers like Python', () => {
    expect(fmtG(6.48e-9)).toBe('6.48e-09');
    expect(fmtG(0.69)).toBe('0.69');
    expect(fmtG(1e-8)).toBe('1e-08');
    expect(fmtG(123456)).toBe('1.23e+05');
    expect(fmtG(1.6e-7)).toBe('1.6e-07');
    expect(fmtG(3.0, 17)).toBe('3');
    expect(fmtG(0.1, 17)).toBe('0.10000000000000001');
  });
  it('nextUp is math.nextafter(x, inf)', () => {
    expect(nextUp(1)).toBe(1 + Number.EPSILON);
    expect(nextUp(-1)).toBe(-1 + Number.EPSILON / 2);
    expect(nextUp(0)).toBe(Number.MIN_VALUE);
  });
  it('the centered parabola interpolates its three points', () => {
    const p = parabolaThrough(1, 2, 0, 5, 3, 7)!;
    const at = (z: number) => p.coef[0] + p.coef[1] * (z - 1) + p.coef[2] * (z - 1) ** 2;
    expect(at(1)).toBeCloseTo(2, 12);
    expect(at(0)).toBeCloseTo(5, 12);
    expect(at(3)).toBeCloseTo(7, 12);
    expect(parabolaThrough(1, 2, 1, 3, 4, 5)).toBeNull();
  });
  it('accepts a bare callable like Python scalar_problem', () => {
    const r = getMethod('golden_section').fn((x: number) => (x - 0.25) ** 2, {
      bracket: [-1, 1],
      xtol: 1e-8,
      max_iter: 200,
    });
    expect(r.converged).toBe(true);
    expect(Math.abs((r.x as number) - 0.25)).toBeLessThan(1e-7);
  });
});
