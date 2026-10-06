/**
 * Differentiation port: strict parity with the Python fixtures
 * (src/generated/fixtures/differentiation.json) — every Step.info key, Result.extra, the message
 * and the evaluation count — plus unit tests for what the fixtures do not cover (fsum, median,
 * %g formatting, complex arithmetic, collapsed levels, failure paths, exactness on polynomials).
 *
 * This file imports only the differentiation port and the calculus problems.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson } from '../../src/core/json';
import { getMethod, listMethods } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Problem, Result } from '../../src/core/types';
import '../../src/problems/calculus';
import {
  collapsed,
  complexF,
  fsum,
  median,
  observedRoundoff,
  pyG,
  type ComplexProblem,
} from '../../src/methods/differentiation/methods';
import { COMPLEX_F, cx, div, mul, powi } from '../../src/methods/differentiation/complex';

const FIXTURE = fileURLToPath(
  new URL('../../src/generated/fixtures/differentiation.json', import.meta.url),
);
const REGISTRY = fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url));

function run(id: string, problem: unknown, params: Record<string, unknown> = {}): Result {
  const { spec, fn } = getMethod(id);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

/** NaN in TS is `null` in the exported JSON (Python NaN → null). */
const norm = (v: unknown) => (typeof v === 'number' && Number.isNaN(v) ? null : v);

function close(a: number, b: number, rtol: number, atol = 0) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return expect(a).toBe(b);
  expect(Math.abs(a - b)).toBeLessThanOrEqual(atol + rtol * Math.abs(b));
}

function sameValue(a: unknown, b: unknown, rtol: number, atol: number, path: string): void {
  a = norm(a);
  if (typeof b === 'number' && typeof a === 'number') {
    try {
      close(a, b, rtol, atol);
    } catch (e) {
      throw new Error(`${path}: ${String(a)} vs ${String(b)} (${(e as Error).message})`, {
        cause: e,
      });
    }
    return;
  }
  if (Array.isArray(b)) {
    expect(Array.isArray(a), path).toBe(true);
    expect((a as unknown[]).length, path).toBe(b.length);
    b.forEach((v, i) => sameValue((a as unknown[])[i], v, rtol, atol, `${path}[${i}]`));
    return;
  }
  expect(a, path).toEqual(b);
}

/**
 * A problem whose f returns Python's own f values (read from the fixture's stencils), so the
 * port can be checked bit for bit: the only TS ↔ Python differences left are libm's last bit
 * in exp/sin (NumPy vs V8), which 1/h amplifies at small h.
 */
function oracle(problem: Problem<number>, trace: { info: Record<string, unknown> }[]) {
  const known = new Map<number, number>();
  for (const s of trace)
    for (const [x, fx] of s.info.stencil as [number, number | null][])
      known.set(x, fx === null ? NaN : fx);
  const f = problem.f as (x: number) => number;
  return { ...problem, f: (x: number) => known.get(x) ?? f(x) };
}

describe.runIf(existsSync(FIXTURE))('differentiation parity with Python fixtures', () => {
  const cases = (JSON.parse(readFileSync(FIXTURE, 'utf8')) as Record<string, unknown>[]).map(
    fixtureCaseFromJson,
  );
  it('has every fixture case', () => expect(cases.length).toBe(8));
  cases.forEach((c, idx) => {
    it(`${c.method} on ${c.problem} #${idx}: same levels, selection and outcome`, () => {
      const result = run(c.method, getProblem(c.problem), c.params as Record<string, unknown>);
      const want = c.result;
      expect(result.nIter).toBe(want.nIter);
      expect(result.nFev).toBe(want.nFev);
      expect(result.converged).toBe(want.converged);
      expect(result.trace.length).toBe(want.trace.length);
      expect(result.extra.k_best).toBe(want.extra.k_best);
      // k_trunc_end / k_confirm_last sit in the round-off regime, where a last-bit libm
      // difference can move them by a level; the oracle test below checks them exactly.
      // The final estimate carries round-off ≈ ε|f|/h at the selected h (harness: 1e-6).
      close(result.x as number, want.x as number, 1e-6, 1e-10);
      // The first ten levels (h ≥ h0/512) agree to 1e-10 relative even with libm differences.
      for (let k = 0; k < Math.min(10, want.trace.length); k++)
        close(result.trace[k].x as number, want.trace[k].x as number, 1e-10);
    });
    it(`${c.method} on ${c.problem} #${idx}: bit for bit on Python's f values`, () => {
      const want = c.result;
      const problem = getProblem(c.problem) as Problem<number>;
      const result = run(
        c.method,
        oracle(problem, want.trace),
        c.params as Record<string, unknown>,
      );
      expect(result.message).toBe(want.message);
      expect(result.nFev).toBe(want.nFev);
      expect(result.converged).toBe(want.converged);
      expect(norm(result.x)).toEqual(want.x);
      result.trace.forEach((s, k) => {
        const w = want.trace[k];
        expect(s.k).toBe(w.k);
        expect(norm(s.x), `trace[${k}].x`).toEqual(w.x);
        expect(s.stepSize).toBe(w.stepSize);
        expect(Object.keys(s.info)).toEqual(Object.keys(w.info));
        for (const key of Object.keys(w.info))
          sameValue(s.info[key], w.info[key], 0, 0, `trace[${k}].info.${key}`);
      });
      for (const key of Object.keys(want.extra)) {
        // h_opt = c·ε^p uses pow(3, 1/3) etc., where V8 and glibc may differ in the last bit.
        if (key === 'h_opt') sameValue(result.extra[key], want.extra[key], 1e-15, 0, 'extra.h_opt');
        else expect(norm(result.extra[key]), `extra.${key}`).toEqual(want.extra[key]);
      }
    });
  });
});

describe.runIf(existsSync(REGISTRY))('registry parity', () => {
  it('has the same ids, names, params, orders, summaries and references as Python', () => {
    const raw = JSON.parse(readFileSync(REGISTRY, 'utf8')) as
      { methods: Record<string, unknown>[] } | Record<string, unknown>[];
    const all = Array.isArray(raw) ? raw : raw.methods;
    const py = all.filter((m) => m.family === 'differentiation');
    const ts = listMethods('differentiation');
    expect(ts.map((m) => m.spec.id).sort()).toEqual(py.map((m) => m.id as string).sort());
    for (const p of py) {
      const { spec } = getMethod(p.id as string);
      expect(spec.name).toBe(p.name);
      expect(spec.order).toBe(p.order);
      expect(spec.summary).toBe(p.summary);
      expect(spec.references).toEqual(p.references);
      expect(spec.needs).toEqual(p.needs);
      const params = (p.params as Record<string, unknown>[]).map((q) => ({
        name: q.name,
        default: q.default,
        kind: q.kind,
        min: q.min,
        max: q.max,
        log: q.log,
        help: q.help,
      }));
      expect(
        spec.params.map((q) => ({
          name: q.name,
          default: q.default,
          kind: q.kind,
          min: q.min,
          max: q.max,
          log: q.log,
          help: q.help,
        })),
      ).toEqual(params);
    }
  });
});

describe('helpers', () => {
  it('fsum is correctly rounded', () => {
    // Python: math.fsum([0.1, 0.2, 0.3, -0.6]) == 2.7755575615628914e-17
    expect(fsum([0.1, 0.2, 0.3, -0.6])).toBe(2.7755575615628914e-17);
    expect(fsum([1e100, 1.0, -1e100, 1e-100])).toBe(1.0);
    expect(fsum([1, 1e-16, 1e-16])).toBe(1.0000000000000002);
    expect(Number.isNaN(fsum([Infinity, -Infinity]))).toBe(true);
    expect(fsum([1.7976931348623157e308, 1.7976931348623157e308])).toBe(Infinity);
  });
  it('median matches Python statistics.median', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([5])).toBe(5);
  });
  it('pyG matches Python %.3g', () => {
    expect(pyG(4.76837158203125e-8)).toBe('4.77e-08');
    expect(pyG(0.025)).toBe('0.025');
    expect(pyG(0.000244140625)).toBe('0.000244');
    expect(pyG(1.6e-10)).toBe('1.6e-10');
    expect(pyG(123456)).toBe('1.23e+05');
    expect(pyG(12.5)).toBe('12.5');
    expect(pyG(100)).toBe('100');
    expect(pyG(Infinity)).toBe('inf');
  });
  it('collapsed detects merged abscissae', () => {
    expect(collapsed(1, 1e-17, [-1, 1])).toBe(true);
    expect(collapsed(1, 1e-10, [-1, 1])).toBe(false);
    expect(collapsed(0, 1e-300, [-2, -1, 1, 2])).toBe(false);
  });
  it('observed round-off carries a later growth back by 2^-q per level', () => {
    const nu = observedRoundoff([null, 1, 0.5, 0.25, 1, null], 1);
    // Δ_4 = 1 grew: ν_3 = 1/(1.5)·2^{-1}, ν_2 = that /2, …
    expect(nu[4]).toBe(0);
    expect(nu[3]).toBeCloseTo(1 / 1.5 / 2, 15);
    expect(nu[2]).toBeCloseTo(1 / 1.5 / 4, 15);
  });
});

describe('complex arithmetic follows CPython', () => {
  it('z**2 is z*z and division is Smith’s algorithm', () => {
    const z = cx(0.5, 1e-3);
    expect(powi(z, 2)).toEqual(mul(cx(1, 0), mul(z, z)));
    const q = div(cx(1, 0), cx(1.25, 1e-3));
    const d = 1.25 ** 2 + 1e-6;
    expect(q.re).toBeCloseTo(1.25 / d, 15);
    expect(q.im).toBeCloseTo(-1e-3 / d, 15);
  });
  it('every calculus problem has a complex twin that agrees with f on the real line', () => {
    for (const id of Object.keys(COMPLEX_F)) {
      const p = getProblem(id) as unknown as { f: (x: number) => number; x0: number };
      const w = COMPLEX_F[id](cx(p.x0, 0));
      expect(w.re).toBeCloseTo(p.f(p.x0), 14);
      expect(Math.abs(w.im)).toBe(0);
    }
  });
});

describe('method behavior', () => {
  const exp = getProblem('exp_0_1');
  it('complex step stays at machine precision for tiny h', () => {
    const r = run('complex_step', exp, { h0: 1e-20, levels: 5 });
    for (const s of r.trace)
      expect(Math.abs((s.x as number) - Math.E)).toBeLessThan(5e-16 * Math.E);
  });
  it('central difference is exact on a quadratic up to rounding', () => {
    const quad: ComplexProblem = {
      id: 'quad',
      name: 'quad',
      latex: 'x^2',
      dim: 1,
      domain: [-1, 1],
      f: (x) => x * x,
      grad: (x) => 2 * x,
      x0: 0.75,
    };
    const r = run('central_difference', quad, { h0: 0.5, levels: 3 });
    for (const s of r.trace) expect(s.x).toBeCloseTo(1.5, 14);
  });
  it('five-point reuses x0 ± 2h_k = x0 ± h_{k−1}: two new evaluations per level', () => {
    const r = run('five_point_stencil', exp, { h0: 0.1, levels: 4 });
    expect(r.nFev).toBe(4 + 2 * 4);
  });
  it('forward difference evaluates f(x0) once', () => {
    const r = run('forward_difference', exp, { h0: 0.1, levels: 4 });
    expect(r.nFev).toBe(1 + 5);
  });
  it('levels = 0 has no error estimate', () => {
    const r = run('central_difference', exp, { levels: 0 });
    expect(r.converged).toBe(false);
    expect(r.message).toBe('levels = 0: a single estimate has no error estimate');
    expect(r.extra.k_best).toBeNull();
  });
  it('a sweep into collapsed abscissae marks those levels and never selects them', () => {
    const r = run('central_difference', exp, { h0: 1e-14, levels: 10 });
    const col = r.trace.filter((s) => s.info.collapsed);
    expect(col.length).toBeGreaterThan(0);
    for (const s of col) {
      expect(s.info.err_est).toBeNull();
      expect(s.info.roundoff).toBeNull();
      expect(s.info.confirms).toBe(false);
    }
    const kBest = r.extra.k_best as number | null;
    if (kBest !== null) expect(r.trace[kBest].info.collapsed).toBe(false);
  });
  it('NaN values (sqrt left of 0) are skipped and reported', () => {
    const r = run('central_difference', getProblem('sqrt_0_1'), { h0: 1, levels: 20 });
    expect(r.message).toMatch(/level\(s\) with non-finite values were skipped/);
    expect(r.trace[0].x).toBeNaN();
    expect(r.trace[0].info.error).toBeNull();
  });
  it('invalid input is rejected like Python', () => {
    expect(() => run('central_difference', exp, { h0: 0 })).toThrow(/h0 must be a positive/);
    expect(() => run('central_difference', exp, { levels: 1.5 })).toThrow(/levels must be/);
    expect(() => run('central_difference', exp, { tol: -1 })).toThrow(/tol must be/);
    expect(() => run('central_difference', exp, { x0: NaN })).toThrow(/x0 must be finite/);
  });
  it('complex step needs a complex-analytic f', () => {
    const custom: ComplexProblem = {
      id: 'custom',
      name: 'custom',
      latex: 'f',
      dim: 1,
      domain: [0, 1],
      f: (x) => x,
      x0: 0.5,
    };
    expect(complexF(custom)).toBeNull();
    expect(() => run('complex_step', custom)).toThrow(/complex/);
    const withC = { ...custom, fComplex: (z: { re: number; im: number }) => mul(z, z) };
    const r = run('complex_step', withC, { levels: 3 });
    expect(r.x).toBeCloseTo(1, 14);
  });
  it('second derivative compares with f″', () => {
    const r = run('second_derivative_central', exp, { h0: 0.1, levels: 12 });
    expect(r.extra.exact).toBe(Math.exp(1));
    expect(r.extra.derivative).toBe(2);
    expect(Math.abs((r.x as number) - Math.E)).toBeLessThan(1e-6);
  });
  it('Richardson rows grow by one entry per level and restart after a collapse', () => {
    const r = run('richardson_extrapolation', getProblem('sin_0_pi'), { h0: 0.5, levels: 6 });
    r.trace.forEach((s, k) => expect((s.info.row as number[]).length).toBe(k + 1));
    expect(Math.abs((r.x as number) - 0.5)).toBeLessThan(1e-12);
  });
});
