/**
 * Parity of the systems port (src/methods/roots/systems.ts, src/problems/systems.ts) with the
 * Python fixtures in src/generated/fixtures/systems.json — the shared harness checks (first
 * min(10, n) iterates within 1e-8, nIter exactly, final x within 1e-6 relative, converged) plus
 * stricter ones the harness does not make: evaluation counts, the message, `extra`, the info keys
 * of every step and the info values of the first ten steps.
 *
 * It imports only the systems modules, so it runs while other ports are being written.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import '../../src/methods/roots/systems';
import '../../src/problems/systems';

const FILE = fileURLToPath(new URL('../../src/generated/fixtures/systems.json', import.meta.url));
const cases = (JSON.parse(readFileSync(FILE, 'utf8')) as Record<string, unknown>[]).map(
  fixtureCaseFromJson,
);

function flat(x: unknown): number[] {
  if (typeof x === 'number') return [x];
  if (x === null) return [NaN];
  if (Array.isArray(x)) return x.flatMap(flat);
  if (x && typeof x === 'object')
    return Object.keys(x)
      .sort()
      .flatMap((k) => flat((x as Record<string, unknown>)[k]));
  return [];
}

function close(a: unknown, b: unknown, rtol: number, atol = rtol) {
  const xa = flat(a),
    xb = flat(b);
  expect(xa.length).toBe(xb.length);
  xa.forEach((v, i) => {
    const w = xb[i];
    if (!Number.isFinite(w) || !Number.isFinite(v)) {
      if (Number.isNaN(w)) return expect(Number.isNaN(v)).toBe(true);
      return expect(v).toBe(w);
    }
    expect(Math.abs(v - w)).toBeLessThanOrEqual(atol + rtol * Math.abs(w));
  });
}

describe('systems parity (strict)', () => {
  it('covers every fixture', () => {
    expect(cases.length).toBe(8);
  });
  cases.forEach((c, idx) => {
    it(`${c.method} on ${c.problem} #${idx}`, () => {
      const { spec, fn } = getMethod(c.method);
      const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
      const got = fn(getProblem(c.problem), {
        ...defaults,
        ...(c.params as Record<string, never>),
      });
      const want = c.result;
      expect(got.nIter).toBe(want.nIter);
      expect(got.converged).toBe(want.converged);
      expect(got.trace.length).toBe(want.trace.length);
      close(got.x, want.x, 1e-6, 1e-10);
      expect(got.nFev).toBe(want.nFev);
      expect(got.nGev).toBe(want.nGev);
      expect(got.nHev).toBe(0);
      // Same message; numbers at roundoff level (‖F‖ ≈ 1e-16) may differ in the last digits.
      const mask = (m: string) => m.replace(/[-+]?\d+(\.\d+)?(e[-+]\d+)?/g, '#');
      expect(mask(got.message)).toBe(mask(want.message));
      expect(got.extra).toEqual(want.extra);
      got.trace.forEach((s, k) => {
        const w = want.trace[k];
        expect(s.k).toBe(w.k);
        expect(Object.keys(s.info).sort()).toEqual(Object.keys(w.info).sort());
        if (k < 10) {
          close(s.x, w.x, 1e-8);
          close(s.fun, w.fun, 1e-8);
          close(s.stepSize, w.stepSize, 1e-8);
          for (const key of Object.keys(w.info)) close(s.info[key], w.info[key], 1e-7, 1e-9);
        }
      });
    });
  });
});
