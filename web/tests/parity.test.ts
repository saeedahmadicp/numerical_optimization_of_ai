/**
 * Parity harness: replays every Python fixture in src/generated/fixtures/<family>.json with the
 * registered TS port of the same method id (docs/architecture.md, "Parity rule").
 *
 *   - final x within 1e-6 (relative), first min(10, n) iterates within 1e-8, nIter exactly
 *   - stochastic methods (`deterministic: false`): first 10 iterates (1e-8) + final fun (1e-6 rel)
 *
 * Cases whose method is not ported yet are skipped and counted; the summary is printed.
 */
import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, type FixtureCase } from '../src/core/json';
import { getMethod, hasMethod } from '../src/core/registry';
import { getProblem, hasProblem } from '../src/problems/registry';
import type { Point } from '../src/core/types';
import '../src/methods';
import '../src/problems';

const DIR = fileURLToPath(new URL('../src/generated/fixtures/', import.meta.url));
const files = existsSync(DIR)
  ? readdirSync(DIR)
      .filter((f) => f.endsWith('.json'))
      .sort()
  : [];

const counts = { cases: 0, ported: 0, missingMethod: 0, missingProblem: 0 };
const missing = new Map<string, number>();

function flat(x: Point | unknown): number[] {
  if (typeof x === 'number') return [x];
  if (Array.isArray(x)) return (x as unknown[]).flatMap(flat);
  return [];
}

function close(a: unknown, b: unknown, rtol: number, atol = rtol) {
  const xa = flat(a),
    xb = flat(b);
  expect(xa.length).toBe(xb.length);
  xa.forEach((v, i) => {
    const w = xb[i];
    if (!Number.isFinite(w) || !Number.isFinite(v)) return expect(v).toBe(w);
    expect(Math.abs(v - w)).toBeLessThanOrEqual(atol + rtol * Math.abs(w));
  });
}

function runCase(c: FixtureCase) {
  const { spec, fn } = getMethod(c.method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return {
    spec,
    result: fn(getProblem(c.problem), { ...defaults, ...(c.params as Record<string, never>) }),
  };
}

describe('parity with Python fixtures', () => {
  if (files.length === 0) {
    it.skip('no fixtures exported yet (run `npm run gen`)', () => {});
  }
  for (const file of files) {
    const family = file.replace(/\.json$/, '');
    const cases = (JSON.parse(readFileSync(DIR + file, 'utf8')) as Record<string, unknown>[]).map(
      fixtureCaseFromJson,
    );
    describe(family, () => {
      cases.forEach((c, idx) => {
        counts.cases++;
        const name = `${c.method} on ${c.problem} #${idx}`;
        if (!hasMethod(c.method)) {
          counts.missingMethod++;
          missing.set(c.method, (missing.get(c.method) ?? 0) + 1);
          it.skip(`${name} (method not ported)`, () => {});
          return;
        }
        if (!hasProblem(c.problem)) {
          counts.missingProblem++;
          it.skip(`${name} (problem not ported)`, () => {});
          return;
        }
        counts.ported++;
        it(name, () => {
          const { spec, result } = runCase(c);
          const want = c.result;
          const n = Math.min(10, want.trace.length);
          expect(result.trace.length).toBeGreaterThanOrEqual(n);
          for (let k = 0; k < n; k++) close(result.trace[k].x, want.trace[k].x, 1e-8);
          if (spec.deterministic) {
            expect(result.nIter).toBe(want.nIter);
            close(result.x, want.x, 1e-6, 1e-10);
            expect(result.converged).toBe(want.converged);
          } else if (want.fun !== null && result.fun !== null) {
            close(result.fun, want.fun, 1e-6, 1e-10);
          }
        });
      });
    });
  }
  // Always runs (afterAll hooks are skipped when every case is skipped).
  it('reports parity coverage', () => {
    const top = [...missing.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([m, n]) => `${m}×${n}`);
    console.info(
      `[parity] ${files.length} fixture files, ${counts.cases} cases: ${counts.ported} checked, ` +
        `${counts.missingMethod} skipped (method not ported), ${counts.missingProblem} skipped (problem not ported)` +
        (top.length ? `\n[parity] not ported yet: ${top.join(', ')}` : ''),
    );
    expect(counts.ported + counts.missingMethod + counts.missingProblem).toBe(counts.cases);
  });
});
