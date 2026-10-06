/**
 * numopt.unconstrained.conjugate_gradient + trust_region — TS port checks beyond the parity
 * harness:
 *   - the nine registered specs equal registry.json;
 *   - every fixture of the nine methods matches Python step by step (x, f, ‖∇f‖, step size and
 *     every info key), with the same counts, message, converged flag and extra.
 *
 * This file imports only the two ports (and the problems), so it runs even while other ports
 * are being written.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, methodSpecFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems/unconstrained';
import { readJson } from '../shared-ports/compare';
import { IDS, expectSameResult, run, type Raw } from './helpers';

describe('conjugate_gradient + trust_region registry', () => {
  const registry = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  const python = registry.filter((m) => IDS.includes(m.id as string)).map(methodSpecFromJson);

  it('registers the nine Python methods with identical specs', () => {
    expect(python.map((s) => s.id).sort()).toEqual(IDS);
    for (const want of python) {
      const got = getMethod(want.id).spec;
      // label / tex are TS-only display fields
      const params = got.params.map(({ label: _l, tex: _t, ...p }) => p);
      expect({ ...got, params }).toEqual(want);
    }
  });

  it('documents every method for the MethodCard', () => {
    for (const id of IDS) {
      const doc = getMethod(id).doc;
      expect(doc?.rule.length).toBeGreaterThan(0);
      expect(doc?.intuition.length).toBeGreaterThan(0);
    }
  });
});

describe('unconstrained fixtures of cg_* and trust_region_*, step by step', () => {
  const cases = readJson<Raw[]>('../../src/generated/fixtures/unconstrained.json')
    .filter((c) => IDS.includes(c.method as string))
    .map((c) => ({ c: fixtureCaseFromJson(c), raw: c.result as Raw }));

  it('has a fixture for every method', () => {
    expect([...new Set(cases.map(({ c }) => c.method))].sort()).toEqual(IDS);
  });

  cases.forEach(({ c, raw }, i) => {
    it(`${c.method} on ${c.problem} #${i}`, () => {
      expectSameResult(run(c.method, getProblem(c.problem), c.params as Raw), raw);
    });
  });
});
