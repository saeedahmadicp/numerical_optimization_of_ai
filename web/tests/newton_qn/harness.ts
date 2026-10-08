/**
 * Shared helpers for tests/newton_qn/*.test.ts: the custom problems of gen_newton_qn_fixture.py and
 * a step-by-step comparison of a TS Result with a Python result (snake_case JSON).
 */
import { expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { methodSpecFromJson, resultFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Matrix, MethodSpec, Problem, Result, Vector } from '../../src/core/types';
import { getProblem } from '../../src/problems/registry';
import { CANONICAL, sameText } from '../fixtures/platform';
import { mismatch, readJson, type Tol } from '../shared-ports/compare';

export type Raw = Record<string, unknown>;

const D2 = [
  [-2, 2],
  [-2, 2],
];

const rosen = (x: Vector) => (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2;
const rosenGrad = (x: Vector): Vector => [
  -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
  200.0 * (x[1] - x[0] ** 2),
];

function custom(
  id: string,
  x0: Vector,
  f: (x: Vector) => number,
  grad?: (x: Vector) => Vector,
  hess?: (x: Vector) => Matrix,
): Problem<Vector> {
  return { id, name: '', latex: '', dim: x0.length, domain: D2, x0, f, grad, hess };
}

/** The custom problems of gen_newton_qn_fixture.py (same formulas, same operation order). */
export const CUSTOM: Record<string, Problem<Vector>> = {
  rosen_nohess: custom('rosen_nohess', [-1.2, 1.0], rosen, rosenGrad),
  rosen_fonly: custom('rosen_fonly', [-1.2, 1.0], rosen),
  saddle: custom(
    'saddle',
    [1.0, 0.5],
    (x) => x[0] ** 2 - x[1] ** 2,
    (x) => [2.0 * x[0], -2.0 * x[1]],
    () => [
      [2.0, 0.0],
      [0.0, -2.0],
    ],
  ),
  bowl_down: custom(
    'bowl_down',
    [1.0, 0.5],
    (x) => -(x[0] ** 2 + x[1] ** 2),
    (x) => [-2.0 * x[0], -2.0 * x[1]],
    () => [
      [-2.0, 0.0],
      [0.0, -2.0],
    ],
  ),
  valley: custom(
    'valley',
    [1.0, 2.0],
    (x) => (x[0] + x[1]) ** 2,
    (x) => [2.0 * (x[0] + x[1]), 2.0 * (x[0] + x[1])],
    () => [
      [2.0, 2.0],
      [2.0, 2.0],
    ],
  ),
  valley_fonly: custom('valley_fonly', [1.0, 2.0], (x) => (x[0] + x[1]) ** 2),
  negsemi: custom(
    'negsemi',
    [0.0, 0.0],
    (x) => -(x[0] ** 2) + x[1] ** 4,
    (x) => [-2.0 * x[0], 4.0 * x[1] ** 3],
    (x) => [
      [-2.0, 0.0],
      [0.0, 12.0 * x[1] ** 2],
    ],
  ),
  soft_abs: custom(
    'soft_abs',
    [1.5, 0.5],
    (x) => Math.sqrt(1.0 + x[0] ** 2) + Math.sqrt(1.0 + x[1] ** 2),
    (x) => [x[0] / Math.sqrt(1.0 + x[0] ** 2), x[1] / Math.sqrt(1.0 + x[1] ** 2)],
    (x) => [
      [(1.0 + x[0] ** 2) ** -1.5, 0.0],
      [0.0, (1.0 + x[1] ** 2) ** -1.5],
    ],
  ),
  quartic3: custom(
    'quartic3',
    [1.5, -1.0, 0.7],
    (x) =>
      x[0] ** 4 + (x[0] - x[1]) ** 2 + 2.0 * (x[1] + x[2]) ** 2 + x[2] ** 4 + 0.5 * x[0] * x[2],
    (x) => [
      4.0 * x[0] ** 3 + 2.0 * (x[0] - x[1]) + 0.5 * x[2],
      -2.0 * (x[0] - x[1]) + 4.0 * (x[1] + x[2]),
      4.0 * (x[1] + x[2]) + 4.0 * x[2] ** 3 + 0.5 * x[0],
    ],
    (x) => [
      [12.0 * x[0] ** 2 + 2.0, -2.0, 0.5],
      [-2.0, 6.0, 4.0],
      [0.5, 4.0, 4.0 + 12.0 * x[2] ** 2],
    ],
  ),
};

export const problemById = (id: string): unknown => CUSTOM[id] ?? getProblem(id);

export function run(method: string, problem: unknown, params: Raw): Result {
  const { spec, fn } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

/** Python specs of the given ids from registry.json. */
export function pythonSpecs(ids: readonly string[]): MethodSpec[] {
  const registry = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  return registry.filter((m) => ids.includes(m.id as string)).map(methodSpecFromJson);
}

/** The registered TS spec without the TS-only display fields (label, tex). */
export function tsSpec(id: string): MethodSpec {
  const got = getMethod(id).spec;
  const params = got.params.map(({ label: _l, tex: _t, ...p }) => p);
  return { ...got, params };
}

export interface Expect {
  /** Tolerance for the steps of the head (k < headSteps) and for the final x. */
  head: Tol;
  /** Tolerance for later steps (long runs accumulate rounding differences). */
  tail: Tol;
  headSteps: number;
}

export const DEFAULT_EXPECT: Expect = {
  head: { rtol: 1e-9, atol: 1e-12 },
  tail: { rtol: 1e-6, atol: 1e-9 },
  headSteps: 10,
};

/**
 * Compare a TS Result with a Python result field by field. The Python trace may be trimmed
 * (gen_newton_qn_fixture.py keeps the first and last steps); steps are matched by k.
 */
export function expectSameResult(got: Result, wantRaw: Raw, ex: Expect = DEFAULT_EXPECT) {
  const want = resultFromJson(wantRaw);
  expect(got.method).toBe(want.method);
  expect(got.message).toBe(want.message);
  expect(got.converged).toBe(want.converged);
  expect([got.nIter, got.nFev, got.nGev, got.nHev]).toEqual([
    want.nIter,
    want.nFev,
    want.nGev,
    want.nHev,
  ]);
  const last = want.trace[want.trace.length - 1];
  expect(got.trace.length).toBe(last.k + 1);
  for (const w of want.trace) {
    const s = got.trace[w.k];
    const tol = w.k < ex.headSteps ? ex.head : ex.tail;
    const m = mismatch(
      { k: s.k, x: s.x, fun: s.fun, gradNorm: s.gradNorm, stepSize: s.stepSize, info: s.info },
      { k: w.k, x: w.x, fun: w.fun, gradNorm: w.gradNorm, stepSize: w.stepSize, info: w.info },
      tol,
      `trace[${w.k}]`,
    );
    if (m !== null) throw new Error(m);
  }
  expect(mismatch(got.x, want.x, ex.tail, 'x')).toBeNull();
  expect(mismatch(got.fun, want.fun, ex.tail, 'fun')).toBeNull();
  expect(mismatch(got.extra, want.extra, undefined, 'extra')).toBeNull();
}

/**
 * Loose comparison for runs whose late iterates are dominated by rounding noise that a port cannot
 * reproduce bit for bit: n ≥ 3 (NumPy's BLAS sums dot products in a SIMD order), and 2-D runs whose
 * first difference is a last-bit difference of the problem's f or ∇f (NumPy's scalar `x ** 2` is
 * not always x·x) or of a central-difference ∇f, amplified where yᵀs or ∇f is at rounding level.
 * Checks the parity rule: the first 10 iterates within 1e-8, the same `converged`, the same
 * message up to its numbers, and the final x within 1e-6 (relative) when the run converged.
 */
export function expectSimilarResult(got: Result, wantRaw: Raw) {
  const want = resultFromJson(wantRaw);
  const words = (m: string) => m.replace(/-?\d+(\.\d+)?(e[+-]\d+)?/g, '#');
  expect(words(got.message)).toBe(words(want.message));
  expect(got.converged).toBe(want.converged);
  for (const w of want.trace.filter((t) => t.k < 10)) {
    const m = mismatch(got.trace[w.k]?.x, w.x, { rtol: 1e-8, atol: 1e-8 }, `trace[${w.k}].x`);
    if (m !== null) throw new Error(m);
  }
  if (want.converged) {
    const m = mismatch(got.x, want.x, { rtol: 1e-6, atol: 1e-8 }, 'x');
    if (m !== null) throw new Error(m);
  }
}

/**
 * Compare a TS Result with a Python reference dump (gen_newton_qn_fixture.py). On the canonical
 * platform (tests/fixtures/platform.ts) this is `expectSameResult`. Elsewhere the dump rounds
 * differently from the arithmetic the ports replay, so it is the parity rule of
 * docs/architecture.md (`expectSimilarResult`: the first 10 iterates within 1e-8, the same
 * `converged`, the message up to its numbers, the final x within 1e-6) plus the same message
 * integers and the same counts. A run that is `chaotic` (its counts, even its outcome, differ
 * between platforms in Python itself) is compared over its first 10 iterates, and by its final x
 * when both runs converged. A `noisy` run (a finite-difference ∇²f
 * from f values: its relative error √ε, amplified by κ(∇²f), moves the iterates by about 1e-6
 * between platforms) keeps the counts and the final x, but not the first iterates.
 */
export function expectSameDump(
  got: Result,
  wantRaw: Raw,
  {
    chaotic = false,
    noisy = false,
    ex = DEFAULT_EXPECT,
  }: { chaotic?: boolean; noisy?: boolean; ex?: Expect } = {},
) {
  if (CANONICAL) return expectSameResult(got, wantRaw, ex);
  const want = resultFromJson(wantRaw);
  if (chaotic) {
    // Another CPU's Python takes another path at some acceptance decision, so even the outcome
    // (converged or max_iter) can differ: the first iterates, and the final x when both runs
    // converged.
    for (const w of want.trace.filter((t) => t.k < 10)) {
      const m = mismatch(got.trace[w.k]?.x, w.x, { rtol: 1e-8, atol: 1e-8 }, `trace[${w.k}].x`);
      if (m !== null) throw new Error(m);
    }
    if (got.converged && want.converged)
      expect(mismatch(got.x, want.x, { rtol: 1e-6, atol: 1e-8 }, 'x')).toBeNull();
    return;
  }
  if (noisy) {
    expect(got.converged).toBe(want.converged);
    if (want.converged) expect(mismatch(got.x, want.x, { rtol: 1e-6, atol: 1e-8 }, 'x')).toBeNull();
  } else expectSimilarResult(got, wantRaw);
  expect(sameText(got.message, want.message), `${got.message} vs ${want.message}`).toBe(true);
  expect([got.nIter, got.nFev, got.nGev, got.nHev]).toEqual([
    want.nIter,
    want.nFev,
    want.nGev,
    want.nHev,
  ]);
  expect(got.trace.length).toBe(want.trace[want.trace.length - 1].k + 1);
}

export interface RefCase {
  method: string;
  problem: string;
  params: Raw;
  result?: Raw;
  error?: string;
}

export function referenceCases(group: 'newton' | 'quasi_newton'): RefCase[] {
  const all = readJson<Record<string, RefCase[]>>('../newton_qn/fixtures/newton_qn_python.json');
  // Python NaN parameters are exported as JSON null.
  return all[group].map((c) => ({
    ...c,
    params: Object.fromEntries(Object.entries(c.params).map(([k, v]) => [k, v ?? NaN])),
  }));
}

/** A stable key for a reference case: `method problem {params}`. */
export const caseKey = (c: RefCase) => `${c.method} ${c.problem} ${JSON.stringify(c.params)}`;

export function generatedFixtures(methods: readonly string[]): RefCase[] {
  return readJson<RefCase[]>('../../src/generated/fixtures/unconstrained.json').filter((c) =>
    methods.includes(c.method),
  );
}

/** Python `ValueError` text of a TS throw (the ports set `name = 'ValueError'`). */
export function errorOf(fn: () => unknown): { name: string; message: string } | null {
  try {
    fn();
    return null;
  } catch (e) {
    const err = e as Error;
    return { name: err.name, message: err.message };
  }
}
