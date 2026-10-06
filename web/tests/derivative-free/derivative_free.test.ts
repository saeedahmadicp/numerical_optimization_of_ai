/**
 * numopt.unconstrained.derivative_free — TS port checks beyond the parity harness:
 *   - the registered specs equal registry.json (ids, params, metadata);
 *   - every derivative-free parity fixture matches step by step (x, fun, step size, every info
 *     key), with the same counts, message and converged flag;
 *   - extra Python runs (fixtures/derivative_free_python.json, from gen_derivative_free_fixture.py):
 *     more problems, explicit x0, max_iter stops, the extreme barrier (NaN / +inf), f = −inf,
 *     non-finite f(x0), a 1-D problem — and the ValueError messages;
 *   - focused unit tests (line minimizer, bare callables, input safety).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, methodSpecFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Problem, Result, Step, Vector } from '../../src/core/types';
import {
  DerivativeFreeInputError,
  compassSearch,
  hookeJeeves,
  lineMinimize,
  nelderMead,
  powell,
} from '../../src/methods/unconstrained/derivative_free';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems';
import { mismatch, readJson, type Tol } from '../shared-ports/compare';

type Raw = Record<string, unknown>;
const IDS = ['nelder_mead', 'powell', 'hooke_jeeves', 'compass_search'];
/** Step-by-step agreement: the port keeps Python's operation order, so values agree to rounding. */
const TOL: Tol = { rtol: 1e-10, atol: 1e-12 };

/** A TS Step in the snake_case shape of the Python export. */
function rawStep(s: Step): Raw {
  return {
    k: s.k,
    x: s.x,
    fun: s.fun,
    grad_norm: s.gradNorm,
    step_size: s.stepSize,
    info: s.info,
  };
}

/**
 * Runs whose last sweep happens on the rounding plateau of f, where the TS problem and numpy
 * differ by an ulp (numpy's float64 `x ** 2` is C `pow`, not always equal to `x * x`; numpy sums a
 * matrix-vector product in another order): the final sweep then evaluates other trial points.
 * Iterations, convergence and the iterates still agree (to 1e-8); the trials of the plateau sweeps,
 * the stop message and n_fev do not.
 */
const ROUNDING = new Set(['powell/himmelblau']);

function checkSummary(got: Result, want: Raw, rounding = false) {
  expect(got.method).toBe(want.method);
  expect(got.converged).toBe(want.converged);
  expect(got.nIter).toBe(want.n_iter);
  expect(got.nGev).toBe(0);
  expect(got.nHev).toBe(0);
  expect(got.extra).toEqual({});
  if (rounding) {
    expect(mismatch(got.x, want.x, { rtol: 1e-6, atol: 1e-10 })).toBeNull();
    expect(mismatch(got.fun, want.fun, { rtol: 1e-6, atol: 1e-20 })).toBeNull();
    return;
  }
  expect(got.message).toBe(want.message);
  expect(got.nFev).toBe(want.n_fev);
  expect(mismatch(got.x, want.x, TOL)).toBeNull();
  expect(mismatch(got.fun, want.fun, TOL)).toBeNull();
}

// ---------------------------------------------------------------------------------------
// Registry metadata
// ---------------------------------------------------------------------------------------

describe('registered specs', () => {
  const registry = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[] | { methods: Raw[] };
  const list = (Array.isArray(registry) ? registry : registry.methods).map(methodSpecFromJson);
  for (const id of IDS) {
    it(`${id} matches registry.json`, () => {
      const want = list.find((m) => m.id === id);
      expect(want).toBeDefined();
      const { spec } = getMethod(id);
      const strip = ({ label: _l, tex: _t, ...p }: (typeof spec.params)[number]) => p;
      expect({ ...spec, params: spec.params.map(strip) }).toEqual(want);
    });
  }
});

// ---------------------------------------------------------------------------------------
// Parity fixtures, every step and every info key
// ---------------------------------------------------------------------------------------

describe('parity fixtures, step by step', () => {
  const raw = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/fixtures/unconstrained.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  const cases = raw.map(fixtureCaseFromJson).filter((c) => IDS.includes(c.method));
  it('covers every derivative-free fixture', () => expect(cases.length).toBe(8));
  for (const c of cases) {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)}`, () => {
      const { fn, spec } = getMethod(c.method);
      const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
      const got = fn(getProblem(c.problem), { ...defaults, ...(c.params as Raw) } as never);
      const want = c.result;
      expect(got.message).toBe(want.message);
      expect(got.converged).toBe(want.converged);
      expect(got.nIter).toBe(want.nIter);
      expect(got.nFev).toBe(want.nFev);
      expect(got.nGev).toBe(want.nGev);
      expect(got.nHev).toBe(want.nHev);
      expect(got.trace.length).toBe(want.trace.length);
      expect(mismatch(got.x, want.x, TOL)).toBeNull();
      expect(mismatch(got.fun, want.fun, TOL)).toBeNull();
      for (let k = 0; k < want.trace.length; k++)
        expect(mismatch(got.trace[k], want.trace[k], TOL, `trace[${k}]`)).toBeNull();
    });
  }
});

// ---------------------------------------------------------------------------------------
// Extra Python runs
// ---------------------------------------------------------------------------------------

// The custom problems of gen_derivative_free_fixture.py.
const CUSTOM: Record<string, Problem<Vector> | Problem<number>> = {
  barrier_2d: {
    id: 'barrier_2d',
    name: 'barrier',
    latex: '',
    dim: 2,
    domain: [],
    x0: [0.0, 0.0],
    f: (x: Vector) => {
      if (x[0] < -0.25) return NaN;
      if (x[1] > 2.6) return Infinity;
      return (x[0] - 1.0) ** 2 + 3.0 * (x[1] - 2.5) ** 2 + 0.5 * x[0] * x[1];
    },
  },
  unbounded_2d: {
    id: 'unbounded_2d',
    name: 'unbounded',
    latex: '',
    dim: 2,
    domain: [],
    x0: [0.0, 0.0],
    f: (x: Vector) => (x[1] < -1.0 ? -Infinity : x[0] ** 2 + x[1]),
  },
  nan_start: {
    id: 'nan_start',
    name: 'nan',
    latex: '',
    dim: 2,
    domain: [],
    x0: [1.0, 2.0],
    f: () => NaN,
  },
  minus_inf_start: {
    id: 'minus_inf_start',
    name: '-inf',
    latex: '',
    dim: 2,
    domain: [],
    x0: [1.0, 2.0],
    f: () => -Infinity,
  },
  one_d: {
    id: 'one_d',
    name: '1-D',
    latex: '',
    dim: 1,
    domain: [-3.0, 3.0],
    x0: 0.5,
    f: (t: number) => (t - 2.0) ** 2 + 0.1 * t ** 4,
  },
};

const problemFor = (id: string) => CUSTOM[id] ?? getProblem(id);

interface ExtraRun {
  method: string;
  problem: string;
  params: Raw;
  result: Raw & { trace_len: number; trace_head: Raw[]; trace_tail: Raw[] };
}
interface ExtraError {
  method: string;
  problem: string;
  params: Raw;
  message: string;
}
const EXTRA = readJson<{ runs: ExtraRun[]; errors: ExtraError[] }>(
  '../derivative-free/fixtures/derivative_free_python.json',
);

function run(method: string, problem: string, params: Raw): Result {
  const { fn, spec } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problemFor(problem), { ...defaults, ...params } as never);
}

describe('extra Python runs', () => {
  for (const r of EXTRA.runs) {
    it(`${r.method} on ${r.problem} ${JSON.stringify(r.params)}`, () => {
      const got = run(r.method, r.problem, r.params);
      const rounding = ROUNDING.has(`${r.method}/${r.problem}`);
      checkSummary(got, r.result, rounding);
      expect(got.trace.length).toBe(r.result.trace_len);
      if (rounding) {
        r.result.trace_head.forEach((s, k) =>
          expect(mismatch(got.trace[k].x, s.x, { rtol: 1e-8, atol: 1e-8 }, `x[${k}]`)).toBeNull(),
        );
        return;
      }
      r.result.trace_head.forEach((s, k) =>
        expect(mismatch(rawStep(got.trace[k]), s, TOL, `trace[${k}]`)).toBeNull(),
      );
      const off = got.trace.length - r.result.trace_tail.length;
      r.result.trace_tail.forEach((s, j) =>
        expect(mismatch(rawStep(got.trace[off + j]), s, TOL, `trace[${off + j}]`)).toBeNull(),
      );
    });
  }

  for (const e of EXTRA.errors) {
    it(`ValueError: ${e.method} ${JSON.stringify(e.params)}`, () => {
      let err: unknown = null;
      try {
        run(e.method, e.problem, e.params);
      } catch (x) {
        err = x;
      }
      expect(err).toBeInstanceOf(DerivativeFreeInputError);
      expect((err as Error).name).toBe('ValueError');
      expect((err as Error).message).toBe(e.message);
    });
  }
});

// ---------------------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------------------

describe('unit', () => {
  it('lineMinimize finds the minimizer of a parabola to Brent tolerance', () => {
    let calls = 0;
    const phi = (a: number) => {
      calls++;
      return (a - 3.7) ** 2 + 1.0;
    };
    const res = lineMinimize(phi, phi(0) /* f0 */);
    expect(res.ok).toBe(true);
    expect(Math.abs(res.alpha - 3.7)).toBeLessThan(1e-6);
    expect(res.f).toBeLessThanOrEqual(1.0 + 1e-12);
    expect(res.trials.length).toBe(calls - 1);
  });

  it('lineMinimize reports −∞ and an unbracketable line without returning −∞', () => {
    const inf = lineMinimize((a) => (a > 5 ? -Infinity : -a), 0);
    expect(inf.ok).toBe(false);
    expect(inf.message).toBe('f = -inf on the line: f is unbounded below');
    expect(Number.isFinite(inf.f)).toBe(true);
    // Decreasing for α > 0 at every scale, finite everywhere: mnbrak gives up after 100 expansions.
    const lin = lineMinimize((a) => -Math.log1p(Math.abs(a)), 0);
    expect(lin.ok).toBe(false);
    expect(lin.message).toBe('could not bracket a minimum along the line (f keeps decreasing)');
    expect(lin.alpha).toBeGreaterThan(0);
  });

  it('accepts a bare callable with an explicit x0 and never mutates x0', () => {
    const x0 = [-1.2, 1.0];
    const f = (x: Vector) => {
      const v = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2;
      x[0] = 1e9; // a hostile f: the methods pass copies
      return v;
    };
    for (const m of [nelderMead, powell, hookeJeeves, compassSearch]) {
      const res = m(f, { x0, max_iter: 50 });
      expect(x0).toEqual([-1.2, 1.0]);
      if (m !== nelderMead) expect(res.trace[0].x).toEqual([-1.2, 1.0]); // NM: best vertex
      expect(Number.isFinite(res.fun)).toBe(true);
      expect(res.nIter).toBe(res.trace[res.trace.length - 1].k);
    }
  });

  it('rejects a missing start point with the Python message', () => {
    expect(() => nelderMead((x: Vector) => x[0] ** 2, {})).toThrow(
      'custom: no starting point given and the problem has no default x0',
    );
  });

  it('counts every evaluation of f', () => {
    for (const m of [nelderMead, powell, hookeJeeves, compassSearch]) {
      let calls = 0;
      const res = m((x: Vector) => (calls++, (x[0] - 1) ** 2 + 2 * (x[1] + 0.5) ** 2), {
        x0: [0, 0],
      });
      expect(res.nFev).toBe(calls);
      expect(res.converged).toBe(true);
    }
  });
});
