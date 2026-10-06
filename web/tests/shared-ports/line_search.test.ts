/**
 * numopt.line_search.methods — TS port checks beyond the parity harness:
 *   - the registered specs equal registry.json;
 *   - every line_search fixture matches step by step (x, fun, ‖∇f‖, α, every info key), with the
 *     same counts, message and extra;
 *   - `search()` matches direct Python calls (results and ValueError messages);
 *   - the demos match Python on Newton directions, failure paths, 1-D problems and
 *     finite-difference derivatives (tests/shared-ports/fixtures, from gen_shared_ports_fixture.py);
 *   - focused unit tests (formatting, interpolation, breakdowns the fixtures cannot show).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, methodSpecFromJson, resultFromJson } from '../../src/core/json';
import { getMethod, listMethods } from '../../src/core/registry';
import type { Matrix, Problem, Result, Vector } from '../../src/core/types';
import {
  KINDS,
  LineSearchInputError,
  LineSearchZeroDivision,
  cubicMinimizer,
  formatG,
  formatRepr,
  quadraticMinimizer,
  search,
  type SearchOptions,
} from '../../src/methods/line_search/methods';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems';
import { mismatch, readJson } from './compare';

type Raw = Record<string, unknown>;
type SmoothProblem = Omit<Problem<Vector>, 'f' | 'grad' | 'hess'> & {
  f: (x: Vector) => number;
  grad: (x: Vector) => Vector;
  hess: (x: Vector) => Matrix;
};

// The custom problems of gen_shared_ports_fixture.py.
const CUSTOM: Record<string, Problem<Vector>> = {
  linear_2d: {
    id: 'linear_2d',
    name: 'linear',
    latex: 'x + 2y',
    dim: 2,
    domain: [
      [-1, 1],
      [-1, 1],
    ],
    f: (x: Vector) => x[0] + 2.0 * x[1],
    grad: () => [1.0, 2.0],
    hess: () => [
      [0, 0],
      [0, 0],
    ],
    x0: [0.0, 0.0],
  },
  rosen_nograd: {
    id: 'rosen_nograd',
    name: 'Rosenbrock (f only)',
    latex: '',
    dim: 2,
    domain: [
      [-2, 2],
      [-1, 3],
    ],
    f: (x: Vector) => (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2,
    x0: [-1.2, 1.0],
  },
};
const problemById = (id: string) => (CUSTOM[id] ?? getProblem(id)) as SmoothProblem;

function run(method: string, problem: unknown, params: Raw): Result {
  const { spec, fn } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

/** Compare a TS Result with a Python result (snake_case JSON) field by field. */
function expectSameResult(got: Result, wantRaw: Raw) {
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
  expect(got.trace.length).toBe(want.trace.length);
  got.trace.forEach((s, k) => {
    const w = want.trace[k];
    const m = mismatch(
      { k: s.k, x: s.x, fun: s.fun, gradNorm: s.gradNorm, stepSize: s.stepSize, info: s.info },
      { k: w.k, x: w.x, fun: w.fun, gradNorm: w.gradNorm, stepSize: w.stepSize, info: w.info },
      undefined,
      `trace[${k}]`,
    );
    expect(m).toBeNull();
  });
  expect(mismatch(got.x, want.x, undefined, 'x')).toBeNull();
  expect(mismatch(got.fun, want.fun, undefined, 'fun')).toBeNull();
  expect(mismatch(got.extra, want.extra, undefined, 'extra')).toBeNull();
}

// ---------------------------------------------------------------------------------------

describe('line_search registry', () => {
  const registry = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  const python = registry.filter((m) => m.family === 'line_search').map(methodSpecFromJson);

  it('registers the five Python demos with identical specs', () => {
    const ts = listMethods('line_search').map((m) => m.spec);
    expect(ts.map((s) => s.id)).toEqual(python.map((s) => s.id).sort());
    for (const want of python) {
      const got = getMethod(want.id).spec;
      // label / tex are TS-only display fields
      const params = got.params.map(({ label: _l, tex: _t, ...p }) => p);
      expect({ ...got, params }).toEqual(want);
    }
  });
});

describe('line_search fixtures, step by step', () => {
  const cases = readJson<Raw[]>('../../src/generated/fixtures/line_search.json').map((c) => ({
    c: fixtureCaseFromJson(c),
    raw: c.result as Raw,
  }));
  cases.forEach(({ c, raw }, i) => {
    it(`${c.method} on ${c.problem} #${i}`, () => {
      expectSameResult(run(c.method, getProblem(c.problem), c.params as Raw), raw);
    });
  });

  it('covers every phase and every zoom interpolation branch', () => {
    const phases = new Set<string>(),
      interps = new Set<string>();
    for (const { c } of cases) {
      for (const s of run(c.method, getProblem(c.problem), c.params as Raw).trace) {
        phases.add(String(s.info.phase));
        if (s.info.interp) interps.add(String(s.info.interp));
      }
    }
    expect([...phases].sort()).toEqual(['backtrack', 'bisect', 'exact', 'expand', 'start', 'zoom']);
    expect([...interps].sort()).toEqual([
      'bisection',
      'cubic',
      'cubic_clamped',
      'quadratic',
      'quadratic_clamped',
    ]);
  });
});

interface Fixture {
  formats: [number, string, string, string][];
  search: {
    kind: string;
    problem: string;
    x: number[];
    p: number[];
    opts: Raw;
    result?: Raw;
    error?: string;
  }[];
  demos: { method: string; problem: string; params: Raw; result?: Raw; error?: string }[];
}
const FIX = readJson<Fixture>('./fixtures/shared_ports_python.json');

describe('search() matches Python', () => {
  FIX.search.forEach((c, i) => {
    it(`${c.kind} on ${c.problem} #${i}${c.error ? ' (ValueError)' : ''}`, () => {
      const prob = problemById(c.problem);
      const o = c.opts;
      const opts: SearchOptions = {};
      if (o.pass_f0g0) {
        opts.f0 = prob.f(c.x);
        opts.g0 = prob.grad(c.x);
      }
      if (o.hess_matrix) opts.hess = prob.hess(c.x);
      if (o.hess_callable) opts.hess = prob.hess;
      if (o.c1 !== undefined) opts.c1 = o.c1 as number;
      if (o.c2 !== undefined) opts.c2 = o.c2 as number;
      if (o.rho !== undefined) opts.rho = o.rho as number;
      if (o.alpha0 !== undefined) opts.alpha0 = o.alpha0 as number;
      if (o.max_iter !== undefined) opts.maxIter = o.max_iter as number;
      if (o.alpha_max !== undefined) opts.alphaMax = o.alpha_max as number;
      if (o.f_err !== undefined) opts.fErr = o.f_err as number;
      const call = () => search(c.kind, prob.f, prob.grad, c.x, c.p, opts);
      if (c.error !== undefined) {
        expect(call).toThrow(LineSearchInputError);
        expect(call).toThrow(c.error);
        return;
      }
      const r = call();
      const w = c.result as Raw;
      expect(r.message).toBe(w.message);
      expect(
        mismatch(
          [r.alpha, r.fNew, r.gNew, r.nFev, r.nGev, r.nHev, r.success, r.trials],
          [w.alpha, w.f_new, w.g_new, w.n_fev, w.n_gev, w.n_hev, w.success, w.trials],
        ),
      ).toBeNull();
    });
  });
});

describe('line-search demos match Python beyond the parity fixtures', () => {
  FIX.demos.forEach((c, i) => {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => {
      const call = () => run(c.method, problemById(c.problem), c.params);
      if (c.error !== undefined) {
        expect(call).toThrow(c.error);
        return;
      }
      expectSameResult(call(), c.result as Raw);
    });
  });
});

// ---------------------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------------------

describe('line_search units', () => {
  it('formatG matches Python format(v, ".6g") / ".3g"', () => {
    const cases: [number, number, string][] = [
      [0.0009765625, 6, '0.000976562'],
      [0.000787243, 6, '0.000787243'],
      [1, 6, '1'],
      [0.25, 6, '0.25'],
      [1e3, 3, '1e+03'],
      [50, 3, '50'],
      [-20816.0, 3, '-2.08e+04'],
      [54227.38, 6, '54227.4'],
      [1e-5, 6, '1e-05'],
      [123456789, 6, '1.23457e+08'],
      [0.1, 6, '0.1'],
      [0, 6, '0'],
      [NaN, 3, 'nan'],
      [-Infinity, 3, '-inf'],
      [1e-320, 6, '9.99989e-321'],
      [5e-324, 6, '4.94066e-324'],
      [9.9999995, 6, '10'],
      [0.00012345675, 6, '0.000123457'],
      [-(2 ** -10), 6, '-0.000976562'],
    ];
    for (const [v, p, want] of cases) expect(formatG(v, p)).toBe(want);
  });

  it('formatG / formatRepr match Python on 500+ awkward floats (ties, subnormals, powers of 2)', () => {
    const bad = FIX.formats.filter(
      ([v, g6, g3, repr]) => formatG(v, 6) !== g6 || formatG(v, 3) !== g3 || formatRepr(v) !== repr,
    );
    expect(bad.slice(0, 5)).toEqual([]);
    expect(FIX.formats.length).toBeGreaterThan(500);
  });

  it('formatRepr matches Python repr(float)', () => {
    const cases: [number, string][] = [
      [1, '1.0'],
      [0.5, '0.5'],
      [1e-5, '1e-05'],
      [1.5e-7, '1.5e-07'],
      [1e16, '1e+16'],
      [123.25, '123.25'],
      [0.0001, '0.0001'],
      [Infinity, 'inf'],
      [NaN, 'nan'],
    ];
    for (const [v, want] of cases) expect(formatRepr(v)).toBe(want);
  });

  it('cubic interpolation recovers the minimizer of a cubic (incl. the removable 0/0 case)', () => {
    // φ(t) = t³ − 3t: φ' = 3t² − 3, minimizer t = 1
    const phi = (t: number) => t ** 3 - 3 * t,
      d = (t: number) => 3 * t * t - 3;
    expect(cubicMinimizer(0, phi(0), d(0), 2, phi(2), d(2))).toBeCloseTo(1, 14);
    expect(cubicMinimizer(2, phi(2), d(2), 0, phi(0), d(0))).toBeCloseTo(1, 14);
    // φ' = 0.75 − 3t² on [0, 1]: the printed eq. (3.59) has a 0/0; the minimizer is −0.5
    const p2 = (t: number) => 0.75 * t - t ** 3,
      d2 = (t: number) => 0.75 - 3 * t * t;
    expect(cubicMinimizer(0, p2(0), d2(0), 1, p2(1), d2(1))).toBeCloseTo(-0.5, 14);
    // no strict local minimizer: φ = t (linear)
    expect(cubicMinimizer(0, 0, 1, 1, 1, 1)).toBeNull();
    expect(cubicMinimizer(1, 0, 1, 1, 1, 1)).toBeNull();
  });

  it('quadratic interpolation and the h² underflow (Python ZeroDivisionError)', () => {
    // q(t) = (t − 0.3)²: q(0) = 0.09, q'(0) = −0.6, q(1) = 0.49 → minimizer 0.3
    expect(quadraticMinimizer(0, 0.09, -0.6, 1, 0.49)).toBeCloseTo(0.3, 15);
    expect(quadraticMinimizer(0, 0, -1, 1, -2)).toBeNull(); // concave: no minimizer
    expect(() => quadraticMinimizer(0, 0, -1, 1e-170, -1e-171)).toThrow(LineSearchZeroDivision);
    expect(() => quadraticMinimizer(0, 0, -1, 1e-170, -1e-171)).toThrow('float division by zero');
  });

  it('search counts f(x) and ∇f(x) only when f0 / g0 are not given', () => {
    const p = getProblem<SmoothProblem>('rosenbrock');
    const x = [-1.2, 1.0];
    const g = p.grad(x);
    const d = g.map((v) => -v);
    const a = search('strong_wolfe', p.f, p.grad, x, d);
    const b = search('strong_wolfe', p.f, p.grad, x, d, { f0: p.f(x), g0: g });
    expect(a.alpha).toBe(b.alpha);
    expect([a.nFev - b.nFev, a.nGev - b.nGev]).toEqual([1, 1]);
    for (const kind of KINDS) {
      const r = search(kind, p.f, p.grad, x, d, {
        hess: p.hess,
        c1: kind === 'goldstein' ? 0.25 : null,
      });
      expect(r.trials.length).toBeGreaterThan(0);
      expect(r.nHev).toBe(kind === 'exact_quadratic' ? 1 : 0);
    }
  });

  it('is offset invariant: f and f + C give the same trials and verdicts', () => {
    const p = getProblem<SmoothProblem>('himmelblau');
    const x = [0.0, 0.0];
    const d = p.grad(x).map((v) => -v);
    const shifted = (z: Vector) => p.f(z) + 1024.0; // exact in float64 for these values
    for (const kind of ['backtracking', 'strong_wolfe', 'weak_wolfe', 'goldstein'] as const) {
      const a = search(kind, p.f, p.grad, x, d);
      const b = search(kind, shifted, p.grad, x, d);
      expect(b.trials.map((t) => t[0])).toEqual(a.trials.map((t) => t[0]));
      expect([b.alpha, b.success]).toEqual([a.alpha, a.success]);
    }
  });

  it('accepts a scalar or [[h]] Hessian for n = 1', () => {
    const f = (x: Vector) => (x[0] - 3) ** 2;
    const g = (x: Vector) => [2 * (x[0] - 3)];
    for (const hess of [2, [[2]], () => 2, () => [[2]]] as const) {
      const r = search('exact_quadratic', f, g, [0], [6], { hess: hess as never });
      expect(r.alpha).toBe(0.5);
      expect(r.fNew).toBe(0);
    }
  });

  it('a Newton demo at a singular Hessian fails without throwing', () => {
    // f = x⁴ + y²: ∇²f(0, 1) = diag(0, 2) is singular, ∇f(0, 1) = (0, 2) ≠ 0
    const quartic: Problem<Vector> = {
      id: 'quartic_y',
      name: 'x⁴ + y²',
      latex: '',
      dim: 2,
      domain: [],
      f: ([a, b]: Vector) => a ** 4 + b ** 2,
      grad: ([a, b]: Vector) => [4 * a ** 3, 2 * b],
      hess: ([a]: Vector) => [
        [12 * a * a, 0],
        [0, 2],
      ],
    };
    for (const id of KINDS) {
      const r = run(id, quartic, { x0: [0, 1], direction: 'newton' });
      expect(r.converged).toBe(false);
      expect(r.message).toBe('the Hessian at x0 is singular: no Newton direction');
      expect(r.trace).toHaveLength(1);
      expect(r.trace[0].info.direction).toEqual([0, 0]);
      expect(Number.isNaN(r.trace[0].info.dphi0)).toBe(true);
      expect([r.nFev, r.nGev, r.nHev]).toEqual([1, 1, 1]);
    }
  });

  it('accepts a bare f(x) (central-difference gradient, counted in nFev)', () => {
    const f = (x: Vector) => (x[0] - 1) ** 2 + 4 * (x[1] + 2) ** 2;
    const r = run('backtracking', f, { x0: [0, 0] });
    expect(r.converged).toBe(true);
    expect(r.nGev).toBe(1);
    expect(r.nFev).toBe(1 + 4 + r.nIter); // f(x0) + 2n for ∇f + one per trial
  });

  it('every demo step carries the documented info keys', () => {
    const base = [
      'accepted',
      'alpha',
      'c1',
      'c2',
      'conditions',
      'direction',
      'dphi',
      'dphi0',
      'interval',
      'phase',
      'phi',
      'phi0',
    ];
    const extra: Record<string, string[]> = {
      backtracking: ['rho'],
      strong_wolfe: ['alpha_hi', 'alpha_lo', 'interp'],
      weak_wolfe: [],
      goldstein: [],
      exact_quadratic: ['model_phi', 'pHp'],
    };
    for (const id of KINDS) {
      const r = run(id, getProblem('rosenbrock'), {});
      for (const s of r.trace)
        expect(Object.keys(s.info).sort()).toEqual([...base, ...extra[id]].sort());
      expect(r.trace.filter((s) => s.info.accepted)).toHaveLength(r.converged ? 1 : 0);
    }
  });
});
