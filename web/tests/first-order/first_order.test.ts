/**
 * numopt.unconstrained.first_order — TS port checks beyond the parity harness:
 *   - the 13 registered specs equal registry.json (Python fields);
 *   - every unconstrained fixture of these methods matches step by step (x, fun, ‖∇f‖, α, every
 *     info key) with the same nIter, counts, converged flag and message;
 *   - more Python runs (tests/first-order/fixtures, from gen_first_order_fixture.py): failure
 *     paths, max_iter, other problems and start points, n-D problems, a bare callable without ∇f;
 *   - the ValueError messages of invalid input;
 *   - focused unit tests for what the Python data cannot show.
 *
 * It imports the port directly (not the `src/methods` glob), so it does not depend on other ports.
 */
import { describe, expect, it } from 'vitest';
import {
  fixtureCaseFromJson,
  methodSpecFromJson,
  resultFromJson,
  stepFromJson,
} from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Matrix, MethodFn, Problem, Result, Step, Vector } from '../../src/core/types';
import {
  FirstOrderInputError,
  barzilaiBorwein,
  gdAlpha0,
  bbSigma,
  bbSteps,
  frexpExp,
  ldexp,
  type FirstOrderProblem,
} from '../../src/methods/unconstrained/first_order';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems/unconstrained';
import { CANONICAL, sameText } from '../fixtures/platform';
import { mismatch, readJson, type Tol } from '../shared-ports/compare';

type Raw = Record<string, unknown>;

const IDS = [
  'gradient_descent',
  'barzilai_borwein',
  'momentum',
  'nesterov',
  'adagrad',
  'rmsprop',
  'adadelta',
  'adam',
  'adamw',
  'adamax',
  'nadam',
  'amsgrad',
  'coordinate_descent',
];

/** Agreement of the early steps (the parity rule asks 1e-8 on the first 10 iterates). */
const STEP_TOL: Tol = { rtol: 1e-9, atol: 1e-12 };
/** Final x and f (parity rule: 1e-6 relative). */
const FINAL_TOL: Tol = { rtol: 1e-6, atol: 1e-10 };

// The custom problems of gen_first_order_fixture.py.
const LINEAR: Problem<Vector> = {
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
};
const rosen = (x: Vector) => (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2;

/** `_tiny_gradient(c)`: f = a·(x₁ + x₂) + ½c‖x‖², a = 1e-148 (yᵀy underflows in BB). */
function tinyGradient(c: number): Problem<Vector> {
  const a = 1e-148;
  return {
    id: `tiny_gradient_${c}`,
    name: 'tiny gradient',
    latex: '',
    dim: 2,
    domain: [
      [-1, 1],
      [-1, 1],
    ],
    f: (x: Vector) => a * (x[0] + x[1]) + 0.5 * c * (x[0] * x[0] + x[1] * x[1]),
    grad: (x: Vector) => x.map((t) => a + c * t),
    x0: [0.0, 0.0],
  };
}

/** `_scaled(pid, e)`: 2ᵉ·f with ∇f and ∇²f scaled to match. */
function scaledProblem(pid: string, e: number): Problem<Vector> {
  const p = getProblem<Problem<Vector>>(pid);
  const c = 2 ** e;
  const { grad, hess } = p;
  return {
    ...p,
    f: (x: Vector) => c * (p.f(x) as number),
    grad: grad && ((x: Vector) => (grad(x) as Vector).map((t) => c * t)),
    hess: hess && ((x: Vector) => (hess(x) as Matrix).map((row) => row.map((t) => c * t))),
  };
}

function problemFor(id: string): FirstOrderProblem {
  if (id === 'linear_2d') return LINEAR;
  if (id === 'rosen_nograd') return rosen;
  if (id.startsWith('tiny_gradient_')) return tinyGradient(Number(id.slice(14)));
  if (id.includes('_x2^')) {
    const [base, e] = id.split('_x2^');
    return scaledProblem(base, Number(e));
  }
  return getProblem<Problem<Vector>>(id);
}

/** Seeded uniform [0, 1) generator (mulberry32), so the property tests are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `float(a @ b)` for small n: a plain left-to-right sum. */
const dot = (a: readonly number[], b: readonly number[]) =>
  a.reduce((acc, t, i) => acc + t * b[i], 0);

function run(method: string, problem: string, params: Record<string, unknown>): Result {
  const { fn } = getMethod<FirstOrderProblem>(method);
  return (fn as MethodFn<FirstOrderProblem>)(problemFor(problem), params as never);
}

/** A Step as plain data, NaN → null (as Python exports it). */
function plain(step: Step): Raw {
  return JSON.parse(
    JSON.stringify(step, (_k, v: unknown) =>
      typeof v === 'number' && !Number.isFinite(v) ? (Number.isNaN(v) ? null : String(v)) : v,
    ),
    (_k, v: unknown) => (v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v),
  ) as Raw;
}

function expectStep(got: Step, want: Step, tol: Tol, where: string) {
  const m = mismatch(plain(got), plain(want), tol, `${where}.k${want.k}`);
  expect(m).toBeNull();
}

/**
 * NumPy's `g @ g` (OpenBLAS ddot) and the plain loop of the port can round the last bit
 * differently for n > 2. Gradient descent on the 10-D Rosenbrock amplifies that ulp about 2.5× per
 * iteration (the cancellation in α₀ = 2(f_{k−2} − f_{k−1})/‖g‖²): the first 12 steps agree to
 * 1e-12, x₅₀ only to 3e-5 and α₅₀ to 2e-3. The counts, nIter and message still agree exactly.
 */
const CHAOTIC: Record<string, Tol> = {
  'gradient_descent/rosenbrock_nd': { rtol: 1e-2, atol: 1e-6 },
};

/**
 * Runs whose Armijo backtracking accepts a trial one step earlier or later on another platform
 * (tests/fixtures/platform.ts), so n_fev differs (measured: 355 against 353 with the generic
 * ARMv8 and Neoverse-N1 OpenBLAS kernels) and the path leaves Python's. nIter, the flag and the
 * first steps still agree; the final state is compared only when the run converged (after 300
 * steps that stop at max_iter it is 0.026 away).
 */
const PLATFORM_CHAOTIC = new Set(['gradient_descent/rosenbrock {"max_iter":300}']);

function expectTotals(got: Result, want: Result, finalTol: Tol = FINAL_TOL, chaotic = false) {
  expect(got.method).toBe(want.method);
  expect(got.nIter).toBe(want.nIter);
  expect(got.converged).toBe(want.converged);
  if (CANONICAL) expect(got.message).toBe(want.message);
  else expect(sameText(got.message, want.message), `${got.message} vs ${want.message}`).toBe(true);
  if (chaotic) {
    if (!want.converged) return;
  } else expect([got.nFev, got.nGev, got.nHev]).toEqual([want.nFev, want.nGev, want.nHev]);
  expect(mismatch(got.x, want.x, finalTol)).toBeNull();
  expect(mismatch(got.fun, want.fun, finalTol)).toBeNull();
}

describe('first_order specs', () => {
  it('registers the 13 Python methods with identical specs', () => {
    const registry = readJson<Raw[]>('../../src/generated/registry.json');
    for (const id of IDS) {
      const raw = registry.find((m) => m.id === id);
      expect(raw, id).toBeDefined();
      const want = methodSpecFromJson(raw!);
      const got = getMethod(id).spec;
      const { params, ...rest } = got;
      const { params: wantParams, ...wantRest } = want;
      expect(rest).toEqual(wantRest);
      // label / tex are TS-only display fields.
      expect(params.map(({ label: _l, tex: _t, ...p }) => p)).toEqual(wantParams);
    }
  });

  it('gives every method a MethodCard doc whose quantities name Step fields', () => {
    for (const id of IDS) {
      const { doc } = getMethod(id);
      expect(doc?.rule, id).toBeTruthy();
      for (const q of doc?.quantities ?? [])
        expect(q.key).toMatch(/^(stepSize|gradNorm|fun|info\.[a-z_0-9A-Z]+)$/);
    }
  });
});

describe('first_order parity fixtures, step by step', () => {
  const cases = readJson<Raw[]>('../../src/generated/fixtures/unconstrained.json')
    .map(fixtureCaseFromJson)
    .filter((c) => IDS.includes(c.method));

  it('covers all 13 methods and every gradient-descent step rule', () => {
    expect(new Set(cases.map((c) => c.method)).size).toBe(13);
    const rules = cases
      .filter((c) => c.method === 'gradient_descent')
      .map((c) => c.params.step_rule ?? 'backtracking');
    expect(new Set(rules)).toEqual(
      new Set(['fixed', 'backtracking', 'strong_wolfe', 'exact_quadratic']),
    );
  });

  cases.forEach((c, i) => {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => {
      const got = run(c.method, c.problem, c.params);
      const want = c.result;
      expectTotals(got, want);
      expect(got.trace.length).toBe(want.trace.length);
      // Parity rule: the first 10 iterates within 1e-8.
      for (let k = 0; k < Math.min(10, want.trace.length); k++)
        expect(mismatch(got.trace[k].x, want.trace[k].x, { rtol: 1e-8, atol: 1e-8 })).toBeNull();
      // Every step and every info key: tight early, then within the final tolerance.
      want.trace.forEach((w, k) =>
        expectStep(got.trace[k], w, k < 20 ? STEP_TOL : FINAL_TOL, `${c.method}`),
      );
    });
  });
});

interface RunCase {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  head: Raw[];
  last: Raw;
  n_steps: number;
  [key: string]: unknown;
}
interface ErrorCase {
  method: string;
  problem: string;
  params: Record<string, unknown>;
  error: string;
}
// readJson resolves paths relative to tests/shared-ports/compare.ts.
const FIX = readJson<{ runs: RunCase[]; errors: ErrorCase[] }>(
  '../first-order/fixtures/first_order_python.json',
);

describe('first_order matches Python beyond the parity fixtures', () => {
  FIX.runs.forEach((c, i) => {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => {
      const got = run(c.method, c.problem, c.params);
      const want = resultFromJson({ ...c, trace: [] });
      const finalTol = CHAOTIC[`${c.method}/${c.problem}`] ?? FINAL_TOL;
      const chaotic =
        !CANONICAL && PLATFORM_CHAOTIC.has(`${c.method}/${c.problem} ${JSON.stringify(c.params)}`);
      expectTotals(got, want, finalTol, chaotic);
      expect(got.trace.length).toBe(c.n_steps);
      c.head.forEach((raw, k) => expectStep(got.trace[k], stepFromJson(raw), STEP_TOL, c.method));
      const last = got.trace[got.trace.length - 1];
      if (CANONICAL) expectStep(last, stepFromJson(c.last), finalTol, c.method);
      else if (!(chaotic && !want.converged)) {
        // Another platform's dump (tests/fixtures/platform.ts): the last step's diagnostics (α
        // after 100 BB steps on the 20-D quadratic: 1.3e-6 apart on x86-64, from BLAS dot
        // products) carry its rounding; its state is held to the final tolerance.
        const want = stepFromJson(c.last);
        expect(
          mismatch({ x: last.x, fun: last.fun }, { x: want.x, fun: want.fun }, finalTol),
        ).toBeNull();
      }
    });
  });

  FIX.errors.forEach((c, i) => {
    it(`${c.method} ${JSON.stringify(c.params)} throws ValueError #${i}`, () => {
      const call = () => run(c.method, c.problem, c.params);
      expect(call).toThrow(FirstOrderInputError);
      expect(call).toThrow(c.error);
      try {
        call();
      } catch (e) {
        expect((e as Error).name).toBe('ValueError');
        expect((e as Error).message).toBe(c.error);
      }
    });
  });
});

describe('first_order unit tests', () => {
  const quadBowl = () => getProblem<Problem<Vector>>('quadratic_bowl');

  it('records the documented info keys at every step', () => {
    const extra: Record<string, string[]> = {
      gradient_descent: ['trials', 'alpha0', 'pHp'],
      barzilai_borwein: ['alpha_bb', 'bb1', 'bb2', 'reset', 'f_ref', 'trials'],
      momentum: ['velocity'],
      nesterov: ['velocity', 'lookahead', 'grad_lookahead'],
      adagrad: ['v', 'lr_eff'],
      rmsprop: ['v', 'lr_eff'],
      adadelta: ['v', 'u', 'lr_eff'],
      adam: ['m', 'v', 'm_hat', 'v_hat', 'lr_eff'],
      adamw: ['m', 'v', 'm_hat', 'v_hat', 'lr_eff', 'decay'],
      adamax: ['m', 'u', 'lr_eff'],
      nadam: ['m', 'v', 'm_bar', 'v_hat', 'lr_eff'],
      amsgrad: ['m', 'v', 'v_max', 'lr_eff'],
      coordinate_descent: ['coordinate', 'sweep', 'curvature', 'newton', 'trials'],
    };
    for (const id of IDS) {
      const r = run(id, 'quadratic_bowl', { max_iter: 5 });
      const want = ['grad', 'direction', 'alpha', ...extra[id]].sort();
      for (const s of r.trace) expect(Object.keys(s.info).sort(), id).toEqual(want);
      expect(r.trace[0].stepSize).toBeNull();
      expect(r.trace[0].info.direction).toBeNull();
      expect(r.nIter).toBe(r.trace[r.trace.length - 1].k);
      for (const s of r.trace.slice(1)) expect(s.info.alpha).toBe(s.stepSize);
    }
  });

  it('every step satisfies x_k = x_{k−1} + α_k p_k', () => {
    for (const id of IDS) {
      const r = run(id, 'himmelblau', { max_iter: 30 });
      for (let k = 1; k < r.trace.length; k++) {
        const prev = r.trace[k - 1].x as Vector;
        const x = r.trace[k].x as Vector;
        const p = r.trace[k].info.direction as Vector;
        const a = r.trace[k].stepSize!;
        x.forEach((xi, i) => expect(xi).toBeCloseTo(prev[i] + a * p[i], 10));
      }
    }
  });

  it('does not mutate x0 or the trace after the run', () => {
    const x0 = [-1.2, 1.0];
    const r = run('adam', 'rosenbrock', { x0, max_iter: 20 });
    expect(x0).toEqual([-1.2, 1.0]);
    expect(r.trace[0].x).toEqual([-1.2, 1.0]);
    (r.x as Vector)[0] = 99;
    expect((r.trace[r.trace.length - 1].x as Vector)[0]).not.toBe(99);
  });

  it('nesterov reuses the look-ahead gradient when μv = 0 (k = 1)', () => {
    const r = run('nesterov', 'quadratic_bowl', { max_iter: 3 });
    // k = 1: one gradient (at x₁); k = 2, 3: two each; plus ∇f(x₀).
    expect(r.nGev).toBe(1 + 1 + 2 + 2);
    expect(r.trace[1].info.lookahead).toEqual(r.trace[0].x);
  });

  it('gdAlpha0 takes the larger of the N&W guesses (3.60, 3.61), else a unit move', () => {
    expect(gdAlpha0(4.0, 1.0, null, NaN, NaN)).toBe(0.5);
    // α_{3.61} = 2(3 − 1)/4 = 1, α_{3.60} = 0.1·8/4 = 0.2.
    expect(gdAlpha0(4.0, 1.0, 0.1, 8.0, 3.0)).toBe(1.0);
    // f did not decrease: α_{3.61} ≤ 0 is skipped.
    expect(gdAlpha0(4.0, 1.0, 0.1, 8.0, 1.0)).toBeCloseTo(0.2, 15);
    expect(gdAlpha0(4.0, 1.0, 0.0, 8.0, 1.0)).toBe(0.5);
  });

  it('bbSteps equals the unscaled quotients on well-scaled data, bit for bit', () => {
    // The fixtures were made with (sᵀs/sᵀy, sᵀy/yᵀy); the power-of-2 scaling must not move them.
    const rng = mulberry32(20261006);
    for (let trial = 0; trial < 2000; trial++) {
      const n = 1 + Math.floor(rng() * 4);
      const s = Array.from({ length: n }, () => (rng() * 4 - 2) * 2 ** Math.floor(rng() * 40 - 20));
      const y = Array.from({ length: n }, () => (rng() * 4 - 2) * 2 ** Math.floor(rng() * 40 - 20));
      const sy = dot(s, y);
      const want = sy > 0 ? [dot(s, s) / sy, sy / dot(y, y)] : [null, null];
      expect(bbSteps(s, y)).toEqual(want);
    }
  });

  it('bbSteps(2ᵃs, 2ᵇy) = 2^(a−b)·bbSteps(s, y) exactly for a, b in [−1000, 1000]', () => {
    // Same property as the Python hypothesis test: unscaled, the products under- or overflow for
    // |a|, |b| ≳ 500; scaled, only a quotient outside the float range leaves the exact relation
    // (then it is +Infinity or rounds toward 0, never NaN or null).
    const rng = mulberry32(7);
    for (let trial = 0; trial < 2000; trial++) {
      const n = 1 + Math.floor(rng() * 4);
      const s = Array.from({ length: n }, () => (0.5 + 1.5 * rng()) * (rng() < 0.5 ? -1 : 1));
      const y = s.map((t) => t * (0.1 + 9.9 * rng())); // sᵢyᵢ > 0
      const [bb1, bb2] = bbSteps(s, y) as [number, number];
      const a = Math.floor(rng() * 2001) - 1000;
      const b = Math.floor(rng() * 2001) - 1000;
      const [big1, big2] = bbSteps(
        s.map((t) => ldexp(t, a)),
        y.map((t) => ldexp(t, b)),
      );
      const want1 = ldexp(bb1, a - b);
      const want2 = ldexp(bb2, a - b);
      const normal = (v: number) => v > 1e-300 && v < 1e300;
      if (normal(want1) && normal(want2)) expect([big1, big2]).toEqual([want1, want2]);
      else {
        expect(big1).toBeGreaterThanOrEqual(0);
        expect(big2).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('bbSteps is undefined when sᵀy ≤ 0 (also y = 0)', () => {
    expect(bbSteps([1, 0], [0, 0])).toEqual([null, null]);
    expect(bbSteps([1, 0], [-1, 0])).toEqual([null, null]);
    expect(bbSteps([1, 0], [0, 1])).toEqual([null, null]);
  });

  it('ldexp rounds once (also into the subnormal range) and frexpExp is math.frexp', () => {
    expect(ldexp(1.5, -1074)).toBe(2 ** -1074 * 2); // 1.5·2⁻¹⁰⁷⁴ rounds to even: 2·2⁻¹⁰⁷⁴
    expect(ldexp(1.25, -1073)).toBe(2 ** -1073); // ties to even, one rounding
    expect(ldexp(1 + 2 ** -52, -1060)).toBe(2 ** -1060); // the low bit is below the subnormal ulp
    expect(ldexp(0.75, 1024)).toBe(1.5 * 2 ** 1023);
    expect(ldexp(1, 1024)).toBe(Infinity);
    expect(ldexp(2 ** -1074, 2097)).toBe(2 ** 1023); // two steps up, exact
    expect(ldexp(2 ** -1074, 2098)).toBe(Infinity);
    expect(ldexp(2 ** 1023, -2097)).toBe(2 ** -1074); // two steps down, exact
    expect(ldexp(3, 0)).toBe(3);
    expect(frexpExp(1)).toBe(1); // 1 = 0.5·2¹
    expect(frexpExp(0.75)).toBe(0);
    expect(frexpExp(-8)).toBe(4);
    expect(frexpExp(2 ** -1074)).toBe(-1073);
    expect(frexpExp(Number.MAX_VALUE)).toBe(1024);
    expect([frexpExp(0), frexpExp(Infinity), frexpExp(NaN)]).toEqual([0, 0, 0]);
  });

  it('bbSigma is the safeguarded quadratic-interpolation factor', () => {
    expect(bbSigma(1.0, Infinity, 0.0, 1.0)).toBe(0.1);
    expect(bbSigma(1.0, -5.0, 0.0, 1.0)).toBe(0.5); // curvature ≤ 0
    // φ(t) = t² − t: c = 1, t* = 1/2, σ = 0.5.
    expect(bbSigma(1.0, 0.0, 0.0, 1.0)).toBe(0.5);
    // φ(t) = 10t² − t: c = 10, t* = 0.05, σ clipped to 0.1.
    expect(bbSigma(1.0, 9.0, 0.0, 1.0)).toBe(0.1);
    // The Python cases of test_bb_sigma_interpolant_minimizer (α = 1e-200: α² underflows).
    for (const [alpha, want] of [
      [2.0, 0.25],
      [1.25, 0.4],
      [10.0, 0.1],
      [0.5, 0.5],
      [1e-200, 0.5],
    ])
      expect(bbSigma(alpha, 1.0 - alpha + alpha * alpha, 1.0, 1.0)).toBe(want);
    // 2cα underflows to 0 with c > 0: σ = gᵀg/(2cα) exceeds every float and is clipped to σ₂.
    expect(bbSigma(5e-324, 1e-300, 1e-300, 1e-300)).toBe(0.5);
  });

  it('bbSigma always lies in [σ₁, σ₂] = [0.1, 0.5]', () => {
    const rng = mulberry32(3);
    const pick = (lo: number, hi: number) =>
      10 ** (lo + (hi - lo) * rng()) * (rng() < 0.5 ? 1 : -1);
    for (let trial = 0; trial < 5000; trial++) {
      const alpha = Math.abs(pick(-323, 10));
      const fAlpha = trial % 50 === 0 ? (trial % 100 === 0 ? Infinity : NaN) : pick(-300, 300);
      const sigma = bbSigma(alpha, fAlpha, pick(-300, 300), Math.abs(pick(-300, 300)));
      expect(sigma).toBeGreaterThanOrEqual(0.1);
      expect(sigma).toBeLessThanOrEqual(0.5);
    }
  });

  it('barzilai_borwein keeps BB values defined when yᵀy underflows (scaled inner products)', () => {
    // f = x₂ + ½h·x₁², h = 1e-15, x₀ = (1e-135, 0): ∇f = (h·x₁, 1). The first (unit) step gives
    // s = (−1.07e-150, −1), y = (−1.09e-165, 0): sᵀy ≈ 1e-315 > 0, but the unscaled yᵀy ≈ 1e-330
    // underflows to 0 (Python raised ZeroDivisionError there before `_bb_steps`). With s and y
    // scaled by powers of 2, sᵀy/yᵀy is the exact quotient rounded once (Python and Fraction give
    // 985162418487296), and sᵀs/sᵀy ≈ 1e315 overflows to Infinity. Both lie outside
    // [1e-10, 1e10], so the step is reset as documented.
    const h = 1e-15;
    const tiny: Problem<Vector> = {
      id: 'tiny_y',
      name: 'tiny y',
      latex: '',
      dim: 2,
      domain: [],
      f: (x) => x[1] + 0.5 * h * x[0] ** 2,
      grad: (x) => [h * x[0], 1.0],
      x0: [1e-135, 0.0],
    };
    for (const variant of ['bb1', 'bb2']) {
      const r = barzilaiBorwein(tiny, { variant, max_iter: 3 });
      expect(r.trace.length).toBe(4);
      expect(r.trace[2].info.bb1).toBe(Infinity);
      expect(r.trace[2].info.bb2).toBe(985162418487296.0);
      expect(r.trace[2].info.reset).toBe(true);
      expect(r.trace[2].info.alpha_bb).toBe(1.0);
    }
  });

  it('a bare callable without a gradient needs x0 and counts 2n f calls per gradient', () => {
    expect(() => run('momentum', 'rosen_nograd', {})).toThrow(
      'custom: no starting point given and the problem has no default x0',
    );
    const r = run('momentum', 'rosen_nograd', { x0: [-1.2, 1.0], max_iter: 4 });
    expect(r.nGev).toBe(5);
    expect(r.nFev).toBe(5 + 2 * 2 * 5);
  });

  it('stops at once when f or ∇f is not finite at x0', () => {
    const p: Problem<Vector> = { ...quadBowl(), id: 'nan_start', f: () => NaN };
    const r = barzilaiBorwein(p, {});
    expect(r.converged).toBe(false);
    expect(r.message).toBe('f(x0) or ∇f(x0) is not finite');
    expect(r.nIter).toBe(0);
  });
});
