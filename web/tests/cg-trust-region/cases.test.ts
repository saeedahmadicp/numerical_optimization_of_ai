/**
 * Conjugate gradient + trust region: Python reference runs beyond the parity fixtures
 * (tests/cg-trust-region/fixtures/cg_tr_python.json, from gen_cg_tr_fixture.py): n-D problems,
 * the exact line search, failure paths, indefinite Hessians (dogleg fallback, Steihaug negative
 * curvature, the hard case), finite-difference derivatives, 1-D problems, the radius clamp and
 * ValueError messages. Plus focused unit tests of the numerical helpers.
 */
import { describe, expect, it } from 'vitest';
import { resultFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Matrix, Problem, Result, Step, Vector } from '../../src/core/types';
import { betaOf, frexpExp, ldexp, norm2 } from '../../src/methods/unconstrained/conjugate_gradient';
import { eigh } from '../../src/methods/unconstrained/trust_region';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems/unconstrained';
import { mismatch, readJson, type Tol } from '../shared-ports/compare';
import { STEP_TOL, run } from './helpers';

type Raw = Record<string, unknown>;

const rosen = (x: Vector) => (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2;
const box: [number, number][] = [
  [-2, 2],
  [-2, 2],
];
const diag = (a: number, b: number): Matrix => [
  [a, 0.0],
  [0.0, b],
];

// The custom problems of gen_cg_tr_fixture.py.
const CUSTOM: Record<string, Problem<Vector> | Problem<number>> = {
  rosen_nograd: {
    id: 'rosen_nograd',
    name: '',
    latex: '',
    f: rosen,
    dim: 2,
    domain: box,
    x0: [-1.2, 1.0],
  },
  saddle: {
    id: 'saddle',
    name: '',
    latex: '',
    f: (x: Vector) => x[0] ** 2 - x[1] ** 2,
    grad: (x: Vector) => [2.0 * x[0], -2.0 * x[1]],
    hess: () => diag(2.0, -2.0),
    dim: 2,
    domain: box,
    x0: [1.0, 0.0],
  },
  quartic_1d: {
    id: 'quartic_1d',
    name: '',
    latex: '',
    f: (x: number) => (x - 2.0) ** 4 + x * x,
    grad: (x: number) => 4.0 * (x - 2.0) ** 3 + 2.0 * x,
    hess: (x: number) => 12.0 * (x - 2.0) ** 2 + 2.0,
    dim: 1,
    domain: [-1.0, 4.0],
    x0: -1.0,
  },
  log_neg: {
    id: 'log_neg',
    name: '',
    latex: '',
    f: (x: Vector) => (x[0] > 0 ? Math.log(x[0]) + x[1] ** 2 : NaN),
    grad: (x: Vector) => [1.0 / x[0], 2.0 * x[1]],
    hess: (x: Vector) => diag(-1.0 / x[0] ** 2, 2.0),
    dim: 2,
    domain: box,
    x0: [-1.0, 0.5],
  },
  tiny_bowl: {
    id: 'tiny_bowl',
    name: '',
    latex: '',
    f: (x: Vector) => 1e-300 * (x[0] ** 2 + x[1] ** 2),
    grad: (x: Vector) => [2e-300 * x[0], 2e-300 * x[1]],
    hess: () => diag(2e-300, 2e-300),
    dim: 2,
    domain: box,
    x0: [1.0, 1.0],
  },
  huge_bowl: {
    id: 'huge_bowl',
    name: '',
    latex: '',
    f: (x: Vector) => 1e300 * (x[0] ** 2 + x[1] ** 2),
    grad: (x: Vector) => [2e300 * x[0], 2e300 * x[1]],
    hess: () => diag(2e300, 2e300),
    dim: 2,
    domain: box,
    x0: [1.0, 1.0],
  },
  only_at_start: {
    id: 'only_at_start',
    name: '',
    latex: '',
    f: (x: Vector) => (x[0] === 1.0 && x[1] === 1.0 ? 2.0 : Infinity),
    grad: (x: Vector) => [2.0 * x[0], 2.0 * x[1]],
    hess: () => diag(2.0, 2.0),
    dim: 2,
    domain: box,
    x0: [1.0, 1.0],
  },
};

/** The crafted quartic problems of gen_cg_tr_fixture.py (data from the JSON, same loops). */
interface QuarticData {
  A: Matrix;
  c: Vector;
  w: Vector;
  v: Vector;
  x0: Vector;
}
function quarticExp(id: string, { A, c, w, v, x0 }: QuarticData): Problem<Vector> {
  const n = c.length;
  return {
    id,
    name: id,
    latex: '',
    dim: n,
    domain: [],
    x0,
    f: (x: Vector) => {
      const d = x.map((t, i) => t - c[i]);
      let s = 0.0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += d[i] * A[i][j] * d[j];
      let r = 0.5 * s;
      for (let i = 0; i < n; i++) r += w[i] * (x[i] * x[i] * x[i] * x[i]);
      for (let i = 0; i < n; i++) r += v[i] * Math.exp(x[i]);
      return r;
    },
    grad: (x: Vector) => {
      const d = x.map((t, i) => t - c[i]);
      return x.map((xi, i) => {
        let s = 0.0;
        for (let j = 0; j < n; j++) s += A[i][j] * d[j];
        return s + 4.0 * w[i] * (xi * xi * xi) + v[i] * Math.exp(xi);
      });
    },
    hess: (x: Vector) =>
      A.map((row, i) =>
        row.map((a, j) =>
          i === j ? a + (12.0 * w[i] * (x[i] * x[i]) + v[i] * Math.exp(x[i])) : a,
        ),
      ),
  };
}

interface RunCase {
  method: string;
  problem: string;
  params: Raw;
  result: Raw & { trace_len: number; trace_head: Raw[]; trace_tail: Raw[] };
}
interface ErrorCase {
  method: string;
  problem: string;
  params: Raw;
  error: string;
}
const FIX = readJson<{
  runs: RunCase[];
  errors: ErrorCase[];
  eigh2: { A: Matrix; w: number[]; Q: Matrix }[];
  quartic: Record<string, QuarticData>;
}>('../cg-trust-region/fixtures/cg_tr_python.json');
for (const [id, data] of Object.entries(FIX.quartic)) CUSTOM[id] = quarticExp(id, data);
const problemById = (id: string) => CUSTOM[id] ?? getProblem(id);

/**
 * Long replays drift by rounding (BLAS summation order, Jacobi vs LAPACK eigh, libm pow): the
 * first ten steps must agree to STEP_TOL in every field, later iterates and the final point to
 * LATE (the parity rule asks 1e-8 for the first ten iterates and 1e-6 for the final x).
 */
const LATE: Tol = { rtol: 1e-6, atol: 1e-9 };

const EPS = 2.220446049250313e-16;

/**
 * The fields of a step to compare. Where Python's predicted reduction is at the rounding level of
 * f (≤ 10⁴ε·max(1, |f|)), the measured reduction and ρ are rounding noise (a last-bit difference
 * in f, e.g. from libm sin, moves them by O(1)), so they are left out.
 */
function stepView(s: Step, ref: Step = s) {
  const info = { ...s.info };
  const pred = ref.info.predicted;
  if (typeof pred === 'number' && pred <= 1e4 * EPS * Math.max(1, Math.abs(ref.fun ?? 0))) {
    delete info.actual;
    delete info.rho;
  }
  return { k: s.k, x: s.x, fun: s.fun, gradNorm: s.gradNorm, stepSize: s.stepSize, info };
}

/** Python's message with the rounding-level numbers of a converged run masked. */
const maskTol = (m: string) => m.replace(/‖∇f‖∞ = \S+ ≤ gtol/, '‖∇f‖∞ = … ≤ gtol');

function expectSameRun(got: Result, c: RunCase, early: Tol = STEP_TOL) {
  const { trace_len: len, trace_head: head, trace_tail: tail, ...rest } = c.result;
  const want = resultFromJson({ ...rest, trace: [...head, ...tail] });
  expect(got.method).toBe(want.method);
  const nHead = head.length;
  want.trace.forEach((w, i) => {
    const k = i < nHead ? i : len - (want.trace.length - i);
    // Late steps: only the iterate (ρ, ‖∇f‖, … near a minimizer are rounding noise).
    const m =
      k < 10
        ? mismatch(stepView(got.trace[k], w), stepView(w), early, `trace[${k}]`)
        : mismatch(
            { k: got.trace[k].k, x: got.trace[k].x },
            { k: w.k, x: w.x },
            LATE,
            `trace[${k}]`,
          );
    expect(m).toBeNull();
  });
  expect(got.trace.length).toBe(len);
  expect([got.nIter, got.nFev, got.nGev, got.nHev]).toEqual([
    want.nIter,
    want.nFev,
    want.nGev,
    want.nHev,
  ]);
  expect(got.converged).toBe(want.converged);
  // ‖∇f‖∞ at convergence is at the rounding level after a long run or when it is ≲ 1e-12
  // (‖∇f‖ of an ulp-level iterate); every other message must match exactly.
  const ginf = Number(/‖∇f‖∞ = (\S+) ≤ gtol/.exec(want.message)?.[1] ?? NaN);
  if (want.converged && (len > 10 || ginf < 1e-12))
    expect(maskTol(got.message)).toBe(maskTol(want.message));
  else expect(got.message).toBe(want.message);
  expect(mismatch(got.x, want.x, LATE, 'x')).toBeNull();
  expect(mismatch(got.fun, want.fun, LATE, 'fun')).toBeNull();
  expect(mismatch(got.extra, want.extra, LATE, 'extra')).toBeNull();
}

/**
 * Replays that amplify rounding: n-D problems (NumPy sums 10- and 20-term dot products with
 * BLAS kernels whose summation order differs from a plain loop) and central-difference Hessians
 * (an ulp in x moves a Hessian entry by ≈ ε|f|/h² ≈ 1e-9 relative). For them the first ten
 * steps must agree to 1e-8 (the parity rule), the run must end the same way and at the same point
 * to 1e-4, and nIter within 30 % (CG on the 20-D quadratic loses conjugacy at a rate that
 * depends on those last bits).
 */
const SENSITIVE = new Set(['rosenbrock_nd', 'quadratic_nd', 'rosen_nograd', 'quartic_nd4']);
const PARITY: Tol = { rtol: 1e-8, atol: 1e-10 };

function expectSimilarRun(got: Result, c: RunCase) {
  const { trace_head: head, ...rest } = c.result;
  const want = resultFromJson({ ...rest, trace: head });
  for (let k = 0; k < Math.min(10, head.length); k++)
    expect(
      mismatch(
        { x: got.trace[k].x, fun: got.trace[k].fun },
        { x: want.trace[k].x, fun: want.trace[k].fun },
        PARITY,
        `trace[${k}]`,
      ),
    ).toBeNull();
  expect(got.converged).toBe(want.converged);
  expect(maskTol(got.message)).toBe(maskTol(want.message));
  expect(Math.abs(got.nIter - want.nIter)).toBeLessThanOrEqual(Math.max(2, 0.3 * want.nIter));
  if (want.converged) expect(mismatch(got.x, want.x, { rtol: 1e-4, atol: 1e-4 }, 'x')).toBeNull();
}

describe('Python reference runs beyond the fixtures', () => {
  FIX.runs.forEach((c, i) => {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => {
      const got = run(c.method, problemById(c.problem), c.params);
      if (SENSITIVE.has(c.problem)) expectSimilarRun(got, c);
      else expectSameRun(got, c);
    });
  });

  it('reaches every restart kind, solver branch and failure message', () => {
    const restarts = new Set<string>(),
      notes = new Set<string>(),
      terminations = new Set<string>();
    let hard = 0;
    for (const c of FIX.runs) {
      const r = run(c.method, problemById(c.problem), c.params);
      for (const s of r.trace) {
        if (typeof s.info.restart === 'string') restarts.add(s.info.restart);
        if (typeof s.info.note === 'string') notes.add(s.info.note.split(':')[0]);
        if (typeof s.info.termination === 'string') terminations.add(s.info.termination);
        if (s.info.hard_case === true) hard++;
      }
    }
    expect([...restarts].sort()).toEqual([
      'breakdown',
      'initial',
      'not_descent',
      'periodic',
      'powell',
    ]);
    expect([...terminations].sort()).toEqual(['boundary', 'negative_curvature', 'residual']);
    expect([...notes].sort()).toEqual([
      'radius0 = 50 > max_radius = 2',
      '∇²f is not positive definite',
    ]);
    expect(hard).toBeGreaterThan(0);
  });
});

describe('ValueError messages', () => {
  FIX.errors.forEach((c) => {
    it(`${c.method} ${JSON.stringify(c.params)}`, () => {
      expect(() => run(c.method, problemById(c.problem), c.params)).toThrow(c.error);
      try {
        run(c.method, problemById(c.problem), c.params);
      } catch (e) {
        expect((e as Error).name).toBe('ValueError');
        expect((e as Error).message).toBe(c.error);
      }
    });
  });
});

describe('numerical helpers', () => {
  it('frexp / ldexp match Python math.frexp / math.ldexp', () => {
    expect(frexpExp(1.0)).toBe(1);
    expect(frexpExp(0.5)).toBe(0);
    expect(frexpExp(3.0)).toBe(2);
    expect(frexpExp(5e-324)).toBe(-1073);
    expect(frexpExp(1.7976931348623157e308)).toBe(1024);
    expect(ldexp(1.0, -1074)).toBe(5e-324);
    expect(ldexp(5e-324, 1073)).toBe(0.5);
    expect(ldexp(0.75, 1024)).toBe(0.75 * 2 ** 1023 * 2);
  });

  it('norm2 does not under- or overflow and equals the plain norm in range', () => {
    expect(norm2([3, 4])).toBe(5);
    expect(norm2([3e-200, 4e-200])).toBeCloseTo(5e-200, 210);
    expect(norm2([3e-200, 4e-200]) / 5e-200).toBeCloseTo(1, 14);
    expect(norm2([3e200, 4e200]) / 5e200).toBeCloseTo(1, 14);
    expect(norm2([0, 0])).toBe(0);
    expect(norm2([NaN, 1])).toBeNaN();
    expect(norm2([Infinity, 1])).toBe(Infinity);
    expect(norm2([5e-324, 0])).toBe(5e-324);
  });

  it('β formulas: values, PR+ truncation, Hager–Zhang bound and breakdown', () => {
    const g = [1.0, -2.0],
      gp = [3.0, 1.0],
      dp = [-3.0, -1.0];
    expect(betaOf('fletcher_reeves', g, gp, dp)).toEqual([5 / 10, 5 / 10]);
    // gᵀy = 1·(−2) + (−2)(−3) = 4 → PR = 0.4; with g, gp swapped PR < 0 and PR+ = 0.
    expect(betaOf('polak_ribiere', g, gp, dp)).toEqual([0.4, 0.4]);
    const [b, raw] = betaOf('polak_ribiere', [0.1, 0.0], gp, dp);
    expect(raw).toBeLessThan(0);
    expect(b).toBe(0);
    // dᵀy = (−3)(−2) + (−1)(−3) = 9
    expect(betaOf('hestenes_stiefel', g, gp, dp)).toEqual([4 / 9, 4 / 9]);
    expect(betaOf('dai_yuan', g, gp, dp)).toEqual([5 / 9, 5 / 9]);
    const [hz, hzRaw] = betaOf('hager_zhang', g, gp, dp);
    expect(hzRaw).toBeCloseTo((4 - (2 * 13 * -1) / 9) / 9, 14);
    expect(hz).toBe(hzRaw);
    // dᵀy ≤ 0 → undefined formula (restart = "breakdown").
    expect(betaOf('hestenes_stiefel', g, gp, [3.0, 1.0])).toEqual([null, null]);
    expect(betaOf('fletcher_reeves', g, [0, 0], dp)).toEqual([null, null]);
  });

  it('eigh reproduces np.linalg.eigh bit for bit on 2×2 matrices (LAPACK dsteqr / dlaev2)', () => {
    for (const { A, w, Q } of FIX.eigh2) {
      const got = eigh(A);
      expect(got.values).toEqual(w);
      expect(got.Q).toEqual(Q);
    }
  });

  it('Jacobi eigendecomposition (n ≥ 3): ascending values, orthonormal vectors, AQ = QΛ', () => {
    const A: Matrix = [
      [4, 1, -2, 0.5],
      [1, 2, 0, 1],
      [-2, 0, 3, -1],
      [0.5, 1, -1, -1],
    ];
    const { values, Q } = eigh(A);
    for (let i = 1; i < values.length; i++) expect(values[i]).toBeGreaterThanOrEqual(values[i - 1]);
    for (let j = 0; j < 4; j++) {
      const q = Q.map((row) => row[j]);
      const Aq = A.map((row) => row.reduce((s, v, k) => s + v * q[k], 0));
      Aq.forEach((v, i) => expect(v).toBeCloseTo(values[j] * q[i], 13));
      for (let l = 0; l < 4; l++) {
        const d = Q.reduce((s, row) => s + row[j] * row[l], 0);
        expect(d).toBeCloseTo(j === l ? 1 : 0, 14);
      }
    }
    const trace = A.reduce((s, row, i) => s + row[i], 0);
    expect(values.reduce((s, v) => s + v, 0)).toBeCloseTo(trace, 13);
  });
});

describe('contract', () => {
  it('a bare f(x) works like Python (central differences, dim from x0)', () => {
    const { fn } = getMethod('trust_region_dogleg');
    const r = fn(rosen, { x0: [-1.2, 1.0] } as never);
    const ref = FIX.runs.find(
      (c) => c.method === 'trust_region_dogleg' && c.problem === 'rosen_nograd',
    )!;
    expectSameRun({ ...r }, ref);
  });

  it('never throws on numerical breakdown and keeps max_iter + 1 steps at most', () => {
    for (const id of ['cg_polak_ribiere', 'trust_region_exact']) {
      for (const p of ['rosenbrock', 'himmelblau', 'ackley', 'rastrigin']) {
        const r = run(id, getProblem(p), { max_iter: 30 });
        expect(r.trace.length).toBeLessThanOrEqual(31);
        expect(r.nIter).toBe(r.trace[r.trace.length - 1].k);
      }
    }
  });
});
