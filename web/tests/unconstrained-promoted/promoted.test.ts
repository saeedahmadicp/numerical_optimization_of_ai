/**
 * The promoted unconstrained methods — TS ports of accelerated.py (silver_gd,
 * silver_gd_strongly_convex, long_step_gd, ogm, fista), anderson.py (anderson_gd) and
 * regularized_newton.py (arc, reg_newton) — beyond the parity harness:
 *   - the eight registered specs equal registry.json;
 *   - every parity fixture of the eight methods matches Python step by step (x, f, ‖∇f‖, step
 *     size and every info key), with the same counts, message, converged flag and extra;
 *   - 36 more Python runs (other problems and parameters, failure paths) and 12 ValueError
 *     messages from fixtures/promoted_python.json (gen_promoted_fixture.py);
 *   - the schedule helpers and the subproblem solvers against their defining equations.
 *
 * One parity fixture cannot be replayed to its end by any port: anderson_gd on rosenbrock (176
 * iterations) is chaotic. A change of x₀[0] by one ulp makes the Python run itself take 227
 * iterations, so its iteration count depends on the last bit of LAPACK's dgelsd. The step-by-step
 * check below covers its first 20 iterates (agreement 10⁻⁹ there) and the parity harness reports
 * the case until the fixture is truncated (see the port's summary).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fixtureCaseFromJson, methodSpecFromJson, resultFromJson } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import { getProblem } from '../../src/problems/registry';
import type { Matrix, Result, Vector } from '../../src/core/types';
import '../../src/problems/unconstrained';
import {
  LONG_STEP_PATTERNS,
  LONG_STEP_RATES,
  ogmThetas,
  scaledNorm,
  silverRate,
  silverScAutoHorizon,
  silverScSchedule,
  silverStep,
  twoAdicValuation,
} from '../../src/methods/unconstrained/accelerated';
import { aaCoefficients, lstsq } from '../../src/methods/unconstrained/anderson';
import {
  cubicCauchy,
  cubicModel,
  cubicSubproblem,
} from '../../src/methods/unconstrained/regularized_newton';
import { mismatch, readJson, type Tol } from '../shared-ports/compare';

type Raw = Record<string, unknown>;

const IDS = [
  'anderson_gd',
  'arc',
  'fista',
  'long_step_gd',
  'ogm',
  'reg_newton',
  'silver_gd',
  'silver_gd_strongly_convex',
];

/** Agreement of a whole replay: rounding differences (libm, BLAS order, Jacobi vs LAPACK). */
const STEP_TOL: Tol = { rtol: 1e-9, atol: 1e-12 };
/**
 * AA solves a least-squares problem per step with cond(ΔF) up to 10⁸ and has no damping of the
 * resulting differences (the SVD of the port and LAPACK's dgelsd differ in the last bits), and a
 * run that ends by GMRES termination has a gradient of pure rounding (10⁻¹²).
 */
const AA_TOL: Tol = { rtol: 1e-7, atol: 1e-10 };

function run(method: string, problem: unknown, params: Raw): Result {
  const { spec, fn } = getMethod(method);
  const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
  return fn(problem, { ...defaults, ...(params as Record<string, never>) });
}

/**
 * Info keys whose value is ill-conditioned by construction: AA's weights c* solve a least-squares
 * problem whose condition number reaches 10⁴–10⁸ near convergence, and cond = σ_max/σ_min carries
 * the error of σ_min. They are compared to 10⁻⁶ relative to max(1, ‖c*‖∞); the iterates they
 * produce are still compared to 10⁻⁹.
 */
const LOOSE_INFO = new Set(['cond', 'coefficients']);
const LOOSE = 1e-6;

/** A message with its numbers masked: AA's final ‖∇f‖ can be pure rounding (GMRES termination). */
const masked = (m: string) => m.replace(/-?\d+(\.\d+)?(e[-+]?\d+)?/g, '#');

function stripLoose(info: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(info).filter(([k]) => !LOOSE_INFO.has(k)));
}

/** Compare a TS Result with a Python result (snake_case JSON), field by field. */
function expectSameResult(got: Result, wantRaw: Raw, steps = Infinity) {
  const want = resultFromJson(wantRaw);
  const full = steps === Infinity;
  const tol = got.method === 'anderson_gd' ? AA_TOL : STEP_TOL;
  if (full) {
    if (got.method === 'anderson_gd' && got.message !== want.message)
      expect(masked(got.message)).toBe(masked(want.message));
    else expect(got.message).toBe(want.message);
    expect(got.converged).toBe(want.converged);
    expect([got.nIter, got.nFev, got.nGev, got.nHev]).toEqual([
      want.nIter,
      want.nFev,
      want.nGev,
      want.nHev,
    ]);
    expect(got.trace.length).toBe(want.trace.length);
  }
  const n = Math.min(steps, want.trace.length);
  for (let k = 0; k < n; k++) {
    const s = got.trace[k];
    const w = want.trace[k];
    const m = mismatch(
      {
        k: s.k,
        x: s.x,
        fun: s.fun,
        gradNorm: s.gradNorm,
        stepSize: s.stepSize,
        info: stripLoose(s.info),
      },
      {
        k: w.k,
        x: w.x,
        fun: w.fun,
        gradNorm: w.gradNorm,
        stepSize: w.stepSize,
        info: stripLoose(w.info),
      },
      tol,
      `trace[${k}]`,
    );
    expect(m).toBeNull();
    if (typeof w.info.cond === 'number' && Number.isFinite(w.info.cond))
      expect(
        Math.abs((s.info.cond as number) - w.info.cond) / w.info.cond,
        `trace[${k}].info.cond`,
      ).toBeLessThan(LOOSE);
    if (Array.isArray(w.info.coefficients)) {
      const c = w.info.coefficients as number[];
      const scale = Math.max(1, ...c.map(Math.abs));
      const loose: Tol = { rtol: 0, atol: LOOSE * scale };
      expect(mismatch(s.info.coefficients, c, loose, `trace[${k}].info.coefficients`)).toBeNull();
    }
  }
  if (full) {
    expect(mismatch(got.x, want.x, tol, 'x')).toBeNull();
    expect(mismatch(got.fun, want.fun, tol, 'fun')).toBeNull();
    expect(mismatch(got.extra, want.extra, tol, 'extra')).toBeNull();
  }
}

// ---------------------------------------------------------------------------------------

describe('registry', () => {
  const registry = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/registry.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  const python = registry.filter((m) => IDS.includes(m.id as string)).map(methodSpecFromJson);

  it('registers the eight Python methods with identical specs', () => {
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

describe('parity fixtures of the eight methods, step by step', () => {
  const cases = readJson<Raw[]>('../../src/generated/fixtures/unconstrained.json')
    .filter((c) => IDS.includes(c.method as string))
    .map((c) => ({ c: fixtureCaseFromJson(c), raw: c.result as Raw }));

  it('has a fixture for every method', () => {
    expect([...new Set(cases.map(({ c }) => c.method))].sort()).toEqual(IDS);
  });

  cases.forEach(({ c, raw }, i) => {
    // The chaotic case (module comment): its first 20 iterates.
    const chaotic = c.method === 'anderson_gd' && c.problem === 'rosenbrock';
    it(`${c.method} on ${c.problem} #${i}${chaotic ? ' (first 20 iterates)' : ''}`, () => {
      expectSameResult(
        run(c.method, getProblem(c.problem), c.params as Raw),
        raw,
        chaotic ? 20 : Infinity,
      );
    });
  });
});

interface Promoted {
  cases: { method: string; problem: string; params: Raw; result: Raw }[];
  errors: { method: string; problem: string; params: Raw; message: string }[];
  helpers: {
    silver_schedule_15: number[];
    silver_rates: number[];
    silver_sc: { kappa: number; n: number; h: number[]; tau: number }[];
    auto_horizon: Record<string, number>;
    ogm_thetas_6: number[];
  };
}

const PY = readJson<Promoted>('../unconstrained-promoted/fixtures/promoted_python.json');

describe('more Python runs, step by step', () => {
  PY.cases.forEach((c, i) => {
    it(`${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${i}`, () => {
      expectSameResult(run(c.method, getProblem(c.problem), c.params), c.result);
    });
  });
});

describe('invalid input', () => {
  PY.errors.forEach((e) => {
    it(`${e.method} ${JSON.stringify(e.params)}: ${e.message}`, () => {
      expect(() => run(e.method, getProblem(e.problem), e.params)).toThrow(e.message);
    });
  });
});

// ---------------------------------------------------------------------------------------

const close = (a: number, b: number, rtol = 1e-14) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(rtol * Math.max(1, Math.abs(b)));

describe('schedules', () => {
  it('silver steps, rates and ν(t) equal Python', () => {
    expect(twoAdicValuation(1)).toBe(0);
    expect(twoAdicValuation(12)).toBe(2);
    PY.helpers.silver_schedule_15.forEach((h, t) => close(silverStep(t), h));
    PY.helpers.silver_rates.forEach((r, k) => close(silverRate(k), r));
    expect(silverRate(0)).toBe(0.5);
  });

  it('the κ-aware silver block and its rate equal Python', () => {
    for (const { kappa, n, h, tau } of PY.helpers.silver_sc) {
      const [hh, tt] = silverScSchedule(kappa, n);
      expect(hh.length).toBe(n);
      hh.forEach((v, i) => close(v, h[i], 1e-13));
      close(tt, tau, 1e-12);
      // Every step lies in (1, (κ + 1)/2] (Part I, Lemma 3.2), so none is longer than 1/μ.
      if (kappa > 1) for (const v of hh) expect(v > 1 && v <= (kappa + 1) / 2 + 1e-12).toBe(true);
    }
    for (const [k, n] of Object.entries(PY.helpers.auto_horizon))
      expect(silverScAutoHorizon(Number(k))).toBe(n);
  });

  it('OGM θ and the long-step patterns', () => {
    ogmThetas(6).forEach((t, i) => close(t, PY.helpers.ogm_thetas_6[i]));
    for (const [t, h] of Object.entries(LONG_STEP_PATTERNS)) {
      expect(h.length).toBe(Number(t));
      // Table 1's rate constant c is close to the pattern's average step.
      close(h.reduce((a, v) => a + v, 0) / h.length, LONG_STEP_RATES[t], 2e-5);
    }
  });

  it('the scaled norm never underflows to 0 for a nonzero vector', () => {
    expect(scaledNorm([1e-200, 1e-200])).toBeCloseTo(Math.SQRT2 * 1e-200, 210);
    expect(scaledNorm([3, 4])).toBe(5);
    expect(scaledNorm([0, 0])).toBe(0);
  });
});

describe('Anderson least squares', () => {
  it('lstsq returns the minimum-norm solution of a wide and a tall system', () => {
    // Wide 2 × 4, full row rank: x = Aᵀ(AAᵀ)⁻¹b.
    const A: Matrix = [
      [1, 2, -1, 0.5],
      [0, 1, 3, -2],
    ];
    const b = [1, -2];
    const { x } = lstsq(A, b);
    const r = A.map((row) => row.reduce((s, v, j) => s + v * x[j], 0) - b[A.indexOf(row)]);
    r.forEach((v) => expect(Math.abs(v)).toBeLessThan(1e-14));
    // x is in the row space of A: x = Aᵀy.
    const AAt = [
      [A[0].reduce((s, v) => s + v * v, 0), A[0].reduce((s, v, j) => s + v * A[1][j], 0)],
      [0, A[1].reduce((s, v) => s + v * v, 0)],
    ];
    AAt[1][0] = AAt[0][1];
    const det = AAt[0][0] * AAt[1][1] - AAt[0][1] ** 2;
    const y = [
      (AAt[1][1] * b[0] - AAt[0][1] * b[1]) / det,
      (AAt[0][0] * b[1] - AAt[0][1] * b[0]) / det,
    ];
    A[0].forEach((_, j) => close(x[j], A[0][j] * y[0] + A[1][j] * y[1], 1e-13));
    // Tall 4 × 2: the normal equations.
    const T: Matrix = [
      [1, 0],
      [1, 1],
      [1, 2],
      [1, 3],
    ];
    const { x: c, s } = lstsq(T, [1, 2, 2, 4]);
    close(c[0], 0.9, 1e-13);
    close(c[1], 0.9, 1e-13);
    expect(s[0]).toBeGreaterThan(s[1]);
  });

  it('a rank-deficient ΔF gives a finite step (the zero singular value is cut)', () => {
    const { x, s } = lstsq(
      [
        [1, 2],
        [2, 4],
        [3, 6],
      ],
      [1, 1, 1],
    );
    expect(x.every(Number.isFinite)).toBe(true);
    expect(s[1]).toBeLessThan(1e-14 * s[0]);
  });

  it('the AA weights sum to 1 and minimize ‖Fc‖ (λ = 0) or ‖Fc‖² + λ′‖c‖² (λ > 0)', () => {
    const F: Matrix = [
      [0.3, -0.1, 0.05, 0.02],
      [-0.2, 0.15, -0.04, 0.01],
      [0.1, 0.05, 0.03, -0.02],
    ];
    for (const lam of [0, 1e-2]) {
      const { c, lamEff } = aaCoefficients(F, lam);
      close(
        c.reduce((a, v) => a + v, 0),
        1,
        1e-14,
      );
      const obj = (w: Vector) =>
        F.reduce((s, row) => s + row.reduce((t, v, j) => t + v * w[j], 0) ** 2, 0) +
        lamEff * w.reduce((s, v) => s + v * v, 0);
      // Feasible perturbations (1ᵀd = 0) never decrease the objective.
      for (const d of [
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [1, 0, 0, -1],
      ])
        for (const t of [1e-3, -1e-3])
          expect(obj(c.map((v, j) => v + t * d[j]))).toBeGreaterThanOrEqual(obj(c) - 1e-16);
      expect(lam === 0 ? lamEff === 0 : lamEff > 0).toBe(true);
    }
  });
});

describe('the cubic subproblem (CGT Thm. 3.1)', () => {
  const cases: [Vector, Matrix, number][] = [
    [
      [1, -2],
      [
        [3, 1],
        [1, 2],
      ],
      1,
    ],
    [
      [0.5, 0.3],
      [
        [-2, 0.4],
        [0.4, 1],
      ],
      0.7,
    ],
    [
      [-4, 1],
      [
        [10, -3],
        [-3, 1],
      ],
      100,
    ],
  ];
  it('(B + λI)s = −g, λ = σ‖s‖ and B + λI ⪰ 0 at the returned step', () => {
    for (const [g, B, sigma] of cases) {
      const r = cubicSubproblem(g, B, sigma);
      expect(r.solved).toBe(true);
      const ns = Math.hypot(...r.s);
      close(r.lam, sigma * ns, 1e-10);
      for (let i = 0; i < 2; i++)
        close(B[i][0] * r.s[0] + B[i][1] * r.s[1] + r.lam * r.s[i], -g[i], 1e-9);
      expect(r.lam + r.lamMin).toBeGreaterThanOrEqual(-1e-12);
      // The predicted decrease is f − m(s), and the step beats the Cauchy point.
      close(r.predicted, -cubicModel(g, B, sigma, r.s), 1e-10);
      expect(cubicModel(g, B, sigma, r.s)).toBeLessThanOrEqual(
        cubicModel(g, B, sigma, cubicCauchy(g, B, sigma)) + 1e-14,
      );
    }
  });

  it('the hard case: g ⊥ the eigenvector of λ₁ < 0 gives ‖s‖ = −λ₁/σ', () => {
    const r = cubicSubproblem(
      [1, 0],
      [
        [2, 0],
        [0, -2],
      ],
      1,
    );
    expect(r.hardCase).toBe(true);
    close(Math.hypot(...r.s), 2, 1e-12);
    close(r.lam, 2, 1e-12);
  });
});
