/**
 * numopt.unconstrained.newton — TS port checks beyond the parity harness:
 *   - the registered specs equal registry.json;
 *   - every Newton fixture of unconstrained.json matches step by step (x, fun, ‖∇f‖, α, every
 *     info key), with the same counts and message;
 *   - more Python reference runs (tests/newton_qn/fixtures, from gen_newton_qn_fixture.py): every
 *     line search, finite-difference ∇f / ∇²f, saddles, maximizers, singular and divergent runs,
 *     max_iter stops, n = 3, and the ValueError messages;
 *   - focused unit tests (the Jacobi eigensolver, Alg. 3.3, the Cholesky solve).
 */
import { describe, expect, it } from 'vitest';
import type { Matrix } from '../../src/core/types';
import {
  choleskyShift,
  choleskySolve,
  eigh,
  fdGradient,
  fdHessian,
  pureNewton,
} from '../../src/methods/unconstrained/newton';
import '../../src/problems';
import {
  caseKey,
  errorOf,
  expectSameDump,
  expectSameResult,
  generatedFixtures,
  problemById,
  pythonSpecs,
  referenceCases,
  run,
  tsSpec,
} from './harness';

const IDS = ['pure_newton', 'damped_newton', 'modified_newton'] as const;

describe('newton registry', () => {
  it('registers the three Python methods with identical specs', () => {
    const python = pythonSpecs(IDS);
    expect(python.map((s) => s.id).sort()).toEqual([...IDS].sort());
    for (const want of python) expect(tsSpec(want.id)).toEqual(want);
  });
});

describe('newton fixtures (unconstrained.json), step by step', () => {
  const cases = generatedFixtures(IDS);
  it('has every Python fixture case', () => expect(cases.length).toBe(8));
  cases.forEach((c, i) => {
    it(`${c.method} on ${c.problem} #${i}`, () => {
      expectSameResult(run(c.method, problemById(c.problem), c.params), c.result!);
    });
  });
});

/**
 * Runs whose ∇f and ∇²f are central differences of f only: on another platform their iterates
 * differ by about 1e-6 (harness.ts, `expectSameDump`, `noisy`).
 */
const FD_ONLY = new Set(['rosen_fonly', 'valley_fonly']);

describe('newton reference runs (gen_newton_qn_fixture.py)', () => {
  referenceCases('newton').forEach((c, i) => {
    it(`${caseKey(c)} #${i}`, () => {
      if (c.error !== undefined) {
        const err = errorOf(() => run(c.method, problemById(c.problem), c.params));
        expect(err).toEqual({ name: 'ValueError', message: c.error });
        return;
      }
      expectSameDump(run(c.method, problemById(c.problem), c.params), c.result!, {
        noisy: FD_ONLY.has(c.problem),
      });
    });
  });
});

describe('newton helpers', () => {
  const reconstruct = (lam: number[], Q: Matrix) =>
    Q.map((_, i) => Q.map((__, j) => lam.reduce((s, l, k) => s + Q[i][k] * l * Q[j][k], 0)));

  it('eigh: ascending eigenvalues and an orthonormal Q with A = Q Λ Qᵀ', () => {
    const A: Matrix = [
      [4, 1, -2, 0.5],
      [1, 3, 0.25, 1],
      [-2, 0.25, -1, 2],
      [0.5, 1, 2, 6],
    ];
    const { lam, Q } = eigh(A);
    expect([...lam].sort((a, b) => a - b)).toEqual(lam);
    const B = reconstruct(lam, Q);
    A.forEach((row, i) => row.forEach((v, j) => expect(B[i][j]).toBeCloseTo(v, 13)));
    const QtQ = Q.map((_, i) => Q.map((__, j) => Q.reduce((s, r) => s + r[i] * r[j], 0)));
    QtQ.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(i === j ? 1 : 0, 14)));
    // trace and determinant-free check: Σλ = tr A
    expect(lam.reduce((a, b) => a + b, 0)).toBeCloseTo(4 + 3 - 1 + 6, 13);
  });

  it('eigh: diagonal, 1×1 and zero matrices are exact', () => {
    expect(
      eigh([
        [3, 0],
        [0, -2],
      ]).lam,
    ).toEqual([-2, 3]);
    expect(eigh([[5]])).toEqual({ lam: [5], Q: [[1]] });
    expect(
      eigh([
        [0, 0],
        [0, 0],
      ]).lam,
    ).toEqual([0, 0]);
  });

  it('eigh: tiny off-diagonal entries do not overflow the rotation', () => {
    const { lam } = eigh([
      [1, 1e-300],
      [1e-300, 2],
    ]);
    expect(lam).toEqual([1, 2]);
  });

  it('choleskyShift: τ = 0 for a positive definite matrix, Alg. 3.3 doubling otherwise', () => {
    const [L0, tau0, n0] = choleskyShift(
      [
        [4, 2],
        [2, 3],
      ],
      1e-3,
    );
    expect([tau0, n0]).toEqual([0, 1]);
    expect(L0![0][0]).toBe(2);
    // diag min = −2 → τ₀ = 2 + β; A + τ₀I = [[β, 0], [0, 6 + β]] factors at once.
    const [, tau1, n1] = choleskyShift(
      [
        [-2, 0],
        [0, 4],
      ],
      1e-3,
    );
    expect([tau1, n1]).toEqual([2.001, 1]);
    // Positive diagonal but indefinite (λ = −2, 4): τ = 0, β, 2β, 4β fail; 8β = 2.4 > 2 works.
    const [, tau2, n2] = choleskyShift(
      [
        [1, 3],
        [3, 1],
      ],
      0.3,
    );
    expect([tau2, n2]).toEqual([0.3 * 8, 5]);
  });

  it('choleskyShift: gives up after 64 doublings with τ ≥ β', () => {
    // τ₀ = +∞: every A + τI has a NaN entry, so all 64 attempts with τ ≥ β fail.
    const [L, tau, attempts] = choleskyShift([[-Infinity]], 1.0);
    expect(L).toBeNull();
    expect(attempts).toBe(64);
    expect(tau).toBe(Infinity);
  });

  it('choleskySolve solves L Lᵀ z = b', () => {
    const L: Matrix = [
      [2, 0],
      [1, 3],
    ];
    // A = L Lᵀ = [[4, 2], [2, 10]]; A [1, 2] = [8, 22]
    expect(choleskySolve(L, [8, 22])).toEqual([1, 2]);
  });

  it('central differences match numopt.core.diff on a quadratic', () => {
    const f = (x: number[]) => x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2;
    const g = fdGradient(f, [1, -2]);
    expect(g[0]).toBeCloseTo(-4, 8);
    expect(g[1]).toBeCloseTo(-5, 8);
    const H = fdHessian((x) => [2 * x[0] + 3 * x[1], 3 * x[0] + 4 * x[1]], [1, -2]);
    expect(H[0][1]).toBe(H[1][0]);
    expect(H[0][0]).toBeCloseTo(2, 8);
    expect(H[1][1]).toBeCloseTo(4, 8);
  });

  it('accepts a bare f(x) with x0 (vector_problem) and counts evaluations', () => {
    const r = pureNewton((x) => (x[0] - 1) ** 2 + 2 * (x[1] + 3) ** 2, { x0: [0, 0] });
    expect(r.converged).toBe(true);
    const x = r.x as number[];
    expect(x[0]).toBeCloseTo(1, 6);
    expect(x[1]).toBeCloseTo(-3, 6);
    // f at x_k plus 2n per finite-difference ∇f; ∇f at x_k plus 2n per finite-difference ∇²f.
    expect(r.nHev).toBe(r.nIter + 1);
    expect(r.nGev).toBe((r.nIter + 1) * 5);
    expect(r.nFev).toBe(r.nGev * 4 + r.nIter + 1);
  });

  it('throws the Python ValueError without a start point', () => {
    const err = errorOf(() =>
      pureNewton({ ...(problemById('rosenbrock') as object), x0: null } as never),
    );
    expect(err).toEqual({
      name: 'ValueError',
      message: 'rosenbrock: no starting point given and the problem has no default x0',
    });
  });
});
