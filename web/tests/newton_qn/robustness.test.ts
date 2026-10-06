/**
 * Newton and quasi-Newton ports at the limits of float64 (mirrors the Python regressions in
 * tests/test_unconstrained_newton.py and tests/test_unconstrained_quasi_newton.py):
 *   - a non-finite slope ∇fᵀp ends the run with converged = false and a message (it used to reach
 *     search(), which throws ValueError on a non-finite slope);
 *   - `eigh` of a 2×2 Hessian scales like LAPACK `dsyevd` when max|a_ij| is outside
 *     [2⁻⁴⁸⁵, 2⁴⁸⁵], so the deflation test cannot overflow (oracle: numpy.linalg.eigh).
 */
import { describe, expect, it } from 'vitest';
import type { Matrix, Problem, Vector } from '../../src/core/types';
import { eigh, slopeFailure } from '../../src/methods/unconstrained/newton';
import '../../src/methods/unconstrained/quasi_newton';
import { getProblem } from '../../src/problems/registry';
import '../../src/problems/unconstrained';
import { CUSTOM, run } from './harness';

const LS_METHODS = [
  'damped_newton',
  'modified_newton',
  'bfgs',
  'dfp',
  'sr1',
  'broyden_class',
  'lbfgs',
] as const;

describe('a non-finite slope ∇fᵀp is a failed run, not an exception', () => {
  for (const method of LS_METHODS) {
    for (const lineSearch of ['backtracking', 'strong_wolfe']) {
      it(`${method} with ${lineSearch} on f = −(x² + y²)`, () => {
        const r = run(method, CUSTOM.bowl_down, { line_search: lineSearch, max_iter: 2000 });
        expect(r.converged).toBe(false);
        expect((r.x as Vector).every(Number.isFinite)).toBe(true);
        expect(Number.isFinite(r.fun)).toBe(true);
        if (lineSearch === 'backtracking') {
          // α = 1 is accepted until ‖x‖ ≈ 1e154, where ∇fᵀp overflows to −inf.
          expect(r.message).toContain('∇fᵀp overflowed to -inf, so no step can be tested');
          expect(r.message.startsWith(`line search failed at iteration ${r.nIter + 1}`)).toBe(true);
          expect(r.trace[r.trace.length - 1].gradNorm).toBeGreaterThan(1e153);
        } else {
          // φ still decreases at α_max on the first search.
          expect(r.message).toContain('phi still decreases at alpha_max');
        }
      });
    }
  }

  it('slopeFailure names the overflow or the underflow (Python _slope_failure)', () => {
    expect(slopeFailure(-Infinity)).toBe('∇fᵀp overflowed to -inf, so no step can be tested');
    expect(slopeFailure(NaN)).toBe('∇fᵀp overflowed to nan, so no step can be tested');
    expect(slopeFailure(0.0)).toBe('∇fᵀp underflowed to 0, so no step can be tested');
  });

  it('matches the Python run of bfgs and modified_newton on f = −(x² + y²)', () => {
    // Values of numopt.run in Python (also in the step-by-step fixture newton_qn_python.json).
    const bfgs = run('bfgs', CUSTOM.bowl_down, { line_search: 'backtracking', max_iter: 2000 });
    expect([bfgs.nIter, bfgs.nFev, bfgs.nGev]).toEqual([323, 325, 324]);
    expect(bfgs.message).toBe(
      'line search failed at iteration 324 (‖∇f‖∞ = 1.72e+154): ∇fᵀp overflowed to -inf, so no ' +
        'step can be tested; predicted decrease |∇fᵀp| = inf vs rounding level of f ' +
        'ε·max(1, |f|) = 2.05e+292',
    );
    const mn = run('modified_newton', CUSTOM.bowl_down, {});
    expect([mn.nIter, mn.nFev, mn.nGev, mn.nHev]).toEqual([47, 52, 48, 48]);
  });
});

describe('eigh of a 2×2 matrix at extreme scales (LAPACK dsyevd scaling)', () => {
  const Q_REF: Matrix = [
    [-0.4464987692873111, -0.8947842471930967],
    [-0.8947842471930967, 0.4464987692873111],
  ];
  // numpy.linalg.eigh(2**e * [[802, -400], [-400, 200]]): the Rosenbrock Hessian at x* scaled.
  const CASES: [number, [number, number]][] = [
    [600, [1.6571537222902177e180, 4.1561574462964646e183]],
    [-600, [9.624274469111938e-182, 2.4137772773861786e-178]],
    [1000, [4.2791849973551815e300, 1.0732237059009043e304]],
    [-1000, [3.7270887495373604e-302, 9.347574368652715e-299]],
    [500, [1.3072637854562304e150, 3.2786301253264778e153]],
  ];
  for (const [e, lam] of CASES) {
    it(`2^${e}·∇²f_rosenbrock(x*) equals numpy.linalg.eigh bit for bit`, () => {
      const c = 2 ** e;
      const got = eigh([
        [c * 802, c * -400],
        [c * -400, c * 200],
      ]);
      expect(got.lam).toEqual(lam);
      expect(got.Q).toEqual(Q_REF);
    });
  }

  it('an indefinite matrix with entries near 1e-160 equals numpy.linalg.eigh bit for bit', () => {
    const got = eigh([
      [1e-160, 3e-160],
      [3e-160, -2e-160],
    ]);
    expect(got.lam).toEqual([-3.8541019662496847e-160, 2.854101966249684e-160]);
    expect(got.Q).toEqual([
      [-0.5257311121191335, -0.8506508083520399],
      [0.8506508083520399, -0.5257311121191335],
    ]);
  });

  it('pure and damped Newton converge on 2⁶⁰⁰·Rosenbrock as in Python', () => {
    const base = getProblem<Problem<Vector>>('rosenbrock');
    const c = 2 ** 600;
    const { grad, hess } = base;
    const scaled: Problem<Vector> = {
      ...base,
      f: (x) => c * (base.f(x) as number),
      grad: (x) => (grad!(x) as Vector).map((t) => c * t),
      hess: (x) => (hess!(x) as Matrix).map((row) => row.map((t) => c * t)),
    };
    const head = '‖∇f‖∞ = 0 ≤ gtol; ∇²f is positive definite (λ_min = 1.66e+180)';
    const pure = run('pure_newton', scaled, {});
    expect([pure.converged, pure.nIter, pure.nFev, pure.nGev, pure.nHev]).toEqual([
      true,
      7,
      8,
      8,
      8,
    ]);
    expect(pure.message).toBe(`${head}: a strict local minimizer`);
    const damped = run('damped_newton', scaled, {});
    expect([damped.converged, damped.nIter, damped.nFev, damped.nGev]).toEqual([true, 22, 30, 23]);
    expect(damped.message).toBe(`${head}: a strict local minimizer`);
    expect(damped.x).toEqual([1, 1]);
  });
});
