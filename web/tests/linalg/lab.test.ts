/**
 * The linear-systems lab's model (src/labs/linalg/model.ts): the geometry it draws must be the
 * mathematics of the methods, so each helper is checked against the method's own output.
 */
import { describe, expect, it } from 'vitest';
import { runMethod } from '../../src/core/registry';
import { listProblems } from '../../src/problems/registry';
import type { Matrix, Vector } from '../../src/core/types';
import type { LinalgProblem } from '../../src/problems/linalg';
import {
  aNorm2,
  defaultStart,
  equationRuns,
  fieldOf,
  gsTargets,
  isDirect,
  jacobiTargets,
  lineInBox,
  lowerFactors,
  planeDomain,
  relaxationCurve,
  stageOf,
  structureOf,
  swapped,
} from '../../src/labs/linalg/model';
import { cellText, texNum } from '../../src/labs/linalg/texfmt';
import '../../src/labs/linalg/setup';

const P = (id: string) => listProblems<LinalgProblem>('linalg').find((p) => p.id === id)!;

describe('structure of A', () => {
  it('reports the documented numbers', () => {
    const s = structureOf(P('spd_2x2').A);
    expect(s.spd).toBe(true);
    expect(s.kappa).toBeCloseTo(3.5, 12);
    expect(s.rhoJ).toBeCloseTo(Math.SQRT2 / 3, 12);
    expect(s.rhoGS).toBeCloseTo(2 / 9, 12);
    expect(s.tridiagonal && s.diagDominant).toBe(true);

    const pois = structureOf(P('poisson_1d_10').A);
    expect(pois.omegaOpt).toBeCloseTo(2 / (1 + Math.sin(Math.PI / 11)), 12);
    expect(pois.kappa).toBeCloseTo(48.374, 2);

    const jd = structureOf(P('jacobi_diverges').A);
    expect(jd.spd).toBe(false);
    expect(jd.rhoJ).toBeCloseTo(Math.sqrt(5) / 2, 12);
    expect(jd.omegaOpt).toBeNull();

    expect(structureOf(P('needs_pivoting').A).zeroDiagonal).toBe(true);
    expect(structureOf(P('singular_3').A).kappa).toBe(Infinity);
    expect(structureOf(P('nonsymmetric_4').A).kappa).toBeCloseTo(3.15, 1);
  });

  it('samples ρ(G_ω) with its minimum at Young’s ω⋆', () => {
    const A = P('poisson_1d_10').A;
    const s = structureOf(A);
    const curve = relaxationCurve(A, [s.omegaOpt!]);
    const best = curve.reduce((m, p) => (p[1] < m[1] ? p : m));
    expect(best[0]).toBeCloseTo(s.omegaOpt!, 12);
    expect(best[1]).toBeCloseTo(s.omegaOpt! - 1, 6);
    // Kahan: ρ(G_ω) ≥ |ω − 1| everywhere.
    curve.forEach(([w, r]) => expect(r).toBeGreaterThanOrEqual(Math.abs(w - 1) - 1e-12));
    expect(relaxationCurve(P('needs_pivoting').A)).toEqual([]);
  });
});

describe('elimination factors', () => {
  for (const [method, problem] of [
    ['gaussian_elimination', 'diag_dominant_3'],
    ['gaussian_elimination_pivoting', 'needs_pivoting'],
    ['gaussian_elimination_pivoting', 'nonsymmetric_4'],
    ['lu_decomposition', 'nonsymmetric_4'],
    ['cholesky', 'hilbert_5'],
  ] as const) {
    it(`${method} on ${problem}: L builds up to the method's final factor`, () => {
      const r = runMethod(method, P(problem));
      const n = P(problem).n;
      const Ls = lowerFactors(method, r.trace, n);
      const final = Ls[Ls.length - 1]!;
      expect(final.flat().some(Number.isNaN)).toBe(false);
      const want = r.extra.L as Matrix;
      final.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(want[i][j], 14)));
      // Column c is known exactly from stage c + 1 on, and unknown before.
      for (let k = 1; k <= n; k++) {
        const Lk = Ls[k]!;
        for (let i = 0; i < n; i++)
          for (let j = 0; j < i; j++) expect(Number.isNaN(Lk[i][j])).toBe(j >= k);
      }
    });
  }

  it('reads the stages of a trace', () => {
    const r = runMethod('gaussian_elimination_pivoting', P('needs_pivoting'));
    const s1 = stageOf(r.trace[1], 3);
    expect(s1.rowSwap).toEqual([0, 1]);
    expect(s1.perm).toEqual([1, 0, 2]);
    expect(s1.pivotValue).toBe(1);
    expect(swapped(stageOf(r.trace[0], 3).matrix, s1.rowSwap)[0]).toEqual([1, -2, -3, 0]);
    expect(stageOf(r.trace[4], 3).phase).toBe('back_substitution');
    expect(isDirect('thomas') && !isDirect('gmres')).toBe(true);
  });
});

describe('plane geometry (2 × 2)', () => {
  const p = P('spd_2x2');
  const A = p.A as Matrix;
  const b = p.b;
  const x0 = [-2, 2];

  it('Jacobi targets lie on the lines and give the next iterate', () => {
    const r = runMethod('jacobi', p, { x0 });
    for (let k = 1; k < 5; k++) {
      const from = r.trace[k - 1].x as Vector;
      const [p1, p2] = jacobiTargets(A, b, from);
      expect(A[0][0] * p1[0] + A[0][1] * p1[1]).toBeCloseTo(b[0], 12);
      expect(A[1][0] * p2[0] + A[1][1] * p2[1]).toBeCloseTo(b[1], 12);
      const next = r.trace[k].x as Vector;
      expect(next[0]).toBeCloseTo(p1[0], 12);
      expect(next[1]).toBeCloseTo(p2[1], 12);
    }
  });

  it('SOR overshoots each Gauss–Seidel target by ω', () => {
    const omega = 1.4;
    const r = runMethod('sor', p, { x0, omega });
    const sweep = r.trace[1].info.sweep as Vector[];
    const [g1, g2] = gsTargets(sweep, omega);
    expect(A[0][0] * g1[0] + A[0][1] * g1[1]).toBeCloseTo(b[0], 12);
    expect(A[1][0] * g2[0] + A[1][1] * g2[1]).toBeCloseTo(b[1], 12);
    // The new component is x_old + ω (x_GS − x_old).
    expect(sweep[1][0]).toBeCloseTo(sweep[0][0] + omega * (g1[0] - sweep[0][0]), 12);
  });

  it('CG steps are tangent to the level ellipse of φ they end on', () => {
    const r = runMethod('conjugate_gradient_linear', p, { x0 });
    const xs = p.solution!;
    const x1 = r.trace[1].x as Vector;
    const d0 = r.trace[0].info.direction as Vector;
    // ∇φ(x₁) = A x₁ − b is orthogonal to the step direction p₀ (exact line search).
    const g = [A[0][0] * x1[0] + A[0][1] * x1[1] - b[0], A[1][0] * x1[0] + A[1][1] * x1[1] - b[1]];
    expect(g[0] * d0[0] + g[1] * d0[1]).toBeCloseTo(0, 10);
    // φ(x) − φ⋆ = ½‖x − x⋆‖²_A: the ellipse level the view draws.
    const f = fieldOf(A, b, true);
    expect(f(x1[0], x1[1]) - f(xs[0], xs[1])).toBeCloseTo(0.5 * aNorm2(A, x1, xs), 12);
  });

  it('clips the equations to the view and labels them', () => {
    const box = planeDomain(p.solution!, x0);
    const seg = lineInBox(A[0], b[0], box)!;
    for (const q of seg) expect(A[0][0] * q[0] + A[0][1] * q[1]).toBeCloseTo(b[0], 12);
    expect(lineInBox([1, 0], 100, box)).toBeNull();
    expect(
      equationRuns([3, 2], 2)
        .map((r) => r.t)
        .join(''),
    ).toBe('3x + 2y = 2');
    expect(
      equationRuns([2, -6], -8)
        .map((r) => r.t)
        .join(''),
    ).toBe('2x − 6y = −8');
    expect(
      equationRuns(P('nearly_singular').A[1], P('nearly_singular').b[1])
        .map((r) => r.t)
        .join(''),
    ).toBe('x + 1.000000001y = 2.000000001');
    expect(defaultStart(p)).toEqual([-2, 2]);
    expect(defaultStart(P('poisson_1d_10'))).toEqual(new Array(10).fill(0));
  });
});

describe('number formatting', () => {
  it('prints matrix entries and TeX numbers', () => {
    expect(cellText(0)).toBe('0');
    expect(cellText(-3)).toBe('−3');
    expect(cellText(0.3333333)).toBe('0.3333');
    expect(cellText(NaN)).toBe('·');
    expect(texNum(-0.5)).toBe('-0.5');
    expect(texNum(1.6e-16)).toBe('1.6\\times 10^{-16}');
    expect(texNum(null)).toBe('\\text{—}');
  });
});
