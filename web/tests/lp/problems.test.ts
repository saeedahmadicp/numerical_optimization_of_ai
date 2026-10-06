/**
 * The LP problem library (src/problems/lp.ts) against src/generated/problems.json, plus unit tests
 * for helpers the reference results do not exercise directly (exact rationals, Python's `:g`
 * format, the dense kernels of the interior-point port, the generated KaTeX).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { linearProgramFromJson } from '../../src/core/json';
import { listProblems } from '../../src/problems/registry';
import { lpLatex, type LPProblem } from '../../src/problems/lp';
import { Q, roundHalfEven } from '../../src/methods/lp/integer';
import { pyG, luFactor, luSolve } from '../../src/methods/lp/simplex';
import { householderQR, matrixRank, singularValues } from '../../src/methods/lp/interior_point';

const META = (
  JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/problems.json', import.meta.url)),
      'utf8',
    ),
  ) as Record<string, unknown>[]
).filter((p) => p.kind === 'lp');

describe('LP problems equal problems.json', () => {
  const ts = listProblems<LPProblem>('lp');
  it('same ids in the same order', () => {
    expect(ts.map((p) => p.id)).toEqual(META.map((p) => p.id));
  });
  for (const raw of META) {
    it(`${String(raw.id)}: data and metadata`, () => {
      const want = linearProgramFromJson(raw);
      const got = ts.find((p) => p.id === want.id)!;
      const { latex, dim, tags, ...data } = got;
      expect(data).toEqual(want);
      expect(dim).toBe(want.c.length);
      expect(latex.length).toBeGreaterThan(10);
      expect(Array.isArray(tags)).toBe(true);
    });
  }
});

describe('lpLatex', () => {
  it('typesets a small program row by row, ≥ rows keep their stored ≤ form', () => {
    const tex = lpLatex(listProblems<LPProblem>('lp').find((p) => p.id === 'diet_2d')!);
    expect(tex).toContain('\\min');
    // A leading minus is braced so KaTeX sets it unary (not spaced as a binary operator).
    expect(tex).toContain('{-}x_{1} - 3x_{2} \\le -6');
    expect(tex).toContain('\\text{s.t.}');
  });
  it('uses matrix form for larger programs and marks integrality', () => {
    const tex = lpLatex(listProblems<LPProblem>('lp').find((p) => p.id === 'transport_small')!);
    expect(tex).toContain('A_{\\text{eq}}');
    const ilp = lpLatex(
      listProblems<LPProblem>('lp').find((p) => p.id === 'ilp_knapsack_like_2d')!,
    );
    expect(ilp).toContain('\\mathbb{Z}^{2}_{\\ge 0}');
  });
});

describe('exact rationals (Fraction semantics)', () => {
  it('normalizes, compares and floors like Python', () => {
    const a = new Q(6n, -4n);
    expect([a.n, a.d]).toEqual([-3n, 2n]);
    expect(a.floor().n).toBe(-2n);
    expect(a.frac().cmp(new Q(1n, 2n))).toBe(0);
    expect(new Q(7n, 3n).frac().cmp(new Q(1n, 3n))).toBe(0);
    expect(new Q(1n, 3n).add(new Q(1n, 6n)).cmp(new Q(1n, 2n))).toBe(0);
  });
  it('Fraction(float) is the exact binary value', () => {
    const q = Q.fromFloat(0.1);
    expect(q.d).toBe(2n ** 55n);
    expect(q.n).toBe(3602879701896397n);
    expect(q.toNumber()).toBe(0.1);
    expect(Q.fromFloat(-2.5).cmp(new Q(-5n, 2n))).toBe(0);
  });
  it('round half to even', () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, 2.4].map(roundHalfEven)).toEqual([0, 2, 2, -0, -2, 2]);
  });
});

describe("Python's general format", () => {
  it.each([
    [2, 6, '2'],
    [1234567, 6, '1.23457e+06'],
    [0.000123456, 3, '0.000123'],
    [0.0000123456, 3, '1.23e-05'],
    [2.0000001, 6, '2'],
    [-0.5, 3, '-0.5'],
  ])('%s with %i digits → %s', (v, p, want) => expect(pyG(v, p)).toBe(want));
});

describe('dense kernels', () => {
  it('LU with row equilibration solves B y = r and Bᵀ y = r', () => {
    const B = [
      [2, 1, 0],
      [1e8, 3, 1],
      [0, 1, 4],
    ];
    const f = luFactor(B)!;
    const y = luSolve(f, [1, 2, 3]);
    B.forEach((row, i) =>
      expect(row.reduce((s, v, j) => s + v * y[j], 0)).toBeCloseTo([1, 2, 3][i], 6),
    ); // the 10⁸ row: residual relative to 10⁸
    const w = luSolve(f, [1, 2, 3], true);
    [0, 1, 2].forEach((j) =>
      expect(B.reduce((s, row, i) => s + row[j] * w[i], 0)).toBeCloseTo([1, 2, 3][j], 9),
    );
    expect(
      luFactor([
        [1, 2],
        [2, 4],
      ]),
    ).toBeNull();
  });
  it('Householder QR has orthonormal columns and reproduces M', () => {
    const M = [
      [1, 2],
      [3, 4],
      [5, 7],
      [0, 1],
    ];
    const { Q: q, R } = householderQR(M);
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 2; b++)
        expect(q.reduce((s, r) => s + r[a] * r[b], 0)).toBeCloseTo(a === b ? 1 : 0, 12);
    M.forEach((row, i) =>
      row.forEach((v, j) => expect(q[i][0] * R[0][j] + q[i][1] * R[1][j]).toBeCloseTo(v, 12)),
    );
  });
  it('singular values and rank', () => {
    expect(
      singularValues([
        [3, 0],
        [0, 4],
      ]),
    ).toEqual([4, 3]);
    expect(
      matrixRank(
        [
          [1, 2, 3],
          [2, 4, 6],
        ],
        1e-10,
      ),
    ).toBe(1);
    expect(
      matrixRank(
        [
          [1, 2, 3],
          [2, 4, 7],
        ],
        1e-10,
      ),
    ).toBe(2);
  });
});
