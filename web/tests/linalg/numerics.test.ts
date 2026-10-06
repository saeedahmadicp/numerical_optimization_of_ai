/**
 * Unit tests for the linalg numerics (src/methods/linalg/numerics.ts) and the problem library
 * (src/problems/linalg.ts): what the parity fixtures do not cover directly.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  eigvals,
  eigvalsh,
  fma,
  frexpExponent,
  hypot,
  npSum,
  pyG,
  spectralRadius,
} from '../../src/methods/linalg/numerics';
import { jacobiMatrix, kappa2, optimalOmega, sorMatrix } from '../../src/methods/linalg/iterative';
import { listProblems } from '../../src/problems/registry';
import { runMethod } from '../../src/core/registry';
import type { LinalgProblem } from '../../src/problems/linalg';
import '../../src/methods/linalg/direct';
import '../../src/methods/linalg/iterative';
import '../../src/problems/linalg';

/** Exact a·b + c rounded to double, with BigInt arithmetic on the binary expansions. */
function exactFma(a: number, b: number, c: number): number {
  const parts = (x: number): [bigint, number] => {
    const buf = new DataView(new ArrayBuffer(8));
    buf.setFloat64(0, x);
    const hi = buf.getUint32(0),
      lo = buf.getUint32(4);
    const sign = hi >>> 31 ? -1n : 1n;
    const e = (hi >>> 20) & 0x7ff;
    let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
    if (e === 0) return [sign * m, -1074];
    m |= 1n << 52n;
    return [sign * m, e - 1075];
  };
  const [ma, ea] = parts(a),
    [mb, eb] = parts(b),
    [mc, ec] = parts(c);
  const ep = ea + eb;
  const e0 = Math.min(ep, ec);
  const total = ((ma * mb) << BigInt(ep - e0)) + (mc << BigInt(ec - e0));
  // Round total·2^e0 to the nearest double (ties to even) via a decimal-free path.
  if (total === 0n) return 0;
  const neg = total < 0n;
  let t = neg ? -total : total;
  let e = e0;
  const bits = t.toString(2).length;
  if (bits > 53) {
    const shift = BigInt(bits - 53);
    const rem = t & ((1n << shift) - 1n);
    t >>= shift;
    e += Number(shift);
    const half = 1n << (shift - 1n);
    if (rem > half || (rem === half && (t & 1n) === 1n)) t += 1n;
  }
  const v = Number(t) * 2 ** e;
  return neg ? -v : v;
}

describe('fma', () => {
  it('is correctly rounded on random and adversarial inputs', () => {
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    for (let i = 0; i < 4000; i++) {
      const a = (rnd() - 0.5) * 2 ** Math.floor(rnd() * 40 - 20);
      const b = (rnd() - 0.5) * 2 ** Math.floor(rnd() * 40 - 20);
      // c near −a·b exposes cancellation; otherwise random.
      const c = i % 3 === 0 ? -(a * b) * (1 + (rnd() - 0.5) * 1e-12) : (rnd() - 0.5) * 4;
      expect(fma(a, b, c)).toBe(exactFma(a, b, c));
    }
  });
  it('handles ties broken by the low-order part', () => {
    // 1 + 2⁻⁵³ is a tie; (1 + 2⁻⁵²)(1 + 2⁻⁵²) + … needs the fused low bits.
    const a = 1 + 2 ** -52;
    expect(fma(a, a, -1)).toBe(exactFma(a, a, -1));
    expect(fma(a, a, 2 ** -53)).toBe(exactFma(a, a, 2 ** -53));
    expect(fma(1e300, 1.5e7, -1e307)).toBe(exactFma(1e300, 1.5e7, -1e307));
  });
});

describe('formatting and scalars', () => {
  it("formats like Python's '.3g'", () => {
    expect(pyG(1.6e-16)).toBe('1.6e-16');
    expect(pyG(0.47140452079103173)).toBe('0.471');
    expect(pyG(12345.678)).toBe('1.23e+04');
    expect(pyG(100)).toBe('100');
    expect(pyG(0.0001234)).toBe('0.000123');
    expect(pyG(1.118033988749895, 4)).toBe('1.118');
    expect(pyG(Infinity)).toBe('inf');
    expect(pyG(NaN)).toBe('nan');
    expect(pyG(0)).toBe('0');
    expect(pyG(-2.5e-5)).toBe('-2.5e-05');
  });
  it('frexp exponents', () => {
    expect(frexpExponent(1)).toBe(1);
    expect(frexpExponent(0.5)).toBe(0);
    expect(frexpExponent(0.75)).toBe(0);
    expect(frexpExponent(2 ** 100)).toBe(101);
    expect(frexpExponent(3e-300)).toBe(Math.floor(Math.log2(3e-300)) + 1);
  });
  it('npSum is pairwise beyond 8 terms', () => {
    const v = Array.from({ length: 20 }, (_, i) => 1 / (i + 1));
    const r = v.slice(0, 8);
    for (let j = 0; j < 8; j++) r[j] += v[8 + j];
    let want = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]));
    for (let i = 16; i < 20; i++) want += v[i];
    expect(npSum(v)).toBe(want);
  });
  it('hypot is accurate', () => {
    expect(hypot(3, 4)).toBe(5);
    expect(hypot(1e300, 1e300)).toBeCloseTo(Math.SQRT2 * 1e300, -285);
    expect(hypot(0, -2)).toBe(2);
  });
});

describe('eigenvalues', () => {
  it('finds a complex pair (rotation) and real eigenvalues', () => {
    const rot = eigvals([
      [0, -1],
      [1, 0],
    ]);
    expect(rot.map((z) => Math.abs(z.im)).sort()).toEqual([1, 1]);
    expect(
      spectralRadius([
        [2, 1],
        [1, 2],
      ]),
    ).toBeCloseTo(3, 14);
  });
  it('matches the documented spectral radii of the problems', () => {
    const A = (id: string) => listProblems<LinalgProblem>('linalg').find((p) => p.id === id)!.A;
    expect(spectralRadius(jacobiMatrix(A('spd_2x2')))).toBeCloseTo(Math.SQRT2 / 3, 14);
    expect(spectralRadius(sorMatrix(A('spd_2x2'), 1))).toBeCloseTo(2 / 9, 14);
    expect(spectralRadius(jacobiMatrix(A('jacobi_diverges')))).toBeCloseTo(Math.sqrt(5) / 2, 13);
    expect(spectralRadius(sorMatrix(A('jacobi_diverges'), 1))).toBeCloseTo(0.5, 13);
    expect(spectralRadius(jacobiMatrix(A('poisson_1d_10')))).toBeCloseTo(
      Math.cos(Math.PI / 11),
      13,
    );
    const young = optimalOmega(A('poisson_1d_10'), 1.5);
    expect(young.omega_opt).toBeCloseTo(2 / (1 + Math.sin(Math.PI / 11)), 12);
    expect(spectralRadius(sorMatrix(A('poisson_1d_10'), young.omega_opt!))).toBeCloseTo(
      young.omega_opt! - 1,
      5, // ρ(G_ω) has a square-root singularity at ω*: a 1e-16 change in ω moves it by 1e-8
    );
    // A complex pair 5.5 ± 2.1i on nonsymmetric_4 (documented)
    const z = eigvals(A('nonsymmetric_4')).filter((e) => Math.abs(e.im) > 1e-9);
    expect(z).toHaveLength(2);
    z.forEach((e) => expect(e.re).toBeCloseTo(5.5, 0));
  });
  it('symmetric eigenvalues and condition numbers', () => {
    expect(
      eigvalsh([
        [3, 2],
        [2, 6],
      ]),
    ).toEqual([expect.closeTo(2, 14), expect.closeTo(7, 14)]);
    const pois = listProblems<LinalgProblem>('linalg').find((p) => p.id === 'poisson_1d_10')!;
    const lam = eigvalsh(pois.A);
    lam.forEach((l, j) => expect(l).toBeCloseTo(2 - 2 * Math.cos(((j + 1) * Math.PI) / 11), 13));
    expect(kappa2(pois.A)).toBeCloseTo(48.374, 2);
    expect(
      kappa2([
        [1, 0],
        [0, -1],
      ]),
    ).toBeNull();
  });
});

describe('problem library', () => {
  type Raw = Record<string, unknown>;
  const META = JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../src/generated/problems.json', import.meta.url)),
      'utf8',
    ),
  ) as Raw[];
  const want = META.filter((m) => m.kind === 'linalg');
  const got = listProblems<LinalgProblem>('linalg');

  it('has the same ids in the same order', () => {
    expect(got.map((p) => p.id)).toEqual(want.map((m) => m.id));
  });
  it('reproduces every field of problems.json exactly', () => {
    for (const m of want) {
      const p = got.find((q) => q.id === m.id)!;
      expect({
        name: p.name,
        A: p.A,
        b: p.b,
        solution: p.solution,
        description: p.description,
        tags: p.tags,
      }).toEqual({
        name: m.name,
        A: m.A,
        b: m.b,
        solution: m.solution,
        description: m.description,
        tags: m.tags,
      });
      expect(p.n).toBe(p.b.length);
      expect(p.latex.length).toBeGreaterThan(5);
    }
  });
  it('freezes the shared data', () => {
    const p = got[0];
    expect(Object.isFrozen(p.A)).toBe(true);
    expect(Object.isFrozen(p.A[0])).toBe(true);
    expect(Object.isFrozen(p.b)).toBe(true);
  });
});

describe('failure paths', () => {
  const P = (id: string) => listProblems<LinalgProblem>('linalg').find((p) => p.id === id)!;
  it('reports zero pivots, non-SPD input and inapplicable methods without throwing', () => {
    const ge = runMethod('gaussian_elimination', P('needs_pivoting'));
    expect(ge.converged).toBe(false);
    expect(ge.message).toMatch(/zero pivot .* at stage 1/);
    expect(ge.trace.at(-1)!.info.zero_pivot).toBe(true);
    expect(runMethod('gaussian_elimination_pivoting', P('needs_pivoting')).converged).toBe(true);
    expect(runMethod('lu_decomposition', P('singular_3')).message).toMatch(/singular/);
    expect(runMethod('cholesky', P('nonsymmetric_4')).message).toMatch(/not symmetric/);
    expect(runMethod('conjugate_gradient_linear', P('nonsymmetric_4')).converged).toBe(false);
    const jd = runMethod('jacobi', P('jacobi_diverges'));
    expect(jd.converged).toBe(false);
    expect(jd.message).toMatch(/ρ\(G\) = 1\.118 ≥ 1/);
    expect(runMethod('jacobi', P('needs_pivoting')).message).toMatch(/a_00 = 0/);
  });
  it('raises on invalid input like Python', () => {
    expect(() => runMethod('thomas', P('nonsymmetric_4'))).toThrow(/tridiagonal/);
    expect(() => runMethod('sor', P('spd_2x2'), { omega: 2 })).toThrow(/omega/);
    expect(() => runMethod('jacobi', P('spd_2x2'), { tol: 0 })).toThrow(/tol/);
    expect(() => runMethod('jacobi', P('spd_2x2'), { x0: [1, 2, 3] })).toThrow(/x0 has 3/);
    expect(() => runMethod('gmres', P('spd_2x2'), { restart: 0 })).toThrow(/restart/);
  });
  it('CG ends in n steps on the 2×2 SPD system (finite termination)', () => {
    const r = runMethod('conjugate_gradient_linear', P('spd_2x2'), { x0: [-2, 2] });
    expect(r.converged).toBe(true);
    expect(r.nIter).toBe(2);
    expect(r.x as number[]).toEqual([expect.closeTo(2, 12), expect.closeTo(-2, 12)]);
  });
  it('Gauss–Seidel sweeps land on the lines of the 2×2 system', () => {
    const p = P('spd_2x2');
    const r = runMethod('gauss_seidel', p, { x0: [-2, 2] });
    const sweep = r.trace[1].info.sweep as number[][];
    // after the first component update, equation 1 holds; after the second, equation 2 holds
    expect(p.A[0][0] * sweep[1][0] + p.A[0][1] * sweep[1][1]).toBeCloseTo(p.b[0], 12);
    expect(p.A[1][0] * sweep[2][0] + p.A[1][1] * sweep[2][1]).toBeCloseTo(p.b[1], 12);
  });
});
