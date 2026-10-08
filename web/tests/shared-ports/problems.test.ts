/**
 * Problem-library ports (src/problems/{unconstrained,calculus,data}.ts) against Python:
 *   - same ids, order and metadata as src/generated/problems.json;
 *   - f, ∇f, ∇²f equal the Python values at x0, the minima and seeded random points
 *     (tests/shared-ports/fixtures, from gen_shared_ports_fixture.py);
 *   - the exact derivatives agree with central differences;
 *   - the datasets (Mulberry32 noise in Python order) equal problems.json.
 */
import { describe, expect, it } from 'vitest';
import { CANONICAL } from '../fixtures/platform';
import type { Dataset, Matrix, Problem, Vector } from '../../src/core/types';
import { listProblems } from '../../src/problems/registry';
import '../../src/problems';
import {
  buildQuadraticNd,
  is2D,
  listUnconstrained2D,
  npSum,
  type UnconstrainedProblem,
} from '../../src/problems/unconstrained';
import type { CalculusProblem } from '../../src/problems/calculus';
import { chebyshevNodesFirstKind, linspace } from '../../src/problems/data';
import { flat, maxRelDiff, mismatch, readJson } from './compare';

type Raw = Record<string, unknown>;
const META = readJson<Raw[]>('../../src/generated/problems.json');
const metaOf = (kind: string) => META.filter((m) => m.kind === kind);

interface ValueCase {
  id: string;
  x: number | number[];
  f: number | null;
  grad: number | number[] | null;
  hess: number | number[][] | null;
}
const VALUES = readJson<{ problems: ValueCase[] }>('./fixtures/shared_ports_python.json').problems;

/** Metadata of a TS problem in the shape of problems.json. */
type Meta = Pick<
  Problem<unknown>,
  'id' | 'name' | 'latex' | 'dim' | 'domain' | 'bracket' | 'constraints' | 'exact'
> &
  Pick<Problem<unknown>, 'description' | 'tags' | 'extra'> & {
    x0?: unknown;
    minima?: unknown;
    roots?: unknown;
  };

function metaFromTs(kind: string, p: Meta): Raw {
  return {
    kind,
    id: p.id,
    name: p.name,
    latex: p.latex,
    dim: p.dim,
    domain: p.domain,
    x0: p.x0 ?? null,
    bracket: p.bracket ?? null,
    minima: p.minima ?? [],
    roots: p.roots ?? [],
    constraints: p.constraints ?? [],
    exact: p.exact ?? null,
    description: p.description ?? '',
    tags: p.tags ?? [],
    extra: p.extra ?? {},
  };
}

const EXACT = { rtol: 0, atol: 0 };

describe('unconstrained problems', () => {
  const ts = listProblems<UnconstrainedProblem>('unconstrained');
  const py = metaOf('unconstrained');

  it('has the Python ids in the Python order', () => {
    expect(ts.map((p) => p.id)).toEqual(py.map((m) => m.id));
  });

  for (const want of py) {
    it(`${String(want.id)}: metadata equals problems.json`, () => {
      const p = ts.find((q) => q.id === want.id)!;
      const got = metaFromTs('unconstrained', p);
      if (want.id === 'quadratic_nd') {
        // A is rebuilt from Rng(20) + a Householder product; BLAS sums in another order.
        const { A, ...rest } = got.extra as Raw;
        const { A: wantA, ...wantRest } = want.extra as Raw;
        expect(maxRelDiff(flat(A), flat(wantA))).toBeLessThan(1e-14);
        expect(mismatch({ ...got, extra: rest }, { ...want, extra: wantRest }, EXACT)).toBeNull();
      } else {
        expect(mismatch(got, want, EXACT)).toBeNull();
      }
    });
  }

  it('reproduces f, ∇f, ∇²f of Python at x0, the minima and random points', () => {
    const cases = VALUES.filter((c) => ts.some((p) => p.id === c.id));
    expect(cases.length).toBeGreaterThan(150);
    let exact = 0,
      total = 0,
      worst = 0;
    for (const c of cases) {
      const p = ts.find((q) => q.id === c.id)!;
      const x = c.x as Vector;
      const got = [p.f(x), ...p.grad(x), ...flat(p.hess(x))];
      const want = [c.f as number, ...flat(c.grad), ...flat(c.hess)];
      // Scale: relative to the largest magnitude of the same quantity (gradients near 0 at minima).
      const scale = (v: number[]) => Math.max(1, ...v.map(Math.abs));
      const groups: [number[], number[]][] = [
        [got.slice(0, 1), want.slice(0, 1)],
        [got.slice(1, 1 + x.length), want.slice(1, 1 + x.length)],
        [got.slice(1 + x.length), want.slice(1 + x.length)],
      ];
      for (const [g, w] of groups) {
        const d = maxRelDiff(g, w, scale(w));
        worst = Math.max(worst, d);
        expect(d, `${c.id} at ${JSON.stringify(x)}`).toBeLessThan(1e-14);
        g.forEach((v, i) => {
          total++;
          if (Object.is(v, w[i]) || v === w[i]) exact++;
        });
      }
    }
    // Most values are bit-identical; the rest differ by libm (pow/sin/exp) or BLAS rounding.
    // That share describes the canonical platform (tests/fixtures/platform.ts); another one
    // rounds its own way (42 % bit-identical on x86-64), so there only the 1e-14 bound holds.
    if (CANONICAL) expect(exact / total).toBeGreaterThan(0.6);
    console.info(
      `[shared-ports] unconstrained: ${exact}/${total} values bit-identical, worst rel diff ${worst.toExponential(2)}`,
    );
  });

  it('exact derivatives agree with central differences at random points', () => {
    let seed = 12345;
    const rand = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    for (const p of ts) {
      for (let t = 0; t < 5; t++) {
        const x = p.domain.map(([lo, hi]) => lo + (hi - lo) * (0.1 + 0.8 * rand()));
        if (p.id === 'ackley' && Math.hypot(...x) < 1e-3) continue;
        const g = p.grad(x),
          H = p.hess(x);
        for (let i = 0; i < x.length; i++) {
          const h = 1e-6 * Math.max(1, Math.abs(x[i]));
          const xp = x.slice(),
            xm = x.slice();
          xp[i] += h;
          xm[i] -= h;
          const fd = (p.f(xp) - p.f(xm)) / (2 * h);
          expect(Math.abs(fd - g[i])).toBeLessThan(
            1e-5 * Math.max(1, Math.abs(g[i]), Math.abs(p.f(x))),
          );
          const gp = p.grad(xp),
            gm = p.grad(xm);
          for (let j = 0; j < x.length; j++) {
            const fdH = (gp[j] - gm[j]) / (2 * h);
            expect(Math.abs(fdH - H[j][i])).toBeLessThan(
              1e-4 * Math.max(1, Math.abs(H[j][i]), ...g.map(Math.abs)),
            );
          }
        }
        // symmetric Hessian (to rounding: Python adds the outer products of goldstein_price in a
        // different order for (0, 1) and (1, 0))
        H.forEach((row, r) =>
          row.forEach((v, c) =>
            expect(Math.abs(v - H[c][r])).toBeLessThanOrEqual(1e-14 * Math.max(1, Math.abs(v))),
          ),
        );
      }
    }
  });

  it('every listed minimizer is stationary and has the listed value', () => {
    for (const p of ts) {
      p.minima.forEach((m, i) => {
        const g = p.grad(m);
        const gs = Math.max(1, Math.abs(p.f(m)));
        expect(Math.max(...g.map(Math.abs)), `${p.id} minimum ${i}`).toBeLessThan(1e-6 * gs * 10);
        expect(Math.abs(p.f(m) - p.extra.minima_f[i])).toBeLessThan(1e-9 * gs);
      });
      expect(p.extra.f_min).toBe(p.extra.minima_f[0]);
    }
  });

  it('quadratic_nd: the TS construction reproduces A, c and λ from Rng(20)', () => {
    const want = py.find((m) => m.id === 'quadratic_nd')!.extra as Raw;
    const { A, c, lam } = buildQuadraticNd();
    expect(c).toEqual(want.c);
    expect(maxRelDiff(lam, want.eigenvalues as number[], 1)).toBeLessThan(1e-15);
    expect(maxRelDiff(flat(A), flat(want.A), 1)).toBeLessThan(1e-14);
    // symmetric
    A.forEach((row, i) => row.forEach((v, j) => expect(v).toBe(A[j][i])));
  });

  it('quadratic_ill: the pinned A equals Q diag(1, 50) Qᵀ', () => {
    const Q = [
      [0.8, -0.6],
      [0.6, 0.8],
    ];
    const D = [1, 50];
    const A: Matrix = [0, 1].map((i) =>
      [0, 1].map((j) => Q[i][0] * D[0] * Q[j][0] + Q[i][1] * D[1] * Q[j][1]),
    );
    const p = ts.find((q) => q.id === 'quadratic_ill')!;
    expect(maxRelDiff(flat(p.extra.A), flat(A), 1)).toBeLessThan(1e-15);
  });

  it('npSum reproduces NumPy pairwise summation', () => {
    // 9 terms: NumPy sums ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)), then + a8
    const a = [1e16, 1, -1e16, 1, 3, 1e-3, 7, -2, 0.5];
    const want = a[0] + a[1] + (a[2] + a[3]) + (a[4] + a[5] + (a[6] + a[7])) + a[8];
    expect(npSum(a)).toBe(want);
    expect(npSum([1, 2, 3])).toBe(6);
    expect(Object.is(npSum([]), -0)).toBe(true);
    const big = Array.from({ length: 300 }, (_, i) => Math.sin(i) * 10 ** (i % 7));
    const seq = big.reduce((s, v) => s + v, 0);
    expect(Math.abs(npSum(big) - seq)).toBeLessThan(1e-9);
  });

  it('listUnconstrained2D returns only 2-D problems (n-D ones stay registered for the CLI/tests)', () => {
    const twoD = listUnconstrained2D();
    expect(twoD.every(is2D)).toBe(true);
    expect(twoD.map((p) => p.id)).not.toContain('rosenbrock_nd');
    expect(twoD.map((p) => p.id)).not.toContain('quadratic_nd');
    expect(twoD.length).toBe(16);
  });
});

describe('calculus problems', () => {
  const ts = listProblems<CalculusProblem>('calculus');
  const py = metaOf('calculus');

  it('has the Python ids in the Python order and identical metadata', () => {
    expect(ts.map((p) => p.id)).toEqual(py.map((m) => m.id));
    for (const want of py) {
      const p = ts.find((q) => q.id === want.id)!;
      expect(mismatch(metaFromTs('calculus', p), want, EXACT), String(want.id)).toBeNull();
    }
  });

  it("reproduces f, f', f'' of Python", () => {
    let exact = 0,
      total = 0;
    for (const c of VALUES.filter((v) => ts.some((p) => p.id === v.id))) {
      const p = ts.find((q) => q.id === c.id)!;
      const x = c.x as number;
      const got = [p.f(x), p.grad(x), p.hess(x)];
      const want = [c.f, c.grad, c.hess].map((v) => (v === null ? NaN : (v as number)));
      got.forEach((v, i) => {
        const w = want[i];
        total++;
        if (Object.is(v, w) || v === w) exact++;
        else
          expect(Math.abs(v - w), `${c.id}[${i}] at ${x}`).toBeLessThan(
            1e-14 * Math.max(1, Math.abs(w)),
          );
      });
    }
    expect(total).toBeGreaterThan(150);
    expect(exact / total).toBeGreaterThan(0.9);
  });

  it("f' and f'' agree with central differences (away from kinks and singularities)", () => {
    for (const p of ts) {
      const [a, b] = p.domain;
      for (let t = 1; t < 10; t++) {
        const x = a + ((b - a) * t) / 10;
        if (p.id === 'abs_kink' && Math.abs(x - 0.3) < 1e-3) continue;
        const h = 1e-5;
        const fd1 = (p.f(x + h) - p.f(x - h)) / (2 * h);
        const fd2 = (p.grad(x + h) - p.grad(x - h)) / (2 * h);
        expect(Math.abs(fd1 - p.grad(x)), `${p.id} f' at ${x}`).toBeLessThan(
          1e-6 * Math.max(1, Math.abs(p.grad(x))),
        );
        expect(Math.abs(fd2 - p.hess(x)), `${p.id} f'' at ${x}`).toBeLessThan(
          1e-5 * Math.max(1, Math.abs(p.hess(x))),
        );
      }
    }
  });
});

describe('datasets', () => {
  const ts = listProblems<Dataset>('data');
  const py = metaOf('data');

  it('has the Python ids in the Python order', () => {
    expect(ts.map((d) => d.id)).toEqual(py.map((m) => m.id));
  });

  for (const want of py) {
    it(`${String(want.id)}: x, y and metadata equal problems.json`, () => {
      const d = ts.find((q) => q.id === want.id)!;
      const { x, y, ...meta } = want;
      expect(
        mismatch(
          {
            kind: 'data',
            id: d.id,
            name: d.name,
            latex: d.latex,
            domain: d.domain,
            description: d.description,
          },
          meta,
          EXACT,
        ),
      ).toBeNull();
      // x is exact (linspace / Chebyshev cos); y to libm precision (Box–Muller log/cos, exp).
      expect(d.x).toEqual(x);
      expect(maxRelDiff(d.y, y as number[], 1)).toBeLessThan(1e-14);
    });
  }

  it('almost every y value is bit-identical to Python (V8 and glibc libm agree on most draws)', () => {
    let exact = 0,
      total = 0;
    for (const want of py) {
      const d = ts.find((q) => q.id === want.id)!;
      (want.y as number[]).forEach((v, i) => {
        total++;
        if (d.y[i] === v) exact++;
      });
    }
    console.info(`[shared-ports] datasets: ${exact}/${total} y values bit-identical`);
    expect(exact / total).toBeGreaterThan(0.95);
  });

  it('fTrue is the noise-free truth (absent for Anscombe)', () => {
    const byId = Object.fromEntries(ts.map((d) => [d.id, d]));
    expect(byId.anscombe_1.fTrue).toBeUndefined();
    expect(byId.noisy_linear.fTrue!(4)).toBe(4);
    expect(byId.noisy_quadratic.fTrue!(2)).toBe(1);
    expect(byId.outliers_linear.fTrue!(3)).toBe(7);
    expect(byId.exponential_growth.fTrue!(0)).toBe(2);
    expect(byId.step_data.fTrue!(0)).toBe(1);
    expect(byId.runge_equispaced.fTrue!(0.2)).toBe(0.5);
  });

  it('arrays are read-only and helpers follow NumPy', () => {
    const d = ts[0];
    expect(Object.isFrozen(d.x)).toBe(true);
    expect(linspace(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(linspace(2, 2, 3)).toEqual([2, 2, 2]);
    const c = chebyshevNodesFirstKind(3);
    expect(c[0]).toBeLessThan(c[1]);
    expect(c[1]).toBeCloseTo(0, 15);
  });
});
