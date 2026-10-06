/**
 * Least-squares ports (src/problems/least_squares.ts, src/methods/unconstrained/least_squares.ts)
 * against Python, beyond the parity fixtures:
 *   - problem metadata and data equal src/generated/problems.json;
 *   - r, J, f, ∇f, ∇²f equal the Python values at x0, the minima and seeded points;
 *   - the exact derivatives agree with central differences;
 *   - 103 extra runs (tests/least-squares/fixtures, from gen_least_squares_fixture.py): other
 *     starts and parameters, budgets, overflow, rank deficiency, a non-finite start and a bare
 *     residual callable — same nIter, converged flag, message, evaluation counts, Step.info key
 *     set, and the trace within the parity tolerances;
 *   - the thin SVD and Python's `.3g` formatting.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { reviveNumbers } from '../../src/core/json';
import { getMethod } from '../../src/core/registry';
import type { Matrix, Result, Vector } from '../../src/core/types';
import { getProblem, listProblems } from '../../src/problems/registry';
import type { LeastSquaresProblem } from '../../src/problems/least_squares';
import { norm2, pyG, svdThin } from '../../src/methods/unconstrained/least_squares';
import '../../src/problems/least_squares';
import '../../src/methods/unconstrained/least_squares';

type Raw = Record<string, unknown>;
const read = <T>(rel: string): T =>
  reviveNumbers(
    JSON.parse(readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')),
  ) as T;

const META = read<Raw[]>('../../src/generated/problems.json').filter(
  (p) => p.kind === 'least_squares',
);
const REF = read<{
  problems: {
    id: string;
    x: Vector;
    r: Vector;
    J: Matrix;
    f: number;
    grad: Vector;
    hess: Matrix;
  }[];
  runs: { method: string; problem: string; params: Raw; result: Raw }[];
}>('./fixtures/least_squares_python.json');

const flat = (v: unknown): number[] =>
  typeof v === 'number' ? [v] : Array.isArray(v) ? v.flatMap(flat) : [];

function maxRel(a: unknown, b: unknown, atol = 1e-300): number {
  const xa = flat(a),
    xb = flat(b);
  if (xa.length !== xb.length) return Infinity;
  let worst = 0;
  xa.forEach((v, i) => {
    const w = xb[i];
    if (!Number.isFinite(w) || !Number.isFinite(v)) {
      if (!(Object.is(v, w) || (Number.isNaN(v) && Number.isNaN(w)))) worst = Infinity;
      return;
    }
    worst = Math.max(worst, Math.abs(v - w) / (atol + Math.abs(w)));
  });
  return worst;
}

/** The message with its numbers masked (numbers are compared through the trace instead). */
const shape = (m: string) => m.replace(/[-+]?\d+\.?\d*(?:e[-+]?\d+)?|inf|nan/g, '#');

const problemsLs = () => listProblems<LeastSquaresProblem>('least_squares');

describe('least-squares problems', () => {
  it('same ids, order and metadata as problems.json', () => {
    expect(problemsLs().map((p) => p.id)).toEqual(META.map((m) => m.id));
    for (const m of META) {
      const p = getProblem<LeastSquaresProblem>(m.id as string);
      expect(p.name).toBe(m.name);
      expect(p.latex).toBe(m.latex);
      expect(p.dim).toBe(m.dim);
      expect(p.domain).toEqual(m.domain);
      expect(p.x0).toEqual(m.x0);
      expect(p.minima).toEqual(m.minima);
      expect(p.description).toBe(m.description);
      expect(p.tags).toEqual(m.tags);
      const want = m.extra as Raw;
      expect(Object.keys(p.extra).sort()).toEqual(Object.keys(want).sort());
      for (const [k, v] of Object.entries(want)) {
        if (typeof v === 'string') expect(p.extra[k]).toBe(v);
        // Mulberry32 data + libm exp/cos/sin/hypot: within a few ulps.
        else expect(maxRel(p.extra[k], v, 1e-300)).toBeLessThanOrEqual(1e-14);
      }
    }
  });

  it('r, J, f, ∇f, ∇²f equal the Python values', () => {
    for (const c of REF.problems) {
      const p = getProblem<LeastSquaresProblem>(c.id);
      expect(maxRel(p.residual(c.x), c.r, 1e-12), `${c.id} r`).toBeLessThan(1e-11);
      expect(maxRel(p.jac(c.x), c.J, 1e-12), `${c.id} J`).toBeLessThan(1e-11);
      expect(maxRel(p.f(c.x), c.f, 1e-12), `${c.id} f`).toBeLessThan(1e-11);
      // ∇f = Jᵀr cancels near a minimizer: compare against the size of its terms, ‖J‖·‖r‖.
      const gScale = Math.max(...flat(c.J).map(Math.abs)) * Math.max(1e-300, ...c.r.map(Math.abs));
      p.grad(c.x).forEach((v, i) =>
        expect(Math.abs(v - c.grad[i]), `${c.id} grad`).toBeLessThanOrEqual(1e-12 * gScale),
      );
      expect(maxRel(p.hess(c.x), c.hess, 1e-6), `${c.id} hess`).toBeLessThan(1e-9);
    }
  });

  it('exact derivatives agree with central differences', () => {
    for (const c of REF.problems) {
      const p = getProblem<LeastSquaresProblem>(c.id);
      const x = c.x;
      const h = x.map((v) => 1e-6 * Math.max(1, Math.abs(v)));
      const shift = (i: number, s: number) => x.map((v, j) => (j === i ? v + s * h[i] : v));
      const g = p.grad(x);
      const H = p.hess(x);
      const J = p.jac(x);
      for (let i = 0; i < 2; i++) {
        const fp = p.f(shift(i, 1)),
          fm = p.f(shift(i, -1));
        const fd = (fp - fm) / (2 * h[i]);
        expect(Math.abs(fd - g[i])).toBeLessThan(1e-5 * (1 + Math.abs(g[i]) + Math.abs(p.f(x))));
        const gp = p.grad(shift(i, 1)),
          gm = p.grad(shift(i, -1));
        for (let j = 0; j < 2; j++) {
          const hd = (gp[j] - gm[j]) / (2 * h[i]);
          expect(Math.abs(hd - H[j][i])).toBeLessThan(1e-4 * (1 + Math.abs(H[j][i])));
        }
        const rp = p.residual(shift(i, 1)),
          rm = p.residual(shift(i, -1));
        rp.forEach((v, r) => {
          const jd = (v - rm[r]) / (2 * h[i]);
          expect(Math.abs(jd - J[r][i])).toBeLessThan(1e-5 * (1 + Math.abs(J[r][i])));
        });
      }
    }
  });

  it('f at the minima equals minima_f and the minima are stationary', () => {
    for (const p of problemsLs()) {
      p.minima.forEach((m, i) => {
        expect(p.f(m)).toBeCloseTo(p.extra.minima_f[i], 10);
        const g = p.grad(m);
        expect(Math.max(...g.map(Math.abs))).toBeLessThan(1e-5 * (1 + p.f(m)));
      });
    }
  });
});

/**
 * Where the two implementations may legitimately part ways. Both evaluate f = ½‖r‖², Jᵀr and the
 * SVD in different summation orders (NumPy/OpenBLAS with FMA vs plain JS; LAPACK vs Jacobi), so
 * they agree to a few ulps, not bit for bit. Two situations amplify those ulps into a different
 * discrete decision (accept/reject, which stopping test fires first), and only these two:
 *   - noise floor: f(𝐱ₖ) already equals the final f to rounding (|Δf| ≤ 10⁻¹²·f), so the gain
 *     ratio ϱ is a ratio of rounding errors (MNT §3.2; the Python module docstring's √ε limit);
 *   - ill-conditioning: κ₂(JᵀJ) ≥ 10¹⁴ at an earlier iterate (κ₂(J) ≥ 10⁷), so ulp perturbations
 *     of J move the step by ≥ 10⁻⁹ relative (Higham (2002), §20.1).
 * Everything up to that step must agree to the parity tolerance, the verdict must be the same,
 * and the final x must agree to 10⁻⁶ (noise floor) or 10⁻³ (ill-conditioned).
 */
function divergenceAllowed(want: Raw[], k: number): 'noise' | 'ill' | null {
  const fEnd = want[want.length - 1].fun as number;
  const fPrev = want[Math.max(0, k - 1)].fun as number;
  if (Number.isFinite(fPrev) && Math.abs(fPrev - fEnd) <= 1e-12 * Math.abs(fEnd)) return 'noise';
  for (let j = 0; j < k; j++) {
    const c = (want[j].info as Raw).jtj_cond;
    if (typeof c === 'number' && c >= 1e14) return 'ill';
  }
  return null;
}

describe('least-squares methods: extra Python runs', () => {
  const rosenR = (x: Vector): Vector => [10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]];
  let diverged = 0;
  REF.runs.forEach((c, idx) => {
    const name = `${c.method} on ${c.problem} ${JSON.stringify(c.params)} #${idx}`;
    it(name, () => {
      const { spec, fn } = getMethod(c.method);
      const problem = c.problem === 'bare:rosen_r' ? rosenR : getProblem(c.problem);
      const defaults = Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
      const got = fn(problem, { ...defaults, ...(c.params as Record<string, never>) }) as Result;
      const want = c.result;
      const wtrace = want.trace as Raw[];
      const scale = Math.max(
        1e-300,
        ...wtrace.map((s) => (s.info as Raw).residual_norm as number).filter(Number.isFinite),
      );

      // The first step where the traces part (iterate beyond 1e-8 or a different decision).
      let split = -1;
      const n = Math.min(got.trace.length, wtrace.length);
      for (let k = 0; k < n && split < 0; k++) {
        if (maxRel(got.trace[k].x, wtrace[k].x, 1e-10) > 1e-8) split = k;
        else if (got.trace[k].info.accepted !== (wtrace[k].info as Raw).accepted) split = k;
      }
      if (split < 0 && (got.trace.length !== wtrace.length || got.nIter !== want.n_iter)) split = n;
      const why = split < 0 ? null : divergenceAllowed(wtrace, split);
      if (split >= 0) {
        diverged++;
        expect(why, `traces part at k = ${split}`).not.toBeNull();
        expect(got.converged).toBe(want.converged);
        expect(maxRel(got.x, want.x, 1e-10)).toBeLessThanOrEqual(why === 'ill' ? 1e-3 : 1e-6);
      } else {
        expect(got.nIter).toBe(want.n_iter);
        expect(got.converged).toBe(want.converged);
        expect(shape(got.message)).toBe(shape(want.message as string));
        expect(got.nFev).toBe(want.n_fev);
        expect(got.nGev).toBe(want.n_gev);
        expect(maxRel(got.x, want.x, 1e-10)).toBeLessThanOrEqual(1e-6);
      }
      expect(got.nHev).toBe(0);
      const upto = split < 0 ? n : split;
      got.trace.slice(0, upto).forEach((s, k) => {
        const wi = wtrace[k].info as Raw;
        expect(Object.keys(s.info).sort(), `info keys at k = ${k}`).toEqual(Object.keys(wi).sort());
        for (const key of ['accepted', 'alpha'])
          if (key in wi) expect(s.info[key]).toEqual(wi[key]);
        for (const key of ['lambda', 'nu']) {
          if (!(key in wi)) continue;
          if (wi[key] === null) expect(s.info[key] ?? null).toBeNull();
          else expect(maxRel(s.info[key], wi[key]), `${key} at k = ${k}`).toBeLessThan(1e-7);
        }
        // ‖r‖ reaches 0 on zero-residual problems: compare against the run's scale.
        const rn = wi.residual_norm;
        if (rn === null) expect(s.info.residual_norm ?? null).toBeNull();
        else
          expect(
            Math.abs((s.info.residual_norm as number) - (rn as number)),
            `residual_norm at k = ${k}`,
          ).toBeLessThanOrEqual(1e-8 * scale);
      });
      // Every step of a method carries the same key set.
      const keys = JSON.stringify(Object.keys(got.trace[0].info).sort());
      for (const s of got.trace) expect(JSON.stringify(Object.keys(s.info).sort())).toBe(keys);
    });
  });
  it('parts ways only in the documented situations, and rarely', () => {
    // 10 of 103 runs end at the noise floor or meet κ₂(JᵀJ) ≥ 10¹⁴ (counted when this file runs).
    expect(diverged).toBeLessThanOrEqual(12);
  });
});

describe('helpers', () => {
  it('thin SVD reconstructs J with orthonormal factors and descending σ', () => {
    const mats: Matrix[] = [
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [-24, 10],
        [-1, 0],
      ],
      [
        [1e-8, 1],
        [2e-8, 1],
        [3e-8, 1.0000001],
      ],
      [
        [0, 0],
        [0, 0],
      ],
      [[1, 2, 3]],
    ];
    for (const A of mats) {
      const { U, s, Vt } = svdThin(A);
      for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThanOrEqual(s[i - 1]);
      A.forEach((row, i) =>
        row.forEach((v, j) => {
          const rec = s.reduce((acc, sk, k) => acc + U[i][k] * sk * Vt[k][j], 0);
          expect(Math.abs(rec - v)).toBeLessThan(1e-12 * (1 + Math.abs(v)));
        }),
      );
    }
    // σ = 2^(1/2)·… for a matrix with a known spectrum.
    const { s } = svdThin([
      [3, 0],
      [0, 4],
      [0, 0],
    ]);
    expect(s).toEqual([4, 3]);
  });

  it('pyG formats like Python format(x, ".3g")', () => {
    const cases: [number, string][] = [
      [6.71e-10, '6.71e-10'],
      [0, '0'],
      [2.04e-8, '2.04e-08'],
      [1.49e-15, '1.49e-15'],
      [5.53e-5, '5.53e-05'],
      [0.000123, '0.000123'],
      [12.1, '12.1'],
      [1171.28, '1.17e+03'],
      [100, '100'],
      [1, '1'],
      [0.5, '0.5'],
      [Infinity, 'inf'],
      [NaN, 'nan'],
      [-3.14159, '-3.14'],
    ];
    for (const [x, s] of cases) expect(pyG(x)).toBe(s);
  });

  it('norm2 neither overflows nor underflows', () => {
    expect(norm2([3e300, 4e300])).toBeCloseTo(5e300, -290);
    expect(norm2([3e-320, 4e-320])).toBeGreaterThan(0);
    expect(norm2([])).toBe(0);
  });
});
