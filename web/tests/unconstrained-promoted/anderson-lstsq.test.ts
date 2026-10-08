/**
 * anderson_gd's `np.linalg.lstsq` and `M @ v` are bit-exact ports (LAPACK dgelsd and the
 * OpenBLAS kernels of the NumPy wheel): the AA iteration on a curved valley amplifies a one-ulp
 * difference in γ* into a different iteration count. Reference values come from
 * gen_anderson_lstsq_fixture.py.
 *
 * Bit for bit holds only against the canonical NumPy wheel, whose kernels the port replays
 * (tests/fixtures/platform.ts). Another platform's dump (an x86-64 OpenBLAS, another LAPACK
 * build) is compared with the error bounds of the computations instead: the same numerical rank
 * (rcond = ε·max(m, n)), σ within 2·max(m, n)·ε·σ₁ (a backward-stable SVD), x within
 * 8·max(m, n)·κ²·ε·max(1, ‖x‖∞) (least-squares perturbation theory, κ = σ₁/σ_rank), and M @ v
 * within n·ε·Σ|M_ij v_j| (the dot-product bound). Measured on x86-64: at most 1.3, 0.75 and 0.17
 * of these bounds without the factors.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fmaFast, lstsq, matvec } from '../../src/methods/unconstrained/anderson';
import { fma } from '../../src/methods/unconstrained/trust_region';
import { CANONICAL } from '../fixtures/platform';

interface Fixture {
  lstsq: { A: number[][]; b: number[]; x: number[]; s: number[] }[];
  matvec: { M: number[][]; v: number[]; y: number[] }[];
}

const FIX = JSON.parse(
  readFileSync(fileURLToPath(new URL('./fixtures/anderson_lstsq.json', import.meta.url)), 'utf8'),
) as Fixture;

const EPS = 2 ** -52;
const maxAbsDiff = (a: readonly number[], b: readonly number[]) =>
  Math.max(0, ...a.map((v, i) => Math.abs(v - b[i])));

/** Off the canonical platform: x and σ of lstsq within their error bounds (module comment). */
function withinLstsqBounds(c: Fixture['lstsq'][number], x: number[], s: number[]): boolean {
  const m = c.A.length;
  const n = c.A[0].length;
  const k = Math.max(m, n);
  const s1 = c.s[0] ?? 0;
  const rank = (v: readonly number[]) => v.filter((t) => t > EPS * k * (v[0] ?? 0)).length;
  if (rank(s) !== rank(c.s) || x.length !== c.x.length || s.length !== c.s.length) return false;
  const r = rank(c.s);
  const kappa = r ? s1 / c.s[r - 1] : 1;
  const xs = Math.max(1, ...c.x.map(Math.abs));
  return (
    maxAbsDiff(s, c.s) <= 2 * k * EPS * s1 && maxAbsDiff(x, c.x) <= 8 * k * kappa * kappa * EPS * xs
  );
}

/** Equal as IEEE doubles (so 0 and −0 differ, NaN equals NaN). */
const same = (a: readonly number[], b: readonly number[]) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

describe('anderson_gd linear algebra, bit for bit against NumPy', () => {
  it('lstsq reproduces np.linalg.lstsq(A, b, rcond=None): x and the singular values', () => {
    const bad = FIX.lstsq.filter((c) => {
      const { x, s } = lstsq(c.A, c.b);
      if (!CANONICAL) return !withinLstsqBounds(c, x, s);
      // JSON has no −0: compare x + 0 (turns −0 into 0) on both sides.
      return (
        !same(
          x.map((t) => t + 0),
          c.x.map((t) => t + 0),
        ) || !same(s, c.s)
      );
    });
    expect(FIX.lstsq.length).toBeGreaterThan(100);
    expect(bad.map((c) => `${c.A.length}×${c.A[0].length}`)).toEqual([]);
  });

  it('matvec reproduces the C-contiguous M @ v of NumPy (two-lane SVE dgemv_t)', () => {
    const bad = FIX.matvec.filter((c) => {
      const y = matvec(c.M, c.v);
      if (CANONICAL) return !same(y, c.y);
      return c.M.some(
        (row, i) =>
          Math.abs(y[i] - c.y[i]) >
          row.length * EPS * row.reduce((t, a, j) => t + Math.abs(a * c.v[j]), 0),
      );
    });
    expect(bad.length).toBe(0);
  });

  it('fmaFast rounds a·b + c once, as the exact BigInt fma does', () => {
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const any = () => (rnd() - 0.5) * 2 ** Math.floor(rnd() * 80 - 40);
    let bad = 0;
    for (let k = 0; k < 50_000; k++) {
      const a = any();
      const b = any();
      const p = a * b;
      const c = [
        any(),
        -p, // exact cancellation of the rounded product
        -p * (1 + (Math.floor(rnd() * 16) - 8) * 2 ** -52),
        -p + any() * 2 ** -60 * Math.abs(p),
        p * 2 ** (Math.floor(rnd() * 120) - 60),
      ][k % 5];
      if (!Object.is(fmaFast(a, b, c), fma(a, b, c))) bad++;
    }
    // Products that need 54 bits plus a small c: ties and near-ties of the final rounding.
    for (let k = 0; k < 20_000; k++) {
      const a = 1 + Math.floor(rnd() * 2 ** 26) * 2 ** -26;
      const b = 1 + Math.floor(rnd() * 2 ** 27) * 2 ** -27;
      const c = (rnd() - 0.5) * 2 ** (Math.floor(rnd() * 10) - 50);
      if (!Object.is(fmaFast(a, b, c), fma(a, b, c))) bad++;
    }
    // Ranges where the error-free transforms would under- or overflow use the BigInt path.
    for (const [a, b, c] of [
      [1e-200, 1e-200, 2 ** -1030],
      [1e300, 1e10, -Infinity],
      [2 ** 1000, 2 ** 20, -(2 ** 1020)],
      [3e-170, 7e-170, -(2 ** -1070)],
    ])
      if (!Object.is(fmaFast(a, b, c), fma(a, b, c))) bad++;
    expect(bad).toBe(0);
  });
});
