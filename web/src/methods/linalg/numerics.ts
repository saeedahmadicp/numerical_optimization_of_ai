/**
 * Shared numerics of the linalg ports — the helpers at the top of
 * `src/numopt/linalg/direct.py` (resolve_system, norm2, pivot_tolerance, triangular solves,
 * Hager–Higham condition estimate, backward error) plus the NumPy pieces the Python modules use:
 * `np.linalg.eigvals` (general real matrix), `np.linalg.eigvalsh` (symmetric), `np.frexp`, and
 * Python's `format(v, '.3g')` for messages.
 *
 * This module registers nothing; `direct.ts` and `iterative.ts` import it.
 */
import type { Matrix, Vector } from '../../core/types';

/** Machine epsilon ε = 2⁻⁵² of IEEE double precision (twice the unit roundoff u = 2⁻⁵³). */
export const EPS = 2 ** -52;

/** A square system as the methods accept it: a LinearSystem-like `{A, b}` or a pair `[A, b]`. */
export type SystemLike =
  | { A: readonly (readonly number[])[]; b: readonly number[] }
  | readonly [readonly (readonly number[])[], readonly number[]];

// ── Small dense helpers (plain arrays; the Python order of operations) ─────────────────

export function zeros(n: number): Vector {
  return new Array<number>(n).fill(0);
}

export function zerosM(n: number, m = n): Matrix {
  return Array.from({ length: n }, () => new Array<number>(m).fill(0));
}

export function eye(n: number): Matrix {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function copyM(A: readonly (readonly number[])[]): Matrix {
  return A.map((r) => r.slice());
}

export function transposeM(A: readonly (readonly number[])[]): Matrix {
  const n = A.length;
  const m = A[0]?.length ?? 0;
  return Array.from({ length: m }, (_, j) => Array.from({ length: n }, (_, i) => A[i][j]));
}

/** Σ a_i b_i, left to right. */
export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** A·x as NumPy computes `A @ x` (see `gemv`). */
export function mv(A: readonly (readonly number[])[], x: readonly number[]): Vector {
  return gemv(A, x);
}

export function allFinite(v: readonly number[]): boolean {
  for (const x of v) if (!Number.isFinite(x)) return false;
  return true;
}

export function allFiniteM(A: readonly (readonly number[])[]): boolean {
  return A.every(allFinite);
}

/** Max |v_i| (0 for an empty v); NaN propagates like `np.max`. */
export function maxAbs(v: readonly number[]): number {
  let s = 0;
  for (const x of v) {
    const a = Math.abs(x);
    if (Number.isNaN(a)) return NaN;
    if (a > s) s = a;
  }
  return s;
}

export function maxAbsM(A: readonly (readonly number[])[]): number {
  let s = 0;
  for (const r of A) {
    const a = maxAbs(r);
    if (Number.isNaN(a)) return NaN;
    if (a > s) s = a;
  }
  return s;
}

// ── Floating-point kernels as NumPy computes them ─────────────────────────────────────
//
// The Python fixtures come from NumPy on OpenBLAS (aarch64, SVE kernels with fused multiply-add).
// Ill-conditioned Krylov runs (CG on the Hilbert matrix) amplify a 1-ulp difference in a dot
// product to 1e-4 within n steps, so the ports reproduce NumPy's arithmetic bit for bit:
//   x @ y   (contiguous)   OpenBLAS dot_kernel_sve: 4 FMA lanes, then a 2-lane tail, see blasDot
//   x @ y   (strided)      OpenBLAS dot_kernel_asimd: one sequential FMA chain, see dotStrided
//   A @ x                  OpenBLAS gemv_t_sve: per row, 2 FMA lanes, see gemv
//   A @ V[:, j]            OpenBLAS gemv_t, strided x: one FMA chain per row, see gemvStrided
//   x @ A                  OpenBLAS gemv_n_sve_v1x3: columns in blocks of three, see vecMat
//   np.sum                 NumPy pairwise summation, see npSum
// Each was checked against NumPy on random data (every case matched exactly).

const SPLIT = 134217729; // 2^27 + 1 (Dekker)
const F64 = new Float64Array(1);
const I64 = new BigInt64Array(F64.buffer);

/** The double next to r in the direction of `dir` (nextafter). */
function nextToward(r: number, dir: number): number {
  if (r === 0) return dir > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  F64[0] = r;
  I64[0] += r > 0 === dir > 0 ? 1n : -1n;
  return F64[0];
}

/** Correctly rounded a·b + c (IEEE fusedMultiplyAdd), by error-free transformations. */
export function fma(a: number, b: number, c: number): number {
  const p = a * b;
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return p + c;
  if (a === 0 || b === 0) return p + c;
  const ap = Math.abs(p);
  if (!(ap < 1e290 && ap > 1e-290 && Math.abs(a) < 1e290 && Math.abs(b) < 1e290)) {
    // Near the ends of the range Dekker's split overflows or its error term underflows: scale a
    // and b to [0.5, 1) by powers of two (exact), fuse there, and scale back (exact in range).
    const sa = 2 ** -frexpExponent(Math.abs(a));
    const sb = 2 ** -frexpExponent(Math.abs(b));
    const c1 = c * sa * sb;
    if (!Number.isFinite(c1)) return p + c;
    return fma(a * sa, b * sb, c1) / sa / sb;
  }
  // a·b = p + e exactly (Dekker's TwoProduct)
  let t = SPLIT * a;
  const ah = t - (t - a),
    al = a - ah;
  t = SPLIT * b;
  const bh = t - (t - b),
    bl = b - bh;
  const e = ah * bh - p + ah * bl + al * bh + al * bl;
  // p + c = s + q exactly (TwoSum)
  const s = p + c;
  let z = s - p;
  const q = p - (s - z) + (c - z);
  // q + e = u + v exactly
  const u = q + e;
  z = u - q;
  const v = q - (u - z) + (e - z);
  // s + u = r + w exactly; the exact result is r + w + v
  const r = s + u;
  z = r - s;
  const w = s - (r - z) + (u - z);
  if (w === 0) return r + v;
  if (v === 0 || Math.sign(v) !== Math.sign(w)) return r;
  // r + w was a tie rounded to even, and v pushes the exact value past the midpoint.
  const next = nextToward(r, w);
  return Math.abs(w) * 2 === Math.abs(next - r) ? next : r;
}

/** `x @ y` for contiguous vectors (OpenBLAS ddot, SVE kernel at 128-bit vector length). */
export function blasDot(x: readonly number[], y: readonly number[]): number {
  const n = x.length;
  const n4 = n & ~3;
  const n2 = n & ~1;
  let s0 = 0,
    s1 = 0,
    z0 = 0,
    z1 = 0;
  for (let i = 0; i < n4; i += 4) {
    s0 = fma(x[i], y[i], s0);
    s1 = fma(x[i + 1], y[i + 1], s1);
    z0 = fma(x[i + 2], y[i + 2], z0);
    z1 = fma(x[i + 3], y[i + 3], z1);
  }
  const d1 = n4 > 0 ? s0 + s1 : 0;
  for (let i = n4; i < n2; i += 2) {
    z0 = fma(x[i], y[i], z0);
    z1 = fma(x[i + 1], y[i + 1], z1);
  }
  if (n2 !== n) z0 = fma(x[n - 1], y[n - 1], z0);
  return d1 + (z0 + z1);
}

/** `x @ y` when one operand is strided (a matrix column): one sequential FMA chain. */
export function dotStrided(x: readonly number[], y: readonly number[]): number {
  let acc = 0;
  for (let i = 0; i < x.length; i++) acc = fma(x[i], y[i], acc);
  return acc;
}

/** One row of `A @ x` (OpenBLAS dgemv_t SVE kernel: two FMA lanes, even and odd indices). */
function gemvRow(a: readonly number[], x: readonly number[], m: number): number {
  let t0 = 0,
    t1 = 0,
    i = 0;
  for (; i + 1 < m; i += 2) {
    t0 = fma(a[i], x[i], t0);
    t1 = fma(a[i + 1], x[i + 1], t1);
  }
  if (i < m) t0 = fma(a[i], x[i], t0);
  return 0 + (t0 + t1);
}

/** `A @ x` (the first `x.length` columns of each row of A). */
export function gemv(A: readonly (readonly number[])[], x: readonly number[]): Vector {
  return A.map((row) => gemvRow(row, x, x.length));
}

/** `A @ v` when v is a strided column (OpenBLAS's scalar gemv path: one FMA chain per row). */
export function gemvStrided(A: readonly (readonly number[])[], x: readonly number[]): Vector {
  return A.map((row) => dotStrided(row, x));
}

/** `x @ M` = Mᵀx (OpenBLAS dgemv_n SVE kernel: rows of M in blocks of three, FMA updates). */
export function vecMat(x: readonly number[], M: readonly (readonly number[])[]): Vector {
  const n = M.length;
  const m = M[0]?.length ?? 0;
  const y = zeros(m);
  const w = Math.floor(n / 3);
  const axpy = (row: number) => {
    const t = 1.0 * x[row];
    const a = M[row];
    for (let q = 0; q < m; q++) y[q] = fma(t, a[q], y[q]);
  };
  for (let j = 0; j < w; j++) {
    axpy(j);
    axpy(j + w);
    axpy(j + 2 * w);
  }
  for (let j = 3 * w; j < n; j++) axpy(j);
  return y;
}

/** `np.sum(v)` for a contiguous float64 vector: NumPy's pairwise summation. */
export function npSum(v: readonly number[], lo = 0, hi = v.length): number {
  return 0.0 + pairwise(v, lo, hi - lo);
}

function pairwise(a: readonly number[], lo: number, n: number): number {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += a[lo + i];
    return res;
  }
  if (n <= 128) {
    const r = a.slice(lo, lo + 8);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
}

/**
 * `np.hypot` as glibc ≥ 2.35 computes it on a machine with FMA (sysdeps/ieee754/dbl-64/e_hypot.c,
 * after Borges, "An improved algorithm for hypot(a, b)", 2019). It is not always correctly
 * rounded, so the ports use this exact kernel (matched NumPy on 3,000 random pairs).
 */
export function hypot(x: number, y: number): number {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return Math.hypot(x, y);
  const X = Math.abs(x),
    Y = Math.abs(y);
  const ax = X < Y ? Y : X;
  const ay = X < Y ? X : Y;
  const EPS54 = 2 ** -54;
  const SCALE = 2 ** -600;
  if (ax > 2 ** 511) {
    if (ay <= ax * EPS54) return ax + ay;
    return hypotKernel(ax * SCALE, ay * SCALE) / SCALE;
  }
  if (ay < 2 ** -511) {
    if (ax >= ay / EPS54) return ax + ay;
    return hypotKernel(ax / SCALE, ay / SCALE) * SCALE;
  }
  if (ay <= ax * EPS54) return ax + ay;
  return hypotKernel(ax, ay);
}

function hypotKernel(ax: number, ay: number): number {
  const t1 = ay + ay;
  const t2 = ax - ay;
  if (t1 >= ax) return Math.sqrt(fma(t1, ax, t2 * t2));
  return Math.sqrt(fma(ax, ax, ay * ay));
}

// ── resolve_system ─────────────────────────────────────────────────────────────────────

/** Fresh copies `(A, b)` of a LinearSystem or an `[A, b]` pair; throws like Python's ValueError. */
export function resolveSystem(problem: SystemLike): { A: Matrix; b: Vector } {
  let Araw: unknown, braw: unknown;
  if (Array.isArray(problem) && problem.length === 2) [Araw, braw] = problem;
  else if (problem && typeof problem === 'object' && 'A' in problem && 'b' in problem)
    ({ A: Araw, b: braw } = problem as { A: unknown; b: unknown });
  else throw new TypeError('problem must be a numopt LinearSystem or a pair (A, b)');
  if (!Array.isArray(Araw) || !Araw.every((r) => Array.isArray(r)))
    throw new Error('A must be a non-empty square matrix');
  const A = (Araw as number[][]).map((r) => r.map(Number));
  const b = (Array.isArray(braw) ? (braw as unknown[]).flat(Infinity) : [braw]).map(Number);
  const n = A.length;
  if (n === 0 || A.some((r) => r.length !== n))
    throw new Error(`A must be a non-empty square matrix, got shape (${n}, ${A[0]?.length ?? 0})`);
  if (b.length !== n) throw new Error(`b has ${b.length} entries, expected ${n}`);
  if (!(allFiniteM(A) && allFinite(b))) throw new Error('A and b must contain only finite values');
  return { A, b };
}

// ── Norms and tolerances ───────────────────────────────────────────────────────────────

/** (s, t) with ‖v‖₂ = s·t, s = max|v_i| and 1 ≤ t ≤ √(size); (0, 1) for a zero v. */
export function norm2Parts(v: readonly number[]): [number, number] {
  const s = maxAbs(v);
  if (s === 0 || !Number.isFinite(s)) return [s, 1];
  return [s, Math.sqrt(npSum(v.map((x) => (Math.abs(x) / s) ** 2)))];
}

/** ‖v‖₂ without overflow or underflow of the squares (LAPACK xNRM2 scaling). */
export function norm2(v: readonly number[]): number {
  const [s, t] = norm2Parts(v);
  return s * t;
}

/** Frobenius norm ‖A‖_F, scaled like `norm2`. */
export function normF(A: readonly (readonly number[])[]): number {
  return norm2(A.flat());
}

/** τ = n·ε·‖A‖_F: a pivot with |p| ≤ τ is treated as zero (finite even when ‖A‖_F overflows). */
export function pivotTolerance(A: readonly (readonly number[])[]): number {
  const n = A.length;
  const [s, t] = norm2Parts(A.flat());
  const aNorm = s * t;
  if (Number.isFinite(aNorm)) return n * EPS * aNorm;
  return n * EPS * s * t;
}

/** max |a_ij − a_ji|. */
export function asymmetry(A: readonly (readonly number[])[]): number {
  let m = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A.length; j++) m = Math.max(m, Math.abs(A[i][j] - A[j][i]));
  return m;
}

/** `True` when max |a_ij − a_ji| ≤ τ. */
export function isSymmetric(A: readonly (readonly number[])[]): boolean {
  return asymmetry(A) <= pivotTolerance(A);
}

/** ‖A‖₁ = max column sum of |a_ij| (`np.linalg.norm(A, 1)`). */
export function norm1(A: readonly (readonly number[])[]): number {
  const n = A.length;
  const m = A[0]?.length ?? 0;
  let best = 0;
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(A[i][j]);
    if (s > best || Number.isNaN(s)) best = s;
  }
  return best;
}

// ── Triangular solves (Golub & Van Loan 2013, Alg. 3.1.1 / 3.1.2, row versions) ───────

/**
 * `strided`: L is a transposed view in Python (`U.T`), so its rows are strided and NumPy's dot
 * takes the sequential FMA kernel.
 */
export function forwardSubstitution(
  L: readonly (readonly number[])[],
  y: readonly number[],
  unit = false,
  strided = false,
): Vector {
  const n = y.length;
  const x = zeros(n);
  const dotf = strided ? dotStrided : blasDot;
  for (let i = 0; i < n; i++) {
    const s = y[i] - dotf(L[i].slice(0, i), x.slice(0, i));
    x[i] = unit ? s : s / L[i][i];
  }
  return x;
}

export function backSubstitution(
  U: readonly (readonly number[])[],
  y: readonly number[],
  strided = false,
): Vector {
  const n = y.length;
  const x = zeros(n);
  const dotf = strided ? dotStrided : blasDot;
  for (let i = n - 1; i >= 0; i--)
    x[i] = (y[i] - dotf(U[i].slice(i + 1, n), x.slice(i + 1, n))) / U[i][i];
  return x;
}

// ── Condition estimate and backward error ─────────────────────────────────────────────

/** Index of the first max |v_i| (`np.argmax(np.abs(v))`; NaN wins like NumPy). */
export function argmaxAbs(v: readonly number[], from = 0): number {
  let best = from;
  let m = Math.abs(v[from]);
  if (Number.isNaN(m)) return from;
  for (let i = from + 1; i < v.length; i++) {
    const a = Math.abs(v[i]);
    if (Number.isNaN(a)) return i;
    if (a > m) {
      m = a;
      best = i;
    }
  }
  return best;
}

/**
 * Lower-bound estimate of ‖A⁻¹‖₁ from solves with A and Aᵀ only (Hager 1984, refined by Higham
 * 1988; LAPACK xLACN2): at most 5 power-like steps, then Higham's alternating test vector.
 */
export function estimateInvNorm1(
  solve: (v: Vector) => Vector,
  solveT: (v: Vector) => Vector,
  n: number,
): number {
  let x: Vector = new Array<number>(n).fill(1.0 / n);
  let est = 0.0;
  let jPrev = -1;
  for (let it = 0; it < 5; it++) {
    const y = solve(x);
    est = Math.max(est, npSum(y.map(Math.abs)));
    const xi = y.map((v) => (v >= 0.0 ? 1.0 : -1.0));
    const z = solveT(xi);
    const j = argmaxAbs(z);
    if ((it > 0 && Math.abs(z[j]) <= blasDot(z, x)) || j === jPrev) break;
    x = zeros(n);
    x[j] = 1.0;
    jPrev = j;
  }
  if (n > 1) {
    const xAlt = Array.from(
      { length: n },
      (_, i) => (i % 2 === 0 ? 1.0 : -1.0) * (1.0 + i / (n - 1)),
    );
    est = Math.max(est, (2.0 * npSum(solve(xAlt).map(Math.abs))) / (3.0 * n));
  }
  return est;
}

/** κ₁(A) ≈ ‖A‖₁ · est(‖A⁻¹‖₁); `Infinity` if the estimate is not finite. */
export function condEstimate(
  A: readonly (readonly number[])[],
  solve: (v: Vector) => Vector,
  solveT: (v: Vector) => Vector,
): number {
  const invEst = estimateInvNorm1(solve, solveT, A.length);
  const a1 = norm1(A);
  let est: number;
  if (Number.isFinite(a1)) est = a1 * invEst;
  else {
    const s = maxAbsM(A);
    est = norm1(A.map((r) => r.map((v) => v / s))) * (s * invEst);
  }
  return Number.isFinite(est) ? est : Infinity;
}

/** `np.logaddexp(a, b)`. */
function logaddexp(a: number, b: number): number {
  if (a === b) return a + Math.LN2;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
}

/** Normwise backward error η∞ = ‖r‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞), evaluated in the log domain. */
export function backwardError(
  A: readonly (readonly number[])[],
  b: readonly number[],
  x: readonly number[],
  r: readonly number[],
): number {
  if (!allFinite(r)) return Infinity;
  const rInf = maxAbs(r);
  if (rInf === 0.0) return 0.0;
  const s = maxAbsM(A);
  let logA = -Infinity;
  if (s > 0) {
    let rowMax = 0;
    for (const row of A) rowMax = Math.max(rowMax, npSum(row.map((v) => Math.abs(v) / s)));
    logA = Math.log(s) + Math.log(rowMax);
  }
  const logX = Math.log(maxAbs(x));
  const logB = Math.log(maxAbs(b));
  const logDen = logaddexp(logA + logX, logB);
  if (logDen === -Infinity) return Infinity;
  return Math.exp(Math.log(rInf) - logDen);
}

// ── Powers of two (np.frexp / np.ldexp) ───────────────────────────────────────────────

/** The exponent e of `np.frexp(s)`: s = m·2^e with 0.5 ≤ m < 1 (s > 0 finite). */
export function frexpExponent(s: number): number {
  let e = Math.floor(Math.log2(s)) + 1;
  // Correct the floating-point log2 near exact powers of two.
  while (2 ** (e - 1) > s) e--;
  while (2 ** e <= s) e++;
  return e;
}

/** A power of 2 within a factor 2 of max|v_i| (1 for v = 0): division by it is exact. */
export function pow2Scale(v: readonly number[]): number {
  const s = maxAbs(v);
  return s > 0.0 && Number.isFinite(s) ? 2 ** frexpExponent(s) : 1.0;
}

/** uᵀv / s² computed as (u/s)ᵀ(v/s). */
export function scaledDot(u: readonly number[], v: readonly number[], s: number): number {
  return blasDot(
    u.map((a) => a / s),
    v.map((a) => a / s),
  );
}

// ── Python's format(v, '.Ng') ──────────────────────────────────────────────────────────

/** `format(v, '.{p}g')` as Python prints it (`1.6e-16`, `0.471`, `inf`, `nan`). */
export function pyG(v: number, p = 3): string {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const prec = Math.max(1, p);
  const [mant, expStr] = v.toExponential(prec - 1).split('e');
  const X = Number(expStr);
  const strip = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
  if (X >= -4 && X < prec) return strip(v.toFixed(prec - 1 - X));
  const e = Math.abs(X);
  return `${strip(mant)}e${X < 0 ? '-' : '+'}${e < 10 ? `0${e}` : e}`;
}

// ── Eigenvalues ────────────────────────────────────────────────────────────────────────

export interface Complex {
  re: number;
  im: number;
}

/**
 * Eigenvalues of a general real matrix (`np.linalg.eigvals`): reduction to upper Hessenberg form
 * by stabilized elementary similarity transforms (EISPACK `elmhes`), then the Francis
 * double-shift QR iteration (EISPACK `hqr`; Press et al., Numerical Recipes, §11.5–11.6).
 * Returns NaN entries if the iteration does not converge (it does for every matrix in the lab).
 */
export function eigvals(Ain: readonly (readonly number[])[]): Complex[] {
  const n = Ain.length;
  const a = copyM(Ain);
  // elmhes
  for (let m = 1; m < n - 1; m++) {
    let x = 0.0;
    let i = m;
    for (let j = m; j < n; j++) {
      if (Math.abs(a[j][m - 1]) > Math.abs(x)) {
        x = a[j][m - 1];
        i = j;
      }
    }
    if (i !== m) {
      for (let j = m - 1; j < n; j++) [a[i][j], a[m][j]] = [a[m][j], a[i][j]];
      for (let j = 0; j < n; j++) [a[j][i], a[j][m]] = [a[j][m], a[j][i]];
    }
    if (x !== 0) {
      for (i = m + 1; i < n; i++) {
        let y = a[i][m - 1];
        if (y !== 0.0) {
          y /= x;
          a[i][m - 1] = y;
          for (let j = m; j < n; j++) a[i][j] -= y * a[m][j];
          for (let j = 0; j < n; j++) a[j][m] += y * a[j][i];
        }
      }
    }
  }
  for (let i = 2; i < n; i++) for (let j = 0; j < i - 1; j++) a[i][j] = 0;

  // hqr
  const wr = zeros(n);
  const wi = zeros(n);
  let anorm = 0.0;
  for (let i = 0; i < n; i++)
    for (let j = Math.max(i - 1, 0); j < n; j++) anorm += Math.abs(a[i][j]);
  let nn = n - 1;
  let t = 0.0;
  const sign = (u: number, v: number) => (v >= 0 ? Math.abs(u) : -Math.abs(u));
  // Declared without initial values: hqr assigns each before use.
  let p = 0,
    q = 0,
    r = 0,
    s: number,
    w: number,
    x: number,
    y: number,
    z: number;
  while (nn >= 0) {
    let its = 0;
    let l: number;
    do {
      for (l = nn; l >= 1; l--) {
        s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]);
        if (s === 0.0) s = anorm;
        if (Math.abs(a[l][l - 1]) + s === s) {
          a[l][l - 1] = 0.0;
          break;
        }
      }
      x = a[nn][nn];
      if (l === nn) {
        wr[nn] = x + t;
        wi[nn--] = 0.0;
      } else {
        y = a[nn - 1][nn - 1];
        w = a[nn][nn - 1] * a[nn - 1][nn];
        if (l === nn - 1) {
          p = 0.5 * (y - x);
          q = p * p + w;
          z = Math.sqrt(Math.abs(q));
          x += t;
          if (q >= 0.0) {
            z = p + sign(z, p);
            wr[nn - 1] = wr[nn] = x + z;
            if (z) wr[nn] = x - w / z;
            wi[nn - 1] = wi[nn] = 0.0;
          } else {
            wr[nn - 1] = wr[nn] = x + p;
            wi[nn - 1] = -(wi[nn] = z);
          }
          nn -= 2;
        } else {
          if (its === 60) {
            for (let i = 0; i <= nn; i++) wr[i] = wi[i] = NaN;
            return wr.map((re, i) => ({ re, im: wi[i] }));
          }
          if (its === 10 || its === 20 || its === 40) {
            t += x;
            for (let i = 0; i <= nn; i++) a[i][i] -= x;
            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]);
            y = x = 0.75 * s;
            w = -0.4375 * s * s;
          }
          ++its;
          let m: number;
          for (m = nn - 2; m >= l; m--) {
            z = a[m][m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
            q = a[m + 1][m + 1] - z - r - s;
            r = a[m + 2][m + 1];
            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m === l) break;
            const u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r));
            const v =
              Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]));
            if (u + v === v) break;
          }
          for (let i = m + 2; i <= nn; i++) {
            a[i][i - 2] = 0.0;
            if (i !== m + 2) a[i][i - 3] = 0.0;
          }
          for (let k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = a[k][k - 1];
              q = a[k + 1][k - 1];
              r = 0.0;
              if (k !== nn - 1) r = a[k + 2][k - 1];
              if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0.0) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            if ((s = sign(Math.sqrt(p * p + q * q + r * r), p)) !== 0) {
              if (k === m) {
                if (l !== m) a[k][k - 1] = -a[k][k - 1];
              } else a[k][k - 1] = -s * x;
              p += s;
              x = p / s;
              y = q / s;
              z = r / s;
              q /= p;
              r /= p;
              for (let j = k; j <= nn; j++) {
                p = a[k][j] + q * a[k + 1][j];
                if (k !== nn - 1) {
                  p += r * a[k + 2][j];
                  a[k + 2][j] -= p * z;
                }
                a[k + 1][j] -= p * y;
                a[k][j] -= p * x;
              }
              const mmin = nn < k + 3 ? nn : k + 3;
              for (let i = l; i <= mmin; i++) {
                p = x * a[i][k] + y * a[i][k + 1];
                if (k !== nn - 1) {
                  p += z * a[i][k + 2];
                  a[i][k + 2] -= p * r;
                }
                a[i][k + 1] -= p * q;
                a[i][k] -= p;
              }
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return wr.map((re, i) => ({ re, im: wi[i] }));
}

/** ρ(G) = max |λ_i(G)| (`np.max(np.abs(np.linalg.eigvals(G)))`). */
export function spectralRadius(G: readonly (readonly number[])[]): number {
  let m = 0;
  for (const { re, im } of eigvals(G)) {
    const a = Math.hypot(re, im);
    if (Number.isNaN(a)) return NaN;
    m = Math.max(m, a);
  }
  return m;
}

/**
 * Eigenvalues of a symmetric matrix, ascending (`np.linalg.eigvalsh`), by the cyclic Jacobi
 * method (Golub & Van Loan 2013, Alg. 8.5.3), which is accurate to O(ε‖A‖) for every eigenvalue.
 */
export function eigvalsh(Ain: readonly (readonly number[])[]): number[] {
  const n = Ain.length;
  const a = copyM(Ain);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off === 0) break;
    let diag = 0;
    for (let i = 0; i < n; i++) diag += a[i][i] * a[i][i];
    if (Math.sqrt(off) <= 1e-17 * Math.sqrt(diag + off)) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (apq === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p],
            akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k],
            aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  return a.map((r, i) => r[i]).sort((u, v) => u - v);
}

/** Solve A X = B column by column with LU and partial pivoting (`np.linalg.solve(A, B)`). */
export function solveMatrix(
  A: readonly (readonly number[])[],
  B: readonly (readonly number[])[],
): Matrix {
  const n = A.length;
  const lu = copyM(A);
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let k = 0; k < n; k++) {
    const p = k + argmaxAbs(lu.map((r) => r[k]).slice(k));
    if (p !== k) {
      [lu[k], lu[p]] = [lu[p], lu[k]];
      [perm[k], perm[p]] = [perm[p], perm[k]];
    }
    for (let i = k + 1; i < n; i++) {
      const m = lu[i][k] / lu[k][k];
      lu[i][k] = m;
      for (let j = k + 1; j < n; j++) lu[i][j] -= m * lu[k][j];
    }
  }
  const m = B[0]?.length ?? 0;
  const X = zerosM(n, m);
  for (let c = 0; c < m; c++) {
    const y = perm.map((p) => B[p][c]);
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) y[i] -= lu[i][j] * y[j];
    for (let i = n - 1; i >= 0; i--) {
      for (let j = i + 1; j < n; j++) y[i] -= lu[i][j] * y[j];
      y[i] /= lu[i][i];
    }
    for (let i = 0; i < n; i++) X[i][c] = y[i];
  }
  return X;
}
