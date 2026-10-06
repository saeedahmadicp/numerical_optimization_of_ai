/**
 * Small dense linear-algebra helpers for ports and visualizations (n ≲ 50).
 * Plain `number[]` / `number[][]`; inputs are never mutated.
 */
import type { Matrix, Vector } from './types';

export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** `alpha * x + y` (a new vector). */
export function axpy(alpha: number, x: readonly number[], y: readonly number[]): Vector {
  const out = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) out[i] = alpha * x[i] + y[i];
  return out;
}

export function add(a: readonly number[], b: readonly number[]): Vector {
  return a.map((v, i) => v + b[i]);
}

export function sub(a: readonly number[], b: readonly number[]): Vector {
  return a.map((v, i) => v - b[i]);
}

export function scale(alpha: number, x: readonly number[]): Vector {
  return x.map((v) => alpha * v);
}

/** Euclidean norm, computed like `numpy.linalg.norm` (plain sqrt of the sum of squares). */
export function norm(x: readonly number[]): number {
  return Math.sqrt(dot(x, x));
}

export function normInf(x: readonly number[]): number {
  let m = 0;
  for (const v of x) m = Math.max(m, Math.abs(v));
  return m;
}

export function matvec(A: Matrix, x: readonly number[]): Vector {
  return A.map((row) => dot(row, x));
}

export function matmul(A: Matrix, B: Matrix): Matrix {
  const n = A.length,
    m = B[0]?.length ?? 0,
    p = B.length;
  const C: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i++)
    for (let k = 0; k < p; k++) {
      const a = A[i][k];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[k][j];
    }
  return C;
}

export function transpose(A: Matrix): Matrix {
  if (A.length === 0) return [];
  return A[0].map((_, j) => A.map((row) => row[j]));
}

export function identity(n: number): Matrix {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function outer(a: readonly number[], b: readonly number[]): Matrix {
  return a.map((ai) => b.map((bj) => ai * bj));
}

export interface LU {
  /** Combined L (unit lower, below diagonal) and U (upper). */
  lu: Matrix;
  /** Row permutation: row i of PA is row perm[i] of A. */
  perm: number[];
  singular: boolean;
}

/** LU factorization with partial pivoting (Golub & Van Loan, Alg. 3.4.1). */
export function luFactor(A: Matrix, tol = 0): LU {
  const n = A.length;
  const lu = A.map((r) => r.slice());
  const perm = Array.from({ length: n }, (_, i) => i);
  let singular = false;
  for (let k = 0; k < n; k++) {
    let p = k;
    let max = Math.abs(lu[k][k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i][k]);
      if (v > max) {
        max = v;
        p = i;
      }
    }
    if (max <= tol) {
      singular = true;
      continue;
    }
    if (p !== k) {
      [lu[k], lu[p]] = [lu[p], lu[k]];
      [perm[k], perm[p]] = [perm[p], perm[k]];
    }
    for (let i = k + 1; i < n; i++) {
      const m = lu[i][k] / lu[k][k];
      lu[i][k] = m;
      if (m === 0) continue;
      for (let j = k + 1; j < n; j++) lu[i][j] -= m * lu[k][j];
    }
  }
  return { lu, perm, singular };
}

export function luSolve({ lu, perm }: LU, b: readonly number[]): Vector {
  const n = lu.length;
  const y = perm.map((p) => b[p]);
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) y[i] -= lu[i][j] * y[j];
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) y[i] -= lu[i][j] * y[j];
    y[i] /= lu[i][i];
  }
  return y;
}

/** Solve `A x = b`; returns `null` when A is (numerically) singular. */
export function solve(A: Matrix, b: readonly number[]): Vector | null {
  const f = luFactor(A);
  if (f.singular) return null;
  const x = luSolve(f, b);
  return x.every(Number.isFinite) ? x : null;
}

/** Lower-triangular L with A = L Lᵀ, or `null` when A is not (numerically) positive definite. */
export function cholesky(A: Matrix): Matrix | null {
  const n = A.length;
  const L: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let d = A[j][j];
    for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k];
    if (!(d > 0)) return null;
    L[j][j] = Math.sqrt(d);
    for (let i = j + 1; i < n; i++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = s / L[j][j];
    }
  }
  return L;
}

/** Solve `A x = b` with a Cholesky factor `L` of A. */
export function choleskySolve(L: Matrix, b: readonly number[]): Vector {
  const n = L.length;
  const y = b.slice();
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < i; k++) y[i] -= L[i][k] * y[k];
    y[i] /= L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i + 1; k < n; k++) y[i] -= L[k][i] * y[k];
    y[i] /= L[i][i];
  }
  return y;
}

/**
 * Eigen-decomposition of a symmetric 2×2 matrix [[a, b], [b, c]].
 * Returns eigenvalues ascending and unit eigenvectors (columns), useful for drawing level-set
 * ellipses and conditioning readouts.
 */
export function eigSym2(A: Matrix): { values: [number, number]; vectors: [Vector, Vector] } {
  const a = A[0][0],
    b = A[0][1],
    c = A[1][1];
  const mean = (a + c) / 2;
  const r = Math.hypot((a - c) / 2, b);
  const l1 = mean - r,
    l2 = mean + r;
  let v1: Vector;
  if (Math.abs(b) > 1e-300) v1 = [l1 - c, b];
  else v1 = a <= c ? [1, 0] : [0, 1];
  const n1 = Math.hypot(v1[0], v1[1]);
  v1 = [v1[0] / n1, v1[1] / n1];
  const v2: Vector = [-v1[1], v1[0]];
  return { values: [l1, l2], vectors: [v1, v2] };
}
