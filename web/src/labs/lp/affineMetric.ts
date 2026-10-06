/**
 * The metric of an affine-scaling step, seen in the plane of the original variables.
 *
 * `affineScaling` iterates on the scaled standard form ẑ = (x̂, ŝ, t) with one artificial column
 * r₀ = b̂ − Â𝟙 (see methods/lp/interior_point.ts). Its step is steepest descent in the norm
 * ‖Z⁻¹d‖ over the directions with A_aug d = 0, so the true Dikin ellipsoid of the step is
 *
 *     E = { d : ‖Z⁻¹d‖ ≤ 1, A_aug d = 0 },
 *
 * and its shadow on x = β x̂ is the ellipse { dx : dxᵀ Q dx ≤ 1 } with
 *
 *     Q⁻¹ = β² [ Z (I − Bᵀ(BBᵀ)⁻¹B) Z ]ₓₓ,   B = A_aug Z.
 *
 * Once t = 0 this is the polygon's Dikin ellipse Σ gᵢgᵢᵀ/sᵢ² (geometry.dikinMatrix); while t > 0
 * the artificial can move too, so the ellipse is larger and need not fit in the polygon.
 * Pure; tested in geometry.test.ts.
 */
import type { LinearProgram } from '../../core/types';
import { scaledForm } from '../../methods/lp/interior_point';

/** Solve M X = R (M square) by Gaussian elimination with partial pivoting; null if singular. */
function solve(M: number[][], R: number[][]): number[][] | null {
  const m = M.length;
  const A = M.map((row, i) => [...row, ...R[i]]);
  const w = R[0]?.length ?? 0;
  const scale = Math.max(1e-300, ...M.flat().map(Math.abs));
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) <= 1e-13 * scale) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      if (f === 0) continue;
      for (let j = c; j < m + w; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map((row, i) => row.slice(m).map((v) => v / A[i][i]));
}

const identity = (n: number) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

/**
 * Q of the projected step ellipse at the iterate with original variables `x` and artificial
 * `t` (the method's `info.artificial`). Null when the iterate is not strictly interior or the
 * ellipse is degenerate in the plane (e.g. an equality row pins x).
 */
export function affineScalingMetric(
  lp: LinearProgram,
  x: readonly number[],
  t: number,
): number[][] | null {
  const sf = scaledForm(lp);
  if (!sf.consistent) return null;
  const { A: A0, b, n, beta } = sf;
  const m = A0.length;
  const N0 = A0.length ? A0[0].length : n;
  const r0 = b.map((v, i) => v - A0[i].reduce((s, a) => s + a, 0));
  const A = A0.map((row, i) => [...row, r0[i]]);
  // ẑ from (x, t): x̂ = x/β, every slack read off the one row it appears in, then t.
  const z = new Array<number>(N0 + 1).fill(NaN);
  for (let j = 0; j < n; j++) z[j] = x[j] / beta;
  z[N0] = t;
  for (let j = n; j < N0; j++) {
    const r = A.findIndex((row) => Math.abs(row[j]) > 0);
    if (r < 0) return null;
    let rest = b[r] - A[r][N0] * t;
    for (let q = 0; q < n; q++) rest -= A[r][q] * z[q];
    z[j] = rest / A[r][j];
  }
  // x̂ and ŝ strictly positive; t may be 0 (then the artificial is pinned and drops out).
  if (!z.every((v, j) => Number.isFinite(v) && (j === N0 ? v >= 0 : v > 0))) return null;

  // Σ = β² [Z (I − Bᵀ(BBᵀ)⁻¹B) Z]ₓₓ.
  const B = A.map((row) => row.map((v, j) => v * z[j]));
  let Sigma: number[][];
  if (m === 0) Sigma = identity(n).map((row, a) => row.map((v, c) => v * z[a] * z[c]));
  else {
    const M = B.map((ri) => B.map((rj) => ri.reduce((s, v, j) => s + v * rj[j], 0)));
    const Y = solve(
      M,
      B.map((row) => row.slice(0, n)),
    );
    if (!Y) return null;
    Sigma = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, c) => {
        let proj = 0;
        for (let r = 0; r < m; r++) proj += B[r][a] * Y[r][c];
        return z[a] * z[c] * ((a === c ? 1 : 0) - proj);
      }),
    );
  }
  Sigma = Sigma.map((row) => row.map((v) => beta * beta * v));
  // Symmetrize, then invert; a (near-)singular Σ means the ellipse is flat in the plane.
  Sigma = Sigma.map((row, a) => row.map((v, c) => (v + Sigma[c][a]) / 2));
  const trace = Sigma.reduce((s, row, a) => s + row[a], 0);
  if (!(trace > 0)) return null;
  const Q = solve(Sigma, identity(n));
  if (!Q) return null;
  // Reject a numerically flat ellipse (its inverse blows up along the pinned direction).
  const qTrace = Q.reduce((s, row, a) => s + row[a], 0);
  if (!(qTrace > 0) || qTrace * trace > 1e12) return null;
  return Q;
}
