/**
 * Step geometry for the systems lab, read from `Step.info` (src/methods/roots/systems.ts).
 *
 * Step j (j ≥ 1) says how 𝐱_j was produced from 𝐱_{j−1}: the model matrix M (J(𝐱_{j−1}) for
 * Newton, B_{j−1} for Broyden) linearizes each component,
 *
 *     Lᵢ(𝐱) = Fᵢ(𝐱_{j−1}) + Mᵢ·(𝐱 − 𝐱_{j−1}),
 *
 * and the full step 𝐩 lands where both linear models vanish: the zero lines ℓ₁ = {L₁ = 0} and
 * ℓ₂ = {L₂ = 0} cross at 𝐱_{j−1} + 𝐩. Pure functions, unit-tested in ./geometry.test.ts.
 */
import type { Matrix, Step, Vector } from '../../core/types';

export type Pt = [number, number];

/** The zero line of Lᵢ, as a long segment centered on the foot of the perpendicular from 𝐱. */
export function linearizationLine(
  x: readonly number[],
  Fi: number,
  row: readonly number[],
  halfLength: number,
): { from: Pt; to: Pt; foot: Pt } | null {
  const g2 = row[0] * row[0] + row[1] * row[1];
  if (!(g2 > 0) || !Number.isFinite(g2) || !Number.isFinite(Fi)) return null;
  const g = Math.sqrt(g2);
  // Foot of the perpendicular from 𝐱 onto {Lᵢ = 0}: 𝐱 − Fᵢ Mᵢ / ‖Mᵢ‖².
  const foot: Pt = [x[0] - (Fi * row[0]) / g2, x[1] - (Fi * row[1]) / g2];
  const d: Pt = [-row[1] / g, row[0] / g];
  return {
    from: [foot[0] - halfLength * d[0], foot[1] - halfLength * d[1]],
    to: [foot[0] + halfLength * d[0], foot[1] + halfLength * d[1]],
    foot,
  };
}

/** Where the two zero lines cross (the model's root), or null when M is singular. */
export function modelRoot(x: readonly number[], F: readonly number[], M: Matrix): Pt | null {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  if (!Number.isFinite(det) || det === 0) return null;
  // p = −M⁻¹F (Cramer's rule, for drawing only).
  const p0 = -(M[1][1] * F[0] - M[0][1] * F[1]) / det;
  const p1 = -(-M[1][0] * F[0] + M[0][0] * F[1]) / det;
  return [x[0] + p0, x[1] + p1];
}

/** The geometry of step j of a systems trace (null for j < 1 or a malformed step). */
export interface StepGeometry {
  j: number;
  /** 𝐱_{j−1}: where the model is built. */
  from: Pt;
  /** 𝐱_j: where the method went. */
  to: Pt;
  /** F(𝐱_{j−1}). */
  F: Vector;
  /** The model matrix (J or B). */
  M: Matrix;
  /** The full step 𝐩 (Newton: `newton_step`; Broyden: `step`). */
  p: Vector;
  /** 𝐱_{j−1} + 𝐩: the crossing of ℓ₁ and ℓ₂. */
  target: Pt;
  /** α taken (1 for Broyden). */
  alpha: number;
  /** Backtracking trials [α, φ] (damped Newton). */
  trials: [number, number][];
  /** Broyden only: s = 𝐱_j − 𝐱_{j−1}, y = F(𝐱_j) − F(𝐱_{j−1}). */
  secant: { s: Vector; y: Vector } | null;
}

export function stepGeometry(trace: readonly Step[], j: number): StepGeometry | null {
  if (j < 1 || j >= trace.length) return null;
  const prev = trace[j - 1];
  const cur = trace[j];
  const F = prev.info.residual as Vector | undefined;
  const M = cur.info.jacobian as Matrix | undefined;
  if (!F || !M || !Array.isArray(prev.x) || !Array.isArray(cur.x)) return null;
  const from = prev.x as Pt;
  const to = cur.x as Pt;
  const p = (cur.info.newton_step as Vector | undefined) ?? (cur.info.step as Vector);
  return {
    j,
    from,
    to,
    F,
    M,
    p,
    target: [from[0] + p[0], from[1] + p[1]],
    alpha: typeof cur.info.alpha === 'number' ? cur.info.alpha : 1,
    trials: (cur.info.trials as [number, number][] | undefined) ?? [],
    secant: (cur.info.secant as { s: Vector; y: Vector } | undefined) ?? null,
  };
}

/** Which step's geometry explains the picture at local time `lt` of a trace with n steps. */
export function geometryIndex(lt: number, n: number): number {
  if (n < 2) return 0;
  // While the head moves from 𝐱_k to 𝐱_{k+1} (and when paused at 𝐱_k), show step k + 1: the
  // model built at 𝐱_k and where it points. At the end, keep the last step.
  const k = Math.floor(lt + 1e-9);
  return Math.min(k + 1, n - 1);
}

/** Frobenius norm. */
export function frob(A: Matrix): number {
  let s = 0;
  for (const r of A) for (const v of r) s += v * v;
  return Math.sqrt(s);
}

/** ‖B − J‖_F / ‖J‖_F: how far a Broyden matrix is from the true Jacobian. */
export function modelError(B: Matrix, J: Matrix): number {
  const D = B.map((r, i) => r.map((v, j) => v - J[i][j]));
  const nj = frob(J);
  return nj > 0 ? frob(D) / nj : Infinity;
}

/** Index of the known root within `tol` (relative) of x, else −1. */
export function rootIndex(x: readonly number[], roots: readonly (readonly number[])[], tol = 1e-6) {
  for (let i = 0; i < roots.length; i++) {
    const r = roots[i];
    const d = Math.hypot(x[0] - r[0], x[1] - r[1]);
    if (d <= tol * (1 + Math.hypot(r[0], r[1]))) return i;
  }
  return -1;
}

/** The known root nearest to x (for the error measure ‖𝐱ₖ − 𝐱⋆‖). */
export function nearestRoot(
  x: readonly number[],
  roots: readonly (readonly number[])[],
): readonly number[] | null {
  let best: readonly number[] | null = null,
    bd = Infinity;
  for (const r of roots) {
    const d = Math.hypot(x[0] - r[0], x[1] - r[1]);
    if (d < bd) {
      bd = d;
      best = r;
    }
  }
  return best;
}
