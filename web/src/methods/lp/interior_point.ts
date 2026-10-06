/**
 * Interior-point methods for linear programs — TS port of `numopt.lp.interior_point`
 * (src/numopt/lp/interior_point.py). Both methods work on the scaled equality standard form
 *
 *   min ĉᵀẑ  s.t.  Â ẑ = b̂,  ẑ ≥ 0,   Â = R A D,  b̂ = R b / β,  ĉ = c̃ / γ
 *
 * (row equilibration R, slack scaling D, β = ‖R b‖∞, γ = ‖c̃‖∞), with linearly dependent rows
 * removed. Infeasibility and unboundedness are detected by approximate Farkas certificates
 * (ε = 1e-8) and, when undecided, a second run on the zero objective (see the Python module
 * docstring for the reasoning behind every NOTE).
 *
 * Info keys (snake_case, as in Python): x, mu, primal_residual, dual_residual, gap,
 * alpha_primal, alpha_dual, alpha_aff_primal, alpha_aff_dual, sigma, x_affine (primal_dual_ipm);
 * alpha, direction, artificial (affine_scaling).
 */
import { registerMethod, param } from '../../core/registry';
import type { LinearProgram, Matrix, Result, Step } from '../../core/types';
import { EPS, TINY, checkLp, dotv, lpArrays, ones, pyG, zeros, zerosM } from './simplex';

/** Tolerance ε of the approximate Farkas certificates (scaled data). */
const CERT_TOL = 1e-8;
/** primal_dual_ipm: consecutive long steps without progress in the primal residual. */
const LOST_STEPS = 5;
/** A ray whose classification is still open is followed while feasibility falls by this factor. */
const PROGRESS = 0.9;
/** affine_scaling: t·‖r̂₀‖ ≤ this multiple of (1 + ‖b̂‖) counts as feasible. */
const FEAS_RTOL = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────────────────
// Dense kernels (LAPACK semantics where it matters for decisions)
// ─────────────────────────────────────────────────────────────────────────────────────────

const matvec = (A: Matrix, x: readonly number[]) => A.map((r) => dotv(r, x));
/** Aᵀ y */
function rmatvec(A: Matrix, y: readonly number[], N: number): number[] {
  const out = zeros(N);
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let i = 0; i < A.length; i++) s += A[i][j] * y[i];
    out[j] = s;
  }
  return out;
}
const sumv = (v: readonly number[]) => v.reduce((s, x) => s + x, 0);
const maxv = (v: readonly number[], initial = -Infinity) =>
  v.reduce((m, x) => (x > m || Number.isNaN(x) ? x : m), initial);
const minv = (v: readonly number[]) =>
  v.reduce((m, x) => (x < m || Number.isNaN(x) ? x : m), Infinity);
const norm2 = (v: readonly number[]) => Math.sqrt(dotv(v, v));
const allFinite = (v: readonly number[]) => v.every(Number.isFinite);

/** ‖v‖₂ without overflow of the squares: m·‖v/m‖₂ with m = ‖v‖∞. */
function safeNorm(v: readonly number[]): number {
  const m = v.reduce((a, x) => (Math.abs(x) > a || Number.isNaN(x) ? Math.abs(x) : a), 0);
  if (m === 0.0 || !Number.isFinite(m)) return m;
  return m * norm2(v.map((x) => x / m));
}

/** Lower Cholesky factor (reads the lower triangle, like LAPACK dpotrf); null if not SPD. */
function choleskyLower(M: Matrix): number[][] | null {
  const n = M.length;
  const L = zerosM(n, n);
  for (let j = 0; j < n; j++) {
    let s = M[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 0)) return null;
    const d = Math.sqrt(s);
    L[j][j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = M[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / d;
    }
  }
  return L;
}

/** Cholesky factor of the SPD normal matrix; one retry with a tiny diagonal shift. */
function cholesky(M: Matrix): number[][] | null {
  const L = choleskyLower(M);
  if (L) return L;
  // NOTE: near a degenerate optimum A D² Aᵀ can lose definiteness to rounding (Wright 1997, §11.1).
  const diag = M.map((r, i) => r[i]);
  const shift = M.length ? 1e-14 * Math.max(1.0, maxv(diag)) : 0.0;
  return choleskyLower(M.map((r, i) => r.map((v, j) => (i === j ? v + shift : v))));
}

/** Solve L Lᵀ y = r by forward and back substitution. */
function cholSolve(L: number[][], r: readonly number[]): number[] {
  const m = L.length;
  const y = [...r];
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let k = 0; k < i; k++) s += L[i][k] * y[k];
    y[i] = (y[i] - s) / L[i][i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = 0;
    for (let k = i + 1; k < m; k++) s += L[k][i] * y[k];
    y[i] = (y[i] - s) / L[i][i];
  }
  return y;
}

/** Reduced Householder QR of a p × q matrix (p ≥ q): Q (p × q) with orthonormal columns, R (q × q). */
export function householderQR(M: Matrix): { Q: number[][]; R: number[][] } {
  const p = M.length,
    q = p ? M[0].length : 0;
  const R = M.map((r) => [...r]);
  const vs: number[][] = [];
  const taus: number[] = [];
  for (let k = 0; k < q; k++) {
    const alpha = R[k][k];
    let xn = 0;
    for (let i = k + 1; i < p; i++) xn += R[i][k] * R[i][k];
    xn = Math.sqrt(xn);
    const v = zeros(p);
    if (xn === 0) {
      vs.push(v);
      taus.push(0);
      continue;
    }
    const beta = -(alpha >= 0 ? 1 : -1) * Math.hypot(alpha, xn);
    const tau = (beta - alpha) / beta;
    v[k] = 1;
    for (let i = k + 1; i < p; i++) v[i] = R[i][k] / (alpha - beta);
    for (let j = k; j < q; j++) {
      let s = 0;
      for (let i = k; i < p; i++) s += v[i] * R[i][j];
      s *= tau;
      for (let i = k; i < p; i++) R[i][j] -= s * v[i];
    }
    vs.push(v);
    taus.push(tau);
  }
  const Q = zerosM(p, q);
  for (let j = 0; j < q; j++) Q[j][j] = 1;
  for (let k = q - 1; k >= 0; k--) {
    const v = vs[k],
      tau = taus[k];
    if (tau === 0) continue;
    for (let j = 0; j < q; j++) {
      let s = 0;
      for (let i = k; i < p; i++) s += v[i] * Q[i][j];
      s *= tau;
      for (let i = k; i < p; i++) Q[i][j] -= s * v[i];
    }
  }
  return { Q, R: R.slice(0, q).map((r) => r.slice(0, q)) };
}

/** Singular values (one-sided Jacobi), for `matrix_rank`. */
export function singularValues(M: Matrix): number[] {
  const rows = M.length,
    cols = rows ? M[0].length : 0;
  if (rows === 0 || cols === 0) return [];
  // Columns of X are the shorter dimension.
  const X =
    rows >= cols
      ? M.map((r) => [...r])
      : Array.from({ length: cols }, (_, j) => M.map((r) => r[j]));
  const p = X.length,
    q = X[0].length;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let a = 0; a < q - 1; a++)
      for (let b = a + 1; b < q; b++) {
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let i = 0; i < p; i++) {
          alpha += X[i][a] * X[i][a];
          beta += X[i][b] * X[i][b];
          gamma += X[i][a] * X[i][b];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta));
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t),
          s = c * t;
        for (let i = 0; i < p; i++) {
          const xa = X[i][a],
            xb = X[i][b];
          X[i][a] = c * xa - s * xb;
          X[i][b] = s * xa + c * xb;
        }
      }
    if (off <= EPS) break;
  }
  return Array.from({ length: q }, (_, j) => norm2(X.map((r) => r[j]))).sort((a, b) => b - a);
}

/** `np.linalg.matrix_rank(M, tol)`: number of singular values > tol. */
export function matrixRank(M: Matrix, tol: number): number {
  return singularValues(M).filter((s) => s > tol).length;
}

/** Least-squares solution of G w ≈ a for G (p × q) with full column rank (QR). */
function lstsq(G: Matrix, a: readonly number[]): number[] {
  const q = G.length ? G[0].length : 0;
  if (q === 0) return [];
  const { Q, R } = householderQR(G);
  const qa = Array.from({ length: q }, (_, j) => {
    let s = 0;
    for (let i = 0; i < G.length; i++) s += Q[i][j] * a[i];
    return s;
  });
  const w = zeros(q);
  for (let i = q - 1; i >= 0; i--) {
    let s = qa[i];
    for (let k = i + 1; k < q; k++) s -= R[i][k] * w[k];
    w[i] = s / R[i][i];
  }
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Scaled standard form
// ─────────────────────────────────────────────────────────────────────────────────────────

/** `A, b, c̃, n, sign` of the equality standard form (unscaled, all rows). */
export function equalityForm(lp: LinearProgram) {
  const { c, aUb, bUb, aEq, bEq } = lpArrays(lp);
  const n = c.length,
    mUb = bUb.length,
    mEq = bEq.length;
  const A = zerosM(mUb + mEq, n + mUb);
  for (let i = 0; i < mUb; i++) {
    for (let j = 0; j < n; j++) A[i][j] = aUb[i][j];
    A[i][n + i] = 1.0;
  }
  for (let i = 0; i < mEq; i++) for (let j = 0; j < n; j++) A[mUb + i][j] = aEq[i][j];
  const b = [...bUb, ...bEq];
  const sign = lp.sense === 'min' ? 1.0 : -1.0;
  return { A, b, c: [...c.map((v) => sign * v), ...zeros(mUb)], n, sign };
}

/** Greedy row selection: keep a row when it raises the rank of the kept rows. */
export function independentRows(
  A: Matrix,
  b: readonly number[],
): { keep: number[]; consistent: boolean } {
  const keep: number[] = [];
  let consistent = true;
  const N = A.length ? A[0].length : 0;
  const scale =
    1.0 +
    (A.length && N
      ? Math.max(...A.map((r) => r.reduce((m, v) => Math.max(m, Math.abs(v)), 0)))
      : 0.0);
  const tol = Math.max(A.length, N) * EPS * scale * 1e3;
  for (let i = 0; i < A.length; i++) {
    const trial = [...keep, i];
    if (
      matrixRank(
        trial.map((r) => A[r]),
        tol,
      ) === trial.length
    ) {
      keep.push(i);
      continue;
    }
    // w = lstsq(A[keep]ᵀ, A[i])
    const G = Array.from({ length: N }, (_, j) => keep.map((r) => A[r][j]));
    const w = lstsq(G, A[i]);
    const wb = keep.reduce((s, r, q) => s + w[q] * b[r], 0);
    const bmax = keep.reduce((m, r) => Math.max(m, Math.abs(b[r])), 0.0);
    if (Math.abs(b[i] - wb) > 1e-9 * (1.0 + Math.abs(b[i]) + bmax)) consistent = false;
  }
  return { keep, consistent };
}

/** The scaled equality form min ĉᵀẑ s.t. Â ẑ = b̂, ẑ ≥ 0. */
export interface ScaledForm {
  A: Matrix;
  b: number[];
  c: number[];
  n: number;
  sign: number;
  col: number[];
  row: number[];
  beta: number;
  gamma: number;
  AFull: Matrix;
  bFull: number[];
  consistent: boolean;
  /** Original z = β D ẑ. */
  z: (zHat: readonly number[]) => number[];
  /** Original variables x = β ẑ[:n]. */
  x: (zHat: readonly number[]) => number[];
}

export function scaledForm(lp: LinearProgram): ScaledForm {
  const { A: AFull, b: bFull, c: cFull, n, sign } = equalityForm(lp);
  const m = AFull.length,
    N = cFull.length;
  const mUb = N - n;
  const rho = AFull.map((r, i) => {
    const v = r.slice(0, n).reduce((a, x) => Math.max(a, Math.abs(x)), 0);
    // A row with no entry on x is scaled by |bᵢ| so that it does not set β by itself.
    return v > 0.0 ? v : Math.abs(bFull[i]) >= TINY ? Math.abs(bFull[i]) : 1.0;
  });
  const col = ones(N);
  for (let i = 0; i < mUb; i++) col[n + i] = rho[i];
  const AHat = AFull.map((r, i) => r.map((v, j) => (v * col[j]) / rho[i]));
  const bRow = bFull.map((v, i) => v / rho[i]);
  let beta = m ? bRow.reduce((a, v) => Math.max(a, Math.abs(v)), 0) : 0.0;
  beta = beta > 0.0 ? beta : 1.0;
  let gamma = cFull.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  gamma = gamma > 0.0 ? gamma : 1.0;
  const bHat = bRow.map((v) => v / beta);
  const { keep, consistent } = independentRows(AHat, bHat);
  return {
    A: keep.map((i) => AHat[i]),
    b: keep.map((i) => bHat[i]),
    c: cFull.map((v) => v / gamma),
    n,
    sign,
    col,
    row: keep.map((i) => rho[i]),
    beta,
    gamma,
    AFull,
    bFull,
    consistent,
    z: (zHat) => zHat.map((v, j) => beta * col[j] * v),
    x: (zHat) => zHat.slice(0, n).map((v) => beta * v),
  };
}

/** Solve the Newton system by the normal equations (Nocedal & Wright §14.2). */
function newton(
  A: Matrix,
  L: number[][],
  z: readonly number[],
  s: readonly number[],
  rB: readonly number[],
  rC: readonly number[],
  rZs: readonly number[],
): [number[], number[], number[]] {
  const N = z.length;
  const t = rZs.map((v, j) => (v + z[j] * rC[j]) / s[j]);
  const At = matvec(A, t);
  const dlam = cholSolve(
    L,
    rB.map((v, i) => -v - At[i]),
  );
  const Atl = rmatvec(A, dlam, N);
  const ds = rC.map((v, j) => -v - Atl[j]);
  const dz = rZs.map((v, j) => (v - z[j] * ds[j]) / s[j]);
  return [dz, dlam, ds];
}

/** Largest α ∈ [0, 1] with v + α dv ≥ 0 (v > 0). */
function maxStep(v: readonly number[], dv: readonly number[]): number {
  let m = Infinity;
  let any = false;
  for (let i = 0; i < v.length; i++)
    if (dv[i] < 0) {
      any = true;
      m = Math.min(m, -v[i] / dv[i]);
    }
  return any ? Math.min(1.0, m) : 1.0;
}

/** True when y = λ/bᵀλ has bᵀy = 1 and max(Aᵀy) ≤ ε (primal infeasible). */
function farkasPrimal(A: Matrix, b: readonly number[], lam: readonly number[], N: number): boolean {
  const bLam = dotv(b, lam);
  return bLam > 0.0 && maxv(rmatvec(A, lam, N)) <= CERT_TOL * bLam;
}

/** True when d = z/(−cᵀz) ≥ 0 has cᵀd = −1 and ‖A d‖∞ ≤ ε (dual infeasible). */
function descentRay(A: Matrix, c: readonly number[], z: readonly number[]): boolean {
  const cz = dotv(c, z);
  return cz < 0.0 && maxv(matvec(A, z).map(Math.abs), 0.0) <= CERT_TOL * -cz;
}

const INCONSISTENT =
  'LP is infeasible: the equality rows are linearly dependent with inconsistent right-hand sides';

function resolveRay(check: Result, rayMessage: string): [string, string] {
  if (check.converged)
    return [
      'unbounded',
      `LP is unbounded: ${rayMessage}, and the LP is feasible (the same method on the ` +
        'zero objective converged)',
    ];
  if (check.extra.status === 'infeasible')
    return [
      'infeasible',
      `LP is infeasible (the same method on the zero objective: ${check.message}); ` +
        `it also has ${rayMessage}`,
    ];
  return [
    'infeasible_or_unbounded',
    `LP is unbounded or infeasible: ${rayMessage}, but feasibility is undecided (the ` +
      `same method on the zero objective: ${check.message})`,
  ];
}

const zeroObjective = (lp: LinearProgram): LinearProgram => ({
  ...lp,
  c: zeros(lp.c.length),
  sense: 'min',
  integer: [],
});

const RAY = 'an approximate ray of descent d (cᵀd = −1, ‖A d‖∞ ≤ 1e-8, scaled data)';
const FARKAS =
  'LP is infeasible: approximate Farkas certificate y = λ/bᵀλ with bᵀy = 1 and ' +
  'max(Aᵀy) ≤ 1e-8 (scaled data), so no feasible point has ‖ẑ‖₁ < 1e8';

const step = (
  k: number,
  x: number[],
  fun: number | null,
  stepSize: number | null,
  info: Record<string, unknown>,
): Step => ({
  k,
  x,
  fun,
  gradNorm: null,
  stepSize,
  info,
});

// ─────────────────────────────────────────────────────────────────────────────────────────
// Mehrotra predictor–corrector
// ─────────────────────────────────────────────────────────────────────────────────────────

interface IpmParams {
  [key: string]: unknown;
  tol?: unknown;
  eta?: unknown;
  max_iter?: unknown;
}

/** Mehrotra's predictor–corrector primal–dual method (Nocedal & Wright, Algorithm 14.3). */
export function primalDualIpm(problem: LinearProgram, o: IpmParams = {}): Result {
  const lp = checkLp(problem);
  const tol = Number(o.tol ?? 1e-8),
    eta = Number(o.eta ?? 0.99),
    maxIter = Number(o.max_iter ?? 100);
  if (!(0 < eta && eta < 1)) throw new Error('eta must be in (0, 1)');
  if (!(tol > 0) || maxIter < 1) throw new Error('tol must be positive and max_iter ≥ 1');
  const sf = scaledForm(lp);
  const { A, b, c } = sf;
  const cOrig = [...lp.c];
  const N = c.length;
  const normB = norm2(b),
    normC = norm2(c);
  const bg = sf.beta * sf.gamma; // zᵀs (original) = βγ ẑᵀŝ
  const trace: Step[] = [];

  const objective = (zHat: readonly number[]) => dotv(cOrig, sf.x(zHat));
  const baseInfo = (zHat: readonly number[]) => {
    const zo = sf.z(zHat);
    const r = sf.AFull.map((row, i) => dotv(row, zo) - sf.bFull[i]);
    return { x: sf.x(zHat), primal_residual: safeNorm(r) };
  };

  if (!sf.consistent) {
    const z0 = zeros(N);
    trace.push(step(0, sf.x(z0), objective(z0), null, baseInfo(z0)));
    return {
      method: 'primal_dual_ipm',
      x: sf.x(z0),
      fun: objective(z0),
      converged: false,
      message: INCONSISTENT,
      nIter: 0,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace,
      extra: { status: 'infeasible' },
    };
  }

  // Starting point (Nocedal & Wright §14.2).
  const AAt = A.map((ri) => A.map((rj) => dotv(ri, rj)));
  const L0 = A.length ? cholesky(AAt) : [];
  if (L0 === null) throw new Error(`${lp.id}: A has dependent rows that could not be removed`);
  let z = rmatvec(A, cholSolve(L0, b), N);
  let lam = cholSolve(L0, matvec(A, c));
  let s = (() => {
    const Al = rmatvec(A, lam, N);
    return c.map((v, j) => v - Al[j]);
  })();
  const zShift = Math.max(-1.5 * minv(z), 0.0);
  z = z.map((v) => v + zShift);
  const sShift = Math.max(-1.5 * minv(s), 0.0);
  s = s.map((v) => v + sShift);
  // Both shifts of N&W (14.42) use the same (ẑ, ŝ).
  const zs0 = dotv(z, s);
  const dzShift = (0.5 * zs0) / Math.max(sumv(s), 1e-300);
  const dsShift = (0.5 * zs0) / Math.max(sumv(z), 1e-300);
  z = z.map((v) => v + dzShift);
  s = s.map((v) => v + dsShift);
  if (!(z.every((v) => v > 0) && s.every((v) => v > 0))) {
    // NOTE: degenerate cases (b = 0 and c = 0): fall back to the all-ones start.
    z = ones(N);
    s = ones(N);
  }

  let last: {
    alpha_primal: number | null;
    alpha_dual: number | null;
    alpha_aff_primal: number | null;
    alpha_aff_dual: number | null;
    sigma: number | null;
    x_affine: number[] | null;
  } = {
    alpha_primal: null,
    alpha_dual: null,
    alpha_aff_primal: null,
    alpha_aff_dual: null,
    sigma: null,
    x_affine: null,
  };
  let alphaLast: number | null = null;
  let feasibleSeen = false;
  let raySeen = false;
  let relPPrev = Infinity;
  let lost = 0;
  let status = 'max_iter',
    message = `reached max_iter=${maxIter}`;
  let k = 0;
  for (;;) {
    const Az = matvec(A, z);
    const rB = Az.map((v, i) => v - b[i]);
    const Atl = rmatvec(A, lam, N);
    const rC = Atl.map((v, j) => v + s[j] - c[j]);
    const zs = dotv(z, s);
    const mu = zs / N;
    const relP = norm2(rB) / (1.0 + normB);
    const relD = norm2(rC) / (1.0 + normC);
    const relGap = zs / (1.0 + Math.abs(dotv(c, z)));
    const dualRes = safeNorm(rC.map((v, j) => (sf.gamma * v) / sf.col[j]));
    const info = {
      ...baseInfo(z),
      mu: bg * mu,
      dual_residual: dualRes,
      gap: bg * zs,
      ...last,
    };
    trace.push(step(k, sf.x(z), objective(z), alphaLast, info));
    if (relP <= tol && relD <= tol && relGap <= tol) {
      status = 'optimal';
      message = 'optimal: relative residuals and gap ≤ tol';
      break;
    }
    if (farkasPrimal(A, b, lam, N)) {
      status = 'infeasible';
      message = FARKAS;
      break;
    }
    feasibleSeen = feasibleSeen || relP <= tol;
    if (descentRay(A, c, z)) {
      if (feasibleSeen) {
        status = 'unbounded';
        message =
          'LP is unbounded: an iterate was feasible (relative residual ≤ tol) and ' +
          'd = z/(−cᵀz) ≥ 0 has cᵀd = −1, ‖A d‖∞ ≤ 1e-8 (scaled data): an ' +
          'approximate ray of descent';
        break;
      }
      raySeen = true;
      if (relP > PROGRESS * relPPrev) {
        status = 'ray';
        break;
      }
    }
    // A z = b is linear: a long step α_p must give r_b ← (1 − α_p) r_b.
    const longStep = k > 0 && (last.alpha_primal as number) >= 0.5;
    lost = longStep && relP > tol && relP > 0.5 * relPPrev ? lost + 1 : 0;
    relPPrev = relP;
    if (k > 0 && Math.max(last.alpha_primal as number, last.alpha_dual as number) < 1e-12) {
      status = 'stalled';
      message = 'step lengths below 1e-12: the method stalled';
      break;
    }
    if (lost >= LOST_STEPS) {
      status = 'stalled';
      message =
        `the primal residual did not fall in ${LOST_STEPS} long steps (it must fall by ` +
        'the factor 1 − α): the normal equations A Z S⁻¹ Aᵀ lost their accuracy';
      break;
    }
    if (k >= maxIter) break;
    const d = z.map((v, j) => v / s[j]); // D² = Z S⁻¹
    const M = A.map((ri) => A.map((rj) => ri.reduce((acc, v, q) => acc + v * d[q] * rj[q], 0)));
    const L = A.length ? cholesky(M) : [];
    if (L === null) {
      status = 'singular';
      message = 'normal matrix A Z S⁻¹ Aᵀ is not positive definite';
      break;
    }
    const [dzA, , dsA] = newton(
      A,
      L,
      z,
      s,
      rB,
      rC,
      z.map((v, j) => -v * s[j]),
    );
    const aAffP = maxStep(z, dzA),
      aAffD = maxStep(s, dsA);
    const muAff =
      dotv(
        z.map((v, j) => v + aAffP * dzA[j]),
        s.map((v, j) => v + aAffD * dsA[j]),
      ) / N;
    const sigma = mu > 0 ? (muAff / mu) ** 3 : 0.0;
    const [dz, dlam, ds] = newton(
      A,
      L,
      z,
      s,
      rB,
      rC,
      z.map((v, j) => -v * s[j] - dzA[j] * dsA[j] + sigma * mu),
    );
    const aP = Math.min(1.0, eta * maxStep(z, dz));
    const aD = Math.min(1.0, eta * maxStep(s, ds));
    const xAff = sf.x(z.map((v, j) => v + aAffP * dzA[j]));
    z = z.map((v, j) => v + aP * dz[j]);
    lam = lam.map((v, i) => v + aD * dlam[i]);
    s = s.map((v, j) => v + aD * ds[j]);
    k += 1;
    alphaLast = aP;
    last = {
      alpha_primal: aP,
      alpha_dual: aD,
      alpha_aff_primal: aAffP,
      alpha_aff_dual: aAffD,
      sigma,
      x_affine: xAff,
    };
    if (!(allFinite(z) && allFinite(lam) && allFinite(s))) {
      status = 'non_finite';
      message = 'non-finite iterate';
      trace.push(step(k, sf.x(z), null, aP, { x: sf.x(z), ...last }));
      break;
    }
  }

  const extra: Record<string, unknown> = {};
  const breakdown = status === 'stalled' || status === 'singular' || status === 'non_finite';
  const rayCase = status === 'ray' || (raySeen && (breakdown || status === 'max_iter'));
  if (rayCase || (breakdown && !feasibleSeen && c.some((v) => v !== 0.0))) {
    // Decide feasibility with a second run on the zero objective.
    const check = primalDualIpm(zeroObjective(lp), { tol, eta, max_iter: maxIter });
    extra.feasibility_check = check.extra.status;
    if (rayCase) [status, message] = resolveRay(check, RAY);
    else if (check.extra.status === 'infeasible') {
      status = 'infeasible';
      message =
        `LP is infeasible (the same method on the zero objective: ${check.message}); ` +
        `the run on the LP itself stopped: ${message}`;
    }
  }
  const final = trace[trace.length - 1];
  return {
    method: 'primal_dual_ipm',
    x: [...(final.x as number[])],
    fun: final.fun,
    converged: status === 'optimal',
    message,
    nIter: final.k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {
      status,
      lambda: lam.map((v, i) => (sf.gamma * v) / sf.row[i]),
      s: s.map((v, j) => (sf.gamma * v) / sf.col[j]),
      z: sf.z(z),
      ...extra,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Primal affine scaling
// ─────────────────────────────────────────────────────────────────────────────────────────

/** Z r = P Z c, the projection of Z·cost onto null(A Z), applied twice (QR of Z Aᵀ). */
function scaledReducedCosts(A: Matrix, z: readonly number[], cost: readonly number[]): number[] {
  let p = z.map((v, j) => v * cost[j]);
  if (A.length === 0) return p;
  const N = z.length,
    m = A.length;
  const AZt = Array.from({ length: N }, (_, j) => A.map((r) => r[j] * z[j]));
  const { Q } = householderQR(AZt);
  for (let pass = 0; pass < 2; pass++) {
    const qtp = Array.from({ length: m }, (_, q) => {
      let s = 0;
      for (let j = 0; j < N; j++) s += Q[j][q] * p[j];
      return s;
    });
    p = p.map((v, j) => {
      let s = 0;
      for (let q = 0; q < m; q++) s += Q[j][q] * qtp[q];
      return v - s;
    });
  }
  return p;
}

interface AffineParams {
  [key: string]: unknown;
  beta?: unknown;
  variant?: unknown;
  big_m?: unknown;
  tol?: unknown;
  max_iter?: unknown;
}

/** Primal affine-scaling algorithm (Bertsimas & Tsitsiklis 1997, §9.2). */
export function affineScaling(problem: LinearProgram, o: AffineParams = {}): Result {
  const lp = checkLp(problem);
  const beta = Number(o.beta ?? 0.66),
    variant = String(o.variant ?? 'long'),
    bigM = Number(o.big_m ?? 1e6),
    tol = Number(o.tol ?? 1e-8),
    maxIter = Number(o.max_iter ?? 500);
  if (!(0 < beta && beta < 1)) throw new Error('beta must be in (0, 1)');
  if (variant !== 'long' && variant !== 'short')
    throw new Error("variant must be 'long' or 'short'");
  if (!(bigM > 0) || !(tol > 0) || maxIter < 1)
    throw new Error('big_m and tol must be positive and max_iter ≥ 1');
  const sf = scaledForm(lp);
  const { A: A0, b, c } = sf;
  const cOrig = [...lp.c];
  const N0 = c.length;
  const r0 = b.map((v, i) => v - sumv(A0[i]));
  const A = A0.map((row, i) => [...row, r0[i]]); // last column: artificial t
  const cAug = [...c, bigM];
  const colAug = [...sf.col, 1.0];
  let z = ones(N0 + 1);
  const normC = c.length ? c.reduce((a, v) => Math.max(a, Math.abs(v)), 0) : 0.0;
  const normR0 = norm2(r0);
  const feasTol = FEAS_RTOL * (1.0 + norm2(b));
  const bg = sf.beta * sf.gamma;
  const trace: Step[] = [];

  const objective = (v: readonly number[]) => dotv(cOrig, sf.x(v));
  const stepInfo = (
    v: readonly number[],
    r: readonly number[],
    alpha: number | null,
    dx: number[] | null,
  ) => {
    // NOTE: r / col can exceed the float range for a row with a tiny ρᵢ (then inf).
    const dualRes = safeNorm(r.map((x, j) => sf.gamma * Math.min(x / colAug[j], 0.0)));
    const zo = sf.z(v.slice(0, N0));
    return {
      x: sf.x(v),
      mu: (bg * dotv(v, r)) / v.length,
      primal_residual: safeNorm(sf.AFull.map((row, i) => dotv(row, zo) - sf.bFull[i])),
      dual_residual: dualRes,
      gap: bg * dotv(v, r),
      alpha,
      direction: dx,
      artificial: v[v.length - 1],
    };
  };
  const tz = () => z[z.length - 1];

  const classifyRay = (d: readonly number[]): [string, string] => {
    const Ad = matvec(A0, d.slice(0, N0)).map(Math.abs);
    if (maxv(Ad, 0.0) <= CERT_TOL) {
      if (tz() * normR0 <= feasTol)
        return [
          'unbounded',
          'LP is unbounded: the iterate is feasible (t ≈ 0) and the big-M problem ' +
            'has a ray of descent d with ‖A d_z‖∞ ≤ 1e-8 (scaled data), a ray of the LP',
        ];
      return [
        'infeasible_or_unbounded',
        `LP is unbounded or infeasible: the LP has an approximate ray of descent ` +
          `(‖A d_z‖∞ ≤ 1e-8, scaled data), but the artificial t = ${pyG(tz(), 3)} > 0, so ` +
          'LP feasibility is not established',
      ];
    }
    return [
      'infeasible_or_big_m_too_small',
      `the big-M problem is unbounded only through the artificial t = ${pyG(tz(), 3)}: ` +
        'the LP is infeasible, or big_m is too small for the data',
    ];
  };

  const result = (status: string, message: string, extra: Record<string, unknown>): Result => {
    const final = trace[trace.length - 1];
    return {
      method: 'affine_scaling',
      x: [...(final.x as number[])],
      fun: final.fun,
      converged: status === 'optimal',
      message,
      nIter: final.k,
      nFev: 0,
      nGev: 0,
      nHev: 0,
      trace,
      extra: { status, ...extra },
    };
  };

  if (!sf.consistent) {
    const v = zeros(N0 + 1);
    trace.push(step(0, sf.x(v), objective(v), null, stepInfo(v, zeros(N0 + 1), null, null)));
    return result('infeasible', INCONSISTENT, { artificial: 0.0 });
  }

  let alpha: number | null = null;
  let dxLast: number[] | null = null;
  let rayOpen: [string, string] | null = null;
  let status = 'max_iter',
    message = `reached max_iter=${maxIter}`;
  let k = 0;
  for (;;) {
    let zr = scaledReducedCosts(A, z, cAug); // zᵢrᵢ
    const r = zr.map((v, j) => v / z[j]);
    trace.push(step(k, sf.x(z), objective(z), alpha, stepInfo(z, r, alpha, dxLast)));
    const gap = dotv(z, r);
    const dualOk = minv(r) >= -tol * (1.0 + normC);
    if (dualOk && gap <= tol * (1.0 + Math.abs(dotv(c, z.slice(0, N0))))) {
      if (tz() * normR0 <= feasTol) {
        status = 'optimal';
        message = 'optimal: r ≥ −tol and zᵀr ≤ tol (B&T §9.2 step 3)';
      } else {
        status = 'infeasible';
        message =
          `optimal for the big-M problem but the artificial t = ${pyG(tz(), 3)} > 0: ` +
          'the LP is infeasible (or big_m is too small)';
      }
      break;
    }
    if (maxv(zr) <= 0.0) {
      // −Z²r ≥ 0 is an exact ray of the big-M problem.
      const ray = z.map((v, j) => -v * zr[j]);
      const cr = -dotv(cAug, ray);
      [status, message] = classifyRay(ray.map((v) => v / cr));
      break;
    }
    if (k >= maxIter) break;
    alpha = variant === 'long' ? beta / maxv(zr) : beta / norm2(zr);
    const a = alpha;
    const dz = z.map((v, j) => -a * v * zr[j]); // −α Z² r
    const tPrev = tz();
    z = z.map((v, j) => v + dz[j]);
    dxLast = sf.x(dz);
    k += 1;
    if (!allFinite(z)) {
      status = 'non_finite';
      message = 'non-finite iterate';
      trace.push(step(k, sf.x(z), null, alpha, { x: sf.x(z) }));
      break;
    }
    if (descentRay(A, cAug, z)) {
      const cz = -dotv(cAug, z);
      rayOpen = classifyRay(z.map((v) => v / cz));
    } else rayOpen = null;
    const finalize =
      rayOpen !== null &&
      (rayOpen[0] === 'unbounded' ||
        (rayOpen[0] === 'infeasible_or_unbounded' && tz() > PROGRESS * tPrev) ||
        (rayOpen[0] === 'infeasible_or_big_m_too_small' && tz() >= tPrev));
    if (rayOpen !== null && finalize) {
      [status, message] = rayOpen;
      if (status === 'infeasible_or_unbounded') status = 'ray';
      zr = scaledReducedCosts(A, z, cAug);
      trace.push(
        step(
          k,
          sf.x(z),
          objective(z),
          alpha,
          stepInfo(
            z,
            zr.map((v, j) => v / z[j]),
            alpha,
            dxLast,
          ),
        ),
      );
      break;
    }
  }

  if (rayOpen !== null && (status === 'max_iter' || status === 'non_finite'))
    [status, message] = [rayOpen[0], `${rayOpen[1]} (then: ${message})`];
  const extra: Record<string, unknown> = {};
  if (
    status === 'ray' ||
    status === 'infeasible_or_unbounded' ||
    status === 'infeasible_or_big_m_too_small'
  ) {
    const check = affineScaling(zeroObjective(lp), {
      beta,
      variant,
      big_m: bigM,
      tol,
      max_iter: maxIter,
    });
    extra.feasibility_check = check.extra.status;
    if (status !== 'infeasible_or_big_m_too_small') [status, message] = resolveRay(check, RAY);
    else if (check.extra.status === 'infeasible') {
      message =
        `LP is infeasible (the same method on the zero objective: ${check.message}); ` +
        `before that: ${message}`;
      status = 'infeasible';
    } else if (check.converged) {
      status = 'big_m_too_small';
      message =
        'the LP is feasible (the same method on the zero objective converged), but the ' +
        `big-M problem is unbounded through the artificial t = ${pyG(tz(), 3)}: big_m is ` +
        'too small for the data, or the LP is unbounded';
    }
  }
  return result(status, message, { artificial: tz(), ...extra });
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────────────────

registerMethod<LinearProgram>(
  {
    id: 'primal_dual_ipm',
    family: 'lp',
    name: 'Primal–dual interior point (Mehrotra)',
    params: [
      param.float('tol', 1e-8, {
        min: 1e-12,
        max: 1e-3,
        log: true,
        help: 'Stop when the relative primal residual, dual residual and gap of the scaled LP are all ≤ tol.',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
      param.float('eta', 0.99, {
        min: 0.5,
        max: 0.9999,
        help: 'Fraction of the step to the boundary (α = min(1, η·α_max)).',
        label: 'Step fraction',
        tex: '\\eta',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 1000,
        help: 'Iteration limit.',
        label: 'Iteration budget',
      }),
    ],
    needs: ['lp'],
    order: 'superlinear in practice (polynomial bound for path-following variants)',
    summary:
      'Follow the central path with Newton steps on the perturbed KKT conditions, using an affine predictor and a centering corrector.',
    references: [
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Algorithm 14.3 and §14.2',
      'Mehrotra (1992), SIAM J. Optim. 2(4):575–601',
      'Wright, Primal-Dual Interior-Point Methods (1997), ch. 9–11',
    ],
  },
  (p, o) => primalDualIpm(p, o),
  {
    rule: '\\begin{bmatrix} 0 & A^{\\top} & I \\\\ A & 0 & 0 \\\\ S & 0 & Z \\end{bmatrix}\\!\\begin{bmatrix} \\Delta\\mathbf{z} \\\\ \\Delta\\boldsymbol{\\lambda} \\\\ \\Delta\\mathbf{s} \\end{bmatrix} = \\begin{bmatrix} -\\mathbf{r}_c \\\\ -\\mathbf{r}_b \\\\ -ZS\\mathbf{e} + \\sigma\\mu\\mathbf{e} \\end{bmatrix}',
    intuition:
      'Instead of walking the boundary, it follows the central path zᵢsᵢ = μ through the interior while μ shrinks to zero. The affine predictor measures how far a pure Newton step gets; the centering σ = (μ_aff/μ)³ says how much to pull back toward the path.',
    order: 'superlinear in practice',
    pros: ['Polynomial-time variants', 'Few iterations, independent of the number of vertices'],
    cons: ['Ends near, not on, a vertex', 'Each step solves a normal system'],
    quantities: [
      { tex: '\\mu_k', key: 'info.mu' },
      { tex: '\\sigma_k', key: 'info.sigma' },
      { tex: '\\|\\mathbf{r}_b\\|', key: 'info.primal_residual' },
    ],
  },
);

registerMethod<LinearProgram>(
  {
    id: 'affine_scaling',
    family: 'lp',
    name: 'Affine scaling (Dikin)',
    params: [
      param.float('beta', 0.66, {
        min: 0.05,
        max: 0.99,
        help: 'Step fraction: of the distance to the boundary (long) or of the Dikin ellipsoid radius (short). β ≤ 2/3 guarantees convergence.',
        label: 'Step fraction',
        tex: '\\beta',
      }),
      param.choice('variant', 'long', ['long', 'short'], {
        help: 'long: α = β / maxᵢ zᵢrᵢ (Vanderbei et al.); short: α = β / ‖Zr‖₂ (Dikin).',
        label: 'Step',
      }),
      param.float('big_m', 1e6, {
        min: 10.0,
        max: 1e9,
        log: true,
        help: 'Cost of the artificial variable that makes z = 1 an interior start (relative to the scaled costs, ‖ĉ‖∞ = 1).',
        label: 'Artificial cost',
        tex: 'M',
      }),
      param.float('tol', 1e-8, {
        min: 1e-12,
        max: 1e-3,
        log: true,
        help: 'Stop when r ≥ −tol·(1+‖c‖∞) and zᵀr ≤ tol·(1+|cᵀz|) on the scaled LP.',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
      param.int('max_iter', 500, {
        min: 1,
        max: 10_000,
        help: 'Iteration limit.',
        label: 'Iteration budget',
      }),
    ],
    needs: ['lp'],
    order: 'linear',
    summary:
      'Rescale so the current point is the center of the orthant, then step along the projected steepest-descent direction.',
    references: [
      'Dikin (1967), Soviet Math. Dokl. 8:674–675',
      'Vanderbei, Meketon & Freedman (1986), Algorithmica 1:395–407',
      'Bertsimas & Tsitsiklis, Introduction to Linear Optimization (1997), §9.2',
      "Hall & Vanderbei (1993), 'Two-thirds is sharp for affine scaling', Oper. Res. Lett. 13:197–201",
    ],
  },
  (p, o) => affineScaling(p, o),
  {
    rule: '\\mathbf{z}_{k+1} = \\mathbf{z}_k - \\alpha_k Z_k^2 \\mathbf{r}_k,\\qquad \\alpha_k = \\frac{\\beta}{\\max_i z_i r_i}',
    intuition:
      'Rescale the variables so the current point sits at e = (1, …, 1), where a ball fits inside the orthant; step along the projected negative cost there, then scale back. Close to a face the rescaling stretches it, so the steps shorten and bend toward the optimum.',
    order: 'linear',
    pros: ['One projection per step', 'Simple geometry: a Dikin ellipsoid step'],
    cons: ['Linear convergence', 'Needs β ≤ 2/3 for guaranteed convergence', 'Big-M start'],
    quantities: [
      { tex: 't_k', key: 'info.artificial' },
      { tex: '\\mathbf{z}^{\\top}\\mathbf{r}', key: 'info.gap' },
    ],
  },
);
