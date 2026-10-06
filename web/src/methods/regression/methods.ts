/**
 * Regression of data (x_i, y_i), i = 1..m, on polynomials in x (mostly the line β₀ + β₁x) — TS
 * port of `numopt.regression.methods` (src/numopt/regression/methods.py).
 *
 * Coefficients are always in increasing powers, β = [β₀, β₁, …, β_d], so ŷ(x) = Σ_j β_j x^j.
 *
 *   - `Result.x`: β.
 *   - `Result.fun`: the objective the method minimizes: RSS for least squares, RSS + λ‖β_{1:}‖²
 *     for ridge, Σ ρ_δ(r_i/σ̂) for Huber, Σ|r_i| for LAD, max|r_i| for the minimax line; the RSS
 *     for Theil–Sen (which minimizes no objective).
 *   - `Result.extra` (snake_case keys, as exported): coefficients, degree, fitted, residuals, rss,
 *     tss, r_squared, adj_r_squared, rmse, sigma, std_errors, eval {x, y}, plus method keys.
 *
 * Numerics as in Python: least squares factors the column-equilibrated design (van der Sluis
 * 1969) by Householder QR; the rank is the number of singular values σ_k > max(m, p)·ε·σ_max of
 * that matrix (here from a one-sided Jacobi SVD, which is accurate to high relative precision);
 * only the explicit `normal_equations` solver forms AᵀA. The robust IRLS lines solve in the
 * centred variable x − x̄. `nFev` is 0: the methods read data and evaluate no function.
 *
 * Info keys (snake_case, as in the Python docstring): solver, rank, cond, cond_gram, lambda,
 * weights, scale, delta, smoothed_objective, slopes, reference, level, entering, max_deviation.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Dataset, Matrix, MethodFn, Result, Step, StepInfo, Vector } from '../../core/types';
import { linspace } from '../../problems/data';

/** Number of points on the plotting grid in `extra.eval`. */
export const N_GRID = 200;
/** Φ⁻¹(3/4): MAD / Φ⁻¹(3/4) estimates σ for Gaussian data (normalized MAD). */
export const MAD_TO_SIGMA = 0.6744897501960817;
const EPS = 2 ** -52;
/** Smallest subnormal 2⁻¹⁰⁷⁴ (the underflow term η of the standard model, Higham eq. (2.8)). */
const ETA = 5e-324;
/** Smallest normal float64, 2⁻¹⁰²². */
const TINY = 2.2250738585072014e-308;
/** κ₂(AᵀA)·ε at or above this: the normal-equations solve is reported as not converged. */
export const NE_MAX_FORWARD_BOUND = 1e-2;
const PRELIM_LAD_EPS = 1e-6;
const PRELIM_LAD_TOL = 1e-10;
const PRELIM_LAD_MAX_ITER = 500;
const MINIMAX_ROUNDING_EPS = 8.0;

/** A Dataset, or a bare `[x, y]` pair (Python accepts an `(x, y)` tuple). */
export type RegressionProblem =
  | (Pick<Dataset, 'x' | 'y'> & Partial<Pick<Dataset, 'domain' | 'id'>>)
  | readonly [readonly number[], readonly number[]];

// ---------------------------------------------------------------------------------------
// Python formatting (messages)
// ---------------------------------------------------------------------------------------

/** Python's `format(v, '.{p}g')`. */
export function pyG(v: number, p = 6): string {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, expStr] = v.toExponential(Math.max(0, p - 1)).split('e');
  const exp = Number(expStr);
  const strip = (s: string) => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
  if (exp < -4 || exp >= p) {
    const e = Math.abs(exp);
    return `${strip(mant)}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  return strip(v.toFixed(Math.max(0, p - 1 - exp)));
}

// ---------------------------------------------------------------------------------------
// Small numerics (NumPy semantics where it matters)
// ---------------------------------------------------------------------------------------

/** `np.sum` of a contiguous vector: NumPy's pairwise summation. */
export function npSum(v: readonly number[]): number {
  return 0.0 + pairwise(v, 0, v.length);
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

const npMean = (v: readonly number[]) => npSum(v) / v.length;

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const maxAbs = (v: readonly number[]) => v.reduce((m, t) => Math.max(m, Math.abs(t)), 0);
const allFinite = (v: readonly number[]) => v.every(Number.isFinite);

/** `np.median` (mean of the two middle values for an even count). */
export function median(v: readonly number[]): number {
  const s = sortAsc(v);
  const n = s.length;
  if (n === 0) return NaN;
  const h = n >> 1;
  return n % 2 === 1 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** Ascending sort with NaN last (NumPy's order). */
function sortAsc(v: readonly number[]): number[] {
  return [...v].sort((a, b) => {
    if (Number.isNaN(a)) return Number.isNaN(b) ? 0 : 1;
    if (Number.isNaN(b)) return -1;
    return a - b;
  });
}

/** V_{ij} = x_i^j, j = 0..degree (increasing powers), built by repeated multiplication. */
export function vandermonde(x: readonly number[], degree: number): Matrix {
  return x.map((xi) => {
    const row = new Array<number>(degree + 1);
    row[0] = 1.0;
    for (let j = 1; j <= degree; j++) row[j] = row[j - 1] * xi;
    return row;
  });
}

/** Horner evaluation of Σ_j β_j t^j. */
export function polyval(beta: readonly number[], t: number): number {
  let out = beta[beta.length - 1];
  for (let j = beta.length - 2; j >= 0; j--) out = beta[j] + t * out;
  return out;
}

const polyvalAll = (beta: readonly number[], t: readonly number[]) =>
  t.map((v) => polyval(beta, v));

/** 2-norms of the columns of `a` (1 for a zero column); sequential sums, as NumPy's axis-0. */
function columnScales(a: Matrix, p: number): number[] {
  const s = new Array<number>(p).fill(0);
  for (const row of a) for (let j = 0; j < p; j++) s[j] += row[j] * row[j];
  return s.map((v) => {
    const r = Math.sqrt(v);
    return r > 0 ? r : 1.0;
  });
}

const divCols = (a: Matrix, s: readonly number[]): Matrix =>
  a.map((row) => row.map((v, j) => v / s[j]));

/** Solve the upper-triangular R β = z (Golub & Van Loan, Alg. 3.1.2). */
function backSubstitute(r: Matrix, z: readonly number[]): Vector {
  const p = z.length;
  const beta = new Array<number>(p);
  for (let i = p - 1; i >= 0; i--) {
    let s = 0;
    for (let j = i + 1; j < p; j++) s += r[i][j] * beta[j];
    beta[i] = (z[i] - s) / r[i][i];
  }
  return beta;
}

/** Solve the lower-triangular L w = z (Golub & Van Loan, Alg. 3.1.1). */
function forwardSubstitute(lo: Matrix, z: readonly number[]): Vector {
  const p = z.length;
  const w = new Array<number>(p);
  for (let i = 0; i < p; i++) {
    let s = 0;
    for (let j = 0; j < i; j++) s += lo[i][j] * w[j];
    w[i] = (z[i] - s) / lo[i][i];
  }
  return w;
}

const transposeM = (a: Matrix): Matrix =>
  a.length === 0 ? [] : a[0].map((_, j) => a.map((row) => row[j]));

/**
 * Thin SVD A = U diag(s) Vᵀ by one-sided Jacobi (Hestenes 1958; Demmel & Veselić 1992: high
 * relative accuracy). Returns r = min(m, p) singular values in descending order with U (m × r)
 * and V (p × r) as column lists.
 */
export function svd(a: Matrix): { u: Vector[]; s: Vector; v: Vector[] } {
  const m = a.length;
  const p = m ? a[0].length : 0;
  if (m < p) {
    const t = svd(transposeM(a));
    return { u: t.v, s: t.s, v: t.u };
  }
  // Columns of W = A V, rotated until mutually orthogonal.
  const w: Vector[] = Array.from({ length: p }, (_, j) => a.map((row) => row[j]));
  const v: Vector[] = Array.from({ length: p }, (_, j) =>
    Array.from({ length: p }, (_, i) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < 80; sweep++) {
    let rotated = false;
    for (let i = 0; i < p - 1; i++)
      for (let j = i + 1; j < p; j++) {
        const wi = w[i],
          wj = w[j];
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let q = 0; q < m; q++) {
          alpha += wi[q] * wi[q];
          beta += wj[q] * wj[q];
          gamma += wi[q] * wj[q];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        if (t === 0 || !Number.isFinite(t)) continue;
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        rotated = true;
        for (let q = 0; q < m; q++) {
          const x = wi[q],
            y = wj[q];
          wi[q] = c * x - s * y;
          wj[q] = s * x + c * y;
        }
        const vi = v[i],
          vj = v[j];
        for (let q = 0; q < p; q++) {
          const x = vi[q],
            y = vj[q];
          vi[q] = c * x - s * y;
          vj[q] = s * x + c * y;
        }
      }
    if (!rotated) break;
  }
  const norms = w.map((col) => {
    const big = maxAbs(col);
    if (big === 0 || !Number.isFinite(big)) return big;
    let ss = 0;
    for (const c of col) ss += (c / big) * (c / big);
    return big * Math.sqrt(ss);
  });
  const order = norms.map((_, j) => j).sort((i, j) => norms[j] - norms[i]);
  return {
    s: order.map((j) => norms[j]),
    u: order.map((j) => (norms[j] > 0 ? w[j].map((c) => c / norms[j]) : w[j].map(() => 0))),
    v: order.map((j) => v[j]),
  };
}

/** Numerical rank (matrix_rank rule) and κ₂ from the descending singular values. */
export function rankCond(sv: readonly number[], shape: [number, number]): [number, number] {
  const smax = sv.length ? sv[0] : 0;
  const tol = Math.max(shape[0], shape[1]) * EPS * smax;
  const rank = sv.filter((s) => s > tol).length;
  const full = sv.length === shape[1] && sv.length > 0 && sv[sv.length - 1] > 0;
  return [rank, full ? smax / sv[sv.length - 1] : Infinity];
}

const singularValues = (a: Matrix) => svd(a).s;

/**
 * Least squares min ‖Aβ − rhs‖₂ by Householder QR (Golub & Van Loan, Alg. 5.3.2), m ≥ p and
 * full column rank. Returns (β, R).
 */
export function qrLstsq(a: Matrix, rhs: readonly number[]): [Vector, Matrix] {
  const m = a.length;
  const p = a[0].length;
  const A = a.map((row) => row.slice());
  const b = rhs.slice();
  for (let j = 0; j < p; j++) {
    let big = 0;
    for (let i = j; i < m; i++) big = Math.max(big, Math.abs(A[i][j]));
    if (big === 0) continue;
    let ss = 0;
    for (let i = j; i < m; i++) ss += (A[i][j] / big) ** 2;
    const nrm = big * Math.sqrt(ss);
    const alpha = A[j][j] > 0 ? -nrm : nrm;
    const v = new Array<number>(m - j);
    for (let i = j; i < m; i++) v[i - j] = A[i][j];
    v[0] -= alpha;
    const vv = dot(v, v);
    if (vv === 0) continue;
    for (let c = j; c < p; c++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i - j] * A[i][c];
      const f = (2 * s) / vv;
      for (let i = j; i < m; i++) A[i][c] -= f * v[i - j];
    }
    let s = 0;
    for (let i = j; i < m; i++) s += v[i - j] * b[i];
    const f = (2 * s) / vv;
    for (let i = j; i < m; i++) b[i] -= f * v[i - j];
    A[j][j] = alpha;
    for (let i = j + 1; i < m; i++) A[i][j] = 0;
  }
  const R = A.slice(0, p).map((row) => row.slice(0, p));
  return [backSubstitute(R, b.slice(0, p)), R];
}

/** diag((RᵀR)⁻¹) (upper R) or diag((LLᵀ)⁻¹) (lower L), one triangular solve per column. */
function invDiagFromTriangle(r: Matrix, upper: boolean): Vector {
  const p = r.length;
  const lo = upper ? transposeM(r) : r;
  return Array.from({ length: p }, (_, j) => {
    const e = new Array<number>(p).fill(0);
    e[j] = 1;
    const z = forwardSubstitute(lo, e);
    return dot(z, z);
  });
}

/** Lower Cholesky factor of a symmetric matrix; null when it is not numerically PD (dpotrf). */
function cholesky(a: Matrix): Matrix | null {
  const n = a.length;
  const L: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let d = a[j][j];
    for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k];
    if (!(d > 0)) return null;
    const ljj = Math.sqrt(d);
    L[j][j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = s / ljj;
    }
  }
  return L;
}

/** Gaussian elimination with partial pivoting; null on an exactly zero pivot (dgesv). */
function solveDense(a: Matrix, b: readonly number[]): Vector | null {
  const n = a.length;
  const A = a.map((r) => r.slice());
  const x = b.slice();
  for (let k = 0; k < n; k++) {
    let piv = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[piv][k])) piv = i;
    if (A[piv][k] === 0 || Number.isNaN(A[piv][k])) return null;
    if (piv !== k) {
      [A[k], A[piv]] = [A[piv], A[k]];
      [x[k], x[piv]] = [x[piv], x[k]];
    }
    for (let i = k + 1; i < n; i++) {
      const f = A[i][k] / A[k][k];
      if (f === 0) continue;
      for (let j = k; j < n; j++) A[i][j] -= f * A[k][j];
      x[i] -= f * x[k];
    }
  }
  return backSubstitute(A, x);
}

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

interface Data {
  x: number[];
  y: number[];
  a: number;
  b: number;
  m: number;
}

function resolve(problem: RegressionProblem, minPoints: number): Data {
  let xr: readonly number[], yr: readonly number[];
  let domain: readonly number[] | null | undefined = null;
  if (Array.isArray(problem) && problem.length === 2) {
    [xr, yr] = problem as readonly [readonly number[], readonly number[]];
  } else if (problem && typeof problem === 'object' && 'x' in problem && 'y' in problem) {
    const p = problem as Pick<Dataset, 'x' | 'y'> & Partial<Pick<Dataset, 'domain'>>;
    xr = p.x;
    yr = p.y;
    domain = p.domain;
  } else {
    throw new TypeError('problem must be a numopt Dataset or an (x, y) pair of arrays');
  }
  if (!Array.isArray(xr) || !Array.isArray(yr) || xr.length !== yr.length)
    throw new Error(
      `x and y must be 1-D of equal length; got (${xr?.length},) and (${yr?.length},)`,
    );
  const x = xr.map(Number);
  const y = yr.map(Number);
  if (x.length < minPoints)
    throw new Error(`need at least ${minPoints} data points, got ${x.length}`);
  if (!allFinite(x) || !allFinite(y)) throw new Error('data contain NaN or infinite values');
  let a: number, b: number;
  if (domain) {
    a = Number(domain[0]);
    b = Number(domain[1]);
  } else {
    a = Math.min(...x);
    b = Math.max(...x);
  }
  if (!(a < b)) {
    a -= 1.0;
    b += 1.0;
  }
  return { x, y, a, b, m: x.length };
}

/** Residual statistics and the plotting curve for a fitted polynomial β (Python `_goodness`). */
export function goodness(
  data: { x: readonly number[]; y: readonly number[]; a: number; b: number },
  beta: readonly number[],
  nParams: number,
  covDiag: readonly number[] | null,
): Record<string, unknown> {
  const fitted = polyvalAll(beta, data.x);
  const resid = data.y.map((v, i) => v - fitted[i]);
  const rss = dot(resid, resid);
  const ybar = npMean(data.y);
  const dev = data.y.map((v) => v - ybar);
  const tss = dot(dev, dev);
  const m = data.x.length;
  // NOTE (Python): constant y (to rounding) gives no meaningful R²; reported as null.
  const flat = maxAbs(dev) <= m * EPS * maxAbs(data.y);
  const r2 = tss > 0 && !flat ? 1.0 - rss / tss : null;
  const dof = m - nParams;
  const adj = r2 !== null && dof > 0 ? 1.0 - ((1.0 - r2) * (m - 1)) / dof : null;
  const sigma = dof > 0 ? Math.sqrt(rss / dof) : null;
  const se = covDiag !== null && sigma !== null ? covDiag.map((c) => sigma * Math.sqrt(c)) : null;
  const grid = linspace(data.a, data.b, N_GRID);
  return {
    coefficients: beta.slice(),
    degree: nParams - 1,
    fitted,
    residuals: resid,
    rss,
    tss,
    r_squared: r2,
    adj_r_squared: adj,
    rmse: Math.sqrt(rss / m),
    sigma,
    std_errors: se,
    eval: { x: grid, y: polyvalAll(beta, grid) },
  };
}

function step(
  k: number,
  x: readonly number[],
  fun: number | null,
  info: StepInfo = {},
  stepSize: number | null = null,
): Step {
  return { k, x: x.slice(), fun, gradNorm: null, stepSize, info };
}

function result(
  method: string,
  x: readonly number[],
  fun: number | null,
  converged: boolean,
  message: string,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  return {
    method,
    x: x.slice(),
    fun,
    converged,
    message,
    nIter: trace[trace.length - 1].k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra,
  };
}

const fail = (method: string, beta: readonly number[], message: string, trace: Step[]) =>
  result(method, beta, null, false, message, trace);

const nanVec = (p: number) => new Array<number>(p).fill(NaN);

// ---------------------------------------------------------------------------------------
// Ordinary least squares
// ---------------------------------------------------------------------------------------

/** Minimum-‖γ‖ least squares by the SVD (Golub & Van Loan, Thm. 5.5.1). */
function svdLstsq(a: Matrix, y: readonly number[]): [Vector, number, number, Vector | null] {
  const p = a[0].length;
  const { u, s, v } = svd(a);
  const [rank, cond] = rankCond(s, [a.length, p]);
  const coefU = Array.from({ length: rank }, (_, k) => dot(u[k], y) / s[k]);
  const gamma = Array.from({ length: p }, (_, j) => {
    let g = 0;
    for (let k = 0; k < rank; k++) g += v[k][j] * coefU[k];
    return g;
  });
  const cov =
    rank === p
      ? Array.from({ length: p }, (_, j) => {
          let c = 0;
          for (let k = 0; k < s.length; k++) c += (v[k][j] / s[k]) ** 2;
          return c;
        })
      : null;
  return [gamma, rank, cond, cov];
}

function notUniqueMessage(data: Data, p: number, rank: number): string {
  return (
    `rank-deficient design (${p} parameters, ${data.m} points, ` +
    `${new Set(data.x).size} distinct x; numerical rank ${rank} < ${p}): the ` +
    'least-squares solution is not unique; returned the minimum-norm one (SVD)'
  );
}

type Solver = 'qr' | 'normal_equations' | 'svd';

function ols(method: string, data: Data, degree: number, solver: Solver, minNorm = false): Result {
  const p = degree + 1;
  const xMat = vandermonde(data.x, degree);
  const nanBeta = nanVec(p);
  const info: StepInfo = { solver };
  let covDiag: Vector | null = null;
  let converged = true;
  let msg: string;
  let beta: Vector;
  let rank: number;
  // NOTE (Python): equilibrate before every rank decision, the SVD included.
  const scale = columnScales(xMat, p);
  const a = divCols(xMat, scale);

  if (solver === 'svd') {
    const [gamma, rk, cond, covGamma] = svdLstsq(a, data.y);
    rank = rk;
    beta = gamma.map((g, j) => g / scale[j]);
    Object.assign(info, { rank, cond });
    if (rank === p && covGamma !== null) {
      covDiag = covGamma.map((c, j) => c / scale[j] ** 2);
      msg = 'least-squares solution by SVD';
    } else {
      converged = false;
      msg = notUniqueMessage(data, p, rank);
    }
  } else {
    const [rk, cond] = rankCond(singularValues(a), [a.length, p]);
    rank = rk;
    Object.assign(info, { rank, cond });
    if (solver === 'normal_equations') info.cond_gram = cond * cond;
    if (rank < p && minNorm) {
      const [gamma, rk2, cond2] = svdLstsq(a, data.y);
      rank = rk2;
      beta = gamma.map((g, j) => g / scale[j]);
      Object.assign(info, { solver: 'svd', rank: rk2, cond: cond2 });
      converged = false;
      msg = notUniqueMessage(data, p, rk2);
    } else if (rank < p) {
      return fail(
        method,
        nanBeta,
        `rank-deficient design (rank ${rank} < ${p}): the least-squares solution ` +
          "is not unique (use solver='svd' for the minimum-norm one)",
        [step(0, nanBeta, null, info)],
      );
    } else if (solver === 'qr') {
      const [gamma, r] = qrLstsq(a, data.y);
      covDiag = invDiagFromTriangle(r, true).map((c, j) => c / scale[j] ** 2);
      msg = 'least-squares solution by Householder QR';
      beta = gamma.map((g, j) => g / scale[j]);
    } else {
      // NOTE (Python): deliberately the normal equations AᵀA γ = Aᵀy (Cholesky).
      const at = transposeM(a);
      const gram = at.map((ri) => at.map((rj) => dot(ri, rj)));
      const chol = cholesky(gram);
      if (chol === null)
        return fail(
          method,
          nanBeta,
          `Cholesky of AᵀA failed (κ(AᵀA) ≈ ${pyG(cond * cond, 2)}): ` +
            'the normal equations are numerically singular',
          [step(0, nanBeta, null, info)],
        );
      const w = forwardSubstitute(
        chol,
        at.map((row) => dot(row, data.y)),
      );
      const gamma = backSubstitute(transposeM(chol), w);
      covDiag = invDiagFromTriangle(chol, false).map((c, j) => c / scale[j] ** 2);
      beta = gamma.map((g, j) => g / scale[j]);
      const condGram = cond * cond;
      const bound = condGram * EPS;
      const lost = `κ₂(AᵀA) ≈ ${pyG(condGram, 3)}: about ${Math.log10(condGram).toFixed(1)} of 16 digits lost`;
      if (bound >= NE_MAX_FORWARD_BOUND) {
        converged = false;
        msg =
          `normal equations solved, but ${lost} (forward error bound ` +
          `κ₂(AᵀA)·ε = ${pyG(bound, 2)} ≥ ${pyG(NE_MAX_FORWARD_BOUND)}): the coefficients ` +
          "may have no correct digits; use solver='qr'";
      } else {
        msg = `least-squares solution by the normal equations (Cholesky); ${lost}`;
      }
    }
  }
  if (!allFinite(beta))
    return fail(method, beta, 'non-finite coefficients', [step(0, beta, null, info)]);
  const stats = goodness(data, beta, p, rank === p ? covDiag : null);
  Object.assign(stats, { rank: info.rank, cond: info.cond, solver: info.solver });
  const rss = stats.rss as number;
  return result(method, beta, rss, converged, msg, [step(0, beta, rss, info)], stats);
}

const linearRegression: MethodFn<RegressionProblem> = (problem, { solver = 'qr' }) => {
  if (solver !== 'qr' && solver !== 'normal_equations' && solver !== 'svd')
    throw new Error(`unknown solver '${String(solver)}'`);
  return ols('linear_regression', resolve(problem, 1), 1, solver);
};

const polynomialRegression: MethodFn<RegressionProblem> = (problem, { degree = 2 }) => {
  const d = Math.trunc(Number(degree));
  if (d < 0) throw new Error('degree must be ≥ 0');
  return ols('polynomial_regression', resolve(problem, 1), d, 'qr', true);
};

// ---------------------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------------------

const ridgeRegression: MethodFn<RegressionProblem> = (problem, { lam = 1.0, degree = 1 }) => {
  const l = Number(lam);
  if (!(l >= 0.0 && Number.isFinite(l))) throw new Error('lam must be a finite number ≥ 0');
  if (Number(degree) < 0) throw new Error('degree must be ≥ 0');
  const data = resolve(problem, 1);
  const d = Math.trunc(Number(degree));
  const p = d + 1;
  const ybar = npMean(data.y);
  const info: StepInfo = { lambda: l };
  let beta: Vector;
  if (d === 0) {
    beta = [ybar];
    Object.assign(info, { rank: 1, cond: 1.0 });
  } else {
    const feats = vandermonde(data.x, d).map((row) => row.slice(1)); // (m, d)
    const xbar = new Array<number>(d).fill(0);
    for (const row of feats) for (let j = 0; j < d; j++) xbar[j] += row[j];
    for (let j = 0; j < d; j++) xbar[j] /= data.m;
    const xc = feats.map((row) => row.map((v, j) => v - xbar[j]));
    const yc = data.y.map((v) => v - ybar);
    const scale = columnScales(xc, d);
    const sl = Math.sqrt(l);
    const aug: Matrix = [
      ...divCols(xc, scale),
      ...Array.from({ length: d }, (_, i) =>
        Array.from({ length: d }, (_, j) => (i === j ? sl / scale[j] : 0)),
      ),
    ];
    const rhs = [...yc, ...new Array<number>(d).fill(0)];
    const [rank, cond] = rankCond(singularValues(aug), [aug.length, d]);
    Object.assign(info, { rank, cond });
    if (rank < d) {
      const nanBeta = nanVec(p);
      return fail(
        'ridge_regression',
        nanBeta,
        `λ = 0 and the design is rank deficient (rank ${rank} < ${d}): no unique solution`,
        [step(0, nanBeta, null, info)],
      );
    }
    const [gamma] = qrLstsq(aug, rhs);
    const slopes = gamma.map((g, j) => g / scale[j]);
    beta = [ybar - dot(xbar, slopes), ...slopes];
  }
  if (!allFinite(beta))
    return fail('ridge_regression', beta, 'non-finite coefficients', [step(0, beta, null)]);
  const stats = goodness(data, beta, p, null);
  const tail = beta.slice(1);
  const objective = (stats.rss as number) + l * dot(tail, tail);
  Object.assign(stats, { lambda: l, rank: info.rank, cond: info.cond });
  return result(
    'ridge_regression',
    beta,
    objective,
    true,
    `ridge solution for λ = ${pyG(l)} by QR of the augmented system`,
    [step(0, beta, objective, info)],
    stats,
  );
};

// ---------------------------------------------------------------------------------------
// Iteratively reweighted least squares (robust lines)
// ---------------------------------------------------------------------------------------

/** Weighted least squares min Σ w_i (y_i − x_iᵀβ)² by QR of diag(√w) X; null if singular. */
function wls(xMat: Matrix, y: readonly number[], w: readonly number[]): Vector | null {
  const p = xMat[0].length;
  const sw = w.map(Math.sqrt);
  const a0 = xMat.map((row, i) => row.map((v) => v * sw[i]));
  const rhs = y.map((v, i) => v * sw[i]);
  if (!a0.every(allFinite) || !allFinite(rhs)) return null;
  const scale = columnScales(a0, p);
  const a = divCols(a0, scale);
  const [rank] = rankCond(singularValues(a), [a.length, p]);
  if (rank < p) return null;
  const [gamma] = qrLstsq(a, rhs);
  return gamma.map((g, j) => g / scale[j]);
}

/** sqrt(mean(r²)) computed as ‖r‖∞·sqrt(mean((r/‖r‖∞)²)), so r² cannot overflow. */
function rms(r: readonly number[]): number {
  const rmax = r.reduce((m, v) => (Math.abs(v) > m || Number.isNaN(v) ? Math.abs(v) : m), 0);
  if (rmax === 0 || !Number.isFinite(rmax)) return rmax;
  return rmax * Math.sqrt(npMean(r.map((v) => (v / rmax) ** 2)));
}

/** Huber (1964): ρ(u) = u²/2 for |u| ≤ δ, δ|u| − δ²/2 otherwise. */
export function huberRho(u: number, delta: number): number {
  const au = Math.abs(u);
  return au <= delta ? 0.5 * u * u : delta * au - 0.5 * delta * delta;
}

/** w(u) = ψ(u)/u = min(1, δ/|u|) (w(0) = 1). */
export function huberWeight(u: number, delta: number): number {
  const au = Math.abs(u);
  return au <= delta ? 1.0 : delta / Math.max(au, delta);
}

/** ρ_ε(r) = |r| for |r| ≥ ε, r²/(2ε) + ε/2 otherwise (the function LAD-IRLS decreases). */
export function ladSmoothed(r: number, eps: number): number {
  const ar = Math.abs(r);
  return ar >= eps ? ar : (r * r) / (2.0 * eps) + 0.5 * eps;
}

function change(a: readonly number[], b: readonly number[]): number {
  return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
}

/** The centred line design X_c = [1, x − c], c = x̄ (see the NOTE in Python `_LineDesign`). */
interface LineDesign {
  mat: Matrix;
  center: number;
}

const designBeta = (d: LineDesign, theta: readonly number[]): Vector => [
  theta[0] - theta[1] * d.center,
  theta[1],
];

const designApply = (d: LineDesign, t: readonly number[]) =>
  d.mat.map((row) => row[0] * t[0] + row[1] * t[1]);

function lineDesign(data: Data): LineDesign {
  if (data.x.every((v) => v === data.x[0]))
    throw new Error('all x values are equal: the line is not identifiable');
  const lo = Math.min(...data.x),
    hi = Math.max(...data.x);
  const mid = 0.5 * lo + 0.5 * hi;
  let center = mid + npMean(data.x.map((v) => v - mid));
  if (!Number.isFinite(center)) center = mid;
  return {
    mat: vandermonde(
      data.x.map((v) => v - center),
      1,
    ),
    center,
  };
}

function olsStart(d: LineDesign, y: readonly number[]): [Vector, Vector] | null {
  const theta = wls(
    d.mat,
    y,
    y.map(() => 1),
  );
  if (theta === null || !allFinite(theta)) return null;
  const fit = designApply(d, theta);
  const r = y.map((v, i) => v - fit[i]);
  return allFinite(r) ? [theta, r] : null;
}

function startFailed(method: string, data: Data, d: LineDesign): Result {
  const theta = wls(
    d.mat,
    data.y,
    data.y.map(() => 1),
  );
  const bad = theta === null ? nanVec(2) : designBeta(d, theta);
  return fail(
    method,
    bad,
    'the least-squares start is singular or not finite (overflow; e.g. the slope ' +
      'exceeds the floating-point range): IRLS cannot start',
    [step(0, bad, null)],
  );
}

interface IrlsRun {
  beta: Vector;
  theta: Vector;
  objective: number;
  weights: Vector;
  converged: boolean;
  message: string;
  trace: Step[];
}

function ladIrls(
  d: LineDesign,
  y: readonly number[],
  theta0: readonly number[],
  eps: number,
  tol: number,
  maxIter: number,
): IrlsRun {
  const state = (t: readonly number[]): [number, number, Vector] => {
    const fit = designApply(d, t);
    const r = y.map((v, i) => v - fit[i]);
    return [
      npSum(r.map(Math.abs)),
      npSum(r.map((v) => ladSmoothed(v, eps))),
      r.map((v) => 1.0 / Math.max(Math.abs(v), eps)),
    ];
  };
  let theta = theta0.slice();
  let beta = designBeta(d, theta);
  let [obj, smooth, w] = state(theta);
  const trace = [step(0, beta, obj, { weights: w, smoothed_objective: smooth })];
  let converged = false;
  let msg = `reached max_iter=${maxIter}`;
  for (let k = 1; k <= maxIter; k++) {
    const next = wls(d.mat, y, w);
    if (next === null) {
      msg = 'weighted design matrix is rank deficient or not finite';
      break;
    }
    const ch = change(next, theta);
    theta = next;
    beta = designBeta(d, theta);
    [obj, smooth, w] = state(theta);
    trace.push(step(k, beta, obj, { weights: w, smoothed_objective: smooth }, ch));
    if (!(allFinite(beta) && Number.isFinite(obj))) {
      msg = 'non-finite coefficients or objective';
      break;
    }
    if (ch <= tol * (1.0 + maxAbs(theta))) {
      converged = true;
      msg = `‖Δθ‖∞ = ${pyG(ch, 3)} ≤ tol·(1 + ‖θ‖∞)`;
      break;
    }
  }
  return { beta, theta, objective: obj, weights: w, converged, message: msg, trace };
}

/** σ̂ = median of the m − p largest |r_i| / Φ⁻¹(3/4), r the residuals of an L1 fit. */
function l1ResidualScale(r: readonly number[], nParams: number): number {
  let ar = sortAsc(r.map(Math.abs));
  if (ar.length > nParams) ar = ar.slice(nParams);
  return median(ar) / MAD_TO_SIGMA;
}

const huberRegression: MethodFn<RegressionProblem> = (
  problem,
  { delta = 1.345, tol = 1e-10, max_iter = 100 },
) => {
  const dl = Number(delta),
    tl = Number(tol),
    maxIter = Math.trunc(Number(max_iter));
  if (!(dl > 0.0)) throw new Error('delta must be > 0');
  const data = resolve(problem, 2);
  const d = lineDesign(data);
  const start = olsStart(d, data.y);
  if (start === null) return startFailed('huber_regression', data, d);
  let [theta] = start;
  const r = start[1];
  let beta = designBeta(d, theta);
  // Residuals below ~16 ulps of the data are rounding noise, not a scale.
  const noise = 16.0 * EPS * Math.max(maxAbs(data.y), TINY);
  const rmsR = rms(r);
  if (rmsR <= noise) {
    const ones = data.y.map(() => 1.0);
    const stats = goodness(data, beta, 2, null);
    Object.assign(stats, {
      weights: ones,
      scale: 0.0,
      delta: dl,
      objective: 0.0,
      x_center: d.center,
    });
    return result(
      'huber_regression',
      beta,
      0.0,
      true,
      'the least-squares line fits every point exactly',
      [step(0, beta, 0.0, { weights: ones, scale: 0.0, delta: dl })],
      stats,
    );
  }
  const l1 = ladIrls(d, data.y, theta, PRELIM_LAD_EPS * rmsR, PRELIM_LAD_TOL, PRELIM_LAD_MAX_ITER);
  const fitL1 = designApply(d, l1.theta);
  const rL1 = data.y.map((v, i) => v - fitL1[i]);
  let scale = allFinite(rL1) ? l1ResidualScale(rL1, 2) : NaN;
  if (Number.isFinite(scale) && scale <= noise) scale = rmsR;
  if (!(Number.isFinite(scale) && Number.isFinite(rmsR)))
    return fail(
      'huber_regression',
      beta,
      `the residual scale σ̂ = ${pyG(scale, 3)} is not finite (overflow)`,
      [step(0, beta, null)],
    );
  const state = (t: readonly number[]): [number, Vector] => {
    const fit = designApply(d, t);
    const u = data.y.map((v, i) => (v - fit[i]) / scale);
    return [npSum(u.map((ui) => huberRho(ui, dl))), u.map((ui) => huberWeight(ui, dl))];
  };
  let [obj, w] = state(theta);
  const info = (wt: Vector): StepInfo => ({ weights: wt, scale, delta: dl });
  if (!Number.isFinite(obj))
    return fail(
      'huber_regression',
      beta,
      'non-finite objective at the least-squares start (overflow)',
      [step(0, beta, null, info(w))],
    );
  const trace = [step(0, beta, obj, info(w))];
  let converged = false;
  let msg = `reached max_iter=${maxIter}`;
  for (let k = 1; k <= maxIter; k++) {
    const next = wls(d.mat, data.y, w);
    if (next === null) {
      msg = 'weighted design matrix is rank deficient or not finite';
      break;
    }
    const ch = change(next, theta);
    theta = next;
    beta = designBeta(d, theta);
    [obj, w] = state(theta);
    trace.push(step(k, beta, obj, info(w), ch));
    if (!(allFinite(beta) && Number.isFinite(obj))) {
      msg = 'non-finite coefficients or objective';
      break;
    }
    if (ch <= tl * (1.0 + maxAbs(theta))) {
      converged = true;
      msg = `‖Δθ‖∞ = ${pyG(ch, 3)} ≤ tol·(1 + ‖θ‖∞)`;
      break;
    }
  }
  const stats = goodness(data, beta, 2, null);
  Object.assign(stats, { weights: w, scale, delta: dl, objective: obj, x_center: d.center });
  return result('huber_regression', beta, obj, converged, msg, trace, stats);
};

const ladRegression: MethodFn<RegressionProblem> = (
  problem,
  { eps = 1e-6, tol = 1e-10, max_iter = 500 },
) => {
  const e = Number(eps);
  if (!(e > 0.0)) throw new Error('eps must be > 0');
  const data = resolve(problem, 2);
  const d = lineDesign(data);
  const start = olsStart(d, data.y);
  if (start === null) return startFailed('lad_regression', data, d);
  const run = ladIrls(d, data.y, start[0], e, Number(tol), Math.trunc(Number(max_iter)));
  const stats = goodness(data, run.beta, 2, null);
  Object.assign(stats, {
    weights: run.weights,
    eps: e,
    objective: run.objective,
    x_center: d.center,
  });
  return result(
    'lad_regression',
    run.beta,
    run.objective,
    run.converged,
    run.message,
    run.trace,
    stats,
  );
};

// ---------------------------------------------------------------------------------------
// Theil–Sen
// ---------------------------------------------------------------------------------------

const theilSen: MethodFn<RegressionProblem> = (problem) => {
  const data = resolve(problem, 2);
  const raw: number[] = [];
  for (let i = 0; i < data.m; i++)
    for (let j = i + 1; j < data.m; j++) {
      const dx = data.x[j] - data.x[i];
      if (dx !== 0.0) raw.push((data.y[j] - data.y[i]) / dx);
    }
  if (raw.length === 0) throw new Error('all x values are equal: no slope is defined');
  const slopes = sortAsc(raw);
  const slope = median(slopes);
  const intercept = median(data.y.map((v, i) => v - slope * data.x[i]));
  const beta = [intercept, slope];
  if (!allFinite(beta))
    return fail(
      'theil_sen',
      beta,
      `non-finite line (slope ${pyG(slope, 3)}, intercept ${pyG(intercept, 3)}): the pairwise ` +
        'slopes overflow the floating-point range',
      [step(0, beta, null, { slopes })],
    );
  const stats = goodness(data, beta, 2, null);
  Object.assign(stats, { n_pairs: slopes.length, slopes });
  const rss = stats.rss as number;
  return result(
    'theil_sen',
    beta,
    rss,
    true,
    `median of ${slopes.length} pairwise slopes`,
    [step(0, beta, rss, { slopes })],
    stats,
  );
};

// ---------------------------------------------------------------------------------------
// Chebyshev (minimax) line
// ---------------------------------------------------------------------------------------

/** Solve β₀ + β₁x_{r_j} + (−1)^j h = y_{r_j}, j = 0, 1, 2 → [β₀, β₁, h]. */
function levelledLine(x: readonly number[], y: readonly number[], ref: readonly number[]) {
  const a = ref.map((r, j) => [1.0, x[r], j % 2 === 0 ? 1.0 : -1.0]);
  const sol = solveDense(
    a,
    ref.map((r) => y[r]),
  );
  return sol !== null && allFinite(sol) ? sol : null;
}

/** Single-point exchange (Stiefel 1959) keeping the residual signs alternating. */
export function exchange(
  ref: readonly number[],
  xs: readonly number[],
  k: number,
  signK: number,
  h: number,
): number[] {
  const sg = h >= 0.0 ? 1.0 : -1.0;
  const sigma = [sg, -sg, sg];
  const [p0, p1, p2] = ref;
  const xk = xs[k];
  if (xk < xs[p0]) return signK === sigma[0] ? [k, p1, p2] : [k, p0, p1];
  if (xk < xs[p1]) return signK === sigma[0] ? [k, p1, p2] : [p0, k, p2];
  if (xk < xs[p2]) return signK === sigma[1] ? [p0, k, p2] : [p0, p1, k];
  return signK === sigma[2] ? [p0, p1, k] : [p1, p2, k];
}

const chebyshevMinimaxLine: MethodFn<RegressionProblem> = (
  problem,
  { tol = 1e-12, max_iter = 100 },
) => {
  const tl = Number(tol),
    maxIter = Math.trunc(Number(max_iter));
  const data = resolve(problem, 3);
  const order = data.x.map((_, i) => i).sort((i, j) => data.x[i] - data.x[j] || i - j);
  const xs = order.map((i) => data.x[i]),
    ys = order.map((i) => data.y[i]);
  for (let i = 1; i < xs.length; i++)
    if (xs[i] - xs[i - 1] === 0.0)
      throw new Error('the minimax line needs distinct x values (Haar condition)');
  const m = data.m;
  const yscale = maxAbs(ys);
  let ref = [0, Math.floor((m - 1) / 2), m - 1];
  const trace: Step[] = [];
  let beta = nanVec(2);
  let h = 0.0,
    dev = Infinity,
    lastRef = ref.slice();
  let converged = false;
  let msg = `reached max_iter=${maxIter}`;
  for (let k = 0; k <= maxIter; k++) {
    const sol = levelledLine(xs, ys, ref);
    if (sol === null) {
      msg = 'singular or non-finite (overflow) reference system';
      break;
    }
    const hNew = sol[2];
    if (k > 0 && Math.abs(hNew) <= Math.abs(h)) {
      msg = 'no ascent of the levelled error |h| (rounding): stopped';
      break;
    }
    beta = [sol[0], sol[1]];
    h = hNew;
    lastRef = ref.slice();
    const r = ys.map((v, i) => v - (1.0 * beta[0] + xs[i] * beta[1]));
    let kmax = 0;
    for (let i = 1; i < r.length; i++) if (Math.abs(r[i]) > Math.abs(r[kmax])) kmax = i;
    dev = Math.abs(r[kmax]);
    let fs = 0;
    for (let i = 0; i < m; i++)
      fs = Math.max(
        fs,
        Math.abs(ys[i]) + Math.abs(beta[0]) + Math.abs(beta[1] * xs[i]) + Math.abs(h),
      );
    const rounding = MINIMAX_ROUNDING_EPS * (EPS * fs + ETA);
    const done = dev - Math.abs(h) <= tl * Math.max(Math.abs(h), yscale) + rounding;
    const stop = done || k === maxIter;
    trace.push(
      step(k, beta, dev, {
        reference: ref.map((i) => order[i]),
        level: h,
        entering: stop ? null : order[kmax],
        max_deviation: dev,
      }),
    );
    if (done) {
      converged = true;
      msg = `max|r| = ${pyG(dev, 6)} equals the levelled error |h|`;
      break;
    }
    if (stop) break;
    ref = exchange(ref, xs, kmax, r[kmax] > 0.0 ? 1.0 : -1.0, h);
  }
  if (trace.length === 0) {
    const nanBeta = nanVec(2);
    return fail('chebyshev_minimax_line', nanBeta, msg, [step(0, nanBeta, null, {})]);
  }
  const last = trace[trace.length - 1];
  if (!converged && last.info.entering !== null)
    trace[trace.length - 1] = { ...last, info: { ...last.info, entering: null } };
  const stats = goodness(data, beta, 2, null);
  Object.assign(stats, { reference: lastRef.map((i) => order[i]), level: h, max_deviation: dev });
  return result('chebyshev_minimax_line', beta, dev, converged, msg, trace, stats);
};

// ---------------------------------------------------------------------------------------
// Registration (same ids, params and ranges as the Python @register)
// ---------------------------------------------------------------------------------------

const DOCS: Record<string, MethodDoc> = {
  linear_regression: {
    rule: '\\begin{aligned} \\hat{\\boldsymbol\\beta} &= \\arg\\min_{\\boldsymbol\\beta}\\|\\mathbf{y} - X\\boldsymbol\\beta\\|_2^2 \\\\ &= R^{-1}Q^{\\mathsf T}\\mathbf{y}, \\quad X = QR \\end{aligned}',
    intuition:
      'Ordinary least squares picks the line that makes the sum of the squared vertical sticks as small as possible. Orthogonal factorization (QR) solves it in about log₁₀κ(X) lost digits; the normal equations lose twice as many.',
    order: 'direct',
    pros: [
      'Unique, closed form, unbiased under Gaussian noise',
      'Standard errors and R² come for free',
    ],
    cons: ['One gross outlier can tilt the whole line (breakdown point 0)'],
  },
  polynomial_regression: {
    rule: '\\begin{aligned} \\hat y(x) &= \\textstyle\\sum_{j=0}^{d}\\beta_j x^j \\\\ \\hat{\\boldsymbol\\beta} &= \\arg\\min_{\\boldsymbol\\beta}\\|\\mathbf{y} - V\\boldsymbol\\beta\\|_2^2 \\end{aligned}',
    intuition:
      'The same least-squares fit with the columns 1, x, …, xᵈ. Each extra degree lowers the training error, but past the true degree the curve starts to follow the noise and oscillates between the points.',
    order: 'direct',
    pros: ['Linear in β: one QR solve', 'Captures curvature a line cannot'],
    cons: [
      'The Vandermonde matrix grows ill-conditioned with d',
      'High degree overfits: training error falls while prediction error rises',
    ],
  },
  ridge_regression: {
    rule: '\\begin{aligned} \\hat{\\boldsymbol\\beta}_\\lambda = \\arg\\min_{\\boldsymbol\\beta}\\ & \\|\\mathbf{y} - X\\boldsymbol\\beta\\|_2^2 \\\\ & + \\lambda\\textstyle\\sum_{j\\ge 1}\\beta_j^2 \\end{aligned}',
    intuition:
      'A penalty on the size of the slope coefficients trades a little bias for much less variance. As λ grows the fit flattens toward the mean ȳ; as λ → 0 it becomes least squares.',
    order: 'direct',
    pros: ['Always unique for λ > 0', 'Tames high-degree fits'],
    cons: ['λ has units here (features are not standardized)', 'Biased: shrinks real effects too'],
  },
  huber_regression: {
    rule: '\\begin{aligned} w_i &= \\min\\!\\Big(1, \\frac{\\delta\\hat\\sigma}{|r_i(\\boldsymbol\\beta_k)|}\\Big) \\\\ \\boldsymbol\\beta_{k+1} &= \\arg\\min_{\\boldsymbol\\beta}\\textstyle\\sum_i w_i\\, r_i(\\boldsymbol\\beta)^2 \\end{aligned}',
    intuition:
      'Squared loss inside the band |r| ≤ δσ̂, absolute loss outside: a point far from the line keeps a vote, but its weight falls like 1/|r|. Each reweighted solve minimizes a quadratic majorizer, so the Huber objective decreases at every step.',
    order: 'linear',
    pros: ['Bounded influence of outliers in y', '95 % efficient under Gaussian noise (δ = 1.345)'],
    cons: ['Needs a robust scale σ̂', 'Leverage points (outliers in x) still pull it'],
    quantities: [
      { tex: '\\hat\\sigma', key: 'info.scale' },
      { tex: '\\delta', key: 'info.delta' },
    ],
  },
  lad_regression: {
    rule: '\\begin{aligned} w_i &= \\frac{1}{\\max(|r_i(\\boldsymbol\\beta_k)|, \\varepsilon)} \\\\ \\boldsymbol\\beta_{k+1} &= \\arg\\min_{\\boldsymbol\\beta}\\textstyle\\sum_i w_i\\, r_i(\\boldsymbol\\beta)^2 \\end{aligned}',
    intuition:
      'Least absolute deviations is the regression analogue of the median. Weighting each squared residual by 1/|rᵢ| turns it back into |rᵢ|; at the optimum the line passes through at least two data points.',
    order: 'linear',
    pros: ['Resists outliers in y', 'No scale to estimate'],
    cons: ['IRLS converges slowly near the optimum', 'The solution need not be unique'],
    quantities: [{ tex: 'S_\\varepsilon(\\boldsymbol\\beta_k)', key: 'info.smoothed_objective' }],
  },
  theil_sen: {
    rule: '\\begin{aligned} \\beta_1 &= \\operatorname*{med}_{i<j,\\ x_i\\ne x_j}\\frac{y_j - y_i}{x_j - x_i} \\\\ \\beta_0 &= \\operatorname*{med}_i\\,(y_i - \\beta_1 x_i) \\end{aligned}',
    intuition:
      'Draw the line through every pair of points and take the median slope. Up to about 29 % of the points can be arbitrarily wrong before the estimate breaks down.',
    order: 'direct',
    pros: ['Breakdown point ≈ 29 %', 'No tuning, no iteration'],
    cons: ['O(m²) pairs', 'Lines only'],
  },
  chebyshev_minimax_line: {
    rule: '\\beta_0 + \\beta_1 x_{r_j} + (-1)^j h = y_{r_j},\\quad j = 0, 1, 2',
    intuition:
      'The minimax line equioscillates: its largest error ±h is attained at three points with alternating signs. Each exchange swaps the worst point into the reference, and |h| rises until no point is worse than the reference.',
    order: 'finite (exchange)',
    pros: ['Optimal in the max norm', 'Terminates in a few exchanges'],
    cons: ['Decided by the most extreme points: the opposite of robust'],
    quantities: [
      { tex: 'h_k', key: 'info.level' },
      { tex: '\\max_i |r_i|', key: 'info.max_deviation' },
    ],
  },
};

const NEEDS = ['data'];

registerMethod<RegressionProblem>(
  {
    id: 'linear_regression',
    family: 'regression',
    name: 'Linear regression (OLS)',
    params: [
      param.choice('solver', 'qr', ['qr', 'normal_equations', 'svd'], {
        help: 'QR (stable), normal equations (squares κ), or SVD (rank-revealing).',
        label: 'Solver',
      }),
    ],
    needs: NEEDS,
    order: 'direct',
    summary: 'Fit the line y = β₀ + β₁x that minimizes the sum of squared vertical residuals.',
    references: [
      'Golub & Van Loan, Matrix Computations (4th ed.), §5.3 (Alg. 5.3.1 normal equations, Alg. 5.3.2 Householder LS) and §5.5 (SVD, rank deficiency)',
      'Higham, Accuracy and Stability of Numerical Algorithms (2nd ed.), Ch. 20',
      'Montgomery, Peck & Vining, Introduction to Linear Regression Analysis (5th ed.), Ch. 2–3 (standard errors, R², adjusted R²)',
    ],
  },
  linearRegression,
  DOCS.linear_regression,
);

registerMethod<RegressionProblem>(
  {
    id: 'polynomial_regression',
    family: 'regression',
    name: 'Polynomial regression',
    params: [
      param.int('degree', 2, {
        min: 0,
        max: 15,
        help: 'Polynomial degree d. With fewer than d + 1 distinct x the fit is not unique: the minimum-norm solution is returned with converged=False.',
        label: 'Degree',
        tex: 'd',
      }),
    ],
    needs: NEEDS,
    order: 'direct',
    summary: 'Least-squares fit of a degree-d polynomial; higher d fits the noise.',
    references: [
      'Golub & Van Loan, Matrix Computations (4th ed.), Alg. 5.3.2 (Householder LS)',
      'van der Sluis (1969), Condition numbers and equilibration of matrices, Numer. Math. 14 (column scaling)',
    ],
  },
  polynomialRegression,
  DOCS.polynomial_regression,
);

registerMethod<RegressionProblem>(
  {
    id: 'ridge_regression',
    family: 'regression',
    name: 'Ridge regression',
    params: [
      param.float('lam', 1.0, {
        min: 1e-8,
        max: 1e4,
        log: true,
        help: 'Penalty λ on Σ_{j≥1} β_j² (λ = 0, plain OLS, is also accepted).',
        label: 'Penalty',
        tex: '\\lambda',
      }),
      param.int('degree', 1, {
        min: 0,
        max: 15,
        help: 'Polynomial degree d.',
        label: 'Degree',
        tex: 'd',
      }),
    ],
    needs: NEEDS,
    order: 'direct',
    summary: 'Least squares plus a penalty λ‖β‖² that shrinks the slope coefficients toward 0.',
    references: [
      'Hastie, Tibshirani & Friedman, The Elements of Statistical Learning (2nd ed.), §3.4.1, eq. (3.41) (intercept not penalized)',
      'Golub & Van Loan, Matrix Computations (4th ed.), §6.1 (Tikhonov regularization as an augmented least-squares problem)',
    ],
  },
  ridgeRegression,
  DOCS.ridge_regression,
);

registerMethod<RegressionProblem>(
  {
    id: 'huber_regression',
    family: 'regression',
    name: 'Huber regression (IRLS)',
    params: [
      param.float('delta', 1.345, {
        min: 0.1,
        max: 10.0,
        help: 'Threshold δ in units of σ̂ (1.345 gives 95% efficiency for Gaussian noise).',
        label: 'Threshold',
        tex: '\\delta',
      }),
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-2,
        log: true,
        help: 'Stop when ‖Δθ‖∞ ≤ tol·(1 + ‖θ‖∞), θ = the line in the centered variable x − x̄.',
        label: 'Step tolerance',
        tex: '\\|\\Delta\\boldsymbol\\theta\\|_\\infty \\le',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 10_000,
        help: 'IRLS iteration limit.',
        label: 'Iteration budget',
      }),
    ],
    needs: NEEDS,
    order: 'linear',
    summary:
      'Quadratic loss for small residuals, linear for large ones; solved by reweighted least squares.',
    references: [
      'Huber (1964), Robust estimation of a location parameter, Ann. Math. Stat. 35',
      'Holland & Welsch (1977), Robust regression using iteratively reweighted least-squares, Commun. Stat. A6',
      'Maronna, Martin & Yohai, Robust Statistics (2006), Ch. 4 (regression M-estimates with a preliminary scale from an L1 fit; IRWLS and its monotone descent)',
    ],
  },
  huberRegression,
  DOCS.huber_regression,
);

registerMethod<RegressionProblem>(
  {
    id: 'lad_regression',
    family: 'regression',
    name: 'Least absolute deviations (IRLS)',
    params: [
      param.float('eps', 1e-6, {
        min: 1e-12,
        max: 1e-1,
        log: true,
        help: 'Weight floor: w_i = 1/max(|r_i|, eps) (in units of y).',
        label: 'Weight floor',
        tex: '\\varepsilon',
      }),
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-2,
        log: true,
        help: 'Stop when ‖Δθ‖∞ ≤ tol·(1 + ‖θ‖∞), θ = the line in the centered variable x − x̄.',
        label: 'Step tolerance',
        tex: '\\|\\Delta\\boldsymbol\\theta\\|_\\infty \\le',
      }),
      param.int('max_iter', 500, {
        min: 1,
        max: 10_000,
        help: 'IRLS iteration limit.',
        label: 'Iteration budget',
      }),
    ],
    needs: NEEDS,
    order: 'linear',
    summary: 'Minimize the sum of absolute residuals by repeatedly solving weighted least squares.',
    references: [
      'Schlossmacher (1973), An iterative technique for absolute deviations curve fitting, JASA 68',
      'Björck, Numerical Methods for Least Squares Problems (1996), Ch. 4 (IRLS for ℓ_p)',
    ],
  },
  ladRegression,
  DOCS.lad_regression,
);

registerMethod<RegressionProblem>(
  {
    id: 'theil_sen',
    family: 'regression',
    name: 'Theil–Sen estimator',
    params: [],
    needs: NEEDS,
    order: 'direct, O(m² log m)',
    summary: 'The slope is the median of the slopes through all pairs of points.',
    references: [
      'Theil (1950), A rank-invariant method of linear and polynomial regression analysis, Indag. Math. 12',
      "Sen (1968), Estimates of the regression coefficient based on Kendall's tau, JASA 63",
    ],
  },
  theilSen,
  DOCS.theil_sen,
);

registerMethod<RegressionProblem>(
  {
    id: 'chebyshev_minimax_line',
    family: 'regression',
    name: 'Minimax (Chebyshev) line',
    params: [
      param.float('tol', 1e-12, {
        min: 1e-15,
        max: 1e-3,
        log: true,
        help: 'Stop when max|r| − |h| ≤ tol·max(|h|, ‖y‖∞) + the rounding level of r.',
        label: 'Gap tolerance',
      }),
      param.int('max_iter', 100, {
        min: 1,
        max: 10_000,
        help: 'Exchange limit.',
        label: 'Exchange budget',
      }),
    ],
    needs: NEEDS,
    order: 'finite (exchange)',
    summary:
      'The line that minimizes the largest vertical error; found by exchanging 3 reference points.',
    references: [
      'Stiefel (1959), Über diskrete und lineare Tschebyscheff-Approximationen, Numer. Math. 1 (exchange algorithm)',
      'Cheney, Introduction to Approximation Theory (1966), Ch. 2 (alternation theorem, de la Vallée Poussin ascent)',
    ],
  },
  chebyshevMinimaxLine,
  DOCS.chebyshev_minimax_line,
);
