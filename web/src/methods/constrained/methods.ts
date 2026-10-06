/**
 * Constrained nonlinear optimization — TS port of `numopt.constrained.methods`
 * (src/numopt/constrained/methods.py): min f(x) subject to c_i(x) ≤ 0 (i ∈ I), c_i(x) = 0 (i ∈ E).
 *
 * Conventions (identical to Python, see the module docstring there):
 *
 *   - Lagrangian L(x, λ) = f(x) + Σ λ_i c_i(x), λ_i ≥ 0 for inequalities.
 *   - KKT residual (∞-norm): stationarity ‖∇f + Σ λ_i ∇c_i‖∞, complementarity max_I |λ_i c_i|,
 *     dual feasibility max_I max(0, −λ_i); violation max(max_I max(0, c_i), max_E |c_i|).
 *     projected_gradient reports ‖x − P(x − ∇f(x))‖∞ and frank_wolfe the gap ∇f(x)ᵀ(x − s).
 *   - Converged when the KKT residual and the violation are both ≤ tol at an iterate.
 *   - The outer and inner loops are flattened into one trace: Step k is the k-th accepted iterate,
 *     `max_iter` bounds the total number of steps.
 *   - Counts: nFev, nGev, nHev exactly as `Counted`; `extra` adds `n_cev`, `n_cgev`, `n_chev`,
 *     `kkt_residual`, `violation` and the final `multipliers` (snake_case, as exported).
 *
 * Step.info keys are the Python ones (snake_case): outer, inner, constraints, active, violation,
 * kkt_residual, and per method from, gradient, unprojected, s, arc, trials, projected_from /
 * vertex, direction, gamma, lmo, gap / mu, tau, merit, multipliers, stationarity, alpha /
 * lambda, omega, eta, update / t, newton_decrement, hessian_shift / working_set, theta.
 * Python `None` is `null`; NaN stays NaN (the exporter writes it as JSON null).
 *
 * Floating-point operations keep the Python order (`x + alpha * p`, `fx - fz >= sigma * pred`,
 * ...). `np.linalg.solve` is an LU with partial pivoting (LAPACK getrf/getrs), `np.linalg.qr` a
 * Householder QR and `np.linalg.lstsq` a minimum-norm SVD least-squares solve (rcond = ε·max(m, n));
 * the ports below do the same, so traces agree with the Python fixtures to rounding.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { cholesky } from '../../core/linalg';
import type {
  Constraint,
  Matrix,
  MethodFn,
  Params,
  Problem,
  Result,
  RunOptions,
  Step,
  StepInfo,
  Vector,
} from '../../core/types';

// ---------------------------------------------------------------------------------------
// Constants (as in Python)
// ---------------------------------------------------------------------------------------

/** An inequality with c_i(x) ≥ −ACTIVE_TOL is reported as active in `info.active`. */
export const ACTIVE_TOL = 1e-6;
/** Armijo sufficient-decrease constant c₁ of the inner line searches. */
const C1 = 1e-4;
/** Backtracking factor of the inner line searches. */
const SHRINK = 0.5;
/** Maximum number of trial steps of one line search. */
const MAX_TRIALS = 60;
/** Smallest trial step of projected_gradient relative to s̄. */
const MIN_TRIAL_RATIO = 1e-18;
/** Safety cap on outer iterations that take no inner step. */
const MAX_OUTER = 200;
/** Relative test for linear dependence of a new constraint normal in `solveQp`. */
const QP_DEP_TOL = 1e-10;
/** Relative feasibility tolerance of `solveQp`. */
const QP_FEAS_TOL = 1e-12;
/** Safety factor on the first-order rounding bound of a dependent constraint's residual. */
const QP_ROUND_FACTOR = 10.0;
/** Machine epsilon of float64. */
export const EPS = Number.EPSILON;
/** Relative resolution of a merit value. */
const RESOLUTION = 100.0 * EPS;
/** Backtracking constants of B&V Alg. 9.2. */
const BARRIER_ALPHA = 0.01;
const BARRIER_BETA = 0.5;
/** N&W Alg. 3.3: the first shift added when the Cholesky factorization fails. */
const SHIFT_BETA = 1e-3;
/** N&W Alg. 18.3 constants. */
const SQP_ETA = 1e-4;
const SQP_TAU = 0.5;
const SQP_RHO = 0.5;

/** Python `ValueError` (invalid input). */
export class ConstrainedInputError extends Error {
  override name = 'ValueError';
}

class ProjectionError extends Error {
  override name = 'ProjectionError';
}

// ---------------------------------------------------------------------------------------
// Python-compatible scalar helpers and formatting
// ---------------------------------------------------------------------------------------

/** Python's builtin `max(a, b, ...)` (keeps the first argument when a comparison is NaN). */
function pyMax(...xs: number[]): number {
  let m = xs[0];
  for (let i = 1; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  return m;
}

/** Python's builtin `min(a, b)`. */
function pyMin(a: number, b: number): number {
  return b < a ? b : a;
}

/** `np.max` of a non-empty array (NaN propagates). */
function npMax(a: readonly number[]): number {
  let m = -Infinity;
  for (const v of a) {
    if (Number.isNaN(v)) return NaN;
    if (v > m) m = v;
  }
  return m;
}

/** `np.sum` of a short array (< 8 entries: a plain loop, as NumPy's pairwise sum does). */
function npSum(a: readonly number[]): number {
  let s = 0;
  for (const v of a) s += v;
  return s;
}

function resolution(phi: number): number {
  return RESOLUTION * pyMax(1.0, Math.abs(phi));
}

/** Armijo's test with gradients only (approximate Wolfe, Hager & Zhang 2005). False for NaN. */
function approxArmijo(slopeNew: number, pred: number, c1: number): boolean {
  return slopeNew <= (1.0 - 2.0 * c1) * pred;
}

/** Python `format(v, '.{p}g')`. */
export function pyG(v: number, p = 3): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, e] = v.toExponential(p - 1).split('e');
  const exp = Number(e);
  const strip = (s: string) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s);
  if (exp < -4 || exp >= p)
    return `${strip(mant)}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  return strip(v.toFixed(Math.max(0, p - 1 - exp)));
}

/** Python `repr(float(v))`. */
export function pyRepr(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0.0' : '0.0';
  const [mant, e] = v.toExponential().split('e');
  const exp = Number(e);
  if (exp < -4 || exp >= 16)
    return `${mant}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  const s = String(v);
  return s.includes('.') ? s : `${s}.0`;
}

/** Python `repr(list_of_floats)`. */
function pyList(x: readonly number[]): string {
  return `[${x.map(pyRepr).join(', ')}]`;
}

// ---------------------------------------------------------------------------------------
// Small dense linear algebra (NumPy / LAPACK semantics)
// ---------------------------------------------------------------------------------------

const zerosV = (n: number): Vector => new Array<number>(n).fill(0);
const eye = (n: number): Matrix =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
const addV = (a: readonly number[], b: readonly number[]): Vector => a.map((v, i) => v + b[i]);
const subV = (a: readonly number[], b: readonly number[]): Vector => a.map((v, i) => v - b[i]);
const scaleV = (s: number, a: readonly number[]): Vector => a.map((v) => s * v);
const negV = (a: readonly number[]): Vector => a.map((v) => -v);
/** `x + alpha * p` in the Python order. */
const axpyP = (x: readonly number[], alpha: number, p: readonly number[]): Vector =>
  x.map((v, i) => v + alpha * p[i]);
const norm2 = (a: readonly number[]): number => Math.sqrt(dot(a, a));
const normInfV = (a: readonly number[]): number => npMax(a.map(Math.abs));
const matvec = (A: Matrix, x: readonly number[]): Vector => A.map((row) => dot(row, x));
/** Aᵀ y for A (m × n) — Σ_i y_i A[i]; zeros(n) when m = 0. */
function matTvec(A: Matrix, y: readonly number[], n: number): Vector {
  const out = zerosV(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < A.length; i++) s += A[i][j] * y[i];
    out[j] = s;
  }
  return out;
}
function matmul(A: Matrix, B: Matrix): Matrix {
  const n = A.length,
    p = B.length,
    m = B[0]?.length ?? 0;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let s = 0;
      for (let k = 0; k < p; k++) s += A[i][k] * B[k][j];
      return s;
    }),
  );
}
const transpose = (A: Matrix, cols?: number): Matrix => {
  const m = cols ?? A[0]?.length ?? 0;
  return Array.from({ length: m }, (_, j) => A.map((row) => row[j]));
};
const outerV = (a: readonly number[], b: readonly number[]): Matrix =>
  a.map((ai) => b.map((bj) => ai * bj));
const symmetrize = (A: Matrix): Matrix => A.map((row, i) => row.map((v, j) => 0.5 * (v + A[j][i])));
const allFinite = (...vals: (number | readonly number[] | Matrix)[]): boolean =>
  vals.every((v) =>
    typeof v === 'number'
      ? Number.isFinite(v)
      : (v as readonly unknown[]).every((r) =>
          typeof r === 'number' ? Number.isFinite(r) : (r as number[]).every(Number.isFinite),
        ),
  );
const arraysEqual = (a: readonly number[], b: readonly number[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * `np.linalg.solve(A, B)`: LU with partial pivoting (LAPACK getrf/getrs). Returns null when a
 * pivot is exactly zero (NumPy raises LinAlgError: "Singular matrix"). `B` is one right-hand
 * side (a vector) or several (an array of vectors, solved with the same factorization).
 */
function luSolveMany(A: Matrix, rhs: readonly (readonly number[])[]): Vector[] | null {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let j = 0; j < n; j++) {
    let p = j,
      best = Math.abs(a[j][j]);
    for (let i = j + 1; i < n; i++) {
      const v = Math.abs(a[i][j]);
      if (v > best || Number.isNaN(v)) {
        best = v;
        p = i;
      }
    }
    if (a[p][j] === 0) return null;
    if (p !== j) {
      [a[p], a[j]] = [a[j], a[p]];
      [perm[p], perm[j]] = [perm[j], perm[p]];
    }
    const piv = a[j][j];
    for (let i = j + 1; i < n; i++) {
      const l = a[i][j] / piv;
      a[i][j] = l;
      if (l !== 0) for (let k = j + 1; k < n; k++) a[i][k] -= l * a[j][k];
    }
  }
  return rhs.map((b) => {
    const y = perm.map((pi) => b[pi]);
    for (let i = 0; i < n; i++) for (let k = 0; k < i; k++) y[i] -= a[i][k] * y[k];
    for (let i = n - 1; i >= 0; i--) {
      for (let k = i + 1; k < n; k++) y[i] -= a[i][k] * y[k];
      y[i] /= a[i][i];
    }
    return y;
  });
}

function luSolve(A: Matrix, b: readonly number[]): Vector | null {
  const r = luSolveMany(A, [b]);
  return r ? r[0] : null;
}

/** Reduced Householder QR of B (n × q, given as q columns): Q (q columns) and R (q × q). */
function qrColumns(cols: readonly (readonly number[])[]): { Q: Vector[]; R: Matrix } {
  const q = cols.length;
  const n = cols[0]?.length ?? 0;
  // Work on A as rows n × q.
  const A: Matrix = Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
  const vs: Vector[] = [];
  const taus: number[] = [];
  for (let j = 0; j < q; j++) {
    const alpha = A[j][j];
    let xnorm = 0;
    for (let i = j + 1; i < n; i++) xnorm = Math.hypot(xnorm, A[i][j]);
    const v = zerosV(n);
    if (xnorm === 0) {
      taus.push(0);
      vs.push(v);
      continue;
    }
    const beta = -Math.sign(alpha || 1) * Math.hypot(alpha, xnorm);
    const tau = (beta - alpha) / beta;
    const scal = 1 / (alpha - beta);
    v[j] = 1;
    for (let i = j + 1; i < n; i++) v[i] = A[i][j] * scal;
    A[j][j] = beta;
    for (let i = j + 1; i < n; i++) A[i][j] = 0;
    for (let k = j + 1; k < q; k++) {
      let s = 0;
      for (let i = j; i < n; i++) s += v[i] * A[i][k];
      s *= tau;
      for (let i = j; i < n; i++) A[i][k] -= s * v[i];
    }
    taus.push(tau);
    vs.push(v);
  }
  const R: Matrix = Array.from({ length: q }, (_, i) =>
    Array.from({ length: q }, (_, k) => (k >= i ? A[i][k] : 0)),
  );
  // Q = H₁ ⋯ H_q applied to the first q columns of the identity.
  const Q: Vector[] = [];
  for (let k = 0; k < q; k++) {
    const e = zerosV(n);
    e[k] = 1;
    for (let j = q - 1; j >= 0; j--) {
      const v = vs[j];
      let s = 0;
      for (let i = j; i < n; i++) s += v[i] * e[i];
      s *= taus[j];
      for (let i = j; i < n; i++) e[i] -= s * v[i];
    }
    Q.push(e);
  }
  return { Q, R };
}

/**
 * `np.linalg.lstsq(A, b, rcond=None)[0]`: the minimum-norm least-squares solution, from a
 * one-sided Jacobi SVD; singular values ≤ ε·max(m, n)·σ_max are treated as zero.
 */
export function lstsq(A: Matrix, b: readonly number[]): Vector {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  if (m === 0 || n === 0) return zerosV(n);
  // SVD of M = A (m ≥ n) or Aᵀ: columns U·σ, rotations V.
  const tall = m >= n;
  const M = tall ? A.map((r) => r.slice()) : transpose(A, n);
  const rows = M.length,
    cols = M[0].length;
  const V = eye(cols);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < cols - 1; p++)
      for (let q = p + 1; q < cols; q++) {
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let i = 0; i < rows; i++) {
          alpha += M[i][p] * M[i][p];
          beta += M[i][q] * M[i][q];
          gamma += M[i][p] * M[i][q];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta));
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t),
          s = c * t;
        for (let i = 0; i < rows; i++) {
          const mp = M[i][p],
            mq = M[i][q];
          M[i][p] = c * mp - s * mq;
          M[i][q] = s * mp + c * mq;
        }
        for (let i = 0; i < cols; i++) {
          const vp = V[i][p],
            vq = V[i][q];
          V[i][p] = c * vp - s * vq;
          V[i][q] = s * vp + c * vq;
        }
      }
    if (off <= EPS) break;
  }
  const sig = Array.from({ length: cols }, (_, j) => norm2(M.map((r) => r[j])));
  const cutoff = EPS * Math.max(m, n) * Math.max(0, ...sig);
  // M = U Σ (columns), so the matrix is U Σ Vᵀ.
  // tall: A = U Σ Vᵀ ⇒ x = V Σ⁺ Uᵀ b.   wide: Aᵀ = U Σ Vᵀ ⇒ A = V Σ Uᵀ ⇒ x = U Σ⁺ Vᵀ b.
  const x = zerosV(n);
  for (let j = 0; j < cols; j++) {
    if (!(sig[j] > cutoff)) continue;
    const u = M.map((r) => r[j] / sig[j]);
    if (tall) {
      const coef = dot(u, b) / sig[j];
      for (let i = 0; i < n; i++) x[i] += coef * V[i][j];
    } else {
      const vj = V.map((r) => r[j]);
      const coef = dot(vj, b) / sig[j];
      for (let i = 0; i < n; i++) x[i] += coef * u[i];
    }
  }
  return x;
}

// ---------------------------------------------------------------------------------------
// Finite-difference fallbacks (numopt.core.diff)
// ---------------------------------------------------------------------------------------

const H_CENTRAL = Math.cbrt(EPS);

function fdGradient(f: (x: Vector) => number, x: Vector): Vector {
  return x.map((xi, i) => {
    const h = H_CENTRAL * pyMax(1.0, Math.abs(xi));
    const e = zerosV(x.length);
    e[i] = h;
    return (f(addV(x, e)) - f(subV(x, e))) / (2.0 * h);
  });
}

function fdHessian(grad: (x: Vector) => Vector, x: Vector): Matrix {
  const n = x.length;
  const H: Matrix = Array.from({ length: n }, () => zerosV(n));
  for (let i = 0; i < n; i++) {
    const h = H_CENTRAL * pyMax(1.0, Math.abs(x[i]));
    const e = zerosV(n);
    e[i] = h;
    const gp = grad(addV(x, e)),
      gm = grad(subV(x, e));
    for (let r = 0; r < n; r++) H[r][i] = (gp[r] - gm[r]) / (2.0 * h);
  }
  return symmetrize(H);
}

// ---------------------------------------------------------------------------------------
// Problem resolution and counted evaluations
// ---------------------------------------------------------------------------------------

/** What the methods read from a problem (the shape of `ConstrainedProblem`). */
export interface ConstrainedLike {
  id: string;
  dim: number;
  f: (x: Vector) => number;
  grad?: (x: Vector) => Vector;
  hess?: (x: Vector) => Matrix;
  x0?: Vector | null;
  constraints?: Constraint[];
  extra?: Record<string, unknown>;
}

function vectorProblem(problem: unknown, x0: unknown): ConstrainedLike {
  if (typeof problem === 'function') {
    const dim = Array.isArray(x0) ? x0.length : 2;
    return { id: 'custom', dim, f: problem as (x: Vector) => number, constraints: [], extra: {} };
  }
  if (problem && typeof problem === 'object' && 'f' in problem) return problem as ConstrainedLike;
  throw new TypeError('problem must be a numopt Problem or a callable f(x)');
}

function startPoint(prob: ConstrainedLike, x0: unknown): Vector {
  const v = x0 ?? prob.x0;
  if (v === null || v === undefined)
    throw new ConstrainedInputError(
      `${prob.id}: no starting point given and the problem has no default x0`,
    );
  const x = (Array.isArray(v) ? (v as unknown[]).flat(Infinity) : [v]).map(Number);
  if (prob.dim && x.length !== prob.dim)
    throw new ConstrainedInputError(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  return x;
}

class Model {
  readonly problem: ConstrainedLike;
  readonly n: number;
  readonly cons: Constraint[];
  readonly m: number;
  readonly isEq: boolean[];
  readonly isIn: boolean[];
  readonly affine: boolean[];
  private readonly chessFns: ((x: Vector) => Matrix)[] | null;
  nf = 0;
  ng = 0;
  nh = 0;
  nCev = 0;
  nCgev = 0;
  nChev = 0;

  constructor(problem: ConstrainedLike) {
    this.problem = problem;
    this.n = problem.dim;
    this.cons = [...(problem.constraints ?? [])];
    this.m = this.cons.length;
    this.isEq = this.cons.map((c) => c.kind === 'eq');
    this.isIn = this.isEq.map((e) => !e);
    const extra = problem.extra ?? {};
    const affine = extra.affine as boolean[] | undefined;
    this.affine =
      affine && affine.length === this.m ? affine.map(Boolean) : new Array(this.m).fill(false);
    const hess = extra.constraint_hess as ((x: Vector) => Matrix)[] | undefined;
    this.chessFns = hess && hess.length === this.m ? hess : null;
  }

  fun(x: Vector): number {
    this.nf++;
    return Number(this.problem.f(x));
  }

  grad(x: Vector): Vector {
    if (!this.problem.grad) return fdGradient((z) => this.fun(z), x);
    this.ng++;
    return this.problem.grad(x).slice();
  }

  hess(x: Vector): Matrix {
    if (!this.problem.hess) return fdHessian((z) => this.grad(z), x);
    this.nh++;
    return this.problem.hess(x).map((r) => r.slice());
  }

  cval(x: Vector): Vector {
    this.nCev += this.m;
    return this.cons.map((c) => Number(c.fun(x)));
  }

  cjac(x: Vector): Matrix {
    this.nCgev += this.m;
    return this.cons.map((c) => c.grad(x).slice());
  }

  chess(i: number, x: Vector): Matrix {
    this.nChev++;
    if (this.chessFns) return this.chessFns[i](x).map((r) => r.slice());
    const gradI = this.cons[i].grad;
    return fdHessian((z) => {
      this.nCgev++;
      return gradI(z).slice();
    }, x);
  }

  counts(): [number, number, number] {
    return [this.nf, this.problem.grad ? this.ng : 0, this.problem.hess ? this.nh : 0];
  }
}

const pick = <T>(a: readonly T[], mask: readonly boolean[]): T[] => a.filter((_, i) => mask[i]);

function violationOf(c: readonly number[], isEq: readonly boolean[]): number {
  if (c.length === 0) return 0.0;
  return npMax(c.map((v, i) => (isEq[i] ? Math.abs(v) : Math.max(v, 0.0))));
}

function activeOf(c: readonly number[], isEq: readonly boolean[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < c.length; i++) if (isEq[i] || c[i] >= -ACTIVE_TOL) out.push(i);
  return out;
}

/** (stationarity, kkt_residual) of (x, λ). */
function kktOf(
  gf: Vector,
  J: Matrix,
  lam: readonly number[],
  c: readonly number[],
  isEq: readonly boolean[],
): [number, number] {
  const r = lam.length ? addV(gf, matTvec(J, lam, gf.length)) : gf;
  const stat = r.length ? npMax(r.map(Math.abs)) : 0.0;
  const ineq = isEq.map((e) => !e);
  if (!ineq.some(Boolean)) return [stat, stat];
  const li = pick(lam, ineq),
    ci = pick(c, ineq);
  const comp = npMax(li.map((l, i) => Math.abs(l * ci[i])));
  const dual = npMax(li.map((l) => Math.max(-l, 0.0)));
  return [stat, pyMax(stat, comp, dual)];
}

function state(
  c: readonly number[],
  isEq: readonly boolean[],
  outer: number,
  inner: number,
  kkt: number,
): StepInfo {
  return {
    outer,
    inner,
    constraints: [...c],
    active: activeOf(c, isEq),
    violation: violationOf(c, isEq),
    kkt_residual: kkt,
  };
}

function step(
  k: number,
  x: Vector,
  fun: number,
  gradNorm: number | null,
  stepSize: number | null,
  info: StepInfo,
): Step {
  return { k, x: x.slice(), fun, gradNorm, stepSize, info };
}

function finish(
  method: string,
  model: Model,
  x: Vector,
  fx: number,
  converged: boolean,
  message: string,
  trace: Step[],
  kkt: number,
  violation: number,
  multipliers?: readonly number[],
): Result {
  const [nFev, nGev, nHev] = model.counts();
  const extra: Record<string, unknown> = {
    n_cev: model.nCev,
    n_cgev: model.nCgev,
    n_chev: model.nChev,
    kkt_residual: kkt,
    violation,
  };
  if (multipliers !== undefined) extra.multipliers = [...multipliers];
  return {
    method,
    x: x.slice(),
    fun: fx,
    converged,
    message,
    nIter: trace[trace.length - 1].k,
    nFev,
    nGev,
    nHev,
    trace,
    extra,
  };
}

const convergedMsg = (kkt: number, viol: number, tol: number) =>
  `KKT residual ${pyG(kkt)} ≤ tol and violation ${pyG(viol)} ≤ tol = ${pyG(tol)}`;

const nonfiniteMsg = (x: Vector) =>
  `non-finite function, constraint or derivative value at x = ${pyList(x)}`;

function excess(kkt: number, viol: number, tol: number): string {
  const failing = (
    [
      ['KKT residual', kkt],
      ['violation', viol],
    ] as const
  )
    .filter(([, v]) => !(v <= tol))
    .map(([name, v]) => `${name} ${pyG(v)}`);
  if (!failing.length) return `KKT residual ${pyG(kkt)}, violation ${pyG(viol)}, tol = ${pyG(tol)}`;
  return failing.join(' and ') + ` > tol = ${pyG(tol)}`;
}

const stalledMsg = (kkt: number, viol: number, tol: number) =>
  `stalled: the accepted step leaves x unchanged (${excess(kkt, viol, tol)}); the ` +
  'merit function cannot resolve further progress in float64';

const nonfiniteTrialsMsg = (kind: string, s: number, kkt: number) =>
  `${kind}: f is non-finite at the trial points down to step ${pyG(s)} (KKT residual ${pyG(kkt)})`;

const nullTrialMsg = (kkt: number, viol: number, tol: number) =>
  'stalled: backtracking shrank the step until the trial point equals x in float64 ' +
  `(${excess(kkt, viol, tol)}); the decrease of f is below its rounding level`;

const roundingLimitMsg = (nTrials: number, s: number, kkt: number, viol: number, tol: number) =>
  `stalled at the rounding level: after ${nTrials} trials (step down to ${pyG(s)}) ` +
  'the predicted decrease of f is below its resolution and the gradient form of the ' +
  `Armijo test fails (${excess(kkt, viol, tol)})`;

const maxIterMsg = (maxIter: number, kkt: number, viol: number) =>
  `reached max_iter=${maxIter} (KKT residual ${pyG(kkt)}, violation ${pyG(viol)})`;

// ---------------------------------------------------------------------------------------
// Strictly convex QP: Goldfarb–Idnani dual active-set method
// ---------------------------------------------------------------------------------------

/** Solution of min ½xᵀGx + aᵀx s.t. A_eq x = b_eq, A_ub x ≤ b_ub (see `solveQp`). */
export interface QPResult {
  ok: boolean;
  x: Vector;
  lamEq: Vector;
  lamUb: Vector;
  /** Inequality rows in the final active set. */
  active: number[];
  nIter: number;
  message: string;
}

/** Primal step z = H n⁺ and dual step r = N* n⁺ of Goldfarb & Idnani (1983), §3. */
function giDirections(
  L: Matrix,
  N: readonly Vector[],
  nP: Vector,
): { z: Vector; r: Vector; dependent: boolean } {
  const n = nP.length;
  const w = luSolve(L, nP) ?? zerosV(n).fill(NaN);
  let wPerp: Vector;
  let r: Vector;
  if (N.length === 0) {
    wPerp = w;
    r = [];
  } else {
    const B = luSolveMany(L, N) ?? N.map(() => zerosV(n).fill(NaN));
    const { Q, R } = qrColumns(B);
    const qtw = Q.map((qc) => dot(qc, w));
    r = luSolve(R, qtw) ?? qtw.map(() => NaN);
    const qq = zerosV(n);
    for (let j = 0; j < Q.length; j++) for (let i = 0; i < n; i++) qq[i] += Q[j][i] * qtw[j];
    wPerp = subV(w, qq);
  }
  const normW = norm2(w);
  const dependent = normW === 0.0 || norm2(wPerp) <= QP_DEP_TOL * normW;
  const z = dependent ? zerosV(n) : (luSolve(transpose(L), wPerp) ?? zerosV(n).fill(NaN));
  return { z, r, dependent };
}

/** Is a constraint n_pᵀx ≥ b_p with n_p ≈ N r consistent with the active rows Nᵀx = b_N? */
function dependentIsConsistent(
  nP: Vector,
  bP: number,
  N: readonly Vector[],
  bN: readonly number[],
  r: readonly number[],
  x: Vector,
): boolean {
  const q = bN.length;
  const n = x.length;
  const xn = norm2(x);
  const sP = dot(nP, x) - bP;
  const sN = q ? N.map((col, j) => dot(col, x) - bN[j]) : [];
  const delta = sP - dot(r, sN);
  const colNorms = q ? N.map(norm2) : [];
  const nPNorm = norm2(nP);
  const scale =
    Math.abs(bP) +
    nPNorm * xn +
    dot(
      r.map(Math.abs),
      bN.map((b, j) => Math.abs(b) + colNorms[j] * xn),
    );
  let defect: number;
  if (q) {
    const Nr = zerosV(n);
    for (let j = 0; j < q; j++) for (let i = 0; i < n; i++) Nr[i] += N[j][i] * r[j];
    defect = norm2(subV(nP, Nr));
  } else defect = nPNorm;
  const tol =
    QP_FEAS_TOL * (1.0 + Math.abs(bP) + nPNorm * xn) +
    QP_ROUND_FACTOR * (n + q + 2) * EPS * scale +
    defect * xn;
  return pyMin(Math.abs(sP), Math.abs(delta)) <= tol;
}

/**
 * Strictly convex QP by the dual active-set method of Goldfarb & Idnani (1983): min ½xᵀGx + aᵀx
 * s.t. A_eq x = b_eq, A_ub x ≤ b_ub, G symmetric positive definite. Starts from −G⁻¹a, adds the
 * violated constraints (equalities first, then the most violated inequality), drops blocking
 * inequalities (partial steps), and skips a dependent constraint that is consistent up to
 * rounding. Multipliers follow Gx + a + A_eqᵀλ_eq + A_ubᵀλ_ub = 0, λ_ub ≥ 0.
 */
export function solveQp(
  G: Matrix,
  a: readonly number[],
  Aeq: Matrix | null = null,
  beq: readonly number[] | null = null,
  Aub: Matrix | null = null,
  bub: readonly number[] | null = null,
): QPResult {
  const av = [...a];
  const n = av.length;
  const Ae = Aeq ?? [],
    be = beq ? [...beq] : [],
    Ai = Aub ?? [],
    bi = bub ? [...bub] : [];
  const me = be.length,
    mi = bi.length;
  if (Ae.length !== me || Ai.length !== mi)
    throw new ConstrainedInputError(
      'solve_qp: constraint matrices and right-hand sides disagree in size',
    );
  const fail = (x: Vector, it: number, msg: string): QPResult => ({
    ok: false,
    x,
    lamEq: zerosV(me),
    lamUb: zerosV(mi),
    active: [],
    nIter: it,
    message: msg,
  });
  const L = cholesky(G);
  if (!L) return fail(zerosV(n), 0, 'the QP Hessian is not positive definite');
  let x = luSolve(transpose(L), luSolve(L, negV(av)) ?? zerosV(n)) ?? zerosV(n);
  let act: number[] = [];
  let sgn: number[] = [];
  let u: Vector = [];
  let skipped = new Set<number>();
  const rowNormI = Ai.map(norm2);
  const cap = 10 * (n + me + mi) + 10;
  let it = 0;

  const normal = (cid: number, s: number): [Vector, number] =>
    cid < me ? [scaleV(s, Ae[cid]), s * be[cid]] : [negV(Ai[cid - me]), -bi[cid - me]];

  for (;;) {
    // Step 1: choose a violated constraint p (equalities first, in order).
    let p = -1;
    let sp = 1.0;
    const xnorm = norm2(x);
    for (let i = 0; i < me; i++) {
      if (!act.includes(i) && !skipped.has(i)) {
        p = i;
        sp = dot(Ae[i], x) - be[i] > 0.0 ? -1.0 : 1.0;
        break;
      }
    }
    if (p < 0 && mi) {
      const slack = bi.map((b, j) => b - dot(Ai[j], x));
      let worst = Infinity;
      for (let j = 0; j < mi; j++) {
        const cid = me + j;
        const tolJ = QP_FEAS_TOL * (1.0 + Math.abs(bi[j]) + rowNormI[j] * xnorm);
        if (!act.includes(cid) && !skipped.has(cid) && slack[j] < -tolJ && slack[j] < worst) {
          p = cid;
          worst = slack[j];
        }
      }
    }
    if (p < 0) break;
    const [nP, bP] = normal(p, sp);
    let uPlus = [...u, 0.0];
    const saved: [number[], number[], Set<number>] = [[...act], [...sgn], new Set(skipped)];
    let xMoved = false;
    // Step 2: move until constraint p is active (full step) or blocked (partial step).
    for (;;) {
      it += 1;
      if (it > cap) return fail(x, it, `QP iteration limit (${cap}) reached`);
      const pairs = act.map((cid, idx) => normal(cid, sgn[idx]));
      const N = pairs.map(([nv]) => nv);
      const bN = pairs.map(([, bv]) => bv);
      const { z, r, dependent } = giDirections(L, N, nP);
      const sP = dot(nP, x) - bP;
      let t1 = Infinity,
        kDrop = -1;
      act.forEach((cid, idx) => {
        if (cid >= me && r[idx] > 0.0) {
          const ratio = uPlus[idx] / r[idx];
          if (ratio < t1) {
            t1 = ratio;
            kDrop = idx;
          }
        }
      });
      if (dependent && t1 === Infinity && !xMoved && dependentIsConsistent(nP, bP, N, bN, r, x)) {
        // p is implied by the active rows; restore the state before p and skip p.
        act = [...saved[0]];
        sgn = [...saved[1]];
        skipped = new Set(saved[2]);
        skipped.add(p);
        break;
      }
      const t2 = dependent ? Infinity : -sP / dot(z, nP);
      const t = pyMin(t1, t2);
      if (!Number.isFinite(t))
        return fail(x, it, 'the QP constraints are inconsistent (no feasible point)');
      if (t2 === Infinity || t2 === -Infinity) {
        // Dual step only: drop the blocking inequality.
        for (let i = 0; i < r.length; i++) uPlus[i] -= t * r[i];
        uPlus[uPlus.length - 1] += t;
        act.splice(kDrop, 1);
        sgn.splice(kDrop, 1);
        uPlus.splice(kDrop, 1);
        skipped = new Set([...skipped].filter((cid) => cid < me));
        continue;
      }
      x = axpyP(x, t, z);
      xMoved = true;
      for (let i = 0; i < r.length; i++) uPlus[i] -= t * r[i];
      uPlus[uPlus.length - 1] += t;
      if (t2 <= t1) {
        act.push(p);
        sgn.push(sp);
        u = uPlus;
        break;
      }
      act.splice(kDrop, 1); // partial step
      sgn.splice(kDrop, 1);
      uPlus = uPlus.filter((_, i) => i !== kDrop);
      skipped = new Set([...skipped].filter((cid) => cid < me));
    }
  }
  const lamEq = zerosV(me),
    lamUb = zerosV(mi);
  act.forEach((cid, idx) => {
    if (cid < me) lamEq[cid] = -sgn[idx] * u[idx];
    else lamUb[cid - me] = u[idx];
  });
  const active = act
    .filter((cid) => cid >= me)
    .map((cid) => cid - me)
    .sort((p, q) => p - q);
  return { ok: true, x, lamEq, lamUb, active, nIter: it, message: 'optimal' };
}

// ---------------------------------------------------------------------------------------
// Simple feasible sets: projection and linear-minimization oracle
// ---------------------------------------------------------------------------------------

/** Exact Euclidean projection P_C and linear-minimization oracle of a box, disk or polyhedron. */
export class FeasibleSet {
  readonly kind: 'box' | 'disk' | 'polyhedron';
  readonly vertices: Vector[] | null = null;
  readonly lo: Vector = [];
  readonly hi: Vector = [];
  readonly center: Vector = [];
  readonly radius: number = 0;
  readonly Aub: Matrix = [];
  readonly bub: Vector = [];
  readonly Aeq: Matrix = [];
  readonly beq: Vector = [];
  private readonly eye: Matrix = [];

  constructor(problem: ConstrainedLike, method: string, compact = false) {
    const extra = problem.extra ?? {};
    const kind = extra.projection;
    if (kind !== 'box' && kind !== 'disk' && kind !== 'polyhedron')
      throw new ConstrainedInputError(
        `${method} needs a convex feasible set with an exact projection ` +
          "(Problem.extra['projection'] = 'box' | 'disk' | 'polyhedron'); " +
          `problem '${problem.id}' has none`,
      );
    this.kind = kind;
    const n = problem.dim;
    if (extra.vertices) this.vertices = (extra.vertices as number[][]).map((v) => [...v]);
    if (kind === 'box') {
      const bounds = extra.bounds as number[][];
      this.lo = bounds.slice(0, n).map((b) => b[0]);
      this.hi = bounds.slice(0, n).map((b) => b[1]);
    } else if (kind === 'disk') {
      const d = extra.disk as { center: number[]; radius: number };
      this.center = [...d.center];
      this.radius = Number(d.radius);
    } else {
      this.Aub = ((extra.A_ub as number[][] | undefined) ?? []).map((r) => [...r]);
      this.bub = [...((extra.b_ub as number[] | undefined) ?? [])];
      this.Aeq = ((extra.A_eq as number[][] | undefined) ?? []).map((r) => [...r]);
      this.beq = [...((extra.b_eq as number[] | undefined) ?? [])];
      this.eye = eye(n);
      if (compact && this.vertices === null)
        throw new ConstrainedInputError(
          `${method} needs a compact feasible set; problem '${problem.id}' is a ` +
            "polyhedron without extra['vertices'] (it may be unbounded)",
        );
    }
  }

  project(z: Vector): Vector {
    if (this.kind === 'box') return z.map((v, i) => Math.min(Math.max(v, this.lo[i]), this.hi[i]));
    if (this.kind === 'disk') {
      const d = subV(z, this.center);
      const nd = norm2(d);
      if (nd <= this.radius) return z.slice();
      return addV(this.center, scaleV(this.radius / nd, d));
    }
    const qp = solveQp(this.eye, negV(z), this.Aeq, this.beq, this.Aub, this.bub);
    if (!qp.ok) throw new ProjectionError(`projection onto the polyhedron failed: ${qp.message}`);
    return qp.x;
  }

  /** argmin_{s ∈ C} gᵀs (ties: lower bound / first vertex; g = 0 returns x). */
  lmo(g: Vector, x: Vector): Vector {
    if (this.kind === 'box')
      return g.map((gi, i) => (gi > 0.0 ? this.lo[i] : gi < 0.0 ? this.hi[i] : x[i]));
    if (this.kind === 'disk') {
      const ng = norm2(g);
      return ng === 0.0 ? x.slice() : subV(this.center, scaleV(this.radius / ng, g));
    }
    const V = this.vertices!;
    let best = 0,
      bestVal = dot(V[0], g);
    for (let i = 1; i < V.length; i++) {
      const v = dot(V[i], g);
      if (v < bestVal || (Number.isNaN(v) && !Number.isNaN(bestVal))) {
        best = i;
        bestVal = v;
      }
      if (Number.isNaN(bestVal)) break;
    }
    return V[best].slice();
  }
}

function projectStart(C: FeasibleSet, x0: Vector): Vector {
  try {
    return C.project(x0);
  } catch (e) {
    if (e instanceof ProjectionError)
      throw new ConstrainedInputError(`cannot project x0 onto the feasible set: ${e.message}`);
    throw e;
  }
}

// ---------------------------------------------------------------------------------------
// Parameter specs
// ---------------------------------------------------------------------------------------

const tolParam = (def: number) =>
  param.float('tol', def, {
    min: 1e-14,
    max: 1e-2,
    log: true,
    help: 'Stop when the KKT residual and the constraint violation are both ≤ tol.',
    label: 'KKT tolerance',
    tex: '\\varepsilon',
  });

const maxIterParam = (def: number) =>
  param.int('max_iter', def, {
    min: 1,
    max: 100_000,
    help: 'Limit on the total number of iterations (inner and outer steps together).',
    label: 'Iteration budget',
  });

type Opts = RunOptions & Params;
const num = (o: Opts, k: string, d: number) => (o[k] === undefined ? d : Number(o[k]));

// ---------------------------------------------------------------------------------------
// Projected gradient
// ---------------------------------------------------------------------------------------

/**
 * Gradient projection with the Armijo rule along the projection arc (Bertsekas 1999, §2.3.1):
 * x_k(s) = P_C(x_k − s∇f(x_k)), accept the first s = s̄βᵐ with
 * f(x_k) − f(x_k(s)) ≥ σ∇f(x_k)ᵀ(x_k − x_k(s)). Converged when ‖x − P_C(x − ∇f)‖∞ ≤ tol.
 */
export function projectedGradient(problem: unknown, o: Opts): Result {
  const method = 'projected_gradient';
  const sBar = num(o, 's_bar', 1.0),
    beta = num(o, 'beta', 0.5),
    sigma = num(o, 'sigma', 1e-4),
    tol = num(o, 'tol', 1e-6),
    maxIter = num(o, 'max_iter', 1000);
  const prob = vectorProblem(problem, o.x0);
  if (!(sBar > 0.0 && beta > 0.0 && beta < 1.0 && sigma > 0.0 && sigma < 1.0))
    throw new ConstrainedInputError('need s_bar > 0, 0 < beta < 1 and 0 < sigma < 1');
  const maxTrials = 1 + Math.ceil(Math.log(MIN_TRIAL_RATIO) / Math.log(beta));
  const model = new Model(prob);
  const C = new FeasibleSet(prob, method);
  const xGiven = startPoint(prob, o.x0);
  let x = projectStart(C, xGiven);
  let fx = model.fun(x);
  let g = model.grad(x);
  let c = model.cval(x);
  const info0: StepInfo = {};
  if (!arraysEqual(x, xGiven)) info0.projected_from = [...xGiven];
  if (!allFinite(fx, g, c)) {
    const trace = [step(0, x, fx, null, null, { ...state(c, model.isEq, 0, 0, NaN), ...info0 })];
    return finish(method, model, x, fx, false, nonfiniteMsg(x), trace, NaN, NaN);
  }
  let xUnit: Vector;
  try {
    xUnit = C.project(subV(x, g));
  } catch (e) {
    if (!(e instanceof ProjectionError)) throw e;
    const trace = [step(0, x, fx, null, null, { ...state(c, model.isEq, 0, 0, NaN), ...info0 })];
    return finish(method, model, x, fx, false, e.message, trace, NaN, 0.0);
  }
  let res = normInfV(subV(x, xUnit));
  let viol = violationOf(c, model.isEq);
  const trace: Step[] = [
    step(0, x, fx, norm2(g), null, { ...state(c, model.isEq, 0, 0, res), ...info0 }),
  ];
  let k = 0;
  for (;;) {
    if (res <= tol && viol <= tol)
      return finish(method, model, x, fx, true, convergedMsg(res, viol, tol), trace, res, viol);
    if (k === maxIter)
      return finish(method, model, x, fx, false, maxIterMsg(maxIter, res, viol), trace, res, viol);
    const resF = resolution(fx);
    let s = sBar;
    const trials: number[][] = [];
    const arc: number[][] = [];
    let accepted = false,
      stalled = false,
      rounding = false;
    let z = x,
      fz = fx;
    let gz: Vector | null = null;
    let gNew: Vector, cNew: Vector, zUnit: Vector;
    try {
      for (let tr = 0; tr < maxTrials; tr++) {
        z = s === 1.0 ? xUnit : C.project(subV(x, scaleV(s, g)));
        if (arraysEqual(z, x)) {
          stalled = true;
          break;
        }
        fz = model.fun(z);
        trials.push([s, fz]);
        arc.push([...z]);
        const pred = dot(g, subV(x, z));
        gz = null;
        rounding = pred <= resF && Number.isFinite(fz);
        if (!rounding) {
          if (fx - fz >= sigma * pred) {
            accepted = true;
            break;
          }
        } else if (fz <= fx + resF) {
          gz = model.grad(z);
          if (approxArmijo(dot(gz, subV(z, x)), pred, sigma)) {
            accepted = true;
            break;
          }
        }
        s *= beta;
      }
      if (stalled || !accepted) {
        let msg: string;
        if (!Number.isFinite(fz))
          msg = nonfiniteTrialsMsg('Armijo rule along the projection arc', s / beta, res);
        else if (stalled) msg = nullTrialMsg(res, viol, tol);
        else if (rounding) msg = roundingLimitMsg(maxTrials, s / beta, res, viol, tol);
        else
          msg =
            `Armijo rule along the projection arc: trial limit reached (${maxTrials} trials, ` +
            `s down to ${pyG(s / beta)}; KKT residual ${pyG(res)}); f may be non-finite ` +
            'along the arc or the gradient inaccurate';
        return finish(method, model, x, fx, false, msg, trace, res, viol);
      }
      gNew = gz !== null ? gz : model.grad(z);
      cNew = model.cval(z);
      if (!allFinite(fz, gNew, cNew))
        return finish(method, model, x, fx, false, nonfiniteMsg(z), trace, res, viol);
      zUnit = C.project(subV(z, gNew));
    } catch (e) {
      if (!(e instanceof ProjectionError)) throw e;
      return finish(method, model, x, fx, false, e.message, trace, res, viol);
    }
    const xPrev = x,
      gPrev = g;
    x = z;
    fx = fz;
    g = gNew;
    c = cNew;
    xUnit = zUnit;
    res = normInfV(subV(x, xUnit));
    viol = violationOf(c, model.isEq);
    k += 1;
    trace.push(
      step(k, x, fx, norm2(g), s, {
        ...state(c, model.isEq, 0, k, res),
        from: [...xPrev],
        gradient: [...gPrev],
        unprojected: subV(xPrev, scaleV(s, gPrev)),
        s,
        arc,
        trials,
      }),
    );
  }
}

// ---------------------------------------------------------------------------------------
// Frank–Wolfe (conditional gradient)
// ---------------------------------------------------------------------------------------

/**
 * Frank–Wolfe / conditional gradient (Jaggi 2013, Algorithm 1): s_k = argmin_{s∈C} ∇f(x_k)ᵀs,
 * x_{k+1} = (1 − γ_k)x_k + γ_k s_k with γ_k = 2/(k + 2) or Armijo backtracking from γ = 1.
 * Converged when the gap ∇f(x_k)ᵀ(x_k − s_k) ≤ tol.
 */
export function frankWolfe(problem: unknown, o: Opts): Result {
  const method = 'frank_wolfe';
  const rule = (o.step ?? 'open_loop') as string;
  const tol = num(o, 'tol', 1e-6),
    maxIter = num(o, 'max_iter', 1000);
  if (rule !== 'open_loop' && rule !== 'armijo')
    throw new ConstrainedInputError(
      `unknown step rule '${rule}'; expected 'open_loop' or 'armijo'`,
    );
  const prob = vectorProblem(problem, o.x0);
  const model = new Model(prob);
  const C = new FeasibleSet(prob, method, true);
  const xGiven = startPoint(prob, o.x0);
  let x = projectStart(C, xGiven);
  let fx = model.fun(x);
  let g = model.grad(x);
  let c = model.cval(x);
  const projected: StepInfo = {};
  if (!arraysEqual(x, xGiven)) projected.projected_from = [...xGiven];
  if (!allFinite(fx, g, c)) {
    const trace = [
      step(0, x, fx, null, null, { ...state(c, model.isEq, 0, 0, NaN), ...projected }),
    ];
    return finish(method, model, x, fx, false, nonfiniteMsg(x), trace, NaN, NaN);
  }
  let s = C.lmo(g, x);
  let gap = dot(g, subV(x, s));
  let viol = violationOf(c, model.isEq);
  const trace: Step[] = [
    step(0, x, fx, norm2(g), null, {
      ...state(c, model.isEq, 0, 0, gap),
      lmo: [...s],
      gap,
      ...projected,
    }),
  ];
  let k = 0;
  for (;;) {
    if (gap <= tol && viol <= tol)
      return finish(method, model, x, fx, true, convergedMsg(gap, viol, tol), trace, gap, viol);
    if (k === maxIter)
      return finish(method, model, x, fx, false, maxIterMsg(maxIter, gap, viol), trace, gap, viol);
    const d = subV(s, x);
    const trials: number[][] = [];
    let gNew: Vector | null = null;
    let gamma: number, xNew: Vector, fNew: number;
    if (rule === 'open_loop') {
      gamma = 2.0 / (k + 2.0);
      xNew = addV(scaleV(1.0 - gamma, x), scaleV(gamma, s));
      fNew = model.fun(xNew);
      trials.push([gamma, fNew]);
    } else {
      const resF = resolution(fx);
      gamma = 1.0;
      xNew = x;
      fNew = fx;
      let accepted = false,
        rounding = false;
      for (let tr = 0; tr < MAX_TRIALS; tr++) {
        xNew = addV(scaleV(1.0 - gamma, x), scaleV(gamma, s));
        if (arraysEqual(xNew, x)) {
          const msg = Number.isFinite(fNew)
            ? nullTrialMsg(gap, viol, tol)
            : nonfiniteTrialsMsg('Armijo backtracking on γ', 2.0 * gamma, gap);
          return finish(method, model, x, fx, false, msg, trace, gap, viol);
        }
        fNew = model.fun(xNew);
        trials.push([gamma, fNew]);
        const pred = gamma * gap;
        gNew = null;
        rounding = pred <= resF && Number.isFinite(fNew);
        if (!rounding) {
          if (fNew <= fx - C1 * pred) {
            accepted = true;
            break;
          }
        } else if (fNew <= fx + resF) {
          gNew = model.grad(xNew);
          if (approxArmijo(gamma * dot(gNew, d), pred, C1)) {
            accepted = true;
            break;
          }
        }
        gamma *= SHRINK;
      }
      if (!accepted) {
        const msg = !Number.isFinite(fNew)
          ? nonfiniteTrialsMsg('Armijo backtracking on γ', 2.0 * gamma, gap)
          : rounding
            ? roundingLimitMsg(MAX_TRIALS, 2.0 * gamma, gap, viol, tol)
            : `Armijo backtracking on γ: trial limit reached (${MAX_TRIALS} trials, ` +
              `γ down to ${pyG(2.0 * gamma)}; gap ${pyG(gap)}); f may be non-finite ` +
              'along the segment or the gradient inaccurate';
        return finish(method, model, x, fx, false, msg, trace, gap, viol);
      }
    }
    const gN = gNew !== null ? gNew : model.grad(xNew);
    const cNew = model.cval(xNew);
    if (!allFinite(fNew, gN, cNew))
      return finish(method, model, x, fx, false, nonfiniteMsg(xNew), trace, gap, viol);
    const xPrev = x,
      sPrev = s;
    x = xNew;
    fx = fNew;
    g = gN;
    c = cNew;
    s = C.lmo(g, x);
    gap = dot(g, subV(x, s));
    viol = violationOf(c, model.isEq);
    k += 1;
    trace.push(
      step(k, x, fx, norm2(g), gamma, {
        ...state(c, model.isEq, 0, k, gap),
        from: [...xPrev],
        vertex: [...sPrev],
        direction: d,
        gamma,
        trials,
        lmo: [...s],
        gap,
      }),
    );
  }
}

// ---------------------------------------------------------------------------------------
// Sequential unconstrained minimization: quadratic penalty and augmented Lagrangian
// ---------------------------------------------------------------------------------------

interface Point {
  x: Vector;
  f: number;
  c: Vector;
  g: Vector;
  J: Matrix;
}

/** Outer-loop rules of a sequential unconstrained minimization method. */
interface Policy {
  merit(f: number, c: Vector): number;
  /** Multiplier estimate w(x) with ∇merit = ∇f + Jᵀw. */
  weights(c: Vector): Vector;
  innerTol(): number;
  /** Update the outer parameters; return a failure message to stop. */
  afterInner(pt: Point): string | null;
  info(): StepInfo;
}

/** Inverse BFGS update (N&W eq. 6.17) with the initial scaling (eq. 6.20); skipped if sᵀy ≤ 0. */
function bfgsUpdate(H: Matrix, s: Vector, y: Vector, first: boolean): [Matrix, boolean] {
  const sy = dot(s, y);
  if (!(sy > 1e-12 * norm2(s) * norm2(y))) return [H, false];
  const n = s.length;
  let Hm = H;
  if (first) Hm = scaleM(sy / dot(y, y), eye(n));
  const rho = 1.0 / sy;
  const I = eye(n);
  const sy_ = outerV(s, y);
  const V = I.map((row, i) => row.map((v, j) => v - rho * sy_[i][j]));
  const VHVt = matmul(matmul(V, Hm), transpose(V));
  const ss = outerV(s, s);
  return [VHVt.map((row, i) => row.map((v, j) => v + rho * ss[i][j])), true];
}

function scaleM(s: number, A: Matrix): Matrix {
  return A.map((r) => r.map((v) => s * v));
}

function sequential(
  method: string,
  problem: unknown,
  x0: unknown,
  tol: number,
  maxIter: number,
  makePolicy: (model: Model) => Policy,
): Result {
  const prob = vectorProblem(problem, x0);
  const model = new Model(prob);
  const policy = makePolicy(model);
  const isEq = model.isEq;
  const n = model.n;
  const x = startPoint(prob, x0);
  const fx = model.fun(x);
  const c = model.cval(x);
  const g = model.grad(x);
  const J = model.cjac(x);
  let pt: Point = { x, f: fx, c, g, J };
  const w = allFinite(c) ? policy.weights(c) : new Array<number>(model.m).fill(NaN);
  let gphi = addV(g, matTvec(J, w, n));
  let phi = policy.merit(fx, c);
  if (!allFinite(fx, c, g, J, gphi)) {
    const trace = [step(0, x, fx, null, null, state(c, isEq, 0, 0, NaN))];
    return finish(method, model, x, fx, false, nonfiniteMsg(x), trace, NaN, NaN);
  }
  let lam = w;
  let [stat, kkt] = kktOf(g, J, lam, c, isEq);
  let viol = violationOf(c, isEq);
  const trace: Step[] = [
    step(0, x, fx, norm2(g), null, {
      ...state(c, isEq, 0, 0, kkt),
      ...policy.info(),
      merit: phi,
      multipliers: [...lam],
      stationarity: stat,
    }),
  ];
  const done = (converged: boolean, msg: string) =>
    finish(method, model, pt.x, pt.f, converged, msg, trace, kkt, viol, lam);

  if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
  let k = 0;
  let outer = 0;
  let idleOuter = 0;
  for (;;) {
    const tau = policy.innerTol();
    let H = eye(n);
    let nUpdates = 0;
    let inner = 0;
    while (normInfV(gphi) > tau) {
      if (k === maxIter) return done(false, maxIterMsg(maxIter, kkt, viol));
      let p = negV(matvec(H, gphi));
      let slope = dot(gphi, p);
      if (!(slope < 0.0)) {
        H = eye(n);
        nUpdates = 0;
        p = negV(gphi);
        slope = -dot(gphi, gphi);
      }
      let alpha = nUpdates ? 1.0 : pyMin(1.0, 1.0 / norm2(p));
      const trials: number[][] = [];
      let accepted = false;
      let xt = pt.x,
        ft = pt.f,
        ct = pt.c,
        phit = phi;
      for (let tr = 0; tr < MAX_TRIALS; tr++) {
        xt = axpyP(pt.x, alpha, p);
        ft = model.fun(xt);
        ct = model.cval(xt);
        phit = allFinite(ft, ct) ? policy.merit(ft, ct) : Infinity;
        trials.push([alpha, phit]);
        if (
          phit <= phi + C1 * alpha * slope ||
          (-alpha * slope <= resolution(phi) && phit <= phi + resolution(phi))
        ) {
          accepted = true;
          break;
        }
        alpha *= SHRINK;
      }
      if (!accepted) return done(false, `inner line search failed after ${MAX_TRIALS} trials`);
      if (arraysEqual(xt, pt.x)) return done(false, stalledMsg(kkt, viol, tol));
      const gt = model.grad(xt);
      const Jt = model.cjac(xt);
      const wt = policy.weights(ct);
      const gphiT = addV(gt, matTvec(Jt, wt, n));
      if (!allFinite(gt, Jt, gphiT)) return done(false, nonfiniteMsg(xt));
      let updated: boolean;
      [H, updated] = bfgsUpdate(H, subV(xt, pt.x), subV(gphiT, gphi), nUpdates === 0);
      nUpdates += updated ? 1 : 0;
      const xPrev = pt.x;
      pt = { x: xt, f: ft, c: ct, g: gt, J: Jt };
      gphi = gphiT;
      phi = phit;
      lam = wt;
      k += 1;
      inner += 1;
      [stat, kkt] = kktOf(gt, Jt, lam, ct, isEq);
      viol = violationOf(ct, isEq);
      trace.push(
        step(k, xt, ft, norm2(gt), alpha, {
          ...state(ct, isEq, outer, inner, kkt),
          ...policy.info(),
          merit: phit,
          multipliers: [...wt],
          stationarity: stat,
          from: [...xPrev],
          direction: [...p],
          alpha,
          trials,
        }),
      );
      if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
    }
    if (inner === 0) {
      // The outer parameters changed without a new step: test the current point with the
      // multiplier estimate of the current merit (no Step is added).
      lam = policy.weights(pt.c);
      kkt = kktOf(pt.g, pt.J, lam, pt.c, isEq)[1];
      if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
      idleOuter += 1;
      if (idleOuter > MAX_OUTER)
        return done(false, `${MAX_OUTER} outer updates in a row without progress`);
    } else idleOuter = 0;
    const msg = policy.afterInner(pt);
    if (msg !== null) return done(false, msg);
    outer += 1;
    gphi = addV(pt.g, matTvec(pt.J, policy.weights(pt.c), n));
    phi = policy.merit(pt.f, pt.c);
  }
}

class PenaltyPolicy implements Policy {
  mu: number;
  private readonly isEq: boolean[];
  private readonly rho: number;
  private readonly muMax: number;
  private readonly tol: number;
  constructor(isEq: boolean[], mu0: number, rho: number, muMax: number, tol: number) {
    this.isEq = isEq;
    this.mu = mu0;
    this.rho = rho;
    this.muMax = muMax;
    this.tol = tol;
  }
  private r(c: Vector): Vector {
    return c.map((v, i) => (this.isEq[i] ? v : Math.max(v, 0.0)));
  }
  merit(f: number, c: Vector): number {
    const r = this.r(c);
    return f + 0.5 * this.mu * dot(r, r);
  }
  weights(c: Vector): Vector {
    return scaleV(this.mu, this.r(c));
  }
  innerTol(): number {
    return pyMax(this.tol, 1.0 / this.mu);
  }
  afterInner(pt: Point): string | null {
    if (this.mu * this.rho > this.muMax)
      return (
        `penalty parameter would exceed mu_max = ${pyG(this.muMax)} ` +
        `(violation ${pyG(violationOf(pt.c, this.isEq))}; the problem may be infeasible ` +
        'or need a larger mu_max)'
      );
    this.mu *= this.rho;
    return null;
  }
  info(): StepInfo {
    return { mu: this.mu, tau: this.innerTol() };
  }
}

/**
 * Quadratic penalty method (N&W Framework 17.1): minimize Q(x; μ) = f + (μ/2)(Σ_E c_i² +
 * Σ_I max(0, c_i)²) with BFGS until ‖∇Q‖∞ ≤ max(tol, 1/μ), then μ ← ρμ.
 */
export function quadraticPenalty(problem: unknown, o: Opts): Result {
  const mu0 = num(o, 'mu0', 1.0),
    rho = num(o, 'rho', 10.0),
    muMax = num(o, 'mu_max', 1e10),
    tol = num(o, 'tol', 1e-6),
    maxIter = num(o, 'max_iter', 1000);
  if (!(mu0 > 0.0 && rho > 1.0 && muMax >= mu0))
    throw new ConstrainedInputError('need mu0 > 0, rho > 1 and mu_max ≥ mu0');
  return sequential(
    'quadratic_penalty',
    problem,
    o.x0,
    tol,
    maxIter,
    (model) => new PenaltyPolicy(model.isEq, mu0, rho, muMax, tol),
  );
}

class ALPolicy implements Policy {
  mu: number;
  lam: Vector;
  omega: number;
  eta: number;
  update = 'start';
  private readonly isEq: boolean[];
  private readonly muFactor: number;
  private readonly muMax: number;
  private readonly tol: number;
  constructor(
    isEq: boolean[],
    m: number,
    mu0: number,
    muFactor: number,
    muMax: number,
    tol: number,
  ) {
    this.isEq = isEq;
    this.muFactor = muFactor;
    this.muMax = muMax;
    this.tol = tol;
    this.mu = mu0;
    this.lam = zerosV(m);
    // N&W Alg. 17.4 initial tolerances: ω₀ = 1/μ₀, η₀ = 1/μ₀^0.1.
    this.omega = pyMax(1.0 / mu0, tol);
    this.eta = 1.0 / mu0 ** 0.1;
  }
  private shifted(c: Vector): Vector {
    return this.lam.map((l, i) => l + this.mu * c[i]);
  }
  merit(f: number, c: Vector): number {
    const lc = this.shifted(c);
    const eq = this.isEq;
    const ce = pick(c, eq),
      le = pick(this.lam, eq);
    const val = f + dot(le, ce) + 0.5 * this.mu * dot(ce, ce);
    const ineq = eq.map((e) => !e);
    const plus = pick(lc, ineq).map((v) => Math.max(v, 0.0));
    const li = pick(this.lam, ineq);
    return val + (dot(plus, plus) - dot(li, li)) / (2.0 * this.mu);
  }
  weights(c: Vector): Vector {
    const lc = this.shifted(c);
    return lc.map((v, i) => (this.isEq[i] ? v : Math.max(v, 0.0)));
  }
  innerTol(): number {
    return this.omega;
  }
  afterInner(pt: Point): string | null {
    // Birgin & Martínez (2014) measure V_i = max(c_i, −λ_i/μ) for inequalities, |c_i| for equalities.
    const V = pt.c.map((ci, i) =>
      this.isEq[i] ? Math.abs(ci) : Math.abs(Math.max(ci, -this.lam[i] / this.mu)),
    );
    if (V.length === 0 || npMax(V) <= this.eta) {
      this.lam = this.weights(pt.c);
      this.eta = this.eta / this.mu ** 0.9;
      this.omega = pyMax(this.omega / this.mu, this.tol);
      this.update = 'multipliers';
      return null;
    }
    if (this.mu * this.muFactor > this.muMax)
      return (
        `penalty parameter would exceed mu_max = ${pyG(this.muMax)} ` +
        `(violation ${pyG(violationOf(pt.c, this.isEq))}; the problem may be infeasible)`
      );
    this.mu *= this.muFactor;
    this.eta = 1.0 / this.mu ** 0.1;
    this.omega = pyMax(1.0 / this.mu, this.tol);
    this.update = 'penalty';
    return null;
  }
  info(): StepInfo {
    return {
      mu: this.mu,
      lambda: [...this.lam],
      omega: this.omega,
      eta: this.eta,
      update: this.update,
    };
  }
}

/**
 * Augmented Lagrangian (N&W Framework 17.3 with the updates of Alg. 17.4), PHR form for
 * inequalities: L_A = f + Σ_E [λ_i c_i + (μ/2)c_i²] + (1/2μ) Σ_I [max(0, λ_i + μc_i)² − λ_i²].
 */
export function augmentedLagrangian(problem: unknown, o: Opts): Result {
  const mu0 = num(o, 'mu0', 10.0),
    muFactor = num(o, 'mu_factor', 100.0),
    muMax = num(o, 'mu_max', 1e12),
    tol = num(o, 'tol', 1e-8),
    maxIter = num(o, 'max_iter', 1000);
  if (!(mu0 > 1.0 && muFactor > 1.0 && muMax >= mu0))
    throw new ConstrainedInputError('need mu0 > 1, mu_factor > 1 and mu_max ≥ mu0');
  return sequential(
    'augmented_lagrangian',
    problem,
    o.x0,
    tol,
    maxIter,
    (model) => new ALPolicy(model.isEq, model.m, mu0, muFactor, muMax, tol),
  );
}

// ---------------------------------------------------------------------------------------
// Log-barrier interior-point method
// ---------------------------------------------------------------------------------------

/** H + τI with the smallest τ of the sequence of N&W Alg. 3.3 that admits Cholesky. */
function makePd(H: Matrix): [Matrix, number] | null {
  const n = H.length;
  const dmin = Math.min(...H.map((r, i) => r[i]));
  let tau = dmin > 0.0 ? 0.0 : -dmin + SHIFT_BETA;
  for (let it = 0; it < 200; it++) {
    const Ht = H.map((r, i) => r.map((v, j) => v + tau * (i === j ? 1 : 0)));
    if (cholesky(Ht)) return [Ht, tau];
    tau = pyMax(2.0 * tau, SHIFT_BETA);
  }
  void n;
  return null;
}

/**
 * Barrier method (Boyd & Vandenberghe, Alg. 11.1) with Newton centering: minimize
 * t f(x) − Σ_I log(−c_i(x)) subject to the affine equalities, then t ← μt.
 */
export function logBarrier(problem: unknown, o: Opts): Result {
  const method = 'log_barrier';
  const t0 = num(o, 't0', 1.0),
    mu = num(o, 'mu', 10.0),
    tol = num(o, 'tol', 1e-6),
    newtonTol = num(o, 'newton_tol', 1e-10),
    maxIter = num(o, 'max_iter', 500);
  if (!(t0 > 0.0 && mu > 1.0 && newtonTol > 0.0))
    throw new ConstrainedInputError('need t0 > 0, mu > 1 and newton_tol > 0');
  const prob = vectorProblem(problem, o.x0);
  const model = new Model(prob);
  const E = model.isEq,
    I = model.isIn;
  const eqIdx = E.flatMap((e, i) => (e ? [i] : []));
  const inIdx = I.flatMap((e, i) => (e ? [i] : []));
  const notAffine = eqIdx.filter((i) => !model.affine[i]);
  if (notAffine.length)
    throw new ConstrainedInputError(
      `log_barrier needs affine equality constraints (B&V §11.1); equality constraint(s) ` +
        `[${notAffine.join(', ')}] of problem '${prob.id}' are not declared affine ` +
        "(set Problem.extra['affine'][i] = True for an affine c_i)",
    );
  const xGiven = startPoint(prob, o.x0);
  let x = xGiven.slice();
  let c = model.cval(x);
  const info0: StepInfo = {};
  if (eqIdx.length && pick(c, E).some((v) => v !== 0.0)) {
    const A = pick(model.cjac(x), E);
    x = subV(x, lstsq(A, pick(c, E)));
    c = model.cval(x);
    info0.projected_from = [...xGiven];
  }
  if (!(allFinite(c) && pick(c, I).every((v) => v < 0.0)))
    throw new ConstrainedInputError(
      'log_barrier needs a strictly feasible x0 (c_i(x0) < 0 for every inequality); ' +
        `got c(x0) = ${pyList(c)}`,
    );
  const mIn = inIdx.length;
  const n = model.n;
  const zero = eye(n).map((r) => r.map((v) => v * 0.0));

  const evaluate = (z: Vector): [Vector, Matrix, Matrix, Matrix[]] => {
    const gz = model.grad(z);
    const Jz = model.cjac(z);
    const Hz = model.hess(z);
    const CH = inIdx.map((i) => (model.affine[i] ? zero : model.chess(i, z)));
    return [gz, Jz, Hz, CH];
  };
  const barrier = (t: number, f: number, cv: Vector) =>
    t * f - npSum(pick(cv, I).map((v) => Math.log(-v)));
  const multipliers = (t: number, gz: Vector, Jz: Matrix, cv: Vector): Vector => {
    const lam = zerosV(model.m);
    for (const i of inIdx) lam[i] = -1.0 / (t * cv[i]);
    if (eqIdx.length) {
      const JI = inIdx.map((i) => Jz[i]);
      const r = addV(
        gz,
        matTvec(
          JI,
          inIdx.map((i) => lam[i]),
          n,
        ),
      );
      const JEt = transpose(
        eqIdx.map((i) => Jz[i]),
        n,
      );
      const nu = lstsq(JEt, negV(r));
      eqIdx.forEach((i, j) => (lam[i] = nu[j]));
    }
    return lam;
  };

  let fx = model.fun(x);
  let [g, J, H, CH] = evaluate(x);
  let t = t0;
  if (!allFinite(fx, g, J, H, ...CH)) {
    const trace = [step(0, x, fx, null, null, { ...state(c, E, 0, 0, NaN), ...info0 })];
    return finish(method, model, x, fx, false, nonfiniteMsg(x), trace, NaN, NaN);
  }
  let lam = multipliers(t, g, J, c);
  let [stat, kkt] = kktOf(g, J, lam, c, E);
  let viol = violationOf(c, E);
  let F = barrier(t, fx, c);
  const trace: Step[] = [
    step(0, x, fx, norm2(g), null, {
      ...state(c, E, 0, 0, kkt),
      t,
      gap: mIn / t,
      merit: F,
      multipliers: [...lam],
      stationarity: stat,
      ...info0,
    }),
  ];
  const done = (converged: boolean, msg: string) =>
    finish(method, model, x, fx, converged, msg, trace, kkt, viol, lam);

  if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
  let k = 0;
  let outer = 0;
  let inner = 0;
  let atFloor = false;
  for (;;) {
    const ci = pick(c, I);
    const Ji = pick(J, I);
    const invC = ci.map((v) => 1.0 / v);
    const dF = subV(scaleV(t, g), matTvec(Ji, invC, n));
    let HF: Matrix = Array.from({ length: n }, (_, a) =>
      Array.from({ length: n }, (_, b) => {
        let s = 0;
        for (let i = 0; i < Ji.length; i++) s += Ji[i][a] * (Ji[i][b] / (ci[i] * ci[i]));
        return t * H[a][b] + s;
      }),
    );
    CH.forEach((ch, idx) => {
      const cv = ci[idx];
      HF = HF.map((row, a) => row.map((v, b) => v - ch[a][b] / cv));
    });
    HF = symmetrize(HF);
    const made = makePd(HF);
    if (made === null) return done(false, 'could not make the barrier Hessian positive definite');
    const [Hm, shift] = made;
    let dx: Vector | null;
    if (eqIdx.length) {
      const A = eqIdx.map((i) => J[i]);
      const p = A.length;
      const K: Matrix = [
        ...Hm.map((row, a) => [...row, ...A.map((ar) => ar[a])]),
        ...A.map((ar) => [...ar, ...zerosV(p)]),
      ];
      const sol = luSolve(K, [...negV(dF), ...negV(pick(c, E))]);
      dx = sol ? sol.slice(0, n) : null;
    } else dx = luSolve(Hm, negV(dF));
    if (dx === null)
      return done(false, 'singular Newton (KKT) system: the equality constraints are dependent');
    const decrement = 0.5 * dot(matTvec(Hm, dx, n), dx);
    const finalStage = mIn === 0 || 1.0 / t < tol;
    if ((decrement <= newtonTol || atFloor) && !finalStage) {
      atFloor = false;
      lam = multipliers(t, g, J, c);
      [stat, kkt] = kktOf(g, J, lam, c, E);
      if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
      t *= mu;
      outer += 1;
      inner = 0;
      F = barrier(t, fx, c);
      continue;
    }
    if (k === maxIter) return done(false, maxIterMsg(maxIter, kkt, viol));
    const slope = dot(dF, dx);
    if (slope > resolution(F))
      return done(
        false,
        `the Newton step is not a descent direction of the barrier function ` +
          `(∇FᵀΔx = ${pyG(slope)} > 0); are the equality constraints affine?`,
      );
    const belowResolution = -slope <= resolution(F);
    let s = 1.0;
    const trials: number[][] = [];
    let accepted = false;
    let xt = x,
      ft = fx,
      ct = c,
      Ft = F;
    for (let tr = 0; tr < MAX_TRIALS; tr++) {
      xt = axpyP(x, s, dx);
      ct = model.cval(xt);
      if (!(allFinite(ct) && pick(ct, I).every((v) => v < 0.0))) {
        trials.push([s, Infinity]);
        s *= BARRIER_BETA;
        continue;
      }
      ft = model.fun(xt);
      Ft = Number.isFinite(ft) ? barrier(t, ft, ct) : Infinity;
      trials.push([s, Ft]);
      if (Number.isFinite(Ft) && (belowResolution || Ft <= F + BARRIER_ALPHA * s * slope)) {
        accepted = true;
        break;
      }
      s *= BARRIER_BETA;
    }
    if (!accepted) return done(false, `backtracking line search failed after ${MAX_TRIALS} trials`);
    if (arraysEqual(xt, x)) {
      const where = mIn ? `; max_I c_i(x) = ${pyG(npMax(pick(c, I)))}` : '';
      return done(false, stalledMsg(kkt, viol, tol) + where);
    }
    const [gt, Jt, Ht, CHt] = evaluate(xt);
    if (!allFinite(gt, Jt, Ht, ...CHt)) return done(false, nonfiniteMsg(xt));
    const statBefore = stat;
    const xPrev = x;
    x = xt;
    fx = ft;
    c = ct;
    F = Ft;
    g = gt;
    J = Jt;
    H = Ht;
    CH = CHt;
    k += 1;
    inner += 1;
    lam = multipliers(t, g, J, c);
    [stat, kkt] = kktOf(g, J, lam, c, E);
    viol = violationOf(c, E);
    trace.push(
      step(k, x, fx, norm2(g), s, {
        ...state(c, E, outer, inner, kkt),
        t,
        gap: mIn / t,
        merit: F,
        multipliers: [...lam],
        stationarity: stat,
        from: [...xPrev],
        direction: [...dx],
        alpha: s,
        trials,
        newton_decrement: decrement,
        hessian_shift: shift,
      }),
    );
    if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
    if (belowResolution && !(stat < statBefore)) {
      if (!finalStage) {
        atFloor = true;
        continue;
      }
      const why = mIn
        ? `for t = ${pyG(t)}: ${excess(kkt, viol, tol)} (near the boundary the ` +
          'rounding error of c_i(x) ≈ −1/(t λ_i) is amplified in λ_i = −1/(t c_i) ' +
          'and in the barrier Hessian; use a larger tol)'
        : `${excess(kkt, viol, tol)} (no inequalities: the full Newton step of the ` +
          'equality-constrained problem no longer reduces the KKT residual in ' +
          'float64; use a larger tol)';
      return done(false, `stalled at the rounding level ${why}`);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Sequential quadratic programming
// ---------------------------------------------------------------------------------------

/**
 * Line-search SQP with a damped-BFGS Hessian of the Lagrangian (N&W Alg. 18.3): solve the QP
 * min ½pᵀBp + ∇fᵀp s.t. ∇c_iᵀp + c_i = 0 (E), ≤ 0 (I) by `solveQp`, backtrack on the ℓ1 merit
 * φ₁ = f + μ‖c⁻‖₁, take λ_{k+1} = λ̂, update B by Powell's damped BFGS.
 */
export function sqp(problem: unknown, o: Opts): Result {
  const method = 'sqp';
  const tol = num(o, 'tol', 1e-8),
    maxIter = num(o, 'max_iter', 200);
  const prob = vectorProblem(problem, o.x0);
  const model = new Model(prob);
  const E = model.isEq,
    I = model.isIn;
  const n = model.n;
  let x = startPoint(prob, o.x0);
  let fx = model.fun(x);
  let c = model.cval(x);
  let g = model.grad(x);
  let J = model.cjac(x);
  let lam = zerosV(model.m);
  let muMerit = 0.0;
  const l1 = (cv: Vector) =>
    npSum(pick(cv, E).map(Math.abs)) + npSum(pick(cv, I).map((v) => Math.max(v, 0.0)));

  if (!allFinite(fx, c, g, J)) {
    const trace = [step(0, x, fx, null, null, state(c, E, 0, 0, NaN))];
    return finish(method, model, x, fx, false, nonfiniteMsg(x), trace, NaN, NaN);
  }
  let [stat, kkt] = kktOf(g, J, lam, c, E);
  let viol = violationOf(c, E);
  let phi = fx + muMerit * l1(c);
  const trace: Step[] = [
    step(0, x, fx, norm2(g), null, {
      ...state(c, E, 0, 0, kkt),
      mu: muMerit,
      merit: phi,
      multipliers: [...lam],
      stationarity: stat,
    }),
  ];
  const done = (converged: boolean, msg: string) =>
    finish(method, model, x, fx, converged, msg, trace, kkt, viol, lam);

  let B = eye(n);
  let k = 0;
  const inIdx = I.flatMap((e, i) => (e ? [i] : []));
  const eqIdx = E.flatMap((e, i) => (e ? [i] : []));
  for (;;) {
    if (kkt <= tol && viol <= tol) return done(true, convergedMsg(kkt, viol, tol));
    if (k === maxIter) return done(false, maxIterMsg(maxIter, kkt, viol));
    const qp = solveQp(B, g, pick(J, E), negV(pick(c, E)), pick(J, I), negV(pick(c, I)));
    if (!qp.ok) return done(false, `QP subproblem failed at x = ${pyList(x)}: ${qp.message}`);
    const p = qp.x;
    const lamHat = zerosV(model.m);
    eqIdx.forEach((i, j) => (lamHat[i] = qp.lamEq[j]));
    inIdx.forEach((i, j) => (lamHat[i] = qp.lamUb[j]));
    const working = [...eqIdx, ...qp.active.map((j) => inIdx[j])].sort((a, b) => a - b);
    const v1 = l1(c);
    const gp = dot(g, p);
    const pBp = dot(matvec(B, p), p);
    if (v1 > 0.0) muMerit = pyMax(muMerit, (gp + 0.5 * pBp) / ((1.0 - SQP_RHO) * v1));
    const D = gp - muMerit * v1;
    phi = fx + muMerit * v1;
    let alpha = 1.0;
    const trials: number[][] = [];
    let accepted = false;
    let xt = x,
      ft = fx,
      ct = c;
    for (let tr = 0; tr < MAX_TRIALS; tr++) {
      xt = axpyP(x, alpha, p);
      ft = model.fun(xt);
      ct = model.cval(xt);
      const phit = allFinite(ft, ct) ? ft + muMerit * l1(ct) : Infinity;
      trials.push([alpha, phit]);
      if (
        phit <= phi + SQP_ETA * alpha * D ||
        (-alpha * D <= resolution(phi) && phit <= phi + resolution(phi))
      ) {
        accepted = true;
        break;
      }
      alpha *= SQP_TAU;
    }
    if (!accepted)
      return done(false, `merit-function line search failed after ${MAX_TRIALS} trials`);
    const gt = model.grad(xt);
    const Jt = model.cjac(xt);
    if (!allFinite(gt, Jt)) return done(false, nonfiniteMsg(xt));
    const lamNew = lamHat;
    if (arraysEqual(xt, x) && arraysEqual(lamNew, lam))
      return done(false, stalledMsg(kkt, viol, tol));
    const s = subV(xt, x);
    const y = subV(addV(gt, matTvec(Jt, lamNew, n)), addV(g, matTvec(J, lamNew, n)));
    const Bs = matvec(B, s);
    const sBs = dot(s, Bs);
    let theta: number | null = null;
    if (sBs > 0.0) {
      const sy = dot(s, y);
      theta = sy >= 0.2 * sBs ? 1.0 : (0.8 * sBs) / (sBs - sy);
      const th = theta;
      const r = addV(scaleV(th, y), scaleV(1.0 - th, Bs));
      const sr = dot(s, r);
      const BsBs = outerV(Bs, Bs),
        rr = outerV(r, r);
      B = B.map((row, i) => row.map((v, j) => v - BsBs[i][j] / sBs + rr[i][j] / sr));
      B = symmetrize(B);
    }
    const xPrev = x;
    x = xt;
    fx = ft;
    c = ct;
    g = gt;
    J = Jt;
    lam = lamNew;
    k += 1;
    [stat, kkt] = kktOf(g, J, lam, c, E);
    viol = violationOf(c, E);
    trace.push(
      step(k, x, fx, norm2(g), alpha, {
        ...state(c, E, 0, k, kkt),
        mu: muMerit,
        merit: fx + muMerit * l1(c),
        multipliers: [...lam],
        stationarity: stat,
        from: [...xPrev],
        direction: [...p],
        alpha,
        trials,
        working_set: working,
        theta,
      }),
    );
  }
}

// ---------------------------------------------------------------------------------------
// Registration (same ids, params and metadata as the Python @register calls)
// ---------------------------------------------------------------------------------------

const asFn = (fn: (p: unknown, o: Opts) => Result) => fn as MethodFn<Problem>;

const DOCS: Record<string, MethodDoc> = {
  projected_gradient: {
    rule: '\\mathbf{x}_{k+1} = P_C\\big(\\mathbf{x}_k - s_k \\nabla f(\\mathbf{x}_k)\\big)',
    intuition:
      'Take a gradient step as if there were no constraints, then return to the feasible set by ' +
      'the nearest-point map P_C. When the step leaves C, the trial points slide along the ' +
      'boundary (the projection arc) and backtracking stops at the first one that decreases f enough.',
    pros: [
      'Cheap when P_C is cheap (boxes, disks)',
      'Identifies the active face in finitely many steps on polyhedra',
    ],
    cons: [
      'Only for sets with an exact projection',
      'Linear rate: zigzags like steepest descent on ill-conditioned f',
    ],
    order: 'linear',
    // Step sizes are those of the move into x_k: index k − 1, as in the rule.
    quantities: [
      { tex: 's_{k-1}', key: 'stepSize' },
      {
        tex: 'r_k',
        key: 'info.kkt_residual',
        label: 'Projection residual ‖x_k − P_C(x_k − ∇f(x_k))‖∞',
      },
    ],
  },
  frank_wolfe: {
    rule: '\\begin{gathered}\\mathbf{s}_k \\in \\arg\\min_{\\mathbf{s} \\in C} \\nabla f(\\mathbf{x}_k)^{\\mathsf T}\\mathbf{s} \\\\ \\mathbf{x}_{k+1} = (1-\\gamma_k)\\,\\mathbf{x}_k + \\gamma_k \\mathbf{s}_k\\end{gathered}',
    intuition:
      'Minimize the linear model of f over the whole feasible set — the answer is an extreme ' +
      'point, the vertex atom s_k — and move part of the way toward it. No projection is ever ' +
      'needed, and the gap ∇f(x_k)ᵀ(x_k − s_k) bounds f(x_k) − f⋆ for convex f.',
    pros: [
      'Projection-free: needs only a linear oracle',
      'Iterates are sparse convex combinations of vertices',
    ],
    order: 'O(1/k)',
    cons: ['Sublinear O(1/k) rate', 'Zigzags between vertices when x⋆ lies on a face'],
    quantities: [
      { tex: '\\gamma_{k-1}', key: 'stepSize' },
      // s_k itself is drawn on the stage; the grid has no room for a vector.
      { tex: 'G_k', key: 'info.gap', label: 'Frank–Wolfe gap' },
    ],
  },
  quadratic_penalty: {
    rule: '\\begin{gathered}Q(\\mathbf{x};\\mu) = f(\\mathbf{x}) + \\tfrac{\\mu}{2}\\Big[\\textstyle\\sum_{E} c_i^2 + \\sum_{I} \\max(0, c_i)^2\\Big] \\\\ \\mathbf{x}(\\mu) \\approx \\arg\\min_{\\mathbf{x}} Q(\\mathbf{x};\\mu),\\quad \\mu \\leftarrow \\rho\\,\\mu\\end{gathered}',
    intuition:
      'Replace the constraints by a quadratic wall that grows with μ, minimize the resulting ' +
      'smooth landscape with BFGS, and repeat with a larger μ. The minimizers approach x⋆ from ' +
      'outside the feasible set, infeasible by O(1/μ); the outer loop converges linearly in the ' +
      'number of μ updates.',
    order: 'linear',
    pros: [
      'Any smooth constraints, no feasible start',
      'Multiplier estimate λᵢ = μcᵢ (E), μ max(0, cᵢ) (I) comes for free',
    ],
    cons: ['Ill-conditioned as μ → ∞', 'Feasibility only in the limit'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'stepSize' },
      { tex: '\\mu', key: 'info.mu' },
      { tex: '\\tau', key: 'info.tau' },
      { tex: '\\boldsymbol{\\lambda}_k', key: 'info.multipliers' },
      { tex: '\\|c^{-}\\|_\\infty', key: 'info.violation' },
    ],
  },
  augmented_lagrangian: {
    rule: '\\begin{gathered}L_A = f + \\textstyle\\sum_E \\big(\\lambda_i c_i + \\tfrac{\\mu}{2}c_i^2\\big) \\\\ + \\tfrac{1}{2\\mu}\\textstyle\\sum_I \\big(\\max(0,\\lambda_i + \\mu c_i)^2 - \\lambda_i^2\\big) \\\\ \\lambda_i \\leftarrow \\lambda_i + \\mu c_i\\ (E),\\quad \\lambda_i \\leftarrow \\max(0, \\lambda_i + \\mu c_i)\\ (I)\\end{gathered}',
    intuition:
      'Add the Lagrangian term λᵀc to the quadratic penalty. After each inner minimization the ' +
      'multipliers move to λᵢ + μcᵢ; once they are right, a moderate μ already makes x⋆ a ' +
      'minimizer, so the subproblems stay well conditioned. The outer loop converges linearly, ' +
      'faster as μ grows.',
    order: 'linear',
    pros: ['Exact solutions with bounded μ', 'Robust on equalities and nonconvex constraints'],
    cons: [
      'Outer loop converges only linearly',
      'Needs tolerances ω, η to be tuned (N&W Alg. 17.4)',
    ],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'stepSize' },
      { tex: '\\mu', key: 'info.mu' },
      { tex: '\\boldsymbol{\\lambda}^{\\text{outer}}', key: 'info.lambda' },
      { tex: '\\omega', key: 'info.omega' },
      { tex: '\\eta', key: 'info.eta' },
    ],
  },
  log_barrier: {
    rule: '\\begin{gathered}\\mathbf{x}^\\star(t) = \\arg\\min_{\\mathbf{x}}\\; t\\,f(\\mathbf{x}) - \\textstyle\\sum_{I} \\log\\big(-c_i(\\mathbf{x})\\big) \\\\ \\mathbf{x} \\leftarrow \\mathbf{x} + \\alpha\\,\\Delta\\mathbf{x}\\ \\text{(Newton)},\\quad t \\leftarrow \\mu\\, t\\end{gathered}',
    intuition:
      'Keep strictly inside the feasible set: the logarithm rises to +∞ at the boundary. Newton ' +
      'centering finds the minimizer x⋆(t) of the barrier problem, and increasing t slides this ' +
      'point along the central path to x⋆, with duality gap |I|/t: the gap shrinks by the factor μ ' +
      'per stage, and Newton converges quadratically inside each stage.',
    order: 'linear',
    pros: ['Every iterate strictly feasible', 'Newton inner steps: few iterations per stage'],
    cons: ['Needs a strictly feasible start', 'Needs Hessians; ill-conditioned near the boundary'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'stepSize' },
      { tex: 't', key: 'info.t' },
      { tex: '|I|/t', key: 'info.gap' },
      // The Newton decrement at x_{k−1} (B&V eq. 9.29), not a multiplier.
      { tex: '\\lambda_{\\mathrm N}^2/2', key: 'info.newton_decrement' },
      { tex: '\\boldsymbol{\\lambda}_k', key: 'info.multipliers' },
    ],
  },
  sqp: {
    rule: '\\begin{gathered}\\mathbf{p}_k = \\arg\\min_{\\mathbf{p}}\\ \\tfrac12\\mathbf{p}^{\\mathsf T}B_k\\mathbf{p} + \\nabla f_k^{\\mathsf T}\\mathbf{p} \\\\ \\text{s.t. } c_i + \\nabla c_i^{\\mathsf T}\\mathbf{p} = 0\\ (E),\\ \\le 0\\ (I) \\\\ \\mathbf{x}_{k+1} = \\mathbf{x}_k + \\alpha_k\\mathbf{p}_k\\end{gathered}',
    intuition:
      'Newton’s method on the KKT conditions: at each iterate solve a quadratic model of the ' +
      'Lagrangian over the linearized constraints, then step along its solution until the ℓ1 ' +
      'merit f + μ‖c⁻‖₁ decreases. The QP also predicts which constraints are active.',
    order: 'superlinear',
    pros: ['Superlinear near x⋆ (quasi-Newton)', 'Handles equalities and inequalities directly'],
    cons: ['Each step solves a QP', 'Linearized constraints can be inconsistent far from x⋆'],
    quantities: [
      { tex: '\\alpha_{k-1}', key: 'stepSize' },
      { tex: '\\mu', key: 'info.mu', label: 'merit penalty' },
      { tex: '\\theta', key: 'info.theta' },
      { tex: '\\boldsymbol{\\lambda}_k', key: 'info.multipliers' },
      { tex: '\\|c^{-}\\|_\\infty', key: 'info.violation' },
    ],
  },
};

registerMethod(
  {
    id: 'projected_gradient',
    family: 'constrained',
    name: 'Projected gradient',
    params: [
      param.float('s_bar', 1.0, {
        min: 1e-4,
        max: 1e2,
        log: true,
        help: 'First trial step s̄ of the Armijo rule along the projection arc.',
        label: 'First trial step',
        tex: '\\bar s',
      }),
      param.float('beta', 0.5, {
        min: 0.1,
        max: 0.9,
        help: 'Backtracking factor β: s ← βs.',
        label: 'Backtracking factor',
        tex: '\\beta',
      }),
      param.float('sigma', 1e-4, {
        min: 1e-6,
        max: 0.5,
        log: true,
        help: 'Sufficient-decrease constant σ of the Armijo rule.',
        label: 'Sufficient decrease',
        tex: '\\sigma',
      }),
      tolParam(1e-6),
      maxIterParam(1000),
    ],
    needs: ['f', 'grad', 'projection'],
    order: 'linear',
    summary:
      'Take a gradient step, project it back onto the feasible set, and backtrack along the ' +
      'projection arc until f decreases enough.',
    references: [
      'Bertsekas (1999), Nonlinear Programming, 2nd ed., §2.3.1 (Armijo rule along the projection arc, eq. 2.43)',
      'Nocedal & Wright (2006), Numerical Optimization, 2nd ed., §16.7',
    ],
  },
  asFn(projectedGradient),
  DOCS.projected_gradient,
);

registerMethod(
  {
    id: 'frank_wolfe',
    family: 'constrained',
    name: 'Frank–Wolfe (conditional gradient)',
    params: [
      param.choice('step', 'open_loop', ['open_loop', 'armijo'], {
        help: 'γ_k = 2/(k+2) (open loop) or Armijo backtracking from γ = 1.',
        label: 'Step rule',
        tex: '\\gamma_k',
      }),
      tolParam(1e-6),
      maxIterParam(1000),
    ],
    needs: ['f', 'grad', 'linear minimization oracle'],
    order: 'sublinear O(1/k)',
    summary:
      'Minimize the linear model of f over the feasible set (a vertex) and move part of the way ' +
      'toward that vertex; no projection needed.',
    references: [
      'Frank & Wolfe (1956), Naval Res. Logist. Q. 3, 95–110',
      'Jaggi (2013), Revisiting Frank–Wolfe, ICML, Algorithm 1 and eq. (2)',
      'Bertsekas (1999), Nonlinear Programming, 2nd ed., §2.2 (conditional gradient)',
    ],
  },
  asFn(frankWolfe),
  DOCS.frank_wolfe,
);

registerMethod(
  {
    id: 'quadratic_penalty',
    family: 'constrained',
    name: 'Quadratic penalty',
    params: [
      param.float('mu0', 1.0, {
        min: 1e-3,
        max: 1e4,
        log: true,
        help: 'Initial penalty parameter μ₀.',
        label: 'Initial penalty',
        tex: '\\mu_0',
      }),
      param.float('rho', 10.0, {
        min: 1.5,
        max: 100.0,
        log: true,
        help: 'Growth factor: μ ← ρμ.',
        label: 'Growth factor',
        tex: '\\rho',
      }),
      param.float('mu_max', 1e10, {
        min: 1e4,
        max: 1e15,
        log: true,
        help: 'Give up when μ would exceed this value.',
        label: 'Penalty cap',
        tex: '\\mu_{\\max}',
      }),
      tolParam(1e-6),
      maxIterParam(1000),
    ],
    needs: ['f', 'grad', 'constraints'],
    order: 'linear in the number of μ updates; violation O(1/μ)',
    summary:
      'Replace the constraints by a quadratic penalty (μ/2)·violation², minimize, and repeat ' +
      'with a larger μ.',
    references: [
      'Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Framework 17.1, eqs. 17.2 and 17.5',
      'Nocedal & Wright (2006), Algorithm 6.1 (BFGS, inner solver)',
    ],
  },
  asFn(quadraticPenalty),
  DOCS.quadratic_penalty,
);

registerMethod(
  {
    id: 'augmented_lagrangian',
    family: 'constrained',
    name: 'Augmented Lagrangian (method of multipliers)',
    params: [
      param.float('mu0', 10.0, {
        min: 2.0,
        max: 1e4,
        log: true,
        help: 'Initial penalty parameter μ₀ (> 1).',
        label: 'Initial penalty',
        tex: '\\mu_0',
      }),
      param.float('mu_factor', 100.0, {
        min: 2.0,
        max: 1000.0,
        log: true,
        help: 'Penalty growth μ ← factor·μ when the constraints did not improve enough.',
        label: 'Penalty growth',
        tex: '\\mu \\leftarrow c\\,\\mu',
      }),
      param.float('mu_max', 1e12, {
        min: 1e4,
        max: 1e16,
        log: true,
        help: 'Give up when μ would exceed this.',
        label: 'Penalty cap',
        tex: '\\mu_{\\max}',
      }),
      tolParam(1e-8),
      maxIterParam(1000),
    ],
    needs: ['f', 'grad', 'constraints'],
    order: 'linear in the outer iterations (rate → 0 as μ grows)',
    summary:
      'Minimize the Lagrangian plus a quadratic penalty, then move the multipliers toward their ' +
      'optimal values; μ need not go to infinity.',
    references: [
      'Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Framework 17.3 and Algorithm 17.4 (update rules for μ, ω, η)',
      'Rockafellar (1973), Math. Program. 5, 354–373 (PHR inequality form)',
      'Birgin & Martínez (2014), Practical Augmented Lagrangian Methods, SIAM, Alg. 4.1',
    ],
  },
  asFn(augmentedLagrangian),
  DOCS.augmented_lagrangian,
);

registerMethod(
  {
    id: 'log_barrier',
    family: 'constrained',
    name: 'Log-barrier interior-point method',
    params: [
      param.float('t0', 1.0, {
        min: 1e-3,
        max: 1e3,
        log: true,
        help: 'Initial barrier parameter t₀.',
        label: 'Initial barrier parameter',
        tex: 't_0',
      }),
      param.float('mu', 10.0, {
        min: 1.5,
        max: 100.0,
        log: true,
        help: 'Barrier growth: t ← μt.',
        label: 'Barrier growth',
        tex: '\\mu\\ (t \\leftarrow \\mu t)',
      }),
      tolParam(1e-6),
      param.float('newton_tol', 1e-10, {
        min: 1e-14,
        max: 1e-2,
        log: true,
        help: 'Centering ends when the Newton decrement λ²/2 ≤ newton_tol.',
        label: 'Centering tolerance',
        tex: '\\lambda_{\\mathrm N}^2/2 \\le',
      }),
      maxIterParam(500),
    ],
    needs: ['f', 'grad', 'hess', 'constraints', 'strictly feasible x0'],
    order: 'outer: gap |I|/t shrinks by μ per stage; inner: Newton (quadratic)',
    summary:
      "Follow the central path: minimize t·f − Σ log(−c_i) with Newton's method for growing t, " +
      'staying strictly inside the feasible set.',
    references: [
      'Boyd & Vandenberghe (2004), Convex Optimization, Algorithm 11.1 (barrier method)',
      "Boyd & Vandenberghe (2004), Algorithms 9.5 and 10.1 (Newton's method), 9.2 (backtracking)",
      'Nocedal & Wright (2006), Algorithm 3.3 (Cholesky with added multiple of the identity)',
    ],
  },
  asFn(logBarrier),
  DOCS.log_barrier,
);

registerMethod(
  {
    id: 'sqp',
    family: 'constrained',
    name: 'SQP (line search, BFGS)',
    params: [tolParam(1e-8), maxIterParam(200)],
    needs: ['f', 'grad', 'constraints'],
    order: 'superlinear (quasi-Newton)',
    summary:
      'Solve a quadratic model of the Lagrangian subject to linearized constraints, then step ' +
      'along its solution with an ℓ1 merit-function line search.',
    references: [
      'Nocedal & Wright (2006), Numerical Optimization, 2nd ed., Algorithm 18.3 (line-search SQP), eqs. 18.11 (QP), 18.36 (penalty update), Procedure 18.2 (damped BFGS)',
      'Nocedal & Wright (2006), Algorithm 18.1 (multiplier update λₖ₊₁ = λ̂)',
      'Powell (1978), A fast algorithm for nonlinearly constrained optimization calculations, LNM 630, 144–157',
      'Goldfarb & Idnani (1983), Math. Program. 27, 1–33 (QP subproblem solver)',
    ],
  },
  asFn(sqp),
  DOCS.sqp,
);
