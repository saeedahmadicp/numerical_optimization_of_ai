/**
 * Newton and Broyden methods for square nonlinear systems F(x) = 0, F: ℝⁿ → ℝⁿ — TS port of
 * `numopt.roots.systems` (src/numopt/roots/systems.py). Same ids, params, stopping tests,
 * evaluation counts, messages and `Step.info` keys (snake_case, as in the Python docstring):
 *
 *   residual: [n]             F(x_k) (every step, including k = 0).
 *   residual_norm: number     ‖F(x_k)‖₂ (every step).
 *   step: [n]                 x_k − x_{k−1} (k ≥ 1).
 *   jacobian: [[n]]           the matrix that produced x_k: J(x_{k−1}) for Newton, the Broyden
 *                             approximation B_{k−1} for Broyden (k ≥ 1).
 *   newton_system only (k ≥ 1):
 *     newton_step: [n]        the full Newton step p = −J⁻¹F(x_{k−1}) (solved, not inverted).
 *     alpha: number           the step length taken (1.0 without damping).
 *     trials: [[α, φ]]        backtracking trials, φ = ½‖F(x_{k−1} + αp)‖² ([] without damping).
 *   broyden only (k ≥ 1):
 *     secant: {s: [n], y: [n]}  s = x_k − x_{k−1}, y = F(x_k) − F(x_{k−1}); the next
 *                             approximation satisfies the secant equation B_k s = y.
 *
 * Stopping test (converged): ‖F(x_k)‖₂ ≤ ftol, or the full step ‖p‖₂ ≤ xtol·(1 + ‖x_{k−1}‖₂)
 * (Broyden also needs the secant correction ‖F‖‖s‖/‖y‖ ≤ the same bound). Failures (singular or
 * non-finite Jacobian, non-finite F, divergence, a failed backtracking search, an undefined
 * Broyden update, a Broyden stall, the iteration budget) return `converged: false`; nothing throws
 * on numerical breakdown.
 *
 * Linear algebra mirrors NumPy/LAPACK: `np.linalg.solve` → LU with partial pivoting (for n = 2
 * with OpenBLAS's rounding, fused multiply-adds included, so long runs replay bit for bit),
 * `s @ H` with a fused accumulation, `np.linalg.cond` → σ_max/σ_min (dgesdd's 2×2 path for n = 2,
 * a one-sided Jacobi SVD otherwise).
 */
import { registerMethod, param } from '../../core/registry';
import type { Matrix, MethodFn, Problem, Result, Step, Vector } from '../../core/types';
import { luFactor, luSolve } from '../../core/linalg';

/** Machine epsilon of float64. */
export const EPS = Number.EPSILON;
/** Divergence is declared when ‖x_k‖ > DIVERGENCE_FACTOR · max(1, ‖x₀‖). */
export const DIVERGENCE_FACTOR = 1e12;
/** Armijo constant c₁ for the merit function φ = ½‖F‖² (Nocedal & Wright, §3.1). */
export const ARMIJO_C1 = 1e-4;
/** Backtracking halves α at most this many times (α ≥ 2⁻³⁰ ≈ 9.3e-10). */
export const MAX_BACKTRACKS = 30;
/** Central-difference step factor ε^{1/3} (numopt.core.diff._H_CENTRAL). */
const H_CENTRAL = Math.pow(EPS, 1.0 / 3.0);

type SystemLike = Pick<Problem<Vector>, 'id' | 'dim' | 'x0'> & {
  f: (x: Vector) => unknown;
  jac?: (x: Vector) => Matrix;
};

// ── Small helpers (NumPy semantics) ─────────────────────────────────────────────────────────

/** True when every entry (of numbers, vectors or matrices) is finite. */
function finite(...values: unknown[]): boolean {
  const ok = (v: unknown): boolean =>
    typeof v === 'number' ? Number.isFinite(v) : Array.isArray(v) ? v.every(ok) : false;
  return values.every(ok);
}

/**
 * ‖v‖₂ without overflow or underflow: scale by max|vᵢ| first (LAPACK dnrm2 idea, Higham 2002,
 * ch. 27). Non-finite entries give ∞ or NaN as usual (NumPy's `max` propagates NaN).
 */
export function scaledNorm(v: readonly number[]): number {
  if (v.length === 0) return 0.0;
  let m = -Infinity;
  for (const x of v) {
    const a = Math.abs(x);
    if (Number.isNaN(a)) {
      m = NaN;
      break;
    }
    if (a > m) m = a;
  }
  if (m === 0.0 || !Number.isFinite(m)) return m;
  let s = 0.0;
  for (const x of v) {
    const w = x / m;
    s += w * w;
  }
  return m * Math.sqrt(s);
}

/** Python's `f"{v:.{p}g}"`. */
export function pyG(v: number, p = 3): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, expStr] = v.toExponential(p - 1).split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= p) {
    const m = mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant;
    return `${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  const s = v.toFixed(Math.max(0, p - 1 - exp));
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** An approximation of `np.array2string(x, precision=6)` for messages: `[ 1.  -0.5]`. */
export function pyArray(x: readonly number[]): string {
  const sci = x.some(
    (v) => Number.isFinite(v) && v !== 0 && (Math.abs(v) >= 1e8 || Math.abs(v) < 1e-4),
  );
  const parts = x.map((v) => {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (sci) return v.toExponential(6).replace(/e([+-])(\d)$/, 'e$10$2');
    const s = v.toFixed(6).replace(/0+$/, '');
    return s;
  });
  const width = Math.max(...parts.map((s) => s.replace(/^-/, '').length));
  return `[${parts
    .map((s) => (s.startsWith('-') ? s : ' ' + s).padEnd(width + 1))
    .join(' ')
    .trimEnd()}]`;
}

/**
 * Singular values of a square matrix, descending (one-sided Jacobi, Demmel & Veselić 1992).
 * Accurate to a few ulps relative to σ_max for the small matrices used here.
 */
export function singularValues(A: Matrix): number[] {
  const n = A.length;
  // Work on columns: U = A, rotate pairs of columns until they are mutually orthogonal.
  const U = A.map((r) => r.slice());
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) {
        let a = 0,
          b = 0,
          c = 0;
        for (let i = 0; i < n; i++) {
          a += U[i][p] * U[i][p];
          b += U[i][q] * U[i][q];
          c += U[i][p] * U[i][q];
        }
        if (c === 0 || Math.abs(c) <= EPS * Math.sqrt(a * b)) continue;
        off = Math.max(off, Math.abs(c) / Math.sqrt(a * b));
        const zeta = (b - a) / (2 * c);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const cs = 1 / Math.sqrt(1 + t * t),
          sn = cs * t;
        for (let i = 0; i < n; i++) {
          const up = U[i][p],
            uq = U[i][q];
          U[i][p] = cs * up - sn * uq;
          U[i][q] = sn * up + cs * uq;
        }
      }
    if (off <= EPS) break;
  }
  const s: number[] = [];
  for (let j = 0; j < n; j++) {
    const col: number[] = [];
    for (let i = 0; i < n; i++) col.push(U[i][j]);
    s.push(scaledNorm(col));
  }
  return s.sort((x, y) => y - x);
}

/** Veltkamp split of a double into two 26-bit halves (Dekker 1971). */
function split(a: number): [number, number] {
  const c = 134217729 * a; // 2^27 + 1
  const hi = c - (c - a);
  return [hi, a - hi];
}

/**
 * a·b + c with a single rounding (an emulated fused multiply-add): Dekker's exact product plus
 * Knuth's TwoSum. Correctly rounded except in rare double-rounding ties, which is enough here.
 */
export function fma(a: number, b: number, c: number): number {
  const p = a * b;
  if (!Number.isFinite(p) || !Number.isFinite(c) || Math.abs(p) > 1e300) return p + c;
  const [ah, al] = split(a);
  const [bh, bl] = split(b);
  const pe = al * bl - (p - ah * bh - al * bh - ah * bl); // p + pe = a·b exactly
  const s = p + c;
  const bb = s - p;
  const se = p - (s - bb) + (c - bb); // s + se = p + c exactly
  return s + (se + pe);
}

/** LAPACK dlapy2: √(x² + y²) without destructive overflow. */
function dlapy2(x: number, y: number): number {
  const xa = Math.abs(x),
    ya = Math.abs(y);
  const w = Math.max(xa, ya),
    z = Math.min(xa, ya);
  if (z === 0 || w > Number.MAX_VALUE) return w;
  return w * Math.sqrt(1 + (z / w) * (z / w));
}

/** LAPACK dlas2: singular values [σ_min, σ_max] of the upper triangular [[f, g], [0, h]]. */
function dlas2(f: number, g: number, h: number): [number, number] {
  const fa = Math.abs(f),
    ga = Math.abs(g),
    ha = Math.abs(h);
  const fhmn = Math.min(fa, ha),
    fhmx = Math.max(fa, ha);
  if (fhmn === 0) {
    const smax =
      fhmx === 0
        ? ga
        : Math.max(fhmx, ga) * Math.sqrt(1 + (Math.min(fhmx, ga) / Math.max(fhmx, ga)) ** 2);
    return [0, smax];
  }
  if (ga < fhmx) {
    const as = 1 + fhmn / fhmx;
    const at = (fhmx - fhmn) / fhmx;
    const au = (ga / fhmx) * (ga / fhmx);
    const c = 2 / (Math.sqrt(as * as + au) + Math.sqrt(at * at + au));
    return [fhmn * c, fhmx / c];
  }
  const au = fhmx / ga;
  if (au === 0) return [(fhmn * fhmx) / ga, ga];
  const as = 1 + fhmn / fhmx;
  const at = (fhmx - fhmn) / fhmx;
  const c = 1 / (Math.sqrt(1 + as * au * (as * au)) + Math.sqrt(1 + at * au * (at * au)));
  const smin = fhmn * c * au;
  return [smin + smin, ga / (c + c)];
}

/**
 * Singular values of a 2×2 matrix the way `np.linalg.svd(A, compute_uv=False)` gets them
 * (LAPACK dgesdd → dgebd2 → dlasq1 → dlas2): one Householder reflection (dlarfg/dlarf) makes A
 * upper triangular, then dlas2. Near-singular matrices then get the same σ_min as NumPy, which
 * matters for the cond₂ > 1/ε test.
 */
export function singularValues2(A: Matrix): [number, number] {
  const a11 = A[0][0],
    a21 = A[1][0],
    a12 = A[0][1],
    a22 = A[1][1];
  let d1 = a11,
    e = a12,
    d2 = a22;
  const xnorm = Math.abs(a21);
  if (xnorm !== 0) {
    const beta = -Math.sign(a11 || 1) * dlapy2(a11, a21);
    const tau = (beta - a11) / beta;
    const v2 = a21 * (1 / (a11 - beta));
    // H = I − τ v vᵀ with v = (1, v2), applied to the second column (dlarf: w = vᵀc, then the
    // rank-one update c += v·(−τw), which OpenBLAS's dger evaluates with a fused multiply-add).
    const w = a12 + v2 * a22;
    const t = -tau * w;
    e = a12 + t;
    d2 = fma(v2, t, a22);
    d1 = beta;
  }
  const [smin, smax] = dlas2(d1, e, d2);
  return [smax, smin];
}

/** `np.linalg.cond(J)` (2-norm): σ_max/σ_min (∞ when σ_min = 0, NaN for non-finite J). */
export function cond2(J: Matrix): number {
  const s = J.length === 2 ? singularValues2(J) : singularValues(J);
  const smin = s[s.length - 1];
  if (!Number.isFinite(s[0]) || Number.isNaN(smin)) return NaN;
  return smin === 0 ? Infinity : s[0] / smin;
}

/** Return a reason when J is not finite or numerically singular (cond₂ > 1/ε). */
function singular(J: Matrix): string | null {
  if (!finite(J)) return 'the Jacobian has non-finite entries';
  const c = cond2(J);
  if (!Number.isFinite(c) || c > 1.0 / EPS)
    return `the Jacobian is singular to working precision (cond₂ = ${pyG(c)})`;
  return null;
}

/**
 * 2×2 LU with partial pivoting the way OpenBLAS dgetrf/dgetrs round it (the kernels NumPy calls):
 * the multiplier is a21·(1/a11), and the Schur complement and the forward and back substitutions
 * are fused multiply-adds. `recip` divides by the pivots as x·(1/u) (dtrsm, several right-hand
 * sides: `np.linalg.solve(B, I)`); otherwise x/u (one right-hand side). Measured against NumPy
 * 2.5 / OpenBLAS on 8000 random systems: bit-identical. Plain LU agrees to ~1 ulp, which long
 * runs on ill-conditioned problems amplify (Broyden on rosenbrock_system ends elsewhere).
 */
function solve2(A: Matrix, b: readonly number[], recip: boolean): Vector {
  let [[a11, a12], [a21, a22]] = A;
  let [b1, b2] = b;
  if (Math.abs(a21) > Math.abs(a11)) {
    [a11, a12, a21, a22] = [a21, a22, a11, a12];
    [b1, b2] = [b2, b1];
  }
  const l = a21 * (1 / a11);
  const u22 = fma(-l, a12, a22);
  const y2 = fma(-l, b1, b2);
  const x2 = recip ? y2 * (1 / u22) : y2 / u22;
  const t = fma(-a12, x2, b1);
  return [recip ? t * (1 / a11) : t / a11, x2];
}

/** `np.linalg.solve(A, b)` for a matrix already known to be nonsingular. */
function solveLU(A: Matrix, b: readonly number[]): Vector {
  if (A.length === 2) return solve2(A, b, false);
  return luSolve(luFactor(A), b);
}

/** `np.linalg.solve(A, np.eye(n))`: the inverse, column by column. */
function inverse(A: Matrix): Matrix {
  const n = A.length;
  const unit = (j: number) => Array.from({ length: n }, (_, i) => (i === j ? 1 : 0));
  const lu = n === 2 ? null : luFactor(A);
  const cols = Array.from({ length: n }, (_, j) =>
    lu ? luSolve(lu, unit(j)) : solve2(A, unit(j), true),
  );
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}

/**
 * The row vector `s @ H` as NumPy computes it (dgemv on the transposed matrix): each entry
 * accumulates with a fused multiply-add, (sᵀH)_j = fma(s₁, H₁ⱼ, s₀H₀ⱼ) for n = 2.
 */
function vecMat(s: readonly number[], H: Matrix): number[] {
  return Array.from({ length: s.length }, (_, j) =>
    H.reduce((acc, row, i) => (i === 0 ? s[0] * row[j] : fma(s[i], row[j], acc)), 0),
  );
}

const matvec = (A: Matrix, x: readonly number[]) =>
  A.map((row) => row.reduce((s, a, j) => s + a * x[j], 0));
const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

/** Central-difference Jacobian (numopt.core.diff.jacobian): 2n + 1 evaluations of F. */
function fdJacobian(F: (x: Vector) => Vector, x: Vector): Matrix {
  const f0 = F(x);
  const J: Matrix = f0.map(() => new Array<number>(x.length).fill(0));
  for (let i = 0; i < x.length; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => (j === i ? v + h : v + 0.0));
    const xm = x.map((v, j) => (j === i ? v - h : v - 0.0));
    const fp = F(xp),
      fm = F(xm);
    for (let r = 0; r < f0.length; r++) J[r][i] = (fp[r] - fm[r]) / (2.0 * h);
  }
  return J;
}

// ── Counted residual and Jacobian with the finite-difference fallback ───────────────────────

class CountedSystem {
  nF = 0;
  nJ = 0;
  readonly prob: SystemLike;
  readonly extra: Record<string, unknown>;
  constructor(prob: SystemLike, extra: Record<string, unknown>) {
    this.prob = prob;
    this.extra = extra;
    if (!prob.jac) extra.jacobian = 'finite_difference';
  }

  /** F(x) (counted); an exception inside F counts as a NaN value. */
  residual = (x: Vector): Vector => {
    this.nF++;
    try {
      const v = this.prob.f(x);
      const arr = Array.isArray(v) ? (v as unknown[]).flat(Infinity) : [v];
      return arr.map((e) => Number(e));
    } catch {
      return x.map(() => NaN);
    }
  };

  jacobian(x: Vector, exact = true): Matrix {
    try {
      if (exact && this.prob.jac) {
        this.nJ++;
        return this.prob.jac(x).map((r) => r.map(Number));
      }
      return fdJacobian(this.residual, x);
    } catch {
      return x.map(() => x.map(() => NaN));
    }
  }
}

function setup(problem: SystemLike | ((x: Vector) => Vector), x0: unknown): [SystemLike, Vector] {
  let prob: SystemLike;
  if (typeof problem === 'function') {
    if (x0 === undefined || x0 === null)
      throw new Error('a starting point x0 is required for a bare callable F');
    prob = { id: 'custom', dim: 0, f: problem, x0: null };
  } else if (problem && typeof problem === 'object' && typeof problem.f === 'function') {
    prob = problem;
  } else {
    throw new TypeError('problem must be a numopt Problem or a callable F(x)');
  }
  const raw = x0 ?? prob.x0;
  if (raw === undefined || raw === null)
    throw new Error(`${prob.id}: no starting point given and the problem has no default x0`);
  const x = (Array.isArray(raw) ? (raw as unknown[]).flat(Infinity) : [raw]).map(Number);
  if (prob.dim && x.length !== prob.dim)
    throw new Error(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  return [prob, x];
}

/** Shared tests after a step; returns [converged, message] to stop, else null. */
function check(
  xNew: Vector,
  FNew: Vector,
  p: Vector,
  xOld: Vector,
  x0Norm: number,
  ftol: number,
  xtol: number,
  secant?: [Vector, Vector],
): [boolean, string] | null {
  if (!finite(xNew, FNew)) return [false, 'diverged: non-finite x or F(x)'];
  const r = scaledNorm(FNew);
  if (r <= ftol) return [true, `‖F(x)‖₂ = ${pyG(r)} ≤ ftol`];
  const pNorm = scaledNorm(p);
  const tol = xtol * (1.0 + scaledNorm(xOld));
  if (!secant) {
    if (pNorm <= tol) return [true, `step ‖p‖₂ = ${pyG(pNorm)} ≤ xtol·(1 + ‖x‖₂)`];
  } else {
    const [s, y] = secant;
    const sNorm = scaledNorm(s),
      yNorm = scaledNorm(y);
    if (sNorm === 0.0)
      return [
        false,
        `stalled: the step ‖p‖₂ = ${pyG(pNorm)} does not change x in floating point ` +
          `while ‖F(x)‖₂ = ${pyG(r)} > ftol (the Broyden matrix no longer models F)`,
      ];
    const correction = yNorm > 0.0 ? r * (sNorm / yNorm) : Infinity;
    if (pNorm <= tol && correction <= tol)
      return [
        true,
        `step ‖p‖₂ = ${pyG(pNorm)} ≤ xtol·(1 + ‖x‖₂) and the secant correction ` +
          `‖F‖‖s‖/‖y‖ = ${pyG(correction)} ≤ xtol·(1 + ‖x‖₂)`,
      ];
  }
  const bound = DIVERGENCE_FACTOR * Math.max(1.0, x0Norm);
  if (scaledNorm(xNew) > bound) return [false, `diverged: ‖x‖₂ > ${pyG(bound)}`];
  return null;
}

function startStep(x: Vector, Fx: Vector): Step {
  const r = scaledNorm(Fx);
  return {
    k: 0,
    x: x.slice(),
    fun: r,
    gradNorm: null,
    stepSize: null,
    info: { residual: Fx.slice(), residual_norm: r },
  };
}

const sub = (a: readonly number[], b: readonly number[]) => a.map((v, i) => v - b[i]);

function makeResult(
  method: string,
  x: Vector,
  Fx: Vector,
  converged: boolean,
  message: string,
  k: number,
  sys: CountedSystem,
  trace: Step[],
): Result {
  return {
    method,
    x: x.slice(),
    fun: scaledNorm(Fx),
    converged,
    message,
    nIter: k,
    nFev: sys.nF,
    nGev: sys.nJ,
    nHev: 0,
    trace,
    extra: sys.extra,
  };
}

// ── Newton's method for systems ───────────────────────────────────────────────────────────────

/** Newton's method for F(x) = 0 (Nocedal & Wright Alg. 11.1), optionally damped (§11.2). */
export const newtonSystem: MethodFn<SystemLike> = (problem, options) => {
  const ftol = Number(options.ftol ?? 1e-10);
  const xtol = Number(options.xtol ?? 1e-12);
  const maxIter = Number(options.max_iter ?? 100);
  const damping = Boolean(options.damping ?? false);
  const [prob, x0] = setup(problem, options.x0);
  let x = x0;
  const extra: Record<string, unknown> = {};
  const sys = new CountedSystem(prob, extra);
  let Fx = sys.residual(x);
  if (Fx.length !== x.length)
    throw new Error(
      `newton_system needs a square system; F has ${Fx.length} rows, x has ${x.length}`,
    );
  const x0Norm = scaledNorm(x);
  const trace: Step[] = [startStep(x, Fx)];
  const result = (converged: boolean, message: string, k: number) =>
    makeResult('newton_system', x, Fx, converged, message, k, sys, trace);

  if (!finite(Fx)) return result(false, 'F(x₀) is not finite', 0);
  if (scaledNorm(Fx) <= ftol) return result(true, `‖F(x)‖₂ = ${pyG(scaledNorm(Fx))} ≤ ftol`, 0);
  let k = 0;
  for (;;) {
    if (k === maxIter) return result(false, `reached max_iter=${maxIter}`, k);
    const J = sys.jacobian(x);
    const why = singular(J);
    if (why !== null) return result(false, `${why} at x = ${pyArray(x)}`, k);
    const p = solveLU(
      J,
      Fx.map((v) => -v),
    );
    let alpha = 1.0;
    const trials: [number, number][] = [];
    let xNew = x.map((v, i) => v + p[i]);
    let FNew = sys.residual(xNew);
    if (damping) {
      const r0 = scaledNorm(Fx);
      const phi0 = 0.5 * r0 * r0;
      for (let i = 0; i <= MAX_BACKTRACKS; i++) {
        const rTry = finite(FNew) ? scaledNorm(FNew) : Infinity;
        const phi = 0.5 * rTry * rTry;
        trials.push([alpha, phi]);
        if (phi <= (1.0 - 2.0 * ARMIJO_C1 * alpha) * phi0) break;
        if (i === MAX_BACKTRACKS)
          return result(
            false,
            'line search failed: no sufficient decrease of ½‖F‖² along the ' +
              'Newton step (x may be near a local minimizer of ‖F‖ that is not a root)',
            k,
          );
        alpha *= 0.5;
        const a = alpha;
        xNew = x.map((v, j) => v + a * p[j]);
        FNew = sys.residual(xNew);
      }
    }
    k += 1;
    const rNew = scaledNorm(FNew);
    const step = sub(xNew, x);
    trace.push({
      k,
      x: xNew.slice(),
      fun: rNew,
      gradNorm: null,
      stepSize: scaledNorm(step),
      info: {
        residual: FNew.slice(),
        residual_norm: rNew,
        step,
        jacobian: J,
        newton_step: p,
        alpha,
        trials,
      },
    });
    const stop = check(xNew, FNew, p, x, x0Norm, ftol, xtol);
    x = xNew;
    Fx = FNew;
    if (stop !== null) return result(stop[0], stop[1], k);
  }
};

// ── Broyden's method ("good" Broyden, inverse update) ─────────────────────────────────────────

/** Broyden's "good" method with the inverse (Sherman–Morrison) update (Broyden 1965, method 1). */
export const broyden: MethodFn<SystemLike> = (problem, options) => {
  const ftol = Number(options.ftol ?? 1e-10);
  const xtol = Number(options.xtol ?? 1e-12);
  const maxIter = Number(options.max_iter ?? 100);
  const jacobian0 = String(options.jacobian0 ?? 'exact');
  if (jacobian0 !== 'exact' && jacobian0 !== 'finite_difference')
    throw new Error(
      `jacobian0 must be 'exact' or 'finite_difference', got ${JSON.stringify(jacobian0)}`,
    );
  const [prob, x0] = setup(problem, options.x0);
  let x = x0;
  const extra: Record<string, unknown> = {};
  const sys = new CountedSystem(prob, extra);
  if (jacobian0 === 'finite_difference') extra.jacobian = 'finite_difference';
  let Fx = sys.residual(x);
  if (Fx.length !== x.length)
    throw new Error(`broyden needs a square system; F has ${Fx.length} rows, x has ${x.length}`);
  const x0Norm = scaledNorm(x);
  const trace: Step[] = [startStep(x, Fx)];
  const result = (converged: boolean, message: string, k: number) =>
    makeResult('broyden', x, Fx, converged, message, k, sys, trace);

  if (!finite(Fx)) return result(false, 'F(x₀) is not finite', 0);
  if (scaledNorm(Fx) <= ftol) return result(true, `‖F(x)‖₂ = ${pyG(scaledNorm(Fx))} ≤ ftol`, 0);
  let B = sys.jacobian(x, jacobian0 === 'exact');
  const why = singular(B);
  if (why !== null) return result(false, `initial ${why}`, 0);
  // H₀ = B₀⁻¹ as the solve B₀H₀ = I (column by column).
  let H: Matrix = inverse(B);
  let k = 0;
  for (;;) {
    if (k === maxIter) return result(false, `reached max_iter=${maxIter}`, k);
    const HF = matvec(H, Fx);
    const p = HF.map((v) => -v);
    const xNew = x.map((v, i) => v + p[i]);
    const FNew = sys.residual(xNew);
    const s = sub(xNew, x); // the step actually taken (differs from p by rounding)
    const y = sub(FNew, Fx);
    k += 1;
    const rNew = scaledNorm(FNew);
    trace.push({
      k,
      x: xNew.slice(),
      fun: rNew,
      gradNorm: null,
      stepSize: scaledNorm(s),
      info: {
        residual: FNew.slice(),
        residual_norm: rNew,
        step: s.slice(),
        jacobian: B.map((r) => r.slice()),
        secant: { s: s.slice(), y: y.slice() },
      },
    });
    const stop = check(xNew, FNew, p, x, x0Norm, ftol, xtol, [s, y]);
    x = xNew;
    Fx = FNew;
    if (stop !== null) return result(stop[0], stop[1], k);
    const Hy = matvec(H, y);
    const denom = dot(s, Hy);
    if (!Number.isFinite(denom) || Math.abs(denom) <= EPS * (scaledNorm(s) * scaledNorm(Hy)))
      return result(false, 'Broyden update undefined: sᵀH y = 0 (H would become singular)', k);
    // sᵀH (row vector): (sᵀH)_j = Σᵢ sᵢ H_ij.
    const sH = vecMat(s, H);
    const u = sub(s, Hy);
    H = H.map((row, i) => row.map((h, j) => h + (u[i] * sH[j]) / denom));
    const Bs = matvec(B, s);
    const ss = dot(s, s);
    const v = sub(y, Bs);
    B = B.map((row, i) => row.map((b, j) => b + (v[i] * s[j]) / ss));
  }
};

// ── Registration ────────────────────────────────────────────────────────────────────────────

const SYSTEM_PARAMS = [
  param.float('ftol', 1e-10, {
    min: 1e-15,
    max: 1e-2,
    log: true,
    help: 'Stop when ‖F(x)‖₂ ≤ ftol.',
    label: 'Residual tolerance',
    tex: '\\|F\\|_2 \\le',
  }),
  param.float('xtol', 1e-12, {
    min: 1e-16,
    max: 1e-2,
    log: true,
    help: 'Stop when the full step ‖p‖₂ ≤ xtol·(1 + ‖x‖₂).',
    label: 'Step tolerance',
    tex: '\\|\\mathbf{p}\\|_2 \\le',
  }),
  param.int('max_iter', 100, {
    min: 1,
    max: 10_000,
    help: 'Iteration limit.',
    label: 'Iteration budget',
  }),
];

registerMethod(
  {
    id: 'newton_system',
    family: 'systems',
    name: "Newton's method for systems",
    params: [
      ...SYSTEM_PARAMS,
      param.bool('damping', false, {
        help: 'Backtrack along the Newton step until ½‖F‖² decreases enough (Armijo).',
        label: 'Damping (backtracking on ½‖F‖²)',
      }),
    ],
    needs: ['f', 'jac', 'x0'],
    order: 'quadratic (nonsingular root)',
    summary: 'Solve the linearization J(x)p = −F(x) and step to x + p (optionally damped).',
    references: [
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), Alg. 11.1',
      'Nocedal & Wright (2006), §11.2 (merit function ½‖F‖², line search) and Alg. 3.1',
      'Dennis & Schnabel, Numerical Methods for Unconstrained Optimization and ' +
        'Nonlinear Equations (1983), §6.5',
    ],
  },
  newtonSystem,
  {
    rule:
      '\\begin{aligned} J(\\mathbf{x}_k)\\,\\mathbf{p}_k &= -F(\\mathbf{x}_k) \\\\ ' +
      '\\mathbf{x}_{k+1} &= \\mathbf{x}_k + \\alpha_k\\,\\mathbf{p}_k \\end{aligned}',
    intuition:
      'Replace each component Fᵢ by its tangent plane at 𝐱ₖ. Each plane meets zero in a line; ' +
      'the two lines cross at the Newton point, and the method jumps there.',
    order: 'quadratic',
    pros: [
      'Quadratic convergence near a root with nonsingular J',
      'Affine invariant: unchanged by a linear change of variables',
    ],
    cons: [
      'Needs J and a linear solve every step',
      'Undefined where J is singular; can cycle or diverge from far away',
    ],
    quantities: [
      { tex: '\\|F(\\mathbf{x}_k)\\|_2', key: 'info.residual_norm' },
      { tex: '\\alpha_{k-1}', key: 'info.alpha' },
      { tex: '\\mathbf{p}_{k-1}', key: 'info.newton_step' },
      { tex: '\\|\\mathbf{x}_k - \\mathbf{x}_{k-1}\\|_2', key: 'stepSize' },
    ],
  },
);

registerMethod(
  {
    id: 'broyden',
    family: 'systems',
    name: "Broyden's method (good Broyden)",
    params: [
      ...SYSTEM_PARAMS,
      param.choice('jacobian0', 'exact', ['exact', 'finite_difference'], {
        help: 'Initial Jacobian B₀: the exact J(x₀) or a central-difference estimate.',
        label: 'Initial Jacobian B₀',
      }),
    ],
    needs: ['f', 'x0'],
    order: 'superlinear',
    summary: 'A quasi-Newton method: one Jacobian at the start, then rank-one secant updates.',
    references: [
      'Broyden (1965), Math. Comp. 19(92), 577–593 (method 1, inverse update)',
      'Nocedal & Wright, Numerical Optimization (2nd ed., 2006), §11.1',
      'Dennis & Schnabel (1983), §8.1',
    ],
  },
  broyden,
  {
    rule:
      '\\begin{aligned} \\mathbf{x}_{k+1} &= \\mathbf{x}_k - H_k F(\\mathbf{x}_k) \\\\ ' +
      'H_{k+1} &= H_k + \\frac{(\\mathbf{s}_k - H_k\\mathbf{y}_k)\\,\\mathbf{s}_k^{\\mathsf T} H_k}' +
      '{\\mathbf{s}_k^{\\mathsf T} H_k \\mathbf{y}_k} \\end{aligned}',
    intuition:
      'Newton with a borrowed Jacobian: B₀ = J(𝐱₀) once, then a rank-one correction so that the ' +
      'model reproduces the last observed change of F, B 𝐬ₖ = 𝐲ₖ. One F evaluation per step, no ' +
      'derivatives, no linear solve.',
    order: 'superlinear',
    pros: ['One evaluation of F per step; no Jacobian after B₀', 'O(n²) work per step'],
    cons: [
      'Superlinear only near the root; no globalization',
      'The model can drift far from J, so a small step alone is not proof of convergence',
    ],
    quantities: [
      { tex: '\\|F(\\mathbf{x}_k)\\|_2', key: 'info.residual_norm' },
      { tex: '\\mathbf{s}_{k-1}', key: 'info.step' },
      { tex: '\\|\\mathbf{s}_{k-1}\\|_2', key: 'stepSize' },
    ],
  },
);
