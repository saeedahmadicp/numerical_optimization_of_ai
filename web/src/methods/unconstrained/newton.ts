/**
 * Newton's method — TS port of `numopt.unconstrained.newton` (src/numopt/unconstrained/newton.py).
 *
 * At the iterate x_k the quadratic model m_k(p) = f(x_k) + ∇f(x_k)ᵀp + ½ pᵀ∇²f(x_k) p has the
 * stationary point given by the Newton equations ∇²f(x_k) p = −∇f(x_k) (N&W eq. 3.30):
 *
 *   pure_newton      x_{k+1} = x_k + p_k (α = 1), no globalization.
 *   damped_newton    x_{k+1} = x_k + α_k p_k, line search from α = 1; −∇f where p_k fails the
 *                    descent test cos θ > 1e-8 or ∇²f is numerically singular.
 *   modified_newton  N&W Alg. 3.2 with Alg. 3.3: (∇²f + τI) p = −∇f, τ ≥ 0 by Cholesky retries.
 *
 * Every rule of the Python module is kept: the spectral solve p = −Q Λ⁻¹ Qᵀ ∇f (a Jacobi
 * eigensolver stands in for `numpy.linalg.eigh`), the eigenvalue tolerance (analytic vs.
 * central-difference ∇²f), the second-order stopping test and its messages, the divergence test,
 * the evaluation counts (finite-difference fallbacks are counted through f and ∇f) and the info
 * keys (snake_case, as documented in the Python module docstring):
 *
 *   every method, every step: grad, hess (n ≤ 2, else null), hess_eigs, direction, alpha, trials,
 *   descent; damped_newton: direction_type; modified_newton: tau, chol_attempts.
 *
 * Shared helpers for the n-D descent ports (counted oracle, central differences, Python-style
 * validation errors, the scaled cos θ) are exported for `quasi_newton.ts`.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Matrix, MethodFn, Point, Problem, Result, Step, Vector } from '../../core/types';
import { formatG, formatRepr, search } from '../line_search/methods';

// ---------------------------------------------------------------------------------------
// Constants (Python module level)
// ---------------------------------------------------------------------------------------

/** Machine epsilon of float64. */
export const EPS = Number.EPSILON;
/** A finite-difference ∇f with |∇fᵀp| below this many ulps of f cannot make progress. */
const FD_STALL = 10.0;
/** Relative accuracy ε^{1/3} of a central-difference ∇²f (`EPS ** (1.0 / 3.0)` in Python). */
const FD_HESS_REL = EPS ** (1.0 / 3.0);
/** Descent test cos θ > η for `damped_newton`. */
const DESCENT_COS = 1e-8;
/** `pure_newton` reports divergence when ‖x_k‖∞ > DIVERGENCE_FACTOR · max(1, ‖x_0‖∞). */
const DIVERGENCE_FACTOR = 1e8;
/** `f"{_DIVERGENCE_FACTOR:.0e}"` in Python. */
const DIVERGENCE_FACTOR_TEXT = '1e+08';
/** N&W Alg. 3.3 gives up after this many failed factorizations with τ ≥ β. */
const MAX_SHIFT_DOUBLINGS = 64;

export const LINE_SEARCHES = ['backtracking', 'strong_wolfe', 'weak_wolfe', 'goldstein'] as const;

// ---------------------------------------------------------------------------------------
// Shared helpers (numopt.core.counting / numopt.core.diff equivalents)
// ---------------------------------------------------------------------------------------

/** Python `ValueError` raised on invalid input (bad params, a non-finite start). */
export class MethodInputError extends Error {
  override name = 'ValueError';
}

/** Python `repr` of a tuple of strings: `('a', 'b')`. */
export function tupleRepr(items: readonly string[]): string {
  return `(${items.map((s) => `'${s}'`).join(', ')})`;
}

/** `numpy.str` of a 1-D float array, close enough for error messages: `[1. 2.5]`. */
function arrayStr(v: readonly number[]): string {
  return `[${v.map((e) => formatRepr(e)).join(' ')}]`;
}

/** `as_vector`: a fresh flat number[] (numbers, nested arrays and typed arrays are accepted). */
export function toVector(v: unknown): Vector {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return (v as unknown[]).flat(Infinity).map(Number);
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>, Number);
  return [Number(v)];
}

/** `finite(*values)`: every entry of every scalar / vector / matrix is finite. */
export function finite(...values: (number | readonly number[] | readonly (readonly number[])[])[]) {
  for (const v of values) {
    if (typeof v === 'number') {
      if (!Number.isFinite(v)) return false;
      continue;
    }
    for (const e of v) {
      if (typeof e === 'number') {
        if (!Number.isFinite(e)) return false;
      } else if (!finite(e)) return false;
    }
  }
  return true;
}

/** `float(np.max(np.abs(v)))` (NaN propagates like numpy). */
export function maxAbs(v: readonly number[]): number {
  let m = -Infinity;
  for (const e of v) {
    const a = Math.abs(e);
    if (Number.isNaN(a)) return NaN;
    if (a > m) m = a;
  }
  return m;
}

/** `float(a @ b)` for 1-D arrays. */
export function dotp(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** `float(np.linalg.norm(v))` (√(vᵀv), no scaling). */
export function norm2(v: readonly number[]): number {
  return Math.sqrt(dotp(v, v));
}

/** `x + alpha * p` elementwise, in the Python order. */
export function step(x: readonly number[], alpha: number, p: readonly number[]): Vector {
  const out = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] + alpha * p[i];
  return out;
}

/**
 * cos θ = −gᵀp / (‖g‖‖p‖); NaN when either vector is zero or not finite. Both vectors are scaled
 * by their largest entries first so that the norms cannot underflow.
 */
export function cosAngle(g: readonly number[], p: readonly number[]): number {
  const gMax = maxAbs(g);
  const pMax = maxAbs(p);
  if (!(Number.isFinite(gMax) && Number.isFinite(pMax) && gMax > 0.0 && pMax > 0.0)) return NaN;
  const gHat = g.map((v) => v / gMax);
  const pHat = p.map((v) => v / pMax);
  return -dotp(gHat, pHat) / (norm2(gHat) * norm2(pHat));
}

/** ε^{1/3}: the central-difference step factor of `numopt.core.diff`. */
const H_CENTRAL = EPS ** (1.0 / 3.0);

/** `numopt.core.diff.gradient`: central-difference ∇f (2n calls of f). */
export function fdGradient(f: (x: Vector) => number, x: readonly number[]): Vector {
  const g = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => v + (j === i ? h : 0.0));
    const xm = x.map((v, j) => v - (j === i ? h : 0.0));
    g[i] = (f(xp) - f(xm)) / (2.0 * h);
  }
  return g;
}

/** `numopt.core.diff.hessian`: symmetrized central difference of ∇f (2n calls of ∇f). */
export function fdHessian(grad: (x: Vector) => Vector, x: readonly number[]): Matrix {
  const n = x.length;
  const H: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => v + (j === i ? h : 0.0));
    const xm = x.map((v, j) => v - (j === i ? h : 0.0));
    const gp = grad(xp);
    const gm = grad(xm);
    for (let r = 0; r < n; r++) H[r][i] = (gp[r] - gm[r]) / (2.0 * h);
  }
  return H.map((row, r) => row.map((v, c) => 0.5 * (v + H[c][r])));
}

type AnyFn = (x: unknown) => unknown;

/** A smooth n-D problem as the ports accept it: a Problem or a bare `f(x)`. */
export type SmoothProblem =
  Problem<Point> | Problem<Vector> | Problem<number> | ((x: Vector) => number);

/** A counted callable (`numopt.core.counting.Counted`). */
export interface Counted<T> {
  (x: Vector): T;
  n: number;
}

function counted<T>(fn: (x: Vector) => T): Counted<T> {
  const c = ((x: Vector) => {
    c.n++;
    return fn(x);
  }) as Counted<T>;
  c.n = 0;
  return c;
}

/** Counted f and ∇f (and ∇²f) of a problem, with the central-difference fallbacks of Python. */
export interface Oracle {
  f: Counted<number>;
  grad: Counted<Vector>;
  hess: Counted<Matrix>;
  /** ∇f is a central difference of f (no analytic gradient). */
  gradFd: boolean;
  /** ∇²f is a central difference of ∇f (no analytic Hessian). */
  hessFd: boolean;
}

/** `vector_problem` + `start_point` + the counted oracle of `_resolve`. */
export function resolve(problemIn: SmoothProblem, x0: unknown): { x: Vector; oracle: Oracle } {
  const prob: Problem<unknown> =
    typeof problemIn === 'function'
      ? {
          id: 'custom',
          name: 'custom',
          latex: 'f(x)',
          f: problemIn as AnyFn,
          dim: x0 === null || x0 === undefined ? 2 : toVector(x0).length,
          domain: [],
        }
      : (problemIn as Problem<unknown>);
  const start = x0 ?? prob.x0;
  if (start === null || start === undefined)
    throw new MethodInputError(
      `${prob.id}: no starting point given and the problem has no default x0`,
    );
  const x = toVector(start);
  if (prob.dim && x.length !== prob.dim)
    throw new MethodInputError(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  const n = x.length;
  // Problems with dim = 1 take and return numbers (TS convention); Python passes a size-1 array.
  const scalar = prob.dim === 1;
  const arg = (v: Vector): unknown => (scalar ? v[0] : v);
  const fRaw = prob.f as AnyFn;
  const gRaw = prob.grad as AnyFn | undefined;
  const hRaw = prob.hess as AnyFn | undefined;
  const f = counted((v: Vector) => Number(fRaw(arg(v))));
  const grad = counted<Vector>((v) =>
    gRaw === undefined ? fdGradient(f, v) : toVector(gRaw(arg(v))),
  );
  const hess = counted<Matrix>((v) => {
    if (hRaw === undefined) return fdHessian(grad, v);
    const flat = toVector(hRaw(arg(v)));
    return Array.from({ length: n }, (_, r) => flat.slice(r * n, (r + 1) * n));
  });
  return {
    x,
    oracle: { f, grad, hess, gradFd: gRaw === undefined, hessFd: hRaw === undefined },
  };
}

/** `_validate` of both modules (gtol, max_iter, line_search). */
export function validateCommon(
  gtol: number,
  maxIter: number,
  lineSearch: string | null,
  choices: readonly string[],
): void {
  if (!(Number.isFinite(gtol) && gtol >= 0.0))
    throw new MethodInputError(`gtol must be finite and ≥ 0, got ${formatRepr(gtol)}`);
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new MethodInputError(`max_iter must be a positive integer, got ${String(maxIter)}`);
  if (lineSearch !== null && !choices.includes(lineSearch))
    throw new MethodInputError(
      `unknown line_search '${lineSearch}'; expected one of ${tupleRepr(choices)}`,
    );
}

// ---------------------------------------------------------------------------------------
// Symmetric eigendecomposition (stands in for numpy.linalg.eigh)
// ---------------------------------------------------------------------------------------

/** x = m·2^e exactly (m a signed BigInt integer) for a finite double x. */
function decompose(x: number): [bigint, number] {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const hi = view.getUint32(0);
  const lo = view.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e = -1074;
  if (biased !== 0) {
    mant |= 1n << 52n;
    e = biased - 1075;
  }
  return [hi >>> 31 ? -mant : mant, e];
}

/** x·2^k for an integer-valued x ≤ 2^53 whose result is representable (steps avoid overflow). */
function ldexp(x: number, k: number): number {
  let r = x;
  let e = k;
  while (e > 1023) {
    r *= 2 ** 1023;
    e -= 1023;
  }
  while (e < -1022) {
    r *= 2 ** -1022;
    e += 1022;
  }
  return r * 2 ** e;
}

/**
 * fma(a, b, c) = a·b + c with ONE rounding (round half to even), computed exactly with BigInt.
 *
 * NumPy's LAPACK/BLAS on this platform contract a·b + c into fused multiply-adds; `eigh2` and
 * `newtonDirection` use this to reproduce `numpy.linalg.eigh` and `Q @ ((Q.T @ g) / λ)` bit for bit
 * for n = 2 (checked on 20,000 random symmetric matrices).
 */
export function fma(a: number, b: number, c: number): number {
  const prod = a * b;
  if (!Number.isFinite(prod) || !Number.isFinite(c) || a === 0 || b === 0 || c === 0)
    return prod + c;
  const [ma, ea] = decompose(a);
  const [mb, eb] = decompose(b);
  const [mc, ec] = decompose(c);
  const ep = ea + eb;
  const e = Math.min(ep, ec);
  const sum = ((ma * mb) << BigInt(ep - e)) + (mc << BigInt(ec - e));
  if (sum === 0n) return 0;
  const neg = sum < 0n;
  const m = neg ? -sum : sum;
  const top = m.toString(2).length - 1 + e;
  // The lowest kept bit: 53 significant bits, or the subnormal grid 2^-1074.
  const low = Math.max(top - 52, -1074);
  let q = m;
  let scaleExp = e;
  if (low > e) {
    const sh = BigInt(low - e);
    q = m >> sh;
    const rem = m - (q << sh);
    const half = 1n << (sh - 1n);
    if (rem > half || (rem === half && (q & 1n) === 1n)) q += 1n;
    scaleExp = low;
  }
  const r = ldexp(Number(q), scaleExp);
  return neg ? -r : r;
}

/**
 * LAPACK `dlaev2`: the eigendecomposition of [[a, b], [b, c]]. Returns (rt1, rt2, cs1, sn1) with
 * |rt1| ≥ |rt2| and (cs1, sn1) the unit eigenvector of rt1. The three contracted products of the
 * compiled LAPACK are fused here as well.
 */
function dlaev2(a: number, b: number, c: number): [number, number, number, number] {
  const sm = a + c;
  const df = a - c;
  const adf = Math.abs(df);
  const tb = b + b;
  const ab = Math.abs(tb);
  const [acmx, acmn] = Math.abs(a) > Math.abs(c) ? [a, c] : [c, a];
  let rt: number;
  if (adf > ab) {
    const r = ab / adf;
    rt = adf * Math.sqrt(fma(r, r, 1.0));
  } else if (adf < ab) {
    const r = adf / ab;
    rt = ab * Math.sqrt(fma(r, r, 1.0));
  } else rt = ab * Math.sqrt(2.0);
  let rt1: number;
  let rt2: number;
  let sgn1: number;
  if (sm < 0.0) {
    rt1 = 0.5 * (sm - rt);
    sgn1 = -1;
    rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
  } else if (sm > 0.0) {
    rt1 = 0.5 * (sm + rt);
    sgn1 = 1;
    rt2 = fma(acmx / rt1, acmn, -((b / rt1) * b));
  } else {
    rt1 = 0.5 * rt;
    rt2 = -0.5 * rt;
    sgn1 = 1;
  }
  let cs: number;
  let sgn2: number;
  if (df >= 0.0) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }
  let cs1: number;
  let sn1: number;
  if (Math.abs(cs) > ab) {
    const ct = -tb / cs;
    sn1 = 1.0 / Math.sqrt(fma(ct, ct, 1.0));
    cs1 = ct * sn1;
  } else if (ab === 0.0) {
    cs1 = 1.0;
    sn1 = 0.0;
  } else {
    const tn = -cs / tb;
    cs1 = 1.0 / Math.sqrt(fma(tn, tn, 1.0));
    sn1 = tn * cs1;
  }
  if (sgn1 === sgn2) {
    const tn = cs1;
    cs1 = -sn1;
    sn1 = tn;
  }
  return [rt1, rt2, cs1, sn1];
}

const SAFMIN = 2.2250738585072014e-308;

/** `dsyevd`'s scaling range [√(safmin/ε), √(ε/safmin)] = [2⁻⁴⁸⁵, 2⁴⁸⁵] (≈ 1e∓146). */
const RMIN = 2 ** -485;
const RMAX = 2 ** 485;

/**
 * `numpy.linalg.eigh` of a symmetric 2×2 matrix as LAPACK computes it (`dsyevd` → `dsytrd` (a
 * no-op for n = 2) → `dsteqr`): the split and deflation tests of `dsteqr`, one `dlaev2` rotation
 * applied to Z = I by `dlasr`, then the ascending selection sort.
 *
 * Like `dsyevd`, a matrix with max|a_ij| outside [RMIN, RMAX] is first multiplied by
 * σ = RMAX/max|a_ij| (or RMIN/max|a_ij|), and the eigenvalues are multiplied by 1/σ at the end.
 * Without it the deflation test squares e, which overflows for |e| ≳ 1e154 (or underflows for
 * |e| ≲ 1e-154) and deflates a matrix that is far from diagonal.
 */
function eigh2(A: Matrix): { lam: Vector; Q: Matrix } {
  // dlansy('M', 'L'): the lower triangle a₁₁, a₂₁, a₂₂.
  const anrm = Math.max(Math.abs(A[0][0]), Math.abs(A[1][0]), Math.abs(A[1][1]));
  if ((anrm > 0.0 && anrm < RMIN) || anrm > RMAX) {
    const sigma = anrm < RMIN ? RMIN / anrm : RMAX / anrm;
    const e = A[1][0] * sigma;
    const { lam, Q } = eigh2Unscaled([
      [A[0][0] * sigma, e],
      [e, A[1][1] * sigma],
    ]);
    const inv = 1.0 / sigma; // dscal(n, 1/σ, w)
    return { lam: lam.map((v) => v * inv), Q };
  }
  return eigh2Unscaled(A);
}

function eigh2Unscaled(A: Matrix): { lam: Vector; Q: Matrix } {
  const d1 = A[0][0];
  const d2 = A[1][1];
  const e = A[1][0];
  let lam: Vector = [d1, d2];
  let Z: Matrix = [
    [1.0, 0.0],
    [0.0, 1.0],
  ];
  const split = e === 0 || Math.abs(e) <= Math.sqrt(Math.abs(d1)) * Math.sqrt(Math.abs(d2)) * EPS;
  // QL when |d2| ≥ |d1|, else QR; their deflation tests differ only in the factor order.
  const [dm, dn] = Math.abs(d2) < Math.abs(d1) ? [d2, d1] : [d1, d2];
  const deflate = Math.abs(e) ** 2 <= EPS * EPS * Math.abs(dm) * Math.abs(dn) + SAFMIN;
  if (!split && !deflate) {
    const [rt1, rt2, c, s] = dlaev2(d1, e, d2);
    lam = [rt1, rt2];
    Z = [
      [c, -s],
      [s, c],
    ];
  }
  if (lam[1] < lam[0]) {
    lam = [lam[1], lam[0]];
    Z = [
      [Z[0][1], Z[0][0]],
      [Z[1][1], Z[1][0]],
    ];
  }
  return { lam, Q: Z };
}

/**
 * ∇²f = Q diag(λ) Qᵀ for a finite symmetric A (stands in for `numpy.linalg.eigh`): eigenvalues
 * ascending, eigenvectors as the columns of Q. n = 1 and n = 2 follow LAPACK exactly (see
 * `eigh2`); n ≥ 3 uses the cyclic Jacobi method (Golub & Van Loan (2013), Alg. 8.5.3), accurate
 * to a small multiple of ε‖A‖ like LAPACK but not bit for bit.
 */
export function eigh(A: Matrix): { lam: Vector; Q: Matrix } {
  const n = A.length;
  if (n === 1) return { lam: [A[0][0]], Q: [[1.0]] };
  if (n === 2) return eigh2(A);
  const a = A.map((row) => row.slice());
  const V: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += Math.abs(a[p][q]);
    if (off === 0) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (apq === 0) continue;
        const app = a[p][p];
        const aqq = a[q][q];
        // Negligible against both diagonal entries (Numerical Recipes' test): drop it.
        const g = 100.0 * Math.abs(apq);
        if (
          sweep > 3 &&
          Math.abs(app) + g === Math.abs(app) &&
          Math.abs(aqq) + g === Math.abs(aqq)
        ) {
          a[p][q] = 0;
          a[q][p] = 0;
          continue;
        }
        const theta = (aqq - app) / (2.0 * apq);
        let t: number;
        if (!Number.isFinite(theta * theta)) t = 1.0 / (2.0 * theta);
        else t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(1.0 + theta * theta));
        const c = 1.0 / Math.sqrt(1.0 + t * t);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        a[p][q] = 0;
        a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {
    lam: order.map((i) => a[i][i]),
    Q: V.map((row) => order.map((i) => row[i])),
  };
}

// ---------------------------------------------------------------------------------------
// Module helpers (Python: _evaluate_hessian, _eigs, _eig_tol, ...)
// ---------------------------------------------------------------------------------------

interface Eig {
  lam: Vector;
  Q: Matrix;
}

function evaluateHessian(oracle: Oracle, x: Vector): Matrix {
  const H = oracle.hess(x);
  // Symmetrize ∇²f once: the spectral solve and the Cholesky factorization read one triangle.
  return H.map((row, i) => row.map((v, j) => 0.5 * (v + H[j][i])));
}

function nanMatrix(n: number): Matrix {
  return Array.from({ length: n }, () => new Array<number>(n).fill(NaN));
}

function eigs(H: Matrix): Eig | null {
  if (!finite(H)) return null;
  return eigh(H);
}

function eigTol(lam: Vector, fx: number, oracle: Oracle): number {
  const n = lam.length;
  const lamMax = maxAbs(lam);
  if (!oracle.hessFd) return n * EPS * lamMax;
  const scale = Math.max(1.0, lamMax, oracle.gradFd ? Math.abs(fx) : 0.0);
  return n * FD_HESS_REL * scale;
}

function isSingular(lam: Vector, tol: number): boolean {
  const absl = lam.map(Math.abs);
  return Math.max(...absl) === 0.0 || Math.min(...absl) <= tol;
}

/**
 * p = −Q diag(1/λ) Qᵀ g (N&W eq. 3.30). For n = 2 the products round like NumPy's BLAS here:
 * `Q.T @ g` (a transposed gemv) as an FMA chain, `Q @ w` as a plain two-term sum.
 */
function newtonDirection(lam: Vector, Q: Matrix, g: Vector): Vector {
  const n = g.length;
  const w = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    let s: number;
    if (n === 2) s = fma(Q[1][j], g[1], Q[0][j] * g[0]);
    else {
      s = 0;
      for (let i = 0; i < n; i++) s += Q[i][j] * g[i];
    }
    w[j] = s / lam[j];
  }
  return Q.map((row) => -dotp(row, w));
}

const g3 = (v: number) => formatG(v, 3);

/** Classify a stationary point by the eigenvalues of ∇²f (N&W Thms 2.3–2.4). */
function secondOrder(lam: Vector, tol: number, fdHessian: boolean): [boolean, string] {
  const lamMin = lam[0];
  const source = fdHessian ? 'the finite-difference ∇²f' : '∇²f';
  if (lamMin < -tol) {
    const lamTop = lam[lam.length - 1];
    let kind: string;
    if (lamTop < -tol) kind = 'a maximizer';
    else if (lamTop > tol) kind = 'a saddle point';
    else
      kind =
        `a saddle point or a maximizer (λ_max = ${g3(lamTop + 0.0)}: ∇²f is singular, so ` +
        'the second-order test cannot decide)';
    return [
      false,
      `stopped at ${kind}, not a minimizer: ${source} has the eigenvalue ` +
        `λ_min = ${g3(lamMin)} < 0`,
    ];
  }
  if (lamMin <= tol) {
    if (fdHessian)
      return [
        true,
        `the finite-difference ∇²f has λ_min = ${g3(lamMin)}, within its accuracy ` +
          `±${g3(tol)} of 0: ∇²f is positive semidefinite to that accuracy, and the ` +
          'second-order sufficient condition is not verified',
      ];
    return [
      true,
      '∇²f is positive semidefinite but numerically singular ' +
        `(λ_min = ${g3(lamMin)}): the second-order sufficient condition is not verified`,
    ];
  }
  return [true, `${source} is positive definite (λ_min = ${g3(lamMin)}): a strict local minimizer`];
}

function hessInfo(H: Matrix, lam: Vector | null): Record<string, unknown> {
  return {
    hess: H.length <= 2 ? H.map((r) => r.slice()) : null,
    hess_eigs: lam !== null ? lam.slice() : new Array<number>(H.length).fill(NaN),
  };
}

function incomingNone(): Record<string, unknown> {
  return { direction: null, alpha: null, trials: [], descent: null };
}

function makeResult(
  method: string,
  x: Vector,
  fx: number,
  converged: boolean,
  message: string,
  k: number,
  oracle: Oracle,
  trace: Step[],
): Result {
  return {
    method,
    x: x.slice(),
    fun: fx,
    converged,
    message,
    nIter: k,
    nFev: oracle.f.n,
    nGev: oracle.grad.n,
    nHev: oracle.hess.n,
    trace,
    extra: {},
  };
}

interface Start {
  x: Vector;
  fx: number;
  g: Vector;
  H: Matrix;
  eig: Eig | null;
  oracle: Oracle;
}

function start(method: string, problem: SmoothProblem, x0: unknown): Start {
  const { x, oracle } = resolve(problem, x0);
  const fx = oracle.f(x);
  const g = oracle.grad(x);
  if (!finite(fx, g))
    throw new MethodInputError(
      `${method}: f and ∇f must be finite at x0; got f=${formatRepr(fx)}, ∇f=${arrayStr(g)}`,
    );
  const H = evaluateHessian(oracle, x);
  return { x, fx, g, H, eig: eigs(H), oracle };
}

/** The shared stopping test: [converged, message] or null to continue. */
function checkStop(
  g: Vector,
  fx: number,
  eig: Eig | null,
  gtol: number,
  oracle: Oracle,
): [boolean, string] | null {
  const gnorm = maxAbs(g);
  if (gnorm > gtol) return null;
  const head = `‖∇f‖∞ = ${g3(gnorm)} ≤ gtol`;
  if (eig === null) return [false, `${head}, but ∇²f is not finite there`];
  const [ok, why] = secondOrder(eig.lam, eigTol(eig.lam, fx, oracle), oracle.hessFd);
  return [ok, `${head}; ${why}`];
}

/** Why no step can be tested when ∇fᵀp is not a finite negative number (`_slope_failure`). */
export function slopeFailure(slope: number): string {
  if (Number.isFinite(slope)) return '∇fᵀp underflowed to 0, so no step can be tested';
  return `∇fᵀp overflowed to ${formatRepr(slope)}, so no step can be tested`;
}

function lineSearchFailure(k: number, g: Vector, p: Vector, fx: number, why: string): string {
  return (
    `line search failed at iteration ${k}: ${why}; predicted decrease ` +
    `|∇fᵀp| = ${g3(Math.abs(dotp(g, p)))} vs rounding level of f ` +
    `ε·max(1, |f|) = ${g3(EPS * Math.max(1.0, Math.abs(fx)))}`
  );
}

// ---------------------------------------------------------------------------------------
// Pure Newton
// ---------------------------------------------------------------------------------------

export interface NewtonOptions {
  x0?: unknown;
  gtol?: number;
  max_iter?: number;
}

/** Pure Newton's method: x_{k+1} = x_k + p_k with ∇²f(x_k) p_k = −∇f(x_k). */
export function pureNewton(problem: SmoothProblem, opts: NewtonOptions = {}): Result {
  const gtol = Number(opts.gtol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 100);
  validateCommon(gtol, maxIter, null, LINE_SEARCHES);
  const method = 'pure_newton';
  const s0 = start(method, problem, opts.x0);
  const { oracle } = s0;
  let { x, fx, g, H, eig } = s0;
  const xScale = Math.max(1.0, maxAbs(x));
  let info: Record<string, unknown> = {
    grad: g.slice(),
    ...hessInfo(H, eig !== null ? eig.lam : null),
    ...incomingNone(),
  };
  const trace: Step[] = [{ k: 0, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: null, info }];
  let k = 0;
  for (;;) {
    const stop = checkStop(g, fx, eig, gtol, oracle);
    if (stop !== null) return makeResult(method, x, fx, stop[0], stop[1], k, oracle, trace);
    // The divergence test runs only after the stopping test.
    const xNorm = maxAbs(x);
    if (xNorm > DIVERGENCE_FACTOR * xScale) {
      const msg =
        `the iterates diverge: ‖x_${k}‖∞ = ${g3(xNorm)} > ` +
        `${DIVERGENCE_FACTOR_TEXT}·max(1, ‖x_0‖∞)`;
      return makeResult(method, x, fx, false, msg, k, oracle, trace);
    }
    if (eig === null)
      return makeResult(
        method,
        x,
        fx,
        false,
        `∇²f is not finite at iteration ${k}`,
        k,
        oracle,
        trace,
      );
    const { lam, Q } = eig;
    const tol = eigTol(lam, fx, oracle);
    if (isSingular(lam, tol)) {
      const source = oracle.hessFd ? 'the finite-difference ∇²f' : '∇²f';
      const msg =
        `${source}(x_${k}) is numerically singular (|λ|_min = ` +
        `${g3(Math.min(...lam.map(Math.abs)))} ≤ tol = ${g3(tol)}); the Newton step is undefined`;
      return makeResult(method, x, fx, false, msg, k, oracle, trace);
    }
    if (k === maxIter)
      return makeResult(method, x, fx, false, `reached max_iter=${maxIter}`, k, oracle, trace);
    const p = newtonDirection(lam, Q, g);
    const descent = dotp(g, p) < 0.0;
    k += 1;
    x = x.map((v, i) => v + p[i]);
    fx = oracle.f(x);
    g = oracle.grad(x);
    H = finite(fx, g) ? evaluateHessian(oracle, x) : nanMatrix(x.length);
    eig = eigs(H);
    info = {
      grad: g.slice(),
      ...hessInfo(H, eig !== null ? eig.lam : null),
      direction: p.slice(),
      alpha: 1.0,
      trials: [],
      descent,
    };
    trace.push({ k, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: 1.0, info });
    if (!finite(fx, g)) {
      const msg = `f or ∇f is not finite at iteration ${k}: the iterates diverged`;
      return makeResult(method, x, fx, false, msg, k, oracle, trace);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Line-search Newton methods
// ---------------------------------------------------------------------------------------

/** N&W Alg. 3.3 (Cholesky with added multiple of the identity): [L | null, τ, attempts]. */
export function choleskyShift(A: Matrix, beta: number): [Matrix | null, number, number] {
  const n = A.length;
  let aMin = Infinity;
  for (let i = 0; i < n; i++) aMin = Math.min(aMin, A[i][i]);
  let tau = aMin > 0.0 ? 0.0 : -aMin + beta;
  let attempts = 0;
  let doublings = 0;
  for (;;) {
    attempts += 1;
    const B = A.map((row, i) => row.map((v, j) => v + tau * (i === j ? 1.0 : 0.0)));
    const L = cholesky(B);
    if (L !== null && finite(L)) return [L, tau, attempts];
    if (tau >= beta) {
      doublings += 1;
      if (doublings >= MAX_SHIFT_DOUBLINGS) return [null, tau, attempts];
    }
    tau = Math.max(2.0 * tau, beta);
  }
}

/**
 * Lower-triangular Cholesky factor (reads the lower triangle), or null when a pivot is not > 0.
 * Rounds like OpenBLAS `potf2` behind `numpy.linalg.cholesky`: each column below the diagonal is
 * scaled by the reciprocal 1/L_jj (bit for bit for n = 2).
 */
function cholesky(A: Matrix): Matrix | null {
  const n = A.length;
  const L: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let d = A[j][j];
    for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k];
    if (!(d > 0)) return null;
    const ljj = Math.sqrt(d);
    L[j][j] = ljj;
    const inv = 1.0 / ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = s * inv;
    }
  }
  return L;
}

/** Solve L Lᵀ z = b by forward then back substitution (Golub & Van Loan, Algs 3.1.1–3.1.2). */
export function choleskySolve(L: Matrix, b: Vector): Vector {
  const n = b.length;
  const w = new Array<number>(n);
  const z = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < i; j++) s += L[i][j] * w[j];
    w[i] = (b[i] - s) / L[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = 0;
    for (let j = i + 1; j < n; j++) s += L[j][i] * z[j];
    z[i] = (w[i] - s) / L[i][i];
  }
  return z;
}

type DirectionFn = (
  H: Matrix,
  g: Vector,
  eig: Eig,
  tol: number,
) => [Vector, Record<string, unknown>] | string;

function lineSearchNewton(
  method: 'damped_newton' | 'modified_newton',
  problem: SmoothProblem,
  x0: unknown,
  gtol: number,
  maxIter: number,
  lineSearch: string,
  direction: DirectionFn,
): Result {
  const s0 = start(method, problem, x0);
  const { oracle } = s0;
  let { x, fx, g, H, eig } = s0;
  const extraNone: Record<string, unknown> =
    method === 'damped_newton' ? { direction_type: null } : { tau: null, chol_attempts: null };
  let info: Record<string, unknown> = {
    grad: g.slice(),
    ...hessInfo(H, eig !== null ? eig.lam : null),
    ...incomingNone(),
    ...extraNone,
  };
  const trace: Step[] = [{ k: 0, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: null, info }];
  let k = 0;
  for (;;) {
    const stop = checkStop(g, fx, eig, gtol, oracle);
    if (stop !== null) return makeResult(method, x, fx, stop[0], stop[1], k, oracle, trace);
    if (eig === null)
      return makeResult(
        method,
        x,
        fx,
        false,
        `∇²f is not finite at iteration ${k}`,
        k,
        oracle,
        trace,
      );
    if (k === maxIter)
      return makeResult(method, x, fx, false, `reached max_iter=${maxIter}`, k, oracle, trace);
    const out = direction(H, g, eig, eigTol(eig.lam, fx, oracle));
    if (typeof out === 'string') return makeResult(method, x, fx, false, out, k, oracle, trace);
    const [p, extra] = out;
    const slope = dotp(g, p);
    if (!(Number.isFinite(slope) && slope < 0.0)) {
      // NOTE (Python): every direction passed a descent test (cos θ > η, −∇f, or −‖L⁻¹∇f‖² < 0),
      // so a non-finite ∇fᵀp overflowed and ∇fᵀp ≥ 0 underflowed (∇f near the underflow
      // threshold). search() needs a finite ∇fᵀp < 0, so either one ends the run.
      const why = slopeFailure(slope);
      return makeResult(
        method,
        x,
        fx,
        false,
        lineSearchFailure(k + 1, g, p, fx, why),
        k,
        oracle,
        trace,
      );
    }
    // The line search calls the counted f and ∇f, so the oracle totals include its calls.
    const ls = search(lineSearch, oracle.f, oracle.grad, x, p, { f0: fx, g0: g, alpha0: 1.0 });
    if (!ls.success) {
      const floor = FD_STALL * EPS * Math.max(1.0, Math.abs(fx));
      if (oracle.gradFd && Math.abs(dotp(g, p)) <= floor) {
        // With a finite-difference ∇f, a predicted decrease at the rounding level of f means x
        // is stationary to the accuracy of the gradient estimate.
        const [ok, why] = secondOrder(eig.lam, eigTol(eig.lam, fx, oracle), oracle.hessFd);
        const msg =
          `stationary to the accuracy of the finite-difference ∇f at iteration ${k}: ` +
          `predicted decrease |∇fᵀp| = ${g3(Math.abs(dotp(g, p)))} ≤ ${g3(floor)} ` +
          `(the rounding level of f); ${why}`;
        return makeResult(method, x, fx, ok, msg, k, oracle, trace);
      }
      const msg = lineSearchFailure(k + 1, g, p, fx, ls.message);
      return makeResult(method, x, fx, false, msg, k, oracle, trace);
    }
    k += 1;
    x = step(x, ls.alpha, p);
    fx = Number(ls.fNew);
    g = ls.gNew !== null ? toVector(ls.gNew) : oracle.grad(x);
    H = finite(g) ? evaluateHessian(oracle, x) : nanMatrix(x.length);
    eig = eigs(H);
    info = {
      grad: g.slice(),
      ...hessInfo(H, eig !== null ? eig.lam : null),
      direction: p.slice(),
      alpha: ls.alpha,
      trials: ls.trials.map(([a, v]) => [a, v]),
      descent: true,
      ...extra,
    };
    trace.push({ k, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: ls.alpha, info });
    if (!finite(fx, g))
      return makeResult(
        method,
        x,
        fx,
        false,
        `f or ∇f is not finite at iteration ${k}`,
        k,
        oracle,
        trace,
      );
  }
}

export interface LineSearchNewtonOptions extends NewtonOptions {
  line_search?: string;
}

/** Damped (line-search) Newton with the −∇f fallback. */
export function dampedNewton(problem: SmoothProblem, opts: LineSearchNewtonOptions = {}): Result {
  const gtol = Number(opts.gtol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 100);
  const lineSearch = String(opts.line_search ?? 'backtracking');
  validateCommon(gtol, maxIter, lineSearch, LINE_SEARCHES);
  const direction: DirectionFn = (_H, g, eig, tol) => {
    if (!isSingular(eig.lam, tol)) {
      const p = newtonDirection(eig.lam, eig.Q, g);
      if (cosAngle(g, p) > DESCENT_COS) return [p, { direction_type: 'newton' }];
    }
    return [g.map((v) => -v), { direction_type: 'steepest' }];
  };
  return lineSearchNewton('damped_newton', problem, opts.x0, gtol, maxIter, lineSearch, direction);
}

export interface ModifiedNewtonOptions extends LineSearchNewtonOptions {
  beta?: number;
}

/** Line-search Newton with Hessian modification (N&W Alg. 3.2 with Alg. 3.3). */
export function modifiedNewton(problem: SmoothProblem, opts: ModifiedNewtonOptions = {}): Result {
  const gtol = Number(opts.gtol ?? 1e-8);
  const maxIter = Number(opts.max_iter ?? 100);
  const lineSearch = String(opts.line_search ?? 'backtracking');
  const beta = Number(opts.beta ?? 1e-3);
  validateCommon(gtol, maxIter, lineSearch, LINE_SEARCHES);
  if (!(Number.isFinite(beta) && beta > 0.0))
    throw new MethodInputError(`beta must be finite and > 0, got ${formatRepr(beta)}`);
  const direction: DirectionFn = (H, g) => {
    const [L, tau, attempts] = choleskyShift(H, beta);
    if (L === null)
      return `Alg. 3.3 found no shift: ∇²f + τI had no Cholesky factor up to τ = ${g3(tau)}`;
    const p = choleskySolve(
      L,
      g.map((v) => -v),
    );
    if (!finite(p)) return `the modified Newton direction is not finite (τ = ${g3(tau)})`;
    return [p, { tau, chol_attempts: attempts }];
  };
  return lineSearchNewton(
    'modified_newton',
    problem,
    opts.x0,
    gtol,
    maxIter,
    lineSearch,
    direction,
  );
}

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

const P_GTOL = param.float('gtol', 1e-8, {
  min: 1e-14,
  max: 1e-2,
  log: true,
  help: 'Stop when ‖∇f(x)‖∞ ≤ gtol (and ∇²f(x) has no negative eigenvalue).',
  label: 'Gradient tolerance',
  tex: '\\|\\nabla f\\|_\\infty \\le',
});
const P_MAX_ITER = param.int('max_iter', 100, {
  min: 1,
  max: 10_000,
  help: 'Iteration limit.',
  label: 'Max iterations',
});
const P_LINE_SEARCH = param.choice('line_search', 'backtracking', [...LINE_SEARCHES], {
  help: 'Step-length rule; every search tries α = 1 (the full Newton step) first.',
  label: 'Line search',
});
const P_BETA = param.float('beta', 1e-3, {
  min: 1e-8,
  max: 10.0,
  log: true,
  help: 'Alg. 3.3: smallest nonzero shift τ; τ doubles until ∇²f + τI has a Cholesky factor.',
  label: 'Smallest shift',
  tex: '\\beta',
});

const pureNewtonFn: MethodFn<SmoothProblem> = (problem, o) =>
  pureNewton(problem, o as NewtonOptions);
const dampedNewtonFn: MethodFn<SmoothProblem> = (problem, o) =>
  dampedNewton(problem, o as LineSearchNewtonOptions);
const modifiedNewtonFn: MethodFn<SmoothProblem> = (problem, o) =>
  modifiedNewton(problem, o as ModifiedNewtonOptions);

const PURE_DOC: MethodDoc = {
  rule: '\\nabla^2 f(x_k)\\,p_k = -\\nabla f(x_k), \\qquad x_{k+1} = x_k + p_k',
  intuition:
    'Fit the local quadratic model and jump straight to its stationary point. Near a minimizer the ' +
    'number of correct digits doubles each step; elsewhere the jump can go uphill, to a saddle, or away.',
  order: 'quadratic',
  pros: ['Quadratic convergence near a minimizer with ∇²f ≻ 0', 'One step on a convex quadratic'],
  cons: [
    'No globalization: attracted to saddles and maxima, can diverge',
    'Needs ∇²f and a linear solve per step',
  ],
  quantities: [
    { tex: '\\lambda(\\nabla^2 f)', key: 'info.hess_eigs', label: 'Hessian eigenvalues' },
    { tex: 'p_{k-1}', key: 'info.direction', label: 'Newton step' },
    { tex: '\\nabla f^\\top p < 0', key: 'info.descent', label: 'Descent' },
  ],
};

const DAMPED_DOC: MethodDoc = {
  rule: 'x_{k+1} = x_k + \\alpha_k p_k, \\quad p_k = \\begin{cases} -\\nabla^2 f(x_k)^{-1}\\nabla f(x_k) & \\text{if descent} \\\\ -\\nabla f(x_k) & \\text{otherwise} \\end{cases}',
  intuition:
    'Take the Newton step, but let a line search shorten it until f decreases enough. Where the ' +
    'Hessian is indefinite or singular, fall back to steepest descent for that step.',
  order: 'quadratic',
  pros: ['Global descent with Newton speed near the minimizer (α = 1 is accepted there)'],
  cons: ['The −∇f fallback can be slow on nonconvex regions', 'Needs ∇²f'],
  quantities: [
    { tex: '\\alpha_k', key: 'stepSize', label: 'Step length' },
    { tex: '\\text{direction}', key: 'info.direction_type', label: 'Direction type' },
    { tex: '\\lambda(\\nabla^2 f)', key: 'info.hess_eigs', label: 'Hessian eigenvalues' },
  ],
};

const MODIFIED_DOC: MethodDoc = {
  rule: '(\\nabla^2 f(x_k) + \\tau_k I)\\,p_k = -\\nabla f(x_k), \\qquad x_{k+1} = x_k + \\alpha_k p_k',
  intuition:
    'Shift the Hessian by τI, doubling τ until a Cholesky factorization succeeds, so the model is ' +
    'convex and p is always downhill. Near a minimizer τ = 0 and the method is Newton’s method.',
  order: 'quadratic',
  pros: [
    'Always a descent direction',
    'Uses negative curvature information instead of discarding it',
  ],
  cons: ['Several Cholesky factorizations per step where ∇²f is indefinite', 'Needs ∇²f'],
  quantities: [
    { tex: '\\tau_k', key: 'info.tau', label: 'Shift' },
    { tex: '\\alpha_k', key: 'stepSize', label: 'Step length' },
    { tex: '\\text{attempts}', key: 'info.chol_attempts', label: 'Cholesky attempts' },
  ],
};

registerMethod(
  {
    id: 'pure_newton',
    family: 'unconstrained',
    name: "Newton's method",
    params: [P_GTOL, P_MAX_ITER],
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic (near a minimizer with ∇²f ≻ 0)',
    summary: 'Jump to the stationary point of the local quadratic model: solve ∇²f p = −∇f.',
    references: [
      'Nocedal & Wright (2006), §2.2 and eq. 3.30; Theorem 3.5',
      'Boyd & Vandenberghe (2004), §9.5',
    ],
  },
  pureNewtonFn,
  PURE_DOC,
);

registerMethod(
  {
    id: 'damped_newton',
    family: 'unconstrained',
    name: 'Damped Newton (line search)',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH],
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic (near a minimizer with ∇²f ≻ 0, where α = 1 is accepted)',
    summary: 'Newton direction with a line search; falls back to −∇f where Newton points uphill.',
    references: [
      "Nocedal & Wright (2006), §3.3 (Newton's method with line search), Alg. 3.1",
      'Boyd & Vandenberghe (2004), Alg. 9.5 (damped Newton)',
    ],
  },
  dampedNewtonFn,
  DAMPED_DOC,
);

registerMethod(
  {
    id: 'modified_newton',
    family: 'unconstrained',
    name: 'Modified Newton (Hessian + τI)',
    params: [P_GTOL, P_MAX_ITER, P_LINE_SEARCH, P_BETA],
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic (near a minimizer with ∇²f ≻ 0, where τ = 0 and α = 1)',
    summary: 'Add τI to the Hessian until it is positive definite, then take a Newton step.',
    references: [
      'Nocedal & Wright (2006), Alg. 3.2 (line search Newton with modification)',
      'Nocedal & Wright (2006), Alg. 3.3 (Cholesky with added multiple of the identity)',
    ],
  },
  modifiedNewtonFn,
  MODIFIED_DOC,
);
