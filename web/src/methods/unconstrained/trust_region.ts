/**
 * Trust-region methods — TS port of `numopt.unconstrained.trust_region`
 * (src/numopt/unconstrained/trust_region.py).
 *
 * At x_k the methods minimize, approximately, the quadratic model
 *
 *     m_k(p) = f_k + g_kᵀp + ½ pᵀB_k p     subject to ‖p‖₂ ≤ Δ_k,      B_k = ∇²f(x_k),
 *
 * share the radius update of Nocedal & Wright (2006), Alg. 4.1, and differ only in the solver:
 *
 *   trust_region_cauchy    the Cauchy point (N&W Alg. 4.2, eq. 4.11–4.12)
 *   trust_region_dogleg    the dogleg step (N&W §4.1, eq. 4.15–4.16); Cauchy point if B ⊁ 0
 *   trust_region_steihaug  Steihaug–Toint truncated CG (N&W Alg. 7.2)
 *   trust_region_exact     the global minimizer of the subproblem (N&W Alg. 4.3, Moré & Sorensen
 *                          1983) including the hard case (N&W eq. 4.45)
 *
 * ρ_k = (actual + δ)/(predicted + δ) with δ = 10³ε·max(|f(x_k)|, |f(x_k + p_k)|) (the rounding
 * allowance of the Python module); ρ < ¼ → Δ ← ¼Δ; ρ > ¾ on the boundary → Δ ← min(2Δ, Δ̂);
 * the step is accepted when ρ > η.
 *
 * Stopping test (converged): ‖g_k‖∞ ≤ `gtol`. Failures (converged = false): non-finite f, ∇f or
 * ∇²f, the radius collapsing below ε·max(1, ‖x_k‖), `max_iter`, or a model step predicting no
 * decrease (the model is at its rounding level). Counts: one f per iteration, one ∇f and one ∇²f
 * per accepted point.
 *
 * Step.info keys (snake_case, as in Python): center, grad, H, radius, new_radius, step,
 * step_norm, hits_boundary, predicted, actual, rho, accepted, trial_point, cauchy_point,
 * newton_point, note, plus per method: tau (cauchy); dogleg_path, tau (dogleg); cg_path,
 * cg_iters, termination, cg_tol (steihaug); lambda, lambda_min, hard_case, lambda_iters (exact).
 * `Result.extra = { n_rejected }`.
 *
 * NOTE (port): `np.linalg.eigh` (LAPACK dsyevd) is reproduced exactly for n ≤ 2 (dsteqr's dlaev2
 * rotation) and replaced by a cyclic Jacobi eigensolver for n ≥ 3 (same eigenpairs up to
 * rounding; the solver only uses sign-invariant products of the eigenvectors, and q₁'s sign is
 * normalized as in Python).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { Matrix, MethodFn, Params, Result, RunOptions, Step, Vector } from '../../core/types';
import { dot, matvec, norm } from '../../core/linalg';
import { npSum } from '../../problems/unconstrained';
import { formatRepr } from '../line_search/methods';
import {
  ValueError,
  allFinite,
  frexpExp,
  g3,
  ldexp,
  maxAbs,
  norm2,
  pyMax,
  pyMin,
  resolveSmooth,
  type SmoothProblem,
} from './conjugate_gradient';

const EPS = 2.220446049250313e-16;
/** Radius update thresholds of N&W Alg. 4.1. */
const RHO_SHRINK = 0.25;
const RHO_EXPAND = 0.75;
/** Rounding allowance δ = ROUNDING·ε·max(|f(x)|, |f(x + p)|) in ρ. */
const ROUNDING = 1e3;
/** Relative accuracy |‖p(λ)‖ − Δ| ≤ LAMBDA_RTOL·Δ of the secular equation. */
const LAMBDA_RTOL = 1e-12;
/** Iteration cap of the secular-equation solver. */
const LAMBDA_MAX_ITER = 100;
/** LAPACK dsyevd's scaling range √(safmin/ε), √(ε/safmin) with ε = 2⁻⁵², safmin = 2⁻¹⁰²². */
const RMIN = 2 ** -485;
const RMAX = 2 ** 485;

const PARAMS = [
  param.float('gtol', 1e-8, {
    min: 1e-14,
    max: 1e-2,
    log: true,
    help: 'Stop when the gradient ‖∇f(x)‖∞ ≤ gtol.',
    label: 'Gradient tolerance',
    tex: '\\|\\nabla f\\|_\\infty \\le',
  }),
  param.int('max_iter', 200, {
    min: 1,
    max: 100_000,
    help: 'Iteration limit.',
    label: 'Max iterations',
  }),
  param.float('radius0', 1.0, {
    min: 1e-3,
    max: 100.0,
    log: true,
    help: 'Initial trust-region radius Δ₀.',
    label: 'Initial radius',
    tex: '\\Delta_0',
  }),
  param.float('max_radius', 100.0, {
    min: 0.1,
    max: 1e4,
    log: true,
    help: 'Largest radius Δ̂ the update may reach.',
    label: 'Largest radius',
    tex: '\\hat\\Delta',
  }),
  param.float('eta', 0.15, {
    min: 0.0,
    max: 0.24,
    help: 'Accept the step when ρ = actual/predicted reduction > η (η ∈ [0, ¼)).',
    label: 'Acceptance threshold',
    tex: '\\eta',
  }),
];

// ---------------------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------------------

const neg = (v: readonly number[]): Vector => v.map((t) => -t);
const scaled = (a: number, v: readonly number[]): Vector => v.map((t) => a * t);
const plus = (a: readonly number[], b: readonly number[]): Vector => a.map((t, i) => t + b[i]);
const minus = (a: readonly number[], b: readonly number[]): Vector => a.map((t, i) => t - b[i]);
const anyNonzero = (v: readonly number[]): boolean => v.some((t) => t !== 0);

/** (ĝ, e) with ĝ = 2⁻ᵉg exactly and ½ ≤ ‖ĝ‖∞ < 1; g must be finite and nonzero. */
function pow2Scaled(g: readonly number[]): [Vector, number] {
  const e = frexpExp(maxAbs(g));
  return [g.map((t) => ldexp(t, -e)), e];
}

/** Lower Cholesky factor (LAPACK dpotrf failure rule: a pivot ≤ 0 or NaN), or null. */
function cholesky(B: Matrix): Matrix | null {
  const n = B.length;
  const L: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < j; k++) s += L[j][k] * L[j][k];
    const ajj = B[j][j] - s;
    if (!(ajj > 0.0)) return null;
    L[j][j] = Math.sqrt(ajj);
    // LAPACK/OpenBLAS potf2 scales the column by the reciprocal 1/ℓⱼⱼ (SCAL), not by division.
    const inv = 1.0 / L[j][j];
    for (let i = j + 1; i < n; i++) {
      let t = 0;
      for (let k = 0; k < j; k++) t += L[i][k] * L[j][k];
      L[i][j] = (B[i][j] - t) * inv;
    }
  }
  return allFinite(L) ? L : null;
}

/** Solve LLᵀx = b by forward and back substitution (Golub & Van Loan, Alg. 3.1.1–3.1.2). */
function choSolve(L: Matrix, b: readonly number[]): Vector {
  const n = b.length;
  const y = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < i; k++) s += L[i][k] * y[k];
    y[i] = (b[i] - s) / L[i][i];
  }
  const x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = 0;
    for (let k = i + 1; k < n; k++) s += L[k][i] * x[k];
    x[i] = (y[i] - s) / L[i][i];
  }
  return x;
}

/** p^B = −B⁻¹g when B is positive definite, else null. */
function newtonStep(g: readonly number[], B: Matrix): Vector | null {
  const L = cholesky(B);
  if (L === null) return null;
  const p = neg(choSolve(L, g));
  return allFinite(p) ? p : null;
}

/** Roots τ₋ ≤ 0 ≤ τ₊ of ‖z + τd‖ = Δ for ‖z‖ ≤ Δ, d ≠ 0 (cancellation-free, Higham §1.8). */
function boundaryTau(z: readonly number[], d: readonly number[], delta: number): [number, number] {
  const a = dot(d, d);
  const b = dot(z, d);
  const c = pyMin(dot(z, z) - delta * delta, 0.0);
  const s = Math.sqrt(pyMax(b * b - a * c, 0.0));
  let tauMinus: number, tauPlus: number;
  if (b <= 0.0) {
    tauPlus = (-b + s) / a;
    tauMinus = tauPlus > 0.0 ? c / (a * tauPlus) : (-b - s) / a;
  } else {
    tauMinus = (-b - s) / a;
    tauPlus = tauMinus < 0.0 ? c / (a * tauMinus) : (-b + s) / a;
  }
  return [tauMinus, tauPlus];
}

/**
 * `np.linalg.eigh` (LAPACK dsyevd, lower triangle): eigenvalues ascending, eigenvectors as the
 * columns of Q.
 *
 * dsyevd scales A by σ when max|a_ij| lies outside [rmin, rmax] = [2⁻⁴⁸⁵, 2⁴⁸⁵] and multiplies the
 * eigenvalues by 1/σ afterwards; both roundings are reproduced. For n = 2 the tridiagonal form is
 * A itself, and dsteqr diagonalizes it with one closed-form rotation (dlaev2): that path is
 * reproduced operation by operation, so 2-D problems agree with NumPy to the last bit. For n ≥ 3
 * a cyclic Jacobi iteration stands in for the divide-and-conquer solver (same eigenpairs up to
 * rounding; the trust-region solver only uses sign-invariant products of the eigenvectors).
 */
export function eigh(A: Matrix): { values: number[]; Q: Matrix } {
  const n = A.length;
  let anrm = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) anrm = Math.max(anrm, Math.abs(A[i][j]));
  const sigma =
    anrm > 0 && anrm < RMIN ? RMIN / anrm : anrm > RMAX && Number.isFinite(anrm) ? RMAX / anrm : 1;
  // dsyevd reads the lower triangle.
  const a = A.map((row, i) =>
    row.map((_, j) => {
      const v = i >= j ? A[i][j] : A[j][i];
      return sigma === 1 ? v : v * sigma;
    }),
  );
  const out = n === 1 ? { values: [a[0][0]], Q: [[1.0]] } : n === 2 ? steqr2(a) : jacobiEig(a);
  if (sigma !== 1) {
    const unscale = 1 / sigma;
    out.values = out.values.map((v) => v * unscale);
  }
  return out;
}

const F64 = new DataView(new ArrayBuffer(8));

/** x = m·2ᵉ exactly, with m a BigInt (x finite). */
function decompose(x: number): [bigint, number] {
  F64.setFloat64(0, x);
  const hi = F64.getUint32(0),
    lo = F64.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e = -1074;
  if (biased !== 0) {
    m |= 1n << 52n;
    e = biased - 1075;
  }
  return [hi >>> 31 ? -m : m, e];
}

/**
 * Fused multiply-add round(a·b + c) with a single rounding (IEEE 754 fusedMultiplyAdd, exact via
 * BigInt). The NumPy wheels' LAPACK is compiled by gfortran with FMA contraction on aarch64 and
 * x86-64-v3, so dlaev2 below needs it to reproduce np.linalg.eigh bit for bit.
 */
export function fma(a: number, b: number, c: number): number {
  const p = a * b;
  if (!Number.isFinite(p) || !Number.isFinite(c) || a === 0 || b === 0) return p + c;
  const [ma, ea] = decompose(a),
    [mb, eb] = decompose(b),
    [mc, ec] = decompose(c);
  const ep = ea + eb;
  const e = Math.min(ep, ec);
  const m = ((ma * mb) << BigInt(ep - e)) + (mc << BigInt(ec - e));
  if (m === 0n) return p + c; // exact zero: the sign rule of a·b + c
  const neg = m < 0n;
  let mag = neg ? -m : m;
  // Keep 53 significant bits, or fewer where the result is subnormal (ulp 2⁻¹⁰⁷⁴).
  const bits = mag.toString(2).length;
  const drop = Math.max(bits - 53, -1074 - e);
  let exp = e;
  if (drop > 0) {
    const q = mag >> BigInt(drop);
    const rem = mag - (q << BigInt(drop));
    const half = 1n << BigInt(drop - 1);
    mag = rem > half || (rem === half && (q & 1n) === 1n) ? q + 1n : q;
    exp += drop;
  }
  const r = ldexp(Number(mag), exp);
  return neg ? -r : r;
}

/**
 * dlaev2: eigen-decomposition of [[a, b], [b, c]] (rt1 has the larger |·|, (cs1, sn1) its
 * vector), with the fused multiply-adds of the compiled LAPACK (verified on 71 matrices).
 */
function dlaev2(a: number, b: number, c: number): [number, number, number, number] {
  const sm = a + c;
  const df = a - c;
  const adf = Math.abs(df);
  const tb = b + b;
  const ab = Math.abs(tb);
  let acmx: number, acmn: number;
  if (Math.abs(a) > Math.abs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  let rt: number;
  if (adf > ab) {
    const q = ab / adf;
    rt = adf * Math.sqrt(fma(q, q, 1.0));
  } else if (adf < ab) {
    const q = adf / ab;
    rt = ab * Math.sqrt(fma(q, q, 1.0));
  } else rt = ab * Math.SQRT2;
  let rt1: number, rt2: number, sgn1: number;
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
  let cs: number, sgn2: number;
  if (df >= 0.0) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }
  let cs1: number, sn1: number;
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

/** LAPACK dlamch('E') = 2⁻⁵³ and dsteqr's derived constants. */
const EPS_HALF = 2 ** -53;
const EPS2 = EPS_HALF * EPS_HALF;
const SAFMIN = 2 ** -1022;
const SSFMAX = Math.sqrt(1 / SAFMIN) / 3.0;
const SSFMIN = Math.sqrt(SAFMIN) / EPS2;

/** dsteqr (COMPZ = 'I') on the symmetric tridiagonal 2×2 matrix [[d1, e], [e, d2]]. */
function steqr2(a: Matrix): { values: number[]; Q: Matrix } {
  let d1 = a[0][0],
    d2 = a[1][1],
    e = a[1][0];
  let Z: Matrix = [
    [1.0, 0.0],
    [0.0, 1.0],
  ];
  const split =
    e === 0 || Math.abs(e) <= Math.sqrt(Math.abs(d1)) * Math.sqrt(Math.abs(d2)) * EPS_HALF;
  if (!split) {
    const anorm = Math.max(Math.abs(d1), Math.abs(d2), Math.abs(e));
    let iscale = 0;
    if (anorm > SSFMAX) iscale = 1;
    else if (anorm < SSFMIN) iscale = 2;
    if (iscale) {
      const mul = (iscale === 1 ? SSFMAX : SSFMIN) / anorm;
      d1 *= mul;
      d2 *= mul;
      e *= mul;
    }
    // QL when |d2| ≥ |d1|, else QR: the small-subdiagonal test multiplies in that order.
    const ql = !(Math.abs(d2) < Math.abs(d1));
    const tst = Math.abs(e) * Math.abs(e);
    const small = ql
      ? tst <= fma(EPS2 * Math.abs(d1), Math.abs(d2), SAFMIN)
      : tst <= fma(EPS2 * Math.abs(d2), Math.abs(d1), SAFMIN);
    if (!small) {
      const [rt1, rt2, c, s] = dlaev2(d1, e, d2);
      // dlasr('R', 'V', ·) of one rotation applied to Z = I.
      Z = [
        [c, -s],
        [s, c],
      ];
      d1 = rt1;
      d2 = rt2;
    }
    if (iscale) {
      const mul = anorm / (iscale === 1 ? SSFMAX : SSFMIN);
      d1 *= mul;
      d2 *= mul;
    }
  }
  // Selection sort into increasing order (swaps the eigenvector columns).
  if (d2 < d1) {
    return {
      values: [d2, d1],
      Q: [
        [Z[0][1], Z[0][0]],
        [Z[1][1], Z[1][0]],
      ],
    };
  }
  return { values: [d1, d2], Q: Z };
}

/** Cyclic Jacobi eigendecomposition (n ≥ 3): eigenvalues ascending, eigenvectors as columns. */
function jacobiEig(A: Matrix): { values: number[]; Q: Matrix } {
  const n = A.length;
  const a = A.map((row) => row.slice());
  const V: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0)),
  );
  for (let sweep = 0; sweep < 100; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (apq === 0) continue;
        const app = a[p][p],
          aqq = a[q][q];
        // Negligible against both diagonal entries: set to zero (classical Jacobi threshold).
        if (Math.abs(apq) <= 0.5 * EPS * 1e-3 * Math.sqrt(Math.abs(app) * Math.abs(aqq))) {
          a[p][q] = a[q][p] = 0;
          continue;
        }
        rotated = true;
        const theta = (aqq - app) / (2.0 * apq);
        const t = (theta >= 0 ? 1.0 : -1.0) / (Math.abs(theta) + Math.hypot(theta, 1.0));
        const c = 1.0 / Math.hypot(t, 1.0);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p],
            akq = a[k][q];
          a[k][p] = a[p][k] = c * akp - s * akq;
          a[k][q] = a[q][k] = s * akp + c * akq;
        }
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = a[q][p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p],
            vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    if (!rotated) break;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return { values: order.map((i) => a[i][i]), Q: V.map((row) => order.map((i) => row[i])) };
}

// ---------------------------------------------------------------------------------------
// Subproblem solvers: each returns p, whether p is on the boundary, and its own info
// ---------------------------------------------------------------------------------------

interface Subproblem {
  p: Vector;
  hitsBoundary: boolean;
  info: Record<string, unknown>;
  note: string | null;
}

const sub = (
  p: Vector,
  hitsBoundary: boolean,
  info: Record<string, unknown>,
  note: string | null = null,
): Subproblem => ({ p, hitsBoundary, info, note });

/** Cauchy point, N&W Alg. 4.2: p^C = τ p^S, p^S = −Δg/‖g‖, τ = min(‖g‖³/(Δ gᵀBg), 1) or 1. */
function cauchy(g: Vector, B: Matrix, delta: number): Subproblem {
  if (!anyNonzero(g)) return sub(new Array<number>(g.length).fill(0), false, { tau: 0.0 });
  const [gHat, e] = pow2Scaled(g);
  const gHatNorm = norm(gHat);
  const curv = dot(gHat, matvec(B, gHat)); // ĝᵀBĝ = gᵀBg/4ᵉ
  const tau = curv <= 0.0 ? 1.0 : pyMin(ldexp(gHatNorm ** 3 / curv / delta, e), 1.0);
  const p = scaled(-((tau * delta) / gHatNorm), gHat);
  return sub(p, tau >= 1.0, { tau });
}

/** Dogleg step, N&W §4.1 (eq. 4.15–4.16); the Cauchy point when B is not positive definite. */
function dogleg(g: Vector, B: Matrix, delta: number): Subproblem {
  if (!anyNonzero(g))
    return sub(new Array<number>(g.length).fill(0), false, { dogleg_path: null, tau: 0.0 });

  const cauchyFallback = (why: string): Subproblem => {
    const s = cauchy(g, B, delta);
    return sub(s.p, s.hitsBoundary, { dogleg_path: null, tau: null }, why);
  };

  const pB = newtonStep(g, B);
  if (pB === null)
    return cauchyFallback('∇²f is not positive definite: dogleg undefined, Cauchy point used');
  const [gHat] = pow2Scaled(g);
  const curv = dot(gHat, matvec(B, gHat));
  const pU = curv > 0.0 ? scaled(-(dot(gHat, gHat) / curv), g) : g.map(() => NaN);
  if (!allFinite(pU))
    return cauchyFallback('gᵀ∇²f g ≤ 0 in floating point: dogleg undefined, Cauchy point used');
  const path = [new Array<number>(g.length).fill(0), pU, pB];
  if (norm(pB) <= delta) return sub(pB, false, { dogleg_path: path, tau: 2.0 });
  const pUNorm = norm(pU);
  if (pUNorm >= delta) {
    const tau = delta / pUNorm;
    return sub(scaled(tau, pU), true, { dogleg_path: path, tau });
  }
  const diff = minus(pB, pU);
  let [, s] = boundaryTau(pU, diff, delta); // s = τ − 1 ∈ (0, 1)
  s = pyMin(pyMax(s, 0.0), 1.0);
  return sub(plus(pU, scaled(s, diff)), true, { dogleg_path: path, tau: 1.0 + s });
}

/** Steihaug–Toint truncated CG, N&W Alg. 7.2, with ε_k = min(½, √‖g‖)‖g‖. */
function steihaug(g: Vector, B: Matrix, delta: number): Subproblem {
  const n = g.length;
  let z: Vector = new Array<number>(n).fill(0);
  if (!anyNonzero(g))
    return sub(z, false, { cg_path: [z, z], cg_iters: 0, termination: 'residual', cg_tol: 0.0 });
  const gnorm = norm2(g);
  const epsK = pyMin(0.5, Math.sqrt(gnorm)) * gnorm;
  // The recurrences run on r̂ = 2⁻ᵉr and d̂ = 2⁻ᵉd (exact scaling, see the Python NOTE).
  const [r0, e] = pow2Scaled(g);
  let r = r0;
  const epsHat = pyMin(0.5, Math.sqrt(gnorm)) * norm(r);
  let d = neg(r);
  let rr = dot(r, r);
  const path: Vector[] = [z.slice()];

  const finish = (p: Vector, boundary: boolean, why: string, iters: number): Subproblem => {
    path.push(p.slice());
    return sub(p, boundary, { cg_path: path, cg_iters: iters, termination: why, cg_tol: epsK });
  };

  const maxInner = 2 * n;
  for (let j = 0; j < maxInner; j++) {
    const Bd = matvec(B, d);
    const dBd = dot(d, Bd);
    if (dBd <= 0.0) {
      const [tauMinus, tauPlus] = boundaryTau(z, d, delta);
      const candidates = [plus(z, scaled(tauMinus, d)), plus(z, scaled(tauPlus, d))];
      const values = candidates.map((p) => dot(g, p) + 0.5 * dot(p, matvec(B, p)));
      // Ties go to τ₊ ≥ 0.
      const p = values[0] < values[1] ? candidates[0] : candidates[1];
      return finish(p, true, 'negative_curvature', j + 1);
    }
    const alpha = rr / dBd;
    const step = ldexp(alpha, e);
    const zNext = z.map((v, i) => v + step * d[i]);
    if (!(norm(zNext) < delta)) {
      const [, tau] = boundaryTau(z, d, delta);
      return finish(plus(z, scaled(tau, d)), true, 'boundary', j + 1);
    }
    r = r.map((v, i) => v + alpha * Bd[i]);
    const rrNext = dot(r, r);
    z = zNext;
    if (Math.sqrt(rrNext) < epsHat) return finish(z, false, 'residual', j + 1);
    path.push(z.slice());
    const beta = rrNext / rr;
    d = r.map((v, i) => -v + beta * d[i]);
    rr = rrNext;
  }
  path.pop();
  return finish(z, false, 'max_iter', maxInner);
}

/** Global minimizer of the subproblem (N&W §4.3, Moré & Sorensen 1983), see the Python docstring. */
function exact(g: Vector, B: Matrix, delta: number): Subproblem {
  const n = g.length;
  const { values: lamAll, Q } = eigh(B);
  const QT = Q[0].map((_, j) => Q.map((row) => row[j])); // columns of Q as rows
  // γ = Qᵀg: NumPy hands the transposed product to OpenBLAS dgemv_t, which accumulates
  // s ← fma(q_ij, g_i, s) for small n (verified on 2 ≤ n ≤ 5); larger n use a plain dot.
  const gamma = QT.map((q) =>
    n <= 5 ? q.reduce((acc, v, i) => fma(v, g[i], acc), 0.0) : dot(q, g),
  );
  const lam1 = lamAll[0];
  const gnorm = norm2(g);
  const bnorm = maxAbs(lamAll);
  let q1 = QT[0].slice();
  // Make the eigenvector's sign reproducible: its largest |entry| is positive (first argmax).
  let iMax = 0;
  for (let i = 1; i < n; i++) if (Math.abs(q1[i]) > Math.abs(q1[iMax])) iMax = i;
  if (q1[iMax] < 0.0) q1 = neg(q1);

  const info = (lam: number, hard: boolean, iters: number) => ({
    lambda: Number.isNaN(lam) ? null : lam,
    lambda_min: lam1,
    hard_case: hard,
    lambda_iters: iters,
  });
  const qTimes = (w: readonly number[]): Vector => matvec(Q, w);

  // 1. Interior solution.
  if (lam1 > 0.0) {
    const p0 = neg(qTimes(gamma.map((v, j) => v / (lamAll[j] + 0.0))));
    const p0Norm = norm(p0);
    if (p0Norm <= delta) return sub(p0, false, info(0.0, false, 0));
  }

  // 2. Hard case (N&W eq. 4.45).
  const tol = 10.0 * n * EPS;
  if (lam1 <= 0.0) {
    const cluster = lamAll.map((l) => l - lam1 <= tol * bnorm);
    const gamma1 = norm2(gamma.filter((_, j) => cluster[j]));
    if (gamma1 <= tol * pyMax(bnorm * delta, gnorm)) {
      const coef = gamma.map((v, j) => (cluster[j] ? 0.0 : v / (lamAll[j] - lam1)));
      const pPerp = neg(qTimes(coef));
      const pp = norm(pPerp);
      if (pp <= delta) {
        const tau = Math.sqrt(pyMax(delta * delta - pp * pp, 0.0));
        let u: Vector;
        if (gamma1 > 0.0) {
          const masked = gamma.map((v, j) => (cluster[j] ? v : 0.0));
          u = neg(qTimes(masked)).map((t) => t / gamma1);
        } else {
          u = q1;
        }
        return sub(plus(pPerp, scaled(tau, u)), true, info(-lam1, true, 0));
      }
    }
  }

  // 3. Newton's method on the secular equation on the shift μ = λ + λ₁, safeguarded.
  const dsh = lamAll.map((l) => l - lam1); // d[0] = 0
  const pOfMu = (mu: number): Vector => neg(qTimes(gamma.map((v, j) => v / (dsh[j] + mu))));

  let lo = pyMax(lam1, 0.0);
  let hi = gnorm / delta;
  let mu = lam1 > 0.0 ? lo : hi;
  let iters = 0;
  while (iters < LAMBDA_MAX_ITER) {
    const w = gamma.map((v, j) => v / (dsh[j] + mu));
    const p = neg(qTimes(w));
    const pnorm = norm2(p);
    const qSq = npSum(w.map((t, j) => (t * t) / (dsh[j] + mu)));
    const ok = qSq > 0.0 && Number.isFinite(qSq);
    let muNew = ok ? mu + (((pnorm * pnorm) / qSq) * (pnorm - delta)) / delta : NaN;
    if (Math.abs(pnorm - delta) <= LAMBDA_RTOL * delta)
      return sub(p, true, info(mu - lam1, false, iters));
    if (pnorm <= delta) hi = mu;
    else lo = mu; // ‖p‖ > Δ, or ‖p‖ not finite
    if (!(lo < muNew && muNew <= hi)) {
      // Moré & Sorensen (1983), §3 safeguard: geometric-mean bisection.
      muNew = pyMax(Math.sqrt(lo) * Math.sqrt(hi), 1e-3 * hi);
    }
    iters++;
    if (!(lo < muNew && muNew <= hi) || muNew === mu) break;
    mu = muNew;
  }

  // 4. Safety net: move from p(hi) (‖p(hi)‖ ≤ Δ) along q₁ to the boundary.
  let p = pOfMu(hi);
  if (allFinite(p)) {
    const [tauMinus, tauPlus] = boundaryTau(p, q1, delta);
    const candidates = [plus(p, scaled(tauMinus, q1)), plus(p, scaled(tauPlus, q1))];
    const values = candidates.map((c) => dot(g, c) + 0.5 * dot(c, matvec(B, c)));
    p = values[0] < values[1] ? candidates[0] : candidates[1];
  }
  if (allFinite(p)) {
    const note = 'secular equation not solved to tolerance; boundary step p(λ) + τq₁ used';
    return sub(p, true, info(hi - lam1, false, iters), note);
  }
  const s = cauchy(g, B, delta);
  const note = 'secular equation not solvable in float64; Cauchy point used';
  return sub(s.p, s.hitsBoundary, info(NaN, false, iters), note);
}

export const SOLVERS: Record<
  TrustRegionMethod,
  (g: Vector, B: Matrix, delta: number) => Subproblem
> = {
  trust_region_cauchy: cauchy,
  trust_region_dogleg: dogleg,
  trust_region_steihaug: steihaug,
  trust_region_exact: exact,
};

export type TrustRegionMethod =
  'trust_region_cauchy' | 'trust_region_dogleg' | 'trust_region_steihaug' | 'trust_region_exact';

/** Method-specific info keys (null at k = 0). */
const EXTRA_KEYS: Record<TrustRegionMethod, string[]> = {
  trust_region_cauchy: ['tau'],
  trust_region_dogleg: ['dogleg_path', 'tau'],
  trust_region_steihaug: ['cg_path', 'cg_iters', 'termination', 'cg_tol'],
  trust_region_exact: ['lambda', 'lambda_min', 'hard_case', 'lambda_iters'],
};

// ---------------------------------------------------------------------------------------
// The shared trust-region driver (N&W Alg. 4.1)
// ---------------------------------------------------------------------------------------

export interface TrustRegionOptions {
  gtol: number;
  maxIter: number;
  radius0: number;
  maxRadius: number;
  eta: number;
}

const symmetrize = (B: Matrix): Matrix => B.map((row, i) => row.map((v, j) => 0.5 * (v + B[j][i])));

export function trustRegion(
  method: TrustRegionMethod,
  problem: SmoothProblem,
  x0: unknown,
  { gtol, maxIter, radius0, maxRadius, eta }: TrustRegionOptions,
): Result {
  if (!(gtol >= 0.0)) throw new ValueError(`gtol must be ≥ 0, got ${formatRepr(gtol)}`);
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new ValueError(`max_iter must be a positive integer, got ${maxIter}`);
  if (!(radius0 > 0.0 && radius0 < Infinity && maxRadius > 0.0 && maxRadius < Infinity))
    throw new ValueError(
      `need 0 < radius0 < ∞ and 0 < max_radius < ∞, got radius0=${formatRepr(radius0)}, max_radius=${formatRepr(maxRadius)}`,
    );
  if (!(eta >= 0.0 && eta < 0.25))
    throw new ValueError(`eta must lie in [0, 1/4), got ${formatRepr(eta)}`);
  const solve = SOLVERS[method];
  const extraKeys = EXTRA_KEYS[method];

  const { x: xStart, f, grad, hess } = resolveSmooth(problem, x0);
  let x = xStart;
  const n = x.length;
  let fx = f(x);
  let g: Vector = grad(x);
  let B: Matrix = hess(x);
  const trace: Step[] = [];
  let nRejected = 0;

  const done = (converged: boolean, message: string, k: number): Result => ({
    method,
    x: x.slice(),
    fun: fx,
    converged,
    message,
    nIter: k,
    nFev: f.n,
    nGev: grad.n,
    nHev: hess.n,
    trace,
    extra: { n_rejected: nRejected },
  });

  const baseInfo = (center: Vector, gc: Vector, Bc: Matrix, radius: number) => {
    const info: Record<string, unknown> = {
      center: center.slice(),
      grad: allFinite(gc) ? gc.slice() : null,
      H: n === 2 && allFinite(Bc) ? Bc.map((row) => row.slice()) : null,
      radius,
      new_radius: radius,
      step: null,
      step_norm: null,
      hits_boundary: null,
      predicted: null,
      actual: null,
      rho: null,
      accepted: null,
      trial_point: null,
      cauchy_point: null,
      newton_point: null,
      note: null,
    };
    for (const key of extraKeys) info[key] = null;
    return info;
  };

  // N&W Alg. 4.1 takes Δ₀ ∈ (0, Δ̂); a radius0 above max_radius is clamped (Step 0 says so).
  let radius = pyMin(radius0, maxRadius);
  const note0 =
    radius0 > maxRadius
      ? `radius0 = ${g3(radius0)} > max_radius = ${g3(maxRadius)}: Δ₀ = Δ̂ used`
      : null;
  const square = B.length === n && B.every((row) => row.length === n);
  if (!(Number.isFinite(fx) && allFinite(g, B)) || !square) {
    const gn = allFinite(g) ? norm2(g) : null;
    trace.push({
      k: 0,
      x: x.slice(),
      fun: fx,
      gradNorm: gn,
      stepSize: null,
      info: { ...baseInfo(x, g, B, radius), note: note0 },
    });
    return done(false, 'f, ∇f or ∇²f is not finite (or ∇²f has the wrong shape) at x0', 0);
  }
  B = symmetrize(B);
  trace.push({
    k: 0,
    x: x.slice(),
    fun: fx,
    gradNorm: norm2(g),
    stepSize: null,
    info: { ...baseInfo(x, g, B, radius), note: note0 },
  });

  for (let k = 1; k <= maxIter; k++) {
    const ginf = maxAbs(g);
    if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, k - 1);

    const s = solve(g, B, radius);
    const p = s.p;
    const predicted = -(dot(g, p) + 0.5 * dot(p, matvec(B, p))); // m(0) − m(p)
    if (!(predicted > 0.0 && Number.isFinite(predicted)))
      return done(
        false,
        `the model step predicts no decrease (m(0) − m(p) = ${g3(predicted)} at iteration ` +
          `${k}): the quadratic model is at its rounding level; ‖∇f‖∞ = ${g3(ginf)} > gtol`,
        k - 1,
      );
    const trial = plus(x, p);
    const fTrial = f(trial);
    let actual: number | null = null;
    let rho: number | null = null;
    if (Number.isFinite(fTrial)) {
      actual = fx - fTrial;
      const deltaRound = ROUNDING * EPS * pyMax(Math.abs(fx), Math.abs(fTrial));
      rho = (fx - fTrial + deltaRound) / (predicted + deltaRound);
    }
    const rhoValue = rho === null ? -Infinity : rho;

    // Radius update, N&W Alg. 4.1.
    let newRadius: number;
    if (rhoValue < RHO_SHRINK) newRadius = RHO_SHRINK * radius;
    else if (rhoValue > RHO_EXPAND && s.hitsBoundary) newRadius = pyMin(2.0 * radius, maxRadius);
    else newRadius = radius;
    const accepted = rhoValue > eta;

    const info = baseInfo(x, g, B, radius);
    const cauchyP = cauchy(g, B, radius).p;
    const newton = newtonStep(g, B);
    Object.assign(info, {
      new_radius: newRadius,
      step: p.slice(),
      step_norm: norm(p),
      hits_boundary: s.hitsBoundary,
      predicted,
      actual,
      rho,
      accepted,
      trial_point: trial.slice(),
      cauchy_point: plus(x, cauchyP),
      newton_point: newton === null ? null : plus(x, newton),
      note: s.note,
    });
    for (const key of extraKeys) {
      let v = s.info[key] ?? null;
      if ((key === 'dogleg_path' || key === 'cg_path') && v !== null) {
        const xc = x;
        v = (v as Vector[]).map((q) => plus(xc, q));
      }
      info[key] = v;
    }

    if (accepted) {
      const gNew = grad(trial);
      const BNew = hess(trial);
      const sq = BNew.length === n && BNew.every((row) => row.length === n);
      if (!(allFinite(gNew, BNew) && sq)) {
        x = trial;
        fx = fTrial;
        const gn = allFinite(gNew) ? norm2(gNew) : null;
        trace.push({ k, x: x.slice(), fun: fx, gradNorm: gn, stepSize: radius, info });
        return done(false, `∇f or ∇²f is not finite at the iterate of step ${k}`, k);
      }
      x = trial;
      fx = fTrial;
      g = gNew;
      B = symmetrize(BNew);
    } else {
      nRejected++;
    }
    trace.push({ k, x: x.slice(), fun: fx, gradNorm: norm2(g), stepSize: radius, info });
    radius = newRadius;
    if (radius < EPS * pyMax(1.0, norm(x)))
      return done(
        false,
        `trust radius collapsed to ${g3(radius)} (no acceptable step); ` +
          `‖∇f‖∞ = ${g3(maxAbs(g))} > gtol`,
        k,
      );
  }

  const ginf = maxAbs(g);
  if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, maxIter);
  return done(false, `reached max_iter=${maxIter}`, maxIter);
}

// ---------------------------------------------------------------------------------------
// Registered methods
// ---------------------------------------------------------------------------------------

const COMMON_REFS = ['Nocedal & Wright (2006), Numerical Optimization, Alg. 4.1'];

function trMethod(id: TrustRegionMethod): MethodFn<SmoothProblem> {
  return (problem: SmoothProblem, options: RunOptions & Params) =>
    trustRegion(id, problem, options.x0, {
      gtol: Number(options.gtol ?? 1e-8),
      maxIter: Number(options.max_iter ?? 200),
      radius0: Number(options.radius0 ?? 1.0),
      maxRadius: Number(options.max_radius ?? 100.0),
      eta: Number(options.eta ?? 0.15),
    });
}

const MODEL_TEX =
  'p_k \\approx \\arg\\min_{\\|p\\| \\le \\Delta_k}\\; f_k + \\nabla f_k^{\\mathsf T} p + \\tfrac12 p^{\\mathsf T} \\nabla^2 f_k\\, p';

const QUANTITIES: NonNullable<MethodDoc['quantities']> = [
  { tex: '\\Delta_k', key: 'info.radius', label: 'trust radius' },
  { tex: '\\rho_k', key: 'info.rho', label: 'actual / predicted' },
  { tex: '\\|p_k\\|', key: 'info.step_norm' },
];

registerMethod(
  {
    id: 'trust_region_cauchy',
    family: 'unconstrained',
    name: 'Trust region (Cauchy point)',
    params: PARAMS,
    needs: ['f', 'grad', 'hess'],
    order: 'linear (steepest descent with a model-based step length)',
    summary: 'Minimize the quadratic model along −∇f inside the trust region: the Cauchy point.',
    references: ['Nocedal & Wright (2006), Alg. 4.2, eq. 4.11–4.12', ...COMMON_REFS],
  },
  trMethod('trust_region_cauchy'),
  {
    rule: `${MODEL_TEX},\\quad p_k = -\\tau_k \\frac{\\Delta_k}{\\|\\nabla f_k\\|}\\nabla f_k`,
    intuition:
      'Trust the quadratic model only inside a ball, and take its minimizer along the ' +
      'steepest-descent ray. Good steps grow the ball; poor ones shrink it.',
    quantities: [...QUANTITIES, { tex: '\\tau_k', key: 'info.tau' }],
  },
);

registerMethod(
  {
    id: 'trust_region_dogleg',
    family: 'unconstrained',
    name: 'Trust region (dogleg)',
    params: PARAMS,
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic near a minimizer with ∇²f ≻ 0',
    summary:
      'Follow the two-segment path 0 → steepest-descent minimizer → Newton point ' +
      'until it leaves the trust region.',
    references: ['Nocedal & Wright (2006), §4.1, eq. 4.15–4.16', 'Powell (1970)', ...COMMON_REFS],
  },
  trMethod('trust_region_dogleg'),
  {
    rule: `${MODEL_TEX},\\quad p_k = \\tilde p(\\tau),\\ 0 \\to p^U \\to p^B`,
    intuition:
      'Walk from the centre toward the steepest-descent minimizer, then turn toward the Newton ' +
      'point, and stop where the path leaves the ball.',
    quantities: [...QUANTITIES, { tex: '\\tau', key: 'info.tau' }],
  },
);

registerMethod(
  {
    id: 'trust_region_steihaug',
    family: 'unconstrained',
    name: 'Trust region (Steihaug–Toint CG)',
    params: PARAMS,
    needs: ['f', 'grad', 'hess'],
    order: 'superlinear (forcing term min(½, √‖g‖))',
    summary:
      'Run CG on the Newton equations and stop at the trust-region boundary, at ' +
      'negative curvature, or when the residual is small.',
    references: [
      'Nocedal & Wright (2006), Alg. 7.2 (CG–Steihaug)',
      'Steihaug (1983), SIAM J. Numer. Anal. 20(3), 626–637',
      'Toint (1981), in Sparse Matrices and Their Uses, 57–88',
      ...COMMON_REFS,
    ],
  },
  trMethod('trust_region_steihaug'),
  {
    rule: `${MODEL_TEX},\\quad \\text{CG on } \\nabla^2 f_k\\, p = -\\nabla f_k \\text{ from } z_0 = 0`,
    intuition:
      'Solve the Newton equations by conjugate gradients, but stop at the edge of the ball or ' +
      'as soon as a direction of negative curvature appears.',
    quantities: [...QUANTITIES, { tex: 'j', key: 'info.cg_iters', label: 'inner CG steps' }],
  },
);

registerMethod(
  {
    id: 'trust_region_exact',
    family: 'unconstrained',
    name: 'Trust region (exact subproblem, Moré–Sorensen)',
    params: PARAMS,
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic near a minimizer with ∇²f ≻ 0',
    summary:
      'Solve the trust-region subproblem exactly: find λ ≥ 0 with (∇²f + λI)p = −∇f ' +
      'and ‖p‖ = Δ, including the hard case.',
    references: [
      'Nocedal & Wright (2006), Alg. 4.3, Thm. 4.1, eq. 4.38–4.45',
      'Moré & Sorensen (1983), SIAM J. Sci. Stat. Comput. 4(3), 553–572',
      ...COMMON_REFS,
    ],
  },
  trMethod('trust_region_exact'),
  {
    rule: `(\\nabla^2 f_k + \\lambda I)\\,p_k = -\\nabla f_k,\\quad \\lambda \\ge 0,\\quad \\lambda(\\Delta_k - \\|p_k\\|) = 0`,
    intuition:
      'Find the exact minimizer of the model in the ball by tuning the shift λ until the ' +
      'shifted Newton step just reaches the boundary.',
    quantities: [...QUANTITIES, { tex: '\\lambda', key: 'info.lambda' }],
  },
);
