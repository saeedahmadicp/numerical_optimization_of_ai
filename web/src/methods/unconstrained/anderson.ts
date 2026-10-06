/**
 * Anderson acceleration of gradient descent, AA(m), with the RNA term — TS port of
 * `numopt.unconstrained.anderson` (src/numopt/unconstrained/anderson.py).
 *
 * Gradient descent with the fixed step α is the fixed-point iteration x ← g(x) = x − α∇f(x). AA(m)
 * combines the residuals f_i = g(x_i) − x_i = −α∇f(x_i) of the last m_k + 1 iterates (Walker & Ni
 * (2011), Alg. AA and eq. (1.1); the Tikhonov term λ' is the RNA regularization of Scieur,
 * d'Aspremont & Bach (2016), Alg. 2):
 *
 *     c* = argmin_{1ᵀc = 1} ‖F_k c‖₂² + λ'‖c‖₂²,        λ' = λ‖F_k‖₂²,
 *     x_{k+1} = Σ_i c*_i [(1 − β) x_i + β g(x_i)],
 *
 * solved in the unconstrained difference form γ* = argmin ‖[f_k; √λ' e] − [ΔF; −√λ' D] γ‖₂,
 * x_{k+1} = x_k + β f_k − (ΔX + βΔF) γ*, c* = e + Dγ*.
 *
 * Stopping test (converged): ‖∇f(x_k)‖₂ ≤ gtol and ∇²f(x_k) has no eigenvalue below
 * −max(tol_H, √(‖∇f‖₂·|λ|_max)). Failures: a saddle point or maximizer, a non-finite value, a
 * failed solve, divergence ‖x_k‖₂ > 10¹²·max(1, ‖x₀‖₂), a stall, max_iter. Counts: one ∇f and one
 * f per iterate, one ∇²f at a point that passes the gradient test.
 *
 * Step.info keys (snake_case, as in Python): grad, alpha, memory, coefficients, history, x_bar,
 * lsq_residual, cond, lam_eff, direction, hess_eigs.
 *
 * NOTE (port): `numpy.linalg.lstsq` is a bit-exact port of the LAPACK dgelsd path that the NumPy
 * wheel runs (see `lstsq` below). An approximation that agrees only to rounding is not enough:
 * on rosenbrock a one-ulp change in γ* at iteration 7 changed the stop from 176 to 433
 * iterations. ‖F‖₂ (for λ' with λ > 0) uses a one-sided Jacobi SVD, and `eigvalsh` uses the
 * eigensolver of trust_region.ts. Both agree with LAPACK to rounding.
 */
import { registerMethod, param } from '../../core/registry';
import type { Matrix, MethodFn, Params, Result, RunOptions, Step, Vector } from '../../core/types';
import { formatG, formatRepr } from '../line_search/methods';
import { ValueError, resolveSmooth, type SmoothProblem } from './conjugate_gradient';
import { eigh, fma } from './trust_region';

/** Stop with converged=false when ‖x_k‖₂ > DIVERGENCE_FACTOR · max(1, ‖x₀‖₂). */
const DIVERGENCE_FACTOR = 1e12;
const EPS = Number.EPSILON;
/** Relative accuracy of a central-difference Hessian (ε^{1/3}). */
const FD_HESS_REL = EPS ** (1.0 / 3.0);

const g3 = (v: number) => formatG(v, 3);

// ---------------------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------------------

function dotv(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** `float(np.linalg.norm(v))`: √(vᵀv). */
const norm = (v: readonly number[]) => Math.sqrt(dotv(v, v));

/**
 * Row · vector as NumPy's C-contiguous `A @ v` rounds it: OpenBLAS dgemv_t (NEOVERSEN2 kernel
 * `gemv_t_sve_v1x3.c`, 128-bit SVE) keeps two lanes of fused multiply-adds, one over the even
 * and one over the odd entries, and adds the two lanes at the end. Checked bit for bit against
 * `M @ v` for 2 ≤ rows ≤ 10 and 1 ≤ cols ≤ 6.
 */
function rowDot(row: readonly number[], v: readonly number[]): number {
  const m = row.length;
  let l0 = 0.0;
  let l1 = 0.0;
  let i = 0;
  for (; i + 1 < m; i += 2) {
    l0 = fmaFast(row[i], v[i], l0);
    l1 = fmaFast(row[i + 1], v[i + 1], l1);
  }
  if (i < m) l0 = fmaFast(row[i], v[i], l0);
  return l0 + l1;
}

export const matvec = (A: Matrix, v: readonly number[]): Vector => A.map((row) => rowDot(row, v));

const allFinite = (...vs: (number | readonly number[] | Matrix)[]): boolean =>
  vs.every((v) =>
    typeof v === 'number'
      ? Number.isFinite(v)
      : (v as readonly (number | readonly number[])[]).every((e) =>
          typeof e === 'number' ? Number.isFinite(e) : e.every((t) => Number.isFinite(t)),
        ),
  );

/** The columns of a matrix given by rows. */
const columns = (A: Matrix): Matrix =>
  A.length === 0 ? [] : A[0].map((_, j) => A.map((row) => row[j]));

/**
 * One-sided Jacobi (Hestenes) on the columns of a tall matrix given as columns C (m × n, m ≥ n):
 * C V = W with orthogonal columns. Returns W's columns, V (n × n, by rows) and σ_i = ‖w_i‖.
 */
function hestenes(C: Matrix, m: number): { W: Matrix; V: Matrix; s: Vector } {
  const n = C.length;
  const W = C.map((c) => c.slice());
  const V: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1.0 : 0.0)),
  );
  for (let sweep = 0; sweep < 80; sweep++) {
    let rotated = false;
    for (let i = 0; i < n - 1; i++)
      for (let j = i + 1; j < n; j++) {
        const a = W[i];
        const b = W[j];
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let r = 0; r < m; r++) {
          alpha += a[r] * a[r];
          beta += b[r] * b[r];
          gamma += a[r] * b[r];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.hypot(1, zeta));
        const c = 1 / Math.hypot(1, t);
        const sn = c * t;
        for (let r = 0; r < m; r++) {
          const ar = a[r];
          const br = b[r];
          a[r] = c * ar - sn * br;
          b[r] = sn * ar + c * br;
        }
        for (let r = 0; r < n; r++) {
          const vi = V[r][i];
          const vj = V[r][j];
          V[r][i] = c * vi - sn * vj;
          V[r][j] = sn * vi + c * vj;
        }
      }
    if (!rotated) break;
  }
  return { W, V, s: W.map((w) => norm(w)) };
}

/** Singular values of A (rows × cols), descending (`np.linalg.svd(A, compute_uv=False)`). */
function singularValues(A: Matrix): Vector {
  const rows = A.length;
  const cols = rows ? A[0].length : 0;
  const tall = rows >= cols;
  const C = tall ? columns(A) : A.map((r) => r.slice());
  return hestenes(C, tall ? rows : cols).s.sort((a, b) => b - a);
}

// ---------------------------------------------------------------------------------------
// np.linalg.lstsq, bit for bit: LAPACK dgelsd as the NumPy wheel runs it
// ---------------------------------------------------------------------------------------
//
// The AA iteration is chaotic on a curved valley (rosenbrock): a one-ulp difference in γ* at one
// step changes the iteration count. So `lstsq` reproduces `numpy.linalg.lstsq` exactly, not only
// to rounding: the reference LAPACK 3.12 routines of the OpenBLAS 0.3.34 wheel (dgelsd → dgeqr2 /
// dgelq2 → dgebd2 → dlalsd → dlasdq → dbdsqr), with
//   - the fused multiply-adds that gfortran's contraction put into dlartg, dlasv2, dlas2, dlapy2,
//     dlasr and dbdsqr (read from the disassembly of the wheel's library), and
//   - the OpenBLAS NEOVERSEN2 kernels they call (dnrm2 `nrm2.S`, dgemv `gemv_[nt]_sve_v1x3.c` with
//     128-bit SVE, daxpy/dger with fmadd, drot `rot.S`, dgemm with n = 1 forwarded to dgemv_t).
// Checked bit for bit against np.linalg.lstsq on 25,000 random m × n systems (1 ≤ m ≤ 10,
// 1 ≤ n ≤ 20, tall, square and wide, rank-deficient and scaled) and every lstsq call of the
// rosenbrock run. Arrays are column-major with 0-based offsets, as in the Fortran.

/** DLAMCH('E') = 2⁻⁵³, DLAMCH('P') = 2⁻⁵², DLAMCH('S') = 2⁻¹⁰²². */
const LA_EPS = 2 ** -53;
const LA_PREC = 2 ** -52;
const SAFMIN = 2 ** -1022;
const SAFMAX = 1 / SAFMIN;
const LA_HUGE = Number.MAX_VALUE;
/** dbdsqr: TOL = max(10, min(100, ε^(−1/8)))·ε with ε = DLAMCH('E'). */
const BDSQR_TOL = Math.max(10.0, Math.min(100.0, LA_EPS ** -0.125)) * LA_EPS;

/** Fortran SIGN(a, b): |a| with the sign bit of b. */
const fsign = (a: number, b: number) => (b < 0 || Object.is(b, -0) ? -Math.abs(a) : Math.abs(a));

// --- fma(a, b, c) with one rounding, fast path ---------------------------------------
//
// Boldo & Melquiond (2008), IEEE Trans. Comput. 57(4), Alg. 5.4: an exact product (Dekker), two
// exact sums (Knuth), one sum rounded to odd, one sum rounded to nearest. Inputs where the
// error-free transforms could overflow or underflow go to the exact BigInt `fma` of trust_region.

const SPLITTER = 134217729; // 2^27 + 1
const RO_BUF = new DataView(new ArrayBuffer(8));

/** x + y rounded to odd (the neighbor with an odd last bit when the sum is inexact). */
function addRoundOdd(x: number, y: number): number {
  const s = x + y;
  const bb = s - x;
  const err = x - (s - bb) + (y - bb);
  if (err === 0) return s;
  RO_BUF.setFloat64(0, s);
  let hi = RO_BUF.getUint32(0);
  let lo = RO_BUF.getUint32(4);
  if ((lo & 1) === 1) return s;
  // s is even and inexact: step one ulp toward the exact value.
  if (err > 0 === s > 0) {
    lo += 1; // lo is even: no carry
  } else if (lo === 0) {
    lo = 0xffffffff;
    hi -= 1;
  } else {
    lo -= 1;
  }
  RO_BUF.setUint32(0, hi);
  RO_BUF.setUint32(4, lo >>> 0);
  return RO_BUF.getFloat64(0);
}

const FMA_BIG = 2 ** 995;
const FMA_SMALL = 2 ** -900;

/** round(a·b + c), one rounding (IEEE fusedMultiplyAdd). */
export function fmaFast(a: number, b: number, c: number): number {
  const p = a * b;
  if (!Number.isFinite(p) || !Number.isFinite(c) || a === 0 || b === 0 || c === 0) return p + c;
  const ap = Math.abs(p);
  const ac = Math.abs(c);
  if (
    Math.abs(a) > FMA_BIG ||
    Math.abs(b) > FMA_BIG ||
    ap > FMA_BIG ||
    ac > FMA_BIG ||
    ap < FMA_SMALL ||
    ac < FMA_SMALL
  )
    return fma(a, b, c);
  // Dekker: a·b = p + pl exactly.
  let t = SPLITTER * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = SPLITTER * b;
  const bh = t - (t - b);
  const bl = b - bh;
  const pl = ah * bh - p + ah * bl + al * bh + al * bl;
  // (th, tl) = c + pl, (vh, vl) = p + th, both exact.
  const th = c + pl;
  let bb = th - c;
  const tl = c - (th - bb) + (pl - bb);
  const vh = p + th;
  bb = vh - p;
  const vl = p - (vh - bb) + (th - bb);
  return vh + addRoundOdd(tl, vl);
}

// --- BLAS: the OpenBLAS NEOVERSEN2 kernels -------------------------------------------

/** dnrm2 (`nrm2.S`, one-pass scaled sum of squares; n = 1 returns |x₁|). */
function dnrm2(n: number, x: number[], ox: number, inc: number): number {
  if (n <= 0) return 0.0;
  if (n === 1) return Math.abs(x[ox]);
  let scale = 0.0;
  let ssq = 1.0;
  for (let k = 0; k < n; k++) {
    const v = x[ox + k * inc];
    if (v === 0.0) continue;
    const a = Math.abs(v);
    if (scale >= a) {
      const d = a / scale;
      ssq = fmaFast(d, d, ssq);
    } else {
      let d = scale / a;
      d = d * d;
      ssq = 1.0 + ssq * d;
      scale = a;
    }
  }
  return scale * Math.sqrt(ssq);
}

function dscal(n: number, a: number, x: number[], ox: number, inc: number) {
  if (n <= 0 || inc <= 0 || a === 1.0) return;
  for (let k = 0; k < n; k++) x[ox + k * inc] *= a;
}

/** daxpy: y ← fma(x, α, y). */
function daxpy(
  n: number,
  da: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
) {
  if (n <= 0 || da === 0.0) return;
  for (let k = 0; k < n; k++) y[oy + k * incy] = fmaFast(x[ox + k * incx], da, y[oy + k * incy]);
}

/**
 * y ← Aᵀx (α = 1, β = 0), A m × n. Unit stride: two SVE lanes (even / odd rows) of fused
 * multiply-adds, then lane 0 + lane 1.
 */
function dgemvT(
  m: number,
  n: number,
  A: number[],
  oa: number,
  lda: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
) {
  if (m === 0 || n === 0) return;
  for (let j = 0; j < n; j++) {
    const col = oa + j * lda;
    let s: number;
    if (incx === 1) {
      let l0 = 0.0;
      let l1 = 0.0;
      let i = 0;
      for (; i + 1 < m; i += 2) {
        l0 = fmaFast(A[col + i], x[ox + i], l0);
        l1 = fmaFast(A[col + i + 1], x[ox + i + 1], l1);
      }
      if (i < m) l0 = fmaFast(A[col + i], x[ox + i], l0);
      s = l0 + l1;
    } else {
      s = 0.0;
      for (let i = 0; i < m; i++) s = fmaFast(A[col + i], x[ox + i * incx], s);
    }
    y[oy + j * incy] = 0.0 + s;
  }
}

/**
 * y ← Ax (α = 1, β = 0), A m × n. Unit stride: columns in the kernel's order (j, j + w, j + 2w
 * for w = ⌊n/3⌋, then the rest), each a fused multiply-add into y.
 */
function dgemvN(
  m: number,
  n: number,
  A: number[],
  oa: number,
  lda: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
) {
  if (m === 0 || n === 0) return;
  for (let i = 0; i < m; i++) y[oy + i * incy] = 0.0;
  const order: number[] = [];
  if (incy === 1) {
    const w = Math.floor(n / 3);
    for (let j = 0; j < w; j++) order.push(j, j + w, j + 2 * w);
    for (let j = 3 * w; j < n; j++) order.push(j);
  } else {
    for (let j = 0; j < n; j++) order.push(j);
  }
  for (const c of order) {
    const t = x[ox + c * incx];
    for (let i = 0; i < m; i++)
      y[oy + i * incy] = fmaFast(t, A[oa + c * lda + i], y[oy + i * incy]);
  }
}

/** A ← A + α x yᵀ (generic `ger.c` over the daxpy kernel). */
function dger(
  m: number,
  n: number,
  alpha: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
  A: number[],
  oa: number,
  lda: number,
) {
  if (m === 0 || n === 0 || alpha === 0.0) return;
  for (let j = 0; j < n; j++) {
    const s = alpha * y[oy + j * incy];
    for (let i = 0; i < m; i++)
      A[oa + j * lda + i] = fmaFast(x[ox + i * incx], s, A[oa + j * lda + i]);
  }
}

/** drot (`rot.S`): x ← fma(s, y, c·x), y ← fma(−s, x, c·y). */
function drot(
  n: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
  c: number,
  s: number,
) {
  for (let k = 0; k < n; k++) {
    const xv = x[ox + k * incx];
    const yv = y[oy + k * incy];
    x[ox + k * incx] = fmaFast(s, yv, c * xv);
    y[oy + k * incy] = fmaFast(-s, xv, c * yv);
  }
}

function dswap(
  n: number,
  x: number[],
  ox: number,
  incx: number,
  y: number[],
  oy: number,
  incy: number,
) {
  for (let k = 0; k < n; k++) {
    const t = x[ox + k * incx];
    x[ox + k * incx] = y[oy + k * incy];
    y[oy + k * incy] = t;
  }
}

// --- LAPACK auxiliaries ------------------------------------------------------------------

function dlapy2(x: number, y: number): number {
  if (Number.isNaN(x)) return x;
  if (Number.isNaN(y)) return y;
  const w = Math.max(Math.abs(x), Math.abs(y));
  const z = Math.min(Math.abs(x), Math.abs(y));
  if (z === 0.0 || w > LA_HUGE) return w;
  const q = z / w;
  return w * Math.sqrt(fmaFast(q, q, 1.0));
}

/** dlarfg: returns [β (the new α), τ]; scales x in place. */
function dlarfg(n: number, alpha: number, x: number[], ox: number, inc: number): [number, number] {
  if (n <= 1) return [alpha, 0.0];
  let xnorm = dnrm2(n - 1, x, ox, inc);
  if (xnorm === 0.0) return [alpha, 0.0];
  let beta = -fsign(dlapy2(alpha, xnorm), alpha);
  const safmin = SAFMIN / LA_EPS;
  let knt = 0;
  if (Math.abs(beta) < safmin) {
    const rsafmn = 1.0 / safmin;
    do {
      knt++;
      dscal(n - 1, rsafmn, x, ox, inc);
      beta *= rsafmn;
      alpha *= rsafmn;
    } while (Math.abs(beta) < safmin && knt < 20);
    xnorm = dnrm2(n - 1, x, ox, inc);
    beta = -fsign(dlapy2(alpha, xnorm), alpha);
  }
  const tau = (beta - alpha) / beta;
  dscal(n - 1, 1.0 / (alpha - beta), x, ox, inc);
  for (let j = 0; j < knt; j++) beta *= safmin;
  return [beta, tau];
}

/** iladlc: the last non-zero column of C (m × n). */
function iladlc(m: number, n: number, C: number[], oc: number, ldc: number): number {
  if (n === 0) return 0;
  if (C[oc + (n - 1) * ldc] !== 0.0 || C[oc + (n - 1) * ldc + m - 1] !== 0.0) return n;
  for (let j = n; j >= 1; j--)
    for (let i = 0; i < m; i++) if (C[oc + (j - 1) * ldc + i] !== 0.0) return j;
  return 0;
}

/** iladlr: the last non-zero row of C (m × n). */
function iladlr(m: number, n: number, C: number[], oc: number, ldc: number): number {
  if (m === 0) return 0;
  if (C[oc + m - 1] !== 0.0 || C[oc + (n - 1) * ldc + m - 1] !== 0.0) return m;
  let res = 0;
  for (let j = 0; j < n; j++) {
    let i = m;
    while (i >= 1 && C[oc + j * ldc + i - 1] === 0.0) i--;
    res = Math.max(res, i);
  }
  return res;
}

/** dlarf1f: apply H = I − τvvᵀ (v₁ = 1 implicit) from the left or the right to C (m × n). */
function dlarf1f(
  left: boolean,
  m: number,
  n: number,
  V: number[],
  ov: number,
  incv: number,
  tau: number,
  C: number[],
  oc: number,
  ldc: number,
) {
  let lastv = 1;
  let lastc = 0;
  if (tau !== 0.0) {
    lastv = left ? m : n;
    let i = incv > 0 ? ov + (lastv - 1) * incv : ov;
    while (lastv > 1 && V[i] === 0.0) {
      lastv--;
      i -= incv;
    }
    lastc = left ? iladlc(lastv, n, C, oc, ldc) : iladlr(m, lastv, C, oc, ldc);
  }
  if (lastc === 0) return;
  const work = new Array<number>(lastc).fill(0.0);
  if (left) {
    if (lastv === 1) {
      dscal(lastc, 1.0 - tau, C, oc, ldc);
    } else {
      dgemvT(lastv - 1, lastc, C, oc + 1, ldc, V, ov + incv, incv, work, 0, 1);
      daxpy(lastc, 1.0, C, oc, ldc, work, 0, 1);
      daxpy(lastc, -tau, work, 0, 1, C, oc, ldc);
      dger(lastv - 1, lastc, -tau, V, ov + incv, incv, work, 0, 1, C, oc + 1, ldc);
    }
  } else if (lastv === 1) {
    dscal(lastc, 1.0 - tau, C, oc, 1);
  } else {
    dgemvN(lastc, lastv - 1, C, oc + ldc, ldc, V, ov + incv, incv, work, 0, 1);
    daxpy(lastc, 1.0, C, oc, 1, work, 0, 1);
    daxpy(lastc, -tau, work, 0, 1, C, oc, 1);
    dger(lastc, lastv - 1, -tau, work, 0, 1, V, ov + incv, incv, C, oc + ldc, ldc);
  }
}

/** dgeqr2: A = QR (m × n), Householder vectors below the diagonal. */
function dgeqr2(m: number, n: number, A: number[], lda: number, tau: number[]) {
  for (let i = 0; i < Math.min(m, n); i++) {
    const d = i * lda + i;
    [A[d], tau[i]] = dlarfg(m - i, A[d], A, i * lda + Math.min(i + 1, m - 1), 1);
    if (i < n - 1) dlarf1f(true, m - i, n - i - 1, A, d, 1, tau[i], A, d + lda, lda);
  }
}

/** dgelq2: A = LQ (m × n), Householder vectors right of the diagonal. */
function dgelq2(m: number, n: number, A: number[], lda: number, tau: number[]) {
  for (let i = 0; i < Math.min(m, n); i++) {
    const d = i * lda + i;
    [A[d], tau[i]] = dlarfg(n - i, A[d], A, Math.min(i + 1, n - 1) * lda + i, lda);
    if (i < m - 1) dlarf1f(false, m - i - 1, n - i, A, d, lda, tau[i], A, d + 1, lda);
  }
}

/** dgebd2: Qᵀ A P = B bidiagonal (upper if m ≥ n, else lower), diagonal D, off-diagonal E. */
function dgebd2(
  m: number,
  n: number,
  A: number[],
  lda: number,
  D: number[],
  E: number[],
  tauq: number[],
  taup: number[],
) {
  if (m >= n) {
    for (let i = 0; i < n; i++) {
      const d = i * lda + i;
      [A[d], tauq[i]] = dlarfg(m - i, A[d], A, i * lda + Math.min(i + 1, m - 1), 1);
      D[i] = A[d];
      if (i < n - 1) {
        dlarf1f(true, m - i, n - i - 1, A, d, 1, tauq[i], A, d + lda, lda);
        const e = d + lda;
        [A[e], taup[i]] = dlarfg(n - i - 1, A[e], A, Math.min(i + 2, n - 1) * lda + i, lda);
        E[i] = A[e];
        dlarf1f(false, m - i - 1, n - i - 1, A, e, lda, taup[i], A, e + 1, lda);
      } else {
        taup[i] = 0.0;
      }
    }
  } else {
    for (let i = 0; i < m; i++) {
      const d = i * lda + i;
      [A[d], taup[i]] = dlarfg(n - i, A[d], A, Math.min(i + 1, n - 1) * lda + i, lda);
      D[i] = A[d];
      if (i < m - 1) {
        dlarf1f(false, m - i - 1, n - i, A, d, lda, taup[i], A, d + 1, lda);
        const e = d + 1;
        [A[e], tauq[i]] = dlarfg(m - i - 1, A[e], A, i * lda + Math.min(i + 2, m - 1), 1);
        E[i] = A[e];
        dlarf1f(true, m - i - 1, n - i - 1, A, e, 1, tauq[i], A, e + lda, lda);
      } else {
        tauq[i] = 0.0;
      }
    }
  }
}

/** dorm2r('L', 'T') on one right-hand side: b ← Qᵀb, k column reflectors at A(o + i(lda + 1)). */
function applyQt(
  m: number,
  k: number,
  A: number[],
  oa: number,
  lda: number,
  tau: number[],
  b: number[],
  ob: number,
) {
  for (let i = 0; i < k; i++)
    dlarf1f(true, m - i, 1, A, oa + i * lda + i, 1, tau[i], b, ob + i, Math.max(m, 1));
}

/** dorml2('L', 'T') on one right-hand side: b ← Qᵀb for k row reflectors (applied k..1). */
function applyLqT(
  m: number,
  k: number,
  A: number[],
  oa: number,
  lda: number,
  tau: number[],
  b: number[],
  ob: number,
) {
  for (let i = k - 1; i >= 0; i--)
    dlarf1f(true, m - i, 1, A, oa + i * lda + i, lda, tau[i], b, ob + i, Math.max(m, 1));
}

/** dlartg (LAPACK 3.10+): [c, s, r] with c·f + s·g = r, −s·f + c·g = 0. */
function dlartg(f: number, g: number): [number, number, number] {
  const rtmin = Math.sqrt(SAFMIN);
  const rtmax = Math.sqrt(SAFMAX / 2);
  const f1 = Math.abs(f);
  const g1 = Math.abs(g);
  if (g === 0.0) return [1.0, 0.0, f];
  if (f === 0.0) return [0.0, fsign(1.0, g), g1];
  if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    const d = Math.sqrt(fmaFast(f, f, g * g));
    const r = fsign(d, f);
    return [f1 / d, g / r, r];
  }
  const u = Math.min(SAFMAX, Math.max(SAFMIN, f1, g1));
  const fs = f / u;
  const gs = g / u;
  const d = Math.sqrt(fmaFast(fs, fs, gs * gs));
  const r = fsign(d, f);
  return [Math.abs(fs) / d, gs / r, r * u];
}

/** dlas2: the singular values [σ_min, σ_max] of [[f, g], [0, h]]. */
function dlas2(f: number, g: number, h: number): [number, number] {
  const fa = Math.abs(f);
  const ga = Math.abs(g);
  const ha = Math.abs(h);
  const fhmn = Math.min(fa, ha);
  const fhmx = Math.max(fa, ha);
  if (fhmn === 0.0) {
    if (fhmx === 0.0) return [0.0, ga];
    const mx = Math.max(fhmx, ga);
    const q = Math.min(fhmx, ga) / mx;
    return [0.0, mx * Math.sqrt(fmaFast(q, q, 1.0))];
  }
  if (ga < fhmx) {
    const as = 1.0 + fhmn / fhmx;
    const at = (fhmx - fhmn) / fhmx;
    const q = ga / fhmx;
    const c = 2.0 / (Math.sqrt(fmaFast(q, q, as * as)) + Math.sqrt(fmaFast(q, q, at * at)));
    return [fhmn * c, fhmx / c];
  }
  const au = fhmx / ga;
  if (au === 0.0) return [(fhmn * fhmx) / ga, ga];
  const as = 1.0 + fhmn / fhmx;
  const at = (fhmx - fhmn) / fhmx;
  const p = as * au;
  const q = at * au;
  const c = 1.0 / (Math.sqrt(fmaFast(p, p, 1.0)) + Math.sqrt(fmaFast(q, q, 1.0)));
  let ssmin = fhmn * c * au;
  ssmin = ssmin + ssmin;
  return [ssmin, ga / (c + c)];
}

/** dlasv2: the SVD of [[f, g], [0, h]] as [σ_min, σ_max, sn_r, cs_r, sn_l, cs_l]. */
function dlasv2(f: number, g: number, h: number): [number, number, number, number, number, number] {
  let ft = f;
  let fa = Math.abs(f);
  let ht = h;
  let ha = Math.abs(h);
  let pmax = 1;
  const swap = ha > fa;
  if (swap) {
    pmax = 3;
    [ft, ht] = [ht, ft];
    [fa, ha] = [ha, fa];
  }
  const gt = g;
  const ga = Math.abs(g);
  let ssmin: number;
  let ssmax: number;
  let clt = 1.0;
  let crt = 1.0;
  let slt = 0.0;
  let srt = 0.0;
  if (ga === 0.0) {
    ssmin = ha;
    ssmax = fa;
  } else {
    let gasmal = true;
    ssmin = 0.0;
    ssmax = 0.0;
    if (ga > fa) {
      pmax = 2;
      if (fa / ga < LA_EPS) {
        gasmal = false;
        ssmax = ga;
        ssmin = ha > 1.0 ? fa / (ga / ha) : (fa / ga) * ha;
        clt = 1.0;
        slt = ht / gt;
        srt = 1.0;
        crt = ft / gt;
      }
    }
    if (gasmal) {
      const d = fa - ha;
      let l = d === fa ? 1.0 : d / fa;
      const m = gt / ft;
      let t = 2.0 - l;
      const mm = m * m;
      const s = Math.sqrt(fmaFast(t, t, mm));
      const r = l === 0.0 ? Math.abs(m) : Math.sqrt(fmaFast(l, l, mm));
      const a = 0.5 * (s + r);
      ssmin = ha / a;
      ssmax = fa * a;
      if (mm === 0.0) {
        t = l === 0.0 ? fsign(2.0, ft) * fsign(1.0, gt) : gt / fsign(d, ft) + m / t;
      } else {
        t = (m / (s + t) + m / (r + l)) * (1.0 + a);
      }
      l = Math.sqrt(fmaFast(t, t, 4.0));
      crt = 2.0 / l;
      srt = t / l;
      clt = fmaFast(srt, m, crt) / a;
      slt = ((ht / ft) * srt) / a;
    }
  }
  const [csl, snl, csr, snr] = swap ? [srt, crt, slt, clt] : [clt, slt, crt, srt];
  let tsign: number;
  if (pmax === 1) tsign = fsign(1.0, csr) * fsign(1.0, csl) * fsign(1.0, f);
  else if (pmax === 2) tsign = fsign(1.0, snr) * fsign(1.0, csl) * fsign(1.0, g);
  else tsign = fsign(1.0, snr) * fsign(1.0, snl) * fsign(1.0, h);
  ssmax = fsign(ssmax, tsign);
  ssmin = fsign(ssmin, tsign * fsign(1.0, f) * fsign(1.0, h));
  return [ssmin, ssmax, snr, csr, snl, csl];
}

/** dlasr('L', 'V', direct): plane rotations (c, s) = (w[oc + j], w[os + j]) on rows of A (m × n). */
function dlasrLV(
  forward: boolean,
  m: number,
  n: number,
  w: number[],
  oc: number,
  os: number,
  A: number[],
  oa: number,
  lda: number,
) {
  for (let jj = 0; jj < m - 1; jj++) {
    const j = forward ? jj : m - 2 - jj;
    const ct = w[oc + j];
    const st = w[os + j];
    if (ct === 1.0 && st === 0.0) continue;
    for (let i = 0; i < n; i++) {
      const p = oa + i * lda + j;
      const temp = A[p + 1];
      A[p + 1] = fmaFast(ct, temp, -(st * A[p]));
      A[p] = fmaFast(st, temp, ct * A[p]);
    }
  }
}

/**
 * dbdsqr('U', n, ncvt, nru = 0, ncc): the SVD of the upper bidiagonal (D, E) by implicit
 * zero-shift / shifted QR, applied to the rows of VT (n × ncvt) and C (n × ncc). Returns INFO.
 */
function dbdsqr(
  n: number,
  D: number[],
  E: number[],
  ncvt: number,
  VT: number[],
  ldvt: number,
  ncc: number,
  C: number[],
  ldc: number,
): number {
  if (n === 0) return 0;
  if (n > 1) {
    const maxitr = 6;
    const nm1 = n - 1;
    const nm12 = nm1 + nm1;
    const nm13 = nm12 + nm1;
    const work = new Array<number>(4 * n).fill(0.0);
    const tol = BDSQR_TOL;
    let idir = 0;
    let smax = 0.0;
    for (let i = 0; i < n; i++) smax = Math.max(smax, Math.abs(D[i]));
    for (let i = 0; i < n - 1; i++) smax = Math.max(smax, Math.abs(E[i]));
    let sminoa = Math.abs(D[0]);
    if (sminoa !== 0.0) {
      let mu = sminoa;
      for (let i = 1; i < n; i++) {
        mu = Math.abs(D[i]) * (mu / (mu + Math.abs(E[i - 1])));
        sminoa = Math.min(sminoa, mu);
        if (sminoa === 0.0) break;
      }
    }
    sminoa = sminoa / Math.sqrt(n);
    const thresh = Math.max(tol * sminoa, maxitr * (n * (n * SAFMIN)));
    const maxitdivn = maxitr * n;
    let iterdivn = 0;
    let iter = -1;
    let oldll = -1;
    let oldm = -1;
    let M = n; // 1-based, as in the Fortran
    let smin: number;
    for (;;) {
      if (M <= 1) break;
      if (iter >= n) {
        iter -= n;
        iterdivn++;
        if (iterdivn >= maxitdivn) {
          let info = 0;
          for (let i = 0; i < n - 1; i++) if (E[i] !== 0.0) info++;
          return info;
        }
      }
      smax = Math.abs(D[M - 1]);
      let ll = 0;
      let split = false;
      for (let lll = 1; lll < M; lll++) {
        ll = M - lll;
        const abss = Math.abs(D[ll - 1]);
        const abse = Math.abs(E[ll - 1]);
        if (abse <= thresh) {
          split = true;
          break;
        }
        smax = Math.max(smax, abss, abse);
      }
      if (split) {
        E[ll - 1] = 0.0;
        if (ll === M - 1) {
          M -= 1;
          continue;
        }
      } else {
        ll = 0;
      }
      ll += 1;
      if (ll === M - 1) {
        const [sigmn, sigmx, sinr, cosr, sinl, cosl] = dlasv2(D[M - 2], E[M - 2], D[M - 1]);
        D[M - 2] = sigmx;
        E[M - 2] = 0.0;
        D[M - 1] = sigmn;
        if (ncvt > 0) drot(ncvt, VT, M - 2, ldvt, VT, M - 1, ldvt, cosr, sinr);
        if (ncc > 0) drot(ncc, C, M - 2, ldc, C, M - 1, ldc, cosl, sinl);
        M -= 2;
        continue;
      }
      if (ll > oldm || M < oldll) idir = Math.abs(D[ll - 1]) >= Math.abs(D[M - 1]) ? 1 : 2;
      let restart = false;
      if (idir === 1) {
        if (Math.abs(E[M - 2]) <= Math.abs(tol) * Math.abs(D[M - 1])) {
          E[M - 2] = 0.0;
          continue;
        }
        let mu = Math.abs(D[ll - 1]);
        smin = mu;
        for (let lll = ll; lll < M; lll++) {
          if (Math.abs(E[lll - 1]) <= tol * mu) {
            E[lll - 1] = 0.0;
            restart = true;
            break;
          }
          mu = Math.abs(D[lll]) * (mu / (mu + Math.abs(E[lll - 1])));
          smin = Math.min(smin, mu);
        }
      } else {
        if (Math.abs(E[ll - 1]) <= Math.abs(tol) * Math.abs(D[ll - 1])) {
          E[ll - 1] = 0.0;
          continue;
        }
        let mu = Math.abs(D[M - 1]);
        smin = mu;
        for (let lll = M - 1; lll >= ll; lll--) {
          if (Math.abs(E[lll - 1]) <= tol * mu) {
            E[lll - 1] = 0.0;
            restart = true;
            break;
          }
          mu = Math.abs(D[lll - 1]) * (mu / (mu + Math.abs(E[lll - 1])));
          smin = Math.min(smin, mu);
        }
      }
      if (restart) continue;
      oldll = ll;
      oldm = M;
      let shift: number;
      if (n * tol * (smin / smax) <= Math.max(LA_EPS, 0.01 * tol)) {
        shift = 0.0;
      } else {
        let sll: number;
        if (idir === 1) {
          sll = Math.abs(D[ll - 1]);
          [shift] = dlas2(D[M - 2], E[M - 2], D[M - 1]);
        } else {
          sll = Math.abs(D[M - 1]);
          [shift] = dlas2(D[ll - 1], E[ll - 1], D[ll]);
        }
        if (sll > 0.0) {
          const q = shift / sll;
          if (q * q < LA_EPS) shift = 0.0;
        }
      }
      iter = iter + M - ll;
      if (shift === 0.0) {
        let cs = 1.0;
        let oldcs = 1.0;
        let oldsn = 0.0;
        if (idir === 1) {
          for (let i = ll; i < M; i++) {
            const rot = dlartg(D[i - 1] * cs, E[i - 1]);
            cs = rot[0];
            const sn = rot[1];
            const r = rot[2];
            if (i > ll) E[i - 2] = oldsn * r;
            [oldcs, oldsn, D[i - 1]] = dlartg(oldcs * r, D[i] * sn);
            work[i - ll] = cs;
            work[i - ll + nm1] = sn;
            work[i - ll + nm12] = oldcs;
            work[i - ll + nm13] = oldsn;
          }
          const h = D[M - 1] * cs;
          D[M - 1] = h * oldcs;
          E[M - 2] = h * oldsn;
          if (ncvt > 0) dlasrLV(true, M - ll + 1, ncvt, work, 0, nm1, VT, ll - 1, ldvt);
          if (ncc > 0) dlasrLV(true, M - ll + 1, ncc, work, nm12, nm13, C, ll - 1, ldc);
          if (Math.abs(E[M - 2]) <= thresh) E[M - 2] = 0.0;
        } else {
          for (let i = M; i > ll; i--) {
            const rot = dlartg(D[i - 1] * cs, E[i - 2]);
            cs = rot[0];
            const sn = rot[1];
            const r = rot[2];
            if (i < M) E[i - 1] = oldsn * r;
            [oldcs, oldsn, D[i - 1]] = dlartg(oldcs * r, D[i - 2] * sn);
            work[i - ll - 1] = cs;
            work[i - ll - 1 + nm1] = -sn;
            work[i - ll - 1 + nm12] = oldcs;
            work[i - ll - 1 + nm13] = -oldsn;
          }
          const h = D[ll - 1] * cs;
          D[ll - 1] = h * oldcs;
          E[ll - 1] = h * oldsn;
          if (ncvt > 0) dlasrLV(false, M - ll + 1, ncvt, work, nm12, nm13, VT, ll - 1, ldvt);
          if (ncc > 0) dlasrLV(false, M - ll + 1, ncc, work, 0, nm1, C, ll - 1, ldc);
          if (Math.abs(E[ll - 1]) <= thresh) E[ll - 1] = 0.0;
        }
      } else if (idir === 1) {
        let f = (Math.abs(D[ll - 1]) - shift) * (fsign(1.0, D[ll - 1]) + shift / D[ll - 1]);
        let g = E[ll - 1];
        for (let i = ll; i < M; i++) {
          const [cosr, sinr, r1] = dlartg(f, g);
          if (i > ll) E[i - 2] = r1;
          f = fmaFast(cosr, D[i - 1], sinr * E[i - 1]);
          E[i - 1] = fmaFast(cosr, E[i - 1], -(sinr * D[i - 1]));
          g = sinr * D[i];
          D[i] = cosr * D[i];
          const [cosl, sinl, r2] = dlartg(f, g);
          D[i - 1] = r2;
          f = fmaFast(cosl, E[i - 1], sinl * D[i]);
          D[i] = fmaFast(cosl, D[i], -(sinl * E[i - 1]));
          if (i < M - 1) {
            g = sinl * E[i];
            E[i] = cosl * E[i];
          }
          work[i - ll] = cosr;
          work[i - ll + nm1] = sinr;
          work[i - ll + nm12] = cosl;
          work[i - ll + nm13] = sinl;
        }
        E[M - 2] = f;
        if (ncvt > 0) dlasrLV(true, M - ll + 1, ncvt, work, 0, nm1, VT, ll - 1, ldvt);
        if (ncc > 0) dlasrLV(true, M - ll + 1, ncc, work, nm12, nm13, C, ll - 1, ldc);
        if (Math.abs(E[M - 2]) <= thresh) E[M - 2] = 0.0;
      } else {
        let f = (Math.abs(D[M - 1]) - shift) * (fsign(1.0, D[M - 1]) + shift / D[M - 1]);
        let g = E[M - 2];
        for (let i = M; i > ll; i--) {
          const [cosr, sinr, r1] = dlartg(f, g);
          if (i < M) E[i - 1] = r1;
          f = fmaFast(cosr, D[i - 1], sinr * E[i - 2]);
          E[i - 2] = fmaFast(cosr, E[i - 2], -(sinr * D[i - 1]));
          g = sinr * D[i - 2];
          D[i - 2] = cosr * D[i - 2];
          const [cosl, sinl, r2] = dlartg(f, g);
          D[i - 1] = r2;
          f = fmaFast(cosl, E[i - 2], sinl * D[i - 2]);
          D[i - 2] = fmaFast(cosl, D[i - 2], -(sinl * E[i - 2]));
          if (i > ll + 1) {
            g = sinl * E[i - 3];
            E[i - 3] = cosl * E[i - 3];
          }
          work[i - ll - 1] = cosr;
          work[i - ll - 1 + nm1] = -sinr;
          work[i - ll - 1 + nm12] = cosl;
          work[i - ll - 1 + nm13] = -sinl;
        }
        E[ll - 1] = f;
        if (Math.abs(E[ll - 1]) <= thresh) E[ll - 1] = 0.0;
        if (ncvt > 0) dlasrLV(false, M - ll + 1, ncvt, work, nm12, nm13, VT, ll - 1, ldvt);
        if (ncc > 0) dlasrLV(false, M - ll + 1, ncc, work, 0, nm1, C, ll - 1, ldc);
      }
    }
  }
  // Make the singular values positive, then sort them into decreasing order.
  for (let i = 0; i < n; i++) {
    if (D[i] === 0.0) D[i] = 0.0;
    if (D[i] < 0.0) {
      D[i] = -D[i];
      if (ncvt > 0) dscal(ncvt, -1.0, VT, i, ldvt);
    }
  }
  for (let i = 1; i < n; i++) {
    let isub = 1;
    let smin = D[0];
    for (let j = 2; j <= n + 1 - i; j++) {
      if (D[j - 1] <= smin) {
        isub = j;
        smin = D[j - 1];
      }
    }
    const tgt = n + 1 - i;
    if (isub !== tgt) {
      D[isub - 1] = D[tgt - 1];
      D[tgt - 1] = smin;
      if (ncvt > 0) dswap(ncvt, VT, isub - 1, ldvt, VT, tgt - 1, ldvt);
      if (ncc > 0) dswap(ncc, C, isub - 1, ldc, C, tgt - 1, ldc);
    }
  }
  return 0;
}

/** The multipliers dlascl('G', cfrom → cto) applies, in order. */
function dlasclMuls(cfrom: number, cto: number): number[] {
  const smlnum = SAFMIN;
  const bignum = 1.0 / smlnum;
  let cfromc = cfrom;
  let ctoc = cto;
  const muls: number[] = [];
  for (;;) {
    const cfrom1 = cfromc * smlnum;
    let mul: number;
    let done: boolean;
    if (cfrom1 === cfromc) {
      mul = ctoc / cfromc;
      done = true;
    } else {
      const cto1 = ctoc / bignum;
      if (cto1 === ctoc) {
        mul = ctoc;
        done = true;
        cfromc = 1.0;
      } else if (Math.abs(cfrom1) > Math.abs(ctoc) && ctoc !== 0.0) {
        mul = smlnum;
        done = false;
        cfromc = cfrom1;
      } else if (Math.abs(cto1) > Math.abs(cfromc)) {
        mul = bignum;
        done = false;
        ctoc = cto1;
      } else {
        mul = ctoc / cfromc;
        done = true;
        if (mul === 1.0) return muls;
      }
    }
    muls.push(mul);
    if (done) return muls;
  }
}

function dlascl(cfrom: number, cto: number, x: number[], ox: number, n: number) {
  for (const mul of dlasclMuls(cfrom, cto)) for (let k = 0; k < n; k++) x[ox + k] *= mul;
}

/** max|A| as dlange('M') / dlanst('M') find it (a NaN wins). */
function maxAbsNan(v: readonly number[], n = v.length): number {
  let r = 0.0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(v[i]);
    if (r < a || Number.isNaN(a)) r = a;
  }
  return r;
}

/** dlalsd for NRHS = 1 and n ≤ SMLSIZ = 25: solve the bidiagonal least-squares problem in b. */
function dlalsd(lower: boolean, n: number, D: number[], E: number[], b: number[], rcond: number) {
  const rcnd = rcond <= 0.0 || rcond >= 1.0 ? LA_EPS : rcond;
  if (n === 0) return;
  if (n === 1) {
    if (D[0] === 0.0) b[0] = 0.0;
    else {
      dlascl(D[0], 1.0, b, 0, 1);
      D[0] = Math.abs(D[0]);
    }
    return;
  }
  if (lower) {
    for (let i = 0; i < n - 1; i++) {
      const [cs, sn, r] = dlartg(D[i], E[i]);
      D[i] = r;
      E[i] = sn * D[i + 1];
      D[i + 1] = cs * D[i + 1];
      drot(1, b, i, 1, b, i + 1, 1, cs, sn);
    }
  }
  // dlanst('M'): |D(n)| first, then |D(i)|, |E(i)|.
  let orgnrm = Math.abs(D[n - 1]);
  for (let i = 0; i < n - 1; i++) {
    for (const v of [Math.abs(D[i]), Math.abs(E[i])]) if (orgnrm < v || Number.isNaN(v)) orgnrm = v;
  }
  if (orgnrm === 0.0) {
    for (let i = 0; i < n; i++) b[i] = 0.0;
    return;
  }
  dlascl(orgnrm, 1.0, D, 0, n);
  dlascl(orgnrm, 1.0, E, 0, n - 1);
  if (n > 25) throw new Error('dlalsd: n > SMLSIZ is not ported');
  // dlasdq('U', 0, n, n, 0, 1, D, E, VT = I, …, C = b): dbdsqr, then an ascending sort.
  const W: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) W.push(i === j ? 1.0 : 0.0);
  const ldb = Math.max(n, 1);
  dbdsqr(n, D, E, n, W, n, 1, b, ldb);
  for (let i = 0; i < n; i++) {
    let isub = i;
    let smin = D[i];
    for (let j = i + 1; j < n; j++)
      if (D[j] < smin) {
        isub = j;
        smin = D[j];
      }
    if (isub !== i) {
      D[isub] = D[i];
      D[i] = smin;
      dswap(n, W, isub, n, W, i, n);
      dswap(1, b, isub, ldb, b, i, ldb);
    }
  }
  let imax = 0;
  for (let i = 1; i < n; i++) if (Math.abs(D[i]) > Math.abs(D[imax])) imax = i;
  const tol = rcnd * Math.abs(D[imax]);
  for (let i = 0; i < n; i++) {
    if (D[i] <= tol) b[i] = 0.0;
    else dlascl(D[i], 1.0, b, i, 1);
  }
  // DGEMM('T', 'N', n, 1, n, 1, VT, n, b, …): OpenBLAS forwards n = 1 to dgemv_t.
  const out = new Array<number>(n).fill(0.0);
  dgemvT(n, n, W, 0, n, b, 0, 1, out, 0, 1);
  for (let i = 0; i < n; i++) b[i] = out[i];
  dlascl(1.0, orgnrm, D, 0, n);
  D.splice(0, n, ...D.slice(0, n).sort((p, q) => q - p));
  dlascl(orgnrm, 1.0, b, 0, n);
}

/**
 * `np.linalg.lstsq(A, b, rcond=None)` (LAPACK dgelsd, rcond = ε·max(rows, cols)): the
 * minimum-norm least-squares solution and the singular values of A (descending), bit for bit.
 */
export function lstsq(Arows: Matrix, bIn: readonly number[]): { x: Vector; s: Vector } {
  const m = Arows.length;
  const n = m ? Arows[0].length : 0;
  const rcond = EPS * Math.max(m, n);
  const minmn = Math.min(m, n);
  if (m === 0 || n === 0) return { x: new Array<number>(n).fill(0.0), s: [] };
  const lda = m;
  const A: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) A.push(Arows[i][j]);
  const B: number[] = [...bIn, ...new Array<number>(Math.max(m, n) - m).fill(0.0)];
  // ILAENV(6, 'DGELSD') = INT(REAL(min(m, n))·1.6E0) in single precision.
  const mnthr = Math.trunc(Math.fround(Math.fround(minmn) * Math.fround(1.6)));
  const smlnum = SAFMIN / LA_PREC;
  const bignum = 1.0 / smlnum;
  const anrm = maxAbsNan(A);
  let iascl = 0;
  if (anrm > 0.0 && anrm < smlnum) {
    dlascl(anrm, smlnum, A, 0, A.length);
    iascl = 1;
  } else if (anrm > bignum) {
    dlascl(anrm, bignum, A, 0, A.length);
    iascl = 2;
  } else if (anrm === 0.0) {
    return { x: new Array<number>(n).fill(0.0), s: new Array<number>(minmn).fill(0.0) };
  }
  const bnrm = maxAbsNan(B, m);
  let ibscl = 0;
  if (bnrm > 0.0 && bnrm < smlnum) {
    dlascl(bnrm, smlnum, B, 0, m);
    ibscl = 1;
  } else if (bnrm > bignum) {
    dlascl(bnrm, bignum, B, 0, m);
    ibscl = 2;
  }
  const S = new Array<number>(minmn).fill(0.0);
  const E = new Array<number>(minmn).fill(0.0);
  const tauq = new Array<number>(minmn).fill(0.0);
  const taup = new Array<number>(minmn).fill(0.0);
  if (m >= n) {
    let mm = m;
    if (m >= mnthr) {
      // Path 1a: QR first, then bidiagonalize R.
      mm = n;
      const tau = new Array<number>(n).fill(0.0);
      dgeqr2(m, n, A, lda, tau);
      applyQt(m, n, A, 0, lda, tau, B, 0);
      for (let j = 0; j < n - 1; j++) for (let i = j + 1; i < n; i++) A[j * lda + i] = 0.0;
    }
    dgebd2(mm, n, A, lda, S, E, tauq, taup);
    applyQt(mm, n, A, 0, lda, tauq, B, 0);
    dlalsd(false, n, S, E, B, rcond);
    if (n > 1) applyLqT(n - 1, n - 1, A, lda, lda, taup, B, 1);
  } else if (n >= mnthr) {
    // Path 2a: LQ first, then bidiagonalize L.
    const tau = new Array<number>(m).fill(0.0);
    dgelq2(m, n, A, lda, tau);
    const L = new Array<number>(m * m).fill(0.0);
    for (let j = 0; j < m; j++) for (let i = j; i < m; i++) L[j * m + i] = A[j * lda + i];
    dgebd2(m, m, L, m, S, E, tauq, taup);
    applyQt(m, m, L, 0, m, tauq, B, 0);
    dlalsd(false, m, S, E, B, rcond);
    if (m > 1) applyLqT(m - 1, m - 1, L, m, m, taup, B, 1);
    for (let i = m; i < n; i++) B[i] = 0.0;
    applyLqT(n, m, A, 0, lda, tau, B, 0);
  } else {
    // Path 2: bidiagonalize A directly (lower bidiagonal).
    dgebd2(m, n, A, lda, S, E, tauq, taup);
    if (m > 1) applyQt(m - 1, m - 1, A, 1, lda, tauq, B, 1);
    dlalsd(true, m, S, E, B, rcond);
    applyLqT(n, m, A, 0, lda, taup, B, 0);
  }
  if (iascl === 1) {
    dlascl(anrm, smlnum, B, 0, n);
    dlascl(smlnum, anrm, S, 0, minmn);
  } else if (iascl === 2) {
    dlascl(anrm, bignum, B, 0, n);
    dlascl(bignum, anrm, S, 0, minmn);
  }
  if (ibscl === 1) dlascl(smlnum, bnrm, B, 0, n);
  else if (ibscl === 2) dlascl(bignum, bnrm, B, 0, n);
  return { x: B.slice(0, n), s: S };
}

// ---------------------------------------------------------------------------------------
// The AA coefficient problem
// ---------------------------------------------------------------------------------------

/**
 * (c*, γ*, cond, λ') for the residual window F (n × (m_k + 1), by rows), m_k ≥ 1:
 * c* = argmin_{1ᵀc=1} ‖Fc‖₂² + λ'‖c‖₂², λ' = λ‖F‖₂², through c* = e + Dγ*.
 */
export function aaCoefficients(
  F: Matrix,
  lam: number,
): { c: Vector; gamma: Vector; cond: number; lamEff: number } {
  const mk = F[0].length - 1;
  const dF: Matrix = F.map((row) => row.slice(1).map((v, j) => v - row[j])); // (n, mk)
  // D: (mk + 1) × mk with D_jj = 1, D_{j+1,j} = −1; e = (0, …, 0, 1).
  const D: Matrix = Array.from({ length: mk + 1 }, (_, i) =>
    Array.from({ length: mk }, (_, j) => (i === j ? 1.0 : i === j + 1 ? -1.0 : 0.0)),
  );
  const e = Array.from({ length: mk + 1 }, (_, i) => (i === mk ? 1.0 : 0.0));
  const lamEff = lam > 0.0 ? lam * singularValues(F)[0] ** 2 : 0.0;
  let A: Matrix;
  let rhs: Vector;
  const last = F.map((row) => row[mk]);
  if (lamEff > 0.0) {
    const r = Math.sqrt(lamEff);
    A = [...dF, ...D.map((row) => row.map((v) => -r * v))];
    rhs = [...last, ...e.map((v) => r * v)];
  } else {
    A = dF;
    rhs = last;
  }
  const { x: gamma, s } = lstsq(A, rhs);
  const sLast = s[s.length - 1];
  const cond = s.length && sLast > 0.0 ? s[0] / sLast : Infinity;
  const Dg = matvec(D, gamma);
  return { c: e.map((v, i) => v + Dg[i]), gamma, cond, lamEff };
}

// ---------------------------------------------------------------------------------------
// Second-order test at a point that passes the gradient test
// ---------------------------------------------------------------------------------------

const maxAbs = (v: readonly number[]) => v.reduce((m, t) => Math.max(m, Math.abs(t)), 0);

/** tol_H: the accuracy of the eigenvalues of ∇²f itself. */
function eigTol(lam: Vector, fx: number, hessFd: boolean, gradFd: boolean): number {
  const n = lam.length;
  const lamMax = maxAbs(lam);
  if (!hessFd) return n * EPS * lamMax;
  const scale = Math.max(1.0, lamMax, gradFd ? Math.abs(fx) : 0.0);
  return n * FD_HESS_REL * scale;
}

/** tol_g = √(‖∇f(x)‖₂·|λ|_max): the eigenvalue shift allowed by the distance from x*. */
const stationarityTol = (lam: Vector, gnorm: number) => Math.sqrt(gnorm * maxAbs(lam));

/** [converged, why] from the ascending eigenvalues of ∇²f (N&W Thms 2.3–2.4). */
function classify(lam: Vector, tolH: number, tolG: number, hessFd: boolean): [boolean, string] {
  const tol = Math.max(tolH, tolG);
  const lamMin = lam[0];
  const lamTop = lam[lam.length - 1];
  const source = hessFd ? 'the finite-difference ∇²f' : '∇²f';
  if (lamMin < -tol) {
    let kind: string;
    if (lamTop < -tol) kind = 'a maximizer';
    else if (lamTop > tol) kind = 'a saddle point';
    else
      kind =
        `a saddle point or a maximizer (λ_max = ${g3(lamTop + 0.0)}: ∇²f is singular, ` +
        'so the second-order test cannot decide)';
    return [
      false,
      `stopped at ${kind}, not a minimizer: ${source} has the eigenvalue ` +
        `λ_min = ${g3(lamMin)} < −${g3(tol)}`,
    ];
  }
  if (lamMin <= tol) {
    if (Math.abs(lamMin) > tolH)
      return [
        true,
        `${source} has λ_min = ${g3(lamMin)}, within √(‖∇f‖·|λ|_max) = ${g3(tolG)} of 0: ` +
          '∇²f is positive semidefinite to the accuracy of the gradient test, and the ' +
          'second-order sufficient condition is not verified',
      ];
    if (hessFd)
      return [
        true,
        `the finite-difference ∇²f has λ_min = ${g3(lamMin)}, within its accuracy ` +
          `±${g3(tolH)} of 0: ∇²f is positive semidefinite to that accuracy, and the ` +
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

// ---------------------------------------------------------------------------------------
// The method
// ---------------------------------------------------------------------------------------

const incomingNone = (): Record<string, unknown> => ({
  memory: null,
  coefficients: [],
  history: [],
  x_bar: null,
  lsq_residual: null,
  cond: null,
  lam_eff: null,
  direction: null,
});

function validate(m: number, lr: number, beta: number, lam: number, gtol: number, maxIter: number) {
  if (!Number.isInteger(m) || m < 0)
    throw new ValueError(`m must be an integer ≥ 0, got ${String(m)}`);
  if (!(Number.isFinite(lr) && lr > 0.0))
    throw new ValueError(`lr must be finite and > 0, got ${formatRepr(lr)}`);
  if (!(Number.isFinite(beta) && beta > 0.0))
    throw new ValueError(`beta must be finite and > 0, got ${formatRepr(beta)}`);
  if (!(Number.isFinite(lam) && lam >= 0.0))
    throw new ValueError(`lam must be finite and ≥ 0, got ${formatRepr(lam)}`);
  if (!(Number.isFinite(gtol) && gtol >= 0.0))
    throw new ValueError(`gtol must be finite and ≥ 0, got ${formatRepr(gtol)}`);
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new ValueError(`max_iter must be a positive integer, got ${String(maxIter)}`);
}

type Opts = RunOptions & Params;
const num = (v: unknown, def: number): number => (typeof v === 'number' ? v : def);

export const andersonGd: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const m = num(o.m, 5);
  const lr = num(o.lr, 1e-3);
  const beta = num(o.beta, 1.0);
  const lam = num(o.lam, 0.0);
  const gtol = num(o.gtol, 1e-6);
  const maxIter = num(o.max_iter, 500);
  validate(m, lr, beta, lam, gtol, maxIter);

  const { x: xStart, f: fc, grad: gc, hess: hc } = resolveSmooth(problem, o.x0);
  const p = typeof problem === 'function' ? null : problem;
  const gradFd = !(p && typeof p.grad === 'function');
  const hessFd = !(p && typeof p.hess === 'function');
  let x = xStart;
  const n = x.length;

  const evaluate = (z: Vector): [number, Vector] => {
    try {
      const gz = gc(z);
      const fz = fc(z);
      return [fz, gz];
    } catch {
      return [NaN, new Array<number>(n).fill(NaN)];
    }
  };

  /** null to continue; else [converged, message, ascending eigenvalues of ∇²f or null]. */
  const stoppingTest = (
    z: Vector,
    fz: number,
    gz: Vector,
  ): [boolean, string, Vector | null] | null => {
    const gnorm = norm(gz);
    if (gnorm > gtol) return null;
    const head = `‖∇f‖₂ = ${g3(gnorm)} ≤ gtol`;
    let H: Matrix;
    try {
      H = hc(z);
    } catch {
      H = Array.from({ length: n }, () => new Array<number>(n).fill(NaN));
    }
    if (!allFinite(H)) return [false, `${head}, but ∇²f is not finite there`, null];
    const S = H.map((row, i) => row.map((v, j) => 0.5 * (v + H[j][i])));
    const eigs = eigh(S).values;
    const tolH = eigTol(eigs, fz, hessFd, gradFd);
    const [ok, why] = classify(eigs, tolH, stationarityTol(eigs, gnorm), hessFd);
    return [ok, `${head}; ${why}`, eigs];
  };

  const result = (
    z: Vector,
    fz: number,
    converged: boolean,
    message: string,
    trace: Step[],
  ): Result => ({
    method: 'anderson_gd',
    x: z.slice(),
    fun: fz,
    converged,
    message,
    nIter: trace[trace.length - 1].k,
    nFev: fc.n,
    nGev: gc.n,
    nHev: hc.n,
    trace,
    extra: { m, lr, beta, lam },
  });

  const x0Scale = Math.max(1.0, norm(x));
  let [fx, g] = evaluate(x);
  if (!allFinite(fx, g)) {
    const info = { grad: g.slice(), alpha: lr, ...incomingNone(), hess_eigs: null };
    const trace: Step[] = [
      { k: 0, x: x.slice(), fun: fx, gradNorm: norm(g), stepSize: null, info },
    ];
    return result(x, fx, false, 'f or ∇f is not finite at x0', trace);
  }
  let stop = stoppingTest(x, fx, g);
  const trace: Step[] = [
    {
      k: 0,
      x: x.slice(),
      fun: fx,
      gradNorm: norm(g),
      stepSize: null,
      info: { grad: g.slice(), alpha: lr, ...incomingNone(), hess_eigs: stop ? stop[2] : null },
    },
  ];
  if (stop !== null) return result(x, fx, stop[0], `at x0: ${stop[1]}`, trace);

  const X: Vector[] = []; // x_{k−m_k}, …, x_k
  const R: Vector[] = []; // f_{k−m_k}, …, f_k with f_i = −α∇f(x_i)
  for (let k = 0; k < maxIter; k++) {
    const fk = g.map((t) => -lr * t);
    X.push(x);
    R.push(fk);
    if (X.length > m + 1) {
      X.shift();
      R.shift();
    }
    let c: Vector;
    let cond: number;
    let lamEff: number;
    let xNew: Vector;
    let Xw: Matrix; // (n, m_k + 1) by rows
    let Fw: Matrix;
    if (X.length === 1) {
      c = [1.0];
      cond = 1.0;
      lamEff = 0.0;
      const xc = x;
      xNew = xc.map((t, i) => t + beta * fk[i]);
      Xw = xc.map((t) => [t]);
      Fw = fk.map((t) => [t]);
    } else {
      Xw = Array.from({ length: n }, (_, i) => X.map((xi) => xi[i]));
      Fw = Array.from({ length: n }, (_, i) => R.map((ri) => ri[i]));
      const dX: Matrix = Xw.map((row) => row.slice(1).map((v, j) => v - row[j]));
      const dF: Matrix = Fw.map((row) => row.slice(1).map((v, j) => v - row[j]));
      if (!allFinite(fk, dX, dF))
        // An overflow in α∇f or in a difference (|entries| near 10³⁰⁸): no SVD is tried.
        return result(x, fx, false, `non-finite residual difference at iteration ${k + 1}`, trace);
      let gamma: Vector;
      ({ c, gamma, cond, lamEff } = aaCoefficients(Fw, lam));
      const M: Matrix = dX.map((row, i) => row.map((v, j) => v + beta * dF[i][j]));
      const Mg = matvec(M, gamma);
      const xc = x;
      xNew = xc.map((t, i) => t + beta * fk[i] - Mg[i]);
    }
    const xBar = matvec(Xw, c);
    const incoming = {
      memory: X.length - 1,
      coefficients: c.slice(),
      history: X.map((xi) => xi.slice()),
      x_bar: xBar,
      lsq_residual: norm(matvec(Fw, c)),
      cond,
      lam_eff: lamEff,
      direction: xNew.map((t, i) => t - x[i]),
    };
    if (!allFinite(xNew))
      return result(x, fx, false, `non-finite iterate at iteration ${k + 1}`, trace);
    const [fxNew, gNew] = evaluate(xNew);
    const okValues = allFinite(fxNew, gNew);
    stop = okValues ? stoppingTest(xNew, fxNew, gNew) : null;
    trace.push({
      k: k + 1,
      x: xNew.slice(),
      fun: fxNew,
      gradNorm: norm(gNew),
      stepSize: norm(xNew.map((t, i) => t - x[i])),
      info: { grad: gNew.slice(), alpha: lr, ...incoming, hess_eigs: stop ? stop[2] : null },
    });
    if (!okValues)
      return result(xNew, fxNew, false, `f or ∇f is not finite at iteration ${k + 1}`, trace);
    if (stop !== null) return result(xNew, fxNew, stop[0], stop[1], trace);
    if (norm(xNew) > DIVERGENCE_FACTOR * x0Scale)
      return result(
        xNew,
        fxNew,
        false,
        `diverged: ‖x‖₂ > 1e+12·max(1, ‖x₀‖₂) at iteration ${k + 1}`,
        trace,
      );
    if (xNew.every((t, i) => t === x[i]))
      return result(xNew, fxNew, false, `stalled: x_${k + 1} = x_${k} in floating point`, trace);
    x = xNew;
    fx = fxNew;
    g = gNew;
  }
  return result(x, fx, false, `max_iter = ${maxIter} reached`, trace);
};

registerMethod(
  {
    id: 'anderson_gd',
    family: 'unconstrained',
    name: 'Anderson-accelerated gradient descent',
    params: [
      param.int('m', 5, {
        min: 0,
        max: 50,
        help: 'Memory: the number of residual differences kept (m = 0 is gradient descent).',
        label: 'Memory',
        tex: 'm',
      }),
      param.float('lr', 1e-3, {
        min: 1e-6,
        max: 1.0,
        log: true,
        help: 'Gradient step α of the map g(x) = x − α∇f(x).',
        label: 'Step size',
        tex: '\\alpha',
      }),
      param.float('beta', 1.0, {
        min: 0.05,
        max: 1.0,
        help: 'Mixing β: x⁺ = Σ cᵢ[(1 − β)xᵢ + β g(xᵢ)]; β = 1 is undamped AA.',
        label: 'Mixing',
        tex: '\\beta',
      }),
      param.float('lam', 0.0, {
        min: 0.0,
        max: 0.1,
        help: 'RNA Tikhonov weight λ, relative to ‖F‖₂² (0 = plain Anderson).',
        label: 'Regularization',
        tex: '\\lambda',
      }),
      param.float('gtol', 1e-6, {
        min: 1e-14,
        max: 1e-2,
        log: true,
        help: 'Stop when ‖∇f(x)‖₂ ≤ gtol (then ∇²f is checked for a minimizer).',
        label: 'Gradient tolerance',
        tex: '\\|\\nabla f\\|_2 \\le',
      }),
      param.int('max_iter', 500, {
        min: 1,
        max: 100_000,
        help: 'Iteration limit.',
        label: 'Max iterations',
      }),
    ],
    needs: ['f', 'grad'],
    order: 'linear (r-linear near a point where g is a contraction)',
    summary:
      'Combine the last m gradient steps with least-squares weights; fast, but it can stop at ' +
      'a saddle point.',
    references: [
      'Walker & Ni (2011), SIAM J. Numer. Anal. 49(4), Alg. AA and eq. (1.1); Thm. 2.2 (GMRES)',
      'Anderson (1965), J. ACM 12(4)',
      'Fang & Saad (2009), Numer. Linear Algebra Appl. 16, Type-II multisecant form',
      "Scieur, d'Aspremont & Bach (2016), arXiv:1606.04133, Alg. 2; λ scaling of Alg. 3, step 3",
      'Nocedal & Wright (2006), Thms 2.3–2.4 (second-order test at the stopping point)',
      'Nesterov & Polyak (2006), Math. Program. 108; Jin et al. (2017), ICML ' +
        '(ε-second-order stationarity, λ_min ≥ −√(ρε))',
    ],
  },
  andersonGd,
  {
    rule: 'c^\\star = \\arg\\min_{\\mathbf 1^{\\top} c = 1} \\|F_k c\\|^2,\\quad x_{k+1} = \\sum_i c_i^\\star\\,\\big(x_i - \\beta\\alpha\\nabla f(x_i)\\big)',
    intuition:
      'Keep the last few gradient steps and mix them with the weights that make the combined ' +
      'residual smallest: on a quadratic this is GMRES, without a single line search.',
    quantities: [
      { tex: 'm_k', key: 'info.memory' },
      { tex: '\\|F_k c^\\star\\|', key: 'info.lsq_residual' },
      { tex: '\\|\\nabla f(x_k)\\|', key: 'gradNorm' },
    ],
  },
);
