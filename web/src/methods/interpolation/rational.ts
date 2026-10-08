/**
 * Barycentric rational approximation: AAA and Floater–Hormann interpolation — TS port of
 * `numopt.interpolation.rational` (src/numopt/interpolation/rational.py).
 *
 * Both methods return a rational function in the barycentric form that `barycentricEval`
 * evaluates (Berrut & Trefethen 2004, eq. (4.2); Nakatsukasa, Sète & Trefethen 2018, eq. (2.1)),
 *
 *     r(t) = Σ_j [w_j/(t − z_j)] f_j / Σ_j [w_j/(t − z_j)],      r(z_j) = f_j when w_j ≠ 0.
 *
 *   - `aaa`: the support points z_j are chosen greedily from the samples Z = x; w is the right
 *     singular vector of the smallest singular value of a Loewner matrix (NST 2018, Fig. 4.1).
 *   - `floater_hormann`: the support points are all data nodes (sorted); w is the explicit
 *     Floater–Hormann blend weight (FH 2007, eq. (18)). No real poles (Theorem 1).
 *
 * `Result.x` = the final weights w; `Result.fun` = max_t |r(t) − f_true(t)| on the 200-point
 * grid (null without f_true); `nFev` = 0. `Result.extra` has the keys of the interpolation
 * family (kind, coefficients, nodes, values, domain, eval, max_error, node_residual); AAA adds
 * scaling, sample_error, errors, sigma_min, poles, residues, n_doublets, n_interval_poles,
 * interval_poles; FH adds d.
 *
 * Info keys (snake_case, as in the Python docstring):
 *   AAA: node_index, node, support, support_values, weights, degree, sample_error, sigma_min,
 *        poles [[re, im]], residues [[re, im]], n_doublets, n_interval_poles, interval_poles,
 *        curve (step 0: the constant mean f).
 *   FH:  node_index, node, weight, window [i_lo, i_hi]; the last step adds curve.
 *
 * Linear algebra without LAPACK: the minimal right singular vector comes from a one-sided
 * Jacobi SVD (Hestenes 1958; Demmel & Veselić 1992), which computes it to high accuracy; the
 * pole pencil is solved by the same shift-and-invert reduction as Python, with the complex
 * eigenvalues from a Householder–Hessenberg reduction and the shifted QR iteration (Golub & Van
 * Loan 2013, Alg. 7.4.2 and §7.5). The singular vector is unique up to sign whenever the
 * Loewner matrix has at least m − 1 rows (every parity fixture), so w agrees with NumPy to
 * rounding; where it is not unique, both take the same basis-free vector (`minimalVector`). The
 * products `C @ v` replay OpenBLAS's dgemv kernel (`gemv`).
 */
import { param, registerMethod, type MethodDoc } from '../../core/registry';
import type { MethodFn, Result, Step } from '../../core/types';
import { blasDot, gemv, npSum } from '../linalg/numerics';
import { Grid, NODE_RTOL, barycentricEval, pyG, type InterpProblem } from './methods';

/** NST 2018, Fig. 5.1: a pole with |residue| < DOUBLET_RTOL·max|f| is a Froissart doublet. */
export const DOUBLET_RTOL = 1e-13;
const EPS = 2 ** -52;
/** Smallest normal float64, 2⁻¹⁰²². */
const TINY = 2.2250738585072014e-308;
/** Newton steps that polish each eigenvalue estimate of a pole. */
export const POLE_NEWTON_STEPS = 3;
/**
 * Fractions s of a gap [lo, hi] at which `realDenominatorRoots` samples d: graded towards both
 * ends (2⁻¹ … 2⁻⁶⁰) plus 31 equispaced interior points (sorted, without repeats).
 */
const GAP_FRACTIONS: number[] = (() => {
  const s = new Set<number>();
  for (let k = 1; k <= 60; k++) s.add(2 ** -k);
  for (let k = 1; k <= 31; k++) s.add(k / 32);
  return [...s].sort((p, q) => p - q);
})();
const HINT_RTOL = 1e-4;
const HINT_ATOL = 1e3 * EPS;
const BISECT_MAX = 2000;

type ProblemArg = InterpProblem | readonly [readonly number[], readonly number[]];
type F = (x: number) => number;

// ---------------------------------------------------------------------------------------
// Data handling (the same rules as methods.ts: Python `_resolve`, `_sorted`, `_finish`)
// ---------------------------------------------------------------------------------------

interface Data {
  x: number[];
  y: number[];
  fTrue: F | null;
  a: number;
  b: number;
  n: number;
}

const allFinite = (v: readonly number[]) => v.every(Number.isFinite);
const maxAbs = (v: readonly number[]) => v.reduce((m, t) => Math.max(m, Math.abs(t)), 0);

function resolve(problem: ProblemArg): Data {
  let xRaw: readonly number[], yRaw: readonly number[];
  let fTrue: F | null = null;
  let domain: readonly number[] | null | undefined = null;
  if (Array.isArray(problem)) {
    if (problem.length !== 2) throw new TypeError('problem must be a Dataset or an (x, y) pair');
    [xRaw, yRaw] = problem as [readonly number[], readonly number[]];
  } else {
    const d = problem as InterpProblem;
    xRaw = d.x;
    yRaw = d.y;
    fTrue = d.fTrue ?? null;
    domain = d.domain;
  }
  const x = Array.from(xRaw, Number);
  const y = Array.from(yRaw, Number);
  if (x.length !== y.length)
    throw new Error(`x and y must be 1-D of equal length; got (${x.length},) and (${y.length},)`);
  if (x.length === 0) throw new Error('need at least one data point');
  if (!(allFinite(x) && allFinite(y))) throw new Error('data contain NaN or infinite values');
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
  return { x, y, fTrue, a, b, n: x.length };
}

function requireDistinct(x: readonly number[]): void {
  const s = [...x].sort((p, q) => p - q);
  for (let i = 1; i < s.length; i++)
    if (s[i] - s[i - 1] === 0.0) throw new Error('interpolation nodes must be distinct');
}

function sortedData(data: Data, minPoints: number): [number[], number[]] {
  if (data.n < minPoints) throw new Error(`need at least ${minPoints} data points, got ${data.n}`);
  const order = data.x.map((_, i) => i).sort((i, j) => data.x[i] - data.x[j] || i - j);
  const x = order.map((i) => data.x[i]);
  const y = order.map((i) => data.y[i]);
  for (let i = 1; i < x.length; i++)
    if (x[i] - x[i - 1] === 0.0) throw new Error('interpolation nodes must be distinct');
  return [x, y];
}

function step(k: number, x: number[], fun: number | null, info: Step['info']): Step {
  return { k, x, fun, gradNorm: null, stepSize: null, info };
}

function broken(method: string, trace: Step[], message: string): Result {
  const last = trace[trace.length - 1];
  return {
    method,
    x: last.x,
    fun: null,
    converged: false,
    message,
    nIter: last.k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {},
  };
}

// ---------------------------------------------------------------------------------------
// Complex arithmetic (pairs [re, im])
// ---------------------------------------------------------------------------------------

export type Complex = [number, number];

const cadd = (a: Complex, b: Complex): Complex => [a[0] + b[0], a[1] + b[1]];
const csub = (a: Complex, b: Complex): Complex => [a[0] - b[0], a[1] - b[1]];
const cmul = (a: Complex, b: Complex): Complex => [
  a[0] * b[0] - a[1] * b[1],
  a[0] * b[1] + a[1] * b[0],
];
const cabs = (a: Complex) => Math.hypot(a[0], a[1]);
const cfinite = (a: Complex) => Number.isFinite(a[0]) && Number.isFinite(a[1]);

/** a / b by Smith's algorithm (as NumPy divides complex numbers). */
function cdiv(a: Complex, b: Complex): Complex {
  const [ar, ai] = a,
    [br, bi] = b;
  if (Math.abs(br) >= Math.abs(bi)) {
    if (br === 0 && bi === 0) return [ar / 0, ai / 0];
    const r = bi / br,
      den = br + bi * r;
    return [(ar + ai * r) / den, (ai - ar * r) / den];
  }
  const r = br / bi,
    den = bi + br * r;
  return [(ar * r + ai) / den, (ai * r - ar) / den];
}

/** Principal square root. */
function csqrt([re, im]: Complex): Complex {
  if (re === 0 && im === 0) return [0, 0];
  const r = Math.hypot(re, im);
  const t = Math.sqrt(0.5 * (r + Math.abs(re)));
  if (re >= 0) return [t, im / (2 * t)];
  return [Math.abs(im) / (2 * t), im >= 0 ? t : -t];
}

/**
 * Eigenvalues of a complex n×n matrix (`np.linalg.eigvals`): Householder reduction to upper
 * Hessenberg form, then the shifted QR iteration with Givens rotations, a Wilkinson shift and
 * deflation at |h_{k,k−1}| ≤ ε(|h_{k−1,k−1}| + |h_{kk}|) (Golub & Van Loan 2013, Alg. 7.4.2,
 * §7.5). Returns null if an eigenvalue needs more than 30 iterations (NumPy: LinAlgError).
 */
export function complexEigvals(
  Are: readonly (readonly number[])[],
  Aim: readonly (readonly number[])[],
): Complex[] | null {
  const n = Are.length;
  const hr = Are.map((r) => Float64Array.from(r));
  const hi = Aim.map((r) => Float64Array.from(r));
  // Householder: zero column k below the subdiagonal, H ← P H P with P = I − 2vvᴴ.
  for (let k = 0; k < n - 2; k++) {
    let norm2 = 0;
    for (let i = k + 1; i < n; i++) norm2 += hr[i][k] * hr[i][k] + hi[i][k] * hi[i][k];
    const alphaAbs = Math.sqrt(norm2);
    if (alphaAbs === 0) continue;
    const x0r = hr[k + 1][k],
      x0i = hi[k + 1][k];
    const x0a = Math.hypot(x0r, x0i);
    const ph: Complex = x0a === 0 ? [1, 0] : [x0r / x0a, x0i / x0a];
    // v = x + e^{iθ}‖x‖ e₁ (no cancellation), then normalized.
    const len = n - k - 1;
    const vr = new Float64Array(len),
      vi = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      vr[i] = hr[k + 1 + i][k];
      vi[i] = hi[k + 1 + i][k];
    }
    vr[0] += ph[0] * alphaAbs;
    vi[0] += ph[1] * alphaAbs;
    let vn = 0;
    for (let i = 0; i < len; i++) vn += vr[i] * vr[i] + vi[i] * vi[i];
    vn = Math.sqrt(vn);
    if (vn === 0) continue;
    for (let i = 0; i < len; i++) {
      vr[i] /= vn;
      vi[i] /= vn;
    }
    // Left: H[k+1:, j] −= 2 v (vᴴ H[k+1:, j]) for every column j ≥ k.
    for (let j = k; j < n; j++) {
      let sr = 0,
        si = 0;
      for (let i = 0; i < len; i++) {
        const a = hr[k + 1 + i][j],
          b = hi[k + 1 + i][j];
        sr += vr[i] * a + vi[i] * b;
        si += vr[i] * b - vi[i] * a;
      }
      for (let i = 0; i < len; i++) {
        hr[k + 1 + i][j] -= 2 * (vr[i] * sr - vi[i] * si);
        hi[k + 1 + i][j] -= 2 * (vr[i] * si + vi[i] * sr);
      }
    }
    // Right: H[i, k+1:] −= 2 (H[i, k+1:] v) vᴴ for every row i.
    for (let i = 0; i < n; i++) {
      let sr = 0,
        si = 0;
      for (let q = 0; q < len; q++) {
        const a = hr[i][k + 1 + q],
          b = hi[i][k + 1 + q];
        sr += a * vr[q] - b * vi[q];
        si += a * vi[q] + b * vr[q];
      }
      for (let q = 0; q < len; q++) {
        hr[i][k + 1 + q] -= 2 * (sr * vr[q] + si * vi[q]);
        hi[i][k + 1 + q] -= 2 * (si * vr[q] - sr * vi[q]);
      }
    }
    for (let i = k + 2; i < n; i++) hr[i][k] = hi[i][k] = 0;
  }
  const get = (i: number, j: number): Complex => [hr[i][j], hi[i][j]];
  const cabs1 = (i: number, j: number) => Math.abs(hr[i][j]) + Math.abs(hi[i][j]);
  let anorm = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) anorm = Math.max(anorm, cabs1(i, j));
  const out: Complex[] = new Array<Complex>(n);
  let hiIdx = n - 1;
  let its = 0;
  while (hiIdx >= 0) {
    // Find the active block [lo, hiIdx].
    let lo = hiIdx;
    while (lo > 0) {
      let s = cabs1(lo - 1, lo - 1) + cabs1(lo, lo);
      if (s === 0) s = anorm;
      if (cabs1(lo, lo - 1) <= EPS * s) {
        hr[lo][lo - 1] = hi[lo][lo - 1] = 0;
        break;
      }
      lo--;
    }
    if (lo === hiIdx) {
      out[hiIdx] = get(hiIdx, hiIdx);
      hiIdx--;
      its = 0;
      continue;
    }
    if (its >= 30 * n) return null;
    its++;
    // Wilkinson shift: the eigenvalue of the trailing 2×2 block nearer to h_{hi,hi}.
    const a = get(hiIdx - 1, hiIdx - 1),
      b = get(hiIdx - 1, hiIdx),
      c = get(hiIdx, hiIdx - 1),
      d = get(hiIdx, hiIdx);
    let mu: Complex;
    if (its % 11 === 10) {
      // Exceptional shift (breaks cycles; LAPACK zlahqr uses the same idea).
      mu = [d[0] + 0.75 * Math.abs(c[0]), d[1] + 0.75 * Math.abs(c[1])];
    } else {
      const half: Complex = [(a[0] - d[0]) / 2, (a[1] - d[1]) / 2];
      const disc = csqrt(cadd(cmul(half, half), cmul(b, c)));
      const mid: Complex = [(a[0] + d[0]) / 2, (a[1] + d[1]) / 2];
      const m1 = cadd(mid, disc),
        m2 = csub(mid, disc);
      mu = cabs(csub(m1, d)) <= cabs(csub(m2, d)) ? m1 : m2;
    }
    if (!cfinite(mu)) return null;
    for (let i = lo; i <= hiIdx; i++) {
      hr[i][i] -= mu[0];
      hi[i][i] -= mu[1];
    }
    // H − μI = QR by Givens rotations G_k on rows k, k+1; then RQ.
    const gc: number[] = [],
      gs: Complex[] = [];
    for (let k = lo; k < hiIdx; k++) {
      const x = get(k, k),
        y = get(k + 1, k);
      const ax = cabs(x),
        nu = Math.hypot(ax, cabs(y));
      let cr: number, s: Complex;
      if (nu === 0) {
        cr = 1;
        s = [0, 0];
      } else if (ax === 0) {
        cr = 0;
        s = [1, 0];
      } else {
        cr = ax / nu;
        const ph: Complex = [x[0] / ax, x[1] / ax];
        s = cmul(ph, [y[0] / nu, -y[1] / nu]);
      }
      gc.push(cr);
      gs.push(s);
      for (let j = k; j <= hiIdx; j++) {
        const u = get(k, j),
          v = get(k + 1, j);
        const nu1 = cadd([cr * u[0], cr * u[1]], cmul(s, v));
        const nv = cadd(cmul([-s[0], s[1]], u), [cr * v[0], cr * v[1]]);
        hr[k][j] = nu1[0];
        hi[k][j] = nu1[1];
        hr[k + 1][j] = nv[0];
        hi[k + 1][j] = nv[1];
      }
    }
    for (let q = 0; q < gc.length; q++) {
      const k = lo + q,
        cr = gc[q],
        s = gs[q];
      const sc: Complex = [s[0], -s[1]];
      for (let i = lo; i <= Math.min(k + 2, hiIdx); i++) {
        const u = get(i, k),
          v = get(i, k + 1);
        const nu1 = cadd([cr * u[0], cr * u[1]], cmul(v, sc));
        const nv = cadd(cmul([-s[0], -s[1]], u), [cr * v[0], cr * v[1]]);
        hr[i][k] = nu1[0];
        hi[i][k] = nu1[1];
        hr[i][k + 1] = nv[0];
        hi[i][k + 1] = nv[1];
      }
    }
    for (let i = lo; i <= hiIdx; i++) {
      hr[i][i] += mu[0];
      hi[i][i] += mu[1];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Poles and residues of a barycentric rational function
// ---------------------------------------------------------------------------------------

/** |d(λ)|/Σ|w_j/(λ − z_j)|, d(λ) and d'(λ) for each λ (Python `rel_residual`). */
function relResidual(lam: readonly Complex[], z: readonly number[], w: readonly number[]) {
  return lam.map((l) => {
    let dr = 0,
      di = 0,
      pr = 0,
      pi = 0,
      sabs = 0;
    for (let j = 0; j < z.length; j++) {
      const diff: Complex = [l[0] - z[j], l[1]];
      const c = cdiv([w[j], 0], diff);
      dr += c[0];
      di += c[1];
      const c2 = cdiv(c, diff);
      pr -= c2[0];
      pi -= c2[1];
      sabs += cabs(c);
    }
    const dval: Complex = [dr, di];
    return { res: cabs(dval) / sabs, dval, dprime: [pr, pi] as Complex };
  });
}

/** Safeguarded Newton steps λ ← λ − d(λ)/d'(λ) (Python `_polish_poles`). */
function polishPoles(poles: Complex[], z: readonly number[], w: readonly number[]): Complex[] {
  let p = poles.map((q) => [q[0], q[1]] as Complex);
  if (p.length === 0) return p;
  for (let it = 0; it < POLE_NEWTON_STEPS; it++) {
    const cur = relResidual(p, z, w);
    const stepV = cur.map((c) => cdiv(c.dval, c.dprime));
    const trial = p.map((q, i) => csub(q, stepV[i]));
    const resTrial = relResidual(trial, z, w).map((c) => c.res);
    const ok = p.map((q, i) => {
      let gap = Infinity;
      for (let j = 0; j < p.length; j++) if (j !== i) gap = Math.min(gap, cabs(csub(q, p[j])));
      return cfinite(trial[i]) && resTrial[i] < cur[i].res && cabs(stepV[i]) < 0.5 * gap;
    });
    if (!ok.some(Boolean)) break;
    p = p.map((q, i) => (ok[i] ? trial[i] : q));
  }
  return p;
}

/** Number q of leading moments Σ_j w_j s_j^k that vanish to rounding (Python `_vanishing_moments`). */
function vanishingMoments(s: readonly number[], w: readonly number[]): number {
  const m = s.length;
  let q = 0;
  let power = s.map(() => 1.0);
  for (let k = 0; k < m - 1; k++) {
    const terms = w.map((wj, j) => wj * power[j]);
    const absTerms = terms.map(Math.abs);
    if (Math.abs(npSum(terms)) > 4.0 * (m + k) * EPS * npSum(absTerms)) break;
    q += 1;
    power = power.map((pj, j) => pj * s[j]);
  }
  return q;
}

/**
 * Poles and residues of r = n/d, n = Σ w_j f_j/(t − z_j), d = Σ w_j/(t − z_j): the finite
 * eigenvalues of the arrowhead pencil (NST 2018, eq. (3.11)) by one exact shift-and-invert step,
 * polished by safeguarded Newton steps on d (see the Python docstring of `barycentric_poles`).
 * Support points with w_j = 0 are dropped. Residues n(λ)/d'(λ).
 */
export function barycentricPoles(
  zIn: readonly number[],
  wIn: readonly number[],
  fIn: readonly number[],
): { poles: Complex[]; residues: Complex[] } {
  const keepIdx = wIn.map((_, j) => j).filter((j) => wIn[j] !== 0.0);
  const z = keepIdx.map((j) => zIn[j]),
    w = keepIdx.map((j) => wIn[j]),
    f = keepIdx.map((j) => fIn[j]);
  const m = z.length;
  if (m <= 1) return { poles: [], residues: [] };
  const center = 0.5 * (Math.min(...z) + Math.max(...z));
  let radius = maxAbs(z.map((zj) => zj - center));
  if (radius === 0.0) radius = 1.0;
  const nFinite =
    m -
    1 -
    vanishingMoments(
      z.map((zj) => (zj - center) / radius),
      w,
    );
  if (nFinite === 0) return { poles: [], residues: [] };
  const nan = (): { poles: Complex[]; residues: Complex[] } => ({
    poles: Array.from({ length: nFinite }, () => [NaN, NaN] as Complex),
    residues: Array.from({ length: nFinite }, () => [NaN, NaN] as Complex),
  });
  let best: { norm: number; sigma: Complex; re: number[][]; im: number[][] } | null = null;
  for (const theta of [0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]) {
    for (const scale of [1.0, 2.0]) {
      const sigma: Complex = [
        center + scale * radius * Math.cos(Math.PI * theta),
        scale * radius * Math.sin(Math.PI * theta),
      ];
      const invD = z.map((zj) => cdiv([1, 0], [zj - sigma[0], -sigma[1]]));
      let s: Complex = [0, 0];
      for (let j = 0; j < m; j++) s = cadd(s, [w[j] * invD[j][0], w[j] * invD[j][1]]);
      if ((s[0] === 0 && s[1] === 0) || !cfinite(s)) continue;
      const re: number[][] = [],
        im: number[][] = [];
      let norm2 = 0;
      for (let i = 0; i < m; i++) {
        const rr = new Array<number>(m),
          ri = new Array<number>(m);
        for (let j = 0; j < m; j++) {
          const o = cdiv(cmul(invD[i], [w[j] * invD[j][0], w[j] * invD[j][1]]), s);
          let v: Complex = [-o[0], -o[1]];
          if (i === j) v = cadd(v, invD[i]);
          rr[j] = v[0];
          ri[j] = v[1];
          norm2 += v[0] * v[0] + v[1] * v[1];
        }
        re.push(rr);
        im.push(ri);
      }
      const norm = Math.sqrt(norm2);
      if (Number.isFinite(norm) && (best === null || norm < best.norm))
        best = { norm, sigma, re, im };
    }
  }
  if (best === null) return nan();
  const mu = complexEigvals(best.re, best.im);
  if (mu === null || !mu.every(cfinite)) return nan();
  // Drop the m − nFinite eigenvalues of smallest |μ| (one Jordan block at 0), then exact zeros.
  const order = mu.map((_, i) => i).sort((i, j) => cabs(mu[i]) - cabs(mu[j]));
  const kept = order
    .slice(m - nFinite)
    .map((i) => mu[i])
    .filter((q) => !(q[0] === 0 && q[1] === 0));
  const sigma = best.sigma;
  const poles = polishPoles(
    kept.map((q) => cadd(sigma, cdiv([1, 0], q))),
    z,
    w,
  );
  const residues = poles.map((lam) => {
    let num: Complex = [0, 0],
      dp: Complex = [0, 0];
    for (let j = 0; j < m; j++) {
      const inv = cdiv([1, 0], [lam[0] - z[j], lam[1]]);
      num = cadd(num, [inv[0] * w[j] * f[j], inv[1] * w[j] * f[j]]);
      const inv2 = cmul(inv, inv);
      dp = csub(dp, [inv2[0] * w[j], inv2[1] * w[j]]);
    }
    return cdiv(num, dp);
  });
  return { poles, residues };
}

/** Sign of d(t) = Σ_j w_j/(t − z_j) where it is certain, else 0 (Python `_signed_d`). */
function signedD(t: number, z: readonly number[], w: readonly number[]): number {
  const c = z.map((zj, j) => w[j] / (t - zj));
  const dval = npSum(c);
  const gamma = 4.0 * (z.length + 2) * EPS;
  if (Math.abs(dval) > gamma * npSum(c.map(Math.abs))) return Math.sign(dval);
  return 0.0;
}

/** Shrink a sign-change bracket while every probe stays certified (Python `_bisect_certified`). */
function bisectCertified(
  loIn: number,
  hiIn: number,
  sLo: number,
  z: readonly number[],
  w: readonly number[],
): [number, number] {
  let lo = loIn,
    hi = hiIn;
  let uLo: number | null = null,
    uHi: number | null = null;
  for (let it = 0; it < BISECT_MAX; it++) {
    let probes: number[], limits: [number, number][];
    if (uLo === null || uHi === null) {
      probes = [lo + 0.5 * (hi - lo)];
      limits = [[lo, hi]];
    } else {
      probes = [lo + 0.5 * (uLo - lo), uHi + 0.5 * (hi - uHi)];
      limits = [
        [lo, uLo],
        [uHi, hi],
      ];
    }
    let moved = false;
    for (let q = 0; q < probes.length; q++) {
      const p = probes[q];
      const [a, b] = limits[q];
      if (!(a < p && p < b && lo < p && p < hi)) continue;
      moved = true;
      const sp = signedD(p, z, w);
      if (sp === sLo) lo = p;
      else if (sp === -sLo) hi = p;
      else {
        uLo = uLo === null ? p : Math.min(uLo, p);
        uHi = uHi === null ? p : Math.max(uHi, p);
      }
      if (uLo !== null && uHi !== null && !(lo < uLo && uHi < hi)) uLo = uHi = null;
    }
    if (!moved) break;
  }
  return [lo, hi];
}

/**
 * Certified real zeros of d(t) = Σ_j w_j/(t − z_j) in [a, b], as brackets [lo, hi] sorted by lo
 * (Python `real_denominator_roots`): each bracket holds a sign change of d (Schneider & Werner
 * 1986), shrunk by bisection to the band where the sign of d is no longer certain.
 */
export function realDenominatorRoots(
  zIn: readonly number[],
  wIn: readonly number[],
  a: number,
  b: number,
  hints: readonly number[] = [],
): [number, number][] {
  const idx = wIn.map((_, j) => j).filter((j) => wIn[j] !== 0.0);
  idx.sort((i, j) => zIn[i] - zIn[j]);
  const z = idx.map((j) => zIn[j]),
    w = idx.map((j) => wIn[j]);
  const sw = w.map(Math.sign);
  const innerIdx = z.map((_, j) => j).filter((j) => z[j] > a && z[j] < b);
  const ends = [a, ...innerIdx.map((j) => z[j]), b];
  const rightSign = [0.0, ...innerIdx.map((j) => sw[j]), 0.0];
  const leftSign = [0.0, ...innerIdx.map((j) => -sw[j]), 0.0];
  for (const [i, e] of [
    [0, a],
    [ends.length - 1, b],
  ] as const) {
    const k = z.indexOf(e);
    if (k >= 0) {
      rightSign[i] = sw[k];
      leftSign[i] = -sw[k];
    } else rightSign[i] = leftSign[i] = signedD(e, z, w);
  }
  const brackets: [number, number, number][] = [];
  for (let s = 0; s < ends.length - 1; s++) {
    const lo = ends[s],
      hi = ends[s + 1];
    if (!(hi > lo)) continue;
    const hs = hints.filter((h) => h > lo && h < hi).sort((p, q) => p - q);
    let t = [
      ...GAP_FRACTIONS.map((g) => lo + g * (hi - lo)),
      ...GAP_FRACTIONS.map((g) => hi - g * (hi - lo)),
      ...hs,
    ];
    for (let i = 0; i + 1 < hs.length; i++) t.push(0.5 * (hs[i + 1] + hs[i]));
    t = [...new Set(t)].sort((p, q) => p - q).filter((v) => v > lo && v < hi);
    const xs = [lo],
      ss = [rightSign[s]];
    for (const v of t) {
      const sg = signedD(v, z, w);
      if (sg !== 0.0) {
        xs.push(v);
        ss.push(sg);
      }
    }
    xs.push(hi);
    ss.push(leftSign[s + 1]);
    const kx: number[] = [],
      ks: number[] = [];
    ss.forEach((v, i) => {
      if (v !== 0.0) {
        kx.push(xs[i]);
        ks.push(v);
      }
    });
    for (let i = 0; i + 1 < ks.length; i++)
      if (ks[i + 1] !== ks[i]) brackets.push([kx[i], kx[i + 1], ks[i]]);
  }
  return brackets.map(([lo, hi, sLo]) => bisectCertified(lo, hi, sLo, z, w));
}

/** Poles, residues, doublet count and certified real poles in [a, b] (Python `_pole_summary`). */
function poleSummary(
  z: readonly number[],
  w: readonly number[],
  f: readonly number[],
  a: number,
  b: number,
  fScale: number,
): Record<string, unknown> {
  const { poles, residues } = barycentricPoles(z, w, f);
  const scale = z.length ? maxAbs(z) : 1.0;
  const hints = poles
    .filter((p) => Math.abs(p[1]) <= HINT_RTOL * cabs(p) + HINT_ATOL * scale)
    .map((p) => p[0]);
  const brackets = realDenominatorRoots(z, w, a, b, hints);
  const limit = DOUBLET_RTOL * Math.max(fScale, TINY);
  return {
    poles: poles.map((p) => [p[0], p[1]]),
    residues: residues.map((r) => [r[0], r[1]]),
    n_doublets: residues.filter((r) => cabs(r) < limit).length,
    n_interval_poles: brackets.length,
    interval_poles: brackets.map(([lo, hi]) => 0.5 * (lo + hi)),
  };
}

// ---------------------------------------------------------------------------------------
// Minimal right singular vector (one-sided Jacobi)
// ---------------------------------------------------------------------------------------

/**
 * One-sided Jacobi on the columns of the r×p matrix `a` (no transposition, so a wide matrix keeps
 * its null vectors): after convergence the columns are A·v_j for the orthonormal right singular
 * vectors v_j, so their norms are the singular values. With r < p, p − r of the norms are 0 up to
 * rounding: their vectors span the null space of `a`. Two columns that are both negligible
 * (norm ≤ ‖a‖_F·max(r, p)·ε) are not rotated: their angle is rounding noise and would never meet
 * the stopping test, and any orthonormal basis of their span is as good (`minimalVector` uses
 * the null space of a wide `a` as a span only; a tall `a` has at most one such column unless
 * its minimal singular vector is not unique anyway). Null when a sweep limit is reached.
 */
function jacobiRight(
  a: readonly (readonly number[])[],
  p: number,
): { norms: number[]; v: Float64Array[] } | null {
  const r = a.length;
  let fro = 0;
  for (const row of a) for (const t of row) fro += t * t;
  const negligible = fro * (Math.max(r, p) * EPS) ** 2;
  const cols: Float64Array[] = Array.from({ length: p }, (_, j) =>
    Float64Array.from(a, (row) => row[j]),
  );
  const v: Float64Array[] = Array.from({ length: p }, (_, j) => {
    const e = new Float64Array(p);
    e[j] = 1;
    return e;
  });
  let done = false;
  for (let sweep = 0; sweep < 100 && !done; sweep++) {
    let rotated = false;
    for (let i = 0; i < p - 1; i++)
      for (let j = i + 1; j < p; j++) {
        const wi = cols[i],
          wj = cols[j];
        let alpha = 0,
          beta = 0,
          gamma = 0;
        for (let q = 0; q < r; q++) {
          alpha += wi[q] * wi[q];
          beta += wj[q] * wj[q];
          gamma += wi[q] * wj[q];
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        if (alpha <= negligible && beta <= negligible) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        if (t === 0 || !Number.isFinite(t)) continue;
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        rotated = true;
        for (let q = 0; q < r; q++) {
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
    done = !rotated;
  }
  if (!done) return null;
  const norms = cols.map((col) => {
    const big = col.reduce((m, t) => Math.max(m, Math.abs(t)), 0);
    if (big === 0 || !Number.isFinite(big)) return big;
    let ss = 0;
    for (const c of col) ss += (c / big) * (c / big);
    return big * Math.sqrt(ss);
  });
  return { norms, v };
}

/**
 * The right singular vector of the smallest singular value of the r×p matrix `a` and that value
 * (one-sided Jacobi, Hestenes 1958; Demmel & Veselić 1992). With r < p the smallest value is 0 up
 * to rounding and its vector is a null vector of `a`. Null when a sweep limit is reached.
 */
export function minRightSingular(
  a: readonly (readonly number[])[],
  p: number,
): { sigma: number; v: number[] } | null {
  const svd = jacobiRight(a, p);
  return svd === null ? null : smallest(svd);
}

/** The last column in descending order of norm (ties: the later column, as V's last row). */
function smallest({ norms, v }: { norms: number[]; v: Float64Array[] }): {
  sigma: number;
  v: number[];
} {
  let best = 0;
  for (let j = 1; j < norms.length; j++) if (norms[j] <= norms[best]) best = j;
  return { sigma: norms[best], v: Array.from(v[best]) };
}

/**
 * A unit minimizer of ‖Av‖ for the r×p matrix `a` (Python `_minimal_vector`) and the smallest
 * singular value. With r ≥ p − 1 this is `minRightSingular`. A wider `a` has a null space N of
 * dimension d = p − r ≥ 2 (the vectors of the d smallest column norms), and the choice is
 * P·1/‖P·1‖ (P the projector onto N): the minimum-norm v ∈ N with Σ v_j = 1. If
 * ‖P·1‖ ≤ 1e-8·√p (N nearly orthogonal to 1) it is the unit vector of N with the largest single
 * entry, P e_j/‖P e_j‖ (j the first index with P_jj within TIE_RTOL of the largest). Neither
 * depends on the basis of N that an SVD returns, so Jacobi and LAPACK give the same w.
 */
export function minimalVector(
  a: readonly (readonly number[])[],
  p: number,
): { sigma: number; v: number[] } | null {
  const svd = jacobiRight(a, p);
  if (svd === null) return null;
  const d = p - a.length;
  if (d <= 1) return smallest(svd);
  const order = svd.norms.map((_, j) => j).sort((i, j) => svd.norms[j] - svd.norms[i] || i - j);
  const basis = order.slice(p - d).map((j) => svd.v[j]);
  const norm = (u: number[]) => Math.sqrt(u.reduce((t, x) => t + x * x, 0));
  const sums = basis.map((b) => b.reduce((t, x) => t + x, 0)); // Bᵀ1
  let v = Array.from({ length: p }, (_, i) => basis.reduce((t, b, k) => t + b[i] * sums[k], 0)); // P·1
  if (!(norm(v) > 1e-8 * Math.sqrt(p))) {
    const diag = Array.from({ length: p }, (_, j) => basis.reduce((t, b) => t + b[j] * b[j], 0));
    const j = firstNearMaxAbs(diag);
    v = Array.from({ length: p }, (_, i) => basis.reduce((t, b) => t + b[i] * b[j], 0)); // P e_j
  }
  const nrm = norm(v);
  return { sigma: svd.norms[order[p - 1]], v: v.map((x) => x / nrm) };
}

// ---------------------------------------------------------------------------------------
// AAA
// ---------------------------------------------------------------------------------------

/** Index of the first maximum of |v| (NumPy argmax: the first NaN wins). */
function argmaxAbs(v: readonly number[]): number {
  let best = 0;
  let bv = -Infinity;
  for (let i = 0; i < v.length; i++) {
    const a = Math.abs(v[i]);
    if (Number.isNaN(a)) return i;
    if (a > bv) {
      bv = a;
      best = i;
    }
  }
  return best;
}

/**
 * AAA ranks two values (sample errors, |w_j|) as tied when they differ by less than TIE_RTOL
 * relative (Python `TIE_RTOL`). Symmetric data ties in exact arithmetic, and the computed values
 * then differ only by rounding, which changes with the CPU and BLAS; 1e-8 is far above that noise.
 */
const TIE_RTOL = 1e-8;

/**
 * The first index i with |v_i| ≥ (1 − TIE_RTOL)·max|v| (Python `_first_near_max`; the first NaN
 * wins). AAA uses it to pick the next support point and the entry of w that is made positive,
 * so neither choice depends on the last bit of a tie (sine_samples: |F − mean F| at π/2 and
 * 3π/2; w = ±0.5587… at step 4).
 */
function firstNearMaxAbs(v: readonly number[]): number {
  const i = argmaxAbs(v);
  const big = Math.abs(v[i]);
  if (Number.isNaN(big)) return i;
  for (let j = 0; j < v.length; j++) if (Math.abs(v[j]) >= (1 - TIE_RTOL) * big) return j;
  return i;
}

const aaa: MethodFn<ProblemArg> = (
  problem,
  { tol = 1e-13, max_terms = 100, scaling = 'columns' },
) => {
  const tolN = Number(tol);
  if (!(tolN >= 0.0)) throw new Error('tol must be ≥ 0');
  if (typeof max_terms === 'boolean' || !Number.isInteger(Number(max_terms)))
    throw new Error('max_terms must be an integer');
  const maxTerms = Number(max_terms);
  if (maxTerms < 1) throw new Error('max_terms must be ≥ 1');
  if (scaling !== 'none' && scaling !== 'columns')
    throw new Error("scaling must be 'none' or 'columns'");
  const data = resolve(problem);
  requireDistinct(data.x);
  const grid = new Grid(data);
  const zAll = data.x,
    fAll = data.y;
  const M = zAll.length;
  const fScale = maxAbs(fAll);
  const atol = tolN * fScale;

  const free = new Array<boolean>(M).fill(true);
  const supportIdx: number[] = [];
  const cauchyCols: number[][] = [];
  const meanF = npSum(fAll) / M;
  let rSamples = new Array<number>(M).fill(meanF);
  const maxDev = (r: readonly number[]) => {
    let e = 0;
    for (let i = 0; i < M; i++) {
      const d = Math.abs(fAll[i] - r[i]);
      if (d > e || Number.isNaN(d)) e = d;
    }
    return e;
  };
  let err = maxDev(rSamples);
  const errors = [err];
  const sigmas: number[] = [];
  const constCurve = grid.t.map(() => meanF);
  const trace: Step[] = [
    step(0, [], grid.error(constCurve), {
      support: [],
      support_values: [],
      weights: [],
      degree: 0,
      sample_error: err,
      curve: constCurve,
      poles: [],
      residues: [],
      n_doublets: 0,
      n_interval_poles: 0,
      interval_poles: [],
    }),
  ];
  let w: number[] = [];
  let converged = false;
  for (let m = 1; m <= maxTerms; m++) {
    const cand = free.flatMap((f, i) => (f ? [i] : []));
    const j = cand[firstNearMaxAbs(cand.map((i) => fAll[i] - rSamples[i]))];
    supportIdx.push(j);
    free[j] = false;
    const zj = zAll[j];
    cauchyCols.push(zAll.map((zi) => 1.0 / (zi - zj)));
    const zs = supportIdx.map((i) => zAll[i]);
    const fs = supportIdx.map((i) => fAll[i]);
    const rows = free.flatMap((f, i) => (f ? [i] : []));
    const cmat = rows.map((i) => cauchyCols.map((col) => col[i]));
    const loewner = cmat.map((row, q) => row.map((c, jj) => fAll[rows[q]] * c - c * fs[jj]));
    const colNorm = new Array<number>(m).fill(1.0);
    if (scaling === 'columns') {
      for (let jj = 0; jj < m; jj++) {
        let s2 = 0;
        for (const row of loewner) s2 += row[jj] * row[jj];
        const nrm = Math.sqrt(s2);
        colNorm[jj] = nrm > 0.0 ? nrm : 1.0;
      }
    }
    const scaled = loewner.map((row) => row.map((v, jj) => v / colNorm[jj]));
    // NOTE: with fewer rows than columns w is a null vector (σ_min reported as 0); when the
    // minimizers of ‖Aw‖ form a subspace of dimension ≥ 2, `minimalVector` takes the same
    // basis-free choice as Python (`_minimal_vector`).
    const finite = scaled.every((row) => row.every(Number.isFinite));
    const svd = !finite ? null : minimalVector(scaled, m);
    if (svd === null) return broken('aaa', trace, `SVD of the Loewner matrix failed at step ${m}`);
    const sigmaMin = rows.length >= m ? svd.sigma : 0.0;
    w = svd.v.map((v, jj) => v / colNorm[jj]);
    const wn = Math.sqrt(blasDot(w, w));
    w = w.map((v) => v / wn);
    const sgn = w[firstNearMaxAbs(w)] >= 0.0 ? 1.0 : -1.0;
    w = w.map((v) => v * sgn);
    const numer = gemv(
      cmat,
      w.map((v, jj) => v * fs[jj]),
    );
    const denom = gemv(cmat, w);
    rSamples = fAll.slice();
    rows.forEach((i, q) => (rSamples[i] = numer[q] / denom[q]));
    if (!(allFinite(w) && allFinite(rSamples)))
      return broken(
        'aaa',
        trace,
        `non-finite weights or values at step ${m} (d(z) = 0 at a sample point)`,
      );
    sigmas.push(sigmaMin);
    err = maxDev(rSamples);
    errors.push(err);
    const curve = barycentricEval(zs, w, fs, grid.t);
    const info: Record<string, unknown> = {
      node_index: j,
      node: [zAll[j], fAll[j]],
      support: zs.slice(),
      support_values: fs.slice(),
      weights: w.slice(),
      degree: m - 1,
      sample_error: err,
      sigma_min: sigmaMin,
      curve,
      ...poleSummary(zs, w, fs, data.a, data.b, fScale),
    };
    trace.push(step(m, w.slice(), grid.error(curve), info));
    if (err <= atol) {
      converged = true;
      break;
    }
    if (!free.some(Boolean)) break;
  }

  const m = supportIdx.length;
  const final = trace[trace.length - 1].info;
  let message = converged
    ? `max sample error ${pyG(err, 3)} ≤ tol·max|f| = ${pyG(atol, 3)} with ${m} support points ` +
      `(type (${m - 1}, ${m - 1}))`
    : `max_terms = ${maxTerms} reached: max sample error ${pyG(err, 3)} > tol·max|f| = ${pyG(atol, 3)}`;
  const nReal = Number(final.n_interval_poles);
  if (nReal)
    message += `; warning: ${nReal} certified real pole(s) of r in [${pyG(data.a, 6)}, ${pyG(data.b, 6)}]`;

  const zs = supportIdx.map((i) => zAll[i]);
  const fs = supportIdx.map((i) => fAll[i]);
  const curve = final.curve as number[];
  const maxErr = grid.error(curve);
  const atSupport = barycentricEval(zs, w, fs, zs);
  const nodeRes = atSupport.reduce((e, v, i) => {
    const d = Math.abs(v - fs[i]);
    return d > e || Number.isNaN(d) ? d : e;
  }, 0);
  return {
    method: 'aaa',
    x: w.slice(),
    fun: maxErr,
    converged,
    message,
    nIter: trace[trace.length - 1].k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {
      kind: 'aaa',
      coefficients: w.slice(),
      nodes: zs,
      values: fs,
      domain: [data.a, data.b],
      eval: { x: grid.t, y: curve, f_true: grid.truth },
      max_error: maxErr,
      node_residual: nodeRes,
      scaling,
      sample_error: err,
      errors,
      sigma_min: sigmas,
      poles: final.poles,
      residues: final.residues,
      n_doublets: final.n_doublets,
      n_interval_poles: final.n_interval_poles,
      interval_poles: final.interval_poles,
    },
  };
};

// ---------------------------------------------------------------------------------------
// Floater–Hormann
// ---------------------------------------------------------------------------------------

/**
 * Floater–Hormann weights for sorted distinct nodes x_0 < … < x_n (FH 2007, eq. (18)):
 * w_k = (−1)^{k−d} Σ_{i∈J_k} Π_{j=i, j≠k}^{i+d} 1/|x_k − x_j|, J_k = {i ∈ 0..n−d : k−d ≤ i ≤ k},
 * each factor scaled by h = (x_n − x_0)/n (a common factor h^d that cancels in r). Returns the
 * weights and the windows [min J_k, max J_k]; throws unless 0 ≤ d ≤ n.
 */
export function floaterHormannWeights(
  x: readonly number[],
  d: number,
): { w: number[]; windows: [number, number][] } {
  const n = x.length - 1;
  if (!(d >= 0 && d <= n))
    throw new Error(`need 0 ≤ d ≤ n = ${n} (number of nodes - 1); got d = ${d}`);
  const h = n > 0 ? (x[n] - x[0]) / n : 1.0;
  const w = new Array<number>(n + 1).fill(0);
  const windows: [number, number][] = [];
  for (let k = 0; k <= n; k++) {
    const iLo = Math.max(0, k - d),
      iHi = Math.min(k, n - d);
    let total = 0.0;
    for (let i = iLo; i <= iHi; i++) {
      let prod = 1.0;
      for (let j = i; j <= i + d; j++) if (j !== k) prod *= h / Math.abs(x[k] - x[j]);
      total += prod;
    }
    w[k] = ((k - d) % 2 === 0 ? 1.0 : -1.0) * total;
    windows.push([iLo, iHi]);
  }
  return { w, windows };
}

const floaterHormann: MethodFn<ProblemArg> = (problem, { d = 3 }) => {
  const data = resolve(problem);
  const [xs, ys] = sortedData(data, 1);
  if (typeof d === 'boolean' || !Number.isInteger(Number(d)))
    throw new Error('d must be an integer');
  const dd = Number(d);
  const { w, windows } = floaterHormannWeights(xs, dd);
  const grid = new Grid(data);
  const n = xs.length - 1;
  const trace: Step[] = [];
  let curve: number[] = [];
  for (let k = 0; k <= n; k++) {
    const info: Record<string, unknown> = {
      node_index: k,
      node: [xs[k], ys[k]],
      weight: w[k],
      window: [windows[k][0], windows[k][1]],
    };
    let fun: number | null = null;
    if (k === n) {
      curve = barycentricEval(xs, w, ys, grid.t);
      info.curve = curve;
      fun = grid.error(curve);
    }
    trace.push(step(k, w.slice(0, k + 1), fun, info));
  }
  // Python `_finish`.
  const atNodes = barycentricEval(xs, w, ys, xs);
  const finite = allFinite(w) && allFinite(curve) && allFinite(atNodes);
  let residual = Infinity;
  if (finite) residual = atNodes.reduce((e, v, i) => Math.max(e, Math.abs(v - ys[i])), 0);
  const limit = NODE_RTOL * Math.max(maxAbs(ys), TINY);
  const ok = finite && residual <= limit;
  const err = grid.error(curve);
  const message = ok
    ? `interpolant through ${xs.length} nodes built; max node residual ${pyG(residual, 3)}`
    : finite
      ? `the interpolant misses the data by ${pyG(residual, 3)} at a node (> ${pyG(limit, 3)}): ` +
        'round-off has destroyed this representation for these nodes'
      : 'non-finite coefficients or values (overflow): the interpolant is unusable';
  return {
    method: 'floater_hormann',
    x: w.slice(),
    fun: err,
    converged: ok,
    message,
    nIter: n,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {
      kind: 'floater_hormann',
      coefficients: w.slice(),
      nodes: xs,
      values: ys,
      domain: [data.a, data.b],
      eval: { x: grid.t, y: curve, f_true: grid.truth },
      max_error: err,
      node_residual: residual,
      d: dd,
    },
  };
};

// ---------------------------------------------------------------------------------------
// Registration (ids, params and references exactly as in Python)
// ---------------------------------------------------------------------------------------

export const RATIONAL_DOCS: Record<'aaa' | 'floater_hormann', MethodDoc> = {
  aaa: {
    order: 'geometric for analytic f',
    rule: 'r(x) = \\frac{\\sum_j \\frac{w_j f_j}{x - z_j}}{\\sum_j \\frac{w_j}{x - z_j}},\\qquad z_m = \\arg\\max_{Z}|f - r_{m-1}|,\\quad \\mathbf{w} = \\arg\\min_{\\|\\mathbf{w}\\|=1}\\|A^{(m)}\\mathbf{w}\\|',
    intuition:
      'Each step adds the sample where the current rational function fits worst as a new support point zₘ; r interpolates every support point. The weights w then fit all other samples in the least-squares sense: w is the smallest singular vector of the Loewner matrix with entries (Fᵢ − fⱼ)/(Zᵢ − zⱼ). On smooth data the poles of r lie off the interval, near the singularities of f; on noisy data AAA can put real poles between the samples, and a pole with a tiny residue pairs with a nearby zero (a Froissart doublet).',
    pros: [
      'near-best rational approximation',
      'handles poles and |x|-type singularities',
      'no parameters to tune',
    ],
    cons: ['fits the samples only: r can have a real pole between two samples', 'O(M m³) work'],
  },
  floater_hormann: {
    order: '$O(h^{d+1})$',
    rule: 'w_k = (-1)^{k-d}\\sum_{i \\in J_k}\\prod_{j=i,\\,j\\ne k}^{i+d}\\frac{1}{|x_k - x_j|},\\qquad r(x) = \\frac{\\sum_k \\frac{w_k}{x - x_k}\\,y_k}{\\sum_k \\frac{w_k}{x - x_k}}',
    intuition:
      'Blend the local polynomials $p_i$ of degree d through the d + 1 consecutive nodes $x_i, \\dots, x_{i+d}$. The blend is a rational interpolant in barycentric form whose weight wₖ sums over the window Jₖ of local polynomials that contain node k. It has no real poles for any d, and no Runge oscillation on equispaced nodes.',
    pros: ['no real poles (Theorem 1)', 'explicit weights, O(n d²)', 'stable on equispaced nodes'],
    cons: ['only $O(h^{d+1})$, not geometric', 'the Lebesgue constant grows like $2^d$'],
  },
};

registerMethod(
  {
    id: 'aaa',
    family: 'interpolation',
    name: 'AAA rational approximation',
    params: [
      param.float('tol', 1e-13, {
        min: 1e-15,
        max: 1e-2,
        log: true,
        help: 'Stop when max over the samples of |f − r| ≤ tol·max|f| (NST 2018 default 10⁻¹³).',
        label: 'Tolerance',
        tex: '\\max_Z|f - r| \\le',
      }),
      param.int('max_terms', 100, {
        min: 1,
        max: 200,
        help: 'Maximum number m of support points; r has type (m − 1, m − 1) (NST 2018 mmax = 100).',
        label: 'Max support points',
        tex: 'm_{\\max}',
      }),
      param.choice('scaling', 'columns', ['columns', 'none'], {
        help:
          "'columns': SVD of the Loewner matrix with unit-norm columns (stable when " +
          "column norms differ by many orders); 'none': NST 2018 Fig. 4.1 exactly.",
        label: 'Loewner scaling',
      }),
    ],
    needs: ['data'],
    order: 'root-exponential for |x|-type singularities, geometric for analytic f',
    summary:
      'Add the worst-fit sample as a support point, then choose the barycentric weights ' +
      'that best fit all other samples (smallest singular vector of a Loewner matrix).',
    references: [
      'Nakatsukasa, Sète & Trefethen (2018), The AAA algorithm for rational approximation, ' +
        'SIAM J. Sci. Comput. 40(3): Fig. 4.1 (algorithm), eqs. (3.5)–(3.10) (Loewner ' +
        'least-squares problem), eq. (3.11) (poles), §5 (Froissart doublets)',
      'Schneider & Werner (1986), Some new aspects of rational interpolation, Math. Comp. ' +
        '47: sign condition on the weights (certified real poles)',
      'van der Sluis (1969), Condition numbers and equilibration of matrices, Numer. Math. ' +
        '14 (column scaling)',
    ],
  },
  aaa,
  RATIONAL_DOCS.aaa,
);

registerMethod(
  {
    id: 'floater_hormann',
    family: 'interpolation',
    name: 'Floater–Hormann rational interpolation',
    params: [
      param.int('d', 3, {
        min: 0,
        max: 8,
        help:
          'Degree of the blended local interpolants (0 ≤ d ≤ number of nodes − 1); ' +
          "error O(hᵈ⁺¹). d = 0 is Berrut's interpolant; d = n is the polynomial.",
        label: 'Blend degree',
        tex: 'd',
      }),
    ],
    needs: ['data'],
    order: 'O(hᵈ⁺¹), no real poles',
    summary:
      'Blend the local degree-d interpolants into one rational interpolant: explicit ' +
      'barycentric weights, no real poles, and no Runge oscillation.',
    references: [
      'Floater & Hormann (2007), Barycentric rational interpolation with no poles and high ' +
        'rates of approximation, Numer. Math. 107: eqs. (4), (5) (blend), (11), (18) ' +
        '(weights), Theorem 1 (no real poles), Theorem 2 (O(hᵈ⁺¹))',
      'Berrut & Trefethen (2004), Barycentric Lagrange Interpolation, SIAM Review 46(3), ' +
        'eq. (4.2) (evaluation)',
    ],
  },
  floaterHormann,
  RATIONAL_DOCS.floater_hormann,
);
