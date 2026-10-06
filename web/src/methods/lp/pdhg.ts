/**
 * Restarted primal–dual hybrid gradient (PDHG) for linear programs, in the style of PDLP — TS
 * port of `numopt.lp.pdhg` (src/numopt/lp/pdhg.py; its module docstring has the mathematics,
 * the references and the reasoning behind every NOTE).
 *
 * The LP is solved in the equality standard form min c̃ᵀz s.t. A z = b, z ≥ 0 (one slack per ≤
 * row) through the saddle point min_{z≥0} max_y c̃ᵀz − yᵀAz + bᵀy with the PDHG step
 *
 *   z⁺ = proj_{z ≥ 0}(z − τ(c̃ − Aᵀy)),   y⁺ = y + σ(b − A(2z⁺ − z)),   τ = η/ω, σ = ηω, η = 0.9/‖Ã‖₂,
 *
 * on the Ruiz + Pock–Chambolle scaled data Ã = D₁AD₂, and restarts to the running average of the
 * epoch (fixed period, or when the normalized duality gap has fallen by β).
 *
 * Floating-point order follows NumPy where it matters for parity: row sums of a contiguous axis
 * use NumPy's pairwise summation (`npSum`), column sums add the rows in order, everything else is
 * a left-to-right loop (BLAS dot/gemv; differences are rounding-level).
 *
 * Info keys (snake_case, as in Python): x, x_pdhg, x_avg, y, kkt_last, kkt_avg, kkt,
 * primal_residual, dual_residual, gap, omega, tau, sigma, restarted, epoch, epoch_len,
 * normalized_gap, restart_threshold, matvecs.
 */
import { registerMethod, param } from '../../core/registry';
import type { LinearProgram, Matrix, Result, Step } from '../../core/types';
import { checkLp, dotv, pyG, zeros } from './simplex';
import { equalityForm, singularValues } from './interior_point';

/** Step-size factor η·‖A‖₂ (Applegate et al. 2021, §2 baseline; 2023, §7). */
export const ETA_FACTOR = 0.9;
/** Primal-weight smoothing θ of PDLP Algorithm 3. */
export const THETA = 0.5;
/** Ruiz passes before the Pock–Chambolle pass. */
export const RUIZ_ITERS = 10;
/** The "zero" of InitializePrimalWeight and of Algorithm 3. */
export const ZERO = 1e-10;
/** First restart length τ⁰ of the adaptive scheme (2023 paper, eq. 30). */
export const TAU0 = 1;

export const RESTARTS = ['none', 'fixed', 'adaptive'] as const;
export const PRIMAL_WEIGHTS = ['unit', 'balanced', 'adaptive'] as const;
export const PRECONDITIONERS = ['none', 'ruiz_pc'] as const;

type Vec = number[];

// ─────────────────────────────────────────────────────────────────────────────────────────
// NumPy-order helpers
// ─────────────────────────────────────────────────────────────────────────────────────────

/** NumPy's pairwise summation of a contiguous float64 array (`np.sum` along one axis). */
function pairwise(a: readonly number[], lo: number, n: number): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[lo + i];
    return res;
  }
  if (n <= 128) {
    const r = [0, 1, 2, 3, 4, 5, 6, 7].map((j) => a[lo + j]);
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
/** `np.sum(v)` for a 1-D array. */
export const npSum = (v: readonly number[]) => 0 + pairwise(v, 0, v.length);

const matvec = (A: Matrix, x: readonly number[]) => A.map((r) => dotv(r, x));
/** Aᵀ y */
function rmatvec(A: Matrix, y: readonly number[], N: number): Vec {
  const out = zeros(N);
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let i = 0; i < A.length; i++) s += A[i][j] * y[i];
    out[j] = s;
  }
  return out;
}
const norm2 = (v: readonly number[]) => Math.sqrt(dotv(v, v));
/** ‖v‖₂ without overflow of the squares (`interior_point._norm`): m·‖v/m‖₂, m = ‖v‖∞. */
function safeNorm(v: readonly number[]): number {
  const m = v.reduce((a, x) => (Math.abs(x) > a || Number.isNaN(x) ? Math.abs(x) : a), 0);
  if (m === 0.0 || !Number.isFinite(m)) return m;
  return m * norm2(v.map((x) => x / m));
}
/** `np.maximum` / `np.minimum` (NaN propagates). */
const npMax = (a: number, b: number) => (Number.isNaN(a) || Number.isNaN(b) ? NaN : Math.max(a, b));
const npMin = (a: number, b: number) => (Number.isNaN(a) || Number.isNaN(b) ? NaN : Math.min(a, b));
const allFinite = (v: readonly number[]) => v.every(Number.isFinite);
const sub = (a: readonly number[], b: readonly number[]) => a.map((v, i) => v - b[i]);
/** Python `f"{v:.2e}"`. */
function pyE2(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const [mant, e] = v.toExponential(2).split('e');
  const en = Number(e);
  return `${mant}e${en < 0 ? '-' : '+'}${String(Math.abs(en)).padStart(2, '0')}`;
}
const pyTuple = (xs: readonly string[]) => `(${xs.map((x) => `'${x}'`).join(', ')})`;

// ─────────────────────────────────────────────────────────────────────────────────────────
// Problem data and preconditioning
// ─────────────────────────────────────────────────────────────────────────────────────────

/** min cᵀz s.t. A z = b, z ≥ 0 (unscaled); the original variables are z[:n]. */
export interface StandardLP {
  A: number[][];
  b: Vec;
  c: Vec;
  n: number;
  mUb: number;
  sign: number;
}

export function pdhgStandardForm(lp: LinearProgram): StandardLP {
  const { A, b, c, n, sign } = equalityForm(lp);
  const N = c.length;
  return { A, b, c, n, mUb: N - n, sign };
}

/** Diagonal scalings d₁ (m), d₂ (N) with Ã = diag(d₁) A diag(d₂): Ruiz, then Pock–Chambolle. */
export function ruizPockChambolle(A: Matrix, ruizIters = RUIZ_ITERS): { d1: Vec; d2: Vec } {
  const m = A.length,
    N = m ? A[0].length : 0;
  const d1 = new Array<number>(m).fill(1),
    d2 = new Array<number>(N).fill(1);
  let K = A.map((r) => [...r]);
  for (let it = 0; it < ruizIters; it++) {
    const r = K.map((row) => Math.sqrt(row.reduce((a, v) => npMax(a, Math.abs(v)), 0)));
    const s = Array.from({ length: N }, (_, j) =>
      Math.sqrt(K.reduce((a, row) => npMax(a, Math.abs(row[j])), 0)),
    );
    for (let i = 0; i < m; i++) if (r[i] === 0) r[i] = 1;
    for (let j = 0; j < N; j++) if (s[j] === 0) s[j] = 1;
    K = K.map((row, i) => row.map((v, j) => v / r[i] / s[j]));
    for (let i = 0; i < m; i++) d1[i] /= r[i];
    for (let j = 0; j < N; j++) d2[j] /= s[j];
  }
  const r = K.map((row) => Math.sqrt(npSum(row.map(Math.abs))));
  const colSum = zeros(N);
  for (const row of K) for (let j = 0; j < N; j++) colSum[j] += Math.abs(row[j]);
  const s = colSum.map(Math.sqrt);
  for (let i = 0; i < m; i++) if (r[i] === 0) r[i] = 1;
  for (let j = 0; j < N; j++) if (s[j] === 0) s[j] = 1;
  return { d1: d1.map((v, i) => v / r[i]), d2: d2.map((v, j) => v / s[j]) };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Normalized duality gap, primal weight, KKT error
// ─────────────────────────────────────────────────────────────────────────────────────────

/** ‖(dz, dy)‖_ω = √(ω‖dz‖² + ‖dy‖²/ω). */
export function weightedNorm(dz: readonly number[], dy: readonly number[], omega: number): number {
  return Math.sqrt(omega * dotv(dz, dz) + dotv(dy, dy) / omega);
}

/**
 * ρ_r(z, y) for min cᵀz s.t. Az = b, z ≥ 0 in the norm ‖·‖_ω (2023 paper, eq. 4a), computed
 * exactly by sorting the break points of the trust-region path (see the Python docstring).
 */
export function normalizedDualityGap(
  z: readonly number[],
  Az: readonly number[],
  ATy: readonly number[],
  b: readonly number[],
  c: readonly number[],
  r: number,
  omega: number,
): number {
  if (!(r >= 0.0)) throw new Error('radius must be ≥ 0');
  const gz = ATy.map((v, i) => v - c[i]);
  const gy = b.map((v, i) => v - Az[i]);
  const qz = gz.map((g) => (g * g) / omega);
  const clampIdx: number[] = [];
  const freeQ: number[] = [];
  gz.forEach((g, i) => {
    if (g < 0.0) clampIdx.push(i);
    else freeQ.push(qz[i]);
  });
  const qFree = npSum(freeQ) + omega * npSum(gy.map((v) => v * v));
  let bp = clampIdx.map((i) => (z[i] * omega) / -gz[i]);
  let s = clampIdx.map((i) => omega * (z[i] * z[i]));
  let q = clampIdx.map((i) => qz[i]);
  if (r === 0.0) return Math.sqrt(qFree + npSum(q.filter((_, k) => bp[k] > 0.0)));
  const order = bp.map((_, k) => k).sort((a, b2) => bp[a] - bp[b2] || a - b2);
  bp = order.map((k) => bp[k]);
  s = order.map((k) => s[k]);
  q = order.map((k) => q[k]);
  const K = bp.length;
  const S = [0.0];
  for (let k = 0; k < K; k++) S.push(S[k] + s[k]);
  // Reverse cumulative sum of q (np.cumsum(q[::-1])[::-1]), then 0, plus q_free.
  const rev = zeros(K);
  let acc = 0;
  for (let k = K - 1; k >= 0; k--) {
    acc += q[k];
    rev[k] = acc;
  }
  const Q = [...rev, 0.0].map((v) => qFree + v);
  const r2 = r * r;
  let hit = -1;
  for (let k = 0; k < K; k++) {
    if (S[k] + bp[k] * bp[k] * Q[k] >= r2) {
      hit = k;
      break;
    }
  }
  let k: number;
  if (hit >= 0) k = hit;
  else if (Q[K] > 0.0) k = K;
  else {
    let num = 0;
    for (const i of clampIdx) num += gz[i] * -z[i];
    return num / r;
  }
  const t = Q[k] > 0.0 ? Math.sqrt(Math.max(r2 - S[k], 0.0) / Q[k]) : 0.0;
  const dz = z.map((zi, i) => npMax(-zi, (t * gz[i]) / omega));
  const dy = gy.map((g) => t * omega * g);
  return (dotv(gz, dz) + dotv(gy, dy)) / r;
}

/** PDLP Algorithm 3: exp(θ log(Δy/Δz) + (1 − θ) log ω) when Δz, Δy > ZERO, else ω. */
export function primalWeightUpdate(dz: number, dy: number, omega: number, theta = THETA): number {
  if (dz > ZERO && dy > ZERO && Number.isFinite(dz) && Number.isFinite(dy))
    return Math.exp(theta * Math.log(dy / dz) + (1.0 - theta) * Math.log(omega));
  return omega;
}

/** The relative KKT error and its three terms (Applegate et al. 2021, eq. 6). */
export interface KKT {
  error: number;
  primal: number;
  dual: number;
  gap: number;
  pobj: number;
  dobj: number;
}

/** Relative KKT error of (z, y) for the unscaled standard form. */
export function kktError(
  lp: StandardLP,
  z: readonly number[],
  y: readonly number[],
  Az: readonly number[],
  ATy: readonly number[],
): KKT {
  const pobj = dotv(lp.c, z);
  const dobj = dotv(lp.b, y);
  const primal = safeNorm(sub(Az, lp.b)) / (1.0 + safeNorm(lp.b));
  const dual = safeNorm(lp.c.map((v, i) => npMin(v - ATy[i], 0.0))) / (1.0 + safeNorm(lp.c));
  const gap = Math.abs(pobj - dobj) / (1.0 + Math.abs(pobj) + Math.abs(dobj));
  const terms = [primal, dual, gap];
  const error = terms.every(Number.isFinite) ? Math.max(...terms) : Infinity;
  return { error, primal, dual, gap, pobj, dobj };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// The method
// ─────────────────────────────────────────────────────────────────────────────────────────

/** z⁰: x0 (negative entries set to 0) and slacks max(b_ub − A_ub x0, 0); zero if absent. */
function start(lp: StandardLP, x0: unknown): Vec {
  const z = zeros(lp.c.length);
  if (x0 === undefined || x0 === null) return z;
  const x = (Array.isArray(x0) ? (x0 as unknown[]).flat(Infinity) : [x0]).map(Number);
  if (x.length !== lp.n || !allFinite(x))
    throw new Error(`x0 must be a finite vector of length ${lp.n}`);
  for (let j = 0; j < lp.n; j++) z[j] = Math.max(x[j], 0.0);
  for (let i = 0; i < lp.mUb; i++) {
    const ax = dotv(lp.A[i].slice(0, lp.n), z.slice(0, lp.n));
    z[lp.n + i] = npMax(lp.b[i] - ax, 0.0);
  }
  return z;
}

export interface PdhgOptions {
  x0?: unknown;
  restart?: string;
  restart_period?: number;
  beta?: number;
  restart_check_every?: number;
  primal_weight?: string;
  precondition?: string;
  tol?: number;
  max_iter?: number;
}

/** Restarted PDHG for LP (Applegate, Hinder, Lu & Lubin 2023, Algorithm 1; PDLP §3). */
export function restartedPdhg(problem: LinearProgram, o: PdhgOptions = {}): Result {
  const restart = o.restart ?? 'adaptive';
  const restartPeriod = o.restart_period ?? 64;
  const beta = o.beta ?? Math.exp(-1.0);
  const checkEvery = o.restart_check_every ?? 40;
  const primalWeight = o.primal_weight ?? 'balanced';
  const precondition = o.precondition ?? 'ruiz_pc';
  const tol = o.tol ?? 1e-8;
  const maxIter = o.max_iter ?? 20_000;

  const lp = checkLp(problem);
  if (!(RESTARTS as readonly string[]).includes(restart))
    throw new Error(`restart must be one of ${pyTuple(RESTARTS)}`);
  if (!(PRIMAL_WEIGHTS as readonly string[]).includes(primalWeight))
    throw new Error(`primal_weight must be one of ${pyTuple(PRIMAL_WEIGHTS)}`);
  if (!(PRECONDITIONERS as readonly string[]).includes(precondition))
    throw new Error(`precondition must be one of ${pyTuple(PRECONDITIONERS)}`);
  if (!(beta > 0.0 && beta < 1.0)) throw new Error('beta must be in (0, 1)');
  if (restartPeriod < 1 || checkEvery < 1 || maxIter < 1)
    throw new Error('restart_period, restart_check_every and max_iter must be ≥ 1');
  if (!(tol > 0.0)) throw new Error('tol must be > 0');

  const std = pdhgStandardForm(lp);
  const m = std.A.length,
    N = std.c.length;
  if (m === 0 || !std.A.some((row) => row.some((v) => v !== 0)))
    throw new Error(`${lp.id}: restarted_pdhg needs at least one nonzero constraint row`);
  const { d1, d2 } =
    precondition === 'ruiz_pc'
      ? ruizPockChambolle(std.A)
      : { d1: new Array<number>(m).fill(1), d2: new Array<number>(N).fill(1) };
  const A = std.A.map((row, i) => row.map((v, j) => v * d1[i] * d2[j]));
  const b = std.b.map((v, i) => d1[i] * v);
  const c = std.c.map((v, j) => d2[j] * v);
  const normA = singularValues(A)[0];
  const eta = ETA_FACTOR / normA;
  let omega = 1.0;
  if (primalWeight !== 'unit') {
    const nc = norm2(c),
      nb = norm2(b);
    if (nc > ZERO && nb > ZERO) omega = nc / nb;
  }
  let tau = eta / omega,
    sigma = eta * omega;

  const unscale = (zs: Vec, ys: Vec, Azs: Vec, ATys: Vec) => ({
    z: zs.map((v, j) => d2[j] * v),
    y: ys.map((v, i) => d1[i] * v),
    Az: Azs.map((v, i) => v / d1[i]),
    ATy: ATys.map((v, j) => v / d2[j]),
  });
  const kktOf = (zs: Vec, ys: Vec, Azs: Vec, ATys: Vec): KKT => {
    const u = unscale(zs, ys, Azs, ATys);
    return kktError(std, u.z, u.y, u.Az, u.ATy);
  };
  const origX = (zs: Vec): Vec => zs.slice(0, std.n).map((v, j) => d2[j] * v);
  const funOf = (x: Vec) => std.sign * dotv(std.c.slice(0, std.n), x);

  interface Flags {
    stepSize: number | null;
    restarted: boolean;
    epoch: number;
    epochLen: number;
    normalizedGap: number | null;
    restartThreshold: number | null;
    matvecs: number;
  }
  const makeStep = (
    k: number,
    zs: Vec,
    ys: Vec,
    zPdhg: Vec,
    zAvg: Vec,
    kl: KKT,
    ka: KKT,
    f: Flags,
  ): Step => {
    const x = origX(zs);
    return {
      k,
      x,
      fun: funOf(x),
      gradNorm: null,
      stepSize: f.stepSize,
      info: {
        x: [...x],
        x_pdhg: origX(zPdhg),
        x_avg: origX(zAvg),
        y: ys.map((v, i) => d1[i] * v),
        kkt_last: kl.error,
        kkt_avg: ka.error,
        kkt: Math.min(kl.error, ka.error),
        primal_residual: kl.primal,
        dual_residual: kl.dual,
        gap: kl.gap,
        omega,
        tau,
        sigma,
        restarted: f.restarted,
        epoch: f.epoch,
        epoch_len: f.epochLen,
        normalized_gap: f.normalizedGap,
        restart_threshold: f.restartThreshold,
        matvecs: f.matvecs,
      },
    };
  };

  // State in the scaled space (see the Python source).
  let z = start(std, o.x0).map((v, j) => v / d2[j]);
  let y = zeros(m);
  let Az = matvec(A, z),
    ATy = rmatvec(A, y, N);
  let matvecs = 2;
  let zStart = [...z],
    yStart = [...y];
  let sums = [zeros(N), zeros(m), zeros(m), zeros(N)];
  let t = 0;
  let epoch = 0;
  let rhoRef = Infinity;

  const k0 = kktOf(z, y, Az, ATy);
  const trace: Step[] = [
    makeStep(0, z, y, z, z, k0, k0, {
      stepSize: null,
      restarted: false,
      epoch: 0,
      epochLen: 0,
      normalizedGap: null,
      restartThreshold: null,
      matvecs,
    }),
  ];
  let best = {
    err: k0.error,
    z: [...z],
    y: [...y],
    Az: [...Az],
    ATy: [...ATy],
    which: 'iterate',
  };
  let status = k0.error <= tol ? 'optimal' : 'max_iter';
  let k = 0;
  while (status === 'max_iter' && k < maxIter) {
    k += 1;
    const tauUsed = tau;
    const zNew = z.map((v, j) => npMax(v - tau * (c[j] - ATy[j]), 0.0));
    const AzOld = Az;
    z = zNew;
    Az = matvec(A, z);
    y = y.map((v, i) => v + sigma * (b[i] - (2.0 * Az[i] - AzOld[i])));
    ATy = rmatvec(A, y, N);
    matvecs += 2;
    t += 1;
    [z, y, Az, ATy].forEach((v, q) => {
      const acc = sums[q];
      for (let i = 0; i < acc.length; i++) acc[i] += v[i];
    });
    const [zAvg, yAvg, AzAvg, ATyAvg] = sums.map((acc) => acc.map((v) => v / t));

    const kLast = kktOf(z, y, Az, ATy);
    const kAvg = kktOf(zAvg, yAvg, AzAvg, ATyAvg);
    if (!(allFinite(z) && allFinite(y))) status = 'nonfinite';
    if (kLast.error < best.err)
      best = {
        err: kLast.error,
        z: [...z],
        y: [...y],
        Az: [...Az],
        ATy: [...ATy],
        which: 'iterate',
      };
    if (kAvg.error < best.err)
      best = { err: kAvg.error, z: zAvg, y: yAvg, Az: AzAvg, ATy: ATyAvg, which: 'average' };
    if (status === 'max_iter' && Math.min(kLast.error, kAvg.error) <= tol) status = 'optimal';

    let rhoAvg: number | null = null;
    let threshold: number | null = null;
    let doRestart = false;
    if (status === 'max_iter') {
      if (restart === 'fixed') doRestart = t >= restartPeriod;
      else if (restart === 'adaptive' && t % checkEvery === 0) {
        if (epoch === 0) doRestart = t >= TAU0;
        else {
          const r = weightedNorm(sub(zAvg, zStart), sub(yAvg, yStart), omega);
          rhoAvg = normalizedDualityGap(zAvg, AzAvg, ATyAvg, b, c, r, omega);
          threshold = beta * rhoRef;
          doRestart = rhoAvg <= threshold;
        }
      }
    }
    const zPdhg = z,
      epochLen = t;
    if (doRestart) {
      const zPrev = zStart,
        yPrev = yStart;
      z = zAvg;
      y = yAvg;
      Az = AzAvg;
      ATy = ATyAvg;
      zStart = [...z];
      yStart = [...y];
      sums = [zeros(N), zeros(m), zeros(m), zeros(N)];
      t = 0;
      epoch += 1;
      if (primalWeight === 'adaptive') {
        omega = primalWeightUpdate(norm2(sub(zStart, zPrev)), norm2(sub(yStart, yPrev)), omega);
        tau = eta / omega;
        sigma = eta * omega;
      }
      if (restart === 'adaptive') {
        const rRef = weightedNorm(sub(zStart, zPrev), sub(yStart, yPrev), omega);
        rhoRef = normalizedDualityGap(zStart, Az, ATy, b, c, rRef, omega);
      }
    }
    trace.push(
      makeStep(k, z, y, zPdhg, zAvg, kLast, kAvg, {
        stepSize: tauUsed,
        restarted: doRestart,
        epoch,
        epochLen,
        normalizedGap: rhoAvg,
        restartThreshold: threshold,
        matvecs,
      }),
    );
  }

  const kb = kktOf(best.z, best.y, best.Az, best.ATy);
  const out = unscale(best.z, best.y, best.Az, best.ATy);
  let message: string;
  if (status === 'optimal')
    message =
      `relative KKT error ${pyE2(best.err)} ≤ tol=${pyG(tol)} at the ${best.which} after ${k} iterations ` +
      `(${matvecs} products with A or Aᵀ, ${epoch} restarts)`;
  else if (status === 'nonfinite') message = `stopped: non-finite iterate at iteration ${k}`;
  else
    message =
      `reached max_iter=${maxIter}; best relative KKT error ${pyE2(best.err)} > tol=${pyG(tol)} ` +
      '(no infeasibility detection: the LP may be infeasible or unbounded)';
  const xOut = out.z.slice(0, std.n);
  return {
    method: 'restarted_pdhg',
    x: xOut,
    fun: funOf(xOut),
    converged: status === 'optimal',
    message,
    nIter: k,
    nFev: 0,
    nGev: 0,
    nHev: 0,
    trace,
    extra: {
      status,
      y: out.y,
      output: best.which,
      kkt: best.err,
      primal_residual: kb.primal,
      dual_residual: kb.dual,
      gap: kb.gap,
      n_matvec: matvecs,
      n_restarts: epoch,
      omega,
      eta,
      norm_A: normA,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────────────────

registerMethod<LinearProgram>(
  {
    id: 'restarted_pdhg',
    family: 'lp',
    name: 'Restarted PDHG (PDLP-style)',
    params: [
      param.choice('restart', 'adaptive', [...RESTARTS], {
        help: 'none: plain PDHG; fixed: restart to the average every restart_period iterations; adaptive: restart when the normalized duality gap falls by beta.',
        label: 'Restart',
      }),
      param.int('restart_period', 64, {
        min: 1,
        max: 100_000,
        help: "Restart length for restart='fixed'.",
        label: 'Restart period',
      }),
      param.float('beta', Math.exp(-1.0), {
        min: 0.01,
        max: 0.99,
        help: "Required decay factor of the normalized duality gap (restart='adaptive').",
        label: 'Gap decay',
        tex: '\\beta',
      }),
      param.int('restart_check_every', 40, {
        min: 1,
        max: 10_000,
        help: 'Evaluate the adaptive restart test every this many iterations (PDLP: 40).',
        label: 'Restart check every',
      }),
      param.choice('primal_weight', 'balanced', [...PRIMAL_WEIGHTS], {
        help: "unit: ω = 1; balanced: ω = ‖c‖/‖b‖; adaptive: balanced start, then PDLP's smoothed update at every restart.",
        label: 'Primal weight',
        tex: '\\omega',
      }),
      param.choice('precondition', 'ruiz_pc', [...PRECONDITIONERS], {
        help: 'Diagonal preconditioning: 10 Ruiz passes then Pock–Chambolle (α = 1).',
        label: 'Preconditioner',
      }),
      param.float('tol', 1e-8, {
        min: 1e-12,
        max: 1e-2,
        log: true,
        help: 'Stop when the relative KKT error (gap, primal and dual residual) is ≤ tol.',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
      param.int('max_iter', 20_000, {
        min: 1,
        max: 1_000_000,
        help: 'Iteration limit.',
        label: 'Iteration budget',
      }),
    ],
    needs: ['lp'],
    order: 'linear with restarts (sharp LPs); O(1/k) for the plain average',
    summary:
      'Take cheap projected primal and dual gradient steps on the Lagrangian, and restart from the running average whenever the normalized duality gap has fallen enough.',
    references: [
      'Applegate, Hinder, Lu & Lubin (2023), Math. Program. 201:133–184, Algorithm 1, restart schemes eqs. 29–30, normalized duality gap eq. 4a',
      'Applegate et al. (2021), NeurIPS, PDLP: eqs. 3–4 (PDHG step), eq. 6 (KKT error), Algorithm 3 (primal weight), §3.5 (preconditioning)',
      'Chambolle & Pock (2011), J. Math. Imaging Vis. 40:120–145, Algorithm 1 (θ = 1)',
      'Pock & Chambolle (2011), ICCV, Lemma 2 (diagonal preconditioning)',
    ],
  },
  (p, o) => restartedPdhg(p, o as PdhgOptions),
  {
    rule:
      '\\begin{aligned} \\mathbf z_{k+1} &= \\max\\!\\big(\\mathbf z_k - \\tau(\\mathbf c - A^{\\top}\\mathbf y_k),\\ \\mathbf 0\\big) \\\\ ' +
      '\\mathbf y_{k+1} &= \\mathbf y_k + \\sigma\\big(\\mathbf b - A(2\\mathbf z_{k+1} - \\mathbf z_k)\\big) \\end{aligned}',
    intuition:
      'A gradient step down in the primal and up in the dual of the Lagrangian, with no linear solve: only products with A and Aᵀ. The iterates spiral around the saddle point; their average converges, and restarting from the average whenever the normalized duality gap has fallen by β turns the slow spiral into linear convergence.',
    order: 'linear with restarts',
    pros: [
      'Only matrix–vector products: scales to huge sparse LPs',
      'Restarts give linear convergence on sharp LPs',
    ],
    cons: [
      'Hundreds of iterations for 8 digits',
      'Iterates can leave the feasible set (only z ≥ 0 is kept)',
      'No infeasibility detection in this version',
    ],
    quantities: [
      { tex: '\\text{KKT}_k', key: 'info.kkt' },
      { tex: '\\omega', key: 'info.omega' },
      { tex: 'n', key: 'info.epoch', label: 'restarts' },
    ],
  },
);
