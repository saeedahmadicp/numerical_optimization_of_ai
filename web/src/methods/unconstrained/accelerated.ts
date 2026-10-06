/**
 * Accelerated first-order methods — TS port of `numopt.unconstrained.accelerated`
 * (src/numopt/unconstrained/accelerated.py): certified step-size schedules, OGM and restarted AGD.
 *
 * Every method minimizes an L-smooth convex f with f and ∇f only. Four of them are gradient
 * descent with normalized steps h_t (OGM: with its momentum), fixed before the run:
 *
 *     x_{t+1} = x_t − (h_t / L) ∇f(x_t),        t = 0, 1, 2, ...                     (GD)
 *
 *   silver_gd                   h_t = 1 + ρ^{ν(t+1)−1}, ρ = 1 + √2 (Altschuler & Parrilo 2024)
 *   silver_gd_strongly_convex   the κ-aware silver block of length n, repeated (JACM 2025, §3)
 *   long_step_gd                Grimmer's periodic patterns (SIAM J. Optim. 2024, Table 1)
 *   ogm                         Kim & Fessler's OGM1 (Math. Program. 2016, §7.1)
 *   fista                       FISTA with g ≡ 0 and O'Donoghue–Candès adaptive restart
 *
 * Conventions (identical to Python): step k = 0 holds x₀, step k ≥ 1 the iterate x_k =
 * x_{k−1} + α_k p_k with `stepSize` = info.alpha = α_k and info.direction = p_k; `gradNorm` =
 * ‖∇f(x_k)‖₂ (scaled as LAPACK dnrm2) for the schedules and OGM, null for FISTA (its gradient is
 * taken at y_k). Stopping test: ‖∇f(x_k)‖₂ ≤ gtol (FISTA: ‖∇f(y_k)‖₂ ≤ gtol). Failures: max_iter,
 * a non-finite f or ∇f, a rise f(x_k) − f(x₀) > 10¹²·max(1, |f(x₀)|), FISTA's failed
 * backtracking or stall. L = 0 means "auto": problem.extra.L, else λ_max(∇²f(x₀)) of a problem
 * tagged "quadratic" (one counted Hessian evaluation, shared with auto μ); otherwise a ValueError.
 *
 * Step.info keys (snake_case, as in Python): direction, alpha; schedules and OGM: grad, h,
 * checkpoint, bound_f (+ bound_dist for the κ-aware silver schedule, y and theta for OGM); FISTA:
 * y, grad_y, grad_norm_y, beta, t, restarted, L, trials.
 *
 * NOTE (port): `np.linalg.eigvalsh` is replaced by the LAPACK-exact 2×2 `eigh` of
 * trust_region.ts (a Jacobi solver for n ≥ 3); JS Math.pow / Math.hypot / Math.log1p may differ
 * from C's libm in the last bit, which moves the schedules by about 10⁻¹⁶ relative.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type {
  Matrix,
  MethodFn,
  Params,
  Problem,
  Result,
  RunOptions,
  Step,
  Vector,
} from '../../core/types';
import { formatG, formatRepr } from '../line_search/methods';
import { ValueError, resolveSmooth, type Counted, type SmoothProblem } from './conjugate_gradient';
import { eigh } from './trust_region';

// ---------------------------------------------------------------------------------------
// Constants (Python module level)
// ---------------------------------------------------------------------------------------

/** The silver ratio ρ = 1 + √2. */
export const RHO = 1.0 + Math.sqrt(2.0);
/** A rise f(x_k) − f(x₀) above F_DIVERGE·max(1, |f(x₀)|) counts as divergence. */
const F_DIVERGE = 1e12;
/** Largest κ-aware silver horizon built in memory: 2²⁰ steps. */
export const MAX_HORIZON = 2 ** 20;
/** Auto horizon: stop doubling once a doubling improves −log(τ_n)/n by less than this. */
const SATURATION_GAIN = 0.01;
/** FISTA backtracking: at most this many trials per iteration. */
export const MAX_BACKTRACK = 60;
/** FISTA backtracking: relative slack for rounding in the test f(p) ≤ Q_L(p, y). */
const BT_RTOL = 1e-12;
/** Auto μ: λ_min(∇²f) ≤ EIG_RESIDUE·n·ε·λ_max is read as 0 (not strongly convex). */
const EIG_RESIDUE = 64;
const EPS = Number.EPSILON;
/** Restart schemes of `fista`. */
export const RESTARTS = ['none', 'function', 'gradient'] as const;

const g3 = (v: number) => formatG(v, 3);

/** Python `format(v, '.3e')`. */
function e3(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const [m, e] = v.toExponential(3).split('e');
  const n = Number(e);
  return `${m}e${n < 0 ? '-' : '+'}${String(Math.abs(n)).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------------------
// Schedules and their certified rates
// ---------------------------------------------------------------------------------------

/** int.bit_length() for 1 ≤ t < 2³¹. */
const bitLength = (t: number): number => (t <= 0 ? 0 : 32 - Math.clz32(t));

/** ν(t): the exponent of the largest power of 2 that divides t ≥ 1 (ν(1) = 0, ν(12) = 2). */
export function twoAdicValuation(t: number): number {
  if (!(t >= 1)) throw new ValueError(`ν(t) needs t ≥ 1, got ${t}`);
  let v = 0;
  let u = t;
  while (u % 2 === 0) {
    u /= 2;
    v++;
  }
  return v;
}

/** The t-th convex silver step h_t = 1 + ρ^{ν(t+1)−1}, t = 0, 1, ... (Part II, eq. 2.1). */
export function silverStep(t: number): number {
  return 1.0 + RHO ** (twoAdicValuation(t + 1) - 1);
}

/** r_k = 1/(1 + √(4ρ^{2k} − 3)): f(x_n) − f* ≤ r_k L‖x₀ − x*‖² at n = 2^k − 1 (Part II, eq. 1.4). */
export function silverRate(k: number): number {
  if (k < 0) throw new ValueError(`k must be ≥ 0, got ${k}`);
  return 1.0 / (1.0 + Math.sqrt(4.0 * RHO ** (2 * k) - 3.0));
}

/** ψ(t) = (1 + κt)/(1 + t) (Part I, eq. 3.5). */
const psi = (t: number, kappa: number) => (1.0 + kappa * t) / (1.0 + t);

/** log τ = 2 log(w/(2 − w)) from log w (−∞ when w = 0), free of cancellation. */
function logTau(logW: number): number {
  if (logW === -Infinity) return -Infinity;
  return 2.0 * (logW - Math.log1p(-Math.expm1(logW)));
}

/** (a_n, b_n, log τ_n) for n = 1, 2, 4, 8, ... (Part I, eqs. 3.1–3.4 and 3.9), one per call. */
function silverScLevels(kappa: number): () => [number, number, number] {
  let z = 1.0 / kappa;
  let logW: number;
  if (kappa === 1.0) logW = -Infinity;
  else if (kappa < 2.0) logW = Math.log((kappa - 1.0) / kappa);
  else logW = Math.log1p(-1.0 / kappa);
  let first = true;
  return () => {
    if (first) {
      first = false;
      const b = psi(z, kappa);
      return [b, b, logTau(logW)];
    }
    const w = Math.exp(logW);
    const u = -Math.expm1(logW);
    const r = Math.hypot(1.0, w);
    const s = w + r;
    const y = z / s;
    logW = 2.0 * logW + Math.log1p(-u / (1.0 + r));
    const wNew = Math.exp(logW);
    z = wNew < 0.5 ? 1.0 - wNew : z * s;
    return [psi(y, kappa), psi(z, kappa), logTau(logW)];
  };
}

function checkKappa(kappa: number): void {
  if (!(Number.isFinite(kappa) && kappa >= 1.0))
    throw new ValueError(`κ = L/μ must be finite and ≥ 1, got ${formatRepr(kappa)}`);
}

function checkHorizon(n: unknown): number {
  const bad = () =>
    new ValueError(`the horizon must be a power of 2 in [1, ${MAX_HORIZON}], got ${String(n)}`);
  if (typeof n !== 'number' || !Number.isFinite(n) || n !== Math.floor(n)) throw bad();
  if (n < 1 || n > MAX_HORIZON || (n & (n - 1)) !== 0) throw bad();
  return n;
}

/** The κ-aware silver schedule h⁽ⁿ⁾ (n a power of 2) and its rate τ_n (Part I, §3, eq. 3.8). */
export function silverScSchedule(kappa: number, n: number): [Vector, number] {
  checkKappa(kappa);
  const horizon = checkHorizon(n);
  const next = silverScLevels(kappa);
  let [, b, lt] = next();
  let h: Vector = [b];
  for (let m = 1; m < horizon; m *= 2) {
    let a: number;
    [a, b, lt] = next();
    const body = h.slice(0, -1);
    h = [...body, a, ...body, b];
  }
  return [h, Math.exp(lt)];
}

/** τ_n of the κ-aware silver schedule (Part I, eq. 3.9), n a power of 2. */
export function silverScRate(kappa: number, n: number): number {
  checkKappa(kappa);
  const horizon = checkHorizon(n);
  const next = silverScLevels(kappa);
  let lt = next()[2];
  for (let i = 0; i < bitLength(horizon) - 1; i++) lt = next()[2];
  return Math.exp(lt);
}

/** The smallest n = 2^j after which a doubling improves −log(τ_n)/n by < 1 % (≤ 2²⁰). */
export function silverScAutoHorizon(kappa: number): number {
  checkKappa(kappa);
  const next = silverScLevels(kappa);
  let lt = next()[2];
  let n = 1;
  while (n < MAX_HORIZON) {
    const lt2 = next()[2];
    if (lt === -Infinity) return n; // κ = 1: one step of 1/L is exact
    if (lt2 / (2 * n) >= ((1.0 + SATURATION_GAIN) * lt) / n) return n;
    lt = lt2;
    n = 2 * n;
  }
  return n;
}

const repeat = (...parts: (number | readonly number[])[]): number[] =>
  parts.flatMap((p) => (typeof p === 'number' ? [p] : [...p]));

const B3 = [1.4, 2.0, 1.4] as const;
const B7 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4] as const;
const S127 = [7.2, 12.6, 7.2, 23.5, 7.2, 12.6, 7.2, 370.0, 7.2, 12.6, 7.2, 23.5, 7.2, 12.6, 7.2];

/** Grimmer (2024), Table 1: the straightforward patterns. */
export const LONG_STEP_PATTERNS: Readonly<Record<string, readonly number[]>> = {
  '2': [2.9, 1.5],
  '3': [1.5, 4.9, 1.5],
  '7': [1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5],
  '15': repeat(B3, 4.5, B3, 29.7, B3, 4.5, B3),
  '31': repeat(B7, 8.2, B7, 72.3, B7, 8.2, B7),
  '63': repeat(B7, 7.2, B7, 14.2, B7, 7.2, B7, 164.0, B7, 7.2, B7, 14.2, B7, 7.2, B7),
  '127': repeat(...S127.flatMap((s) => [B7, s] as const), B7),
};
const PATTERN_KEYS = ['2', '3', '7', '15', '31', '63', '127'];

/** Grimmer (2024), Table 1: the proved rate f(x_T) − f* ≤ L D²/(c·T) + O(1/T²) has this c. */
export const LONG_STEP_RATES: Readonly<Record<string, number>> = {
  '2': 2.2,
  '3': 2.6333333333333333,
  '7': 3.1999999,
  '15': 3.8599999,
  '31': 4.6032258,
  '63': 5.2253968,
  '127': 5.8346303,
};

/** θ₀, ..., θ_N of OGM1 (Kim & Fessler 2016, §7.1). */
export function ogmThetas(N: number): Vector {
  if (N < 1) throw new ValueError(`N must be ≥ 1, got ${N}`);
  const th = new Array<number>(N + 1);
  th[0] = 1.0;
  for (let i = 0; i < N; i++) {
    const c = i === N - 1 ? 8.0 : 4.0;
    th[i + 1] = 0.5 * (1.0 + Math.sqrt(1.0 + c * th[i] ** 2));
  }
  return th;
}

// ---------------------------------------------------------------------------------------
// Shared machinery
// ---------------------------------------------------------------------------------------

/** `np.max(np.abs(v))` (NaN propagates). */
function maxAbs(v: readonly number[]): number {
  let m = 0;
  for (const t of v) {
    const a = Math.abs(t);
    if (Number.isNaN(a)) return NaN;
    if (a > m) m = a;
  }
  return m;
}

function dotv(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** ‖v‖₂ with scaling (as LAPACK's dnrm2): m·√(uᵀu), u = v/m; 0 only when v = 0. */
export function scaledNorm(v: readonly number[]): number {
  const m = v.length ? maxAbs(v) : 0.0;
  if (m === 0.0 || !Number.isFinite(m)) return m;
  const u = v.map((t) => t / m);
  return m * Math.sqrt(dotv(u, u));
}

const allFinite = (v: readonly number[]) => v.every((t) => Number.isFinite(t));
const arraysEqual = (a: readonly number[], b: readonly number[]) =>
  a.length === b.length && a.every((t, i) => t === b[i]);

function checkCommon(gtol: number, maxIter: unknown): number {
  if (!(Number.isFinite(gtol) && gtol >= 0.0))
    throw new ValueError(`gtol must be a finite number ≥ 0, got ${formatRepr(gtol)}`);
  if (
    typeof maxIter !== 'number' ||
    !Number.isFinite(maxIter) ||
    maxIter !== Math.floor(maxIter) ||
    maxIter < 1
  )
    throw new ValueError(`max_iter must be a positive integer, got ${String(maxIter)}`);
  return maxIter;
}

/** Counted f, ∇f and ∇²f of the problem, and its tags / extra (for auto L and μ). */
class Oracle {
  readonly f: Counted<number>;
  readonly grad: Counted<Vector>;
  readonly hess: Counted<Matrix>;
  readonly id: string;
  readonly tags: readonly string[];
  readonly extra: Readonly<Record<string, unknown>>;
  readonly hasHess: boolean;
  readonly x0: Vector;
  private eigs: { x: Vector; lam: Vector } | null = null;

  constructor(problem: SmoothProblem, x0: unknown) {
    const r = resolveSmooth(problem, x0);
    this.f = r.f;
    this.grad = r.grad;
    this.hess = r.hess;
    this.x0 = r.x;
    const p = typeof problem === 'function' ? null : (problem as Problem<unknown>);
    this.id = p?.id ?? 'custom';
    this.tags = p?.tags ?? [];
    this.extra = p?.extra ?? {};
    this.hasHess = typeof p?.hess === 'function';
  }

  value(x: Vector): number {
    return this.f(x);
  }

  gradient(x: Vector): Vector {
    return this.grad(x);
  }

  /** Ascending eigenvalues of the symmetrized ∇²f(x); cached so auto L and μ cost one ∇²f. */
  hessianEigs(x: Vector): Vector {
    if (this.eigs && arraysEqual(this.eigs.x, x)) return this.eigs.lam;
    const H = this.hess(x);
    const S = H.map((row, i) => row.map((v, j) => 0.5 * (v + H[j][i])));
    const lam = eigh(S).values;
    this.eigs = { x: x.slice(), lam };
    return lam;
  }

  result(
    method: string,
    trace: Step[],
    converged: boolean,
    message: string,
    extra: Record<string, unknown>,
  ): Result {
    const last = trace[trace.length - 1];
    return {
      method,
      x: (last.x as Vector).slice(),
      fun: last.fun,
      converged,
      message,
      nIter: last.k,
      nFev: this.f.n,
      nGev: this.grad.n,
      nHev: this.hess.n,
      trace,
      extra,
    };
  }
}

/** Resolve L (`which = −1`, the largest eigenvalue) or μ (`which = 0`, the smallest). */
function constant(
  name: string,
  given: number,
  oracle: Oracle,
  x0: Vector,
  which: -1 | 0,
): [number, string] {
  if (!(Number.isFinite(given) && given >= 0.0))
    throw new ValueError(`${name} must be finite and ≥ 0 (0 = auto), got ${formatRepr(given)}`);
  if (given > 0.0) return [given, 'given'];
  const stated = oracle.extra[name];
  if (typeof stated === 'number' && Number.isFinite(stated) && stated > 0.0)
    return [stated, `problem.extra["${name}"]`];
  if (oracle.tags.includes('quadratic') && oracle.hasHess) {
    // λ(∇²f(x₀)) is a global constant only because f is quadratic (constant Hessian).
    const eigs = oracle.hessianEigs(x0);
    const lam = which === 0 ? eigs[0] : eigs[eigs.length - 1];
    const floor = which === 0 ? EIG_RESIDUE * eigs.length * EPS * maxAbs(eigs) : 0.0;
    if (lam > floor) return [lam, 'eigenvalue of the (constant) Hessian'];
    if (which === 0 && lam > 0.0)
      throw new ValueError(
        `${oracle.id}: give ${name} > 0 explicitly; λ_min(∇²f) = ${g3(lam)} is at the ` +
          `rounding level of eigvalsh (≤ ${EIG_RESIDUE}·n·ε·λ_max = ${g3(floor)}), so f is ` +
          'not certifiably strongly convex',
      );
  }
  throw new ValueError(
    `${oracle.id}: give ${name} > 0 explicitly; the certificates need a global constant ` +
      'and the problem states none',
  );
}

/** [converged, message] when the run must stop at iterate x_k, else null. */
function status(
  k: number,
  fx: number,
  g: Vector,
  f0: number,
  gtol: number,
): [boolean, string] | null {
  if (!(Number.isFinite(fx) && allFinite(g))) {
    if (k === 0) return [false, 'f(x0) or ∇f(x0) is not finite'];
    return [false, `diverged: f(x) or ∇f(x) is not finite at iteration ${k}`];
  }
  const riseMax = F_DIVERGE * Math.max(1.0, Math.abs(f0));
  if (fx - f0 > riseMax)
    return [
      false,
      `diverged: f(x) − f(x0) = ${g3(fx - f0)} > 1e+12·max(1, |f(x0)|) ` +
        `= ${g3(riseMax)} at iteration ${k} (L too small?)`,
    ];
  const gn = scaledNorm(g);
  if (gn <= gtol) {
    const where = k === 0 ? 'at the start point' : `after ${k} iterations`;
    return [true, `‖∇f(x)‖ = ${g3(gn)} ≤ gtol ${where}`];
  }
  return null;
}

const maxIterMessage = (maxIter: number, g: Vector) =>
  `reached max_iter=${maxIter} (‖∇f(x)‖ = ${g3(scaledNorm(g))} > gtol)`;

interface Marks {
  checkpoint: boolean;
  bound_f: number | null;
  [key: string]: unknown;
}

function makeStep(
  k: number,
  x: Vector,
  fx: number,
  g: Vector,
  alpha: number | null,
  p: Vector | null,
  h: number | null,
  marks: Marks,
  more: Record<string, unknown> = {},
): Step {
  return {
    k,
    x: x.slice(),
    fun: fx,
    gradNorm: scaledNorm(g),
    stepSize: alpha,
    info: {
      grad: g.slice(),
      direction: p === null ? null : p.slice(),
      alpha,
      h,
      ...marks,
      ...more,
    },
  };
}

/** Run (GD) with normalized steps `step(t)`; `marks(k)` gives the checkpoint keys. */
function scheduleGd(
  method: string,
  oracle: Oracle,
  xStart: Vector,
  L: number,
  step: (t: number) => number,
  marks: (k: number) => Marks,
  gtol: number,
  maxIter: number,
  extra: Record<string, unknown>,
): Result {
  let x = xStart;
  let fx = oracle.value(x);
  let g = oracle.gradient(x);
  const f0 = fx;
  const trace: Step[] = [makeStep(0, x, fx, g, null, null, null, marks(0))];
  let stop = status(0, fx, g, f0, gtol);
  if (stop !== null) return oracle.result(method, trace, stop[0], stop[1], extra);
  for (let k = 1; k <= maxIter; k++) {
    const h = step(k - 1);
    const alpha = h / L;
    const p = g.map((t) => -t);
    x = x.map((t, i) => t + alpha * p[i]);
    fx = oracle.value(x);
    g = oracle.gradient(x);
    trace.push(makeStep(k, x, fx, g, alpha, p, h, marks(k)));
    stop = status(k, fx, g, f0, gtol);
    if (stop !== null) return oracle.result(method, trace, stop[0], stop[1], extra);
  }
  return oracle.result(method, trace, false, maxIterMessage(maxIter, g), extra);
}

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

const P_L = param.float('L', 0.0, {
  min: 0.0,
  max: 1e6,
  help: 'Global Lipschitz constant of ∇f (0 = auto: problem.extra["L"] or the Hessian of a quadratic).',
  label: 'Lipschitz constant',
  tex: 'L',
});
const P_GTOL = param.float('gtol', 1e-6, {
  min: 1e-14,
  max: 1e-2,
  log: true,
  help: 'Stop when ‖∇f(x_k)‖₂ ≤ gtol (0 = never).',
  label: 'Gradient tolerance',
  tex: '\\|\\nabla f\\|_2 \\le',
});
const pMaxIter = (def: number, help = 'Iteration limit.') =>
  param.int('max_iter', def, {
    min: 1,
    max: 100_000,
    help,
    label: 'Max iterations',
  });

type Opts = RunOptions & Params;
const num = (v: unknown, def: number): number => (typeof v === 'number' ? v : def);

// ---------------------------------------------------------------------------------------
// Certified step-size schedules
// ---------------------------------------------------------------------------------------

export const silverGd: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 1023);
  const oracle = new Oracle(problem, o.x0);
  const x = oracle.x0;
  const [L, src] = constant('L', num(o.L, 0.0), oracle, x, -1);
  const marks = (k: number): Marks => {
    const j = bitLength(k + 1) - 1;
    const cp = k >= 1 && k + 1 === 2 ** j;
    return { checkpoint: cp, bound_f: cp ? silverRate(j) : null };
  };
  return scheduleGd('silver_gd', oracle, x, L, silverStep, marks, gtol, maxIter, {
    L,
    L_source: src,
  });
};

export const silverGdStronglyConvex: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 1024);
  const horizon = o.horizon ?? 0;
  if (horizon !== 0) checkHorizon(horizon);
  const oracle = new Oracle(problem, o.x0);
  const x = oracle.x0;
  const [L, srcL] = constant('L', num(o.L, 0.0), oracle, x, -1);
  const [mu, srcMu] = constant('mu', num(o.mu, 0.0), oracle, x, 0);
  if (mu > L) throw new ValueError(`μ = ${formatG(mu)} exceeds L = ${formatG(L)}`);
  const kappa = L / mu;
  const n = horizon === 0 ? silverScAutoHorizon(kappa) : checkHorizon(horizon);
  const [h, tau] = silverScSchedule(kappa, n);
  const marks = (k: number): Marks => {
    const cp = k >= 1 && k % n === 0;
    const bd = cp ? tau ** Math.floor(k / n) : null;
    return { checkpoint: cp, bound_f: bd === null ? null : 0.5 * bd, bound_dist: bd };
  };
  const extra = {
    L,
    L_source: srcL,
    mu,
    mu_source: srcMu,
    kappa,
    horizon: n,
    tau_horizon: tau,
  };
  return scheduleGd(
    'silver_gd_strongly_convex',
    oracle,
    x,
    L,
    (t) => h[t % n],
    marks,
    gtol,
    maxIter,
    extra,
  );
};

export const longStepGd: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-6);
  const maxIter = checkCommon(gtol, o.max_iter ?? 1000);
  const pattern = String(o.pattern ?? '7');
  if (!PATTERN_KEYS.includes(pattern))
    throw new ValueError(
      `pattern must be one of (${PATTERN_KEYS.map((p) => `'${p}'`).join(', ')}), got '${pattern}'`,
    );
  const oracle = new Oracle(problem, o.x0);
  const x = oracle.x0;
  const [L, src] = constant('L', num(o.L, 0.0), oracle, x, -1);
  const h = LONG_STEP_PATTERNS[pattern];
  const t = h.length;
  const marks = (k: number): Marks => ({ checkpoint: k >= 1 && k % t === 0, bound_f: null });
  const extra = {
    L,
    L_source: src,
    pattern: [...h],
    avg_h: h.reduce((a, v) => a + v, 0) / t,
    rate_c: LONG_STEP_RATES[pattern],
  };
  return scheduleGd('long_step_gd', oracle, x, L, (i) => h[i % t], marks, gtol, maxIter, extra);
};

export const ogm: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const gtol = num(o.gtol, 1e-6);
  const N = checkCommon(gtol, o.max_iter ?? 1000);
  const oracle = new Oracle(problem, o.x0);
  let x = oracle.x0;
  const [L, src] = constant('L', num(o.L, 0.0), oracle, x, -1);
  const theta = ogmThetas(N);
  const boundN = 0.5 / theta[N] ** 2;
  let y = x.slice();
  let fx = oracle.value(x);
  let g = oracle.gradient(x);
  const f0 = fx;
  const extra = { L, L_source: src, theta_N: theta[N] };
  const trace: Step[] = [
    makeStep(
      0,
      x,
      fx,
      g,
      null,
      null,
      null,
      { checkpoint: false, bound_f: null },
      {
        y: y.slice(),
        theta: 1.0,
      },
    ),
  ];
  let stop = status(0, fx, g, f0, gtol);
  if (stop !== null) return oracle.result('ogm', trace, stop[0], stop[1], extra);
  const alpha = 1.0 / L;
  for (let i = 0; i < N; i++) {
    const yNew = x.map((t, j) => t - alpha * g[j]);
    const t0 = theta[i];
    const t1 = theta[i + 1];
    const c1 = (t0 - 1.0) / t1;
    const c2 = t0 / t1;
    const xNew = yNew.map((t, j) => t + c1 * (t - y[j]) + c2 * (t - x[j]));
    const p = xNew.map((t, j) => (t - x[j]) / alpha);
    x = xNew;
    y = yNew;
    fx = oracle.value(x);
    g = oracle.gradient(x);
    const k = i + 1;
    const last = k === N;
    trace.push(
      makeStep(
        k,
        x,
        fx,
        g,
        alpha,
        p,
        1.0,
        { checkpoint: last, bound_f: last ? boundN : null },
        {
          y: y.slice(),
          theta: t1,
        },
      ),
    );
    stop = status(k, fx, g, f0, gtol);
    if (stop !== null) return oracle.result('ogm', trace, stop[0], stop[1], extra);
  }
  return oracle.result('ogm', trace, false, maxIterMessage(N, g), extra);
};

// ---------------------------------------------------------------------------------------
// FISTA / accelerated gradient with adaptive restart
// ---------------------------------------------------------------------------------------

/** Beck–Teboulle backtracking (§3, eq. 2.9 with g ≡ 0) from the estimate L. */
function backtrack(
  oracle: Oracle,
  y: Vector,
  gy: Vector,
  fy: number,
  L0: number,
  eta: number,
): [boolean, Vector, number, number, number[][]] {
  const trials: number[][] = [];
  let L = L0;
  let p = y;
  let fp = NaN;
  for (let i = 0; i < MAX_BACKTRACK; i++) {
    const Lc = L;
    p = y.map((t, j) => t - gy[j] / Lc);
    fp = oracle.value(p);
    trials.push([L, fp]);
    const d = p.map((t, j) => t - y[j]);
    const q = fy + dotv(d, gy) + 0.5 * L * dotv(d, d);
    if (Number.isFinite(fp) && fp <= q + BT_RTOL * Math.max(Math.abs(fy), Math.abs(fp)))
      return [true, p, fp, L, trials];
    L *= eta;
  }
  return [false, p, fp, L, trials];
}

export const fista: MethodFn<SmoothProblem> = (problem, o: Opts) => {
  const restart = String(o.restart ?? 'gradient');
  const lr = num(o.lr, 1.0);
  const backtracking = o.backtracking === undefined ? true : Boolean(o.backtracking);
  const eta = num(o.eta, 2.0);
  const gtol = num(o.gtol, 1e-6);
  const maxIt = checkCommon(gtol, o.max_iter ?? 5000);
  if (!(gtol > 0.0))
    throw new ValueError(`gtol must be a finite number > 0, got ${formatRepr(gtol)}`);
  if (!(RESTARTS as readonly string[]).includes(restart))
    throw new ValueError(
      `restart must be one of ('none', 'function', 'gradient'), got '${restart}'`,
    );
  if (!(Number.isFinite(lr) && lr > 0.0))
    throw new ValueError(`lr must be a finite number > 0, got ${formatRepr(lr)}`);
  if (!(Number.isFinite(eta) && eta > 1.0))
    throw new ValueError(`eta must be a finite number > 1, got ${formatRepr(eta)}`);
  const oracle = new Oracle(problem, o.x0);
  let x = oracle.x0; // x_{k−1} at the top of iteration k
  const restarts: number[] = [];
  const trace: Step[] = [];
  const done = (converged: boolean, message: string): Result =>
    oracle.result('fista', trace, converged, message, {
      n_restart: restarts.length,
      restarts: [...restarts],
    });

  let y = x.slice(); // y_k (y₁ = x₀)
  let t = 1.0; // t_k (t₁ = 1)
  let gnLast = Infinity;
  let beta = 0.0; // the coefficient that formed y_k
  let L = 1.0 / lr;
  let fx = oracle.value(x);
  const f0 = fx;
  trace.push({
    k: 0,
    x: x.slice(),
    fun: fx,
    gradNorm: null,
    stepSize: null,
    info: {
      direction: null,
      alpha: null,
      y: null,
      grad_y: null,
      grad_norm_y: null,
      beta: null,
      t,
      restarted: false,
      L,
      trials: [],
    },
  });
  if (!Number.isFinite(fx)) return done(false, 'f(x0) is not finite');

  for (let k = 1; k <= maxIt; k++) {
    const gy = oracle.gradient(y);
    if (!allFinite(gy)) return done(false, `diverged: ∇f is not finite at y_k (iteration ${k})`);
    let p: Vector;
    let fp: number;
    let trials: number[][];
    if (backtracking) {
      // f(y_k) is already known when y_k = x_{k−1} (no momentum): reuse it.
      const fy = arraysEqual(y, x) ? fx : oracle.value(y);
      let found: boolean;
      [found, p, fp, L, trials] = backtrack(oracle, y, gy, fy, L, eta);
      if (!found)
        return done(
          false,
          `backtracking failed: no L̄ = ηⁱL_(k−1) with f(p) ≤ Q_L(p, y) in ` +
            `${MAX_BACKTRACK} trials (iteration ${k}, last L̄ = ${e3(trials[trials.length - 1][0])})`,
        );
    } else {
      trials = [];
      const Lc = L;
      p = y.map((v, j) => v - gy[j] / Lc);
      fp = oracle.value(p);
    }
    let restarted = false;
    if (restart === 'function') restarted = fp > fx;
    else if (restart === 'gradient')
      restarted =
        dotv(
          gy,
          p.map((v, j) => v - x[j]),
        ) > 0.0;
    let tNext: number;
    let betaNext: number;
    let yNext: Vector;
    if (restarted) {
      tNext = 1.0;
      betaNext = 0.0;
      yNext = p.slice();
      restarts.push(k);
    } else {
      tNext = 0.5 * (1.0 + Math.sqrt(1.0 + 4.0 * t * t)); // (4.2)
      betaNext = (t - 1.0) / tNext;
      const bn = betaNext;
      yNext = p.map((v, j) => v + bn * (v - x[j])); // (4.3)
    }
    const alpha = 1.0 / L;
    const gn = scaledNorm(gy);
    trace.push({
      k,
      x: p.slice(),
      fun: fp,
      gradNorm: null,
      stepSize: alpha,
      info: {
        direction: p.map((v, j) => (v - x[j]) / alpha),
        alpha,
        y: y.slice(),
        grad_y: gy.slice(),
        grad_norm_y: gn,
        beta,
        t: tNext,
        restarted,
        L,
        trials,
      },
    });
    if (!(Number.isFinite(fp) && allFinite(p)))
      return done(false, `diverged: f(x) is not finite at iteration ${k}`);
    const riseMax = F_DIVERGE * Math.max(1.0, Math.abs(f0));
    if (fp - f0 > riseMax)
      return done(
        false,
        `diverged: f(x) − f(x0) = ${g3(fp - f0)} > 1e+12·max(1, |f(x0)|) ` +
          `= ${g3(riseMax)} at iteration ${k} (step 1/L too long?)`,
      );
    if (gn <= gtol) return done(true, `‖∇f(y_k)‖ = ${g3(gn)} ≤ gtol after ${k} iterations`);
    if (arraysEqual(p, x) && arraysEqual(yNext, x))
      return done(
        false,
        `stalled: x_k = x_(k−1) in floating point at iteration ${k} ` +
          `(step ‖∇f(y_k)‖/L_k = ${g3(gn / L)}; gtol below the attainable precision?)`,
      );
    x = p;
    fx = fp;
    y = yNext;
    t = tNext;
    beta = betaNext;
    gnLast = gn;
  }
  return done(false, `reached max_iter=${maxIt} (‖∇f(y_k)‖ = ${g3(gnLast)} > gtol)`);
};

// ---------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------

const STUDY_SCHEDULES = 'numopt study research/certified-stepsize-schedules';

const SCHEDULE_QUANTITIES: MethodDoc['quantities'] = [
  { tex: 'h_k', key: 'info.h' },
  { tex: '\\alpha_k = h_k/L', key: 'stepSize' },
  { tex: '\\|\\nabla f(x_k)\\|', key: 'gradNorm' },
];

registerMethod(
  {
    id: 'silver_gd',
    family: 'unconstrained',
    name: 'Silver step-size gradient descent',
    params: [P_L, P_GTOL, pMaxIter(1023, 'Iterations; the certificate holds at k = 2ʲ − 1.')],
    needs: ['f', 'grad'],
    order: 'sublinear: f − f⋆ ≤ r_k L‖x₀ − x⋆‖² ≈ L‖x₀ − x⋆‖²/(2n^{1.2716}) at n = 2ᵏ − 1',
    summary:
      'Gradient descent with a fixed, fractal schedule of long and short steps that is ' +
      'provably faster than any constant step on convex functions.',
    references: [
      'Altschuler & Parrilo (2024), Math. Program., eq. 2.1 (schedule), Theorem 1.1 and ' +
        'eq. 1.4 (bound)',
      STUDY_SCHEDULES,
    ],
  },
  silverGd,
  {
    rule: 'x_{k+1} = x_k - \\frac{h_k}{L}\\nabla f(x_k),\\quad h_k = 1 + \\rho^{\\nu(k+1)-1},\\ \\rho = 1 + \\sqrt2',
    intuition:
      'Mostly short steps, with a long step of growing length at every power of two: the long ' +
      'steps overshoot on purpose and the short ones repair the damage, faster than any constant step.',
    quantities: SCHEDULE_QUANTITIES,
  },
);

registerMethod(
  {
    id: 'silver_gd_strongly_convex',
    family: 'unconstrained',
    name: 'Silver step-size gradient descent (strongly convex)',
    params: [
      P_L,
      param.float('mu', 0.0, {
        min: 0.0,
        max: 1e6,
        help: 'Strong-convexity constant μ ≤ L (0 = auto: problem.extra["mu"] or the Hessian of a quadratic).',
        label: 'Strong convexity',
        tex: '\\mu',
      }),
      param.int('horizon', 0, {
        min: 0,
        max: 4096,
        help: 'Block length n to repeat: a power of 2 (0 = auto, near saturation of the rate).',
        label: 'Block length',
        tex: 'n',
      }),
      P_GTOL,
      pMaxIter(1024),
    ],
    needs: ['f', 'grad'],
    order: 'linear: ‖x − x⋆‖² shrinks by τ_n per block of n steps, O(κ^{0.7864} log(1/ε)) steps',
    summary:
      'Gradient descent with a κ-aware silver schedule: certified linear convergence ' +
      'faster than any constant step on strongly convex functions.',
    references: [
      'Altschuler & Parrilo (2025), J. ACM 72(2), §3, eqs. 3.1–3.9 (schedule) and ' +
        'Theorem 1.1 (rate)',
      STUDY_SCHEDULES,
    ],
  },
  silverGdStronglyConvex,
  {
    rule: 'x_{k+1} = x_k - \\frac{h_{k \\bmod n}}{L}\\nabla f(x_k),\\quad h^{(n)} = [\\tilde h^{(n/2)}, a_n, \\tilde h^{(n/2)}, b_n]',
    intuition:
      'The silver schedule tuned to κ = L/μ: one block of n steps contracts ‖x − x*‖² by a ' +
      'certified factor τ_n, and the block repeats.',
    quantities: SCHEDULE_QUANTITIES,
  },
);

registerMethod(
  {
    id: 'long_step_gd',
    family: 'unconstrained',
    name: 'Long-step gradient descent',
    params: [
      P_L,
      param.choice('pattern', '7', [...PATTERN_KEYS], {
        help: "Length t of Grimmer's straightforward step pattern (Table 1), cycled.",
        label: 'Pattern length',
        tex: 't',
      }),
      P_GTOL,
      pMaxIter(1000),
    ],
    needs: ['f', 'grad'],
    order: 'sublinear: f − f⋆ ≤ L D²/(avg(h)·T) + O(1/T²), avg(h) up to 5.83 (pattern 127)',
    summary:
      'Gradient descent that cycles a short step pattern with one very long step; its ' +
      'proved O(1/T) constant grows with the average step.',
    references: [
      'Grimmer (2024), SIAM J. Optim. 34(3), eq. 1.3 (method), Table 1 (patterns), ' +
        'Theorem 2.1 and eq. 1.4 (rate)',
      STUDY_SCHEDULES,
    ],
  },
  longStepGd,
  {
    rule: 'x_{k+1} = x_k - \\frac{h_{k \\bmod t}}{L}\\nabla f(x_k)',
    intuition:
      'A fixed pattern of steps, most below the classical limit 2/L and one far above it, ' +
      'repeated: on average the pattern moves further than any safe constant step.',
    quantities: SCHEDULE_QUANTITIES,
  },
);

registerMethod(
  {
    id: 'ogm',
    family: 'unconstrained',
    name: 'Optimized gradient method (OGM)',
    params: [P_L, P_GTOL, pMaxIter(1000, 'The horizon N (the last step uses a modified θ_N).')],
    needs: ['f', 'grad'],
    order: 'sublinear: f(x_N) − f⋆ ≤ L‖x₀ − x⋆‖²/(2θ_N²) ≤ L‖x₀ − x⋆‖²/(N + 1)², tight',
    summary:
      "An accelerated gradient method whose worst-case bound is half of Nesterov's: the " +
      'best possible for a fixed number of gradient steps.',
    references: [
      'Kim & Fessler (2016), Math. Program. 159, §7.1 (Algorithm OGM1), Theorem 2 and ' +
        'eq. 6.17 (bound), Theorem 3 (tightness)',
      STUDY_SCHEDULES,
    ],
  },
  ogm,
  {
    rule: 'y_{k+1} = x_k - \\tfrac1L\\nabla f(x_k),\\quad x_{k+1} = y_{k+1} + \\tfrac{\\theta_k - 1}{\\theta_{k+1}}(y_{k+1} - y_k) + \\tfrac{\\theta_k}{\\theta_{k+1}}(y_{k+1} - x_k)',
    intuition:
      'A gradient step, then two momentum terms: Nesterov’s, and a second one along the ' +
      'gradient step itself. Its worst case is the best any fixed-step method can achieve.',
    quantities: [
      { tex: '\\theta_k', key: 'info.theta' },
      { tex: '\\|\\nabla f(x_k)\\|', key: 'gradNorm' },
    ],
  },
);

registerMethod(
  {
    id: 'fista',
    family: 'unconstrained',
    name: 'Accelerated gradient with adaptive restart (FISTA)',
    params: [
      param.choice('restart', 'gradient', [...RESTARTS], {
        help:
          "Adaptive restart test (O'Donoghue & Candès 2015, §3.2): none = plain FISTA; " +
          'function: f(x_k) > f(x_{k−1}); gradient: ∇f(y_k)ᵀ(x_k − x_{k−1}) > 0.',
        label: 'Restart test',
      }),
      param.float('lr', 1.0, {
        min: 1e-8,
        max: 1e4,
        log: true,
        help:
          'Step 1/L. With backtracking, the first estimate L₀ = 1/lr (it only grows, so ' +
          'choose L₀ ≤ L).',
        label: 'Step size',
        tex: '1/L_0',
      }),
      param.bool('backtracking', true, {
        help: 'Beck–Teboulle backtracking: multiply L by η until f(p) ≤ f(y) − ‖∇f(y)‖²/(2L).',
        label: 'Backtracking',
      }),
      param.float('eta', 2.0, {
        min: 1.1,
        max: 10.0,
        help: 'Backtracking factor η > 1.',
        label: 'Backtracking factor',
        tex: '\\eta',
      }),
      param.float('gtol', 1e-6, {
        min: 1e-14,
        max: 1e-2,
        log: true,
        help: 'Stop when ‖∇f(y_k)‖₂ ≤ gtol (the gradient at the extrapolated point).',
        label: 'Gradient tolerance',
        tex: '\\|\\nabla f(y_k)\\|_2 \\le',
      }),
      param.int('max_iter', 5000, {
        min: 1,
        max: 1_000_000,
        help: 'Iteration limit.',
        label: 'Max iterations',
      }),
    ],
    needs: ['f', 'grad'],
    order:
      'sublinear O(1/k²) on convex f; with restart, linear O(√κ log(1/ε)) on strongly ' +
      'convex quadratics without knowing μ',
    summary:
      "Nesterov's accelerated gradient with growing momentum, reset whenever the " +
      'momentum starts to work against the descent.',
    references: [
      'Beck & Teboulle (2009), SIAM J. Imaging Sci. 2(1), §4, eqs. 4.1–4.3 (FISTA), ' +
        'eq. 2.9 and §3 (backtracking), Theorem 4.4 (rate)',
      "O'Donoghue & Candès (2015), Found. Comput. Math. 15, §3.2 (restart tests), " +
        'Algorithm 3 (reset), §4.5 (restart interval)',
      'numopt study research/restarted-accelerated-gradient',
    ],
  },
  fista,
  {
    rule: 'x_k = y_k - \\tfrac{1}{L_k}\\nabla f(y_k),\\quad y_{k+1} = x_k + \\tfrac{t_k - 1}{t_{k+1}}(x_k - x_{k-1})',
    intuition:
      'Gradient steps from an extrapolated point whose momentum grows each iteration; when ' +
      'the momentum starts to push uphill, it is reset to zero.',
    quantities: [
      { tex: 't_{k+1}', key: 'info.t' },
      { tex: 'L_k', key: 'info.L' },
      { tex: '\\|\\nabla f(y_k)\\|', key: 'info.grad_norm_y' },
    ],
  },
);
