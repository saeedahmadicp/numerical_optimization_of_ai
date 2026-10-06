/**
 * Stochastic gradient methods for finite sums f(w) = (1/n) Σᵢ fᵢ(w) — TS port of
 * `numopt.stochastic.methods` (src/numopt/stochastic/methods.py).
 *
 * Every method runs on a `FiniteSumProblem` (src/problems/stochastic.ts) and shares one driver,
 * so the methods differ only in their update rule. Conventions (identical to Python):
 *
 * - **Sampling.** b = min(batch_size, n). Epoch e = 1, 2, … first draws `perm = rng.permutation(n)`
 *   (one Fisher–Yates shuffle per epoch, from one `Rng(seed)` created once per run), then performs
 *   U = ⌈n/b⌉ updates with the mini-batches perm[jb : (j+1)b]. No other random numbers are drawn.
 * - **Learning rate.** Update t = 0 … T−1 (T = epochs·U) uses η_t = `learningRate(...)`:
 *   constant η₀; step η₀·½^⌊⌊τ⌋/s⌋ with s = max(1, ⌊epochs/4⌋); inv_sqrt η₀/√(1 + τ); cosine
 *   η₀·½(1 + cos(πt/T)), with τ = t/U.
 * - **Trace.** Step k is the iterate after k updates; it is recorded when k is a multiple of
 *   `every` = max(record_every, ⌈T/398⌉, 1) and at the final iterate (≤ 400 steps). `fun` is the
 *   FULL loss f(w_k), `gradNorm` = ‖∇f(w_k)‖₂, `stepSize` the η of the update that produced w_k.
 * - **Stopping test.** ‖∇f(w)‖₂ ≤ gtol, tested at w₀ and after the last update of every epoch.
 * - **Start test.** f(w₀) or ‖∇f(w₀)‖ not finite → stop at k = 0, message "not finite at x0 …".
 * - **Divergence test.** Non-finite iterate/loss/gradient, or ‖∇f‖ > 10⁸(1 + ‖∇f(w₀)‖) where ∇f is
 *   evaluated, or f > 10⁸(1 + |f(w₀)|) at a recorded step → "diverged: …; try a smaller lr".
 * - **Counts.** nGev = component gradients ∇fᵢ the update rule evaluates (a full gradient counts
 *   n); nFev = full-loss evaluations for the trace; monitoring gradients go to
 *   `extra.monitor_grad_evals`.
 *
 * Info keys (snake_case, as in Python): epoch, batch (null when b > 32 and at k = 0), stoch_grad,
 * full_grad, lr, update, ifo; per method velocity, lookahead (Nesterov), accum / sq_avg / m / v /
 * m_hat / v_hat / scaled_lr (adaptive), snapshot / snapshot_grad (SVRG), table_mean (SAGA),
 * seen (SAG).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { Rng } from '../../core/rng';
import type {
  MethodFn,
  ParamSpec,
  Params,
  Result,
  RunOptions,
  Step,
  Vector,
} from '../../core/types';
import { isFiniteSumProblem, type FiniteSumProblem } from '../../problems/stochastic';

export const SCHEDULES = ['constant', 'step', 'inv_sqrt', 'cosine'] as const;
export type Schedule = (typeof SCHEDULES)[number];
/** The `step` schedule multiplies η by STEP_GAMMA every ⌊epochs/STEP_DROPS⌋ epochs. */
export const STEP_GAMMA = 0.5;
export const STEP_DROPS = 4;
/** Largest mini-batch whose indices are written to `info.batch`. */
export const BATCH_INFO_MAX = 32;
/** The recording interval is raised as needed so the trace has ≤ TRACE_MAX steps. */
export const TRACE_MAX = 400;
/** Divergence test (b): ‖∇f‖ or f above BLOWUP·(1 + its value at w₀) stops the run. */
export const BLOWUP = 1e8;

/** η_t of update t (0-based) for a run of `totalUpdates` = epochs·U updates. */
export function learningRate(
  schedule: string,
  lr: number,
  t: number,
  updatesPerEpoch: number,
  totalUpdates: number,
): number {
  const U = updatesPerEpoch,
    T = totalUpdates;
  if (schedule === 'constant') return lr;
  if (schedule === 'step') {
    const s = Math.max(1, Math.floor(Math.floor(T / U) / STEP_DROPS));
    return lr * STEP_GAMMA ** Math.floor(Math.floor(t / U) / s);
  }
  if (schedule === 'inv_sqrt') return lr / Math.sqrt(1.0 + t / U);
  if (schedule === 'cosine') return lr * 0.5 * (1.0 + Math.cos((Math.PI * t) / T));
  throw new Error(`unknown lr_schedule '${schedule}'; expected one of ${SCHEDULES.join(', ')}`);
}

// ── Python number formatting for messages ──────────────────────────────────────────────

/** Python's `format(x, '.{p}g')`: 3 significant digits, exponent form outside [1e-4, 10^p). */
export function pyG(x: number, p = 3): string {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const [mant, expStr] = x.toExponential(p - 1).split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= p) {
    const m = mant.includes('.') ? mant.replace(/0+$/, '').replace(/\.$/, '') : mant;
    return `${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  const fixed = x.toFixed(Math.max(0, p - 1 - exp));
  return fixed.includes('.') ? fixed.replace(/0+$/, '').replace(/\.$/, '') : fixed;
}

// ── Small vector helpers (Python order: w - eta * g, ...) ─────────────────────────────

const finiteVec = (v: readonly number[]) => v.every(Number.isFinite);
const norm2 = (v: readonly number[]) => {
  // np.linalg.norm of a 1-D float vector: sqrt(dot(v, v)).
  let s = 0.0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
};

// ── Oracle and update rules ───────────────────────────────────────────────────────────

/** Component-gradient access with an exact count of the ∇fᵢ evaluated (nGev). */
class Oracle {
  n = 0;
  readonly p: FiniteSumProblem;
  constructor(p: FiniteSumProblem) {
    this.p = p;
  }
  batch(w: Vector, idx: readonly number[]): Vector {
    this.n += idx.length;
    return this.p.gradBatch(w, idx);
  }
  samples(w: Vector, idx: readonly number[]): Vector[] {
    this.n += idx.length;
    return this.p.gradSamples(w, idx);
  }
  full(w: Vector): Vector {
    this.n += this.p.nSamples;
    return this.p.grad(w);
  }
}

interface Rule {
  /** Called once before the first epoch. */
  start(w: Vector, oracle: Oracle): void;
  /** Called at the start of every epoch, before its first update. */
  beginEpoch(w: Vector, oracle: Oracle): void;
  /** Returns [w_{t+1}, g_t]: the new iterate and the gradient estimate it used. */
  update(w: Vector, batch: number[], eta: number, t: number, oracle: Oracle): [Vector, Vector];
  /** Method-specific Step.info entries for the current state. */
  info(): Record<string, unknown>;
}

const copy = (v: Vector | null): Vector | null => (v === null ? null : [...v]);

class SGDRule implements Rule {
  start() {}
  beginEpoch() {}
  update(w: Vector, batch: number[], eta: number, _t: number, oracle: Oracle): [Vector, Vector] {
    const g = oracle.batch(w, batch);
    return [w.map((wi, i) => wi - eta * g[i]), g];
  }
  info() {
    return {};
  }
}

class MomentumRule implements Rule {
  v: Vector | null = null;
  lookahead: Vector | null = null;
  readonly beta: number;
  readonly nesterov: boolean;
  constructor(beta: number, nesterov: boolean) {
    this.beta = beta;
    this.nesterov = nesterov;
  }
  start(w: Vector) {
    this.v = w.map(() => 0.0);
  }
  beginEpoch() {}
  update(w: Vector, batch: number[], eta: number, _t: number, oracle: Oracle): [Vector, Vector] {
    const v = this.v!;
    let g: Vector;
    if (this.nesterov) {
      this.lookahead = w.map((wi, i) => wi + this.beta * v[i]);
      g = oracle.batch(this.lookahead, batch);
    } else {
      g = oracle.batch(w, batch);
    }
    const vNew = v.map((vi, i) => this.beta * vi - eta * g[i]);
    this.v = vNew;
    return [w.map((wi, i) => wi + vNew[i]), g];
  }
  info() {
    const out: Record<string, unknown> = { velocity: copy(this.v) };
    if (this.nesterov) out.lookahead = copy(this.lookahead);
    return out;
  }
}

type AdaptiveKind = 'adagrad' | 'rmsprop' | 'adam';

/** AdaGrad, RMSProp and Adam: per-coordinate steps η/(√r + ε). */
class AdaptiveRule implements Rule {
  r: Vector | null = null; // Σ g², EMA of g², or Adam's v
  m: Vector | null = null;
  mHat: Vector | null = null;
  vHat: Vector | null = null;
  scaled: Vector | null = null;
  readonly kind: AdaptiveKind;
  readonly eps: number;
  readonly rho: number;
  readonly beta1: number;
  readonly beta2: number;
  constructor(kind: AdaptiveKind, eps: number, rho = 0.9, beta1 = 0.9, beta2 = 0.999) {
    this.kind = kind;
    this.eps = eps;
    this.rho = rho;
    this.beta1 = beta1;
    this.beta2 = beta2;
  }
  start(w: Vector) {
    this.r = w.map(() => 0.0);
    this.m = w.map(() => 0.0);
  }
  beginEpoch() {}
  update(w: Vector, batch: number[], eta: number, t: number, oracle: Oracle): [Vector, Vector] {
    const g = oracle.batch(w, batch);
    const r = this.r!;
    if (this.kind === 'adagrad') {
      this.r = r.map((ri, i) => ri + g[i] * g[i]);
      const scaled = this.r.map((ri) => eta / (Math.sqrt(ri) + this.eps));
      this.scaled = scaled;
      return [w.map((wi, i) => wi - scaled[i] * g[i]), g];
    }
    if (this.kind === 'rmsprop') {
      this.r = r.map((ri, i) => this.rho * ri + (1.0 - this.rho) * (g[i] * g[i]));
      const scaled = this.r.map((ri) => eta / (Math.sqrt(ri) + this.eps));
      this.scaled = scaled;
      return [w.map((wi, i) => wi - scaled[i] * g[i]), g];
    }
    const step = t + 1; // Adam's bias-correction counter starts at 1
    const m = this.m!;
    this.m = m.map((mi, i) => this.beta1 * mi + (1.0 - this.beta1) * g[i]);
    this.r = r.map((ri, i) => this.beta2 * ri + (1.0 - this.beta2) * (g[i] * g[i]));
    const c1 = 1.0 - this.beta1 ** step;
    const c2 = 1.0 - this.beta2 ** step;
    const mHat = this.m.map((mi) => mi / c1);
    this.mHat = mHat;
    this.vHat = this.r.map((ri) => ri / c2);
    const scaled = this.vHat.map((vi) => eta / (Math.sqrt(vi) + this.eps));
    this.scaled = scaled;
    return [w.map((wi, i) => wi - scaled[i] * mHat[i]), g];
  }
  info() {
    if (this.kind === 'adagrad') return { accum: copy(this.r), scaled_lr: copy(this.scaled) };
    if (this.kind === 'rmsprop') return { sq_avg: copy(this.r), scaled_lr: copy(this.scaled) };
    return {
      m: copy(this.m),
      v: copy(this.r),
      m_hat: copy(this.mHat),
      v_hat: copy(this.vHat),
      scaled_lr: copy(this.scaled),
    };
  }
}

class SVRGRule implements Rule {
  snapshot: Vector | null = null;
  mu: Vector | null = null;
  start() {}
  beginEpoch(w: Vector, oracle: Oracle) {
    // Option I of Johnson & Zhang (2013): the next snapshot is the last inner iterate.
    this.snapshot = [...w];
    this.mu = oracle.full(this.snapshot);
  }
  update(w: Vector, batch: number[], eta: number, _t: number, oracle: Oracle): [Vector, Vector] {
    const gw = oracle.batch(w, batch);
    const gs = oracle.batch(this.snapshot!, batch);
    const mu = this.mu!;
    const v = gw.map((gi, i) => gi - gs[i] + mu[i]);
    return [w.map((wi, i) => wi - eta * v[i]), v];
  }
  info() {
    return { snapshot: copy(this.snapshot), snapshot_grad: copy(this.mu) };
  }
}

/** SAG and SAGA: a table φ of the last component gradient seen for every sample. */
class TableRule implements Rule {
  table: Vector[] | null = null; // (n, d)
  total: Vector | null = null; // Σᵢ φᵢ, kept up to date incrementally
  seen: boolean[] | null = null;
  readonly unbiased: boolean; // true: SAGA, false: SAG
  constructor(unbiased: boolean) {
    this.unbiased = unbiased;
  }
  start(w: Vector, oracle: Oracle) {
    const n = oracle.p.nSamples;
    if (this.unbiased) {
      // SAGA (Defazio et al. 2014, §2): φᵢ⁰ = w₀, i.e. the table holds ∇fᵢ(w₀).
      this.table = oracle.samples(
        w,
        Array.from({ length: n }, (_, i) => i),
      );
      this.seen = new Array<boolean>(n).fill(true);
    } else {
      // SAG (Schmidt et al. 2017, Alg. 1): yᵢ = 0 and d = 0.
      this.table = Array.from({ length: n }, () => w.map(() => 0.0));
      this.seen = new Array<boolean>(n).fill(false);
    }
    // table.sum(axis=0): rows added in order.
    let total = w.map(() => 0.0);
    for (const row of this.table) total = total.map((s, j) => s + row[j]);
    this.total = total;
  }
  beginEpoch() {}
  update(w: Vector, batch: number[], eta: number, _t: number, oracle: Oracle): [Vector, Vector] {
    const table = this.table!,
      seen = this.seen!;
    const n = table.length,
      d = w.length,
      b = batch.length;
    const fresh = oracle.samples(w, batch); // rows ∇fᵢ(w), i ∈ B
    const delta = fresh.map((row, r) => row.map((v, j) => v - table[batch[r]][j]));
    // delta.sum(axis=0), rows in order
    let dsum = new Array<number>(d).fill(0.0);
    for (const row of delta) dsum = dsum.map((s, j) => s + row[j]);
    // SAGA: v = (1/b) Σ_{i∈B} (∇fᵢ(w) − φᵢ) + (1/n) Σⱼ φⱼ (the table before the update).
    const total = this.total!;
    const vSaga = dsum.map((s, j) => s / b + total[j] / n);
    batch.forEach((i, r) => {
      table[i] = fresh[r];
      seen[i] = true;
    });
    this.total = total.map((s, j) => s + dsum[j]);
    // SAG: v = d/m with d = Σⱼ yⱼ after the update and m = #samples seen.
    let v = vSaga;
    if (!this.unbiased) {
      const m = seen.reduce((c, s) => c + (s ? 1 : 0), 0);
      v = this.total.map((s) => s / m);
    }
    return [w.map((wi, j) => wi - eta * v[j]), v];
  }
  info() {
    if (this.unbiased) {
      const mean =
        this.total === null || this.table === null
          ? null
          : this.total.map((s) => s / this.table!.length);
      return { table_mean: mean };
    }
    return { seen: this.seen === null ? 0 : this.seen.reduce((c, s) => c + (s ? 1 : 0), 0) };
  }
}

// ── Driver ─────────────────────────────────────────────────────────────────────────────

function checkCount(name: string, value: unknown, minimum: number): number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    value !== Math.floor(value) ||
    value < minimum
  )
    throw new Error(`${name} must be an integer ≥ ${minimum}, got ${String(value)}`);
  return value;
}

function checkUnit(name: string, value: number): void {
  if (!(value >= 0.0 && value < 1.0)) throw new Error(`${name} must lie in [0, 1), got ${value}`);
}

function checkEps(eps: number): void {
  if (!(eps > 0.0)) throw new Error(`eps must be positive, got ${eps}`);
}

type Options = RunOptions & Params;

function drive(method: string, problem: unknown, rule: Rule, o: Options): Result {
  if (!isFiniteSumProblem(problem))
    throw new TypeError(`${method}: problem must be a FiniteSumProblem (kind 'stochastic')`);
  const lrSchedule = String(o.lr_schedule);
  if (!(SCHEDULES as readonly string[]).includes(lrSchedule))
    throw new Error(`unknown lr_schedule '${lrSchedule}'; expected one of ${SCHEDULES.join(', ')}`);
  const lr = Number(o.lr);
  if (!(lr > 0.0 && Number.isFinite(lr)))
    throw new Error(`lr must be positive and finite, got ${lr}`);
  const batchSize = checkCount('batch_size', o.batch_size, 1);
  const epochs = checkCount('epochs', o.epochs, 1);
  const recordEvery = checkCount('record_every', o.record_every ?? 0, 0);
  const gtol = Number(o.gtol);
  if (!(gtol >= 0.0)) throw new Error(`gtol must be ≥ 0, got ${gtol}`);
  const x0 = o.x0 === undefined ? problem.x0 : o.x0;
  let w: Vector = Array.isArray(x0) ? x0.map(Number) : [Number(x0)];
  if (w.length !== problem.dim || !finiteVec(w))
    throw new Error(
      `${problem.id}: x0 must be ${problem.dim} finite numbers, got [${w.join(', ')}]`,
    );
  const seed = o.seed ?? 0;

  const p = problem;
  const n = p.nSamples;
  const b = Math.min(batchSize, n);
  const U = Math.ceil(n / b); // ⌈n/b⌉ updates per epoch
  const T = epochs * U;
  // 1 (k = 0) + ⌊T/every⌋ ≤ TRACE_MAX − 2 multiples + 1 (the final step) ≤ TRACE_MAX steps.
  const every = Math.max(recordEvery, Math.ceil(T / (TRACE_MAX - 2)), 1);
  const rng = new Rng(seed);
  const oracle = new Oracle(p);
  let nFev = 0;
  let nMon = 0; // full gradients evaluated only for monitoring
  const trace: Step[] = [];

  const stop = (
    converged: boolean,
    message: string,
    k: number,
    fw: number,
    epochsRun: number,
  ): Result => ({
    method,
    x: [...w],
    fun: fw,
    converged,
    message,
    nIter: k,
    nFev,
    nGev: oracle.n,
    nHev: 0,
    trace,
    extra: {
      ifo: oracle.n,
      monitor_grad_evals: nMon,
      epochs_run: epochsRun,
      updates_per_epoch: U,
      batch_size: b,
      record_every: every,
    },
  });

  rule.start(w, oracle); // SAGA fills its gradient table at w₀ here (n gradients)
  let gFull = p.grad(w);
  let fw = p.f(w);
  nMon = 1;
  nFev = 1;
  let gnorm = norm2(gFull);
  trace.push({
    k: 0,
    x: [...w],
    fun: fw,
    gradNorm: gnorm,
    stepSize: null,
    info: {
      epoch: 0,
      batch: null,
      stoch_grad: null,
      full_grad: [...gFull],
      lr: null,
      update: null,
      ifo: oracle.n,
      ...rule.info(),
    },
  });
  if (!(Number.isFinite(fw) && Number.isFinite(gnorm))) {
    // Start test: no update was made, so this is not a divergence (no lr helps).
    const msg =
      `not finite at x0: f(x0) = ${pyG(fw)}, ‖∇f(x0)‖ = ${pyG(gnorm)} (the start point ` +
      'or the data is out of range); no update was made';
    return stop(false, msg, 0, fw, 0);
  }
  if (gnorm <= gtol) return stop(true, `‖∇f(x0)‖ = ${pyG(gnorm)} ≤ gtol at the start`, 0, fw, 0);

  // Divergence test (b): the blow-up thresholds, relative to the start.
  const gCap = BLOWUP * (1.0 + gnorm);
  const fCap = BLOWUP * (1.0 + Math.abs(fw));
  const nonfinite = 'non-finite iterate, loss or gradient';
  const blowup = '1e+08';
  let t = 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const perm = rng.permutation(n);
    rule.beginEpoch(w, oracle);
    for (let j = 0; j < U; j++) {
      const batch = perm.slice(j * b, (j + 1) * b);
      const eta = learningRate(lrSchedule, lr, t, U, T);
      const wPrev = w;
      const [wNext, gEst] = rule.update(w, batch, eta, t, oracle);
      w = wNext;
      t += 1;
      const endOfEpoch = j === U - 1;
      let why = finiteVec(w) ? '' : nonfinite; // non-empty: the run diverged
      if (t % every === 0 || endOfEpoch || why) {
        gFull = p.grad(w);
        nMon += 1;
        gnorm = norm2(gFull);
        if (!(why || Number.isFinite(gnorm))) why = nonfinite;
        else if (!why && gnorm > gCap)
          why = `‖∇f‖ = ${pyG(gnorm)} > ${blowup}·(1 + ‖∇f(x0)‖) = ${pyG(gCap)}`;
      }
      let converged = endOfEpoch && !why && gnorm <= gtol;
      const done = Boolean(why) || converged || t === T;
      if (t % every !== 0 && !done) continue;
      fw = p.f(w);
      nFev += 1;
      if (!(why || Number.isFinite(fw))) why = nonfinite;
      else if (!why && fw > fCap) why = `f = ${pyG(fw)} > ${blowup}·(1 + |f(x0)|) = ${pyG(fCap)}`;
      converged = converged && !why;
      trace.push({
        k: t,
        x: [...w],
        fun: fw,
        gradNorm: gnorm,
        stepSize: eta,
        info: {
          epoch,
          batch: b <= BATCH_INFO_MAX ? [...batch] : null,
          stoch_grad: [...gEst],
          full_grad: [...gFull],
          lr: eta,
          update: w.map((wi, i) => wi - wPrev[i]),
          ifo: oracle.n,
          ...rule.info(),
        },
      });
      if (why) {
        const msg = `diverged: ${why} after update ${t} (epoch ${epoch}); try a smaller lr`;
        return stop(false, msg, t, fw, epoch);
      }
      if (converged) {
        const msg = `‖∇f‖ = ${pyG(gnorm)} ≤ gtol after ${epoch} epochs (${t} updates)`;
        return stop(true, msg, t, fw, epoch);
      }
    }
  }
  const msg = `completed ${epochs} epochs (${T} updates); ‖∇f‖ = ${pyG(gnorm)} > gtol`;
  return stop(false, msg, T, fw, epochs);
}

// ── Parameters ─────────────────────────────────────────────────────────────────────────

function common(lr: number): ParamSpec[] {
  return [
    param.float('lr', lr, {
      min: 1e-5,
      max: 10.0,
      log: true,
      help: 'Base learning rate η₀ (step size).',
      label: 'Learning rate',
      tex: '\\eta_0',
    }),
    param.int('batch_size', 10, {
      min: 1,
      max: 1000,
      help: 'Samples per mini-batch (values above n use the full batch).',
      label: 'Batch size',
      tex: 'b',
    }),
    param.int('epochs', 20, {
      min: 1,
      max: 1000,
      help: 'Passes over the data.',
      label: 'Epochs',
    }),
    param.choice('lr_schedule', 'constant', [...SCHEDULES], {
      help: 'η_t: constant, halve every epochs/4 (step), η₀/√(1+τ), or cosine to 0.',
      label: 'Schedule',
      tex: '\\eta_t',
    }),
    param.float('gtol', 1e-6, {
      min: 1e-14,
      max: 1e-1,
      log: true,
      help: 'Stop when the full-gradient norm ‖∇f‖ ≤ gtol (tested after each epoch).',
      label: 'Gradient tolerance',
      tex: '\\|\\nabla f\\| \\le',
    }),
    param.int('record_every', 0, {
      min: 0,
      max: 100_000,
      help:
        'Record every k-th update in the trace (0 = automatic). The trace is capped at 400 ' +
        'steps: k is raised to ⌈updates/398⌉ when it is smaller.',
      label: 'Record every',
    }),
  ];
}

const EPS_PARAM = param.float('eps', 1e-8, {
  min: 1e-12,
  max: 1e-2,
  log: true,
  help: 'ε in η/(√r + ε): guards the division.',
  label: 'Epsilon',
  tex: '\\varepsilon',
});
const momentumParam = () =>
  param.float('momentum', 0.9, {
    min: 0.0,
    max: 0.999,
    help: 'β: the velocity decay.',
    label: 'Momentum',
    tex: '\\beta',
  });

// ── Teaching material for the MethodCard ───────────────────────────────────────────────

const Q_ETA = { tex: '\\eta_k', key: 'stepSize' };
const Q_EPOCH = { tex: '\\text{epoch}', key: 'info.epoch' };
const Q_G = { tex: '\\mathbf{g}_k', key: 'info.stoch_grad' };
const Q_IFO = { tex: '\\#\\nabla f_i', key: 'info.ifo' };

const DOCS: Record<string, MethodDoc> = {
  sgd: {
    rule: String.raw`\begin{aligned}\mathbf{w}_{k+1} &= \mathbf{w}_k - \eta_k\,\mathbf{g}_k\\ \mathbf{g}_k &= \frac{1}{|B_k|}\sum_{i\in B_k}\nabla f_i(\mathbf{w}_k)\end{aligned}`,
    intuition:
      'Each step follows the gradient of a random mini-batch, an unbiased but noisy estimate of ∇f. With a constant η the iterates settle into a noise ball of radius O(η) around w⋆; only a decaying η (Σηₖ = ∞, Σηₖ² < ∞) shrinks it to zero.',
    order: 'sublinear',
    pros: ['Cost per step is b component gradients, not n', 'Noise helps escape flat regions'],
    cons: [
      'Constant η stalls at an O(η) loss floor',
      'Needs η < 2/L; κ-ill-conditioned valleys force tiny steps',
    ],
    quantities: [Q_ETA, Q_EPOCH, Q_G, Q_IFO],
  },
  sgd_momentum: {
    rule: String.raw`\begin{aligned}\mathbf{v}_{k+1} &= \beta\,\mathbf{v}_k - \eta_k\,\mathbf{g}_k\\ \mathbf{w}_{k+1} &= \mathbf{w}_k + \mathbf{v}_{k+1}\end{aligned}`,
    intuition:
      'The velocity is an exponentially weighted sum of past stochastic gradients: it averages out noise across steps and builds speed along a consistent direction, at the price of overshooting.',
    order: 'sublinear',
    pros: ['Accelerates along shallow valleys', 'Averages gradient noise over ~1/(1 − β) steps'],
    cons: ['Overshoots and oscillates when βη is too large', 'Effective step is η/(1 − β)'],
    quantities: [Q_ETA, { tex: '\\mathbf{v}_k', key: 'info.velocity' }, Q_G, Q_EPOCH],
  },
  sgd_nesterov: {
    rule: String.raw`\begin{aligned}\tilde{\mathbf{w}}_k &= \mathbf{w}_k + \beta\,\mathbf{v}_k\\ \mathbf{v}_{k+1} &= \beta\,\mathbf{v}_k - \eta_k\nabla f_{B_k}(\tilde{\mathbf{w}}_k)\\ \mathbf{w}_{k+1} &= \mathbf{w}_k + \mathbf{v}_{k+1}\end{aligned}`,
    intuition:
      'Momentum that looks before it leaps: the mini-batch gradient is taken at the look-ahead point w + βv, where the velocity is about to carry the iterate, so the correction arrives one step earlier.',
    order: 'sublinear',
    pros: ['Less overshoot than heavy-ball momentum at the same β'],
    cons: ['Still a noise ball with constant η'],
    quantities: [
      Q_ETA,
      { tex: '\\mathbf{v}_k', key: 'info.velocity' },
      { tex: '\\tilde{\\mathbf{w}}_{k-1}', key: 'info.lookahead' },
      Q_EPOCH,
    ],
  },
  stochastic_adagrad: {
    rule: String.raw`\begin{aligned}\mathbf{r}_{k+1} &= \mathbf{r}_k + \mathbf{g}_k\odot\mathbf{g}_k\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \frac{\eta_k}{\sqrt{\mathbf{r}_{k+1}}+\varepsilon}\odot\mathbf{g}_k\end{aligned}`,
    intuition:
      'Every coordinate gets its own step, divided by the root of its accumulated squared gradients. Steep coordinates slow down, flat ones keep moving, and the effective step decays like 1/√k by itself.',
    order: 'sublinear',
    pros: ['Rescales badly scaled coordinates automatically', 'Built-in step decay'],
    cons: ['The accumulator only grows: steps can die out too early'],
    quantities: [
      Q_ETA,
      { tex: '\\mathbf{r}_k', key: 'info.accum' },
      { tex: '\\eta/(\\sqrt{\\mathbf{r}}+\\varepsilon)', key: 'info.scaled_lr' },
      Q_EPOCH,
    ],
  },
  stochastic_rmsprop: {
    rule: String.raw`\begin{aligned}\mathbf{r}_{k+1} &= \rho\,\mathbf{r}_k + (1-\rho)\,\mathbf{g}_k\odot\mathbf{g}_k\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \frac{\eta_k}{\sqrt{\mathbf{r}_{k+1}}+\varepsilon}\odot\mathbf{g}_k\end{aligned}`,
    intuition:
      'AdaGrad with a moving average instead of a sum: each coordinate is divided by its recent root-mean-square gradient, so the step stays near η in every direction instead of dying out.',
    order: 'sublinear',
    pros: ['Per-coordinate scaling without AdaGrad’s decay'],
    cons: ['Steps of size ~η never vanish: needs a schedule to settle'],
    quantities: [
      Q_ETA,
      { tex: '\\mathbf{r}_k', key: 'info.sq_avg' },
      { tex: '\\eta/(\\sqrt{\\mathbf{r}}+\\varepsilon)', key: 'info.scaled_lr' },
      Q_EPOCH,
    ],
  },
  stochastic_adam: {
    rule: String.raw`\begin{aligned}\hat{\mathbf{m}} &= \frac{\mathbf{m}_{k+1}}{1-\beta_1^{\,k+1}},\quad \hat{\mathbf{v}} = \frac{\mathbf{v}_{k+1}}{1-\beta_2^{\,k+1}}\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \eta_k\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\varepsilon}\end{aligned}`,
    intuition:
      'Momentum on the gradient (m) divided by an RMS of the gradient (v), both bias-corrected for their zero start. The step is about η per coordinate whatever the gradient’s scale, so badly scaled problems look round.',
    order: 'sublinear',
    pros: ['Scale-invariant per coordinate', 'Robust defaults (β₁ = 0.9, β₂ = 0.999)'],
    cons: ['Does not converge with constant η in general (Reddi et al. 2018)'],
    quantities: [
      Q_ETA,
      { tex: '\\hat{\\mathbf{m}}_k', key: 'info.m_hat' },
      { tex: '\\hat{\\mathbf{v}}_k', key: 'info.v_hat' },
      { tex: '\\eta/(\\sqrt{\\hat{\\mathbf{v}}}+\\varepsilon)', key: 'info.scaled_lr' },
    ],
  },
  svrg: {
    rule: String.raw`\begin{aligned}\mathbf{v}_k &= \nabla f_{B_k}(\mathbf{w}_k) - \nabla f_{B_k}(\tilde{\mathbf{w}}) + \nabla f(\tilde{\mathbf{w}})\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \eta\,\mathbf{v}_k\end{aligned}`,
    intuition:
      'Once per epoch take one full gradient at a snapshot w̃; every mini-batch gradient is then corrected by the same batch’s gradient at w̃. The estimate stays unbiased and its variance vanishes as w and w̃ approach w⋆, so a constant η converges linearly.',
    order: 'linear (strongly convex)',
    pros: ['Linear convergence with a constant step', 'No n × d gradient table'],
    cons: ['A full gradient (n component gradients) per epoch', '2b gradients per step'],
    quantities: [
      Q_ETA,
      { tex: '\\tilde{\\mathbf{w}}', key: 'info.snapshot' },
      { tex: '\\nabla f(\\tilde{\\mathbf{w}})', key: 'info.snapshot_grad' },
      Q_IFO,
    ],
  },
  saga: {
    rule: String.raw`\begin{aligned}\mathbf{v}_k &= \frac{1}{|B_k|}\sum_{i\in B_k}\big[\nabla f_i(\mathbf{w}_k) - \boldsymbol{\phi}_i\big] + \frac{1}{n}\sum_{j=1}^{n}\boldsymbol{\phi}_j\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \eta\,\mathbf{v}_k,\quad \boldsymbol{\phi}_i \leftarrow \nabla f_i(\mathbf{w}_k)\ \ (i\in B_k)\end{aligned}`,
    intuition:
      'Keep the last gradient φᵢ of every sample. A new mini-batch gradient is corrected by what the table predicted for it, plus the table’s mean: unbiased, and the correction cancels the noise once the table is fresh.',
    order: 'linear (strongly convex)',
    pros: ['Linear convergence with a constant step', 'b gradients per step, no full passes'],
    cons: ['Stores an n × d table of gradients'],
    quantities: [
      Q_ETA,
      { tex: '\\tfrac1n\\textstyle\\sum_j\\boldsymbol{\\phi}_j', key: 'info.table_mean' },
      Q_G,
      Q_IFO,
    ],
  },
  sag: {
    rule: String.raw`\begin{aligned}\mathbf{d} &\leftarrow \mathbf{d} + \sum_{i\in B_k}\big[\nabla f_i(\mathbf{w}_k) - \mathbf{y}_i\big],\quad \mathbf{y}_i \leftarrow \nabla f_i(\mathbf{w}_k)\\ \mathbf{w}_{k+1} &= \mathbf{w}_k - \frac{\eta}{m}\,\mathbf{d}\end{aligned}`,
    intuition:
      'Step along the average of the most recent gradient of every sample seen so far. Stale gradients make the direction biased, but its variance is small; with reshuffled epochs it needs a smaller step than SAGA.',
    order: 'linear (strongly convex)',
    pros: ['Low-variance direction from b gradients per step'],
    cons: [
      'Biased estimate',
      'Stores an n × d table',
      'Unstable for η ≳ 0.1/L_max under reshuffling',
    ],
    quantities: [Q_ETA, { tex: 'm', key: 'info.seen' }, Q_G, Q_IFO],
  },
};

// ── Registered methods ─────────────────────────────────────────────────────────────────

const NEEDS = ['f', 'grad_batch'];

interface Meta {
  order: string;
  summary: string;
  references: string[];
}

function register(
  id: string,
  name: string,
  lr: number,
  extra: ParamSpec[],
  meta: Meta,
  makeRule: (o: Options) => Rule,
): void {
  const fn: MethodFn = (problem, o) => drive(id, problem, makeRule(o), o);
  registerMethod(
    {
      id,
      family: 'stochastic',
      name,
      params: [...common(lr), ...extra],
      needs: NEEDS,
      deterministic: false,
      ...meta,
    },
    fn,
    DOCS[id],
  );
}

register(
  'sgd',
  'Stochastic gradient descent',
  0.05,
  [],
  {
    order: 'sublinear (O(1/k) with decaying η, strongly convex)',
    summary: 'Step against the gradient of a random mini-batch instead of the full gradient.',
    references: [
      'Robbins & Monro (1951)',
      'Bottou, Curtis & Nocedal (2018), SIAM Review 60(2), Alg. 4.1',
    ],
  },
  () => new SGDRule(),
);

register(
  'sgd_momentum',
  'SGD with momentum (heavy ball)',
  0.02,
  [momentumParam()],
  {
    order: 'sublinear (stochastic); accelerates ill-conditioned valleys',
    summary: 'Accumulate a velocity of past stochastic gradients and move along it.',
    references: ['Polyak (1964)', 'Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.2'],
  },
  (o) => {
    checkUnit('momentum', Number(o.momentum));
    return new MomentumRule(Number(o.momentum), false);
  },
);

register(
  'sgd_nesterov',
  'SGD with Nesterov momentum',
  0.02,
  [momentumParam()],
  {
    order: 'sublinear (stochastic)',
    summary: 'Momentum that evaluates the gradient at the look-ahead point w + βv.',
    references: [
      'Nesterov (1983)',
      'Sutskever, Martens, Dahl & Hinton (2013), ICML, eqs. 3-4',
      'Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.3',
    ],
  },
  (o) => {
    checkUnit('momentum', Number(o.momentum));
    return new MomentumRule(Number(o.momentum), true);
  },
);

register(
  'stochastic_adagrad',
  'AdaGrad',
  0.5,
  [EPS_PARAM],
  {
    order: 'sublinear (O(1/√k) regret bound)',
    summary: "Divide each coordinate's step by the root of its summed squared gradients.",
    references: [
      'Duchi, Hazan & Singer (2011), JMLR 12 (diagonal variant)',
      'Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.4',
    ],
  },
  (o) => {
    checkEps(Number(o.eps));
    return new AdaptiveRule('adagrad', Number(o.eps));
  },
);

register(
  'stochastic_rmsprop',
  'RMSProp',
  0.01,
  [
    param.float('rho', 0.9, {
      min: 0.0,
      max: 0.9999,
      help: 'ρ: decay of the squared-gradient average.',
      label: 'Decay',
      tex: '\\rho',
    }),
    EPS_PARAM,
  ],
  {
    order: 'sublinear (stochastic)',
    summary: 'AdaGrad with an exponential moving average, so the step does not die out.',
    references: [
      'Tieleman & Hinton (2012), COURSERA Neural Networks, Lecture 6.5',
      'Goodfellow, Bengio & Courville (2016), Deep Learning, Alg. 8.5',
    ],
  },
  (o) => {
    checkEps(Number(o.eps));
    checkUnit('rho', Number(o.rho));
    return new AdaptiveRule('rmsprop', Number(o.eps), Number(o.rho));
  },
);

register(
  'stochastic_adam',
  'Adam',
  0.05,
  [
    param.float('beta1', 0.9, {
      min: 0.0,
      max: 0.999,
      help: 'β₁: decay of the first moment.',
      label: 'First-moment decay',
      tex: '\\beta_1',
    }),
    param.float('beta2', 0.999, {
      min: 0.0,
      max: 0.99999,
      help: 'β₂: decay of the second moment.',
      label: 'Second-moment decay',
      tex: '\\beta_2',
    }),
    EPS_PARAM,
  ],
  {
    order: 'sublinear (stochastic)',
    summary: 'Momentum on the gradient plus RMSProp scaling, both with bias correction.',
    references: ['Kingma & Ba (2015), ICLR, Algorithm 1'],
  },
  (o) => {
    checkEps(Number(o.eps));
    checkUnit('beta1', Number(o.beta1));
    checkUnit('beta2', Number(o.beta2));
    return new AdaptiveRule('adam', Number(o.eps), 0.9, Number(o.beta1), Number(o.beta2));
  },
);

register(
  'svrg',
  'SVRG (stochastic variance-reduced gradient)',
  0.05,
  [],
  {
    order: 'linear (strongly convex, η < 1/(4L_max))',
    summary: 'Correct each stochastic gradient with a full gradient taken once per epoch.',
    references: ['Johnson & Zhang (2013), NeurIPS, Procedure SVRG (Fig. 1), option I'],
  },
  () => new SVRGRule(),
);

register(
  'saga',
  'SAGA',
  0.05,
  [],
  {
    order: 'linear (strongly convex, η = 1/(3L_max))',
    summary: 'Keep the last gradient of every sample; correct each new one by the table.',
    references: ['Defazio, Bach & Lacoste-Julien (2014), NeurIPS, §2 (SAGA update)'],
  },
  () => new TableRule(true),
);

register(
  'sag',
  'SAG (stochastic average gradient)',
  0.05,
  [],
  {
    order: 'linear (strongly convex, η = 1/(16L_max))',
    summary: 'Step along the average of the last gradient seen for every sample.',
    references: [
      'Le Roux, Schmidt & Bach (2012), NeurIPS',
      'Schmidt, Le Roux & Bach (2017), Math. Programming 162, Alg. 1 and §4.1',
    ],
  },
  () => new TableRule(false),
);
