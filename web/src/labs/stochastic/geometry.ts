/**
 * The geometry of one stochastic step, computed from `Step.info` and the problem (pure, tested).
 *
 * For the update that produced 𝐰ₖ the lab draws, at the point where the method stood:
 *
 *   base   𝐰ₖ₋₁ = 𝐰ₖ − info.update (exact even when the trace skips updates);
 *   push   the deterministic part of the step that does not depend on the mini-batch:
 *          momentum βvₖ₋₁ = update + η·g (heavy ball and Nesterov), else 0;
 *   eval   where the mini-batch gradient was evaluated (Nesterov: the look-ahead 𝐰ₖ₋₁ + βvₖ₋₁);
 *   mean   the expected step: base + push − η∇f(eval) (SGD, momentum, SVRG, SAGA: the
 *          estimator is unbiased; SAG: the full-gradient step it is compared with);
 *   cov    the exact covariance of the random part −η·g over all mini-batches of size b drawn
 *          without replacement: η² (n − b)/(b(n − 1)) · S, with S the population covariance of
 *          the per-sample vectors (∇fᵢ for plain mini-batches, ∇fᵢ(𝐰) − ∇fᵢ(𝐰̃) for SVRG).
 *          Each mini-batch of a shuffled epoch is, on its own, a uniform b-subset, so this is the
 *          covariance of the step marginalized over the epoch's permutation. Within an epoch the
 *          batches are not independent (given the earlier batches, the last one is determined),
 *          and the step is a discrete distribution over b-subsets, not a Gaussian: the 2σ ellipse
 *          holds ≈ 86 % of it only to the extent the step is Gaussian. SAG/SAGA (gradient table)
 *          and the adaptive methods (the step depends nonlinearly on g) have no closed form
 *          here: `cov` is null.
 */
import type { Step, Vector } from '../../core/types';
import type { FiniteSumProblem } from '../../problems/stochastic';

export type RuleKind = 'sgd' | 'momentum' | 'nesterov' | 'adaptive' | 'svrg' | 'table';

export function ruleKind(methodId: string): RuleKind {
  switch (methodId) {
    case 'sgd':
      return 'sgd';
    case 'sgd_momentum':
      return 'momentum';
    case 'sgd_nesterov':
      return 'nesterov';
    case 'svrg':
      return 'svrg';
    case 'saga':
    case 'sag':
      return 'table';
    default:
      return 'adaptive';
  }
}

export type Pt = [number, number];

export interface StepGeometry {
  kind: RuleKind;
  base: Pt;
  /** The new iterate 𝐰ₖ. */
  to: Pt;
  /** base + deterministic push (momentum), where the random part starts. */
  pushTo: Pt;
  /** Where the stochastic gradient was evaluated. */
  evalAt: Pt;
  /** Expected position after the step (null for adaptive methods). */
  mean: Pt | null;
  /** Covariance of the step's random part (null when no closed form). */
  cov: [[number, number], [number, number]] | null;
  /** SVRG snapshot 𝐰̃. */
  snapshot: Pt | null;
  /** −∇f(evalAt) scaled to the update's length (adaptive methods: the direction it rescales). */
  gradDir: Pt | null;
  eta: number;
  /** Mini-batch indices of the update (null when b > 32 or at k = 0). */
  batch: number[] | null;
  /** ‖g − ∇f(eval)‖: this step's gradient noise (for the table and card). */
  noise: number | null;
  /**
   * The step as a head-to-tail chain of its exact per-sample parts, from `pushTo` (or `base`) to
   * `to`: SGD/momentum −(η/|B|)∇fᵢ(eval) for i ∈ B in sampled order; SVRG first −η∇f(𝐰̃), then
   * −(η/|B|)(∇fᵢ(𝐰) − ∇fᵢ(𝐰̃)); AdaGrad/RMSProp −(η/(√r + ε)) ⊙ ∇fᵢ/|B|. Null when the batch was
   * not recorded (b > 32) or the step is not a sum over the batch (Adam, SAG, SAGA).
   */
  chain: Pt[] | null;
}

const vec = (v: unknown): Vector | null =>
  Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === 'number')
    ? (v as Vector)
    : null;

/** Population covariance (1/n)Σ(rᵢ − r̄)(rᵢ − r̄)ᵀ of 2-vectors. */
export function covariance(
  rows: readonly (readonly number[])[],
): [[number, number], [number, number]] {
  const n = rows.length;
  let m0 = 0,
    m1 = 0;
  for (const r of rows) {
    m0 += r[0];
    m1 += r[1];
  }
  m0 /= n;
  m1 /= n;
  let a = 0,
    b = 0,
    c = 0;
  for (const r of rows) {
    const d0 = r[0] - m0,
      d1 = r[1] - m1;
    a += d0 * d0;
    b += d0 * d1;
    c += d1 * d1;
  }
  return [
    [a / n, b / n],
    [b / n, c / n],
  ];
}

/** Var of a mean of b draws without replacement from n: S·(n − b)/(b(n − 1)). */
export function batchFactor(n: number, b: number): number {
  if (b >= n) return 0;
  return (n - b) / (b * (n - 1));
}

/** Inverse of a 2×2 SPD matrix scaled for `ellipse` overlays; null when (near) singular. */
export function inverse2(
  M: readonly (readonly number[])[],
): [[number, number], [number, number]] | null {
  const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const scale = Math.max(Math.abs(M[0][0]), Math.abs(M[1][1]), 1e-300);
  if (!(det > 1e-14 * scale * scale) || !Number.isFinite(det)) return null;
  return [
    [M[1][1] / det, -M[0][1] / det],
    [-M[1][0] / det, M[0][0] / det],
  ];
}

/**
 * βvₖ₋₁ of a heavy-ball or Nesterov update: vₖ = βvₖ₋₁ − ηg and 𝐰ₖ = 𝐰ₖ₋₁ + vₖ give
 * βvₖ₋₁ = update + ηg. The sum cancels: v₀ = 0 exactly at k = 1, and a component that is
 * rounding residue (below 10⁻¹² of the terms it came from) is set to 0.
 */
export function momentumPush(
  k: number,
  update: readonly number[],
  g: readonly number[],
  eta: number,
): Pt {
  if (k <= 1) return [0, 0];
  const c = (i: number) => {
    const v = update[i] + eta * g[i];
    return Math.abs(v) <= 1e-12 * Math.max(Math.abs(update[i]), Math.abs(eta * g[i])) ? 0 : v;
  };
  return [c(0), c(1)];
}

export function stepGeometry(
  problem: FiniteSumProblem,
  methodId: string,
  step: Step,
  batchSize: number,
): StepGeometry | null {
  if (step.k === 0) return null;
  const x = vec(step.x),
    update = vec(step.info.update),
    g = vec(step.info.stoch_grad);
  const eta = step.stepSize;
  if (!x || !update || !g || eta === null || !Number.isFinite(eta)) return null;
  const kind = ruleKind(methodId);
  const base: Pt = [x[0] - update[0], x[1] - update[1]];
  if (!base.every(Number.isFinite)) return null;
  const push: Pt =
    kind === 'momentum' || kind === 'nesterov' ? momentumPush(step.k, update, g, eta) : [0, 0];
  const pushTo: Pt = [base[0] + push[0], base[1] + push[1]];
  const evalAt: Pt = kind === 'nesterov' ? pushTo : base;
  const full = problem.grad(evalAt);
  const n = problem.nSamples;
  const bFull = Math.min(batchSize, n);
  // Update t = k − 1 is the j-th of its epoch; the last mini-batch of an epoch is smaller when b ∤ n.
  const U = Math.ceil(n / bFull);
  const j = (step.k - 1) % U;
  const b = j === U - 1 ? n - (U - 1) * bFull : bFull;
  const snapshot = kind === 'svrg' ? (vec(step.info.snapshot) as Pt | null) : null;

  let mean: Pt | null = null;
  let gradDir: Pt | null = null;
  if (kind === 'adaptive') {
    const len = Math.hypot(update[0], update[1]);
    const gn = Math.hypot(full[0], full[1]);
    if (gn > 0 && len > 0)
      gradDir = [base[0] - (full[0] / gn) * len, base[1] - (full[1] / gn) * len];
  } else {
    mean = [pushTo[0] - eta * full[0], pushTo[1] - eta * full[1]];
  }

  let cov: StepGeometry['cov'] = null;
  const factor = batchFactor(n, b);
  if (
    factor > 0 &&
    (kind === 'sgd' || kind === 'momentum' || kind === 'nesterov' || kind === 'svrg')
  ) {
    const all = Array.from({ length: n }, (_, i) => i);
    let rows = problem.gradSamples(evalAt, all);
    if (kind === 'svrg') {
      if (!snapshot) return null;
      const rs = problem.gradSamples(snapshot, all);
      rows = rows.map((r, i) => [r[0] - rs[i][0], r[1] - rs[i][1]]);
    }
    const S = covariance(rows);
    const s = eta * eta * factor;
    cov = [
      [s * S[0][0], s * S[0][1]],
      [s * S[1][0], s * S[1][1]],
    ];
  }
  const batch = Array.isArray(step.info.batch) ? (step.info.batch as number[]) : null;
  let chain: Pt[] | null = null;
  const scaled = vec(step.info.scaled_lr);
  if (batch && batch.length > 0) {
    const m = batch.length;
    const add = (
      start: Pt,
      parts: readonly (readonly number[])[],
      w: (v: number, c: number) => number,
    ) => {
      const out: Pt[] = [start];
      let p = start;
      for (const r of parts) {
        p = [p[0] + w(r[0], 0), p[1] + w(r[1], 1)];
        out.push(p);
      }
      return out;
    };
    if (kind === 'sgd' || kind === 'momentum' || kind === 'nesterov') {
      chain = add(pushTo, problem.gradSamples(evalAt, batch), (v) => (-eta * v) / m);
    } else if (kind === 'svrg' && snapshot) {
      const mu = vec(step.info.snapshot_grad);
      if (mu) {
        const rw = problem.gradSamples(base, batch);
        const rs = problem.gradSamples(snapshot, batch);
        const start: Pt = [base[0] - eta * mu[0], base[1] - eta * mu[1]];
        chain = [
          base,
          ...add(
            start,
            rw.map((r, i) => [r[0] - rs[i][0], r[1] - rs[i][1]]),
            (v) => (-eta * v) / m,
          ),
        ];
      }
    } else if ((methodId === 'stochastic_adagrad' || methodId === 'stochastic_rmsprop') && scaled) {
      chain = add(base, problem.gradSamples(base, batch), (v, c) => (-scaled[c] * v) / m);
    }
  }
  return {
    kind,
    base,
    to: [x[0], x[1]],
    pushTo,
    evalAt,
    mean,
    cov,
    snapshot,
    gradDir,
    eta,
    batch,
    noise: Math.hypot(g[0] - full[0], g[1] - full[1]),
    chain,
  };
}

/** Mahalanobis radius of the 1 − p region of a 2-D Gaussian: r² = −2 ln p (r = 2: 86.5 %). */
export const ELLIPSE_R = 2;
export const ELLIPSE_MASS = 1 - Math.exp(-(ELLIPSE_R * ELLIPSE_R) / 2);

/**
 * What one step of this lab is called everywhere (player, lens, caption, status chips, card): an
 * update of 𝐰, counted by k. An epoch is a pass over the data (n / b updates).
 */
export const UPDATE_NOUN = ['update', 'updates'] as const;

/**
 * The epoch position of a trace step: k / U (updates per epoch). Every method in the lab shares
 * the batch size, so U and the trace's recording interval are the same for all of them.
 */
export const epochOf = (k: number, updatesPerEpoch: number) => k / Math.max(1, updatesPerEpoch);

export type DataMode = 'line' | 'logistic' | 'predicted';

/** How the data panel shows a problem: a 1-D regressor with intercept, labels, or ŷ vs y. */
export function dataMode(p: FiniteSumProblem): DataMode {
  if (p.loss === 'logistic') return 'logistic';
  return p.X.every((r) => r[0] === 1) ? 'line' : 'predicted';
}
