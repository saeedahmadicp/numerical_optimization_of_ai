/**
 * Finite-sum objectives of machine learning (kind `"stochastic"`) — TS port of
 * `numopt.problems.stochastic` (src/numopt/problems/stochastic.py).
 *
 * Every problem is a regularized empirical risk over n samples (aᵢ, yᵢ) with a linear predictor
 * zᵢ = aᵢᵀw (Bottou, Curtis & Nocedal (2018), eq. 2.3 / §3):
 *
 *     f(w) = (1/n) Σᵢ fᵢ(w),        fᵢ(w) = φ(aᵢᵀw, yᵢ) + (λ/2)‖w‖²,
 *     ∇fᵢ(w) = φ'(aᵢᵀw, yᵢ) aᵢ + λw,      ∇²f(w) = (1/n) Aᵀ diag(φ''(Aw, y)) A + λI.
 *
 * Per-sample losses φ(z, y): `squared` ½(z − y)²; `logistic` log(1 + eᶻ) − yz (evaluated without
 * overflow); `huber` (δ = `huberDelta`) ½r² for |r| ≤ δ, δ(|r| − ½δ) otherwise, r = z − y.
 *
 * `gradBatch(w, idx)` = (1/|B|) Σ_{i∈B} ∇fᵢ(w); it sorts `idx` first (like Python), so the value
 * does not depend on the order of `idx`, and `grad(w)` is `gradBatch(w, 0..n−1)`.
 * `gradSamples(w, idx)` returns the rows ∇fᵢ(w) in the order of `idx`.
 *
 * Data come only from `Rng` (Mulberry32) in the order each builder documents, so the samples equal
 * the Python ones (to libm precision in `log`/`cos`/`exp`; tests/stochastic.test.ts compares them
 * with `problems.json`). The minimizer, f⋆ and the smoothness constants are the Python reference
 * values (`problems.json`), not recomputed here; the tests check ∇f(w⋆) ≈ 0.
 *
 * Floating-point order follows NumPy where it matters: zᵢ = aᵢ₀w₀ + aᵢ₁w₁; `f` averages with
 * NumPy's pairwise summation (`np.mean` over a contiguous axis); `gradBatch` sums aᵢφ'ᵢ in sorted
 * index order (NumPy calls BLAS here, so the last bit can differ).
 *
 * Ids (n = 200 samples, d = 2 parameters): linreg_2d, logreg_2d, ill_conditioned_ls,
 * huber_regression_2d.
 */
import { Rng } from '../core/rng';
import type { Matrix, Vector } from '../core/types';
import { addProblem } from './registry';

export type Loss = 'squared' | 'logistic' | 'huber';

/** Number of samples of every library problem. */
export const N_SAMPLES = 200;

export interface StochasticExtra {
  seed: number;
  /** The parameters that generated the data. */
  true_w: [number, number];
  /** Standard deviation of the Gaussian label noise (regression problems). */
  noise_std?: number;
  /** f(minima[0]). */
  f_min: number;
  /** f is L-smooth. */
  L: number;
  /** Every fᵢ is L_max-smooth. */
  L_max: number;
  /** A global strong-convexity constant (0 for Huber). */
  mu: number;
  /** LaTeX of the predictor. */
  model: string;
  /** huber_regression_2d: indices of the corrupted samples. */
  outliers?: number[];
}

/** f(w) = (1/n) Σᵢ [φ(aᵢᵀw, yᵢ) + (λ/2)‖w‖²] over a data set (X, y). */
export interface FiniteSumProblem {
  readonly kind: 'stochastic';
  id: string;
  name: string;
  latex: string;
  /** Feature rows aᵢ (n × d). Frozen. */
  X: readonly (readonly number[])[];
  /** Targets (labels in {0, 1} for the logistic loss). Frozen. */
  y: readonly number[];
  loss: Loss;
  /** Plotting box [[lo₀, hi₀], [lo₁, hi₁]] in parameter space. */
  domain: [[number, number], [number, number]];
  x0: [number, number];
  minima: [number, number][];
  l2: number;
  huberDelta: number;
  description: string;
  tags: string[];
  extra: StochasticExtra;
  dim: number;
  nSamples: number;
  /** Full average loss f(w). */
  f: (w: readonly number[]) => number;
  /** ∇f(w) = gradBatch(w, 0..n−1). */
  grad: (w: readonly number[]) => Vector;
  /** (1/|B|) Σ_{i∈B} ∇fᵢ(w) for the multiset B = idx (sorted first). */
  gradBatch: (w: readonly number[], idx: readonly number[]) => Vector;
  /** Rows ∇fᵢ(w), one per entry of idx, in the order of idx. */
  gradSamples: (w: readonly number[], idx: readonly number[]) => Matrix;
  /** ∇²f(w) (Huber: φ'' = 1 on |r| ≤ δ). */
  hess: (w: readonly number[]) => Matrix;
  /** Per-sample loss φ(z, y) (no ridge term), for plots. */
  phi: (z: number, y: number) => number;
}

// ── Per-sample losses φ(z, y) of the linear predictor z = aᵀw ───────────────────────────

/** (σ(z), σ(−z), e^{−|z|}) without overflow. */
function sigmoids(z: number): [number, number, number] {
  const e = Math.exp(-Math.abs(z));
  const big = 1.0 / (1.0 + e); // σ(|z|)
  const small = e / (1.0 + e); // σ(−|z|)
  return z >= 0 ? [big, small, e] : [small, big, e];
}

export function phi(loss: Loss, z: number, y: number, delta: number): number {
  if (loss === 'squared') {
    const r = z - y;
    return 0.5 * r * r;
  }
  if (loss === 'logistic') return Math.max(z, 0.0) + Math.log1p(Math.exp(-Math.abs(z))) - y * z;
  const r = Math.abs(z - y);
  return r <= delta ? 0.5 * r * r : delta * (r - 0.5 * delta);
}

export function dphi(loss: Loss, z: number, y: number, delta: number): number {
  if (loss === 'squared') return z - y;
  if (loss === 'logistic') {
    const [sPos, sNeg] = sigmoids(z);
    return (1.0 - y) * sPos - y * sNeg;
  }
  // np.clip(z − y, −δ, δ)
  return Math.min(Math.max(z - y, -delta), delta);
}

export function d2phi(loss: Loss, z: number, y: number, delta: number): number {
  if (loss === 'squared') return 1.0;
  if (loss === 'logistic') {
    const e = sigmoids(z)[2];
    return e / ((1.0 + e) * (1.0 + e));
  }
  return Math.abs(z - y) <= delta ? 1.0 : 0.0;
}

/**
 * NumPy's pairwise summation of `a[start .. start + n)` (`pairwise_sum` in loops_utils.h: blocks
 * of ≤ 128 with eight accumulators, halves split at a multiple of 8), as `np.add.reduce` uses it
 * on a contiguous axis. `np.mean(a)` = `(0 + pairwiseSum(a)) / n`.
 */
export function pairwiseSum(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += a[start + i];
    return res;
  }
  if (n <= 128) {
    const r = [0, 1, 2, 3, 4, 5, 6, 7].map((j) => a[start + j]);
    const m = n - (n % 8);
    for (let i = 8; i < m; i += 8) for (let j = 0; j < 8; j++) r[j] += a[start + i + j];
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]));
    for (let i = m; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2);
}

// ── The problem type ───────────────────────────────────────────────────────────────────

interface Spec {
  id: string;
  name: string;
  latex: string;
  X: number[][];
  y: number[];
  loss: Loss;
  l2: number;
  huberDelta?: number;
  domain: [[number, number], [number, number]];
  x0: [number, number];
  minimum: [number, number];
  description: string;
  tags: string[];
  extra: StochasticExtra;
}

function checkIndex(idx: readonly number[], n: number): void {
  if (idx.length === 0) throw new Error('a mini-batch needs at least one index');
  for (const i of idx)
    if (!(Number.isInteger(i) && i >= 0 && i < n))
      throw new Error(`sample indices must lie in [0, ${n})`);
}

function build(s: Spec): FiniteSumProblem {
  const X = Object.freeze(
    s.X.map((row) => Object.freeze([...row])),
  ) as readonly (readonly number[])[];
  const y = Object.freeze([...s.y]) as readonly number[];
  const n = X.length;
  const d = X[0].length;
  const { loss, l2 } = s;
  const delta = s.huberDelta ?? 1.0;
  const all = Array.from({ length: n }, (_, i) => i);
  const zOf = (a: readonly number[], w: readonly number[]) => {
    // A @ w row by row: a₀w₀ + a₁w₁ (+ …), left to right.
    let z = a[0] * w[0];
    for (let j = 1; j < d; j++) z += a[j] * w[j];
    return z;
  };

  const f = (w: readonly number[]): number => {
    const vals = new Float64Array(n);
    for (let i = 0; i < n; i++) vals[i] = phi(loss, zOf(X[i], w), y[i], delta);
    const mean = (0.0 + pairwiseSum(vals)) / n;
    let ww = 0.0;
    for (let j = 0; j < d; j++) ww += w[j] * w[j];
    return mean + 0.5 * l2 * ww;
  };

  const gradBatch = (w: readonly number[], idx: readonly number[]): Vector => {
    checkIndex(idx, n);
    const ii = [...idx].sort((a, b) => a - b);
    const acc = new Array<number>(d).fill(0.0);
    for (const i of ii) {
      const a = X[i];
      const sI = dphi(loss, zOf(a, w), y[i], delta);
      for (let j = 0; j < d; j++) acc[j] += a[j] * sI;
    }
    return acc.map((v, j) => v / ii.length + l2 * w[j]);
  };

  const gradSamples = (w: readonly number[], idx: readonly number[]): Matrix => {
    checkIndex(idx, n);
    return idx.map((i) => {
      const a = X[i];
      const sI = dphi(loss, zOf(a, w), y[i], delta);
      return a.map((aj, j) => sI * aj + l2 * w[j]);
    });
  };

  const hess = (w: readonly number[]): Matrix => {
    const H = Array.from({ length: d }, () => new Array<number>(d).fill(0.0));
    for (let i = 0; i < n; i++) {
      const a = X[i];
      const c = d2phi(loss, zOf(a, w), y[i], delta);
      for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) H[p][q] += c * a[p] * a[q];
    }
    const M = H.map((row, p) => row.map((v, q) => v / n + (p === q ? l2 : 0.0)));
    return M.map((row, p) => row.map((v, q) => 0.5 * (v + M[q][p])));
  };

  return {
    kind: 'stochastic',
    id: s.id,
    name: s.name,
    latex: s.latex,
    X,
    y,
    loss,
    domain: s.domain,
    x0: s.x0,
    minima: [s.minimum],
    l2,
    huberDelta: delta,
    description: s.description,
    tags: ['finite-sum', ...s.tags],
    extra: s.extra,
    dim: d,
    nSamples: n,
    f,
    grad: (w) => gradBatch(w, all),
    gradBatch,
    gradSamples,
    hess,
    phi: (z, yy) => phi(loss, z, yy, delta),
  };
}

const withIntercept = (x: readonly number[]) => x.map((v) => [1.0, v]);

// ── Library problems ───────────────────────────────────────────────────────────────────

/** y ≈ w₀ + w₁x, squared loss, x ~ U(−1, 3). Rng(11): xᵢ = uniform(−1, 3), then yᵢ = 1 + 2xᵢ + normal(0, 0.5). */
function linreg2d(): FiniteSumProblem {
  const seed = 11,
    sigma = 0.5,
    trueW: [number, number] = [1.0, 2.0];
  const rng = new Rng(seed);
  const x: number[] = [],
    y: number[] = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    x.push(rng.uniform(-1.0, 3.0));
    y.push(trueW[0] + trueW[1] * x[i] + rng.normal(0.0, sigma));
  }
  return build({
    id: 'linreg_2d',
    name: 'Linear regression (2 parameters)',
    latex: String.raw`f(w) = \frac{1}{2n}\sum_{i=1}^{n} (w_0 + w_1 x_i - y_i)^2`,
    X: withIntercept(x),
    y,
    loss: 'squared',
    l2: 0.0,
    domain: [
      [-3.0, 4.0],
      [-1.5, 4.5],
    ],
    x0: [-2.0, -1.0],
    minimum: [1.0146625582587192, 1.9849236559665782],
    description:
      'Fit a line to 200 noisy points. The loss is a quadratic bowl; the intercept and the slope are correlated because x is not centered (κ ≈ 6).',
    tags: ['regression', 'quadratic', 'convex'],
    extra: {
      seed,
      true_w: trueW,
      noise_std: sigma,
      model: String.raw`\hat y = w_0 + w_1 x`,
      f_min: 0.10615560009538562,
      L: 2.77375489338458,
      L_max: 9.833074260408432,
      mu: 0.4550680797050327,
    },
  });
}

/** P(y = 1 | x) = σ(w₀ + w₁x). Rng(12): xᵢ = uniform(−3, 3), then yᵢ = [random() < σ(0.5 + 2.5xᵢ)]. */
function logreg2d(): FiniteSumProblem {
  const seed = 12,
    trueW: [number, number] = [0.5, 2.5],
    l2 = 1e-2;
  const rng = new Rng(seed);
  const x: number[] = [],
    y: number[] = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    x.push(rng.uniform(-3.0, 3.0));
    const prob = 1.0 / (1.0 + Math.exp(-(trueW[0] + trueW[1] * x[i])));
    y.push(rng.random() < prob ? 1.0 : 0.0);
  }
  return build({
    id: 'logreg_2d',
    name: 'Logistic regression (2 parameters)',
    latex:
      String.raw`f(w) = \frac{1}{n}\sum_{i=1}^{n} \left[\log(1 + e^{z_i}) - y_i z_i\right]` +
      String.raw` + \frac{\lambda}{2}\|w\|^2,\ z_i = w_0 + w_1 x_i,\ \lambda = 10^{-2}`,
    X: withIntercept(x),
    y,
    loss: 'logistic',
    l2,
    domain: [
      [-4.0, 4.0],
      [-2.0, 6.0],
    ],
    x0: [-3.0, -1.0],
    minimum: [0.4073844030797963, 2.0146988267693358],
    description:
      'Classify 200 points on a line into two nearly separable classes. The ridge term λ = 10⁻² makes the minimizer unique; far from it the loss is almost linear, near it almost quadratic.',
    tags: ['classification', 'convex', 'strongly-convex'],
    extra: {
      seed,
      true_w: trueW,
      model: String.raw`P(y=1) = \sigma(w_0 + w_1 x)`,
      f_min: 0.28000181298393756,
      L: 0.6940738091383543,
      L_max: 2.502699857640877,
      mu: 0.01,
    },
  });
}

/** y ≈ w₀u + w₁(30v). Rng(13): uᵢ = normal(), vᵢ = normal(), aᵢ = (uᵢ, 30vᵢ), yᵢ = aᵢ·(1, 0.5) + normal(0, 0.5). */
function illConditionedLs(): FiniteSumProblem {
  const seed = 13,
    sigma = 0.5,
    trueW: [number, number] = [1.0, 0.5],
    scale = 30.0;
  const rng = new Rng(seed);
  const X: number[][] = [],
    y: number[] = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    const u = rng.normal();
    const v = scale * rng.normal();
    X.push([u, v]);
    y.push(trueW[0] * u + trueW[1] * v + rng.normal(0.0, sigma));
  }
  return build({
    id: 'ill_conditioned_ls',
    name: 'Ill-conditioned least squares',
    latex: String.raw`f(w) = \frac{1}{2n}\sum_{i=1}^{n} (w_0 u_i + 30\,w_1 v_i - y_i)^2`,
    X,
    y,
    loss: 'squared',
    l2: 0.0,
    domain: [
      [-2.0, 4.0],
      [-1.0, 2.0],
    ],
    x0: [-1.5, 1.5],
    minimum: [1.0633956601199213, 0.49975953410014634],
    description:
      'Two features on scales 1 and 30: the Hessian has κ ≈ 1070, a long narrow valley. Plain SGD must use a tiny step; per-coordinate methods (AdaGrad, RMSProp, Adam) rescale the axes.',
    tags: ['regression', 'quadratic', 'ill-conditioned'],
    extra: {
      seed,
      true_w: trueW,
      noise_std: sigma,
      model: String.raw`\hat y = w_0 u + w_1 (30 v)`,
      f_min: 0.11641117614086988,
      L: 940.6762929025842,
      L_max: 7510.934855775995,
      mu: 0.8826495974276637,
    },
  });
}

/**
 * y ≈ w₀ + w₁x with 10 % gross outliers, Huber loss (δ = 1). Rng(14): xᵢ = uniform(−1, 3),
 * eᵢ = normal(0, 0.3), uᵢ = random(), oᵢ = uniform(5, 15) (always drawn); yᵢ = 1 + 2xᵢ + eᵢ, plus oᵢ
 * when uᵢ < 0.1.
 */
function huberRegression2d(): FiniteSumProblem {
  const seed = 14,
    sigma = 0.3,
    trueW: [number, number] = [1.0, 2.0],
    delta = 1.0,
    pOut = 0.1;
  const rng = new Rng(seed);
  const x: number[] = [],
    y: number[] = [],
    outliers: number[] = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    x.push(rng.uniform(-1.0, 3.0));
    const e = rng.normal(0.0, sigma);
    const u = rng.random();
    const o = rng.uniform(5.0, 15.0);
    let yi = trueW[0] + trueW[1] * x[i] + e;
    if (u < pOut) {
      yi += o;
      outliers.push(i);
    }
    y.push(yi);
  }
  return build({
    id: 'huber_regression_2d',
    name: 'Huber regression (2 parameters)',
    latex:
      String.raw`f(w) = \frac{1}{n}\sum_{i=1}^{n} h_\delta(w_0 + w_1 x_i - y_i),\ ` +
      String.raw`h_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ ` +
      String.raw`\delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases},\ \delta = 1`,
    X: withIntercept(x),
    y,
    loss: 'huber',
    l2: 0.0,
    huberDelta: delta,
    domain: [
      [-3.0, 5.0],
      [-1.5, 4.5],
    ],
    x0: [-2.0, -1.0],
    minimum: [1.1671213873115458, 1.9697596176768577],
    description:
      'A line fit with 10% gross outliers. The Huber loss is quadratic for small residuals and linear for large ones, so the outliers pull the fit far less than in least squares. f is convex and C¹ but its Hessian jumps.',
    tags: ['regression', 'robust', 'convex', 'nonsmooth-hessian'],
    extra: {
      seed,
      true_w: trueW,
      noise_std: sigma,
      outliers,
      model: String.raw`\hat y = w_0 + w_1 x`,
      f_min: 0.974441939747105,
      L: 3.25874618754091,
      L_max: 9.948618310480715,
      mu: 0.0,
    },
  });
}

export const STOCHASTIC_PROBLEMS: readonly FiniteSumProblem[] = [
  linreg2d(),
  logreg2d(),
  illConditionedLs(),
  huberRegression2d(),
];

for (const p of STOCHASTIC_PROBLEMS) addProblem('stochastic', p);

/** Type guard for the stochastic methods (Python raises TypeError for other problems). */
export function isFiniteSumProblem(p: unknown): p is FiniteSumProblem {
  return (
    typeof p === 'object' &&
    p !== null &&
    (p as { kind?: unknown }).kind === 'stochastic' &&
    typeof (p as { gradBatch?: unknown }).gradBatch === 'function'
  );
}
