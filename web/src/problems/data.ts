/**
 * Datasets (x_i, y_i) for the interpolation and regression families — TS port of
 * `numopt.problems.data` (src/numopt/problems/data.py).
 *
 * When the data come from a known function, `fTrue` is that function. For the noisy sets it is
 * the noise-free regression function: E[y | x] for additive noise, and the median of y | x for
 * `exponential_growth` (multiplicative log-normal noise).
 *
 * Noise is drawn ONLY from `Rng` (Mulberry32), one `normal` draw per point in increasing index
 * order, exactly as Python does, so the TS data equal `problems.json` (to libm precision in
 * `log`/`cos`/`exp`; tests/problems_data.test.ts checks every value). Arrays are frozen so no
 * method can mutate the shared library copy.
 *
 * Ids: runge_equispaced, runge_chebyshev, sine_samples, step_data, noisy_linear,
 * noisy_quadratic, anscombe_1, outliers_linear, exponential_growth.
 */
import { Rng } from '../core/rng';
import type { Dataset, Vector } from '../core/types';
import { addProblem } from './registry';

export interface DataProblem extends Dataset {
  x: Vector;
  y: Vector;
  domain: [number, number];
}

/** `np.linspace(start, stop, num)` (endpoint included), with NumPy's rounding. */
export function linspace(start: number, stop: number, num: number): number[] {
  const div = num - 1;
  const delta = stop - start;
  const step = delta / div;
  const out = Array.from({ length: num }, (_, i) =>
    step === 0 ? (i / div) * delta + start : i * step + start,
  );
  if (num > 1) out[num - 1] = stop;
  return out;
}

/** `n` draws of N(0, std²) from Mulberry32 seeded with `seed`, in index order. */
function normalNoise(seed: number, n: number, std: number): number[] {
  const rng = new Rng(seed);
  return Array.from({ length: n }, () => rng.normal(0.0, std));
}

/** Runge's function f(x) = 1 / (1 + 25 x²). */
export function rungeFn(x: number): number {
  return 1.0 / (1.0 + 25.0 * (x * x));
}

/**
 * The n roots of T_n mapped to [a, b], ascending: u_j = cos(π(2j + 1)/(2n)), j = n−1..0,
 * x_j = (a + b)/2 + (b − a)/2 · u_j (Burden & Faires, Theorem 8.9).
 */
export function chebyshevNodesFirstKind(n: number, a = -1.0, b = 1.0): number[] {
  if (n < 1) throw new Error('need at least one Chebyshev node');
  const out: number[] = [];
  for (let j = n - 1; j >= 0; j--) {
    const u = Math.cos((Math.PI * (2.0 * j + 1.0)) / (2.0 * n));
    out.push(0.5 * (a + b) + 0.5 * (b - a) * u);
  }
  return out;
}

/** Heaviside step H(x) = 1 for x ≥ 0, else 0. */
export function unitStep(x: number): number {
  return x >= 0.0 ? 1.0 : 0.0;
}

/** E[y | x] for noisy_quadratic: 1 − x + 0.5 x². */
export function quadraticTruth(x: number): number {
  return 1.0 - x + 0.5 * x * x;
}

/** Median growth curve for exponential_growth: 2 exp(0.3 x). */
export function exponentialTruth(x: number): number {
  return 2.0 * Math.exp(0.3 * x);
}

function line(beta0: number, beta1: number): (x: number) => number {
  return (x) => beta0 + beta1 * x;
}

export const NOISY_LINEAR_SEED = 42;
export const NOISY_QUADRATIC_SEED = 7;
export const OUTLIERS_LINEAR_SEED = 11;
export const EXPONENTIAL_GROWTH_SEED = 3;
/** Indices and additive offsets of the two gross outliers in `outliers_linear`. */
export const OUTLIER_OFFSETS: readonly (readonly [number, number])[] = [
  [5, 15.0],
  [16, -20.0],
];

function dataset(
  id: string,
  name: string,
  x: number[],
  y: number[],
  fTrue: ((x: number) => number) | undefined,
  latex: string,
  domain: [number, number],
  description: string,
): DataProblem {
  const d: DataProblem = {
    id,
    name,
    x: Object.freeze(x.slice()) as Vector,
    y: Object.freeze(y.slice()) as Vector,
    latex,
    domain,
    description,
  };
  if (fTrue) d.fTrue = fTrue;
  return addProblem('data', d);
}

// ---------------------------------------------------------------------------------------
// Interpolation datasets (noise-free samples of a known function)
// ---------------------------------------------------------------------------------------

export const rungeEquispaced = (() => {
  const x = linspace(-1.0, 1.0, 11);
  return dataset(
    'runge_equispaced',
    'Runge function, 11 equispaced nodes',
    x,
    x.map(rungeFn),
    rungeFn,
    'f(x) = \\frac{1}{1 + 25x^2}',
    [-1.0, 1.0],
    "Runge's example: the degree-10 interpolant on equispaced nodes oscillates wildly " +
      'near ±1 (max error ≈ 1.9), and the error grows with the degree.',
  );
})();

export const rungeChebyshev = (() => {
  const x = chebyshevNodesFirstKind(11);
  return dataset(
    'runge_chebyshev',
    'Runge function, 11 Chebyshev nodes',
    x,
    x.map(rungeFn),
    rungeFn,
    'f(x) = \\frac{1}{1 + 25x^2},\\ x_j = \\cos\\frac{(2j+1)\\pi}{22}',
    [-1.0, 1.0],
    'The same function sampled at the roots of T₁₁: the node polynomial is minimal in ' +
      'the max norm, and the interpolation error converges as the degree grows.',
  );
})();

export const sineSamples = (() => {
  const x = linspace(0.0, 2.0 * Math.PI, 9);
  return dataset(
    'sine_samples',
    'sin x, 9 equispaced samples',
    x,
    x.map(Math.sin),
    Math.sin,
    'f(x) = \\sin x',
    [0.0, 2.0 * Math.PI],
    'A smooth, analytic function: every interpolant converges quickly.',
  );
})();

export const stepData = (() => {
  const x = linspace(-1.0, 1.0, 12); // an even count: no node at the jump x = 0
  return dataset(
    'step_data',
    'Unit step (discontinuous)',
    x,
    x.map(unitStep),
    unitStep,
    'H(x) = \\begin{cases} 0 & x < 0 \\\\ 1 & x \\ge 0 \\end{cases}',
    [-1.0, 1.0],
    'Monotone data with a jump: polynomials and cubic splines overshoot (Gibbs-like), ' +
      'while PCHIP and the linear spline stay monotone. The jump has height 1, so the ' +
      'sup-norm error of any continuous interpolant is at least 1/2; the 200-point error ' +
      'grid has no point at x = 0, so the reported max errors of the splines are a ' +
      'little lower (about 0.46–0.48).',
  );
})();

// ---------------------------------------------------------------------------------------
// Regression datasets
// ---------------------------------------------------------------------------------------

export const noisyLinear = (() => {
  const x = linspace(0.0, 10.0, 20);
  const truth = line(2.0, 0.5);
  const noise = normalNoise(NOISY_LINEAR_SEED, x.length, 0.5);
  return dataset(
    'noisy_linear',
    'Noisy line',
    x,
    x.map((v, i) => truth(v) + noise[i]),
    truth,
    'y = 2 + 0.5x + \\varepsilon,\\ \\varepsilon \\sim N(0, 0.5^2)',
    [0.0, 10.0],
    'Twenty points on a line with Gaussian noise (Mulberry32 seed 42, Box–Muller).',
  );
})();

export const noisyQuadratic = (() => {
  const x = linspace(-2.0, 4.0, 25);
  const noise = normalNoise(NOISY_QUADRATIC_SEED, x.length, 0.4);
  return dataset(
    'noisy_quadratic',
    'Noisy parabola',
    x,
    x.map((v, i) => quadraticTruth(v) + noise[i]),
    quadraticTruth,
    'y = 1 - x + \\tfrac12 x^2 + \\varepsilon,\\ \\varepsilon \\sim N(0, 0.4^2)',
    [-2.0, 4.0],
    'Twenty-five points on a parabola with Gaussian noise (seed 7): a line underfits, ' +
      'degree 2 fits, high degrees overfit.',
  );
})();

/** Anscombe (1973), "Graphs in Statistical Analysis", Am. Stat. 27(1), Table 1, set I. */
export const ANSCOMBE_I_X = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
export const ANSCOMBE_I_Y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];

export const anscombe1 = dataset(
  'anscombe_1',
  "Anscombe's quartet I",
  ANSCOMBE_I_X,
  ANSCOMBE_I_Y,
  undefined,
  '\\hat y = 3.00 + 0.500\\,x,\\ R^2 = 0.67',
  [4.0, 14.0],
  'Anscombe (1973), set I, in the published order: the well-behaved member of the ' +
    'quartet. OLS gives ŷ = 3.0001 + 0.5001x with R² = 0.6665.',
);

export const outliersLinear = (() => {
  const x = linspace(0.0, 10.0, 20);
  const truth = line(1.0, 2.0);
  const noise = normalNoise(OUTLIERS_LINEAR_SEED, x.length, 0.3);
  const y = x.map((v, i) => truth(v) + noise[i]);
  for (const [i, offset] of OUTLIER_OFFSETS) y[i] += offset;
  return dataset(
    'outliers_linear',
    'Line with two gross outliers',
    x,
    y,
    truth,
    'y = 1 + 2x + \\varepsilon,\\ \\varepsilon \\sim N(0, 0.3^2)',
    [0.0, 10.0],
    'A clean line (seed 11) with y₅ raised by 15 and y₁₆ lowered by 20: ordinary least squares is ' +
      'pulled away, while Huber, LAD and Theil–Sen stay within the noise of the ' +
      'least-squares line through the 18 clean points.',
  );
})();

export const exponentialGrowth = (() => {
  const x = linspace(0.0, 10.0, 11);
  const noise = normalNoise(EXPONENTIAL_GROWTH_SEED, x.length, 0.05);
  return dataset(
    'exponential_growth',
    'Exponential growth',
    x,
    x.map((v, i) => 2.0 * Math.exp(0.3 * v + noise[i])),
    exponentialTruth,
    'y = 2\\,e^{0.3x + \\varepsilon},\\ \\varepsilon \\sim N(0, 0.05^2)',
    [0.0, 10.0],
    'Multiplicative noise on an exponential (seed 3): a straight line fits badly; ' +
      'a line fitted to log y recovers log 2 and 0.3.',
  );
})();

/** Every dataset, in the Python registration order. */
export const DATASETS: DataProblem[] = [
  rungeEquispaced,
  rungeChebyshev,
  sineSamples,
  stepData,
  noisyLinear,
  noisyQuadratic,
  anscombe1,
  outliersLinear,
  exponentialGrowth,
];
