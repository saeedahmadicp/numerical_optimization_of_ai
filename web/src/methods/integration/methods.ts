/**
 * Numerical integration (quadrature) of I = ∫ₐᵇ f(x) dx — TS port of
 * `numopt.integration.methods` (src/numopt/integration/methods.py).
 *
 * The interval is `problem.domain` unless the option `bracket = [a, b]` overrides it (a < b is
 * required). The exact value `problem.exact` is used only to report errors.
 *
 * Method groups (ids, params, stopping tests and Step.info keys exactly as in Python):
 *
 *   - composite rules (`left_riemann`, `right_riemann`, `midpoint_rule`, `trapezoid`, `simpson`,
 *     `simpson_38`, `boole`): step k is the rule on N_k = n·2^k subintervals, k = 0, …, levels;
 *     the error estimate is the observed-ratio Richardson estimate, and `converged` needs the
 *     asymptotic regime to be confirmed (de Boor's CADRE ratio tests);
 *   - `romberg`: step k is row k of the Romberg table (Burden & Faires, Alg. 4.2);
 *   - `gauss_legendre`: step k is the (k + 1)-point rule (Golub–Welsch nodes and weights);
 *   - `adaptive_simpson`: step k is the k-th interval accepted by the two-level Lyness test;
 *   - `monte_carlo_integration`: step k uses the first n·2^k uniform samples of `Rng(seed)`;
 *   - nested rules (`clenshaw_curtis`, `gauss_patterson`): step k is the Clenshaw–Curtis rule on
 *     n·2^k + 1 Chebyshev points, or the Gauss–Kronrod–Patterson rule on 2^{k+1} − 1 points;
 *     every node of step k − 1 is reused and d_k = |I_k − I_{k−1}| is the error estimate.
 *
 * Sums are correctly rounded (`fsum`, a port of CPython's `math.fsum`); when Python's fsum would
 * raise (inf + (−inf), intermediate overflow) the plain IEEE sum is used, as in Python. Function
 * values are cached by abscissa (exact float equality), so `nFev` counts distinct points.
 *
 * Info keys (snake_case, as in the Python docstring): estimate, error, err_est, ratio, confirmed,
 * h, n_panels, panels, nodes, weights (composite rules); estimate, error, err_est, ratio,
 * confirmed, h, n_panels, row, nodes (Romberg); estimate, error, err_est, ratio, n_points, nodes,
 * weights (Gauss); estimate, error, err_est, interval, pending, depth, tol_local, passed,
 * parent_passed, forced, nodes (adaptive Simpson); estimate, error, err_est, samples, n_samples
 * (Monte Carlo); estimate, error, err_est, n_points, nodes, weights, new_nodes, plus n and
 * cheb_coeffs for Clenshaw–Curtis (nested rules; nodes in decreasing x). Python `None` is `null`.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { Rng } from '../../core/rng';
import type { MethodFn, Problem, Result, RunOptions, Params, Step } from '../../core/types';

/** Upper bound on subintervals (or samples) per run, to keep traces and work bounded. */
export const MAX_POINTS = 2 ** 18;
export const MAX_SAMPLES = 2 ** 20;
/** The largest number of nodes (or samples) one Step stores for display. */
export const MAX_DISPLAY = 4096;

const EPS = Number.EPSILON;

type F = (x: number) => number;
type Pt = [number, number];

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

/**
 * CPython's `math.fsum` (Shewchuk's msum with the half-even correction). Returns `null` where
 * Python raises: ValueError (−inf + inf) or OverflowError (intermediate overflow).
 */
export function fsumExact(values: readonly number[]): number | null {
  const p: number[] = [];
  let n = 0;
  let specialSum = 0.0;
  let infSum = 0.0;
  for (const item of values) {
    let x = item;
    const xsave = x;
    let i = 0;
    for (let j = 0; j < n; j++) {
      let y = p[j];
      if (Math.abs(x) < Math.abs(y)) {
        const t = x;
        x = y;
        y = t;
      }
      const hi = x + y;
      const yr = hi - x;
      const lo = y - yr;
      if (lo !== 0.0) p[i++] = lo;
      x = hi;
    }
    n = i;
    if (x !== 0.0) {
      if (!Number.isFinite(x)) {
        // Intermediate overflow, or a nan/inf summand.
        if (Number.isFinite(xsave)) return null;
        if (xsave === Infinity || xsave === -Infinity) infSum += xsave;
        specialSum += xsave;
        n = 0;
      } else {
        p[n++] = x;
      }
    }
  }
  if (specialSum !== 0.0) {
    // NaN != 0 is true in C as well: a NaN summand makes the result NaN.
    if (Number.isNaN(infSum)) return null;
    return specialSum;
  }
  let hi = 0.0;
  if (n > 0) {
    hi = p[--n];
    let lo = 0.0;
    // Sum the partials from the top, stop when the sum becomes inexact.
    while (n > 0) {
      const x = hi;
      const y = p[--n];
      hi = x + y;
      const yr = hi - x;
      lo = y - yr;
      if (lo !== 0.0) break;
    }
    // Make half-even rounding work across multiple partials.
    if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
      const y = lo * 2.0;
      const x = hi + y;
      const yr = x - hi;
      if (y === yr) hi = x;
    }
  }
  return hi;
}

/** Correctly rounded Σ terms, with IEEE semantics when it cannot be (Python `_fsum`). */
export function fsum(values: readonly number[]): number {
  const s = fsumExact(values);
  if (s !== null) return s;
  let t = 0;
  for (const v of values) t += v;
  return t;
}

/** f(x) as a number; NaN when the evaluation throws (Python `_safe_eval`). */
function safeEval(f: F, x: number): number {
  try {
    const v = f(x);
    return typeof v === 'number' ? v : Number(v);
  } catch {
    return NaN;
  }
}

/** Counted evaluation of f (Python `Counted`). */
class Counted {
  n = 0;
  private readonly fn: F;
  constructor(fn: F) {
    this.fn = fn;
  }
  call = (x: number): number => {
    this.n += 1;
    return this.fn(x);
  };
}

/** Counted evaluation of f with a cache keyed by the abscissa (Python `_CachedF`). */
class CachedF {
  readonly counted: Counted;
  private readonly cache = new Map<number, number>();
  constructor(f: F) {
    this.counted = new Counted(f);
  }
  call = (x: number): number => {
    // Map keys use SameValueZero: −0 and +0 share an entry, as in a Python dict.
    let v = this.cache.get(x);
    if (v === undefined) {
      v = safeEval(this.counted.call, x);
      this.cache.set(x, v);
    }
    return v;
  };
  get n(): number {
    return this.counted.n;
  }
}

/** The integration problem's view the methods need (Python `scalar_problem`). */
interface ScalarProblem {
  id: string;
  f: F;
  domain: unknown;
  exact: number | null;
}

function scalarProblem(problem: Problem<number> | F): ScalarProblem {
  if (typeof problem === 'function')
    return { id: 'custom', f: problem, domain: [-1.0, 1.0], exact: null };
  if (!problem || typeof problem.f !== 'function')
    throw new TypeError('problem must be a numopt Problem or a callable f(x)');
  return {
    id: problem.id,
    f: problem.f as F,
    domain: problem.domain,
    exact: problem.exact ?? null,
  };
}

function resolveInterval(problem: ScalarProblem, bracket: unknown): [number, number] {
  const ab = (bracket ?? problem.domain) as unknown[] | null | undefined;
  if (!Array.isArray(ab) || ab.length !== 2)
    throw new Error(`${problem.id}: an integration interval (a, b) is required`);
  const a = Number(ab[0]),
    b = Number(ab[1]);
  if (!(Number.isFinite(a) && Number.isFinite(b)))
    throw new Error(`integration interval must be finite, got (${pyRepr(a)}, ${pyRepr(b)})`);
  if (!(a < b))
    throw new Error(`invalid integration interval: need a < b, got (${pyRepr(a)}, ${pyRepr(b)})`);
  return [a, b];
}

const exactOf = (p: ScalarProblem): number | null => (p.exact === null ? null : Number(p.exact));

const errorOf = (estimate: number, exact: number | null): number | null =>
  exact === null ? null : Math.abs(estimate - exact);

/** The documented acceptance test err_est ≤ tol·max(1, |estimate|). */
function passes(errEst: number | null, estimate: number, tol: number): boolean {
  return (
    errEst !== null &&
    Number.isFinite(errEst) &&
    Number.isFinite(estimate) &&
    errEst <= tol * Math.max(1.0, Math.abs(estimate))
  );
}

/**
 * Error estimate of the newest term of a sequence and the observed ratio ρ = d_{k−1}/d_k
 * (Python `_sequence_error`). `rate` is R = 2^p for a composite rule of order p, null for Gauss.
 */
export function sequenceError(
  diffs: readonly number[],
  noises: readonly number[],
  rate: number | null,
): [number, number | null] {
  const d = diffs[diffs.length - 1];
  const dPrev = diffs.length >= 2 ? diffs[diffs.length - 2] : null;
  const ratio = dPrev !== null && d > 0.0 ? dPrev / d : null;
  const base = rate !== null ? d / (rate - 1.0) : d;
  if (dPrev === null || (d <= noises[noises.length - 1] && dPrev <= noises[noises.length - 2]))
    return [base, ratio];
  let est: number;
  if (ratio !== null && ratio <= 1.0) {
    est = d;
  } else {
    const candidates = [base];
    if (ratio !== null) candidates.push(d / (ratio - 1.0));
    if (rate !== null) candidates.push(dPrev / (rate * (rate - 1.0)));
    est = pyMax(candidates);
  }
  if (rate === null) est = pyMax([est, ...diffs.slice(-3)]);
  return [est, ratio];
}

/** Python's max(): the first of equal maxima; NaN compares false (keeps the earlier value). */
function pyMax(values: readonly number[]): number {
  let m = values[0];
  for (let i = 1; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}

/** Two consecutive observed ratios "agree" when they differ by at most this factor. */
const RATIO_SPREAD = 2.0;
/** Number of observed ratios that must agree to confirm the asymptotic regime. */
const N_RATIOS = 3;

/** True when the newest terms of a sequence are in their asymptotic regime (Python `_confirmed`). */
export function confirmedRegime(deltas: readonly number[], noises: readonly number[]): boolean {
  const L = deltas.length;
  if (
    L >= 2 &&
    Math.abs(deltas[L - 1]) <= noises[L - 1] &&
    Math.abs(deltas[L - 2]) <= noises[L - 2]
  )
    return true;
  if (L < N_RATIOS + 1) return false;
  const last = deltas.slice(-(N_RATIOS + 1));
  if (!(last.every((d) => d > 0.0) || last.every((d) => d < 0.0))) return false;
  const rho: number[] = [];
  for (let i = 0; i + 1 < last.length; i++) rho.push(Math.abs(last[i]) / Math.abs(last[i + 1]));
  if (!rho.every((r) => r > 1.0)) return false;
  for (let i = 0; i + 1 < rho.length; i++) {
    const q = rho[i + 1] / rho[i];
    if (!(1.0 / RATIO_SPREAD <= q && q <= RATIO_SPREAD)) return false;
  }
  return true;
}

function checkInt(name: string, value: unknown, lo: number): number {
  const v = Number(value);
  if (!Number.isInteger(v) || v < lo)
    throw new Error(`${name} must be an integer ≥ ${lo}, got ${pyRepr(value)}`);
  return v;
}

function checkTol(tol: unknown): number {
  const t = Number(tol);
  if (!(Number.isFinite(t) && t > 0.0))
    throw new Error(`tol must be a positive finite number, got ${pyRepr(tol)}`);
  return t;
}

/** The first abscissa whose f value is not finite (null if all are finite). */
function firstBad(points: readonly Pt[]): number | null {
  for (const [x, fx] of points) if (!Number.isFinite(fx)) return x;
  return null;
}

function makeResult(
  method: string,
  estimate: number,
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  return {
    method,
    x: estimate,
    fun: estimate,
    converged,
    message,
    nIter,
    nFev,
    nGev: 0,
    nHev: 0,
    trace,
    extra,
  };
}

function nonfiniteResult(
  method: string,
  estimate: number,
  xBad: number | null,
  k: number,
  fc: { n: number },
  trace: Step[],
  extra: Record<string, unknown> = {},
): Result {
  const msg = xBad === null ? 'the estimate overflowed' : `f is not finite at x = ${pyG(xBad, 6)}`;
  return makeResult(method, estimate, false, msg, k, fc.n, trace, extra);
}

function step(k: number, estimate: number, stepSize: number | null, info: Step['info']): Step {
  return { k, x: estimate, fun: estimate, gradNorm: null, stepSize, info };
}

/** Python's `format(x, '.<p>g')`. */
export function pyG(x: number, p: number): string {
  if (Number.isNaN(x)) return 'nan';
  if (x === Infinity) return 'inf';
  if (x === -Infinity) return '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const exp = Number(x.toExponential(p - 1).split('e')[1]);
  if (exp < -4 || exp >= p) {
    const [mant, e] = x.toExponential(p - 1).split('e');
    const m = mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant;
    const en = Number(e);
    return `${m}e${en < 0 ? '-' : '+'}${String(Math.abs(en)).padStart(2, '0')}`;
  }
  const s = x.toFixed(Math.max(0, p - 1 - exp));
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** A rough Python repr for error messages. */
function pyRepr(v: unknown): string {
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
    return Number.isInteger(v) ? `${v}.0` : String(v);
  }
  if (typeof v === 'string') return `'${v}'`;
  return String(v);
}

// ---------------------------------------------------------------------------------------
// Composite Newton–Cotes rules (and Riemann sums)
// ---------------------------------------------------------------------------------------

/** Each panel of `span` subintervals contributes (num/den)·h·Σⱼ coeffs[j]·f(x_{start + j}). */
interface CompositeRule {
  span: number;
  coeffs: readonly number[];
  num: number;
  den: number;
  /** p: the error is O(h^p) for smooth f. */
  order: number;
  /** Exact for polynomials of degree ≤ degree. */
  degree: number;
  midpoint?: boolean;
}

export const RULES: Readonly<Record<string, CompositeRule>> = {
  left_riemann: { span: 1, coeffs: [1, 0], num: 1, den: 1, order: 1, degree: 0 },
  right_riemann: { span: 1, coeffs: [0, 1], num: 1, den: 1, order: 1, degree: 0 },
  midpoint_rule: { span: 1, coeffs: [1], num: 1, den: 1, order: 2, degree: 1, midpoint: true },
  trapezoid: { span: 1, coeffs: [1, 1], num: 1, den: 2, order: 2, degree: 1 },
  simpson: { span: 2, coeffs: [1, 4, 1], num: 1, den: 3, order: 4, degree: 3 },
  simpson_38: { span: 3, coeffs: [1, 3, 3, 1], num: 3, den: 8, order: 4, degree: 3 },
  boole: { span: 4, coeffs: [7, 32, 12, 32, 7], num: 2, den: 45, order: 6, degree: 5 },
};

/**
 * Nodes, weights and basic-rule panels of a composite rule on `nSub` subintervals. Grid nodes
 * are xᵢ = a + i·h for i < N and x_N = b exactly, so halving h reproduces every old node.
 */
export function compositeNodesWeights(
  method: string,
  a: number,
  b: number,
  nSub: number,
): [number[], number[], [number, number][]] {
  const rule = RULES[method];
  if (nSub % rule.span)
    throw new Error(`${method}: the number of subintervals must be a multiple of ${rule.span}`);
  const h = (b - a) / nSub;
  const grid = (i: number) => (i === nSub ? b : a + i * h);
  const m = rule.span;
  const panels: [number, number][] = [];
  for (let p = 0; p < nSub / m; p++) panels.push([grid(p * m), grid((p + 1) * m)]);
  if (rule.midpoint) {
    const nodes: number[] = [];
    for (let i = 0; i < nSub; i++) nodes.push(a + (i + 0.5) * h);
    return [nodes, new Array<number>(nSub).fill(h), panels];
  }
  const counts = new Array<number>(nSub + 1).fill(0);
  rule.coeffs.forEach((c, j) => {
    if (c) for (let i = j; i < nSub - m + j + 1; i += m) counts[i] += c;
  });
  const nodes: number[] = [];
  const weights: number[] = [];
  for (let i = 0; i <= nSub; i++) {
    if (!counts[i]) continue;
    nodes.push(grid(i));
    weights.push((h * (counts[i] * rule.num)) / rule.den);
  }
  return [nodes, weights, panels];
}

function composite(
  method: string,
  problem: Problem<number> | F,
  bracket: unknown,
  nIn: unknown,
  levelsIn: unknown,
  tolIn: unknown,
): Result {
  const rule = RULES[method];
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, bracket);
  const n = checkInt('n', nIn, rule.span);
  if (n % rule.span) throw new Error(`${method}: n must be a multiple of ${rule.span}, got ${n}`);
  const levels = checkInt('levels', levelsIn, 0);
  const tol = checkTol(tolIn);
  if (n * 2 ** levels > MAX_POINTS)
    throw new Error(`n·2^levels = ${n * 2 ** levels} exceeds the limit ${MAX_POINTS}`);
  const exact = exactOf(prob);
  const fc = new CachedF(prob.f);

  const trace: Step[] = [];
  let estimate = NaN;
  let errEst: number | null = null;
  let ratio: number | null = null;
  let confirmed = false;
  const diffs: number[] = [];
  const deltas: number[] = [];
  const noises: number[] = [];
  let massPrev = NaN;
  for (let k = 0; k <= levels; k++) {
    const nSub = n * 2 ** k;
    const h = (b - a) / nSub;
    const [nodes, weights, panels] = compositeNodesWeights(method, a, b, nSub);
    const values = nodes.map((x) => fc.call(x));
    const prev = estimate;
    estimate = fsum(weights.map((w, i) => w * values[i]));
    const mass = fsum(weights.map((w, i) => Math.abs(w * values[i])));
    const finite = Number.isFinite(estimate) && Number.isFinite(mass);
    if (k > 0 && finite) {
      deltas.push(estimate - prev);
      diffs.push(Math.abs(deltas[deltas.length - 1]));
      noises.push(4.0 * EPS * (mass + massPrev));
      [errEst, ratio] = sequenceError(diffs, noises, 2.0 ** rule.order);
      confirmed = confirmedRegime(deltas, noises);
    }
    massPrev = mass;
    const points: Pt[] = nodes.map((x, i) => [x, values[i]]);
    const shown = nodes.length <= MAX_DISPLAY;
    trace.push(
      step(k, estimate, h, {
        estimate,
        error: errorOf(estimate, exact),
        err_est: errEst,
        ratio,
        confirmed,
        h,
        n_panels: nSub,
        panels: shown ? panels : null,
        nodes: shown ? points : null,
        weights: shown ? weights : null,
      }),
    );
    if (!finite || !values.every((v) => Number.isFinite(v)))
      return nonfiniteResult(method, estimate, firstBad(points), k, fc, trace, { exact });
  }

  const extra = {
    exact,
    error: errorOf(estimate, exact),
    err_est: errEst,
    order: rule.order,
    degree: rule.degree,
    n_panels: n * 2 ** levels,
  };
  if (levels <= 1) {
    const msg =
      levels === 0
        ? 'levels = 0: a single estimate has no error estimate'
        : 'levels = 1: one difference cannot confirm the order p; use levels ≥ 2';
    return makeResult(method, estimate, false, msg, levels, fc.n, trace, extra);
  }
  const passed = passes(errEst, estimate, tol);
  const converged = passed && confirmed;
  const e = pyG(errEst as number, 3);
  let msg: string;
  if (converged) msg = `error estimate ${e} ≤ tol·max(1, |I|), asymptotic regime confirmed`;
  else if (passed)
    msg =
      `error estimate ${e} ≤ tol·max(1, |I|), but the asymptotic regime is not ` +
      'confirmed (the last 3 ratios ρ disagree or the differences change sign); ' +
      'increase levels';
  else msg = `error estimate ${e} > tol·max(1, |I|); increase n or levels`;
  return makeResult(method, estimate, converged, msg, levels, fc.n, trace, extra);
}

/** The largest `levels` of a composite rule (n_max·2^LEVELS_MAX ≤ MAX_POINTS). */
const LEVELS_MAX = 12;

function compositeParams(nDefault: number, nMin: number, nHelp: string, nMax = 64) {
  return [
    param.int('n', nDefault, {
      min: nMin,
      max: nMax,
      help: nHelp,
      label: 'Subintervals at step 0',
      tex: 'N_0',
    }),
    param.int('levels', 6, {
      min: 0,
      max: LEVELS_MAX,
      help: 'Number of halvings: step k uses n·2ᵏ subintervals.',
      label: 'Halvings',
      tex: 'K',
    }),
    param.float('tol', 1e-8, {
      min: 1e-15,
      max: 1e-1,
      log: true,
      help: 'Converged when the final error estimate ≤ tol·max(1, |I|).',
      label: 'Tolerance',
      tex: '\\varepsilon',
    }),
  ];
}

const N_HELP = 'Subintervals at step 0.';

type Opts = RunOptions & Params;

function compositeFn(id: string): MethodFn<Problem<number> | F> {
  return (problem, o: Opts) => composite(id, problem, o.bracket, o.n, o.levels, o.tol);
}

/** "This step" quantities of a composite rule whose estimate is written `sym`_N. */
const seqQuantities = (sym: string) => [
  { tex: 'N', key: 'info.n_panels' },
  { tex: 'h', key: 'stepSize' },
  { tex: `|${sym}_N - I|`, key: 'info.error' },
  { tex: '\\hat\\varepsilon_N', key: 'info.err_est' },
  { tex: '\\rho_k', key: 'info.ratio' },
];

export const DOCS: Record<string, MethodDoc> = {
  left_riemann: {
    rule: 'L_N = h\\sum_{i=0}^{N-1} f(x_i),\\qquad x_i = a + ih',
    intuition:
      'Each subinterval becomes a rectangle whose height is f at its left end. On a rising f every rectangle falls short, so the error is first order: halving h only halves it.',
    order: 'O(h)',
    pros: ['The simplest rule; exact for constants', 'Converges for any Riemann-integrable f'],
    cons: ['First order: each extra digit costs 10× the work', 'Biased below on increasing f'],
    quantities: seqQuantities('L'),
  },
  right_riemann: {
    rule: 'R_N = h\\sum_{i=1}^{N} f(x_i),\\qquad x_i = a + ih',
    intuition:
      'Rectangles take their height from the right end of each subinterval. The error has the opposite sign to the left sum, so the mean of the two is the trapezoidal rule.',
    order: 'O(h)',
    pros: ['Exact for constants', 'Its error mirrors the left sum'],
    cons: ['First order', 'Biased above on increasing f'],
    quantities: seqQuantities('R'),
  },
  midpoint_rule: {
    rule: 'M_N = h\\sum_{i=0}^{N-1} f\\!\\left(a + \\left(i + \\tfrac12\\right)h\\right)',
    intuition:
      'Sampling at the centre makes the linear part of the error cancel inside each subinterval, so the rule is second order with half the error constant of the trapezoid, and of opposite sign.',
    order: 'O(h²)',
    pros: ['Exact for degree ≤ 1', 'Open: never evaluates f at a or b'],
    cons: ['Halving h reuses no old node', 'At a kink, levels can agree on a wrong value'],
    quantities: seqQuantities('M'),
  },
  trapezoid: {
    rule: 'T_N = h\\left[\\tfrac12 f(x_0) + f(x_1) + \\cdots + f(x_{N-1}) + \\tfrac12 f(x_N)\\right]',
    intuition:
      'Join neighbouring samples with straight chords and add the trapezoids. The error is −(b − a)h²f″/12: chords lie above a convex f, so the rule overestimates it.',
    order: 'O(h²)',
    pros: [
      'Nested grids: halving h reuses every node',
      'Spectrally accurate for smooth periodic f',
    ],
    cons: ['Second order on non-periodic f', 'Loses its order at a kink'],
    quantities: seqQuantities('T'),
  },
  simpson: {
    rule: 'S_N = \\tfrac{h}{3}\\left[f_0 + 4f_1 + 2f_2 + 4f_3 + \\cdots + 4f_{N-1} + f_N\\right]',
    intuition:
      'Each pair of subintervals gets the parabola through its three samples, integrated exactly. By symmetry the cubic error term cancels as well, so the rule is fourth order.',
    order: 'O(h⁴)',
    pros: ['Exact for cubics', 'Simpson = (4T₂ₙ − Tₙ)/3: one Richardson step'],
    cons: ['Needs an even N', 'Drops to h^{3/2} on √x'],
    quantities: seqQuantities('S'),
  },
  simpson_38: {
    rule: 'S^{3/8}_N = \\tfrac{3h}{8}\\left[f_0 + 3f_1 + 3f_2 + 2f_3 + \\cdots + 3f_{N-1} + f_N\\right]',
    intuition:
      'Each group of three subintervals gets the cubic through its four samples. It has the same order as Simpson with a slightly larger constant, and it allows N to be a multiple of 3.',
    order: 'O(h⁴)',
    pros: ['Exact for cubics', 'Works when N is odd (multiple of 3)'],
    cons: ['Error constant 9/4 of Simpson’s per panel width', 'N must be a multiple of 3'],
    quantities: seqQuantities('S^{3/8}'),
  },
  boole: {
    rule: 'B_N = \\tfrac{2h}{45}\\left[7f_0 + 32f_1 + 12f_2 + 32f_3 + 14f_4 + \\cdots + 7f_N\\right]',
    intuition:
      'Each group of four subintervals gets the quartic through its five samples. Boole is one Richardson step above Simpson, (16S₂ₙ − Sₙ)/15, and gains two orders.',
    order: 'O(h⁶)',
    pros: ['Exact for degree ≤ 5', 'Sixth order on smooth f'],
    cons: ['Large alternating weights amplify noise', 'The order collapses at a singularity'],
    quantities: seqQuantities('B'),
  },
  romberg: {
    rule: 'R_{k,j} = R_{k,j-1} + \\frac{R_{k,j-1} - R_{k-1,j-1}}{4^{j} - 1}',
    intuition:
      'The trapezoid error is a series in h² (Euler–Maclaurin). Each column of the table eliminates one more power of h², so the diagonal R(k, k) is exact for degree 2k + 1.',
    order: 'O(h²ᵏ⁺²)',
    pros: ['Very fast on smooth f', 'Reuses every trapezoid node'],
    cons: ['Relies on the h² expansion: a singularity breaks it', 'Cost doubles per row'],
    quantities: [
      { tex: 'N = 2^k', key: 'info.n_panels' },
      { tex: 'h', key: 'stepSize' },
      { tex: '|R_{k,k} - I|', key: 'info.error' },
      { tex: '|R_{k,k} - R_{k-1,k-1}|', key: 'info.err_est' },
      { tex: '\\rho_k', key: 'info.ratio' },
    ],
  },
  gauss_legendre: {
    rule: 'G_m = \\frac{b-a}{2}\\sum_{i=1}^{m} w_i\\, f\\!\\left(\\frac{a+b}{2} + \\frac{b-a}{2}\\,t_i\\right)',
    intuition:
      'Free the nodes as well as the weights: with tᵢ the roots of the Legendre polynomial Pₘ, m points integrate every polynomial of degree 2m − 1 exactly. For analytic f the error falls geometrically in m.',
    order: 'spectral',
    pros: ['Optimal degree for m nodes', 'Geometric convergence for analytic f'],
    cons: [
      'Nodes are not nested: no reuse between m',
      'Only algebraic convergence at a singularity',
    ],
    quantities: [
      { tex: 'm', key: 'info.n_points' },
      { tex: '|G_m - I|', key: 'info.error' },
      { tex: '\\hat\\varepsilon_m', key: 'info.err_est' },
      { tex: '\\rho_m', key: 'info.ratio' },
    ],
  },
  adaptive_simpson: {
    rule: '|S_2 - S_1| \\le 15\\,\\tau \\;\\Rightarrow\\; \\int_\\alpha^\\beta f \\approx S_2 + \\frac{S_2 - S_1}{15}',
    intuition:
      'Compare Simpson on an interval with Simpson on its two halves. Where they disagree, bisect and halve the tolerance; where they agree on two levels, accept. Work concentrates where f is rough.',
    order: 'O(h⁴) locally',
    pros: ['Puts nodes where f needs them', 'Handles endpoint singularities'],
    cons: ['A narrow feature no node sees is missed', 'The error estimate is heuristic'],
    quantities: [
      { tex: '[\\alpha,\\beta]', key: 'info.interval' },
      { tex: '\\text{depth}', key: 'info.depth' },
      { tex: '\\tau', key: 'info.tol_local' },
      { tex: '|S_2 - S_1|/15', key: 'info.err_est' },
      { tex: '|I_k - I|', key: 'info.error' },
    ],
  },
  monte_carlo_integration: {
    rule: 'I_N = \\frac{b-a}{N}\\sum_{i=1}^{N} f(U_i),\\qquad U_i \\sim \\mathcal{U}(a, b)',
    intuition:
      'The integral is (b − a) times the mean of f. Average f at random points: the error is random, with standard deviation (b − a)σ_f/√N, whatever the smoothness of f.',
    order: 'O(1/√N)',
    pros: ['Indifferent to smoothness and dimension', 'Comes with a standard error'],
    cons: ['Very slow in one dimension', 'Each extra digit costs 100× the samples'],
    quantities: [
      { tex: 'N', key: 'info.n_samples' },
      { tex: '|I_N - I|', key: 'info.error' },
      { tex: '\\text{s.e.}', key: 'info.err_est' },
    ],
  },
};

const REF_BF = 'Burden & Faires, Numerical Analysis (10th ed.)';

registerMethod(
  {
    id: 'left_riemann',
    family: 'integration',
    name: 'Left Riemann sum',
    params: compositeParams(4, 1, N_HELP),
    needs: ['f', 'interval'],
    order: 'O(h); exact for constants',
    summary: 'Add up rectangles whose height is f at the left end of each subinterval.',
    references: [`${REF_BF}, §4.3–4.4`],
  },
  compositeFn('left_riemann'),
  DOCS.left_riemann,
);

registerMethod(
  {
    id: 'right_riemann',
    family: 'integration',
    name: 'Right Riemann sum',
    params: compositeParams(4, 1, N_HELP),
    needs: ['f', 'interval'],
    order: 'O(h); exact for constants',
    summary: 'Add up rectangles whose height is f at the right end of each subinterval.',
    references: [`${REF_BF}, §4.3–4.4`],
  },
  compositeFn('right_riemann'),
  DOCS.right_riemann,
);

registerMethod(
  {
    id: 'midpoint_rule',
    family: 'integration',
    name: 'Composite midpoint rule',
    params: compositeParams(4, 1, N_HELP),
    needs: ['f', 'interval'],
    order: 'O(h²); exact for degree ≤ 1',
    summary: 'Rectangles whose height is f at the center of each subinterval.',
    references: [`${REF_BF}, Theorem 4.6`],
  },
  compositeFn('midpoint_rule'),
  DOCS.midpoint_rule,
);

registerMethod(
  {
    id: 'trapezoid',
    family: 'integration',
    name: 'Composite trapezoidal rule',
    params: compositeParams(4, 1, N_HELP),
    needs: ['f', 'interval'],
    order: 'O(h²); exact for degree ≤ 1',
    summary: 'Join neighboring points with straight lines and add up the trapezoids.',
    references: [`${REF_BF}, Theorem 4.5`],
  },
  compositeFn('trapezoid'),
  DOCS.trapezoid,
);

registerMethod(
  {
    id: 'simpson',
    family: 'integration',
    name: "Composite Simpson's rule",
    params: compositeParams(4, 2, 'Subintervals at step 0 (must be even).'),
    needs: ['f', 'interval'],
    order: 'O(h⁴); exact for degree ≤ 3',
    summary: 'Fit a parabola through each pair of subintervals and integrate it exactly.',
    references: [`${REF_BF}, Theorem 4.4, Alg. 4.1`],
  },
  compositeFn('simpson'),
  DOCS.simpson,
);

registerMethod(
  {
    id: 'simpson_38',
    family: 'integration',
    name: "Composite Simpson's 3/8 rule",
    params: compositeParams(3, 3, 'Subintervals at step 0 (must be a multiple of 3).', 63),
    needs: ['f', 'interval'],
    order: 'O(h⁴); exact for degree ≤ 3',
    summary: 'Fit a cubic through each group of three subintervals and integrate it exactly.',
    references: [`${REF_BF}, §4.3 (closed Newton–Cotes, n = 3)`],
  },
  compositeFn('simpson_38'),
  DOCS.simpson_38,
);

registerMethod(
  {
    id: 'boole',
    family: 'integration',
    name: "Composite Boole's rule",
    params: compositeParams(4, 4, 'Subintervals at step 0 (must be a multiple of 4).'),
    needs: ['f', 'interval'],
    order: 'O(h⁶); exact for degree ≤ 5',
    summary: 'Fit a quartic through each group of four subintervals and integrate it exactly.',
    references: [`${REF_BF}, §4.3 (closed Newton–Cotes, n = 4)`],
  },
  compositeFn('boole'),
  DOCS.boole,
);

// ---------------------------------------------------------------------------------------
// Romberg integration
// ---------------------------------------------------------------------------------------

/** Row k uses 2^k subintervals; 2^18 = MAX_POINTS. */
const ROMBERG_LEVELS_MAX = 18;

const romberg: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const tol = checkTol(o.tol);
  const maxLevels = checkInt('max_levels', o.max_levels, 1);
  if (2 ** maxLevels > MAX_POINTS)
    throw new Error(
      `max_levels = ${maxLevels}: 2^max_levels exceeds the limit ${MAX_POINTS} ` +
        `(use max_levels ≤ ${ROMBERG_LEVELS_MAX})`,
    );
  const exact = exactOf(prob);
  const fc = new CachedF(prob.f);

  const fa = fc.call(a),
    fb = fc.call(b);
  let h = b - a;
  let row = [0.5 * h * (fa + fb)];
  let mass = 0.5 * h * (Math.abs(fa) + Math.abs(fb));
  const trace: Step[] = [
    step(0, row[0], h, {
      estimate: row[0],
      error: errorOf(row[0], exact),
      err_est: null,
      ratio: null,
      confirmed: false,
      h,
      n_panels: 1,
      row: [...row],
      nodes: [
        [a, fa],
        [b, fb],
      ],
    }),
  ];
  if (!(Number.isFinite(fa) && Number.isFinite(fb)))
    return nonfiniteResult('romberg', row[0], !Number.isFinite(fa) ? a : b, 0, fc, trace);
  if (!(Number.isFinite(row[0]) && Number.isFinite(mass)))
    return nonfiniteResult('romberg', row[0], null, 0, fc, trace);

  let errEst = Infinity;
  const deltas: number[] = [];
  const noises: number[] = [];
  let passedPrev = false;
  let passed = false;
  let confirmed = false;
  for (let k = 1; k <= maxLevels; k++) {
    const nSub = 2 ** k;
    h = (b - a) / nSub;
    const newX: number[] = [];
    for (let i = 1; i <= nSub / 2; i++) newX.push(a + (2 * i - 1) * h);
    const newF = newX.map((x) => fc.call(x));
    const prev = row,
      massPrev = mass;
    row = [0.5 * prev[0] + h * fsum(newF)];
    mass = 0.5 * massPrev + h * fsum(newF.map((v) => Math.abs(v)));
    for (let j = 1; j <= k; j++)
      row.push(row[j - 1] + (row[j - 1] - prev[j - 1]) / (4.0 ** j - 1.0));
    const estimate = row[k];
    errEst = Math.abs(estimate - prev[k - 1]);
    const finite = row.every((v) => Number.isFinite(v)) && Number.isFinite(mass);
    let ratio: number | null = null;
    if (finite) {
      deltas.push(row[0] - prev[0]);
      noises.push(4.0 * EPS * (mass + massPrev));
      if (k >= 2 && deltas[deltas.length - 1] !== 0.0)
        ratio = Math.abs(deltas[deltas.length - 2]) / Math.abs(deltas[deltas.length - 1]);
      confirmed = confirmedRegime(deltas, noises);
      passedPrev = passed;
      passed = k >= 2 && passes(errEst, estimate, tol);
    }
    const points: Pt[] = newX.map((x, i) => [x, newF[i]]);
    trace.push(
      step(k, estimate, h, {
        estimate,
        error: errorOf(estimate, exact),
        err_est: errEst,
        ratio,
        confirmed,
        h,
        n_panels: nSub,
        row: [...row],
        nodes: points.length <= MAX_DISPLAY ? points : null,
      }),
    );
    if (!newF.every((v) => Number.isFinite(v)))
      return nonfiniteResult('romberg', estimate, firstBad(points), k, fc, trace);
    if (!finite) return nonfiniteResult('romberg', estimate, null, k, fc, trace);
    if (passed && passedPrev && confirmed)
      return makeResult(
        'romberg',
        estimate,
        true,
        `|R(k,k) − R(k−1,k−1)| = ${pyG(errEst, 3)} ≤ tol·max(1, |I|) at k = ${k - 1} and ` +
          `k = ${k}, trapezoid column in its asymptotic regime`,
        k,
        fc.n,
        trace,
        { exact, error: errorOf(estimate, exact), err_est: errEst },
      );
  }
  const estimate = row[row.length - 1];
  let msg = `reached max_levels=${maxLevels} (|R(k,k) − R(k−1,k−1)| = ${pyG(errEst, 3)}`;
  if (passed && !confirmed) msg += '; the trapezoid column is not in its asymptotic regime';
  msg += ')';
  if (maxLevels < 3) msg += '; the stopping test needs rows k − 1 ≥ 2 and k, use max_levels ≥ 3';
  return makeResult('romberg', estimate, false, msg, maxLevels, fc.n, trace, {
    exact,
    error: errorOf(estimate, exact),
    err_est: errEst,
  });
};

registerMethod(
  {
    id: 'romberg',
    family: 'integration',
    name: 'Romberg integration',
    params: [
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-1,
        log: true,
        help:
          'Converged when |R(k,k) − R(k−1,k−1)| ≤ tol·max(1, |R(k,k)|) at two consecutive ' +
          'k ≥ 2 and the trapezoid column is in its asymptotic regime.',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
      param.int('max_levels', 16, {
        min: 3,
        max: ROMBERG_LEVELS_MAX,
        help: 'Maximum number of rows − 1 (row k uses 2ᵏ subintervals; tested from k = 3).',
        label: 'Rows',
        tex: 'k_{\\max}',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'O(h^{2k+2}) in row k for smooth f',
    summary: 'Extrapolate trapezoid estimates on halved grids to h = 0 (Richardson in h²).',
    references: [
      `${REF_BF}, Alg. 4.2`,
      'Dahlquist & Björck, Numerical Methods in Scientific Computing I (2008), §5.2.3',
      'de Boor, CADRE: an algorithm for numerical quadrature, in Rice (ed.), ' +
        'Mathematical Software (1971)',
    ],
  },
  romberg,
  DOCS.romberg,
);

// ---------------------------------------------------------------------------------------
// Gauss–Legendre quadrature
// ---------------------------------------------------------------------------------------

/**
 * Eigenvalues and first eigenvector components of the symmetric tridiagonal matrix with
 * diagonal `d` and off-diagonal `e` (e[i] couples i and i + 1): the implicit QL algorithm with
 * Wilkinson shifts (Numerical Recipes `tqli`), tracking only row 0 of the eigenvector matrix.
 * Returns the eigenvalues ascending, with their first components.
 */
function tridiagEig(dIn: readonly number[], offIn: readonly number[]): [number[], number[]] {
  const n = dIn.length;
  const d = [...dIn];
  const e = [...offIn, 0.0];
  const z = new Array<number>(n).fill(0.0);
  z[0] = 1.0;
  for (let l = 0; l < n; l++) {
    let iter = 0;
    let m: number;
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= EPS * dd) break;
      }
      if (m !== l) {
        if (iter++ === 60) break;
        let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        let r = Math.hypot(g, 1.0);
        g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1.0,
          c = 1.0,
          p = 0.0;
        let i: number;
        for (i = m - 1; i >= l; i--) {
          let f = s * e[i];
          const bb = c * e[i];
          e[i + 1] = r = Math.hypot(f, g);
          if (r === 0.0) {
            d[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * bb;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - bb;
          f = z[i + 1];
          z[i + 1] = s * z[i] + c * f;
          z[i] = c * z[i] - s * f;
        }
        if (r === 0.0 && i >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m !== l);
  }
  const order = d.map((_, i) => i).sort((p, q) => d[p] - d[q]);
  return [order.map((i) => d[i]), order.map((i) => z[i])];
}

/**
 * Nodes tᵢ and weights wᵢ of the n-point Gauss–Legendre rule on [−1, 1] (Golub–Welsch 1969):
 * the eigenvalues of the Jacobi matrix (zero diagonal, β_k = k/√(4k² − 1)) and wᵢ = 2·v₁ᵢ²,
 * symmetrized as in Python: tᵢ ← (tᵢ − t_{n+1−i})/2, wᵢ ← (wᵢ + w_{n+1−i})/2.
 */
export function gaussLegendreRule(nIn: number): [number[], number[]] {
  const n = checkInt('n', nIn, 1);
  const beta: number[] = [];
  for (let k = 1; k < n; k++) beta.push(k / Math.sqrt(4.0 * k * k - 1.0));
  const [theta, v0] = tridiagEig(new Array<number>(n).fill(0.0), beta);
  const w = v0.map((v) => 2.0 * v ** 2);
  const t = theta.map((th, i) => 0.5 * (th - theta[n - 1 - i]));
  const ws = w.map((wi, i) => 0.5 * (wi + w[n - 1 - i]));
  return [t, ws];
}

const gaussLegendre: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const n = checkInt('n', o.n, 1);
  const tol = checkTol(o.tol);
  const exact = exactOf(prob);
  const fc = new CachedF(prob.f);
  const half = 0.5 * (b - a),
    mid = 0.5 * (a + b);

  const trace: Step[] = [];
  let estimate = NaN;
  let errEst: number | null = null;
  let ratio: number | null = null;
  const diffs: number[] = [];
  const noises: number[] = [];
  let massPrev = NaN;
  for (let k = 0; k < n; k++) {
    const [t, w] = gaussLegendreRule(k + 1);
    const nodes = t.map((ti) => mid + half * ti);
    const weights = w.map((wi) => half * wi);
    const values = nodes.map((x) => fc.call(x));
    const prev = estimate;
    estimate = fsum(weights.map((wi, i) => wi * values[i]));
    const mass = fsum(weights.map((wi, i) => Math.abs(wi * values[i])));
    const finite = Number.isFinite(estimate) && Number.isFinite(mass);
    if (k > 0 && finite) {
      diffs.push(Math.abs(estimate - prev));
      noises.push(4.0 * EPS * (mass + massPrev));
      [errEst, ratio] = sequenceError(diffs, noises, null);
    }
    massPrev = mass;
    const points: Pt[] = nodes.map((x, i) => [x, values[i]]);
    trace.push(
      step(k, estimate, null, {
        estimate,
        error: errorOf(estimate, exact),
        err_est: errEst,
        ratio,
        n_points: k + 1,
        nodes: points,
        weights,
      }),
    );
    if (!finite || !values.every((v) => Number.isFinite(v)))
      return nonfiniteResult('gauss_legendre', estimate, firstBad(points), k, fc, trace, {
        exact,
      });
  }
  const extra = { exact, error: errorOf(estimate, exact), err_est: errEst };
  if (n <= 2) {
    const msg =
      n === 1
        ? 'n = 1: a single rule has no error estimate'
        : 'n = 2: one difference cannot confirm the convergence; use n ≥ 3';
    return makeResult('gauss_legendre', estimate, false, msg, n - 1, fc.n, trace, extra);
  }
  const converged = passes(errEst, estimate, tol);
  let msg = `error estimate ${pyG(errEst as number, 3)} ${converged ? '≤' : '>'} tol·max(1, |I|)`;
  if (!converged) msg += '; increase n';
  return makeResult('gauss_legendre', estimate, converged, msg, n - 1, fc.n, trace, extra);
};

registerMethod(
  {
    id: 'gauss_legendre',
    family: 'integration',
    name: 'Gauss–Legendre quadrature',
    params: [
      param.int('n', 10, {
        min: 1,
        max: 64,
        help: 'Number of Gauss points of the final rule; step k uses k + 1 points.',
        label: 'Gauss points',
        tex: 'm_{\\max}',
      }),
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-1,
        log: true,
        help: 'Converged when the error estimate from |G_n − G_{n−1}| ≤ tol·max(1, |G_n|).',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'exact for degree ≤ 2n − 1; spectral for analytic f',
    summary: 'Place n nodes at the roots of the Legendre polynomial so degree 2n − 1 is exact.',
    references: [
      'Golub & Welsch, Calculation of Gauss quadrature rules, Math. Comp. 23 (1969)',
      `${REF_BF}, §4.7`,
      'Trefethen, Approximation Theory and Approximation Practice (2013), Ch. 19',
    ],
  },
  gaussLegendre,
  DOCS.gauss_legendre,
);

// ---------------------------------------------------------------------------------------
// Adaptive Simpson
// ---------------------------------------------------------------------------------------

/** Stack entry: (alpha, beta, f_alpha, f_mid, f_beta, S1, tau, depth, parent_passed). */
type Entry = [number, number, number, number, number, number, number, number, boolean];

const adaptiveSimpson: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const tol = checkTol(o.tol);
  let minDepth = checkInt('min_depth', o.min_depth, 0);
  const maxDepth = checkInt('max_depth', o.max_depth, 1);
  minDepth = Math.min(minDepth, maxDepth);
  const maxIter = checkInt('max_iter', o.max_iter, 1);
  const exact = exactOf(prob);
  const fc = new CachedF(prob.f);
  const trace: Step[] = [];

  const failed = (msg: string, estimate: number, extra: Record<string, unknown>) =>
    makeResult('adaptive_simpson', estimate, false, msg, trace.length - 1, fc.n, trace, extra);

  const m = a + 0.5 * (b - a);
  const fa = fc.call(a),
    fm = fc.call(m),
    fb = fc.call(b);
  const sWhole = ((b - a) / 6.0) * (fa + 4.0 * fm + fb);
  const stack: Entry[] = [[a, b, fa, fm, fb, sWhole, tol, 0, true]];
  trace.push(
    step(0, sWhole, null, {
      estimate: sWhole,
      error: errorOf(sWhole, exact),
      err_est: null,
      interval: null,
      pending: [[a, b]],
      depth: 0,
      tol_local: tol,
      passed: null,
      parent_passed: null,
      forced: false,
      nodes: [
        [a, fa],
        [m, fm],
        [b, fb],
      ],
    }),
  );
  if (![fa, fm, fb].every((v) => Number.isFinite(v))) {
    const bad = firstBad([
      [a, fa],
      [m, fm],
      [b, fb],
    ]);
    return nonfiniteResult('adaptive_simpson', sWhole, bad, 0, fc, trace, { exact });
  }
  if (!Number.isFinite(sWhole))
    return nonfiniteResult('adaptive_simpson', sWhole, null, 0, fc, trace, { exact });

  const accepted: [number, number][] = [];
  const contributions: number[] = [];
  let errTotal = 0.0;
  let nForced = 0;
  let estimate = sWhole;
  while (stack.length) {
    const [alpha, beta, fAl, fMu, fBe, s1, tau, depth, parentPassed] = stack.pop()!;
    const mu = alpha + 0.5 * (beta - alpha);
    // The quarter points are computed exactly as each child computes its midpoint.
    const xL = alpha + 0.5 * (mu - alpha),
      xR = mu + 0.5 * (beta - mu);
    const fL = fc.call(xL),
      fR = fc.call(xR);
    if (!(Number.isFinite(fL) && Number.isFinite(fR))) {
      const bad = !Number.isFinite(fL) ? xL : xR;
      return nonfiniteResult('adaptive_simpson', estimate, bad, trace.length - 1, fc, trace, {
        exact,
      });
    }
    const sLeft = ((mu - alpha) / 6.0) * (fAl + 4.0 * fL + fMu);
    const sRight = ((beta - mu) / 6.0) * (fMu + 4.0 * fR + fBe);
    const s2 = sLeft + sRight;
    const diff = s2 - s1;
    if (!(Number.isFinite(sLeft) && Number.isFinite(sRight) && Number.isFinite(diff)))
      return nonfiniteResult('adaptive_simpson', estimate, null, trace.length - 1, fc, trace, {
        exact,
      });
    const unsplittable = !(alpha < xL && xL < mu && mu < xR && xR < beta);
    const passed = Math.abs(diff) <= 15.0 * tau;
    const ok = passed && parentPassed;
    if ((ok && depth >= minDepth) || depth >= maxDepth || unsplittable) {
      const forced = !ok;
      nForced += forced ? 1 : 0;
      contributions.push(s2 + diff / 15.0);
      accepted.push([alpha, beta]);
      errTotal += Math.abs(diff) / 15.0;
      estimate = fsum(contributions) + fsum(stack.map((s) => s[5]));
      trace.push(
        step(trace.length, estimate, beta - alpha, {
          estimate,
          error: errorOf(estimate, exact),
          err_est: Math.abs(diff) / 15.0,
          interval: [alpha, beta],
          pending: [...stack].reverse().map((s) => [s[0], s[1]]),
          depth,
          tol_local: tau,
          passed,
          parent_passed: parentPassed,
          forced,
          nodes: [
            [alpha, fAl],
            [xL, fL],
            [mu, fMu],
            [xR, fR],
            [beta, fBe],
          ],
        }),
      );
      const extra = {
        exact,
        error: errorOf(estimate, exact),
        err_est: errTotal,
        min_depth: minDepth,
        intervals: accepted,
      };
      if (!Number.isFinite(estimate)) return failed('the estimate overflowed', estimate, extra);
      if (accepted.length >= maxIter && stack.length)
        return failed(
          `reached max_iter=${maxIter} accepted intervals with ${stack.length} pending`,
          estimate,
          extra,
        );
    } else {
      stack.push([mu, beta, fMu, fR, fBe, sRight, 0.5 * tau, depth + 1, passed]);
      stack.push([alpha, mu, fAl, fL, fMu, sLeft, 0.5 * tau, depth + 1, passed]);
    }
  }

  const extra = {
    exact,
    error: errorOf(estimate, exact),
    err_est: errTotal,
    min_depth: minDepth,
    n_intervals: accepted.length,
    n_forced: nForced,
    intervals: accepted,
  };
  if (nForced)
    return failed(
      `${nForced} interval(s) accepted without passing the Lyness test on two levels ` +
        `(max_depth=${maxDepth} or too small to bisect)`,
      estimate,
      extra,
    );
  const msg =
    `all ${accepted.length} intervals and their parents passed |S₂ − S₁| ≤ 15τ ` +
    `(estimated error ${pyG(errTotal, 3)})`;
  return makeResult('adaptive_simpson', estimate, true, msg, trace.length - 1, fc.n, trace, extra);
};

registerMethod(
  {
    id: 'adaptive_simpson',
    family: 'integration',
    name: 'Adaptive Simpson',
    params: [
      param.float('tol', 1e-8, {
        min: 1e-14,
        max: 1e-1,
        log: true,
        help: 'Absolute error target; each half of an interval gets half the tolerance.',
        label: 'Tolerance',
        tex: '\\tau_0',
      }),
      param.int('min_depth', 2, {
        min: 0,
        max: 10,
        help: 'Bisect at least this deep before an interval may be accepted (≤ max_depth).',
        label: 'Minimum depth',
      }),
      param.int('max_depth', 30, {
        min: 1,
        max: 50,
        help: 'Maximum bisection depth; deeper intervals are accepted and flagged.',
        label: 'Maximum depth',
      }),
      param.int('max_iter', 1000, {
        min: 1,
        max: 10_000,
        help: 'Maximum number of accepted intervals (trace steps).',
        label: 'Interval budget',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'O(h⁴) locally; work concentrates where f is rough',
    summary: 'Bisect only the intervals where two Simpson estimates disagree.',
    references: [
      'Lyness, Notes on the adaptive Simpson quadrature routine, J. ACM 16 (1969)',
      `${REF_BF}, Alg. 4.3`,
    ],
  },
  adaptiveSimpson,
  DOCS.adaptive_simpson,
);

// ---------------------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------------------

const monteCarlo: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const n = checkInt('n', o.n, 1);
  const levels = checkInt('levels', o.levels, 0);
  const tol = checkTol(o.tol);
  if (n * 2 ** levels > MAX_SAMPLES)
    throw new Error(`n·2^levels = ${n * 2 ** levels} exceeds the limit ${MAX_SAMPLES}`);
  const exact = exactOf(prob);
  const rng = new Rng(o.seed ?? 0);
  const fc = new Counted(prob.f);
  const width = b - a;

  let count = 0,
    mean = 0.0,
    m2 = 0.0;
  const trace: Step[] = [];
  let estimate = NaN;
  let stdErr: number | null = null;
  let xBad: number | null = null;
  for (let k = 0; k <= levels; k++) {
    const target = n * 2 ** k;
    const samples: Pt[] = [];
    while (count < target) {
      const x = rng.uniform(a, b);
      const fx = safeEval(fc.call, x);
      if (samples.length < MAX_DISPLAY) samples.push([x, fx]);
      if (xBad === null && !Number.isFinite(fx)) xBad = x;
      count += 1;
      const delta = fx - mean;
      mean += delta / count;
      m2 += delta * (fx - mean);
    }
    estimate = width * mean;
    stdErr = count > 1 ? width * Math.sqrt(m2 / (count - 1) / count) : null;
    trace.push(
      step(k, estimate, null, {
        estimate,
        error: errorOf(estimate, exact),
        err_est: stdErr,
        samples,
        n_samples: count,
      }),
    );
    if (!Number.isFinite(estimate))
      return nonfiniteResult('monte_carlo_integration', estimate, xBad, k, fc, trace, { exact });
  }
  const extra = { exact, error: errorOf(estimate, exact), err_est: stdErr, n_samples: count };
  let msg: string;
  let converged: boolean;
  if (stdErr === null) {
    msg = 'one sample: no standard error';
    converged = false;
  } else {
    converged = passes(stdErr, estimate, tol);
    msg = `standard error ${pyG(stdErr, 3)} ${converged ? '≤' : '>'} tol·max(1, |I|) with N = ${count}`;
  }
  return makeResult(
    'monte_carlo_integration',
    estimate,
    converged,
    msg,
    levels,
    fc.n,
    trace,
    extra,
  );
};

registerMethod(
  {
    id: 'monte_carlo_integration',
    family: 'integration',
    name: 'Monte Carlo integration',
    params: [
      param.int('n', 100, {
        min: 1,
        max: 128,
        help: 'Samples at step 0.',
        label: 'Samples at step 0',
        tex: 'N_0',
      }),
      param.int('levels', 6, {
        min: 0,
        max: 13,
        help: 'Number of doublings: step k uses n·2ᵏ samples.',
        label: 'Doublings',
        tex: 'K',
      }),
      param.float('tol', 1e-2, {
        min: 1e-6,
        max: 1.0,
        log: true,
        help: 'Converged when the standard error ≤ tol·max(1, |I|).',
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'O(N^{-1/2}) in probability',
    summary: 'Average f at uniformly random points and multiply by the interval length.',
    references: [
      'Owen, Monte Carlo theory, methods and examples (2013), Ch. 2',
      'Welford, Technometrics 4 (1962) (running variance)',
    ],
    deterministic: false,
  },
  monteCarlo,
  DOCS.monte_carlo_integration,
);

// ---------------------------------------------------------------------------------------
// Nested rules with a free error estimate: Clenshaw–Curtis and Gauss–Kronrod–Patterson
// ---------------------------------------------------------------------------------------
//
// Both methods apply a sequence of rules whose node sets are nested (every node of rule k is a
// node of rule k + 1), so a run that stops at level K costs only the nodes of rule K, and
// d_k = |I_k − I_{k−1}| is a free error estimate. Both use the stopping test of the study
// research/clenshaw-curtis-vs-gauss: converged at the first k ≥ 2 with d_k ≤ tol·max(1, |I_k|).
// The Python section comment measures the limits of this test (abs_kink, narrow peaks).

/** Upper bound on the degree n·2^max_levels of a Clenshaw–Curtis rule (= MAX_POINTS). */
const CC_MAX_DEGREE = MAX_POINTS;
/** The largest `max_levels` of Clenshaw–Curtis; n ≤ CC_N_MAX keeps n·2^max_levels ≤ the limit. */
const CC_LEVELS_MAX = 12;
const CC_N_MAX = CC_MAX_DEGREE >> CC_LEVELS_MAX;

/**
 * Discrete Fourier transform X_k = Σ_j x_j·exp(sign·2πi·jk/N) of a complex sequence (re, im).
 * Radix-2 decimation in time down to the odd factor of N, which is transformed directly; the
 * Clenshaw–Curtis lengths are n·2^k with n ≤ 64, so the direct part costs at most 63 per point.
 * Twiddles come from one table cos/sin(2πj/N) of the full length (an O(ε) error per factor,
 * as in any FFT; the parity tests compare at 1e-9).
 */
export function dft(
  re: readonly number[],
  im: readonly number[],
  sign: 1 | -1,
): [number[], number[]] {
  const N = re.length;
  if (N === 0) return [[], []];
  const cosT = new Float64Array(N);
  const sinT = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    const th = (2 * Math.PI * j) / N;
    cosT[j] = Math.cos(th);
    sinT[j] = sign * Math.sin(th);
  }
  // rec(offset, stride, n): the length-n DFT of x[offset + stride·j], j < n (n divides N).
  const rec = (offset: number, stride: number, n: number): [Float64Array, Float64Array] => {
    const outR = new Float64Array(n);
    const outI = new Float64Array(n);
    const step = N / n; // the twiddle exp(sign·2πi·k/n) is table entry k·step
    if (n % 2 === 1) {
      for (let k = 0; k < n; k++) {
        let sr = 0,
          si = 0;
        for (let j = 0; j < n; j++) {
          const t = ((j * k) % n) * step;
          const xr = re[offset + stride * j],
            xi = im[offset + stride * j];
          sr += xr * cosT[t] - xi * sinT[t];
          si += xr * sinT[t] + xi * cosT[t];
        }
        outR[k] = sr;
        outI[k] = si;
      }
      return [outR, outI];
    }
    const h = n / 2;
    const [er, ei] = rec(offset, 2 * stride, h);
    const [or, oi] = rec(offset + stride, 2 * stride, h);
    for (let k = 0; k < h; k++) {
      const t = k * step;
      const wr = cosT[t],
        wi = sinT[t];
      const tr = wr * or[k] - wi * oi[k];
      const ti = wr * oi[k] + wi * or[k];
      outR[k] = er[k] + tr;
      outI[k] = ei[k] + ti;
      outR[k + h] = er[k] - tr;
      outI[k + h] = ei[k] - ti;
    }
    return [outR, outI];
  };
  const [r, i] = rec(0, 1, N);
  return [Array.from(r), Array.from(i)];
}

/**
 * Chebyshev extreme points t_k = cos(kπ/n), k = 0, …, n (decreasing), on [−1, 1], computed as
 * sin(π(n − 2k)/(2n)) as in Python: exactly antisymmetric, t_0 = 1, t_n = −1, and t_k of n is
 * t_{2k} of 2n bit for bit (the scalings by 2 of the argument are exact), so the evaluation
 * cache is exact on nested grids.
 */
export function clenshawCurtisNodes(nIn: number): number[] {
  const n = checkInt('n', nIn, 1);
  const t: number[] = [];
  for (let k = 0; k <= n; k++) t.push(Math.sin((Math.PI * (n - 2 * k)) / (2.0 * n)));
  return t;
}

/**
 * Clenshaw–Curtis weights w_0, …, w_n on [−1, 1] for the nodes cos(kπ/n) (Python
 * `clenshaw_curtis_weights`): Waldvogel (2006), §5, w = F_n⁻¹(v + g) with v of (3.10), g of
 * (4.2) and w_0 = 1/(n² − 1 + n mod 2) of (2.6), as in Waldvogel's `fejer.m`; then w_n := w_0
 * and the weights are symmetrized, w_k ← (w_k + w_{n−k})/2. n = 1 is the trapezoid rule (1, 1).
 */
export function clenshawCurtisWeights(nIn: number): number[] {
  const n = checkInt('n', nIn, 1);
  if (n === 1) return [1.0, 1.0];
  const odd: number[] = [];
  for (let j = 1; j < n; j += 2) odd.push(j);
  const nOdd = odd.length;
  const nRest = n - nOdd;
  const v0 = [...odd.map((o) => 2.0 / o / (o - 2.0)), 1.0 / odd[nOdd - 1]];
  for (let j = 0; j < nRest; j++) v0.push(0.0);
  // v2[i] = −v0[i] − v0[n − i] (Python: −v0[:−1] − v0[:0:−1]).
  const v2: number[] = [];
  for (let i = 0; i < n; i++) v2.push(-v0[i] - v0[n - i]);
  const g0 = new Array<number>(n).fill(-1.0);
  g0[nOdd] += n;
  g0[nRest] += n;
  const den = n * n - 1 + (n % 2);
  const re = v2.map((v, i) => v + g0[i] / den);
  // numpy's ifft: (1/n)·Σ x_j·exp(+2πi·jk/n); only the real part is kept.
  const [wr] = dft(re, new Array<number>(n).fill(0.0), 1);
  const w = wr.map((v) => v / n);
  w.push(w[0]);
  return w.map((wk, k) => 0.5 * (wk + w[n - k]));
}

/** Nodes t_k (decreasing) and weights w_k of the (n + 1)-point Clenshaw–Curtis rule on [−1, 1]. */
export function clenshawCurtisRule(n: number): [number[], number[]] {
  return [clenshawCurtisNodes(n), clenshawCurtisWeights(n)];
}

/**
 * Coefficients a_0, …, a_n of the interpolant p = Σ a_j T_j through (cos(kπ/n), values[k]):
 * the DCT-I by an FFT of the even extension (Python `chebyshev_coefficients`; Trefethen 2008,
 * §2): g = Re FFT([f_0, …, f_n, f_{n−1}, …, f_1])/(2n), a_0 = g_0, a_j = 2g_j, a_n = g_n.
 */
export function chebyshevCoefficients(values: readonly number[]): number[] {
  const n = values.length - 1;
  if (n < 1) throw new Error('chebyshev_coefficients needs at least two values');
  const ext = [...values];
  for (let k = n - 1; k >= 1; k--) ext.push(values[k]);
  const [gr] = dft(ext, new Array<number>(2 * n).fill(0.0), -1);
  const g = gr.map((v) => v / (2.0 * n));
  const a = g.slice(0, n + 1).map((v) => 2.0 * v);
  a[0] = g[0];
  a[n] = g[n];
  return a;
}

// Gauss–Kronrod–Patterson rules on [−1, 1]: levels 0, …, 6 with N_k = 2^{k+1} − 1 = 1, 3, 7,
// 15, 31, 63, 127 points (midpoint, Gauss G₃, Kronrod K₇, then Patterson's extensions). The
// tables are the Python `_GKP_NODES` / `_GKP_WEIGHTS` (nearest doubles of the binary128 and
// 300-digit computations of research/clenshaw-curtis-vs-gauss), copied digit for digit.
// GKP_NODES holds the 63 positive nodes of the 127-point rule, decreasing; level k uses the
// entries i with (i + 1) divisible by 2^{6−k}. GKP_WEIGHTS[k] holds the weights of the 2^k
// non-negative nodes of level k, in decreasing order of the node (the last is at t = 0).
const GKP_MAX_LEVEL = 6;
const GKP_NODES: readonly number[] = [
  0.9999824303548916, 0.9998728881203576, 0.9995987996719107, 0.9990981249676676,
  0.9983166353184074, 0.997206259372222, 0.9957241046984072, 0.993831963212755, 0.9914957211781061,
  0.9886847575474295, 0.9853714995985203, 0.9815311495537401, 0.9771415146397057,
  0.9721828747485818, 0.9666378515584165, 0.9604912687080203, 0.9537300064257611,
  0.9463428583734029, 0.9383203977795929, 0.9296548574297401, 0.9203400254700124,
  0.9103711569570043, 0.89974489977694, 0.888459232872257, 0.8765134144847053, 0.8639079381936905,
  0.8506444947683502, 0.8367259381688688, 0.8221562543649804, 0.8069405319502176,
  0.7910849337998483, 0.7745966692414834, 0.7574839663805136, 0.7397560443526947,
  0.7214230853700989, 0.7024962064915271, 0.6829874310910792, 0.6629096600247806,
  0.6422766425097595, 0.6211029467372264, 0.5994039302422429, 0.5771957100520458,
  0.5544951326319325, 0.5313197436443756, 0.5076877575337166, 0.48361802694584105,
  0.45913001198983233, 0.43424374934680254, 0.4089798212298887, 0.38335932419873037,
  0.3574038378315322, 0.3311353932579768, 0.30457644155671404, 0.2777498220218243,
  0.2506787303034832, 0.2233866864289669, 0.19589750271110015, 0.16823525155220748,
  0.14042423315256017, 0.11248894313318662, 0.08445404008371088, 0.05634431304659279,
  0.028184648949745695,
];
const GKP_WEIGHTS: readonly (readonly number[])[] = [
  [2.0],
  [0.5555555555555556, 0.8888888888888888],
  [0.10465622602646726, 0.26848808986833345, 0.40139741477596225, 0.45091653865847414],
  [
    0.01700171962994026, 0.05160328299707974, 0.09292719531512454, 0.13441525524378423,
    0.1715119091363914, 0.20062852937698902, 0.2191568584015875, 0.2255104997982067,
  ],
  [
    0.0025447807915618746, 0.008434565739321106, 0.01644604985438781, 0.025807598096176654,
    0.03595710330712932, 0.04646289326175799, 0.05697950949412336, 0.0672077542959907,
    0.07687962049900353, 0.08575592004999034, 0.09362710998126447, 0.10031427861179558,
    0.1056698935802348, 0.10957842105592464, 0.11195687302095346, 0.11275525672076869,
  ],
  [
    0.00036322148184553065, 0.001265156556230068, 0.0025790497946856883, 0.004217630441558855,
    0.006115506822117246, 0.00822300795723593, 0.010498246909621322, 0.012903800100351265,
    0.015406750466559498, 0.01797855156812827, 0.02059423391591271, 0.02323144663991027,
    0.025869679327214748, 0.02848975474583355, 0.031073551111687966, 0.03360387714820773,
    0.03606443278078257, 0.03843981024945553, 0.04071551011694432, 0.04287796002500773,
    0.0449145316536322, 0.04681355499062801, 0.0485643304066732, 0.05015713930589954,
    0.051583253952048456, 0.05283494679011652, 0.05390549933526606, 0.054789210527962866,
    0.05548140435655936, 0.05597843651047632, 0.0562776998312543, 0.056377628360384714,
  ],
  [
    5.053609520786252e-5, 0.00018073956444538837, 0.00037774664632698465, 0.0006326073193626335,
    0.0009383698485423815, 0.0012895240826104174, 0.00168114286542147, 0.002108815245726633,
    0.0025687649437940202, 0.003057753410175531, 0.0035728927835172995, 0.004111503978654693,
    0.0046710503721143215, 0.0052491234548088595, 0.00584344987583564, 0.006451900050175737,
    0.007072489995433555, 0.007703375233279742, 0.008342838753968157, 0.008989275784064136,
    0.009641177729702537, 0.010297116957956355, 0.010955733387837901, 0.011615723319955135,
    0.01227583056008277, 0.012934839663607374, 0.013591571009765546, 0.014244877372916775,
    0.014893641664815181, 0.015536775555843983, 0.01617321872957772, 0.016801938574103864,
    0.017421930159464173, 0.018032216390391285, 0.01863184825613879, 0.019219905124727765,
    0.019795495048097498, 0.02035775505847216, 0.020905851445812022, 0.021438980012503866,
    0.021956366305317825, 0.0224572658268161, 0.02294096422938775, 0.023406777495314005,
    0.02385405210603854, 0.0242821652033366, 0.024690524744487678, 0.02507856965294977,
    0.025445769965464767, 0.025791626976024228, 0.0261156733767061, 0.02641747339505826,
    0.02669662292745036, 0.02695274966763303, 0.027185513229624793, 0.027394605263981433,
    0.027579749566481872, 0.02774070217827968, 0.027877251476613702, 0.02798921825523816,
    0.028076455793817248, 0.02813884991562715, 0.0281763190330166, 0.028188814180192357,
  ],
];

/**
 * Nodes t_i (decreasing) and weights w_i of the Gauss–Kronrod–Patterson rule of `level`
 * (0, …, 6; N = 2^{level+1} − 1 points) on [−1, 1]. Exact for polynomials of degree
 * ≤ 3·2^level − 1 (level ≥ 1; degree 1 at level 0).
 */
export function gaussPattersonRule(levelIn: number): [number[], number[]] {
  const level = checkInt('level', levelIn, 0);
  if (level > GKP_MAX_LEVEL)
    throw new Error(`level must be ≤ ${GKP_MAX_LEVEL} (127 points), got ${level}`);
  const stride = 2 ** (GKP_MAX_LEVEL - level);
  const pos: number[] = [];
  for (let i = stride - 1; i < GKP_NODES.length; i += stride) pos.push(GKP_NODES[i]);
  const half = GKP_WEIGHTS[level];
  const t = [...pos, 0.0, ...pos.map((p) => -p).reverse()];
  const w = [...half, ...half.slice(0, -1).reverse()];
  return [t, w];
}

/**
 * The shared driver of the nested rules (Python `_nested_quadrature`): step k applies rule(k)
 * mapped to [a, b], I_k = Σᵢ ((b − a)/2)·wᵢ·f((a + b)/2 + ((b − a)/2)·tᵢ). `closed` rules have
 * t_0 = 1 and t_last = −1, which are mapped to b and a exactly. Converged at the first k ≥ 2
 * with d_k = |I_k − I_{k−1}| ≤ tol·max(1, |I_k|); converged = false after `maxLevels` or on a
 * non-finite value. `levelInfo(k, values)` adds the method-specific Info keys.
 */
function nestedQuadrature(
  method: string,
  prob: ScalarProblem,
  a: number,
  b: number,
  maxLevels: number,
  tol: number,
  rule: (k: number) => [number[], number[]],
  closed: boolean,
  levelInfo: (k: number, values: number[]) => Record<string, unknown>,
): Result {
  const exact = exactOf(prob);
  const fc = new CachedF(prob.f);
  const half = 0.5 * (b - a),
    mid = 0.5 * (a + b);

  const trace: Step[] = [];
  let estimate = NaN;
  let errEst: number | null = null;
  let nPoints = 0;
  for (let k = 0; k <= maxLevels; k++) {
    const [t, w] = rule(k);
    // The same doubles for a shared t, so the cache is exact on nested grids.
    const nodes = t.map((ti) => mid + half * ti);
    if (closed) {
      nodes[0] = b;
      nodes[nodes.length - 1] = a;
    }
    const weights = w.map((wi) => half * wi);
    nPoints = nodes.length;
    const nBefore = fc.n;
    const values = nodes.map((x) => fc.call(x));
    const prev = estimate;
    estimate = fsum(weights.map((wi, i) => wi * values[i]));
    if (k > 0) errEst = Math.abs(estimate - prev);
    const finite = Number.isFinite(estimate) && values.every((v) => Number.isFinite(v));
    const shown = nPoints <= MAX_DISPLAY;
    const points: Pt[] = nodes.map((x, i) => [x, values[i]]);
    trace.push(
      step(k, estimate, null, {
        estimate,
        error: errorOf(estimate, exact),
        err_est: errEst,
        n_points: nPoints,
        nodes: shown ? points : null,
        weights: shown ? weights : null,
        new_nodes: fc.n - nBefore,
        ...levelInfo(k, finite && shown ? values : []),
      }),
    );
    if (!finite)
      return nonfiniteResult(method, estimate, firstBad(points), k, fc, trace, {
        exact,
        error: errorOf(estimate, exact),
        err_est: errEst,
        n_points: nPoints,
      });
    if (k >= 2 && passes(errEst, estimate, tol))
      return makeResult(
        method,
        estimate,
        true,
        `|I_k − I_(k−1)| = ${pyG(errEst as number, 3)} ≤ tol·max(1, |I|) with ${nPoints} points`,
        k,
        fc.n,
        trace,
        { exact, error: errorOf(estimate, exact), err_est: errEst, n_points: nPoints },
      );
  }
  return makeResult(
    method,
    estimate,
    false,
    `reached max_levels=${maxLevels} (${nPoints} points) with ` +
      `|I_k − I_(k−1)| = ${pyG(errEst as number, 3)} > tol·max(1, |I|)`,
    maxLevels,
    fc.n,
    trace,
    { exact, error: errorOf(estimate, exact), err_est: errEst, n_points: nPoints },
  );
}

const NESTED_TOL_HELP = 'Converged at the first k ≥ 2 with |I_k − I_(k−1)| ≤ tol·max(1, |I_k|).';

const clenshawCurtis: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const n = checkInt('n', o.n, 1);
  const maxLevels = checkInt('max_levels', o.max_levels, 2);
  const tol = checkTol(o.tol);
  if (n * 2 ** maxLevels > CC_MAX_DEGREE)
    throw new Error(`n·2^max_levels = ${n * 2 ** maxLevels} exceeds the limit ${CC_MAX_DEGREE}`);
  return nestedQuadrature(
    'clenshaw_curtis',
    prob,
    a,
    b,
    maxLevels,
    tol,
    (k) => clenshawCurtisRule(n * 2 ** k),
    true,
    (k, values) => ({
      n: n * 2 ** k,
      cheb_coeffs: values.length ? chebyshevCoefficients(values) : null,
    }),
  );
};

const gaussPatterson: MethodFn<Problem<number> | F> = (problem, o: Opts) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveInterval(prob, o.bracket);
  const maxLevels = checkInt('max_levels', o.max_levels, 2);
  if (maxLevels > GKP_MAX_LEVEL)
    throw new Error(`max_levels must be ≤ ${GKP_MAX_LEVEL} (127 points), got ${maxLevels}`);
  const tol = checkTol(o.tol);
  return nestedQuadrature(
    'gauss_patterson',
    prob,
    a,
    b,
    maxLevels,
    tol,
    gaussPattersonRule,
    false,
    () => ({}),
  );
};

/** Docs of the nested rules (the MethodCard). */
export const NESTED_DOCS: Record<string, MethodDoc> = {
  clenshaw_curtis: {
    rule: 'C_n = \\frac{b-a}{2}\\sum_{j=0}^{n} w_j\\, f(x_j),\\qquad x_j = \\frac{a+b}{2} + \\frac{b-a}{2}\\cos\\frac{j\\pi}{n}',
    intuition:
      'Sample f at the Chebyshev points, the shadows of equally spaced points on a semicircle, and integrate the polynomial through them exactly. Doubling n keeps every old point, so each level costs only the new half and the change |C_n − C_{n/2}| is a free error estimate.',
    order: 'close to Gauss for the same points',
    pros: [
      'Nested: doubling n reuses every old value of f',
      'Weights in O(n log n) by one FFT (Waldvogel)',
    ],
    cons: [
      'Exact only to degree n (Gauss: 2n + 1)',
      'The stopping test can accept a wrong value at a kink',
    ],
    quantities: [
      { tex: 'n', key: 'info.n', label: 'degree' },
      { tex: 'n + 1', key: 'info.n_points', label: 'nodes' },
      { tex: '\\text{new}', key: 'info.new_nodes', label: 'new nodes' },
      { tex: '|C_n - I|', key: 'info.error' },
      { tex: '|C_n - C_{n/2}|', key: 'info.err_est' },
    ],
  },
  gauss_patterson: {
    rule: 'P_N = \\frac{b-a}{2}\\sum_{i=1}^{N} w_i\\, f\\!\\left(\\frac{a+b}{2} + \\frac{b-a}{2}\\,t_i\\right),\\qquad N = 2^{k+1} - 1',
    intuition:
      'Start from Gauss with 3 points, then add N + 1 new points between the old ones, placed so that the rule on all of them has the highest degree. Every old value of f is reused, and the change between two levels is a free error estimate.',
    order: 'exact for degree ≤ (3N + 1)/2',
    pros: [
      'Nested, like Clenshaw–Curtis, with a higher degree',
      'Open: never evaluates f at a or b',
    ],
    cons: [
      'Only 7 levels (127 points) can be computed in double precision',
      'Two coarse levels can agree by accident on a narrow peak',
    ],
    quantities: [
      { tex: 'N', key: 'info.n_points', label: 'nodes' },
      { tex: '\\text{new}', key: 'info.new_nodes', label: 'new nodes' },
      { tex: '|P_N - I|', key: 'info.error' },
      { tex: '|P_N - P_{N_{k-1}}|', key: 'info.err_est' },
    ],
  },
};

registerMethod(
  {
    id: 'clenshaw_curtis',
    family: 'integration',
    name: 'Clenshaw–Curtis quadrature',
    params: [
      param.int('n', 2, {
        min: 1,
        max: CC_N_MAX,
        help: 'Degree of the first rule (n + 1 Chebyshev points); step k uses n·2ᵏ.',
        label: 'Degree at step 0',
        tex: 'n_0',
      }),
      param.int('max_levels', 12, {
        min: 2,
        max: CC_LEVELS_MAX,
        help: 'Maximum number of doublings L; the last rule has n·2ᴸ + 1 points.',
        label: 'Doublings',
        tex: 'K',
      }),
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-1,
        log: true,
        help: NESTED_TOL_HELP,
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'exact for degree ≤ n; ≈ Gauss accuracy unless f is analytic in a large ellipse',
    summary:
      'Integrate the polynomial through f at Chebyshev points; doubling n reuses every old ' +
      'point and gives a free error estimate.',
    references: [
      'Clenshaw & Curtis, A method for numerical integration on an automatic computer, ' +
        'Numer. Math. 2 (1960)',
      'Trefethen, Is Gauss quadrature better than Clenshaw–Curtis?, SIAM Rev. 50 (2008), ' +
        'eqs. (2.2)–(2.3), Thm. 5.2',
      'Waldvogel, Fast construction of the Fejér and Clenshaw–Curtis quadrature rules, ' +
        'BIT 46 (2006), §5, eqs. (2.6), (3.10), (4.2)',
    ],
  },
  clenshawCurtis,
  NESTED_DOCS.clenshaw_curtis,
);

registerMethod(
  {
    id: 'gauss_patterson',
    family: 'integration',
    name: 'Gauss–Kronrod–Patterson quadrature',
    params: [
      param.int('max_levels', GKP_MAX_LEVEL, {
        min: 2,
        max: GKP_MAX_LEVEL,
        help: 'Last level; step k uses 2ᵏ⁺¹ − 1 points (1, 3, 7, …, 127).',
        label: 'Last level',
        tex: 'K',
      }),
      param.float('tol', 1e-10, {
        min: 1e-15,
        max: 1e-1,
        log: true,
        help: NESTED_TOL_HELP,
        label: 'Tolerance',
        tex: '\\varepsilon',
      }),
    ],
    needs: ['f', 'interval'],
    order: 'the N-point rule is exact for degree ≤ (3N + 1)/2 (N ≥ 3)',
    summary:
      "Gauss's 3 points, then repeatedly add the optimal new points between the old ones: " +
      'every old value is reused and each level gives a free error estimate.',
    references: [
      'Patterson, The optimum addition of points to quadrature formulae, Math. Comp. 22 (1968)',
      'Kronrod, Nodes and weights of quadrature formulas (1965)',
      'Gautschi, Orthogonal Polynomials: Computation and Approximation (2004), §3.1.2',
    ],
  },
  gaussPatterson,
  NESTED_DOCS.gauss_patterson,
);
