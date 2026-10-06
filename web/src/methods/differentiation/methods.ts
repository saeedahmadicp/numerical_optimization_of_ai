/**
 * Numerical differentiation: estimate f′(x0) (or f″(x0)) from values of f — TS port of
 * `numopt.differentiation.methods` (src/numopt/differentiation/methods.py).
 *
 * Every method sweeps the step sizes h_k = h0/2^k, k = 0, …, `levels`, and records one Step per
 * h. The sweep never stops early, so the trace always shows the V-shaped error curve: for large
 * h the truncation error C·h^p dominates, for small h the round-off error ε·|f|/h^q of the
 * cancelling differences dominates (Burden & Faires §4.1; Sauer §5.1.2).
 *
 * Each formula is D(h) = (1/h^q)·Σᵢ cᵢ·f(x0 + oᵢh):
 *
 *   forward_difference         o = 0, 1            c = −1, 1                  q = 1  p = 1
 *   backward_difference        o = −1, 0           c = −1, 1                  q = 1  p = 1
 *   central_difference         o = −1, 1           c = −1/2, 1/2              q = 1  p = 2
 *   five_point_stencil         o = −2, −1, 1, 2    c = (1, −8, 8, −1)/12      q = 1  p = 4
 *   second_derivative_central  o = −1, 0, 1        c = 1, −2, 1               q = 2  p = 2
 *
 * `richardson_extrapolation` extrapolates central differences to h = 0; `complex_step` uses
 * Im f(x0 + ih)/h. The level selection (err_est, the round-off model, the observed round-off ν,
 * the truncation run, the confirming cut c and the bound B_k) is the Python algorithm line by
 * line; see the Python module docstring for the derivation. Sums are correctly rounded (a port
 * of CPython's `math.fsum`), function values are cached by abscissa (`nFev` counts distinct
 * points), and the exact derivative (`problem.grad` / `problem.hess`) only fills the `error`
 * column.
 *
 * The TS calculus problems are real-only; `complex_step` evaluates the complex-analytic twin of
 * f from ./complex.ts (keyed by problem id), or `problem.fComplex` when a caller supplies one.
 *
 * Info keys (snake_case, as in Python): h, estimate, error, err_est, roundoff, roundoff_obs,
 * confirms, collapsed, stencil, weights, x0, plus row (Richardson) and imag (complex step).
 * Python `None` is `null`; a non-finite estimate is NaN (Python NaN → JSON null).
 * Result.extra keys: exact, error, k_best, h_best, bound_best, k_trunc_end, k_confirm_last,
 * h_opt, h_opt_rule, h_min_error, order, derivative.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { MethodFn, Params, Problem, Result, RunOptions, Step } from '../../core/types';
import { COMPLEX_F, cx, type Complex } from './complex';

const EPS = Number.EPSILON;

type F = (x: number) => number;

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
    if (Number.isNaN(infSum)) return null;
    return specialSum;
  }
  let hi = 0.0;
  if (n > 0) {
    hi = p[--n];
    let lo = 0.0;
    while (n > 0) {
      const x = hi;
      const y = p[--n];
      hi = x + y;
      const yr = hi - x;
      lo = y - yr;
      if (lo !== 0.0) break;
    }
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

/** Python `statistics.median` of a non-empty list of finite floats. */
export function median(values: readonly number[]): number {
  const d = [...values].sort((a, b) => a - b);
  const n = d.length;
  if (n % 2 === 1) return d[(n - 1) / 2];
  const i = n / 2;
  return (d[i - 1] + d[i]) / 2;
}

/** Python's `format(x, '.{digits}g')`: `4.77e-08`, `0.025`, `1.6e-10`, `inf`. */
export function pyG(x: number, digits = 3): string {
  if (Number.isNaN(x)) return 'nan';
  if (x === Infinity) return 'inf';
  if (x === -Infinity) return '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const [mant, expStr] = x.toExponential(digits - 1).split('e');
  const exp = Number(expStr);
  const strip = (s: string) => (s.includes('.') ? s.replace(/\.?0+$/, '') : s);
  if (exp >= -4 && exp < digits) return strip(x.toFixed(Math.max(0, digits - 1 - exp)));
  return `${strip(mant)}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
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

/** Counted evaluation of f with a cache keyed by the abscissa (Python `_CachedF`). */
class CachedF {
  n = 0;
  private readonly cache = new Map<number, number>();
  private readonly fn: F;
  constructor(fn: F) {
    this.fn = fn;
  }
  call = (x: number): number => {
    const hit = this.cache.get(x);
    if (hit !== undefined) return hit;
    this.n++;
    const v = safeEval(this.fn, x);
    this.cache.set(x, v);
    return v;
  };
}

interface Stencil {
  offsets: readonly number[];
  /** Integer weights: ints[i]·f is exact for ±1, ±2, ±8. */
  ints: readonly number[];
  /** Common denominator. */
  den: number;
  /** Derivative order. */
  q: number;
  /** Truncation order. */
  p: number;
}

export type StencilMethod =
  | 'forward_difference'
  | 'backward_difference'
  | 'central_difference'
  | 'five_point_stencil'
  | 'second_derivative_central';

export const STENCILS: Readonly<Record<StencilMethod, Stencil>> = {
  forward_difference: { offsets: [0, 1], ints: [-1, 1], den: 1, q: 1, p: 1 },
  backward_difference: { offsets: [-1, 0], ints: [-1, 1], den: 1, q: 1, p: 1 },
  central_difference: { offsets: [-1, 1], ints: [-1, 1], den: 2, q: 1, p: 2 },
  five_point_stencil: { offsets: [-2, -1, 1, 2], ints: [1, -8, 8, -1], den: 12, q: 1, p: 4 },
  second_derivative_central: { offsets: [-1, 0, 1], ints: [1, -2, 1], den: 1, q: 2, p: 2 },
};

const isFiniteNum = (v: number | null | undefined): v is number =>
  v !== null && v !== undefined && Number.isFinite(v);

/** Python `_finite(*values)`: every value is present and finite. */
const allFinite = (...values: (number | null | undefined)[]) => values.every(isFiniteNum);

/** The point of differentiation (Python `_start` → `start_scalar`); it must be finite. */
function startOf(problem: Problem<number>, x0: RunOptions['x0']): number {
  let x: unknown = x0 ?? problem.x0;
  if (x === undefined || x === null)
    throw new Error(`${problem.id}: no starting point given and the problem has no default x0`);
  if (Array.isArray(x)) x = x[0];
  const v = Number(x);
  if (!Number.isFinite(v)) throw new Error(`x0 must be finite, got ${String(v)}`);
  return v;
}

/** True when rounding merges two abscissae: fl(x0 + oᵢh) (x0 included) not strictly increasing. */
export function collapsed(x0: number, h: number, offsets: readonly number[]): boolean {
  const os = [...new Set([0, ...offsets])].sort((a, b) => a - b);
  const grid = os.map((o) => x0 + o * h);
  for (let i = 1; i < grid.length; i++) if (grid[i] <= grid[i - 1]) return true;
  return false;
}

function validate(h0: unknown, levels: unknown, tol: unknown): [number, number, number] {
  const h = Number(h0);
  if (!(Number.isFinite(h) && h > 0.0))
    throw new Error(`h0 must be a positive finite number, got ${String(h0)}`);
  const l = Number(levels);
  if (!Number.isInteger(l) || l < 0)
    throw new Error(`levels must be an integer ≥ 0, got ${String(levels)}`);
  const t = Number(tol);
  if (!(Number.isFinite(t) && t > 0.0))
    throw new Error(`tol must be a positive finite number, got ${String(tol)}`);
  return [h, l, t];
}

/** The exact f′(x0) or f″(x0) from the problem, for the error column only (not counted). */
function exactDerivative(problem: Problem<number>, x0: number, order: number): number | null {
  const fn = order === 1 ? problem.grad : problem.hess;
  if (!fn) return null;
  let v: number;
  try {
    v = Number(fn(x0));
  } catch {
    return null;
  }
  return Number.isFinite(v) ? v : null;
}

function errorOf(estimate: number, exact: number | null): number | null {
  if (exact === null || !Number.isFinite(estimate)) return null;
  return Math.abs(estimate - exact);
}

/** h_opt = constant·ε^power·max(1, |x0|) (N&W §8.1 scaling by max(1, |x0|)). */
const hOpt = (x0: number, constant: number, power: number) =>
  constant * EPS ** power * Math.max(1.0, Math.abs(x0));

/** Per-level numbers that feed the step selection, plus the data its Step displays. */
interface Level {
  h: number;
  /** NaN when not finite (fixed stencils, complex step). */
  estimate: number;
  /** Δ_k = |D_k − D_{k−1}| (null exactly when err_est is null). */
  diff: number | null;
  errEst: number | null;
  roundoff: number | null;
  collapsed: boolean;
  stencil: [number, number][];
  weights: number[];
  more: Record<string, unknown>;
  /** Richardson only: (Δ, roundoff) of the base column D(k, 0); null means (diff, roundoff). */
  base: [number, number] | null;
}

/** The observed round-off may exceed the a-priori model by this factor (Python `_MODEL_BROKEN`). */
const MODEL_BROKEN = 4.0;
/** Minimum length of the truncation run before one jump counts as round-off (`_MIN_RUN`). */
const MIN_RUN = 3;

export interface Selection {
  bounds: Map<number, number>;
  confirmed: Set<number>;
  /** Observed round-off ν_k (null when the level is not usable). */
  nu: (number | null)[];
  truncEnd: number | null;
  confirmLast: number | null;
}

/** ν_k = max_{i > k, Δ_i > Δ_{i−1}} Δ_i/(1 + 2^−q)·2^{−q(i−k)} by one backward pass. */
export function observedRoundoff(diffs: readonly (number | null)[], q: number): number[] {
  const shrink = 2.0 ** q;
  const nu = new Array<number>(diffs.length).fill(0.0);
  let carry = 0.0;
  for (let k = diffs.length - 1; k >= 0; k--) {
    nu[k] = carry;
    const d = diffs[k];
    const dPrev = k > 0 ? diffs[k - 1] : null;
    if (d !== null && dPrev !== null && d > dPrev) {
      const c = d / (1.0 + 1.0 / shrink);
      if (c > carry) carry = c;
    }
    carry /= shrink;
  }
  return nu;
}

/** B_k for every usable level (Python `_select`). */
function select(levels: readonly Level[], q: number): Selection {
  const n = levels.length;
  const usable = levels.map(
    (lev) =>
      !lev.collapsed &&
      lev.errEst !== null &&
      lev.diff !== null &&
      lev.roundoff !== null &&
      allFinite(lev.estimate, lev.errEst + lev.roundoff),
  );
  const diffs = levels.map((lev, k) => (usable[k] ? lev.diff : null));
  const nuAll = observedRoundoff(diffs, q);
  const nu = nuAll.map((v, k) => (usable[k] ? v : null));
  // The regime signals: the estimate itself, or Richardson's base column.
  const reg: [number | null, number | null][] = levels.map((lev, k) =>
    usable[k] ? (lev.base ?? [lev.diff, lev.roundoff]) : [null, null],
  );
  const regDiff = reg.map(([d]) => d);
  const regNu = observedRoundoff(regDiff, q);

  // The truncation run: the longest run of usable levels with strictly decreasing Δ.
  let truncEnd: number | null = null;
  let bestLen = 0;
  let start: number | null = null;
  for (let k = 0; k < n; k++) {
    const d = regDiff[k];
    if (d === null) {
      start = null;
      continue;
    }
    const dPrev = k > 0 ? regDiff[k - 1] : null;
    if (start === null || dPrev === null || !(d < dPrev)) start = k;
    if (k - start + 1 > bestLen) {
      bestLen = k - start + 1;
      truncEnd = k;
    }
  }
  if (truncEnd === null)
    return { bounds: new Map(), confirmed: new Set(), nu, truncEnd: null, confirmLast: null };

  // c: the first usable level after the run whose observed round-off breaks the model.
  let confirmLast = n - 1;
  const shrink = 2.0 ** q;
  for (let j = truncEnd + 1; j < n; j++) {
    const d = regDiff[j];
    const dPrev = regDiff[j - 1];
    const model = reg[j][1];
    const grew = d !== null && dPrev !== null && d > dPrev;
    const own = grew && d !== null && bestLen >= MIN_RUN ? d / (1.0 + 1.0 / shrink) : 0;
    if (model !== null && Math.max(regNu[j], own) > MODEL_BROKEN * model) {
      confirmLast = j;
      break;
    }
  }

  // b_k = err_est_k + max(roundoff_k, ν_k)
  const local: (number | null)[] = levels.map((lev, k) => {
    const nuK = nu[k];
    return lev.errEst !== null && lev.roundoff !== null && nuK !== null
      ? lev.errEst + Math.max(lev.roundoff, nuK)
      : null;
  });
  const bounds = new Map<number, number>();
  const confirmed = new Set<number>();
  local.forEach((bK, k) => {
    if (bK === null) return;
    // |e_k| ≥ |D_k − D_j| − b_j for every later confirming level j.
    let bound = bK;
    for (let j = k + 1; j <= confirmLast; j++) {
      const bJ = local[j];
      if (bJ !== null) {
        const lower = Math.abs(levels[k].estimate - levels[j].estimate) - bJ;
        if (lower > bound) bound = lower;
        confirmed.add(k);
      }
    }
    bounds.set(k, bound);
  });
  return { bounds, confirmed, nu, truncEnd, confirmLast };
}

function makeStep(
  k: number,
  lev: Level,
  x0: number,
  exact: number | null,
  roundoffObs: number | null,
  confirms: boolean,
): Step {
  return {
    k,
    x: lev.estimate,
    fun: lev.estimate,
    gradNorm: null,
    stepSize: lev.h,
    info: {
      h: lev.h,
      estimate: lev.estimate,
      error: errorOf(lev.estimate, exact),
      err_est: lev.errEst,
      roundoff: lev.roundoff,
      roundoff_obs: roundoffObs,
      confirms,
      collapsed: lev.collapsed,
      stencil: lev.stencil.map((pt) => [...pt]),
      weights: [...lev.weights],
      x0,
      ...lev.more,
    },
  };
}

/** Select k* = argmin B_k and build the trace and Result (Python `_finish`). */
function finish(
  method: string,
  x0: number,
  levels: readonly Level[],
  nFev: number,
  tol: number,
  exact: number | null,
  order: number,
  derivative: number,
  hOptValue: number | null,
  hOptRule: string,
): Result {
  const sel = select(levels, derivative);
  const { bounds, confirmed } = sel;
  const keys = [...bounds.keys()];
  // Prefer confirmed levels; an unconfirmed level is chosen only when none is confirmed.
  const confirmedKeys = keys.filter((k) => confirmed.has(k));
  const pool = confirmedKeys.length ? confirmedKeys : keys;
  let bestK: number | null = null;
  for (const k of pool) {
    if (bestK === null) {
      bestK = k;
      continue;
    }
    const b = bounds.get(k)!,
      bb = bounds.get(bestK)!;
    if (b < bb || (b === bb && k < bestK)) bestK = k;
  }
  const bestBound = bestK !== null ? bounds.get(bestK)! : Infinity;
  const last = sel.confirmLast;
  const trace = levels.map((lev, k) =>
    makeStep(k, lev, x0, exact, sel.nu[k], sel.nu[k] !== null && last !== null && k <= last),
  );

  // h with the smallest actual error (ties: the smaller h, like Python's tuple min).
  let hMinError: number | null = null;
  let eMin = Infinity;
  for (const lev of levels) {
    const e = errorOf(lev.estimate, exact);
    if (e === null) continue;
    if (hMinError === null || e < eMin || (e === eMin && lev.h < hMinError)) {
      eMin = e;
      hMinError = lev.h;
    }
  }
  const nNonfinite = levels.filter((lev) => !Number.isFinite(lev.estimate)).length;
  const extraBase = {
    k_trunc_end: sel.truncEnd,
    k_confirm_last: sel.confirmLast,
    h_opt: hOptValue,
    h_opt_rule: hOptRule,
    h_min_error: hMinError,
    order,
    derivative,
  };
  const result = (
    estimate: number,
    converged: boolean,
    message: string,
    extra: Record<string, unknown>,
  ): Result => ({
    method,
    x: estimate,
    fun: estimate,
    converged,
    message,
    nIter: levels.length - 1,
    nFev,
    nGev: 0,
    nHev: 0,
    trace,
    extra,
  });

  if (bestK === null) {
    const finiteLevels = levels.filter((lev) => Number.isFinite(lev.estimate));
    const estimate = finiteLevels.length ? finiteLevels[finiteLevels.length - 1].estimate : NaN;
    const msg =
      levels.length === 1
        ? 'levels = 0: a single estimate has no error estimate'
        : !finiteLevels.length
          ? 'every estimate is non-finite (f is not finite on the stencils)'
          : 'no level has an error estimate (it needs two consecutive finite levels ' +
            'whose abscissae do not collapse)';
    return result(estimate, false, msg, {
      exact,
      error: errorOf(estimate, exact),
      k_best: null,
      h_best: null,
      bound_best: null,
      ...extraBase,
    });
  }

  const best = levels[bestK];
  const within = bestBound <= tol * Math.max(1.0, Math.abs(best.estimate));
  const converged = within && confirmed.has(bestK);
  const rel = within ? '≤' : '>';
  let msg = `best h = ${pyG(best.h)} (level ${bestK}): error bound ${pyG(bestBound)} ${rel} tol·max(1, |D|)`;
  if (!confirmed.has(bestK)) {
    if (last !== null && last < levels.length - 1)
      msg +=
        `; no later level confirms this bound (the levels after ${last} are ` +
        'dominated by round-off: increase h0)';
    else msg += '; no later level confirms this bound (increase levels)';
  }
  if (nNonfinite) msg += `; ${nNonfinite} level(s) with non-finite values were skipped`;
  return result(best.estimate, converged, msg, {
    exact,
    error: errorOf(best.estimate, exact),
    k_best: bestK,
    h_best: best.h,
    bound_best: bestBound,
    ...extraBase,
  });
}

/** (Δ_k, err_est_k) = (|D_k − D_{k−1}|, Δ_k/factor), or (null, null) without a usable previous level. */
function diffAndErrEst(
  estimate: number,
  prev: Level | undefined,
  factor: number,
): [number | null, number | null] {
  if (!prev || prev.collapsed || !allFinite(estimate, prev.estimate)) return [null, null];
  const diff = Math.abs(estimate - prev.estimate);
  return [diff, diff / factor];
}

// ---------------------------------------------------------------------------------------
// Fixed-stencil finite differences
// ---------------------------------------------------------------------------------------

/** (constant, power of ε, formula) of the textbook optimal step (Burden & Faires §4.1; Sauer §5.1.2). */
export const H_OPT: Readonly<Record<StencilMethod, [number, number, string]>> = {
  forward_difference: [2.0, 1 / 2, '2·√ε·max(1,|x0|)'],
  backward_difference: [2.0, 1 / 2, '2·√ε·max(1,|x0|)'],
  central_difference: [3.0 ** (1 / 3), 1 / 3, '(3ε)^(1/3)·max(1,|x0|)'],
  five_point_stencil: [11.25 ** (1 / 5), 1 / 5, '(45ε/4)^(1/5)·max(1,|x0|)'],
  second_derivative_central: [48.0 ** 0.25, 1 / 4, '(48ε)^(1/4)·max(1,|x0|)'],
};

/** h^q with h² as one correctly rounded product (Python's `h**2` is glibc's exact pow). */
const powQ = (h: number, q: number) => (q === 1 ? h : h * h);

function finiteDifference(
  method: StencilMethod,
  problem: Problem<number>,
  options: RunOptions & Params,
): Result {
  const st = STENCILS[method];
  const x = startOf(problem, options.x0);
  const [h0, nLevels, tol] = validate(options.h0, options.levels, options.tol);
  const exact = exactDerivative(problem, x, st.q);
  const fc = new CachedF(problem.f as F);
  const factor = 2.0 ** st.p - 1.0;

  const rows: Level[] = [];
  const slopes: number[] = [];
  for (let k = 0; k <= nLevels; k++) {
    const h = h0 / 2.0 ** k;
    const xs = st.offsets.map((o) => x + o * h);
    const fs = xs.map((xi) => fc.call(xi));
    const scale = st.den * powQ(h, st.q);
    let estimate = fsum(st.ints.map((c, i) => c * fs[i])) / scale;
    if (!Number.isFinite(estimate)) estimate = NaN;
    const isCollapsed = collapsed(x, h, st.offsets);
    let diff: number | null = null;
    let errEst: number | null = null;
    let roundoff: number | null = null;
    if (!isCollapsed) {
      let slope = st.q === 1 ? Math.abs(estimate) : Math.abs(fs[2] - fs[0]) / (2.0 * h);
      if (Number.isFinite(slope)) slopes.push(slope);
      slope = slopes.length ? median(slopes) : NaN;
      const model =
        (EPS *
          fsum(
            st.ints.map(
              (c, i) =>
                Math.abs(c) * (Math.abs(fs[i]) + (st.offsets[i] ? Math.abs(xs[i]) * slope : 0.0)),
            ),
          )) /
        scale;
      roundoff = Number.isFinite(model) ? model : null;
      [diff, errEst] = diffAndErrEst(estimate, rows[rows.length - 1], factor);
    }
    rows.push({
      h,
      estimate,
      diff,
      errEst,
      roundoff,
      collapsed: isCollapsed,
      stencil: xs.map((xi, i) => [xi, fs[i]]),
      weights: st.ints.map((c) => c / scale),
      more: {},
      base: null,
    });
  }
  const [c, power, rule] = H_OPT[method];
  return finish(method, x, rows, fc.n, tol, exact, st.p, st.q, hOpt(x, c, power), rule);
}

function fdParams(levelsDefault = 30) {
  return [
    param.float('h0', 0.1, {
      min: 1e-12,
      max: 10.0,
      log: true,
      help: 'Largest step; level k uses h₀/2ᵏ.',
      label: 'Largest step',
      tex: 'h_0',
    }),
    param.int('levels', levelsDefault, {
      min: 0,
      max: 60,
      help: 'Number of halvings of h (the sweep always runs to the end).',
      label: 'Halvings of h',
      tex: 'K',
    }),
    param.float('tol', 1e-6, {
      min: 1e-15,
      max: 1e-1,
      log: true,
      help: "Converged when the best level's error bound ≤ tol·max(1, |D|).",
      label: 'Tolerance',
      tex: '\\mathrm{tol}',
    }),
  ];
}

const REF_BF = 'Burden & Faires, Numerical Analysis (10th ed.)';

const QUANTITIES: MethodDoc['quantities'] = [
  { tex: 'h_k', key: 'info.h' },
  { tex: 'D(h_k)', key: 'info.estimate' },
  { tex: "|D(h_k) - f'(x_0)|", key: 'info.error' },
  { tex: '\\hat e_{\\mathrm{trunc}}', key: 'info.err_est' },
  { tex: '\\hat e_{\\mathrm{round}}', key: 'info.roundoff' },
];

export const DOCS: Readonly<Record<string, MethodDoc>> = {
  forward_difference: {
    rule: "\\begin{aligned} D(h) &= \\frac{f(x_0 + h) - f(x_0)}{h} \\\\ &= f'(x_0) + \\tfrac{h}{2} f''(\\xi) \\end{aligned}",
    intuition:
      'The slope of the secant from x₀ to x₀ + h. The secant leans with the curvature, so the error shrinks only like h; below h ≈ √ε the two values of f agree in almost every digit and their difference is rounding noise.',
    order: 'O(h)',
    pros: [
      'One new evaluation per step: f(x₀) is reused',
      'Works where f is defined only on one side',
    ],
    cons: ['First order: half the digits at best (√ε ≈ 1.5×10⁻⁸)', 'Biased by the curvature f″'],
    quantities: QUANTITIES,
  },
  backward_difference: {
    rule: "\\begin{aligned} D(h) &= \\frac{f(x_0) - f(x_0 - h)}{h} \\\\ &= f'(x_0) - \\tfrac{h}{2} f''(\\xi) \\end{aligned}",
    intuition:
      'The mirror image of the forward difference: the secant from x₀ − h to x₀. Its error has the opposite sign, so the mean of the two is the central difference.',
    order: 'O(h)',
    pros: ['Uses only values left of x₀ (e.g. past data)', 'One new evaluation per step'],
    cons: ['First order', 'Round-off ≈ 2ε|f|/h like the forward difference'],
    quantities: QUANTITIES,
  },
  central_difference: {
    rule: "\\begin{aligned} D(h) &= \\frac{f(x_0 + h) - f(x_0 - h)}{2h} \\\\ &= f'(x_0) + \\tfrac{h^2}{6} f'''(\\xi) \\end{aligned}",
    intuition:
      'The chord across x₀ is parallel to the tangent up to h²: the even Taylor terms cancel in the difference. The best step is about ε¹ᐟ³, which gives about two thirds of the digits.',
    order: 'O(h²)',
    pros: [
      'Second order for the price of two evaluations',
      'Even error expansion: Richardson doubles the order per level',
    ],
    cons: [
      'Needs f on both sides of x₀',
      'A stencil that straddles a kink returns the chord slope',
    ],
    quantities: QUANTITIES,
  },
  five_point_stencil: {
    rule: "\\begin{aligned} D(h) &= \\tfrac{f(x_0 - 2h) - 8f(x_0 - h) + 8f(x_0 + h) - f(x_0 + 2h)}{12h} \\\\ &= f'(x_0) - \\tfrac{h^4}{30} f^{(5)}(\\xi) \\end{aligned}",
    intuition:
      'Four neighbours weighted so that the error terms up to h³ cancel: it is one Richardson step on the central difference, 4/3 of the h-chord minus 1/3 of the 2h-chord. Truncation falls like h⁴, so the V bottoms out at a larger h.',
    order: 'O(h⁴)',
    pros: [
      'Fourth order: about four fifths of the digits',
      'Each new level costs two evaluations (x₀ ± 2hₖ were x₀ ± hₖ₋₁)',
    ],
    cons: ['Wider stencil: needs f smooth on [x₀ − 2h, x₀ + 2h]', 'Round-off constant 1.5ε|f|/h'],
    quantities: QUANTITIES,
  },
  second_derivative_central: {
    rule: "\\begin{aligned} D_2(h) &= \\frac{f(x_0 - h) - 2f(x_0) + f(x_0 + h)}{h^2} \\\\ &= f''(x_0) + \\tfrac{h^2}{12} f^{(4)}(\\xi) \\end{aligned}",
    intuition:
      'The curvature of the parabola through three equally spaced points. Dividing by h² instead of h makes round-off grow like ε|f|/h², so the left branch of the V is twice as steep and the best h is near ε¹ᐟ⁴.',
    order: 'O(h²)',
    pros: ['Second order, three evaluations', 'The building block of finite-difference Hessians'],
    cons: ['Round-off grows like h⁻²: about half the digits', 'Error measured against f″, not f′'],
    quantities: [
      { tex: 'h_k', key: 'info.h' },
      { tex: 'D_2(h_k)', key: 'info.estimate' },
      { tex: "|D_2(h_k) - f''(x_0)|", key: 'info.error' },
      { tex: '\\hat e_{\\mathrm{trunc}}', key: 'info.err_est' },
      { tex: '\\hat e_{\\mathrm{round}}', key: 'info.roundoff' },
    ],
  },
  richardson_extrapolation: {
    rule: '\\begin{aligned} D(k, j) &= D(k, j-1) \\\\ &\\quad + \\frac{D(k, j-1) - D(k-1, j-1)}{4^{j} - 1} \\end{aligned}',
    intuition:
      'The central difference has an even expansion f′ + c₁h² + c₂h⁴ + …; each column of the table eliminates one more power of h². The diagonal D(k, k) is the value at h = 0 of the polynomial in h² through the first k + 1 central differences.',
    order: 'O(h²ᵏ⁺²) on row k',
    pros: [
      'Near machine precision from moderate steps',
      'Its own error estimate: successive diagonal entries',
    ],
    cons: [
      'Assumes a smooth even expansion (fails at kinks)',
      'Round-off is amplified by the extrapolation weights',
    ],
    quantities: [
      { tex: 'h_k', key: 'info.h' },
      { tex: 'D(k, k)', key: 'info.estimate' },
      { tex: "|D(k, k) - f'(x_0)|", key: 'info.error' },
      { tex: '\\hat e_{\\mathrm{trunc}}', key: 'info.err_est' },
      { tex: '\\hat e_{\\mathrm{round}}', key: 'info.roundoff' },
    ],
  },
  complex_step: {
    rule: "\\begin{aligned} D(h) &= \\frac{\\operatorname{Im} f(x_0 + ih)}{h} \\\\ &= f'(x_0) - \\tfrac{h^2}{6} f'''(x_0) + O(h^4) \\end{aligned}",
    intuition:
      'Step into the complex plane instead of along the real line: Im f(x₀ + ih) = h f′(x₀) − h³f‴/6 + …. Nothing is subtracted, so there is no cancellation: once h² is below ε the estimate is exact to rounding, and h can be 10⁻²⁰⁰.',
    order: 'O(h²), no cancellation',
    pros: ['Full machine precision for every small h', 'One complex evaluation per step'],
    cons: [
      'f must be complex-analytic (no abs, no real-only branches)',
      'Needs a complex implementation of f',
    ],
    quantities: [
      { tex: 'h_k', key: 'info.h' },
      { tex: '\\operatorname{Im} f(x_0 + ih_k)', key: 'info.imag' },
      { tex: 'D(h_k)', key: 'info.estimate' },
      { tex: "|D(h_k) - f'(x_0)|", key: 'info.error' },
      { tex: '\\hat e_{\\mathrm{trunc}}', key: 'info.err_est' },
    ],
  },
};

const fdMethod =
  (id: StencilMethod): MethodFn<Problem<number>> =>
  (problem, options) =>
    finiteDifference(id, problem, options);

registerMethod(
  {
    id: 'forward_difference',
    family: 'differentiation',
    name: 'Forward difference',
    params: fdParams(),
    needs: ['f', 'x0'],
    order: 'O(h)',
    summary: 'Slope of the secant from x₀ to x₀ + h.',
    references: [`${REF_BF}, Eq. (4.1)`],
  },
  fdMethod('forward_difference'),
  DOCS.forward_difference,
);

registerMethod(
  {
    id: 'backward_difference',
    family: 'differentiation',
    name: 'Backward difference',
    params: fdParams(),
    needs: ['f', 'x0'],
    order: 'O(h)',
    summary: 'Slope of the secant from x₀ − h to x₀.',
    references: [`${REF_BF}, §4.1 (h < 0 in Eq. 4.1)`],
  },
  fdMethod('backward_difference'),
  DOCS.backward_difference,
);

registerMethod(
  {
    id: 'central_difference',
    family: 'differentiation',
    name: 'Central difference',
    params: fdParams(),
    needs: ['f', 'x0'],
    order: 'O(h²)',
    summary: 'Slope of the chord from x₀ − h to x₀ + h; the odd error terms cancel.',
    references: [`${REF_BF}, Eq. (4.5) (three-point midpoint)`],
  },
  fdMethod('central_difference'),
  DOCS.central_difference,
);

registerMethod(
  {
    id: 'five_point_stencil',
    family: 'differentiation',
    name: 'Five-point stencil',
    params: fdParams(),
    needs: ['f', 'x0'],
    order: 'O(h⁴)',
    summary: 'Combine four neighbors so the error terms up to h³ cancel.',
    references: [`${REF_BF}, Eq. (4.6) (five-point midpoint)`],
  },
  fdMethod('five_point_stencil'),
  DOCS.five_point_stencil,
);

registerMethod(
  {
    id: 'second_derivative_central',
    family: 'differentiation',
    name: 'Second derivative (central)',
    params: fdParams(),
    needs: ['f', 'x0'],
    order: 'O(h²)',
    summary: 'Estimate f″(x₀) from the curvature of three equally spaced points.',
    references: [`${REF_BF}, Eq. (4.9)`],
  },
  fdMethod('second_derivative_central'),
  DOCS.second_derivative_central,
);

// ---------------------------------------------------------------------------------------
// Richardson extrapolation of central differences
// ---------------------------------------------------------------------------------------

const richardsonExtrapolation: MethodFn<Problem<number>> = (problem, options) => {
  const x = startOf(problem, options.x0);
  const [h0, nLevels, tol] = validate(options.h0, options.levels, options.tol);
  const exact = exactDerivative(problem, x, 1);
  const fc = new CachedF(problem.f as F);

  const rows: Level[] = [];
  let table: number[] = []; // previous row D(k−1, ·); empty after a restart
  let rhoRow: number[] = [];
  const slopes: number[] = [];
  for (let k = 0; k <= nLevels; k++) {
    const h = h0 / 2.0 ** k;
    const xp = x + h,
      xm = x - h;
    const fp = fc.call(xp),
      fm = fc.call(xm);
    const base = (fp - fm) / (2.0 * h);
    const isCollapsed = collapsed(x, h, [-1, 1]);
    if (!isCollapsed && Number.isFinite(base)) slopes.push(Math.abs(base));
    const slope = slopes.length ? median(slopes) : NaN;
    const rho0 =
      (EPS * (Math.abs(fp) + Math.abs(fm) + (Math.abs(xp) + Math.abs(xm)) * slope)) / (2.0 * h);
    let diff: number | null = null;
    let errEst: number | null = null;
    let roundoff: number | null = null;
    let baseSignals: [number, number] | null = null;
    let row: number[];
    let estimate: number;
    if (isCollapsed || !allFinite(base, rho0)) {
      row = [base];
      table = [];
      rhoRow = [];
      estimate = isCollapsed && Number.isFinite(base) ? base : NaN;
    } else {
      row = [base];
      const rho = [rho0];
      for (let j = 1; j <= table.length; j++) {
        const c = 4.0 ** j;
        row.push(row[j - 1] + (row[j - 1] - table[j - 1]) / (c - 1.0));
        rho.push((c * rho[j - 1] + rhoRow[j - 1]) / (c - 1.0));
      }
      estimate = row[row.length - 1];
      roundoff = rho[rho.length - 1];
      if (table.length) {
        diff = errEst = Math.abs(row[row.length - 1] - table[table.length - 1]);
        baseSignals = [Math.abs(base - table[0]), rho0];
      }
      table = row;
      rhoRow = rho;
    }
    rows.push({
      h,
      estimate,
      diff,
      errEst,
      roundoff,
      collapsed: isCollapsed,
      stencil: [
        [xm, fm],
        [xp, fp],
      ],
      weights: [-0.5 / h, 0.5 / h],
      more: { row: [...row] },
      base: baseSignals,
    });
  }
  return finish(
    'richardson_extrapolation',
    x,
    rows,
    fc.n,
    tol,
    exact,
    2,
    1,
    null,
    'none (depends on the column)',
  );
};

registerMethod(
  {
    id: 'richardson_extrapolation',
    family: 'differentiation',
    name: 'Richardson extrapolation',
    params: fdParams(20),
    needs: ['f', 'x0'],
    order: 'O(h^{2k+2}) on row k',
    summary: 'Extrapolate central differences on halved steps to h = 0.',
    references: [`${REF_BF}, §4.2`, 'Press et al., Numerical Recipes (3rd ed.), §5.7'],
  },
  richardsonExtrapolation,
  DOCS.richardson_extrapolation,
);

// ---------------------------------------------------------------------------------------
// Complex step
// ---------------------------------------------------------------------------------------

/** A problem may carry its own complex-analytic f (TS-only; Python's f accepts complex input). */
export type ComplexProblem = Problem<number> & { fComplex?: (z: Complex) => Complex };

/** The complex-analytic f of a problem, or null when none is known. */
export function complexF(problem: ComplexProblem): ((z: Complex) => Complex) | null {
  return problem.fComplex ?? COMPLEX_F[problem.id] ?? null;
}

const complexStep: MethodFn<ComplexProblem> = (problem, options) => {
  const x = startOf(problem, options.x0);
  const [h0, nLevels, tol] = validate(options.h0, options.levels, options.tol);
  const exact = exactDerivative(problem, x, 1);
  const f = complexF(problem);
  if (!f)
    throw new Error(
      `complex_step: f must accept complex input; ${problem.id} has no complex-analytic form`,
    );
  let nFev = 0;

  const rows: Level[] = [];
  for (let k = 0; k <= nLevels; k++) {
    const h = h0 / 2.0 ** k;
    let w: Complex;
    nFev++;
    try {
      w = f(cx(x, h));
    } catch {
      w = cx(NaN, NaN);
    }
    let estimate = w.im / h;
    if (!Number.isFinite(estimate)) estimate = NaN;
    const roundoff = EPS * Math.abs(estimate);
    const [diff, errEst] = diffAndErrEst(estimate, rows[rows.length - 1], 3.0);
    rows.push({
      h,
      estimate,
      diff,
      errEst,
      roundoff: Number.isFinite(roundoff) ? roundoff : null,
      collapsed: false,
      stencil: [[x, w.re]],
      weights: [1.0 / h],
      more: { imag: w.im },
      base: null,
    });
  }
  return finish(
    'complex_step',
    x,
    rows,
    nFev,
    tol,
    exact,
    2,
    1,
    hOpt(x, Math.sqrt(6.0), 0.5),
    '√(6ε)·max(1,|x0|)',
  );
};

registerMethod(
  {
    id: 'complex_step',
    family: 'differentiation',
    name: 'Complex-step derivative',
    params: [
      param.float('h0', 0.1, {
        min: 1e-30,
        max: 10.0,
        log: true,
        help: 'Largest step; level k uses h₀/2ᵏ.',
        label: 'Largest step',
        tex: 'h_0',
      }),
      param.int('levels', 30, {
        min: 0,
        max: 60,
        help: 'Number of halvings of h (the sweep always runs to the end).',
        label: 'Halvings of h',
        tex: 'K',
      }),
      param.float('tol', 1e-6, {
        min: 1e-15,
        max: 1e-1,
        log: true,
        help: "Converged when the best level's error bound ≤ tol·max(1, |D|).",
        label: 'Tolerance',
        tex: '\\mathrm{tol}',
      }),
    ],
    needs: ['f', 'x0', 'complex'],
    order: 'O(h²), no subtractive cancellation',
    summary: 'Take Im f(x₀ + ih)/h: no difference of nearly equal numbers, so h can be tiny.',
    references: [
      'Squire & Trapp, SIAM Review 40 (1998)',
      'Martins, Sturdza & Alonso, ACM TOMS 29 (2003)',
    ],
  },
  complexStep,
  DOCS.complex_step,
);
