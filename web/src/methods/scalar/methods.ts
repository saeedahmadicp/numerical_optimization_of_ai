/**
 * One-dimensional minimization — TS port of `numopt.scalar.methods`
 * (src/numopt/scalar/methods.py): interval elimination (golden section, Fibonacci, dichotomous,
 * ternary), successive parabolic interpolation, Brent's localmin, Newton's method with a
 * negative-curvature safeguard, and minimum bracketing (NR mnbrak).
 *
 * Interval elimination (Bazaraa, Sherali & Shetty (2006), §8.2): for unimodal f on [a, b] and
 * a ≤ x₁ < x₂ ≤ b, f(x₁) ≤ f(x₂) ⇒ a minimizer lies in [a, x₂], otherwise in [x₁, b]. Every
 * method here uses exactly this rule, ties included.
 *
 * The port mirrors Python line by line: same ids, ParamSpecs, iteration logic, stopping tests,
 * evaluation counts, messages and Step.info keys (snake_case, as documented in the Python
 * module docstring). Floating-point operations keep the Python order.
 *
 * Info keys (every Step, k = 0 included):
 *   all but newton_1d:  bracket [a, b]
 *   golden_section, fibonacci_search, dichotomous_search, ternary_search:
 *     interior [x1, x2], f_interior, evaluated [x…], cut 'left' | 'right' | null;
 *     fibonacci_search also ratio, eps
 *   parabolic_interpolation: triple, f_triple, pattern, step, trial, f_trial, parabola, vertex, max_step
 *   brent_minimize: xwv, f_xwv, step, trial, f_trial, parabola, vertex, tol
 *   newton_1d: step, g, h, parabola, alpha, trials
 *   bracket_minimum: triple, f_triple, step, trials
 * `parabola` is `{center: c, coef: [c0, c1, c2]}` for p(z) = c0 + c1 (z − c) + c2 (z − c)².
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type { MethodFn, Params, Problem, Result, Step, StepInfo } from '../../core/types';

// ---------------------------------------------------------------------------------------
// Constants (same values as Python)
// ---------------------------------------------------------------------------------------

export const EPS = Number.EPSILON;
export const SQRT_EPS = Math.sqrt(EPS);
/** Golden-section fraction ρ = (3 − √5)/2 = 1 − 1/φ ≈ 0.381966. */
export const RHO = 0.5 * (3.0 - Math.sqrt(5.0));
/** NR mnbrak magnification ratio (the 7-digit φ of NR 3rd ed. §10.1). */
export const GOLD = 1.618034;
/** NR mnbrak guard against division by zero in the parabolic extrapolation. */
const TINY = 1e-20;
/** Armijo constant and maximum number of halvings of Newton's gradient safeguard. */
const ARMIJO_C1 = 1e-4;
const MAX_BACKTRACK = 60;
/** No interval can be resolved below a few ulps of its end points. */
const RESOLUTION_ULPS = 4.0;
/** Fibonacci search plans for xtol − FIB_SLACK_ULPS·ε·max(|a|, |b|). */
const FIB_SLACK_ULPS = 16.0;
/** Smallest UI xtol of dichotomous_search. */
const DICHOTOMOUS_XTOL_MIN = 1e-10;
/** dichotomous_search: a comparison within TIE_ULPS ulps of the larger value is a rounding tie. */
const TIE_ULPS = 4.0;
/** Largest Fibonacci number a plan may use. */
const FIB_MAX = 2 ** 1000;
/** core.diff central-difference step factor. */
const H_CENTRAL = 6.055454452393343e-6;

/** Python `ValueError`: invalid input (the lab shows it as an input error). */
export class ScalarInputError extends Error {
  override name = 'ValueError';
}

type Func = (x: number) => number;
/** A 1-D problem (f, and for Newton f′, f″) or a bare callable f(x). */
export type ScalarProblemInput = Problem<number> | Func;

// ---------------------------------------------------------------------------------------
// Python-style formatting for messages
// ---------------------------------------------------------------------------------------

/** `format(v, ".{p}g")` (round-half-even on exact ties aside, as Python prints it). */
export function fmtG(v: number, p = 3): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, expStr] = v.toExponential(p - 1).split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= p) {
    const m = mant.includes('.') ? mant.replace(/0+$/, '').replace(/\.$/, '') : mant;
    const e = Math.abs(exp);
    return `${m}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  const fixed = v.toFixed(Math.max(0, p - 1 - exp));
  return fixed.includes('.') ? fixed.replace(/0+$/, '').replace(/\.$/, '') : fixed;
}

/** `str(v)` of a Python float (`1.0`, `nan`, `inf`). */
function pyStr(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const [mant, expStr] = v.toExponential().split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= 16) {
    const e = Math.abs(exp);
    return `${mant}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  const s = String(v);
  return s.includes('.') ? s : `${s}.0`;
}

const g17 = (v: number) => fmtG(v, 17);

/** Python `math.nextafter(x, inf)`. */
export function nextUp(x: number): number {
  if (Number.isNaN(x) || x === Infinity) return x;
  if (x === 0) return Number.MIN_VALUE;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  let bits = view.getBigUint64(0);
  bits = x > 0 ? bits + 1n : bits - 1n;
  view.setBigUint64(0, bits);
  return view.getFloat64(0);
}

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

/** Python `core.counting.scalar_problem`: a Problem, or a bare f(x) wrapped as one. */
function scalarProblem(problem: ScalarProblemInput): Problem<number> {
  if (typeof problem === 'function') {
    return { id: 'custom', name: 'custom', latex: 'f(x)', f: problem, dim: 1, domain: [-1, 1] };
  }
  if (!problem || typeof problem.f !== 'function')
    throw new TypeError('problem must be a numopt Problem or a callable f(x)');
  return problem;
}

/** Python `Counted`: `fc.n` counts the calls. */
interface Counted extends Func {
  n: number;
}
function counted(fn: Func): Counted {
  const c = ((x: number) => {
    c.n += 1;
    return fn(x);
  }) as Counted;
  c.n = 0;
  return c;
}

function resolveBracket(problem: Problem<number>, bracket: unknown): [number, number] {
  const br = (bracket ?? problem.bracket) as readonly unknown[] | null | undefined;
  if (br === null || br === undefined)
    throw new ScalarInputError(`${problem.id}: a bracket (a, b) is required`);
  const a = Number(br[0]),
    b = Number(br[1]);
  if (!(Number.isFinite(a) && Number.isFinite(b)))
    throw new ScalarInputError(
      `invalid bracket: end points must be finite, got (${pyStr(a)}, ${pyStr(b)})`,
    );
  if (!(a < b))
    throw new ScalarInputError(`invalid bracket: need a < b, got (${pyStr(a)}, ${pyStr(b)})`);
  if (!Number.isFinite(b - a))
    throw new ScalarInputError(
      `invalid bracket: the width b - a overflows, got (${pyStr(a)}, ${pyStr(b)})`,
    );
  return [a, b];
}

function requirePositive(values: Record<string, number>): void {
  for (const [name, value] of Object.entries(values)) {
    if (!(Number.isFinite(value) && value > 0.0))
      throw new ScalarInputError(`${name} must be a positive finite number, got ${pyStr(value)}`);
  }
}

function requireMaxIter(maxIter: number): void {
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new ScalarInputError(`max_iter must be a positive integer, got ${maxIter}`);
}

function startScalar(problem: Problem<number>, x0: unknown): number {
  let v = x0 ?? problem.x0;
  if (v === null || v === undefined)
    throw new ScalarInputError(
      `${problem.id}: no starting point given and the problem has no default x0`,
    );
  while (Array.isArray(v)) v = v[0];
  return Number(v);
}

/** The bracket-width stopping threshold: xtol, floored at floating-point resolution. */
function widthTol(xtol: number, a: number, b: number): number {
  return Math.max(xtol, RESOLUTION_ULPS * EPS * Math.max(Math.abs(a), Math.abs(b)));
}

function widthReason(width: number, xtol: number): string {
  if (width <= xtol) return `bracket width ${fmtG(width)} ≤ xtol`;
  return `bracket width ${fmtG(width)} reached floating-point resolution (xtol=${fmtG(xtol)})`;
}

function nonfinite(method: string, x: number, fx: number): string {
  return `${method}: f(${g17(x)}) = ${pyStr(fx)} is not finite; stopped`;
}

/** Centered parabola p(z) = c0 + c1 (z − c) + c2 (z − c)². */
export interface ParabolaInfo {
  center: number;
  coef: [number, number, number];
}

/** The parabola through three points in centered form about x0 (Newton divided differences), or null. */
export function parabolaThrough(
  x0: number,
  f0: number,
  x1: number,
  f1: number,
  x2: number,
  f2: number,
): ParabolaInfo | null {
  if (x0 === x1 || x0 === x2 || x1 === x2) return null;
  const d01 = (f1 - f0) / (x1 - x0);
  const d02 = (f2 - f0) / (x2 - x0);
  const c2 = (d02 - d01) / (x2 - x1);
  const c1 = d01 + c2 * (x0 - x1);
  if (!(Number.isFinite(c1) && Number.isFinite(c2))) return null;
  return { center: x0, coef: [f0, c1, c2] };
}

/** Evaluate a centered parabola. */
export function evalParabola(p: ParabolaInfo, z: number): number {
  const d = z - p.center;
  return p.coef[0] + p.coef[1] * d + p.coef[2] * d * d;
}

function result(
  method: string,
  x: number,
  fx: number,
  converged: boolean,
  message: string,
  nIter: number,
  trace: Step[],
  counts: { nFev: number; nGev?: number; nHev?: number },
  extra: Record<string, unknown> = {},
): Result {
  return {
    method,
    x,
    fun: fx,
    converged,
    message,
    nIter,
    nFev: counts.nFev,
    nGev: counts.nGev ?? 0,
    nHev: counts.nHev ?? 0,
    trace,
    extra,
  };
}

function step(
  k: number,
  x: number,
  fun: number,
  info: StepInfo,
  o: { stepSize?: number | null; gradNorm?: number | null } = {},
): Step {
  return { k, x, fun, gradNorm: o.gradNorm ?? null, stepSize: o.stepSize ?? null, info };
}

function intervalInfo(
  a: number,
  b: number,
  x1: number,
  x2: number,
  f1: number,
  f2: number,
  evaluated: number[],
  cut: 'left' | 'right' | null,
): StepInfo {
  return { bracket: [a, b], interior: [x1, x2], f_interior: [f1, f2], evaluated, cut };
}

const num = (o: Params, key: string, def: number): number => {
  const v = o[key];
  return v === undefined || v === null ? def : Number(v);
};

const XTOL_HELP =
  'Stop when the bracket width b − a ≤ xtol. For unimodal f, |x − x⋆| ≤ max(xtol, h): f-comparisons cannot resolve x⋆ below h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).';
const XTOL_PARAM = param.float('xtol', 1e-8, {
  min: 1e-12,
  max: 1e-1,
  log: true,
  help: XTOL_HELP,
  label: 'Bracket tolerance',
  tex: 'b - a \\le',
});

function maxIterParam(def: number) {
  return param.int('max_iter', def, {
    min: 1,
    max: 10_000,
    help: 'Iteration limit.',
    label: 'Iteration budget',
  });
}

// ---------------------------------------------------------------------------------------
// Golden-section search
// ---------------------------------------------------------------------------------------

export const goldenSection: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  let [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-8);
  const maxIter = num(o, 'max_iter', 200);
  requirePositive({ xtol });
  requireMaxIter(maxIter);
  const f = counted(prob.f as Func);
  const method = 'golden_section';

  let x1 = a + RHO * (b - a);
  let x2 = x1 + RHO * (b - x1);
  let f1 = f(x1),
    f2 = f(x2);
  let [xb, fb] = f1 <= f2 ? [x1, f1] : [x2, f2];
  const trace: Step[] = [
    step(0, xb, fb, intervalInfo(a, b, x1, x2, f1, f2, [x1, x2], null), { stepSize: b - a }),
  ];
  for (const [xe, fe] of [
    [x1, f1],
    [x2, f2],
  ]) {
    if (!Number.isFinite(fe))
      return result(method, xe, fe, false, nonfinite(method, xe, fe), 0, trace, { nFev: f.n });
  }

  for (let k = 1; k <= maxIter; k++) {
    if (b - a <= widthTol(xtol, a, b))
      return result(
        method,
        xb,
        fb,
        true,
        widthReason(b - a, xtol),
        k - 1,
        trace,
        { nFev: f.n },
        {
          bracket: [a, b],
        },
      );
    let fnew: number, xnew: number, cut: 'left' | 'right';
    if (f1 <= f2) {
      // minimizer in [a, x2]: discard (x2, b]
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = x2 - RHO * (x2 - a);
      f1 = fnew = f(x1);
      xnew = x1;
      cut = 'right';
    } else {
      // minimizer in [x1, b]: discard [a, x1)
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = x1 + RHO * (b - x1);
      f2 = fnew = f(x2);
      xnew = x2;
      cut = 'left';
    }
    [xb, fb] = f1 <= f2 ? [x1, f1] : [x2, f2];
    trace.push(
      step(k, xb, fb, intervalInfo(a, b, x1, x2, f1, f2, [xnew], cut), { stepSize: b - a }),
    );
    if (!Number.isFinite(fnew))
      return result(method, xnew, fnew, false, nonfinite(method, xnew, fnew), k, trace, {
        nFev: f.n,
      });
  }

  const converged = b - a <= widthTol(xtol, a, b);
  const msg = converged ? widthReason(b - a, xtol) : `reached max_iter=${maxIter}`;
  return result(method, xb, fb, converged, msg, maxIter, trace, { nFev: f.n }, { bracket: [a, b] });
};

// ---------------------------------------------------------------------------------------
// Fibonacci search
// ---------------------------------------------------------------------------------------

/**
 * F₀ = F₁ = 1, F_{j+1} = F_j + F_{j−1}, extended until n ≥ 3 and F_n ≥ target (Bazaraa et al.
 * §8.2 indexing); also stops at n = `nMax` or once F_n > 2¹⁰⁰⁰.
 */
export function fibonacciNumbers(target: number, nMax: number | null = null): number[] {
  const fib = [1, 1];
  while (
    fib.length < 4 ||
    (fib[fib.length - 1] < target &&
      (nMax === null || fib.length <= nMax) &&
      fib[fib.length - 1] <= FIB_MAX)
  ) {
    fib.push(fib[fib.length - 1] + fib[fib.length - 2]);
  }
  return fib;
}

export const fibonacciSearch: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  let [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-8);
  const epsRatio = num(o, 'eps_ratio', 0.05);
  const maxIter = num(o, 'max_iter', 200);
  requirePositive({ xtol, eps_ratio: epsRatio });
  if (epsRatio >= 1.0)
    throw new ScalarInputError(
      `eps_ratio must be < 1 (ε must be smaller than the last interval), got ${pyStr(epsRatio)}`,
    );
  requireMaxIter(maxIter);
  const f = counted(prob.f as Func);
  const method = 'fibonacci_search';

  const L1 = b - a;
  const slack = FIB_SLACK_ULPS * EPS * Math.max(Math.abs(a), Math.abs(b));
  const planTol = xtol > 2.0 * slack ? xtol - slack : xtol * (1.0 - FIB_SLACK_ULPS * EPS);
  const target = ((1.0 + epsRatio) * L1) / planTol;
  const fib = fibonacciNumbers(target, maxIter + 2);
  const n = fib.length - 1;
  const capped = fib[n] < target;
  const eps = (epsRatio * L1) / fib[n];
  const extra: Record<string, unknown> = { n_planned: n, eps };

  let lam = a + (fib[n - 2] / fib[n]) * L1;
  let mu = a + (fib[n - 1] / fib[n]) * L1;
  let flam = f(lam),
    fmu = f(mu);

  const info = (
    evaluated: number[],
    cut: 'left' | 'right' | null,
    ratio: number | null,
    shift: number | null,
  ): StepInfo => ({
    ...intervalInfo(a, b, lam, mu, flam, fmu, evaluated, cut),
    ratio,
    eps: shift,
  });
  const best = (): [number, number] => (flam <= fmu ? [lam, flam] : [mu, fmu]);

  let [xb, fb] = best();
  const trace: Step[] = [step(0, xb, fb, info([lam, mu], null, null, null), { stepSize: b - a })];
  const finish = (converged: boolean, msg: string): Result => {
    extra.bracket = [a, b];
    return result(
      method,
      xb,
      fb,
      converged,
      msg,
      trace[trace.length - 1].k,
      trace,
      { nFev: f.n },
      extra,
    );
  };
  for (const [xe, fe] of [
    [lam, flam],
    [mu, fmu],
  ]) {
    if (!Number.isFinite(fe))
      return result(
        method,
        xe,
        fe,
        false,
        nonfinite(method, xe, fe),
        0,
        trace,
        { nFev: f.n },
        extra,
      );
  }

  const nIterPlanned = n - 2;
  for (let k = 1; k <= nIterPlanned; k++) {
    if (b - a <= RESOLUTION_ULPS * EPS * Math.max(Math.abs(a), Math.abs(b)))
      return finish(true, `${widthReason(b - a, xtol)} before the planned n = ${n} evaluations`);
    if (!(a < lam && lam < mu && mu < b))
      return finish(false, `rounding broke the order a < λ < μ < b at k = ${k - 1}; stopped`);
    let fnew: number, xnew: number;
    if (k < nIterPlanned) {
      const c = fib[n - k - 2] / fib[n - k - 1];
      let cut: 'left' | 'right';
      if (flam > fmu) {
        // discard [a, λ)
        a = lam;
        lam = mu;
        flam = fmu;
        mu = b - c * (b - lam);
        fmu = fnew = f(mu);
        xnew = mu;
        cut = 'left';
      } else {
        // discard (μ, b]
        b = mu;
        mu = lam;
        fmu = flam;
        lam = a + c * (mu - a);
        flam = fnew = f(lam);
        xnew = lam;
        cut = 'right';
      }
      [xb, fb] = best();
      const ratio = fib[n - k - 1] / fib[n - k];
      trace.push(step(k, xb, fb, info([xnew], cut, ratio, null), { stepSize: b - a }));
    } else {
      // Last iteration: reduce once more (p lands on the midpoint), compare p with q = p + ε.
      let aNew: number, bNew: number, p: number, fp: number;
      if (flam > fmu) [aNew, bNew, p, fp] = [lam, b, mu, fmu];
      else [aNew, bNew, p, fp] = [a, mu, lam, flam];
      const shift = p + eps > p ? eps : nextUp(p) - p;
      const q = p + shift;
      if (!(q < bNew)) {
        a = aNew;
        b = bNew;
        const msg =
          `no point fits between p and b at width ${fmtG(b - a)} (the bracket after the ` +
          'last comparison); stopped';
        return finish(b - a <= widthTol(xtol, a, b), msg);
      }
      a = aNew;
      b = bNew;
      const fq = (fnew = f(q));
      xnew = q;
      let cut: 'left' | 'right';
      if (fp > fq) {
        a = p;
        cut = 'left';
      } else {
        b = q;
        cut = 'right';
      }
      [lam, flam, mu, fmu] = [p, fp, q, fq];
      [xb, fb] = best();
      trace.push(step(k, xb, fb, info([xnew], cut, 0.5, shift), { stepSize: b - a }));
    }
    if (!Number.isFinite(fnew))
      return result(
        method,
        xnew,
        fnew,
        false,
        nonfinite(method, xnew, fnew),
        k,
        trace,
        {
          nFev: f.n,
        },
        extra,
      );
  }

  const converged = b - a <= widthTol(xtol, a, b);
  let msg: string;
  if (converged) msg = `${widthReason(b - a, xtol)} after the planned n = ${n} evaluations`;
  else if (capped) {
    const limit =
      n === maxIter + 2 ? `max_iter + 2 = ${maxIter + 2} evaluations` : 'a plan with F_n ≤ 2^1000';
    msg =
      `xtol = ${fmtG(xtol)} needs more than ${limit}; the planned n = ${n} point ` +
      `Fibonacci plan ends with width ${fmtG(b - a)} > xtol`;
  } else msg = `final width ${fmtG(b - a)} > xtol after n = ${n} evaluations (rounding)`;
  return finish(converged, msg);
};

// ---------------------------------------------------------------------------------------
// Dichotomous search and ternary search (two new evaluations per iteration)
// ---------------------------------------------------------------------------------------

function roundingTie(f1: number, f2: number): boolean {
  return Math.abs(f1 - f2) <= TIE_ULPS * EPS * Math.max(Math.abs(f1), Math.abs(f2));
}

function twoPointSearch(
  method: string,
  prob: Problem<number>,
  a: number,
  b: number,
  place: (lo: number, hi: number) => [number, number],
  xtol: number,
  maxIter: number,
  stopOnTie = false,
): Result {
  const f = counted(prob.f as Func);
  let [x1, x2] = place(a, b);
  let f1 = f(x1),
    f2 = f(x2);
  let [xb, fb] = f1 <= f2 ? [x1, f1] : [x2, f2];
  const trace: Step[] = [
    step(0, xb, fb, intervalInfo(a, b, x1, x2, f1, f2, [x1, x2], null), { stepSize: b - a }),
  ];
  for (const [xe, fe] of [
    [x1, f1],
    [x2, f2],
  ]) {
    if (!Number.isFinite(fe))
      return result(method, xe, fe, false, nonfinite(method, xe, fe), 0, trace, { nFev: f.n });
  }

  for (let k = 1; k <= maxIter; k++) {
    if (b - a <= widthTol(xtol, a, b))
      return result(
        method,
        xb,
        fb,
        true,
        widthReason(b - a, xtol),
        k - 1,
        trace,
        { nFev: f.n },
        {
          bracket: [a, b],
        },
      );
    if (stopOnTie && roundingTie(f1, f2)) {
      const msg =
        `comparison resolution reached at bracket width ${fmtG(b - a)} > xtol: ` +
        `f(x1) and f(x2), ${fmtG(x2 - x1)} apart, agree to rounding, so neither end ` +
        'can be discarded (increase delta_ratio·xtol)';
      return result(method, xb, fb, false, msg, k - 1, trace, { nFev: f.n }, { bracket: [a, b] });
    }
    let cut: 'left' | 'right';
    if (f1 <= f2) {
      b = x2;
      cut = 'right';
    } else {
      a = x1;
      cut = 'left';
    }
    [x1, x2] = place(a, b);
    f1 = f(x1);
    f2 = f(x2);
    [xb, fb] = f1 <= f2 ? [x1, f1] : [x2, f2];
    trace.push(
      step(k, xb, fb, intervalInfo(a, b, x1, x2, f1, f2, [x1, x2], cut), { stepSize: b - a }),
    );
    for (const [xe, fe] of [
      [x1, f1],
      [x2, f2],
    ]) {
      if (!Number.isFinite(fe))
        return result(method, xe, fe, false, nonfinite(method, xe, fe), k, trace, { nFev: f.n });
    }
  }

  const converged = b - a <= widthTol(xtol, a, b);
  const msg = converged ? widthReason(b - a, xtol) : `reached max_iter=${maxIter}`;
  return result(method, xb, fb, converged, msg, maxIter, trace, { nFev: f.n }, { bracket: [a, b] });
}

export const dichotomousSearch: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-6);
  const deltaRatio = num(o, 'delta_ratio', 0.1);
  const maxIter = num(o, 'max_iter', 200);
  requirePositive({ xtol, delta_ratio: deltaRatio });
  if (deltaRatio >= 1.0)
    throw new ScalarInputError(
      `delta_ratio must be < 1 (the width converges to δ), got ${pyStr(deltaRatio)}`,
    );
  requireMaxIter(maxIter);
  const halfDelta = 0.5 * deltaRatio * xtol;
  if (2.0 * halfDelta <= RESOLUTION_ULPS * EPS * Math.max(Math.abs(a), Math.abs(b)))
    throw new ScalarInputError(
      `δ = delta_ratio·xtol = ${fmtG(2.0 * halfDelta)} is below floating-point resolution on ` +
        `[${pyStr(a)}, ${pyStr(b)}]: the two points would coincide and every comparison would be a tie`,
    );
  if (2.0 * halfDelta >= b - a)
    throw new ScalarInputError(
      `δ = delta_ratio·xtol = ${fmtG(2.0 * halfDelta)} must be smaller than the bracket width ` +
        `b - a = ${fmtG(b - a)}: the points m ± δ/2 would lie outside [${pyStr(a)}, ${pyStr(b)}] ` +
        '(reduce xtol or delta_ratio, or widen the bracket)',
    );
  const place = (lo: number, hi: number): [number, number] => {
    const m = lo + 0.5 * (hi - lo);
    // The clamp only acts on rounding (δ < hi − lo puts m ± δ/2 inside in exact arithmetic).
    return [Math.max(m - halfDelta, lo), Math.min(m + halfDelta, hi)];
  };
  return twoPointSearch('dichotomous_search', prob, a, b, place, xtol, maxIter, true);
};

export const ternarySearch: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-8);
  const maxIter = num(o, 'max_iter', 200);
  requirePositive({ xtol });
  requireMaxIter(maxIter);
  const place = (lo: number, hi: number): [number, number] => {
    const third = (hi - lo) / 3.0;
    return [lo + third, hi - third];
  };
  return twoPointSearch('ternary_search', prob, a, b, place, xtol, maxIter);
};

// ---------------------------------------------------------------------------------------
// Successive parabolic interpolation (safeguarded)
// ---------------------------------------------------------------------------------------

/** Vertex of the parabola through three points (NR eq. 10.3.1); NaN for collinear points. */
export function parabolaVertex(
  xl: number,
  xm: number,
  xr: number,
  fl: number,
  fm: number,
  fr: number,
): number {
  const p = (xm - xl) * (fm - fr);
  const q = (xm - xr) * (fm - fl);
  const den = p - q;
  if (den === 0.0) return NaN;
  return xm - (0.5 * ((xm - xl) * p - (xm - xr) * q)) / den;
}

export type ParabolicStep = 'init' | 'parabolic' | 'probe' | 'golden' | 'bisect';

export const parabolicInterpolation: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  const [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-8);
  const maxIter = num(o, 'max_iter', 200);
  requirePositive({ xtol });
  requireMaxIter(maxIter);
  const f = counted(prob.f as Func);
  const method = 'parabolic_interpolation';

  let xl = a,
    xm = a + 0.5 * (b - a),
    xr = b;
  let fl = f(xl),
    fm = f(xm),
    fr = f(xr);

  const best = (): [number, number] => {
    // Python min(((xm, fm), (xl, fl), (xr, fr)), key=f): the first minimal entry.
    let out: [number, number] = [xm, fm];
    if (fl < out[1]) out = [xl, fl];
    if (fr < out[1]) out = [xr, fr];
    return out;
  };
  const info = (
    kind: ParabolicStep,
    trial: number | null,
    fTrial: number | null,
    parab: ParabolaInfo | null,
    vertex: number | null,
    maxStep: number | null,
  ): StepInfo => ({
    bracket: [xl, xr],
    triple: [xl, xm, xr],
    f_triple: [fl, fm, fr],
    pattern: fm <= fl && fm <= fr,
    step: kind,
    trial,
    f_trial: fTrial,
    parabola: parab,
    vertex,
    max_step: maxStep,
  });

  // |u − x_m| of the last two iterations (Brent's d and e); ∞ until two steps exist.
  let lastStep = Infinity,
    stepBeforeLast = Infinity;
  const trace: Step[] = [
    step(0, xm, fm, info('init', null, null, null, null, null), { stepSize: xr - xl }),
  ];
  for (const [xe, fe, where] of [
    [xl, fl, 'a'],
    [xm, fm, ''],
    [xr, fr, 'b'],
  ] as const) {
    if (!Number.isFinite(fe)) {
      let msg = nonfinite(method, xe, fe);
      if (where)
        msg =
          `${method}: f(${g17(xe)}) = ${pyStr(fe)} at the bracket end point ${where} is not ` +
          'finite; this method evaluates f at both bracket ends, so choose a bracket ' +
          'strictly inside the domain of f; stopped';
      return result(method, xe, fe, false, msg, 0, trace, { nFev: f.n });
    }
  }
  let [xb, fb] = best();
  trace[0] = step(0, xb, fb, trace[0].info, { stepSize: xr - xl });

  for (let k = 1; k <= maxIter; k++) {
    const width = xr - xl;
    const delta = 0.5 * widthTol(xtol, xl, xr);
    if (Math.max(xm - xl, xr - xm) <= delta)
      return result(
        method,
        xb,
        fb,
        true,
        widthReason(width, xtol),
        k - 1,
        trace,
        { nFev: f.n },
        {
          bracket: [xl, xr],
        },
      );
    let parab: ParabolaInfo | null = null;
    let vertex: number | null = null;
    const maxStep = 0.5 * stepBeforeLast;
    const xmOld = xm;
    let kind: ParabolicStep;
    let u: number, fu: number;
    if (!(fm <= fl && fm <= fr)) {
      kind = 'bisect';
      if (fl <= fr) {
        // f(x_m) > f(x_l): minimizer in [x_l, x_m]
        xr = xm;
        fr = fm;
      } else {
        // f(x_m) > f(x_r): minimizer in [x_m, x_r]
        xl = xm;
        fl = fm;
      }
      u = xl + 0.5 * (xr - xl);
      fu = f(u);
      xm = u;
      fm = fu;
    } else {
      parab = parabolaThrough(xm, fm, xl, fl, xr, fr);
      u = parabolaVertex(xl, xm, xr, fl, fm, fr);
      if (Number.isFinite(u) && xl < u && u < xr && Math.abs(u - xm) < maxStep) {
        vertex = u;
        kind = 'parabolic';
        if (Math.abs(u - xm) < delta) {
          // Minimum step (Brent 1973, Ch. 5): never evaluate closer than δ to x_m.
          kind = 'probe';
          let right = u !== xm ? u > xm : xr - xm > xm - xl;
          if (right && xr - xm <= delta) right = false;
          else if (!right && xm - xl <= delta) right = true;
          u = right ? xm + delta : xm - delta;
          if (!(xl < u && u < xr)) u = right ? xm + 0.5 * (xr - xm) : xm - 0.5 * (xm - xl);
        }
      } else {
        // Safeguard: unusable fit or failed progress test → golden step into the larger segment.
        vertex = Number.isFinite(u) ? u : null;
        kind = 'golden';
        u = xm - xl > xr - xm ? xm - RHO * (xm - xl) : xm + RHO * (xr - xm);
      }
      if (!(xl < u && u < xr) || u === xm) {
        const msg = 'the trial point cannot leave x_m in floating point; xtol is below resolution';
        return result(
          method,
          xb,
          fb,
          false,
          msg,
          k - 1,
          trace,
          { nFev: f.n },
          {
            bracket: [xl, xr],
          },
        );
      }
      fu = f(u);
      if (Number.isFinite(fu)) {
        if (u < xm) {
          if (fu <= fm) [xr, fr, xm, fm] = [xm, fm, u, fu];
          else [xl, fl] = [u, fu];
        } else if (fu <= fm) [xl, fl, xm, fm] = [xm, fm, u, fu];
        else [xr, fr] = [u, fu];
      }
    }
    // As in FMM fmin, a probe records the vertex step |vertex − x_m|, not δ.
    const moved = Math.abs(
      ((kind === 'parabolic' || kind === 'probe') && vertex !== null ? vertex : u) - xmOld,
    );
    stepBeforeLast = lastStep;
    lastStep = moved;
    [xb, fb] = best();
    trace.push(
      step(k, xb, fb, info(kind, u, fu, parab, vertex, Number.isFinite(maxStep) ? maxStep : null), {
        stepSize: xr - xl,
      }),
    );
    if (!Number.isFinite(fu))
      return result(method, u, fu, false, nonfinite(method, u, fu), k, trace, { nFev: f.n });
  }

  const width = xr - xl;
  const converged = Math.max(xm - xl, xr - xm) <= 0.5 * widthTol(xtol, xl, xr);
  const msg = converged ? widthReason(width, xtol) : `reached max_iter=${maxIter}`;
  return result(
    method,
    xb,
    fb,
    converged,
    msg,
    maxIter,
    trace,
    { nFev: f.n },
    {
      bracket: [xl, xr],
    },
  );
};

// ---------------------------------------------------------------------------------------
// Brent's method (localmin)
// ---------------------------------------------------------------------------------------

export const brentMinimize: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  let [a, b] = resolveBracket(prob, o.bracket);
  const xtol = num(o, 'xtol', 1e-8);
  const rtol = num(o, 'rtol', SQRT_EPS);
  const maxIter = num(o, 'max_iter', 500);
  requirePositive({ xtol, rtol });
  requireMaxIter(maxIter);
  const f = counted(prob.f as Func);
  const method = 'brent_minimize';

  let x = a + RHO * (b - a);
  let w = x,
    v = x;
  let fx = f(x);
  let fw = fx,
    fv = fx;
  let d = 0.0,
    e = 0.0;
  let tol = rtol * Math.abs(x) + xtol / 3.0;

  const info = (
    kind: 'init' | 'parabolic' | 'golden',
    u: number | null,
    fu: number | null,
    parab: ParabolaInfo | null,
    vertex: number | null,
  ): StepInfo => ({
    bracket: [a, b],
    xwv: [x, w, v],
    f_xwv: [fx, fw, fv],
    step: kind,
    trial: u,
    f_trial: fu,
    parabola: parab,
    vertex,
    tol,
  });

  const trace: Step[] = [step(0, x, fx, info('init', null, null, null, null), { stepSize: b - a })];
  if (!Number.isFinite(fx))
    return result(method, x, fx, false, nonfinite(method, x, fx), 0, trace, { nFev: f.n });

  for (let k = 1; k <= maxIter; k++) {
    const m = 0.5 * (a + b);
    tol = rtol * Math.abs(x) + xtol / 3.0;
    const t2 = 2.0 * tol;
    if (Math.abs(x - m) <= t2 - 0.5 * (b - a)) {
      const msg = `x is within 2·tol = ${fmtG(t2)} of both bracket ends`;
      return result(method, x, fx, true, msg, k - 1, trace, { nFev: f.n }, { bracket: [a, b] });
    }
    let parab: ParabolaInfo | null = null;
    let vertex: number | null = null;
    let p = 0.0,
      q = 0.0,
      r = 0.0;
    if (Math.abs(e) > tol) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      else q = -q;
      r = e;
      e = d;
      parab = parabolaThrough(x, fx, w, fw, v, fv);
      if (q !== 0.0) vertex = x + p / q;
    }
    let kind: 'parabolic' | 'golden';
    if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x)) {
      kind = 'parabolic';
      d = p / q;
      const u = x + d;
      // f must not be evaluated too close to a or b.
      if (u - a < t2 || b - u < t2) d = x <= m ? tol : -tol;
    } else {
      kind = 'golden';
      e = x < m ? b - x : a - x;
      d = RHO * e;
    }
    const u = Math.abs(d) >= tol ? x + d : x + (d >= 0.0 ? tol : -tol);
    const xPrev = x;
    const fu = f(u);
    if (!Number.isFinite(fu)) {
      trace.push(step(k, x, fx, info(kind, u, fu, parab, vertex), { stepSize: Math.abs(u - x) }));
      return result(method, u, fu, false, nonfinite(method, u, fu), k, trace, { nFev: f.n });
    }
    if (fu <= fx) {
      if (u < x) b = x;
      else a = x;
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x) a = u;
      else b = u;
      if (fu <= fw || w === x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v === x || v === w) {
        v = u;
        fv = fu;
      }
    }
    trace.push(step(k, x, fx, info(kind, u, fu, parab, vertex), { stepSize: Math.abs(u - xPrev) }));
  }

  const m = 0.5 * (a + b);
  tol = rtol * Math.abs(x) + xtol / 3.0;
  const converged = Math.abs(x - m) <= 2.0 * tol - 0.5 * (b - a);
  const msg = converged
    ? `x is within 2·tol = ${fmtG(2.0 * tol)} of both bracket ends`
    : `reached max_iter=${maxIter}`;
  return result(method, x, fx, converged, msg, maxIter, trace, { nFev: f.n }, { bracket: [a, b] });
};

// ---------------------------------------------------------------------------------------
// Newton's method for minimization
// ---------------------------------------------------------------------------------------

/** core.diff.derivative: central-difference f′(x). */
function derivative(f: Func, x: number): number {
  const h = H_CENTRAL * Math.max(1.0, Math.abs(x));
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

/** core.diff.second_derivative: central-difference f″(x) with h ≈ ε^{1/4}. */
function secondDerivative(f: Func, x: number): number {
  const h = EPS ** 0.25 * Math.max(1.0, Math.abs(x));
  return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

export const newton1d: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  const gtol = num(o, 'gtol', 1e-8);
  const alpha0 = num(o, 'alpha0', 1.0);
  const maxIter = num(o, 'max_iter', 100);
  requirePositive({ gtol, alpha0 });
  requireMaxIter(maxIter);
  let x = startScalar(prob, o.x0);
  if (!Number.isFinite(x)) throw new ScalarInputError(`x0 must be finite, got ${pyStr(x)}`);
  const method = 'newton_1d';
  const f = counted(prob.f as Func);
  const gFn: Func = prob.grad ? (prob.grad as Func) : (z) => derivative(f, z);
  const hFn: Func = prob.hess ? (prob.hess as Func) : (z) => secondDerivative(f, z);
  const g = counted(gFn);
  const h = counted(hFn);

  let fx = f(x),
    gx = g(x),
    hx = h(x);
  const trace: Step[] = [
    step(
      0,
      x,
      fx,
      { step: 'init', g: gx, h: hx, parabola: null, alpha: null, trials: [] },
      { gradNorm: Math.abs(gx) },
    ),
  ];
  const finish = (converged: boolean, msg: string, nIter: number): Result =>
    result(method, x, fx, converged, msg, nIter, trace, {
      nFev: f.n,
      nGev: prob.grad ? g.n : 0,
      nHev: prob.hess ? h.n : 0,
    });
  if (!(Number.isFinite(fx) && Number.isFinite(gx) && Number.isFinite(hx)))
    return finish(false, `non-finite f, f' or f'' at x0 = ${g17(x)}`, 0);

  for (let k = 1; k <= maxIter; k++) {
    if (Math.abs(gx) <= gtol) {
      if (hx >= 0.0)
        return finish(true, `|f'(x)| = ${fmtG(Math.abs(gx))} ≤ gtol and f''(x) ≥ 0`, k - 1);
      return finish(
        false,
        `|f'(x)| ≤ gtol but f''(x) = ${fmtG(hx)} < 0: x is near a local maximum`,
        k - 1,
      );
    }
    const model: ParabolaInfo = { center: x, coef: [fx, gx, 0.5 * hx] };
    const trials: [number, number][] = [];
    let alpha: number | null = null;
    let kind: 'newton' | 'gradient';
    let xNew: number, fNew: number;
    if (hx > 0.0) {
      kind = 'newton';
      xNew = x - gx / hx;
      fNew = f(xNew);
    } else {
      kind = 'gradient';
      alpha = alpha0;
      let accepted = false;
      xNew = x;
      fNew = fx;
      for (let i = 0; i < MAX_BACKTRACK; i++) {
        xNew = x - alpha * gx;
        fNew = f(xNew);
        trials.push([alpha, fNew]);
        if (Number.isFinite(fNew) && fNew <= fx - ARMIJO_C1 * alpha * gx * gx) {
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }
      if (!accepted)
        return finish(
          false,
          `gradient-step backtracking failed after ${MAX_BACKTRACK} halvings at x = ${g17(x)}`,
          k - 1,
        );
    }
    const gNew = g(xNew),
      hNew = h(xNew);
    trace.push(
      step(
        k,
        xNew,
        fNew,
        { step: kind, g: gNew, h: hNew, parabola: model, alpha, trials },
        { gradNorm: Math.abs(gNew), stepSize: Math.abs(xNew - x) },
      ),
    );
    [x, fx, gx, hx] = [xNew, fNew, gNew, hNew];
    if (!(Number.isFinite(fx) && Number.isFinite(gx) && Number.isFinite(hx)))
      return finish(false, `non-finite f, f' or f'' at x = ${g17(x)} (left the domain?)`, k);
  }

  const converged = Math.abs(gx) <= gtol && hx >= 0.0;
  const msg = converged
    ? `|f'(x)| = ${fmtG(Math.abs(gx))} ≤ gtol and f''(x) ≥ 0`
    : `reached max_iter=${maxIter}`;
  return finish(converged, msg, maxIter);
};

// ---------------------------------------------------------------------------------------
// Bracketing a minimum (NR mnbrak)
// ---------------------------------------------------------------------------------------

export type BracketStep = 'init' | 'golden' | 'parabolic' | 'parabolic_far' | 'limit';

export const bracketMinimum: MethodFn<ScalarProblemInput> = (problem, o) => {
  const prob = scalarProblem(problem);
  const stepLen = num(o, 'step', 0.1);
  const growLimit = num(o, 'grow_limit', 100.0);
  const maxIter = num(o, 'max_iter', 50);
  requireMaxIter(maxIter);
  if (!(Number.isFinite(stepLen) && stepLen !== 0.0))
    throw new ScalarInputError(`step must be a non-zero finite number, got ${pyStr(stepLen)}`);
  if (!(Number.isFinite(growLimit) && growLimit > 1.0))
    throw new ScalarInputError(`grow_limit must be a finite number > 1, got ${pyStr(growLimit)}`);
  let xa = startScalar(prob, o.x0);
  if (!Number.isFinite(xa)) throw new ScalarInputError(`x0 must be finite, got ${pyStr(xa)}`);
  const method = 'bracket_minimum';
  const f = counted(prob.f as Func);

  let xb = xa + stepLen;
  if (xb === xa)
    throw new ScalarInputError(
      `step = ${fmtG(stepLen)} is lost in rounding at x0 = ${g17(xa)} (x0 + step == x0); ` +
        'use a larger step',
    );
  let fa = f(xa),
    fb = f(xb);
  const initTrials: [number, number][] = [
    [xa, fa],
    [xb, fb],
  ];
  if (!(Number.isFinite(fa) && Number.isFinite(fb))) {
    const [bad, fbad] = !Number.isFinite(fa) ? [xa, fa] : [xb, fb];
    const trace0 = [
      step(0, xa, fa, {
        bracket: [Math.min(xa, xb), Math.max(xa, xb)],
        triple: [xa, xb, xb],
        f_triple: [fa, fb, fb],
        step: 'init',
        trials: initTrials,
      }),
    ];
    return result(method, bad, fbad, false, nonfinite(method, bad, fbad), 0, trace0, { nFev: f.n });
  }
  if (fb > fa) [xa, xb, fa, fb] = [xb, xa, fb, fa]; // make a → b the downhill direction
  let xc = xb + GOLD * (xb - xa);
  let fc = f(xc);
  initTrials.push([xc, fc]);

  const info = (kind: BracketStep, trials: [number, number][]): StepInfo => ({
    bracket: [Math.min(xa, xc), Math.max(xa, xc)],
    triple: [xa, xb, xc],
    f_triple: [fa, fb, fc],
    step: kind,
    trials,
  });
  const sortedTriple = (): { triple: number[]; f_triple: number[] } => {
    const t = (
      [
        [xa, fa],
        [xb, fb],
        [xc, fc],
      ] as [number, number][]
    ).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    return { triple: t.map((e) => e[0]), f_triple: t.map((e) => e[1]) };
  };
  const trace: Step[] = [
    step(0, xb, fb, info('init', initTrials), { stepSize: Math.abs(xb - xa) }),
  ];
  const done = (k: number): Result => {
    const extra = sortedTriple();
    const [lo, mid, hi] = extra.triple;
    if (!(lo < mid && mid < hi)) {
      const msg =
        `the triple collapsed in floating point (${g17(xa)}, ${g17(xb)}, ${g17(xc)}): ` +
        'two points are equal, so f(b) ≤ f(c) is no bracket';
      return result(method, xb, fb, false, msg, k, trace, { nFev: f.n }, extra);
    }
    const msg = `bracket found: f(b) ≤ f(a) and f(b) ≤ f(c) with width ${fmtG(Math.abs(xc - xa))}`;
    return result(method, xb, fb, true, msg, k, trace, { nFev: f.n }, extra);
  };
  if (!Number.isFinite(fc))
    return result(
      method,
      xc,
      fc,
      false,
      nonfinite(method, xc, fc),
      0,
      trace,
      { nFev: f.n },
      sortedTriple(),
    );

  for (let k = 1; k <= maxIter; k++) {
    if (fb <= fc) return done(k - 1);
    const r = (xb - xa) * (fb - fc);
    const q = (xb - xc) * (fb - fa);
    const qr = q - r;
    const den = 2.0 * (qr >= 0.0 ? Math.max(Math.abs(qr), TINY) : -Math.max(Math.abs(qr), TINY));
    let u = xb - ((xb - xc) * q - (xb - xa) * r) / den;
    const ulim = xb + growLimit * (xc - xb);
    const trials: [number, number][] = [];
    const ev = (z: number): number => {
      const fz = f(z);
      trials.push([z, fz]);
      return fz;
    };
    let found = false;
    let kind: BracketStep;
    let fu: number;
    if ((xb - u) * (u - xc) > 0.0) {
      // parabolic u between b and c
      kind = 'parabolic';
      fu = ev(u);
      if (!Number.isFinite(fu)) {
        // reported below as a non-finite stop
      } else if (fu < fc) {
        // minimum between b and c
        [xa, xb, fa, fb] = [xb, u, fb, fu];
        found = true;
      } else if (fu > fb) {
        // minimum between a and u
        [xc, fc] = [u, fu];
        found = true;
      } else {
        // the parabola did not help: default magnification
        kind = 'golden';
        u = xc + GOLD * (xc - xb);
        fu = ev(u);
      }
    } else if ((xc - u) * (u - ulim) > 0.0) {
      // parabolic u between c and its limit
      kind = 'parabolic_far';
      fu = ev(u);
      if (Number.isFinite(fu) && fu < fc) {
        [xb, xc, u] = [xc, u, u + GOLD * (u - xc)];
        [fb, fc] = [fc, fu];
        fu = ev(u);
      }
    } else if ((u - ulim) * (ulim - xc) >= 0.0) {
      // limit u to its maximum allowed value
      kind = 'limit';
      u = ulim;
      fu = ev(u);
    } else {
      // reject the parabolic u: default magnification
      kind = 'golden';
      u = xc + GOLD * (xc - xb);
      fu = ev(u);
    }
    if (!Number.isFinite(fu)) {
      trace.push(step(k, xb, fb, info(kind, trials), { stepSize: Math.abs(u - xb) }));
      return result(
        method,
        u,
        fu,
        false,
        nonfinite(method, u, fu),
        k,
        trace,
        { nFev: f.n },
        sortedTriple(),
      );
    }
    if (!found) {
      [xa, xb, xc] = [xb, xc, u];
      [fa, fb, fc] = [fb, fc, fu];
    }
    trace.push(step(k, xb, fb, info(kind, trials), { stepSize: Math.abs(xc - xa) }));
    if (found) return done(k);
  }

  if (fb <= fc) return done(maxIter);
  const msg = `no bracket after max_iter=${maxIter} expansions (f may be unbounded below in this direction)`;
  return result(method, xb, fb, false, msg, maxIter, trace, { nFev: f.n }, sortedTriple());
};

// ---------------------------------------------------------------------------------------
// Registration (ids, params, needs, order, summaries and references as in Python)
// ---------------------------------------------------------------------------------------

const INTERVAL_QUANTITIES: MethodDoc['quantities'] = [
  { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
  { tex: 'b_k - a_k', key: 'stepSize' },
  { tex: '[x_1,\\, x_2]', key: 'info.interior' },
];

registerMethod<ScalarProblemInput>(
  {
    id: 'golden_section',
    family: 'scalar',
    name: 'Golden-section search',
    params: [XTOL_PARAM, maxIterParam(200)],
    needs: ['f', 'bracket'],
    order: 'linear (rate 1/φ ≈ 0.618)',
    summary:
      'Compare f at two golden-ratio points, discard the worse end; one of the old points is ' +
      'reused, so each step costs one new evaluation.',
    references: [
      'Kiefer (1953), Sequential minimax search for a maximum, Proc. AMS 4, 502–506',
      'Press et al., Numerical Recipes (3rd ed.), §10.2 (routine Golden)',
      'Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2',
    ],
  },
  goldenSection,
  {
    rule: '\\begin{aligned} x_1 &= a_k + \\rho\\,(b_k - a_k)\\\\ x_2 &= b_k - \\rho\\,(b_k - a_k), \\quad \\rho = \\tfrac{3-\\sqrt5}{2} \\approx 0.382\\\\ [a_{k+1}, b_{k+1}] &= \\begin{cases} [a_k, x_2] & f(x_1) \\le f(x_2)\\\\ [x_1, b_k] & \\text{otherwise} \\end{cases} \\end{aligned}',
    intuition:
      'Two probes at the golden points split the bracket ρ : 1 − 2ρ : ρ; the worse probe’s ' +
      'outer piece is cut away. Because ρ² − 3ρ + 1 = 0, the surviving probe already sits at a ' +
      'golden point of the new bracket, so each step costs one evaluation and shrinks the width by 1/φ.',
    order: 'linear · 1/φ',
    pros: [
      'Needs only f-comparisons; no derivatives, no smoothness',
      'Guaranteed width (b₀ − a₀)·φ²⁻ⁿ after n ≥ 2 evaluations',
    ],
    cons: [
      'Linear only: ignores curvature that parabolic steps exploit',
      'Finds a local minimizer when f is not unimodal',
    ],
    quantities: INTERVAL_QUANTITIES,
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'fibonacci_search',
    family: 'scalar',
    name: 'Fibonacci search',
    params: [
      XTOL_PARAM,
      param.float('eps_ratio', 0.05, {
        min: 1e-3,
        max: 0.5,
        log: true,
        help: 'Shift ε of the last step as a fraction of the final Fibonacci interval (b₀ − a₀)/Fₙ.',
        label: 'Last-step shift',
        tex: '\\varepsilon F_n / L_1',
      }),
      maxIterParam(200),
    ],
    needs: ['f', 'bracket'],
    order: 'linear (optimal for a fixed number of evaluations)',
    summary:
      'Plan n evaluations in advance so the final interval is as small as possible; points sit ' +
      'at Fibonacci ratios and each step reuses one old point.',
    references: [
      'Kiefer (1953), Sequential minimax search for a maximum, Proc. AMS 4, 502–506',
      'Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (Fibonacci search)',
      'Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (Fibonacci and golden section search)',
    ],
  },
  fibonacciSearch,
  {
    rule: '\\begin{aligned} \\lambda_k &= b_k - r_k\\,(b_k - a_k)\\\\ \\mu_k &= a_k + r_k\\,(b_k - a_k), \\quad r_k = \\tfrac{F_{n-k-1}}{F_{n-k}}\\\\ [a_{k+1}, b_{k+1}] &= \\begin{cases} [\\lambda_k, b_k] & f(\\lambda_k) > f(\\mu_k)\\\\ [a_k, \\mu_k] & \\text{otherwise} \\end{cases} \\end{aligned}',
    intuition:
      'With the number of evaluations n fixed in advance, Fibonacci ratios Fₙ₋ₖ₋₁/Fₙ₋ₖ ' +
      'give the smallest possible final interval, L₁/Fₙ + ε. The ratios tend to 1/φ, so the run ' +
      'looks like golden section until the last steps, where the ratio drops to ½ and an ε-shifted pair decides.',
    order: 'linear · minimax optimal',
    pros: ['Optimal worst-case final width for n evaluations (Kiefer 1953)'],
    cons: [
      'n must be planned from the tolerance before the first evaluation',
      'Gains over golden section only about 17 % in width',
    ],
    quantities: [
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: 'b_k - a_k', key: 'stepSize' },
      { tex: '[\\lambda_k,\\, \\mu_k]', key: 'info.interior' },
      { tex: 'F_{n-k-1}/F_{n-k}', key: 'info.ratio' },
    ],
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'dichotomous_search',
    family: 'scalar',
    name: 'Dichotomous search',
    params: [
      param.float('xtol', 1e-6, {
        min: DICHOTOMOUS_XTOL_MIN,
        max: 1e-1,
        log: true,
        help: 'Stop when the bracket width b − a ≤ xtol; then |x − x⋆| ≤ max(xtol, h) for unimodal f, with h ≈ √ε·(1 + |x⋆|). A small δ = delta_ratio·xtol can make rounding decide a comparison first: the run then stops with converged=False.',
        label: 'Bracket tolerance',
        tex: 'b - a \\le',
      }),
      param.float('delta_ratio', 0.1, {
        min: 1e-3,
        max: 0.9,
        log: true,
        help: 'Separation δ = delta_ratio·xtol of the two points around the midpoint (must be < xtol and < the bracket width b − a). A smaller δ gives smaller f-differences that rounding hides sooner.',
        label: 'Probe separation',
        tex: '\\delta / \\text{xtol}',
      }),
      maxIterParam(200),
    ],
    needs: ['f', 'bracket'],
    order: 'linear (rate ½ per two evaluations)',
    summary:
      'Evaluate f just left and right of the midpoint and keep the half that contains the lower ' +
      'value; the bracket almost halves each step.',
    references: [
      'Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (dichotomous search)',
    ],
  },
  dichotomousSearch,
  {
    rule: '\\begin{aligned} x_{1,2} &= \\tfrac{a_k + b_k}{2} \\mp \\tfrac{\\delta}{2}\\\\ [a_{k+1}, b_{k+1}] &= \\begin{cases} [a_k, x_2] & f(x_1) \\le f(x_2)\\\\ [x_1, b_k] & \\text{otherwise} \\end{cases} \\end{aligned}',
    intuition:
      'Two probes δ apart at the midpoint tell which half slopes down. The width obeys ' +
      'L′ = L/2 + δ/2, so it nearly halves per step but costs two evaluations: √½ ≈ 0.707 per evaluation.',
    order: 'linear · 0.707 / eval',
    pros: ['Simplest elimination rule; nearly halves the bracket each step'],
    cons: [
      'Two new evaluations per step, none reused',
      'Probes only δ apart: rounding decides comparisons early',
    ],
    quantities: INTERVAL_QUANTITIES,
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'ternary_search',
    family: 'scalar',
    name: 'Ternary search',
    params: [XTOL_PARAM, maxIterParam(200)],
    needs: ['f', 'bracket'],
    order: 'linear (rate 2/3 per two evaluations)',
    summary:
      'Split the bracket into thirds, compare f at the two inner points and drop the outer third ' +
      'next to the worse one.',
    references: [
      'Bazaraa, Sherali & Shetty, Nonlinear Programming (3rd ed.), §8.2 (interval elimination)',
    ],
  },
  ternarySearch,
  {
    rule: '\\begin{aligned} x_1 &= a_k + \\tfrac{b_k - a_k}{3}, \\quad x_2 = b_k - \\tfrac{b_k - a_k}{3}\\\\ [a_{k+1}, b_{k+1}] &= \\begin{cases} [a_k, x_2] & f(x_1) \\le f(x_2)\\\\ [x_1, b_k] & \\text{otherwise} \\end{cases} \\end{aligned}',
    intuition:
      'Thirds look balanced, but neither old probe lands on a third of the new bracket, so both ' +
      'are thrown away: 2/3 per two evaluations is √(2/3) ≈ 0.816 per evaluation, worse than golden section’s 0.618.',
    order: 'linear · 0.816 / eval',
    pros: ['Easy to state and to prove'],
    cons: ['Wastes an evaluation per step compared with golden section'],
    quantities: INTERVAL_QUANTITIES,
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'parabolic_interpolation',
    family: 'scalar',
    name: 'Successive parabolic interpolation',
    params: [
      param.float('xtol', 1e-8, {
        min: 1e-12,
        max: 1e-1,
        log: true,
        help: 'Stop when xₘ is within xtol/2 of both ends of the bracket (so its width ≤ xtol). f-comparisons limit the accuracy to h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).',
        label: 'Bracket tolerance',
        tex: 'b - a \\le',
      }),
      maxIterParam(200),
    ],
    needs: ['f', 'bracket'],
    order: 'linear when an end point stays fixed (the unbracketed variant is superlinear, ≈ 1.324)',
    summary:
      'Fit a parabola through three points that bracket the minimum and jump to its vertex; fall ' +
      'back to golden-section steps when the fit is unusable or makes too little progress, and to ' +
      'bisection until the three points bracket the minimum. The setup evaluates f at both ' +
      "bracket ends, so f must be finite at a and b (choose a bracket inside the domain; Brent's " +
      'method evaluates interior points only).',
    references: [
      'Press et al., Numerical Recipes (3rd ed.), §10.3, eq. 10.3.1',
      'Brent (1973), Algorithms for Minimization without Derivatives, Ch. 5 (progress test)',
      'Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (line search by quadratic fit)',
      'Antoniou & Lu, Practical Optimization (2007), Ch. 4 (quadratic interpolation method)',
    ],
  },
  parabolicInterpolation,
  {
    rule: '\\begin{aligned} u &= x_m - \\tfrac{1}{2}\\,N/D\\\\ N &= (x_m - x_l)^2 (f_m - f_r) - (x_m - x_r)^2 (f_m - f_l)\\\\ D &= (x_m - x_l)(f_m - f_r) - (x_m - x_r)(f_m - f_l) \\end{aligned}',
    intuition:
      'Near a smooth minimum f looks like a parabola, so the vertex of the parabola through the ' +
      'three points xₗ < xₘ < xᵣ is a good guess. The new point replaces an end so the ' +
      'pattern f(xₘ) ≤ min(f(xₗ), f(xᵣ)) holds again; golden and bisection steps guard against bad fits.',
    order: 'linear',
    pros: [
      'Uses the curvature: few steps on smooth, nearly quadratic minima',
      'One evaluation per step',
    ],
    cons: [
      'An end point that never moves makes it only linear',
      'Evaluates f at both bracket ends',
    ],
    quantities: [
      { tex: '(x_l, x_m, x_r)', key: 'info.triple' },
      { tex: 'b_k - a_k', key: 'stepSize' },
      { tex: 'u', key: 'info.trial' },
    ],
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'brent_minimize',
    family: 'scalar',
    name: "Brent's method (minimization)",
    params: [
      param.float('xtol', 1e-8, {
        min: 1e-12,
        max: 1e-1,
        log: true,
        help: 'Absolute tolerance t: stop when x is within 2(rtol·|x| + xtol/3) of both bracket ends. f-comparisons limit the accuracy to h ≈ √ε·(1 + |x⋆|) (about 10⁻⁸).',
        label: 'Absolute tolerance',
        tex: 't',
      }),
      param.float('rtol', SQRT_EPS, {
        min: 1e-10,
        max: 1e-3,
        log: true,
        help: "Relative tolerance (Brent's eps; √ε by default). Below √ε the guarantee 2(rtol·|x| + xtol/3) can fail: f-comparisons resolve x⋆ only to about √ε·(1 + |x⋆|).",
        label: 'Relative tolerance',
        tex: '\\epsilon_r',
      }),
      maxIterParam(500),
    ],
    needs: ['f', 'bracket'],
    order: 'superlinear (≈ 1.324) for smooth f; guaranteed convergence otherwise',
    summary:
      'Golden-section search that switches to parabolic interpolation whenever the parabola is ' +
      'trustworthy; the default 1-D minimizer in most libraries. Its accuracy is ' +
      '2(rtol·|x| + xtol/3), so far from 0 the relative part rtol·|x| dominates xtol.',
    references: [
      'Brent (1973), Algorithms for Minimization without Derivatives, Ch. 5 (localmin)',
      'Forsythe, Malcolm & Moler (1977), Computer Methods for Mathematical Computations, fmin',
      'Press et al., Numerical Recipes (3rd ed.), §10.3 (routine Brent)',
    ],
  },
  brentMinimize,
  {
    rule: '\\begin{aligned} u &= x + \\tfrac{p}{q} \\ \\text{(parabola through } x, w, v\\text{)} \\quad \\text{if } \\left|\\tfrac{p}{q}\\right| < \\tfrac12 |e| \\text{ and } u \\in (a, b)\\\\ u &= x + \\rho\\,(b - x \\text{ or } a - x) \\quad \\text{(golden step) otherwise} \\end{aligned}',
    intuition:
      'Brent keeps the best point x, the second best w and the previous w, v. It takes the ' +
      'parabola’s vertex when the step is less than half the step before last, and a golden ' +
      'step into the larger part otherwise — superlinear on smooth minima, never slower than golden section by much.',
    order: 'superlinear · ≈ 1.324',
    pros: [
      'Superlinear on smooth f, guaranteed on any unimodal f',
      'Never evaluates at the bracket ends',
    ],
    cons: [
      'Accuracy 2(rtol·|x| + xtol/3), relative far from 0',
      'More bookkeeping than golden section',
    ],
    quantities: [
      { tex: '(x, w, v)', key: 'info.xwv' },
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: 'u', key: 'info.trial' },
      { tex: '\\text{tol}', key: 'info.tol' },
    ],
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'newton_1d',
    family: 'scalar',
    name: "Newton's method (1-D minimization)",
    params: [
      param.float('gtol', 1e-8, {
        min: 1e-14,
        max: 1e-2,
        log: true,
        help: 'Stop when |f′(x)| ≤ gtol and f″(x) ≥ 0 (necessary conditions only: x can be an inflection point).',
        label: 'Derivative tolerance',
        tex: "|f'| \\le",
      }),
      param.float('alpha0', 1.0, {
        min: 1e-4,
        max: 10.0,
        log: true,
        help: 'First trial length of the gradient step used when f″(x) ≤ 0.',
        label: 'Safeguard step',
        tex: '\\alpha_0',
      }),
      maxIterParam(100),
    ],
    needs: ['f', 'grad', 'hess'],
    order: 'quadratic near a minimizer with f″ > 0',
    summary:
      'Jump to the minimizer of the local quadratic model x − f′/f″; when the model is not convex (f″ ≤ 0), take a backtracking gradient step instead.',
    references: [
      'Nocedal & Wright, Numerical Optimization (2nd ed.), §3.3 (Newton) and Alg. 3.1 (backtracking)',
      "Luenberger & Ye, Linear and Nonlinear Programming (4th ed.), Ch. 8 (Newton's method for line search)",
    ],
  },
  newton1d,
  {
    rule: "x_{k+1} = \\begin{cases} x_k - \\dfrac{f'(x_k)}{f''(x_k)} & f''(x_k) > 0\\\\[4pt] x_k - \\alpha_k f'(x_k) \\ \\text{(Armijo)} & \\text{otherwise} \\end{cases}",
    intuition:
      'Fit the Taylor parabola f(xₖ) + f′(xₖ)(z − xₖ) + ½ f″(xₖ)(z − xₖ)² and jump to its ' +
      'vertex. Near a minimizer with f″ > 0 the error squares each step; where f″ ≤ 0 the model has ' +
      'no minimum, so a backtracking gradient step is taken instead.',
    order: 'quadratic',
    pros: ['Quadratic convergence near a nondegenerate minimizer'],
    cons: ['Needs f′ and f″', 'Converges to any stationary point with f″ ≥ 0, even an inflection'],
    quantities: [
      { tex: "f'(x_k)", key: 'info.g' },
      { tex: "f''(x_k)", key: 'info.h' },
      { tex: '|x_k - x_{k-1}|', key: 'stepSize' },
    ],
  },
);

registerMethod<ScalarProblemInput>(
  {
    id: 'bracket_minimum',
    family: 'scalar',
    name: 'Bracketing a minimum',
    params: [
      param.float('step', 0.1, {
        min: 1e-6,
        max: 10.0,
        log: true,
        help: 'First step: the second point is x₀ + step.',
        label: 'First step',
        tex: 'h',
      }),
      param.float('grow_limit', 100.0, {
        min: 1.5,
        max: 1000.0,
        log: true,
        help: 'A parabolic extrapolation may reach at most uₗᵢₘ = b + grow_limit·(c − b) (must be > 1, so uₗᵢₘ lies beyond c).',
        label: 'Growth limit',
        tex: 'g',
      }),
      maxIterParam(50),
    ],
    needs: ['f'],
    order: 'geometric expansion (ratio φ)',
    summary:
      'Walk downhill with steps that grow by the golden ratio (or a parabolic jump) until f turns ' +
      'up: the last three points bracket a minimum.',
    references: [
      'Press et al., Numerical Recipes (3rd ed.), §10.1 (Bracketmethod::bracket, mnbrak)',
    ],
  },
  bracketMinimum,
  {
    rule: 'c_{k+1} = c_k + \\varphi\\,(c_k - b_k) \\quad \\text{until} \\quad f(b) \\le f(a),\\ f(b) \\le f(c)',
    intuition:
      'Start with a downhill pair a → b and step on with lengths that grow by φ (or jump to a ' +
      'parabolic extrapolation) until f turns up. The last three points a, b, c then bracket a ' +
      'minimum: the input every interval method needs.',
    order: 'expansion · φ',
    pros: ['Needs only f and a start point'],
    cons: [
      'Finds a bracket, not a minimizer',
      'Runs away when f is unbounded below in the downhill direction',
    ],
    quantities: [
      { tex: '(a, b, c)', key: 'info.triple' },
      { tex: '|c - a|', key: 'stepSize' },
    ],
  },
);
