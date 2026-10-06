/**
 * Bracketing root finders — TS port of `numopt.roots.bracketing` (src/numopt/roots/bracketing.py).
 *
 * f(x) = 0 on an interval [a, b] with f(a)·f(b) < 0. The methods keep a sign change inside the
 * current interval, so they cannot diverge. Conventions (identical to Python):
 *
 *   - Start: `bracket = [a, b]` (or the problem's default) with finite a < b whose width does not
 *     overflow, f finite at both ends and of opposite signs, xtol ≥ 0 (> 0 for ITP); else an
 *     `Error` (Python `ValueError`). If f(a) or f(b) is exactly zero, that end is returned at once.
 *   - Trace: step k holds the k-th newly evaluated estimate x_k and f(x_k); k = 0 is the first
 *     estimate computed from the initial bracket. `info.bracket` is the bracket x_k was computed
 *     from; `info.new_bracket` the bracket after the sign test. `nIter == trace.at(-1).k`.
 *   - Tolerance: tol(x) = xtol + 2ε|x| (Brent 1973, ch. 4).
 *   - Converged: f(x_k) == 0, |f(x_k)| ≤ ftol, or the new bracket has half-width ≤ tol (strict <
 *     for Chandrupatla; ITP: b − a ≤ 2·xtol). The false-position family and Ridders use the a
 *     posteriori step test only as a trigger for a verifying probe x_k ± tol.
 *   - Failure: max_iter or a non-finite f(x_k) → converged = false.
 *
 * Step.info keys are the Python ones (snake_case): bracket, new_bracket; bisection a, b, fa, fb;
 * false-position family step, chord, scale, estimate; ridders step, midpoint, f_mid, exp_factor,
 * transformed, estimate; brent step, attempted, points, tol, best; chandrupatla step, t, points,
 * xi, phi, best; itp x_half, x_f, x_t, delta, r, projected, best. Python `None` is `null`.
 *
 * The module also exports the Python float semantics both root modules need (`copysign`,
 * `nextafter`, `ulp`, Python `min`/`max`, `repr` and `format(x, "g")` for messages).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type {
  ParamSpec,
  Params,
  Problem,
  Result,
  RunOptions,
  Step,
  StepInfo,
} from '../../core/types';
import { crHypot } from './crmath';

// ---------------------------------------------------------------------------------------
// Python float semantics
// ---------------------------------------------------------------------------------------

/** Machine epsilon of float64 (`sys.float_info.epsilon`). */
export const EPS = Number.EPSILON;

/** True when the sign bit of x is set (−0 included). */
export function signbit(x: number): boolean {
  return x < 0 || Object.is(x, -0);
}

/** `math.copysign(mag, sgn)`. */
export function copysign(mag: number, sgn: number): number {
  const a = Math.abs(mag);
  return signbit(sgn) ? -a : a;
}

/** Python `max(a, b)`: the first argument unless the second is strictly greater (NaN-aware). */
export function pyMax(a: number, b: number): number {
  return b > a ? b : a;
}

/** Python `min(a, b)`: the first argument unless the second is strictly smaller. */
export function pyMin(a: number, b: number): number {
  return b < a ? b : a;
}

/** `numopt.core.counting.finite`: every value is a finite number. */
export function finite(...values: number[]): boolean {
  return values.every((v) => Number.isFinite(v));
}

const F64 = new Float64Array(1);
const I64 = new BigInt64Array(F64.buffer);

/** `math.nextafter(x, toward)`: the next float after x in the direction of `toward`. */
export function nextafter(x: number, toward: number): number {
  if (Number.isNaN(x) || Number.isNaN(toward)) return NaN;
  if (x === toward) return toward;
  if (x === 0) return toward > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  F64[0] = x;
  // Moving away from zero increments the magnitude bits; toward zero decrements them.
  if (toward > x === x > 0) I64[0] += 1n;
  else I64[0] -= 1n;
  return F64[0];
}

/** `math.ulp(x)`: the value of the least significant bit of |x|. */
export function ulp(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  if (!Number.isFinite(a)) return a;
  if (a === 0) return Number.MIN_VALUE;
  const up = nextafter(a, Infinity);
  if (!Number.isFinite(up)) return a - nextafter(a, -Infinity);
  return up - a;
}

/** `math.ldexp(m, e)` = m·2ᵉ exactly (±Infinity where Python raises OverflowError). */
export function ldexp(m: number, e: number): number {
  if (m === 0 || !Number.isFinite(m)) return m;
  let r = m;
  let n = e;
  while (n > 1000) {
    r *= 2 ** 1000;
    n -= 1000;
  }
  while (n < -1000) {
    r *= 2 ** -1000;
    n += 1000;
  }
  // Split the last factor so no intermediate power of two underflows.
  if (n < -1022) {
    r *= 2 ** -1022;
    n += 1022;
  }
  return r * 2 ** n;
}

/** Exact decomposition of a finite float: |x| = mant·2^exp with an integer mantissa. */
function decompose(x: number): { mant: bigint; exp: number } {
  F64[0] = Math.abs(x);
  const bits = I64[0];
  const biased = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & 0xfffffffffffffn;
  if (biased === 0) return { mant, exp: -1074 };
  mant |= 1n << 52n;
  return { mant, exp: biased - 1075 };
}

const bitLength = (v: bigint) => (v === 0n ? 0 : v.toString(2).length);

/**
 * Python `repr(float)`: the shortest round-trip digits, in fixed notation for decimal exponents
 * −4 ≤ e < 16 and scientific (`1e-05`, `1.5e+16`) otherwise; integral values keep `.0`.
 */
export function pyRepr(x: number): string {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return signbit(x) ? '-0.0' : '0.0';
  const sign = x < 0 ? '-' : '';
  const [mant, e] = Math.abs(x).toExponential().split('e');
  const digits = mant.replace('.', '');
  const exp = Number(e);
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      const int = digits.slice(0, exp + 1).padEnd(exp + 1, '0');
      const frac = digits.slice(exp + 1) || '0';
      return `${sign}${int}.${frac}`;
    }
    return `${sign}0.${'0'.repeat(-exp - 1)}${digits}`;
  }
  const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${sign}${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
}

/**
 * |x| rounded to p significant decimal digits, half-to-even on the EXACT binary value (as
 * CPython does; `toPrecision` rounds exact ties up, e.g. 2⁻¹⁰). Returns the p digits and the
 * decimal exponent of the first one.
 */
function decimalDigits(x: number, p: number): { digits: string; exp: number } {
  const { mant, exp: e2 } = decompose(x);
  let exp = Number(Math.abs(x).toExponential().split('e')[1]);
  const n0 = e2 >= 0 ? mant << BigInt(e2) : mant;
  const d0 = e2 >= 0 ? 1n : 1n << BigInt(-e2);
  const lowest = 10n ** BigInt(p - 1);
  const scaled = (e: number) => {
    const shift = p - 1 - e;
    const n = shift >= 0 ? n0 * 10n ** BigInt(shift) : n0;
    const d = shift >= 0 ? d0 : d0 * 10n ** BigInt(-shift);
    return { n, d, q: n / d };
  };
  let { n, d, q } = scaled(exp);
  while (q < lowest) {
    exp -= 1;
    ({ n, d, q } = scaled(exp));
  }
  const r2 = 2n * (n - q * d);
  if (r2 > d || (r2 === d && q % 2n === 1n)) q += 1n;
  if (q === 10n ** BigInt(p)) {
    q /= 10n;
    exp += 1;
  }
  return { digits: q.toString(), exp };
}

/** Python `format(x, ".{p}g")`, digit for digit (round-half-even on the exact value). */
export function fmtG(x: number, precision = 6): string {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  const p = precision === 0 ? 1 : precision;
  if (x === 0) return signbit(x) ? '-0' : '0';
  const sign = x < 0 ? '-' : '';
  const { digits, exp } = decimalDigits(x, p);
  if (exp >= -4 && exp < p) {
    const int = exp >= 0 ? digits.slice(0, exp + 1) : '0';
    const frac = (exp >= 0 ? digits.slice(exp + 1) : '0'.repeat(-exp - 1) + digits).replace(
      /0+$/,
      '',
    );
    return `${sign}${int}${frac ? `.${frac}` : ''}`;
  }
  const tail = digits.slice(1).replace(/0+$/, '');
  return `${sign}${digits[0]}${tail ? `.${tail}` : ''}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------------------
// Evaluation counting and problem resolution (numopt.core.counting)
// ---------------------------------------------------------------------------------------

export type ScalarFn = (x: number) => number;

/**
 * `Counted(SafeScalar(fn))`: counts calls; `error` names a RangeError thrown by f (Python turns
 * ArithmeticError/ValueError into nan; JavaScript arithmetic never throws, so this is rare).
 */
export class Counted {
  n = 0;
  error: string | null = null;
  readonly fn: (x: number) => unknown;
  constructor(fn: (x: number) => unknown) {
    this.fn = fn;
  }
  call(x: number): number {
    this.n += 1;
    this.error = null;
    try {
      return Number(this.fn(x));
    } catch (e) {
      if (!(e instanceof RangeError)) throw e;
      this.error = `${e.name}: ${e.message}`;
      return NaN;
    }
  }
}

/** Message suffix naming the exception of the last evaluation of f ("" if none). */
export function raised(f: Counted): string {
  return f.error !== null ? `; f raised ${f.error}` : '';
}

/** A 1-D problem as the root methods see it (a library problem or a bare callable). */
export interface ScalarProblem {
  id: string;
  f: (x: number) => unknown;
  grad?: ((x: number) => unknown) | null;
  hess?: ((x: number) => unknown) | null;
  x0?: number | number[] | null;
  bracket?: readonly [number, number] | readonly number[] | null;
}

/** `scalar_problem`: accept a Problem or a bare `f(x)`. */
export function scalarProblem(problem: unknown): ScalarProblem {
  if (typeof problem === 'function') return { id: 'custom', f: problem as ScalarFn };
  if (problem && typeof problem === 'object' && typeof (problem as Problem).f === 'function')
    return problem as ScalarProblem;
  throw new TypeError('problem must be a numopt Problem or a callable f(x)');
}

export function makeStep(
  k: number,
  x: number,
  fun: number,
  stepSize: number | null = null,
  info: StepInfo = {},
): Step {
  return { k, x, fun, gradNorm: null, stepSize, info };
}

export function makeResult(
  method: string,
  x: number,
  fun: number,
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
  more: { nGev?: number; nHev?: number; extra?: Record<string, unknown> } = {},
): Result {
  return {
    method,
    x,
    fun,
    converged,
    message,
    nIter,
    nFev,
    nGev: more.nGev ?? 0,
    nHev: more.nHev ?? 0,
    trace,
    extra: more.extra ?? {},
  };
}

/** A float parameter with its Python default. */
export function num(v: unknown, def: number): number {
  return v === undefined || v === null ? def : Number(v);
}

// ---------------------------------------------------------------------------------------
// Shared bracketing machinery
// ---------------------------------------------------------------------------------------

const COMMON_PARAMS: ParamSpec[] = [
  param.float('xtol', 1e-10, {
    min: 1e-15,
    max: 1e-2,
    log: true,
    help: 'Stop when the bracket half-width ≤ xtol + 2ε|x|.',
    label: 'Bracket tolerance',
    tex: '\\tfrac{b-a}{2} \\le',
  }),
  param.float('ftol', 0.0, {
    min: 0.0,
    max: 1e-2,
    help: 'Also stop when |f(x)| ≤ ftol (0 disables).',
    label: 'Residual tolerance',
    tex: '|f| \\le',
  }),
  param.int('max_iter', 200, {
    min: 1,
    max: 10_000,
    help: 'Iteration limit.',
    label: 'Iteration budget',
  }),
];

/** Brent's tolerance xtol + 2ε|x| (Brent 1973, ch. 4). */
export function tolOf(x: number, xtol: number): number {
  return xtol + 2.0 * EPS * Math.abs(x);
}

export function sameSign(u: number, v: number): boolean {
  return signbit(u) === signbit(v);
}

function sorted2(u: number, v: number): [number, number] {
  return u <= v ? [u, v] : [v, u];
}

function resolveBracket(prob: ScalarProblem, bracket: unknown): [number, number] {
  const br = (bracket ?? prob.bracket) as readonly number[] | null | undefined;
  if (br === null || br === undefined) throw new Error(`${prob.id}: a bracket (a, b) is required`);
  const a = Number(br[0]),
    b = Number(br[1]);
  if (!(Number.isFinite(a) && Number.isFinite(b)))
    throw new Error(`invalid bracket: the ends must be finite, got (${pyRepr(a)}, ${pyRepr(b)})`);
  if (!(a < b)) throw new Error(`invalid bracket: need a < b, got (${pyRepr(a)}, ${pyRepr(b)})`);
  if (!Number.isFinite(b - a))
    throw new Error(
      `invalid bracket: the width b − a of (${pyRepr(a)}, ${pyRepr(b)}) overflows the float range; ` +
        'use a narrower bracket',
    );
  return [a, b];
}

function checkXtol(xtol: number, positive = false): void {
  const ok = Number.isFinite(xtol) && (positive ? xtol > 0.0 : xtol >= 0.0);
  if (!ok)
    throw new Error(`xtol must be finite and ${positive ? '> 0' : '≥ 0'}, got ${pyRepr(xtol)}`);
}

interface Start {
  f: Counted;
  a: number;
  b: number;
  fa: number;
  fb: number;
  early: Result | null;
}

/** Validate xtol and the bracket, evaluate both ends and check the sign change. */
function start(method: string, problem: unknown, bracket: unknown, xtol: number): Start {
  checkXtol(xtol, method === 'itp');
  const prob = scalarProblem(problem);
  const [a, b] = resolveBracket(prob, bracket);
  const f = new Counted(prob.f);
  const fa = f.call(a);
  let note = raised(f);
  const fb = f.call(b);
  note = note || raised(f);
  if (!finite(fa, fb))
    throw new Error(
      `f must be finite at the bracket ends; got f(${pyRepr(a)})=${pyRepr(fa)}, ` +
        `f(${pyRepr(b)})=${pyRepr(fb)}${note}`,
    );
  for (const x of [a, b]) {
    const fx = x === a ? fa : fb;
    if (fx === 0.0) {
      const info = { bracket: [a, b], new_bracket: [x, x] };
      const trace = [makeStep(0, x, 0.0, null, info)];
      const msg = 'f is exactly zero at a bracket end';
      return { f, a, b, fa, fb, early: makeResult(method, x, 0.0, true, msg, 0, f.n, trace) };
    }
  }
  if (sameSign(fa, fb))
    throw new Error(
      `f(a) and f(b) must have opposite signs; got f(${pyRepr(a)})=${pyRepr(fa)}, ` +
        `f(${pyRepr(b)})=${pyRepr(fb)}`,
    );
  return { f, a, b, fa, fb, early: null };
}

/** The shared bracketing stopping test; returns the reason, or null to continue. */
function stopReason(fx: number, halfWidth: number, tol: number, ftol: number): string | null {
  if (fx === 0.0) return 'f(x) is exactly zero';
  if (Math.abs(fx) <= ftol) return `|f(x)| = ${fmtG(Math.abs(fx), 3)} ≤ ftol`;
  if (halfWidth <= tol) return `bracket half-width ${fmtG(halfWidth, 3)} ≤ tol = ${fmtG(tol, 3)}`;
  return null;
}

/**
 * A posteriori error estimate s_k·max(1, ρ/(1 − ρ)) with ρ = s_k/s_{k−1} < 1, else null
 * (`step_estimate`; Burden & Faires §2.2). Only a trigger for the verifying probe.
 */
export function stepEstimate(step: number, prevStep: number | null): number | null {
  if (prevStep === null || !(step < prevStep)) return null;
  const rho = step / prevStep;
  return step * pyMax(1.0, rho / (1.0 - rho));
}

/** Step-test value over consecutive points of one rule (0 when the last two coincide). */
function estimateOf(points: number[]): number | null {
  if (points.length < 2) return null;
  const n = points.length;
  const step = Math.abs(points[n - 1] - points[n - 2]);
  if (step === 0.0) return 0.0;
  const prev = n >= 3 ? Math.abs(points[n - 2] - points[n - 3]) : null;
  return stepEstimate(step, prev);
}

function nonfinite(
  method: string,
  f: Counted,
  x: number,
  fx: number,
  k: number,
  trace: Step[],
): Result {
  const msg = `f(x) is not finite at x = ${pyRepr(x)} (f = ${pyRepr(fx)}${raised(f)})`;
  return makeResult(method, x, fx, false, msg, k, f.n, trace);
}

function maxIterResult(
  method: string,
  x: number,
  fx: number,
  maxIter: number,
  nfev: number,
  trace: Step[],
): Result {
  return makeResult(method, x, fx, false, `reached max_iter=${maxIter}`, maxIter, nfev, trace);
}

interface Common {
  bracket: unknown;
  xtol: number;
  ftol: number;
  maxIter: number;
}

function common(o: RunOptions & Params): Common {
  return {
    bracket: o.bracket,
    xtol: num(o.xtol, 1e-10),
    ftol: num(o.ftol, 0.0),
    maxIter: num(o.max_iter, 200),
  };
}

// ---------------------------------------------------------------------------------------
// Bisection
// ---------------------------------------------------------------------------------------

/**
 * Bisection (Burden & Faires, Alg. 2.1): m = a + (b − a)/2; keep [a, m] if f(a)·f(m) < 0, else
 * [m, b]. Stops when the half-width of the bracket that contains m is ≤ xtol + 2ε|m|, when
 * |f(m)| ≤ ftol, or when f(m) == 0.
 */
export function bisection(problem: unknown, o: RunOptions & Params): Result {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const s = start('bisection', problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let { a, b, fa, fb } = s;
  const info = (a: number, b: number, fa: number, fb: number, m: number, fm: number) => {
    const nw = fm === 0.0 ? [m, m] : !sameSign(fa, fm) ? [a, m] : [m, b];
    return { bracket: [a, b], new_bracket: nw, a, b, fa, fb };
  };
  let m = a + 0.5 * (b - a);
  let fm = f.call(m);
  const trace = [makeStep(0, m, fm, null, info(a, b, fa, fb, m, fm))];
  let k = 0;
  for (;;) {
    if (!finite(fm)) return nonfinite('bisection', f, m, fm, k, trace);
    // NOTE: the half-width test uses the bracket that contains m (before the split).
    const reason = stopReason(fm, 0.5 * (b - a), tolOf(m, xtol), ftol);
    if (reason !== null) return makeResult('bisection', m, fm, true, reason, k, f.n, trace);
    if (k === maxIter) return maxIterResult('bisection', m, fm, maxIter, f.n, trace);
    k += 1;
    if (!sameSign(fa, fm)) {
      b = m;
      fb = fm;
    } else {
      a = m;
      fa = fm;
    }
    m = a + 0.5 * (b - a);
    fm = f.call(m);
    trace.push(makeStep(k, m, fm, 0.5 * (b - a), info(a, b, fa, fb, m, fm)));
  }
}

// ---------------------------------------------------------------------------------------
// False position and its Illinois-type modifications
// ---------------------------------------------------------------------------------------

/** Zero b − f_b(b − a)/(f_b − f_a) of the chord through (a, fa), (b, fb), fa·fb < 0. */
function chordPoint(a: number, fa: number, b: number, fb: number): number {
  const x = b - (fb * (b - a)) / (fb - fa);
  if (Number.isFinite(x)) return x;
  const den = fb - fa;
  const w = Number.isFinite(den) ? fb / den : (0.5 * fb) / (0.5 * fb - 0.5 * fa);
  return b - w * (b - a);
}

/** x clamped to [lo, hi] and, if it lands on an end, moved one ulp inward (lo < hi). */
function inside(x: number, lo: number, hi: number): number {
  let y = x;
  if (y <= lo) y = nextafter(lo, hi);
  else if (y >= hi) y = nextafter(hi, lo);
  return pyMin(pyMax(y, lo), hi);
}

/** `chordPoint` kept inside [a, b] (rounding can push it one ulp outside). */
export function chordRoot(a: number, fa: number, b: number, fb: number): number {
  const x = chordPoint(a, fa, b, fb);
  const [lo, hi] = a <= b ? [a, b] : [b, a];
  return pyMin(pyMax(x, lo), hi);
}

type FalsePositionMethod = 'regula_falsi' | 'illinois' | 'pegasus' | 'anderson_bjorck';

/** Factor m for the retained end when f_k and f_{k+1} have the same sign. */
function scaleFactor(method: FalsePositionMethod, fK: number, fNext: number): number {
  if (method === 'regula_falsi') return 1.0;
  if (method === 'illinois') return 0.5;
  if (method === 'pegasus') return fK / (fK + fNext);
  const m = 1.0 - fNext / fK;
  return m > 0.0 ? m : 0.5;
}

/**
 * Generic Illinois-type false position (Ford 1995, Algorithm 1) with the verified step test of
 * the module docstring: an a posteriori step estimate ≤ tol triggers the probe b ± tol toward a;
 * if it keeps the sign of f(b), the next point is the bisection point.
 */
function falsePosition(method: FalsePositionMethod, problem: unknown, o: RunOptions & Params) {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const s = start(method, problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let { a, b, fa, fb } = s;
  const trace: Step[] = [];
  let chordPoints: number[] = [];
  let kind: 'chord' | 'verify' | 'bisection' = 'chord';
  let k = 0;
  for (;;) {
    const oldBracket = sorted2(a, b);
    let chord: number[][] | null = null;
    let x: number;
    if (kind === 'chord') {
      chord = [
        [a, fa],
        [b, fb],
      ];
      x = chordRoot(a, fa, b, fb);
    } else if (kind === 'verify') {
      x = b + copysign(tolOf(b, xtol), a - b);
    } else {
      x = oldBracket[0] + 0.5 * (oldBracket[1] - oldBracket[0]);
    }
    const fx = f.call(x);
    const step = trace.length ? Math.abs(x - (trace[trace.length - 1].x as number)) : null;
    const info: StepInfo = { bracket: oldBracket, step: kind, chord };
    if (!finite(fx)) {
      trace.push(makeStep(k, x, fx, step, info));
      return nonfinite(method, f, x, fx, k, trace);
    }
    let scale = 1.0;
    if (fx !== 0.0) {
      if (!sameSign(fx, fb)) {
        a = b;
        fa = fb;
      } else if (kind === 'chord') {
        scale = scaleFactor(method, fb, fx);
        fa *= scale;
      }
    }
    b = x;
    fb = fx;
    const newBracket = fx === 0.0 ? [x, x] : sorted2(a, b);
    let estimate: number | null = null;
    if (kind === 'chord') {
      chordPoints.push(x);
      estimate = estimateOf(chordPoints);
    }
    Object.assign(info, { new_bracket: newBracket, scale, estimate });
    trace.push(makeStep(k, x, fx, step, info));

    const tol = tolOf(x, xtol);
    let reason = stopReason(fx, 0.5 * (newBracket[1] - newBracket[0]), tol, ftol);
    if (reason !== null) {
      if (kind === 'verify' && reason.startsWith('bracket'))
        reason += ' (f changes sign across the probe xₖ₋₁ ± tol)';
      return makeResult(method, x, fx, true, reason, k, f.n, trace);
    }
    if (k === maxIter) return maxIterResult(method, x, fx, maxIter, f.n, trace);
    if (kind === 'chord') {
      kind = estimate !== null && estimate <= tol ? 'verify' : 'chord';
    } else {
      // The probe kept the sign: the step test was misled, so bisect once.
      kind = kind === 'verify' ? 'bisection' : 'chord';
      chordPoints = [];
    }
    k += 1;
  }
}

export const regulaFalsi = (p: unknown, o: RunOptions & Params) =>
  falsePosition('regula_falsi', p, o);
export const illinois = (p: unknown, o: RunOptions & Params) => falsePosition('illinois', p, o);
export const pegasus = (p: unknown, o: RunOptions & Params) => falsePosition('pegasus', p, o);
export const andersonBjorck = (p: unknown, o: RunOptions & Params) =>
  falsePosition('anderson_bjorck', p, o);

// ---------------------------------------------------------------------------------------
// Ridders
// ---------------------------------------------------------------------------------------

/**
 * Ridders' method (Ridders 1979; Numerical Recipes §9.2.1): with m = (a + b)/2 choose Q = e^{λd}
 * so that f(a), f(m)Q, f(b)Q² are collinear; x = m + d·sign(f(a) − f(b))·f(m)/√(f(m)² − f(a)f(b)).
 * The tightest of [m, x], [a, x], [x, b] that keeps the sign change becomes the new bracket.
 */
export function ridders(problem: unknown, o: RunOptions & Params): Result {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const s = start('ridders', problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let lo = s.a,
    hi = s.b,
    flo = s.fa,
    fhi = s.fb;
  const trace: Step[] = [];
  let points: number[] = [];
  let kind: 'ridders' | 'verify' = 'ridders';
  let k = 0;
  for (;;) {
    const oldBracket = [lo, hi];
    const info: StepInfo = {
      bracket: oldBracket,
      step: kind,
      midpoint: null,
      f_mid: null,
      exp_factor: null,
      transformed: null,
    };
    let m = NaN,
      fm = NaN;
    let x: number, fx: number;
    if (kind === 'verify') {
      // The last Ridders point is an end of [lo, hi]; probe tol inside, toward the other.
      const xLast = trace[trace.length - 1].x as number;
      x = xLast + copysign(tolOf(xLast, xtol), xLast === lo ? 1.0 : -1.0);
      fx = f.call(x);
    } else {
      m = lo + 0.5 * (hi - lo);
      fm = f.call(m);
      Object.assign(info, { midpoint: m, f_mid: fm });
      if (!finite(fm)) {
        trace.push(makeStep(k, m, fm, null, { bracket: oldBracket }));
        return nonfinite('ridders', f, m, fm, k, trace);
      }
      if (fm === 0.0) {
        // NOTE: the Ridders point equals m exactly when f(m) = 0; skip re-evaluating it.
        x = m;
        fx = 0.0;
      } else {
        // √(f(m)² − f(lo)f(hi)) written as hypot to avoid overflow; f(lo)f(hi) < 0.
        // NOTE: correctly rounded like CPython's math.hypot (V8's Math.hypot is not).
        const sq = crHypot(fm, Math.sqrt(Math.abs(flo)) * Math.sqrt(Math.abs(fhi)));
        x = m + ((m - lo) * copysign(1.0, flo - fhi) * fm) / sq;
        // NOTE: clamp against a one-ulp overshoot of the rounded Ridders point.
        x = pyMin(pyMax(x, lo), hi);
        fx = f.call(x);
        // Q is the positive root of f(hi)Q² − 2f(m)Q + f(lo) = 0 (form without cancellation).
        const q = sameSign(fm, fhi)
          ? (fm + copysign(sq, fhi)) / fhi
          : flo / (fm - copysign(sq, fhi));
        Object.assign(info, {
          exp_factor: q,
          transformed: [
            [lo, flo],
            [m, fm * q],
            [hi, fhi * q * q],
          ],
        });
      }
    }
    const step = trace.length ? Math.abs(x - (trace[trace.length - 1].x as number)) : null;
    if (!finite(fx)) {
      trace.push(makeStep(k, x, fx, step, info));
      return nonfinite('ridders', f, x, fx, k, trace);
    }
    if (fx === 0.0) {
      lo = hi = x;
    } else if (kind === 'ridders' && !sameSign(fm, fx)) {
      // sorted(((m, fm), (x, fx))): by abscissa, then by value.
      if (m < x || (m === x && fm <= fx)) {
        lo = m;
        flo = fm;
        hi = x;
        fhi = fx;
      } else {
        lo = x;
        flo = fx;
        hi = m;
        fhi = fm;
      }
    } else if (!sameSign(flo, fx)) {
      hi = x;
      fhi = fx;
    } else {
      lo = x;
      flo = fx;
    }
    let estimate: number | null = null;
    if (kind === 'ridders') {
      points.push(x);
      estimate = estimateOf(points);
    }
    Object.assign(info, { new_bracket: [lo, hi], estimate });
    trace.push(makeStep(k, x, fx, step, info));
    const tol = tolOf(x, xtol);
    let reason = stopReason(fx, 0.5 * (hi - lo), tol, ftol);
    if (reason !== null) {
      if (kind === 'verify' && reason.startsWith('bracket'))
        reason += ' (f changes sign across the probe xₖ₋₁ ± tol)';
      return makeResult('ridders', x, fx, true, reason, k, f.n, trace);
    }
    if (k === maxIter) return maxIterResult('ridders', x, fx, maxIter, f.n, trace);
    if (kind === 'ridders' && estimate !== null && estimate <= tol) {
      kind = 'verify';
    } else {
      if (kind === 'verify') points = []; // the probe kept the sign: the step test was misled
      kind = 'ridders';
    }
    k += 1;
  }
}

// ---------------------------------------------------------------------------------------
// Brent
// ---------------------------------------------------------------------------------------

/**
 * Brent's method, a transcription of procedure `zero` (Brent 1973, ch. 4, §4.6): inverse
 * quadratic or secant steps from the best point b, accepted only inside ¾ of the way to the
 * contrapoint c and when shorter than half the step before last; otherwise bisection; never a
 * step shorter than tol. Stops when ½|c − b| ≤ tol, f(b) == 0 or |f(b)| ≤ ftol; returns b.
 */
export function brent(problem: unknown, o: RunOptions & Params): Result {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const s = start('brent', problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let { a, b, fa, fb } = s;
  let c = a,
    fc = fa;
  let d = b - a,
    e = b - a;
  if (Math.abs(fc) < Math.abs(fb)) {
    [a, b, c] = [b, c, b];
    [fa, fb, fc] = [fb, fc, fb];
  }
  const convergedReason = () => stopReason(fb, 0.5 * Math.abs(c - b), tolOf(b, xtol), ftol);

  let reason = convergedReason();
  if (reason !== null) {
    const info0 = { bracket: sorted2(b, c), new_bracket: sorted2(b, c), best: b };
    return makeResult('brent', b, fb, true, reason, 0, f.n, [makeStep(0, b, fb, null, info0)]);
  }

  const trace: Step[] = [];
  let k = 0;
  for (;;) {
    const tol = tolOf(b, xtol);
    const m = 0.5 * (c - b);
    const oldBracket = sorted2(b, c);
    let attempted: 'secant' | 'inverse_quadratic' | null = null;
    let points: number[][] = [];
    let kind: 'bisection' | 'secant' | 'inverse_quadratic' = 'bisection';
    if (Math.abs(e) >= tol && Math.abs(fa) > Math.abs(fb)) {
      const sr = fb / fa;
      let p: number, q: number;
      if (a === c) {
        attempted = 'secant';
        points = [
          [a, fa],
          [b, fb],
        ];
        p = 2.0 * m * sr;
        q = 1.0 - sr;
      } else {
        attempted = 'inverse_quadratic';
        points = [
          [a, fa],
          [b, fb],
          [c, fc],
        ];
        q = fa / fc;
        const r = fb / fc;
        p = sr * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (sr - 1.0);
      }
      if (p > 0.0) q = -q;
      else p = -p;
      const eOld = e;
      e = d;
      if (2.0 * p < 3.0 * m * q - Math.abs(tol * q) && p < Math.abs(0.5 * eOld * q)) {
        d = p / q;
        kind = attempted;
      } else {
        d = e = m;
      }
    } else {
      d = e = m;
    }
    a = b;
    fa = fb;
    const step = Math.abs(d) > tol ? d : copysign(tol, m);
    b = b + step;
    fb = f.call(b);
    const xNew = b,
      fNew = fb;
    const info: StepInfo = { bracket: oldBracket, step: kind, attempted, points, tol };
    if (!finite(fb)) {
      trace.push(makeStep(k, xNew, fNew, Math.abs(step), info));
      return nonfinite('brent', f, xNew, fNew, k, trace);
    }
    if (fb > 0.0 === fc > 0.0) {
      // Brent's "int": the sign change is between a and b.
      c = a;
      fc = fa;
      d = e = b - a;
    }
    if (Math.abs(fc) < Math.abs(fb)) {
      // Brent's "ext": keep the better point in b.
      [a, b, c] = [b, c, b];
      [fa, fb, fc] = [fb, fc, fb];
    }
    info.new_bracket = fb === 0.0 ? [b, b] : sorted2(b, c);
    info.best = b;
    trace.push(makeStep(k, xNew, fNew, Math.abs(step), info));
    reason = convergedReason();
    if (reason !== null) return makeResult('brent', b, fb, true, reason, k, f.n, trace);
    if (k === maxIter) return maxIterResult('brent', b, fb, maxIter, f.n, trace);
    k += 1;
  }
}

// ---------------------------------------------------------------------------------------
// Chandrupatla
// ---------------------------------------------------------------------------------------

/**
 * Chandrupatla's hybrid quadratic/bisection method (Chandrupatla 1997): the next point is
 * x₁ + t(x₂ − x₁) with the inverse quadratic t when 1 − √(1 − ξ) < Φ < √ξ, else t = ½; t is
 * clipped to [t_l, 1 − t_l], t_l = tol/|x₂ − x₁|. Stops when |x₂ − x₁| < 2·tol (strict), or
 * f(x_m) == 0, or |f(x_m)| ≤ ftol, where x_m is the end with the smaller |f|; returns x_m.
 */
export function chandrupatla(problem: unknown, o: RunOptions & Params): Result {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const s = start('chandrupatla', problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let x1 = s.a,
    x2 = s.b,
    f1 = s.fa,
    f2 = s.fb;
  // Python initializes x₃ = x₂; the first step always overwrites it before reading it.
  let x3: number, f3: number;
  let t = 0.5;
  let pending: StepInfo = { step: 'bisection', t, points: [], xi: null, phi: null };
  const trace: Step[] = [];
  let k = 0;
  for (;;) {
    const oldBracket = sorted2(x1, x2);
    const xt = inside(x1 + t * (x2 - x1), oldBracket[0], oldBracket[1]);
    const ft = f.call(xt);
    const info: StepInfo = { bracket: oldBracket, ...pending };
    if (!finite(ft)) {
      trace.push(makeStep(k, xt, ft, null, info));
      return nonfinite('chandrupatla', f, xt, ft, k, trace);
    }
    if (sameSign(ft, f1)) {
      x3 = x1;
      f3 = f1;
    } else {
      x3 = x2;
      f3 = f2;
      x2 = x1;
      f2 = f1;
    }
    x1 = xt;
    f1 = ft;
    const [xm, fm] = Math.abs(f1) < Math.abs(f2) ? [x1, f1] : [x2, f2];
    const tol = tolOf(xm, xtol);
    const dx = Math.abs(x2 - x1);
    info.new_bracket = ft === 0.0 ? [xt, xt] : sorted2(x1, x2);
    info.best = xm;
    trace.push(makeStep(k, xt, ft, null, info));
    let reason: string | null = null;
    if (fm === 0.0) reason = 'f(x) is exactly zero';
    else if (Math.abs(fm) <= ftol) reason = `|f(x)| = ${fmtG(Math.abs(fm), 3)} ≤ ftol`;
    else if (0.5 * dx < tol)
      // Chandrupatla's strict test t_l > ½ ⇔ |x₂ − x₁| < 2·tol
      reason = `bracket half-width ${fmtG(0.5 * dx, 3)} < tol = ${fmtG(tol, 3)}`;
    if (reason !== null) return makeResult('chandrupatla', xm, fm, true, reason, k, f.n, trace);
    if (k === maxIter) return maxIterResult('chandrupatla', xm, fm, maxIter, f.n, trace);
    // Choose the next t.
    const xi = (x1 - x2) / (x3 - x2);
    const phi = (f1 - f2) / (f3 - f2);
    let kind: 'inverse_quadratic' | 'bisection';
    let points: number[][];
    if (1.0 - Math.sqrt(pyMax(0.0, 1.0 - xi)) < phi && phi < Math.sqrt(pyMax(0.0, xi))) {
      const alpha = (x3 - x1) / (x2 - x1);
      t = ((f1 / (f1 - f2)) * f3) / (f3 - f2) - (((alpha * f1) / (f3 - f1)) * f2) / (f2 - f3);
      kind = 'inverse_quadratic';
      points = [
        [x1, f1],
        [x2, f2],
        [x3, f3],
      ];
    } else {
      t = 0.5;
      kind = 'bisection';
      points = [];
    }
    const tl = tol / dx;
    t = pyMin(1.0 - tl, pyMax(tl, t));
    pending = { step: kind, t, points, xi, phi };
    k += 1;
  }
}

// ---------------------------------------------------------------------------------------
// ITP
// ---------------------------------------------------------------------------------------

/**
 * n½ = ⌈log₂((b − a)/(2ε))⌉ ≥ 0, computed exactly: the least n ≥ 0 with 2ε·2ⁿ ≥ b − a (the ratio
 * is compared as an exact rational, like Python's `Fraction`).
 */
export function itpNHalf(a: number, b: number, eps: number): number {
  if (!(eps > 0.0 && Number.isFinite(eps) && Number.isFinite(b - a)))
    throw new Error(
      `itp_n_half needs ε > 0 and a finite width, got ε=${pyRepr(eps)}, b − a=${pyRepr(b - a)}`,
    );
  if (!(b - a > 0)) return 0;
  // (b − a)/(2ε) = m1·2^e1 / (m2·2^(e2+1)) = p/q
  const w = decompose(b - a),
    d = decompose(eps);
  const shift = w.exp - d.exp - 1;
  const p = shift >= 0 ? w.mant << BigInt(shift) : w.mant;
  const q = shift >= 0 ? d.mant : d.mant << BigInt(-shift);
  let n = Math.max(0, bitLength(p) - bitLength(q));
  while (q << BigInt(n) < p) n += 1;
  while (n > 0 && q << BigInt(n - 1) >= p) n -= 1;
  return n;
}

/**
 * The ITP method (Oliveira & Takahashi 2020, Algorithm 1) with ε = xtol: interpolation (regula
 * falsi point x_f), truncation toward x½ by δ = max(κ₁(b − a)^κ₂, ε/2), projection onto the
 * ball of radius r = ε′·2^{n_max − j} − (b − a)/2 around x½. Stops when b − a ≤ 2ε, f(x) == 0,
 * |f(x)| ≤ ftol, or the bracket ends are adjacent floats; returns the end with the smaller |f|.
 */
export function itp(problem: unknown, o: RunOptions & Params): Result {
  const { bracket, xtol, ftol, maxIter } = common(o);
  const kappa1 = num(o.kappa1, 0.1),
    kappa2 = num(o.kappa2, 2.0),
    n0 = num(o.n0, 1);
  if (!(kappa1 > 0.0)) throw new Error(`kappa1 must be > 0, got ${pyRepr(kappa1)}`);
  if (!(kappa2 >= 1.0 && kappa2 < 1.0 + 0.5 * (1.0 + Math.sqrt(5.0))))
    throw new Error(`kappa2 must lie in [1, 1 + φ), got ${pyRepr(kappa2)}`);
  if (n0 < 0) throw new Error(`n0 must be ≥ 0, got ${n0}`);
  const s = start('itp', problem, bracket, xtol);
  if (s.early) return s.early;
  const f = s.f;
  let { a, b, fa, fb } = s;
  const eps = xtol;
  const nHalf = itpNHalf(a, b, eps);
  const nMax = nHalf + n0;
  const epsR = pyMax(eps - 2.0 * ulp(pyMax(Math.abs(a), Math.abs(b))), 0.5 * eps);

  const best = (): [number, number] => (Math.abs(fa) <= Math.abs(fb) ? [a, fa] : [b, fb]);

  if (b - a <= 2.0 * eps) {
    const [x, fx] = best();
    const info0 = { bracket: [a, b], new_bracket: [a, b], best: x };
    const msg = `bracket width ${fmtG(b - a, 3)} ≤ 2·xtol`;
    return makeResult('itp', x, fx, true, msg, 0, f.n, [makeStep(0, x, fx, null, info0)]);
  }

  const trace: Step[] = [];
  let k = 0;
  for (;;) {
    const oldBracket = [a, b];
    const width = b - a;
    const xHalf = a + 0.5 * width;
    // NOTE: ε′ < ε absorbs rounding; r is clipped at 0 (it would turn negative once j > n_max).
    const radius = ldexp(epsR, nMax - k);
    const r = pyMax(0.0, radius - 0.5 * width);
    // NOTE: δ is floored at ε/2 (see the Python docstring).
    const delta = pyMax(kappa1 * width ** kappa2, 0.5 * eps);
    const xF = chordRoot(a, fa, b, fb);
    const sigma = xHalf !== xF ? copysign(1.0, xHalf - xF) : 0.0;
    const xT = delta <= Math.abs(xHalf - xF) ? xF + sigma * delta : xHalf;
    const projected = Math.abs(xT - xHalf) > r;
    const x = projected ? xHalf - sigma * r : xT;
    const fx = f.call(x);
    const info: StepInfo = {
      bracket: oldBracket,
      x_half: xHalf,
      x_f: xF,
      x_t: xT,
      delta,
      r,
      projected,
    };
    if (!finite(fx)) {
      trace.push(makeStep(k, x, fx, null, info));
      return nonfinite('itp', f, x, fx, k, trace);
    }
    if (fx === 0.0) {
      a = b = x;
      fa = fb = 0.0;
    } else if (sameSign(fx, fa)) {
      a = x;
      fa = fx;
    } else {
      b = x;
      fb = fx;
    }
    const [xb, fxb] = best();
    info.new_bracket = [a, b];
    info.best = xb;
    trace.push(makeStep(k, x, fx, null, info));
    let reason: string | null = null;
    if (fx === 0.0) reason = 'f(x) is exactly zero';
    else if (Math.abs(fx) <= ftol) reason = `|f(x)| = ${fmtG(Math.abs(fx), 3)} ≤ ftol`;
    else if (b - a <= 2.0 * eps) reason = `bracket width ${fmtG(b - a, 3)} ≤ 2·xtol`;
    else if (nextafter(a, Infinity) >= b)
      reason = 'bracket ends are adjacent floating-point numbers';
    if (reason !== null) return makeResult('itp', xb, fxb, true, reason, k, f.n, trace);
    if (k === maxIter) return maxIterResult('itp', xb, fxb, maxIter, f.n, trace);
    k += 1;
  }
}

// ---------------------------------------------------------------------------------------
// Registration (same ids, params and metadata as the Python @register calls)
// ---------------------------------------------------------------------------------------

const BRACKET_Q = [
  { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
  { tex: '[a_{k+1},\\, b_{k+1}]', key: 'info.new_bracket' },
];

const DOCS: Record<string, MethodDoc> = {
  bisection: {
    rule: 'x_k = \\frac{a_k + b_k}{2}, \\qquad [a_{k+1}, b_{k+1}] = \\begin{cases} [a_k, x_k] & f(a_k) f(x_k) < 0 \\\\ [x_k, b_k] & \\text{otherwise} \\end{cases}',
    intuition:
      'Evaluate f at the midpoint and keep the half on which f changes sign. Each step halves the bracket, whatever f looks like — one binary digit of the root per evaluation.',
    order: 'linear, rate ½',
    pros: [
      'Cannot fail once f(a)·f(b) < 0',
      'Iteration count known in advance: ⌈log₂((b − a)/tol)⌉ − 1 steps',
    ],
    cons: [
      'Ignores the values of f: 33 steps for 10 digits',
      'Finds only roots with a sign change',
    ],
    quantities: BRACKET_Q,
  },
  regula_falsi: {
    rule: 'x_k = b_k - f(b_k)\\,\\frac{b_k - a_k}{f(b_k) - f(a_k)}',
    intuition:
      'Replace f by the chord through the bracket ends and take its zero. On a convex or concave f one end never moves, so the chord pivots about it and the bracket stops shrinking.',
    order: 'linear (one end fixed)',
    pros: ['Uses the values of f, not only their signs', 'Keeps the bracket'],
    cons: ['Stalls when one end stays fixed (x¹⁰ − 1: 95 steps)'],
    quantities: [
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: '\\text{step}', key: 'info.step' },
    ],
  },
  illinois: {
    rule: 'x_k = b_k - f_b\\,\\frac{b_k - a_k}{f_b - f_a}, \\qquad f_a \\leftarrow \\tfrac12 f_a \\ \\text{if } a \\text{ is retained}',
    intuition:
      'Regula falsi, but when the same end is kept twice its stored value of f is halved. The next chord then swings toward that end and frees it.',
    order: 'superlinear, ≈ 1.442 per evaluation',
    pros: ['Removes the fixed-end stall of regula falsi', 'One evaluation per step'],
    cons: ['Order below the secant method'],
    quantities: [
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: 'm', key: 'info.scale' },
    ],
  },
  pegasus: {
    rule: 'x_k = b_k - f_b\\,\\frac{b_k - a_k}{f_b - f_a}, \\qquad f_a \\leftarrow \\frac{f_k}{f_k + f_{k+1}}\\, f_a',
    intuition:
      'Like Illinois, but the retained end is scaled by f_k/(f_k + f_{k+1}), a factor that the latest two values of f choose. The chord turns faster toward the stuck end.',
    order: 'superlinear, ≈ 1.642 per evaluation',
    pros: ['Faster than Illinois at no extra cost'],
    cons: ['Still a chord method: steep f slows the first steps'],
    quantities: [
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: 'm', key: 'info.scale' },
    ],
  },
  anderson_bjorck: {
    rule: 'x_k = b_k - f_b\\,\\frac{b_k - a_k}{f_b - f_a}, \\qquad f_a \\leftarrow \\Bigl(1 - \\frac{f_{k+1}}{f_k}\\Bigr) f_a',
    intuition:
      'The retained end is scaled by 1 − f_{k+1}/f_k, the factor that gives the parabola through the last three points the right slope; ½ when that factor is not positive.',
    order: 'superlinear, ≈ 1.7 per evaluation',
    pros: ['The fastest Illinois-type method on smooth f'],
    cons: ['Falls back to ½ (Illinois) when the factor is ≤ 0'],
    quantities: [
      { tex: '[a_k,\\, b_k]', key: 'info.bracket' },
      { tex: 'm', key: 'info.scale' },
    ],
  },
  ridders: {
    rule: 'x_k = m + (m - a)\\,\\frac{\\operatorname{sign}(f_a - f_b)\\, f_m}{\\sqrt{f_m^2 - f_a f_b}}, \\quad m = \\tfrac{a+b}{2}',
    intuition:
      "Multiply f by an exponential e^{λ(x−a)} chosen so that the three values at a, m, b lie on a line, then take that line's zero. The bracket at least halves per step.",
    order: 'quadratic per step, √2 per evaluation',
    pros: ['Never worse than bisection', 'Robust and simple'],
    cons: ['Two evaluations per step'],
    quantities: [
      { tex: 'm', key: 'info.midpoint' },
      { tex: 'Q = e^{\\lambda (m-a)}', key: 'info.exp_factor' },
    ],
  },
  brent: {
    rule: 'b_{k+1} = b_k + \\begin{cases} p/q & \\text{interpolation accepted} \\\\ \\tfrac12 (c_k - b_k) & \\text{otherwise} \\end{cases}',
    intuition:
      'Try inverse quadratic interpolation (or a secant step) from the best point; accept it only when it stays well inside the bracket and shrinks the step, else bisect. Fast on smooth f, never much slower than bisection.',
    order: 'superlinear, ≈ 1.84; never much worse than bisection',
    pros: ['The default bracketing solver (zeroin, fzero, brentq)', 'Guaranteed convergence'],
    cons: ['The acceptance tests are subtle to reimplement'],
    quantities: [
      { tex: '\\text{step}', key: 'info.step' },
      { tex: '\\text{tried}', key: 'info.attempted' },
      { tex: 'b \\ (\\text{best})', key: 'info.best' },
    ],
  },
  chandrupatla: {
    rule: 'x_{k} = x_1 + t\\,(x_2 - x_1), \\quad t = \\begin{cases} t_{\\text{IQI}} & 1 - \\sqrt{1-\\xi} < \\Phi < \\sqrt{\\xi} \\\\ \\tfrac12 & \\text{otherwise} \\end{cases}',
    intuition:
      'Inverse quadratic interpolation is used only where a test on the last three points says the interpolant is monotone on the bracket; otherwise bisect. Simpler than Brent, with the same speed.',
    order: 'superlinear (inverse quadratic when valid)',
    pros: ['One clear validity test instead of Brent’s heuristics'],
    cons: ['Less widely known; few library implementations'],
    quantities: [
      { tex: 't', key: 'info.t' },
      { tex: '\\xi', key: 'info.xi' },
      { tex: '\\Phi', key: 'info.phi' },
    ],
  },
  itp: {
    rule: 'x_{\\text{ITP}} = \\begin{cases} x_t & |x_t - x_{1/2}| \\le r_k \\\\ x_{1/2} - \\sigma r_k & \\text{otherwise} \\end{cases}, \\quad x_t = x_f + \\sigma \\delta',
    intuition:
      'Interpolate with regula falsi, truncate the point toward the midpoint by δ, then project it into a ball around the midpoint whose radius keeps the worst case within n₀ steps of bisection.',
    order: 'superlinear; worst case n½ + n₀ steps',
    pros: ['Worst case provably at most n₀ steps more than bisection', 'Superlinear on smooth f'],
    cons: ['Three parameters (κ₁, κ₂, n₀)'],
    quantities: [
      { tex: 'x_{1/2}', key: 'info.x_half' },
      { tex: 'x_f', key: 'info.x_f' },
      { tex: '\\delta', key: 'info.delta' },
      { tex: 'r_k', key: 'info.r' },
    ],
  },
};

type Meta = {
  id: string;
  name: string;
  order: string;
  summary: string;
  references: string[];
  params?: ParamSpec[];
};

function reg(meta: Meta, fn: (p: unknown, o: RunOptions & Params) => Result) {
  registerMethod(
    {
      id: meta.id,
      family: 'roots',
      name: meta.name,
      params: meta.params ?? COMMON_PARAMS,
      needs: ['f', 'bracket'],
      order: meta.order,
      summary: meta.summary,
      references: meta.references,
    },
    fn,
    DOCS[meta.id],
  );
}

reg(
  {
    id: 'bisection',
    name: 'Bisection',
    order: 'linear (rate ½)',
    summary: 'Halve the bracket and keep the half where f changes sign.',
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.1'],
  },
  bisection,
);
reg(
  {
    id: 'regula_falsi',
    name: 'Regula falsi (false position)',
    order: 'linear (one end usually stays fixed)',
    summary:
      'Draw the chord between the bracket ends and keep the sub-bracket where f changes sign.',
    references: ['Burden & Faires, Numerical Analysis (10th ed.), Alg. 2.5'],
  },
  regulaFalsi,
);
reg(
  {
    id: 'illinois',
    name: 'Illinois',
    order: 'superlinear (≈ 1.442 per evaluation)',
    summary: "Regula falsi that halves the retained end's f-value when that end is kept again.",
    references: [
      'Dowell & Jarratt (1971), BIT 11, 168–174',
      'Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1',
    ],
  },
  illinois,
);
reg(
  {
    id: 'pegasus',
    name: 'Pegasus',
    order: 'superlinear (≈ 1.642 per evaluation)',
    summary: "Regula falsi that scales the retained end's f-value by f_k/(f_k + f_{k+1}).",
    references: [
      'Dowell & Jarratt (1972), BIT 12, 503–508',
      'Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1',
    ],
  },
  pegasus,
);
reg(
  {
    id: 'anderson_bjorck',
    name: 'Anderson–Björck',
    order: 'superlinear (≈ 1.7 per evaluation)',
    summary: "Regula falsi that scales the retained end's f-value by 1 − f_{k+1}/f_k (or ½).",
    references: [
      'Anderson & Björck (1973), BIT 13, 253–264',
      'Ford (1995), Improved algorithms of Illinois-type, Univ. of Essex CSM-257, Alg. 1',
    ],
  },
  andersonBjorck,
);
reg(
  {
    id: 'ridders',
    name: 'Ridders',
    order: 'superlinear (quadratic per iteration, √2 per evaluation)',
    summary:
      "Factor out an exponential so the midpoint lies on a line, then take that line's zero.",
    references: [
      'Ridders (1979), IEEE Trans. Circuits Syst. 26(11), 979–980',
      'Press et al., Numerical Recipes (3rd ed.), §9.2.1 (zriddr)',
    ],
  },
  ridders,
);
reg(
  {
    id: 'brent',
    name: 'Brent (zeroin)',
    order: 'superlinear (≈ 1.84 with inverse quadratic steps); never much worse than bisection',
    summary:
      'Inverse quadratic / secant steps, with a bisection fallback whenever they are unsafe.',
    references: [
      'Brent (1973), Algorithms for Minimization without Derivatives, ch. 4, procedure zero',
      'Press et al., Numerical Recipes (3rd ed.), §9.3 (zbrent)',
    ],
  },
  brent,
);
reg(
  {
    id: 'chandrupatla',
    name: 'Chandrupatla',
    order: 'superlinear (inverse quadratic when locally valid, else bisection)',
    summary:
      'Inverse quadratic interpolation only where the data say it is valid; bisect otherwise.',
    references: ['Chandrupatla (1997), Advances in Engineering Software 28(3), 145–149'],
  },
  chandrupatla,
);
reg(
  {
    id: 'itp',
    name: 'ITP (interpolate–truncate–project)',
    order: 'superlinear on smooth f; worst case ⌈log₂((b−a)/(2·xtol))⌉ + n₀ iterations',
    summary:
      "Regula falsi, nudged toward the midpoint and kept within bisection's worst-case budget.",
    references: ['Oliveira & Takahashi (2020), ACM Trans. Math. Softw. 47(1), Art. 5, Alg. 1'],
    params: [
      ...COMMON_PARAMS,
      param.float('kappa1', 0.1, {
        min: 1e-4,
        max: 10.0,
        log: true,
        help: 'Truncation size κ₁ in δ = κ₁(b − a)^κ₂ (κ₁ > 0).',
        label: 'Truncation size',
        tex: '\\kappa_1',
      }),
      param.float('kappa2', 2.0, {
        min: 1.0,
        max: 2.6,
        help: 'Truncation exponent κ₂ ∈ [1, 1 + φ) (φ = golden ratio).',
        label: 'Truncation exponent',
        tex: '\\kappa_2',
      }),
      param.int('n0', 1, {
        min: 0,
        max: 20,
        help: 'Slack: at most n0 more iterations than bisection in the worst case.',
        label: 'Slack over bisection',
        tex: 'n_0',
      }),
    ],
  },
  itp,
);

/** Parity fixtures of the Python module (method id, problem id, params) — for tests. */
export const FIXTURE_CASES: [string, string, Record<string, unknown>][] = [
  ['bisection', 'sqrt2', {}],
  ['regula_falsi', 'x10_minus_1', {}],
  ['illinois', 'x10_minus_1', {}],
  ['pegasus', 'steep_exp', {}],
  ['anderson_bjorck', 'cubic', {}],
  ['ridders', 'kepler', {}],
  ['brent', 'wilkinson5', {}],
  ['chandrupatla', 'cos_minus_x', {}],
  ['itp', 'steep_exp', {}],
  ['illinois', 'steep_exp', { bracket: [-50.0, 60.0] }],
];
