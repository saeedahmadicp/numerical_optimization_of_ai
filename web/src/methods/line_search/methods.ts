/**
 * Line searches — TS port of `numopt.line_search.methods` (src/numopt/line_search/methods.py).
 *
 * All searches work on the one-dimensional restriction
 *
 *     φ(α) = f(x + α p),   φ'(α) = ∇f(x + α p)ᵀ p,   φ'(0) < 0,
 *
 * and accept a step by one of these tests (Nocedal & Wright (2006), §3.1):
 *
 *   Armijo / sufficient decrease (3.4):  φ(α) − φ(0) ≤ c₁ α φ'(0)
 *   curvature (3.6b):                    φ'(α) ≥ c₂ φ'(0)
 *   strong curvature (3.7b):             |φ'(α)| ≤ c₂ |φ'(0)|
 *   Goldstein (3.11):                    (1 − c) α φ'(0) ≤ φ(α) − φ(0) ≤ c α φ'(0)
 *
 * Two public surfaces, exactly as in Python:
 *
 *   - `search(kind, f, grad, x, p, opts)` — the helper every n-D descent method calls. It returns
 *     `{alpha, fNew, gNew, nFev, nGev, nHev, success, trials, message}` and counts exactly the
 *     calls it makes (including f(x) / ∇f(x) when `f0` / `g0` are not given). It throws
 *     `LineSearchInputError` (Python `ValueError`) on invalid input and `LineSearchZeroDivision`
 *     (Python `ZeroDivisionError`) where Python's float division by zero raises.
 *   - five registered demo methods (family `line_search`; ids `backtracking`, `strong_wolfe`,
 *     `weak_wolfe`, `goldstein`, `exact_quadratic`) that run ONE line search from x0 along the
 *     steepest-descent or Newton direction and record one Step per trial step.
 *
 * Step.info keys are the Python ones (snake_case): alpha, phi, dphi, phi0, dphi0, c1, c2,
 * direction, phase, interval, accepted, conditions, and per kind rho / alpha_lo, alpha_hi, interp /
 * pHp, model_phi. Python `None` is `null`; a NaN float stays NaN here (the Python exporter writes it
 * as JSON null).
 *
 * Floating-point operations keep the Python order (`x + alpha * p`, `phi - phi0 <= c * alpha *
 * dphi0`, ...), so traces agree with the Python fixtures to rounding.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { luFactor, luSolve, norm } from '../../core/linalg';
import type {
  Matrix,
  MethodFn,
  Params,
  Point,
  Problem,
  Result,
  RunOptions,
  Step,
  StepInfo,
  Vector,
} from '../../core/types';

// ---------------------------------------------------------------------------------------
// Public constants and types
// ---------------------------------------------------------------------------------------

export const KINDS = [
  'backtracking',
  'strong_wolfe',
  'weak_wolfe',
  'goldstein',
  'exact_quadratic',
] as const;
export type LineSearchKind = (typeof KINDS)[number];

/** Default Armijo constant c₁ (N&W p. 33). */
export const C1_DEFAULT = 1e-4;
/** Default Goldstein constant c ∈ (0, ½). */
export const C_GOLDSTEIN_DEFAULT = 0.25;
/** Default curvature constant c₂ (N&W p. 34: 0.9 for Newton and quasi-Newton directions). */
export const C2_DEFAULT = 0.9;

/** Zoom safeguard: an interpolated trial is clamped into [a + δw, b − δw]. */
const SAFEGUARD = 0.1;
/** Zoom stall test (Moré & Thuente 1994, §4): bisect when the bracket did not shrink by 0.66. */
const STALL = 0.66;
/** exact_quadratic: c₂ of the strong curvature test that confirms the line minimizer. */
const EXACT_C2 = 0.1;

/** Python `ValueError`: invalid parameters or a non-descent direction. */
export class LineSearchInputError extends Error {
  override name = 'ValueError';
}

/**
 * Python `ZeroDivisionError` ("float division by zero"): the zoom's quadratic interpolant divides
 * by h², which underflows to 0 for brackets narrower than ≈ 1.5e-162. Python callers that guard
 * against `ArithmeticError` (conjugate gradient) catch this one.
 */
export class LineSearchZeroDivision extends Error {
  override name = 'ZeroDivisionError';
  constructor() {
    super('float division by zero');
  }
}

/** Outcome of `search` (camelCase mirror of Python's `LineSearchResult`). */
export interface LineSearchResult {
  /** The accepted step (0 if the search failed). */
  alpha: number;
  /** f(x + alpha p) (equals f(x) on failure). */
  fNew: number;
  /** ∇f(x + alpha p) when it was computed (∇f(x) on failure), else null. */
  gNew: Vector | null;
  nFev: number;
  nGev: number;
  /** ∇²f evaluations (`exact_quadratic` with a callable `hess` only). */
  nHev: number;
  success: boolean;
  /** Every (alpha, φ(alpha)) evaluated, in order (φ(0) is not a trial). */
  trials: [number, number][];
  message: string;
}

export type HessianInput = ((x: Vector) => Matrix | number) | Matrix | number;

export interface SearchOptions {
  /** f(x) if already known (not re-evaluated). */
  f0?: number | null;
  /** ∇f(x) if already known (not re-evaluated). */
  g0?: readonly number[] | null;
  /** First trial step (unused by `exact_quadratic`). Default 1. */
  alpha0?: number;
  /** ∇²f as a callable or a matrix (a number for n = 1); required by `exact_quadratic`. */
  hess?: HessianInput | null;
  /** Armijo constant; for `goldstein` the Goldstein constant c. Default 1e-4 (goldstein 0.25). */
  c1?: number | null;
  /** Curvature constant c₂ ∈ (c₁, 1) for the Wolfe kinds. Default 0.9. */
  c2?: number;
  /** Backtracking contraction factor ρ ∈ (0, 1). Default 0.5. */
  rho?: number;
  /** Maximum number of trial steps. Default 50. */
  maxIter?: number;
  /** Finite upper limit on α for the expanding kinds. Default 1e3. */
  alphaMax?: number;
  /** `exact_quadratic` only: bound (≥ 0, absolute) on the rounding error of each value of f. */
  fErr?: number;
}

// ---------------------------------------------------------------------------------------
// Python-style number formatting for messages (`f"{v:.6g}"`)
// ---------------------------------------------------------------------------------------

/**
 * |v| rounded to `p` significant decimal digits with round-half-even on the EXACT binary value
 * (as CPython does; JS `toFixed`/`toPrecision` round exact ties up, e.g. 2⁻¹⁰). Returns the digit
 * string (length p) and the decimal exponent of its first digit.
 */
function decimalDigits(v: number, p: number): { digits: string; exp: number } {
  const a = Math.abs(v);
  // The shortest round-trip repr gives floor(log10 |v|) (or one more, for subnormals).
  let exp = Number(a.toExponential().split('e')[1]);
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, a);
  const hi = view.getUint32(0),
    lo = view.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e2 = -1074;
  if (biased !== 0) {
    mant |= 1n << 52n;
    e2 = biased - 1075;
  }
  // a = n0 / d0 exactly; scaled by 10^(p − 1 − exp) the integer part must have p digits.
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
  // The estimate can be one too high for subnormals (1e-320 is 9.99989e-321 exactly).
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

/** `format(v, ".{p}g")` exactly as Python prints it (`formatG(2 ** -10, 6)` = "0.000976562"). */
export function formatG(v: number, precision = 6): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const p = precision === 0 ? 1 : precision;
  const { digits, exp } = decimalDigits(v, p);
  const sign = v < 0 ? '-' : '';
  if (exp >= -4 && exp < p) {
    const int = exp >= 0 ? digits.slice(0, exp + 1) : '0';
    const frac = (exp >= 0 ? digits.slice(exp + 1) : '0'.repeat(-exp - 1) + digits).replace(
      /0+$/,
      '',
    );
    return `${sign}${int}${frac ? `.${frac}` : ''}`;
  }
  const rest = digits.slice(1).replace(/0+$/, '');
  const e = Math.abs(exp);
  return `${sign}${digits[0]}${rest ? `.${rest}` : ''}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
}

/** `repr(v)` / `str(v)` of a Python float (`1.0`, `1e-05`, `0.25`, `inf`, `nan`). */
export function formatRepr(v: number): string {
  if (Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const [mant, expStr] = v.toExponential().split('e');
  const exp = Number(expStr);
  if (exp < -4 || exp >= 16) {
    const e = Math.abs(exp);
    return `${mant}e${exp < 0 ? '-' : '+'}${e < 10 ? '0' : ''}${e}`;
  }
  const s = Object.is(v, -0) ? '-0' : String(v);
  return s.includes('.') ? s : `${s}.0`;
}

const g6 = (v: number) => formatG(v, 6);
const r = (v: number) => formatRepr(v);
const g3 = (v: number) => formatG(v, 3);

// ---------------------------------------------------------------------------------------
// The line function φ and the trial record
// ---------------------------------------------------------------------------------------

function toVector(v: unknown): Vector {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return (v as unknown[]).flat(Infinity).map(Number);
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>);
  return [Number(v)];
}

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** φ(α) = f(x + αp) and φ'(α) = ∇f(x + αp)ᵀp with exact call counts. */
class Line {
  nFev = 0;
  nGev = 0;
  readonly f: (x: Vector) => number;
  readonly grad: (x: Vector) => unknown;
  readonly x: Vector;
  readonly p: Vector;

  constructor(f: (x: Vector) => number, grad: (x: Vector) => unknown, x: Vector, p: Vector) {
    this.f = f;
    this.grad = grad;
    this.x = x;
    this.p = p;
  }

  point(alpha: number): Vector {
    const out = new Array<number>(this.x.length);
    for (let i = 0; i < out.length; i++) out[i] = this.x[i] + alpha * this.p[i];
    return out;
  }

  /** False when fl(x + αp) = x: the step is lost to rounding (always for α = 0). */
  moves(alpha: number): boolean {
    const pt = this.point(alpha);
    for (let i = 0; i < pt.length; i++) if (pt[i] !== this.x[i]) return true;
    return false;
  }

  phi(alpha: number): number {
    this.nFev++;
    return Number(this.f(this.point(alpha)));
  }

  gradient(alpha: number): Vector {
    this.nGev++;
    return toVector(this.grad(this.point(alpha)));
  }
}

type Phase = 'backtrack' | 'expand' | 'zoom' | 'bisect' | 'exact';

/** One trial step α with what the search learned there. */
interface Trial {
  alpha: number;
  phi: number;
  phase: Phase;
  interval: [number, number] | null;
  dphi: number | null;
  g: Vector | null;
  alphaLo: number | null;
  alphaHi: number | null;
  interp: string | null;
}

function trial(
  alpha: number,
  phi: number,
  phase: Phase,
  interval: [number, number] | null,
  zoom: { alphaLo: number; alphaHi: number; interp: string } | null = null,
): Trial {
  return {
    alpha,
    phi,
    phase,
    interval,
    dphi: null,
    g: null,
    alphaLo: zoom?.alphaLo ?? null,
    alphaHi: zoom?.alphaHi ?? null,
    interp: zoom?.interp ?? null,
  };
}

interface Outcome {
  success: boolean;
  message: string;
  trials: Trial[];
}

// The tests compare the difference φ(α) − φ(0) with the slope term (offset invariance), exactly
// like the Python module.

function armijo(alpha: number, phi: number, phi0: number, dphi0: number, c: number): boolean {
  return Number.isFinite(phi) && phi - phi0 <= c * alpha * dphi0;
}

function goldsteinLower(
  alpha: number,
  phi: number,
  phi0: number,
  dphi0: number,
  c: number,
): boolean {
  return Number.isFinite(phi) && phi - phi0 >= (1.0 - c) * alpha * dphi0;
}

function decrease(phi: number, phi0: number): boolean {
  return Number.isFinite(phi) && phi <= phi0;
}

function curvature(dphi: number, dphi0: number, c2: number): boolean {
  return Number.isFinite(dphi) && dphi >= c2 * dphi0;
}

function strongCurvature(dphi: number, dphi0: number, c2: number): boolean {
  return Number.isFinite(dphi) && Math.abs(dphi) <= -c2 * dphi0;
}

/** The acceptance tests of `kind` at one trial (Step.info.conditions). */
function conditions(
  kind: LineSearchKind,
  alpha: number,
  phi: number,
  dphi: number | null,
  phi0: number,
  dphi0: number,
  c1: number,
  c2: number,
): Record<string, boolean | null> {
  const out: Record<string, boolean | null> = { armijo: armijo(alpha, phi, phi0, dphi0, c1) };
  if (kind === 'strong_wolfe' || kind === 'weak_wolfe') {
    out.curvature = dphi === null ? null : curvature(dphi, dphi0, c2);
    if (kind === 'strong_wolfe')
      out.strong_curvature = dphi === null ? null : strongCurvature(dphi, dphi0, c2);
  } else if (kind === 'goldstein') {
    out.goldstein_lower = goldsteinLower(alpha, phi, phi0, dphi0, c1);
  } else if (kind === 'exact_quadratic') {
    out.decrease = decrease(phi, phi0);
    out.strong_curvature = dphi === null ? null : strongCurvature(dphi, dphi0, EXACT_C2);
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Interpolation for the zoom (N&W §3.5)
// ---------------------------------------------------------------------------------------

/** Minimizer of the cubic through φ, φ' at a and b (N&W eq. 3.59, stable form), or null. */
export function cubicMinimizer(
  a: number,
  fa: number,
  da: number,
  b: number,
  fb: number,
  db: number,
): number | null {
  const h = b - a;
  if (h === 0.0) return null;
  const d1 = da + db + (3.0 * (fa - fb)) / h;
  const disc = d1 * d1 - da * db;
  if (!Number.isFinite(disc) || disc <= 0.0) return null;
  // math.copysign(√disc, h); h ≠ 0 here, so the sign of h decides.
  const d2 = h < 0 || Object.is(h, -0) ? -Math.sqrt(disc) : Math.sqrt(disc);
  const m = d1 + da;
  let num: number, den: number;
  if (m >= 0.0 === d2 >= 0.0) {
    num = m + d2;
    den = da + db + 2.0 * d1;
  } else {
    num = da;
    den = m - d2;
  }
  if (den === 0.0 || !Number.isFinite(den)) return null;
  const t = a + h * (num / den);
  return Number.isFinite(t) ? t : null;
}

/** Minimizer of the quadratic through φ(a), φ'(a) and φ(b) (N&W eq. 3.58 generalized), or null. */
export function quadraticMinimizer(
  a: number,
  fa: number,
  da: number,
  b: number,
  fb: number,
): number | null {
  const h = b - a;
  if (h === 0.0) return null;
  const hh = h * h;
  // Python raises ZeroDivisionError when h² underflows to 0.
  if (hh === 0.0) throw new LineSearchZeroDivision();
  const curv = (fb - fa - da * h) / hh;
  if (!Number.isFinite(curv) || curv <= 0.0) return null;
  const t = a - da / (2.0 * curv);
  return Number.isFinite(t) ? t : null;
}

// ---------------------------------------------------------------------------------------
// The searches (each returns every trial; the accepted trial, if any, is the last one)
// ---------------------------------------------------------------------------------------

/** Backtracking, N&W Alg. 3.1: α ← ρα until the Armijo condition holds. */
function backtracking(
  line: Line,
  phi0: number,
  dphi0: number,
  alpha0: number,
  c1: number,
  rho: number,
  maxIter: number,
): Outcome {
  const out: Outcome = { success: false, message: '', trials: [] };
  let alpha = alpha0;
  for (let i = 0; i < maxIter; i++) {
    if (!line.moves(alpha)) {
      out.message =
        `the step underflowed before the Armijo condition held: x + αp = x ` +
        `in floating point at α = ${g6(alpha)}`;
      return out;
    }
    const val = line.phi(alpha);
    out.trials.push(trial(alpha, val, 'backtrack', null));
    if (armijo(alpha, val, phi0, dphi0, c1)) {
      out.success = true;
      out.message = `Armijo condition holds at α = ${g6(alpha)}`;
      return out;
    }
    alpha *= rho;
  }
  out.message =
    `no step satisfied the Armijo condition within max_iter=${maxIter} trials ` +
    `(last α = ${g3(out.trials[out.trials.length - 1].alpha)})`;
  return out;
}

/** Strong Wolfe, N&W Alg. 3.5 (bracketing) + Alg. 3.6 (zoom). */
function strongWolfe(
  line: Line,
  phi0: number,
  dphi0: number,
  alpha0: number,
  c1: number,
  c2: number,
  maxIter: number,
  alphaMax: number,
): Outcome {
  const out: Outcome = { success: false, message: '', trials: [] };
  let aPrev = 0.0,
    phiPrev = phi0,
    dphiPrev = dphi0;
  let alpha = alpha0;
  let i = 1;
  while (out.trials.length < maxIter) {
    if (!line.moves(alpha)) {
      // NOTE (Python): a step that does not move x is too short; α_{i−1} = α carries φ(0), φ'(0).
      if (alpha >= alphaMax) {
        out.message = `x + alpha p == x in floating point even at alpha_max=${g3(alphaMax)}`;
        return out;
      }
      aPrev = alpha;
      alpha = Math.min(2.0 * alpha, alphaMax);
      continue;
    }
    const val = line.phi(alpha);
    const t = trial(alpha, val, 'expand', [aPrev, alphaMax]);
    out.trials.push(t);
    if (!armijo(alpha, val, phi0, dphi0, c1) || (i > 1 && val >= phiPrev)) {
      return zoom(
        line,
        out,
        phi0,
        dphi0,
        c1,
        c2,
        maxIter,
        [aPrev, phiPrev, dphiPrev],
        alpha,
        val,
        null,
      );
    }
    const g = line.gradient(alpha);
    const d = dot(g, line.p);
    t.dphi = d;
    t.g = g;
    if (!Number.isFinite(d)) {
      out.message = `non-finite gradient at α = ${g6(alpha)}`;
      return out;
    }
    if (strongCurvature(d, dphi0, c2)) {
      out.success = true;
      out.message = `strong Wolfe conditions hold at α = ${g6(alpha)}`;
      return out;
    }
    if (d >= 0.0) {
      return zoom(
        line,
        out,
        phi0,
        dphi0,
        c1,
        c2,
        maxIter,
        [alpha, val, d],
        aPrev,
        phiPrev,
        dphiPrev,
      );
    }
    if (alpha >= alphaMax) {
      out.message =
        `phi still decreases at alpha_max=${g3(alphaMax)}; f may be unbounded below ` + 'along p';
      return out;
    }
    aPrev = alpha;
    phiPrev = val;
    dphiPrev = d;
    alpha = Math.min(2.0 * alpha, alphaMax);
    i += 1;
  }
  out.message = `reached max_iter=${maxIter} trials in the bracketing phase`;
  return out;
}

/** Zoom, N&W Alg. 3.6, with cubic/quadratic interpolation, clamping and the stall rule. */
function zoom(
  line: Line,
  out: Outcome,
  phi0: number,
  dphi0: number,
  c1: number,
  c2: number,
  maxIter: number,
  lo: [number, number, number],
  aHiIn: number,
  phiHiIn: number,
  dphiHiIn: number | null,
): Outcome {
  let [aLo, phiLo, dphiLo] = lo;
  let aHi = aHiIn,
    phiHi = phiHiIn,
    dphiHi = dphiHiIn;
  const widths: number[] = []; // bracket width before each zoom trial
  while (out.trials.length < maxIter) {
    const [a, b] = aLo < aHi ? [aLo, aHi] : [aHi, aLo];
    const w = b - a;
    widths.push(w);
    let t: number | null;
    let how: string;
    if (dphiHi !== null && Number.isFinite(phiHi)) {
      t = cubicMinimizer(aLo, phiLo, dphiLo, aHi, phiHi, dphiHi);
      how = 'cubic';
    } else if (Number.isFinite(phiHi)) {
      t = quadraticMinimizer(aLo, phiLo, dphiLo, aHi, phiHi);
      how = 'quadratic';
    } else {
      t = null;
      how = 'bisection';
    }
    const stalled = widths.length >= 3 && w > STALL * widths[widths.length - 3];
    if (t !== null && !stalled) {
      const loSafe = a + SAFEGUARD * w,
        hiSafe = b - SAFEGUARD * w;
      if (!(loSafe <= t && t <= hiSafe)) {
        t = Math.min(Math.max(t, loSafe), hiSafe);
        how = `${how}_clamped`;
      }
    }
    if (t === null || stalled || !(a < t && t < b)) {
      t = a + 0.5 * w;
      how = 'bisection';
    }
    if (!(a < t && t < b)) {
      out.message =
        `the zoom interval [${g6(a)}, ${g6(b)}] collapsed to machine precision ` +
        'before the strong Wolfe conditions held';
      return out;
    }
    const val = line.phi(t);
    const tr = trial(t, val, 'zoom', [a, b], { alphaLo: aLo, alphaHi: aHi, interp: how });
    out.trials.push(tr);
    if (!armijo(t, val, phi0, dphi0, c1) || val >= phiLo) {
      aHi = t;
      phiHi = val;
      dphiHi = null;
      continue;
    }
    const g = line.gradient(t);
    const d = dot(g, line.p);
    tr.dphi = d;
    tr.g = g;
    if (!Number.isFinite(d)) {
      out.message = `non-finite gradient at α = ${g6(t)}`;
      return out;
    }
    if (strongCurvature(d, dphi0, c2)) {
      out.success = true;
      out.message = `strong Wolfe conditions hold at α = ${g6(t)}`;
      return out;
    }
    if (d * (aHi - aLo) >= 0.0) {
      aHi = aLo;
      phiHi = phiLo;
      dphiHi = dphiLo;
    }
    aLo = t;
    phiLo = val;
    dphiLo = d;
  }
  out.message = `reached max_iter=${maxIter} trials before the zoom found a strong Wolfe step`;
  return out;
}

/** Bisection/doubling bracket search for the weak Wolfe (Lewis & Overton 2013) or Goldstein tests. */
function bisectDouble(
  kind: 'weak_wolfe' | 'goldstein',
  line: Line,
  phi0: number,
  dphi0: number,
  alpha0: number,
  c1: number,
  c2: number,
  maxIter: number,
  alphaMax: number,
): Outcome {
  const out: Outcome = { success: false, message: '', trials: [] };
  const name = kind === 'weak_wolfe' ? 'weak Wolfe' : 'Goldstein';
  let lo = 0.0,
    hi = Infinity;
  let alpha = alpha0;
  while (out.trials.length < maxIter) {
    if (!line.moves(alpha)) {
      lo = alpha;
    } else {
      const val = line.phi(alpha);
      const t = trial(alpha, val, hi === Infinity ? 'expand' : 'bisect', [lo, hi]);
      out.trials.push(t);
      if (!armijo(alpha, val, phi0, dphi0, c1)) {
        hi = alpha;
      } else {
        let tooShort: boolean;
        if (kind === 'weak_wolfe') {
          const g = line.gradient(alpha);
          const d = dot(g, line.p);
          t.dphi = d;
          t.g = g;
          if (!Number.isFinite(d)) {
            out.message = `non-finite gradient at α = ${g6(alpha)}`;
            return out;
          }
          tooShort = !curvature(d, dphi0, c2);
        } else {
          tooShort = !goldsteinLower(alpha, val, phi0, dphi0, c1);
        }
        if (!tooShort) {
          out.success = true;
          out.message = `${name} conditions hold at α = ${g6(alpha)}`;
          return out;
        }
        lo = alpha;
      }
    }
    if (hi === Infinity) {
      if (lo >= alphaMax) {
        out.message = !line.moves(lo)
          ? `x + alpha p == x in floating point even at alpha_max=${g3(alphaMax)}`
          : `phi still decreases at alpha_max=${g3(alphaMax)}; ` +
            'f may be unbounded below along p';
        return out;
      }
      alpha = Math.min(2.0 * lo, alphaMax);
    } else {
      alpha = lo + 0.5 * (hi - lo);
      if (!(lo < alpha && alpha < hi)) {
        out.message =
          `the bracket [${g6(lo)}, ${g6(hi)}] collapsed to machine precision ` +
          `before the ${name} conditions held`;
        return out;
      }
    }
  }
  out.message = `reached max_iter=${maxIter} trials before the ${name} conditions held`;
  return out;
}

/** α = −φ'(0)/pᵀHp, the minimizer of the quadratic model along p (N&W eq. 5.6). */
function exactQuadratic(line: Line, phi0: number, dphi0: number, pHp: number, fErr = 0.0): Outcome {
  const out: Outcome = { success: false, message: '', trials: [] };
  if (!(Number.isFinite(pHp) && pHp > 0.0)) {
    out.message = `pᵀ∇²f p = ${g3(pHp)} ≤ 0: the quadratic model has no minimizer along p`;
    return out;
  }
  const alpha = -dphi0 / pHp;
  const val = line.phi(alpha);
  const t = trial(alpha, val, 'exact', null);
  out.trials.push(t);
  if (!Number.isFinite(val)) {
    out.message = `f(x + αp) is not finite at α = ${g6(alpha)}`;
    return out;
  }
  const head = `exact minimizer of the quadratic model along p: α = ${g6(alpha)}`;
  if (decrease(val, phi0)) {
    out.success = true;
    out.message = head;
    return out;
  }
  const rise = val - phi0;
  const g = line.gradient(alpha);
  const d = dot(g, line.p);
  t.dphi = d;
  t.g = g;
  if (!Number.isFinite(d)) {
    out.message = `f increased at α = ${g6(alpha)} and the gradient there is not finite`;
    return out;
  }
  const slope = `|φ′(α)| = ${g3(Math.abs(d))}`;
  const bound = `${formatG(EXACT_C2)}·|φ′(0)| = ${g3(-EXACT_C2 * dphi0)}`;
  const increased = `f increased by ${g3(rise)} (from f(x) = ${g6(phi0)}) at the model step`;
  if (!strongCurvature(d, dphi0, EXACT_C2)) {
    const side = d > 0.0 ? 'overshoots' : 'falls short of';
    out.message =
      `${increased} α = ${g6(alpha)} and ${slope} > ${bound}: the quadratic model ${side} ` +
      'the line minimizer (f is not quadratic along p, or ∇f is dominated by rounding ' +
      'error)';
    return out;
  }
  if (rise <= 2.0 * fErr) {
    out.success = true;
    out.message =
      `${head}; f rose by ${g3(rise)} ≤ 2·f_err = ${g3(2.0 * fErr)}, within the rounding ` +
      `error of f, and ${slope} ≤ ${bound} confirms the line minimizer`;
    return out;
  }
  out.message =
    `${increased} α = ${g6(alpha)} although ${slope} ≤ ${bound}: the rise exceeds the ` +
    `rounding bound 2·f_err = ${g3(2.0 * fErr)}, so either α is near a local maximum ` +
    'of φ (f is not quadratic along p) or the values of f are dominated by rounding ' +
    'error (pass f_err, a bound on the rounding error of f)';
  return out;
}

// ---------------------------------------------------------------------------------------
// Public helper
// ---------------------------------------------------------------------------------------

function defaultC1(kind: string, c1: number | null | undefined): number {
  if (c1 !== null && c1 !== undefined) return Number(c1);
  return kind === 'goldstein' ? C_GOLDSTEIN_DEFAULT : C1_DEFAULT;
}

function isKind(kind: string): kind is LineSearchKind {
  return (KINDS as readonly string[]).includes(kind);
}

function validate(
  kind: string,
  c1: number,
  c2: number,
  rho: number,
  alpha0: number,
  alphaMax: number,
  maxIter: unknown,
): asserts kind is LineSearchKind {
  const fail = (msg: string): never => {
    throw new LineSearchInputError(msg);
  };
  if (!isKind(kind))
    fail(
      `unknown line search kind '${kind}'; expected one of (${KINDS.map((k) => `'${k}'`).join(', ')})`,
    );
  if (kind === 'goldstein') {
    if (!(0.0 < c1 && c1 < 0.5)) fail(`goldstein needs 0 < c < 1/2 (passed as c1), got ${r(c1)}`);
  } else if (!(0.0 < c1 && c1 < 1.0)) fail(`need 0 < c1 < 1, got c1=${r(c1)}`);
  if ((kind === 'strong_wolfe' || kind === 'weak_wolfe') && !(c1 < c2 && c2 < 1.0))
    fail(`${kind} needs 0 < c1 < c2 < 1, got c1=${r(c1)}, c2=${r(c2)}`);
  if (kind === 'backtracking' && !(0.0 < rho && rho < 1.0))
    fail(`need 0 < rho < 1, got rho=${r(rho)}`);
  if (kind !== 'exact_quadratic' && !(Number.isFinite(alpha0) && alpha0 > 0.0))
    fail(`need a finite alpha0 > 0, got ${r(alpha0)}`);
  if (kind === 'strong_wolfe' || kind === 'weak_wolfe' || kind === 'goldstein') {
    if (!Number.isFinite(alphaMax)) fail(`need a finite alpha_max, got ${r(alphaMax)}`);
    if (!(alpha0 <= alphaMax))
      fail(`need alpha0 <= alpha_max, got alpha0=${r(alpha0)}, alpha_max=${r(alphaMax)}`);
  }
  if (!(typeof maxIter === 'number' && Number.isInteger(maxIter) && maxIter >= 1))
    fail(`max_iter must be a positive integer, got ${String(maxIter)}`);
}

function run(
  kind: LineSearchKind,
  line: Line,
  phi0: number,
  dphi0: number,
  o: {
    alpha0: number;
    pHp: number;
    c1: number;
    c2: number;
    rho: number;
    maxIter: number;
    alphaMax: number;
    fErr?: number;
  },
): Outcome {
  switch (kind) {
    case 'backtracking':
      return backtracking(line, phi0, dphi0, o.alpha0, o.c1, o.rho, o.maxIter);
    case 'strong_wolfe':
      return strongWolfe(line, phi0, dphi0, o.alpha0, o.c1, o.c2, o.maxIter, o.alphaMax);
    case 'weak_wolfe':
    case 'goldstein':
      return bisectDouble(kind, line, phi0, dphi0, o.alpha0, o.c1, o.c2, o.maxIter, o.alphaMax);
    default:
      return exactQuadratic(line, phi0, dphi0, o.pHp, o.fErr ?? 0.0);
  }
}

function asMatrix(H: Matrix | number, n: number): Matrix {
  if (typeof H === 'number') return [[H]];
  // The Problem convention for 1-D problems: hess may return [h] or [[h]].
  if (n === 1 && Array.isArray(H) && H.length === 1 && typeof H[0] === 'number')
    return [[H[0] as unknown as number]];
  return H;
}

/**
 * Find a step length α along the descent direction `p` from `x` (Python `search`).
 *
 * `kind` ∈ backtracking | strong_wolfe | weak_wolfe | goldstein | exact_quadratic. Every call of
 * `f` / `grad` made here is counted in `nFev` / `nGev` (if they are already counted wrappers, add
 * only one of the two to your totals). On failure `alpha = 0` and `fNew` / `gNew` refer to x.
 *
 * @throws LineSearchInputError on invalid parameters, a non-finite f(x) or ∇f(x)ᵀp, a non-descent
 *   direction, or (exact_quadratic) a missing `hess`, pᵀ∇²f p ≤ 0 or an invalid `fErr`.
 * @throws LineSearchZeroDivision where Python raises ZeroDivisionError (h² underflow in the zoom).
 */
export function search(
  kind: LineSearchKind | string,
  f: (x: Vector) => number,
  grad: (x: Vector) => unknown,
  x: readonly number[],
  p: readonly number[],
  opts: SearchOptions = {},
): LineSearchResult {
  const alpha0 = Number(opts.alpha0 ?? 1.0);
  const c2 = Number(opts.c2 ?? C2_DEFAULT);
  const rho = Number(opts.rho ?? 0.5);
  const maxIter = opts.maxIter ?? 50;
  const alphaMax = Number(opts.alphaMax ?? 1e3);
  const fErr = opts.fErr ?? 0.0;
  const c1 = defaultC1(kind, opts.c1);
  validate(kind, c1, c2, rho, alpha0, alphaMax, maxIter);
  if (
    kind === 'exact_quadratic' &&
    !(typeof fErr === 'number' && Number.isFinite(fErr) && fErr >= 0.0)
  )
    throw new LineSearchInputError(
      `f_err must be a finite number >= 0, got ${typeof fErr === 'number' ? r(fErr) : String(fErr)}`,
    );
  const xv = Array.from(x, Number);
  const pv = Array.from(p, Number);
  if (xv.length !== pv.length)
    throw new LineSearchInputError(
      `x and p must have the same shape, got (${xv.length},) and (${pv.length},)`,
    );
  const line = new Line(f, grad, xv, pv);
  const phi0 = opts.f0 === null || opts.f0 === undefined ? line.phi(0.0) : Number(opts.f0);
  const gv =
    opts.g0 === null || opts.g0 === undefined ? line.gradient(0.0) : Array.from(opts.g0, Number);
  if (gv.length !== xv.length)
    throw new LineSearchInputError(`g0 must have shape (${xv.length},), got (${gv.length},)`);
  const dphi0 = dot(gv, pv);
  if (!(Number.isFinite(phi0) && Number.isFinite(dphi0)))
    throw new LineSearchInputError(`f(x)=${r(phi0)} and ∇f(x)ᵀp=${r(dphi0)} must be finite`);
  if (dphi0 >= 0.0)
    throw new LineSearchInputError(`p is not a descent direction: ∇f(x)ᵀp = ${g6(dphi0)} ≥ 0`);

  let nHev = 0;
  let pHp = NaN;
  if (kind === 'exact_quadratic') {
    const hess = opts.hess;
    if (hess === null || hess === undefined)
      throw new LineSearchInputError('exact_quadratic needs the Hessian (pass hess=...)');
    let H: Matrix;
    if (typeof hess === 'function') {
      nHev = 1;
      H = asMatrix(hess(xv), xv.length);
    } else {
      H = asMatrix(hess, xv.length);
    }
    if (H.length !== xv.length || H.some((row) => row.length !== xv.length))
      throw new LineSearchInputError(
        `hess must have shape (${xv.length}, ${xv.length}), got (${H.length}, ${H[0]?.length ?? 0})`,
      );
    pHp = dot(
      pv,
      H.map((row) => dot(row, pv)),
    );
    if (!(Number.isFinite(pHp) && pHp > 0.0))
      throw new LineSearchInputError(`exact_quadratic needs pᵀ∇²f p > 0, got ${g6(pHp)}`);
  }

  const out = run(kind, line, phi0, dphi0, {
    alpha0,
    pHp,
    c1,
    c2,
    rho,
    maxIter,
    alphaMax,
    fErr,
  });
  const trials = out.trials.map((t) => [t.alpha, t.phi] as [number, number]);
  if (out.success) {
    const last = out.trials[out.trials.length - 1];
    return {
      alpha: last.alpha,
      fNew: last.phi,
      gNew: last.g,
      nFev: line.nFev,
      nGev: line.nGev,
      nHev,
      success: true,
      trials,
      message: out.message,
    };
  }
  return {
    alpha: 0.0,
    fNew: phi0,
    gNew: gv,
    nFev: line.nFev,
    nGev: line.nGev,
    nHev,
    success: false,
    trials,
    message: out.message,
  };
}

// ---------------------------------------------------------------------------------------
// Finite-difference fallbacks (port of numopt.core.diff.gradient / hessian)
// ---------------------------------------------------------------------------------------

/** ε^{1/3} for float64, the central-difference step factor of `numopt.core.diff`. */
const H_CENTRAL = 6.055454452393343e-6;

function fdGradient(f: (x: Vector) => number, x: Vector): Vector {
  const g = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.slice(),
      xm = x.slice();
    xp[i] = x[i] + h;
    xm[i] = x[i] - h;
    g[i] = (f(xp) - f(xm)) / (2.0 * h);
  }
  return g;
}

function fdHessian(grad: (x: Vector) => Vector, x: Vector): Matrix {
  const n = x.length;
  const H: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.slice(),
      xm = x.slice();
    xp[i] = x[i] + h;
    xm[i] = x[i] - h;
    const gp = grad(xp),
      gm = grad(xm);
    for (let r = 0; r < n; r++) H[r][i] = (gp[r] - gm[r]) / (2.0 * h);
  }
  return H.map((row, r) => row.map((v, c) => 0.5 * (v + H[c][r])));
}

// ---------------------------------------------------------------------------------------
// Registered demo methods: one line search from x0, one Step per trial
// ---------------------------------------------------------------------------------------

const P_DIRECTION = param.choice('direction', 'steepest', ['steepest', 'newton'], {
  help: 'Search direction p: steepest descent −∇f(x₀) or Newton −∇²f(x₀)⁻¹∇f(x₀).',
  label: 'Direction',
});
const P_ALPHA0 = param.float('alpha0', 1.0, {
  min: 1e-4,
  max: 100.0,
  log: true,
  help: 'First trial step length α₀.',
  label: 'First trial step',
  tex: '\\alpha_0',
});
const P_C1 = param.float('c1', C1_DEFAULT, {
  min: 1e-6,
  max: 0.5,
  log: true,
  help: "Armijo constant c₁: accept only φ(α) ≤ φ(0) + c₁αφ'(0).",
  label: 'Armijo constant',
  tex: 'c_1',
});
// Disjoint slider ranges where the searches need an order (c₁ < c₂, α₀ ≤ α_max).
const P_C1_WOLFE = param.float('c1', C1_DEFAULT, {
  min: 1e-6,
  max: 0.09,
  log: true,
  help: "Armijo constant c₁ < c₂: accept only φ(α) ≤ φ(0) + c₁αφ'(0).",
  label: 'Armijo constant',
  tex: 'c_1',
});
const P_C2 = param.float('c2', C2_DEFAULT, {
  min: 0.1,
  max: 0.999,
  help: 'Curvature constant c₂ ∈ (c₁, 1) (0.9 for Newton/quasi-Newton, 0.1 for CG).',
  label: 'Curvature constant',
  tex: 'c_2',
});
const P_RHO = param.float('rho', 0.5, {
  min: 0.05,
  max: 0.95,
  help: 'Contraction factor: α ← ρα.',
  label: 'Contraction factor',
  tex: '\\rho',
});
const P_MAX_ITER = param.int('max_iter', 50, {
  min: 1,
  max: 200,
  help: 'Maximum number of trial steps.',
  label: 'Max trials',
});
const P_ALPHA_MAX = param.float('alpha_max', 1e3, {
  min: 100.0,
  max: 1e6,
  log: true,
  help: 'Largest step length the search may try (≥ α₀).',
  label: 'Largest step',
  tex: '\\alpha_{\\max}',
});
const P_C_GOLDSTEIN = param.float('c1', C_GOLDSTEIN_DEFAULT, {
  min: 0.01,
  max: 0.49,
  help: "Goldstein constant c ∈ (0, ½): φ(0)+(1-c)αφ'(0) ≤ φ(α) ≤ φ(0)+cαφ'(0).",
  label: 'Goldstein constant',
  tex: 'c',
});

/** A problem the demos accept: a library Problem (1-D or n-D) or a bare f(x) on vectors. */
export type LineSearchProblem = Problem<Point> | Problem<number> | Problem<Vector>;
type AnyFn = (x: unknown) => unknown;

function allFinite(...values: (number | readonly number[] | Matrix)[]): boolean {
  return values.every((v) =>
    typeof v === 'number'
      ? Number.isFinite(v)
      : (v as unknown[]).every((e) =>
          typeof e === 'number' ? Number.isFinite(e) : allFinite(e as number[]),
        ),
  );
}

interface DemoOptions {
  alpha0?: number;
  c1?: number;
  c2?: number;
  rho?: number;
  maxIter?: number;
  alphaMax?: number;
}

function demo(
  kind: LineSearchKind,
  problemIn: LineSearchProblem | ((x: Vector) => number),
  x0: Point | null | undefined,
  direction: string,
  o: DemoOptions = {},
): Result {
  const alpha0 = o.alpha0 ?? 1.0;
  const c1 = o.c1 ?? C1_DEFAULT;
  const c2 = o.c2 ?? C2_DEFAULT;
  const rho = o.rho ?? 0.5;
  const maxIterIn = o.maxIter ?? 50;
  const alphaMax = o.alphaMax ?? 1e3;
  validate(kind, c1, c2, rho, alpha0, alphaMax, maxIterIn);
  const maxIter = maxIterIn as number;
  if (direction !== 'steepest' && direction !== 'newton')
    throw new LineSearchInputError(
      `direction must be 'steepest' or 'newton', got '${String(direction)}'`,
    );

  // vector_problem + start_point
  const prob: Problem<unknown> =
    typeof problemIn === 'function'
      ? {
          id: 'custom',
          name: 'custom',
          latex: 'f(x)',
          f: problemIn as AnyFn,
          dim: x0 === null || x0 === undefined ? 2 : toVector(x0).length,
          domain: [],
        }
      : (problemIn as Problem<unknown>);
  const start = x0 ?? (prob.x0 as Point | null | undefined);
  if (start === null || start === undefined)
    throw new LineSearchInputError(
      `${prob.id}: no starting point given and the problem has no default x0`,
    );
  const x = toVector(start);
  if (prob.dim && x.length !== prob.dim)
    throw new LineSearchInputError(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  const needsHess = direction === 'newton' || kind === 'exact_quadratic';
  const scalar = prob.dim === 1;
  const gradFn = prob.grad as AnyFn | undefined;
  const hessFn = prob.hess as AnyFn | undefined;
  const arg = (v: Vector): unknown => (scalar ? v[0] : v);
  const asX = (v: Vector): Point => (scalar ? v[0] : v);

  // Counted f, ∇f, ∇²f (missing derivatives are central differences of the counted f / ∇f).
  let nF = 0,
    nG = 0,
    nH = 0;
  const fc = (v: Vector): number => {
    nF++;
    return Number((prob.f as AnyFn)(arg(v)));
  };
  const gc = (v: Vector): Vector => {
    nG++;
    return gradFn === undefined ? fdGradient(fc, v) : toVector(gradFn(arg(v)));
  };
  const hc = (v: Vector): Matrix => {
    nH++;
    if (hessFn === undefined) return fdHessian(gc, v);
    const H = hessFn(arg(v));
    const flat = toVector(H);
    const n = v.length;
    return Array.from({ length: n }, (_, r) => flat.slice(r * n, (r + 1) * n));
  };

  const phi0 = fc(x);
  const g0 = gc(x);
  const H = needsHess ? hc(x) : null;

  // The search direction p, φ'(0) and (exact_quadratic) the model curvature pᵀHp.
  let p: Vector = x.map(() => 0);
  let dphi0 = NaN;
  let pHp = NaN;
  let failure: string | null = null;
  if (!allFinite(phi0, g0)) {
    failure = 'f(x0) or ∇f(x0) is not finite';
  } else if (direction === 'steepest') {
    p = g0.map((v) => -v);
  } else {
    const lu = luFactor(H as Matrix);
    if (lu.singular) failure = 'the Hessian at x0 is singular: no Newton direction';
    else p = luSolve(lu, g0).map((v) => -v);
  }
  if (failure === null) {
    dphi0 = dot(g0, p);
    if (!allFinite(p, dphi0)) {
      failure = 'the search direction is not finite';
    } else if (!g0.some((v) => v !== 0)) {
      failure = '∇f(x0) = 0: x0 is a stationary point, so there is no descent direction';
    } else if (dphi0 >= 0.0) {
      failure = `p is not a descent direction (∇f(x0)ᵀp = ${g3(dphi0)} ≥ 0)`;
    } else if (kind === 'exact_quadratic') {
      pHp = dot(
        p,
        (H as Matrix).map((row) => dot(row, p)),
      );
      if (!(Number.isFinite(pHp) && pHp > 0.0))
        failure = `pᵀ∇²f(x0)p = ${g3(pHp)} ≤ 0: the quadratic model has no minimizer along p`;
    }
  }

  const c2Info = kind === 'strong_wolfe' || kind === 'weak_wolfe' ? c2 : null;
  const directionList = p.slice();

  const info = (t: Trial | null, accepted: boolean): StepInfo => {
    const alpha = t === null ? 0.0 : t.alpha;
    const phi = t === null ? phi0 : t.phi;
    const dphi = t === null ? dphi0 : t.dphi;
    const d: StepInfo = {
      alpha,
      phi,
      dphi,
      phi0,
      dphi0,
      c1,
      c2: c2Info,
      direction: directionList,
      phase: t === null ? 'start' : t.phase,
      interval: t === null || t.interval === null ? null : [t.interval[0], t.interval[1]],
      accepted,
      conditions: conditions(kind, alpha, phi, dphi, phi0, dphi0, c1, c2),
    };
    if (kind === 'backtracking') {
      d.rho = rho;
    } else if (kind === 'strong_wolfe') {
      d.alpha_lo = t === null ? null : t.alphaLo;
      d.alpha_hi = t === null ? null : t.alphaHi;
      d.interp = t === null ? null : t.interp;
    } else if (kind === 'exact_quadratic') {
      d.pHp = pHp;
      d.model_phi = Number.isFinite(pHp) ? phi0 + alpha * dphi0 + 0.5 * alpha * alpha * pHp : null;
    }
    return d;
  };

  const extra = (alpha: number) => ({ alpha, direction: directionList, phi0, dphi0 });
  const trace: Step[] = [
    { k: 0, x: asX(x), fun: phi0, gradNorm: norm(g0), stepSize: 0.0, info: info(null, false) },
  ];
  if (failure !== null) {
    return {
      method: kind,
      x: asX(x),
      fun: phi0,
      converged: false,
      message: failure,
      nIter: 0,
      nFev: nF,
      nGev: nG,
      nHev: nH,
      trace,
      extra: extra(0.0),
    };
  }

  const line = new Line(fc, gc, x, p);
  const out = run(kind, line, phi0, dphi0, { alpha0, pHp, c1, c2, rho, maxIter, alphaMax });
  const n = out.trials.length;
  out.trials.forEach((t, i) => {
    const k = i + 1;
    trace.push({
      k,
      x: asX(line.point(t.alpha)),
      fun: t.phi,
      gradNorm: t.g === null ? null : norm(t.g),
      stepSize: t.alpha,
      info: info(t, out.success && k === n),
    });
  });
  // A failed search can have no trial (backtracking whose α₀ does not move x).
  const last = out.trials[n - 1];
  const alphaStar = out.success ? last.alpha : 0.0;
  const fStar = out.success ? last.phi : phi0;
  return {
    method: kind,
    x: out.success ? asX(line.point(alphaStar)) : asX(x),
    fun: fStar,
    converged: out.success,
    message: out.message,
    nIter: n,
    nFev: nF,
    nGev: nG,
    nHev: nH,
    trace,
    extra: extra(alphaStar),
  };
}

type DemoFn = MethodFn<LineSearchProblem>;
const opt = (o: RunOptions & Params, k: string): number | undefined =>
  o[k] === undefined ? undefined : Number(o[k]);
const dir = (o: RunOptions & Params): string => String(o.direction ?? 'steepest');

const backtrackingDemo: DemoFn = (problem, o) =>
  demo('backtracking', problem, o.x0, dir(o), {
    alpha0: opt(o, 'alpha0'),
    c1: opt(o, 'c1'),
    rho: opt(o, 'rho'),
    maxIter: opt(o, 'max_iter'),
  });

const strongWolfeDemo: DemoFn = (problem, o) =>
  demo('strong_wolfe', problem, o.x0, dir(o), {
    alpha0: opt(o, 'alpha0'),
    c1: opt(o, 'c1'),
    c2: opt(o, 'c2'),
    maxIter: opt(o, 'max_iter'),
    alphaMax: opt(o, 'alpha_max'),
  });

const weakWolfeDemo: DemoFn = (problem, o) =>
  demo('weak_wolfe', problem, o.x0, dir(o), {
    alpha0: opt(o, 'alpha0'),
    c1: opt(o, 'c1'),
    c2: opt(o, 'c2'),
    maxIter: opt(o, 'max_iter'),
    alphaMax: opt(o, 'alpha_max'),
  });

const goldsteinDemo: DemoFn = (problem, o) =>
  demo('goldstein', problem, o.x0, dir(o), {
    alpha0: opt(o, 'alpha0'),
    c1: opt(o, 'c1') ?? C_GOLDSTEIN_DEFAULT,
    maxIter: opt(o, 'max_iter'),
    alphaMax: opt(o, 'alpha_max'),
  });

const exactQuadraticDemo: DemoFn = (problem, o) =>
  demo('exact_quadratic', problem, o.x0, dir(o), { c1: opt(o, 'c1') });

const NW = 'Nocedal & Wright (2006), Numerical Optimization, 2nd ed.';
const PHI_QUANTITIES = [
  { tex: '\\alpha', key: 'info.alpha', label: 'trial step' },
  { tex: '\\varphi(\\alpha)', key: 'info.phi', label: 'f(x₀ + αp)' },
  { tex: "\\varphi'(\\alpha)", key: 'info.dphi', label: 'slope along p' },
];

const DOCS: Record<LineSearchKind, MethodDoc> = {
  backtracking: {
    rule: "\\alpha \\leftarrow \\rho\\,\\alpha \\quad\\text{until}\\quad \\varphi(\\alpha) \\le \\varphi(0) + c_1\\,\\alpha\\,\\varphi'(0)",
    intuition:
      'Try a long step first. If f did not drop by at least a fixed fraction of what the slope promised, shrink the step by ρ and try again.',
    order: 'one f evaluation per trial',
    pros: ['Needs only f values after the first gradient', 'Simple and robust'],
    cons: ['Never lengthens a step that is too short'],
    quantities: [...PHI_QUANTITIES.slice(0, 2), { tex: '\\rho', key: 'info.rho' }],
  },
  strong_wolfe: {
    rule: "\\varphi(\\alpha) \\le \\varphi(0) + c_1\\alpha\\varphi'(0), \\qquad |\\varphi'(\\alpha)| \\le c_2\\,|\\varphi'(0)|",
    intuition:
      'Double the step until a good step is bracketed, then zoom in on the bracket with cubic or quadratic interpolation until f has dropped enough and the slope has flattened.',
    order: 'superlinear zoom (interpolation)',
    pros: ['Guarantees yᵀs > 0 for quasi-Newton updates', 'Lengthens short steps'],
    cons: ['Needs a gradient at most trials'],
    quantities: [
      ...PHI_QUANTITIES,
      { tex: '\\alpha_{lo}', key: 'info.alpha_lo' },
      { tex: '\\alpha_{hi}', key: 'info.alpha_hi' },
    ],
  },
  weak_wolfe: {
    rule: "\\varphi(\\alpha) \\le \\varphi(0) + c_1\\alpha\\varphi'(0), \\qquad \\varphi'(\\alpha) \\ge c_2\\,\\varphi'(0)",
    intuition:
      'Halve a step that is too long, double a step that is too short, until f has dropped enough and the slope is no longer steeply negative.',
    order: 'linear (bisection)',
    pros: ['Simple bracket logic', 'Works for nonsmooth f (Lewis & Overton)'],
    cons: ['Bisection is slower than interpolation'],
    quantities: [...PHI_QUANTITIES, { tex: '[\\ell, u]', key: 'info.interval' }],
  },
  goldstein: {
    rule: "\\varphi(0) + (1-c)\\,\\alpha\\varphi'(0) \\le \\varphi(\\alpha) \\le \\varphi(0) + c\\,\\alpha\\varphi'(0)",
    intuition:
      'Keep f(x + αp) between two lines through f(x): below the shallow one (enough decrease) and above the steep one (the step is not too short).',
    order: 'linear (bisection)',
    pros: ['Needs no gradient at the trials'],
    cons: ['Can exclude every minimizer of φ'],
    quantities: [...PHI_QUANTITIES.slice(0, 2), { tex: '[\\ell, u]', key: 'info.interval' }],
  },
  exact_quadratic: {
    rule: '\\alpha = -\\frac{\\nabla f(x)^\\top p}{p^\\top \\nabla^2 f(x)\\, p}',
    intuition:
      'Fit the quadratic model of f along p and jump straight to its minimizer. On a quadratic this is the exact line minimizer.',
    order: 'one step',
    pros: ['Exact on quadratics', 'One f evaluation'],
    cons: ['Needs the Hessian', 'No guarantee on non-quadratic f'],
    quantities: [
      ...PHI_QUANTITIES.slice(0, 2),
      { tex: 'p^\\top \\nabla^2 f\\, p', key: 'info.pHp' },
      { tex: 'q(\\alpha)', key: 'info.model_phi', label: 'model value' },
    ],
  },
};

registerMethod(
  {
    id: 'backtracking',
    family: 'line_search',
    name: 'Backtracking (Armijo)',
    params: [P_DIRECTION, P_ALPHA0, P_C1, P_RHO, P_MAX_ITER],
    needs: ['f', 'grad'],
    summary: 'Start with a long step and shrink it by ρ until f decreases enough (Armijo).',
    references: [`${NW}, Algorithm 3.1 and eq. (3.4)`, 'Armijo (1966), Pacific J. Math. 16'],
  },
  backtrackingDemo,
  DOCS.backtracking,
);

registerMethod(
  {
    id: 'strong_wolfe',
    family: 'line_search',
    name: 'Strong Wolfe (bracket + zoom)',
    params: [P_DIRECTION, P_ALPHA0, P_C1_WOLFE, P_C2, P_MAX_ITER, P_ALPHA_MAX],
    needs: ['f', 'grad'],
    summary: 'Expand until a good step is bracketed, then zoom in with cubic interpolation.',
    references: [`${NW}, Algorithms 3.5 and 3.6, eqs. (3.7), (3.58), (3.59)`],
  },
  strongWolfeDemo,
  DOCS.strong_wolfe,
);

registerMethod(
  {
    id: 'weak_wolfe',
    family: 'line_search',
    name: 'Weak Wolfe (bisection)',
    params: [P_DIRECTION, P_ALPHA0, P_C1_WOLFE, P_C2, P_MAX_ITER, P_ALPHA_MAX],
    needs: ['f', 'grad'],
    summary: 'Halve a too-long step, double a too-short one, until both Wolfe conditions hold.',
    references: [
      'Lewis & Overton (2013), Nonsmooth optimization via quasi-Newton methods, ' +
        'Math. Program. 141:135–163',
      `${NW}, eq. (3.6)`,
    ],
  },
  weakWolfeDemo,
  DOCS.weak_wolfe,
);

registerMethod(
  {
    id: 'goldstein',
    family: 'line_search',
    name: 'Goldstein (bisection)',
    params: [P_DIRECTION, P_ALPHA0, P_C_GOLDSTEIN, P_MAX_ITER, P_ALPHA_MAX],
    needs: ['f', 'grad'],
    summary: 'Keep f(x+αp) between two lines through f(x): not too long, not too short.',
    references: [`${NW}, eq. (3.11)`, 'Goldstein (1965), SIAM J. Control 3'],
  },
  goldsteinDemo,
  DOCS.goldstein,
);

registerMethod(
  {
    id: 'exact_quadratic',
    family: 'line_search',
    name: 'Exact step (quadratic model)',
    params: [P_DIRECTION, P_C1],
    needs: ['f', 'grad', 'hess'],
    summary: 'Jump straight to the minimizer of the local quadratic model along p.',
    references: [`${NW}, eq. (5.6)`],
  },
  exactQuadraticDemo,
  DOCS.exact_quadratic,
);
