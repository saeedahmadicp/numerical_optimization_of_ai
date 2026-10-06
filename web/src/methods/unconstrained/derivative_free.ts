/**
 * Derivative-free local minimization of f: ℝⁿ → ℝ — TS port of
 * `numopt.unconstrained.derivative_free` (src/numopt/unconstrained/derivative_free.py).
 *
 * Methods (ids as in Python): `nelder_mead`, `powell`, `hooke_jeeves`, `compass_search`.
 * Shared conventions (see the Python module docstring):
 *
 *   - Start: `x0` (or the problem's default). If f(x0) is not finite the method returns at once
 *     with `converged: false` and a one-step trace.
 *   - Extreme barrier: a trial value NaN or +∞ is treated as +∞, so such a point is never
 *     accepted; a value −∞ stops the method with `converged: false` (f is unbounded below).
 *   - Trace: one Step for k = 0 and one per iteration, `nIter === trace.at(-1).k`.
 *   - Counts: `nFev` counts every evaluation of f exactly; `nGev = nHev = 0`.
 *
 * Step.info keys are the Python ones (snake_case), listed in the Python module docstring:
 *   nelder_mead     simplex, simplex_f, operation, centroid, worst, trials, size, f_spread, coefficients
 *   powell          directions, lines, extrapolated, f_extrapolated, replace_test, replaced,
 *                   new_directions, largest_decrease, largest_index
 *   hooke_jeeves    move, outcome, base, previous_base, pattern_point, probes, step, new_step
 *   compass_search  polls, success, direction, step, new_step
 *
 * Floating-point operations keep the Python/numpy order (`origin + a * d`, `centroid + (ρχ)(x̄ −
 * x_{n+1})`, the centroid as a sequential sum divided by n), so traces agree with the fixtures to
 * the last bit. Invalid parameters throw `DerivativeFreeInputError` (Python `ValueError`) with the
 * Python message.
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import { norm } from '../../core/linalg';
import type { MethodFn, Params, Problem, Result, RunOptions, Step, Vector } from '../../core/types';
import { formatG, formatRepr } from '../line_search/methods';

/** Machine epsilon of float64. */
const EPS = Number.EPSILON;

/** Python `ValueError`: invalid parameters or start point. */
export class DerivativeFreeInputError extends Error {
  override name = 'ValueError';
}

/** A bare objective f(x) → number (Python's `vector_problem` accepts one). */
export type VectorFn = (x: Vector) => number;

/**
 * What the methods accept: an n-D problem, a 1-D problem (f gets a number, as the TS problem
 * library defines 1-D problems; Python passes a one-element array) or a bare callable.
 */
export type DerivativeFreeProblem = Problem<Vector> | Problem<number> | VectorFn;

type Options = RunOptions & Params;

const g3 = (v: number) => formatG(v, 3);

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

/** f with exact counting and the extreme-barrier convention (NaN, +∞ → +∞). */
class Objective {
  n = 0;
  private readonly fn: (x: Vector) => unknown;

  constructor(fn: (x: Vector) => unknown) {
    this.fn = fn;
  }

  call(x: readonly number[]): number {
    this.n++;
    // Pass a copy: a user f that mutates its argument cannot corrupt the method's state.
    const v = toScalar(this.fn(x.slice()));
    return Number.isNaN(v) ? Infinity : v;
  }
}

/** Python `float(v)`: a number, or a one-element array. */
function toScalar(v: unknown): number {
  if (Array.isArray(v)) {
    const flat = (v as unknown[]).flat(Infinity);
    if (flat.length !== 1)
      throw new TypeError('only length-1 arrays can be converted to Python scalars');
    return Number(flat[0]);
  }
  return Number(v);
}

/** Python `as_vector`: a fresh flat array of numbers. */
function asVector(x: unknown): Vector {
  if (typeof x === 'number') return [x];
  if (Array.isArray(x)) return (x as unknown[]).flat(Infinity).map(Number);
  if (ArrayBuffer.isView(x)) return Array.from(x as unknown as ArrayLike<number>);
  return [Number(x)];
}

/** `vector_problem` + `start_point`: the counted objective and the start vector. */
function setup(problem: DerivativeFreeProblem, x0: unknown): [Objective, Vector] {
  let id: string, dim: number, f: (x: Vector) => unknown, def: unknown;
  if (typeof problem === 'function') {
    id = 'custom';
    dim = x0 !== undefined && x0 !== null ? asVector(x0).length : 2;
    f = problem;
    def = null;
  } else if (problem && typeof problem.f === 'function') {
    id = problem.id;
    dim = problem.dim;
    const pf = problem.f as (x: unknown) => unknown;
    f = dim === 1 ? (x: Vector) => pf(x[0]) : (x: Vector) => pf(x);
    def = problem.x0;
  } else {
    throw new TypeError('problem must be a numopt Problem or a callable f(x)');
  }
  const start = x0 ?? def;
  if (start === undefined || start === null)
    throw new DerivativeFreeInputError(
      `${id}: no starting point given and the problem has no default x0`,
    );
  const x = asVector(start);
  if (dim && x.length !== dim)
    throw new DerivativeFreeInputError(`${id}: x0 has ${x.length} entries, expected ${dim}`);
  if (x.length < 1) throw new DerivativeFreeInputError('x0 must have at least one entry');
  return [new Objective(f), x];
}

function step(
  k: number,
  x: Vector,
  fun: number,
  stepSize: number | null,
  info: Record<string, unknown>,
): Step {
  return { k, x, fun, gradNorm: null, stepSize, info };
}

function result(
  method: string,
  x: Vector,
  fun: number,
  converged: boolean,
  message: string,
  nIter: number,
  nFev: number,
  trace: Step[],
): Result {
  return { method, x, fun, converged, message, nIter, nFev, nGev: 0, nHev: 0, trace, extra: {} };
}

function badStart(method: string, x0: Vector, f0: number, nfev: number): Result {
  const msg = `f(x0) is not finite (f = ${formatRepr(f0)}); cannot start`;
  return result(method, x0, f0, false, msg, 0, nfev, [step(0, x0.slice(), f0, null, {})]);
}

/** Python `str(x.tolist())` for a float vector: `[1.0, -2.5]`. */
function listRepr(x: readonly number[]): string {
  return `[${x.map(formatRepr).join(', ')}]`;
}

function unbounded(method: string, x: Vector, trace: Step[], k: number, nfev: number): Result {
  const msg = `f = -inf at x = ${listRepr(x)}: f is unbounded below`;
  return result(method, x, -Infinity, false, msg, k, nfev, trace);
}

function maxIterResult(
  method: string,
  x: Vector,
  fx: number,
  maxIter: number,
  nfev: number,
  trace: Step[],
): Result {
  return result(method, x, fx, false, `reached max_iter=${maxIter}`, maxIter, nfev, trace);
}

function checkPositive(values: Record<string, number>): void {
  for (const [name, v] of Object.entries(values)) {
    if (!(v > 0.0)) throw new DerivativeFreeInputError(`${name} must be > 0, got ${formatRepr(v)}`);
  }
}

/** Reject a `max_iter` that the `k == max_iter` test can never meet (0, < 0, non-integer). */
function checkMaxIter(maxIter: number): void {
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new DerivativeFreeInputError(`max_iter must be a positive integer, got ${maxIter}`);
}

function checkShrink(shrink: number): void {
  if (!(0.0 < shrink && shrink < 1.0))
    throw new DerivativeFreeInputError(`shrink must lie in (0, 1), got ${formatRepr(shrink)}`);
}

/** `np.max(np.abs(v))` (NaN propagates, like numpy). */
function maxAbs(v: readonly number[]): number {
  let m = -Infinity;
  for (const x of v) {
    const a = Math.abs(x);
    if (Number.isNaN(a)) return NaN;
    if (a > m) m = a;
  }
  return m;
}

/** `np.max(np.abs(a − b))`. */
function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (Number.isNaN(d)) return NaN;
    if (d > m) m = d;
  }
  return m;
}

/** Python `math.copysign(a, b)` (the sign of b, including −0). */
function copysign(a: number, b: number): number {
  const neg = b < 0 || Object.is(b, -0);
  return neg ? -Math.abs(a) : Math.abs(a);
}

function num(v: unknown): number {
  return Number(v);
}

// ---------------------------------------------------------------------------------------
// Nelder–Mead
// ---------------------------------------------------------------------------------------

interface NmTrial {
  op: string;
  x: Vector;
  f: number;
}

/** (ρ, χ, γ, σ): LRWW (1998) standard values, or Gao & Han (2012) eq. (4.1) adaptive ones. */
function nmCoefficients(n: number, adaptive: boolean): [number, number, number, number] {
  if (!adaptive) return [1.0, 2.0, 0.5, 0.5];
  if (n < 2)
    throw new DerivativeFreeInputError(
      'adaptive Nelder–Mead needs n ≥ 2 (its shrink σ = 1 − 1/n is 0 for n = 1)',
    );
  return [1.0, 1.0 + 2.0 / n, 0.75 - 0.5 / n, 1.0 - 1.0 / n];
}

/** origin + s·(a − b), elementwise in numpy's order. */
function along(origin: readonly number[], s: number, a: readonly number[], b: readonly number[]) {
  const out = new Array<number>(origin.length);
  for (let i = 0; i < out.length; i++) out[i] = origin[i] + s * (a[i] - b[i]);
  return out;
}

export const nelderMead: MethodFn<DerivativeFreeProblem> = (problem, options: Options) => {
  const xtol = num(options.xtol ?? 1e-8);
  const ftol = num(options.ftol ?? 1e-8);
  const initialStep = num(options.initial_step ?? 0.5);
  const adaptive = Boolean(options.adaptive ?? false);
  const maxIter = num(options.max_iter ?? 1000);
  checkMaxIter(maxIter);
  checkPositive({ xtol, ftol, initial_step: initialStep });
  const [fobj, xStart] = setup(problem, options.x0);
  const n = xStart.length;
  const [rho, chi, gamma, sigma] = nmCoefficients(n, adaptive);
  const coefficients = { rho, chi, gamma, sigma };

  const fStart = fobj.call(xStart);
  if (!Number.isFinite(fStart)) return badStart('nelder_mead', xStart, fStart, fobj.n);

  let simplex: Vector[] = [xStart.slice()];
  for (let i = 0; i < n; i++) {
    const v = xStart.slice();
    v[i] += initialStep;
    simplex.push(v);
  }
  let fvals: number[] = [fStart];
  for (let i = 1; i <= n; i++) fvals.push(fobj.call(simplex[i]));

  const sort = () => {
    // Stable sort (LRWW tie rules), with Python's `<` comparison.
    const order = Array.from({ length: n + 1 }, (_, i) => i).sort((i, j) =>
      fvals[i] < fvals[j] ? -1 : fvals[j] < fvals[i] ? 1 : 0,
    );
    simplex = order.map((i) => simplex[i]);
    fvals = order.map((i) => fvals[i]);
  };

  const size = () => {
    let m = -Infinity;
    for (let i = 1; i <= n; i++) {
      const d = maxAbsDiff(simplex[i], simplex[0]);
      if (d > m || Number.isNaN(d)) m = d;
    }
    return m;
  };

  const info = (
    op: string | null,
    centroid: Vector | null,
    worst: Vector | null,
    trials: NmTrial[],
  ) => ({
    simplex: simplex.map((v) => v.slice()),
    simplex_f: fvals.slice(),
    operation: op,
    centroid,
    worst,
    trials,
    size: size(),
    f_spread: fvals[n] - fvals[0],
    coefficients,
  });

  sort();
  const trace: Step[] = [step(0, simplex[0].slice(), fvals[0], size(), info(null, null, null, []))];
  if (fvals[0] === -Infinity) return unbounded('nelder_mead', simplex[0].slice(), trace, 0, fobj.n);
  let k = 0;
  for (;;) {
    const tolX = xtol + 2.0 * EPS * maxAbs(simplex[0]);
    const tolF = ftol + 2.0 * EPS * Math.abs(fvals[0]);
    const sz = size(),
      spread = fvals[n] - fvals[0];
    if (sz <= tolX && spread <= tolF) {
      const msg = `simplex size ${g3(sz)} ≤ ${g3(tolX)} and f-spread ${g3(spread)} ≤ ${g3(tolF)}`;
      return result('nelder_mead', simplex[0].slice(), fvals[0], true, msg, k, fobj.n, trace);
    }
    if (k === maxIter)
      return maxIterResult('nelder_mead', simplex[0].slice(), fvals[0], maxIter, fobj.n, trace);
    k += 1;

    const worst = simplex[n].slice();
    // np.mean(np.stack(simplex[:n]), axis=0): a sequential sum over the rows, divided by n.
    const centroid = simplex[0].slice();
    for (let i = 1; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j];
    for (let j = 0; j < n; j++) centroid[j] /= n;
    const trials: NmTrial[] = [];
    const trial = (op: string, point: Vector) => {
      const fp = fobj.call(point);
      trials.push({ op, x: point, f: fp });
      return fp;
    };

    let op: string;
    let accepted: [Vector, number] | null;
    const xR = along(centroid, rho, centroid, worst);
    const fR = trial('reflect', xR);
    if (fvals[0] <= fR && fR < fvals[n - 1]) {
      op = 'reflect';
      accepted = [xR, fR];
    } else if (fR < fvals[0]) {
      const xE = along(centroid, rho * chi, centroid, worst);
      const fE = trial('expand', xE);
      if (fE < fR) {
        op = 'expand';
        accepted = [xE, fE];
      } else {
        op = 'reflect';
        accepted = [xR, fR];
      }
    } else if (fR < fvals[n]) {
      const xC = along(centroid, rho * gamma, centroid, worst);
      const fC = trial('contract_outside', xC);
      op = fC <= fR ? 'contract_outside' : 'shrink';
      accepted = fC <= fR ? [xC, fC] : null;
    } else {
      // x̄ − γ(x̄ − x_{n+1}), a subtraction as in Python.
      const xCC = new Array<number>(n);
      for (let j = 0; j < n; j++) xCC[j] = centroid[j] - gamma * (centroid[j] - worst[j]);
      const fCC = trial('contract_inside', xCC);
      op = fCC < fvals[n] ? 'contract_inside' : 'shrink';
      accepted = fCC < fvals[n] ? [xCC, fCC] : null;
    }

    if (accepted !== null) {
      simplex[n] = accepted[0];
      fvals[n] = accepted[1];
    } else {
      for (let i = 1; i <= n; i++) {
        simplex[i] = along(simplex[0], sigma, simplex[i], simplex[0]);
        fvals[i] = fobj.call(simplex[i]);
      }
    }
    sort();
    trace.push(step(k, simplex[0].slice(), fvals[0], size(), info(op, centroid, worst, trials)));
    if (fvals[0] === -Infinity)
      return unbounded('nelder_mead', simplex[0].slice(), trace, k, fobj.n);
  }
};

// ---------------------------------------------------------------------------------------
// One-dimensional minimization along a line: bracketing (NR §10.1) + Brent (NR §10.3)
// ---------------------------------------------------------------------------------------

/** Golden-ratio magnification of mnbrak and the limit of a parabolic extrapolation (NR §10.1). */
const GOLD = 0.5 * (1.0 + Math.sqrt(5.0));
const GLIMIT = 100.0;
const TINY = 1e-20;
/** Give up bracketing after this many expansions (f is then decreasing along the whole line). */
const BRACKET_MAX_ITER = 100;
/** Brent: golden fraction (3 − √5)/2, relative tolerance, absolute floor and iteration limit. */
const CGOLD = 0.5 * (3.0 - Math.sqrt(5.0));
const BRENT_TOL = 3.0e-8;
const BRENT_ZEPS = EPS * 1.0e-3;
const BRENT_MAX_ITER = 100;

export interface LineResult {
  alpha: number;
  f: number;
  /** Every [α, φ(α)] evaluated, in order. */
  trials: [number, number][];
  ok: boolean;
  message: string;
}

/**
 * Minimize φ(α) = f(x + α d) from α = 0 (where φ = f0): NR's `mnbrak` from the points 0 and 1,
 * then Brent's `localmin` with tolerance tol·|α| + ZEPS. Returns the best α found (φ(α) ≤ f0).
 * `ok` is false when f is −∞ somewhere or no bracket is found in 100 expansions.
 */
export function lineMinimize(phi: (a: number) => number, f0: number): LineResult {
  const trials: [number, number][] = [];
  const ev = (a: number) => {
    const v = phi(a);
    trials.push([a, v]);
    return v;
  };
  const fail = (message: string): LineResult => {
    // Report the best finite point seen (never −∞) so the caller keeps a finite iterate.
    let best: [number, number] = [0.0, f0];
    for (const t of trials) if (Number.isFinite(t[1]) && t[1] < best[1]) best = t;
    return { alpha: best[0], f: best[1], trials, ok: false, message };
  };
  const UNBOUNDED = 'f = -inf on the line: f is unbounded below';

  // ---- mnbrak ----
  let ax = 0.0,
    bx = 1.0;
  let fa = f0,
    fb = ev(bx);
  if (fb === -Infinity) return fail(UNBOUNDED);
  if (fb > fa) {
    [ax, bx, fa, fb] = [bx, ax, fb, fa];
  }
  let cx = bx + GOLD * (bx - ax);
  let fc = ev(cx);
  let nExpand = 0;
  while (fb > fc) {
    if (fc === -Infinity) return fail(UNBOUNDED);
    nExpand += 1;
    if (nExpand > BRACKET_MAX_ITER)
      return fail('could not bracket a minimum along the line (f keeps decreasing)');
    const r = (bx - ax) * (fb - fc);
    const q = (bx - cx) * (fb - fa);
    let u =
      bx -
      ((bx - cx) * q - (bx - ax) * r) / (2.0 * copysign(Math.max(Math.abs(q - r), TINY), q - r));
    const ulim = bx + GLIMIT * (cx - bx);
    let fu: number;
    if (!Number.isFinite(u)) {
      // A parabola through a +inf (barrier) value is undefined: use the golden magnification.
      u = cx + GOLD * (cx - bx);
      fu = ev(u);
    } else if ((bx - u) * (u - cx) > 0.0) {
      // parabolic u between b and c
      fu = ev(u);
      if (fu < fc) {
        // minimum between b and c
        [ax, bx, fb] = [bx, u, fu]; // (fa ← fb is not needed after the loop)
        break;
      }
      if (fu > fb) {
        // minimum between a and u
        cx = u;
        fc = fu;
        break;
      }
      u = cx + GOLD * (cx - bx);
      fu = ev(u);
    } else if ((cx - u) * (u - ulim) > 0.0) {
      // parabolic u between c and its limit
      fu = ev(u);
      if (fu < fc) {
        [bx, cx, u] = [cx, u, u + GOLD * (u - cx)];
        [fb, fc] = [fc, fu];
        fu = ev(u);
      }
    } else if ((u - ulim) * (ulim - cx) >= 0.0) {
      // limit parabolic u to its maximum value
      u = ulim;
      fu = ev(u);
    } else {
      // reject parabolic u, use default magnification
      u = cx + GOLD * (cx - bx);
      fu = ev(u);
    }
    if (fu === -Infinity) return fail(UNBOUNDED);
    [ax, bx, cx] = [bx, cx, u];
    [fa, fb, fc] = [fb, fc, fu];
  }
  if (fc === -Infinity || fb === -Infinity) return fail(UNBOUNDED);

  // ---- Brent's localmin on [min(a, c), max(a, c)] starting from the bracket's middle b ----
  let [a, b] = ax < cx ? [ax, cx] : [cx, ax];
  let x = bx,
    w = bx,
    v = bx;
  let fx = fb,
    fw = fb,
    fv = fb;
  let d = 0.0,
    e = 0.0;
  for (let it = 0; it < BRENT_MAX_ITER; it++) {
    const xm = 0.5 * (a + b);
    const tol1 = BRENT_TOL * Math.abs(x) + BRENT_ZEPS;
    const tol2 = 2.0 * tol1;
    if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) break;
    let golden = true;
    if (Math.abs(e) > tol1) {
      const r = (x - w) * (fx - fv);
      let q = (x - v) * (fx - fw);
      let p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = Math.abs(q);
      const etemp = e;
      e = d;
      // With a +inf (barrier) value among fx, fw, fv the parabola is undefined (NaN): reject it.
      const acceptable =
        Number.isFinite(p) &&
        Number.isFinite(q) &&
        !(Math.abs(p) >= Math.abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x));
      if (acceptable) {
        golden = false;
        d = p / q;
        const u = x + d;
        if (u - a < tol2 || b - u < tol2) d = copysign(tol1, xm - x);
      }
    }
    if (golden) {
      e = x >= xm ? a - x : b - x;
      d = CGOLD * e;
    }
    const u = Math.abs(d) >= tol1 ? x + d : x + copysign(tol1, d);
    const fu = ev(u);
    if (fu === -Infinity) return fail(UNBOUNDED);
    if (fu <= fx) {
      if (u >= x) a = x;
      else b = x;
      [v, w, x] = [w, x, u];
      [fv, fw, fx] = [fw, fx, fu];
    } else {
      if (u < x) a = u;
      else b = u;
      if (fu <= fw || w === x) {
        [v, w] = [w, u];
        [fv, fw] = [fw, fu];
      } else if (fu <= fv || v === x || v === w) {
        v = u;
        fv = fu;
      }
    }
  }
  // NR treats ITMAX as an error; Python returns Brent's best point (φ(x) ≤ φ(0) still holds).
  if (fx > f0)
    return { alpha: 0.0, f: f0, trials, ok: true, message: 'no decrease along the line' };
  return { alpha: x, f: fx, trials, ok: true, message: '' };
}

// ---------------------------------------------------------------------------------------
// Powell's conjugate-direction method
// ---------------------------------------------------------------------------------------

/** Absolute floor of Powell's relative stopping test (NR3 `Powell`: TINY = 1e-25). */
const POWELL_TINY = 1.0e-25;

interface PowellLine {
  origin: Vector;
  direction: Vector;
  alpha: number;
  point: Vector;
  f: number;
  trials: [number, number][];
}

/** origin + a·d, elementwise. */
function pointOn(origin: readonly number[], a: number, d: readonly number[]): Vector {
  const out = new Array<number>(origin.length);
  for (let i = 0; i < out.length; i++) out[i] = origin[i] + a * d[i];
  return out;
}

function unitVectors(n: number, sign = 1.0): Vector[] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? sign * 1.0 : sign * 0.0)),
  );
}

export const powell: MethodFn<DerivativeFreeProblem> = (problem, options: Options) => {
  const ftol = num(options.ftol ?? 1e-10);
  const xtol = num(options.xtol ?? 1e-8);
  const maxIter = num(options.max_iter ?? 200);
  checkMaxIter(maxIter);
  checkPositive({ ftol, xtol });
  const [fobj, p0] = setup(problem, options.x0);
  let p = p0;
  const n = p.length;
  let fret = fobj.call(p);
  if (!Number.isFinite(fret)) return badStart('powell', p, fret, fobj.n);
  const directions: Vector[] = unitVectors(n);
  const info0 = {
    directions: directions.map((d) => d.slice()),
    new_directions: directions.map((d) => d.slice()),
    lines: [],
    extrapolated: null,
    f_extrapolated: null,
    largest_decrease: 0.0,
    largest_index: 0,
    replace_test: null,
    replaced: null,
  };
  const trace: Step[] = [step(0, p.slice(), fret, null, info0)];

  const line = (origin: Vector, d: Vector, fOrigin: number): [LineResult, PowellLine] => {
    const res = lineMinimize((a) => fobj.call(pointOn(origin, a, d)), fOrigin);
    const point = pointOn(origin, res.alpha, d);
    const rec: PowellLine = {
      origin: origin.slice(),
      direction: d.slice(),
      alpha: res.alpha,
      point,
      f: res.f,
      trials: res.trials,
    };
    return [res, rec];
  };

  let k = 0;
  for (;;) {
    k += 1;
    const f0 = fret,
      xStart = p.slice();
    const used = directions.map((d) => d.slice());
    const lines: PowellLine[] = [];
    let delta = 0.0,
      ibig = 0;
    const info: Record<string, unknown> = {
      directions: used,
      lines,
      extrapolated: null,
      f_extrapolated: null,
      replace_test: null,
      replaced: null,
    };

    let failure: string | null = null;
    let unboundedAt: Vector | null = null;
    let stopMsg: string | null = null;
    for (let i = 0; i < n; i++) {
      const fBefore = fret;
      const [res, rec] = line(p, directions[i], fret);
      lines.push(rec);
      p = rec.point;
      fret = res.f;
      if (!res.ok) {
        failure = `line minimization along direction ${i} failed: ${res.message}`;
        break;
      }
      if (fBefore - fret > delta) {
        delta = fBefore - fret;
        ibig = i;
      }
    }

    const fTest = 2.0 * (f0 - fret) <= ftol * (Math.abs(f0) + Math.abs(fret)) + POWELL_TINY;
    const sweepStep = maxAbsDiff(p, xStart);
    const xTest = sweepStep <= xtol + 2.0 * EPS * maxAbs(p);
    const plateau = f0 - fret <= 2.0 * EPS * Math.abs(fret);
    if (failure === null && fTest && (xTest || plateau)) {
      stopMsg =
        `sweep decrease 2(f₀ − f_n) = ${g3(2.0 * (f0 - fret))} ≤ ftol·(|f₀| + |f_n|) + 1e-25` +
        (xTest
          ? ` and sweep step ${g3(sweepStep)} ≤ xtol + 2ε‖x‖∞`
          : ' and f decreased only at rounding level (≤ 2ε|f|)');
    } else if (failure === null) {
      const xE = new Array<number>(n);
      for (let j = 0; j < n; j++) xE[j] = 2.0 * p[j] - xStart[j];
      const fE = fobj.call(xE);
      info.extrapolated = xE;
      info.f_extrapolated = fE;
      if (fE === -Infinity) {
        unboundedAt = xE;
      } else if (fE < f0) {
        // Python's `** 2` (C pow, correctly rounded for squares) = the product, which JS `**`
        // does not guarantee.
        const s1 = f0 - fret - delta,
          s2 = f0 - fE;
        const t = 2.0 * (f0 - 2.0 * fret + fE) * (s1 * s1) - delta * (s2 * s2);
        info.replace_test = t;
        if (t < 0.0) {
          const u = new Array<number>(n);
          for (let j = 0; j < n; j++) u[j] = p[j] - xStart[j];
          const [res, rec] = line(p, u, fret);
          lines.push(rec);
          p = rec.point;
          fret = res.f;
          if (res.ok) {
            directions[ibig] = directions[n - 1];
            directions[n - 1] = u;
            info.replaced = ibig;
          } else {
            failure = `line minimization along the new direction failed: ${res.message}`;
          }
        }
      }
    }

    info.new_directions = directions.map((d) => d.slice());
    info.largest_decrease = delta;
    info.largest_index = ibig;
    const prev = trace[trace.length - 1].x as Vector;
    const stepLen = norm(p.map((v, j) => v - prev[j]));
    trace.push(step(k, p.slice(), fret, stepLen, info));
    if (unboundedAt !== null) return unbounded('powell', unboundedAt, trace, k, fobj.n);
    if (failure !== null)
      return result('powell', p.slice(), fret, false, failure, k, fobj.n, trace);
    if (stopMsg !== null) return result('powell', p.slice(), fret, true, stopMsg, k, fobj.n, trace);
    if (k === maxIter) return maxIterResult('powell', p.slice(), fret, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Hooke–Jeeves pattern search
// ---------------------------------------------------------------------------------------

interface Probe {
  x: Vector;
  f: number;
}

function stepParams(options: Options) {
  const step = num(options.step ?? 0.5);
  const shrink = num(options.shrink ?? 0.5);
  const xtol = num(options.xtol ?? 1e-8);
  const maxIter = num(options.max_iter ?? 1000);
  checkMaxIter(maxIter);
  checkPositive({ step, xtol });
  checkShrink(shrink);
  return { step, shrink, xtol, maxIter };
}

export const hookeJeeves: MethodFn<DerivativeFreeProblem> = (problem, options: Options) => {
  const { step: step0, shrink, xtol, maxIter } = stepParams(options);
  const [fobj, base0] = setup(problem, options.x0);
  let base = base0;
  const n = base.length;
  let fBase = fobj.call(base);
  if (!Number.isFinite(fBase)) return badStart('hooke_jeeves', base, fBase, fobj.n);
  let h = step0;
  let prevBase: Vector | null = null;
  let mode: 'explore' | 'pattern' = 'explore';
  const info0 = {
    move: null,
    outcome: null,
    base: base.slice(),
    previous_base: null,
    pattern_point: null,
    probes: [],
    step: h,
    new_step: h,
  };
  const trace: Step[] = [step(0, base.slice(), fBase, h, info0)];

  const explore = (y0: Vector, fy0: number, probes: Probe[]): [Vector, number] => {
    let y = y0.slice(),
      fy = fy0;
    for (let i = 0; i < n; i++) {
      for (const sgn of [1.0, -1.0]) {
        const trial = y.slice();
        trial[i] += sgn * h;
        const ft = fobj.call(trial);
        probes.push({ x: trial, f: ft });
        if (ft < fy) {
          y = trial;
          fy = ft;
          break;
        }
      }
    }
    return [y, fy];
  };

  let k = 0;
  for (;;) {
    k += 1;
    const probes: Probe[] = [];
    const hUsed = h;
    let patternPoint: Vector | null = null;
    const move = mode;
    let outcome: string;
    if (mode === 'pattern') {
      const pb = prevBase as Vector;
      patternPoint = base.map((v, j) => 2.0 * v - pb[j]);
      const fP = fobj.call(patternPoint);
      probes.push({ x: patternPoint.slice(), f: fP });
      const [x, fx] = explore(patternPoint, fP, probes);
      if (fx < fBase) {
        [prevBase, base, fBase] = [base, x, fx];
        outcome = 'pattern_success';
      } else {
        outcome = 'pattern_failed';
        mode = 'explore';
      }
    } else {
      const [x, fx] = explore(base, fBase, probes);
      if (fx < fBase) {
        [prevBase, base, fBase] = [base, x, fx];
        outcome = 'explore_success';
        mode = 'pattern';
      } else {
        h *= shrink;
        outcome = 'step_reduced';
      }
    }
    const info = {
      move,
      outcome,
      base: base.slice(),
      previous_base: prevBase === null ? null : prevBase.slice(),
      pattern_point: patternPoint,
      probes,
      step: hUsed,
      new_step: h,
    };
    trace.push(step(k, base.slice(), fBase, hUsed, info));
    if (fBase === -Infinity) return unbounded('hooke_jeeves', base.slice(), trace, k, fobj.n);
    if (outcome === 'step_reduced' && h <= xtol) {
      const msg = `step h = ${g3(h)} ≤ xtol after an unsuccessful exploratory move`;
      return result('hooke_jeeves', base.slice(), fBase, true, msg, k, fobj.n, trace);
    }
    if (k === maxIter)
      return maxIterResult('hooke_jeeves', base.slice(), fBase, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Compass search
// ---------------------------------------------------------------------------------------

export const compassSearch: MethodFn<DerivativeFreeProblem> = (problem, options: Options) => {
  const { step: step0, shrink, xtol, maxIter } = stepParams(options);
  const [fobj, x0] = setup(problem, options.x0);
  let x = x0;
  const n = x.length;
  let fx = fobj.call(x);
  if (!Number.isFinite(fx)) return badStart('compass_search', x, fx, fobj.n);
  let delta = step0;
  const info0 = { polls: [], success: null, direction: null, step: delta, new_step: delta };
  const trace: Step[] = [step(0, x.slice(), fx, delta, info0)];
  // D⊕ = (e₁, …, e_n, −e₁, …, −e_n)
  const compass = [...unitVectors(n), ...unitVectors(n, -1.0)];
  let k = 0;
  for (;;) {
    k += 1;
    const polls: Probe[] = [];
    let direction: number | null = null;
    for (let j = 0; j < compass.length; j++) {
      const y = pointOn(x, delta, compass[j]);
      const fy = fobj.call(y);
      polls.push({ x: y, f: fy });
      if (fy < fx) {
        x = y;
        fx = fy;
        direction = j;
        break;
      }
    }
    const deltaUsed = delta;
    if (direction === null) delta *= shrink;
    const info = {
      polls,
      success: direction !== null,
      direction,
      step: deltaUsed,
      new_step: delta,
    };
    trace.push(step(k, x.slice(), fx, deltaUsed, info));
    if (fx === -Infinity) return unbounded('compass_search', x.slice(), trace, k, fobj.n);
    if (direction === null && delta < xtol) {
      const msg = `step Δ = ${g3(delta)} < xtol after an unsuccessful poll`;
      return result('compass_search', x.slice(), fx, true, msg, k, fobj.n, trace);
    }
    if (k === maxIter)
      return maxIterResult('compass_search', x.slice(), fx, maxIter, fobj.n, trace);
  }
};

// ---------------------------------------------------------------------------------------
// Registration (ids, params, metadata exactly as the Python @register)
// ---------------------------------------------------------------------------------------

const STEP_PARAMS = [
  param.float('step', 0.5, {
    min: 1e-3,
    max: 5.0,
    log: true,
    help: 'Initial step length along each coordinate.',
    label: 'Initial step',
    tex: 'h_0',
  }),
  param.float('shrink', 0.5, {
    min: 0.05,
    max: 0.95,
    help: 'Factor that multiplies the step after an unsuccessful iteration.',
    label: 'Shrink factor',
    tex: '\\theta',
  }),
  param.float('xtol', 1e-8, {
    min: 1e-14,
    max: 1e-1,
    log: true,
    help: 'Stop when the step falls below xtol.',
    label: 'Step tolerance',
    tex: 'h \\le',
  }),
  param.int('max_iter', 1000, {
    min: 1,
    max: 100_000,
    help: 'Iteration limit.',
    label: 'Max iterations',
  }),
];

const DOCS: Record<'nelder_mead' | 'powell' | 'hooke_jeeves' | 'compass_search', MethodDoc> = {
  nelder_mead: {
    rule: '\\mathbf{x}_r = \\bar{\\mathbf{x}} + \\rho\\,(\\bar{\\mathbf{x}} - \\mathbf{x}_{n+1}), \\qquad \\bar{\\mathbf{x}} = \\tfrac{1}{n}\\textstyle\\sum_{i \\le n} \\mathbf{x}_i',
    intuition:
      'Keep n + 1 points. Reflect the worst one through the centroid of the others; stretch the move if it went well, pull it back if it did not, and shrink everything toward the best point when nothing helps.',
    order: 'no general rate (may stall on non-stationary points, McKinnon 1998)',
    pros: ['Needs only values of f', 'Adapts its shape to the valley'],
    cons: ['Can converge to a non-stationary point', 'Slow in high dimension'],
    quantities: [
      { tex: '\\text{op}', key: 'info.operation', label: 'Operation' },
      { tex: '\\max_i \\|\\mathbf{x}_i - \\mathbf{x}_1\\|_\\infty', key: 'info.size' },
      { tex: 'f_{n+1} - f_1', key: 'info.f_spread' },
    ],
  },
  powell: {
    rule: '\\mathbf{x}_i = \\mathbf{x}_{i-1} + \\alpha_i \\mathbf{u}_i, \\quad \\alpha_i = \\arg\\min_\\alpha f(\\mathbf{x}_{i-1} + \\alpha\\,\\mathbf{u}_i)',
    intuition:
      'Minimize exactly along each direction in turn. After a sweep, the net displacement becomes a new direction and replaces the one that helped most, so the set drifts toward conjugate directions.',
    order: 'no quadratic termination with the discarding rule; Powell (1964): n sweeps',
    pros: ['Needs only values of f', 'Exact line searches make big moves along valleys'],
    cons: ['Many evaluations per sweep', 'Directions can become nearly dependent'],
    quantities: [
      { tex: '\\Delta f', key: 'info.largest_decrease', label: 'Largest decrease' },
      { tex: 'i_{\\text{big}}', key: 'info.largest_index' },
      { tex: 't', key: 'info.replace_test', label: 'Replace test' },
    ],
  },
  hooke_jeeves: {
    rule: '\\mathbf{p} = \\mathbf{b} + (\\mathbf{b} - \\mathbf{b}_{\\text{old}}), \\qquad \\text{explore } \\mathbf{p} \\pm h\\,\\mathbf{e}_i',
    intuition:
      'Probe each coordinate by ±h and keep what helps. After a success, leap again in the same direction (the pattern move); after a failure, halve the step.',
    order: 'linear at best; ‖∇f‖ = O(step) at unsuccessful iterations',
    pros: ['Needs only values of f', 'Pattern moves accelerate along valleys'],
    cons: ['Coordinate-aligned probes', 'Slow final convergence'],
    quantities: [
      { tex: 'h', key: 'info.step', label: 'Step' },
      { tex: '\\text{move}', key: 'info.move' },
      { tex: '\\text{outcome}', key: 'info.outcome' },
    ],
  },
  compass_search: {
    rule: '\\mathbf{x}_{k+1} = \\mathbf{x}_k + \\Delta_k \\mathbf{d}_k, \\quad \\mathbf{d}_k \\in \\{\\pm \\mathbf{e}_i\\}, \\qquad \\Delta_{k+1} = \\theta\\,\\Delta_k \\text{ on failure}',
    intuition:
      'Try a step north, east, south and west; move to the first point that is lower. When none is, the minimizer is closer than the step, so halve it.',
    order: 'linear at best; ‖∇f‖ ≤ √n·M·Δ at unsuccessful iterations',
    pros: ['Simplest provably convergent direct search', 'Stopping test certifies stationarity'],
    cons: ['Up to 2n evaluations per iteration', 'Slow on ill-conditioned problems'],
    quantities: [
      { tex: '\\Delta_k', key: 'info.step', label: 'Step' },
      { tex: '\\text{success}', key: 'info.success' },
    ],
  },
};

registerMethod(
  {
    id: 'nelder_mead',
    family: 'unconstrained',
    name: 'Nelder–Mead simplex',
    params: [
      param.float('xtol', 1e-8, {
        min: 1e-14,
        max: 1e-1,
        log: true,
        help: 'Stop when the simplex size max‖x_i − x_1‖∞ ≤ xtol (and the f-spread test holds).',
        label: 'Size tolerance',
      }),
      param.float('ftol', 1e-8, {
        min: 1e-14,
        max: 1e-1,
        log: true,
        help: 'Stop when f(worst) − f(best) ≤ ftol (and the size test holds).',
        label: 'f-spread tolerance',
      }),
      param.float('initial_step', 0.5, {
        min: 1e-3,
        max: 5.0,
        log: true,
        help: 'Edge length h of the initial simplex x₀, x₀ + h e₁, …, x₀ + h eₙ.',
        label: 'Initial edge',
        tex: 'h',
      }),
      param.bool('adaptive', false, {
        help: 'Use the dimension-dependent coefficients of Gao & Han (2012).',
        label: 'Adaptive coefficients',
      }),
      param.int('max_iter', 1000, {
        min: 1,
        max: 100_000,
        help: 'Iteration limit.',
        label: 'Max iterations',
      }),
    ],
    needs: ['f'],
    order: 'no general rate (may stall on non-stationary points, McKinnon 1998)',
    summary:
      'Move a simplex of n + 1 points by reflecting, expanding, contracting or shrinking it.',
    references: [
      'Nelder & Mead (1965), Comput. J. 7(4), 308–313',
      'Lagarias, Reeds, Wright & Wright (1998), SIAM J. Optim. 9(1), 112–147, §2',
      'Gao & Han (2012), Comput. Optim. Appl. 51(1), 259–277, eq. (4.1)',
    ],
  },
  nelderMead,
  DOCS.nelder_mead,
);

registerMethod(
  {
    id: 'powell',
    family: 'unconstrained',
    name: 'Powell (conjugate directions)',
    params: [
      param.float('ftol', 1e-10, {
        min: 1e-15,
        max: 1e-2,
        log: true,
        help:
          'Relative f test: a sweep decreases f by ≤ ftol·(|f_old| + |f_new|)/2 (+10⁻²⁵). ' +
          'Relative, so it depends on the size of f, not only on its variation.',
        label: 'Relative f tolerance',
      }),
      param.float('xtol', 1e-8, {
        min: 1e-14,
        max: 1e-1,
        log: true,
        help:
          'Also required: the sweep moved x by ≤ xtol + 2ε‖x‖∞ (∞-norm), ' +
          'or decreased f only at rounding level (≤ 2ε|f|).',
        label: 'Sweep step tolerance',
      }),
      param.int('max_iter', 200, {
        min: 1,
        max: 10_000,
        help: 'Sweep limit.',
        label: 'Max sweeps',
      }),
    ],
    needs: ['f'],
    order:
      "no quadratic termination with NR's discarding rule (2n–3n sweeps on random " +
      "quadratics); Powell's (1964) basic method: n sweeps",
    summary: 'Minimize along n directions in turn, then swap in the net displacement as a new one.',
    references: [
      'Powell (1964), Comput. J. 7(2), 155–162',
      'Press et al., Numerical Recipes (3rd ed.), §10.7 (Powell) with §10.1 (mnbrak), §10.3 (Brent)',
    ],
  },
  powell,
  DOCS.powell,
);

registerMethod(
  {
    id: 'hooke_jeeves',
    family: 'unconstrained',
    name: 'Hooke–Jeeves pattern search',
    params: STEP_PARAMS,
    needs: ['f'],
    order: 'linear at best; ‖∇f‖ = O(step) at unsuccessful iterations',
    summary: 'Probe each coordinate, then jump along the direction of the last success.',
    references: [
      'Hooke & Jeeves (1961), J. ACM 8(2), 212–229',
      'Kelley (1999), Iterative Methods for Optimization, SIAM, §8.3',
      'Torczon (1997), SIAM J. Optim. 7(1), 1–25 (convergence)',
    ],
  },
  hookeJeeves,
  DOCS.hooke_jeeves,
);

registerMethod(
  {
    id: 'compass_search',
    family: 'unconstrained',
    name: 'Compass search',
    params: STEP_PARAMS,
    needs: ['f'],
    order: 'linear at best; ‖∇f‖ ≤ √n·M·Δ at unsuccessful iterations',
    summary:
      'Try a step north, south, east and west; move on the first success, else halve the step.',
    references: [
      'Kolda, Lewis & Torczon (2003), SIAM Review 45(3), 385–482, Alg. 3.1 and eq. (3.3)',
    ],
  },
  compassSearch,
  DOCS.compass_search,
);
