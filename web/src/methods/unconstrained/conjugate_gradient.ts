/**
 * Nonlinear conjugate gradient (CG) — TS port of `numopt.unconstrained.conjugate_gradient`
 * (src/numopt/unconstrained/conjugate_gradient.py).
 *
 * Every method generates search directions
 *
 *     d₀ = −g₀,        d_k = −g_k + β_k d_{k−1}   (k ≥ 1),        g_k = ∇f(x_k),
 *
 * and steps x_{k+1} = x_k + α_k d_k with a line search (Nocedal & Wright (2006), Alg. 5.4). The
 * methods differ only in β_k (y_{k−1} = g_k − g_{k−1}):
 *
 *   cg_fletcher_reeves   β^FR  = ‖g_k‖² / ‖g_{k−1}‖²                         (N&W eq. 5.41a)
 *   cg_polak_ribiere     β^PR+ = max(g_kᵀy_{k−1} / ‖g_{k−1}‖², 0)           (N&W eq. 5.44, 5.45)
 *   cg_hestenes_stiefel  β^HS  = g_kᵀy_{k−1} / d_{k−1}ᵀy_{k−1}              (N&W eq. 5.46)
 *   cg_dai_yuan          β^DY  = ‖g_k‖² / d_{k−1}ᵀy_{k−1}                    (N&W eq. 5.49)
 *   cg_hager_zhang       β̄^N  = max(β^N, η_k)                               (Hager & Zhang 2005)
 *
 * Restarts (d_k = −g_k, β_k = 0), tested in this order: "periodic" (n iterations since the last
 * steepest-descent direction), "powell" (|g_kᵀg_{k−1}| ≥ 0.1‖g_k‖², N&W eq. 5.52), "breakdown"
 * (β undefined, or the CG direction / its slope overflows), "not_descent" (g_kᵀd_k ≥ 0).
 *
 * Line search: `strong_wolfe` (c₁ = 1e-4, c₂ = `c2`, first trial 1/‖g₀‖ at k = 0, then
 * α_{k−1} g_{k−1}ᵀd_{k−1} / g_kᵀd_k) or `exact_quadratic` (α = −gᵀd / dᵀ∇²f d).
 *
 * Stopping test (converged): ‖g_k‖∞ ≤ `gtol`. Failures (converged = false): non-finite f / ∇f,
 * a failed line search, dᵀ∇²f d ≤ 0 with the exact search, `max_iter`, or ‖g_k‖₂² under- or
 * overflowing. The same evaluation counts as Python (`n_fev` / `n_gev` include the line search,
 * `n_hev` counts the Hessians of the exact search).
 *
 * Step.info keys (snake_case, as in Python): direction, beta, beta_formula, restart,
 * powell_ratio, descent, alpha, trials. `Result.extra = { n_restarts }`.
 *
 * The module also exports the small numerical helpers that the trust-region port shares
 * (`resolveSmooth`, `norm2`, `frexpExp`, `ldexp`, `allFinite`, `ValueError`).
 */
import { registerMethod, param, type MethodDoc } from '../../core/registry';
import type {
  Matrix,
  MethodFn,
  Params,
  Problem,
  Result,
  RunOptions,
  Step,
  Vector,
} from '../../core/types';
import { dot, matvec, norm } from '../../core/linalg';
import { LineSearchZeroDivision, formatG, formatRepr, search } from '../line_search/methods';

// ---------------------------------------------------------------------------------------
// Shared helpers (also used by trust_region.ts)
// ---------------------------------------------------------------------------------------

/** Python `ValueError`: invalid input (parameters, start point, shapes). */
export class ValueError extends Error {
  override name = 'ValueError';
}

/** `format(v, ".3g")` as Python prints it. */
export const g3 = (v: number): string => formatG(v, 3);

/** True when every number (scalar, vector or matrix entries) is finite. */
export function allFinite(
  ...values: (number | readonly number[] | Matrix | null | undefined)[]
): boolean {
  for (const v of values) {
    if (v === null || v === undefined) return false;
    if (typeof v === 'number') {
      if (!Number.isFinite(v)) return false;
      continue;
    }
    for (const e of v as readonly (number | readonly number[])[]) {
      if (typeof e === 'number') {
        if (!Number.isFinite(e)) return false;
      } else if (!allFinite(e)) return false;
    }
  }
  return true;
}

/** `np.max(np.abs(v))` (NaN propagates; 0 for an empty vector). */
export function maxAbs(v: readonly number[]): number {
  let m = 0;
  for (const t of v) m = Math.max(m, Math.abs(t));
  return m;
}

const VIEW = new DataView(new ArrayBuffer(8));

/** The exponent e of `math.frexp(x)`: x = m·2ᵉ with ½ ≤ |m| < 1 (x finite and nonzero). */
export function frexpExp(x: number): number {
  VIEW.setFloat64(0, x);
  const biased = (VIEW.getUint32(0) >>> 20) & 0x7ff;
  if (biased === 0) return frexpExp(x * 2 ** 64) - 64; // subnormal
  return biased - 1022;
}

/** `math.ldexp(x, n)` = x·2ⁿ, rounded once like C's ldexp in the ranges used here. */
export function ldexp(x: number, n: number): number {
  if (n > 1023) return x * 2 ** 1023 * 2 ** (n - 1023);
  if (n < -1022) return x * 2 ** (n + 1022) * 2 ** -1022;
  return x * 2 ** n;
}

/**
 * ‖v‖₂ without underflow or overflow of the squares (Higham (2002), §27.8 / LAPACK dnrm2): v is
 * scaled by 2⁻ᵉ (exact) before squaring, so the result equals `norm(v)` whenever no square under-
 * or overflows, and stays > 0 for v ≠ 0.
 */
export function norm2(v: readonly number[]): number {
  const vinf = v.length ? maxAbs(v) : 0.0;
  if (!(vinf > 0.0 && vinf < Infinity)) return vinf; // 0, inf or nan
  const e = frexpExp(vinf);
  return ldexp(norm(v.map((t) => ldexp(t, -e))), e);
}

/** Python's `max(a, b)`: `a` unless `b > a`. */
export const pyMax = (a: number, b: number): number => (b > a ? b : a);
/** Python's `min(a, b)`: `a` unless `b < a`. */
export const pyMin = (a: number, b: number): number => (b < a ? b : a);

/** ε^{1/3} for float64, the central-difference step factor of `numopt.core.diff`. */
const H_CENTRAL = 6.055454452393343e-6;

/** `numopt.core.diff.gradient`: central differences, 2n evaluations of f. */
function fdGradient(f: (x: Vector) => number, x: Vector): Vector {
  const g = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const xp = x.map((v, j) => v + (j === i ? h : 0.0));
    const xm = x.map((v, j) => v - (j === i ? h : 0.0));
    g[i] = (f(xp) - f(xm)) / (2.0 * h);
  }
  return g;
}

/** `numopt.core.diff.hessian`: symmetrized central differences of ∇f, 2n gradient evaluations. */
function fdHessian(grad: (x: Vector) => Vector, x: Vector): Matrix {
  const n = x.length;
  const H: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    const h = H_CENTRAL * Math.max(1.0, Math.abs(x[i]));
    const gp = grad(x.map((v, j) => v + (j === i ? h : 0.0)));
    const gm = grad(x.map((v, j) => v - (j === i ? h : 0.0)));
    for (let r = 0; r < n; r++) H[r][i] = (gp[r] - gm[r]) / (2.0 * h);
  }
  return H.map((row, r) => row.map((v, c) => 0.5 * (v + H[c][r])));
}

function flatten(v: unknown): number[] {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) return (v as unknown[]).flat(Infinity).map(Number);
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>, Number);
  return [Number(v)];
}

/** f(x) as a number; a size-1 array is accepted, a larger one is invalid input. */
function value(v: unknown): number {
  const arr = flatten(v);
  if (arr.length !== 1)
    throw new ValueError(`f(x) must return a scalar, got an array of shape (${arr.length},)`);
  return arr[0];
}

/** A problem accepted by the n-D smooth methods: a library Problem or a bare f(x). */
export type SmoothProblem = Problem<Vector> | Problem<number> | ((x: Vector) => unknown);

/** A counted callable: `fn(x)` increments `fn.n`. */
export interface Counted<T> {
  (x: Vector): T;
  readonly n: number;
}

function counted<T>(fn: (x: Vector) => T): Counted<T> {
  let n = 0;
  const c = ((x: Vector) => {
    n++;
    return fn(x);
  }) as Counted<T>;
  Object.defineProperty(c, 'n', { get: () => n });
  return c;
}

export interface Resolved {
  x: Vector;
  f: Counted<number>;
  grad: Counted<Vector>;
  hess: Counted<Matrix>;
}

/**
 * `_resolve` of the Python modules: (x0, counted f, counted ∇f, counted ∇²f). A Problem with
 * dim == 1 follows the scalar convention (its callables receive and return numbers); missing
 * derivatives are replaced by central differences of the counted lower-order callable.
 */
export function resolveSmooth(problem: SmoothProblem, x0: unknown): Resolved {
  let prob: Problem<unknown>;
  if (typeof problem === 'function') {
    const dim = x0 !== undefined && x0 !== null ? flatten(x0).length : 2;
    prob = {
      id: 'custom',
      name: 'custom',
      latex: 'f(x)',
      f: problem as (x: unknown) => unknown,
      dim,
      domain: [],
    };
  } else {
    prob = problem as Problem<unknown>;
  }
  const start = x0 ?? prob.x0;
  if (start === undefined || start === null)
    throw new ValueError(`${prob.id}: no starting point given and the problem has no default x0`);
  const x = flatten(start);
  if (prob.dim && x.length !== prob.dim)
    throw new ValueError(`${prob.id}: x0 has ${x.length} entries, expected ${prob.dim}`);
  const scalar = prob.dim === 1;
  const arg = (z: Vector): unknown => (scalar ? z[0] : z.slice());
  const n = x.length;

  const f = counted((z: Vector) => value(prob.f(arg(z))));
  const gFn = prob.grad;
  const grad = gFn
    ? counted((z: Vector) => flatten(gFn(arg(z))))
    : counted((z: Vector) => fdGradient(f, z));
  const hFn = prob.hess;
  const hess = hFn
    ? counted((z: Vector) => {
        const flat = flatten(hFn(arg(z)));
        if (flat.length !== n * n)
          throw new ValueError(
            `cannot reshape array of size ${flat.length} into shape (${n},${n})`,
          );
        return Array.from({ length: n }, (_, i) => flat.slice(i * n, (i + 1) * n));
      })
    : counted((z: Vector) => fdHessian(grad, z));
  return { x, f, grad, hess };
}

// ---------------------------------------------------------------------------------------
// Constants and parameters
// ---------------------------------------------------------------------------------------

/** Armijo constant c₁ (N&W p. 33). */
export const C1 = 1e-4;
/** Default curvature constant c₂ for CG (N&W §5.2, p. 125). */
export const C2_DEFAULT = 0.1;
/** Powell's restart threshold ν (N&W eq. 5.52). */
export const POWELL_NU = 0.1;
/** Hager–Zhang lower-bound constant η (Hager & Zhang 2005, eq. 1.6). */
export const HZ_ETA = 0.01;
/** Smallest positive normal float64. */
export const TINY = 2.2250738585072014e-308;
/** Largest step length the strong Wolfe search may try. */
export const ALPHA_MAX = 1e10;

export const LINE_SEARCHES = ['strong_wolfe', 'exact_quadratic'] as const;
export const RULES = [
  'fletcher_reeves',
  'polak_ribiere',
  'hestenes_stiefel',
  'dai_yuan',
  'hager_zhang',
] as const;
export type BetaRule = (typeof RULES)[number];

const PARAMS = [
  param.float('gtol', 1e-5, {
    min: 1e-14,
    max: 1e-2,
    log: true,
    help:
      'Stop when the gradient ‖∇f(x)‖∞ ≤ gtol. Below ≈ √(2ε|f⋆|λₘₐₓ) the f-based line ' +
      "search cannot certify a decrease and the run stops unconverged (10⁻⁵ = SciPy's default).",
    label: 'Gradient tolerance',
    tex: '\\|\\nabla f\\|_\\infty \\le',
  }),
  param.int('max_iter', 1000, {
    min: 1,
    max: 100_000,
    help: 'Iteration limit.',
    label: 'Max iterations',
  }),
  param.choice('line_search', 'strong_wolfe', [...LINE_SEARCHES], {
    help: 'Strong Wolfe search, or the exact step −gᵀd/dᵀ∇²f d (exact for quadratics).',
    label: 'Line search',
  }),
  param.float('c2', C2_DEFAULT, {
    min: 0.01,
    max: 0.9,
    help: 'Strong Wolfe curvature constant c₂ (0.1 for CG; Fletcher–Reeves needs c₂ < ½).',
    label: 'Curvature constant',
    tex: 'c_2',
  }),
];

// ---------------------------------------------------------------------------------------
// β formulas
// ---------------------------------------------------------------------------------------

/** (β used, raw β formula) for `rule`; [null, null] when the formula is undefined. */
export function betaOf(
  rule: BetaRule,
  g: readonly number[],
  gPrev: readonly number[],
  dPrev: readonly number[],
): [number | null, number | null] {
  const y = g.map((v, i) => v - gPrev[i]);
  let den: number, num: number;
  if (rule === 'fletcher_reeves' || rule === 'polak_ribiere') {
    den = dot(gPrev, gPrev);
    num = rule === 'fletcher_reeves' ? dot(g, g) : dot(g, y);
  } else {
    den = dot(dPrev, y);
    if (rule === 'hestenes_stiefel') num = dot(g, y);
    else if (rule === 'dai_yuan') num = dot(g, g);
    // hager_zhang: β^N = (y − 2d‖y‖²/dᵀy)ᵀg / dᵀy  (Hager & Zhang 2005, eq. 1.4)
    else num = den > 0.0 ? dot(g, y) - (2.0 * dot(y, y) * dot(dPrev, g)) / den : 0.0;
  }
  if (!(den > 0.0 && Number.isFinite(den))) return [null, null];
  const raw = num / den;
  if (!Number.isFinite(raw)) return [null, null];
  if (rule === 'polak_ribiere') return [pyMax(raw, 0.0), raw]; // PR+ (N&W eq. 5.45)
  if (rule === 'hager_zhang') {
    // η_k = −1/(‖d_{k−1}‖ min(η, ‖g_{k−1}‖)) (Hager & Zhang 2005, eq. 1.6).
    const etaK = -1.0 / (norm(dPrev) * pyMin(HZ_ETA, norm(gPrev)));
    return Number.isFinite(etaK) ? [pyMax(raw, etaK), raw] : [raw, raw];
  }
  return [raw, raw];
}

interface CgInfo {
  direction: Vector | null;
  beta: number | null;
  beta_formula: number | null;
  restart: string | null;
  powell_ratio: number | null;
  descent: number | null;
  alpha: number | null;
  trials: [number, number][];
  [key: string]: unknown;
}

const neg = (v: readonly number[]): Vector => v.map((t) => -t);

// ---------------------------------------------------------------------------------------
// The shared nonlinear CG driver (N&W Alg. 5.4 with a choice of β)
// ---------------------------------------------------------------------------------------

export interface CgOptions {
  gtol: number;
  maxIter: number;
  lineSearch: string;
  c2: number;
}

export function nonlinearCg(
  method: string,
  rule: BetaRule,
  problem: SmoothProblem,
  x0: unknown,
  { gtol, maxIter, lineSearch, c2 }: CgOptions,
): Result {
  if (!(LINE_SEARCHES as readonly string[]).includes(lineSearch))
    throw new ValueError(
      `line_search must be one of ('strong_wolfe', 'exact_quadratic'), got '${lineSearch}'`,
    );
  if (!(gtol >= 0.0)) throw new ValueError(`gtol must be ≥ 0, got ${formatRepr(gtol)}`);
  if (!Number.isInteger(maxIter) || maxIter < 1)
    throw new ValueError(`max_iter must be a positive integer, got ${maxIter}`);
  if (!(C1 < c2 && c2 < 1.0))
    throw new ValueError(`c2 must lie in (c1, 1) = (${formatRepr(C1)}, 1), got ${formatRepr(c2)}`);

  const { x: xStart, f, grad, hess } = resolveSmooth(problem, x0);
  let x = xStart;
  const n = x.length;
  let fx = f(x);
  let g: Vector = Number.isFinite(fx) ? grad(x) : new Array<number>(n).fill(NaN);
  const trace: Step[] = [];
  let nRestarts = 0;

  const done = (converged: boolean, message: string, k: number): Result => ({
    method,
    x: x.slice(),
    fun: fx,
    converged,
    message,
    nIter: k,
    nFev: f.n,
    nGev: grad.n,
    nHev: hess.n,
    trace,
    extra: { n_restarts: nRestarts },
  });

  const info = (o: Partial<CgInfo>): CgInfo => ({
    direction: null,
    beta: null,
    beta_formula: null,
    restart: null,
    powell_ratio: null,
    descent: null,
    alpha: null,
    trials: [],
    ...o,
  });

  if (!allFinite(fx, g)) {
    const gn = allFinite(g) ? norm2(g) : null;
    trace.push({ k: 0, x: x.slice(), fun: fx, gradNorm: gn, stepSize: null, info: info({}) });
    return done(false, 'f or ∇f is not finite at x0', 0);
  }

  // State carried between iterations (values at x_{k−1}).
  let gPrev: Vector | null = null;
  let dPrev: Vector | null = null;
  let slopePrev = NaN; // g_{k−1}ᵀd_{k−1}
  let alphaPrev: number | null = null;
  let trials: [number, number][] = [];
  let sinceRestart = 0;
  let k = 0;
  for (;;) {
    const gnorm = norm2(g);
    const ginf = maxAbs(g);
    const gnormSq = gnorm * gnorm;
    const underflow = gnormSq < TINY;
    const overflow = !Number.isFinite(gnormSq);
    if (ginf <= gtol || underflow || overflow) {
      trace.push({
        k,
        x: x.slice(),
        fun: fx,
        gradNorm: gnorm,
        stepSize: alphaPrev,
        info: info({ alpha: alphaPrev, trials }),
      });
      if (ginf <= gtol) return done(true, `gradient ‖∇f‖∞ = ${g3(ginf)} ≤ gtol`, k);
      if (overflow)
        return done(
          false,
          `‖∇f‖₂ = ${g3(gnorm)} is so large that ‖∇f‖₂² overflows: β and the slope ∇fᵀd ` +
            'cannot be formed in float64; rescale f',
          k,
        );
      return done(
        false,
        `‖∇f‖₂ = ${g3(gnorm)} is so small that ‖∇f‖₂² underflows (‖∇f‖∞ = ${g3(ginf)} > ` +
          'gtol): β and the slope ∇fᵀd cannot be formed in float64; rescale f',
        k,
      );
    }

    // ---- search direction d_k ------------------------------------------------------
    let beta: number | null = null;
    let betaFormula: number | null = null;
    let powellRatio: number | null = null;
    let restart: string | null = null;
    let d: Vector;
    if (gPrev === null || dPrev === null) {
      restart = 'initial';
      d = neg(g);
    } else {
      [beta, betaFormula] = betaOf(rule, g, gPrev, dPrev);
      powellRatio = Math.abs(dot(g, gPrev)) / gnormSq; // inf ≥ ν restarts
      d = neg(g);
      if (sinceRestart + 1 >= n) restart = 'periodic';
      else if (powellRatio >= POWELL_NU) restart = 'powell';
      else if (beta === null) restart = 'breakdown';
      else {
        const b = beta,
          dp = dPrev;
        d = g.map((v, i) => -v + b * dp[i]);
        const gd = dot(g, d);
        if (!(allFinite(d) && Number.isFinite(gd))) {
          // β_k d_{k−1} or g_kᵀd_k overflowed: the CG direction is not representable.
          restart = 'breakdown';
          d = neg(g);
        } else if (!(gd < 0.0)) {
          restart = 'not_descent';
          d = neg(g);
        }
      }
      if (restart !== null) {
        beta = 0.0;
        nRestarts++;
      }
    }
    sinceRestart = restart !== null ? 0 : sinceRestart + 1;
    const slope = dot(g, d);
    trace.push({
      k,
      x: x.slice(),
      fun: fx,
      gradNorm: gnorm,
      stepSize: alphaPrev,
      info: info({
        direction: d.slice(),
        beta,
        beta_formula: betaFormula,
        restart,
        powell_ratio: powellRatio,
        descent: slope,
        alpha: alphaPrev,
        trials,
      }),
    });
    if (k === maxIter) return done(false, `reached max_iter=${maxIter}`, k);

    // ---- line search along d_k --------------------------------------------------------
    let ls: ReturnType<typeof search>;
    try {
      if (lineSearch === 'exact_quadratic') {
        const H = hess(x);
        const curvature = dot(d, matvec(H, d));
        if (!Number.isFinite(curvature))
          return done(
            false,
            `dᵀ∇²f d = ${g3(curvature)} is not finite at iteration ${k}: the exact line ` +
              'search cannot form the step −gᵀd/dᵀ∇²f d (rescale f)',
            k,
          );
        if (!(curvature > 0.0))
          return done(
            false,
            `dᵀ∇²f d = ${g3(curvature)} ≤ 0 at iteration ${k}: the exact line search ` +
              'needs positive curvature along d (f is not a convex quadratic)',
            k,
          );
        ls = search('exact_quadratic', f, grad, x, d, { f0: fx, g0: g, hess: H });
      } else {
        let alpha0 = k === 0 ? 1.0 / gnorm : ((alphaPrev as number) * slopePrev) / slope;
        if (!(Number.isFinite(alpha0) && alpha0 > 0.0)) alpha0 = 1.0;
        alpha0 = pyMin(alpha0, ALPHA_MAX);
        ls = search('strong_wolfe', f, grad, x, d, {
          f0: fx,
          g0: g,
          alpha0,
          c1: C1,
          c2,
          alphaMax: ALPHA_MAX,
        });
      }
    } catch (exc) {
      // The contract forbids raising on numerical breakdown (Python catches ArithmeticError).
      if (exc instanceof LineSearchZeroDivision)
        return done(
          false,
          `line search broke down at iteration ${k} (${exc.name}: ${exc.message}); the step ` +
            'lengths are at the limit of float64 (rescale f)',
          k,
        );
      throw exc;
    }
    trials = ls.trials.map(([a, v]) => [a, v]);
    if (!ls.success)
      return done(
        false,
        `line search failed at iteration ${k} (‖∇f‖∞ = ${g3(maxAbs(g))}): ${ls.message}`,
        k,
      );

    const alpha = ls.alpha;
    const xNew = x.map((v, i) => v + alpha * d[i]);
    const fNew = ls.fNew;
    const gNew = ls.gNew !== null ? ls.gNew.slice() : grad(xNew);
    if (!allFinite(fNew, gNew))
      return done(false, `f or ∇f is not finite at the new iterate (iteration ${k})`, k);
    gPrev = g;
    dPrev = d;
    slopePrev = slope;
    alphaPrev = alpha;
    x = xNew;
    fx = fNew;
    g = gNew;
    k++;
  }
}

// ---------------------------------------------------------------------------------------
// Registered methods
// ---------------------------------------------------------------------------------------

const COMMON_REFS = [
  'Nocedal & Wright (2006), Numerical Optimization, Alg. 5.4 and §5.2',
  'Nocedal & Wright (2006), Alg. 3.5–3.6 (strong Wolfe line search)',
];

type CgProblem = SmoothProblem;

function cgMethod(id: string, rule: BetaRule): MethodFn<CgProblem> {
  return (problem: CgProblem, options: RunOptions & Params) =>
    nonlinearCg(id, rule, problem, options.x0, {
      gtol: Number(options.gtol ?? 1e-5),
      maxIter: Number(options.max_iter ?? 1000),
      lineSearch: String(options.line_search ?? 'strong_wolfe'),
      c2: Number(options.c2 ?? C2_DEFAULT),
    });
}

const QUANTITIES: MethodDoc['quantities'] = [
  { tex: '\\beta_k', key: 'info.beta', label: 'β used' },
  { tex: '\\alpha_{k-1}', key: 'stepSize', label: 'accepted step' },
  { tex: '\\nabla f_k^{\\mathsf T} d_k', key: 'info.descent', label: 'slope' },
  {
    tex: '|g_k^{\\mathsf T} g_{k-1}| / \\|g_k\\|^2',
    key: 'info.powell_ratio',
    label: 'Powell ratio',
  },
  { tex: '\\text{restart}', key: 'info.restart' },
];

const RULE_TEX = 'x_{k+1} = x_k + \\alpha_k d_k,\\quad d_k = -\\nabla f_k + \\beta_k d_{k-1}';

registerMethod(
  {
    id: 'cg_fletcher_reeves',
    family: 'unconstrained',
    name: 'Conjugate gradient (Fletcher–Reeves)',
    params: PARAMS,
    needs: ['f', 'grad'],
    order: 'linear (n-step quadratic with restarts)',
    summary: 'Steepest descent plus a multiple of the previous direction, β = ‖g_k‖²/‖g_{k−1}‖².',
    references: ['Fletcher & Reeves (1964), Comput. J. 7, 149–154', ...COMMON_REFS],
  },
  cgMethod('cg_fletcher_reeves', 'fletcher_reeves'),
  {
    rule: `${RULE_TEX},\\quad \\beta_k^{FR} = \\frac{\\|\\nabla f_k\\|^2}{\\|\\nabla f_{k-1}\\|^2}`,
    intuition:
      'Each new direction is the steepest-descent direction bent toward the previous one, so the ' +
      'zig-zag of steepest descent straightens out. On a quadratic with exact steps it ends in n steps.',
    quantities: QUANTITIES,
  },
);

registerMethod(
  {
    id: 'cg_polak_ribiere',
    family: 'unconstrained',
    name: 'Conjugate gradient (Polak–Ribière+)',
    params: PARAMS,
    needs: ['f', 'grad'],
    order: 'linear (n-step quadratic with restarts)',
    summary: 'CG with β = max(g_kᵀ(g_k − g_{k−1})/‖g_{k−1}‖², 0): restarts by itself when stuck.',
    references: [
      'Polak & Ribière (1969), Rev. Française Inform. Rech. Opér. 16, 35–43',
      'Gilbert & Nocedal (1992), SIAM J. Optim. 2(1), 21–42 (PR+)',
      ...COMMON_REFS,
    ],
  },
  cgMethod('cg_polak_ribiere', 'polak_ribiere'),
  {
    rule: `${RULE_TEX},\\quad \\beta_k^{PR+} = \\max\\Big(\\frac{\\nabla f_k^{\\mathsf T}(\\nabla f_k - \\nabla f_{k-1})}{\\|\\nabla f_{k-1}\\|^2},\\,0\\Big)`,
    intuition:
      'When a step makes little progress the gradient barely changes, β drops to zero and the ' +
      'method falls back to steepest descent on its own.',
    quantities: QUANTITIES,
  },
);

registerMethod(
  {
    id: 'cg_hestenes_stiefel',
    family: 'unconstrained',
    name: 'Conjugate gradient (Hestenes–Stiefel)',
    params: PARAMS,
    needs: ['f', 'grad'],
    order: 'linear (n-step quadratic with restarts)',
    summary:
      'CG with β = g_kᵀy/d_{k−1}ᵀy, y = g_k − g_{k−1}: d_k is conjugate to d_{k−1} ' +
      'with respect to the average Hessian.',
    references: [
      'Hestenes & Stiefel (1952), J. Res. Nat. Bur. Standards 49(6), 409–436',
      ...COMMON_REFS,
    ],
  },
  cgMethod('cg_hestenes_stiefel', 'hestenes_stiefel'),
  {
    rule: `${RULE_TEX},\\quad \\beta_k^{HS} = \\frac{\\nabla f_k^{\\mathsf T} y_{k-1}}{d_{k-1}^{\\mathsf T} y_{k-1}}`,
    intuition:
      'β is chosen so that the new direction is conjugate to the last one with respect to the ' +
      'average Hessian along the step; a negative β can break descent and forces a restart.',
    quantities: QUANTITIES,
  },
);

registerMethod(
  {
    id: 'cg_dai_yuan',
    family: 'unconstrained',
    name: 'Conjugate gradient (Dai–Yuan)',
    params: PARAMS,
    needs: ['f', 'grad'],
    order: 'linear (n-step quadratic with restarts)',
    summary: 'CG with β = ‖g_k‖²/d_{k−1}ᵀy: a descent direction under any Wolfe line search.',
    references: ['Dai & Yuan (1999), SIAM J. Optim. 10(1), 177–182', ...COMMON_REFS],
  },
  cgMethod('cg_dai_yuan', 'dai_yuan'),
  {
    rule: `${RULE_TEX},\\quad \\beta_k^{DY} = \\frac{\\|\\nabla f_k\\|^2}{d_{k-1}^{\\mathsf T} y_{k-1}}`,
    intuition:
      'The Fletcher–Reeves numerator over the Hestenes–Stiefel denominator: every direction ' +
      'goes downhill as long as the line search meets the Wolfe conditions.',
    quantities: QUANTITIES,
  },
);

registerMethod(
  {
    id: 'cg_hager_zhang',
    family: 'unconstrained',
    name: 'Conjugate gradient (Hager–Zhang)',
    params: PARAMS,
    needs: ['f', 'grad'],
    order: 'linear (n-step quadratic with restarts)',
    summary:
      'The CG_DESCENT direction: Hestenes–Stiefel plus a correction that guarantees ' +
      'gᵀd ≤ −⅞‖g‖² for any line search.',
    references: [
      'Hager & Zhang (2005), SIAM J. Optim. 16(1), 170–192, eq. 1.4–1.6',
      'Hager & Zhang (2006), Pacific J. Optim. 2(1), 35–58 (survey)',
      ...COMMON_REFS,
    ],
  },
  cgMethod('cg_hager_zhang', 'hager_zhang'),
  {
    rule: `${RULE_TEX},\\quad \\beta_k = \\max\\Big(\\frac{(y - 2d\\frac{\\|y\\|^2}{d^{\\mathsf T}y})^{\\mathsf T}\\nabla f_k}{d^{\\mathsf T} y},\\ \\eta_k\\Big)`,
    intuition:
      'Hestenes–Stiefel with a correction term that keeps every direction a sufficient descent ' +
      'direction, and a lower bound η_k that stops β from going too negative.',
    quantities: QUANTITIES,
  },
);
